"""Tests for aone plan_test."""

import unittest

from rtp_llm.test.perf_test.case_planning import plan_cases
from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.config_schema import ConfigError
from rtp_llm.test.perf_test.errors import PlanningError, ResourceError
from rtp_llm.test.perf_test.plan import ExecutionPlan, build_execution_plan, case_id
from rtp_llm.test.perf_test.tests.support.plan import plan_cases, plan_config


class AOnePlanningTest(unittest.TestCase):
    def test_cross_host_dp_layout_is_validated_before_submission(self):
        from rtp_llm.test.perf_test.aone.config_schema import parse_execution
        from rtp_llm.test.perf_test.aone.plan import plan_resources
        from rtp_llm.test.perf_test.capabilities import validate_aone_capability

        for topology, message in (
            (dict(tp=2, dp=2, pp=1, ep=4), "whole local TP"),
            (dict(tp=1, dp=2, pp=2, ep=2), "PP=1"),
        ):
            with self.subTest(topology=topology):
                source = plan_config("aone")
                source["model"]["topology"] = topology
                source["execution"].pop("shard_count")
                source["execution"].update(gpus_per_machine=1, max_parallel=4)
                error_type = ConfigError if topology["pp"] != 1 else PlanningError
                with self.assertRaisesRegex(error_type, message):
                    build_execution_plan(
                        mode="decode",
                        action="run",
                        resolved=resolve_config(
                            source, execution_parser=parse_execution
                        ),
                        cases=plan_cases(),
                        result_directory="unused",
                        resource_planner=plan_resources,
                        capability_validator=validate_aone_capability,
                    )

    def _plan(
        self, *, mode="prefill", execution=None, world_size=2, gang=False, cases=None
    ):
        from rtp_llm.test.perf_test.aone.config_schema import parse_execution
        from rtp_llm.test.perf_test.aone.plan import plan_resources
        from rtp_llm.test.perf_test.capabilities import validate_aone_capability

        source = plan_config("aone")
        source["model"]["topology"].update(tp=world_size, ep=1)
        if gang:
            source["execution"].pop("shard_count")
            source["execution"]["gpus_per_machine"] = 8
        if execution:
            source["execution"].update(execution)
        return build_execution_plan(
            mode=mode,
            action="run",
            resolved=resolve_config(source, execution_parser=parse_execution),
            cases=plan_cases() if cases is None else cases,
            result_directory="runs/aone",
            resource_planner=plan_resources,
            capability_validator=validate_aone_capability,
        )

    def test_single_prefill_and_both_gang_modes_round_trip(self):
        for mode, gang in (("prefill", False), ("prefill", True), ("decode", True)):
            with self.subTest(mode=mode, gang=gang):
                plan = self._plan(mode=mode, world_size=16 if gang else 2, gang=gang)
                self.assertEqual(2 if gang else 1, plan.resource_plan.machine_count)
                self.assertEqual(8 if gang else 2, plan.resource_plan.local_world_size)
                self.assertEqual(plan, ExecutionPlan.from_dict(plan.to_dict()))

    def test_invalid_resources_and_case_gang_combinations_rejected(self):
        invalid = [
            ({"resource": "unknown"}, 2, "resource"),
            ({"gpu_type": "unknown"}, 2, "GPU"),
            ({"shard_count": 3}, 2, "case count"),
            ({"max_parallel": 3}, 2, "max_parallel"),
            ({"gpus_per_machine": 8}, 16, "shard_count"),
            ({"gpus_per_machine": 3}, 8, "divide"),
            ({"gpus_per_machine": 16}, 16, "GPU|capacity|host"),
        ]
        for execution, world_size, message in invalid:
            with self.subTest(execution=execution), self.assertRaisesRegex(
                (PlanningError, ResourceError), message
            ):
                self._plan(execution=execution, world_size=world_size)

    def test_gang_requires_enough_parallel_machines(self):
        with self.assertRaisesRegex(ResourceError, "max_parallel"):
            self._plan(world_size=16, gang=True, execution={"max_parallel": 1})

    def test_single_machine_requires_case_shards(self):
        with self.assertRaisesRegex(PlanningError, "shard_count"):
            self._plan(world_size=8, gang=True)

    def test_oversized_machine_setting_normalizes_and_round_trips(self):
        plan = self._plan(execution={"gpus_per_machine": 8})
        self.assertEqual(2, plan.resource_plan.gpus_per_machine)
        self.assertEqual(plan, ExecutionPlan.from_dict(plan.to_dict()))

    def test_persisted_machine_setting_rejects_non_positive_and_non_integers(self):
        for value in (0, -1, True, "2", 1.5):
            with self.subTest(value=value):
                payload = self._plan().to_dict()
                payload["resolved_config"]["execution"]["gpus_per_machine"] = value
                payload["resource_plan"].update(
                    gpus_per_machine=1, machine_count=2, local_world_size=1
                )
                with self.assertRaises(ResourceError):
                    ExecutionPlan.from_dict(payload)

    def test_case_shards_balance_cost_and_assign_every_case_once(self):
        from rtp_llm.test.perf_test.aone.sharding import plan_shards

        shard_plan = plan_shards(self._plan())
        self.assertEqual(
            (524288, 262144),
            tuple(shard.estimated_cost for shard in shard_plan.shards),
        )
        self.assertEqual(786432, shard_plan.total_estimated_cost)
        self.assertEqual(
            (1, 1), tuple(len(shard.case_ids) for shard in shard_plan.shards)
        )
        self.assertEqual(
            {case_id("prefill", case) for case in plan_cases()},
            {
                identifier
                for shard in shard_plan.shards
                for identifier in shard.case_ids
            },
        )

    def test_cost_shards_place_longest_cases_on_the_lightest_shard(self):
        from rtp_llm.test.perf_test.aone.sharding import plan_shards

        cases = [
            {"batch_size": 1, "input_len": length, "cache_len": 0}
            for length in (10, 9, 8, 1)
        ]
        forward = plan_shards(self._plan(cases=cases))
        reverse = plan_shards(self._plan(cases=list(reversed(cases))))
        self.assertEqual(forward, reverse)
        self.assertEqual((101, 145), tuple(s.estimated_cost for s in forward.shards))
        self.assertEqual(
            {case_id("prefill", cases[0]), case_id("prefill", cases[3])},
            set(forward.shards[0].case_ids),
        )

    def test_grouped_case_cost_counts_each_request(self):
        from rtp_llm.test.perf_test.aone.sharding import case_weight

        self.assertEqual(
            105,
            case_weight(
                {
                    "request_groups": [
                        {"count": 2, "input_len": 10, "cache_len": 6},
                        {"count": 1, "input_len": 5, "cache_len": 0},
                    ]
                }
            ),
        )
        with self.assertRaises(PlanningError):
            case_weight(
                {
                    "request_groups": [
                        {"count": 0, "input_len": 10, "cache_len": 6},
                    ]
                }
            )

    def test_gang_has_no_case_ownership_shards(self):
        from rtp_llm.test.perf_test.aone.sharding import plan_shards

        with self.assertRaisesRegex(PlanningError, "gang"):
            plan_shards(self._plan(world_size=16, gang=True))

    def test_single_machine_decode_remains_unsupported(self):
        with self.assertRaisesRegex(PlanningError, "decode.*gang"):
            self._plan(mode="decode")

    def test_schema_rejects_invalid_numbers_and_unknown_fields(self):
        from rtp_llm.test.perf_test.aone.config_schema import parse_execution
        from rtp_llm.test.perf_test.config_schema import ConfigError

        for changes in (
            {"max_parallel": True},
            {"gpus_per_machine": 0},
            {"shard_count": None},
            {"devices": [0]},
        ):
            with self.subTest(changes=changes), self.assertRaises(ConfigError):
                parse_execution({**plan_config("aone")["execution"], **changes})


if __name__ == "__main__":
    unittest.main()

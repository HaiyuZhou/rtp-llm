import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.config_schema import parse_config
from rtp_llm.test.perf_test.decode.distribution import parse_distribution_inputs
from rtp_llm.test.perf_test.decode.model import (
    DecodeCaseSpec,
    DecodeProfilingConfig,
    ExactBatchPolicy,
    FixedBatchConfig,
    TargetTPOTConfig,
    TargetTPOTPolicy,
)
from rtp_llm.test.perf_test.decode.planner import plan_decode_cases
from rtp_llm.test.perf_test.entry import _planned_cases
from rtp_llm.test.perf_test.errors import PlanningError
from rtp_llm.test.perf_test.plan import case_id


def _payload(workload):
    return {
        "schema_version": 1,
        "model": {"label": "test-model", "type": "registered", "checkpoint": "/m"},
        "engine": {
            "max_seq_len": 4096,
            "max_batch_tokens": 8192,
            "concurrency_limit": 8,
        },
        "workload": workload,
        "execution": {"backend": "local"},
        "output": {},
        "regression": {},
        "publish": {},
    }


def _resolved_grid(*, inputs, outputs, sizes):
    payload = _payload(
        {
            "kind": "grid",
            "input_lengths": inputs,
            "output_lengths": outputs,
            "batch": {"strategy": "fixed", "sizes": sizes},
        }
    )
    return parse_config(payload), resolve_config(payload)


def _resolved_tpot(*, inputs, outputs, max_size):
    payload = _payload(
        {
            "kind": "grid",
            "input_lengths": inputs,
            "output_lengths": outputs,
            "batch": {
                "strategy": "target_tpot",
                "max_size": max_size,
                "candidate_step": 2,
                "target_tpot_ms": 10.0,
            },
        }
    )
    return parse_config(payload), resolve_config(payload)


def _resolved_distribution(*, distribution, max_size=None, sizes=None):
    batch = (
        {"strategy": "fixed", "sizes": sizes}
        if sizes is not None
        else {
            "strategy": "target_tpot",
            "max_size": max_size,
            "candidate_step": 2,
            "target_tpot_ms": 10.0,
        }
    )
    payload = _payload(
        {
            "kind": "distribution",
            "distribution_file": "requests.json",
            "output_length": 16,
            "batch": batch,
        }
    )
    with tempfile.TemporaryDirectory() as directory:
        Path(directory, "requests.json").write_text(json.dumps(distribution))
        return parse_config(payload), resolve_config(payload, source_dir=directory)


class DecodePlannerTest(unittest.TestCase):
    def test_exact_grid_is_input_by_output_by_batch(self):
        config, resolved = _resolved_grid(
            inputs=[128, 512], outputs=[16, 64], sizes=[1, 4]
        )
        cases = plan_decode_cases(
            config.workload,
            config.engine,
            dp_size=2,
            resolved_workload=resolved.resolved_config["workload"],
        )
        specs = [DecodeCaseSpec.from_identity(case) for case in cases]
        self.assertEqual(8, len(specs))
        self.assertEqual(
            [
                (128, 16, 1),
                (128, 16, 4),
                (128, 64, 1),
                (128, 64, 4),
                (512, 16, 1),
                (512, 16, 4),
                (512, 64, 1),
                (512, 64, 4),
            ],
            [
                (item.input_lengths[0], item.output_length, item.batch_policy.size)
                for item in specs
            ],
        )
        self.assertEqual((128, 128, 128, 128), specs[1].input_lengths)

    def test_tpot_grid_has_one_case_per_input_output_pair(self):
        config, resolved = _resolved_tpot(
            inputs=[128, 512], outputs=[16, 64], max_size=8
        )
        cases = plan_decode_cases(
            config.workload,
            config.engine,
            dp_size=1,
            resolved_workload=resolved.resolved_config["workload"],
        )
        specs = [DecodeCaseSpec.from_identity(case) for case in cases]
        self.assertEqual(4, len(specs))
        self.assertTrue(all(len(item.input_lengths) == 8 for item in specs))
        self.assertEqual(
            TargetTPOTPolicy("target_tpot", 8, 2, 10.0), specs[0].batch_policy
        )

    def test_case_ids_are_stable_and_distinguish_output_and_policy(self):
        fixed, fixed_resolved = _resolved_grid(
            inputs=[128], outputs=[16, 64], sizes=[4]
        )
        tpot, tpot_resolved = _resolved_tpot(inputs=[128], outputs=[16], max_size=4)
        fixed_identities = plan_decode_cases(
            fixed.workload,
            fixed.engine,
            dp_size=2,
            resolved_workload=fixed_resolved.resolved_config["workload"],
        )
        repeated = plan_decode_cases(
            fixed.workload,
            fixed.engine,
            dp_size=2,
            resolved_workload=fixed_resolved.resolved_config["workload"],
        )
        tpot_identity = plan_decode_cases(
            tpot.workload,
            tpot.engine,
            dp_size=2,
            resolved_workload=tpot_resolved.resolved_config["workload"],
        )[0]
        self.assertEqual(fixed_identities, repeated)
        ids = [case_id("decode", item) for item in fixed_identities]
        self.assertEqual(2, len(set(ids)))
        self.assertNotEqual(ids[0], case_id("decode", tpot_identity))

    def test_known_sequence_capacity_fails_during_planning(self):
        config, _ = _resolved_grid(inputs=[4090], outputs=[16], sizes=[1])
        with self.assertRaisesRegex(PlanningError, "max_seq_len"):
            plan_decode_cases(config.workload, config.engine, dp_size=2)

    def test_capacity_is_checked_per_dp_worker(self):
        config, _ = _resolved_grid(inputs=[128], outputs=[16], sizes=[3])
        engine = replace(config.engine, concurrency_limit=2, max_batch_tokens=256)
        self.assertEqual(1, len(plan_decode_cases(config.workload, engine, dp_size=2)))
        for overrides, field in (
            ({"concurrency_limit": 1}, "concurrency_limit"),
            ({"max_batch_tokens": 255}, "max_batch_tokens"),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(PlanningError, field):
                    plan_decode_cases(
                        config.workload, replace(engine, **overrides), dp_size=2
                    )

    def test_distribution_tpot_uses_sequence_prefixes_and_rejects_batch_map(self):
        config, resolved = _resolved_distribution(
            distribution={"sequence_lengths": [64, 96, 128, 160]}, max_size=4
        )
        case = plan_decode_cases(
            config.workload,
            config.engine,
            dp_size=2,
            resolved_workload=resolved.resolved_config["workload"],
        )[0]
        self.assertEqual([64, 96, 128, 160], case["input_lengths"])
        self.assertEqual({"source_kind": "sequence_lengths"}, case["distribution"])
        mapped_config, mapped_resolved = _resolved_distribution(
            distribution={"batch_seq_len_map": {"4": [64, 96, 128, 160]}}, max_size=4
        )
        with self.assertRaisesRegex(PlanningError, "batch_seq_len_map.*target_tpot"):
            plan_decode_cases(
                mapped_config.workload,
                mapped_config.engine,
                dp_size=2,
                resolved_workload=mapped_resolved.resolved_config["workload"],
            )

    def test_distribution_fixed_prefixes_and_maps_follow_batch_order(self):
        for distribution, source_kind in (
            ([64, 96, 128], "sequence_lengths"),
            ({"sequence_lengths": [64, 96, 128]}, "sequence_lengths"),
            (
                {"batch_seq_len_map": {"1": [64], "3": [64, 96, 128]}},
                "batch_seq_len_map",
            ),
        ):
            with self.subTest(source_kind=source_kind):
                config, resolved = _resolved_distribution(
                    distribution=distribution, sizes=[3, 1]
                )
                cases = plan_decode_cases(
                    config.workload,
                    config.engine,
                    dp_size=2,
                    resolved_workload=resolved.resolved_config["workload"],
                )
                self.assertEqual(
                    [[64, 96, 128], [64]], [case["input_lengths"] for case in cases]
                )
                self.assertEqual(
                    {"strategy": "fixed", "size": 3}, cases[0]["batch_policy"]
                )
                self.assertEqual({"source_kind": source_kind}, cases[0]["distribution"])
                self.assertTrue(all(case["output_length"] == 16 for case in cases))

    def test_invalid_distribution_is_rejected(self):
        batch = FixedBatchConfig("fixed", (2,))
        for distribution in (
            [],
            [64],
            [64, True],
            {"sequence_lengths": []},
            {"sequence_lengths": [64, 96], "extra": 1},
            {"batch_seq_len_map": {"1": [64]}},
            {"batch_seq_len_map": {"02": [64, 96]}},
            {"batch_seq_len_map": {"2": [64]}},
            {"batch_seq_len_map": []},
        ):
            with self.subTest(distribution=distribution):
                with self.assertRaises(PlanningError):
                    parse_distribution_inputs(distribution, batch)
        with self.assertRaisesRegex(PlanningError, "enough sequence lengths"):
            parse_distribution_inputs([64], TargetTPOTConfig("target_tpot", 2, 1, 10.0))

    def test_distribution_capacity_uses_actual_contiguous_worker_lengths(self):
        config, resolved = _resolved_distribution(
            distribution=[128, 128, 64], sizes=[3]
        )
        with self.assertRaisesRegex(PlanningError, "max_batch_tokens"):
            plan_decode_cases(
                config.workload,
                replace(config.engine, max_batch_tokens=200),
                dp_size=2,
                resolved_workload=resolved.resolved_config["workload"],
            )

    def test_distribution_requires_resolved_content(self):
        config, _ = _resolved_distribution(distribution=[64], sizes=[1])
        with self.assertRaisesRegex(PlanningError, "resolved distribution content"):
            plan_decode_cases(config.workload, config.engine, dp_size=1)

    def test_invalid_dp_or_empty_grid_is_rejected(self):
        config, _ = _resolved_grid(inputs=[128], outputs=[16], sizes=[1])
        for dp_size in (0, True):
            with self.subTest(dp_size=dp_size):
                with self.assertRaisesRegex(PlanningError, "dp_size"):
                    plan_decode_cases(config.workload, config.engine, dp_size=dp_size)
        with self.assertRaisesRegex(PlanningError, "no cases"):
            plan_decode_cases(
                replace(config.workload, input_lengths=()), config.engine, dp_size=1
            )
        with self.assertRaisesRegex(PlanningError, "grid or distribution"):
            plan_decode_cases(object(), config.engine, dp_size=1)

    def test_identity_round_trip_and_nested_data_are_independent(self):
        spec = DecodeCaseSpec(
            workload_kind="distribution",
            input_lengths=(64, 96),
            output_length=16,
            batch_policy=ExactBatchPolicy("fixed", 2),
            measure_runs=2,
            warmup_runs=0,
            generate_config={"stop_words_list": [[1, 2]]},
            profiling=DecodeProfilingConfig(enabled=True, runs=2),
            distribution={"source_kind": "sequence_lengths", "nested": [1]},
        )
        identity = spec.to_identity()
        restored = DecodeCaseSpec.from_identity(json.loads(json.dumps(identity)))
        self.assertEqual(spec, restored)
        identity["generate_config"]["stop_words_list"][0].append(3)
        identity["distribution"]["nested"].append(2)
        self.assertEqual([[1, 2]], spec.generate_config["stop_words_list"])
        self.assertEqual([1], restored.distribution["nested"])
        original = copy.deepcopy(identity)
        restored = DecodeCaseSpec.from_identity(identity)
        restored.generate_config["stop_words_list"][0].append(4)
        self.assertEqual(original, identity)
        identity["batch_policy"]["strategy"] = "unknown"
        with self.assertRaisesRegex(ValueError, "unknown decode batch policy"):
            DecodeCaseSpec.from_identity(identity)

    def test_entry_dispatch_uses_unified_grid_and_distribution_identities(self):
        for workload in (
            {
                "kind": "grid",
                "input_lengths": [128],
                "output_lengths": [16, 64],
                "batch": {"strategy": "fixed", "sizes": [1]},
            },
            {
                "kind": "distribution",
                "distribution_file": "requests.json",
                "output_length": 16,
                "batch": {"strategy": "fixed", "sizes": [1]},
            },
        ):
            with self.subTest(kind=workload["kind"]):
                payload = _payload(workload)
                with tempfile.TemporaryDirectory() as directory:
                    Path(directory, "requests.json").write_text("[64]")
                    resolved = resolve_config(payload, source_dir=directory)
                cases = _planned_cases("decode", payload, resolved)
                self.assertEqual(
                    ["fixed"] * len(cases),
                    [case["batch_policy"]["strategy"] for case in cases],
                )
                self.assertEqual(
                    [16, 64] if workload["kind"] == "grid" else [16],
                    [case["output_length"] for case in cases],
                )


if __name__ == "__main__":
    unittest.main()

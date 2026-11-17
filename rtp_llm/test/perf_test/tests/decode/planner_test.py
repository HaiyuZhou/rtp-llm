"""Tests for decode planner_test."""

import unittest
from dataclasses import replace

from rtp_llm.test.perf_test.config_schema import parse_config
from rtp_llm.test.perf_test.decode.model import DecodeCaseSpec
from rtp_llm.test.perf_test.decode.planner import plan_decode_cases
from rtp_llm.test.perf_test.errors import PlanningError
from rtp_llm.test.perf_test.tests.support.plan import (
    decode_planner_payload,
    decode_planner_resolved_distribution,
    decode_planner_resolved_grid,
)


class DecodePlannerTest(unittest.TestCase):

    def test_fixed_batches_reject_uneven_active_dp_workers(self):
        for size in (1, 3):
            grid, _ = decode_planner_resolved_grid(
                inputs=[128], outputs=[16], sizes=[size]
            )
            distribution, resolved = decode_planner_resolved_distribution(
                distribution=[64, 96, 128], sizes=[size]
            )
            for config, workload in (
                (grid, None),
                (distribution, resolved.resolved_config["workload"]),
            ):
                with self.subTest(kind=config.workload.kind, size=size):
                    with self.assertRaisesRegex(PlanningError, "uniform"):
                        plan_decode_cases(
                            config.workload,
                            config.engine,
                            dp_size=2,
                            resolved_workload=workload,
                        )

    def test_tpot_allows_uneven_maximum_for_filtering_at_search_time(self):
        config, resolved = decode_planner_resolved_distribution(
            distribution=[64, 96, 128], max_size=3
        )
        cases = plan_decode_cases(
            config.workload,
            config.engine,
            dp_size=2,
            resolved_workload=resolved.resolved_config["workload"],
        )
        self.assertEqual(3, cases[0]["batch_policy"]["max_size"])

    def test_tpot_capacity_uses_largest_supported_candidate(self):
        grid = parse_config(
            decode_planner_payload(
                {
                    "kind": "grid",
                    "input_lengths": [128],
                    "output_lengths": [16],
                    "batch": {
                        "strategy": "target_tpot",
                        "max_size": 5,
                        "candidate_step": 2,
                        "target_tpot_ms": 10.0,
                    },
                }
            )
        )
        distribution, resolved = decode_planner_resolved_distribution(
            distribution=[128] * 5, max_size=5
        )
        for config, workload in (
            (grid, None),
            (distribution, resolved.resolved_config["workload"]),
        ):
            with self.subTest(kind=config.workload.kind):
                cases = plan_decode_cases(
                    config.workload,
                    replace(config.engine, concurrency_limit=2, max_batch_tokens=256),
                    dp_size=2,
                    resolved_workload=workload,
                )
                self.assertEqual(5, len(cases[0]["input_lengths"]))

    def test_rank_batch_grid_is_canonical_global_batch(self):
        payload = decode_planner_payload(
            {
                "kind": "grid",
                "input_lengths": [128],
                "output_lengths": [16],
                "batch": {"strategy": "fixed", "sizes": [2, 4], "unit": "per_dp_rank"},
            }
        )
        config = parse_config(payload)
        cases = plan_decode_cases(config.workload, config.engine, dp_size=4)
        self.assertEqual([8, 16], [case["batch_policy"]["size"] for case in cases])
        self.assertEqual([8, 16], [len(case["input_lengths"]) for case in cases])
        self.assertEqual(["per_dp_rank"] * 2, [case["batch_unit"] for case in cases])

    def test_exact_grid_is_input_by_output_by_batch(self):
        (config, resolved) = decode_planner_resolved_grid(
            inputs=[128, 512], outputs=[16, 64], sizes=[2, 4]
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
                (128, 16, 2),
                (128, 16, 4),
                (128, 64, 2),
                (128, 64, 4),
                (512, 16, 2),
                (512, 16, 4),
                (512, 64, 2),
                (512, 64, 4),
            ],
            [
                (item.input_lengths[0], item.output_length, item.batch_policy.size)
                for item in specs
            ],
        )
        self.assertEqual((128, 128, 128, 128), specs[1].input_lengths)

    def test_capacity_is_checked_per_dp_worker(self):
        (config, _) = decode_planner_resolved_grid(
            inputs=[128], outputs=[16], sizes=[4]
        )
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
                (config, resolved) = decode_planner_resolved_distribution(
                    distribution=distribution, sizes=[3, 1]
                )
                cases = plan_decode_cases(
                    config.workload,
                    config.engine,
                    dp_size=1,
                    resolved_workload=resolved.resolved_config["workload"],
                )
                self.assertEqual(
                    [[64, 96, 128], [64]], [case["input_lengths"] for case in cases]
                )
                self.assertEqual(
                    {"strategy": "fixed", "size": 3}, cases[0]["batch_policy"]
                )
                self.assertEqual({"source_kind": source_kind}, cases[0]["distribution"])
                self.assertTrue(all((case["output_length"] == 16 for case in cases)))


if __name__ == "__main__":
    unittest.main()

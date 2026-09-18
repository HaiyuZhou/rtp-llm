import argparse
import unittest

from rtp_llm.test.perf_test.batch_decode_test import _configure_cache_batch_limits
from rtp_llm.test.perf_test.cache_grid.runner.generate_random_batch_grid import (
    generate_grid,
    parse_args,
)
from rtp_llm.test.perf_test.cache_grid.runner.workspace_budget import (
    FIXED_POLICY,
    WORKSPACE_TOKENS,
    fixed_workspace_grid,
    grid_token_budget,
    input_limit,
    validate_fixed_workspace,
)
from rtp_llm.test.perf_test.dataset import extract_arg


class WorkspaceBudgetTest(unittest.TestCase):
    def case(self, batch=32, length=32512):
        return {
            "case_id": 1,
            "batch_size": batch,
            "input_len": length,
            "cache_len": 4096,
        }

    def test_rectangle_not_just_token_sum(self):
        case = {
            "batch_size": 8,
            "request_groups": [
                {"count": 1, "input_len": 200000, "cache_len": 0},
                {"count": 7, "input_len": 8192, "cache_len": 0},
            ],
        }
        with self.assertRaisesRegex(ValueError, "rectangle"):
            validate_fixed_workspace([case])

    def test_cp_padding_and_output_reserve(self):
        self.assertEqual(input_limit(32), 32767)
        with self.assertRaisesRegex(ValueError, "rectangle"):
            validate_fixed_workspace([self.case(length=32768)])
        stats = validate_fixed_workspace([self.case()])
        self.assertEqual(stats[1]["workspace_tokens"], 1040896)

    def test_seed_counts_for_both_prefix_policies(self):
        case = self.case(batch=4, length=8192)
        self.assertEqual(validate_fixed_workspace([case])[-1]["requests"], 4)
        case["prefix_policy"] = "shared_by_group"
        self.assertEqual(validate_fixed_workspace([case])[-1]["requests"], 1)

    def test_invalid_seed_and_probe_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "seed tail"):
            validate_fixed_workspace([self.case(length=4097)])
        with self.assertRaisesRegex(ValueError, "probe"):
            validate_fixed_workspace([self.case()], token_budget=4096)
        with self.assertRaisesRegex(ValueError, "probe"):
            validate_fixed_workspace([self.case()], commit_tail=WORKSPACE_TOKENS)

    def test_generated_mixed_batch_grid_respects_capacity(self):
        args = parse_args(
            [
                "--output",
                "/tmp/not-written.json",
                "--batch-sizes",
                "2",
                "4",
                "8",
                "16",
                "32",
                "--num-cases",
                "200",
            ]
        )
        grid = generate_grid(args)
        self.assertTrue(fixed_workspace_grid(grid))
        self.assertEqual(grid["generator"]["workspace_policy"], FIXED_POLICY)
        self.assertTrue(
            all(
                s["workspace_tokens"] <= WORKSPACE_TOKENS
                for s in validate_fixed_workspace(grid["cases"])
            )
        )
        self.assertEqual(grid, generate_grid(args))

    def test_backend_does_not_expand_context_capacity(self):
        args = argparse.Namespace(
            cache_fixed_workspace=True,
            dp_size=1,
            concurrency_limit=1,
            max_seq_len=8192,
            cache_shared_seed=False,
            cache_commit_tail_tokens=4096,
            expected_cache_block_size=4096,
            decode_test_length=1,
        )
        remaining = [
            "--tp_size=8",
            "--cp_rotate_method=ALL_GATHER",
            "--prefill_cp_kv_cache_sharded=1",
            "--max_context_batch_size=32",
        ]
        _configure_cache_batch_limits(args, remaining, [self.case()])
        self.assertEqual(args.max_seq_len, WORKSPACE_TOKENS)
        self.assertEqual(extract_arg(remaining, "max_context_batch_size"), "1")
        self.assertIsNone(extract_arg(remaining, "max_generate_batch_size"))
        self.assertEqual(args.concurrency_limit, 32)
        with self.assertRaisesRegex(ValueError, "rectangle"):
            _configure_cache_batch_limits(args, remaining, [self.case(length=32768)])

    def test_policy_validation(self):
        self.assertFalse(fixed_workspace_grid({}))
        with self.assertRaisesRegex(ValueError, "unsupported"):
            fixed_workspace_grid({"generator": {"workspace_policy": "unknown"}})

    def test_grid_budget_cannot_exceed_workspace(self):
        self.assertEqual(grid_token_budget({}), WORKSPACE_TOKENS)
        for budget in (0, -1, WORKSPACE_TOKENS + 1):
            with self.assertRaisesRegex(ValueError, "budget"):
                grid_token_budget(
                    {"generator": {"parameters": {"max_batch_tokens": budget}}}
                )


if __name__ == "__main__":
    unittest.main()

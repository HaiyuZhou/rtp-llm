import unittest

from rtp_llm.test.perf_test.aone.sharding import case_weight, plan_shards
from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.plan import build_execution_plan


def _config():
    return {
        "schema_version": 1,
        "model": {
            "label": "test",
            "type": "registered",
            "checkpoint": "/models/test",
        },
        "engine": {
            "max_seq_len": 4096,
            "max_batch_tokens": 8192,
            "concurrency_limit": 4,
        },
        "workload": {
            "kind": "cache_grid",
            "max_input_len": 2048,
            "cache_alignment": 64,
            "commit_tail_tokens": 64,
            "compute_steps": [256],
            "batch_sizes": [1],
        },
        "execution": {
            "backend": "aone",
            "resource": "daily",
            "gpu_type": "H20",
            "max_parallel": 2,
            "shard_count": 2,
        },
        "output": {},
        "regression": {},
        "publish": {},
    }


def _plan():
    cases = [
        {"input_len": 10, "cache_len": 0, "compute_len": 10, "batch_size": 1},
        {"input_len": 10, "cache_len": 1, "compute_len": 9, "batch_size": 1},
        {"input_len": 10, "cache_len": 9, "compute_len": 1, "batch_size": 1},
        {"input_len": 5, "cache_len": 1, "compute_len": 4, "batch_size": 1},
    ]
    return build_execution_plan(
        mode="prefill",
        action="run",
        resolved=resolve_config(_config()),
        cases=cases,
        result_directory="runs/aone",
    )


class AOneShardingTest(unittest.TestCase):
    def test_cost_formula_matches_legacy_attention_work(self):
        cases = [
            {"input_len": 10, "cache_len": 0, "batch_size": 1},
            {"input_len": 10, "cache_len": 5, "batch_size": 1},
            {"input_len": 8, "cache_len": 4, "batch_size": 1},
        ]

        self.assertEqual([100, 50, 32], [case_weight(case) for case in cases])

    def test_lpt_plan_is_complete_exclusive_deterministic_and_balanced(self):
        plan = _plan()

        first = plan_shards(plan)
        second = plan_shards(plan)

        self.assertEqual(first, second)
        self.assertEqual(2, len(first.shards))
        flattened = [case_id for shard in first.shards for case_id in shard.case_ids]
        self.assertEqual(sorted(case.case_id for case in plan.cases), sorted(flattened))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual([110, 110], [shard.estimated_cost for shard in first.shards])
        self.assertEqual(220, first.total_estimated_cost)
        self.assertEqual(
            ("shard-0000", "shard-0001"),
            tuple(shard.shard_id for shard in first.shards),
        )


if __name__ == "__main__":
    unittest.main()

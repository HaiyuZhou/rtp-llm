import dataclasses
import unittest

from rtp_llm.test.perf_test.config_schema import ConfigError, parse_config
from rtp_llm.test.perf_test.errors import PlanningError
from rtp_llm.test.perf_test.prefill.planner import plan_prefill_cases


def _payload():
    return {
        "schema_version": 1,
        "model": {"label": "test", "type": "registered", "checkpoint": "/m"},
        "engine": {
            "max_seq_len": 4097,
            "max_batch_tokens": 8192,
            "concurrency_limit": 2,
        },
        "workload": {
            "kind": "cache_grid",
            "max_input_len": 2048,
            "cache_alignment": 64,
            "commit_tail_tokens": 128,
            "compute_steps": [512, 256],
            "min_compute_len": 256,
            "cache_lengths": [0, 512],
            "batch_sizes": [1, 2],
            "measure_runs": 3,
            "seed": 17,
            "max_cases": 100,
        },
        "execution": {"backend": "local"},
        "output": {},
        "regression": {},
        "publish": {},
    }


def _parsed(payload=None):
    config = parse_config(payload or _payload())
    return config.workload, config.engine


class PrefillPlannerTest(unittest.TestCase):
    def test_fixed_cache_compute_sweep_is_deterministic_and_deduplicated(self):
        workload, engine = _parsed()

        first = plan_prefill_cases(workload, engine)
        second = plan_prefill_cases(workload, engine)

        self.assertEqual(first, second)
        geometries = [
            (case["batch_size"], case["input_len"], case["cache_len"]) for case in first
        ]
        self.assertEqual(len(geometries), len(set(geometries)))
        self.assertEqual({1, 2}, {case["batch_size"] for case in first})
        self.assertEqual({0, 512}, {case["cache_len"] for case in first})
        for case in first:
            self.assertEqual(case["input_len"] - case["cache_len"], case["compute_len"])
            self.assertGreaterEqual(case["compute_len"], 256)
            self.assertLessEqual(case["input_len"], 2048)
            self.assertEqual(3, case["measure_runs"])
            self.assertEqual(17, case["seed"])

    def test_random_cache_sampling_is_reproducible_and_seeded(self):
        payload = _payload()
        payload["workload"].pop("cache_lengths")
        payload["workload"].update(
            random_cache_count=4, min_cache_len=0, max_cache_len=1024
        )
        workload, engine = _parsed(payload)

        first = plan_prefill_cases(workload, engine)
        repeat = plan_prefill_cases(workload, engine)
        changed = plan_prefill_cases(dataclasses.replace(workload, seed=18), engine)

        first_lengths = sorted({case["cache_len"] for case in first})
        changed_lengths = sorted({case["cache_len"] for case in changed})
        self.assertEqual(first, repeat)
        self.assertEqual(4, len(first_lengths))
        self.assertNotEqual(first_lengths, changed_lengths)
        self.assertTrue(all(value % 128 == 0 for value in first_lengths))

    def test_explicit_cache_lengths_disable_default_random_sampling(self):
        workload, _ = _parsed()
        self.assertEqual((0, 512), workload.cache_lengths)
        self.assertEqual(0, workload.random_cache_count)

        payload = _payload()
        payload["workload"]["random_cache_count"] = 2
        with self.assertRaisesRegex(ConfigError, "cache_lengths.*random_cache_count"):
            parse_config(payload)

    def test_max_cases_is_enforced_after_all_batches(self):
        workload, engine = _parsed()
        workload = dataclasses.replace(workload, max_cases=2)
        with self.assertRaisesRegex(PlanningError, "max_cases=2"):
            plan_prefill_cases(workload, engine)

    def test_every_invalid_planner_boundary_is_rejected(self):
        workload, engine = _parsed()
        cases = [
            (
                "commit tail alignment",
                dataclasses.replace(workload, commit_tail_tokens=96),
                engine,
                "commit_tail_tokens",
            ),
            (
                "engine max sequence",
                workload,
                dataclasses.replace(engine, max_seq_len=2048),
                "max_seq_len",
            ),
            (
                "minimum compute",
                dataclasses.replace(workload, min_compute_len=4096),
                engine,
                "min_compute_len",
            ),
            (
                "cache bounds",
                dataclasses.replace(workload, min_cache_len=1024, max_cache_len=512),
                engine,
                "cache",
            ),
            (
                "unaligned fixed cache",
                dataclasses.replace(
                    workload, cache_lengths=(65,), random_cache_count=0
                ),
                engine,
                "aligned",
            ),
            (
                "duplicate fixed cache",
                dataclasses.replace(
                    workload, cache_lengths=(512, 512), random_cache_count=0
                ),
                engine,
                "unique",
            ),
            (
                "random count exceeds candidates",
                dataclasses.replace(
                    workload,
                    cache_lengths=(),
                    random_cache_count=100,
                    min_cache_len=0,
                    max_cache_len=128,
                ),
                engine,
                "random cache lengths",
            ),
            (
                "batch concurrency",
                dataclasses.replace(workload, batch_sizes=(3,)),
                engine,
                "concurrency_limit",
            ),
            (
                "batch token capacity",
                dataclasses.replace(workload, max_input_len=4096, batch_sizes=(2,)),
                dataclasses.replace(engine, max_seq_len=4097, max_batch_tokens=4096),
                "max_batch_tokens",
            ),
            (
                "compute order",
                dataclasses.replace(workload, compute_steps=(256, 512)),
                engine,
                "coarse-to-fine",
            ),
        ]
        for name, candidate, candidate_engine, diagnostic in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(PlanningError, diagnostic):
                    plan_prefill_cases(candidate, candidate_engine)


if __name__ == "__main__":
    unittest.main()

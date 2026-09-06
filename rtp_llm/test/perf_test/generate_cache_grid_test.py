import argparse
import unittest

from rtp_llm.test.perf_test.generate_cache_grid import (
    DEFAULT_SEED,
    build_grid,
    generate_cache_lengths,
    generate_input_lengths,
)


class GenerateCacheGridTest(unittest.TestCase):
    def test_input_points_are_aligned_and_keep_strict_1m_boundary(self):
        values = generate_input_lengths(
            256, 1048575, 128, 32, "stratified", DEFAULT_SEED
        )
        self.assertIn(1048575, values)
        self.assertTrue(all(x % 128 == 0 for x in values if x != 1048575))

    def test_same_seed_is_reproducible(self):
        first = generate_cache_lengths(65536, 128, 16, 7, DEFAULT_SEED)
        second = generate_cache_lengths(65536, 128, 16, 7, DEFAULT_SEED)
        self.assertEqual(first, second)
        self.assertEqual(first[0], 0)
        self.assertEqual(first[-1], 65408)
        self.assertEqual(len(first), len(set(first)))
        self.assertTrue(all(x % 128 == 0 for x in first))

    def test_different_seed_preserves_boundaries(self):
        first = generate_cache_lengths(65536, 128, 16, 7, DEFAULT_SEED)
        second = generate_cache_lengths(65536, 128, 16, 7, 104723)
        self.assertNotEqual(first, second)
        self.assertEqual((first[0], first[-1]), (second[0], second[-1]))

    def test_cache_alignment_governs_cache_dimension_only(self):
        args = argparse.Namespace(
            min_input_len=2048,
            max_input_len=65536,
            alignment=128,
            cache_alignment=512,
            input_points=8,
            input_mode="stratified",
            seed=DEFAULT_SEED,
            cache_points_per_input=6,
            cache_ratio_points=2,
            batch_size=1,
            max_cases=1000,
            allow_large_grid=False,
            measure_runs=3,
        )
        plan = build_grid(args)
        for case in plan["cases"]:
            self.assertEqual(case["cache_len"] % 512, 0)
        self.assertEqual(plan["generator"]["cache_sampling"]["alignment"], 512)
        # The input dimension keeps its own alignment.
        inputs = {case["input_len"] for case in plan["cases"]}
        self.assertTrue(inputs)
        self.assertTrue(all(x % 128 == 0 for x in inputs))

    def test_cache_alignment_zero_falls_back_to_input_alignment(self):
        args = argparse.Namespace(
            min_input_len=2048,
            max_input_len=65536,
            alignment=128,
            cache_alignment=0,
            input_points=8,
            input_mode="stratified",
            seed=DEFAULT_SEED,
            cache_points_per_input=6,
            cache_ratio_points=2,
            batch_size=1,
            max_cases=1000,
            allow_large_grid=False,
            measure_runs=3,
        )
        plan = build_grid(args)
        self.assertEqual(plan["generator"]["cache_sampling"]["alignment"], 128)
        for case in plan["cases"]:
            self.assertEqual(case["cache_len"] % 128, 0)

    def test_case_limit_rejects_oversized_plan(self):
        args = argparse.Namespace(
            min_input_len=256,
            max_input_len=8192,
            alignment=128,
            cache_alignment=0,
            input_points=32,
            input_mode="stratified",
            seed=DEFAULT_SEED,
            cache_points_per_input=16,
            cache_ratio_points=7,
            batch_size=1,
            max_cases=5,
            allow_large_grid=False,
            measure_runs=3,
        )
        with self.assertRaisesRegex(ValueError, "exceeding --max-cases"):
            build_grid(args)


if __name__ == "__main__":
    unittest.main()

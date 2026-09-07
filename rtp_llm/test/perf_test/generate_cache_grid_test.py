import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.generate_cache_grid import (
    DEFAULT_SEED,
    build_grid,
    generate_cache_lengths,
    generate_input_lengths,
)
from rtp_llm.test.perf_test.perf_profile import fingerprint as profile_fingerprint


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


class GenerateCacheGridProfileTest(unittest.TestCase):
    COMMON_ARGS = [
        "--min-input-len",
        "2048",
        "--max-input-len",
        "65536",
        "--alignment",
        "128",
        "--input-points",
        "8",
        "--cache-points-per-input",
        "6",
        "--cache-ratio-points",
        "2",
    ]

    def _run_grid(self, extra_args, tmp):
        output = Path(tmp) / "grid.json"
        cmd = (
            [
                sys.executable,
                "-m",
                "rtp_llm.test.perf_test.generate_cache_grid",
                "--output",
                str(output),
            ]
            + self.COMMON_ARGS
            + extra_args
        )
        subprocess.run(cmd, check=True, capture_output=True)
        return json.loads(output.read_text(encoding="utf-8"))

    def test_without_profile_no_profile_keys_in_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._run_grid([], tmp)
        self.assertNotIn("profile", payload)
        self.assertNotIn("profile_sha256", payload)

    def test_with_profile_embeds_profile_and_sha256(self):
        profile = {
            "schema_version": 1,
            "label": "test",
            "cache_grid": {"cache_alignment": 512},
        }
        with tempfile.TemporaryDirectory() as tmp:
            profile_path = Path(tmp) / "profile.json"
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            payload = self._run_grid(["--profile", str(profile_path)], tmp)
        self.assertIn("profile", payload)
        self.assertIn("profile_sha256", payload)
        self.assertEqual(payload["profile"], profile)
        self.assertEqual(payload["profile_sha256"], profile_fingerprint(profile))
        for case in payload["cases"]:
            self.assertEqual(case["cache_len"] % 512, 0)

    def test_cli_cache_alignment_wins_over_profile(self):
        profile = {
            "schema_version": 1,
            "cache_grid": {"cache_alignment": 512},
        }
        with tempfile.TemporaryDirectory() as tmp:
            profile_path = Path(tmp) / "profile.json"
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            payload = self._run_grid(
                ["--profile", str(profile_path), "--cache-alignment", "256"], tmp
            )
        for case in payload["cases"]:
            self.assertEqual(case["cache_len"] % 256, 0)
        self.assertEqual(payload["generator"]["cache_sampling"]["alignment"], 256)

    def test_byte_identity_without_profile(self):
        """Output without --profile must be identical to pre-profile code."""
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._run_grid(["--cache-alignment", "0"], tmp)
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
            max_cases=20000,
            allow_large_grid=False,
            measure_runs=3,
        )
        expected = build_grid(args)
        self.assertEqual(payload["cases"], expected["cases"])
        self.assertEqual(payload["generator"], expected["generator"])
        self.assertEqual(payload["summary"], expected["summary"])


if __name__ == "__main__":
    unittest.main()

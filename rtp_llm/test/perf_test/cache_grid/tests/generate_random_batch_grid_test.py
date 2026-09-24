import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rtp_llm.test.perf_test.cache_grid.runner.generate_random_batch_grid import (
    build_plans,
    generate_grid,
    main,
    parse_args,
    round_up,
)
from rtp_llm.test.perf_test.cache_grid.runner.run_random_batch_grids import inspect_grid
from rtp_llm.test.perf_test.cache_grid.runner.run_random_batch_grids import (
    main as run_main,
)


class RandomBatchGridTest(unittest.TestCase):
    def setUp(self):
        stdout = patch("sys.stdout", new_callable=io.StringIO)
        stdout.start()
        self.addCleanup(stdout.stop)

    def args(self, *extra):
        return parse_args(["--output", "/tmp/random_batch_test.json", *extra])

    def test_reproducible_constrained_batches(self):
        args = self.args("--num-cases", "300", "--kv-budget-tokens", "400000")
        grid = generate_grid(args)
        self.assertEqual(grid, generate_grid(args))
        for case in grid["cases"]:
            groups = case["request_groups"]
            self.assertEqual(sum(g["count"] for g in groups), case["batch_size"])
            self.assertLessEqual(
                sum(g["input_len"] for g in groups), args.max_batch_tokens
            )
            peak = 0
            for g in groups:
                self.assertLessEqual(g["input_len"], 262144)
                self.assertEqual(g["input_len"] % 256, 0)
                self.assertEqual(g["cache_len"] % 4096, 0)
                self.assertLess(g["cache_len"], g["input_len"])
                if g["cache_len"]:
                    self.assertGreaterEqual(g["input_len"] - g["cache_len"], 4096)
                peak += round_up(g["input_len"], 4096) + (4096 if g["cache_len"] else 0)
            self.assertEqual(peak, case["estimated_peak_kv_tokens"])
            self.assertLessEqual(peak, 400000)

    def test_small_unaligned_minimum_with_tight_kv_budget(self):
        grid = generate_grid(
            self.args(
                "--num-cases",
                "1",
                "--batch-sizes",
                "2",
                "--min-input-tokens",
                "257",
                "--max-input-tokens",
                "512",
                "--kv-budget-tokens",
                "8192",
            )
        )
        self.assertEqual(grid["cases"][0]["estimated_peak_kv_tokens"], 8192)

    def test_impossible_and_oversized_inputs_rejected(self):
        for extra in [
            ("--max-input-tokens", "0"),
            ("--max-batch-tokens", "100"),
            ("--kv-budget-tokens", "100"),
            ("--commit-tail-tokens", "1"),
        ]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                generate_grid(self.args(*extra))

    def test_exhausted_shape_space_rejected(self):
        with self.assertRaisesRegex(ValueError, "distinct batches"):
            generate_grid(
                self.args(
                    "--num-cases",
                    "2",
                    "--batch-sizes",
                    "1",
                    "--min-input-tokens",
                    "256",
                    "--max-input-tokens",
                    "256",
                )
            )

    def test_directory_plans_shrink_large_batch_lengths(self):
        args = parse_args(
            [
                "--output-dir",
                "/tmp/batch_plans",
                "--num-cases",
                "5",
                "--batch-sizes",
                "1",
                "8",
                "31",
            ]
        )
        plans = build_plans(args)
        self.assertEqual(plans, build_plans(args))
        for _, grid in plans:
            batches = {case["batch_size"] for case in grid["cases"]}
            self.assertEqual(len(batches), 1)
            batch = batches.pop()
            limit = grid["summary"]["max_request_tokens"]
            self.assertLessEqual(limit, 262144)
            self.assertLessEqual(batch * limit, 1048576)

    def test_batch_limit_override(self):
        args = parse_args(
            [
                "--output-dir",
                "/tmp/batch_plans",
                "--num-cases",
                "5",
                "--batch-sizes",
                "31",
                "--batch-input-limit",
                "31:65536",
            ]
        )
        self.assertEqual(
            build_plans(args)[0][1]["summary"]["max_request_tokens"], 33792
        )
        args.batch_input_limit = ["31:262145"]
        with self.assertRaises(ValueError):
            build_plans(args)

    def test_custom_commit_tail_alignment(self):
        grid = generate_grid(self.args("--commit-tail-tokens", "8192"))
        for case in grid["cases"]:
            for group in case["request_groups"]:
                self.assertEqual(group["cache_len"] % 8192, 0)

    def profile(self, root, model_type="qwen_2", engine_env=None):
        path = root / "profile.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "engine": {
                        "model_type": model_type,
                        "checkpoint_path": "/weights/model",
                        "tokenizer_path": "/weights/model",
                        "tp_size": 1,
                        "dp_size": 1,
                        "max_seq_len": 2097152,
                    },
                    "cache_grid": {
                        "expected_block_size": 4096,
                        "commit_tail_tokens": 4096,
                    },
                    "engine_env": engine_env or {},
                    "bazel": {"configs": ["cuda12"]},
                }
            )
        )
        return path

    def test_generic_grid_accepts_other_geometry_and_larger_budget(self):
        grid = generate_grid(
            self.args(
                "--num-cases",
                "10",
                "--batch-sizes",
                "2",
                "--min-input-tokens",
                "524288",
                "--max-input-tokens",
                "1048576",
                "--max-batch-tokens",
                "4194304",
                "--cache-alignment",
                "64",
                "--input-alignment",
                "64",
                "--commit-tail-tokens",
                "64",
            )
        )
        self.assertEqual(grid["generator"]["workspace_policy"], "none")
        for case in grid["cases"]:
            for group in case["request_groups"]:
                self.assertGreater(group["input_len"], 262144)
                self.assertEqual(group["cache_len"] % 64, 0)

    def test_directory_writer_and_profile_runner(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            main(
                [
                    "--output-dir",
                    str(root / "grids"),
                    "--num-cases",
                    "3",
                    "--batch-sizes",
                    "2",
                    "16",
                ]
            )
            profile = self.profile(root)
            argv = [
                "--grid-dir",
                str(root / "grids"),
                "--profile",
                str(profile),
                "--result-root",
                str(root / "results"),
                "--output-base",
                str(root / "bazel"),
                "--skip-reuse-validation",
            ]
            with patch.dict("os.environ", {}, clear=True), patch(
                "rtp_llm.test.perf_test.cache_grid.runner.cache_perf.subprocess.run"
            ) as run:
                run.return_value.returncode = 0
                self.assertEqual(run_main(argv + ["--dry-run"]), 0)
                run.assert_not_called()
                self.assertFalse((root / "results").exists())
                self.assertEqual(run_main(argv), 0)
                self.assertEqual(run.call_count, 2)
                for call in run.call_args_list:
                    command = call.args[0]
                    self.assertIn("--config=cuda12", command)
                    self.assertNotIn("--config=sm10x", command)
                    self.assertIn("--test_arg=--cache_skip_reuse_validation", command)
                    self.assertNotIn("--test_arg=--cache_fixed_workspace", command)
                    self.assertNotIn("DSV4", " ".join(command))
                    self.assertNotIn("deepseek", " ".join(command))
            manifest = json.loads((root / "results" / "batch_runs.json").read_text())
            self.assertEqual([p["status"] for p in manifest["runs"]], ["completed"] * 2)
            for plan in manifest["runs"]:
                result = Path(plan["result_dir"])
                saved = json.loads((result / "profile.snapshot.json").read_text())
                self.assertEqual(saved["engine"]["model_type"], "qwen_2")
                self.assertEqual(saved["engine"]["tp_size"], 1)
                self.assertTrue((result / "cache_perf_launch.json").exists())
                self.assertTrue((result / "grid.snapshot.json").exists())
            with self.assertRaises(SystemExit):
                run_main(argv)

    def test_model_specific_environment_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            main(
                [
                    "--output-dir",
                    str(root / "grids"),
                    "--num-cases",
                    "1",
                    "--batch-sizes",
                    "1",
                ]
            )
            profile = self.profile(root, "deepseek_v4", {"DSV4_USE_MEGA_MOE": "1"})
            with patch.dict("os.environ", {}, clear=True), patch(
                "rtp_llm.test.perf_test.cache_grid.runner.cache_perf.subprocess.run"
            ) as run:
                run.return_value.returncode = 0
                self.assertEqual(
                    run_main(
                        [
                            "--grid-dir",
                            str(root / "grids"),
                            "--profile",
                            str(profile),
                            "--result-root",
                            str(root / "results"),
                            "--env",
                            "DSV4_USE_MEGA_MOE_SE=0",
                        ]
                    ),
                    0,
                )
                command = run.call_args.args[0]
                self.assertIn("--test_env=DSV4_USE_MEGA_MOE=1", command)
                self.assertIn("--test_env=DSV4_USE_MEGA_MOE_SE=0", command)

    def test_runner_stops_on_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            main(
                [
                    "--output-dir",
                    str(root / "grids"),
                    "--num-cases",
                    "2",
                    "--batch-sizes",
                    "2",
                    "4",
                ]
            )
            with patch(
                "rtp_llm.test.perf_test.cache_grid.runner.cache_perf.subprocess.run"
            ) as run:
                run.return_value.returncode = 3
                code = run_main(
                    [
                        "--grid-dir",
                        str(root / "grids"),
                        "--profile",
                        str(self.profile(root)),
                        "--result-root",
                        str(root / "results"),
                    ]
                )
                self.assertEqual(code, 3)
                self.assertEqual(run.call_count, 1)
            manifest = json.loads((root / "results" / "batch_runs.json").read_text())
            self.assertEqual(
                [p["status"] for p in manifest["runs"]], ["failed", "pending"]
            )

    def test_generic_inspection_and_mixed_batch_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "grid.json"
            case = {"case_id": 0, "batch_size": 1, "input_len": 524288, "cache_len": 64}
            path.write_text(json.dumps({"cases": [case]}))
            self.assertEqual(inspect_grid(path)["batch_size"], 1)
            path.write_text(json.dumps({"cases": [case, {**case, "batch_size": 2}]}))
            with self.assertRaisesRegex(ValueError, "mixed batch"):
                inspect_grid(path)


if __name__ == "__main__":
    unittest.main()

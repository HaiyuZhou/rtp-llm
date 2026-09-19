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
            ("--max-input-tokens", "262145"),
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

    def test_pro_keeps_mega_moe_family_enabled(self):
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
            with patch(
                "rtp_llm.test.perf_test.cache_grid.runner.run_random_batch_grids.subprocess.run"
            ) as run:
                run.return_value.returncode = 0
                self.assertEqual(
                    run_main(
                        [
                            "--grid-dir",
                            str(root / "grids"),
                            "--model-dir",
                            "/weights/Pro",
                            "--result-root",
                            str(root / "results"),
                        ]
                    ),
                    0,
                )
                command = run.call_args.args[0]
                self.assertIn("--test_env=DSV4_USE_MEGA_MOE_SE=1", command)
                self.assertIn("--test_env=DSV4_USE_MEGA_MOE=1", command)

    def test_directory_writer_and_serial_runner(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            grids = root / "grids"
            main(
                [
                    "--output-dir",
                    str(grids),
                    "--num-cases",
                    "3",
                    "--batch-sizes",
                    "2",
                    "16",
                ]
            )
            files = sorted(grids.glob("*.json"))
            self.assertEqual(len(files), 2)
            inspected = [inspect_grid(path) for path in files]
            self.assertEqual(inspected[0]["max_seq_len"], inspected[1]["max_seq_len"])
            runner_args = [
                "--grid-dir",
                str(grids),
                "--model-dir",
                "/weights/Pro",
                "--result-root",
                str(root / "results"),
                "--output-base",
                str(root / "bazel"),
                "--mega-moe-se",
                "0",
            ]
            with patch(
                "rtp_llm.test.perf_test.cache_grid.runner.run_random_batch_grids.subprocess.run"
            ) as run:
                run.return_value.returncode = 0
                self.assertEqual(run_main(runner_args + ["--dry-run"]), 0)
                run.assert_not_called()
                self.assertFalse((root / "results").exists())
                self.assertEqual(run_main(runner_args), 0)
                self.assertEqual(run.call_count, 2)
                for call, plan in zip(run.call_args_list, inspected):
                    command = call.args[0]
                    self.assertEqual(command[1], f"--output_base={root / 'bazel'}")
                    self.assertEqual(command[2], "test")
                    self.assertIn("--test_env=DSV4_USE_MEGA_MOE_SE=0", command)
                    self.assertIn("--test_env=DSV4_USE_MEGA_MOE=1", command)
                    self.assertIn(
                        f"--test_arg=--max_seq_len={plan['max_seq_len']}", command
                    )
                    self.assertIn(
                        "--test_arg=--max_context_batch_size=1",
                        command,
                    )
                    self.assertNotIn("--test_arg=--cache_shared_seed", command)
            manifest = json.loads((root / "results" / "batch_runs.json").read_text())
            self.assertEqual(
                [p["status"] for p in manifest["runs"]], ["completed", "completed"]
            )
            with self.assertRaises(SystemExit):
                run_main(runner_args)

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
                "rtp_llm.test.perf_test.cache_grid.runner.run_random_batch_grids.subprocess.run"
            ) as run:
                run.return_value.returncode = 3
                code = run_main(
                    [
                        "--grid-dir",
                        str(root / "grids"),
                        "--model-dir",
                        "/weights/Pro",
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

    def test_probe_length_floor_and_mixed_batch_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "grid.json"
            case = {"case_id": 0, "batch_size": 1, "input_len": 256, "cache_len": 0}
            path.write_text(json.dumps({"cases": [case]}))
            self.assertEqual(inspect_grid(path)["max_seq_len"], 1048576)
            path.write_text(json.dumps({"cases": [case, {**case, "batch_size": 2}]}))
            with self.assertRaisesRegex(ValueError, "mixed batch"):
                inspect_grid(path)


if __name__ == "__main__":
    unittest.main()

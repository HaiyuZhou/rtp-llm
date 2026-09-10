import argparse
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.run_cache_grid_pipeline import (
    MODULES,
    build_commands,
    run_pipeline,
)


class CacheGridPipelineTest(unittest.TestCase):
    def _args(self, root: Path, **overrides):
        values = {
            "cache_grid_json": root / "grid.json",
            "result_dir": root / "results",
            "profile": root / "profile.json",
            "batch_size": 1,
            "estimator": "min",
            "formula_output_dir": None,
            "svg_output": None,
            "cold_svg_output": None,
            "html_output": None,
            "skip_test": True,
            "runner_args": ["--", "--cache_measure_runs=3", "--tp_size=8"],
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_build_commands_forwards_runner_args_and_owns_shared_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            commands = build_commands(self._args(Path(temporary)))
        self.assertIn("--partial=2", commands["runner"])
        self.assertIn("--cache_measure_runs=3", commands["runner"])
        self.assertIn("--tp_size=8", commands["runner"])
        self.assertEqual(commands["fit"][2], MODULES["fit"])
        self.assertIn("--estimator", commands["fit"])
        svg_path = commands["svg"][commands["svg"].index("--output") + 1]
        html_path = commands["html"][commands["html"].index("--output") + 1]
        self.assertTrue(svg_path.endswith("prefill_3d.svg"))
        self.assertTrue(html_path.endswith("prefill_3d.interactive.html"))

    def test_rejects_duplicate_pipeline_owned_runner_argument(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self._args(Path(temporary), runner_args=["--result_dir=/other"])
            with self.assertRaisesRegex(ValueError, "managed by the pipeline"):
                build_commands(args)

    def test_fit_rejection_still_generates_both_charts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self._args(root)
            args.result_dir.mkdir()
            (args.result_dir / "cache_grid_results.json").write_text(
                json.dumps(
                    {"complete": True, "completed_cases": 10, "total_cases": 10}
                ),
                encoding="utf-8",
            )
            called = []

            def fake_run(command, check):
                called.append(command[2])
                code = 3 if command[2] == MODULES["fit"] else 0
                return subprocess.CompletedProcess(command, code)

            self.assertEqual(run_pipeline(args, fake_run), 3)
            self.assertEqual(called, [MODULES["fit"], MODULES["svg"], MODULES["html"]])
            manifest = json.loads(
                (args.result_dir / "pipeline_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "fit_rejected")

    def test_incomplete_result_is_not_post_processed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self._args(root)
            args.result_dir.mkdir()
            (args.result_dir / "cache_grid_results.json").write_text(
                json.dumps({"complete": False, "status": "in_progress"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                run_pipeline(args, lambda *unused, **kwargs: None)


if __name__ == "__main__":
    unittest.main()

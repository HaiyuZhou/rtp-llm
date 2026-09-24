import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rtp_llm.test.perf_test.cache_grid.runner import cache_perf as cli


class CachePerfPipelineTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.result = self.root / "results"
        self.profile = self.root / "profile.json"
        self.profile.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "model_label": "Original Model",
                    "engine": {"model_type": "qwen_2", "tp_size": 1},
                    "engine_env": {"PERF_PROFILE_RUNS": "0"},
                    "bazel": {"configs": ["cuda12"]},
                }
            )
        )
        self.grid = self.root / "grid.json"
        self.grid.write_text(
            json.dumps(
                {
                    "cases": [
                        {"case_id": 0, "input_len": 8192, "cache_len": 0},
                    ]
                }
            )
        )
        output = patch("sys.stdout", new_callable=io.StringIO)
        output.start()
        self.addCleanup(output.stop)

    def argv(self, *extra, launch=True):
        argv = ["pipeline", "--result-dir", str(self.result), *extra]
        if launch:
            argv += ["--profile", str(self.profile), "--grid", str(self.grid)]
        return argv

    def result_file(self, complete=True):
        self.result.mkdir(exist_ok=True)
        (self.result / "cache_grid_results.json").write_text(
            json.dumps(
                {
                    "complete": complete,
                    "completed_cases": 1,
                    "total_cases": 1,
                }
            )
        )

    def manifest(self):
        return json.loads((self.result / "pipeline_summary.json").read_text())

    def test_pipeline_uses_bazel_snapshots_and_frozen_profile(self):
        calls = []

        def run(command, **kwargs):
            calls.append(command)
            if command[1] == "test":
                self.assertIn(cli.TARGET, command)
                self.assertIn("--config=cuda12", command)
                self.assertIn("--test_arg=--cache_measure_runs=4", command)
                self.assertEqual(kwargs["env"]["PERF_PROFILE_RUNS"], "0")
                self.assertTrue((self.result / cli.MANIFEST).exists())
                self.assertTrue((self.result / "grid.snapshot.json").exists())
                self.profile.write_text("{}")
                self.result_file()
            else:
                self.assertIn(
                    "--profile=" + str(self.result / "profile.snapshot.json"), command
                )
                frozen = json.loads((self.result / "profile.snapshot.json").read_text())
                self.assertEqual(frozen["model_label"], "Original Model")
            return subprocess.CompletedProcess(command, 0)

        with patch.object(cli.subprocess, "run", side_effect=run):
            self.assertEqual(cli.main(self.argv("--runs", "4")), 0)
        self.assertEqual(
            [c[2] for c in calls[1:]], list(cli.POSTPROCESS_MODULES.values())
        )
        self.assertEqual(self.manifest()["status"], "completed")

    def test_dry_run_writes_nothing_and_launches_nothing(self):
        with patch.object(cli.subprocess, "run") as run:
            self.assertEqual(cli.main(self.argv("--dry-run")), 0)
            run.assert_not_called()
        self.assertFalse(self.result.exists())

    def test_skip_test_keeps_quality_gate_and_custom_outputs(self):
        self.result_file()
        calls = []

        def run(command, **kwargs):
            calls.append(command)
            code = 3 if command[2] == cli.POSTPROCESS_MODULES["fit"] else 0
            return subprocess.CompletedProcess(command, code)

        with patch.object(cli.subprocess, "run", side_effect=run):
            code = cli.main(
                self.argv(
                    "--skip-test",
                    "--estimator",
                    "min",
                    "--batch-size",
                    "2",
                    "--svg-output",
                    str(self.root / "custom.svg"),
                    launch=False,
                )
            )
        self.assertEqual(code, 3)
        self.assertEqual(len(calls), 4)
        self.assertIn("min", calls[0])
        self.assertIn(str(self.root / "custom.svg"), calls[1])
        self.assertIn("--all-runs", calls[2])
        self.assertIn("--all-runs", calls[3])
        self.assertIn("tpm-effective", calls[3])
        self.assertNotIn("tpm-compute", " ".join(arg for call in calls for arg in call))
        self.assertEqual(self.manifest()["status"], "fit_rejected")
        self.assertEqual(self.manifest()["stages"]["test"], {"skipped": True})

    def test_test_failure_stops_postprocessing_and_preserves_code(self):
        with patch.object(
            cli.subprocess, "run", return_value=subprocess.CompletedProcess([], 7)
        ) as run:
            self.assertEqual(cli.main(self.argv()), 7)
            self.assertEqual(run.call_count, 1)
        self.assertEqual(self.manifest()["status"], "failed")
        self.assertEqual(self.manifest()["stages"], {"test": {"returncode": 7}})

    def test_incomplete_results_are_rejected_without_postprocessing(self):
        self.result_file(complete=False)
        args = cli.parser().parse_args(self.argv("--skip-test", launch=False))
        with patch.object(cli.subprocess, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                cli.run_pipeline(args)
            run.assert_not_called()
        self.assertFalse((self.result / "pipeline_summary.json").exists())

    def test_postprocessing_failure_is_not_a_quality_gate(self):
        self.result_file()
        with patch.object(
            cli.subprocess, "run", return_value=subprocess.CompletedProcess([], 2)
        ) as run:
            self.assertEqual(cli.main(self.argv("--skip-test", launch=False)), 2)
            self.assertEqual(run.call_count, 1)
        self.assertEqual(self.manifest()["status"], "failed")

    def test_chart_failure_is_recorded(self):
        self.result_file()
        with patch.object(
            cli.subprocess,
            "run",
            side_effect=[
                subprocess.CompletedProcess([], 0),
                subprocess.CompletedProcess([], 9),
            ],
        ) as run:
            self.assertEqual(cli.main(self.argv("--skip-test", launch=False)), 9)
            self.assertEqual(run.call_count, 2)
        self.assertEqual(self.manifest()["stages"]["svg"]["returncode"], 9)

    def test_resume_uses_existing_launch_configuration(self):
        args = cli.parser().parse_args(self.argv())
        args.mode = "run"
        plan = cli.build_plan(args, {})
        self.result.mkdir()
        for name, data in plan["artifacts"].items():
            (self.result / name).write_bytes(data)
        self.result_file(complete=False)
        frozen = (self.result / "profile.snapshot.json").read_bytes()
        self.profile.unlink()
        self.grid.unlink()

        def run(command, **kwargs):
            if command[1] == "test":
                self.assertIn("--test_arg=--require_cache_resume", command)
                self.result_file()
            return subprocess.CompletedProcess(command, 0)

        with patch.object(cli.subprocess, "run", side_effect=run):
            self.assertEqual(
                cli.main(self.argv("--test-mode", "resume", launch=False)), 0
            )
        self.assertEqual((self.result / "profile.snapshot.json").read_bytes(), frozen)

    def test_skip_test_rejects_launch_overrides(self):
        args = cli.parser().parse_args(self.argv("--skip-test"))
        with self.assertRaisesRegex(ValueError, "launch overrides"):
            cli.run_pipeline(args)

    def test_pipeline_does_not_accept_profiler_mode(self):
        args = cli.parser().parse_args(self.argv("--profile-backend", "nsys"))
        with self.assertRaisesRegex(ValueError, "formal measurements"):
            cli.run_pipeline(args)


if __name__ == "__main__":
    unittest.main()

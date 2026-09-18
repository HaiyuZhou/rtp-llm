import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rtp_llm.test.perf_test.cache_grid.config.perf_profile import (
    ProfileError,
    fingerprint,
    load_profile,
    profile_environment,
    strip_json_comments,
)
from rtp_llm.test.perf_test.cache_grid.runner import cache_perf as cli


class CachePerfTest(unittest.TestCase):
    def test_canonical_modules_have_no_legacy_forwarding_files(self):
        package = "rtp_llm.test.perf_test"
        root = Path(__file__).resolve().parents[1]
        for category in ("runner", "plot", "formula", "config"):
            for path in (root / category).glob("*.py"):
                if path.stem in ("__init__", "cache_perf"):
                    continue
                with self.subTest(module=path.stem):
                    module = importlib.import_module(
                        f"{package}.cache_grid.{category}.{path.stem}"
                    )
                    self.assertEqual(Path(module.__file__).resolve(), path.resolve())
                    self.assertFalse((root.parent / path.name).exists())

    def test_launchers_resolve_repository_after_move(self):
        from rtp_llm.test.perf_test.cache_grid.runner.run_random_batch_grids import (
            REPO_ROOT,
        )

        self.assertEqual(cli.REPO, REPO_ROOT)
        self.assertEqual(cli.REPO, Path(__file__).resolve().parents[5])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.profile = self.root / "local.jsonc"
        self.profile.write_text(
            """{
          // local paths are literal, not shell expressions
          "schema_version":1,
          "engine":{"checkpoint_path":"/model","tokenizer_path":"/model"},
          "cache_grid":{"expected_block_size":4096,"measure_runs":5},
          "engine_env":{"DSV4_CHUNK_TOKENS":"8192"},
          "runtime_env":{"CC":"/gcc"}
        }"""
        )
        self.grid = self.root / "grid.json"
        self.grid.write_text(
            json.dumps(
                {
                    "cases": [
                        {"case_id": 7, "input_len": 8192, "cache_len": 4096},
                        {"case_id": 9, "input_len": 16384, "cache_len": 0},
                    ]
                }
            )
        )
        self.result = self.root / "results"

    def args(self, mode="run", extra=(), explicit=True):
        args = [mode, "--result-dir", str(self.result), *extra]
        if explicit:
            args += ["--profile", str(self.profile), "--grid", str(self.grid)]
        return cli.parser().parse_args(args)

    def save_run(self):
        plan = cli.build_plan(
            self.args(), {"PATH": "/usr/bin", "DSV4_CHUNK_TOKENS": "1"}
        )
        self.result.mkdir()
        for name, data in plan["artifacts"].items():
            (self.result / name).write_bytes(data)
        (self.result / "cache_grid_results.json").write_text("{}")
        return plan

    def test_jsonc_preserves_urls_strings_and_fingerprint(self):
        text = '{"schema_version":1,"url":"https://a/b/*c*/","quote":"a\\"//b" /* x */}'
        parsed = json.loads(strip_json_comments(text))
        self.assertEqual(parsed["url"], "https://a/b/*c*/")
        self.assertEqual(
            fingerprint(parsed), fingerprint(json.loads(json.dumps(parsed)))
        )
        with self.assertRaises(ProfileError):
            strip_json_comments("{ /* missing close")

    def test_standard_json_still_rejects_comments(self):
        path = self.root / "strict.json"
        path.write_text(self.profile.read_text())
        with self.assertRaises(ProfileError):
            load_profile(path)

    def test_invalid_or_sensitive_env_rejected(self):
        for values in (
            {"DSV4_X": True},
            {"BAD-NAME": "x"},
            {"API_KEY": "secret"},
            {"PATH": "$PATH:/x"},
        ):
            with self.assertRaises(ProfileError):
                profile_environment({"engine_env": values})
        with self.assertRaises(ProfileError):
            profile_environment({"runtime_env": {"CC": "a"}, "engine_env": {"CC": "b"}})

    def test_env_priority_and_bazel_command(self):
        plan = cli.build_plan(
            self.args(
                extra=["--env", "DSV4_CHUNK_TOKENS=4096", "--output-base", "/tmp/build"]
            ),
            {"PATH": "/usr/bin", "DSV4_CHUNK_TOKENS": "1"},
        )
        self.assertEqual(plan["process_env"]["DSV4_CHUNK_TOKENS"], "4096")
        self.assertEqual(
            plan["command"][:3], ["bazelisk", "--output_base=/tmp/build", "test"]
        )
        self.assertIn("--test_env=CC=/gcc", plan["command"])
        self.assertIn("--test_arg=--engine_env=CC=/gcc", plan["command"])
        self.assertFalse(self.result.exists())

    def test_dry_run_does_not_write_or_launch(self):
        with patch.object(cli.subprocess, "run") as run, patch("builtins.print"):
            rc = cli.main(
                [
                    "run",
                    "--profile",
                    str(self.profile),
                    "--grid",
                    str(self.grid),
                    "--result-dir",
                    str(self.result),
                    "--dry-run",
                ]
            )
        self.assertEqual(rc, 0)
        run.assert_not_called()
        self.assertFalse(self.result.exists())

    def test_run_refuses_existing_results(self):
        self.save_run()
        with self.assertRaisesRegex(ValueError, "empty/new"):
            cli.build_plan(self.args(), {})

    def test_resume_restores_env_and_requires_checkpoint(self):
        before = self.save_run()
        after = cli.build_plan(
            self.args("resume", explicit=False), {"DSV4_CHUNK_TOKENS": "999"}
        )
        self.assertEqual(
            before["summary"]["environment"], after["summary"]["environment"]
        )
        self.assertIn("--test_arg=--require_cache_resume", after["command"])
        self.assertEqual(after["artifacts"], {})

    def test_resume_missing_checkpoint_or_overrides_rejected(self):
        with self.assertRaisesRegex(ValueError, "existing"):
            cli.build_plan(self.args("resume", explicit=False), {})
        self.save_run()
        with self.assertRaisesRegex(ValueError, "overrides"):
            cli.build_plan(
                self.args("resume", extra=["--runs", "3"], explicit=False), {}
            )

    def test_changed_snapshot_rejected(self):
        self.save_run()
        (self.result / "grid.snapshot.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "changed"):
            cli.build_plan(self.args("resume", explicit=False), {})

    def test_retest_selects_only_requested_cases(self):
        self.save_run()
        plan = cli.build_plan(
            self.args("retest", extra=["--cases", "9", "--runs", "3"], explicit=False),
            {},
        )
        grid = json.loads(plan["artifacts"]["grid.snapshot.json"])
        self.assertEqual([c["case_id"] for c in grid["cases"]], [9])
        self.assertIn("--test_arg=--cache_profile_runs=0", plan["command"])
        self.assertIn("--test_arg=--cache_measure_runs=3", plan["command"])
        self.assertNotIn("--test_arg=--require_cache_resume", plan["command"])
        self.assertTrue(
            plan["destination"].is_relative_to(self.result / "cache_perf_replays")
        )

    def test_profile_never_resumes_formal_grid(self):
        plan = cli.build_plan(
            self.args("profile", extra=["--cases", "7", "--runs", "2"]), {}
        )
        self.assertIn("--test_arg=--cache_profile_only", plan["command"])
        self.assertIn("--test_arg=--cache_profile_runs=2", plan["command"])
        self.assertEqual(plan["summary"]["planned_cases"], 1)
        self.assertFalse(plan["summary"]["reads_checkpoint"])

    def test_unknown_case_and_invalid_runs(self):
        for options in (["--cases", "888"], ["--cases", "7", "--runs", "0"]):
            with self.assertRaises(ValueError):
                cli.build_plan(self.args("retest", extra=options), {})

    def test_main_propagates_bazel_exit_code(self):
        with patch.object(cli.subprocess, "run") as run, patch("builtins.print"):
            run.return_value.returncode = 9
            rc = cli.main(
                [
                    "run",
                    "--profile",
                    str(self.profile),
                    "--grid",
                    str(self.grid),
                    "--result-dir",
                    str(self.result),
                ]
            )
        self.assertEqual(rc, 9)
        self.assertTrue((self.result / cli.MANIFEST).exists())
        self.assertEqual(run.call_args.kwargs["cwd"], cli.REPO)

    def test_explicit_default_value_overrides_profile(self):
        from rtp_llm.test.perf_test.batch_decode_test import parse_args

        args, _ = parse_args(
            ["--profile", str(self.profile), "--cache_measure_runs", "3"]
        )
        self.assertEqual(args.cache_measure_runs, 3)

    def test_legacy_metadata_retest(self):
        self.result.mkdir()
        info = {
            "argv": [
                "main.py",
                "--partial=2",
                "--decode_test_length=1",
                "--checkpoint_path=/model",
                "--cache_grid_json=" + str(self.grid),
                "--result_dir=" + str(self.result),
            ],
            "cache_grid_json": str(self.grid),
            "engine_environment": {},
            "profile": None,
        }
        (self.result / "test_info.json").write_text(json.dumps(info))
        plan = cli.build_plan(
            self.args("retest", extra=["--cases", "7"], explicit=False), {}
        )
        self.assertIn("--test_arg=--checkpoint_path=/model", plan["command"])


if __name__ == "__main__":
    unittest.main()

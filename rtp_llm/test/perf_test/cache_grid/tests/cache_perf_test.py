import importlib
import json
import shlex
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

    def test_nsys_profile_wraps_bazel_and_forwards_capture_options(self):
        plan = cli.build_plan(
            self.args(
                "profile",
                extra=[
                    "--cases=7",
                    "--runs=3",
                    "--profile-backend=nsys",
                    "--nsys-path=/opt/nsys tools/nsys",
                    "--nsys-session=case19314",
                    "--nsys-tail-seconds=0.2",
                ],
            ),
            {},
        )
        command = plan["command"]
        wrapper = shlex.split(
            next(x.split("=", 1)[1] for x in command if x.startswith("--run_under="))
        )
        self.assertEqual(wrapper[:2], ["/opt/nsys tools/nsys", "launch"])
        self.assertIn("--session-new=case19314", wrapper)
        self.assertIn("--wait=all", wrapper)
        self.assertIn("--test_arg=--cache_profile_backend=nsys", command)
        self.assertIn("--test_arg=--cache_profile_runs=3", command)
        self.assertIn("--test_arg=--cache_nsys_tail_seconds=0.2", command)
        self.assertIn("--test_env=GEN_TIMELINE_SYNC=0", command)
        self.assertIn("--test_arg=--gen_timeline_sync=False", command)
        manifest = json.loads(plan["artifacts"][cli.MANIFEST])
        self.assertIn("--cache_nsys_session=case19314", manifest["runner_args"])
        self.assertFalse(plan["destination"].exists())

    def test_nsys_default_session_is_unique_and_kineto_has_no_wrapper(self):
        args = self.args("profile", extra=["--cases=7", "--profile-backend=nsys"])
        commands = [cli.build_plan(args, {})["command"] for _ in range(2)]
        sessions = [
            next(x for x in c if x.startswith("--test_arg=--cache_nsys_session="))
            for c in commands
        ]
        self.assertNotEqual(*sessions)
        normal = cli.build_plan(self.args("profile", extra=["--cases=7"]), {})
        self.assertFalse(any(x.startswith("--run_under=") for x in normal["command"]))

    def test_nsys_rejects_nonprofile_and_invalid_tail(self):
        for mode, extra in [
            ("run", ["--profile-backend=nsys"]),
            ("profile", ["--cases=7", "--nsys-tail-seconds=-1"]),
            ("profile", ["--cases=7", "--nsys-tail-seconds=nan"]),
        ]:
            with self.subTest(mode=mode, extra=extra), self.assertRaises(ValueError):
                cli.build_plan(self.args(mode, extra=extra), {})

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

    def test_run_forwards_and_resume_freezes_skip_reuse_validation(self):
        first = cli.build_plan(
            self.args(extra=["--skip-reuse-validation"]),
            {"PATH": "/usr/bin"},
        )
        self.assertIn("--test_arg=--cache_skip_reuse_validation", first["command"])
        self.assertTrue(first["summary"]["skip_reuse_validation"])
        self.result.mkdir()
        for name, data in first["artifacts"].items():
            (self.result / name).write_bytes(data)
        (self.result / "cache_grid_results.json").write_text("{}")
        resumed = cli.build_plan(self.args("resume", explicit=False), {})
        self.assertIn("--test_arg=--cache_skip_reuse_validation", resumed["command"])
        with self.assertRaisesRegex(ValueError, "overrides"):
            cli.build_plan(
                self.args(
                    "resume",
                    extra=["--skip-reuse-validation"],
                    explicit=False,
                ),
                {},
            )

    def test_launch_v2_stores_profile_and_environment_once(self):
        plan = self.save_run()
        raw = json.loads(plan["artifacts"][cli.MANIFEST])
        self.assertEqual(raw["schema_version"], 2)
        self.assertNotIn("profile", raw)
        self.assertEqual(raw["profile_file"], "profile.snapshot.json")
        self.assertFalse(any(a.startswith("--engine_env=") for a in raw["runner_args"]))
        saved = cli.load_saved(self.result)
        self.assertEqual(saved["profile"], load_profile(self.profile))
        for key, value in saved["env"].items():
            self.assertEqual(
                saved["runner_args"].count(f"--engine_env={key}={value}"), 1
            )

    def test_v1_launch_manifest_remains_replayable(self):
        plan = self.save_run()
        saved = cli.load_saved(self.result)
        saved["schema_version"] = 1
        saved.pop("profile_file")
        (self.result / cli.MANIFEST).write_text(json.dumps(saved))
        replay = cli.build_plan(self.args("resume", explicit=False), {})
        before = [a for a in plan["command"] if a.startswith("--test_env=")]
        after = [a for a in replay["command"] if a.startswith("--test_env=")]
        self.assertEqual(before, after)
        self.assertEqual(replay["artifacts"], {})

    def test_compact_test_info_preserves_attempts_and_resume_fingerprint(self):
        from rtp_llm.test.perf_test.batch_decode_test import (
            _build_cache_resume_config,
            _write_test_info,
            parse_args,
        )
        from rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner import (
            resume_config_fingerprint,
        )

        plan = self.save_run()
        saved = cli.load_saved(self.result)
        args, remaining = parse_args(saved["runner_args"])
        with patch.dict("os.environ", saved["env"], clear=True):
            config = _build_cache_resume_config(
                args, remaining, list(saved["env"]), 4096
            )
            for status in ("running", "completed", "running", "completed"):
                _write_test_info(
                    args,
                    remaining,
                    list(saved["env"]),
                    status=status,
                    expected_cache_block_size=4096,
                    resume_config=config,
                )
        text = (self.result / "test_info.json").read_text()
        info = json.loads(text)
        self.assertEqual(info["schema_version"], 4)
        self.assertEqual(info["attempt_count"], 2)
        self.assertEqual(info["status"], "completed")
        self.assertEqual(
            info["resume_config_sha256"], resume_config_fingerprint(config)
        )
        for field in (
            "engine_env_names",
            "engine_environment",
            "engine_args",
            "argv",
            "resume_config",
            "profile",
        ):
            self.assertNotIn(field, info)
        self.assertNotIn("DSV4_CHUNK_TOKENS", text)
        self.assertEqual(
            info["config_file_sha256"][cli.MANIFEST],
            cli.sha(plan["artifacts"][cli.MANIFEST]),
        )
        resumed = cli.build_plan(self.args("resume", explicit=False), {})
        self.assertEqual(resumed["summary"]["environment"], saved["env"])
        resumed_args = [
            a[len("--test_arg=") :]
            for a in resumed["command"]
            if a.startswith("--test_arg=")
        ]
        new_args, new_remaining = parse_args(resumed_args)
        with patch.dict("os.environ", saved["env"], clear=True):
            new_config = _build_cache_resume_config(
                new_args, new_remaining, list(saved["env"]), 4096
            )
        self.assertEqual(
            resume_config_fingerprint(new_config), resume_config_fingerprint(config)
        )
        raw = json.loads((self.result / cli.MANIFEST).read_text())
        raw["env"]["DSV4_CHUNK_TOKENS"] = "999"
        (self.result / cli.MANIFEST).write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError, "saved configuration changed"):
            cli.load_saved(self.result)
        (self.result / cli.MANIFEST).unlink()
        with self.assertRaisesRegex(ValueError, "compact test_info requires"):
            cli.load_saved(self.result)

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

    def test_fixed_workspace_capacities_are_frozen_and_replayed(self):
        payload = json.loads(self.grid.read_text())
        payload["generator"] = {
            "workspace_policy": "fixed_cp8_1m_v1",
            "parameters": {"max_batch_tokens": 65536},
        }
        for case in payload["cases"]:
            case["batch_size"] = 4
        self.grid.write_text(json.dumps(payload))
        first = self.save_run()
        plans = [first]
        for mode in ("resume", "retest"):
            extra = [] if mode == "resume" else ["--cases", "7"]
            plans.append(
                cli.build_plan(self.args(mode, extra=extra, explicit=False), {})
            )
        for plan in plans:
            for arg in (
                "--max_seq_len=1048576",
                "--max_context_batch_size=1",
                "--max_batch_tokens_size=65536",
                "--concurrency_limit=4",
                "--cache_fixed_workspace",
            ):
                self.assertIn("--test_arg=" + arg, plan["command"])

    def test_fixed_workspace_rejects_oversized_rectangle_before_launch(self):
        self.grid.write_text(
            json.dumps(
                {
                    "generator": {"workspace_policy": "fixed_cp8_1m_v1"},
                    "cases": [
                        {
                            "case_id": 1,
                            "batch_size": 8,
                            "request_groups": [
                                {"count": 1, "input_len": 200000, "cache_len": 0},
                                {"count": 7, "input_len": 8192, "cache_len": 0},
                            ],
                        }
                    ],
                }
            )
        )
        with self.assertRaisesRegex(ValueError, "rectangle"):
            cli.build_plan(self.args(), {})
        self.assertFalse(self.result.exists())

    def test_fixed_workspace_batch_one_profile_keeps_capacity(self):
        payload = json.loads(self.grid.read_text())
        payload["generator"] = {"workspace_policy": "fixed_cp8_1m_v1"}
        self.grid.write_text(json.dumps(payload))
        self.save_run()
        plan = cli.build_plan(
            self.args("profile", extra=["--cases", "7"], explicit=False), {}
        )
        self.assertIn("--test_arg=--cache_fixed_workspace", plan["command"])
        self.assertIn("--test_arg=--max_context_batch_size=1", plan["command"])
        self.assertIn("--test_arg=--max_seq_len=1048576", plan["command"])

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
        self.assertIn("--test_arg=--cache_profile_flat_output", plan["command"])
        self.assertIn("--test_arg=--cache_profile_runs=2", plan["command"])
        self.assertEqual(plan["summary"]["planned_cases"], 1)
        self.assertFalse(plan["summary"]["reads_checkpoint"])

    def test_unknown_case_and_invalid_runs(self):
        for options in (["--cases", "888"], ["--cases", "7", "--runs", "0"]):
            with self.assertRaises(ValueError):
                cli.build_plan(self.args("retest", extra=options), {})

    def test_profile_plan_uses_same_isolated_directory_in_backend(self):
        from rtp_llm.test.perf_test.batch_decode_test import (
            _prepare_cache_profile_result_dir,
            parse_args,
        )

        plan = cli.build_plan(self.args("profile", extra=["--cases", "7"]), {})
        directory = plan["destination"]
        directory.mkdir(parents=True)
        for name, data in plan["artifacts"].items():
            (directory / name).write_bytes(data)
        argv = json.loads(plan["artifacts"][cli.MANIFEST])["runner_args"]
        args, _ = parse_args(argv)
        _prepare_cache_profile_result_dir(args)
        self.assertEqual(Path(args.result_dir), directory)
        self.assertTrue(args.cache_profile_flat_output)
        self.assertTrue(args.cache_profile_only)
        self.assertFalse((directory / "cache_profile_replays").exists())
        for name, data in plan["artifacts"].items():
            self.assertEqual((directory / name).read_bytes(), data)

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

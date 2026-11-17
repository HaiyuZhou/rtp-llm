"""Tests for core entry_test."""

import builtins
import io
import json
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from rtp_llm.test.perf_test.cli import parse_cli
from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.entry import (
    build_request_plan,
    create_preview,
    dispatch_plan,
    main,
    run_request,
)
from rtp_llm.test.perf_test.errors import ConfigurationError, PlanningError
from rtp_llm.test.perf_test.plan import ExecutionPlan, build_execution_plan
from rtp_llm.test.perf_test.run_directory import initialize_run_directory
from rtp_llm.test.perf_test.tests.support.entry import (
    dry_run_cases,
    dry_run_config,
    entry_LOCAL,
    entry_ROOT,
    entry_SCRIPT,
    public_boundary_ROOT,
)
from rtp_llm.test.perf_test.tests.support.fixtures import (
    fixture_engine,
    fixture_execution,
    fixture_model,
    fixture_workload,
    run_context_path,
)


class UnifiedEntryTest(unittest.TestCase):

    def test_incompatible_ep_is_rejected_before_outputs_or_executor_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = json.loads(
                (entry_LOCAL.parent / "decode_grid_local.json").read_text()
            )
            source["model"]["topology"]["ep"] = 1
            source["output"]["result_root"] = str(root / "results")
            path = root / "config.json"
            path.write_text(json.dumps(source))
            factory = Mock()
            errors = io.StringIO()
            code = main(
                ["--mode=decode", "--action=run", f"--config={path}"],
                stderr=errors,
                executor_factory=factory,
            )
            self.assertEqual(2, code)
            self.assertIn("model.topology.ep", errors.getvalue())
            factory.assert_not_called()
            self.assertEqual(["config.json"], sorted(x.name for x in root.iterdir()))

    def test_profile_cli_exit_code_matches_real_profile_outcome(self):
        from rtp_llm.test.perf_test.prefill.profile import PrefillProfiler
        from rtp_llm.test.perf_test.tests.support.fixtures import prefill_profile_setup

        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "run"
                plan = prefill_profile_setup(root)

                def factory(profile_dir):
                    def run_case(case):
                        if fail:
                            raise RuntimeError("trace collection failed")
                        (profile_dir / "trace.json").write_text("{}")
                        return dict(
                            case_id=case["case_id"],
                            status="ok",
                            trace_files=["trace.json"],
                        )

                    return Mock(run_case=run_case)

                def execute(_plan):
                    return PrefillProfiler(factory).run(
                        root, [plan.cases[0].case_id], session_id="cpu-profile"
                    )

                output = io.StringIO()
                code = main(
                    ["--mode=prefill", "--action=profile", "--run-dir=" + str(root)],
                    stdout=output,
                    executor_factory=lambda: {"local": execute},
                )
                self.assertEqual(5 if fail else 0, code)
                self.assertEqual(
                    "failed" if fail else "completed",
                    json.loads(output.getvalue())["status"],
                )

    def test_multi_dp_batch_rejected_before_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = json.loads(entry_LOCAL.read_text())
            source["model"]["topology"].update(dp=2, ep=2)
            source["execution"]["devices"] = [0, 1]
            source["output"]["result_root"] = str(root / "results")
            path = root / "config.json"
            for batch_size in (4, 1):
                source["workload"]["batch_sizes"] = [batch_size]
                path.write_text(json.dumps(source))
                request = parse_cli(
                    ["--mode=prefill", "--action=run", f"--config={path}"]
                )
                if batch_size > 1:
                    with self.assertRaisesRegex(PlanningError, "DP.*batch"):
                        build_request_plan(request)
                else:
                    self.assertTrue(build_request_plan(request)[0].cases)
                self.assertEqual(
                    ["config.json"], sorted((item.name for item in root.iterdir()))
                )

    def _run(self, *arguments, cwd=None):
        return subprocess.run(
            [str(entry_SCRIPT), *arguments],
            cwd=cwd or entry_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_repository_entry_help(self):
        completed = self._run("--help")
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertIn("--mode", completed.stdout)
        self.assertIn("--action", completed.stdout)
        self.assertIn("--skip-test", completed.stdout)
        self.assertNotIn("--partial", completed.stdout)

    def test_local_dry_run_prints_plan_without_creating_result_root(self):
        with tempfile.TemporaryDirectory() as directory:
            completed = self._run(
                "--mode=prefill",
                "--action=run",
                f"--config={entry_LOCAL}",
                "--dry-run",
                cwd=directory,
            )
            self.assertEqual(0, completed.returncode, completed.stderr)
            preview = json.loads(completed.stdout)
            self.assertTrue(preview["dry_run"])
            self.assertEqual("local", preview["backend"])
            self.assertGreater(preview["cases"]["count"], 0)
            self.assertFalse((Path(directory) / "perf_test_results").exists())

    def test_bazel_main_module_uses_the_same_help_contract(self):
        completed = subprocess.run(
            ["python", "-m", "rtp_llm.test.perf_test.main", "--help"],
            cwd=entry_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertIn("--dry-run", completed.stdout)


class EntryExecutionTest(unittest.TestCase):

    def test_run_directory_is_initialized_before_backend_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_root = root / "results"
            config = {
                "schema_version": 1,
                "model": fixture_model(label="model"),
                "engine": fixture_engine(
                    max_seq_len=1024, max_batch_tokens=1024, concurrency_limit=1
                ),
                "workload": fixture_workload(max_input_len=512, cache_lengths=[0]),
                "execution": fixture_execution(),
                "output": {"result_root": str(result_root)},
                "regression": {},
                "publish": {},
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            observed = {}

            def execute(plan):
                run_root = Path(plan.result_directory)
                observed["root"] = run_root
                observed["files"] = {
                    name: (run_root / name).is_file()
                    for name in (
                        "source_config.json",
                        "resolved_config.json",
                        "execution_plan.json",
                        "manifest.json",
                    )
                }
                return Mock(exit_code=0)

            request = parse_cli(
                ["--mode=prefill", "--action=run", f"--config={config_path}"]
            )
            original_import = builtins.__import__

            def entry_import(name, *args, **kwargs):
                if name == "rtp_llm.test.perf_test.prefill.checkpoint":
                    raise ModuleNotFoundError(name)
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=entry_import):
                run_request(request, executor_factory=lambda: {"local": execute})
            self.assertTrue(all(observed["files"].values()))
            source = json.loads(
                (observed["root"] / "source_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(config, source)


class DryRunTest(unittest.TestCase):

    def _assert_side_effect_free_preview(self, backend):
        create_run_directory = Mock()
        build_target = Mock()
        start_engine = Mock()
        submit_job = Mock()
        publisher = Mock()

        def execute(_plan):
            create_run_directory()
            build_target()
            start_engine()
            submit_job()
            publisher()

        with tempfile.TemporaryDirectory() as temp_dir:
            result_directory = Path(temp_dir) / "formal-run"
            plan = build_execution_plan(
                mode="prefill",
                action="run",
                resolved=resolve_config(dry_run_config(backend)),
                cases=dry_run_cases(),
                result_directory=str(result_directory),
            )
            preview = dispatch_plan(plan, dry_run=True, execute=execute)
            self.assertFalse(result_directory.exists())
            self.assertTrue(preview["dry_run"])
            self.assertEqual(backend, preview["backend"])
            self.assertEqual(
                plan.to_dict()["resolved_config"], preview["normalized_config"]
            )
            self.assertEqual(2, preview["cases"]["count"])
            self.assertEqual(
                [case.case_id for case in plan.cases], preview["cases"]["case_ids"]
            )
            self.assertEqual(plan.resource_plan.to_dict(), preview["resource_summary"])
        create_run_directory.assert_not_called()
        build_target.assert_not_called()
        start_engine.assert_not_called()
        submit_job.assert_not_called()
        publisher.assert_not_called()
        return preview

    def test_local_dry_run_is_complete_and_side_effect_free(self):
        preview = self._assert_side_effect_free_preview("local")
        self.assertEqual(1, len(preview["commands"]))
        self.assertEqual("local-run", preview["commands"][0]["stage"])
        self.assertIn("execution_plan.json", preview["commands"][0]["argv"][-1])


class PublicBoundaryTest(unittest.TestCase):

    def test_single_machine_aone_decode_rejected_before_side_effects(self):
        for mode in ("decode",):
            for dry in (False, True):
                with self.subTest(
                    mode=mode, dry=dry
                ), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    source = json.loads(
                        (
                            public_boundary_ROOT
                            / "examples"
                            / (
                                "prefill_local.json"
                                if mode == "prefill"
                                else "decode_grid_local.json"
                            )
                        ).read_text()
                    )
                    source["execution"] = {
                        "backend": "aone",
                        "resource": "daily",
                        "gpu_type": "H20",
                        "max_parallel": 1,
                        "shard_count": 1,
                    }
                    source["output"]["result_root"] = str(root / "results")
                    config = root / "config.json"
                    config.write_text(json.dumps(source))
                    args = [f"--mode={mode}", "--action=run", f"--config={config}"] + (
                        ["--dry-run"] if dry else []
                    )
                    executor_factory = Mock()
                    self.assertEqual(3, main(args, executor_factory=executor_factory))
                    executor_factory.assert_not_called()
                    self.assertEqual(
                        ["config.json"], sorted((x.name for x in root.iterdir()))
                    )

    def test_aone_dry_run_plans_single_prefill_and_gang_without_runtime_imports(self):
        for mode, gang in (("prefill", False), ("prefill", True), ("decode", True)):
            with self.subTest(
                mode=mode, gang=gang
            ), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                example = (
                    "prefill_local.json"
                    if mode == "prefill"
                    else "decode_grid_local.json"
                )
                source = json.loads(
                    (public_boundary_ROOT / "examples" / example).read_text()
                )
                source["model"]["topology"].update(
                    tp=16 if gang else 2, dp=1, pp=1, ep=1
                )
                source["engine"]["concurrency_limit"] = 8
                source["execution"] = {
                    "backend": "aone",
                    "resource": "daily",
                    "gpu_type": "H20",
                    "max_parallel": 2 if gang else 1,
                    **({"gpus_per_machine": 8} if gang else {"shard_count": 2}),
                }
                source["output"]["result_root"] = str(root / "results")
                config = root / "config.json"
                config.write_text(json.dumps(source))
                request = parse_cli(
                    [
                        f"--mode={mode}",
                        "--action=run",
                        f"--config={config}",
                        "--dry-run",
                    ]
                )
                original_import = builtins.__import__

                def forbid_runtime(name, *args, **kwargs):
                    if name.startswith("internal_source") or name.endswith(
                        ("execution.local", "execution.aone_bazel")
                    ):
                        raise AssertionError(f"dry-run imported runtime: {name}")
                    return original_import(name, *args, **kwargs)

                with patch("builtins.__import__", side_effect=forbid_runtime):
                    preview = run_request(
                        request,
                        executor_factory=lambda: self.fail("dry-run created executor"),
                    )
                self.assertEqual("aone", preview["backend"])
                self.assertEqual(
                    2 if gang else 1, preview["resource_summary"]["machine_count"]
                )
                self.assertEqual(
                    ["build", "remote-test"],
                    [command["stage"] for command in preview["commands"]],
                )
                test_argv = preview["commands"][1]["argv"]
                if gang:
                    self.assertIn("//generated/perf_test:gang", test_argv)
                    self.assertEqual(
                        [8, 9, 10, 11, 12, 13, 14, 15],
                        preview["gang"]["members"][1]["world_ranks"],
                    )
                else:
                    self.assertIn("//generated/perf_test:shard_0000", test_argv)
                    self.assertIn("//generated/perf_test:shard_0001", test_argv)
                    owned = [
                        case_id
                        for shard in preview["shards"]["items"]
                        for case_id in shard["case_ids"]
                    ]
                    self.assertCountEqual(preview["cases"]["case_ids"], owned)
                    self.assertEqual(len(owned), len(set(owned)))
                self.assertEqual(
                    ["config.json"], sorted(p.name for p in root.iterdir())
                )

    def test_aone_run_dispatches_injected_executor_after_persisting_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = json.loads(entry_LOCAL.read_text())
            source["execution"] = {
                "backend": "aone",
                "resource": "daily",
                "gpu_type": "H20",
                "max_parallel": 1,
                "shard_count": 1,
            }
            source["output"]["result_root"] = str(root / "results")
            config = root / "config.json"
            config.write_text(json.dumps(source))

            def execute(plan):
                plan_path = Path(plan.result_directory) / "execution_plan.json"
                restored = ExecutionPlan.from_dict(json.loads(plan_path.read_text()))
                self.assertEqual(plan, restored)
                return {"backend": plan.backend, "cases": len(plan.cases)}

            result = run_request(
                parse_cli(["--mode=prefill", "--action=run", f"--config={config}"]),
                executor_factory=lambda: {"aone": execute},
            )
            self.assertEqual("aone", result["backend"])
            self.assertGreater(result["cases"], 0)

    def test_invalid_aone_case_gang_combinations_rejected_before_side_effects(self):
        invalid = (
            {"gpus_per_machine": 1},
            {"shard_count": 100000},
            {"gpu_type": "unknown"},
        )
        for changes in invalid:
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = json.loads(entry_LOCAL.read_text())
                source["model"]["topology"].update(tp=2, dp=1, pp=1, ep=1)
                source["execution"] = {
                    "backend": "aone",
                    "resource": "daily",
                    "gpu_type": "H20",
                    "max_parallel": 2,
                    "shard_count": 2,
                    **changes,
                }
                source["output"]["result_root"] = str(root / "results")
                config = root / "config.json"
                config.write_text(json.dumps(source))
                with self.assertRaises(PlanningError):
                    run_request(
                        parse_cli(
                            ["--mode=prefill", "--action=run", f"--config={config}"]
                        ),
                        executor_factory=lambda: self.fail(
                            "invalid request created executor"
                        ),
                    )
                self.assertEqual(
                    ["config.json"], sorted(p.name for p in root.iterdir())
                )

    def test_aone_report_is_offline_and_follow_up_execution_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = json.loads(entry_LOCAL.read_text())
            source["execution"] = {
                "backend": "aone",
                "resource": "daily",
                "gpu_type": "H20",
                "max_parallel": 1,
                "shard_count": 1,
            }
            source["output"]["result_root"] = str(root / "results")
            config = root / "config.json"
            config.write_text(json.dumps(source))
            plan, source = build_request_plan(
                parse_cli(["--mode=prefill", "--action=run", f"--config={config}"])
            )
            initialize_run_directory(
                plan.result_directory,
                source_config=source,
                plan=plan,
                run_id=Path(plan.result_directory).name,
                code_version="test",
            )
            report = run_request(
                parse_cli(
                    [
                        "--mode=prefill",
                        "--action=report",
                        f"--run-dir={plan.result_directory}",
                        "--dry-run",
                    ]
                ),
                executor_factory=lambda: self.fail("offline preview created executor"),
            )
            self.assertEqual([], report["commands"])
            for action in ("resume", "profile"):
                with self.subTest(action=action), self.assertRaises(ConfigurationError):
                    run_request(
                        parse_cli(
                            [
                                "--mode=prefill",
                                f"--action={action}",
                                f"--run-dir={plan.result_directory}",
                            ]
                        ),
                        executor_factory=lambda: self.fail(
                            "unsupported action created executor"
                        ),
                    )

    def test_historical_plan_is_rejected(self):
        plan, _ = build_request_plan(
            parse_cli(
                [
                    "--config="
                    + str(public_boundary_ROOT / "examples/prefill_local.json"),
                    "--dry-run",
                ]
            )
        )
        payload = plan.to_dict()
        payload["schema_version"] = 1
        with self.assertRaisesRegex(PlanningError, "schema_version"):
            ExecutionPlan.from_dict(payload)

    def test_decode_follow_up_actions_rejected_before_output_or_executor(self):
        for action in ("resume", "profile", "report"):
            with self.subTest(action=action), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                run_directory = root / "run"
                (plan, source) = build_request_plan(
                    parse_cli(
                        [
                            "--mode=decode",
                            "--action=run",
                            f"--config={public_boundary_ROOT / 'examples/decode_grid_local.json'}",
                        ]
                    )
                )
                plan = replace(plan, result_directory=str(run_directory))
                initialize_run_directory(
                    run_directory,
                    source_config=source,
                    plan=plan,
                    run_id="run",
                    code_version="test",
                )
                before = {
                    str(path.relative_to(root)): path.read_bytes()
                    for path in root.rglob("*")
                    if path.is_file()
                }
                executor_factory = Mock()
                self.assertEqual(
                    2,
                    main(
                        [
                            "--mode=decode",
                            f"--action={action}",
                            f"--run-dir={run_directory}",
                        ],
                        executor_factory=executor_factory,
                    ),
                )
                executor_factory.assert_not_called()
                self.assertEqual(
                    before,
                    {
                        str(path.relative_to(root)): path.read_bytes()
                        for path in root.rglob("*")
                        if path.is_file()
                    },
                )


class CliParsingTest(unittest.TestCase):

    def test_skip_test_alias_selects_offline_prefill_report(self):
        request = parse_cli(["--mode=prefill", "--skip-test", "--run-dir=runs/example"])
        self.assertEqual("report", request.action)
        self.assertEqual("runs/example", request.run_dir)
        explicit = parse_cli(
            [
                "--mode=prefill",
                "--action=report",
                "--skip-test",
                "--run-dir=runs/example",
            ]
        )
        self.assertEqual(request, explicit)

    def test_action_source_conflicts_are_exit_code_two(self):
        invalid = [
            ["--mode", "prefill", "--action", "run"],
            [
                "--mode",
                "prefill",
                "--action",
                "run",
                "--config",
                "config.json",
                "--run-dir",
                "runs/example",
            ],
        ]
        for argv in invalid:
            with self.subTest(argv=argv):
                with self.assertRaises(ConfigurationError) as context:
                    parse_cli(argv)
                self.assertEqual(2, context.exception.exit_code)


class RepositoryProvenanceTest(unittest.TestCase):
    def test_invalid_revision_mapping_is_rejected_before_run_creation(self):
        for value in ('{"repo": "main"}', "[]", '"not a mapping"', "{invalid"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = json.loads(entry_LOCAL.read_text())
                source["output"]["result_root"] = str(root / "results")
                path = root / "source.json"
                path.write_text(json.dumps(source))
                request = parse_cli(
                    ["--mode=prefill", "--action=run", f"--config={path}"]
                )
                context_path = root / "run_context.json"
                context_path.write_text(
                    '{"schema_version":1,"repository_revisions":' + value + "}"
                )
                request = replace(request, run_context=str(context_path))
                with self.assertRaises((PlanningError, ValueError)):
                    run_request(request)
                self.assertFalse((root / "results").exists())

    def test_revisions_survive_success_and_executor_failure(self):
        revisions = {"public": "a" * 40, "internal": "b" * 40}
        for failure in (False, True):
            with self.subTest(
                failure=failure
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = json.loads(entry_LOCAL.read_text())
                source["output"]["result_root"] = str(root / "results")
                path = root / "source.json"
                path.write_text(json.dumps(source))
                request = parse_cli(
                    ["--mode=prefill", "--action=run", f"--config={path}"]
                )

                def execute(plan):
                    if failure:
                        raise OSError("execution failed")
                    return 0

                request = replace(
                    request,
                    run_context=run_context_path(root, repository_revisions=revisions),
                )
                with patch.dict(
                    "os.environ",
                    {
                        "BUILD_SCM_REVISION": revisions["public"],
                    },
                ):
                    if failure:
                        with self.assertRaisesRegex(OSError, "execution failed"):
                            run_request(
                                request, executor_factory=lambda: {"local": execute}
                            )
                    else:
                        run_request(
                            request, executor_factory=lambda: {"local": execute}
                        )
                manifest = json.loads(
                    next((root / "results").glob("*/manifest.json")).read_text()
                )
                self.assertEqual(revisions, manifest["repository_revisions"])
                self.assertEqual(revisions["public"], manifest["code_version"])


if __name__ == "__main__":
    unittest.main()


class InferredCliTest(unittest.TestCase):

    def test_invalid_workload_shape_is_a_configuration_error(self):
        from rtp_llm.test.perf_test.entry import plan_from_source

        for workload in (None, [], "grid"):
            with self.subTest(workload=workload), self.assertRaises(ConfigurationError):
                plan_from_source({"workload": workload}, source_dir=Path.cwd())

    def test_single_config_defaults_to_run_and_infers_mode(self):
        request = parse_cli(["--config=" + str(entry_LOCAL), "--dry-run"])
        self.assertEqual("run", request.action)
        self.assertIsNone(request.mode)
        plan, _ = build_request_plan(request)
        self.assertEqual("prefill", plan.mode)

    def test_explicit_mode_is_assertion(self):
        request = parse_cli(["--mode=decode", "--config=" + str(entry_LOCAL)])
        with self.assertRaises(ConfigurationError):
            build_request_plan(request)

    def test_maintenance_only_accepts_existing_run(self):
        request = parse_cli(["--run-dir=/tmp/existing-run", "--maintenance=status"])
        self.assertEqual("status", request.maintenance)
        for argv in (
            ["--maintenance=cancel"],
            ["--run-dir=x", "--maintenance=clean", "--action=run"],
        ):
            with self.assertRaises(ConfigurationError):
                parse_cli(argv)


class SshEntryContractTest(unittest.TestCase):
    def test_one_config_freezes_git_and_roundtrips_without_original_worktree(self):
        import shutil

        from rtp_llm.test.perf_test.entry import plan_from_source
        from rtp_llm.test.perf_test.run_directory import inspect_run_directory

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "public source"
            repo.mkdir()
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            (repo / "source.py").write_text("print('source')\n")
            subprocess.run(["git", "-C", str(repo), "add", "source.py"], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "-c",
                    "user.name=Perf Test",
                    "-c",
                    "user.email=perf@example.invalid",
                    "commit",
                    "-qm",
                    "fixture",
                ],
                check=True,
            )
            source = json.loads(entry_LOCAL.read_text())
            source["execution"] = {
                "backend": "ssh",
                "hosts": [
                    {"ssh_target": "gpu-via-jump", "advertise_host": "192.0.2.11"}
                ],
                "gpus_per_machine": 1,
                "source": {"build_policy": "reuse", "worktree": repo.name},
            }
            plan, frozen_source = plan_from_source(
                source, source_dir=root, result_directory=str(root / "run")
            )
            self.assertEqual(
                40, len(plan.resolved_config["execution"]["source"]["ref"])
            )
            self.assertEqual(1, plan.resource_plan.machine_count)
            self.assertEqual(plan, ExecutionPlan.from_dict(plan.to_dict()))
            initialize_run_directory(
                root / "run",
                source_config=frozen_source,
                plan=plan,
                run_id="run",
                code_version="test",
            )
            shutil.rmtree(repo)
            self.assertEqual(plan, inspect_run_directory(root / "run").execution_plan)
            with self.assertRaises(TypeError):
                plan.resource_plan.hosts[0]["ssh_target"] = "changed"

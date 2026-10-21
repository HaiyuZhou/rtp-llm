import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from rtp_llm.test.perf_test.cache_grid.runner import cache_perf as cli
from rtp_llm.test.perf_test.cache_grid.runner import (
    cache_perf_orchestrator as orchestrator,
)


class CachePerfOrchestratorTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.repo = Path(temporary.name) / "workspace"
        self.repo.mkdir()

    def local_args(self, **updates):
        values = dict(
            mode="local-run",
            model_json='{"type":"qwen_2","checkpoint":"/models/qwen"}',
            engine_json='{"max_seq_len":4096,"seq_size_per_block":64,'
            '"args":{"act_type":"BF16"}}',
            grid_json='{"cache_alignment":64,"compute_step":[1024,256]}',
            engine_env_json="{}",
            runtime_env_json="{}",
            result_root=None,
            devices="0",
            bazel="bazelisk",
            bazel_arg=[],
            dry_run=False,
        )
        values.update(updates)
        return argparse.Namespace(**values)

    def aone_args(self, **updates):
        values = dict(
            mode="aone-run",
            resource="daily",
            gpu_type="SM103_CU13",
            build_jobs=None,
            max_parallel=1,
        )
        values.update(updates)
        return self.local_args(**values)

    def test_aone_plan_expands_resource_and_separates_concurrency(self):
        args = self.aone_args(build_jobs=64, max_parallel=2)
        plan = orchestrator.build_aone_plan(
            args, repo=self.repo, run_id="unit", inherited={}
        )
        common = [
            "--config=cuda13",
            "--config=cicd",
            "--config=daily_aone_bazel_cache",
            "--config=daily_aone_bazel_remote",
        ]
        self.assertEqual(
            plan.commands["build"],
            [
                "bazelisk",
                "build",
                plan.ci_plan["suite_target"],
                *common,
                "--jobs=64",
                "--remote_download_outputs=minimal",
                "--keep_going",
            ],
        )
        self.assertEqual(
            plan.commands["test"],
            [
                "bazelisk",
                "test",
                plan.ci_plan["suite_target"],
                *common,
                "--strategy=TestRunner=remote",
                "--jobs=2",
                "--test_timeout=7000",
                "--test_output=errors",
                "--keep_going",
                "--remote_download_outputs=all",
                "--nocache_test_results",
                "--run_under=//rtp_llm/test/utils:gpu_lock",
            ],
        )
        self.assertEqual(plan.ci_plan["gpu_count"], 1)
        self.assertEqual(plan.mode, "aone-run")
        self.assertEqual(plan.summary["effective_case_count"], len(plan.request_cases))

    def test_aone_plan_maps_supported_pools_and_online_resource(self):
        for gpu_type, platform in (
            ("SM100_ARM_CU13", "cuda13_arm"),
            ("SM103_CU13", "cuda13"),
            ("L20_CU13", "cuda13"),
            ("L20D_DEV", "cuda13"),
        ):
            with self.subTest(gpu_type=gpu_type):
                plan = orchestrator.build_aone_plan(
                    self.aone_args(resource="online", gpu_type=gpu_type),
                    repo=self.repo,
                    run_id="pool",
                    inherited={},
                )
                expected = [
                    f"--config={platform}",
                    "--config=cicd",
                    "--config=online_aone_bazel_cache",
                    "--config=online_aone_bazel_remote",
                ]
                for command in plan.commands.values():
                    self.assertEqual(command[3:7], expected)
                profile = json.loads(
                    plan.artifacts[plan.paths.inputs_dir / "profile.json"]
                )
                self.assertEqual(
                    profile["bazel"]["configs"],
                    [
                        platform,
                        "cicd",
                        "online_aone_bazel_cache",
                        "online_aone_bazel_remote",
                    ],
                )
                self.assertEqual(plan.ci_plan["gpu_type"], gpu_type)

    def test_aone_plan_rejects_unknown_gpu_pool_before_artifacts(self):
        with patch.object(orchestrator.ci, "build_input_artifacts") as inputs:
            with self.assertRaisesRegex(ValueError, "supported GPU types"):
                orchestrator.build_aone_plan(
                    self.aone_args(gpu_type="UNKNOWN_CU13"),
                    repo=self.repo,
                    run_id="bad",
                )
        inputs.assert_not_called()
        self.assertEqual(list(self.repo.iterdir()), [])

    def test_aone_plan_rejects_unknown_resource(self):
        with self.assertRaisesRegex(ValueError, "supported resources"):
            orchestrator.build_aone_plan(
                self.aone_args(resource="unknown"), repo=self.repo, run_id="bad"
            )
        self.assertEqual(list(self.repo.iterdir()), [])

    def test_aone_plan_defaults_build_jobs_to_available_cpus(self):
        for cpu_count, expected in (
            (8, "--jobs=8"),
            (None, "--jobs=1"),
            (0, "--jobs=1"),
        ):
            with self.subTest(cpu_count=cpu_count), patch.object(
                orchestrator.os, "cpu_count", return_value=cpu_count
            ):
                plan = orchestrator.build_aone_plan(
                    self.aone_args(), repo=self.repo, run_id="cpus", inherited={}
                )
                self.assertIn(expected, plan.commands["build"])
                self.assertIn("--jobs=1", plan.commands["test"])

    def test_aone_plan_derives_gpu_count_from_topology_and_keeps_env_ownership(self):
        inherited = {"PATH": "/bin", "CC": "old-gcc", "API_TOKEN": "inherited-only"}
        args = self.aone_args(
            model_json='{"type":"qwen_2","checkpoint":"/models/qwen",'
            '"topology":{"tp":2,"dp":2,"pp":2,"ep":4}}',
            engine_env_json='{"MODEL_SETTING":"00123"}',
            runtime_env_json='{"CC":"/usr/bin/gcc"}',
        )
        plan = orchestrator.build_aone_plan(
            args, repo=self.repo, run_id="env", inherited=inherited
        )
        self.assertEqual(plan.ci_plan["gpu_count"], 8)
        self.assertEqual(
            plan.process_env,
            {
                "PATH": "/bin",
                "CC": "/usr/bin/gcc",
                "API_TOKEN": "inherited-only",
            },
        )
        self.assertEqual(inherited["CC"], "old-gcc")
        build_file = plan.artifacts[plan.paths.generated_dir / "BUILD.bazel"].decode()
        self.assertIn("gpu_count = 8", build_file)
        self.assertIn('"WORLD_SIZE": "8"', build_file)
        self.assertIn('"MODEL_SETTING": "00123"', build_file)
        self.assertIn('"CC": "/usr/bin/gcc"', build_file)
        self.assertNotIn("inherited-only", b"".join(plan.artifacts.values()).decode())

    def test_aone_plan_is_write_free_and_assigns_each_effective_case_once(self):
        args = self.aone_args(bazel_arg=["--jobs=7", "--remote_timeout=45m"])
        with patch.object(Path, "mkdir") as mkdir, patch.object(
            Path, "open", side_effect=AssertionError("planning must not open files")
        ), patch.object(Path, "write_bytes") as write_bytes, patch.object(
            Path, "write_text"
        ) as write_text, patch.object(
            orchestrator.subprocess, "run"
        ) as run:
            plan = orchestrator.build_aone_plan(
                args, repo=self.repo, run_id="dry", inherited={}
            )
        mkdir.assert_not_called()
        write_bytes.assert_not_called()
        write_text.assert_not_called()
        run.assert_not_called()
        assigned = [
            case_id for shard in plan.ci_plan["shards"] for case_id in shard["case_ids"]
        ]
        self.assertEqual(sorted(assigned), plan.ci_plan["case_ids"])
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertEqual(
            sorted(case["case_id"] for case in plan.request_cases), sorted(assigned)
        )
        self.assertEqual(plan.commands["build"][-2:], args.bazel_arg)
        self.assertEqual(plan.commands["test"][-2:], args.bazel_arg)
        self.assertEqual(list(self.repo.iterdir()), [])
        self.assertEqual(
            {path.parent for path in plan.artifacts},
            {
                plan.paths.inputs_dir,
                plan.paths.generated_dir,
            },
        )
        self.assertEqual(
            plan.ci_plan["suite_target"], "//cache_grid_aone_generated_qwen_dry:all"
        )
        self.assertEqual(
            plan.ci_plan["profile_source"], str(plan.paths.inputs_dir / "profile.json")
        )
        self.assertEqual(
            plan.ci_plan["grid_source"], str(plan.paths.inputs_dir / "grid.json")
        )
        for shard in plan.ci_plan["shards"]:
            payload = json.loads(
                plan.artifacts[plan.paths.generated_dir / shard["grid_file"]]
            )
            self.assertEqual(
                [case["case_id"] for case in payload["cases"]], shard["case_ids"]
            )

    def test_aone_plan_automatically_shards_large_grid_independent_of_parallelism(self):
        args = self.aone_args(
            engine_json='{"max_seq_len":1302,"seq_size_per_block":1,"args":{"act_type":"BF16"}}',
            grid_json='{"cache_alignment":1,"compute_step":[1],"fixed_cache_len":0,"max_cases":1400}',
            max_parallel=1,
        )
        plan = orchestrator.build_aone_plan(
            args, repo=self.repo, run_id="large", inherited={}
        )
        self.assertEqual(len(plan.request_cases), 1301)
        self.assertEqual(plan.ci_plan["shard_count"], 2)
        self.assertIn("--jobs=1", plan.commands["test"])
        assigned = [
            case_id for shard in plan.ci_plan["shards"] for case_id in shard["case_ids"]
        ]
        self.assertEqual(sorted(assigned), list(range(1301)))

    def test_local_plan_uses_one_bazel_run_without_test_timeout(self):
        plan = orchestrator.build_local_plan(
            self.local_args(),
            repo=self.repo,
            run_id="unit",
            machine="x86_64",
            inherited={},
        )
        command = plan.commands["run"]
        self.assertEqual(command[:3], ["bazelisk", "run", cli.TARGET])
        self.assertEqual(set(plan.commands), {"run"})
        self.assertIn("--config=cuda13", command)
        self.assertIn("--run_under=//rtp_llm/test/utils:gpu_lock", command)
        self.assertNotIn("--test_timeout", " ".join(command))
        self.assertNotIn("shard", " ".join(command))
        self.assertEqual(plan.summary["effective_case_count"], len(plan.request_cases))
        self.assertGreater(len(plan.request_cases), 0)
        self.assertEqual(plan.mode, "local-run")
        self.assertIsNone(plan.ci_plan)
        self.assertFalse(plan.paths.result_root.exists())

    def test_local_plan_rejects_device_count_and_duplicates(self):
        for devices, message in (
            ("0,1", "requires 1 devices"),
            ("0,0", "must be unique"),
        ):
            with self.subTest(devices=devices), self.assertRaisesRegex(
                ValueError, message
            ):
                orchestrator.build_local_plan(
                    self.local_args(devices=devices), repo=self.repo
                )
        self.assertEqual(list(self.repo.iterdir()), [])

    def test_local_plan_rejects_invalid_device_syntax_before_side_effects(self):
        for devices in ("", "0,", ",0", "gpu0", "-1", "0,,1"):
            with self.subTest(devices=devices), self.assertRaisesRegex(
                ValueError, "GPU indices"
            ):
                orchestrator.build_local_plan(
                    self.local_args(devices=devices), repo=self.repo
                )
        self.assertEqual(list(self.repo.iterdir()), [])

    def test_local_plan_derives_device_count_from_tp_dp_pp(self):
        args = self.local_args(
            devices=" 2, 5 ",
            model_json='{"type":"qwen_2","checkpoint":"/models/qwen",'
            '"topology":{"tp":2,"ep":2}}',
        )
        plan = orchestrator.build_local_plan(
            args, repo=self.repo, run_id="tp", machine="x86_64", inherited={}
        )
        self.assertEqual(plan.resolved.gpu_count, 2)
        self.assertEqual(plan.process_env["CUDA_VISIBLE_DEVICES"], "2,5")

    def test_local_plan_selects_arm_config_and_rejects_other_architectures(self):
        plan = orchestrator.build_local_plan(
            self.local_args(),
            repo=self.repo,
            run_id="arm",
            machine="aarch64",
            inherited={},
        )
        self.assertIn("--config=cuda13_arm", plan.commands["run"])
        with self.assertRaisesRegex(ValueError, "unsupported local architecture"):
            orchestrator.build_local_plan(
                self.local_args(), repo=self.repo, machine="ppc64le"
            )
        self.assertEqual(list(self.repo.iterdir()), [])

    def test_local_plan_rejects_explicit_device_environment_before_paths_or_artifacts(
        self,
    ):
        result_root = self.repo.parent / "external-results"
        for category in ("engine_env", "runtime_env"):
            for value in ("0", "7"):
                with self.subTest(category=category, value=value):
                    args = self.local_args(
                        result_root=result_root,
                        **{
                            f"{category}_json": json.dumps(
                                {"CUDA_VISIBLE_DEVICES": value}
                            )
                        },
                    )
                    with patch.object(
                        orchestrator,
                        "make_run_paths",
                        wraps=orchestrator.make_run_paths,
                    ) as paths, patch.object(
                        orchestrator.ci,
                        "build_input_artifacts",
                        wraps=orchestrator.ci.build_input_artifacts,
                    ) as artifacts:
                        with self.assertRaisesRegex(
                            ValueError,
                            rf"{category}\.CUDA_VISIBLE_DEVICES.*managed by --devices",
                        ):
                            orchestrator.build_local_plan(
                                args,
                                repo=self.repo,
                                run_id="device-env",
                                machine="x86_64",
                                inherited={},
                            )
                    paths.assert_not_called()
                    artifacts.assert_not_called()
                    self.assertFalse(result_root.exists())
                    self.assertEqual(list(self.repo.iterdir()), [])

    def test_local_plan_freezes_categorized_environment_and_launch_manifest(self):
        inherited = {
            "PATH": "/bin",
            "CC": "old-gcc",
            "CUDA_VISIBLE_DEVICES": "7",
            "API_TOKEN": "inherited-only",
        }
        args = self.local_args(
            engine_env_json='{"MODEL_SETTING":"00123"}',
            runtime_env_json='{"CC":"/usr/bin/gcc"}',
        )
        plan = orchestrator.build_local_plan(
            args, repo=self.repo, run_id="env", machine="x86_64", inherited=inherited
        )
        self.assertEqual(plan.process_env["CC"], "/usr/bin/gcc")
        self.assertEqual(plan.process_env["MODEL_SETTING"], "00123")
        self.assertEqual(plan.process_env["API_TOKEN"], "inherited-only")
        self.assertEqual(plan.process_env["PATH"], "/bin")
        self.assertEqual(plan.process_env["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(inherited["CC"], "old-gcc")
        command = plan.commands["run"]
        self.assertIn("--engine_env=CC=/usr/bin/gcc", command)
        self.assertIn("--engine_env=MODEL_SETTING=00123", command)
        self.assertNotIn("inherited-only", " ".join(command))
        root = plan.paths.result_root
        profile = json.loads(plan.artifacts[root / "profile.snapshot.json"])
        self.assertEqual(profile["runtime_env"], {"CC": "/usr/bin/gcc"})
        self.assertEqual(profile["engine_env"]["MODEL_SETTING"], "00123")
        self.assertNotIn("CC", profile["engine_env"])
        self.assertNotIn("MODEL_SETTING", profile["runtime_env"])
        launch = json.loads(plan.artifacts[root / cli.MANIFEST])
        self.assertEqual(launch["schema_version"], 2)
        self.assertEqual(launch["mode"], "run")
        self.assertEqual(launch["profile_file"], "profile.snapshot.json")
        self.assertEqual(launch["grid"], str(root / "grid.snapshot.json"))
        self.assertEqual(launch["devices"], ["0"])
        self.assertEqual(
            launch["bazel"],
            {
                "command": "run",
                "configs": ["cuda13"],
                "args": [],
                "executable": "bazelisk",
            },
        )
        self.assertFalse(
            any(value.startswith("--engine_env=") for value in launch["runner_args"])
        )
        self.assertEqual(launch["env"]["CC"], "/usr/bin/gcc")
        self.assertEqual(launch["engine_env"], profile["engine_env"])
        self.assertEqual(launch["runtime_env"], profile["runtime_env"])
        self.assertNotIn("inherited-only", b"".join(plan.artifacts.values()).decode())
        for name in ("profile.snapshot.json", "grid.snapshot.json"):
            self.assertEqual(
                launch["snapshots"][name],
                hashlib.sha256(plan.artifacts[root / name]).hexdigest(),
            )
        grid = json.loads(plan.artifacts[root / "grid.snapshot.json"])
        self.assertEqual(plan.request_cases, grid["cases"])

    def test_local_runner_args_and_bazel_overrides_are_forwarded_in_order(self):
        args = self.local_args(
            bazel="/opt/bazelisk", bazel_arg=["--jobs=2", "--remote_timeout=45m"]
        )
        plan = orchestrator.build_local_plan(
            args, repo=self.repo, run_id="argv", machine="x86_64", inherited={}
        )
        command = plan.commands["run"]
        separator = command.index("--")
        self.assertEqual(command[0], "/opt/bazelisk")
        self.assertEqual(command[separator - 2 : separator], args.bazel_arg)
        runner_args = command[separator + 1 :]
        for value in (
            "--partial=2",
            "--decode_test_length=1",
            "--batch_size=1",
            "--cache_request_transport=dashsc_input_ids",
            "--cache_measure_runs=3",
            "--cache_commit_tail_tokens=64",
        ):
            self.assertIn(value, runner_args)
        self.assertIn(
            f"--profile={plan.paths.result_root / 'profile.snapshot.json'}", runner_args
        )
        self.assertIn(
            f"--cache_grid_json={plan.paths.result_root / 'grid.snapshot.json'}",
            runner_args,
        )
        self.assertIn(f"--result_dir={plan.paths.result_root}", runner_args)

    def test_local_paths_use_sanitized_label_and_no_extra_directories(self):
        paths = orchestrator.make_run_paths(
            self.local_args(),
            repo=self.repo,
            mode="local",
            run_id="unit",
            model_label="Qwen /中文.2",
        )
        self.assertEqual(
            paths.result_root, self.repo / "cache_grid_local_results_Qwen____.2_unit"
        )
        self.assertEqual(paths.inputs_dir, paths.result_root)
        self.assertIsNone(paths.generated_dir)
        self.assertEqual(list(self.repo.iterdir()), [])

    def test_aone_paths_reserve_top_level_input_and_generated_names(self):
        paths = orchestrator.make_run_paths(
            self.local_args(),
            repo=self.repo,
            mode="aone",
            run_id="unit",
            model_label="qwen",
        )
        self.assertEqual(
            paths.result_root, self.repo / "cache_grid_aone_results_qwen_unit"
        )
        self.assertEqual(
            paths.inputs_dir, self.repo / "cache_grid_aone_inputs_qwen_unit"
        )
        self.assertEqual(
            paths.generated_dir, self.repo / "cache_grid_aone_generated_qwen_unit"
        )
        self.assertEqual(list(self.repo.iterdir()), [])

    def test_explicit_result_root_can_be_outside_workspace_but_must_be_empty(self):
        root = self.repo.parent / "external-results"
        args = self.local_args(result_root=root)
        plan = orchestrator.build_local_plan(
            args, repo=self.repo, run_id="external", machine="x86_64", inherited={}
        )
        self.assertEqual(plan.paths.result_root, root)
        self.assertFalse(root.exists())
        root.mkdir()
        orchestrator.build_local_plan(
            args, repo=self.repo, run_id="empty", machine="x86_64", inherited={}
        )
        (root / "existing").write_text("keep")
        with self.assertRaisesRegex(ValueError, "new or empty"):
            orchestrator.build_local_plan(args, repo=self.repo)
        self.assertEqual((root / "existing").read_text(), "keep")

    def test_result_root_file_is_rejected(self):
        root = self.repo / "file"
        root.write_text("keep")
        with self.assertRaisesRegex(ValueError, "new or empty"):
            orchestrator.build_local_plan(
                self.local_args(result_root=root), repo=self.repo
            )

    def test_planning_does_not_write_or_launch_and_resolves_request_once(self):
        with patch.object(
            orchestrator, "resolve_run_request", wraps=orchestrator.resolve_run_request
        ) as resolve, patch.object(Path, "mkdir") as mkdir, patch.object(
            orchestrator.subprocess, "run"
        ) as run:
            orchestrator.build_local_plan(
                self.local_args(),
                repo=self.repo,
                run_id="dry",
                machine="x86_64",
                inherited={},
            )
        self.assertEqual(resolve.call_count, 1)
        mkdir.assert_not_called()
        run.assert_not_called()

    @patch.object(orchestrator.subprocess, "run")
    @patch.object(orchestrator.unified_pipeline, "run")
    def test_execute_local_writes_snapshots_before_launch_and_postprocesses(
        self, pipeline, run
    ):
        plan = orchestrator.build_local_plan(
            self.local_args(),
            repo=self.repo,
            run_id="ok",
            machine="x86_64",
            inherited={},
        )

        def launch(command, **kwargs):
            self.assertEqual(command, plan.commands["run"])
            self.assertEqual(
                kwargs,
                {"cwd": orchestrator.REPO, "env": plan.process_env, "check": False},
            )
            for path, content in plan.artifacts.items():
                self.assertEqual(path.read_bytes(), content)
            return CompletedProcess(command, 0)

        run.side_effect = launch
        pipeline.return_value = 0
        self.assertEqual(orchestrator.execute_local_plan(plan), 0)
        self.assertEqual(run.call_count, 1)
        pipeline.assert_called_once()
        namespace = pipeline.call_args.args[0]
        self.assertEqual(namespace.result_dir, plan.paths.result_root)
        self.assertTrue(namespace.skip_test)
        self.assertFalse(namespace.partial)
        self.assertFalse(namespace.skip_fit)
        saved = cli.load_saved(plan.paths.result_root)
        self.assertEqual(
            saved["grid"], str(plan.paths.result_root / "grid.snapshot.json")
        )

    @patch.object(orchestrator.subprocess, "run")
    @patch.object(orchestrator.unified_pipeline, "run")
    def test_execute_local_preserves_runner_failure(self, pipeline, run):
        run.return_value = CompletedProcess([], 9)
        plan = orchestrator.build_local_plan(
            self.local_args(),
            repo=self.repo,
            run_id="fail",
            machine="x86_64",
            inherited={},
        )
        self.assertEqual(orchestrator.execute_local_plan(plan), 9)
        self.assertEqual(run.call_count, 1)
        self.assertNotIn("timeout", run.call_args.kwargs)
        pipeline.assert_not_called()
        self.assertTrue((plan.paths.result_root / cli.MANIFEST).is_file())

    @patch.object(orchestrator.subprocess, "run")
    @patch.object(orchestrator.unified_pipeline, "run")
    def test_execute_local_returns_postprocess_failure(self, pipeline, run):
        run.return_value = CompletedProcess([], 0)
        pipeline.return_value = 23
        plan = orchestrator.build_local_plan(
            self.local_args(),
            repo=self.repo,
            run_id="reportfail",
            machine="x86_64",
            inherited={},
        )
        self.assertEqual(orchestrator.execute_local_plan(plan), 23)

    @patch.object(orchestrator.subprocess, "run")
    def test_execute_local_checks_all_artifact_collisions_before_writing(self, run):
        plan = orchestrator.build_local_plan(
            self.local_args(),
            repo=self.repo,
            run_id="collision",
            machine="x86_64",
            inherited={},
        )
        root = plan.paths.result_root
        root.mkdir()
        (root / cli.MANIFEST).write_bytes(b"existing manifest")
        with self.assertRaises(FileExistsError):
            orchestrator.execute_local_plan(plan)
        self.assertFalse((root / "profile.snapshot.json").exists())
        self.assertFalse((root / "grid.snapshot.json").exists())
        self.assertEqual((root / cli.MANIFEST).read_bytes(), b"existing manifest")
        run.assert_not_called()

    def test_write_artifacts_preflights_every_target_before_creating_parents(self):
        first = self.repo / "new" / "snapshot.json"
        existing = self.repo / "existing.json"
        existing.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            orchestrator.write_artifacts({first: b"new", existing: b"replace"})
        self.assertFalse(first.parent.exists())
        self.assertEqual(existing.read_bytes(), b"keep")

    def test_write_artifacts_rejects_dangling_symlink_before_writes(self):
        first = self.repo / "new" / "snapshot.json"
        link = self.repo / "link.json"
        link.symlink_to(self.repo / "absent")
        with self.assertRaises(FileExistsError):
            orchestrator.write_artifacts({first: b"new", link: b"replace"})
        self.assertFalse(first.parent.exists())
        self.assertFalse((self.repo / "absent").exists())

    def test_postprocess_namespace_matches_pipeline_defaults_and_partial_behavior(self):
        for partial in (False, True):
            with self.subTest(partial=partial):
                namespace = orchestrator.postprocess_namespace(
                    self.repo, partial=partial
                )
                self.assertEqual(namespace.mode, "pipeline")
                self.assertEqual(namespace.result_dir, self.repo)
                self.assertTrue(namespace.skip_test)
                self.assertEqual(namespace.partial, partial)
                self.assertEqual(namespace.skip_fit, partial)
                self.assertEqual(namespace.batch_size, 1)
                self.assertEqual(namespace.profile_backend, "kineto")
                self.assertEqual(namespace.estimator, "median")
                self.assertIsNone(namespace.result_root)
                self.assertIsNone(namespace.profile)
                self.assertIsNone(namespace.grid)
                self.assertIsNone(namespace.bazel)
                self.assertIsNone(namespace.oss_destination)


if __name__ == "__main__":
    unittest.main()

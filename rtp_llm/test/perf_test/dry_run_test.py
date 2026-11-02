import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.dry_run import create_preview, dispatch_plan
from rtp_llm.test.perf_test.plan import build_execution_plan


def _config(backend):
    execution = {"backend": "local", "devices": [0]}
    if backend == "aone":
        execution = {
            "backend": "aone",
            "resource": "daily",
            "gpu_type": "H20",
            "max_parallel": 2,
            "shard_count": 2,
        }
    return {
        "schema_version": 1,
        "model": {
            "label": "test-model",
            "type": "registered_model",
            "checkpoint": "/models/test",
        },
        "engine": {
            "max_seq_len": 4096,
            "max_batch_tokens": 8192,
            "concurrency_limit": 8,
        },
        "workload": {
            "kind": "cache_grid",
            "max_input_len": 2048,
            "cache_alignment": 64,
            "commit_tail_tokens": 64,
            "compute_steps": [128],
            "batch_sizes": [1],
        },
        "execution": execution,
        "output": {},
        "regression": {},
        "publish": {},
    }


def _gang_execution():
    return {
        "backend": "aone",
        "resource": "daily",
        "gpu_type": "H20",
        "max_parallel": 2,
        "gpus_per_machine": 1,
    }


def _gang_config():
    return {
        "schema_version": 1,
        "model": {
            "label": "test-model",
            "type": "registered_model",
            "checkpoint": "/models/test",
            "topology": {"tp": 2, "dp": 1, "pp": 1, "ep": 1},
        },
        "engine": {
            "max_seq_len": 4096,
            "max_batch_tokens": 8192,
            "concurrency_limit": 8,
        },
        "workload": {
            "kind": "cache_grid",
            "max_input_len": 2048,
            "cache_alignment": 64,
            "commit_tail_tokens": 64,
            "compute_steps": [128],
            "batch_sizes": [1],
        },
        "execution": _gang_execution(),
        "output": {},
        "regression": {},
        "publish": {},
    }


def _cases():
    return [
        {"input_len": 512, "cache_len": 0, "batch_size": 1},
        {"input_len": 1024, "cache_len": 512, "batch_size": 1},
    ]


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
                resolved=resolve_config(_config(backend)),
                cases=_cases(),
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

    def test_aone_dry_run_is_complete_and_side_effect_free(self):
        preview = self._assert_side_effect_free_preview("aone")

        self.assertEqual(2, len(preview["commands"]))
        self.assertEqual(
            ["build", "remote-test"],
            [command["stage"] for command in preview["commands"]],
        )
        for command in preview["commands"]:
            for shard in range(2):
                self.assertIn(
                    f"//generated/perf_test:shard_{shard:04d}", command["argv"]
                )
        self.assertIn("--jobs=2", preview["commands"][1]["argv"])

    def test_non_dry_run_delegates_to_executor(self):
        plan = build_execution_plan(
            mode="prefill",
            action="run",
            resolved=resolve_config(_config("local")),
            cases=_cases(),
            result_directory="runs/example",
        )
        execute = Mock(return_value={"status": "executed"})

        result = dispatch_plan(plan, dry_run=False, execute=execute)

        self.assertEqual({"status": "executed"}, result)
        execute.assert_called_once_with(plan)

    def test_gang_dry_run_is_complete_and_side_effect_free(self):
        create_run_directory = Mock()
        start_broker = Mock()
        generate_package = Mock()
        submit_job = Mock()

        def execute(_plan):
            create_run_directory()
            start_broker()
            generate_package()
            submit_job()

        with tempfile.TemporaryDirectory() as temp_dir:
            result_directory = Path(temp_dir) / "gang-run"
            plan = build_execution_plan(
                mode="prefill",
                action="run",
                resolved=resolve_config(_gang_config()),
                cases=_cases(),
                result_directory=str(result_directory),
            )

            with patch("subprocess.run", create_run_directory) as run_subprocess, patch(
                "subprocess.Popen", submit_job
            ) as popen_subprocess, patch(
                "rtp_llm.test.perf_test.aone.broker.run_gang_broker", start_broker
            ), patch(
                "rtp_llm.test.perf_test.aone.gang_package.generate_gang_package",
                generate_package,
            ), patch(
                "rtp_llm.test.perf_test.run_directory.initialize_run_directory",
                create_run_directory,
            ):
                preview = dispatch_plan(plan, dry_run=True, execute=execute)

            self.assertFalse(result_directory.exists())

        create_run_directory.assert_not_called()
        submit_job.assert_not_called()
        run_subprocess.assert_not_called()
        popen_subprocess.assert_not_called()
        start_broker.assert_not_called()
        generate_package.assert_not_called()

        self.assertTrue(preview["dry_run"])
        self.assertEqual("aone", preview["backend"])
        self.assertEqual(
            plan.to_dict()["resolved_config"], preview["normalized_config"]
        )
        self.assertEqual(plan.resource_plan.to_dict(), preview["resource_summary"])
        self.assertEqual(2, preview["cases"]["count"])
        self.assertNotIn("shards", preview)
        self.assertEqual("//generated/perf_test:gang", preview["gang"]["target"])

    def test_gang_dry_run_prints_topology_members_and_gang_commands(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            plan = build_execution_plan(
                mode="prefill",
                action="run",
                resolved=resolve_config(_gang_config()),
                cases=_cases(),
                result_directory=str(Path(temp_dir) / "gang-run"),
            )

            preview = create_preview(plan)

        gang = preview["gang"]
        self.assertEqual(2, gang["machine_count"])
        self.assertEqual(1, gang["gpus_per_machine"])
        self.assertEqual(1, gang["local_world_size"])
        self.assertEqual(2, gang["world_size"])
        self.assertEqual(0, gang["master_shard_index"])
        self.assertEqual(
            [
                {"shard_index": 0, "world_ranks": [0], "is_master": True},
                {"shard_index": 1, "world_ranks": [1], "is_master": False},
            ],
            gang["members"],
        )
        self.assertEqual(
            ["build", "remote-test"],
            [command["stage"] for command in preview["commands"]],
        )
        for command in preview["commands"]:
            self.assertIn("//generated/perf_test:gang", command["argv"])
            self.assertIn("--config=cuda13", command["argv"])
        self.assertIn("--config=daily_aone_bazel_cache", preview["commands"][0]["argv"])
        self.assertIn(
            "--config=daily_aone_bazel_remote", preview["commands"][1]["argv"]
        )
        self.assertIn("--strategy=TestRunner=remote", preview["commands"][1]["argv"])
        self.assertIn("--jobs=2", preview["commands"][1]["argv"])


if __name__ == "__main__":
    unittest.main()

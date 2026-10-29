import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.dry_run import dispatch_plan
from rtp_llm.test.perf_test.plan import build_execution_plan


def _config(backend):
    execution = {"backend": "local", "devices": [0]}
    if backend == "aone":
        execution = {
            "backend": "aone",
            "resource": "gpu",
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

        self.assertEqual(4, len(preview["commands"]))
        self.assertEqual(
            ["build-shard", "submit-shard", "build-shard", "submit-shard"],
            [command["stage"] for command in preview["commands"]],
        )
        self.assertEqual(
            [0, 0, 1, 1], [command["shard"] for command in preview["commands"]]
        )

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


if __name__ == "__main__":
    unittest.main()

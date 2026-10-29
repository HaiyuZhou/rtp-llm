import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.errors import PlanningError, ResourceError
from rtp_llm.test.perf_test.plan import build_execution_plan
from rtp_llm.test.perf_test.run_directory import (
    RunDirectoryLock,
    initialize_run_directory,
    inspect_run_directory,
)


def _source_config():
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
        "execution": {"backend": "local", "devices": [0]},
        "output": {},
        "regression": {},
        "publish": {},
    }


def _plan(result_directory):
    return build_execution_plan(
        mode="prefill",
        action="run",
        resolved=resolve_config(_source_config()),
        cases=[{"input_len": 512, "cache_len": 0, "batch_size": 1}],
        result_directory=str(result_directory),
    )


def _tree_bytes(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


class RunDirectoryTest(unittest.TestCase):
    def test_new_run_directory_is_self_describing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run-001"
            plan = _plan(root)

            paths = initialize_run_directory(
                root,
                source_config=_source_config(),
                plan=plan,
                run_id="run-001",
                code_version="deadbeef",
            )
            snapshot = inspect_run_directory(root)

            self.assertEqual(root, paths.root)
            for relative in (
                "workload",
                "checkpoints",
                "results/cases",
                "reports",
                "profiles",
                "traces",
            ):
                self.assertTrue((root / relative).is_dir(), relative)
            self.assertEqual(_source_config(), snapshot.source_config)
            self.assertEqual(
                plan.to_dict()["resolved_config"], snapshot.resolved_config
            )
            self.assertEqual(plan, snapshot.execution_plan)
            self.assertEqual("run-001", snapshot.manifest["run_id"])
            self.assertEqual("planned", snapshot.manifest["state"])
            self.assertEqual("deadbeef", snapshot.manifest["code_version"])
            self.assertEqual(
                [case.case_id for case in plan.cases],
                snapshot.manifest["case_ids"],
            )
            self.assertEqual(
                plan.resource_plan.to_dict(), snapshot.manifest["resource_plan"]
            )
            self.assertFalse(paths.writer_lock.exists())

    def test_second_writer_is_rejected_without_mutation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run-001"
            initialize_run_directory(
                root,
                source_config=_source_config(),
                plan=_plan(root),
                run_id="run-001",
                code_version="deadbeef",
            )
            with RunDirectoryLock(root, action="run"):
                before = _tree_bytes(root)
                with self.assertRaisesRegex(
                    ResourceError, "already has a writer"
                ) as error:
                    with RunDirectoryLock(root, action="resume"):
                        self.fail("second writer entered lock")
                self.assertEqual(3, error.exception.exit_code)
                self.assertEqual(before, _tree_bytes(root))

            self.assertFalse((root / ".writer.lock").exists())
            self.assertEqual("run-001", inspect_run_directory(root).manifest["run_id"])

    def test_existing_target_is_rejected_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run-001"
            root.mkdir()
            sentinel = root / "sentinel"
            sentinel.write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(PlanningError, "already exists"):
                initialize_run_directory(
                    root,
                    source_config=_source_config(),
                    plan=_plan(root),
                    run_id="run-001",
                    code_version="deadbeef",
                )

            self.assertEqual("keep", sentinel.read_text(encoding="utf-8"))
            self.assertEqual({"sentinel"}, {path.name for path in root.iterdir()})

    def test_snapshots_are_valid_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run-001"
            initialize_run_directory(
                root,
                source_config=_source_config(),
                plan=_plan(root),
                run_id="run-001",
                code_version="deadbeef",
            )
            for name in (
                "source_config.json",
                "resolved_config.json",
                "execution_plan.json",
                "manifest.json",
            ):
                with (root / name).open(encoding="utf-8") as stream:
                    self.assertIsInstance(json.load(stream), dict)


if __name__ == "__main__":
    unittest.main()

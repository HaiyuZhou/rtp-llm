import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.storage import RunStore, atomic_write_json


class AtomicStorageTest(unittest.TestCase):
    def test_fault_before_replace_preserves_committed_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.json"
            atomic_write_json(path, {"state": "planned"})

            def interrupt(stage, _target, _temporary):
                if stage == "before_replace":
                    raise KeyboardInterrupt("interrupted")

            with self.assertRaises(KeyboardInterrupt):
                atomic_write_json(path, {"state": "running"}, fault_injector=interrupt)

            self.assertEqual({"state": "planned"}, json.loads(path.read_text()))
            self.assertEqual(
                ["manifest.json"], [item.name for item in path.parent.iterdir()]
            )

    def test_fault_before_first_commit_exposes_no_case_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "case-a.json"

            def interrupt(stage, _target, _temporary):
                if stage == "before_replace":
                    raise RuntimeError("fault")

            with self.assertRaises(RuntimeError):
                atomic_write_json(path, {"status": "ok"}, fault_injector=interrupt)

            self.assertFalse(path.exists())
            self.assertEqual([], list(path.parent.iterdir()))

    def test_fault_after_replace_leaves_complete_new_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "summary.json"
            atomic_write_json(path, {"status": "old"})

            def interrupt(stage, _target, _temporary):
                if stage == "after_replace":
                    raise RuntimeError("fault")

            with self.assertRaises(RuntimeError):
                atomic_write_json(path, {"status": "new"}, fault_injector=interrupt)

            self.assertEqual({"status": "new"}, json.loads(path.read_text()))
            self.assertEqual(
                ["summary.json"], [item.name for item in path.parent.iterdir()]
            )

    def test_run_store_commits_every_mutable_artifact_atomically(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "checkpoints").mkdir()
            (root / "results" / "cases").mkdir(parents=True)
            store = RunStore(root)

            store.write_manifest({"state": "running"})
            store.write_checkpoint("progress", {"completed": ["case-a"]})
            store.write_case("case-a", {"case_id": "case-a", "status": "ok"})
            store.write_final_result({"status": "completed"})
            store.write_summary({"successful_cases": 1})

            expected = {
                "manifest.json": {"state": "running"},
                "checkpoints/progress.json": {"completed": ["case-a"]},
                "results/cases/case-a.json": {"case_id": "case-a", "status": "ok"},
                "results/result.json": {"status": "completed"},
                "results/summary.json": {"successful_cases": 1},
            }
            for relative, value in expected.items():
                with self.subTest(relative=relative):
                    self.assertEqual(value, json.loads((root / relative).read_text()))

    def test_committed_cases_ignore_temporary_files_and_are_sorted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cases = root / "results" / "cases"
            cases.mkdir(parents=True)
            store = RunStore(root)
            store.write_case("case-b", {"case_id": "case-b", "status": "ok"})
            store.write_case("case-a", {"case_id": "case-a", "status": "ok"})
            (cases / ".case-c.json.interrupted.tmp").write_text(
                '{"case_id":', encoding="utf-8"
            )

            committed = store.read_committed_cases()

            self.assertEqual(["case-a", "case-b"], list(committed))

    def test_invalid_artifact_names_cannot_escape_run_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "checkpoints").mkdir()
            (root / "results" / "cases").mkdir(parents=True)
            store = RunStore(root)

            for method, name in (
                (store.write_checkpoint, "../outside"),
                (store.write_case, "nested/case"),
                (store.write_case, ".hidden"),
            ):
                with self.subTest(name=name):
                    with self.assertRaises(ValueError):
                        method(name, {})


if __name__ == "__main__":
    unittest.main()

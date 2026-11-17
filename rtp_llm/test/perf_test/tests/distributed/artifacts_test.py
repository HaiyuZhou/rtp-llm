"""Gang collection validates identities and never imports controller snapshots."""

import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.errors import CollectionError
from rtp_llm.test.perf_test.execution.artifacts import collect_master_artifacts
from rtp_llm.test.perf_test.tests.support.gang import (
    RUN_ID,
    _setup_run,
    _write_master_run_tree,
)


class ArtifactsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name)
        self.root = self.home / "controller"
        self.plan = _setup_run(self.root)
        self.outputs = self.home / "remote"
        self.outputs.mkdir()
        _write_master_run_tree(self.outputs, self.plan)
        self.tree = self.outputs / "run"

    def collect(self):
        return collect_master_artifacts(
            self.root, self.tree, plan=self.plan, run_id=RUN_ID
        )

    def test_controller_snapshots_survive_valid_collection(self):
        manifest = (self.root / "manifest.json").read_bytes()
        (self.tree / "manifest.json").write_text('{"state":"forged"}')
        result = self.collect()
        self.assertEqual("completed", result.state.value)
        self.assertEqual(len(self.plan.cases), len(result.cases))
        self.assertEqual(manifest, (self.root / "manifest.json").read_bytes())

    def test_stale_run_identity_rejected_before_copy(self):
        path = self.tree / "results/result.json"
        envelope = json.loads(path.read_text())
        envelope["run_id"] = "stale"
        path.write_text(json.dumps(envelope))
        with self.assertRaisesRegex(CollectionError, "run_id"):
            self.collect()
        self.assertFalse((self.root / "results/result.json").exists())

    def test_remote_symlink_cannot_escape_collection(self):
        (self.tree / "results/leak").symlink_to(self.root / "manifest.json")
        with self.assertRaisesRegex(CollectionError, "symlink"):
            self.collect()
        self.assertFalse((self.root / "results/result.json").exists())


if __name__ == "__main__":
    unittest.main()

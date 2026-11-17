import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.errors import ConfigurationError
from rtp_llm.test.perf_test.ssh.source import (
    build_identity,
    resolve_source,
    validate_receipt,
)


class SourceTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.git("init")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "user.name", "Test")
        (self.root / "code.py").write_text("value = 1\n")
        self.git("add", "code.py")
        self.git("commit", "-m", "test")

    def git(self, *args):
        return subprocess.check_output(
            ["git", "-C", str(self.root), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()

    def source(self):
        return {
            "worktree": str(self.root),
            "ref": "auto",
            "build_policy": "reuse",
            "build_driver": None,
            "bazel_config": "cuda13",
            "bazel_targets": ["//:target"],
            "bazel_args": [],
        }

    def test_auto_freezes_exact_public_commit_and_untracked_is_excluded(self):
        (self.root / "personal.json").write_text("{}")
        frozen = resolve_source(self.source(), source_dir=self.root)
        self.assertEqual(frozen["ref"], self.git("rev-parse", "HEAD"))
        self.assertEqual(frozen["worktree"], str(self.root))

    def test_dirty_tracked_code_is_rejected(self):
        (self.root / "code.py").write_text("value = 2\n")
        with self.assertRaisesRegex(ConfigurationError, "uncommitted"):
            resolve_source(self.source(), source_dir=self.root)

    def test_build_identity_changes_with_flags(self):
        source = resolve_source(self.source(), source_dir=self.root)
        changed = {**source, "bazel_args": ["--jobs=2"]}
        self.assertNotEqual(build_identity(source), build_identity(changed))

    def test_receipt_checks_actual_artifact_content(self):
        import hashlib

        artifact = self.root / "lib.so"
        artifact.write_bytes(b"native")
        source = resolve_source(self.source(), source_dir=self.root)
        receipt = {
            "schema_version": 1,
            "status": "completed",
            "bazel_exit_code": 0,
            "source_sha": source["ref"],
            "artifacts": {str(artifact): hashlib.sha256(b"native").hexdigest()},
        }
        validate_receipt(receipt, source=source, source_dir=self.root)
        artifact.write_bytes(b"changed")
        with self.assertRaisesRegex(ConfigurationError, "artifact"):
            validate_receipt(receipt, source=source, source_dir=self.root)
        with self.assertRaises(ConfigurationError):
            validate_receipt(
                {**receipt, "source_sha": "0" * 40}, source=source, source_dir=self.root
            )


if __name__ == "__main__":
    unittest.main()

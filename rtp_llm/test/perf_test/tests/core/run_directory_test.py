"""Tests for core run_directory_test."""

import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.errors import ResourceError
from rtp_llm.test.perf_test.run_directory import (
    RunDirectoryLock,
    initialize_run_directory,
    inspect_run_directory,
)
from rtp_llm.test.perf_test.tests.support.run_storage import (
    run_directory_plan,
    run_directory_source_config,
    run_directory_tree_bytes,
)


class RunDirectoryTest(unittest.TestCase):

    def test_second_writer_is_rejected_without_mutation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run-001"
            initialize_run_directory(
                root,
                source_config=run_directory_source_config(),
                plan=run_directory_plan(root),
                run_id="run-001",
                code_version="deadbeef",
            )
            with RunDirectoryLock(root, action="run"):
                before = run_directory_tree_bytes(root)
                with self.assertRaisesRegex(
                    ResourceError, "already has a writer"
                ) as error:
                    with RunDirectoryLock(root, action="resume"):
                        self.fail("second writer entered lock")
                self.assertEqual(3, error.exception.exit_code)
                self.assertEqual(before, run_directory_tree_bytes(root))
            with RunDirectoryLock(root, action="verify-released"):
                pass
            self.assertEqual("run-001", inspect_run_directory(root).manifest["run_id"])


if __name__ == "__main__":
    unittest.main()

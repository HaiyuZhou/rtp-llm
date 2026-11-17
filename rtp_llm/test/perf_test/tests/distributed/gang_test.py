"""Shared stage bookkeeping preserves failure precedence across backends."""

import unittest
from unittest.mock import Mock

from rtp_llm.test.perf_test.execution.gang import reconcile_terminal_state, record_stage
from rtp_llm.test.perf_test.state import RunState


class GangBookkeepingTest(unittest.TestCase):
    def test_submit_failure_overrides_completed_and_partial_payload(self):
        for state in (RunState.COMPLETED, RunState.PARTIAL):
            self.assertEqual(
                RunState.FAILED, reconcile_terminal_state(state, RuntimeError("failed"))
            )
        self.assertEqual(
            RunState.INTERRUPTED,
            reconcile_terminal_state(RunState.INTERRUPTED, RuntimeError("cleanup")),
        )

    def test_stage_writes_existing_manifest_without_another_state_store(self):
        store = Mock()
        manifest = {"state": "planned"}
        record_stage(
            store,
            manifest,
            stage="submit",
            status="failed",
            state=RunState.FAILED,
            details={"member_id": "1"},
        )
        self.assertEqual("failed", manifest["state"])
        self.assertEqual(2, manifest["stages"]["submit"]["stage_index"])
        self.assertEqual("1", manifest["stages"]["submit"]["member_id"])
        store.write_manifest.assert_called_once_with(manifest)


if __name__ == "__main__":
    unittest.main()

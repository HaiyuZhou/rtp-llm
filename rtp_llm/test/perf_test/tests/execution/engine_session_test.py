"""Tests for execution engine_session_test."""

import signal
import unittest
from unittest.mock import Mock

from rtp_llm.test.perf_test.errors import EngineLifecycleError
from rtp_llm.test.perf_test.execution.engine_session import (
    EngineInterrupted,
    EngineSession,
)
from rtp_llm.test.perf_test.tests.support.execution import engine_session_FakeHandle


class EngineSessionTest(unittest.TestCase):

    def _session(self, handle, **kwargs):
        launcher = Mock(return_value=handle)
        timelines = Mock(return_value=["traces/timeline.json"])
        coredumps = Mock(return_value={"count": 1, "files": ["core.123"]})
        session = EngineSession(
            launcher=launcher,
            readiness_timeout_seconds=30,
            timeline_collector=timelines,
            coredump_collector=coredumps,
            install_signal_handlers=False,
            **kwargs,
        )
        return (session, launcher, timelines, coredumps)

    def test_success_exposes_client_and_collects_diagnostics_on_teardown(self):
        handle = engine_session_FakeHandle()
        (session, launcher, timelines, coredumps) = self._session(handle)
        with session as active:
            self.assertIs(handle.client, active.client)
            self.assertEqual("/tmp/process.log", active.log_path)
            active.assert_healthy()
        launcher.assert_called_once_with()
        self.assertEqual([30], handle.wait_ready_calls)
        self.assertEqual(1, handle.stop_calls)
        timelines.assert_called_once_with()
        coredumps.assert_called_once_with()
        self.assertEqual(("traces/timeline.json",), session.diagnostics.timeline_files)
        self.assertEqual(1, session.diagnostics.coredumps["count"])

    def test_startup_failure_stops_and_preserves_diagnostics(self):
        handle = engine_session_FakeHandle(ready=False)
        handle.exit_code = 17
        (session, launcher, timelines, coredumps) = self._session(handle)
        with self.assertRaisesRegex(EngineLifecycleError, "failed readiness") as error:
            session.start()
        launcher.assert_called_once_with()
        self.assertEqual(1, handle.stop_calls)
        timelines.assert_called_once_with()
        coredumps.assert_called_once_with()
        self.assertEqual(17, error.exception.diagnostics.exit_code)
        session.close()
        self.assertEqual(1, handle.stop_calls)

    def test_interrupt_closes_session_and_carries_signal(self):
        handle = engine_session_FakeHandle()
        (session, _, _, _) = self._session(handle)
        with self.assertRaises(EngineInterrupted) as error:
            with session:
                try:
                    session.handle_signal(signal.SIGINT)
                except EngineInterrupted:
                    self.assertEqual(0, handle.stop_calls)
                    raise
        self.assertEqual(signal.SIGINT, error.exception.signal_number)
        self.assertEqual(1, handle.stop_calls)
        session.close()
        self.assertEqual(1, handle.stop_calls)

    def test_multiple_cases_share_exactly_one_startup(self):
        handle = engine_session_FakeHandle()
        (session, launcher, _, _) = self._session(handle)
        with session:
            session.start()
            for _case in range(5):
                self.assertIs(handle.client, session.client)
                session.assert_healthy()
        launcher.assert_called_once_with()
        self.assertEqual([30], handle.wait_ready_calls)
        self.assertEqual(1, handle.stop_calls)


if __name__ == "__main__":
    unittest.main()

import signal
import unittest
from unittest.mock import Mock

from rtp_llm.test.perf_test.errors import EngineLifecycleError
from rtp_llm.test.perf_test.execution.engine_session import (
    EngineInterrupted,
    EngineSession,
)


class _FakeHandle:
    def __init__(self, *, ready=True):
        self.ready = ready
        self.alive = True
        self.exit_code = None
        self.client = object()
        self.log_path = "/tmp/process.log"
        self.wait_ready_calls = []
        self.stop_calls = 0

    def wait_ready(self, timeout_seconds):
        self.wait_ready_calls.append(timeout_seconds)
        return self.ready

    def is_alive(self):
        return self.alive

    def stop(self):
        self.stop_calls += 1
        self.alive = False


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
        return session, launcher, timelines, coredumps

    def test_success_exposes_client_and_collects_diagnostics_on_teardown(self):
        handle = _FakeHandle()
        session, launcher, timelines, coredumps = self._session(handle)

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
        handle = _FakeHandle(ready=False)
        handle.exit_code = 17
        session, launcher, timelines, coredumps = self._session(handle)

        with self.assertRaisesRegex(EngineLifecycleError, "failed readiness") as error:
            session.start()

        launcher.assert_called_once_with()
        self.assertEqual(1, handle.stop_calls)
        timelines.assert_called_once_with()
        coredumps.assert_called_once_with()
        self.assertEqual(17, error.exception.diagnostics.exit_code)
        session.close()
        self.assertEqual(1, handle.stop_calls)

    def test_mid_run_crash_is_detected_before_client_access(self):
        handle = _FakeHandle()
        session, _, _, _ = self._session(handle)

        with self.assertRaisesRegex(EngineLifecycleError, "exited during run") as error:
            with session:
                handle.alive = False
                handle.exit_code = 9
                _ = session.client

        self.assertEqual(9, error.exception.diagnostics.exit_code)
        self.assertEqual(1, handle.stop_calls)

    def test_interrupt_closes_session_and_carries_signal(self):
        handle = _FakeHandle()
        session, _, _, _ = self._session(handle)
        session.start()

        with self.assertRaises(EngineInterrupted) as error:
            session.handle_signal(signal.SIGINT)

        self.assertEqual(signal.SIGINT, error.exception.signal_number)
        self.assertEqual(1, handle.stop_calls)
        session.close()
        self.assertEqual(1, handle.stop_calls)

    def test_multiple_cases_share_exactly_one_startup(self):
        handle = _FakeHandle()
        session, launcher, _, _ = self._session(handle)

        with session:
            session.start()
            for _case in range(5):
                self.assertIs(handle.client, session.client)
                session.assert_healthy()

        launcher.assert_called_once_with()
        self.assertEqual([30], handle.wait_ready_calls)
        self.assertEqual(1, handle.stop_calls)

    def test_context_exception_still_tears_down(self):
        handle = _FakeHandle()
        session, _, timelines, coredumps = self._session(handle)

        with self.assertRaisesRegex(RuntimeError, "case failed"):
            with session:
                raise RuntimeError("case failed")

        self.assertEqual(1, handle.stop_calls)
        timelines.assert_called_once_with()
        coredumps.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

"""Bounded continuous request supply, without engine imports."""

import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm.test.perf_test.cache_grid.runner.continuous_pressure import (
    ArrivalClock,
    BoundedSender,
    CancellableRequestHandle,
)


class BoundedSenderTest(unittest.TestCase):
    def test_does_not_construct_or_submit_when_full(self):
        sender = BoundedSender(2)
        first, second, third = Future(), Future(), Future()
        calls = []
        self.assertTrue(sender.submit(0, lambda: first))
        self.assertTrue(sender.submit(1, lambda: second))
        self.assertFalse(sender.submit(2, lambda: calls.append(2) or third))
        self.assertEqual(calls, [])
        first.set_result({"success": True})
        self.assertEqual(sender.collect(), [(0, {"success": True})])
        self.assertTrue(sender.submit(2, lambda: third))
        sender.cancel_all()
        self.assertTrue(second.cancelled())
        self.assertTrue(third.cancelled())

    def test_failure_becomes_record_and_releases_slot(self):
        sender = BoundedSender(1)
        failed = Future()
        failed.set_exception(RuntimeError("broken"))
        sender.submit(1, lambda: failed)
        self.assertEqual(
            sender.collect(),
            [(1, {"success": False, "error": "RuntimeError('broken')"})],
        )
        self.assertEqual(sender.collect(), [])
        self.assertTrue(sender.submit(2, Future))

    def test_invalid_limit_and_duplicate_id(self):
        with self.assertRaises(ValueError):
            BoundedSender(0)
        sender = BoundedSender(1)
        sender.submit(0, Future)
        with self.assertRaises(ValueError):
            sender.submit(0, Future)

    def test_backpressure_does_not_advance_schedule(self):
        clock = ArrivalClock(2, False, 0)
        sender = BoundedSender(1)
        first = Future()
        due = clock.next_offset()
        sender.submit(0, lambda: first)
        due = clock.next_offset()
        self.assertFalse(sender.submit(1, Future))
        self.assertEqual(due, 0.5)
        first.set_result({})
        sender.collect()
        self.assertTrue(sender.submit(1, Future))
        self.assertEqual(clock.next_offset(), 1.0)


class ArrivalClockTest(unittest.TestCase):
    def test_burst_and_constant_offsets(self):
        burst = ArrivalClock(0, False, 0)
        regular = ArrivalClock(2, False, 0)
        self.assertEqual([burst.next_offset() for _ in range(3)], [0, 0, 0])
        self.assertEqual([regular.next_offset() for _ in range(3)], [0, 0.5, 1])

    def test_poisson_replays_seed(self):
        first, second = ArrivalClock(2, True, 9), ArrivalClock(2, True, 9)
        offsets = [first.next_offset() for _ in range(10)]
        self.assertEqual(offsets, [second.next_offset() for _ in range(10)])
        self.assertTrue(all(b > a for a, b in zip(offsets, offsets[1:])))


class CancellableRequestHandleTest(unittest.TestCase):
    def test_cancels_running_rpc(self):
        started, cancelled = Event(), Event()

        class Call:
            def cancel(self):
                cancelled.set()
                return True

        def request(register_call):
            register_call(Call())
            started.set()
            if not cancelled.wait(2):
                raise AssertionError("RPC was not cancelled")
            return {"success": False}

        with ThreadPoolExecutor(max_workers=1) as executor:
            handle = CancellableRequestHandle(executor, request)
            self.assertTrue(started.wait(2))
            self.assertTrue(handle.cancel())
            self.assertTrue(cancelled.is_set())
            self.assertEqual(handle.result(), {"success": False})
            self.assertTrue(handle.done())

    def test_cancel_queued_worker_does_not_start_request(self):
        entered, release = Event(), Event()
        calls = []

        def occupy_worker():
            entered.set()
            release.wait(2)

        with ThreadPoolExecutor(max_workers=1) as executor:
            blocker = executor.submit(occupy_worker)
            self.assertTrue(entered.wait(2))
            handle = CancellableRequestHandle(
                executor, lambda register: calls.append("started") or {}
            )
            self.assertTrue(handle.cancel())
            release.set()
            blocker.result()
            self.assertTrue(handle.done())
            self.assertEqual(calls, [])

    def test_cancel_before_rpc_registration_cancels_late_call(self):
        entered, proceed, cancelled = Event(), Event(), Event()

        class Call:
            def cancel(self):
                cancelled.set()
                return True

        def request(register_call):
            entered.set()
            proceed.wait(2)
            register_call(Call())
            return {}

        with ThreadPoolExecutor(max_workers=1) as executor:
            handle = CancellableRequestHandle(executor, request)
            self.assertTrue(entered.wait(2))
            handle.cancel()
            proceed.set()
            self.assertEqual(handle.result(), {})
            self.assertTrue(cancelled.is_set())


class TransportRegistrationTest(unittest.TestCase):
    def test_post_prefill_ids_registers_actual_call_before_iteration(self):
        from rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner import (
            _post_prefill_ids,
        )

        registered = []

        class Call:
            cancelled = False

            def cancel(self):
                self.cancelled = True
                return True

            def __iter__(self):
                self_test.assertEqual(registered, [self])
                self_test.assertTrue(self.cancelled)
                return iter(())

        self_test = self
        call = Call()
        stub = Mock()
        stub.ModelStreamInfer.return_value = call

        def register_call(value):
            registered.append(value)
            value.cancel()

        # Only wire builders are replaced: importing them otherwise loads the
        # engine extension. Exercise the real adapter and stream lifecycle.
        dependencies = {
            "rtp_llm.dash_sc.client": SimpleNamespace(
                build_model_infer_request=lambda **kwargs: kwargs
            ),
            "rtp_llm.dash_sc.codec": SimpleNamespace(
                SamplingParams=lambda **kwargs: kwargs
            ),
        }
        with patch.dict("sys.modules", dependencies):
            result = _post_prefill_ids(
                stub, [1, 2], 17, "registered", register_call=register_call
            )
        self.assertTrue(result["success"])
        self.assertEqual(stub.ModelStreamInfer.call_args.kwargs["timeout"], 17)


if __name__ == "__main__":
    unittest.main()

"""Bounded request handles and cumulative session arrival schedules.

The supervisor owns the sender and clock. It retains the next arrival while
backpressured and only advances it after a successful submission; workers never
sleep waiting for scheduled arrivals.
"""

from __future__ import annotations

import random
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Protocol


class RequestHandle(Protocol):
    def done(self) -> bool: ...

    def result(self) -> dict: ...

    def cancel(self) -> bool: ...


class BoundedSender:
    """Supervisor-owned bounded set of started requests (no submission queue)."""

    def __init__(self, limit: int):
        if limit < 1:
            raise ValueError("positive sender limit required")
        self.limit = limit
        self.pending: dict[int, RequestHandle] = {}

    def submit(self, job_id: int, make_handle: Callable[[], RequestHandle]) -> bool:
        if job_id in self.pending:
            raise ValueError("duplicate job ID")
        if len(self.pending) >= self.limit:
            return False
        self.pending[job_id] = make_handle()
        return True

    def collect(self) -> list[tuple[int, dict]]:
        ready = []
        for job_id, handle in list(self.pending.items()):
            if not handle.done():
                continue
            try:
                result = handle.result()
            except Exception as exc:
                result = {"success": False, "error": repr(exc)}
            ready.append((job_id, result))
            del self.pending[job_id]
        return ready

    def cancel_all(self) -> None:
        for handle in self.pending.values():
            handle.cancel()


class ArrivalClock:
    """Cumulative offsets from one formal-distribution session origin."""

    def __init__(self, rate: float, poisson: bool, seed: int):
        self.rate, self.poisson = rate, poisson
        self.rng = random.Random(seed)
        self.offset = 0.0
        self.count = 0

    def next_offset(self) -> float:
        if self.count and self.rate > 0:
            self.offset += (
                self.rng.expovariate(self.rate) if self.poisson else 1.0 / self.rate
            )
        self.count += 1
        return self.offset


class CancellableRequestHandle:
    """A bounded worker plus the actual RPC's cancellation, including start races.

    Construct only inside ``BoundedSender.submit``'s factory. The request receives
    a registration callback to pass as ``_post_prefill_ids(register_call=...)``.
    The supervisor owns a dedicated executor with at least the sender limit in
    workers. Generate inputs inside the accepted factory or request, never in an
    unbounded job list. On shutdown, stop submissions, call ``cancel_all``, then
    collect until a bounded drain deadline. Executor shutdown must not wait
    indefinitely; RPC deadlines remain the final bound on transport workers.
    """

    def __init__(
        self,
        executor: ThreadPoolExecutor,
        request: Callable[[Callable[[Any], None]], dict],
    ):
        self._lock = threading.Lock()
        self._call = None
        self._cancel_requested = False
        self._future = executor.submit(request, self._register_call)

    def _register_call(self, call: Any) -> None:
        with self._lock:
            self._call = call
            cancelled = self._cancel_requested
        if cancelled:
            call.cancel()

    def done(self) -> bool:
        return self._future.done()

    def result(self) -> dict:
        return self._future.result()

    def cancel(self) -> bool:
        with self._lock:
            self._cancel_requested = True
            call = self._call
        worker_cancelled = self._future.cancel()
        if call is not None:
            return bool(call.cancel()) or worker_cancelled
        return worker_cancelled

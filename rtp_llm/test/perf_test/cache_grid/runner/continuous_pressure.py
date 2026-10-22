"""Bounded request handles and cumulative session arrival schedules.

The supervisor owns the sender and clock. It retains the next arrival while
backpressured and only advances it after a successful submission; workers never
sleep waiting for scheduled arrivals.
"""

from __future__ import annotations

import hashlib
import math
import random
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, Callable, Protocol

from .pressure_diagnostics import (
    PressureCoverage,
    PressureTarget,
    assess_activity,
    group_valid,
    select_targets,
    validate_inventory,
)
from .pressure_seed_plan import SeedPlan


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
        if isinstance(rate, bool) or not math.isfinite(rate) or rate < 0:
            raise ValueError("arrival rate must be finite and nonnegative")
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


class AdaptiveBacklogController:
    """Piecewise-linear offered rate derived from unfinished RPC count."""

    def __init__(self, *, rate_min, rate_max, backlog_low, backlog_high):
        if (
            isinstance(rate_min, bool)
            or isinstance(rate_max, bool)
            or not math.isfinite(rate_min)
            or not math.isfinite(rate_max)
            or rate_min <= 0
            or rate_max < rate_min
        ):
            raise ValueError("adaptive rates must satisfy 0 < min <= max")
        if (
            type(backlog_low) is not int
            or type(backlog_high) is not int
            or backlog_low < 0
            or backlog_high <= backlog_low
        ):
            raise ValueError("adaptive backlog must satisfy 0 <= low < high")
        self.rate_min = float(rate_min)
        self.rate_max = float(rate_max)
        self.backlog_low = backlog_low
        self.backlog_high = backlog_high

    def rate(self, outstanding: int) -> float:
        if type(outstanding) is not int or outstanding < 0:
            raise ValueError("outstanding must be a nonnegative integer")
        if outstanding <= self.backlog_low:
            return self.rate_max
        if outstanding >= self.backlog_high:
            return self.rate_min
        remaining = (self.backlog_high - outstanding) / (
            self.backlog_high - self.backlog_low
        )
        return self.rate_min + (self.rate_max - self.rate_min) * remaining


@dataclass(frozen=True)
class TieredRollingAssignment:
    cohort_id: int
    cohort_lane: int
    family: int
    case_index: int
    expected_cache_tier: str
    promoted_from_cohort: int | None = None


class TieredRollingPlanner:
    """Build fixed cohorts whose memory half becomes the next device half."""

    def __init__(
        self,
        *,
        effective_seed_count: int,
        cohort_size: int,
        memory_reuse_ratio: float,
        case_indices: tuple[int, ...],
    ):
        if type(effective_seed_count) is not int or effective_seed_count <= 0:
            raise ValueError("effective_seed_count must be positive")
        if type(cohort_size) is not int or cohort_size <= 0:
            raise ValueError("cohort_size must be positive")
        if (
            isinstance(memory_reuse_ratio, bool)
            or not math.isfinite(memory_reuse_ratio)
            or not 0 < memory_reuse_ratio < 1
        ):
            raise ValueError("memory_reuse_ratio must be between zero and one")
        memory_count = cohort_size * memory_reuse_ratio
        if not memory_count.is_integer():
            raise ValueError("memory reuse ratio must produce an integral cohort")
        self.memory_count = int(memory_count)
        self.device_count = cohort_size - self.memory_count
        if self.memory_count != self.device_count:
            raise ValueError("tiered rolling currently requires a 50% memory ratio")
        if not case_indices or self.memory_count % len(case_indices):
            raise ValueError("cache cases must divide each tier half evenly")
        self.effective_seed_count = effective_seed_count
        self.cohort_size = cohort_size
        self.case_indices = tuple(case_indices)
        self.hot_start = effective_seed_count - self.device_count
        self.memory_window_end = (
            self.hot_start // self.memory_count
        ) * self.memory_count
        if self.memory_window_end < 2 * self.memory_count:
            raise ValueError("effective seed count must provide two memory windows")
        self.memory_cursor = 0
        self.cohort_id = 0
        self.previous_memory: tuple[TieredRollingAssignment, ...] | None = None

    def _cases(self) -> tuple[int, ...]:
        repeats = self.memory_count // len(self.case_indices)
        return self.case_indices * repeats

    def assignment(self, sequence: int) -> TieredRollingAssignment:
        if type(sequence) is not int or sequence < 0:
            raise ValueError("tiered rolling sequence must be nonnegative")
        cohort_id, cohort_lane = divmod(sequence, self.cohort_size)
        lane = cohort_lane // 2
        cases = self._cases()
        case_index = cases[lane]
        memory_start = (cohort_id * self.memory_count) % self.memory_window_end
        if cohort_lane % 2:
            return TieredRollingAssignment(
                cohort_id=cohort_id,
                cohort_lane=cohort_lane,
                family=memory_start + lane,
                case_index=case_index,
                expected_cache_tier="memory",
            )
        if cohort_id == 0:
            return TieredRollingAssignment(
                cohort_id=cohort_id,
                cohort_lane=cohort_lane,
                family=self.hot_start + lane,
                case_index=case_index,
                expected_cache_tier="device",
            )
        previous_start = ((cohort_id - 1) * self.memory_count) % (
            self.memory_window_end
        )
        return TieredRollingAssignment(
            cohort_id=cohort_id,
            cohort_lane=cohort_lane,
            family=previous_start + lane,
            case_index=case_index,
            expected_cache_tier="device",
            promoted_from_cohort=cohort_id - 1,
        )

    def next_cohort(self) -> tuple[TieredRollingAssignment, ...]:
        start = self.cohort_id * self.cohort_size
        result = tuple(
            self.assignment(start + lane) for lane in range(self.cohort_size)
        )
        self.cohort_id += 1
        return result


def freeze_seed_families(seed_ids) -> int:
    """Return the populated prefix length, rejecting holes in seed families."""
    effective = len(seed_ids)
    for index, ids in enumerate(seed_ids):
        if ids is None:
            effective = index
            break
    if any(ids is not None for ids in seed_ids[effective:]):
        raise ValueError("completed seed families must form a contiguous prefix")
    return effective


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
        on_done: Callable[[], None] | None = None,
    ):
        self._lock = threading.Lock()
        self._call = None
        self._cancel_requested = False
        self._future = executor.submit(request, self._register_call)
        if on_done is not None:
            self._future.add_done_callback(lambda future: on_done())

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


@dataclass(frozen=True)
class ExecutionLimits:
    """Verified service limits, in tokens and per-request physical KV blocks.

    available_kv_blocks is the conservative simultaneous request budget after
    non-reclaimable reservations, not total configured cache capacity.
    When supplied, owner_kv_budgets replaces the single-pool projection with
    (owner/group/pool identity, available blocks, tokens per block) per pool.
    """

    model_tokens: int
    admission_requests: int
    batch_tokens: int
    available_kv_blocks: int
    block_tokens: int
    owner_kv_budgets: tuple[tuple[str, int, int], ...] = ()

    def __post_init__(self):
        if (
            any(
                type(value) is not int or value <= 0
                for value in (
                    self.model_tokens,
                    self.admission_requests,
                    self.batch_tokens,
                    self.block_tokens,
                )
            )
            or type(self.available_kv_blocks) is not int
            or self.available_kv_blocks < 0
        ):
            raise ValueError("execution limits must be positive integers")
        if type(self.owner_kv_budgets) is not tuple or any(
            type(budget) is not tuple
            or len(budget) != 3
            or not isinstance(budget[0], str)
            or not budget[0]
            or type(budget[1]) is not int
            or budget[1] < 0
            or type(budget[2]) is not int
            or budget[2] <= 0
            for budget in self.owner_kv_budgets
        ):
            raise ValueError("invalid immutable owner KV budgets")

    def concurrency_for(self, input_tokens):
        if type(input_tokens) is not int or not 0 < input_tokens < self.model_tokens:
            raise ValueError(
                f"input_tokens={input_tokens} plus one output exceeds "
                f"model token limit={self.model_tokens}"
            )
        tokens = input_tokens + 1
        if tokens > self.batch_tokens:
            raise ValueError(
                f"input_tokens={input_tokens} plus one output exceeds "
                f"batch token limit={self.batch_tokens}"
            )
        count = min(self.admission_requests, self.batch_tokens // tokens)
        budgets = self.owner_kv_budgets or (
            ("service", self.available_kv_blocks, self.block_tokens),
        )
        for name, available, block_tokens in budgets:
            blocks = (tokens + block_tokens - 1) // block_tokens
            if blocks > available:
                raise ValueError(
                    f"{name}: KV admission requires {blocks} blocks "
                    f"for input_tokens={input_tokens} plus one output; "
                    f"available_blocks={available} block_tokens={block_tokens}"
                )
            count = min(count, available // blocks)
        return count

    def require_request(self, input_tokens: int) -> None:
        self.concurrency_for(input_tokens)


class PressurePhases:
    def __init__(self, config, started_s):
        self.config = config
        self.phase = "fill"
        self.entered = started_s
        self.measurement_start = None
        self.measurement_end = None
        self.failure = None

    def fail(self, reason):
        if self.failure is None:
            self.failure = reason
        self.phase = "drain"

    def advance(
        self, now_s, *, fill_ready=False, steady_covered_s=0.0, profile_pending=False
    ):
        if self.phase in ("fill", "steady"):
            timeout = (
                self.config.fill_timeout_s
                if self.phase == "fill"
                else self.config.steady_timeout_s
            )
            if now_s - self.entered >= timeout:
                self.fail("pressure_not_reached")
            elif self.phase == "fill" and fill_ready:
                self.phase, self.entered = "steady", now_s
            elif self.phase == "steady" and steady_covered_s >= self.config.steady_s:
                self.phase = "measurement"
                self.measurement_start = now_s
                self.measurement_end = now_s + self.config.measure_s
        elif self.phase == "measurement" and now_s >= self.measurement_end:
            self.phase = "support" if profile_pending else "drain"
        if self.phase == "support":
            if not profile_pending:
                self.phase = "drain"
            elif now_s >= self.measurement_end + self.config.support_s:
                self.fail("profile_timeout")
        return self.phase


@dataclass(frozen=True)
class PressurePoll:
    samples: tuple
    raw: dict
    activity: tuple = ()
    step_page_full: bool = False
    step_history_complete: bool = True
    sample_complete: bool = True


class DispatchClosed(Exception):
    """Input preparation crossed the final supply cutoff; no RPC was sent."""


class PressureSession:
    """Bounded supervisor. Callbacks do not retain an experiment's event history.

    poll(timeout) returns PressurePoll; request(payload, register, before) invokes
    before(payload) immediately before its real RPC and uses the returned ID.
    make_payload(kind, job_id, formal_sequence) runs only in accepted request
    workers. sink(event) must be fast, thread safe and persist/aggregate
    incrementally. Controller start(now), pending(now), stop(reason, now) must
    be nonblocking.
    """

    def __init__(
        self,
        config,
        *,
        session_id,
        limits,
        concurrency,
        scheduler,
        batch_size,
        max_input_tokens,
        seed_plan: SeedPlan,
        startup_targets: tuple[PressureTarget, ...],
        make_payload,
        request,
        poll,
        sink,
        set_batch,
        controller=None,
        diagnostics_control=None,
        rate=0,
        poisson=False,
        random_seed=0,
        clock=time.monotonic,
        sleep=time.sleep,
        executor_factory=ThreadPoolExecutor,
    ):
        self.config, self.limits = config, limits
        self.seed_plan = seed_plan
        self.startup_targets = startup_targets
        seed_count = seed_plan.seed_count
        seed_input_tokens = seed_plan.seed_input_tokens
        self.arrivals = ArrivalClock(rate, poisson, random_seed)
        if scheduler != "fifo":
            raise ValueError("continuous full-pressure preload is FIFO-only")
        if batch_size != 1 or type(batch_size) is not int:
            raise ValueError("FIFO full-pressure preload has no fixed batch")
        if type(concurrency) is not int or concurrency < 1:
            raise ValueError("positive concurrency required")
        limits.require_request(max_input_tokens)
        limits.require_request(seed_input_tokens)
        self.fill_batch = min(
            concurrency, limits.concurrency_for(config.fill_input_tokens)
        )
        if type(seed_count) is not int or seed_count < 0:
            raise ValueError("invalid seed count")
        self.session_id, self.scheduler, self.batch_size = (
            session_id,
            scheduler,
            batch_size,
        )
        self.seed_count = seed_count
        self.adaptive = None
        if config.rate_control == "adaptive_backlog":
            if poisson:
                raise ValueError("adaptive backlog requires constant arrivals")
            if rate != 0:
                raise ValueError(
                    "adaptive backlog uses request_rate_max instead of fixed rate"
                )
            self.adaptive = AdaptiveBacklogController(
                rate_min=config.request_rate_min,
                rate_max=config.request_rate_max,
                backlog_low=config.backlog_low,
                backlog_high=config.backlog_high,
            )
        elif config.family_schedule == "tiered_rolling" and rate != 0:
            raise ValueError("tiered rolling requires an unlimited arrival rate")
        self.max_input_tokens, self.seed_input_tokens = (
            max_input_tokens,
            seed_input_tokens,
        )
        self.make_payload, self.request, self.poll, self.sink = (
            make_payload,
            request,
            poll,
            sink,
        )
        self.controller = controller
        self.diagnostics_control = diagnostics_control
        self.control_jobs = []
        self.clock, self.sleep, self.executor_factory = clock, sleep, executor_factory
        self.phase_concurrency = concurrency
        sender_limit = config.max_outstanding if self.adaptive else concurrency
        self.sender = BoundedSender(sender_limit)
        self.lock = threading.RLock()
        self.wakeup = threading.Event()
        self.jobs = {}
        self.job_id = self.sequence = self.formal_submitted = 0
        self.job_kinds = {}
        self.cancel_requested = set()
        self.seed_submitted = 0
        self.seed_completed = 0
        self.seed_failed = 0
        self.seed_cancelled = 0
        self.max_seed_pending = 0
        self.seed_started = self.seed_finished = None
        self.seed_deadline = float("inf")
        self.seed_stop_requested = False
        self.seed_stop_snapshot = ()
        self.fill_sent = 0
        self.fill_headroom_reached = False
        self.fill_headroom_snapshot = ()
        self.closed = False
        self.supply_deadline = float("inf")
        self.profile_started = False
        self.profile_pending = False
        self.profile_stopped = False
        self.support_started = None
        self.last_poll_s = float("-inf")
        self.poll_future = None
        self.next_poll = float("inf")
        self.last_progress = self.last_dispatch = clock()
        self.last_execution_progress = clock()
        self.progress = None
        self.ready = False
        self.supply_stats = dict(
            max_outstanding=0,
            rate_count=0,
            rate_sum=0.0,
            rate_min=None,
            rate_max=None,
            backlog_regions=dict(low=0, interpolated=0, high=0),
            saturated=False,
        )

    def emit(self, event, **values):
        with self.lock:
            record = dict(event=event, session_id=self.session_id, **values)
            self.sink(record)
            if event == "phase":
                self.last_emitted_phase = values["phase"]
            if self.controller and hasattr(self.controller, "consume"):
                self.controller.consume(record)
            if event == "phase" and self.diagnostics_control:
                if len(self.control_jobs) >= 16:
                    raise RuntimeError("diagnostic control queue full")
                self.control_jobs.append(
                    (
                        "phase",
                        values["phase"],
                        self.control_executor.submit(
                            self.diagnostics_control,
                            "phase",
                            values["phase"],
                            self.config.control_timeout_s,
                        ),
                    )
                )

    def _control_result(self, action, phase, future, timeout=0):
        try:
            result = future.result(timeout=timeout)
            self.emit(
                "control",
                action=action,
                phase=phase,
                result=result,
                received_s=self.clock(),
            )
            if result is not None and not result.get("success", False):
                raise RuntimeError("diagnostic rank acknowledgement failed")
        except Exception as exc:
            self.emit(
                "error", action=action, phase=phase, error=repr(exc), at_s=self.clock()
            )
            self.fail("diagnostics_control_failed")
            return False
        return True

    def _collect_controls(self):
        with self.lock:
            pending = []
            for action, phase, future in self.control_jobs:
                if future.done():
                    self._control_result(action, phase, future)
                else:
                    pending.append((action, phase, future))
            self.control_jobs = pending

    def _disable_diagnostics(self):
        # Pending phase markers cannot delay cleanup behind a growing work queue.
        for action, phase, future in self.control_jobs:
            if future.cancel():
                self.emit("control_cancelled", action=action, phase=phase)
            else:
                self._control_result(
                    action,
                    phase,
                    future,
                    timeout=self.config.control_timeout_s + 0.1,
                )
        self.control_jobs.clear()
        future = self.control_executor.submit(
            self.diagnostics_control,
            "disable",
            "",
            self.config.control_timeout_s,
        )
        self._control_result(
            "disable",
            "",
            future,
            timeout=self.config.control_timeout_s + 0.1,
        )

    def _final_steps(self):
        deadline = self.clock() + 2.0
        try:
            if self.poll_future is not None:
                evidence = self.poll_future.result(timeout=1.0)
                self.poll_future = None
                if not self._record_pressure(evidence):
                    raise RuntimeError(evidence.raw["diagnostics_error"])
            # One extra empty page handles an exactly full final page.
            for _ in range(17):
                remaining = deadline - self.clock()
                if remaining <= 0:
                    break
                future = self.poll_executor.submit(self.poll, min(1.0, remaining))
                evidence = future.result(timeout=min(1.1, remaining))
                if not self._record_pressure(evidence):
                    raise RuntimeError(evidence.raw["diagnostics_error"])
                if evidence.step_history_complete and not evidence.step_page_full:
                    return
            raise RuntimeError("final step drain limit reached")
        except Exception as exc:
            self.emit("error", action="final_steps", error=repr(exc), at_s=self.clock())
            self.fail("diagnostics_drain_failed")

    def fail(self, reason):
        with self.lock:
            if self.state.phase != "drain":
                self.supply_deadline = min(
                    self.supply_deadline, self.clock() + self.config.drain_timeout_s
                )
            self.state.fail(reason)

    def _advance(self):
        with self.lock:
            now = self.clock()
            previous = self.state.phase
            if self.state.phase == "seed" and now >= self.seed_deadline:
                self.fail("seed_timeout")
            if (
                self.profile_started
                and self.profile_pending
                and now >= self.state.measurement_start + self.config.capture_timeout_s
            ):
                self.state.fail("profile_timeout")
            covered = 0.0
            if self.state.phase == "steady" and now > self.state.entered:
                intervals = self.coverage.intervals(self.state.entered, now)
                if (
                    intervals
                    and self.ready
                    and now - intervals[-1][1] <= self.config.max_sample_age_s
                ):
                    covered = intervals[-1][1] - intervals[-1][0]
            self.state.advance(
                now, steady_covered_s=covered, profile_pending=self.profile_pending
            )
            if self.state.phase == "support" and self.support_started is None:
                self.support_started = self.state.measurement_end
            if self.state.phase != previous:
                if self.state.phase == "drain":
                    self.supply_deadline = min(
                        self.supply_deadline, now + self.config.drain_timeout_s
                    )
                self.emit("phase", phase=self.state.phase, at_s=now)

    def _before(self, job_id, kind, scheduled, payload, dispatch_context):
        # Serialize phase classification and identity allocation at the transport
        # boundary, after token generation. Workers never wait for arrival times.
        with self.lock:
            self._advance()
            phase = self.state.phase if kind == "formal" else kind
            if (
                self.closed
                or self.clock() >= self.supply_deadline
                or self.state.phase == "drain"
                or (kind == "seed" and self.state.phase != "seed")
                or (
                    kind == "formal"
                    and phase not in ("steady", "measurement", "support")
                )
            ):
                raise DispatchClosed()
            ids = payload["input_ids"]
            expected = (
                self.config.fill_input_tokens
                if kind == "fill"
                else self.seed_input_tokens if kind == "seed" else self.max_input_tokens
            )
            if len(ids) > expected or (
                kind in ("seed", "fill") and len(ids) != expected
            ):
                raise ValueError("generated input violates frozen token bound")
            self.limits.concurrency_for(len(ids))
            input_sha256 = hashlib.sha256(
                struct.pack(f"<{len(ids)}i", *ids)
            ).hexdigest()
            metadata = {k: v for k, v in payload.items() if k != "input_ids"}
            # Validation and hashing can consume the remaining seed budget.
            # Prepare them before allocating or publishing a dispatch record.
            if kind == "seed" and self.clock() >= self.seed_deadline:
                self.fail("seed_timeout")
                raise DispatchClosed()
            now = self.clock()
            sequence = self.sequence
            self.sequence += 1
            identity = f"{self.session_id}:{phase}:{sequence}"
            record = dict(
                job_id=job_id,
                kind=kind,
                sequence=sequence,
                phase=phase,
                request_id=identity,
                dispatched_s=now,
                scheduled_offset_s=scheduled,
                input_len=len(ids),
                input_sha256=input_sha256,
                metadata=metadata,
            )
            record.update(dispatch_context)
            if scheduled is not None:
                record["schedule_lag_s"] = max(0, now - self.formal_origin - scheduled)
            self.jobs[job_id] = record
            self.last_dispatch = now
            if kind == "formal":
                self.formal_submitted += 1
            self.emit("dispatch", **record)
            # A durable sink/controller can also cross the deadline. The caller
            # may start transport only after this final permission check. If a
            # record was already published, collection retains not_dispatched.
            if kind == "seed" and self.clock() >= self.seed_deadline:
                self.fail("seed_timeout")
                raise DispatchClosed()
            return identity

    def _submit(self, kind, scheduled=None, formal_sequence=None, **dispatch_context):
        job_id = self.job_id
        if formal_sequence is not None:
            dispatch_context = dict(
                dispatch_context, formal_sequence=formal_sequence
            )

        def work(register):
            try:
                payload = self.make_payload(kind, job_id, formal_sequence)
                result = self.request(
                    payload,
                    register,
                    lambda p: self._before(
                        job_id, kind, scheduled, p, dispatch_context
                    ),
                )
                return dict(result, _pressure_completed_s=self.clock())
            except DispatchClosed:
                return {"not_dispatched": True}

        accepted = self.sender.submit(
            job_id,
            lambda: CancellableRequestHandle(
                self.executor, work, on_done=self.wakeup.set
            ),
        )
        if accepted:
            self.job_kinds[job_id] = kind
            self.job_id += 1
        return accepted

    def _collect(self):
        for job_id, result in self.sender.collect():
            kind = self.job_kinds.pop(job_id, None)
            cancelled = job_id in self.cancel_requested
            self.cancel_requested.discard(job_id)
            with self.lock:
                record = self.jobs.pop(job_id, None)
            if record is None:
                if kind == "seed" and not cancelled:
                    self.seed_failed += 1
                    self.fail("request_failed")
                if not result.get("not_dispatched"):
                    self.fail("request_failed")
                    self.emit(
                        "completion",
                        job_id=job_id,
                        completed_s=self.clock(),
                        result=result,
                    )
                continue
            result = dict(result)
            now = result.pop("_pressure_completed_s", self.clock())
            self.emit(
                "completion",
                **record,
                completed_s=now,
                result=result,
                crossing=record["phase"] == "measurement"
                and now >= self.state.measurement_end,
            )
            if result.get("success"):
                self.last_execution_progress = max(self.last_execution_progress, now)
            success = (
                result.get("success")
                and result.get("input_len") == record["input_len"]
                and result.get("output_len") == 1
            )
            if kind == "seed":
                if success:
                    self.seed_completed += 1
                elif not cancelled:
                    self.seed_failed += 1
            if not success and not cancelled:
                self.fail("request_failed")

    def _consume_poll(self, evidence):
        with self.lock:
            self._apply_poll(evidence)

    def _record_pressure(self, evidence):
        self.emit("pressure", at_s=self.clock(), raw=evidence.raw)
        error = evidence.raw.get("diagnostics_error")
        if error:
            self.emit(
                "error", action="diagnostics_poll", error=error, at_s=self.clock()
            )
        return not error

    def _apply_poll(self, evidence):
        now = self.clock()
        self.last_poll_s = min(
            (s.client_receive_s for s in evidence.samples), default=now
        )
        if not self._record_pressure(evidence):
            self.fail("diagnostics_poll_failed")
            return
        self.coverage.add(evidence.samples, now, complete=evidence.sample_complete)
        if self.config.family_schedule == "tiered_rolling":
            occupancy = self._host_pool_occupancy(evidence.samples)
            at_limit = any(
                item["ratio"] >= self.config.fill_host_max_ratio
                for item in occupancy
            )
            if (
                self.state.phase == "seed"
                and not self.seed_stop_requested
                and at_limit
            ):
                self.seed_stop_requested = True
                self.seed_stop_snapshot = occupancy
                self.emit(
                    "seed_limit",
                    at_s=now,
                    reason="host_max_ratio",
                    configured_limit=self.seed_count,
                    submitted=self.seed_submitted,
                    completed=self.seed_completed,
                    host_pools=list(occupancy),
                )
            elif (
                self.state.phase == "fill"
                and not self.fill_headroom_reached
                and at_limit
            ):
                self.fill_headroom_reached = True
                self.fill_headroom_snapshot = occupancy
                self.emit(
                    "fill_limit",
                    at_s=now,
                    reason="host_max_ratio",
                    sent=self.fill_sent,
                    host_pools=list(occupancy),
                )
        self.ready = False
        if not self.coverage.epoch_valid:
            self.fail("pressure_epoch_invalid")
            return
        if evidence.sample_complete and len(evidence.samples) == len(
            self.coverage.ranks
        ):
            try:
                validate_inventory(
                    select_targets(evidence.samples, self.coverage.ranks),
                    self.coverage.targets,
                )
            except ValueError:
                self.coverage.epoch_valid = False
                self.fail("pressure_epoch_invalid")
                return
            try:
                self.ready = all(
                    self.coverage.target_valid(t, evidence.samples)
                    for t in self.coverage.targets
                )
                self.ready = self.ready and all(
                    0
                    <= now
                    - (
                        s.server_s
                        + (s.offset_bounds or (s.client_send_s - s.server_s,))[0]
                    )
                    <= self.config.max_sample_age_s
                    for s in evidence.samples
                )
            except (ValueError, KeyError, StopIteration):
                self.ready = False
        if not self.ready and self.state.phase in ("measurement", "support"):
            self.emit("pressure_lost", phase=self.state.phase, at_s=now)
        progress = tuple(
            (
                s.rank,
                tuple((g.group_id, g.triggers, g.gpu_host_published) for g in s.groups),
                tuple(
                    (p.pool_id, p.used - p.pending_release, p.host_resident)
                    for p in s.pools
                ),
            )
            for s in evidence.samples
        )
        if progress and progress != self.progress:
            self.last_progress, self.progress = now, progress
        if evidence.activity:
            for activity in evidence.activity:
                activity = type(activity)(
                    self.last_dispatch,
                    max(
                        self.last_progress,
                        self.last_execution_progress,
                        activity.last_progress_s,
                    ),
                    activity.active_step_started_s,
                    activity.running,
                    activity.waiting,
                )
                assessment = assess_activity(
                    activity, now, self.config.activity_timeout_s
                )
                if not assessment.valid:
                    self.fail(assessment.reasons[0])
        elif now - self.last_progress >= self.config.activity_timeout_s:
            self.fail("unknown_activity")

    @staticmethod
    def _host_pool_occupancy(samples):
        occupancy = []
        for sample in samples:
            pools = {pool.pool_id: pool for pool in sample.pools}
            for group in sample.groups:
                pool = pools.get(group.host_pool_id)
                if pool is None or pool.capacity <= 0:
                    continue
                resident = max(0, pool.used - pool.pending_release)
                occupancy.append(
                    dict(
                        rank=sample.rank,
                        group_id=group.group_id,
                        pool_id=pool.pool_id,
                        used=resident,
                        capacity=pool.capacity,
                        ratio=resident / pool.capacity,
                    )
                )
        return tuple(occupancy)

    def _tick(self):
        self._collect_controls()
        self._collect()
        now = self.clock()
        if self.poll_future is not None and self.poll_future.done():
            try:
                evidence = self.poll_future.result()
            except Exception as exc:
                evidence = PressurePoll((), {"error": repr(exc)})
            self.poll_future = None
            self._consume_poll(evidence)
        if self.poll_future is None and now >= self.next_poll:
            self.next_poll = now + self.config.sample_period_s
            self.poll_future = self.poll_executor.submit(
                self.poll, self.config.sample_period_s
            )
        if self.controller and self.profile_started and not self.profile_stopped:
            self.profile_pending = self.controller.pending(now)
        self._advance()
        if (
            self.state.phase == "fill"
            and now - self.last_progress >= self.config.fill_no_progress_s
        ):
            self.fail("pressure_not_reached")
        if (
            self.state.phase == "measurement"
            and self.controller
            and not self.profile_started
        ):
            self.controller.start(self.state.measurement_start)
            self.profile_started = True
            self.profile_pending = self.controller.pending(now)

    def _pause(self, until=None):
        delay = min(0.01, self.config.sample_period_s / 10)
        if until is not None:
            delay = min(delay, max(0.0, until - self.clock()))
        if self.sleep is time.sleep:
            self.wakeup.wait(delay)
            self.wakeup.clear()
        else:
            self.sleep(delay)

    def _batch(self, kind, count):
        if self.state.phase == "drain":
            return
        for _ in range(count):
            self._submit(kind)
        while self.sender.pending and self.state.phase != "drain":
            self._tick()
            self._pause()

    def _seed_result(self):
        planned = (
            self.seed_submitted
            if self.config.family_schedule == "tiered_rolling"
            and self.seed_stop_requested
            else self.seed_count
        )
        result = dict(
            planned=planned,
            submitted=self.seed_submitted,
            completed=self.seed_completed,
            failed=self.seed_failed,
            cancelled=self.seed_cancelled,
            pending=sum(kind == "seed" for kind in self.job_kinds.values()),
            max_inflight=self.max_seed_pending,
            elapsed_s=(
                (self.seed_finished - self.seed_started)
                if self.seed_started is not None and self.seed_finished is not None
                else 0.0
            ),
        )
        if self.config.family_schedule == "tiered_rolling":
            result.update(
                configured_limit=self.seed_count,
                effective_seed_count=self.seed_completed,
                stop_reason=(
                    "host_max_ratio"
                    if self.seed_stop_requested
                    else "configured_limit"
                ),
                stop_host_pools=list(self.seed_stop_snapshot),
            )
        return result

    def _run_seed_phase(self) -> dict:
        self.seed_started = self.clock()
        self.seed_deadline = self.seed_started + self.config.seed_timeout_s
        self.state.phase, self.state.entered = "seed", self.seed_started
        if self.seed_count:
            self.emit("phase", phase="seed", at_s=self.seed_started)
        while self.seed_completed + self.seed_failed < self.seed_count:
            self._advance()
            while (
                self.state.phase == "seed"
                and not self.seed_stop_requested
                and self.seed_submitted < self.seed_count
                and sum(
                    kind == "seed" for kind in self.job_kinds.values()
                )
                < self.phase_concurrency
            ):
                self._advance()
                if self.state.phase != "seed" or not self._submit("seed"):
                    break
                self.seed_submitted += 1
                self.max_seed_pending = max(
                    self.max_seed_pending, len(self.sender.pending)
                )
            self._tick()
            if self.state.phase == "drain":
                break
            if self.seed_stop_requested and not self.sender.pending:
                break
            if self.seed_completed == self.seed_count:
                break
            self._pause()
        self.seed_finished = self.clock()
        return self._seed_result()

    def _start_fill(self):
        # Rebase after the last seed completion. A pre-existing asynchronous poll
        # may have sampled seed transfers; consume it before taking this snapshot.
        if self.seed_count:
            if self.poll_future is not None:
                evidence = self.poll_future.result(timeout=self.config.sample_period_s)
                self.poll_future = None
                self._consume_poll(evidence)
            baseline = self.poll(self.config.sample_period_s)
            self._consume_poll(baseline)
            if self.state.phase == "drain":
                return
            if not baseline.sample_complete:
                raise RuntimeError("post-seed pressure inventory is incomplete")
            self.coverage = PressureCoverage(tuple(baseline.samples), self.config)
            self.emit("fill_baseline", at_s=self.clock(), raw_baseline=baseline.raw)
        now = self.clock()
        self.state.phase, self.state.entered = "fill", now
        self.last_progress = self.last_execution_progress = now
        self.progress, self.ready = None, False
        self.fill_sent = 0
        self.emit("phase", phase="fill", at_s=now)

    def _run_formal_supply(self):
        self.formal_origin = self.clock()
        if self.adaptive:
            self._run_adaptive_formal_supply()
            return
        due = self.arrivals.next_offset()
        formal_sequence = 0
        while self.state.phase in ("steady", "measurement", "support"):
            self._tick()
            if self.state.phase == "drain":
                break
            while (
                self.state.phase in ("steady", "measurement", "support")
                and self.clock() + 1e-8 >= self.formal_origin + due
            ):
                if not self._submit("formal", due, formal_sequence):
                    break
                formal_sequence += 1
                due = self.arrivals.next_offset()
            if self.state.phase != "drain":
                self._pause()

    def _record_adaptive_dispatch(self, outstanding, rate):
        stats = self.supply_stats
        stats["max_outstanding"] = max(stats["max_outstanding"], outstanding + 1)
        stats["rate_count"] += 1
        stats["rate_sum"] += rate
        stats["rate_min"] = rate if stats["rate_min"] is None else min(
            stats["rate_min"], rate
        )
        stats["rate_max"] = rate if stats["rate_max"] is None else max(
            stats["rate_max"], rate
        )
        region = (
            "low"
            if outstanding <= self.config.backlog_low
            else "high"
            if outstanding >= self.config.backlog_high
            else "interpolated"
        )
        stats["backlog_regions"][region] += 1

    def _run_adaptive_formal_supply(self):
        due = 0.0
        formal_sequence = 0
        while self.state.phase in ("steady", "measurement", "support"):
            self._tick()
            if self.state.phase == "drain":
                break
            while (
                self.state.phase in ("steady", "measurement", "support")
                and self.clock() + 1e-8 >= self.formal_origin + due
            ):
                outstanding = len(self.sender.pending)
                rate = self.adaptive.rate(outstanding)
                if outstanding >= self.sender.limit:
                    self.supply_stats["saturated"] = True
                    self.emit(
                        "sender_saturated",
                        at_s=self.clock(),
                        formal_sequence=formal_sequence,
                        scheduled_offset_s=due,
                        outstanding=outstanding,
                        max_outstanding=self.sender.limit,
                        effective_rate=rate,
                    )
                    self.fail("sender_saturated")
                    break
                if not self._submit(
                    "formal",
                    due,
                    formal_sequence,
                    outstanding_before=outstanding,
                    effective_rate=rate,
                ):
                    raise RuntimeError("adaptive sender rejected below hard limit")
                self._record_adaptive_dispatch(outstanding, rate)
                formal_sequence += 1
                due += 1.0 / rate
            if self.state.phase != "drain":
                self._pause(self.formal_origin + due)

    def _supply_result(self):
        stats = self.supply_stats
        return dict(
            max_outstanding=stats["max_outstanding"],
            configured_max_outstanding=self.sender.limit,
            rate_min=stats["rate_min"],
            rate_max=stats["rate_max"],
            rate_average=(
                stats["rate_sum"] / stats["rate_count"]
                if stats["rate_count"]
                else None
            ),
            rate_count=stats["rate_count"],
            backlog_regions=dict(stats["backlog_regions"]),
            saturated=stats["saturated"],
        )

    def run(self):
        self.state = PressurePhases(self.config, self.clock())
        self.executor = self.executor_factory(
            max_workers=self.sender.limit, thread_name_prefix="pressure-request"
        )
        self.poll_executor = self.executor_factory(
            max_workers=1, thread_name_prefix="pressure-poll"
        )
        if self.diagnostics_control:
            self.control_executor = self.executor_factory(
                max_workers=1, thread_name_prefix="pressure-control"
            )
        try:
            if self.diagnostics_control:
                future = self.control_executor.submit(
                    self.diagnostics_control,
                    "enable",
                    "",
                    self.config.control_timeout_s,
                )
                if not self._control_result(
                    "enable",
                    "",
                    future,
                    timeout=self.config.control_timeout_s + 0.1,
                ):
                    raise RuntimeError("diagnostic enable failed")
            baseline = self.poll(self.config.sample_period_s)
            try:
                if baseline.raw.get("diagnostics_error"):
                    raise RuntimeError(baseline.raw["diagnostics_error"])
                if not baseline.sample_complete:
                    raise RuntimeError("initial pressure inventory is incomplete")
                validate_inventory(
                    select_targets(
                        tuple(baseline.samples),
                        tuple(sorted({target.rank for target in self.startup_targets})),
                    ),
                    self.startup_targets,
                )
                self.coverage = PressureCoverage(tuple(baseline.samples), self.config)
                # Validate reachability before sending any requests.
                for sample in baseline.samples:
                    for group in sample.groups:
                        if group.group_type == "FULL":
                            group_valid(
                                group,
                                {p.pool_id: p for p in sample.pools},
                                group,
                                self.config,
                            )
            except Exception:
                self._record_pressure(baseline)
                raise
            self.emit(
                "frozen",
                config=asdict(self.config),
                limits=asdict(self.limits),
                seed_plan=asdict(self.seed_plan),
                targets=[asdict(t) for t in self.coverage.targets],
                raw_baseline=baseline.raw,
            )
            self._record_pressure(baseline)
            self.next_poll = self.clock()
            seed = self._run_seed_phase()
            if (
                self.state.phase != "drain"
                and seed["completed"] == seed["planned"]
                and seed["failed"] == 0
                and seed["pending"] == 0
            ):
                self._start_fill()
            elif self.state.phase != "drain":
                self.fail("seed_incomplete")
            while self.state.phase == "fill":
                self._tick()
                if self.state.phase == "drain":
                    break
                if self.fill_headroom_reached:
                    self.fail("fill_headroom_exhausted")
                    break
                if self.fill_sent and self.ready and not self.sender.pending:
                    self.state.advance(self.clock(), fill_ready=True)
                    self.emit("phase", phase="steady", at_s=self.clock())
                    break
                if (
                    self.fill_sent >= self.config.fill_max_requests
                    or self.clock() - self.last_progress
                    >= self.config.fill_no_progress_s
                ):
                    self.fail("pressure_not_reached")
                    break
                count = min(
                    self.fill_batch, self.config.fill_max_requests - self.fill_sent
                )
                self._batch("fill", count)
                self.fill_sent += count
                completed = self.clock()
                while self.last_poll_s < completed and self.state.phase != "drain":
                    self._tick()
                    if self.state.phase != "drain":
                        self._pause()
            if self.state.phase == "steady":
                self._run_formal_supply()
        except Exception as exc:
            self.fail("supervisor_failed")
            self.emit("error", error=repr(exc), at_s=self.clock())
        finally:
            try:
                self._drain()
            finally:
                if self.diagnostics_control:
                    try:
                        self._disable_diagnostics()
                        self._final_steps()
                    finally:
                        self.control_executor.shutdown(wait=False, cancel_futures=True)
                self.executor.shutdown(wait=False, cancel_futures=True)
                self.poll_executor.shutdown(wait=False, cancel_futures=True)
        profile_result = None
        if self.controller and hasattr(self.controller, "finish"):
            try:
                profile_result = self.controller.finish(getattr(self, "coverage", None))
            except Exception as exc:
                profile_result = dict(
                    trace_valid=False, reasons=[repr(exc)], pressure_valid=False
                )
            if not profile_result.get("trace_valid"):
                self.state.fail("profile_invalid")
        seed = self._seed_result()
        self.emit("seed_summary", **seed)
        result = dict(
            seed=seed,
            failure=self.state.failure,
            measurement_start=self.state.measurement_start,
            measurement_end=self.state.measurement_end,
            supply=self._supply_result(),
        )
        if profile_result is not None:
            result["profile"] = profile_result
        if self.state.measurement_end is not None:
            result["pressure"] = asdict(
                self.coverage.assess(
                    self.state.measurement_start, self.state.measurement_end
                )
            )
            result["migration"] = asdict(
                self.coverage.assess_migration(
                    self.state.measurement_start, self.state.measurement_end
                )
            )
        if self.support_started is not None and self.support_end > self.support_started:
            result["support_pressure"] = asdict(
                self.coverage.assess(self.support_started, self.support_end)
            )
        self.emit("finished", **result)
        return result

    def _drain(self):
        self.state.phase = "drain"
        started = self.clock()
        if getattr(self, "last_emitted_phase", None) != "drain":
            self.emit("phase", phase="drain", at_s=started)
        self.support_end = started
        deadline = min(self.supply_deadline, started + self.config.drain_timeout_s)
        self.supply_deadline = deadline
        if self.controller and not self.profile_stopped:
            self.profile_stopped = True
            try:
                self.controller.stop(self.state.failure or "complete", started)
            except Exception as exc:
                self.fail("profile_stop_failed")
                self.emit("error", error=repr(exc), at_s=self.clock())
        # A failed seed cannot make progress into another phase. Request
        # cancellation immediately, then use the bounded drain to reap it.
        if self.state.failure and (
            self.state.failure == "sender_saturated"
            or any(kind == "seed" for kind in self.job_kinds.values())
        ):
            self._cancel_pending()
        while self.clock() < deadline:
            self._collect()
            if hasattr(self, "coverage"):
                self._tick()
            edge = (
                self.support_end
                if self.support_started is not None
                else self.state.measurement_end
            )
            edge_pending = edge is not None and self.last_poll_s <= edge
            if not self.sender.pending and not edge_pending:
                break
            self._pause()
        with self.lock:
            self.closed = True
        if self.sender.pending:
            self.fail("drain_timeout")
            self._cancel_pending()
            self._collect()
        if self.poll_future is not None:
            self.poll_future.cancel()

    def _cancel_pending(self):
        for job_id, handle in list(self.sender.pending.items()):
            if job_id in self.cancel_requested:
                continue
            self.cancel_requested.add(job_id)
            if self.job_kinds.get(job_id) == "seed":
                self.seed_cancelled += 1
            error = None
            try:
                accepted = handle.cancel()
            except Exception as exc:
                accepted, error = False, repr(exc)
            self.emit(
                "cancellation",
                job_id=job_id,
                at_s=self.clock(),
                accepted=accepted,
                error=error,
                dispatch=self.jobs.get(job_id),
            )

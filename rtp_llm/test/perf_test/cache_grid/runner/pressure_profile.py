"""Independent, bounded Kineto control and conservative workload validation."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import asdict
from pathlib import Path

from .pressure_nsys import classify_gpu_gap, normalize_nsys


def run_nsys_window(command, *, duration_s, cancelled):
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("positive finite Nsight duration required")
    try:
        command("start")
        cancelled.wait(duration_s)
    except BaseException:
        try:
            command("stop")
        except BaseException:
            # The production command adapter records both failures.
            pass
        raise
    else:
        command("stop")


class ProfileController:
    """Operations must honor their transport deadlines/cancellation event.

    Daemon isolation also bounds the supervisor when an injected operation violates
    that contract. Python cannot forcibly terminate such an operation safely.
    """

    def __init__(
        self,
        start_capture,
        wait_capture,
        stop_capture,
        *,
        start_timeout_s=5,
        wait_timeout_s=120,
        stop_timeout_s=2,
    ):
        self.operations = dict(
            start=start_capture, wait=wait_capture, stop=stop_capture
        )
        self.timeouts = dict(
            start=start_timeout_s, wait=wait_timeout_s, stop=stop_timeout_s
        )
        if any(not math.isfinite(v) or v <= 0 for v in self.timeouts.values()):
            raise ValueError("profile operation deadlines must be finite and positive")
        self.cancel = threading.Event()
        self.log = []
        self.lock = threading.RLock()
        self.future = Future()
        self.stop_future = None
        self.started = False
        self.start_settled = threading.Event()
        self.stopped_during_start = False

    def _call(self, operation):
        future = Future()
        entry = dict(operation=operation, started_s=time.monotonic())
        with self.lock:
            self.log.append(entry)

        def invoke():
            try:
                value = self.operations[operation]()
                future.set_result(value)
            except BaseException as exc:
                future.set_exception(exc)

        threading.Thread(
            target=invoke, name="pressure-profile-" + operation, daemon=True
        ).start()
        try:
            value = future.result(self.timeouts[operation])
            entry.update(acknowledged_s=time.monotonic())
            if operation != "wait":
                entry["result"] = value
            return value
        except BaseException as exc:
            entry.update(failed_s=time.monotonic(), error=repr(exc))
            raise

    def start(self):
        with self.lock:
            if self.started:
                raise RuntimeError("profile controller already started")
            self.started = True

        def run():
            result = {}
            errors = []
            try:
                try:
                    if self.cancel.is_set():
                        raise RuntimeError("profile_cancelled_before_start")
                    self._call("start")
                finally:
                    self.start_settled.set()
                if self.cancel.is_set():
                    raise RuntimeError("profile_cancelled_during_start")
                result = self._call("wait") or {}
            except BaseException as exc:
                errors.append(repr(exc))
            finally:
                self.request_stop()
                try:
                    self.stop_future.result(self.timeouts["stop"] + 0.1)
                except BaseException as exc:
                    errors.append(repr(exc))
                if self.stopped_during_start:
                    try:
                        self._call("stop")
                    except BaseException as exc:
                        errors.append(repr(exc))
            if errors:
                result.update(trace_valid=False, control_errors=errors)
            result["control_log"] = list(self.log)
            self.future.set_result(result)

        threading.Thread(
            target=run, name="pressure-profile-controller", daemon=True
        ).start()

    def request_stop(self):
        self.cancel.set()
        with self.lock:
            if self.stop_future is not None:
                return
            self.stopped_during_start = not self.start_settled.is_set()
            self.stop_future = Future()

        def stop():
            try:
                self.stop_future.set_result(self._call("stop"))
            except BaseException as exc:
                self.stop_future.set_exception(exc)

        threading.Thread(
            target=stop, name="pressure-profile-cleanup", daemon=True
        ).start()

    def done(self):
        return self.future.done()

    def result(self, timeout_s):
        if not math.isfinite(timeout_s) or timeout_s < 0:
            raise ValueError("invalid profile result deadline")
        try:
            return self.future.result(timeout_s)
        except FutureTimeoutError as exc:
            self.request_stop()
            raise TimeoutError("profile result deadline expired") from exc


def _range(event):
    start, duration = event.get("ts"), event.get("dur")
    if (
        event.get("ph") != "X"
        or not isinstance(start, (int, float))
        or not isinstance(duration, (int, float))
        or not math.isfinite(start)
        or not math.isfinite(duration)
        or duration <= 0
    ):
        return None
    return start, start + duration


def _contains(outer, inner):
    a, b = _range(outer), _range(inner)
    return (
        a is not None
        and b is not None
        and outer.get("pid") == inner.get("pid")
        and outer.get("tid") == inner.get("tid")
        and a[0] <= b[0]
        and b[1] <= a[1]
    )


def validate_kineto(trace, steps, requested_steps):
    """Join observed CPU markers, forward scopes, launches and CUDA kernels.

    Only the installed Kineto schema's cpu_op/user_annotation, cuda_runtime/
    cuda_driver and kernel categories are accepted. GPU pid/tid are not OS IDs.
    Sidecar timestamps are never compared to Kineto timestamps.
    """
    events = trace.get("traceEvents", []) if isinstance(trace, dict) else []
    if (
        not isinstance(trace, dict)
        or not isinstance(events, list)
        or any(not isinstance(e, dict) for e in events)
    ):
        return dict(trace_valid=False, reasons=["invalid_trace_schema"])
    reasons = []
    if trace.get("schemaVersion", 1) != 1:
        reasons.append("unsupported_trace_schema")
    if any(not isinstance(e.get("args", {}), dict) for e in events):
        return dict(trace_valid=False, reasons=["invalid_trace_arguments"])
    cpu = [
        e for e in events if e.get("cat") in ("cpu_op", "user_annotation") and _range(e)
    ]
    forwards = [
        e for e in cpu if str(e.get("name", "")).startswith("executor.model_forward")
    ]
    kernels = [e for e in events if e.get("cat") == "kernel" and _range(e)]
    runtimes = [
        e
        for e in events
        if e.get("cat") in ("cuda_runtime", "cuda_driver") and _range(e)
    ]
    if not kernels:
        reasons.append("missing_gpu_kernels")
    if not forwards:
        reasons.append("missing_cpu_forward_ranges")
    markers = {}
    for event in cpu:
        name = str(event.get("name", ""))
        if not name.startswith("pressure.step/"):
            continue
        try:
            step_id = int(name.split("/", 1)[1])
            if step_id <= 0 or step_id in markers:
                raise ValueError()
            markers[step_id] = event
        except ValueError:
            reasons.append("invalid_step_identity")
    if not markers:
        reasons.append("missing_step_markers")
    sidecar = {int(s["step_id"]): s for s in steps}
    matched, linked = {}, set()
    for step_id, marker in markers.items():
        scopes = [f for f in forwards if _contains(marker, f) or _contains(f, marker)]
        if len(scopes) != 1:
            reasons.append("ambiguous_step_forward")
            continue
        forward = scopes[0]
        calls = [r for r in runtimes if _contains(forward, r) and _contains(marker, r)]
        correlations = {r.get("args", {}).get("correlation") for r in calls}
        correlations.discard(None)
        work = [
            k for k in kernels if k.get("args", {}).get("correlation") in correlations
        ]
        linked.update(id(k) for k in work)
        if not work:
            reasons.append("missing_cuda_correlation")
        if step_id not in sidecar:
            reasons.append("missing_step_sidecar")
            continue
        matched[step_id] = (marker, forward, work, sidecar[step_id])
    non_forward_correlated = 0
    if forwards:
        low = min(_range(f)[0] for f in forwards)
        high = max(
            [_range(f)[1] for f in forwards]
            + [_range(k)[1] for k in kernels if id(k) in linked]
        )
        runtime_by_correlation = {}
        for call in runtimes:
            correlation = call.get("args", {}).get("correlation")
            if correlation is not None:
                runtime_by_correlation.setdefault(correlation, []).append(call)
        for kernel in kernels:
            if id(kernel) in linked or not low <= _range(kernel)[0] <= high:
                continue
            calls = runtime_by_correlation.get(
                kernel.get("args", {}).get("correlation"), []
            )
            if len(calls) != 1:
                reasons.append("missing_cuda_correlation")
                continue
            call = calls[0]
            overlaps = any(
                scope.get("pid") == call.get("pid")
                and scope.get("tid") == call.get("tid")
                and _range(scope)[0] < _range(call)[1]
                and _range(call)[0] < _range(scope)[1]
                for scope in [*forwards, *markers.values()]
            )
            if overlaps:
                reasons.append("missing_cuda_correlation")
            else:
                non_forward_correlated += 1
    observed = {
        i for i, (_, _, _, s) in matched.items() if s.get("context_batch", 0) > 0
    }
    if len(observed) < requested_steps:
        reasons.append("insufficient_observed_steps")
    usable, excluded = set(), {}
    first = min(markers, key=lambda i: _range(markers[i])[0]) if markers else None
    ends = [
        e["ts"]
        for e in events
        if e.get("name") == "Record Window End"
        and isinstance(e.get("ts"), (int, float))
    ]
    if len(ends) != 1 or not math.isfinite(ends[0]):
        reasons.append("missing_capture_trace_boundary")
    gathers = [e for e in cpu if e.get("name") == "executor.gather_model_input"]
    for step_id, (marker, forward, work, step) in matched.items():
        exclusion = None
        if not step.get("complete") or step.get("profile_boundary"):
            exclusion = "incomplete_or_boundary"
        elif not step.get("context_batch", 0) or step.get("decode_batch", 0):
            exclusion = "not_pure_prefill"
        elif step_id == first and not any(
            g.get("pid") == forward.get("pid")
            and g.get("tid") == forward.get("tid")
            and _range(g)[1] <= _range(forward)[0]
            for g in gathers
        ):
            exclusion = "first_step_gather_omitted"
        elif ends and max(
            [_range(marker)[1], _range(forward)[1]] + [_range(k)[1] for k in work]
        ) >= min(ends):
            exclusion = "flush_boundary"
        elif not work:
            exclusion = "missing_cuda_correlation"
        if exclusion:
            excluded[str(step_id)] = exclusion
        else:
            usable.add(step_id)
    pairs = sum(i + 1 in usable for i in usable)
    if not pairs:
        reasons.append("no_complete_internal_step_pair")
    return dict(
        trace_valid=not reasons,
        reasons=sorted(set(reasons)),
        observed_steps=len(observed),
        usable_steps=len(usable),
        usable_pairs=pairs,
        observed_step_ids=sorted(observed),
        usable_step_ids=sorted(usable),
        exclusions=excluded,
        non_forward_correlated=non_forward_correlated,
    )


def _analyze_profile(request, deadline):
    """Exec a fresh interpreter; timeout kills and reaps it before returning."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return dict(ranks={}, reasons=["profile_analysis_timeout"])
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(sys.path)
    try:
        completed = subprocess.run(
            [sys.executable, "-m", __package__ + ".pressure_profile_worker"],
            input=json.dumps(request),
            text=True,
            capture_output=True,
            timeout=remaining,
            check=True,
            env=environment,
        )
        return json.loads(completed.stdout)
    except subprocess.TimeoutExpired:
        return dict(ranks={}, reasons=["profile_analysis_timeout"])
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        return dict(ranks={}, reasons=["profile_analysis_failed"], error=repr(exc))


def analyze_kineto_files(paths, steps, requested_steps, deadline):
    return _analyze_profile(
        dict(
            backend="kineto", paths=paths, steps=steps, requested_steps=requested_steps
        ),
        deadline,
    )


def analyze_nsys_file(path, steps, output, deadline):
    return _analyze_profile(
        dict(backend="nsys", path=str(path), steps=steps, output=str(output)), deadline
    )


class KinetoCapture:
    """PressureSession adapter. Supervisor hooks only copy bounded evidence.

    Transport callbacks are injected with explicit deadlines. Online readiness
    never parses traces; isolated final validation follows step drain.
    """

    def __init__(
        self,
        *,
        start_capture,
        read_status,
        stop_capture,
        trace_name,
        directory,
        ranks,
        instances,
        session_id,
        timeout_s,
        requested_steps=20,
        state_capacity=4096,
    ):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", trace_name):
            raise ValueError("invalid continuous trace name")
        self.trace_name, self.directory = trace_name, Path(directory)
        self.ranks, self.instances = tuple(ranks), dict(instances)
        self.session_id, self.requested_steps = session_id, requested_steps
        self.read_status, self.timeout_s = read_status, timeout_s
        self.steps = {rank: OrderedDict() for rank in ranks}
        self.offsets, self.statuses = {}, {}
        self.state_capacity, self.evictions = state_capacity, 0
        self.errors = set()
        self.started = False
        self.controller = ProfileController(
            start_capture,
            self._wait,
            stop_capture,
            start_timeout_s=5,
            wait_timeout_s=timeout_s + 2,
            stop_timeout_s=2,
        )

    def consume(self, event):
        if event.get("event") != "pressure":
            return
        raw = event.get("raw", {})
        for page in raw.get("response", {}).get("ranks", []):
            rank = int(page.get("world_rank", 0))
            if rank not in self.steps:
                self.errors.add("unexpected_step_rank")
                continue
            if page.get("service_instance") != self.instances[rank]:
                self.errors.add("step_instance_changed")
                continue
            bounds = raw.get("rank_offset_bounds_s", {}).get(str(rank))
            if bounds is not None:
                self.offsets[rank] = bounds
            status = page.get("profile", {})
            if status.get("trace_name") != self.trace_name:
                status = self.statuses.get(rank, {})
            begin, end = int(status.get("capture_begin_ns", 0)), int(
                status.get("capture_end_ns", 0)
            )
            for record in page.get("steps", []):
                if record.get("session_id") != self.session_id:
                    self.errors.add("step_session_changed")
                    continue
                entered = int(record.get("forward_enter_ns", 0))
                if (begin and entered < begin) or (end and entered > end):
                    continue
                step_id = int(record.get("step_id", 0))
                values = dict(
                    step_id=step_id,
                    context_batch=int(record.get("context_count", 0)),
                    decode_batch=int(record.get("decode_count", 0)),
                    complete=record.get("completed", False),
                    begin_ns=int(record.get("preparation_begin_ns", 0)),
                    forward_enter_ns=int(record.get("forward_enter_ns", 0)),
                    end_ns=int(record.get("end_ns", 0)),
                )
                old = self.steps[rank].get(step_id)
                if old and old["complete"] and not values["complete"]:
                    continue
                self.steps[rank][step_id] = values
                if len(self.steps[rank]) > self.state_capacity:
                    self.steps[rank].popitem(last=False)
                    self.evictions += 1

    def start(self, now):
        self.started = True
        self.started_s = now
        self.deadline = time.monotonic() + self.timeout_s
        for records in self.steps.values():
            records.clear()
        self.evictions = 0
        self.controller.start()

    def pending(self, now):
        if self.started and now >= self.started_s + self.timeout_s:
            self.controller.request_stop()
            return False
        return self.started and not self.controller.done()

    def stop(self, reason, now):
        if self.started:
            self.controller.request_stop()

    def _wait(self):
        deadline = self.deadline
        cancel_deadline = None
        paths, last_error = {}, None
        while time.monotonic() < deadline:
            if self.controller.cancel.is_set():
                if cancel_deadline is None:
                    cancel_deadline = time.monotonic() + 2
                if time.monotonic() >= cancel_deadline:
                    break
            try:
                raw = self.read_status(min(1, max(0.01, deadline - time.monotonic())))
                pages = raw.get("response", {}).get("ranks", [])
                actual = [int(page.get("world_rank", 0)) for page in pages]
                if sorted(actual) != sorted(self.ranks):
                    raise ValueError("missing_or_duplicate_profile_rank")
                for page in pages:
                    rank = int(page.get("world_rank", 0))
                    status = page.get("profile", {})
                    self.statuses[rank] = status
                    if (
                        page.get("service_instance") != self.instances[rank]
                        or status.get("trace_name") != self.trace_name
                    ):
                        raise ValueError("profile_identity_mismatch")
                    if page.get("error") or not status.get("supported"):
                        raise ValueError("profile_rank_error")
                    if status.get("collection_error") or status.get("export_error"):
                        raise ValueError("profile_collection_or_export_error")
                    if not (
                        status.get("collection_complete")
                        and status.get("export_complete")
                        and status.get("state") == "finished"
                    ):
                        continue
                    path = Path(status.get("file_name", ""))
                    if not path.is_absolute():
                        path = self.directory / path
                    if (
                        not path.name.startswith(f"{self.trace_name}_wr{rank}_")
                        or path.suffix != ".json"
                    ):
                        raise ValueError("profile_artifact_identity_mismatch")
                    if not path.is_file() or path.stat().st_size == 0:
                        raise OSError("profile_artifact_not_ready")
                    paths[rank] = str(path)
                if len(paths) == len(self.ranks):
                    return dict(trace_files=paths)
            except (OSError, ValueError) as exc:
                last_error = repr(exc)
                if isinstance(exc, ValueError):
                    return dict(
                        trace_valid=False,
                        reasons=[str(exc)],
                        trace_files=paths,
                    )
            except Exception as exc:
                last_error = repr(exc)
            time.sleep(0.05)
        return dict(
            trace_valid=False,
            reasons=["profile_trace_timeout"],
            last_error=last_error,
            trace_files=paths,
        )

    def finish(self, coverage):
        if not self.started:
            return dict(
                trace_valid=False, reasons=["profile_not_started"], pressure_valid=False
            )
        try:
            result = self.controller.result(5)
        except TimeoutError:
            result = dict(
                trace_valid=False,
                reasons=["profile_cleanup_timeout"],
                control_log=list(self.controller.log),
            )
        paths = result.get("trace_files", {})
        rank_steps, windows = {}, []
        reasons = set(result.get("reasons", [])) | set(self.errors)
        for rank in self.ranks:
            status = self.statuses.get(rank, {})
            begin, end = int(status.get("capture_begin_ns", 0)), int(
                status.get("capture_end_ns", 0)
            )
            bounds = self.offsets.get(rank)
            if begin <= 0 or end <= begin or not bounds:
                reasons.add("missing_capture_clock_boundaries")
            else:
                windows.append((begin / 1e9 + bounds[0], end / 1e9 + bounds[1]))
            if rank not in paths:
                reasons.add("missing_rank_trace")
                continue
            steps = [
                dict(
                    step,
                    profile_boundary=(step["begin_ns"] < begin or step["end_ns"] > end),
                )
                for step in self.steps[rank].values()
                if begin <= step["forward_enter_ns"] < end
            ]
            rank_steps[str(rank)] = steps
        analysis_started = time.monotonic()
        analysis = analyze_kineto_files(
            paths, rank_steps, self.requested_steps, self.deadline
        )
        analysis["remaining_budget_s"] = max(0, self.deadline - analysis_started)
        analysis["wall_elapsed_s"] = time.monotonic() - analysis_started
        rank_results = analysis.get("ranks", {})
        reasons.update(analysis.get("reasons", []))
        if self.evictions:
            reasons.add("profile_step_history_overflow")
        if result.get("control_errors"):
            reasons.add("profile_control_failed")
        pressure = None
        if len(windows) == len(self.ranks) and coverage is not None:
            # Conservatively cover every rank's entire capture including flush;
            # all target FULL groups must hold across their union in client time.
            start, end = min(w[0] for w in windows), max(w[1] for w in windows)
            pressure = dict(
                start_s=start, end_s=end, **asdict(coverage.assess(start, end))
            )
        valid = (
            result.get("trace_valid", True)
            and not reasons
            and len(rank_results) == len(self.ranks)
            and all(r["trace_valid"] for r in rank_results.values())
        )
        result.update(
            trace_valid=bool(valid),
            reasons=sorted(reasons),
            ranks=rank_results,
            statuses={str(k): v for k, v in self.statuses.items()},
            pressure=pressure,
            pressure_valid=pressure["valid"] if pressure else False,
            trace_name=self.trace_name,
            state_evictions=self.evictions,
            analysis=analysis,
        )
        result["partial_files"] = [
            str(path)
            for path in sorted(self.directory.glob(f"{self.trace_name}_wr*_*.json"))
        ]
        return result


class NsightCapture(KinetoCapture):
    """Duration window with the same bounded sidecar ingestion as Kineto.

    The client command envelope is conservative pressure evidence, not an exact
    GPU activity boundary. SQLite timestamps never participate in that envelope.
    """

    def __init__(
        self,
        *,
        command,
        duration_s,
        trace_name,
        directory,
        ranks,
        instances,
        session_id,
        timeout_s,
        command_timeout_s,
    ):
        if duration_s is None or not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("positive finite Nsight duration required")
        if not math.isfinite(command_timeout_s) or command_timeout_s <= 0:
            raise ValueError("positive finite Nsight command timeout required")
        self.duration_s = min(duration_s, timeout_s)
        self.command = command
        self.output = (Path(directory) / "nsys" / trace_name).resolve()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.report = self.output.with_suffix(".nsys-rep")
        self.sqlite = self.output.with_suffix(".sqlite")
        if self.report.exists() or self.sqlite.exists():
            raise FileExistsError(self.output)
        self.capture_begin_s = self.capture_end_s = None
        self.capture_ready_s = None
        self.command_record = dict(backend="nsys", trace_name=trace_name)
        super().__init__(
            start_capture=self._start_capture,
            read_status=None,
            stop_capture=self._stop_capture,
            trace_name=trace_name,
            directory=directory,
            ranks=ranks,
            instances=instances,
            session_id=session_id,
            timeout_s=timeout_s,
            requested_steps=0,
        )
        # stop includes report flushing and a separate SQLite export, both with
        # their own subprocess deadline. Sender workers continue throughout.
        self.controller.timeouts.update(
            start=command_timeout_s + 1, stop=2 * command_timeout_s + 2
        )

    def _start_capture(self):
        self.capture_begin_s = time.monotonic()
        self.command(
            self.command_record,
            "start",
            "--sample=process-tree",
            "--cpuctxsw=process-tree",
            "--output=" + str(self.output),
        )
        self.capture_ready_s = time.monotonic()

    def consume(self, event):
        if event.get("event") != "pressure":
            return
        raw = event.get("raw", {})
        pages = []
        for page in raw.get("response", {}).get("ranks", []):
            bounds = raw.get("rank_offset_bounds_s", {}).get(
                str(int(page.get("world_rank", 0)))
            )
            records = []
            for step in page.get("steps", []):
                entered = int(step.get("forward_enter_ns", 0)) / 1e9
                if (
                    bounds
                    and self.capture_begin_s is not None
                    and entered + bounds[1] < self.capture_begin_s
                ):
                    continue
                if (
                    bounds
                    and self.capture_end_s is not None
                    and entered + bounds[0] > self.capture_end_s
                ):
                    continue
                records.append(step)
            pages.append(dict(page, steps=records))
        super().consume(
            dict(
                event,
                raw=dict(raw, response=dict(raw.get("response", {}), ranks=pages)),
            )
        )

    def _wait(self):
        self.controller.cancel.wait(
            max(
                0,
                min(self.capture_ready_s + self.duration_s, self.deadline)
                - time.monotonic(),
            )
        )
        return {}

    def _stop_capture(self):
        self.command(self.command_record, "stop")
        self.capture_end_s = time.monotonic()
        if not self.report.is_file() or self.report.stat().st_size == 0:
            raise RuntimeError("nsys stop did not produce a nonempty report")
        self.command(
            self.command_record,
            "export",
            "--type=sqlite",
            "--lazy=false",
            "--output=" + str(self.sqlite),
            str(self.report),
        )
        if not self.sqlite.is_file() or self.sqlite.stat().st_size == 0:
            raise RuntimeError("nsys export did not produce a nonempty SQLite file")

    def pending(self, now):
        # Stop at the capture deadline but keep cleanup/export visible as pending
        # until the session's own support deadline bounds further traffic.
        if self.started and now >= self.started_s + self.timeout_s:
            self.controller.request_stop()
        return self.started and not self.controller.done()

    def finish(self, coverage):
        if not self.started:
            return dict(
                trace_valid=False, pressure_valid=False, reasons=["profile_not_started"]
            )
        try:
            result = self.controller.result(5)
        except TimeoutError:
            result = dict(
                trace_valid=False,
                reasons=["profile_cleanup_timeout"],
                control_log=list(self.controller.log),
            )
        reasons = set(result.get("reasons", [])) | self.errors
        if result.get("control_errors"):
            reasons.add("profile_control_failed")
        if self.evictions:
            reasons.add("profile_step_history_overflow")
        snapshot = {
            rank: list(records.values()) for rank, records in self.steps.items()
        }
        analysis_started = time.monotonic()
        normalized = analyze_nsys_file(
            self.sqlite,
            snapshot,
            self.output.with_suffix(".normalized.json"),
            self.deadline,
        )
        normalized.setdefault("trace_valid", False)
        normalized["analysis_remaining_budget_s"] = max(
            0, self.deadline - analysis_started
        )
        normalized["analysis_wall_elapsed_s"] = time.monotonic() - analysis_started
        reasons.update(normalized.get("reasons", []))
        pressure = None
        if (
            self.capture_begin_s is not None
            and self.capture_end_s is not None
            and coverage is not None
        ):
            pressure = dict(
                start_s=self.capture_begin_s,
                end_s=self.capture_end_s,
                bounds_kind="client_start_call_to_stop_ack",
                **asdict(coverage.assess(self.capture_begin_s, self.capture_end_s)),
            )
        else:
            reasons.add("missing_capture_operation_bounds")
        result.update(normalized)
        result.update(
            trace_valid=not reasons,
            reasons=sorted(reasons),
            pressure=pressure,
            pressure_valid=pressure["valid"] if pressure else False,
            trace_name=self.trace_name,
            backend="nsys",
            capture_ready_s=self.capture_ready_s,
            duration_s=self.duration_s,
            operation_timeout_s=dict(self.controller.timeouts),
            nsys_commands=list(self.command_record.get("nsys_commands", [])),
            trace_files=[str(self.report), str(self.sqlite)],
            partial_files=[
                str(p) for p in sorted(self.output.parent.glob(self.trace_name + "*"))
            ],
            state_evictions=self.evictions,
        )
        return result

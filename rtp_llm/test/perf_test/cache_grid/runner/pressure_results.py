"""Incremental diagnostic evidence, deliberately separate from fit observations."""

import json
import math
import os
import threading
from pathlib import Path


class RequestWindow:
    """Terminal dispatched cohort counters; callers submit each request once."""

    def __init__(self, start_s, end_s):
        if not all(math.isfinite(v) for v in (start_s, end_s)) or end_s <= start_s:
            raise ValueError("invalid measurement window")
        self.start, self.end = start_s, end_s
        self.members = self.crossing = self.completed_inside = 0
        self.successes = self.cancelled = 0

    def observe(self, record):
        if record["kind"] != "formal" or record["phase"] != "measurement":
            return
        if not self.start <= record["dispatch_s"] < self.end:
            raise ValueError("measurement dispatch outside window")
        self.members += 1
        completed = record.get("completed_s")
        if completed is not None and completed >= self.end:
            self.crossing += 1
        elif completed is not None and self.start <= completed < self.end:
            self.completed_inside += 1
            self.successes += record.get("success") is True
        self.cancelled += record.get("terminal") == "cancellation"

    def summary(self):
        return dict(
            measurement_requests=self.members,
            crossing_requests=self.crossing,
            in_window_completions=self.completed_inside,
            request_throughput=self.completed_inside / (self.end - self.start),
            in_window_successes=self.successes,
            success_throughput=self.successes / (self.end - self.start),
            cancelled_requests=self.cancelled,
        )


class PressureWriter:
    def __init__(self, directory: Path, manifest: dict):
        self.directory = Path(directory) / "pressure"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.streams = {}
        self.closed = False
        self.manifest = dict(manifest, complete=False)
        self._atomic("manifest.json", self.manifest)

    def _atomic(self, name, value):
        encoded = json.dumps(value, allow_nan=False, indent=2) + "\n"
        path = self.directory / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, path)

    def append(self, kind: str, record: dict):
        if kind not in {"requests", "cache_samples", "steps"}:
            raise ValueError("unsupported pressure evidence stream")
        encoded = json.dumps(record, allow_nan=False) + "\n"
        with self.lock:
            if self.closed:
                raise ValueError("pressure writer closed")
            if kind not in self.streams:
                self.streams[kind] = (self.directory / (kind + ".jsonl")).open(
                    "a", encoding="utf-8"
                )
            stream = self.streams[kind]
            stream.write(encoded)
            # Each callback is durable to process termination; no queued records.
            stream.flush()

    def finish(self, summary: dict):
        with self.lock:
            if self.closed:
                raise ValueError("pressure writer closed")
            self._atomic("summary.json", summary)
            self.manifest.update(complete=True, validity=summary.get("validity"))
            self._atomic("manifest.json", self.manifest)

    def close(self):
        with self.lock:
            for stream in self.streams.values():
                stream.flush()
                stream.close()
            self.streams.clear()
            self.closed = True


# Task 9 projects these from resolved inputs/configuration; missing is unknown,
# never a fabricated identity or a passing applicability check.
MANIFEST_PROJECTION = (
    "code_hash",
    "profile_hash",
    "grid_hash",
    "resolved_settings",
    "input_identity",
    "epoch",
    "clock_bounds",
    "control_times",
)


class PressureEvidence:
    """Serialized supervisor sink retaining only outstanding dispatched jobs."""

    def __init__(self, writer, downstream=None, *, profile_requested=False):
        self.writer, self.downstream = writer, downstream
        self.profile_requested = profile_requested
        self.pending = {}
        self.window = None
        self.duration = None
        self.lock = threading.RLock()

    def abort(self, error):
        with self.lock:
            self.writer.append("steps", dict(event="error", error=error))
            for record, cancellation in self.pending.values():
                self._terminal(record, "cancellation", cancellation or {})
            self.pending.clear()

    def _terminal(self, record, terminal, event):
        row = dict(record)
        row.setdefault(
            "kind",
            (
                "formal"
                if row["phase"] in {"steady", "measurement", "support"}
                else row["phase"]
            ),
        )
        row["dispatch_s"] = row.pop("dispatched_s")
        row.update(terminal=terminal, completed_s=event.get("completed_s"))
        row.update(event.get("result", {}))
        row["success"] = terminal == "completion" and row.get("success") is True
        if terminal == "cancellation":
            row.update(
                cancel_requested_s=event.get("at_s"),
                transport_cancel_accepted=event.get("accepted"),
            )
        self.writer.append("requests", row)
        if self.window:
            self.window.observe(row)

    def __call__(self, event):
        with self.lock:
            kind = event["event"]
            self.writer.append(
                "cache_samples" if kind == "pressure" else "steps", event
            )
            if kind == "frozen":
                self.duration = event["config"]["measure_s"]
                self.writer.manifest.update(
                    session_id=event["session_id"], frozen=event
                )
                self.writer._atomic("manifest.json", self.writer.manifest)
            elif kind == "phase" and event["phase"] == "measurement":
                self.window = RequestWindow(
                    event["at_s"], event["at_s"] + self.duration
                )
            elif kind == "dispatch":
                self.pending[event["job_id"]] = (event, None)
            elif kind == "completion":
                entry = self.pending.pop(event["job_id"], None)
                if entry:
                    self._terminal(entry[0], kind, event)
            elif kind == "cancellation":
                entry = self.pending.get(event["job_id"])
                if entry:
                    self.pending[event["job_id"]] = (entry[0], event)
            elif kind == "finished":
                for record, cancellation in self.pending.values():
                    self._terminal(record, "cancellation", cancellation or {})
                self.pending.clear()
                failure = event.get("failure")
                profile_failure = bool(failure and failure.startswith("profile"))
                summary = dict(
                    execution_mode="full_pressure_diagnostic_v1",
                    metrics=[],
                    window=self.window.summary() if self.window else None,
                    result=event,
                    validity=dict(
                        execution_valid=not failure or profile_failure,
                        pressure_valid=dict(
                            measurement=event.get("pressure", {}).get("valid"),
                            # The actual capture window is supplied by Task 13;
                            # support coverage alone cannot certify that window.
                            profile=None,
                            support=event.get("support_pressure", {}).get("valid"),
                        ),
                        reuse_valid=None,
                        trace_valid=None,
                        trace_applicability=(
                            "pending" if self.profile_requested else "not_applicable"
                        ),
                    ),
                )
                self.writer.finish(summary)
            if self.downstream:
                self.downstream(event)

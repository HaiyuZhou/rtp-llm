"""Deterministic baseline candidate generation from completed results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Union

from rtp_llm.test.perf_test.plan import case_id
from rtp_llm.test.perf_test.regression.baseline import parse_baseline
from rtp_llm.test.perf_test.results import parse_result_envelope
from rtp_llm.test.perf_test.storage import atomic_write_json

_METRICS = {
    "prefill": (
        ("latency_ms", "lower_is_better"),
        ("throughput_tokens_per_second", "higher_is_better"),
    ),
    "decode": (
        ("latency_ms.wait", "lower_is_better"),
        ("latency_ms.prefill", "lower_is_better"),
        ("latency_ms.decode", "lower_is_better"),
        ("latency_ms.tpot", "lower_is_better"),
        ("latency_ms.total", "lower_is_better"),
        ("throughput_tokens_per_second", "higher_is_better"),
    ),
}


class BaselineCandidateError(ValueError):
    """Raised when a result cannot produce a formal baseline candidate."""


def _metric(metrics: Mapping[str, Any], path: str) -> Any:
    value: Any = metrics
    for name in path.split("."):
        value = value[name]
    return value


def build_baseline_candidate(
    result: Mapping[str, Any], *, threshold: float
) -> Dict[str, Any]:
    """Build a sorted candidate without mutating any accepted baseline."""
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise BaselineCandidateError("threshold must be a number from zero to one")
    threshold_value = float(threshold)
    if not 0 <= threshold_value <= 1:
        raise BaselineCandidateError("threshold must be a number from zero to one")
    envelope = parse_result_envelope(result)
    if envelope.status.value != "completed":
        raise BaselineCandidateError(
            f"baseline candidate requires completed result, got {envelope.status.value}"
        )

    entries = []
    for item in envelope.cases:
        stable_key = case_id(envelope.mode, item.identity)
        if item.case_id != stable_key:
            raise BaselineCandidateError(
                f"stable case key mismatch: result has {item.case_id}, expected {stable_key}"
            )
        if item.status.value != "ok" or item.metrics is None:
            raise BaselineCandidateError(
                f"baseline candidate requires ok case: {item.case_id}"
            )
        metrics = item.metrics.to_dict()
        for metric, direction in _METRICS[envelope.mode]:
            entries.append(
                {
                    "case_key": stable_key,
                    "metric": metric,
                    "value": _metric(metrics, metric),
                    "direction": direction,
                    "threshold": threshold_value,
                }
            )
    entries.sort(key=lambda entry: (entry["case_key"], entry["metric"]))
    candidate = {"schema_version": 1, "entries": entries}
    parse_baseline(candidate)
    return candidate


def write_baseline_candidate(
    root: Union[str, Path], result: Mapping[str, Any], *, threshold: float
) -> Path:
    """Write only reports/baseline_candidate.json under an initialized run."""
    path = Path(root) / "reports" / "baseline_candidate.json"
    atomic_write_json(path, build_baseline_candidate(result, threshold=threshold))
    return path

"""Bounded target-TPOT batch search with complete attempt observations."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


class DecodeCapacityError(RuntimeError):
    """A decode batch candidate cannot fit available engine capacity."""


@dataclass(frozen=True)
class TPOTSearchObservation:
    batch_size: int
    status: str
    tpot_ms: Optional[float]
    success_rate: Optional[float]
    throughput_tokens_per_second: Optional[float]
    satisfies_target: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "status": self.status,
            "tpot_ms": self.tpot_ms,
            "success_rate": self.success_rate,
            "throughput_tokens_per_second": self.throughput_tokens_per_second,
            "satisfies_target": self.satisfies_target,
            "error": self.error,
        }


@dataclass(frozen=True)
class TPOTSearchResult:
    target_tpot_ms: float
    max_batch_size: int
    outcome: str
    best_batch_size: Optional[int]
    actual_tpot_ms: Optional[float]
    throughput_tokens_per_second: Optional[float]
    observations: Tuple[TPOTSearchObservation, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_tpot_ms": self.target_tpot_ms,
            "max_batch_size": self.max_batch_size,
            "outcome": self.outcome,
            "best_batch_size": self.best_batch_size,
            "actual_tpot_ms": self.actual_tpot_ms,
            "throughput_tokens_per_second": self.throughput_tokens_per_second,
            "observations": [item.to_dict() for item in self.observations],
        }


def _candidates(max_batch_size: int, step: int) -> Tuple[int, ...]:
    values = [1]
    candidate = step
    while candidate <= max_batch_size:
        values.append(candidate)
        candidate += step
    if values[-1] != max_batch_size:
        values.append(max_batch_size)
    return tuple(dict.fromkeys(values))


def _finite_number(path: str, value: Any, *, allow_zero: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0 or not allow_zero and parsed == 0:
        raise ValueError(f"{path} must be a finite positive number")
    return parsed


def _measurement(value: Any) -> Tuple[float, float, Optional[float]]:
    if not isinstance(value, Mapping):
        raise ValueError("measurement must be an object")
    status = value.get("status", "ok")
    if status == "capacity_failed":
        raise DecodeCapacityError(str(value.get("error", "capacity failure")))
    if status != "ok":
        raise ValueError(str(value.get("error", f"measurement status {status}")))
    tpot = _finite_number("tpot_ms", value.get("tpot_ms"), allow_zero=False)
    success_rate = _finite_number(
        "success_rate", value.get("success_rate"), allow_zero=True
    )
    if success_rate > 1:
        raise ValueError("success_rate must be between zero and one")
    raw_throughput = value.get("throughput_tokens_per_second")
    throughput = (
        None
        if raw_throughput is None
        else _finite_number(
            "throughput_tokens_per_second", raw_throughput, allow_zero=True
        )
    )
    return tpot, success_rate, throughput


def search_target_tpot(
    measure: Callable[[int], Mapping[str, Any]],
    *,
    target_tpot_ms: float,
    max_batch_size: int,
    candidate_step: int = 4,
) -> TPOTSearchResult:
    """Find the largest measured batch meeting target TPOT and full success."""
    target = _finite_number("target_tpot_ms", target_tpot_ms, allow_zero=False)
    if type(max_batch_size) is not int or max_batch_size <= 0:
        raise ValueError("max_batch_size must be a positive integer")
    if type(candidate_step) is not int or candidate_step <= 0:
        raise ValueError("candidate_step must be a positive integer")
    candidates = _candidates(max_batch_size, candidate_step)
    lower, upper = 0, len(candidates) - 1
    best_batch: Optional[int] = None
    best_tpot: Optional[float] = None
    best_throughput: Optional[float] = None
    measured = False
    observations: List[TPOTSearchObservation] = []
    while lower <= upper:
        middle = (lower + upper) // 2
        batch_size = candidates[middle]
        try:
            tpot, success_rate, throughput = _measurement(measure(batch_size))
        except DecodeCapacityError as error:
            observations.append(
                TPOTSearchObservation(
                    batch_size=batch_size,
                    status="capacity_failed",
                    tpot_ms=None,
                    success_rate=None,
                    throughput_tokens_per_second=None,
                    satisfies_target=False,
                    error=str(error),
                )
            )
            upper = middle - 1
            continue
        except Exception as error:
            observations.append(
                TPOTSearchObservation(
                    batch_size=batch_size,
                    status="measurement_failed",
                    tpot_ms=None,
                    success_rate=None,
                    throughput_tokens_per_second=None,
                    satisfies_target=False,
                    error=str(error),
                )
            )
            upper = middle - 1
            continue

        measured = True
        satisfies = success_rate == 1.0 and tpot <= target
        observations.append(
            TPOTSearchObservation(
                batch_size=batch_size,
                status="measured",
                tpot_ms=tpot,
                success_rate=success_rate,
                throughput_tokens_per_second=throughput,
                satisfies_target=satisfies,
            )
        )
        if satisfies:
            best_batch = batch_size
            best_tpot = tpot
            best_throughput = throughput
            lower = middle + 1
        else:
            upper = middle - 1

    outcome = (
        "found"
        if best_batch is not None
        else "no_satisfying_candidate" if measured else "all_candidates_failed"
    )
    return TPOTSearchResult(
        target_tpot_ms=target,
        max_batch_size=max_batch_size,
        outcome=outcome,
        best_batch_size=best_batch,
        actual_tpot_ms=best_tpot,
        throughput_tokens_per_second=best_throughput,
        observations=tuple(observations),
    )

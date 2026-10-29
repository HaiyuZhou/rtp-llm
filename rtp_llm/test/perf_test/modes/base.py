"""Narrow contracts shared by prefill and decode mode adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

from rtp_llm.test.perf_test.errors import PerfTestError
from rtp_llm.test.perf_test.plan import ExecutionPlan
from rtp_llm.test.perf_test.state import RunState


@dataclass(frozen=True)
class ModeExecutionContext:
    plan: ExecutionPlan
    session: Any
    store: Any


@dataclass(frozen=True)
class ModeExecutionResult:
    state: RunState
    case_results: Tuple[Mapping[str, Any], ...]
    failure: Optional[PerfTestError] = None

    def __post_init__(self) -> None:
        if self.state not in {
            RunState.COMPLETED,
            RunState.PARTIAL,
            RunState.FAILED,
        }:
            raise ValueError("mode execution result requires a terminal state")

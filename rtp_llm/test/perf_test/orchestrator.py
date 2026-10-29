"""Backend dispatch for immutable unified perf-test plans."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from rtp_llm.test.perf_test.errors import PlanningError
from rtp_llm.test.perf_test.plan import ExecutionPlan


class Orchestrator:
    def __init__(self, executors: Mapping[str, Callable[[ExecutionPlan], Any]]):
        self._executors = dict(executors)

    def execute(self, plan: ExecutionPlan) -> Any:
        executor = self._executors.get(plan.backend)
        if executor is None:
            raise PlanningError(f"no executor registered for backend {plan.backend!r}")
        return executor(plan)

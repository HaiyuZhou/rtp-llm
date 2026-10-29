"""Local execution of one immutable plan through one engine session."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Tuple

from rtp_llm.test.perf_test.errors import (
    EngineLifecycleError,
    PerfTestError,
    PlanningError,
)
from rtp_llm.test.perf_test.execution.engine_session import (
    EngineDiagnostics,
    EngineInterrupted,
)
from rtp_llm.test.perf_test.modes.base import ModeExecutionContext, ModeExecutionResult
from rtp_llm.test.perf_test.plan import ExecutionPlan
from rtp_llm.test.perf_test.state import RunOutcome, RunState


@dataclass(frozen=True)
class ExecutionOutcome:
    state: RunState
    case_results: Tuple[Mapping[str, Any], ...] = ()
    failure: Optional[PerfTestError] = None
    signal_number: Optional[int] = None
    engine_diagnostics: EngineDiagnostics = field(default_factory=EngineDiagnostics)

    @property
    def exit_code(self) -> int:
        return RunOutcome(
            state=self.state,
            failure=self.failure,
            signal_number=self.signal_number,
        ).exit_code


class LocalExecutor:
    def __init__(
        self,
        *,
        session_factory: Callable[[ExecutionPlan], Any],
        store_factory: Callable[[str], Any],
        adapters: Mapping[str, Callable[[ModeExecutionContext], ModeExecutionResult]],
    ):
        self._session_factory = session_factory
        self._store_factory = store_factory
        self._adapters = dict(adapters)

    def execute(self, plan: ExecutionPlan) -> ExecutionOutcome:
        if plan.backend != "local":
            raise PlanningError("LocalExecutor requires a local execution plan")
        adapter = self._adapters.get(plan.mode)
        if adapter is None:
            raise PlanningError(f"no local adapter registered for mode {plan.mode!r}")
        session = self._session_factory(plan)
        store = self._store_factory(plan.result_directory)
        try:
            with session as active_session:
                result = adapter(
                    ModeExecutionContext(
                        plan=plan,
                        session=active_session,
                        store=store,
                    )
                )
        except EngineInterrupted as error:
            return ExecutionOutcome(
                state=RunState.INTERRUPTED,
                signal_number=error.signal_number,
                engine_diagnostics=session.diagnostics,
            )
        except EngineLifecycleError as error:
            diagnostics = getattr(error, "diagnostics", session.diagnostics)
            return ExecutionOutcome(
                state=RunState.FAILED,
                failure=error,
                engine_diagnostics=diagnostics,
            )
        except PerfTestError as error:
            return ExecutionOutcome(
                state=RunState.FAILED,
                failure=error,
                engine_diagnostics=session.diagnostics,
            )
        return ExecutionOutcome(
            state=result.state,
            case_results=result.case_results,
            failure=result.failure,
            engine_diagnostics=session.diagnostics,
        )

"""Prefill mode adapter with atomic per-case progress."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from rtp_llm.test.perf_test.errors import CaseExecutionError
from rtp_llm.test.perf_test.modes.base import ModeExecutionContext, ModeExecutionResult
from rtp_llm.test.perf_test.plan import thaw
from rtp_llm.test.perf_test.prefill.checkpoint import resume_fingerprints
from rtp_llm.test.perf_test.state import RunState


class PrefillAdapter:
    def __init__(self, runner_factory: Callable[[ModeExecutionContext], Any]):
        self._runner_factory = runner_factory

    @staticmethod
    def _request_failure(case: Dict[str, Any], error: Exception) -> Dict[str, Any]:
        return {
            "case_id": case["case_id"],
            "identity": {key: value for key, value in case.items() if key != "case_id"},
            "status": "request_failed",
            "runs": int(case.get("measure_runs", 0)),
            "errors": [{"code": "request_failed", "message": str(error)}],
            "metrics": None,
        }

    def run(
        self,
        context: ModeExecutionContext,
        *,
        selected_case_ids=None,
        existing_results=(),
    ) -> ModeExecutionResult:
        if context.plan.mode != "prefill":
            raise ValueError("PrefillAdapter requires a prefill execution plan")
        runner = self._runner_factory(context)
        results: List[Dict[str, Any]] = list(existing_results)
        statuses: Dict[str, str] = {
            result["case_id"]: result["status"] for result in results
        }
        completed: List[str] = [result["case_id"] for result in results]
        selected = (
            {case.case_id for case in context.plan.cases}
            if selected_case_ids is None
            else set(selected_case_ids)
        )
        for planned_case in context.plan.cases:
            if planned_case.case_id not in selected:
                continue
            case = thaw(planned_case.identity)
            case["case_id"] = planned_case.case_id
            try:
                result = runner.run_case(case)
            except Exception as error:
                result = self._request_failure(case, error)
            context.store.write_case(planned_case.case_id, result)
            results.append(result)
            completed.append(planned_case.case_id)
            statuses[planned_case.case_id] = result["status"]
            context.store.write_checkpoint(
                "prefill_progress",
                {
                    "schema_version": 1,
                    **resume_fingerprints(context.plan),
                    "completed_case_ids": completed,
                    "case_statuses": statuses,
                },
            )
        failed = [result for result in results if result["status"] != "ok"]
        if failed:
            return ModeExecutionResult(
                state=RunState.PARTIAL,
                case_results=tuple(results),
                failure=CaseExecutionError(
                    f"{len(failed)} of {len(results)} prefill cases failed validation"
                ),
            )
        return ModeExecutionResult(
            state=RunState.COMPLETED,
            case_results=tuple(results),
        )

    def __call__(self, context: ModeExecutionContext) -> ModeExecutionResult:
        return self.run(context)

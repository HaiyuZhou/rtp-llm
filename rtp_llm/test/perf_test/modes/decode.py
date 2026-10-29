"""Decode mode adapter for fixed and distribution case runners."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from rtp_llm.test.perf_test.errors import CaseExecutionError
from rtp_llm.test.perf_test.modes.base import ModeExecutionContext, ModeExecutionResult
from rtp_llm.test.perf_test.plan import thaw
from rtp_llm.test.perf_test.state import RunState


class DecodeAdapter:
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

    def __call__(self, context: ModeExecutionContext) -> ModeExecutionResult:
        if context.plan.mode != "decode":
            raise ValueError("DecodeAdapter requires a decode execution plan")
        runner = self._runner_factory(context)
        results: List[Dict[str, Any]] = []
        statuses: Dict[str, str] = {}
        for planned_case in context.plan.cases:
            case = thaw(planned_case.identity)
            case["case_id"] = planned_case.case_id
            try:
                result = runner.run_case(case)
            except Exception as error:
                result = self._request_failure(case, error)
            context.store.write_case(planned_case.case_id, result)
            results.append(result)
            statuses[planned_case.case_id] = result["status"]
            context.store.write_checkpoint(
                "decode_progress",
                {
                    "schema_version": 1,
                    "config_fingerprint": context.plan.config_fingerprint,
                    "workload_fingerprint": context.plan.workload_fingerprint,
                    "completed_case_ids": [item["case_id"] for item in results],
                    "case_statuses": statuses,
                },
            )
        failed = [result for result in results if result["status"] != "ok"]
        if failed:
            return ModeExecutionResult(
                state=RunState.PARTIAL,
                case_results=tuple(results),
                failure=CaseExecutionError(
                    f"{len(failed)} of {len(results)} decode cases failed"
                ),
            )
        return ModeExecutionResult(
            state=RunState.COMPLETED, case_results=tuple(results)
        )

"""Finalize local prefill measurements into the common result envelope."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Union

from rtp_llm.test.perf_test.errors import PlanningError
from rtp_llm.test.perf_test.plan import ExecutionPlan, thaw
from rtp_llm.test.perf_test.results import RESULT_SCHEMA_VERSION, parse_result_envelope
from rtp_llm.test.perf_test.state import RunState
from rtp_llm.test.perf_test.storage import RunStore


def finalize_prefill_run(
    root: Union[str, Path],
    *,
    plan: ExecutionPlan,
    run_id: str,
    case_results: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    if (
        plan.mode != "prefill"
        or plan.action not in {"run", "resume"}
        or plan.backend != "local"
    ):
        raise PlanningError("prefill finalization requires a local prefill run plan")
    if not isinstance(run_id, str) or not run_id:
        raise PlanningError("prefill run_id must be a non-empty string")

    provided: Dict[str, Dict[str, Any]] = {}
    for result in case_results:
        if not isinstance(result, Mapping) or not isinstance(
            result.get("case_id"), str
        ):
            raise PlanningError("prefill case result must contain case_id")
        case_id = result["case_id"]
        if case_id in provided:
            raise PlanningError(f"duplicate prefill case result: {case_id}")
        provided[case_id] = copy.deepcopy(dict(result))
    planned_ids = {case.case_id for case in plan.cases}
    if set(provided) != planned_ids:
        raise PlanningError(
            "prefill results must cover every planned case exactly once"
        )

    ordered = []
    for case in plan.cases:
        result = provided[case.case_id]
        if result.get("identity") != thaw(case.identity):
            raise PlanningError(f"prefill case identity mismatch: {case.case_id}")
        ordered.append(result)
    successful = sum(result.get("status") == "ok" for result in ordered)
    state = RunState.COMPLETED if successful == len(ordered) else RunState.PARTIAL
    summary = {
        "planned_cases": len(ordered),
        "successful_cases": successful,
        "failed_cases": len(ordered) - successful,
    }
    resolved_model = plan.to_dict()["resolved_config"]["model"]
    envelope = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": run_id,
        "mode": "prefill",
        "backend": "local",
        "status": state.value,
        "config_fingerprint": plan.config_fingerprint,
        "workload_fingerprint": plan.workload_fingerprint,
        "model": {
            name: copy.deepcopy(resolved_model[name])
            for name in ("label", "type", "checkpoint", "topology")
        },
        "summary": summary,
        "cases": ordered,
    }
    parse_result_envelope(envelope)

    store = RunStore(root)
    for result in ordered:
        store.write_case(result["case_id"], result)
    store.write_final_result(envelope)
    store.write_summary(
        {
            "schema_version": 1,
            "run_id": run_id,
            "mode": "prefill",
            "backend": "local",
            "status": state.value,
            **summary,
        }
    )
    return envelope

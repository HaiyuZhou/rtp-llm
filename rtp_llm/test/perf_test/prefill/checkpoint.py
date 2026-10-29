"""Strict local-prefill resume identity and selection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

from rtp_llm.test.perf_test.config_resolver import canonical_json
from rtp_llm.test.perf_test.errors import PlanningError
from rtp_llm.test.perf_test.modes.base import ModeExecutionContext, ModeExecutionResult
from rtp_llm.test.perf_test.plan import ExecutionPlan
from rtp_llm.test.perf_test.run_directory import RunDirectoryLock

RUNNER_CONTRACT = {
    "name": "unified-prefill-runner",
    "version": 1,
    "prompt_contract": "exact-token-prefix-v1",
    "measurement_contract": "exact-reuse-and-positive-latency-v1",
}


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def resume_fingerprints(plan: ExecutionPlan) -> Dict[str, str]:
    return {
        "config_fingerprint": plan.config_fingerprint,
        "workload_fingerprint": plan.workload_fingerprint,
        "model_fingerprint": _sha(plan.to_dict()["resolved_config"]["model"]),
        "runner_fingerprint": _sha(RUNNER_CONTRACT),
    }


@dataclass(frozen=True)
class ResumeSelection:
    rerun_case_ids: Tuple[str, ...]
    preserved_results: Tuple[Dict[str, Any], ...]


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PlanningError(
            f"required resume {label} is missing or invalid: {path}"
        ) from error


def _validate_resume(context: ModeExecutionContext) -> ResumeSelection:
    plan = context.plan
    if plan.mode != "prefill" or plan.backend != "local" or plan.action != "resume":
        raise PlanningError("prefill resume requires a local prefill resume plan")
    root = Path(plan.result_directory)
    persisted = ExecutionPlan.from_dict(
        _read_json(root / "execution_plan.json", "execution plan")
    )
    if persisted.mode != "prefill" or persisted.backend != "local":
        raise PlanningError("persisted execution plan is not local prefill")
    expected = resume_fingerprints(plan)
    persisted_fingerprints = resume_fingerprints(persisted)
    for name in (
        "config_fingerprint",
        "workload_fingerprint",
        "model_fingerprint",
        "runner_fingerprint",
    ):
        if persisted_fingerprints[name] != expected[name]:
            raise PlanningError(f"resume {name} mismatch")
    if [case.to_dict() for case in persisted.cases] != [
        case.to_dict() for case in plan.cases
    ]:
        raise PlanningError("resume planned case identity mismatch")
    checkpoint = _read_json(
        root / "checkpoints" / "prefill_progress.json", "checkpoint"
    )
    if not isinstance(checkpoint, dict):
        raise PlanningError("resume checkpoint must be an object")
    for name, value in expected.items():
        if checkpoint.get(name) != value:
            raise PlanningError(f"resume checkpoint {name} mismatch")
    committed = context.store.read_committed_cases()
    planned_ids = {case.case_id for case in plan.cases}
    unknown = set(committed) - planned_ids
    if unknown:
        raise PlanningError(f"resume found unknown case: {sorted(unknown)[0]}")
    preserved = []
    rerun = []
    for case in plan.cases:
        result = committed.get(case.case_id)
        if result is not None and result.get("status") == "ok":
            preserved.append(result)
        else:
            rerun.append(case.case_id)
    return ResumeSelection(tuple(rerun), tuple(preserved))


class PrefillResume:
    def __init__(self, adapter: Any):
        self._adapter = adapter

    def __call__(self, context: ModeExecutionContext) -> ModeExecutionResult:
        with RunDirectoryLock(context.plan.result_directory, action="resume"):
            selection = _validate_resume(context)
            return self._adapter.run(
                context,
                selected_case_ids=selection.rerun_case_ids,
                existing_results=selection.preserved_results,
            )

"""Lightweight prefill plan validation shared by entry and runtime paths."""

from rtp_llm.test.perf_test.errors import PlanningError
from rtp_llm.test.perf_test.plan import ExecutionPlan
from rtp_llm.test.perf_test.prefill.measurement import PREFILL_MEASUREMENT_CONTRACT


def require_concurrent_prefill_plan(plan: ExecutionPlan) -> None:
    """Reject persisted measurements whose batch semantics cannot be resumed."""
    if any(
        case.identity.get("measurement_contract") != PREFILL_MEASUREMENT_CONTRACT
        for case in plan.cases
    ):
        raise PlanningError(
            f"prefill resume/profile requires {PREFILL_MEASUREMENT_CONTRACT}; "
            "rerun from config in a new directory for new measurements. "
            "Existing directories remain available for offline reports."
        )

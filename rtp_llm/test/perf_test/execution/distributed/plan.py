"""Bind an already frozen plan to a member without resolving source inputs."""

from dataclasses import replace
from pathlib import Path

from rtp_llm.test.perf_test.plan import ExecutionPlan


def bind_member_plan(plan: ExecutionPlan, *, result_directory: str) -> ExecutionPlan:
    if (
        not isinstance(result_directory, str)
        or not Path(result_directory).is_absolute()
    ):
        raise ValueError("member result_directory must be an absolute path")
    if plan.action != "run":
        raise ValueError("member execution requires a frozen run plan")
    return replace(plan, result_directory=result_directory)

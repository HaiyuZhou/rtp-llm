"""Side-effect-free execution-plan previews."""

from __future__ import annotations

from pathlib import PurePath
from typing import Any, Callable, Dict, List

from rtp_llm.test.perf_test.aone.sharding import plan_shards
from rtp_llm.test.perf_test.plan import AOneResourcePlan, ExecutionPlan


def _command_preview(plan: ExecutionPlan) -> List[Dict[str, Any]]:
    if isinstance(plan.resource_plan, AOneResourcePlan):
        commands = []
        for shard in range(plan.resource_plan.shard_count):
            target = f"//generated/perf_test:shard_{shard:04d}"
            commands.extend(
                [
                    {
                        "stage": "build-shard",
                        "shard": shard,
                        "argv": ["bazel", "build", target],
                    },
                    {
                        "stage": "submit-shard",
                        "shard": shard,
                        "argv": ["aone", "submit", "--target", target],
                    },
                ]
            )
        return commands
    plan_path = str(PurePath(plan.result_directory) / "execution_plan.json")
    return [
        {
            "stage": "local-run",
            "argv": ["perf_test_worker", "--plan", plan_path],
        }
    ]


def create_preview(plan: ExecutionPlan) -> Dict[str, Any]:
    """Render all resolved plan inputs without executing or writing them."""
    plan_dict = plan.to_dict()
    preview = {
        "schema_version": 1,
        "dry_run": True,
        "mode": plan.mode,
        "action": plan.action,
        "backend": plan.backend,
        "result_directory": plan.result_directory,
        "config_fingerprint": plan.config_fingerprint,
        "workload_fingerprint": plan.workload_fingerprint,
        "normalized_config": plan_dict["resolved_config"],
        "cases": {
            "count": len(plan.cases),
            "case_ids": [case.case_id for case in plan.cases],
            "items": [case.to_dict() for case in plan.cases],
        },
        "resource_summary": plan.resource_plan.to_dict(),
        "commands": _command_preview(plan),
    }
    if isinstance(plan.resource_plan, AOneResourcePlan):
        shard_plan = plan_shards(plan)
        preview["shards"] = {
            "count": len(shard_plan.shards),
            "total_estimated_cost": shard_plan.total_estimated_cost,
            "items": [
                {
                    **shard.to_dict(),
                    "target": f"//generated/perf_test:shard_{index:04d}",
                }
                for index, shard in enumerate(shard_plan.shards)
            ],
        }
    return preview


def dispatch_plan(
    plan: ExecutionPlan,
    *,
    dry_run: bool,
    execute: Callable[[ExecutionPlan], Any],
) -> Any:
    """Keep dry-run on a branch that cannot invoke the execution callback."""
    if dry_run:
        return create_preview(plan)
    return execute(plan)

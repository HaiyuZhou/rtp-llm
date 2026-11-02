"""Side-effect-free execution-plan previews."""

from __future__ import annotations

from pathlib import PurePath
from typing import Any, Callable, Dict, List

from rtp_llm.test.perf_test.aone.sharding import plan_shards
from rtp_llm.test.perf_test.plan import (
    AONE_GPU_PLATFORMS,
    AOneResourcePlan,
    ExecutionPlan,
)

# Symbolic location of the single gang target a generated gang package
# exposes (Bazel expands it into one shard per machine at submission time).
GANG_TARGET = "//generated/perf_test:gang"


def _is_gang_resource(resource: AOneResourcePlan) -> bool:
    return resource.machine_count > 1


def _gang_member_layout(resource: AOneResourcePlan) -> List[Dict[str, Any]]:
    return [
        {
            "shard_index": shard_index,
            "world_ranks": [
                shard_index * resource.local_world_size + offset
                for offset in range(resource.local_world_size)
            ],
            "is_master": shard_index == 0,
        }
        for shard_index in range(resource.machine_count)
    ]


def _command_preview(plan: ExecutionPlan) -> List[Dict[str, Any]]:
    resource = plan.resource_plan
    if isinstance(resource, AOneResourcePlan):
        if _is_gang_resource(resource):
            targets = [GANG_TARGET]
            # The whole gang must be placed as one wave, exactly like the
            # gang executor's submission stage.
            jobs = max(resource.max_parallel, resource.machine_count)
        else:
            targets = [
                f"//generated/perf_test:shard_{shard:04d}"
                for shard in range(resource.shard_count)
            ]
            jobs = resource.max_parallel
        common = [f"--config={AONE_GPU_PLATFORMS[resource.gpu_type]}", "--config=cicd"]
        return [
            {
                "stage": "build",
                "argv": [
                    "bazelisk",
                    "build",
                    *targets,
                    *common,
                    f"--config={resource.resource}_aone_bazel_cache",
                ],
            },
            {
                "stage": "remote-test",
                "argv": [
                    "bazelisk",
                    "test",
                    *targets,
                    *common,
                    f"--config={resource.resource}_aone_bazel_remote",
                    "--strategy=TestRunner=remote",
                    f"--jobs={jobs}",
                ],
            },
        ]
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
    resource = plan.resource_plan
    if isinstance(resource, AOneResourcePlan) and _is_gang_resource(resource):
        # A gang plan has no case shards: every machine hosts the full plan's
        # rank group, so the preview prints the gang topology instead.
        preview["gang"] = {
            "machine_count": resource.machine_count,
            "gpus_per_machine": resource.gpus_per_machine,
            "local_world_size": resource.local_world_size,
            "world_size": resource.machine_count * resource.local_world_size,
            "master_shard_index": 0,
            "target": GANG_TARGET,
            "members": _gang_member_layout(resource),
        }
    elif isinstance(resource, AOneResourcePlan):
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

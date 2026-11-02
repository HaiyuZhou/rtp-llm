"""Generate self-contained Bazel gang packages for multi-machine AOne runs.

A gang package holds the frozen normalized inputs plus a single
``perf_test_gang_shard`` target that Bazel expands into ``machine_count``
remote test shards. The generated BUILD text references only
package-relative runfile names, so no submit-host path leaks into the
package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from rtp_llm.test.perf_test.aone.package import (
    GeneratedAOnePackage,
    _atomic_write_text,
    _validated_package_name,
)
from rtp_llm.test.perf_test.errors import PlanningError
from rtp_llm.test.perf_test.plan import (
    AONE_GPU_HOSTS,
    AOneResourcePlan,
    ExecutionPlan,
    thaw,
)
from rtp_llm.test.perf_test.storage import atomic_write_json

GANG_PLAN_SCHEMA_VERSION = 1

# Base port of the per-machine local rank group. Every gang member reserves
# ``local_world_size`` consecutive ports from here, so the worker servers of
# one gang never collide with each other on their host.
GANG_BASE_PORT = 12333


def _per_host_gpu_count(gpu_type: str) -> int:
    """Return the registered full-host GPU count for ``gpu_type``."""
    per_host = AONE_GPU_HOSTS.get(gpu_type)
    if per_host is None:
        supported = ", ".join(sorted(AONE_GPU_HOSTS))
        raise PlanningError(
            f"unsupported GPU type {gpu_type!r} for gang execution: gang "
            "shards request full hosts, so the GPU type must have a "
            "per-host GPU count in AONE_GPU_HOSTS; "
            f"supported GPU types: {supported}"
        )
    return per_host


def build_gang_plan_payload(plan: ExecutionPlan, run_id: str) -> Dict[str, Any]:
    """Build the ``gang_plan.json`` payload shared by every gang member."""
    if not isinstance(run_id, str) or not run_id:
        raise PlanningError("AOne run_id must be a non-empty string")
    resource = plan.resource_plan
    if not isinstance(resource, AOneResourcePlan):
        raise PlanningError("gang plan requires an AOne resource plan")
    return {
        "schema_version": GANG_PLAN_SCHEMA_VERSION,
        "run_id": run_id,
        "config_fingerprint": plan.config_fingerprint,
        "workload_fingerprint": plan.workload_fingerprint,
        "world_size": resource.machine_count * resource.local_world_size,
        "machine_count": resource.machine_count,
        "local_world_size": resource.local_world_size,
        "gpus_per_machine": resource.gpus_per_machine,
        "gpus_per_host": _per_host_gpu_count(resource.gpu_type),
        "gpu_type": resource.gpu_type,
        "topology": thaw(plan.resolved_config)["model"]["topology"],
        "master_shard_index": 0,
        "port_range": {
            "start": GANG_BASE_PORT,
            "count": resource.local_world_size,
        },
    }


def _build_file(resource: AOneResourcePlan) -> str:
    lines = [
        'load("//rtp_llm/test/perf_test:defs.bzl", "perf_test_gang_shard")',
        "",
        "perf_test_gang_shard(",
        '    name = "gang",',
        '    resolved_config = "resolved_config.json",',
        '    resolved_workload = "resolved_workload.json",',
        '    gang_plan = "gang_plan.json",',
        f'    gpu_type = "{resource.gpu_type}",',
        f"    gpus_per_host = {_per_host_gpu_count(resource.gpu_type)},",
        f"    machine_count = {resource.machine_count},",
        ")",
        "",
        "test_suite(",
        '    name = "all",',
        '    tests = [":gang"],',
        '    tags = ["manual"],',
        ")",
        "",
    ]
    return "\n".join(lines)


def generate_gang_package(
    root: Union[str, Path],
    plan: ExecutionPlan,
    *,
    run_id: str,
    package: Optional[str] = None,
) -> GeneratedAOnePackage:
    """Materialize the frozen gang inputs without submit-host source paths."""
    if not isinstance(run_id, str) or not run_id:
        raise PlanningError("AOne run_id must be a non-empty string")
    resource = plan.resource_plan
    if not isinstance(resource, AOneResourcePlan):
        raise PlanningError("gang package requires an AOne resource plan")
    gang_payload = build_gang_plan_payload(plan, run_id)
    resolved = thaw(plan.resolved_config)
    workload = resolved.pop("workload")

    output = Path(root)
    package_name = _validated_package_name(output.name if package is None else package)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise PlanningError(
            f"generated gang package already exists: {output}"
        ) from error
    except OSError as error:
        raise PlanningError(
            f"cannot create generated gang package {output}: {error}"
        ) from error

    resolved_config = output / "resolved_config.json"
    resolved_workload = output / "resolved_workload.json"
    gang_plan = output / "gang_plan.json"
    build_file = output / "BUILD.bazel"
    atomic_write_json(resolved_config, resolved)
    atomic_write_json(resolved_workload, workload)
    atomic_write_json(gang_plan, gang_payload)
    _atomic_write_text(build_file, _build_file(resource))
    return GeneratedAOnePackage(
        root=output,
        resolved_config=resolved_config,
        resolved_workload=resolved_workload,
        shard_plan=gang_plan,
        build_file=build_file,
        targets=(f"//{package_name}:gang",),
    )

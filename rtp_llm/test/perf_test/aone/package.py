"""Generate self-contained Bazel runfile packages for AOne prefill shards."""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

from rtp_llm.test.perf_test.errors import PlanningError
from rtp_llm.test.perf_test.plan import AOneResourcePlan, ExecutionPlan, thaw
from rtp_llm.test.perf_test.storage import atomic_write_json

PACKAGE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class GeneratedAOnePackage:
    root: Path
    resolved_config: Path
    resolved_workload: Path
    shard_plan: Path
    build_file: Path
    targets: Tuple[str, ...]


def _validated_ownership(
    plan: ExecutionPlan, ownership: Sequence[Sequence[str]]
) -> Tuple[Tuple[str, ...], ...]:
    resource = plan.resource_plan
    if (
        plan.mode != "prefill"
        or plan.action != "run"
        or not isinstance(resource, AOneResourcePlan)
    ):
        raise PlanningError("AOne package requires an AOne prefill run plan")
    if len(ownership) != resource.shard_count:
        raise PlanningError("AOne ownership count must match the planned shard count")
    normalized: List[Tuple[str, ...]] = []
    flattened: List[str] = []
    for index, case_ids in enumerate(ownership):
        shard = tuple(case_ids)
        if not shard:
            raise PlanningError(f"AOne shard {index} cannot be empty")
        if len(set(shard)) != len(shard):
            raise PlanningError(f"AOne shard {index} contains duplicate cases")
        normalized.append(tuple(sorted(shard)))
        flattened.extend(shard)
    if len(set(flattened)) != len(flattened):
        raise PlanningError("AOne case ownership overlaps across shards")
    planned = {case.case_id for case in plan.cases}
    owned = set(flattened)
    if owned != planned:
        missing = sorted(planned - owned)
        unknown = sorted(owned - planned)
        raise PlanningError(
            f"AOne ownership does not cover the plan: missing={missing}, unknown={unknown}"
        )
    return tuple(normalized)


def _build_file(resource: AOneResourcePlan, ownership: Iterable[Sequence[str]]) -> str:
    lines = [
        'load("//rtp_llm/test/perf_test:defs.bzl", "perf_test_aone_shard")',
        "",
    ]
    for index, _case_ids in enumerate(ownership):
        lines.extend(
            [
                "perf_test_aone_shard(",
                f'    name = "shard_{index:04d}",',
                '    resolved_config = "resolved_config.json",',
                '    resolved_workload = "resolved_workload.json",',
                '    shard_plan = "shard_plan.json",',
                f'    shard_id = "shard-{index:04d}",',
                f'    gpu_type = "{resource.gpu_type}",',
                f"    gpu_count = {resource.gpus_per_shard},",
                ")",
                "",
            ]
        )
    targets = ", ".join(f'":shard_{index:04d}"' for index, _ in enumerate(ownership))
    lines.extend(
        [
            "test_suite(",
            '    name = "all",',
            f"    tests = [{targets}],",
            '    tags = ["manual"],',
            ")",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write_text(path: Path, value: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    replaced = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        replaced = True
    finally:
        if not replaced:
            temporary.unlink(missing_ok=True)


def generate_aone_package(
    root: Union[str, Path],
    plan: ExecutionPlan,
    *,
    run_id: str,
    ownership: Sequence[Sequence[str]],
    package: Optional[str] = None,
) -> GeneratedAOnePackage:
    """Materialize normalized inputs without retaining submit-host source paths."""
    if not isinstance(run_id, str) or not run_id:
        raise PlanningError("AOne run_id must be a non-empty string")
    normalized_ownership = _validated_ownership(plan, ownership)
    resource = plan.resource_plan
    assert isinstance(resource, AOneResourcePlan)
    cases = {case.case_id: case.to_dict() for case in plan.cases}
    resolved = thaw(plan.resolved_config)
    workload = resolved.pop("workload")
    shard_payload = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "run_id": run_id,
        "mode": plan.mode,
        "config_fingerprint": plan.config_fingerprint,
        "workload_fingerprint": plan.workload_fingerprint,
        "shards": [
            {
                "shard_id": f"shard-{index:04d}",
                "cases": [cases[case_id] for case_id in case_ids],
            }
            for index, case_ids in enumerate(normalized_ownership)
        ],
    }

    output = Path(root)
    package_name = output.name if package is None else package
    if not isinstance(package_name, str) or not re.fullmatch(
        r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", package_name
    ):
        raise PlanningError(f"invalid generated Bazel package: {package_name!r}")
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise PlanningError(
            f"generated AOne package already exists: {output}"
        ) from error
    except OSError as error:
        raise PlanningError(
            f"cannot create generated AOne package {output}: {error}"
        ) from error

    resolved_config = output / "resolved_config.json"
    resolved_workload = output / "resolved_workload.json"
    shard_plan = output / "shard_plan.json"
    build_file = output / "BUILD.bazel"
    atomic_write_json(resolved_config, resolved)
    atomic_write_json(resolved_workload, workload)
    atomic_write_json(shard_plan, shard_payload)
    _atomic_write_text(build_file, _build_file(resource, normalized_ownership))
    return GeneratedAOnePackage(
        root=output,
        resolved_config=resolved_config,
        resolved_workload=resolved_workload,
        shard_plan=shard_plan,
        build_file=build_file,
        targets=tuple(
            f"//{package_name}:shard_{index:04d}"
            for index in range(len(normalized_ownership))
        ),
    )

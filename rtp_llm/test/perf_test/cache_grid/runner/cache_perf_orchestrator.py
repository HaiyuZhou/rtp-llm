"""Plan cache-perf workflows in memory before writing inputs or launching Bazel."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import re
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from rtp_llm.test.perf_test.cache_grid.config.grid_config import dedupe_cache_grid_cases
from rtp_llm.test.perf_test.cache_grid.config.perf_profile import profile_environment
from rtp_llm.test.perf_test.cache_grid.config.run_request import (
    ResolvedRunRequest,
    resolve_run_request,
    to_prepare_namespace,
)
from rtp_llm.test.perf_test.cache_grid.runner import cache_perf as cli
from rtp_llm.test.perf_test.cache_grid.runner import cache_perf_ci as ci
from rtp_llm.test.perf_test.cache_grid.runner import unified_pipeline

REPO = Path(__file__).resolve().parents[5]
LOCAL_PLATFORM_CONFIGS = {
    "x86_64": "cuda13",
    "aarch64": "cuda13_arm",
}


@dataclass(frozen=True)
class RunPaths:
    result_root: Path
    inputs_dir: Path
    generated_dir: Optional[Path]


@dataclass
class OrchestrationPlan:
    mode: str
    paths: RunPaths
    resolved: ResolvedRunRequest
    request_cases: List[Dict[str, Any]]
    artifacts: Dict[Path, bytes]
    commands: Dict[str, List[str]]
    process_env: Dict[str, str]
    summary: Dict[str, Any]
    ci_plan: Optional[Dict[str, Any]] = None


def _new_or_empty(directory: Path, option: str) -> Path:
    directory = directory.resolve()
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ValueError(f"{option} must be new or empty")
    return directory


def make_run_paths(
    args: argparse.Namespace,
    *,
    repo: Path,
    mode: str,
    run_id: str,
    model_label: str,
) -> RunPaths:
    """Reserve names without creating directories or reparsing the request."""
    label = re.sub(r"[^A-Za-z0-9._-]", "_", model_label)
    repo = repo.resolve()
    result_root = _new_or_empty(
        args.result_root or repo / f"cache_grid_{mode}_results_{label}_{run_id}",
        "--result-root",
    )
    if mode == "local":
        return RunPaths(result_root, result_root, None)
    inputs_dir = _new_or_empty(
        repo / f"cache_grid_{mode}_inputs_{label}_{run_id}", "inputs directory"
    )
    generated_dir = _new_or_empty(
        repo / f"cache_grid_{mode}_generated_{label}_{run_id}",
        "generated directory",
    )
    return RunPaths(result_root, inputs_dir, generated_dir)


def build_local_plan(
    args: argparse.Namespace,
    *,
    repo: Path,
    run_id: str | None = None,
    machine: str | None = None,
    inherited: Mapping[str, str] | None = None,
) -> OrchestrationPlan:
    resolved = resolve_run_request(
        args.model_json,
        args.engine_json,
        args.grid_json,
        args.engine_env_json,
        args.runtime_env_json,
    )
    for category, values in (
        ("engine_env", resolved.engine_env),
        ("runtime_env", resolved.runtime_env),
    ):
        if "CUDA_VISIBLE_DEVICES" in values:
            raise ValueError(f"{category}.CUDA_VISIBLE_DEVICES is managed by --devices")
    bazel_config = LOCAL_PLATFORM_CONFIGS.get(machine or platform.machine())
    if bazel_config is None:
        supported = ", ".join(sorted(LOCAL_PLATFORM_CONFIGS))
        raise ValueError(
            f"unsupported local architecture; expected one of: {supported}"
        )
    devices = [value.strip() for value in args.devices.split(",")]
    if not devices or any(not value or not value.isdigit() for value in devices):
        raise ValueError("--devices must be a comma-separated list of GPU indices")
    if len(devices) != len(set(devices)):
        raise ValueError("--devices entries must be unique")
    if len(devices) != resolved.gpu_count:
        raise ValueError(
            f"topology requires {resolved.gpu_count} devices, got {len(devices)}"
        )
    paths = make_run_paths(
        args,
        repo=repo,
        mode="local",
        run_id=run_id if run_id is not None else uuid.uuid4().hex,
        model_label=resolved.model["label"],
    )
    input_bundle = ci.build_input_artifacts(
        to_prepare_namespace(
            resolved,
            inputs_dir=paths.inputs_dir,
            configs=[bazel_config],
            bazel=args.bazel,
        )
    )
    request_cases = dedupe_cache_grid_cases(
        input_bundle["grid_payload"]["cases"], resolved.grid["cache_alignment"]
    )
    runner_args = [
        f"--profile={paths.result_root / 'profile.snapshot.json'}",
        f"--cache_grid_json={paths.result_root / 'grid.snapshot.json'}",
        f"--result_dir={paths.result_root}",
        "--partial=2",
        "--decode_test_length=1",
        "--batch_size=1",
        "--cache_request_transport=dashsc_input_ids",
        f"--cache_measure_runs={resolved.grid['measure_runs']}",
        f"--cache_commit_tail_tokens={resolved.grid['commit_tail_tokens']}",
    ]
    effective_env = profile_environment(
        {"runtime_env": resolved.runtime_env, "engine_env": resolved.engine_env}
    )
    runner_args.extend(
        f"--engine_env={key}={value}" for key, value in sorted(effective_env.items())
    )
    command = [
        args.bazel,
        "run",
        cli.TARGET,
        f"--config={bazel_config}",
        "--run_under=//rtp_llm/test/utils:gpu_lock",
        *args.bazel_arg,
        "--",
        *runner_args,
    ]
    snapshot_bytes = {
        "profile.snapshot.json": input_bundle["artifacts"][ci.PROFILE_NAME],
        "grid.snapshot.json": input_bundle["artifacts"][ci.GRID_NAME],
    }
    launch = {
        "schema_version": 2,
        "mode": "run",
        "profile_file": "profile.snapshot.json",
        "grid": str(paths.result_root / "grid.snapshot.json"),
        "env": effective_env,
        "runtime_env": resolved.runtime_env,
        "engine_env": resolved.engine_env,
        "runner_args": [
            value for value in runner_args if not value.startswith("--engine_env=")
        ],
        "bazel": {
            "command": "run",
            "configs": [bazel_config],
            "args": list(args.bazel_arg),
            "executable": args.bazel,
        },
        "devices": devices,
        "snapshots": {
            name: hashlib.sha256(content).hexdigest()
            for name, content in snapshot_bytes.items()
        },
    }
    artifacts = {
        paths.result_root / name: content for name, content in snapshot_bytes.items()
    }
    artifacts[paths.result_root / cli.MANIFEST] = cli.encoded(launch)
    process_env = dict(os.environ if inherited is None else inherited)
    process_env.update(effective_env)
    process_env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    summary = {
        "mode": "local-run",
        "result_root": str(paths.result_root),
        "effective_case_count": len(request_cases),
        "gpu_count": resolved.gpu_count,
        "devices": devices,
        "engine_env": resolved.engine_env,
        "runtime_env": resolved.runtime_env,
        "commands": {"run": command},
    }
    return OrchestrationPlan(
        mode="local-run",
        paths=paths,
        resolved=resolved,
        request_cases=request_cases,
        artifacts=artifacts,
        commands={"run": command},
        process_env=process_env,
        summary=summary,
    )


def write_artifacts(artifacts: Mapping[Path, bytes]) -> None:
    """Preflight every destination, then exclusively create frozen inputs."""
    for path in artifacts:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"artifact already exists: {path}")
    for path, content in artifacts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(content)


def postprocess_namespace(result_dir: Path, *, partial: bool) -> argparse.Namespace:
    return argparse.Namespace(
        mode="pipeline",
        profile=None,
        grid=None,
        result_dir=result_dir,
        cases=None,
        runs=None,
        env=[],
        config=None,
        output_base=None,
        bazel=None,
        trace_timeout=180,
        profile_backend="kineto",
        nsys_path="nsys",
        nsys_session=None,
        nsys_tail_seconds=0.1,
        skip_reuse_validation=False,
        dry_run=False,
        test_mode="run",
        skip_test=True,
        batch_size=1,
        grid_dir=None,
        result_root=None,
        skip_fit=partial,
        partial=partial,
        cards=None,
        estimator="median",
        formula_output_dir=None,
        svg_output=None,
        cold_svg_output=None,
        html_output=None,
        oss_destination=None,
    )


def execute_local_plan(plan: OrchestrationPlan) -> int:
    write_artifacts(plan.artifacts)
    completed = subprocess.run(
        plan.commands["run"], cwd=REPO, env=plan.process_env, check=False
    )
    if completed.returncode:
        return completed.returncode
    return unified_pipeline.run(
        postprocess_namespace(result_dir=plan.paths.result_root, partial=False)
    )

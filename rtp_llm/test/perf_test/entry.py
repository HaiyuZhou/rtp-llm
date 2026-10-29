"""Unified perf-test command entry: strict input, planning, and dispatch."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, TextIO, Tuple

from rtp_llm.test.perf_test.capabilities import validate_capability
from rtp_llm.test.perf_test.cli import CliRequest, parse_cli
from rtp_llm.test.perf_test.config_resolver import ResolvedConfig, load_resolved_config
from rtp_llm.test.perf_test.config_schema import (
    DecodeDistributionWorkloadConfig,
    DecodeGridWorkloadConfig,
    PrefillWorkloadConfig,
    parse_config,
)
from rtp_llm.test.perf_test.decode.distribution import plan_decode_distribution_cases
from rtp_llm.test.perf_test.decode.planner import plan_decode_grid_cases
from rtp_llm.test.perf_test.dry_run import create_preview
from rtp_llm.test.perf_test.errors import ConfigurationError, PerfTestError
from rtp_llm.test.perf_test.orchestrator import Orchestrator
from rtp_llm.test.perf_test.plan import ExecutionPlan, build_execution_plan
from rtp_llm.test.perf_test.prefill.planner import plan_prefill_cases
from rtp_llm.test.perf_test.run_directory import (
    initialize_run_directory,
    inspect_run_directory,
)

ExecutorFactory = Callable[[], Mapping[str, Callable[[ExecutionPlan], Any]]]


def _run_id(mode: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{mode}-{now}"


def _source_config(path: str) -> Dict[str, Any]:
    source_path = Path(path)
    try:
        value = json.loads(source_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as error:
        raise ConfigurationError(f"config {source_path} is not UTF-8") from error
    except json.JSONDecodeError as error:
        raise ConfigurationError(
            f"config {source_path} is not valid JSON: {error.msg}"
        ) from error
    except OSError as error:
        raise ConfigurationError(
            f"cannot read config {source_path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise ConfigurationError(f"config {source_path} must contain an object")
    return value


def _planned_cases(
    mode: str, source: Mapping[str, Any], resolved: ResolvedConfig
) -> Tuple[Dict[str, object], ...]:
    config = parse_config(source)
    workload = config.workload
    if mode == "prefill":
        if not isinstance(workload, PrefillWorkloadConfig):
            raise ConfigurationError("--mode=prefill requires workload.kind=cache_grid")
        return plan_prefill_cases(workload, config.engine)
    if isinstance(workload, DecodeGridWorkloadConfig):
        return plan_decode_grid_cases(
            workload, config.engine, dp_size=config.model.topology.dp
        )
    if isinstance(workload, DecodeDistributionWorkloadConfig):
        return plan_decode_distribution_cases(
            resolved.resolved_config["workload"],
            config.engine,
            dp_size=config.model.topology.dp,
        )
    raise ConfigurationError(
        "--mode=decode requires workload.kind=grid or distribution"
    )


def _new_plan(request: CliRequest) -> tuple[ExecutionPlan, Dict[str, Any]]:
    assert request.config is not None
    source = _source_config(request.config)
    resolved = load_resolved_config(request.config)
    cases = _planned_cases(request.mode, source, resolved)
    root = Path(resolved.resolved_config["output"]["result_root"])
    result_directory = str(root / _run_id(request.mode))
    return (
        build_execution_plan(
            mode=request.mode,
            action=request.action,
            resolved=resolved,
            cases=cases,
            result_directory=result_directory,
        ),
        source,
    )


def _persisted_plan(request: CliRequest) -> tuple[ExecutionPlan, Dict[str, Any]]:
    assert request.run_dir is not None
    snapshot = inspect_run_directory(request.run_dir)
    plan = snapshot.execution_plan
    if request.mode != plan.mode:
        raise ConfigurationError(
            f"--mode={request.mode} does not match persisted mode {plan.mode}"
        )
    validate_capability(request.mode, request.action, plan.backend)
    if request.cases:
        known = {item.case_id for item in plan.cases}
        unknown = sorted(set(request.cases) - known)
        if unknown:
            raise ConfigurationError(f"unknown --cases value: {unknown[0]}")
    return replace(plan, action=request.action), snapshot.source_config


def build_request_plan(request: CliRequest) -> tuple[ExecutionPlan, Dict[str, Any]]:
    """Resolve a CLI request without mutating its result directory."""
    return _new_plan(request) if request.action == "run" else _persisted_plan(request)


def _unsupported_aone(_plan: ExecutionPlan) -> Any:
    raise ConfigurationError(
        "AOne submission requires an installed CI integration; generated shard "
        "targets use the unified AOne worker directly"
    )


def _default_executors(
    request: Optional[CliRequest] = None,
) -> Mapping[str, Callable[[ExecutionPlan], Any]]:
    from rtp_llm.test.perf_test.execution.defaults import create_default_local_executor

    selected = () if request is None else request.cases
    return {
        "local": create_default_local_executor(selected_case_ids=selected),
        "aone": _unsupported_aone,
    }


def run_request(
    request: CliRequest,
    *,
    executor_factory: ExecutorFactory = _default_executors,
) -> Any:
    """Plan first; keep dry-run outside every filesystem/execution side effect."""
    plan, source = build_request_plan(request)
    if request.dry_run:
        preview = create_preview(plan)
        if request.cases:
            preview["selected_case_ids"] = list(request.cases)
        return preview
    if request.action == "run":
        initialize_run_directory(
            plan.result_directory,
            source_config=source,
            plan=plan,
            run_id=Path(plan.result_directory).name,
            code_version=os.environ.get("BUILD_SCM_REVISION", "unknown"),
        )
    executors = (
        _default_executors(request)
        if executor_factory is _default_executors
        else executor_factory()
    )
    orchestrator = Orchestrator(executors)
    return orchestrator.execute(plan)


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    stdout: Optional[TextIO] = None,
    stderr: Optional[TextIO] = None,
    executor_factory: ExecutorFactory = _default_executors,
) -> int:
    output = sys.stdout if stdout is None else stdout
    errors = sys.stderr if stderr is None else stderr
    try:
        request = parse_cli(sys.argv[1:] if argv is None else argv)
        result = run_request(request, executor_factory=executor_factory)
        if isinstance(result, Mapping):
            output.write(json.dumps(result, sort_keys=True, indent=2) + "\n")
        exit_code = getattr(result, "exit_code", 0)
        return int(exit_code)
    except PerfTestError as error:
        errors.write(f"perf_test: {error}\n")
        return error.exit_code
    except (ValueError, OSError) as error:
        errors.write(f"perf_test: {error}\n")
        return 1

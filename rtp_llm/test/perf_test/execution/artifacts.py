"""Validate and import a gang master's artifacts without platform path knowledge."""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rtp_llm.test.perf_test.errors import (
    CaseExecutionError,
    CollectionError,
    PartialRunError,
    PerfTestError,
)
from rtp_llm.test.perf_test.plan import thaw
from rtp_llm.test.perf_test.results import parse_result_envelope
from rtp_llm.test.perf_test.state import RunState

MASTER_ARTIFACT_TREES = ("results", "reports", "traces", "profiles", "checkpoints")


@dataclass(frozen=True)
class RunCollection:
    state: RunState
    cases: tuple
    failure: Optional[PerfTestError] = None


def collect_master_artifacts(root, run_tree, *, plan, run_id):
    """Import validated output subtrees; controller plans/manifests stay intact."""
    root, run_tree = Path(root), Path(run_tree)
    envelope_path = run_tree / "results" / "result.json"
    if (
        run_tree.is_symlink()
        or (run_tree / "results").is_symlink()
        or envelope_path.is_symlink()
    ):
        raise CollectionError("artifact symlink is forbidden in master envelope path")
    if not envelope_path.is_file():
        raise CollectionError(
            f"gang master run tree is missing its result envelope: {envelope_path}"
        )
    try:
        envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
        parsed = parse_result_envelope(envelope)
    except (OSError, ValueError) as error:
        raise CollectionError(
            f"gang master result envelope is unreadable: {error}"
        ) from error
    expected = {
        "run_id": run_id,
        "mode": plan.mode,
        "backend": plan.backend,
        "config_fingerprint": plan.config_fingerprint,
        "workload_fingerprint": plan.workload_fingerprint,
    }
    for field, value in expected.items():
        if envelope.get(field) != value:
            raise CollectionError(f"gang master result {field} does not match the plan")
    planned = {case.case_id: thaw(case.identity) for case in plan.cases}
    if {case.case_id for case in parsed.cases} != set(planned) or len(
        parsed.cases
    ) != len(planned):
        raise CollectionError(
            "gang master results must cover every planned case exactly once"
        )
    for case in parsed.cases:
        if case.identity != planned[case.case_id]:
            raise CollectionError(
                f"gang master case identity does not match: {case.case_id}"
            )
    # Validate all paths before importing any file, including directory symlinks.
    sources = []
    for name in MASTER_ARTIFACT_TREES:
        source = run_tree / name
        if source.is_symlink():
            raise CollectionError(f"artifact symlink is forbidden: {source}")
        if not source.is_dir():
            continue
        for path in sorted(source.rglob("*")):
            if path.is_symlink():
                raise CollectionError(f"artifact symlink is forbidden: {path}")
            target = root / path.relative_to(run_tree)
            if any(
                parent.is_symlink()
                for parent in (target, *target.parents)
                if parent != root.parent
            ):
                raise CollectionError(
                    f"artifact destination symlink is forbidden: {target}"
                )
            if path.is_file():
                sources.append((path, target))
    for source, target in sources:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    cases = tuple(dict(case) for case in envelope.get("cases", []))
    if parsed.status == RunState.COMPLETED:
        return RunCollection(RunState.COMPLETED, cases)
    if parsed.status in {RunState.PARTIAL, RunState.INTERRUPTED}:
        return RunCollection(
            RunState.PARTIAL,
            cases,
            PartialRunError("remote gang run completed part of the plan"),
        )
    return RunCollection(
        RunState.FAILED, cases, CaseExecutionError("remote gang run failed")
    )


def copy_member_traces(source_directory, trace_root, member_id):
    """Copy actual Chrome traces, preserving paths and member provenance."""
    source_directory, trace_root = Path(source_directory), Path(trace_root)
    if (
        not isinstance(member_id, str)
        or not member_id
        or Path(member_id).name != member_id
        or member_id in {".", ".."}
    ):
        raise CollectionError("invalid trace member identity")
    if source_directory.is_symlink():
        raise CollectionError("trace source symlink is forbidden")
    for source in sorted(source_directory.rglob("*.json")):
        if source.is_symlink() or not source.is_file():
            continue
        try:
            value = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError):
            continue
        if not isinstance(value, dict) or not isinstance(
            value.get("traceEvents"), list
        ):
            continue
        target = trace_root / member_id / source.relative_to(source_directory)
        if any(path.is_symlink() for path in (target, *target.parents)):
            raise CollectionError("trace destination symlink is forbidden")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

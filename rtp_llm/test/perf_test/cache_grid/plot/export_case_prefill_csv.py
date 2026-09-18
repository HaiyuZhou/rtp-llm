#!/usr/bin/env python3
"""Export selected cache-grid cases and all prefill runs to one CSV row per case."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ResultSource:
    label: str
    path: Path
    started_at: str
    metrics: dict[int, dict[str, Any]]
    max_runs: int


def parse_case_ids(values: Iterable[str]) -> list[int]:
    """Parse comma- or whitespace-separated case IDs while preserving order."""
    case_ids: list[int] = []
    seen: set[int] = set()
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                case_id = int(token)
            except ValueError as error:
                raise ValueError(f"invalid case ID: {token!r}") from error
            if case_id < 0:
                raise ValueError(f"case ID must be non-negative: {case_id}")
            if case_id not in seen:
                seen.add(case_id)
                case_ids.append(case_id)
    if not case_ids:
        raise ValueError("at least one case ID is required")
    return case_ids


def _metrics_from_payload(payload: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        metrics = payload
    elif isinstance(payload, dict):
        metrics = payload.get("metrics", payload.get("results"))
    else:
        metrics = None
    if not isinstance(metrics, list):
        raise ValueError(f"{path}: expected a top-level metrics/results list")
    return [item for item in metrics if isinstance(item, dict)]


def _safe_label(value: str) -> str:
    label = re.sub(r"[^0-9A-Za-z_.-]+", "_", value).strip("_")
    return label or "replay"


def _replay_label(path: Path) -> str:
    """Return a compact label such as re_7797bd for retest_<hash>."""
    directory = path.parent.name
    match = re.search(r"(?:^|_)([0-9a-fA-F]{6,})(?:$|_)", directory)
    identifier = match.group(1)[:6].lower() if match else _safe_label(directory)[:6]
    return f"re_{identifier}"


def _load_source(path: Path, label: str, requested: set[int]) -> ResultSource:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to read {path}: {error}") from error

    metrics: dict[int, dict[str, Any]] = {}
    for item in _metrics_from_payload(payload, path):
        try:
            case_id = int(item.get("case_id"))
        except (TypeError, ValueError):
            continue
        if case_id not in requested:
            continue
        if case_id in metrics:
            raise ValueError(f"{path}: duplicate case_id {case_id}")
        metrics[case_id] = item

    max_runs = max(
        (
            len(item.get("runs", []))
            for item in metrics.values()
            if isinstance(item.get("runs"), list)
        ),
        default=0,
    )
    started_at = payload.get("started_at", "") if isinstance(payload, dict) else ""
    return ResultSource(
        label=_safe_label(label),
        path=path,
        started_at=str(started_at),
        metrics=metrics,
        max_runs=max_runs,
    )


def discover_sources(result_dir: Path, case_ids: list[int]) -> list[ResultSource]:
    """Load the root result and every replay result below cache_perf_replays."""
    if not result_dir.is_dir():
        raise ValueError(f"result directory does not exist: {result_dir}")

    requested = set(case_ids)
    original_path = result_dir / "cache_grid_results.json"
    sources: list[ResultSource] = []
    if original_path.is_file():
        sources.append(_load_source(original_path, "ori", requested))

    replay_root = result_dir / "cache_perf_replays"
    replay_sources: list[ResultSource] = []
    if replay_root.is_dir():
        for path in replay_root.rglob("cache_grid_results.json"):
            replay_sources.append(_load_source(path, _replay_label(path), requested))
    replay_sources.sort(key=lambda source: (source.started_at, source.label))
    labels = [source.label for source in replay_sources]
    duplicate_labels = sorted({label for label in labels if labels.count(label) > 1})
    if duplicate_labels:
        raise ValueError(
            "replay hash-prefix collision for CSV labels: "
            + ", ".join(duplicate_labels)
        )
    sources.extend(replay_sources)

    if not sources:
        raise ValueError(
            f"no cache_grid_results.json found in {result_dir} or cache_perf_replays"
        )
    return sources


def _integer(value: Any, field: str, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{context}: invalid {field} value {value!r}")
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context}: missing or invalid {field}: {value!r}") from error
    return number


def _consistent_integer(values: list[Any], field: str, context: str) -> int:
    numbers = {_integer(value, field, context) for value in values if value is not None}
    if not numbers:
        raise ValueError(f"{context}: no {field} value found")
    if len(numbers) != 1:
        raise ValueError(f"{context}: inconsistent {field} values: {sorted(numbers)}")
    return numbers.pop()


def _case_metadata(item: dict[str, Any], context: str) -> dict[str, Any]:
    runs = item.get("runs") if isinstance(item.get("runs"), list) else []
    run_dicts = [run for run in runs if isinstance(run, dict)]

    input_values = [item.get("input_len")]
    input_values.extend(run.get("input_len") for run in run_dicts)
    input_len = _consistent_integer(input_values, "input_len", context)

    reuse_values = [run.get("reuse_len") for run in run_dicts]
    observed = item.get("cache_len_observed")
    if isinstance(observed, list):
        reuse_values.extend(observed)
    elif observed is not None:
        reuse_values.append(observed)
    if not any(value is not None for value in reuse_values):
        reuse_values.extend(
            [item.get("expected_reuse_len"), item.get("cache_len_requested")]
        )
    reuse_len = _consistent_integer(reuse_values, "reuse_len", context)

    return {
        "case_key": str(item.get("case_key", "")),
        "batch_size": _integer(item.get("batch_size", 1), "batch_size", context),
        "input_len": input_len,
        "reuse_len": reuse_len,
        "compute_len": input_len - reuse_len,
    }


def _prefill_values(item: dict[str, Any], run_count: int) -> list[Any]:
    runs = item.get("runs") if isinstance(item.get("runs"), list) else []
    values: list[Any] = []
    for index in range(run_count):
        if index >= len(runs) or not isinstance(runs[index], dict):
            values.append("")
            continue
        run = runs[index]
        value = run.get("prefill_time_ms") if run.get("success", True) else None
        values.append("" if value is None else value)
    return values


def build_table(
    sources: list[ResultSource], case_ids: list[int]
) -> tuple[list[str], list[list[Any]], list[str]]:
    fixed_header = [
        "case_id",
        "case_key",
        "batch_size",
        "input_len",
        "reuse_len",
        "compute_len",
    ]
    run_header = [
        f"{source.label}_run{run_index}"
        for source in sources
        for run_index in range(source.max_runs)
    ]
    rows: list[list[Any]] = []
    warnings: list[str] = []

    for case_id in case_ids:
        metadata_by_source: list[tuple[ResultSource, dict[str, Any]]] = []
        for source in sources:
            item = source.metrics.get(case_id)
            if item is None:
                warnings.append(f"case {case_id} is missing from {source.path}")
                continue
            context = f"{source.path}: case {case_id}"
            metadata_by_source.append((source, _case_metadata(item, context)))

        if not metadata_by_source:
            raise ValueError(f"case {case_id} was not found in any result file")

        metadata = metadata_by_source[0][1]
        for source, candidate in metadata_by_source[1:]:
            if candidate != metadata:
                raise ValueError(
                    f"case {case_id}: metadata mismatch in {source.path}: "
                    f"{candidate!r} != {metadata!r}"
                )

        row: list[Any] = [case_id]
        row.extend(metadata[key] for key in fixed_header[1:])
        for source in sources:
            item = source.metrics.get(case_id)
            if item is None:
                row.extend([""] * source.max_runs)
            else:
                row.extend(_prefill_values(item, source.max_runs))
        rows.append(row)

    return fixed_header + run_header, rows, warnings


def export_csv(
    result_dir: Path, case_ids: list[int], output_path: Path
) -> tuple[int, list[str]]:
    sources = discover_sources(result_dir, case_ids)
    header, rows, warnings = build_table(sources, case_ids)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(header)
        writer.writerows(rows)
    return len(rows), warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export selected cases from a cache-grid result directory and all "
            "cache_perf_replays results to one CSV row per case."
        )
    )
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument(
        "--case-ids",
        required=True,
        nargs="+",
        help="Case IDs separated by spaces and/or commas.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path (default: RESULT_DIR/case_prefill_times.csv).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        case_ids = parse_case_ids(args.case_ids)
        output_path = args.output or args.result_dir / "case_prefill_times.csv"
        row_count, warnings = export_csv(args.result_dir, case_ids, output_path)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    print(f"wrote {row_count} cases to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

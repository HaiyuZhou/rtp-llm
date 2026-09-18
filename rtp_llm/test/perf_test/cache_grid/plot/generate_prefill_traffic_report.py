#!/usr/bin/env python3
"""Compare streamed production TTFT traffic with an offline prefill grid."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from rtp_llm.test.perf_test.cache_grid.plot.generate_prefill_interactive_chart import (
    representative_levels,
    representative_slice,
)

REQUIRED_CSV_COLUMNS = (
    "ds",
    "hh",
    "mm",
    "input_len",
    "reuse_len",
    "first_token_cost_time",
)


class CsvSchemaError(ValueError):
    pass


def finite_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def integer(value: Any) -> int | None:
    result = finite_number(value)
    return int(result) if result is not None and result.is_integer() else None


def quantile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _stable_seed(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass
class Reservoir:
    capacity: int
    seed: int
    key: str
    seen: int = 0
    values: list[float] = field(default_factory=list)
    _random: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._random = random.Random(_stable_seed(self.seed, self.key))

    @property
    def exact(self) -> bool:
        return self.seen <= self.capacity

    def add(self, value: float) -> None:
        self.seen += 1
        if len(self.values) < self.capacity:
            self.values.append(value)
            return
        slot = self._random.randrange(self.seen)
        if slot < self.capacity:
            self.values[slot] = value


@dataclass
class BucketAggregate:
    input_bucket: int
    reuse_bucket: int
    bucket_tokens: int
    reservoir_size: int
    seed: int
    count: int = 0
    input_sum: int = 0
    reuse_sum: int = 0
    ttft_sum_ms: float = 0.0
    reservoir: Reservoir = field(init=False)

    def __post_init__(self) -> None:
        self.reservoir = Reservoir(
            self.reservoir_size,
            self.seed,
            f"bucket:{self.input_bucket}:{self.reuse_bucket}",
        )

    def add(self, input_len: int, reuse_len: int, ttft_ms: float) -> None:
        self.count += 1
        self.input_sum += input_len
        self.reuse_sum += reuse_len
        self.ttft_sum_ms += ttft_ms
        self.reservoir.add(ttft_ms)

    def to_row(self, total_requests: int, total_input_tokens: int) -> dict[str, Any]:
        input_mean = self.input_sum / self.count
        reuse_mean = self.reuse_sum / self.count
        samples = self.reservoir.values
        return {
            "input_bucket_start": self.input_bucket,
            "input_bucket_end": self.input_bucket + self.bucket_tokens,
            "reuse_bucket_start": self.reuse_bucket,
            "reuse_bucket_end": self.reuse_bucket + self.bucket_tokens,
            "request_count": self.count,
            "input_token_sum": self.input_sum,
            "reuse_token_sum": self.reuse_sum,
            "compute_token_sum": self.input_sum - self.reuse_sum,
            "request_share": self.count / total_requests if total_requests else 0.0,
            "input_token_share": (
                self.input_sum / total_input_tokens if total_input_tokens else 0.0
            ),
            "mean_input_len": input_mean,
            "mean_reuse_len": reuse_mean,
            "mean_compute_len": input_mean - reuse_mean,
            "online_ttft_mean_ms": self.ttft_sum_ms / self.count,
            "online_ttft_p50_ms": quantile(samples, 0.50),
            "online_ttft_p95_ms": quantile(samples, 0.95),
            "online_ttft_p99_ms": quantile(samples, 0.99),
            "ttft_sample_count": len(samples),
            "ttft_quantiles_exact": self.reservoir.exact,
        }


@dataclass(frozen=True)
class BenchmarkPoint:
    compute_len: int
    cache_len: int
    offline_prefill_ms: float

    @property
    def input_len(self) -> int:
        return self.compute_len + self.cache_len


def bucket_start(value: int, bucket_tokens: int) -> int:
    return (value // bucket_tokens) * bucket_tokens


def _row_reason(row: dict[str, str | None]) -> tuple[int, int, float] | str:
    if any(not row.get(column) for column in REQUIRED_CSV_COLUMNS):
        return "missing_required_field"
    input_len = integer(row.get("input_len"))
    reuse_len = integer(row.get("reuse_len"))
    ttft_ms = finite_number(row.get("first_token_cost_time"))
    if input_len is None or reuse_len is None:
        return "non_integral_token_length"
    if ttft_ms is None:
        return "non_finite_ttft"
    if input_len < 0 or reuse_len < 0:
        return "negative_token_length"
    if reuse_len > input_len:
        return "reuse_exceeds_input"
    if ttft_ms <= 0:
        return "non_positive_ttft"
    return input_len, reuse_len, ttft_ms


def aggregate_production_csvs(
    paths: Iterable[Path], bucket_tokens: int, reservoir_size: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any], Reservoir]:
    if bucket_tokens <= 0:
        raise ValueError("bucket_tokens must be positive")
    if reservoir_size <= 0:
        raise ValueError("reservoir_size must be positive")

    buckets: dict[tuple[int, int], BucketAggregate] = {}
    source_audit: dict[str, dict[str, Any]] = {}
    global_reservoir = Reservoir(reservoir_size, seed, "global")
    total_requests = 0
    total_input_tokens = 0

    for path in paths:
        audit = {
            "path": str(path),
            "rows_total": 0,
            "rows_accepted": 0,
            "rows_rejected": 0,
            "rejected_by_reason": Counter(),
        }
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            header = reader.fieldnames or []
            missing = [
                column for column in REQUIRED_CSV_COLUMNS if column not in header
            ]
            if missing:
                raise CsvSchemaError(
                    f"{path}: missing required columns: {', '.join(missing)}"
                )
            for row in reader:
                audit["rows_total"] += 1
                parsed = _row_reason(row)
                if isinstance(parsed, str):
                    audit["rows_rejected"] += 1
                    audit["rejected_by_reason"][parsed] += 1
                    continue
                input_len, reuse_len, ttft_ms = parsed
                key = (
                    bucket_start(input_len, bucket_tokens),
                    bucket_start(reuse_len, bucket_tokens),
                )
                aggregate = buckets.get(key)
                if aggregate is None:
                    aggregate = BucketAggregate(
                        key[0], key[1], bucket_tokens, reservoir_size, seed
                    )
                    buckets[key] = aggregate
                aggregate.add(input_len, reuse_len, ttft_ms)
                global_reservoir.add(ttft_ms)
                audit["rows_accepted"] += 1
                total_requests += 1
                total_input_tokens += input_len
        audit["rejected_by_reason"] = dict(sorted(audit["rejected_by_reason"].items()))
        source_audit[str(path)] = audit

    rows = [
        aggregate.to_row(total_requests, total_input_tokens)
        for _, aggregate in sorted(buckets.items())
    ]
    summary = {
        "rows_total": sum(item["rows_total"] for item in source_audit.values()),
        "rows_accepted": total_requests,
        "rows_rejected": sum(item["rows_rejected"] for item in source_audit.values()),
        "input_token_sum": total_input_tokens,
        "bucket_count": len(rows),
        "source_files": source_audit,
    }
    return rows, summary, global_reservoir


def _metric_status_ok(item: dict[str, Any]) -> bool:
    status = str(item.get("status", "")).lower()
    return not status or status in {"ok", "success", "passed"}


def _benchmark_metric(
    item: dict[str, Any], batch_size: int
) -> tuple[BenchmarkPoint | None, str | None]:
    if integer(item.get("batch_size", 1)) != batch_size:
        return None, "different_batch_size"
    if not _metric_status_ok(item):
        return None, "status_not_ok"
    input_len = integer(item.get("input_len", item.get("seq_len")))
    requested_cache = integer(item.get("cache_len_requested", item.get("cache_len", 0)))
    expected_runs = integer(item.get("measure_runs")) or 3
    success_runs = integer(item.get("success_runs"))
    runs = item.get("runs")
    if input_len is None or requested_cache is None:
        return None, "missing_geometry"
    if input_len < 0 or requested_cache < 0 or requested_cache > input_len:
        return None, "invalid_geometry"
    if item.get("reuse_exact") is False and not item.get(
        "reuse_validation_skipped", False
    ):
        return None, "reuse_not_exact"
    if not isinstance(runs, list) or len(runs) != expected_runs:
        return None, "incomplete_runs"
    if success_runs != expected_runs:
        return None, "success_runs_mismatch"
    observed = item.get("cache_len_observed")
    observed_values = (
        [integer(value) for value in observed] if isinstance(observed, list) else []
    )
    observed_values = [value for value in observed_values if value is not None]
    if len(observed_values) != expected_runs:
        observed_values = []
        for run in runs:
            if not isinstance(run, dict):
                return None, "invalid_run"
            value = integer(run.get("reuse_len"))
            if value is None:
                return None, "missing_observed_reuse"
            observed_values.append(value)
    if len(set(observed_values)) != 1:
        return None, "observed_reuse_not_constant"
    cache_len = observed_values[0]
    if cache_len != requested_cache:
        return None, "requested_reuse_mismatch"

    latencies: list[float] = []
    for run in runs:
        if not isinstance(run, dict) or run.get("success") is not True:
            return None, "run_failed"
        if integer(run.get("input_len")) != input_len:
            return None, "run_input_mismatch"
        if integer(run.get("reuse_len")) != cache_len:
            return None, "run_reuse_mismatch"
        latency = finite_number(run.get("prefill_time_ms"))
        if latency is None or latency <= 0:
            return None, "invalid_prefill_time"
        latencies.append(latency)
    return BenchmarkPoint(input_len - cache_len, cache_len, median(latencies)), None


def load_benchmark(
    path: Path, batch_size: int
) -> tuple[list[BenchmarkPoint], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("metrics") if isinstance(data, dict) else None
    if not isinstance(metrics, list):
        raise ValueError(f"{path}: expected a JSON object containing metrics[]")
    grouped: defaultdict[tuple[int, int], list[float]] = defaultdict(list)
    rejected: Counter[str] = Counter()
    selected = 0
    for item in metrics:
        if not isinstance(item, dict):
            rejected["invalid_metric"] += 1
            continue
        point, reason = _benchmark_metric(item, batch_size)
        if reason == "different_batch_size":
            continue
        selected += 1
        if point is None:
            rejected[reason or "invalid_metric"] += 1
            continue
        grouped[(point.compute_len, point.cache_len)].append(point.offline_prefill_ms)
    points = [
        BenchmarkPoint(compute, cache, median(values))
        for (compute, cache), values in sorted(grouped.items())
    ]
    audit = {
        "path": str(path),
        "batch_size": batch_size,
        "metrics_selected_batch": selected,
        "geometries_accepted": len(points),
        "metrics_rejected": sum(rejected.values()),
        "rejected_by_reason": dict(sorted(rejected.items())),
    }
    return points, audit


def _benchmark_index(
    points: list[BenchmarkPoint], cell_size: int
) -> dict[tuple[int, int], list[BenchmarkPoint]]:
    index: dict[tuple[int, int], list[BenchmarkPoint]] = defaultdict(list)
    for point in points:
        index[(point.compute_len // cell_size, point.cache_len // cell_size)].append(
            point
        )
    return index


def map_buckets_to_benchmark(
    rows: list[dict[str, Any]], points: list[BenchmarkPoint], max_distance_tokens: int
) -> list[dict[str, Any]]:
    if max_distance_tokens <= 0:
        raise ValueError("max_map_distance_tokens must be positive")
    index = _benchmark_index(points, max_distance_tokens)
    mapped: list[dict[str, Any]] = []
    for row in rows:
        compute = float(row["mean_compute_len"])
        cache = float(row["mean_reuse_len"])
        compute_cell = math.floor(compute / max_distance_tokens)
        cache_cell = math.floor(cache / max_distance_tokens)
        candidates = [
            point
            for compute_offset in (-1, 0, 1)
            for cache_offset in (-1, 0, 1)
            for point in index.get(
                (compute_cell + compute_offset, cache_cell + cache_offset), []
            )
        ]
        selected: BenchmarkPoint | None = None
        if candidates:
            selected = min(
                candidates,
                key=lambda point: (
                    (point.compute_len - compute) ** 2 + (point.cache_len - cache) ** 2,
                    point.compute_len,
                    point.cache_len,
                ),
            )
        enriched = dict(row)
        if selected is None:
            enriched.update(
                {
                    "covered": False,
                    "matched_compute_len": None,
                    "matched_cache_len": None,
                    "compute_distance_tokens": None,
                    "cache_distance_tokens": None,
                    "offline_prefill_ms": None,
                    "online_p50_minus_offline_ms": None,
                    "online_p95_minus_offline_ms": None,
                    "online_p99_minus_offline_ms": None,
                }
            )
        else:
            compute_distance = abs(compute - selected.compute_len)
            cache_distance = abs(cache - selected.cache_len)
            covered = (
                compute_distance <= max_distance_tokens
                and cache_distance <= max_distance_tokens
            )
            offline = selected.offline_prefill_ms if covered else None
            enriched.update(
                {
                    "covered": covered,
                    "matched_compute_len": selected.compute_len if covered else None,
                    "matched_cache_len": selected.cache_len if covered else None,
                    "compute_distance_tokens": compute_distance if covered else None,
                    "cache_distance_tokens": cache_distance if covered else None,
                    "offline_prefill_ms": offline,
                    "online_p50_minus_offline_ms": (
                        row["online_ttft_p50_ms"] - offline
                        if offline is not None
                        else None
                    ),
                    "online_p95_minus_offline_ms": (
                        row["online_ttft_p95_ms"] - offline
                        if offline is not None
                        else None
                    ),
                    "online_p99_minus_offline_ms": (
                        row["online_ttft_p99_ms"] - offline
                        if offline is not None
                        else None
                    ),
                }
            )
        mapped.append(enriched)
    return mapped


def _weighted_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    entries = [
        (float(row[key]), int(row["request_count"]))
        for row in rows
        if row.get(key) is not None
    ]
    denominator = sum(weight for _, weight in entries)
    if not denominator:
        return None
    return sum(value * weight for value, weight in entries) / denominator


def summarize(
    rows: list[dict[str, Any]],
    production_audit: dict[str, Any],
    global_reservoir: Reservoir,
) -> dict[str, Any]:
    covered = [row for row in rows if row["covered"]]
    total_requests = production_audit["rows_accepted"]
    total_input_tokens = production_audit["input_token_sum"]
    covered_requests = sum(int(row["request_count"]) for row in covered)
    covered_input_tokens = sum(int(row["input_token_sum"]) for row in covered)
    top_request = sorted(
        rows,
        key=lambda row: (
            -int(row["request_count"]),
            row["input_bucket_start"],
            row["reuse_bucket_start"],
        ),
    )[:10]
    top_traffic = sorted(
        rows,
        key=lambda row: (
            -int(row["input_token_sum"]),
            row["input_bucket_start"],
            row["reuse_bucket_start"],
        ),
    )[:10]
    top_impact = sorted(
        rows,
        key=lambda row: (
            -(float(row["request_share"]) * float(row["online_ttft_p95_ms"])),
            row["input_bucket_start"],
            row["reuse_bucket_start"],
        ),
    )[:10]
    samples = global_reservoir.values
    return {
        "coverage": {
            "covered_requests": covered_requests,
            "uncovered_requests": total_requests - covered_requests,
            "covered_request_share": (
                covered_requests / total_requests if total_requests else 0.0
            ),
            "covered_input_tokens": covered_input_tokens,
            "uncovered_input_tokens": total_input_tokens - covered_input_tokens,
            "covered_input_token_share": (
                covered_input_tokens / total_input_tokens if total_input_tokens else 0.0
            ),
        },
        "global_online_ttft": {
            "p50_ms": quantile(samples, 0.50),
            "p95_ms": quantile(samples, 0.95),
            "p99_ms": quantile(samples, 0.99),
            "sample_count": len(samples),
            "quantiles_exact": global_reservoir.exact,
        },
        "request_weighted_bucket_metrics": {
            "online_p50_ms": _weighted_mean(rows, "online_ttft_p50_ms"),
            "online_p95_ms": _weighted_mean(rows, "online_ttft_p95_ms"),
            "covered_online_p50_ms": _weighted_mean(covered, "online_ttft_p50_ms"),
            "covered_online_p95_ms": _weighted_mean(covered, "online_ttft_p95_ms"),
            "covered_offline_engine_prefill_ms": _weighted_mean(
                covered, "offline_prefill_ms"
            ),
            "covered_p50_gap_ms": _weighted_mean(
                covered, "online_p50_minus_offline_ms"
            ),
            "covered_p95_gap_ms": _weighted_mean(
                covered, "online_p95_minus_offline_ms"
            ),
        },
        "top_by_request_count": top_request,
        "top_by_input_token_traffic": top_traffic,
        "top_by_request_share_times_p95": top_impact,
    }


def _fmt_number(value: float | int | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value:,.{digits}f}"


def _fmt_percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


def _density_color(value: int, maximum: int) -> str:
    palette = ("#eff6ff", "#bfdbfe", "#60a5fa", "#2563eb", "#1e3a8a")
    if maximum <= 1:
        return palette[0]
    level = math.log1p(value) / math.log1p(maximum)
    return palette[min(len(palette) - 1, round(level * (len(palette) - 1)))]


def _traffic_mesh(
    rows: list[dict[str, Any]], z_base: float
) -> tuple[dict[str, list[Any]], list[list[Any]]]:
    maximum = max((int(row["request_count"]) for row in rows), default=1)
    vertices: dict[str, list[Any]] = {
        "x": [],
        "y": [],
        "z": [],
        "i": [],
        "j": [],
        "k": [],
        "facecolor": [],
    }
    hover_rows: list[list[Any]] = []
    for index, row in enumerate(rows):
        input_low = int(row["input_bucket_start"])
        input_high = int(row["input_bucket_end"])
        reuse_low = int(row["reuse_bucket_start"])
        reuse_high = int(row["reuse_bucket_end"])
        base = index * 4
        corners = (
            (input_low - reuse_low, reuse_low),
            (input_high - reuse_low, reuse_low),
            (input_high - reuse_high, reuse_high),
            (input_low - reuse_high, reuse_high),
        )
        vertices["x"].extend(corner[0] for corner in corners)
        vertices["y"].extend(corner[1] for corner in corners)
        vertices["z"].extend([z_base] * 4)
        vertices["i"].extend((base, base))
        vertices["j"].extend((base + 1, base + 2))
        vertices["k"].extend((base + 2, base + 3))
        color = _density_color(int(row["request_count"]), maximum)
        vertices["facecolor"].extend((color, color))
        hover_rows.append(
            [
                row["input_bucket_start"],
                row["input_bucket_end"],
                row["reuse_bucket_start"],
                row["reuse_bucket_end"],
                row["request_count"],
                row["request_share"],
                row["input_token_share"],
                row["online_ttft_p50_ms"],
                row["online_ttft_p95_ms"],
                row["online_ttft_p99_ms"],
                "已覆盖" if row["covered"] else "未覆盖",
                row["offline_prefill_ms"],
            ]
        )
    return vertices, hover_rows


def render_report(
    output: Path,
    benchmark: list[BenchmarkPoint],
    rows: list[dict[str, Any]],
    production_audit: dict[str, Any],
    benchmark_audit: dict[str, Any],
    decision_summary: dict[str, Any],
    settings: dict[str, Any],
) -> None:
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Plotly is required. Install it with: python3 -m pip install --user plotly"
        ) from error

    benchmark_rows = [
        {
            "compute_len": float(point.compute_len),
            "cache_len": float(point.cache_len),
            "prefill_rt": point.offline_prefill_ms,
        }
        for point in benchmark
    ]
    offline_values = [point.offline_prefill_ms for point in benchmark]
    z_min = min(offline_values)
    z_max = max(offline_values)
    z_base = z_min - max(1.0, (z_max - z_min) * 0.07)
    mesh, hover_rows = _traffic_mesh(rows, z_base)

    traces: list[Any] = []
    cache_levels = representative_levels(benchmark_rows, "cache_len")
    compute_levels = representative_levels(benchmark_rows, "compute_len")
    warm_palette = ("#b42318", "#d97706", "#15803d", "#0369a1")
    cool_palette = ("#7c3aed", "#c026d3", "#0369a1", "#0f766e")
    for index, level in enumerate(cache_levels):
        subset = representative_slice(benchmark_rows, "cache_len", level, "compute_len")
        traces.append(
            go.Scatter3d(
                x=[row["compute_len"] for row in subset],
                y=[row["cache_len"] for row in subset],
                z=[row["prefill_rt"] for row in subset],
                mode="lines",
                line={"color": warm_palette[index % len(warm_palette)], "width": 5},
                name=f"离线趋势：cache={level:,.0f}",
                hoverinfo="skip",
            )
        )
    for index, level in enumerate(compute_levels):
        subset = representative_slice(benchmark_rows, "compute_len", level, "cache_len")
        traces.append(
            go.Scatter3d(
                x=[row["compute_len"] for row in subset],
                y=[row["cache_len"] for row in subset],
                z=[row["prefill_rt"] for row in subset],
                mode="lines",
                line={
                    "color": cool_palette[index % len(cool_palette)],
                    "width": 4,
                    "dash": "dash",
                },
                name=f"离线趋势：compute={level:,.0f}",
                hoverinfo="skip",
            )
        )
    traces.extend(
        [
            go.Mesh3d(
                **mesh,
                flatshading=True,
                opacity=0.82,
                hoverinfo="skip",
                name="线上请求密度（底面）",
                showlegend=True,
            ),
            go.Scatter3d(
                x=[row["mean_compute_len"] for row in rows],
                y=[row["mean_reuse_len"] for row in rows],
                z=[z_base] * len(rows),
                mode="markers",
                marker={"size": 2, "color": "rgba(0,0,0,0.02)"},
                customdata=hover_rows,
                hovertemplate=(
                    "线上 input 桶: %{customdata[0]:,.0f}–%{customdata[1]:,.0f}<br>"
                    "线上 reuse 桶: %{customdata[2]:,.0f}–%{customdata[3]:,.0f}<br>"
                    "请求数: %{customdata[4]:,.0f}<br>"
                    "请求占比: %{customdata[5]:.2%}<br>"
                    "input-token 占比: %{customdata[6]:.2%}<br>"
                    "线上 TTFT p50 / p95 / p99: %{customdata[7]:.1f} / %{customdata[8]:.1f} / %{customdata[9]:.1f} ms<br>"
                    "网格映射: %{customdata[10]}<br>"
                    "离线引擎 prefill: %{customdata[11]:.1f} ms<extra></extra>"
                ),
                name="线上桶详情",
                showlegend=False,
            ),
            go.Scatter3d(
                x=[point.compute_len for point in benchmark],
                y=[point.cache_len for point in benchmark],
                z=offline_values,
                mode="markers",
                marker={
                    "size": 2.2,
                    "color": offline_values,
                    "colorscale": "Viridis",
                    "reversescale": True,
                    "colorbar": {"title": {"text": "离线引擎 prefill (ms，深色更慢)"}},
                    "opacity": 0.8,
                },
                customdata=[[point.input_len] for point in benchmark],
                hovertemplate=(
                    "Compute tokens: %{x:,.0f}<br>"
                    "Cached tokens: %{y:,.0f}<br>"
                    "离线引擎 prefill: %{z:.2f} ms<br>"
                    "Input length: %{customdata[0]:,.0f}<extra></extra>"
                ),
                name="离线 benchmark 点",
                showlegend=False,
            ),
        ]
    )
    figure_3d = go.Figure(data=traces)
    figure_3d.update_layout(
        title="离线引擎 prefill 基准面 + 线上请求密度",
        scene={
            "xaxis_title": "非缓存 compute tokens (X)",
            "yaxis_title": "缓存 reuse tokens (Y)",
            "zaxis_title": "离线引擎 prefill (ms，Z)",
            "aspectmode": "manual",
            "aspectratio": {"x": 1, "y": 1, "z": 0.65},
        },
        legend={"x": 0.01, "y": 0.99},
        margin={"l": 0, "r": 0, "b": 0, "t": 50},
    )

    input_buckets = sorted({int(row["input_bucket_start"]) for row in rows})
    reuse_buckets = sorted({int(row["reuse_bucket_start"]) for row in rows})
    by_bucket = {
        (int(row["input_bucket_start"]), int(row["reuse_bucket_start"])): row
        for row in rows
    }
    z_matrix: list[list[float | None]] = []
    custom_matrix: list[list[list[Any] | None]] = []
    for reuse in reuse_buckets:
        z_line: list[float | None] = []
        custom_line: list[list[Any] | None] = []
        for input_bucket in input_buckets:
            row = by_bucket.get((input_bucket, reuse))
            z_line.append(None if row is None else row["online_ttft_p95_ms"])
            custom_line.append(
                None
                if row is None
                else [
                    row["request_count"],
                    row["request_share"],
                    row["input_token_share"],
                    row["online_ttft_p50_ms"],
                    row["online_ttft_p95_ms"],
                    row["online_ttft_p99_ms"],
                    "已覆盖" if row["covered"] else "未覆盖",
                    row["offline_prefill_ms"],
                    row["online_p95_minus_offline_ms"],
                ]
            )
        z_matrix.append(z_line)
        custom_matrix.append(custom_line)
    figure_2d = go.Figure(
        data=[
            go.Heatmap(
                x=input_buckets,
                y=reuse_buckets,
                z=z_matrix,
                customdata=custom_matrix,
                colorscale="YlOrRd",
                colorbar={"title": "线上 TTFT p95 (ms)"},
                hovertemplate=(
                    "input 桶: %{x:,.0f}<br>"
                    "reuse 桶: %{y:,.0f}<br>"
                    "请求数: %{customdata[0]:,.0f}<br>"
                    "请求占比: %{customdata[1]:.2%}<br>"
                    "input-token 占比: %{customdata[2]:.2%}<br>"
                    "线上 TTFT p50 / p95 / p99: %{customdata[3]:.1f} / %{customdata[4]:.1f} / %{customdata[5]:.1f} ms<br>"
                    "网格映射: %{customdata[6]}<br>"
                    "离线引擎 prefill: %{customdata[7]:.1f} ms<br>"
                    "线上 p95 - 离线 prefill: %{customdata[8]:.1f} ms<extra></extra>"
                ),
            )
        ]
    )
    figure_2d.update_layout(
        title="线上真实体验：按 input/reuse 桶聚合的 TTFT p95",
        xaxis_title="Input bucket 起点 (tokens)",
        yaxis_title="Reuse bucket 起点 (tokens)",
        margin={"l": 65, "r": 20, "b": 65, "t": 50},
    )

    coverage = decision_summary["coverage"]
    global_ttft = decision_summary["global_online_ttft"]
    weighted = decision_summary["request_weighted_bucket_metrics"]
    cards = [
        ("线上有效请求", f"{production_audit['rows_accepted']:,}"),
        ("聚合热点桶", f"{production_audit['bucket_count']:,}"),
        ("离线网格请求覆盖", _fmt_percent(coverage["covered_request_share"])),
        ("离线网格 token 覆盖", _fmt_percent(coverage["covered_input_token_share"])),
        ("线上 TTFT p50", f"{_fmt_number(global_ttft['p50_ms'])} ms"),
        ("线上 TTFT p95", f"{_fmt_number(global_ttft['p95_ms'])} ms"),
        (
            "覆盖区域线上 p95 - 离线 prefill",
            f"{_fmt_number(weighted['covered_p95_gap_ms'])} ms",
        ),
    ]
    cards_html = "".join(
        f"<section class='card'><div>{html.escape(label)}</div><strong>{html.escape(value)}</strong></section>"
        for label, value in cards
    )

    def table_html(title: str, values: list[dict[str, Any]]) -> str:
        body = "".join(
            "<tr>"
            f"<td>{row['input_bucket_start']:,}–{row['input_bucket_end']:,}</td>"
            f"<td>{row['reuse_bucket_start']:,}–{row['reuse_bucket_end']:,}</td>"
            f"<td>{row['request_count']:,}</td>"
            f"<td>{_fmt_percent(row['request_share'])}</td>"
            f"<td>{_fmt_number(row['online_ttft_p95_ms'])}</td>"
            f"<td>{'已覆盖' if row['covered'] else '未覆盖'}</td>"
            f"<td>{_fmt_number(row['offline_prefill_ms'])}</td>"
            "</tr>"
            for row in values
        )
        return (
            f"<h3>{html.escape(title)}</h3>"
            "<div class='table-wrap'><table><thead><tr>"
            "<th>input 桶</th><th>reuse 桶</th><th>请求数</th><th>请求占比</th>"
            "<th>线上 TTFT p95 (ms)</th><th>网格映射</th><th>离线引擎 prefill (ms)</th>"
            f"</tr></thead><tbody>{body}</tbody></table></div>"
        )

    source_rows = "".join(
        "<tr>"
        f"<td>{html.escape(Path(path).name)}</td>"
        f"<td>{audit['rows_total']:,}</td><td>{audit['rows_accepted']:,}</td>"
        f"<td>{audit['rows_rejected']:,}</td>"
        f"<td>{html.escape(json.dumps(audit['rejected_by_reason'], ensure_ascii=False))}</td>"
        "</tr>"
        for path, audit in production_audit["source_files"].items()
    )
    source_audit_html = (
        "<h3>线上 CSV 行审计</h3><div class='table-wrap'><table><thead><tr>"
        "<th>文件</th><th>总行数</th><th>有效行</th><th>跳过行</th><th>跳过原因</th>"
        f"</tr></thead><tbody>{source_rows}</tbody></table></div>"
    )
    settings_text = html.escape(json.dumps(settings, ensure_ascii=False, indent=2))
    benchmark_text = html.escape(
        json.dumps(benchmark_audit, ensure_ascii=False, indent=2)
    )
    figure_3d_html = figure_3d.to_html(include_plotlyjs=True, full_html=False)
    figure_2d_html = figure_2d.to_html(include_plotlyjs=False, full_html=False)
    report = f"""<!doctype html>
<html lang='zh-CN'>
<head>
<meta charset='utf-8'>
<title>线上流量与离线 prefill 基准对比</title>
<style>
body {{ font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; margin: 24px; color: #172033; background: #f8fafc; }}
h1,h2,h3 {{ color: #0f172a; }}
.note {{ padding: 12px 16px; background: #e0f2fe; border-left: 4px solid #0284c7; border-radius: 4px; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(190px,1fr)); gap: 12px; margin: 16px 0; }}
.card {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; }}
.card div {{ color: #475569; font-size: 13px; }} .card strong {{ display: block; margin-top: 6px; font-size: 21px; }}
.panel {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 8px; margin: 18px 0; }}
.table-wrap {{ max-height: 420px; overflow: auto; background: white; border: 1px solid #e2e8f0; border-radius: 8px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }} th,td {{ padding: 8px; text-align: right; border-bottom: 1px solid #e2e8f0; white-space: nowrap; }} th {{ position: sticky; top: 0; background: #f1f5f9; }} th:first-child, td:first-child {{ text-align: left; }}
pre {{ overflow:auto; background:#0f172a; color:#e2e8f0; padding:14px; border-radius:8px; }}
</style>
</head>
<body>
<h1>线上流量与离线 prefill 基准对比</h1>
<p class='note'>线上指标是 <code>first_token_cost_time</code>（含线上调度、排队、网络等开销）；离线指标是引擎侧 <code>prefill_time_ms</code>。两者的差值表示观测到的线上额外开销，不代表单独的引擎性能回归。未被离线网格覆盖的线上桶不会做插值。</p>
<div class='cards'>{cards_html}</div>
<div class='panel'>{figure_3d_html}</div>
<div class='panel'>{figure_2d_html}</div>
{source_audit_html}
{table_html('Top 10：按请求数', decision_summary['top_by_request_count'])}
{table_html('Top 10：按 input-token 流量', decision_summary['top_by_input_token_traffic'])}
{table_html('Top 10：按请求占比 × 线上 p95', decision_summary['top_by_request_share_times_p95'])}
<h3>报告参数</h3><pre>{settings_text}</pre>
<h3>离线 benchmark 审计</h3><pre>{benchmark_text}</pre>
</body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream production TTFT CSVs and compare them with an offline prefill grid."
    )
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--production-csv", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--traffic-json", required=True, type=Path)
    parser.add_argument("--bucket-tokens", type=int, default=1024)
    parser.add_argument("--benchmark-batch-size", type=int, default=1)
    parser.add_argument("--max-map-distance-tokens", type=int, default=1024)
    parser.add_argument("--reservoir-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()

    rows, production_audit, global_reservoir = aggregate_production_csvs(
        args.production_csv, args.bucket_tokens, args.reservoir_size, args.seed
    )
    benchmark, benchmark_audit = load_benchmark(
        args.benchmark, args.benchmark_batch_size
    )
    if not benchmark:
        raise SystemExit("no valid offline benchmark geometries")
    rows = map_buckets_to_benchmark(rows, benchmark, args.max_map_distance_tokens)
    settings = {
        "benchmark": str(args.benchmark),
        "production_csv": [str(path) for path in args.production_csv],
        "bucket_tokens": args.bucket_tokens,
        "benchmark_batch_size": args.benchmark_batch_size,
        "max_map_distance_tokens": args.max_map_distance_tokens,
        "reservoir_size": args.reservoir_size,
        "reservoir_seed": args.seed,
        "online_latency_field": "first_token_cost_time",
        "offline_latency_field": "runs[].prefill_time_ms",
    }
    decision_summary = summarize(rows, production_audit, global_reservoir)
    traffic_payload = {
        "schema_version": 1,
        "kind": "production_prefill_traffic_histogram",
        "settings": settings,
        "production_audit": production_audit,
        "benchmark_audit": benchmark_audit,
        "buckets": rows,
    }
    summary_payload = {
        "schema_version": 1,
        "kind": "production_prefill_traffic_comparison",
        "settings": settings,
        "production_audit": production_audit,
        "benchmark_audit": benchmark_audit,
        "quantiles": {
            "method": "linear_interpolation_on_deterministic_reservoir",
            "reservoir_size": args.reservoir_size,
            "seed": args.seed,
        },
        "decision_summary": decision_summary,
        "traffic_histogram": str(args.traffic_json),
    }
    write_json(args.traffic_json, traffic_payload)
    write_json(args.summary_json, summary_payload)
    render_report(
        args.output,
        benchmark,
        rows,
        production_audit,
        benchmark_audit,
        decision_summary,
        settings,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "summary_json": str(args.summary_json),
                "traffic_json": str(args.traffic_json),
                "accepted_rows": production_audit["rows_accepted"],
                "rejected_rows": production_audit["rows_rejected"],
                "traffic_buckets": len(rows),
                "benchmark_geometries": len(benchmark),
                "covered_request_share": decision_summary["coverage"][
                    "covered_request_share"
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

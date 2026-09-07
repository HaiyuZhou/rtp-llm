#!/usr/bin/env python3
"""Generate an interactive, rotatable 3D prefill chart from a result JSON.

The chart mirrors the static SVG's convention:
  X = uncached compute tokens = input_len - observed_cache_len
  Y = observed cached tokens
  Z = measured prefill RT / TTFT (ms)

Every valid geometry is shown as a dot.  Coloured lines are representative
fixed-cache (warm, solid) and fixed-compute (cool, dashed) trend guides.
Duplicate geometries are collapsed by median RT.
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from rtp_llm.test.perf_test.perf_profile import extract_embedded_profile
from rtp_llm.test.perf_test.perf_profile import fingerprint as profile_fingerprint
from rtp_llm.test.perf_test.perf_profile import (
    load_profile,
    resolve_label,
    resolve_title,
)


def number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def consistent_value(values: list[float]) -> float | None:
    return values[0] if values and len(set(values)) == 1 else None


def observed_cache_len(item: dict[str, Any]) -> float | None:
    observed = item.get("cache_len_observed")
    if isinstance(observed, list):
        values = [number(value) for value in observed if value is not None]
        return consistent_value(values)
    if observed is not None:
        return number(observed)

    runs = item.get("runs")
    if isinstance(runs, list):
        values = [number(run.get("reuse_len")) for run in runs if isinstance(run, dict)]
        values = [value for value in values if value is not None]
        return consistent_value(values)
    return None


def prefill_rt(item: dict[str, Any]) -> float | None:
    for key in (
        "avg_prefill_time",
        "target_ms",
        "ttft_ms",
        "prefill_time_ms",
        "prefill_ms",
    ):
        value = number(item.get(key))
        if value is not None:
            return value

    runs = item.get("runs")
    if isinstance(runs, list):
        values = [
            number(run.get("prefill_time_ms"))
            for run in runs
            if isinstance(run, dict) and run.get("success", True)
        ]
        values = [value for value in values if value is not None]
        if values:
            return median(values)
    return None


def load_rows(input_path: Path, batch_size: int) -> list[dict[str, float]]:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    metrics = (
        data if isinstance(data, list) else data.get("metrics", data.get("results", []))
    )
    rows: list[dict[str, float]] = []

    for item in metrics:
        if not isinstance(item, dict):
            continue
        if int(number(item.get("batch_size", 1)) or 1) != batch_size:
            continue
        status = str(item.get("status", "")).lower()
        if status and status not in {
            "ok",
            "success",
            "passed",
            "unknown",
            "invalid_reuse",
        }:
            continue

        input_len = number(item.get("input_len", item.get("seq_len")))
        cache_len = observed_cache_len(item)
        rt = prefill_rt(item)
        if (
            input_len is None
            or cache_len is None
            or rt is None
            or input_len < 0
            or cache_len < 0
            or cache_len > input_len
        ):
            continue

        rows.append(
            {
                "input_len": input_len,
                "cache_len": cache_len,
                "compute_len": input_len - cache_len,
                "prefill_rt": rt,
            }
        )

    grouped: defaultdict[tuple[float, float, float], list[dict[str, float]]] = (
        defaultdict(list)
    )
    for row in rows:
        grouped[(row["input_len"], row["cache_len"], row["compute_len"])].append(row)
    deduped = [
        {**values[0], "prefill_rt": median(item["prefill_rt"] for item in values)}
        for _, values in sorted(grouped.items())
    ]
    return deduped


def top_levels(rows: list[dict[str, float]], key: str, limit: int = 8) -> list[float]:
    counts = Counter(row[key] for row in rows)
    return sorted(
        level
        for level, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[
            :limit
        ]
    )


def line_trace(
    rows: list[dict[str, float]],
    fixed_key: str,
    fixed_value: float,
    varying_key: str,
    color: str,
    dash: str,
    name: str,
) -> Any:
    import plotly.graph_objects as go

    subset = sorted(
        (row for row in rows if row[fixed_key] == fixed_value),
        key=lambda row: row[varying_key],
    )
    return go.Scatter3d(
        x=[row["compute_len"] for row in subset],
        y=[row["cache_len"] for row in subset],
        z=[row["prefill_rt"] for row in subset],
        mode="lines",
        line={"color": color, "width": 5, "dash": dash},
        name=name,
        hoverinfo="skip",
        showlegend=True,
    )


def fmt_tokens(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}K"
    return f"{value:.0f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an interactive 3D prefill chart from a result JSON file."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to Prefill_Result.json or cache_grid_results.json",
    )
    parser.add_argument(
        "--output", type=Path, help="Output HTML path; defaults beside the input file"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size to display"
    )
    parser.add_argument("--title", default=None, help="Chart title")
    parser.add_argument(
        "--model-label", default=None, help="Model label used to build the title."
    )
    parser.add_argument(
        "--profile", default=None, help="JSON profile for parameter defaults."
    )
    parser.add_argument(
        "--log-rt",
        action="store_true",
        help="Use log scale for the Prefill RT (Z) axis",
    )
    parser.add_argument(
        "--log-x",
        action="store_true",
        help="Deprecated alias for --log-rt",
    )
    parser.add_argument(
        "--stretch-rt",
        type=float,
        default=None,
        help="Visual stretch factor for the Prefill RT axis (e.g. 5 makes RT changes 5x taller)",
    )
    parser.add_argument(
        "--stretch-x",
        type=float,
        default=None,
        help="Deprecated alias for --stretch-rt",
    )
    args = parser.parse_args()

    if args.log_x and not args.log_rt:
        warnings.warn(
            "--log-x is deprecated; use --log-rt", DeprecationWarning, stacklevel=2
        )
        args.log_rt = True
    if args.stretch_x is not None and args.stretch_rt is None:
        warnings.warn(
            "--stretch-x is deprecated; use --stretch-rt",
            DeprecationWarning,
            stacklevel=2,
        )
        args.stretch_rt = args.stretch_x
    if args.stretch_rt is None:
        args.stretch_rt = 1.0

    profile = None
    profile_sha256 = None
    if args.profile:
        profile = load_profile(args.profile)
        profile_sha256 = profile_fingerprint(profile)
    else:
        try:
            input_data = json.loads(args.input.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            input_data = None
        if isinstance(input_data, dict):
            embedded_sha = input_data.get("profile_sha256")
            embedded = extract_embedded_profile(input_data)
            if embedded is not None:
                profile = embedded
                profile_sha256 = (
                    embedded_sha
                    if isinstance(embedded_sha, str)
                    else profile_fingerprint(profile)
                )

    model_label = resolve_label(profile, args.model_label, "DeepSeek-V4-Pro")

    rows = load_rows(args.input, args.batch_size)
    if not rows:
        parser.error(f"no usable rows for batch_size={args.batch_size} in {args.input}")

    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Plotly is required. Install it with: python3 -m pip install --user plotly"
        ) from error

    output = args.output or args.input.with_name(f"{args.input.stem}.interactive.html")
    output.parent.mkdir(parents=True, exist_ok=True)

    default_title = (
        f"{model_label} Prefill — interactive 3D view (batch size {args.batch_size})"
    )
    title = resolve_title(profile, args.title, default_title)

    warm_palette = (
        "#b42318",
        "#d97706",
        "#15803d",
        "#0369a1",
        "#6d28d9",
        "#be185d",
        "#475569",
        "#0f766e",
    )
    cool_palette = (
        "#7c3aed",
        "#c026d3",
        "#0369a1",
        "#0f766e",
        "#b42318",
        "#d97706",
        "#15803d",
        "#475569",
    )

    cache_levels = top_levels(rows, "cache_len")
    compute_levels = top_levels(rows, "compute_len")

    traces: list[Any] = []

    for index, level in enumerate(cache_levels):
        traces.append(
            line_trace(
                rows,
                fixed_key="cache_len",
                fixed_value=level,
                varying_key="compute_len",
                color=warm_palette[index % len(warm_palette)],
                dash="solid",
                name=f"cache = {fmt_tokens(level)}",
            )
        )

    for index, level in enumerate(compute_levels):
        traces.append(
            line_trace(
                rows,
                fixed_key="compute_len",
                fixed_value=level,
                varying_key="cache_len",
                color=cool_palette[index % len(cool_palette)],
                dash="dash",
                name=f"compute = {fmt_tokens(level)}",
            )
        )

    traces.append(
        go.Scatter3d(
            x=[row["compute_len"] for row in rows],
            y=[row["cache_len"] for row in rows],
            z=[row["prefill_rt"] for row in rows],
            mode="markers",
            marker={
                "size": 4,
                "color": [row["prefill_rt"] for row in rows],
                "colorscale": "Viridis",
                "colorbar": {"title": "Prefill RT (ms)"},
                "opacity": 0.8,
            },
            customdata=[
                [row["input_len"], row["cache_len"], row["compute_len"]] for row in rows
            ],
            hovertemplate=(
                "Compute tokens: %{x:,.0f}<br>"
                "Cached tokens: %{y:,.0f}<br>"
                "Prefill RT: %{z:.2f} ms<br>"
                "Input length: %{customdata[0]:,.0f}<extra></extra>"
            ),
            name="measurements",
            showlegend=False,
        )
    )

    figure = go.Figure(data=traces)
    figure.update_layout(
        title=title,
        scene={
            "xaxis_title": "Compute tokens (X)",
            "yaxis_title": "Observed cached tokens (Y)",
            "zaxis_title": "Prefill RT (Z, ms)",
            "zaxis_type": "log" if args.log_rt else "linear",
            "aspectmode": "manual",
            "aspectratio": {"x": 1, "y": 1, "z": args.stretch_rt},
        },
        legend={
            "title": {"text": "Trend guides"},
            "yanchor": "top",
            "y": 0.99,
            "xanchor": "left",
            "x": 0.01,
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 50},
    )
    figure.write_html(output, include_plotlyjs=True, full_html=True)
    print(
        json.dumps(
            {
                "input": str(args.input),
                "rows": len(rows),
                "output": str(output),
                "title": title,
                "profile": profile,
                "profile_sha256": profile_sha256,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

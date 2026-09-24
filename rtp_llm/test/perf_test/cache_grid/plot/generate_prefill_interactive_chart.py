#!/usr/bin/env python3
"""Generate an interactive, rotatable 3D prefill chart from a result JSON.

The chart mirrors the static SVG's convention:
  X = uncached compute tokens = input_len - observed_cache_len
  Y = observed cached tokens
  Z = measured prefill RT / TTFT (ms)

Every valid geometry is shown as a dot.  Coloured lines are representative
fixed-cache (warm, solid) and fixed-compute (cool, dashed) trend guides.
Duplicate geometries are collapsed by median RT.  Pass ``--all-runs`` to plot
every successful run as its own dot instead of one median dot per case.
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from rtp_llm.test.perf_test.cache_grid.config.perf_profile import (
    extract_embedded_profile,
)
from rtp_llm.test.perf_test.cache_grid.config.perf_profile import (
    fingerprint as profile_fingerprint,
)
from rtp_llm.test.perf_test.cache_grid.config.perf_profile import (
    load_profile,
    resolve_label,
    resolve_title,
)
from rtp_llm.test.perf_test.cache_grid.runner.result_schema import (
    MetricFormatError,
    single_request_metric,
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
        "median_ttft_ms",
        "avg_ttft_ms",
        "ttft_ms",
        "client_wall_time_ms",
        "avg_prefill_time",
        "target_ms",
        "prefill_time_ms",
        "prefill_ms",
    ):
        value = number(item.get(key))
        if value is not None:
            return value

    runs = item.get("runs")
    if isinstance(runs, list):
        values = [
            run_prefill_rt(run)
            for run in runs
            if isinstance(run, dict) and run.get("success", True)
        ]
        values = [value for value in values if value is not None]
        if values:
            return median(values)
    return None


def run_prefill_rt(run: dict[str, Any]) -> float | None:
    for key in ("ttft_ms", "client_wall_time_ms", "prefill_time_ms", "total_time_ms"):
        value = number(run.get(key))
        if value is not None:
            return value
    return None


def load_rows(
    input_path: Path, batch_size: int, all_runs: bool = False
) -> list[dict[str, float]]:
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
        try:
            item = single_request_metric(item)
        except MetricFormatError:
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

        runs = item.get("runs") if isinstance(item.get("runs"), list) else []
        if all_runs and runs:
            input_len = number(item.get("input_len", item.get("seq_len")))
            case_cache_len = observed_cache_len(item)
            for run_index, run in enumerate(runs, 1):
                if not isinstance(run, dict) or not run.get("success", True):
                    continue
                rt = run_prefill_rt(run)
                cache_len = number(run.get("reuse_len"))
                if cache_len is None:
                    cache_len = case_cache_len
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
                        "run_index": run_index,
                    }
                )
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

    if all_runs:
        return rows

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


def representative_levels(rows: list[dict[str, float]], key: str) -> list[float]:
    """Choose guide planes near 0%, 33%, 67%, and 100% of an axis."""
    values = sorted({row[key] for row in rows})
    maximum = values[-1]
    chosen: list[float] = []
    for target in (0.0, 0.33 * maximum, 0.67 * maximum, maximum):
        level = min(values, key=lambda value: abs(value - target))
        if level not in chosen:
            chosen.append(level)
    return chosen


def representative_slice(
    rows: list[dict[str, float]],
    fixed_key: str,
    fixed_value: float,
    varying_key: str,
) -> list[dict[str, float]]:
    """Build the same nearby, median-binned guide used by the static SVG."""
    axis_maximum = max(
        max(row["cache_len"] for row in rows),
        max(row["compute_len"] for row in rows),
    )
    tolerance = max(4_096.0, axis_maximum * 0.025)
    subset = [row for row in rows if abs(row[fixed_key] - fixed_value) <= tolerance]
    if len(subset) < 8:
        subset = sorted(rows, key=lambda row: abs(row[fixed_key] - fixed_value))[
            : max(8, min(24, len(rows)))
        ]

    ordered = sorted(subset, key=lambda row: row[varying_key])
    bins = min(12, len(ordered))
    points: list[dict[str, float]] = []
    for index in range(bins):
        lo = (index * len(ordered)) // bins
        hi = ((index + 1) * len(ordered)) // bins
        group = ordered[lo : max(hi, lo + 1)]
        point = dict(group[len(group) // 2])
        point[fixed_key] = fixed_value
        point[varying_key] = median(row[varying_key] for row in group)
        point["prefill_rt"] = median(row["prefill_rt"] for row in group)
        points.append(point)
    return points


def line_trace(
    rows: list[dict[str, float]],
    fixed_key: str,
    fixed_value: float,
    varying_key: str,
    color: str,
    dash: str,
    name: str,
    z_key: str = "prefill_rt",
) -> Any:
    import plotly.graph_objects as go

    subset = representative_slice(rows, fixed_key, fixed_value, varying_key)
    return go.Scatter3d(
        x=[row["compute_len"] for row in subset],
        y=[row["cache_len"] for row in subset],
        z=[row[z_key] for row in subset],
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


Z_METRICS = {
    "rt": {
        "key": "prefill_rt",
        "label": "Prefill RT",
        "unit": "ms",
        "derive": lambda row: row["prefill_rt"],
    },
    "tpm-compute": {
        "key": "tpm_compute",
        "label": "Compute TPM",
        "unit": "tokens/min",
        "derive": lambda row: (
            row["compute_len"] * 60000.0 / row["prefill_rt"]
            if row["prefill_rt"] > 0
            else None
        ),
    },
    "tpm-effective": {
        "key": "tpm_effective",
        "label": "Effective TPM",
        "unit": "tokens/min",
        "derive": lambda row: (
            row["input_len"] * 60000.0 / row["prefill_rt"]
            if row["prefill_rt"] > 0
            else None
        ),
    },
}


def apply_z_metric(
    rows: list[dict[str, float]],
    z_metric: str,
    cards: float = 1.0,
    rt_cards: float | None = None,
) -> list[dict[str, float]]:
    spec = Z_METRICS[z_metric]
    # rt latency is a whole-system measurement; only throughput is per card.
    divisor = rt_cards if z_metric == "rt" and rt_cards is not None else cards
    enriched: list[dict[str, float]] = []
    for row in rows:
        value = spec["derive"](row)
        if value is None or not math.isfinite(value):
            continue
        if divisor and divisor != 1.0:
            value = value / divisor
        enriched.append({**row, spec["key"]: value})
    return enriched


def profile_cards(profile: dict | None) -> float | None:
    """Use total workers, or TP x DP x PP; EP/CP are not extra GPU factors."""
    engine = (profile or {}).get("engine") or {}

    def count(value):
        parsed = number(value)
        if (
            isinstance(value, bool)
            or parsed is None
            or parsed <= 0
            or not parsed.is_integer()
        ):
            raise ValueError("profile GPU counts must be positive integers")
        return parsed

    if "world_size" in engine:
        return count(engine["world_size"])
    if any(key in engine for key in ("tp_size", "dp_size", "pp_size")):
        return math.prod(
            count(engine.get(key, 1)) for key in ("tp_size", "dp_size", "pp_size")
        )
    return None


def detect_cards(input_path: Path, profile: dict | None = None) -> float | None:
    """Prefer profile topology, then legacy result run_config metadata."""
    cards = profile_cards(profile)
    if cards is not None:
        return cards
    try:
        data = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    cards = profile_cards(extract_embedded_profile(data))
    if cards is not None:
        return cards
    run_config = data.get("run_config")
    if not isinstance(run_config, dict):
        return None
    engine = run_config.get("engine", {})
    if "world_size" in engine:
        return profile_cards({"engine": engine})
    tp_size = number(engine.get("tp_size"))
    if tp_size is None or tp_size <= 0:
        # tp_size often only survives inside the engine's CLI args.
        for arg in (
            engine.get("args", []) if isinstance(engine.get("args"), list) else []
        ):
            if isinstance(arg, str) and arg.startswith("--tp_size="):
                tp_size = number(arg.split("=", 1)[1])
                break
    if tp_size is None or tp_size <= 0:
        return None
    return tp_size


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
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help=(
            "Plot every successful run as its own point (raw scatter) instead "
            "of one median point per case"
        ),
    )
    parser.add_argument(
        "--z-metric",
        default="rt",
        choices=sorted(Z_METRICS),
        help=(
            "Z axis metric: rt = prefill latency (ms); tpm-compute = uncached "
            "compute tokens per minute; tpm-effective = input tokens (cache "
            "hit + compute) per minute"
        ),
    )
    parser.add_argument(
        "--cards",
        type=float,
        default=None,
        help=(
            "Accelerator count used to normalize TPM to per-card throughput. "
            "Defaults to profile engine.world_size (or TP x DP x PP), then "
            "legacy run_config tp_size; use 1 "
            "to keep the whole-system TPM."
        ),
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
    parser.add_argument(
        "--marker-size",
        type=float,
        default=2.0,
        help="Measurement marker size in pixels (default: 2)",
    )
    args = parser.parse_args()

    if args.marker_size <= 0:
        parser.error("--marker-size must be greater than 0")

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

    model_label = resolve_label(profile, args.model_label, "Model")

    rows = load_rows(args.input, args.batch_size, all_runs=args.all_runs)
    if not rows:
        parser.error(f"no usable rows for batch_size={args.batch_size} in {args.input}")

    z_spec = Z_METRICS[args.z_metric]
    z_key = z_spec["key"]
    cards = 1.0
    if args.z_metric != "rt":
        try:
            cards = (
                args.cards
                if args.cards is not None
                else (detect_cards(args.input, profile) or 1.0)
            )
        except ValueError as error:
            parser.error(str(error))
        if not math.isfinite(cards) or cards <= 0 or not cards.is_integer():
            parser.error("--cards must be a positive integer")
    rows = apply_z_metric(rows, args.z_metric, cards=cards)
    if not rows:
        parser.error(f"no usable rows for z_metric={args.z_metric} in {args.input}")

    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Plotly is required. Install it with: python3 -m pip install --user plotly"
        ) from error

    suffix = ".interactive.html"
    if args.all_runs:
        suffix = ".all_runs.interactive.html"
    if args.z_metric != "rt":
        stem, ext = suffix.split(".", 1)
        per_card = f"_per_card{cards:g}" if cards != 1.0 else ""
        suffix = f"{stem}.{args.z_metric}{per_card}.{ext}"
    output = args.output or args.input.with_name(f"{args.input.stem}{suffix}")
    output.parent.mkdir(parents=True, exist_ok=True)

    per_card_note = (
        f", per card (÷{cards:g} cards)"
        if args.z_metric != "rt" and cards != 1.0
        else ""
    )
    default_title = (
        f"{model_label} Prefill — interactive 3D view (batch size {args.batch_size}"
        f"{', all runs' if args.all_runs else ''}, Z = {z_spec['label']}"
        f"{per_card_note})"
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

    cache_levels = representative_levels(rows, "cache_len")
    compute_levels = representative_levels(rows, "compute_len")

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
                z_key=z_key,
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
                z_key=z_key,
            )
        )

    show_run_index = args.all_runs and all("run_index" in row for row in rows)
    customdata = [
        [row["input_len"], row["cache_len"], row["compute_len"]]
        + ([row["run_index"]] if show_run_index else [])
        for row in rows
    ]
    z_format = ":,.0f" if args.z_metric != "rt" else ":.2f"
    hovertemplate = (
        "Compute tokens: %{x:,.0f}<br>"
        "Cached tokens: %{y:,.0f}<br>"
        f"{z_spec['label']}: %{{z{z_format}}}{'' if args.z_metric == 'rt' else ' tokens/min'}<br>"
        "Prefill RT: %{customdata[" + str(4 if show_run_index else 4) + "]:,.2f} ms<br>"
        "Input length: %{customdata[0]:,.0f}<br>"
        + (
            "Run: %{customdata[3]}<extra></extra>"
            if show_run_index
            else "<extra></extra>"
        )
    )
    customdata = [entry + [row["prefill_rt"]] for entry, row in zip(customdata, rows)]
    colorbar_title = (
        f"{z_spec['label']} ({z_spec['unit']}, darker = slower)"
        if args.z_metric == "rt"
        else (
            f"{z_spec['label']} per card ({z_spec['unit']}, darker = lower throughput)"
            if cards != 1.0
            else f"{z_spec['label']} ({z_spec['unit']}, darker = lower throughput)"
        )
    )

    traces.append(
        go.Scatter3d(
            x=[row["compute_len"] for row in rows],
            y=[row["cache_len"] for row in rows],
            z=[row[z_key] for row in rows],
            mode="markers",
            marker={
                "size": args.marker_size,
                "color": [row[z_key] for row in rows],
                "colorscale": "Viridis",
                # Higher latency (lower throughput) is darker; keep raw values.
                "reversescale": True,
                "colorbar": {"title": {"text": colorbar_title}},
                "opacity": 0.8,
            },
            customdata=customdata,
            hovertemplate=hovertemplate,
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
            "zaxis_title": (
                f"{z_spec['label']} per card (Z, {z_spec['unit']})"
                if args.z_metric != "rt" and cards != 1.0
                else f"{z_spec['label']} (Z, {z_spec['unit']})"
            ),
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
                "aggregation": "all_runs" if args.all_runs else "median",
                "z_metric": args.z_metric,
                "cards": cards if args.z_metric != "rt" else 1.0,
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

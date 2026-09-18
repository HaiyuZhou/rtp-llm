#!/usr/bin/env python3
"""Generate a 3D heatmap of production TTFT traffic."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_traffic_histogram(input_path: Path) -> list[dict[str, Any]]:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    return data.get("buckets", [])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a 3D heatmap of production TTFT traffic."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to prefill_traffic_histogram.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output HTML path; defaults beside the input file",
    )
    parser.add_argument(
        "--z-metric",
        default="p95",
        choices=["p50", "p95", "p99", "mean"],
        help="TTFT metric for Z axis",
    )
    parser.add_argument(
        "--log-z",
        action="store_true",
        help="Use log scale for Z axis",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=3.0,
        help="Marker size in pixels",
    )
    parser.add_argument(
        "--log-color",
        action="store_true",
        help="Use log scale for color (request count)",
    )
    parser.add_argument(
        "--color-min-percentile",
        type=float,
        default=0.0,
        help="Clip color scale minimum to this percentile (0-100)",
    )
    parser.add_argument(
        "--color-max-percentile",
        type=float,
        default=100.0,
        help="Clip color scale maximum to this percentile (0-100)",
    )
    args = parser.parse_args()

    rows = load_traffic_histogram(args.input)
    if not rows:
        parser.error(f"no traffic buckets in {args.input}")

    z_field = {
        "p50": "online_ttft_p50_ms",
        "p95": "online_ttft_p95_ms",
        "p99": "online_ttft_p99_ms",
        "mean": "online_ttft_mean_ms",
    }[args.z_metric]

    points = []
    for row in rows:
        z_value = row.get(z_field)
        if z_value is None or not math.isfinite(z_value):
            continue
        points.append(
            {
                "x": row["mean_compute_len"],
                "y": row["mean_reuse_len"],
                "z": z_value,
                "count": row["request_count"],
                "input_len": row["mean_input_len"],
                "reuse_len": row["mean_reuse_len"],
                "compute_len": row["mean_compute_len"],
                "p50": row.get("online_ttft_p50_ms"),
                "p95": row.get("online_ttft_p95_ms"),
                "p99": row.get("online_ttft_p99_ms"),
            }
        )

    if not points:
        parser.error(f"no valid TTFT values for z_metric={args.z_metric}")

    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Plotly is required. Install it with: python3 -m pip install --user plotly"
        ) from error

    output = args.output or args.input.with_name(
        f"{args.input.stem}.ttft_3d.{args.z_metric}.html"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    counts = [p["count"] for p in points]
    color_values = counts

    if args.log_color:
        color_values = [math.log1p(c) for c in counts]

    if args.color_min_percentile > 0 or args.color_max_percentile < 100:
        sorted_counts = sorted(color_values)
        n = len(sorted_counts)
        color_min = sorted_counts[int(n * args.color_min_percentile / 100)]
        color_max = sorted_counts[
            min(n - 1, int(n * args.color_max_percentile / 100) - 1)
        ]
        color_values = [max(color_min, min(color_max, v)) for v in color_values]
    figure = go.Figure(
        data=[
            go.Scatter3d(
                x=[p["x"] for p in points],
                y=[p["y"] for p in points],
                z=[p["z"] for p in points],
                mode="markers",
                marker={
                    "size": args.marker_size,
                    "color": color_values,
                    "colorscale": "Hot",
                    "reversescale": True,
                    "colorbar": {
                        "title": "请求数（热度）" + (" (log)" if args.log_color else "")
                    },
                    "opacity": 0.8,
                },
                customdata=[
                    [
                        p["input_len"],
                        p["reuse_len"],
                        p["p50"],
                        p["p95"],
                        p["p99"],
                        p["count"],
                    ]
                    for p in points
                ],
                hovertemplate=(
                    "Compute tokens: %{x:,.0f}<br>"
                    "Reuse tokens: %{y:,.0f}<br>"
                    f"TTFT {args.z_metric}: %{{z:.1f}} ms<br>"
                    "Input: %{customdata[0]:,.0f}<br>"
                    "Reuse: %{customdata[1]:,.0f}<br>"
                    "TTFT p50/p95/p99: %{customdata[2]:.1f} / %{customdata[3]:.1f} / %{customdata[4]:.1f} ms<br>"
                    "请求数: %{customdata[5]:,d}"
                    "<extra></extra>"
                ),
                name="线上 TTFT 热力点",
            )
        ]
    )
    figure.update_layout(
        title=f"线上 TTFT 3D 热力图（Z = {args.z_metric.upper()}，颜色 = 请求热度）",
        scene={
            "xaxis_title": "非缓存 compute tokens (X)",
            "yaxis_title": "缓存 reuse tokens (Y)",
            "zaxis_title": f"线上 TTFT {args.z_metric.upper()} (ms, Z)",
            "zaxis_type": "log" if args.log_z else "linear",
            "aspectmode": "manual",
            "aspectratio": {"x": 1, "y": 1, "z": 0.7},
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 50},
    )
    figure.write_html(output, include_plotlyjs=True, full_html=True)
    print(
        json.dumps(
            {
                "input": str(args.input),
                "points": len(points),
                "z_metric": args.z_metric,
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate a 3D heatmap of production TTFT traffic."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from rtp_llm.test.perf_test.cache_grid.plot.traffic_chart import render_traffic_chart


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

    output = args.output or args.input.with_name(
        f"{args.input.stem}.ttft_3d.{args.z_metric}.html"
    )

    render_traffic_chart(
        points,
        output,
        title=(f"线上 TTFT 3D 热力图（Z = {args.z_metric.upper()}，颜色 = 请求热度）"),
        z_title=(f"线上 TTFT {args.z_metric.upper()} (ms, Z)"),
        trace_name=("线上 TTFT 热力点"),
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
        custom_fields=(("input_len", "reuse_len", "p50", "p95", "p99", "count")),
        marker_size=args.marker_size,
        log_z=args.log_z,
        log_color=args.log_color,
        color_max_percentile=args.color_max_percentile,
        color_min_percentile=args.color_min_percentile,
    )
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

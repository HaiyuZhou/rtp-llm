#!/usr/bin/env python3
"""Generate a 3D heatmap of production single-card TPM (tokens per minute).

Token count = input_len = compute + reuse.  Single-card TPM is derived from
per-bucket mean input length and mean TTFT, divided by the number of cards.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from rtp_llm.test.perf_test.cache_grid.config.topology import (
    detect_cards as infer_cards,
)
from rtp_llm.test.perf_test.cache_grid.plot.traffic_chart import render_traffic_chart


def detect_cards(benchmark_path: Path, default_cards: int | None) -> int:
    if default_cards is not None:
        return default_cards
    return infer_cards(benchmark_path) or 1


MS_PER_MINUTE = 60_000.0


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compute_tpm(mean_input_len: float, ttft_ms: float, cards: int) -> float | None:
    if not math.isfinite(mean_input_len) or mean_input_len <= 0:
        return None
    if not math.isfinite(ttft_ms) or ttft_ms <= 0:
        return None
    if cards <= 0:
        return None
    return (mean_input_len * MS_PER_MINUTE / ttft_ms) / cards


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a 3D heatmap of production single-card TPM."
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
        "--ttft-metric",
        default="mean",
        choices=["mean", "p50", "p95", "p99"],
        help="TTFT metric used to derive TPM throughput (default: mean)",
    )
    parser.add_argument(
        "--cards",
        type=int,
        default=None,
        help="Number of cards for single-card normalization; auto-detect from benchmark if omitted",
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
        "--color-max-percentile",
        type=float,
        default=75.0,
        help="Clip color scale maximum to this percentile (0-100)",
    )
    args = parser.parse_args()

    data = load_json(args.input)
    buckets = data.get("buckets", [])
    if not buckets:
        parser.error(f"no traffic buckets in {args.input}")

    settings = data.get("settings", {}) if isinstance(data, dict) else {}
    benchmark_path = (
        Path(settings.get("benchmark", "")) if isinstance(settings, dict) else Path()
    )
    if not benchmark_path.is_absolute() or not benchmark_path.exists():
        benchmark_path = args.input.parent / benchmark_path.name
    cards = detect_cards(benchmark_path, args.cards)

    ttft_field = {
        "mean": "online_ttft_mean_ms",
        "p50": "online_ttft_p50_ms",
        "p95": "online_ttft_p95_ms",
        "p99": "online_ttft_p99_ms",
    }[args.ttft_metric]

    points: list[dict[str, Any]] = []
    total_requests = 0
    total_input_tokens = 0
    weighted_request_tpm = 0.0
    weighted_token_tpm = 0.0
    max_tpm: float | None = None
    max_tpm_bucket: dict[str, Any] | None = None

    for row in buckets:
        ttft_ms = row.get(ttft_field)
        mean_input_len = row.get("mean_input_len")
        mean_reuse_len = row.get("mean_reuse_len")
        mean_compute_len = row.get("mean_compute_len")
        request_count = row.get("request_count", 0)
        input_token_sum = row.get("input_token_sum", 0)

        tpm = compute_tpm(mean_input_len, ttft_ms, cards)
        if tpm is None:
            continue

        points.append(
            {
                "x": mean_compute_len,
                "y": mean_reuse_len,
                "z": tpm,
                "count": request_count,
                "input_len": mean_input_len,
                "reuse_len": mean_reuse_len,
                "compute_len": mean_compute_len,
                "ttft_ms": ttft_ms,
                "input_token_sum": input_token_sum,
            }
        )

        total_requests += request_count
        total_input_tokens += input_token_sum
        weighted_request_tpm += request_count * tpm
        weighted_token_tpm += input_token_sum * tpm
        if max_tpm is None or tpm > max_tpm:
            max_tpm = tpm
            max_tpm_bucket = row

    if not points:
        parser.error(f"no valid TPM values for ttft_metric={args.ttft_metric}")

    request_weighted_tpm = (
        weighted_request_tpm / total_requests if total_requests > 0 else None
    )
    token_weighted_tpm = (
        weighted_token_tpm / total_input_tokens if total_input_tokens > 0 else None
    )

    output = args.output or args.input.with_name(
        f"{args.input.stem}.tpm_3d.{args.ttft_metric}.html"
    )

    render_traffic_chart(
        points,
        output,
        title=(
            f"线上单卡 TPM 3D 热力图（Z = TPM，TTFT 口径 = {args.ttft_metric.upper()}，"
            f"颜色 = 请求热度，{cards} 卡）"
        ),
        z_title=(f"单卡 TPM (tokens/min, Z, TTFT={args.ttft_metric.upper()})"),
        trace_name=("线上单卡 TPM"),
        hovertemplate=(
            "Compute tokens: %{x:,.0f}<br>"
            "Reuse tokens: %{y:,.0f}<br>"
            "单卡 TPM: %{z:,.0f}<br>"
            "Input: %{customdata[0]:,.0f}<br>"
            "Reuse: %{customdata[1]:,.0f}<br>"
            "Compute: %{customdata[2]:,.0f}<br>"
            "TTFT: %{customdata[3]:.1f} ms<br>"
            "请求数: %{customdata[4]:,d}<br>"
            "Input token sum: %{customdata[5]:,d}"
            "<extra></extra>"
        ),
        custom_fields=(
            (
                "input_len",
                "reuse_len",
                "compute_len",
                "ttft_ms",
                "count",
                "input_token_sum",
            )
        ),
        marker_size=args.marker_size,
        log_z=args.log_z,
        log_color=args.log_color,
        color_max_percentile=args.color_max_percentile,
    )

    summary = {
        "input": str(args.input),
        "points": len(points),
        "cards": cards,
        "ttft_metric": args.ttft_metric,
        "total_requests": total_requests,
        "total_input_tokens": total_input_tokens,
        "request_weighted_single_card_tpm": request_weighted_tpm,
        "token_weighted_single_card_tpm": token_weighted_tpm,
        "max_single_card_tpm": max_tpm,
        "max_tpm_bucket": {
            "input_bucket": (
                f"[{max_tpm_bucket['input_bucket_start']}, {max_tpm_bucket['input_bucket_end']})"
                if max_tpm_bucket
                else None
            ),
            "reuse_bucket": (
                f"[{max_tpm_bucket['reuse_bucket_start']}, {max_tpm_bucket['reuse_bucket_end']})"
                if max_tpm_bucket
                else None
            ),
            "mean_input_len": (
                max_tpm_bucket.get("mean_input_len") if max_tpm_bucket else None
            ),
            "ttft_ms": max_tpm_bucket.get(ttft_field) if max_tpm_bucket else None,
            "request_count": (
                max_tpm_bucket.get("request_count") if max_tpm_bucket else None
            ),
        },
        "output": str(output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

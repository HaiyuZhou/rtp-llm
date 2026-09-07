#!/usr/bin/env python3
"""Generate a reproducible, block-aligned prefix-cache performance grid.

Input lengths are aligned to ``--alignment`` (128 by default).  Cache lengths
should instead be aligned to the engine's physical reuse granularity
(``--cache-alignment``): the prefix cache only reuses whole blocks, so two
requested cache lengths that floor onto the same block bucket measure the
identical geometry and waste one seed plus ``--measure-runs`` full prefills.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

from rtp_llm.test.perf_test.perf_profile import cache_grid_section, engine_section
from rtp_llm.test.perf_test.perf_profile import fingerprint as profile_fingerprint
from rtp_llm.test.perf_test.perf_profile import load_profile

DEFAULT_SEED = 104729  # A fixed prime, recorded in every generated plan.
DEFAULT_BOUNDARIES = (
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    786432,
    1048448,
    1048575,
)


def align_down(value: int, alignment: int) -> int:
    return value // alignment * alignment


def _stable_rng(seed: int, namespace: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{namespace}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _stratified_pick(values: list[int], count: int, seed: int) -> list[int]:
    if count >= len(values):
        return values
    rng = _stable_rng(seed, "input")
    picked = []
    for index in range(count):
        lo = index * len(values) // count
        hi = (index + 1) * len(values) // count
        picked.append(values[rng.randrange(lo, max(lo + 1, hi))])
    return sorted(set(picked))


def generate_input_lengths(
    minimum: int,
    maximum: int,
    alignment: int,
    count: int,
    mode: str,
    seed: int,
    boundaries: Iterable[int] = DEFAULT_BOUNDARIES,
) -> list[int]:
    if minimum <= 0 or maximum < minimum or alignment <= 0:
        raise ValueError("invalid input length bounds or alignment")
    first = ((minimum + alignment - 1) // alignment) * alignment
    aligned = list(range(first, maximum + 1, alignment))
    if not aligned:
        raise ValueError("input range contains no aligned value")
    forced = {x for x in boundaries if minimum <= x <= maximum}
    if mode == "stride":
        selected_set = set(aligned)
    else:
        if count < len(forced):
            raise ValueError(
                f"input point count {count} is smaller than {len(forced)} boundaries"
            )
        candidates = [x for x in aligned if x not in forced]
        selected_set = set(_stratified_pick(candidates, count - len(forced), seed))
    selected_set.update(forced)
    return sorted(selected_set)


def generate_cache_lengths(
    input_len: int,
    alignment: int,
    points: int,
    ratio_points: int,
    seed: int,
) -> list[int]:
    if points < 2 or ratio_points < 0 or ratio_points > points - 2:
        raise ValueError("cache points require two boundaries and valid ratio_points")
    max_cache = align_down(input_len - 1, alignment)
    if max_cache <= 0:
        return [0]
    result = {0, max_cache}
    interior_count = points - 2
    compute_points = interior_count - ratio_points
    rng = _stable_rng(seed, f"cache:{input_len}")

    def add_strata(count: int, reverse: bool) -> None:
        for index in range(count):
            # Pick an aligned index from the interior of each equal-width
            # stratum. Using indices avoids tokenizer/block-size assumptions.
            lo = 1 + index * max(1, max_cache // alignment - 1) // count
            hi = 1 + (index + 1) * max(1, max_cache // alignment - 1) // count
            cache_index = rng.randrange(lo, max(lo + 1, hi))
            value = min(cache_index * alignment, max_cache)
            result.add(max_cache - value if reverse else value)

    if ratio_points:
        add_strata(ratio_points, False)
    if compute_points:
        add_strata(compute_points, True)
    return sorted(x for x in result if 0 <= x < input_len)


def build_grid(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size <= 0 or args.measure_runs <= 0 or args.max_cases <= 0:
        raise ValueError("batch size, measure runs, and max cases must be positive")
    inputs = generate_input_lengths(
        args.min_input_len,
        args.max_input_len,
        args.alignment,
        args.input_points,
        args.input_mode,
        args.seed,
    )
    # Cache reuse granularity is the engine's physical block size
    # (--seq_size_per_block, multiplied by CP size when
    # PREFILL_CP_KV_CACHE_SHARDED=1), not the input alignment.  Points finer
    # than one block floor onto the same physical bucket, so callers should
    # pin --cache-alignment to the block size; 0 keeps --alignment for
    # compatibility with grids generated before this distinction existed.
    cache_alignment = getattr(args, "cache_alignment", 0) or args.alignment
    cases = []
    for input_len in inputs:
        cache_lengths = generate_cache_lengths(
            input_len,
            cache_alignment,
            args.cache_points_per_input,
            args.cache_ratio_points,
            args.seed,
        )
        for cache_len in cache_lengths:
            cases.append(
                {
                    "case_id": len(cases),
                    "batch_size": args.batch_size,
                    "input_len": input_len,
                    "cache_len": cache_len,
                }
            )
    if len(cases) > args.max_cases and not args.allow_large_grid:
        raise ValueError(
            f"generated {len(cases)} cases, exceeding --max-cases={args.max_cases}; "
            "reduce sampling or pass --allow-large-grid"
        )
    generator = {
        "name": "aligned_stratified_cache_grid",
        "version": 1,
        "alignment": args.alignment,
        "seed": args.seed,
        "input_sampling": {
            "mode": args.input_mode,
            "min": args.min_input_len,
            "max": args.max_input_len,
            "requested_count": args.input_points,
        },
        "cache_sampling": {
            "alignment": cache_alignment,
            "points_per_input": args.cache_points_per_input,
            "ratio_points": args.cache_ratio_points,
            "compute_points": args.cache_points_per_input - 2 - args.cache_ratio_points,
            "include_cold": True,
            "include_near_full": True,
        },
        "unaligned_boundary_exceptions": [
            value for value in inputs if value % args.alignment
        ],
    }
    return {
        "schema_version": 2,
        "kind": "aligned_stratified_cache_grid",
        "generator": generator,
        "summary": {
            "input_count": len(inputs),
            "case_count": len(cases),
            "estimated_measure_requests": len(cases) * args.measure_runs,
            "estimated_seed_requests": sum(c["cache_len"] > 0 for c in cases),
        },
        "cases": cases,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-input-len", type=int, default=256)
    parser.add_argument("--max-input-len", type=int, default=1048575)
    parser.add_argument(
        "--alignment",
        type=int,
        default=128,
        help="Alignment in tokens for sampled input lengths",
    )
    parser.add_argument(
        "--cache-alignment",
        type=int,
        default=None,
        help=(
            "Alignment in tokens for cache lengths; defaults to --alignment. "
            "Set to the engine's physical reuse granularity "
            "(--seq_size_per_block, multiplied by CP size when "
            "PREFILL_CP_KV_CACHE_SHARDED=1) so every cache point is exactly "
            "reusable and no two points collapse onto the same block bucket."
        ),
    )
    parser.add_argument(
        "--input-mode", choices=("stratified", "stride"), default="stratified"
    )
    parser.add_argument("--input-points", type=int, default=1024)
    parser.add_argument("--cache-points-per-input", type=int, default=16)
    parser.add_argument("--cache-ratio-points", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--measure-runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-cases", type=int, default=20000)
    parser.add_argument("--allow-large-grid", action="store_true")
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        help=(
            "JSON profile for parameter defaults and downstream embedding. "
            "cache_grid.cache_alignment overrides --cache-alignment; "
            "engine.max_seq_len is informational only."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    profile = None
    profile_sha256 = None
    if args.profile is not None:
        profile = load_profile(args.profile)
        profile_sha256 = profile_fingerprint(profile)
        cache_grid = cache_grid_section(profile)
        profile_cache_alignment = cache_grid.get("cache_alignment")
        if args.cache_alignment is None and profile_cache_alignment is not None:
            args.cache_alignment = int(profile_cache_alignment)
    if args.cache_alignment is None:
        args.cache_alignment = 0
    payload = build_grid(args)
    if profile is not None:
        payload["profile"] = profile
        payload["profile_sha256"] = profile_sha256
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()

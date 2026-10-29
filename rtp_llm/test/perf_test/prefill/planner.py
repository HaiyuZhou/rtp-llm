"""Deterministic cache-aware prefill grid planning."""

from __future__ import annotations

import hashlib
import math
import random
from typing import Dict, List, Tuple

from rtp_llm.test.perf_test.config_schema import EngineConfig, PrefillWorkloadConfig
from rtp_llm.test.perf_test.errors import PlanningError


def _stable_rng(seed: int, namespace: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{namespace}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _validate_contract(workload: PrefillWorkloadConfig, engine: EngineConfig) -> int:
    if workload.kind != "cache_grid":
        raise PlanningError("prefill planner requires workload.kind=cache_grid")
    if workload.max_input_len + 1 > engine.max_seq_len:
        raise PlanningError(
            "workload.max_input_len plus one output token exceeds engine.max_seq_len"
        )
    if workload.commit_tail_tokens % workload.cache_alignment:
        raise PlanningError(
            "workload.commit_tail_tokens must be aligned to cache_alignment"
        )
    if workload.min_compute_len > workload.max_input_len:
        raise PlanningError(
            "workload.min_compute_len cannot exceed workload.max_input_len"
        )
    if len(set(workload.compute_steps)) != len(workload.compute_steps) or any(
        left <= right
        for left, right in zip(workload.compute_steps, workload.compute_steps[1:])
    ):
        raise PlanningError(
            "workload.compute_steps must be unique and ordered coarse-to-fine"
        )
    if workload.max_cache_len is not None and (
        workload.max_cache_len < workload.min_cache_len
    ):
        raise PlanningError("workload cache minimum cannot exceed cache maximum")
    if any(batch > engine.concurrency_limit for batch in workload.batch_sizes):
        raise PlanningError(
            "workload.batch_sizes cannot exceed engine.concurrency_limit"
        )
    if any(
        batch * workload.max_input_len > engine.max_batch_tokens
        for batch in workload.batch_sizes
    ):
        raise PlanningError(
            "batch_size times max_input_len exceeds engine.max_batch_tokens"
        )
    return math.lcm(workload.cache_alignment, workload.commit_tail_tokens)


def _cache_lengths(workload: PrefillWorkloadConfig, cache_step: int) -> Tuple[int, ...]:
    compute_floor = max(workload.min_compute_len, workload.commit_tail_tokens)
    largest_usable = workload.max_input_len - compute_floor
    if workload.min_cache_len == 0:
        largest_usable = max(0, largest_usable)
    maximum = (
        largest_usable if workload.max_cache_len is None else workload.max_cache_len
    )
    if maximum > largest_usable:
        raise PlanningError(
            "workload.max_cache_len must leave the required compute and commit tail"
        )

    if workload.cache_lengths:
        values = workload.cache_lengths
        if workload.random_cache_count:
            raise PlanningError(
                "cache_lengths and random_cache_count are mutually exclusive"
            )
        if len(values) != len(set(values)):
            raise PlanningError("workload.cache_lengths must be unique")
        for value in values:
            if value < workload.min_cache_len or value > maximum:
                raise PlanningError(
                    f"cache length {value} is outside the usable cache range"
                )
            if value % cache_step:
                raise PlanningError(
                    f"cache length {value} must be aligned to {cache_step}"
                )
        return tuple(values)

    if workload.random_cache_count <= 0:
        raise PlanningError(
            "prefill workload requires cache_lengths or positive random_cache_count"
        )
    first = (workload.min_cache_len + cache_step - 1) // cache_step * cache_step
    candidates = list(range(first, maximum + 1, cache_step))
    if workload.random_cache_count > len(candidates):
        raise PlanningError(
            f"requested {workload.random_cache_count} random cache lengths, "
            f"but only {len(candidates)} aligned values exist"
        )
    rng = _stable_rng(workload.seed, "prefill-cache-lengths")
    return tuple(sorted(rng.sample(candidates, workload.random_cache_count)))


def plan_prefill_cases(
    workload: PrefillWorkloadConfig, engine: EngineConfig
) -> Tuple[Dict[str, object], ...]:
    """Expand fixed/random cache slices and compute refinements into identities."""
    cache_step = _validate_contract(workload, engine)
    cache_lengths = _cache_lengths(workload, cache_step)
    cases: List[Dict[str, object]] = []
    seen = set()
    compute_floor = max(workload.min_compute_len, workload.commit_tail_tokens)
    for batch_size in workload.batch_sizes:
        for cache_len in cache_lengths:
            maximum_compute = workload.max_input_len - cache_len
            first_compute = compute_floor if cache_len else workload.min_compute_len
            for refinement_level, compute_step in enumerate(workload.compute_steps):
                for compute_len in range(
                    first_compute, maximum_compute + 1, compute_step
                ):
                    input_len = cache_len + compute_len
                    geometry = (batch_size, input_len, cache_len)
                    if geometry in seen:
                        continue
                    seen.add(geometry)
                    cases.append(
                        {
                            "batch_size": batch_size,
                            "input_len": input_len,
                            "cache_len": cache_len,
                            "compute_len": compute_len,
                            "measure_runs": workload.measure_runs,
                            "seed": workload.seed,
                            "prefix_policy": (
                                "shared_seed" if batch_size == 1 else "independent"
                            ),
                            "refinement_level": refinement_level,
                            "compute_step": compute_step,
                        }
                    )
    if not cases:
        raise PlanningError("prefill workload generated no valid cases")
    if workload.max_cases is not None and len(cases) > workload.max_cases:
        raise PlanningError(
            f"generated {len(cases)} cases, exceeding max_cases={workload.max_cases}"
        )
    return tuple(cases)

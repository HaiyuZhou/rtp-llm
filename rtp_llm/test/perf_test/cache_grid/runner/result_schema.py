"""CPU-only adapters for measured cache-grid result formats."""

import math


class MetricFormatError(ValueError):
    """A metric cannot be represented as a single-request observation."""


def request_runs(item):
    """Iterate actual requests for timing-source auditing in either schema."""
    for run in item.get("runs", []):
        if not isinstance(run, dict):
            continue
        if "requests" in run:
            if isinstance(run["requests"], list):
                yield from (r for r in run["requests"] if isinstance(r, dict))
        else:
            yield run


def single_request_metric(item):
    """Unwrap scheduler batch=1 without changing the request latency contract.

    The batch barrier wall time and server forward time are diagnostics, not
    substitutes for the nested request's client TTFT. Multi-request batches
    cannot be reduced to one input/cache pair and are deliberately rejected.
    The input object is never modified.
    """
    runs = item.get("runs")
    grouped = (
        "request_groups" in item
        or item.get("execution_mode") == "scheduler_fixed_batch"
    )
    if isinstance(runs, list):
        grouped = grouped or any(isinstance(r, dict) and "requests" in r for r in runs)
    if not grouped:
        return item
    if item.get("batch_size") != 1:
        raise MetricFormatError("unsupported_grouped_batch")
    groups = item.get("request_groups")
    if (
        not isinstance(groups, list)
        or len(groups) != 1
        or not isinstance(groups[0], dict)
    ):
        raise MetricFormatError("invalid_request_groups")
    group = groups[0]
    inp, cache = group.get("input_len"), group.get("cache_len")
    if (
        group.get("count") != 1
        or type(inp) is not int
        or type(cache) is not int
        or not 0 <= cache < inp
    ):
        raise MetricFormatError("invalid_request_groups")
    for key, expected in (
        ("input_len", inp),
        ("cache_len", cache),
        ("cache_len_requested", cache),
    ):
        if key in item and item[key] != expected:
            raise MetricFormatError("request_shape_mismatch")
    if not isinstance(runs, list) or not runs:
        raise MetricFormatError("missing_runs")
    flat = []
    skip_reuse = item.get("reuse_validation_skipped", False)
    for run in runs:
        requests = run.get("requests") if isinstance(run, dict) else None
        if (
            not isinstance(requests, list)
            or len(requests) != 1
            or not isinstance(requests[0], dict)
        ):
            raise MetricFormatError("invalid_grouped_run")
        request = requests[0]
        if run.get("valid") is not True or request.get("success") is not True:
            raise MetricFormatError("run_failed")
        if (
            run.get("completed_requests", 1) != 1
            or request.get("input_len") != inp
            or request.get("output_len") != 1
            or request.get("shape_exact") is False
        ):
            raise MetricFormatError("request_shape_mismatch")
        observed = request.get("reuse_len")
        if type(observed) is not int or not 0 <= observed < inp:
            raise MetricFormatError("invalid_observed_geometry")
        if not skip_reuse and (
            observed != cache or request.get("reuse_exact") is False
        ):
            raise MetricFormatError("requested_reuse_mismatch")
        latency = request.get("ttft_ms", request.get("client_wall_time_ms"))
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(latency)
            or latency <= 0
            or request.get("timing_valid") is False
        ):
            raise MetricFormatError("invalid_latency")
        flat.append({**request, "ttft_ms": latency, "run_index": run.get("run_index")})
    if item.get("measure_runs", len(flat)) != len(flat) or item.get(
        "success_runs", len(flat)
    ) != len(flat):
        raise MetricFormatError("incomplete_runs")
    return {
        **item,
        "input_len": inp,
        "cache_len_requested": cache,
        "cache_len_observed": [r["reuse_len"] for r in flat],
        "measure_runs": len(flat),
        "success_runs": len(flat),
        "runs": flat,
    }

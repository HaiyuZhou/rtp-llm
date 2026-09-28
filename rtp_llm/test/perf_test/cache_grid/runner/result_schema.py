"""CPU-only adapters for measured cache-grid result formats."""

import math
from collections import defaultdict
from statistics import median
from typing import Any


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


def measurement_status(
    success, shape_exact, reuse_exact, timing_valid, *, skip_reuse=False
):
    """Classify failures consistently; skipping reuse never hides shape/timing errors."""
    for valid, status in (
        (success, "failed"),
        (shape_exact, "invalid_shape"),
        (timing_valid, "invalid_timing"),
        (reuse_exact or skip_reuse, "invalid_reuse"),
    ):
        if not valid:
            return status
    return "ok"


SUCCESS_STATUSES = frozenset(("ok", "success", "passed"))
DIAGNOSTIC_STATUSES = SUCCESS_STATUSES | {"unknown", "invalid_reuse"}
STATIC_RUN_TIME_FIELDS = (
    "ttft_ms",
    "client_wall_time_ms",
    "prefill_time_ms",
    "prefill_ms",
    "avg_prefill_time",
    "first_token_time_ms",
)
STATIC_TIME_FIELDS = (
    "median_ttft_ms",
    "avg_ttft_ms",
    "avg_prefill_time",
    "target_ms",
    "ttft_ms",
)
DIAGNOSTIC_RUN_TIME_FIELDS = (
    "ttft_ms",
    "client_wall_time_ms",
    "prefill_time_ms",
    "total_time_ms",
)
DIAGNOSTIC_TIME_FIELDS = (
    "median_ttft_ms",
    "avg_ttft_ms",
    "ttft_ms",
    "client_wall_time_ms",
    "avg_prefill_time",
    "target_ms",
    "prefill_time_ms",
    "prefill_ms",
)


def finite_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def integer(value: Any) -> int | None:
    number = finite_number(value)
    return int(number) if number is not None and number.is_integer() else None


def status_ok(item, allowed=SUCCESS_STATUSES):
    status = str(item.get("status", "")).lower()
    return not status or status in allowed


def first_number(item, fields):
    for field in fields:
        value = finite_number(item.get(field))
        if value is not None:
            return value
    return None


def observed_reuse_values(item, parse=finite_number):
    """Read per-run reuse, falling back to runs when the observed list is empty."""
    observed = item.get("cache_len_observed")
    values = [parse(value) for value in observed] if isinstance(observed, list) else []
    values = [value for value in values if value is not None]
    if not values and isinstance(item.get("runs"), list):
        values = [
            parse(run.get("reuse_len")) for run in item["runs"] if isinstance(run, dict)
        ]
        values = [value for value in values if value is not None]
    return values


def aggregate_latency(item, fields, run_fields, *, successful_only=True):
    value = first_number(item, fields)
    if value is not None:
        return value
    runs = item.get("runs")
    if isinstance(runs, list):
        values = [
            first_number(run, run_fields)
            for run in runs
            if isinstance(run, dict)
            and (not successful_only or run.get("success", True))
        ]
        values = [value for value in values if value is not None]
        if values:
            return median(values)
    return None


def consistent_value(values: list[float]) -> float | None:
    return values[0] if values and len(set(values)) == 1 else None


def observed_cache_len(item: dict[str, Any]) -> float | None:
    observed = item.get("cache_len_observed")
    if isinstance(observed, list):
        values = [finite_number(value) for value in observed if value is not None]
        return consistent_value(values)
    if observed is not None:
        return finite_number(observed)

    runs = item.get("runs")
    if isinstance(runs, list):
        values = [
            finite_number(run.get("reuse_len")) for run in runs if isinstance(run, dict)
        ]
        values = [value for value in values if value is not None]
        return consistent_value(values)
    return None


def prefill_rt(item):
    return aggregate_latency(item, DIAGNOSTIC_TIME_FIELDS, DIAGNOSTIC_RUN_TIME_FIELDS)


def run_prefill_rt(run):
    return first_number(run, DIAGNOSTIC_RUN_TIME_FIELDS)


def measurement_rows(data, batch_size, *, strict, all_runs=False):
    """Normalize chart geometry with explicit strict or diagnostic filtering.

    Strict plots require complete runs and requested reuse; diagnostic plots
    keep successful observations even when the requested cache was not hit.
    Fitting adds its stronger request-level checks separately.
    """
    metrics = (
        data if isinstance(data, list) else data.get("metrics", data.get("results", []))
    )
    rows = []
    for item in metrics:
        if not isinstance(item, dict):
            continue
        selected_batch = (
            item.get("batch_size", 1)
            if strict
            else (finite_number(item.get("batch_size", 1)) or 1)
        )
        try:
            if int(selected_batch) != batch_size:
                continue
        except (TypeError, ValueError, OverflowError):
            continue
        try:
            item = single_request_metric(item)
        except MetricFormatError:
            continue
        if not status_ok(item, SUCCESS_STATUSES if strict else DIAGNOSTIC_STATUSES):
            continue
        input_len = finite_number(item.get("input_len", item.get("seq_len")))
        if strict:
            if item.get("reuse_exact") is False and not item.get(
                "reuse_validation_skipped", False
            ):
                continue
            if item.get("success_runs") is not None:
                try:
                    if int(item["success_runs"]) != int(item.get("measure_runs", 3)):
                        continue
                except (TypeError, ValueError):
                    continue
            cache_len = consistent_value(observed_reuse_values(item))
            requested = finite_number(
                item.get(
                    "target_cache_len",
                    item.get("cache_len_requested", item.get("cache_len")),
                )
            )
            if cache_len != (requested if requested is not None else 0):
                continue
            rt = aggregate_latency(
                item, STATIC_TIME_FIELDS, STATIC_RUN_TIME_FIELDS, successful_only=False
            )
        else:
            cache_len = observed_cache_len(item)
            rt = prefill_rt(item)

        runs = item.get("runs") if isinstance(item.get("runs"), list) else []
        if all_runs and runs:
            samples = []
            for index, run in enumerate(runs, 1):
                if not isinstance(run, dict) or not run.get("success", True):
                    continue
                reuse = finite_number(run.get("reuse_len"))
                samples.append(
                    (cache_len if reuse is None else reuse, run_prefill_rt(run), index)
                )
        else:
            samples = [(cache_len, rt, None)]
        for cache, latency, index in samples:
            if (
                input_len is None
                or cache is None
                or latency is None
                or input_len < 0
                or cache < 0
                or cache > input_len
                or (strict and cache == input_len)
            ):
                continue
            row = {
                "input_len": input_len,
                "cache_len": cache,
                "compute_len": input_len - cache,
                "prefill_rt": latency,
            }
            if index is not None:
                row["run_index"] = index
            rows.append(row)
    if all_runs:
        return rows
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["input_len"], row["cache_len"], row["compute_len"])].append(row)
    return [
        {**values[0], "prefill_rt": median(item["prefill_rt"] for item in values)}
        for _, values in sorted(grouped.items())
    ]

"""Validation and aggregation of raw per-request decode measurements."""

from __future__ import annotations

import copy
import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional

from rtp_llm.test.perf_test.plan import freeze, thaw

_LATENCIES = ("wait", "prefill", "decode", "tpot", "total")


class _MeasurementError(ValueError):
    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


def _number(path: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _MeasurementError("invalid_timing", f"{path} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise _MeasurementError(
            "invalid_timing", f"{path} must be a non-negative finite number"
        )
    return parsed


def _identity(case: Mapping[str, Any]) -> Dict[str, Any]:
    return thaw(
        freeze(
            {
                key: copy.deepcopy(value)
                for key, value in case.items()
                if key != "case_id"
            }
        )
    )


def _failure(case: Mapping[str, Any], status: str, message: str) -> Dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "identity": _identity(case),
        "status": status,
        "runs": int(case["measure_runs"]),
        "errors": [{"code": status, "message": message}],
        "metrics": None,
    }


def _planned_lengths(case: Mapping[str, Any]) -> List[int]:
    if case["workload_kind"] == "distribution":
        lengths = list(case["input_lengths"])
    else:
        lengths = [int(case["input_length"])] * int(case["global_batch_size"])
    if len(lengths) != int(case["global_batch_size"]):
        raise _MeasurementError(
            "invalid_shape", "planned input lengths do not match global batch"
        )
    return lengths


def build_decode_case_result(
    case: Mapping[str, Any],
    *,
    measurements: Sequence[Mapping[str, Any]],
    topology: Mapping[str, int],
    search: Optional[Mapping[str, Any]] = None,
    distribution: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate raw runs and produce one common-schema decode case result."""
    try:
        expected_lengths = _planned_lengths(case)
        expected_runs = int(case["measure_runs"])
        if (
            not isinstance(measurements, (list, tuple))
            or len(measurements) != expected_runs
        ):
            raise _MeasurementError(
                "invalid_shape", "measurement count does not match planned runs"
            )
        parsed_topology = dict(topology)
        if set(parsed_topology) != {"tp", "dp", "pp", "ep"} or any(
            type(value) is not int or value <= 0 for value in parsed_topology.values()
        ):
            raise _MeasurementError(
                "invalid_shape", "decode topology requires positive tp/dp/pp/ep"
            )
        output_length = int(case["output_length"])
        normalized_measurements = []
        latency_values = {name: [] for name in _LATENCIES}
        throughputs = []
        successful = 0
        failed = 0
        request_errors = []
        for run_index, measurement in enumerate(measurements):
            if not isinstance(measurement, Mapping):
                raise _MeasurementError(
                    "invalid_shape", f"measurements[{run_index}] must be an object"
                )
            if measurement.get("run") != run_index:
                raise _MeasurementError(
                    "invalid_shape", "measurement run indices must be contiguous"
                )
            requests = measurement.get("requests")
            if not isinstance(requests, list) or len(requests) != len(expected_lengths):
                raise _MeasurementError(
                    "invalid_shape", "request count does not match global batch"
                )
            normalized_requests = []
            seen_indices = set()
            for request in requests:
                if not isinstance(request, Mapping):
                    raise _MeasurementError(
                        "invalid_shape", "request outcome must be an object"
                    )
                request_index = request.get("request_index")
                if type(request_index) is not int or request_index not in range(
                    len(expected_lengths)
                ):
                    raise _MeasurementError(
                        "invalid_shape", "request index is outside the global batch"
                    )
                if request_index in seen_indices:
                    raise _MeasurementError(
                        "invalid_shape", "request index appears more than once"
                    )
                seen_indices.add(request_index)
                if (
                    request.get("input_length") != expected_lengths[request_index]
                    or request.get("output_length") != output_length
                ):
                    raise _MeasurementError(
                        "invalid_shape",
                        "request input/output shape does not match plan",
                    )
                if type(request.get("success")) is not bool:
                    raise _MeasurementError(
                        "invalid_shape", "request success must be boolean"
                    )
                normalized_request = copy.deepcopy(dict(request))
                if request["success"]:
                    latency = request.get("latency_ms")
                    if not isinstance(latency, Mapping) or set(latency) != set(
                        _LATENCIES
                    ):
                        raise _MeasurementError(
                            "invalid_timing",
                            "successful request requires all latency fields",
                        )
                    normalized_latency = {
                        name: _number(f"latency_ms.{name}", latency[name])
                        for name in _LATENCIES
                    }
                    normalized_request["latency_ms"] = normalized_latency
                    for name, value in normalized_latency.items():
                        latency_values[name].append(value)
                    successful += 1
                else:
                    failed += 1
                    message = str(request.get("error", "request failed"))
                    request_errors.append(
                        {"code": "request_failed", "message": message}
                    )
                normalized_requests.append(normalized_request)
            if seen_indices != set(range(len(expected_lengths))):
                raise _MeasurementError(
                    "invalid_shape", "measurement does not cover every request"
                )
            throughput = _number(
                "throughput_tokens_per_second",
                measurement.get("throughput_tokens_per_second"),
            )
            throughputs.append(throughput)
            normalized_measurement = copy.deepcopy(dict(measurement))
            normalized_measurement["requests"] = normalized_requests
            normalized_measurement["throughput_tokens_per_second"] = throughput
            normalized_measurements.append(normalized_measurement)
    except _MeasurementError as error:
        return _failure(case, error.status, str(error))

    aggregate_latency = {
        name: statistics.fmean(values) if values else 0.0
        for name, values in latency_values.items()
    }
    distribution_details = None
    if case["workload_kind"] == "distribution":
        distribution_details = copy.deepcopy(
            dict(distribution or case.get("distribution", {}))
        )
        distribution_details.setdefault(
            "actual_sequence_lengths", list(expected_lengths)
        )
    metrics = {
        "workload_kind": case["workload_kind"],
        "input_lengths": list(expected_lengths),
        "output_length": int(case["output_length"]),
        "global_batch_size": int(case["global_batch_size"]),
        "topology": parsed_topology,
        "successful_requests": successful,
        "failed_requests": failed,
        "latency_ms": aggregate_latency,
        "throughput_tokens_per_second": statistics.fmean(throughputs),
        "measurements": normalized_measurements,
        "search": None if search is None else copy.deepcopy(dict(search)),
        "distribution": distribution_details,
    }
    status = "ok" if failed == 0 else "request_failed"
    return {
        "case_id": case["case_id"],
        "identity": _identity(case),
        "status": status,
        "runs": len(normalized_measurements),
        "errors": request_errors,
        "metrics": metrics,
    }

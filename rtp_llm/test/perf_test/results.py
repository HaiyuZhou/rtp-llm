"""Versioned common result envelope with discriminated mode metrics."""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from rtp_llm.test.perf_test.state import CaseState, RunState

RESULT_SCHEMA_VERSION = 1
_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")


class ResultSchemaError(ValueError):
    """Raised when a persisted result does not match the common envelope."""


def _object(path: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResultSchemaError(f"{path}: expected an object")
    return value


def _required(path: str, value: Mapping[str, Any], fields: set[str]) -> None:
    missing = sorted(fields - set(value))
    if missing:
        raise ResultSchemaError(f"{path}.{missing[0]}: required field is missing")


def _exact_fields(path: str, value: Mapping[str, Any], fields: set[str]) -> None:
    _required(path, value, fields)
    unknown = sorted(set(value) - fields)
    if unknown:
        raise ResultSchemaError(f"{path}.{unknown[0]}: unknown field")


def _string(path: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ResultSchemaError(f"{path}: expected a non-empty string")
    return value


def _count(path: str, value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ResultSchemaError(f"{path}: expected a non-negative integer")
    return value


def _positive_int(path: str, value: Any) -> int:
    parsed = _count(path, value)
    if parsed == 0:
        raise ResultSchemaError(f"{path}: expected a positive integer")
    return parsed


def _number(path: str, value: Any, *, positive: bool = False) -> Union[int, float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResultSchemaError(f"{path}: expected a finite number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0 or positive and parsed == 0:
        raise ResultSchemaError(f"{path}: expected a non-negative finite number")
    return value


def _record_list(path: str, value: Any) -> Tuple[Dict[str, Any], ...]:
    if not isinstance(value, list):
        raise ResultSchemaError(f"{path}: expected an array")
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ResultSchemaError(f"{path}[{index}]: expected an object")
        result.append(copy.deepcopy(item))
    return tuple(result)


@dataclass(frozen=True)
class PrefillMetrics:
    input_length: int
    expected_reuse_length: int
    observed_reuse_length: int
    compute_length: int
    latency_ms: Union[int, float]
    throughput_tokens_per_second: Union[int, float]
    validation: Dict[str, bool]
    seed: int
    prefix_policy: Dict[str, Any]
    measurements: Tuple[Dict[str, Any], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_length": self.input_length,
            "expected_reuse_length": self.expected_reuse_length,
            "observed_reuse_length": self.observed_reuse_length,
            "compute_length": self.compute_length,
            "latency_ms": self.latency_ms,
            "throughput_tokens_per_second": self.throughput_tokens_per_second,
            "validation": dict(self.validation),
            "seed": self.seed,
            "prefix_policy": copy.deepcopy(self.prefix_policy),
            "measurements": copy.deepcopy(list(self.measurements)),
        }


@dataclass(frozen=True)
class DecodeMetrics:
    workload_kind: str
    input_lengths: Tuple[int, ...]
    output_length: int
    global_batch_size: int
    topology: Dict[str, int]
    successful_requests: int
    failed_requests: int
    latency_ms: Dict[str, Union[int, float]]
    throughput_tokens_per_second: Union[int, float]
    measurements: Tuple[Dict[str, Any], ...]
    search: Optional[Dict[str, Any]]
    distribution: Optional[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workload_kind": self.workload_kind,
            "input_lengths": list(self.input_lengths),
            "output_length": self.output_length,
            "global_batch_size": self.global_batch_size,
            "topology": dict(self.topology),
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "latency_ms": dict(self.latency_ms),
            "throughput_tokens_per_second": self.throughput_tokens_per_second,
            "measurements": copy.deepcopy(list(self.measurements)),
            "search": copy.deepcopy(self.search),
            "distribution": copy.deepcopy(self.distribution),
        }


ModeMetrics = Union[PrefillMetrics, DecodeMetrics]


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    identity: Dict[str, Any]
    status: CaseState
    runs: int
    errors: Tuple[Dict[str, str], ...]
    metrics: Optional[ModeMetrics]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "identity": copy.deepcopy(self.identity),
            "status": self.status.value,
            "runs": self.runs,
            "errors": copy.deepcopy(list(self.errors)),
            "metrics": None if self.metrics is None else self.metrics.to_dict(),
        }


@dataclass(frozen=True)
class ResultEnvelope:
    schema_version: int
    run_id: str
    mode: str
    backend: str
    status: RunState
    config_fingerprint: str
    workload_fingerprint: str
    model: Dict[str, Any]
    summary: Dict[str, int]
    cases: Tuple[CaseResult, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "mode": self.mode,
            "backend": self.backend,
            "status": self.status.value,
            "config_fingerprint": self.config_fingerprint,
            "workload_fingerprint": self.workload_fingerprint,
            "model": copy.deepcopy(self.model),
            "summary": dict(self.summary),
            "cases": [case.to_dict() for case in self.cases],
        }


def _parse_prefill_metrics(path: str, value: Any) -> PrefillMetrics:
    raw = _object(path, value)
    fields = {
        "input_length",
        "expected_reuse_length",
        "observed_reuse_length",
        "compute_length",
        "latency_ms",
        "throughput_tokens_per_second",
        "validation",
        "seed",
        "prefix_policy",
        "measurements",
    }
    _exact_fields(path, raw, fields)
    input_length = _positive_int(f"{path}.input_length", raw["input_length"])
    expected = _count(f"{path}.expected_reuse_length", raw["expected_reuse_length"])
    observed = _count(f"{path}.observed_reuse_length", raw["observed_reuse_length"])
    compute = _positive_int(f"{path}.compute_length", raw["compute_length"])
    if expected + compute != input_length:
        raise ResultSchemaError(f"{path}: reuse plus compute must equal input length")
    validation = _object(f"{path}.validation", raw["validation"])
    _exact_fields(f"{path}.validation", validation, {"shape", "reuse", "timing"})
    if any(type(item) is not bool for item in validation.values()):
        raise ResultSchemaError(f"{path}.validation: values must be boolean")
    prefix_policy = _object(f"{path}.prefix_policy", raw["prefix_policy"])
    return PrefillMetrics(
        input_length=input_length,
        expected_reuse_length=expected,
        observed_reuse_length=observed,
        compute_length=compute,
        latency_ms=_number(f"{path}.latency_ms", raw["latency_ms"], positive=True),
        throughput_tokens_per_second=_number(
            f"{path}.throughput_tokens_per_second",
            raw["throughput_tokens_per_second"],
            positive=True,
        ),
        validation=dict(validation),
        seed=_count(f"{path}.seed", raw["seed"]),
        prefix_policy=copy.deepcopy(dict(prefix_policy)),
        measurements=_record_list(f"{path}.measurements", raw["measurements"]),
    )


def _parse_decode_metrics(path: str, value: Any) -> DecodeMetrics:
    raw = _object(path, value)
    fields = {
        "workload_kind",
        "input_lengths",
        "output_length",
        "global_batch_size",
        "topology",
        "successful_requests",
        "failed_requests",
        "latency_ms",
        "throughput_tokens_per_second",
        "measurements",
        "search",
        "distribution",
    }
    _exact_fields(path, raw, fields)
    kind = raw["workload_kind"]
    if kind not in {"grid", "distribution"}:
        raise ResultSchemaError(f"{path}.workload_kind: expected grid or distribution")
    input_lengths = raw["input_lengths"]
    if not isinstance(input_lengths, list) or not input_lengths:
        raise ResultSchemaError(f"{path}.input_lengths: expected a non-empty array")
    parsed_inputs = tuple(
        _positive_int(f"{path}.input_lengths[{index}]", item)
        for index, item in enumerate(input_lengths)
    )
    global_batch = _positive_int(f"{path}.global_batch_size", raw["global_batch_size"])
    if len(parsed_inputs) != global_batch:
        raise ResultSchemaError(
            f"{path}.global_batch_size: must equal the number of input lengths"
        )
    successful = _count(f"{path}.successful_requests", raw["successful_requests"])
    failed = _count(f"{path}.failed_requests", raw["failed_requests"])
    if successful + failed != global_batch:
        raise ResultSchemaError(
            f"{path}: request outcomes must equal global_batch_size"
        )
    topology = _object(f"{path}.topology", raw["topology"])
    _exact_fields(f"{path}.topology", topology, {"tp", "dp", "pp", "ep"})
    parsed_topology = {
        name: _positive_int(f"{path}.topology.{name}", topology[name])
        for name in ("tp", "dp", "pp", "ep")
    }
    latency = _object(f"{path}.latency_ms", raw["latency_ms"])
    latency_fields = {"wait", "prefill", "decode", "tpot", "total"}
    _exact_fields(f"{path}.latency_ms", latency, latency_fields)
    parsed_latency = {
        name: _number(f"{path}.latency_ms.{name}", latency[name])
        for name in sorted(latency_fields)
    }
    optional_records = {}
    for name in ("search", "distribution"):
        item = raw[name]
        if item is not None and not isinstance(item, dict):
            raise ResultSchemaError(f"{path}.{name}: expected an object or null")
        optional_records[name] = copy.deepcopy(item)
    return DecodeMetrics(
        workload_kind=kind,
        input_lengths=parsed_inputs,
        output_length=_positive_int(f"{path}.output_length", raw["output_length"]),
        global_batch_size=global_batch,
        topology=parsed_topology,
        successful_requests=successful,
        failed_requests=failed,
        latency_ms=parsed_latency,
        throughput_tokens_per_second=_number(
            f"{path}.throughput_tokens_per_second",
            raw["throughput_tokens_per_second"],
        ),
        measurements=_record_list(f"{path}.measurements", raw["measurements"]),
        search=optional_records["search"],
        distribution=optional_records["distribution"],
    )


def _parse_case(index: int, mode: str, value: Any) -> CaseResult:
    path = f"cases[{index}]"
    raw = _object(path, value)
    fields = {"case_id", "identity", "status", "runs", "errors", "metrics"}
    _exact_fields(path, raw, fields)
    try:
        status = CaseState(raw["status"])
    except ValueError as error:
        raise ResultSchemaError(f"{path}.status: unknown case state") from error
    identity = _object(f"{path}.identity", raw["identity"])
    errors = _record_list(f"{path}.errors", raw["errors"])
    parsed_errors = []
    for error_index, item in enumerate(errors):
        error_path = f"{path}.errors[{error_index}]"
        _exact_fields(error_path, item, {"code", "message"})
        parsed_errors.append(
            {
                "code": _string(f"{error_path}.code", item["code"]),
                "message": _string(f"{error_path}.message", item["message"]),
            }
        )
    metrics_value = raw["metrics"]
    if status == CaseState.OK and metrics_value is None:
        raise ResultSchemaError(f"{path}.metrics: required for ok case")
    metrics = None
    if metrics_value is not None:
        metrics = (
            _parse_prefill_metrics(f"{path}.metrics", metrics_value)
            if mode == "prefill"
            else _parse_decode_metrics(f"{path}.metrics", metrics_value)
        )
    return CaseResult(
        case_id=_string(f"{path}.case_id", raw["case_id"]),
        identity=copy.deepcopy(dict(identity)),
        status=status,
        runs=_count(f"{path}.runs", raw["runs"]),
        errors=tuple(parsed_errors),
        metrics=metrics,
    )


def parse_result_envelope(value: Any) -> ResultEnvelope:
    raw = _object("result", value)
    fields = {
        "schema_version",
        "run_id",
        "mode",
        "backend",
        "status",
        "config_fingerprint",
        "workload_fingerprint",
        "model",
        "summary",
        "cases",
    }
    _exact_fields("", raw, fields)
    if raw["schema_version"] != RESULT_SCHEMA_VERSION:
        raise ResultSchemaError("schema_version: unsupported result schema")
    mode = raw["mode"]
    if mode not in {"prefill", "decode"}:
        raise ResultSchemaError("mode: expected prefill or decode")
    backend = raw["backend"]
    if backend not in {"local", "aone"} or mode == "decode" and backend != "local":
        raise ResultSchemaError("backend: unsupported for result mode")
    try:
        status = RunState(raw["status"])
    except ValueError as error:
        raise ResultSchemaError("status: unknown run state") from error
    if status not in {
        RunState.COMPLETED,
        RunState.PARTIAL,
        RunState.FAILED,
        RunState.INTERRUPTED,
    }:
        raise ResultSchemaError("status: result requires a terminal run state")
    for name in ("config_fingerprint", "workload_fingerprint"):
        if not isinstance(raw[name], str) or not _FINGERPRINT.fullmatch(raw[name]):
            raise ResultSchemaError(f"{name}: expected a lowercase SHA-256 fingerprint")
    model = _object("model", raw["model"])
    _exact_fields("model", model, {"label", "type", "checkpoint", "topology"})
    topology = _object("model.topology", model["topology"])
    _exact_fields("model.topology", topology, {"tp", "dp", "pp", "ep"})
    parsed_model = {
        "label": _string("model.label", model["label"]),
        "type": _string("model.type", model["type"]),
        "checkpoint": _string("model.checkpoint", model["checkpoint"]),
        "topology": {
            name: _positive_int(f"model.topology.{name}", topology[name])
            for name in ("tp", "dp", "pp", "ep")
        },
    }
    summary = _object("summary", raw["summary"])
    summary_fields = {"planned_cases", "successful_cases", "failed_cases"}
    _exact_fields("summary", summary, summary_fields)
    parsed_summary = {
        name: _count(f"summary.{name}", summary[name]) for name in summary_fields
    }
    cases_value = raw["cases"]
    if not isinstance(cases_value, list):
        raise ResultSchemaError("cases: expected an array")
    cases = tuple(
        _parse_case(index, mode, item) for index, item in enumerate(cases_value)
    )
    if parsed_summary["planned_cases"] != len(cases):
        raise ResultSchemaError("summary.planned_cases: must equal case count")
    if parsed_summary["successful_cases"] + parsed_summary["failed_cases"] != len(
        cases
    ):
        raise ResultSchemaError("summary: successful plus failed must equal case count")
    return ResultEnvelope(
        schema_version=RESULT_SCHEMA_VERSION,
        run_id=_string("run_id", raw["run_id"]),
        mode=mode,
        backend=backend,
        status=status,
        config_fingerprint=raw["config_fingerprint"],
        workload_fingerprint=raw["workload_fingerprint"],
        model=parsed_model,
        summary=parsed_summary,
        cases=cases,
    )

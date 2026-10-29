"""Strict, side-effect-free configuration models for the unified perf test."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from rtp_llm.test.perf_test.errors import ConfigurationError

SCHEMA_VERSION = 1


class ConfigError(ConfigurationError, ValueError):
    """Raised when a unified perf-test config violates its schema."""


def _is_sensitive_name(name: str) -> bool:
    lower = name.lower()
    return (
        any(
            fragment in lower
            for fragment in (
                "password",
                "secret",
                "access_key",
                "api_key",
                "credential",
            )
        )
        or lower == "token"
        or lower.endswith("_token")
    )


def _reject_unsupported_forms(value: Any, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for name, item in value.items():
            if not isinstance(name, str):
                raise ConfigError(f"{path}: object keys must be strings")
            item_path = name if path == "config" else f"{path}.{name}"
            lower = name.lower()
            if lower in {"include", "includes", "$include"}:
                raise ConfigError(f"{item_path}: include directives are not supported")
            if _is_sensitive_name(name):
                raise ConfigError(
                    f"{item_path}: credentials and secret values must remain external"
                )
            _reject_unsupported_forms(item, item_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_unsupported_forms(item, f"{path}[{index}]")
    elif isinstance(value, str):
        if "$" in value or "`" in value:
            raise ConfigError(
                f"{path}: interpolation and shell evaluation are not supported"
            )
        if value.lower().startswith(("secret:", "vault:", "env:")):
            raise ConfigError(f"{path}: secret references are not supported")


def _object(path: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path}: expected an object")
    return value


def _reject_unknown(path: str, value: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        field_path = unknown[0] if not path else f"{path}.{unknown[0]}"
        expected = ", ".join(sorted(allowed))
        raise ConfigError(f"{field_path}: unknown field; expected one of: {expected}")


def _required(path: str, value: Mapping[str, Any], name: str) -> Any:
    if name not in value:
        field_path = name if not path else f"{path}.{name}"
        raise ConfigError(f"{field_path}: required field is missing")
    return value[name]


def _non_empty_string(path: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{path}: expected a non-empty string")
    return value


def _positive_int(path: str, value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise ConfigError(f"{path}: expected a positive integer")
    return value


def _non_negative_int(path: str, value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ConfigError(f"{path}: expected a non-negative integer")
    return value


def _positive_number(path: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path}: expected a positive finite number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ConfigError(f"{path}: expected a positive finite number")
    return parsed


def _boolean(path: str, value: Any) -> bool:
    if type(value) is not bool:
        raise ConfigError(f"{path}: expected a boolean")
    return value


def _int_tuple(
    path: str, value: Any, *, allow_zero: bool = False, allow_empty: bool = False
) -> Tuple[int, ...]:
    if not isinstance(value, list):
        raise ConfigError(f"{path}: expected an array of integers")
    if not value and not allow_empty:
        raise ConfigError(f"{path}: expected at least one value")
    parser = _non_negative_int if allow_zero else _positive_int
    return tuple(parser(f"{path}[{index}]", item) for index, item in enumerate(value))


def _extension_map(path: str, value: Any, *, scalar_only: bool) -> Dict[str, Any]:
    raw = _object(path, value)
    result: Dict[str, Any] = {}
    for name, item in raw.items():
        item_path = f"{path}.{name}"
        if scalar_only and (
            isinstance(item, bool)
            or not isinstance(item, (str, int, float))
            or isinstance(item, float)
            and not math.isfinite(item)
        ):
            raise ConfigError(f"{item_path}: expected a string or finite number")
        result[name] = copy.deepcopy(item)
    return result


@dataclass(frozen=True)
class TopologyConfig:
    tp: int = 1
    dp: int = 1
    pp: int = 1
    ep: int = 1

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.tp, self.dp, self.pp, self.ep


@dataclass(frozen=True)
class ModelConfig:
    label: str
    type: str
    checkpoint: str
    tokenizer: str
    topology: TopologyConfig


@dataclass(frozen=True)
class EngineConfig:
    max_seq_len: int
    max_batch_tokens: int
    concurrency_limit: int
    args: Dict[str, Any] = field(default_factory=dict)
    env: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PrefillWorkloadConfig:
    kind: str
    max_input_len: int
    cache_alignment: int
    commit_tail_tokens: int
    compute_steps: Tuple[int, ...]
    batch_sizes: Tuple[int, ...]
    measure_runs: int = 3
    min_compute_len: int = 1
    min_cache_len: int = 0
    max_cache_len: Optional[int] = None
    cache_lengths: Tuple[int, ...] = ()
    random_cache_count: int = 8
    seed: int = 1
    max_cases: Optional[int] = None


@dataclass(frozen=True)
class DecodeGridWorkloadConfig:
    kind: str
    input_lengths: Tuple[int, ...]
    output_length: int
    batch_sizes: Tuple[int, ...]
    measure_runs: int = 3
    warmup_runs: int = 1
    target_tpot_ms: Optional[float] = None
    generate_config: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecodeDistributionWorkloadConfig:
    kind: str
    distribution_file: str
    output_length: int
    batch_sizes: Tuple[int, ...]
    measure_runs: int = 3
    warmup_runs: int = 1
    target_tpot_ms: Optional[float] = None
    generate_config: Dict[str, Any] = field(default_factory=dict)


WorkloadConfig = Union[
    PrefillWorkloadConfig,
    DecodeGridWorkloadConfig,
    DecodeDistributionWorkloadConfig,
]


@dataclass(frozen=True)
class LocalExecutionConfig:
    backend: str
    devices: Tuple[int, ...] = ()


@dataclass(frozen=True)
class AOneExecutionConfig:
    backend: str
    resource: str
    gpu_type: str
    max_parallel: int
    shard_count: int


ExecutionConfig = Union[LocalExecutionConfig, AOneExecutionConfig]


@dataclass(frozen=True)
class OutputConfig:
    result_root: str = "perf_test_results"


@dataclass(frozen=True)
class RegressionConfig:
    enabled: bool = False
    baseline: Optional[str] = None
    threshold_pct: float = 10.0
    generate_candidate: bool = False


@dataclass(frozen=True)
class PublisherConfig:
    enabled: bool = False
    required: bool = False
    destination: Optional[str] = None
    artifacts: Tuple[str, ...] = ()


@dataclass(frozen=True)
class PublishConfig:
    oss: PublisherConfig = field(default_factory=PublisherConfig)
    odps: PublisherConfig = field(default_factory=PublisherConfig)


@dataclass(frozen=True)
class PerfTestConfig:
    schema_version: int
    model: ModelConfig
    engine: EngineConfig
    workload: WorkloadConfig
    execution: ExecutionConfig
    output: OutputConfig
    regression: RegressionConfig
    publish: PublishConfig


def _parse_topology(value: Any) -> TopologyConfig:
    raw = _object("model.topology", value)
    _reject_unknown("model.topology", raw, {"tp", "dp", "pp", "ep"})
    return TopologyConfig(
        **{
            name: _positive_int(f"model.topology.{name}", raw.get(name, 1))
            for name in ("tp", "dp", "pp", "ep")
        }
    )


def _parse_model(value: Any) -> ModelConfig:
    raw = _object("model", value)
    _reject_unknown(
        "model", raw, {"label", "type", "checkpoint", "tokenizer", "topology"}
    )
    checkpoint = _non_empty_string(
        "model.checkpoint", _required("model", raw, "checkpoint")
    )
    return ModelConfig(
        label=_non_empty_string("model.label", _required("model", raw, "label")),
        type=_non_empty_string("model.type", _required("model", raw, "type")),
        checkpoint=checkpoint,
        tokenizer=_non_empty_string(
            "model.tokenizer", raw.get("tokenizer", checkpoint)
        ),
        topology=_parse_topology(raw.get("topology", {})),
    )


def _parse_engine(value: Any) -> EngineConfig:
    raw = _object("engine", value)
    _reject_unknown(
        "engine",
        raw,
        {"max_seq_len", "max_batch_tokens", "concurrency_limit", "args", "env"},
    )
    return EngineConfig(
        max_seq_len=_positive_int(
            "engine.max_seq_len", _required("engine", raw, "max_seq_len")
        ),
        max_batch_tokens=_positive_int(
            "engine.max_batch_tokens",
            _required("engine", raw, "max_batch_tokens"),
        ),
        concurrency_limit=_positive_int(
            "engine.concurrency_limit",
            _required("engine", raw, "concurrency_limit"),
        ),
        args=_extension_map("engine.args", raw.get("args", {}), scalar_only=True),
        env=_extension_map("engine.env", raw.get("env", {}), scalar_only=True),
    )


def _parse_prefill_workload(raw: Mapping[str, Any]) -> PrefillWorkloadConfig:
    allowed = {
        "kind",
        "max_input_len",
        "cache_alignment",
        "commit_tail_tokens",
        "compute_steps",
        "batch_sizes",
        "measure_runs",
        "min_compute_len",
        "min_cache_len",
        "max_cache_len",
        "cache_lengths",
        "random_cache_count",
        "seed",
        "max_cases",
    }
    _reject_unknown("workload", raw, allowed)
    compute_steps = _int_tuple(
        "workload.compute_steps", _required("workload", raw, "compute_steps")
    )
    if len(set(compute_steps)) != len(compute_steps) or any(
        left <= right for left, right in zip(compute_steps, compute_steps[1:])
    ):
        raise ConfigError(
            "workload.compute_steps: expected unique coarse-to-fine values"
        )
    max_cache_len = raw.get("max_cache_len")
    max_cases = raw.get("max_cases")
    cache_lengths = _int_tuple(
        "workload.cache_lengths",
        raw.get("cache_lengths", []),
        allow_zero=True,
        allow_empty=True,
    )
    random_cache_count = _non_negative_int(
        "workload.random_cache_count",
        raw.get("random_cache_count", 0 if cache_lengths else 8),
    )
    if cache_lengths and random_cache_count:
        raise ConfigError(
            "workload.cache_lengths conflicts with workload.random_cache_count"
        )
    return PrefillWorkloadConfig(
        kind="cache_grid",
        max_input_len=_positive_int(
            "workload.max_input_len", _required("workload", raw, "max_input_len")
        ),
        cache_alignment=_positive_int(
            "workload.cache_alignment",
            _required("workload", raw, "cache_alignment"),
        ),
        commit_tail_tokens=_positive_int(
            "workload.commit_tail_tokens",
            _required("workload", raw, "commit_tail_tokens"),
        ),
        compute_steps=compute_steps,
        batch_sizes=_int_tuple(
            "workload.batch_sizes", _required("workload", raw, "batch_sizes")
        ),
        measure_runs=_positive_int("workload.measure_runs", raw.get("measure_runs", 3)),
        min_compute_len=_positive_int(
            "workload.min_compute_len", raw.get("min_compute_len", compute_steps[-1])
        ),
        min_cache_len=_non_negative_int(
            "workload.min_cache_len", raw.get("min_cache_len", 0)
        ),
        max_cache_len=(
            None
            if max_cache_len is None
            else _non_negative_int("workload.max_cache_len", max_cache_len)
        ),
        cache_lengths=cache_lengths,
        random_cache_count=random_cache_count,
        seed=_non_negative_int("workload.seed", raw.get("seed", 1)),
        max_cases=(
            None
            if max_cases is None
            else _positive_int("workload.max_cases", max_cases)
        ),
    )


def _decode_common(raw: Mapping[str, Any]) -> Dict[str, Any]:
    target = raw.get("target_tpot_ms")
    return {
        "output_length": _positive_int(
            "workload.output_length", _required("workload", raw, "output_length")
        ),
        "batch_sizes": _int_tuple(
            "workload.batch_sizes", _required("workload", raw, "batch_sizes")
        ),
        "measure_runs": _positive_int(
            "workload.measure_runs", raw.get("measure_runs", 3)
        ),
        "warmup_runs": _non_negative_int(
            "workload.warmup_runs", raw.get("warmup_runs", 1)
        ),
        "target_tpot_ms": (
            None
            if target is None
            else _positive_number("workload.target_tpot_ms", target)
        ),
        "generate_config": _extension_map(
            "workload.generate_config",
            raw.get("generate_config", {}),
            scalar_only=False,
        ),
    }


def _parse_workload(value: Any) -> WorkloadConfig:
    raw = _object("workload", value)
    kind = _non_empty_string("workload.kind", _required("workload", raw, "kind"))
    if kind == "cache_grid":
        return _parse_prefill_workload(raw)
    common = {
        "kind",
        "output_length",
        "batch_sizes",
        "measure_runs",
        "warmup_runs",
        "target_tpot_ms",
        "generate_config",
    }
    if kind == "grid":
        _reject_unknown("workload", raw, common | {"input_lengths"})
        return DecodeGridWorkloadConfig(
            kind=kind,
            input_lengths=_int_tuple(
                "workload.input_lengths",
                _required("workload", raw, "input_lengths"),
            ),
            **_decode_common(raw),
        )
    if kind == "distribution":
        _reject_unknown("workload", raw, common | {"distribution_file"})
        return DecodeDistributionWorkloadConfig(
            kind=kind,
            distribution_file=_non_empty_string(
                "workload.distribution_file",
                _required("workload", raw, "distribution_file"),
            ),
            **_decode_common(raw),
        )
    raise ConfigError("workload.kind: expected one of: cache_grid, distribution, grid")


def _parse_execution(value: Any) -> ExecutionConfig:
    raw = _object("execution", value)
    backend = _non_empty_string(
        "execution.backend", _required("execution", raw, "backend")
    )
    if backend == "local":
        _reject_unknown("execution", raw, {"backend", "devices"})
        return LocalExecutionConfig(
            backend=backend,
            devices=_int_tuple(
                "execution.devices",
                raw.get("devices", []),
                allow_zero=True,
                allow_empty=True,
            ),
        )
    if backend == "aone":
        _reject_unknown(
            "execution",
            raw,
            {"backend", "resource", "gpu_type", "max_parallel", "shard_count"},
        )
        return AOneExecutionConfig(
            backend=backend,
            resource=_non_empty_string(
                "execution.resource", _required("execution", raw, "resource")
            ),
            gpu_type=_non_empty_string(
                "execution.gpu_type", _required("execution", raw, "gpu_type")
            ),
            max_parallel=_positive_int(
                "execution.max_parallel",
                _required("execution", raw, "max_parallel"),
            ),
            shard_count=_positive_int(
                "execution.shard_count", _required("execution", raw, "shard_count")
            ),
        )
    raise ConfigError("execution.backend: expected one of: aone, local")


def _parse_output(value: Any) -> OutputConfig:
    raw = _object("output", value)
    _reject_unknown("output", raw, {"result_root"})
    return OutputConfig(
        result_root=_non_empty_string(
            "output.result_root", raw.get("result_root", "perf_test_results")
        )
    )


def _parse_regression(value: Any) -> RegressionConfig:
    raw = _object("regression", value)
    _reject_unknown(
        "regression",
        raw,
        {"enabled", "baseline", "threshold_pct", "generate_candidate"},
    )
    baseline = raw.get("baseline")
    config = RegressionConfig(
        enabled=_boolean("regression.enabled", raw.get("enabled", False)),
        baseline=(
            None
            if baseline is None
            else _non_empty_string("regression.baseline", baseline)
        ),
        threshold_pct=_positive_number(
            "regression.threshold_pct", raw.get("threshold_pct", 10.0)
        ),
        generate_candidate=_boolean(
            "regression.generate_candidate", raw.get("generate_candidate", False)
        ),
    )
    if config.enabled and config.baseline is None:
        raise ConfigError("regression.baseline: required when regression is enabled")
    return config


def _parse_publisher(path: str, value: Any) -> PublisherConfig:
    raw = _object(path, value)
    _reject_unknown(path, raw, {"enabled", "required", "destination", "artifacts"})
    destination = raw.get("destination")
    artifacts = raw.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise ConfigError(f"{path}.artifacts: expected an array of strings")
    normalized_artifacts = tuple(
        _non_empty_string(f"{path}.artifacts[{index}]", item)
        for index, item in enumerate(artifacts)
    )
    return PublisherConfig(
        enabled=_boolean(f"{path}.enabled", raw.get("enabled", False)),
        required=_boolean(f"{path}.required", raw.get("required", False)),
        destination=(
            None
            if destination is None
            else _non_empty_string(f"{path}.destination", destination)
        ),
        artifacts=normalized_artifacts,
    )


def _parse_publish(value: Any) -> PublishConfig:
    raw = _object("publish", value)
    _reject_unknown("publish", raw, {"oss", "odps"})
    return PublishConfig(
        oss=_parse_publisher("publish.oss", raw.get("oss", {})),
        odps=_parse_publisher("publish.odps", raw.get("odps", {})),
    )


def parse_config(value: Any) -> PerfTestConfig:
    """Validate an already-decoded config without reading files or environment."""
    raw = _object("config", value)
    _reject_unsupported_forms(raw)
    allowed = {
        "schema_version",
        "model",
        "engine",
        "workload",
        "execution",
        "output",
        "regression",
        "publish",
    }
    _reject_unknown("", raw, allowed)
    version = _required("", raw, "schema_version")
    if type(version) is not int or version != SCHEMA_VERSION:
        raise ConfigError(
            f"schema_version: expected integer {SCHEMA_VERSION}, got {version!r}"
        )
    return PerfTestConfig(
        schema_version=version,
        model=_parse_model(_required("", raw, "model")),
        engine=_parse_engine(_required("", raw, "engine")),
        workload=_parse_workload(_required("", raw, "workload")),
        execution=_parse_execution(_required("", raw, "execution")),
        output=_parse_output(_required("", raw, "output")),
        regression=_parse_regression(_required("", raw, "regression")),
        publish=_parse_publish(_required("", raw, "publish")),
    )

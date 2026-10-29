"""Canonical config resolution and fingerprints for unified perf tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from rtp_llm.test.perf_test.config_schema import ConfigError, parse_config

DEFAULT_MAX_EXTERNAL_BYTES = 1024 * 1024


class ConfigResolutionError(ConfigError):
    """Raised when source config or external workload materialization fails."""


@dataclass(frozen=True)
class ResolvedConfig:
    resolved_config: Dict[str, Any]
    config_fingerprint: str
    workload_fingerprint: str


def canonical_json(value: Any) -> str:
    """Return the single canonical JSON representation used for fingerprints."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ConfigResolutionError(f"config is not canonical JSON: {error}") from error


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _load_external_json(path: Path, max_bytes: int) -> Any:
    field = "workload.distribution_file"
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ConfigResolutionError("max_external_bytes must be a positive integer")
    try:
        size = path.stat().st_size
    except OSError as error:
        raise ConfigResolutionError(f"{field}: cannot read {path}: {error}") from error
    if size > max_bytes:
        raise ConfigResolutionError(
            f"{field}: {path} is too large ({size} bytes; limit {max_bytes})"
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as error:
        raise ConfigResolutionError(f"{field}: {path} is not UTF-8") from error
    except json.JSONDecodeError as error:
        raise ConfigResolutionError(
            f"{field}: {path} is not valid JSON: {error.msg}"
        ) from error
    except OSError as error:
        raise ConfigResolutionError(f"{field}: cannot read {path}: {error}") from error


def _materialize_workload(
    source: Mapping[str, Any],
    normalized: Dict[str, Any],
    source_dir: Path,
    max_external_bytes: int,
) -> None:
    source_workload = source["workload"]
    if source_workload.get("kind") != "distribution":
        return
    source_path = Path(source_workload["distribution_file"])
    if not source_path.is_absolute():
        source_path = source_dir / source_path
    normalized_workload = normalized["workload"]
    normalized_workload.pop("distribution_file", None)
    normalized_workload["distribution"] = _load_external_json(
        source_path, max_external_bytes
    )


def resolve_config(
    value: Any,
    *,
    source_dir: Optional[Union[str, Path]] = None,
    max_external_bytes: int = DEFAULT_MAX_EXTERNAL_BYTES,
) -> ResolvedConfig:
    """Validate, normalize, materialize, and fingerprint a decoded config."""
    parsed = parse_config(value)
    normalized = asdict(parsed)
    base_dir = Path.cwd() if source_dir is None else Path(source_dir)
    _materialize_workload(value, normalized, base_dir, max_external_bytes)

    config_fields = {
        name: section for name, section in normalized.items() if name != "workload"
    }
    return ResolvedConfig(
        resolved_config=normalized,
        config_fingerprint=_fingerprint(config_fields),
        workload_fingerprint=_fingerprint(normalized["workload"]),
    )


def load_resolved_config(
    path: Union[str, Path],
    *,
    max_external_bytes: int = DEFAULT_MAX_EXTERNAL_BYTES,
) -> ResolvedConfig:
    """Load a source JSON config and resolve relative workload inputs."""
    config_path = Path(path)
    try:
        source = json.loads(config_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as error:
        raise ConfigResolutionError(f"config {config_path} is not UTF-8") from error
    except json.JSONDecodeError as error:
        raise ConfigResolutionError(
            f"config {config_path} is not valid JSON: {error.msg}"
        ) from error
    except OSError as error:
        raise ConfigResolutionError(
            f"cannot read config {config_path}: {error}"
        ) from error
    return resolve_config(
        source,
        source_dir=config_path.parent,
        max_external_bytes=max_external_bytes,
    )

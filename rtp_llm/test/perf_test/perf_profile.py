"""Versioned JSON profile for the perf-test toolchain.

A profile centralises model paths, engine topology, cache-grid defaults,
chart labels, and formula-fit knobs that were previously hard-coded for
DeepSeek-V4-Pro across four scripts.  Every consumer resolves values through
the same priority chain:

    CLI explicit (including explicit ``0``) > profile field > input result
    metadata (embedded profile) > legacy default.

The fingerprint is ``sha256`` of the canonical JSON (sorted keys, no
whitespace, ``ensure_ascii``).  Two profiles with the same observable
fields always share the same fingerprint; adding an optional field that
was previously absent changes the fingerprint (the field is part of the
canonical form).

``engine.dp_size``, ``engine.max_seq_len``, and ``engine.concurrency_limit``
are consumed by the perf-test ``args`` namespace; all other engine keys
(``model_type``, ``checkpoint_path``, ``tokenizer_path``, ``tp_size``,
``seq_size_per_block``, ``engine_args.*``) are injected into the forwarded
``remaining`` argv.  CLI ``--test_arg`` values always win over profile.

``engine_args`` values must be strings or integers.  ``False`` is rejected
because the engine CLI does not distinguish it from ``"False"``; use ``0``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCHEMA_VERSION = 1

_ENGINE_ARGS_NAMESPACE_KEYS = frozenset(
    {
        "dp_size",
        "max_seq_len",
        "concurrency_limit",
    }
)


class ProfileError(ValueError):
    """Raised when a profile fails validation."""


def load_profile(path: str | Path) -> Dict[str, Any]:
    """Load and validate a profile from disk."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        profile = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProfileError(f"profile {path}: invalid JSON: {exc}") from exc
    validate_profile(profile)
    return profile


def validate_profile(profile: Any) -> None:
    """Validate the structure of a profile dict.

    Raises :class:`ProfileError` on any schema violation.  Unknown top-level
    keys are allowed (forward compatibility); missing optional sections are
    filled with empty dicts by the caller.
    """
    if not isinstance(profile, dict):
        raise ProfileError("profile must be a JSON object")
    version = profile.get("schema_version")
    if version is None:
        raise ProfileError("profile is missing schema_version")
    try:
        version = int(version)
    except (TypeError, ValueError):
        raise ProfileError(f"schema_version must be an integer, got {version!r}")
    if version != SCHEMA_VERSION:
        raise ProfileError(
            f"unsupported schema_version {version}; this tool supports {SCHEMA_VERSION}"
        )
    engine = profile.get("engine")
    if engine is not None and not isinstance(engine, dict):
        raise ProfileError("engine must be a JSON object")
    engine_args = profile.get("engine_args")
    if engine_args is not None and not isinstance(engine_args, dict):
        raise ProfileError("engine_args must be a JSON object")
    if isinstance(engine_args, dict):
        for key, value in engine_args.items():
            if isinstance(value, bool):
                raise ProfileError(
                    f"engine_args.{key}: bool is not a valid CLI value; use 0 or 1"
                )
            if not isinstance(value, (str, int, float)):
                raise ProfileError(
                    f"engine_args.{key}: expected string or number, got {type(value).__name__}"
                )
    cache_grid = profile.get("cache_grid")
    if cache_grid is not None and not isinstance(cache_grid, dict):
        raise ProfileError("cache_grid must be a JSON object")
    chart = profile.get("chart")
    if chart is not None and not isinstance(chart, dict):
        raise ProfileError("chart must be a JSON object")


def fingerprint(profile: Dict[str, Any]) -> str:
    """Deterministic SHA-256 fingerprint of a profile dict."""
    canonical = json.dumps(
        profile, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _section(profile: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = profile.get(key)
    return value if isinstance(value, dict) else {}


def engine_section(profile: Dict[str, Any]) -> Dict[str, Any]:
    return _section(profile, "engine")


def engine_args_section(profile: Dict[str, Any]) -> Dict[str, Any]:
    return _section(profile, "engine_args")


def cache_grid_section(profile: Dict[str, Any]) -> Dict[str, Any]:
    return _section(profile, "cache_grid")


def chart_section(profile: Dict[str, Any]) -> Dict[str, Any]:
    return _section(profile, "chart")


def resolve_int(
    profile: Dict[str, Any],
    section_name: str,
    key: str,
    cli_value: Optional[int],
    default: int,
) -> int:
    """Resolve an integer through the priority chain.

    CLI explicit (including ``0``) wins.  Then profile section key.  Then
    the supplied default.  ``None`` in the profile is treated as absent.
    """
    if cli_value is not None:
        return int(cli_value)
    section = _section(profile, section_name)
    raw = section.get(key)
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            logging.warning(
                "profile: %s.%s = %r is not a valid integer; using default %d",
                section_name,
                key,
                raw,
                default,
            )
    return default


def resolve_str(
    profile: Dict[str, Any],
    section_name: str,
    key: str,
    cli_value: Optional[str],
    default: str,
) -> str:
    """Resolve a string through the priority chain."""
    if cli_value is not None and cli_value != "":
        return cli_value
    section = _section(profile, section_name)
    raw = section.get(key)
    if raw is not None and raw != "":
        return str(raw)
    return default


def resolve_label(
    profile: Optional[Dict[str, Any]],
    cli_model_label: Optional[str],
    default: str,
) -> str:
    """Resolve a display label: CLI > profile.chart.model_label > default."""
    if cli_model_label is not None and cli_model_label != "":
        return cli_model_label
    if profile is not None:
        chart = chart_section(profile)
        raw = chart.get("model_label")
        if raw is not None and raw != "":
            return str(raw)
    return default


def resolve_title(
    profile: Optional[Dict[str, Any]],
    cli_title: Optional[str],
    default: str,
) -> str:
    """Resolve a chart title: CLI > profile.chart.title > default."""
    if cli_title is not None and cli_title != "":
        return cli_title
    if profile is not None:
        chart = chart_section(profile)
        raw = chart.get("title")
        if raw is not None and raw != "":
            return str(raw)
    return default


def engine_args_tokens(profile: Dict[str, Any]) -> List[str]:
    """Flatten ``engine_args`` into CLI token pairs.

    Keys may optionally start with ``--``; the output always uses ``--key
    value`` form.  Both ``engine_args`` and ``engine`` sections contribute:
    ``engine`` keys that are not in the namespace set (``dp_size``,
    ``max_seq_len``, ``concurrency_limit``) are emitted as ``--key value``.
    """
    tokens: List[str] = []
    engine = engine_section(profile)
    for key, value in sorted(engine.items()):
        if key in _ENGINE_ARGS_NAMESPACE_KEYS:
            continue
        cli_key = key if key.startswith("--") else f"--{key}"
        tokens.extend([cli_key, str(value)])
    engine_args = engine_args_section(profile)
    for key, value in sorted(engine_args.items()):
        cli_key = key if key.startswith("--") else f"--{key}"
        tokens.extend([cli_key, str(value)])
    return tokens


def _extract_arg_value(
    argv: Sequence[str], key: str
) -> Tuple[Optional[str], Optional[int]]:
    """Find the value of ``--key`` in an argv list.

    Returns ``(value, index)`` where ``index`` is the position of the flag
    token.  Handles ``--key value``, ``--key=value``, and returns ``(None,
    None)`` when absent.
    """
    flag = f"--{key}"
    prefix = f"--{key}="
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return argv[i + 1], i
        if arg.startswith(prefix):
            return arg[len(prefix) :], i
    return None, None


def set_engine_arg(remaining: List[str], key: str, value: str) -> List[str]:
    """Set or update ``--key value`` in a forwarded argv list.

    Returns a new list.  Handles ``--key value``, ``--key=value``, and
    bare ``--key`` (boolean flag) forms.  If the key is absent, appends
    ``--key value`` at the end.
    """
    out = list(remaining)
    flag = f"--{key}"
    prefix = f"--{key}="
    for i, arg in enumerate(out):
        if arg == flag:
            if i + 1 < len(out) and not out[i + 1].startswith("--"):
                out[i + 1] = str(value)
            else:
                out.insert(i + 1, str(value))
            return out
        if arg.startswith(prefix):
            out[i] = f"--{key}={value}"
            return out
    out.extend([f"--{key}", str(value)])
    return out


def merge_engine_args(
    profile: Dict[str, Any],
    remaining: List[str],
    *,
    skip_namespace_keys: bool = True,
) -> List[str]:
    """Merge profile engine args into the forwarded argv list.

    CLI values already in ``remaining`` always win.  Keys in
    ``_ENGINE_ARGS_NAMESPACE_KEYS`` (``dp_size``, ``max_seq_len``,
    ``concurrency_limit``) are skipped by default — they belong in the
    ``args`` namespace, not the forwarded argv.
    """
    out = list(remaining)
    engine = engine_section(profile)
    for key, value in sorted(engine.items()):
        if skip_namespace_keys and key in _ENGINE_ARGS_NAMESPACE_KEYS:
            continue
        existing, _ = _extract_arg_value(out, key)
        if existing is not None:
            continue
        out = set_engine_arg(out, key, str(value))
    engine_args = engine_args_section(profile)
    for key, value in sorted(engine_args.items()):
        clean_key = key.lstrip("-")
        existing, _ = _extract_arg_value(out, clean_key)
        if existing is not None:
            continue
        out = set_engine_arg(out, clean_key, str(value))
    return out


def extract_embedded_profile(data: Any) -> Optional[Dict[str, Any]]:
    """Extract an embedded profile from a result JSON payload.

    The runner writes ``profile`` as a top-level key in the result JSON.
    Returns ``None`` when the key is absent or not a dict.
    """
    if not isinstance(data, dict):
        return None
    embedded = data.get("profile")
    if isinstance(embedded, dict):
        return embedded
    return None

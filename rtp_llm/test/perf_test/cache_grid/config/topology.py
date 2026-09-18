"""GPU topology shared by benchmark and production throughput charts."""

import json
import math
from pathlib import Path


def _count(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float("nan")
    if (
        isinstance(value, bool)
        or not math.isfinite(parsed)
        or parsed <= 0
        or not parsed.is_integer()
    ):
        raise ValueError("profile GPU counts must be positive integers")
    return int(parsed)


def profile_cards(profile: dict | None) -> int | None:
    """Use world size or TP x DP x PP; EP and CP do not multiply GPU counts."""
    engine = (profile or {}).get("engine") or {}
    if "world_size" in engine:
        return _count(engine["world_size"])
    if any(key in engine for key in ("tp_size", "dp_size", "pp_size")):
        return math.prod(
            _count(engine.get(key, 1)) for key in ("tp_size", "dp_size", "pp_size")
        )
    return None


def detect_cards(input_path: Path, profile: dict | None = None) -> int | None:
    """Prefer explicit/embedded profile topology, then legacy run configuration."""
    cards = profile_cards(profile)
    if cards is not None:
        return cards
    try:
        data = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    embedded = data.get("profile")
    cards = profile_cards(embedded if isinstance(embedded, dict) else None)
    if cards is not None:
        return cards
    config = data.get("run_config")
    if not isinstance(config, dict):
        return None
    engine = config.get("engine")
    engine = engine if isinstance(engine, dict) else {}
    keys = ("world_size", "tp_size", "dp_size", "pp_size")
    topology = {key: config[key] for key in keys if key in config}
    argv = engine.get("args", [])
    if isinstance(argv, list):
        for index, arg in enumerate(argv):
            if not isinstance(arg, str):
                continue
            name, sep, value = arg.partition("=")
            key = name.removeprefix("--")
            if name.startswith("--") and key in keys:
                if not sep and index + 1 < len(argv):
                    value = argv[index + 1]
                try:
                    topology[key] = _count(value)
                except ValueError:
                    continue
    topology.update({key: engine[key] for key in keys if key in engine})
    return profile_cards({"engine": topology})

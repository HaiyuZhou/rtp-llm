"""Deliver mode-aware AOne CI configuration without importing the runtime."""

import argparse
import json
import os
from pathlib import Path
from typing import Mapping


def deliver_config(env: Mapping[str, str], checkout: Path, destination: Path) -> dict:
    """Apply explicit CI parameters to a config and write the delivered JSON.

    Missing, empty, and ``auto`` parameters inherit the selected config. The
    default config is mode-specific; an explicit config retains its own values.
    """

    def parameter(name):
        value = env.get("PERF_TEST_" + name, "").strip()
        return None if value in ("", "auto") else value

    mode = parameter("MODE") or "decode"
    if mode not in ("decode", "prefill"):
        raise ValueError("PERF_TEST_MODE must be 'decode' or 'prefill'")
    spec = parameter("CONFIG") or (
        "rtp_llm/test/perf_test/examples/"
        + ("prefill_aone_ci.json" if mode == "prefill" else "decode_aone_gang.json")
    )
    source = (
        spec if spec.startswith("{") else (checkout / spec).read_text(encoding="utf-8")
    )
    base = json.loads(source)
    if not isinstance(base, dict):
        raise ValueError("config must be a JSON object")
    model = base.setdefault("model", {})
    for key, name in [
        ("label", "MODEL_LABEL"),
        ("type", "MODEL_TYPE"),
        ("checkpoint", "CHECKPOINT_PATH"),
        ("tokenizer", "TOKENIZER_PATH"),
    ]:
        value = parameter(name)
        if value is not None:
            model[key] = value

    def positive(value, name):
        if isinstance(value, bool) or not str(value).isdigit() or int(value) < 1:
            raise ValueError(f"{name} must be a positive integer")
        return int(value)

    topology = model.setdefault("topology", {})
    for key in ("tp", "dp", "pp", "ep"):
        name = key.upper() + "_SIZE"
        topology[key] = positive(parameter(name) or topology.get(key, 1), name)
    execution = base.setdefault("execution", {})
    gpus = positive(
        parameter("GPUS_PER_MACHINE") or execution.get("gpus_per_machine", 1),
        "GPUS_PER_MACHINE",
    )
    world = topology["tp"] * topology["dp"] * topology["pp"]
    if world % gpus:
        raise ValueError(
            f"world_size {world} is not divisible by gpus_per_machine {gpus}"
        )
    execution["backend"] = "aone"
    for key in ("gpu_type", "resource"):
        value = parameter(key.upper())
        if value is not None:
            execution[key] = value
    execution["gpus_per_machine"] = gpus
    execution["max_parallel"] = max(
        int(execution.get("max_parallel") or 0), world // gpus
    )
    if mode == "prefill":
        # Gate discovery and CI upload both consume this fixed artifact root.
        base.setdefault("output", {})["result_root"] = "perf_test_results"
    destination.write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, default=Path.cwd())
    parser.add_argument(
        "--destination", type=Path, default=Path("multi_node_perf_config.json")
    )
    args = parser.parse_args()
    print(
        json.dumps(
            deliver_config(os.environ, args.checkout, args.destination), indent=2
        )
    )


if __name__ == "__main__":
    main()

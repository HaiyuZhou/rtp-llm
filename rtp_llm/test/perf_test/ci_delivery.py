"""Deliver mode-aware AOne CI configuration without importing the runtime."""

import argparse
import json
import os
import subprocess
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
    spec = parameter("CONFIG")
    source_dir = checkout.resolve()
    if spec is None:
        relative = Path("rtp_llm/test/perf_test/examples") / (
            "prefill_aone_ci.json" if mode == "prefill" else "decode_aone_gang.json"
        )
        public_checkout = (
            checkout / "github-opensource"
            if (checkout / "github-opensource").is_dir()
            else checkout
        )
        source_path = (public_checkout / relative).resolve()
        source_dir = source_path.parent
        source = source_path.read_text(encoding="utf-8")
    else:
        if spec.startswith("{"):
            source = spec
        else:
            source_path = (checkout / spec).resolve()
            source_dir = source_path.parent
            source = source_path.read_text(encoding="utf-8")
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
    for key in ("tp", "dp", "pp"):
        name = key.upper() + "_SIZE"
        topology[key] = positive(parameter(name) or topology.get(key, 1), name)
    topology["ep"] = positive(
        parameter("EP_SIZE") or topology.get("ep", topology["tp"] * topology["dp"]),
        "EP_SIZE",
    )
    execution = base.setdefault("execution", {})
    gpu_override = parameter("GPUS_PER_MACHINE")
    gpus = gpu_override or execution.get("gpus_per_machine")
    world = topology["tp"] * topology["dp"] * topology["pp"]
    if gpus is not None:
        gpus = positive(gpus, "GPUS_PER_MACHINE")
        if world > gpus and world % gpus:
            raise ValueError(
                f"world_size {world} is not divisible by gpus_per_machine {gpus}"
            )
        execution["gpus_per_machine"] = gpus
        machine_count = max(1, world // gpus)
        if gpu_override is not None and machine_count > 1:
            # An explicit gang override replaces single-host case sharding.
            execution.pop("shard_count", None)
        execution["max_parallel"] = max(
            int(execution.get("max_parallel") or 0), machine_count
        )
    execution["backend"] = "aone"
    for key in ("gpu_type", "resource"):
        value = parameter(key.upper())
        if value is not None:
            execution[key] = value

    # The public entry materializes these files before packaging remote inputs.
    # Preserve their original base directory when relocating the source JSON.
    def rebase_distributions(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "distribution_file" and isinstance(child, str):
                    path = Path(child)
                    if not path.is_absolute():
                        value[key] = str((source_dir / path).resolve())
                else:
                    rebase_distributions(child)
        elif isinstance(value, list):
            for child in value:
                rebase_distributions(child)

    rebase_distributions(base.get("workload", {}))
    # Both mode gates and the always-run upload consume this artifact root.
    base.setdefault("output", {})["result_root"] = "perf_test_results"
    destination.write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")
    return base


def record_repository_revisions(checkout: Path) -> dict:
    """Record available checkouts; the paired internal parent is optional."""
    checkout = checkout.resolve()
    repositories = [("public", checkout)]
    if (checkout.parent / "internal_source").is_dir():
        repositories.append(("internal", checkout.parent))
    revisions = {
        name: subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        for name, repo in repositories
    }
    destination = checkout / "perf_test_results/repository_revisions.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(revisions, sort_keys=True) + "\n", encoding="utf-8"
    )
    return revisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, default=Path.cwd())
    parser.add_argument(
        "--destination", type=Path, default=Path("multi_node_perf_config.json")
    )
    parser.add_argument("--record-provenance", action="store_true")
    args = parser.parse_args()
    if args.record_provenance:
        print(json.dumps(record_repository_revisions(args.checkout), sort_keys=True))
        return
    print(
        json.dumps(
            deliver_config(os.environ, args.checkout, args.destination), indent=2
        )
    )


if __name__ == "__main__":
    main()

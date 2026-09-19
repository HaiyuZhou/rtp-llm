#!/usr/bin/env python3
"""Run explicit fixed-batch grids serially with a fresh service for each file."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

# Keep the canonical script usable both as a module and by file path.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from rtp_llm.test.perf_test.cache_grid.runner.workspace_budget import (
    WORKSPACE_TOKENS,
    grid_token_budget,
    validate_fixed_workspace,
)

REPO_ROOT = Path(__file__).resolve().parents[5]
TARGET = "//rtp_llm/test/perf_test:cache_grid_perf_test"


def inspect_grid(path):
    with path.open(encoding="utf-8") as stream:
        grid = json.load(stream)
    cases = grid.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{path}: requires a nonempty explicit cases list")
    batch_sizes = set()
    metadata = grid.get("generator", {})
    block = int(metadata.get("cache_alignment", 4096))
    tail = int(metadata.get("parameters", {}).get("commit_tail_tokens", 4096))
    # This launcher retains the user's CP8 / physical block512 configuration.
    if block != 4096 or tail <= 0 or tail % block:
        raise ValueError(
            f"{path}: requires cache alignment4096 and aligned positive commit tail"
        )
    for case in cases:
        batch = int(case["batch_size"])
        groups = case.get("request_groups") or [
            {
                "count": batch,
                "input_len": case["input_len"],
                "cache_len": case.get("cache_len", 0),
            }
        ]
        if batch <= 0 or sum(int(g["count"]) for g in groups) != batch:
            raise ValueError(f"{path}: group counts must sum to positive batch size")
        for group in groups:
            count = int(group["count"])
            length = int(group["input_len"])
            cached = int(group.get("cache_len", 0))
            if count <= 0 or not 0 < length <= 262144 or not 0 <= cached < length:
                raise ValueError(f"{path}: invalid group or input above256K")
            if cached and (cached % block or cached % tail or cached + tail > length):
                raise ValueError(f"{path}: invalid cache alignment or commit space")
        batch_sizes.add(batch)
    if len(batch_sizes) != 1:
        raise ValueError(f"{path}: split mixed batch sizes into separate files")
    budget = grid_token_budget(grid)
    capacity = validate_fixed_workspace(
        cases, commit_tail=tail, block=block, token_budget=budget
    )
    return {
        "grid_json": str(path.resolve()),
        "batch_size": batch_sizes.pop(),
        "max_seq_len": WORKSPACE_TOKENS,
        "max_context_batch_size": 1,
        "max_batch_tokens": budget,
        "workspace_capacity": capacity,
        "commit_tail_tokens": tail,
    }


def build_command(args, plan):
    command = [args.bazelisk]
    if args.output_base:
        command.append(f"--output_base={args.output_base.resolve()}")
    command += [
        "test",
        TARGET,
        "--config=cuda13",
        "--config=sm10x",
        "--test_timeout=345600",
        "--test_output=streamed",
        "--nocache_test_results",
    ]
    batch = plan["batch_size"]
    engine = {
        "partial": 2,
        "decode_test_length": 1,
        "concurrency_limit": batch,
        "max_context_batch_size": 1,
        "model_type": "deepseek_v4",
        "checkpoint_path": args.model_dir,
        "tokenizer_path": args.model_dir,
        "max_seq_len": plan["max_seq_len"],
        "max_batch_tokens_size": plan["max_batch_tokens"],
        "tp_size": 8,
        "ep_size": 8,
        "world_size": 8,
        "dp_size": 1,
        "cp_rotate_method": "ALL_GATHER",
        "prefill_cp_kv_cache_sharded": 1,
        "seq_size_per_block": 512,
        "kernel_seq_size_per_block": 128,
        "fp8_kv_cache": 1,
        "use_deepep_moe": 1,
        "use_deepep_low_latency": 0,
        "act_type": "BF16",
        "load_method": "fastsafetensors",
        "reserver_runtime_mem_mb": args.reserve_runtime_mem_mb,
        "cache_grid_json": plan["grid_json"],
        "result_dir": plan["result_dir"],
        "expected_cache_block_size": 4096,
        "cache_commit_tail_tokens": plan["commit_tail_tokens"],
        "cache_profile_runs": 0,
        "cache_request_transport": "dashsc_input_ids",
    }
    command += [f"--test_arg=--{name}={value}" for name, value in engine.items()]
    command.append("--test_arg=--cache_fixed_workspace")
    cc = shutil.which("gcc") or "gcc"
    cxx = shutil.which("g++") or "g++"
    env = {
        "WORLD_SIZE": "8",
        "DG_JIT_CPP_STANDARD": "20",
        "DSV4_USE_MEGA_MOE_SE": str(args.mega_moe_se),
        # Mega-SE belongs to the MegaMoE family; both flags must stay enabled.
        "DSV4_USE_MEGA_MOE": "1",
        "DSV4_CHUNK_TOKENS": "8192",
        "DSV4_PREFILL_CP_OVERLAP": "0",
        "PERF_PROFILE_RUNS": "0",
        "TOKENIZERS_PARALLELISM": "false",
        "CC": cc,
        "CXX": cxx,
        "CUDAHOSTCXX": cxx,
        "NVCC_PREPEND_FLAGS": f"-ccbin={cxx}",
        "PATH": os.environ.get("PATH", ""),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
    }
    if args.jit_cache_dir:
        root = args.jit_cache_dir.resolve()
        env.update(
            {
                "DG_JIT_CACHE_DIR": str(root / "jit_cache"),
                "TRITON_CACHE_DIR": str(root / "triton_cache"),
                "TILELANG_CACHE_DIR": str(root / "tilelang_cache"),
            }
        )
    command += [f"--test_env={name}={value}" for name, value in env.items()]
    extra = args.bazel_args[1:] if args.bazel_args[:1] == ["--"] else args.bazel_args
    # Prevent hidden overrides of the per-grid length/batch configuration.
    for value in extra:
        if value.startswith("--test_arg") or not value.startswith("--"):
            raise ValueError(
                "extra options must be Bazel flags; test_arg overrides are unsupported"
            )
    return command + extra


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    grids = parser.add_mutually_exclusive_group(required=True)
    grids.add_argument("--grid-dir", type=Path)
    grids.add_argument("--grid-json", type=Path, nargs="+")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-base", type=Path)
    parser.add_argument("--jit-cache-dir", type=Path)
    parser.add_argument(
        "--mega-moe-se",
        type=int,
        choices=[0, 1],
        default=1,
        help="1 for the current Pro setup; set0 for Flash MegaMoE",
    )
    parser.add_argument("--reserve-runtime-mem-mb", type=int, default=81920)
    parser.add_argument("--bazelisk", default="bazelisk")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("bazel_args", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    files = sorted(args.grid_dir.glob("*.json")) if args.grid_dir else args.grid_json
    if not files:
        raise SystemExit("No JSON grids found")
    if len({p.stem for p in files}) != len(files):
        raise SystemExit("Grid filenames must have distinct stems")
    plans = []
    try:
        for path in files:
            plan = inspect_grid(path)
            plan["result_dir"] = str((args.result_root / path.stem).resolve())
            if Path(plan["result_dir"]).exists() and not args.dry_run:
                raise ValueError(
                    f"Result already exists; use a new result-root: {plan['result_dir']}"
                )
            plan["command"] = build_command(args, plan)
            plan["status"] = "pending"
            plans.append(plan)
    except (ValueError, KeyError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
    for plan in plans:
        print(
            f"B={plan['batch_size']} max_seq_len={plan['max_seq_len']} result={plan['result_dir']}",
            flush=True,
        )
        print(shlex.join(plan["command"]), flush=True)
    if args.dry_run:
        return 0
    args.result_root.mkdir(parents=True, exist_ok=True)
    manifest = args.result_root / "batch_runs.json"
    if manifest.exists():
        raise SystemExit(f"Refusing to overwrite {manifest}")

    def save():
        manifest.write_text(
            json.dumps({"runs": plans}, indent=2) + "\n", encoding="utf-8"
        )

    save()
    for plan in plans:
        plan["status"] = "running"
        save()
        try:
            code = subprocess.run(plan["command"], cwd=REPO_ROOT).returncode
        except KeyboardInterrupt:
            plan["status"] = "interrupted"
            save()
            return 130
        except OSError as exc:
            plan["error"] = str(exc)
            code = 127
        plan["returncode"] = code
        plan["status"] = "completed" if code == 0 else "failed"
        save()
        if code and not args.continue_on_error:
            return code if code > 0 else 1
    return int(any(p["status"] == "failed" for p in plans))


if __name__ == "__main__":
    raise SystemExit(main())

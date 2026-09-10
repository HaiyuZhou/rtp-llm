import argparse
import glob
import hashlib
import json
import logging
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rtp_llm.test.perf_test.cache_grid_runner import (
    CacheGridRunner,
    MaterializedCaseStore,
    PrefixPromptFactory,
    resume_config_fingerprint,
    validate_cache_grid_resume,
)
from rtp_llm.test.perf_test.dataset import KNOWN_DATASETS, extract_arg
from rtp_llm.test.perf_test.distribution_runner import DistributionRunner
from rtp_llm.test.perf_test.grid_runner import GridRunner
from rtp_llm.test.perf_test.hub_download import (
    needs_perf_hub_resolve,
    resolve_checkpoint_or_tokenizer_for_perf,
)
from rtp_llm.test.perf_test.perf_profile import (
    cache_grid_section,
    engine_section,
    extract_embedded_profile,
)
from rtp_llm.test.perf_test.perf_profile import fingerprint as profile_fingerprint
from rtp_llm.test.perf_test.perf_profile import (
    load_profile,
    merge_engine_args,
    resolve_int,
)
from rtp_llm.test.perf_test.sampling import prepare_distribution_config
from rtp_llm.test.perf_test.server import EngineServer
from rtp_llm.test.perf_test.test_util import create_query


def run_single(
    port: int,
    dp_size: int,
    batch_size_list: List[int],
    input_len_list: List[int],
    input_query_dict: Dict[int, str],
    is_decode: bool = True,
    dump_json_path: str = ".",
    decode_test_length: int = 20,
    tp_size: int = 1,
    generate_config: Optional[Dict[str, Any]] = None,
):
    """Backward-compatible wrapper — delegates to GridRunner."""
    return GridRunner(
        port,
        dp_size,
        batch_size_list,
        input_len_list,
        input_query_dict,
        is_decode=is_decode,
        dump_json_path=dump_json_path,
        decode_test_length=decode_test_length,
        tp_size=tp_size,
        generate_config=generate_config,
    ).run()


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description="RTP-LLM batch decode performance test runner. "
        "Unrecognized arguments are forwarded to the engine server.",
    )

    perf = parser.add_argument_group("perf test configuration")
    perf.add_argument(
        "--batch_size",
        type=str,
        default="1,8,16",
        help="Comma-separated batch sizes for grid mode",
    )
    perf.add_argument(
        "--input_len",
        type=str,
        default="1024,4096",
        help="Comma-separated input lengths for grid mode",
    )
    dataset_group = perf.add_mutually_exclusive_group()
    dataset_group.add_argument(
        "--dataset_name",
        type=str,
        default="",
        help="Known dataset name (auto-downloads via ModelScope / HF). "
        f"Choices: {list(KNOWN_DATASETS.keys())}",
    )
    dataset_group.add_argument(
        "--dataset_path",
        type=str,
        default="",
        help="Local path to dataset JSON (conversation or prompt format).",
    )
    perf.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Path to distribution.csv for runtime stratified sampling. "
        "Requires --max_seq_len and --concurrency_limit.",
    )
    perf.add_argument(
        "--test_json",
        type=str,
        default="",
        help="Path to previously saved test config JSON for replay",
    )
    perf.add_argument(
        "--partial",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="0: test all, 1: decode only, 2: prefill only",
    )
    perf.add_argument("--generate_config", type=str, default="{}")
    perf.add_argument(
        "--result_dir",
        type=str,
        default=os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "./perf_results"),
    )
    perf.add_argument("--decode_test_length", type=int, default=10)
    perf.add_argument(
        "--cache_grid_json",
        type=str,
        default="",
        help=(
            "JSON describing an explicit total-seq x prefix-cache grid. "
            "Each case inserts a prefix, then verifies aux_info.reuse_len."
        ),
    )
    perf.add_argument(
        "--cache_measure_runs",
        type=int,
        default=3,
        help="Measured requests per cache-grid case (default: 3)",
    )
    perf.add_argument(
        "--cache_request_timeout",
        type=int,
        default=int(os.environ.get("PERF_REQUEST_TIMEOUT", "7200")),
        help="Per-request timeout in seconds for cache-grid mode (default: 7200)",
    )
    perf.add_argument(
        "--cache_commit_tail_tokens",
        type=int,
        default=int(os.environ.get("CACHE_COMMIT_TAIL_TOKENS", "4096")),
        help=(
            "Extra seed tokens used to commit the requested cache prefix "
            "before measurement (default: 4096)"
        ),
    )
    perf.add_argument(
        "--warmup_runs",
        type=int,
        default=None,
        help="Override PERF_FORMAL_WARMUP_RUNS for every case.",
    )
    perf.add_argument(
        "--measure_runs",
        type=int,
        default=None,
        help="Override PERF_MEASURE_RUNS for every case.",
    )
    perf.add_argument(
        "--profile_runs",
        type=int,
        default=None,
        help="Override PERF_PROFILE_RUNS for every case.",
    )
    perf.add_argument(
        "--cache_profile_runs",
        type=int,
        default=0,
        help=(
            "Diagnostic replays per selected cache-grid case (default: disabled). "
            "Independent of --profile_runs."
        ),
    )
    perf.add_argument(
        "--cache_profile_case_ids",
        type=int,
        nargs="+",
        default=[],
        help="Explicit case IDs from the cache grid/report to profile.",
    )
    perf.add_argument(
        "--cache_profile_only",
        action="store_true",
        help=(
            "Skip formal measurements; write a new isolated replay directory "
            "below result_dir."
        ),
    )
    perf.add_argument(
        "--cache_profile_trace_timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for complete trace JSON from every TP rank.",
    )
    perf.add_argument(
        "--engine_arg",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Repeatable engine argument shorthand, e.g. tp_size=8.",
    )
    perf.add_argument(
        "--engine_env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Repeatable engine environment default; --test_env wins.",
    )
    perf.add_argument(
        "--expected_cache_block_size",
        type=int,
        default=0,
        help=(
            "Physical prefix-cache reuse granularity in tokens: "
            "--seq_size_per_block (DSV4 defaults to 256 when unset), "
            "multiplied by CP size when PREFILL_CP_KV_CACHE_SHARDED=1. "
            "0 = read from the grid JSON generator metadata, or skip. "
            "Used to drop cases collapsing onto the same block bucket and "
            "to probe reuse_len before measuring."
        ),
    )
    perf.add_argument(
        "--materialize_cache_cases",
        type=str,
        default="",
        help=(
            "Cache-grid mode: build every case's prompts once with the "
            "tokenizer, store them compactly in this directory, and exit "
            "without starting the engine.  Later runs with "
            "--cache_case_files reuse the store and skip per-case prompt "
            "construction entirely."
        ),
    )
    perf.add_argument(
        "--cache_request_transport",
        choices=("http_prompt", "dashsc_input_ids"),
        default="dashsc_input_ids",
        help=(
            "Cache-grid request transport. dashsc_input_ids sends the already "
            "verified token IDs as binary INT32 and skips server tokenization "
            "(default: dashsc_input_ids)."
        ),
    )
    perf.add_argument(
        "--cache_grpc_port",
        type=int,
        default=0,
        help="Dash-SC gRPC port for input_ids mode; 0 uses HTTP port + 8.",
    )
    perf.add_argument(
        "--cache_case_files",
        type=str,
        default="",
        help=(
            "Cache-grid mode: read prompts from a store directory created "
            "by --materialize_cache_cases instead of constructing them with "
            "the tokenizer at run time.  Requires the same --cache_grid_json."
        ),
    )
    perf.add_argument(
        "--profile",
        type=str,
        default="",
        help=(
            "JSON profile for parameter defaults and engine arg injection. "
            "Priority: CLI explicit > profile > legacy default."
        ),
    )
    perf.add_argument(
        "--cache_checkpoint_every",
        type=int,
        default=100,
        help=(
            "Compact the append-only per-case journal into the full result "
            "JSON after this many cases (default: 100)."
        ),
    )
    perf.add_argument(
        "--require_cache_resume",
        action="store_true",
        help=(
            "Require cache_grid_results.json in --result_dir. Use this on a "
            "restart to prevent an accidental fresh run from a mistyped path."
        ),
    )
    perf.add_argument(
        "--allow_resume_mismatch",
        action="store_true",
        help=(
            "When resuming from existing results, warn instead of aborting "
            "on grid/profile, model/engine config, measure runs, transport, "
            "commit-tail, or block-size mismatches."
        ),
    )

    engine = parser.add_argument_group(
        "engine args consumed by perf test (also forwarded to server)"
    )
    engine.add_argument("--dp_size", type=int, default=1)
    engine.add_argument("--max_seq_len", type=int, default=8192)
    engine.add_argument("--concurrency_limit", type=int, default=64)

    args, remaining = parser.parse_known_args(argv)
    if args.cache_profile_runs < 0 or args.cache_profile_trace_timeout <= 0:
        parser.error(
            "cache profile runs must be non-negative and trace timeout positive"
        )
    if bool(args.cache_profile_runs) != bool(args.cache_profile_case_ids):
        parser.error(
            "--cache_profile_runs and --cache_profile_case_ids must be supplied together"
        )
    if args.cache_profile_only and args.require_cache_resume:
        parser.error(
            "--cache_profile_only creates a fresh replay directory; "
            "omit --require_cache_resume"
        )
    if args.cache_profile_only and not args.cache_profile_runs:
        parser.error("--cache_profile_only requires --cache_profile_runs and case IDs")
    if args.cache_profile_runs and (not args.cache_grid_json or args.partial != 2):
        parser.error("cache profiling requires --cache_grid_json and --partial=2")

    profile = None
    profile_sha256 = None
    if args.profile:
        profile = load_profile(args.profile)
        profile_sha256 = profile_fingerprint(profile)
        cache_grid = cache_grid_section(profile)
        engine = engine_section(profile)
        args.cache_measure_runs = resolve_int(
            profile,
            "cache_grid",
            "measure_runs",
            args.cache_measure_runs if args.cache_measure_runs != 3 else None,
            3,
        )
        args.expected_cache_block_size = resolve_int(
            profile,
            "cache_grid",
            "expected_block_size",
            (
                args.expected_cache_block_size
                if args.expected_cache_block_size != 0
                else None
            ),
            0,
        )
        if "dp_size" in engine:
            args.dp_size = resolve_int(
                profile,
                "engine",
                "dp_size",
                args.dp_size if args.dp_size != 1 else None,
                1,
            )
        if "max_seq_len" in engine:
            args.max_seq_len = resolve_int(
                profile,
                "engine",
                "max_seq_len",
                args.max_seq_len if args.max_seq_len != 8192 else None,
                8192,
            )
        if "concurrency_limit" in engine:
            args.concurrency_limit = resolve_int(
                profile,
                "engine",
                "concurrency_limit",
                args.concurrency_limit if args.concurrency_limit != 64 else None,
                64,
            )
        remaining = merge_engine_args(profile, remaining)

    args._profile = profile
    args._profile_sha256 = profile_sha256
    return args, remaining


def _parse_name_value(value: str, option: str) -> Tuple[str, str]:
    """Parse NAME=VALUE options used by the generic Bazel entrypoint."""
    if "=" not in value:
        raise ValueError(f"{option} expects NAME=VALUE, got {value!r}")
    name, parsed = value.split("=", 1)
    name = name.strip().lstrip("-")
    if not name or any(ch.isspace() for ch in name):
        raise ValueError(f"{option} has invalid name in {value!r}")
    return name, parsed


def _engine_arg_argv(engine_args: List[str]) -> List[str]:
    """Convert repeatable ``--engine_arg NAME=VALUE`` options to engine argv."""
    result: List[str] = []
    for item in engine_args:
        name, value = _parse_name_value(item, "--engine_arg")
        result.extend([f"--{name}", value])
    return result


def _apply_engine_env(engine_env: List[str]) -> List[str]:
    """Apply engine env defaults without overriding Bazel ``--test_env`` values."""
    names: List[str] = []
    for item in engine_env:
        name, value = _parse_name_value(item, "--engine_env")
        if name not in os.environ:
            os.environ[name] = value
        names.append(name)
    return sorted(set(names))


def _apply_run_overrides(args: argparse.Namespace) -> None:
    """Map explicit perf run controls to the legacy environment contract."""
    for option, env_name in (
        ("warmup_runs", "PERF_FORMAL_WARMUP_RUNS"),
        ("measure_runs", "PERF_MEASURE_RUNS"),
        ("profile_runs", "PERF_PROFILE_RUNS"),
    ):
        value = getattr(args, option)
        if value is None:
            continue
        if value < 0:
            raise ValueError(f"--{option} must be >= 0, got {value}")
        os.environ[env_name] = str(value)


def _is_sensitive_name(name: str) -> bool:
    normalized = name.strip().lstrip("-").lower()
    return (
        any(token in normalized for token in ("password", "secret", "access_key"))
        or normalized == "token"
        or normalized.endswith("_token")
    )


def _redact_argv(argv: List[str]) -> List[str]:
    """Redact likely credentials before persisting invocation metadata."""
    redacted: List[str] = []
    redact_next = False
    for item in argv:
        if redact_next:
            redacted.append("***")
            redact_next = False
            continue
        key = item.split("=", 1)[0].lstrip("-").lower()
        embedded_key = ""
        if key in ("engine_arg", "engine_env") and "=" in item:
            embedded_key = item.split("=", 1)[1].split("=", 1)[0].lower()
        sensitive = _is_sensitive_name(key) or (
            bool(embedded_key) and _is_sensitive_name(embedded_key)
        )
        if sensitive:
            if "=" in item:
                redacted.append(item.split("=", 1)[0] + "=***")
            else:
                redacted.append(item)
                redact_next = True
        else:
            redacted.append(item)
    return redacted


_REPRODUCTION_ENV_NAMES = {
    "CUDA_VISIBLE_DEVICES",
    "LOCAL_WORLD_SIZE",
    "START_PORT",
    "TOKENIZERS_PARALLELISM",
    "WORLD_SIZE",
}
_REPRODUCTION_ENV_PREFIXES = (
    "CACHE_",
    "DG_JIT_",
    "DSV4_",
    "PERF_",
    "PREFILL_",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TILELANG_",
    "TRITON_",
)


def _capture_reproduction_env(
    engine_env_names: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Capture performance-relevant environment values with credential redaction."""
    requested = set(engine_env_names or [])
    requested.update(
        name
        for name in os.environ
        if name in _REPRODUCTION_ENV_NAMES
        or name.startswith(_REPRODUCTION_ENV_PREFIXES)
    )
    captured = {}
    for name in sorted(requested):
        if name not in os.environ:
            continue
        lowered = name.lower()
        sensitive = _is_sensitive_name(lowered)
        captured[name] = "***" if sensitive else os.environ[name]
    return captured


def _build_cache_resume_config(
    args: argparse.Namespace,
    remaining_args: List[str],
    engine_env_names: Optional[List[str]],
    expected_cache_block_size: int,
) -> Dict[str, Any]:
    """Build the stable model/workload config guarded across resumed attempts."""
    return {
        "schema_version": 1,
        "model": {
            "model_type": extract_arg(remaining_args, "model_type"),
            "checkpoint_path": extract_arg(remaining_args, "checkpoint_path"),
            "tokenizer_path": extract_arg(remaining_args, "tokenizer_path"),
        },
        "engine": {
            "args": _redact_argv(remaining_args),
            "environment": _capture_reproduction_env(engine_env_names),
            "dp_size": args.dp_size,
            "max_seq_len": args.max_seq_len,
            "concurrency_limit": args.concurrency_limit,
        },
        "workload": {
            "partial": args.partial,
            "decode_test_length": args.decode_test_length,
            "cache_measure_runs": args.cache_measure_runs,
            "cache_commit_tail_tokens": args.cache_commit_tail_tokens,
            "expected_cache_block_size": expected_cache_block_size,
            "cache_request_transport": args.cache_request_transport,
        },
    }


def _replace_cli_value(argv: List[str], key: str, new_value: str) -> None:
    flag = f"--{key}"
    prefix = f"--{key}="
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            argv[i + 1] = new_value
            return
        if arg.startswith(prefix):
            argv[i] = prefix + new_value
            return
    raise ValueError(
        f"perf_test: missing {flag} in argv, cannot replace with local path"
    )


def resolve_perf_engine_paths(remaining: List[str]) -> List[str]:
    """在转发给引擎前，将 Hub 链接 / repo id 等解析为本地路径并写回 argv（同一远程引用只下载一次）。"""
    out = list(remaining)
    resolved_cache: Dict[str, str] = {}
    for k in ("checkpoint_path", "tokenizer_path"):
        val = extract_arg(out, k)
        if not val or not needs_perf_hub_resolve(val):
            continue
        if val not in resolved_cache:
            local = resolve_checkpoint_or_tokenizer_for_perf(val)
            logging.info(f"perf_test: resolved --{k} -> {local}")
            resolved_cache[val] = local
        _replace_cli_value(out, k, resolved_cache[val])
    return out


def _load_cache_grid_cases(path: str) -> List[Dict[str, int]]:
    """Load and validate an explicit total-sequence × cache-length grid.

    The cache runner measures batch=1 only.  Cache length is a request-level
    workload dimension, not an engine CLI argument, so the case file is kept
    separate from the forwarded engine args.
    """
    with open(path, encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("cache grid must be a JSON object")
    explicit_cases = "cases" in config
    if explicit_cases:
        raw_cases = config["cases"]
    else:
        seq_lens = config.get("seq_lens")
        if seq_lens is None:
            generation = config.get("seq_generation", {})
            if generation.get("kind") != "linear_with_dense_prefix":
                raise ValueError(
                    "cache grid requires cases, seq_lens, or "
                    "seq_generation.kind=linear_with_dense_prefix"
                )
            count = int(generation.get("count", 489))
            max_seq_len = int(generation.get("max_seq_len", 1048575))
            seq_block = int(config.get("seq_block_size", 256))
            if count < 2 or max_seq_len <= seq_block:
                raise ValueError("invalid seq_generation bounds")
            values = set(range(seq_block, min(16384, max_seq_len), seq_block))
            target_nonmax = count - 1
            i = 0
            while len(values) < target_nonmax:
                raw = seq_block + round(
                    i * (max_seq_len - 2 * seq_block) / max(1, target_nonmax - 1)
                )
                aligned = max(
                    seq_block,
                    min(
                        max_seq_len - seq_block,
                        round(raw / seq_block) * seq_block,
                    ),
                )
                values.add(aligned)
                i += 1
                if i > target_nonmax * 20:
                    raise ValueError("unable to generate unique seq lengths")
            seq_lens = sorted(values)[:target_nonmax] + [max_seq_len]
            if len(seq_lens) != count or len(set(seq_lens)) != count:
                raise ValueError("generated sequence lengths are not unique")
        ratios = [
            float(x)
            for x in config.get(
                "cache_ratios",
                [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.95],
            )
        ]
        block = int(config.get("cache_block_size", 4096))
        if block <= 0 or any(ratio < 0.0 or ratio >= 1.0 for ratio in ratios):
            raise ValueError(
                "cache_ratios must be in [0, 1) and block must be positive"
            )
        raw_cases = []
        case_id = 0
        for seq_len in seq_lens:
            seq_len = int(seq_len)
            max_cache_len = max(0, ((seq_len - block) // block) * block)
            for ratio in ratios:
                cache_len = int((max(0, seq_len - 1) * ratio) // block) * block
                raw_cases.append(
                    {
                        "case_id": case_id,
                        "batch_size": 1,
                        "input_len": seq_len,
                        "cache_len": min(cache_len, max_cache_len),
                    }
                )
                case_id += 1
            if max_cache_len > 0:
                raw_cases.append(
                    {
                        "case_id": case_id,
                        "batch_size": 1,
                        "input_len": seq_len,
                        "cache_len": max_cache_len,
                    }
                )
                case_id += 1

    if not isinstance(raw_cases, list):
        raise ValueError("cache grid cases must be a list")
    cases: List[Dict[str, int]] = []
    seen = set()
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, dict):
            raise ValueError(f"cache grid case {index} must be an object")
        case = {
            "case_id": int(raw.get("case_id", index)),
            "batch_size": int(raw.get("batch_size", 1)),
            "input_len": int(raw["input_len"]),
            "cache_len": int(raw.get("cache_len", 0)),
        }
        if case["batch_size"] != 1:
            raise ValueError("cache grid currently requires batch_size=1")
        if case["input_len"] <= 0 or not 0 <= case["cache_len"] < case["input_len"]:
            raise ValueError(f"invalid cache grid case: {case}")
        key = (case["batch_size"], case["input_len"], case["cache_len"])
        if key in seen:
            if not explicit_cases:
                continue
            raise ValueError(f"duplicate cache grid case: {case}")
        seen.add(key)
        cases.append(case)
    if not cases:
        raise ValueError(f"cache grid {path} contains no cases")
    return cases


def _resolve_cache_block_size(grid_payload: Any, cli_value: int) -> int:
    """Resolve the physical reuse granularity for dedup and probing.

    Prefer the explicit CLI value; otherwise fall back to the alignment the
    grid was generated with (generate_cache_grid.py records it as
    generator.cache_alignment or generator.cache_sampling.alignment).  0
    means unknown — skip dedup and probing.
    """
    if cli_value > 0:
        return cli_value
    generator = (
        grid_payload.get("generator") if isinstance(grid_payload, dict) else None
    )
    if not isinstance(generator, dict):
        return 0
    for source in (
        generator.get("cache_alignment"),
        (
            generator.get("cache_sampling", {}).get("alignment")
            if isinstance(generator.get("cache_sampling"), dict)
            else None
        ),
    ):
        try:
            value = int(source or 0)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return 0


def _dedupe_cache_grid_cases(
    cases: List[Dict[str, int]], block_size: int
) -> List[Dict[str, int]]:
    """Collapse cases whose requested cache lengths share one physical bucket.

    The engine reuses prefix KV only in whole block_size chunks, so requests
    whose cache_len floors to the same block count measure the identical
    geometry.  Keep one representative per bucket: prefer the aligned request
    (its observed reuse then equals what it asked for), otherwise the largest
    request in the bucket, which stays closest to the next boundary if the
    block-size assumption is slightly off.
    """
    buckets: Dict[tuple, Dict[str, int]] = {}
    order: List[tuple] = []
    for case in cases:
        key = (case["batch_size"], case["input_len"], case["cache_len"] // block_size)
        kept = buckets.get(key)
        if kept is None:
            buckets[key] = case
            order.append(key)
        elif kept["cache_len"] % block_size and not case["cache_len"] % block_size:
            buckets[key] = case
        elif (
            kept["cache_len"] % block_size
            and case["cache_len"] % block_size
            and case["cache_len"] > kept["cache_len"]
        ):
            buckets[key] = case
    return [buckets[key] for key in order]


def _load_materialized_case_store(
    root: str,
    cases: List[Dict[str, int]],
    measure_runs: int,
    grid_sha256: str,
) -> MaterializedCaseStore:
    """Load a precomputed prompt store and prove it matches this grid."""
    store = MaterializedCaseStore(root)
    stored = store.load_cases()
    if store.run_count != measure_runs:
        raise ValueError(
            f"case store {root} was materialized with run_count="
            f"{store.run_count}, but --cache_measure_runs={measure_runs}"
        )
    if store.grid_sha256 and store.grid_sha256 != grid_sha256:
        raise ValueError(
            f"case store {root} was materialized from a different grid JSON "
            f"(sha256 {store.grid_sha256[:12]} != {grid_sha256[:12]}); "
            "rerun --materialize_cache_cases"
        )

    def geometry(case: Dict[str, int]) -> tuple:
        return (
            case["case_id"],
            case["batch_size"],
            case["input_len"],
            case["cache_len"],
        )

    if [geometry(c) for c in stored] != [geometry(c) for c in cases]:
        raise ValueError(
            f"case store {root} holds {len(stored)} cases but this run plans "
            f"{len(cases)}; the grid JSON or block-size dedup changed since "
            "materialization. Rerun --materialize_cache_cases."
        )
    return store


def _collect_timeline_files(result_dir: str) -> None:
    """Wait for async profiler saves and collect timeline JSON files into a timelines/ subdirectory."""
    # C++ engine's ProfilerSaveWorker writes timeline JSONs asynchronously in a
    # background thread. Sleep to allow pending writes to flush before we move files.
    time.sleep(3)
    timeline_dir = os.path.join(result_dir, "timelines")
    pattern = os.path.join(result_dir, "*.json")
    timeline_files = [
        f
        for f in glob.glob(pattern)
        if os.path.basename(f).startswith(("profiler_ts", "profiler_"))
        or "_wr" in os.path.basename(f)
    ]
    if timeline_files:
        os.makedirs(timeline_dir, exist_ok=True)
        for f in timeline_files:
            dst = os.path.join(timeline_dir, os.path.basename(f))
            shutil.move(f, dst)
            logging.info(f"Collected timeline: {dst}")
    else:
        logging.info("No timeline files found in %s", result_dir)


def _write_test_info(
    args: argparse.Namespace,
    remaining_args: List[str],
    engine_env_names: Optional[List[str]] = None,
    status: str = "completed",
    expected_cache_block_size: int = 0,
    resume_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist a reproducible, credential-safe test configuration."""
    model_type = extract_arg(remaining_args, "model_type") or os.environ.get(
        "MODEL_TYPE"
    )
    checkpoint_path = extract_arg(remaining_args, "checkpoint_path") or os.environ.get(
        "CHECKPOINT_PATH"
    )
    tokenizer_path = extract_arg(remaining_args, "tokenizer_path") or os.environ.get(
        "TOKENIZER_PATH"
    )
    profile = getattr(args, "_profile", None)
    profile_sha256 = getattr(args, "_profile_sha256", None)
    path = os.path.join(args.result_dir, "test_info.json")
    previous: Dict[str, Any] = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as stream:
                previous = json.load(stream)
        except (OSError, ValueError):
            logging.warning("Ignoring unreadable previous test info at %s", path)
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    attempt_count = int(previous.get("attempt_count", 0))
    if status == "running":
        attempt_count += 1
    info = {
        "schema_version": 3,
        "status": status,
        "started_at": previous.get("started_at", now),
        "updated_at": now,
        "last_attempt_started_at": (
            now if status == "running" else previous.get("last_attempt_started_at", now)
        ),
        "attempt_count": attempt_count,
        "model_type": model_type,
        "checkpoint_path": checkpoint_path,
        "tokenizer_path": tokenizer_path,
        "tp_size": extract_arg(remaining_args, "tp_size", "1"),
        "dp_size": args.dp_size,
        "max_seq_len": args.max_seq_len,
        "concurrency_limit": args.concurrency_limit,
        "decode_test_length": args.decode_test_length,
        "seq_size_per_block": extract_arg(remaining_args, "seq_size_per_block", None),
        "cache_grid_json": args.cache_grid_json or None,
        "cache_measure_runs": (
            args.cache_measure_runs if args.cache_grid_json else None
        ),
        "cache_request_timeout": (
            args.cache_request_timeout if args.cache_grid_json else None
        ),
        "cache_commit_tail_tokens": (
            args.cache_commit_tail_tokens if args.cache_grid_json else None
        ),
        "partial": args.partial,
        "warmup_runs": int(os.environ.get("PERF_FORMAL_WARMUP_RUNS", "1")),
        "measure_runs": int(os.environ.get("PERF_MEASURE_RUNS", "1")),
        "profile_runs": int(os.environ.get("PERF_PROFILE_RUNS", "1")),
        "expected_cache_block_size": (
            expected_cache_block_size if args.cache_grid_json else None
        ),
        "cache_case_store": args.cache_case_files or None,
        "cache_profile_runs": args.cache_profile_runs,
        "cache_profile_case_ids": args.cache_profile_case_ids,
        "cache_profile_only": args.cache_profile_only,
        "cache_profile_trace_timeout": args.cache_profile_trace_timeout,
        "cache_request_transport": (
            args.cache_request_transport if args.cache_grid_json else None
        ),
        "cache_grpc_port": (
            args.cache_grpc_port or None if args.cache_grid_json else None
        ),
        "dataset_name": args.dataset_name or None,
        "dataset_path": args.dataset_path or args.dataset or None,
        "engine_args": _redact_argv(remaining_args),
        "engine_env_names": sorted(engine_env_names or []),
        "engine_environment": _capture_reproduction_env(engine_env_names),
        "argv": _redact_argv(sys.argv),
        "resume_config": resume_config,
        "resume_config_sha256": resume_config_fingerprint(resume_config),
        "profile": profile,
        "profile_sha256": profile_sha256,
    }
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as stream:
        json.dump(info, stream, indent=2)
    os.replace(tmp_path, path)
    logging.info("Wrote test info to %s", path)


def _effective_grid_max_seq_len(
    args: argparse.Namespace, input_len_list: List[int]
) -> int:
    needed_seq_len = max(input_len_list) + args.decode_test_length
    return max(needed_seq_len, args.max_seq_len)


def _ensure_xgrammar_lib_path() -> None:
    import sys

    so_name = "libxgrammar_bindings.so"
    for p in sys.path:
        xgrammar_dir = os.path.join(p, "xgrammar")
        if os.path.isfile(os.path.join(xgrammar_dir, so_name)):
            current = os.environ.get("LD_LIBRARY_PATH", "")
            if xgrammar_dir not in current.split(":"):
                os.environ["LD_LIBRARY_PATH"] = (
                    f"{xgrammar_dir}:{current}" if current else xgrammar_dir
                )
                logging.info(f"Added {xgrammar_dir} to LD_LIBRARY_PATH")
            return


def main() -> str:
    from rtp_llm.config.log_config import setup_logging

    setup_logging()
    _ensure_xgrammar_lib_path()

    args, remaining = parse_args()
    engine_env_names = _apply_engine_env(args.engine_env)
    _apply_run_overrides(args)
    # Model/parallelism-specific flags are intentionally not hard-coded in
    # this runner.  The generic target can forward any engine flag through
    # repeatable --engine_arg=NAME=VALUE options; raw unknown args remain
    # supported for backwards compatibility.
    remaining.extend(_engine_arg_argv(args.engine_arg))
    remaining = resolve_perf_engine_paths(remaining)
    generate_config = json.loads(args.generate_config)
    if args.cache_profile_runs and args.dp_size != 1:
        raise ValueError(
            "cache-grid profiling currently requires DP=1; all TP ranks are captured"
        )
    if args.cache_profile_only:
        # Preserve the original report, manifest and resume checkpoint byte-for-byte.
        args.result_dir = str(
            Path(args.result_dir) / "cache_profile_replays" / uuid.uuid4().hex
        )
    os.makedirs(args.result_dir, exist_ok=True)
    # Cache-grid mode writes its manifest after resolving the grid and resume
    # fingerprint, but still before tokenizer/model initialization.
    if not args.cache_grid_json:
        _write_test_info(args, remaining, engine_env_names, status="running")
    EngineServer.propagate_engine_env(remaining)

    logging.info(f"Result directory: {args.result_dir}")
    logging.info(f"Engine args forwarded to server: {remaining}")

    if args.cache_grid_json:
        if args.partial != 2:
            raise ValueError("--cache_grid_json is prefill-only; use --partial=2")
        if args.cache_measure_runs <= 0:
            raise ValueError("--cache_measure_runs must be positive")
        if args.cache_request_timeout <= 0:
            raise ValueError("--cache_request_timeout must be positive")
        if args.cache_commit_tail_tokens <= 0:
            raise ValueError("--cache_commit_tail_tokens must be positive")
        if args.cache_checkpoint_every <= 0:
            raise ValueError("--cache_checkpoint_every must be positive")
        if args.materialize_cache_cases and args.cache_case_files:
            raise ValueError(
                "--materialize_cache_cases and --cache_case_files are mutually "
                "exclusive"
            )

        cases = _load_cache_grid_cases(args.cache_grid_json)
        with open(args.cache_grid_json, "rb") as stream:
            grid_bytes = stream.read()
        grid_payload = json.loads(grid_bytes)
        grid_metadata = {
            key: grid_payload.get(key)
            for key in ("schema_version", "kind", "generator", "summary")
            if key in grid_payload
        }
        grid_sha256 = hashlib.sha256(grid_bytes).hexdigest()
        expected_block_size = _resolve_cache_block_size(
            grid_payload, args.expected_cache_block_size
        )
        if expected_block_size > 0:
            deduped = _dedupe_cache_grid_cases(cases, expected_block_size)
            if len(deduped) < len(cases):
                logging.warning(
                    "cache grid: dropped %d of %d cases that collapse onto the same "
                    "%d-token physical block bucket",
                    len(cases) - len(deduped),
                    len(cases),
                    expected_block_size,
                )
                cases = deduped
        logging.info(
            "cache grid plan: cases=%d sha256=%s expected_block_size=%d metadata=%s",
            len(cases),
            grid_sha256,
            expected_block_size,
            grid_metadata,
        )
        unknown_profile_ids = set(args.cache_profile_case_ids) - {
            int(c["case_id"]) for c in cases
        }
        if unknown_profile_ids:
            raise ValueError(
                f"unknown/deduplicated cache profile case IDs: {sorted(unknown_profile_ids)}"
            )
        if args.cache_profile_runs and args.materialize_cache_cases:
            raise ValueError(
                "cache profiling cannot be combined with --materialize_cache_cases"
            )
        resume_config = _build_cache_resume_config(
            args, remaining, engine_env_names, expected_block_size
        )
        checkpoint = validate_cache_grid_resume(
            args.result_dir,
            grid_sha256=grid_sha256,
            profile_sha256=getattr(args, "_profile_sha256", None),
            measure_runs=args.cache_measure_runs,
            cache_commit_tail_tokens=args.cache_commit_tail_tokens,
            expected_block_size=expected_block_size,
            request_transport=args.cache_request_transport,
            run_config=resume_config,
            allow_resume_mismatch=args.allow_resume_mismatch,
            require_resume=args.require_cache_resume,
        )
        if checkpoint is not None:
            logging.info(
                "cache grid: preflight resume accepted progress=%s",
                checkpoint.get("progress"),
            )
        _write_test_info(
            args,
            remaining,
            engine_env_names,
            status="running",
            expected_cache_block_size=expected_block_size,
            resume_config=resume_config,
        )
        checkpoint_ok_keys = (
            {
                str(row.get("case_key"))
                for row in checkpoint.get("metrics", [])
                if isinstance(row, dict) and row.get("status") == "ok"
            }
            if checkpoint is not None
            else set()
        )
        planned_case_keys = {CacheGridRunner.case_key(case) for case in cases}
        if (
            checkpoint is not None
            and checkpoint_ok_keys == planned_case_keys
            and not args.cache_profile_runs
        ):
            logging.info(
                "cache grid: checkpoint already contains all %d successful cases; "
                "skipping tokenizer and model startup",
                len(cases),
            )
            _write_test_info(
                args,
                remaining,
                engine_env_names,
                status="completed",
                expected_cache_block_size=expected_block_size,
                resume_config=resume_config,
            )
            return args.result_dir
        for case in cases:
            cache_len = int(case["cache_len"])
            input_len = int(case["input_len"])
            if cache_len and cache_len % args.cache_commit_tail_tokens:
                raise ValueError(
                    "cache-grid cache_len must align to "
                    f"--cache_commit_tail_tokens={args.cache_commit_tail_tokens}: "
                    f"{case}"
                )
            if cache_len and cache_len + args.cache_commit_tail_tokens > input_len:
                raise ValueError(
                    "cache-grid cache_len must leave one commit tail before "
                    f"server startup: {case}"
                )
        max_input_len = max(int(case["input_len"]) for case in cases)
        tokenizer_path = (
            extract_arg(remaining, "tokenizer_path")
            or extract_arg(remaining, "checkpoint_path")
            or os.environ.get("TOKENIZER_PATH", "")
        )
        if not tokenizer_path:
            raise ValueError(
                "cache-grid mode requires --tokenizer_path or --checkpoint_path"
            )

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )

        if args.materialize_cache_cases:
            store = MaterializedCaseStore(args.materialize_cache_cases)
            stats = store.materialize(
                cases,
                PrefixPromptFactory(tokenizer),
                args.cache_measure_runs,
                grid_metadata=grid_metadata,
                grid_sha256=grid_sha256,
                profile_sha256=getattr(args, "_profile_sha256", "") or "",
            )
            logging.info(
                "materialized %d cache-grid cases (%d cached) to %s: %s",
                stats["cases"],
                stats["cached_cases"],
                args.materialize_cache_cases,
                stats,
            )
            return args.result_dir

        case_store = None
        if args.cache_case_files:
            case_store = _load_materialized_case_store(
                args.cache_case_files,
                cases,
                args.cache_measure_runs,
                grid_sha256,
            )
            logging.info(
                "cache grid: using materialized case store %s", args.cache_case_files
            )

        server = EngineServer(args, remaining)
        server.start(
            max_seq_len=max(max_input_len + args.decode_test_length, args.max_seq_len),
            # CacheGridRunner keeps model forwards serial. Preserve the
            # requested service admission limit instead of rewriting it to
            # the workload's fixed batch size of one.
            max_concurrency=args.concurrency_limit,
        )
        try:
            CacheGridRunner(
                server.port,
                tokenizer,
                cases,
                args.result_dir,
                request_timeout=args.cache_request_timeout,
                measure_runs=args.cache_measure_runs,
                checkpoint_every=args.cache_checkpoint_every,
                cache_commit_tail_tokens=args.cache_commit_tail_tokens,
                grid_metadata=grid_metadata,
                grid_sha256=grid_sha256,
                expected_block_size=expected_block_size,
                case_store=case_store,
                profile=getattr(args, "_profile", None),
                profile_sha256=getattr(args, "_profile_sha256", None),
                allow_resume_mismatch=args.allow_resume_mismatch,
                require_resume=args.require_cache_resume,
                request_transport=args.cache_request_transport,
                grpc_port=args.cache_grpc_port or None,
                run_config=resume_config,
                profile_runs=args.cache_profile_runs,
                profile_case_ids=args.cache_profile_case_ids,
                profile_only=args.cache_profile_only,
                profile_tp_size=int(
                    extract_arg(remaining, "tp_size") or os.environ.get("TP_SIZE", "1")
                ),
                profile_trace_timeout=args.cache_profile_trace_timeout,
            ).run()
            _collect_timeline_files(args.result_dir)
        finally:
            server.stop()
        _write_test_info(
            args,
            remaining,
            engine_env_names,
            status="completed",
            expected_cache_block_size=expected_block_size,
            resume_config=resume_config,
        )
        return args.result_dir

    distribution_mode = (
        args.dataset_name or args.dataset_path or args.dataset or args.test_json
    )
    if distribution_mode:
        tokenizer_path = (
            extract_arg(remaining, "tokenizer_path")
            or extract_arg(remaining, "checkpoint_path")
            or os.environ.get("TOKENIZER_PATH", "")
        )
        test_config = prepare_distribution_config(
            tokenizer_path=tokenizer_path,
            max_seq_len=args.max_seq_len,
            max_concurrency=args.concurrency_limit * args.dp_size,
            result_dir=args.result_dir,
            dataset_name=args.dataset_name,
            dataset_path=args.dataset_path,
            dataset_csv=args.dataset,
            test_json=args.test_json,
        )

        batch_seq_len_map = test_config["batch_seq_len_map"]
        all_seq_lens = sorted(
            set(sl for sls in batch_seq_len_map.values() for sl in sls)
        )

        needed_seq_len = max(all_seq_lens) + args.decode_test_length
        effective_max_seq_len = max(needed_seq_len, args.max_seq_len)

        server = EngineServer(args, remaining)
        server.start(
            max_seq_len=effective_max_seq_len,
            max_concurrency=max(int(k) for k in batch_seq_len_map),
        )

        input_query_dict = create_query(input_len_list=all_seq_lens)
        DistributionRunner(
            server.port,
            args.dp_size,
            test_config,
            input_query_dict,
            dump_json_path=args.result_dir,
            decode_test_length=args.decode_test_length,
            generate_config=generate_config,
        ).run()
        _collect_timeline_files(args.result_dir)
        server.stop()
    else:
        batch_size_list = [int(x) for x in args.batch_size.split(",")]
        input_len_list = [int(x) for x in args.input_len.split(",")]
        needed_seq_len = max(input_len_list) + args.decode_test_length
        effective_max_seq_len = max(needed_seq_len, args.max_seq_len)

        server = EngineServer(args, remaining)
        server.start(
            max_seq_len=_effective_grid_max_seq_len(args, input_len_list),
            max_concurrency=max(batch_size_list),
        )

        input_query_dict = create_query(input_len_list=input_len_list)

        if args.partial in (0, 1):
            GridRunner(
                server.port,
                args.dp_size,
                batch_size_list,
                input_len_list,
                input_query_dict,
                is_decode=True,
                dump_json_path=args.result_dir,
                decode_test_length=args.decode_test_length,
                generate_config=generate_config,
            ).run()
        if args.partial in (0, 2):
            GridRunner(
                server.port,
                args.dp_size,
                batch_size_list,
                input_len_list,
                input_query_dict,
                is_decode=False,
                dump_json_path=args.result_dir,
                decode_test_length=args.decode_test_length,
                generate_config=generate_config,
            ).run()
        _collect_timeline_files(args.result_dir)
        server.stop()

    _write_test_info(args, remaining, engine_env_names, status="completed")
    return args.result_dir


if __name__ == "__main__":
    main()

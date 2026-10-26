"""Explicit prefix-cache performance grid runner.

The ordinary ``GridRunner`` varies only batch size and total input length.  This
runner adds a cache dimension without adding an engine/server argument: each
case first inserts a unique prefix into the normal prefix cache, then sends
three unique continuations sharing exactly that prefix.  The measured
``aux_info.reuse_len`` is recorded, so a requested cache length is never
silently treated as a hit.

When ``expected_block_size`` is set (the engine's ``--seq_size_per_block``,
multiplied by CP size under ``PREFILL_CP_KV_CACHE_SHARDED=1``), a probe runs
before any case: it seeds exactly one block of tokens and verifies a 2-block
request reports ``reuse_len == block_size``.  A mismatch would silently
invalidate every cache-hitting case, so the runner aborts immediately instead
of burning GPU-hours on a misaligned grid.

Prompt construction happens at the token-id layer (``build_case_prompts``):
every prompt is ``marker + " hello" * k``, so the exact-length ids are marker
ids plus filler repeats, and the text is the same string concatenation.  The
structure is proven once per shape by probing the tokenizer and by fully
re-encoding the first built case; tokenizers that do not tokenize the filler
pattern structurally fall back to exact text construction.
Precomputed prompts can also be materialized to disk once and reloaded
without a tokenizer (``MaterializedCaseStore``).

Grouped cases support explicit request counts and independent or group-shared
prefixes. Seed and measurement phases configure the dedicated DP=1 test scheduler
to their respective batch sizes before concurrent submission. Each member's
geometry is validated; see ``cache_grid/docs/reference.md``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import random
import statistics
import struct
import subprocess
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter

from rtp_llm.test.perf_test.cache_grid.runner.continuous_pressure import (
    DispatchClosed,
    TieredRollingPlanner,
    freeze_seed_families,
)
from rtp_llm.test.perf_test.cache_grid.runner.result_schema import (
    SERVER_LATENCY_CONTRACT,
    current_metrics,
    measurement_status,
    run_prefill_rt,
)
from rtp_llm.test.perf_test.cache_grid.runner.scheduler import configure_scheduler
from rtp_llm.test.perf_test.perf_utils import _sanitize_nested_provenance

_STARTUP_PRESSURE_RETRY_BUDGET_S = 10.0
_STARTUP_PRESSURE_RETRY_DELAY_S = 0.25
_TRANSIENT_PRESSURE_RANK_ERRORS = (
    "connect failed",
    "connection refused",
    "failed to connect",
    "temporarily unavailable",
)


def _is_transient_pressure_rank_failure(response, error: ValueError) -> bool:
    errors = [rank.error.strip().lower() for rank in response.ranks if rank.error]
    raised = str(error).strip().lower()
    rank_failures = {
        f"rank {rank.world_rank}: {rank.error}".strip().lower()
        for rank in response.ranks
        if rank.error
    }
    return (
        bool(errors)
        and raised in rank_failures
        and all(
            any(marker in error for marker in _TRANSIENT_PRESSURE_RANK_ERRORS)
            for error in errors
        )
    )


def profile_trace_name(session_id: str, case_key: str, run_index: int) -> str:
    """Return a stable Kineto-safe trace name below the engine's 64-byte limit."""
    case_digest = hashlib.sha256(case_key.encode("utf-8")).hexdigest()[:16]
    name = f"cg_{session_id[:12]}_{case_digest}_r{run_index}"
    if len(name.encode("utf-8")) > 64:
        raise ValueError("generated profile trace name exceeds 64 bytes")
    return name


def normalize_cache_case(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Validate explicit cases; derive only grouped summary geometry."""
    if not isinstance(raw, dict):
        raise ValueError("cache grid case must be an object")

    def integer(source, key):
        if key not in source or type(source[key]) is not int:
            raise ValueError(f"cache grid requires integer {key}: {source}")
        return source[key]

    case = dict(raw)
    case_id = integer(raw, "case_id")
    batch = integer(raw, "batch_size")
    if case_id < 0 or batch <= 0:
        raise ValueError("case_id must be nonnegative and batch_size positive")
    if "request_groups" not in raw:
        input_len = integer(raw, "input_len")
        cache_len = integer(raw, "cache_len")
        if batch != 1 or "prefix_policy" in raw:
            raise ValueError(
                "batch cases require explicit request_groups and prefix_policy"
            )
        if not 0 <= cache_len < input_len:
            raise ValueError("require 0 <= cache_len < input_len")
        return case
    if raw.get("prefix_policy") not in ("independent", "shared_by_group"):
        raise ValueError("grouped cases require explicit valid prefix_policy")
    groups = raw["request_groups"]
    if not isinstance(groups, list) or not groups:
        raise ValueError("request_groups must be a nonempty list")
    normalized = []
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("each request group must be an object")
        item = {k: integer(group, k) for k in ("count", "input_len", "cache_len")}
        if item["count"] <= 0 or not 0 <= item["cache_len"] < item["input_len"]:
            raise ValueError(f"invalid request group: {group}")
        normalized.append(item)
    if sum(g["count"] for g in normalized) != batch:
        raise ValueError("request group counts must sum to batch_size")
    for key in ("input_len", "cache_len"):
        value = max(g[key] for g in normalized)
        if key in raw and integer(raw, key) != value:
            raise ValueError(f"grouped summary {key} must equal group maximum")
        case[key] = value
    case["request_groups"] = normalized
    return case


def is_grouped_case(case: Dict[str, Any]) -> bool:
    return "request_groups" in case


def validate_full_pressure_cases(cases, family_schedule):
    """Share case compatibility between offline plans and real pressure entry."""
    if any(
        is_grouped_case(case) or int(case.get("batch_size", 1)) != 1 for case in cases
    ):
        raise ValueError("full pressure requires ungrouped batch_size=1")
    if family_schedule == "tiered_rolling" and any(
        int(case["cache_len"]) == 0 for case in cases
    ):
        raise ValueError("tiered rolling excludes cache_len=0 cases")


def build_grouped_prompts(factory, case, run_count):
    """Build per-request prompts with distinct suffixes across all rounds."""
    case = normalize_cache_case(case)
    members = []
    for group_index, group in enumerate(case["request_groups"]):
        shared = case["prefix_policy"] == "shared_by_group"
        shared_prompts = None
        for member in range(group["count"]):
            identity = f"{case['case_id']}_g{group_index}_r{0 if shared else member}"
            if shared_prompts is None:
                prompts = factory.build_case_prompts(
                    identity,
                    group["input_len"],
                    group["cache_len"],
                    run_count * group["count"] if shared else run_count,
                )
                if shared:
                    shared_prompts = prompts
            else:
                prompts = shared_prompts
            indices = (
                [r * group["count"] + member for r in range(run_count)]
                if shared
                else list(range(run_count))
            )
            members.append(
                {
                    "group_index": group_index,
                    "request_index": len(members),
                    "prefix_id": identity,
                    "input_len": group["input_len"],
                    "cache_len": group["cache_len"],
                    "prompts": CasePrompts(
                        prompts.seed_text,
                        prompts.prefix_ids,
                        [prompts.run_texts[r] for r in indices],
                        [prompts.run_ids[r] for r in indices],
                        prompts.built_len,
                    ),
                }
            )
    return members


def resume_config_fingerprint(config: Dict[str, Any] | None) -> str | None:
    """Return a stable fingerprint for the model/workload configuration."""
    if config is None:
        return None
    config = _sanitize_nested_provenance("run_config", config)
    canonical = json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def validate_cache_grid_resume(
    result_dir: str,
    *,
    grid_sha256: str | None,
    profile_sha256: str | None,
    measure_runs: int,
    cache_commit_tail_tokens: int,
    expected_block_size: int,
    request_transport: str,
    run_config: Dict[str, Any] | None = None,
    shared_seed: bool = False,
    pressure_mode: str = "case",
    allow_resume_mismatch: bool = False,
    require_resume: bool = False,
) -> Dict[str, Any] | None:
    """Validate a checkpoint before expensive tokenizer/model initialization."""
    result_path = Path(result_dir) / "cache_grid_results.json"
    if not result_path.exists():
        if require_resume:
            raise ValueError(
                f"cache grid: --require_cache_resume was set but checkpoint "
                f"does not exist: {result_path}"
            )
        return None
    with result_path.open(encoding="utf-8") as stream:
        payload = json.load(stream)

    current_metrics(payload, str(result_path))
    required = {
        "seed_mode",
        "grid_sha256",
        "profile_sha256",
        "measure_runs",
        "cache_commit_tail_tokens",
        "request_transport",
        "expected_block_size",
        "run_config_sha256",
    }
    if required - payload.keys():
        raise ValueError("checkpoint lacks current resume guards; retest")
    if payload.get("measurement_contract") != SERVER_LATENCY_CONTRACT:
        raise ValueError(
            "checkpoint measurement_contract differs; retest with server first_token_cost_time"
        )
    expected_seed_mode = (
        "continuous_shared_prefix_v1"
        if pressure_mode == "continuous"
        else ("shared_prefix_v1" if shared_seed else "independent")
    )
    if payload["seed_mode"] != expected_seed_mode:
        raise ValueError("cache seed mode mismatch; use a new result directory")
    mismatches = []
    checks = (
        ("grid_sha256", payload.get("grid_sha256"), grid_sha256),
        ("profile_sha256", payload.get("profile_sha256"), profile_sha256),
        ("measure_runs", payload.get("measure_runs"), measure_runs),
        (
            "cache_commit_tail_tokens",
            payload.get("cache_commit_tail_tokens"),
            cache_commit_tail_tokens,
        ),
        (
            "request_transport",
            payload["request_transport"],
            request_transport,
        ),
    )
    for name, old, new in checks:
        if old != new:
            mismatches.append(f"{name}: {old} vs {new}")
    old_block_size = payload.get("expected_block_size")
    if old_block_size != expected_block_size:
        mismatches.append(
            f"expected_block_size: {old_block_size} vs {expected_block_size}"
        )
    old_config_sha = payload.get("run_config_sha256")
    new_config_sha = resume_config_fingerprint(run_config)
    if old_config_sha != new_config_sha:
        mismatches.append(f"run_config_sha256: {old_config_sha} vs {new_config_sha}")

    if mismatches:
        detail = "; ".join(mismatches)
        if allow_resume_mismatch:
            logging.warning(
                "cache grid: resuming with mismatched parameters "
                "(--allow_resume_mismatch): %s",
                detail,
            )
        else:
            raise ValueError(
                f"cache grid: existing results at {result_path} have "
                f"incompatible parameters: {detail}. Pass "
                f"--allow_resume_mismatch to override."
            )
    return payload


def _make_http_session() -> requests.Session:
    """Keep one localhost connection alive while forwards remain serial."""
    session = requests.Session()
    session.mount(
        "http://",
        HTTPAdapter(pool_connections=1, pool_maxsize=1, max_retries=0),
    )
    session.headers.update({"Connection": "keep-alive"})
    return session


def _encode(tokenizer: Any, text: str) -> List[int]:
    return list(tokenizer.encode(text))


class FastConstructionUnavailable(Exception):
    """Internal: the id-layer prompt construction cannot be trusted here."""


@dataclass
class CasePrompts:
    """All prompts needed to execute one cache-grid case.

    ``run_ids`` and ``prefix_ids`` may be empty when the prompts were loaded
    from a precomputed store (geometry was verified at materialization time).
    """

    seed_text: str
    prefix_ids: List[int] = field(default_factory=list)
    run_texts: List[str] = field(default_factory=list)
    run_ids: List[List[int]] = field(default_factory=list)
    built_len: int = 0
    run_specs: Optional[List[Dict[str, Any]]] = None


class PrefixPromptFactory:
    """Build exact-length prompts and verify their shared token prefix."""

    SHARED_PREFIX_MARKER = "cache_grid_shared_prefix_"

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        # BPE tokenizers generally preserve this family exactly.  We still
        # verify every generated prompt before sending it to the server.
        self._word = " hello"
        # The filler is a single stable token for this tokenizer.  Use a
        # one-pass candidate and retain its token ids instead of repeatedly
        # re-encoding exponentially growing strings.
        self._filler_ids = _encode(self.tokenizer, self._word)
        self._fast_path = len(self._filler_ids) == 1
        self._last_exact_key = None
        self._last_exact_ids = None
        self._last_exact_text = None
        # Id-layer caches: marker -> ids (None when the marker is unsafe) and
        # the set of prompt shapes whose first fully verified case passed.
        self._marker_ids_cache: Dict[str, Optional[List[int]]] = {}
        self._tail_ids_cache: Dict[str, Optional[List[int]]] = {}
        self._verified_shapes: set = set()
        self._unsafe_shapes: set = set()

    def _store_exact(self, key, text, ids):
        self._last_exact_key = key
        self._last_exact_text = text
        self._last_exact_ids = ids
        return text, ids

    def _exact_text_and_ids(self, target_len: int, prefix: str):
        if target_len <= 0:
            raise ValueError(f"input length must be positive, got {target_len}")
        key = (int(target_len), prefix)
        if self._last_exact_key == key and self._last_exact_ids is not None:
            return self._last_exact_text, self._last_exact_ids

        # Fast path: with a one-token filler, construct the exact target in one
        # full encode.  Prefix/suffix boundaries are checked by the caller.
        if self._fast_path:
            prefix_ids = _encode(self.tokenizer, prefix)
            needed = target_len - len(prefix_ids)
            if needed >= 0:
                candidate = prefix + self._word * needed
                candidate_ids = _encode(self.tokenizer, candidate)
                if len(candidate_ids) == target_len:
                    return self._store_exact(key, candidate, candidate_ids)

        # Conservative fallback for tokenizers whose filler merges or expands.
        text = prefix
        repeats = 1
        while len(_encode(self.tokenizer, text + self._word * repeats)) < target_len:
            repeats *= 2
        lo, hi = 0, repeats
        while lo < hi:
            mid = (lo + hi) // 2
            if len(_encode(self.tokenizer, text + self._word * mid)) < target_len:
                lo = mid + 1
            else:
                hi = mid
        candidate = text + self._word * lo
        candidate_ids = _encode(self.tokenizer, candidate)
        if len(candidate_ids) == target_len:
            return self._store_exact(key, candidate, candidate_ids)

        lo, hi = 0, len(candidate)
        best = prefix
        best_ids = _encode(self.tokenizer, best)
        while lo <= hi:
            mid = (lo + hi) // 2
            cur = candidate[:mid]
            cur_ids = _encode(self.tokenizer, cur)
            n = len(cur_ids)
            if n <= target_len:
                if n == target_len:
                    return self._store_exact(key, cur, cur_ids)
                best, best_ids = cur, cur_ids
                lo = mid + 1
            else:
                hi = mid - 1
        if len(best_ids) != target_len:
            raise ValueError(
                f"cannot construct exact tokenizer length {target_len}; "
                f"got {len(best_ids)}"
            )
        return self._store_exact(key, best, best_ids)

    def _exact_text(self, target_len: int, prefix: str) -> str:
        text, _ = self._exact_text_and_ids(target_len, prefix)
        return text

    def make_case(
        self, case_id: int | str, total_len: int, cache_len: int
    ) -> Tuple[str, str, int]:
        if cache_len < 0 or cache_len >= total_len:
            raise ValueError(
                f"cache_len must satisfy 0 <= cache_len < total_len, "
                f"got {cache_len}/{total_len}"
            )
        # Isolate every geometry. Reusing one marker across cases lets a larger
        # previously seeded prefix contaminate later, smaller cache requests.
        marker = f"{self.SHARED_PREFIX_MARKER}{case_id}_"
        if cache_len == 0:
            target, target_ids = self._exact_text_and_ids(total_len, marker)
            return target, "", len(target_ids)

        prefix, prefix_ids = self._exact_text_and_ids(cache_len, marker)
        target, target_ids = self._exact_text_and_ids(total_len, prefix + " __suffix_")
        if len(target_ids) != total_len or target_ids[:cache_len] != prefix_ids:
            target, target_ids = self._exact_text_and_ids(total_len, prefix)
        if len(target_ids) != total_len or target_ids[:cache_len] != prefix_ids:
            raise ValueError(
                f"unable to preserve exact prefix for case={case_id}: "
                f"prefix={cache_len}, target={len(target_ids)}"
            )
        return target, prefix, len(target_ids)

    def make_seed(
        self, case_id: int | str, prefix: str, cache_len: int, commit_tail_tokens: int
    ) -> str:
        return self.make_seed_and_ids(case_id, prefix, cache_len, commit_tail_tokens)[0]

    def make_seed_and_ids(
        self, case_id: int | str, prefix: str, cache_len: int, commit_tail_tokens: int
    ) -> Tuple[str, List[int]]:
        if cache_len <= 0:
            return "", []
        seed_len = cache_len + commit_tail_tokens
        seed, seed_ids = self._exact_text_and_ids(
            seed_len, prefix + f" __seed_commit_{case_id}_"
        )
        prefix_ids = _encode(self.tokenizer, prefix)
        if len(prefix_ids) != cache_len or seed_ids[:cache_len] != prefix_ids:
            seed, seed_ids = self._exact_text_and_ids(seed_len, prefix)
        if len(seed_ids) != seed_len or seed_ids[:cache_len] != prefix_ids:
            raise ValueError(
                f"unable to build cache seed for case={case_id}: "
                f"prefix={cache_len}, seed={len(seed_ids)}"
            )
        return seed, seed_ids

    def _safe_marker_ids(self, marker: str) -> Optional[List[int]]:
        """Marker ids when ``marker + filler*k`` tokenizes structurally.

        The probe encodes ``marker`` plus eight fillers once; when the result
        is exactly the marker ids followed by eight filler ids, any filler
        count shares the same structure and prompts of every length can be
        assembled without re-encoding them.
        """
        if not self._fast_path:
            return None
        if marker in self._marker_ids_cache:
            return self._marker_ids_cache[marker]
        marker_ids = _encode(self.tokenizer, marker)
        probe = _encode(self.tokenizer, marker + self._word * 8)
        result = marker_ids if probe == marker_ids + self._filler_ids * 8 else None
        self._marker_ids_cache[marker] = result
        return result

    def _safe_tail_ids(self, tail: str) -> Optional[List[int]]:
        """Tail ids when filler->tail->filler boundaries tokenize structurally."""
        if not self._fast_path:
            return None
        if tail in self._tail_ids_cache:
            return self._tail_ids_cache[tail]
        tail_ids = _encode(self.tokenizer, tail)
        probe = _encode(self.tokenizer, self._word * 4 + tail + self._word * 4)
        expected = self._filler_ids * 4 + tail_ids + self._filler_ids * 4
        result = tail_ids if probe == expected else None
        self._tail_ids_cache[tail] = result
        return result

    def build_case_prompts(
        self, case_id: int | str, total_len: int, cache_len: int, run_count: int
    ) -> CasePrompts:
        """Build the seed prefix and the measured run prompts for one case.

        The fast path assembles exact-length token ids and texts from the
        marker ids plus filler repeats, costing string/list concatenation only.
        The first built case of each shape (cached / cold) is fully re-encoded
        and compared token by token; any mismatch, or a tokenizer without the
        structural filler property, falls back to exact text construction.
        """
        if cache_len < 0 or cache_len >= total_len:
            raise ValueError(
                f"cache_len must satisfy 0 <= cache_len < total_len, "
                f"got {cache_len}/{total_len}"
            )
        if run_count <= 0:
            raise ValueError(f"run_count must be positive, got {run_count}")
        shape = "cached" if cache_len else "cold"
        if shape not in self._unsafe_shapes:
            try:
                prompts = self._fast_case_prompts(
                    case_id, total_len, cache_len, run_count
                )
                if shape not in self._verified_shapes:
                    self._verify_case_prompts(shape, prompts)
                return prompts
            except FastConstructionUnavailable:
                pass
        return self._text_case_prompts(case_id, total_len, cache_len, run_count)

    def _fast_case_prompts(
        self, case_id: int | str, total_len: int, cache_len: int, run_count: int
    ) -> CasePrompts:
        if cache_len == 0:
            run_texts: List[str] = []
            run_ids: List[List[int]] = []
            run_specs: List[Dict[str, Any]] = []
            for run_idx in range(run_count):
                marker = f"case_{case_id}_cold_run_{run_idx}_"
                marker_ids = self._safe_marker_ids(marker)
                if marker_ids is None or len(marker_ids) > total_len:
                    raise FastConstructionUnavailable()
                fillers = total_len - len(marker_ids)
                run_texts.append(marker + self._word * fillers)
                run_ids.append(marker_ids + self._filler_ids * fillers)
                run_specs.append({"marker": marker, "fillers": fillers})
            return CasePrompts("", [], run_texts, run_ids, total_len, run_specs)

        marker = f"{self.SHARED_PREFIX_MARKER}{case_id}_"
        marker_ids = self._safe_marker_ids(marker)
        if marker_ids is None or len(marker_ids) > cache_len:
            raise FastConstructionUnavailable()
        prefix_fillers = cache_len - len(marker_ids)
        prefix_ids = marker_ids + self._filler_ids * prefix_fillers
        seed_text = marker + self._word * prefix_fillers
        run_texts = []
        run_ids = []
        run_specs = []
        for run_idx in range(run_count):
            tail = f" __run_{run_idx}_"
            tail_ids = self._safe_tail_ids(tail)
            if tail_ids is None:
                raise FastConstructionUnavailable()
            fillers = total_len - cache_len - len(tail_ids)
            if fillers < 0:
                raise FastConstructionUnavailable()
            run_texts.append(seed_text + tail + self._word * fillers)
            run_ids.append(prefix_ids + tail_ids + self._filler_ids * fillers)
            run_specs.append({"tail": tail, "fillers": fillers})
        return CasePrompts(
            seed_text, prefix_ids, run_texts, run_ids, total_len, run_specs
        )

    def _verify_case_prompts(self, shape: str, prompts: CasePrompts) -> None:
        """Fully re-encode the first case of a shape to prove the id layer."""
        texts = ([prompts.seed_text] if prompts.seed_text else []) + prompts.run_texts
        ids_lists = (
            [prompts.prefix_ids] if prompts.seed_text else []
        ) + prompts.run_ids
        for text, ids in zip(texts, ids_lists):
            if _encode(self.tokenizer, text) != ids:
                self._unsafe_shapes.add(shape)
                logging.warning(
                    "cache grid: id-layer prompt construction failed full "
                    "verification for shape=%s; falling back to text layer",
                    shape,
                )
                raise FastConstructionUnavailable()
        self._verified_shapes.add(shape)
        logging.info(
            "cache grid: id-layer construction verified for shape=%s "
            "(seed=%d tokens, %d runs, %d run tokens)",
            shape,
            len(prompts.prefix_ids),
            len(prompts.run_ids),
            sum(len(x) for x in prompts.run_ids),
        )

    def _text_case_prompts(
        self, case_id: int | str, total_len: int, cache_len: int, run_count: int
    ) -> CasePrompts:
        _, prefix, built_len = self.make_case(case_id, total_len, cache_len)
        prefix_ids = _encode(self.tokenizer, prefix) if cache_len else []
        run_texts: List[str] = []
        run_ids: List[List[int]] = []
        for run_idx in range(run_count):
            if cache_len:
                run_target, ids = self._exact_text_and_ids(
                    total_len, prefix + f" __run_{run_idx}_"
                )
                if ids[:cache_len] != prefix_ids:
                    raise ValueError(f"run {run_idx} did not preserve cache prefix")
            else:
                run_target, ids = self._exact_text_and_ids(
                    total_len, f"case_{case_id}_cold_run_{run_idx}_"
                )
            run_texts.append(run_target)
            run_ids.append(ids)
        return CasePrompts(prefix, prefix_ids, run_texts, run_ids, built_len, None)


class MaterializedCaseStore:
    """Current prompt records, with explicit seed text or marker/filler counts.

    Verbatim records handle tokenizer-dependent prompts that cannot be compressed.
    """

    SCHEMA_VERSION = 2

    def __init__(self, root: str):
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.jsonl"
        self.info_path = self.root / "store_info.json"
        self._records: Dict[int, Dict[str, Any]] = {}
        self._loaded = False
        self._word = " hello"
        self.run_count = -1
        self.grid_metadata: Dict[str, Any] = {}
        self.grid_sha256 = ""
        self.profile_sha256 = ""

    def materialize(
        self,
        cases: Iterable[Dict[str, Any]],
        factory: PrefixPromptFactory,
        run_count: int,
        *,
        grid_metadata: Optional[Dict[str, Any]] = None,
        grid_sha256: str = "",
        profile_sha256: str = "",
    ) -> Dict[str, Any]:
        if run_count <= 0:
            raise ValueError("run_count must be positive")
        self.root.mkdir(parents=True, exist_ok=True)
        stats = {
            "cases": 0,
            "cached_cases": 0,
            "verbatim_seed": 0,
            "verbatim_runs": 0,
            "bytes": 0,
        }
        case_list = [normalize_cache_case(case) for case in cases]
        self._records = {}
        for case in case_list:
            if is_grouped_case(case):
                members = build_grouped_prompts(factory, case, run_count)
                self._records[case["case_id"]] = {
                    **case,
                    "schema_version": self.SCHEMA_VERSION,
                    "members": [
                        {**member, "prompts": asdict(member["prompts"])}
                        for member in members
                    ],
                }
                stats["cases"] += 1
                stats["cached_cases"] += int(any(m["cache_len"] for m in members))
                continue
            prompts = factory.build_case_prompts(
                int(case["case_id"]),
                int(case["input_len"]),
                int(case["cache_len"]),
                run_count,
            )
            record = {
                "schema_version": self.SCHEMA_VERSION,
                "case_id": int(case["case_id"]),
                "batch_size": int(case.get("batch_size", 1)),
                "input_len": int(case["input_len"]),
                "cache_len": int(case["cache_len"]),
            }
            if prompts.seed_text:
                stats["cached_cases"] += 1
                cache_len = int(case["cache_len"])
                marker = f"{factory.SHARED_PREFIX_MARKER}{int(case['case_id'])}_"
                case_marker_ids = _encode(factory.tokenizer, marker)
                fillers = cache_len - len(case_marker_ids)
                decomposed = marker + factory._word * fillers
                if fillers >= 0 and prompts.seed_text == decomposed:
                    record["seed_marker"] = marker
                    record["seed_fillers"] = fillers
                else:
                    record["seed_text"] = prompts.seed_text
                    stats["verbatim_seed"] += 1
            runs = []
            for idx, text in enumerate(prompts.run_texts):
                spec = None
                if prompts.run_specs and idx < len(prompts.run_specs):
                    spec = prompts.run_specs[idx]
                if spec is not None and "tail" in spec and prompts.seed_text:
                    runs.append(spec)
                elif spec is not None and "marker" in spec:
                    runs.append(spec)
                else:
                    runs.append({"text": text})
                    stats["verbatim_runs"] += 1
            record["runs"] = runs
            self._records[int(case["case_id"])] = record
            stats["cases"] += 1
        with self.manifest_path.open("w", encoding="utf-8") as manifest:
            for record in self._records.values():
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.info_path.write_text(
            json.dumps(
                {
                    "schema_version": self.SCHEMA_VERSION,
                    "run_count": run_count,
                    "word": factory._word,
                    "case_count": stats["cases"],
                    "grid_metadata": grid_metadata or {},
                    "grid_sha256": grid_sha256,
                    "profile_sha256": profile_sha256,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        stats["bytes"] = sum(
            f.stat().st_size for f in self.root.rglob("*") if f.is_file()
        )
        self._loaded = True
        self.run_count = run_count
        self.grid_metadata = grid_metadata or {}
        self.grid_sha256 = grid_sha256
        self.profile_sha256 = profile_sha256
        return stats

    def load_cases(self) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        return [
            {
                "case_id": record["case_id"],
                "batch_size": record["batch_size"],
                "input_len": record["input_len"],
                "cache_len": record["cache_len"],
                **(
                    {
                        "request_groups": record["request_groups"],
                        "prefix_policy": record["prefix_policy"],
                    }
                    if "members" in record
                    else {}
                ),
            }
            for record in self._records.values()
        ]

    def load_case(self, case: Dict[str, Any]) -> CasePrompts:
        self._ensure_loaded()
        record = self._records[int(case["case_id"])]
        cache_len = int(record["cache_len"])
        seed_text = ""
        if cache_len:
            if "seed_text" in record:
                seed_text = record["seed_text"]
            elif "seed_marker" in record:
                seed_text = record["seed_marker"] + self._word * int(
                    record["seed_fillers"]
                )
            else:
                raise ValueError(
                    "materialized prefix requires seed_text or seed_marker/seed_fillers"
                )
        run_texts: List[str] = []
        for spec in record["runs"]:
            if "text" in spec:
                run_texts.append(spec["text"])
            elif "tail" in spec:
                run_texts.append(
                    seed_text + spec["tail"] + self._word * int(spec["fillers"])
                )
            else:
                run_texts.append(spec["marker"] + self._word * int(spec["fillers"]))
        return CasePrompts(seed_text, [], run_texts, [], int(record["input_len"]))

    def load_grouped_case(self, case):
        self._ensure_loaded()
        record = self._records[int(case["case_id"])]
        if CacheGridRunner.case_key(record) != CacheGridRunner.case_key(case):
            raise ValueError("materialized batch distribution does not match case")
        return [
            {**member, "prompts": CasePrompts(**member["prompts"])}
            for member in record["members"]
        ]

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"{self.root} is not a materialized case store "
                f"(missing manifest.jsonl)"
            )
        if not self.info_path.is_file():
            raise ValueError("materialized store_info.json required")
        info = json.loads(self.info_path.read_text(encoding="utf-8"))
        if info.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError("materialized schema v2 required")
        if (
            "word" not in info
            or type(info.get("run_count")) is not int
            or info["run_count"] <= 0
        ):
            raise ValueError(
                "materialized v2 store requires word and positive run_count"
            )
        self._word = info["word"]
        self.run_count = info["run_count"]
        self.grid_metadata = info.get("grid_metadata", {})
        self.grid_sha256 = info.get("grid_sha256", "")
        self.profile_sha256 = info.get("profile_sha256", "")
        with self.manifest_path.open(encoding="utf-8") as manifest:
            for line in manifest:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("schema_version") != self.SCHEMA_VERSION:
                    raise ValueError("materialized record schema v2 required")
                self._records[int(record["case_id"])] = record
        self._loaded = True


def _post_prefill(
    port: int,
    prompt: str,
    timeout: int,
    request_id: str,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    body = {
        "prompt": prompt,
        "generate_config": {
            "max_new_tokens": 1,
            "min_new_tokens": 1,
            "force_sp_accept": True,
        },
    }
    started = time.perf_counter()
    try:
        response = (session or requests).post(
            f"http://127.0.0.1:{port}",
            json=body,
            headers={"X-DashScope-LoggingConsent": "none"},
            timeout=timeout,
        )
    except Exception as exc:
        return {
            "success": False,
            "error": repr(exc),
            "request_id": request_id,
            "client_wall_time_ms": (time.perf_counter() - started) * 1000.0,
        }
    client_wall_time_ms = (time.perf_counter() - started) * 1000.0
    if response.status_code != 200:
        return {
            "success": False,
            "error": f"HTTP {response.status_code}: {response.text[:500]}",
            "request_id": request_id,
            "client_wall_time_ms": client_wall_time_ms,
        }
    try:
        data = response.json()
    except Exception as exc:
        return {
            "success": False,
            "error": f"invalid JSON: {exc}",
            "request_id": request_id,
            "client_wall_time_ms": client_wall_time_ms,
        }
    aux = data.get("aux_info") or {}
    return {
        "success": True,
        "request_id": request_id,
        "input_len": int(aux.get("input_len", 0)),
        "output_len": int(aux.get("output_len", 0)),
        "reuse_len": int(aux.get("reuse_len", 0)),
        "prefill_time_ms": float(aux.get("first_token_cost_time", 0.0)),
        "total_time_ms": float(aux.get("cost_time", 0.0)),
        "wait_time_ms": float(aux.get("wait_time", 0.0)),
        "client_wall_time_ms": client_wall_time_ms,
        "ttft_ms": client_wall_time_ms,
        "ttft_source": "client_http_wall_max_new_tokens_1",
    }


def _post_prefill_ids(
    stub,
    input_ids,
    timeout: int,
    request_id: str,
    *,
    register_call: Optional[Callable[[Any], None]] = None,
    before_dispatch: Optional[Callable[[], str]] = None,
    resolve_mock_memory_cache_request: Optional[Callable[[], Optional[bool]]] = None,
) -> Dict[str, Any]:
    """Send token ids and optionally register the cancellable call before reading."""
    from rtp_llm.dash_sc.client import build_model_infer_request
    from rtp_llm.dash_sc.codec import SamplingParams

    started = time.perf_counter()
    try:
        request = build_model_infer_request(
            request_id=request_id,
            model_name="rtp_llm",
            input_ids=input_ids,
            sampling=SamplingParams(max_new_tokens=1, min_new_tokens=1, top_k=1),
            enable_thinking=False,
            force_sp_accept=True,
        )
        if before_dispatch is not None:
            request_id = before_dispatch()
            request.id = request_id
        # Resolve after phase classification, while retaining token/protobuf
        # preparation before the final dispatch permission check.
        if resolve_mock_memory_cache_request is not None:
            mock_request = resolve_mock_memory_cache_request()
            if mock_request is not None:
                request.parameters["mock_memory_cache_request"].bool_param = (
                    mock_request
                )
        responses = stub.ModelStreamInfer(
            iter((request,)),
            timeout=timeout,
            metadata=(("x-dashscope-inner-loggingconsent", "none"),),
        )
        if register_call is not None:
            register_call(responses)
        values: Dict[str, Any] = {}
        output_len = 0
        received_inference = False
        for response in responses:
            if response.error_message:
                raise RuntimeError(response.error_message)
            if not response.HasField("infer_response"):
                continue
            received_inference = True
            infer = response.infer_response
            for name in (
                "engine_cost_time_us",
                "engine_first_token_cost_time_us",
                "engine_wait_time_us",
            ):
                if name in infer.parameters:
                    values[name] = infer.parameters[name].int64_param / 1000.0
            for index, output in enumerate(infer.outputs):
                if index >= len(infer.raw_output_contents):
                    continue
                raw = infer.raw_output_contents[index]
                if output.name == "generated_ids" and output.datatype == "INT32":
                    shape = list(output.shape)
                    output_len = max(
                        output_len, int(shape[-1]) if shape else len(raw) // 4
                    )
                elif output.datatype == "INT32" and len(raw) >= 4:
                    values[output.name] = struct.unpack_from("<i", raw)[0]
                elif output.datatype == "FP64" and len(raw) >= 8:
                    values[output.name] = struct.unpack_from("<d", raw)[0]
        if not received_inference:
            raise RuntimeError("Dash-SC stream ended without an inference response")
        wall_ms = (time.perf_counter() - started) * 1000.0
        return {
            "success": True,
            "request_id": request_id,
            "input_len": int(values.get("prompt_token_num", 0)),
            "output_len": output_len,
            "reuse_len": int(values.get("prompt_cached_token_num", 0)),
            "prefill_time_ms": float(
                values.get("engine_first_token_cost_time_us", 0.0)
            ),
            "total_time_ms": float(values.get("engine_cost_time_us", 0.0)),
            "wait_time_ms": float(values.get("engine_wait_time_us", 0.0)),
            "client_wall_time_ms": wall_ms,
            "ttft_ms": wall_ms,
            "ttft_source": "client_dashsc_grpc_input_ids_wall_max_new_tokens_1",
        }
    except DispatchClosed:
        raise
    except Exception as exc:
        return {
            "success": False,
            "error": repr(exc),
            "request_id": request_id,
            "client_wall_time_ms": (time.perf_counter() - started) * 1000.0,
        }


class CacheGridRunner:
    """Run and checkpoint a total-seq × prefix-cache grid."""

    def __init__(
        self,
        port: int,
        tokenizer: Any,
        cases: Iterable[Dict[str, Any]],
        result_dir: str,
        *,
        request_timeout: int = 7200,
        shared_seed: bool = False,
        pressure_mode: str = "case",
        pressure_scheduler: str = "batch_decode",
        pressure_batch_size: int = 1,
        pressure_client_concurrency: int = 2,
        pressure_seed_count: int | None = None,
        pressure_request_rate: float = 0.0,
        pressure_arrival: str = "constant",
        pressure_random_seed: int = 0,
        pressure_config=None,
        measure_runs: int | None = None,
        checkpoint_every: int = 100,
        cache_commit_tail_tokens: int = 4096,
        fail_fast: bool = True,
        skip_reuse_validation: bool = False,
        grid_metadata: Dict[str, Any] | None = None,
        grid_sha256: str | None = None,
        expected_block_size: int = 0,
        case_store: "MaterializedCaseStore | None" = None,
        profile: Dict[str, Any] | None = None,
        profile_sha256: str | None = None,
        allow_resume_mismatch: bool = False,
        require_resume: bool = False,
        request_transport: str = "http_prompt",
        grpc_port: int | None = None,
        run_config: Dict[str, Any] | None = None,
        profile_runs: int = 0,
        profile_case_ids: Iterable[int] = (),
        profile_only: bool = False,
        profile_flat_output: bool = False,
        profile_tp_size: int = 1,
        profile_trace_timeout: float = 120.0,
        profile_backend: str = "kineto",
        nsys_path: str = "nsys",
        nsys_session: str = "",
        nsys_tail_seconds: float = 0.1,
    ):
        self.port = port
        if pressure_config is None and pressure_seed_count is None:
            pressure_seed_count = 4
        self.pressure_config = pressure_config
        if pressure_config is not None:
            for name in ("explicit_smoke", "diagnostic_only"):
                marker = (profile or {}).get("cache_grid", {}).get(name, False)
                if type(marker) is not bool:
                    raise ValueError(f"cache_grid.{name} must be a boolean")
            from dataclasses import asdict

            from .pressure_diagnostics import resolve_pressure_config

            self.pressure_config = resolve_pressure_config(asdict(pressure_config))
            startup = (run_config or {}).get("full_pressure_startup", {})
            if pressure_mode != "continuous":
                raise ValueError("pressure preload requires continuous mode")
            if pressure_scheduler != "fifo" or pressure_batch_size != 1:
                raise ValueError("pressure preload requires FIFO and batch size 1")
            if measure_runs is not None:
                raise ValueError(
                    "pressure time window conflicts with explicit count/runs"
                )
            if allow_resume_mismatch:
                raise ValueError("full pressure resume requires immutable settings")
            if grpc_port not in (None, port + 8):
                raise ValueError(
                    "full pressure requires the owned input-ID service endpoint"
                )
            if (
                startup.get("dp_size") != 1
                or startup.get("exclusive_service") is not True
            ):
                raise ValueError("full pressure requires DP=1 and exclusive service")
            if (run_config or {}).get("workspace_tokens") is not None or (
                grid_metadata or {}
            ).get("generator", {}).get("workspace_tokens") is not None:
                raise ValueError("full pressure cannot use fixed workspace")
            run_config = dict(
                run_config or {},
                cache_pressure_preload=True,
                pressure_settings=asdict(self.pressure_config),
            )
            run_config["workload"] = dict(
                run_config.get("workload", {}),
                cache_pressure_mode=pressure_mode,
                cache_pressure_scheduler=pressure_scheduler,
                cache_pressure_batch_size=pressure_batch_size,
                cache_pressure_client_concurrency=pressure_client_concurrency,
                cache_pressure_seed_count=pressure_seed_count,
                cache_pressure_request_rate=pressure_request_rate,
                cache_pressure_arrival=pressure_arrival,
                cache_pressure_random_seed=pressure_random_seed,
            )
        measure_runs = 3 if measure_runs is None else measure_runs
        if request_transport not in {"http_prompt", "dashsc_input_ids"}:
            raise ValueError(
                f"unsupported cache request transport: {request_transport}"
            )
        self.request_transport = request_transport
        self.grpc_port = int(grpc_port if grpc_port is not None else port + 8)
        self._grpc_channel = None
        self._grpc_stub = None
        if self.request_transport == "dashsc_input_ids":
            import grpc

            from rtp_llm.dash_sc.client import dash_sc_grpc_client_channel_options
            from rtp_llm.dash_sc.proto import predict_v2_pb2_grpc

            self._grpc_channel = grpc.insecure_channel(
                f"127.0.0.1:{self.grpc_port}",
                options=dash_sc_grpc_client_channel_options(),
            )
            self._grpc_stub = predict_v2_pb2_grpc.GRPCInferenceServiceStub(
                self._grpc_channel
            )
        self.factory = PrefixPromptFactory(tokenizer)
        self.cases = [normalize_cache_case(case) for case in cases]
        if pressure_config is not None and (
            not self.cases
            or any(
                c["cache_len"]
                and (
                    cache_commit_tail_tokens <= 0
                    or c["cache_len"] % cache_commit_tail_tokens
                    or c["cache_len"] + cache_commit_tail_tokens > c["input_len"]
                )
                for c in self.cases
                if not is_grouped_case(c)
            )
        ):
            raise ValueError("full pressure requires cases with complete commit tails")
        if len({c["case_id"] for c in self.cases}) != len(self.cases):
            raise ValueError("cache grid case_id values must be unique")
        self.shared_seed = shared_seed
        self._shared_seed_ids = []
        self._shared_seed_record = None
        self._shared_seed_refreshes = []
        self.pressure_mode = pressure_mode
        self.pressure_scheduler = pressure_scheduler
        self.pressure_batch_size = pressure_batch_size
        self.pressure_client_concurrency = pressure_client_concurrency
        self.pressure_seed_count = pressure_seed_count
        self.pressure_request_rate = pressure_request_rate
        self.pressure_arrival = pressure_arrival
        self.pressure_random_seed = pressure_random_seed
        self._pressure_seed_plan = None
        self._continuous_seed_ids: List[List[int]] = []
        self._continuous_seed_records: List[Dict[str, Any]] = []
        self._continuous_summary: Dict[str, Any] | None = None
        self._continuous_prompt_lock = threading.Lock()
        if pressure_mode not in {"case", "continuous"}:
            raise ValueError(f"unsupported cache pressure mode: {pressure_mode}")
        if pressure_scheduler not in {"batch_decode", "fifo"}:
            raise ValueError(
                f"unsupported continuous pressure scheduler: {pressure_scheduler}"
            )
        if pressure_mode != "continuous" and pressure_scheduler != "batch_decode":
            raise ValueError("FIFO cache pressure requires continuous mode")
        if pressure_mode == "continuous":
            if pressure_batch_size <= 0 or pressure_client_concurrency <= 0:
                raise ValueError(
                    "continuous pressure batch size and concurrency must be positive"
                )
            if pressure_seed_count is None:
                if pressure_config is None:
                    raise ValueError(
                        "automatic seed count requires full-pressure preload"
                    )
            elif type(pressure_seed_count) is not int or pressure_seed_count <= 0:
                raise ValueError("continuous pressure seed count must be positive")
            if not math.isfinite(pressure_request_rate) or pressure_request_rate < 0:
                raise ValueError(
                    "continuous pressure request rate must be non-negative"
                )
            if pressure_arrival not in {"constant", "poisson"}:
                raise ValueError(
                    "continuous pressure arrival must be constant or poisson"
                )
            if (
                pressure_scheduler == "batch_decode"
                and pressure_client_concurrency < pressure_batch_size
            ):
                raise ValueError(
                    "continuous pressure client concurrency must be at least batch size"
                )
            if pressure_scheduler == "fifo" and pressure_batch_size != 1:
                raise ValueError(
                    "continuous FIFO pressure does not use pressure_batch_size; "
                    "leave it at 1"
                )
            if shared_seed:
                raise ValueError(
                    "continuous pressure cannot be combined with shared seed"
                )
            if request_transport != "dashsc_input_ids":
                raise ValueError(
                    "continuous pressure requires dashsc_input_ids transport"
                )
            if any(is_grouped_case(c) or c["batch_size"] != 1 for c in self.cases):
                raise ValueError(
                    "continuous pressure supports only ungrouped batch_size=1"
                )
            continuous_profile = (
                self.pressure_config is not None
                and profile_only
                and profile_runs == 1
                and not tuple(profile_case_ids)
            )
            if case_store is not None or (
                (profile_runs or profile_only) and not continuous_profile
            ):
                raise ValueError(
                    "continuous pressure does not support materialized cases or profiling"
                )
            if (
                not expected_block_size
                or cache_commit_tail_tokens < expected_block_size
            ):
                raise ValueError(
                    "continuous pressure requires a known block size and at least "
                    "one block commit tail"
                )
            run_config = dict(run_config or {})
            run_config["cache_seed_mode"] = "continuous_shared_prefix_v1"
            if pressure_scheduler == "fifo":
                run_config["cache_pressure_scheduler"] = "fifo"
        if shared_seed:
            if request_transport != "dashsc_input_ids":
                raise ValueError("shared seed requires dashsc_input_ids transport")
            if any(is_grouped_case(c) or c["batch_size"] != 1 for c in self.cases):
                raise ValueError("shared seed supports only ungrouped batch_size=1")
            if case_store is not None or profile_runs or profile_only:
                raise ValueError(
                    "shared seed does not support materialized cases or profiling"
                )
            if (
                not expected_block_size
                or cache_commit_tail_tokens < expected_block_size
            ):
                raise ValueError(
                    "shared seed requires a known block size and at least one block commit tail"
                )
            run_config = dict(run_config or {})
            run_config["cache_seed_mode"] = "shared_prefix_v1"
        self.profile_runs = profile_runs
        self.profile_case_ids = set(profile_case_ids)
        if any(
            is_grouped_case(c) and c["case_id"] in self.profile_case_ids
            for c in self.cases
        ):
            raise ValueError(
                "cache profiling currently supports only ungrouped batch_size=1 cases"
            )
        self.profile_only = profile_only
        self.profile_flat_output = profile_flat_output
        if profile_flat_output and not profile_only:
            raise ValueError("flat profile output requires profile-only mode")
        self.profile_tp_size = profile_tp_size
        self.profile_trace_timeout = profile_trace_timeout
        self.profile_backend = profile_backend
        self.nsys_path = nsys_path
        self.nsys_session = nsys_session
        self.nsys_tail_seconds = nsys_tail_seconds
        if profile_backend not in {"kineto", "nsys"}:
            raise ValueError("unknown profile backend")
        if profile_backend == "nsys" and (not profile_only or not nsys_session):
            raise ValueError("nsys requires profile-only mode and a named session")
        if (
            profile_backend == "nsys"
            and self.pressure_config is not None
            and self.pressure_config.nsys_duration_s is None
        ):
            raise ValueError("continuous Nsight requires explicit nsys_duration_s")
        if not 0 <= nsys_tail_seconds <= 60:
            raise ValueError("nsys tail seconds must be between 0 and 60")
        if profile_runs < 0 or profile_trace_timeout <= 0 or profile_tp_size <= 0:
            raise ValueError("invalid cache profile runs, timeout or TP size")
        if not (
            pressure_mode == "continuous"
            and self.pressure_config is not None
            and profile_only
        ) and (
            bool(profile_runs) != bool(self.profile_case_ids)
            or (profile_only and not profile_runs)
        ):
            raise ValueError(
                "cache profiling requires positive runs and explicit case IDs"
            )
        unknown = self.profile_case_ids - {int(c["case_id"]) for c in self.cases}
        if unknown:
            raise ValueError(f"unknown cache profile case IDs: {sorted(unknown)}")
        self.case_store = case_store
        if case_store is not None and case_store.run_count not in (-1, measure_runs):
            raise ValueError(
                f"materialized case store was built for run_count="
                f"{case_store.run_count} but the runner uses measure_runs="
                f"{measure_runs}"
            )
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        if request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        if measure_runs <= 0:
            raise ValueError("measure_runs must be positive")
        if cache_commit_tail_tokens <= 0:
            raise ValueError("cache_commit_tail_tokens must be positive")
        if expected_block_size < 0:
            raise ValueError("expected_block_size must be non-negative")
        self.request_timeout = request_timeout
        self.measure_runs = measure_runs
        self.checkpoint_every = max(1, checkpoint_every)
        self.cache_commit_tail_tokens = cache_commit_tail_tokens
        self.fail_fast = fail_fast
        self.skip_reuse_validation = skip_reuse_validation
        self.grid_metadata = grid_metadata or {}
        self.grid_sha256 = grid_sha256
        self.expected_block_size = expected_block_size
        self.profile = _sanitize_nested_provenance("profile", profile)
        self.profile_sha256 = profile_sha256
        self.run_config = _sanitize_nested_provenance("run_config", run_config)
        self.run_config_sha256 = resume_config_fingerprint(self.run_config)
        self.result_path = self.result_dir / "cache_grid_results.json"
        self.progress_path = self.result_dir / "cache_grid_progress.json"
        self.journal_path = self.result_dir / "cache_grid_results.journal.jsonl"
        self._results: Dict[str, Dict[str, Any]] = {}
        self._resume_count = 0
        self._started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        payload = validate_cache_grid_resume(
            str(self.result_dir),
            grid_sha256=self.grid_sha256,
            profile_sha256=self.profile_sha256,
            measure_runs=self.measure_runs,
            cache_commit_tail_tokens=self.cache_commit_tail_tokens,
            expected_block_size=self.expected_block_size,
            request_transport=self.request_transport,
            run_config=self.run_config,
            shared_seed=self.shared_seed,
            pressure_mode=self.pressure_mode,
            allow_resume_mismatch=allow_resume_mismatch,
            require_resume=require_resume,
        )
        if payload is None and self.journal_path.exists():
            raise ValueError(
                f"cache grid: found {self.journal_path} without the base "
                f"{self.result_path}; restore the base checkpoint or use a new "
                f"result directory"
            )
        if payload is not None:
            planned_keys = {self.case_key(case) for case in self.cases}
            self._results = {
                str(row["case_key"]): row
                for row in payload.get("metrics", [])
                if isinstance(row, dict) and row.get("case_key") in planned_keys
            }
            self._load_journal(planned_keys)
            if self.shared_seed:
                self._shared_seed_record = payload.get("shared_seed")
                self._shared_seed_refreshes = payload.get("shared_seed_refreshes", [])
            if self.pressure_mode == "continuous":
                self._continuous_seed_records = payload.get("continuous_seeds", [])
                self._continuous_summary = payload.get("continuous_pressure")
            self._resume_count = int(payload.get("resume_count", 0)) + 1
            self._started_at = payload.get("started_at", self._started_at)
            progress = payload.get("progress") or {}
            logging.info(
                "cache grid: checkpoint loaded resume_count=%d completed=%s/%s "
                "next=%s",
                self._resume_count,
                progress.get("completed_cases", len(self._results)),
                progress.get("total_cases", len(self.cases)),
                progress.get("next_pending_case"),
            )
        self._http_session = _make_http_session()
        self._checkpoint_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="checkpoint-writer"
        )
        self._checkpoint_future: Optional[Future] = None

    @staticmethod
    def case_key(case: Dict[str, Any]) -> str:
        if is_grouped_case(case):
            case = normalize_cache_case(case)
            digest = resume_config_fingerprint(
                {
                    "request_groups": case["request_groups"],
                    "prefix_policy": case["prefix_policy"],
                    "execution_mode": "scheduler_fixed_batch_v1",
                }
            )
            return f"bs{case['batch_size']}_dist{digest[:16]}"
        return f"bs{case['batch_size']}_seq{case['input_len']}_cache{case['cache_len']}"

    def _load_journal(self, planned_keys: set[str]) -> None:
        if not self.journal_path.exists():
            return
        loaded = 0
        with self.journal_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    logging.warning(
                        "cache grid: ignoring malformed journal record at %s:%d",
                        self.journal_path,
                        line_number,
                    )
                    continue
                current_metrics(
                    {
                        "schema_version": 2,
                        "mode": "prefix_cache_grid",
                        "metrics": [row],
                    },
                    str(self.journal_path),
                )
                if row.get("case_key") in planned_keys:
                    self._results[str(row["case_key"])] = row
                    loaded += 1
        if loaded:
            logging.info(
                "cache grid: replayed %d durable case records from %s",
                loaded,
                self.journal_path,
            )

    def _append_journal(self, metric: Dict[str, Any]) -> None:
        encoded = (
            json.dumps(metric, separators=(",", ":"), ensure_ascii=False) + "\n"
        ).encode("utf-8")
        with self.journal_path.open("a+b") as stream:
            size = stream.tell()
            if size:
                stream.seek(-1, os.SEEK_END)
                if stream.read(1) != b"\n":
                    stream.write(b"\n")
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())

    def _write_json_atomic(self, path: Path, payload: Dict[str, Any]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
        os.replace(tmp, path)

    def _write_checkpoint(self, payload: Dict[str, Any]) -> None:
        self._write_json_atomic(self.result_path, payload)

    def _write_progress_checkpoint(self) -> None:
        self._write_json_atomic(
            self.progress_path,
            {
                "schema_version": 1,
                "mode": "prefix_cache_grid_progress",
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "resume_count": self._resume_count,
                "run_config_sha256": self.run_config_sha256,
                "progress": self._progress(),
            },
        )

    @staticmethod
    def _case_descriptor(case: Dict[str, Any]) -> Dict[str, Any]:
        return {
            **case,
            "case_key": CacheGridRunner.case_key(case),
            "case_id": int(case["case_id"]),
            "batch_size": int(case.get("batch_size", 1)),
            "input_len": int(case["input_len"]),
            "cache_len": int(case["cache_len"]),
        }

    def _progress(self) -> Dict[str, Any]:
        completed = []
        failed = []
        next_pending = None
        for case in self.cases:
            row = self._results.get(self.case_key(case))
            if row is not None and row.get("status") == "ok":
                completed.append(case)
            else:
                if next_pending is None:
                    next_pending = case
                if row is not None:
                    failed.append(case)
        completed_count = len(completed)
        return {
            "total_cases": len(self.cases),
            "attempted_cases": completed_count + len(failed),
            "completed_cases": completed_count,
            "failed_or_invalid_cases": len(failed),
            "pending_cases": len(self.cases) - completed_count,
            "progress_pct": (
                round(100.0 * completed_count / len(self.cases), 6)
                if self.cases
                else 100.0
            ),
            "last_completed_case": (
                self._case_descriptor(completed[-1]) if completed else None
            ),
            "next_pending_case": (
                self._case_descriptor(next_pending) if next_pending else None
            ),
        }

    def _save(self, *, complete: bool = False, asynchronous: bool = False) -> None:
        progress = self._progress()
        payload = {
            "schema_version": 2,
            "measurement_contract": SERVER_LATENCY_CONTRACT,
            "mode": "prefix_cache_grid",
            "seed_mode": (
                "continuous_shared_prefix_v1"
                if self.pressure_mode == "continuous"
                else ("shared_prefix_v1" if self.shared_seed else "independent")
            ),
            "complete": complete,
            "status": "completed" if complete else "in_progress",
            "started_at": self._started_at,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "resume_count": self._resume_count,
            "total_cases": progress["total_cases"],
            "completed_cases": progress["completed_cases"],
            "progress": progress,
            "grid_metadata": self.grid_metadata,
            "grid_sha256": self.grid_sha256,
            "measure_runs": self.measure_runs,
            "cache_commit_tail_tokens": self.cache_commit_tail_tokens,
            "checkpoint_every": self.checkpoint_every,
            "skip_reuse_validation": self.skip_reuse_validation,
            "expected_block_size": self.expected_block_size,
            "request_transport": self.request_transport,
            "run_config": self.run_config,
            "run_config_sha256": self.run_config_sha256,
            "profile": self.profile,
            "profile_sha256": self.profile_sha256,
            "metrics": list(self._results.values()),
        }
        if self.shared_seed:
            payload["shared_seed"] = self._shared_seed_record
            payload["shared_seed_refreshes"] = self._shared_seed_refreshes
        if self.pressure_mode == "continuous":
            payload["pressure_scheduler"] = self.pressure_scheduler
            payload["continuous_seeds"] = self._continuous_seed_records
            payload["continuous_pressure"] = self._continuous_summary
        if asynchronous:
            if self._checkpoint_future is not None:
                self._checkpoint_future.result()
            self._checkpoint_future = self._checkpoint_executor.submit(
                self._write_checkpoint, copy.deepcopy(payload)
            )
            return
        if self._checkpoint_future is not None:
            self._checkpoint_future.result()
            self._checkpoint_future = None
        self._write_checkpoint(payload)
        self._write_progress_checkpoint()

    def _close_resources(self) -> None:
        if self._checkpoint_future is not None:
            self._checkpoint_future.result()
            self._checkpoint_future = None
        self._checkpoint_executor.shutdown(wait=True)
        self._http_session.close()
        if self._grpc_channel is not None:
            self._grpc_channel.close()

    def _post_request(self, prompt: str, input_ids, request_id: str) -> Dict[str, Any]:
        if self.request_transport == "dashsc_input_ids":
            if self._grpc_stub is None:
                raise RuntimeError("Dash-SC gRPC transport is not initialized")
            return _post_prefill_ids(
                self._grpc_stub,
                input_ids,
                self.request_timeout,
                request_id,
                resolve_mock_memory_cache_request=lambda: (
                    False
                    if self.pressure_config is not None
                    and self.pressure_config.evidence_policy == "tree_then_mock_memory"
                    else None
                ),
            )
        return _post_prefill(
            self.port, prompt, self.request_timeout, request_id, self._http_session
        )

    def _probe_reuse_granularity(self) -> None:
        """Fail fast when the assumed physical reuse granularity is wrong."""
        block = self.expected_block_size
        if block <= 0:
            return
        prefix_text, prefix_ids = self.factory._exact_text_and_ids(
            block, PrefixPromptFactory.SHARED_PREFIX_MARKER
        )
        seed_text, seed_ids = self.factory.make_seed_and_ids(
            -1, prefix_text, block, self.cache_commit_tail_tokens
        )
        seed = self._post_request(seed_text, seed_ids, "block_size_probe:seed")
        if not seed.get("success"):
            raise RuntimeError(f"block-size probe seed failed: {seed.get('error')}")
        hit_text, hit_ids = self.factory._exact_text_and_ids(
            2 * block, prefix_text + " __block_probe_"
        )
        if hit_ids[:block] != prefix_ids:
            hit_text, hit_ids = self.factory._exact_text_and_ids(2 * block, prefix_text)
        if hit_ids[:block] != prefix_ids:
            raise RuntimeError(
                "block-size probe could not build a prompt preserving its own prefix"
            )
        hit = self._post_request(hit_text, hit_ids, "block_size_probe:hit")
        if not hit.get("success"):
            raise RuntimeError(f"block-size probe request failed: {hit.get('error')}")
        observed = int(hit.get("reuse_len", -1))
        if observed != block:
            raise RuntimeError(
                f"cache reuse granularity mismatch: probe observed reuse_len={observed}, "
                f"expected {block} tokens per physical block. Every cache-hitting case "
                "would be reported as invalid_reuse. Check --seq_size_per_block (x CP "
                "size when PREFILL_CP_KV_CACHE_SHARDED=1) or fix "
                "grid.generator.cache_alignment and rerun."
            )
        logging.info("[CACHE_GRID] block-size probe passed: reuse_len=%d", block)

    def _build_prompts(
        self, case: Dict[str, Any], total_len: int, cache_len: int
    ) -> CasePrompts:
        if self.case_store is not None:
            prompts = self.case_store.load_case(case)
        else:
            prompts = self.factory.build_case_prompts(
                int(case["case_id"]), total_len, cache_len, self.measure_runs
            )
        if len(prompts.run_texts) != self.measure_runs:
            raise ValueError(
                f"expected {self.measure_runs} run prompts, got "
                f"{len(prompts.run_texts)}"
            )
        return prompts

    def _initialize_shared_seed(self, pending):
        max_cache = max((int(c["cache_len"]) for c in pending), default=0)
        if not max_cache:
            return
        seed_len = max_cache + self.cache_commit_tail_tokens
        started = time.perf_counter()
        seed_id = uuid.uuid4().hex
        _, self._shared_seed_ids = self.factory._exact_text_and_ids(
            seed_len, f"shared_seed_{seed_id}_"
        )
        result = self._post_request("", self._shared_seed_ids, f"shared_seed:{seed_id}")
        self._shared_seed_record = {
            "seed_id": seed_id,
            "input_len": seed_len,
            "max_cache_len": max_cache,
            "result": result,
            "elapsed_s": time.perf_counter() - started,
        }
        self._write_json_atomic(
            self.result_dir / f"shared_seed_{seed_id}.json", self._shared_seed_record
        )
        self._save(complete=False)
        if not (
            result.get("success")
            and result.get("input_len") == seed_len
            and result.get("output_len") == 1
            and result.get("reuse_len") == 0
        ):
            raise RuntimeError(f"shared seed failed shape/reuse validation: {result}")
        logging.info(
            "[CACHE_SHARED_SEED] length=%d elapsed_s=%.3f",
            seed_len,
            self._shared_seed_record["elapsed_s"],
        )

    def _post_shared_measurement(self, case, ids, request_id):
        result = self._post_request("", ids, request_id)
        expected = int(case["cache_len"])
        if not (
            result.get("success")
            and result.get("input_len") == case["input_len"]
            and result.get("output_len") == 1
            and 0 <= result.get("reuse_len", -1) < expected
        ):
            return result
        # A miss can evict the seed or populate this suffix. Never retry that
        # suffix: record the excluded attempt and build an independent branch.
        # Shared cases run in descending cache_len order, so the current case
        # also covers every remaining case. Keep the original IDs immutable:
        # next-case payloads may already have been prepared in another thread.
        refresh_len = expected + self.cache_commit_tail_tokens
        refresh_ids = self._shared_seed_ids[:refresh_len]
        event = {
            "case_id": case["case_id"],
            "request_id": request_id,
            "excluded_request": result,
            "seed_id": self._shared_seed_record["seed_id"],
            "refresh_input_len": refresh_len,
        }
        self._shared_seed_refreshes.append(event)
        event_path = self.result_dir / f"shared_seed_refresh_{uuid.uuid4().hex}.json"
        self._write_json_atomic(event_path, event)
        started = time.perf_counter()
        refresh = self._post_request("", refresh_ids, request_id + ":refresh_seed")
        event["seed_result"] = refresh
        event["refresh_elapsed_s"] = time.perf_counter() - started
        self._write_json_atomic(event_path, event)
        if not (
            refresh.get("success")
            and refresh.get("input_len") == refresh_len
            and refresh.get("output_len") == 1
        ):
            raise RuntimeError(f"shared seed refresh failed: {refresh}")
        fresh_ids = self._prepare_shared_payload(case)["run_ids"][0]
        retry = self._post_request("", fresh_ids, request_id + ":retry")
        event["retry_request_id"] = request_id + ":retry"
        self._write_json_atomic(event_path, event)
        logging.info(
            "[CACHE_SHARED_SEED] refreshed length=%d after reuse=%s expected=%d; retry reuse=%s",
            refresh_len,
            result.get("reuse_len"),
            expected,
            retry.get("reuse_len"),
        )
        return retry  # Normal validation rejects a second miss; no unbounded retries.

    def _prepare_shared_payload(self, case):
        total_len, cache_len = int(case["input_len"]), int(case["cache_len"])
        if (
            cache_len % self.expected_block_size
            or total_len - cache_len < self.cache_commit_tail_tokens
        ):
            raise ValueError(
                "shared seed case requires aligned cache and a full commit tail"
            )
        # Diverge from the seed immediately; nonce tokens distinguish every replay
        # before another full reusable block, even when first tokens coincide.
        prefix = self._shared_seed_ids[:cache_len]
        targets = []
        for _ in range(self.measure_runs):
            nonce = _encode(self.factory.tokenizer, "branch_" + uuid.uuid4().hex)
            different = next(
                (x for x in nonce if x != self._shared_seed_ids[cache_len]), None
            )
            if different is None or len(nonce) + 1 >= self.expected_block_size:
                raise ValueError("cannot construct an isolated shared seed suffix")
            suffix = [different] + nonce
            suffix += (self.factory._filler_ids * (total_len - cache_len))[
                : total_len - cache_len - len(suffix)
            ]
            targets.append(prefix + suffix)
        return {
            "total_len": total_len,
            "cache_len": cache_len,
            "batch_size": 1,
            "built_len": total_len,
            "seed": "",
            "seed_ids": [],
            "run_targets": [""] * self.measure_runs,
            "run_ids": targets,
        }

    @staticmethod
    def _shared_seed_case_order(
        case: Dict[str, Any],
    ) -> tuple[int, int, int, int, int]:
        """Keep the longest shared prefixes hot and defer cold cache pressure."""
        cache_len = int(case["cache_len"])
        return (
            1 if cache_len == 0 else 0,
            -cache_len,
            int(case.get("refinement_level", 0)),
            int(case["input_len"]),
            int(case["case_id"]),
        )

    def _initialize_continuous_seeds(self, pending) -> None:
        max_cache = max((int(case["cache_len"]) for case in pending), default=0)
        self._continuous_seed_ids = []
        self._continuous_seed_records = []
        if max_cache == 0:
            return
        seed_len = max_cache + self.cache_commit_tail_tokens
        seeds = []
        for family in range(self.pressure_seed_count):
            seed_id = uuid.uuid4().hex
            _, ids = self.factory._exact_text_and_ids(
                seed_len, f"continuous_seed_{family}_{seed_id}_"
            )
            seeds.append((family, seed_id, ids))
        if self.pressure_scheduler == "batch_decode":
            self._set_prefill_batch_size(len(seeds))
        barrier = threading.Barrier(len(seeds))

        def send(item):
            family, seed_id, ids = item
            started = time.perf_counter()
            try:
                barrier.wait(timeout=self.request_timeout)
                result = self._post_request(
                    "", ids, f"continuous_seed:{family}:{seed_id}"
                )
            except Exception as exc:
                result = {"success": False, "error": repr(exc)}
            return {
                "family": family,
                "seed_id": seed_id,
                "input_len": seed_len,
                "max_cache_len": max_cache,
                "elapsed_s": time.perf_counter() - started,
                "result": result,
                "ids": ids,
            }

        with ThreadPoolExecutor(
            max_workers=len(seeds), thread_name_prefix="continuous-seed"
        ) as pool:
            futures = [pool.submit(send, seed) for seed in seeds]
            records = [future.result() for future in futures]
        records.sort(key=lambda record: record["family"])
        for record in records:
            result = record["result"]
            if not (
                result.get("success")
                and result.get("input_len") == seed_len
                and result.get("output_len") == 1
                and result.get("reuse_len") == 0
            ):
                raise RuntimeError(
                    "continuous pressure seed failed shape/reuse validation: "
                    f"{result}"
                )
        self._continuous_seed_ids = [record.pop("ids") for record in records]
        self._continuous_seed_records = records
        logging.info(
            "[CACHE_CONTINUOUS] seeded families=%d length=%d",
            len(records),
            seed_len,
        )

    def _continuous_request_ids(self, case, family, nonce):
        total_len = int(case["input_len"])
        cache_len = int(case["cache_len"])
        if cache_len == 0:
            with self._continuous_prompt_lock:
                _, ids = self.factory._exact_text_and_ids(
                    total_len, f"continuous_cold_{nonce}_"
                )
            return ids
        seed_ids = self._continuous_seed_ids[family]
        prefix = seed_ids[:cache_len]
        with self._continuous_prompt_lock:
            nonce_ids = _encode(self.factory.tokenizer, f"pressure_branch_{nonce}_")
        different = next(
            (token for token in nonce_ids if token != seed_ids[cache_len]), None
        )
        suffix_len = total_len - cache_len
        if (
            different is None
            or len(nonce_ids) + 1 >= self.expected_block_size
            or len(nonce_ids) + 1 > suffix_len
        ):
            raise ValueError("cannot construct an isolated continuous-pressure suffix")
        suffix = [different] + nonce_ids
        suffix += (self.factory._filler_ids * suffix_len)[: suffix_len - len(suffix)]
        return prefix + suffix

    def _run_full_pressure(
        self,
        pending,
        config,
        limits,
        pressure_stub,
        target_ranks,
        sink,
        controller=None,
        *,
        seed_plan,
        startup_targets,
        evidence_manifest=None,
    ):
        """Internal opt-in entry; CLI validation/wiring is owned by the caller.

        pressure_stub is the generated RpcServiceStub on the real rank-zero
        endpoint. limits must project resolved server admission/model/token/KV
        bounds, never guesses based on the requested workload.
        """
        from google.protobuf.json_format import MessageToDict

        from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2

        from .continuous_pressure import PressurePoll, PressureSession
        from .pressure_diagnostics import (
            ActivitySample,
            decode_pressure_response,
            raw_pressure_record,
        )
        from .pressure_results import (
            MANIFEST_PROJECTION,
            PressureEvidence,
            PressureWriter,
        )

        cases = tuple(dict(case) for case in pending)
        if not cases or self._grpc_stub is None:
            raise ValueError("full pressure requires cases and input-ID transport")
        validate_full_pressure_cases(cases, config.family_schedule)
        session_id = uuid.uuid4().hex
        if controller is None and getattr(self, "profile_only", False):
            from .pressure_profile import KinetoCapture, NsightCapture

            trace_name = "pressure_" + session_id
            instances = {
                int(rank): instance
                for rank, instance in (evidence_manifest or {}).get("epoch", {}).items()
            }

            def start_capture():
                response = self._http_session.post(
                    f"http://127.0.0.1:{self.port}/start_profile",
                    json=dict(
                        gen_timeline=True,
                        trace_name=trace_name,
                        start_step=0,
                        num_steps=config.profile_steps,
                        enable_all_rank=True,
                    ),
                    timeout=5,
                )
                response.raise_for_status()
                ack = response.json()
                if ack.get("status") != "ok" or ack.get("error"):
                    raise RuntimeError(f"start_profile failed: {ack}")
                return ack

            def read_status(timeout):
                sent = time.monotonic()
                response = pressure_stub.GetCachePressure(
                    model_rpc_service_pb2.CachePressureRequestPB(
                        target_ranks=target_ranks
                    ),
                    timeout=timeout,
                )
                received = time.monotonic()
                # Standard decoder enforces rank identities and clock bounds.
                decode_pressure_response(response, sent, received, target_ranks)
                return raw_pressure_record(response, sent, received)

            def stop_capture():
                response = pressure_stub.StopProfile(
                    model_rpc_service_pb2.StopProfileRequestPB(
                        trace_name=trace_name,
                        enable_all_rank=True,
                        target_ranks=target_ranks,
                    ),
                    timeout=1,
                )
                value = MessageToDict(response, preserving_proto_field_name=True)
                pages = value.get("ranks", [])
                if sorted(int(p.get("world_rank", 0)) for p in pages) != sorted(
                    target_ranks
                ):
                    raise RuntimeError(f"missing stop rank: {value}")
                for page in pages:
                    rank = int(page.get("world_rank", 0))
                    if (
                        not page.get("accepted")
                        or page.get("error")
                        or page.get("service_instance") != instances.get(rank)
                        or page.get("profile", {}).get("trace_name") != trace_name
                    ):
                        raise RuntimeError(f"profile stop failed: {value}")
                return value

            capture_options = dict(
                trace_name=trace_name,
                directory=self.result_dir,
                ranks=target_ranks,
                instances=instances,
                session_id=session_id,
                timeout_s=config.capture_timeout_s,
            )
            if self.profile_backend == "nsys":
                controller = NsightCapture(
                    command=self._nsys_command,
                    duration_s=config.nsys_duration_s,
                    command_timeout_s=self.profile_trace_timeout,
                    **capture_options,
                )
            else:
                capture_options["timeout_s"] = min(
                    config.capture_timeout_s, self.profile_trace_timeout
                )
                controller = KinetoCapture(
                    start_capture=start_capture,
                    read_status=read_status,
                    stop_capture=stop_capture,
                    requested_steps=config.profile_steps,
                    **capture_options,
                )
        seed_count = seed_plan.seed_count
        seed_tokens = seed_plan.seed_input_tokens
        self._continuous_seed_ids = [None] * seed_count
        producer_lock = threading.Lock()
        rng = random.Random(self.pressure_random_seed)
        order = list(range(len(cases)))
        seed_index = 0
        rewarm_index = 0
        tiered_planner = None
        cursors = {}
        instances = {
            int(rank.get("world_rank", 0)): rank["service_instance"]
            for rank in (evidence_manifest or {})
            .get("startup_pressure", {})
            .get("response", {})
            .get("ranks", [])
            if rank.get("service_instance")
        }

        def diagnostics_control(action, phase, timeout):
            actions = dict(enable=1, phase=2, disable=3)
            sent = time.monotonic()
            record = dict(action=action, phase=phase, client_send_s=sent)
            try:
                response = pressure_stub.ControlPressureDiagnostics(
                    model_rpc_service_pb2.PressureSessionControlRequestPB(
                        session_id=session_id,
                        action=actions[action],
                        phase=phase,
                        recorder_capacity=4096 if action == "enable" else 0,
                        target_ranks=target_ranks,
                    ),
                    timeout=timeout,
                )
                record["response"] = MessageToDict(
                    response,
                    preserving_proto_field_name=True,
                    use_integers_for_enums=True,
                )
                errors = []
                seen = set()
                if response.schema_version != 1:
                    errors.append("unsupported control schema")
                for ack in response.ranks:
                    rank = ack.world_rank
                    if rank in seen or rank not in target_ranks:
                        errors.append("duplicate or unexpected control rank")
                    seen.add(rank)
                    if (
                        ack.error
                        or not ack.service_instance
                        or ack.applied_monotonic_ns <= 0
                    ):
                        errors.append("rank acknowledgement failed: " + str(rank))
                    elif rank in instances and instances[rank] != ack.service_instance:
                        errors.append("rank instance changed: " + str(rank))
                    elif rank in target_ranks:
                        instances[rank] = ack.service_instance
                if seen != set(target_ranks):
                    errors.append("missing control ranks")
                record.update(success=not errors, errors=errors)
            except Exception as exc:
                record.update(success=False, errors=[repr(exc)])
            record["client_receive_s"] = time.monotonic()
            return record

        def get_tiered_planner():
            nonlocal tiered_planner
            if tiered_planner is None:
                effective_seed_count = freeze_seed_families(self._continuous_seed_ids)
                tiered_planner = TieredRollingPlanner(
                    effective_seed_count=effective_seed_count,
                    cohort_size=(
                        config.cohort_size or self.pressure_client_concurrency
                    ),
                    memory_reuse_ratio=config.memory_reuse_ratio,
                    case_indices=tuple(range(len(cases))),
                )
            return tiered_planner

        def make_payload(kind, job_id, formal_sequence):
            nonlocal seed_index, rewarm_index
            nonce = f"{session_id}_{job_id}"
            with producer_lock:
                if kind in ("seed", "fill"):
                    tokens = (
                        seed_tokens if kind == "seed" else seed_plan.fill_input_tokens
                    )
                    with self._continuous_prompt_lock:
                        _, ids = self.factory._exact_text_and_ids(
                            tokens, f"diagnostic_{kind}_{nonce}_"
                        )
                    payload = dict(input_ids=ids, nonce=nonce)
                    if kind == "seed":
                        family = seed_index
                        seed_index += 1
                        self._continuous_seed_ids[family] = ids
                        payload["family"] = family
                    return payload
                if kind == "rewarm":
                    if config.family_schedule != "tiered_rolling":
                        raise ValueError("rewarm requires tiered rolling")
                    assignments = get_tiered_planner().initial_device_assignments()
                    if rewarm_index >= len(assignments):
                        raise ValueError("rewarm assignment overflow")
                    assignment = assignments[rewarm_index]
                    rewarm_index += 1
                    index = assignment.cohort_lane
                elif formal_sequence is None:
                    raise ValueError("formal payload requires a supervisor sequence")
                else:
                    index = formal_sequence
                    assignment = (
                        get_tiered_planner().assignment(index)
                        if config.family_schedule == "tiered_rolling"
                        else None
                    )
                if assignment is not None:
                    case_index = assignment.case_index
                    family = assignment.family
                else:
                    if index % len(cases) == 0:
                        rng.shuffle(order)
                    case_index = order[index % len(cases)]
                    family = index % seed_count if seed_count else 0
                case = cases[case_index]
                ids = self._continuous_request_ids(case, family, nonce)
                payload = dict(
                    input_ids=ids,
                    nonce=nonce,
                    family=family,
                    epoch=(
                        assignment.cohort_id
                        if assignment is not None
                        else index // len(cases)
                    ),
                    case_index=case_index,
                    requested_cache_len=int(case["cache_len"]),
                )
                if assignment is not None:
                    payload.update(
                        cohort_id=assignment.cohort_id,
                        cohort_lane=assignment.cohort_lane,
                        expected_cache_tier=assignment.expected_cache_tier,
                        promoted_from_cohort=assignment.promoted_from_cohort,
                        effective_seed_count=tiered_planner.effective_seed_count,
                    )
                if kind == "rewarm":
                    payload["rewarm"] = True
                return payload

        def request(payload, register, before):
            return _post_prefill_ids(
                self._grpc_stub,
                payload["input_ids"],
                self.request_timeout,
                "pending",
                register_call=register,
                before_dispatch=lambda: before(payload),
                resolve_mock_memory_cache_request=lambda: payload.get(
                    "mock_memory_cache_request"
                ),
            )

        def poll(timeout):
            sent = time.monotonic()
            response = pressure_stub.GetCachePressure(
                model_rpc_service_pb2.CachePressureRequestPB(
                    session_id=session_id,
                    target_ranks=target_ranks,
                    after_step_sequence=cursors,
                    max_step_records=256,
                ),
                timeout=timeout,
            )
            received = time.monotonic()
            raw = dict(
                client_send_s=sent,
                client_receive_s=received,
                response=MessageToDict(
                    response,
                    preserving_proto_field_name=True,
                    use_integers_for_enums=True,
                ),
            )
            try:
                raw = raw_pressure_record(response, sent, received)
                observed_ranks = [rank.world_rank for rank in response.ranks]
                if response.schema_version != 1:
                    raise ValueError("unsupported pressure schema")
                if (
                    len(response.requested_ranks) != len(target_ranks)
                    or set(response.requested_ranks) != set(target_ranks)
                    or len(observed_ranks) != len(set(observed_ranks))
                    or not set(observed_ranks).issubset(target_ranks)
                ):
                    raise ValueError("pressure rank protocol mismatch")
                payload_ranks = [
                    rank
                    for rank in response.ranks
                    if not rank.error
                    or any(
                        field.name
                        not in {
                            "world_rank",
                            "service_instance",
                            "error",
                            "coordinator_send_ns",
                            "coordinator_receive_ns",
                        }
                        for field, _ in rank.ListFields()
                    )
                ]
                payload_rank_ids = {rank.world_rank for rank in payload_ranks}
                if any(
                    rank.world_rank not in instances
                    or (
                        rank.service_instance
                        and instances[rank.world_rank] != rank.service_instance
                    )
                    or (
                        rank.world_rank in payload_rank_ids
                        and (
                            not rank.service_instance or not rank.step_history_available
                        )
                    )
                    or any(step.session_id != session_id for step in rank.steps)
                    for rank in response.ranks
                ):
                    raise ValueError("rank instance or diagnostic history mismatch")
                if len(observed_ranks) != len(target_ranks) or any(
                    rank.error for rank in response.ranks
                ):
                    concrete_samples = ()
                    if payload_ranks:
                        # Preserve concrete evidence for frozen inventory/reset
                        # checks, without asserting a complete shared scope.
                        available = model_rpc_service_pb2.CachePressureResponsePB()
                        available.CopyFrom(response)
                        available.ClearField("ranks")
                        available.ClearField("requested_ranks")
                        for rank in payload_ranks:
                            copied = available.ranks.add()
                            copied.CopyFrom(rank)
                            copied.ClearField("error")
                            available.requested_ranks.append(rank.world_rank)
                        concrete_samples = decode_pressure_response(
                            available,
                            sent,
                            received,
                            tuple(available.requested_ranks),
                            complete=False,
                        ).samples
                    raw["sampling_error"] = "missing rank or failed rank sample"
                    # A partial page provides no complete coverage or cursor boundary.
                    return PressurePoll(
                        concrete_samples,
                        raw,
                        step_history_complete=False,
                        sample_complete=False,
                    )
                decoded = decode_pressure_response(
                    response, sent, received, tuple(target_ranks)
                )
            except Exception as exc:
                raw["diagnostics_error"] = repr(exc)
                return PressurePoll((), raw)
            samples = {s.rank: s for s in decoded.samples}
            activity = []
            for rank in response.ranks:
                sample = samples[rank.world_rank]
                low = sample.offset_bounds[0]
                active = [
                    step.wall_start_ns / 1e9 + low
                    for step in rank.steps
                    if not step.completed
                ]
                progress = [
                    step.wall_end_ns / 1e9 + low
                    for step in rank.steps
                    if step.completed
                ]
                a = rank.activity
                activity.append(
                    ActivitySample(
                        0,
                        max(progress, default=0),
                        max(active, default=None),
                        (
                            a.running_requests.value
                            if a.HasField("running_requests")
                            else None
                        ),
                        (
                            a.waiting_requests.value
                            if a.HasField("waiting_requests")
                            else None
                        ),
                    )
                )
                cursors[rank.world_rank] = rank.next_step_sequence
            return PressurePoll(
                decoded.samples,
                raw,
                tuple(activity),
                any(
                    sum(step.completed for step in rank.steps) >= 256
                    for rank in response.ranks
                ),
            )

        manifest = {key: None for key in MANIFEST_PROJECTION}
        manifest.update(evidence_manifest or {})
        manifest["seed_plan"] = asdict(seed_plan)
        manifest["missing_metadata"] = [
            key for key in MANIFEST_PROJECTION if manifest[key] is None
        ]
        writer = PressureWriter(
            getattr(self, "_pressure_epoch_dir", self.result_dir), manifest
        )
        evidence = PressureEvidence(
            writer,
            sink,
            profile_requested=controller is not None,
            target_ranks=target_ranks,
            session_id=session_id,
        )
        session = PressureSession(
            config,
            session_id=session_id,
            limits=limits,
            concurrency=self.pressure_client_concurrency,
            scheduler=self.pressure_scheduler,
            batch_size=self.pressure_batch_size,
            max_input_tokens=max(int(c["input_len"]) for c in cases),
            seed_plan=seed_plan,
            startup_targets=startup_targets,
            make_payload=make_payload,
            request=request,
            poll=poll,
            sink=evidence,
            set_batch=self._set_prefill_batch_size,
            controller=controller,
            diagnostics_control=diagnostics_control,
            rate=self.pressure_request_rate,
            poisson=self.pressure_arrival == "poisson",
            random_seed=self.pressure_random_seed,
        )
        try:
            return session.run()
        except BaseException as exc:
            evidence.abort(repr(exc))
            raise
        finally:
            writer.close()

    def _continuous_metric(self, case, runs):
        total_len = int(case["input_len"])
        cache_len = int(case["cache_len"])
        successful = [result for result in runs if result.get("success")]
        reuse_values = [int(result.get("reuse_len", -1)) for result in successful]
        reuse_exact = bool(
            len(successful) == self.measure_runs
            and all(value == cache_len for value in reuse_values)
        )
        shape_exact = bool(
            len(successful) == self.measure_runs
            and all(
                int(result.get("input_len", -1)) == total_len for result in successful
            )
            and all(int(result.get("output_len", -1)) == 1 for result in successful)
        )
        prefill_values = [
            float(result["prefill_time_ms"])
            for result in successful
            if run_prefill_rt(result) is not None
        ]
        timing_valid = len(prefill_values) == self.measure_runs
        validation_status = measurement_status(
            len(successful) == self.measure_runs,
            shape_exact,
            reuse_exact,
            timing_valid,
        )
        status = measurement_status(
            len(successful) == self.measure_runs,
            shape_exact,
            reuse_exact,
            timing_valid,
            skip_reuse=self.skip_reuse_validation,
        )
        offsets = [float(result["pressure_completed_offset_ms"]) for result in runs]
        return {
            "case_key": self.case_key(case),
            "case_id": int(case["case_id"]),
            "batch_size": 1,
            "input_len": total_len,
            "input_len_built": total_len,
            "cache_len_requested": cache_len,
            "cache_len_observed": reuse_values,
            "input_len_observed": [
                int(result.get("input_len", 0)) for result in successful
            ],
            "expected_reuse_len": cache_len,
            "success_runs": len(successful),
            "measure_runs": self.measure_runs,
            "cache_commit_tail_tokens": self.cache_commit_tail_tokens,
            "seed": {},
            "seed_mode": "continuous_shared_prefix_v1",
            "execution_mode": "continuous_pressure_v1",
            "pressure_scheduler": self.pressure_scheduler,
            "pressure_batch_size": (
                self.pressure_batch_size
                if self.pressure_scheduler == "batch_decode"
                else None
            ),
            "pressure_client_concurrency": self.pressure_client_concurrency,
            "runs": runs,
            "reuse_exact": reuse_exact,
            "reuse_validation_skipped": self.skip_reuse_validation,
            "validation_status": validation_status,
            "shape_exact": shape_exact,
            "timing_valid": timing_valid,
            "prefill_time_ms": prefill_values,
            "median_prefill_time_ms": (
                statistics.median(prefill_values) if timing_valid else None
            ),
            "avg_prefill_time_ms": (
                statistics.fmean(prefill_values) if timing_valid else None
            ),
            "elapsed_s": (max(offsets) - min(offsets)) / 1000.0 if offsets else 0.0,
            "status": status,
        }

    def _run_continuous(self, pending) -> List[Dict[str, Any]]:
        rng = random.Random(self.pressure_random_seed)
        try:
            self._initialize_continuous_seeds(pending)
            jobs = []
            for case in pending:
                for run_index in range(self.measure_runs):
                    jobs.append(
                        {
                            "case": case,
                            "run_index": run_index,
                            "padding": False,
                        }
                    )
            rng.shuffle(jobs)
            padding_count = (
                (-len(jobs)) % self.pressure_batch_size
                if self.pressure_scheduler == "batch_decode"
                else 0
            )
            for index in range(padding_count):
                padding = dict(jobs[index % len(jobs)])
                padding["padding"] = True
                jobs.append(padding)
            for sequence, job in enumerate(jobs):
                job["sequence"] = sequence
                job["nonce"] = uuid.uuid4().hex
                if int(job["case"]["cache_len"]):
                    job["family"] = sequence % len(self._continuous_seed_ids)
                else:
                    job["family"] = None

            offsets = []
            offset = 0.0
            for index in range(len(jobs)):
                if index and self.pressure_request_rate:
                    if self.pressure_arrival == "poisson":
                        offset += rng.expovariate(self.pressure_request_rate)
                    else:
                        offset += 1.0 / self.pressure_request_rate
                offsets.append(offset)

            if self.pressure_scheduler == "batch_decode":
                self._set_prefill_batch_size(self.pressure_batch_size)
            pressure_started_wall = time.time()
            pressure_started = time.perf_counter()

            def send(job, scheduled_offset):
                case = job["case"]
                key = self.case_key(case)
                request_id = (
                    f"{key}:run{job['run_index']}:pressure{job['sequence']}:"
                    f"family{job['family']}"
                )
                try:
                    ids = self._continuous_request_ids(
                        case, job["family"], job["nonce"]
                    )
                    due = pressure_started + scheduled_offset
                    remaining = due - time.perf_counter()
                    if remaining > 0:
                        time.sleep(remaining)
                    dispatched = time.perf_counter()
                    result = self._post_request("", ids, request_id)
                except Exception as exc:
                    dispatched = time.perf_counter()
                    result = {
                        "success": False,
                        "error": repr(exc),
                        "request_id": request_id,
                    }
                completed = time.perf_counter()
                result = dict(result)
                result.update(
                    {
                        "run_index": job["run_index"],
                        "seed_family": job["family"],
                        "pressure_sequence": job["sequence"],
                        "pressure_padding": job["padding"],
                        "pressure_scheduled_offset_ms": scheduled_offset * 1000.0,
                        "pressure_dispatch_offset_ms": (dispatched - pressure_started)
                        * 1000.0,
                        "pressure_schedule_lag_ms": max(
                            0.0,
                            (dispatched - pressure_started - scheduled_offset) * 1000.0,
                        ),
                        "pressure_completed_offset_ms": (completed - pressure_started)
                        * 1000.0,
                    }
                )
                return result

            with ThreadPoolExecutor(
                max_workers=self.pressure_client_concurrency,
                thread_name_prefix="continuous-pressure",
            ) as pool:
                futures = [
                    pool.submit(send, job, offsets[index])
                    for index, job in enumerate(jobs)
                ]
                results = [future.result() for future in futures]
        finally:
            if self.pressure_scheduler == "batch_decode":
                self._set_prefill_batch_size(1)

        by_key = {self.case_key(case): [] for case in pending}
        for job, result in zip(jobs, results):
            if not job["padding"]:
                by_key[self.case_key(job["case"])].append(result)
        elapsed = time.perf_counter() - pressure_started
        schedule_lags = [
            float(result["pressure_schedule_lag_ms"])
            for job, result in zip(jobs, results)
            if not job["padding"]
        ]
        server_waits = [
            float(result["wait_time_ms"])
            for job, result in zip(jobs, results)
            if not job["padding"]
            and result.get("success")
            and isinstance(result.get("wait_time_ms"), (int, float))
        ]
        padding_failures = sum(
            not result.get("success")
            for job, result in zip(jobs, results)
            if job["padding"]
        )
        self._continuous_summary = {
            "schema_version": 1,
            "started_at_epoch_s": pressure_started_wall,
            "execution_mode": "continuous_pressure_v1",
            "scheduler": self.pressure_scheduler,
            "arrival": self.pressure_arrival,
            "request_rate": self.pressure_request_rate,
            "random_seed": self.pressure_random_seed,
            "scheduler_batch_size": (
                self.pressure_batch_size
                if self.pressure_scheduler == "batch_decode"
                else None
            ),
            "client_concurrency": self.pressure_client_concurrency,
            "configured_seed_count": self.pressure_seed_count,
            "effective_seed_count": len(self._continuous_seed_ids),
            "measured_requests": len(jobs) - padding_count,
            "padding_requests": padding_count,
            "padding_failures": padding_failures,
            "elapsed_s": elapsed,
            "achieved_request_rate": (len(jobs) / elapsed if elapsed else None),
            "median_schedule_lag_ms": (
                statistics.median(schedule_lags) if schedule_lags else None
            ),
            "max_schedule_lag_ms": max(schedule_lags) if schedule_lags else None,
            "queued_requests": sum(wait > 0 for wait in server_waits),
            "median_server_wait_time_ms": (
                statistics.median(server_waits) if server_waits else None
            ),
            "max_server_wait_time_ms": max(server_waits) if server_waits else None,
        }
        metrics = []
        for case in pending:
            key = self.case_key(case)
            runs = sorted(by_key[key], key=lambda result: result["run_index"])
            metric = self._continuous_metric(case, runs)
            self._results[key] = metric
            self._append_journal(metric)
            self._write_progress_checkpoint()
            metrics.append(metric)
        complete = all(
            self._results.get(self.case_key(case), {}).get("status") == "ok"
            for case in self.cases
        )
        self._save(complete=complete)
        failures = [metric for metric in metrics if metric["status"] != "ok"]
        if padding_failures:
            raise RuntimeError(
                f"continuous cache pressure had {padding_failures} failed padding requests"
            )
        if failures and self.fail_fast:
            raise RuntimeError(
                "continuous cache pressure completed with invalid cases: "
                + ", ".join(metric["case_key"] for metric in failures)
            )
        return list(self._results.values())

    def _prepare_case_payload(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Build prompts for one case without issuing a model request."""
        if is_grouped_case(case):
            return self._prepare_grouped_payload(case)
        if self.shared_seed and int(case["cache_len"]) > 0:
            return self._prepare_shared_payload(case)
        key = self.case_key(case)
        total_len = int(case["input_len"])
        cache_len = int(case["cache_len"])
        batch_size = int(case.get("batch_size", 1))
        if batch_size != 1:
            raise ValueError("prefix cache grid currently requires batch_size=1")
        if cache_len and cache_len % self.cache_commit_tail_tokens:
            raise ValueError(
                f"cache_len must align to commit tail "
                f"{self.cache_commit_tail_tokens}, got {cache_len}"
            )
        if cache_len and cache_len + self.cache_commit_tail_tokens > total_len:
            raise ValueError(
                f"cache_len must leave at least {self.cache_commit_tail_tokens} "
                f"tokens for seed commit, got {cache_len}/{total_len}"
            )
        prompts = self._build_prompts(case, total_len, cache_len)
        prefix = prompts.seed_text
        built_len = prompts.built_len
        if self.request_transport == "dashsc_input_ids" and not prompts.run_ids:
            prompts.run_ids = [
                _encode(self.factory.tokenizer, text) for text in prompts.run_texts
            ]
        run_ids = prompts.run_ids or [None] * len(prompts.run_texts)
        if cache_len:
            seed, seed_ids = self.factory.make_seed_and_ids(
                int(case["case_id"]),
                prefix,
                cache_len,
                self.cache_commit_tail_tokens,
            )
        else:
            seed, seed_ids = "", []
        return {
            "key": key,
            "total_len": total_len,
            "cache_len": cache_len,
            "batch_size": batch_size,
            "built_len": built_len,
            "seed": seed,
            "seed_ids": seed_ids,
            "run_targets": prompts.run_texts,
            "run_ids": run_ids,
        }

    def _prepare_grouped_payload(self, case):
        case = normalize_cache_case(case)
        for group in case["request_groups"]:
            cache_len = group["cache_len"]
            if cache_len and (
                cache_len % self.cache_commit_tail_tokens
                or cache_len + self.cache_commit_tail_tokens > group["input_len"]
                or (self.expected_block_size and cache_len % self.expected_block_size)
            ):
                raise ValueError(
                    f"invalid cache alignment or seed commit space: {group}"
                )
        members = (
            self.case_store.load_grouped_case(case)
            if self.case_store is not None
            else build_grouped_prompts(self.factory, case, self.measure_runs)
        )
        seeds = {}
        for member in members:
            prompts = member["prompts"]
            if len(prompts.run_texts) != self.measure_runs:
                raise ValueError("materialized batch measure_runs mismatch")
            if member["cache_len"] and member["prefix_id"] not in seeds:
                seeds[member["prefix_id"]] = self.factory.make_seed_and_ids(
                    member["prefix_id"],
                    prompts.seed_text,
                    member["cache_len"],
                    self.cache_commit_tail_tokens,
                )
        return {"case": case, "members": members, "seeds": seeds}

    def _set_prefill_batch_size(self, batch_size, *, timeout=None):
        """Wait for scheduler acknowledgement before submitting a complete batch."""
        return configure_scheduler(
            self.port,
            batch_size=batch_size,
            mode="prefill",
            post=self._http_session.post,
            timeout=min(
                self.request_timeout, 60, timeout if timeout is not None else 60
            ),
        )

    def _seed_grouped_payload(self, payload):
        seeds = list(payload["seeds"].items())
        if not seeds:
            return []
        self._set_prefill_batch_size(len(seeds))
        barrier = threading.Barrier(len(seeds))
        key = self.case_key(payload["case"])

        def send(seed):
            identity, (text, ids) = seed
            request_id = f"{key}:seed:{identity}"
            with _make_http_session() as session:
                barrier.wait(timeout=self.request_timeout)
                if self.request_transport == "dashsc_input_ids":
                    return _post_prefill_ids(
                        self._grpc_stub, ids, self.request_timeout, request_id
                    )
                return _post_prefill(
                    self.port, text, self.request_timeout, request_id, session
                )

        with ThreadPoolExecutor(
            max_workers=len(seeds), thread_name_prefix="cache-seed"
        ) as pool:
            futures = [pool.submit(send, seed) for seed in seeds]
            results = [future.result() for future in futures]
        if not all(result.get("success") for result in results):
            raise RuntimeError(f"batch seed failed: {results}")
        return results

    def _measure_grouped_payload(self, payload, started):
        try:
            return self._measure_fixed_batch(payload, started)
        finally:
            # The following case/probe/profile can contain a single request.
            self._set_prefill_batch_size(1)

    def _measure_fixed_batch(self, payload, started):
        case, members = payload["case"], payload["members"]
        key = self.case_key(case)
        seeds = self._seed_grouped_payload(payload)
        self._set_prefill_batch_size(len(members))

        # Sessions are owned by logical request slots, never shared concurrently.
        sessions = [_make_http_session() for _ in members]
        rounds = []
        try:
            with ThreadPoolExecutor(
                max_workers=len(members), thread_name_prefix="cache-batch"
            ) as pool:
                for run_index in range(self.measure_runs):
                    released = []
                    barrier = threading.Barrier(
                        len(members),
                        action=lambda: released.append(time.perf_counter()),
                    )

                    def send(member):
                        slot = member["request_index"]
                        prompts = member["prompts"]
                        request_id = f"{key}:run{run_index}:request{slot}"
                        try:
                            barrier.wait(timeout=self.request_timeout)
                            if self.request_transport == "dashsc_input_ids":
                                result = _post_prefill_ids(
                                    self._grpc_stub,
                                    prompts.run_ids[run_index],
                                    self.request_timeout,
                                    request_id,
                                )
                            else:
                                result = _post_prefill(
                                    self.port,
                                    prompts.run_texts[run_index],
                                    self.request_timeout,
                                    request_id,
                                    sessions[slot],
                                )
                        except Exception as exc:
                            result = {
                                "success": False,
                                "error": repr(exc),
                                "request_id": request_id,
                            }
                        completed = time.perf_counter()
                        result = dict(result)
                        result.update(
                            {
                                k: member[k]
                                for k in ("request_index", "group_index", "prefix_id")
                            }
                        )
                        result["expected_input_len"] = member["input_len"]
                        result["expected_reuse_len"] = member["cache_len"]
                        result["shape_exact"] = bool(
                            result.get("success")
                            and result.get("input_len") == member["input_len"]
                            and result.get("output_len") == 1
                        )
                        result["reuse_exact"] = bool(
                            result.get("success")
                            and result.get("reuse_len") == member["cache_len"]
                        )
                        result["timing_valid"] = bool(
                            result.get("success") and run_prefill_rt(result) is not None
                        )
                        return result, completed

                    futures = [pool.submit(send, member) for member in members]
                    completed = [future.result() for future in futures]
                    results = [result for result, _ in completed]
                    wall_ms = (
                        (max(t for _, t in completed) - released[0]) * 1000
                        if released
                        else None
                    )
                    valid = all(
                        r["shape_exact"]
                        and (r["reuse_exact"] or self.skip_reuse_validation)
                        and r["timing_valid"]
                        for r in results
                    )
                    rounds.append(
                        {
                            "run_index": run_index,
                            "requests": results,
                            "batch_wall_time_ms": wall_ms,
                            "valid": valid,
                            "completed_requests": sum(
                                bool(r.get("success")) for r in results
                            ),
                            "total_input_tokens": sum(
                                r.get("input_len", 0)
                                for r in results
                                if r.get("success")
                            ),
                            "new_prefill_tokens": sum(
                                r.get("input_len", 0) - r.get("reuse_len", 0)
                                for r in results
                                if r.get("success")
                            ),
                        }
                    )
                    if not all(r.get("success") for r in results):
                        # A missing request can leave a fixed-size scheduler waiting.
                        # Never mix the remaining requests with a subsequent round.
                        break
        finally:
            for session in sessions:
                session.close()
        results = [r for run in rounds for r in run["requests"]]
        shape = all(r["shape_exact"] for r in results)
        reuse = all(r["reuse_exact"] for r in results)
        timing = all(r["timing_valid"] for r in results)
        success = all(r.get("success") for r in results)
        validation_status = measurement_status(success, shape, reuse, timing)
        status = measurement_status(
            success, shape, reuse, timing, skip_reuse=self.skip_reuse_validation
        )
        return {
            **case,
            "case_key": key,
            "execution_mode": "scheduler_fixed_batch",
            "seed_batch_size": len(seeds),
            "scheduler_batch_size": len(members),
            "requested_batch_size": len(members),
            "measure_runs": self.measure_runs,
            "seed": seeds,
            "runs": rounds,
            "status": status,
            "validation_status": validation_status,
            "reuse_validation_skipped": self.skip_reuse_validation,
            "success_runs": sum(r["valid"] for r in rounds),
            "shape_exact": shape,
            "reuse_exact": reuse,
            "timing_valid": timing,
            "cache_len_observed": [r.get("reuse_len") for r in results],
            "median_batch_first_token_cost_time_ms": (
                statistics.median(
                    max(r["prefill_time_ms"] for r in round_["requests"])
                    for round_ in rounds
                )
                if status == "ok"
                else None
            ),
            "median_batch_wall_time_ms": (
                statistics.median(r["batch_wall_time_ms"] for r in rounds)
                if status == "ok"
                else None
            ),
            "elapsed_s": time.time() - started,
        }

    def _wait_profile_traces(self, trace_name: str) -> List[str]:
        """Wait for valid JSON from every TP rank; files are saved asynchronously."""
        deadline = time.monotonic() + self.profile_trace_timeout
        complete: Dict[int, Path] = {}
        while True:
            for rank in range(self.profile_tp_size):
                if rank in complete:
                    continue
                for path in self.result_dir.glob(f"{trace_name}_wr{rank}_*.json"):
                    try:
                        with path.open() as stream:
                            trace = json.load(stream)
                        if isinstance(trace, dict) and trace.get("traceEvents"):
                            complete[rank] = path
                            break
                    except (OSError, ValueError):
                        pass  # The background saver may still be writing.
            if len(complete) == self.profile_tp_size:
                target = self.result_dir / "timelines"
                target.mkdir(exist_ok=True)
                paths = []
                for rank in sorted(complete):
                    path = complete[rank]
                    destination = target / path.name
                    path.rename(destination)
                    paths.append(str(destination.relative_to(self.result_dir)))
                return paths
            if time.monotonic() >= deadline:
                missing = sorted(set(range(self.profile_tp_size)) - complete.keys())
                raise RuntimeError(
                    f"profile trace timeout for {trace_name}; missing TP ranks {missing}"
                )
            time.sleep(0.5)

    def _nsys_command(self, record, action, *options):
        command = [self.nsys_path, action]
        if action != "export":
            command.append("--session=" + self.nsys_session)
        command.extend(options)
        entry = {"command": command, "started_at": time.time()}
        record.setdefault("nsys_commands", []).append(entry)
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.profile_trace_timeout,
                check=False,
            )
            entry.update(
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
            if result.returncode:
                raise RuntimeError(
                    f"nsys {action} failed ({result.returncode}): "
                    f"{result.stderr or result.stdout}"
                )
        except BaseException as exc:
            entry["error"] = repr(exc)
            raise
        finally:
            entry["finished_at"] = time.time()

    def _run_nsys_request(self, record, prompts, directory):
        output = (directory / "nsys" / record["trace_name"]).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        report = output.with_suffix(".nsys-rep")
        if report.exists():
            raise FileExistsError(report)
        record["capture_scope"] = (
            "target request plus tail; not an all-rank GPU barrier"
        )
        record["nsys_tail_seconds"] = self.nsys_tail_seconds
        record["nsys_report_expected"] = str(report)
        # A start timeout can occur after activation. Attempt stop even on that
        # path, but never send the target unless start completed successfully.
        try:
            self._nsys_command(
                record,
                "start",
                "--sample=process-tree",
                "--cpuctxsw=process-tree",
                "--output=" + str(output),
            )
            record["result"] = self._post_request(
                prompts.run_texts[0],
                prompts.run_ids[0],
                record["trace_name"] + ":profile",
            )
            if self.nsys_tail_seconds:
                time.sleep(self.nsys_tail_seconds)
        except BaseException:
            try:
                self._nsys_command(record, "stop")
            except BaseException as stop_error:
                record["nsys_stop_error"] = repr(stop_error)
            raise
        else:
            self._nsys_command(record, "stop")
        # Stop/export completes before the caller tears down the model service.
        if not report.is_file() or report.stat().st_size == 0:
            raise RuntimeError(f"nsys stop did not produce a nonempty report: {report}")
        record["trace_files"] = [str(report.relative_to(self.result_dir.resolve()))]
        return record["result"]

    def _run_kineto_request(self, record, prompts, trace_name):
        # Await the all-TP acknowledgement before submitting the target.
        response = self._http_session.post(
            f"http://127.0.0.1:{self.port}/start_profile",
            json={
                "gen_timeline": True,
                "trace_name": trace_name,
                "start_step": 0,
                "num_steps": 1,
                "enable_all_rank": True,
            },
            timeout=60,
        )
        response.raise_for_status()
        acknowledgement = response.json()
        if acknowledgement.get("status") != "ok" or acknowledgement.get("error"):
            raise RuntimeError(f"start_profile failed: {acknowledgement}")
        record["profile_request"] = acknowledgement
        result = self._post_request(
            prompts.run_texts[0], prompts.run_ids[0], f"{trace_name}:profile"
        )
        record["result"] = result
        if not result.get("success"):
            raise RuntimeError(f"profile request failed: {result}")
        record["trace_files"] = self._wait_profile_traces(trace_name)
        return result

    def _run_profiles(self) -> List[Dict[str, Any]]:
        """Replay selected geometries after measurements, never adding to runs[]."""
        if not self.profile_runs:
            return []
        session_id = uuid.uuid4().hex
        directory = (
            self.result_dir
            if self.profile_flat_output
            else self.result_dir / "cache_profiles" / session_id
        )
        directory.mkdir(parents=True, exist_ok=self.profile_flat_output)
        manifest_path = directory / "manifest.json"
        records: List[Dict[str, Any]] = []
        manifest = {
            "schema_version": 1,
            "session_id": session_id,
            "diagnostic_only": True,
            "profile_backend": self.profile_backend,
            "nsys_session": (
                self.nsys_session if self.profile_backend == "nsys" else None
            ),
            "profile_runs": self.profile_runs,
            "profile_case_ids": sorted(self.profile_case_ids),
            "profile_tp_size": self.profile_tp_size,
            "request_transport": self.request_transport,
            "grid_sha256": self.grid_sha256,
            "run_config": self.run_config,
            "status": "running",
            "records": records,
        }

        def save():
            temporary = manifest_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            temporary.replace(manifest_path)

        # Reserve the manifest before arming the profiler; never replace an old session.
        with manifest_path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
        try:
            for case in self.cases:
                if int(case["case_id"]) not in self.profile_case_ids:
                    continue
                key = self.case_key(case)
                total_len, cache_len = int(case["input_len"]), int(case["cache_len"])
                for index in range(self.profile_runs):
                    # New identity for EACH replay, including cold requests. Replaying
                    # formal prompts would accidentally hit their previously cached tails.
                    prompt_case_id = uuid.uuid4().int >> 64
                    prompts = self.factory.build_case_prompts(
                        prompt_case_id, total_len, cache_len, 1
                    )
                    trace_name = profile_trace_name(session_id, key, index)
                    record = {
                        "case_id": int(case["case_id"]),
                        "case_key": key,
                        "profile_run": index,
                        "prompt_case_id": prompt_case_id,
                        "input_len": total_len,
                        "cache_len_requested": cache_len,
                        "trace_name": trace_name,
                        "status": "running",
                    }
                    records.append(record)
                    save()
                    if cache_len:
                        seed, seed_ids = self.factory.make_seed_and_ids(
                            prompt_case_id,
                            prompts.seed_text,
                            cache_len,
                            self.cache_commit_tail_tokens,
                        )
                        record["seed"] = self._post_request(
                            seed, seed_ids, f"{trace_name}:seed"
                        )
                        if not record["seed"].get("success"):
                            raise RuntimeError(f"profile seed failed: {record['seed']}")
                    if self.profile_backend == "nsys":
                        result = self._run_nsys_request(record, prompts, directory)
                    else:
                        result = self._run_kineto_request(record, prompts, trace_name)
                    record["result"] = result
                    if not result.get("success"):
                        raise RuntimeError(f"profile request failed: {result}")
                    if (
                        result.get("input_len") != total_len
                        or result.get("reuse_len") != cache_len
                        or result.get("output_len") != 1
                    ):
                        raise RuntimeError(f"profile shape/reuse mismatch: {result}")
                    record["status"] = "ok"
                    save()
                    logging.info(
                        "[CACHE_PROFILE] %s run=%d traces=%s",
                        key,
                        index,
                        record["trace_files"],
                    )
            manifest["status"] = "completed"
            save()
            return records
        except BaseException as exc:
            manifest["status"] = "failed"
            manifest["error"] = repr(exc)
            if records and records[-1]["status"] == "running":
                records[-1]["status"] = "failed"
                records[-1]["error"] = repr(exc)
            save()
            # Abort rather than let a partially armed profiler affect later requests.
            raise

    def _run_full_pressure_public(self):
        epoch_id = uuid.uuid4().hex
        self._pressure_epoch_dir = self.result_dir / "pressure_epochs" / epoch_id
        self._pressure_epoch_dir.mkdir(parents=True, exist_ok=False)
        self._write_json_atomic(
            self.result_dir / "pressure_latest.json",
            dict(epoch_id=epoch_id, directory=str(self._pressure_epoch_dir)),
        )
        self._pressure_preflight_complete = False
        try:
            return self._run_full_pressure_epoch(epoch_id)
        except BaseException as exc:
            if not self._pressure_preflight_complete:
                self._write_json_atomic(
                    self._pressure_epoch_dir / "preflight_failure.json",
                    dict(epoch_id=epoch_id, error=repr(exc)),
                )
            raise

    def _run_full_pressure_epoch(self, epoch_id):
        from dataclasses import asdict, replace

        import grpc

        from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2 as pb
        from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2_grpc as rpc

        from .continuous_pressure import ExecutionLimits
        from .pressure_diagnostics import (
            decode_pressure_response,
            raw_pressure_record,
            select_targets,
        )
        from .pressure_seed_plan import build_seed_plan

        startup = self.run_config["full_pressure_startup"]
        ranks = tuple(startup["target_ranks"])
        # A root checkpoint guards the workload only; each epoch reruns every phase.
        self._pressure_seed_plan = None
        self._continuous_seed_ids = []
        self._results.clear()
        self._save(complete=False)
        with grpc.insecure_channel(f"127.0.0.1:{self.port + 1}") as channel:
            stub = rpc.RpcServiceStub(channel)
            retry_deadline = time.monotonic() + min(
                _STARTUP_PRESSURE_RETRY_BUDGET_S,
                self.pressure_config.fill_timeout_s,
            )
            startup_attempts = []
            while True:
                sent = time.perf_counter()
                response = stub.GetCachePressure(
                    pb.CachePressureRequestPB(target_ranks=ranks),
                    timeout=min(
                        self.request_timeout,
                        self.pressure_config.fill_timeout_s,
                        max(0.1, retry_deadline - time.monotonic()),
                    ),
                )
                received = time.perf_counter()
                raw = raw_pressure_record(response, sent, received)
                startup_attempts.append(raw)
                (self._pressure_epoch_dir / "startup_pressure.json").write_text(
                    json.dumps(raw, allow_nan=False, indent=2) + "\n"
                )
                (
                    self._pressure_epoch_dir / "startup_pressure_attempts.json"
                ).write_text(
                    json.dumps(startup_attempts, allow_nan=False, indent=2) + "\n"
                )
                try:
                    decoded = decode_pressure_response(response, sent, received, ranks)
                    break
                except ValueError as error:
                    remaining = retry_deadline - time.monotonic()
                    if remaining <= 0 or not _is_transient_pressure_rank_failure(
                        response, error
                    ):
                        raise
                    logging.warning(
                        "startup pressure snapshot has transient rank errors; "
                        "retrying attempt=%d remaining_s=%.3f",
                        len(startup_attempts),
                        remaining,
                    )
                    time.sleep(min(_STARTUP_PRESSURE_RETRY_DELAY_S, remaining))
            targets = select_targets(decoded.samples, ranks)
            seed_plan = build_seed_plan(
                decoded.samples,
                targets,
                cases=tuple(self.cases),
                cache_alignment=self.grid_metadata["generator"]["cache_alignment"],
                commit_tail_tokens=self.cache_commit_tail_tokens,
                client_inflight=self.pressure_client_concurrency,
                manual_seed_count=self.pressure_seed_count,
                fill_target_tokens=self.pressure_config.fill_input_tokens,
                test_block_num=startup.get("test_block_num", 0),
                explicit_smoke=(
                    (self.profile or {})
                    .get("cache_grid", {})
                    .get("explicit_smoke", False)
                    or self.grid_metadata.get("kind") == "explicit_cache_grid_smoke"
                ),
                diagnostic_only=(
                    (self.profile or {})
                    .get("cache_grid", {})
                    .get("diagnostic_only", False)
                ),
            )
            self._pressure_seed_plan = seed_plan
            effective_config = replace(
                self.pressure_config, fill_input_tokens=seed_plan.fill_input_tokens
            )
            owner_kv_budgets = tuple(
                (
                    f"owner={group.owner_rank} group={group.owner_group_id} "
                    f"pool={pool.pool_id}",
                    pool.available_blocks,
                    pool.block_tokens,
                )
                for group in seed_plan.groups
                for pool in group.device_pools
            )
            limits = ExecutionLimits(
                model_tokens=startup["model_tokens"],
                admission_requests=startup["admission_requests"],
                batch_tokens=startup["batch_tokens"],
                available_kv_blocks=min(pool[1] for pool in owner_kv_budgets),
                block_tokens=startup["block_tokens"],
                owner_kv_budgets=owner_kv_budgets,
            )
            shapes = [
                ("fill", seed_plan.fill_input_tokens),
                ("seed", seed_plan.seed_input_tokens),
                *[
                    (f"formal case_id={case['case_id']}", int(case["input_len"]))
                    for case in self.cases
                ],
                (
                    "reuse probe seed",
                    self.expected_block_size + self.cache_commit_tail_tokens,
                ),
                ("reuse probe hit", 2 * self.expected_block_size),
            ]
            for shape, tokens in shapes:
                try:
                    limits.require_request(tokens)
                except ValueError as exc:
                    raise ValueError(f"{shape} input_tokens={tokens}: {exc}") from exc
            source_root = Path(__file__).parent
            source_files = sorted(source_root.glob("*.py")) + sorted(
                (source_root.parent / "config").glob("*.py")
            )
            source_files += [
                source_root.parent.parent / name
                for name in ("perf_config.py", "batch_decode_test.py")
            ]
            source_hashes = {
                str(path.relative_to(source_root.parent.parent)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in source_files
            }
            profile_hash = (
                self.profile_sha256
                or hashlib.sha256(
                    json.dumps(self.profile or {}, sort_keys=True).encode()
                ).hexdigest()
            )
            grid_hash = (
                self.grid_sha256
                or hashlib.sha256(
                    json.dumps(self.cases, sort_keys=True).encode()
                ).hexdigest()
            )
            manifest = dict(
                code_hash=hashlib.sha256(
                    json.dumps(source_hashes, sort_keys=True).encode()
                ).hexdigest(),
                code_hash_scope="python_pressure_toolchain",
                source_hashes=source_hashes,
                profile_hash=profile_hash,
                grid_hash=grid_hash,
                resolved_settings=dict(
                    self.run_config, pressure=asdict(effective_config)
                ),
                input_identity=dict(
                    grid_hash=grid_hash,
                    tokenizer=getattr(
                        self.factory.tokenizer,
                        "name_or_path",
                        type(self.factory.tokenizer).__name__,
                    ),
                    random_seed=self.pressure_random_seed,
                ),
                epoch={str(sample.rank): sample.instance for sample in decoded.samples},
                clock_bounds={
                    str(sample.rank): sample.offset_bounds for sample in decoded.samples
                },
                control_times=dict(capacity_send_s=sent, capacity_receive_s=received),
                startup_pressure=raw,
                startup_targets=[asdict(target) for target in targets],
                seed_plan=asdict(seed_plan),
                epoch_id=epoch_id,
            )
            logging.info("full pressure epoch evidence: %s", self._pressure_epoch_dir)
            self._probe_reuse_granularity()
            self._pressure_preflight_complete = True
            result = self._run_full_pressure(
                self.cases,
                effective_config,
                limits,
                stub,
                ranks,
                None,
                seed_plan=seed_plan,
                startup_targets=targets,
                evidence_manifest=manifest,
            )
            if result.get("failure"):
                raise RuntimeError(
                    f"full pressure failed: {result['failure']}; see {self._pressure_epoch_dir}"
                )
            return []

    def run(self) -> List[Dict[str, Any]]:
        if self.pressure_config is not None:
            try:
                return self._run_full_pressure_public()
            finally:
                self._close_resources()
        if self.profile_only:
            try:
                return self._run_profiles()
            finally:
                self._close_resources()
        # Establish a small, configuration-guarded base checkpoint before the
        # first model request. Per-case journal records then provide durable
        # one-case resume granularity without rewriting the growing full JSON.
        self._save(complete=False)
        self._probe_reuse_granularity()
        pending = [
            case
            for case in self.cases
            if self._results.get(self.case_key(case), {}).get("status") != "ok"
        ]
        if self.shared_seed:
            pending.sort(key=self._shared_seed_case_order)
            logging.info(
                "cache grid: shared seed execution order=%s",
                [int(case["case_id"]) for case in pending],
            )
        logging.info(
            "cache grid: %d total cases, %d already complete, %d pending",
            len(self.cases),
            len(self.cases) - len(pending),
            len(pending),
        )
        if not pending:
            self._save(complete=True)
            try:
                self._run_profiles()
                return list(self._results.values())
            finally:
                self._close_resources()

        if self.pressure_mode == "continuous":
            try:
                return self._run_continuous(pending)
            except BaseException:
                try:
                    self._save(complete=False)
                except Exception:
                    logging.exception(
                        "cache grid: failed to flush continuous-pressure checkpoint"
                    )
                raise
            finally:
                self._close_resources()

        completed_before = len(self.cases) - len(pending)
        completed_this_attempt = 0
        try:
            if self.shared_seed:
                self._initialize_shared_seed(pending)
            with ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="query-prefetch"
            ) as prep:
                prepared: Future = prep.submit(self._prepare_case_payload, pending[0])
                for idx, case in enumerate(pending, 1):
                    key = self.case_key(case)
                    started = time.time()
                    refresh_start = len(self._shared_seed_refreshes)
                    try:
                        try:
                            payload = prepared.result()
                        finally:
                            if idx < len(pending):
                                prepared = prep.submit(
                                    self._prepare_case_payload, pending[idx]
                                )
                        if "members" in payload:
                            metric = self._measure_grouped_payload(payload, started)
                        else:
                            total_len = payload["total_len"]
                            cache_len = payload["cache_len"]
                            batch_size = payload["batch_size"]
                            built_len = payload["built_len"]
                            seed_result: Dict[str, Any] = {}
                            if cache_len and not self.shared_seed:
                                seed_result = self._post_request(
                                    payload["seed"], payload["seed_ids"], f"{key}:seed"
                                )
                                if not seed_result.get("success"):
                                    raise RuntimeError(
                                        seed_result.get("error", "seed request failed")
                                    )

                            runs = [
                                (
                                    self._post_request(
                                        run_target,
                                        payload["run_ids"][run_idx],
                                        f"{key}:run{run_idx}",
                                    )
                                    if not (self.shared_seed and cache_len)
                                    else self._post_shared_measurement(
                                        case,
                                        payload["run_ids"][run_idx],
                                        f"{key}:run{run_idx}",
                                    )
                                )
                                for run_idx, run_target in enumerate(
                                    payload["run_targets"]
                                )
                            ]

                            successful = [r for r in runs if r.get("success")]
                            expected_reuse = cache_len
                            reuse_values = [
                                int(r.get("reuse_len", -1)) for r in successful
                            ]
                            reuse_exact = bool(
                                len(successful) == self.measure_runs
                                and all(x == expected_reuse for x in reuse_values)
                            )
                            shape_exact = bool(
                                len(successful) == self.measure_runs
                                and all(
                                    int(r.get("input_len", -1)) == total_len
                                    for r in successful
                                )
                                and all(
                                    int(r.get("output_len", -1)) == 1
                                    for r in successful
                                )
                            )
                            prefill_values = [
                                float(r["prefill_time_ms"])
                                for r in successful
                                if run_prefill_rt(r) is not None
                            ]
                            timing_valid = len(prefill_values) == self.measure_runs
                            validation_status = measurement_status(
                                len(successful) == self.measure_runs,
                                shape_exact,
                                reuse_exact,
                                timing_valid,
                            )
                            status = measurement_status(
                                len(successful) == self.measure_runs,
                                shape_exact,
                                reuse_exact,
                                timing_valid,
                                skip_reuse=self.skip_reuse_validation,
                            )
                            metric = {
                                "case_key": key,
                                "case_id": int(case["case_id"]),
                                "batch_size": batch_size,
                                "input_len": total_len,
                                "input_len_built": built_len,
                                "cache_len_requested": cache_len,
                                "cache_len_observed": reuse_values,
                                "input_len_observed": [
                                    int(r.get("input_len", 0)) for r in successful
                                ],
                                "expected_reuse_len": expected_reuse,
                                "success_runs": len(successful),
                                "measure_runs": self.measure_runs,
                                "cache_commit_tail_tokens": self.cache_commit_tail_tokens,
                                "seed": seed_result,
                                "runs": runs,
                                "reuse_exact": reuse_exact,
                                "reuse_validation_skipped": self.skip_reuse_validation,
                                "validation_status": validation_status,
                                "shape_exact": shape_exact,
                                "timing_valid": timing_valid,
                                "prefill_time_ms": prefill_values,
                                "median_prefill_time_ms": (
                                    statistics.median(prefill_values)
                                    if timing_valid
                                    else None
                                ),
                                "avg_prefill_time_ms": (
                                    statistics.fmean(prefill_values)
                                    if timing_valid
                                    else None
                                ),
                                "elapsed_s": time.time() - started,
                                "status": status,
                            }
                    except Exception as exc:
                        metric = {
                            **case,
                            "case_key": key,
                            "case_id": int(case["case_id"]),
                            "batch_size": int(case.get("batch_size", 1)),
                            "input_len": int(case["input_len"]),
                            "cache_len_requested": int(case["cache_len"]),
                            "status": "error",
                            "error": repr(exc),
                            "batch_execution_failed": is_grouped_case(case),
                            "elapsed_s": time.time() - started,
                        }
                    if self.shared_seed:
                        metric["seed_mode"] = "shared_prefix_v1"
                        metric["shared_seed_refreshes"] = self._shared_seed_refreshes[
                            refresh_start:
                        ]
                        metric["shared_seed_id"] = (
                            self._shared_seed_record["seed_id"]
                            if self._shared_seed_record and int(case["cache_len"])
                            else None
                        )
                    self._results[key] = metric
                    self._append_journal(metric)
                    self._write_progress_checkpoint()
                    if idx % self.checkpoint_every == 0:
                        self._save(asynchronous=True)
                    if metric.get("status") == "ok":
                        completed_this_attempt += 1
                    logging.info(
                        "[CACHE_GRID] completed=%d/%d resume_attempt=%d/%d "
                        "%s status=%s reuse=%s query_prefetch=next",
                        completed_before + completed_this_attempt,
                        len(self.cases),
                        idx,
                        len(pending),
                        key,
                        metric.get("status"),
                        metric.get("cache_len_observed", []),
                    )
                    if (self.fail_fast and metric.get("status") != "ok") or (
                        is_grouped_case(case)
                        and metric.get("status") in {"failed", "error"}
                    ):
                        self._save()
                        raise RuntimeError(
                            f"cache grid stopped at {key}: "
                            f"status={metric.get('status')}"
                        )
            complete = all(
                self._results.get(self.case_key(case), {}).get("status") == "ok"
                for case in self.cases
            )
            self._save(complete=complete)
        except BaseException:
            try:
                self._save(complete=False)
            except Exception:
                logging.exception("cache grid: failed to flush interruption checkpoint")
            raise
        else:
            self._run_profiles()
            return list(self._results.values())
        finally:
            self._close_resources()

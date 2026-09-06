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
pattern structurally fall back to the legacy text-layer construction.
Precomputed prompts can also be materialized to disk once and reloaded
without a tokenizer (``MaterializedCaseStore``).
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter


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
        self, case_id: int, total_len: int, cache_len: int
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
        self, case_id: int, prefix: str, cache_len: int, commit_tail_tokens: int
    ) -> str:
        if cache_len <= 0:
            return ""
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
        return seed

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
        self, case_id: int, total_len: int, cache_len: int, run_count: int
    ) -> CasePrompts:
        """Build the seed prefix and the measured run prompts for one case.

        The fast path assembles exact-length token ids and texts from the
        marker ids plus filler repeats, costing string/list concatenation only.
        The first built case of each shape (cached / cold) is fully re-encoded
        and compared token by token; any mismatch, or a tokenizer without the
        structural filler property, falls back to the legacy text-layer path.
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
        return self._legacy_case_prompts(case_id, total_len, cache_len, run_count)

    def _fast_case_prompts(
        self, case_id: int, total_len: int, cache_len: int, run_count: int
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

    def _legacy_case_prompts(
        self, case_id: int, total_len: int, cache_len: int, run_count: int
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
    """Precomputed cache-grid prompts persisted on disk (plan A).

    All cached-case seed prefixes share one marker and a filler pattern, so
    every seed text is a character prefix of the longest one.  The store keeps
    that single base text under ``prefixes/base.txt`` plus one tiny JSON
    record per case (tail markers and filler counts); a case is rebuilt by
    string slicing and concatenation without touching the tokenizer.  Cases
    whose prompts could not be decomposed (legacy fallback) fall back to
    verbatim text records.
    """

    SCHEMA_VERSION = 1

    def __init__(self, root: str):
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.jsonl"
        self.info_path = self.root / "store_info.json"
        self._records: Dict[int, Dict[str, Any]] = {}
        self._loaded = False
        self._word = " hello"
        self._marker = PrefixPromptFactory.SHARED_PREFIX_MARKER
        self._marker_ids_len = -1
        self._base_text: Optional[str] = None
        self.run_count = -1
        self.grid_metadata: Dict[str, Any] = {}
        self.grid_sha256 = ""

    def materialize(
        self,
        cases: Iterable[Dict[str, int]],
        factory: PrefixPromptFactory,
        run_count: int,
        *,
        grid_metadata: Optional[Dict[str, Any]] = None,
        grid_sha256: str = "",
    ) -> Dict[str, Any]:
        if run_count <= 0:
            raise ValueError("run_count must be positive")
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "prefixes").mkdir(exist_ok=True)
        stats = {
            "cases": 0,
            "cached_cases": 0,
            "verbatim_seed": 0,
            "verbatim_runs": 0,
            "bytes": 0,
        }
        base_text = ""
        base_cache_len = -1
        case_list = list(cases)
        marker_ids = _encode(factory.tokenizer, factory.SHARED_PREFIX_MARKER)
        for case in case_list:
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
        (self.root / "prefixes" / "base.txt").write_text(base_text, encoding="utf-8")
        self.info_path.write_text(
            json.dumps(
                {
                    "schema_version": self.SCHEMA_VERSION,
                    "run_count": run_count,
                    "word": factory._word,
                    "marker": factory.SHARED_PREFIX_MARKER,
                    "marker_ids_len": len(marker_ids),
                    "max_cache_len": base_cache_len,
                    "case_count": stats["cases"],
                    "grid_metadata": grid_metadata or {},
                    "grid_sha256": grid_sha256,
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
        self._marker_ids_len = len(marker_ids)
        self._base_text = base_text
        return stats

    def load_cases(self) -> List[Dict[str, int]]:
        self._ensure_loaded()
        return [
            {
                "case_id": record["case_id"],
                "batch_size": record["batch_size"],
                "input_len": record["input_len"],
                "cache_len": record["cache_len"],
            }
            for record in self._records.values()
        ]

    def load_case(self, case: Dict[str, int]) -> CasePrompts:
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
                seed_text = self._base_prefix_text(cache_len)
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

    def _base_prefix_text(self, cache_len: int) -> str:
        if self._base_text is None:
            path = self.root / "prefixes" / "base.txt"
            self._base_text = path.read_text(encoding="utf-8")
        if self._marker_ids_len < 0:
            raise ValueError("store_info.json is missing marker_ids_len")
        chars = len(self._marker) + len(self._word) * (cache_len - self._marker_ids_len)
        if chars < 0 or chars > len(self._base_text):
            raise ValueError(
                f"prefix of {cache_len} tokens needs {chars} chars but the "
                f"materialized base text only has {len(self._base_text)}"
            )
        return self._base_text[:chars]

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"{self.root} is not a materialized case store "
                f"(missing manifest.jsonl)"
            )
        if self.info_path.exists():
            info = json.loads(self.info_path.read_text(encoding="utf-8"))
            self._word = info.get("word", self._word)
            self._marker = info.get("marker", self._marker)
            self._marker_ids_len = int(info.get("marker_ids_len", -1))
            self.run_count = int(info.get("run_count", -1))
            self.grid_metadata = info.get("grid_metadata", {})
            self.grid_sha256 = info.get("grid_sha256", "")
        with self.manifest_path.open(encoding="utf-8") as manifest:
            for line in manifest:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
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
            f"http://127.0.0.1:{port}", json=body, timeout=timeout
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


class CacheGridRunner:
    """Run and checkpoint a total-seq × prefix-cache grid."""

    def __init__(
        self,
        port: int,
        tokenizer: Any,
        cases: Iterable[Dict[str, int]],
        result_dir: str,
        *,
        request_timeout: int = 7200,
        measure_runs: int = 3,
        checkpoint_every: int = 1,
        cache_commit_tail_tokens: int = 4096,
        fail_fast: bool = True,
        grid_metadata: Dict[str, Any] | None = None,
        grid_sha256: str | None = None,
        expected_block_size: int = 0,
        case_store: "MaterializedCaseStore | None" = None,
    ):
        self.port = port
        self.factory = PrefixPromptFactory(tokenizer)
        self.cases = list(cases)
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
        self.grid_metadata = grid_metadata or {}
        self.grid_sha256 = grid_sha256
        self.expected_block_size = expected_block_size
        self.result_path = self.result_dir / "cache_grid_results.json"
        self._results: Dict[str, Dict[str, Any]] = {}
        self._http_session = _make_http_session()
        self._checkpoint_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="checkpoint-writer"
        )
        self._checkpoint_future: Optional[Future] = None
        if self.result_path.exists():
            with self.result_path.open(encoding="utf-8") as f:
                payload = json.load(f)
            self._results = {str(x["case_key"]): x for x in payload.get("metrics", [])}

    @staticmethod
    def case_key(case: Dict[str, int]) -> str:
        return f"bs{case['batch_size']}_seq{case['input_len']}_cache{case['cache_len']}"

    def _write_checkpoint(self, payload: Dict[str, Any]) -> None:
        tmp = self.result_path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, self.result_path)

    def _save(self, *, complete: bool = False, asynchronous: bool = False) -> None:
        payload = {
            "schema_version": 1,
            "mode": "prefix_cache_grid",
            "complete": complete,
            "total_cases": len(self.cases),
            "completed_cases": sum(
                row.get("status") == "ok" for row in self._results.values()
            ),
            "grid_metadata": self.grid_metadata,
            "grid_sha256": self.grid_sha256,
            "metrics": list(self._results.values()),
        }
        if asynchronous:
            if self._checkpoint_future is not None:
                self._checkpoint_future.result()
            self._checkpoint_future = self._checkpoint_executor.submit(
                self._write_checkpoint, payload
            )
            return
        if self._checkpoint_future is not None:
            self._checkpoint_future.result()
            self._checkpoint_future = None
        self._write_checkpoint(payload)

    def _close_resources(self) -> None:
        if self._checkpoint_future is not None:
            self._checkpoint_future.result()
            self._checkpoint_future = None
        self._checkpoint_executor.shutdown(wait=True)
        self._http_session.close()

    def _probe_reuse_granularity(self) -> None:
        """Fail fast when the assumed physical reuse granularity is wrong."""
        block = self.expected_block_size
        if block <= 0:
            return
        prefix_text, prefix_ids = self.factory._exact_text_and_ids(
            block, PrefixPromptFactory.SHARED_PREFIX_MARKER
        )
        seed_text = self.factory.make_seed(
            -1, prefix_text, block, self.cache_commit_tail_tokens
        )
        hit_text, hit_ids = self.factory._exact_text_and_ids(
            2 * block, prefix_text + " __block_probe_"
        )
        if hit_ids[:block] != prefix_ids:
            hit_text, hit_ids = self.factory._exact_text_and_ids(2 * block, prefix_text)
        if hit_ids[:block] != prefix_ids:
            raise RuntimeError(
                "block-size probe could not build a prompt preserving its own prefix"
            )
        seed = _post_prefill(
            self.port,
            seed_text,
            self.request_timeout,
            "block_size_probe:seed",
            self._http_session,
        )
        if not seed.get("success"):
            raise RuntimeError(f"block-size probe seed failed: {seed.get('error')}")
        hit = _post_prefill(
            self.port,
            hit_text,
            self.request_timeout,
            "block_size_probe:hit",
            self._http_session,
        )
        if not hit.get("success"):
            raise RuntimeError(f"block-size probe request failed: {hit.get('error')}")
        observed = int(hit.get("reuse_len", -1))
        if observed != block:
            raise RuntimeError(
                f"cache reuse granularity mismatch: probe observed reuse_len={observed}, "
                f"expected {block} tokens per physical block. Every cache-hitting case "
                "would be reported as invalid_reuse. Check --seq_size_per_block (x CP "
                "size when PREFILL_CP_KV_CACHE_SHARDED=1) or fix "
                "--expected_cache_block_size and rerun."
            )
        logging.info("[CACHE_GRID] block-size probe passed: reuse_len=%d", block)

    def _build_prompts(
        self, case: Dict[str, int], total_len: int, cache_len: int
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

    def _prepare_case_payload(self, case: Dict[str, int]) -> Dict[str, Any]:
        """Build prompts for one case without issuing a model request."""
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
        seed = (
            self.factory.make_seed(
                int(case["case_id"]),
                prefix,
                cache_len,
                self.cache_commit_tail_tokens,
            )
            if cache_len
            else ""
        )
        return {
            "key": key,
            "total_len": total_len,
            "cache_len": cache_len,
            "batch_size": batch_size,
            "built_len": built_len,
            "seed": seed,
            "run_targets": prompts.run_texts,
        }

    def run(self) -> List[Dict[str, Any]]:
        self._probe_reuse_granularity()
        pending = [
            case
            for case in self.cases
            if self._results.get(self.case_key(case), {}).get("status") != "ok"
        ]
        logging.info(
            "cache grid: %d total cases, %d already complete, %d pending",
            len(self.cases),
            len(self.cases) - len(pending),
            len(pending),
        )
        if not pending:
            self._save(complete=True)
            self._close_resources()
            return list(self._results.values())

        try:
            with ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="query-prefetch"
            ) as prep:
                prepared: Future = prep.submit(self._prepare_case_payload, pending[0])
                for idx, case in enumerate(pending, 1):
                    key = self.case_key(case)
                    started = time.time()
                    try:
                        try:
                            payload = prepared.result()
                        finally:
                            if idx < len(pending):
                                prepared = prep.submit(
                                    self._prepare_case_payload, pending[idx]
                                )
                        total_len = payload["total_len"]
                        cache_len = payload["cache_len"]
                        batch_size = payload["batch_size"]
                        built_len = payload["built_len"]
                        seed_result: Dict[str, Any] = {}
                        if cache_len:
                            seed_result = _post_prefill(
                                self.port,
                                payload["seed"],
                                self.request_timeout,
                                f"{key}:seed",
                                self._http_session,
                            )
                            if not seed_result.get("success"):
                                raise RuntimeError(
                                    seed_result.get("error", "seed request failed")
                                )

                        runs = [
                            _post_prefill(
                                self.port,
                                run_target,
                                self.request_timeout,
                                f"{key}:run{run_idx}",
                                self._http_session,
                            )
                            for run_idx, run_target in enumerate(payload["run_targets"])
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
                                int(r.get("output_len", -1)) == 1 for r in successful
                            )
                        )
                        ttft_values = [
                            float(r["ttft_ms"])
                            for r in successful
                            if float(r.get("ttft_ms", 0.0)) > 0.0
                        ]
                        timing_valid = len(ttft_values) == self.measure_runs
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
                            "shape_exact": shape_exact,
                            "timing_valid": timing_valid,
                            "ttft_ms": ttft_values,
                            "median_ttft_ms": (
                                statistics.median(ttft_values)
                                if timing_valid
                                else None
                            ),
                            "avg_ttft_ms": (
                                statistics.fmean(ttft_values)
                                if timing_valid
                                else None
                            ),
                            "elapsed_s": time.time() - started,
                            "status": (
                                "ok"
                                if reuse_exact and shape_exact and timing_valid
                                else "invalid_shape"
                                if len(successful) == self.measure_runs
                                and not shape_exact
                                else "invalid_timing"
                                if len(successful) == self.measure_runs
                                and not timing_valid
                                else "invalid_reuse"
                                if len(successful) == self.measure_runs
                                else "failed"
                            ),
                        }
                    except Exception as exc:
                        metric = {
                            "case_key": key,
                            "case_id": int(case["case_id"]),
                            "batch_size": int(case.get("batch_size", 1)),
                            "input_len": int(case["input_len"]),
                            "cache_len_requested": int(case["cache_len"]),
                            "status": "error",
                            "error": repr(exc),
                            "elapsed_s": time.time() - started,
                        }
                    self._results[key] = metric
                    if idx % self.checkpoint_every == 0:
                        self._save(asynchronous=True)
                    logging.info(
                        "[CACHE_GRID] %d/%d %s status=%s reuse=%s "
                        "query_prefetch=next",
                        idx,
                        len(pending),
                        key,
                        metric.get("status"),
                        metric.get("cache_len_observed", []),
                    )
                    if self.fail_fast and metric.get("status") != "ok":
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
            return list(self._results.values())
        finally:
            self._close_resources()

# Adaptive FIFO Continuous Pressure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make formal FIFO continuous-pressure runs derive a reusable seed bank from live FULL cache capacity, sustain at least 16 queued client requests, and require measurement-window D2H and H2D publication evidence.

**Architecture:** A new pure Python seed-planning module consumes the frozen `GetCachePressure` owner topology and produces a serialized `SeedPlan`; the runner freezes that plan before any seed request. `PressureSession` becomes FIFO-only for this preload workflow and initializes families through a bounded rolling sender. The existing pressure evaluator gains bounded, clock-conservative migration event intervals, while result writing exposes configuration, pressure, execution, and bidirectional validity independently.

**Tech Stack:** Python 3 standard-library dataclasses/concurrency/unittest, existing protobuf/gRPC pressure diagnostics, Bazel test launcher, DeepSeek-V4-Flash TP4 GPU service.

**Spec:** `openspec/changes/refine-continuous-pressure-seeding/design.md` and `openspec/changes/refine-continuous-pressure-seeding/specs/adaptive-fifo-cache-pressure/spec.md`; depends on the implemented `openspec/changes/add-continuous-full-pressure-diagnostics` change.

## Global Constraints

- Continuous full-pressure preload SHALL accept `fifo` only. Fixed-batch preload, padding, and batch-tail behavior are removed from this workflow; ordinary non-pressure batch execution remains unchanged.
- Formal seed count SHALL be derived only after the startup `GetCachePressure` snapshot. An explicit seed count is smoke/debug input and makes the run non-formal.
- Shared TP pools SHALL be counted once from the declared owner. Participant identity, layout, and freshness remain required evidence.
- A formal grid SHALL have `max(cache_len) >= 32768`; reusable tokens are `align_up(max(32768, max(cache_len)), cache_alignment)` and exclude the commit tail.
- Formal fill SHALL target `100 * 1024 = 102400` unique tokens before alignment. It SHALL fail rather than shrink when the live service cannot admit one request.
- Formal client in-flight depth SHALL default to 16 and SHALL be at least 16. It describes submitted/running/waiting RPCs, not the scheduler's active batch.
- Automatic seed planning SHALL make the reusable bank strictly larger than every required FULL owner's GPU family capacity and no larger than low-water GPU retention plus live host retention.
- `test_block_num > 0`, explicit seed count, client depth below 16, or a nonstandard formal fill target SHALL prevent formal acceptance. Smoke execution remains available and records every reason.
- Bidirectional validity SHALL require, for every required FULL owner group, successful DEVICE-to-HOST STORE/EVICT publication and successful HOST-to-DEVICE LOAD publication conservatively located inside `[measurement_start, measurement_end)`.
- Production GPU acceptance SHALL use `/mnt/hf3fs/3fs/models/DeepSeek-V4-Flash` without `test_block_num`; the prior 127-block run remains smoke evidence.
- No protobuf or C++ cache schema change is planned. Existing group geometry and pool `total_blocks`/`available_blocks` fields are the authoritative inputs.
- Read the dirty worktree before every commit. Stage only the files or hunks owned by the task; never stage all untracked OpenSpec, documentation, or user changes.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `rtp_llm/test/perf_test/cache_grid/runner/pressure_seed_plan.py` | Formal constants, run classification, owner-pool deduplication, token/block conversion, feasible seed interval, serializable plan |
| `rtp_llm/test/perf_test/cache_grid/tests/pressure_seed_plan_test.py` | Pure planner tests with no model service |
| `rtp_llm/test/perf_test/cache_grid/config/perf_profile.py` | Add seed timeout option and resolve the 102400-token fill default |
| `rtp_llm/test/perf_test/perf_config.py` | Direct runner defaults: FIFO preload, client depth 16, optional automatic seed count |
| `rtp_llm/test/perf_test/batch_decode_test.py` | Freeze startup limits/test-capacity provenance and preserve ordinary scheduler behavior |
| `rtp_llm/test/perf_test/cache_grid/runner/cache_perf.py` | Public CLI defaults, preload validation, and forwarding without an invented seed count |
| `rtp_llm/test/perf_test/cache_grid/runner/cache_grid_runner.py` | Post-startup plan creation, per-request admission checks, request construction, and manifest integration |
| `rtp_llm/test/perf_test/cache_grid/runner/continuous_pressure.py` | FIFO-only bounded rolling seed lifecycle and independent seed timeout |
| `rtp_llm/test/perf_test/cache_grid/runner/pressure_diagnostics.py` | Decode live available blocks and evaluate bounded measurement-window transfer deltas |
| `rtp_llm/test/perf_test/cache_grid/runner/pressure_results.py` | Persist seed outcome and expose four independent validity gates plus aggregate formal acceptance |
| `rtp_llm/test/perf_test/cache_grid/tests/cache_perf_test.py` | Public CLI/default/forwarding regressions |
| `rtp_llm/test/perf_test/cache_grid/tests/perf_profile_test.py` | Pressure config default and seed-timeout validation |
| `rtp_llm/test/perf_test/cache_grid/tests/continuous_pressure_test.py` | Direct config, preflight, rolling sender, timeout, cleanup, and runner integration tests |
| `rtp_llm/test/perf_test/cache_grid/tests/pressure_diagnostics_test.py` | Owner topology and conservative migration-window tests |
| `rtp_llm/test/perf_test/cache_grid/tests/pressure_results_test.py` | Manifest/summary and independent validity tests |
| `rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_dsv4.example.jsonc` | Explicit reduced-capacity non-formal smoke profile |
| `rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_dsv4.production.example.jsonc` | Production-capacity DeepSeek-V4-Flash formal profile |
| `rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_grid.example.json` | Reduced smoke grid retained for quick validation |
| `rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_grid.production.example.json` | Formal grid with reusable 32K/64K families |
| `rtp_llm/test/perf_test/cache_grid/docs/continuous_full_pressure_guide_zh.md` | Chinese production and smoke commands and interpretation |
| `rtp_llm/test/perf_test/cache_grid/docs/reference.md` | Stable field/formula/result contract and acceptance status |
| `rtp_llm/test/perf_test/cache_grid/docs/continuous_full_pressure_plan.md` | Mark old fixed-batch/manual-seed decisions as superseded |
| `rtp_llm/test/perf_test/cache_grid/docs/continuous_full_pressure_acceptance_zh.md` | Commands and evidence paths from smoke and production GPU acceptance |

### Task 1: Freeze FIFO Defaults and Formal Classification Inputs

**Files:**
- Create: `rtp_llm/test/perf_test/cache_grid/runner/pressure_seed_plan.py`
- Modify: `rtp_llm/test/perf_test/cache_grid/config/perf_profile.py:38-77`
- Modify: `rtp_llm/test/perf_test/perf_config.py:113-147,392-415`
- Modify: `rtp_llm/test/perf_test/cache_grid/runner/cache_perf.py:162-332,400-430,626-656`
- Modify: `rtp_llm/test/perf_test/batch_decode_test.py:920-974,1029-1057,1168-1193,1215-1226`
- Test: `rtp_llm/test/perf_test/cache_grid/tests/perf_profile_test.py`
- Test: `rtp_llm/test/perf_test/cache_grid/tests/cache_perf_test.py`
- Test: `rtp_llm/test/perf_test/batch_decode_test_test.py`

**Interfaces:**
- Consumes: existing `pressure_settings(profile, args, direct=...)`, public `--pressure-*` arguments, direct `--cache_pressure_*` arguments, and `_full_pressure_startup()`.
- Produces: constants `MIN_REUSE_TOKENS`, `FORMAL_FILL_TARGET_TOKENS`, `MIN_FORMAL_CLIENT_INFLIGHT`; `formal_config_reasons(...) -> tuple[str, ...]`; optional `cache_pressure_seed_count`; startup field `test_block_num`; launch-summary fields `formal_candidate` and `formal_candidate_reasons`.

- [ ] **Step 1: Write failing constant/classification tests in a new planner test module.**

Create `rtp_llm/test/perf_test/cache_grid/tests/pressure_seed_plan_test.py` with this first test case:

```python
import unittest

from rtp_llm.test.perf_test.cache_grid.runner.pressure_seed_plan import (
    FORMAL_FILL_TARGET_TOKENS,
    MIN_FORMAL_CLIENT_INFLIGHT,
    MIN_REUSE_TOKENS,
    formal_config_reasons,
)


class FormalConfigTest(unittest.TestCase):
    def test_constants_and_monotonic_reasons(self):
        self.assertEqual(MIN_REUSE_TOKENS, 32768)
        self.assertEqual(FORMAL_FILL_TARGET_TOKENS, 102400)
        self.assertEqual(MIN_FORMAL_CLIENT_INFLIGHT, 16)
        self.assertEqual(
            formal_config_reasons(
                manual_seed_count=None,
                client_inflight=16,
                test_block_num=0,
                fill_target_tokens=102400,
            ),
            (),
        )
        self.assertEqual(
            formal_config_reasons(
                manual_seed_count=2,
                client_inflight=2,
                test_block_num=128,
                fill_target_tokens=16384,
            ),
            (
                "manual_seed_count",
                "client_inflight_below_16",
                "synthetic_test_block_capacity",
                "nonstandard_fill_target",
            ),
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the new test and confirm the module is missing.**

Run:

```bash
python3 -m unittest rtp_llm.test.perf_test.cache_grid.tests.pressure_seed_plan_test -v
```

Expected: FAIL with `ModuleNotFoundError` for `pressure_seed_plan`.

- [ ] **Step 3: Add the constants and exact classification helper.**

Start `pressure_seed_plan.py` with:

```python
from __future__ import annotations

MIN_REUSE_TOKENS = 32 * 1024
FORMAL_FILL_TARGET_TOKENS = 100 * 1024
MIN_FORMAL_CLIENT_INFLIGHT = 16


def formal_config_reasons(
    *,
    manual_seed_count: int | None,
    client_inflight: int,
    test_block_num: int,
    fill_target_tokens: int,
) -> tuple[str, ...]:
    reasons = []
    if manual_seed_count is not None:
        reasons.append("manual_seed_count")
    if client_inflight < MIN_FORMAL_CLIENT_INFLIGHT:
        reasons.append("client_inflight_below_16")
    if test_block_num > 0:
        reasons.append("synthetic_test_block_capacity")
    if fill_target_tokens != FORMAL_FILL_TARGET_TOKENS:
        reasons.append("nonstandard_fill_target")
    return tuple(reasons)
```

Reject booleans and non-positive numeric inputs before returning reasons; keep the tuple order shown above so manifests and tests are deterministic.

- [ ] **Step 4: Write failing CLI/profile tests for the new defaults and FIFO-only preload.**

Extend `CachePerfTest` and the direct parser tests with assertions equivalent to:

```python
plan = cli.build_plan(
    self.args(extra=["--pressure-mode=continuous", "--pressure-preload"]), {}
)
runner = json.loads(plan["artifacts"][cli.MANIFEST])["runner_args"]
self.assertIn("--cache_pressure_scheduler=fifo", runner)
self.assertIn("--cache_pressure_client_concurrency=16", runner)
self.assertFalse(any("cache_pressure_seed_count" in arg for arg in runner))
self.assertEqual(
    json.loads(plan["artifacts"]["profile.snapshot.json"])["cache_grid"][
        "pressure"
    ]["fill_input_tokens"],
    102400,
)

with self.assertRaisesRegex(ValueError, "FIFO-only"):
    cli.build_plan(
        self.args(
            extra=[
                "--pressure-mode=continuous",
                "--pressure-preload",
                "--pressure-scheduler=batch_decode",
            ]
        ),
        {},
    )
```

Also assert that continuous execution without `--pressure-preload` still resolves `batch_decode` and seed count 4, preserving the ordinary finite workflow.

- [ ] **Step 5: Run focused configuration tests and observe the old values.**

Run:

```bash
python3 -m unittest \
  rtp_llm.test.perf_test.cache_grid.tests.perf_profile_test \
  rtp_llm.test.perf_test.cache_grid.tests.cache_perf_test \
  rtp_llm.test.perf_test.batch_decode_test_test -v
```

Expected: FAIL because preload still defaults to `batch_decode`, client concurrency `0`, seed count `4`, and fill input has no 102400 default.

- [ ] **Step 6: Implement context-sensitive defaults without changing ordinary execution.**

In `pressure_settings()`, set only the formal fill target when absent:

```python
from rtp_llm.test.perf_test.cache_grid.runner.pressure_seed_plan import (
    FORMAL_FILL_TARGET_TOKENS,
)

values = dict(profile.get("cache_grid", {}).get("pressure", {}))
values.setdefault("fill_input_tokens", FORMAL_FILL_TARGET_TOKENS)
```

In both public and direct argument flows, resolve omitted values as:

```python
scheduler = supplied_scheduler or ("fifo" if pressure_preload else "batch_decode")
client_inflight = supplied_concurrency or (
    MIN_FORMAL_CLIENT_INFLIGHT if pressure_preload else 0
)
seed_count = supplied_seed_count if supplied_seed_count is not None else (
    None if pressure_preload else 4
)
```

Change the direct `--cache_pressure_seed_count` default from `4` to `None`. Do not forward a `cache_pressure_seed_count` argument from `cache_perf.py` when it is `None`. Reject any resolved non-FIFO scheduler when preload is enabled, but retain the existing scheduler selection for non-preload execution.

In `_full_pressure_startup()`, freeze the exact synthetic override:

```python
test_block_num = int(extract_arg(remaining, "test_block_num") or 0)
return dict(
    # existing fields remain
    test_block_num=test_block_num,
)
```

Remove the full-pressure comparison that requires `concurrency_limit` to be at least the client backlog or seed count. Keep positive service admission validation; active service concurrency is still used later for single-request feasibility.

In `cache_perf.build_plan()`, call `formal_config_reasons()` with the resolved public arguments and the frozen profile's `engine_args["--test_block_num"]`. Store `formal_candidate=not reasons` and the ordered reasons in both `cache_perf_launch.json` and the returned dry-run summary. Label these fields as static configuration screening; they never replace the live post-startup plan.

- [ ] **Step 7: Run the focused configuration tests and confirm ordinary batch regressions pass.**

Run the Step 5 command again. Expected: PASS, including the new preload defaults and existing non-preload batch tests.

- [ ] **Step 8: Commit the configuration contract.**

```bash
git add \
  rtp_llm/test/perf_test/cache_grid/runner/pressure_seed_plan.py \
  rtp_llm/test/perf_test/cache_grid/tests/pressure_seed_plan_test.py \
  rtp_llm/test/perf_test/cache_grid/config/perf_profile.py \
  rtp_llm/test/perf_test/perf_config.py \
  rtp_llm/test/perf_test/cache_grid/runner/cache_perf.py \
  rtp_llm/test/perf_test/batch_decode_test.py \
  rtp_llm/test/perf_test/cache_grid/tests/perf_profile_test.py \
  rtp_llm/test/perf_test/cache_grid/tests/cache_perf_test.py \
  rtp_llm/test/perf_test/batch_decode_test_test.py
git diff --cached --check
git commit -m "feat(cache-grid): define formal FIFO pressure defaults"
```

### Task 2: Implement the Live FULL Seed Planner

**Files:**
- Modify: `rtp_llm/test/perf_test/cache_grid/runner/pressure_seed_plan.py`
- Modify: `rtp_llm/test/perf_test/cache_grid/runner/pressure_diagnostics.py:36-100,162-212,530-689`
- Modify: `rtp_llm/test/perf_test/cache_grid/tests/pressure_seed_plan_test.py`
- Modify: `rtp_llm/test/perf_test/cache_grid/tests/pressure_diagnostics_test.py`

**Interfaces:**
- Consumes: `tuple[RankSample, ...]`, `tuple[PressureTarget, ...]`, normalized grid cases, cache alignment, commit tail, configured fill target, optional manual seed count, client depth, and frozen `test_block_num`.
- Produces: `SeedPoolPlan`, `SeedGroupPlan`, `SeedPlan`, `align_up()`, and `build_seed_plan(...) -> SeedPlan`. Later tasks serialize `dataclasses.asdict(plan)` unchanged.

- [ ] **Step 1: Extend decoded pools with live free/available values and write failing decode tests.**

Append defaulted fields to `PoolSample` so existing positional fixtures remain compatible:

```python
@dataclass(frozen=True)
class PoolSample:
    # existing fields remain in their current order
    free: int = 0
    available: int = 0
```

Add a decoder test that sets `total_blocks=128`, `free_blocks=17`, and `available_blocks=23` and expects `PoolSample(capacity=128, free=17, available=23)`. Add invalid cases for `free > capacity` and `available > capacity`.

- [ ] **Step 2: Run the decoder test and confirm the fields are not projected.**

Run:

```bash
python3 -m unittest \
  rtp_llm.test.perf_test.cache_grid.tests.pressure_diagnostics_test -v
```

Expected: FAIL because decoded samples do not yet retain live free/available blocks.

- [ ] **Step 3: Decode and validate authoritative live pool availability.**

In `decode_pressure_response()`, pass `pool.free_blocks` and `pool.available_blocks` into `PoolSample`. In `_pool_map()`, validate all six occupancy fields and these invariants:

```python
if not (
    0 <= pool.pending_release <= pool.used <= pool.capacity
    and 0 <= pool.free <= pool.capacity
    and 0 <= pool.available <= pool.capacity
    and pool.host_resident <= pool.used
):
    raise ValueError("invalid physical occupancy")
```

Do not add dynamic `free` or `available` to `_layout()`; topology continuity remains based on identity, capacity, geometry, and watermarks.

- [ ] **Step 4: Write failing planner tests for owner deduplication and the feasible interval.**

Use existing `RankSample`, `GroupSample`, `PoolSample`, `PressureOwnership`, and `select_targets()` fixtures. Cover this exact single-owner calculation:

```python
plan = build_seed_plan(
    samples,
    select_targets(samples, (0, 1)),
    cases=(dict(input_len=36864, cache_len=32768, batch_size=1),),
    cache_alignment=4096,
    commit_tail_tokens=4096,
    client_inflight=16,
    manual_seed_count=None,
    fill_target_tokens=102400,
    test_block_num=0,
)
self.assertEqual(plan.reusable_tokens, 32768)
self.assertEqual(plan.seed_input_tokens, 36864)
self.assertEqual(len(plan.groups), 1)  # shared TP owner counted once
self.assertEqual(plan.groups[0].seed_blocks, 64)  # 32768 / 512
self.assertEqual(plan.groups[0].gpu_family_capacity, 1)  # 127 // 64
self.assertEqual(plan.groups[0].minimum_families, 2)
self.assertEqual(plan.seed_count, 2)
self.assertEqual(plan.source, "automatic")
self.assertTrue(plan.formal_config_valid)
```

Add cases for multiple FULL owner groups, duplicate participant views, `max(cache_len)=32767`, unaligned `cache_len=33000`, heterogeneous group `block_tokens`, host upper-bound failure, manual smoke count, and deterministic non-formal reasons.

- [ ] **Step 5: Run the planner tests and confirm the dataclasses/functions are missing.**

Run:

```bash
python3 -m unittest rtp_llm.test.perf_test.cache_grid.tests.pressure_seed_plan_test -v
```

Expected: FAIL on imports for `SeedPlan`/`build_seed_plan`.

- [ ] **Step 6: Implement immutable plan types and exact block arithmetic.**

Add these public types to `pressure_seed_plan.py`:

```python
from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class SeedPoolPlan:
    pool_id: str
    capacity_blocks: int
    available_blocks: int
    low_blocks: int
    seed_blocks: int


@dataclass(frozen=True)
class SeedGroupPlan:
    owner_rank: int
    owner_group_id: int
    block_tokens: int
    device_pools: tuple[SeedPoolPlan, ...]
    host_pool: SeedPoolPlan
    gpu_family_capacity: int
    gpu_low_families: int
    host_families: int
    minimum_families: int
    maximum_families: int

    @property
    def seed_blocks(self) -> int:
        return max(pool.seed_blocks for pool in self.device_pools)


@dataclass(frozen=True)
class SeedPlan:
    source: str
    seed_count: int
    grid_reuse_tokens: int
    reusable_tokens: int
    commit_tail_tokens: int
    seed_input_tokens: int
    fill_target_tokens: int
    fill_input_tokens: int
    client_inflight: int
    formal_config_valid: bool
    nonformal_reasons: tuple[str, ...]
    groups: tuple[SeedGroupPlan, ...]


def align_up(value: int, alignment: int) -> int:
    if type(value) is not int or type(alignment) is not int or value <= 0 or alignment <= 0:
        raise ValueError("positive integer value/alignment required")
    return ((value + alignment - 1) // alignment) * alignment
```

Implement:

```python
def build_seed_plan(
    samples: tuple[Any, ...],
    targets: tuple[Any, ...],
    *,
    cases: tuple[dict, ...],
    cache_alignment: int,
    commit_tail_tokens: int,
    client_inflight: int,
    manual_seed_count: int | None,
    fill_target_tokens: int,
    test_block_num: int,
) -> SeedPlan:
```

The implementation must:

1. Build unique logical group keys from `(target.owner_rank, target.owner_group_id)`.
2. Read only the owner sample/group for capacity and counters; never sum participant pools.
3. Use `group.geometry[0]` as the live group block-token span and fail if it is non-positive.
4. Deduplicate owner device pool IDs before calculating `seed_blocks = ceil(reusable_tokens / block_tokens)`.
5. Calculate `gpu_family_capacity = min(capacity // seed_blocks)`, `gpu_low_families = min(floor(capacity * low_ratio) // seed_blocks)`, and `host_families = host.available // host_seed_blocks`.
6. Set `minimum_families = gpu_family_capacity + 1`, `maximum_families = gpu_low_families + host_families`, global lower bound to the maximum minimum, and global upper bound to the minimum maximum.
7. Raise `ValueError` containing the owner/group and both bounds when an automatic plan has no feasible interval.
8. Align the formal fill target upward with `cache_alignment`; never round down.
9. For an otherwise formal candidate, reject a grid below 32K instead of silently downgrading it. Permit that grid only when an independent smoke reason already exists, and append `grid_reuse_below_32768`.
10. For a manual smoke count, require a positive integer and `manual_seed_count <= global_upper`; record source `manual` and do not pretend it exceeds GPU capacity.

- [ ] **Step 7: Run pure planner and decoder tests.**

Run:

```bash
python3 -m unittest \
  rtp_llm.test.perf_test.cache_grid.tests.pressure_seed_plan_test \
  rtp_llm.test.perf_test.cache_grid.tests.pressure_diagnostics_test -v
```

Expected: PASS for owner deduplication, exact-capacity `+1`, host feasibility, alignment, manual smoke, and decoder validation.

- [ ] **Step 8: Commit the pure live-capacity planner.**

```bash
git add \
  rtp_llm/test/perf_test/cache_grid/runner/pressure_seed_plan.py \
  rtp_llm/test/perf_test/cache_grid/runner/pressure_diagnostics.py \
  rtp_llm/test/perf_test/cache_grid/tests/pressure_seed_plan_test.py \
  rtp_llm/test/perf_test/cache_grid/tests/pressure_diagnostics_test.py
git diff --cached --check
git commit -m "feat(cache-grid): plan seed bank from live FULL capacity"
```

### Task 3: Integrate the Frozen Seed Plan into Runtime Preflight

**Files:**
- Modify: `rtp_llm/test/perf_test/cache_grid/runner/cache_grid_runner.py:987-1205,1839-2081,2230-2265,3087-3222`
- Modify: `rtp_llm/test/perf_test/cache_grid/runner/continuous_pressure.py:145-175,231-332`
- Modify: `rtp_llm/test/perf_test/cache_grid/tests/continuous_pressure_test.py:1090-1620`
- Modify: `rtp_llm/test/perf_test/cache_grid/tests/pressure_results_test.py`

**Interfaces:**
- Consumes: Task 2 `build_seed_plan()` and serialized `SeedPlan`.
- Produces: `CacheGridRunner.pressure_seed_count: int | None`, `CacheGridRunner._pressure_seed_plan: SeedPlan`, `PressureSession(..., seed_plan: SeedPlan, ...)`, frozen event field `seed_plan`, and manifest field `seed_plan`.

- [ ] **Step 1: Write failing post-startup preflight tests.**

Extend the existing fake `GetCachePressure` integration fixture to assert:

```python
self.assertIsNone(runner.pressure_seed_count)
result = runner._run_full_pressure_epoch("epoch")
manifest = json.loads(
    (runner._pressure_epoch_dir / "pressure/manifest.json").read_text()
)
self.assertEqual(manifest["seed_plan"]["source"], "automatic")
self.assertEqual(manifest["seed_plan"]["seed_count"], 2)
self.assertEqual(len(manifest["seed_plan"]["groups"]), 1)
```

Add negative cases for missing FULL owner pool, live `available_blocks > total_blocks`, a formal grid below 32K, a 102400-token fill exceeding model/batch/KV admission, and a manual count that is recorded as non-formal.

- [ ] **Step 2: Run the runner integration tests and confirm seed count is still chosen before startup.**

Run:

```bash
python3 -m unittest \
  rtp_llm.test.perf_test.cache_grid.tests.continuous_pressure_test \
  rtp_llm.test.perf_test.cache_grid.tests.pressure_results_test -v
```

Expected: FAIL because `_run_full_pressure()` still derives seed length/count from constructor fields and the manifest has no `seed_plan`.

- [ ] **Step 3: Make the direct runner contract optional-seed and FIFO-only for preload.**

Change the constructor annotation/default to `pressure_seed_count: int | None = None`. When `pressure_config is not None`, require `pressure_scheduler == "fifo"`, `pressure_batch_size == 1`, and allow `pressure_seed_count is None`; validate an explicit count only as a positive smoke override. Preserve the old positive integer default/resolution in non-preload continuous mode.

Initialize:

```python
self._pressure_seed_plan = None
self._continuous_seed_ids = []
```

Do not allocate seed ID slots in the constructor or before the live plan exists.

- [ ] **Step 4: Separate single-request admission from client backlog.**

Keep `ExecutionLimits.concurrency_for(input_tokens)` for active fill sizing, and add a clearer wrapper:

```python
def require_request(self, input_tokens: int) -> None:
    self.concurrency_for(input_tokens)
```

Delete both comparisons that reject `pressure_client_concurrency > concurrency_for(max_input)`. Preflight must call `require_request()` independently for the effective fill length, seed input length, and every distinct formal `input_len`; a positive result proves one request can execute, while the sender bound controls queued RPCs.

- [ ] **Step 5: Build and freeze the plan immediately after the startup snapshot.**

In `_run_full_pressure_epoch()`, after `decode_pressure_response()` and `select_targets()`:

```python
from dataclasses import asdict, replace
from .pressure_seed_plan import build_seed_plan

alignment = int(self.grid_metadata["generator"]["cache_alignment"])
seed_plan = build_seed_plan(
    decoded.samples,
    targets,
    cases=tuple(self.cases),
    cache_alignment=alignment,
    commit_tail_tokens=self.cache_commit_tail_tokens,
    client_inflight=self.pressure_client_concurrency,
    manual_seed_count=self.pressure_seed_count,
    fill_target_tokens=self.pressure_config.fill_input_tokens,
    test_block_num=int(startup.get("test_block_num", 0)),
)
self._pressure_seed_plan = seed_plan
effective_config = replace(
    self.pressure_config, fill_input_tokens=seed_plan.fill_input_tokens
)
```

Validate each request through `limits.require_request()`. Add `seed_plan=asdict(seed_plan)` to the evidence manifest and pass the immutable plan plus `effective_config` into `_run_full_pressure()`.

- [ ] **Step 6: Construct seed/fill/formal requests exclusively from the frozen plan.**

Change `_run_full_pressure(..., seed_plan, ...)` so it uses:

```python
seed_count = seed_plan.seed_count
seed_tokens = seed_plan.seed_input_tokens
self._continuous_seed_ids = [None] * seed_count
```

Use `seed_plan.fill_input_tokens` for unique fill payloads. Continue forming formal requests from `seed_ids[family][:case["cache_len"]]` plus a unique suffix, which makes only the grid-requested prefix reusable. Include `seed_plan` in the session's `frozen` event; never infer it again from cases.

Replace the old `seed_count` and `seed_input_tokens` constructor parameters with the plan itself:

```python
class PressureSession:
    def __init__(self, config, *, seed_plan, **existing_dependencies):
        self.seed_plan = seed_plan
        self.seed_count = seed_plan.seed_count
        self.seed_input_tokens = seed_plan.seed_input_tokens
```

Construct it with `seed_plan=seed_plan`. The session may read these frozen values but must not modify or recalculate the dataclass.

- [ ] **Step 7: Verify manifest reproducibility and admission behavior.**

Run the Step 2 command again. Add one assertion that reconstructs the group lower/upper bounds from `manifest["seed_plan"]` and matches the serialized `seed_count`. Expected: PASS, with 16 client RPCs allowed even when `limits.concurrency_for(long_request) == 1`.

- [ ] **Step 8: Commit runtime planning integration.**

```bash
git add \
  rtp_llm/test/perf_test/cache_grid/runner/cache_grid_runner.py \
  rtp_llm/test/perf_test/cache_grid/runner/continuous_pressure.py \
  rtp_llm/test/perf_test/cache_grid/tests/continuous_pressure_test.py \
  rtp_llm/test/perf_test/cache_grid/tests/pressure_results_test.py
git diff --cached --check
git commit -m "feat(cache-grid): freeze live seed plan before pressure requests"
```

### Task 4: Replace Seed Batch Barriers with Rolling FIFO Supply

**Files:**
- Modify: `rtp_llm/test/perf_test/cache_grid/runner/pressure_diagnostics.py:12-34,133-159`
- Modify: `rtp_llm/test/perf_test/cache_grid/config/perf_profile.py:39-60`
- Modify: `rtp_llm/test/perf_test/cache_grid/runner/continuous_pressure.py:140-175,176-217,231-332,529-590,723-950`
- Modify: `rtp_llm/test/perf_test/cache_grid/runner/cache_grid_runner.py:2245-2265`
- Modify: `rtp_llm/test/perf_test/cache_grid/tests/perf_profile_test.py`
- Modify: `rtp_llm/test/perf_test/cache_grid/tests/continuous_pressure_test.py:1-820`

**Interfaces:**
- Consumes: frozen `SeedPlan.seed_count`, `SeedPlan.seed_input_tokens`, sender limit, and `PressureConfig.seed_timeout_s`.
- Produces: FIFO-only `PressureSession`, `_run_seed_phase() -> dict`, finished result field `seed`, and deterministic seed lifecycle events.

- [ ] **Step 1: Add a required independent seed timeout and failing validation tests.**

Insert `seed_timeout_s: float` with the other required `PressureConfig` fields before defaulted fields. Add `("seed-timeout", "seed_timeout_s", float)` to `PRESSURE_OPTIONS`. Update all test fixture constructors with an explicit value; use `seed_timeout_s=30` in unit fixtures, `300` in the reduced smoke profile, and `3600` in the production profile created later.

Test zero, negative, boolean, and nonfinite values through `resolve_pressure_config()`.

- [ ] **Step 2: Write failing rolling-supply and timeout tests.**

Use controllable request handles to verify the sequence:

```python
session = make_session(seed_count=20, concurrency=16)
result = session.run()
self.assertEqual(session.max_seed_pending, 16)
self.assertEqual(result["seed"]["submitted"], 20)
self.assertEqual(result["seed"]["completed"], 20)
self.assertEqual(result["seed"]["failed"], 0)
self.assertEqual(result["seed"]["pending"], 0)
self.assertFalse(any(e.get("kind") == "padding" for e in events))
```

The handle fixture must complete one initial seed while fifteen remain pending and assert that family 16 is submitted before those fifteen drain. Add timeout/failure cases that assert phase `fill` is never emitted and all accepted handles are completed or cancelled.

- [ ] **Step 3: Run focused tests and observe batch-barrier behavior.**

Run:

```bash
python3 -m unittest \
  rtp_llm.test.perf_test.cache_grid.tests.perf_profile_test \
  rtp_llm.test.perf_test.cache_grid.tests.continuous_pressure_test -v
```

Expected: FAIL because the current seed loop calls `_batch()` and waits for the entire `seed_batch` before submitting replacements.

- [ ] **Step 4: Make `PressureSession` reject non-FIFO preload directly.**

Keep the constructor arguments for a small migration diff, but enforce:

```python
if scheduler != "fifo":
    raise ValueError("continuous full-pressure preload is FIFO-only")
if batch_size != 1:
    raise ValueError("FIFO full-pressure preload has no fixed batch")
```

Remove `padding_needed()`, fixed-batch warmup state, `_configure_batch()`, final tail submission, and `set_batch` calls from this class. This class is the preload session; ordinary non-pressure fixed batching continues through the existing runner paths outside it.

- [ ] **Step 5: Track seed outcomes in `_collect()` and implement rolling supply.**

Initialize exact counters:

```python
self.seed_submitted = 0
self.seed_completed = 0
self.seed_failed = 0
self.seed_cancelled = 0
self.max_seed_pending = 0
```

Increment completion/failure counters from the stored dispatch record's `kind`. When `_drain()` requests cancellation for a pending seed record, increment `seed_cancelled` exactly once even if the transport rejects cancellation. Implement:

```python
def _run_seed_phase(self) -> dict:
    started = self.clock()
    deadline = started + self.config.seed_timeout_s
    self.state.phase = "seed"
    self.state.entered = started
    self.emit("phase", phase="seed", at_s=started)
    while self.seed_completed + self.seed_failed < self.seed_count:
        while self.seed_submitted < self.seed_count and len(self.sender.pending) < self.sender.limit:
            if not self._submit("seed"):
                break
            self.seed_submitted += 1
            self.max_seed_pending = max(self.max_seed_pending, len(self.sender.pending))
        self._tick()
        if self.state.phase == "drain":
            break
        if self.clock() >= deadline:
            self.fail("seed_timeout")
            break
        self._pause()
    return dict(
        planned=self.seed_count,
        submitted=self.seed_submitted,
        completed=self.seed_completed,
        failed=self.seed_failed,
        cancelled=self.seed_cancelled,
        pending=len(self.sender.pending),
        max_inflight=self.max_seed_pending,
        elapsed_s=self.clock() - started,
    )
```

Only enter fill when `completed == planned`, `failed == 0`, and no seed handles remain. On timeout, stop new submissions and let the existing bounded drain cancel/reap accepted work.

- [ ] **Step 6: Emit and persist the seed summary independently from fill timing.**

Call `_run_seed_phase()` after the initial frozen baseline. Reset `state.phase`, `state.entered`, `last_progress`, and fill counters only after successful seed completion. Include the returned object in the final session result as `seed`; seed dispatch/completion timestamps remain in raw evidence but do not contribute to fill timeout, steady coverage, or measurement migration baselines.

- [ ] **Step 7: Run lifecycle tests and the entire continuous-pressure module.**

Run the Step 3 command again. Expected: PASS with immediate replacement, maximum pending exactly bounded, no padding, seed timeout cleanup, and unchanged ordinary runner tests.

- [ ] **Step 8: Commit the rolling FIFO lifecycle.**

```bash
git add \
  rtp_llm/test/perf_test/cache_grid/runner/pressure_diagnostics.py \
  rtp_llm/test/perf_test/cache_grid/config/perf_profile.py \
  rtp_llm/test/perf_test/cache_grid/runner/continuous_pressure.py \
  rtp_llm/test/perf_test/cache_grid/runner/cache_grid_runner.py \
  rtp_llm/test/perf_test/cache_grid/tests/perf_profile_test.py \
  rtp_llm/test/perf_test/cache_grid/tests/continuous_pressure_test.py
git diff --cached --check
git commit -m "feat(cache-grid): seed FULL pressure with rolling FIFO supply"
```

### Task 5: Evaluate Measurement-Window D2H and H2D Publication

**Files:**
- Modify: `rtp_llm/test/perf_test/cache_grid/runner/pressure_diagnostics.py:695-780,824-1110`
- Modify: `rtp_llm/test/perf_test/cache_grid/runner/continuous_pressure.py:591-715,879-897`
- Modify: `rtp_llm/test/perf_test/cache_grid/tests/pressure_diagnostics_test.py`
- Modify: `rtp_llm/test/perf_test/cache_grid/tests/continuous_pressure_test.py`

**Interfaces:**
- Consumes: cumulative `RankSample.counters`, frozen `PressureTarget` owner mappings, adjacent valid samples, and `[measurement_start, measurement_end)`.
- Produces: `MigrationGroupAssessment`, `MigrationAssessment`, `PressureCoverage.assess_migration(start, end)`, and session result field `migration`.

- [ ] **Step 1: Write failing direction/window tests around real wire counter keys.**

Build adjacent decoded samples with these endpoint increments:

```python
# operation: 1=LOAD, 2=EVICT, 3=STORE; tier: 1=DEVICE, 2=HOST
d2h = ("published_blocks", group_id, host_pool_id, 2, 1, 2)
h2d = ("published_blocks", group_id, device_pool_id, 1, 2, 1)
```

Assert all of the following:

```python
assessment = coverage.assess_migration(10.0, 20.0)
self.assertTrue(assessment.valid)
self.assertEqual(assessment.reasons, ())
self.assertTrue(all(group.valid for group in assessment.groups))
```

Separate tests must reject D2H-only, H2D-only, copied-but-unpublished, failed operations, a counter increment whose adjacent-sample uncertainty straddles `t0`, an increment after `t1`, one missing shared TP participant, and a counter reset. A shared TP fixture with four participant views must yield one logical owner-group assessment.

- [ ] **Step 2: Run diagnostics tests and confirm no migration assessment exists.**

Run:

```bash
python3 -m unittest rtp_llm.test.perf_test.cache_grid.tests.pressure_diagnostics_test -v
```

Expected: FAIL because `PressureCoverage.assess_migration` and result dataclasses are absent.

- [ ] **Step 3: Add immutable migration result types and the outer uncertainty interval.**

Add:

```python
@dataclass(frozen=True)
class MigrationGroupAssessment:
    owner_rank: int
    owner_group_id: int
    d2h_intervals: tuple[Interval, ...]
    h2d_intervals: tuple[Interval, ...]
    valid: bool


@dataclass(frozen=True)
class MigrationAssessment:
    epoch_valid: bool
    groups: tuple[MigrationGroupAssessment, ...]
    valid: bool
    reasons: tuple[str, ...]


def possible_event_segment(
    start_server: float,
    end_server: float,
    start_bounds: tuple[float, float],
    end_bounds: tuple[float, float],
) -> Interval:
    return (
        start_server + min(start_bounds[0], end_bounds[0]),
        end_server + max(start_bounds[1], end_bounds[1]),
    )
```

`possible_event_segment` is intentionally the outer event envelope. Do not use `confirmed_segment()`, whose inner interval is correct for continuous coverage but would overstate certainty about when a counter increment occurred.

- [ ] **Step 4: Record bounded owner-group direction intervals during `PressureCoverage.add()`.**

Initialize one list for each unique `(owner_rank, owner_group_id, direction)` from frozen targets. For every adjacent, complete, topology-valid owner sample:

1. Sum `published_blocks` counters matching the owner group.
2. D2H accepts operation 2 or 3 with source tier 1 and target tier 2.
3. H2D accepts only operation 1 with source tier 2 and target tier 1.
4. If a cumulative direction total increased, append its `possible_event_segment`.
5. Keep at most `max_segments` intervals per key; set `history_overflow` if an old interval is dropped.

Use one unique logical owner-group key even when multiple participant targets point to it. Existing inventory validation must complete before adding an interval, so a missing participant cannot certify the event.

- [ ] **Step 5: Implement conservative window assessment.**

Implement:

```python
def assess_migration(self, start: float, end: float) -> MigrationAssessment:
    coverage([], start, end)
    groups = []
    reasons = set()
    for owner_rank, owner_group_id in self.owner_groups:
        d2h = tuple(
            interval
            for interval in self.migration_intervals[(owner_rank, owner_group_id, "d2h")]
            if start <= interval[0] and interval[1] <= end
        )
        h2d = tuple(
            interval
            for interval in self.migration_intervals[(owner_rank, owner_group_id, "h2d")]
            if start <= interval[0] and interval[1] <= end
        )
        if not d2h:
            reasons.add(f"missing_d2h:{owner_rank}:{owner_group_id}")
        if not h2d:
            reasons.add(f"missing_h2d_load:{owner_rank}:{owner_group_id}")
        groups.append(
            MigrationGroupAssessment(
                owner_rank, owner_group_id, d2h, h2d, bool(d2h and h2d)
            )
        )
    valid = self.epoch_valid and bool(groups) and all(group.valid for group in groups)
    return MigrationAssessment(
        self.epoch_valid, tuple(groups), valid, tuple(sorted(reasons))
    )
```

If the epoch reset or bounded history overflow can hide required evidence, return `valid=False` and retain the epoch reason.

- [ ] **Step 6: Add migration evidence to the terminal session result.**

After the measurement pressure assessment, add:

```python
result["migration"] = asdict(
    self.coverage.assess_migration(
        self.state.measurement_start, self.state.measurement_end
    )
)
```

Drain must continue polling through the measurement end exactly as it does for pressure coverage; transfers observed only during seed/fill/support remain outside this assessment.

- [ ] **Step 7: Run diagnostics and session tests.**

Run:

```bash
python3 -m unittest \
  rtp_llm.test.perf_test.cache_grid.tests.pressure_diagnostics_test \
  rtp_llm.test.perf_test.cache_grid.tests.continuous_pressure_test -v
```

Expected: PASS for both directions, window boundaries, owner deduplication, missing participants, resets, and session serialization.

- [ ] **Step 8: Commit window-local migration qualification.**

```bash
git add \
  rtp_llm/test/perf_test/cache_grid/runner/pressure_diagnostics.py \
  rtp_llm/test/perf_test/cache_grid/runner/continuous_pressure.py \
  rtp_llm/test/perf_test/cache_grid/tests/pressure_diagnostics_test.py \
  rtp_llm/test/perf_test/cache_grid/tests/continuous_pressure_test.py
git diff --cached --check
git commit -m "feat(cache-grid): qualify bidirectional cache migration"
```

### Task 6: Split Formal Validity and Persist Seed Outcomes

**Files:**
- Modify: `rtp_llm/test/perf_test/cache_grid/runner/pressure_results.py:346-560`
- Modify: `rtp_llm/test/perf_test/cache_grid/runner/result_schema.py:191-209`
- Modify: `rtp_llm/test/perf_test/cache_grid/tests/pressure_results_test.py`
- Modify: `rtp_llm/test/perf_test/cache_grid/tests/result_schema_test.py`

**Interfaces:**
- Consumes: frozen event `seed_plan`, terminal result `seed`, `pressure`, and `migration`.
- Produces: summary validity fields `execution_valid`, `full_pressure_valid`, `formal_config_valid`, `bidirectional_migration_valid`, and `formal_acceptance_valid`; manifest field `seed_outcome`.

- [ ] **Step 1: Write failing independent-validity tests.**

Drive `PressureEvidence` with a frozen event and four terminal variants. The passing fixture is:

```python
seed_plan = dict(
    seed_count=2,
    formal_config_valid=True,
    nonformal_reasons=[],
)
finished = dict(
    event="finished",
    failure=None,
    seed=dict(planned=2, submitted=2, completed=2, failed=0, cancelled=0, pending=0),
    pressure=dict(valid=True),
    migration=dict(valid=True, reasons=[]),
)
```

Assert:

```python
self.assertEqual(
    summary["validity"],
    {
        "execution_valid": True,
        "full_pressure_valid": True,
        "formal_config_valid": True,
        "bidirectional_migration_valid": True,
        "formal_acceptance_valid": True,
        "pressure_valid": {
            "measurement": True,
            "profile": None,
            "support": None,
        },
        "reuse_valid": None,
        "trace_valid": None,
        "trace_applicability": "not_applicable",
    },
)
```

Add one test per independent false gate and one test where planned/completed seeds differ. Confirm a D2H-only migration failure does not rewrite `execution_valid` or `full_pressure_valid`.

- [ ] **Step 2: Run result tests and confirm the new fields are absent.**

Run:

```bash
python3 -m unittest \
  rtp_llm.test.perf_test.cache_grid.tests.pressure_results_test \
  rtp_llm.test.perf_test.cache_grid.tests.result_schema_test -v
```

Expected: FAIL because current validity contains only execution and nested pressure/trace fields.

- [ ] **Step 3: Persist the frozen plan and terminal seed outcome.**

On the `frozen` event, require and copy `event["seed_plan"]` into `writer.manifest["seed_plan"]`. On `finished`, copy the terminal seed object to `writer.manifest["seed_outcome"]` before `finish()` writes `complete=true`. Keep the raw frozen/session event in the existing manifest for backward auditability.

Treat `planned != completed`, nonzero failed/cancelled/pending, or missing seed outcome as execution invalid. Do not turn a manual smoke plan into formal merely because all requests completed.

- [ ] **Step 4: Calculate the four gates and aggregate acceptance explicitly.**

Use:

```python
seed_ok = (
    seed.get("planned") == seed.get("completed")
    and seed.get("failed") == 0
    and seed.get("cancelled") == 0
    and seed.get("pending") == 0
)
execution_valid = (
    not self.execution_failed and seed_ok and (not failure or profile_failure)
)
full_pressure_valid = event.get("pressure", {}).get("valid") is True
formal_config_valid = seed_plan.get("formal_config_valid") is True
bidirectional_migration_valid = event.get("migration", {}).get("valid") is True
formal_acceptance_valid = all(
    (
        execution_valid,
        full_pressure_valid,
        formal_config_valid,
        bidirectional_migration_valid,
    )
)
```

Retain the nested `pressure_valid` field for existing profile/support consumers. Add the four top-level gates and aggregate field; do not overload `reuse_valid` or `trace_valid`.

- [ ] **Step 5: Update result-schema accessors without admitting diagnostic rows to formula fits.**

If `measurement_status()` or validation helpers enumerate validity keys, recognize the new booleans. Preserve `execution_mode="full_pressure_diagnostic_v1"`, empty `metrics`, and the existing exclusion from exact-reuse fitting.

- [ ] **Step 6: Run result/schema tests and inspect serialized JSON.**

Run the Step 2 command again. Expected: PASS, and `json.dumps(..., allow_nan=False)` succeeds for manifest and summary fixtures.

- [ ] **Step 7: Commit the explicit formal validity schema.**

```bash
git add \
  rtp_llm/test/perf_test/cache_grid/runner/pressure_results.py \
  rtp_llm/test/perf_test/cache_grid/runner/result_schema.py \
  rtp_llm/test/perf_test/cache_grid/tests/pressure_results_test.py \
  rtp_llm/test/perf_test/cache_grid/tests/result_schema_test.py
git diff --cached --check
git commit -m "feat(cache-grid): report independent formal pressure gates"
```

### Task 7: Add Production/Smoke Assets and Chinese Operator Guidance

**Files:**
- Modify: `rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_dsv4.example.jsonc`
- Create: `rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_dsv4.production.example.jsonc`
- Modify: `rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_grid.example.json`
- Create: `rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_grid.production.example.json`
- Modify: `rtp_llm/test/perf_test/cache_grid/docs/continuous_full_pressure_guide_zh.md`
- Modify: `rtp_llm/test/perf_test/cache_grid/docs/reference.md:450-537,855-878`
- Modify: `rtp_llm/test/perf_test/cache_grid/docs/continuous_full_pressure_plan.md`
- Modify: `rtp_llm/test/perf_test/cache_grid/tests/cache_perf_test.py`

**Interfaces:**
- Consumes: the public CLI/defaults and result schema from Tasks 1-6.
- Produces: one explicitly non-formal reduced smoke pair, one production-formal pair, and runnable Chinese commands that omit manual seed count for formal execution.

- [ ] **Step 1: Write failing profile/grid resolution tests.**

Load both example pairs through `load_profile()` and `load_cases()`. Assert the production pair has:

```python
self.assertEqual(profile["engine"]["checkpoint_path"], "/mnt/hf3fs/3fs/models/DeepSeek-V4-Flash")
self.assertNotIn("--test_block_num", profile["engine_args"])
self.assertNotIn("--dsv4_fixed_pool_blocks", profile["engine_args"])
self.assertGreaterEqual(profile["engine"]["max_seq_len"], 102401)
self.assertGreaterEqual(max(case["cache_len"] for case in cases), 32768)
```

Build a dry-run plan with no scheduler, concurrency, or seed overrides and assert FIFO, client depth 16, no seed-count flag, `formal_candidate=true`, and no static candidate reasons. Build the smoke pair with explicit concurrency/seed 2 and assert `formal_candidate=false`, the launch snapshot retains `test_block_num=128`, and the reasons enumerate every known static disqualifier. These dry-run fields are preliminary; only the live plan can produce `formal_config_valid`.

- [ ] **Step 2: Run the asset tests and confirm production files are absent.**

Run:

```bash
python3 -m unittest rtp_llm.test.perf_test.cache_grid.tests.cache_perf_test -v
```

Expected: FAIL because the production profile/grid files do not exist.

- [ ] **Step 3: Label the existing reduced assets as smoke and add the production grid.**

Keep the current small profile executable, but change its label/description to state `NON-FORMAL GPU smoke`, add `seed_timeout_s: 300`, and retain `test_block_num=128`, fixed pools, 2-way limits, and the 16K fill override.

Create `continuous_full_pressure_grid.production.example.json` with this exact case set:

```json
{
  "schema_version": 2,
  "kind": "continuous_full_pressure_formal",
  "description": "DeepSeek-V4-Flash FIFO production FULL pressure workload",
  "generator": {
    "name": "continuous_pressure_formal_acceptance",
    "version": 2,
    "alignment": 128,
    "cache_alignment": 4096,
    "commit_tail_constraint": {"tokens": 4096}
  },
  "cases": [
    {"case_id": 0, "batch_size": 1, "input_len": 32768, "cache_len": 0},
    {"case_id": 1, "batch_size": 1, "input_len": 36864, "cache_len": 32768},
    {"case_id": 2, "batch_size": 1, "input_len": 65536, "cache_len": 0},
    {"case_id": 3, "batch_size": 1, "input_len": 69632, "cache_len": 65536}
  ]
}
```

- [ ] **Step 4: Create the production DeepSeek-V4-Flash profile.**

Copy the model/toolchain/environment sections from the smoke profile, then apply these exact capacity/workload changes:

```json
{
  "label": "DeepSeek-V4-Flash FIFO production FULL pressure",
  "engine": {
    "max_seq_len": 131072,
    "max_batch_tokens_size": 131072,
    "max_context_batch_size": 16,
    "concurrency_limit": 16,
    "seq_size_per_block": 512,
    "kernel_seq_size_per_block": 128
  },
  "engine_args": {
    "--enable_device_cache": 1,
    "--enable_memory_cache": 1,
    "--memory_cache_size_mb": 65536,
    "--block_tree_device_evict_low_watermark_ratio": 0.82,
    "--block_tree_device_evict_high_watermark_ratio": 0.9
  },
  "cache_grid": {
    "cache_alignment": 4096,
    "commit_tail_tokens": 4096,
    "pressure": {
      "fill_timeout_s": 1800,
      "fill_max_requests": 1000,
      "fill_no_progress_s": 300,
      "seed_timeout_s": 3600,
      "gpu_tolerance_blocks": 2,
      "activity_timeout_s": 300,
      "drain_timeout_s": 300,
      "capture_timeout_s": 600,
      "nsys_duration_s": 10
    }
  }
}
```

Retain the smoke profile's model path, TP4/EP4/DP1 topology, compiler/runtime paths, cache enablement, FP8 KV configuration, and CUDA device selection. Omit both `--test_block_num` and `--dsv4_fixed_pool_blocks`; omit `fill_input_tokens` to exercise the 102400 default.

- [ ] **Step 5: Rewrite the Chinese guide around formal automatic planning and an explicit smoke section.**

The formal baseline command must be exactly this shape and must not contain `--pressure-seed-count`:

```bash
python3 tools/cache_perf run \
  --profile rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_dsv4.production.example.jsonc \
  --grid rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_grid.production.example.json \
  --result-dir /data0/zhouhaiyu.zhy/tmp/dsv4-formal-pressure \
  --pressure-mode continuous --pressure-preload \
  --pressure-client-concurrency 16 \
  --pressure-request-rate 0 --pressure-arrival constant \
  --pressure-measure-seconds 60
```

Document the automatic formula, owner-pool deduplication, reserved block interpretation, 32K grid requirement, 102400-token unique fill, rolling seed flow, four validity fields, and evidence files. Add the smoke command using the existing reduced profile/grid plus `--pressure-client-concurrency 2 --pressure-seed-count 2`, and state that its result must be non-formal.

- [ ] **Step 6: Update reference and source-plan decisions.**

In `reference.md`, make `fill_input_tokens=102400` the default formal target, add required `seed_timeout_s`, describe the seed-plan manifest schema, and replace the old claim that client concurrency is bounded by active execution capacity. Reclassify the prior 127-block evidence as reduced smoke and state that old fixed-batch pressure acceptance does not satisfy this FIFO-only capability.

In `continuous_full_pressure_plan.md`, mark manual formal seed sizing, fixed-batch preload comparison, padding, and the old seed batch barrier as superseded by `refine-continuous-pressure-seeding`. Preserve historical evidence without presenting it as current formal acceptance.

- [ ] **Step 7: Verify profiles, grids, CLI help, dry-run plans, and documentation references.**

Run:

```bash
python3 -m unittest rtp_llm.test.perf_test.cache_grid.tests.cache_perf_test -v
python3 tools/cache_perf --help
python3 tools/cache_perf run \
  --profile rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_dsv4.production.example.jsonc \
  --grid rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_grid.production.example.json \
  --result-dir /tmp/dsv4-pressure-plan-check \
  --pressure-mode continuous --pressure-preload --dry-run
rg -n "127.*production|fixed batch.*formal|pressure-seed-count" \
  rtp_llm/test/perf_test/cache_grid/docs
```

Expected: tests and dry-run PASS; help lists seed timeout; search results mention 127 only as smoke/history and manual seed count only in the smoke/debug instructions.

- [ ] **Step 8: Commit production assets and operator documentation.**

```bash
git add \
  rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_dsv4.example.jsonc \
  rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_dsv4.production.example.jsonc \
  rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_grid.example.json \
  rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_grid.production.example.json \
  rtp_llm/test/perf_test/cache_grid/docs/continuous_full_pressure_guide_zh.md \
  rtp_llm/test/perf_test/cache_grid/docs/reference.md \
  rtp_llm/test/perf_test/cache_grid/docs/continuous_full_pressure_plan.md \
  rtp_llm/test/perf_test/cache_grid/tests/cache_perf_test.py
git diff --cached --check
git commit -m "docs(cache-grid): add formal FIFO pressure workflow"
```

### Task 8: Run Integrated Tests and Two-Tier DeepSeek GPU Acceptance

**Files:**
- Create: `rtp_llm/test/perf_test/cache_grid/docs/continuous_full_pressure_acceptance_zh.md`
- Modify: `rtp_llm/test/perf_test/cache_grid/docs/reference.md`
- Modify: `openspec/changes/refine-continuous-pressure-seeding/tasks.md`

**Interfaces:**
- Consumes: all implementation tasks, both example pairs, the local model directory, and generated pressure artifacts.
- Produces: passing focused test suite, non-formal reduced smoke evidence, production formal evidence, and a checked-in acceptance record with reproducible commands and artifact paths.

- [ ] **Step 1: Run the complete focused Python suite once.**

Run:

```bash
python3 -m unittest \
  rtp_llm.test.perf_test.cache_grid.tests.perf_profile_test \
  rtp_llm.test.perf_test.cache_grid.tests.cache_perf_test \
  rtp_llm.test.perf_test.cache_grid.tests.cache_perf_pipeline_test \
  rtp_llm.test.perf_test.cache_grid.tests.pressure_seed_plan_test \
  rtp_llm.test.perf_test.cache_grid.tests.continuous_pressure_test \
  rtp_llm.test.perf_test.cache_grid.tests.pressure_diagnostics_test \
  rtp_llm.test.perf_test.cache_grid.tests.pressure_results_test \
  rtp_llm.test.perf_test.cache_grid.tests.result_schema_test \
  rtp_llm.test.perf_test.batch_decode_test_test -v
```

Expected: PASS with no skipped tests for the newly added planner, rolling seed, or migration cases.

- [ ] **Step 2: Run the full cache-grid Python regression suite.**

Run:

```bash
python3 -m unittest discover \
  -s rtp_llm/test/perf_test/cache_grid/tests \
  -p '*_test.py' -v
```

Expected: PASS. Fix any regression before using GPU time.

- [ ] **Step 3: Verify model and four-GPU prerequisites, then run the reduced smoke.**

Run:

```bash
test -f /mnt/hf3fs/3fs/models/DeepSeek-V4-Flash/config.json
nvidia-smi
SMOKE_ROOT=/data0/zhouhaiyu.zhy/tmp/dsv4-fifo-pressure-smoke-20261021
python3 tools/cache_perf run \
  --profile rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_dsv4.example.jsonc \
  --grid rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_grid.example.json \
  --result-dir "$SMOKE_ROOT" \
  --pressure-mode continuous --pressure-preload \
  --pressure-client-concurrency 2 --pressure-seed-count 2 \
  --pressure-request-rate 0 --pressure-arrival constant \
  --pressure-measure-seconds 10
```

Expected: request lifecycle and cleanup complete; manifest reasons include manual seed, client depth below 16, synthetic test capacity, and nonstandard fill target; `formal_config_valid=false` and `formal_acceptance_valid=false` even if execution/pressure pass.

- [ ] **Step 4: Run the production-capacity formal acceptance.**

Ensure GPUs `2,3,4,5` are free as selected by the profile; if the available four-card set differs, override only `CUDA_VISIBLE_DEVICES` and record the actual value. Run:

```bash
FORMAL_ROOT=/data0/zhouhaiyu.zhy/tmp/dsv4-fifo-pressure-formal-20261021
python3 tools/cache_perf run \
  --profile rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_dsv4.production.example.jsonc \
  --grid rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_grid.production.example.json \
  --result-dir "$FORMAL_ROOT" \
  --env CUDA_VISIBLE_DEVICES=2,3,4,5 \
  --pressure-mode continuous --pressure-preload \
  --pressure-client-concurrency 16 \
  --pressure-request-rate 0 --pressure-arrival constant \
  --pressure-measure-seconds 60
```

Expected: no `test_block_num` in the launch snapshot; startup evidence reports actual allocatable FULL owner capacity; automatic seed count is feasible and completes; measurement records at least one successful D2H publication and H2D LOAD publication for every required owner group.

- [ ] **Step 5: Audit the production artifacts and independently recalculate the plan.**

Resolve the epoch path from `pressure_latest.json`, then inspect:

```bash
jq . "$FORMAL_ROOT/pressure_latest.json"
find "$FORMAL_ROOT/pressure_epochs" -maxdepth 3 -type f -print | sort
jq '.seed_plan, .seed_outcome, .validity' \
  "$FORMAL_ROOT"/pressure_epochs/*/pressure/manifest.json
jq '.validity, .result.seed, .result.pressure, .result.migration' \
  "$FORMAL_ROOT"/pressure_epochs/*/pressure/summary.json
```

For each serialized group, verify `seed_count >= gpu_family_capacity + 1` and `seed_count <= maximum_families`; confirm planned/completed counts match and all four gates plus aggregate formal acceptance are true. Inspect `requests.jsonl` to confirm `max_inflight >= 16`, and `cache_samples.jsonl` to confirm both publication directions occur inside the measurement timestamps.

- [ ] **Step 6: Record exact evidence and any environment substitutions in Chinese.**

Create `continuous_full_pressure_acceptance_zh.md` with:

- source commit and dirty-state note;
- exact smoke/formal commands and CUDA device selection;
- model/profile/grid hashes;
- startup FULL owner pool capacities and TP participant mapping;
- planner lower/upper bounds, selected count, and completed count;
- measurement timestamps, coverage, longest invalid interval, D2H/H2D deltas;
- all validity booleans and direct paths to retained artifacts;
- failures/retries as failures, rather than replacing them with only the successful attempt.

Update the validated-runtime section of `reference.md` to distinguish the new production formal run from the prior reduced smoke evidence.

- [ ] **Step 7: Validate OpenSpec completion and repository formatting.**

After checking off only tasks whose implementation and evidence passed, run:

```bash
openspec validate refine-continuous-pressure-seeding --strict
git diff --check
git status --short
```

Expected: OpenSpec validation passes, no whitespace errors, and the status contains only intended implementation, plan, OpenSpec, documentation, and evidence changes.

- [ ] **Step 8: Commit the acceptance record.**

```bash
git add \
  rtp_llm/test/perf_test/cache_grid/docs/continuous_full_pressure_acceptance_zh.md \
  rtp_llm/test/perf_test/cache_grid/docs/reference.md \
  openspec/changes/refine-continuous-pressure-seeding/tasks.md
git diff --cached --check
git commit -m "test(cache-grid): validate production FIFO cache migration"
```

## Completion Criteria

- Automatic plans are derived from the live owner topology after startup and can be recalculated from `manifest.json`.
- The formal reusable bank exceeds every required FULL GPU family capacity and fits the combined low-water GPU plus live host retention bound.
- Seed initialization maintains bounded rolling FIFO supply, completes exactly the planned families, and has an independent timeout/cleanup path.
- Full-pressure preload rejects fixed-batch scheduling at public, direct, runner, and session boundaries while ordinary batch tests continue to pass.
- The result schema independently reports execution, FULL pressure, formal configuration, and bidirectional migration validity.
- Reduced DeepSeek smoke is explicitly non-formal; the production run has no synthetic block limit and passes both D2H and H2D measurement gates.
- The Chinese guide's commands match `--help`, dry-run successfully, and point to the generated formal/smoke assets.

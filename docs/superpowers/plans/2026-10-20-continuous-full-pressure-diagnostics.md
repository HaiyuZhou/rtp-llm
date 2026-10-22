# Continuous FULL Pressure Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish real FULL-attention GPU/host cache pressure, sustain prefill supply, and produce trustworthy baseline and continuous-profile diagnostics.

**Architecture:** Cache settlement provides structured per-rank pressure evidence through additive diagnostic RPCs. A bounded client phase machine consumes that evidence while independent sampling/profile controllers operate beside the sender. An opt-in executor recorder supplies actual step membership and wall/thread-CPU timing; diagnostic artifacts remain separate from exact-reuse metrics.

**Tech Stack:** Existing C++ cache/engine, protobuf/gRPC, Python standard-library dataclasses/concurrency/unittest, Bazel/gtest, PyTorch Kineto, and Nsight Systems/NVTX.

**Spec:** `openspec/changes/add-continuous-full-pressure-diagnostics/design.md`; `specs/cache-pressure-observability/spec.md`, `specs/continuous-cache-pressure/spec.md`, and `specs/prefill-step-diagnostics/spec.md` beneath that same change directory. The approved narrative is `rtp_llm/test/perf_test/cache_grid/docs/continuous_full_pressure_plan.md`.

## Global Constraints

- Full-pressure diagnostics SHALL require explicit opt-in and retain DP=1, exclusive service, ungrouped batch_size=1 grid, input-ID transport, known block size, and complete commit-tail constraints.
- Fixed workspace, materialized inputs, and shared-seed mode SHALL remain incompatible.
- The run SHALL require every target TP rank's FULL groups to pass independently.
- Supported shared TP metadata ownership SHALL be explicitly declared: retain raw local snapshots, attribute shared slot/publication evidence to its owner and all-participant completion scope, freeze every participant's identity/layout/capacity, and intersect owner-clock pressure with all participant freshness. Missing peers or incompatible layouts SHALL NOT be replaced by owner values.
- SWA/LINEAR SHALL be observed without capacity gates.
- Published host residency SHALL default to at least ceil(host_capacity * 0.25).
- Sampling SHALL default to 1 second with maximum sample gap and age of 2 seconds.
- Default acceptance SHALL require at least 95% valid duration and at most 2 seconds of consecutive invalid/unknown time.
- Steady SHALL default to 3 continuous seconds of valid FULL pressure coverage and a total wait limit of 60 seconds from entry into the formal distribution.
- The default measurement SHALL last 60 seconds using [t0, t1), mutually exclusive with request-count stopping.
- Continuous Kineto SHALL default to 20 actual nonempty execution steps on every target rank.
- Unfinished profiling SHALL receive same-distribution support for at most 60 additional seconds by default, subject to earlier capture deadlines.
- Kineto and Nsight SHALL not collect CUDA activity simultaneously.
- No implementation of this plan changes the C++ block-tree benchmark or replaces the real scheduler.
- All code blocks below are instructions for future implementation, not claims that these APIs already exist. Existing integration points are identified explicitly; proposed interfaces are defined in the task that introduces them.

---

## Execution Map and Prerequisites

This is one dependent feature delivered in three milestones: evidence/API (Tasks 1-3), pressure baseline (Tasks 4-9), and step/profile diagnosis (Tasks 10-14). Task 15 validates the whole workflow. Keep the original OpenSpec task boxes pending until their actual deliverables and tests pass.

Runtime integration addendum (2026-10-21, Task 15): actual Flash TP startup demonstrated that only TP0 owns mutable cache metadata while all ranks use synchronized block IDs. Task 15 must implement the explicit shared-owner evidence contract now recorded in the linked specs/design, plus the missing optional scheduler waiting/loading count producer. Preserve raw peer counters and every rank's validation; do not require replicated trees, infer ownership from zeros, or lower acceptance gates. Cover ownership/scope/layout/instance failures and owner-clock coverage before repeating GPU acceptance. Earlier task completion records do not imply this newly discovered integration requirement has passed.

Kineto runtime addendum (2026-10-21, Task 15): real four-rank traces total over 1GB, and online JSON parsing starves Python sampling even in a separate thread. Online artifact waiting must only check completion, identity and readiness. Perform load/validation after drain in a terminable, reaped child within the original capture deadline's remaining budget; retain raw artifacts on timeout and never extend measurement/support. Uniquely CUDA-runtime-correlated GPU work launched outside all forward/step scopes is separate non-forward activity. Missing/ambiguous correlations and forwards without correctly associated GPU work remain invalid. Cover lifecycle, timeout/cleanup and association negative cases, then validate the retained real traces and repeat the independent GPU capture.

Nsight runtime addendum (2026-10-21, Task 15): the production report exceeded the fixed five-second analysis allowance and its daemon kept writing after timeout. Move post-drain normalization/serialization into a terminable, reaped exec process using the original capture deadline's remaining budget. Preserve raw and explicitly partial outputs on failure; measure phase durations/artifact sizes and verify real rank markers and counts before rerunning capture. Do not extend measurement, support or capture windows.

Nsight duration correction (Task 15): count the requested duration from successful start acknowledgement, retaining the earlier start-call timestamp only for the conservative pressure envelope. Startup command latency must not consume the requested workload collection window; original capture/support deadlines still apply. Cover slow acknowledgement and deadline interruption, and verify actual backend timing in the production report.

Nsight clock correction (Task 15): observed exporter tail NVTX ranges retain absolute clocks while internal events are relative. Only normalize when unique consistent session/analysis metadata places the full raw range inside the absolute capture interval and outside the relative interval. Preserve raw values/provenance; validate CUDA links but exclude repaired boundary steps from usable pairs. Repaired internal steps, missing/inconsistent metadata and ambiguous/out-of-range bounds remain invalid. Test these negatives and the retained real report; do not infer repair from magnitude or an arbitrary tolerance.

Commands and paths beginning with `rtp_llm/`, `openspec/`, or `tools/` are relative to repository root. In task file lists, `runner/`, `config/`, `tests/`, and `docs/reference.md` abbreviate paths beneath `rtp_llm/test/perf_test/cache_grid/`; C++ sibling filenames are relative to the fully qualified directory in that task. In commands, `--config=cuda13` is the existing x86 CUDA configuration, not a statement about the current host. Execute C++ tests on a configured CUDA13 worker matching the BUILD target's GPU tags. Use the corresponding existing platform configuration when targeting another supported architecture and record that substitution. Do not bypass build failures with ad hoc compiler commands. This plan-writing turn does not run builds or GPU experiments.

Read the existing worktree state before execution; do not replace unrelated files or stage the whole repository. For each task, commit only its listed files and necessary owning BUILD changes, after reviewing the staged diff. Commit commands below assume those files contain only the intended task changes; otherwise stage the intended hunks. Documentation updates belong to their implementation task, not only the final acceptance task.

| Files to create | Responsibility |
| --- | --- |
| `rtp_llm/cpp/cache/block_tree_cache/CachePressureState.{h,cc}` | Bounded residency ledger and cumulative transfer/trigger evidence |
| `rtp_llm/cpp/cache/block_tree_cache/test/CachePressureStateTest.cc` | Ledger semantics without a model |
| `rtp_llm/cpp/engine_base/PressureDiagnostics.h` | Engine-facing value types and control/result contracts |
| `rtp_llm/cpp/normal_engine/PressureStepRecorder.{h,cc}` | Bounded step records, cursors, and session state |
| `rtp_llm/cpp/normal_engine/test/PressureStepRecorderTest.cc` | Recorder, timing, and overflow tests |
| `rtp_llm/test/perf_test/cache_grid/runner/pressure_diagnostics.py` | Frozen settings, decoded samples, target selection, coverage |
| `rtp_llm/test/perf_test/cache_grid/runner/continuous_pressure.py` | Bounded producer and phase transitions |
| `rtp_llm/test/perf_test/cache_grid/runner/pressure_results.py` | Incremental artifacts, cohort metrics, step classification |
| `rtp_llm/test/perf_test/cache_grid/runner/pressure_profile.py` | Async capture control, trace validation, attribution |
| `rtp_llm/test/perf_test/cache_grid/tests/pressure_diagnostics_test.py` | Config/predicate/interval tests |
| `rtp_llm/test/perf_test/cache_grid/tests/continuous_pressure_test.py` | Producer, phases, cancellation, no-model integration |
| `rtp_llm/test/perf_test/cache_grid/tests/pressure_results_test.py` | Schema, attribution, resume, and fitting isolation |
| `rtp_llm/test/perf_test/cache_grid/tests/pressure_profile_test.py` | Profile lifecycle and synthetic trace tests |

`pressure_results.py` is a responsibility split of the design's runner logic, not an additional product capability. Reuse existing packaging and BUILD helpers. Python test snippets use `unittest`; place functions inside the named `TestCase` when shown indented as methods. C++ snippets go inside `namespace rtp_llm` and use the listed production headers and gtest.

### Task 1: Settlement Evidence Ledger

**Files:** Create `rtp_llm/cpp/cache/block_tree_cache/CachePressureState.h`, `.cc`, and `test/CachePressureStateTest.cc`; modify the owning `BUILD` and `test/BUILD`.

**Interfaces:** Consumes stable group IDs, physical pool IDs, node cache keys, and settled ownership events. Produces `CachePressureState`, `PoolEvidence`, and `TransferEvidence` below. The cache mutex protects ownership methods; a private ledger mutex protects copy callbacks and snapshot copies. Lock order is cache mutex then ledger mutex, never the reverse.

```cpp
struct PoolEvidence {
    uint64_t published_host_blocks{0};
};
struct TransferEvidence {
    uint64_t copied_blocks{0}, copied_bytes{0};
    uint64_t published_blocks{0}, published_bytes{0}, failed_operations{0};
};
using PressureOwner = std::pair<size_t, int64_t>;  // group, node cache key
using PressureBlock = std::pair<std::string, int64_t>;  // physical pool, block
// group, physical pool, operation, source tier, target tier
using PressureTransferKey =
    std::tuple<size_t, std::string, CacheTransferOperation, Tier, Tier>;

class CachePressureState {
public:
    void setHostOwner(PressureOwner owner, std::optional<PressureBlock> block);
    PoolEvidence pool(const std::string& pool_id) const;
    void recordCopy(const PressureTransferKey& key, uint64_t blocks,
                    uint64_t bytes, bool success);
    void recordPublication(const PressureTransferKey& key, uint64_t blocks,
                           uint64_t bytes);
    TransferEvidence transfer(const PressureTransferKey& key) const;
    void recordWatermarkTrigger(size_t group_id);
    uint64_t watermarkTriggers(size_t group_id) const;
};
```

- [ ] **Step 1: Add failing ledger tests.** Include the new header and `<gtest/gtest.h>`; the declarations above are introduced by implementation Step 3.

```cpp
TEST(CachePressureStateTest, ReservationsAndDetachedCopiesAreNotPublished) {
    CachePressureState state;
    PressureTransferKey key{0, "host-full", CacheTransferOperation::EVICT,
                            Tier::DEVICE, Tier::HOST};
    state.recordCopy(key, 1, 4096, true);
    EXPECT_EQ(state.transfer(key).copied_blocks, 1u);
    EXPECT_EQ(state.transfer(key).published_blocks, 0u);
    EXPECT_EQ(state.pool("host-full").published_host_blocks, 0u);
}

TEST(CachePressureStateTest, DeduplicatesPhysicalBlockAndRetainsCumulativeTotals) {
    CachePressureState state;
    PressureTransferKey key{0, "host-full", CacheTransferOperation::EVICT,
                            Tier::DEVICE, Tier::HOST};
    state.setHostOwner({0, 101}, PressureBlock{"host-full", 7});
    state.setHostOwner({1, 202}, PressureBlock{"host-full", 7});
    state.recordPublication(key, 1, 4096);
    EXPECT_EQ(state.pool("host-full").published_host_blocks, 1u);
    state.setHostOwner({0, 101}, std::nullopt);
    EXPECT_EQ(state.pool("host-full").published_host_blocks, 1u);
    state.setHostOwner({1, 202}, std::nullopt);
    EXPECT_EQ(state.pool("host-full").published_host_blocks, 0u);
    EXPECT_EQ(state.transfer(key).published_blocks, 1u);
}
```

- [ ] **Step 2: Register and run the focused test target.** Add `cache_pressure_state_test` beside the existing cache tests, using their local dependencies and copts. Run `bazel test --config=cuda13 //rtp_llm/cpp/cache/block_tree_cache/test:cache_pressure_state_test --test_output=errors`. Expected first failure: the new ledger implementation is absent, not an unrelated toolchain failure.

- [ ] **Step 3: Implement bounded state and counters.** Add private `owners_` (`std::map<PressureOwner, PressureBlock>`), `owners_per_block_` (`std::map<PressureBlock, uint64_t>`), `pools_`, `transfers_`, `triggers_`, and a mutable mutex. Topology keys are registered from existing pools/groups, not from arbitrary request strings. Ownership memory is bounded by resident resources and entries are erased when removed. The central mutation is:

```cpp
void CachePressureState::setHostOwner(
    PressureOwner owner, std::optional<PressureBlock> block) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto previous = owners_.find(owner);
    if (previous != owners_.end() && block && previous->second == *block) return;
    if (previous != owners_.end()) {
        const auto old = previous->second;
        auto count = owners_per_block_.find(old);
        if (--count->second == 0) {
            --pools_[old.first].published_host_blocks;
            owners_per_block_.erase(count);
        }
        owners_.erase(previous);
    }
    if (block) {
        if (owners_per_block_[*block]++ == 0) {
            ++pools_[block->first].published_host_blocks;
        }
        owners_.emplace(owner, *block);
    }
}
```

Implement copy/publication/trigger updates under the same mutex. A failed copy increments failed_operations; successful partial descriptors contribute copied blocks/bytes separately. Publication calls are emitted once per actual accepted ownership transition, not by the polling loop. Getters return zero for a registered key with no activity. Include `<map>`, `<mutex>`, `<optional>`, `<tuple>`, `<string>`, `<cstdint>`, the existing cache tier header, and `BlockTreeCacheMetricsReporter.h` for the existing LOAD/EVICT/STORE enum. Reuse that enum without depending on a metrics reporter instance; the new counters remain active when external metrics are disabled.

- [ ] **Step 4: Run the target again and add replacement/idempotence cases.** Verify changing an owner's block decrements the old pool exactly once, repeated owner updates are idempotent, copy failure cannot change residency, and removal leaves no per-block entry. Document the lock order in the header.
- [ ] **Step 5: Commit.**

```bash
git add rtp_llm/cpp/cache/block_tree_cache/CachePressureState.h rtp_llm/cpp/cache/block_tree_cache/CachePressureState.cc rtp_llm/cpp/cache/block_tree_cache/test/CachePressureStateTest.cc rtp_llm/cpp/cache/block_tree_cache/BUILD rtp_llm/cpp/cache/block_tree_cache/test/BUILD
git diff --cached --check
git commit -m "feat(cache): add pressure publication evidence ledger"
```

### Task 2: Real Cache Hooks and Structured Snapshots

**Files:** Modify `rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.{h,cc}`, `BlockTree.{h,cc}`, `evict/BlockTreeEvictor.{h,cc}`, `load/BlockTreeLoader.{h,cc}`, `store/BlockTreeStorer.{h,cc}`, transfer completion reporting, `test/BlockTreeCacheTest.cc`, and relevant eviction/load/store tests. Update `rtp_llm/test/perf_test/cache_grid/docs/reference.md`.

**Interfaces:** Consumes Task 1 ledger. Produces `CachePressureSnapshot BlockTreeCache::pressureSnapshot() const`, `uint64_t BlockTreeEvictor::pendingReleaseBlocks(const IBlockPool*) const`, and group/pool snapshot records. `CachePressureSnapshot` contains vectors of physical pool records, group mappings, and transfer counters; no request data or tree-key arrays. Pool records include stable ID, tier, capacity/used/free/available, block bytes, reference categories, pending release, and published residency. Group records include ID/type, physical pool IDs, watermarks, and trigger count.

- [ ] **Step 1: Extend the actual cache fixture with a failing residency test.** Add the helper returning the sum over unique HOST pool records, named `publishedHostBlocks(const CachePressureSnapshot&)`, to this test file. Use the existing `block_tree_cache_test::FullSWAEnvironment` APIs:

```cpp
TEST(BlockTreePressureTest, CountsSettledHostDataAndRemovesItOnReclaim) {
    auto env = block_tree_cache_test::FullSWAEnvironment::create();
    env->insertRequestPath();
    env->releaseRequestRefs();
    EXPECT_EQ(publishedHostBlocks(env->cache->pressureSnapshot()), 0u);
    env->demoteAll(Tier::DEVICE);
    EXPECT_GT(publishedHostBlocks(env->cache->pressureSnapshot()), 0u);
    env->reclaimAll();
    EXPECT_EQ(publishedHostBlocks(env->cache->pressureSnapshot()), 0u);
    env->expectFullyReclaimed();
}
```

- [ ] **Step 2: Run the fixture before implementation.** Run `bazel test --config=cuda13 //rtp_llm/cpp/cache/block_tree_cache/test:block_tree_cache_test --test_filter='BlockTreePressureTest.*' --test_output=errors`. Expected failure: the snapshot API is absent.
- [ ] **Step 3: Wire ownership changes, then snapshot under existing locks.** Keep the ledger owned by the cache/tree and inject references into collaborators, with destruction ordered after outstanding callbacks settle. Add a focused `refreshPressureHostOwner(const TreeNode*, size_t group_id)` method at tree ownership updates; it uses node key/group ID plus the actual host pool identity. Do not reuse `logicalCacheSnapshot()` or its event-publisher gate: that path describes device publication and returns early when no event publisher exists.

```cpp
// After a resource mutation, while the cache mutex is held:
const auto& resource = node->group_set_resources.at(group_id);
if (resource.transfer_detached || !resource.hasTier(Tier::HOST)) {
    pressure_state_.setHostOwner({group_id, node->cache_key}, std::nullopt);
} else {
    const auto& host_pool = group_sets_.at(group_id)->hostPool();
    pressure_state_.setHostOwner(
        {group_id, node->cache_key},
        PressureBlock{host_pool->poolName(), resource.host_block});
}
```

Use a generated stable pool ID if poolName is not unique; freeze that mapping once. Call this helper on accepted tree insertion, successful `completeEvict`, `completeDrop`, successful `settleLoadLocked`, detachment/invalidation, and before node deletion. Valid LOADING host sources stay counted until removed; unpublished targets are not installed in the resource. Count transfer publication only in the successful, non-detached settlement branch, using source/target physical block sizes. Count successful copies from the transfer result before any external-metrics early return. Count watermark triggers at a false-to-true triggered transition; expose pending releases with the existing pending-release lock. Do not decrement counters based on a failed copy.

Snapshot algorithm: acquire cache mutex; copy pool metrics and topology; read pending-release/ledger counters in the documented lock order; release locks; serialize outside locks. Complexity is O(pool + group count), not O(tree nodes). Verify pointer-deduplicated device pools and stable IDs agree with group mappings.

- [ ] **Step 4: Run lifecycle and eviction tests.** Run cache test plus `//rtp_llm/cpp/cache/block_tree_cache/evict/test:block_tree_evictor_test`, `:full_eviction_test`, and `:full_swa_eviction_test`. Extend existing detached/failure/load/store fixtures to assert new counters before reservation, after copy, after publication, after invalidation, and after reclaim. Document copy/publication/residency units and unavailable fields in reference.md.
- [ ] **Step 5: Commit reviewed cache hooks and documentation.**

```bash
git add rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.cc rtp_llm/cpp/cache/block_tree_cache/BlockTree.h rtp_llm/cpp/cache/block_tree_cache/BlockTree.cc rtp_llm/cpp/cache/block_tree_cache/evict/BlockTreeEvictor.h rtp_llm/cpp/cache/block_tree_cache/evict/BlockTreeEvictor.cc rtp_llm/cpp/cache/block_tree_cache/load/BlockTreeLoader.h rtp_llm/cpp/cache/block_tree_cache/load/BlockTreeLoader.cc rtp_llm/cpp/cache/block_tree_cache/store/BlockTreeStorer.h rtp_llm/cpp/cache/block_tree_cache/store/BlockTreeStorer.cc rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTest.cc rtp_llm/test/perf_test/cache_grid/docs/reference.md
git diff --cached --stat
git diff --cached --check
git commit -m "feat(cache): expose settlement-backed pressure snapshots"
```

Explicitly stage changed transfer runners, lifecycle tests, and owning BUILD files after reviewing their diffs. Stage only this task's changes; do not include the existing untracked benchmark plan.

### Task 3: Diagnostic Protocol, Engine Access, and Rank Fan-Out

**Files:** Modify `rtp_llm/cpp/model_rpc/proto/model_rpc_service.proto`, `rtp_llm/cpp/model_rpc/LocalRpcServer.{h,cc}`, `rtp_llm/cpp/engine_base/EngineBase.h`, `rtp_llm/cpp/normal_engine/NormalEngine.{h,cc}`, `rtp_llm/cpp/cache/KVCacheManager.{h,cc}`, `rtp_llm/cpp/model_rpc/test/LocalRpcServerTest.cc`, and owning BUILD files. Create `rtp_llm/cpp/engine_base/PressureDiagnostics.h`.

**Interfaces:** Add `GetCachePressure`/`GetCachePressureInternal` to `ModelRpcService`. Engine-facing `ErrorResult<CachePressureSnapshot> getCachePressure() const` defaults to unsupported; NormalEngine delegates to KVCacheManager. The public RPC returns one record per requested rank, including error records. The internal RPC reads only its local rank. Session mutation is added in Task 10; this task defines its wire types.

- [ ] **Step 1: Add the unsupported-engine test before adding the service methods.** Extend existing `TestLocalRpcServer` and its `FailingEngine` fixture:

```cpp
TEST(PressureRpcTest, UnsupportedEngineDoesNotReturnEmptySuccess) {
    TestLocalRpcServer server;
    server.setEngineForTest(std::make_shared<FailingEngine>(EngineFailureMode::REJECT_ALL));
    CachePressureRequestPB request;
    request.set_session_id("pressure-test");
    CachePressureResponsePB response;
    grpc::ServerContext context;
    const auto status = server.GetCachePressureInternal(&context, &request, &response);
    EXPECT_EQ(status.error_code(), grpc::StatusCode::UNIMPLEMENTED);
}
```

- [ ] **Step 2: Run `bazel test --config=cuda13 //rtp_llm/cpp/model_rpc/test:local_rpc_server_test --test_output=errors`.** Expected failure: missing message/method declarations. Do not treat a generated-code import/build-environment failure as the intended regression signal.
- [ ] **Step 3: Add versioned messages and conversion.** Use unused protobuf field numbers and existing generation rules. Define all nested records with typed fields, preserving optional activity via explicit presence. The envelope is:

```proto
message CachePressureRequestPB {
  string session_id = 1;
  repeated int32 target_ranks = 2;
  map<int32, uint64> after_step_sequence = 3;
  uint32 max_step_records = 4;
}
message CachePressureRankPB {
  int32 world_rank = 1;
  string service_instance = 2;
  uint64 sequence = 3;
  int64 sampled_monotonic_ns = 4;
  int64 collect_duration_ns = 5;
  string error = 6;
  repeated PressurePoolPB pools = 7;
  repeated PressureGroupPB groups = 8;
  repeated PressureTransferPB transfers = 9;
  repeated PressureStepPB steps = 10;
  uint64 next_step_sequence = 11;
  bool step_records_lost = 12;
  int64 coordinator_send_ns = 13;
  int64 coordinator_receive_ns = 14;
}
message CachePressureResponsePB {
  uint32 schema_version = 1;
  repeated int32 requested_ranks = 2;
  repeated CachePressureRankPB ranks = 3;
  int64 coordinator_received_ns = 4;
  int64 coordinator_sent_ns = 5;
}
```

Define `PressurePoolPB` from Task 2 pool fields; `PressureGroupPB` from group fields; `PressureTransferPB` from Task 1 key/count fields. Define `PressureStepPB` now with session ID, step ID, sequence, wall/CPU boundary timestamps, context/decode counts, tokens, request IDs, and completion state; Task 10 populates it. All timestamp fields use nanoseconds from the reported process clock. Use numeric tier/group enums consistent with existing cache types or explicit stable protocol enums, never C++ enum ordinal assumptions without conversion tests.

Follow `StartProfile` broadcaster usage with bounded per-peer deadlines, verify TP rank zero for public fan-out, and call only `GetCachePressureInternal` on peers. Validate duplicate/out-of-topology targets and cap page size. Include per-peer errors in an otherwise successful aggregate so partial evidence survives; a wholly unsupported local backend returns UNIMPLEMENTED. Do not collect keys or invoke snapshot conversion from ordinary health/status handlers.

- [ ] **Step 4: Verify serialization and fan-out cases.** Extend RPC tests with a fake engine returning a known pool/group snapshot; assert exact values and mappings. Add mixed success/missing rank, duplicate target, page cap, sequence reset, no-recursion, and ordinary-health no-snapshot tests. Run local RPC and worker-status test targets. Document the optional protocol and timing units in reference.md.
- [ ] **Step 5: Commit the additive API.**

```bash
git add rtp_llm/cpp/model_rpc/proto/model_rpc_service.proto rtp_llm/cpp/model_rpc/LocalRpcServer.h rtp_llm/cpp/model_rpc/LocalRpcServer.cc rtp_llm/cpp/model_rpc/test/LocalRpcServerTest.cc rtp_llm/cpp/engine_base/PressureDiagnostics.h rtp_llm/cpp/engine_base/EngineBase.h rtp_llm/cpp/normal_engine/NormalEngine.h rtp_llm/cpp/normal_engine/NormalEngine.cc rtp_llm/cpp/cache/KVCacheManager.h rtp_llm/cpp/cache/KVCacheManager.cc rtp_llm/test/perf_test/cache_grid/docs/reference.md
git diff --cached --check
git commit -m "feat(rpc): add opt-in per-rank cache pressure diagnostics"
```

Include changed owning BUILD files explicitly in that commit after inspecting their diff.

### Task 4: Frozen Python Contracts and FULL Predicate

**Files:** Create `rtp_llm/test/perf_test/cache_grid/runner/pressure_diagnostics.py` and `tests/pressure_diagnostics_test.py`; modify `rtp_llm/test/perf_test/BUILD` and `docs/reference.md`.

**Interfaces:** Produce frozen dataclasses `PressureConfig`, `PoolSample`, `GroupSample`, `RankSample`, and `PressureTarget`; `resolve_pressure_config(values: dict) -> PressureConfig`; `select_targets(samples: tuple[RankSample, ...], ranks: tuple[int, ...]) -> tuple[PressureTarget, ...]`; `group_valid(group: GroupSample, pools: dict[str, PoolSample], baseline: GroupSample, config: PressureConfig) -> bool`. Identity/schema validation happens before predicate evaluation.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class PressureConfig:
    fill_timeout_s: float
    fill_max_requests: int
    fill_no_progress_s: float
    fill_input_tokens: int
    gpu_tolerance_blocks: int
    activity_timeout_s: float
    drain_timeout_s: float
    capture_timeout_s: float
    host_ratio: float = 0.25
    sample_period_s: float = 1.0
    max_sample_gap_s: float = 2.0
    max_sample_age_s: float = 2.0
    steady_s: float = 3.0
    steady_timeout_s: float = 60.0
    measure_s: float = 60.0
    min_coverage: float = 0.95
    max_invalid_s: float = 2.0
    profile_steps: int = 20
    support_s: float = 60.0
    nsys_duration_s: float | None = None

@dataclass(frozen=True)
class PoolSample:
    pool_id: str
    capacity: int
    used: int
    pending_release: int
    low_ratio: float
    high_ratio: float
    host_resident: int = 0

@dataclass(frozen=True)
class GroupSample:
    group_id: int
    group_type: str
    device_pool_ids: tuple[str, ...]
    host_pool_id: str
    triggers: int
    gpu_host_published: int

@dataclass(frozen=True)
class RankSample:
    rank: int
    instance: str
    sequence: int
    server_s: float
    client_send_s: float
    client_receive_s: float
    pools: tuple[PoolSample, ...]
    groups: tuple[GroupSample, ...]

@dataclass(frozen=True)
class PressureTarget:
    rank: int
    instance: str
    group_id: int
    device_pool_ids: tuple[str, ...]
    host_pool_id: str
```

- [ ] **Step 1: Add failing defaults and gate tests.** In the unittest class use this complete explicit experiment settings fixture:

```python
def configured():
    return resolve_pressure_config(dict(
        fill_timeout_s=120, fill_max_requests=1000, fill_no_progress_s=30,
        fill_input_tokens=4096, gpu_tolerance_blocks=0,
        activity_timeout_s=30, drain_timeout_s=60, capture_timeout_s=120,
    ))

class PressurePredicateTest(unittest.TestCase):
    def test_full_gate_needs_publication_and_effective_occupancy(self):
        cfg = configured()
        pools = {
            "gpu": PoolSample("gpu", 100, 90, 15, 0.8, 0.9),
            "host": PoolSample("host", 100, 30, 0, 0.8, 0.9, 25),
        }
        base = GroupSample(0, "FULL", ("gpu",), "host", 0, 0)
        current = GroupSample(0, "FULL", ("gpu",), "host", 1, 1)
        self.assertFalse(group_valid(current, pools, base, cfg))
        pools["gpu"] = PoolSample("gpu", 100, 90, 0, 0.8, 0.9)
        self.assertTrue(group_valid(current, pools, base, cfg))
        self.assertFalse(group_valid(base, pools, base, cfg))
        self.assertEqual((cfg.steady_s, cfg.measure_s, cfg.profile_steps), (3, 60, 20))
```

- [ ] **Step 2: Run `python3 -m unittest rtp_llm.test.perf_test.cache_grid.tests.pressure_diagnostics_test -v`.** Expected initial failure: new module/functions absent.
- [ ] **Step 3: Implement config resolution, strict decoding, target freezing, and predicate.** Reject missing required settings, NaN/infinity, bool-as-integer, nonpositive budgets, ratios outside `(0, 1]`, sample age/gap below period, and GPU tolerance greater than a selected pool's low threshold. Require `0 < low < high <= 1`, `0 <= pending <= used <= capacity`, valid IDs/mappings, every expected rank, FULL groups, and positive host capacity. Preserve the full raw wire sample for output; these dataclasses are the evaluator projection.

```python
import math

def group_valid(group, pools, baseline, config):
    device_ok = any(
        pools[key].used - pools[key].pending_release
        >= math.floor(pools[key].capacity * pools[key].low_ratio)
           - config.gpu_tolerance_blocks
        for key in group.device_pool_ids
    )
    host = pools[group.host_pool_id]
    return (
        device_ok
        and host.host_resident >= math.ceil(host.capacity * config.host_ratio)
        and group.triggers > baseline.triggers
        and group.gpu_host_published > baseline.gpu_host_published
    )
```

`select_targets` freezes all FULL groups for every expected rank, raises on no FULL or changed topology, and leaves SWA/LINEAR in raw observations. Deduplicate pool records by physical ID; reject conflicting duplicates. Compare counters to the before-seed epoch baseline, not the preceding sample. A counter decrease or instance change is a discontinuity, not a negative progress delta.

- [ ] **Step 4: Run predicates/config tests and document settings.** Add cases for one-of-many device pools, rounding at nonmultiple capacities, all required ranks, missing host, low SWA occupancy, counter reset, no FULL, invalid numbers, and frozen dataclass mutation. Add the test to BUILD. The explicit fixture values above are test inputs, not new production defaults.
- [ ] **Step 5: Commit.**

```bash
git add rtp_llm/test/perf_test/cache_grid/runner/pressure_diagnostics.py rtp_llm/test/perf_test/cache_grid/tests/pressure_diagnostics_test.py rtp_llm/test/perf_test/cache_grid/docs/reference.md rtp_llm/test/perf_test/BUILD
git diff --cached --check
git commit -m "feat(cache-perf): define frozen FULL pressure predicates"
```

### Task 5: Conservative Time Coverage and Supply Assessment

**Files:** Modify `rtp_llm/test/perf_test/cache_grid/runner/pressure_diagnostics.py`, `tests/pressure_diagnostics_test.py`, and `docs/reference.md`.

**Interfaces:** Produce `Interval = tuple[float, float]`, `merge_intervals(values) -> list[Interval]`, `intersect_intervals(left, right) -> list[Interval]`, `coverage(values, start, end) -> tuple[float, float]` (ratio, longest invalid duration), and `confirmed_segment(start_server, end_server, offset_low, offset_high) -> Interval | None`. Offset means client-clock minus server-clock. Add `ActivitySample` below and `activity_valid(sample, now, timeout_s) -> bool`.

- [ ] **Step 1: Add interval regression tests.**

```python
class CoverageTest(unittest.TestCase):
    def test_average_does_not_hide_a_long_loss(self):
        ratio, longest = coverage([(0.0, 57.0)], 0.0, 60.0)
        self.assertAlmostEqual(ratio, 0.95)
        self.assertEqual(longest, 3.0)

    def test_merge_before_mapping_avoids_artificial_sample_gaps(self):
        merged = merge_intervals([(0, 1), (1, 2), (2, 3), (3, 4)])
        self.assertEqual(merged, [(0, 4)])
        mapped = confirmed_segment(0, 4, -0.01, 0.01)
        self.assertEqual(mapped, (0.01, 3.99))
        self.assertGreater(mapped[1] - mapped[0], 3.0)

    def test_required_rank_intersection(self):
        valid = intersect_intervals([(0, 60)], [(1, 59)])
        self.assertEqual(valid, [(1, 59)])
        ratio, longest = coverage(valid, 0, 60)
        self.assertAlmostEqual(ratio, 58 / 60)
        self.assertEqual(longest, 1)
```

- [ ] **Step 2: Run `python3 -m unittest rtp_llm.test.perf_test.cache_grid.tests.pressure_diagnostics_test -v`.** Expected failure: interval APIs absent.
- [ ] **Step 3: Implement interval operations with explicit units.**

```python
def merge_intervals(values):
    merged = []
    for start, end in sorted(values):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged

def confirmed_segment(start_server, end_server, offset_low, offset_high):
    start = start_server + offset_high
    end = end_server + offset_low
    return (start, end) if end > start else None

def intersect_intervals(left, right):
    left, right = merge_intervals(left), merge_intervals(right)
    i = j = 0
    output = []
    while i < len(left) and j < len(right):
        start, end = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if end > start:
            output.append((start, end))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return output

def coverage(values, start, end):
    if end <= start:
        raise ValueError("coverage requires a positive window")
    clipped = merge_intervals((max(a, start), min(b, end)) for a, b in values)
    valid = 0.0
    cursor = start
    longest = 0.0
    for a, b in clipped:
        longest = max(longest, a - cursor)
        valid += b - a
        cursor = b
    longest = max(longest, end - cursor)
    return valid / (end - start), longest

@dataclass(frozen=True)
class ActivitySample:
    last_dispatch_s: float
    last_progress_s: float
    active_step_started_s: float | None
    running: int | None
    waiting: int | None

def activity_valid(sample, now, timeout_s):
    progressing = now - sample.last_progress_s <= timeout_s
    active = (sample.active_step_started_s is not None
              and now - sample.active_step_started_s <= timeout_s)
    supplied = (now - sample.last_dispatch_s <= timeout_s
                or active or (sample.waiting is not None and sample.waiting > 0))
    return supplied and (progressing or active)
```

For direct local samples, infer offset bounds `[client_send - server_sample, client_receive - server_sample]`. For fan-out, compose the client's coordinator offset interval with the coordinator's per-rank send/receive interval. Convert nanoseconds to seconds once on decoding. Keep an offset bound valid only within a continuous service epoch; widen for observed drift and terminate the segment if bounds become inconsistent. Unknown age, missing rank, stale (>2s default), sample gap >2s, nonincreasing sequence, counter decrease, and instance reset break candidate segments. Do not retain an old sample as fresh. Save mapping bounds with raw samples.

The caller tracks completion, allocation, publication, and actual-step activity to update `last_progress_s`; a request-only completion pause is not automatically failure. The activity timeout remains explicit experiment configuration. A waiting queue with no progress cannot keep the run alive forever. Return structured reasons in addition to the predicate for manifests.

- [ ] **Step 4: Run tests for gaps, edges, resets, and long prefill.** Add adjacent samples at t=0/1/2 (insufficient steady), t=0/1/2/3 (3s with zero uncertainty), and five samples with nonzero uncertainty. Test exactly 2s versus >2s invalid gaps, empty intervals, all-rank intersection, and active step within/over its configured timeout. Document that coverage is a sampled estimate.
- [ ] **Step 5: Commit.**

```bash
git add rtp_llm/test/perf_test/cache_grid/runner/pressure_diagnostics.py rtp_llm/test/perf_test/cache_grid/tests/pressure_diagnostics_test.py rtp_llm/test/perf_test/cache_grid/docs/reference.md
git diff --cached --check
git commit -m "feat(cache-perf): evaluate conservative pressure time coverage"
```

### Task 6: Bounded Submission and Arrival Schedules

**Files:** Create `rtp_llm/test/perf_test/cache_grid/runner/continuous_pressure.py` and `tests/continuous_pressure_test.py`; modify runner transport integration and `rtp_llm/test/perf_test/BUILD`.

**Interfaces:** Produce `ArrivalClock(rate: float, poisson: bool, seed: int)`, `BoundedSender(limit: int)`, and `RequestHandle` protocol with `done() -> bool`, `result() -> dict`, `cancel() -> bool`. `BoundedSender.submit(job_id: int, make_handle: Callable[[], RequestHandle]) -> bool`, `collect() -> list[tuple[int, dict]]`, and `cancel_all() -> None`. A handle represents an actually started RPC or a bounded transport worker; job IDs are monotonic within the session.

- [ ] **Step 1: Write bounded-queue tests using standard futures as test handles.**

```python
from concurrent.futures import Future

class BoundedSenderTest(unittest.TestCase):
    def test_does_not_construct_or_submit_when_full(self):
        sender = BoundedSender(limit=2)
        first, second, third = Future(), Future(), Future()
        calls = []
        self.assertTrue(sender.submit(0, lambda: first))
        self.assertTrue(sender.submit(1, lambda: second))
        self.assertFalse(sender.submit(2, lambda: calls.append(2) or third))
        self.assertEqual(calls, [])
        first.set_result({"success": True})
        self.assertEqual(sender.collect(), [(0, {"success": True})])
        self.assertTrue(sender.submit(2, lambda: third))
        sender.cancel_all()
        self.assertTrue(second.cancelled())
        self.assertTrue(third.cancelled())
```

- [ ] **Step 2: Run `python3 -m unittest rtp_llm.test.perf_test.cache_grid.tests.continuous_pressure_test -v`.** Expected failure: sender module absent.
- [ ] **Step 3: Implement the bounded sender and cumulative arrival offsets.**

```python
class BoundedSender:
    def __init__(self, limit):
        if limit < 1:
            raise ValueError("positive sender limit required")
        self.limit = limit
        self.pending = {}

    def submit(self, job_id, make_handle):
        if job_id in self.pending:
            raise ValueError("duplicate job ID")
        if len(self.pending) >= self.limit:
            return False
        self.pending[job_id] = make_handle()
        return True

    def collect(self):
        ready = []
        for job_id, handle in list(self.pending.items()):
            if not handle.done():
                continue
            try:
                result = handle.result()
            except Exception as exc:
                result = {"success": False, "error": repr(exc)}
            ready.append((job_id, result))
            del self.pending[job_id]
        return ready

    def cancel_all(self):
        for handle in self.pending.values():
            handle.cancel()

class ArrivalClock:
    def __init__(self, rate, poisson, seed):
        self.rate, self.poisson = rate, poisson
        self.rng = random.Random(seed)
        self.offset = 0.0
        self.count = 0

    def next_offset(self):
        if self.count and self.rate > 0:
            self.offset += (self.rng.expovariate(self.rate) if self.poisson
                            else 1.0 / self.rate)
        self.count += 1
        return self.offset
```

Import `random`; reject invalid rates in resolved config. Keep one ArrivalClock across formal steady/measurement/support. Do not skip an overdue offset when backpressured. The supervisor sleeps interruptibly until the earlier next arrival/poll/phase deadline; it must not block all workers with future scheduled sleeps. Build tokens only when a bounded send slot is available. For gRPC streaming requests, retain the actual call/iterator and invoke its cancellation on shutdown; cancelling only a Python Future is insufficient. Extend the existing `_post_prefill_ids` transport adapter with an optional call-registration callback while preserving its default interface and request deadlines.

- [ ] **Step 4: Verify deterministic arrival and cancellation behavior.** Test burst offsets, 2RPS offsets `[0, .5, 1]`, same-seed Poisson replay, delayed slot acquisition preserving offsets, RPC cancel invocation, and failures becoming result records. Keep legacy `ContinuousPressureTest` passing.
- [ ] **Step 5: Commit.**

```bash
git add rtp_llm/test/perf_test/cache_grid/runner/continuous_pressure.py rtp_llm/test/perf_test/cache_grid/runner/cache_grid_runner.py rtp_llm/test/perf_test/cache_grid/tests/continuous_pressure_test.py rtp_llm/test/perf_test/BUILD
git diff --cached --check
git commit -m "feat(cache-perf): add bounded continuous request supply"
```

### Task 7: Fill, Steady, Measurement, Support, and Drain

**Files:** Modify `runner/continuous_pressure.py`, `runner/cache_grid_runner.py`, `tests/continuous_pressure_test.py`, and `docs/reference.md` under `rtp_llm/test/perf_test/cache_grid/`.

**Interfaces:** Produce `PressurePhases(config, started_s)`, `.phase`, `.measurement_start`, `.measurement_end`, `.failure`, and `advance(now_s, *, fill_ready=False, steady_covered_s=0.0, profile_pending=False) -> str`. Produce `padding_needed(submitted: int, scheduler: str, batch_size: int) -> int`. The supervisor feeds Task 5 evidence into this pure state machine and owns actual RPCs.

- [ ] **Step 1: Write the boundary test with no service.**

```python
class PhaseTest(unittest.TestCase):
    def test_measurement_ends_without_waiting_for_profile(self):
        cfg = resolve_pressure_config(dict(
            fill_timeout_s=120, fill_max_requests=1000, fill_no_progress_s=30,
            fill_input_tokens=4096, gpu_tolerance_blocks=0,
            activity_timeout_s=30, drain_timeout_s=60, capture_timeout_s=120,
        ))
        state = PressurePhases(cfg, started_s=0)
        self.assertEqual(state.advance(2, fill_ready=True), "steady")
        self.assertEqual(state.advance(5, steady_covered_s=3), "measurement")
        self.assertEqual(state.measurement_end, 65)
        self.assertEqual(state.advance(65, profile_pending=True), "support")
        self.assertEqual(state.advance(125, profile_pending=True), "drain")
        self.assertEqual(state.failure, "profile_timeout")
        self.assertEqual(padding_needed(6, "batch_decode", 4), 2)
        self.assertEqual(padding_needed(6, "fifo", 4), 0)
```

- [ ] **Step 2: Run the continuous-pressure unittest module.** Expected failure: phase API absent.
- [ ] **Step 3: Implement a pure phase state and orchestrate around it.**

```python
def padding_needed(submitted, scheduler, batch_size):
    return (-submitted) % batch_size if scheduler == "batch_decode" else 0

class PressurePhases:
    def __init__(self, config, started_s):
        self.config = config
        self.phase = "fill"
        self.entered = started_s
        self.measurement_start = None
        self.measurement_end = None
        self.failure = None

    def advance(self, now_s, *, fill_ready=False, steady_covered_s=0.0,
                profile_pending=False):
        if self.phase == "fill":
            if now_s - self.entered >= self.config.fill_timeout_s:
                self.failure, self.phase = "pressure_not_reached", "drain"
            elif fill_ready:
                self.phase, self.entered = "steady", now_s
        elif self.phase == "steady":
            if now_s - self.entered >= self.config.steady_timeout_s:
                self.failure, self.phase = "pressure_not_reached", "drain"
            elif steady_covered_s >= self.config.steady_s:
                self.phase = "measurement"
                self.measurement_start = now_s
                self.measurement_end = now_s + self.config.measure_s
        elif self.phase == "measurement" and now_s >= self.measurement_end:
            self.phase = "support" if profile_pending else "drain"
        elif self.phase == "support":
            if not profile_pending:
                self.phase = "drain"
            elif now_s >= self.measurement_end + self.config.support_s:
                self.failure, self.phase = "profile_timeout", "drain"
        return self.phase
```

Check stricter capture, request-count, no-progress, and drain deadlines in the supervisor; the pure phase timeout must not override an earlier failure. Begin with existing seed initialization (after capturing counter baselines), then fill using independent unique prefixes/suffixes. Respect the exact input/token/KV/admission budget and only one generated token. Do not batch more long fill requests than can execute together. At fill cutoff, stop new fill, finish/pad its batch, and recheck pressure before steady. All-cold grids skip seed initialization and still execute capacity fill.

At the actual send site, read the phase after waiting for capacity/arrival and build the request ID from session, phase, and monotonic sequence. Never use the job's enqueue phase for membership. In steady/measurement/support, reuse the same producer and cumulative arrival schedule; do not drain at boundaries or refresh seeds. At final supply cessation, count all actual formal and support submissions for padding. Never use fixed-batch controls in FIFO mode. On failure stop production, request profile stop, cancel live RPC calls at the drain deadline, persist cancellation outcomes, and shut down workers without an unbounded `with ThreadPoolExecutor` join.

- [ ] **Step 4: Run fake-clock lifecycle tests.** Cover all-cold fill, insufficient residency, time/request/no-progress exhaustion, 3-second steady from timestamps, 60-second steady timeout, no boundary drain, request dispatched exactly at t1, support completion before cap, earlier capture deadline, padding failure, and blocked RPC cancellation. Update phase/cleanup documentation in reference.md.
- [ ] **Step 5: Commit.**

```bash
git add rtp_llm/test/perf_test/cache_grid/runner/continuous_pressure.py rtp_llm/test/perf_test/cache_grid/runner/cache_grid_runner.py rtp_llm/test/perf_test/cache_grid/tests/continuous_pressure_test.py rtp_llm/test/perf_test/cache_grid/docs/reference.md
git diff --cached --check
git commit -m "feat(cache-perf): implement full-pressure phase lifecycle"
```

### Task 8: Incremental Evidence and Window Metrics

**Files:** Create `rtp_llm/test/perf_test/cache_grid/runner/pressure_results.py` and `tests/pressure_results_test.py`; modify `runner/cache_grid_runner.py`, owning BUILD, and `docs/reference.md`.

**Interfaces:** Produce `PressureWriter(directory: Path, manifest: dict)`, `.append(kind: str, record: dict) -> None`, `.finish(summary: dict) -> None`, and `.close() -> None`; permitted kinds are `requests`, `cache_samples`, and `steps`. Produce `RequestWindow(start_s: float, end_s: float)`, `.observe(record: dict)`, and `.summary() -> dict`. Request records contain `phase`, `kind`, `request_id`, `dispatch_s`, `completed_s`, `success`, shape/reuse/timing fields, and schedule lag.

- [ ] **Step 1: Add crossing and persistence tests.**

```python
class PressureResultsTest(unittest.TestCase):
    def test_late_completion_stays_in_cohort_not_throughput(self):
        window = RequestWindow(10, 70)
        window.observe(dict(phase="measurement", kind="formal", request_id="a",
                            dispatch_s=69, completed_s=71, success=True))
        window.observe(dict(phase="support", kind="formal", request_id="b",
                            dispatch_s=70, completed_s=71, success=True))
        summary = window.summary()
        self.assertEqual(summary["measurement_requests"], 1)
        self.assertEqual(summary["crossing_requests"], 1)
        self.assertEqual(summary["in_window_completions"], 0)
        self.assertEqual(summary["request_throughput"], 0)

    def test_partial_evidence_survives_close_without_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = PressureWriter(Path(directory), {"session_id": "s"})
            writer.append("cache_samples", {"rank": 0, "sequence": 1})
            writer.close()
            rows = (Path(directory) / "pressure/cache_samples.jsonl").read_text()
            self.assertEqual(json.loads(rows)["sequence"], 1)
```

- [ ] **Step 2: Run `python3 -m unittest rtp_llm.test.perf_test.cache_grid.tests.pressure_results_test -v`.** Expected failure: new results module absent.
- [ ] **Step 3: Implement dispatch-cohort counters and incremental files.**

```python
class RequestWindow:
    def __init__(self, start_s, end_s):
        self.start, self.end = start_s, end_s
        self.members = self.crossing = self.completed_inside = 0

    def observe(self, record):
        if record["kind"] != "formal" or record["phase"] != "measurement":
            return
        if not self.start <= record["dispatch_s"] < self.end:
            raise ValueError("measurement dispatch outside window")
        self.members += 1
        completed = record.get("completed_s")
        if completed is not None and completed >= self.end:
            self.crossing += 1
        elif completed is not None and self.start <= completed < self.end:
            self.completed_inside += 1

    def summary(self):
        return dict(measurement_requests=self.members,
                    crossing_requests=self.crossing,
                    in_window_completions=self.completed_inside,
                    request_throughput=self.completed_inside / (self.end - self.start))
```

Observe each completed/cancelled request once; emit a terminal cancellation record for unfinished members. Report success throughput separately if desired, retaining raw completion/success counts and overall execution validity. Keep only bounded in-flight state and bounded distribution accumulators, not all completed requests. Store raw latency/reuse data in JSONL for exact offline reanalysis.

For `PressureWriter`, create `pressure/`, write manifest through a sibling temporary file and `os.replace`, open allowed JSONL streams lazily, encode with `json.dumps(..., allow_nan=False)`, append newline, and flush periodically and on failure. `finish` writes summary atomically then updates manifest validity; `close` always flushes/closes opened files without claiming success. Retain all raw failure records. Record code/profile/grid hashes, resolved settings, target inventory, clock bounds, input identity, session/epoch, and control/window times. `trace_valid` is null/not-applicable in baseline. Use separate measurement/profile `pressure_valid` entries.

- [ ] **Step 4: Run schema/fitting isolation tests.** Add window-edge, padding, failed request, and cancellation cases; verify diagnostic results never populate exact-reuse `metrics[]`. Extend `prefill_formula_fit_test.py` with a diagnostic-only input rejection case using that module's existing loader fixtures. Document throughput and result-status meanings.
- [ ] **Step 5: Commit.**

```bash
git add rtp_llm/test/perf_test/cache_grid/runner/pressure_results.py rtp_llm/test/perf_test/cache_grid/runner/cache_grid_runner.py rtp_llm/test/perf_test/cache_grid/tests/pressure_results_test.py rtp_llm/test/perf_test/cache_grid/tests/prefill_formula_fit_test.py rtp_llm/test/perf_test/cache_grid/docs/reference.md rtp_llm/test/perf_test/BUILD
git diff --cached --check
git commit -m "feat(cache-perf): persist isolated pressure diagnostics"
```

### Task 9: CLI/Profile Wiring, Compatibility, and Resume

**Files:** Modify `rtp_llm/test/perf_test/cache_grid/runner/cache_perf.py`, `rtp_llm/test/perf_test/cache_grid/config/perf_profile.py`, `rtp_llm/test/perf_test/perf_config.py`, `rtp_llm/test/perf_test/batch_decode_test.py`, `rtp_llm/test/perf_test/cache_grid/runner/cache_grid_runner.py`, `rtp_llm/test/perf_test/cache_grid/tests/cache_perf_test.py`, `rtp_llm/test/perf_test/cache_grid/tests/perf_profile_test.py`, `rtp_llm/test/perf_test/batch_decode_test_test.py`, `rtp_llm/test/perf_test/cache_grid/docs/reference.md`, and `rtp_llm/test/perf_test/cache_grid/README.md`.

**Interfaces:** Add `--pressure-preload` and the plan's pressure flags, mapped to `cache_pressure_*` arguments using existing hyphen/underscore conventions. Store the complete resolved pressure settings under the existing frozen run/profile configuration. Produce a new full-pressure runner branch selected only by continuous + preload. Finite continuous and single-case profile keep their existing paths.

- [ ] **Step 1: Add CLI tests in the existing `CachePerfTest` fixture.**

```python
def test_pressure_preload_requires_continuous(self):
    with self.assertRaisesRegex(ValueError, "continuous"):
        cli.build_plan(self.args(extra=["--pressure-preload"]), {})

def test_explicit_count_conflicts_with_pressure_time_window(self):
    with self.assertRaisesRegex(ValueError, "runs|count"):
        cli.build_plan(self.args(extra=["--pressure-mode=continuous",
            "--pressure-preload", "--pressure-measure-seconds=60", "--runs=5"]), {})
```

- [ ] **Step 2: Run `python3 -m unittest rtp_llm.test.perf_test.cache_grid.tests.cache_perf_test -v`.** Expected initial failure: new CLI flags are unrecognized; after parsing is added the tests must fail for the intended validation reason until validation is implemented.
- [ ] **Step 3: Map explicit settings consistently through all layers.** Add remaining required experiment settings as optional CLI overrides of existing profile values: fill timeout/max requests/no-progress/input tokens, GPU tolerance blocks, activity/drain/capture deadlines, and Nsight seconds. They resolve through Task 4, not through another YAML format. Distinguish explicitly supplied `--runs` from a legacy default so time mode does not reject every invocation. Core branch validation:

```python
if args.pressure_preload and args.pressure_mode != "continuous":
    raise ValueError("pressure preload requires continuous mode")
if args.pressure_preload and args.runs is not None:
    raise ValueError("pressure time window conflicts with explicit --runs")
```

Use an explicit-argument sentinel if `args.runs` currently receives a parser default; retain the old default for old modes after validation. Enforce DP=1, exclusive service, ungrouped request geometry, input-ID transport, known block/commit-tail, and incompatibilities in both wrapper and direct runner. Keep continuous-profile rejection until Tasks 11-14 provide its lifecycle; do not accidentally route to single-case replay. Resume keeps immutable settings, starts a fresh session/epoch, reruns seed/fill/steady, and never skips pressure establishment because a disk checkpoint exists. Expose startup validation errors before seed requests.

- [ ] **Step 4: Verify frozen config and old modes.** Use existing profile fixture JSON to supply Task 4 required settings, then assert forwarded CLI values and frozen artifacts. Test negative/NaN values, missing required settings, conflicting overrides, all unsupported combinations, and unchanged finite continuous/case profile behavior. Run `python3 -m unittest rtp_llm.test.perf_test.cache_grid.tests.cache_perf_test rtp_llm.test.perf_test.cache_grid.tests.perf_profile_test rtp_llm.test.perf_test.batch_decode_test_test -v`. Update baseline invocation documentation with explicit required settings.
- [ ] **Step 5: Commit.**

```bash
git add rtp_llm/test/perf_test/cache_grid/runner/cache_perf.py rtp_llm/test/perf_test/cache_grid/config/perf_profile.py rtp_llm/test/perf_test/perf_config.py rtp_llm/test/perf_test/batch_decode_test.py rtp_llm/test/perf_test/cache_grid/runner/cache_grid_runner.py rtp_llm/test/perf_test/cache_grid/tests/cache_perf_test.py rtp_llm/test/perf_test/cache_grid/tests/perf_profile_test.py rtp_llm/test/perf_test/batch_decode_test_test.py rtp_llm/test/perf_test/cache_grid/docs/reference.md rtp_llm/test/perf_test/cache_grid/README.md
git diff --cached --check
git commit -m "feat(cache-perf): wire full-pressure configuration and resume"
```

### Task 10: Actual Step Recorder and TP Identity

**Files:** Create `rtp_llm/cpp/normal_engine/PressureStepRecorder.{h,cc}` and `test/PressureStepRecorderTest.cc`; modify `NormalEngine.{h,cc}`, `NormalExecutor.{h,cc}`, `rtp_llm/cpp/models/ModelTypes.{h,cc}`, `rtp_llm/cpp/models/test/ModelDataTest.cc`, diagnostic RPC conversion, and owning BUILD files.

**Interfaces:** Produce the following value types and bounded recorder. Task 3's `PressureStepPB` serializes these fields. `read` uses an exclusive sequence cursor and returns `lost=true` when the requested next record was overwritten.

```cpp
struct PressureStepRecord {
    uint64_t sequence{0}, step_id{0};
    int64_t begin_ns{0}, forward_enter_ns{0}, forward_return_ns{0}, end_ns{0};
    int64_t forward_enter_cpu_ns{0}, forward_return_cpu_ns{0};
    uint64_t context_batch{0}, decode_batch{0}, tokens{0};
    std::vector<std::string> request_ids;
    bool complete{false};
};
struct PressureStepPage {
    std::vector<PressureStepRecord> records;
    uint64_t next_sequence{0};
    bool lost{false};
};
class PressureStepRecorder {
public:
    explicit PressureStepRecorder(size_t capacity);
    void enable(const std::string& session_id);
    bool disable(const std::string& session_id);
    void append(PressureStepRecord record);
    PressureStepPage read(uint64_t after_sequence, size_t limit) const;
};
```

- [ ] **Step 1: Add a failing bounded-recorder test.**

```cpp
TEST(PressureStepRecorderTest, ReportsLostRecordsWithoutBridgingSteps) {
    PressureStepRecorder recorder(2);
    recorder.enable("s");
    for (uint64_t id = 1; id <= 3; ++id) {
        PressureStepRecord record;
        record.step_id = id;
        record.context_batch = 1;
        record.complete = true;
        recorder.append(record);
    }
    auto page = recorder.read(0, 10);
    ASSERT_EQ(page.records.size(), 2u);
    EXPECT_TRUE(page.lost);
    EXPECT_EQ(page.records.front().step_id, 2u);
    EXPECT_EQ(page.next_sequence, 3u);
    EXPECT_FALSE(recorder.disable("another-session"));
    EXPECT_TRUE(recorder.disable("s"));
}
```

- [ ] **Step 2: Add `pressure_step_recorder_test` to `normal_engine/test/BUILD` and run it.** Command: `bazel test --config=cuda13 //rtp_llm/cpp/normal_engine/test:pressure_step_recorder_test --test_output=errors`. Expected initial failure: recorder implementation absent.
- [ ] **Step 3: Implement the recorder and execution hooks.** Use a mutex-protected fixed-capacity deque and monotonically increasing sequence. Enabling a new session resets the recorder only after validation; reject a conflicting active session. Append only actual nonempty forward records; read copies at most min(requested limit, configured page cap). Keep active-step information separately so polling can distinguish execution from a stalled sender. Timestamp reads use:

```cpp
int64_t pressureClockNs(clockid_t clock) {
    timespec value{};
    if (clock_gettime(clock, &value) != 0) {
        throw std::runtime_error("pressure clock read failed");
    }
    return int64_t(value.tv_sec) * 1000000000LL + value.tv_nsec;
}
// On the executor thread at the matching forward boundaries:
record.forward_enter_ns = pressureClockNs(CLOCK_MONOTONIC);
record.forward_enter_cpu_ns = pressureClockNs(CLOCK_THREAD_CPUTIME_ID);
// Existing model_->forward(...) remains in its existing location.
record.forward_return_cpu_ns = pressureClockNs(CLOCK_THREAD_CPUTIME_ID);
record.forward_return_ns = pressureClockNs(CLOCK_MONOTONIC);
```

Include `<time.h>` and `<stdexcept>`. Do not call CUDA synchronization to obtain these times. Capture request membership from the authoritative dispatched stream grouping on rank zero, not client completion order.

Add `pressure_step_id` (zero means disabled/no actual step), context/decode counts, and executed-token metadata to `GptModelInputs` and its shape-hint encoding/decoding in `ModelTypes.{h,cc}`. Rank zero assigns a candidate ID after input preparation knows the nonempty batch and before existing TP sync; peers receive it in that same sync. Commit/count the ID only when the batch actually executes. Non-root ranks use synchronized metadata, not their locally empty stream list. No extra per-step collective is introduced. Record preparation boundaries at executor entry, even though trace collection may start later.

Add `ControlPressureDiagnostics` public/internal RPCs with session ID, action enum ENABLE/MARK_PHASE/DISABLE, phase string, and recorder capacity. Validate actions and bounds, queue control to the owning engine thread, and return actual per-rank applied timestamps. Poll controls at idle boundaries before early returns. Do not claim all ranks apply at one instant. Snapshots expose recorder pages and active-step progress only for the matching session.

- [ ] **Step 4: Verify recorder, TP metadata, and idle behavior.** Extend ModelDataTest round-trip tests with nonzero diagnostic metadata and disabled zero metadata. Test empty loops, multiple requests in one forward, decode violation, ring overflow, sequence cursor, wrong-session cleanup, active-step visibility, and normal mode with recorder disabled. Run recorder/model/engine focused targets. Document fields in reference.md.
- [ ] **Step 5: Commit.**

```bash
git add rtp_llm/cpp/normal_engine/PressureStepRecorder.h rtp_llm/cpp/normal_engine/PressureStepRecorder.cc rtp_llm/cpp/normal_engine/test/PressureStepRecorderTest.cc rtp_llm/cpp/normal_engine/NormalEngine.h rtp_llm/cpp/normal_engine/NormalEngine.cc rtp_llm/cpp/normal_engine/NormalExecutor.h rtp_llm/cpp/normal_engine/NormalExecutor.cc rtp_llm/cpp/models/ModelTypes.h rtp_llm/cpp/models/ModelTypes.cc rtp_llm/cpp/models/test/ModelDataTest.cc rtp_llm/cpp/model_rpc/proto/model_rpc_service.proto rtp_llm/cpp/model_rpc/LocalRpcServer.h rtp_llm/cpp/model_rpc/LocalRpcServer.cc rtp_llm/test/perf_test/cache_grid/docs/reference.md
git diff --cached --check
git commit -m "feat(engine): record bounded prefill step diagnostics"
```

Also stage the owning BUILD updates reviewed for this task.

### Task 11: Step Classification and Baseline Distributions

**Files:** Modify `rtp_llm/test/perf_test/cache_grid/runner/pressure_results.py`, `tests/pressure_results_test.py`, and `docs/reference.md`.

**Interfaces:** Produce `step_eligible(step: dict, start_ns: int, end_ns: int) -> bool` and `step_gap(previous: dict, current: dict) -> dict | None`. Steps contain Task 10 fields plus decoded `member_phases`, `member_kinds`, and instance/session identity. Decoder obtains phase/kind from recorded request identity, not ordering; unknown membership makes a step ineligible. Window arguments must be in that rank's monotonic clock: for client-minus-server offset bounds `[low, high]`, pass `[ceil(t0_ns - low), floor(t1_ns - high)]`. Reject uncertain boundary steps; never compare client timestamps directly to server timestamps. Same-rank gap subtraction uses original server values.

- [ ] **Step 1: Add boundary and adjacency tests.**

```python
class StepClassificationTest(unittest.TestCase):
    def test_gap_requires_adjacent_full_measurement_steps(self):
        previous = dict(step_id=1, complete=True, context_batch=1, decode_batch=0,
            begin_ns=10, end_ns=20, forward_return_ns=18, forward_return_cpu_ns=8,
            member_phases=["measurement"], member_kinds=["formal"],
            session_id="s", instance="i")
        current = dict(previous, step_id=2, begin_ns=30, end_ns=40,
            forward_enter_ns=32, forward_enter_cpu_ns=11)
        self.assertTrue(step_eligible(previous, 0, 100))
        self.assertEqual(step_gap(previous, current), {"wall_ns": 14, "cpu_ns": 3})
        self.assertIsNone(step_gap(previous, dict(current, step_id=3)))
        self.assertFalse(step_eligible(dict(current, member_phases=["support"]), 0, 100))
        self.assertFalse(step_eligible(dict(current, begin_ns=-1), 0, 100))
```

- [ ] **Step 2: Run the pressure-results unittest module.** Expected failure: classification functions absent.
- [ ] **Step 3: Implement classification, then bounded summaries.**

```python
def step_eligible(step, start_ns, end_ns):
    phases, kinds = step.get("member_phases", []), step.get("member_kinds", [])
    return bool(
        step["complete"] and step["context_batch"] > 0 and step["decode_batch"] == 0
        and start_ns <= step["begin_ns"] < step["end_ns"] <= end_ns
        and phases and len(phases) == len(kinds)
        and all(value == "measurement" for value in phases)
        and all(value == "formal" for value in kinds)
    )

def step_gap(previous, current):
    if (previous["session_id"], previous["instance"]) != (
            current["session_id"], current["instance"]):
        return None
    if current["step_id"] != previous["step_id"] + 1:
        return None
    wall = current["forward_enter_ns"] - previous["forward_return_ns"]
    cpu = current["forward_enter_cpu_ns"] - previous["forward_return_cpu_ns"]
    if wall < 0 or cpu < 0:
        return None
    return {"wall_ns": wall, "cpu_ns": cpu}
```

Call `step_gap` only after both steps pass eligibility and no cursor-loss marker intervenes. Keep per-rank accumulators; never join steps across processes for CPU-clock subtraction. Write all raw records before filtering. For means/max use streaming counters; for quantiles use a deterministic bounded reservoir, reporting reservoir size, algorithm, and total count. Treat p99 as insufficient when fewer than 1000 eligible intervals are available; this is a documented reporting heuristic, not a statistical confidence guarantee. With 20-step traces, emphasize structure rather than tail estimates. Classify pure support, mixed phase, padding, incomplete boundary, unknown membership, and decode separately with exclusion counts. Background-thread wall durations are not added to executor wall time.

- [ ] **Step 4: Verify clock and classification cases.** Test CPU wall overlap, nonadjacent IDs, missing ring records, different sessions/instances, unknown membership, mixed padding, all-support, exactly-at-window boundary, and insufficient sample labeling. Document the quantile heuristic and estimator error limitations.
- [ ] **Step 5: Commit.**

```bash
git add rtp_llm/test/perf_test/cache_grid/runner/pressure_results.py rtp_llm/test/perf_test/cache_grid/tests/pressure_results_test.py rtp_llm/test/perf_test/cache_grid/docs/reference.md
git diff --cached --check
git commit -m "feat(cache-perf): classify prefill steps and baseline gaps"
```

### Task 12: Thread-Affine Kineto Cancellation

**Files:** Modify `rtp_llm/cpp/engine_base/TorchProfiler.{h,cc}`, `test/StepWindowProfilerConfigureTest.cc`, `test/TorchProfilerTest.cc`, `rtp_llm/cpp/normal_engine/NormalEngine.{h,cc}`, `rtp_llm/cpp/model_rpc/proto/model_rpc_service.proto`, `LocalRpcServer.{h,cc}`, RPC tests, and reference.md.

**Interfaces:** Add `bool StepWindowProfiler::requestStop(const std::string& trace_name)` (thread-safe request only) and `void StepWindowProfiler::serviceControl()` (owner-thread only). Add `StopProfile`/`StopProfileInternal` RPCs with `trace_name` and all-rank selection. Each trace name contains the session identity and window identity within the existing sanitization/length limit. Snapshot status exposes armed/active/stopping/finished and the owning trace name; an accepted stop acknowledgement is not completion.

- [ ] **Step 1: Add stop-before-start and mismatched-trace tests to the existing configure fixture.**

```cpp
TEST_F(StepWindowProfilerConfigureTest, StopIsScopedAndCanDisarmBeforeStart) {
    profiler_.configure(true, "sessionA_window0", 0, 20);
    EXPECT_FALSE(profiler_.requestStop("sessionB_window0"));
    EXPECT_TRUE(profiler_.enabled());
    EXPECT_TRUE(profiler_.requestStop("sessionA_window0"));
    EXPECT_FALSE(profiler_.enabled());
    profiler_.serviceControl();
    EXPECT_FALSE(profiler_.has_profiler_.load());
    EXPECT_TRUE(profiler_.requestStop("sessionA_window0"));
}
```

- [ ] **Step 2: Run `bazel test --config=cuda13 //rtp_llm/cpp/engine_base/test:step_window_profiler_configure_test --test_output=errors`.** Expected failure: stop/control methods absent.
- [ ] **Step 3: Request cancellation under the configuration mutex and service it only on the owner thread.**

```cpp
bool StepWindowProfiler::requestStop(const std::string& trace_name) {
    std::lock_guard<std::mutex> lock(mu_);
    if (trace_name != trace_name_) return false;
    enabled_.store(false, std::memory_order_release);
    return true;
}

void StepWindowProfiler::serviceControl() {
    if (!enabled_.load(std::memory_order_acquire)
        && has_profiler_.load(std::memory_order_acquire)) {
        stopProfiler("diagnostic_stop");
    }
}
```

Integrate the request with configure's locking so a concurrent configure cannot resurrect a stopped session or overwrite a different active trace. Preserve the last finished trace identity for idempotent cleanup; reject attempts to stop another trace. Only the engine thread calls `serviceControl`, before early-return idle branches and after safe execution completion. Existing startStep/finishStep keep their same-thread behavior; do not start or advance profiler windows during idle control servicing. A stuck model call cannot be force-stopped on the RPC thread: bound the external wait, report cleanup failure, and use existing service teardown/cancellation policy.

Add stop RPC fan-out using the existing start-profile pattern, and local-only internal handling. Stop/export completion remains separately observable through status and artifact availability. Do not modify existing case-start argument sanitization or the accepted step limits.

- [ ] **Step 4: Run configure, real profiler, RPC, and idle shutdown tests.** Extend TorchProfilerTest with start and stop on the same engine worker while cancellation is requested from another thread; record thread IDs and verify collection occurs on the worker. Cover idle stop without an extra forward, duplicate stop, timeout, wrong trace, and async saver failure. Add stop protocol documentation.
- [ ] **Step 5: Commit.**

```bash
git add rtp_llm/cpp/engine_base/TorchProfiler.h rtp_llm/cpp/engine_base/TorchProfiler.cc rtp_llm/cpp/engine_base/test/StepWindowProfilerConfigureTest.cc rtp_llm/cpp/engine_base/test/TorchProfilerTest.cc rtp_llm/cpp/normal_engine/NormalEngine.h rtp_llm/cpp/normal_engine/NormalEngine.cc rtp_llm/cpp/model_rpc/proto/model_rpc_service.proto rtp_llm/cpp/model_rpc/LocalRpcServer.h rtp_llm/cpp/model_rpc/LocalRpcServer.cc rtp_llm/cpp/model_rpc/test/LocalRpcServerTest.cc rtp_llm/test/perf_test/cache_grid/docs/reference.md
git diff --cached --check
git commit -m "feat(profiling): add thread-affine continuous capture cancellation"
```

### Task 13: Independent Kineto Control and Trace Validation

**Files:** Create `rtp_llm/test/perf_test/cache_grid/runner/pressure_profile.py` and `tests/pressure_profile_test.py`; modify runner integration, all three continuous-profile validation layers, owning BUILD, and docs/reference.md.

**Interfaces:** Produce `ProfileController(start_capture: Callable[[], None], wait_capture: Callable[[], dict], stop_capture: Callable[[], None])` with `start() -> None`, `done() -> bool`, `result(timeout_s: float) -> dict`, and `request_stop() -> None`. Produce `validate_kineto(trace: dict, steps: list[dict], requested_steps: int) -> dict`. The controller uses its own executor/thread; never the sender's pool. All injected operations have explicit timeouts.

- [ ] **Step 1: Add independent-control and invalid-trace tests.**

```python
class PressureProfileTest(unittest.TestCase):
    def test_waiting_for_trace_does_not_block_sender(self):
        release = threading.Event()
        entered = threading.Event()
        def wait_capture():
            entered.set()
            if not release.wait(2):
                raise TimeoutError("fixture capture timeout")
            return {"trace_valid": True}
        controller = ProfileController(lambda: None, wait_capture, release.set)
        controller.start()
        self.assertTrue(entered.wait(1))
        sender = BoundedSender(1)
        handle = Future()
        self.assertTrue(sender.submit(1, lambda: handle))
        handle.set_result({"success": True})
        self.assertEqual(len(sender.collect()), 1)
        release.set()
        self.assertTrue(controller.result(1)["trace_valid"])

    def test_nonempty_trace_without_gpu_work_is_invalid(self):
        trace = {"traceEvents": [{"name": "executor.model_forward", "cat": "cpu_op",
                                  "ph": "X", "ts": 0, "dur": 10, "pid": 1, "tid": 1}]}
        result = validate_kineto(trace, [], requested_steps=20)
        self.assertFalse(result["trace_valid"])
        self.assertIn("missing_gpu_kernels", result["reasons"])
```

Import threading, Future, BoundedSender, and the new module functions explicitly.

- [ ] **Step 2: Run `python3 -m unittest rtp_llm.test.perf_test.cache_grid.tests.pressure_profile_test -v`.** Expected initial failure: profile module absent.
- [ ] **Step 3: Implement control and normalized trace checks.** The worker performs start, wait, and bounded cleanup in try/finally; retain a structured control log for call start/ack/failure times. `request_stop` signals cancellation and invokes the bounded stop operation without depending on the blocked wait task; the wait operation must observe the cancellation event or expire. On result/cleanup, shut down the controller worker with bounded transport/file-wait deadlines.

```python
def validate_kineto(trace, steps, requested_steps):
    events = trace.get("traceEvents", [])
    reasons = []
    kernels = [e for e in events if e.get("cat") == "kernel"]
    forwards = [e for e in events
                if str(e.get("name", "")).startswith("executor.model_forward")]
    runtimes = [e for e in events if e.get("cat") in ("cuda_runtime", "cuda_driver")]
    if not kernels:
        reasons.append("missing_gpu_kernels")
    if not forwards:
        reasons.append("missing_cpu_forward_ranges")
    runtime_ids = {e.get("args", {}).get("correlation") for e in runtimes}
    runtime_ids.discard(None)
    if kernels and not all(e.get("args", {}).get("correlation") in runtime_ids
                           for e in kernels):
        reasons.append("missing_cuda_correlation")
    observed = len({s["step_id"] for s in steps if s.get("context_batch", 0) > 0})
    if observed < requested_steps:
        reasons.append("insufficient_observed_steps")
    usable = [s for s in steps if s.get("complete") and not s.get("profile_boundary")
              and s.get("context_batch", 0) > 0 and s.get("decode_batch", 0) == 0]
    ids = {s["step_id"] for s in usable}
    pairs = sum(step_id + 1 in ids for step_id in ids)
    if not pairs:
        reasons.append("no_complete_internal_step_pair")
    return dict(trace_valid=not reasons, reasons=reasons,
                observed_steps=observed, usable_steps=len(usable), usable_pairs=pairs)
```

Normalize actual Kineto category/argument variants at the parser boundary using fixture traces from the installed runtime; never silently assume an unknown schema is valid. Restrict validation to the model workload's correlated events/window; unrelated kernels outside it are retained but not used to invent missing links. Connect runtime calls to CPU forward scopes by process/thread/time and emitted step markers, and to kernels by correlation. Sidecar steps must be clipped to each rank's actual capture boundaries, not the full 60-second measurement. Mark first/flush boundary exclusions from observed events, not solely client acknowledgement. Require all target ranks and preserve incomplete files.

Start only after steady qualifies, with unique sanitized trace name, `start_step=0`, `num_steps=20`, and `enable_all_rank=true`. Reuse current trace-file collection but strengthen its validation; file waiting stays off sending workers. Wire Task 7 support continuation and earlier capture deadlines. Capture finishing early does not shorten measurement. Enable continuous Kineto through CLI/startup/runner gates without invoking case replay or requiring `--cases`.

- [ ] **Step 4: Run trace/controller/compatibility tests.** Add actual minimal valid trace fixtures, one missing rank, missing correlation, truncated JSON, first-step gather omission, flush boundary, insufficient step count, stop timeout, and early profile completion. Verify every failure preserves request/cache artifacts and does not rewrite a separate baseline result. Document continuous Kineto usage.
- [ ] **Step 5: Commit.**

```bash
git add rtp_llm/test/perf_test/cache_grid/runner/pressure_profile.py rtp_llm/test/perf_test/cache_grid/tests/pressure_profile_test.py rtp_llm/test/perf_test/cache_grid/runner/cache_grid_runner.py rtp_llm/test/perf_test/cache_grid/runner/cache_perf.py rtp_llm/test/perf_test/batch_decode_test.py rtp_llm/test/perf_test/cache_grid/docs/reference.md rtp_llm/test/perf_test/BUILD
git diff --cached --check
git commit -m "feat(cache-perf): capture continuous multi-step Kineto profiles"
```

### Task 14: Nsight Windows, NVTX, and GPU-Gap Attribution

**Files:** Modify `runner/pressure_profile.py`, `runner/cache_grid_runner.py`, `tests/pressure_profile_test.py`, `tests/cache_perf_test.py`, `rtp_llm/cpp/utils/ProfilingScope.h`, executor/cache critical marker sites, owning BUILD files, `rtp_llm/test/perf_test/cache_grid/README.md`, and `docs/reference.md`.

**Interfaces:** Add `run_nsys_window(command: Callable[..., None], *, duration_s: float, cancelled: threading.Event) -> None` using the existing `_nsys_command(record, action, *options)` adapter; `classify_gpu_gap(submit_ns: int, prior_gpu_end_ns: int, next_gpu_start_ns: int, dependency_complete: bool) -> dict`; optional diagnostic-only `PressureNvtxScope` RAII, gated on capture mode. Nsight produces a report path and backend-normalized events with the same units and step IDs used by Kineto analysis.

- [ ] **Step 1: Add command-order and attribution tests.**

```python
class NsightPressureTest(unittest.TestCase):
    def test_stop_attempted_after_start_failure(self):
        calls = []
        def command(action):
            calls.append(action)
            if action == "start":
                raise RuntimeError("injected start failure")
        with self.assertRaises(RuntimeError):
            run_nsys_window(command, duration_s=0.01, cancelled=threading.Event())
        self.assertEqual(calls, ["start", "stop"])

    def test_early_submission_is_not_cpu_delay(self):
        result = classify_gpu_gap(90, 100, 130, dependency_complete=False)
        self.assertEqual(result["classification"], "unresolved_gpu_dependency")
        self.assertIsNone(result["exposed_idle_ns"])
```

- [ ] **Step 2: Run the pressure-profile unittest module.** Expected failure: new Nsight functions absent.
- [ ] **Step 3: Implement bounded capture and evidence-based attribution.**

```python
def run_nsys_window(command, *, duration_s, cancelled):
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("positive finite Nsight duration required")
    try:
        command("start")
        cancelled.wait(duration_s)
    finally:
        command("stop")

def classify_gpu_gap(submit_ns, prior_gpu_end_ns, next_gpu_start_ns,
                     dependency_complete):
    candidate = max(0, next_gpu_start_ns - prior_gpu_end_ns)
    if not dependency_complete:
        label = "unresolved_gpu_dependency"
    elif submit_ns > prior_gpu_end_ns:
        label = "late_host_submission"
    else:
        label = "submitted_before_prior_gpu_completion"
    return dict(classification=label, candidate_gap_ns=candidate,
                exposed_idle_ns=candidate if dependency_complete else None)
```

The production wrapper retains both start and stop exceptions in the manifest if both fail, rather than replacing the original cause. Existing command timeout bounds stop/export. Require explicit `nsys_duration_s`, run from steady on the controller worker, and stop at the earlier duration/support/capture deadline. Keep the existing launch-session and report paths; verify nonempty report plus parseable step evidence before declaring trace valid.

Use the installed Nsight export command's documented SQLite output (confirm with `nsys export --help` on the capture worker), read it through `sqlite3`, and map CUPTI runtime/kernel/memcpy/communication records and NVTX ranges to normalized events. Check required tables/columns before analysis and return an unsupported-schema reason on mismatch. Add export-schema fixtures to tests. Export command planning is local to the installed tool; do not assume one binary version from the repository.

Add explicit NVTX scoped ranges for `pressure.step/<step_id>` and critical stage names during Nsight diagnostics. Use the project's linked NVTX facility and compile guards; ordinary mode must not incur dynamic name formatting or NVTX emission. Preserve ATen markers for Kineto. Scope ranges correctly on exception paths with RAII. Do not run both CUDA activity collectors together.

For each candidate GPU gap, identify the next model compute kernel, its host submission, previous relevant GPU completion, and overlapping copy/communication/other-stream dependencies. `dependency_complete` means those dependencies were actually checked, not just that a correlation ID exists. Missing cross-stream or OS evidence yields unresolved attribution. Keep CPU wall/thread-CPU measurements separate; perf/OS Runtime analysis is linked evidence, not arithmetic subtraction.

- [ ] **Step 4: Verify compatibility and backend artifacts.** Run profile and CLI tests for start/stop ordering, export failure, explicit duration, backend exclusivity, continuation during controls, and unchanged single-case replay. On the capture worker verify NVTX markers and actual step count in a short report. Update baseline/Kineto/Nsight docs with actual installed export command and version.
- [ ] **Step 5: Commit.**

```bash
git add rtp_llm/test/perf_test/cache_grid/runner/pressure_profile.py rtp_llm/test/perf_test/cache_grid/runner/cache_grid_runner.py rtp_llm/test/perf_test/cache_grid/tests/pressure_profile_test.py rtp_llm/test/perf_test/cache_grid/tests/cache_perf_test.py rtp_llm/cpp/utils/ProfilingScope.h rtp_llm/cpp/normal_engine/NormalExecutor.cc rtp_llm/test/perf_test/cache_grid/docs/reference.md rtp_llm/test/perf_test/cache_grid/README.md
git diff --cached --check
git commit -m "feat(cache-perf): add continuous Nsight pressure attribution"
```

Explicitly add any changed cache marker sites and owning BUILD files after reviewing them; avoid staging unrelated cache work.

### Task 15: End-to-End Acceptance and Final Documentation

**Files:** Extend `tests/continuous_pressure_test.py`, `tests/pressure_profile_test.py`, and existing pipeline tests; update `rtp_llm/test/perf_test/cache_grid/README.md`, `docs/reference.md`, and `docs/continuous_full_pressure_plan.md` after acceptance. Store run evidence under selected result directories, not source directories.

**Interfaces:** Consumes all prior contracts. Produces a recorded acceptance matrix with command, code/profile/grid hashes, hardware/tool versions, result paths, and independent execution/pressure/trace statuses. The source plan stays "not implemented" until runtime checks actually pass.

- [ ] **Step 1: Add a no-model end-to-end failure test.** Build the fixture using Task 4 decoded samples, Task 6 sender handles, Task 7 phases, and Task 8 writer. The controlling assertion is:

```python
def test_missing_rank_cannot_produce_valid_pressure(self):
    required_ranks = (0, 1)
    sample = RankSample(
        rank=0, instance="i0", sequence=1, server_s=0,
        client_send_s=0, client_receive_s=0.001,
        pools=(PoolSample("g", 100, 90, 0, .8, .9),
               PoolSample("h", 100, 30, 0, .8, .9, 25)),
        groups=(GroupSample(0, "FULL", ("g",), "h", 1, 1),),
    )
    with self.assertRaisesRegex(ValueError, "rank"):
        select_targets((sample,), required_ranks)
```

Add integrated fake-transport cases that lose rank 1 after steady and inspect the final manifest, not only the predicate. Provide rank 1 initially using `dataclasses.replace(sample, rank=1, instance="i1")`; after measurement starts, return only rank 0 until its unknown interval exceeds 2 seconds. Expect `pressure_valid=false`, preserved samples, and bounded drain. Keep all fixture clocks injected; no sleeps to simulate 60 seconds.

- [ ] **Step 2: Run Python and focused C++ regressions.**

```bash
python3 -m unittest discover -s rtp_llm/test/perf_test/cache_grid/tests -p '*_test.py'
python3 -m unittest rtp_llm.test.perf_test.batch_decode_test_test
bazel test --config=cuda13 //rtp_llm/cpp/cache/block_tree_cache/test:cache_pressure_state_test //rtp_llm/cpp/cache/block_tree_cache/test:block_tree_cache_test //rtp_llm/cpp/cache/block_tree_cache/evict/test:block_tree_evictor_test //rtp_llm/cpp/model_rpc/test:local_rpc_server_test //rtp_llm/cpp/normal_engine/test:pressure_step_recorder_test //rtp_llm/cpp/engine_base/test:step_window_profiler_configure_test //rtp_llm/cpp/engine_base/test:torch_profiler_test --test_output=errors
```

Also run the touched load/store, ModelDataTest, engine, worker-status, and formula-fit targets selected from their owning BUILD files. Record exact invocations and results; an unavailable toolchain/GPU is an unmet acceptance item, not a passing test.

- [ ] **Step 3: Run real small-capacity smoke and independent profiles.** These commands are for execution after CLI implementation. The operator supplies actual existing profile/grid paths and a new result root; they must not be invented or silently downloaded. The profile contains explicit fill length/budgets, GPU tolerance, activity/drain/capture deadlines, and host-tier capacity. Its engine settings must create reachable pressure for the selected FULL model.

```bash
: "${CACHE_PRESSURE_PROFILE:?Set the actual frozen profile path}"
: "${CACHE_PRESSURE_GRID:?Set the actual grid path}"
: "${CACHE_PRESSURE_RESULTS:?Set a new result root}"
tools/cache_perf run --profile "$CACHE_PRESSURE_PROFILE" --grid "$CACHE_PRESSURE_GRID" --result-dir "$CACHE_PRESSURE_RESULTS/baseline" --pressure-mode continuous --pressure-preload --pressure-scheduler fifo --pressure-measure-seconds 60
tools/cache_perf profile --profile "$CACHE_PRESSURE_PROFILE" --grid "$CACHE_PRESSURE_GRID" --result-dir "$CACHE_PRESSURE_RESULTS/kineto" --pressure-mode continuous --pressure-preload --pressure-scheduler fifo --profile-backend kineto --pressure-profile-steps 20
tools/cache_perf profile --profile "$CACHE_PRESSURE_PROFILE" --grid "$CACHE_PRESSURE_GRID" --result-dir "$CACHE_PRESSURE_RESULTS/nsys" --pressure-mode continuous --pressure-preload --pressure-scheduler fifo --profile-backend nsys --pressure-profile-seconds 10
```

The explicit 10-second Nsight value is an acceptance-run choice, not a universal default. Run baseline/profile separately. Repeat a small smoke with fixed batching, then all-cold and hit/cold grids, and TP>1 within DP=1. Use the existing profile syntax for model/hardware details. Acceptance requires real successful GPU-to-host publication on every required FULL rank/group, 25% default host residency, 3-second steady coverage, default measurement coverage >=95%, longest invalid/unknown <=2 seconds, actual nonempty pure-prefill steps, complete rank traces, and bounded cleanup. Show actual host reuse separately; host residency does not imply host hits.

- [ ] **Step 4: Exercise the failure matrix and review artifacts.** Use deterministic integration fixtures for no FULL, missing host, unreachable thresholds, fill/steady timeout, stale/missing rank, counter reset, pressure loss, fixed-tail padding failure, profile start/stop/export failure, and support expiry. Use real GPU cases for transfer completion and cancellation paths that mocks cannot establish. Verify no invalid run reports valid pressure, support does not extend measurement, diagnostics never enter fitting, and neither restart nor resume joins epochs. Record actual baseline/profile perturbation without asserting identical cache contents.

- [ ] **Step 5: Update implementation status and commit documentation/tests only after evidence passes.**

```bash
git add rtp_llm/test/perf_test/cache_grid/tests/continuous_pressure_test.py rtp_llm/test/perf_test/cache_grid/tests/pressure_profile_test.py rtp_llm/test/perf_test/cache_grid/README.md rtp_llm/test/perf_test/cache_grid/docs/reference.md rtp_llm/test/perf_test/cache_grid/docs/continuous_full_pressure_plan.md
git diff --cached --check
git commit -m "test(cache-perf): validate continuous FULL pressure diagnostics"
```

Record evidence paths in the implementation review. Do not commit large trace/model artifacts. Update the corresponding OpenSpec tasks only after their checks complete; do not archive or merge as part of this implementation task without the applicable workflow request.

## Coverage and Interface Audit

| OpenSpec task group | Detailed tasks | Required behavior covered |
| --- | --- | --- |
| 1. Cache Snapshot and Publication Evidence | 1, 2 | Pool/group identity, published residency, copy/publication separation, resets, cost |
| 2. Diagnostic Protocol and Rank Collection | 3, 10 | Opt-in queries, capability errors, all-rank timing, controls, health compatibility |
| 3. Frozen Configuration and Pure Pressure Evaluation | 4, 5 | FULL-only gates, defaults, immutable targets, interval intersection, supply checks |
| 4. Bounded Continuous Request Lifecycle | 6, 7 | Fill, all-cold, bounded jobs, arrival lag, uninterrupted transitions, padding/cancellation |
| 5. CLI Wiring and Diagnostic Results | 8, 9 | Config propagation, immutable resume, cohort metrics, independent validity, fitting isolation |
| 6. Actual Step Recording and Baseline Analysis | 10, 11 | Actual step identity, TP membership, wall/CPU meanings, complete adjacent pairs, distributions |
| 7. Continuous Kineto Lifecycle | 12, 13 | Owner-thread stop, independent control, 20-step capture, rank/trace validation, support cap |
| 8. Continuous Nsight and Trace Attribution | 14 | Bounded capture, NVTX, export, CPU/GPU attribution, backend exclusivity |
| 9. End-to-End Acceptance | 15 | Real demotion, multi-rank pure prefill, both schedulers, baseline/profile comparison, failures |

Interface ownership is fixed: Task 1 owns cache evidence; Task 2 owns the complete cache snapshot; Task 3 owns its wire envelope; Task 4 owns evaluator dataclasses/config; Task 5 owns time coverage; Task 6 owns bounded send handles; Task 7 owns phases; Task 8 owns persisted request evidence; Task 10 owns step records/control; Task 11 owns measurement step selection; Task 12 owns stop requests; Tasks 13-14 own profile control/analysis. Future code must not introduce a second conflicting definition of these contracts.

## Planning Verification

This file is a plan, not implementation or runtime evidence. Before handoff, verify that every task has files, consumed/produced interfaces, test code, expected initial failure, implementation code/algorithm, a success command or observable check, and scoped commit instructions. Check all referenced existing paths and target names. Re-run OpenSpec strict validation; keep all implementation task checkboxes unchecked.

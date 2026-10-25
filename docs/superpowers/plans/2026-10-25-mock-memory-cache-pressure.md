# Mock Memory Cache Pressure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to execute this plan task by task. Use `superpowers:test-driven-development` for every production change and `superpowers:verification-before-completion` before claiming completion.

**Goal:** Add an opt-in, performance-only cache mode that controls the cumulative block-aligned Host-cache hit ratio, performs real H2D loads from a bounded synthetic Host reservoir, and writes completed requests back through the real D2H Host-store path.

**Architecture:** A serialized cumulative controller makes one persisted decision per request in `CoordinatorCacheManager`. A cache-lifetime `MockMemoryCacheSource` supplies immutable, CRC-valid Host blocks to a dedicated `BlockTreeLoader` context that uses the existing transfer dispatcher without entering the normal cache tree. `StreamCacheResource` forces successful requests to the existing Host store path while tagged metrics distinguish planned, loaded, and stored mock traffic.

**Tech Stack:** C++17, CUDA/HIP cache transfer layer, Bazel, GoogleTest, Python server-argument tests, existing RTP-LLM pressure runner and profiling tools.

**Spec:** [2026-10-25-mock-memory-cache-pressure-design.md](../specs/2026-10-25-mock-memory-cache-pressure-design.md)

## Global Constraints

- The feature is disabled by default and must have no observable behavior change when disabled.
- Enabling it marks model output invalid for correctness use; startup logs must say this explicitly.
- Initial lookup bypasses natural Device, Host, Disk, and remote matches. Decode allocation remains unchanged.
- The ratio denominator and numerator are logical reusable blocks. Unaligned tails and invalid reuse prefixes do not enter the denominator.
- A request is charged to the cumulative controller once. Allocation retries reuse the stored plan.
- Mock loads and stores must traverse the existing asynchronous transfer stack. A metadata-only simulated hit is not acceptable.
- Reservoir payloads are finite. When CRC is active, each Host record includes a valid CRC32C footer over its payload.
- The reservoir remains outside the normal cache tree and holds a distinct `MOCK_SOURCE` block reference for its lifetime.
- All Bazel commands below must be run through the repository `test-execution` precheck and `bazel_run.sh` workflow, as a non-root container user, with the exact GPU lock selected from the test target. Resolve the active container, user, cache root, and CUDA config before the first invocation; do not invoke raw `bazel test`.

---

## Task 1: Configuration Contract And Startup Validation

**Files:**

- Modify: `rtp_llm/cpp/config/ConfigModules.h`
- Modify: `rtp_llm/cpp/config/ConfigModules.cc`
- Modify: `rtp_llm/cpp/pybind/ConfigInit.cc`
- Modify: `rtp_llm/cpp/pybind/config_pickle_test.py`
- Modify: `rtp_llm/server/server_args/kv_cache_group_args.py`
- Modify: `rtp_llm/server/server_args/test/server_args_test.py`
- Modify: `rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheFactory.cc`
- Modify: `rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheFactoryTest.cc`

### Step 1: Write failing server-argument and pickle tests

Add assertions for these defaults and explicit values:

```text
mock_memory_cache = false
mock_memory_cache_hit_ratio = 0.0
mock_memory_cache_seed = 1
mock_memory_cache_source_blocks = 64
```

Append the new fields to the pybind pickle tuple. Bump the pickle schema version from 7 to 8, preserve loading of version-7 tuples with the defaults above, and reject unsupported newer versions.

Run:

```bash
python3 -m unittest rtp_llm.server.server_args.test.server_args_test
# Via test-execution wrapper:
//rtp_llm/cpp/pybind:config_pickle_test --config=cuda13
```

Expected: tests fail because the fields and pickle schema do not exist.

### Step 2: Add config fields and argument parsing

Append fields to `KVCacheConfig`:

```cpp
bool     mock_memory_cache{false};
double   mock_memory_cache_hit_ratio{0.0};
uint64_t mock_memory_cache_seed{1};
size_t   mock_memory_cache_source_blocks{64};
```

Expose them through `kv_cache_group_args.py`, `ConfigInit.cc`, and `KVCacheConfig::toString()`. Keep environment/argument naming consistent with adjacent cache options.

### Step 3: Write failing startup-validation tests

In `BlockTreeCacheFactoryTest.cc`, cover:

- disabled mode accepts the default ratio and source count;
- ratio `0.0` and `1.0` are accepted when prerequisites are present;
- ratio below `0.0`, above `1.0`, NaN, and infinity fail;
- enabled mode fails when `reuse_cache`, `enable_memory_cache`, or positive `memory_cache_size_mb` is absent;
- enabled mode fails when `mock_memory_cache_source_blocks == 0`;
- enabled mode logs the invalid-output warning.

### Step 4: Implement fail-fast validation

Add one validation block before pool construction in `BlockTreeCacheFactory`. Log the specific invalid field and return `nullptr`, matching the factory's existing failure contract, rather than silently disabling mock mode. Keep validation independent of current free Host capacity; actual reservation failure is an initialization error in Task 3.

### Step 5: Verify and commit

Run the Python test and these wrapper-managed Bazel targets:

```text
//rtp_llm/cpp/pybind:config_pickle_test --config=cuda13
//rtp_llm/cpp/cache/block_tree_cache/test:block_tree_cache_factory_test --config=cuda13
```

Commit:

```text
feat(cache): add mock memory cache configuration
```

---

## Task 2: Cumulative Controller And Retry-Stable Request Plan

**Files:**

- Create: `rtp_llm/cpp/cache/MockMemoryHitController.h`
- Create: `rtp_llm/cpp/cache/MockMemoryHitController.cc`
- Create: `rtp_llm/cpp/cache/test/MockMemoryHitControllerTest.cc`
- Modify: `rtp_llm/cpp/cache/BUILD`
- Modify: `rtp_llm/cpp/cache/test/BUILD`
- Modify: `rtp_llm/cpp/cache/BatchKVCacheResource.h`
- Create: `rtp_llm/cpp/cache/test/BatchKVCacheResourceTest.cc`

### Step 1: Write failing controller tests

Define the public contract:

```cpp
struct MockMemoryHitDecision {
    size_t   eligible_blocks{0};
    size_t   planned_host_blocks{0};
    uint64_t cumulative_eligible_blocks{0};
    uint64_t cumulative_planned_host_blocks{0};
};

struct MockMemoryHitSnapshot {
    uint64_t cumulative_eligible_blocks{0};
    uint64_t cumulative_planned_host_blocks{0};
};

class MockMemoryHitController {
public:
    explicit MockMemoryHitController(double ratio);
    MockMemoryHitDecision plan(
        size_t eligible_blocks,
        const std::function<bool(size_t)>& is_valid_prefix);
    MockMemoryHitSnapshot snapshot() const;
};
```

Test:

- ratios `0`, `0.5`, `1`, and a non-dyadic fraction such as `0.333333`;
- variable eligible lengths across a sequence;
- result never exceeds the current request's eligible blocks;
- invalid desired prefixes round down to the largest valid prefix;
- the resulting deficit is carried into later requests;
- `floor(cumulative_eligible * ratio) - cumulative_planned` stays below one admissible prefix quantum when later requests permit recovery;
- concurrent callers produce the same final counters as some serialized order and never overshoot the cumulative target;
- overflow guards reject or saturate before arithmetic wraps.

Run the new CPU-only target through the wrapper and confirm red.

### Step 2: Implement deterministic cumulative arithmetic

At construction, validate the ratio and quantize it once to millionths:

```cpp
ratio_ppm_ = std::llround(ratio * 1'000'000.0);
target = static_cast<uint64_t>(
    (static_cast<unsigned __int128>(eligible_next) * ratio_ppm_) / 1'000'000);
```

Serialize `plan()` and `snapshot()` with one mutex. Compute the desired request contribution as `clamp(target - planned_so_far, 0, eligible_blocks)`, then decrement until `is_valid_prefix(candidate)` succeeds. Update both cumulative counters only after a valid decision is chosen.

Document the effective ratio resolution as `1e-6`. Do not use request randomness; admission order is the sole input.

### Step 3: Write failing request-plan lifetime tests

Add a request-local value type:

```cpp
struct MockMemoryCachePlan {
    size_t eligible_blocks{0};
    size_t planned_host_blocks{0};
};
```

Store it as `std::optional<MockMemoryCachePlan>` in `BatchKVCacheResource`. Test:

- the first allocation stores it;
- a retry reads the exact same value;
- `swap()` moves it with the rest of the request state;
- reset for a new stream clears it;
- batch resize does not duplicate an existing decision.

### Step 4: Implement plan persistence

Add narrow `mockMemoryCachePlan()`, `setMockMemoryCachePlan()`, and `clearMockMemoryCachePlan()` methods. Include the optional in all existing swap/reset paths. Do not expose controller state through `BatchKVCacheResource`.

### Step 5: Verify and commit

Run wrapper-managed targets:

```text
//rtp_llm/cpp/cache/test:mock_memory_hit_controller_test --config=cuda13
//rtp_llm/cpp/cache/test:batch_kv_cache_resource_test --config=cuda13
```

Commit:

```text
feat(cache): add cumulative mock memory hit planning
```

---

## Task 3: CRC-Valid Synthetic Host Reservoir

**Files:**

- Create: `rtp_llm/cpp/cache/block_tree_cache/load/MockMemoryCacheSource.h`
- Create: `rtp_llm/cpp/cache/block_tree_cache/load/MockMemoryCacheSource.cc`
- Create: `rtp_llm/cpp/cache/block_tree_cache/load/test/MockMemoryCacheSourceTest.cc`
- Modify: `rtp_llm/cpp/cache/block_tree_cache/BUILD`
- Modify: `rtp_llm/cpp/cache/block_tree_cache/load/test/BUILD`
- Modify: `rtp_llm/cpp/cache/block_tree_cache/block_pool/IBlockPool.h`
- Modify: `rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h`
- Modify: `rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.cc`
- Modify: `rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheFactory.cc`
- Modify: `rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheFactoryTest.cc`

### Step 1: Write failing reservoir tests

Define:

```cpp
class MockMemoryCacheSource {
public:
    static absl::StatusOr<std::shared_ptr<MockMemoryCacheSource>> create(
        const std::vector<GroupSetPtr>& group_sets,
        size_t source_blocks,
        uint64_t seed);

    BlockIdxType blockFor(size_t group_set_id, size_t logical_block) const;
    size_t sourceBlockCount(size_t group_set_id) const;
};
```

Test:

- exactly the configured count is reserved in every `GroupSet` Host pool;
- reservation uses a new `BlockTreeRefType::MOCK_SOURCE` reference;
- the same `(seed, group_set_id, logical_block)` maps deterministically;
- logical indices beyond the reservoir wrap without extra allocation;
- every payload byte is finite-safe zero in the initial implementation;
- each record's padding is initialized, not left uninitialized;
- CRC-disabled records have `storageBytes() == payloadBytes()` behavior;
- CRC-enabled records contain the same CRC32C footer expected by `CrcTransferService`;
- a real `hostToDevice` transfer of an initialized source block succeeds with CRC checking enabled;
- insufficient Host capacity returns initialization failure and releases partial reservations.

### Step 2: Add a dedicated lifetime reference

Extend `BlockTreeRefType` with `MOCK_SOURCE` before `COUNT`. Update all fixed-size arrays, names, debug output, and pool metric snapshots. Add `mock_source_ref_blocks` so operators can distinguish reserved source capacity from cache, load, eviction, and store pressure.

### Step 3: Implement allocation and deterministic ring mapping

For each GroupSet:

```cpp
auto blocks = group_set->allocateBlocks(source_blocks, Tier::HOST,
                                        BlockTreeRefType::MOCK_SOURCE);
```

Retain those IDs until after loader and transfer workers drain. Derive each group's ring offset from `mock_memory_cache_seed` and `group_set_id`; map `logical_block` modulo that group's source count. The mapping changes which source record is used but never changes transfer length.

### Step 4: Initialize payload and CRC records correctly

Zero the full Host stride. For CRC-enabled groups:

1. Compute CRC32C over exactly `GroupSet::payloadBytes()` using polynomial `0x82F63B78` and the same initial/final convention as the existing transfer CRC implementation.
2. Write the 32-bit footer at the offset defined by `CrcBlockCopyBatch::encodedBytes(payload_bytes) - sizeof(uint32_t)`.
3. Leave any alignment padding deterministically zero.
4. Assert the encoded size fits `HostBlockPool::strideBytes()`.

Factor the CPU CRC helper beside `MockMemoryCacheSource` and verify its output against the existing CRC path in the test. Do not fill raw bytes from a PRNG because arbitrary FP8/BF16/FP16 bit patterns may become NaN or Inf.

### Step 5: Construct the source during cache initialization

When mock mode is enabled, create the source after GroupSet Host pools exist and before accepting traffic. Attach it to the loader/cache object. Log source blocks and bytes per GroupSet plus the invalid-output warning. Any reserve or initialization failure aborts cache construction.

### Step 6: Verify and commit

Run wrapper-managed targets:

```text
//rtp_llm/cpp/cache/block_tree_cache/load/test:mock_memory_cache_source_test --config=cuda13
//rtp_llm/cpp/cache/block_tree_cache/test:block_tree_cache_factory_test --config=cuda13
```

Commit:

```text
feat(cache): reserve CRC-valid mock host cache source
```

---

## Task 4: Dedicated Mock H2D Load Lifecycle

**Files:**

- Modify: `rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h`
- Modify: `rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.cc`
- Modify: `rtp_llm/cpp/cache/block_tree_cache/load/BlockTreeLoader.h`
- Modify: `rtp_llm/cpp/cache/block_tree_cache/load/BlockTreeLoader.cc`
- Modify: `rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h`
- Modify: `rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.cc`
- Modify: `rtp_llm/cpp/cache/CoordinatorCacheManager.h`
- Modify: `rtp_llm/cpp/cache/CoordinatorCacheManager.cc`
- Modify: `rtp_llm/cpp/cache/BUILD`
- Modify: `rtp_llm/cpp/cache/block_tree_cache/load/test/LoadAsyncContextTest.cc`
- Modify: `rtp_llm/cpp/cache/block_tree_cache/load/test/BlockTreeLoaderTest.cc`
- Modify: `rtp_llm/cpp/cache/test/KVCacheManagerWithTierCacheBasicTest.cc`
- Modify: `rtp_llm/cpp/cache/test/KVCacheManagerWithTierCacheLoadFailureTest.cc`

### Step 1: Write failing async-context and loader tests

Add:

```cpp
enum class LoadSourceKind { TREE, MOCK_MEMORY };
```

The default remains `TREE`. Test a mock context with null tree nodes and assert that:

- tier reconstruction reports `Tier::HOST` logical blocks once across GroupSets;
- `commitLoad()` does not dereference a tree node or register a tree-load join;
- each descriptor is `HOST -> DEVICE` and has real Device destinations;
- multi-group geometry uses `GroupSet::computeReuseBlockCount()`;
- reservoir wrap still emits one descriptor per required group mapping;
- success publishes the target blocks only after transfer completion;
- failure, timeout, and cancellation release every target `LOAD` reference;
- settlement never inserts source or target blocks into the normal block tree.

### Step 2: Add a separate mock-load API

Expose through `BlockTreeCache` and `BlockTreeLoader`:

```cpp
std::shared_ptr<LoadAsyncContext> mockMemoryLoad(size_t logical_blocks);
```

`mockMemoryLoad()` builds descriptors from `MockMemoryCacheSource::blockFor()` and leaves `LoadDescriptor::node` null. It does not call `match()`, the join registry, or any tree-state mutation.

### Step 3: Route commit, abort, and settlement by source kind

In `BlockTreeLoader`:

- normal contexts retain the exact current code path;
- mock commit adds Device `LOAD` references and submits the existing `LoadTaskRunner` task;
- mock success retains request-owned Device blocks and settles the async context as Host reuse;
- mock failure/abort calls the existing task-resource release helper and publishes zero reuse;
- mock settlement never releases reservoir `MOCK_SOURCE` references.

Keep shutdown ordering explicit: stop new admissions, drain/cancel loader work, stop transfer workers, then destroy the reservoir.

### Step 4: Write failing coordinator tests

Cover admission behavior:

- disabled mode calls existing natural lookup unchanged;
- enabled mode computes eligible blocks using the existing max-prefix and `isValidReuseLength()` rules;
- enabled mode ignores an existing natural Device, Host, or Disk match;
- a non-zero plan creates the mock Host context;
- a zero plan follows cold allocation and does not perform natural lookup;
- allocator retry reuses `BatchKVCacheResource::mockMemoryCachePlan()` and does not increment controller counters;
- only the coordinator plans; TP peers consume the coordinated descriptor count;
- successful finalize reports `memoryReuseBlockNum` and `host_reuse_length` from completed blocks;
- failed transfer reports no cache reuse.

### Step 5: Integrate controller at initial allocation

Instantiate one controller when attaching the block-tree cache in enabled mode. In `initMallocForCommonLen()`:

1. Derive the canonical, block-aligned eligible key prefix with the same maximum-prefix and valid-length helpers used by normal matching.
2. Reuse the request plan if present; otherwise call the controller once and persist the decision.
3. Skip `prepareKVCache(match_keys, ...)` completely in mock mode.
4. For `planned_host_blocks > 0`, request a mock load and mark load attempted.
5. For zero planned blocks, continue through cold Device allocation without a lookup context.
6. Let existing `materializeInitialBlocks()` allocate request destinations and attach them to the context.

Add a dedicated profiling scope around planning and descriptor construction so CPU overhead is separate from actual H2D spans.

### Step 6: Verify and commit

Run wrapper-managed targets:

```text
//rtp_llm/cpp/cache/block_tree_cache/load/test:load_async_context_test --config=cuda13
//rtp_llm/cpp/cache/block_tree_cache/load/test:block_tree_loader_test --config=cuda13
//rtp_llm/cpp/cache/test:kv_cache_manager_with_tier_cache_test --config=cuda13
```

Commit:

```text
feat(cache): execute planned mock host loads through H2D
```

---

## Task 5: Forced Host Writeback And Mock Observability

**Files:**

- Modify: `rtp_llm/cpp/cache/KVCacheManager.h`
- Modify: `rtp_llm/cpp/cache/KVCacheManager.cc`
- Modify: `rtp_llm/cpp/cache/Types.h`
- Modify: `rtp_llm/cpp/engine_base/stream/StreamCacheResource.cc`
- Modify: `rtp_llm/cpp/engine_base/stream/test/StreamCacheResourceTest.cc`
- Modify: `rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h`
- Modify: `rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.cc`
- Modify: `rtp_llm/cpp/cache/block_tree_cache/store/BlockTreeStorer.h`
- Modify: `rtp_llm/cpp/cache/block_tree_cache/store/BlockTreeStorer.cc`
- Modify: `rtp_llm/cpp/cache/block_tree_cache/store/StoreTaskRunner.h`
- Modify: `rtp_llm/cpp/cache/block_tree_cache/store/StoreTaskRunner.cc`
- Modify: `rtp_llm/cpp/cache/block_tree_cache/store/test/BlockTreeStorerTest.cc`
- Modify: `rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h`
- Modify: `rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.cc`
- Modify: `rtp_llm/cpp/metrics/RtpLLMMetrics.h`
- Modify: `rtp_llm/cpp/metrics/RtpLLMMetrics.cc`

### Step 1: Write failing store-target tests

Extend `StreamCacheResourceTest` with:

- disabled mode retains Device-first, then Host, then Disk selection;
- enabled mock mode selects `Tier::HOST` whenever memory cache is available, even when Device cache storage is available;
- cancelled, errored, and unfinished requests still return `Tier::NONE` under existing rules;
- memory cache becoming unavailable returns the established failure/no-store result, not Device fallback.

Expose a read-only `KVCacheManager::mockMemoryCacheEnabled()` for this selection.

### Step 2: Tag mock stores end to end

Add `bool mock_memory_cache{false}` to `InsertInfo`. Propagate it through `BlockTreeCache::insert()`, `BlockTreeStorer`, and `StoreTaskRunner::Task`. Existing callers rely on the default false value.

In `StreamCacheResource::tryReleaseKVBlock()`, set the tag from the manager mode and pass the request's actual keys and Device resources unchanged.

### Step 3: Write failing D2H and publication tests

Test:

- successful request completion submits a real `DEVICE -> HOST` descriptor;
- Device source blocks remain pinned with `STORE` references after request references are released;
- Host tree publication occurs only after D2H succeeds;
- transfer failure frees destination Host blocks, publishes nothing, and increments failure metrics;
- duplicate-key publication may be rejected after the D2H copy, but submitted/completed work remains counted;
- response completion does not wait for the asynchronous store;
- failed/cancelled requests do not submit a store.

### Step 4: Add bounded metrics

Add one collector/group with these fields:

```text
mock_memory_cache_eligible_blocks
mock_memory_cache_planned_host_blocks
mock_memory_cache_successful_h2d_blocks
mock_memory_cache_failed_h2d_blocks
mock_memory_cache_d2h_submitted_blocks
mock_memory_cache_d2h_completed_blocks
mock_memory_cache_d2h_failed_blocks
mock_memory_cache_planned_ratio_ppm
mock_memory_cache_success_ratio_ppm
mock_memory_cache_source_blocks
mock_memory_cache_source_bytes
```

Update cumulative gauges from controller/load snapshots and increment transfer outcomes only at terminal completion. Preserve the existing general transfer metrics as the authoritative byte and latency series. Do not add request IDs as metric labels.

Add per-request debug logging with request ID, eligible, planned, successful Host blocks, and retry flag. Keep it at debug level to avoid pressure-test log amplification.

### Step 5: Verify and commit

Run wrapper-managed targets:

```text
//rtp_llm/cpp/engine_base/stream/test:stream_cache_resource_test --config=cuda13
//rtp_llm/cpp/cache/block_tree_cache/store/test:block_tree_storer_test --config=cuda13
//rtp_llm/cpp/cache/test:kv_cache_manager_with_tier_cache_test --config=cuda13
```

Commit:

```text
feat(cache): force mock requests through host writeback
```

---

## Task 6: Pressure Profile, Documentation, And Acceptance

**Files:**

- Create: `rtp_llm/test/perf_test/cache_grid/config/continuous_full_pressure_dsv4.mock_memory.example.jsonc`
- Modify: `rtp_llm/test/perf_test/cache_grid/tests/perf_profile_test.py`
- Modify: `rtp_llm/test/perf_test/cache_grid/tests/pressure_profile_test.py`
- Modify: `rtp_llm/test/perf_test/cache_grid/docs/continuous_full_pressure_guide_zh.md`
- Modify: `docs/backend/server_arguments.md`

### Step 1: Write failing profile-contract tests

Create a dedicated mock profile from the production example so normal production and smoke profiles remain unchanged. Set these engine arguments in the new profile:

```jsonc
"--mock_memory_cache": true,
"--mock_memory_cache_hit_ratio": 0.5,
"--mock_memory_cache_seed": 1,
"--mock_memory_cache_source_blocks": 64
```

Test that snapshot/freeze handling retains all four fields and that dry-run output includes them verbatim. Assert the profile is marked diagnostic-only because output correctness is intentionally invalid.

### Step 2: Document operating semantics

Document in Chinese:

- ratio is cumulative reusable-token coverage expressed in whole blocks, not a percentage of request count;
- the ratio applies to initial prompt allocation only;
- natural cache matches are bypassed;
- planned ratio and successful ratio are separate;
- H2D should appear during admission and D2H after successful completion;
- cache output cannot be evaluated for correctness;
- source block count bounds Host capacity use but does not cap transferred logical blocks because the ring wraps;
- CRC initialization happens before the pressure/profile window;
- profile traces must retain the existing transfer and scheduler spans.

### Step 3: Run focused regression suites

Run:

```bash
python3 -m unittest \
  rtp_llm.server.server_args.test.server_args_test \
  rtp_llm.test.perf_test.cache_grid.tests.perf_profile_test \
  rtp_llm.test.perf_test.cache_grid.tests.pressure_profile_test
```

Then run every Bazel target from Tasks 1-5 through `test-execution`, grouped only when their GPU type and resolved environment are identical.

### Step 4: Run a real continuous profile acceptance

Start the normal service with the mock profile and run the existing independent adaptive continuous sender. Do not use cohort barriers or completion-coupled issuance. Use a fresh result directory and preserve the frozen profile, grid, launch record, metrics, and timeline.

Acceptance checks:

1. `abs(cumulative_planned_host_blocks - floor(cumulative_eligible_blocks * ratio))` is no more than one valid reuse block at the end of the measurement window.
2. Successful ratio is reported separately and any gap is explained by explicit load failures/cancellations.
3. Timeline contains real Host-to-Device cache-load events during request admission.
4. Timeline contains real Device-to-Host cache-store events after successful request completion.
5. Initial mock reuse attribution contains Host only; Device and Disk reuse are zero.
6. The mock source reservation is constant throughout traffic and no source block enters the normal tree.
7. The sender maintains its configured in-flight pressure without 32-request cohort barriers.
8. CPU analysis reports planning, descriptor construction, scheduler, H2D wait/wakeup, D2H worker, and model-forward intervals separately.

### Step 5: Final regression and commit

Run the repository formatting/lint command for touched C++ and Python files, rerun the focused tests after formatting, and inspect `git diff --check` plus `git status --short`.

Commit:

```text
docs(cache): add mock memory pressure profile and acceptance
```

---

## Completion Criteria

- Disabled-mode behavior is covered and unchanged.
- Configuration, pybind pickle compatibility, and profile freezing are tested.
- Cumulative planning converges under variable request lengths and concurrent admission.
- Retries cannot charge the controller twice.
- Mock Host records pass real CRC-enabled H2D validation.
- The load lifecycle has correct success, failure, cancellation, and shutdown ownership.
- Every successful mock-mode request attempts real asynchronous D2H Host writeback.
- Metrics distinguish planned hits, completed H2D, and D2H outcomes.
- A continuous profile demonstrates the configured cumulative ratio and visible H2D/D2H operations without cohort barriers.
- The final branch passes all focused tests through the mandated wrapper workflow.

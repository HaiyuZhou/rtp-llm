# Mock Memory Cache Pressure Design

## Goal

Add an opt-in pressure-test mode that assigns a configured cumulative fraction
of reusable, block-aligned prompt tokens to the memory-cache tier. Assigned
prefixes must execute the real Host-to-Device transfer path, and every
successfully completed request must execute the normal Device-to-Host cache
store path. Model output is intentionally not valid in this mode and must not
be used for correctness evaluation.

## Scope

The first implementation supports synthetic memory-cache hits only. It does
not synthesize device, disk, or remote-cache hits. Disk support may later reuse
the same cumulative controller and add a disk-backed source provider.

Mock mode applies only to the initial allocation of an online request. Decode
incremental allocation keeps its existing behavior. When mock mode is disabled,
cache lookup, storage, metrics, and model output remain unchanged.

## Configuration

Add these fields to `KVCacheConfig` and expose them through the existing server
argument and Python binding paths:

```text
mock_memory_cache = false
mock_memory_cache_hit_ratio = 0.0
mock_memory_cache_seed = 1
mock_memory_cache_source_blocks = 64
```

`mock_memory_cache_hit_ratio` is a fraction in the inclusive range `[0.0, 1.0]`.
Enabling mock mode requires `reuse_cache=true`, `enable_memory_cache=true`, and
`memory_cache_size_mb > 0`. `mock_memory_cache_source_blocks` is the number of
immutable source blocks reserved from each `GroupSet` Host pool and must be
positive when mock mode is enabled. Invalid combinations fail during
configuration validation before the engine accepts traffic.

Mock mode is a service-level setting. Requests do not carry mock-cache fields.
The mode bypasses natural Device, Host, Disk, and remote matching for initial
allocation so that an assigned mock hit is always attributed to Host and always
performs H2D.

## Cumulative Hit Controller

The ratio denominator is the number of prompt tokens in valid, block-aligned,
reusable prefixes. It excludes unaligned tail tokens and any prefix forbidden
by the existing maximum-prefix and `isValidReuseLength()` constraints.

The controller operates in reusable logical blocks. For request `i`, let
`eligible_i` be its valid reusable prefix block count, `E` the cumulative
eligible block count, `H` the cumulative planned Host-hit block count, and `R`
the configured ratio. Admission computes:

```text
E_next = E + eligible_i
target = floor(E_next * R)
planned_i = clamp(target - H, 0, eligible_i)
E = E_next
H = H + planned_i
```

If topology-specific valid-prefix rules prevent exactly `planned_i` blocks,
the request uses the largest valid prefix not exceeding that value. The
remaining deficit is retained and may be filled by later requests. The normal
maximum prefix reuse rule remains in force, including any token or block that
must remain for prefill.

The controller is serialized in request-admission order. Its decision is made
once per request and stored with the request's cache allocation state. An
allocator retry reuses the same plan and does not update cumulative counters a
second time. A request cancellation or transfer failure is reported separately;
planned and successful ratios are both observable.

The coordinating allocator makes the decision once. TP ranks consume the same
planned prefix length through the existing coordinated transfer request rather
than maintaining independent ratio counters.

## Synthetic Host Source

Introduce a dedicated mock Host source owned by the block-tree cache subsystem.
It reserves exactly `mock_memory_cache_source_blocks` pinned Host blocks from
each `GroupSet` and holds them for the cache lifetime. Logical mock prefix
blocks map across that reservoir in a deterministic ring derived from
`mock_memory_cache_seed`.

Source payloads are initialized once, outside request admission and profiling.
Initialization must produce finite values for every physical cache layout. A
zero-valued payload is the safe fallback when dtype-aware bounded random
initialization is unavailable; arbitrary random bytes are forbidden because
they can encode NaN or Inf and destabilize kernels.

The source blocks are immutable after initialization and may be read by
concurrent requests. Each planned logical block still creates its own transfer
descriptor, target Device block, and H2D copy, even when the bounded reservoir
wraps and a source block is reused. This preserves transfer bytes, descriptor
processing, Device allocation, asynchronous completion, and scheduler wakeups.

Mock source blocks do not enter the normal cache tree. This prevents a prior
mock load from turning a later forced Host hit into a Device hit and prevents
synthetic keys from polluting normal eviction and matching behavior. Reservoir
capacity and reserved bytes are logged at startup and visible in pool-pressure
metrics.

## Load Path

During initial allocation, `CoordinatorCacheManager` computes the request's
eligible block count and obtains its persisted mock plan. A non-zero plan uses
a dedicated mock-load preparation path instead of `BlockTreeCache::match()`.

The mock-load path:

1. Allocates the normal request-owned Device blocks required by the planned
   prefix for every applicable `GroupSet`.
2. Creates Host-to-Device descriptors from the mock source reservoir to those
   Device blocks, respecting each group's reuse geometry.
3. Submits the descriptors through the existing
   `BlockTransferDispatcher`/per-rank transfer engine.
4. Returns an async context compatible with the existing allocator wait and
   cancellation lifecycle.
5. Publishes the request's Device block IDs and Host reuse length only after
   all H2D copies complete successfully.

The async context reports the matched tier as `Tier::HOST`, so existing
`StreamCacheResource::finalizeAllocatorLoad()` behavior sets
`memoryReuseBlockNum`, `host_reuse_length`, and cache-load metrics from real
completed work. A partial or failed transfer publishes no reuse and releases
all allocated target blocks.

A zero-block plan follows the normal cold allocation path without natural
cache matching. It therefore represents the non-hit portion of the configured
mock workload rather than accidentally consuming an existing cache entry.

## Completion Writeback

For a successfully finished mock-mode request, `StreamCacheResource` selects
`Tier::HOST` as its store target regardless of the normal Device-first store
preference. It passes the request's actual cache keys and Device resources to
the existing `KVCacheManager::insertIntoCache()` and `BlockTreeStorer` path.

The existing asynchronous store semantics remain unchanged:

- the store task pins Device source blocks before request references are freed;
- it performs a real D2H transfer;
- it publishes Host cache-tree resources only after a successful copy;
- it rolls back allocated Host blocks and publishes nothing on failure;
- the client response does not wait for background D2H completion.

Errored, cancelled, or unfinished requests retain the current rule and are not
stored. Duplicate cache keys may be rejected during publication, but the store
submission and D2H work remain observable.

## Concurrency And Lifetime

The cumulative controller uses a mutex because initial allocations can be
concurrent. The per-request plan lives with `BatchKVCacheResource`, survives
retryable allocation failures, and is cleared when that resource is reset for a
new stream.

The mock Host reservoir is initialized before traffic and released only after
the loader and transfer workers are drained during cache shutdown. In-flight
mock load cancellation follows the same terminal-state and Device-block
rollback contract as normal lower-tier loads.

## Observability

Expose bounded counters and gauges for:

- cumulative eligible blocks and tokens;
- cumulative planned Host-hit blocks and configured ratio;
- successful and failed mock H2D blocks;
- mock D2H store blocks submitted, completed, and failed;
- planned and successful cumulative hit ratios;
- mock source reservoir blocks and bytes.

Per-request debug logs contain request ID, eligible blocks, planned Host blocks,
successful Host blocks, and whether the allocation is a retry. Existing cache
load/store transfer profiling remains authoritative for H2D and D2H durations;
mock planning receives a separate profiling scope so CPU control overhead is
distinguishable from transfer preparation.

Startup logs must state that outputs are invalid while mock mode is enabled.

## Failure Behavior

Configuration errors fail startup. Exhausting Device capacity uses the existing
retryable/permanent allocation classification. Failure to reserve or initialize
the mock Host source fails cache initialization rather than silently disabling
mock mode.

An H2D preparation, queue, timeout, cancellation, or copy failure publishes no
mock hit for that request and releases temporary Device references. D2H store
failure leaves the completed request successful but records the store failure
and does not publish a Host cache entry, matching existing asynchronous store
semantics.

## Verification

Tests must cover:

1. Configuration defaults, argument/binding propagation, valid boundaries, and
   invalid prerequisite combinations.
2. Cumulative ratio convergence, block rounding, valid-prefix deficits, zero
   and one ratios, concurrent admission, and retry idempotence.
3. Disabled mode preserving normal cache matching and Device-first storage.
4. Mock mode ignoring natural cache entries and producing only Host-attributed
   reuse.
5. Real H2D descriptors and Device destinations for every planned logical
   block, including reservoir wraparound and multi-group layouts.
6. Transfer success publishing reuse only after completion; failure and
   cancellation rolling back all target references.
7. Successful request completion submitting a real Host store, successful D2H
   publishing normal cache-tree entries, and failed/error requests not storing.
8. Metrics and logs distinguishing planned hits, successful H2D, and D2H store
   outcomes.

An end-to-end pressure acceptance run must show the configured cumulative Host
hit ratio within one valid reuse block, visible Host-to-Device load events,
visible Device-to-Host store events after request completion, and no Device or
Disk reuse attribution for initial mock allocation.

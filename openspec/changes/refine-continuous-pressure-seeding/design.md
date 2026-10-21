# Design

## Context

See `proposal.md` for motivation and `specs/adaptive-fifo-cache-pressure/spec.md` for normative behavior. This change builds on the completed, unarchived `add-continuous-full-pressure-diagnostics` change, so that change must be integrated first.

The current runner resolves an omitted seed count to four before startup and checks client concurrency against an aggregate active-execution budget. Seed requests are issued in batches separated by full drains. This keeps a small set of prefixes hot and prevents a deep FIFO backlog. The existing `GetCachePressure` response already exposes FULL group membership, owner/participant identity, logical pool capacity, block geometry, watermarks, host residency, and settlement-backed transfer counters; no RPC schema extension is needed.

The FULL cache can contain several named member pools for one group, and TP participants can expose views of pools whose mutable logical owner is one rank. Pool block zero is reserved, so planning must consume the allocatable logical capacity returned by the live snapshot rather than reconstructing it from configuration. The earlier 127-block DeepSeek-V4-Flash result came from a deliberately forced 128-block smoke setup and is not a production-capacity measurement.

## Goals / Non-Goals

**Goals:**

- Produce one deterministic seed plan from the live post-startup FULL topology and persist enough inputs to recalculate it.
- Make the reusable family working set strictly exceed every required FULL group's GPU family capacity while remaining retainable across GPU low-water residency and host capacity.
- Keep FIFO supplied through a bounded client backlog without treating queued RPCs as simultaneously executing requests.
- Separate execution, FULL-pressure, formal-configuration, and bidirectional-migration validity.
- Preserve a small explicit smoke path for fast GPU and unit validation.

**Non-Goals:**

- No fixed-batch or batch-decode preload support, padding, or partial-batch tail handling in this workflow.
- No automatic GPU/host cache resizing, cache pinning, direct block manipulation, or scheduler changes.
- No formal guarantee for SWA or LINEAR cache groups; they remain observable where existing diagnostics expose them.
- No change to ordinary grid/case execution outside the continuous full-pressure preload path.

## Decisions

### 1. Plan after startup from frozen owner topology

The runner takes the existing startup `GetCachePressure` snapshot before constructing seed requests. It freezes required FULL groups, owner ranks, TP participants, physical pool identities, block-token spans, allocatable logical capacities, low/high ratios, current host availability, and publication scope for the epoch. Existing inventory validation rejects missing owners, incompatible participant layouts, zero capacity, invalid watermarks, or topology changes.

Planning uses only the named owner record for logical capacity, residency, and publication counters. Participant records remain required for topology and freshness validation but are not summed. Within one group, each physical pool ID is deduplicated before calculating capacity. The plan uses reported allocatable capacity directly, including the cache's reserved-block effect.

Configuration-derived capacity was rejected because flags such as `test_block_num` can intentionally replace the production pool size and because shared TP views make rank summation incorrect. A pre-start estimate is useful only for configuration diagnostics and cannot become the formal plan.

### 2. Derive one feasible reusable-family count

The reusable token span is:

```text
grid_reuse_tokens = max(grid.cache_len)
reuse_tokens = align_up(max(32768, grid_reuse_tokens), cache_alignment)
```

A formal grid must have `grid_reuse_tokens >= 32768`; otherwise the extra initialization tokens would never appear in formal requests and would not be reusable. The seed commit tail is calculated and recorded separately and is excluded from every working-set calculation.

For each required FULL group `g` and each deduplicated owner device pool `p`:

```text
seed_blocks[g,p] = ceil(reuse_tokens / block_tokens[g,p])
gpu_family_capacity[g] = min_p floor(capacity[g,p] / seed_blocks[g,p])
gpu_low_families[g] = min_p floor(low_blocks[g,p] / seed_blocks[g,p])
low_blocks[g,p] = floor(capacity[g,p] * low_ratio[g,p])
```

The host budget uses the live host pool's available retainable blocks after non-reclaimable occupancy and any configured safety reserve. It converts device-family tokens through the host pool's own block-token span rather than assuming device and host blocks are interchangeable:

```text
host_seed_blocks[g] = ceil(reuse_tokens / host_block_tokens[g])
host_families[g] = floor(host_retention_blocks[g] / host_seed_blocks[g])
minimum[g] = gpu_family_capacity[g] + 1
maximum[g] = gpu_low_families[g] + host_families[g]
seed_count = max_g minimum[g]
```

The single `seed_count` is feasible only if it is no greater than `min_g maximum[g]`. The strict `+1` makes the reusable family bank larger than the GPU capacity of every required group; high-water reachability is also checked from the reported high ratios. The upper bound keeps the bank retainable after eviction settles around the low watermark. If the bounds do not overlap, preflight reports each group's calculated bounds and stops rather than weakening the target.

This conservative family-level calculation can leave some host capacity unused when pools have different geometry. Optimizing a mixed packing problem was rejected because it would make seed identity and replay nonuniform and harder to audit.

An explicit seed count bypasses this derivation only in smoke/debug mode. The planner still records the live topology and evaluates feasibility, but the run receives a non-formal reason and cannot qualify as production acceptance.

### 3. Use a fixed formal request shape and validate it independently

The default formal fill input target is `100 * 1024 = 102400` tokens, rounded up to the common request/cache alignment. Every fill request uses a unique token family. Its one-shot suffix contributes to immediate pressure but never to reusable seed capacity. The runner fails formal preflight if the aligned fill request plus required output cannot fit the model sequence limit, per-request token limit, or per-request KV admission limit; it does not shorten the request.

The same individual-admission check applies to seed requests and every formal grid shape. `ExecutionLimits.concurrency_for(shape)` can continue to calculate active execution feasibility, but the preflight only requires its result to be positive for each shape. It no longer caps the number of outstanding client RPCs. A separate admission/queue check verifies that the transport and service can accept the configured bounded backlog.

The formal client in-flight default and minimum are both 16. This count includes submitted RPCs whether they are waiting or running. The FIFO scheduler decides the active batch from its own token and KV limits. Silent concurrency clamping was rejected because it can turn a requested formal run into a shallow, non-reloading workload without visible failure.

### 4. Run seed initialization as a rolling FIFO producer

The full-pressure path validates `scheduler == fifo` at CLI/profile resolution and again at the runner boundary so direct callers cannot bypass it. FIFO always uses scheduler batch size one and never invokes batch controls or padding. Existing fixed-batch behavior remains available to ordinary cases outside this path.

Seed initialization reuses the bounded sender used by the continuous phase:

1. Lazily construct and submit requests until the in-flight bound is reached or the derived target has been submitted.
2. Poll completions continuously; after every successful completion, submit the next family immediately when the target remains.
3. Stop producing once exactly `seed_count` families have been accepted, then drain all accepted seed requests within a dedicated seed deadline.
4. Record submitted, completed, failed, cancelled, and pending family counts. Any unsuccessful or missing family prevents measurement.

There is no barrier between groups of 16. This bounds memory and threads while maintaining queue pressure. The seed deadline is independent of fill and steady deadlines so a large production-derived bank cannot consume the fill budget or be mistaken for fill progress.

The phase sequence remains `validate -> seed -> fill -> steady -> measurement -> support -> drain`. Seed traffic is excluded from fill timing and measurement transfer evidence. Fill continues to use unique 102400-token requests until the existing FULL pressure predicates qualify; formal grid requests alone sustain steady, measurement, and profile-support phases.

### 5. Make formal classification explicit and monotonic

Run classification starts as formal-eligible and accumulates non-formal reasons. At minimum, any of these make it non-formal:

- explicit/manual seed count;
- client in-flight depth below 16;
- forced synthetic GPU capacity such as `test_block_num > 0`;
- an explicitly selected reduced smoke profile.

Auxiliary-pool sizing that is part of an actual deployment topology is recorded but is not automatically called synthetic; the explicit test-capacity override is the decisive signal. A non-formal run can still execute and report smoke results. No later success can erase its classification reasons.

The epoch manifest records the raw startup snapshot, resolved owner mappings, all formula terms, auto/manual source, planned and completed seed counts, alignment, reusable/commit/fill lengths, configured backlog, scheduler, relevant capacity flags, and classification reasons. Summary JSON and console output expose separate `execution_valid`, `full_pressure_valid`, `formal_config_valid`, and `bidirectional_migration_valid` fields; aggregate formal acceptance requires all four.

### 6. Qualify H2D and D2H inside the measurement window

For every required FULL owner group, the evaluator freezes transfer-counter baselines at the measurement boundary using the existing clock-aligned diagnostic timeline. It accepts only settlement-backed publication deltas whose conservative sample interval lies within the measurement window:

- D2H: successful STORE or EVICT publication from DEVICE to HOST;
- H2D: successful LOAD publication from HOST to DEVICE.

Both deltas must be positive for every required FULL group. Shared TP publication is attributed once to the owner and remains valid only while all declared participants have fresh, compatible topology evidence. Copy-start, reservation, bytes copied without publication, transfers outside the measurement interval, and failed/cancelled transfers do not qualify.

This becomes a new bidirectional validity result rather than an extra condition hidden inside general FULL-pressure coverage. A run can therefore show that requests executed and pressure remained valid while clearly failing because it produced only D2H traffic. Counter decrease, service-instance change, missing boundary evidence, or topology change invalidates the epoch under the existing rules.

### 7. Keep production acceptance distinct from the small-capacity smoke

Unit tests cover formula boundaries, TP owner deduplication, heterogeneous block spans, infeasible host retention, grid reuse rejection, independent backlog admission, rolling replenishment, deadline cleanup, classification, and window-local transfer deltas. Integration tests cover CLI/profile/direct-call FIFO enforcement and preserve ordinary batch regressions.

GPU validation has two tiers:

- A reduced-capacity DeepSeek-V4-Flash smoke proves request construction, seed/fill phase transitions, cleanup, and artifact generation. It must report non-formal.
- Final acceptance starts `/mnt/hf3fs/3fs/models/DeepSeek-V4-Flash` without `test_block_num`, generates a profile and grid with `max(cache_len) >= 32768`, uses automatic seed count and at least 16 client requests in flight, and runs long enough to evaluate both transfer directions. Its artifacts must independently reproduce the plan and pass every formal gate.

The previous 127-block evidence remains useful as smoke evidence only and must not be cited as the production FULL pool capacity.

## Risks / Trade-offs

- [A production cache can require many long seed requests] -> Generate lazily, bound in-flight work at 16 by default, give seed initialization its own explicit deadline, and report partial completion without entering measurement.
- [Host retention capacity may be too small for a bank larger than GPU capacity] -> Detect the empty feasible interval before request submission and report the limiting FULL group rather than accepting a D2H-only workload.
- [Pool geometries can differ across groups or tiers] -> Convert tokens to blocks separately for every pool and retain the raw terms in the manifest.
- [A deep client backlog can increase latency and cancellation time] -> Use bounded request deadlines, stop production before drain, cancel outstanding RPCs, and enforce a global drain deadline.
- [Conservative measurement boundaries can miss transfers near t0 or t1] -> Keep sampling around the window edges and fail bidirectional qualification when the event interval cannot be placed safely inside the window.
- [Removing batch preload support breaks existing pressure profiles] -> Reject them with a migration message; convert pressure profiles to FIFO while preserving ordinary batch case execution.

## Migration Plan

1. Integrate `add-continuous-full-pressure-diagnostics` so its RPC, counter, evaluator, and bounded-session contracts are present.
2. Add the live seed planner and manifest fields behind the existing continuous full-pressure opt-in; keep explicit counts as non-formal smoke compatibility.
3. Switch pressure-profile defaults to FIFO, 16 in-flight requests, automatic seed count, 32K minimum reusable span, and 102400-token fill inputs. Reject batch pressure profiles at every public entry point.
4. Add rolling seed production and the separate seed deadline, then add measurement-window bidirectional qualification and split validity fields.
5. Run focused unit/integration tests, the reduced-capacity GPU smoke, and the production-sized DeepSeek-V4-Flash acceptance. Publish the generated Chinese operator guide with both command paths and artifact checks.

Rollback disables the full-pressure preload opt-in or reverts these runner changes. Ordinary case execution remains available. New manifests retain their classification and planner-version fields so they are not interpreted by older acceptance tooling as formal results.

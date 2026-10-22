# Design

## Context

See `proposal.md` for motivation and the three capability specs for normative behavior. The approved source is `rtp_llm/test/perf_test/cache_grid/docs/continuous_full_pressure_plan.md`.

Observed implementation constraints:

- `cache_grid_runner.py::_run_continuous()` builds a finite jobs list and submits all futures. Worker concurrency is bounded, but this cannot become an unbounded time-window producer unchanged. `_initialize_continuous_seeds()` skips all-cold grids and validates request shape/reuse, not capacity pressure.
- `cache_perf.py`, `batch_decode_test.py`, and the runner constructor each reject continuous profiling. Existing `_run_kineto_request()` records one step and waits synchronously; `_run_nsys_request()` surrounds one request plus a tail.
- `BlockTreeCache::poolMetricsSnapshots()` and `BlockTreeCacheMetricsReporter` already expose physical occupancy and overlapping reference categories. `KVCacheManager::getKVCacheInfo()` exposes aggregate token capacity, which cannot prove host publication.
- `BlockTreeEvictor::computePoolWatermarkRequired()` uses `used - pending_release`; group eviction triggers when any associated pool reaches high water. Successful copy and settled publication differ, including the detached-destination path in `scheduleEvictionSettlement()`.
- Real Flash TP validation confirmed `BlockTreeCacheFactory` gives only TP0 the mutable tree. Peers hold physical pools and execute shared block IDs without updating local allocator/publication metadata. `ModelTypes` synchronizes IDs by pool tags; `MultiRankBlockTransferEngine` requires every worker/block result before owner settlement. Therefore peer-local zero counters are not evidence of empty physical data, and shared-owner provenance is necessary.
- `LocalRpcServer` already has per-TP broadcast for profile control. `CacheVersionPB`/`CacheStatusPB` are normal cache-status messages and should keep their existing lightweight semantics.
- `NormalExecutor` starts profiling after input gather and TP input synchronization. `StepWindowProfiler` in `engine_base/TorchProfiler.{h,cc}` owns thread-affine collection and asynchronous file saving. There is no existing public stop-profile RPC in the inspected protocol.
- `ProfilingScope.h` emits ATen record-function ranges, not direct NVTX markers. Existing trace collection checks parseability and nonempty events, not GPU/step completeness.

## Goals / Non-Goals

**Goals:**

- Establish one auditable contract from cache settlement through per-rank diagnostics to pressure qualification and result validity.
- Reuse transport, prompt generation, real scheduling, profile launch infrastructure, and existing compatibility paths.
- Keep pressure evaluation and interval accounting deterministic and independently testable without a model service.
- Preserve bounded submission, observation, artifact buffering, and cancellation behavior throughout long experiments.

**Non-Goals:**

- No C++ cache benchmark changes, simulated capacity, direct pool manipulation, pinned seeds, automatic engine-capacity increases, or new scheduler.
- No SWA/LINEAR capacity pass/fail gates, guaranteed host-hit mix, DP>1 support, decode/chunked-prefill workload, or identical cache contents between independent runs.
- No universal automatic explanation of off-CPU waits without OS evidence, and no per-step GPU synchronization added for timing.

## Decisions

### 1. Keep three ownership boundaries

| Boundary | Existing integration | Proposed focused additions |
| --- | --- | --- |
| Cache evidence | `BlockTreeCache`, evict/load/store settlement, `KVCacheManager` | Snapshot value types and settlement-backed residency/publication counters |
| Run orchestration | `cache_grid_runner.py`, `cache_perf.py`, `perf_config.py`, `batch_decode_test.py` | `runner/pressure_diagnostics.py` for immutable config, snapshots, and validity; `runner/continuous_pressure.py` for bounded phase orchestration |
| Step/profile evidence | `NormalEngine`, `NormalExecutor`, `TorchProfiler`, existing profile helpers | Opt-in step recorder, thread-affine stop control, and `runner/pressure_profile.py` for continuous control and trace validation |

New runner paths are relative to `rtp_llm/test/perf_test/cache_grid/`. They are proposed modules, not existing APIs. Keep the legacy finite `_run_continuous()` behavior until equivalent helper reuse is covered by regression tests; do not refactor unrelated case execution. The alternative of growing all logic inside `CacheGridRunner` would couple clocks, HTTP waits, pressure predicates, and request accounting unnecessarily.

### 2. Add explicit diagnostic RPCs outside normal status probes

Add `GetCachePressure` and `GetCachePressureInternal` request/response messages and methods to `model_rpc_service.proto`. The public rank-zero endpoint gathers bounded per-rank responses through existing worker endpoints, retaining partial failures under a common deadline; the internal method is local-only to avoid recursive fan-out. Diagnostic channels share each request-local completion queue's lifetime, because reusing channels across destroyed queues caused reproducible multi-peer completion stalls. Request fields include diagnostic session identity, requested ranks, optional step-record cursor, and bounded page size. Responses carry a schema version, capability/error information, requested/returned rank lists, and per-rank records. The runner uses its existing gRPC channel; a new frontend HTTP endpoint is unnecessary.

Add a diagnostic engine access method with a default unsupported response so non-normal engines and unrelated mocks do not accidentally claim support. `NormalEngine` delegates cache snapshots to `KVCacheManager`/`BlockTreeCache`. A separate session control RPC enables/disables the bounded step recorder and publishes phase/window markers. Controls are explicit, session-scoped, idempotent for cleanup, and report actual per-rank application times.

Snapshot fields:

- Identity: schema version, service-instance ID, session/epoch, world/TP rank, monotonic timestamp, sequence, and capture duration.
- Topology: group-set ID/type, stable physical pool ID/name, tier, mappings, block bytes, capacity, used/free/available, reference categories, and pending release blocks.
- Ownership: independent-local or supported shared-TP mode, owner world rank, participant ranks, and whether publication requires successful transfer completion on all participants. Derive this from the actual cache/transfer configuration.
- Pressure: configured low/high ratios, group trigger counter, valid published host residency, and successful-copy/successful-publication/failure counters by operation/direction/group/pool.
- Activity: available waiting/running counts, completed nonempty-step count, active-step timing, and allocation/publication progress counters. Unavailable values are explicit rather than zero.

Waiting means local waiting plus asynchronous-cache-loading entries, excluding running entries. Use an optional constant-time scheduler accessor under existing queue locks; unsupported grouped states remain unavailable. NormalEngine delegates and EngineBase remains unsupported by default.

Normal `/worker_status` and `GetCacheStatus` do not request these records. An alternative extension that computes diagnostics during every status call would increase hot-path health-check cost and make capability detection ambiguous.

### 3. Count publication at ownership transitions

Reuse pool snapshots for physical accounting, but maintain current published host residency and cumulative publication totals at successful ownership transitions under the cache's existing synchronization. Cover eviction settlement, load completion, store publication, invalidation, detached transfer cleanup, and host removal; count physical blocks once. Copy completion can advance copy bytes even when later publication is rejected. Keep these counters independent of the optional external metrics reporter.

The host source remains resident while it is valid during reload. Reservation alone never creates residency. Do not reconstruct residency from `used - eviction_ref - load_ref`, because reference sets overlap. Do not scan all tree keys on each 1Hz query. Snapshot collection copies bounded pool/group records under established lock ordering and releases cache locks before serialization or TP communication.

Counters are cumulative within a service instance. Capture baselines before seed/fill; an instance change or counter decrease terminates the continuous epoch and invalidates evidence crossing it. Watermark trigger counters preserve short high-water events that 1Hz polling could miss. Successful-copy and publication counters retain separate units: physical blocks and bytes; packed host blocks are not assumed equal to all device-pool block counts.

### 4. Freeze FULL targets and evaluate group semantics

Build the target inventory before sending requests. Every target rank's FULL group is required; SWA/LINEAR is observe-only. Deduplicate physical pools, retain group mappings, and reject missing FULL groups, host tiers, capabilities, zero capacities, invalid watermarks, or known unreachable thresholds. Do not turn unavailable required fields into observe-only targets.

For shared TP addressing, freeze owner/participant instances, the complete publication scope, and compatible pool tags/group mappings/geometry/capacities. Keep raw peer allocator counts intact. The evaluator attributes logical occupied slots, pending releases, trigger/publication counters and published host residency to the named owner, then checks every participant against its validated physical capacity and watermarks. Unsupported asymmetric layouts fail explicitly. Root settlement certifies a published block only after all declared workers complete that block successfully; local copy completion alone is insufficient. Missing or changed participants, missing owner, scope changes and mismatched layouts cannot qualify. Derived target/manifest metadata records the evidence source and mappings, and admission budgeting uses that same ownership contract without multiplying shared slots.

For pool p, use `effective_used[p] = used[p] - pending_release[p]`, high blocks `ceil(capacity[p] * high_ratio[p])`, and low blocks `floor(capacity[p] * low_ratio[p])`. A FULL group's stable GPU predicate is true if any associated device pool meets its low-block threshold minus the frozen block tolerance. Above-high occupancy is evidence to report, not a loss. Each group also needs current-epoch DEVICE high-watermark trigger evidence, positive GPU-to-host publication delta, and host residency at least `ceil(host_capacity * host_target_ratio)`. `watermark_trigger_count` counts only DEVICE transitions; HOST/DISK remain separately observed. STORE and EVICT publications both qualify after the DEVICE trigger.

Coverage results expose cumulative `epoch_reasons`, including earlier fill losses, separately from clipped per-window coverage and validity. The latest-started epoch pointer is atomically published at directory creation; preflight failures retain their own evidence there.

The initial defaults are:

| Setting | Value |
| --- | --- |
| Host published residency ratio | 0.25 |
| Poll period | 1 second |
| Maximum sample gap / age | 2 seconds each |
| Continuous steady coverage / total steady timeout | 3 seconds / 60 seconds |
| Measurement duration | 60 seconds |
| Minimum valid coverage | 0.95 |
| Maximum consecutive invalid or unknown duration | 2 seconds |
| Kineto actual step target | 20 |
| Maximum support after measurement | 60 seconds |

Settings without an approved universal default remain explicit experiment configuration: fill timeout/request budget/no-progress timeout, fill request length, GPU tolerance, Nsight duration, and activity/drain/capture deadlines. Resolve them from the existing profile/CLI/config chain before launch; reject unresolved or invalid values. This is an intentional startup contract, not permission to invent adaptive thresholds during measurement. Input generation uses a frozen random stream and records actual token-stream identity.

### 5. Use conservative intervals and a common client timeline

Retain client send/receive monotonic times for each rank-local sample and server monotonic timestamps. Use bounded request timing to estimate each rank's clock offset range; rank-zero fan-out reports internal per-rank timing so the client can propagate uncertainty. Do not compare raw monotonic clocks across processes. A sample whose age cannot be bounded within the configured maximum is unknown.

For each rank/group, adjacent valid samples with an allowed gap establish a candidate interval on that server's clock. First merge contiguous candidate intervals, then conservatively map the outer boundaries of each maximal valid segment using clock-alignment uncertainty, and finally intersect required rank/group timelines. Do not independently shrink every adjacent interval and create artificial gaps at shared sample boundaries. Clip the intersection to the requested window. Missing samples, restarts, resets, stale data, and uncovered edges contribute no valid time. Coverage is intersection duration divided by window duration; longest invalid duration is the largest connected component of its complement. Sample-count averages and unlimited last-value hold are rejected alternatives.

Shared-owner counter validity remains on the owner's clock and is intersected with every participant's freshness/layout-valid intervals. Never re-stamp owner measurements onto peer clocks or substitute owner data for missing peer responses. This retains all-rank coverage without inventing replicated allocator state.

Use the same interval evaluator for steady qualification, measurement, and profile windows. At 1Hz, steady needs coverage of a full 3 seconds, normally four samples. Preserve raw samples so results can be recomputed. Supply/progress is a separate gate based on dispatch records, queued/running activity, active steps, and allocation/publication progress with frozen deadlines; lack of a completion during an active long prefill is not by itself starvation.

### 6. Run a bounded phase machine with independent controllers

```text
validate -> seed -> fill -> steady -> measurement -> support (if needed) -> drain
                       |       |            |               |
                       +-------+------------+---------------+--> failure cleanup
```

The continuous producer keeps at most a configured bounded number of outstanding jobs, using the existing transport and request construction primitives. Workers do not wait on profile controls or artifact export. Generate formal grid epochs lazily, shuffle deterministically, attach unique nonce/family identity, and replenish jobs as completions arrive. The arrival schedule remains cumulative across steady/measurement/support so backpressure appears as schedule lag. The dispatch-time phase check handles jobs queued before a boundary but sent afterward.

Seed requests remain separate. Fill uses independent unique prefixes/suffixes and bounded feasible batches; all-cold grids still fill. Successful requests naturally publish and release cache ownership. Once fill passes, use only the formal workload in steady and measurement; never refresh/pin seeds or inject fill to repair a measured loss. Steady must qualify before measurement and is bounded by 60 seconds by default.

For fixed batching, a deliberate fill-boundary drain can complete a bounded fill batch; steady-to-measurement/profile never drains. At final supply cessation, include support submissions in the tail count and pad only to complete the fixed batch. FIFO never receives padding or fixed-batch scheduler controls. Record padding failure as execution failure.

At t1 stop new measurement sends. If collection remains unfinished, continue formal-distribution support until collection completion or the earlier capture/support deadline. Continue sampling through cleanup so the final measurement edge can be evaluated. Set per-request transport deadlines and a global drain deadline, cancel RPCs and stop producer work on expiry, and avoid an executor context manager that waits indefinitely for blocked workers. Cleanup attempts profile stop/export before service destruction and retains all failure reasons.

### 7. Record actual execution boundaries on the executor thread

Add an opt-in bounded recorder near `NormalEngine`/`NormalExecutor`. Allocate session-scoped step IDs only for actual nonempty forwards after TP input metadata confirms execution. Carry the rank-zero execution identity through TP metadata so peer ranks share the step ID; attach authoritative request membership at rank zero and reference it from peer records. Do not infer membership from client completion order or unsynchronized local counters.

Record start/end boundaries, forward entry/return, monotonic wall time, same-executor-thread CPU time, context/decode counts, executed tokens, and explicit waits. Measure a host-forward gap from previous forward return to next forward entry; use CPU-clock readings at those same points. A bounded ring is retrieved with sequence cursors; overflow is explicit and makes affected timing evidence incomplete, never silently drops samples. Long-lived summaries use streaming accumulation or bounded documented quantile estimation with sample counts.

Phase markers applied on the engine thread and client/server time mapping define windows. Only full steps with all-measurement members enter formal distributions, and adjacent-gap statistics require adjacent qualifying steps. Keep mixed/padding/support/boundary steps separately. A pure support step inside a profile can explain structure, but cannot enlarge the measurement cohort. Short traces cannot establish representative p99; baseline distribution reporting carries sample sufficiency metadata.

### 8. Extend profile lifecycle instead of replaying one request

Reuse `/start_profile` with all-rank enablement and 20 actual steps by default. Add public/internal `StopProfile` control to the RPC service, scoped to the diagnostic session/trace identity. A stop request only signals cancellation; the owning engine thread services it at safe execution and idle-loop boundaries before early return. This preserves Kineto enable/disable thread affinity even if request supply stops. A stuck model call is bounded by external shutdown, and cleanup failure is reported rather than claimed successful. Existing single-case profile helpers retain their behavior.

The control worker records request/acknowledgement times and waits for all target-rank artifacts asynchronously. Validate GPU kernels, CPU ranges, CUDA correlation, rank coverage, step count, and at least one complete internal adjacent pair. Report collected and usable counts separately and exclude initial/flush boundaries. A nonempty JSON file is not sufficient. Baseline and profile run separately and compare observed pressure/distribution rather than presumed identical cache state.

Online Kineto artifact waiting checks status, identity and file readiness without parsing trace JSON. After request drain, a terminable child process loads and validates the files within the original capture deadline's remaining budget; timeout terminates and reaps the child and retains raw artifacts. Analysis never extends measurement or support. Real Flash traces exceed 1GB across four ranks, and in-process JSON parsing can starve the sampler despite using a separate Python thread. GPU work with a unique CUDA runtime correlation launched outside every forward/step scope is retained separately as non-forward activity. Missing or ambiguous correlation remains invalid, and every analyzed forward still requires correctly associated GPU work.

Nsight reuses launch sessions and start/stop/export wrappers, with explicit duration. Add diagnostic-only NVTX step and critical-stage markers alongside ATen ranges, gated so normal execution does not emit them. Reconcile CUDA correlation, copy, communication, and multiple streams before labeling exposed GPU gaps; missing evidence yields unavailable/uncertain attribution, not fabricated CPU cost. Kineto and Nsight are mutually exclusive collectors.

Measure the configured Nsight duration from successful start acknowledgement, since command setup latency is not collected workload time. Keep the pre-start call time as the conservative pressure-envelope boundary, record acknowledgement/backend timing separately, and retain the original capture/support deadlines.

Nsight normalization and normalized-artifact serialization also run after drain in a terminable, reaped exec process, bounded by the original capture deadline's remaining budget. A fixed five-second daemon timeout was insufficient for real Flash reports and allowed output writes to continue after failure. Preserve report/SQLite and clearly identified partial analysis artifacts on timeout; record phase durations and artifact sizes without extending pressure or capture windows.

For the observed Nsight exporter mixed-clock NVTX tail, convert only ranges whose raw bounds fit the absolute capture interval and not its relative interval, using unique consistent session-start and analysis metadata. Retain raw values and conversion provenance. Validate repaired boundary-step CUDA correlations but exclude these steps from usable pairs because completeness is not established; the same anomaly in an internal step is fatal. Missing/inconsistent metadata or ambiguous/out-of-range timestamps remain invalid. No magnitude guess, arbitrary tolerance or unconditional final-step exemption is allowed.

### 9. Keep diagnostic persistence outside exact-reuse metrics

Write `pressure/manifest.json`, append-only `pressure/cache_samples.jsonl`, `pressure/steps.jsonl`, and `pressure/summary.json`, plus incremental diagnostic request records. Reuse backend-specific `timelines/` and `nsys/` output organization. Flush evidence incrementally and write manifests/checkpoints atomically so exceptions retain useful output.

Freeze run/profile/grid hashes, code identity, input identity, targets, capacities, thresholds, clock uncertainty, epochs, and all phase/control boundaries. Separate measurement/profile pressure validity, execution status, trace validity (not applicable for baseline), reuse exactness, and diagnostic-only labeling. Measurement request membership is dispatch-based; window throughput excludes late/drain completions and support/padding. Diagnostic requests never enter the original `metrics[]` exact-reuse fitting path. Resume creates a new epoch and redoes seed/fill/steady.

## Risks / Trade-offs

- Settlement instrumentation can miscount shared or detached blocks -> test ownership transitions and use existing locks; never subtract overlapping references.
- Instrumentation can perturb pressure or latency -> keep polling bounded, gate step/NVTX recording, record query cost, and compare separate baseline/profile runs.
- Conservative clock mapping can reject an otherwise loaded run -> preserve uncertainty and classify unsupported intervals as unknown; never relax thresholds after seeing results.
- FULL-only capacity does not imply a host-hit workload or healthy SWA pressure -> report actual reuse and observe-only tiers; execution failures still invalidate the run.
- Fixed batching plus cancellation can strand a partial batch -> compute tail padding after final support submission and bound request/drain deadlines.
- Detailed step membership can overflow buffers -> paginate a bounded recorder, flush incrementally, and flag gaps explicitly.
- Kineto may stop before 60-second measurement ends -> keep the measurement sender running and evaluate the actual capture window separately.
- Test execution guidance named in the source plan is not present as a local skill in this checkout -> use the discovered README commands and Bazel target metadata, recording toolchain/GPU constraints; do not claim runtime acceptance from planning validation.

## Migration Plan

1. Land additive cache snapshot/counter and RPC support with unsupported defaults for other engines; ordinary status probes remain unchanged.
2. Land isolated pressure evaluation, bounded workflow, CLI/profile configuration, and diagnostic output behind explicit preload opt-in.
3. Land step recording and thread-safe continuous profile lifecycle, then backend-specific validation and analysis.
4. Run focused unit/interface suites, GPU small-capacity smoke, and independent baseline/Kineto/Nsight acceptance runs before documenting the mode as implemented.
5. Roll back operationally by disabling preload diagnostics; legacy case and finite continuous modes remain usable. Preserve diagnostic schema/epoch metadata so old evidence is not mixed with new runs.

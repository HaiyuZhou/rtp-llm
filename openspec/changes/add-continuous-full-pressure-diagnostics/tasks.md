# Tasks

Implement against `design.md` and all three `specs/*/spec.md` files. The approved source plan is `rtp_llm/test/perf_test/cache_grid/docs/continuous_full_pressure_plan.md`. All implementation tasks and GPU experiment acceptance are reviewed and verified. Whole-branch review and its scoped fix review passed at `9b937dbf673fcf8283524917892358474ee68ce7`; the existing branch/worktree is retained without merge, push, or archive.

Paths beginning with `runner/`, `tests/`, or `docs/` below are relative to `rtp_llm/test/perf_test/cache_grid/`. New file names are proposed implementation locations. Use focused regression tests for each task, demonstrate the relevant failure before its fix, and retain existing behavior outside explicit preload diagnostics.

## 1. Cache Snapshot and Publication Evidence

- [x] 1.1 Add diagnostic snapshot value types and accessors in `rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.{h,cc}` and the nearby metrics snapshot definitions, including rank-local physical pool identity, group mappings, watermarks, used/free/available, and pending releases; verify `BlockTreeCacheTest.cc` fixtures distinguish evictable occupancy, overlapping references, and shared-pool deduplication.
- [x] 1.2 Add current-epoch watermark-trigger and successful copy/publication/failure counters around `evict/BlockTreeEvictor.cc` settlement, independent of the external metrics reporter; verify eviction tests cover any-pool triggering, pending-release subtraction, copy-success-with-detachment, rollback, and a disabled metrics reporter.
- [x] 1.3 Maintain published host residency through eviction/load/store publication, valid-source reload, invalidation, detachment, and removal in the existing ownership paths; verify cache/load/store tests prove reservations never count as published data, valid host sources persist until release, and cumulative publication remains after residency decreases.
- [x] 1.4 Document snapshot units, deduplication, lock ordering, and copy/publication/residency distinctions in `docs/reference.md`; verify the documented fields match serialized fixtures and run the affected cache and eviction test targets without adding per-poll tree traversal.

## 2. Diagnostic Protocol and Rank Collection

- [x] 2.1 Add versioned `GetCachePressure`/`GetCachePressureInternal` protocol messages and local-only internal routing in `rtp_llm/cpp/model_rpc/proto/model_rpc_service.proto` and `LocalRpcServer.{h,cc}`, with default unsupported engine behavior and `NormalEngine`/`KVCacheManager` accessors; verify protobuf generation and `LocalRpcServerTest.cc` cover supported and unsupported engines.
- [x] 2.2 Add bounded rank-zero fan-out using the existing TP broadcaster, returning requested/returned ranks, per-rank timestamps/sequences, timing bounds, errors, and bounded step-cursor fields; verify RPC tests cover missing ranks, stale responses, no recursive fan-out, and non-atomic acknowledgement semantics.
- [x] 2.3 Add session-scoped enable/disable and phase-marker control for diagnostics, with actual per-rank application times and idempotent cleanup; verify duplicate cleanup and mismatched-session controls do not mutate another session.
- [x] 2.4 Preserve ordinary `GetCacheStatus` and `/worker_status` cost and semantics; verify existing RPC/worker-status tests and spies show no diagnostic snapshot, step history, or cache-key scan on ordinary health requests, and document the optional diagnostic protocol in `docs/reference.md`.

## 3. Frozen Configuration and Pure Pressure Evaluation

- [x] 3.1 Create `runner/pressure_diagnostics.py` with immutable resolved settings, snapshot parsing, and explicit unsupported/missing-field errors; verify new `tests/pressure_diagnostics_test.py` fixtures encode all confirmed defaults and reject invalid/unresolved fill, tolerance, capture, and cleanup settings.
- [x] 3.2 Implement the frozen FULL-only target inventory and group predicate using current-epoch trigger/publication evidence, any-device-pool effective occupancy, and published host residency; verify no-FULL rejection, missing host, per-rank failure, SWA/LINEAR observe-only behavior, pool deduplication, rounding, and 25% host thresholds.
- [x] 3.3 Implement server-clock valid-segment merging, conservative common-time mapping, all-required-target intersection, window clipping, coverage, and longest-invalid duration; verify tests cover four nominal samples spanning 3 seconds, nonzero clock uncertainty without artificial internal gaps, stale/absent samples, resets, uncovered edges, and 95% coverage with a failing 3-second loss.
- [x] 3.4 Implement independent activity/supply assessment with frozen progress deadlines and explicit unavailable fields; verify active long-prefill fixtures do not fail solely for lack of completions while real client starvation and stalled execution remain detectable, and document the distinction in `docs/reference.md`.

## 4. Bounded Continuous Request Lifecycle

- [x] 4.1 Create `runner/continuous_pressure.py` and integrate it as an opt-in runner path, reusing request construction and transport; verify fake-transport tests show bounded outstanding jobs, lazy token generation, cumulative constant/poisson schedules, preserved schedule lag, and auditable sequence/epoch/family/input identity.
- [x] 4.2 Implement seed and unique capacity-fill requests with real one-token prefill semantics and explicit resource/time/request/no-progress limits; verify all-cold grids still fill, fill inputs are isolated from formal seeds, unreachable pressure ends with pressure_not_reached, and admission/token/KV budgets are respected.
- [x] 4.3 Implement steady-to-measurement transition using the pure evaluator, default 3-second steady coverage, 60-second steady deadline, and 60-second measurement; verify transitions retain in-flight requests without cache clearing/restart, label at actual dispatch, and never reset arrival schedules or inject fill into measurement.
- [x] 4.4 Implement scheduler-specific final tail handling across measurement/support and bounded drain/cancellation; verify fixed batches are padded exactly once at final supply cessation, padding failures invalidate execution, FIFO sends no padding, and blocked transports cannot keep cleanup alive beyond resolved deadlines.
- [x] 4.5 Add lifecycle tests to `tests/continuous_pressure_test.py` and retain `batch_decode_test_test.py::ContinuousPressureTest` compatibility coverage; document phase transitions, drain semantics, and finite-mode preservation in `docs/reference.md`, then run both focused suites.

## 5. CLI Wiring and Diagnostic Results

- [x] 5.1 Wire preload/settings through `cache_perf.py`, `perf_config.py`, `batch_decode_test.py`, profile freezing, and runner construction; verify `cache_perf_test.py` and config tests cover defaults, DP/transport/workspace/materialization/shared-seed rejection, mutually exclusive time/count stopping, and immutable resume settings.
- [x] 5.2 Add incremental request/cache output and atomic manifests under `pressure/`, with separate execution, measurement/profile pressure, trace, reuse, and diagnostic-only fields; verify schema tests preserve partial evidence on failure, do not accumulate unbounded request lists, and record phase/rank/epoch/config/input identity.
- [x] 5.3 Implement dispatch-cohort request metrics, crossing flags, in-window completion throughput, and separate support/padding/drain reporting; verify boundary tests exclude late/drain completions from throughput and retain their request outcomes without assigning queued-but-unsent work to measurement.
- [x] 5.4 Keep diagnostic output out of original exact-reuse metrics and formula fitting, and restart seed/fill/steady in a new epoch on resume; verify `prefill_formula_fit_test.py` rejects diagnostic inputs and resume tests never join discontinuous timelines, then document output/status semantics in `docs/reference.md`.

## 6. Actual Step Recording and Baseline Analysis

- [x] 6.1 Add an opt-in bounded step recorder near `rtp_llm/cpp/normal_engine/NormalEngine` and `NormalExecutor`, carrying authoritative rank-zero step identity through existing TP input metadata and request membership through diagnostic records; verify engine tests count actual nonempty forwards, share step IDs across TP ranks, and flag unexpected decode.
- [x] 6.2 Record monotonic execution/forward boundaries and same-thread CPU readings with explicit waits; verify deterministic recorder fixtures distinguish wall and thread-CPU deltas and preserve the previous-forward-return to next-forward-entry interval without introducing GPU synchronization.
- [x] 6.3 Expose recorder pages and phase markers through the diagnostic session API, including cursor overflow and active-step progress; verify bounded buffer tests report missing records explicitly and retain idle/cancellation servicing without falsely incrementing step counts.
- [x] 6.4 Add step classification and baseline distribution logic in the diagnostic runner modules; verify fixtures exclude partial-window, mixed, padding, and support steps, never bridge an excluded step into an adjacent-gap sample, and label small-sample p99 as insufficient. Document timing definitions and bounded aggregation in `docs/reference.md`.

## 7. Continuous Kineto Lifecycle

- [x] 7.1 Add session/trace-scoped public/internal stop-profile RPCs and engine-thread cancellation servicing in `rtp_llm/cpp/engine_base/TorchProfiler.{h,cc}` and `NormalEngine`; verify profiler/RPC tests cover stop before start, stop during execution, stop while idle, same-thread collector shutdown, wrong-session rejection, and idempotent cleanup.
- [x] 7.2 Create `runner/pressure_profile.py` with an independent control worker using the existing start-profile protocol and all-rank 20-step default; verify fake blocking acknowledgements and file waits do not occupy sender workers, and actual control/rank boundaries are recorded.
- [x] 7.3 Add continuous trace validation for rank coverage, parseability, GPU kernels, CPU ranges, CUDA correlation, observed steps, and usable internal adjacent pairs; verify fixtures reject nonempty-but-unusable traces, missing ranks, incomplete files, and startup/flush contamination while reporting collected versus usable counts.
- [x] 7.4 Integrate support continuation up to t1 plus 60 seconds, earlier capture deadlines, independent profile-pressure evaluation, explicit stop, and partial-artifact retention; verify timeout/short-capture tests keep the fixed measurement window intact and preserve baseline evidence, and document the continuous Kineto workflow in `docs/reference.md`.

## 8. Continuous Nsight and Trace Attribution

- [x] 8.1 Extend existing Nsight launch/start/stop/export helpers to a bounded explicit-duration continuous window and make backend selection mutually exclusive; verify mocked command tests keep supply active, honor the support cap, and attempt stop/export before service destruction on success and failures.
- [x] 8.2 Add diagnostic-only NVTX markers for step identity and critical executor/cache stages alongside existing ATen ranges; verify build guards preserve non-diagnostic behavior and trace fixtures distinguish visible Nsight markers from record-function-only events.
- [x] 8.3 Add correlation-based analysis that separates foreground CPU wall/CPU intervals from exposed GPU gaps while considering copies, communication, and multiple streams; verify synthetic traces with overlapping preparation, delayed submission, and stream dependencies produce distinct explanations or explicit unavailable attribution rather than subtracting GPU duration from wall time.
- [x] 8.4 Enable continuous profiling through all three CLI/startup/runner validation layers while preserving ordinary case replay/retest; verify `cache_perf_test.py`, pipeline tests, and existing single-case profile tests, then document separate baseline/Kineto/Nsight invocations and required explicit experiment settings in `README.md` and `docs/reference.md`.

## 9. End-to-End Acceptance

- [x] 9.1 Run the documented Python regression commands and affected C++ cache/RPC/profiler/engine targets with the required repository toolchain and target GPU settings; verify all results are recorded with exact commands, versions, and any environment limitations, including a cross-module failure-path test from snapshot loss through final manifest.
- [x] 9.2 Run a small-capacity GPU/host smoke with FULL attention, real demotion/publication, all-cold and hit/cold workloads, and both FIFO and fixed batching; verify every required rank/group has pressure evidence, no decode occurs, phase transitions retain supply, and all requests/transfer work finish or cancel boundedly.
- [x] 9.3 Run independent baseline, Kineto, and Nsight experiments from the same frozen workload configuration; verify default time coverage and longest-loss gates, actual per-rank step/trace completeness, diagnostic-only result isolation, and a report comparing observed pressure/distribution and collection overhead.
- [x] 9.4 Exercise missing-host/no-FULL, fill timeout, steady timeout, missing/stale rank, counter reset, pressure loss, padding failure, profile timeout, and stop/export failure across integration fixtures and targeted GPU cases where required; verify each retains stage-specific evidence and never reports an invalid run as a valid full-pressure result. Update the source plan's implementation status only after these acceptance results are available.
- [x] 9.5 Reconcile real-runtime contracts: expose optional constant-time waiting/loading activity; declare TP shared metadata ownership and all-participant publication scope from the actual cache/transfer configuration; preserve raw peer-local counters while freezing every participant's identity/layout and owner evidence provenance. Verify missing/stale/restarted peers, missing owner, scope/layout mismatches, partial transfer failures and owner-clock boundaries cannot qualify, then repeat actual Flash acceptance. Earlier completed components remain implemented; this runtime integration requirement has passed review and actual Flash verification.

## Verification Commands

The README documents these Python commands; run them during apply, not as a substitute for implementation:

```bash
python3 -m unittest discover -s rtp_llm/test/perf_test/cache_grid/tests -p '*_test.py'
python3 -m unittest rtp_llm.test.perf_test.batch_decode_test_test
```

Focused existing Bazel target identities include the following. Use the repository's configured CUDA/toolchain options and respect each BUILD target's GPU tags; these lines identify targets and do not assert this environment is ready to execute them:

```text
//rtp_llm/cpp/cache/block_tree_cache/test:block_tree_cache_test
//rtp_llm/cpp/cache/block_tree_cache/evict/test:block_tree_evictor_test
//rtp_llm/cpp/cache/block_tree_cache/evict/test:full_eviction_test
//rtp_llm/cpp/cache/block_tree_cache/evict/test:full_swa_eviction_test
//rtp_llm/cpp/model_rpc/test:local_rpc_server_test
//rtp_llm/cpp/normal_engine/test:engine_test
//rtp_llm/test/perf_test:cache_perf_test
//rtp_llm/test/perf_test:prefill_formula_fit_test
```

Add focused new test files to their owning BUILD targets in the same task that introduces them. Extend the existing `StepWindowProfilerConfigureTest.cc` and `TorchProfilerTest.cc` coverage for lifecycle changes. Implementation and GPU verification evidence is recorded in /data0/zhouhaiyu.zhy/tmp/continuous-full-pressure-20261020-flash/acceptance-final-report.md and acceptance-matrix.json.

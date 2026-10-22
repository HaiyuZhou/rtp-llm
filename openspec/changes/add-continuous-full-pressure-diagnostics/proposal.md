# Proposal

## Why

The existing cache-perf continuous sender exercises real prefill requests but cannot prove sustained cache capacity pressure or capture a continuous profile. Implementing the approved `rtp_llm/test/perf_test/cache_grid/docs/continuous_full_pressure_plan.md` will make FULL-attention pressure, completed GPU-to-host publication, and gaps between actual execution steps measurable without contaminating exact-reuse performance fitting.

## What Changes

- Add opt-in, structured per-rank/group/pool diagnostics with physical occupancy, pending releases, watermarks, published host residency, and distinct copy/publication counters. Keep normal health probes inexpensive.
- Add a bounded seed/fill/steady/measurement/support/drain workflow driven by real requests and the existing FIFO or fixed-batch scheduler. Preserve continuous supply across measurement and profiling boundaries.
- Require every target TP rank's FULL groups to satisfy the frozen pressure contract; observe SWA/LINEAR without using their occupancy as pass/fail gates. Reject models with no FULL groups.
- Declare cache metadata ownership explicitly. For the existing TP0-owned tree with shared block addressing, retain every rank's raw local snapshot and use owner-settled logical occupancy/publication only with complete participant identity/layout validation and an explicit all-participant transfer-completion scope.
- Freeze the confirmed defaults: 25% host residency, 1-second polling, 2-second maximum sample gap/age, 3-second steady coverage, 60-second steady timeout and measurement, at least 95% valid time coverage, at most 2 seconds of consecutive invalid/unknown time, 20 Kineto execution steps, and at most 60 seconds of support after measurement.
- Add continuous Kineto and bounded-time Nsight collection, engine step/request correlation, baseline wall/thread-CPU statistics, and profile-based GPU gap attribution.
- Write independent diagnostic manifests, pressure samples, step records, and summaries; report execution, pressure, trace, and reuse validity separately. Preserve original case profiling, finite continuous runs, exact-reuse validation, and formula-fit inputs when the new mode is disabled.

## Capabilities

### New Capabilities

- `cache-pressure-observability`: On-demand evidence of cache capacity, valid residency, and completed migration across ranks, groups, and tiers.
- `continuous-cache-pressure`: Real-request FULL-attention pressure establishment, bounded continuous windows, time-based validity, and isolated diagnostic results.
- `prefill-step-diagnostics`: Actual prefill-step timing and request mapping, uninterrupted continuous profiling, and correctly scoped CPU/GPU attribution.

### Modified Capabilities

None. The OpenSpec capability inventory is currently empty; compatibility with existing implementation behavior is specified in the new capabilities.

## Impact

- Cache layer: `BlockTreeCache`, `BlockTreeEvictor`, load/store settlement, pool snapshots, and `KVCacheManager`.
- Service layer: additive diagnostic protocol/routing in `model_rpc_service.proto`, `LocalRpcServer`, and engine interfaces, with per-TP-rank collection and capability errors.
- Engine layer: `NormalEngine`, `NormalExecutor`, step-profiler lifecycle, and optional NVTX markers.
- Tooling: `cache_perf.py`, `cache_grid_runner.py`, `batch_decode_test.py`, `perf_config.py`, frozen profiles, result analysis, and nearby Python/C++ tests.
- Requires a real GPU/host-tier setup for smoke and baseline/profile validation; reuses the existing Kineto/Nsight toolchains. No new simulator, scheduler, standalone configuration format, or changes to the C++ cache benchmark are included.

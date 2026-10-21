# Proposal

## Why

Continuous FULL-pressure runs currently use a manually chosen, small seed-family count, so repeated prefixes can remain GPU-hot while one-shot fill and suffix blocks produce mostly GPU-to-host traffic. The workload needs to derive its reusable working set from the live cache topology and keep FIFO supplied deeply enough to exercise host-to-GPU reloads under production-sized cache capacity.

## What Changes

- Derive the effective seed-family count after startup from the authoritative `GetCachePressure` FULL-group topology, capacities, ownership, block spans, watermarks, and host retention budget. An explicit count remains available only for smoke/debug runs and makes the result non-formal.
- Require a formally reusable prefix span of at least 32K tokens, aligned to the grid/cache contract, and reject formal grids that never request that span. Keep the seed commit tail separate from the reusable span.
- Use an aligned 100K-token unique fill request by default to establish capacity pressure; do not count its one-shot suffix as reusable seed-bank state.
- Replace seed batch barriers with bounded FIFO rolling supply. Keep at least 16 client requests in flight by default for formal runs, replenish on completion, and distinguish client backlog from the server's actual active batch.
- **BREAKING** Restrict the full-pressure preload workflow to FIFO scheduling. Fixed-batch scheduling, padding, and batch-tail acceptance are removed from this workflow; ordinary case execution and non-pressure batch scheduling remain unchanged.
- Distinguish production-formal runs from deliberately reduced GPU smoke runs. Formal runs reject forced test block capacity and client concurrency below 16; smoke runs retain explicit smaller overrides and are labeled non-formal.
- Require measurement-window evidence of both successful GPU-to-host publication and successful host-to-GPU LOAD publication for bidirectional migration validity, separately from execution and general FULL-pressure validity.

## Capabilities

### New Capabilities

- `adaptive-fifo-cache-pressure`: Live-capacity seed planning, reusable-prefix validation, bounded FIFO pressure supply, production/smoke classification, and bidirectional cache-migration qualification.

### Modified Capabilities

None. The durable capability inventory is empty; this capability builds on the completed but unarchived `add-continuous-full-pressure-diagnostics` change and must be integrated after that change.

## Impact

- Pressure CLI and profile resolution in `rtp_llm/test/perf_test/cache_grid/runner/cache_perf.py` and related configuration tests.
- Seed/fill request construction, live topology preflight, manifests, and summaries in `cache_grid_runner.py`.
- FIFO sender lifecycle, concurrency semantics, phase timeouts, and migration gates in `continuous_pressure.py` and `pressure_diagnostics.py`.
- Continuous-pressure unit/integration tests, DeepSeek-V4-Flash production-profile GPU acceptance, example profiles/grids, and Chinese/reference documentation.
- No cache RPC schema change is expected: the existing `GetCachePressure` response already carries the required ownership, pool, group, watermark, residency, and transfer counters.

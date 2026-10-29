# Design

## Context

See `proposal.md` for motivation and the three delta specs for observable behavior.

The unified perf entry already admits only `decode + local + run`, freezes decode cases into an execution plan, and writes common case/result artifacts. Its decode implementation is not unified internally: fixed and distribution runners duplicate topology, TPOT, failure, and result handling; TPOT search also owns batch execution; the HTTP batch runner owns scheduling, warmup, measurement, transport, and parsing; and multi-node execution still calls a second grid runner built on `BatchPerfImpl`.

The current fixed schema has `input_lengths`, one scalar `output_length`, `batch_sizes`, and optional top-level `target_tpot_ms`. A TPOT case is currently first planned at a configured maximum batch, searches candidates during execution, then builds top-level metrics from the maximum batch rather than necessarily from the selected batch. The multi-node path also emits `Decode_Result.json` and infers profile metadata from filenames and logs.

The refactor must preserve the existing prefill implementation and capability matrix, keep distribution output scalar, retain global-batch DP accounting, retain optional decode timeline capture, and support both a locally owned model session and an externally managed multi-node service.

## Goals / Non-Goals

**Goals:**

- Represent fixed input-by-output grids and exact or target-TPOT batch policies without ambiguous optional fields.
- Make planning, strategy selection, case execution, transport, metrics, profiling, and reporting independently testable.
- Execute fixed, distribution, normal local, and externally managed multi-node cases through one decode core.
- Make the selected TPOT batch authoritative for formal metrics and optional profiling.
- Produce stable shape-complete identities and reports that preserve every grid axis.
- Remove the old callable grid API and result format after all repository consumers migrate.

**Non-Goals:**

- Adding decode resume, standalone profile/report actions, AOne decode, or decode regression publishing.
- Changing prefill configuration, execution, profiling, reporting, or AOne behavior.
- Generalizing prefill and decode into one workload engine.
- Replacing the model server, its scheduler endpoint, or the existing distributed service launcher.
- Persisting and resuming an individual TPOT search candidate; a TPOT case remains the atomic execution unit.

## Decisions

### 1. Use explicit batch policy types

Fixed grid config becomes:

```json
{
  "kind": "grid",
  "input_lengths": [128, 512],
  "output_lengths": [32, 128],
  "batch": {"strategy": "fixed", "sizes": [1, 4, 8]},
  "warmup_runs": 1,
  "measure_runs": 3,
  "generate_config": {},
  "profiling": {"enabled": false}
}
```

Target-TPOT replaces `sizes` with `max_size`, `candidate_step`, and `target_tpot_ms`. Distribution uses the same `batch` and `profiling` objects but retains scalar `output_length` and `distribution_file`. Parsing produces frozen `FixedBatchPolicy` or `TargetTPOTPolicy` values; it never leaves a partially valid dictionary for execution to interpret.

This is intentionally incompatible. Keeping top-level `batch_sizes` and `target_tpot_ms` would preserve ambiguous combinations and require compatibility branches throughout planning. A generic list of dimensions was rejected because it would obscure workload-specific validation and over-generalize this decode-only change.

`generate_config` remains an extension map, but planned output shape is authoritative: request construction applies user generation settings first and then forces `min_new_tokens` and `max_new_tokens` to the case output length.

### 2. Plan typed cases before model startup

`DecodeCaseSpec` carries the case identity, workload kind, materialized input lengths, scalar output length, frozen batch policy, warmup/measurement settings, generation settings, profiling settings, and optional distribution metadata.

Planning behavior is deterministic:

- Exact fixed grid expands `input_lengths x output_lengths x batch.sizes`.
- Target-TPOT fixed grid expands `input_lengths x output_lengths`; each case materializes `max_size` repeated input lengths for candidate slicing.
- Exact distribution materializes the real length list for each configured batch.
- Target-TPOT distribution requires one sequence-length list of at least `max_size`; candidates use prefixes. Explicit per-batch maps are rejected for search.

Case identifiers derive from normalized shape and policy identity, not loop position. Config order determines plan/report order, while equivalent normalized cases retain equivalent identifiers.

Known-invalid values and configured engine bounds are rejected during planning, including non-positive dimensions, input plus output beyond maximum sequence length, and a configured upper batch that cannot satisfy declared concurrency or token bounds. Runtime capacity failures such as allocation failure remain search observations.

An alternative that passed loose dictionaries into the existing runners was rejected because it would keep validation, identity, and execution coupled.

### 3. Separate policy from measurement

`DecodeCaseExecutor` resolves a strategy from the case batch policy. Both strategies receive one callable boundary:

```python
measure(case: DecodeCaseSpec, batch_size: int) -> DecodeBatchMeasurement
```

`FixedBatchStrategy` calls it once. `TargetTPOTStrategy` owns candidate construction, bounded search, observation recording, and final selection, but knows nothing about HTTP, prompts, DP scheduling, warmup, or metric persistence.

The existing candidate sequence is retained: `1`, step multiples through the maximum, and the maximum when it is not already present. Bounded binary search assumes that feasibility under the TPOT target is monotonic enough for capacity discovery. Every attempted candidate is cached and recorded once.

On success, the best satisfying batch is the only measurement promoted to formal case metrics. The executor does not force a maximum-batch measurement after search. On failure, observations are retained and the case becomes `target_tpot_not_met` rather than selecting a diagnostic candidate as a formal result.

Linear exhaustive search was rejected because large batch ranges would make grid cost unacceptable. Keeping search wrapped around the current case runner was rejected because it causes duplicate fixed/distribution behavior and makes selected-batch metrics ambiguous.

### 4. Use one batch executor and a narrow HTTP transport

`DecodeBatchExecutor` performs candidate slicing, DP allocation, per-worker engine-bound checks, scheduler configuration, warmup rounds, measurement rounds, elapsed-time accounting, and construction of `DecodeBatchMeasurement`.

`DecodeHttpTransport` performs only service-facing operations: update scheduler information, construct prompts and requests, send one concurrent request round, and parse per-request response data. It receives an endpoint, `prompt_for_length`, HTTP callable, timeout, monotonic clock, and sleep function through dependency injection.

The existing deterministic contiguous DP allocator is retained. Global batch remains the public workload size; worker allocations may be uneven and may include idle workers. Scheduler configuration uses the largest local batch while measurements retain all worker allocations.

Keeping warmup and metric aggregation inside the HTTP object was rejected because fake execution, non-HTTP reuse, and strategy tests would still require transport-shaped mocks.

### 5. Build one case result for every workload kind

Fixed and distribution runner classes are replaced by one `DecodeCaseExecutor` and one result builder. The builder validates that every measured request matches its planned input and output shape, retains per-run/per-request data, aggregates latency and throughput, attaches search/distribution/profile data, and applies one failure policy.

Exact-batch results report the configured global batch. TPOT results report the selected global batch. Search limits and observations remain under `search`; they are not substituted for the effective batch in top-level metrics.

The result schema version is advanced because fixed workload identities, batch policy, search semantics, report shape, and profile references are incompatible with the existing decode result contract.

### 6. Keep inline profiling inside `run` but outside formal measurement

`workload.profiling` contains `enabled`, `runs`, `steps`, `enable_all_ranks`, and `flush_timeout_seconds`; defaults are disabled, one run, three steps, all ranks, and a bounded flush timeout.

After a successful exact case, the profiler repeats that exact batch. After successful TPOT search, it repeats only the selected batch. It may pre-arm `/start_profile`, adds the server timeline fields to the profile request, assigns a shape-complete trace name containing input, output, and batch, and waits for the server's trace-save signal or artifact discovery.

Each attempt writes `profiles/<case_id>/profile_manifest.json`. Profile failure is recorded in the case and summary but does not invalidate already completed formal metrics or turn a complete performance run into partial. Report generation failure remains fatal under the existing result contract.

Making profiling a decode action was rejected because decode follow-up actions remain out of scope and the old capability was an inline post-measurement pass. Profiling every TPOT candidate was rejected because it would distort search cost and generate traces that do not describe the chosen operating point.

### 7. Generate flat observations and derived matrix views

Finalization first persists common case files and the versioned result envelope, then derives `decode_report.json` only from committed cases. The report contains ordered flat observations for all workload kinds.

For an exact fixed grid it additionally emits one input-by-output matrix per exact batch. For target-TPOT it emits an input-by-output matrix whose cells contain status, best batch, actual TPOT, throughput, and search outcome. Distribution remains a table because arbitrary sequence-length batches do not form a rectangular grid.

Invalid or missing required cases remain visible but are excluded from valid aggregation, making the report diagnostic. A separate decode report action is not added.

### 8. Attach both local and multi-node runtimes to the same core

The normal local executor continues to own model startup and shutdown, then constructs the transport and unified decode executor for its frozen plan.

The multi-node launcher distributes one normalized perf config artifact to all ranks. The existing distributed launcher still owns service setup and shutdown. All ranks validate the config and actual topology; only the master constructs the shared runtime endpoint, executes the frozen plan, finalizes the standard run directory, and signals completion. Worker ranks never execute cases or write final results.

Repository multi-node collection moves from `*Result.json` to the standard run directory. Result merging reads `results/result.json`; trace analysis reads profile manifests rather than inferring shape solely from trace filenames or PrettyTable log text.

A legacy adapter around `run_single` was rejected because the user explicitly chose a clean break and an adapter would preserve the second data/result contract.

### 9. Replace modules rather than retain parallel runners

The target decode package contains typed model, planner, strategy, executor, transport, profiling, scheduler, metrics, report, and finalization modules. `modes/decode.py` remains the plan-level case loop and failure-isolation boundary; `execution/defaults.py` only assembles the runtime dependencies.

After repository consumers migrate, remove `decode/fixed.py`, `decode/grid_runner.py`, `decode/http_batch.py`, `batch_perf_impl.py`, and its now-unused legacy metric structures. Distribution file loading may remain separate, but distribution case execution does not.

## Risks / Trade-offs

- **[TPOT binary search assumes a mostly monotonic boundary]** -> Preserve every observation, document the assumption, and keep the strategy isolated so a different search policy can replace it without changing execution.
- **[The Cartesian output axis can multiply runtime substantially]** -> Expose the complete dry-run case count and shape before model startup; do not silently cap or sample requested cases.
- **[Profile trace export varies by server/runtime]** -> Use a bounded flush timeout, persist a manifest even on failure, and keep formal metrics independent.
- **[Multi-node config and service topology can drift]** -> Validate TP/DP/PP/EP on every rank before measurement and allow only the master to write results.
- **[Breaking config and result formats affect downstream scripts]** -> Migrate every in-repository example, launcher, collector, merger, and analyzer in the same change; provide no dual-write period that could hide stale consumers.
- **[Materializing distribution inputs up to max batch increases plan size]** -> Store only the resolved integer sequence needed by the case and rely on existing small workload-artifact expectations; model data remains external.
- **[Profile failure with a completed run can be overlooked]** -> Surface `profiling_failed_cases` in summary and command output while preserving the distinction from performance failure.

## Migration Plan

1. Add the new config types, strict parser, typed case model, and planner with unit tests while the old execution code still exists as a behavior reference.
2. Add the batch executor, narrow transport, exact and TPOT strategies, unified metrics, and fake-transport tests.
3. Add inline profiling manifests and the matrix report, including failure-path tests.
4. Switch the normal local decode runtime to the unified executor and update examples, docs, dry-run expectations, result schema, and BUILD targets.
5. Migrate the multi-node launcher to the normalized config and shared executor, then migrate collection, merge, and trace-analysis consumers to the standard artifacts.
6. Delete the old callable API, runner modules, legacy result writer, and obsolete tests; add import/target assertions so they cannot return accidentally.
7. Run focused Python/Bazel tests, strict OpenSpec validation, dry runs, and real GPU smoke tests with `Qwen2.5-0.5B-Instruct` for exact grid, target-TPOT, and one inline profile case.

There is no runtime rollback or compatibility mode. Reverting the change restores the old schema and runner; persisted new-format run directories remain self-describing but old tools are not expected to read them.

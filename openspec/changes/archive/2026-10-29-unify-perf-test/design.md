# Design

## Context

See `proposal.md` for motivation and the four delta specs for observable behavior.

The current implementation has two intertwined control planes:

- `batch_decode_test.py` owns the original argument flow, model lifecycle, ordinary prefill path, decode dispatch, and cache-prefill integration helpers.
- `perf_config.py` combines perf arguments, engine arguments, environment mutation, run overrides, and legacy path resolution.
- `test_entry.py` wraps the test with baseline comparison, golden behavior, OSS upload, ODPS publishing, and environment-dependent policy.
- `cache_grid/runner/cache_perf.py` and `cache_perf_orchestrator.py` expose a second CLI and local/AOne workflow.
- `cache_perf_ci.py` materializes CI inputs, estimates case weights, creates generated Bazel packages, shards cases, and collects outputs.
- `cache_grid_runner.py` combines prompt materialization, exact reuse construction, decode Cartesian execution, prefill execution, checkpointing, and profiling.
- `result_schema.py`, formula, and plot modules already contain reusable prefill validation and report logic, but consume cache-perf-specific artifacts.
- `defs.bzl` and `BUILD` bind many targets directly to the old entry files and macros.

The change is intentionally allowed to break old configs and results. The implementation must preserve validated algorithms, but it must not preserve the ownership structure or implicit behavior of the old entry points.

## Goals / Non-Goals

**Goals:**

- Establish dependency direction from CLI and config resolution to a frozen plan, then to execution backends and mode-specific runners.
- Make local and AOne prefill consume the same normalized config, workload, case identity, and result contracts.
- Separate engine lifecycle from workload algorithms so prefill and decode can share process handling without sharing metric semantics.
- Make interruption, resume, profile, partial collection, reporting, and publishing explicit state transitions.
- Migrate verified prompt, reuse, decode, formula, chart, baseline, and publishing logic behind the new contracts.
- Keep the final user-facing switch atomic: one new entry remains and all old entries are removed in the same release change.

**Non-Goals:**

- Designing a general benchmark framework outside the existing perf-test domain.
- Preserving old Python imports, shell commands, Bazel macros, config fields, run directories, or baseline formats.
- Adding AOne decode, decode resume/profile, or an input-length by output-length decode grid.
- Redesigning multi-node performance tooling except where deleted imports require it to consume a new stable API.
- Moving model checkpoints or tokenizer data into generated AOne packages.

## Decisions

### 1. Use a unified control plane with mode adapters and execution backends

The new dependency flow is:

```text
tools/perf_test
  -> CLI
  -> ConfigResolver
  -> CapabilityValidator
  -> ExecutionPlan
  -> Orchestrator
      -> LocalExecutor | AOneExecutor
      -> PrefillAdapter | DecodeAdapter
  -> Result/Report/Publish
```

The control plane owns parsing, validation, plans, states, exit codes, and artifact locations. Execution backends own devices, model process lifecycle, and remote submission. Mode adapters own case generation, request construction, correctness checks, and metrics.

Suggested ownership layout:

```text
rtp_llm/test/perf_test/
  main.py
  config_schema.py
  capabilities.py
  plan.py
  orchestrator.py
  results.py
  execution/
    local.py
    aone.py
    engine_session.py
  modes/
    prefill.py
    decode.py
  prefill/
    grid.py
    planner.py
    runner.py
    checkpoint.py
    profile.py
    report.py
  decode/
    batch.py
    grid_runner.py
    distribution.py
    tpot.py
    report.py
  publish/
    baseline.py
    archive.py
    odps.py
```

This tree is an ownership target, not a requirement to create one file per class immediately.

**Alternative considered:** Extend `batch_decode_test.py` and call cache-perf orchestration conditionally. Rejected because that keeps argument parsing, engine lifecycle, result state, and reporting coupled to historical mode detection.

**Alternative considered:** Keep two top-level commands over shared helpers. Rejected because config, capability, exit-code, and CI behavior would remain duplicated and users would still see two products.

### 2. Use one strict discriminated config schema

The top-level schema is versioned and contains:

```text
schema_version
model
engine
workload
execution
output
regression
publish
```

`model.topology` carries TP/DP/PP/EP. `engine` carries explicit capacity values plus declared `args` and `env` extension maps. `workload.kind` selects `cache_grid`, `grid`, or `distribution`. `execution.backend` selects `local` or `aone` and validates backend-specific fields.

Unknown structured fields fail validation. Environment interpolation, shell evaluation, includes, and secrets are not supported. The source file is copied to the run directory, but only canonical `resolved_config.json` is executable input.

Small external workload inputs are parsed and materialized into the resolved workload. Checkpoint and tokenizer paths remain external runtime dependencies.

**Alternative considered:** Preserve `perf_config.py` argument passthrough and progressively add typed fields. Rejected because the effective model configuration would still depend on argument ordering and ambient environment.

**Alternative considered:** Support old configs through adapters. Rejected by scope and because adapters would force the new schema to encode ambiguous old defaults.

### 3. Freeze an immutable execution plan before side effects

`ExecutionPlan` contains at least:

```text
mode
action
backend
resolved_config
config_fingerprint
workload_fingerprint
cases
resource_plan
result_directory
```

Fingerprints use canonical JSON and SHA-256. Case identifiers derive only from normalized case identity, not list position or runtime order. The plan is persisted before execution and runners cannot reinterpret source config.

Capability validation occurs before the run directory, Bazel package, engine, or remote job is created. Dry-run returns the normalized plan, resource and shard summaries, and command preview without executing it.

**Alternative considered:** Let each runner resolve defaults. Rejected because local and AOne could then generate different cases or fingerprints from the same source config.

### 4. Encode the capability matrix centrally

The central matrix is:

| Mode | Backend | run | resume | profile | report |
| --- | --- | --- | --- | --- | --- |
| prefill | local | yes | yes | yes | yes |
| prefill | aone | yes | no | no | offline after collect |
| decode | local | yes | no | no | generated by run only |
| decode | aone | no | no | no | no |

Mode adapters may assume the matrix has already admitted the request. They still validate mode-specific workload shape.

**Alternative considered:** Distribute capability checks through command handlers. Rejected because rejected operations could perform partial setup before reaching the relevant handler.

### 5. Give one engine session ownership of each local plan

`EngineSession` becomes the only component allowed to start and stop the serving process. It builds the explicit environment and arguments, waits for readiness, exposes a client, captures stdout/stderr, timeline and coredump metadata, and closes the process on all exits.

A local run starts one session and runs all cases through it. AOne starts one session per shard action. Offline report never creates a session.

Signal handling first records interruption and flushes already committed case state, then tears down the engine and returns 130 or 143.

**Alternative considered:** Keep model management in `batch_decode_test.main()` and expose helper callbacks. Rejected because cache, decode, resume, and future mode work would remain dependent on an entry-point module.

### 6. Rebuild prefill around the cache runner algorithms only

The prefill planner owns alignment, commit-tail, maximum input, minimum compute, batch, sampling, seed, and maximum-case constraints. It creates deterministic cases with expected reuse and request-construction metadata.

The prefill runner reuses the proven exact-token prompt construction and shared-seed behavior from `cache_grid_runner.py`. It creates cache state, sends measurements, and validates response shape, observed reuse, and timings. Reuse mismatch is a correctness failure, never a warning.

Ordinary `_run_prefill` behavior from `batch_decode_test.py` is not migrated.

Resume reads the persisted plan and fingerprints, preserves only `ok` cases, and reruns all other planned cases. Profile reconstructs selected original cases and writes under a separate profile session without changing formal results or checkpoints. Report reads persisted cases only.

**Alternative considered:** Keep the existing combined `CacheGridRunner` class. Rejected because it also owns decode Cartesian execution, persistence, request transport, and profiling, which prevents a clean mode boundary. Its algorithms are extracted and separately tested.

### 7. Split decode behavior by workload while retaining old algorithms

The following responsibilities move out of `batch_decode_test.py`:

| Current responsibility | New owner |
| --- | --- |
| `_run_decode` dispatch | decode mode adapter |
| `run_single` fixed case execution | decode grid runner |
| distribution construction/execution | decode distribution runner |
| TPOT/TPS batch binary search | decode TPOT search runner |
| DP request scheduling | decode batch scheduler plus engine session |

Fixed decode grid is input lengths by batch sizes with one scalar output length. Distribution input is resolved during planning and actual sequence lengths are persisted. TPOT search preserves every attempted batch, capacity failure, measured TPOT, and the final largest satisfying batch.

The plan and result distinguish global batch from DP worker allocation. `CacheGridRunner.run_cartesian_grid` is removed after its needed loop behavior is covered in the new decode runner.

**Alternative considered:** Add output-length arrays while migrating the runner. Rejected because decode needs separate feature reduction and validation before multiplying the workload dimension.

### 8. Package AOne plans as generated Bazel runfiles

The submit host creates a generated Bazel package containing:

```text
resolved_config.json
resolved_workload.json
shard_plan.json
BUILD.bazel
```

Targets pass each artifact using Bazel runfiles and location expansion. Remote actions never read the original source-config path. Each shard receives only its immutable ownership plan and returns run/shard identity, config and workload fingerprints, owned/completed case IDs, and case results.

The existing `case_weight` and LPT-style `split_cases` behavior in `cache_perf_ci.py` is retained behind the new planner. Collection rejects missing shards, duplicate or unknown cases, ownership violations, missing declared results, and fingerprint mismatches.

**Alternative considered:** Pass source config and local paths as action arguments. Rejected because AOne workers do not share submit-host filesystem semantics and the effective input would be hard to audit.

**Alternative considered:** Let each shard independently expand the grid. Rejected because shard ownership and full-workload completeness could not be proven centrally.

### 9. Use per-case atomic files and a versioned result envelope

The run directory is:

```text
source_config.json
resolved_config.json
execution_plan.json
manifest.json
workload/
checkpoints/
results/cases/<case_id>.json
results/result.json
results/summary.json
reports/
profiles/
traces/
```

Manifest, checkpoint, case, and aggregate JSON writes use temporary files followed by atomic rename. A run-directory lock prevents two writers. Aggregate results derive only from committed case files.

Run stages are `created`, `planned`, `running`, optional `collecting`, `reporting`, and `completed`, with terminal `partial`, `failed`, and `interrupted`. Case states are `planned`, `running`, `ok`, `request_failed`, `invalid_shape`, `invalid_reuse`, and `invalid_timing`.

The common result envelope contains schema version, run identity, mode, backend, status, fingerprints, model, summary, and mode-specific metrics. Only case identity, status, runs, and errors are common at case level.

**Alternative considered:** Keep one append-only journal as the primary result. Rejected because downstream consumers need deterministic case ownership and atomic independent recovery. A journal may remain an internal diagnostic, but case files are authoritative.

### 10. Separate measurements, reports, regression, and publishers

Formal reports consume only valid shape/reuse/timing measurements. Partial input can produce a diagnostic report, never a formal regression result. Report failure changes run completion state but does not rewrite measurements.

Baseline entries use stable case key, metric, expected value, direction, and threshold. Report can create `baseline_candidate.json` but cannot overwrite the accepted baseline. Golden generation and validation are explicit build/CI operations.

`test_entry.py` behavior is split into baseline, archive, and ODPS modules. Publishers receive finalized summaries and artifact selections. Credentials remain in the runtime environment. Each publisher declares `required`; required failure is a run failure, optional failure is recorded without changing case outcomes.

**Alternative considered:** Keep `test_entry.py` as a wrapper around the new command. Rejected because it would preserve a second policy and exit-code layer outside the unified result state.

### 11. Use stable error classes and exit codes

The orchestrator maps failures to:

| Code | Class |
| --- | --- |
| 0 | complete success |
| 2 | config or capability |
| 3 | plan or resource |
| 4 | engine lifecycle |
| 5 | case failure or partial run |
| 6 | collect, merge, report, or required publisher |
| 130 / 143 | interrupt / termination signal |

Mode runners return typed outcomes rather than exiting directly. This keeps CLI, Bazel, and AOne behavior consistent.

**Alternative considered:** Preserve existing command-specific exit codes. Rejected because CI could not interpret the same failure consistently across modes and backends.

## Risks / Trade-offs

- **[Large replacement touches many targets]** -> Implement contracts and adapters behind tests first, then perform one repository-wide entry-point switch after all release gates pass.
- **[Algorithm changes can invalidate performance comparability]** -> Extract existing prompt, reuse, decode, search, formula, and chart logic with characterization tests before restructuring it.
- **[Strict reuse checks can turn previously reported points into failures]** -> Preserve invalid cases for diagnostics and make the new correctness rule explicit in result and report schemas.
- **[AOne generated packages can accidentally capture submit-host paths]** -> Test runfiles in a clean execution root and reject non-runtime paths during plan validation.
- **[Per-case files increase filesystem operations]** -> Keep payloads small, write once atomically, and aggregate only at checkpoints and completion.
- **[No compatibility layer forces coordinated CI migration]** -> Land new targets and configurations before the atomic switch, but do not expose both as supported user interfaces in the final state.
- **[Existing multi-node code imports `run_single`]** -> Either migrate that call site to a stable decode runner API in the switch or explicitly retain a non-user internal API; do not retain `batch_decode_test.py` for it.
- **[Publisher behavior can obscure successful measurements]** -> Persist measurement completion before publishing and represent publisher failure separately in manifest and summary.
- **[The old OpenSpec change can cause conflicting implementation work]** -> Mark this change as its successor and stop applying `add-cache-perf-local-aone-runners`; archive or close it once this plan is accepted.

## Migration Plan

### Phase 1: Contracts and shared infrastructure

1. Add strict config models, capability matrix, immutable plan, fingerprints, state models, atomic result writer, and run-directory lock.
2. Add result envelope, baseline/golden contracts, publisher interfaces, and fake publisher tests.
3. Extract engine lifecycle into one session abstraction with fake-server tests.
4. Keep old entry points unchanged during this internal build-out.

### Phase 2: Prefill

1. Extract cache grid planning, exact prompt construction, reuse validation, and metric collection behind the new prefill adapter.
2. Implement local run, resume, profile, and offline report on the new run directory.
3. Replace AOne preparation with normalized runfile artifacts, LPT shard plans, and strict collection.
4. Validate local and AOne prefill with unit, integration, and GPU smoke tests.

### Phase 3: Decode and switch

1. Move fixed grid, distribution, DP scheduling, and TPOT batch search into decode-owned modules.
2. Migrate regression, golden, OSS, and ODPS consumers to the common result contract.
3. Add the final `tools/perf_test` entry and new Bazel/CI targets.
4. Migrate internal importers such as multi-node tooling to stable new APIs.
5. Run all release gates and a repository-wide reference scan.
6. In one final change, delete `tools/cache_perf`, `batch_decode_test.py`, `test_entry.py`, `perf_config.py`, old cache CLI/orchestrator/CI planner, old macros, old-only tests, and obsolete documentation.

### Rollback

Before the atomic switch, rollback consists of reverting the new isolated modules and targets because supported old entry points remain unchanged. After the switch, rollback is a source-control revert of the complete switch commit; no runtime data migration rollback is promised because old and new result/config schemas are deliberately incompatible.

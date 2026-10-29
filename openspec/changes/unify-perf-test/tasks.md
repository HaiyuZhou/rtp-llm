# Tasks

## 1. Control-plane contracts

- [x] 1.1 Add characterization tests for reusable cache prompt/reuse construction, decode fixed-grid execution, distribution generation, TPOT search, AOne case weighting/sharding, formula inputs, and existing baseline/publisher behavior; verify the focused legacy test targets pass before extraction.
- [x] 1.2 Create the new perf-test package boundaries and strict versioned config models for model, engine, prefill/decode workload, local/AOne execution, output, regression, and publish sections; verify config tests cover defaults, range/type errors, unknown fields, extension maps, and rejection of interpolation/include/secret forms.
- [x] 1.3 Implement canonical config normalization and SHA-256 config/workload fingerprints, including submit-host materialization of small external workload files; verify equivalent configs hash identically and content changes alter the correct fingerprint.
- [x] 1.4 Implement the centralized mode/action/backend capability matrix and typed failure classes; verify every supported combination is accepted and every unsupported combination fails before filesystem, build, engine, or submission fakes are called.
- [x] 1.5 Implement deterministic case identity, immutable execution plans, and resource-plan validation; verify plan serialization round-trips and case IDs do not depend on input ordering or runtime order.
- [x] 1.6 Implement CLI parsing for `--mode`, `--action`, `--config`, `--run-dir`, `--cases`, and `--dry-run` without exposing a production entry switch yet; verify action source conflicts and persisted-run overrides return exit code 2.
- [x] 1.7 Implement dry-run output for local and AOne plans; verify tests prove it creates no formal run directory, starts no engine, builds no target, submits no job, and invokes no publisher.

## 2. Run state, results, and engine lifecycle

- [x] 2.1 Implement the run-directory layout, source/resolved config and plan snapshots, manifest schema, and exclusive writer lock; verify an isolated run directory is self-describing and a second writer is rejected without mutation.
- [x] 2.2 Implement atomic JSON writes for manifests, checkpoints, per-case files, final results, and summaries; verify interruption/fault-injection tests never expose a truncated file as committed state.
- [x] 2.3 Implement run and case state transitions plus stable exit-code mapping; verify table-driven tests cover complete, partial, failed, interrupted, invalid-shape, invalid-reuse, invalid-timing, engine, and post-execution failures.
- [x] 2.4 Implement the versioned common result envelope with separate prefill and decode metric payloads; verify schema/golden tests reject missing common fields and prevent mode-specific fields from being interpreted as shared metrics.
- [x] 2.5 Extract model startup, readiness, client access, logs, timeline, coredump metadata, signal handling, and teardown into the shared engine session; verify fake-process tests cover success, startup failure, mid-run crash, interrupt, and exactly one startup per local plan.
- [x] 2.6 Implement the local executor and orchestrator dispatch against immutable plans and typed outcomes; verify fake prefill/decode adapters receive one session and cannot access source-config parsing.

## 3. Local prefill cache performance

- [x] 3.1 Extract and test prefill grid planning for alignment, commit-tail tokens, maximum input, minimum compute, fixed/random cache lengths, batch sizes, deterministic sampling, and maximum case count; verify planner tests include every invalid boundary from the prefill spec.
- [x] 3.2 Extract exact-token prompt materialization, shared-prefix seed construction, and request reconstruction from `cache_grid_runner.py`; verify characterization outputs match the old algorithms for representative tokenizers and grouped/fixed batches.
- [x] 3.3 Implement the prefill runner with cache seeding, measurement loops, response-shape checks, exact observed-reuse validation, timing validation, and per-run metrics; verify fake-server tests classify each invalid condition and exclude it from valid aggregation.
- [x] 3.4 Implement atomic local prefill progress and checkpoint updates while continuing independent cases after request-level failures; verify interruption tests preserve committed cases and produce partial/interrupted state as specified.
- [x] 3.5 Implement local prefill resume using config, workload, model, and runner fingerprints; verify `ok` cases are preserved, all other planned cases rerun, mismatches cause no mutation, and concurrent resume is locked out.
- [x] 3.6 Implement isolated prefill profile sessions for selected original cases with exact request reconstruction and reuse validation; verify successful and failed profiles leave formal results, checkpoints, and summaries byte-for-byte unchanged.
- [x] 3.7 Adapt existing formula, chart, and observation logic to generate complete or diagnostic offline prefill reports from the new result schema; verify report golden tests exclude invalid points and rerunning report does not launch a model or alter case files.
- [x] 3.8 Add local prefill example configs and action documentation for run/resume/profile/report; verify every documented dry-run command succeeds against the new parser.

## 4. AOne prefill execution

- [x] 4.1 Implement generated Bazel packages containing normalized config, resolved workload, shard plan, and BUILD targets with declared runfiles; verify an action launched from a clean runfiles root never reads the submit-host source-config path.
- [x] 4.2 Move the existing estimated-cost and LPT-style sharding algorithm behind the new plan contract; verify shard tests prove complete coverage, exclusive ownership, deterministic output, and cost-aware balancing.
- [x] 4.3 Implement one engine session per remote prefill shard and return run/shard identity, fingerprints, ownership, completion claims, and atomic case results; verify a multi-case shard starts and stops the fake engine exactly once.
- [x] 4.4 Implement strict AOne collection for missing shards/results, duplicate, unknown or out-of-scope cases, and fingerprint mismatches; verify each inconsistency prevents completed status and returns exit code 5 or 6 as classified.
- [x] 4.5 Implement AOne orchestration for build, submit, collect, and offline report with manifest stage updates; verify fake AOne integration tests cover single shard, multiple shards, remote failure, partial diagnostics, and report failure.
- [x] 4.6 Add AOne prefill example config and CI documentation describing runtime checkpoint paths, generated runfiles, sharding, collection, and unsupported actions; verify the documented dry-run emits the expected targets and ownership summary.

## 5. Local decode performance

- [x] 5.1 Extract the fixed decode case runner from `run_single` and the Cartesian loop from `CacheGridRunner.run_cartesian_grid`; verify input-length by batch-size tests retain one scalar output length and reject output arrays/ranges.
- [x] 5.2 Migrate distribution workload resolution and execution to the decode adapter; verify malformed/empty distributions fail during planning and results retain actual sequence lengths and request outcomes.
- [x] 5.3 Migrate decode batch scheduling for DP greater than one; verify deterministic worker allocation covers every request exactly once and reports global batch separately from worker-local allocation, including uneven batches.
- [x] 5.4 Migrate target-TPOT batch search with bounded capacity handling and full search observations; verify tests cover a found boundary, capacity failures, all candidates failing, and no candidate meeting the target.
- [x] 5.5 Implement decode measurement validation, result aggregation, summary, and run-generated basic report; verify wait/prefill/decode/TPOT/total latency, throughput, successful/failed request counts, distribution details, and search trajectory survive schema round-trip.
- [x] 5.6 Add local decode grid, distribution, and TPOT example configs and documentation; verify documented dry-runs succeed and decode resume/profile/report plus AOne decode fail with exit code 2 and no side effects.

## 6. Regression, golden, and publishers

- [x] 6.1 Implement stable baseline case keys and metric/value/direction/threshold comparison against the new result schema; verify higher/lower-is-better, threshold boundary, missing metric, partial result, and mismatch diagnostics.
- [x] 6.2 Implement `baseline_candidate.json` generation without baseline overwrite and add explicit golden generation/validation targets; verify normal runs leave accepted baseline/golden files unchanged and golden targets are deterministic.
- [x] 6.3 Move archive selection and OSS upload behavior from `test_entry.py` into the publisher contract; verify fake uploads receive only configured artifacts and credentials never appear in config, manifest, result, report, or captured logs.
- [x] 6.4 Move ODPS summary publication behind an optional publisher using only finalized aggregate data; verify fake clients cover success, sanitized credential failure, and payload fields for both modes.
- [x] 6.5 Implement publisher `required` policy and manifest/summary outcomes; verify required failure returns exit code 6, optional failure preserves measurement success, and neither path rewrites case outcomes.
- [x] 6.6 Add regression/golden/publishing configuration reference documentation; verify example configs pass strict schema validation and contain no credential values.

## 7. Atomic entry-point and build migration

- [x] 7.1 Add the final `tools/perf_test` entry and new Bazel libraries/tests/macros wired only to the new orchestrator; verify help, local dry-run, and AOne prefill dry-run work through both the repository script and Bazel runfiles.
- [x] 7.2 Migrate all CI definitions and internal callers to the new config, result, and stable runner APIs, including the multi-node `run_single` importer; verify repository targets no longer import old entry modules.
- [x] 7.3 Replace user and CI examples with the new prefill/decode config files and update running/reference docs for the capability matrix, run directory, exit codes, resume/profile/report, AOne delivery, regression, and publishers; verify every documented command is covered by a parser or dry-run test.
- [x] 7.4 Delete `tools/cache_perf`, `batch_decode_test.py`, `test_entry.py`, `perf_config.py`, old cache CLI/orchestrator/CI planner, old decode Cartesian method, old macros, old-only tests, and obsolete docs in the same switch; verify the build and test graph contains no missing sources.
- [x] 7.5 Remove `--partial`, `--cache_grid_json`, implicit engine-argument forwarding, old config/result/resume/baseline compatibility, and cache-file-based behavior selection; verify negative tests prove the removed flags and old config shapes are rejected rather than silently accepted.
- [x] 7.6 Run a repository-wide reference scan for deleted files, symbols, commands, flags, macros, result names, and old documentation examples; verify only intentional historical/OpenSpec references remain.

## 8. Integration and release gates

- [ ] 8.1 Run all focused unit and integration targets for config, planning, state, engine session, prefill, AOne, decode, reports, regression, golden, and publishers; verify the complete target set passes with no skipped required coverage.
- [ ] 8.2 Run local GPU prefill smoke for run/resume/profile/report and validate the common envelope, exact reuse, profile isolation, and report artifacts; record the command and result directory in the change evidence.
- [ ] 8.3 Run single-shard and multi-shard AOne prefill smoke and validate runfiles independence, ownership/fingerprints, strict collection, and offline report; record remote job and collected result evidence.
- [ ] 8.4 Run local GPU decode fixed-grid smoke and, after the core path is stable, distribution and TPOT smoke; validate scalar output length, global DP batch accounting, metric envelope, and search trajectory.
- [ ] 8.5 Run baseline/golden CI gates plus fake required/optional OSS and ODPS failure workflows; verify exit codes, state, and secret-redaction expectations match the specs.
- [ ] 8.6 Run strict OpenSpec validation, formatting/linting, the repository-wide old-reference scan, and the complete affected Bazel test suite; verify all release gates pass before marking the implementation complete.

## Workflow follow-up

- Stop applying `add-cache-perf-local-aone-runners`; after this change is implemented and reviewed, archive or close the superseded change according to project policy.
- Sync the four new capability specs to the main spec set and archive `unify-perf-test` only after implementation and release-gate evidence are complete.

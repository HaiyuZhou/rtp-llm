# Tasks

## 1. Decode Config And Planning

- [ ] 1.1 Add frozen exact-batch, target-TPOT, and inline-profiling config types and strict parsing; verify config tests accept only the new nested strategy fields and reject fixed scalar output, mixed strategy fields, and invalid profile settings.
- [ ] 1.2 Add the typed decode case and measurement models plus shape-complete identity serialization; verify round-trip and fingerprint tests distinguish output lengths and batch policies while keeping equivalent normalized configs stable.
- [ ] 1.3 Replace fixed planning with `input_lengths x output_lengths x batch.sizes` exact expansion and `input_lengths x output_lengths` TPOT expansion; verify case-count, order, identifier, maximum-sequence, concurrency, and token-bound tests.
- [ ] 1.4 Refactor distribution loading and planning for the shared batch policy; verify exact sequence lists/maps work, TPOT sequence prefixes work, and searchable per-batch maps or insufficient sequences fail before execution.
- [ ] 1.5 Update decode example configs and decode documentation to the new grid/distribution/TPOT schema; verify documented commands parse and dry-run with the expected cases and old schema examples are absent.

## 2. Unified Decode Execution Core

- [ ] 2.1 Implement the common batch executor around candidate slicing, DP allocation, warmup, measurement, and structured batch output; verify fake-runtime tests cover uneven `10 -> 4/3/3`, idle workers, worker limits, and measurement counts.
- [ ] 2.2 Replace the HTTP batch runner with a narrow injected transport for scheduler configuration, prompt construction, concurrent rounds, and response parsing; verify transport tests cover retries, timeouts, request failures, and forced planned `min_new_tokens`/`max_new_tokens`.
- [ ] 2.3 Implement the exact-batch strategy and registry boundary; verify one planned exact case performs exactly one formal candidate measurement and preserves its configured global batch.
- [ ] 2.4 Refactor target-TPOT into an execution-independent strategy; verify candidate generation, observation caching, capacity and measurement failures, no-satisfying outcome, and largest-satisfying selection without a forced maximum-batch result.
- [ ] 2.5 Replace fixed and distribution case runners with one case executor and result builder; verify both workload kinds share topology, failure, metric, search, and distribution handling while top-level TPOT metrics come from the selected batch.
- [ ] 2.6 Update decode metrics tests for planned/actual shape validation, per-request outcomes, worker allocations, selected-batch throughput, and `target_tpot_not_met`; verify invalid shape/timing remains excluded from valid metrics.

## 3. Inline Profiling And Decode Reports

- [ ] 3.1 Implement inline profiling after formal measurement with bounded trace flush and per-case manifests; verify fixed cases profile their exact batch, TPOT cases profile only the best batch, disabled profiling sends no request, and failures are recorded without changing valid performance status.
- [ ] 3.2 Extend decode result/finalization schemas with batch policy, search, profiling, and `profiling_failed_cases`; verify case files, common envelope, summary, and profile references validate and are written atomically.
- [ ] 3.3 Implement report generation from committed cases with flat observations, per-batch input-by-output matrices, TPOT matrices, and distribution tables; verify complete grids are formal and invalid, failed, or missing required cases produce diagnostic reports.
- [ ] 3.4 Document decode result, report, and profile-manifest layouts with fixed and TPOT examples; verify documentation tests reference only the standard run-directory artifacts.

## 4. Unified Local Entry Integration

- [ ] 4.1 Wire entry planning, `DecodeAdapter`, and the default local executor to the typed planner and unified case executor; verify local executor tests use one model session and no longer branch to fixed/distribution runner classes.
- [ ] 4.2 Preserve the decode capability boundary at local `run` only; verify capability and CLI tests continue rejecting decode resume, standalone profile/report, and AOne before filesystem, build, engine, or submission side effects.
- [ ] 4.3 Update Bazel BUILD targets and Python imports for the new decode modules and tests; verify focused decode test targets build and run without undeclared module dependencies.

## 5. Multi-Node Migration

- [ ] 5.1 Change the multi-node launcher to distribute and parse one normalized perf config, validate runtime topology on every rank, execute cases only on the master, and retain external lifecycle ownership; verify fake multi-rank tests cover master, worker, mismatch, completion, and failure paths.
- [ ] 5.2 Change multi-node collection scripts to transfer the standard run directory instead of `*Result.json`; verify a fixture run preserves results, reports, profiles, and diagnostics through collection.
- [ ] 5.3 Migrate decode result merging to `results/result.json`, including output length and batch strategy columns; verify merger fixtures cover exact grids, TPOT-selected batches, partial runs, and multiple DP configurations.
- [ ] 5.4 Migrate trace analysis to profile manifests and shape-complete trace identity rather than filename-only inference; verify traces with the same input/batch and different output lengths remain distinct.
- [ ] 5.5 Update multi-node benchmark configuration and documentation for normalized config delivery and standard artifacts; verify repository searches find no log-table or `Decode_Result.json` dependency in active multi-node paths.

## 6. Legacy Decode Removal

- [ ] 6.1 Delete `run_single`, `run_cartesian_grid`, `decode/grid_runner.py`, `decode/fixed.py`, `decode/http_batch.py`, `batch_perf_impl.py`, and now-unused legacy metric data structures after all consumers migrate; verify repository import searches and Bazel query find no references or compatibility aliases.
- [ ] 6.2 Remove obsolete grid API and legacy runner tests and replace them with assertions for the unified core and removed APIs; verify the focused perf-test suite has no test collection dependency on deleted modules.
- [ ] 6.3 Remove old fixed scalar-output configs and legacy result names from examples, docs, scripts, and BUILD metadata; verify repository searches find them only in historical archived specifications where applicable.

## 7. Integration Verification

- [ ] 7.1 Run the complete focused Python/Bazel perf-test suite and record exact commands and outcomes; verify all decode, shared orchestration, result, and unchanged prefill tests pass, documenting any demonstrably unrelated repository dependency blocker.
- [ ] 7.2 Run exact-grid and target-TPOT dry runs and inspect case counts, identities, policies, and resource bounds; verify dry-run creates no formal run directory or model process.
- [ ] 7.3 Run a real-GPU exact input-by-output grid with `/mnt/hf3fs/3fs/models/Qwen2.5-0.5B-Instruct/`; verify every requested input/output/batch cell completes with matching shapes and appears in the matrix report.
- [ ] 7.4 Run a small real-GPU target-TPOT search with the same model; verify observations are preserved and formal metrics/report cells use `best_batch_size` rather than `max_size`.
- [ ] 7.5 Run one real-GPU inline profile case; verify formal metrics remain separate, the manifest references the generated trace, and profile failure handling can be exercised without corrupting case results.
- [ ] 7.6 Run strict OpenSpec validation and final repository status/diff review; verify planning contracts, implementation, tests, docs, and BUILD files are coherent and no unrelated worktree changes are included.

## Workflow follow-up

- Sync the completed delta specs into the main specs after implementation review.
- Archive `refactor-decode-perf-test` only after sync and final verification succeed.

# Tasks

## 1. Decode Config And Planning

- [x] 1.1 Add frozen exact-batch, target-TPOT, and inline-profiling config types and strict parsing; verify config tests accept only the new nested strategy fields and reject fixed scalar output, mixed strategy fields, and invalid profile settings.
- [x] 1.2 Add the typed decode case and measurement models plus shape-complete identity serialization; verify round-trip and fingerprint tests distinguish output lengths and batch policies while keeping equivalent normalized configs stable.
- [x] 1.3 Replace fixed planning with `input_lengths x output_lengths x batch.sizes` exact expansion and `input_lengths x output_lengths` TPOT expansion; verify case-count, order, identifier, maximum-sequence, concurrency, and token-bound tests.
- [x] 1.4 Refactor distribution loading and planning for the shared batch policy; verify exact sequence lists/maps work, TPOT sequence prefixes work, and searchable per-batch maps or insufficient sequences fail before execution.
- [x] 1.5 Update decode example configs and decode documentation to the new grid/distribution/TPOT schema; verify documented commands parse and dry-run with the expected cases and old schema examples are absent.

## 2. Unified Decode Execution Core

- [x] 2.1 Implement the common batch executor around candidate slicing, DP allocation, warmup, measurement, and structured batch output; verify fake-runtime tests cover uneven `10 -> 4/3/3`, idle workers, worker limits, and measurement counts.
- [x] 2.2 Replace the HTTP batch runner with a narrow injected transport for scheduler configuration, prompt construction, concurrent rounds, and response parsing; verify transport tests cover retries, timeouts, request failures, and forced planned `min_new_tokens`/`max_new_tokens`.
- [x] 2.3 Implement the exact-batch strategy and registry boundary; verify one planned exact case performs exactly one formal candidate measurement and preserves its configured global batch.
- [x] 2.4 Refactor target-TPOT into an execution-independent strategy; verify candidate generation, observation caching, capacity and measurement failures, no-satisfying outcome, and largest-satisfying selection without a forced maximum-batch result.
- [x] 2.5 Replace fixed and distribution case runners with one case executor and result builder; verify both workload kinds share topology, failure, metric, search, and distribution handling while top-level TPOT metrics come from the selected batch.
- [x] 2.6 Update decode metrics tests for planned/actual shape validation, per-request outcomes, worker allocations, selected-batch throughput, and `target_tpot_not_met`; verify invalid shape/timing remains excluded from valid metrics.

## 3. Inline Profiling And Decode Reports

- [x] 3.1 Implement inline profiling after formal measurement with bounded trace flush and per-case manifests; verify fixed cases profile their exact batch, TPOT cases profile only the best batch, disabled profiling sends no request, and failures are recorded without changing valid performance status.
- [x] 3.2 Extend decode result/finalization schemas with batch policy, search, profiling, and `profiling_failed_cases`; verify case files, common envelope, summary, and profile references validate and are written atomically.
- [x] 3.3 Implement report generation from committed cases with flat observations, per-batch input-by-output matrices, TPOT matrices, and distribution tables; verify complete grids are formal and invalid, failed, or missing required cases produce diagnostic reports.
- [x] 3.4 Document decode result, report, and profile-manifest layouts with fixed and TPOT examples; verify documentation tests reference only the standard run-directory artifacts.

## 4. Unified Local Entry Integration

- [x] 4.1 Wire entry planning, `DecodeAdapter`, and the default local executor to the typed planner and unified case executor; verify local executor tests use one model session and no longer branch to fixed/distribution runner classes.
- [x] 4.2 Preserve the decode capability boundary at local `run` only; verify capability and CLI tests continue rejecting decode resume, standalone profile/report, and AOne before filesystem, build, engine, or submission side effects.
- [x] 4.3 Update Bazel BUILD targets and Python imports for the new decode modules and tests; verify focused decode test targets build and run without undeclared module dependencies. The earlier "missing `@pip_cpu_torch//flash_attn`" analysis failure was misdiagnosed as an environment defect: the recorded command omitted `--config=cuda13`, so `arch_config/arch_select.bzl` resolved `flash_attn` through the `//conditions:default` branch to the CPU pip repository, where flash_attn is intentionally absent from `deps/requirements_lock_torch_cpu.txt`. With `--config=cuda13` the dependency resolves to `@pip_gpu_cuda13_torch//flash_attn` and the exact Task 8 26-target Bazel command passes (see the 2026-11-01 evidence below).

## 5. Multi-Node Migration

- [x] 5.1 Change the multi-node launcher to distribute and parse one normalized perf config, validate runtime topology on every rank, execute cases only on the master, and retain external lifecycle ownership; verify fake multi-rank tests cover master, worker, mismatch, completion, and failure paths.
- [x] 5.2 Change multi-node collection scripts to transfer the standard run directory instead of `*Result.json`; verify a fixture run preserves results, reports, profiles, and diagnostics through collection.
- [x] 5.3 Migrate decode result merging to `results/result.json`, including output length and batch strategy columns; verify merger fixtures cover exact grids, TPOT-selected batches, partial runs, and multiple DP configurations.
- [x] 5.4 Migrate trace analysis to profile manifests and shape-complete trace identity rather than filename-only inference; verify traces with the same input/batch and different output lengths remain distinct.
- [x] 5.5 Update multi-node benchmark configuration and documentation for normalized config delivery and standard artifacts; verify repository searches find no log-table or `Decode_Result.json` dependency in active multi-node paths.

## 6. Legacy Decode Removal

- [x] 6.1 Delete `run_single`, `run_cartesian_grid`, `decode/grid_runner.py`, `decode/fixed.py`, `decode/http_batch.py`, `batch_perf_impl.py`, and now-unused legacy metric data structures after all consumers migrate; verify repository import searches and Bazel query find no references or compatibility aliases.
- [x] 6.2 Remove obsolete grid API and legacy runner tests and replace them with assertions for the unified core and removed APIs; verify the focused perf-test suite has no test collection dependency on deleted modules.
- [x] 6.3 Remove old fixed scalar-output configs and legacy result names from examples, docs, scripts, and BUILD metadata; verify repository searches find them only in historical archived specifications where applicable.

## 7. Integration Verification

- [x] 7.1 Run the complete focused Python/Bazel perf-test suite and record exact commands and outcomes; verify all decode, shared orchestration, result, and unchanged prefill tests pass, documenting any demonstrably unrelated repository dependency blocker.
- [x] 7.2 Run exact-grid and target-TPOT dry runs and inspect case counts, identities, policies, and resource bounds; verify dry-run creates no formal run directory or model process.
- [x] 7.3 Run a real-GPU exact input-by-output grid with `/mnt/hf3fs/3fs/models/Qwen2.5-0.5B-Instruct/`; verify every requested input/output/batch cell completes with matching shapes and appears in the matrix report.
- [x] 7.4 Run a small real-GPU target-TPOT search with the same model; verify observations are preserved and formal metrics/report cells use `best_batch_size` rather than `max_size`.
- [x] 7.5 Run one real-GPU inline profile case; verify formal metrics remain separate, the manifest references the generated trace, and profile failure handling can be exercised without corrupting case results.
- [x] 7.6 Run strict OpenSpec validation and final repository status/diff review; verify planning contracts, implementation, tests, docs, and BUILD files are coherent and no unrelated worktree changes are included.

## Workflow follow-up

- Sync the completed delta specs into the main specs after implementation review.
- Archive `refactor-decode-perf-test` only after sync and final verification succeed.

## Verification evidence (2026-10-30)

- Groups 1–6: the focused direct Python suite plus migrated local/scheduler/distribution/docs consumers passes 221 tests (exit 0). Legacy API/module searches and the scoped Bazel target query find no deleted references. Multi-node collection/config/plan consistency is covered by the strict-TDD fix `2f8505cfe0`; real profile discovery/retention is covered by `5f10531c67`.
- 7.1: executed the exact Task 8 26-target Bazel command; exit 1 before tests because `@pip_cpu_torch//flash_attn` has no BUILD package. The equivalent 26 direct Python modules plus five migrated/documentation consumers pass (221 tests, exit 0). Exact argv, exits and full logs: `/tmp/decode-task9-evidence/focused-bazel.{json,log}` and `focused-python.{json,log}`. The lightweight analyzer Bazel target also passes (cached).
- 7.2: both documented example dry-runs exit 0; exact case count 18 and TPOT case count 6. Unique identities, policies, sequence/concurrency bounds, absent preview run directories and unchanged empty model-process snapshots verified in `/tmp/decode-task9-evidence/dry-evidence.json`.
- 7.3: exact brief config with the local Qwen 0.5B checkpoint exits 0 on NVIDIA L20D GPU 0. `/tmp/decode-perf-grid-results/decode-20261030T032129815921Z`: 8/8 successful cases, every actual request shape matches the plan, formal report with batch 1/2 matrices and four cells each.
- 7.4: exact brief TPOT config exits 0. `/tmp/decode-perf-tpot-results/decode-20261030T033230016161Z`: search `found`, observations for batches 2/3/4, best batch 4, matching formal top-level metrics and report cell, 1/1 successful case. This smoke target selects the maximum; smaller-than-maximum selection is covered by direct strategy/executor/profile tests.
- 7.5: the same TPOT run profiles only selected batch 4. `profiles/decode-83b870403543d07e81bbfd90/profile_manifest.json` records input `[32,32,32,32]`, output 8, runs 1, steps 3, all ranks, completed status and a complete JSON trace retained under `traces/`. Summary has zero profile failures. Earlier real run `/tmp/decode-perf-tpot-results/decode-20261030T032257925517Z` reproduces nonfatal profile timeout handling: performance remains successful/formal with `profiling_failed_cases=1`; the discovered logger-routing defect was then fixed with strict TDD and rerun successfully.
- 7.6: `openspec validate refactor-decode-perf-test --strict` and whitespace/scope checks pass. Full run/request/report/profile assertions are recorded by `/tmp/decode-task9-inspect.py` and `/tmp/decode-task9-evidence/result-inspection.json`. Initial source/runtime failures and the existing dependency environment required a `/tmp` bootstrap using already-generated Bazel modules and complete cached GPU dependencies; exact configs and CLI commands were retained. Every model/server process was allowed to terminate; no generated result or unrelated user files are committed.

## Verification evidence (2026-11-01)

- 4.3: the exact Task 8 26-target Bazel command now passes on the local x86_64 host (CUDA 13.2, NVIDIA L20D) once `--config=cuda13` is supplied: `Executed 26 out of 26 tests: 26 tests pass`, exit 0, elapsed 736.274s, critical path 719.65s. Command, exit code, and the full log are retained at `/tmp/decode-task9-evidence/focused-bazel-v3.json` and `/tmp/decode-task9-evidence/focused-bazel-v3.log`. The original failure was not a `@pip_cpu_torch//flash_attn` environment defect: without a platform config, `arch_config/arch_select.bzl` resolves `flash_attn` to the CPU pip repository, whose lock (`deps/requirements_lock_torch_cpu.txt`) intentionally carries no flash_attn; `--config=cuda13` resolves it to `@pip_gpu_cuda13_torch//flash_attn` (present at `deps/requirements_lock_torch_gpu_cuda13.txt:652`).
- 4.3 network note: the local proxy environment variables (`http_proxy`/`https_proxy` at `127.0.0.1:18091`) point at a nonfunctional proxy, and direct git-smart-HTTP access to `github.com` is unstable (`Empty reply from server`), which blocked the `@xgrammar` git fetch in fresh output bases. This was worked around without any source change by reusing the marker-complete output user root `/home/zhouhaiyu.zhy/.cache/bazel_cuda13_cache_grid_main_local_cache` (no github fetch required); internal pip/wheel mirrors remained directly reachable.

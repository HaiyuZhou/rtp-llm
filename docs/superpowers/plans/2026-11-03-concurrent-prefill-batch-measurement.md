# Concurrent Prefill Batch Measurement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make prefill batch sizes correspond to concurrently submitted requests under the prefill scheduler, with throughput derived from actual batch duration.

**Architecture:** Keep one shared `PrefillRunner` for local, AOne shard, and gang paths. Extend its HTTP client with acknowledged scheduler updates; seed caches under batch 1, then release each measured batch concurrently. Mark new case identities with `concurrent_batch_v1` and validate round-level evidence before reporting.

**Tech Stack:** Python 3, `unittest`, `concurrent.futures.ThreadPoolExecutor`, `threading.Event`, HTTP `/update_scheduler_info`, Bazel tests, OpenSpec.

**Spec:** `openspec/changes/fix-concurrent-prefill-batch-measurement/design.md`, `specs/prefill-cache-performance/spec.md`, and `specs/perf-test-results/spec.md` under that change.

## Global Constraints

- Preserve `./tools/perf_test` and the versioned JSON config shape; do not add a public CLI option.
- Use `measurement_contract="concurrent_batch_v1"` in every new prefill case identity; do not change decode case identities or the global result-envelope schema version 2.
- Configure `{batch_size: 1, mode: "prefill"}` before cached seeds, then `{batch_size: B, mode: "prefill"}` before measured rounds. Require a successful acknowledgement before sending that phase's requests.
- Reject `DP>1` with any prefill `batch_size>1` during planning; preserve `DP>1, batch_size=1` and `DP=1` TP/PP gangs.
- Keep server-reported first-token latency separate from client-monotonic round duration. Case throughput is the arithmetic mean of `B * compute_length * 1000 / round_elapsed_ms` over valid rounds.
- Preserve old run directories for offline report, but reject resume/profile of an old prefill plan. Do not silently compare old throughput baselines with new case IDs or auto-rewrite baselines.

## Review Focus

- A scheduler acknowledgement that is HTTP 200 but contains `error` or lacks `status: ok` must fail before seeds/measurements (Task 3 tests).
- An early-failing concurrent member must not leave peers unawaited or produce a successful point (Task 4 tests).
- A zero, non-finite, or inconsistent round duration must be excluded even when the top-level throughput looks plausible (Task 5 tests).
- `DP>1` with batch 4 must fail before a run directory or remote submission, while batch 1 remains valid (Task 1 tests).
- A legacy plan must remain reportable but cannot be resumed/profiled into a new-contract result (Tasks 2 and 6 tests).

---

## File map

- Create `rtp_llm/test/perf_test/prefill/measurement.py`: `PREFILL_MEASUREMENT_CONTRACT` and pure validation of new per-round evidence; used by planner, result parser, and report.
- Modify `prefill/planner.py`, `entry.py`, `multi_node/prefill_runtime.py`: contract-bearing cases and early DP validation in ordinary and externally managed planning.
- Modify `prefill/checkpoint.py`, `prefill/profile.py`: old-plan resume/profile guard; `entry.py` applies the same guard before execution side effects.
- Modify `execution/runtime.py`: scheduler control on `HttpPrefillClient`, shared by local, shard, and gang sessions.
- Modify `prefill/runner.py`: phase transitions, concurrent measured rounds, per-member collection, and measured throughput.
- Modify `results.py`, `prefill/report.py`: validate new evidence, retain legacy read compatibility, and label report contract.
- Modify `rtp_llm/test/perf_test/BUILD`: add the shared measurement module to `perf_test_entry_lib` and the standalone `perf_test_lib`. `perf_test_runtime_lib` receives it through its `perf_test_entry_lib` dependency.
- Extend neighboring `*_test.py` files rather than adding a separate parallel test framework; update `docs/prefill.md`, `docs/multi_node_aone.md`, and `docs/regression_publish.md` beside the relevant implementation tasks.

### Task 1: Freeze the new prefill identity and reject unsafe DP plans

**Files:** Create `rtp_llm/test/perf_test/prefill/measurement.py`; modify `prefill/planner.py:106-149`, `entry.py:57-72`, `multi_node/prefill_runtime.py:20-43`, `BUILD:36-190`; test `prefill_planner_test.py`, `entry_test.py`, `prefill_external_test.py`.

**Interfaces:** Produce `PREFILL_MEASUREMENT_CONTRACT: str = "concurrent_batch_v1"` and `plan_prefill_cases(workload: PrefillWorkloadConfig, engine: EngineConfig, *, dp_size: int = 1) -> Tuple[Dict[str, object], ...]`. Ordinary and external entry points must pass configured `model.topology.dp`, not rely on the default.

- [ ] **Step 1: Write failing tests.** Add `test_concurrent_contract_changes_case_id` asserting every planned identity has `measurement_contract == "concurrent_batch_v1"` and differs from the same old identity's `case_id`; add `test_multi_dp_batch_rejected_before_side_effects` for `dp_size=2, batch_sizes=(4,)`, while `batch_sizes=(1,)` passes. Exercise both `build_request_plan` and `build_external_prefill_plan` with no run directory created.
- [ ] **Step 2: Verify red.** Run `bazel test //rtp_llm/test/perf_test:prefill_planner_test //rtp_llm/test/perf_test:entry_test //rtp_llm/test/perf_test:prefill_external_test --test_output=errors`; expect the new assertions to fail on the missing contract or `dp_size` parameter.
- [ ] **Step 3: Implement.** Add the constant; pass `dp_size` from both entry paths, reject `dp_size>1 and any(batch>1)` with `PlanningError`, and add the contract to each identity before `case_id` is derived. Register the new module in `BUILD` wherever imported.
- [ ] **Step 4: Verify green.** Rerun the three Bazel targets; expect all tests to pass, including existing deterministic-planning tests.
- [ ] **Step 5: Commit this slice.** Stage only the files above and commit `fix(perf): version prefill batch identities and guard DP plans`.

### Task 2: Guard old prefill plans while preserving offline reports

**Files:** Modify `prefill/checkpoint.py:14-33`, `prefill/profile.py:28-47`, `entry.py:102-120`; test `prefill_resume_test.py`, `prefill_profile_test.py`, `prefill_report_test.py`; document in `docs/prefill.md`.

**Interfaces:** Consume `PREFILL_MEASUREMENT_CONTRACT` from Task 1. Produce `require_concurrent_prefill_plan(plan: ExecutionPlan) -> None` in `prefill/checkpoint.py`; call it for prefill `resume`/`profile` in `entry._persisted_plan`, `PrefillResume` validation, and `PrefillProfiler._load_plan`. Update `RUNNER_CONTRACT` to identify the changed runner semantics.

- [ ] **Step 1: Write failing tests.** `test_legacy_plan_resume_and_profile_rejected` constructs a persisted prefill plan without the marker and asserts a diagnostic containing `concurrent_batch_v1` before executor startup; `test_legacy_plan_report_still_reads` asserts report action remains allowed.
- [ ] **Step 2: Verify red.** Run `bazel test //rtp_llm/test/perf_test:prefill_resume_test //rtp_llm/test/perf_test:prefill_profile_test //rtp_llm/test/perf_test:prefill_report_test --test_output=errors`; expect the new migration assertions to fail.
- [ ] **Step 3: Implement and document.** Add the guard and update runner fingerprint; explain that old directories can be reported but must be rerun from config for new measurements. Do not mutate old plans.
- [ ] **Step 4: Verify green.** Rerun those targets; expect all to pass, including new-plan resume/profile tests.
- [ ] **Step 5: Commit this slice.** Stage only the touched files and commit `fix(perf): reject legacy prefill resume and profile`.

### Task 3: Add acknowledged prefill scheduler control

**Files:** Modify `execution/runtime.py:91-160`; test `runtime_adapter_test.py`, `prefill_external_test.py`.

**Interfaces:** Produce `HttpPrefillClient.configure_scheduler(self, batch_size: int, mode: str = "prefill") -> None`; add injectable `sleep: Callable[[float], None] = time.sleep` and `scheduler_attempts: int = 20` constructor arguments. Use the client's full endpoint for `/update_scheduler_info`, not the loopback-only helper. A successful response is HTTP 200 with JSON `{"status": "ok"}` and no `error`.

- [ ] **Step 1: Write failing tests.** `test_scheduler_update_acknowledged_on_explicit_endpoint` asserts POST to `<endpoint>/update_scheduler_info` with `{"batch_size": 4, "mode": "prefill"}`; `test_scheduler_rejects_bad_ack_and_exhausted_retry` covers HTTP error, `{"error": "..."}`, and HTTP 200 without `status: ok`, with injected no-op sleep and two attempts.
- [ ] **Step 2: Verify red.** Run `bazel test //rtp_llm/test/perf_test:runtime_adapter_test //rtp_llm/test/perf_test:prefill_external_test --test_output=errors`; expect `configure_scheduler`-related failures.
- [ ] **Step 3: Implement.** Validate positive integer batch and prefill mode, retry bounded failures, and raise a diagnostic error after the final attempt. Keep request-timeout behavior unchanged. Confirm `ExternalPrefillSession` and `MagaEngineHandle` both expose this same client.
- [ ] **Step 4: Verify green.** Rerun those targets; expect all tests to pass.
- [ ] **Step 5: Commit this slice.** Stage only the touched files and commit `feat(perf): configure prefill scheduler through shared client`.

### Task 4: Execute real prefill batches and time complete rounds

**Files:** Modify `prefill/runner.py:12-139`; test `prefill_runner_test.py`, `prefill_external_test.py`; document behavior in `docs/prefill.md`.

**Interfaces:** Consume `client.configure_scheduler(batch_size: int, mode: str = "prefill")` from Task 3. Add `monotonic: Callable[[], float] = time.monotonic` to `PrefillRunner.__init__` for deterministic time tests; keep `run_case(case: Dict[str, Any]) -> Dict[str, Any]`. Each member measurement keeps `run`, `request_index`, `latency_ms`, `observed_reuse_length` and adds `round_elapsed_ms` plus `measurement_contract`.

- [ ] **Step 1: Write failing tests.** Use a fake client with a four-party barrier to prove `test_batch_four_measure_requests_overlap`; assert scheduler calls `[1, 4]` for a cached case, `[4]` for a cold case, and no measurement after a seed or scheduler failure. Test one member throwing while all submitted peers finish. With injected monotonic times `0.0` and `0.2`, batch 4 and `compute_len=8` must yield `160.0` tokens/s despite server latencies of 10–40 ms; batch 1 uses the same denominator.
- [ ] **Step 2: Verify red.** Run `bazel test //rtp_llm/test/perf_test:prefill_runner_test //rtp_llm/test/perf_test:prefill_external_test --test_output=errors`; expect overlap, scheduler-order, and throughput assertions to fail.
- [ ] **Step 3: Implement.** Configure batch 1 before cached seeds, then target batch before measurement. Prepare prompts outside timing; use a reusable `ThreadPoolExecutor(max_workers=B)` and release event per round; submit all members before awaiting any, record elapsed with `monotonic`, and join every future before classifying failures. Return `request_failed`/`invalid_shape`/`invalid_reuse`/`invalid_timing` with `metrics=None` if any member or elapsed value is invalid. Compute top-level latency as median server first-token time and throughput as mean of round throughputs. Make existing response fixtures keyed by member rather than pop order so concurrent tests are deterministic.
- [ ] **Step 4: Verify green.** Rerun both targets; expect all tests to pass. Document seed-versus-measure scheduler phases and wall-clock throughput versus server latency.
- [ ] **Step 5: Commit this slice.** Stage only the touched files and commit `feat(perf): measure concurrent prefill batches`.

### Task 5: Validate persisted new-contract prefill evidence

**Files:** Extend `prefill/measurement.py`; modify `results.py:197-240,540-600`; test `result_envelope_test.py`.

**Interfaces:** Produce `validate_concurrent_prefill_metrics(identity: Mapping[str, Any], metrics: Mapping[str, Any], runs: int) -> None` in `prefill/measurement.py`. It raises `ValueError` for a new-contract mismatch and is a no-op for legacy identities; `results.parse_case_result` translates it to `ResultSchemaError`. Task 6 reuses the same function in the offline report.

- [ ] **Step 1: Write failing tests.** For batch 4 and two rounds, accept exactly eight distinct `(run, request_index)` records with one positive finite `round_elapsed_ms` per round and the correct marker. Reject duplicate/missing members, inconsistent/zero/NaN elapsed, incorrect member reuse or latency, and top-level latency or throughput not matching the raw records; confirm decode envelope parsing is unchanged.
- [ ] **Step 2: Verify red.** Run `bazel test //rtp_llm/test/perf_test:result_envelope_test --test_output=errors`; expect tampered new-contract cases to be accepted incorrectly until validation is added.
- [ ] **Step 3: Implement.** Validate exact round indices `0..runs-1`, member indices `0..B-1`, marker equality, per-member reuse and positive finite latency, round elapsed consistency, and `math.isclose` for median latency and mean round throughput. Call only for successful new-contract prefill results; leave legacy parsing and decode untouched.
- [ ] **Step 4: Verify green.** Rerun the target; expect all tests to pass.
- [ ] **Step 5: Commit this slice.** Stage only the touched files and commit `fix(perf): validate prefill batch measurement evidence`.

### Task 6: Expose contract in reports and protect baseline migration

**Files:** Modify `prefill/report.py:35-130,145-255`; test `prefill_report_test.py`, `baseline_regression_test.py`, `golden_test.py`; document in `docs/prefill.md`, `docs/multi_node_aone.md`, `docs/regression_publish.md`.

**Interfaces:** Consume `validate_concurrent_prefill_metrics(...)` from Task 5. New report observations expose `measurement_contract`; legacy observations use an explicit `legacy_latency_derived_v1` label. Keep existing report schema fields and offline-only operation.

- [ ] **Step 1: Write failing tests.** `test_report_excludes_tampered_concurrent_round` asserts diagnostic report and excluded case; `test_report_labels_legacy_without_rewriting` asserts legacy output stays readable with a legacy label; `test_old_baseline_key_does_not_match_new_prefill_case` asserts regression failure from the changed stable ID; keep golden validation explicit, with no automatic baseline or golden rewrite.
- [ ] **Step 2: Verify red.** Run `bazel test //rtp_llm/test/perf_test:prefill_report_test //rtp_llm/test/perf_test:baseline_regression_test //rtp_llm/test/perf_test:golden_test --test_output=errors`; expect new report-label/validation assertions to fail.
- [ ] **Step 3: Implement and document.** Validate new observations through the shared helper, expose contract in JSON/HTML, retain legacy offline reads, and explain the DP restriction, changed case IDs, throughput source, and explicit baseline regeneration in the three docs. Update only fixtures that encode the new result contract.
- [ ] **Step 4: Verify green.** Rerun those targets and `//rtp_llm/test/perf_test:prefill_examples_test`; expect all to pass.
- [ ] **Step 5: Commit this slice.** Stage only the touched files and commit `docs(perf): report concurrent prefill contract and migration`.

### Task 7: Verify all prefill execution paths and the OpenSpec change

**Files:** Test-only updates if needed in `aone_worker_test.py`, `aone_gang_worker_test.py`, `default_executor_test.py`, `gang_examples_test.py`; update matching `openspec/changes/fix-concurrent-prefill-batch-measurement/tasks.md` checkboxes as each OpenSpec task is finished during apply.

**Interfaces:** No new production API; verifies Tasks 1–6 as one end-to-end contract.

- [ ] **Step 1: Add integration assertions if missing.** Local, single-machine AOne, and gang prefill must use the shared client/runner; dry-run must reject `DP>1, B>1` without output creation; batch-4 config must reach scheduler-update then concurrent execution on a fake endpoint. Assert no duplicate or lost case results under AOne collection.
- [ ] **Step 2: Verify focused paths.** Run `bazel test //rtp_llm/test/perf_test:default_executor_test //rtp_llm/test/perf_test:aone_worker_test //rtp_llm/test/perf_test:aone_gang_worker_test //rtp_llm/test/perf_test:gang_examples_test --test_output=errors`; expect all to pass.
- [ ] **Step 3: Verify the complete suite.** Run `bazel test //rtp_llm/test/perf_test/... --test_output=errors` and `openspec validate fix-concurrent-prefill-batch-measurement --strict`; record exact pass/fail counts and distinguish any pre-existing failure from a regression.
- [ ] **Step 4: Commit integration tests only if added.** Stage only new test changes and commit `test(perf): cover prefill batch execution across backends`.

## Execution notes

- Keep the existing dirty worktree's unrelated untracked files untouched. Stage explicit paths only; no broad `git add .`.
- During apply, OpenSpec task checkboxes are a separate progress ledger from this plan. Complete each only after its behavior and verification are done.
- Do not archive or sync the OpenSpec change as an implicit part of this plan; request that follow-up separately after review.

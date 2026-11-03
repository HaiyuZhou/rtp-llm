# Unified Prefill Cache Restoration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore cache-aware prefill charting, FlexLB fitting, grouped batches, trace capture, cache probe, DashSC input-ID transport, ordered multi-grid runs, and offline reanalysis in the unified perf-test entry.

**Architecture:** Keep one immutable execution plan and one result directory. Generated and explicit grouped cases share the prefill adapter; each case records a versioned measurement contract. The offline report normalizes valid observations, renders charts, fits a candidate formula, and can be rerun without an engine.

**Tech Stack:** Python 3, unittest, Plotly, current Bazel perf-test package, historical `81b06495f4` cache-grid algorithms as reference.

**Spec:** `docs/superpowers/specs/2026-11-03-unified-prefill-cache-restoration-design.md`

## Global Constraints

- Preserve flat `workload.kind=cache_grid` inputs and `concurrent_batch_v1` results.
- Grouped cases use `concurrent_grouped_batch_v1`; never reinterpret an old case with the new contract.
- `request_groups` entries contain `count`, `input_len`, `cache_len`; total count is batch size. Identity summary input/cache lengths are maxima; metric/report aggregate lengths are sums; validation uses each member's geometry.
- Grid order is declared order, not lexical order. Every grid name is unique and appears in case identity and report provenance.
- Ordered multi-grid execution is local-only; reject AOne multi-grid planning because AOne shards cannot enforce a global grid barrier.
- Server first-token latency and client round wall time remain separate. Fit target is median over rounds of the maximum member server latency. Throughput uses measured round wall time.
- Formal results and checkpoints never change during profile, report, or `--skip-test`.
- No formula receives `production_acceptance=true` from partial or historical latency-derived observations. The two validated concurrent contracts may fit together using the same target definition.
- Local and AOne prefill retain the same shared runner. Trace capture is local only.

## Review Focus

- A grouped case with one member over `max_seq_len` fails planning even if total budget is high: Task 1 test.
- A grouped case whose padded token sum exceeds workspace fails planning: Task 1 test.
- One member reporting the wrong reuse length fails the whole case: Task 2 test.
- A failed block probe prevents any formal measured request: Task 2 test.
- Offline regeneration leaves committed case bytes and checkpoints intact: Task 7 test.

---

### Task 1: Explicit grouped batches and ordered grid planning

**Files:** Modify `config_schema.py`, `prefill/planner.py`, `aone/sharding.py`, `prefill/plan_validation.py`; test `unified_config_test.py`, `prefill_external_test.py`, `aone_sharding_test.py`.

**Interfaces:** Extend `PrefillWorkloadConfig` with validated ordered `grids` and explicit `request_batches`. Add shared `workload.transport`, `workload.profiling`, `workload.probe_block_size`, and `workload.workspace_tokens` (required for explicit groups). `plan_prefill_cases(...)` returns old homogeneous identities unchanged and grouped identities with `grid_name`, `request_groups`, maximum summary lengths, and `measurement_contract=concurrent_grouped_batch_v1`. AOne shard weight uses the sum of member attention costs.

- [ ] Write tests for named grid order, duplicate grid names, grouped geometry and static padded budget, DP>1 batch rejection, AOne multi-grid rejection, and AOne grouped weight.
- [ ] Run those tests and confirm expected feature-missing failures.
- [ ] Implement strict config parsing and planner admission with one output token reserve per member and physical block alignment.
- [ ] Run the focused tests and existing planner/config/sharding tests; commit.

### Task 2: Grouped concurrent runner and cache-block probe

**Files:** Modify `prefill/runner.py`, `prefill/measurement.py`, `modes/prefill.py`, `execution/defaults.py`; create `prefill/probe.py`; test `prefill_runner_test.py`, `prefill_resume_test.py`, `default_executor_test.py`.

**Interfaces:** `PrefillRunner.run_case(case)` handles homogeneous and grouped identities, sends independent sequential seeds at scheduler batch 1, then releases member requests concurrently at target B. `probe_cache_block(client, tokenizer, block_size, commit_tail_tokens)` checks a one-block hit before selected formal cases, outside the adapter's per-case catch. `validate_concurrent_prefill_metrics` accepts the new grouped contract and preserves old behavior. Extend `results.py` metric parser for grouped sums without changing old case round trips.

- [ ] Write failing tests for differing member lengths, exact per-member reuse, round wall time, failed probe before first case, and resume handling of grouped identities.
- [ ] Run and inspect failures.
- [ ] Implement grouped prompt construction, seed/measure flow, contract validation, and a persisted probe artifact outside case results.
- [ ] Run focused runner/adapter/resume tests; commit.

### Task 3: DashSC gRPC input-ID transport

**Files:** Modify `config_schema.py`, `execution/runtime.py`, `execution/defaults.py`; create `prefill/dashsc_transport.py`; test `runtime_adapter_test.py`, `default_executor_test.py`.

**Interfaces:** `workload.transport` selects `http_text` (default) or `dashsc_input_ids`. Both client implementations expose `configure_scheduler(batch_size, mode)` and `request(...)` with the existing response dictionary. DashSC sends precomputed INT32 IDs, reads engine latency microseconds as milliseconds, and returns input/output/reuse lengths.

- [ ] Write failing tests for strict transport selection, exact transmitted IDs, response field parity, and missing gRPC endpoint failure.
- [ ] Run and inspect failures.
- [ ] Port the historical DashSC transport behavior into the unified runtime with lazy imports and explicit failure on unavailable dependencies.
- [ ] Run transport and executor tests; commit.

### Task 4: Real isolated prefill trace capture

**Files:** Modify `prefill/profile.py`, `execution/defaults.py`, `execution/runtime.py`; create `prefill/trace.py`; test `prefill_profile_test.py`, `default_executor_test.py`.

**Interfaces:** Local `profile` accepts Kineto or nsys backend in the persisted config, selected case IDs, and trace timeout. Each selected case gets a trace session, manifest and artifact paths under `profiles/<session>/`; formal results are untouched. Trace files must be observed before success is recorded. Kineto manifests label the one-forward-step scope; nsys captures the complete selected round. Interrupted sessions persist a terminal manifest state.

- [ ] Write failing tests for Kineto/NSys command and artifact handling, honest capture scope, timeout/failure/interrupted status, and formal result isolation.
- [ ] Run and inspect failures.
- [ ] Implement trace session management using historical profiler hooks, with process lifecycle cleanup and no trace timing in formal metrics.
- [ ] Run focused tests; commit.

### Task 5: Offline merged observations and charts

**Files:** Modify `prefill/report.py`; create `prefill/chart.py`; test `prefill_report_test.py`, new `prefill_chart_test.py`.

**Interfaces:** `generate_prefill_report(root)` validates homogeneous and grouped evidence and writes `reports/prefill_observations.json`, `reports/prefill_report.json`, `reports/prefill_report.html`, `reports/latency.interactive.html`, `reports/tpm.interactive.html`, plus static fallback. Merged rows carry `grid_name` and per-member lengths.

- [ ] Write failing tests for mixed-grid provenance, invalid member exclusion, batch latency as median round maximum, TPM from round wall time, and standalone HTML output.
- [ ] Run and inspect failures.
- [ ] Implement normalization and Plotly chart rendering with safe deterministic SVG fallback.
- [ ] Run report/chart tests; commit.

### Task 6: Restricted symbolic FlexLB candidate formula

**Files:** Create `prefill/formula.py`, `prefill/restricted_symbolic.py`; modify `prefill/report.py`; test new `prefill_formula_test.py`, `prefill_report_test.py`.

**Interfaces:** `fit_prefill_formula(observations, output_dir, *, complete)` writes `model.json`, `fit_report.json`, `predictions.csv`, `fit_gap.svg`, and `PREFILL_TIME_FORMULA.txt` when data is sufficient. Use historical restricted grammar and deterministic train/validation/test separation; include grouped request distributions. `production_acceptance` is separate from candidate generation.

- [ ] Write failing tests for generated syntax, held-out split, per-batch error gate, grouped and homogeneous combined features, insufficient rows, partial/legacy rejection, and clearing stale accepted artifacts.
- [ ] Run and inspect failures.
- [ ] Port and adapt the historical fit algorithm to validated unified observations, retaining FlexLB parser grammar and report artifacts.
- [ ] Run formula/report tests; commit.

### Task 7: Offline `--skip-test`, one-command integration, examples and docs

**Files:** Modify `cli.py`, `entry.py`, `capabilities.py`, `docs/prefill.md`, `docs/reference.md`, examples, `BUILD`; test `entry_test.py`, `entry_execution_test.py`, `unified_docs_test.py`, new integration test.

**Interfaces:** `./tools/perf_test --mode=prefill --skip-test --run-dir=...` is an alias for offline report regeneration. Existing `--action=report` remains supported. Successful local/AOne run auto-generates charts and fit report; no second command is required.

- [ ] Write failing tests for skip-test no-engine behavior, unchanged formal results, rejected decode skip-test, one-command artifacts, and documented grouped/multi-grid config.
- [ ] Run and inspect failures.
- [ ] Wire the CLI alias, Bazel dependencies, examples and docs.
- [ ] Run relevant perf-test unittest suite, compile check, and final diff review; commit.

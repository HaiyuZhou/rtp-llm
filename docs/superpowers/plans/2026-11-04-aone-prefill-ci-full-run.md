# AOne Prefill CI Full Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make one AOne prefill `run` exercise grouped concurrent batches, the cache-block probe, DashSC `input_ids`, collection, charts, and FlexLB fitting, then verify the pipeline and artifacts.

**Architecture:** Add a single-grid AOne gang fixture with at least 30 distinct grouped request geometries, including heterogeneous batch size 4. Route `mode=prefill` in the existing internal `.aoneci/multi-node-perf.yaml` pipeline to that fixture while preserving the decode defaults. Add a small, testable artifact gate that verifies the collected run's prefill contract; the pipeline uploads the run directory even on failure.

**Tech Stack:** Python 3, JSON, Bazel, AOne CI YAML, unified `./tools/perf_test`.

**Spec:** `docs/superpowers/specs/2026-11-03-unified-prefill-cache-restoration-design.md`; existing AOne contract in `rtp_llm/test/perf_test/docs/multi_node_aone.md`.

## Global Constraints

- AOne supports one grid per run; do not add multi-grid ordering or `profile` to this pipeline.
- Grouped batches require `model.topology.dp == 1`, declared token workspace, and bounded member lengths.
- The formula report may reject production acceptance; the CI gate requires the report and a real candidate from at least 30 valid geometries, never production acceptance.
- Do not change the existing decode default behavior when `mode=decode`.
- Internal CI YAML is in the parent repository; edit its `multi-node-perf.yaml` pipeline and only the runtime dependency lock entries needed to supply embedded Plotly charts.

## Review Focus

- A `mode=prefill` invocation with default parameters selects the new prefill fixture and its model values; pin this in the delivery test.
- Explicit config/model overrides continue to win; pin this in the delivery test.
- Missing probe, grouped contract, interactive charts, or formula candidate causes CI artifact verification to fail; pin each in verifier tests.
- An incomplete/diagnostic report cannot pass the artifact gate; pin this in verifier tests.
- The prefill fixture remains one AOne grid and plans at least 30 distinct bounded grouped cases, including batch size 4; pin this in fixture tests.

---

### Task 1: CI prefill fixture

**Files:**
- Create: `rtp_llm/test/perf_test/examples/prefill_aone_ci.json`
- Modify: `rtp_llm/test/perf_test/gang_examples_test.py`
- Modify: `rtp_llm/test/perf_test/BUILD` if needed for test data

**Interfaces:** Produces a versioned config accepted by `parse_config`, `plan_prefill_cases`, and AOne gang dry-run, with `transport=dashsc_input_ids`, `probe_block_size=64`, `workspace_tokens`, and at least 30 unique explicit grouped batches including batch size 4.

- [ ] Write a failing fixture test that checks parser, plan, AOne gang preview, 30 unique geometries, cache and no-cache members, heterogeneous groups, and batch concurrency greater than one.
- [ ] Run the focused test and confirm it fails because the fixture is absent.
- [ ] Add the smallest bounded fixture; preserve TP2/DP1 two-machine gang and one grid.
- [ ] Run focused Bazel test and dry-run; inspect all planned cases and resource commands.
- [ ] Commit Task 1 files.

### Task 2: CI config delivery and routing

**Files:**
- Create: `rtp_llm/test/perf_test/ci_delivery.py`
- Create: `rtp_llm/test/perf_test/ci_delivery_test.py`
- Modify: `rtp_llm/test/perf_test/BUILD`
- Modify: `../.aoneci/multi-node-perf.yaml` (parent internal repository)

**Interfaces:** `deliver_config(env: Mapping[str, str], checkout: Path, destination: Path) -> dict` loads the mode-specific default or explicit config, applies explicit model/topology/execution overrides, pins AOne, and writes the delivered JSON. The YAML calls `python3 -m rtp_llm.test.perf_test.ci_delivery`.

- [ ] Write failing delivery tests for prefill defaults, preserved decode defaults, explicit overrides, invalid divisibility, and one-grid admission.
- [ ] Run the focused test and confirm the expected failure.
- [ ] Implement the pure-standard-library delivery module, then replace the YAML inline delivery block with its invocation and mode-aware defaults.
- [ ] Run delivery tests and validate the YAML structure; dry-run the delivered prefill config.
- [ ] Commit Task 2 files in each owning repository, subject to workspace permissions.

### Task 3: Collected-artifact gate

**Files:**
- Create: `rtp_llm/test/perf_test/ci_prefill_verify.py`
- Create: `rtp_llm/test/perf_test/ci_prefill_verify_test.py`
- Modify: `rtp_llm/test/perf_test/BUILD`
- Modify: `../.aoneci/multi-node-perf.yaml`
- Modify: the parent internal CUDA13 Python dependency lock only if production Bazel runfiles lack Plotly
- Modify: `rtp_llm/test/perf_test/docs/multi_node_aone.md`

**Interfaces:** `verify_prefill_run(root: Path) -> dict` checks complete AOne prefill result, successful cache probe, grouped measurements with concurrent members and observed reuse, formal report, standalone interactive HTML/SVG charts, and a FlexLB candidate with at least 30 valid geometries. CLI finds the one fresh `perf_test_results/prefill-*` directory after the pipeline run and exits nonzero on missing evidence.

- [ ] Write failing tests using a small synthetic run tree and missing-artifact mutations.
- [ ] Run focused tests and confirm the verifier is absent.
- [ ] Implement the verifier and invoke it only for prefill after `./tools/perf_test --action=run`; keep artifact upload under `always()`.
- [ ] Run focused tests and Bazel tests; update CI documentation with the exact prefill invocation and artifact contract.
- [ ] Commit Task 3 files in each owning repository, subject to workspace permissions.

### Task 4: End-to-end validation

**Files:** No required source changes; save observed commands/results in plan progress ledger.

- [ ] Run all focused Bazel targets and `./tools/perf_test --mode=prefill --action=run --config=<delivered> --dry-run`.
- [ ] Run the Qwen3-4B local equivalent if available to validate grouped DashSC/probe/report on a real model, or reuse an existing fresh equivalent run only if its artifacts match this fixture contract.
- [ ] Validate AOne pipeline syntax and, where credentials/permissions allow, trigger a real prefill pipeline run against the pinned source commit; inspect uploaded artifacts and gate result.
- [ ] Run final code review and record any remote-only limitation accurately.

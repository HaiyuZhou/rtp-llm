# Adaptive Pressure Sender Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dispatch the tiered rolling formal request sequence from an adaptive timer whose positive rate decreases with outstanding RPC backlog and whose hard capacity fails explicitly instead of waiting for completion.

**Architecture:** Add a pure backlog-rate controller to the existing continuous-pressure runner, keep seed/fill concurrency separate from the formal outstanding safety limit, and make the supervisor own deterministic formal sequence allocation. Existing fixed constant/Poisson behavior remains the default; adaptive behavior is explicitly configured through full-pressure settings.

**Tech Stack:** Python 3, dataclasses, `unittest`, fake monotonic clocks, existing cache-grid CLI/profile configuration.

**Spec:** `docs/superpowers/specs/2026-10-22-adaptive-pressure-sender-design.md`

## Global Constraints

- Seed and fill phase behavior and concurrency remain unchanged.
- Adaptive high-backlog rate is positive; it never waits at a soft watermark.
- Reaching `max_outstanding` records `sender_saturated` and drains without retrying the due sequence item.
- Tiered rolling sequence composition remains deterministic and has no cohort completion barrier.
- Existing fixed constant and Poisson arrival modes remain compatible.
- Commit and push the complete workspace change set after verification, then pull it into the target container.

---

### Task 1: Adaptive Rate Contract And Validation

**Files:**
- Modify: `rtp_llm/test/perf_test/cache_grid/runner/continuous_pressure.py`
- Modify: `rtp_llm/test/perf_test/cache_grid/runner/pressure_diagnostics.py`
- Modify: `rtp_llm/test/perf_test/cache_grid/config/perf_profile.py`
- Test: `rtp_llm/test/perf_test/cache_grid/tests/continuous_pressure_test.py`

**Interfaces:**
- Produces: `AdaptiveBacklogController(rate_min, rate_max, backlog_low, backlog_high).rate(outstanding) -> float`.
- Produces: immutable `PressureConfig` fields `rate_control`, `request_rate_min`, `request_rate_max`, `backlog_low`, `backlog_high`, `max_outstanding`, and `cohort_size`.

- [x] **Step 1: Write failing controller and validation tests**

Add tests asserting the controller returns max rate below the low watermark,
linearly interpolates, returns a positive minimum above the high watermark,
and rejects invalid rates/watermarks. Add profile-resolution tests for each new
field and require explicit adaptive settings only when `rate_control` is
`adaptive_backlog`.

- [x] **Step 2: Run the focused tests and confirm RED**

Run `python3 -m unittest rtp_llm.test.perf_test.cache_grid.tests.continuous_pressure_test`.

Expected: failures for the missing controller and configuration fields.

- [x] **Step 3: Implement the pure controller and configuration validation**

Implement the approved piecewise-linear formula. Extend `PRESSURE_OPTIONS` so
profile and `--pressure-*` CLI values resolve through the existing immutable
configuration path. Keep `rate_control="fixed"` defaults neutral and validate
adaptive fields as:

```text
0 < request_rate_min <= request_rate_max
0 <= backlog_low < backlog_high < max_outstanding
cohort_size > 0
family_schedule == tiered_rolling
arrival == constant
```

Session-level validation handles values that live outside `PressureConfig`,
namely the legacy fixed `pressure_request_rate` and arrival type.

- [x] **Step 4: Run focused tests and confirm GREEN**

Run the Task 1 command and confirm all tests pass.

- [x] **Step 5: Include the task in the consolidated verified commit**

Stage the Task 1 files and commit with message `feat(cache-grid): add adaptive backlog rate control`.

### Task 2: Timer-Driven Formal Dispatch And Saturation

**Files:**
- Modify: `rtp_llm/test/perf_test/cache_grid/runner/continuous_pressure.py`
- Modify: `rtp_llm/test/perf_test/cache_grid/runner/cache_grid_runner.py`
- Test: `rtp_llm/test/perf_test/cache_grid/tests/continuous_pressure_test.py`

**Interfaces:**
- Consumes: `AdaptiveBacklogController.rate(outstanding)` from Task 1.
- Produces: `PressureSession` adaptive summary under `result["supply"]`.
- Produces: formal dispatch fields `formal_sequence`, `outstanding_before`, and `effective_rate`.
- Changes internal payload callback to `make_payload(kind, job_id, formal_sequence)` where `formal_sequence` is `None` outside formal supply.

- [x] **Step 1: Write failing fake-clock dispatch tests**

Use request handles that never complete and assert due requests continue to be
accepted at decreasing timer-derived rates until `max_outstanding`. Assert the
next due request at the hard limit produces `sender_saturated` without waiting,
retrying, or allocating a later sequence. Add a completion test proving a
completion only changes the rate at the next timer dispatch and does not submit
from its callback.

- [x] **Step 2: Run the new tests and confirm RED**

Run individual new unittest methods so failures are attributable to the old
bounded refill behavior.

- [x] **Step 3: Separate phase concurrency from formal capacity**

Construct `BoundedSender` and the request executor with `max_outstanding` in
adaptive mode while retaining the existing `concurrency` value as the seed and
fill in-flight limit. Count seed jobs explicitly in `_run_seed_phase` rather
than using the sender's total capacity.

- [x] **Step 4: Implement adaptive absolute-time dispatch**

In `_run_formal_supply`, retain one absolute `due` time, dispatch only when the
timer reaches it, and advance it by `1 / effective_rate` after an accepted
request. When the sender is full, emit a saturation event containing sequence,
due time, outstanding count, cap, and effective rate; set failure to
`sender_saturated`; cancel through the bounded drain path. Extend `_pause` to
honor the next due time while remaining interruptible by completion and control
events.

- [x] **Step 5: Make sequence ownership deterministic**

Allocate `formal_sequence` in the supervisor before starting the worker and
increment it only after `BoundedSender.submit` accepts the handle. Pass the
captured value to lazy payload construction. Index tiered assignments by that
sequence rather than worker lock-acquisition order, preserving exact cohort
composition without constructing token IDs ahead of dispatch.

- [x] **Step 6: Add bounded supply telemetry**

Track max observed formal outstanding, rate count/sum/min/max, low/interpolated/high
region counts, and saturation state without storing an unbounded per-request
side list. Add these fields to `result["supply"]`; add per-dispatch context to
the existing durable dispatch events.

- [x] **Step 7: Run session tests and confirm GREEN**

Run `python3 -m unittest rtp_llm.test.perf_test.cache_grid.tests.continuous_pressure_test`.

- [x] **Step 8: Include the task in the consolidated verified commit**

Stage the Task 2 files and commit with message `feat(cache-grid): dispatch adaptive pressure independently`.

### Task 3: CLI Propagation, Example Profile, And Regression Coverage

**Files:**
- Modify: `rtp_llm/test/perf_test/cache_grid/runner/cache_perf.py`
- Modify: `rtp_llm/test/perf_test/cache_grid/runner/cache_grid_runner.py`
- Modify: `/data0/zhouhaiyu.zhy/tmp/continuous_pressure_dsv41_flash_tp4_mem128g_h2d.profile.jsonc`
- Modify: `rtp_llm/test/perf_test/cache_grid/docs/continuous_full_pressure_guide_zh.md`
- Test: `rtp_llm/test/perf_test/cache_grid/tests/cache_perf_test.py`
- Test: `rtp_llm/test/perf_test/cache_grid/tests/perf_profile_test.py`
- Test: `rtp_llm/test/perf_test/cache_grid/tests/continuous_pressure_test.py`

**Interfaces:**
- Consumes: new `PRESSURE_OPTIONS` fields and adaptive `PressureSession` behavior.
- Produces: reproducible H2D profile settings for a 32-request logical cohort and 256-RPC hard capacity without changing the mixed cold/hit production baseline.

- [x] **Step 1: Write failing propagation tests**

Assert profile and explicit CLI settings are frozen into the profile snapshot,
then resolve through the perf-test parser into `CacheGridRunner`, the session
manifest, and the frozen event. Assert adaptive tiered rolling accepts a
positive request rate while fixed tiered rolling preserves compatibility.

- [x] **Step 2: Run propagation tests and confirm RED**

Run `python3 -m unittest rtp_llm.test.perf_test.cache_grid.tests.cache_perf_test rtp_llm.test.perf_test.cache_grid.tests.perf_profile_test rtp_llm.test.perf_test.cache_grid.tests.continuous_pressure_test`.

- [x] **Step 3: Complete CLI and runner propagation**

Use the existing generated `PRESSURE_OPTIONS` CLI path for adaptive settings,
include resolved values in the frozen profile snapshot, and let the perf-test
parser reconstruct the immutable pressure configuration from that snapshot.
Ensure the runner uses the resolved `request_rate_max` only for adaptive mode.
Preserve the legacy `pressure_request_rate` names and defaults for
non-adaptive runs.

- [x] **Step 4: Configure and document the production example**

Set the dedicated DSV4.1 H2D profile to:

```json
{
  "rate_control": "adaptive_backlog",
  "request_rate_min": 50.0,
  "request_rate_max": 400.0,
  "backlog_low": 64,
  "backlog_high": 192,
  "max_outstanding": 256,
  "cohort_size": 32,
  "family_schedule": "tiered_rolling",
  "memory_reuse_ratio": 0.5,
  "fill_host_max_ratio": 0.8
}
```

Document that adaptive min/max rates come from the pressure profile, that
successful adaptive runs never hit saturation, and that the requested 16/16
tier split is a sequence property rather than a batch barrier.

- [x] **Step 5: Run the focused suite; run the static dry run after target sync**

Run the Task 3 unittest command, then run the documented `tools/cache_perf`
dry-run command and parse the synced profile through the direct perf-test
configuration path. Pressure window/controller settings live in the frozen
profile snapshot; only workload settings use generated runner arguments.

- [x] **Step 6: Run syntax and diff checks**

Run Python compilation, the repository-available focused suites, and
`git diff --check`. Black is not installed in the local workspace.

- [x] **Step 7: Include the task in the consolidated verified commit**

Stage the Task 3 files and commit with message `feat(cache-grid): configure adaptive H2D pressure stream`.

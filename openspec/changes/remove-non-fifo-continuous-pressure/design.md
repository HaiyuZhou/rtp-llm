# Design

## Context

See `proposal.md` for motivation and `specs/fifo-continuous-cache-pressure/spec.md` for the behavior contract. The full-pressure preload path already validates FIFO at profile, CLI, and direct-runner boundaries, but the older finite continuous path still defaults to `batch_decode`. That branch changes server batch size for seed and measurement, creates padding jobs to complete a tail batch, derives automatic client concurrency from pressure batch size, and serializes scheduler-dependent resume fields.

The same runner also owns ordinary case-mode fixed-batch measurement. Those helpers and CLI controls cannot be removed globally: only their use from `pressure_mode=continuous` is obsolete.

## Goals / Non-Goals

**Goals:**

- Resolve every continuous entry point to FIFO and reject explicit non-FIFO input consistently.
- Collapse finite continuous execution onto its existing FIFO branch without changing its count-based stopping behavior.
- Remove continuous-only scheduler mutation, padding, and fixed-batch concurrency logic.
- Keep existing result files readable and make their effective FIFO behavior explicit.

**Non-Goals:**

- No removal of batch-decode scheduler support from ordinary case/grid benchmarks.
- No change to full-pressure phase thresholds, live seed planning, mock-memory policies, adaptive backlog control, or profiling backends.
- No conversion of old non-FIFO checkpoints into FIFO epochs.
- No engine scheduler implementation changes.

## Decisions

### 1. Enforce FIFO wherever continuous mode is resolved

The public cache-perf wrapper will default continuous mode to FIFO even without preload. The underlying perf-test parser and `CacheGridRunner` constructor will reject `pressure_mode=continuous` with any other scheduler. Retaining all three checks protects profile users, direct perf-test users, resume paths, and unit-level direct callers.

Silently coercing an explicit `batch_decode` selection was rejected because it could resume or label an experiment with execution semantics different from the request. The error will name FIFO-only continuous pressure and direct users to ordinary case mode when they need fixed batching.

### 2. Keep one finite continuous producer and delete scheduler-dependent branches

The legacy finite `_run_continuous` path remains count-based, but always uses the dynamic FIFO behavior that already exists: seed requests do not call scheduler control, formal jobs are submitted exactly once, and tail padding is never generated. Automatic client concurrency resolves from service concurrency, while explicit positive bounds retain current validation.

This is preferable to redirecting finite continuous runs into full-pressure preload because preload has different live-capacity planning, time-window validity, diagnostics, and result contracts.

### 3. Preserve ordinary batch helpers and narrow their call sites

Scheduler-control helpers, fixed-batch measurement, grouped cases, and their tests remain for `pressure_mode=case`. Implementation edits will remove only calls and branches reachable from continuous execution. Tests will explicitly exercise both sides of this boundary so broad cleanup cannot accidentally remove ordinary batch coverage.

### 4. Keep result compatibility fields neutral where inexpensive

Artifacts continue to record `pressure_scheduler="fifo"`. Existing fields such as `scheduler_batch_size`, `padding_requests`, and `padding_failures` may remain with `null` or zero values to avoid unnecessary reader breakage, but no execution decision reads them. Resume compatibility compares the FIFO scheduler and rejects checkpoints that recorded `batch_decode`; obsolete pressure batch size is not part of a new FIFO run's behavior.

Removing every field immediately was rejected because the requested simplification concerns execution functionality, while schema removal would create unrelated downstream migration work.

## Risks / Trade-offs

- [Existing finite continuous commands explicitly select batch-decode] → Fail early with a migration message and point users to FIFO continuous mode or ordinary fixed-batch case mode.
- [Conditional cleanup accidentally affects ordinary batch measurements] → Retain dedicated ordinary fixed-batch regression tests and avoid deleting shared scheduler helpers.
- [Old checkpoints carry non-FIFO scheduler identity] → Reject resume rather than relabeling or partially reusing their results.
- [Neutral compatibility fields look like supported features] → Document that they are serialization compatibility only and assert that no continuous code reads them for control flow.

## Migration Plan

1. Change wrapper/parser defaults and validation so all new continuous runs resolve to FIFO and old non-FIFO commands fail before launch.
2. Remove continuous fixed-batch seed switching, padding, batch-size validation, and concurrency derivation while retaining case-mode helpers.
3. Update checkpoint/result serialization and resume tests around the FIFO-only contract.
4. Run focused parser/runner tests plus the complete ordinary batch regression suite.

Rollback restores the removed continuous batch-decode branches. Existing FIFO result directories and ordinary case-mode results require no migration.

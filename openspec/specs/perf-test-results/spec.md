# perf-test-results Specification

## Purpose

Defines durable run state, atomic case storage, versioned result and report contracts, regression artifacts, and optional publication outcomes.

## Requirements

### Requirement: Run directory is self-describing
The system SHALL persist the source config snapshot, normalized config, execution plan, manifest, workload artifacts, checkpoints, case results, summaries, reports, and traces under one run directory.

#### Scenario: New run is planned
- **WHEN** planning succeeds and execution is about to begin
- **THEN** the run directory identifies the normalized inputs, fingerprints, cases, resources, code version, and current state

#### Scenario: Inspect a completed run
- **WHEN** a downstream tool receives only the run directory
- **THEN** it can determine what was requested, what ran, which cases succeeded, and which artifacts were produced

### Requirement: Run directory has one writer
The system SHALL enforce exclusive write ownership for a run directory and SHALL allow read-only reporting only when no conflicting write action is active.

#### Scenario: Concurrent resume
- **WHEN** a second writer attempts to resume a run that is already being modified
- **THEN** the second writer fails without modifying the run

#### Scenario: Offline report without writer
- **WHEN** no run or resume action holds the directory lock
- **THEN** report generation can read persisted results

### Requirement: Case and control data are atomic
The system SHALL write case results, manifests, checkpoints, and final summaries atomically so interruption cannot expose a partially written JSON document as committed state.

#### Scenario: Interruption during case write
- **WHEN** the process stops before a case file rename completes
- **THEN** the incomplete temporary data is not treated as a committed case

#### Scenario: Aggregate results
- **WHEN** final result and summary files are generated
- **THEN** they are derived only from committed case files

### Requirement: Run state distinguishes completeness
The system SHALL represent planned execution stages and terminal `completed`, `partial`, `failed`, and `interrupted` outcomes without treating partial data as success.

#### Scenario: Complete execution
- **WHEN** every required case, collection step, report, and required publisher succeeds
- **THEN** the run state is `completed` and exit code is zero

#### Scenario: Some valid cases remain
- **WHEN** at least one valid case exists but the required plan does not complete
- **THEN** the run state is `partial`, diagnostics are preserved, and exit code is nonzero

#### Scenario: No acceptable result
- **WHEN** a run-level failure prevents acceptable case output
- **THEN** the run state is `failed` and exit code is nonzero

#### Scenario: Signal interruption
- **WHEN** execution receives an interrupt or termination signal
- **THEN** the run state is `interrupted`, committed cases remain available, and the process returns the signal-derived exit code

### Requirement: Case state records correctness failures
The system SHALL distinguish planned, running, successful, request-failed, invalid-shape, invalid-reuse, and invalid-timing case states.

#### Scenario: Invalid response shape
- **WHEN** a response does not match the planned request/output shape
- **THEN** the case is retained as `invalid_shape` and excluded from valid metrics

#### Scenario: Invalid timing
- **WHEN** required timing values are absent or inconsistent
- **THEN** the case is retained as `invalid_timing` and excluded from valid metrics

### Requirement: Results use a versioned common envelope
The system SHALL produce a versioned result envelope containing run identity, mode, backend, status, fingerprints, model metadata, summary, common case state, and mode-specific metrics sufficient to reconstruct the planned and effective workload shape.

#### Scenario: Prefill result
- **WHEN** a prefill result is written
- **THEN** its metrics include input, expected and observed reuse, compute, latency, throughput, validation, seed, prefix policy, and per-run measurements

#### Scenario: Decode result
- **WHEN** an exact-batch decode result is written
- **THEN** its metrics include workload kind, actual input lengths, requested output length, global batch, topology, worker allocation, request outcomes, latency, throughput, and applicable distribution details

#### Scenario: Target-TPOT decode result
- **WHEN** target-TPOT search selects a best batch
- **THEN** top-level decode metrics describe that selected batch and search data preserves the target, maximum, candidate step, outcome, best batch, and every attempted candidate

#### Scenario: Shared case fields
- **WHEN** a consumer reads either mode's cases
- **THEN** it can rely on common case identity, status, runs, and errors without assuming mode-specific metrics have identical meaning

### Requirement: Reports consume only valid measurements
The system SHALL exclude cases with invalid shape, reuse, or timing from formal aggregation, fitting, regression comparison, and performance charts.

#### Scenario: Complete formal report
- **WHEN** all required report inputs are valid and complete
- **THEN** the report is marked formal and may be used for regression gating

#### Scenario: Diagnostic partial report
- **WHEN** inputs are incomplete or contain invalid required cases
- **THEN** the report is visibly marked diagnostic and cannot be mistaken for a complete regression result

#### Scenario: Report generation fails
- **WHEN** measurements are committed but report generation fails
- **THEN** committed measurements remain unchanged and the run does not become `completed`

### Requirement: Baseline regression has an explicit schema
The system SHALL compare stable case keys and metrics using an expected value, comparison direction, and threshold, and SHALL generate a baseline candidate without overwriting the accepted baseline.

#### Scenario: Regression passes
- **WHEN** every required result satisfies its configured direction and threshold
- **THEN** the regression gate succeeds

#### Scenario: Regression fails
- **WHEN** a required result violates its threshold or is missing
- **THEN** the regression gate fails with the affected case key and metric

#### Scenario: Generate candidate
- **WHEN** report generation requests baseline output
- **THEN** the system writes a separate baseline candidate and leaves the accepted baseline unchanged

### Requirement: Golden generation is explicit
The system SHALL provide an explicit CI or build target for generating or validating golden artifacts and SHALL NOT silently rewrite committed golden data during a normal run.

#### Scenario: Normal performance run
- **WHEN** a standard run completes
- **THEN** committed golden data remains unchanged

#### Scenario: Golden target
- **WHEN** the explicit golden operation is invoked
- **THEN** it generates or validates artifacts using the versioned result schema

### Requirement: Publishers are policy controlled
The system SHALL publish only configured outputs, SHALL record each publisher outcome, and SHALL let a `required` policy determine whether publisher failure makes the run unsuccessful.

#### Scenario: Required publisher fails
- **WHEN** a required OSS archive or ODPS publisher fails
- **THEN** the run exits nonzero and records the publisher error

#### Scenario: Optional publisher fails
- **WHEN** a non-required publisher fails
- **THEN** measurement results remain available and the failure is recorded without converting successful measurement cases into failed cases

#### Scenario: Publisher succeeds
- **WHEN** configured publishers complete
- **THEN** the summary records their destination identity and success status

### Requirement: Publisher credentials remain external
The system SHALL obtain publisher credentials from the runtime environment and SHALL NOT persist secret values in configs, manifests, results, reports, or logs.

#### Scenario: Credentials are present
- **WHEN** a publisher reads required credentials from its runtime environment
- **THEN** publication may proceed without copying the values into run artifacts

#### Scenario: Credentials are missing
- **WHEN** required credentials are absent
- **THEN** the publisher reports a sanitized error governed by its required policy

### Requirement: Exit codes classify failures
The system SHALL use stable nonzero exit-code classes for configuration/capability, planning/resource, engine, case/partial, and collection/merge/report/publisher failures.

#### Scenario: Invalid capability combination
- **WHEN** capability validation fails
- **THEN** the process exits with the configuration/capability exit-code class

#### Scenario: Partial case execution
- **WHEN** case execution ends in a partial run
- **THEN** the process exits with the case/partial exit-code class

#### Scenario: Collection or report failure
- **WHEN** collection, merge, report, or a required publisher fails
- **THEN** the process exits with the corresponding post-execution failure class

### Requirement: Decode case identity is shape complete
The system SHALL identify each decode case by workload kind, input shape, requested output length, and batch policy so cases from different output-grid cells or strategies cannot collide.

#### Scenario: Different requested outputs
- **WHEN** two fixed cases have the same input length and batch but different output lengths
- **THEN** they receive different stable case identifiers and separate persisted case files

#### Scenario: Different batch strategies
- **WHEN** exact-batch and target-TPOT cases share the same input and output shape
- **THEN** their identities remain distinct because the normalized batch policies differ

### Requirement: Decode report preserves grid dimensions
The system SHALL persist ordered flat decode observations and deterministic matrix views that retain input, output, batch policy, status, and selected performance metrics.

#### Scenario: Exact-batch matrix groups
- **WHEN** a fixed workload contains multiple exact batch sizes
- **THEN** the report contains one input-by-output matrix group for each batch size

#### Scenario: TPOT matrix cells
- **WHEN** a fixed target-TPOT workload completes
- **THEN** every input-by-output cell retains the best batch, actual TPOT, throughput, and search outcome

#### Scenario: Diagnostic matrix
- **WHEN** a grid contains failed or invalid cases
- **THEN** their cells retain status and diagnostics without contributing valid aggregate metrics and the report is marked diagnostic

### Requirement: Inline profiling artifacts are explicit
The system SHALL persist a profile manifest for each attempted inline decode profile and SHALL distinguish profile artifact failure from formal performance failure.

#### Scenario: Profile succeeds
- **WHEN** trace collection completes
- **THEN** the case result references a completed profile manifest containing the profiled shape, settings, trace identity, and discovered artifacts

#### Scenario: Profile fails after valid measurement
- **WHEN** a case has valid formal metrics but profile triggering, execution, or trace flush fails
- **THEN** the case remains performance-successful, its profiling status is failed, and the summary increments the profiling failure count

#### Scenario: Profiling is disabled
- **WHEN** inline profiling is not enabled
- **THEN** no profile request is sent and the result records profiling as disabled or omitted according to the result schema

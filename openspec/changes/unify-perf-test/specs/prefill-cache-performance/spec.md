# Spec Delta

## Purpose

Defines cache-aware prefill workload planning, reuse correctness, local recovery and diagnostics, and AOne prefill execution behavior.

## ADDED Requirements

### Requirement: Prefill uses only cache-aware grid semantics
The system SHALL define prefill performance cases by batch size, total input length, expected reused tokens, and compute tokens, and SHALL NOT provide the legacy ordinary full-request prefill mode.

#### Scenario: Plan a valid prefill case
- **WHEN** a requested cache length, compute step, and batch size satisfy all workload constraints
- **THEN** the plan contains a case with frozen expected reuse, compute length, seed, and prefix policy

#### Scenario: Request legacy prefill behavior
- **WHEN** a config requests ordinary prefill without controlled reuse
- **THEN** the system rejects the workload as unsupported

### Requirement: Prefill planner enforces workload constraints
The system SHALL enforce cache alignment, committed cache-tail tokens, maximum input length, minimum compute tokens, allowed batch sizes, and maximum case count before execution.

#### Scenario: Aligned case
- **WHEN** a candidate satisfies alignment and length constraints
- **THEN** the planner emits a deterministic case identifier for it

#### Scenario: Invalid cache and compute combination
- **WHEN** a candidate exceeds maximum input length or leaves fewer than the minimum compute tokens
- **THEN** the planner rejects the candidate before starting a model

#### Scenario: Excessive case expansion
- **WHEN** a fixed or sampled grid would exceed the configured maximum case count
- **THEN** planning fails with the expanded count and configured limit

### Requirement: Prefill request construction is reproducible
The system SHALL persist the seed and prefix policy needed to reconstruct each planned request and SHALL use them for execution, resume, and profile actions.

#### Scenario: Repeat a case
- **WHEN** the same case is reconstructed from its persisted plan
- **THEN** it produces the same shared prefix and measured request token sequence

#### Scenario: Sample cache lengths
- **WHEN** the workload uses constrained random cache-length sampling
- **THEN** the same seed and normalized config produce the same sampled cases

### Requirement: Observed reuse is a correctness condition
The system SHALL compare observed reuse with expected reuse for every measured case and SHALL exclude a mismatch from valid performance aggregation.

#### Scenario: Exact reuse match
- **WHEN** response shape and timing are valid and observed reuse equals expected reuse
- **THEN** the measurement is eligible for prefill metrics and reporting

#### Scenario: Reuse mismatch
- **WHEN** observed reuse differs from the planned expected reuse
- **THEN** the case is marked `invalid_reuse`, retained for diagnostics, and excluded from valid performance points

#### Scenario: Reuse telemetry missing
- **WHEN** the response does not provide data sufficient to validate reuse
- **THEN** the case is invalid rather than treated as a cache hit or cache miss by assumption

### Requirement: Local prefill run persists case progress
The system SHALL atomically persist each completed local prefill case so an interruption does not discard earlier valid cases.

#### Scenario: Process interrupted between cases
- **WHEN** one or more cases are committed and the process is interrupted before the plan finishes
- **THEN** committed cases remain readable and the run is marked interrupted

#### Scenario: Case request fails
- **WHEN** one case request fails but the engine remains usable
- **THEN** the system records that case failure and continues with independent planned cases

### Requirement: Local prefill resume validates identity
The system SHALL support resume only for local prefill runs and SHALL require matching config, workload, model, and runner fingerprints.

#### Scenario: Resume matching run
- **WHEN** all required fingerprints match
- **THEN** resume preserves `ok` cases and reruns planned, interrupted, and failed cases

#### Scenario: Resume mismatched run
- **WHEN** any required fingerprint differs or a required snapshot is missing
- **THEN** resume fails without changing existing case results

### Requirement: Local prefill profile is isolated
The system SHALL profile selected cases from a local prefill run by recreating their original requests and SHALL write profile artifacts outside formal results and checkpoints.

#### Scenario: Profile selected cases
- **WHEN** a user supplies case identifiers from the original plan
- **THEN** the system validates exact reuse and writes a new profile session for those cases

#### Scenario: Profile invalid case identifier
- **WHEN** a requested case identifier is absent from the original plan
- **THEN** the profile action rejects it without modifying the run

#### Scenario: Profile completes
- **WHEN** profile collection succeeds or fails
- **THEN** formal case results, checkpoints, and summaries remain unchanged

### Requirement: Prefill report is reproducible offline
The system SHALL generate prefill reports from persisted run data without launching a model and SHALL allow the report action to be repeated.

#### Scenario: Complete run report
- **WHEN** all planned cases needed by the report are valid
- **THEN** the system produces a formal report and marks it complete

#### Scenario: Partial run report
- **WHEN** persisted data contains missing or invalid cases
- **THEN** the system produces only a clearly marked diagnostic report and excludes invalid measurements

### Requirement: AOne prefill shards have exclusive ownership
The system SHALL partition all planned prefill cases into non-overlapping AOne shards using estimated case cost and SHALL run each shard with one model startup.

#### Scenario: Multi-shard plan
- **WHEN** an AOne prefill run uses multiple shards
- **THEN** every planned case belongs to exactly one shard and the shard plans collectively cover the full workload

#### Scenario: Unequal case cost
- **WHEN** planned cases have different estimated costs
- **THEN** sharding balances estimated work rather than only the number of cases

### Requirement: AOne prefill collection is strict
The system SHALL verify run identity, config and workload fingerprints, shard ownership, completion claims, and case uniqueness before accepting collected results.

#### Scenario: Valid collection
- **WHEN** every expected shard returns only its owned cases with matching fingerprints
- **THEN** the system merges them into one run result and permits offline reporting

#### Scenario: Fingerprint or ownership mismatch
- **WHEN** a shard has a mismatched fingerprint, unknown case, out-of-scope case, or duplicate case
- **THEN** collection rejects the inconsistent data and the run does not complete successfully

#### Scenario: Missing shard or case
- **WHEN** an expected shard or declared completed case result is missing
- **THEN** available diagnostics may be retained but the run is marked partial or failed and exits nonzero

# decode-performance Specification

## Purpose

Defines the supported local decode workloads, output-length constraint, data-parallel request accounting, and target-TPOT batch optimization behavior.

## Requirements

### Requirement: Decode runs locally
The system SHALL execute decode performance workloads only with the local backend and SHALL support the `run` action without decode resume or profile actions.

#### Scenario: Local decode run
- **WHEN** a valid decode workload selects the local backend and `run`
- **THEN** the system plans and executes the decode workload

#### Scenario: Decode follow-up action
- **WHEN** a user requests decode `resume` or `profile`
- **THEN** the system rejects the action before modifying a run directory or launching a model

### Requirement: Fixed decode grid
The system SHALL generate exact-batch fixed decode cases from the Cartesian product of configured input lengths, output lengths, and batch sizes, and SHALL generate target-TPOT cases from the Cartesian product of input and output lengths.

#### Scenario: Generate fixed grid
- **WHEN** input lengths contain three values, output lengths contain two values, and fixed batch sizes contain four values
- **THEN** the planner produces twenty-four decode cases with deterministic identifiers

#### Scenario: Generate target-TPOT grid
- **WHEN** input lengths contain three values, output lengths contain two values, and the batch strategy is target-TPOT
- **THEN** the planner produces six decode cases and each case searches its batch independently

#### Scenario: Invalid grid dimension
- **WHEN** an input length, output length, batch value, or known input-plus-output capacity constraint is invalid
- **THEN** the system rejects the invalid value before model startup

### Requirement: Distribution decode workload
The system SHALL support a validated request distribution that determines actual input sequence lengths, applies one scalar output length, and uses the configured exact-batch or target-TPOT strategy and measurement settings.

#### Scenario: Valid distribution input
- **WHEN** the distribution source and fixed batch policy are valid
- **THEN** the plan records the resolved lengths for each exact batch and execution records the actual sequence lengths used

#### Scenario: Valid searchable distribution
- **WHEN** target-TPOT uses a sequence-length distribution containing at least the configured maximum batch
- **THEN** each candidate uses the corresponding prefix of the resolved sequence lengths

#### Scenario: Invalid distribution input
- **WHEN** the distribution is malformed, empty, contains invalid sequence lengths, or cannot supply the selected batch strategy
- **THEN** planning fails before model startup

### Requirement: Decode measurements preserve request outcomes
The system SHALL retain successful and failed request counts and the per-run wait, prefill, decode, TPOT, total latency, and token-throughput measurements needed by the selected workload.

#### Scenario: Partially failed batch
- **WHEN** some requests in a measured batch fail
- **THEN** the case records both successful and failed counts and is not represented as a fully successful measurement

#### Scenario: Valid measured batch
- **WHEN** all requests complete with valid timing and output shape
- **THEN** the case is eligible for decode metric aggregation

### Requirement: Data-parallel accounting uses global workload size
The system SHALL schedule decode requests across configured data-parallel workers and SHALL report the global batch and topology separately from worker-local allocation.

#### Scenario: DP greater than one
- **WHEN** a global decode batch is distributed over multiple DP workers
- **THEN** the plan and result retain the global batch size, DP topology, and deterministic worker allocation

#### Scenario: Uneven worker allocation
- **WHEN** global batch size is not evenly divisible by the DP degree
- **THEN** every request is assigned once and the reported global batch is not replaced by a worker-local batch

### Requirement: Target-TPOT batch search
The system SHALL search each planned input/output workload for the largest usable candidate satisfying the TPOT target, SHALL preserve every attempted candidate and outcome, and SHALL report formal case metrics for the selected best batch.

#### Scenario: Search finds a boundary
- **WHEN** multiple candidates succeed and only a prefix satisfies the TPOT target
- **THEN** the result identifies the largest satisfying batch, uses that batch for top-level metrics, and includes the complete search trajectory

#### Scenario: Capacity failure during search
- **WHEN** a candidate batch fails because of capacity limits
- **THEN** the failure is recorded as a search observation and the bounded search continues or terminates without representing that candidate as formal metrics

#### Scenario: No batch satisfies target
- **WHEN** no valid candidate satisfies the TPOT target
- **THEN** the case reports `target_tpot_not_met`, preserves all observations, and contributes to a partial nonzero run

#### Scenario: Maximum batch is not selected
- **WHEN** a best batch smaller than the configured maximum is found
- **THEN** the system does not substitute or force an additional maximum-batch measurement as the case result

### Requirement: Decode report is generated by run
The system SHALL generate a machine-readable decode report during `run`, with flat observations and fixed-grid input-by-output matrix views, and SHALL NOT expose a separate decode report action.

#### Scenario: Decode run completes
- **WHEN** an exact fixed-grid run completes
- **THEN** the report groups one input-by-output matrix per configured batch and retains status, TPOT, throughput, total latency, and success rate for each cell

#### Scenario: Target-TPOT grid report
- **WHEN** a target-TPOT grid run completes
- **THEN** each input-by-output cell reports search outcome, best batch, actual TPOT, and throughput

#### Scenario: Distribution report
- **WHEN** a distribution run completes
- **THEN** the report emits ordered case observations without claiming the distribution is a rectangular matrix

#### Scenario: Invalid or incomplete case
- **WHEN** any required decode case is invalid or incomplete
- **THEN** the report is marked diagnostic and cannot be mistaken for a complete formal result

#### Scenario: Separate decode report requested
- **WHEN** a user requests `--mode=decode --action=report`
- **THEN** capability validation rejects the action

### Requirement: Decode output length contract
The system SHALL require a non-empty array of positive output lengths for fixed-grid workloads and exactly one positive scalar output length for distribution workloads.

#### Scenario: Fixed output grid
- **WHEN** a fixed decode config supplies multiple positive output lengths
- **THEN** the planner treats every output length as an independent grid dimension

#### Scenario: Distribution scalar output
- **WHEN** a distribution decode config supplies one positive output length
- **THEN** every case derived from that distribution uses the configured scalar output length

#### Scenario: Invalid output form
- **WHEN** a fixed workload supplies scalar `output_length` or a distribution workload supplies `output_lengths`
- **THEN** config validation fails before model startup

#### Scenario: Generation config attempts to override output shape
- **WHEN** decode `generate_config` supplies `min_new_tokens` or `max_new_tokens` that conflicts with the planned output length
- **THEN** the request uses the planned output length so the grid shape remains authoritative

### Requirement: Explicit decode batch strategy
The system SHALL require each decode workload to select either an exact fixed-batch strategy or a target-TPOT strategy with mutually exclusive parameters.

#### Scenario: Exact fixed batches
- **WHEN** `batch.strategy` is `fixed`
- **THEN** the config requires a non-empty array of positive batch sizes and rejects TPOT search parameters

#### Scenario: Target-TPOT policy
- **WHEN** `batch.strategy` is `target_tpot`
- **THEN** the config requires a positive maximum batch, candidate step, and TPOT target and rejects fixed batch sizes

#### Scenario: Distribution batch map with search
- **WHEN** a distribution uses an explicit per-batch sequence-length map with target-TPOT search
- **THEN** planning fails because unlisted search candidates cannot be constructed reliably

### Requirement: Inline decode profiling
The system SHALL optionally profile decode cases after formal measurement during `run`, SHALL keep profiling disabled by default, and SHALL keep profiling outcomes separate from performance correctness.

#### Scenario: Fixed case profiling
- **WHEN** inline profiling is enabled for a successful exact-batch case
- **THEN** the system profiles the measured batch after formal metrics are complete and records a profile manifest

#### Scenario: TPOT case profiling
- **WHEN** inline profiling is enabled and target-TPOT search finds a best batch
- **THEN** the system profiles only the selected best batch and does not profile search candidates

#### Scenario: Profiling failure
- **WHEN** formal measurement succeeds but profile triggering or trace collection fails
- **THEN** case performance remains successful while profiling is marked failed and the run summary reports the artifact failure

#### Scenario: Standalone decode profile action
- **WHEN** a user requests `--mode=decode --action=profile`
- **THEN** capability validation still rejects the action before side effects

### Requirement: Unified externally managed decode execution
The system SHALL execute an externally managed multi-node decode workload from the same normalized plan, case semantics, batch strategies, and result contract as a normal local decode run.

#### Scenario: Multi-node master executes cases
- **WHEN** all ranks load the same normalized decode config and the externally managed server is ready
- **THEN** the master rank executes the unified decode plan while non-master ranks only participate in service lifetime coordination

#### Scenario: Multi-node topology mismatch
- **WHEN** the normalized topology differs from the running distributed service topology
- **THEN** execution fails before measuring a decode case

#### Scenario: Removed callable grid API
- **WHEN** a caller imports the former `run_single` or `run_cartesian_grid` API
- **THEN** no compatibility wrapper or alias is provided

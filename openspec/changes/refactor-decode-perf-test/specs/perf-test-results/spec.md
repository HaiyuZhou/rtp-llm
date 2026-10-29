# Spec Delta

## ADDED Requirements

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

## MODIFIED Requirements

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

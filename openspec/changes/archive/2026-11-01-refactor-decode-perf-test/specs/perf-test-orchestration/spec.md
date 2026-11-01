# Spec Delta

## ADDED Requirements

### Requirement: Externally managed local runtime boundary
The system SHALL allow an externally managed local or multi-node server to supply runtime transport and prompt services to a normalized decode plan without transferring server lifecycle ownership to the decode case executor.

#### Scenario: Externally managed server is ready
- **WHEN** a launcher has started the distributed model service and validated its topology
- **THEN** the launcher supplies the runtime endpoint to the unified decode executor and retains responsibility for service shutdown

#### Scenario: Worker rank receives the plan
- **WHEN** a non-master rank loads the normalized plan for an externally managed run
- **THEN** it does not execute duplicate performance cases or produce a competing final result

## MODIFIED Requirements

### Requirement: Normalized immutable execution plan
The system SHALL normalize defaults and paths, compute stable config and workload fingerprints, and freeze cases, resources, backend, mode, action, batch policy, and result location into an immutable execution plan before execution, including externally managed decode execution.

#### Scenario: Equivalent config normalization
- **WHEN** two source configs resolve to the same canonical values
- **THEN** they produce the same config fingerprint and equivalent execution plans

#### Scenario: Runner consumes a plan
- **WHEN** execution begins
- **THEN** workload runners use the frozen plan and do not reinterpret the source config or infer undeclared environment parameters

#### Scenario: Multi-node config distribution
- **WHEN** an externally managed multi-node decode run is launched
- **THEN** every rank loads the same normalized config artifact and the master executes the same frozen case identities as a normal local decode run

#### Scenario: Runtime topology validation
- **WHEN** an externally managed runtime is attached to a decode plan
- **THEN** its actual TP, DP, PP, and EP topology is validated against the frozen plan before case execution

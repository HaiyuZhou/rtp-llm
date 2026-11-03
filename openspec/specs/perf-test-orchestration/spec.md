# perf-test-orchestration Specification

## Purpose

Defines the single performance-test entry point, configuration contract, capability validation, and execution planning shared by prefill and decode workloads.

## Requirements

### Requirement: Single user entry point
The system SHALL expose `./tools/perf_test` as the only supported user entry point and SHALL require an explicit `prefill` or `decode` mode and an explicit action.

#### Scenario: Start a new run
- **WHEN** a user invokes `./tools/perf_test --mode=prefill --action=run --config=<path>`
- **THEN** the system resolves the supplied config and plans a prefill run

#### Scenario: Use a removed entry point
- **WHEN** a user or CI target refers to an old perf test or cache perf entry point
- **THEN** the repository provides no compatibility wrapper or alias for that entry point

### Requirement: Action input source
The system SHALL accept a config file for `run` and a run directory for `resume`, `profile`, and `report`, and SHALL reject conflicting sources or command-line overrides of persisted run parameters.

#### Scenario: Conflicting inputs
- **WHEN** an invocation supplies both `--config` and `--run-dir`
- **THEN** the system exits with a configuration error before creating or modifying a run directory

#### Scenario: Resume override
- **WHEN** a resume invocation attempts to override model, workload, or execution settings
- **THEN** the system rejects the invocation instead of mixing parameters with the persisted run

### Requirement: Strict versioned config
The system SHALL parse one versioned config containing model, engine, workload, execution, output, regression, and publish sections, and SHALL reject unknown fields except within declared extension maps.

#### Scenario: Valid extension map
- **WHEN** engine arguments, engine environment values, or decode generation values are supplied in their declared extension maps
- **THEN** the system preserves them in the normalized config and manifest

#### Scenario: Unknown structured field
- **WHEN** a config contains an unknown field outside a declared extension map
- **THEN** validation fails with the complete field path and expected contract

#### Scenario: Unsupported interpolation
- **WHEN** a config contains an include directive, shell expression, secret reference, or environment-variable interpolation
- **THEN** the system rejects it rather than evaluating it

### Requirement: Capability validation before side effects
The system SHALL validate the mode, action, and backend combination before creating outputs, building targets, launching a model, or submitting a remote job.

#### Scenario: Supported local prefill action
- **WHEN** mode is `prefill`, backend is `local`, and action is `run`, `resume`, `profile`, or `report`
- **THEN** capability validation succeeds

#### Scenario: Supported AOne prefill run
- **WHEN** mode is `prefill`, backend is `aone`, and action is `run`
- **THEN** capability validation succeeds

#### Scenario: Unsupported AOne action
- **WHEN** backend is `aone` and the action is `resume` or `profile`
- **THEN** validation fails before any local or remote side effect

#### Scenario: Supported AOne decode run
- **WHEN** mode is `decode`, backend is `aone`, and action is `run`
- **THEN** capability validation succeeds and single-machine rejection, if any, is decided later by gang plan validation

#### Scenario: Unsupported AOne decode
- **WHEN** mode is `decode`, backend is `aone`, and the action is anything other than `run`
- **THEN** validation fails before any local or remote side effect

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

### Requirement: Dry-run has no execution side effects
The system SHALL support dry-run validation and planning without creating a formal run directory, building targets, launching a model, submitting jobs, or publishing outputs.

#### Scenario: Local dry-run
- **WHEN** a valid local config is invoked with `--dry-run`
- **THEN** the system reports normalized configuration, cases, resources, and commands without starting the engine

#### Scenario: AOne dry-run
- **WHEN** a valid AOne prefill config is invoked with `--dry-run`
- **THEN** the system reports shard and Bazel target plans without building or submitting them

#### Scenario: Gang dry-run
- **WHEN** a valid gang-mode AOne config is invoked with `--dry-run`
- **THEN** the system reports the derived machine count, member layout, and final build and test commands without starting a broker or submitting jobs

### Requirement: One model lifecycle per local plan
The system SHALL start the model once for a local run plan, share that session across its cases, and close it on success, failure, or process interruption.

#### Scenario: Multiple local cases
- **WHEN** a local plan contains multiple cases
- **THEN** all cases execute against one model startup for that plan

#### Scenario: Interrupted engine session
- **WHEN** the process receives an interruption while a model is running
- **THEN** the system closes the model session and preserves already committed case results

### Requirement: AOne configuration is delivered as runfiles
The system SHALL resolve and validate AOne input on the submit host and SHALL deliver normalized config, workload, and shard or gang ownership files as Bazel runfiles without depending on submit-host local paths.

#### Scenario: Remote shard starts
- **WHEN** an AOne shard action starts
- **THEN** it reads normalized config, workload, and shard plan from declared runfiles

#### Scenario: Gang member starts
- **WHEN** a gang member action starts
- **THEN** it reads normalized config and the gang plan from declared runfiles

#### Scenario: External workload input
- **WHEN** the source config references a small grid or distribution file
- **THEN** the submit host validates and materializes its content into the remote workload artifact

#### Scenario: Model path remains external
- **WHEN** the config identifies a model checkpoint or tokenizer
- **THEN** the normalized config retains a runtime-accessible path and does not package model data into the generated Bazel target

### Requirement: Gang execution derivation and validation
The system SHALL derive the gang machine count from the optional `execution.gpus_per_machine` field with `world_size` as its default, SHALL reject an explicit `machine_count` field, a `shard_count` field when the derived machine count exceeds one, a `world_size` that does not divide evenly, and `max_parallel` below the machine count during config validation, and SHALL keep machine count one on the existing single-machine AOne path.

#### Scenario: Default keeps single machine
- **WHEN** an AOne config omits `gpus_per_machine`
- **THEN** the derived machine count is one and the existing single-machine path is selected unchanged

#### Scenario: Gang derivation
- **WHEN** an AOne config sets `gpus_per_machine` to a value below `world_size` that divides it evenly
- **THEN** the derived machine count exceeds one and gang mode is selected

#### Scenario: Explicit machine count rejected
- **WHEN** a config supplies `execution.machine_count`
- **THEN** validation fails with the reserved-field reason before any side effect

#### Scenario: Shard count rejected in gang mode
- **WHEN** a gang-mode config also supplies `shard_count`
- **THEN** validation fails because case-shard and gang semantics are mutually exclusive

#### Scenario: Non-divisible world size rejected
- **WHEN** `gpus_per_machine` does not divide `world_size` evenly
- **THEN** validation fails with the derived machine count and world size

#### Scenario: Insufficient parallelism rejected
- **WHEN** `max_parallel` is below the derived machine count
- **THEN** validation fails rather than clamping the value

#### Scenario: Single-machine decode AOne rejected
- **WHEN** mode is `decode`, backend is `aone`, and the derived machine count is one
- **THEN** planning fails with a message naming `gpus_per_machine` before any side effect

### Requirement: Externally managed local runtime boundary
The system SHALL allow an externally managed local or multi-node server to supply runtime transport and prompt services to a normalized decode plan without transferring server lifecycle ownership to the decode case executor.

#### Scenario: Externally managed server is ready
- **WHEN** a launcher has started the distributed model service and validated its topology
- **THEN** the launcher supplies the runtime endpoint to the unified decode executor and retains responsibility for service shutdown

#### Scenario: Worker rank receives the plan
- **WHEN** a non-master rank loads the normalized plan for an externally managed run
- **THEN** it does not execute duplicate performance cases or produce a competing final result

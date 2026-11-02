# Spec Delta

## MODIFIED Requirements

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

## ADDED Requirements

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

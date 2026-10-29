# Spec Delta

## Purpose

Defines the single performance-test entry point, configuration contract, capability validation, and execution planning shared by prefill and decode workloads.

## ADDED Requirements

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

#### Scenario: Unsupported AOne decode
- **WHEN** mode is `decode` and backend is `aone`
- **THEN** validation fails before any local or remote side effect

### Requirement: Normalized immutable execution plan
The system SHALL normalize defaults and paths, compute stable config and workload fingerprints, and freeze cases, resources, backend, mode, action, and result location into an immutable execution plan before execution.

#### Scenario: Equivalent config normalization
- **WHEN** two source configs resolve to the same canonical values
- **THEN** they produce the same config fingerprint and equivalent execution plans

#### Scenario: Runner consumes a plan
- **WHEN** execution begins
- **THEN** workload runners use the frozen plan and do not reinterpret the source config or infer undeclared environment parameters

### Requirement: Dry-run has no execution side effects
The system SHALL support dry-run validation and planning without creating a formal run directory, building targets, launching a model, submitting jobs, or publishing outputs.

#### Scenario: Local dry-run
- **WHEN** a valid local config is invoked with `--dry-run`
- **THEN** the system reports normalized configuration, cases, resources, and commands without starting the engine

#### Scenario: AOne dry-run
- **WHEN** a valid AOne prefill config is invoked with `--dry-run`
- **THEN** the system reports shard and Bazel target plans without building or submitting them

### Requirement: One model lifecycle per local plan
The system SHALL start the model once for a local run plan, share that session across its cases, and close it on success, failure, or process interruption.

#### Scenario: Multiple local cases
- **WHEN** a local plan contains multiple cases
- **THEN** all cases execute against one model startup for that plan

#### Scenario: Interrupted engine session
- **WHEN** the process receives an interruption while a model is running
- **THEN** the system closes the model session and preserves already committed case results

### Requirement: AOne configuration is delivered as runfiles
The system SHALL resolve and validate AOne input on the submit host and SHALL deliver normalized config, workload, and shard ownership files as Bazel runfiles without depending on submit-host local paths.

#### Scenario: Remote shard starts
- **WHEN** an AOne shard action starts
- **THEN** it reads normalized config, workload, and shard plan from declared runfiles

#### Scenario: External workload input
- **WHEN** the source config references a small grid or distribution file
- **THEN** the submit host validates and materializes its content into the remote workload artifact

#### Scenario: Model path remains external
- **WHEN** the config identifies a model checkpoint or tokenizer
- **THEN** the normalized config retains a runtime-accessible path and does not package model data into the generated Bazel target

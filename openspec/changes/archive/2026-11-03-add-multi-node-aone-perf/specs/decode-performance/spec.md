# Spec Delta

## MODIFIED Requirements

### Requirement: Decode runs locally
The system SHALL execute decode performance workloads with the local backend and, when the derived gang machine count exceeds one, with the AOne backend, and SHALL support the `run` action without decode resume or profile actions.

#### Scenario: Local decode run
- **WHEN** a valid decode workload selects the local backend and `run`
- **THEN** the system plans and executes the decode workload

#### Scenario: Gang AOne decode run
- **WHEN** a valid decode workload selects the AOne backend and derives a gang machine count above one
- **THEN** the gang executes the unified decode plan on the master member with the same case semantics as a local run

#### Scenario: Decode follow-up action
- **WHEN** a user requests decode `resume` or `profile`
- **THEN** the system rejects the action before modifying a run directory or launching a model

#### Scenario: Single-machine AOne decode rejected
- **WHEN** a decode workload selects the AOne backend and derives a machine count of one
- **THEN** the system rejects the combination during planning before any side effect

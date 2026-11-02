# Spec Delta

## Purpose

Defines how one AOne performance-test task occupies multiple machines as one gang: machine exclusivity, token-authenticated rendezvous, the gang worker contract, gang result collection, and unified-entry execution staging.

## ADDED Requirements

### Requirement: Gang members hold exclusive machines
The system SHALL request the registered per-host GPU count of the selected GPU type for every gang shard action and SHALL restrict each worker to its local world size in devices, so two gang members cannot occupy the same machine.

#### Scenario: Full-host request per member
- **WHEN** a gang of two members runs on eight-card hosts for a two-GPU topology
- **THEN** each member's test action requests the registered per-host GPU count, not the local world size

#### Scenario: Worker device restriction
- **WHEN** a gang member starts its local rank group
- **THEN** the member exposes only its local world size in devices to the distributed server

### Requirement: Gang formation uses a token-authenticated rendezvous
The system SHALL form a gang through a broker on the submit host that authenticates every request with a bearer token, serves only bounded run metadata, accepts one registration per shard, records each member's HTTP source address as its routable address with a hostname cross-check, and rejects duplicate-host registrations and fingerprint mismatches.

#### Scenario: Register a member
- **WHEN** a gang member registers with its shard index, hostname, and port
- **THEN** the broker records the registration's source address as the member's routable address and confirms the hostname matches

#### Scenario: Invalid token
- **WHEN** a request reaches the broker without the valid bearer token
- **THEN** the broker rejects it and no run metadata is disclosed

#### Scenario: Duplicate shard registration
- **WHEN** two registrations claim the same shard index
- **THEN** the broker rejects the second registration

#### Scenario: Two members on one host
- **WHEN** two registrations resolve to the same host address or hostname
- **THEN** the broker rejects the registration and the gang fails

#### Scenario: Fingerprint mismatch
- **WHEN** a member registers with a config fingerprint different from the gang's frozen plan
- **THEN** the broker rejects the registration

### Requirement: Rendezvous deadlines bound gang formation
The system SHALL enforce a register deadline, a run deadline, and a heartbeat expiry on the broker so an incomplete or stalled gang fails fast instead of hanging.

#### Scenario: Register deadline exceeded
- **WHEN** not all members register before the register deadline
- **THEN** the broker fails the gang and registered members exit nonzero

#### Scenario: Heartbeat expiry
- **WHEN** a member stops sending heartbeats beyond the expiry interval
- **THEN** the broker marks the member lost and fails the gang

#### Scenario: Run deadline exceeded
- **WHEN** the gang has not completed before the run deadline
- **THEN** the broker fails the gang and every member exits nonzero

### Requirement: Gang executes one plan on the master member
The system SHALL execute the frozen unified plan on the master member against the gang's distributed server while non-master members serve their local ranks and wait for master completion, and SHALL propagate any member failure to a nonzero exit for every member while preserving partial artifacts.

#### Scenario: Master executes the whole plan
- **WHEN** all members are ready
- **THEN** the master member executes every planned case through the gang's distributed server and non-master members wait

#### Scenario: Member failure propagates
- **WHEN** one member fails during gang execution
- **THEN** every member exits nonzero and the master's partial run artifacts remain collectable

### Requirement: Gang results match local run artifacts
The system SHALL collect the master member's complete run artifacts and non-master member logs, and SHALL produce the same versioned result envelope and reports as a local run of the same config, recording gang topology and broker identity in the manifest.

#### Scenario: Local parity
- **WHEN** a gang run completes successfully
- **THEN** the collected run directory contains the same envelope, report, and trace artifacts a local run of the same config would produce, with backend recorded as AOne

#### Scenario: Non-master diagnostics
- **WHEN** a gang run finishes
- **THEN** non-master members contribute logs through their test outputs and no partial case results

### Requirement: Gang execution is driven by the unified entry
The system SHALL drive gang execution from the same unified entry as single-machine AOne runs, starting the broker in the entry process on the submit host, persisting stage states, and preserving the original test failure exit code over collection errors while always attempting partial collection.

#### Scenario: Staged execution
- **WHEN** a gang run progresses through rendezvous, build, submit, collect, and report stages
- **THEN** each stage transition is persisted to the manifest and a stage failure records the failing stage

#### Scenario: Test failure wins over collection errors
- **WHEN** the remote test fails and collection also reports errors
- **THEN** the run exits with the test failure class and partial collection results are retained

#### Scenario: CI pipeline consumes the entry
- **WHEN** a CI pipeline job runs the unified entry command on the CI runner
- **THEN** the entry performs broker start, package generation, remote build, remote test, and collection without additional orchestration commands

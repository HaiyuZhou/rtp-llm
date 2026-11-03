# Spec Delta

## ADDED Requirements

### Requirement: Prefill scheduler matches each request phase
The system SHALL explicitly configure the performance scheduler for prefill and SHALL use a batch target that can be satisfied by the requests issued in the current phase before waiting for responses.

#### Scenario: Cached case seed phase
- **WHEN** a cached prefill case needs prefix-seed requests before measurement
- **THEN** the scheduler accepts each seed request without waiting for not-yet-issued measurement members, and every seed completes or fails before the measurement phase

#### Scenario: Measured batch phase
- **WHEN** a prefill case begins a measured round with batch size greater than one
- **THEN** the scheduler is configured in prefill mode for the effective batch size on each participating worker before the round's requests are sent

#### Scenario: Scheduler configuration fails
- **WHEN** the scheduler does not acknowledge the required prefill mode or batch target
- **THEN** the case fails without emitting successful measurement metrics or blocking on an incomplete batch

### Requirement: Prefill measured rounds issue concurrent requests
The system SHALL issue all planned members of one measured prefill round concurrently, await every member's terminal response, and validate each member's shape, reuse, and timing before accepting the round.

#### Scenario: Batch greater than one
- **WHEN** a planned prefill case has batch size four
- **THEN** each measured round presents four overlapping requests to the engine and records one response for each planned member

#### Scenario: Batch size one
- **WHEN** a planned prefill case has batch size one
- **THEN** the single measured request remains valid without requiring additional requests

#### Scenario: One member fails
- **WHEN** any member of a measured round fails, has the wrong shape, or reports incorrect reuse or timing
- **THEN** the case is non-successful and no successful batch-throughput point is emitted, while the runner waits for or accounts for every submitted member

#### Scenario: Distributed prefill
- **WHEN** a prefill case runs with more than one DP worker
- **THEN** batch sizes greater than one are rejected during planning until deterministic per-DP request routing is available; batch size one remains supported

#### Scenario: Resume an old prefill plan
- **WHEN** a resume or profile action loads a prefill plan without the concurrent-batch measurement contract
- **THEN** the action fails with an explicit migration error rather than writing new-contract measurements under old case identities

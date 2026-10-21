# Spec Delta

## Purpose

Defines a production-qualified FIFO continuous-pressure workload that adapts its reusable seed working set to the live FULL cache capacity and verifies bidirectional cache migration.

## ADDED Requirements

### Requirement: Formal runs derive seed capacity from live FULL topology
After service startup, the system SHALL use the authoritative `GetCachePressure` snapshot to derive the effective reusable seed-family count for every required FULL cache group. It SHALL use owner-device logical pool capacity and SHALL count a shared TP-owned pool only once.

#### Scenario: Shared TP pool is counted once
- **WHEN** a formal run observes one FULL pool owned by one device and shared by multiple TP participants
- **THEN** the seed planner uses the owner's logical block capacity once rather than summing the participants' reported views

#### Scenario: Multiple required FULL groups
- **WHEN** a formal run requires more than one FULL cache group
- **THEN** the derived seed-family count satisfies the pressure and retention constraints of every required group

#### Scenario: No feasible live-capacity plan
- **WHEN** the live FULL capacities, configured watermarks, reusable block spans, and host retention budgets have no common feasible seed-family count
- **THEN** preflight fails with the constraining group and capacity inputs instead of starting a formal measurement

### Requirement: Manual seed counts are limited to non-formal runs
The system SHALL allow an explicit seed-family count only for smoke or debug execution and SHALL classify that execution as non-formal.

#### Scenario: Explicit seed count requested
- **WHEN** an operator supplies a seed-family count
- **THEN** the run records the manual source and effective count and cannot report formal acceptance

#### Scenario: Formal count omitted
- **WHEN** a formal run omits a seed-family count
- **THEN** the system derives the count after the startup pressure snapshot without requiring operator capacity input

### Requirement: Formal seeds expose at least 32K reusable tokens
The system SHALL set the reusable seed span to the aligned maximum of 32K tokens and the grid's maximum requested cache length. A formal run SHALL reject a grid whose requests never consume at least 32K reusable seed tokens.

#### Scenario: Grid reaches the minimum reusable span
- **WHEN** a grid has a maximum cache length of at least 32K tokens
- **THEN** the reusable seed span is aligned from the greater of that maximum and 32K

#### Scenario: Grid never reuses 32K
- **WHEN** every grid entry has a cache length below 32K tokens
- **THEN** formal preflight fails because extending only the seed initialization request would not create a reusable 32K prefix

#### Scenario: Commit tail is present
- **WHEN** seed initialization adds tokens needed to commit the reusable prefix
- **THEN** those tail tokens are recorded separately and are excluded from reusable seed working-set capacity

### Requirement: Formal fill uses an aligned 100K unique request
The system SHALL default the formal fill phase to a unique request targeting 100K tokens and aligned to the request/cache contract. The system SHALL exclude the request's one-shot suffix from reusable seed-bank capacity.

#### Scenario: Service admits the formal fill request
- **WHEN** the aligned 100K fill request fits the live service and model request limits
- **THEN** the system uses it to establish FULL capacity pressure and records its effective aligned length

#### Scenario: Service cannot admit the formal fill request
- **WHEN** the aligned 100K fill request exceeds a live model or service limit
- **THEN** formal preflight fails with the violated limit instead of silently shortening the request

### Requirement: Seed initialization uses bounded FIFO rolling supply
The system SHALL initialize seed families through FIFO with a bounded client backlog, submit replacements as requests complete, stop submitting at the derived target, and drain all accepted requests within a bounded seed-phase deadline.

#### Scenario: A seed request completes
- **WHEN** seed initialization has not reached its target and one in-flight request completes
- **THEN** the system immediately submits the next seed request while preserving the configured in-flight bound

#### Scenario: Seed target is reached
- **WHEN** the target number of seed requests has been accepted for submission
- **THEN** the system stops producing new seed requests and waits boundedly for all accepted requests to finish

#### Scenario: Seed initialization is incomplete
- **WHEN** a seed request fails or the seed-phase deadline expires before every target seed finishes successfully
- **THEN** the run fails seed initialization and does not start a formal measurement window

### Requirement: Formal client backlog is independent of active server batch
The system SHALL treat client in-flight depth as submitted, running, or waiting RPCs and SHALL keep it at 16 or more for formal runs. It SHALL validate per-request admission separately from the server scheduler's active execution batch.

#### Scenario: FIFO queues requests beyond active batch
- **WHEN** each pressure request is individually admissible and the FIFO server can queue the configured client backlog
- **THEN** formal preflight permits at least 16 in-flight client requests even when fewer requests can execute concurrently

#### Scenario: Formal client depth is too small
- **WHEN** configured client in-flight depth is below 16
- **THEN** the run is classified as non-formal and cannot report formal acceptance

#### Scenario: Individual pressure request is inadmissible
- **WHEN** any fill, seed, or formal grid request cannot be admitted individually under the live limits
- **THEN** preflight fails with the request shape and violated limit

### Requirement: Full-pressure preload is FIFO-only
The system SHALL accept FIFO scheduling only for this continuous full-pressure preload workflow. It SHALL reject fixed-batch scheduling without padding or tail exceptions, while leaving ordinary non-pressure case execution unchanged.

#### Scenario: FIFO pressure workflow
- **WHEN** the continuous full-pressure workflow is requested with FIFO scheduling
- **THEN** the system proceeds to live-capacity preflight

#### Scenario: Batch scheduling requested
- **WHEN** the continuous full-pressure workflow is requested with fixed-batch or batch-decode scheduling
- **THEN** validation rejects the configuration before pressure requests are submitted

#### Scenario: Ordinary batch case
- **WHEN** fixed-batch scheduling is used outside the continuous full-pressure preload workflow
- **THEN** existing ordinary case behavior remains available

### Requirement: Formal and smoke executions are explicitly classified
The system SHALL report whether an execution is production-formal or non-formal smoke/debug and SHALL record every reason preventing formal classification. Forced synthetic GPU block capacity SHALL prevent formal classification.

#### Scenario: Production-formal configuration
- **WHEN** seed count is automatic, client depth is at least 16, no synthetic test block capacity is forced, and all formal preflight requirements pass
- **THEN** the run is eligible for formal acceptance

#### Scenario: Reduced GPU smoke configuration
- **WHEN** synthetic GPU block capacity, a manual seed count, or a client depth below 16 is used
- **THEN** the run remains executable for smoke/debug purposes and reports each non-formal reason

### Requirement: Formal plans and outcomes are reproducible
The system SHALL persist the raw startup pressure snapshot, owner-pool capacity inputs, watermarks, host budgets, block-token spans, alignment inputs, seed-count source and result, effective lengths, client depth, classification, and completed seed count.

#### Scenario: Automatic plan completes
- **WHEN** an automatically planned run reaches measurement
- **THEN** its artifacts contain enough raw inputs and derived outputs to independently recalculate the effective seed-family count

#### Scenario: Planned and completed seed counts differ
- **WHEN** fewer seed families complete than the derived target
- **THEN** the discrepancy is recorded and the run cannot report formal validity

### Requirement: Bidirectional migration validity requires window-local evidence
The system SHALL evaluate bidirectional migration validity independently from execution validity and general FULL-pressure validity. A valid result SHALL include successful GPU-to-host publication and successful host-to-GPU LOAD publication during the measurement window.

#### Scenario: Both directions publish successfully
- **WHEN** measurement-window evidence contains at least one successful device-to-host publication and at least one successful host-to-device LOAD publication
- **THEN** the run may report bidirectional migration valid if its other formal gates pass

#### Scenario: Only device-to-host traffic is observed
- **WHEN** the measurement window contains successful device-to-host publication but no successful host-to-device LOAD publication
- **THEN** bidirectional migration is invalid even if execution and FULL-pressure checks pass

#### Scenario: Transfer occurred outside measurement
- **WHEN** a required transfer direction appears only before or after the measurement window
- **THEN** that transfer does not satisfy bidirectional migration validity

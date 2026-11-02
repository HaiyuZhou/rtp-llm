# Spec Delta

## Purpose

Define a single FIFO execution contract for continuous cache pressure, eliminating fixed-batch pressure behavior while keeping ordinary non-continuous batch measurement available.

## ADDED Requirements

### Requirement: Continuous pressure is FIFO-only

Every continuous cache-pressure run SHALL use FIFO scheduling, whether or not full-pressure preload is enabled. An explicit non-FIFO scheduler SHALL be rejected before requests are submitted.

#### Scenario: Default continuous invocation
- **WHEN** a caller selects continuous cache pressure without a scheduler override
- **THEN** the resolved scheduler is FIFO

#### Scenario: Explicit non-FIFO invocation
- **WHEN** a caller combines continuous cache pressure with a fixed-batch or batch-decode scheduler
- **THEN** startup fails with a FIFO-only migration error before service traffic begins

#### Scenario: Direct runner construction
- **WHEN** a direct runner caller selects continuous mode with a non-FIFO scheduler
- **THEN** construction fails instead of bypassing public CLI validation

### Requirement: Continuous pressure has no fixed-batch lifecycle

Continuous pressure SHALL NOT switch scheduler batch size, wait for fixed-batch barriers, pad a tail batch, or derive client concurrency from a pressure batch size. FIFO backlog and request deadlines SHALL remain bounded by the continuous configuration.

#### Scenario: Finite continuous tail
- **WHEN** a finite continuous run has a request count that is not divisible by any historical batch size
- **THEN** it submits exactly the requested workload with no padding requests

#### Scenario: Seed initialization
- **WHEN** continuous seed families are initialized
- **THEN** they use bounded FIFO supply without changing server scheduler batch size

#### Scenario: Automatic client concurrency
- **WHEN** continuous client concurrency uses its automatic sentinel
- **THEN** it resolves from the FIFO/service concurrency contract and not from a fixed pressure batch size

### Requirement: FIFO semantics remain auditable

Continuous manifests, resume configuration, metrics, and summaries SHALL identify FIFO scheduling. Any compatibility fields retained for scheduler batch size or padding SHALL be neutral and SHALL NOT affect request construction, execution, validity, or resume compatibility.

#### Scenario: Continuous result serialization
- **WHEN** a continuous run writes its checkpoint and summary
- **THEN** the artifacts identify FIFO and report no effective scheduler batch size or padding work

#### Scenario: Resume old non-FIFO run
- **WHEN** a checkpoint records non-FIFO continuous scheduling
- **THEN** resume rejects it with a compatibility error instead of translating it silently

### Requirement: Ordinary batch execution is preserved

Non-continuous case execution SHALL retain its existing fixed-batch and batch-decode scheduler behavior. FIFO-only validation SHALL be scoped to continuous pressure and SHALL NOT disable ordinary cache-grid batch measurement.

#### Scenario: Ordinary fixed-batch case
- **WHEN** a caller runs cache-grid case mode with the batch-decode scheduler
- **THEN** existing scheduler controls, batching, measurement, and result semantics remain available

#### Scenario: Existing non-pressure regression
- **WHEN** ordinary grouped or fixed-batch cache-grid tests run after the change
- **THEN** they observe no continuous-pressure validation or FIFO coercion

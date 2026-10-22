# Spec Delta

## Purpose

Provide auditable, on-demand cache pressure evidence across ranks, groups, and storage tiers without making ordinary health checks scan cache contents.

## ADDED Requirements

### Requirement: Opt-in diagnostic snapshots
The service SHALL expose structured cache pressure snapshots only on explicit diagnostic requests. Ordinary health and cache-status requests SHALL retain their existing response semantics and SHALL NOT collect diagnostic cache contents or step histories.

#### Scenario: Ordinary health probe
- **WHEN** a caller requests worker health without diagnostics
- **THEN** no diagnostic snapshot or cache-key scan is added to the probe

#### Scenario: Unsupported diagnostics
- **WHEN** pressure diagnostics are requested from an unsupported backend or service version
- **THEN** the caller receives an explicit capability error instead of treating absent fields as zero occupancy

### Requirement: Physical pool and logical group identity
Each snapshot SHALL identify the service instance, rank, sequence, server monotonic timestamp, group types, tiers, and physical pool mappings. It SHALL expose capacity, used/free/available blocks, pending eviction releases, configured watermarks, trigger evidence, and available reference categories. Shared physical pools SHALL be counted once while retaining all logical mappings.

#### Scenario: Shared device pool
- **WHEN** two groups refer to the same physical device pool
- **THEN** the snapshot has one physical occupancy record and both group mappings

#### Scenario: Evictable occupied blocks
- **WHEN** cached blocks remain occupied but are available for eviction
- **THEN** used/free and available are reported separately and overlapping reference counts are not summed into used

### Requirement: Published host residency
Host residency SHALL count distinct, successfully published and still-valid host cache blocks, including valid host sources of in-flight migrations. It SHALL exclude unpublished targets, detached data, and mere reservations. Residency SHALL NOT be inferred by subtracting overlapping reference counts.

#### Scenario: Reserved demotion target
- **WHEN** a host destination has been allocated but its data has not been published
- **THEN** physical used blocks can increase while published host residency does not

#### Scenario: Valid source during reload
- **WHEN** a published host block remains valid during a reload
- **THEN** it remains resident until its cache ownership becomes invalid or is released

#### Scenario: Detached destination
- **WHEN** a copy succeeds after its destination has been detached
- **THEN** the destination does not count as published host residency

### Requirement: Explicit cache metadata ownership
Diagnostics SHALL distinguish independent rank-local metadata from supported shared TP block addressing with one owner. A shared declaration SHALL identify the owner rank, complete participant ranks, and all-participant publication-completion scope derived from actual cache/transfer configuration. Unknown or inconsistent ownership SHALL fail explicitly.

#### Scenario: Invalid ownership declaration
- **WHEN** ownership is absent, unknown, or inconsistent with the actual shared-address and transfer configuration
- **THEN** shared pressure is unsupported and no owner is inferred from zero counters or TP size

### Requirement: Frozen ownership provenance
The collector SHALL freeze owner/participant service instances and compatible pool/group/tag/geometry/capacity mappings. Raw local allocator/reference/counter records SHALL remain unchanged. Derived evidence SHALL name its owner and publication scope rather than relabel owner values as peer-local measurements.

#### Scenario: TP peer has no mutable cache tree
- **WHEN** a peer holds physical pools and executes shared block IDs but its local allocator counters are zero
- **THEN** those raw zero counters are retained, and only an explicit validated shared-ownership declaration permits owner logical slot and publication evidence to apply to that participant

#### Scenario: Partial participant transfer
- **WHEN** any declared participant fails a block transfer or its response is absent
- **THEN** that block cannot advance owner publication certified for all participants, even if other participants copied it successfully

#### Scenario: Incompatible participant layout
- **WHEN** a participant's group mappings, tags, block geometry, capacities, identity, or declared ownership differ from the frozen compatible layout
- **THEN** shared pressure cannot qualify and the discrepancy remains in raw evidence

#### Scenario: Missing peer with healthy owner
- **WHEN** the metadata owner reports pressure but a declared participant is missing or stale
- **THEN** owner evidence cannot replace that participant's freshness or identity evidence and the interval remains unknown

### Requirement: Distinct transfer and publication counters
Diagnostics SHALL expose cumulative successful copy and successful publication blocks/bytes separately, classified by rank, group/pool, operation, and source/target tiers, with failure counters. Publication counters SHALL advance only after successful cache settlement. Counters SHALL be available independently of external metrics reporting.

#### Scenario: Copy without publication
- **WHEN** a GPU-to-host copy completes but its target is discarded before publication
- **THEN** successful copy counters advance and successful GPU-to-host publication counters do not

#### Scenario: Publication followed by eviction
- **WHEN** successfully published host data is later removed
- **THEN** cumulative publication remains recorded while current residency decreases

### Requirement: Complete rank evidence and freshness
Collection SHALL retain requested and returned rank identities, per-rank timestamps, sequence numbers, query timing, and sampling skew. Missing ranks, stale samples, instance changes, and counter resets SHALL be explicit. Independent rank responses SHALL NOT be represented as an atomic all-rank snapshot.

#### Scenario: One rank does not respond
- **WHEN** another rank reports healthy pressure but a required rank times out
- **THEN** that rank's interval is unknown and cannot be replaced by the responding rank's values

#### Scenario: Restart resets counters
- **WHEN** a service instance changes or a cumulative counter decreases
- **THEN** the collector records a discontinuity and does not compute progress across the boundary

### Requirement: Diagnostic cost and activity evidence
Snapshots SHALL report available scheduler waiting/running activity and diagnostic sampling timing. Collection SHALL use bounded work and SHALL NOT require full cache-key traversal on every pressure poll. Missing optional activity fields SHALL be distinguished from measured zero activity.

#### Scenario: Missing scheduler counts
- **WHEN** a backend cannot provide waiting/running counts
- **THEN** the response marks them unavailable and the diagnostic report uses other execution and supply evidence without inventing zero counts

#### Scenario: Slow query
- **WHEN** snapshot collection exceeds the allowed freshness budget
- **THEN** its query cost remains visible and the sample cannot establish fresh pressure coverage

### Requirement: Waiting request accounting
Available waiting counts SHALL include local waiting and asynchronous-cache-loading entries and exclude running entries. Supported schedulers SHALL read bounded queue sizes under existing synchronization. Unsupported scheduler/grouped states SHALL remain unavailable rather than returning an invented zero.

#### Scenario: Supported local waiting queues
- **WHEN** a supported scheduler has waiting or asynchronous-cache-loading requests
- **THEN** their local count is exposed without scanning requests or including running entries

#### Scenario: Unsupported grouped queue state
- **WHEN** the scheduler cannot provide the defined bounded count for its current grouped state
- **THEN** waiting activity is unavailable rather than a fabricated zero

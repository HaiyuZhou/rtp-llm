# Spec Delta

## MODIFIED Requirements

### Requirement: AOne prefill shards have exclusive ownership
The system SHALL partition all planned prefill cases into non-overlapping AOne shards using estimated case cost and SHALL run each shard with one model startup when the derived gang machine count is one.

#### Scenario: Multi-shard plan
- **WHEN** an AOne prefill run uses multiple shards with a derived machine count of one
- **THEN** every planned case belongs to exactly one shard and the shard plans collectively cover the full workload

#### Scenario: Unequal case cost
- **WHEN** planned cases have different estimated costs
- **THEN** sharding balances estimated work rather than only the number of cases

#### Scenario: Gang mode bypasses case sharding
- **WHEN** an AOne prefill run derives a gang machine count above one
- **THEN** no case partitioning happens and the whole plan executes as one gang

## ADDED Requirements

### Requirement: AOne gang prefill executes one distributed model
The system SHALL execute the entire prefill plan on the master member of a gang against one externally managed distributed server and SHALL produce the standard prefill result envelope, reports, and reuse-correctness behavior of a local run.

#### Scenario: Gang prefill run
- **WHEN** a prefill config selects the AOne backend with a gang machine count above one
- **THEN** the master member executes every planned case against the gang's distributed server and observed reuse remains a correctness condition

#### Scenario: Gang prefill results match local parity
- **WHEN** a gang prefill run completes
- **THEN** the collected run directory contains the same versioned envelope and report artifacts a local run of the same config would produce, with backend recorded as AOne

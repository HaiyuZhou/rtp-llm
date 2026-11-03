# Spec Delta

## ADDED Requirements

### Requirement: Prefill batch throughput is based on observed rounds
The system SHALL calculate prefill throughput from each complete measured round's elapsed time and planned compute tokens across its concurrent members, while retaining per-request first-token latency as a separate metric.

#### Scenario: Successful concurrent batch
- **WHEN** a prefill round of batch size four completes with four valid member responses
- **THEN** its throughput is `4 * compute_length * 1000 / round_elapsed_ms`, not `4 * compute_length * 1000 / median_request_latency_ms`

#### Scenario: Multiple measured rounds
- **WHEN** a prefill case has multiple valid measured rounds
- **THEN** the case throughput is the arithmetic mean of their measured round throughputs, and the persisted measurements identify each round's elapsed time and member responses

#### Scenario: Invalid round duration or missing member
- **WHEN** a measured round has a non-positive or non-finite elapsed time or lacks a valid response from any planned member
- **THEN** the case does not produce a successful throughput value and formal reporting excludes it

#### Scenario: Single-member batch
- **WHEN** a prefill case has batch size one
- **THEN** its throughput uses that request's measured round elapsed time under the same contract, and its latency remains the server-reported first-token value

#### Scenario: Historical result comparison
- **WHEN** a consumer compares a corrected prefill result with a result produced under the former latency-derived throughput contract
- **THEN** the case identity distinguishes the measurement contracts, and regression comparison fails rather than silently matching the old throughput baseline

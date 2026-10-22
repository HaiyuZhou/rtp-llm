# Spec Delta

## Purpose

Measure actual prefill execution steps and capture uninterrupted pressure profiles while distinguishing foreground CPU work, waits, and exposed GPU gaps.

## ADDED Requirements

### Requirement: Execution step identity and request membership
Diagnostics SHALL identify each actual nonempty batch forward with a session-scoped step ID, rank, monotonic boundaries, context/decode batch sizes, executed tokens, and server-derived request membership. Empty iterations SHALL NOT count as execution steps. Pure-prefill qualification SHALL require context batch greater than zero and decode batch equal to zero.

#### Scenario: Several requests form one batch
- **WHEN** multiple client requests execute in one forward
- **THEN** they map to one execution step rather than one step per request

#### Scenario: Unexpected decode
- **WHEN** a measured execution step contains decode work
- **THEN** the run records a pure-prefill violation rather than including it silently

### Requirement: Complete window-scoped step pairs
Formal measurement step distributions SHALL include only complete steps inside the server-mapped measurement window whose members are all measurement requests. Inter-step samples SHALL require two adjacent qualifying steps. Mixed-phase batches, padding, support, and profile boundary exclusions SHALL retain explicit categories and reasons in raw evidence.

#### Scenario: Mixed measurement and support batch
- **WHEN** a batch contains requests from both measurement and support
- **THEN** it is retained as mixed and excluded from formal measurement step distributions

#### Scenario: Nonadjacent qualifying steps
- **WHEN** an excluded step lies between two otherwise qualifying steps
- **THEN** their combined gap is not reported as one adjacent-step overhead sample

#### Scenario: Support-only profile step
- **WHEN** a complete profile step contains only support requests
- **THEN** it can be analyzed separately for structure but cannot enter measurement distributions

### Requirement: Separate timing meanings
Diagnostics SHALL distinguish executor-thread forward-to-forward wall gaps, same-thread CPU-clock deltas, and exposed GPU gaps derived from correlated GPU activity. They SHALL NOT equate CPU forward return with GPU completion, subtract GPU duration from wall time to invent CPU cost, or introduce per-step GPU synchronization for measurement.

#### Scenario: CPU preparation overlaps GPU execution
- **WHEN** the CPU prepares the next batch while the previous batch is still executing
- **THEN** overlapping preparation is not automatically classified as exposed GPU idle time

#### Scenario: Thread is descheduled
- **WHEN** executor wall time increases without comparable thread CPU time
- **THEN** the difference is reported as non-on-CPU time without claiming a precise cause unless scheduling evidence supports it

### Requirement: Lightweight baseline statistics
Baseline runs SHALL record bounded lightweight step evidence and sample counts, wall/thread-CPU distributions, batch/token distributions, and known wait intervals. Background-thread costs SHALL remain separately attributable. Tail statistics from insufficient samples SHALL be labeled insufficient rather than presented as representative p99 conclusions.

#### Scenario: Short trace sample
- **WHEN** only a short 20-step profile is available
- **THEN** the report uses it for structural diagnosis and does not assert a representative p99 tail from it

### Requirement: Independent continuous profile control
Baseline and profile SHALL run separately and independently establish seed/fill/steady pressure. Profile start, stop, and artifact waiting SHALL not occupy request-sending workers or interrupt supply. Per-rank control and trace boundaries SHALL be recorded without treating all-rank acknowledgement as a synchronized execution barrier.

#### Scenario: Slow profile acknowledgement
- **WHEN** the profile control request or trace-file wait is delayed
- **THEN** request supply continues and the delay remains visible in control timing

#### Scenario: Different rank start times
- **WHEN** ranks begin recording at different execution boundaries
- **THEN** actual rank windows and pressure evidence are reported instead of assuming identical starts

#### Scenario: Large Kineto artifacts
- **WHEN** completed trace files require expensive JSON parsing
- **THEN** online artifact waiting avoids parsing, and analysis runs after drain in a terminable child within the original capture deadline, retaining raw files on timeout without extending measurement or support

### Requirement: Multi-step Kineto collection
Continuous Kineto SHALL default to 20 actual nonempty execution steps on every target rank. Validation SHALL require complete parseable traces, GPU kernels, CPU ranges, CUDA correlation, and usable internal adjacent-step evidence. Startup and flush boundaries SHALL be excluded, with requested, observed, and usable step/interval counts reported separately.

#### Scenario: Nonempty but unusable trace
- **WHEN** a rank emits nonempty JSON without GPU activity or usable step correlation
- **THEN** trace_valid is false and the artifact is retained with a reason

#### Scenario: Boundary preprocessing missing
- **WHEN** collection starts after the first step's input preparation
- **THEN** that incomplete boundary is excluded and 20 collected steps are not claimed as 20 usable intervals

#### Scenario: Missing rank trace
- **WHEN** one target rank's trace is absent at the bounded deadline
- **THEN** capture fails explicitly, collection is cleaned up, and available rank traces are retained

#### Scenario: Correlated GPU work outside model forward
- **WHEN** sampler or asynchronous cache work has a unique CUDA runtime correlation whose launch is outside every forward and execution-step scope
- **THEN** it is retained as separate non-forward activity, while missing or ambiguous correlations and forwards without correctly associated GPU work still invalidate the trace

### Requirement: Bounded Nsight capture and visible markers
Continuous Nsight SHALL use an explicitly configured bounded time window and validate actual execution-step coverage afterward. Step and critical-stage markers SHALL be visible to that backend. Stop/export SHALL finish or fail explicitly before service destruction. Kineto and Nsight SHALL not collect CUDA activity simultaneously.

#### Scenario: Time window includes fewer steps than expected
- **WHEN** an Nsight window contains few actual forwards despite many completed requests
- **THEN** the report uses the observed step count and does not substitute request count

#### Scenario: Slow Nsight start acknowledgement
- **WHEN** starting collection consumes noticeable command setup time
- **THEN** the configured collection duration starts after successful acknowledgement, while pressure uses the conservative pre-start boundary and the original capture/support deadlines still apply

#### Scenario: Stop or export fails
- **WHEN** Nsight cannot complete stop/export
- **THEN** cleanup records the failure and preserves partial artifacts without reporting a complete trace

#### Scenario: Production-size analysis exceeds a short allowance
- **WHEN** Nsight normalization or serialization requires more than a fixed short cleanup allowance
- **THEN** post-drain analysis uses only the original capture deadline's remaining budget in a terminable and reaped process, retaining raw and explicitly partial artifacts on timeout without extending measurement, support or capture

#### Scenario: Mixed-clock NVTX capture boundary
- **WHEN** NVTX boundary ranges fit only the absolute capture interval established by unique consistent session and analysis clock metadata
- **THEN** normalization retains the raw values and provenance, validates CUDA association and excludes repaired boundary steps from usable pairs; internal clock anomalies, inconsistent metadata and ambiguous or out-of-range values remain invalid

### Requirement: Profile result isolation and compatibility
Continuous profiles SHALL use the full pressure workflow and retain diagnostic-only results, separate from baseline and exact-reuse fitting. Existing single-case profiling/retest behavior SHALL remain unchanged. Capture failure SHALL not erase request/cache evidence or invalidate a separate successful baseline run.

#### Scenario: Existing case profile
- **WHEN** a user selects ordinary case profiling without continuous full-pressure mode
- **THEN** the existing case selection, seed/replay, and artifact behavior are preserved

#### Scenario: Continuous profile invocation
- **WHEN** a user selects continuous full-pressure profiling
- **THEN** the run does not require a single replay case and does not substitute a one-request replay for the sustained workflow

# Spec Delta

## Purpose

Establish and measure sustained FULL-attention cache pressure using real prefill requests, continuous bounded supply, and auditable diagnostic results separate from exact-reuse benchmarks.

## ADDED Requirements

### Requirement: Explicit opt-in and frozen configuration
Full-pressure diagnostics SHALL require explicit opt-in and retain DP=1, exclusive service, ungrouped batch_size=1 grid, input-ID transport, known block size, and complete commit-tail constraints. Fixed workspace, materialized inputs, and shared-seed mode SHALL remain incompatible. All thresholds, budgets, target mappings, and stop policies SHALL resolve and freeze before execution.

#### Scenario: Unsupported combination
- **WHEN** diagnostics are requested with DP greater than one or materialized inputs
- **THEN** execution fails with a configuration error before sending workload requests

#### Scenario: Existing continuous invocation
- **WHEN** continuous mode is invoked without full-pressure opt-in
- **THEN** its finite request count, scheduler, padding, reuse validation, and output behavior are preserved

#### Scenario: Missing required experiment settings
- **WHEN** fill budgets, GPU tolerance, or a requested Nsight duration cannot resolve to explicit valid settings
- **THEN** startup fails rather than choosing or changing those values during the run

### Requirement: FULL-only pressure targets
The run SHALL require every target TP rank's FULL groups to pass independently. SWA/LINEAR SHALL be observed without capacity gates. Physical pools SHALL be deduplicated while retaining group mappings. Models without FULL groups, missing required host tiers, and known unreachable targets SHALL fail explicitly; targets SHALL NOT be removed to obtain a pass.

#### Scenario: Other attention groups are below watermarks
- **WHEN** all required FULL groups meet pressure conditions while SWA/LINEAR occupancy is low
- **THEN** FULL capacity pressure passes and the result does not claim all-group saturation

#### Scenario: One FULL rank fails
- **WHEN** one required rank/group is below its target while others exceed theirs
- **THEN** cross-rank or cross-group averaging cannot make the pressure check pass

#### Scenario: No FULL group
- **WHEN** the model has no FULL attention cache group
- **THEN** the run reports the criterion as unsupported and does not substitute another attention type

### Requirement: Shared-owner FULL evaluation
For explicitly declared shared TP ownership, each rank/group SHALL retain its own freshness, identity, topology and capacity validation while using named owner slot, pending-release and publication evidence under the all-participant completion contract. The run SHALL NOT require replicated mutable trees, infer ownership from zero counters, or substitute owner responses for missing peers.

#### Scenario: Shared owner with complete participants
- **WHEN** owner pressure qualifies, every declared participant has a fresh compatible physical layout and stable identity, and publication requires successful completion on all participants
- **THEN** each required participant group is evaluated against its validated capacity using explicitly attributed owner evidence while raw peer-local allocator counters remain unchanged

#### Scenario: Shared owner without complete scope
- **WHEN** the owner, a participant, or the all-participant publication scope is absent or inconsistent
- **THEN** shared pressure cannot qualify even when the available owner counters exceed all thresholds

### Requirement: Shared-slot admission accounting
Preflight/admission budgets and frozen result provenance SHALL use the same declared ownership model as pressure evaluation, without counting shared slots multiple times.

#### Scenario: Shared slots across physical TP pools
- **WHEN** multiple participant pools use one owner's allocated block IDs
- **THEN** admission does not add their local allocator availability as independently allocatable shared slots

### Requirement: Group watermark and residency gates
Each required FULL group SHALL have current-epoch DEVICE high-watermark-trigger evidence and positive successful GPU-to-host publication delta. GPU pressure SHALL use effective_used = used - pending_release. Stable pressure requires at least one associated device pool at or above floor(capacity * low_ratio) minus frozen tolerance. Published host residency SHALL default to at least ceil(host_capacity * 0.25).

#### Scenario: One pool triggers the group
- **WHEN** one device pool reaches ceil(capacity * high_ratio) while another associated pool does not
- **THEN** the real group trigger qualifies without requiring every pool to reach its high watermark

#### Scenario: Pending releases or over-high occupancy
- **WHEN** pending releases reduce effective occupancy below all low-watermark thresholds
- **THEN** the group fails the stable GPU gate even if physical used remains high
- **AND** occupancy above the high watermark is reported separately and does not itself mean lost pressure

#### Scenario: Host allocation without publication
- **WHEN** host reservations reach 25% but published residency or successful publication delta is insufficient
- **THEN** the capacity gate remains unsatisfied

### Requirement: DEVICE-specific watermark evidence
The existing watermark_trigger_count SHALL count DEVICE transitions only; HOST and DISK triggers SHALL NOT qualify GPU pressure. Successful STORE and EVICT publications SHALL both remain eligible after a DEVICE trigger.

#### Scenario: Non-device trigger with STORE publication
- **WHEN** only HOST or DISK has crossed its high watermark and a GPU-to-host STORE has published
- **THEN** the group still lacks the required DEVICE-trigger evidence
- **AND** the same STORE publication remains eligible once an actual DEVICE trigger occurs

### Requirement: Bounded real-request fill
Capacity fill SHALL use unique inputs isolated from formal seeds, real prefill execution with one output token, and natural cache publication/eviction. It SHALL obey model, admission, token, and KV limits and frozen time/request/no-progress budgets. Completed request or estimated block counts SHALL NOT replace actual pressure evidence.

#### Scenario: All-cold workload
- **WHEN** the formal grid requires no shared prefix seeds
- **THEN** fill still runs and must establish the same FULL pressure evidence

#### Scenario: Fill budget exhausted
- **WHEN** pressure is not reached before a configured fill limit
- **THEN** the run ends boundedly with pressure_not_reached and preserves samples and failure-stage evidence

### Requirement: Continuous bounded phase transitions
Steady, measurement, and profile SHALL share uninterrupted supply without restart, cache clearing, or an in-flight drain at transitions. Submission buffers SHALL be bounded and inputs generated on demand. Arrival schedules SHALL advance continuously across phases, retaining rate-limit lag. Fixed batching SHALL pad its final tail; FIFO SHALL not pad or wait for a configured fixed batch.

#### Scenario: Switch to measurement under load
- **WHEN** steady qualifies with requests still in flight
- **THEN** new sends receive measurement labels without draining the sender or resetting scheduled arrival times

#### Scenario: Long time-window experiment
- **WHEN** the grid repeats for the measurement duration
- **THEN** queued work remains bounded and sequence, epoch, nonce, and family assignments are auditable

#### Scenario: Scheduler-specific termination
- **WHEN** the final submitted request count is not a multiple of the fixed batch size
- **THEN** fixed-batch mode sends the required tail padding and treats padding failure as invalid execution
- **AND** FIFO mode sends no padding requests

### Requirement: Steady readiness and bounded waiting
Steady SHALL default to 3 continuous seconds of valid FULL pressure coverage and a total wait limit of 60 seconds from entry into the formal distribution. Capacity validity SHALL use time intervals, not sample count. Supply/progress checks SHALL consider execution activity and SHALL NOT equate a long active prefill with client starvation solely because completions pause.

#### Scenario: Three samples do not span three seconds
- **WHEN** valid samples arrive at seconds 0, 1, and 2
- **THEN** they do not qualify a 3-second steady window

#### Scenario: Steady timeout
- **WHEN** continuous qualifying coverage is not reached within 60 seconds under default settings
- **THEN** the run reports pressure_not_reached and performs bounded cleanup without valid measurement output

### Requirement: Conservative time-based pressure coverage
Sampling SHALL default to 1 second with maximum sample gap and age of 2 seconds. Only intervals between adjacent fresh, valid samples qualify. Coverage SHALL intersect every required rank/group's intervals on a common timeline, clip to the measurement window, and count unknown time as invalid. Default acceptance SHALL require at least 95% valid duration and at most 2 seconds of consecutive invalid/unknown time.

#### Scenario: Stale sample or uncovered edge
- **WHEN** a sample is missing, stale, discontinuous, or cannot cover a window edge
- **THEN** the unsupported time is unknown and no last-value carry-forward makes it valid

#### Scenario: Average coverage hides a long loss
- **WHEN** a default 60-second window has 57 valid seconds and one continuous 3-second invalid interval
- **THEN** it fails the longest-invalid-interval limit despite reaching 95% coverage

#### Scenario: Cross-rank skew
- **WHEN** required ranks have different valid time intervals or uncertain clock alignment
- **THEN** only their conservatively mapped intersection counts and the report states sampling-scale validity

#### Scenario: Pressure lost during measurement
- **WHEN** FULL pressure falls below a gate
- **THEN** a pressure_lost event is retained, no fill requests are secretly injected, and window validity follows the frozen coverage and failure rules

### Requirement: Shared-owner clock provenance
Shared-owner counter intervals SHALL retain the owner's timestamps and sequences and intersect all participants' freshness/layout-valid intervals. Owner observations SHALL NOT be re-stamped onto peer clocks or extended beyond the owner's own valid interval.

#### Scenario: Peer interval extends past owner evidence
- **WHEN** a fresh peer interval begins earlier or ends later than the owner's confirmed pressure interval
- **THEN** that extra time cannot become valid shared pressure through timestamp substitution

### Requirement: Fixed measurement window and request attribution
The default measurement SHALL last 60 seconds using [t0, t1), mutually exclusive with request-count stopping. Requests SHALL belong to their actual dispatch phase. Measurement requests finishing after t1 SHALL retain membership with a crossing flag. Throughput SHALL count only measurement-member completions inside the window divided by its duration; support/padding and drain completions SHALL be excluded.

#### Scenario: Queued before a boundary but dispatched after it
- **WHEN** a queued job is actually sent after t1
- **THEN** it cannot become a measurement request because of its enqueue time

#### Scenario: Late completion
- **WHEN** a measurement request completes during drain
- **THEN** its latency and outcome remain in the measurement cohort but its completion does not increase window throughput

### Requirement: Independent validity and reproducible evidence
Diagnostic results SHALL separate execution_status, per-window pressure_valid, trace_valid, reuse_exact, and diagnostic_only. They SHALL retain frozen configuration, code/profile/grid identity, actual input identity, service/epoch/rank timing, raw request/cache/step evidence, and failure reasons. They SHALL NOT enter exact-reuse fitting datasets or claim that a shuffle seed alone reproduces UUID-based inputs.

#### Scenario: Seed evicted by genuine pressure
- **WHEN** a valid-shaped request succeeds with less reuse than requested
- **THEN** actual reuse is reported without automatically making execution fail, and the result remains diagnostic-only

#### Scenario: Trace failure
- **WHEN** profiling fails after requests and pressure samples have been recorded
- **THEN** trace validity fails independently and collected evidence and any separate baseline result are retained

### Requirement: Explicit cumulative diagnostic scope
Coverage assessments SHALL name cumulative diagnostics `epoch_reasons`, explicitly covering the whole epoch so far, including fill. They SHALL NOT present those reasons as window-specific failures; per-window acceptance SHALL use clipped coverage, longest-invalid duration, and epoch continuity.

#### Scenario: Clean window after fill pressure loss
- **WHEN** fill recorded pressure loss but the later measurement window has complete valid coverage
- **THEN** epoch_reasons retains that history without labeling it a measurement-window failure

### Requirement: Latest-started epoch discoverability
The latest-started epoch pointer SHALL be published atomically at epoch creation, and preflight failures SHALL retain evidence in the newly referenced directory.

#### Scenario: New epoch fails preflight
- **WHEN** a newly created epoch fails checkpoint or initial RPC preflight
- **THEN** the latest pointer identifies that epoch and its directory retains the failure evidence

### Requirement: Bounded support, drain, and resume
At t1 the sender SHALL stop new measurement requests. Unfinished profiling SHALL receive same-distribution support for at most 60 additional seconds by default, subject to earlier capture deadlines. Support pressure SHALL be evaluated separately. Cleanup SHALL stop collection and finish or boundedly cancel in-flight requests. Resume SHALL establish a new epoch and rebuild pressure instead of joining discontinuous timelines.

#### Scenario: Capture needs more steps
- **WHEN** the measurement window closes before profiling completes
- **THEN** support continues with distinct labels and pressure evidence, without extending measurement statistics

#### Scenario: Support deadline expires
- **WHEN** capture has not completed by the support deadline
- **THEN** collection is explicitly stopped, partial artifacts retained, and requests drained or boundedly cancelled

#### Scenario: Resume from disk checkpoint
- **WHEN** a stopped experiment resumes
- **THEN** seed/fill/steady are re-established in a new session/epoch and prior cache pressure is not assumed to survive

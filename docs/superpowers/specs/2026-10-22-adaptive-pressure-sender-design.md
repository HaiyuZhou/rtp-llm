# Adaptive Continuous-Pressure Sender Design

## Goal

Make the tiered rolling pressure workload dispatch its predetermined request
sequence from a timer instead of replenishing one request for each completed
request. The sender should start quickly, reduce its offered rate as unfinished
RPCs accumulate, and fail explicitly rather than silently becoming
completion-driven when its safety capacity is exhausted.

## Scope

This change applies to the FIFO continuous full-pressure preload workflow. It
does not change the seed or fill phases, ordinary finite continuous execution,
the FIFO scheduler, or cache migration behavior.

## Configuration

The formal request stream separates four independent controls:

- `family_schedule` and `cohort_size` determine the request identity sequence.
- `request_rate_max` is the initial and low-backlog offered rate.
- `request_rate_min` is the non-zero maintenance rate at high backlog.
- `backlog_low`, `backlog_high`, and `max_outstanding` control adaptive
  backpressure and the hard safety boundary.

For compatibility, the existing fixed constant and Poisson arrival modes retain
their current behavior when adaptive rate control is not selected. Tiered
rolling adaptive runs use constant arrival and require positive finite minimum and maximum rates,
`request_rate_min <= request_rate_max`, and
`0 <= backlog_low < backlog_high < max_outstanding`.

The tiered rolling cohort size is independent of `max_outstanding`. A cohort
continues to contain equal device and memory halves, but it is only a logical
sequence grouping and never a dispatch or completion barrier.

## Rate Control

Let `q` be the number of submitted RPCs without a collected completion. The
effective rate is:

```text
q <= backlog_low:
    rate = request_rate_max

backlog_low < q < backlog_high:
    rate = request_rate_min
         + (request_rate_max - request_rate_min)
         * (backlog_high - q) / (backlog_high - backlog_low)

q >= backlog_high:
    rate = request_rate_min
```

The next absolute due time advances by `1 / rate` after each accepted request.
It is never reset to the current time, so supervisor delays remain visible as
schedule lag. Completion callbacks may wake the supervisor for collection but
may not directly submit or schedule another request.

If a request becomes due while `q == max_outstanding`, the run records a
`sender_saturated` failure and drains. It does not wait for a completion,
retry the same due request, or skip to a later sequence item. This makes silent
closed-loop degradation impossible in a successful run.

## Sequence Semantics

A single supervisor owns sequence allocation and dispatch. The next
`TieredRollingAssignment` is consumed only when its RPC is accepted by the
bounded sender. Rate adaptation changes timing but not identity order. Every
consecutive logical cohort therefore retains its configured device/memory
composition, and the previous cohort's memory assignments remain the next
cohort's promoted device assignments.

Expected tiers remain workload intent, not a completion barrier. Actual cache
residency and transfer counters remain the acceptance evidence.

## Observability And Acceptance

Each formal dispatch records:

- sequence and tiered assignment metadata;
- scheduled and actual dispatch timestamps and schedule lag;
- outstanding count before dispatch;
- effective rate used to schedule the next request.

The session summary records maximum outstanding, minimum/maximum/average
effective rate, backlog-region dwell counts, and saturation status. A formal
profile is invalid if the sender saturates, drops, or reorders a sequence item.

Enqueue timestamps must remain available in trace analysis. A scheduler wake
caused by a new enqueue must not be classified as an H2D completion wake.

## Failure And Shutdown

Configuration errors fail before any formal request is sent. Runtime sender
saturation emits its diagnostic event, changes the pressure phase to drain,
cancels outstanding work using the existing bounded drain path, and marks the
run unsuccessful. Other request, diagnostics, and profiling failures retain
their existing behavior.

## Verification

Deterministic fake-clock tests cover:

1. Non-completing requests continue to dispatch at timer-selected intervals
   until the hard outstanding limit.
2. Effective rate decreases monotonically across the interpolation region and
   remains non-zero above the high watermark.
3. Reaching the hard limit fails with `sender_saturated` instead of waiting for
   a completion.
4. Completion changes later rate calculations but never immediately dispatches
   a replacement request.
5. Tiered rolling assignments remain ordered with the expected 16/16 split and
   no cohort completion barrier.
6. Existing constant/Poisson and seed/fill behavior remains unchanged.

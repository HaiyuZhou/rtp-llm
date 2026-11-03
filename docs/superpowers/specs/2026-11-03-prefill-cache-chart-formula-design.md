# Prefill Cache Report Charts and Formula Design

## Intent

Restore the one-command cache-aware prefill deliverable after the unified perf-test migration. A completed `./tools/perf_test --mode=prefill --action=run --config=...` should produce an offline HTML performance chart and a fitted prefill latency formula when the measured data supports one. `--action=report --run-dir=...` should recreate the same derived artifacts without starting a model. Local, AOne shard, and AOne gang runs already call the shared prefill report generator.

## Current state and constraints

- `prefill/report.py` validates committed cases and writes JSON plus a table-only `prefill_report.html`. Its `chart` and `formula_input` entries are metadata, not rendered charts or a fit.
- The former cache-grid pipeline produced offline interactive HTML, formula text, `model.json`, `fit_report.json`, `predictions.csv`, and an error graphic. That pipeline read the retired `cache_grid_results.json` format and cannot be invoked on the unified run directory.
- Current `concurrent_batch_v1` cases contain one server first-token latency per member and one shared client round elapsed time. The case's `latency_ms` is the median of all member latencies; `throughput_tokens_per_second` is based on complete round wall time. These are different measurements.
- The existing OpenSpec requires invalid shape, reuse, and timing cases to be excluded from fitting and charts. Legacy prefill runs remain readable offline but have a different measurement contract.

## Approaches

1. Restore the retired `tools/cache_perf` pipeline with its old result format. This preserves its original command and code but duplicates the unified runner and cannot interpret concurrent batches without a second migration layer.
2. Extend the unified offline report path with chart and fit modules, adapting the historical restricted symbolic library to the current validated observations. This keeps one run command and one result contract. **Proposed choice:** approach 2.

## Behavior

- Every successful prefill `run` writes `reports/prefill_report.html` with links to `reports/latency.interactive.html` and `reports/tpm.interactive.html`. The chart files are self-contained for offline viewing. They support batch filtering and show input, cache reuse, compute, server first-token latency, and measured throughput. When Plotly is unavailable, the files contain a dependency-free SVG chart and clearly state that interaction is unavailable.
- Plot latency for a concurrent case as the median across rounds of each round's **maximum member server first-token latency**. Show member observations separately. For throughput plots, use each round's recorded `round_elapsed_ms`; never derive throughput from server latency. Label the displayed metric and measurement contract.
- Fit `PREFILL_TIME_FORMULA` to the same median-of-round-max server first-token latency. Map a homogeneous batch of size B to B requests with the case's input and cache lengths. The fitted formula covers only these measured workload geometries; the report must not imply validation for heterogeneous batches. Port the historical restricted symbolic feature library and deterministic train/validation/test split, with a held-out error audit and explicit production acceptance flag.
- Write `reports/formula/model.json`, `fit_report.json`, `predictions.csv`, `fit_gap.svg`, and `PREFILL_TIME_FORMULA.txt` when fitting has enough valid, distinct geometry and nonempty train/validation/test splits. A rejected quality gate may retain a candidate formula but sets `production_acceptance=false`. Insufficient data writes an audit and rejection reason, never a deployable formula.
- A partial or invalid run can produce clearly marked diagnostic charts from valid cases, but cannot produce a production-accepted formula. Legacy observations are labeled in the HTML and excluded from a `concurrent_batch_v1` fit; never merge the two measurement contracts into one model.
- A fit quality rejection does not change the measurement run's success status; it is visible in the report and artifacts. Unexpected chart or fit implementation errors remain report failures. Report regeneration replaces only derived report artifacts and never edits case results, checkpoints, accepted baselines, or goldens.

## Implementation boundaries

- Add focused `prefill/chart.py` and `prefill/formula.py`; reuse only the historical restricted symbolic fitting logic that remains compatible with the FlexLB parser. Keep `prefill/report.py` responsible for validation, orchestration, and artifact references.
- Use the existing `plotly` dependency for self-contained interactive HTML and declare it in the Bazel runtime where needed. Keep an SVG fallback so a missing optional renderer does not remove the HTML chart.
- Add integration tests against persisted unified run fixtures: concurrent B>1 timing separation, offline regeneration, partial/legacy labeling, insufficient data, formula acceptance/rejection, and unchanged formal case files. Update the public perf-test reference and example output documentation.

## Open decision for review

The proposed formula is a FlexLB-compatible **candidate** with validation and a separate production acceptance flag. If only an exploratory formula is wanted, the parser compatibility and production gate can be omitted.

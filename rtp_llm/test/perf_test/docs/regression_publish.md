# Regression, golden, and publishing reference

The unified perf test accepts regression and publisher policy only from the
versioned config. `examples/prefill_regression_publish.json` is a complete
strict example. It contains destinations and artifact names, but no credential
values or secret references.

## Regression

`regression.enabled` requires `regression.baseline`. `threshold_pct` is the
allowed relative degradation percentage used when a candidate is generated.
`generate_candidate` writes `reports/baseline_candidate.json`; it never
overwrites the accepted baseline.

A baseline is a versioned object containing stable case and metric entries:

```json
{
  "schema_version": 1,
  "entries": [
    {
      "case_key": "prefill-0123456789abcdef01234567",
      "metric": "latency_ms",
      "value": 10.0,
      "direction": "lower_is_better",
      "threshold": 0.05
    }
  ]
}
```

Threshold boundaries pass. A lower-is-better metric fails only above
`value * (1 + threshold)`; a higher-is-better metric fails only below
`value * (1 - threshold)`. A missing case, missing metric, invalid stable case
key, or partial result fails the regression gate.

## Golden operations

Normal runs never update committed golden files. Generation and validation are
separate Bazel operations that consume a finalized versioned result:

```bash
bazel run //rtp_llm/test/perf_test:generate_perf_test_golden -- \
  --result=perf_test_results/prefill-run/results/result.json \
  --output=rtp_llm/test/perf_test/testdata/prefill_result.golden.json
```

```bash
bazel run //rtp_llm/test/perf_test:validate_perf_test_golden -- \
  --result=perf_test_results/prefill-run/results/result.json \
  --golden=rtp_llm/test/perf_test/testdata/prefill_result.golden.json
```

Validation is read-only. Generation performs an explicit atomic replacement of
the selected golden path.

## OSS archive publisher

`publish.oss.artifacts` is an exact allowlist of run-relative regular files.
Absolute paths, parent traversal, directories, duplicate entries, and missing
files are rejected. Only the listed files enter the archive. `destination` is
the remote prefix; the object name is `<run_id>.tar.gz`.

The runtime environment must provide `MAGA_OSS_KEY_ID` and
`MAGA_OSS_KEY_SECRET`. `OSS_ENDPOINT` is optional. CI must inject their values;
do not place values or secret references in config files, manifests, results,
reports, command arguments, or logs.

## ODPS summary publisher

`publish.odps.destination` names the summary table. ODPS receives only finalized
aggregate fields: run identity, mode/backend/status, fingerprints, model,
case counts, and mode-specific latency/throughput/request aggregates. It never
receives raw cases or measurements. `publish.odps.artifacts` is unused and
should remain empty.

The runtime environment must provide `ALIBABA_CLOUD_ACCESS_KEY_ID` and
`ALIBABA_CLOUD_ACCESS_KEY_SECRET`. `ODPS_PROJECT` and `ODPS_ENDPOINT` are
optional. The same no-secret-in-artifacts rule applies.

## Failure policy

Each enabled publisher records a success or failure outcome in both
`manifest.json` and `results/summary.json`. A failed publisher with
`required: true` returns exit code 6 after recording the failure. A failed
optional publisher preserves measurement success. Neither policy rewrites
`results/result.json` or any per-case result.

# Regression, golden, and publishing reference

The unified perf test accepts regression and publisher policy only from the
versioned config. `examples/prefill_regression_publish.json` is a complete
strict example. It contains destinations and artifact names, but no credential
values or secret references.

Current integration boundary: the config parser accepts these policies and
the regression and publisher helpers are implemented, but the `tools/perf_test`
run entry does not yet call `compare_result_to_baseline`,
`write_baseline_candidate`, or `run_publishers`. The sections below describe
the explicit helper contracts, not automatic post-run actions.

## Regression

`regression.enabled` requires `regression.baseline`. `threshold_pct` is the
allowed relative degradation percentage for a candidate. The config parser
accepts `generate_candidate`, but the current run entry does not invoke the
baseline comparison or candidate writer. Generate a candidate explicitly from
a completed `results/result.json` until those operations are wired into the
run lifecycle; neither operation changes the accepted baseline.

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

`compare_result_to_baseline` applies these rules when called explicitly.
Threshold boundaries pass. A lower-is-better metric fails only above
`value * (1 + threshold)`; a higher-is-better metric fails only below
`value * (1 - threshold)`. A missing case, missing metric, invalid stable case
key, or partial result yields a failed comparison. The current run entry does
not apply that comparison as an automatic gate.

Preview a candidate from a completed result without writing any file:

```bash
python - /path/to/run/results/result.json 0.05 <<'PY'
import json
import sys
from pathlib import Path
from rtp_llm.test.perf_test.regression.candidate import build_baseline_candidate

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
candidate = build_baseline_candidate(result, threshold=float(sys.argv[2]))
print(json.dumps(candidate, indent=2, sort_keys=True))
PY
```

Here `0.05` permits a 5% degradation. Review the output before saving it as a
baseline candidate and deliberately promoting it to the accepted baseline.

## Golden operations

Normal runs never update committed golden files. Generation and validation are
separate Bazel operations that consume a finalized versioned result. Generate
to a candidate file in an existing directory before reviewing or replacing an
accepted golden:

```bash
bazel run //rtp_llm/test/perf_test:generate_perf_test_golden -- \
  --result=perf_test_results/prefill-run/results/result.json \
  --output=/tmp/prefill_result.candidate.golden.json
```

```bash
bazel run //rtp_llm/test/perf_test:validate_perf_test_golden -- \
  --result=perf_test_results/prefill-run/results/result.json \
  --golden=/tmp/prefill_result.candidate.golden.json
```

Validation is read-only. Generation performs an explicit atomic replacement of
the selected output path. Review the candidate diff, then explicitly promote
it to the golden file used by the relevant test target and validate that file.
The output parent directory must already exist.

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

When `run_publishers` is called explicitly, it records success or failure
outcomes in both `manifest.json` and `results/summary.json`. A failed required
publisher raises a publisher error with exit code 6 after recording the failure;
a failed optional publisher preserves measurement success. The current run entry
does not invoke this policy. Neither helper rewrites `results/result.json` or
any per-case result.

## Concurrent prefill baseline migration

New prefill runs use `concurrent_batch_v1`. Plans containing any `batch_size > 1`
require `model.topology.dp == 1`; plans containing only batch size 1 still support
DP > 1. Throughput is calculated from complete concurrent round wall time rather
than server first-token latency. The contract marker changes stable case IDs; an old
baseline case key produces a missing-case comparison failure rather than matching the
new measurement. The common result schema stays at v2.

Create a fresh run, inspect its validated raw round evidence and report, then
explicitly generate a baseline candidate using the command above. Review and
promote that candidate to the accepted baseline deliberately. Use the explicit
`generate_perf_test_golden` operation above only when approving a new golden;
`validate_perf_test_golden` remains read-only and reports migration mismatches.
No run or report automatically rewrites accepted baselines or goldens.

Old directories remain readable offline. Their JSON/HTML observations carry
`legacy_latency_derived_v1`; new observations carry `concurrent_batch_v1`.
Reports exclude inconsistent new-contract rounds and become diagnostic, so use
only a complete validated fresh run to regenerate reference artifacts.

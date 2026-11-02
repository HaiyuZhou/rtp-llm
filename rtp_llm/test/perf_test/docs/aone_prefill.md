# AOne prefill performance tests

AOne CI supports unified prefill `run`. Pass one versioned config file to the
submit-host entry; decode, prefill `resume`, and prefill `profile` are rejected
before build or submission. After collection, `report` runs offline against the
new run directory and does not submit another AOne job.

The checkpoint and tokenizer in `examples/prefill_aone.json` are paths visible
inside the AOne runtime. Model data is not copied into the generated Bazel
package. Adapt those paths, the topology, GPU type, and resource name for the CI
environment.

```bash
./tools/perf_test \
  --mode=prefill \
  --action=run \
  --config=rtp_llm/test/perf_test/examples/prefill_aone.json \
  --dry-run
```

The submit host strictly validates and normalizes the config, expands the full
cache grid, computes config and workload fingerprints, and assigns every case
to one shard with deterministic cost-aware LPT scheduling. Dry-run prints the
shard ownership, estimated costs, GPU resources, generated targets, and build
and submission commands without creating a run directory or remote job.

For execution, each generated Bazel target declares
`resolved_config.json`, `resolved_workload.json`, and `shard_plan.json` as
runfiles and passes their locations to the worker. The source config path is not
used remotely. Each shard starts one model session and returns its run identity,
shard identity, fingerprints, owned and completed case IDs, and atomic case
results.

Collection rejects missing or duplicate shard claims, unknown or out-of-scope
cases, mismatched fingerprints, and completion claims without case files. A
remote measurement failure preserves valid cases and produces a diagnostic
offline report with a nonzero exit. Only complete valid ownership produces a
formal report and the common `results/result.json` envelope with backend
`aone`.

The submit host runs one Bazel build for all generated shard targets, followed
by one remote Bazel test with `execution.max_parallel` as its job limit. The
adapter selects the platform config from `execution.gpu_type` and expands
`execution.resource` to the matching `daily` or `online` cache/remote configs.
`PERF_TEST_BAZEL` can override the `bazelisk` executable;
`PERF_TEST_BAZEL_BUILD_JOBS` controls build concurrency;
`PERF_TEST_BAZEL_STARTUP_ARGS` supplies startup options such as an isolated
`--output_user_root`;
`PERF_TEST_BAZEL_ARGS` supplies additional Bazel command arguments; and
`PERF_TEST_BAZEL_TESTLOGS` overrides the default workspace `bazel-testlogs`
collection root.

When the derived machine count exceeds one, the run becomes one multi-machine
gang instead of case shards; `multi_node_aone.md` covers the gang config,
broker, and collection contract.

# Proposal

## Why

The repository currently has separate perf test and cache perf test entry points with overlapping orchestration, incompatible configuration and result formats, and different CI wrappers. A single contract is needed so prefill and decode retain their intended workload semantics without continuing two control planes.

## What Changes

- **BREAKING**: Replace the existing perf test and cache perf test user entry points with one `./tools/perf_test` command using explicit `--mode`, `--action`, and a single versioned config file.
- **BREAKING**: Remove compatibility with old CLI flags, configs, result directories, resume data, baselines, wrappers, and Bazel macros.
- Retain only cache-aware prefill grid testing, including local run/resume/profile/report and AOne run with offline reporting.
- Retain only the old local decode workloads: fixed input-length/batch grid, distribution workloads, and target-TPOT batch search with a single output length.
- Reject decode on AOne and all other unsupported mode/action/backend combinations before creating outputs, building targets, launching models, or submitting jobs.
- Introduce a normalized immutable execution plan, shared engine lifecycle, strict run/case state model, atomic per-case results, and a versioned result envelope.
- Transfer normalized AOne config and shard plans through generated Bazel runfiles instead of submit-host paths, with strict ownership and fingerprint checks during collection.
- Preserve baseline regression, golden generation, report generation, OSS archive, and ODPS summary behavior behind the new result and publishing contracts.
- Supersede the in-progress `add-cache-perf-local-aone-runners` change, whose retained low-level commands and cache-only control plane conflict with this replacement.

## Capabilities

### New Capabilities

- `perf-test-orchestration`: Unified CLI, strict config resolution, capability validation, immutable execution plans, dry-run behavior, and local/AOne execution boundaries.
- `prefill-cache-performance`: Cache-aware prefill grid planning and execution, reuse correctness, local resume/profile/report, and AOne prefill sharding.
- `decode-performance`: Local decode grid, distribution workload, DP scheduling, and target-TPOT batch search behavior.
- `perf-test-results`: Run and case state, atomic result storage, versioned result/report contracts, regression comparison, golden generation, and optional result publishers.

### Modified Capabilities

None. The project currently has no main capability specs; this change introduces the new contracts and supersedes the conflicting in-progress change rather than modifying an established spec.

## Impact

- Affected entry points and build integration: `tools/`, `rtp_llm/test/perf_test/BUILD`, and `rtp_llm/test/perf_test/defs.bzl`.
- Affected implementations: `batch_decode_test.py`, `test_entry.py`, `perf_config.py`, decode runners, and `cache_grid` config/runner/report modules.
- Affected CI systems: local GPU tests, AOne generated Bazel packages and collection, baseline/golden gates, OSS archival, and ODPS reporting.
- Affected users must migrate to the new config and result schemas; there is no transition compatibility layer.
- Multi-node perf tooling outside the unified entry point is not redesigned unless an existing dependency on a deleted symbol must be migrated.

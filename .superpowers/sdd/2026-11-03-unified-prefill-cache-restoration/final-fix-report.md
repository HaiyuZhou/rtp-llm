# Final whole-branch fix wave — 2026-11-03

All three final-review findings addressed. No unrelated untracked files changed.

## Changes

1. Added `prefill/block_size.py` as the single planner/runtime resolver. It recognizes the actual engine sharding argument `prefill_cp_kv_cache_sharded` and environment key `PREFILL_CP_KV_CACHE_SHARDED`, with CLI argument precedence and engine boolean spellings. Enabled sharding requires explicit `workload.probe_block_size`; no generic CP multiplier is inferred. The engine's related settings are `cp_rotate_method` / `CP_ROTATE_METHOD` and `prefill_cp_size` / `PREFILL_CP_SIZE`, verified against `server/server_args/parallel_group_args.py`. Argument precedence was checked against `EnvArgumentParser` in `server_args.py`. Grouped alignment/padding and runtime probing now share the same resolver. Registered the module in all three relevant Bazel source lists.
2. Every `plan_prefill_cases` invocation now validates probe seed and continuation lengths plus an output reserve against sequence and batch limits and optional block-padded workspace. This includes generated flat and named grids with inferred block size. Tests cover a 128-token probe against a 64-token limit for each budget and both alignment/engine-argument inference paths.
3. SVG subtitle now counts valid observations and describes the target as the across-round median of per-round maximum server first-token latency. It no longer labels observation count as unique geometries or assumes three requests. A generated-SVG regression checks the wording using repeated observations.

Updated the design and prefill user guide with the CP override rule and universal probe admission checks. CP integration tests additionally verify missing overrides at planning/runtime, explicit overrides, CLI-over-env disable behavior, grouped alignment, and the 96-token padded requirement for two 33-token members at block size 16.

## TDD and verification evidence

Commands ran from repository root with `python3` (Python 3.10). The first sandbox shell failed with `bwrap: pivot_root: Operation not permitted`; subsequent commands used reviewed escalated execution.

Initial test construction exposed three fixture errors (boolean rejected by scalar config map, a top-level seed conflicting with named grids, and missing SVG cache_len). Corrected fixtures before implementation. The same red command then had only expected behavioral failures:

```text
python3 -m unittest rtp_llm.test.perf_test.prefill_planner_test rtp_llm.test.perf_test.prefill_formula_test > /tmp/final-fix-red.log 2>&1
Ran 15 tests in 1.208s
FAILED (failures=16)
```

Failures were missing CP rejection, missing inferred probe budget rejection across flat/named cases, and old SVG wording. No production implementation was changed before this red run.

After implementation, the initial focused run passed 35 tests. After adding CP alignment/padding and runtime precedence coverage and formatting, the final focused command/output was:

```text
python3 -m unittest rtp_llm.test.perf_test.prefill_planner_test rtp_llm.test.perf_test.prefill_probe_test rtp_llm.test.perf_test.prefill_external_test rtp_llm.test.perf_test.prefill_formula_test > /tmp/final-fix-focused.log 2>&1
.....................................
----------------------------------------------------------------------
Ran 37 tests in 2.215s

OK
```

Broad unified suite (including all five added test methods):

```text
python3 -m unittest discover -s rtp_llm/test/perf_test -p '*_test.py' > /tmp/final-fix-broad.log 2>&1
Ran 546 tests in 28.043s

OK
```

The broad output includes expected failure-path logs from archive/ODPS publication tests (`upload rejected` and `ODPS rejected`, secrets masked), plus analyzer diagnostics; these are not unittest failures. Full output is in the temporary logs above. Broad verification preceded final formatting/documentation/Bazel list edits; the final focused suite was rerun after those edits.

```text
python3 -m compileall -q rtp_llm/test/perf_test
# exit 0, no output

git diff --check
# exit 0, no output
```

## Limitations

No live GPU, CP cache descriptor, HTTP/DashSC server, AOne, Kineto, nsys, or FlexLB Java execution was performed in this fix wave. The explicit CP block expectation is conservatively required even if a model's effective block equals its base block; the live admission probe remains responsible for validating actual reuse. Host environment settings outside persisted `engine.env` are not inferred by planning. `command -v bazel` returned exit 1, so Bazel execution was unavailable; source lists were updated and reviewed. `python3 -m black --version` reported `No module named black`, so formatting was manual and checked by compileall/diff checks.

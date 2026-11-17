# AOne performance execution

Run from the public workspace root. `./tools/perf_test` and the Bazel binary
`//rtp_llm/test/perf_test:perf_test` use the same public entry. AOne execution
needs site-provided Bazel remote configuration and authorization; endpoint and
authentication settings do not belong in performance JSON or generated runfiles.
The resource labels `daily`/`online` select the corresponding site-defined
`<resource>_aone_bazel_cache` and `<resource>_aone_bazel_remote` configs.
GPU types map to platforms in `aone/plan.py`.

## Three execution paths

| Workload | Config backend | Scheduling | Result backend |
| --- | --- | --- | --- |
| Single-machine prefill | `aone` | Independent case-shard targets, one model session per target | `aone` |
| Multi-machine prefill/decode | `aone` | One gang target, one Bazel test shard per machine | `aone` |
| Ordinary Bazel single-machine decode | `local` | One unsharded test action runs all cases | `local` |

AOne supports `run`, plus offline prefill `report`. It rejects `resume` and
`profile`; decode has no standalone report action. `backend=aone` single-machine
decode is rejected before execution. Public runs do not compare organization
baselines or publish to external services.

## Configure and preview

Ready-to-adapt AOne configs are `examples/prefill_aone.json` for single-machine
prefill, `examples/prefill_aone_gang.json` and `examples/decode_aone_gang.json`
for gangs, and `examples/prefill_aone_ci.json` for the grouped prefill workload
used by the internal CI adapter. Preview them with `./tools/perf_test
--mode=<prefill|decode> --action=run --config=<example-path> --dry-run`.
The examples contain no credentials or model weights. Their `daily` resource
and GPU labels require matching site configuration.

You can also start from `examples/prefill_local.json` or
`examples/decode_grid_local.json` and adapt the execution settings below.
Use worker-readable checkpoint/tokenizer paths or identifiers supported by the
worker runtime. Model weights are not packaged or copied from the submit host.
Keep `output.result_root` inside the public workspace: generated AOne packages
must have workspace-relative Bazel labels.

For single-machine prefill replace the complete `execution` object with:

```json
{"backend": "aone", "resource": "daily", "gpu_type": "H20",
 "shard_count": 2, "max_parallel": 2}
```

The model world size is `tp * dp * pp`. With `gpus_per_machine` omitted, all
ranks fit on one machine per case shard. `shard_count` is required, must not
exceed the case count, and `max_parallel` must not exceed `shard_count`.
Deterministic cost-aware scheduling assigns every case to exactly one shard.
Each target starts one model session for its owned cases. There is one build
and one remote test invocation for all targets.

For a two-machine gang set `model.topology` to TP2/DP1/PP1/EP1 and replace
`execution` with:

```json
{"backend": "aone", "resource": "daily", "gpu_type": "H20",
 "gpus_per_machine": 1, "max_parallel": 2}
```

When adapting `decode_grid_local.json` to DP1, also set
`engine.concurrency_limit` to at least 8 for its largest batch (the original
DP2 example uses four slots per worker).

`machine_count = ceil(world_size / gpus_per_machine)` is derived, never an
input field. A gang must divide world size evenly; `local_world_size` is the
number of ranks on each machine. A value at or above world size normalizes to
one machine, where prefill again requires `shard_count`. Gangs forbid
`shard_count` and require `max_parallel >= machine_count`. Each gang action
reserves the registered full-host GPU capacity to prevent members co-locating;
only its local rank group uses devices. Concurrent scheduling does not guarantee
atomic allocation of the whole gang.

AOne prefill accepts one grid per run. Concurrent prefill batches larger than
one require DP1. Cross-host gang decode with DP>1 uses synchronized local DP
frontends for every batch unit; this requires complete local TP groups. TP-only
gangs use the master frontend. All runs require PP1. Decode batches must keep
every DP worker active with equal allocations; inline profiling requires DP1.

Gang runfiles support decode distributions and generated, random-batch, or
traffic-sampled prefill workloads, including a named grid. The worker restores
the parser-facing shape while retaining the frozen workload separately. Startup
and master execution share the same plan builder, check both fingerprints, and
preserve case identities and traffic sampling metadata. Original controller
config/distribution files are not required on workers.

Save the adapted configs, then validate without creating results or remote jobs:

```bash
./tools/perf_test --mode=prefill --action=run --config=/path/to/prefill-aone.json --dry-run
./tools/perf_test --mode=decode --action=run --config=/path/to/decode-gang.json --dry-run
```

Remove `--dry-run` only with the required workers, models and submit-host
connectivity prepared. The entry owns package generation, build, submission,
collection and reporting. When using the built entry, launch
`./bazel-bin/rtp_llm/test/perf_test/perf_test` outside Bazel so it can drive this
lifecycle. Download the executable's runfiles and runtime dependencies as well.

## Submit host and broker reachability

CI control hosts are the defined gang submission environment. A development
machine uses the same public code, but local gang submission is **unverified**.
Before claiming it works, run a small remote Bazel probe in the same worker pool
that connects from a worker to the proposed broker address and port. Verify a
real authenticated broker request without printing its token, then complete a
two-machine model run. Local DNS, a local TCP probe, unit tests and dry-run do
not establish worker-to-host reachability. If the remote probe fails, stop
before allocating a full gang and use the CI control host.

The broker binds and advertises the same hostname, defaulting to the submit
host FQDN; `--broker-host` changes both. It must remain reachable for
the entire run. Separate bind/advertise addresses or a relay are not implemented.
Broker HTTP clients bypass ambient proxy settings. The per-run token travels
through inherited worker environment, never command-line arguments or saved
artifacts, and is removed before engine/rank subprocess launch. The broker URL
and run ID are declared in `gang_plan.json`. Workers register,
exchange members, send heartbeats, and cast terminal votes; any member failure
fails the gang. A bare `bazel test` does not start this submit-host broker.

The local path saves collected artifacts on the submit machine. Site CI supplies
its own configuration, acceptance checks, and artifact upload.

## Bazel inputs and ordinary single-machine decode

Generated prefill rules declare `resolved_config.json`, `resolved_workload.json`
and `shard_plan.json`; gang rules declare the first two plus `gang_plan.json`.
They use the public `aone_defs.bzl` rules, runtime and SDK. Workers consume these
frozen runfiles, not the submit host's original config path. Generated files
alone do not make undeclared inputs available on a remote machine.

For single-machine decode use `unified_decode_remote_test` in a BUILD file:

```starlark
load("//rtp_llm/test/perf_test:aone_defs.bzl", "unified_decode_remote_test")

unified_decode_remote_test(
    name = "decode_on_worker",
    config = "decode.json",  # execution.backend remains local
    gpu_type = "H20",
    gpu_count = 2,
    data = ["lengths.json"],  # only if referenced by the config
)
```

Declare the config and any distribution inputs as runfiles, preserving relative
paths beside the config. Match `gpu_count` and topology; `execution.devices`
selects logical slots in the wrapper's allocated device list, not host IDs.
The provided manual target `//rtp_llm/test/perf_test:decode_remote_example`
uses the public decode example and requires adapting its model paths. A site
with the named remote configs can invoke it as follows, adding its required
cache, authentication and platform policy flags:

```bash
bazelisk test //rtp_llm/test/perf_test:decode_remote_example \
  --config=cuda13 --config=daily_aone_bazel_remote \
  --experimental_remote_download_toplevel --test_timeout=7000
```

This target runs all planned cases once inside one test action. Its wrapper
sets the result root to `TEST_UNDECLARED_OUTPUTS_DIR`; retrieved results live
under `bazel-testlogs/<package>/<target>/test.outputs/`, possibly in
`outputs.zip`, with a `decode-*/` run directory. This path uses neither the
AOne case collector nor the gang broker.

The AOne entry always invokes `bazelisk`. Execution settings use explicit
command-line arguments:

| Argument | Purpose |
| --- | --- |
| `--bazel-startup-arg=VALUE` | Repeat for startup options, including an isolated output root |
| `--bazel-arg=VALUE` | Repeat for build/test arguments required by the site |
| `--broker-host=HOST` | Routable broker host for gangs; default submit-host FQDN |
| `--run-context=PATH` | Run metadata/provenance JSON |

For options beginning with `--`, use the equals form, for example
`--bazel-startup-arg=--output_user_root=/tmp/perf-bazel` or
`--bazel-arg=--jobs=32`. Each value remains one argv element; no shell splitting
is applied. The workspace comes from Bazel's `BUILD_WORKSPACE_DIRECTORY`,
falling back to the current directory. Results are collected through its
`bazel-testlogs` symlink, including with a separate output root. Build concurrency
defaults to the submit host's CPU count. Extra Bazel arguments apply to both
build and test, so `--bazel-arg=--jobs=N` also overrides test concurrency derived
from `execution.max_parallel`.

The CI delivery adapter accepts `--mode` and `--config`. The CI platform binds
its inputs at the script boundary; the script passes quoted arguments and drops
the bindings before launching the programs. Public entry points and executors do
not read those bindings as environment overrides.
An empty or `auto` config selects `examples/prefill_aone_ci.json` or
`examples/decode_aone_gang.json` by mode. Otherwise, provide a config file path
or an inline JSON object. Put model paths, topology, GPU type, resource pool,
and parallelism in that JSON; individual environment overrides are not read.
For gangs, configure `gpus_per_machine` and `max_parallel` explicitly and omit
single-host `shard_count`. Delivery pins `execution.backend` to `aone` and
`output.result_root` to `perf_test_results` for collection and CI upload. Relative
distribution paths resolve against the config file's directory, or the checkout
directory for inline JSON.

## Results and failure evidence

Every run preserves the [common run directory](reference.md#run-directory).
The result envelope stays schema v2; execution plans and historical AOne
fixtures remain readable. No baseline or historical result rewrite is required.

Single prefill adds `workload/aone_package/` and validates shard ownership,
case IDs, run identity and fingerprints before accepting results. Measurement
failures retain valid partial cases. Integrity failures exclude invalid evidence
from formal results; `results/aone_diagnostics.json` preserves independently
validated cases and rejection reasons where available.

Gang collection restores the master's `run/` tree, including `results/`,
`reports/`, `checkpoints/`, `profiles/` and `traces/`. Broker diagnostics are in
`workload/gang_broker/{broker_state.json,broker_events.jsonl}`; member logs and
summaries are in `workload/gang_shards/shard_N/` (one-based collection indices).
Additional worker traces are retained under `traces/`. Collection still attempts
later members when master output is missing. A failed remote test or broker
cannot become success merely because partial outputs exist.

Regenerate prefill reports without submitting any job:

```bash
./tools/perf_test --mode=prefill --action=report --run-dir=perf_test_results/prefill-run
```

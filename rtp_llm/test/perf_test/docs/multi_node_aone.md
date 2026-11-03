# Multi-machine AOne performance tests

One AOne task can occupy several machines as one gang that serves one
distributed model. Gang mode is part of the unified AOne backend: the entry
invocation is unchanged, and `execution.backend` stays `"aone"` in the config.
A derived machine count above one selects the gang path; a machine count of
one keeps the single-machine behavior of `aone_prefill.md` exactly. Both
decode and prefill `run` support gangs, with the shipped examples:

```bash
./tools/perf_test \
  --mode=decode \
  --action=run \
  --config=rtp_llm/test/perf_test/examples/decode_aone_gang.json \
  --dry-run
```

```bash
./tools/perf_test \
  --mode=prefill \
  --action=run \
  --config=rtp_llm/test/perf_test/examples/prefill_aone_gang.json \
  --dry-run
```

Both examples request TP2 across two single-GPU machines of `H20_CU13` in the
`daily` pool with `Qwen/Qwen2.5-0.5B-Instruct`. Checkpoint and tokenizer paths
must be readable inside the AOne runtime; adapt them together with the
topology and GPU type for the target environment.

## Gang config contract

`execution.gpus_per_machine` is the only new field and it is optional. It
defaults to the model world size `tp*dp*pp`, so omitting it derives
`machine_count == 1` and preserves single-machine AOne execution. The machine
count is always derived, never configured:

- `machine_count = ceil(world_size / gpus_per_machine)`. An explicit
  `execution.machine_count` field is rejected as unknown.
- A `gpus_per_machine` at or above `world_size` derives one machine and is a
  no-op, not an error.
- A gang must divide the world size evenly: `gpus_per_machine 4` on a world
  size of 10 is rejected because the derived three machines cannot host a
  uniform local world size. Each machine then hosts
  `local_world_size = world_size / machine_count` ranks.
- `execution.shard_count` is rejected in gang mode: case-shard and gang
  semantics are mutually exclusive. When the derivation yields one machine,
  `shard_count` stays required exactly as in single-machine mode.
- `execution.max_parallel` must be at least `machine_count`, because the
  whole gang is submitted as one wave.

Build parallelism is not a config field; `PERF_TEST_BAZEL_BUILD_JOBS`
controls it as in single-machine mode. Every rejected combination fails
during config or plan validation, before a run directory, model process,
build, or remote submission exists.

## Gang selection semantics

`machine_count > 1` selects the gang path. Decode plus AOne is accepted by
the capability matrix only for `run`, and the single-machine combination is
rejected at plan validation with a message naming `gpus_per_machine` and
`machine_count`; only a derived gang reaches execution. Decode `resume`,
`profile`, and `report` remain unsupported for every backend. Prefill keeps
its offline `report` after collection and otherwise follows the same gang
path: the whole plan executes against one distributed server instead of
case-sharded single-machine shards.

## Rendezvous broker

The submit host runs one broker in-process inside the entry. It binds the
host's fully qualified domain name by default so remote workers can reach it;
`PERF_TEST_BROKER_HOST` overrides the advertised host. Members discover each
other through a small authenticated HTTP API under
`/v1/runs/<run_id>/<operation>` with the operations `register`, `members`,
`heartbeat`, `result`, and `cleanup`. Every request carries a bearer token in
the authorization header; the token reaches the remote workers through
`PERF_TEST_BROKER_URL`, `PERF_TEST_BROKER_TOKEN`, and
`PERF_TEST_BROKER_RUN_ID` — exported into the submit environment of the
bazel test invocation and carried to every remote shard by the generated
rule's `env_inherit`, never on a command line — and is never persisted to
the manifest, the run directory, or any log.

The broker records each registration's HTTP source address, claimed hostname,
and best-effort reverse-lookup result. Container hostnames may differ from
machine FQDNs, so that mismatch is tolerated; duplicate source addresses or
claimed hostnames, duplicate shard registrations, and config fingerprint
mismatches are rejected. Three deadlines fail the gang instead of hanging it:
registration must complete within 3600 seconds, the whole run ends at 43200
seconds, and a member silent for 120 seconds is marked lost. Workers send a
heartbeat every 30 seconds while they wait. Rendezvous state and events are
persisted under the run directory as `workload/gang_broker/broker_state.json`
and `broker_events.jsonl`.

## Machine exclusivity

A gang shard always requests the full registered per-host GPU count of the
selected GPU type, so two members cannot co-land on one machine. The
registered types are `H20`, `H20_CU13`, `L20D_DEV`, `L20_CU13`,
`SM100_ARM_CU13`, and `SM103_CU13`, each with eight GPUs per host; a gang GPU
type without a registered per-host count is rejected during planning. Each
worker restricts itself to its `local_world_size` devices through
`CUDA_VISIBLE_DEVICES`, and the broker's duplicate-host rejection remains as
a backstop.

## Worker flow

Bazel expands the single generated `perf_test_gang_shard` target into
`machine_count` remote test shards, each one worker process per machine. The
worker loads `resolved_config.json`, `resolved_workload.json`, and
`gang_plan.json` from runfiles, registers with the broker, waits for the full
member list, builds the gang config string ordered by shard index with
`WORLD_RANK = shard_index * local_world_size`, and starts its local rank
group on consecutive ports beginning at 12333. The master member — shard
index 0 — then executes the whole unified plan against the gang's server:
decode drives the external decode core, prefill drives the external prefill
session. Non-master members finish once their rank group is up and wait for
the master's outcome. Every exit path casts exactly one terminal vote; any
member failure, heartbeat expiry, or deadline miss makes all members exit
nonzero while the master's partial `run/` artifacts remain on disk.

## Dry run

`--dry-run` prints the normalized config, config and workload fingerprints,
planned cases, the derived gang topology with its member layout and generated
target, and the final build and remote-test commands — the remote test is
shown with `--jobs=max(max_parallel, machine_count)`. No broker starts, no
Bazel subprocess runs, and no run directory or remote submission is created.

## Collection and artifacts

The gang executor drives the staged lifecycle rendezvous, build, submit,
collect, and report, and every transition persists to the run manifest.
Remote execution runs one Bazel build for the gang package followed by one
remote Bazel test with a 7000-second test timeout. Collection copies the
master member's full `run/` tree from the Bazel testlogs into the run
directory — `results/` with the common envelope recording backend `aone`,
`reports/`, `traces/`, and `profiles/` — giving local-run artifact parity.
Non-master members contribute logs only, collected under
`workload/gang_shards/shard_N/` together with their token-free
`gang_worker.json` summaries. The manifest's `gang` block records
`machine_count`, `local_world_size`, `gpu_type`, the broker URL and run id,
and one sanitized `{shard_index, hostname}` entry per member. Collection
always runs, and the original test failure exit code wins over collect or
report errors.

## CI pipeline

The internal repository ships `.aoneci/multi-node-perf.yaml`. Its
control-plane job checks out the pinned github-opensource sources, patches
the selected gang config with the pipeline's model, topology, and execution
parameters while pinning `execution.backend` to `"aone"`, and invokes the
same entry:

```bash
./tools/perf_test \
  --mode=decode \
  --action=run \
  --config=multi_node_perf_config.json
```

A concurrency group named `multi-node-perf` serializes runs so overlapping
gangs cannot double-book pool machines. Every run directory, including the
manifest, reports, result envelope, and `workload/gang_broker` diagnostics,
is uploaded as a pipeline artifact. The pipeline's 12-hour job timeout
matches the broker's run deadline.

Local dev-machine submission is not validated: the CI runner is the only
supported submit host. This phase A boundary exists because submitting from a
development machine (phase B) additionally requires worker-to-host
reachability and remote Bazel access from the submit host, and that
connectivity is unproven; a gang whose members cannot reach the broker fails
fast — in worker setup or at the register deadline — instead of hanging.

## Prefill measurement contract

Gang prefill uses the same `concurrent_batch_v1` contract as local execution.
Prefill plans containing any `batch_size > 1` require `model.topology.dp == 1`;
plans containing only batch size 1 still support DP > 1. TP/PP may span gang
members, but independent DP engines cannot share one measured batch. Throughput
comes from monotonic elapsed time for each complete concurrent round, excluding scheduler
updates and serial seeding; server first-token latency is reported separately.

The contract changes stable prefill case IDs. Existing baseline keys do not match
new runs. Generate and review baseline candidates and goldens explicitly using
`regression_publish.md`; submission, collection, and reporting never migrate
accepted files automatically. Offline reports retain old directories with the
`legacy_latency_derived_v1` label and expose `concurrent_batch_v1` on new points.
Inconsistent new-contract evidence is excluded from diagnostic reports. The
common result envelope remains schema v2.

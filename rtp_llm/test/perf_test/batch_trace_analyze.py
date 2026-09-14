"""Join engine request/batch records with Kineto CUDA launch correlations.

GPU timestamps are never matched against CPU ranges. Only CPU launch events
are assigned by CPU thread containment, then joined to GPU events by correlation.
"""

import argparse
import collections
import html
import json
import re
from pathlib import Path


def read_jsonl(path):
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{number}: invalid JSONL") from error


def request_key(row):
    return tuple(
        row.get(k) for k in ("session_id", "replica_id", "dp_rank", "request_id")
    )


def request_latencies(rows):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[request_key(row)].append(row)
    result = []
    for key, events in groups.items():
        events.sort(key=lambda row: row["event_seq"])
        by_event = collections.defaultdict(list)
        for row in events:
            by_event[row["event"]].append(row)
        problems = []
        if [e["event_seq"] for e in events] != list(range(len(events))):
            problems.append("missing_or_duplicate_sequence")
        if any(len(v) > 1 for v in by_event.values()):
            problems.append("duplicate_event")
        terminals = [e for e in events if e["event"] in ("finish", "cancel", "error")]
        if len(terminals) != 1 or len(by_event["enqueue"]) != 1:
            problems.append("missing_or_duplicate_endpoint")
        if len({(e.get("owner_instance_id"), e.get("clock_id")) for e in events}) != 1:
            problems.append("clock_mismatch")
        times = [e["timestamp_monotonic_ns"] for e in events]
        if times != sorted(times):
            problems.append("non_monotonic_events")
        if events[0]["event"] != "enqueue" or (
            terminals and events[-1] not in terminals
        ):
            problems.append("invalid_event_order")
        if by_event["first_token"] and not by_event["first_scheduled"]:
            problems.append("missing_first_scheduled")
        if (
            by_event["first_token"]
            and by_event["first_scheduled"]
            and by_event["first_token"][0]["event_seq"]
            < by_event["first_scheduled"][0]["event_seq"]
        ):
            problems.append("first_token_before_schedule")
        if any(
            e["event"]
            not in (
                "enqueue",
                "first_scheduled",
                "first_token",
                "finish",
                "cancel",
                "error",
            )
            for e in events
        ):
            problems.append("unknown_event")

        def delta(end, start):
            if problems or not end or not start:
                return None
            return end[0]["timestamp_monotonic_ns"] - start[0]["timestamp_monotonic_ns"]

        result.append(
            dict(
                zip(("session_id", "replica_id", "dp_rank", "request_id"), key),
                status="partial" if problems else "complete",
                problems=problems,
                terminal=terminals[0]["event"] if len(terminals) == 1 else None,
                output_mode=events[0].get("output_mode"),
                engine_request_latency_ns=delta(terminals, by_event["enqueue"]),
                initial_queue_latency_ns=delta(
                    by_event["first_scheduled"], by_event["enqueue"]
                ),
                engine_first_token_latency_ns=delta(
                    by_event["first_token"], by_event["enqueue"]
                ),
                post_first_schedule_latency_ns=delta(
                    terminals, by_event["first_scheduled"]
                ),
            )
        )
    return result


EXECUTION = re.compile(r"^rtp\.execution\(id=(\d+)\)$")
GRAPH = re.compile(r"^rtp\.graph_replay\(bucket=(\d+),prefill=(\d+)\)$")


def contains(scope, event):
    return (scope.get("pid"), scope.get("tid")) == (
        event.get("pid"),
        event.get("tid"),
    ) and (
        scope["ts"] <= event["ts"]
        and event["ts"] + event.get("dur", 0) <= scope["ts"] + scope["dur"]
    )


def correlate_trace(trace, identity):
    events = trace.get("traceEvents", [])
    runtime_pids = {
        e.get("pid") for e in events if e.get("cat") in ("cuda_runtime", "cuda_driver")
    }
    if len(runtime_pids) > 1:
        raise ValueError(
            "merged multi-process traces must be split by owner before correlation"
        )
    scopes = [
        e for e in events if e.get("ph") == "X" and EXECUTION.match(e.get("name", ""))
    ]
    launches = collections.defaultdict(set)
    execution_modes = {}
    for event in events:
        if event.get("ph") != "X":
            continue
        graph = GRAPH.match(event.get("name", ""))
        if event.get("name") == "py_model.forward(normal)":
            parents = [s for s in scopes if contains(s, event)]
            if len(parents) == 1:
                eid = int(EXECUTION.match(parents[0]["name"])[1])
                execution_modes[eid] = {"execution_mode": "eager", "graph_bucket": None}
        if graph:
            parents = [s for s in scopes if contains(s, event)]
            if len(parents) == 1:
                eid = int(EXECUTION.match(parents[0]["name"])[1])
                execution_modes[eid] = {
                    "execution_mode": "cuda_graph",
                    "graph_bucket": int(graph[1]),
                }
        if event.get("cat") not in ("cuda_runtime", "cuda_driver"):
            continue
        corr = event.get("args", {}).get("correlation")
        if corr is None:
            continue
        parents = [s for s in scopes if contains(s, event)]
        for parent in parents:
            eid = int(EXECUTION.match(parent["name"])[1])
            if eid:
                launches[corr].add(eid)

    kernels = []
    for event in events:
        if event.get("ph") != "X" or event.get("cat") != "kernel":
            continue
        args = event.get("args", {})
        corr = args.get("correlation")
        owners = launches.get(corr, set())
        matched = len(owners) == 1
        kernels.append(
            dict(
                identity,
                kernel_name=event["name"],
                execution_id=next(iter(owners)) if matched else None,
                start_ns=round(event["ts"] * 1000),
                duration_ns=round(event["dur"] * 1000),
                device=args.get("device"),
                context=args.get("context"),
                stream=args.get("stream"),
                correlation_id=corr,
                graph_node_id=args.get("graph node id"),
                association_method="cuda_launch_correlation" if matched else None,
                association_status=(
                    "matched" if matched else "ambiguous" if owners else "unmatched"
                ),
            )
        )
    return kernels, execution_modes


def validate_batch(batch):
    if batch.get("schema_version") != 1:
        raise ValueError("unsupported batch schema")
    rows = batch["requests"]
    if [r["batch_slot"] for r in rows] != list(range(len(rows))):
        raise ValueError("batch slots must be contiguous and ordered")
    seen_prefill = False
    for row in rows:
        if row["phase"] not in ("prefill", "decode") or row.get("is_fake"):
            raise ValueError("unsupported phase or fake request")
        if row["phase"] == "prefill":
            seen_prefill = True
        elif seen_prefill or row["q_tokens"] != 1:
            raise ValueError("ordinary decode must precede prefill and have q=1")
        if any(type(row[k]) is not int for k in ("q_tokens", "kv_tokens_before")):
            raise ValueError("lengths must be integers")
        if row["q_tokens"] <= 0 or row["kv_tokens_before"] < 0:
            raise ValueError("invalid q/KV lengths")
    return rows


def summarize_kernels(kernels):
    groups = collections.defaultdict(list)
    for kernel in kernels:
        groups[(kernel.get("world_rank"), kernel["kernel_name"])].append(
            kernel["duration_ns"]
        )
    return [
        dict(
            world_rank=rank,
            kernel_name=name,
            count=len(values),
            total_ns=sum(values),
            mean_ns=sum(values) / len(values),
            min_ns=min(values),
            max_ns=max(values),
        )
        for (rank, name), values in sorted(
            groups.items(), key=lambda item: -sum(item[1])
        )
    ]


def execution_timings(kernels):
    groups = collections.defaultdict(list)
    for kernel in kernels:
        if kernel["execution_id"] is not None:
            groups[
                (kernel["execution_id"], kernel.get("world_rank"), kernel.get("device"))
            ].append(kernel)
    result = []
    for (eid, rank, device), rows in groups.items():
        intervals = sorted(
            (r["start_ns"], r["start_ns"] + r["duration_ns"]) for r in rows
        )
        begin, end = intervals[0]
        union = 0
        for start, stop in intervals[1:]:
            if start > end:
                union += end - begin
                begin, end = start, stop
            else:
                end = max(end, stop)
        union += end - begin
        result.append(
            dict(
                execution_id=eid,
                world_rank=rank,
                device=device,
                kernel_count=len(rows),
                kernel_sum_ns=sum(r["duration_ns"] for r in rows),
                kernel_union_ns=union,
                gpu_span_ns=max(stop for _, stop in intervals) - intervals[0][0],
            )
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", required=True, type=Path)
    parser.add_argument(
        "--batch-dir",
        type=Path,
        help="TP input owner's directory; defaults to record-dir",
    )
    parser.add_argument(
        "--trace", type=Path, help="One Kineto trace from the specified owner/rank"
    )
    parser.add_argument("--world-rank", type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = json.loads((args.record_dir / "manifest.json").read_text())
    rows = []
    for path in sorted(args.record_dir.glob("request_events.jsonl*")):
        rows.extend(read_jsonl(path))
    batches = []
    for path in sorted((args.batch_dir or args.record_dir).glob("batches.jsonl*")):
        batches.extend(read_jsonl(path))
    for batch in batches:
        if any(
            batch.get(key) != manifest.get(key)
            for key in ("session_id", "replica_id", "dp_rank")
        ):
            parser.error(
                "batch owner and trace owner belong to different recording domains"
            )
    requests = request_latencies(rows)
    kernels, modes = [], {}
    if args.trace:
        if args.world_rank is None or args.world_rank != manifest["world_rank"]:
            parser.error("--world-rank must match the recording owner manifest")
        identity = {
            k: manifest[k]
            for k in ("session_id", "replica_id", "dp_rank", "world_rank")
        }
        identity["capture_id"] = args.trace.name
        kernels, modes = correlate_trace(json.loads(args.trace.read_text()), identity)
    # A truncated recording cannot certify any request as complete.
    if not manifest.get("complete"):
        for request in requests:
            request["status"] = "partial"
            request["problems"].append("recording_incomplete")
            for name in (
                "engine_request_latency_ns",
                "initial_queue_latency_ns",
                "engine_first_token_latency_ns",
                "post_first_schedule_latency_ns",
            ):
                request[name] = None
    report = dict(
        recording_complete=manifest.get("complete", False),
        requests=requests,
        batches=batches,
        missing_batch_execution_ids=sorted(
            {k["execution_id"] for k in kernels if k["execution_id"] is not None}
            - {b["execution_id"] for b in batches}
        ),
        execution_timings=execution_timings(kernels),
        kernel_summary=summarize_kernels(kernels),
        execution_modes=modes,
        kernel_count=len(kernels),
        matched_kernel_count=sum(k["association_status"] == "matched" for k in kernels),
        limitations=[
            "shared batch GPU time is not per-request GPU time",
            "graph node coverage requires backend validation",
        ],
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output / "kernel_events.jsonl").open("w") as stream:
        for kernel in kernels:
            stream.write(json.dumps(kernel) + "\n")
    (args.output / "report.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Batch execution report</title>"
        "<pre>" + html.escape(json.dumps(report, indent=2)) + "</pre>"
    )


if __name__ == "__main__":
    main()

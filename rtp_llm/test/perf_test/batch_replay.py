"""Prepare a native engine replay plan; optionally run a dedicated engine process.

Example: python -m rtp_llm.test.perf_test.batch_replay --batches batches.jsonl
--output /tmp/replay -- python -m rtp_llm.start_server <model arguments>
The engine command must use a fresh instance and the recording's model/topology.
"""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from rtp_llm.test.perf_test.batch_trace_analyze import read_jsonl, validate_batch


def make_plan(batches, warmup=5, repeat=20):
    if warmup < 0 or not 0 < repeat <= 10000 or warmup > 10000 or not batches:
        raise ValueError("invalid warmup/repeat or empty batches")
    lines = [f"RTP_BATCH_REPLAY_V1 {warmup} {repeat} {len(batches)}"]
    seen = set()
    for batch in batches:
        rows = validate_batch(batch)
        eid = batch["execution_id"]
        if type(eid) is not int or not 0 < eid < 2**63 or eid in seen or not rows:
            raise ValueError(
                "execution IDs must be unique positive int64 values; batch cannot be empty"
            )
        seen.add(eid)
        lines.append(f"{eid} {len(rows)}")
        for row in rows:
            prompt = row["prompt_tokens"]
            if type(prompt) is not int or prompt <= 0:
                raise ValueError("prompt_tokens must be positive")
            lines.append(
                f"{int(row['phase'] == 'prefill')} {row['q_tokens']} {row['kv_tokens_before']} {prompt}"
            )
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=Path, required=True)
    parser.add_argument("--execution-ids", help="Comma-separated IDs; default all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.world_size <= 0 or args.timeout <= 0:
        parser.error("world-size and timeout must be positive")
    manifest_path = args.batches.parent / "manifest.json"
    source_manifest = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    )
    best_effort = bool(
        source_manifest and source_manifest.get("delivery_policy") == "best_effort"
    )
    batches = list(read_jsonl(args.batches, skip_invalid=best_effort))
    if args.execution_ids:
        wanted = {int(value) for value in args.execution_ids.split(",")}
        batches = [batch for batch in batches if batch["execution_id"] in wanted]
        if {batch["execution_id"] for batch in batches} != wanted:
            parser.error("some requested execution IDs are missing")
    plan = make_plan(batches, args.warmup, args.repeat)
    # A fresh directory prevents accepting stale completion files or overwriting a capture.
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output.resolve() / "replay.plan"
    path.write_text(plan)
    if source_manifest is not None:
        (args.output / "source_manifest.json").write_text(
            json.dumps(source_manifest, indent=2) + "\n"
        )
    (args.output / "source_batches.jsonl").write_text(
        "".join(json.dumps(b) + "\n" for b in batches)
    )
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        print(
            f"Plan written: {path}; run a dedicated engine with RTP_LLM_REPLAY_PLAN={path}"
        )
        return
    env = dict(os.environ, RTP_LLM_REPLAY_PLAN=str(path))
    env.pop("RTP_LLM_RECORD_DIR", None)
    with (args.output / "engine.log").open("w") as log:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                complete = 0
                for rank in range(args.world_size):
                    result_path = Path(f"{path}.rank{rank}.result.jsonl")
                    if not result_path.exists():
                        continue
                    # Writer may be midway through the final line; retry incomplete tails.
                    lines = result_path.read_text().splitlines(keepends=True)
                    if not lines or not lines[-1].endswith("\n"):
                        continue
                    last = json.loads(lines[-1])
                    if last.get("status") == "error":
                        raise RuntimeError(f"rank {rank}: {last['reason']}")
                    complete += last.get("status") == "complete"
                if complete == args.world_size:
                    print(f"Replay completed for {complete} ranks: {args.output}")
                    return
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Engine exited with {process.returncode}; see engine.log"
                    )
                time.sleep(0.25)
            raise TimeoutError("Replay did not complete before timeout; see engine.log")
        finally:
            # Only the process group created by this launcher is signalled.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            if process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


if __name__ == "__main__":
    main()

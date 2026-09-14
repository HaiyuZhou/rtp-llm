"""Small opt-in CUDA/Kineto correlation check, run on an explicitly selected GPU."""

import argparse
import json
from pathlib import Path

import torch

from rtp_llm.test.perf_test.batch_trace_analyze import correlate_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    x = torch.ones((128, 128), device="cuda")
    y = torch.empty_like(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            torch.mm(x, x, out=y)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.mm(x, x, out=y)
    torch.cuda.synchronize()
    summary = {}
    for mode in ("eager", "graph"):
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler:
            for eid in (101, 102):
                with torch.profiler.record_function(f"rtp.execution(id={eid})"):
                    if mode == "graph":
                        with torch.profiler.record_function(
                            "rtp.graph_replay(bucket=1,prefill=0)"
                        ):
                            graph.replay()
                    else:
                        torch.mm(x, x, out=y)
            torch.cuda.synchronize()
        path = args.output / f"{mode}.json"
        profiler.export_chrome_trace(str(path))
        kernels, modes = correlate_trace(
            json.loads(path.read_text()), {"world_rank": 0}
        )
        matched = {
            k["execution_id"] for k in kernels if k["association_status"] == "matched"
        }
        summary[mode] = dict(
            kernel_count=len(kernels),
            matched_ids=sorted(matched),
            execution_modes=modes,
            supported=matched == {101, 102}
            and all(k["association_status"] == "matched" for k in kernels),
        )
    (args.output / "coverage.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["eager"]["supported"]:
        raise RuntimeError("eager correlation validation failed")


if __name__ == "__main__":
    main()

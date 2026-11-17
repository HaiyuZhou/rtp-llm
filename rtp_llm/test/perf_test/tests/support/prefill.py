"""Fixtures shared by tests split from prefill_runner_test."""

import threading

from rtp_llm.test.perf_test.modes import ModeExecutionContext, PrefillAdapter
from rtp_llm.test.perf_test.prefill.runner import PrefillRunner
from rtp_llm.test.perf_test.storage import RunStore
from rtp_llm.test.perf_test.tests.support.fixtures import (
    prefill_external_COMMIT_TAIL_TOKENS,
    prefill_external_Tokenizer,
)


def prefill_external_run_through_session(plan, session, run_root):
    """Drive the prefill runner factory exactly like the unified executor."""
    (run_root / "results").mkdir(parents=True, exist_ok=True)
    (run_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    adapter = PrefillAdapter(
        lambda context: PrefillRunner(
            context.session.client,
            prefill_external_Tokenizer(),
            commit_tail_tokens=prefill_external_COMMIT_TAIL_TOKENS,
            max_batch_tokens=plan.resolved_config["engine"]["max_batch_tokens"],
            monotonic=iter([0.0, 0.2]).__next__,
        )
    )
    return adapter(
        ModeExecutionContext(plan=plan, session=session, store=RunStore(run_root))
    )


class prefill_runner_Tokenizer:

    def encode(self, text):
        return text.split()


class prefill_runner_Client:

    def __init__(self, responses, batch_size=1):
        self.responses = {}
        seed_index = measure_index = 0
        for response in responses:
            if response["input_len"] <= 12:
                key = ("seed", None, seed_index)
                seed_index += 1
            else:
                key = (
                    "measure",
                    measure_index // batch_size,
                    measure_index % batch_size,
                )
                measure_index += 1
            self.responses[key] = response
        self.requests = []
        self.scheduler_calls = []

    def configure_scheduler(self, batch_size, mode="prefill"):
        assert mode == "prefill"
        self.scheduler_calls.append(batch_size)

    def request(self, **request):
        self.requests.append(request)
        if request["phase"] == "warmup":
            return self.responses["measure", 0, request["request_index"]]
        return self.responses[
            request["phase"], request["run_index"], request["request_index"]
        ]


def prefill_runner_response(
    input_len, reuse_len, latency=10.0, success=True, output_len=1
):
    return {
        "success": success,
        "input_len": input_len,
        "output_len": output_len,
        "reuse_len": reuse_len,
        "latency_ms": latency,
    }


def prefill_runner_case(cache_len=8, batch_size=1, measure_runs=2):
    return {
        "case_id": "prefill-case",
        "input_len": 16,
        "cache_len": cache_len,
        "compute_len": 16 - cache_len,
        "batch_size": batch_size,
        "measure_runs": measure_runs,
        "seed": 7,
        "prefix_policy": "shared_seed" if batch_size == 1 else "independent",
    }


def prefill_runner_grouped_case():
    return dict(
        prefill_runner_case(batch_size=3, measure_runs=1),
        input_len=24,
        cache_len=8,
        compute_len=16,
        prefix_policy="independent",
        measurement_contract="concurrent_grouped_batch_v1",
        request_groups=[
            {"count": 1, "input_len": 16, "cache_len": 8},
            {"count": 2, "input_len": 24, "cache_len": 0},
        ],
    )


class SeedBatchClient:
    """Require complete concurrent scheduler groups without a GPU server."""

    def __init__(self, cache_lengths, *, failed_member=None, raised_member=None):
        self.cache_lengths = cache_lengths
        self.failed_member = failed_member
        self.raised_member = raised_member
        self.batches = []
        self.completed_seeds = set()
        self.lock = threading.Lock()

    def configure_scheduler(self, batch_size, mode="prefill"):
        assert mode == "prefill"
        if self.batches:
            assert self.batches[-1]["completed"] == len(self.batches[-1]["requests"])
        self.batches.append(
            {
                "barrier": threading.Barrier(batch_size, timeout=2),
                "requests": [],
                "completed": 0,
            }
        )

    def request(self, **request):
        with self.lock:
            batch = self.batches[-1]
            batch["requests"].append(request)
        try:
            batch["barrier"].wait()
            member = request["request_index"]
            seed = request["phase"] == "seed"
            if not seed:
                assert self.completed_seeds == {
                    index for index, length in enumerate(self.cache_lengths) if length
                }
            if seed and member == self.raised_member:
                raise RuntimeError("seed transport failed")
            return prefill_runner_response(
                len(request["input_ids"]),
                0 if seed else self.cache_lengths[member],
                success=not (seed and member == self.failed_member),
            )
        finally:
            with self.lock:
                batch["completed"] += 1
                if request["phase"] == "seed":
                    self.completed_seeds.add(request["request_index"])

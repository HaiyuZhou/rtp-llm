import unittest
from unittest.mock import Mock

from rtp_llm.test.perf_test.config_schema import EngineConfig
from rtp_llm.test.perf_test.decode.executor import (
    DecodeBatchExecutor,
    DecodeCaseExecutor,
)
from rtp_llm.test.perf_test.decode.model import DecodeCaseSpec, ExactBatchPolicy
from rtp_llm.test.perf_test.decode.scheduler import schedule_decode_requests


class DecodeSchedulerTest(unittest.TestCase):
    def test_even_and_uneven_batches_cover_each_request_exactly_once(self):
        for global_batch, dp_size, local_batches in (
            (8, 2, [4, 4]),
            (5, 2, [3, 2]),
            (2, 4, [1, 1, 0, 0]),
        ):
            with self.subTest(global_batch=global_batch, dp_size=dp_size):
                lengths = tuple(100 + index for index in range(global_batch))

                first = schedule_decode_requests(lengths, dp_size=dp_size)
                second = schedule_decode_requests(lengths, dp_size=dp_size)

                self.assertEqual(first, second)
                self.assertEqual(global_batch, first.global_batch_size)
                self.assertEqual(
                    local_batches, [item.local_batch_size for item in first.workers]
                )
                indices = [
                    index
                    for worker in first.workers
                    for index in worker.request_indices
                ]
                self.assertEqual(list(range(global_batch)), indices)
                self.assertEqual(len(indices), len(set(indices)))

    def test_executor_reports_global_batch_topology_and_worker_allocation(self):
        transport = Mock()
        transport.run_round.return_value = tuple(
            {
                "request_index": index,
                "success": True,
                "input_length": 128,
                "output_length": 32,
                "latency_ms": {
                    "wait": 1.0,
                    "prefill": 2.0,
                    "decode": 3.0,
                    "tpot": 0.1,
                    "total": 5.0,
                },
            }
            for index in range(5)
        )
        topology = {"tp": 2, "dp": 2, "pp": 1, "ep": 1}
        case = DecodeCaseSpec(
            workload_kind="grid",
            input_lengths=(128,) * 5,
            output_length=32,
            batch_policy=ExactBatchPolicy("fixed", 5),
            measure_runs=1,
            warmup_runs=0,
        )
        batch_executor = DecodeBatchExecutor(
            transport,
            engine=EngineConfig(
                max_seq_len=512, max_batch_tokens=512, concurrency_limit=3
            ),
            dp_size=2,
        )
        result = DecodeCaseExecutor(batch_executor, topology=topology).execute(
            {"case_id": "decode-case", **case.to_identity()}
        )

        self.assertEqual(5, len(result["identity"]["input_lengths"]))
        self.assertEqual(5, result["metrics"]["global_batch_size"])
        self.assertEqual(topology, result["metrics"]["topology"])
        self.assertEqual(
            [3, 2],
            [
                worker["local_batch_size"]
                for worker in result["metrics"]["measurements"][0]["worker_allocations"]
            ],
        )
        self.assertEqual(5, len(transport.run_round.call_args.kwargs["input_lengths"]))
        self.assertEqual(
            [3, 2],
            [
                worker["local_batch_size"]
                for worker in transport.configure_scheduler.call_args.args[0]
            ],
        )


if __name__ == "__main__":
    unittest.main()

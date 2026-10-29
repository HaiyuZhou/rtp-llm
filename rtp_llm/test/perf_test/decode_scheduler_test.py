import unittest
from unittest.mock import Mock

from rtp_llm.test.perf_test.decode.fixed import FixedDecodeCaseRunner
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

    def test_fixed_runner_reports_global_batch_topology_and_worker_allocation(self):
        def execute(**kwargs):
            return {
                "measurements": [
                    {
                        "run": 0,
                        "throughput_tokens_per_second": 1000.0,
                        "worker_allocations": kwargs["worker_allocations"],
                        "requests": [
                            {
                                "request_index": index,
                                "success": True,
                                "input_length": input_length,
                                "output_length": kwargs["output_length"],
                                "latency_ms": {
                                    "wait": 1.0,
                                    "prefill": 2.0,
                                    "decode": 3.0,
                                    "tpot": 0.1,
                                    "total": 5.0,
                                },
                            }
                            for index, input_length in enumerate(
                                kwargs["input_lengths"]
                            )
                        ],
                    }
                ]
            }

        batch_runner = Mock(side_effect=execute)
        topology = {"tp": 2, "dp": 2, "pp": 1, "ep": 1}
        case = {
            "case_id": "decode-case",
            "workload_kind": "grid",
            "input_length": 128,
            "output_length": 32,
            "global_batch_size": 5,
            "measure_runs": 1,
            "warmup_runs": 0,
            "generate_config": {},
            "target_tpot_ms": None,
        }

        result = FixedDecodeCaseRunner(batch_runner, topology=topology).run_case(case)

        self.assertEqual(5, result["identity"]["global_batch_size"])
        self.assertEqual(5, result["metrics"]["global_batch_size"])
        self.assertEqual(topology, result["metrics"]["topology"])
        self.assertEqual(
            [3, 2],
            [
                worker["local_batch_size"]
                for worker in result["metrics"]["measurements"][0]["worker_allocations"]
            ],
        )
        self.assertEqual(5, batch_runner.call_args.kwargs["global_batch_size"])
        self.assertEqual(
            [3, 2],
            [
                worker["local_batch_size"]
                for worker in batch_runner.call_args.kwargs["worker_allocations"]
            ],
        )


if __name__ == "__main__":
    unittest.main()

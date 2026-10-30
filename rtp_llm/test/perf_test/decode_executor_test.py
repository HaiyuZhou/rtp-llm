import unittest
from dataclasses import replace
from unittest.mock import Mock, call

from rtp_llm.test.perf_test.config_schema import EngineConfig
from rtp_llm.test.perf_test.decode.executor import (
    DecodeBatchExecutor,
    DecodeCapacityError,
)
from rtp_llm.test.perf_test.decode.model import DecodeCaseSpec, ExactBatchPolicy
from rtp_llm.test.perf_test.errors import PlanningError


def _request_round(input_lengths, output_length):
    return tuple(
        {
            "request_index": index,
            "success": True,
            "input_length": input_length,
            "output_length": output_length,
            "latency_ms": {
                "wait": 1.0,
                "prefill": 2.0,
                "decode": 15.0,
                "tpot": 1.0,
                "total": 17.0,
            },
        }
        for index, input_length in enumerate(input_lengths)
    )


def _exact_case(*, batch, warmup=0, runs=1):
    return DecodeCaseSpec(
        workload_kind="grid",
        input_lengths=(128,) * batch,
        output_length=16,
        batch_policy=ExactBatchPolicy(strategy="fixed", size=batch),
        measure_runs=runs,
        warmup_runs=warmup,
        generate_config={"top_p": 0.8},
    )


class DecodeBatchExecutorTest(unittest.TestCase):
    def test_measure_configures_dp_then_runs_warmup_and_formal_rounds(self):
        transport = Mock()
        transport.run_round.side_effect = [
            _request_round((128, 128), 16),
            _request_round((128, 128), 16),
            _request_round((128, 128), 16),
        ]
        executor = DecodeBatchExecutor(
            transport,
            engine=EngineConfig(
                max_seq_len=512, max_batch_tokens=512, concurrency_limit=2
            ),
            dp_size=2,
            monotonic=Mock(side_effect=[1.0, 1.1, 2.0, 2.2]),
        )
        measurement = executor.measure(_exact_case(batch=2, warmup=1, runs=2), 2)
        self.assertEqual(2, measurement.batch_size)
        self.assertEqual(2, len(measurement.measurements))
        self.assertEqual(3, transport.run_round.call_count)
        allocations = transport.configure_scheduler.call_args.args[0]
        self.assertEqual([1, 1], [item["local_batch_size"] for item in allocations])
        self.assertEqual(
            [call.configure_scheduler(allocations)]
            + [
                call.run_round(
                    input_lengths=(128, 128),
                    output_length=16,
                    generate_config={"top_p": 0.8},
                )
            ]
            * 3,
            transport.method_calls,
        )
        self.assertEqual([0, 1], [item["run"] for item in measurement.measurements])
        self.assertAlmostEqual(
            320.0, measurement.measurements[0]["throughput_tokens_per_second"]
        )
        self.assertAlmostEqual(
            160.0, measurement.measurements[1]["throughput_tokens_per_second"]
        )

    def test_measure_supports_uneven_and_idle_dp_workers(self):
        for batch, dp, expected in ((10, 3, [4, 3, 3]), (2, 4, [1, 1, 0, 0])):
            with self.subTest(batch=batch, dp=dp):
                transport = Mock()
                transport.run_round.return_value = _request_round((128,) * batch, 16)
                executor = DecodeBatchExecutor(
                    transport,
                    engine=EngineConfig(
                        max_seq_len=512,
                        max_batch_tokens=5120,
                        concurrency_limit=10,
                    ),
                    dp_size=dp,
                )
                measurement = executor.measure(_exact_case(batch=batch), batch)
                self.assertEqual(
                    expected,
                    [
                        item["local_batch_size"]
                        for item in measurement.worker_allocations
                    ],
                )

    def test_measure_slices_inputs_and_checks_worker_limits_not_global_batch(self):
        case = replace(_exact_case(batch=5), input_lengths=(100, 200, 300, 400, 500))
        transport = Mock()
        transport.run_round.return_value = _request_round((100, 200, 300), 16)
        executor = DecodeBatchExecutor(
            transport,
            engine=EngineConfig(
                max_seq_len=1024, max_batch_tokens=300, concurrency_limit=2
            ),
            dp_size=2,
        )
        measurement = executor.measure(case, 3)
        self.assertEqual((100, 200, 300), measurement.input_lengths)
        self.assertEqual(
            [[100, 200], [300]],
            [item["input_lengths"] for item in measurement.worker_allocations],
        )
        self.assertEqual(
            (100, 200, 300), transport.run_round.call_args.kwargs["input_lengths"]
        )

    def test_measure_rejects_capacity_before_calling_transport(self):
        for inputs, batch, concurrency, tokens, message in (
            ((128,), 2, 2, 512, "inputs"),
            ((128, 128, 128), 3, 1, 512, "concurrency_limit"),
            ((200, 200, 100), 3, 2, 300, "max_batch_tokens"),
        ):
            with self.subTest(message=message):
                transport = Mock()
                executor = DecodeBatchExecutor(
                    transport,
                    engine=EngineConfig(
                        max_seq_len=512,
                        max_batch_tokens=tokens,
                        concurrency_limit=concurrency,
                    ),
                    dp_size=2,
                )
                case = replace(_exact_case(batch=batch), input_lengths=inputs)
                with self.assertRaisesRegex(DecodeCapacityError, message) as error:
                    executor.measure(case, batch)
                self.assertIsInstance(error.exception, PlanningError)
                self.assertEqual([], transport.method_calls)

    def test_measure_counts_only_successful_output_tokens_and_retains_failures(self):
        outcomes = _request_round((128, 128), 16)
        outcomes[1].update(success=False, error="HTTP 503")
        transport = Mock()
        transport.run_round.return_value = outcomes
        executor = DecodeBatchExecutor(
            transport,
            engine=EngineConfig(
                max_seq_len=512, max_batch_tokens=512, concurrency_limit=2
            ),
            dp_size=1,
            monotonic=Mock(side_effect=[1.0, 1.5]),
        )
        measurement = executor.measure(_exact_case(batch=2), 2)
        self.assertEqual(
            32.0, measurement.measurements[0]["throughput_tokens_per_second"]
        )
        self.assertEqual(
            "HTTP 503", measurement.measurements[0]["requests"][1]["error"]
        )
        outcomes[0]["latency_ms"]["tpot"] = 100.0
        self.assertEqual(
            1.0, measurement.measurements[0]["requests"][0]["latency_ms"]["tpot"]
        )

    def test_measure_handles_nonpositive_elapsed_time(self):
        for finished in (1.0, 0.9):
            with self.subTest(finished=finished):
                transport = Mock()
                transport.run_round.return_value = _request_round((128,), 16)
                executor = DecodeBatchExecutor(
                    transport,
                    engine=EngineConfig(
                        max_seq_len=512, max_batch_tokens=512, concurrency_limit=2
                    ),
                    dp_size=1,
                    monotonic=Mock(side_effect=[1.0, finished]),
                )
                measurement = executor.measure(_exact_case(batch=1), 1)
                self.assertEqual(
                    0.0, measurement.measurements[0]["throughput_tokens_per_second"]
                )

    def test_measurement_serialization_copies_nested_results(self):
        transport = Mock()
        transport.run_round.return_value = _request_round((128,), 16)
        executor = DecodeBatchExecutor(
            transport,
            engine=EngineConfig(
                max_seq_len=512, max_batch_tokens=512, concurrency_limit=2
            ),
            dp_size=1,
            monotonic=Mock(side_effect=[1.0, 1.5]),
        )
        measurement = executor.measure(_exact_case(batch=1), 1)
        value = measurement.to_dict()
        self.assertEqual(1, value["batch_size"])
        self.assertEqual([128], value["input_lengths"])
        self.assertEqual(32.0, value["measurements"][0]["throughput_tokens_per_second"])
        value["worker_allocations"][0]["input_lengths"][0] = 999
        value["measurements"][0]["requests"][0]["latency_ms"]["tpot"] = 999.0
        self.assertEqual([128], measurement.worker_allocations[0]["input_lengths"])
        self.assertEqual(
            1.0, measurement.measurements[0]["requests"][0]["latency_ms"]["tpot"]
        )


if __name__ == "__main__":
    unittest.main()

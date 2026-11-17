"""Core performance behavior contracts."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, call

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.config_schema import EngineConfig, parse_config
from rtp_llm.test.perf_test.decode.executor import (
    DecodeBatchExecutor,
    DecodeCapacityError,
    DecodeCaseExecutor,
)
from rtp_llm.test.perf_test.decode.model import (
    DecodeBatchMeasurement,
    DecodeCaseSpec,
    ExactBatchPolicy,
)
from rtp_llm.test.perf_test.decode.planner import plan_decode_cases
from rtp_llm.test.perf_test.decode.scheduler import schedule_decode_requests
from rtp_llm.test.perf_test.decode.transport import DecodeHttpTransport
from rtp_llm.test.perf_test.errors import PlanningError
from rtp_llm.test.perf_test.modes import DecodeAdapter, ModeExecutionContext
from rtp_llm.test.perf_test.plan import build_execution_plan
from rtp_llm.test.perf_test.state import RunState
from rtp_llm.test.perf_test.storage import RunStore
from rtp_llm.test.perf_test.tests.support.fixtures import (
    fixture_engine,
    fixture_execution,
    fixture_model,
)


def decode_fixed_config(input_lengths=None, batch_sizes=None, output_length=32):
    return {
        "schema_version": 1,
        "model": fixture_model(),
        "engine": fixture_engine(),
        "workload": {
            "kind": "grid",
            "input_lengths": input_lengths or [128, 256, 512],
            "output_lengths": [output_length],
            "batch": {"strategy": "fixed", "sizes": batch_sizes or [1, 2, 4, 8]},
            "measure_runs": 2,
            "warmup_runs": 1,
            "generate_config": {"top_k": 1},
        },
        "execution": fixture_execution(),
        "output": {},
        "regression": {},
        "publish": {},
    }


def decode_fixed_measurement(case, batch, *, tpot=0.1):
    return DecodeBatchMeasurement(
        batch_size=batch,
        input_lengths=case.input_lengths[:batch],
        worker_allocations=(),
        measurements=tuple(
            (
                {
                    "run": run,
                    "throughput_tokens_per_second": 1000.0,
                    "worker_allocations": [],
                    "requests": [
                        {
                            "request_index": index,
                            "success": True,
                            "input_length": input_length,
                            "output_length": case.output_length,
                            "latency_ms": {
                                "wait": 1.0,
                                "prefill": 2.0,
                                "decode": 3.0,
                                "tpot": tpot,
                                "total": 5.0,
                            },
                        }
                        for (index, input_length) in enumerate(
                            case.input_lengths[:batch]
                        )
                    ],
                }
                for run in range(case.measure_runs)
            )
        ),
    )


def decode_fixed_executor(measure):
    batch_executor = Mock()
    batch_executor.measure.side_effect = measure
    return DecodeCaseExecutor(
        batch_executor, topology={"tp": 1, "dp": 1, "pp": 1, "ep": 1}
    )


def decode_fixed_case(batch=4):
    return DecodeCaseSpec(
        workload_kind="grid",
        input_lengths=(128,) * batch,
        output_length=16,
        batch_policy=ExactBatchPolicy(strategy="fixed", size=batch),
        measure_runs=2,
        warmup_runs=0,
    )


def decode_distribution_executor(batch_runner):

    def measure(spec, batch):
        inputs = spec.input_lengths[:batch]
        raw = batch_runner(
            input_lengths=inputs,
            output_length=spec.output_length,
            measure_runs=spec.measure_runs,
            worker_allocations=[],
        )
        return DecodeBatchMeasurement(batch, inputs, (), tuple(raw["measurements"]))

    batch_executor = Mock()
    batch_executor.measure.side_effect = measure
    return DecodeCaseExecutor(
        batch_executor, topology={"tp": 1, "dp": 1, "pp": 1, "ep": 1}
    )


def decode_distribution_payload():
    spec = DecodeCaseSpec(
        workload_kind="distribution",
        input_lengths=(64, 256),
        output_length=32,
        batch_policy=ExactBatchPolicy("fixed", 2),
        measure_runs=2,
        warmup_runs=0,
        distribution={"source_kind": "sequence_lengths"},
    )
    return {"case_id": "distribution", **spec.to_identity()}


def decode_executor_request_round(input_lengths, output_length):
    return tuple(
        (
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
            for (index, input_length) in enumerate(input_lengths)
        )
    )


def decode_executor_exact_case(*, batch, warmup=0, runs=1):
    return DecodeCaseSpec(
        workload_kind="grid",
        input_lengths=(128,) * batch,
        output_length=16,
        batch_policy=ExactBatchPolicy(strategy="fixed", size=batch),
        measure_runs=runs,
        warmup_runs=warmup,
        generate_config={"top_p": 0.8},
    )


class DecodeDistributionTest(unittest.TestCase):

    def test_fixed_distribution_measures_declared_lengths(self):

        def execute(**kwargs):
            measurements = []
            for run in range(kwargs["measure_runs"]):
                measurements.append(
                    {
                        "run": run,
                        "throughput_tokens_per_second": 100.0,
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
                                    "tpot": 5.0,
                                    "total": 5.0,
                                },
                            }
                            for (index, input_length) in enumerate(
                                kwargs["input_lengths"]
                            )
                        ],
                    }
                )
            return {"measurements": measurements}

        batch_runner = Mock(side_effect=execute)
        result = decode_distribution_executor(batch_runner).execute(
            decode_distribution_payload()
        )
        self.assertEqual("ok", result["status"])
        self.assertEqual(2, result["metrics"]["global_batch_size"])
        self.assertNotIn("search", result["metrics"])
        self.assertEqual(
            [(64, 256)],
            [call.kwargs["input_lengths"] for call in batch_runner.call_args_list],
        )
        self.assertEqual(
            [64, 256],
            result["metrics"]["distribution"]["actual_sequence_lengths"],
        )


class DecodeBatchExecutorTest(unittest.TestCase):

    def test_measure_configures_dp_then_runs_warmup_and_formal_rounds(self):
        transport = Mock()
        transport.run_round.side_effect = [
            decode_executor_request_round((128, 128), 16),
            decode_executor_request_round((128, 128), 16),
            decode_executor_request_round((128, 128), 16),
        ]
        executor = DecodeBatchExecutor(
            transport,
            engine=EngineConfig(
                max_seq_len=512, max_batch_tokens=512, concurrency_limit=2
            ),
            dp_size=2,
            monotonic=Mock(side_effect=[1.0, 1.1, 2.0, 2.2]),
        )
        measurement = executor.measure(
            decode_executor_exact_case(batch=2, warmup=1, runs=2), 2
        )
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

    def test_measure_supports_uniform_dp_and_rejects_idle_workers(self):
        for batch, dp, expected in ((9, 3, [3, 3, 3]), (2, 4, [1, 1, 0, 0])):
            with self.subTest(batch=batch, dp=dp):
                transport = Mock()
                transport.run_round.return_value = decode_executor_request_round(
                    (128,) * batch, 16
                )
                executor = DecodeBatchExecutor(
                    transport,
                    engine=EngineConfig(
                        max_seq_len=512, max_batch_tokens=5120, concurrency_limit=10
                    ),
                    dp_size=dp,
                )
                if batch < dp:
                    with self.assertRaisesRegex(DecodeCapacityError, "positive"):
                        executor.measure(decode_executor_exact_case(batch=batch), batch)
                    self.assertEqual([], transport.method_calls)
                    continue
                measurement = executor.measure(
                    decode_executor_exact_case(batch=batch), batch
                )
                self.assertEqual(
                    expected,
                    [
                        item["local_batch_size"]
                        for item in measurement.worker_allocations
                    ],
                )

    def test_uneven_active_dp_batches_fail_before_measurement_or_profiling(self):
        for batch, dp in ((3, 2), (10, 3), (1, 2), (2, 4)):
            for operation in ("measure", "profile"):
                with self.subTest(batch=batch, dp=dp, operation=operation):
                    transport = Mock()
                    transport.run_round.return_value = decode_executor_request_round(
                        (128,) * batch, 16
                    )
                    executor = DecodeBatchExecutor(
                        transport,
                        engine=EngineConfig(
                            max_seq_len=512, max_batch_tokens=5120, concurrency_limit=10
                        ),
                        dp_size=dp,
                    )
                    case = decode_executor_exact_case(batch=batch)
                    with self.assertRaisesRegex(DecodeCapacityError, "uniform"):
                        if operation == "measure":
                            executor.measure(case, batch)
                        else:
                            executor.profile(
                                case,
                                batch,
                                trace_name="unused",
                                steps=1,
                                enable_all_ranks=False,
                            )
                    self.assertEqual([], transport.method_calls)

    def test_measure_rejects_capacity_before_calling_transport(self):
        for inputs, batch, concurrency, tokens, message in (
            ((128,), 2, 2, 512, "inputs"),
            ((128, 128, 128, 128), 4, 1, 512, "concurrency_limit"),
            ((200, 200, 100, 100), 4, 2, 300, "max_batch_tokens"),
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
                case = replace(
                    decode_executor_exact_case(batch=batch), input_lengths=inputs
                )
                with self.assertRaisesRegex(DecodeCapacityError, message) as error:
                    executor.measure(case, batch)
                self.assertIsInstance(error.exception, PlanningError)
                self.assertEqual([], transport.method_calls)

    def test_measure_counts_only_successful_output_tokens_and_retains_failures(self):
        outcomes = decode_executor_request_round((128, 128), 16)
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
        measurement = executor.measure(decode_executor_exact_case(batch=2), 2)
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


class DecodeFixedTest(unittest.TestCase):

    def test_rank_batch_report_shows_rank_and_global_sizes(self):
        from rtp_llm.test.perf_test.decode.metrics import build_decode_case_result
        from rtp_llm.test.perf_test.decode.report import generate_decode_report

        spec = replace(decode_fixed_case(batch=4), batch_unit="per_dp_rank")
        case = {**spec.to_identity(), "case_id": "decode-rank-grid"}
        allocations = [
            worker.to_dict()
            for worker in schedule_decode_requests(
                spec.input_lengths, dp_size=2
            ).workers
        ]
        selected = decode_fixed_measurement(spec, 4)
        selected = replace(
            selected,
            worker_allocations=tuple(allocations),
            measurements=tuple(
                {**round_data, "worker_allocations": allocations}
                for round_data in selected.measurements
            ),
        )
        result = build_decode_case_result(
            case,
            selected=selected,
            topology={"tp": 1, "dp": 2, "pp": 1, "ep": 2},
            profiling={"status": "disabled", "manifest_path": None, "error": None},
        )
        report = generate_decode_report([result])
        self.assertEqual("formal", report["report_kind"])
        self.assertEqual(2, report["observations"][0]["rank_batch_size"])
        self.assertEqual(4, report["observations"][0]["global_batch_size"])
        self.assertEqual(2, report["matrices"][0]["rank_batch_size"])
        self.assertEqual(4, report["matrices"][0]["global_batch_size"])

    def test_rank_batch_result_rejects_missing_dp_allocation(self):
        from rtp_llm.test.perf_test.decode.metrics import build_decode_case_result
        from rtp_llm.test.perf_test.results import ResultSchemaError, parse_case_result

        spec = replace(decode_fixed_case(batch=4), batch_unit="per_dp_rank")
        case = {**spec.to_identity(), "case_id": "decode-rank-grid"}
        result = build_decode_case_result(
            case,
            selected=decode_fixed_measurement(spec, 4),
            topology={"tp": 1, "dp": 2, "pp": 1, "ep": 2},
            profiling={"status": "disabled", "manifest_path": None, "error": None},
        )
        with self.assertRaisesRegex(ResultSchemaError, "worker_allocations"):
            parse_case_result(0, "decode", result)

    def test_adapter_runs_every_fixed_case_and_checkpoints_through_executor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = parse_config(
                decode_fixed_config(input_lengths=[128, 256], batch_sizes=[1, 2])
            )
            cases = plan_decode_cases(config.workload, config.engine, dp_size=1)
            plan = build_execution_plan(
                mode="decode",
                action="run",
                resolved=resolve_config(
                    decode_fixed_config(input_lengths=[128, 256], batch_sizes=[1, 2])
                ),
                cases=cases,
                result_directory=str(root),
            )
            (root / "checkpoints").mkdir(parents=True)
            (root / "results" / "cases").mkdir(parents=True)
            measure = Mock(side_effect=decode_fixed_measurement)
            adapter = DecodeAdapter(lambda _context: decode_fixed_executor(measure))
            outcome = adapter(
                ModeExecutionContext(plan=plan, session=object(), store=RunStore(root))
            )
            self.assertEqual(RunState.COMPLETED, outcome.state)
            self.assertEqual(4, measure.call_count)
            self.assertEqual(
                {32}, {call.args[0].output_length for call in measure.call_args_list}
            )
            self.assertEqual(4, len(RunStore(root).read_committed_cases()))
            checkpoint = json.loads(
                (root / "checkpoints" / "decode_progress.json").read_text()
            )
            self.assertEqual(
                [case.case_id for case in plan.cases], checkpoint["completed_case_ids"]
            )
            self.assertEqual({"ok"}, set(checkpoint["case_statuses"].values()))

    def test_adapter_converts_case_exception_and_continues_checkpointing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = decode_fixed_config(input_lengths=[128, 256], batch_sizes=[1])
            config = parse_config(source)
            plan = build_execution_plan(
                mode="decode",
                action="run",
                resolved=resolve_config(source),
                cases=plan_decode_cases(config.workload, config.engine, dp_size=1),
                result_directory=temp_dir,
            )

            def measure(spec, batch):
                if spec.input_lengths[0] == 128:
                    raise RuntimeError("transport disconnected")
                return decode_fixed_measurement(spec, batch)

            store = RunStore(Path(temp_dir))
            (Path(temp_dir) / "results" / "cases").mkdir(parents=True)
            (Path(temp_dir) / "checkpoints").mkdir()
            outcome = DecodeAdapter(lambda _context: decode_fixed_executor(measure))(
                ModeExecutionContext(plan=plan, session=object(), store=store)
            )
            self.assertEqual(RunState.PARTIAL, outcome.state)
            self.assertEqual(
                ["request_failed", "ok"],
                [item["status"] for item in outcome.case_results],
            )
            self.assertIsNone(outcome.case_results[0]["metrics"])
            self.assertEqual(
                "transport disconnected",
                outcome.case_results[0]["errors"][0]["message"],
            )
            self.assertEqual(2, len(store.read_committed_cases()))
            checkpoint = json.loads(
                (Path(temp_dir) / "checkpoints" / "decode_progress.json").read_text()
            )
            self.assertEqual(2, len(checkpoint["completed_case_ids"]))


class DecodeTransportTest(unittest.TestCase):

    def test_scheduler_rejects_uneven_active_batches_before_http(self):
        post = Mock(return_value=Mock(status_code=200, json=lambda: {"status": "ok"}))
        transport = DecodeHttpTransport(
            "http://unused", prompt_for_length=str, post=post
        )
        with self.assertRaisesRegex(PlanningError, "uniform"):
            transport.configure_scheduler(
                ({"local_batch_size": 2}, {"local_batch_size": 1})
            )
        post.assert_not_called()

    def test_scheduler_configures_uniform_batches_and_rejects_idle_workers(self):
        for local_batches, expected in (([2, 2], 2), ([1, 1, 0, 0], 1)):
            with self.subTest(local_batches=local_batches):
                post = Mock(
                    return_value=Mock(status_code=200, json=lambda: {"status": "ok"})
                )
                transport = DecodeHttpTransport(
                    "http://unused", prompt_for_length=str, post=post
                )
                if 0 in local_batches:
                    with self.assertRaisesRegex(PlanningError, "positive"):
                        transport.configure_scheduler(
                            tuple({"local_batch_size": size} for size in local_batches)
                        )
                    post.assert_not_called()
                    continue
                transport.configure_scheduler(
                    tuple({"local_batch_size": size} for size in local_batches)
                )
                self.assertEqual(
                    {"batch_size": expected, "mode": "decode"},
                    post.call_args.kwargs["json"],
                )


class DecodeSchedulerTest(unittest.TestCase):

    def test_even_and_uneven_batches_cover_each_request_exactly_once(self):
        for global_batch, dp_size, local_batches in (
            (8, 2, [4, 4]),
            (5, 2, [3, 2]),
            (2, 4, [1, 1, 0, 0]),
        ):
            with self.subTest(global_batch=global_batch, dp_size=dp_size):
                lengths = tuple((100 + index for index in range(global_batch)))
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


if __name__ == "__main__":
    unittest.main()

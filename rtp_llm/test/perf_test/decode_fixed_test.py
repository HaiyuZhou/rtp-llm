import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.config_schema import ConfigError, parse_config
from rtp_llm.test.perf_test.decode.executor import DecodeCaseExecutor
from rtp_llm.test.perf_test.decode.model import DecodeBatchMeasurement
from rtp_llm.test.perf_test.decode.planner import plan_decode_cases
from rtp_llm.test.perf_test.errors import PlanningError
from rtp_llm.test.perf_test.modes.base import ModeExecutionContext
from rtp_llm.test.perf_test.modes.decode import DecodeAdapter
from rtp_llm.test.perf_test.plan import build_execution_plan
from rtp_llm.test.perf_test.state import RunState
from rtp_llm.test.perf_test.storage import RunStore


def _config(input_lengths=None, batch_sizes=None, output_length=32):
    return {
        "schema_version": 1,
        "model": {"label": "test", "type": "registered", "checkpoint": "/m"},
        "engine": {
            "max_seq_len": 4096,
            "max_batch_tokens": 8192,
            "concurrency_limit": 8,
        },
        "workload": {
            "kind": "grid",
            "input_lengths": input_lengths or [128, 256, 512],
            "output_lengths": [output_length],
            "batch": {"strategy": "fixed", "sizes": batch_sizes or [1, 2, 4, 8]},
            "measure_runs": 2,
            "warmup_runs": 1,
            "generate_config": {"top_k": 1},
        },
        "execution": {"backend": "local", "devices": [0]},
        "output": {},
        "regression": {},
        "publish": {},
    }


def _measurement(case, batch, *, tpot=0.1):
    return DecodeBatchMeasurement(
        batch_size=batch,
        input_lengths=case.input_lengths[:batch],
        worker_allocations=(),
        measurements=tuple(
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
                    for index, input_length in enumerate(case.input_lengths[:batch])
                ],
            }
            for run in range(case.measure_runs)
        ),
    )


def _executor(measure):
    batch_executor = Mock()
    batch_executor.measure.side_effect = measure
    return DecodeCaseExecutor(
        batch_executor, topology={"tp": 1, "dp": 1, "pp": 1, "ep": 1}
    )


def _tpot_payload(target):
    source = _config(input_lengths=[128], batch_sizes=[8])
    source["engine"]["concurrency_limit"] = 16
    source["workload"]["batch"] = {
        "strategy": "target_tpot",
        "max_size": 16,
        "candidate_step": 4,
        "target_tpot_ms": target,
    }
    config = parse_config(source)
    case = plan_decode_cases(config.workload, config.engine, dp_size=1)[0]
    return {"case_id": "search", **case}


class DecodeFixedTest(unittest.TestCase):
    def test_three_by_four_grid_has_twelve_deterministic_scalar_output_cases(self):
        config = parse_config(_config())

        first = plan_decode_cases(config.workload, config.engine, dp_size=1)
        second = plan_decode_cases(config.workload, config.engine, dp_size=1)

        self.assertEqual(first, second)
        self.assertEqual(12, len(first))
        self.assertEqual({32}, {case["output_length"] for case in first})
        self.assertEqual(
            {
                (input_length, batch)
                for input_length in (128, 256, 512)
                for batch in (1, 2, 4, 8)
            },
            {
                (case["input_lengths"][0], case["batch_policy"]["size"])
                for case in first
            },
        )

    def test_nested_output_arrays_and_ranges_are_rejected(self):
        for output_length in ([16, 32], {"min": 16, "max": 32}):
            with self.subTest(output_length=output_length):
                with self.assertRaisesRegex(ConfigError, "workload.output_lengths"):
                    parse_config(_config(output_length=output_length))

    def test_known_sequence_and_batch_capacity_fail_before_execution(self):
        sequence = parse_config(_config(input_lengths=[4080], output_length=32))
        with self.assertRaisesRegex(PlanningError, "max_seq_len"):
            plan_decode_cases(sequence.workload, sequence.engine, dp_size=1)

        batch = parse_config(_config(input_lengths=[2048], batch_sizes=[8]))
        with self.assertRaisesRegex(PlanningError, "max_batch_tokens"):
            plan_decode_cases(batch.workload, batch.engine, dp_size=1)

    def test_adapter_runs_every_fixed_case_and_checkpoints_through_executor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = parse_config(_config(input_lengths=[128, 256], batch_sizes=[1, 2]))
            cases = plan_decode_cases(config.workload, config.engine, dp_size=1)
            plan = build_execution_plan(
                mode="decode",
                action="run",
                resolved=resolve_config(
                    _config(input_lengths=[128, 256], batch_sizes=[1, 2])
                ),
                cases=cases,
                result_directory=str(root),
            )
            (root / "checkpoints").mkdir(parents=True)
            (root / "results" / "cases").mkdir(parents=True)
            measure = Mock(side_effect=_measurement)
            adapter = DecodeAdapter(lambda _context: _executor(measure))

            outcome = adapter(
                ModeExecutionContext(plan=plan, session=object(), store=RunStore(root))
            )

            self.assertEqual(RunState.COMPLETED, outcome.state)
            self.assertEqual(4, measure.call_count)
            self.assertEqual(
                {32},
                {call.args[0].output_length for call in measure.call_args_list},
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
            source = _config(input_lengths=[128, 256], batch_sizes=[1])
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
                return _measurement(spec, batch)

            store = RunStore(Path(temp_dir))
            (Path(temp_dir) / "results" / "cases").mkdir(parents=True)
            (Path(temp_dir) / "checkpoints").mkdir()
            outcome = DecodeAdapter(lambda _context: _executor(measure))(
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

    def test_target_tpot_executes_bounded_search_and_preserves_trajectory(self):
        measure = Mock(
            side_effect=lambda case, batch: _measurement(
                case, batch, tpot=8.0 if batch <= 8 else 12.0
            )
        )
        result = _executor(measure).execute(_tpot_payload(10.0))

        search = result["metrics"]["search"]
        self.assertEqual("found", search["outcome"])
        self.assertEqual(8, search["best_batch_size"])
        self.assertEqual(
            [8, 12], [item["batch_size"] for item in search["observations"]]
        )
        self.assertEqual(
            [8, 12],
            [call.args[1] for call in measure.call_args_list],
        )
        self.assertEqual(8, result["metrics"]["global_batch_size"])
        self.assertEqual(16, result["metrics"]["successful_requests"])

    def test_target_tpot_without_satisfying_batch_is_not_successful(self):
        result = _executor(_measurement).execute(_tpot_payload(0.05))

        self.assertEqual("target_tpot_not_met", result["status"])
        self.assertIsNone(result["metrics"])
        self.assertEqual(
            "no_satisfying_candidate",
            result["errors"][0]["details"]["search"]["outcome"],
        )
        self.assertEqual("target_tpot_not_met", result["errors"][-1]["code"])


if __name__ == "__main__":
    unittest.main()

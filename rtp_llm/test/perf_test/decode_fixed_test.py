import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.config_schema import ConfigError, parse_config
from rtp_llm.test.perf_test.decode.fixed import FixedDecodeCaseRunner
from rtp_llm.test.perf_test.decode.planner import plan_decode_grid_cases
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
            "output_length": output_length,
            "batch_sizes": batch_sizes or [1, 2, 4, 8],
            "measure_runs": 2,
            "warmup_runs": 1,
            "generate_config": {"top_k": 1},
        },
        "execution": {"backend": "local", "devices": [0]},
        "output": {},
        "regression": {},
        "publish": {},
    }


def _raw_batch(**kwargs):
    return {
        "measurements": [
            {
                "run": run,
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
                    for index, input_length in enumerate(kwargs["input_lengths"])
                ],
            }
            for run in range(kwargs["measure_runs"])
        ]
    }


class DecodeFixedTest(unittest.TestCase):
    def test_three_by_four_grid_has_twelve_deterministic_scalar_output_cases(self):
        config = parse_config(_config())

        first = plan_decode_grid_cases(config.workload, config.engine, dp_size=1)
        second = plan_decode_grid_cases(config.workload, config.engine, dp_size=1)

        self.assertEqual(first, second)
        self.assertEqual(12, len(first))
        self.assertEqual({32}, {case["output_length"] for case in first})
        self.assertEqual(
            {
                (input_length, batch)
                for input_length in (128, 256, 512)
                for batch in (1, 2, 4, 8)
            },
            {(case["input_length"], case["global_batch_size"]) for case in first},
        )

    def test_output_arrays_and_ranges_remain_rejected(self):
        for output_length in ([16, 32], {"min": 16, "max": 32}):
            with self.subTest(output_length=output_length):
                with self.assertRaisesRegex(ConfigError, "workload.output_length"):
                    parse_config(_config(output_length=output_length))

    def test_known_sequence_and_batch_capacity_fail_before_execution(self):
        sequence = parse_config(_config(input_lengths=[4080], output_length=32))
        with self.assertRaisesRegex(PlanningError, "max_seq_len"):
            plan_decode_grid_cases(sequence.workload, sequence.engine, dp_size=1)

        batch = parse_config(_config(input_lengths=[2048], batch_sizes=[8]))
        with self.assertRaisesRegex(PlanningError, "max_batch_tokens"):
            plan_decode_grid_cases(batch.workload, batch.engine, dp_size=1)

    def test_adapter_runs_every_fixed_case_once_through_scalar_runner(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = parse_config(_config(input_lengths=[128, 256], batch_sizes=[1, 2]))
            cases = plan_decode_grid_cases(config.workload, config.engine, dp_size=1)
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
            batch_runner = Mock(side_effect=_raw_batch)
            runner = FixedDecodeCaseRunner(batch_runner)
            adapter = DecodeAdapter(lambda _context: runner)

            outcome = adapter(
                ModeExecutionContext(plan=plan, session=object(), store=RunStore(root))
            )

            self.assertEqual(RunState.COMPLETED, outcome.state)
            self.assertEqual(4, batch_runner.call_count)
            self.assertEqual(
                {32},
                {call.kwargs["output_length"] for call in batch_runner.call_args_list},
            )
            self.assertEqual(4, len(RunStore(root).read_committed_cases()))


if __name__ == "__main__":
    unittest.main()

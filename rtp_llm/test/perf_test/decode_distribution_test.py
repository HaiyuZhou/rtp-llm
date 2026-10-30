import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from rtp_llm.test.perf_test.config_resolver import load_resolved_config
from rtp_llm.test.perf_test.config_schema import parse_config
from rtp_llm.test.perf_test.decode.executor import DecodeCaseExecutor
from rtp_llm.test.perf_test.decode.model import (
    DecodeBatchMeasurement,
    DecodeCaseSpec,
    ExactBatchPolicy,
    TargetTPOTPolicy,
)
from rtp_llm.test.perf_test.decode.planner import plan_decode_cases
from rtp_llm.test.perf_test.errors import PlanningError


def _source(distribution_file="requests.json"):
    return {
        "schema_version": 1,
        "model": {"label": "test", "type": "registered", "checkpoint": "/m"},
        "engine": {
            "max_seq_len": 4096,
            "max_batch_tokens": 8192,
            "concurrency_limit": 8,
        },
        "workload": {
            "kind": "distribution",
            "distribution_file": distribution_file,
            "output_length": 32,
            "batch": {"strategy": "fixed", "sizes": [2, 3]},
            "measure_runs": 2,
            "warmup_runs": 1,
            "generate_config": {"top_k": 1},
        },
        "execution": {"backend": "local", "devices": [0]},
        "output": {},
        "regression": {},
        "publish": {},
    }


def _executor(batch_runner):
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


def _payload(*, search=False):
    spec = DecodeCaseSpec(
        workload_kind="distribution",
        input_lengths=(64, 128, 256, 512) if search else (64, 256),
        output_length=32,
        batch_policy=(
            TargetTPOTPolicy("target_tpot", 4, 2, 10.0)
            if search
            else ExactBatchPolicy("fixed", 2)
        ),
        measure_runs=1 if search else 2,
        warmup_runs=0,
        distribution={"source_kind": "sequence_lengths"},
    )
    return {"case_id": "distribution", **spec.to_identity()}


class DecodeDistributionTest(unittest.TestCase):
    def test_materialized_sequence_lengths_generate_deterministic_actual_cases(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "requests.json").write_text(
                json.dumps({"sequence_lengths": [64, 128, 256, 512]}),
                encoding="utf-8",
            )
            config_path = root / "perf.json"
            config_path.write_text(json.dumps(_source()), encoding="utf-8")
            resolved = load_resolved_config(config_path)
            parsed = parse_config(_source())

            first = plan_decode_cases(
                parsed.workload,
                parsed.engine,
                dp_size=1,
                resolved_workload=resolved.resolved_config["workload"],
            )
            second = plan_decode_cases(
                parsed.workload,
                parsed.engine,
                dp_size=1,
                resolved_workload=resolved.resolved_config["workload"],
            )

            self.assertEqual(first, second)
            self.assertEqual(
                [[64, 128], [64, 128, 256]], [case["input_lengths"] for case in first]
            )
            self.assertEqual([2, 3], [case["batch_policy"]["size"] for case in first])
            self.assertEqual({32}, {case["output_length"] for case in first})

    def test_legacy_batch_map_is_sorted_and_validated_against_config(self):
        parsed = parse_config(_source())
        workload = {
            **parsed.workload.__dict__,
            "distribution": {
                "batch_seq_len_map": {"3": [64, 128, 256], "2": [512, 1024]}
            },
        }
        workload.pop("distribution_file")

        cases = plan_decode_cases(
            parsed.workload, parsed.engine, dp_size=1, resolved_workload=workload
        )

        self.assertEqual(
            [[512, 1024], [64, 128, 256]], [case["input_lengths"] for case in cases]
        )

    def test_malformed_empty_and_invalid_distributions_fail_planning(self):
        parsed = parse_config(_source())
        invalid = [
            [],
            {},
            {"sequence_lengths": []},
            {"sequence_lengths": [64, 0]},
            {"sequence_lengths": [64, True, 128]},
            {"sequence_lengths": [64, 128]},
            {"batch_seq_len_map": {"2": [64], "3": [64, 128, 256]}},
            {"batch_seq_len_map": {"2": [64, 128]}},
        ]
        for distribution in invalid:
            with self.subTest(distribution=distribution):
                workload = {
                    **parsed.workload.__dict__,
                    "distribution": distribution,
                }
                workload.pop("distribution_file")
                with self.assertRaises(PlanningError):
                    plan_decode_cases(
                        parsed.workload,
                        parsed.engine,
                        dp_size=1,
                        resolved_workload=workload,
                    )

    def test_executor_preserves_actual_lengths_and_request_outcomes(self):
        def execute(**kwargs):
            measurements = []
            for run in range(kwargs["measure_runs"]):
                requests = []
                for index, input_length in enumerate(kwargs["input_lengths"]):
                    request = {
                        "request_index": index,
                        "success": index == 0,
                        "input_length": input_length,
                        "output_length": kwargs["output_length"],
                    }
                    if request["success"]:
                        request["latency_ms"] = {
                            "wait": 1.0,
                            "prefill": 2.0,
                            "decode": 3.0,
                            "tpot": 0.1,
                            "total": 5.0,
                        }
                    else:
                        request["error"] = "failed"
                    requests.append(request)
                measurements.append(
                    {
                        "run": run,
                        "throughput_tokens_per_second": 100.0,
                        "worker_allocations": kwargs["worker_allocations"],
                        "requests": requests,
                    }
                )
            return {"measurements": measurements}

        batch_runner = Mock(side_effect=execute)
        result = _executor(batch_runner).execute(_payload())

        self.assertEqual([64, 256], result["metrics"]["input_lengths"])
        self.assertEqual(
            [True, False],
            [
                item["success"]
                for item in result["metrics"]["measurements"][0]["requests"]
            ],
        )
        self.assertEqual((64, 256), batch_runner.call_args.kwargs["input_lengths"])
        self.assertEqual(32, batch_runner.call_args.kwargs["output_length"])
        self.assertEqual(2, result["metrics"]["successful_requests"])
        self.assertEqual(2, result["metrics"]["failed_requests"])
        self.assertEqual(
            [64, 256], result["metrics"]["distribution"]["actual_sequence_lengths"]
        )

    def test_target_tpot_searches_distribution_prefix_batches(self):
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
                            for index, input_length in enumerate(
                                kwargs["input_lengths"]
                            )
                        ],
                    }
                )
            return {"measurements": measurements}

        batch_runner = Mock(side_effect=execute)
        result = _executor(batch_runner).execute(_payload(search=True))

        self.assertEqual(4, result["metrics"]["search"]["best_batch_size"])
        self.assertEqual(
            [(64, 128), (64, 128, 256, 512)],
            [call.kwargs["input_lengths"] for call in batch_runner.call_args_list],
        )
        self.assertEqual(
            [64, 128, 256, 512],
            result["metrics"]["distribution"]["actual_sequence_lengths"],
        )


if __name__ == "__main__":
    unittest.main()

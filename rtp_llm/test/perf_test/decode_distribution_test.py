import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from rtp_llm.test.perf_test.config_resolver import load_resolved_config
from rtp_llm.test.perf_test.config_schema import parse_config
from rtp_llm.test.perf_test.decode.distribution import (
    DistributionDecodeCaseRunner,
    plan_decode_distribution_cases,
)
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
            "batch_sizes": [2, 3],
            "measure_runs": 2,
            "warmup_runs": 1,
            "generate_config": {"top_k": 1},
        },
        "execution": {"backend": "local", "devices": [0]},
        "output": {},
        "regression": {},
        "publish": {},
    }


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

            first = plan_decode_distribution_cases(
                resolved.resolved_config["workload"], parsed.engine, dp_size=1
            )
            second = plan_decode_distribution_cases(
                resolved.resolved_config["workload"], parsed.engine, dp_size=1
            )

            self.assertEqual(first, second)
            self.assertEqual(
                [(64, 128), (64, 128, 256)], [case["input_lengths"] for case in first]
            )
            self.assertEqual([2, 3], [case["global_batch_size"] for case in first])
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

        cases = plan_decode_distribution_cases(workload, parsed.engine, dp_size=1)

        self.assertEqual(
            [(512, 1024), (64, 128, 256)], [case["input_lengths"] for case in cases]
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
                    plan_decode_distribution_cases(workload, parsed.engine, dp_size=1)

    def test_runner_preserves_actual_lengths_and_request_outcomes(self):
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
        case = {
            "case_id": "decode-case",
            "workload_kind": "distribution",
            "input_lengths": (64, 256),
            "output_length": 32,
            "global_batch_size": 2,
            "measure_runs": 2,
            "warmup_runs": 1,
            "generate_config": {"top_k": 1},
            "target_tpot_ms": None,
            "distribution": {"source_kind": "sequence_lengths"},
        }

        result = DistributionDecodeCaseRunner(batch_runner).run_case(case)

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


if __name__ == "__main__":
    unittest.main()

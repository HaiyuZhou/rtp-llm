import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.decode.finalize import finalize_decode_run
from rtp_llm.test.perf_test.decode.metrics import build_decode_case_result
from rtp_llm.test.perf_test.decode.tpot import search_target_tpot
from rtp_llm.test.perf_test.plan import build_execution_plan
from rtp_llm.test.perf_test.results import parse_result_envelope
from rtp_llm.test.perf_test.run_directory import initialize_run_directory

TOPOLOGY = {"tp": 2, "dp": 2, "pp": 1, "ep": 1}


def _case():
    return {
        "case_id": "decode-case",
        "workload_kind": "distribution",
        "input_lengths": (64, 128),
        "output_length": 4,
        "global_batch_size": 2,
        "measure_runs": 1,
        "warmup_runs": 0,
        "generate_config": {},
        "target_tpot_ms": 10.0,
        "distribution": {"source_kind": "sequence_lengths"},
    }


def _request(index, input_length, *, success=True):
    value = {
        "request_index": index,
        "success": success,
        "input_length": input_length,
        "output_length": 4,
    }
    if success:
        value["latency_ms"] = {
            "wait": 1.0 + index,
            "prefill": 2.0 + index,
            "decode": 8.0 + index,
            "tpot": 2.0 + index,
            "total": 11.0 + index,
        }
    else:
        value["error"] = "request failed"
    return value


def _measurements(second_success=True):
    return [
        {
            "run": 0,
            "throughput_tokens_per_second": 640.0,
            "requests": [
                _request(0, 64),
                _request(1, 128, success=second_success),
            ],
            "worker_allocations": [
                {"worker_index": 0, "request_indices": [0]},
                {"worker_index": 1, "request_indices": [1]},
            ],
        }
    ]


def _source():
    return {
        "schema_version": 1,
        "model": {
            "label": "test",
            "type": "registered",
            "checkpoint": "/m",
            "topology": TOPOLOGY,
        },
        "engine": {
            "max_seq_len": 4096,
            "max_batch_tokens": 8192,
            "concurrency_limit": 2,
        },
        "workload": {
            "kind": "grid",
            "input_lengths": [64],
            "output_length": 4,
            "batch_sizes": [2],
        },
        "execution": {"backend": "local", "devices": [0, 1, 2, 3]},
        "output": {},
        "regression": {},
        "publish": {},
    }


class DecodeMetricsTest(unittest.TestCase):
    def test_valid_and_partially_failed_batches_preserve_all_metrics(self):
        valid = build_decode_case_result(
            _case(), measurements=_measurements(), topology=TOPOLOGY
        )

        self.assertEqual("ok", valid["status"])
        self.assertEqual(2, valid["metrics"]["successful_requests"])
        self.assertEqual(0, valid["metrics"]["failed_requests"])
        self.assertEqual(
            {"wait": 1.5, "prefill": 2.5, "decode": 8.5, "tpot": 2.5, "total": 11.5},
            valid["metrics"]["latency_ms"],
        )
        self.assertEqual(640.0, valid["metrics"]["throughput_tokens_per_second"])

        partial = build_decode_case_result(
            _case(), measurements=_measurements(second_success=False), topology=TOPOLOGY
        )

        self.assertEqual("request_failed", partial["status"])
        self.assertEqual(1, partial["metrics"]["successful_requests"])
        self.assertEqual(1, partial["metrics"]["failed_requests"])
        self.assertFalse(
            partial["metrics"]["measurements"][0]["requests"][1]["success"]
        )

    def test_shape_and_timing_failures_are_classified(self):
        shape = _measurements()
        shape[0]["requests"][1]["output_length"] = 5
        result = build_decode_case_result(
            _case(), measurements=shape, topology=TOPOLOGY
        )
        self.assertEqual("invalid_shape", result["status"])
        self.assertIsNone(result["metrics"])

        timing = _measurements()
        timing[0]["requests"][0]["latency_ms"]["decode"] = -1.0
        result = build_decode_case_result(
            _case(), measurements=timing, topology=TOPOLOGY
        )
        self.assertEqual("invalid_timing", result["status"])
        self.assertIsNone(result["metrics"])

    def test_finalize_round_trip_preserves_distribution_and_search_trajectory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            case = _case()
            plan = build_execution_plan(
                mode="decode",
                action="run",
                resolved=resolve_config(_source()),
                cases=[{key: value for key, value in case.items() if key != "case_id"}],
                result_directory=str(root),
            )
            case["case_id"] = plan.cases[0].case_id
            search = search_target_tpot(
                lambda batch: {
                    "tpot_ms": float(batch),
                    "success_rate": 1.0,
                    "throughput_tokens_per_second": batch * 100.0,
                },
                target_tpot_ms=10.0,
                max_batch_size=16,
            ).to_dict()
            result = build_decode_case_result(
                case,
                measurements=_measurements(),
                topology=TOPOLOGY,
                search=search,
                distribution={
                    "source_kind": "sequence_lengths",
                    "actual_sequence_lengths": [64, 128],
                },
            )
            initialize_run_directory(
                root,
                source_config=_source(),
                plan=plan,
                run_id="decode-run",
                code_version="rev",
            )

            envelope = finalize_decode_run(
                root, plan=plan, run_id="decode-run", case_results=[result]
            )
            parsed = parse_result_envelope(envelope)

            metrics = parsed.cases[0].metrics.to_dict()
            self.assertEqual([64, 128], metrics["input_lengths"])
            self.assertEqual(4, metrics["output_length"])
            self.assertEqual(2, metrics["global_batch_size"])
            self.assertEqual(TOPOLOGY, metrics["topology"])
            self.assertEqual(
                [64, 128], metrics["distribution"]["actual_sequence_lengths"]
            )
            self.assertEqual(
                [8, 12],
                [item["batch_size"] for item in metrics["search"]["observations"]],
            )
            self.assertTrue((root / "results" / "result.json").is_file())
            self.assertTrue((root / "results" / "summary.json").is_file())
            report = root / "reports" / "decode_report.json"
            self.assertEqual(
                "basic", __import__("json").loads(report.read_text())["report_kind"]
            )


if __name__ == "__main__":
    unittest.main()

import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.decode.finalize import finalize_decode_run
from rtp_llm.test.perf_test.decode.metrics import build_decode_case_result
from rtp_llm.test.perf_test.decode.model import (
    DecodeBatchMeasurement,
    DecodeCaseSpec,
    ExactBatchPolicy,
)
from rtp_llm.test.perf_test.plan import build_execution_plan
from rtp_llm.test.perf_test.results import ResultSchemaError, parse_result_envelope
from rtp_llm.test.perf_test.run_directory import initialize_run_directory

TOPOLOGY = {"tp": 2, "dp": 2, "pp": 1, "ep": 1}


def _case():
    spec = DecodeCaseSpec(
        workload_kind="distribution",
        input_lengths=(64, 128),
        output_length=4,
        batch_policy=ExactBatchPolicy("fixed", 2),
        measure_runs=1,
        warmup_runs=0,
        distribution={"source_kind": "sequence_lengths"},
    )
    return {"case_id": "decode-case", **spec.to_identity()}


def _selected(measurements=None):
    return DecodeBatchMeasurement(
        batch_size=2,
        input_lengths=(64, 128),
        worker_allocations=(),
        measurements=tuple(_measurements() if measurements is None else measurements),
    )


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
            "output_lengths": [4],
            "batch": {"strategy": "fixed", "sizes": [2]},
        },
        "execution": {"backend": "local", "devices": [0, 1, 2, 3]},
        "output": {},
        "regression": {},
        "publish": {},
    }


class DecodeMetricsTest(unittest.TestCase):
    def test_valid_and_partially_failed_batches_preserve_all_metrics(self):
        valid = build_decode_case_result(
            _case(), selected=_selected(), topology=TOPOLOGY
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
            _case(),
            selected=_selected(_measurements(second_success=False)),
            topology=TOPOLOGY,
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
            _case(), selected=_selected(shape), topology=TOPOLOGY
        )
        self.assertEqual("invalid_shape", result["status"])
        self.assertIsNone(result["metrics"])

        timing = _measurements()
        timing[0]["requests"][0]["latency_ms"]["decode"] = -1.0
        result = build_decode_case_result(
            _case(), selected=_selected(timing), topology=TOPOLOGY
        )
        self.assertEqual("invalid_timing", result["status"])
        self.assertIsNone(result["metrics"])

    def test_finalize_round_trip_preserves_distribution_and_search_trajectory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            case = _case()
            case["input_lengths"] = [64, 128, 256, 512]
            case["batch_policy"] = {
                "strategy": "target_tpot",
                "max_size": 4,
                "candidate_step": 2,
                "target_tpot_ms": 10.0,
            }
            case["profiling"]["enabled"] = True
            plan = build_execution_plan(
                mode="decode",
                action="run",
                resolved=resolve_config(_source()),
                cases=[{key: value for key, value in case.items() if key != "case_id"}],
                result_directory=str(root),
            )
            case["case_id"] = plan.cases[0].case_id
            search = {
                "target_tpot_ms": 10.0,
                "max_size": 4,
                "candidate_step": 2,
                "outcome": "found",
                "best_batch_size": 2,
                "observations": [
                    {"batch_size": 2, "outcome": "satisfied"},
                    {"batch_size": 4, "outcome": "target_exceeded"},
                ],
            }
            result = build_decode_case_result(
                case,
                selected=_selected(),
                topology=TOPOLOGY,
                search=search,
                profiling={
                    "status": "failed",
                    "manifest_path": "profiles/test/profile_manifest.json",
                    "error": "trace flush timeout",
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
                [2, 4],
                [item["batch_size"] for item in metrics["search"]["observations"]],
            )
            self.assertTrue((root / "results" / "result.json").is_file())
            self.assertTrue((root / "results" / "summary.json").is_file())
            report = root / "reports" / "decode_report.json"
            self.assertEqual("formal", json.loads(report.read_text())["report_kind"])
            self.assertEqual("completed", envelope["status"])
            self.assertEqual(1, envelope["summary"]["profiling_failed_cases"])
            self.assertEqual(
                1,
                json.loads((root / "results" / "summary.json").read_text())[
                    "profiling_failed_cases"
                ],
            )
            paths = [
                root / "results" / "result.json",
                root / "results" / "summary.json",
                report,
                root / "results" / "cases" / f"{case['case_id']}.json",
            ]
            before = {path: path.read_bytes() for path in paths}
            invalid = copy.deepcopy(result)
            invalid["metrics"]["profiling"]["manifest_path"] = None
            with self.assertRaises(ResultSchemaError):
                finalize_decode_run(
                    root, plan=plan, run_id="decode-run", case_results=[invalid]
                )
            self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_finalize_failed_search_is_partial_and_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            case = _case()
            case["batch_policy"] = {
                "strategy": "target_tpot",
                "max_size": 2,
                "candidate_step": 1,
                "target_tpot_ms": 1.0,
            }
            plan = build_execution_plan(
                mode="decode",
                action="run",
                resolved=resolve_config(_source()),
                cases=[{key: value for key, value in case.items() if key != "case_id"}],
                result_directory=str(root),
            )
            identity = {key: value for key, value in case.items() if key != "case_id"}
            search = {
                "target_tpot_ms": 1.0,
                "max_size": 2,
                "candidate_step": 1,
                "outcome": "no_satisfying_candidate",
                "best_batch_size": None,
                "observations": [
                    {
                        "batch_size": 1,
                        "outcome": "target_exceeded",
                        "tpot_ms": 2.0,
                        "success_rate": 1.0,
                    }
                ],
            }
            result = {
                "case_id": plan.cases[0].case_id,
                "identity": identity,
                "status": "target_tpot_not_met",
                "runs": 1,
                "metrics": None,
                "errors": [
                    {
                        "code": "target_tpot_not_met",
                        "message": "target exceeded",
                        "details": {"search": search},
                    }
                ],
            }
            initialize_run_directory(
                root,
                source_config=_source(),
                plan=plan,
                run_id="decode-failed",
                code_version="rev",
            )
            invalid = copy.deepcopy(result)
            invalid["errors"][0]["details"]["search"]["candidate_step"] = 0
            with self.assertRaises(ResultSchemaError):
                finalize_decode_run(
                    root, plan=plan, run_id="decode-failed", case_results=[invalid]
                )
            self.assertFalse((root / "results" / "result.json").exists())
            self.assertEqual([], list((root / "results" / "cases").iterdir()))
            self.assertFalse((root / "reports" / "decode_report.json").exists())
            envelope = finalize_decode_run(
                root, plan=plan, run_id="decode-failed", case_results=[result]
            )
            self.assertEqual("partial", envelope["status"])
            self.assertEqual(
                {
                    "planned_cases": 1,
                    "successful_cases": 0,
                    "failed_cases": 1,
                    "profiling_failed_cases": 0,
                },
                envelope["summary"],
            )
            self.assertIsNone(envelope["cases"][0]["metrics"])
            report = json.loads((root / "reports" / "decode_report.json").read_text())
            self.assertEqual("diagnostic", report["report_kind"])
            self.assertEqual(search, report["distribution_cases"][0]["search"])

    def test_metrics_count_every_formal_round_and_use_selected_shape(self):
        case = _case()
        case["measure_runs"] = 2
        case["input_lengths"] = [64, 128, 256, 512]
        case["batch_policy"] = {
            "strategy": "target_tpot",
            "max_size": 4,
            "candidate_step": 2,
            "target_tpot_ms": 10.0,
        }
        measurements = _measurements() + _measurements(second_success=False)
        measurements[1]["run"] = 1
        selected = _selected(measurements)
        result = build_decode_case_result(case, selected=selected, topology=TOPOLOGY)
        self.assertEqual("request_failed", result["status"])
        metrics = result["metrics"]
        self.assertEqual(2, metrics["global_batch_size"])
        self.assertEqual([64, 128], metrics["input_lengths"])
        self.assertEqual(3, metrics["successful_requests"])
        self.assertEqual(1, metrics["failed_requests"])
        self.assertEqual(2, len(metrics["measurements"]))
        self.assertEqual([64, 128], metrics["distribution"]["actual_sequence_lengths"])
        measurements[0]["requests"][0]["latency_ms"]["tpot"] = 999.0
        self.assertEqual(
            2.0, metrics["measurements"][0]["requests"][0]["latency_ms"]["tpot"]
        )

    def test_selected_batch_and_every_request_shape_are_validated(self):
        for selected in (
            replace(_selected(), batch_size=3),
            replace(_selected(), input_lengths=(64, 256)),
            replace(_selected(), measurements=()),
        ):
            with self.subTest(selected=selected):
                result = build_decode_case_result(
                    _case(), selected=selected, topology=TOPOLOGY
                )
                self.assertEqual("invalid_shape", result["status"])
                self.assertIsNone(result["metrics"])

    def test_result_validation_failure_preserves_independent_search_details(self):
        search = {
            "outcome": "found",
            "best_batch_size": 2,
            "observations": [{"batch_size": 2, "outcome": "satisfied"}],
        }
        for invalid in ("shape", "timing", "topology"):
            with self.subTest(invalid=invalid):
                measurements = _measurements()
                topology = TOPOLOGY
                if invalid == "shape":
                    measurements[0]["requests"][1]["output_length"] = 1
                elif invalid == "timing":
                    measurements[0]["requests"][0]["latency_ms"]["total"] = -1.0
                else:
                    topology = {"dp": 1}
                result = build_decode_case_result(
                    _case(),
                    selected=_selected(measurements),
                    topology=topology,
                    search=search,
                )
                self.assertIn(result["status"], ("invalid_shape", "invalid_timing"))
                self.assertIsNone(result["metrics"])
                details = result["errors"][0].get("details", {})
                self.assertEqual(search, details.get("search"))
                details["search"]["observations"][0]["outcome"] = "changed"
                self.assertEqual("satisfied", search["observations"][0]["outcome"])


if __name__ == "__main__":
    unittest.main()

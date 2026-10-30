import copy
import itertools
import unittest
from dataclasses import replace

from rtp_llm.test.perf_test.decode.metrics import build_decode_case_result
from rtp_llm.test.perf_test.decode.model import (
    DecodeBatchMeasurement,
    DecodeCaseSpec,
    ExactBatchPolicy,
    TargetTPOTPolicy,
)
from rtp_llm.test.perf_test.decode.report import generate_decode_report
from rtp_llm.test.perf_test.plan import case_id

TOPOLOGY = {"tp": 1, "dp": 1, "pp": 1, "ep": 1}
DISABLED = {"status": "disabled", "manifest_path": None, "error": None}


def _exact_case(batch):
    return DecodeCaseSpec(
        workload_kind="grid",
        input_lengths=(128,) * batch,
        output_length=16,
        batch_policy=ExactBatchPolicy("fixed", batch),
        measure_runs=1,
        warmup_runs=0,
    )


def _measurement(batch, inputs, output):
    return DecodeBatchMeasurement(
        batch_size=batch,
        input_lengths=inputs,
        worker_allocations=(),
        measurements=(
            {
                "run": 0,
                "throughput_tokens_per_second": 100.0,
                "worker_allocations": [],
                "requests": [
                    {
                        "request_index": index,
                        "success": True,
                        "input_length": length,
                        "output_length": output,
                        "latency_ms": {
                            "wait": 1.0,
                            "prefill": 2.0,
                            "decode": 15.0,
                            "tpot": 1.0,
                            "total": 17.0,
                        },
                    }
                    for index, length in enumerate(inputs)
                ],
            },
        ),
    )


def _result(spec, batch, search=None):
    identity = spec.to_identity()
    return build_decode_case_result(
        {"case_id": case_id("decode", identity), **identity},
        selected=_measurement(batch, spec.input_lengths[:batch], spec.output_length),
        topology=TOPOLOGY,
        search=search,
        profiling=DISABLED,
    )


def _exact_cases(inputs, outputs, batches):
    return [
        _result(
            replace(
                _exact_case(batch),
                input_lengths=(input_length,) * batch,
                output_length=output_length,
            ),
            batch,
        )
        for batch, input_length, output_length in itertools.product(
            batches, inputs, outputs
        )
    ]


def _tpot_cases(inputs, outputs):
    cases = []
    for index, (input_length, output_length) in enumerate(
        itertools.product(inputs, outputs)
    ):
        batch = (8, 12)[index]
        spec = replace(
            _exact_case(16),
            input_lengths=(input_length,) * 16,
            output_length=output_length,
            batch_policy=TargetTPOTPolicy("target_tpot", 16, 4, 10.0),
        )
        search = {
            "target_tpot_ms": 10.0,
            "max_size": 16,
            "candidate_step": 4,
            "outcome": "found",
            "best_batch_size": batch,
            "observations": [{"batch_size": batch, "outcome": "satisfied"}],
        }
        cases.append(_result(spec, batch, search))
    return cases


class DecodeReportTest(unittest.TestCase):
    def test_exact_report_groups_input_output_matrix_by_batch(self):
        cases = _exact_cases((128, 512), (16, 64), (1, 4))
        before = copy.deepcopy(cases)
        report = generate_decode_report(cases)
        self.assertEqual(2, report["schema_version"])
        self.assertEqual("formal", report["report_kind"])
        self.assertEqual({"planned": 8, "valid": 8, "excluded": 0}, report["counts"])
        self.assertEqual(
            [1, 4], [matrix["batch_size"] for matrix in report["matrices"]]
        )
        first = report["matrices"][0]
        self.assertEqual([128, 512], first["input_lengths"])
        self.assertEqual([16, 64], first["output_lengths"])
        self.assertEqual(4, len(first["cells"]))
        self.assertEqual(1.0, first["cells"][0]["tpot_ms"])
        self.assertEqual(100.0, first["cells"][0]["throughput_tokens_per_second"])
        self.assertEqual(17.0, first["cells"][0]["total_latency_ms"])
        self.assertEqual(1.0, first["cells"][0]["success_rate"])
        self.assertEqual(
            [item["case_id"] for item in cases],
            [item["case_id"] for item in report["observations"]],
        )
        self.assertEqual(before, cases)
        first["cells"][0]["errors"].append({"message": "changed"})
        self.assertEqual([], cases[0]["errors"])

    def test_matrix_axes_preserve_config_order(self):
        report = generate_decode_report(_exact_cases((512, 128), (64, 16), (4, 1)))
        self.assertEqual(
            [4, 1], [matrix["batch_size"] for matrix in report["matrices"]]
        )
        self.assertEqual([512, 128], report["matrices"][0]["input_lengths"])
        self.assertEqual([64, 16], report["matrices"][0]["output_lengths"])

    def test_tpot_report_cells_expose_selected_batch(self):
        report = generate_decode_report(_tpot_cases((128,), (16, 64)))
        self.assertEqual("formal", report["report_kind"])
        self.assertEqual(1, len(report["matrices"]))
        self.assertEqual(
            [8, 12],
            [cell["best_batch_size"] for cell in report["matrices"][0]["cells"]],
        )
        self.assertEqual(
            ["found", "found"],
            [cell["search_outcome"] for cell in report["matrices"][0]["cells"]],
        )

    def test_invalid_case_remains_diagnostic_without_formal_metrics(self):
        cases = _exact_cases((128,), (16,), (1,))
        cases[0].update(
            status="invalid_shape",
            metrics=None,
            errors=[{"code": "invalid_shape", "message": "wrong output"}],
        )
        report = generate_decode_report(cases)
        self.assertEqual("diagnostic", report["report_kind"])
        self.assertEqual({"planned": 1, "valid": 0, "excluded": 1}, report["counts"])
        cell = report["matrices"][0]["cells"][0]
        self.assertEqual("invalid_shape", cell["status"])
        self.assertIsNone(cell["tpot_ms"])
        self.assertEqual("wrong output", cell["errors"][0]["message"])

    def test_partial_request_case_does_not_contribute_formal_metrics(self):
        cases = _exact_cases((128,), (16,), (4,))
        cases[0]["status"] = "request_failed"
        cases[0]["metrics"].update(successful_requests=3, failed_requests=1)
        report = generate_decode_report(cases)
        self.assertEqual("diagnostic", report["report_kind"])
        self.assertIsNone(report["matrices"][0]["cells"][0]["tpot_ms"])

    def test_missing_grid_cell_and_empty_input_are_diagnostic(self):
        cases = _exact_cases((128, 512), (16, 64), (1,))[:-1]
        report = generate_decode_report(cases)
        self.assertEqual("diagnostic", report["report_kind"])
        self.assertEqual(4, len(report["matrices"][0]["cells"]))
        self.assertEqual("missing", report["matrices"][0]["cells"][-1]["status"])
        self.assertEqual("diagnostic", generate_decode_report([])["report_kind"])

    def test_failed_search_retains_trajectory_without_selecting_candidate(self):
        cases = _tpot_cases((128,), (16, 64))[:1]
        search = cases[0]["metrics"]["search"]
        search.update(
            outcome="no_satisfying_candidate",
            best_batch_size=None,
            observations=[
                {"batch_size": 8, "outcome": "capacity_failed", "error": "oom"}
            ],
        )
        cases[0].update(
            status="target_tpot_not_met",
            metrics=None,
            errors=[
                {
                    "code": "target_tpot_not_met",
                    "message": "no candidate",
                    "details": {"search": search},
                }
            ],
        )
        report = generate_decode_report(cases)
        cell = report["matrices"][0]["cells"][0]
        self.assertEqual("no_satisfying_candidate", cell["search_outcome"])
        self.assertIsNone(cell["best_batch_size"])
        self.assertIsNone(cell["throughput_tokens_per_second"])
        self.assertEqual(search, report["observations"][0]["search"])

    def test_distribution_is_an_ordered_table(self):
        spec = replace(
            _exact_case(2),
            workload_kind="distribution",
            input_lengths=(64, 128),
            distribution={"source_kind": "sequence_lengths"},
        )
        result = _result(spec, 2)
        report = generate_decode_report([result])
        self.assertEqual("formal", report["report_kind"])
        self.assertEqual([], report["matrices"])
        self.assertEqual([64, 128], report["distribution_cases"][0]["input_lengths"])

    def test_malformed_metrics_remain_diagnostic_cells(self):
        for metrics in ({}, [], [1], {"latency_ms": {"tpot": float("nan")}}):
            with self.subTest(metrics=metrics):
                cases = _exact_cases((128,), (16,), (1,))
                cases[0]["metrics"] = metrics
                report = generate_decode_report(cases)
                self.assertEqual("diagnostic", report["report_kind"])
                self.assertEqual(
                    "invalid_result", report["matrices"][0]["cells"][0]["status"]
                )
                self.assertIsNone(report["matrices"][0]["cells"][0]["tpot_ms"])

    def test_missing_cells_follow_matrix_axis_order(self):
        cases = _exact_cases((128, 512), (16, 64), (1,))
        del cases[1]
        report = generate_decode_report(cases)
        self.assertEqual(
            [(128, 16), (128, 64), (512, 16), (512, 64)],
            [
                (cell["input_length"], cell["output_length"])
                for cell in report["matrices"][0]["cells"]
            ],
        )


if __name__ == "__main__":
    unittest.main()

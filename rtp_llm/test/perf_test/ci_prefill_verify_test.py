"""Collected evidence gate tests: config text cannot satisfy these checks."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


class PrefillVerificationTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            importlib.util.find_spec("rtp_llm.test.perf_test.ci_prefill_verify"),
            "collected-artifact verifier is absent",
        )
        from rtp_llm.test.perf_test.ci_prefill_verify import verify_prefill_run

        self.verify = verify_prefill_run
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cases = []
        for i in range(30):
            length = 128 + i * 64
            identity = dict(
                input_len=length + 64,
                cache_len=64,
                compute_len=length + 64,
                batch_size=2,
                measure_runs=1,
                prefix_policy="independent",
                measurement_contract="concurrent_grouped_batch_v1",
                request_groups=[
                    dict(count=1, input_len=length, cache_len=64),
                    dict(count=1, input_len=length + 64, cache_len=0),
                ],
            )
            members = [
                dict(
                    run=0,
                    request_index=j,
                    group_index=j,
                    input_length=n,
                    expected_reuse_length=c,
                    observed_reuse_length=c,
                    compute_length=n - c,
                    latency_ms=10.0,
                    round_elapsed_ms=20.0,
                    measurement_contract=identity["measurement_contract"],
                )
                for j, (n, c) in enumerate([(length, 64), (length + 64, 0)])
            ]
            compute = 2 * length
            metrics = dict(
                input_length=compute + 64,
                expected_reuse_length=64,
                observed_reuse_length=64,
                compute_length=compute,
                latency_ms=10.0,
                throughput_tokens_per_second=compute * 50.0,
                validation=dict(shape=True, reuse=True, timing=True),
                seed=1,
                prefix_policy=dict(kind="independent"),
                measurements=members,
            )
            self.cases.append(
                dict(
                    case_id=f"case_{i}",
                    identity=identity,
                    status="ok",
                    runs=1,
                    errors=[],
                    metrics=metrics,
                )
            )
        self.result = dict(
            schema_version=2,
            run_id="prefill-test",
            mode="prefill",
            backend="aone",
            status="completed",
            config_fingerprint="a" * 64,
            workload_fingerprint="b" * 64,
            model=dict(
                label="test",
                type="test",
                checkpoint="/m",
                topology=dict(tp=2, dp=1, pp=1, ep=1),
            ),
            summary=dict(planned_cases=30, successful_cases=30, failed_cases=0),
            cases=self.cases,
        )
        self.write("results/result.json", self.result)
        from types import SimpleNamespace

        from rtp_llm.test.perf_test.prefill.report import _observation

        rows = [
            _observation(
                SimpleNamespace(case_id=c["case_id"], identity=c["identity"]), c
            )[0]
            for c in self.cases
        ]
        self.report = dict(
            mode="prefill",
            report_kind="formal",
            counts=dict(planned=30, valid=30, excluded=0),
            observations=rows,
            excluded=[],
            config_fingerprint="a" * 64,
            workload_fingerprint="b" * 64,
        )
        self.write("reports/prefill_report.json", self.report)
        self.write(
            "checkpoints/prefill_cache_probe.json",
            dict(
                status="ok",
                block_size=64,
                responses=dict(seed=dict(reuse_len=0), measure=dict(reuse_len=64)),
            ),
        )
        self.write(
            "reports/formula/fit_report.json",
            dict(
                candidate_generated=True,
                complete=True,
                unique_geometry_count=30,
                production_acceptance=False,
            ),
        )
        self.write(
            "reports/formula/model.json",
            dict(formula="1+I", production_acceptance=False),
        )
        for name, content in [
            ("PREFILL_TIME_FORMULA.txt", "PREFILL_TIME_FORMULA=1+I"),
            ("predictions.csv", "source,predicted_ms\ncase_0,10"),
            ("fit_gap.svg", "<svg></svg>"),
        ]:
            self.write_text("reports/formula/" + name, content)
        for chart in ("latency", "tpm"):
            self.write_text(f"reports/{chart}.svg", "<svg><circle/></svg>")
            self.write_text(
                f"reports/{chart}.interactive.html",
                '<!doctype html><html><script>window.Plotly={};Plotly.newPlot("chart",[],{updatemenus:[]});</script></html>',
            )
        self.write_text("reports/prefill_report.html", "<html>formal report</html>")

    def write_text(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def write(self, name, value):
        self.write_text(name, json.dumps(value))

    def test_candidate_without_production_acceptance_passes(self):
        self.assertEqual(self.verify(self.root)["valid_geometries"], 30)

    def test_missing_artifacts_fail(self):
        for name in (
            "results/result.json",
            "checkpoints/prefill_cache_probe.json",
            "reports/prefill_report.json",
            "reports/prefill_report.html",
            "reports/latency.interactive.html",
            "reports/tpm.svg",
            "reports/formula/model.json",
            "reports/formula/PREFILL_TIME_FORMULA.txt",
            "reports/formula/predictions.csv",
            "reports/formula/fit_gap.svg",
        ):
            with self.subTest(name=name):
                path = self.root / name
                saved = path.read_bytes()
                path.unlink()
                with self.assertRaises(ValueError):
                    self.verify(self.root)
                path.write_bytes(saved)

    def test_invalid_run_evidence_fails(self):
        for mutate in (
            lambda r: r.update(backend="local"),
            lambda r: r.update(status="partial"),
            lambda r: r["cases"][0].update(status="request_failed"),
            lambda r: r["cases"][0]["metrics"]["measurements"].pop(),
            lambda r: r["cases"][0]["metrics"]["measurements"][0].update(
                observed_reuse_length=0
            ),
            lambda r: r["cases"][0]["identity"].update(
                measurement_contract="concurrent_batch_v1"
            ),
        ):
            with self.subTest(mutate=mutate):
                result = copy.deepcopy(self.result)
                mutate(result)
                self.write("results/result.json", result)
                with self.assertRaises(ValueError):
                    self.verify(self.root)
        self.write("results/result.json", self.result)

    def test_diagnostic_or_forged_report_fails(self):
        for mutate in (
            lambda r: r.update(report_kind="diagnostic"),
            lambda r: r["observations"].pop(),
            lambda r: r["observations"][0].update(batch_latency_ms=99),
        ):
            report = copy.deepcopy(self.report)
            mutate(report)
            self.write("reports/prefill_report.json", report)
            with self.assertRaises(ValueError):
                self.verify(self.root)

    def test_static_or_external_chart_fails(self):
        for html in (
            "<html><svg/></html>",
            '<html><script src="https://cdn/plotly.js"></script><script>Plotly.newPlot();</script></html>',
        ):
            self.write_text("reports/latency.interactive.html", html)
            with self.assertRaises(ValueError):
                self.verify(self.root)

    def test_insufficient_or_missing_candidate_fails(self):
        for data in (
            dict(candidate_generated=False, complete=True, unique_geometry_count=30),
            dict(candidate_generated=True, complete=True, unique_geometry_count=29),
        ):
            self.write("reports/formula/fit_report.json", data)
            with self.assertRaises(ValueError):
                self.verify(self.root)

    def test_failed_cache_probe_fails(self):
        for probe in (
            dict(status="failed"),
            dict(
                status="ok",
                block_size=64,
                responses=dict(seed=dict(reuse_len=0), measure=dict(reuse_len=0)),
            ),
        ):
            self.write("checkpoints/prefill_cache_probe.json", probe)
            with self.assertRaises(ValueError):
                self.verify(self.root)

    def test_reported_geometry_count_cannot_replace_measurements(self):
        result = copy.deepcopy(self.result)
        result["cases"][-1] = dict(copy.deepcopy(result["cases"][0]), case_id="case_29")
        self.write("results/result.json", result)
        with self.assertRaises(ValueError):
            self.verify(self.root)

    def test_cli_returns_nonzero_for_missing_evidence(self):
        from contextlib import redirect_stderr
        from io import StringIO

        from rtp_llm.test.perf_test.ci_prefill_verify import main

        (self.root / "prefill-empty").mkdir()
        error = StringIO()
        with redirect_stderr(error):
            self.assertEqual(
                main(["--results-root", str(self.root), "--started-at", "0"]), 1
            )
        self.assertIn("results/result.json", error.getvalue())

    def test_cli_requires_exactly_one_fresh_run(self):
        from rtp_llm.test.perf_test.ci_prefill_verify import find_fresh_run

        (self.root / "prefill-old").mkdir()
        import os

        os.utime(self.root / "prefill-old", (1, 1))
        with self.assertRaises(ValueError):
            find_fresh_run(self.root, 2)
        (self.root / "prefill-new").mkdir()
        self.assertEqual(find_fresh_run(self.root, 2).name, "prefill-new")
        (self.root / "prefill-another").mkdir()
        with self.assertRaises(ValueError):
            find_fresh_run(self.root, 2)


if __name__ == "__main__":
    unittest.main()

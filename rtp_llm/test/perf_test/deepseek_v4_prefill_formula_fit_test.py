#!/usr/bin/env python3

import argparse
import json
import re
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.deepseek_v4_prefill_formula_fit import (
    DEFAULT_TOKEN_UNIT,
    FEATURE_NAMES,
    Observation,
    build_feature_names,
    build_parser,
    error_metrics,
    feature_values,
    fit_coefficients,
    formula_text,
    load_observations,
    predict,
    run_analyze_anomalies,
    run_fit,
    split_rows,
)


class DeepseekV4PrefillFormulaFitTest(unittest.TestCase):

    def test_features_use_compute_and_hit_tokens(self) -> None:
        row = Observation(
            batch_size=1,
            input_len=4096,
            cache_len=1024,
            target_ms=10.0,
            source="synthetic",
        )
        self.assertEqual(feature_values(row), [1.0, 3.0, 1.0, 9.0, 3.0, 1.0])

    def test_exported_formula_uses_only_prefill_time_formula_names(self) -> None:
        expression = formula_text([1.0, 2.0, -3.0, 4.0, -5.0, 6.0])
        identifiers = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression))
        self.assertEqual(identifiers, {"sum", "computeTokens", "hitCacheTokens"})
        self.assertNotIn("tokens", identifiers)

    def test_feature_expressions_match_flexlb_aggregate_grammar(self) -> None:
        self.assertEqual(len(FEATURE_NAMES), 6)
        for expression in FEATURE_NAMES[1:]:
            self.assertTrue(expression.startswith("sum("), expression)
            self.assertTrue(expression.endswith(")"), expression)

    def test_random_half_split_is_exact_and_reproducible(self) -> None:
        rows = [
            Observation(1, 1024 + index, index, 10.0 + index, f"row-{index}")
            for index in range(11)
        ]
        first = split_rows(rows, mode="random-50-50", seed=17)
        second = split_rows(rows, mode="random-50-50", seed=17)
        self.assertEqual(len(first["train"]), 5)
        self.assertEqual(len(first["validation"]), 0)
        self.assertEqual(len(first["test"]), 6)
        self.assertEqual(first, second)
        self.assertEqual(
            set(first["train"]) | set(first["test"]),
            set(rows),
        )
        self.assertFalse(set(first["train"]) & set(first["test"]))


def _make_observations(count: int = 20) -> list[Observation]:
    rows = []
    for i in range(count):
        input_len = 1024 * (i + 1)
        cache_len = 512 * i
        compute_len = input_len - cache_len
        target_ms = 10.0 + 0.5 * (compute_len / 1024.0) - 0.1 * (
            cache_len / 1024.0
        )
        rows.append(
            Observation(
                batch_size=1,
                input_len=input_len,
                cache_len=cache_len,
                target_ms=max(target_ms, 1.0),
                source=f"synth:{i}",
                requested_cache_len=cache_len,
            )
        )
    return rows


def _write_cache_grid_result(
    path: Path, rows: list[Observation], measure_runs: int = 3
):
    metrics = []
    for idx, row in enumerate(rows):
        runs = []
        for r in range(measure_runs):
            runs.append(
                {
                    "success": True,
                    "input_len": row.input_len,
                    "output_len": 1,
                    "reuse_len": row.cache_len,
                    "prefill_time_ms": row.target_ms,
                }
            )
        metrics.append(
            {
                "case_key": f"bs1_seq{row.input_len}_cache{row.cache_len}",
                "case_id": idx,
                "batch_size": 1,
                "input_len": row.input_len,
                "cache_len_requested": row.cache_len,
                "cache_len_observed": [row.cache_len] * measure_runs,
                "expected_reuse_len": row.cache_len,
                "success_runs": measure_runs,
                "measure_runs": measure_runs,
                "status": "ok",
                "reuse_exact": True,
                "runs": runs,
            }
        )
    payload = {
        "schema_version": 1,
        "mode": "prefix_cache_grid",
        "complete": True,
        "metrics": metrics,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class BuildFeatureNamesTest(unittest.TestCase):
    def test_default_token_unit(self):
        names = build_feature_names()
        self.assertEqual(len(names), 6)
        self.assertEqual(names[0], "1")
        self.assertIn("1024", names[1])
        self.assertEqual(names, build_feature_names(DEFAULT_TOKEN_UNIT))

    def test_custom_token_unit(self):
        names = build_feature_names(2048)
        self.assertEqual(len(names), 6)
        self.assertIn("2048", names[1])
        self.assertIn("2048", names[2])
        self.assertNotEqual(names, build_feature_names())


class FeatureValuesTest(unittest.TestCase):
    def test_default_unit(self):
        row = Observation(1, 2048, 1024, 10.0, "test", 1024)
        values = feature_values(row)
        self.assertAlmostEqual(values[0], 1.0)
        self.assertAlmostEqual(values[1], (2048 - 1024) / 1024.0)
        self.assertAlmostEqual(values[2], 1024 / 1024.0)

    def test_custom_unit(self):
        row = Observation(1, 2048, 1024, 10.0, "test", 1024)
        values_default = feature_values(row)
        values_custom = feature_values(row, 2048)
        self.assertAlmostEqual(values_default[0], values_custom[0])
        self.assertNotAlmostEqual(values_default[1], values_custom[1])
        self.assertAlmostEqual(values_custom[1], (2048 - 1024) / 2048.0)


class FitAndPredictTest(unittest.TestCase):
    def test_fit_with_default_token_unit(self):
        rows = _make_observations(20)
        coefficients, backend = fit_coefficients(rows, objective="mae")
        self.assertEqual(len(coefficients), 6)
        self.assertTrue(backend)

    def test_fit_with_custom_token_unit(self):
        rows = _make_observations(20)
        coefficients, backend = fit_coefficients(rows, objective="mae", token_unit=2048)
        self.assertEqual(len(coefficients), 6)

    def test_predict_roundtrip(self):
        rows = _make_observations(20)
        coefficients, _ = fit_coefficients(rows, objective="mae")
        predicted = predict(coefficients, rows[0])
        self.assertIsInstance(predicted, float)

    def test_predict_with_custom_token_unit(self):
        rows = _make_observations(20)
        coefficients, _ = fit_coefficients(rows, objective="mae", token_unit=2048)
        predicted = predict(coefficients, rows[0], token_unit=2048)
        self.assertIsInstance(predicted, float)

    def test_error_metrics_with_token_unit(self):
        rows = _make_observations(20)
        coefficients, _ = fit_coefficients(rows, objective="mae")
        metrics = error_metrics(rows, coefficients)
        self.assertIn("mape_pct", metrics)
        self.assertIn("p95_ape_pct", metrics)
        metrics_custom = error_metrics(rows, coefficients, token_unit=2048)
        self.assertIn("mape_pct", metrics_custom)


class FormulaTextTest(unittest.TestCase):
    def test_default_token_unit(self):
        names = build_feature_names()
        coefficients = [1.0, 2.0, 0.0, 0.0, 0.0, 0.0]
        text = formula_text(coefficients)
        self.assertIn("1024", text)
        self.assertIn(names[1], text)

    def test_custom_token_unit(self):
        coefficients = [1.0, 2.0, 0.0, 0.0, 0.0, 0.0]
        text = formula_text(coefficients, token_unit=2048)
        self.assertIn("2048", text)
        self.assertNotIn("1024", text)

    def test_zero_coefficients(self):
        text = formula_text([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(text, "0")


class RunFitTest(unittest.TestCase):
    def test_fit_without_profile(self):
        rows = _make_observations(40)
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "cache_grid_results.json"
            _write_cache_grid_result(result_path, rows)
            output_dir = Path(tmpdir) / "output"
            args = argparse.Namespace(
                inputs=[str(result_path)],
                output_dir=str(output_dir),
                batch_size=1,
                min_valid_rows=6,
                max_mape_pct=100.0,
                max_p95_ape_pct=100.0,
                max_max_ape_pct=100.0,
                objective="mae",
                estimator="median",
                allow_insufficient_data=False,
                profile=None,
                model_label=None,
                token_unit=None,
                formula_filename=None,
                formula_key=None,
            )
            exit_code = run_fit(args)
            self.assertIn(exit_code, (0, 3))
            report = json.loads((output_dir / "fit_report.json").read_text())
            self.assertIsNone(report["profile"])
            self.assertIsNone(report["profile_sha256"])
            self.assertEqual(report["token_unit"], DEFAULT_TOKEN_UNIT)
            self.assertTrue((output_dir / "deepseek_v4_prefill_formula.txt").exists())
            formula_content = (
                output_dir / "deepseek_v4_prefill_formula.txt"
            ).read_text()
            self.assertTrue(formula_content.startswith("PREFILL_TIME_FORMULA="))

    def test_fit_with_profile(self):
        rows = _make_observations(40)
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "cache_grid_results.json"
            _write_cache_grid_result(result_path, rows)
            profile = {
                "schema_version": 1,
                "label": "Test Model",
                "chart": {
                    "model_label": "Test Model",
                    "token_unit": 2048,
                    "formula_filename": "test_formula.txt",
                    "formula_key": "TEST_KEY",
                },
            }
            profile_path = Path(tmpdir) / "profile.json"
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            output_dir = Path(tmpdir) / "output"
            args = argparse.Namespace(
                inputs=[str(result_path)],
                output_dir=str(output_dir),
                batch_size=1,
                min_valid_rows=6,
                max_mape_pct=100.0,
                max_p95_ape_pct=100.0,
                max_max_ape_pct=100.0,
                objective="mae",
                estimator="median",
                allow_insufficient_data=False,
                profile=str(profile_path),
                model_label=None,
                token_unit=None,
                formula_filename=None,
                formula_key=None,
            )
            exit_code = run_fit(args)
            self.assertIn(exit_code, (0, 3))
            report = json.loads((output_dir / "fit_report.json").read_text())
            self.assertEqual(report["model"], "Test Model")
            self.assertIsNotNone(report["profile"])
            self.assertIsNotNone(report["profile_sha256"])
            self.assertEqual(report["token_unit"], 2048)
            self.assertTrue((output_dir / "test_formula.txt").exists())
            formula_content = (output_dir / "test_formula.txt").read_text()
            self.assertTrue(formula_content.startswith("TEST_KEY="))
            self.assertIn("2048", formula_content)

    def test_cli_overrides_profile(self):
        rows = _make_observations(40)
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "cache_grid_results.json"
            _write_cache_grid_result(result_path, rows)
            profile = {
                "schema_version": 1,
                "label": "Profile Label",
                "chart": {
                    "model_label": "Profile Label",
                    "token_unit": 2048,
                },
            }
            profile_path = Path(tmpdir) / "profile.json"
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            output_dir = Path(tmpdir) / "output"
            args = argparse.Namespace(
                inputs=[str(result_path)],
                output_dir=str(output_dir),
                batch_size=1,
                min_valid_rows=6,
                max_mape_pct=100.0,
                max_p95_ape_pct=100.0,
                max_max_ape_pct=100.0,
                objective="mae",
                estimator="median",
                allow_insufficient_data=False,
                profile=str(profile_path),
                model_label="CLI Label",
                token_unit=512,
                formula_filename="cli_formula.txt",
                formula_key="CLI_KEY",
            )
            exit_code = run_fit(args)
            self.assertIn(exit_code, (0, 3))
            report = json.loads((output_dir / "fit_report.json").read_text())
            self.assertEqual(report["model"], "CLI Label")
            self.assertEqual(report["token_unit"], 512)
            self.assertTrue((output_dir / "cli_formula.txt").exists())
            formula_content = (output_dir / "cli_formula.txt").read_text()
            self.assertTrue(formula_content.startswith("CLI_KEY="))
            self.assertIn("512", formula_content)


class AnalyzeAnomaliesTest(unittest.TestCase):
    def test_basic_analysis(self):
        rows = _make_observations(40)
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "cache_grid_results.json"
            _write_cache_grid_result(result_path, rows)
            output_dir = Path(tmpdir) / "output"
            args = argparse.Namespace(
                inputs=[str(result_path)],
                output_dir=str(output_dir),
                batch_size=1,
                estimator="median",
                min_rt_ms=5.0,
                min_compute_tokens=16384,
                max_anomaly_ape_pct=25.0,
                profile=None,
                model_label=None,
                token_unit=None,
            )
            exit_code = run_analyze_anomalies(args)
            self.assertEqual(exit_code, 0)
            report = json.loads((output_dir / "anomaly_report.json").read_text())
            self.assertIn("summary", report)
            self.assertIn("anomalies", report)
            self.assertIn("total_anomalies", report["summary"])
            self.assertIsNone(report["profile"])

    def test_with_profile(self):
        rows = _make_observations(40)
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "cache_grid_results.json"
            _write_cache_grid_result(result_path, rows)
            profile = {
                "schema_version": 1,
                "label": "Anomaly Test Model",
                "chart": {
                    "model_label": "Anomaly Test Model",
                },
            }
            profile_path = Path(tmpdir) / "profile.json"
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            output_dir = Path(tmpdir) / "output"
            args = argparse.Namespace(
                inputs=[str(result_path)],
                output_dir=str(output_dir),
                batch_size=1,
                estimator="median",
                min_rt_ms=5.0,
                min_compute_tokens=16384,
                max_anomaly_ape_pct=25.0,
                profile=str(profile_path),
                model_label=None,
                token_unit=None,
            )
            exit_code = run_analyze_anomalies(args)
            self.assertEqual(exit_code, 0)
            report = json.loads((output_dir / "anomaly_report.json").read_text())
            self.assertEqual(report["model"], "Anomaly Test Model")
            self.assertIsNotNone(report["profile"])
            self.assertIsNotNone(report["profile_sha256"])

    def test_detects_cache_monotonicity_violation(self):
        rows = [
            Observation(1, 4096, 0, 20.0, "a", 0),
            Observation(1, 4096, 2048, 25.0, "b", 2048),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "cache_grid_results.json"
            _write_cache_grid_result(result_path, rows)
            output_dir = Path(tmpdir) / "output"
            args = argparse.Namespace(
                inputs=[str(result_path)],
                output_dir=str(output_dir),
                batch_size=1,
                estimator="median",
                min_rt_ms=1.0,
                min_compute_tokens=1024,
                max_anomaly_ape_pct=25.0,
                profile=None,
                model_label=None,
                token_unit=None,
            )
            run_analyze_anomalies(args)
            report = json.loads((output_dir / "anomaly_report.json").read_text())
            cache_anomalies = [
                a for a in report["anomalies"] if a["check"] == "cache_monotonicity"
            ]
            self.assertTrue(len(cache_anomalies) > 0)


class ParserTest(unittest.TestCase):
    def test_fit_has_profile_flags(self):
        parser = build_parser()
        args = parser.parse_args(
            ["fit", "--inputs", "a.json", "--output-dir", "/tmp/out"]
        )
        self.assertIsNone(args.profile)
        self.assertIsNone(args.token_unit)
        self.assertIsNone(args.formula_filename)
        self.assertIsNone(args.formula_key)

    def test_analyze_anomalies_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "analyze-anomalies",
                "--inputs",
                "a.json",
                "--output-dir",
                "/tmp/out",
            ]
        )
        self.assertEqual(args.command, "analyze-anomalies")
        self.assertEqual(args.min_rt_ms, 5.0)
        self.assertEqual(args.min_compute_tokens, 16384)
        self.assertEqual(args.max_anomaly_ape_pct, 25.0)

    def test_validate_has_profile_flags(self):
        parser = build_parser()
        args = parser.parse_args(["validate-inputs", "--inputs", "a.json"])
        self.assertIsNone(args.profile)
        self.assertIsNone(args.model_label)


if __name__ == "__main__":
    unittest.main()

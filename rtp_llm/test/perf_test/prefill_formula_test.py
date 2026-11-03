import csv
import json
import re
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.prefill import formula
from rtp_llm.test.perf_test.prefill.restricted_symbolic import build_candidate_library


def observations(n=120):
    rows = []
    for i in range(n):
        requests = [(256 + 64 * i, 64 * (i % 3)), (128 + 32 * i, 0)]
        if i % 2 == 0:
            requests = [requests[0], requests[0]]
        rows.append(
            dict(
                case_id=str(i),
                batch_size=2,
                fit_eligible=True,
                measurement_contract=(
                    "concurrent_batch_v1"
                    if i % 2 == 0
                    else "concurrent_grouped_batch_v1"
                ),
                request_distribution=[
                    dict(input_length=x, expected_reuse_length=c) for x, c in requests
                ],
                batch_latency_ms=3 + sum(x - c for x, c in requests) / 1024,
            )
        )
    return rows


class FormulaTest(unittest.TestCase):
    def test_svg_labels_observations_and_round_maximum_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gap.svg"
            formula.write_fit_gap_svg(
                [{"target_ms": 4, "predicted_ms": 5, "cache_len": 0}] * 2, path
            )
            svg = path.read_text()
            self.assertIn("2 个有效观测", svg)
            self.assertIn("每轮最大服务端首 token 延迟的跨轮中位数", svg)
            self.assertNotIn("3 次", svg)
            self.assertNotIn("geometry", svg)

    def test_full_distribution_features_and_grammar(self):
        row = formula.normalize_observations(observations(2))[0][1]
        terms = {t.name: t for t in build_candidate_library(1024)}
        self.assertAlmostEqual(
            terms["u**2"].evaluate(row),
            sum(((x - c) / 1024) ** 2 for x, c in row.requests),
        )
        allowed = {
            "sum",
            "sqrt",
            "log",
            "exp",
            "abs",
            "max",
            "min",
            "pow",
            "inputTokens",
            "hitCacheTokens",
            "computeTokens",
            "hasHitCache",
            "batchSize",
            "maxInputTokens",
            "maxComputeTokens",
        }
        for term in terms.values():
            self.assertNotIn("**", term.expression)
            self.assertLessEqual(
                set(re.findall(r"[A-Za-z]+", term.expression)), allowed
            )

    def test_split_is_permutation_and_order_invariant(self):
        data = observations()
        duplicate = dict(
            data[3],
            case_id="duplicate",
            request_distribution=list(reversed(data[3]["request_distribution"])),
        )
        rows, excluded = formula.normalize_observations(data + [duplicate])
        splits = formula.split_rows(rows)
        assignment = {r.source: name for name, group in splits.items() for r in group}
        self.assertEqual(assignment["3"], assignment["duplicate"])
        self.assertEqual(
            assignment,
            {
                r.source: name
                for name, group in formula.split_rows(list(reversed(rows))).items()
                for r in group
            },
        )
        self.assertTrue(all(splits.values()))
        self.assertEqual(len({r.geometry_key for r in rows}), 120)

    def test_quality_thresholds_and_per_batch_coverage(self):
        good = dict(n=1, mape_pct=5.0, p95_ape_pct=10.0, max_ape_pct=40.0)
        metrics = dict(
            test=good, by_batch={"1": dict(train=good, validation=good, test=good)}
        )
        self.assertTrue(formula.quality_passes(metrics))
        for key in ("mape_pct", "p95_ape_pct", "max_ape_pct"):
            bad = dict(good, **{key: good[key] + 0.01})
            self.assertFalse(formula.quality_passes(dict(metrics, test=bad)))
            self.assertFalse(
                formula.quality_passes(
                    dict(
                        metrics,
                        by_batch={"1": dict(train=good, validation=good, test=bad)},
                    )
                )
            )
        self.assertFalse(
            formula.quality_passes(
                dict(
                    metrics,
                    by_batch={
                        "1": dict(train=good, validation=dict(good, n=0), test=good)
                    },
                )
            )
        )

    def test_mixed_contract_fit_and_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = formula.fit_prefill_formula(observations(), tmp, complete=True)
            self.assertTrue(report["production_acceptance"], report)
            for name in (
                "model.json",
                "fit_report.json",
                "predictions.csv",
                "fit_gap.svg",
                "PREFILL_TIME_FORMULA.txt",
            ):
                self.assertTrue((Path(tmp) / name).is_file())
            with (Path(tmp) / "predictions.csv").open() as stream:
                predictions = list(csv.DictReader(stream))
            self.assertEqual(len(predictions), 120)
            self.assertEqual(
                set(p["split"] for p in predictions), {"train", "validation", "test"}
            )
            self.assertIn("sum(", (Path(tmp) / "PREFILL_TIME_FORMULA.txt").read_text())

    def test_insufficient_duplicates_legacy_partial_and_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("model.json", "PREFILL_TIME_FORMULA.txt"):
                (Path(tmp) / name).write_text("previous accepted")
            result = formula.fit_prefill_formula(
                observations(2) * 20, tmp, complete=True
            )
            self.assertFalse(result["production_acceptance"])
            self.assertEqual(result["reason"], "insufficient_valid_geometries")
            self.assertFalse((Path(tmp) / "model.json").exists())
            self.assertFalse((Path(tmp) / "PREFILL_TIME_FORMULA.txt").exists())
            rows = observations()
            rows[0]["measurement_contract"] = "legacy_latency_derived_v1"
            result = formula.fit_prefill_formula(rows, tmp, complete=True)
            self.assertFalse(result["production_acceptance"])
            self.assertTrue(result["excluded"])
            result = formula.fit_prefill_formula(observations(), tmp, complete=False)
            self.assertFalse(result["production_acceptance"])
            self.assertEqual(result["reason"], "partial_run")
            self.assertFalse(
                json.loads((Path(tmp) / "model.json").read_text())[
                    "production_acceptance"
                ]
            )


if __name__ == "__main__":
    unittest.main()

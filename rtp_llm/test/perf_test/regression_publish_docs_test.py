import json
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.config_schema import parse_config

ROOT = Path(__file__).resolve().parent
EXAMPLE = ROOT / "examples" / "prefill_regression_publish.json"
DOC = ROOT / "docs" / "regression_publish.md"
BUILD = ROOT / "BUILD"


class RegressionPublishDocsTest(unittest.TestCase):
    def test_example_passes_strict_schema_and_has_explicit_policy(self):
        source = json.loads(EXAMPLE.read_text(encoding="utf-8"))

        config = parse_config(source)

        self.assertTrue(config.regression.enabled)
        self.assertTrue(config.regression.generate_candidate)
        self.assertEqual(5.0, config.regression.threshold_pct)
        self.assertTrue(config.publish.oss.enabled)
        self.assertTrue(config.publish.oss.required)
        self.assertTrue(config.publish.odps.enabled)
        self.assertFalse(config.publish.odps.required)
        self.assertEqual((), config.publish.odps.artifacts)
        self.assertEqual(
            len(config.publish.oss.artifacts), len(set(config.publish.oss.artifacts))
        )

    def test_example_contains_no_credential_fields_or_values(self):
        text = EXAMPLE.read_text(encoding="utf-8")
        lower = text.lower()

        for forbidden in (
            "access_key",
            "key_secret",
            "credential",
            "password",
            "secret:",
            "vault:",
            "env:",
        ):
            self.assertNotIn(forbidden, lower)

    def test_reference_names_runtime_variables_and_explicit_golden_targets(self):
        text = DOC.read_text(encoding="utf-8")
        build = BUILD.read_text(encoding="utf-8")

        for name in (
            "MAGA_OSS_KEY_ID",
            "MAGA_OSS_KEY_SECRET",
            "ALIBABA_CLOUD_ACCESS_KEY_ID",
            "ALIBABA_CLOUD_ACCESS_KEY_SECRET",
        ):
            self.assertIn(name, text)
        for target in (
            "generate_perf_test_golden",
            "validate_perf_test_golden",
        ):
            self.assertIn(f":{target}", text)
            self.assertIn(f'name = "{target}"', build)
        self.assertIn("results/result.json", text)
        self.assertIn("exit code 6", text)


if __name__ == "__main__":
    unittest.main()

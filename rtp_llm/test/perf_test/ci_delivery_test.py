import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.ci_delivery import deliver_config


class CiDeliveryTest(unittest.TestCase):
    def setUp(self):
        self.checkout = Path(__file__).resolve().parents[3]
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.destination = Path(self.temp.name) / "delivered.json"

    def deliver(self, **values):
        return deliver_config(
            {"PERF_TEST_" + key: value for key, value in values.items()},
            self.checkout,
            self.destination,
        )

    def test_prefill_defaults_preserve_full_single_grid(self):
        config = self.deliver(MODE="prefill", CONFIG="auto", MODEL_LABEL="auto")
        fixture = json.loads(
            (
                self.checkout / "rtp_llm/test/perf_test/examples/prefill_aone_ci.json"
            ).read_text()
        )
        self.assertEqual(config, fixture)
        self.assertEqual(len(config["workload"]["request_batches"]), 30)
        self.assertEqual(config["execution"]["max_parallel"], 2)
        self.assertEqual(json.loads(self.destination.read_text()), fixture)

    def test_decode_effective_defaults_are_unchanged(self):
        config = self.deliver()
        fixture = json.loads(
            (
                self.checkout / "rtp_llm/test/perf_test/examples/decode_aone_gang.json"
            ).read_text()
        )
        self.assertEqual(config, fixture)

    def test_explicit_inline_config_and_overrides_win(self):
        base = self.deliver(MODE="prefill")
        base["model"]["label"] = "custom"
        base["execution"]["backend"] = "local"
        config = self.deliver(
            MODE="prefill",
            CONFIG=json.dumps(base),
            MODEL_TYPE="other",
            CHECKPOINT_PATH="/model",
            TOKENIZER_PATH="/tokenizer",
            TP_SIZE="4",
            DP_SIZE="2",
            PP_SIZE="1",
            EP_SIZE="2",
            GPUS_PER_MACHINE="2",
            GPU_TYPE="custom-gpu",
            RESOURCE="online",
        )
        self.assertEqual(config["model"]["label"], "custom")
        self.assertEqual(config["model"]["type"], "other")
        self.assertEqual(config["model"]["checkpoint"], "/model")
        self.assertEqual(config["model"]["tokenizer"], "/tokenizer")
        self.assertEqual(config["model"]["topology"], dict(tp=4, dp=2, pp=1, ep=2))
        self.assertEqual(
            config["execution"],
            dict(
                backend="aone",
                resource="online",
                gpu_type="custom-gpu",
                max_parallel=4,
                gpus_per_machine=2,
            ),
        )
        self.assertEqual(config["workload"], base["workload"])

    def test_explicit_prefill_output_root_matches_ci_gate_and_upload(self):
        for mode in ("prefill", "decode"):
            with self.subTest(mode=mode):
                source = self.deliver(MODE=mode)
                source["output"]["result_root"] = "custom-results"
                delivered = self.deliver(MODE=mode, CONFIG=json.dumps(source))
                expected = (
                    "perf_test_results" if mode == "prefill" else "custom-results"
                )
                self.assertEqual(expected, delivered["output"]["result_root"])
                self.assertEqual(
                    expected,
                    json.loads(self.destination.read_text())["output"]["result_root"],
                )

    def test_explicit_path_is_relative_to_checkout(self):
        config = self.deliver(
            CONFIG="rtp_llm/test/perf_test/examples/decode_aone_gang.json",
            MODEL_LABEL="explicit",
        )
        self.assertEqual(config["model"]["label"], "explicit")

    def test_invalid_divisibility_rejected_before_write(self):
        with self.assertRaisesRegex(ValueError, "not divisible"):
            self.deliver(TP_SIZE="3", GPUS_PER_MACHINE="2")
        self.assertFalse(self.destination.exists())

    def test_invalid_modes_and_nonpositive_topology_rejected(self):
        for overrides in [dict(MODE="other"), dict(TP_SIZE="0"), dict(EP_SIZE="-1")]:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.deliver(**overrides)


if __name__ == "__main__":
    unittest.main()

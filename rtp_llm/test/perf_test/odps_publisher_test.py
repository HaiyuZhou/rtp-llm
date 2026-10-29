import copy
import io
import logging
import unittest

from rtp_llm.test.perf_test.plan import case_id
from rtp_llm.test.perf_test.publish.odps import OdpsPublisher


def _base(mode, case):
    return {
        "schema_version": 1,
        "run_id": f"{mode}-run-001",
        "mode": mode,
        "backend": "local",
        "status": "completed",
        "config_fingerprint": "a" * 64,
        "workload_fingerprint": "b" * 64,
        "model": {
            "label": "test-model",
            "type": "registered_model",
            "checkpoint": "/models/test",
            "topology": {"tp": 1, "dp": 1, "pp": 1, "ep": 1},
        },
        "summary": {"planned_cases": 1, "successful_cases": 1, "failed_cases": 0},
        "cases": [case],
    }


def _prefill_result():
    identity = {
        "batch_size": 1,
        "cache_len": 512,
        "compute_len": 512,
        "input_len": 1024,
    }
    return _base(
        "prefill",
        {
            "case_id": case_id("prefill", identity),
            "identity": identity,
            "status": "ok",
            "runs": 1,
            "errors": [],
            "metrics": {
                "input_length": 1024,
                "expected_reuse_length": 512,
                "observed_reuse_length": 512,
                "compute_length": 512,
                "latency_ms": 10.0,
                "throughput_tokens_per_second": 51200.0,
                "validation": {"shape": True, "reuse": True, "timing": True},
                "seed": 7,
                "prefix_policy": {"kind": "shared_seed"},
                "measurements": [{"run": 0, "raw": "must-not-publish"}],
            },
        },
    )


def _decode_result():
    identity = {"input_len": 128, "output_len": 16, "batch_size": 2}
    return _base(
        "decode",
        {
            "case_id": case_id("decode", identity),
            "identity": identity,
            "status": "ok",
            "runs": 1,
            "errors": [],
            "metrics": {
                "workload_kind": "grid",
                "input_lengths": [128, 128],
                "output_length": 16,
                "global_batch_size": 2,
                "topology": {"tp": 1, "dp": 1, "pp": 1, "ep": 1},
                "successful_requests": 2,
                "failed_requests": 0,
                "latency_ms": {
                    "wait": 1.0,
                    "prefill": 2.0,
                    "decode": 8.0,
                    "tpot": 0.5,
                    "total": 11.0,
                },
                "throughput_tokens_per_second": 3200.0,
                "measurements": [{"run": 0, "raw": "must-not-publish"}],
                "search": None,
                "distribution": None,
            },
        },
    )


class _Client:
    def __init__(self):
        self.calls = []

    def write_summary(self, destination, payload):
        self.calls.append((destination, copy.deepcopy(payload)))


class _Factory:
    def __init__(self, failure=False):
        self.failure = failure
        self.credentials = None
        self.client = _Client()

    def create(self, credentials):
        self.credentials = credentials
        if self.failure:
            raise RuntimeError(
                f"ODPS rejected {credentials.access_key_id}/{credentials.access_key_secret}"
            )
        return self.client


class OdpsPublisherTest(unittest.TestCase):
    def _publisher(self, factory):
        return OdpsPublisher(
            client_factory=factory,
            environment={
                "ALIBABA_CLOUD_ACCESS_KEY_ID": "runtime-id",
                "ALIBABA_CLOUD_ACCESS_KEY_SECRET": "runtime-secret",
                "ODPS_PROJECT": "perf-project",
                "ODPS_ENDPOINT": "https://odps.example.test/api",
            },
        )

    def test_prefill_payload_contains_only_finalized_aggregate_data(self):
        factory = _Factory()

        outcome = self._publisher(factory).publish(
            _prefill_result(), destination="perf_prefill_summary"
        )

        destination, payload = factory.client.calls[0]
        self.assertEqual("perf_prefill_summary", destination)
        self.assertEqual("prefill", payload["mode"])
        self.assertEqual(
            {
                "valid_cases": 1,
                "mean_latency_ms": 10.0,
                "mean_throughput_tokens_per_second": 51200.0,
            },
            payload["aggregate"],
        )
        self.assertNotIn("cases", payload)
        self.assertNotIn("measurements", repr(payload))
        self.assertNotIn("runtime-secret", repr(payload))
        self.assertEqual("succeeded", outcome["status"])

    def test_decode_payload_has_request_latency_and_throughput_aggregates(self):
        factory = _Factory()

        self._publisher(factory).publish(
            _decode_result(), destination="perf_decode_summary"
        )

        payload = factory.client.calls[0][1]
        self.assertEqual("decode", payload["mode"])
        self.assertEqual(2, payload["aggregate"]["successful_requests"])
        self.assertEqual(0, payload["aggregate"]["failed_requests"])
        self.assertEqual(11.0, payload["aggregate"]["mean_latency_ms"]["total"])
        self.assertEqual(
            3200.0, payload["aggregate"]["mean_throughput_tokens_per_second"]
        )

    def test_credential_failure_is_sanitized_in_exception_and_logs(self):
        factory = _Factory(failure=True)
        captured = io.StringIO()
        logger = logging.getLogger("perf-test-odps-test")
        logger.handlers = [logging.StreamHandler(captured)]
        logger.setLevel(logging.INFO)

        with self.assertRaisesRegex(RuntimeError, "\*\*\*") as raised:
            self._publisher(factory).publish(
                _prefill_result(),
                destination="perf_summary",
                logger=logger,
            )

        text = captured.getvalue() + str(raised.exception)
        self.assertNotIn("runtime-id", text)
        self.assertNotIn("runtime-secret", text)

    def test_missing_credentials_fail_before_client_creation(self):
        factory = _Factory()
        publisher = OdpsPublisher(client_factory=factory, environment={})

        with self.assertRaisesRegex(RuntimeError, "ALIBABA_CLOUD_ACCESS_KEY_ID"):
            publisher.publish(_prefill_result(), destination="perf_summary")

        self.assertIsNone(factory.credentials)


if __name__ == "__main__":
    unittest.main()

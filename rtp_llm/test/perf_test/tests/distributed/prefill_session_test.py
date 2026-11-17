"""Tests for multi_node prefill_session_test."""

import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.entry import plan_from_source
from rtp_llm.test.perf_test.errors import CaseExecutionError
from rtp_llm.test.perf_test.execution.distributed.prefill import ExternalPrefillSession
from rtp_llm.test.perf_test.state import RunState
from rtp_llm.test.perf_test.tests.support.fixtures import (
    prefill_external_aux,
    prefill_external_CASE_CACHE_LEN,
    prefill_external_CASE_INPUT_LEN,
    prefill_external_FakeModelServer,
    prefill_external_TOPOLOGY,
    prefill_external_write_prefill_config,
)
from rtp_llm.test.perf_test.tests.support.prefill import (
    prefill_external_run_through_session,
)


class ExternalPrefillSessionTest(unittest.TestCase):

    def test_prefill_runner_completes_through_the_external_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = prefill_external_write_prefill_config(root)
            (plan, _source) = plan_from_source(
                json.loads(Path(path).read_text()),
                source_dir=Path(path).parent,
                result_directory=str(str(root / "run")),
            )
            with prefill_external_FakeModelServer(
                [
                    prefill_external_aux(prefill_external_CASE_INPUT_LEN, 0, 5.0),
                    prefill_external_aux(
                        prefill_external_CASE_INPUT_LEN,
                        prefill_external_CASE_CACHE_LEN,
                        9000.0,
                    ),
                    prefill_external_aux(
                        prefill_external_CASE_INPUT_LEN,
                        prefill_external_CASE_CACHE_LEN,
                        12.5,
                    ),
                ]
            ) as server:
                with ExternalPrefillSession(
                    server.endpoint,
                    topology=prefill_external_TOPOLOGY,
                    transport="http_text",
                ) as session:
                    outcome = prefill_external_run_through_session(
                        plan, session, root / "run"
                    )
            self.assertEqual(RunState.COMPLETED, outcome.state)
            self.assertIsNone(outcome.failure)
            self.assertEqual(1, len(outcome.case_results))
            result = outcome.case_results[0]
            for field in ("case_id", "identity", "status", "runs", "errors", "metrics"):
                self.assertIn(field, result)
            self.assertEqual("ok", result["status"])
            self.assertEqual(
                prefill_external_CASE_INPUT_LEN, result["metrics"]["input_length"]
            )
            self.assertEqual(
                prefill_external_CASE_CACHE_LEN,
                result["metrics"]["observed_reuse_length"],
            )
            self.assertEqual(12.5, result["metrics"]["latency_ms"])
            self.assertEqual(40.0, result["metrics"]["throughput_tokens_per_second"])
            persisted = json.loads(
                (root / "run" / "results" / "cases.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual("ok", persisted["status"])
            self.assertEqual(5, len(server.requests))
            (configure_seed, seed, configure_measure, warmup, measure) = server.requests
            for update in (configure_seed, configure_measure):
                self.assertEqual("/update_scheduler_info", update["path"])
                self.assertEqual({"batch_size": 1, "mode": "prefill"}, update["body"])
            self.assertEqual("/", seed["path"])
            self.assertEqual("none", seed["headers"]["X-DashScope-LoggingConsent"])
            self.assertEqual(1, seed["body"]["generate_config"]["max_new_tokens"])
            self.assertNotEqual(seed["body"]["prompt"], measure["body"]["prompt"])
            self.assertNotEqual(warmup["body"]["prompt"], measure["body"]["prompt"])

    def test_prefill_runner_flags_invalid_reuse_through_the_external_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = prefill_external_write_prefill_config(root)
            (plan, _source) = plan_from_source(
                json.loads(Path(path).read_text()),
                source_dir=Path(path).parent,
                result_directory=str(str(root / "run")),
            )
            with prefill_external_FakeModelServer(
                [
                    prefill_external_aux(prefill_external_CASE_INPUT_LEN, 0, 5.0),
                    prefill_external_aux(prefill_external_CASE_INPUT_LEN, 0, 12.5),
                ]
            ) as server:
                session = ExternalPrefillSession(
                    server.endpoint,
                    topology=prefill_external_TOPOLOGY,
                    transport="http_text",
                )
                outcome = prefill_external_run_through_session(
                    plan, session, root / "run"
                )
                session.close()
            self.assertEqual(RunState.PARTIAL, outcome.state)
            self.assertIsInstance(outcome.failure, CaseExecutionError)
            result = outcome.case_results[0]
            self.assertEqual("invalid_reuse", result["status"])
            self.assertIsNone(result["metrics"])
            self.assertEqual("invalid_reuse", result["errors"][0]["code"])
            self.assertIn("expected reuse 16", result["errors"][0]["message"])
            persisted = json.loads(
                (root / "run" / "results" / "cases.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual("invalid_reuse", persisted["status"])


if __name__ == "__main__":
    unittest.main()

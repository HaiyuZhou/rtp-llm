"""Tests for prefill profile_test."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call

from rtp_llm.test.perf_test.prefill.profile import PrefillProfiler
from rtp_llm.test.perf_test.prefill.trace import TraceCapture, TraceClient
from rtp_llm.test.perf_test.tests.support.fixtures import (
    prefill_profile_formal_bytes,
    prefill_profile_setup,
)


class PrefillProfileTest(unittest.TestCase):

    def test_selected_case_is_reconstructed_into_isolated_profile_session(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            plan = prefill_profile_setup(root)
            before = prefill_profile_formal_bytes(root)
            runner = Mock()
            runner.run_case.side_effect = lambda case: {
                "case_id": case["case_id"],
                "status": "ok",
                "metrics": {"latency_ms": 1.0},
                "errors": [],
            }

            def factory(profile_dir):
                (profile_dir / "trace.json").write_text(
                    '{"traceEvents": [{"name": "kernel"}]}'
                )
                original = runner.run_case.side_effect
                runner.run_case.side_effect = lambda case: dict(
                    original(case), trace_files=["trace.json"]
                )
                return runner

            profiler = PrefillProfiler(factory)
            outcome = profiler.run(
                root, [plan.cases[0].case_id], session_id="profile-001"
            )
            self.assertEqual("completed", outcome["status"])
            reconstructed = runner.run_case.call_args.args[0]
            self.assertEqual(plan.cases[0].case_id, reconstructed["case_id"])
            self.assertEqual(512, reconstructed["input_len"])
            self.assertTrue(
                (root / "profiles" / "profile-001" / "profile_result.json").is_file()
            )
            self.assertEqual(before, prefill_profile_formal_bytes(root))

    def test_failed_profile_keeps_formal_artifacts_unchanged(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            plan = prefill_profile_setup(root)
            before = prefill_profile_formal_bytes(root)
            runner = Mock()
            runner.run_case.return_value = {
                "case_id": plan.cases[0].case_id,
                "status": "invalid_reuse",
                "metrics": None,
                "errors": [{"code": "invalid_reuse", "message": "mismatch"}],
            }
            outcome = PrefillProfiler(lambda _profile_dir: runner).run(
                root, [plan.cases[0].case_id], session_id="failed"
            )
            self.assertEqual("failed", outcome["status"])
            self.assertEqual(before, prefill_profile_formal_bytes(root))


class TraceTest(unittest.TestCase):

    def test_kineto_all_ranks_and_measure_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            post = Mock(return_value=Mock(json=lambda: {"status": "ok"}))
            capture = TraceCapture(root, "case", port=123, tp_size=2, post=post)
            client = TraceClient(Mock(), capture)
            client.request(phase="seed")
            post.assert_not_called()
            client.request(phase="measure")
            client.request(phase="measure")
            self.assertEqual(1, post.call_count)
            self.assertEqual(
                "http://127.0.0.1:123/start_profile", post.call_args.args[0]
            )
            for rank in range(2):
                (root / f"case_wr{rank}_1.json").write_text(
                    json.dumps({"traceEvents": [{"name": "kernel"}]})
                )
            self.assertEqual(2, len(capture.finish()))

    def test_nsys_stop_exports_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def command(argv, **kwargs):
                if argv[1] == "stop":
                    (root / "case.nsys-rep").write_bytes(b"report")
                return Mock(returncode=0, stdout="", stderr="")

            run = Mock(side_effect=command)
            capture = TraceCapture(
                root, "case", backend="nsys", nsys_session="session", run=run
            )
            capture.start()
            self.assertEqual(["case.nsys-rep"], capture.finish())
            self.assertEqual(
                ["start", "stop"], [call.args[0][1] for call in run.call_args_list]
            )


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from rtp_llm.test.perf_test.prefill.trace import TraceCapture, TraceClient


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

    def test_invalid_or_missing_trace_times_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "case_wr0_1.json").write_text('{"traceEvents": []}')
            capture = TraceCapture(
                root,
                "case",
                port=123,
                timeout=0.001,
                post=Mock(return_value=Mock(json=lambda: {"status": "ok"})),
            )
            capture.start()
            with self.assertRaisesRegex(RuntimeError, "timeout"):
                capture.finish()

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

    def test_nsys_start_failure_still_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Mock(side_effect=[RuntimeError("start failed"), Mock(returncode=0)])
            capture = TraceCapture(
                Path(tmp), "case", backend="nsys", nsys_session="session", run=run
            )
            with self.assertRaisesRegex(RuntimeError, "start failed"):
                capture.start()
            capture.abort()
            self.assertEqual("stop", run.call_args.args[0][1])

    def test_nsys_missing_report_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture = TraceCapture(
                Path(tmp),
                "case",
                backend="nsys",
                nsys_session="session",
                run=Mock(return_value=Mock(returncode=0)),
            )
            capture.start()
            with self.assertRaisesRegex(RuntimeError, "nonempty"):
                capture.finish()

    def test_concurrent_start_gate_blocks_all_members_until_acknowledgement(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor

        armed = threading.Event()
        release = threading.Event()
        post_entered = threading.Event()

        def post(*args, **kwargs):
            post_entered.set()
            self.assertTrue(release.wait(2))
            armed.set()
            return Mock(json=lambda: {"status": "ok"})

        client = Mock()
        client.request.side_effect = lambda **kwargs: self.assertTrue(armed.is_set())
        with tempfile.TemporaryDirectory() as tmp:
            wrapped = TraceClient(client, TraceCapture(tmp, "case", post=post))
            with ThreadPoolExecutor(2) as pool:
                first = pool.submit(wrapped.request, phase="measure")
                self.assertTrue(post_entered.wait(2))
                second = pool.submit(wrapped.request, phase="measure")
                client.request.assert_not_called()
                release.set()
                first.result(timeout=2)
                second.result(timeout=2)
        self.assertEqual(2, client.request.call_count)

    def test_nsys_stops_after_staggered_concurrent_members_and_excludes_seeds(self):
        import threading
        from unittest.mock import patch

        from rtp_llm.test.perf_test.default_executor_test import _Tokenizer
        from rtp_llm.test.perf_test.prefill.trace import ProfiledRunner

        events = []
        first_done = threading.Event()

        class Client:
            port = 123

            def configure_scheduler(self, *args, **kwargs):
                pass

            def request(self, **kwargs):
                if kwargs["phase"] == "seed":
                    events.append("seed")
                elif kwargs["request_index"] == 0:
                    events.append("member0")
                    first_done.set()
                else:
                    self_outer.assertTrue(first_done.wait(2))
                    events.append("member1")
                return {
                    "success": True,
                    "input_len": len(kwargs["input_ids"]),
                    "output_len": 1,
                    "reuse_len": 0 if kwargs["phase"] == "seed" else 4,
                    "latency_ms": 1,
                }

        self_outer = self
        with tempfile.TemporaryDirectory() as tmp:

            def command(argv, **kwargs):
                events.append(argv[1])
                if argv[1] == "stop":
                    (Path(tmp) / "case.nsys-rep").write_bytes(b"report")
                return Mock(returncode=0, stdout="", stderr="")

            config = {
                "model": {"topology": {"tp": 1}},
                "workload": {"commit_tail_tokens": 4, "profiling": {"backend": "nsys"}},
            }
            with patch(
                "rtp_llm.test.perf_test.prefill.trace.TraceCapture",
                side_effect=lambda *a, **k: TraceCapture(*a, **k, run=command),
            ):
                result = ProfiledRunner(
                    Client(), _Tokenizer(), tmp, config, "session"
                ).run_case(
                    {
                        "case_id": "case",
                        "input_len": 16,
                        "cache_len": 4,
                        "compute_len": 12,
                        "batch_size": 2,
                        "measure_runs": 3,
                        "seed": 1,
                        "prefix_policy": "independent",
                    }
                )
            self.assertEqual("ok", result["status"], result)
            self.assertEqual(
                ["seed", "seed", "start", "member0", "member1", "stop"], events
            )
            self.assertEqual(1, result["runs"])
            self.assertEqual(
                "one measured concurrent round; seeds excluded", result["capture_scope"]
            )

    def test_kineto_result_describes_one_forward_step(self):
        from unittest.mock import patch

        from rtp_llm.test.perf_test.prefill.trace import ProfiledRunner

        config = {
            "model": {"topology": {"tp": 1}},
            "workload": {"commit_tail_tokens": 4, "profiling": {"backend": "kineto"}},
        }
        with tempfile.TemporaryDirectory() as tmp, patch(
            "rtp_llm.test.perf_test.prefill.runner.PrefillRunner"
        ) as runner, patch(
            "rtp_llm.test.perf_test.prefill.trace.TraceCapture"
        ) as capture:
            runner.return_value.run_case.return_value = {"status": "ok"}
            capture.return_value.finish.return_value = ["trace.json"]
            result = ProfiledRunner(Mock(), Mock(), tmp, config, "session").run_case(
                {"case_id": "case"}
            )
            self.assertEqual(
                "one forward step; seeds excluded", result["capture_scope"]
            )

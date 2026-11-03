import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.modes.base import ModeExecutionContext
from rtp_llm.test.perf_test.modes.prefill import PrefillAdapter
from rtp_llm.test.perf_test.prefill_resume_test import _config, _plan
from rtp_llm.test.perf_test.prefill_runner_test import _response, _Tokenizer
from rtp_llm.test.perf_test.run_directory import initialize_run_directory
from rtp_llm.test.perf_test.storage import RunStore


class ProbeTest(unittest.TestCase):
    def test_probe_checks_seed_and_hit(self):
        from rtp_llm.test.perf_test.prefill import probe

        for broken_phase in (None, "seed", "measure"):
            requests = []

            class Client:
                def configure_scheduler(self, batch_size, mode):
                    self.batch = batch_size

                def request(self, **request):
                    requests.append(request)
                    reuse = 0 if request["phase"] == "seed" else 8
                    if request["phase"] == broken_phase:
                        reuse += 1
                    return _response(len(request["input_ids"]), reuse)

            client = Client()
            if broken_phase:
                with self.assertRaisesRegex(ValueError, "reuse"):
                    probe.probe_cache_block(client, _Tokenizer(), 8, 4)
            else:
                result = probe.probe_cache_block(client, _Tokenizer(), 8, 4)
                self.assertEqual("ok", result["status"])
                self.assertEqual([12, 16], [len(r["input_ids"]) for r in requests])
                self.assertEqual(
                    requests[0]["input_ids"][:8], requests[1]["input_ids"][:8]
                )
                self.assertNotEqual(
                    requests[0]["input_ids"][8], requests[1]["input_ids"][8]
                )
            self.assertEqual(1 if broken_phase == "seed" else 2, len(requests))

    def test_probe_rejects_invalid_shape_timing_and_failed_request(self):
        from rtp_llm.test.perf_test.prefill.probe import probe_cache_block

        for phase in ("seed", "measure"):
            for field, value in (
                ("success", False),
                ("input_len", 9),
                ("output_len", 2),
                ("latency_ms", 0),
                ("latency_ms", float("nan")),
            ):

                class Client:
                    def configure_scheduler(self, *args, **kwargs):
                        pass

                    def request(self, **request):
                        result = _response(
                            len(request["input_ids"]),
                            0 if request["phase"] == "seed" else 8,
                        )
                        if request["phase"] == phase:
                            result[field] = value
                        return result

                with self.subTest(
                    phase=phase, field=field, value=value
                ), self.assertRaises(ValueError):
                    probe_cache_block(Client(), _Tokenizer(), 8, 4)

    def test_probe_failure_persisted_before_any_case(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            plan = _plan(root)
            initialize_run_directory(
                root,
                source_config=_config(),
                plan=plan,
                run_id="probe",
                code_version="test",
            )
            calls = []

            class Runner:
                def run_case(self, case):
                    calls.append(case)

            def fail(_context):
                raise ValueError("wrong block reuse")

            adapter = PrefillAdapter(lambda context: Runner(), probe=fail)
            with self.assertRaisesRegex(ValueError, "wrong block reuse"):
                adapter(
                    ModeExecutionContext(
                        plan=plan, session=object(), store=RunStore(root)
                    )
                )
            self.assertEqual([], calls)
            self.assertEqual({}, RunStore(root).read_committed_cases())
            import json

            artifact = json.loads(
                (root / "checkpoints" / "prefill_cache_probe.json").read_text()
            )
            self.assertEqual("failed", artifact["status"])

    def test_block_size_precedence(self):
        from rtp_llm.test.perf_test.prefill.probe import resolve_probe_block_size

        config = {"workload": {"cache_alignment": 4}, "engine": {"args": {}}}
        self.assertEqual(4, resolve_probe_block_size(config))
        config["engine"]["args"]["seq_size_per_block"] = 8
        self.assertEqual(8, resolve_probe_block_size(config))
        config["workload"]["probe_block_size"] = 16
        self.assertEqual(16, resolve_probe_block_size(config))

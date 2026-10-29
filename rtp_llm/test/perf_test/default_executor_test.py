import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.execution.defaults import UnifiedLocalExecutor
from rtp_llm.test.perf_test.plan import build_execution_plan
from rtp_llm.test.perf_test.run_directory import initialize_run_directory
from rtp_llm.test.perf_test.state import RunState


class _Tokenizer:
    def encode(self, text):
        return text.split()


class _Client:
    port = 18000

    def request(self, **_request):
        return {
            "success": True,
            "input_len": 16,
            "output_len": 1,
            "reuse_len": 0,
            "latency_ms": 10.0,
        }


class _Session:
    def __init__(self):
        self.client = _Client()
        self.entered = 0
        self.exited = 0
        self.diagnostics = Mock()

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, _type, _value, _traceback):
        self.exited += 1


def _source(result_root):
    return {
        "schema_version": 1,
        "model": {"label": "model", "type": "registered", "checkpoint": "/m"},
        "engine": {
            "max_seq_len": 64,
            "max_batch_tokens": 64,
            "concurrency_limit": 1,
        },
        "workload": {
            "kind": "cache_grid",
            "max_input_len": 16,
            "cache_alignment": 4,
            "commit_tail_tokens": 4,
            "compute_steps": [4],
            "cache_lengths": [0],
            "batch_sizes": [1],
            "measure_runs": 1,
        },
        "execution": {"backend": "local", "devices": [0]},
        "output": {"result_root": str(result_root)},
        "regression": {},
        "publish": {},
    }


class DefaultExecutorTest(unittest.TestCase):
    def test_prefill_run_uses_one_session_and_finalizes_all_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            source = _source(root.parent)
            plan = build_execution_plan(
                mode="prefill",
                action="run",
                resolved=resolve_config(source),
                cases=[
                    {
                        "input_len": 16,
                        "cache_len": 0,
                        "compute_len": 16,
                        "batch_size": 1,
                        "measure_runs": 1,
                        "seed": 1,
                        "prefix_policy": "shared_seed",
                    }
                ],
                result_directory=str(root),
            )
            initialize_run_directory(
                root,
                source_config=source,
                plan=plan,
                run_id="run-local",
                code_version="test",
            )
            session = _Session()
            executor = UnifiedLocalExecutor(
                session_factory=Mock(return_value=session),
                tokenizer_loader=lambda _plan: _Tokenizer(),
                query_builder=lambda _plan, _lengths: {},
            )

            outcome = executor(plan)

            self.assertEqual(RunState.COMPLETED, outcome.state)
            self.assertEqual(1, session.entered)
            self.assertEqual(1, session.exited)
            self.assertTrue((root / "results" / "result.json").is_file())
            self.assertTrue((root / "results" / "summary.json").is_file())
            self.assertTrue((root / "reports" / "prefill_report.json").is_file())


if __name__ == "__main__":
    unittest.main()

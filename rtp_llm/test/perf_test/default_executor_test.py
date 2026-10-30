import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.config_schema import parse_config
from rtp_llm.test.perf_test.decode.model import DecodeBatchMeasurement
from rtp_llm.test.perf_test.decode.planner import plan_decode_cases
from rtp_llm.test.perf_test.errors import EngineLifecycleError
from rtp_llm.test.perf_test.execution.defaults import UnifiedLocalExecutor
from rtp_llm.test.perf_test.execution.engine_session import EngineDiagnostics
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


class _FailingSession:
    diagnostics = EngineDiagnostics(log_path="failed-engine.log", exit_code=2)

    def __enter__(self):
        raise EngineLifecycleError("engine failed readiness check")

    def __exit__(self, _type, _value, _traceback):
        return None


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


def _decode_source_and_plan(root, workload=None):
    source = _source(root.parent)
    source["model"]["topology"] = {"tp": 1, "dp": 1, "pp": 1, "ep": 1}
    source["workload"] = workload or {
        "kind": "grid",
        "input_lengths": [32],
        "output_lengths": [8],
        "batch": {"strategy": "fixed", "sizes": [1]},
        "measure_runs": 1,
        "warmup_runs": 0,
        "profiling": {"enabled": False},
    }
    resolved = resolve_config(source)
    parsed = parse_config(source)
    cases = plan_decode_cases(
        parsed.workload,
        parsed.engine,
        dp_size=1,
        resolved_workload=resolved.resolved_config["workload"],
    )
    return source, build_execution_plan(
        mode="decode",
        action="run",
        resolved=resolved,
        cases=cases,
        result_directory=str(root),
    )


def _request_round(input_lengths, output_length):
    return tuple(
        {
            "request_index": index,
            "success": True,
            "input_length": input_length,
            "output_length": output_length,
            "latency_ms": {
                "wait": 1.0,
                "prefill": 2.0,
                "decode": 15.0,
                "tpot": 1.0,
                "total": 17.0,
            },
        }
        for index, input_length in enumerate(input_lengths)
    )


def _measurement(batch, inputs, output):
    return DecodeBatchMeasurement(
        batch_size=batch,
        input_lengths=inputs,
        worker_allocations=(),
        measurements=(
            {
                "run": 0,
                "throughput_tokens_per_second": 100.0,
                "worker_allocations": [],
                "requests": list(_request_round(inputs, output)),
            },
        ),
    )


class DefaultExecutorTest(unittest.TestCase):
    def test_decode_run_uses_one_session_and_unified_case_executor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            source, plan = _decode_source_and_plan(root)
            initialize_run_directory(
                root,
                source_config=source,
                plan=plan,
                run_id="decode-local",
                code_version="test",
            )
            session = _Session()
            session.log_path = None
            batch_executor = Mock()
            batch_executor.measure.return_value = _measurement(
                batch=1, inputs=(32,), output=8
            )
            executor = UnifiedLocalExecutor(
                session_factory=Mock(return_value=session),
                tokenizer_loader=lambda _plan: _Tokenizer(),
                query_builder=lambda _plan, lengths: {
                    length: f"prompt-{length}" for length in lengths
                },
                decode_batch_executor_factory=Mock(return_value=batch_executor),
            )

            outcome = executor(plan)

            self.assertEqual(RunState.COMPLETED, outcome.state)
            self.assertEqual((1, 1), (session.entered, session.exited))
            self.assertEqual("ok", outcome.case_results[0]["status"])
            self.assertEqual(8, outcome.case_results[0]["metrics"]["output_length"])
            self.assertTrue((root / "reports" / "decode_report.json").is_file())

    def test_decode_output_grid_builds_distinct_queries_and_assembles_in_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            source, plan = _decode_source_and_plan(
                root,
                workload={
                    "kind": "grid",
                    "input_lengths": [32, 16],
                    "output_lengths": [4, 8],
                    "batch": {"strategy": "fixed", "sizes": [1]},
                    "measure_runs": 1,
                    "warmup_runs": 0,
                    "profiling": {"enabled": True},
                },
            )
            initialize_run_directory(
                root,
                source_config=source,
                plan=plan,
                run_id="decode-matrix",
                code_version="test",
            )
            session = _Session()
            session.log_path = None
            queried_lengths = []

            def build_queries(_plan, lengths):
                queried_lengths.append(list(lengths))
                return {length: f"prompt-{length}" for length in lengths}

            def measure(case, batch):
                self.assertEqual((1, 0), (session.entered, session.exited))
                return _measurement(
                    batch, case.input_lengths[:batch], case.output_length
                )

            def profile(*_args, **_kwargs):
                self.assertEqual((1, 0), (session.entered, session.exited))
                raise RuntimeError("profile unavailable")

            def batch_factory(_transport, *, engine, dp_size):
                self.assertEqual((1, 0), (session.entered, session.exited))
                self.assertEqual((64, 1), (engine.max_seq_len, dp_size))
                return Mock(measure=measure, profile=profile)

            executor = UnifiedLocalExecutor(
                session_factory=lambda _plan: session,
                query_builder=build_queries,
                decode_batch_executor_factory=batch_factory,
            )

            outcome = executor(plan)

            self.assertEqual(RunState.COMPLETED, outcome.state)
            self.assertEqual((1, 1), (session.entered, session.exited))
            self.assertEqual([[16, 32]], queried_lengths)
            self.assertEqual(4, len(outcome.case_results))
            self.assertCountEqual(
                [(32, 4), (32, 8), (16, 4), (16, 8)],
                [
                    (
                        item["metrics"]["input_lengths"][0],
                        item["metrics"]["output_length"],
                    )
                    for item in outcome.case_results
                ],
            )
            for result in outcome.case_results:
                self.assertEqual("ok", result["status"])
                profiling = result["metrics"]["profiling"]
                self.assertEqual("failed", profiling["status"])
                self.assertEqual("profile unavailable", profiling["error"])
                self.assertTrue((root / profiling["manifest_path"]).is_file())
            summary = json.loads((root / "results" / "summary.json").read_text())
            self.assertEqual(4, summary["profiling_failed_cases"])

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

    def test_engine_failure_preserves_typed_outcome_without_finalizing_cases(self):
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
                run_id="run-engine-failure",
                code_version="test",
            )
            executor = UnifiedLocalExecutor(
                session_factory=Mock(return_value=_FailingSession()),
                tokenizer_loader=lambda _plan: _Tokenizer(),
                query_builder=lambda _plan, _lengths: {},
            )

            outcome = executor(plan)

            self.assertEqual(RunState.FAILED, outcome.state)
            self.assertEqual(4, outcome.exit_code)
            self.assertFalse((root / "results" / "result.json").exists())
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual("failed", manifest["state"])


if __name__ == "__main__":
    unittest.main()

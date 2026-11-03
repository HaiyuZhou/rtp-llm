import json
import struct
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

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

    def configure_scheduler(self, batch_size, mode="prefill"):
        assert batch_size == 1
        assert mode == "prefill"

    def request(self, **_request):
        return {
            "success": True,
            "input_len": len(_request["input_ids"]),
            "output_len": 1,
            "reuse_len": (
                4
                if _request["case_id"].startswith("cache_probe_")
                and _request["phase"] == "measure"
                else 0
            ),
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
    def test_profile_nsys_launch_is_scoped_and_preserves_default_launch(self):
        from dataclasses import replace

        from rtp_llm.test.perf_test.execution.runtime import create_maga_handle
        from rtp_llm.test.perf_test.plan import freeze, thaw
        from rtp_llm.test.perf_test.prefill_profile_test import _setup

        with tempfile.TemporaryDirectory() as directory:
            plan = _setup(Path(directory) / "run")
            config = thaw(plan.resolved_config)
            config["workload"]["profiling"] = {
                "backend": "nsys",
                "nsys_path": "/opt/nsys",
            }
            config["engine"]["args"]["torch_cuda_profiler_dir"] = "/formal"
            plan = replace(plan, resolved_config=freeze(config))
            manager = Mock()
            manager.return_value.port = 123
            with patch.dict(
                "sys.modules",
                {
                    "rtp_llm.test.utils.maga_server_manager": SimpleNamespace(
                        MagaServerManager=manager
                    )
                },
            ):
                handle = create_maga_handle(
                    plan, profile_directory=directory, nsys_session="unique"
                )
                options = manager.call_args.kwargs
                self.assertEqual(
                    [
                        "/opt/nsys",
                        "launch",
                        "--session-new=unique",
                        "--show-output=true",
                        "--trace=cuda,nvtx,osrt",
                        "--wait=all",
                    ],
                    options["process_prefix"],
                )
                self.assertIn(
                    "--torch_cuda_profiler_dir=" + directory, options["smoke_args_str"]
                )
                self.assertNotIn("/formal", options["smoke_args_str"])
                handle.stop()
                manager.return_value.stop_server.assert_called_once()
                create_maga_handle(plan)
                self.assertNotIn("process_prefix", manager.call_args.kwargs)
                self.assertIn("/formal", manager.call_args.kwargs["smoke_args_str"])

    def test_profile_missing_trace_fails_and_closes_engine(self):
        from dataclasses import replace

        from rtp_llm.test.perf_test.prefill.trace import TraceCapture
        from rtp_llm.test.perf_test.prefill_profile_test import _formal_bytes, _setup

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            plan = replace(_setup(root), action="profile")
            before = _formal_bytes(root)
            session = _Session()

            def capture(*args, **kwargs):
                kwargs["timeout"] = 0.001
                return TraceCapture(
                    *args,
                    **kwargs,
                    post=Mock(return_value=Mock(json=lambda: {"status": "ok"})),
                )

            with patch(
                "rtp_llm.test.perf_test.prefill.trace.TraceCapture", side_effect=capture
            ):
                outcome = UnifiedLocalExecutor(
                    session_factory=lambda _: session,
                    tokenizer_loader=lambda _: _Tokenizer(),
                )(plan)
            self.assertEqual("failed", outcome["status"])
            self.assertIn(
                "trace timeout", outcome["results"][0]["errors"][0]["message"]
            )
            self.assertEqual(1, session.exited)
            self.assertEqual(before, _formal_bytes(root))

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
            self.assertTrue(
                (root / "checkpoints" / "prefill_cache_probe.json").is_file()
            )
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


class PrefillBackendIntegrationTest(unittest.TestCase):
    def test_batch_four_uses_shared_http_runner_across_backends(self):
        self._batch_four_across_backends("http_text")

    def test_batch_four_uses_exact_dashsc_ids_across_backends(self):
        self._batch_four_across_backends("dashsc_input_ids")

    def _batch_four_across_backends(self, transport):
        from rtp_llm.test.perf_test.aone.collector import collect_aone_results
        from rtp_llm.test.perf_test.aone.gang_worker import (
            _default_prefill_master_executor,
        )
        from rtp_llm.test.perf_test.aone.package import generate_aone_package
        from rtp_llm.test.perf_test.aone.sharding import plan_shards
        from rtp_llm.test.perf_test.aone.worker import (
            _default_executor,
            load_remote_inputs,
        )
        from rtp_llm.test.perf_test.execution.runtime import HttpPrefillClient
        from rtp_llm.test.perf_test.prefill.planner import plan_prefill_cases

        for backend in ("local", "aone", "gang"):
            with self.subTest(
                backend=backend
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run_root = root / "run"
                source = _source(root)
                source["workload"]["transport"] = transport
                source["engine"]["concurrency_limit"] = 4
                source["workload"].update(batch_sizes=[4], compute_steps=[8, 4])
                if backend == "aone":
                    source["execution"] = {
                        "backend": "aone",
                        "resource": "daily",
                        "gpu_type": "H20",
                        "max_parallel": 1,
                        "shard_count": 1,
                    }
                elif backend == "gang":
                    example = Path(__file__).parent / "examples/prefill_aone_gang.json"
                    gang_source = json.loads(example.read_text())
                    source["execution"] = gang_source["execution"]
                    source["model"]["topology"] = gang_source["model"]["topology"]
                parsed = parse_config(source)
                plan = build_execution_plan(
                    mode="prefill",
                    action="run",
                    resolved=resolve_config(source),
                    cases=plan_prefill_cases(parsed.workload, parsed.engine),
                    result_directory=str(run_root),
                )
                events = []
                wire_requests = []
                barrier = threading.Barrier(4, timeout=3)
                lock = threading.Lock()
                endpoint = "http://fake-prefill:18000"

                def post(url, *, json, **_kwargs):
                    with lock:
                        events.append((url, json))
                    if url.endswith("/update_scheduler_info"):
                        self.assertIn(
                            json,
                            (
                                {"batch_size": 4, "mode": "prefill"},
                                {"batch_size": 1, "mode": "prefill"},
                            ),
                        )
                        return Mock(status_code=200, json=lambda: {"status": "ok"})
                    self.assertEqual(endpoint, url)
                    is_probe = "cache_probe_" in json["prompt"]
                    if not is_probe:
                        barrier.wait()  # Sequential submission cannot complete this round.
                    return Mock(
                        status_code=200,
                        json=lambda: {
                            "aux_info": {
                                "input_len": len(json["prompt"].split()),
                                "output_len": 1,
                                "reuse_len": (
                                    4 if is_probe and "__run_" in json["prompt"] else 0
                                ),
                                "first_token_cost_time": 10.0,
                            }
                        },
                    )

                http_client = HttpPrefillClient(
                    endpoint, post=post, scheduler_attempts=1
                )
                client = http_client
                tokenizer = _Tokenizer()
                if transport == "dashsc_input_ids":
                    from rtp_llm.dash_sc.proto import predict_v2_pb2
                    from rtp_llm.test.perf_test.prefill.dashsc_transport import (
                        DashScPrefillClient,
                    )

                    class Tokenizer:
                        def __init__(self):
                            self.tokens = {}

                        def encode(self, text):
                            with lock:
                                return [
                                    self.tokens.setdefault(word, len(self.tokens) + 1)
                                    for word in text.split()
                                ]

                    tokenizer = Tokenizer()

                    class Stub:
                        def ModelStreamInfer(self, requests, **_kwargs):
                            (request,) = list(requests)
                            request = predict_v2_pb2.ModelInferRequest.FromString(
                                request.SerializeToString()
                            )
                            wire_requests.append(request)
                            ids = next(
                                raw
                                for tensor, raw in zip(
                                    request.inputs, request.raw_input_contents
                                )
                                if tensor.name == "input_ids"
                            )
                            values = struct.unpack("<" + "i" * (len(ids) // 4), ids)
                            # Decode IDs solely to reuse the same fake server contract;
                            # the real client sent no text prompt.
                            with lock:
                                words = {
                                    value: word
                                    for word, value in tokenizer.tokens.items()
                                }
                            prompt = " ".join(words[value] for value in values)
                            payload = post(endpoint, json={"prompt": prompt}).json()[
                                "aux_info"
                            ]
                            response = predict_v2_pb2.ModelStreamInferResponse()
                            infer = response.infer_response
                            infer.parameters[
                                "engine_first_token_cost_time_us"
                            ].int64_param = 10000
                            for name, value in (
                                ("prompt_token_num", payload["input_len"]),
                                ("prompt_cached_token_num", payload["reuse_len"]),
                                ("generated_ids", 42),
                            ):
                                output = infer.outputs.add(name=name, datatype="INT32")
                                output.shape.append(1)
                                infer.raw_output_contents.append(
                                    struct.pack("<i", value)
                                )
                            return iter([response])

                    client = DashScPrefillClient(
                        http_client, channel=Mock(), stub=Stub()
                    )
                    client.wait_ready = Mock()

                session = _Session()
                session.client = client
                session.diagnostics = EngineDiagnostics()
                with patch(
                    "rtp_llm.test.perf_test.test_util._load_tokenizer",
                    return_value=tokenizer,
                ):
                    if backend == "local":
                        initialize_run_directory(
                            run_root,
                            source_config=source,
                            plan=plan,
                            run_id="integration",
                            code_version="test",
                        )
                        outcome = UnifiedLocalExecutor(
                            session_factory=lambda _plan: session,
                            tokenizer_loader=lambda _plan: tokenizer,
                        )(plan)
                        self.assertEqual(RunState.COMPLETED, outcome.state)
                        results = outcome.case_results
                    elif backend == "aone":
                        shards = plan_shards(plan)
                        package = generate_aone_package(
                            root / "package",
                            plan,
                            run_id="integration",
                            ownership=shards.ownership,
                        )
                        inputs = load_remote_inputs(
                            resolved_config=package.resolved_config,
                            resolved_workload=package.resolved_workload,
                            shard_plan=package.shard_plan,
                            shard_id="shard-0000",
                        )
                        with patch(
                            "rtp_llm.test.perf_test.aone.worker.resolve_remote_runtime_inputs",
                            side_effect=lambda value: value,
                        ), patch(
                            "rtp_llm.test.perf_test.execution.defaults._default_session_factory",
                            return_value=session,
                        ):
                            remote = _default_executor(inputs).execute(inputs, run_root)
                        collected = collect_aone_results(
                            plan,
                            run_id="integration",
                            shard_plan=shards,
                            results=[remote.to_dict()],
                        )
                        self.assertEqual(RunState.COMPLETED, collected.state)
                        results = collected.cases
                    else:
                        config_path = root / "config.json"
                        config_path.write_text(json.dumps(source))
                        topology = source["model"]["topology"]
                        context = SimpleNamespace(
                            gang_plan={
                                "topology": topology,
                                "port_range": {"start": 18000},
                            }
                        )
                        with patch(
                            "rtp_llm.test.perf_test.aone.gang_worker._resolve_model_paths",
                            return_value=source["model"],
                        ), patch(
                            "rtp_llm.test.perf_test.multi_node.prefill_runtime.HttpPrefillClient",
                            return_value=http_client,
                        ) as client_factory, patch(
                            "rtp_llm.test.perf_test.prefill.dashsc_transport.DashScPrefillClient",
                            return_value=client,
                        ):
                            source["model"]["tokenizer"] = "/m"
                            code = _default_prefill_master_executor(
                                SimpleNamespace(),
                                context=context,
                                config_path=str(config_path),
                                result_directory=str(run_root),
                            )
                        self.assertEqual(0, code)
                        client_factory.assert_called_once_with("http://127.0.0.1:18000")
                        from rtp_llm.test.perf_test.storage import RunStore

                        results = tuple(
                            RunStore(run_root).read_committed_cases().values()
                        )
                self.assertCountEqual(
                    [case.case_id for case in plan.cases],
                    [result["case_id"] for result in results],
                )
                self.assertGreater(len(results), 1)
                self.assertEqual(len(plan.cases) * 5 + 3, len(events))
                if transport == "dashsc_input_ids":
                    self.assertEqual(len(plan.cases) * 4 + 2, len(wire_requests))
                self.assertEqual({"batch_size": 1, "mode": "prefill"}, events[0][1])
                self.assertTrue(
                    (run_root / "checkpoints" / "prefill_cache_probe.json").is_file()
                )
                events = events[3:]
                for offset in range(0, len(events), 5):
                    self.assertEqual(
                        endpoint + "/update_scheduler_info", events[offset][0]
                    )
                    self.assertEqual(
                        [endpoint] * 4,
                        [url for url, _ in events[offset + 1 : offset + 5]],
                    )
                for result in results:
                    self.assertEqual("ok", result["status"])
                    self.assertEqual(
                        "concurrent_batch_v1",
                        result["identity"]["measurement_contract"],
                    )
                    records = result["metrics"]["measurements"]
                    self.assertEqual(
                        [0, 1, 2, 3], [record["request_index"] for record in records]
                    )
                    self.assertTrue(
                        all(record["round_elapsed_ms"] > 0 for record in records)
                    )


if __name__ == "__main__":
    unittest.main()

"""Tests for aone gang_worker_test."""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm.test.perf_test.aone.broker import BrokerMember
from rtp_llm.test.perf_test.aone.gang_worker import (
    EXIT_RENDEZVOUS_ERROR,
    GangBrokerFailure,
    _default_decode_worker,
    _default_master_executor,
    _default_prefill_master_executor,
    _default_server_runner,
    build_worker_context,
    run_gang_worker,
)
from rtp_llm.test.perf_test.execution.aone_gang import BROKER_TOKEN_ENV
from rtp_llm.test.perf_test.results import parse_result_envelope
from rtp_llm.test.perf_test.state import RunState
from rtp_llm.test.perf_test.tests.support.gang import _gang_source
from rtp_llm.test.perf_test.tests.support.paths import PERF_TEST_ROOT


class GangWorkerTest(unittest.TestCase):

    def test_worker_acknowledges_bazel_machine_sharding(self):
        status = self.root / "shard.status"
        with patch.dict("os.environ", {"TEST_SHARD_STATUS_FILE": str(status)}):
            self.assertEqual(0, self._run(0, lambda *args, **kwargs: 0))
        self.assertTrue(status.is_file())

    def test_broker_token_is_not_in_rank_process_environment_or_failure_artifacts(self):
        import os

        secret = "test-broker-secret-do-not-persist"

        def launch(context):
            self.assertNotIn(BROKER_TOKEN_ENV, os.environ)
            return self.handle

        with patch.dict(
            "os.environ",
            {
                BROKER_TOKEN_ENV: secret,
                "TEST_SHARD_INDEX": "0",
                "TEST_TOTAL_SHARDS": "2",
            },
        ):
            code = run_gang_worker(
                self.config_path,
                self.plan_path,
                broker=self.broker,
                server_runner=launch,
                test_outputs_dir=self.root / "outputs",
                master_executor=lambda *args, **kwargs: 0,
            )
        self.assertEqual(0, code)
        self.assertNotIn(secret, (self.root / "outputs/gang_worker.json").read_text())

    def test_decode_worker_resolves_its_own_tokenizer(self):
        self._example("decode")
        handle = Mock(tokenizer_path=None)
        with patch(
            "rtp_llm.test.perf_test.aone.gang_worker._resolve_model_paths",
            return_value={"tokenizer": "/local/tokenizer"},
        ) as resolve, patch(
            "rtp_llm.test.perf_test.test_util.create_query",
            return_value={8: "prompt"},
        ) as create, patch(
            "rtp_llm.test.perf_test.multi_node.decode_sync.serve_decode_worker"
        ) as serve:
            _default_decode_worker(
                handle,
                context=build_worker_context(self.gang, self.members, 1),
                config_path=str(self.config_path),
            )
        resolve.assert_called_once()
        self.assertEqual("/local/tokenizer", create.call_args.kwargs["tokenizer_path"])
        serve.assert_called_once()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.gang = {
            "schema_version": 1,
            "run_id": "gang-core",
            "config_fingerprint": "fp",
            "workload_fingerprint": "wf",
            "world_size": 2,
            "machine_count": 2,
            "local_world_size": 1,
            "gpus_per_machine": 1,
            "gpus_per_host": 8,
            "gpu_type": "H20_CU13",
            "topology": {"tp": 2, "dp": 1, "pp": 1, "ep": 1},
            "master_shard_index": 0,
            "port_range": {"start": 18000, "count": 1},
        }
        self.members = [
            BrokerMember(index, f"10.0.0.{index}", f"host-{index}", 18000)
            for index in range(2)
        ]
        self.context = build_worker_context(self.gang, self.members, 0)
        self.handle = Mock(
            tokenizer_path="/m",
            log_file_path=str(self.root / "server.log"),
            log_dir=str(self.root / "logs"),
        )
        self.handle.start_server.return_value = True
        self.broker = Mock(gang_state="ready")
        self.broker.members.return_value = self.members
        self.plan_path = self.root / "gang.json"
        self.plan_path.write_text(json.dumps(self.gang))
        self.config_path = self.root / "config.json"
        self.config_path.write_text("{}")
        self.addCleanup(self._assert_no_heartbeats)

    def _assert_no_heartbeats(self):
        self.assertEqual(
            [],
            [
                t
                for t in threading.enumerate()
                if t.name.startswith("gang-worker-heartbeat") and t.is_alive()
            ],
        )

    def _example(self, mode):
        source = _gang_source()
        source["model"]["topology"] = {"tp": 2, "dp": 1, "pp": 1, "ep": 1}
        source["execution"]["gpus_per_machine"] = 1
        if mode == "decode":
            source["workload"] = {
                "kind": "grid",
                "input_lengths": [8],
                "output_lengths": [2],
                "batch": {"strategy": "fixed", "sizes": [1]},
                "warmup_runs": 0,
                "measure_runs": 1,
                "profiling": {"enabled": False},
            }
        source["engine"].update(
            max_seq_len=64, max_batch_tokens=64, concurrency_limit=1
        )
        if mode == "prefill":
            source["workload"].update(
                max_input_len=16,
                cache_alignment=4,
                commit_tail_tokens=4,
                compute_steps=[4],
                cache_lengths=[0],
                batch_sizes=[1],
            )
        else:
            source["workload"].update(
                input_lengths=[8],
                output_lengths=[2],
                warmup_runs=0,
                measure_runs=1,
                profiling={"enabled": False},
            )
        self.config_path.write_text(json.dumps(source))
        return source

    def test_model_startup_propagates_args_rank_layout_and_disables_cross_host_all_reduce(
        self,
    ):
        self._example("prefill")
        runner = Mock()
        runner.LocalRankGroupContext.side_effect = lambda **kwargs: SimpleNamespace(
            **kwargs
        )
        configs = SimpleNamespace(
            parallelism_config=object(), prefill_cp_config=object()
        )
        with patch.dict(
            "os.environ",
            {
                "TEST_UNDECLARED_OUTPUTS_DIR": str(self.root),
                "RTP_LLM_GPU_LOCKED_DEVICES": "4,5,6,7,0,1,2,3",
            },
        ), patch(
            "rtp_llm.test.perf_test.aone.gang_worker._load_local_server_runner",
            return_value=runner,
        ), patch(
            "rtp_llm.test.perf_test.aone.gang_worker._resolve_model_paths",
            return_value={
                "type": "qwen_2",
                "checkpoint": "/resolved/model",
                "tokenizer": "/resolved/tokenizer",
            },
        ), patch(
            "rtp_llm.server.server_args.server_args.setup_args", return_value=configs
        ) as setup, patch(
            "rtp_llm.config.server_config_setup.set_parallelism_config"
        ):
            context = build_worker_context(self.gang, self.members, 1)
            _default_server_runner(context, config_path=str(self.config_path))
            import os

            self.assertEqual("1", os.environ["FT_DISABLE_CUSTOM_AR"])
        (args, kwargs) = runner.start_local_rank_group.call_args
        self.assertEqual(1, args[0].world_rank)
        self.assertEqual("4", context.cuda_visible_devices)
        self.assertIn("host-1", context.gang_config_string)
        self.assertIn("--tp_size=2", kwargs["engine_argv"])
        self.assertIn("--reuse_cache=1", kwargs["engine_argv"])
        self.assertIn("--checkpoint_path=/resolved/model", kwargs["engine_argv"])
        self.assertIn("--tokenizer_path=/resolved/tokenizer", kwargs["engine_argv"])
        self.assertEqual(list(kwargs["engine_argv"]), setup.call_args.args[0])
        self.assertEqual("2", kwargs["engine_environment"]["WORLD_SIZE"])
        self.assertEqual("4", kwargs["engine_environment"]["CUDA_VISIBLE_DEVICES"])

    def test_prefill_master_executes_cases_probe_and_finalizes_owned_results(self):
        source = self._example("prefill")
        client = Mock()

        def request(**kwargs):
            return {
                "success": True,
                "input_len": len(kwargs["input_ids"]),
                "output_len": 1,
                "reuse_len": (
                    4
                    if kwargs["case_id"].startswith("cache_probe_")
                    and kwargs["phase"] == "measure"
                    else 0
                ),
                "latency_ms": 10.0,
            }

        client.request.side_effect = request
        session = SimpleNamespace(client=client, close=Mock())
        output = self.root / "prefill"
        with patch(
            "rtp_llm.test.perf_test.aone.gang_worker._resolve_model_paths",
            return_value=source["model"],
        ), patch(
            "rtp_llm.test.perf_test.test_util._load_tokenizer",
            return_value=SimpleNamespace(encode=lambda text: text.split()),
        ), patch(
            "rtp_llm.test.perf_test.multi_node.prefill_runtime.ExternalPrefillSession",
            return_value=session,
        ):
            code = _default_prefill_master_executor(
                self.handle,
                context=self.context,
                config_path=str(self.config_path),
                result_directory=str(output),
            )
        self.assertEqual(0, code)
        parsed = parse_result_envelope(
            json.loads((output / "results/result.json").read_text())
        )
        self.assertEqual(RunState.COMPLETED, parsed.status)
        self.assertTrue(parsed.cases)
        self.assertTrue(all((case.status == "ok" for case in parsed.cases)))
        self.assertTrue((output / "checkpoints/prefill_cache_probe.json").is_file())
        session.close.assert_called_once()

    def test_prefill_tokenizer_failure_marks_initialized_run_failed(self):
        source = self._example("prefill")
        output = self.root / "prefill-tokenizer-failure"
        with patch(
            "rtp_llm.test.perf_test.aone.gang_worker._resolve_model_paths",
            return_value=source["model"],
        ), patch(
            "rtp_llm.test.perf_test.test_util._load_tokenizer",
            side_effect=ValueError("tokenizer unavailable"),
        ):
            with self.assertRaisesRegex(ValueError, "tokenizer unavailable"):
                _default_prefill_master_executor(
                    self.handle,
                    context=self.context,
                    config_path=str(self.config_path),
                    result_directory=str(output),
                )
        self.assertEqual(
            "failed", json.loads((output / "manifest.json").read_text())["state"]
        )

    def test_decode_master_executes_and_finalizes_aone_envelope(self):
        self._example("decode")
        transport = Mock()
        transport.run_round.return_value = (
            {
                "request_index": 0,
                "input_length": 8,
                "output_length": 2,
                "success": True,
                "latency_ms": {
                    "wait": 0.0,
                    "prefill": 2.0,
                    "decode": 8.0,
                    "tpot": 4.0,
                    "total": 10.0,
                },
            },
        )
        output = self.root / "decode"
        with patch(
            "rtp_llm.test.perf_test.test_util.create_query", return_value={8: "prompt"}
        ), patch(
            "rtp_llm.test.perf_test.multi_node.decode_runtime.DecodeHttpTransport",
            return_value=transport,
        ) as factory:
            code = _default_master_executor(
                self.handle,
                context=self.context,
                config_path=str(self.config_path),
                result_directory=str(output),
            )
        self.assertEqual(0, code)
        parsed = parse_result_envelope(
            json.loads((output / "results/result.json").read_text())
        )
        self.assertEqual("decode", parsed.mode)
        self.assertEqual("aone", parsed.backend)
        self.assertTrue(parsed.cases)
        self.assertTrue(all((case.status == "ok" for case in parsed.cases)))
        self.assertEqual("http://127.0.0.1:18000", factory.call_args.kwargs["endpoint"])
        transport.configure_scheduler.assert_called()

    def test_rank_batch_decode_master_uses_cluster_transport(self):
        self._check_decode_master_cluster(
            {
                "kind": "grid",
                "input_lengths": [8],
                "batch": {"strategy": "fixed", "sizes": [1], "unit": "per_dp_rank"},
            }
        )

    def test_global_grid_and_distribution_master_use_cluster_transport(self):
        for workload in (
            {
                "kind": "grid",
                "input_lengths": [8],
                "batch": {"strategy": "fixed", "sizes": [2], "unit": "global"},
            },
            {
                "kind": "distribution",
                "distribution_file": "not-on-worker.json",
                "batch": {"strategy": "fixed", "sizes": [2]},
            },
        ):
            with self.subTest(kind=workload["kind"], batch=workload["batch"]):
                self._check_decode_master_cluster(workload)

    def _check_decode_master_cluster(self, workload):
        source = self._example("decode")
        source["model"]["topology"] = {"tp": 1, "dp": 2, "pp": 1, "ep": 2}
        source["workload"] = workload
        self.config_path.write_text(json.dumps(source))
        self.gang["topology"] = source["model"]["topology"]
        context = build_worker_context(self.gang, self.members, 0)
        plan = SimpleNamespace(cases=[SimpleNamespace(identity={"input_lengths": [8]})])
        cluster = Mock()
        with patch(
            "rtp_llm.test.perf_test.multi_node.decode_runtime.build_external_decode_plan",
            return_value=(plan, source),
        ), patch(
            "rtp_llm.test.perf_test.test_util.create_query", return_value={8: "prompt"}
        ), patch(
            "rtp_llm.test.perf_test.multi_node.decode_sync.ClusterDecodeTransport",
            return_value=cluster,
        ), patch(
            "rtp_llm.test.perf_test.multi_node.decode_runtime.execute_master_decode",
            return_value=SimpleNamespace(state=RunState.COMPLETED),
        ) as execute:
            code = _default_master_executor(
                self.handle,
                context=context,
                config_path=str(self.config_path),
                result_directory=str(self.root / "run"),
            )
        self.assertEqual(0, code)
        cluster.connect_workers.assert_called_once()
        cluster.close.assert_called_once()
        self.assertIs(cluster, execute.call_args.kwargs["transport"])

    def _run(self, rank, executor=None):
        with patch.dict(
            "os.environ", {"TEST_SHARD_INDEX": str(rank), "TEST_TOTAL_SHARDS": "2"}
        ):
            return run_gang_worker(
                self.config_path,
                self.plan_path,
                broker=self.broker,
                server_runner=lambda context: self.handle,
                test_outputs_dir=self.root / "outputs",
                master_executor=executor,
            )

    def test_master_failure_votes_failed_stops_server_and_preserves_partial_artifacts(
        self,
    ):

        def execute(handle, *, result_directory, **kwargs):
            output = Path(result_directory) / "partial.json"
            output.parent.mkdir(parents=True)
            output.write_text("{}")
            return 7

        self.assertEqual(7, self._run(0, execute))
        self.assertEqual("failed", self.broker.report.call_args.args[1])
        self.assertTrue((self.root / "outputs/run/partial.json").is_file())
        self.handle.stop_server.assert_called_once()
        self.assertEqual(
            "failed",
            json.loads((self.root / "outputs/gang_worker.json").read_text())["state"],
        )

    def test_waiting_worker_terminates_on_gang_failure_and_stops_server(self):
        self.broker.members.side_effect = [
            self.members,
            GangBrokerFailure("member_failed"),
        ]
        self.assertEqual(EXIT_RENDEZVOUS_ERROR, self._run(1))
        self.assertEqual(2, self.broker.members.call_count)
        self.handle.start_server.assert_called_once()
        self.handle.stop_server.assert_called_once()

    def test_rank_batch_decode_worker_measures_before_completion_vote(self):
        self._check_decode_worker_dispatch("per_dp_rank")

    def test_global_decode_worker_measures_before_completion_vote(self):
        self._check_decode_worker_dispatch("global")

    def _check_decode_worker_dispatch(self, unit):
        source = self._example("decode")
        source["model"]["topology"] = {"tp": 1, "dp": 2, "pp": 1, "ep": 2}
        source["workload"]["batch"] = {
            "strategy": "fixed",
            "sizes": [1],
            "unit": unit,
        }
        self.config_path.write_text(json.dumps(source))
        self.gang["topology"] = source["model"]["topology"]
        self.plan_path.write_text(json.dumps(self.gang))
        with patch(
            "rtp_llm.test.perf_test.aone.gang_worker._default_decode_worker"
        ) as measure, patch(
            "rtp_llm.test.perf_test.aone.gang_worker._run_waiter",
            return_value=0,
        ) as waiter:
            self.assertEqual(0, self._run(1))
        measure.assert_called_once()
        waiter.assert_called_once()


class FrozenGangInputsTest(unittest.TestCase):
    def test_accepted_workloads_reach_worker_startup_with_identical_plans(self):
        from rtp_llm.test.perf_test.aone import gang_worker
        from rtp_llm.test.perf_test.aone.package import generate_gang_package
        from rtp_llm.test.perf_test.cli import parse_cli
        from rtp_llm.test.perf_test.entry import build_request_plan

        examples = PERF_TEST_ROOT / "examples"
        shapes = (
            ("decode_distribution_local.json", "decode", 2),
            ("prefill_random_batch_local.json", "prefill", 12),
            ("prefill_traffic_distribution_local.json", "prefill", 12),
            ("prefill_local.json", "prefill", None),
        )
        for example, mode, count in shapes:
            for named in (
                (False, True)
                if "batch" in example or "traffic" in example
                else (False,)
            ):
                with self.subTest(
                    example=example, named=named
                ), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    source = json.loads((examples / example).read_text())
                    source["model"]["topology"].update(
                        tp=8,
                        dp=2 if mode == "decode" else 1,
                        ep=16 if mode == "decode" else 1,
                    )
                    source["execution"] = dict(
                        backend="aone",
                        resource="daily",
                        gpu_type="H20",
                        gpus_per_machine=8 if mode == "decode" else 4,
                        max_parallel=4,
                    )
                    workload = source["workload"]
                    spec = (
                        workload
                        if mode == "decode"
                        else workload.get("traffic_request_batches", {})
                    )
                    if "distribution_file" in spec:
                        external = root / "external.json"
                        external.write_text(
                            (examples / spec["distribution_file"]).read_text()
                        )
                        spec["distribution_file"] = str(external)
                    if named:
                        key = (
                            "traffic_request_batches"
                            if "traffic" in example
                            else "random_request_batches"
                        )
                        workload["grids"] = [
                            dict(name="sampled", **{key: workload.pop(key)})
                        ]
                    source_path = root / "source.json"
                    source_path.write_text(json.dumps(source))
                    original, _ = build_request_plan(
                        parse_cli(
                            [
                                f"--mode={mode}",
                                "--action=run",
                                f"--config={source_path}",
                            ]
                        )
                    )
                    if count is not None:
                        self.assertEqual(count, len(original.cases))
                    package = generate_gang_package(
                        root / "package", original, run_id="frozen"
                    )
                    source_path.unlink()
                    (root / "external.json").unlink(missing_ok=True)
                    outputs = root / "outputs"
                    context = SimpleNamespace(
                        gang_plan=json.loads(package.shard_plan.read_text())
                    )
                    seen = []

                    def startup(config_path, *_args, **_kwargs):
                        plan, _ = gang_worker._build_rank_group_plan(
                            str(config_path), context, outputs
                        )
                        self.assertEqual(
                            original.config_fingerprint, plan.config_fingerprint
                        )
                        self.assertEqual(
                            original.workload_fingerprint, plan.workload_fingerprint
                        )
                        self.assertEqual(original.cases, plan.cases)
                        self.assertEqual(original.resolved_config, plan.resolved_config)
                        if mode == "decode":
                            worker_context = SimpleNamespace(
                                gang_plan=context.gang_plan,
                                local_world_size=original.resource_plan.local_world_size,
                                world_rank=original.resource_plan.local_world_size,
                                members=[
                                    SimpleNamespace(shard_index=0, hostname="master")
                                ],
                            )
                            with patch(
                                "rtp_llm.test.perf_test.test_util.create_query",
                                return_value={},
                            ) as queries, patch(
                                "rtp_llm.test.perf_test.multi_node.decode_sync.serve_decode_worker"
                            ) as serve:
                                gang_worker._default_decode_worker(
                                    SimpleNamespace(tokenizer_path="/tokenizer"),
                                    context=worker_context,
                                    config_path=str(config_path),
                                )
                            self.assertEqual(
                                sorted(
                                    {
                                        n
                                        for case in plan.cases
                                        for n in case.identity["input_lengths"]
                                    }
                                ),
                                queries.call_args.kwargs["input_len_list"],
                            )
                            serve.assert_called_once()
                        for field in ("config_fingerprint", "workload_fingerprint"):
                            expected = context.gang_plan[field]
                            context.gang_plan[field] = "0" * 64
                            with self.assertRaisesRegex(Exception, field + " mismatch"):
                                gang_worker._build_rank_group_plan(
                                    str(config_path), context, outputs
                                )
                            context.gang_plan[field] = expected
                        seen.append(plan)
                        return 0

                    gang_payload = json.loads(package.shard_plan.read_text())
                    gang_payload["broker_run_id"] = "frozen"
                    gang_payload["broker_url"] = "http://localhost:1234"
                    package.shard_plan.write_text(json.dumps(gang_payload))
                    with patch.dict(
                        "os.environ",
                        {
                            "TEST_UNDECLARED_OUTPUTS_DIR": str(outputs),
                            "PERF_TEST_BROKER_TOKEN": "test",
                        },
                    ), patch.object(
                        gang_worker, "run_gang_worker", side_effect=startup
                    ):
                        code = gang_worker.main(
                            [
                                "--resolved-config",
                                str(package.resolved_config),
                                "--resolved-workload",
                                str(package.resolved_workload),
                                "--gang-plan",
                                str(package.shard_plan),
                            ],
                            broker_factory=Mock(),
                        )
                    self.assertEqual(0, code)
                    self.assertEqual(1, len(seen))


if __name__ == "__main__":
    unittest.main()

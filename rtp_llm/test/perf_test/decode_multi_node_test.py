import ast
import copy
import itertools
import json
import logging
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from rtp_llm.test.perf_test.cli import CliRequest
from rtp_llm.test.perf_test.decode.transport import DecodeHttpTransport
from rtp_llm.test.perf_test.entry import build_request_plan
from rtp_llm.test.perf_test.errors import (
    ConfigurationError,
    PartialRunError,
    PlanningError,
)
from rtp_llm.test.perf_test.multi_node import multi_benchmark
from rtp_llm.test.perf_test.multi_node.decode_runtime import (
    build_external_decode_plan,
    execute_master_decode,
    run_external_rank,
)
from rtp_llm.test.perf_test.multi_node.multi_benchmark import load_config
from rtp_llm.test.perf_test.state import RunState


def _decode_source(*, inputs=(128,), outputs=(16, 64), batch=None):
    return {
        "schema_version": 1,
        "model": {
            "label": "external-decode",
            "type": "registered",
            "checkpoint": "/m",
            "topology": {"tp": 1, "dp": 2, "pp": 1, "ep": 1},
        },
        "engine": {
            "max_seq_len": 4096,
            "max_batch_tokens": 8192,
            "concurrency_limit": 8,
        },
        "workload": {
            "kind": "grid",
            "input_lengths": list(inputs),
            "output_lengths": list(outputs),
            "batch": batch or {"strategy": "fixed", "sizes": [1, 4]},
            "measure_runs": 1,
            "warmup_runs": 0,
        },
        "execution": {"backend": "local", "devices": [0, 1]},
        "output": {},
        "regression": {},
        "publish": {},
    }


def _write_decode_config(root, *, dp=2):
    source = _decode_source()
    source["model"]["topology"]["dp"] = dp
    source["execution"]["devices"] = list(range(dp))
    source["output"]["result_root"] = str(root / "results")
    path = root / f"decode-dp{dp}.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    return path


TOPOLOGY = {"tp": 1, "dp": 2, "pp": 1, "ep": 1}


def _marker(plan, run_id, state):
    return {
        "state": state,
        "run_id": run_id,
        "world_rank": 0,
        "config_fingerprint": plan.config_fingerprint,
        "workload_fingerprint": plan.workload_fingerprint,
    }


def _persist_run(plan, source, root, state, run_id):
    class Transport:
        def configure_scheduler(self, _allocations):
            pass

        def run_round(self, *, input_lengths, output_length, **_kwargs):
            return tuple(
                {
                    "request_index": index,
                    "success": state == "completed",
                    "input_length": length,
                    "output_length": output_length,
                    "error": "request failed",
                    "latency_ms": {
                        "wait": 1.0,
                        "prefill": 2.0,
                        "decode": 15.0,
                        "tpot": 1.0,
                        "total": 17.0,
                    },
                }
                for index, length in enumerate(input_lengths)
            )

    plan = replace(plan, result_directory=str(root))
    with patch.dict(os.environ, {"PERF_TEST_RUN_ID": run_id}):
        with patch(
            "rtp_llm.test.perf_test.multi_node.decode_runtime.DecodeHttpTransport",
            return_value=Transport(),
        ):
            execute_master_decode(
                plan,
                source_config=source,
                endpoint="http://test",
                prompt_for_length=str,
                server_log_path=None,
                profiler_directory=root / "traces",
            )


class DecodeMultiNodeTest(unittest.TestCase):
    def test_collection_rejects_missing_or_inconsistent_plan_and_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, source = build_external_decode_plan(
                _write_decode_config(root), "preflight", TOPOLOGY
            )
            run = root / "collected-run"
            _persist_run(plan, source, run, "completed", "attempt")
            verify = lambda: multi_benchmark.verify_collected_decode_run(
                run, plan=plan, run_id="attempt", state="completed"
            )
            verify()
            changes = (
                ("resolved_config.json", None, None),
                ("resolved_config.json", (), []),
                ("resolved_config.json", ("engine", "concurrency_limit"), 9),
                ("resolved_config.json", ("model", "topology", "ep"), True),
                ("execution_plan.json", None, None),
                ("execution_plan.json", (), []),
                ("execution_plan.json", ("action",), "profile"),
                ("execution_plan.json", ("config_fingerprint",), "stale"),
                ("execution_plan.json", ("cases", 0, "identity", "output_length"), 999),
                ("execution_plan.json", ("resource_plan", "devices"), [2, 3]),
                (
                    "execution_plan.json",
                    ("resolved_config", "engine", "max_seq_len"),
                    999,
                ),
                ("execution_plan.json", ("result_directory",), ""),
            )
            for artifact, keys, value in changes:
                path = run / artifact
                original = path.read_text()
                with self.subTest(artifact=artifact, keys=keys):
                    if keys is None:
                        path.unlink()
                    else:
                        changed = json.loads(original)
                        if not keys:
                            changed = value
                        else:
                            parent = changed
                            for key in keys[:-1]:
                                parent = parent[key]
                            parent[keys[-1]] = value
                        path.write_text(json.dumps(changed))
                    try:
                        with self.assertRaisesRegex(RuntimeError, "persisted"):
                            verify()
                    finally:
                        path.write_text(original)

    def test_collection_selects_current_attempt_instead_of_newest_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "work with spaces/RTP-LLM/github-opensource"
            current = repository / "TEST_OUTPUT_current_0"
            stale = repository / "TEST_OUTPUT_stale_0"
            current.mkdir(parents=True)
            stale.mkdir()
            os.utime(current, (1, 1))
            os.utime(stale, (2, 2))
            env = {
                "PATH": os.environ["PATH"],
                "FT_WORK_DIR": str(root),
                "FT_SUB_DIR": "work with spaces",
                "SUB_CMD": "copy",
                "PERF_TEST_RUN_ID": "current",
                "WORLD_RANK": "0",
            }
            script = Path(__file__).parent / "multi_node/multi_local_executor.sh"
            completed = subprocess.run(
                ["bash", str(script)], env=env, capture_output=True, text=True
            )
            self.assertEqual(0, completed.returncode, completed.stderr)
            self.assertEqual(str(current), completed.stdout.strip())
            completed = subprocess.run(
                ["bash", str(script)],
                env={**env, "PERF_TEST_RUN_ID": "missing"},
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(0, completed.returncode)

    def test_partial_benchmark_propagates_nonzero_launcher_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {
                "test_config": {"build_from_scratch": 0},
                "build_config": {"ft_sub_dir": "perf"},
                "benchmarks": [{"benchmark_name": "partial", "fixed_config": {}}],
            }
            with patch.multiple(
                multi_benchmark,
                get_args=lambda: SimpleNamespace(config="unused"),
                load_config=lambda _path: config,
                check_machine_and_build_config=lambda *_args: None,
                ensure_test_output_dir=lambda: None,
                ensure_experiment_output_dir=lambda _name: None,
                setup_logging=lambda: "host.log",
                multi_kill_script=lambda *_args, **_kwargs: None,
                multi_clean_script=lambda *_args, **_kwargs: None,
                experiment_output_dir=directory,
                logger=logging.getLogger("decode-launcher-test"),
            ):
                with patch.object(multi_benchmark.time, "sleep"):
                    with patch.object(
                        multi_benchmark,
                        "multi_test_script",
                        side_effect=PartialRunError("target_tpot_not_met"),
                    ):
                        self.assertEqual(5, multi_benchmark.main())

    def test_matching_marker_without_persisted_artifacts_fails_before_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _write_decode_config(root)
            plan, _source = build_external_decode_plan(path, "unused", TOPOLOGY)
            task = {
                "perf_test_config": str(path),
                "tp_size": 1,
                "dp_size": 2,
                "ep_size": 1,
                "ip_lists": "node0",
                "run_user": "test",
                "ssh_port": 2222,
                "ft_sub_dir": "perf",
                "sub_cmd": "test",
            }
            actions = []

            def run(args, *, env, stdout, **_kwargs):
                actions.append(env["SUB_CMD"])
                if env["SUB_CMD"] == "test":
                    stdout.write(
                        "PERF_TEST_DECODE_RESULT "
                        + json.dumps(
                            _marker(plan, env["PERF_TEST_RUN_ID"], "completed")
                        )
                        + "\n"
                    )
                return subprocess.CompletedProcess(
                    args, 7 if env["SUB_CMD"] == "test" else 0
                )

            with patch.object(
                multi_benchmark, "logger", logging.getLogger("decode-launcher-test")
            ):
                with patch.object(multi_benchmark.subprocess, "run", side_effect=run):
                    with self.assertRaisesRegex(RuntimeError, "persisted"):
                        multi_benchmark.execute_task(task, str(root), {}, False, 0)
            self.assertEqual(["test", "kill", "copy"], actions)

    def test_completion_requires_current_master_and_persisted_matching_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, source = build_external_decode_plan(
                _write_decode_config(root), str(root / "run"), TOPOLOGY
            )
            log = root / "log"
            run_id = "current-attempt"
            marker = _marker(plan, run_id, "completed")
            for field, value in (
                ("run_id", "stale"),
                ("world_rank", 1),
                ("config_fingerprint", "other"),
                ("workload_fingerprint", "other"),
            ):
                log.write_text(
                    "PERF_TEST_DECODE_RESULT "
                    + json.dumps({**marker, field: value})
                    + "\n"
                )
                self.assertIsNone(
                    multi_benchmark.extract_decode_state(
                        str(log), plan=plan, run_id=run_id
                    )
                )
            log.write_text("PERF_TEST_DECODE_RESULT " + json.dumps(marker) + "\n")
            self.assertEqual(
                "completed",
                multi_benchmark.extract_decode_state(
                    str(log), plan=plan, run_id=run_id
                ),
            )
            with self.assertRaisesRegex(RuntimeError, "persisted"):
                multi_benchmark.verify_collected_decode_run(
                    root / "run", plan=plan, run_id=run_id, state="completed"
                )
            _persist_run(plan, source, root / "run", "completed", run_id)
            multi_benchmark.verify_collected_decode_run(
                root / "run", plan=plan, run_id=run_id, state="completed"
            )
            for artifact, field, value in (
                ("manifest.json", "run_id", "stale"),
                ("results/result.json", "run_id", "stale"),
                ("results/result.json", "config_fingerprint", "other"),
                ("results/result.json", "status", "partial"),
                ("manifest.json", None, []),
            ):
                path = root / "run" / artifact
                original = path.read_text()
                changed = json.loads(original)
                if field is None:
                    changed = value
                else:
                    changed[field] = value
                path.write_text(json.dumps(changed))
                with self.subTest(artifact=artifact, field=field):
                    with self.assertRaisesRegex(RuntimeError, "persisted"):
                        multi_benchmark.verify_collected_decode_run(
                            root / "run", plan=plan, run_id=run_id, state="completed"
                        )
                path.write_text(original)

    def test_runtime_topology_mismatch_precedes_model_setup_or_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _write_decode_config(root)
            parallelism = SimpleNamespace(
                tp_size=1, dp_size=1, pp_size=1, ep_size=1, world_rank=0
            )
            configs = SimpleNamespace(
                parallelism_config=parallelism,
                prefill_cp_config=None,
                server_config=SimpleNamespace(start_port=12333),
                model_args=SimpleNamespace(model_type="registered"),
            )
            effects = []

            def configure(_configs):
                effects.append("model-download")
                raise RuntimeError("model setup reached")

            script = Path(__file__).parent / "multi_node/local_server_runner.py"
            entry = ast.parse(script.read_text()).body[-1]
            compiled = compile(
                ast.Module(body=entry.body, type_ignores=[]), str(script), "exec"
            )
            namespace = dict(
                os=os,
                pathlib=__import__("pathlib"),
                time=__import__("time"),
                PyEnvConfigs=object,
                setup_args=lambda: configs,
                setup_and_configure_server=configure,
                set_parallelism_config=lambda *_args, **_kwargs: effects.append(
                    "topology"
                ),
                build_external_decode_plan=build_external_decode_plan,
            )
            with patch.dict(os.environ, {"PERF_TEST_CONFIG_PATH": str(path)}):
                with self.assertRaisesRegex(PlanningError, "runtime topology"):
                    exec(compiled, namespace)
            self.assertEqual(["topology"], effects)
            self.assertFalse(list(root.glob("TEST_OUTPUT*")))

    def test_environment_newline_cannot_spoof_a_persisted_completed_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = {
                "perf_test_config": str(_write_decode_config(root)),
                "tp_size": 1,
                "dp_size": 2,
                "ep_size": 1,
                "ip_lists": "node0",
                "run_user": "test",
                "ssh_port": 2222,
                "ft_sub_dir": "perf",
                "sub_cmd": "test",
                "injected": '\nPERF_TEST_DECODE_RESULT {"state":"completed"}\n',
            }

            def run(args, *, env, stdout, **_kwargs):
                if env["SUB_CMD"] == "test":
                    stdout.write(env["INJECTED"])
                return subprocess.CompletedProcess(
                    args, 7 if env["SUB_CMD"] == "test" else 0
                )

            with patch.object(
                multi_benchmark, "logger", logging.getLogger("decode-launcher-test")
            ):
                with patch.object(multi_benchmark.subprocess, "run", side_effect=run):
                    with self.assertRaisesRegex(RuntimeError, "did not finalize"):
                        multi_benchmark.execute_task(task, str(root), {}, True, 0)
            self.assertFalse((root / "Task_/run").exists())

    def test_launcher_does_not_accept_nonzero_without_persisted_run_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = {
                "perf_test_config": str(_write_decode_config(root)),
                "tp_size": 1,
                "dp_size": 2,
                "ep_size": 1,
                "ip_lists": "node0",
                "run_user": "test",
                "ssh_port": 2222,
                "ft_sub_dir": "perf",
                "sub_cmd": "test",
            }
            actions = []

            def run(args, *, env, check, **_kwargs):
                actions.append(env["SUB_CMD"])
                code = 7 if env["SUB_CMD"] == "test" else 0
                if check and code:
                    raise subprocess.CalledProcessError(code, args)
                return subprocess.CompletedProcess(args, code)

            with patch.object(
                multi_benchmark, "logger", logging.getLogger("decode-launcher-test")
            ):
                with patch.object(multi_benchmark.subprocess, "run", side_effect=run):
                    with self.assertRaisesRegex(RuntimeError, "did not finalize"):
                        multi_benchmark.execute_task(task, str(root), {}, True, 0)
            self.assertEqual(["test", "kill", "clean"], actions)
            self.assertFalse((root / "Task_/run").exists())

    def test_launcher_snapshots_configs_and_collects_finalized_runs_despite_shutdown_signal(
        self,
    ):
        for state, kind in itertools.product(
            ("completed", "partial"), ("grid", "distribution")
        ):
            with self.subTest(
                state=state, kind=kind
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = _decode_source()
                if kind == "distribution":
                    source["workload"] = {
                        "kind": "distribution",
                        "distribution_file": "requests.json",
                        "output_length": 16,
                        "batch": {"strategy": "fixed", "sizes": [1, 4]},
                        "measure_runs": 1,
                        "warmup_runs": 0,
                    }
                (root / "requests.json").write_text(
                    json.dumps({"sequence_lengths": [32, 64, 96, 128]})
                )
                path = root / "config.json"
                path.write_text(json.dumps(source))
                expected, _ = build_external_decode_plan(path, "unused", TOPOLOGY)
                task = {
                    "perf_test_config": str(path),
                    "tp_size": 1,
                    "dp_size": 2,
                    "ep_size": 1,
                    "ip_lists": "node0",
                    "run_user": "test",
                    "ssh_port": 2222,
                    "ft_sub_dir": "perf",
                    "sub_cmd": "test",
                }
                actions = []
                delivered = []

                def run(args, *, env, stdout, check, **_kwargs):
                    action = env["SUB_CMD"]
                    actions.append(action)
                    code = 0
                    if action == "test":
                        # Rank delivery must read the frozen launch snapshot,
                        # even if the original user config changes meanwhile.
                        path.write_text("{}")
                        delivered_plan, delivered_source = build_external_decode_plan(
                            env["PERF_TEST_CONFIG_PATH"], "unused", TOPOLOGY
                        )
                        delivered.append(
                            (delivered_plan, delivered_source, env["PERF_TEST_RUN_ID"])
                        )
                        stdout.write(
                            "PERF_TEST_DECODE_RESULT "
                            + json.dumps(
                                _marker(delivered_plan, env["PERF_TEST_RUN_ID"], state)
                            )
                            + "\n"
                        )
                        code = 137
                    if action == "copy":
                        _persist_run(
                            delivered[0][0],
                            delivered[0][1],
                            Path(env["TASK_OUTPUT_DIR"]) / "run",
                            state,
                            delivered[0][2],
                        )
                    if check and code:
                        raise subprocess.CalledProcessError(code, args)
                    return subprocess.CompletedProcess(args, code)

                with patch.object(
                    multi_benchmark, "logger", logging.getLogger("decode-launcher-test")
                ):
                    with patch.object(
                        multi_benchmark.subprocess, "run", side_effect=run
                    ):
                        if state == "partial":
                            with self.assertRaises(PartialRunError):
                                multi_benchmark.execute_task(
                                    task, str(root), {}, True, 0
                                )
                        else:
                            multi_benchmark.execute_task(task, str(root), {}, True, 0)
                self.assertEqual(["test", "kill", "copy", "clean"], actions)
                self.assertEqual(
                    expected.workload_fingerprint, delivered[0][0].workload_fingerprint
                )
                self.assertEqual(
                    [case.case_id for case in expected.cases],
                    [case.case_id for case in delivered[0][0].cases],
                )
                self.assertEqual(
                    state,
                    json.loads((root / "Task_/run/results/result.json").read_text())[
                        "status"
                    ],
                )

    def test_standalone_launcher_help_works_outside_repository(self):
        script = Path(__file__).parent / "multi_node/multi_benchmark.py"
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [sys.executable, str(script.resolve()), "--help"],
                cwd=directory,
                env={"PATH": os.environ["PATH"]},
                capture_output=True,
                text=True,
            )
            self.assertEqual(0, completed.returncode, completed.stderr)
            self.assertIn("--config", completed.stdout)

    def test_invalid_runtime_topology_never_coerces_to_configured_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _write_decode_config(root)
            for value in (True, 1.5, None, "1"):
                with self.subTest(value=value):
                    actual = {**TOPOLOGY, "tp": value}
                    with self.assertRaisesRegex(PlanningError, "runtime topology"):
                        build_external_decode_plan(path, str(root / "run"), actual)

    def test_runner_delivers_same_config_path_to_every_rank_and_propagates_failure(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config source.json"
            config.write_text("{}")
            bin_dir = root / "bin"
            bin_dir.mkdir()
            calls = root / "calls.jsonl"
            for tool in ("ssh", "scp"):
                path = bin_dir / tool
                path.write_text(
                    f"#!{sys.executable}\nimport json, os, sys\n"
                    "with open(os.environ['FAKE_CALLS'], 'a') as stream:\n"
                    " stream.write(json.dumps([os.path.basename(sys.argv[0]), sys.argv[1:]]) + '\\n')\n"
                    "if os.path.basename(sys.argv[0]) == 'scp' and os.environ.get('FAKE_FAIL') == '1': sys.exit(7)\n"
                )
                path.chmod(0o755)
            env = dict(
                PATH=f"{bin_dir}:{os.environ['PATH']}",
                SUB_CMD="test",
                IP_LISTS="node0,node1",
                RUN_USER="test",
                SSH_PORT="2222",
                OPEN_SOURCE_REF="main",
                TP_SIZE="1",
                DP_SIZE="2",
                EP_SIZE="2",
                START_PORT="12333",
                MODEL_TYPE="registered",
                TOKENIZER_PATH="/m",
                CHECKPOINT_PATH="/m",
                PERF_TEST_CONFIG_PATH=str(config),
                PERF_TEST_RUN_ID="delivery_test",
                FAKE_CALLS=str(calls),
            )
            script = Path(__file__).parent / "multi_node/multi_runner.sh"
            completed = subprocess.run(
                ["bash", str(script)], env=env, capture_output=True, text=True
            )
            self.assertEqual(0, completed.returncode, completed.stderr)
            recorded = [json.loads(line) for line in calls.read_text().splitlines()]
            copies = [
                args for tool, args in recorded if tool == "scp" and str(config) in args
            ]
            self.assertEqual(2, len(copies))
            for args in copies:
                self.assertTrue(
                    args[-1].endswith(":/tmp/rtp_llm_perf_delivery_test/config.json")
                )
            environments = [
                shlex.split(args[-1])
                for tool, args in recorded
                if tool == "ssh" and "bash /tmp/multi_local_executor.sh" in args[-1]
            ]
            self.assertEqual(2, len(environments))
            for remote_command in environments:
                self.assertIn(
                    "PERF_TEST_CONFIG_PATH=/tmp/rtp_llm_perf_delivery_test/config.json",
                    remote_command,
                )
                self.assertIn(
                    "GANG_CONFIG_STRING=name:test_part0,ip:node0,port:12333;name:test_part1,ip:node1,port:12333",
                    remote_command,
                )
            completed = subprocess.run(
                ["bash", str(script)],
                env={**env, "FAKE_FAIL": "1"},
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(0, completed.returncode)

    def test_sample_benchmark_schema_loads_with_matching_user_configs(self):
        sample_path = Path(__file__).parent / "multi_node/multi_benchmark_config.yaml"
        sample = yaml.safe_load(sample_path.read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for benchmark in sample["benchmarks"]:
                fixed = benchmark.get("fixed_config", {})
                # Supply the user-managed normalized JSON artifacts named by
                # the example, then exercise its real launcher preflight.
                self.assertIn("perf_test_config", fixed)
                path = root / fixed["perf_test_config"]
                path.parent.mkdir(parents=True, exist_ok=True)
                source = _decode_source()
                topology = {name: fixed.get(f"{name}_size", 1) for name in TOPOLOGY}
                source["model"]["topology"] = topology
                source["execution"]["devices"] = list(
                    range(topology["tp"] * topology["dp"] * topology["pp"])
                )
                path.write_text(json.dumps(source))
            launcher_path = root / "benchmarks.yaml"
            launcher_path.write_text(yaml.safe_dump(sample))
            loaded = load_config(str(launcher_path))
            self.assertEqual(3, len(loaded["benchmarks"]))

    def test_launcher_exports_required_config_path(self):
        env = multi_benchmark.convert_to_env(
            {"perf_test_config": "/tmp/config source.json"}
        )
        self.assertEqual("/tmp/config source.json", env.get("PERF_TEST_CONFIG_PATH"))

    def test_launcher_reads_unified_completion_instead_of_legacy_table(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, _ = build_external_decode_plan(
                _write_decode_config(root), "unused", TOPOLOGY
            )
            path = root / "test.log"
            for state in ("completed", "partial"):
                path.write_text(
                    "worker waiting\nPERF_TEST_DECODE_RESULT "
                    + json.dumps(_marker(plan, "attempt", state))
                    + "\n"
                )
                self.assertEqual(
                    state,
                    multi_benchmark.extract_decode_state(
                        str(path), plan=plan, run_id="attempt"
                    ),
                )
            path.write_text("PERF_TEST_DECODE_RESULT malformed\n")
            self.assertIsNone(
                multi_benchmark.extract_decode_state(
                    str(path), plan=plan, run_id="attempt"
                )
            )

    def test_test_launcher_rejects_missing_config_before_remote_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            calls = root / "calls"
            for tool in ("ssh", "scp"):
                path = bin_dir / tool
                path.write_text(
                    f"#!{sys.executable}\nfrom pathlib import Path\nPath({str(calls)!r}).write_text('called')\n"
                )
                path.chmod(0o755)
            env = dict(
                PATH=f"{bin_dir}:{os.environ['PATH']}",
                SUB_CMD="test",
                IP_LISTS="node0",
                RUN_USER="test",
                SSH_PORT="2222",
                OPEN_SOURCE_REF="main",
                TP_SIZE="1",
                DP_SIZE="2",
                EP_SIZE="2",
                START_PORT="12333",
                MODEL_TYPE="registered",
                TOKENIZER_PATH="/m",
                CHECKPOINT_PATH="/m",
            )
            script = Path(__file__).parent / "multi_node/multi_runner.sh"
            completed = subprocess.run(
                ["bash", str(script)], env=env, capture_output=True, text=True
            )
            self.assertNotEqual(0, completed.returncode)
            self.assertIn("PERF_TEST_CONFIG_PATH", completed.stdout + completed.stderr)
            self.assertFalse(calls.exists())

    def test_launcher_requires_decode_config_and_prevalidates_all_topologies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _write_decode_config(root)
            benchmark = {"fixed_config": {"tp_size": 1, "dp_size": 2, "ep_size": 1}}
            launcher_path = root / "benchmarks.yaml"
            launcher_path.write_text(json.dumps({"benchmarks": [benchmark]}))
            with self.assertRaisesRegex(ConfigurationError, "perf_test_config"):
                load_config(str(launcher_path))
            benchmark["fixed_config"]["perf_test_config"] = path.name
            launcher_path.write_text(json.dumps({"benchmarks": [benchmark]}))
            loaded = load_config(str(launcher_path))
            self.assertEqual(
                str(path.resolve()),
                loaded["benchmarks"][0]["fixed_config"]["perf_test_config"],
            )
            benchmark["iterative_search_config"] = {"tp_size": [1, 2]}
            launcher_path.write_text(json.dumps({"benchmarks": [benchmark]}))
            with self.assertRaisesRegex(PlanningError, "runtime topology"):
                load_config(str(launcher_path))
            benchmark.pop("iterative_search_config")
            benchmark["recursive_search_config"] = {"pp_size": [1, 2]}
            launcher_path.write_text(json.dumps({"benchmarks": [benchmark]}))
            with self.assertRaisesRegex(PlanningError, "runtime topology"):
                load_config(str(launcher_path))
            self.assertFalse((root / "results").exists())

    def test_launcher_rejects_old_flat_benchmark_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.yaml"
            path.write_text(
                json.dumps({"benchmarks": [{"name": "decode", "dp_size": 2}]})
            )
            with self.assertRaisesRegex(ConfigurationError, "fixed_config"):
                load_config(str(path))

    def test_runner_collects_recursive_run_artifacts_and_separate_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remote = root / "remote path"
            output = root / "output path"
            for relative in (
                "run/results/cases/a.json",
                "run/reports/decode_report.json",
                "run/profiles/a/profile_manifest.json",
                "run/manifest.json",
                "run/execution_plan.json",
                "run/resolved_config.json",
                "run/source_config.json",
                "main_logs/process.log",
                "normal_trace.json",
                "normal_trace_dir/trace.json",
            ):
                target = remote / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(relative)
            (output / "process_logs").mkdir(parents=True)
            (output / "trace_files").mkdir()
            bin_dir = root / "bin"
            bin_dir.mkdir()
            (bin_dir / "ssh").write_text(
                f"#!{sys.executable}\nimport os, pathlib, shlex, sys\n"
                "command = shlex.split(sys.argv[-1])\n"
                "if 'bash' in command:\n"
                " if command[0] != 'env' or ('TASK_OUTPUT_DIR=' + os.environ['TASK_OUTPUT_DIR']) not in command: sys.exit(9)\n"
                " if ('FT_SUB_DIR=' + os.environ['FT_SUB_DIR']) not in command: sys.exit(9)\n"
                " print(os.environ['FAKE_REMOTE_ROOT'])\n"
                "elif command[0] == 'find':\n"
                " if os.environ.get('FAKE_FAIL_LIST') == '1': sys.exit(8)\n"
                " if list(pathlib.Path(os.environ['FAKE_REMOTE_ROOT']).glob('normal_*')): print('x')\n"
                "else: sys.exit(10)\n"
            )
            (bin_dir / "scp").write_text(
                f"#!{sys.executable}\n"
                "import glob, os, pathlib, shutil, sys\n"
                "args = sys.argv[1:]\n"
                "if '-P' in args:\n i = args.index('-P'); del args[i:i+2]\n"
                "recursive = '-r' in args\n"
                "args = [arg for arg in args if arg != '-r']\n"
                "destination = pathlib.Path(args[-1])\n"
                "for source in args[:-1]:\n"
                " if os.environ.get('FAKE_FAIL_HOST', '') and source.startswith('test@' + os.environ['FAKE_FAIL_HOST'] + ':') and os.environ['FAKE_FAIL_ARTIFACT'] in source: sys.exit(7)\n"
                " source = source.split(':', 1)[-1]\n"
                " matches = glob.glob(source)\n"
                " if not matches: sys.exit(1)\n"
                " for name in matches:\n"
                "  path = pathlib.Path(name)\n"
                "  if path.is_dir():\n"
                "   if not recursive: sys.exit(1)\n"
                "   shutil.copytree(path, destination / path.name, dirs_exist_ok=True)\n"
                "  else: shutil.copy2(path, destination)\n"
            )
            for tool in bin_dir.iterdir():
                tool.chmod(0o755)
            env = dict(
                PATH=f"{bin_dir}:{os.environ['PATH']}",
                SUB_CMD="copy",
                IP_LISTS="node0,node1",
                RUN_USER="test",
                SSH_PORT="2222",
                TASK_OUTPUT_DIR=str(output),
                FAKE_REMOTE_ROOT=str(remote),
                FT_SUB_DIR="work with spaces",
            )
            script = Path(__file__).parent / "multi_node/multi_runner.sh"
            completed = subprocess.run(
                ["bash", str(script)], env=env, capture_output=True, text=True
            )
            self.assertEqual(0, completed.returncode, completed.stderr)
            for relative in (
                "run/results/cases/a.json",
                "run/reports/decode_report.json",
                "run/profiles/a/profile_manifest.json",
                "run/manifest.json",
                "run/execution_plan.json",
                "run/resolved_config.json",
                "run/source_config.json",
                "trace_files/normal_trace.json",
                "trace_files/normal_trace_dir/trace.json",
            ):
                self.assertTrue((output / relative).is_file(), relative)
            self.assertTrue(list((output / "process_logs").glob("process_*.log")))
            for host, artifact in itertools.product(
                ("node0", "node1"), ("process.log", "normal_*")
            ):
                with self.subTest(host=host, artifact=artifact):
                    failed = subprocess.run(
                        ["bash", str(script)],
                        env={
                            **env,
                            "FAKE_FAIL_HOST": host,
                            "FAKE_FAIL_ARTIFACT": artifact,
                        },
                        capture_output=True,
                        text=True,
                    )
                    self.assertNotEqual(0, failed.returncode)
            failed = subprocess.run(
                ["bash", str(script)],
                env={**env, "FAKE_FAIL_LIST": "1"},
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(0, failed.returncode)
            (remote / "normal_trace.json").unlink()
            (remote / "normal_trace_dir/trace.json").unlink()
            (remote / "normal_trace_dir").rmdir()
            optional_absent = subprocess.run(
                ["bash", str(script)], env=env, capture_output=True, text=True
            )
            self.assertEqual(0, optional_absent.returncode, optional_absent.stderr)
            (remote / "run/manifest.json").unlink()
            completed = subprocess.run(
                ["bash", str(script)], env=env, capture_output=True, text=True
            )
            self.assertNotEqual(0, completed.returncode)

    def test_master_executes_grid_and_writes_standard_artifacts_and_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _write_decode_config(root)
            source = json.loads(path.read_text())
            source["workload"]["profiling"] = {"enabled": True}
            path.write_text(json.dumps(source))
            plan, source = build_external_decode_plan(path, str(root / "run"), TOPOLOGY)
            log = root / "server.log"
            requests = []

            class Response:
                status_code = 200

                def __init__(self, body):
                    self.body = body

                def json(self):
                    return self.body

            def post(url, *, json, **_kwargs):
                requests.append((url, copy.deepcopy(json)))
                if url.endswith("/start_profile"):
                    trace = root / (json["trace_name"] + ".json")
                    trace.write_text("{}")
                    with log.open("a") as stream:
                        stream.write(f"profiler trace saved: {trace}\n")
                    return Response({"status": "ok"})
                if url.endswith("/update_scheduler_info"):
                    return Response({"status": "ok"})
                return Response(
                    {
                        "aux_info": {
                            "input_len": int(json["prompt"].split("-")[1]),
                            "output_len": json["generate_config"]["max_new_tokens"],
                            "wait_time": 1.0,
                            "first_token_cost_time": 3.0,
                            "cost_time": 18.0,
                        }
                    }
                )

            def transport(**kwargs):
                return DecodeHttpTransport(**kwargs, post=post)

            with patch(
                "rtp_llm.test.perf_test.multi_node.decode_runtime.DecodeHttpTransport",
                side_effect=transport,
            ):
                result = execute_master_decode(
                    plan,
                    source_config=source,
                    endpoint="http://external:12333",
                    prompt_for_length=lambda length: f"prompt-{length}",
                    server_log_path=str(log),
                    profiler_directory=root / "traces",
                )
            run = root / "run"
            self.assertEqual(RunState.COMPLETED, result.state)
            self.assertEqual(
                "completed", json.loads((run / "manifest.json").read_text())["state"]
            )
            envelope = json.loads((run / "results/result.json").read_text())
            self.assertEqual(4, envelope["summary"]["successful_cases"])
            self.assertEqual(4, len(list((run / "results/cases").glob("*.json"))))
            self.assertEqual(
                4, len(list((run / "profiles").glob("*/profile_manifest.json")))
            )
            self.assertEqual(
                "formal",
                json.loads((run / "reports/decode_report.json").read_text())[
                    "report_kind"
                ],
            )
            self.assertTrue((run / "source_config.json").is_file())
            self.assertTrue((run / "resolved_config.json").is_file())
            self.assertTrue((run / "execution_plan.json").is_file())
            self.assertTrue((run / "results/summary.json").is_file())
            self.assertTrue(
                all(url.startswith("http://external:12333") for url, _ in requests)
            )
            measured_outputs = {
                payload["generate_config"]["max_new_tokens"]
                for url, payload in requests
                if url == "http://external:12333"
            }
            self.assertEqual({16, 64}, measured_outputs)
            self.assertTrue(
                all(
                    case["metrics"]["topology"] == TOPOLOGY
                    for case in envelope["cases"]
                )
            )

    def test_finalization_error_marks_manifest_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, source = build_external_decode_plan(
                _write_decode_config(root), str(root / "run"), TOPOLOGY
            )
            with patch(
                "rtp_llm.test.perf_test.multi_node.decode_runtime.DecodeHttpTransport"
            ) as transport:
                transport.return_value.configure_scheduler.side_effect = RuntimeError(
                    "service failed"
                )
                with patch(
                    "rtp_llm.test.perf_test.multi_node.decode_runtime.finalize_decode_run",
                    side_effect=OSError("disk failed"),
                ):
                    with self.assertRaisesRegex(OSError, "disk failed"):
                        execute_master_decode(
                            plan,
                            source_config=source,
                            endpoint="http://external",
                            prompt_for_length=str,
                            server_log_path=None,
                            profiler_directory=root / "traces",
                        )
            self.assertEqual(
                "failed", json.loads((root / "run/manifest.json").read_text())["state"]
            )

    def test_external_plan_matches_local_case_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = _write_decode_config(root)
            plan, source = build_external_decode_plan(
                config_path, str(root / "run"), TOPOLOGY
            )
            local_plan, _ = build_request_plan(
                CliRequest("decode", "run", str(config_path), None, False, ())
            )
            self.assertEqual(
                [case.case_id for case in local_plan.cases],
                [case.case_id for case in plan.cases],
            )
            self.assertEqual(4, len(plan.cases))
            self.assertEqual(local_plan.config_fingerprint, plan.config_fingerprint)
            self.assertEqual(source["workload"]["output_lengths"], [16, 64])
            self.assertFalse((root / "run").exists())

    def test_external_plan_rejects_every_runtime_topology_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _write_decode_config(root)
            for name in TOPOLOGY:
                actual = dict(TOPOLOGY)
                actual[name] += 1
                with self.subTest(dimension=name):
                    with self.assertRaisesRegex(PlanningError, "runtime topology"):
                        build_external_decode_plan(path, str(root / "run"), actual)
            self.assertFalse((root / "run").exists())

    def test_config_read_errors_are_configuration_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "invalid.json"
            for contents, message in (
                (None, "cannot read config"),
                (b"\xff", "not UTF-8"),
                (b"{", "not valid JSON"),
                (b"[]", "must contain an object"),
            ):
                if contents is not None:
                    path.write_bytes(contents)
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ConfigurationError, message):
                        build_external_decode_plan(path, str(root / "run"), TOPOLOGY)

    def test_worker_rank_only_waits_and_does_not_write_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            events = []

            def execute():
                root.mkdir()
                events.append("execute")

            result = run_external_rank(
                world_rank=1,
                wait_for_master=lambda: events.append("wait"),
                execute_master=execute,
            )
            self.assertIsNone(result)
            self.assertEqual(["wait"], events)
            self.assertFalse(root.exists())

    def test_rank_zero_executes_once_and_returns_result(self):
        events = []

        def execute():
            events.append("execute")
            return "complete"

        result = run_external_rank(
            world_rank=0,
            wait_for_master=lambda: events.append("wait"),
            execute_master=execute,
        )
        self.assertEqual("complete", result)
        self.assertEqual(["execute"], events)


if __name__ == "__main__":
    unittest.main()

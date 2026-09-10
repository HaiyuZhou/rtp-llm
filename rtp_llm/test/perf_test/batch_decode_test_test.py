import argparse
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from unittest.mock import Mock, patch

from rtp_llm.test.perf_test.batch_decode_test import (
    _capture_reproduction_env,
    _dedupe_cache_grid_cases,
    _effective_grid_max_seq_len,
    _engine_arg_argv,
    _load_cache_grid_cases,
    _load_materialized_case_store,
    _parse_name_value,
    _redact_argv,
    _resolve_cache_block_size,
    _write_test_info,
    parse_args,
)
from rtp_llm.test.perf_test.cache_grid_runner import (
    CacheGridRunner,
    MaterializedCaseStore,
    PrefixPromptFactory,
    _post_prefill,
    _post_prefill_ids,
    resume_config_fingerprint,
    validate_cache_grid_resume,
)


class _WhitespaceTokenizer:
    def encode(self, text):
        return text.split()


class _WordTokenizer:
    """Minimal stand-in: every whitespace-separated word is one token."""

    def encode(self, text):
        return text.split()


class _TailMergingTokenizer:
    """Whitespace tokenizer that fuses a ``__run_`` tail into the previous token."""

    def encode(self, text):
        words = text.split()
        out = []
        for word in words:
            if word.startswith("__run_") and out and out[-1] == "hello":
                out[-1] = "hello" + word
            else:
                out.append(word)
        return out


class BatchDecodeTest(unittest.TestCase):
    def test_effective_grid_max_seq_len_uses_decode_need(self):
        args = argparse.Namespace(max_seq_len=8192, decode_test_length=30)
        self.assertEqual(_effective_grid_max_seq_len(args, [1024, 65536]), 65566)

    def test_effective_grid_max_seq_len_respects_explicit_headroom(self):
        args = argparse.Namespace(max_seq_len=65664, decode_test_length=30)
        self.assertEqual(_effective_grid_max_seq_len(args, [65536]), 65664)

    def test_cache_grid_loader_validates_explicit_cases(self):
        payload = {
            "cases": [
                {"case_id": 7, "batch_size": 1, "input_len": 4096, "cache_len": 0},
                {
                    "case_id": 8,
                    "batch_size": 1,
                    "input_len": 4096,
                    "cache_len": 2048,
                },
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache_grid.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(_load_cache_grid_cases(str(path)), payload["cases"])

    def test_cache_grid_loader_rejects_duplicate_geometry(self):
        payload = {
            "cases": [
                {"batch_size": 1, "input_len": 4096, "cache_len": 2048},
                {"batch_size": 1, "input_len": 4096, "cache_len": 2048},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache_grid.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate cache grid case"):
                _load_cache_grid_cases(str(path))

    def test_parse_args_exposes_cache_runner_controls(self):
        args, remaining = parse_args()
        self.assertEqual(args.cache_measure_runs, 3)
        self.assertGreater(args.cache_request_timeout, 0)
        self.assertEqual(args.cache_commit_tail_tokens, 4096)
        self.assertEqual(args.cache_grid_json, "")
        self.assertEqual(args.expected_cache_block_size, 0)
        self.assertEqual(args.materialize_cache_cases, "")
        self.assertEqual(args.cache_case_files, "")
        self.assertEqual(args.cache_request_transport, "dashsc_input_ids")
        self.assertEqual(args.cache_grpc_port, 0)
        self.assertEqual(args.cache_checkpoint_every, 100)
        self.assertFalse(args.require_cache_resume)
        self.assertIsInstance(remaining, list)

    def test_generated_cache_grid_uses_independent_seq_and_cache_alignment(self):
        payload = {
            "seq_generation": {
                "kind": "linear_with_dense_prefix",
                "count": 20,
                "max_seq_len": 65535,
            },
            "seq_block_size": 256,
            "cache_block_size": 4096,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache_grid.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            cases = _load_cache_grid_cases(str(path))
        self.assertTrue(any(case["input_len"] == 256 for case in cases))
        self.assertTrue(
            all(
                case["cache_len"] == 0 or case["cache_len"] % 4096 == 0
                for case in cases
            )
        )
        self.assertTrue(
            all(
                case["cache_len"] == 0 or case["cache_len"] + 4096 <= case["input_len"]
                for case in cases
            )
        )

    def test_cache_seed_commits_one_tail_and_preserves_exact_prefix(self):
        tokenizer = _WhitespaceTokenizer()
        factory = PrefixPromptFactory(tokenizer)
        target, prefix, built_len = factory.make_case(7, 32, 16)
        seed = factory.make_seed(7, prefix, 16, 8)
        prefix_ids = tokenizer.encode(prefix)
        self.assertEqual(built_len, 32)
        self.assertEqual(len(prefix_ids), 16)
        self.assertEqual(len(tokenizer.encode(seed)), 24)
        self.assertEqual(tokenizer.encode(target)[:16], prefix_ids)
        self.assertEqual(tokenizer.encode(seed)[:16], prefix_ids)

    def test_case_prefixes_are_isolated(self):
        tokenizer = _WhitespaceTokenizer()
        factory = PrefixPromptFactory(tokenizer)
        _, prefix_a, _ = factory.make_case(1, 32, 16)
        _, prefix_b, _ = factory.make_case(2, 32, 16)
        self.assertNotEqual(tokenizer.encode(prefix_a), tokenizer.encode(prefix_b))

    @patch("rtp_llm.test.perf_test.cache_grid_runner.requests.post")
    def test_post_prefill_records_client_ttft_separately(self, post):
        response = Mock(status_code=200)
        response.json.return_value = {
            "aux_info": {
                "input_len": 1048575,
                "output_len": 1,
                "reuse_len": 0,
                "first_token_cost_time": 173.0,
                "cost_time": 175.0,
                "wait_time": 2.0,
            }
        }
        post.return_value = response
        result = _post_prefill(12345, "prompt", 10, "case:run0")
        self.assertTrue(result["success"])
        self.assertEqual(result["prefill_time_ms"], 173.0)
        self.assertGreaterEqual(result["ttft_ms"], 0.0)
        self.assertEqual(result["ttft_ms"], result["client_wall_time_ms"])
        self.assertEqual(result["ttft_source"], "client_http_wall_max_new_tokens_1")

    def test_post_prefill_ids_parses_grpc_metrics(self):
        from rtp_llm.dash_sc.proto import predict_v2_pb2

        response = predict_v2_pb2.ModelStreamInferResponse()
        infer = response.infer_response
        for name, value in (("prompt_token_num", 4), ("prompt_cached_token_num", 2)):
            output = infer.outputs.add(name=name, datatype="INT32")
            output.shape.append(1)
            infer.raw_output_contents.append(struct.pack("<i", value))
        output = infer.outputs.add(name="generated_ids", datatype="INT32")
        output.shape.extend([1, 1])
        infer.raw_output_contents.append(struct.pack("<i", 7))
        infer.parameters["engine_cost_time_us"].int64_param = 12500
        infer.parameters["engine_first_token_cost_time_us"].int64_param = 12000
        infer.parameters["engine_wait_time_us"].int64_param = 500

        class Stub:
            def ModelStreamInfer(self, requests, timeout):
                request = next(requests)
                self.assert_timeout = timeout
                self.request = request
                return iter((response,))

        stub = Stub()
        result = _post_prefill_ids(stub, [1, 2, 3, 4], 10, "case:run0")
        self.assertTrue(result["success"])
        self.assertEqual(result["input_len"], 4)
        self.assertEqual(result["reuse_len"], 2)
        self.assertEqual(result["output_len"], 1)
        self.assertEqual(result["prefill_time_ms"], 12.0)
        self.assertEqual(result["total_time_ms"], 12.5)
        self.assertEqual(result["wait_time_ms"], 0.5)
        self.assertEqual(stub.assert_timeout, 10)
        self.assertTrue(stub.request.parameters["force_sp_accept"].bool_param)
        self.assertEqual(
            result["ttft_source"],
            "client_dashsc_grpc_input_ids_wall_max_new_tokens_1",
        )

    @patch("rtp_llm.test.perf_test.cache_grid_runner._post_prefill")
    def test_runner_accepts_only_exact_shape_reuse_and_ttft(self, post):
        post.side_effect = [
            {"success": True},
            {
                "success": True,
                "input_len": 16,
                "output_len": 1,
                "reuse_len": 8,
                "ttft_ms": 12.0,
            },
            {
                "success": True,
                "input_len": 16,
                "output_len": 1,
                "reuse_len": 8,
                "ttft_ms": 10.0,
            },
            {
                "success": True,
                "input_len": 16,
                "output_len": 1,
                "reuse_len": 8,
                "ttft_ms": 11.0,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            rows = CacheGridRunner(
                12345,
                _WhitespaceTokenizer(),
                [{"case_id": 1, "batch_size": 1, "input_len": 16, "cache_len": 8}],
                tmp,
                cache_commit_tail_tokens=8,
            ).run()
        self.assertEqual(rows[0]["status"], "ok")
        self.assertTrue(rows[0]["shape_exact"])
        self.assertTrue(rows[0]["reuse_exact"])
        self.assertTrue(rows[0]["timing_valid"])
        self.assertEqual(rows[0]["median_ttft_ms"], 11.0)

    @patch("rtp_llm.test.perf_test.cache_grid_runner._post_prefill")
    def test_runner_fails_fast_and_checkpoints_invalid_reuse(self, post):
        post.side_effect = [
            {"success": True},
            *[
                {
                    "success": True,
                    "input_len": 16,
                    "output_len": 1,
                    "reuse_len": 0,
                    "ttft_ms": 10.0,
                }
                for _ in range(3)
            ],
        ]
        with tempfile.TemporaryDirectory() as tmp:
            runner = CacheGridRunner(
                12345,
                _WhitespaceTokenizer(),
                [{"case_id": 1, "batch_size": 1, "input_len": 16, "cache_len": 8}],
                tmp,
                cache_commit_tail_tokens=8,
            )
            with self.assertRaisesRegex(RuntimeError, "invalid_reuse"):
                runner.run()
            with (Path(tmp) / "cache_grid_results.json").open() as f:
                result = json.load(f)
        self.assertFalse(result["complete"])
        self.assertEqual(result["completed_cases"], 0)
        self.assertEqual(result["metrics"][0]["status"], "invalid_reuse")

    def test_engine_arg_shorthand_is_forwarded(self):
        self.assertEqual(
            _engine_arg_argv(["tp_size=8", "fp8_kv_cache=1"]),
            ["--tp_size", "8", "--fp8_kv_cache", "1"],
        )

    def test_name_value_rejects_missing_separator(self):
        with self.assertRaises(ValueError):
            _parse_name_value("tp_size", "--engine_arg")

    def test_parse_args_exposes_runtime_overrides(self):
        args, remaining = parse_args(
            [
                "--engine_arg=tp_size=8",
                "--engine_env=FP8_KV_CACHE=1",
                "--measure_runs=3",
                "--model_type=example_model",
            ]
        )
        self.assertEqual(args.engine_arg, ["tp_size=8"])
        self.assertEqual(args.engine_env, ["FP8_KV_CACHE=1"])
        self.assertEqual(args.measure_runs, 3)
        self.assertIn("--model_type=example_model", remaining)

    def test_dsv4_profile_injects_model_identity_and_paths(self):
        profile = Path(__file__).parent / "profiles" / "dsv4_pro_prefill.json"
        _, remaining = parse_args(["--profile", str(profile)])

        self.assertIn("--model_type", remaining)
        self.assertEqual(remaining[remaining.index("--model_type") + 1], "deepseek_v4")
        self.assertIn("--checkpoint_path", remaining)
        self.assertEqual(
            remaining[remaining.index("--checkpoint_path") + 1],
            "/data5/nanjun.cp/DeepSeek-V4-Pro",
        )
        self.assertIn("--tokenizer_path", remaining)
        self.assertEqual(
            remaining[remaining.index("--tokenizer_path") + 1],
            "/data5/nanjun.cp/DeepSeek-V4-Pro",
        )

    def test_redact_argv_hides_embedded_engine_secret(self):
        self.assertEqual(
            _redact_argv(
                [
                    "--engine_env=OSS_ACCESS_KEY_ID=secret",
                    "--engine_arg=tp_size=8",
                ]
            ),
            ["--engine_env=***", "--engine_arg=tp_size=8"],
        )

    def test_redact_argv_preserves_noncredential_token_parameters(self):
        argv = [
            "--tokenizer_path",
            "/weights/model",
            "--cache_commit_tail_tokens",
            "128",
            "--max_batch_tokens_size",
            "1048576",
        ]
        self.assertEqual(_redact_argv(argv), argv)

    def test_capture_reproduction_env_records_values_and_redacts_secrets(self):
        with patch.dict(
            "os.environ",
            {"DSV4_CHUNK_TOKENS": "8192", "PRIVATE_TOKEN": "secret-value"},
            clear=False,
        ):
            captured = _capture_reproduction_env(["PRIVATE_TOKEN"])
        self.assertEqual(captured["DSV4_CHUNK_TOKENS"], "8192")
        self.assertEqual(captured["PRIVATE_TOKEN"], "***")

    def test_test_info_records_resume_config_env_and_attempt_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _ = parse_args(
                [
                    "--result_dir",
                    tmp,
                    "--cache_grid_json",
                    "grid.json",
                    "--partial",
                    "2",
                ]
            )
            remaining = [
                "--model_type",
                "deepseek_v4",
                "--checkpoint_path",
                "/weights/model",
                "--tokenizer_path",
                "/weights/model",
            ]
            resume_config = {"model": {"checkpoint_path": "/weights/model"}}
            with patch.dict("os.environ", {"DSV4_CHUNK_TOKENS": "8192"}):
                _write_test_info(
                    args,
                    remaining,
                    status="running",
                    expected_cache_block_size=512,
                    resume_config=resume_config,
                )
                _write_test_info(
                    args,
                    remaining,
                    status="completed",
                    expected_cache_block_size=512,
                    resume_config=resume_config,
                )
            info = json.loads(
                (Path(tmp) / "test_info.json").read_text(encoding="utf-8")
            )
        self.assertEqual(info["schema_version"], 3)
        self.assertEqual(info["status"], "completed")
        self.assertEqual(info["attempt_count"], 1)
        self.assertEqual(info["engine_environment"]["DSV4_CHUNK_TOKENS"], "8192")
        self.assertEqual(info["resume_config"], resume_config)
        self.assertEqual(
            info["resume_config_sha256"], resume_config_fingerprint(resume_config)
        )

    def test_resolve_cache_block_size_prefers_cli_value(self):
        payload = {"generator": {"cache_alignment": 512}}
        self.assertEqual(_resolve_cache_block_size(payload, 256), 256)
        self.assertEqual(_resolve_cache_block_size(payload, 0), 512)

    def test_resolve_cache_block_size_handles_missing_metadata(self):
        self.assertEqual(_resolve_cache_block_size({}, 0), 0)
        self.assertEqual(_resolve_cache_block_size({"generator": {}}, 0), 0)
        self.assertEqual(_resolve_cache_block_size(None, 0), 0)
        self.assertEqual(
            _resolve_cache_block_size({"generator": {"cache_alignment": "junk"}}, 0), 0
        )

    def test_dedupe_collapses_cases_in_same_block_bucket(self):
        cases = [
            {"case_id": 0, "batch_size": 1, "input_len": 4096, "cache_len": 2048},
            {"case_id": 1, "batch_size": 1, "input_len": 4096, "cache_len": 2304},
            {"case_id": 2, "batch_size": 1, "input_len": 4096, "cache_len": 2400},
        ]
        deduped = _dedupe_cache_grid_cases(cases, 512)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0]["cache_len"], 2048)

    def test_dedupe_prefers_aligned_representative(self):
        cases = [
            {"case_id": 0, "batch_size": 1, "input_len": 4096, "cache_len": 2304},
            {"case_id": 1, "batch_size": 1, "input_len": 4096, "cache_len": 2048},
        ]
        deduped = _dedupe_cache_grid_cases(cases, 512)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0]["cache_len"], 2048)

    def test_dedupe_prefers_cold_over_partial_first_block(self):
        cases = [
            {"case_id": 0, "batch_size": 1, "input_len": 4096, "cache_len": 300},
            {"case_id": 1, "batch_size": 1, "input_len": 4096, "cache_len": 0},
        ]
        deduped = _dedupe_cache_grid_cases(cases, 512)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0]["cache_len"], 0)

    def test_dedupe_keeps_distinct_inputs_and_buckets(self):
        cases = [
            {"case_id": 0, "batch_size": 1, "input_len": 4096, "cache_len": 2048},
            {"case_id": 1, "batch_size": 1, "input_len": 8192, "cache_len": 2048},
            {"case_id": 2, "batch_size": 1, "input_len": 4096, "cache_len": 4096 - 512},
        ]
        deduped = _dedupe_cache_grid_cases(cases, 512)
        self.assertEqual(len(deduped), 3)


class CacheGridProfileTest(unittest.TestCase):
    cases = [
        {"case_id": 1, "batch_size": 1, "input_len": 16, "cache_len": 8},
        {"case_id": 2, "batch_size": 1, "input_len": 16, "cache_len": 0},
    ]

    def runner(self, tmp, **kwargs):
        return CacheGridRunner(
            12345,
            _WhitespaceTokenizer(),
            self.cases,
            tmp,
            cache_commit_tail_tokens=8,
            profile_trace_timeout=0.01,
            **kwargs,
        )

    def wire(self, runner, *, bad_reuse=False, write_traces=True):
        events = []
        active = []

        def arm(url, json, timeout):
            self.assertTrue(url.endswith("/start_profile"))
            self.assertEqual(json["num_steps"], 1)
            self.assertTrue(json["enable_all_rank"])
            events.append("arm")
            active.append(json["trace_name"])
            response = Mock()
            response.json.return_value = {"status": "ok"}
            return response

        def post(text, ids, request_id):
            is_profile = request_id.endswith(":profile")
            events.append(
                "profile"
                if is_profile
                else "seed" if request_id.endswith(":seed") else "measure"
            )
            if is_profile and write_traces:
                for rank in range(runner.profile_tp_size):
                    (runner.result_dir / f"{active[-1]}_wr{rank}_1.json").write_text(
                        json.dumps({"traceEvents": [{"name": "kernel", "dur": 1}]})
                    )
            cached = "seq16_cache8" in request_id
            return {
                "success": True,
                "request_id": request_id,
                "input_len": len(text.split()),
                "output_len": 1,
                "reuse_len": 0 if bad_reuse and is_profile else 8 if cached else 0,
                "ttft_ms": 1000.0 if is_profile else 10.0,
            }

        runner._http_session.post = Mock(side_effect=arm)
        runner._post_request = Mock(side_effect=post)
        return events

    def test_disabled_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = self.runner(tmp)
            events = self.wire(runner)
            rows = runner.run()
            self.assertNotIn("arm", events)
            self.assertEqual([r["median_ttft_ms"] for r in rows], [10.0, 10.0])
            self.assertFalse((Path(tmp) / "cache_profiles").exists())

    def test_selected_profiles_follow_measurements_and_preserve_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = self.runner(
                tmp, profile_runs=2, profile_case_ids=[1], profile_tp_size=2
            )
            events = self.wire(runner)
            rows = runner.run()
            self.assertEqual(events[-6:], ["seed", "arm", "profile"] * 2)
            self.assertEqual(events.count("measure"), 6)
            self.assertEqual([r["median_ttft_ms"] for r in rows], [10.0, 10.0])
            self.assertEqual([len(r["runs"]) for r in rows], [3, 3])
            manifest = json.loads(
                next((Path(tmp) / "cache_profiles").glob("*/manifest.json")).read_text()
            )
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual([r["case_id"] for r in manifest["records"]], [1, 1])
            self.assertNotEqual(
                manifest["records"][0]["prompt_case_id"],
                manifest["records"][1]["prompt_case_id"],
            )
            for record in manifest["records"]:
                self.assertEqual(len(record["trace_files"]), 2)
                for path in record["trace_files"]:
                    self.assertTrue((Path(tmp) / path).is_file())

    def test_profile_only_cold_replays_are_unique_and_do_not_write_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = self.runner(
                tmp, profile_runs=2, profile_case_ids=[2], profile_only=True
            )
            events = self.wire(runner)
            records = runner.run()
            self.assertEqual(events, ["arm", "profile"] * 2)
            texts = [c.args[0] for c in runner._post_request.call_args_list]
            self.assertNotEqual(texts[0], texts[1])
            self.assertEqual([r["result"]["reuse_len"] for r in records], [0, 0])
            self.assertFalse((Path(tmp) / "cache_grid_results.json").exists())

    def test_completed_baseline_can_be_profiled_without_remeasurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = self.runner(tmp)
            self.wire(first)
            first.run()
            baseline = (Path(tmp) / "cache_grid_results.json").read_bytes()
            runner = self.runner(
                tmp, profile_runs=1, profile_case_ids=[1], profile_only=True
            )
            events = self.wire(runner)
            runner.run()
            self.assertEqual(events, ["seed", "arm", "profile"])
            self.assertEqual(
                (Path(tmp) / "cache_grid_results.json").read_bytes(), baseline
            )

    def test_profile_failure_does_not_invalidate_completed_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = self.runner(tmp, profile_runs=1, profile_case_ids=[1])
            self.wire(runner, bad_reuse=True)
            with self.assertRaisesRegex(RuntimeError, "shape/reuse mismatch"):
                runner.run()
            baseline = json.loads((Path(tmp) / "cache_grid_results.json").read_text())
            self.assertTrue(baseline["complete"])
            self.assertEqual(
                [r["median_ttft_ms"] for r in baseline["metrics"]], [10.0, 10.0]
            )
            manifest = json.loads(
                next((Path(tmp) / "cache_profiles").glob("*/manifest.json")).read_text()
            )
            self.assertEqual(manifest["status"], "failed")
            self.assertTrue(manifest["records"][0]["trace_files"])

    def test_profile_api_error_prevents_target_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = self.runner(
                tmp, profile_runs=1, profile_case_ids=[1], profile_only=True
            )
            events = self.wire(runner)
            runner._http_session.post.side_effect = None
            runner._http_session.post.return_value.json.return_value = {
                "error": "broadcast failed"
            }
            with self.assertRaisesRegex(RuntimeError, "start_profile failed"):
                runner.run()
            self.assertEqual(events, ["seed"])

    def test_missing_trace_is_reported_as_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = self.runner(
                tmp, profile_runs=1, profile_case_ids=[2], profile_only=True
            )
            self.wire(runner, write_traces=False)
            with self.assertRaisesRegex(RuntimeError, "missing TP ranks"):
                runner.run()

    def test_incomplete_json_is_not_accepted_as_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = self.runner(
                tmp, profile_runs=1, profile_case_ids=[2], profile_only=True
            )
            (Path(tmp) / "example_wr0_1.json").write_text('{"traceEvents": [')
            try:
                with self.assertRaisesRegex(RuntimeError, "trace timeout"):
                    runner._wait_profile_traces("example")
            finally:
                runner._close_resources()

    def test_cli_profile_only_preserves_existing_report_directory(self):
        from rtp_llm.test.perf_test import batch_decode_test as entry

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            grid = root / "grid.json"
            grid.write_text(json.dumps({"cases": self.cases}))
            original_info = root / "test_info.json"
            original_info.write_text('{"original": true}')
            argv = [
                "--result_dir",
                tmp,
                "--cache_grid_json",
                str(grid),
                "--partial",
                "2",
                "--cache_commit_tail_tokens",
                "8",
                "--tokenizer_path",
                "/mock/tokenizer",
                "--tp_size",
                "2",
                "--cache_profile_only",
                "--cache_profile_runs",
                "1",
                "--cache_profile_case_ids",
                "1",
            ]
            parsed = parse_args(argv)
            with patch.object(entry, "parse_args", return_value=parsed), patch.object(
                entry, "_ensure_xgrammar_lib_path"
            ), patch.object(
                entry, "resolve_perf_engine_paths", side_effect=lambda x: x
            ), patch.object(
                entry, "EngineServer"
            ) as server, patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=_WhitespaceTokenizer(),
            ), patch.object(
                entry, "_collect_timeline_files"
            ), patch.object(
                CacheGridRunner, "run", autospec=True, return_value=[]
            ) as run, patch.dict(
                "os.environ", {}, clear=False
            ):
                server.return_value.port = 12345
                result_dir = Path(entry.main())
                runner = run.call_args.args[0]
                self.assertTrue(runner.profile_only)
                self.assertEqual(runner.profile_tp_size, 2)
                self.assertEqual(runner.profile_case_ids, {1})
                runner._close_resources()
                self.assertEqual(result_dir.parent, root / "cache_profile_replays")
                self.assertEqual(original_info.read_text(), '{"original": true}')
                self.assertTrue((result_dir / "test_info.json").is_file())
                server.return_value.stop.assert_called_once()

    def test_profile_cli_defaults_and_validation(self):
        args, _ = parse_args([])
        self.assertEqual(args.cache_profile_runs, 0)
        self.assertFalse(args.cache_profile_only)
        args, remaining = parse_args(
            [
                "--cache_grid_json",
                "grid.json",
                "--partial",
                "2",
                "--cache_profile_runs",
                "2",
                "--cache_profile_case_ids",
                "1",
                "2",
                "--cache_profile_only",
            ]
        )
        self.assertEqual(args.cache_profile_case_ids, [1, 2])
        self.assertFalse(remaining)
        for options in [
            ["--cache_profile_only"],
            ["--cache_profile_runs", "1"],
            ["--cache_profile_case_ids", "1"],
            ["--cache_profile_runs", "-1"],
            ["--cache_profile_trace_timeout", "0"],
        ]:
            with self.subTest(options=options), patch("sys.stderr"), self.assertRaises(
                SystemExit
            ):
                parse_args(options)


class CacheGridRunnerBlockProbeTest(unittest.TestCase):
    def _make_runner(self, tmp, block_size):
        cases = [{"case_id": 0, "batch_size": 1, "input_len": 1024, "cache_len": 512}]
        return CacheGridRunner(
            0,
            _WordTokenizer(),
            cases,
            tmp,
            expected_block_size=block_size,
        )

    def test_probe_passes_on_matching_reuse_len(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._make_runner(tmp, 512)
            with mock.patch(
                "rtp_llm.test.perf_test.cache_grid_runner._post_prefill",
                return_value={"success": True, "reuse_len": 512},
            ) as post:
                runner._probe_reuse_granularity()
            self.assertEqual(post.call_count, 2)

    def test_probe_aborts_on_granularity_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._make_runner(tmp, 512)
            with mock.patch(
                "rtp_llm.test.perf_test.cache_grid_runner._post_prefill",
                return_value={"success": True, "reuse_len": 0},
            ):
                with self.assertRaisesRegex(RuntimeError, "granularity mismatch"):
                    runner._probe_reuse_granularity()

    def test_probe_aborts_on_failed_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._make_runner(tmp, 512)
            with mock.patch(
                "rtp_llm.test.perf_test.cache_grid_runner._post_prefill",
                return_value={"success": False, "error": "HTTP 500"},
            ):
                with self.assertRaisesRegex(RuntimeError, "probe seed failed"):
                    runner._probe_reuse_granularity()

    def test_probe_skipped_without_block_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._make_runner(tmp, 0)
            with mock.patch(
                "rtp_llm.test.perf_test.cache_grid_runner._post_prefill"
            ) as post:
                runner._probe_reuse_granularity()
            post.assert_not_called()


class PrefixPromptFactoryFastPathTest(unittest.TestCase):
    GEOMETRIES = [(5, 100, 50), (5, 100, 0), (9, 7, 3), (9, 7, 0)]

    def test_fast_prompts_match_legacy_construction(self):
        for case_id, total_len, cache_len in self.GEOMETRIES:
            fast = PrefixPromptFactory(_WordTokenizer()).build_case_prompts(
                case_id, total_len, cache_len, 3
            )
            legacy = PrefixPromptFactory(_WordTokenizer())._legacy_case_prompts(
                case_id, total_len, cache_len, 3
            )
            self.assertEqual(fast.seed_text, legacy.seed_text)
            self.assertEqual(fast.prefix_ids, legacy.prefix_ids)
            self.assertEqual(fast.run_texts, legacy.run_texts)
            self.assertEqual(fast.run_ids, legacy.run_ids)
            self.assertEqual(fast.built_len, total_len)

    def test_first_case_of_each_shape_is_fully_verified(self):
        factory = PrefixPromptFactory(_WordTokenizer())
        factory.build_case_prompts(1, 100, 50, 3)
        factory.build_case_prompts(2, 100, 0, 3)
        self.assertEqual(factory._verified_shapes, {"cached", "cold"})
        self.assertEqual(factory._unsafe_shapes, set())

    def test_merging_tail_falls_back_to_legacy_path(self):
        factory = PrefixPromptFactory(_TailMergingTokenizer())
        # The legacy path itself cannot preserve the prefix for this tokenizer,
        # so the fallback surfaces its original safety error.
        with self.assertRaisesRegex(ValueError, "preserve cache prefix"):
            factory.build_case_prompts(1, 100, 50, 3)


class MaterializedCaseStoreTest(unittest.TestCase):
    CASES = [
        {"case_id": 0, "batch_size": 1, "input_len": 100, "cache_len": 0},
        {"case_id": 1, "batch_size": 1, "input_len": 100, "cache_len": 50},
        {"case_id": 2, "batch_size": 1, "input_len": 60, "cache_len": 40},
    ]

    def test_round_trip_reproduces_constructed_prompts(self):
        factory = PrefixPromptFactory(_WordTokenizer())
        with tempfile.TemporaryDirectory() as tmp:
            store = MaterializedCaseStore(tmp)
            stats = store.materialize(self.CASES, factory, 3, grid_sha256="s" * 64)
            self.assertEqual(stats["cases"], 3)
            self.assertEqual(stats["cached_cases"], 2)
            self.assertEqual(stats["verbatim_seed"], 0)
            self.assertEqual(stats["verbatim_runs"], 0)

            loaded = MaterializedCaseStore(tmp)
            self.assertEqual(loaded.load_cases(), self.CASES)
            self.assertEqual(loaded.run_count, 3)
            self.assertEqual(loaded.grid_sha256, "s" * 64)
            for case in self.CASES:
                expected = factory.build_case_prompts(
                    case["case_id"], case["input_len"], case["cache_len"], 3
                )
                got = loaded.load_case(case)
                self.assertEqual(got.seed_text, expected.seed_text)
                self.assertEqual(got.run_texts, expected.run_texts)

    def test_load_missing_store_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(
                FileNotFoundError, "not a materialized case store"
            ):
                MaterializedCaseStore(tmp).load_cases()


class MaterializedStoreValidationTest(unittest.TestCase):
    CASES = [
        {"case_id": 0, "batch_size": 1, "input_len": 100, "cache_len": 0},
        {"case_id": 1, "batch_size": 1, "input_len": 100, "cache_len": 50},
    ]

    def _materialize(self, tmp, run_count=3, sha="a" * 64):
        store = MaterializedCaseStore(tmp)
        store.materialize(
            self.CASES,
            PrefixPromptFactory(_WordTokenizer()),
            run_count,
            grid_sha256=sha,
        )

    def test_accepts_matching_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._materialize(tmp)
            store = _load_materialized_case_store(tmp, self.CASES, 3, "a" * 64)
            self.assertEqual(store.run_count, 3)

    def test_rejects_run_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._materialize(tmp)
            with self.assertRaisesRegex(ValueError, "run_count"):
                _load_materialized_case_store(tmp, self.CASES, 2, "a" * 64)

    def test_rejects_grid_sha_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._materialize(tmp)
            with self.assertRaisesRegex(ValueError, "different grid JSON"):
                _load_materialized_case_store(tmp, self.CASES, 3, "b" * 64)

    def test_rejects_geometry_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._materialize(tmp)
            other = [dict(self.CASES[0])]
            with self.assertRaisesRegex(
                ValueError, r"holds 2 cases but this run plans 1"
            ):
                _load_materialized_case_store(tmp, other, 3, "a" * 64)


class CacheGridRunnerStoreTest(unittest.TestCase):
    def test_run_sends_materialized_prompts_without_tokenizer_construction(self):
        cases = [
            {"case_id": 0, "batch_size": 1, "input_len": 100, "cache_len": 0},
            {"case_id": 1, "batch_size": 1, "input_len": 100, "cache_len": 50},
        ]
        factory = PrefixPromptFactory(_WordTokenizer())
        with tempfile.TemporaryDirectory() as tmp:
            store = MaterializedCaseStore(str(Path(tmp) / "store"))
            store.materialize(cases, factory, 2)
            expected = {
                case["case_id"]: factory.build_case_prompts(
                    case["case_id"], case["input_len"], case["cache_len"], 2
                )
                for case in cases
            }
            runner = CacheGridRunner(
                0,
                _WordTokenizer(),
                cases,
                str(Path(tmp) / "results"),
                measure_runs=2,
                cache_commit_tail_tokens=10,
                fail_fast=False,
                case_store=store,
            )
            with mock.patch(
                "rtp_llm.test.perf_test.cache_grid_runner._post_prefill",
                return_value={
                    "success": True,
                    "reuse_len": 50,
                    "input_len": 100,
                    "output_len": 1,
                },
            ) as post:
                runner.run()
            sent = [call.args[1] for call in post.call_args_list]
            self.assertEqual(
                sent,
                [
                    expected[0].run_texts[0],
                    expected[0].run_texts[1],
                    factory.make_seed(1, expected[1].seed_text, 50, 10),
                    expected[1].run_texts[0],
                    expected[1].run_texts[1],
                ],
            )

    def test_runner_rejects_store_built_for_other_run_count(self):
        cases = [{"case_id": 0, "batch_size": 1, "input_len": 100, "cache_len": 0}]
        with tempfile.TemporaryDirectory() as tmp:
            store = MaterializedCaseStore(str(Path(tmp) / "store"))
            store.materialize(cases, PrefixPromptFactory(_WordTokenizer()), 3)
            with self.assertRaisesRegex(ValueError, "run_count"):
                CacheGridRunner(
                    0, _WordTokenizer(), cases, tmp, measure_runs=2, case_store=store
                )


class CacheGridRunnerResumeGuardTest(unittest.TestCase):
    def _write_existing_results(self, result_dir, payload):
        import json

        path = Path(result_dir) / "cache_grid_results.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_resume_with_matching_results_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_existing_results(
                tmp,
                {
                    "metrics": [],
                    "grid_sha256": "abc",
                    "profile_sha256": "def",
                    "measure_runs": 3,
                    "expected_block_size": 512,
                },
            )
            runner = CacheGridRunner(
                0,
                _WordTokenizer(),
                [{"case_id": 0, "batch_size": 1, "input_len": 1024, "cache_len": 512}],
                tmp,
                grid_sha256="abc",
                profile_sha256="def",
                expected_block_size=512,
            )
            self.assertEqual(runner._results, {})

    def test_resume_with_profile_sha_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_existing_results(
                tmp,
                {
                    "metrics": [],
                    "grid_sha256": "abc",
                    "profile_sha256": "old_sha",
                    "measure_runs": 3,
                },
            )
            with self.assertRaisesRegex(ValueError, "profile_sha256"):
                CacheGridRunner(
                    0,
                    _WordTokenizer(),
                    [
                        {
                            "case_id": 0,
                            "batch_size": 1,
                            "input_len": 1024,
                            "cache_len": 512,
                        }
                    ],
                    tmp,
                    grid_sha256="abc",
                    profile_sha256="new_sha",
                )

    def test_resume_mismatch_allowed_with_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_existing_results(
                tmp,
                {
                    "metrics": [
                        {
                            "case_key": "bs1_seq1024_cache512",
                            "status": "ok",
                        }
                    ],
                    "grid_sha256": "abc",
                    "profile_sha256": "old",
                    "measure_runs": 3,
                },
            )
            runner = CacheGridRunner(
                0,
                _WordTokenizer(),
                [{"case_id": 0, "batch_size": 1, "input_len": 1024, "cache_len": 512}],
                tmp,
                grid_sha256="abc",
                profile_sha256="new",
                allow_resume_mismatch=True,
            )
            self.assertEqual(len(runner._results), 1)

    def test_require_resume_rejects_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "require_cache_resume"):
                validate_cache_grid_resume(
                    tmp,
                    grid_sha256="grid",
                    profile_sha256="profile",
                    measure_runs=3,
                    cache_commit_tail_tokens=128,
                    expected_block_size=512,
                    request_transport="http_prompt",
                    require_resume=True,
                )

    def test_resume_config_mismatch_is_rejected_before_model_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_config = {"model": {"checkpoint_path": "/weights/old"}}
            self._write_existing_results(
                tmp,
                {
                    "metrics": [],
                    "run_config_sha256": resume_config_fingerprint(old_config),
                },
            )
            with self.assertRaisesRegex(ValueError, "run_config_sha256"):
                validate_cache_grid_resume(
                    tmp,
                    grid_sha256=None,
                    profile_sha256=None,
                    measure_runs=3,
                    cache_commit_tail_tokens=128,
                    expected_block_size=0,
                    request_transport="http_prompt",
                    run_config={"model": {"checkpoint_path": "/weights/new"}},
                )

    @patch("rtp_llm.test.perf_test.cache_grid_runner._post_prefill")
    def test_resume_skips_ok_case_and_starts_at_next_input_cache(self, post):
        cases = [
            {"case_id": 0, "batch_size": 1, "input_len": 8, "cache_len": 0},
            {"case_id": 1, "batch_size": 1, "input_len": 16, "cache_len": 0},
        ]
        post.return_value = {
            "success": True,
            "input_len": 16,
            "output_len": 1,
            "reuse_len": 0,
            "ttft_ms": 5.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            self._write_existing_results(
                tmp,
                {
                    "metrics": [
                        {
                            "case_key": "bs1_seq8_cache0",
                            "case_id": 0,
                            "status": "ok",
                        }
                    ],
                    "measure_runs": 1,
                    "request_transport": "http_prompt",
                    "started_at": "2026-09-08T00:00:00+0800",
                },
            )
            rows = CacheGridRunner(
                12345,
                _WordTokenizer(),
                cases,
                tmp,
                measure_runs=1,
                cache_commit_tail_tokens=4,
            ).run()
            result = json.loads(
                (Path(tmp) / "cache_grid_results.json").read_text(encoding="utf-8")
            )
        self.assertEqual(post.call_count, 1)
        self.assertEqual(len(rows), 2)
        self.assertTrue(result["complete"])
        self.assertEqual(result["resume_count"], 1)
        self.assertEqual(result["progress"]["completed_cases"], 2)
        self.assertIsNone(result["progress"]["next_pending_case"])

    def test_resume_replays_case_journal_newer_than_full_checkpoint(self):
        cases = [
            {"case_id": 0, "batch_size": 1, "input_len": 8, "cache_len": 0},
            {"case_id": 1, "batch_size": 1, "input_len": 16, "cache_len": 0},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            first = CacheGridRunner(
                0,
                _WordTokenizer(),
                cases,
                tmp,
                measure_runs=1,
                cache_commit_tail_tokens=4,
            )
            first._save(complete=False)
            metric = {
                "case_key": first.case_key(cases[0]),
                "case_id": 0,
                "status": "ok",
            }
            first._append_journal(metric)
            first._close_resources()

            resumed = CacheGridRunner(
                0,
                _WordTokenizer(),
                cases,
                tmp,
                measure_runs=1,
                cache_commit_tail_tokens=4,
            )
            try:
                self.assertEqual(
                    resumed._results[resumed.case_key(cases[0])]["status"], "ok"
                )
                progress = resumed._progress()
                self.assertEqual(progress["completed_cases"], 1)
                self.assertEqual(progress["next_pending_case"]["input_len"], 16)
            finally:
                resumed._close_resources()

    @patch("rtp_llm.test.perf_test.cache_grid_runner._post_prefill")
    def test_keyboard_interrupt_flushes_resumable_next_case(self, post):
        post.side_effect = KeyboardInterrupt()
        case = {
            "case_id": 7,
            "batch_size": 1,
            "input_len": 16,
            "cache_len": 0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            runner = CacheGridRunner(
                12345,
                _WordTokenizer(),
                [case],
                tmp,
                measure_runs=1,
                cache_commit_tail_tokens=4,
            )
            with self.assertRaises(KeyboardInterrupt):
                runner.run()
            result = json.loads(
                (Path(tmp) / "cache_grid_results.json").read_text(encoding="utf-8")
            )
        self.assertFalse(result["complete"])
        self.assertEqual(result["progress"]["completed_cases"], 0)
        self.assertEqual(
            result["progress"]["next_pending_case"],
            {
                "case_key": "bs1_seq16_cache0",
                "case_id": 7,
                "batch_size": 1,
                "input_len": 16,
                "cache_len": 0,
            },
        )

    def test_save_includes_profile_and_measure_runs(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            profile = {"schema_version": 1, "label": "test"}
            cases = [
                {"case_id": 0, "batch_size": 1, "input_len": 1024, "cache_len": 0},
                {"case_id": 1, "batch_size": 1, "input_len": 2048, "cache_len": 512},
            ]
            run_config = {"model": {"checkpoint_path": "/weights/model"}}
            runner = CacheGridRunner(
                0,
                _WordTokenizer(),
                cases,
                tmp,
                measure_runs=5,
                expected_block_size=256,
                profile=profile,
                profile_sha256="fp123",
                run_config=run_config,
            )
            runner._results[runner.case_key(cases[0])] = {
                "case_key": runner.case_key(cases[0]),
                "status": "ok",
            }
            runner._save(complete=False)
            path = Path(tmp) / "cache_grid_results.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["measure_runs"], 5)
            self.assertEqual(payload["expected_block_size"], 256)
            self.assertEqual(payload["profile"], profile)
            self.assertEqual(payload["profile_sha256"], "fp123")
            self.assertEqual(payload["schema_version"], 2)
            self.assertEqual(payload["run_config"], run_config)
            self.assertEqual(
                payload["run_config_sha256"], resume_config_fingerprint(run_config)
            )
            self.assertEqual(payload["progress"]["completed_cases"], 1)
            self.assertEqual(
                payload["progress"]["last_completed_case"]["input_len"], 1024
            )
            self.assertEqual(
                payload["progress"]["next_pending_case"]["input_len"], 2048
            )
            self.assertEqual(payload["progress"]["next_pending_case"]["cache_len"], 512)


class MaterializedCaseStoreProfileTest(unittest.TestCase):
    def test_materialize_persists_profile_sha256(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            store = MaterializedCaseStore(tmp)
            factory = PrefixPromptFactory(_WordTokenizer())
            cases = [
                {"case_id": 0, "batch_size": 1, "input_len": 1024, "cache_len": 512}
            ]
            store.materialize(
                cases, factory, run_count=3, profile_sha256="sha_profile_42"
            )
            info = json.loads(
                (Path(tmp) / "store_info.json").read_text(encoding="utf-8")
            )
            self.assertEqual(info["profile_sha256"], "sha_profile_42")
            self.assertEqual(store.profile_sha256, "sha_profile_42")

    def test_load_reads_profile_sha256(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            store1 = MaterializedCaseStore(tmp)
            factory = PrefixPromptFactory(_WordTokenizer())
            cases = [
                {"case_id": 0, "batch_size": 1, "input_len": 1024, "cache_len": 512}
            ]
            store1.materialize(
                cases, factory, run_count=3, profile_sha256="persisted_sha"
            )
            store2 = MaterializedCaseStore(tmp)
            store2.load_cases()
            self.assertEqual(store2.profile_sha256, "persisted_sha")


if __name__ == "__main__":
    unittest.main()

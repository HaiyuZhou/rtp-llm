"""Backend-neutral distributed runtime regression tests."""

import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm.test.perf_test.entry import plan_from_source
from rtp_llm.test.perf_test.execution.distributed import prefill as prefill_runtime
from rtp_llm.test.perf_test.tests.support.fixtures import (
    prefill_external_aux,
    prefill_external_CASE_CACHE_LEN,
    prefill_external_CASE_INPUT_LEN,
    prefill_external_FakeModelServer,
    prefill_external_Tokenizer,
    prefill_external_write_prefill_config,
)


class DistributedRuntimeTest(unittest.TestCase):

    def test_external_prefill_uses_same_deadline_for_http_and_dashsc(self):
        from rtp_llm.test.perf_test.execution.runtime import HttpPrefillClient

        for options, seconds in (({}, 7200), ({"timeout_seconds": 1234}, 1234)):
            with self.subTest(options=options):
                post = Mock(
                    return_value=SimpleNamespace(
                        status_code=200, json=lambda: {"aux_info": {}}
                    )
                )

                def http(endpoint, **kwargs):
                    return HttpPrefillClient(endpoint, post=post, **kwargs)

                with patch.object(
                    prefill_runtime, "HttpPrefillClient", side_effect=http
                ), patch.object(
                    importlib.import_module(
                        "rtp_llm.test.perf_test.prefill.dashsc_transport"
                    ),
                    "DashScPrefillClient",
                ) as dashsc:
                    session = prefill_runtime.ExternalPrefillSession(
                        "http://localhost:18000",
                        topology={},
                        transport="http_text",
                        **options,
                    )
                    session.client.request(
                        phase="probe",
                        case_id="probe",
                        request_index=0,
                        run_index=0,
                        prompt="hello",
                        input_ids=[1],
                        output_length=1,
                    )
                    self.assertEqual(seconds, post.call_args.kwargs["timeout"])
                    prefill_runtime.ExternalPrefillSession(
                        "http://localhost:18000",
                        topology={},
                        transport="dashsc_input_ids",
                        **options,
                    )
                    self.assertEqual(
                        seconds, dashsc.call_args.kwargs["timeout_seconds"]
                    )

    def test_external_prefill_master_completes_against_http_service(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (plan, source) = plan_from_source(
                json.loads(
                    Path(prefill_external_write_prefill_config(root)).read_text()
                ),
                source_dir=Path(prefill_external_write_prefill_config(root)).parent,
                result_directory=str(root / "run"),
            )
            with prefill_external_FakeModelServer(
                [
                    prefill_external_aux(16, 0, 5.0),
                    prefill_external_aux(16, 8, 5.0),
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
                with patch(
                    "rtp_llm.test.perf_test.test_util._load_tokenizer",
                    return_value=prefill_external_Tokenizer(),
                ):
                    result = prefill_runtime.execute_master_prefill(
                        plan,
                        source_config=source,
                        endpoint=server.endpoint,
                        tokenizer_path="/m",
                        server_log_path=None,
                        profiler_directory=root,
                    )
            self.assertEqual("completed", result.state.value)
            self.assertEqual(
                "completed",
                json.loads((root / "run/results/result.json").read_text())["status"],
            )
            self.assertTrue((root / "run/results/summary.json").is_file())

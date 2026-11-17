"""Unified prefill planning and session access for an externally owned service."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from rtp_llm.test.perf_test.config_schema import (
    DEFAULT_PREFILL_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_PREFILL_TRANSPORT,
)
from rtp_llm.test.perf_test.execution.engine_session import EngineDiagnostics
from rtp_llm.test.perf_test.execution.runtime import HttpPrefillClient
from rtp_llm.test.perf_test.modes import ModeExecutionContext, create_prefill_adapter
from rtp_llm.test.perf_test.plan import thaw
from rtp_llm.test.perf_test.prefill.finalize import finalize_prefill_run
from rtp_llm.test.perf_test.prefill.report import generate_prefill_report
from rtp_llm.test.perf_test.run_directory import initialize_run_directory
from rtp_llm.test.perf_test.storage import RunStore


class ExternalPrefillSession:
    """Prefill-runner session handle for a service another lifecycle owns.

    Satisfies the session contract the prefill runner consumes (``client``,
    ``diagnostics``, context-manager ``close``) without ever starting or
    stopping a process: the gang worker owns the model service.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        topology: Dict[str, int],
        transport: str = DEFAULT_PREFILL_TRANSPORT,
        timeout_seconds: float = DEFAULT_PREFILL_REQUEST_TIMEOUT_SECONDS,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.topology = dict(topology)
        if transport not in ("http_text", "dashsc_input_ids"):
            raise ValueError(f"unsupported prefill transport: {transport}")
        # Use the same configured request budget for both inference transports.
        self._client = HttpPrefillClient(self.endpoint, timeout_seconds=timeout_seconds)
        if transport == "dashsc_input_ids":
            from rtp_llm.test.perf_test.prefill.dashsc_transport import (
                DashScPrefillClient,
            )

            self._client = DashScPrefillClient(
                self._client, timeout_seconds=timeout_seconds
            )
            try:
                self._client.wait_ready(600)
            except BaseException:
                self._client.close()
                raise
        self._diagnostics = EngineDiagnostics()

    @property
    def client(self) -> Any:
        return self._client

    @property
    def diagnostics(self) -> EngineDiagnostics:
        return self._diagnostics

    def close(self) -> None:
        """Release the inference channel; the caller owns the service process."""
        if hasattr(self._client, "close"):
            self._client.close()

    def __enter__(self) -> "ExternalPrefillSession":
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> Optional[bool]:
        self.close()
        return None


def execute_master_prefill(
    plan,
    *,
    source_config,
    endpoint,
    tokenizer_path,
    server_log_path,
    profiler_directory,
    run_context=None,
    model_paths: Optional[Mapping[str, Any]] = None,
):
    """Run the shared prefill measurement against a caller-owned rank group."""
    del server_log_path, profiler_directory
    root = Path(plan.result_directory)
    run_context = run_context or {}
    run_id = run_context.get("run_id", root.name)
    initialize_run_directory(
        root,
        source_config=source_config,
        plan=plan,
        run_id=run_id,
        code_version=os.environ.get("BUILD_SCM_REVISION", "unknown"),
        repository_revisions=run_context.get("repository_revisions", {}),
    )
    store = RunStore(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["state"] = "running"
    store.write_manifest(manifest)
    config = thaw(plan.resolved_config)
    workload = config["workload"]
    model = config["model"]
    tokenizer_model = model if model_paths is None else model_paths
    session = None
    try:
        from rtp_llm.test.perf_test.test_util import _load_tokenizer

        tokenizer = _load_tokenizer(
            tokenizer_path,
            checkpoint_path=str(tokenizer_model["checkpoint"]),
            model_type=str(tokenizer_model["type"]),
        )
        session = ExternalPrefillSession(
            endpoint,
            topology=dict(model["topology"]),
            transport=workload.get("transport", DEFAULT_PREFILL_TRANSPORT),
            timeout_seconds=workload.get(
                "request_timeout_seconds", DEFAULT_PREFILL_REQUEST_TIMEOUT_SECONDS
            ),
        )
        adapter = create_prefill_adapter(config, tokenizer)
        result = adapter(ModeExecutionContext(plan=plan, session=session, store=store))
        finalize_prefill_run(
            root,
            plan=plan,
            run_id=run_id,
            case_results=result.case_results,
            terminal_state=result.state,
        )
    except Exception:
        manifest["state"] = "failed"
        store.write_manifest(manifest)
        raise
    finally:
        if session is not None:
            session.close()
    try:
        generate_prefill_report(root, acquire_lock=False)
    except Exception:
        manifest["state"] = "failed"
        store.write_manifest(manifest)
        raise
    manifest["state"] = result.state.value
    store.write_manifest(manifest)
    return result

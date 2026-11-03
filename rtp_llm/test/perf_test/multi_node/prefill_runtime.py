"""Unified prefill planning and session access for an externally owned service."""

from __future__ import annotations

from typing import Any, Dict, Optional

from rtp_llm.test.perf_test.config_resolver import load_resolved_config
from rtp_llm.test.perf_test.config_schema import PrefillWorkloadConfig, parse_config
from rtp_llm.test.perf_test.entry import _source_config
from rtp_llm.test.perf_test.errors import ConfigurationError, PlanningError
from rtp_llm.test.perf_test.execution.engine_session import EngineDiagnostics
from rtp_llm.test.perf_test.execution.runtime import HttpPrefillClient
from rtp_llm.test.perf_test.plan import build_execution_plan
from rtp_llm.test.perf_test.prefill.planner import plan_prefill_cases


def build_external_prefill_plan(config_path, result_directory, actual_topology):
    """Build the local prefill case identities without creating run artifacts.

    Mirrors ``build_external_decode_plan``: the configured topology must
    match the topology the running service was started with, and the
    returned plan is fully frozen (with standard case IDs and fingerprints)
    before any run directory is initialized.
    """
    source = _source_config(str(config_path))
    resolved = load_resolved_config(config_path)
    config = parse_config(source)
    if not isinstance(config.workload, PrefillWorkloadConfig):
        raise ConfigurationError("external prefill requires workload.kind=cache_grid")
    configured = config.model.topology.as_tuple()
    actual = tuple(actual_topology.get(name) for name in ("tp", "dp", "pp", "ep"))
    if any(type(value) is not int or value <= 0 for value in actual):
        raise PlanningError(
            "runtime topology must supply positive integer TP, DP, PP, and EP sizes"
        )
    if configured != actual:
        raise PlanningError(
            f"runtime topology {actual} does not match configured topology {configured}"
        )
    cases = plan_prefill_cases(
        config.workload, config.engine, dp_size=config.model.topology.dp
    )
    plan = build_execution_plan(
        mode="prefill",
        action="run",
        resolved=resolved,
        cases=cases,
        result_directory=str(result_directory),
    )
    return plan, source


class ExternalPrefillSession:
    """Prefill-runner session handle for a service another lifecycle owns.

    Satisfies the session contract the prefill runner consumes (``client``,
    ``diagnostics``, context-manager ``close``) without ever starting or
    stopping a process: the gang worker owns the model service.
    """

    def __init__(
        self, endpoint: str, *, topology: Dict[str, int], transport: str = "http_text"
    ):
        self.endpoint = endpoint.rstrip("/")
        self.topology = dict(topology)
        if transport not in ("http_text", "dashsc_input_ids"):
            raise ValueError(f"unsupported prefill transport: {transport}")
        self._client = HttpPrefillClient(self.endpoint)
        if transport == "dashsc_input_ids":
            from rtp_llm.test.perf_test.prefill.dashsc_transport import (
                DashScPrefillClient,
            )

            self._client = DashScPrefillClient(self._client)
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

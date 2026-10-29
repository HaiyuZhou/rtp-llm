"""ODPS publisher for finalized aggregate performance summaries."""

from __future__ import annotations

import copy
import logging
import os
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Protocol

from rtp_llm.test.perf_test.results import parse_result_envelope


@dataclass(frozen=True)
class OdpsCredentials:
    access_key_id: str = field(repr=False)
    access_key_secret: str = field(repr=False)
    project: str
    endpoint: str

    @classmethod
    def from_environment(cls, environment: Mapping[str, str]) -> "OdpsCredentials":
        access_key_id = environment.get("ALIBABA_CLOUD_ACCESS_KEY_ID", "")
        access_key_secret = environment.get("ALIBABA_CLOUD_ACCESS_KEY_SECRET", "")
        if not access_key_id or not access_key_secret:
            raise RuntimeError(
                "ALIBABA_CLOUD_ACCESS_KEY_ID / ALIBABA_CLOUD_ACCESS_KEY_SECRET "
                "are required for ODPS publication"
            )
        return cls(
            access_key_id=access_key_id,
            access_key_secret=access_key_secret,
            project=environment.get("ODPS_PROJECT", "rank_service_dev"),
            endpoint=environment.get(
                "ODPS_ENDPOINT", "http://service-corp.odps.aliyun-inc.com/api"
            ),
        )


class SummaryClient(Protocol):
    def write_summary(self, destination: str, payload: Mapping[str, Any]) -> None:
        """Write one aggregate summary record."""


class SummaryClientFactory(Protocol):
    def create(self, credentials: OdpsCredentials) -> SummaryClient:
        """Create a client using runtime-only credentials."""


class _PyOdpsSummaryClient:
    def __init__(self, client: Any):
        self.client = client

    def write_summary(self, destination: str, payload: Mapping[str, Any]) -> None:
        table = self.client.get_table(destination)
        with table.open_writer() as writer:
            writer.write([dict(payload)])


class PyOdpsClientFactory:
    """Lazy PyODPS adapter so importing perf-test needs no ODPS dependency."""

    def create(self, credentials: OdpsCredentials) -> SummaryClient:
        from odps import ODPS

        client = ODPS(
            credentials.access_key_id,
            credentials.access_key_secret,
            credentials.project,
            endpoint=credentials.endpoint,
        )
        return _PyOdpsSummaryClient(client)


def _mean(values: list[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def _prefill_aggregate(cases: Any) -> Dict[str, Any]:
    metrics = [item.metrics for item in cases if item.metrics is not None]
    return {
        "valid_cases": len(metrics),
        "mean_latency_ms": _mean([float(item.latency_ms) for item in metrics]),
        "mean_throughput_tokens_per_second": _mean(
            [float(item.throughput_tokens_per_second) for item in metrics]
        ),
    }


def _decode_aggregate(cases: Any) -> Dict[str, Any]:
    metrics = [item.metrics for item in cases if item.metrics is not None]
    latency_names = ("wait", "prefill", "decode", "tpot", "total")
    return {
        "valid_cases": len(metrics),
        "successful_requests": sum(item.successful_requests for item in metrics),
        "failed_requests": sum(item.failed_requests for item in metrics),
        "mean_latency_ms": {
            name: _mean([float(item.latency_ms[name]) for item in metrics])
            for name in latency_names
        },
        "mean_throughput_tokens_per_second": _mean(
            [float(item.throughput_tokens_per_second) for item in metrics]
        ),
    }


def build_summary_payload(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Project a terminal result onto the stable aggregate ODPS contract."""
    envelope = parse_result_envelope(result)
    aggregate = (
        _prefill_aggregate(envelope.cases)
        if envelope.mode == "prefill"
        else _decode_aggregate(envelope.cases)
    )
    return {
        "schema_version": 1,
        "run_id": envelope.run_id,
        "mode": envelope.mode,
        "backend": envelope.backend,
        "status": envelope.status.value,
        "config_fingerprint": envelope.config_fingerprint,
        "workload_fingerprint": envelope.workload_fingerprint,
        "model": copy.deepcopy(envelope.model),
        "summary": copy.deepcopy(envelope.summary),
        "aggregate": aggregate,
    }


def _redact(message: str, credentials: OdpsCredentials) -> str:
    redacted = message
    for value in (credentials.access_key_id, credentials.access_key_secret):
        if value:
            redacted = redacted.replace(value, "***")
    return redacted


class OdpsPublisher:
    """Publish only the finalized aggregate projection of one result."""

    def __init__(
        self,
        *,
        client_factory: SummaryClientFactory,
        environment: Optional[Mapping[str, str]] = None,
    ):
        self.client_factory = client_factory
        self.environment = dict(os.environ if environment is None else environment)

    def publish(
        self,
        result: Mapping[str, Any],
        *,
        destination: str,
        logger: Optional[logging.Logger] = None,
    ) -> Dict[str, Any]:
        if not isinstance(destination, str) or not destination:
            raise ValueError("ODPS destination must be a non-empty string")
        payload = build_summary_payload(result)
        credentials = OdpsCredentials.from_environment(self.environment)
        active_logger = logger or logging.getLogger(__name__)
        try:
            client = self.client_factory.create(credentials)
            client.write_summary(destination, payload)
        except Exception as error:
            message = _redact(str(error), credentials)
            active_logger.error("ODPS summary publication failed: %s", message)
            raise RuntimeError(f"ODPS summary publication failed: {message}") from error
        active_logger.info(
            "published %s aggregate summary for run %s to ODPS",
            payload["mode"],
            payload["run_id"],
        )
        return {
            "publisher": "odps",
            "status": "succeeded",
            "destination": destination,
            "mode": payload["mode"],
            "run_id": payload["run_id"],
        }

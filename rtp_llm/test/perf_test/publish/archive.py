"""Allowlisted run artifact archives and OSS upload adapter."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Union

_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


class ArchiveSelectionError(ValueError):
    """Raised when an artifact allowlist is invalid or unsafe."""


@dataclass(frozen=True)
class OssCredentials:
    key_id: str = field(repr=False)
    key_secret: str = field(repr=False)
    endpoint: str = field(repr=False)

    @classmethod
    def from_environment(cls, environment: Mapping[str, str]) -> "OssCredentials":
        key_id = environment.get("MAGA_OSS_KEY_ID", "")
        key_secret = environment.get("MAGA_OSS_KEY_SECRET", "")
        endpoint = environment.get("OSS_ENDPOINT", "oss-cn-zhangjiakou.aliyuncs.com")
        if not key_id or not key_secret:
            raise RuntimeError(
                "MAGA_OSS_KEY_ID / MAGA_OSS_KEY_SECRET are required for OSS publication"
            )
        return cls(key_id=key_id, key_secret=key_secret, endpoint=endpoint)


class ArchiveUploader(Protocol):
    def upload(
        self, archive: Path, destination: str, credentials: OssCredentials
    ) -> str:
        """Upload one local archive and return its remote URI."""


class OsscmdUploader:
    """Upload archives through the legacy osscmd executable."""

    def __init__(self, *, timeout_seconds: float = 600):
        self.timeout_seconds = timeout_seconds

    def upload(
        self, archive: Path, destination: str, credentials: OssCredentials
    ) -> str:
        command = [
            "osscmd",
            "put",
            str(archive),
            destination,
            "-H",
            credentials.endpoint,
            "-i",
            credentials.key_id,
            "-k",
            credentials.key_secret,
            "--replace=true",
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"osscmd exited {completed.returncode}: {completed.stderr.strip()}"
            )
        return destination


def _selected_files(root: Path, artifacts: Sequence[str]) -> Dict[str, Path]:
    if not artifacts:
        raise ArchiveSelectionError("artifact allowlist must not be empty")
    resolved_root = root.resolve()
    selected: Dict[str, Path] = {}
    for index, raw in enumerate(artifacts):
        if not isinstance(raw, str) or not raw:
            raise ArchiveSelectionError(f"artifacts[{index}] must be a relative path")
        relative = Path(raw)
        if relative.is_absolute() or ".." in relative.parts:
            raise ArchiveSelectionError(f"artifact path must be relative: {raw}")
        normalized = relative.as_posix()
        if normalized in selected:
            raise ArchiveSelectionError(f"duplicate artifact path: {normalized}")
        source = (resolved_root / relative).resolve()
        try:
            source.relative_to(resolved_root)
        except ValueError as error:
            raise ArchiveSelectionError(
                f"artifact path must remain relative to run root: {raw}"
            ) from error
        if not source.exists():
            raise ArchiveSelectionError(f"configured artifact is missing: {normalized}")
        if not source.is_file():
            raise ArchiveSelectionError(
                f"configured artifact must be a regular file: {normalized}"
            )
        selected[normalized] = source
    return dict(sorted(selected.items()))


def _redact(message: str, credentials: OssCredentials) -> str:
    redacted = message
    for value in (credentials.key_id, credentials.key_secret):
        if value:
            redacted = redacted.replace(value, "***")
    return redacted


class OssArchivePublisher:
    """Create one allowlisted archive and upload it without persisting secrets."""

    def __init__(
        self,
        *,
        uploader: ArchiveUploader,
        environment: Optional[Mapping[str, str]] = None,
    ):
        self.uploader = uploader
        self.environment = dict(os.environ if environment is None else environment)

    def publish(
        self,
        root: Union[str, Path],
        *,
        destination: str,
        artifacts: Sequence[str],
        run_id: str,
        logger: Optional[logging.Logger] = None,
    ) -> Dict[str, Any]:
        if not isinstance(destination, str) or not destination:
            raise ArchiveSelectionError("OSS destination must be a non-empty string")
        if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
            raise ArchiveSelectionError("run_id is not safe for an archive name")
        selected = _selected_files(Path(root), artifacts)
        credentials = OssCredentials.from_environment(self.environment)
        remote_uri = f"{destination.rstrip('/')}/{run_id}.tar.gz"
        active_logger = logger or logging.getLogger(__name__)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f"perf-test-{run_id}-", suffix=".tar.gz"
        )
        os.close(descriptor)
        archive = Path(temporary_name)
        try:
            with tarfile.open(archive, "w:gz") as stream:
                for relative, source in selected.items():
                    stream.add(source, arcname=relative, recursive=False)
            active_logger.info(
                "publishing %d perf-test artifacts to %s",
                len(selected),
                remote_uri,
            )
            try:
                uploaded_uri = self.uploader.upload(archive, remote_uri, credentials)
            except Exception as error:
                message = _redact(str(error), credentials)
                active_logger.error("OSS archive publication failed: %s", message)
                raise RuntimeError(
                    f"OSS archive publication failed: {message}"
                ) from error
        finally:
            archive.unlink(missing_ok=True)
        return {
            "publisher": "oss",
            "status": "succeeded",
            "uri": uploaded_uri,
            "artifacts": list(selected),
        }

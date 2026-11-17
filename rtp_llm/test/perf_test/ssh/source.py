"""Read-only source freezing and verified native-build identities."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from rtp_llm.test.perf_test.errors import ConfigurationError


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise ConfigurationError(f"cannot resolve SSH source: {result.stderr.strip()}")
    return result.stdout.strip()


def resolve_source(source: Mapping[str, Any], *, source_dir: Path) -> dict:
    """Freeze local Git input before any SSH side effect; never fetch a branch."""
    result = dict(source)
    configured = result.get("worktree")
    root = Path(configured) if configured else Path(__file__).resolve().parents[4]
    if not root.is_absolute():
        root = Path(source_dir) / root
    root = Path(_git(root.resolve(), "rev-parse", "--show-toplevel")).resolve()
    if _git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ConfigurationError(
            "SSH source contains uncommitted tracked changes; commit them before running"
        )
    ref = result.get("ref", "auto")
    ref = "HEAD" if ref == "auto" else ref
    if not isinstance(ref, str) or not ref or ref.startswith("-"):
        raise ConfigurationError("SSH source.ref must be a local Git reference")
    result["ref"] = _git(root, "rev-parse", "--verify", f"{ref}^{{commit}}")
    result["worktree"] = str(root)
    return result


def build_identity(source: Mapping[str, Any]) -> str:
    fields = {
        key: source.get(key)
        for key in ("ref", "bazel_config", "bazel_targets", "bazel_args")
    }
    return hashlib.sha256(
        json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_receipt(
    receipt: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    source_dir: Path,
    output_user_root: Path = None,
) -> None:
    """A driver launch acknowledgement is insufficient: verify native bytes."""
    if (
        receipt.get("schema_version") != 1
        or receipt.get("status") != "completed"
        or receipt.get("bazel_exit_code") != 0
        or receipt.get("source_sha") != source["ref"]
    ):
        raise ConfigurationError(
            "build receipt does not prove completion for the frozen source"
        )
    if receipt.get("build_identity", build_identity(source)) != build_identity(source):
        raise ConfigurationError("build receipt has a different build identity")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ConfigurationError("build receipt requires nonempty artifact hashes")
    for name, expected in artifacts.items():
        if (
            not isinstance(name, str)
            or not isinstance(expected, str)
            or len(expected) != 64
        ):
            raise ConfigurationError("invalid build artifact receipt")
        path = Path(name)
        if not path.is_absolute():
            path = source_dir / path
        allowed = [source_dir.resolve()] + (
            [output_user_root.resolve()] if output_user_root is not None else []
        )
        if not any(path.resolve().is_relative_to(root) for root in allowed):
            raise ConfigurationError(
                f"build artifact is outside approved roots: {name}"
            )
        if not path.is_file() or file_digest(path) != expected:
            raise ConfigurationError(f"build artifact does not match receipt: {name}")

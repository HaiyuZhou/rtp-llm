"""Remote AOne worker input loading from declared Bazel runfiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

from rtp_llm.test.perf_test.config_resolver import canonical_json
from rtp_llm.test.perf_test.errors import ConfigurationError


@dataclass(frozen=True)
class RemoteInputs:
    run_id: str
    shard_id: str
    config_fingerprint: str
    workload_fingerprint: str
    config: Dict[str, Any]
    owned_case_ids: Tuple[str, ...]
    owned_cases: Tuple[Dict[str, Any], ...]


def _load_object(path: Union[str, Path], name: str) -> Dict[str, Any]:
    artifact = Path(path)
    try:
        value = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigurationError(
            f"cannot read {name} runfile {artifact}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise ConfigurationError(f"{name} runfile must contain an object")
    return value


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_remote_inputs(
    *,
    resolved_config: Union[str, Path],
    resolved_workload: Union[str, Path],
    shard_plan: Union[str, Path],
    shard_id: str,
) -> RemoteInputs:
    """Load and validate one shard without consulting submit-host paths."""
    config = _load_object(resolved_config, "resolved config")
    workload = _load_object(resolved_workload, "resolved workload")
    ownership = _load_object(shard_plan, "shard plan")
    if ownership.get("schema_version") != 1:
        raise ConfigurationError("unsupported AOne shard plan schema_version")
    expected_config = ownership.get("config_fingerprint")
    expected_workload = ownership.get("workload_fingerprint")
    if _fingerprint(config) != expected_config:
        raise ConfigurationError("AOne resolved config fingerprint mismatch")
    if _fingerprint(workload) != expected_workload:
        raise ConfigurationError("AOne resolved workload fingerprint mismatch")

    shards = ownership.get("shards")
    if not isinstance(shards, list):
        raise ConfigurationError("AOne shard plan shards must be an array")
    selected: Mapping[str, Any] | None = None
    seen_ids = set()
    for item in shards:
        if not isinstance(item, Mapping) or not isinstance(item.get("shard_id"), str):
            raise ConfigurationError("AOne shard plan contains an invalid shard")
        current = item["shard_id"]
        if current in seen_ids:
            raise ConfigurationError(f"duplicate AOne shard ID: {current}")
        seen_ids.add(current)
        if current == shard_id:
            selected = item
    if selected is None:
        raise ConfigurationError(f"unknown AOne shard ID: {shard_id}")
    raw_cases = selected.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ConfigurationError(f"AOne shard {shard_id} has no owned cases")
    cases = []
    case_ids = []
    for item in raw_cases:
        if not isinstance(item, dict) or not isinstance(item.get("case_id"), str):
            raise ConfigurationError(f"AOne shard {shard_id} contains an invalid case")
        cases.append(dict(item))
        case_ids.append(item["case_id"])
    if len(set(case_ids)) != len(case_ids):
        raise ConfigurationError(f"AOne shard {shard_id} contains duplicate cases")

    combined = dict(config)
    combined["workload"] = workload
    run_id = ownership.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ConfigurationError("AOne shard plan run_id must be a non-empty string")
    return RemoteInputs(
        run_id=run_id,
        shard_id=shard_id,
        config_fingerprint=expected_config,
        workload_fingerprint=expected_workload,
        config=combined,
        owned_case_ids=tuple(case_ids),
        owned_cases=tuple(cases),
    )


def _default_executor(inputs: RemoteInputs):
    from rtp_llm.test.perf_test.aone.executor import RemoteShardExecutor
    from rtp_llm.test.perf_test.execution.defaults import _default_session_factory
    from rtp_llm.test.perf_test.modes.prefill import PrefillAdapter
    from rtp_llm.test.perf_test.prefill.runner import PrefillRunner
    from rtp_llm.test.perf_test.test_util import _load_tokenizer

    model = inputs.config["model"]
    tokenizer = _load_tokenizer(
        model["tokenizer"],
        checkpoint_path=model["checkpoint"],
        model_type=model["type"],
    )
    commit_tail = int(inputs.config["workload"]["commit_tail_tokens"])
    adapter = PrefillAdapter(
        lambda context: PrefillRunner(
            context.session.client,
            tokenizer,
            commit_tail_tokens=commit_tail,
        )
    )
    return RemoteShardExecutor(
        session_factory=_default_session_factory,
        adapter=adapter,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    executor_factory: Optional[Callable[[RemoteInputs], Any]] = None,
) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-config", required=True)
    parser.add_argument("--resolved-workload", required=True)
    parser.add_argument("--shard-plan", required=True)
    parser.add_argument("--shard-id", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    loaded = load_remote_inputs(
        resolved_config=args.resolved_config,
        resolved_workload=args.resolved_workload,
        shard_plan=args.shard_plan,
        shard_id=args.shard_id,
    )
    output_dir = args.output_dir
    if output_dir is None:
        undeclared = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if not undeclared:
            raise ConfigurationError(
                "AOne worker requires --output-dir or TEST_UNDECLARED_OUTPUTS_DIR"
            )
        output_dir = str(Path(undeclared) / f"perf_test_{loaded.shard_id}")
    factory = _default_executor if executor_factory is None else executor_factory
    result = factory(loaded).execute(loaded, Path(output_dir))
    print(
        json.dumps(
            {
                "run_id": loaded.run_id,
                "shard_id": loaded.shard_id,
                "case_ids": loaded.owned_case_ids,
                "output_dir": output_dir,
                "exit_code": result.exit_code,
            },
            sort_keys=True,
        )
    )
    return int(result.exit_code)

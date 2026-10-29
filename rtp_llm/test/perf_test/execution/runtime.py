"""Concrete engine launch and HTTP request adapters for unified local runs."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import requests

from rtp_llm.test.perf_test.errors import PlanningError
from rtp_llm.test.perf_test.plan import ExecutionPlan, LocalResourcePlan, thaw


@dataclass(frozen=True)
class EngineLaunchSpec:
    argv: Tuple[str, ...]
    environment: Dict[str, str]
    device_ids: Tuple[int, ...]
    checkpoint: str
    model_type: str
    tokenizer: str


def _option_value(value: Any) -> str:
    if type(value) is bool:
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def build_engine_launch_spec(plan: ExecutionPlan) -> EngineLaunchSpec:
    """Project an immutable local plan onto the model server launch contract."""
    if not isinstance(plan.resource_plan, LocalResourcePlan):
        raise PlanningError("engine launch requires a local resource plan")
    config = thaw(plan.resolved_config)
    model = config["model"]
    engine = config["engine"]
    topology = model["topology"]
    reserved = {
        "model_type": model["type"],
        "checkpoint_path": model["checkpoint"],
        "tokenizer_path": model["tokenizer"],
        "tp_size": topology["tp"],
        "dp_size": topology["dp"],
        "pp_size": topology["pp"],
        "ep_size": topology["ep"],
        "world_size": topology["tp"] * topology["dp"] * topology["pp"],
        "max_seq_len": engine["max_seq_len"],
        "max_batch_tokens_size": engine["max_batch_tokens"],
        "concurrency_limit": engine["concurrency_limit"],
    }
    collisions = sorted(set(engine["args"]) & set(reserved))
    if collisions:
        raise PlanningError(
            f"engine.args.{collisions[0]} duplicates a first-class config field"
        )
    options = {**reserved, **engine["args"]}
    argv = tuple(f"--{name}={_option_value(value)}" for name, value in options.items())
    environment = {name: str(value) for name, value in engine["env"].items()}
    environment.update(
        {
            "WORLD_SIZE": str(reserved["world_size"]),
            "USE_BATCH_DECODE_SCHEDULER": "1",
            "FAKE_BALANCE_EXPERT": "1",
        }
    )
    return EngineLaunchSpec(
        argv=argv,
        environment=environment,
        device_ids=plan.resource_plan.devices,
        checkpoint=model["checkpoint"],
        model_type=model["type"],
        tokenizer=model["tokenizer"],
    )


class HttpPrefillClient:
    """Translate model-server HTTP responses to the prefill runner contract."""

    def __init__(
        self,
        port: int,
        *,
        post: Callable[..., Any] = requests.post,
        timeout_seconds: float = 300,
    ):
        self.port = int(port)
        self._post = post
        self._timeout_seconds = timeout_seconds

    def request(
        self,
        *,
        phase: str,
        case_id: str,
        request_index: int,
        run_index: Optional[int],
        prompt: str,
        input_ids: Any,
        output_length: int,
    ) -> Dict[str, Any]:
        del phase, input_ids
        body = {
            "prompt": prompt,
            "generate_config": {
                "max_new_tokens": output_length,
                "min_new_tokens": output_length,
                "force_sp_accept": True,
                "top_k": 1,
            },
        }
        try:
            response = self._post(
                f"http://127.0.0.1:{self.port}",
                json=body,
                headers={"X-DashScope-LoggingConsent": "none"},
                timeout=self._timeout_seconds,
            )
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text[:500]}",
                }
            payload = response.json()
            aux = payload.get("aux_info") or {}
            return {
                "success": True,
                "input_len": int(aux.get("input_len", 0)),
                "output_len": int(aux.get("output_len", 0)),
                "reuse_len": int(aux.get("reuse_len", 0)),
                "latency_ms": float(aux.get("first_token_cost_time", 0.0)),
            }
        except Exception as error:
            return {
                "success": False,
                "error": (
                    f"request {case_id}/{request_index}/{run_index} failed: {error}"
                ),
            }


class MagaEngineHandle:
    """Adapt MagaServerManager to the generic EngineSession handle protocol."""

    def __init__(
        self,
        manager: Any,
        *,
        checkpoint: str,
        model_type: str,
        tokenizer: str,
    ):
        self._manager = manager
        self._checkpoint = checkpoint
        self._model_type = model_type
        self._tokenizer = tokenizer
        self._stopped = False
        self.client = HttpPrefillClient(manager.port)

    @property
    def log_path(self) -> Optional[str]:
        return self._manager.log_file_path

    @property
    def exit_code(self) -> Optional[int]:
        return self._manager.exit_code

    def wait_ready(self, timeout_seconds: int) -> bool:
        return bool(
            self._manager.start_server(
                model_path=self._checkpoint,
                model_type=self._model_type,
                tokenizer_path=self._tokenizer,
                timeout=timeout_seconds,
            )
        )

    def is_alive(self) -> bool:
        return (
            not self._stopped
            and self._manager.server_pid is not None
            and self._manager.exit_code is None
        )

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._manager.stop_server()


def create_maga_handle(plan: ExecutionPlan) -> MagaEngineHandle:
    """Create, but do not start, the real local model server handle."""
    from rtp_llm.test.utils.maga_server_manager import MagaServerManager

    spec = build_engine_launch_spec(plan)
    manager = MagaServerManager(
        env_args=spec.environment,
        device_ids=list(spec.device_ids),
        process_file_name="process.log",
        smoke_args_str=shlex.join(spec.argv),
    )
    return MagaEngineHandle(
        manager,
        checkpoint=spec.checkpoint,
        model_type=spec.model_type,
        tokenizer=spec.tokenizer,
    )

import argparse
import logging
import os
import time
from typing import Dict, List, Optional

import requests

from rtp_llm.test.perf_test.dataset import extract_arg
from rtp_llm.test.utils.maga_server_manager import MagaServerManager


class EngineServer:
    """Engine server lifecycle management (start / stop / port)."""

    def __init__(self, args: argparse.Namespace, remaining_args: List[str]):
        self._args = args
        self._remaining_args = remaining_args
        self._server: Optional[MagaServerManager] = None

    def start(self, max_seq_len: int, max_concurrency: int) -> None:
        """Assemble CLI args and start MagaServerManager."""
        engine_cli = self._build_engine_cli(max_seq_len, max_concurrency)

        env: Dict[str, str] = {
            "USE_BATCH_DECODE_SCHEDULER": "1",
            "FAKE_BALANCE_EXPERT": "1",
            "REUSE_CACHE": "1",
            "BATCH_DECODE_SCHEDULER_WARMUP_TYPE": (
                "0" if self._args.partial in (0, 1) else "1"
            ),
            "TORCH_CUDA_PROFILER_DIR": self._args.result_dir,
        }

        logging.info(f"Starting server with engine CLI: {engine_cli}")
        logging.info(f"remaining_args (raw list): {self._remaining_args}")
        self._server = MagaServerManager(
            env_args=env,
            process_file_name="process.log",
            smoke_args_str=engine_cli,
        )
        if not self._server.start_server():
            self._server.print_process_log()
            raise RuntimeError(
                "Engine server failed to start. Check process.log above for details."
            )

    def stop(self) -> None:
        if self._server is not None:
            self._server.stop_server()

    def set_scheduler_mode(self, *, batch_size: int, mode: str) -> Dict[str, object]:
        """Configure BatchDecodeScheduler before issuing performance requests."""
        if batch_size <= 0:
            raise ValueError("scheduler batch_size must be positive")
        if mode not in ("decode", "prefill"):
            raise ValueError(f"unsupported scheduler mode: {mode}")

        payload = {"batch_size": batch_size, "mode": mode}
        last_error = None
        for attempt in range(1, 21):
            try:
                response = requests.post(
                    f"http://127.0.0.1:{self.port}/update_scheduler_info",
                    json=payload,
                    timeout=60,
                )
                if (
                    response.status_code == 200
                    and response.json().get("status", "ok") == "ok"
                ):
                    logging.info(
                        "scheduler configured before perf requests: payload=%s "
                        "response=%s",
                        payload,
                        response.json(),
                    )
                    return response.json()
                last_error = f"{response.text}, {response.status_code}"
            except Exception as error:
                last_error = repr(error)
            logging.warning(
                "failed to configure scheduler, retrying (%d/20): %s",
                attempt,
                last_error,
            )
            time.sleep(3)
        raise RuntimeError(f"failed to configure scheduler after retries: {last_error}")

    @property
    def port(self) -> int:
        assert self._server is not None, "Server not started"
        return self._server.port

    def _build_engine_cli(self, max_seq_len: int, max_concurrency: int) -> str:
        """Assemble CLI args string for the engine server subprocess.

        remaining_args already contains all engine args (model_type, checkpoint_path,
        tp_size, etc.) that were not consumed by the perf test parser.
        We only put back the three args that the perf test consumed and may modify.
        """
        parts: List[str] = list(self._remaining_args)
        parts.extend(["--dp_size", str(self._args.dp_size)])
        parts.extend(["--max_seq_len", str(max_seq_len)])
        parts.extend(["--concurrency_limit", str(max_concurrency)])
        return " ".join(parts)

    @staticmethod
    def propagate_engine_env(remaining_args: List[str]) -> None:
        """Set engine CLI args as env vars so downstream consumers work automatically.

        create_query() and MagaServerManager.start_server() fall back to env vars
        (MODEL_TYPE, CHECKPOINT_PATH, TOKENIZER_PATH) when explicit params are not
        provided.  Setting them once here avoids threading engine-specific values
        through the perf-test layer.
        """
        for cli_key, env_key in [
            ("model_type", "MODEL_TYPE"),
            ("checkpoint_path", "CHECKPOINT_PATH"),
            ("tokenizer_path", "TOKENIZER_PATH"),
        ]:
            val = extract_arg(remaining_args, cli_key)
            if val and env_key not in os.environ:
                os.environ[env_key] = val
        if "TOKENIZER_PATH" not in os.environ and "CHECKPOINT_PATH" in os.environ:
            os.environ["TOKENIZER_PATH"] = os.environ["CHECKPOINT_PATH"]

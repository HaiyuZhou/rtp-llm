import os

for key, value in os.environ.items():
    print(f"start env {key}={value}")

import json
import logging
import pathlib
import signal
import socket
import sys
import time
from typing import Dict

current_file_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(current_file_path.parent.parent.parent.absolute()))
from rtp_llm.utils.import_util import has_internal_source

if has_internal_source():
    from internal_source.rtp_llm.test.util.set_internal_env import (
        configure_optional_env,
    )

    configure_optional_env()

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.server_config_setup import (
    set_parallelism_config,
    setup_and_configure_server,
)
from rtp_llm.distribute.distributed_server import members_from_test_env
from rtp_llm.server.server_args.server_args import setup_args
from rtp_llm.test.perf_test.multi_node.decode_runtime import (
    build_external_decode_plan,
    execute_master_decode,
    run_external_rank,
)
from rtp_llm.test.perf_test.test_util import create_query
from rtp_llm.test.utils.maga_server_manager import MagaServerManager
from rtp_llm.utils.fuser import fetch_remote_file_to_local

# from uvicorn.loops.uvloop import uvloop_setup
# uvloop_setup()


def wait_master_done(env_dict: Dict[str, str] = {}, world_rank: int = 0) -> None:
    # Get gang_config_string from environment variable or env_dict
    dist_config_str = env_dict.get(
        "GANG_CONFIG_STRING", PyEnvConfigs.distribute_config.gang_config_string
    )
    if not dist_config_str:
        raise RuntimeError("no gang config string, unexpected!")
    dist_members = members_from_test_env(dist_config_str)
    master_member = dist_members[0]
    master_host = master_member.ip
    master_port = master_member.server_port
    while True:
        logging.info(
            f"rank [{world_rank}] waiting for master {master_host}:{master_port} done"
        )
        time.sleep(10)

        # 添加3次重试机制
        retry_count = 0
        max_retries = 5
        connection_failed = True

        while retry_count < max_retries and connection_failed:
            try:
                sock = socket.create_connection(
                    (master_host, master_port), timeout=1000
                )
                sock.close()
                connection_failed = False
            except (socket.error, ConnectionRefusedError) as e:
                retry_count += 1
                if retry_count < max_retries:
                    logging.info(
                        f"rank [{world_rank}] connection attempt {retry_count} failed, retrying... Error: {e}"
                    )
                    time.sleep(5)  # 重试前等待2秒
                else:
                    logging.info(
                        f"rank [{world_rank}] all {max_retries} connection attempts failed, master is done"
                    )
                    break

        if connection_failed:
            break

    logging.info(
        f"rank [{world_rank}] master {master_host}:{master_port} done, this worker exit!"
    )
    return


def script_exit(pgrp_set: bool = False):
    sys.stdout.flush()
    if pgrp_set:
        os.killpg(0, signal.SIGKILL)
        os._exit(0)
    else:
        os._exit(0)


def try_upload_log(log_dir_path: str, upload_path: str):
    import shutil

    if not os.path.exists(log_dir_path) or not os.path.isdir(log_dir_path):
        logging.info(f"{log_dir_path} not exist, skip upload")
        return
    if not os.listdir(log_dir_path):
        logging.info(f"{log_dir_path} is empty, skip upload")
        return

    zip_path = f"{log_dir_path}.zip"
    shutil.make_archive(log_dir_path, "zip", log_dir_path)
    logging.info(f"zip {log_dir_path} to {zip_path}")

    logging.info(f"upload {zip_path} ...")
    os.system(f"osscmd put {zip_path} {upload_path}/{zip_path}")

    # shutil.rmtree(log_dir_path)
    os.remove(zip_path)


if __name__ == "__main__":
    config_path = os.environ.get("PERF_TEST_CONFIG_PATH")
    if not config_path:
        raise RuntimeError("PERF_TEST_CONFIG_PATH is required for multi-node decode")

    os.environ["USE_BATCH_DECODE_SCHEDULER"] = "1"
    os.environ["FAKE_BALANCE_EXPERT"] = "1"

    py_env_configs: PyEnvConfigs = setup_args()
    set_parallelism_config(
        py_env_configs.parallelism_config,
        py_prefill_cp_config=py_env_configs.prefill_cp_config,
    )
    port = py_env_configs.server_config.start_port
    world_rank = py_env_configs.parallelism_config.world_rank
    run_id = os.environ.setdefault("PERF_TEST_RUN_ID", f"decode-{time.time_ns()}")
    log_dir_name = f"TEST_OUTPUT_{run_id}_{world_rank}"
    log_dir_path = os.path.abspath(log_dir_name)
    runtime_profiler_directory = pathlib.Path(log_dir_path)
    parallelism = py_env_configs.parallelism_config
    plan, source = build_external_decode_plan(
        config_path,
        result_directory=str(pathlib.Path(log_dir_path) / "run"),
        actual_topology={
            "tp": parallelism.tp_size,
            "dp": parallelism.dp_size,
            "pp": parallelism.pp_size,
            "ep": parallelism.ep_size,
        },
    )
    setup_and_configure_server(py_env_configs)
    input_len_list = sorted(
        {length for case in plan.cases for length in case.identity["input_lengths"]}
    )
    engine = plan.resolved_config["engine"]
    os.environ["MAX_SEQ_LEN"] = str(engine["max_seq_len"])
    os.environ["MAX_BATCH_TOKENS_SIZE"] = str(engine["max_batch_tokens"])
    os.environ["CONCURRENCY_LIMIT"] = str(engine["concurrency_limit"])
    os.makedirs(log_dir_path, exist_ok=True)
    os.environ["TORCH_CUDA_PROFILER_DIR"] = str(runtime_profiler_directory)
    # for maga server files
    os.environ["TEST_UNDECLARED_OUTPUTS_DIR"] = log_dir_path
    os.environ["MAGA_SERVER_WORK_DIR"] = os.getcwd()

    pgrp_set = False
    try:
        os.setpgrp()
        pgrp_set = True
    except Exception as e:
        logging.info(f"setpgrp error: {e}")

    input_query_dict = {}
    if world_rank == 0:
        raw_tok = py_env_configs.model_args.tokenizer_path
        if not raw_tok:
            raise RuntimeError(
                "tokenizer_path is missing; set model_args.tokenizer_path or ckpt_path"
            )
        tokenizer_path = fetch_remote_file_to_local(
            os.path.expanduser(str(raw_tok).strip())
        )
        input_query_dict = create_query(
            tokenizer_path=tokenizer_path, input_len_list=input_len_list
        )

    server = MagaServerManager(port=str(port))
    try:
        if not server.start_server():
            server.print_process_log()
            raise Exception("server start failed")
        result = run_external_rank(
            world_rank=world_rank,
            wait_for_master=lambda: wait_master_done(world_rank=world_rank),
            execute_master=lambda: execute_master_decode(
                plan,
                source_config=source,
                endpoint=f"http://127.0.0.1:{port}",
                prompt_for_length=lambda length: input_query_dict[length],
                server_log_path=server.log_file_path,
                profiler_directory=runtime_profiler_directory,
            ),
        )
        if result is not None:
            print(
                "PERF_TEST_DECODE_RESULT "
                + json.dumps(
                    {
                        "state": result.state.value,
                        "run_id": run_id,
                        "world_rank": world_rank,
                        "config_fingerprint": plan.config_fingerprint,
                        "workload_fingerprint": plan.workload_fingerprint,
                    }
                )
            )
    finally:
        server.stop_server()
        upload_path = os.environ.get("UPLOAD_OSS_PATH", "")
        if upload_path != "":
            logging.info(f"upload log to {upload_path}")
            try_upload_log(log_dir_path, upload_path)
        script_exit(pgrp_set)

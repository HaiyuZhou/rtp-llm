"""SSH members preserve locked physical GPU allocation and immutable plans."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm.test.perf_test.entry import plan_from_source
from rtp_llm.test.perf_test.execution.distributed.member import run_member
from rtp_llm.test.perf_test.execution.runtime import build_engine_launch_spec
from rtp_llm.test.perf_test.state import RunState
from rtp_llm.test.perf_test.tests.support.paths import PERF_TEST_ROOT


class MemberLaunchTest(unittest.TestCase):
    def make_plan(self, directory):
        source = json.loads(
            (PERF_TEST_ROOT / "examples/prefill_local.json").read_text()
        )
        source["model"]["topology"] = {"tp": 2, "dp": 1, "pp": 1, "ep": 1}
        source["execution"] = {
            "backend": "ssh",
            "hosts": [{"ssh_target": "gpu", "advertise_host": "192.0.2.11"}],
            "gpus_per_machine": 2,
            "source": {
                "build_policy": "reuse",
                "worktree": "/frozen/source",
                "ref": "a" * 40,
            },
        }
        with patch(
            "rtp_llm.test.perf_test.ssh.source.resolve_source",
            side_effect=lambda value, **kwargs: dict(value),
        ):
            plan, frozen_source = plan_from_source(
                source,
                source_dir=directory,
                result_directory=str(directory / "controller"),
            )
        return plan, frozen_source

    def test_launch_spec_has_only_this_members_device_slots(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, _source = self.make_plan(Path(directory))
            self.assertEqual((0, 1), build_engine_launch_spec(plan).device_ids)

    def test_locked_physical_devices_reach_server_without_mutating_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, source = self.make_plan(root)
            original = plan.to_dict()
            descriptor = dict(
                run_id="member-test",
                member_id="0",
                member_index=0,
                members=[dict(hostname="192.0.2.11", port=12333)],
                world_rank=0,
                local_world_size=2,
                devices=[0, 1],
                start_port=12333,
                log_dir=str(root / "logs"),
                result_directory=str(root / "worker"),
            )
            configs = SimpleNamespace(
                parallelism_config=object(),
                prefill_cp_config=object(),
                model_args=SimpleNamespace(tokenizer_path="/m"),
            )
            handle = Mock(tokenizer_path="/m", log_file_path=None)
            handle.start_server.return_value = True
            with patch.dict("os.environ", {"RTP_LLM_GPU_LOCKED_DEVICES": "3,5"}), patch(
                "rtp_llm.server.server_args.server_args.setup_args",
                return_value=configs,
            ), patch(
                "rtp_llm.config.server_config_setup.set_parallelism_config"
            ), patch(
                "rtp_llm.test.perf_test.execution.distributed.rank_group.start_local_rank_group",
                return_value=handle,
            ) as start, patch(
                "rtp_llm.test.perf_test.execution.distributed.prefill.execute_master_prefill",
                return_value=SimpleNamespace(state=RunState.COMPLETED),
            ):
                self.assertEqual(
                    0, run_member(plan, source_config=source, descriptor=descriptor)
                )
                environment = start.call_args.kwargs["engine_environment"]
                self.assertEqual("3,5", environment["CUDA_VISIBLE_DEVICES"])
                self.assertEqual("3,5", environment["HIP_VISIBLE_DEVICES"])
            self.assertEqual(original, plan.to_dict())
            handle.stop_server.assert_called_once()


if __name__ == "__main__":
    unittest.main()

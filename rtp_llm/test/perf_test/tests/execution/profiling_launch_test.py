"""Tests for execution profiling_launch_test."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm.test.perf_test.tests.support.fixtures import (
    prefill_profile_setup as _setup,
)


class DefaultExecutorTest(unittest.TestCase):

    def test_profile_nsys_launch_is_scoped_and_preserves_default_launch(self):
        from dataclasses import replace

        from rtp_llm.test.perf_test.errors import PlanningError
        from rtp_llm.test.perf_test.execution.runtime import create_maga_handle
        from rtp_llm.test.perf_test.plan import freeze, thaw
        from rtp_llm.test.perf_test.tests.support.fixtures import (
            prefill_profile_setup as _setup,
        )

        with tempfile.TemporaryDirectory() as directory:
            plan = _setup(Path(directory) / "run")
            config = thaw(plan.resolved_config)
            config["workload"]["profiling"] = {
                "backend": "nsys",
                "nsys_path": "/opt/nsys",
            }
            config["engine"]["args"]["torch_cuda_profiler_dir"] = "/formal"
            plan = replace(plan, resolved_config=freeze(config))
            manager = Mock()
            manager.return_value.port = 123
            with patch.dict(
                "sys.modules",
                {
                    "rtp_llm.test.utils.maga_server_manager": SimpleNamespace(
                        MagaServerManager=manager
                    )
                },
            ):
                with self.assertRaisesRegex(PlanningError, "torch_cuda_profiler_dir"):
                    create_maga_handle(
                        plan, profile_directory=directory, nsys_session="unique"
                    )
                manager.assert_not_called()
                del config["engine"]["args"]["torch_cuda_profiler_dir"]
                plan = replace(plan, resolved_config=freeze(config))
                handle = create_maga_handle(
                    plan, profile_directory=directory, nsys_session="unique"
                )
                options = manager.call_args.kwargs
                self.assertEqual(
                    [
                        "/opt/nsys",
                        "launch",
                        "--session-new=unique",
                        "--show-output=true",
                        "--trace=cuda,nvtx,osrt",
                        "--wait=all",
                    ],
                    options["process_prefix"],
                )
                self.assertIn(
                    "--torch_cuda_profiler_dir=" + directory, options["smoke_args_str"]
                )
                self.assertNotIn("/formal", options["smoke_args_str"])
                handle.stop()
                manager.return_value.stop_server.assert_called_once()
                create_maga_handle(plan)
                self.assertNotIn("process_prefix", manager.call_args.kwargs)
                self.assertNotIn(
                    "--torch_cuda_profiler_dir",
                    manager.call_args.kwargs["smoke_args_str"],
                )
                self.assertNotIn(
                    "TORCH_CUDA_PROFILER_DIR", manager.call_args.kwargs["env_args"]
                )


if __name__ == "__main__":
    unittest.main()

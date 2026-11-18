"""Runtime buffers must be created after the weight loading domain retires."""

import os
import sys
import types
import unittest
from contextlib import ExitStack
from unittest import mock

import torch

from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors import (
    mega_moe,
    mega_moe_se,
)
from rtp_llm.utils import phase_allocator


class MegaMoePhaseInitializationTest(unittest.TestCase):
    def _exercise(self, executor_class, enabled):
        executor = executor_class.__new__(executor_class)
        torch.nn.Module.__init__(executor)
        executor.cfg = types.SimpleNamespace(
            dim=64,
            moe_inter_dim=32,
            n_routed_experts=8,
            max_tokens_per_rank=16,
            n_activated_experts=2,
            n_shared_experts=1,
        )
        fake_allocator = types.SimpleNamespace(
            state="loading", check_runtime=mock.Mock()
        )
        buf = types.SimpleNamespace(num_max_tokens_per_rank=16)
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.dict(os.environ, {"DSV4_PHASE_ALLOCATOR": str(int(enabled))})
            )
            stack.enter_context(
                mock.patch.dict(sys.modules, {"deep_gemm": types.SimpleNamespace()})
            )
            stack.enter_context(
                mock.patch.object(mega_moe, "_mega_intermediate_size", return_value=32)
            )
            stack.enter_context(
                mock.patch.object(
                    mega_moe_se, "_mega_intermediate_size", return_value=32
                )
            )
            transform = stack.enter_context(
                mock.patch.object(
                    executor,
                    "_transform_routed_weights",
                    return_value=torch.device("cpu"),
                )
            )
            if executor_class is mega_moe_se.MegaMoeSEExecutor:
                stack.enter_context(
                    mock.patch.object(executor, "_setup_shared_expert_weights")
                )
            buffers = []
            for module, suffix in ((mega_moe, ""), (mega_moe_se, "_se")):
                stack.enter_context(
                    mock.patch.object(
                        module, "_get_validated_world_ep_group", return_value=object()
                    )
                )
                buffers.append(
                    stack.enter_context(
                        mock.patch.object(
                            module,
                            "_get_or_create_mega" + suffix + "_buf",
                            return_value=buf,
                        )
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        module,
                        "_get_or_create_mega" + suffix + "_output",
                        side_effect=lambda *args: torch.empty(16, 64),
                    )
                )
            warmup = stack.enter_context(
                mock.patch.object(executor, "_maybe_warmup_jit_once")
            )
            stack.enter_context(
                mock.patch.object(
                    phase_allocator,
                    "get_phase_allocator",
                    return_value=fake_allocator if enabled else None,
                )
            )
            executor.setup_weights({})
            transform.assert_called_once()
            if enabled:
                self.assertEqual(sum(p.call_count for p in buffers), 0)
                warmup.assert_not_called()
                with self.assertRaisesRegex(RuntimeError, "loading"):
                    executor.initialize_runtime()
                fake_allocator.state = "runtime"
                executor.initialize_runtime()
            self.assertTrue(executor._mega_runtime_ready)
            self.assertEqual(sum(p.call_count for p in buffers), 1)
            warmup.assert_called_once()
            executor.initialize_runtime()
            self.assertEqual(sum(p.call_count for p in buffers), 1)
            warmup.assert_called_once()
        return executor

    def test_routed_runtime_is_deferred_until_loading_finishes(self):
        self._exercise(mega_moe.MegaMoeExecutor, True)

    def test_shared_runtime_is_deferred_until_loading_finishes(self):
        self._exercise(mega_moe_se.MegaMoeSEExecutor, True)

    def test_disabled_routed_behavior_stays_eager(self):
        self._exercise(mega_moe.MegaMoeExecutor, False)

    def test_disabled_shared_behavior_stays_eager(self):
        self._exercise(mega_moe_se.MegaMoeSEExecutor, False)


if __name__ == "__main__":
    unittest.main()

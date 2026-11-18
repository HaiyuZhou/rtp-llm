"""AUTO must not mix a WORLD-collective loader with independent loading."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.model_loader.load_config import LoadMethod
from rtp_llm.model_loader.loader import ModelLoader


class PhaseLoadMethodTest(unittest.TestCase):
    def select(
        self, local_fast, peer_fast, *, enabled=True, initialized=True, backend="nccl"
    ):
        loader = object.__new__(ModelLoader)
        loader._load_method = LoadMethod.AUTO
        loader._load_config = SimpleNamespace(
            database=SimpleNamespace(
                is_safetensor=True, get_pretrain_tensor_names=lambda: ["weight"]
            )
        )
        loader._supports_fastsafetensors_iteration = lambda: True
        loader._choose_weight_convert_device = lambda device: device
        loader._is_memory_enough_for_fastsafetensor = lambda: local_fast
        loader._load_from_fastsafetensor = MagicMock(return_value="fast")
        loader._load_from_scratch = MagicMock(return_value="scratch")
        votes = []

        def reduce(vote, op):
            votes.append(int(vote.item()))
            vote.fill_(int(local_fast and peer_fast))

        # Keep the real tensor/reduction value contract, without requiring CUDA.
        real_tensor = torch.tensor

        def tensor(data, **kwargs):
            self.assertEqual(kwargs["device"], "cuda:0" if backend == "nccl" else "cpu")
            return real_tensor(data, dtype=kwargs["dtype"])

        with patch(
            "rtp_llm.utils.phase_allocator.phase_allocator_enabled",
            return_value=enabled,
        ), patch("rtp_llm.model_loader.loader.has_module", return_value=True), patch(
            "torch.distributed.is_initialized", return_value=initialized
        ), patch(
            "torch.distributed.get_backend", return_value=backend
        ), patch(
            "torch.distributed.all_reduce", side_effect=reduce
        ) as collective, patch(
            "torch.tensor", side_effect=tensor
        ):
            result = loader._load_weight("cuda:0")
        return result, votes, collective

    def test_peer_memory_veto_selects_scratch_before_any_loader_io(self):
        result, votes, collective = self.select(True, False)
        self.assertEqual(result, "scratch")
        self.assertEqual(votes, [1])
        collective.assert_called_once()
        self.assertEqual(
            collective.call_args.kwargs["op"], torch.distributed.ReduceOp.MIN
        )

    def test_local_veto_still_participates_in_consensus(self):
        result, votes, _ = self.select(False, True)
        self.assertEqual(result, "scratch")
        self.assertEqual(votes, [0])

    def test_all_eligible_keep_fast_loading(self):
        result, votes, _ = self.select(True, True)
        self.assertEqual((result, votes), ("fast", [1]))

    def test_gloo_consensus_uses_cpu(self):
        result, votes, _ = self.select(True, False, backend="gloo")
        self.assertEqual((result, votes), ("scratch", [1]))

    def test_disabled_phase_also_coordinates_loader(self):
        result, votes, collective = self.select(True, False, enabled=False)
        self.assertEqual((result, votes), ("scratch", [1]))
        collective.assert_called_once()

    def test_single_process_keeps_existing_selection(self):
        result, votes, collective = self.select(True, False, initialized=False)
        self.assertEqual((result, votes), ("fast", []))
        collective.assert_not_called()

    def test_explicit_scratch_does_not_require_world_participation(self):
        loader = object.__new__(ModelLoader)
        loader._load_method = LoadMethod.SCRATCH
        loader._supports_fastsafetensors_iteration = lambda: True
        loader._load_from_scratch = MagicMock(return_value="scratch")
        with patch("torch.distributed.all_reduce") as collective:
            self.assertEqual(loader._load_weight("cuda:0"), "scratch")
        collective.assert_not_called()

    def test_explicit_fast_preserves_existing_loader_contract(self):
        loader = object.__new__(ModelLoader)
        loader._load_method = LoadMethod.FASTSAFETENSORS
        loader._supports_fastsafetensors_iteration = lambda: True
        loader._load_from_fastsafetensor = MagicMock(return_value="fast")
        with patch("torch.distributed.all_reduce") as collective:
            self.assertEqual(loader._load_weight("cuda:0"), "fast")
        collective.assert_not_called()


if __name__ == "__main__":
    unittest.main()

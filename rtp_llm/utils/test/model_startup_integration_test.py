"""Exercise the real shared loader/model entries with lightweight model hooks."""

import os
import types
import unittest
from unittest import mock

import torch

from rtp_llm.models.base_model import BaseModel
from rtp_llm.models_py.model_desc.deepseek_v4_model import DeepSeekV4Model
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.utils.model_startup import ModelStartupLifecycle
from rtp_llm.utils.test.model_startup_test import Allocator, Model


class SharedModel(GptModelBase, Model):
    def __init__(self, events):
        torch.nn.Module.__init__(self)
        Model.__init__(self, events)

    _bind_init_resources = Model._bind_init_resources
    startup_tensors = Model.startup_tensors
    startup_kv_tensors = Model.startup_kv_tensors


class ModelStartupIntegrationTest(unittest.TestCase):
    def test_target_verify_is_a_decode_audit_even_with_prefill_shaped_inputs(self):
        model = GptModelBase.__new__(GptModelBase)
        torch.nn.Module.__init__(model)
        model.kv_cache = object()
        attn = types.SimpleNamespace(is_prefill=True, is_target_verify=True)
        inputs = types.SimpleNamespace(attention_inputs=attn)
        with mock.patch(
            "rtp_llm.models_py.model_desc.module_base.get_attention_inputs_value",
            return_value=attn,
        ):
            self.assertEqual(model.startup_forward_kind(inputs), "decode")

    def test_empty_kv_group_is_rejected_at_binding_checkpoint(self):
        model = GptModelBase.__new__(GptModelBase)
        torch.nn.Module.__init__(model)
        group = types.SimpleNamespace(
            tag="swa", kv_cache_base=torch.empty(0), kv_scale_base=None
        )
        model.kv_cache = types.SimpleNamespace(
            layer_count=1, get_layer_cache_groups=lambda _: [group]
        )
        with self.assertRaisesRegex(RuntimeError, "KV.*empty"):
            list(model.startup_kv_tensors())

    def test_missing_kv_groups_are_rejected(self):
        model = GptModelBase.__new__(GptModelBase)
        torch.nn.Module.__init__(model)
        model.kv_cache = types.SimpleNamespace(
            layer_count=1, get_layer_cache_groups=lambda _: []
        )
        with self.assertRaisesRegex(RuntimeError, "KV.*groups"):
            list(model.startup_kv_tensors())

    def test_common_model_initialize_drives_all_hooks(self):
        events = []
        model = SharedModel(events)
        model._startup_lifecycle = lifecycle = ModelStartupLifecycle(Allocator(events))
        with lifecycle.loading_weights():
            pass
        model.initialize(types.SimpleNamespace(kv_cache=None))
        self.assertEqual(lifecycle.state, "ready")
        self.assertEqual(events.count("convert"), 1)
        model.initialize(types.SimpleNamespace(kv_cache=object()))
        self.assertEqual(events.count("convert"), 1)

    def test_dsv4_initialize_delegates_to_shared_coordinator(self):
        model = DeepSeekV4Model.__new__(DeepSeekV4Model)
        torch.nn.Module.__init__(model)
        model._startup_lifecycle = types.SimpleNamespace(
            initialize=mock.Mock(return_value=True)
        )
        with mock.patch.object(
            model, "_initialize_impl", side_effect=AssertionError("legacy path")
        ):
            self.assertTrue(model.initialize(types.SimpleNamespace(kv_cache=None)))
        model._startup_lifecycle.initialize.assert_called_once()

    def test_dsv4_direct_forward_cannot_skip_shared_guard(self):
        model = DeepSeekV4Model.__new__(DeepSeekV4Model)
        torch.nn.Module.__init__(model)
        model._startup_lifecycle = types.SimpleNamespace(
            forward=mock.Mock(return_value="guarded")
        )
        self.assertEqual(model.forward(object()), "guarded")
        model._startup_lifecycle.forward.assert_called_once()

    def test_common_loader_covers_python_model_construction(self):
        model = BaseModel.__new__(BaseModel)
        model.parallelism_config = types.SimpleNamespace(pp_size=1)
        model.hw_kernel_config = types.SimpleNamespace(enable_cuda_graph=False)
        model.model_config = types.SimpleNamespace(ckpt_path="test")
        model._init_custom_module = lambda: None
        model.create_model_loader = lambda: types.SimpleNamespace(_py_eplb=None)
        model._get_device_str = lambda: "cuda:0"
        model._validate_phase_allocator = lambda: None
        model._weight_alias_names = set()
        model.device = "cuda:0"
        model.weight = object()
        events = []
        allocator = Allocator(events)
        model._load = lambda _: events.append("load")

        def create():
            self.assertEqual(events[-1], "load")
            self.assertNotIn("load_weights.exit", events)
            model.py_model = SharedModel(events)

        model._create_python_model = create
        with mock.patch.dict(os.environ, {"DSV4_PHASE_ALLOCATOR": "1"}), mock.patch(
            "rtp_llm.utils.phase_allocator.get_phase_allocator", return_value=allocator
        ), mock.patch("rtp_llm.model_loader.weight_manager.WeightManager"):
            model.load()
        self.assertIsInstance(model.py_model._startup_lifecycle, ModelStartupLifecycle)
        self.assertEqual(model.py_model._startup_lifecycle.state, "loaded")


if __name__ == "__main__":
    unittest.main()

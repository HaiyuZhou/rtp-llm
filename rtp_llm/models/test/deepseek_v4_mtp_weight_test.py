"""MTP quantized globals must have a single producer for checkpoint scales."""

from types import SimpleNamespace
from unittest import TestCase, main, mock

import torch

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.model_loader.per_block_fp8_quant_weight import V4PerBlockFp8Weight
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.deepseek_v4 import DeepSeekV4MtpWeight, DeepSeekV4Weight
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity


class MtpProjectionScaleTest(TestCase):
    def info(self, reverse=False):
        weights = []
        for prefix, kernel, scale in (
            ("mtp.0.e_proj", W.v4_mtp_e_proj_w, W.v4_mtp_e_proj_s),
            ("mtp.0.h_proj", W.v4_mtp_h_proj_w, W.v4_mtp_h_proj_s),
        ):
            source = AtomicWeight(
                kernel, [CkptWeightInfo(prefix + ".weight", identity)], identity
            )
            weights.extend(
                [
                    V4PerBlockFp8Weight(
                        source, Fp8BlockWiseQuantConfig(is_quanted=True), name=kernel
                    ),
                    AtomicWeight(
                        scale, [CkptWeightInfo(prefix + ".scale", identity)], identity
                    ),
                ]
            )
        if reverse:
            weights.reverse()
        info = ModelWeightInfo(weights=weights, layer_weights=[])
        descriptor = DeepSeekV4MtpWeight.__new__(DeepSeekV4MtpWeight)
        with mock.patch.object(DeepSeekV4Weight, "get_weight_info", return_value=info):
            return descriptor.get_weight_info()

    def test_quantized_scale_has_one_producer(self):
        info = self.info()
        for scale in (W.v4_mtp_e_proj_s, W.v4_mtp_h_proj_s):
            producers = [
                w
                for w in info.weights
                if w.name == scale or scale in getattr(w, "sub_weights", {})
            ]
            self.assertEqual(len(producers), 1, scale)
            self.assertIsInstance(producers[0], V4PerBlockFp8Weight)

    def test_fast_checkpoint_scale_maps_to_quantized_owner(self):
        loader = ModelLoader.__new__(ModelLoader)
        loader._model_weights_info = self.info()
        loader._misc_weights_info = []
        loader._load_config = SimpleNamespace(database=object(), layer_range=range(0))
        loader._maybe_skip_weight = lambda _: False
        mapping, _ = loader._generate_weight_info()
        for name in ("mtp.0.e_proj.scale", "mtp.0.h_proj.scale"):
            self.assertIsInstance(mapping[name].weight, V4PerBlockFp8Weight)

    def test_scratch_loading_preserves_ue8m0_bytes_in_either_order(self):
        scale = torch.tensor([[0.5]], dtype=torch.float32).to(torch.float8_e8m0fnu)
        raw = {
            prefix + suffix: value
            for prefix in ("mtp.0.e_proj", "mtp.0.h_proj")
            for suffix, value in (
                (".weight", torch.full((128, 128), 0.25).to(torch.float8_e4m3fn)),
                (".scale", scale),
            )
        }
        source = SimpleNamespace(load_tensor=lambda name, dtype: [raw[name].to(dtype)])
        config = SimpleNamespace(
            compute_dtype=torch.bfloat16,
            merge_lora=False,
            tp_size=1,
            tp_rank=0,
            ep_size=1,
            ep_rank=0,
            dp_size=1,
            dp_rank=0,
            ffn_tp_size=1,
            ffn_tp_rank=0,
            hidden_size=128,
            head_num=1,
            head_num_kv=1,
            size_per_head=128,
            moe_pure_tp_mode=False,
            bit=16,
            use_swizzleA=False,
            exported_device=SimpleNamespace(
                maybe_rewrite_weight_by_key=lambda name, tensor, **kwargs: tensor
            ),
        )
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                loaded = {}
                for weight in self.info(reverse).weights:
                    loaded.update(weight.load(source, None, "cpu", config))
                for key in (W.v4_mtp_e_proj_s, W.v4_mtp_h_proj_s):
                    self.assertEqual(loaded[key].dtype, torch.float8_e8m0fnu)
                    self.assertTrue(
                        torch.equal(
                            loaded[key].view(torch.uint8), scale.view(torch.uint8)
                        )
                    )

    def test_standalone_scales_without_quantized_owner_are_preserved(self):
        standalone = AtomicWeight(W.v4_mtp_e_proj_s, [], identity)
        info = ModelWeightInfo(weights=[standalone], layer_weights=[])
        descriptor = DeepSeekV4MtpWeight.__new__(DeepSeekV4MtpWeight)
        with mock.patch.object(DeepSeekV4Weight, "get_weight_info", return_value=info):
            self.assertEqual(descriptor.get_weight_info().weights, [standalone])


if __name__ == "__main__":
    main()

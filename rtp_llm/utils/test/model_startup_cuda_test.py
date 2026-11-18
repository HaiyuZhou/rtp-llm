"""Actual native-pool retirement across a target/draft startup group."""

import gc
import types
import unittest

import torch

from rtp_llm.utils.model_startup import (
    ModelStartupLifecycle,
    prepare_engine_models,
    register_engine_models,
)
from rtp_llm.utils.phase_allocator import PhaseAllocator
from rtp_llm.utils.test.phase_allocator_test import MIB, RUNTIME_SETTINGS, owner


class CudaModel:
    def __init__(self, allocator):
        self._startup_lifecycle = ModelStartupLifecycle(allocator)
        self.runtime = None
        self.conversions = 0
        with self._startup_lifecycle.loading_weights():
            self.weight = torch.full((21 * MIB,), 7, dtype=torch.uint8, device="cuda:0")

    def _bind_init_resources(self, resources):
        self.kv_cache = resources.kv_cache

    def _prepare_startup(self, resources):
        pass

    def _materialize_startup_weights(self):
        self.conversions += 1
        self.weight = self.weight.clone()

    def _initialize_startup_runtime(self):
        self.runtime = torch.empty(21 * MIB, dtype=torch.uint8, device="cuda:0")

    def startup_tensors(self):
        yield "weight", self.weight
        if self.runtime is not None:
            yield "runtime", self.runtime

    def startup_kv_tensors(self):
        return iter(())


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class ModelStartupCudaTest(unittest.TestCase):
    def test_two_models_keep_weights_in_one_pool_and_runtime_in_default(self):
        torch.cuda.set_device(0)
        original = torch.cuda.memory._snapshot()["allocator_settings"][
            "PYTORCH_CUDA_ALLOC_CONF"
        ]
        allocator = None
        models = []
        try:
            torch._C._accelerator_setAllocatorSettings(RUNTIME_SETTINGS)
            allocator = PhaseAllocator("cuda:0")
            models = [CudaModel(allocator), CudaModel(allocator)]
            register_engine_models(models)
            resources = types.SimpleNamespace(kv_cache=None)
            prepare_engine_models(models, resources)
            pointers = [m.weight.data_ptr() for m in models]
            self.assertEqual(allocator.state, "runtime")
            self.assertEqual({owner(m.weight) for m in models}, {allocator.pool_id})
            self.assertEqual({owner(m.runtime) for m in models}, {(0, 0)})
            for m in models:
                self.assertEqual(int(m.weight[0]), 7)
                self.assertEqual(int(m.weight[-1]), 7)
                m._startup_lifecycle.initialize(m, resources)
                self.assertEqual(m.conversions, 1)
            self.assertEqual([m.weight.data_ptr() for m in models], pointers)
            allocator.check_runtime()
        finally:
            models.clear()
            m = None
            if allocator is not None:
                allocator.close()
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch._C._accelerator_setAllocatorSettings(original)


if __name__ == "__main__":
    unittest.main()

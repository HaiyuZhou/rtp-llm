"""Lifecycle and actual CUDA allocation-domain checks for phase allocation."""

import gc
import os
import threading
import unittest
from unittest import mock

import torch

from rtp_llm.utils.phase_allocator import PhaseAllocator, get_phase_allocator

MIB = 1024**2
RUNTIME_SETTINGS = "backend:native,expandable_segments:True,large_segment_size_mb:1024"


def owner(tensor):
    ptr = tensor.data_ptr()
    for segment in torch.cuda.memory_snapshot():
        if segment["address"] <= ptr < segment["address"] + segment["total_size"]:
            return tuple(segment["segment_pool_id"])
    raise AssertionError("Tensor has no allocator segment")


class DisabledPhaseAllocatorTest(unittest.TestCase):
    def test_disabled_does_not_touch_cuda(self):
        with mock.patch.dict(os.environ, {"DSV4_PHASE_ALLOCATOR": "0"}), mock.patch(
            "torch.cuda.current_device", side_effect=AssertionError("CUDA touched")
        ):
            self.assertIsNone(get_phase_allocator("cuda:0"))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class PhaseAllocatorCudaTest(unittest.TestCase):
    def setUp(self):
        torch.cuda.set_device(0)
        self.original = torch.cuda.memory._snapshot()["allocator_settings"][
            "PYTORCH_CUDA_ALLOC_CONF"
        ]
        torch._C._accelerator_setAllocatorSettings(RUNTIME_SETTINGS)
        self.allocator = None

    def tearDown(self):
        if self.allocator is not None:
            self.allocator.close()
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch._C._accelerator_setAllocatorSettings(self.original)

    def test_worker_threads_and_reentered_scope_share_loading_pool(self):
        self.allocator = allocator = PhaseAllocator("cuda:0")
        tensors = []

        def allocate():
            torch.cuda.set_device(0)
            tensors.append(torch.full((21 * MIB,), 7, dtype=torch.uint8, device="cuda"))

        with allocator.loading("loader"):
            allocate()
            thread = threading.Thread(target=allocate)
            thread.start()
            thread.join()
        with allocator.loading("conversion"):
            allocate()
        self.assertEqual({owner(t) for t in tensors}, {allocator.pool_id})
        pointers = [t.data_ptr() for t in tensors]
        allocator.finish_loading()
        allocator.finish_loading()  # Executor rebuild is harmless.
        self.assertEqual(allocator.state, "runtime")
        self.assertEqual([t.data_ptr() for t in tensors], pointers)
        self.assertTrue(all(t[0].item() == 7 and t[-1].item() == 7 for t in tensors))
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            runtime = torch.empty(257 * MIB, dtype=torch.uint8, device="cuda")
        stream.synchronize()
        self.assertEqual(owner(runtime), (0, 0))
        allocator.check_runtime()
        runtime_segment = next(
            s
            for s in torch.cuda.memory_snapshot()
            if s["address"] <= runtime.data_ptr() < s["address"] + s["total_size"]
        )
        self.assertTrue(runtime_segment["is_expandable"])
        self.assertEqual(runtime_segment["total_size"] % (1024 * MIB), 0)
        with self.assertRaisesRegex(RuntimeError, "runtime"):
            with allocator.loading("late_load"):
                pass
        tensors.clear()
        del runtime

    def test_exception_restores_runtime_and_releases_global_route(self):
        self.allocator = allocator = PhaseAllocator("cuda:0")
        with self.assertRaisesRegex(ValueError, "conversion failed"):
            with allocator.loading("conversion"):
                tensor = torch.empty(21 * MIB, dtype=torch.uint8, device="cuda")
                raise ValueError("conversion failed")
        self.assertEqual(allocator.state, "failed")
        self.assertEqual(
            torch.cuda.memory._snapshot()["allocator_settings"][
                "PYTORCH_CUDA_ALLOC_CONF"
            ],
            RUNTIME_SETTINGS,
        )
        outside = torch.empty(21 * MIB, dtype=torch.uint8, device="cuda")
        self.assertEqual(owner(outside), (0, 0))
        with self.assertRaisesRegex(RuntimeError, "failed"):
            allocator.finish_loading()
        del tensor, outside

    def test_nested_scope_is_rejected_without_losing_outer_route(self):
        self.allocator = allocator = PhaseAllocator("cuda:0")
        with allocator.loading("outer"):
            with self.assertRaisesRegex(RuntimeError, "active"):
                with allocator.loading("inner"):
                    pass
            tensor = torch.empty(21 * MIB, dtype=torch.uint8, device="cuda")
            self.assertEqual(owner(tensor), allocator.pool_id)
        allocator.finish_loading()
        del tensor

    def test_requires_runtime_1024_before_loading(self):
        torch._C._accelerator_setAllocatorSettings(
            "backend:native,expandable_segments:True,large_segment_size_mb:20"
        )
        with self.assertRaisesRegex(ValueError, "1024"):
            PhaseAllocator("cuda:0")

    def test_preserves_other_allocator_options(self):
        settings = RUNTIME_SETTINGS + ",garbage_collection_threshold:0.8"
        torch._C._accelerator_setAllocatorSettings(settings)
        self.allocator = allocator = PhaseAllocator("cuda:0")
        with allocator.loading("loader"):
            self.assertEqual(
                torch.cuda.memory._snapshot()["allocator_settings"][
                    "garbage_collection_threshold"
                ],
                0.8,
            )
        self.assertEqual(
            torch.cuda.memory._snapshot()["allocator_settings"][
                "PYTORCH_CUDA_ALLOC_CONF"
            ],
            settings,
        )
        allocator.finish_loading()

    def test_retired_loading_domain_rejects_new_live_allocation(self):
        self.allocator = allocator = PhaseAllocator("cuda:0")
        with allocator.loading("loader"):
            weight = torch.empty(21 * MIB, dtype=torch.uint8, device="cuda")
        # Simulate another component incorrectly retaining the loading handle.
        held_pool = allocator._pool
        allocator.finish_loading()
        torch._C._cuda_beginAllocateToPool(0, allocator.pool_id)
        try:
            escaped = torch.empty(21 * MIB, dtype=torch.uint8, device="cuda")
        finally:
            torch._C._cuda_endAllocateToPool(0, allocator.pool_id)
            torch._C._cuda_releasePool(0, allocator.pool_id)
        with self.assertRaisesRegex(RuntimeError, "retired loading"):
            allocator.check_runtime()
        del weight, escaped, held_pool

    def test_resource_inventory_checks_domains_without_retaining_tensors(self):
        import weakref

        self.allocator = allocator = PhaseAllocator("cuda:0")
        with allocator.loading("loader"):
            weight = torch.empty(21 * MIB, dtype=torch.uint8, device="cuda")
        allocator.finish_loading()
        runtime = torch.empty(21 * MIB, dtype=torch.uint8, device="cuda")
        inventory = allocator.inspect_tensors([("weight", weight)], allocator.pool_id)
        self.assertEqual(inventory["weight"]["pool_id"], allocator.pool_id)
        self.assertEqual(
            inventory["weight"]["address"], weight.untyped_storage().data_ptr()
        )
        with self.assertRaisesRegex(RuntimeError, "weight.*domain"):
            allocator.inspect_tensors([("weight", runtime)], allocator.pool_id)
        with self.assertRaisesRegex(RuntimeError, "runtime.*domain"):
            allocator.inspect_tensors([("runtime", weight)], (0, 0))
        allocator.inspect_tensors([("runtime", runtime)], (0, 0))
        allocator.check_runtime()  # Ordinary runtime growth is valid.
        ref = weakref.ref(weight)
        del weight, runtime
        self.assertIsNone(ref())

    def test_runtime_settings_tampering_is_rejected_after_retirement(self):
        self.allocator = allocator = PhaseAllocator("cuda:0")
        allocator.finish_loading()
        torch._C._accelerator_setAllocatorSettings(allocator.loading_settings)
        with self.assertRaisesRegex(RuntimeError, "loading settings"):
            allocator.check_runtime()

    def test_rejects_preexisting_large_segment_even_if_size_is_1024_multiple(self):
        torch._C._accelerator_setAllocatorSettings(
            "backend:native,expandable_segments:True,large_segment_size_mb:20"
        )
        stale = torch.empty(5101 * MIB, dtype=torch.uint8, device="cuda")
        segment = next(
            s
            for s in torch.cuda.memory_snapshot()
            if s["address"] <= stale.data_ptr() < s["address"] + s["total_size"]
        )
        self.assertEqual(segment["total_size"], 5120 * MIB)
        torch._C._accelerator_setAllocatorSettings(RUNTIME_SETTINGS)
        with self.assertRaisesRegex(RuntimeError, "before.*large|preexisting"):
            PhaseAllocator("cuda:0")
        del stale

    def test_exit_diagnostic_failure_retires_loading_pool(self):
        self.allocator = allocator = PhaseAllocator("cuda:0")
        original_record = allocator.record

        def record(stage):
            if stage == "conversion.exit":
                raise OSError("trace filesystem full")
            return original_record(stage)

        with mock.patch.object(allocator, "record", side_effect=record):
            with self.assertRaisesRegex(OSError, "filesystem full"):
                with allocator.loading("conversion"):
                    tensor = torch.empty(21 * MIB, dtype=torch.uint8, device="cuda")
        self.assertEqual(allocator.state, "failed")
        self.assertIsNone(allocator._pool)
        self.assertEqual(
            torch.cuda.memory._snapshot()["allocator_settings"][
                "PYTORCH_CUDA_ALLOC_CONF"
            ],
            RUNTIME_SETTINGS,
        )
        outside = torch.empty(21 * MIB, dtype=torch.uint8, device="cuda")
        self.assertEqual(owner(outside), (0, 0))
        del tensor, outside

    def test_finalization_diagnostic_failure_retires_loading_pool(self):
        self.allocator = allocator = PhaseAllocator("cuda:0")
        with allocator.loading("loader"):
            tensor = torch.empty(21 * MIB, dtype=torch.uint8, device="cuda")
        with mock.patch.object(
            allocator, "record", side_effect=OSError("trace unavailable")
        ):
            with self.assertRaisesRegex(OSError, "trace unavailable"):
                allocator.finish_loading()
        self.assertEqual(allocator.state, "failed")
        self.assertIsNone(allocator._pool)
        del tensor

    def test_exit_sync_failure_restores_route_and_original_body_error(self):
        self.allocator = allocator = PhaseAllocator("cuda:0")
        synchronize = torch.cuda.synchronize
        count = 0

        def fail_exit_once(*args, **kwargs):
            nonlocal count
            count += 1
            # Initial scope sync, entry diagnostics sync, then scope exit.
            if count == 3:
                raise RuntimeError("injected exit sync error")
            return synchronize(*args, **kwargs)

        with mock.patch("torch.cuda.synchronize", side_effect=fail_exit_once):
            with self.assertRaisesRegex(ValueError, "original body failure"):
                with allocator.loading("conversion"):
                    raise ValueError("original body failure")
        self.assertEqual(allocator.state, "failed")
        self.assertIsNone(allocator._pool)
        self.assertEqual(
            torch.cuda.memory._snapshot()["allocator_settings"][
                "PYTORCH_CUDA_ALLOC_CONF"
            ],
            RUNTIME_SETTINGS,
        )


if __name__ == "__main__":
    unittest.main()

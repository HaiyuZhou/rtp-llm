"""Opt-in native CUDA loading pool for DSV4 startup.

The process default domain is always the runtime domain. During a serialized
loading scope, the native all-stream route also covers loader producer threads
and C++ callbacks. This deliberately uses exact-wheel private Torch APIs; it
fails closed when those APIs or the required initial configuration are absent.
"""

import gc
import json
import logging
import os
import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import torch

_MIB = 1024**2
_INSTANCE = None
_INSTANCE_LOCK = threading.Lock()


def phase_allocator_enabled() -> bool:
    return os.environ.get("DSV4_PHASE_ALLOCATOR", "0") == "1"


def get_phase_allocator(device):
    if not phase_allocator_enabled():
        return None
    global _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is None:
            _INSTANCE = PhaseAllocator(device)
        elif _INSTANCE.device != torch.device(device).index:
            raise RuntimeError(
                "DSV4 phase allocation supports one CUDA device per process"
            )
        return _INSTANCE


class PhaseAllocator:
    """Own one loading pool until final weights are ready, then retire it.

    Loading scopes may be reentered sequentially (loader, then conversion).
    Runtime is a one-way transition; executor reconstruction must not reload.
    Concurrent or nested scopes are rejected because allocator settings and
    the all-stream route are process-wide.
    """

    def __init__(self, device):
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("DSV4 phase allocation requires CUDA")
        self.device = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        torch.cuda.set_device(self.device)
        for name in (
            "_cuda_beginAllocateToPool",
            "_cuda_endAllocateToPool",
            "_cuda_releasePool",
            "_accelerator_setAllocatorSettings",
        ):
            if not hasattr(torch._C, name):
                raise RuntimeError(f"DSV4 phase allocation requires Torch API {name}")
        if torch.cuda.memory.get_allocator_backend() != "native":
            raise ValueError("DSV4 phase allocation requires the native allocator")
        settings = torch.cuda.memory._snapshot()["allocator_settings"]
        self.runtime_settings = settings["PYTORCH_CUDA_ALLOC_CONF"]
        pattern = r"(?<![\w])large_segment_size_mb\s*:\s*1024(?=,|$)"
        if not settings["expandable_segments"] or not re.search(
            pattern, self.runtime_settings
        ):
            raise ValueError(
                "Start DSV4 with expandable_segments:True,large_segment_size_mb:1024 "
                "before enabling DSV4_PHASE_ALLOCATOR"
            )
        self.loading_settings = re.sub(
            pattern, "large_segment_size_mb:20", self.runtime_settings
        )
        # Mapped size does not reveal an existing segment's physical page
        # granularity (e.g. 256 * 20 MiB is also 5 * 1024 MiB). Require a clean
        # default large domain before the first loading scope instead.
        self._check_default_segments(require_empty=True)
        self._lock = threading.Lock()
        self._active = False
        self._sealed_loading_blocks = None
        self.state = "loading"
        self._pool = torch.cuda.MemPool()
        self.pool_id = self._pool.id
        self.record("created")

    def _check_default_segments(self, require_empty=False):
        for segment in torch.cuda.memory_snapshot():
            if (
                segment["device"] == self.device
                and tuple(segment["segment_pool_id"]) == (0, 0)
                and segment["segment_type"] == "large"
            ):
                if require_empty:
                    raise RuntimeError(
                        "Enable DSV4 phase allocation before the first default large allocation; "
                        "preexisting segment granularity cannot be verified"
                    )
                if not segment["is_expandable"] or segment["total_size"] % (
                    1024 * _MIB
                ):
                    raise RuntimeError(
                        "Default runtime domain contains a segment incompatible with 1024 MiB"
                    )

    @contextmanager
    def loading(self, stage: str):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("A phase allocator loading scope is already active")
        try:
            if self.state != "loading":
                raise RuntimeError(
                    f"Cannot load weights in phase allocator state {self.state}"
                )
            torch.cuda.set_device(self.device)
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Weight loading cannot run during CUDA Graph capture"
                )
            routed = False
            failure = None
            try:
                torch.cuda.synchronize(self.device)
                # Route first, change settings second. Reverse that order on
                # exit so a default segment cannot be created with 20 MiB pages.
                torch._C._cuda_beginAllocateToPool(self.device, self.pool_id)
                routed = True
                self._active = True
                torch._C._accelerator_setAllocatorSettings(self.loading_settings)
                self.record(stage + ".enter")
                yield
            except BaseException as exc:
                failure = exc
            finally:
                if routed:
                    # Attempt every cleanup operation even if synchronization
                    # fails, and never mask the original loading exception.
                    cleanup = (
                        lambda: torch.cuda.synchronize(self.device),
                        lambda: torch._C._accelerator_setAllocatorSettings(
                            self.runtime_settings
                        ),
                        lambda: torch._C._cuda_endAllocateToPool(
                            self.device, self.pool_id
                        ),
                        lambda: torch._C._cuda_releasePool(self.device, self.pool_id),
                    )
                    for operation in cleanup:
                        try:
                            operation()
                        except BaseException as exc:
                            if failure is None:
                                failure = exc
                            else:
                                logging.exception("Phase allocator cleanup also failed")
                    self._active = False
            if failure is None:
                try:
                    self.record(stage + ".exit")
                except BaseException as exc:
                    failure = exc
            if failure is not None:
                self._fail_loading()
                raise failure
        finally:
            self._lock.release()

    def _fail_loading(self):
        self.state = "failed"
        try:
            self.close()
        except BaseException:
            logging.exception(
                "Failed to reclaim phase allocator pool after startup failure"
            )

    def finish_loading(self):
        if self.state == "runtime":
            return
        if self.state != "loading" or self._active:
            raise RuntimeError(
                f"Cannot finish loading in phase allocator state {self.state}, active={self._active}"
            )
        try:
            self.record("loading.before_reclaim")
            self.close()
            self._sealed_loading_blocks = self._loading_blocks()
            self.state = "runtime"
            self.check_runtime()
            self.record("loading.finalized")
        except BaseException:
            self._fail_loading()
            raise

    def close(self):
        if self._active:
            raise RuntimeError("Cannot release an active loading pool")
        if self._pool is not None:
            torch.cuda.synchronize(self.device)
            # Active weight storages retain their allocator ownership. Dropping
            # the MemPool handle makes its unused pages eligible for reclamation.
            self._pool = None
            gc.collect()
            torch.cuda.empty_cache()

    def check_runtime(self):
        settings = torch.cuda.memory._snapshot()["allocator_settings"]
        if self._active or settings["PYTORCH_CUDA_ALLOC_CONF"] != self.runtime_settings:
            raise RuntimeError(
                "Runtime allocation attempted while loading settings are active"
            )
        self._check_default_segments()
        if self._sealed_loading_blocks is not None:
            if self._loading_blocks() - self._sealed_loading_blocks:
                raise RuntimeError(
                    "New allocations appeared in the retired loading domain"
                )

    def _loading_blocks(self):
        return {
            (block["address"], block["size"])
            for segment in torch.cuda.memory_snapshot()
            if segment["device"] == self.device
            and tuple(segment["segment_pool_id"]) == self.pool_id
            for block in segment["blocks"]
            if block["state"] != "inactive"
        }

    def inspect_tensors(self, named_tensors, expected_pool):
        """Validate registered storage ownership; return metadata, never tensors.

        CUDA storages absent from the native snapshot are reported as external.
        They are not evidence of either native allocation domain.
        """
        segments = [
            s for s in torch.cuda.memory_snapshot() if s["device"] == self.device
        ]
        inventory = {}
        for name, tensor in named_tensors:
            if tensor.device.type != "cuda" or tensor.numel() == 0:
                continue
            if tensor.device.index != self.device:
                raise RuntimeError(f"Resource {name} belongs to another CUDA device")
            storage = tensor.untyped_storage()
            address, size = storage.data_ptr(), storage.nbytes()
            segment = next(
                (
                    s
                    for s in segments
                    if s["address"] <= address
                    and address + size <= s["address"] + s["total_size"]
                ),
                None,
            )
            pool_id = tuple(segment["segment_pool_id"]) if segment else None
            if segment is not None and pool_id != expected_pool:
                raise RuntimeError(
                    f"Resource {name} has allocation domain {pool_id}, expected {expected_pool}"
                )
            inventory[name] = dict(
                address=address, bytes=size, pool_id=pool_id, shape=list(tensor.shape)
            )
        return inventory

    def record(self, stage: str):
        torch.cuda.synchronize(self.device)
        free, total = torch.cuda.mem_get_info(self.device)
        row = {
            "stage": stage,
            "time": time.time(),
            "pid": os.getpid(),
            "rank": os.environ.get("WORLD_RANK"),
            "thread": threading.current_thread().name,
            "device": self.device,
            "state": self.state,
            "loading_pool_id": self.pool_id,
            "allocated": torch.cuda.memory_allocated(self.device),
            "reserved": torch.cuda.memory_reserved(self.device),
            "cuda_free": free,
            "cuda_total": total,
            "allocator_settings": torch.cuda.memory._snapshot()["allocator_settings"],
        }
        logging.info("[PHASE_ALLOCATOR] %s", json.dumps(row))
        trace_dir = os.environ.get("DSV4_PHASE_ALLOCATOR_TRACE_DIR")
        if trace_dir:
            row["segments"] = [
                {
                    k: s[k]
                    for k in (
                        "device",
                        "address",
                        "total_size",
                        "allocated_size",
                        "active_size",
                        "stream",
                        "segment_type",
                        "segment_pool_id",
                        "is_expandable",
                    )
                }
                for s in torch.cuda.memory_snapshot()
            ]
            path = Path(trace_dir)
            path.mkdir(parents=True, exist_ok=True)
            with (path / f"phase_{os.getpid()}.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")

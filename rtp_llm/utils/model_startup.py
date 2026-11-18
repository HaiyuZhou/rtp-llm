"""Shared startup sequencing for models opting into phase allocation.

The engine's existing initialize/forward callbacks drive this coordinator. The
native loading route covers threads; the coordinator serializes transitions.
Only pointer metadata is retained. External CUDA storage is reported separately
because the Torch snapshot cannot prove ownership of other allocators.
"""

import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

_ENGINE_STARTUP_LOCK = threading.Lock()


def iter_tensor_values(name, value):
    import torch

    if isinstance(value, torch.Tensor):
        yield name, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from iter_tensor_values(f"{name}.{key}", child)
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from iter_tensor_values(f"{name}.{index}", child)


def iter_module_tensors(module):
    """Include plain tensor attributes and containers, not just nn.Parameters."""
    for path, child in module.named_modules():
        values = {**vars(child), **child._parameters, **child._buffers}
        for name, value in values.items():
            if name in ("_modules", "_parameters", "_buffers"):
                continue
            yield from iter_tensor_values(f"{path}.{name}", value)


class ModelStartupLifecycle:
    def __init__(self, allocator):
        self.allocator = allocator
        self.state = "created"
        self._lock = threading.Lock()
        self._weights = {}
        self._audited_forward_kinds = set()
        self._expected_participant_ids = None
        self._engine_ready = False
        self._model_label = "model"

    @contextmanager
    def _transition(self, allowed):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Model startup transition is already active")
        try:
            if self.state not in allowed:
                raise RuntimeError(f"Model startup is {self.state}, expected {allowed}")
            try:
                yield
            except BaseException:
                self.state = "failed"
                try:
                    self.allocator.close()
                except BaseException:
                    logging.exception("Model startup cleanup also failed")
                raise
        finally:
            self._lock.release()

    @contextmanager
    def loading_weights(self):
        with self._transition(("created",)):
            self.state = "loading"
            with self.allocator.loading("load_weights"):
                yield
            self.state = "loaded"
            self._record("weights.loaded")

    def initialize(self, model, resources):
        if self._expected_participant_ids is not None and not self._engine_ready:
            raise RuntimeError(
                "Registered model requires the native engine preparation hook before initialize"
            )
        with self._transition(("loaded", "ready")):
            if self.state == "ready":
                self._checkpoint(model, "before_rebind")
                model._bind_init_resources(resources)
                self._checkpoint(model, "resources.bound")
                self._audited_forward_kinds.clear()
                return True
            self._prepare_weights(model, resources)
            self.allocator.finish_loading()
            self._prepare_runtime(model)
            return True

    def _prepare_weights(self, model, resources):
        if resources.kv_cache is not None:
            raise RuntimeError(
                "KV is present before weight conversion; first initialization must have no KV "
                "(use the native engine preparation hook before temporary-KV warmup)"
            )
        model._bind_init_resources(resources)
        self.state = "converting"
        with self.allocator.loading("convert_weights"):
            model._prepare_startup(resources)
            model._materialize_startup_weights()
        self._weights = self.allocator.inspect_tensors(
            model.startup_tensors(), self.allocator.pool_id
        )
        self.state = "weights_ready"
        self._record("weights.prepared", weights=self._weights)

    def _prepare_runtime(self, model):
        self.state = "initializing_runtime"
        model._initialize_startup_runtime()
        self._checkpoint(model, "runtime.ready")
        self.state = "ready"
        self._record("startup.ready")

    def _checkpoint(self, model, stage):
        try:
            self.allocator.check_runtime()
            named = list(model.startup_tensors())
            weights = self.allocator.inspect_tensors(
                ((name, tensor) for name, tensor in named if name in self._weights),
                self.allocator.pool_id,
            )
            if weights != self._weights:
                changed = next(
                    name
                    for name in self._weights
                    if weights.get(name) != self._weights[name]
                )
                raise RuntimeError(
                    f"Resident weight {changed} storage changed after conversion"
                )
            runtime = self.allocator.inspect_tensors(
                ((name, tensor) for name, tensor in named if name not in self._weights),
                (0, 0),
            )
            kv = self.allocator.inspect_tensors(model.startup_kv_tensors(), (0, 0))
            self._record(stage, weights=weights, runtime=runtime, kv=kv)
        except BaseException:
            self.state = "failed"
            raise

    def forward(self, model, callback, inputs, *args, **kwargs):
        if self._expected_participant_ids is not None and not self._engine_ready:
            raise RuntimeError("Engine models are not ready for forward")
        if self.state != "ready":
            raise RuntimeError(f"Cannot forward while model startup is {self.state}")
        if len(self._audited_forward_kinds) == 2:
            return callback(inputs, *args, **kwargs)
        kind = model.startup_forward_kind(inputs)
        if kind not in ("prefill", "decode") or kind in self._audited_forward_kinds:
            return callback(inputs, *args, **kwargs)
        self._checkpoint(model, f"first_{kind}.before")
        result = callback(inputs, *args, **kwargs)
        self._checkpoint(model, f"first_{kind}.after")
        self._audited_forward_kinds.add(kind)
        return result

    def _record(self, stage, **inventories):
        row = dict(
            stage=stage,
            state=self.state,
            model=self._model_label,
            pid=os.getpid(),
            rank=os.environ.get("WORLD_RANK"),
            time=time.time(),
        )
        for role, inventory in inventories.items():
            storages = {(v["address"], v["bytes"]): v for v in inventory.values()}
            row[role] = dict(
                storages=len(storages),
                bytes=sum(k[1] for k in storages),
                external_storages=sum(v["pool_id"] is None for v in storages.values()),
            )
            if role == "kv":
                row[role]["groups"] = {
                    name: value.get("shape")
                    for name, value in inventory.items()
                    if name.endswith("kv_cache_base")
                }
        logging.info("[MODEL_STARTUP] %s", json.dumps(row))
        directory = os.environ.get("DSV4_PHASE_ALLOCATOR_TRACE_DIR")
        if directory:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            with (path / f"startup_{os.getpid()}.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")


def _engine_lifecycles(models):
    ids = frozenset(id(model) for model in models)
    if len(ids) != len(models):
        raise RuntimeError("Engine startup contains duplicate model participants")
    lifecycles = [getattr(model, "_startup_lifecycle", None) for model in models]
    if not any(lc is not None for lc in lifecycles):
        return []
    if any(lc is None for lc in lifecycles):
        raise RuntimeError("Every engine model participant must support staged startup")
    allocator = lifecycles[0].allocator
    if any(lc.allocator is not allocator for lc in lifecycles):
        raise RuntimeError("Engine participants must share one native allocator")
    if any(
        lc._expected_participant_ids is not None and lc._expected_participant_ids != ids
        for lc in lifecycles
    ):
        raise RuntimeError(
            "Engine startup participant model set differs from registration"
        )
    return lifecycles


def register_engine_models(models):
    """Register identities only; native preparation must include every model."""
    lifecycles = _engine_lifecycles(models)
    ids = frozenset(id(model) for model in models)
    for index, lifecycle in enumerate(lifecycles):
        lifecycle._expected_participant_ids = ids
        lifecycle._model_label = "target" if index == 0 else f"draft{index-1}"


def assert_engine_models_ready(models):
    lifecycles = _engine_lifecycles(models)
    for model, lifecycle in zip(models, lifecycles):
        if lifecycle.state != "ready" or not lifecycle._engine_ready:
            raise RuntimeError(
                f"Engine model {lifecycle._model_label} is not ready: {lifecycle.state}"
            )
        lifecycle._checkpoint(model, "engine.before_kv")
    return bool(lifecycles)


def prepare_engine_models(models, resources):
    """Called by C++ before any warmup/cache manager, once for the whole group."""
    lifecycles = _engine_lifecycles(models)
    if not lifecycles:
        return False
    if not _ENGINE_STARTUP_LOCK.acquire(blocking=False):
        raise RuntimeError("Another engine startup group is active")
    try:
        if all(lc.state == "ready" and lc._engine_ready for lc in lifecycles):
            return assert_engine_models_ready(models)
        if any(lc.state != "loaded" for lc in lifecycles):
            raise RuntimeError(
                "Engine participants must all be loaded before preparation"
            )
        if isinstance(resources, dict):
            resources = SimpleNamespace(**dict(kv_cache=None, **resources))
        try:
            for model, lifecycle in zip(models, lifecycles):
                with lifecycle._transition(("loaded",)):
                    lifecycle._prepare_weights(model, resources)
            lifecycles[0].allocator.finish_loading()
            for model, lifecycle in zip(models, lifecycles):
                with lifecycle._transition(("weights_ready",)):
                    lifecycle._prepare_runtime(model)
            for lifecycle in lifecycles:
                lifecycle._engine_ready = True
                lifecycle._record("engine.ready")
        except BaseException:
            for lifecycle in lifecycles:
                lifecycle.state = "failed"
                lifecycle._engine_ready = False
            try:
                lifecycles[0].allocator.close()
            except BaseException:
                logging.exception("Engine startup group cleanup also failed")
            raise
        return True
    finally:
        _ENGINE_STARTUP_LOCK.release()

"""Shared startup sequencing, failures and bounded runtime checkpoints."""

import contextlib
import types
import unittest

from rtp_llm.utils.model_startup import ModelStartupLifecycle


class Allocator:
    device = 0
    pool_id = (0, 7)

    def __init__(self, events):
        self.events = events
        self.state = "loading"

    @contextlib.contextmanager
    def loading(self, name):
        self.events.append(name + ".enter")
        yield
        self.events.append(name + ".exit")

    def finish_loading(self):
        self.events.append("retire")
        self.state = "runtime"

    def close(self):
        self.events.append("close")

    def check_runtime(self):
        self.events.append("check")

    def inspect_tensors(self, tensors, pool):
        return {
            name: dict(address=tensor.address, bytes=8, pool_id=pool)
            for name, tensor in tensors
        }


class Model:
    def __init__(self, events):
        self.events = events
        self.weights = []
        self.kv_cache = None

    def _bind_init_resources(self, resources):
        self.kv_cache = resources.kv_cache
        self.events.append("bind")

    def _prepare_startup(self, resources):
        self.events.append("prepare")

    def _materialize_startup_weights(self):
        self.events.append("convert")

    def _initialize_startup_runtime(self):
        self.events.append("runtime")

    def startup_tensors(self):
        return iter(self.weights)

    def startup_kv_tensors(self):
        return iter(())

    def startup_forward_kind(self, inputs):
        return inputs.kind


class ModelStartupTest(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.allocator = Allocator(self.events)
        self.lifecycle = ModelStartupLifecycle(self.allocator)
        self.model = Model(self.events)
        self.resources = types.SimpleNamespace(kv_cache=None)

    def loaded(self):
        with self.lifecycle.loading_weights():
            self.events.append("load")
        return self.lifecycle

    def ready(self):
        self.loaded().initialize(self.model, self.resources)
        return self.lifecycle

    def test_common_sequence_retires_before_runtime_and_initializes_once(self):
        lifecycle = self.ready()
        important = [e for e in self.events if e != "check"]
        self.assertEqual(
            important,
            [
                "load_weights.enter",
                "load",
                "load_weights.exit",
                "bind",
                "convert_weights.enter",
                "prepare",
                "convert",
                "convert_weights.exit",
                "retire",
                "runtime",
            ],
        )
        self.resources.kv_cache = object()
        lifecycle.initialize(self.model, self.resources)
        self.assertIs(self.model.kv_cache, self.resources.kv_cache)
        self.assertEqual(self.events.count("convert"), 1)
        self.assertEqual(self.events.count("runtime"), 1)
        self.assertEqual(lifecycle.state, "ready")

    def test_early_production_kv_is_rejected(self):
        self.loaded()
        self.resources.kv_cache = object()
        with self.assertRaisesRegex(RuntimeError, "KV.*before"):
            self.lifecycle.initialize(self.model, self.resources)
        self.assertNotIn("convert", self.events)

    def test_initialize_before_load_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "created"):
            self.lifecycle.initialize(self.model, self.resources)

    def test_loading_cannot_reenter_or_restart(self):
        with self.lifecycle.loading_weights():
            with self.assertRaisesRegex(RuntimeError, "active|loading"):
                with self.lifecycle.loading_weights():
                    pass
        self.lifecycle.initialize(self.model, self.resources)
        with self.assertRaisesRegex(RuntimeError, "ready"):
            with self.lifecycle.loading_weights():
                pass

    def test_conversion_failure_closes_pool_and_is_terminal(self):
        self.loaded()

        def fail():
            raise ValueError("bad weights")

        self.model._materialize_startup_weights = fail
        with self.assertRaisesRegex(ValueError, "bad weights"):
            self.lifecycle.initialize(self.model, self.resources)
        self.assertEqual(self.lifecycle.state, "failed")
        self.assertIn("close", self.events)
        self.assertNotIn("runtime", self.events)
        with self.assertRaisesRegex(RuntimeError, "failed"):
            self.lifecycle.initialize(self.model, self.resources)

    def test_runtime_setup_failure_is_terminal(self):
        self.loaded()
        self.model._initialize_startup_runtime = lambda: (_ for _ in ()).throw(
            ValueError("runtime failed")
        )
        with self.assertRaisesRegex(ValueError, "runtime failed"):
            self.lifecycle.initialize(self.model, self.resources)
        self.assertEqual(self.lifecycle.state, "failed")

    def test_weight_storage_replacement_is_rejected_on_rebind(self):
        self.model.weights = [("weight", types.SimpleNamespace(address=123))]
        self.ready()
        self.model.weights = [("weight", types.SimpleNamespace(address=456))]
        with self.assertRaisesRegex(RuntimeError, "weight.*changed"):
            self.lifecycle.initialize(self.model, self.resources)

    def test_first_prefill_and_decode_audit_before_and_after_only_once(self):
        self.ready()
        self.events.clear()
        for kind in ("prefill", "prefill", "decode", "decode"):
            inputs = types.SimpleNamespace(kind=kind)
            result = self.lifecycle.forward(self.model, lambda x: "answer", inputs)
            self.assertEqual(result, "answer")
        self.assertEqual(self.events.count("check"), 4)

    def test_production_binding_rearms_checks_after_temporary_kv_forward(self):
        lifecycle = self.ready()
        temporary = types.SimpleNamespace(kv_cache=object())
        production = types.SimpleNamespace(kv_cache=object())
        inputs = types.SimpleNamespace(kind="prefill")
        lifecycle.initialize(self.model, temporary)
        lifecycle.forward(self.model, lambda _: None, inputs)
        lifecycle.initialize(self.model, self.resources)  # Native warmup reset.
        lifecycle.initialize(self.model, production)
        self.events.clear()
        lifecycle.forward(self.model, lambda _: None, inputs)
        self.assertEqual(self.events.count("check"), 2)

    def test_preparation_inherits_conversion_loading_scope(self):
        self.ready()
        self.assertLess(
            self.events.index("convert_weights.enter"), self.events.index("prepare")
        )

    def test_forward_before_ready_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "created"):
            self.lifecycle.forward(
                self.model, lambda _: None, types.SimpleNamespace(kind="prefill")
            )

    def test_failed_forward_does_not_consume_first_forward_audit(self):
        self.ready()
        inputs = types.SimpleNamespace(kind="decode")

        def fail(_):
            raise ValueError("forward failed")

        with self.assertRaisesRegex(ValueError, "forward failed"):
            self.lifecycle.forward(self.model, fail, inputs)
        self.events.clear()
        self.lifecycle.forward(self.model, lambda _: None, inputs)
        self.assertEqual(self.events.count("check"), 2)


if __name__ == "__main__":
    unittest.main()

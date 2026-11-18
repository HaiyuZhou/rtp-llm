"""An engine must prepare every local model before retiring shared loading."""

import types
import unittest

from rtp_llm.utils import model_startup
from rtp_llm.utils.test.model_startup_test import Allocator, Model


class GroupModel(Model):
    def __init__(self, name, events):
        super().__init__(events)
        self.name = name
        self._startup_lifecycle = None

    def _prepare_startup(self, resources):
        self.events.append(self.name + ".prepare")

    def _materialize_startup_weights(self):
        self.events.append(self.name + ".convert")

    def _initialize_startup_runtime(self):
        self.events.append(self.name + ".runtime")


class ModelStartupGroupTest(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.allocator = Allocator(self.events)
        self.models = [GroupModel(n, self.events) for n in ("target", "draft")]
        self.resources = types.SimpleNamespace(
            kv_cache=None,
            is_speculative=True,
            is_decode_role=False,
            max_context_batch_size=1,
        )
        for model in self.models:
            model._startup_lifecycle = model_startup.ModelStartupLifecycle(
                self.allocator
            )
            with model._startup_lifecycle.loading_weights():
                pass

    def register(self):
        self.assertTrue(
            callable(getattr(model_startup, "register_engine_models", None)),
            "Engine participant registration is missing",
        )
        model_startup.register_engine_models(self.models)

    def prepare(self):
        self.register()
        return model_startup.prepare_engine_models(self.models, self.resources)

    def test_all_weights_precede_one_retirement_and_every_runtime(self):
        self.prepare()
        e = self.events
        self.assertLess(e.index("target.convert"), e.index("retire"))
        self.assertLess(e.index("draft.convert"), e.index("retire"))
        self.assertLess(e.index("retire"), e.index("target.runtime"))
        self.assertLess(e.index("retire"), e.index("draft.runtime"))
        self.assertEqual(e.count("retire"), 1)
        self.assertTrue(model_startup.assert_engine_models_ready(self.models))

    def test_registered_model_cannot_implicitly_seal_without_engine_hook(self):
        self.register()
        with self.assertRaisesRegex(RuntimeError, "engine.*prepar|engine.*hook"):
            self.models[0]._startup_lifecycle.initialize(self.models[0], self.resources)
        self.assertNotIn("retire", self.events)

    def test_missing_draft_is_rejected_before_any_conversion(self):
        self.register()
        with self.assertRaisesRegex(RuntimeError, "participant|model set"):
            model_startup.prepare_engine_models(self.models[:1], self.resources)
        self.assertNotIn("target.convert", self.events)

    def test_duplicate_model_is_rejected(self):
        self.register()
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            model_startup.prepare_engine_models(
                [self.models[0], self.models[0]], self.resources
            )

    def test_models_with_different_allocators_are_rejected(self):
        self.models[1]._startup_lifecycle.allocator = Allocator(self.events)
        with self.assertRaisesRegex(RuntimeError, "allocator"):
            self.prepare()
        self.assertNotIn("retire", self.events)

    def test_group_and_wrapper_initialization_are_idempotent(self):
        self.prepare()
        model_startup.prepare_engine_models(self.models, self.resources)
        for model in self.models:
            for _ in range(2):
                model._startup_lifecycle.initialize(model, self.resources)
        self.assertEqual(self.events.count("retire"), 1)
        self.assertEqual(self.events.count("target.convert"), 1)
        self.assertEqual(self.events.count("draft.convert"), 1)
        self.assertEqual(self.events.count("target.runtime"), 1)
        self.assertEqual(self.events.count("draft.runtime"), 1)

    def test_failed_draft_conversion_poisons_target_and_group(self):
        self.register()
        self.models[1]._materialize_startup_weights = lambda: (_ for _ in ()).throw(
            ValueError("draft bad")
        )
        with self.assertRaisesRegex(ValueError, "draft bad"):
            model_startup.prepare_engine_models(self.models, self.resources)
        self.assertTrue(
            all(m._startup_lifecycle.state == "failed" for m in self.models)
        )
        self.assertNotIn("retire", self.events)
        self.assertNotIn("target.runtime", self.events)

    def test_failed_draft_runtime_does_not_publish_ready_target(self):
        self.register()
        self.models[1]._initialize_startup_runtime = lambda: (_ for _ in ()).throw(
            ValueError("draft runtime bad")
        )
        with self.assertRaisesRegex(ValueError, "draft runtime bad"):
            model_startup.prepare_engine_models(self.models, self.resources)
        self.assertTrue(
            all(m._startup_lifecycle.state == "failed" for m in self.models)
        )
        with self.assertRaisesRegex(RuntimeError, "ready|failed"):
            model_startup.assert_engine_models_ready(self.models)

    def test_native_context_dictionary_preserves_role_and_batch_settings(self):
        self.register()
        contexts = []
        self.models[0]._prepare_startup = lambda r: contexts.append(r)
        model_startup.prepare_engine_models(
            self.models,
            dict(is_speculative=True, is_decode_role=True, max_context_batch_size=3),
        )
        self.assertIsNone(contexts[0].kv_cache)
        self.assertTrue(contexts[0].is_decode_role)
        self.assertEqual(contexts[0].max_context_batch_size, 3)

    def test_disabled_models_need_no_coordinator(self):
        models = [types.SimpleNamespace(_startup_lifecycle=None)]
        self.assertTrue(
            callable(getattr(model_startup, "register_engine_models", None))
        )
        model_startup.register_engine_models(models)
        self.assertFalse(model_startup.prepare_engine_models(models, {}))
        self.assertFalse(model_startup.assert_engine_models_ready(models))


if __name__ == "__main__":
    unittest.main()

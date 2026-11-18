"""Registration must guard native startup without importing engine bindings."""

import importlib.util
import types
import unittest
from pathlib import Path

# The wrapper itself has no native imports until construction. Load its source
# directly so this regression also runs without a built engine extension.
_spec = importlib.util.spec_from_file_location(
    "startup_op_wrapper",
    Path(__file__).resolve().parents[2] / "ops/rtp_llm/rtp_llm_op.py",
)
_wrapper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_wrapper)
RtpLLMOp = _wrapper.RtpLLMOp
from rtp_llm.utils.model_startup import ModelStartupLifecycle
from rtp_llm.utils.test.model_startup_test import Allocator, Model


class EngineStartupRegistrationTest(unittest.TestCase):
    def operation(self, draft=True):
        op = RtpLLMOp.__new__(RtpLLMOp)
        allocator = Allocator([])
        models = [Model([]) for _ in range(2 if draft else 1)]
        for model in models:
            model._startup_lifecycle = ModelStartupLifecycle(allocator)
        op.model = types.SimpleNamespace(py_model=models[0])
        op.propose_model = (
            types.SimpleNamespace(model=types.SimpleNamespace(py_model=models[1]))
            if draft
            else None
        )
        return op, models

    def test_target_and_draft_registered_before_native_entry(self):
        op, models = self.operation()
        op._register_startup_models()
        for model in models:
            self.assertEqual(
                model._startup_lifecycle._expected_participant_ids,
                frozenset(map(id, models)),
            )
        with self.assertRaisesRegex(RuntimeError, "native engine"):
            models[0]._startup_lifecycle.initialize(
                models[0], types.SimpleNamespace(kv_cache=None)
            )

    def test_target_only_registered(self):
        op, models = self.operation(False)
        op._register_startup_models()
        self.assertEqual(
            models[0]._startup_lifecycle._expected_participant_ids,
            frozenset(map(id, models)),
        )

    def test_missing_draft_support_is_rejected(self):
        op, models = self.operation()
        models[1]._startup_lifecycle = None
        with self.assertRaisesRegex(RuntimeError, "Every engine model"):
            op._register_startup_models()

    def test_disabled_models_unchanged(self):
        op = RtpLLMOp.__new__(RtpLLMOp)
        op.model = types.SimpleNamespace(py_model=None)
        op.propose_model = None
        op._register_startup_models()

    def test_disabled_deterministic_proposal_has_no_draft_model(self):
        op = RtpLLMOp.__new__(RtpLLMOp)
        op.model = types.SimpleNamespace(
            py_model=types.SimpleNamespace(_startup_lifecycle=None)
        )
        op.propose_model = types.SimpleNamespace(model=None)
        op._register_startup_models()


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import Mock

from rtp_llm.test.perf_test.capabilities import CapabilityError, validate_capability
from rtp_llm.test.perf_test.errors import (
    CaseExecutionError,
    CollectionError,
    ConfigurationError,
    EngineLifecycleError,
    PartialRunError,
    PlanningError,
    PostExecutionError,
    ResourceError,
)


class CapabilitiesTest(unittest.TestCase):
    def test_complete_capability_matrix(self):
        supported = {
            ("prefill", "local", "run"),
            ("prefill", "local", "resume"),
            ("prefill", "local", "profile"),
            ("prefill", "local", "report"),
            ("prefill", "aone", "run"),
            ("prefill", "aone", "report"),
            ("decode", "local", "run"),
        }
        for mode in ("prefill", "decode"):
            for backend in ("local", "aone"):
                for action in ("run", "resume", "profile", "report"):
                    combination = (mode, backend, action)
                    with self.subTest(combination=combination):
                        if combination in supported:
                            capability = validate_capability(mode, action, backend)
                            self.assertEqual(combination, capability.key)
                            self.assertEqual(
                                combination == ("prefill", "aone", "report"),
                                capability.offline_only,
                            )
                        else:
                            with self.assertRaises(CapabilityError) as context:
                                validate_capability(mode, action, backend)
                            self.assertEqual(2, context.exception.exit_code)

    def test_unknown_values_are_typed_configuration_failures(self):
        for args, path in (
            (("unknown", "run", "local"), "mode"),
            (("prefill", "unknown", "local"), "action"),
            (("prefill", "run", "unknown"), "backend"),
        ):
            with self.subTest(args=args):
                with self.assertRaisesRegex(CapabilityError, path) as context:
                    validate_capability(*args)
                self.assertIsInstance(context.exception, ConfigurationError)

    def test_unsupported_combination_fails_before_all_side_effects(self):
        create_filesystem = Mock()
        build_target = Mock()
        start_engine = Mock()
        submit_job = Mock()

        def admitted_operation():
            validate_capability("decode", "run", "aone")
            create_filesystem()
            build_target()
            start_engine()
            submit_job()

        with self.assertRaises(CapabilityError):
            admitted_operation()

        create_filesystem.assert_not_called()
        build_target.assert_not_called()
        start_engine.assert_not_called()
        submit_job.assert_not_called()

    def test_failure_types_have_stable_exit_code_classes(self):
        expected = {
            ConfigurationError: 2,
            PlanningError: 3,
            ResourceError: 3,
            EngineLifecycleError: 4,
            CaseExecutionError: 5,
            PartialRunError: 5,
            PostExecutionError: 6,
            CollectionError: 6,
        }
        for error_type, exit_code in expected.items():
            with self.subTest(error_type=error_type):
                error = error_type("failure")
                self.assertEqual(exit_code, error.exit_code)
                self.assertEqual("failure", str(error))


if __name__ == "__main__":
    unittest.main()

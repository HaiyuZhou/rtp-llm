import unittest
from unittest.mock import patch

from rtp_llm.test.perf_test.engine_paths import resolve_engine_paths


class EnginePathsTest(unittest.TestCase):
    @patch(
        "rtp_llm.test.perf_test.engine_paths.resolve_checkpoint_or_tokenizer_for_perf"
    )
    @patch("rtp_llm.test.perf_test.engine_paths.needs_perf_hub_resolve")
    def test_shared_hub_path_is_resolved_once(self, needs_resolve, resolve):
        needs_resolve.return_value = True
        resolve.return_value = "/models/qwen"

        actual = resolve_engine_paths(
            [
                "--checkpoint_path",
                "Qwen/Model",
                "--tokenizer_path=Qwen/Model",
            ]
        )

        self.assertEqual(
            [
                "--checkpoint_path",
                "/models/qwen",
                "--tokenizer_path=/models/qwen",
            ],
            actual,
        )
        resolve.assert_called_once_with("Qwen/Model")


if __name__ == "__main__":
    unittest.main()

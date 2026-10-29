import unittest
from unittest.mock import Mock

from rtp_llm.test.perf_test.execution.scheduler import configure_scheduler


class ExecutionSchedulerTest(unittest.TestCase):
    def test_requires_explicit_success_acknowledgement(self):
        response = Mock()
        response.json.return_value = {"status": "error", "error": "bad mode"}
        post = Mock(return_value=response)

        with self.assertRaisesRegex(RuntimeError, "scheduler rejected"):
            configure_scheduler(1234, batch_size=2, mode="decode", post=post)

        response.raise_for_status.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

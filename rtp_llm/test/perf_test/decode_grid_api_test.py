import unittest
from pathlib import Path
from unittest.mock import patch

from rtp_llm.test.perf_test.decode.grid_runner import run_single


class DecodeGridApiTest(unittest.TestCase):
    def test_stable_run_single_forwards_existing_server_grid(self):
        with patch(
            "rtp_llm.test.perf_test.decode.grid_runner.run_cartesian_grid",
            return_value=["metric"],
        ) as run:
            result = run_single(
                1234,
                2,
                [4],
                [128],
                {128: "prompt"},
                dump_json_path="/tmp/output",
                decode_test_length=16,
                generate_config={"top_k": 1},
            )

        self.assertEqual(["metric"], result)
        run.assert_called_once_with(
            1234,
            2,
            [4],
            [128],
            {128: "prompt"},
            is_decode=True,
            dump_json_path="/tmp/output",
            decode_test_length=16,
            tp_size=1,
            generate_config={"top_k": 1},
        )

    def test_multi_node_caller_does_not_import_old_entry_module(self):
        source = (
            Path(__file__).resolve().parent / "multi_node" / "local_server_runner.py"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "from rtp_llm.test.perf_test.decode.grid_runner import run_single", source
        )
        self.assertNotIn("perf_test.batch_decode_test import run_single", source)


if __name__ == "__main__":
    unittest.main()

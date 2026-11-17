"""Real sockets verify completion cannot be inferred from peer disappearance."""

import json
import socket
import unittest

from rtp_llm.test.perf_test.execution.distributed.member import CompletionGroup


class CompletionGroupTest(unittest.TestCase):
    def test_worker_receives_explicit_master_failure(self):
        master = CompletionGroup(
            run_id="run-a",
            world_rank=0,
            member_ranks=(0, 2),
            master_host="127.0.0.1",
            port=0,
            timeout_seconds=2,
        )
        port = master.listener.getsockname()[1]
        worker = CompletionGroup(
            run_id="run-a",
            world_rank=2,
            member_ranks=(0, 2),
            master_host="127.0.0.1",
            port=port,
            timeout_seconds=2,
        )
        try:
            master.ready()
            master.finish(7)
            self.assertEqual(7, worker.wait())
        finally:
            master.close()
            worker.close()

    def test_stale_run_identity_is_rejected(self):
        master = CompletionGroup(
            run_id="current",
            world_rank=0,
            member_ranks=(0, 1),
            master_host="127.0.0.1",
            port=0,
            timeout_seconds=2,
        )
        peer = socket.create_connection(("127.0.0.1", master.listener.getsockname()[1]))
        try:
            peer.sendall(
                json.dumps({"run_id": "stale", "world_rank": 1}).encode() + b"\n"
            )
            with self.assertRaisesRegex(RuntimeError, "identity"):
                master.ready()
        finally:
            peer.close()
            master.close()

    def test_disconnect_is_failure_and_wait_remains_cancellable(self):
        master = CompletionGroup(
            run_id="run",
            world_rank=0,
            member_ranks=(0, 1),
            master_host="127.0.0.1",
            port=0,
            timeout_seconds=2,
        )
        worker = CompletionGroup(
            run_id="run",
            world_rank=1,
            member_ranks=(0, 1),
            master_host="127.0.0.1",
            port=master.listener.getsockname()[1],
            timeout_seconds=2,
        )
        try:
            master.ready()
            master.close()
            with self.assertRaisesRegex(RuntimeError, "disconnected"):
                worker.wait()
        finally:
            worker.close()
        cancelled = CompletionGroup(
            run_id="run",
            world_rank=0,
            member_ranks=(0, 1),
            master_host="127.0.0.1",
            port=0,
            timeout_seconds=2,
            check_cancelled=lambda: (_ for _ in ()).throw(RuntimeError("cancelled")),
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                cancelled.ready()
        finally:
            cancelled.close()


if __name__ == "__main__":
    unittest.main()

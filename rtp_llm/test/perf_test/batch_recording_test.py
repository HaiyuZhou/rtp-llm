import unittest

from rtp_llm.test.perf_test.batch_replay import make_plan
from rtp_llm.test.perf_test.batch_trace_analyze import (
    compare_kernels,
    correlate_trace,
    execution_timings,
    request_latencies,
    validate_batch,
)


class RecordingTest(unittest.TestCase):
    def events(self, names):
        return [
            dict(
                session_id="s",
                replica_id="d0",
                dp_rank=0,
                request_id="r1",
                owner_instance_id="o",
                clock_id="c",
                event=name,
                event_seq=i,
                timestamp_monotonic_ns=100 + i * 20,
            )
            for i, name in enumerate(names)
        ]

    def test_lifecycle(self):
        events = self.events(["enqueue", "first_scheduled", "first_token", "finish"])
        result = request_latencies(list(reversed(events)))[0]
        self.assertEqual(result["engine_request_latency_ns"], 60)
        self.assertEqual(result["initial_queue_latency_ns"], 20)
        self.assertEqual(result["engine_first_token_latency_ns"], 40)
        self.assertEqual(result["post_first_schedule_latency_ns"], 40)

    def test_cancel_before_schedule(self):
        result = request_latencies(self.events(["enqueue", "cancel"]))[0]
        self.assertEqual(result["status"], "complete")
        self.assertIsNone(result["initial_queue_latency_ns"])
        self.assertIsNone(result["post_first_schedule_latency_ns"])

    def test_missing_and_duplicate(self):
        events = self.events(["enqueue", "first_scheduled", "first_token", "finish"])
        self.assertIsNone(
            request_latencies(events[:1] + events[2:])[0]["engine_request_latency_ns"]
        )
        self.assertEqual(
            request_latencies(events + [events[-1]])[0]["status"], "partial"
        )
        events[-1]["clock_id"] = "other"
        self.assertEqual(request_latencies(events)[0]["status"], "partial")

    def test_gpu_outside_cpu_scope(self):
        trace = {
            "traceEvents": [
                dict(ph="X", name="rtp.execution(id=42)", pid=1, tid=7, ts=10, dur=10),
                dict(
                    ph="X",
                    name="cudaLaunchKernel",
                    cat="cuda_runtime",
                    pid=1,
                    tid=7,
                    ts=12,
                    dur=1,
                    args={"correlation": 99},
                ),
                dict(
                    ph="X",
                    name="kernel",
                    cat="kernel",
                    pid=0,
                    tid=3,
                    ts=100,
                    dur=4,
                    args={"correlation": 99},
                ),
                dict(
                    ph="X",
                    name="unmatched",
                    cat="kernel",
                    pid=0,
                    tid=3,
                    ts=13,
                    dur=1,
                    args={"correlation": 100},
                ),
            ]
        }
        kernels, _ = correlate_trace(trace, {"world_rank": 0})
        self.assertEqual(kernels[0]["execution_id"], 42)
        self.assertEqual(kernels[0]["duration_ns"], 4000)
        self.assertIsNone(kernels[1]["execution_id"])

    def test_overlapping_kernel_times(self):
        kernels = [
            dict(
                execution_id=42,
                world_rank=0,
                device=0,
                start_ns=start,
                duration_ns=duration,
            )
            for start, duration in ((0, 10), (5, 10), (20, 5))
        ]
        result = execution_timings(kernels)[0]
        self.assertEqual(result["kernel_sum_ns"], 25)
        self.assertEqual(result["kernel_union_ns"], 20)
        self.assertEqual(result["gpu_span_ns"], 25)

    def test_compare_does_not_align_by_call_index(self):
        original = [
            dict(execution_id=42, world_rank=0, kernel_name="k", duration_ns=10)
        ]
        replay = [
            dict(execution_id=42, world_rank=0, kernel_name="k", duration_ns=15),
            dict(execution_id=42, world_rank=0, kernel_name="extra", duration_ns=1),
        ]
        result = {r["kernel_name"]: r for r in compare_kernels(original, replay)}
        self.assertEqual(result["k"]["change_percent"], 50)
        self.assertIsNone(result["extra"]["change_percent"])

    def test_large_batch_and_replay(self):
        batch = dict(
            schema_version=1,
            execution_id=42,
            requests=[
                dict(
                    batch_slot=i,
                    phase="decode",
                    q_tokens=1,
                    kv_tokens_before=i + 1,
                    prompt_tokens=1,
                )
                for i in range(100)
            ],
        )
        self.assertEqual(len(validate_batch(batch)), 100)
        self.assertEqual(len(make_plan([batch]).splitlines()), 102)
        batch["requests"][0]["q_tokens"] = 2
        with self.assertRaises(ValueError):
            make_plan([batch])


if __name__ == "__main__":
    unittest.main()

import copy
import json
import sqlite3
import tempfile
import threading
import time
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import Mock

from rtp_llm.test.perf_test.cache_grid.runner.continuous_pressure import BoundedSender
from rtp_llm.test.perf_test.cache_grid.runner.pressure_profile import (
    KinetoCapture,
    NsightCapture,
    ProfileController,
    classify_gpu_gap,
    run_nsys_window,
    validate_kineto,
)


class NsightPressureTest(unittest.TestCase):
    def capture(self, fault=None):
        from rtp_llm.test.perf_test.cache_grid.runner.pressure_diagnostics import (
            CoverageAssessment,
        )

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        calls = []

        def command(record, action, *options):
            calls.append(action)
            record.setdefault("nsys_commands", []).append(dict(action=action))
            if action in (fault or ()):
                raise RuntimeError("injected " + action)
            if action == "stop":
                capture.report.write_bytes(b"report")
            if action == "export":
                nsys_fixture(capture.sqlite)

        capture = NsightCapture(
            command=command,
            duration_s=0.01,
            trace_name="pressure_fixture",
            directory=root,
            ranks=(0, 1),
            instances={0: "a", 1: "b"},
            session_id="s",
            timeout_s=1,
            command_timeout_s=0.1,
        )
        capture.start(time.monotonic())
        for rank in (0, 1):
            capture.steps[rank].update((s["step_id"], s) for s in steps())
        capture.controller.result(2)
        coverage = Mock()
        coverage.assess.return_value = CoverageAssessment(
            True, 4096, 0.9, 3, True, (), ()
        )
        return capture.finish(coverage), calls, capture, coverage

    def test_capture_is_duration_based_with_separate_export_and_pressure_bounds(self):
        result, calls, capture, coverage = self.capture()
        self.assertEqual(calls, ["start", "stop", "export"])
        self.assertTrue(result["trace_valid"], result)
        coverage.assess.assert_called_once_with(
            capture.capture_begin_s, capture.capture_end_s
        )
        self.assertEqual(
            result["pressure"]["bounds_kind"], "client_start_call_to_stop_ack"
        )

    def test_start_stop_and_export_failures_remain_invalid_and_preserve_partial_files(
        self,
    ):
        for fault in (("start", "stop"), ("export",), ("stop",)):
            with self.subTest(fault=fault):
                result, calls, capture, _ = self.capture(fault)
                self.assertFalse(result["trace_valid"], result)
                self.assertIn("stop", calls)
                errors = repr(result["control_errors"])
                for action in fault:
                    self.assertIn("injected " + action, errors)
                if "export" in fault:
                    self.assertTrue(capture.report.exists())
                    self.assertIn(str(capture.report), result["partial_files"])

    def test_stop_attempted_after_start_failure(self):
        calls = []

        def command(action):
            calls.append(action)
            if action == "start":
                raise RuntimeError("injected start failure")

        with self.assertRaisesRegex(RuntimeError, "start failure"):
            run_nsys_window(command, duration_s=0.01, cancelled=threading.Event())
        self.assertEqual(calls, ["start", "stop"])

    def test_early_submission_is_not_cpu_delay(self):
        result = classify_gpu_gap(90, 100, 130, dependency_complete=False)
        self.assertEqual(result["classification"], "unresolved_gpu_dependency")
        self.assertIsNone(result["exposed_idle_ns"])

    def test_cancellation_and_explicit_duration(self):
        cancelled = threading.Event()
        cancelled.set()
        calls = []
        run_nsys_window(calls.append, duration_s=60, cancelled=cancelled)
        self.assertEqual(calls, ["start", "stop"])
        for duration in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                run_nsys_window(calls.append, duration_s=duration, cancelled=cancelled)

    def test_schema_identity_correlations_and_unknown_dependencies(self):
        from rtp_llm.test.perf_test.cache_grid.runner.pressure_nsys import (
            normalize_nsys,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.sqlite"
            nsys_fixture(path)
            result = normalize_nsys(path, {0: steps(), 1: steps()})
            self.assertTrue(result["trace_valid"], result)
            self.assertEqual(result["ranks"]["0"]["observed_step_ids"], [1, 2, 3])
            self.assertEqual(result["ranks"]["1"]["observed_step_ids"], [1, 2, 3])
            self.assertTrue(result["gpu_gaps"])
            self.assertTrue(
                all(g["exposed_idle_ns"] is None for g in result["gpu_gaps"])
            )
            with sqlite3.connect(path) as db:
                db.execute("UPDATE NVTX_EVENTS SET uint64Value=NULL")
            result = normalize_nsys(path, {0: steps(), 1: steps()})
            self.assertFalse(result["trace_valid"])
            self.assertIn("missing_explicit_rank_metadata", result["reasons"])

    def test_required_schema_checked_optional_tables_absent(self):
        from rtp_llm.test.perf_test.cache_grid.runner.pressure_nsys import (
            normalize_nsys,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.sqlite"
            nsys_fixture(path)
            with sqlite3.connect(path) as db:
                db.execute("DROP TABLE CUPTI_ACTIVITY_KIND_RUNTIME")
            result = normalize_nsys(path, {0: steps()})
            self.assertFalse(result["trace_valid"])
            self.assertIn("unsupported_schema", result["reasons"])

    def test_device_mapping_is_process_local_and_optional_schema_is_validated(self):
        from rtp_llm.test.perf_test.cache_grid.runner.pressure_nsys import (
            normalize_nsys,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.sqlite"
            nsys_fixture(path)
            with sqlite3.connect(path) as db:
                db.executescript(
                    """
                    CREATE TABLE TARGET_INFO_CUDA_DEVICE(gpuId INTEGER, cudaId INTEGER, pid INTEGER, uuid TEXT);
                    INSERT INTO TARGET_INFO_CUDA_DEVICE VALUES(5,0,629566,'gpu-five'),(6,0,629567,'gpu-six');
                    UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET deviceId=0;
                """
                )
            result = normalize_nsys(path, {0: steps(), 1: steps()})
            self.assertTrue(result["trace_valid"], result)
            kernels = [e for e in result["events"] if e["kind"] == "kernel"]
            self.assertEqual(
                {e["physical_device"] for e in kernels if e["rank"] == 0}, {"gpu-five"}
            )
            self.assertEqual(
                {e["physical_device"] for e in kernels if e["rank"] == 1}, {"gpu-six"}
            )
            with sqlite3.connect(path) as db:
                db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER(start INTEGER)")
            self.assertIn(
                "unsupported_schema",
                normalize_nsys(path, {0: steps(), 1: steps()})["reasons"],
            )


def nsys_fixture(path):
    # Nsight 2025.6.3 column projection. Preserve NVIDIA's encoded PID/TID join.
    with sqlite3.connect(path) as db:
        db.executescript(
            """
          CREATE TABLE StringIds(id INTEGER, value TEXT);
          CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT,textId INTEGER,
            globalTid INTEGER,uint64Value INTEGER);
          CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,
            globalTid INTEGER,correlationId INTEGER,nameId INTEGER);
          CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,
            globalPid INTEGER,correlationId INTEGER,deviceId INTEGER,contextId INTEGER,
            streamId INTEGER,demangledName INTEGER);
          INSERT INTO StringIds VALUES(1,'cudaLaunchKernel'),(2,'model_matmul');
        """
        )
        for rank in (0, 1):
            pid = (1 << 48) | ((629566 + rank) << 24)
            tid = pid | (629566 + rank)
            for step in (1, 2, 3):
                start = step * 1000
                db.execute(
                    "INSERT INTO NVTX_EVENTS VALUES(?,?,?,?,?,?)",
                    (start, start + 100, f"pressure.step/{step}", None, tid, rank),
                )
                db.execute(
                    "INSERT INTO NVTX_EVENTS VALUES(?,?,?,?,?,?)",
                    (start + 1, start + 99, "executor.model_forward", None, tid, None),
                )
                db.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?,?)",
                    (start + 10, start + 20, tid, step, 1),
                )
                db.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?,?,?,?)",
                    (start + 30, start + 90, pid, step, rank, 1, 7, 2),
                )


class PressureProfileTest(unittest.TestCase):
    def test_result_deadline_requests_independent_cleanup(self):
        entered, release, stopped = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )

        def wait():
            entered.set()
            release.wait(1)
            return {}

        def stop():
            stopped.set()
            release.set()

        controller = ProfileController(lambda: None, wait, stop)
        controller.start()
        self.assertTrue(entered.wait(1))
        try:
            with self.assertRaises(TimeoutError):
                controller.result(0.01)
            self.assertTrue(stopped.wait(0.1))
        finally:
            release.set()
        controller.result(1)

    def test_stop_during_start_retries_after_start_acknowledges(self):
        entered, release = threading.Event(), threading.Event()
        active = []
        calls = []

        def start():
            entered.set()
            release.wait(1)
            active.append(True)

        def stop():
            calls.append(bool(active))
            active.clear()

        controller = ProfileController(start, lambda: {}, stop)
        controller.start()
        self.assertTrue(entered.wait(1))
        controller.request_stop()
        controller.stop_future.result(1)
        release.set()
        result = controller.result(1)
        self.assertFalse(active)
        self.assertEqual(calls, [False, True])
        self.assertFalse(result["trace_valid"])

    def capture_case(self, fault=None):
        from rtp_llm.test.perf_test.cache_grid.runner.pressure_diagnostics import (
            CoverageAssessment,
        )

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        pages = []
        for rank in (0, 1):
            path = root / f"pressure_fixture_wr{rank}_1.json"
            path.write_text(
                "{"
                if fault == "truncated" and rank == 1
                else json.dumps(actual_trace())
            )
            pages.append(
                dict(
                    world_rank=rank,
                    service_instance=f"instance{rank}",
                    profile=dict(
                        supported=True,
                        state="finished",
                        trace_name="pressure_fixture",
                        collection_complete=True,
                        export_complete=True,
                        file_name=str(path),
                        capture_begin_ns="1000000000",
                        capture_end_ns="4000000000",
                    ),
                )
            )
        if fault == "rank":
            pages.pop()
        if fault == "identity":
            pages[1]["service_instance"] = "other"
        if fault == "export":
            pages[1]["profile"]["export_error"] = "disk full"
        raw = dict(
            response=dict(ranks=pages),
            rank_offset_bounds_s={"0": [10, 10.1], "1": [10, 10.1]},
        )
        controller = KinetoCapture(
            start_capture=lambda: {},
            read_status=lambda timeout: raw,
            stop_capture=lambda: {},
            trace_name="pressure_fixture",
            directory=root,
            ranks=(0, 1),
            instances={0: "instance0", 1: "instance1"},
            session_id="session",
            timeout_s=0.1,
            requested_steps=3,
        )
        controller.start(99)
        sample = copy.deepcopy(raw)
        for page in sample["response"]["ranks"]:
            page["steps"] = [
                dict(
                    step_id=str(i),
                    session_id="session",
                    completed=True,
                    context_count="1",
                    preparation_begin_ns=str(i * 1000000000),
                    forward_enter_ns=str(i * 1000000000 + 1000000),
                    end_ns=str(i * 1000000000 + 2000000),
                )
                for i in (1, 2, 3)
            ]
        controller.consume(dict(event="pressure", raw=sample))
        controller.controller.result(2)
        coverage = Mock()
        coverage.assess.return_value = CoverageAssessment(
            True, 4096, 0.9, 3, False, (), ()
        )
        return controller.finish(coverage), root, coverage

    def test_rank_identity_and_real_capture_pressure_window_are_independent(self):
        result, _, coverage = self.capture_case()
        self.assertTrue(result["trace_valid"], result)
        self.assertFalse(result["pressure_valid"])
        coverage.assess.assert_called_once_with(11, 14.1)
        self.assertEqual(set(result["ranks"]), {"0", "1"})

    def test_missing_rank_identity_export_and_partial_json_fail_preserving_files(self):
        for fault in ("rank", "identity", "export", "truncated"):
            with self.subTest(fault=fault):
                result, root, _ = self.capture_case(fault)
                self.assertFalse(result["trace_valid"], result)
                self.assertTrue((root / "pressure_fixture_wr0_1.json").exists())
                self.assertTrue((root / "pressure_fixture_wr1_1.json").exists())
                if fault == "truncated":
                    self.assertEqual(
                        (root / "pressure_fixture_wr1_1.json").read_text(), "{"
                    )

    def test_actual_driver_schema_and_first_gather_boundary(self):
        result = validate_kineto(actual_trace(), steps(), 3)
        self.assertTrue(result["trace_valid"], result)
        self.assertEqual(result["usable_step_ids"], [2, 3])
        self.assertEqual(result["usable_pairs"], 1)
        self.assertEqual(result["exclusions"]["1"], "first_step_gather_omitted")

    def test_missing_correlation_and_flush_boundary(self):
        trace = actual_trace()
        trace["traceEvents"][3]["args"].pop("correlation")
        self.assertIn(
            "missing_cuda_correlation", validate_kineto(trace, steps(), 3)["reasons"]
        )
        trace = actual_trace()
        trace["traceEvents"][-1]["ts"] = 1652691328180.0
        result = validate_kineto(trace, steps(), 3)
        self.assertEqual(result["exclusions"]["3"], "flush_boundary")
        self.assertFalse(result["trace_valid"])

    def test_steps_outside_trace_do_not_count(self):
        self.assertIn(
            "insufficient_observed_steps",
            validate_kineto(
                actual_trace(),
                steps() + [dict(step_id=4, context_batch=1, complete=True)],
                4,
            )["reasons"],
        )

    def test_unknown_schema_and_missing_trace_end_fail_closed(self):
        trace = actual_trace()
        trace["schemaVersion"] = 999
        self.assertFalse(validate_kineto(trace, steps(), 3)["trace_valid"])
        trace = actual_trace()
        trace["traceEvents"].pop()
        self.assertFalse(validate_kineto(trace, steps(), 3)["trace_valid"])

    def test_stop_runs_while_wait_is_blocked_and_is_bounded(self):
        entered, release = threading.Event(), threading.Event()
        stopped = threading.Event()

        def wait():
            entered.set()
            release.wait(2)
            return {}

        def stop():
            stopped.set()
            release.wait(2)

        controller = ProfileController(
            lambda: None, wait, stop, wait_timeout_s=0.1, stop_timeout_s=0.05
        )
        controller.start()
        self.assertTrue(entered.wait(1))
        controller.request_stop()
        self.assertTrue(stopped.wait(1))
        result = controller.result(1)
        release.set()
        self.assertFalse(result["trace_valid"])
        self.assertTrue(result["control_errors"])

    def test_waiting_for_trace_does_not_block_sender(self):
        release = threading.Event()
        entered = threading.Event()

        def wait_capture():
            entered.set()
            if not release.wait(2):
                raise TimeoutError("fixture capture timeout")
            return {"trace_valid": True}

        controller = ProfileController(lambda: None, wait_capture, release.set)
        controller.start()
        self.assertTrue(entered.wait(1))
        sender = BoundedSender(1)
        handle = Future()
        self.assertTrue(sender.submit(1, lambda: handle))
        handle.set_result({"success": True})
        self.assertEqual(len(sender.collect()), 1)
        release.set()
        self.assertTrue(controller.result(1)["trace_valid"])

    def test_nonempty_trace_without_gpu_work_is_invalid(self):
        trace = {
            "traceEvents": [
                {
                    "name": "executor.model_forward",
                    "cat": "cpu_op",
                    "ph": "X",
                    "ts": 0,
                    "dur": 10,
                    "pid": 1,
                    "tid": 1,
                }
            ]
        }
        result = validate_kineto(trace, [], requested_steps=20)
        self.assertFalse(result["trace_valid"])
        self.assertIn("missing_gpu_kernels", result["reasons"])


def steps():
    return [
        dict(step_id=i, context_batch=1, decode_batch=0, complete=True)
        for i in (1, 2, 3)
    ]


def actual_trace():
    # Compact projection of kineto-cuda-schema-v1.json, PyTorch 2.11/cu130.
    # Preserve real timestamps, categories, CPU/GPU namespaces and correlations.
    values = [
        (
            1,
            1652691158900.723,
            168952.889,
            1652691158955.581,
            168885.321,
            1652691327624.188,
            58.041,
            1652691327680.280,
            42.880,
            1364,
        ),
        (
            2,
            1652691327960.095,
            154.582,
            1652691327987.552,
            122.944,
            1652691328088.032,
            11.248,
            1652691328101.273,
            42.145,
            1381,
        ),
        (
            3,
            1652691328152.240,
            33.630,
            1652691328157.810,
            24.540,
            1652691328170.765,
            5.131,
            1652691328179.002,
            42.368,
            1398,
        ),
    ]
    events = []
    for i, ts, dur, fts, fdur, rts, rdur, kts, kdur, corr in values:
        base = dict(ph="X", pid=644596, tid=644596)
        events.extend(
            [
                dict(
                    base,
                    cat="user_annotation",
                    name=f"pressure.step/{i}",
                    ts=ts,
                    dur=dur,
                ),
                dict(
                    base,
                    cat="user_annotation",
                    name="executor.model_forward",
                    ts=fts,
                    dur=fdur,
                ),
                dict(
                    base,
                    cat="cuda_driver",
                    name="cuLaunchKernelEx",
                    ts=rts,
                    dur=rdur,
                    args={"External id": i * 4, "correlation": corr},
                ),
                dict(
                    ph="X",
                    pid=0,
                    tid=7,
                    cat="kernel",
                    name="cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_128x64x16_1x1x1_3_nnn_align1_bias_f32_relu",
                    ts=kts,
                    dur=kdur,
                    args={
                        "External id": i * 4,
                        "device": 0,
                        "context": 1,
                        "stream": 7,
                        "correlation": corr,
                    },
                ),
            ]
        )
    events.append(
        dict(name="Record Window End", ph="i", ts=1652691328405.796, pid="", tid="")
    )
    return dict(
        schemaVersion=1,
        baseTimeNanoseconds=1790857026000000000,
        displayTimeUnit="ms",
        traceEvents=events,
    )


if __name__ == "__main__":
    unittest.main()

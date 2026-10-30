# Decode Performance Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 decode perf 重构为单一 typed execution pipeline，并为 fixed workload 增加 `input_lengths x output_lengths x batch` 网格，同时保留 distribution、DP、target-TPOT、inline profiling 和 multi-node 能力。

**Architecture:** 配置层产生互斥的 exact-batch/target-TPOT 类型，统一 planner 将 fixed 或 distribution 物化为 `DecodeCaseSpec`，策略层只决定测量哪些 batch，`DecodeBatchExecutor` 与 `DecodeHttpTransport` 负责如何测量。所有结果通过同一个 metrics/finalizer/report 路径落盘；本地和 externally managed multi-node runtime 只负责提供生命周期与 endpoint。

**Tech Stack:** Python 3、frozen dataclasses、`unittest`/`unittest.mock`、`requests`、现有 `ThreadPoolExecutor`、Bazel/Bazelisk、OpenSpec、RTP-LLM HTTP scheduler/profile endpoints。

**Spec:** `openspec/changes/refactor-decode-perf-test/design.md`；行为契约见 `openspec/changes/refactor-decode-perf-test/specs/`；实施追踪见 `openspec/changes/refactor-decode-perf-test/tasks.md`。

## Global Constraints

- Fixed decode 必须使用非空正整数 `output_lengths` 数组；即使只有一个值也不得回退到标量 `output_length`。
- Distribution decode 必须继续使用一个正整数标量 `output_length`，不得接受 `output_lengths`。
- `batch.strategy=fixed` 只接受 `sizes`；`batch.strategy=target_tpot` 只接受 `max_size`、`candidate_step`、`target_tpot_ms`。
- Planned output shape 是权威值；`generate_config` 中冲突的 `min_new_tokens`/`max_new_tokens` 不得改变 case output length。
- Exact fixed planner 生成 `input_lengths x output_lengths x batch.sizes`；target-TPOT planner 生成 `input_lengths x output_lengths`。
- Target-TPOT 顶层 metrics 必须来自 `best_batch_size`，不得强制用 `max_size` 作为最终 measurement。
- Batch size 始终表示 global batch；DP worker allocation 必须单独记录，并支持不整除与 idle worker。
- Inline profiling 属于 `run` 的可选后置阶段，默认关闭；profile 失败不得把有效 performance case 变为失败。
- Decode capability 仍只允许 `local + run`；decode resume、独立 profile/report 和 AOne decode 必须在副作用前失败。
- Multi-node launcher 继续拥有 server lifecycle；只有 master rank 执行 case 与写最终结果。
- 删除 `run_single`、`run_cartesian_grid`、`Decode_Result.json` 及旧 runner，不保留 alias、wrapper 或 dual-write。
- Prefill schema、execution、resume/profile/report 和 AOne behavior 不得改变。
- 不增加第三方运行时依赖；所有新增结构必须是 canonical JSON 可序列化值。

## File Structure

- Create `rtp_llm/test/perf_test/decode/model.py`: decode config policy、case、batch measurement、strategy outcome 的 frozen domain types 与 JSON identity 转换。
- Modify `rtp_llm/test/perf_test/config_schema.py`: 解析新 fixed/distribution batch policy、output contract 和 inline profiling 配置。
- Rewrite `rtp_llm/test/perf_test/decode/planner.py`: 统一 fixed/distribution planning、capacity validation 和 stable identity。
- Reduce `rtp_llm/test/perf_test/decode/distribution.py`: 只负责 materialized distribution 内容解析，不再执行 case。
- Create `rtp_llm/test/perf_test/decode/transport.py`: scheduler endpoint、profile endpoint、并发 HTTP round 和 response parsing。
- Create `rtp_llm/test/perf_test/decode/executor.py`: candidate batch measurement、DP validation、warmup/formal rounds、统一 case executor。
- Create `rtp_llm/test/perf_test/decode/strategy.py`: exact-batch 与 target-TPOT 选择逻辑。
- Create `rtp_llm/test/perf_test/decode/profiling.py`: selected-batch profile、trace flush、manifest persistence。
- Create `rtp_llm/test/perf_test/decode/report.py`: flat observations、exact matrices、TPOT matrices、distribution table。
- Modify `rtp_llm/test/perf_test/decode/metrics.py`: 统一 workload result aggregation，并以 selected measurement 为正式 metrics。
- Modify `rtp_llm/test/perf_test/decode/finalize.py`: result v2、summary、report、profile reference 收口。
- Modify `rtp_llm/test/perf_test/results.py`: result schema v2 和 decode profiling/search 校验。
- Modify `rtp_llm/test/perf_test/modes/decode.py`: 调用 `DecodeCaseExecutor.execute()`，保留逐 case 故障隔离和 checkpoint。
- Modify `rtp_llm/test/perf_test/execution/defaults.py`: 为本地 session 组装 transport/executor/profiler，不再选择 fixed/distribution runner。
- Modify `rtp_llm/test/perf_test/entry.py`: 所有 decode workload 走 `plan_decode_cases()`。
- Modify `rtp_llm/test/perf_test/multi_node/local_server_runner.py`: master 调用统一 plan/executor，worker 只等待。
- Modify `rtp_llm/test/perf_test/multi_node/multi_benchmark.py`, `multi_runner.sh`, `multi_local_executor.sh`: 传递 perf config 并收集标准 run directory。
- Modify `rtp_llm/test/perf_test/multi_node/analyzer/merge_decode_results.py`: 读取 `results/result.json`。
- Modify `rtp_llm/test/perf_test/multi_node/analyzer/batch_analyze.py`: 按 profile manifest 发现 trace 和 output length。
- Delete `rtp_llm/test/perf_test/decode/fixed.py`, `decode/grid_runner.py`, `decode/http_batch.py`, `batch_perf_impl.py`, `dataclass.py` after consumers migrate.
- Create focused tests: `decode_planner_test.py`, `decode_transport_test.py`, `decode_executor_test.py`, `decode_strategy_test.py`, `decode_profiling_test.py`, `decode_report_test.py`, `decode_multi_node_test.py`, `decode_multi_node_analyzer_test.py`.
- Modify existing config/result/default-executor/example/capability tests and `rtp_llm/test/perf_test/BUILD` with each owning task.

## Test Fixture Contract

Snippets below use small private helpers so each assertion stays focused. Define each helper in the test file where it is first used; when a later task modifies that same test file, reuse the existing helper rather than introducing a second representation. These are the canonical fixture bodies for Tasks 1-5:

```python
import copy
from dataclasses import asdict

from rtp_llm.test.perf_test.config_resolver import ResolvedConfig, resolve_config
from rtp_llm.test.perf_test.config_schema import parse_config
from rtp_llm.test.perf_test.decode.model import (
    DecodeBatchMeasurement,
    DecodeCaseSpec,
    DecodeProfilingConfig,
    ExactBatchPolicy,
    TargetTPOTPolicy,
)
from rtp_llm.test.perf_test.plan import case_id

TOPOLOGY = {"tp": 1, "dp": 1, "pp": 1, "ep": 1}

class _Response:
    def __init__(self, payload=None, *, status_code=200, text="ok"):
        self.status_code = status_code
        self.text = text
        self._payload = {} if payload is None else payload

    def json(self):
        return copy.deepcopy(self._payload)

def _exact_case(*, batch=2, warmup=0, runs=1, profile=False):
    return DecodeCaseSpec(
        workload_kind="grid",
        input_lengths=(128,) * batch,
        output_length=16,
        batch_policy=ExactBatchPolicy(strategy="fixed", size=batch),
        measure_runs=runs,
        warmup_runs=warmup,
        profiling=DecodeProfilingConfig(enabled=profile),
    )

def _tpot_case(*, max_size=16, step=4, target=10.0, profile=False):
    return DecodeCaseSpec(
        workload_kind="grid",
        input_lengths=(128,) * max_size,
        output_length=16,
        batch_policy=TargetTPOTPolicy(
            strategy="target_tpot",
            max_size=max_size,
            candidate_step=step,
            target_tpot_ms=target,
        ),
        measure_runs=1,
        warmup_runs=0,
        profiling=DecodeProfilingConfig(enabled=profile),
    )

def _distribution_case(*, batch=2):
    lengths = (64, 192)[:batch]
    return DecodeCaseSpec(
        workload_kind="distribution",
        input_lengths=lengths,
        output_length=16,
        batch_policy=ExactBatchPolicy(strategy="fixed", size=batch),
        measure_runs=1,
        warmup_runs=0,
        distribution={"source_kind": "sequence_lengths"},
    )

def _request_round(inputs, output, *, tpot=8.0):
    return tuple(
        {
            "request_index": index,
            "success": True,
            "input_length": input_length,
            "output_length": output,
            "latency_ms": {
                "wait": 1.0,
                "prefill": 2.0,
                "decode": tpot * max(output - 1, 0),
                "tpot": tpot,
                "total": 3.0 + tpot * max(output - 1, 0),
            },
        }
        for index, input_length in enumerate(inputs)
    )

def _measurement(*, batch, inputs=None, output=16, tpot=8.0):
    planned_inputs = tuple(inputs or ((128,) * batch))
    requests = _request_round(planned_inputs, output, tpot=tpot)
    return DecodeBatchMeasurement(
        batch_size=batch,
        input_lengths=planned_inputs,
        worker_allocations=({
            "worker_index": 0,
            "local_batch_size": batch,
            "request_indices": list(range(batch)),
            "input_lengths": list(planned_inputs),
        },),
        measurements=({
            "run": 0,
            "throughput_tokens_per_second": float(batch * output),
            "requests": list(requests),
        },),
    )

def _case_payload(spec):
    identity = spec.to_identity()
    return {"case_id": case_id("decode", identity), **identity}
```

Planner tests define source fixtures through the real parser, rather than constructing workload dataclasses manually:

```python
def _decode_source(*, kind="grid", inputs=(128,), outputs=(16,), batch=None):
    workload = {
        "kind": kind,
        "input_lengths": list(inputs),
        "output_lengths": list(outputs),
        "batch": batch or {"strategy": "fixed", "sizes": [1]},
        "measure_runs": 1,
        "warmup_runs": 0,
        "profiling": {"enabled": False},
    }
    return {
        "schema_version": 1,
        "model": {
            "label": "test",
            "type": "registered",
            "checkpoint": "/m",
            "tokenizer": "/m",
            "topology": {"tp": 1, "dp": 2, "pp": 1, "ep": 1},
        },
        "engine": {
            "max_seq_len": 4096,
            "max_batch_tokens": 8192,
            "concurrency_limit": 16,
        },
        "workload": workload,
        "execution": {"backend": "local", "devices": [0, 1]},
        "output": {},
        "regression": {},
        "publish": {},
    }

def _resolved_grid(*, inputs, outputs, sizes):
    source = _decode_source(
        inputs=inputs,
        outputs=outputs,
        batch={"strategy": "fixed", "sizes": sizes},
    )
    return parse_config(source), resolve_config(source)

def _resolved_tpot(*, inputs, outputs, max_size):
    source = _decode_source(
        inputs=inputs,
        outputs=outputs,
        batch={
            "strategy": "target_tpot",
            "max_size": max_size,
            "candidate_step": 4,
            "target_tpot_ms": 10.0,
        },
    )
    return parse_config(source), resolve_config(source)

def _resolved_distribution(*, distribution, max_size):
    source = _decode_source()
    source["workload"] = {
        "kind": "distribution",
        "distribution_file": "/tmp/requests.json",
        "output_length": 16,
        "batch": {
            "strategy": "target_tpot",
            "max_size": max_size,
            "candidate_step": 1,
            "target_tpot_ms": 10.0,
        },
        "measure_runs": 1,
        "warmup_runs": 0,
        "profiling": {"enabled": False},
    }
    config = parse_config(source)
    normalized = asdict(config)
    normalized["workload"].pop("distribution_file")
    normalized["workload"]["distribution"] = copy.deepcopy(distribution)
    resolved = ResolvedConfig(
        resolved_config=normalized,
        config_fingerprint="a" * 64,
        workload_fingerprint="b" * 64,
    )
    return config, resolved
```

For result/report tests, derive cases from `_measurement()` through `build_decode_case_result()` instead of hand-maintaining a second metrics schema. For filesystem tests, write JSON with `atomic_write_json()` and return the path. Do not add any helper that bypasses `parse_config()`, `build_execution_plan()`, `build_decode_case_result()`, or `parse_result_envelope()` at the boundary being tested.

---

### Task 1: Add Typed Decode Config And Unified Planning

**Files:**
- Create: `rtp_llm/test/perf_test/decode/model.py`
- Rewrite: `rtp_llm/test/perf_test/decode/planner.py`
- Modify: `rtp_llm/test/perf_test/decode/distribution.py`
- Modify: `rtp_llm/test/perf_test/config_schema.py`
- Modify: `rtp_llm/test/perf_test/entry.py`
- Modify: `rtp_llm/test/perf_test/unified_config_test.py`
- Create: `rtp_llm/test/perf_test/decode_planner_test.py`
- Modify: `rtp_llm/test/perf_test/BUILD`

**Interfaces:**
- Consumes: `EngineConfig`, resolved distribution content from `ResolvedConfig.resolved_config["workload"]`, existing `plan.case_id()` canonical identity semantics.
- Produces: `FixedBatchConfig`, `TargetTPOTConfig`, `DecodeProfilingConfig`, `ExactBatchPolicy`, `TargetTPOTPolicy`, `DecodeCaseSpec.to_identity()`, `DecodeCaseSpec.from_identity()`, `parse_distribution_inputs(distribution, batch) -> tuple[tuple[tuple[int, ...], str], ...]`, and `plan_decode_cases(workload, engine, *, dp_size, resolved_workload=None) -> tuple[dict[str, object], ...]`.

- [ ] **Step 1: Write failing config tests for the new strict protocol**

Replace the old scalar-output decode assertions in `unified_config_test.py` with exact policy checks:

```python
from rtp_llm.test.perf_test.decode.model import (
    DecodeProfilingConfig,
    FixedBatchConfig,
    TargetTPOTConfig,
)

def test_decode_grid_uses_output_axis_and_fixed_batch_policy(self):
    payload = _base_config()
    payload["workload"] = {
        "kind": "grid",
        "input_lengths": [128, 256],
        "output_lengths": [16, 64],
        "batch": {"strategy": "fixed", "sizes": [1, 4]},
        "profiling": {"enabled": True, "runs": 2, "steps": 3},
        "generate_config": {"top_k": 1},
    }
    workload = parse_config(payload).workload
    self.assertEqual((16, 64), workload.output_lengths)
    self.assertEqual(FixedBatchConfig(strategy="fixed", sizes=(1, 4)), workload.batch)
    self.assertEqual(
        DecodeProfilingConfig(enabled=True, runs=2, steps=3),
        workload.profiling,
    )

def test_decode_batch_strategies_are_mutually_exclusive(self):
    payload = _base_config()
    payload["workload"] = {
        "kind": "grid",
        "input_lengths": [128],
        "output_lengths": [16],
        "batch": {
            "strategy": "target_tpot",
            "sizes": [4],
            "max_size": 16,
            "candidate_step": 4,
            "target_tpot_ms": 10.0,
        },
    }
    with self.assertRaisesRegex(ConfigError, "workload.batch.sizes"):
        parse_config(payload)

def test_fixed_scalar_and_distribution_array_outputs_are_rejected(self):
    fixed = _base_config()
    fixed["workload"] = {
        "kind": "grid",
        "input_lengths": [128],
        "output_length": 16,
        "batch": {"strategy": "fixed", "sizes": [1]},
    }
    distribution = copy.deepcopy(fixed)
    distribution["workload"] = {
        "kind": "distribution",
        "distribution_file": "requests.json",
        "output_lengths": [16],
        "batch": {"strategy": "fixed", "sizes": [1]},
    }
    for payload, path in ((fixed, "workload.output_length"), (distribution, "workload.output_lengths")):
        with self.subTest(path=path):
            with self.assertRaisesRegex(ConfigError, path):
                parse_config(payload)
```

- [ ] **Step 2: Run the config tests and verify the old schema fails the new expectations**

Run: `bazelisk test //rtp_llm/test/perf_test:unified_config_test --test_output=errors`

Expected: FAIL because `decode.model` and `workload.output_lengths`/`workload.batch` are not defined.

- [ ] **Step 3: Implement exact domain and config types**

Create `decode/model.py` with these public structures and JSON conversions:

```python
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple, Union

@dataclass(frozen=True)
class FixedBatchConfig:
    strategy: str
    sizes: Tuple[int, ...]

@dataclass(frozen=True)
class TargetTPOTConfig:
    strategy: str
    max_size: int
    candidate_step: int
    target_tpot_ms: float

DecodeBatchConfig = Union[FixedBatchConfig, TargetTPOTConfig]

@dataclass(frozen=True)
class DecodeProfilingConfig:
    enabled: bool = False
    runs: int = 1
    steps: int = 3
    enable_all_ranks: bool = True
    flush_timeout_seconds: int = 60

@dataclass(frozen=True)
class ExactBatchPolicy:
    strategy: str
    size: int

@dataclass(frozen=True)
class TargetTPOTPolicy:
    strategy: str
    max_size: int
    candidate_step: int
    target_tpot_ms: float

CaseBatchPolicy = Union[ExactBatchPolicy, TargetTPOTPolicy]

@dataclass(frozen=True)
class DecodeCaseSpec:
    workload_kind: str
    input_lengths: Tuple[int, ...]
    output_length: int
    batch_policy: CaseBatchPolicy
    measure_runs: int
    warmup_runs: int
    generate_config: Dict[str, Any] = field(default_factory=dict)
    profiling: DecodeProfilingConfig = field(default_factory=DecodeProfilingConfig)
    distribution: Optional[Dict[str, Any]] = None

    def to_identity(self) -> Dict[str, Any]:
        return {
            "workload_kind": self.workload_kind,
            "input_lengths": list(self.input_lengths),
            "output_length": self.output_length,
            "batch_policy": asdict(self.batch_policy),
            "measure_runs": self.measure_runs,
            "warmup_runs": self.warmup_runs,
            "generate_config": copy.deepcopy(self.generate_config),
            "profiling": asdict(self.profiling),
            "distribution": copy.deepcopy(self.distribution),
        }

    @classmethod
    def from_identity(cls, value: Mapping[str, Any]) -> "DecodeCaseSpec":
        raw_policy = dict(value["batch_policy"])
        policy: CaseBatchPolicy
        if raw_policy["strategy"] == "fixed":
            policy = ExactBatchPolicy(strategy="fixed", size=int(raw_policy["size"]))
        elif raw_policy["strategy"] == "target_tpot":
            policy = TargetTPOTPolicy(
                strategy="target_tpot",
                max_size=int(raw_policy["max_size"]),
                candidate_step=int(raw_policy["candidate_step"]),
                target_tpot_ms=float(raw_policy["target_tpot_ms"]),
            )
        else:
            raise ValueError(
                f"unknown decode batch policy: {raw_policy['strategy']!r}"
            )
        return cls(
            workload_kind=str(value["workload_kind"]),
            input_lengths=tuple(int(item) for item in value["input_lengths"]),
            output_length=int(value["output_length"]),
            batch_policy=policy,
            measure_runs=int(value["measure_runs"]),
            warmup_runs=int(value["warmup_runs"]),
            generate_config=copy.deepcopy(dict(value["generate_config"])),
            profiling=DecodeProfilingConfig(**dict(value["profiling"])),
            distribution=None if value.get("distribution") is None else copy.deepcopy(dict(value["distribution"])),
        )
```

In `config_schema.py`, change the decode workload dataclasses to `output_lengths + batch + profiling` for grid and `output_length + batch + profiling` for distribution. Add `_parse_decode_batch()` and `_parse_decode_profiling()` that call `_reject_unknown()` with these exact field sets:

```python
def _parse_decode_batch(value: Any) -> DecodeBatchConfig:
    raw = _object("workload.batch", value)
    strategy = _non_empty_string(
        "workload.batch.strategy", _required("workload.batch", raw, "strategy")
    )
    if strategy == "fixed":
        _reject_unknown("workload.batch", raw, {"strategy", "sizes"})
        return FixedBatchConfig(
            strategy="fixed",
            sizes=_int_tuple("workload.batch.sizes", _required("workload.batch", raw, "sizes")),
        )
    if strategy == "target_tpot":
        _reject_unknown(
            "workload.batch",
            raw,
            {"strategy", "max_size", "candidate_step", "target_tpot_ms"},
        )
        return TargetTPOTConfig(
            strategy="target_tpot",
            max_size=_positive_int("workload.batch.max_size", _required("workload.batch", raw, "max_size")),
            candidate_step=_positive_int("workload.batch.candidate_step", _required("workload.batch", raw, "candidate_step")),
            target_tpot_ms=_positive_number("workload.batch.target_tpot_ms", _required("workload.batch", raw, "target_tpot_ms")),
        )
    raise ConfigError("workload.batch.strategy: expected one of: fixed, target_tpot")
```

Implement profiling parsing with the same strict field discipline:

```python
def _parse_decode_profiling(value: Any) -> DecodeProfilingConfig:
    defaults = DecodeProfilingConfig()
    if value is None:
        return defaults
    raw = _object("workload.profiling", value)
    fields = {
        "enabled", "runs", "steps", "enable_all_ranks", "flush_timeout_seconds"
    }
    _reject_unknown("workload.profiling", raw, fields)
    return DecodeProfilingConfig(
        enabled=_boolean(
            "workload.profiling.enabled", raw.get("enabled", defaults.enabled)
        ),
        runs=_positive_int(
            "workload.profiling.runs", raw.get("runs", defaults.runs)
        ),
        steps=_positive_int(
            "workload.profiling.steps", raw.get("steps", defaults.steps)
        ),
        enable_all_ranks=_boolean(
            "workload.profiling.enable_all_ranks",
            raw.get("enable_all_ranks", defaults.enable_all_ranks),
        ),
        flush_timeout_seconds=_positive_int(
            "workload.profiling.flush_timeout_seconds",
            raw.get("flush_timeout_seconds", defaults.flush_timeout_seconds),
        ),
    )
```

- [ ] **Step 4: Write failing planner tests for exact grid, TPOT grid, distribution prefixes, and capacity**

Create `decode_planner_test.py`:

```python
class DecodePlannerTest(unittest.TestCase):
    def test_exact_grid_is_input_by_output_by_batch(self):
        config, resolved = _resolved_grid(
            inputs=[128, 512], outputs=[16, 64], sizes=[1, 4]
        )
        cases = plan_decode_cases(
            config.workload,
            config.engine,
            dp_size=2,
            resolved_workload=resolved.resolved_config["workload"],
        )
        specs = [DecodeCaseSpec.from_identity(case) for case in cases]
        self.assertEqual(8, len(specs))
        self.assertEqual(
            {(128, 16, 1), (128, 16, 4), (128, 64, 1), (128, 64, 4),
             (512, 16, 1), (512, 16, 4), (512, 64, 1), (512, 64, 4)},
            {(item.input_lengths[0], item.output_length, item.batch_policy.size) for item in specs},
        )

    def test_tpot_grid_has_one_case_per_input_output_pair(self):
        config, resolved = _resolved_tpot(inputs=[128, 512], outputs=[16, 64], max_size=8)
        cases = plan_decode_cases(
            config.workload,
            config.engine,
            dp_size=1,
            resolved_workload=resolved.resolved_config["workload"],
        )
        specs = [DecodeCaseSpec.from_identity(case) for case in cases]
        self.assertEqual(4, len(specs))
        self.assertTrue(all(len(item.input_lengths) == 8 for item in specs))

    def test_case_ids_are_stable_and_distinguish_output_and_policy(self):
        fixed, fixed_resolved = _resolved_grid(
            inputs=[128], outputs=[16, 64], sizes=[4]
        )
        tpot, tpot_resolved = _resolved_tpot(
            inputs=[128], outputs=[16], max_size=4
        )
        fixed_identities = plan_decode_cases(
            fixed.workload,
            fixed.engine,
            dp_size=2,
            resolved_workload=fixed_resolved.resolved_config["workload"],
        )
        repeated = plan_decode_cases(
            fixed.workload,
            fixed.engine,
            dp_size=2,
            resolved_workload=fixed_resolved.resolved_config["workload"],
        )
        tpot_identity = plan_decode_cases(
            tpot.workload,
            tpot.engine,
            dp_size=2,
            resolved_workload=tpot_resolved.resolved_config["workload"],
        )[0]
        self.assertEqual(fixed_identities, repeated)
        ids = [case_id("decode", item) for item in fixed_identities]
        self.assertEqual(2, len(set(ids)))
        self.assertNotEqual(ids[0], case_id("decode", tpot_identity))

    def test_known_sequence_capacity_fails_during_planning(self):
        config, resolved = _resolved_grid(
            inputs=[4090], outputs=[16], sizes=[1]
        )
        with self.assertRaisesRegex(PlanningError, "max_seq_len"):
            plan_decode_cases(
                config.workload,
                config.engine,
                dp_size=2,
                resolved_workload=resolved.resolved_config["workload"],
            )

    def test_distribution_tpot_uses_sequence_prefixes_and_rejects_batch_map(self):
        config, resolved = _resolved_distribution(
            distribution={"sequence_lengths": [64, 96, 128, 160]}, max_size=4
        )
        case = plan_decode_cases(
            config.workload,
            config.engine,
            dp_size=2,
            resolved_workload=resolved.resolved_config["workload"],
        )[0]
        self.assertEqual([64, 96, 128, 160], case["input_lengths"])
        mapped_config, mapped_resolved = _resolved_distribution(
            distribution={"batch_seq_len_map": {"4": [64, 96, 128, 160]}},
            max_size=4,
        )
        with self.assertRaisesRegex(PlanningError, "batch_seq_len_map.*target_tpot"):
            plan_decode_cases(
                mapped_config.workload,
                mapped_config.engine,
                dp_size=2,
                resolved_workload=mapped_resolved.resolved_config["workload"],
            )
```

- [ ] **Step 5: Run planner tests and verify the unified planner is missing**

Add a `decode_planner_test` `py_test` target to `BUILD`.

Run: `bazelisk test //rtp_llm/test/perf_test:decode_planner_test --test_output=errors`

Expected: FAIL because `plan_decode_cases` and typed case identities do not exist.

- [ ] **Step 6: Implement the unified planner and switch entry planning**

Rename the existing distribution parser boundary and replace its batch handling with:

```python
def parse_distribution_inputs(distribution, batch):
    if isinstance(distribution, list):
        sequence_lengths = _lengths("distribution", distribution)
    elif isinstance(distribution, Mapping) and set(distribution) == {"sequence_lengths"}:
        sequence_lengths = _lengths(
            "distribution.sequence_lengths", distribution["sequence_lengths"]
        )
    else:
        sequence_lengths = ()
    if sequence_lengths:
        required = (
            max(batch.sizes)
            if isinstance(batch, FixedBatchConfig)
            else batch.max_size
        )
        if len(sequence_lengths) < required:
            raise PlanningError(
                "distribution does not contain enough sequence lengths"
            )
        sizes = batch.sizes if isinstance(batch, FixedBatchConfig) else (batch.max_size,)
        return tuple(
            (sequence_lengths[:size], "sequence_lengths") for size in sizes
        )
    if not isinstance(distribution, Mapping) or set(distribution) != {"batch_seq_len_map"}:
        raise PlanningError(
            "distribution must contain sequence_lengths or batch_seq_len_map"
        )
    if isinstance(batch, TargetTPOTConfig):
        raise PlanningError("batch_seq_len_map cannot be used with target_tpot")
    raw_map = distribution["batch_seq_len_map"]
    expected_keys = {str(size) for size in batch.sizes}
    if not isinstance(raw_map, Mapping) or set(raw_map) != expected_keys:
        raise PlanningError(
            "distribution batch map keys must exactly match fixed batch sizes"
        )
    result = []
    for size in batch.sizes:
        lengths = _lengths(
            f"distribution.batch_seq_len_map.{size}", raw_map[str(size)]
        )
        if len(lengths) != size:
            raise PlanningError(
                f"distribution batch {size} does not match its sequence count"
            )
        result.append((lengths, "batch_seq_len_map"))
    return tuple(result)
```

Implement the planner with these shared helpers and loop order:

```python
def _validate_capacity(inputs, output, engine, dp_size):
    if any(input_length + output > engine.max_seq_len for input_length in inputs):
        raise PlanningError("decode input plus output exceeds engine.max_seq_len")
    schedule = schedule_decode_requests(inputs, dp_size=dp_size)
    for worker in schedule.workers:
        if worker.local_batch_size > engine.concurrency_limit:
            raise PlanningError(
                "decode worker-local batch exceeds engine.concurrency_limit"
            )
        if sum(worker.input_lengths) > engine.max_batch_tokens:
            raise PlanningError(
                "decode worker-local input exceeds engine.max_batch_tokens"
            )

def _policy(batch, size=None):
    if isinstance(batch, FixedBatchConfig):
        assert size is not None
        return ExactBatchPolicy(strategy="fixed", size=size)
    return TargetTPOTPolicy(
        strategy="target_tpot",
        max_size=batch.max_size,
        candidate_step=batch.candidate_step,
        target_tpot_ms=batch.target_tpot_ms,
    )

def _spec(workload, inputs, output, policy, distribution=None):
    return DecodeCaseSpec(
        workload_kind=workload.kind,
        input_lengths=tuple(inputs),
        output_length=output,
        batch_policy=policy,
        measure_runs=workload.measure_runs,
        warmup_runs=workload.warmup_runs,
        generate_config=copy.deepcopy(workload.generate_config),
        profiling=workload.profiling,
        distribution=distribution,
    )

def plan_decode_cases(
    workload,
    engine,
    *,
    dp_size,
    resolved_workload=None,
):
    cases = []
    if isinstance(workload, DecodeGridWorkloadConfig):
        for input_length in workload.input_lengths:
            for output_length in workload.output_lengths:
                sizes = (
                    workload.batch.sizes
                    if isinstance(workload.batch, FixedBatchConfig)
                    else (workload.batch.max_size,)
                )
                for size in sizes:
                    inputs = (input_length,) * size
                    _validate_capacity(inputs, output_length, engine, dp_size)
                    cases.append(_spec(
                        workload,
                        inputs,
                        output_length,
                        _policy(workload.batch, size),
                    ))
    elif isinstance(workload, DecodeDistributionWorkloadConfig):
        if resolved_workload is None or "distribution" not in resolved_workload:
            raise PlanningError("resolved distribution content is required")
        materialized = parse_distribution_inputs(
            resolved_workload["distribution"], workload.batch
        )
        for inputs, source_kind in materialized:
            _validate_capacity(inputs, workload.output_length, engine, dp_size)
            size = len(inputs) if isinstance(workload.batch, FixedBatchConfig) else None
            cases.append(_spec(
                workload,
                inputs,
                workload.output_length,
                _policy(workload.batch, size),
                distribution={"source_kind": source_kind},
            ))
    else:
        raise PlanningError("decode planner requires grid or distribution workload")
    if not cases:
        raise PlanningError("decode workload generated no cases")
    return tuple(spec.to_identity() for spec in cases)
```

For target-TPOT grid, the single `size` above is only the materialized upper-bound input list; `_policy()` still records target-TPOT rather than an exact batch.

Use this exact entry dispatch:

```python
if isinstance(workload, (DecodeGridWorkloadConfig, DecodeDistributionWorkloadConfig)):
    return plan_decode_cases(
        workload,
        config.engine,
        dp_size=config.model.topology.dp,
        resolved_workload=resolved.resolved_config["workload"],
    )
```

Keep distribution parsing functions `_lengths()` and `parse_distribution_inputs()` in `decode/distribution.py`; remove its runner class only after Task 8 migrates all imports.

- [ ] **Step 7: Run config/planner/entry tests**

Run: `bazelisk test //rtp_llm/test/perf_test:unified_config_test //rtp_llm/test/perf_test:decode_planner_test //rtp_llm/test/perf_test:entry_execution_test --test_output=errors`

Expected: PASS; exact grid count is 8, TPOT count is 4, old scalar fixed output is rejected before execution.

- [ ] **Step 8: Commit the typed config and planner**

```bash
git add rtp_llm/test/perf_test/config_schema.py \
  rtp_llm/test/perf_test/decode/model.py \
  rtp_llm/test/perf_test/decode/planner.py \
  rtp_llm/test/perf_test/decode/distribution.py \
  rtp_llm/test/perf_test/entry.py \
  rtp_llm/test/perf_test/unified_config_test.py \
  rtp_llm/test/perf_test/decode_planner_test.py \
  rtp_llm/test/perf_test/BUILD
git commit -m "feat(perf): add typed decode workload planning"
```

### Task 2: Split HTTP Transport From Batch Measurement

**Files:**
- Create: `rtp_llm/test/perf_test/decode/transport.py`
- Create: `rtp_llm/test/perf_test/decode/executor.py`
- Modify: `rtp_llm/test/perf_test/decode/scheduler.py`
- Create: `rtp_llm/test/perf_test/decode_transport_test.py`
- Create: `rtp_llm/test/perf_test/decode_executor_test.py`
- Modify: `rtp_llm/test/perf_test/BUILD`

**Interfaces:**
- Consumes: `DecodeCaseSpec`, `EngineConfig`, `schedule_decode_requests(input_lengths, dp_size)`.
- Produces: `DecodeHttpTransport.configure_scheduler()`, `DecodeHttpTransport.run_round()`, `DecodeHttpTransport.start_profile()`, `DecodeCapacityError`, `DecodeBatchExecutor.measure(case, batch_size) -> DecodeBatchMeasurement`.

- [ ] **Step 1: Write failing transport tests for forced output shape and request preservation**

Create `decode_transport_test.py` with the existing `_Response` fixture and these assertions:

```python
def test_round_forces_planned_output_after_user_generate_config(self):
    post = Mock(return_value=_Response({
        "aux_info": {
            "input_len": 128,
            "output_len": 16,
            "wait_time": 1.0,
            "first_token_cost_time": 3.0,
            "cost_time": 18.0,
        }
    }))
    transport = DecodeHttpTransport(
        endpoint="http://127.0.0.1:18000",
        prompt_for_length=lambda length: f"prompt-{length}",
        post=post,
    )
    outcomes = transport.run_round(
        input_lengths=(128,),
        output_length=16,
        generate_config={"min_new_tokens": 2, "max_new_tokens": 2, "top_p": 0.8},
    )
    request = post.call_args.kwargs["json"]
    self.assertEqual(16, request["generate_config"]["min_new_tokens"])
    self.assertEqual(16, request["generate_config"]["max_new_tokens"])
    self.assertEqual(0.8, request["generate_config"]["top_p"])
    self.assertTrue(outcomes[0]["success"])

def test_round_preserves_http_failure_as_request_outcome(self):
    transport = DecodeHttpTransport(
        endpoint="http://127.0.0.1:18000",
        prompt_for_length=lambda _length: "prompt",
        post=Mock(return_value=_Response(status_code=503, text="not ready")),
    )
    outcome = transport.run_round(
        input_lengths=(128,), output_length=8, generate_config={}
    )[0]
    self.assertFalse(outcome["success"])
    self.assertIn("HTTP 503", outcome["error"])

def test_round_preserves_timeout_as_request_outcome(self):
    transport = DecodeHttpTransport(
        endpoint="http://127.0.0.1:18000",
        prompt_for_length=lambda _length: "prompt",
        post=Mock(side_effect=requests.Timeout("request timed out")),
    )
    outcome = transport.run_round(
        input_lengths=(128,), output_length=8, generate_config={}
    )[0]
    self.assertFalse(outcome["success"])
    self.assertIn("request timed out", outcome["error"])

def test_scheduler_retries_then_succeeds(self):
    post = Mock(side_effect=[
        _Response(status_code=503, text="warming"),
        _Response({"status": "ok"}),
    ])
    sleep = Mock()
    transport = DecodeHttpTransport(
        endpoint="http://127.0.0.1:18000",
        prompt_for_length=lambda _length: "prompt",
        post=post,
        sleep=sleep,
        scheduler_attempts=2,
    )
    transport.configure_scheduler(({"local_batch_size": 2},))
    self.assertEqual(2, post.call_count)
    sleep.assert_called_once_with(3)
```

- [ ] **Step 2: Run the transport test and confirm the new class is absent**

Run: `bazelisk test //rtp_llm/test/perf_test:decode_transport_test --test_output=errors`

Expected: FAIL because `decode.transport.DecodeHttpTransport` does not exist.

- [ ] **Step 3: Implement the narrow HTTP transport**

Move request parsing from `decode/http_batch.py` and use this concrete public skeleton; `_request()` is the current parser moved unchanged except for the forced generation-config order shown below:

```python
class DecodeHttpTransport:
    def __init__(
        self,
        endpoint: str,
        *,
        prompt_for_length: Callable[[int], str],
        post: Callable[..., Any] = requests.post,
        sleep: Callable[[float], None] = time.sleep,
        timeout_seconds: float = 300,
        scheduler_attempts: int = 20,
    ):
        self._endpoint = endpoint.rstrip("/")
        self._prompt_for_length = prompt_for_length
        self._post = post
        self._sleep = sleep
        self._timeout_seconds = timeout_seconds
        self._scheduler_attempts = scheduler_attempts

    def configure_scheduler(self, worker_allocations: Sequence[Mapping[str, Any]]) -> None:
        local_batch = max(int(item["local_batch_size"]) for item in worker_allocations)
        last_error = "unknown scheduler error"
        for attempt in range(self._scheduler_attempts):
            try:
                response = self._post(
                    f"{self._endpoint}/update_scheduler_info",
                    json={"batch_size": local_batch, "mode": "decode"},
                    timeout=60,
                )
                body = response.json() if response.status_code == 200 else {}
                if response.status_code == 200 and body.get("status", "ok") == "ok":
                    return
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
            except Exception as error:
                last_error = str(error)
            if attempt + 1 < self._scheduler_attempts:
                self._sleep(3)
        raise RuntimeError(
            f"failed to configure decode scheduler after "
            f"{self._scheduler_attempts} attempts: {last_error}"
        )

    def run_round(
        self,
        *,
        input_lengths: Sequence[int],
        output_length: int,
        generate_config: Mapping[str, Any],
    ) -> Tuple[Dict[str, Any], ...]:
        lengths = tuple(input_lengths)
        with ThreadPoolExecutor(max_workers=len(lengths)) as pool:
            futures = [
                pool.submit(
                    self._request,
                    index,
                    input_length,
                    output_length,
                    generate_config,
                )
                for index, input_length in enumerate(lengths)
            ]
            return tuple(future.result() for future in futures)

    def start_profile(
        self, *, trace_name: str, steps: int, enable_all_ranks: bool
    ) -> None:
        response = self._post(
            f"{self._endpoint}/start_profile",
            json={
                "gen_timeline": True,
                "trace_name": trace_name,
                "start_step": 0,
                "num_steps": steps,
                "enable_all_rank": enable_all_ranks,
            },
            timeout=60,
        )
        body = response.json() if response.status_code == 200 else {}
        if response.status_code != 200 or body.get("status", "ok") != "ok":
            raise RuntimeError(
                f"failed to start decode profile: HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )
```

Construct generation config in this order:

```python
config = copy.deepcopy(dict(generate_config))
config.update({
    "max_new_tokens": output_length,
    "min_new_tokens": output_length,
    "force_sp_accept": True,
    "top_k": config.get("top_k", 1),
})
```

`start_profile()` posts to `<endpoint>/start_profile` with `gen_timeline=True`, `trace_name`, `start_step=0`, `num_steps=steps`, and `enable_all_rank=enable_all_ranks`, and raises on non-200/non-ok response.

- [ ] **Step 4: Write failing batch-executor tests for warmup, elapsed throughput, and DP allocation**

Create `decode_executor_test.py`:

```python
def test_measure_configures_dp_then_runs_warmup_and_formal_rounds(self):
    transport = Mock()
    transport.run_round.side_effect = [
        _request_round((128, 128), 16),
        _request_round((128, 128), 16),
        _request_round((128, 128), 16),
    ]
    executor = DecodeBatchExecutor(
        transport,
        engine=EngineConfig(max_seq_len=512, max_batch_tokens=512, concurrency_limit=2),
        dp_size=2,
        monotonic=Mock(side_effect=[1.0, 1.1, 2.0, 2.2]),
    )
    measurement = executor.measure(_exact_case(batch=2, warmup=1, runs=2), 2)
    self.assertEqual(2, measurement.batch_size)
    self.assertEqual(2, len(measurement.measurements))
    self.assertEqual(3, transport.run_round.call_count)
    allocations = transport.configure_scheduler.call_args.args[0]
    self.assertEqual([1, 1], [item["local_batch_size"] for item in allocations])

def test_measure_supports_uneven_and_idle_dp_workers(self):
    for batch, dp, expected in ((10, 3, [4, 3, 3]), (2, 4, [1, 1, 0, 0])):
        with self.subTest(batch=batch, dp=dp):
            transport = Mock()
            transport.run_round.return_value = _request_round((128,) * batch, 16)
            executor = DecodeBatchExecutor(
                transport,
                engine=EngineConfig(
                    max_seq_len=512,
                    max_batch_tokens=5120,
                    concurrency_limit=10,
                ),
                dp_size=dp,
            )
            measurement = executor.measure(_exact_case(batch=batch), batch)
            self.assertEqual(expected, [item["local_batch_size"] for item in measurement.worker_allocations])
```

- [ ] **Step 5: Implement `DecodeBatchMeasurement` and `DecodeBatchExecutor.measure()`**

Add to `decode/model.py`:

```python
@dataclass(frozen=True)
class DecodeBatchMeasurement:
    batch_size: int
    input_lengths: Tuple[int, ...]
    worker_allocations: Tuple[Dict[str, Any], ...]
    measurements: Tuple[Dict[str, Any], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "input_lengths": list(self.input_lengths),
            "worker_allocations": copy.deepcopy(list(self.worker_allocations)),
            "measurements": copy.deepcopy(list(self.measurements)),
        }
```

Define `DecodeCapacityError(PlanningError)` in `decode/executor.py`. In `DecodeBatchExecutor.measure()`, slice `case.input_lengths[:batch_size]`; raise `DecodeCapacityError` when the candidate exceeds available inputs, a worker local batch exceeds `engine.concurrency_limit`, or a worker input sum exceeds `engine.max_batch_tokens`. Use this loop after validating the schedule:

```python
self._transport.configure_scheduler(
    tuple(worker.to_dict() for worker in schedule.workers)
)
for _ in range(case.warmup_runs):
    self._transport.run_round(
        input_lengths=inputs,
        output_length=case.output_length,
        generate_config=case.generate_config,
    )
measurements = []
for run_index in range(case.measure_runs):
    started = self._monotonic()
    requests = self._transport.run_round(
        input_lengths=inputs,
        output_length=case.output_length,
        generate_config=case.generate_config,
    )
    elapsed = self._monotonic() - started
    successful_tokens = sum(
        item["output_length"] for item in requests if item["success"]
    )
    measurements.append({
        "run": run_index,
        "throughput_tokens_per_second": (
            successful_tokens / elapsed if elapsed > 0 else 0.0
        ),
        "worker_allocations": [worker.to_dict() for worker in schedule.workers],
        "requests": copy.deepcopy(list(requests)),
    })
return DecodeBatchMeasurement(
    batch_size=batch_size,
    input_lengths=inputs,
    worker_allocations=tuple(worker.to_dict() for worker in schedule.workers),
    measurements=tuple(measurements),
)
```

- [ ] **Step 6: Run transport/executor/scheduler tests**

Run: `bazelisk test //rtp_llm/test/perf_test:decode_transport_test //rtp_llm/test/perf_test:decode_executor_test //rtp_llm/test/perf_test:decode_scheduler_test --test_output=errors`

Expected: PASS; `10/3` allocates `4/3/3`, `2/4` allocates `1/1/0/0`, and user generation config cannot change planned output length.

- [ ] **Step 7: Commit transport and batch execution**

```bash
git add rtp_llm/test/perf_test/decode/model.py \
  rtp_llm/test/perf_test/decode/transport.py \
  rtp_llm/test/perf_test/decode/executor.py \
  rtp_llm/test/perf_test/decode/scheduler.py \
  rtp_llm/test/perf_test/decode_transport_test.py \
  rtp_llm/test/perf_test/decode_executor_test.py \
  rtp_llm/test/perf_test/decode_scheduler_test.py \
  rtp_llm/test/perf_test/BUILD
git commit -m "refactor(perf): separate decode transport and measurement"
```

### Task 3: Implement Batch Strategies And The Unified Case Executor

**Files:**
- Create: `rtp_llm/test/perf_test/decode/strategy.py`
- Modify: `rtp_llm/test/perf_test/decode/executor.py`
- Modify: `rtp_llm/test/perf_test/decode/metrics.py`
- Modify: `rtp_llm/test/perf_test/modes/decode.py`
- Create: `rtp_llm/test/perf_test/decode_strategy_test.py`
- Modify: `rtp_llm/test/perf_test/decode_metrics_test.py`
- Modify: `rtp_llm/test/perf_test/BUILD`

**Interfaces:**
- Consumes: `DecodeBatchExecutor.measure(case, batch_size)`, `ExactBatchPolicy`, `TargetTPOTPolicy`.
- Produces: `DecodeStrategyOutcome`, `FixedBatchStrategy.execute()`, `TargetTPOTStrategy.execute()`, `strategy_for(policy)`, `DecodeCaseExecutor.execute(case: Mapping[str, Any]) -> dict[str, Any]`; `case` includes the planner identity plus its stable `case_id`.

- [ ] **Step 1: Write failing strategy tests that make selected-batch semantics explicit**

Create `decode_strategy_test.py`:

```python
def test_fixed_strategy_measures_exactly_once(self):
    measure = Mock(return_value=_measurement(batch=4, tpot=8.0))
    outcome = FixedBatchStrategy().execute(_exact_case(batch=4), measure)
    measure.assert_called_once()
    self.assertEqual(4, outcome.selected.batch_size)
    self.assertIsNone(outcome.search)

def test_tpot_promotes_best_batch_without_forcing_maximum(self):
    measured = []
    def measure(_case, batch_size):
        measured.append(batch_size)
        return _measurement(batch=batch_size, tpot=8.0 if batch_size <= 8 else 12.0)
    outcome = TargetTPOTStrategy().execute(
        _tpot_case(max_size=16, step=4, target=10.0), measure
    )
    self.assertEqual(8, outcome.selected.batch_size)
    self.assertEqual(len(measured), len(set(measured)))
    self.assertEqual("found", outcome.search["outcome"])
    self.assertEqual(8, outcome.search["best_batch_size"])

def test_tpot_failure_retains_observations_without_formal_measurement(self):
    outcome = TargetTPOTStrategy().execute(
        _tpot_case(max_size=8, step=4, target=1.0),
        lambda _case, batch: _measurement(batch=batch, tpot=20.0),
    )
    self.assertIsNone(outcome.selected)
    self.assertEqual("no_satisfying_candidate", outcome.search["outcome"])
    self.assertGreater(len(outcome.search["observations"]), 0)

def test_tpot_records_capacity_failure_and_keeps_best_lower_candidate(self):
    def measure(_case, batch):
        if batch >= 12:
            raise DecodeCapacityError("worker token capacity exceeded")
        return _measurement(batch=batch, tpot=8.0)
    outcome = TargetTPOTStrategy().execute(
        _tpot_case(max_size=16, step=4, target=10.0), measure
    )
    self.assertEqual(8, outcome.selected.batch_size)
    self.assertIn(
        "capacity_failed",
        [item["outcome"] for item in outcome.search["observations"]],
    )
```

- [ ] **Step 2: Run the strategy test and verify it fails before implementation**

Run: `bazelisk test //rtp_llm/test/perf_test:decode_strategy_test --test_output=errors`

Expected: FAIL because the strategy module and `DecodeStrategyOutcome` do not exist.

- [ ] **Step 3: Implement strategy outcomes and bounded search**

Add these types to `decode/model.py`:

```python
@dataclass(frozen=True)
class DecodeStrategyOutcome:
    selected: Optional[DecodeBatchMeasurement]
    search: Optional[Dict[str, Any]]
    error_code: Optional[str] = None
```

Implement the candidate list, measurement summary, and bounded search with this control flow:

```python
def _candidates(max_size: int, step: int) -> Tuple[int, ...]:
    values = [1]
    values.extend(range(step, max_size + 1, step))
    values.append(max_size)
    return tuple(sorted(set(values)))

def _error_text(error: Exception) -> str:
    return " ".join(str(error).splitlines())[:500]

def _summarize(measurement: DecodeBatchMeasurement, target: float) -> Dict[str, Any]:
    requests = [request for run in measurement.measurements
                for request in run["requests"]]
    successful = [request for request in requests if request["success"]]
    success_rate = len(successful) / len(requests) if requests else 0.0
    tpot = (
        statistics.fmean(request["latency_ms"]["tpot"] for request in successful)
        if successful else None
    )
    throughput = statistics.fmean(
        run["throughput_tokens_per_second"] for run in measurement.measurements
    )
    satisfied = success_rate == 1.0 and tpot is not None and tpot <= target
    return {
        "batch_size": measurement.batch_size,
        "outcome": "satisfied" if satisfied else "target_exceeded",
        "tpot_ms": tpot,
        "success_rate": success_rate,
        "throughput_tokens_per_second": throughput,
    }

class FixedBatchStrategy:
    def execute(self, case, measure):
        selected = measure(case, case.batch_policy.size)
        return DecodeStrategyOutcome(selected=selected, search=None)

class TargetTPOTStrategy:
    def execute(self, case, measure):
        policy = case.batch_policy
        candidates = _candidates(policy.max_size, policy.candidate_step)
        cache: Dict[int, DecodeBatchMeasurement] = {}
        observations = []
        best = None
        low, high = 0, len(candidates) - 1
        while low <= high:
            middle = (low + high) // 2
            batch_size = candidates[middle]
            try:
                if batch_size not in cache:
                    cache[batch_size] = measure(case, batch_size)
                selected = cache[batch_size]
                observation = _summarize(selected, policy.target_tpot_ms)
            except DecodeCapacityError as error:
                observation = {"batch_size": batch_size,
                               "outcome": "capacity_failed",
                               "error": _error_text(error)}
            except Exception as error:
                observation = {"batch_size": batch_size,
                               "outcome": "measurement_failed",
                               "error": _error_text(error)}
            observations.append(observation)
            if observation["outcome"] == "satisfied":
                best = cache[batch_size]
                low = middle + 1
            else:
                high = middle - 1
        search = {
            "target_tpot_ms": policy.target_tpot_ms,
            "max_size": policy.max_size,
            "candidate_step": policy.candidate_step,
            "outcome": "found" if best is not None else "no_satisfying_candidate",
            "best_batch_size": None if best is None else best.batch_size,
            "observations": observations,
        }
        return DecodeStrategyOutcome(
            selected=best,
            search=search,
            error_code=None if best is not None else "target_tpot_not_met",
        )
```

Do not re-run the selected candidate after search: the cached `DecodeBatchMeasurement` is the formal measurement.

Use this resolver:

```python
def strategy_for(policy: CaseBatchPolicy) -> Union[FixedBatchStrategy, TargetTPOTStrategy]:
    if isinstance(policy, ExactBatchPolicy):
        return FixedBatchStrategy()
    if isinstance(policy, TargetTPOTPolicy):
        return TargetTPOTStrategy()
    raise TypeError(f"unsupported decode batch policy: {type(policy).__name__}")
```

- [ ] **Step 4: Write a failing unified case-executor test**

Add to `decode_executor_test.py`:

```python
def test_case_executor_uses_one_path_for_grid_and_distribution(self):
    batch_executor = Mock()
    batch_executor.measure.side_effect = [
        _measurement(batch=2, inputs=(128, 128), output=16, tpot=8.0),
        _measurement(batch=2, inputs=(64, 192), output=16, tpot=9.0),
    ]
    executor = DecodeCaseExecutor(batch_executor, topology=TOPOLOGY)
    fixed = executor.execute(_case_payload(_exact_case(batch=2)))
    distribution = executor.execute(_case_payload(_distribution_case(batch=2)))
    self.assertEqual("ok", fixed["status"])
    self.assertEqual("ok", distribution["status"])
    self.assertIsNone(fixed["metrics"]["distribution"])
    self.assertEqual([64, 192], distribution["metrics"]["distribution"]["actual_sequence_lengths"])
```

- [ ] **Step 5: Implement `DecodeCaseExecutor` and selected-measurement metrics**

`DecodeCaseExecutor.execute()` separates `case_id` from the identity and uses this control flow:

```python
def execute(self, case: Mapping[str, Any]) -> Dict[str, Any]:
    identity = {name: copy.deepcopy(value) for name, value in case.items()
                if name != "case_id"}
    spec = DecodeCaseSpec.from_identity(identity)
    outcome = strategy_for(spec.batch_policy).execute(
        spec, self._batch_executor.measure
    )
    if outcome.selected is None:
        return {
            "case_id": case["case_id"],
            "identity": identity,
            "status": "target_tpot_not_met",
            "runs": spec.measure_runs,
            "errors": [{
                "code": outcome.error_code or "target_tpot_not_met",
                "message": "no candidate batch satisfied target_tpot_ms",
                "details": {"search": copy.deepcopy(outcome.search)},
            }],
            "metrics": None,
        }
    profiling = {"status": "disabled", "manifest_path": None, "error": None}
    return build_decode_case_result(
        case,
        selected=outcome.selected,
        topology=self._topology,
        search=outcome.search,
        profiling=profiling,
    )
```

Change `_planned_lengths()` in `metrics.py` to use `selected.input_lengths` instead of branching on workload kind. Metrics must use `selected.batch_size`, keep `search` and optional distribution metadata, validate every request input/output against the selected shape, and preserve all selected formal rounds. Define `successful_requests` and `failed_requests` across every formal round, so their sum equals `selected.batch_size * len(selected.measurements)`; `success_rate` in reports uses those totals.

Change `DecodeAdapter` from `runner.run_case(case)` to `executor.execute(case)`; keep its per-case exception conversion and checkpoint writes unchanged.

- [ ] **Step 6: Run strategy, executor, metrics, and adapter tests**

Run: `bazelisk test //rtp_llm/test/perf_test:decode_strategy_test //rtp_llm/test/perf_test:decode_executor_test //rtp_llm/test/perf_test:decode_metrics_test //rtp_llm/test/perf_test:decode_fixed_test //rtp_llm/test/perf_test:decode_distribution_test --test_output=errors`

Expected: PASS after updating old fixed/distribution tests to instantiate `DecodeCaseExecutor`; TPOT top-level batch is the selected batch and no maximum-batch fallback occurs.

- [ ] **Step 7: Commit the unified strategy and case path**

```bash
git add rtp_llm/test/perf_test/decode/model.py \
  rtp_llm/test/perf_test/decode/strategy.py \
  rtp_llm/test/perf_test/decode/executor.py \
  rtp_llm/test/perf_test/decode/metrics.py \
  rtp_llm/test/perf_test/modes/decode.py \
  rtp_llm/test/perf_test/decode_strategy_test.py \
  rtp_llm/test/perf_test/decode_executor_test.py \
  rtp_llm/test/perf_test/decode_metrics_test.py \
  rtp_llm/test/perf_test/decode_fixed_test.py \
  rtp_llm/test/perf_test/decode_distribution_test.py \
  rtp_llm/test/perf_test/BUILD
git commit -m "refactor(perf): unify decode batch strategies"
```

### Task 4: Preserve Inline Profiling As A Non-Fatal Post-Measurement Stage

**Files:**
- Create: `rtp_llm/test/perf_test/decode/profiling.py`
- Modify: `rtp_llm/test/perf_test/decode/transport.py`
- Modify: `rtp_llm/test/perf_test/decode/executor.py`
- Create: `rtp_llm/test/perf_test/decode_profiling_test.py`
- Modify: `rtp_llm/test/perf_test/BUILD`

**Interfaces:**
- Consumes: selected `DecodeBatchMeasurement`, `DecodeProfilingConfig`, `DecodeHttpTransport.start_profile()` and `run_round()`.
- Produces: `DecodeProfiler.profile(case, batch_size) -> dict[str, Any]` and `profiles/<case_id>/profile_manifest.json` with `disabled|completed|failed` status.

- [ ] **Step 1: Write failing profile-selection and failure-isolation tests**

Create `decode_profiling_test.py`; copy the relevant canonical fixtures from the Test Fixture Contract and import `replace` from `dataclasses`:

```python
def test_tpot_profiles_only_selected_batch(self):
    with tempfile.TemporaryDirectory() as directory:
        trace = Path(directory) / "trace.json"
        trace.write_text("{}", encoding="utf-8")
        batch_executor = Mock()
        profiler = DecodeProfiler(
            batch_executor,
            run_root=Path(directory),
            log_path=None,
            sleep=lambda _seconds: None,
            artifact_discoverer=lambda _log, _offset, _name: (str(trace),),
        )
        manifest = profiler.profile(_tpot_case(profile=True), batch_size=8)
        batch_executor.profile.assert_called_once()
        self.assertEqual(8, batch_executor.profile.call_args.args[1])
        self.assertEqual("completed", manifest["status"])
        self.assertTrue((Path(directory) / manifest["manifest_path"]).is_file())

def test_profile_failure_does_not_change_valid_performance_status(self):
    profiler = Mock()
    profiler.profile.return_value = {
        "status": "failed",
        "manifest_path": "profiles/decode-case/profile_manifest.json",
        "error": "trace flush timeout",
    }
    batch_executor = Mock()
    batch_executor.measure.return_value = _measurement(batch=4)
    executor = DecodeCaseExecutor(
        batch_executor, topology=TOPOLOGY, profiler=profiler
    )
    result = executor.execute(_case_payload(_exact_case(batch=4, profile=True)))
    self.assertEqual("ok", result["status"])
    self.assertEqual("failed", result["metrics"]["profiling"]["status"])
    self.assertIn("trace flush timeout", result["metrics"]["profiling"]["error"])

def test_disabled_profile_sends_no_profile_request(self):
    profiler = Mock()
    batch_executor = Mock()
    batch_executor.measure.return_value = _measurement(batch=1)
    result = DecodeCaseExecutor(
        batch_executor, topology=TOPOLOGY, profiler=profiler
    ).execute(_case_payload(_exact_case(batch=1, profile=False)))
    profiler.profile.assert_not_called()
    self.assertEqual("disabled", result["metrics"]["profiling"]["status"])

def test_flush_timeout_writes_failed_manifest(self):
    with tempfile.TemporaryDirectory() as directory:
        profiler = DecodeProfiler(
            Mock(),
            run_root=Path(directory),
            log_path=None,
            monotonic=Mock(side_effect=[0.0, 0.0, 2.0]),
            sleep=lambda _seconds: None,
            artifact_discoverer=lambda _log, _offset, _name: (),
        )
        case = replace(
            _exact_case(batch=1, profile=True),
            profiling=DecodeProfilingConfig(
                enabled=True, flush_timeout_seconds=1
            ),
        )
        manifest = profiler.profile(case, batch_size=1)
        self.assertEqual("failed", manifest["status"])
        self.assertIn("trace flush timeout", manifest["error"])
        self.assertTrue(
            (Path(directory) / manifest["manifest_path"]).is_file()
        )
```

- [ ] **Step 2: Run the profiling test and confirm the module is absent**

Run: `bazelisk test //rtp_llm/test/perf_test:decode_profiling_test --test_output=errors`

Expected: FAIL because `DecodeProfiler` and batch profile execution do not exist.

- [ ] **Step 3: Implement profile execution and atomic manifest writing**

Extract the candidate slicing/capacity checks from `measure()` into a private `_prepare(case, batch_size) -> tuple[tuple[int, ...], DecodeBatchSchedule]`, then add:

```python
def profile(
    self,
    case: DecodeCaseSpec,
    batch_size: int,
    *,
    trace_name: str,
    steps: int,
    enable_all_ranks: bool,
) -> None:
    inputs, schedule = self._prepare(case, batch_size)
    self._transport.configure_scheduler(
        tuple(worker.to_dict() for worker in schedule.workers)
    )
    self._transport.start_profile(
        trace_name=trace_name,
        steps=steps,
        enable_all_ranks=enable_all_ranks,
    )
    profile_config = copy.deepcopy(case.generate_config)
    profile_config.update({
        "gen_timeline": True,
        "profile_step": steps,
        "profile_trace_name": trace_name,
    })
    for _ in range(case.profiling.runs):
        self._transport.run_round(
            input_lengths=inputs,
            output_length=case.output_length,
            generate_config=profile_config,
        )
```

Implement `DecodeProfiler.profile()` with constructor-injected artifact discovery so tests do not depend on server logs:

```python
class DecodeProfiler:
    def __init__(
        self,
        batch_executor,
        *,
        run_root: Path,
        log_path: Optional[str],
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        artifact_discoverer: Callable[[Optional[str], int, str], Tuple[str, ...]]
            = discover_profile_artifacts,
    ):
        self._batch_executor = batch_executor
        self._run_root = run_root
        self._log_path = log_path
        self._monotonic = monotonic
        self._sleep = sleep
        self._artifact_discoverer = artifact_discoverer

    def profile(self, case: DecodeCaseSpec, batch_size: int) -> Dict[str, Any]:
        identity = case.to_identity()
        case_id_value = case_id("decode", identity)
        trace_name = (
            f"decode_in{case.input_lengths[0]}_out{case.output_length}_bs{batch_size}_"
            f"{case_id_value.rsplit('-', 1)[-1]}"
        )
        relative_manifest = Path("profiles") / case_id_value / "profile_manifest.json"
        manifest_path = self._run_root / relative_manifest
        log_offset = (
            Path(self._log_path).stat().st_size
            if self._log_path and Path(self._log_path).is_file()
            else 0
        )
        status = "completed"
        error_message = None
        artifacts: Tuple[str, ...] = ()
        try:
            self._batch_executor.profile(
                case,
                batch_size,
                trace_name=trace_name,
                steps=case.profiling.steps,
                enable_all_ranks=case.profiling.enable_all_ranks,
            )
            deadline = self._monotonic() + case.profiling.flush_timeout_seconds
            while self._monotonic() < deadline:
                artifacts = self._artifact_discoverer(
                    self._log_path, log_offset, trace_name
                )
                if artifacts:
                    break
                self._sleep(0.2)
            if not artifacts:
                raise RuntimeError("trace flush timeout")
        except Exception as error:
            status = "failed"
            error_message = sanitize_error(error)
            artifacts = ()
        manifest = {
            "schema_version": 1,
            "status": status,
            "case_id": case_id_value,
            "batch_size": batch_size,
            "input_lengths": list(case.input_lengths[:batch_size]),
            "output_length": case.output_length,
            "runs": case.profiling.runs,
            "steps": case.profiling.steps,
            "enable_all_ranks": case.profiling.enable_all_ranks,
            "trace_name": trace_name,
            "artifacts": list(artifacts),
            "manifest_path": str(relative_manifest),
            "error": error_message,
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(manifest_path, manifest)
        return manifest
```

Use these helpers above `DecodeProfiler`; the C++ logger emits `profiler trace saved: <file>`:

```python
def discover_profile_artifacts(
    log_path: Optional[str], offset: int, trace_name: str
) -> Tuple[str, ...]:
    if not log_path or not Path(log_path).is_file():
        return ()
    with Path(log_path).open("r", encoding="utf-8", errors="replace") as stream:
        stream.seek(offset)
        lines = stream.readlines()
    marker = "profiler trace saved:"
    paths = []
    for line in lines:
        if marker not in line:
            continue
        candidate = line.split(marker, 1)[1].strip()
        if trace_name in candidate and Path(candidate).is_file():
            paths.append(str(Path(candidate).resolve()))
    return tuple(sorted(set(paths)))

def sanitize_error(error: Exception) -> str:
    return " ".join(str(error).splitlines())[:500]
```

The bounded loop above is the only flush wait; failure returns a failed manifest rather than raising.

- [ ] **Step 4: Integrate profiling after strategy selection**

Extend `DecodeCaseExecutor.__init__` with `profiler: Optional[DecodeProfiler] = None`. In the successful branch from Task 3, replace the fixed disabled value with:

```python
profiling = {"status": "disabled", "manifest_path": None, "error": None}
if spec.profiling.enabled:
    if self._profiler is None:
        raise RuntimeError("decode profiling is enabled but no profiler is configured")
    profiling = self._profiler.profile(spec, outcome.selected.batch_size)
```

`DecodeProfiler.profile()` catches every ordinary profile/flush exception and returns a persisted failed manifest, so this call cannot downgrade valid formal metrics. For target-TPOT, use `outcome.selected.batch_size`; never invoke profiling from within the candidate measurement callback.

- [ ] **Step 5: Run profiling and strategy regression tests**

Run: `bazelisk test //rtp_llm/test/perf_test:decode_profiling_test //rtp_llm/test/perf_test:decode_strategy_test //rtp_llm/test/perf_test:decode_executor_test --test_output=errors`

Expected: PASS; one selected batch is profiled, search candidates are not, and a profile error leaves case status `ok`.

- [ ] **Step 6: Commit inline profiling**

```bash
git add rtp_llm/test/perf_test/decode/profiling.py \
  rtp_llm/test/perf_test/decode/transport.py \
  rtp_llm/test/perf_test/decode/executor.py \
  rtp_llm/test/perf_test/decode/metrics.py \
  rtp_llm/test/perf_test/decode_profiling_test.py \
  rtp_llm/test/perf_test/BUILD
git commit -m "feat(perf): retain selected-batch decode profiling"
```

### Task 5: Version Decode Results And Generate Matrix Reports

**Files:**
- Create: `rtp_llm/test/perf_test/decode/report.py`
- Modify: `rtp_llm/test/perf_test/decode/finalize.py`
- Modify: `rtp_llm/test/perf_test/results.py`
- Modify: `rtp_llm/test/perf_test/result_envelope_test.py`
- Modify: `rtp_llm/test/perf_test/decode_metrics_test.py`
- Create: `rtp_llm/test/perf_test/decode_report_test.py`
- Modify: `rtp_llm/test/perf_test/BUILD`

**Interfaces:**
- Consumes: committed decode case result dictionaries containing `identity`, selected metrics, `search`, `distribution`, and `profiling` under mode-specific metrics.
- Produces: `RESULT_SCHEMA_VERSION = 2`, `generate_decode_report(case_results) -> dict[str, Any]`, `reports/decode_report.json`, and decode summary field `profiling_failed_cases`.

- [ ] **Step 1: Write failing result-v2 tests**

In `result_envelope_test.py`, replace the old decode fixture with one exact and one TPOT case. Use this constructor so every newly required field is explicit:

```python
def _decode_case_v2(*, case_id, policy, batch, profile, search):
    inputs = [128] * batch
    return {
        "case_id": case_id,
        "identity": {
            "workload_kind": "grid",
            "input_lengths": inputs,
            "output_length": 16,
            "batch_policy": policy,
            "measure_runs": 1,
            "warmup_runs": 0,
            "generate_config": {},
            "profiling": {"enabled": profile["status"] != "disabled", "runs": 1,
                          "steps": 3, "enable_all_ranks": True,
                          "flush_timeout_seconds": 60},
            "distribution": None,
        },
        "status": "ok",
        "runs": 1,
        "errors": [],
        "metrics": {
            "workload_kind": "grid",
            "input_lengths": inputs,
            "output_length": 16,
            "global_batch_size": batch,
            "topology": {"tp": 1, "dp": 1, "pp": 1, "ep": 1},
            "successful_requests": batch,
            "failed_requests": 0,
            "latency_ms": {"wait": 1.0, "prefill": 2.0, "decode": 8.0,
                           "tpot": 1.0, "total": 11.0},
            "throughput_tokens_per_second": 1000.0,
            "measurements": [{"run": 0, "worker_allocations": [], "requests": []}],
            "search": search,
            "distribution": None,
            "profiling": profile,
        },
    }

def _decode_envelope_v2():
    value = _base_result("decode")
    value["schema_version"] = 2
    value["summary"] = {
        "planned_cases": 2,
        "successful_cases": 2,
        "failed_cases": 0,
        "profiling_failed_cases": 1,
    }
    value["cases"] = [
        _decode_case_v2(
            case_id="decode-exact",
            policy={"strategy": "fixed", "size": 4},
            batch=4,
            profile={"status": "disabled", "manifest_path": None, "error": None},
            search=None,
        ),
        _decode_case_v2(
            case_id="decode-tpot",
            policy={"strategy": "target_tpot", "max_size": 16,
                    "candidate_step": 4, "target_tpot_ms": 10.0},
            batch=8,
            profile={"status": "failed",
                     "manifest_path": "profiles/decode-tpot/profile_manifest.json",
                     "error": "trace flush timeout"},
            search={"target_tpot_ms": 10.0, "max_size": 16, "candidate_step": 4,
                    "outcome": "found", "best_batch_size": 8,
                    "observations": [{"batch_size": 8, "outcome": "satisfied"}]},
        ),
    ]
    return value
```

Then assert:

```python
def test_decode_v2_validates_selected_batch_search_and_profiling(self):
    envelope = _decode_envelope_v2()
    parsed = parse_result_envelope(envelope)
    exact, tpot = [case.metrics.to_dict() for case in parsed.cases]
    self.assertEqual(4, exact["global_batch_size"])
    self.assertEqual("disabled", exact["profiling"]["status"])
    self.assertEqual(8, tpot["global_batch_size"])
    self.assertEqual(8, tpot["search"]["best_batch_size"])
    self.assertEqual(1, parsed.summary["profiling_failed_cases"])

def test_decode_v2_rejects_shape_incomplete_identity(self):
    envelope = _decode_envelope_v2()
    del envelope["cases"][0]["identity"]["output_length"]
    with self.assertRaisesRegex(ResultSchemaError, "identity.output_length"):
        parse_result_envelope(envelope)
```

- [ ] **Step 2: Run the result test and verify v1 rejects the new fields**

Run: `bazelisk test //rtp_llm/test/perf_test:result_envelope_test --test_output=errors`

Expected: FAIL because schema version 2, decode profiling, and `profiling_failed_cases` are unknown.

- [ ] **Step 3: Implement result schema v2 without changing prefill metrics**

Set `RESULT_SCHEMA_VERSION = 2`. Add `profiling: Dict[str, Any]` to `DecodeMetrics`; require exact decode metric fields including profiling. Validate profiling status in `{"disabled", "completed", "failed"}` and require `manifest_path` for completed and failed attempts. Require decode identity fields `workload_kind`, `input_lengths`, `output_length`, `batch_policy`, `measure_runs`, `warmup_runs`, `generate_config`, `profiling`, and `distribution`.

Update decode metric count validation from `successful + failed == global_batch_size` to `successful + failed == global_batch_size * len(measurements)`, and require `len(measurements) == case.runs`. This resolves the existing single-round assumption while preserving every formal round.

Change `CaseResult.errors` to `Tuple[Dict[str, Any], ...]`. For `target_tpot_not_met`, validate `errors[0].details.search` with the same target/max/step/outcome/observation rules used for successful `metrics.search`; this is the only location for a failed search trajectory when `metrics` is `None`. Do not synthesize formal latency, throughput, or batch fields for a diagnostic candidate.

Parse summary fields conditionally:

```python
summary_fields = {"planned_cases", "successful_cases", "failed_cases"}
if mode == "decode":
    summary_fields.add("profiling_failed_cases")
_exact_fields("summary", summary, summary_fields)
```

Prefill output automatically writes schema version 2 through the shared constant but retains its current metrics and three summary counts.

- [ ] **Step 4: Write failing exact-grid, TPOT-grid, and diagnostic report tests**

Create `decode_report_test.py`. Copy the relevant canonical fixtures from the Test Fixture Contract and add these result builders:

```python
def _exact_cases(*, inputs, outputs, batches):
    cases = []
    for batch in batches:
        for input_length in inputs:
            for output_length in outputs:
                spec = replace(
                    _exact_case(batch=batch),
                    input_lengths=(input_length,) * batch,
                    output_length=output_length,
                )
                selected = _measurement(
                    batch=batch,
                    inputs=spec.input_lengths,
                    output=output_length,
                )
                cases.append(build_decode_case_result(
                    _case_payload(spec),
                    selected=selected,
                    topology=TOPOLOGY,
                    search=None,
                    profiling={"status": "disabled", "manifest_path": None,
                               "error": None},
                ))
    return cases

def _tpot_cases(*, inputs, outputs):
    cases = []
    for index, (input_length, output_length) in enumerate(
        itertools.product(inputs, outputs)
    ):
        selected_batch = (8, 12)[index]
        spec = replace(
            _tpot_case(max_size=16),
            input_lengths=(input_length,) * 16,
            output_length=output_length,
        )
        search = {
            "target_tpot_ms": 10.0,
            "max_size": 16,
            "candidate_step": 4,
            "outcome": "found",
            "best_batch_size": selected_batch,
            "observations": [{"batch_size": selected_batch,
                              "outcome": "satisfied"}],
        }
        cases.append(build_decode_case_result(
            _case_payload(spec),
            selected=_measurement(
                batch=selected_batch,
                inputs=(input_length,) * selected_batch,
                output=output_length,
            ),
            topology=TOPOLOGY,
            search=search,
            profiling={"status": "disabled", "manifest_path": None,
                       "error": None},
        ))
    return cases
```

Import `itertools` and `replace` from `dataclasses`, then add these assertions:

```python
def test_exact_report_groups_input_output_matrix_by_batch(self):
    report = generate_decode_report(_exact_cases(inputs=(128, 512), outputs=(16, 64), batches=(1, 4)))
    self.assertEqual("formal", report["report_kind"])
    self.assertEqual([1, 4], [matrix["batch_size"] for matrix in report["matrices"]])
    first = report["matrices"][0]
    self.assertEqual([128, 512], first["input_lengths"])
    self.assertEqual([16, 64], first["output_lengths"])
    self.assertEqual(4, len(first["cells"]))

def test_tpot_report_cells_expose_selected_batch(self):
    report = generate_decode_report(_tpot_cases(inputs=(128,), outputs=(16, 64)))
    self.assertEqual([8, 12], [cell["best_batch_size"] for cell in report["matrices"][0]["cells"]])

def test_invalid_case_makes_report_diagnostic_without_dropping_cell(self):
    cases = _exact_cases(inputs=(128,), outputs=(16,), batches=(1,))
    cases[0]["status"] = "invalid_shape"
    cases[0]["metrics"] = None
    report = generate_decode_report(cases)
    self.assertEqual("diagnostic", report["report_kind"])
    self.assertEqual("invalid_shape", report["matrices"][0]["cells"][0]["status"])
```

- [ ] **Step 5: Implement report derivation and finalization**

`generate_decode_report()` must produce:

```python
{
    "schema_version": 2,
    "mode": "decode",
    "report_kind": "formal" if all_valid else "diagnostic",
    "counts": {"planned": len(cases), "valid": valid, "excluded": excluded},
    "observations": observations,
    "matrices": matrices,
    "distribution_cases": distribution_rows,
}
```

Exact matrix keys are `(batch_size, input_length, output_length)`. TPOT matrix keys are `(input_length, output_length)`. Each cell retains `case_id`, `status`, `tpot_ms`, `throughput_tokens_per_second`, `total_latency_ms`, `success_rate`; TPOT cells additionally retain `best_batch_size` and `search_outcome`.

Change `finalize_decode_run()` to count failed profile manifests, include the decode-specific summary count, call `parse_result_envelope()` before writes, and atomically write the report produced from ordered cases.

- [ ] **Step 6: Run result/report/finalizer tests**

Run: `bazelisk test //rtp_llm/test/perf_test:result_envelope_test //rtp_llm/test/perf_test:decode_metrics_test //rtp_llm/test/perf_test:decode_report_test --test_output=errors`

Expected: PASS; exact reports have one matrix per batch, TPOT cells use best batch, partial grids are diagnostic, and prefill envelope tests remain unchanged except schema version.

- [ ] **Step 7: Commit result v2 and matrix reports**

```bash
git add rtp_llm/test/perf_test/results.py \
  rtp_llm/test/perf_test/decode/report.py \
  rtp_llm/test/perf_test/decode/finalize.py \
  rtp_llm/test/perf_test/result_envelope_test.py \
  rtp_llm/test/perf_test/decode_metrics_test.py \
  rtp_llm/test/perf_test/decode_report_test.py \
  rtp_llm/test/perf_test/BUILD
git commit -m "feat(perf): report decode input output matrices"
```

### Task 6: Wire The Unified Local Decode Runtime And Documentation

**Files:**
- Modify: `rtp_llm/test/perf_test/execution/defaults.py`
- Modify: `rtp_llm/test/perf_test/default_executor_test.py`
- Modify: `rtp_llm/test/perf_test/decode_examples_test.py`
- Modify: `rtp_llm/test/perf_test/capabilities_test.py`
- Modify: `rtp_llm/test/perf_test/examples/decode_grid_local.json`
- Modify: `rtp_llm/test/perf_test/examples/decode_tpot_local.json`
- Modify: `rtp_llm/test/perf_test/examples/decode_distribution_local.json`
- Modify: `rtp_llm/test/perf_test/docs/decode.md`
- Modify: `rtp_llm/test/perf_test/BUILD`

**Interfaces:**
- Consumes: local `EngineSession`, tokenizer query map, `DecodeHttpTransport`, `DecodeBatchExecutor`, `DecodeCaseExecutor`, `DecodeProfiler`.
- Produces: one-session local decode execution and updated documented configs/dry runs; capability matrix remains unchanged.

- [ ] **Step 1: Write a failing default-executor decode integration test**

Add to `default_executor_test.py` a decode source/plan helper. Reuse the file's existing `_Session` and `_Tokenizer` fixtures and copy `_request_round()` plus `_measurement()` from the Test Fixture Contract:

```python
def _decode_source_and_plan(root):
    source = _source(root.parent)
    source["model"]["topology"] = {"tp": 1, "dp": 1, "pp": 1, "ep": 1}
    source["workload"] = {
        "kind": "grid",
        "input_lengths": [32],
        "output_lengths": [8],
        "batch": {"strategy": "fixed", "sizes": [1]},
        "measure_runs": 1,
        "warmup_runs": 0,
        "profiling": {"enabled": False},
    }
    resolved = resolve_config(source)
    parsed = parse_config(source)
    cases = plan_decode_cases(
        parsed.workload,
        parsed.engine,
        dp_size=1,
        resolved_workload=resolved.resolved_config["workload"],
    )
    return source, build_execution_plan(
        mode="decode",
        action="run",
        resolved=resolved,
        cases=cases,
        result_directory=str(root),
    )
```

Then inject the batch-executor factory:

```python
def test_decode_run_uses_one_session_and_unified_case_executor(self):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "run"
        source, plan = _decode_source_and_plan(root)
        initialize_run_directory(root, source_config=source, plan=plan, run_id="decode-local", code_version="test")
        session = _Session()
        session.log_path = None
        batch_executor = Mock()
        batch_executor.measure.return_value = _measurement(
            batch=1, inputs=(32,), output=8
        )
        executor = UnifiedLocalExecutor(
            session_factory=Mock(return_value=session),
            tokenizer_loader=lambda _plan: _Tokenizer(),
            query_builder=lambda _plan, lengths: {length: f"prompt-{length}" for length in lengths},
            decode_batch_executor_factory=Mock(return_value=batch_executor),
        )
        outcome = executor(plan)
        self.assertEqual(RunState.COMPLETED, outcome.state)
        self.assertEqual((1, 1), (session.entered, session.exited))
        self.assertTrue((root / "reports" / "decode_report.json").is_file())
```

- [ ] **Step 2: Run the integration test and confirm the factory seam is absent**

Run: `bazelisk test //rtp_llm/test/perf_test:default_executor_test --test_output=errors`

Expected: FAIL because `decode_batch_executor_factory` and unified decode assembly are not implemented.

- [ ] **Step 3: Replace runtime branching with unified assembly**

In `UnifiedLocalExecutor._decode_adapter()`, collect all distinct planned input lengths once, build the query map, and assemble inside the adapter factory so the active session is available:

```python
config = thaw(plan.resolved_config)
topology = config["model"]["topology"]
input_lengths = sorted({
    int(length)
    for planned_case in plan.cases
    for length in thaw(planned_case.identity)["input_lengths"]
})
queries = self._query_builder(plan, input_lengths)

def executor_factory(context):
    session = context.session
    transport = DecodeHttpTransport(
        endpoint=f"http://127.0.0.1:{session.client.port}",
        prompt_for_length=lambda length: queries[length],
    )
    batch_executor = self._decode_batch_executor_factory(
        transport,
        engine=EngineConfig(**config["engine"]),
        dp_size=int(topology["dp"]),
    )
    profiler = DecodeProfiler(
        batch_executor,
        run_root=Path(plan.result_directory),
        log_path=session.log_path,
    )
    return DecodeCaseExecutor(
        batch_executor,
        topology=topology,
        profiler=profiler,
    )

return DecodeAdapter(executor_factory)
```

Remove imports and factory selection for `FixedDecodeCaseRunner`, `DistributionDecodeCaseRunner`, and `DecodeHttpBatchRunner`. Keep the outer session/context/finalizer flow unchanged.

- [ ] **Step 4: Update all three decode examples and documentation**

Replace only the workload objects in the three example files with these exact values; leave their model, engine, execution, output, regression, and publish sections unchanged:

```json
// decode_grid_local.json
{"kind":"grid","input_lengths":[128,512,1024],"output_lengths":[32,64],
 "batch":{"strategy":"fixed","sizes":[1,3,8]},"measure_runs":3,
 "warmup_runs":1,"generate_config":{"top_k":1},
 "profiling":{"enabled":false}}

// decode_tpot_local.json
{"kind":"grid","input_lengths":[128,512,1024],"output_lengths":[64,128],
 "batch":{"strategy":"target_tpot","max_size":32,"candidate_step":4,
          "target_tpot_ms":10.0},"measure_runs":5,"warmup_runs":1,
 "generate_config":{"top_k":1},"profiling":{"enabled":false}}

// decode_distribution_local.json
{"kind":"distribution","distribution_file":"decode_distribution_requests.json",
 "output_length":64,"batch":{"strategy":"fixed","sizes":[2,5]},
 "measure_runs":3,"warmup_runs":1,"generate_config":{"top_k":1},
 "profiling":{"enabled":false}}
```

Update `docs/decode.md` to document input-by-output matrices, selected-batch TPOT metrics, inline profile settings, and the unchanged unsupported action list. Do not add an AOne command or decode follow-up action.

- [ ] **Step 5: Update docs/dry-run tests and verify capability failures remain side-effect free**

In `decode_examples_test.py`, import only `plan_decode_cases`, expect exact fixed count `len(inputs) * len(outputs) * len(sizes)`, and assert TPOT count `len(inputs) * len(outputs)`. Keep the existing four unsupported capability cases.

Run: `bazelisk test //rtp_llm/test/perf_test:default_executor_test //rtp_llm/test/perf_test:decode_examples_test //rtp_llm/test/perf_test:capabilities_test //rtp_llm/test/perf_test:unified_docs_test --test_output=errors`

Expected: PASS; all documented decode commands dry-run, only local run is admitted, and prefill tests stay green.

- [ ] **Step 6: Commit local integration and docs**

```bash
git add rtp_llm/test/perf_test/execution/defaults.py \
  rtp_llm/test/perf_test/default_executor_test.py \
  rtp_llm/test/perf_test/decode_examples_test.py \
  rtp_llm/test/perf_test/capabilities_test.py \
  rtp_llm/test/perf_test/examples/decode_grid_local.json \
  rtp_llm/test/perf_test/examples/decode_tpot_local.json \
  rtp_llm/test/perf_test/examples/decode_distribution_local.json \
  rtp_llm/test/perf_test/docs/decode.md \
  rtp_llm/test/perf_test/BUILD
git commit -m "feat(perf): wire unified local decode execution"
```

### Task 7: Move Multi-Node Decode Onto The Unified Plan

**Files:**
- Create: `rtp_llm/test/perf_test/multi_node/decode_runtime.py`
- Modify: `rtp_llm/test/perf_test/multi_node/local_server_runner.py`
- Modify: `rtp_llm/test/perf_test/multi_node/multi_benchmark.py`
- Modify: `rtp_llm/test/perf_test/multi_node/multi_benchmark_config.yaml`
- Modify: `rtp_llm/test/perf_test/multi_node/multi_local_executor.sh`
- Modify: `rtp_llm/test/perf_test/multi_node/multi_runner.sh`
- Create: `rtp_llm/test/perf_test/decode_multi_node_test.py`
- Modify: `rtp_llm/test/perf_test/BUILD`

**Interfaces:**
- Consumes: `PERF_TEST_CONFIG_PATH`, actual `PyEnvConfigs` topology, externally managed server port and prompt provider.
- Produces: `build_external_decode_plan(config_path, result_directory, actual_topology) -> tuple[ExecutionPlan, dict]`, `execute_master_decode(plan, *, source_config, endpoint, prompt_for_length, server_log_path)`, standard run directory collection; no `run_single` import.

- [ ] **Step 1: Write failing pure multi-node runtime tests**

Create `decode_multi_node_test.py` against the new pure helper rather than importing side-effectful `local_server_runner.py`. Copy `_decode_source()` from the Test Fixture Contract and add:

```python
def _write_decode_config(root, *, dp=2):
    source = _decode_source(
        inputs=(128,),
        outputs=(16, 64),
        batch={"strategy": "fixed", "sizes": [1, 4]},
    )
    source["model"]["topology"]["dp"] = dp
    source["execution"]["devices"] = list(range(dp))
    source["output"]["result_root"] = str(root / "results")
    path = root / f"decode-dp{dp}.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    return path
```

Use it in these tests:

```python
def test_external_plan_matches_local_case_identities(self):
    with tempfile.TemporaryDirectory() as directory:
        config_path = _write_decode_config(Path(directory))
        plan, source = build_external_decode_plan(
            config_path,
            result_directory=str(Path(directory) / "run"),
            actual_topology={"tp": 1, "dp": 2, "pp": 1, "ep": 1},
        )
        local_plan, _ = build_request_plan(
            CliRequest(mode="decode", action="run", config=str(config_path), run_dir=None, dry_run=False, cases=())
        )
        self.assertEqual(
            [case.case_id for case in local_plan.cases],
            [case.case_id for case in plan.cases],
        )
        self.assertEqual(source["workload"]["output_lengths"], [16, 64])

def test_external_plan_rejects_runtime_topology_mismatch(self):
    with tempfile.TemporaryDirectory() as directory:
        config_path = _write_decode_config(Path(directory), dp=2)
        with self.assertRaisesRegex(PlanningError, "runtime topology"):
            build_external_decode_plan(
                config_path,
                result_directory=str(Path(directory) / "run"),
                actual_topology={"tp": 1, "dp": 1, "pp": 1, "ep": 1},
            )

def test_worker_rank_never_executes_cases(self):
    executor = Mock()
    run_external_rank(world_rank=1, wait_for_master=Mock(), execute_master=executor)
    executor.assert_not_called()
```

- [ ] **Step 2: Run the multi-node test and confirm the helper module is absent**

Run: `bazelisk test //rtp_llm/test/perf_test:decode_multi_node_test --test_output=errors`

Expected: FAIL because `multi_node.decode_runtime` does not exist.

- [ ] **Step 3: Implement pure external-plan and rank-control helpers**

Implement the pure plan/rank boundary as follows; wrap JSON I/O failures in `ConfigurationError` using the same messages as `entry._source_config()`:

```python
def build_external_decode_plan(config_path, result_directory, actual_topology):
    path = Path(config_path)
    source = json.loads(path.read_text(encoding="utf-8"))
    resolved = load_resolved_config(path)
    config = parse_config(source)
    configured = config.model.topology.as_tuple()
    actual = tuple(int(actual_topology[name]) for name in ("tp", "dp", "pp", "ep"))
    if configured != actual:
        raise PlanningError(
            f"runtime topology {actual} does not match configured topology {configured}"
        )
    cases = plan_decode_cases(
        config.workload,
        config.engine,
        dp_size=config.model.topology.dp,
        resolved_workload=resolved.resolved_config["workload"],
    )
    plan = build_execution_plan(
        mode="decode",
        action="run",
        resolved=resolved,
        cases=cases,
        result_directory=str(result_directory),
    )
    return plan, source

def run_external_rank(*, world_rank, wait_for_master, execute_master):
    if world_rank == 0:
        return execute_master()
    wait_for_master()
    return None
```

Use the shared components directly, without wrapping the externally managed server in `EngineSession` or `LocalExecutor`:

```python
def execute_master_decode(
    plan,
    *,
    source_config,
    endpoint,
    prompt_for_length,
    server_log_path,
):
    root = Path(plan.result_directory)
    run_id = root.name
    initialize_run_directory(
        root,
        source_config=source_config,
        plan=plan,
        run_id=run_id,
        code_version=os.environ.get("BUILD_SCM_REVISION", "unknown"),
    )
    store = RunStore(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["state"] = "running"
    store.write_manifest(manifest)
    resolved = thaw(plan.resolved_config)
    engine = EngineConfig(**resolved["engine"])
    topology = resolved["model"]["topology"]
    transport = DecodeHttpTransport(
        endpoint=endpoint, prompt_for_length=prompt_for_length
    )
    batch_executor = DecodeBatchExecutor(
        transport, engine=engine, dp_size=int(topology["dp"])
    )
    profiler = DecodeProfiler(
        batch_executor, run_root=root, log_path=server_log_path
    )
    case_executor = DecodeCaseExecutor(
        batch_executor, topology=topology, profiler=profiler
    )
    try:
        result = DecodeAdapter(lambda _context: case_executor)(
            ModeExecutionContext(plan=plan, session=None, store=store)
        )
        finalize_decode_run(
            root,
            plan=plan,
            run_id=run_id,
            case_results=result.case_results,
        )
    except Exception:
        manifest["state"] = "failed"
        store.write_manifest(manifest)
        raise
    manifest["state"] = result.state.value
    store.write_manifest(manifest)
    return result
```

- [ ] **Step 4: Replace legacy environment grid parsing in `local_server_runner.py`**

Require `PERF_TEST_CONFIG_PATH`; remove `BATCH_SIZE_LIST`, `INPUT_LEN_LIST`, and `DECODE_TEST_LENGTH` parsing. Read planned input lengths to build `input_query_dict`, validate config topology against `py_env_configs.parallelism_config`, and call the pure rank helper. Keep current `MagaServerManager`, `wait_master_done()`, upload, process-group, and shutdown ownership.

Use this master call boundary:

```python
run_external_rank(
    world_rank=world_rank,
    wait_for_master=lambda: wait_master_done(world_rank=world_rank),
    execute_master=lambda: execute_master_decode(
        plan,
        source_config=source,
        endpoint=f"http://127.0.0.1:{port}",
        prompt_for_length=lambda length: input_query_dict[length],
        server_log_path=server.log_file_path,
    ),
)
```

- [ ] **Step 5: Change launcher config delivery and run-directory collection**

Add a required `perf_test_config` path to each decode benchmark in `multi_benchmark_config.yaml`. In `multi_benchmark.py`, resolve it to an absolute path, validate it by building the external plan before SSH, and export `PERF_TEST_CONFIG_PATH`. In `multi_runner.sh`, copy the config to a rank-stable remote path and set that path in the remote environment.

Replace the master-only `*Result.json` copy with recursive collection of:

```text
results/
reports/
profiles/
manifest.json
execution_plan.json
resolved_config.json
```

Preserve process logs and raw server trace collection separately.

- [ ] **Step 6: Run multi-node helper and shell syntax tests**

Run: `bazelisk test //rtp_llm/test/perf_test:decode_multi_node_test --test_output=errors`

Run: `bash -n rtp_llm/test/perf_test/multi_node/multi_runner.sh rtp_llm/test/perf_test/multi_node/multi_local_executor.sh`

Expected: both commands PASS; rank zero is the sole case writer and no active path imports `decode.grid_runner`.

- [ ] **Step 7: Commit multi-node runtime migration**

```bash
git add rtp_llm/test/perf_test/multi_node/decode_runtime.py \
  rtp_llm/test/perf_test/multi_node/local_server_runner.py \
  rtp_llm/test/perf_test/multi_node/multi_benchmark.py \
  rtp_llm/test/perf_test/multi_node/multi_benchmark_config.yaml \
  rtp_llm/test/perf_test/multi_node/multi_local_executor.sh \
  rtp_llm/test/perf_test/multi_node/multi_runner.sh \
  rtp_llm/test/perf_test/decode_multi_node_test.py \
  rtp_llm/test/perf_test/BUILD
git commit -m "refactor(perf): share decode core with multi node runs"
```

### Task 8: Migrate Result And Trace Analyzers, Then Remove Legacy Decode Code

**Files:**
- Modify: `rtp_llm/test/perf_test/multi_node/analyzer/merge_decode_results.py`
- Modify: `rtp_llm/test/perf_test/multi_node/analyzer/batch_analyze.py`
- Create: `rtp_llm/test/perf_test/decode_multi_node_analyzer_test.py`
- Delete: `rtp_llm/test/perf_test/decode/grid_runner.py`
- Delete: `rtp_llm/test/perf_test/decode/fixed.py`
- Delete: `rtp_llm/test/perf_test/decode/http_batch.py`
- Delete: `rtp_llm/test/perf_test/decode/tpot.py`
- Delete: `rtp_llm/test/perf_test/batch_perf_impl.py`
- Delete: `rtp_llm/test/perf_test/dataclass.py`
- Delete: `rtp_llm/test/perf_test/decode_grid_api_test.py`
- Delete: `rtp_llm/test/perf_test/decode_http_batch_test.py`
- Modify: `rtp_llm/test/perf_test/BUILD`

**Interfaces:**
- Consumes: collected standard run directories and `profiles/<case_id>/profile_manifest.json`.
- Produces: CSV rows with `input_length`, `output_length`, `batch_strategy`, `global_batch_size`, `best_batch_size`, latency/TPOT/throughput; trace configs keyed by DP/input/output/batch.

- [ ] **Step 1: Write failing analyzer fixtures for result v2 and profile manifests**

Create `decode_multi_node_analyzer_test.py`. Copy `_decode_envelope_v2()` from Task 5 and add these filesystem fixtures:

```python
def _write_collected_run(root, *, input_length, output_length, strategy, batch_size):
    task = root / "Task_dp_size-2"
    result_path = task / "results" / "result.json"
    result_path.parent.mkdir(parents=True)
    envelope = _decode_envelope_v2()
    envelope["cases"] = [envelope["cases"][1]]
    envelope["summary"] = {
        "planned_cases": 1,
        "successful_cases": 1,
        "failed_cases": 0,
        "profiling_failed_cases": 0,
    }
    case = envelope["cases"][0]
    case["identity"]["input_lengths"] = [input_length] * batch_size
    case["identity"]["output_length"] = output_length
    case["identity"]["batch_policy"] = {
        "strategy": strategy,
        "max_size": 16,
        "candidate_step": 4,
        "target_tpot_ms": 10.0,
    }
    case["metrics"]["input_lengths"] = [input_length] * batch_size
    case["metrics"]["output_length"] = output_length
    case["metrics"]["global_batch_size"] = batch_size
    case["metrics"]["successful_requests"] = batch_size
    case["metrics"]["search"]["best_batch_size"] = batch_size
    result_path.write_text(json.dumps(envelope), encoding="utf-8")
    return task

def _write_profile_manifest(root, *, input_length, output_length, batch):
    case_id_value = f"in{input_length}-out{output_length}-bs{batch}"
    trace = root / "Task_dp_size-2" / "trace_files" / f"{case_id_value}.json"
    trace.parent.mkdir(parents=True, exist_ok=True)
    trace.write_text("{}", encoding="utf-8")
    manifest_path = (
        root / "Task_dp_size-2" / "profiles" / case_id_value /
        "profile_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "case_id": case_id_value,
        "dp_size": 2,
        "input_lengths": [input_length] * batch,
        "output_length": output_length,
        "batch_size": batch,
        "trace_name": case_id_value,
        "artifacts": [str(trace)],
        "error": None,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path
```

Then add:

```python
def test_result_merger_reads_standard_decode_envelope(self):
    with tempfile.TemporaryDirectory() as directory:
        task = _write_collected_run(
            Path(directory),
            input_length=128,
            output_length=64,
            strategy="target_tpot",
            batch_size=8,
        )
        files = merge.scan_result_files(directory)
        rows = merge.merge_all_results(files)
        self.assertEqual(1, len(rows))
        metric = rows[0][1][0]
        self.assertEqual(64, metric["output_length"])
        self.assertEqual("target_tpot", metric["batch_strategy"])
        self.assertEqual(8, metric["best_batch_size"])

def test_trace_scan_uses_manifest_and_keeps_output_axis(self):
    with tempfile.TemporaryDirectory() as directory:
        _write_profile_manifest(Path(directory), input_length=128, output_length=16, batch=4)
        _write_profile_manifest(Path(directory), input_length=128, output_length=64, batch=4)
        configs = batch_analyze.scan_trace_files(directory)
        self.assertEqual(
            [(128, 16, 4), (128, 64, 4)],
            [(item.input_length, item.output_length, item.batch_size) for item in configs],
        )
```

- [ ] **Step 2: Run analyzer tests and verify old file discovery fails**

Run: `bazelisk test //rtp_llm/test/perf_test:decode_multi_node_analyzer_test --test_output=errors`

Expected: FAIL because merger scans `Decode_Result.json` and trace discovery depends on the old filename regex.

- [ ] **Step 3: Implement standard result and manifest discovery**

Change `scan_result_files()` to find `Task_*/results/result.json`. `merge_all_results()` must iterate `envelope["cases"]`, skip cases without metrics while preserving their status in diagnostics, and emit one normalized row per case. CSV fields must be:

```python
DECODE_FIELDS = [
    "input_length",
    "output_length",
    "batch_strategy",
    "global_batch_size",
    "best_batch_size",
    "success_rate",
    "wait_ms",
    "prefill_ms",
    "decode_ms",
    "tpot_ms",
    "total_ms",
    "throughput_tokens_per_second",
    "status",
]
```

Add a frozen `TraceConfig(dp_size, input_length, output_length, batch_size, trace_file)` record. Change trace discovery to load every `Task_*/profiles/*/profile_manifest.json`, require `status == "completed"`, derive DP from the `Task_dp_size-*` directory, and derive input/output/batch/artifact path only from the manifest. Do not parse shape from `normal_profiler_*.json` filenames.

- [ ] **Step 4: Run analyzer tests**

Run: `bazelisk test //rtp_llm/test/perf_test:decode_multi_node_analyzer_test --test_output=errors`

Expected: PASS; same input/batch with output 16 and 64 remains two trace configs.

- [ ] **Step 5: Delete legacy runners and remove every active reference**

Delete the listed modules and tests. Remove their sources from all three perf-test libraries and their old test targets from `BUILD`. Point remaining imports to `model`, `planner`, `strategy`, `transport`, and `executor`.

Run these searches:

```bash
rg -n "run_single|run_cartesian_grid|BatchPerfImpl|DecodeHttpBatchRunner|FixedDecodeCaseRunner|DistributionDecodeCaseRunner" rtp_llm/test/perf_test --glob '!openspec/**'
rg -n "Decode_Result\.json|Prefill_Result\.json" rtp_llm/test/perf_test/multi_node
```

Expected: no active matches. Historical OpenSpec archives are outside the searched tree.

- [ ] **Step 6: Run all focused decode tests after deletion**

Run:

```bash
bazelisk test \
  //rtp_llm/test/perf_test:unified_config_test \
  //rtp_llm/test/perf_test:decode_planner_test \
  //rtp_llm/test/perf_test:decode_transport_test \
  //rtp_llm/test/perf_test:decode_executor_test \
  //rtp_llm/test/perf_test:decode_strategy_test \
  //rtp_llm/test/perf_test:decode_metrics_test \
  //rtp_llm/test/perf_test:decode_profiling_test \
  //rtp_llm/test/perf_test:decode_report_test \
  //rtp_llm/test/perf_test:decode_examples_test \
  //rtp_llm/test/perf_test:decode_multi_node_test \
  //rtp_llm/test/perf_test:decode_multi_node_analyzer_test \
  //rtp_llm/test/perf_test:default_executor_test \
  //rtp_llm/test/perf_test:result_envelope_test \
  //rtp_llm/test/perf_test:capabilities_test \
  //rtp_llm/test/perf_test:entry_execution_test \
  //rtp_llm/test/perf_test:prefill_planner_test \
  //rtp_llm/test/perf_test:prefill_runner_test \
  //rtp_llm/test/perf_test:prefill_adapter_test \
  //rtp_llm/test/perf_test:prefill_resume_test \
  //rtp_llm/test/perf_test:prefill_profile_test \
  //rtp_llm/test/perf_test:prefill_report_test \
  //rtp_llm/test/perf_test:aone_package_test \
  //rtp_llm/test/perf_test:aone_sharding_test \
  //rtp_llm/test/perf_test:aone_worker_test \
  //rtp_llm/test/perf_test:aone_collector_test \
  //rtp_llm/test/perf_test:aone_orchestrator_test \
  --test_output=errors
```

Expected: PASS. If Bazel reaches the known unrelated `cache_store_rdma_impl` missing `RdmaMempool*.h` BUILD declaration, capture the exact failing target and also run the corresponding Python test files directly; do not describe the Bazel suite as passing until the external BUILD dependency is repaired.

- [ ] **Step 7: Commit analyzers and legacy removal**

```bash
git add -A rtp_llm/test/perf_test
git commit -m "refactor(perf): remove legacy decode grid runners"
```

Before committing, run `git diff --cached --name-status` and verify no user result directories or unrelated perf files are staged.

### Task 9: Run Dry-Run, GPU, Profile, And Specification Verification

**Files:**
- Create: `/tmp/decode-grid-qwen-0.5b.json` during verification only
- Create: `/tmp/decode-tpot-qwen-0.5b.json` during verification only
- Modify: `openspec/changes/refactor-decode-perf-test/tasks.md` only to mark tasks complete and attach evidence references

**Interfaces:**
- Consumes: final unified CLI and `/mnt/hf3fs/3fs/models/Qwen2.5-0.5B-Instruct/`.
- Produces: verified exact-grid result/report, verified TPOT selected-batch result, verified profile manifest/trace, strict OpenSpec pass, final scoped diff.

- [ ] **Step 1: Run exact and TPOT dry runs before allocating GPU**

Run:

```bash
./tools/perf_test --mode=decode --action=run \
  --config=rtp_llm/test/perf_test/examples/decode_grid_local.json --dry-run
./tools/perf_test --mode=decode --action=run \
  --config=rtp_llm/test/perf_test/examples/decode_tpot_local.json --dry-run
```

Expected: exact preview case count equals `input x output x sizes`; TPOT preview count equals `input x output`; neither command creates a formal run directory or model process.

- [ ] **Step 2: Create a minimal real-model exact-grid config**

Write `/tmp/decode-grid-qwen-0.5b.json` with this complete payload:

```json
{
  "schema_version": 1,
  "model": {
    "label": "qwen2.5-0.5b-decode-grid",
    "type": "qwen_2",
    "checkpoint": "/mnt/hf3fs/3fs/models/Qwen2.5-0.5B-Instruct/",
    "tokenizer": "/mnt/hf3fs/3fs/models/Qwen2.5-0.5B-Instruct/",
    "topology": {"tp": 1, "dp": 1, "pp": 1, "ep": 1}
  },
  "engine": {
    "max_seq_len": 256,
    "max_batch_tokens": 512,
    "concurrency_limit": 2,
    "args": {},
    "env": {}
  },
  "workload": {
    "kind": "grid",
    "input_lengths": [32, 64],
    "output_lengths": [4, 8],
    "batch": {"strategy": "fixed", "sizes": [1, 2]},
    "measure_runs": 1,
    "warmup_runs": 0,
    "generate_config": {"top_k": 1},
    "profiling": {"enabled": false}
  },
  "execution": {"backend": "local", "devices": [0]},
  "output": {"result_root": "/tmp/decode-perf-grid-results"},
  "regression": {},
  "publish": {}
}
```

- [ ] **Step 3: Run the real exact grid and inspect every matrix cell**

Run: `./tools/perf_test --mode=decode --action=run --config=/tmp/decode-grid-qwen-0.5b.json`

Expected: exit 0; result has 8 successful cases; each request actual shape equals its planned input/output; `decode_report.json` has batch matrices 1 and 2 with four cells each.

Inspect with:

```bash
jq '{status,summary,cases:[.cases[]|{status,input:.metrics.input_lengths[0],output:.metrics.output_length,batch:.metrics.global_batch_size}]}' \
  /tmp/decode-perf-grid-results/decode-*/results/result.json
jq '{report_kind,counts,matrices}' \
  /tmp/decode-perf-grid-results/decode-*/reports/decode_report.json
```

- [ ] **Step 4: Create and run a small TPOT plus inline-profile config**

Write `/tmp/decode-tpot-qwen-0.5b.json` with this complete payload:

```json
{
  "schema_version": 1,
  "model": {
    "label": "qwen2.5-0.5b-decode-tpot",
    "type": "qwen_2",
    "checkpoint": "/mnt/hf3fs/3fs/models/Qwen2.5-0.5B-Instruct/",
    "tokenizer": "/mnt/hf3fs/3fs/models/Qwen2.5-0.5B-Instruct/",
    "topology": {"tp": 1, "dp": 1, "pp": 1, "ep": 1}
  },
  "engine": {
    "max_seq_len": 256,
    "max_batch_tokens": 512,
    "concurrency_limit": 4,
    "args": {},
    "env": {}
  },
  "workload": {
    "kind": "grid",
    "input_lengths": [32],
    "output_lengths": [8],
    "batch": {
      "strategy": "target_tpot",
      "max_size": 4,
      "candidate_step": 1,
      "target_tpot_ms": 1000.0
    },
    "measure_runs": 1,
    "warmup_runs": 0,
    "generate_config": {"top_k": 1},
    "profiling": {
      "enabled": true,
      "runs": 1,
      "steps": 3,
      "enable_all_ranks": true,
      "flush_timeout_seconds": 60
    }
  },
  "execution": {"backend": "local", "devices": [0]},
  "output": {"result_root": "/tmp/decode-perf-tpot-results"},
  "regression": {},
  "publish": {}
}
```

Run: `./tools/perf_test --mode=decode --action=run --config=/tmp/decode-tpot-qwen-0.5b.json`

Expected: exit 0; search outcome is `found`; top-level global batch equals `best_batch_size`; only the selected batch has a profile attempt; the manifest records output length 8 and a completed trace artifact. If trace export fails, performance status remains `ok`, `profiling.status` is `failed`, and `profiling_failed_cases` is 1.

- [ ] **Step 5: Run full focused regression and strict OpenSpec validation**

Run the focused Bazel command from Task 8, then run:

```bash
openspec validate refactor-decode-perf-test --strict
git diff --check
git status --short
```

Expected: OpenSpec validation and whitespace check exit 0. Repository status contains only intended implementation/spec task updates plus pre-existing unrelated user files.

- [ ] **Step 6: Mark OpenSpec tasks with evidence and commit verification**

Update each completed checkbox in `openspec/changes/refactor-decode-perf-test/tasks.md`. For the integration verification group, include the exact Bazel/GPU result directories or blocker evidence in the task text without embedding generated result files in Git.

```bash
git add openspec/changes/refactor-decode-perf-test/tasks.md
git commit -m "test(perf): verify unified decode performance flow"
```

- [ ] **Step 7: Request code review before spec sync or archive**

Run `git log --oneline --decorate -10` and `git diff origin/main...HEAD --stat`, then invoke `superpowers:requesting-code-review`. Resolve findings and rerun affected tests before using `openspec-sync-specs`; do not archive from this implementation plan.

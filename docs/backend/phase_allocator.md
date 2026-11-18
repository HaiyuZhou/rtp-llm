# DSV4 phase allocator

DSV4 can use a private Torch CUDA allocation pool with 20 MiB pages while
loading and converting weights, then use the default pool with 1024 MiB pages
for runtime resources and inference. This reduces unused memory retained during
startup while preserving the runtime allocator configuration. The feature is
experimental, opt-in, and disabled by default.

## Requirements and supported paths

- CUDA with the native Torch caching allocator, one CUDA device per process.
- A Torch build supporting `large_segment_size_mb`, `torch.cuda.MemPool`, and
  the native allocation-to-pool APIs used by `rtp_llm/utils/phase_allocator.py`.
  A generic upstream Torch installation is not sufficient evidence of support.
- Matching Python code and rebuilt RTP-LLM native libraries containing the
  `EngineModelStartup` bridge. Replacing only the Python files is insufficient.
- A MegaMoE or MegaMoE-SE executor backend. A backend that creates no
  `MegaMoeExecutor` is rejected during runtime initialization, after weight
  conversion; select the supported backend before starting the engine.
- DSV4 with PP1 and CUDA Graph disabled. Other models, DSpARK, other speculative
  model types, pipeline parallel startup and CUDA Graph are outside the current
  support scope.

| Path | Validation status |
| --- | --- |
| DSV4 without MTP, TP8/EP8, CP ALL_GATHER, PDFUSION | Full generation and allocator A/B tested |
| DSV4 ordinary MTP, TP8/EP8, CP ALL_GATHER, PREFILL | Target/draft startup, warmup on/off and prefill tested |
| MTP with CP and PDFUSION or DECODE | Rejected before preparation/KV; full decode layout is unsupported |
| MTP full multistep decode and acceptance | Not validated |
| Other parallel layouts and runtime backends | Not certified by these results |

CP MTP prefill requires synchronous COMPACT draft inputs. For the tested
multistep configuration, set both `RTP_LLM_STREAM_ASYNC=0` and
`RTP_LLM_MTP_ASYNC_DEVICE_STATE=0`.

## Enable and disable

Set these variables before starting a fresh engine process:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,large_segment_size_mb:1024
export DSV4_PHASE_ALLOCATOR=1
```

If `PYTORCH_ALLOC_CONF` is also set, keep it consistent with
`PYTORCH_CUDA_ALLOC_CONF`. Keep the existing model/checkpoint and parallelism
arguments, with `--pp_size 1 --enable_cuda_graph 0`. For the ordinary MTP CP
path, use the PREFILL role and the synchronous settings above.

Enable the feature before the first default large CUDA allocation. Startup
rejects preexisting default large segments because their original page
granularity cannot be verified. Do not switch the feature on or off in a live
engine process.

To return to the fixed allocator baseline, restart with
`DSV4_PHASE_ALLOCATOR=0` and keep the 1024 MiB allocator setting. Existing
checkpoint files need no conversion. Both `--warm_up 1` and `--warm_up 0` have
been tested in the MTP PREFILL profile.

## Engine lifecycle

1. `BaseModel.load` covers checkpoint loading and Python model construction in
   the private 20 MiB pool, including allocations from loading worker threads.
2. `RtpLLMOp.start` registers the actual target/draft model identities. Before
   warmup or cache-manager creation, C++ `NormalEngine` invokes the grouped
   Python startup coordinator under inference mode.
3. Every participant prepares and converts its weights in the same loading
   pool. Only after all participants finish is that pool retired once.
4. Each model initializes runtime resources and JIT work using the default
   1024 MiB domain. Group readiness is published only after all participants
   succeed.
5. Warmup may temporarily bind KV. Its Python KV references and temporary
   cache manager are released before production available-memory sampling.
   The native engine checks readiness again before production KV planning.
6. Ordinary `initialize` calls bind or rebind runtime/KV resources; they do not
   reconvert weights. First prefill/decode audits run before and after forward
   and are rearmed after rebinding. Target verification is audited as decode.

Resident weights remain in the retired loading domain with unchanged storage
addresses. Retirement reclaims unused loading-pool space; it does not move or
free live weights. Runtime and production KV allocations belong to the default
domain. A preparation failure prevents the group from becoming ready and
attempts to restore allocator settings and release loading routing.

The loading route and allocator settings are process-wide. Concurrent or
nested loading scopes are rejected; sequential loader/conversion scopes are
supported. Runtime transition is one-way, so reloading weights requires a new
engine process.

## Diagnostics and model integration

Set `DSV4_PHASE_ALLOCATOR_TRACE_DIR` to a writable directory to retain per-PID
allocator and startup JSONL records. Startup summaries are also logged with
`[MODEL_STARTUP]`. Check all ranks for the following:

- All target/draft conversions precede the single `loading.finalized` event.
- Each participant reaches `runtime.ready` and `engine.ready` before formal KV.
- Runtime settings remain 1024 MiB; retired loading-pool blocks receive no new
  allocations; registered resident weight storage remains unchanged.
- KV is bound in the default domain, and first prefill/decode audits succeed
  for the request types exercised by the deployment.

The shared lifecycle is in `rtp_llm/utils/model_startup.py`; DSV4 supplies
preparation, weight materialization, runtime initialization and tensor-inventory
hooks. New DSV4 code placed inside these hooks inherits the stage. Add runtime
resources to runtime initialization and expose their tensors through the
inventory hooks. Avoid creating runtime buffers during weight conversion.
Adding another model requires an adapter and validation; the common lifecycle
does not automatically enable unsupported models.

Audits retain storage metadata rather than Tensor references and do not take
snapshots on every steady-state token. Torch snapshots cannot infer the
business purpose of undeclared tensors, prove external allocator page sizes,
or enumerate every tensor hidden inside ordinary DeepGEMM communication
objects. These checks detect several allocation-domain errors, not every
possible future misuse.

Distributed `AUTO` loading performs a WORLD eligibility vote before choosing
fastsafetensors. Every rank must enter AUTO in the same order. Loading only a
subset of ranks must use explicit SCRATCH, as the existing last-PP-stage draft
path does. This loader correctness rule also applies with phase allocation
disabled.

## Validation and measured results

The validation used two four-GPU L20A nodes, TP8/EP8/DP1/PP1,
DeepSeek-V4-Pro, CP ALL_GATHER, Graph0, 8192 maximum sequence length, and
`reserver_runtime_mem_mb=28672`. The MTP PREFILL comparison kept MegaMoE-SE and
DeepEP enabled and used gamma3:

| Mode | Worst-rank KV budget (MiB) | Aggregate KV tokens | Startup including weights, master/worker (s) |
| --- | ---: | ---: | ---: |
| Fixed 1024 MiB, warmup on | 4241 | 2072576 | 245.63 / 234.52 |
| Phase allocator, warmup on | 20609 | 10935296 | 280.70 / 258.50 |
| Phase allocator, warmup off | 21399 | 11363328 | 244.64 / 242.93 |

Aggregate KV capacity is not a larger single-request context limit. Startup
includes target and draft weight loading; filesystem caching was uncontrolled.
The 4k/6k prefill first-response medians were 350.374/345.392 ms for fixed and
335.839/336.817 ms for phase with warmup. These are measurements of one profile,
not a guarantee of universal speedup or zero regression.

An independently rebuilt pre-change engine at
`8a7caa21c51cff4a74d7ac048f4b8f3c9d66b5c7`, with only dependency patch
`175460a4cf720a18a9882deb0d8c385c8e6b04ef`, was also compared under the same
TP8/checkpoint/runtime and frozen no-MTP requests. Against each modified mode,
all 32 complete generated sequences, 49 response texts, and eight strictly
comparable first-step full-vocabulary logits matched exactly (maximum absolute
logit difference 0). Eight later-step records lacked the full conditioning
trajectory and were excluded from that numerical claim.

This establishes scoped regression evidence, not absolute model quality.
The original engine and both modified modes scored 1/32 on strict answer
format and 12/32 on secondary final-number matching, with five responses at the
128-token limit. The historical CP4 golden smoke adapted to TP8 remained 0/6.
Neither the full official benchmark nor complete MTP decode/acceptance passed
acceptance through this work.

Long-input native TopK order/tie selection is also nondeterministic. An
external stable diagnostic reference, checked against native 4k controls,
produced identical complete final hidden-state hashes for 64 phase/fixed
target/draft pairs at 4k/6k. That reference is not installed in production.

## Targeted checks

With the matching runtime and built native libraries available, the CPU
coordinator/registration tests can be run from the repository root:

```bash
python -m unittest \
  rtp_llm.utils.test.model_startup_test \
  rtp_llm.utils.test.model_startup_group_test \
  rtp_llm.utils.test.engine_startup_registration_test
```

On an idle CUDA device with the required Torch extensions:

```bash
python -m unittest \
  rtp_llm.utils.test.phase_allocator_test \
  rtp_llm.utils.test.model_startup_cuda_test
```

The native bridge target is
`//rtp_llm/cpp/normal_engine/test:engine_model_startup_test`. Use the repository's
matching architecture configuration to build and run it. Full-model checks
must additionally verify actual library/module origins, all-rank lifecycle
order, KV budgets, complete startup time and matched-input output/performance
against the fixed 1024 MiB baseline.

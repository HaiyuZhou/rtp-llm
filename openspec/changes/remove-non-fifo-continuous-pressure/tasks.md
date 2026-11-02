# Tasks

## 1. FIFO-only resolution and boundaries

- [x] 1.1 Change `cache_perf.py` and `batch_decode_test.py` so every continuous invocation defaults to FIFO, rejects explicit `batch_decode`, and resolves automatic client concurrency from the FIFO/service limit; verify focused parser/profile tests cover preload and non-preload continuous commands.
- [x] 1.2 Enforce the same FIFO-only contract in direct `CacheGridRunner` construction and resume compatibility, including an actionable migration error for old non-FIFO checkpoints; verify direct-runner tests cannot bypass CLI validation.
- [x] 1.3 Preserve `pressure_mode=case` batch-decode selection and scheduler startup wiring; verify existing ordinary fixed-batch parser and server-argument tests still observe batch-decode behavior.

## 2. Continuous lifecycle simplification

- [x] 2.1 Remove continuous seed-phase scheduler batch switching and fixed-batch concurrency checks while retaining bounded FIFO seed submission; verify continuous seed tests assert no scheduler-control call.
- [x] 2.2 Remove continuous formal tail padding, padding submission/failure branches, and scheduler batch-size transitions so finite continuous mode submits exactly `cases * runs`; verify a non-divisible historical tail produces no extra requests.
- [x] 2.3 Simplify continuous metric and summary construction to unconditional FIFO semantics, retaining only neutral compatibility values where readers require them; verify results report FIFO, no effective scheduler batch size, and zero padding.

## 3. Compatibility, documentation, and integration verification

- [x] 3.1 Update continuous-pressure documentation and CLI help text to remove non-FIFO instructions and explain that fixed batching remains available only in ordinary case mode; verify repository search finds no supported continuous `batch_decode` example.
- [x] 3.2 Run the focused cache-perf/cache-grid continuous suites and `rtp_llm.test.perf_test.batch_decode_test_test`; verify FIFO continuous, full-pressure preload, mock-memory policy wiring, resume rejection, and ordinary fixed-batch cases all pass.
- [x] 3.3 Run `openspec validate remove-non-fifo-continuous-pressure --strict`, Python formatting/static checks for touched files, and `git diff --check`; verify no unrelated working-tree files are modified.

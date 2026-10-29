# Local decode performance tests

Decode supports local `run` only. Every workload has one positive scalar
`output_length`; output-length arrays and ranges are rejected. The run itself
writes the common result, summary, and basic decode report, so there is no
separate decode report action.

## Fixed grid

The fixed grid is `input_lengths` by global `batch_sizes`. With DP greater than
one, every global batch is deterministically divided across workers and the
result records both the global batch and worker-local allocation. Uneven batches
are supported.

```bash
./tools/perf_test \
  --mode=decode \
  --action=run \
  --config=rtp_llm/test/perf_test/examples/decode_grid_local.json \
  --dry-run
```

## Distribution

The submit process reads and validates the small distribution file, then embeds
its content in the resolved workload. The remote source path is not retained.
`sequence_lengths` must contain enough positive lengths for the largest global
batch. Each result preserves the actual lengths and all request outcomes.

```bash
./tools/perf_test \
  --mode=decode \
  --action=run \
  --config=rtp_llm/test/perf_test/examples/decode_distribution_local.json \
  --dry-run
```

## Target TPOT

Set `target_tpot_ms` to search for the largest usable global batch that meets
the limit. The largest configured `batch_sizes` value is the bounded search
maximum. Every attempted batch, capacity or measurement failure, TPOT value,
success rate, and final selection is retained in the result.

```bash
./tools/perf_test \
  --mode=decode \
  --action=run \
  --config=rtp_llm/test/perf_test/examples/decode_tpot_local.json \
  --dry-run
```

Decode `resume`, `profile`, and `report` are unsupported. AOne decode is also
unsupported. These combinations return configuration exit code 2 before a run
directory, model process, build, or remote submission is created.

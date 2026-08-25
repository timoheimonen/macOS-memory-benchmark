# Apple M5 MacBook Air: LLM CPU Decode Working-Set Samples

These two `memory_benchmark` 0.63.0 runs provide a side-by-side view of CPU decode with contiguous KV on an Apple M5
MacBook Air. The second configuration scales both the weight mapping and context length by four, increasing the total
data mapping from 384 MiB to 1,536 MiB.

## Result files

- [256 MiB weights, 32,768-token context](apple_m5_cpu_decode_contiguous_weights_256mib_context_32768.json)
- [1,024 MiB weights, 131,072-token context](apple_m5_cpu_decode_contiguous_weights_1024mib_context_131072.json)

## Test setup

Both runs use the same backend, methodology, layer and head geometry, detected 10-worker policy, automatic calibration,
six-loop balanced scenario order, and seed. Equivalent command lines for collecting the same workload configurations
are:

```bash
memory_benchmark --llm-memory --weight-size-mb 256 --layers 8 \
  --query-heads 8 --kv-heads 2 --head-dim 64 --context-tokens 32768 \
  --count 6 --seed 42 \
  --output results/0.63.0/apple_m5_cpu_decode_contiguous_weights_256mib_context_32768.json

memory_benchmark --llm-memory --weight-size-mb 1024 --layers 8 \
  --query-heads 8 --kv-heads 2 --head-dim 64 --context-tokens 131072 \
  --count 6 --seed 42 \
  --output results/0.63.0/apple_m5_cpu_decode_contiguous_weights_1024mib_context_131072.json
```

## Working sets

| Configuration | 32,768-token sample | 131,072-token sample |
|---|---:|---:|
| Weight mapping | 256 MiB | 1,024 MiB |
| K + V mapping | 128 MiB | 512 MiB |
| Total data mapping | 384 MiB | 1,536 MiB |

## Effective model-payload bandwidth

All values are in GB/s. P50 is the benchmark headline, average is the arithmetic mean of the six measured loops, and
maximum is the highest value observed in those loops.

| Scenario | 384 MiB P50 | 384 MiB average | 384 MiB maximum | 1,536 MiB P50 | 1,536 MiB average | 1,536 MiB maximum |
|---|---:|---:|---:|---:|---:|---:|
| Weights only | 138.26 | 138.52 | 140.80 | 139.01 | 138.02 | 139.60 |
| KV only | 145.41 | 145.07 | 146.59 | 136.35 | 136.80 | 140.72 |
| Mixed | 136.19 | 136.07 | 137.03 | 138.05 | 137.79 | 140.30 |

The KV-only scenario shows the clearest difference between the two working sets: the 384 MiB sample averages
145.07 GB/s and reaches 146.59 GB/s, compared with 136.80 GB/s and 140.72 GB/s for the 1,536 MiB sample. These values
are the most relevant part of the pair when considering a possible SLC contribution. Weights-only averages remain
close, while the larger mixed workload records the higher average and maximum.

## Run quality and comparison scope

Both runs report `status: "complete"`, `results_complete: true`, and `conclusions_valid: true`. Start and end thermal
states are nominal, Low Power Mode is disabled, requested QoS is applied successfully, and neither run reports quality
warnings. All scenario distributions are marked stable: coefficients of variation range from 0.70% to 1.11% for the
384 MiB sample and from 1.44% to 2.33% for the 1,536 MiB sample.

Weight size and context length differ between the samples, so each has its own model geometry and frozen work plan.
They form a working-set scaling comparison rather than one pooled performance distribution.

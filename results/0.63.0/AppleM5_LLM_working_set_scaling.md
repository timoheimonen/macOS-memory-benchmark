# Apple M5 MacBook Air: LLM CPU Decode Working-Set Samples

These two `memory_benchmark` 0.63.0 runs are complete schema-1 examples for the CPU decode path with contiguous KV.
They use the same Apple M5 MacBook Air, backend, methodology, layer/head geometry, detected 10-worker policy, automatic
calibration, six-loop balanced scenario order, and seed. The weight size and context length both scale by four.

## Result files

- [256 MiB weights, 32,768-token context](apple_m5_cpu_decode_contiguous_weights_256mib_context_32768.json)
- [1,024 MiB weights, 131,072-token context](apple_m5_cpu_decode_contiguous_weights_1024mib_context_131072.json)

Both runs report `status: "complete"`, `results_complete: true`, `conclusions_valid: true`, nominal start/end thermal
state, Low Power Mode disabled, successful requested QoS, and no quality warnings.

The JSON files record these invocations before they were given their neutral archival names:

```bash
memory_benchmark --llm-memory --weight-size-mb 256 --layers 8 \
  --query-heads 8 --kv-heads 2 --head-dim 64 --context-tokens 32768 \
  --count 6 --seed 42 --output /tmp/output.json

memory_benchmark --llm-memory --weight-size-mb 1024 --layers 8 \
  --query-heads 8 --kv-heads 2 --head-dim 64 --context-tokens 131072 \
  --count 6 --seed 42 --output /tmp/output.json
```

## Configuration and headline results

The values below are median effective logical model-payload bandwidth.

| Configuration or scenario | 32,768-token sample | 131,072-token sample | Larger versus smaller |
|---|---:|---:|---:|
| Weight mapping | 256 MiB | 1,024 MiB | 4x |
| K + V mapping | 128 MiB | 512 MiB | 4x |
| Total data mapping | 384 MiB | 1,536 MiB | 4x |
| Weights-only bandwidth | 138.26 GB/s | 139.01 GB/s | +0.5% |
| KV-only bandwidth | 145.41 GB/s | 136.35 GB/s | -6.2% |
| Mixed bandwidth | 136.19 GB/s | 138.05 GB/s | +1.4% |

All six-loop scenario distributions are marked stable. Their coefficients of variation range from 0.70% to 1.11%
for the smaller sample and from 1.44% to 2.33% for the larger sample.

## Scope

This pair is useful as a current-version audit example and an empirical working-set scaling observation:

- Weight size and context differ, so the frozen work plans and model geometry are not identical. The runs must not be
  pooled as one performance distribution.
- The two sizes were captured as separate commands rather than an interleaved size sequence. Alternating sizes across
  repeated rounds would provide stronger evidence for a causal working-set comparison.

The nearly unchanged weights-only and mixed headline bandwidth, together with the 6.2% lower KV-only result at the
larger working set, describes the observed scaling across these two configurations.

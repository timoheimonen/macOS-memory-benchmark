# Apple M4 Mac mini vs. Apple M5 MacBook Air: LLM Memory Benchmark

## Benchmark command

```bash
memory_benchmark --llm-memory --weight-size-mb 256 --layers 8 \
  --query-heads 8 --kv-heads 2 --head-dim 64 --context-tokens 32768 \
  --count 3 --seed 42
```

> **Note:** The M4 Mac mini was not completely idle during the benchmark; a Docker container was running in the
> background.

## Results

### Effective model payload bandwidth at 32k context

```text
M4 theoretical maximum (120 GB/s)   | [========================================] 120.00 GB/s
M5 theoretical maximum (153.6 GB/s) | [==================================================>] 153.60 GB/s
------------------------------------+-------------------------------------------------------
Weights only — M4 Mac mini          | [====================================....] 108.54 GB/s
Weights only — M5 MacBook Air       | [===============================================>] 141.81 GB/s (+30.7%)
------------------------------------+-------------------------------------------------------
KV only — M4 Mac mini               | [======================================..] 115.70 GB/s
KV only — M5 MacBook Air            | [================================================>] 147.90 GB/s (+27.8%)
------------------------------------+-------------------------------------------------------
Mixed — M4 Mac mini                 | [====================================....] 108.34 GB/s (269 steps/s)
Mixed — M5 MacBook Air              | [==============================================.] 138.44 GB/s (344 steps/s) (+27.8%)
------------------------------------+-------------------------------------------------------
```

### Detailed comparison

Workload configuration: 256 MB weights and 128 MB KV cache.

| Scenario | M4 Mac mini latency | M4 bandwidth | M5 MacBook Air latency | M5 bandwidth | M5 advantage |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Weights only** | 2.473 ms | 108.54 GB/s | **1.893 ms** | **141.81 GB/s** | **+30.7%** |
| **KV only** | 1.160 ms | 115.70 GB/s | **0.907 ms** | **147.90 GB/s** | **+27.8%** |
| **Mixed** | 3.717 ms (269 steps/s) | 108.34 GB/s | **2.909 ms (344 steps/s)** | **138.44 GB/s** | **+27.8%** |

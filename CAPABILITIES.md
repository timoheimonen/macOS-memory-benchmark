# Measurement Capabilities

macOS-memory-benchmark / `memory_benchmark` is a low-level benchmark and analysis tool for characterizing memory-system behavior on macOS
running on Apple Silicon.

The tool is intended for practical microarchitectural investigation rather than abstract synthetic scoring. It measures
bandwidth, latency, access-pattern sensitivity, TLB behavior, and core-to-core cache-line handoff characteristics using
native ARM64 code paths.

## Main Memory and Cache Bandwidth

The standard benchmark mode can measure read, write, and copy bandwidth for both main memory and cache-sized working
sets.

Supported bandwidth targets include:

- Main memory / DRAM bandwidth
- L1/L2 cache-oriented bandwidth paths
- Custom cache-sized working sets via `--cache-size`

Bandwidth results are reported in GB/s. These results are useful for comparing throughput behavior across buffer sizes,
thread counts, access modes, and system conditions.

## Memory Latency

The tool measures latency using dependent pointer-chase chains. This approach serializes memory accesses so that each
load depends on the result of the previous load, making the result more representative of load-to-use latency than bulk
throughput.

Latency tests can target:

- Main memory
- Cache-sized working sets
- Custom pointer-chain stride values via `--latency-stride-bytes`
- Custom locality windows via `--latency-tlb-locality-kb`

Latency results are reported in nanoseconds per access.

## Access Pattern Analysis

Pattern mode measures how bandwidth changes under different memory access patterns. This helps expose how sequential,
strided, and random access interact with cache behavior, TLB pressure, and hardware prefetching.

The pattern suite includes:

- `sequential_forward`
- `sequential_reverse`
- `strided_64`
- `strided_4096`
- `strided_16384`
- `strided_2mb`
- `random`

These patterns are useful for comparing favorable access streams against page-sized, superpage-sized, and unpredictable
access behavior.

## TLB Behavior

The standalone TLB analysis mode, `--analyze-tlb`, estimates translation-related behavior by sweeping locality windows and
measuring pointer-chase latency changes. Locality points are measured once per round in a reproducible seeded cyclic
Latin order, so each locality rotates through different elapsed-time positions instead of being measured as one
contiguous block.

It can report:

- Likely L1/L2 TLB boundary candidates
- Locality-window sensitivity
- Large-locality latency delta, explicitly not treated as an isolated page-table-walk cost
- Ambiguity caused by private-cache effects
- Boundary confidence and inferred entry ranges
- Seed, round/order metadata, and derived chain seed for each raw measurement

These values should be interpreted as practical microarchitectural estimates, not as guaranteed architectural TLB sizes.
Apple Silicon does not expose every internal translation structure directly to user-space code, so the analysis is based
on observable latency behavior.

## Hardware Prefetch Investigation

The tool does not directly control or measure the hardware prefetcher. However, it can be used to investigate prefetch
effects indirectly.

Sequential, reverse, strided, and random access patterns can be compared to observe where prefetching appears effective
and where it no longer helps. Custom stride and locality controls can also be used to deliberately stress specific access
patterns and "tickle" the prefetcher:

- Small strides can favor cache-line reuse and regular streams.
- Page-sized strides can increase TLB and page-walk pressure.
- Large strides can reduce spatial locality.
- Global random chains make hardware prefetching much less useful.

This makes the tool useful for exploring the interaction between hardware prefetching, cache hierarchy, TLB behavior, and
main-memory latency.

## Custom Stride and Locality Experiments

Advanced latency experiments can be configured with:

- `--latency-stride-bytes <bytes>`: controls the distance between pointer-chain nodes.
- `--latency-tlb-locality-kb <KB>`: controls the locality window used when building pointer chains.
- `--latency-chain-mode <mode>`: controls pointer-chain ordering policy.

These options allow manual exploration of boundary cases where cache locality, TLB locality, prefetch behavior, and DRAM
access begin to dominate results differently.

## Built-in Parameter Sweeps

Sweep mode runs repeated measurements across parameter lists and stores every run in one combined JSON file. This makes
buffer-size, cache-size, thread-scaling, stride, locality, chain-mode, TLB-density, and core-to-core sample-depth
experiments reproducible without external shell orchestration.

Supported sweep targets include:

- Main buffer size via `--sweep buffer-size=...`
- Custom cache target via `--sweep cache-size=...`
- Bandwidth thread count via `--sweep threads=...`
- Latency locality windows via `--sweep latency-tlb-locality-kb=...`
- Pointer-chain stride via `--sweep latency-stride-bytes=...`
- Pointer-chain construction mode via `--sweep latency-chain-mode=...`
- TLB analysis stride, chain mode, and density via `--analyze-tlb --sweep latency-stride-bytes=...`,
  `--analyze-tlb --sweep latency-chain-mode=...`, and `--analyze-tlb --sweep tlb-density=...`
- Core-to-core loop count via `--sweep count=...`
- Core-to-core sample depth via `--sweep latency-samples=...`

Multiple `--sweep` options are combined as a Cartesian product. `--sweep-max-runs` caps the generated run count, and
`--output` is required for the combined JSON result.

## Core-to-Core Cache-Line Handoff

The standalone core-to-core mode, `--analyze-core2core`, measures two-thread cache-line handoff behavior using a ping-pong
style benchmark.

It reports:

- Round-trip latency
- One-way latency estimate
- Percentile statistics
- Scheduler-affinity hint scenarios

On macOS, user-space programs cannot guarantee exact physical core pinning. For that reason, core-to-core results should
be interpreted as scheduler-influenced cache-line handoff measurements rather than strict physical-core topology probes.

## Interpretation Notes

The measurements are sensitive to system state, thermal conditions, power management, background load, and buffer sizes.
For long or comparative runs, use repeated samples and prevent sleep, for example:

```bash
caffeinate -i -d memory_benchmark --benchmark --count 10 --buffer-size 1024
```

For DRAM-focused results, use sufficiently large buffers so that the working set is not dominated by cache residency.
For latency and TLB experiments, compare multiple stride and locality settings rather than relying on a single run.

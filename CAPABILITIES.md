# Measurement Capabilities

`memory_benchmark` characterizes memory-system behavior on Apple Silicon Macs using native ARM64 CPU paths and a standalone Metal compute path. It is designed for controlled comparisons and microarchitectural investigation, not for producing one synthetic performance score.

Bandwidth is reported as **effective workload payload divided by measured time**. It is not a hardware-counter measurement of physical DRAM or cache-bus traffic. A large working set can make a run more useful for main-memory-focused analysis, but buffer size alone does not prove where every byte was served.

| Capability | What it measures | What it does not establish |
|---|---|---|
| CPU bandwidth (`--benchmark`) | Effective read, write, and copy payload rate for cache-sized or main-memory-sized working sets | Physical DRAM traffic, cache residency, or memory-controller utilization |
| GPU bandwidth (`--gpu-bandwidth`) | Effective Metal compute-kernel payload rate using GPU timestamps | Verified DRAM residency, separate VRAM performance, or CPU-to-GPU transfer rate |
| Latency (`--benchmark`) | Dependent pointer-chase load-to-use latency | Bulk bandwidth or an isolated TLB/page-walk cost |
| Access patterns (`--patterns`) | Payload-rate sensitivity to access order, regularity, and virtual stride | Which single cache, prefetch, translation, or scheduling mechanism caused a difference |
| TLB analysis (`--analyze-tlb`) | Paired spread/packed latency deltas and empirical boundary estimates | Guaranteed architectural TLB sizes or direct DRAM latency |
| Core-to-core (`--analyze-core2core`) | Effective round-trip time of a repeated two-thread acquire/release token exchange under scheduler hints | Isolated physical cache-line migration or coherence-path latency, exact physical-core placement, or a definitive topology map |
| Sweeps and JSON (`--sweep`, `--output`) | Repeated configurations and auditable measurement evidence | Comparability when commands, software, hardware, or run conditions differ |

## CPU Memory and Cache Bandwidth

Standard `--benchmark` mode measures ARM64 read, write, and copy throughput. Targets include a large-buffer main-memory working set, automatically detected L1- and L2-sized working sets, and a custom cache-sized target selected with `--cache-size`. `--only-bandwidth` omits latency work.

Buffer size, worker count, loop count, and work can be controlled. When `--iterations` is omitted, excluded calibration work selects the measured pass count; an explicit value fixes the work. Repeated loops reuse the resolved plan and rotate enabled phases and operation order to distribute position effects.

Large-buffer bandwidth defaults to all detected CPU cores. Cache bandwidth defaults to one worker unless `--threads` is specified. Worker QoS is a best-effort macOS scheduler hint, not physical-core pinning.

These results are useful for comparing thread scaling, operation types, working-set sizes, machines, and software revisions under matched conditions. Read, write, and copy values are effective payload rates. In particular, copy payload accounting describes logical workload bytes rather than observed traffic on a physical bus.

## Metal GPU Memory Bandwidth

Standalone `--gpu-bandwidth` mode measures effective read, write, and copy payload rates for versioned Metal compute kernels. GPU command-buffer timestamps define the primary timing interval. Read and write count one full-buffer payload per pass; copy counts both its read and write sides. Copy operates between Metal buffers and is not a CPU-to-GPU transfer test.

The buffers use private Metal storage on Apple Silicon unified memory. “Private” means GPU-only resource access through the API; it neither means separate VRAM nor proves physical DRAM residency. The mode requires unified memory and `MTLGPUFamilyApple7` capability or a compatible later family. Capability support is not a performance guarantee.

Automatic mode calibrates work per operation, while explicit iterations provide fixed work. Measurements include warmup, preconditioning, GPU timing, and validation, so they describe steady-state warm-memory kernel execution rather than cold-cache behavior. GPU caches, dispatch overhead, other GPU activity, thermals, power state, the driver, and runtime compilation can all affect results.

CPU and GPU GB/s should not be compared as if they were the same workload: their kernels, parallelism, resource models, and clocks differ. See the [GPU Bandwidth Whitepaper](GPU_BANDWIDTH_WHITEPAPER.md) for the full timing, resource, validation, and provenance contracts.

## Memory and Cache Latency

Standard latency tests use dependent pointer-chase chains: each load determines the address of the next. This serial dependency suppresses memory-level parallelism and measures load-to-use latency rather than throughput.

Latency can target large-memory, detected cache-sized, or custom cache-sized working sets. Custom stride and locality experiments are part of this capability:

- `--latency-stride-bytes` controls the virtual distance between chain nodes.
- `--latency-tlb-locality-kb` controls the locality window used to build the chain.
- `--latency-chain-mode` selects the chain-ordering policy.

Each per-loop headline value comes from one retained continuous pointer-chase timing pass. A separate sample pass runs by default with 1,000 windows; `--latency-samples` controls that positive window count. Sample windows continue from the preceding window's terminal pointer and form a distribution that neither defines nor weights the headline.

When locality is not explicitly selected, the standard benchmark also performs a paired 16 KiB-locality versus global-random comparison. Its delta combines cache, address-locality, and translation effects; it is not an isolated page-table-walk cost or a TLB-capacity result. A seed reproduces chain and schedule identity, not identical timings. See the [Latency Whitepaper](LATENCY_WHITEPAPER.md) for construction and timing details.

## Access-Pattern Analysis

Standalone `--patterns` mode compares effective read, write, and copy bandwidth for sequential forward, sequential reverse, 64 B, 4096 B, 16384 B, and 2 MiB virtual strides, and random access.

The suite exposes sensitivity to access order, spatial locality, regularity, stride, and worker count. Differences can motivate hypotheses about cache reuse, hardware prefetching, translation, scheduling, or memory-controller behavior, but the tool does not directly control or measure the prefetcher and cannot identify one mechanism as the cause.

Pattern work uses warmup and, in automatic mode, excluded same-shape calibration. Sparse-stride workloads may use fewer effective workers than requested so active workers still perform meaningful work. Exact completed payload is the authoritative bandwidth numerator.

`strided_2mb` means a 2 MiB virtual-address interval; it does not assert physical superpage backing. Use matched buffer, thread, seed, and system settings for comparisons, and use `--analyze-tlb` for controlled translation-related questions.

## TLB Analysis

Standalone `--analyze-tlb` mode estimates translation-related boundaries by sweeping exact active-page counts. At each point it compares a one-node-per-page spread chain with a packed control matched for node count and active cache-line footprint. The primary signal is their same-round latency delta.

Pairs are measured in a reproducible balanced order across adaptive rounds. Confidence intervals, persistence checks, and an independent validation pass gate accepted boundary candidates. Density profiles trade runtime for measurement depth and refinement.

Results can include likely L1/L2 boundary candidates and inferred entry ranges, spread/packed controls, translation deltas, locality sensitivity, confidence, ambiguity, and rejected-candidate evidence. These are empirical estimates, not guaranteed architectural capacities. Cache and locality effects can remain because user space cannot observe every translation structure directly. The large-locality comparison is neither direct DRAM latency nor an isolated page-table-walk measurement.

See the [TLB Analysis Whitepaper](TLB_ANALYSIS_WHITEPAPER.md) for the methodology and validation contract.

## Core-to-Core Cache-Line Handoff

Standalone `--analyze-core2core` mode measures the effective elapsed time of a repeated two-thread acquire/release token-exchange protocol. The result includes the protocol instructions, coherence behavior, and scheduler effects; it is not a direct observation of physical cache-line migration or an isolated coherence-fabric latency. The mode reports a continuous round-trip headline, an estimated one-way value, repeatability across loops, and a separate distribution of sample-window means.

Scheduler-hint scenarios receive their own calibrated work plans, and their order rotates across loops to reduce systematic position bias. macOS user space cannot guarantee exact physical-core pinning, so results are scheduler-influenced handoff measurements rather than definitive topology probes. For affinity-policy comparisons, require complete measurement evidence and successful affinity API returns in every measured affinity record; inspect QoS outcomes separately because the affinity-interpretability field does not include them.

See the [Core-to-Core Whitepaper](CORE_TO_CORE_WHITEPAPER.md) for the assembly protocol and result contract.

## Parameter Sweeps and JSON Evidence

Built-in sweeps execute supported parameter lists without shell orchestration. Standard and pattern modes can sweep buffer size and thread count. Standard latency also supports cache size, stride, locality, and chain-mode sweeps; TLB mode supports its stride, chain-mode, and density controls; core-to-core mode supports loop count and sample depth. Multiple sweep options form a Cartesian product, and combined output requires `--output`. GPU schema 1 does not support sweeps.

JSON is designed as auditable evidence, not merely a list of numbers. It preserves the resolved configuration, work and seed identity, measurements, statistics, status, and enough completion information to distinguish complete, partial, interrupted, and unavailable results. Missing measurements are nullable rather than represented by numeric zero.

Sweep output retains completed evidence even when a later run stops. Consumers should check the mode-specific status and completeness indicators before using aggregate conclusions. Exact schemas, checkpoint behavior, and inspection examples are in the [User Manual](MANUAL.md#json-output-format) and [Technical Specification](TECHNICAL_SPECIFICATION.md#18-json-output-contract).

## Interpretation and Further Documentation

Results depend on scheduling, background load, thermals, power management, working-set size, and software state. For comparisons:

1. Keep commands, seeds, versions, relevant hardware settings, and run conditions matched.
2. Prevent sleep and minimize unrelated activity during long runs.
3. Prefer repeated loops; examine variability and tail percentiles alongside medians.
4. Use sufficiently large buffers for main-memory-focused work while retaining the effective-payload interpretation boundary.
5. Compare multiple strides and locality settings before attributing a latency change to translation or prefetch behavior.
6. Treat `--non-cacheable` as a best-effort cache-discouraging hint, not true uncached memory.
7. Reject incomplete runs when valid conclusions require complete evidence.

For example:

```bash
caffeinate -i -d memory_benchmark --benchmark --count 10 --buffer-size 1024 --output baseline.json
```

See the [User Manual](MANUAL.md) for complete option semantics and workflows, the [Technical Specification](TECHNICAL_SPECIFICATION.md) for implementation contracts, and the mode-specific whitepapers for measurement methodology.

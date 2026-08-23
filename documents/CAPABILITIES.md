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
| LLM memory profile (`--llm-memory`) | Effective logical model-payload rate and synthetic work-unit latency for CPU decode/prefill with contiguous or paged KV and an experimental Metal decode preview with either layout | Transformer computation, inference tokens/s, TTFT, physical DRAM traffic, runtime page allocation, ANE execution, or framework performance |
| JSON output (`--output`) and sweeps (`--sweep`) | Auditable measurement evidence through recoverable files or one final stdout document for every result-producing direct mode and supported CPU sweep | Comparability when commands, software, hardware, or run conditions differ |

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

## Synthetic LLM Memory Profile

Standalone `--llm-memory` uses generic backend/phase/layout/work-unit vocabulary. Active profiles are CPU/decode with
contiguous or paged KV, CPU/prefill with contiguous or paged KV, and Metal/decode with either layout. Exact
methodologies are
`llm-memory-v1-cpu-decode-contiguous`, `llm-memory-v1-cpu-decode-paged`,
`llm-memory-v1-cpu-prefill-contiguous`, `llm-memory-v1-cpu-prefill-paged`,
`llm-memory-v1-metal-decode-contiguous`, and `llm-memory-v1-metal-decode-paged`. Metal is explicitly selected with
`--llm-memory-backend metal`, rejects prefill and `--threads`, and never receives an implicit CPU fallback.

Metal decode with contiguous or paged KV is an experimental preview.

Each active phase executes three scenarios:

- `weights_only` reads the active-weight mapping once per work unit;
- `kv_only` performs the phase-specific K/V write and reads;
- `mixed` performs both traffic components in one timed scenario. CPU uses worker-local layer order inside one
  synchronized worker interval; Metal uses its scenario-specialized grid-stride kernel inside one GPU timestamp interval.

Metal decode specializes one pipeline for each scenario and uses one workload dispatch per task, with every requested
work unit looped inside that kernel. `GPUStartTime`/`GPUEndTime` define authoritative elapsed time. Reset, expected
dual-mod32 checksum construction, and append validation are excluded. Private/tracked W/K/V resources use exact-tail
segments no larger than 256 MiB and a Tier 2 argument buffer. Paged decode additionally uses block-aligned K/V
segments, segmented private table storage, named-lane volatile table loads with threadgroup publication/barriers, and
excluded padding-canary validation. Runtime admission requires unified memory, Apple7-or-later
capability, Tier 2 argument buffers, `maxBufferLength >= 256 MiB`, and MSL 2.3 runtime compilation. Unsupported
capability does not authorize CPU fallback. Runtime compiler, pipeline, resource, or task failures remain terminal
failed/invalid evidence rather than changing backend. Metal results remain synthetic effective payload rates; GPU
cache/SLC/DRAM residency is not measured.

The current source identity is `llm-metal-decode-contiguous-paged-msl23-v2`; the result records its exact runtime
SHA-256 and scenario-specific `membenchmark.llm-metal.pipeline.decode-contiguous.*` or
`membenchmark.llm-metal.pipeline.decode-paged.*` pipeline labels and limits.

With active-weight bytes `W`, layer count `L`, KV heads `h_kv`, head dimension `d_h`, KV element bytes `s_kv`, batch
`B`, and visible context `A`, define `K = L * 2 * h_kv * d_h * s_kv`. Per-work-unit effective model payload is `W` for
weights-only, `B*A*K + B*K` for KV-only, and `W + B*A*K + B*K` for mixed. The versioned crossover/classification
compares `W` with KV-read payload `B*A*K`; exact equality alone is `near_crossover`, and no class identifies a measured
hardware bottleneck.

For prefill, `--prompt-tokens P` and `--attention-query-tile-tokens Q` are required with `P >= 1` and `1 <= Q <= P`;
`--context-tokens` is rejected. One `prefill_operation` rewrites all `P` K/V records for every batch sequence, then
reads each causal prefix at tile ends `e_j = min((j+1)*Q, P)`. With `C = ceil(P/Q)` and
`S(P,Q) = sum(e_j)`, weight read is `W`, KV write is `B*P*K`, and KV read is `B*S(P,Q)*K`. Thus prefill payload is
`W`, `B*(P+S(P,Q))*K`, or `W+B*(P+S(P,Q))*K` for weights-only, KV-only, or mixed. `Q=P` reads one full prefix and
`Q=1` yields `S=triangular(P)`. Causal-pair and logical-FMA counts are audit metadata only; no FMA is executed.
CPU prefill also reports scenario-specific cost-balanced partition identities and exact per-worker minimum, maximum,
and imbalance evidence with `worker-cost` units.

CPU execution allocates active weights and K/V resources at their full derived sizes in ordinary cacheable anonymous
memory; Metal uses exact-tail private/tracked W/K/V segments on unified memory. Initialization and pre-touch occur before
measurement. Contiguous layout is layer, batch sequence, token, head, then head dimension. `--kv-layout` defaults to
contiguous, which rejects `--kv-block-tokens`. Paged layout uses complete
`G`-token physical blocks plus one uint32 block table and requires exactly one explicit positive power-of-two
`--kv-block-tokens G` no greater than `UINT32_MAX`; `G` may exceed the active phase length. For phase length `A`
(decode context or prefill prompt), `R = h_kv*d_h*s_kv`, and `N = ceil(A/G)`, K physical bytes are `L*B*N*G*R`, K
logical bytes remain `L*B*A*R`, and their difference is reported padding; V is identical. The table has `B*N` entries
and is a seeded, versioned bijection over the same physical-ID domain. MHA, GQA, and MQA are represented by
query-head/KV-head geometry, but physical KV payload is determined by the KV-head count.

Paged KV-bearing scenarios perform `L*B*(2*N+1)` explicit timed table loads per decode step: one paired append lookup,
`N` K-block lookups, and `N` V-block lookups per layer/batch pair. The four bytes per lookup are separately reported
layout metadata and count toward task admission, but never enter effective-model-payload GB/s. `weights_only` does not
touch the table. Terminal suffix padding is initialized and canary-validated but is neither timed payload nor accessed
by the kernel.

For paged prefill, let `m_j = ceil(e_j/G)` and `M = sum(m_j)`. A KV-bearing operation performs
`L*B*(N+2*M)` timed loads: `N` paired-write lookups plus separate `M` K-prefix and `M` V-prefix lookups per layer/batch
pair. Partial prefixes stop at the exact tile end instead of widening to a full terminal block, and the lookup total is
independent of worker count.

The frozen SplitMix64/Fisher-Yates permutation is validated and hashed before execution. Paged data patterns depend on
pool, physical ID, and physical offset, while the timed checksum binds logical table index, loaded physical ID,
semantic visit kind, and work-unit ordinal. Decode current-token writes or prefill final-ordinal prompt samples, plus
padding canaries, are validated after each task.
Permutation generation, initialization, expected-checksum construction, and post-validation remain outside the
synchronized CPU timing interval.

The reported decimal GB/s is exact logical effective model payload divided by backend-authoritative elapsed time:
synchronized worker time for CPU or `GPUStartTime`/`GPUEndTime` for Metal. A synthetic work unit is one `decode_step`
or one full-prompt `prefill_operation`, not an inference token. The mode does not run GEMM/GEMV, dequantization, RoPE,
attention math, softmax, layer normalization, framework dispatch, model loading, ANE work, or GPU execution outside the
defined Metal decode kernels. Prefill results do not predict TTFT. The mode also does not model growing
context, runtime page allocation,
sliding-window KV, prefix sharing,
speculative decoding, or compute-memory overlap. The active paged layout measures frozen table indirection and physical
scatter; it is not a runtime allocator and does not provide prefix sharing, eviction, copy-on-write, or sliding-window
behavior. Full-size
resources reduce the risk of accidentally benchmarking a recycled proxy buffer, but they do not prove physical DRAM
service.

The three scenarios are independently calibrated when `--iterations` is omitted and then frozen before loop zero.
Their order rotates across count loops; the default count of three gives each scenario one first, middle, and last
position. Only measured, validly timed, checksum-accepted records enter aggregates. A file target receives an atomic
checkpoint after each terminal scenario measurement and at command terminal; exact `--output -` performs the same
logical state transitions but emits only one final schema 1 document. See the
[LLM Memory Profile Whitepaper](LLM_MEMORY_PROFILE_WHITEPAPER.md) for formulas, timing, validation, and interpretation.

Comparisons require matching backend/phase/layout, schema/methodology and component identities, model/phase geometry,
fixed or automatic work policy, frozen work-plan identity, software, hardware, applicable worker counts, and
sufficiently similar thermal/power/load conditions. A comparative consumer must also require complete,
position-balanced schema state, every planned measurement to be measured, and the selected metric to be non-null;
process success alone is insufficient. Paged cohorts additionally require matching block size, physical/padding/table
geometry, and permutation identity/hash; contiguous and paged samples are not interchangeable.

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

Built-in sweeps execute supported parameter lists without shell orchestration. Standard and pattern modes can sweep
buffer size and thread count. Standard latency also supports cache size, stride, locality, and chain-mode sweeps; TLB
mode supports its stride, chain-mode, and density controls; core-to-core mode supports loop count and sample depth.
Multiple sweep options form a Cartesian product, and combined output requires `--output`. GPU schema 1 and LLM schema 1
do not support sweeps.

JSON is designed as auditable evidence, not merely a list of numbers. It preserves the resolved configuration, work and seed identity, measurements, statistics, status, and enough completion information to distinguish complete, partial, interrupted, and unavailable results. Missing measurements are nullable rather than represented by numeric zero.

Every result-producing direct mode and the CPU modes' supported sweeps reserve the exact target `--output -` for
machine-readable output: stdout contains one final JSON document and the post-parse human transcript is routed to
stderr. An empty output value disables JSON for a direct command but is missing/invalid for a sweep. Every other
non-empty value is a file target, including `./-` and flag-shaped names such as `-G`. File output is atomic; standard
commands, sweeps, GPU, and LLM retain their mode-specific checkpoints, while stdout remains final-only. GPU and LLM
stdout execute their logical checkpoint transitions and stop observations without intermediate serialization. The
current support matrix and process acceptance procedure are in the [Machine-Readable CLI API](API.md).

Current standard results use schema 3, which requires `configuration.mode: "benchmark"`, a string
`configuration.output_file`, plus boolean `results_complete` and `conclusions_valid`. Bundled standard-memory examples
track the current producer, perform local sanity checks, and read its current schema-3 metric paths directly. They do
not support released standard schema 2, unversioned historical standard JSON layouts, or any other explicit standard
version.

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

For a direct machine-consumer run:

```bash
memory_benchmark --patterns --buffer-size 512 --count 5 --seed 42 --output - \
  >patterns.json 2>patterns.log
```

See the [Machine-Readable CLI API](API.md) for process integration, the [User Manual](MANUAL.md) for complete option
semantics and workflows, the [Technical Specification](TECHNICAL_SPECIFICATION.md) for implementation contracts, and
the mode-specific whitepapers for measurement methodology.

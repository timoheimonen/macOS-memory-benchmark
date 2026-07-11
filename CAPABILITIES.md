# Measurement Capabilities

macOS-memory-benchmark / `memory_benchmark` is a low-level benchmark and analysis tool for characterizing memory-system behavior on macOS
running on Apple Silicon.

The tool is intended for practical microarchitectural investigation rather than abstract synthetic scoring. It measures
bandwidth, latency, access-pattern sensitivity, TLB behavior, and core-to-core cache-line handoff characteristics using
native ARM64 code paths.

Long-running benchmark entry points block SIGINT/SIGTERM while worker threads exist, poll for interruption between safe
phases, and restore the caller's exact prior signal mask at scope exit. Live progress is emitted only to an interactive
`stderr` terminal; redirected output remains free of spinner carriage-return sequences.

## Main Memory and Cache Bandwidth

The standard benchmark mode can measure read, write, and copy bandwidth for both main memory and cache-sized working
sets.

Supported bandwidth targets include:

- Main memory / DRAM bandwidth
- L1/L2 cache-oriented bandwidth paths
- Custom cache-sized working sets via `--cache-size`

Bandwidth results are reported in GB/s. These results are useful for comparing throughput behavior across buffer sizes,
thread counts, access modes, and system conditions.

When `--iterations` is omitted, each target/operation uses an excluded same-shape pilot and bounded correction to select
a measured pass count near 150 ms. The exact pass count, finalized worker boundaries, and payload accounting are reused
across `--count` loops. Explicit `--iterations` remains an exact override. Read/write/copy order and enabled phase order
rotate across repeated loops. Worker QoS is a best-effort scheduler hint; it is not core pinning.

Main-memory bandwidth defaults to all detected CPU cores. Cache bandwidth defaults to one worker when `--threads` is
omitted; an explicit thread count applies to both standard bandwidth targets. Pattern mode also defaults to all detected
cores, although sparse strided work may reduce its effective worker count.

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

Every standard main/L1/L2/custom headline comes from one continuous pointer chase calibrated near 250 ms, evaluated
against a 100–300 ms window, and rounded to complete chain cycles. Optional sample windows run separately and continue from one window's terminal pointer to the
next. A resolved `--seed` reproduces chain and schedule metadata across loops, not performance values. When locality is
not explicit, the standard benchmark reports a paired 16 KiB-locality/global-random comparison and median same-round
delta. That result combines cache, locality, and translation effects; only `--analyze-tlb` supports controlled
translation-boundary conclusions.

The effective standard sample count is capped to the measurement access count. Stride is positive and pointer-aligned,
and each enabled main/cache target and configured locality window must retain at least two stride-spaced nodes. The
automatic fixed 16 KiB comparison additionally requires stride at most 8192 bytes; if that setup cannot form two nodes,
the comparison is unavailable without redefining the validated target measurements. The system-page-size maximum
applies only to standalone TLB analysis.

Standard schema-v2 JSON records completion, nullable measurement state, exact work, calibration, seed, schedule, and
requested/effective worker metadata. Only measured values enter median/CV/MAD summaries. Output is atomically
checkpointed after completed loops, and `results_complete` lets consumers reject partial runs.

## Access Pattern Analysis

Pattern mode compares effective bandwidth under different access orders, regularities, and virtual strides. It does not
isolate cache, translation, prefetch, or DRAM effects, so a result difference must not be attributed to one mechanism
without a controlled follow-up experiment.

The pattern suite includes:

- `sequential_forward`
- `sequential_reverse`
- `strided_64`
- `strided_4096`
- `strided_16384`
- `strided_2mb`
- `random`

These patterns are useful for comparing regular streams, different virtual-address strides, and unpredictable access
orders. In particular, `strided_2mb` means a 2 MiB virtual-address interval; it does not assert 2 MiB physical-page
backing. Use `--analyze-tlb` for controlled translation-related conclusions.

Pattern schema-3 JSON records measurement-level evidence plus top-level status/reason, exact planned/completed loop and
measurement counters, and `results_complete`. Each loop plans 21 operations. Numeric measured values and intentional
skips count as completed; invalid or failed execution does not. Interrupted, partial, and failed runs preserve completed
evidence, but only Complete loops feed aggregate values and headlines. Consumers that require completeness must require
both `status: "complete"` and `results_complete: true`.

## TLB Behavior

The standalone TLB analysis mode, `--analyze-tlb`, estimates translation-related behavior by sweeping exact active-page
counts. Every scheduled point measures a verified one-node-per-page spread chain and a packed control with the same node
and unique-cache-line counts. Locality pairs are measured once per round in a reproducible seeded cyclic Latin order, so
each locality rotates through different elapsed-time positions instead of being measured as one contiguous block. A pilot
calibrates accesses toward the selected quick/standard/exhaustive target duration, and complete rounds stop when every
point meets the profile CI-width target or the maximum round count is reached. Buffer choice is gated by a predicted
buffer-plus-scratch peak and a conservative available-memory budget.

It can report:

- Likely L1/L2 TLB boundary candidates
- Locality-window sensitivity
- Large-locality latency delta, explicitly not treated as an isolated page-table-walk cost
- Ambiguity caused by private-cache effects
- Boundary confidence and inferred entry ranges
- Seed, round/order metadata, and derived chain seed for each raw measurement
- Same-round spread/packed raw latencies, physical-chain diagnostics, and paired translation delta
- Runtime profile, calibrated access counts, memory/work estimate, and realized adaptive-round completion metadata

These values should be interpreted as practical microarchitectural estimates, not as guaranteed architectural TLB sizes.
Apple Silicon does not expose every internal translation structure directly to user-space code, so the analysis is based
on observable latency behavior.

## Access-Regularity and Prefetch Hypotheses

The tool does not directly control or measure the hardware prefetcher. Sequential, reverse, strided, and random workloads
can establish effective-bandwidth differences that motivate a prefetch hypothesis, but the same difference can also
contain cache, translation, memory-controller, or scheduling effects.

- Small virtual strides increase address regularity and potential cache-line reuse.
- Page-sized and larger virtual strides reduce spatial locality, but do not by themselves identify a TLB boundary.
- Global-random pointer chains reduce address regularity, but their latency is not an isolated prefetch measurement.

Treat pattern results as access-regularity comparisons. Test prefetch hypotheses with matched buffer, stride, thread, and
system conditions, and use `--analyze-tlb` when the question specifically concerns translation behavior.

## Custom Stride and Locality Experiments

Advanced latency experiments can be configured with:

- `--latency-stride-bytes <bytes>`: controls the distance between pointer-chain nodes.
- `--latency-tlb-locality-kb <KB>`: controls the locality window used when building pointer chains.
- `--latency-chain-mode <mode>`: controls pointer-chain ordering policy.

These options allow manual exploration of boundary cases where cache locality, TLB locality, prefetch behavior, and DRAM
access begin to dominate results differently.

`auto` resolves to `global-random` when locality is zero and to `random-box` otherwise. The locality-using box modes
require a non-zero locality window; standalone TLB analysis rejects explicit `global-random`.

## Built-in Parameter Sweeps

Sweep mode runs repeated measurements across parameter lists and stores every attempted run in one combined JSON file.
This makes buffer-size, cache-size, thread-scaling, stride, locality, chain-mode, TLB-density, and core-to-core
sample-depth experiments reproducible without external shell orchestration.

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

Multiple `--sweep` options are combined as a Cartesian product. `--sweep-max-runs` caps the generated run count (default
`16` for `--analyze-tlb`, `256` otherwise), and
`--output` is required for the combined JSON result. A sweep parameter key may appear only once. Combined sweep JSON is
atomically checkpointed and exposes status/reason, planned/attempted/completed run counts, and conclusion validity.
`attempted_runs` always equals the number of stored `runs` entries. An entry increments `completed_runs` only when its
nested command is genuinely complete: standard and pattern require `status: "complete"` plus `results_complete: true`,
TLB requires `tlb_analysis.status: "complete"` plus `tlb_analysis.conclusions_valid: true`, and core-to-core requires
`core_to_core_latency.status: "complete"` plus `measurements_complete: true`. Partial, interrupted, and failed attempts
remain auditable, stop further attempts, and never make the sweep conclusions valid.

## Core-to-Core Cache-Line Handoff

The standalone core-to-core mode, `--analyze-core2core`, measures two-thread cache-line handoff behavior using a ping-pong
style benchmark. Each scheduler-hint scenario uses an excluded pilot after a 1,000,000-round-trip calibration warmup to
resolve a 25 ms final warmup, a 250 ms continuous headline, and 1 ms sample windows while retaining minimum work. The
scenario order rotates across repeated loops to balance position effects. Core-to-core mode defaults to three loops, so
the bare command produces a median plus CV/MAD; the 1,000 approximately 1 ms sample windows per scenario/loop make this
default intentionally longer than the former single-loop run.

It reports:

- Median P50 round-trip latency across completed continuous loop windows
- One-way latency estimate
- Headline repeatability statistics including CV/MAD and a 7.5% CV warning
- A separate pooled sample-window percentile distribution
- Scheduler-affinity hint scenarios
- Schema-2 work plans, per-loop audit records, completion metadata, and affinity-comparison interpretability

On macOS, user-space programs cannot guarantee exact physical core pinning. For that reason, core-to-core results should
be interpreted as scheduler-influenced cache-line handoff measurements rather than strict physical-core topology probes.
Unavailable measurements carry status/reason and nullable values rather than numeric zeroes.

## Interpretation Notes

The measurements are sensitive to system state, thermal conditions, power management, background load, and buffer sizes.
For long or comparative runs, use repeated samples and prevent sleep, for example:

```bash
caffeinate -i -d memory_benchmark --benchmark --count 10 --buffer-size 1024
```

For DRAM-focused results, use sufficiently large buffers so that the working set is not dominated by cache residency.
For latency and TLB experiments, compare multiple stride and locality settings rather than relying on a single run.

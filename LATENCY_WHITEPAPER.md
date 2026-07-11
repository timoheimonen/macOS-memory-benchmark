# Latency Whitepaper: Dependent Pointer-Chase Methodology

## 1. Purpose

This document explains how `macOS-memory-benchmark` version 0.61.0 measures memory latency on Apple Silicon.

The latency path is designed to measure **load-to-use delay** (pointer chasing), not bulk throughput.
It combines:

- A dependency-serialized ARM64 assembly chase kernel (`src/asm/memory_latency.s`)
- A configurable randomized pointer-chain builder (`src/core/memory/memory_utils.cpp`)
- A timing/sampling wrapper that preserves pointer continuity (`src/benchmark/latency_tests.cpp`)
- Unit tests validating chain setup boundary conditions (`tests/test_memory_utils.cpp`)

## 2. Why Pointer Chasing

A conventional loop over contiguous memory can be heavily assisted by OoO execution and prefetchers,
which tends to reflect throughput, not true per-access latency.

This project uses a circular linked list in memory, then repeatedly dereferences the next address:

```text
p = *p
```

Because each load address depends on the previous load result, the out-of-order engine cannot issue
load `n+1` until the result of load `n` is committed to the register file — a true RAW (read-after-write)
register dependency. The load pipeline is therefore serialized on the critical-path latency of each
individual cache or DRAM access, making the measurement suitable for latency characterization.

## 3. Assembly Kernel Design (`src/asm/memory_latency.s`)

### 3.1 Contract

- Prototype: `uintptr_t* memory_latency_chase_asm(uintptr_t* start_pointer, size_t count)`
- Inputs:
  - `x0` = start pointer
  - `x1` = dereference count
- Output:
  - `x0` = final pointer value after `count` dependent dereferences (returned to prevent dead-code
    elimination of the chase loop by the compiler or linker)

### 3.2 Core Behavior

The kernel uses three key safeguards:

1. **Zero-access fast return**
   - If `count == 0`, it returns immediately.
2. **Exact--count execution with 8x unroll**
   - `count / 8` iterations of 8 dependent loads, then `count % 8` remainder loads.
3. **Correct unsigned loop termination**
   - Remainder loop uses `subs`/`b.ne` rather than compare-and-branch to correctly handle the full
     unsigned 64-bit `count` range without introducing a signed-comparison hazard.

The unrolled body remains dependency-serialized because each `ldr x0, [x0]` consumes the prior result.
Unrolling primarily reduces branch overhead per dereference; it does not remove the data dependency chain.

### 3.3 Measurement Purity Notes

The chase kernel intentionally avoids extra ordering/fence overhead in the hot path.
The kernel does not prefetch, touch pages in advance, or insert memory barrier instructions (`DMB`/`DSB`)
in the hot path, as none of these are required for functional correctness of the chase loop.

## 4. Pointer-Chain Construction (`src/core/memory/memory_utils.cpp`)

`setup_latency_chain()` prepares the memory into a circular pointer chain before timing begins.

### 4.1 Input Validation

The function returns `EXIT_FAILURE` for invalid setup cases, including:

- `buffer == nullptr`
- `stride == 0`
- fewer than 2 pointers fit in the provided buffer (`buffer_size / stride < 2`)
- a locality-using `LatencyChainMode` is selected explicitly while `tlb_locality_bytes == 0`
  (the mode requires a locality window but none is provided)
- `tlb_locality_bytes > 0` but the locality window is too small to hold 2 pointers
  (`tlb_locality_bytes / stride < 2`)

### 4.2 Chain Construction Flow

1. Build index vector `0..num_pointers-1`
2. Reorder indices according to the effective chain mode (mode-dispatched):
   - `GlobalRandom`: full-space `std::shuffle` across all indices
   - All locality modes: delegate to `reorder_indices_with_mode()`, which applies the
     appropriate within-window and window-ordering strategy (see Section 4.3)
3. Link reordered indices into a circular chain
4. Bounds-check each write target and next-pointer target

Standard `--benchmark` execution always calls the seeded overload. One command-level `--seed` is generated or parsed,
then domain-separated seeds are derived for main memory, L1, L2, custom-cache, and the per-round automatic-locality
chains. The separate sample pass continues on the already prepared target chain.
The unseeded overload retains `std::random_device` behavior for direct callers and legacy tests, but it is not the
standard benchmark workload policy.

### 4.3 Chain Construction Modes

The `--latency-chain-mode` flag selects the pointer-chain construction policy.
`--latency-tlb-locality-kb` sets the locality window size used by all non-global modes.

| Mode enum | CLI name | `--latency-tlb-locality-kb` required | Behavior |
|---|---|---|---|
| `GlobalRandom` | `global-random` | No (ignored if provided) | Single `std::shuffle` across the full index space. Maximum address randomness; highest TLB and cache pressure. |
| `RandomInBoxRandomBox` | `random-box` | Yes | Independent random permutation within each locality window; window visit order is also randomized. Default when `tlb_locality_bytes > 0` and mode is `auto`. |
| `SameRandomInBoxIncreasingBox` | `same-random-in-box` | Yes | One random permutation is generated and applied identically to every locality window; windows are visited in sequential order. Useful for isolating within-window access variance. |
| `DiffRandomInBoxIncreasingBox` | `diff-random-in-box` | Yes | Independent random permutation per window; windows are visited in sequential order. Combines per-window randomness with a predictable inter-window traversal order. |

When `--latency-chain-mode auto` (the default), the effective mode is `global-random` if
`--latency-tlb-locality-kb 0`, and `random-box` otherwise.

This is a chain-construction policy, not a hardware TLB control primitive. The locality window
influences which addresses appear in the same traversal neighborhood; it does not guarantee
those addresses reside in any particular TLB level during execution.

### 4.4 Stride Control

`--latency-stride-bytes` controls spacing between pointer-chain nodes (default `256`).

- Smaller stride (e.g., `32`) increases same-page cache-line density — more nodes per page,
  lower page turnover.
- Larger stride (e.g., `128`) increases page turnover and can amplify translation pressure.

Non-zero stride must be a multiple of the pointer size. On AArch64, pointer size is 8 bytes,
so stride must be a multiple of 8.

## 5. Test-Backed Correctness Guarantees (`tests/test_memory_utils.cpp`)

The following behaviors are explicitly covered by unit tests:

- **Null buffer rejected** (`SetupLatencyChainNullBuffer`)
- **Zero stride rejected** (`SetupLatencyChainZeroStride`)
- **Insufficient buffer capacity rejected** (`SetupLatencyChainBufferSmallerThanStride`, `SetupLatencyChainBufferEqualToStride`)
- **Minimum valid chain accepted** (`SetupLatencyChainBufferJustLargerThanStride`)
- **Small valid chains accepted** (`SetupLatencyChainThreePointers`)
- **Constructed pointers stay inside buffer bounds** (`SetupLatencyChainCreatesValidChain`)
- **Page-sized locality accepted** (`SetupLatencyChainWithTlbLocality`)
- **Too-small locality-span rejected** (`SetupLatencyChainWithTooSmallTlbLocalityFails`)
- **Diagnostics populated correctly** (`SetupLatencyChainCollectsDiagnostics`)
- **`SameRandomInBoxIncreasingBox` mode accepted** (`SetupLatencyChainWithSameRandomInBoxMode`)
- **Locality-using mode with zero locality window rejected** (`SetupLatencyChainWithBoxModeAndZeroLocalityFails`)
- **Explicit seed reconstructs the same chain** (`SetupLatencyChainExplicitSeedIsReproducible`)

These tests validate setup correctness and failure handling before timing is run.

## 6. Timing and Sampling Semantics (`src/benchmark/latency_tests.cpp`)

The low-level timing wrapper supports a continuous pass and a segmented sample-collection pass. Standard benchmark
orchestration uses both as separate populations:

1. On the first loop, an excluded continuous pilot resolves work toward a 250 ms target. The final access count is
   rounded to at least 16 complete chain cycles and evaluated against a 100-300 ms intended window.
2. Every benchmark loop runs one continuous headline chase. This pass alone produces the loop's reported latency
   headline.
3. If sampling is enabled, a second pass executes the same total access count in continuing sample windows. These
   windows populate the separate sample distribution and never replace or weight the continuous headline.

The resolved access count is reused across `--count` loops. Key sample-pass behavior:

- `effective_samples = min(requested_samples, num_accesses)`
- Access remainder is distributed over early samples
- Each sample times `accesses_this_sample` dereferences and stores per-sample latency
- The returned pointer from sample `i` becomes the starting pointer for sample `i+1`

That pointer continuity is important for measurement integrity. If each sample always restarted
from the same chain head, the chain head and its immediate successors would be warm in cache
at the start of every sample (left there by the prior sample), causing the first few accesses
of each sample to measure a hot-cache hit rather than the steady-state latency. This would bias
per-sample latencies systematically downward. By passing the final pointer of sample `i` as the
start of sample `i+1`, every sample begins from an arbitrary point in the chain, preventing
that cold-start artifact.

The sample pass executes exactly `num_accesses` dereferences in addition to the headline pass; `--latency-samples` is
therefore a distribution-depth control, not a way to divide or redefine headline work.

`run_latency_test()` is the shared entry point for main-memory, L1, L2, and custom-cache-size
latency measurements. Every case uses the same internal `run_latency_measurement()` path and
assembly kernel; the executor selects the buffer and stores the result in the appropriate field.

## 7. Experimental Protocols

### Parameter Quick Reference

| Flag | Default | Effect |
|---|---|---|
| `--latency-stride-bytes` | `256` | Byte spacing between pointer-chain nodes. Must be a multiple of 8 (pointer size on AArch64). |
| `--latency-tlb-locality-kb` | `1024` | Locality window size in KB. `0` selects global-random mode when `--latency-chain-mode auto` is used; explicit box modes require non-zero locality. |
| `--latency-samples` | `1000` | Number of windows in the separate sample pass per benchmark loop. It does not change the continuous headline definition. |
| `--latency-chain-mode` | `auto` | Chain construction policy. `auto` selects `global-random` when locality is 0, `random-box` otherwise. |
| `--seed` | generated once | Exact uint64 command seed used to derive reproducible target/layout chains. |
| `--count` | `1` | Number of measured loops. Repeated-loop headline is the median P50 of completed continuous loop headlines. |

### 7.1 Baseline Main-Memory Latency (Global Random Chain)

```bash
./memory_benchmark \
  --benchmark \
  --only-latency \
  --buffer-size 1024 \
  --count 10 \
  --latency-samples 1000 \
  --latency-tlb-locality-kb 0 \
  --output latency_global.json
```

### 7.2 Locality-Window Latency Runs

```bash
./memory_benchmark --benchmark --only-latency --buffer-size 1024 --count 10 --latency-samples 1000 --latency-tlb-locality-kb 16 --output latency_tlb16.json
./memory_benchmark --benchmark --only-latency --buffer-size 1024 --count 10 --latency-samples 1000 --latency-tlb-locality-kb 2048 --output latency_tlb2048.json
./memory_benchmark --benchmark --only-latency --buffer-size 1024 --count 10 --latency-samples 1000 --latency-tlb-locality-kb 32768 --output latency_tlb32768.json
```

### 7.3 Cache-Scale Sweep (Custom Cache Size)

Use the repository script to sweep cache sizes with multiple locality settings:

```bash
script-examples/latency_test_script.sh
```

Then visualize trends:

```bash
python3 script-examples/plot_cache_percentiles.py script-examples/final_output.txt --metric median
```

A second sweep script, `script-examples/latency_test_script_stride_tlb.sh`, explores stride and
TLB-locality combinations and produces a combined CSV-style output suitable for cross-dimensional
analysis.

## 8. Reading the Results

- Prefer **median (P50)** for the trend curve.
- Use **P95/P99** to evaluate tail sensitivity and instability.
- Treat isolated very large `max` values as outlier diagnostics, not central tendency.
- Compare locality settings at the same cache size, then compare cache-size transitions.
- Read the continuous aggregate from `main_memory.latency.headline_ns`; cache aggregates use
  `cache.custom.latency.headline_ns`, `cache.l1.latency.headline_ns`, or `cache.l2.latency.headline_ns`.
- Exact loop work and chain metadata are under `loops[].measurements.main_latency`, `l1_latency`, `l2_latency`, or
  `custom_latency`. These records include access count, chain-node count, complete cycles, seed, elapsed time,
  calibration quality, and status.
- When present, segmented windows are under each headline aggregate's `pooled_sample_distribution`, with values and
  loop-boundary metadata kept separate from continuous loop headlines.
- The current standard schema 2 does not serialize the legacy `chain_diagnostics.unique_pages_touched` blocks. Do not
  use the old `main_memory.latency.chain_diagnostics` or `cache.*.latency.chain_diagnostics` paths for version 0.61.0
  output.

When `--latency-tlb-locality-kb` is not explicitly supplied, standard main-memory latency also runs three paired rounds
of 16 KiB-locality and global-random chains. The first-measured layout alternates by round, and
`main_memory.latency.automatic_locality_comparison.locality_latency_delta_ns` is the median of same-round
`global - locality` deltas. This mixed locality/cache comparison is not an isolated page-table-walk measurement; use
`--analyze-tlb` for validated translation-boundary analysis.

### 8a. ARM64 Reference Latency Ranges

The table below provides approximate latency ranges for Apple Silicon as a sanity-check reference.
Exact values vary by M-series generation, operating frequency, and thermal state.

| Memory level | Typical cycles | Typical latency (@ ~4 GHz) |
|---|---|---|
| L1 data cache hit | ~4 cycles | ~1 ns |
| L2 cache hit | ~12 cycles | ~3 ns |
| Main memory (DRAM) | ~100–150 cycles | ~25–40 ns |

If measured latency deviates substantially from these ranges at the expected cache size (e.g.,
measured L1 latency is 20 ns), the buffer size or stride likely places accesses outside the
intended cache level, or system interference is present.

## 9. Limitations and Non-Goals

- The benchmark runs in user space and does not create truly uncached memory mappings.
- TLB-locality windows influence chain layout, but they do not force a specific hardware TLB residency outcome.
- Observed latency includes full system effects along the dependent-load path (cache hierarchy, translation effects when present, and runtime noise).
- Cross-run variance can still occur due to scheduling, thermal behavior, and background activity.

## 10. Summary

The latency path in this project is built around a strict dependent pointer chase with exact--count execution and tested chain setup boundaries.
It is suitable for comparative latency studies across cache sizes and locality-window policies when interpreted with robust statistics (median and tails) rather than single-point extremes.

## 11. Related Documents

- [TLB_ANALYSIS_WHITEPAPER.md](TLB_ANALYSIS_WHITEPAPER.md) — Standalone TLB boundary detection mode (`--analyze-tlb`): sweep methodology, boundary/guard rules, confidence model, and JSON contract.
- [CORE_TO_CORE_WHITEPAPER.md](CORE_TO_CORE_WHITEPAPER.md) — Core-to-Core Cache-Line Handoff Latency Benchmark (`--analyze-core2core`): LDAR/STLR assembly protocol, scheduler-hint scenarios, and JSON contract.
- [TECHNICAL_SPECIFICATION.md](TECHNICAL_SPECIFICATION.md) — Runtime architecture, execution flow, and output contracts for all benchmark modes.

# Latency Whitepaper: Dependent Pointer-Chase Methodology

## 1. Purpose

This document explains how `macOS-memory-benchmark` measures memory latency on Apple Silicon.

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
2. **Exact-count execution with 8x unroll**
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

### 4.3 Chain Construction Modes

The `-latency-chain-mode` flag selects the pointer-chain construction policy.
`-latency-tlb-locality-kb` sets the locality window size used by all non-global modes.

| Mode enum | CLI name | `-latency-tlb-locality-kb` required | Behavior |
|---|---|---|---|
| `GlobalRandom` | `global-random` | No (ignored if provided) | Single `std::shuffle` across the full index space. Maximum address randomness; highest TLB and cache pressure. |
| `RandomInBoxRandomBox` | `random-box` | Yes | Independent random permutation within each locality window; window visit order is also randomized. Default when `tlb_locality_bytes > 0` and mode is `auto`. |
| `SameRandomInBoxIncreasingBox` | `same-random-in-box` | Yes | One random permutation is generated and applied identically to every locality window; windows are visited in sequential order. Useful for isolating within-window access variance. |
| `DiffRandomInBoxIncreasingBox` | `diff-random-in-box` | Yes | Independent random permutation per window; windows are visited in sequential order. Combines per-window randomness with a predictable inter-window traversal order. |

When `-latency-chain-mode auto` (the default), the effective mode is `global-random` if
`-latency-tlb-locality-kb 0`, and `random-box` otherwise.

This is a chain-construction policy, not a hardware TLB control primitive. The locality window
influences which addresses appear in the same traversal neighborhood; it does not guarantee
those addresses reside in any particular TLB level during execution.

### 4.4 Stride Control

`-latency-stride-bytes` controls spacing between pointer-chain nodes (default `512`).

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

These tests validate setup correctness and failure handling before timing is run.

## 6. Timing and Sampling Semantics (`src/benchmark/latency_tests.cpp`)

The timing wrapper provides two modes:

- **Single-shot mode** (no sample vector): one continuous chase of `num_accesses`
- **Sampled mode**: split total accesses across samples while preserving exact total accesses

Key sampled-mode behavior:

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

`run_latency_test()` is the entry point for main-memory latency measurement.
`run_cache_latency_test()` is the corresponding entry point for L1, L2, and custom-cache-size
latency measurements. Both functions share the same internal `run_latency_measurement()` path
and the same assembly kernel; the distinction is only in which buffer is passed and which
result field is populated.

## 7. Experimental Protocols

### Parameter Quick Reference

| Flag | Default | Effect |
|---|---|---|
| `-latency-stride-bytes` | `64` | Byte spacing between pointer-chain nodes. Must be a multiple of 8 (pointer size on AArch64). |
| `-latency-tlb-locality-kb` | `16` | Locality window size in KB. `0` forces global-random mode regardless of `-latency-chain-mode`. |
| `-latency-samples` | (see help) | Number of sub-samples per benchmark loop. Higher values improve statistical resolution. |
| `-latency-chain-mode` | `auto` | Chain construction policy. `auto` selects `global-random` when locality is 0, `random-box` otherwise. |

### 7.1 Baseline Main-Memory Latency (Global Random Chain)

```bash
memory_benchmark \
  -only-latency \
  -buffersize 1024 \
  -count 10 \
  -latency-samples 1000 \
  -latency-tlb-locality-kb 0 \
  -output latency_global.json
```

### 7.2 Locality-Window Latency Runs

```bash
memory_benchmark -only-latency -buffersize 1024 -count 10 -latency-samples 1000 -latency-tlb-locality-kb 16 -output latency_tlb16.json
memory_benchmark -only-latency -buffersize 1024 -count 10 -latency-samples 1000 -latency-tlb-locality-kb 2048 -output latency_tlb2048.json
memory_benchmark -only-latency -buffersize 1024 -count 10 -latency-samples 1000 -latency-tlb-locality-kb 32768 -output latency_tlb32768.json
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
- For page-pressure investigations, inspect `chain_diagnostics.unique_pages_touched` in the JSON
  output. The full paths are:
  - Main memory: `main_memory.latency.chain_diagnostics.unique_pages_touched`
  - Custom cache size: `cache.custom.latency.chain_diagnostics.unique_pages_touched`
  - L1: `cache.l1.latency.chain_diagnostics.unique_pages_touched`
  - L2: `cache.l2.latency.chain_diagnostics.unique_pages_touched`

  **Note:** `chain_diagnostics` is only emitted when `-latency-stride-bytes` is explicitly
  specified on the command line with a non-default value. Runs using the default stride will not
  include this block.

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

The latency path in this project is built around a strict dependent pointer chase with exact-count execution and tested chain setup boundaries.
It is suitable for comparative latency studies across cache sizes and locality-window policies when interpreted with robust statistics (median and tails) rather than single-point extremes.

## 11. Related Documents

- [TLB_ANALYSIS_WHITEPAPER.md](TLB_ANALYSIS_WHITEPAPER.md) — Standalone TLB boundary detection mode (`-analyze-tlb`): sweep methodology, boundary/guard rules, confidence model, and JSON contract.
- [CORE_TO_CORE_WHITEPAPER.md](CORE_TO_CORE_WHITEPAPER.md) — Core-to-Core Cache-Line Handoff Latency Benchmark (`-analyze-core2core`): LDAR/STLR assembly protocol, scheduler-hint scenarios, and JSON contract.
- [TECHNICAL_SPECIFICATION.md](TECHNICAL_SPECIFICATION.md) — Runtime architecture, execution flow, and output contracts for all benchmark modes.

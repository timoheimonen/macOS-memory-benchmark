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

Because each load address depends on the previous load result, the hardware cannot freely overlap the chain.
This forces a serialized load-to-use path that is suitable for latency characterization.

## 3. Assembly Kernel Design (`src/asm/memory_latency.s`)

### 3.1 Contract

- Prototype: `uintptr_t* memory_latency_chase_asm(uintptr_t* start_pointer, size_t count)`
- Inputs:
  - `x0` = start pointer
  - `x1` = dereference count
- Output:
  - `x0` = final pointer value after `count` dependent dereferences

### 3.2 Core Behavior

The kernel uses three key safeguards:

1. **Zero-access fast return**
   - If `count == 0`, it returns immediately.
2. **Exact-count execution with 8x unroll**
   - `count / 8` iterations of 8 dependent loads, then `count % 8` remainder loads.
3. **Counter-based unsigned-safe loop termination**
   - Uses decrement-and-branch patterns (`subs` + `b.ne`) for full `size_t` ranges.

The unrolled body remains dependency-serialized because each `ldr x0, [x0]` consumes the prior result.
Unrolling primarily reduces branch overhead per dereference; it does not remove the data dependency chain.

### 3.3 Measurement Purity Notes

The chase kernel intentionally avoids extra ordering/fence overhead in the hot path.
No pre-touch or barrier instructions are required for functional correctness of the chase loop.

## 4. Pointer-Chain Construction (`src/core/memory/memory_utils.cpp`)

`setup_latency_chain()` prepares the memory into a circular pointer chain before timing begins.

### 4.1 Input Validation

The function returns `EXIT_FAILURE` for invalid setup cases, including:

- `buffer == nullptr`
- `stride == 0`
- fewer than 2 pointers fit in the provided buffer (`buffer_size / stride < 2`)

### 4.2 Chain Construction Flow

1. Build index vector `0..num_pointers-1`
2. Shuffle indices
3. Link shuffled indices into a circular chain
4. Bounds-check each write target and next-pointer target

### 4.3 TLB-Locality Mode

`-latency-tlb-locality-kb` controls chain randomization scope:

- `0`: fully global randomization across all indices
- `>0`: locality-window mode
  - randomize inside each locality window
  - randomize the order of locality windows

This is a chain-construction policy, not a hardware TLB control primitive.

## 5. Test-Backed Correctness Guarantees (`tests/test_memory_utils.cpp`)

The following behaviors are explicitly covered by unit tests:

- **Null buffer rejected** (`SetupLatencyChainNullBuffer`)
- **Zero stride rejected** (`SetupLatencyChainZeroStride`)
- **Insufficient buffer capacity rejected** (`SetupLatencyChainBufferSmallerThanStride`, `SetupLatencyChainBufferEqualToStride`)
- **Minimum valid chain accepted** (`SetupLatencyChainMinimumValid`)
- **Small valid chains accepted** (`SetupLatencyChainThreePointers`)
- **Constructed pointers stay inside buffer bounds** (`SetupLatencyChainCreatesValidChain`)
- **Page-sized locality accepted** (`SetupLatencyChainWithTlbLocality`)
- **Too-small locality-span rejected** (`SetupLatencyChainWithTooSmallTlbLocalityFails`)

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

That pointer continuity avoids repeatedly restarting from the same chain head and reduces cache-prefix bias.

## 7. Experimental Protocols

### 7.1 Baseline Main-Memory Latency (Global Random Chain)

```bash
./memory_benchmark \
  -only-latency \
  -buffersize 1024 \
  -count 10 \
  -latency-samples 1000 \
  -latency-tlb-locality-kb 0 \
  -output latency_global.json
```

### 7.2 Locality-Window Latency Runs

```bash
./memory_benchmark -only-latency -buffersize 1024 -count 10 -latency-samples 1000 -latency-tlb-locality-kb 16 -output latency_tlb16.json
./memory_benchmark -only-latency -buffersize 1024 -count 10 -latency-samples 1000 -latency-tlb-locality-kb 2048 -output latency_tlb2048.json
./memory_benchmark -only-latency -buffersize 1024 -count 10 -latency-samples 1000 -latency-tlb-locality-kb 32768 -output latency_tlb32768.json
```

### 7.3 Cache-Scale Sweep (Custom Cache Size)

Use the repository script to sweep cache sizes with multiple locality settings:

```bash
./script-examples/latency_test_script.sh
```

Then visualize trends:

```bash
python3 script-examples/plot_cache_percentiles.py script-examples/final_output.txt --metric median
```

## 8. Reading the Results

- Prefer **median (P50)** for the trend curve.
- Use **P95/P99** to evaluate tail sensitivity and instability.
- Treat isolated very large `max` values as outlier diagnostics, not central tendency.
- Compare locality settings at the same cache size, then compare cache-size transitions.

## 9. Limitations and Non-Goals

- The benchmark runs in user space and does not create truly uncached memory mappings.
- TLB-locality windows influence chain layout, but they do not force a specific hardware TLB residency outcome.
- Observed latency includes full system effects along the dependent-load path (cache hierarchy, translation effects when present, and runtime noise).
- Cross-run variance can still occur due to scheduling, thermal behavior, and background activity.

## 10. Summary

The latency path in this project is built around a strict dependent pointer chase with exact-count execution and tested chain setup boundaries.
It is suitable for comparative latency studies across cache sizes and locality-window policies when interpreted with robust statistics (median and tails) rather than single-point extremes.

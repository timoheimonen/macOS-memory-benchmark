# macOS Apple Silicon Memory Performance Test

Copyright 2025 Timo Heimonen <timo.heimonen@proton.me>  
License: GPL-3.0 license  
  
A simple tool to measure memory write/read/copy bandwidth and memory access latency on macOS with Apple Silicon (ARM64).  
  
## Description

This tool measures memory performance on macOS Apple Silicon systems. It provides comprehensive benchmarks for:

1. **Main Memory Bandwidth**: How fast data can be read, written, and copied between large memory blocks (measures DRAM performance).
2. **Cache Bandwidth**: How fast data can be read, written, and copied within L1 and L2 cache levels.
3. **Main Memory Latency**: The average time it takes to access memory (measures DRAM access time).
4. **Cache Latency**: The average time it takes to access L1 and L2 cache levels.
5. **Access Pattern Performance**: Performance across different memory access patterns (sequential, strided, random) to analyze prefetcher effectiveness and cache behavior.

The tool automatically detects your system's cache sizes and uses optimized assembly code for accurate measurements. All tests include warm-up runs to ensure stable and consistent results.

## Why This Tool?

The primary motivation for developing this tool is to provide a straightforward and reliable method for measuring and comparing the memory performance characteristics across different generations of Apple Silicon chips (M1, M2, M3, M4, M5, etc.).

Many general-purpose benchmarks are not specifically designed to expose Apple Silicon–specific cache behavior or access patterns.
This tool focuses on:
- Native ARM64 execution (no Rosetta)
- Cache-size aware access patterns
- mmap-based memory allocation

## Disclaimer

**Use this software entirely at your own risk.** This tool performs intensive memory operations. The author(s) are not responsible for any potential system instability, data loss, or hardware issues resulting from its use.  

## Target Platform

macOS on Apple Silicon.

## Features

* **Main Memory Bandwidth**: Measures read, write, and copy speeds using all available CPU cores for maximum throughput (customizable with `-threads` parameter).
* **Cache Bandwidth**: Measures L1 and L2 cache read, write, and copy speeds. By default uses single-threaded tests for accuracy, but can be configured with `-threads` parameter.
* **Memory Latency**: Measures access latency for both main memory and cache levels.
* **Access Pattern Analysis**: Tests different memory access patterns (sequential forward/reverse, strided, random) to analyze prefetcher effectiveness and cache behavior.
* **Advanced Statistics**: When running multiple test loops, provides detailed statistics including percentiles (P50/P90/P95/P99) and standard deviation.
* **Automatic Cache Detection**: Automatically detects and uses your system's L1 and L2 cache sizes.
* **JSON Export**: Save benchmark results to JSON format for further analysis or visualization.
* **Optimized Performance**: Uses highly optimized assembly code for accurate and fast measurements.

## Install with Homebrew

In the Terminal, Run:  
```bash
brew install timoheimonen/macOS-memory-benchmark/memory-benchmark
```

## Prerequisites

* macOS (Apple Silicon).
* Xcode Command Line Tools (includes `clang++` compiler and `as` assembler).
    * Install with: `xcode-select --install` in the Terminal.

## Building

In the Terminal, go to the directory with source code. Run:

**Compile C++/ASM code:**
    ```bash
    make
    ```
This makes the program file named `memory_benchmark`.

## Testing

Prerequisites:
```bash
brew install googletest
```

Run:
```bash
make test
```

(149 unit tests covering config, buffers, memory, benchmarks, and messages.)

## Usage

In the Terminal, go to the directory with `memory_benchmark` and use these commands:

1. **Help**
    ```bash
    ./memory_benchmark -h
    ```
    Example output:
    ```text
    Version: 0.51 by Timo Heimonen <timo.heimonen@proton.me>
    License: GNU GPL v3. See <https://www.gnu.org/licenses/>
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
    Link: https://github.com/timoheimonen/macOS-memory-benchmark

    Usage: ./memory_benchmark [options]
    Options:
      -iterations <count>   Number of iterations for R/W/Copy tests (default: 1000)
      -buffersize <size_mb> Size for EACH of the 3 buffers in Megabytes (MB) as integer (default: 512).
                            The maximum allowed <size_mb> is automatically determined such that
                            3 * <size_mb> does not exceed ~80% of available system memory.
      -count <count>        Number of full loops (read/write/copy/latency) (default: 1).
                            When count > 1, statistics include percentiles (P50/P90/P95/P99) and stddev.
      -latency-samples <count> Number of latency samples to collect per test (default: 1000)
      -threads <count>      Number of threads to use for benchmarks (default: detected
                            CPU core count). Applies to all benchmarks including cache tests.
                            If specified value exceeds available cores, it will be capped to
                            the maximum number of cores with a warning.
      -cache-size <size_kb> Custom cache size in Kilobytes (KB) as integer (16 KB to 524288 KB).
                            Minimum is 16 KB (system page size). When set, skips automatic
                            L1/L2 cache size detection and only performs bandwidth and latency
                            tests for the custom cache size.
      -patterns             Run pattern benchmarks (sequential forward/reverse, strided,
                            and random access patterns). When set, only pattern benchmarks
                            are executed, skipping standard bandwidth and latency tests.
                            use with -buffersize <size_mb> to set the buffer size for the pattern benchmarks.
      -non-cacheable        Apply cache-discouraging hints to src/dst buffers.
                            Uses madvise() hints to discourage caching, but does NOT provide
                            true non-cacheable memory (user-space cannot modify page tables).
                            Best-effort approach that may reduce but not eliminate caching.
      -output <file>        Save benchmark results to JSON file. If path is relative,
                            file is saved in current working directory.
      -h, --help            Show this help message and exit

    Example: ./memory_benchmark -iterations 2000 -buffersize 1024 -output results.json
    ```
2. **Run with default parameters**
    ```bash
    ./memory_benchmark
    ```
3. **Run with custom parameters example**
    ```bash
    ./memory_benchmark -iterations 500 -buffersize 512 -cache-size 1024 -threads 4 -non-cacheable 
    ```

## Example output (Mac Mini M4 24GB)
```text
----- macOS-memory-benchmark v0.51 -----
Copyright 2025 Timo Heimonen <timo.heimonen@proton.me>
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See <https://www.gnu.org/licenses/> for more details.

Buffer Size (per buffer): 512.00 MiB (512 MB requested/capped)
Total Allocation Size: ~1536.00 MiB (for 3 buffers)
Iterations (per R/W/Copy test per loop): 1000
Loop Count (total benchmark repetitions): 1
Non-Cacheable Memory Hints: Enabled

Processor Name: Apple M4
  Performance Cores: 4
  Efficiency Cores: 6
  Total CPU Cores Detected: 10

Detected Cache Sizes:
  L1 Cache Size: 128.00 KB (per P-core)
  L2 Cache Size: 16.00 MB (per P-core cluster)

Running benchmarks...
\ Running tests...
--- Results (Loop 1) ---
Main Memory Bandwidth Tests (multi-threaded, 10 threads):
  Read : 114.74709 GB/s (Total time: 4.67873 s)
  Write: 68.55840 GB/s (Total time: 7.83086 s)
  Copy : 105.60609 GB/s (Total time: 10.16742 s)

Main Memory Latency Test (single-threaded, pointer chase):
  Total time: 19.47850 s
  Average latency: 97.39 ns

Cache Bandwidth Tests (single-threaded):
  L1 Cache:
    Read : 139.59918 GB/s (Buffer size: 96.00 KB)
    Write: 72.70513 GB/s
    Copy : 163.58213 GB/s
  L2 Cache:
    Read : 120.46392 GB/s (Buffer size: 1.60 MB)
    Write: 45.08152 GB/s
    Copy : 126.15097 GB/s

Cache Latency Tests (single-threaded, pointer chase):
  L1 Cache: 0.69 ns (Buffer size: 96.00 KB)
  L2 Cache: 4.90 ns (Buffer size: 1.60 MB)
--------------

Done. Total execution time: 43.48166 s
```
```text
----- macOS-memory-benchmark v0.51 -----
Copyright 2025 Timo Heimonen <timo.heimonen@proton.me>
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See <https://www.gnu.org/licenses/> for more details.

Buffer Size (per buffer): 512.00 MiB (512 MB requested/capped)
Total Allocation Size: ~1536.00 MiB (for 3 buffers)
Iterations (per R/W/Copy test per loop): 1000
Loop Count (total benchmark repetitions): 1
Non-Cacheable Memory Hints: Enabled

Processor Name: Apple M4
  Performance Cores: 4
  Efficiency Cores: 6
  Total CPU Cores Detected: 10

Detected Cache Sizes:
  L1 Cache Size: 128.00 KB (per P-core)
  L2 Cache Size: 16.00 MB (per P-core cluster)

Running Pattern Benchmarks...
- Running tests...
================================

Sequential Forward:
  Read : 114.756 GB/s
  Write: 65.681 GB/s
  Copy : 52.868 GB/s

Sequential Reverse:
  Read : 69.604 GB/s (-39.3%)
  Write: 38.266 GB/s (-41.7%)
  Copy : 44.317 GB/s (-16.2%)

Strided (Cache Line - 64B):
  Read : 35.567 GB/s (-69.0%)
  Write: 17.807 GB/s (-72.9%)
  Copy : 52.520 GB/s (-0.7%)

Strided (Page - 4096B):
  Read : 8.979 GB/s (-92.2%)
  Write: 20.918 GB/s (-68.2%)
  Copy : 12.740 GB/s (-75.9%)

Random Uniform:
  Read : 6.339 GB/s (-94.5%)
  Write: 21.694 GB/s (-67.0%)
  Copy : 5.572 GB/s (-89.5%)

Pattern Efficiency Analysis:
- Sequential coherence: 65.2%
- Prefetcher effectiveness: 45.4%
- Cache thrashing potential: High
- TLB pressure: Minimal


Done. Total execution time: 103.52740 s
```

![Mac Mini M4 Cache Latency from multiple JSON-files](pictures/MacMiniM4_cache_latency.png)  
Mac Mini M4 Cache Latency from multiple JSON-files

## Technical Details

This section provides implementation details for developers and those interested in the technical aspects of the benchmark.

### Implementation Overview

The benchmark is implemented in C++ with performance-critical memory operations written in ARM64 assembly. The main read/write/copy and latency tests are implemented in separate assembly files located in `src/asm/`.

### Assembly Implementation (`src/asm/`)

The performance-critical memory operations are implemented in ARM64 assembly across multiple source files:

* **Core Functions:**
    * `memory_copy.s` - `_memory_copy_loop_asm`: Copies data between buffers using non-temporal instructions (`stnp`) to minimize cache impact. Processes 512-byte blocks sequentially forward.
    * `memory_read.s` - `_memory_read_loop_asm`: Reads memory and calculates an XOR checksum to prevent optimization and ensure data is actually read. Processes 512-byte blocks sequentially forward.
    * `memory_write.s` - `_memory_write_loop_asm`: Writes zeros to memory using non-temporal instructions (`stnp`). Processes 512-byte blocks sequentially forward.
    * `memory_latency.s` - `_memory_latency_chase_asm`: Measures memory access latency via pointer chasing with dependent loads (`ldr x0, [x0]`). Uses 8-way loop unrolling.

* **Pattern Benchmark Functions:**
    * **Sequential Reverse:**
        * `memory_copy_reverse.s` - `_memory_copy_reverse_loop_asm`: Copies data backwards (reverse sequential) using 512-byte blocks with non-temporal stores.
        * `memory_read_reverse.s` - `_memory_read_reverse_loop_asm`: Reads memory backwards with XOR checksum accumulation.
        * `memory_write_reverse.s` - `_memory_write_reverse_loop_asm`: Writes zeros backwards using non-temporal stores.
    * **Strided Access:**
        * `memory_copy_strided.s` - `_memory_copy_strided_loop_asm`: Copies data using strided access patterns (64B cache line stride and 4096B page stride, tested separately) with 32-byte operations, wrapping around buffer via modulo arithmetic.
        * `memory_read_strided.s` - `_memory_read_strided_loop_asm`: Reads memory using strided access patterns (64B and 4096B strides, tested separately) with XOR checksum accumulation.
        * `memory_write_strided.s` - `_memory_write_strided_loop_asm`: Writes zeros using strided access patterns (64B and 4096B strides, tested separately) with non-temporal stores.
    * **Random Access:**
        * `memory_copy_random.s` - `_memory_copy_random_loop_asm`: Copies data using random access pattern defined by pre-generated indices array. Processes 32-byte cache lines per access to maximize cache misses and TLB pressure.
        * `memory_read_random.s` - `_memory_read_random_loop_asm`: Reads memory using random access pattern with XOR checksum accumulation.
        * `memory_write_random.s` - `_memory_write_random_loop_asm`: Writes zeros using random access pattern with non-temporal stores.

* **Key Optimizations:**
    * Processing data in large 512-byte blocks.
    * Extensive use of NEON SIMD registers (q0-q7 and q16-q31) for high throughput, avoiding callee-saved registers (q8-q15) per AAPCS64 for ABI compliance.
    * Non-temporal stores (`stnp`) to reduce cache pollution during bandwidth tests.
    * Careful register management ensuring ABI compatibility without stack operations (leaf function optimization).
    * 8-way loop unrolling in the latency test.

### Benchmark Execution Flow

The benchmark performs the following steps:

1. Detects L1 and L2 cache sizes using system calls (`sysctlbyname`).
2. Allocates memory blocks using `mmap` (source and destination buffers for bandwidth tests; latency buffer for main memory latency; separate buffers for L1/L2 cache latency).
3. Initializes bandwidth buffers to ensure the OS maps the memory.
4. Creates random pointer chains inside the latency buffers (this also maps their memory).
5. Performs warm-up runs for read/write/copy and latency tests to let the CPU speed and caches settle.
6. **Main Memory Bandwidth Tests**: Times read/write/copy operations from source to destination buffer multiple times using precise timing (multi-threaded by default, using all available logical CPU cores, customizable with `-threads` parameter).
7. **Cache Bandwidth Tests**: Times read/write/copy operations in buffers sized to fit within L1 and L2 cache levels (single-threaded by default, using more iterations for accuracy, but can be configured with `-threads` parameter).
8. **Cache Latency Tests**: Times many dependent pointer reads in buffers sized to fit within L1 and L2 cache levels.
9. **Main Memory Latency Test**: Times many dependent pointer reads (following the chain) using precise timing.
10. Calculates and displays memory bandwidth, cache latencies, and average main memory access latency.
11. Releases memory using `munmap` (via RAII with `std::unique_ptr` and a custom deleter).

### Technical Specifications

* **Memory Allocation**: Uses `mmap` for memory blocks (large blocks for bandwidth/main memory latency; cache-sized blocks for cache bandwidth and latency tests).
* **Timing**: Uses `mach_absolute_time` for precise timing measurements.
* **Threading**: Uses multiple threads (`std::thread`) for bandwidth tests. By default, main memory bandwidth tests use all available logical cores (performance cores + efficiency cores), while cache tests are single-threaded. The thread count can be customized using the `-threads` parameter, which applies to all benchmarks including cache tests. If the specified thread count exceeds available cores, it is automatically capped with a warning.
* **Optimizations**: All memory operations use optimized non-temporal pair instructions (`ldnp`/`stnp`) for high-throughput bandwidth tests. Latency tests use pointer chasing with dependent loads (`ldr x0, [x0]`).

## Known Issues and Limitations

* **Non-cacheable memory limitations**:  
  The `-non-cacheable` flag provides **cache-discouraging hints**, not true non-cacheable memory. This is a fundamental limitation of user-space applications on macOS ARM64:
  * **User-space cannot modify page table attributes** (requires kernel privileges)
  * **User-space cannot set MAIR** (Memory Attribute Indirection Register)
  * **User-space cannot create truly uncached mappings**
  
  The implementation uses `madvise(ptr, size, MADV_RANDOM)` to hint that the memory access pattern is random, which may discourage aggressive caching. However, this is a **best-effort approach** that provides hints to the memory system, not guarantees of cache-bypass behavior.
  
  **Important**: The actual cache behavior will depend on the CPU and kernel implementation. This feature may reduce but not eliminate caching. For true non-cacheable memory, kernel-level modifications would be required, which is not feasible for user-space applications.

* **Small buffer sizes (< 512 MB–1 GB) are cache-dominated**:  
  When you run the main memory bandwidth or latency tests with small buffer sizes, a significant portion of the accesses can be served from the CPU caches instead of true main memory (DRAM). This is especially true on Apple Silicon, which has large and complex shared caches. As a result, small buffers tend to measure *cache* performance rather than pure DRAM performance, and may report unrealistically high bandwidth or low latency.
* **No explicit cache flush on Apple Silicon**:  
  Unlike x86 (`CLFLUSH`), there is no user-space instruction on Apple Silicon to reliably flush data caches. The benchmark cannot force a “cold cache” state before each measurement; it relies on large buffer sizes and warm-up behavior to approximate steady-state memory behavior.
* **Recommendation**:  
  For more realistic main memory measurements, prefer buffer sizes in the range of **512 MB–1 GB (or larger, if RAM allows)**. Smaller values are still useful to study cache behavior, but they should not be interpreted as pure main memory bandwidth/latency.
# macOS Apple Silicon Memory Performance Test

Copyright 2025 Timo Heimonen <timo.heimonen@proton.me>  
License: GPL-3.0 license  
  
A simple tool to measure memory write/read/copy bandwidth and memory access latency on macOS with Apple Silicon (ARM64).  

## Disclaimer

**Use this software entirely at your own risk.** This tool performs intensive memory operations. The author(s) are not responsible for any potential system instability, data loss, or hardware issues resulting from its use.  
  
## Description

This C++/asm program measures:
1. How fast data can be read/written/copied between two big memory blocks allocated using `mmap` (main memory bandwidth).
2. How fast data can be read/written/copied in buffers sized to fit within L1 and L2 cache levels (cache bandwidth).
3. The average time (latency) it takes to do dependent memory reads in a large buffer, similar to random access (main memory latency).
4. L1 and L2 cache latency using pointer chasing methodology with buffers sized to fit within each cache level.

The main read/write/copy and latency tests are done in separate ARM64 assembly files located in `src/asm/`. The read/write/copy loops use optimized non-temporal instructions (`ldnp`/`stnp`) for faster testing, and the latency test uses dependent loads (`ldr x0, [x0]`) to measure access time.

### Assembly Implementation (`src/asm/`)

The performance-critical memory operations are implemented in ARM64 assembly for maximum efficiency across four separate source files:

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

The benchmark does these steps:

1.  Detects L1 and L2 cache sizes using system calls (`sysctlbyname`).
2.  Gets memory blocks using `mmap` (src, dst for bandwidth; lat for main memory latency; separate buffers for L1/L2 cache latency).
3.  Writes to the bandwidth buffers to make sure the OS maps the memory.
4.  Creates random pointer chains inside the latency buffers (this also maps their memory).
5.  Does warm-up runs for read/write/copy and latency tests to let the CPU speed and caches settle.
6.  Main Memory Bandwidth Tests: Times the read/write/copy from source to destination buffer multiple times using the precise `mach_absolute_time` timer (multi-threaded, using all available logical CPU cores).
7.  Cache Bandwidth Tests: Times read/write/copy operations in buffers sized to fit within L1 and L2 cache levels (single-threaded, using 10x more iterations for accuracy).
8.  Cache Latency Tests: Times doing many dependent pointer reads in buffers sized to fit within L1 and L2 cache levels.
9.  Main Memory Latency Test: Times doing many dependent pointer reads (following the chain) using `mach_absolute_time`.
10. Calculates and shows the memory bandwidth, cache latencies, and the average main memory access latency.
11. Releases the memory using `munmap` (via RAII with `std::unique_ptr` and a custom deleter).

## Why This Tool?

The primary motivation for developing this tool is to provide a straightforward and reliable method for measuring and comparing the memory performance characteristics across different generations of Apple Silicon chips (M1, M2, M3, M4, M5, etc.).

## Target Platform

macOS on Apple Silicon.

## Features

* Checks main memory read, write and copy speeds (multi-threaded, using all available logical CPU cores).
* Checks L1 and L2 cache bandwidth (read/write/copy) using single-threaded tests.
* Checks L1 and L2 cache latency using pointer chasing methodology.
* Checks main memory access latency.
* Check memory performance with different access patterns (sequential forward/reverse, strided 64B and 4096B tested separately, random). Measures read/write/copy bandwidth for each pattern using optimized ARM64 assembly.
* When running with multiple loops (`-count > 1`), calculates detailed statistics including percentiles (P50/P90/P95/P99) and standard deviation for bandwidth and latency tests.
* Automatically detects cache sizes (L1, L2) for Apple Silicon processors.
* Uses `mmap` for memory blocks (large blocks for bandwidth/main memory latency; cache-sized blocks for cache bandwidth and latency tests).
* All memory operations are implemented in optimized ARM64 assembly.
* Uses optimized non-temporal pair instructions (`ldnp`/`stnp`) for high-throughput bandwidth tests in the assembly loop.
* Checks latency by pointer chasing with dependent loads (`ldr x0, [x0]`) in assembly.
* Uses multiple threads (`std::thread`) for main memory bandwidth tests, with thread count equal to total logical cores (performance cores + efficiency cores). Cache tests are single-threaded.
* Uses `mach_absolute_time` for precise timing.
* Initializes memory and does warm-ups for more stable results.
* Save results to JSON-file

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

(31 unit tests covering config, buffers, memory, benchmarks.)

## Usage

In the Terminal, go to the directory with `memory_benchmark` and use these commands:

1. **Help**
    ```bash
    ./memory_benchmark -h
    ```
    Example output:
    ```text
    Version: 0.50 by Timo Heimonen <timo.heimonen@proton.me>
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
                            See "Non-Cacheable Memory Limitations" section below for details.
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
    ./memory_benchmark -iterations 500 -buffersize 512 -cache-size 1024
    ```

## Example output (Mac Mini M4 24GB)
```text
----- macOS-memory-benchmark v0.50 -----
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
  Read : 116.539 GB/s (Total time: 4.607 s)
  Write: 66.293 GB/s (Total time: 8.098 s)
  Copy : 106.634 GB/s (Total time: 10.069 s)

Main Memory Latency Test (single-threaded, pointer chase):
  Total time: 19.720 s
  Average latency: 98.60 ns

Cache Bandwidth Tests (single-threaded):
  L1 Cache:
    Read : 139.461 GB/s (Buffer size: 96.00 KB)
    Write: 72.589 GB/s
    Copy : 176.591 GB/s
  L2 Cache:
    Read : 122.078 GB/s (Buffer size: 1.60 MB)
    Write: 44.904 GB/s
    Copy : 129.392 GB/s

Cache Latency Tests (single-threaded, pointer chase):
  L1 Cache: 0.69 ns (Buffer size: 96.00 KB)
  L2 Cache: 4.86 ns (Buffer size: 1.60 MB)
--------------

Done. Total execution time: 43.826 s
```
```text
----- macOS-memory-benchmark v0.50 -----
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
  Read : 115.410 GB/s
  Write: 66.226 GB/s
  Copy : 53.140 GB/s

Sequential Reverse:
  Read : 70.693 GB/s (-38.7%)
  Write: 37.107 GB/s (-44.0%)
  Copy : 44.577 GB/s (-16.1%)

Strided (Cache Line - 64B):
  Read : 35.190 GB/s (-69.5%)
  Write: 18.697 GB/s (-71.8%)
  Copy : 53.308 GB/s (+0.3%)

Strided (Page - 4096B):
  Read : 9.000 GB/s (-92.2%)
  Write: 23.320 GB/s (-64.8%)
  Copy : 13.001 GB/s (-75.5%)

Random Uniform:
  Read : 6.344 GB/s (-94.5%)
  Write: 21.539 GB/s (-67.5%)
  Copy : 5.582 GB/s (-89.5%)

Pattern Efficiency Analysis:
- Sequential coherence: 64.9%
- Prefetcher effectiveness: 45.7%
- Cache thrashing potential: High
- TLB pressure: Minimal


Done. Total execution time: 102.93132 s
```

![Mac Mini M4 Cache Latency from multiple JSON-files](pictures/MacMiniM4_cache_latency.png)  
Mac Mini M4 Cache Latency from multiple JSON-files

## Non-Cacheable Memory Limitations

The `-non-cacheable` flag provides **cache-discouraging hints**, not true non-cacheable memory. This is a fundamental limitation of user-space applications on macOS ARM64:

* **User-space cannot modify page table attributes** (requires kernel privileges)
* **User-space cannot set MAIR** (Memory Attribute Indirection Register)
* **User-space cannot create truly uncached mappings**

The implementation uses `madvise(ptr, size, MADV_RANDOM)` to hint that the memory access pattern is random, which may discourage aggressive caching. However, this is a **best-effort approach** that provides hints to the memory system, not guarantees of cache-bypass behavior.

**Important**: The actual cache behavior will depend on the CPU and kernel implementation. This feature may reduce but not eliminate caching. For true non-cacheable memory, kernel-level modifications would be required, which is not feasible for user-space applications.

## Known Issues and Limitations

* **Small buffer sizes (< 512 MB–1 GB) are cache-dominated**:  
  When you run the main memory bandwidth or latency tests with small buffer sizes, a significant portion of the accesses can be served from the CPU caches instead of true main memory (DRAM). This is especially true on Apple Silicon, which has large and complex shared caches. As a result, small buffers tend to measure *cache* performance rather than pure DRAM performance, and may report unrealistically high bandwidth or low latency.
* **No explicit cache flush on Apple Silicon**:  
  Unlike x86 (`CLFLUSH`), there is no user-space instruction on Apple Silicon to reliably flush data caches. The benchmark cannot force a “cold cache” state before each measurement; it relies on large buffer sizes and warm-up behavior to approximate steady-state memory behavior.
* **Recommendation**:  
  For more realistic main memory measurements, prefer buffer sizes in the range of **512 MB–1 GB (or larger, if RAM allows)**. Smaller values are still useful to study cache behavior, but they should not be interpreted as pure main memory bandwidth/latency.
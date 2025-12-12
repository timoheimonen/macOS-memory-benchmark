# macOS Apple Silicon Memory Performance Test

Copyright 2025 Timo Heimonen <timo.heimonen@proton.me>  
License: GPL-3.0 license  
  
A simple tool to measure memory write/read/copy bandwidth and memory access latency on macOS with Apple Silicon (ARM64).  

## Disclaimer

**Use this software entirely at your own risk.** This tool performs intensive memory operations. The author(s) are not responsible for any potential system instability, data loss, or hardware issues resulting from its use.  
  
## Description

This C++/asm program measures:
1. How fast data can be written/read/copied between two big memory blocks allocated using `mmap`.
2. The average time (latency) it takes to do dependent memory reads in a large buffer, similar to random access.
3. L1 and L2 cache latency using pointer chasing methodology with buffers sized to fit within each cache level.

The main write/read/copy and latency tests are done in a separate ARM64 assembly file (`loops.s`). The write/read/copy loop uses optimized non-temporal instructions (`ldnp`/`stnp`) for faster testing, and the latency test uses dependent loads (`ldr x0, [x0]`) to measure access time.

### Assembly Implementation (`loops.s`)

The performance-critical memory operations are implemented in ARM64 assembly for maximum efficiency:

* **Core Functions:**
    * `_memory_copy_loop_asm`: Copies data between buffers using non-temporal instructions (`stnp`) to minimize cache impact.
    * `_memory_read_loop_asm`: Reads memory and calculates an XOR checksum to prevent optimization and ensure data is actually read.
    * `_memory_write_loop_asm`: Writes zeros to memory using non-temporal instructions (`stnp`).
    * `_memory_latency_chase_asm`: Measures memory access latency via pointer chasing with dependent loads (`ldr x0, [x0]`).  

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
5.  Does warm-up runs for write/read/copy and latency tests to let the CPU speed and caches settle.
6.  Bandwidth Test: Times the **write**/read/copy from source to destination buffer multiple times using the precise `mach_absolute_time` timer.
7.  Cache Latency Tests: Times doing many dependent pointer reads in buffers sized to fit within L1 and L2 cache levels.
8.  Main Memory Latency Test: Times doing many dependent pointer reads (following the chain) using `mach_absolute_time`.
9.  Calculates and shows the memory bandwidth, cache latencies, and the average main memory access latency.
10. Releases the memory using `munmap` (via RAII with `std::unique_ptr` and a custom deleter).

## Why This Tool?

The primary motivation for developing this tool is to provide a straightforward and reliable method for measuring and comparing the memory performance characteristics across different generations of Apple Silicon chips (M1, M2, M3, M4, etc.).

## Target Platform

macOS on Apple Silicon.

## Features

* Checks memory write, read and copy speeds.
* Checks L1 and L2 cache latency using pointer chasing methodology.
* Checks main memory access latency.
* Automatically detects cache sizes (L1, L2) for Apple Silicon processors.
* Uses `mmap` for memory blocks (large blocks for bandwidth/main memory latency; cache-sized blocks for cache latency tests).
* Main write/read/copy and latency loops are in ARM64 assembly (`loops.s`).
* Uses optimized non-temporal pair instructions (`ldnp`/`stnp`) for high-throughput **bandwidth tests** in the assembly loop.
* Checks latency by pointer chasing with dependent loads (`ldr x0, [x0]`) in assembly.
* Uses multiple threads (`std::thread`) for the bandwidth test.
* Uses `mach_absolute_time` for precise timing.
* Initializes memory and does warm-ups for more stable results.

## Prerequisites

* macOS (Apple Silicon).
* Xcode Command Line Tools (includes `clang++` compiler and `as` assembler).
    * Install with: `xcode-select --install` in the Terminal.

## Install with Homebrew

In the Terminal, Run:  
```bash
brew install timoheimonen/macOS-memory-benchmark/memory-benchmark
```

## Building

In the Terminal, go to the directory with source code. Run:

1.  **Compile C++/ASM code:**
    ```bash
    make
    ```
This makes the program file named `memory_benchmark`.

## Usage

In the Terminal, go to the directory with `memory_benchmark` and use these commands:

1. **Help**
    ```bash
    ./memory_benchmark -h
    ```
    Example output:
    ```text
    Version: 0.4 by Timo Heimonen <timo.heimonen@proton.me>
    License: GNU GPL v3. See <https://www.gnu.org/licenses/>
    Link: https://github.com/timoheimonen/macOS-memory-benchmark

    Usage: ./memory_benchmark [options]
    Options:
      -iterations <count>   Number of iterations for R/W/Copy tests (default: 1000)
      -buffersize <size_mb> Size for EACH of the 3 buffers in Megabytes (MB) as integer (default: 512).
                            The maximum allowed <size_mb> is automatically determined such that
                            3 * <size_mb> does not exceed ~80% of available system memory.
      -count <count>        Number of full loops (read/write/copy/latency) (default: 1)
      -h, --help            Show this help message and exit

    Example: ./memory_benchmark -iterations 500 -buffersize 1024
    ```
2. **Run with default parameters**
    ```bash
    ./memory_benchmark
    ```
3. **Run with custom parameters example**
    ```bash
    ./memory_benchmark -iterations 500 -buffersize 512
    ```

## Example output (Mac Mini M4 24GB)
```text
----- macOS-memory-benchmark v0.42 -----
Program is licensed under GNU GPL v3. See <https://www.gnu.org/licenses/>
Copyright 2025 Timo Heimonen <timo.heimonen@proton.me>

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
| Running tests...
--- Results (Loop 1) ---
Bandwidth Tests (multi-threaded, 10 threads):
  Read : 115.639 GB/s (Total time: 4.643 s)
  Write: 66.164 GB/s (Total time: 8.114 s)
  Copy : 106.756 GB/s (Total time: 10.058 s)

Cache Bandwidth Tests (single-threaded):
  L1 Cache:
    Read : 135.919 GB/s (Buffer size: 96.00 KB)
    Write: 72.616 GB/s
    Copy : 176.416 GB/s
  L2 Cache:
    Read : 112.459 GB/s (Buffer size: 12.00 MB)
    Write: 37.651 GB/s
    Copy : 122.726 GB/s

Cache Latency Tests (single-threaded, pointer chase):
  L1 Cache: 0.68 ns (Buffer size: 96.00 KB)
  L2 Cache: 4.83 ns (Buffer size: 1.60 MB)

Main Memory Latency Test (single-threaded, pointer chase):
  Total time: 19.797 s
  Average latency: 98.99 ns
--------------

Done. Total execution time: 50.122 s
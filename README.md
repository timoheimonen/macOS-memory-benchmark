# macOS Apple Silicon Memory Performance Test

Copyright 2025 Timo Heimonen <timo.heimonen@gmail.com>  
License: GPL-3.0 license  
  
A simple tool to measure memory write/read/copy bandwidth and memory access latency on macOS with Apple Silicon (ARM64).  

## Disclaimer

**Use this software entirely at your own risk.** This tool performs intensive memory operations. The author(s) are not responsible for any potential system instability, data loss, or hardware issues resulting from its use.  
  
## Description

This C++/asm program measures:
1. How fast data can be written/read/copied between two big memory blocks allocated using `mmap`.
2. The average time (latency) it takes to do dependent memory reads in a large buffer, similar to random access.

The main write/read/copy and latency tests are done in a separate ARM64 assembly file (`loops.s`). The write/read/copy loop uses optimized non-temporal instructions (`ldnp`/`stnp`) for faster testing, and the latency test uses dependent loads (`ldr x0, [x0]`) to measure access time.

The benchmark does these steps:

1.  Gets three large memory blocks using `mmap` (src, dst for bandwidth; lat for latency).
2.  Writes to the bandwidth buffers to make sure the OS maps the memory.
3.  Creates a random pointer chain inside the latency buffer (this also maps its memory).
4.  Does warm-up runs for write/read/copy and latency tests to let the CPU speed and caches settle.
5.  Bandwidth Test: Times the wirte/read/copy from source to destination buffer multiple times using the precise `mach_absolute_time` timer.
6.  Latency Test: Times doing many dependent pointer reads (following the chain) using `mach_absolute_time`.
7.  Calculates and shows the memory bandwidth and the average memory access latency.
8.  Releases the memory using `munmap`.

## Target Platform

macOS on Apple Silicon (M1, M2, M3, M4, etc.).

## Features

* Checks memory write, read and copy speeds.
* Checks memory access latency.
* Uses `mmap` for big memory blocks (bigger than CPU caches).
* Main write/read/copy and latency loops are in ARM64 assembly (`loops.s`).
* Uses optimized non-temporal pair instructions (`ldnp`/`stnp`) for high-throughput copying in the assembly loop.
* Checks latency by pointer chasing with dependent loads (`ldr x0, [x0]`) in assembly.
* Uses multiple threads (`std::thread`) for the bandwidth test.
* Uses `mach_absolute_time` for precise timing.
* Initializes memory and does warm-ups for more stable results.

## Prerequisites

* macOS (Apple Silicon).
* Xcode Command Line Tools (includes `clang++` compiler and `as` assembler).
    * Install with: `xcode-select --install` in the Terminal.

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
    Usage: ./memory_benchmark [options]
    Version: 0.25

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
----- macOS-memory-benchmark v0.25 -----
Program is licensed under GNU GPL v3. See <https://www.gnu.org/licenses/>
Copyright 2025 Timo Heimonen <timo.heimonen@gmail.com>

Buffer Size (per buffer): 512.00 MiB (512 MB requested/capped)
Total Allocation Size: ~1536.00 MiB (for 3 buffers)
Iterations (per R/W/Copy test per loop): 1000
Loop Count (total benchmark repetitions): 1

Processor Name: Apple M4
  Performance Cores: 4
  Efficiency Cores: 6
  Total CPU Cores Detected: 10

--- Allocating Buffers ---
Allocating src buffer (512.00 MiB)...
Allocating dst buffer (512.00 MiB)...
Allocating lat buffer (512.00 MiB)...
Buffers allocated.
Initializing src/dst buffers...
Src/Dst buffers initialized.
Setting up pointer chain (stride 128 bytes, 4194304 pointers)...
Pointer chain setup complete.

Performing warm-up runs...
  Read warm-up...
  Write warm-up...
  Copy warm-up...
  Latency warm-up (single thread)...
Warm-up complete.

--- Starting Measurements (1 loops) ---

Starting Loop 1 of 1...
Measuring Read Bandwidth...
Read complete.
Measuring Write Bandwidth...
Write complete.
Measuring Copy Bandwidth...
Copy complete.
Measuring Latency (single thread)...
Latency complete.

--- Results (Loop 1) ---
Bandwidth Tests (multi-threaded, 10 threads):
  Read : 111.822 GB/s (Total time: 4.801 s)
  Write: 66.096 GB/s (Total time: 8.123 s)
  Copy : 105.599 GB/s (Total time: 10.168 s)

Latency Test (single-threaded, pointer chase):
  Total time: 19.744 s
  Total accesses: 200000000
  Stride: 128 bytes
  Average latency: 98.72 ns
--------------

Freeing memory...
Memory freed.

Done. Total execution time: 43.273 s
# macOS ARM64 Memory Performance Test

A simple tool to measure memory write/read/copy bandwidth and memory access latency on macOS with Apple Silicon (ARM64).

## Description

This C++/asm program measures:
1. How fast data can be copied between two big memory blocks allocated using `mmap`.
2. The average time (latency) it takes to do dependent memory reads in a large buffer, similar to random access.

The main copy and latency tests are done in a separate ARM64 assembly file (`loops.s`). The copy loop uses optimized non-temporal instructions (`ldnp`/`stnp`) for faster copying, and the latency test uses dependent loads (`ldr x0, [x0]`) to measure access time.

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

In the Terminal, go to the directory with `main.cpp` and `loops.s`. Run these commands:

1.  **Compile C++ code:**
    ```bash
    clang++ -O3 -std=c++17 -c main.cpp -o main.o -arch arm64
    ```

2.  **Assemble assembly code:**
    ```bash
    as loops.s -o loops.o -arch arm64
    ```

3.  **Link to create the program:**
    ```bash
    # Note: -pthread might be needed on some systems, but often not with clang++ on macOS
    clang++ main.o loops.o -o memory_benchmark -arch arm64 -pthread
    ```
This makes the program file named `memory_benchmark`.

## Example output (Mac Mini M4 24GB)
```text
--- macOS-memory-benchmark v0.16 ---
by Timo Heimonen <timo.heimonen@gmail.com>
Buffer Size: 512 MiB

--- Allocating Buffers ---
Allocating src buffer (512 MiB)...
Allocating dst buffer (512 MiB)...
Allocating lat buffer (512 MiB)...
Buffers allocated.

Initializing src/dst buffers...
Src/Dst buffers initialized.
Setting up pointer chain (stride 128 bytes, 4194304 pointers)...
Pointer chain setup complete.

Performing warm-up runs (10 threads for bandwidth)...
  Read warm-up...
  Write warm-up...
  Copy warm-up...
  Latency warm-up (single thread)...
Warm-up complete.

--- Starting Measurements (10 threads, 500 iterations each) ---
Measuring Read Bandwidth...
Read complete.
Measuring Write Bandwidth...
Write complete.
Measuring Copy Bandwidth...
Copy complete.
Measuring Latency...
Latency complete.

--- Results ---
Configuration:
  Buffer Size: 512 MiB
  Iterations: 500
  Threads: 10

Bandwidth Tests (multi-threaded):
  Read : 112.143 GB/s
         (Total time: 2.39369 s)
  Write: 65.4454 GB/s
         (Total time: 4.10167 s)
  Copy : 105.528 GB/s
         (Total time: 5.08745 s)

Latency Test (single-threaded):
  Total time: 19.6304 s
  Total accesses: 200000000
  Average latency: 98.152 ns
--------------

Freeing memory...
Memory freed.
Done.
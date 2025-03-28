# macOS ARM64 Memory Performance Test

A simple tool to measure memory copy bandwidth and memory access latency on macOS with Apple Silicon (ARM64).

## Description

This C++/asm program measures:
1. How fast data can be copied between two big memory blocks allocated using `mmap`.
2. The average time (latency) it takes to do dependent memory reads in a large buffer, similar to random access.

The main copy and latency tests are done in a separate ARM64 assembly file (`loops.s`). The copy loop uses optimized non-temporal instructions (`ldnp`/`stnp`) for faster copying, and the latency test uses dependent loads (`ldr x0, [x0]`) to measure access time.

The benchmark does these steps:

1.  Gets three large memory blocks using `mmap` (src, dst for bandwidth; lat for latency).
2.  Writes to the bandwidth buffers to make sure the OS maps the memory.
3.  Creates a random pointer chain inside the latency buffer (this also maps its memory).
4.  Does warm-up runs for both copy and latency tests to let the CPU speed and caches settle.
5.  Bandwidth Test: Times the copy from source to destination buffer multiple times using the precise `mach_absolute_time` timer.
6.  Latency Test: Times doing many dependent pointer reads (following the chain) using `mach_absolute_time`.
7.  Calculates and shows the memory bandwidth and the average memory access latency.
8.  Releases the memory using `munmap`.

## Target Platform

macOS on Apple Silicon (M1, M2, M3, M4, etc.).

## Features

* Checks memory copy speed (Read + Write).
* Checks memory access latency.
* Uses `mmap` for big memory blocks (bigger than CPU caches).
* Main copy and latency loops are in ARM64 assembly (`loops.s`).
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
--- macOS-memory-benchmark v0.13 ---
by Timo Heimonen <timo.heimonen@gmail.com>

--- Bandwidth Test Setup ---
Buffer size: 512 MiB
Iterations: 500
Allocating bandwidth buffers (1024 MiB total)...
Bandwidth buffers allocated.

--- Latency Test Setup ---
Buffer size: 512 MiB
Stride: 128 bytes
Accesses: 200000000
Allocating latency buffer (512 MiB)...
Latency buffer allocated.

Initializing bandwidth buffers...
Bandwidth buffers initialized.
Setting up pointer chain (stride 128 bytes, 4194304 pointers)...
Pointer chain setup complete.

Performing bandwidth warm-up run...
Bandwidth warm-up complete.
Performing latency warm-up run...
Latency warm-up complete.

Starting bandwidth measurement...
Bandwidth measurement complete.

Starting latency measurement...
Latency measurement complete.

--- Results ---
Bandwidth Test:
  Total time: 5.93758 s
  Data copied: 536.871 GB
  Memory bandwidth (copy): 90.4192 GB/s

Latency Test:
  Total time: 19.6515 s
  Total accesses: 200000000
  Average latency: 98.2576 ns
--------------

Freeing memory...
Memory freed.
Done.
# macOS ARM64 Memory Bandwidth Test

A simple benchmark tool designed to measure memory copy bandwidth and memory access latency on macOS systems running on Apple Silicon (ARM64) processors.

## Description

This project provides a basic C++ program that measures:
1. The speed of copying data between two large memory buffers allocated using `mmap`.
2. The average time (latency) for dependent memory reads across a large buffer, simulating random access patterns.

The core memory operations (copy and latency measurement) are implemented in a separate ARM64 assembly file (`loops.s`). The copy loop utilizes NEON instructions for potentially higher throughput, while the latency test uses dependent loads to measure access time accurately.

The benchmark performs the following steps:

1.  Allocates three large memory buffers using `mmap` (src, dst for bandwidth; lat for latency).
2.  Initializes the bandwidth buffer memory ("touches" each page) to ensure physical pages are mapped by the OS.
3.  Sets up a pseudo-random pointer chain within the latency buffer (also ensures page mapping).
4.  Performs separate warm-up runs for both copy and latency tests to allow CPU frequency scaling and cache/TLB stabilization.
5.  Bandwidth Test: Measures the time taken to copy the contents of the source buffer to the destination buffer repeatedly using a high-resolution timer (`mach_absolute_time`).
6.  Latency Test: Measures the time taken to perform a large number of dependent pointer loads (chasing the chain) using `mach_absolute_time`.
7.  Calculates and prints the effective memory bandwidth and the average memory access latency.
8.  Frees all allocated memory using `munmap`.

## Target Platform

macOS on Apple Silicon (ARM64) processors (e.g., M1, M2, M3, M4 series).

## Features

* Measures memory copy bandwidth (Read + Write).
* Measures memory access latency.
* Uses `mmap` for large memory allocation (intended to exceed caches).
* Core copy loop and latency loop implemented in hand-written ARM64 assembly (`loops.s`).
* Utilizes NEON vector instructions (`ldr`/`str` on `q` registers) in the copy loop.
* Latency measured via pointer chasing using dependent loads (`ldr x0, [x0]`) in assembly.
* Employs `mach_absolute_time` for high-resolution timing.
* Includes memory initialization/setup and warm-up phases for more stable results.

## Prerequisites

* macOS (Apple Silicon).
* Xcode Command Line Tools: Provides the `clang++` compiler and `as` assembler.
    * Install them by running: `xcode-select --install` in the Terminal.

## Building

Open your Terminal and navigate to the directory containing `main.cpp` and `loops.s`. Run the following commands step-by-step:

1.  **Compile the C++ source file into an object file:** 
    ```bash
    clang++ -O3 -std=c++17 -c main.cpp -o main.o -arch arm64
    ```

2.  **Assemble the assembly code into an object file:** 
    ```bash
    as loops.s -o loops.o -arch arm64
    ```

3.  **Link the object files into the final executable:** 
    ```bash
    clang++ main.o loops.o -o memory_benchmark -arch arm64
    ```
This sequence will create an executable file named `memory_benchmark`.

## Example output (Mac Mini M4 24GB)
```text
--- Memory Performance Test v0.12 ---

--- Bandwidth Test Setup ---
Buffer size: 512 MiB
Iterations: 100
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
  Total time: 1.23802 s
  Data copied: 107.374 GB
  Memory bandwidth (copy): 86.7308 GB/s

Latency Test:
  Total time: 19.4706 s
  Total accesses: 200000000
  Average latency: 97.3529 ns
--------------

Freeing memory...
Memory freed.
Done.
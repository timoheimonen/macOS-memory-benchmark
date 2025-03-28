# macOS ARM64 Memory Bandwidth Test  

A simple benchmark tool designed to measure memory copy bandwidth on macOS systems running on Apple Silicon (ARM64) processors.  

## Description  

This project provides a basic C++ program that measures the speed of copying data between two large memory buffers allocated using `mmap`. The core memory copy operation is implemented in a separate ARM64 assembly file (`loops.s`) which utilizes NEON instructions for potentially higher throughput compared to a standard C library `memcpy`.  

The benchmark performs the following steps:  

1.  Allocates two large memory buffers using `mmap`.  
2.  Initializes the memory ("touches" each page) to ensure physical pages are mapped by the OS.  
3.  Performs a warm-up run to allow CPU frequency scaling and cache/TLB stabilization.  
4.  Measures the time taken to copy the contents of the source buffer to the destination buffer repeatedly using a high-resolution timer (`mach_absolute_time`).  
5.  Calculates and prints the effective memory bandwidth based on the total data transferred (read + write) and the elapsed time.  
6.  Frees the allocated memory using `munmap`.  

## Target Platform  
  
macOS on Apple Silicon (ARM64) processors (e.g., M1, M2, M3, M4 series).  
  
## Features  

* Measures memory copy bandwidth (Read + Write).  
* Uses `mmap` for large memory allocation (avoids cache effects).  
* Core copy loop implemented in hand-written ARM64 assembly.  
* Utilizes NEON vector instructions (`ldr`/`str` on `q` registers) in the assembly loop.  
* Employs `mach_absolute_time` for high-resolution timing.  
* Includes memory initialization and warm-up phases for more stable results.  

## Prerequisites  

* macOS (Apple Silicon recommended).  
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
'''text
--- Memory Speed Test ---  
Buffer size: 512 MiB  
Iterations / measurement: 100  
Allocating memory (1024 MiB)...  
Memory allocated.  
Initializing memory...  
Memory initialized.  
Performing warm-up run...  
Warm-up complete.  
Starting measurement...  
Measurement complete.  
--- Results ---  
Total time: 1.21302 s  
Data transferred: 107.374 GB  
Memory bandwidth (copy): 88.5178 GB/s  
--------------  
Freeing memory...  
Done.
```
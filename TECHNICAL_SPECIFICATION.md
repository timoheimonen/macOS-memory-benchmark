# Technical Specification: macOS-memory-benchmark Tool (macOS Apple Silicon)

## 1. Introduction

This document describes the technical structure and implementation of a memory performance measurement tool optimized for the Apple Silicon (ARM64) architecture. The software is designed to provide precise data on memory bus bandwidth, access latencies, and cache hierarchy behavior by utilizing low-level optimizations.

## 2. System Architecture

The software is implemented modularly in C++, with performance-critical sections written in hand-optimized ARM64 assembly. The architecture follows a strict separation of concerns and utilizes the RAII (Resource Acquisition Is Initialization) design pattern for resource management.

### 2.1 Memory Management ([`src/core/memory/`](src/core/memory/))

The system employs advanced memory management techniques to ensure the reproducibility of measurement results:

- **Smart Memory Allocation**: Utilizes `mmap` calls wrapped in `MmapPtr` for automatic resource deallocation. See [`memory_manager.h`](src/core/memory/memory_manager.h) and [`memory_manager.cpp`](src/core/memory/memory_manager.cpp).
- **MmapDeleter**: A specialized custom deleter that prevents memory leaks even in the event of exceptions.
- **Cache Control**: Implements `madvise()` hints to simulate non-cacheable memory (acknowledging user-space limitations).
- **Buffer Manager**: Provides centralized management for all test buffers (main memory, L1/L2 caches). See [`buffer_manager.h`](src/core/memory/buffer_manager.h) and [`buffer_manager.cpp`](src/core/memory/buffer_manager.cpp).

## 3. Low-Level Implementation (ARM64 Assembly)

The core of the performance measurement resides in the [`src/asm/`](src/asm/) directory, specifically optimized for the Apple Silicon microarchitecture:

### 3.1 Sequential Operations

- **Read**: 512-byte block reads with XOR checksums to prevent dead code elimination by the compiler. See [`memory_read.s`](src/asm/memory_read.s).
- **Write**: Utilizes non-temporal stores (`stnp`) to minimize cache pollution. See [`memory_write.s`](src/asm/memory_write.s).
- **Copy**: Optimized using pair loads and stores to achieve maximum throughput. See [`memory_copy.s`](src/asm/memory_copy.s).

### 3.2 Latency Measurements

- **Pointer-chasing**: Implemented with 8-way loop unrolling. See [`memory_latency.s`](src/asm/memory_latency.s).
- **Memory Barriers**: `dsb`, `isb`, and `dmb` instructions are used to strictly isolate measurement windows.
- **TLB Pre-touch**: The Translation Lookaside Buffer (TLB) is pre-loaded to ensure address translation does not contaminate latency measurements.
- **Dependencies**: Built with true data dependencies, ensuring the measurement reflects actual latency rather than throughput.

### 3.3 Register Usage and SIMD

- Utilizes NEON SIMD registers (`q0-q7`, `q16-q31`) for parallel data movement.
- Complies with the AAPCS64 ABI standard by avoiding the use of callee-saved registers (`q8-q15`).
- 4-way XOR accumulation is used to reduce the dependency depth between instructions.

## 4. Benchmark Logic and Execution

The measurement process is divided into specialized executors. See [`src/benchmark/`](src/benchmark/) for implementation details:

- **Multi-threading**: Bandwidth tests utilize all available system cores by default. See [`bandwidth_tests.cpp`](src/benchmark/bandwidth_tests.cpp) and [`parallel_test_framework.h`](src/benchmark/parallel_test_framework.h).
- **Single-threading**: Latency tests are executed on a single thread to ensure maximum accuracy. See [`latency_tests.cpp`](src/benchmark/latency_tests.cpp).
- **Statistical Analysis**: Results are aggregated across multiple runs, calculating means and percentiles (P50, P90, P95, P99). See [`benchmark_results.cpp`](src/benchmark/benchmark_results.cpp) and [`benchmark_runner.cpp`](src/benchmark/benchmark_runner.cpp).

### Test Types

- **Main Memory Bandwidth**: Read, write, and copy operations at the DRAM level. See [`benchmark_executor.cpp`](src/benchmark/benchmark_executor.cpp).
- **Cache Bandwidth**: Specific throughput tests for L1 and L2 caches.
- **Access Patterns**: Reverse sequential, strided, and random access patterns for analyzing prefetcher efficiency and cache behavior. See [`src/pattern_benchmark/`](src/pattern_benchmark/) and assembly files like [`memory_read_reverse.s`](src/asm/memory_read_reverse.s), [`memory_read_strided.s`](src/asm/memory_read_strided.s), and [`memory_read_random.s`](src/asm/memory_read_random.s).

## 5. System Integration

- **Timing**: Uses the macOS `mach_absolute_time()` interface for nanosecond-precision timing. See [`timer.h`](src/core/timing/timer.h) and [`timer.cpp`](src/core/timing/timer.cpp).
- **Hardware Detection**: Automatically detects L1/L2 cache sizes and CPU models via `sysctlbyname` calls. See [`system_info.h`](src/core/system/system_info.h) and [`system_info.cpp`](src/core/system/system_info.cpp).
- **QoS Configuration**: Sets the system Quality of Service (QoS) state to levels appropriate for latency-critical testing.

## 6. Output and Reporting

The software supports two primary output formats:

- **Console**: A structured messaging system providing real-time statistical analysis. See [`src/output/console/`](src/output/console/).
- **JSON Export**: Comprehensive structured data containing system information, configuration details, and raw measurements for external analysis. See [`src/output/json/`](src/output/json/).

## 7. Technical Constraints and Considerations

- **Platform**: Designed exclusively for the macOS Apple Silicon platform.
- **User-Space Limitations**: As a user-space application, the tool does not have direct access to explicit cache flushing (cache flush) or true non-cacheable memory.
- **Buffer Sizes**: DRAM measurements require large buffer sizes (recommended > 512 MB) to minimize the impact of the cache hierarchy.

## 8. Quality Assurance

The software includes a comprehensive testing suite:

- Over 190 unit tests using the GoogleTest framework. See [`tests/`](tests/) directory.
- Doxygen-compliant documentation at the source code level.
- Validation logic for all user inputs and detected system values.

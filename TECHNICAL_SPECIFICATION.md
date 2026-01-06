# Technical Specification: macOS-memory-benchmark Tool (macOS Apple Silicon)

## 1. Introduction

This document describes the technical structure and implementation of a memory performance measurement tool optimized for the Apple Silicon (ARM64) architecture. The software is designed to provide precise data on memory bus bandwidth, access latencies, and cache hierarchy behavior by utilizing low-level optimizations.

## 2. System Architecture

The software is implemented modularly in C++, with performance-critical sections written in ARM64 assembly. The architecture follows a strict separation of concerns and utilizes the RAII (Resource Acquisition Is Initialization) design pattern for resource management.

### 2.1 Memory Management ([`src/core/memory/`](src/core/memory/))

The system employs advanced memory management techniques to ensure the reproducibility of measurement results:

- **Smart Memory Allocation**: Utilizes `mmap` calls wrapped in `MmapPtr` for automatic resource deallocation. See [`memory_manager.h`](src/core/memory/memory_manager.h) and [`memory_manager.cpp`](src/core/memory/memory_manager.cpp).
- **MmapDeleter**: A specialized custom deleter that prevents memory leaks even in the event of exceptions.
- **Cache Control**: Implements `madvise()` hints to simulate non-cacheable memory (acknowledging user-space limitations).
- **Buffer Manager**: Provides centralized management for all test buffers (main memory, L1/L2 caches). The interface is defined in [`buffer_manager.h`](src/core/memory/buffer_manager.h), with implementation split (v0.52.4) into [`buffer_allocator.cpp`](src/core/memory/buffer_allocator.cpp) for memory allocation with validation and [`buffer_initializer.cpp`](src/core/memory/buffer_initializer.cpp) for buffer initialization.

### 2.2 Configuration Management ([`src/core/config/`](src/core/config/))

The configuration system manages all benchmark parameters through a two-pass argument parsing strategy and comprehensive validation:

- **Configuration Structure**: The central [`BenchmarkConfig`](src/core/config/config.h) structure contains all benchmark settings including buffer sizes, iteration counts, thread configuration, and test mode flags.
- **Two-Pass Argument Parsing**: [`argument_parser.cpp`](src/core/config/argument_parser.cpp) implements a two-stage parsing process. The first pass extracts `-cache-size` early to enable proper cache detection before the second pass parses all remaining arguments.
- **Configuration Validation**: [`config_validator.cpp`](src/core/config/config_validator.cpp) validates flag compatibility (e.g., mutual exclusion of `-only-bandwidth` and `-only-latency`), enforces memory limits (80% of available RAM), and validates buffer size constraints.
- **Buffer Calculations**: [`buffer_calculator.cpp`](src/core/config/buffer_calculator.cpp) computes cache buffer sizes (L1 at 75% of cache, L2 at 10%) with stride alignment and page boundary enforcement, and scales latency access counts based on buffer size.
- **System Constants**: [`constants.h`](src/core/config/constants.h) defines all default values including 512MB default buffer size, 1000 iterations, and platform-specific constraints like 16KB minimum cache size and 1GB maximum.
- **Version Management**: [`version.h`](src/core/config/version.h) maintains the software version using semantic versioning.

#### Command-Line Parameters

The tool supports the following command-line flags:

**Core Configuration:**
- `-buffersize <size_mb>`: Buffer size in MB (default: 512, automatically capped to 80% of available memory / 3)
- `-iterations <count>`: Number of bandwidth test iterations (default: 1000, incompatible with `-only-latency`)
- `-count <count>`: Number of full benchmark loops (default: 1, enables percentile statistics when > 1)
- `-latency-samples <count>`: Latency samples per test (default: 1000, incompatible with `-only-bandwidth`)
- `-threads <count>`: Thread count for parallel tests (default: auto-detected CPU cores, auto-capped to maximum available)
- `-cache-size <size_kb>`: Custom cache size in KB for testing (range: 16KB to 1GB, skips L1/L2 detection)

**Test Mode Selection:**
- `-patterns`: Run pattern benchmarks only (forward, reverse, strided, random access)
- `-only-bandwidth`: Run only bandwidth tests (main memory and cache), skip latency
- `-only-latency`: Run only latency tests (main memory and cache), skip bandwidth
- `-non-cacheable`: Apply cache-discouraging memory hints via `madvise()` (best-effort)

**Output Control:**
- `-output <file>`: Save results to JSON file
- `-h, --help`: Display help message and exit

The configuration flow integrates with system detection ([`system_info.cpp`](src/core/system/system_info.cpp)) to automatically determine CPU model, core counts, cache sizes, and available memory, providing intelligent defaults while allowing user override of all parameters.

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

### 3.4 Warmup Strategies ([`src/warmup/`](src/warmup/))

Accurate memory benchmarking requires eliminating cold-start effects through systematic warmup. The warmup module implements four distinct strategies optimized for different test scenarios:

**Purpose and Critical Importance:**
- **Page Fault Elimination**: Touches all memory pages to map virtual-to-physical addresses before timing, preventing kernel traps (100-1000× overhead) during measurement
- **TLB Population**: Pre-loads Translation Lookaside Buffer entries to eliminate address translation overhead (10-50 cycle penalty per miss)
- **Cache Priming**: Loads data into cache hierarchy to measure steady-state performance rather than cold-start behavior (10-100× difference)
- **Prefetcher Training**: Establishes access patterns to ensure hardware prefetchers operate at full efficiency during timed runs

**Warmup Strategies:**

1. **Basic Warmup** ([`basic_warmup.cpp`](src/warmup/basic_warmup.cpp)): Multi-threaded sequential read/write/copy warmup for main memory bandwidth tests. Uses adaptive sizing (10% of buffer or 64MB maximum) to prevent excessive overhead on large buffers while ensuring meaningful coverage on small buffers.

2. **Cache Warmup** ([`cache_warmup.cpp`](src/warmup/cache_warmup.cpp)): Multi-threaded full-buffer warmup for cache-specific tests (L1/L2/custom). Warms entire buffer since cache-sized buffers are small (128KB-16MB). Fixed in v0.52.5 to properly honor `num_threads` parameter and added null/size validation guards.

3. **Latency Warmup** ([`latency_warmup.cpp`](src/warmup/latency_warmup.cpp)): Single-threaded page-granular prefaulting (1-byte read + write per page) that eliminates page faults without warming caches, preserving "cold-ish" cache state for accurate latency measurement. Uses volatile pointers to prevent compiler optimization.

4. **Pattern Warmup** ([`pattern_warmup.cpp`](src/warmup/pattern_warmup.cpp)): Specialized warmup for strided and random access patterns. Validates stride parameters (≥32 bytes) and random indices (bounds + alignment). Distributes indices across threads rather than buffer chunks to avoid race conditions.

**Implementation Architecture:**

The warmup system is built on template-based infrastructure ([`warmup_internal.h`](src/warmup/warmup_internal.h)) with two core functions:
- `warmup_parallel<ChunkOp>()`: Multi-threaded orchestrator handling cache-line alignment (64-byte boundaries), chunk distribution, prefix/gap processing to ensure complete buffer coverage, and thread synchronization with QoS settings
- `warmup_single<Op>()`: Single-threaded wrapper for latency tests requiring memory locality

**Integration with Benchmarks:**

Every benchmark type has a corresponding warmup function executed immediately before timing begins:
- Main memory bandwidth: `warmup_read()`, `warmup_write()`, `warmup_copy()`
- Cache bandwidth: `warmup_cache_read()`, `warmup_cache_write()`, `warmup_cache_copy()`
- Latency tests: `warmup_latency()`, `warmup_cache_latency()`
- Pattern tests: `warmup_read_strided()`, `warmup_write_random()`, etc.

The warmup functions call the same ARM64 assembly loops as the benchmarks, ensuring identical access patterns. Atomic XOR accumulation (`std::atomic<uint64_t>`) prevents compiler dead code elimination while maintaining thread safety.

## 4. Benchmark Logic and Execution

The measurement process is divided into specialized executors. See [`src/benchmark/`](src/benchmark/) for implementation details:

- **Multi-threading**: Bandwidth tests utilize all available system cores by default. See [`bandwidth_tests.cpp`](src/benchmark/bandwidth_tests.cpp) and [`parallel_test_framework.h`](src/benchmark/parallel_test_framework.h).
- **Single-threading**: Latency tests are executed on a single thread to ensure maximum accuracy. See [`latency_tests.cpp`](src/benchmark/latency_tests.cpp).
- **Statistical Analysis**: Results are aggregated across multiple runs, calculating means and percentiles (P50, P90, P95, P99). See [`benchmark_results.cpp`](src/benchmark/benchmark_results.cpp) and [`benchmark_runner.cpp`](src/benchmark/benchmark_runner.cpp).

### 4.1 Test Types

- **Main Memory Bandwidth**: Read, write, and copy operations at the DRAM level. See [`benchmark_executor.cpp`](src/benchmark/benchmark_executor.cpp).
- **Cache Bandwidth**: Specific throughput tests for L1 and L2 caches.

### 4.2 Pattern Benchmark Architecture ([`src/pattern_benchmark/`](src/pattern_benchmark/))

The pattern benchmark subsystem analyzes how memory access patterns affect performance, complementing the main bandwidth tests' focus on maximum throughput. While main tests answer "How fast can the memory go?", pattern tests answer "How does access pattern degrade performance?"

**Architectural Components:**

- **Pattern Coordinator** ([`pattern_coordinator.cpp`](src/pattern_benchmark/pattern_coordinator.cpp)): Orchestrates execution of five access patterns in sequence: sequential forward (baseline), sequential reverse, strided 64-byte (cache-line), strided 4096-byte (page), and random uniform. Added in v0.52.4 refactoring.

- **Statistics Manager** ([`pattern_statistics_manager.cpp`](src/pattern_benchmark/pattern_statistics_manager.cpp)): Manages multi-loop execution and collects bandwidth results across iterations for statistical analysis. Refactored from monolithic structure in v0.52.4.

- **Execution Modules**: Specialized executors for each pattern type in [`execution_patterns.cpp`](src/pattern_benchmark/execution_patterns.cpp) (forward/reverse) and [`execution_strided.cpp`](src/pattern_benchmark/execution_strided.cpp) (strided patterns). Random index generation handled in [`execution_utils.cpp`](src/pattern_benchmark/execution_utils.cpp).

**Access Pattern Types:**

1. **Sequential Forward**: Standard linear forward access establishing baseline performance
2. **Sequential Reverse**: Backward linear access testing backward prefetcher efficiency (percentage vs. forward baseline)
3. **Strided 64B**: Cache-line aligned strides (64-byte) measuring hardware prefetcher effectiveness with predictable patterns
4. **Strided 4096B**: Page-level strides (4096-byte) revealing TLB pressure and cache thrashing potential
5. **Random Uniform**: Completely unpredictable access pattern (1,000-1,000,000 accesses, 32-byte aligned) representing worst-case performance

**Efficiency Metrics:**

Pattern benchmarks calculate performance ratios to quantify degradation:
- **Sequential Coherence**: `(reverse_bandwidth / forward_bandwidth) × 100%` - Measures backward prefetcher capability
- **Prefetcher Effectiveness**: `(strided_64_bandwidth / forward_bandwidth) × 100%` - Cache-line stride should be well-prefetched
- **Cache Thrashing Potential**: `(strided_4096_bandwidth / forward_bandwidth) × 100%` with thresholds (>70% = low, 40-70% = medium, <40% = high)
- **TLB Pressure**: `(random_bandwidth / strided_4096_bandwidth) × 100%` with thresholds (>50% = minimal, 20-50% = moderate, <20% = high)

**Assembly Implementation:**

Dedicated ARM64 assembly functions for each pattern:
- Reverse: [`memory_read_reverse.s`](src/asm/memory_read_reverse.s), [`memory_write_reverse.s`](src/asm/memory_write_reverse.s), [`memory_copy_reverse.s`](src/asm/memory_copy_reverse.s)
- Strided: [`memory_read_strided.s`](src/asm/memory_read_strided.s), [`memory_write_strided.s`](src/asm/memory_write_strided.s), [`memory_copy_strided.s`](src/asm/memory_copy_strided.s)
- Random: [`memory_read_random.s`](src/asm/memory_read_random.s), [`memory_write_random.s`](src/asm/memory_write_random.s), [`memory_copy_random.s`](src/asm/memory_copy_random.s)

All assembly functions use NEON SIMD (32-byte loads/stores), modulo arithmetic for bounds safety, and XOR checksum accumulation.

**Integration:**

Pattern benchmarks run via `-patterns` flag, allocating only 2× buffer size (src + dst, no latency buffer). Results include absolute bandwidth plus percentage differences from baseline, providing quantitative analysis of pattern sensitivity across read, write, and copy operations.

## 5. System Integration

- **Timing**: Uses the macOS `mach_absolute_time()` interface for nanosecond-precision timing. See [`timer.h`](src/core/timing/timer.h) and [`timer.cpp`](src/core/timing/timer.cpp).
- **Hardware Detection**: Automatically detects L1/L2 cache sizes and CPU models via `sysctlbyname` calls. See [`system_info.h`](src/core/system/system_info.h) and [`system_info.cpp`](src/core/system/system_info.cpp).
- **QoS Configuration**: Sets the system Quality of Service (QoS) state to levels appropriate for latency-critical testing.

## 6. Output and Reporting

The software implements a sophisticated dual-output system designed for both human readability and machine processing:

### 6.1 Console Output System ([`src/output/console/`](src/output/console/))

**Message Architecture:**

The console output uses a modular message system ([`src/output/console/messages/`](src/output/console/messages/)) with 10 specialized message modules:
- [`cache_messages.cpp`](src/output/console/messages/cache_messages.cpp): Cache test result formatting
- [`config_messages.cpp`](src/output/console/messages/config_messages.cpp): Configuration display
- [`error_messages.cpp`](src/output/console/messages/error_messages.cpp): Error reporting with centralized prefixes
- [`info_messages.cpp`](src/output/console/messages/info_messages.cpp): Informational messages
- [`pattern_messages.cpp`](src/output/console/messages/pattern_messages.cpp): Pattern benchmark formatting with efficiency metrics
- [`program_messages.cpp`](src/output/console/messages/program_messages.cpp): Program lifecycle messages (startup, completion)
- [`results_messages.cpp`](src/output/console/messages/results_messages.cpp): Main memory benchmark results
- [`statistics_messages.cpp`](src/output/console/messages/statistics_messages.cpp): Multi-loop statistical summaries
- [`warning_messages.cpp`](src/output/console/messages/warning_messages.cpp): Warning messages with standardized format

**Output Components:**

- **Output Printer** ([`output_printer.h`](src/output/console/output_printer.h) and [`output_printer.cpp`](src/output/console/output_printer.cpp)): Main output coordinator managing message dispatch and formatting consistency
- **Statistics Formatter** ([`statistics.h`](src/output/console/statistics.h) and [`statistics.cpp`](src/output/console/statistics.cpp)): Calculates and formats percentile statistics (P50/P90/P95/P99), standard deviation (sample stddev with Bessel's correction), min/max values
- **Progress Indicators**: Real-time dot indicators during benchmark execution and spinner animation for long-running operations

**Formatting Standards:**

All numeric output uses precision constants from [`constants.h`](src/core/config/constants.h):
- Bandwidth: 5 decimal places (e.g., "45.12345 GB/s")
- Latency: 2 decimal places (e.g., "125.47 ns")
- Pattern percentages: 1 decimal place (e.g., "+15.3%")

### 6.2 JSON Output System ([`src/output/json/`](src/output/json/))

**JSON Architecture:**

The JSON export subsystem ([`src/output/json/json_output/`](src/output/json/json_output/)) consists of 7 specialized modules:

- **JSON Builder** ([`builder.cpp`](src/output/json/json_output/builder.cpp)): Constructs the root JSON structure with standardized field ordering (v0.52.4)
- **File Writer** ([`file_writer.cpp`](src/output/json/json_output/file_writer.cpp)): Handles file I/O with error handling
- **Main Memory Formatter** ([`main_memory.cpp`](src/output/json/json_output/main_memory.cpp)): Formats bandwidth and latency results for main memory tests
- **Cache Formatter** ([`cache.cpp`](src/output/json/json_output/cache.cpp)): Formats L1/L2/custom cache test results
- **Pattern Formatter** ([`patterns.cpp`](src/output/json/json_output/patterns.cpp)): Formats pattern benchmark results with efficiency metrics
- **JSON Output Coordinator** ([`json_output.h`](src/output/json/json_output/json_output.h) and [`json_output.cpp`](src/output/json/json_output/json_output.cpp)): Orchestrates the complete JSON export process

**JSON Structure:**

The output includes comprehensive metadata and results:
- **System Information**: CPU model, core counts (performance/efficiency), cache sizes, available memory, macOS version
- **Configuration**: All benchmark parameters (buffer sizes, iterations, thread count, test modes)
- **Timestamp**: ISO 8601 formatted execution time
- **Per-Loop Raw Values**: Individual measurements for each benchmark loop iteration
- **Aggregated Statistics**: Mean, median, percentiles (P90/P95/P99), standard deviation, min/max across all loops (when loop_count > 1)
- **Pattern Efficiency Metrics**: Sequential coherence, prefetcher effectiveness, cache thrashing potential, TLB pressure

**Third-Party Dependency:**

JSON parsing and generation uses the nlohmann/json library ([`src/third_party/nlohmann/json.hpp`](src/third_party/nlohmann/json.hpp)), a header-only C++ JSON library providing type-safe serialization.

**Output Modes:**

- **Console Only** (default): Real-time human-readable output with progress indicators
- **Console + JSON** (`-output <file>`): Both console display and machine-readable JSON export for automated analysis and result archiving

## 7. Technical Constraints and Considerations

- **Platform**: Designed exclusively for the macOS Apple Silicon platform.
- **User-Space Limitations**: As a user-space application, the tool does not have direct access to explicit cache flushing (cache flush) or true non-cacheable memory. The `-non-cacheable` flag uses `madvise()` hints which do not make memory truly non-cacheable; they only reduce caching likelihood and affect paging and cache behavior heuristically.
- **Buffer Sizes**: DRAM measurements require large buffer sizes (recommended > 512 MB) to minimize the impact of the cache hierarchy.

## 8. Quality Assurance

The software includes a comprehensive testing suite:

- **Testing Framework**: 183 tests across 8 test files (3,200 lines) using the GoogleTest framework. See [`tests/`](tests/) directory.
- **Test Organization**: Tests are separated into unit tests and integration tests (v0.52.1). Three test targets available: `make test` (unit tests only, faster development cycle), `make test-integration` (integration tests only), and `make test-all` (complete test suite).
- **Coverage Areas**: Memory allocation and initialization, buffer management, configuration validation and parsing, benchmark execution, pattern validation, message formatting, memory utilities.
- **Doxygen-compliant documentation** at the source code level.
- **Validation logic** for all user inputs and detected system values.

## 9. Known Sources of Bias and Limitations

The following factors may influence measurement accuracy and should be considered when interpreting results:

- **Hardware Prefetcher Effects**: Pattern benchmarks intentionally measure prefetcher behavior, but this means sequential and strided patterns may show optimistically high bandwidth compared to real-world irregular access patterns. The prefetcher effectiveness metric quantifies this bias.

- **Core Heterogeneity**: Apple Silicon features performance (P) and efficiency (E) cores with different cache sizes and clock speeds. The tool does not set thread affinity, so the macOS scheduler determines core allocation. Multi-threaded tests may distribute work across heterogeneous cores, potentially showing lower aggregate bandwidth than homogeneous core setups would achieve. Note that even if thread affinity were set via `thread_policy_set()`, macOS does not guarantee strict core binding and may migrate threads based on thermal or power management policies.

- **Operating System Interference**: macOS background services, system daemons, and the scheduler itself introduce non-deterministic overhead. The QoS settings (`QOS_CLASS_USER_INTERACTIVE`) request priority but cannot eliminate all OS interference. Running multiple benchmark loops (`-count`) and examining percentile statistics (P95, P99) helps identify outliers caused by OS activity.

- **Thermal Throttling and DVFS**: Extended benchmark runs may trigger thermal throttling, reducing CPU clock speeds and memory controller performance. Dynamic Voltage and Frequency Scaling (DVFS) adjusts power states based on load and temperature. Results from later loops may show degraded performance if thermal limits are reached. Consider monitoring system temperature and allowing cooldown periods between intensive test runs.

- **Unified Memory Architecture**: Apple Silicon uses unified memory shared between CPU, GPU, and Neural Engine (ANE). Concurrent GPU or ANE workloads can compete for memory bandwidth, affecting CPU benchmark results. For isolated CPU measurements, ensure minimal GPU/ANE activity (close graphics-intensive applications, disable background ML tasks). The unified architecture also means memory bandwidth measurements reflect the entire SoC memory subsystem, not just CPU-specific paths.

- **Lack of SMT/Hyper-Threading**: Apple Silicon does not implement Simultaneous Multi-Threading (SMT). Each thread runs on a dedicated physical core, eliminating SMT-related resource contention but also preventing measurements of SMT bandwidth scaling behavior present on x86 architectures.

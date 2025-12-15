# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.47] - 2025-12-15

### Added
- **Percentile statistics (P50, P90, P95, P99) and standard deviation**: Enhanced statistics reporting to include median (P50), P90, P95, P99 percentiles, and standard deviation for all metrics:
  - Main memory bandwidth (read, write, copy)
  - Cache bandwidth (L1, L2, Custom - read, write, copy)
  - Cache latency (L1, L2, Custom)
  - Main memory latency
  - Percentiles calculated using linear interpolation method for accurate results
- **Full sample distribution collection for latency tests**: Added ability to collect individual latency samples per test iteration for meaningful percentile calculations (used for cache latency tests; main memory latency keeps single-measurement behavior to avoid cache state bias):
  - New `-latency-samples <count>` CLI parameter (default: 1000) to control number of samples collected per latency test
  - Modified `run_latency_test()` and `run_cache_latency_test()` to support per-sample latencies when sample collection is enabled
  - Samples are collected across all benchmark loops and concatenated for final percentile calculation
  - Statistics now show percentiles from full sample distributions (e.g., "from 10000 samples") when available, providing much more meaningful percentile statistics than loop averages alone
  - Falls back to loop average percentiles if sample collection is disabled (backward compatible)
- **Enhanced statistics display**: Statistics output now clearly indicates when percentiles are calculated from full sample distributions vs. loop averages, showing sample count in output

### Changed
- **Extended `Statistics` struct**: Added `median`, `p90`, `p95`, `p99`, and `stddev` fields to support comprehensive statistical analysis
- **Updated `calculate_statistics()` function**: Now calculates percentiles using linear interpolation and standard deviation in addition to average, min, and max
- **Statistics calculation improvements**: All print functions (`print_metric_statistics()`, `print_cache_bandwidth_statistics()`, `print_cache_latency_statistics()`) now display the new percentile and standard deviation statistics
- **Latency test function signatures**: Updated `run_latency_test()` and `run_cache_latency_test()` to accept optional sample collection parameters while maintaining backward compatibility
- **Benchmark results structures**: Added latency sample vectors to `BenchmarkResults` and `BenchmarkStatistics` to store full sample distributions
- **Configuration system**: Added `latency_sample_count` to `BenchmarkConfig` with default value from `Constants::DEFAULT_LATENCY_SAMPLE_COUNT` (1000)

## [0.46] - 2025-12-14

### Added
- Testing section to 'README.md'

### Changed
- **Code organization: Split `src/utils.cpp` into focused modules**: Refactored the monolithic 443-line `utils.cpp` file into three focused files following single responsibility principle:
  - `src/utils.cpp` (43 lines): Contains only thread utilities (`join_threads()`) and progress indicator (`show_progress()`)
  - `src/output_printer.cpp` (269 lines, new): Contains all output/printing functions (`print_usage()`, `print_configuration()`, `print_results()`, `print_cache_info()`)
  - `src/statistics.cpp` (167 lines, new): Contains all statistics calculation and printing functions (`print_statistics()` and helper functions)
  - Updated `src/benchmark.h` with organized function declarations grouped by source file
  - Updated `Makefile` to include new source files in build
  - All tests continue to pass without modification (31 tests)
  - Benefits: Improved code navigation, better maintainability, reduced cognitive load, clearer separation of concerns
- **Code organization: Split `src/benchmark_runner.cpp` into focused modules**: Refactored the 344-line `benchmark_runner.cpp` file into three focused files following single responsibility principle:
  - `src/benchmark_runner.cpp` (161 lines): Contains main orchestration functions (`run_single_benchmark_loop()`, `run_all_benchmarks()`)
  - `src/benchmark_executor.cpp` (129 lines, new): Contains all test execution helper functions (`run_main_memory_bandwidth_tests()`, `run_cache_bandwidth_tests()`, `run_cache_latency_tests()`, `run_main_memory_latency_test()`, and helper functions)
  - `src/benchmark_results.cpp` (71 lines, new): Contains results calculation functions (`calculate_single_bandwidth()`, `calculate_bandwidth_results()`)
  - `src/benchmark_executor.h` (new): Header with `TimingResults` struct and test execution function declarations
  - `src/benchmark_results.h` (new): Header with results calculation function declarations
  - Updated `Makefile` to include new source files in build
  - Benefits: Improved separation of concerns (test execution, results calculation, and orchestration), better code navigation, easier maintenance
- **Consolidated default configuration values**: Moved all default configuration values from `config.h` to `constants.h` to eliminate duplication and ensure single source of truth:
  - `DEFAULT_BUFFER_SIZE_MB` (512): Already existed in constants, now used as default in `config.h`
  - `DEFAULT_ITERATIONS` (1000): Moved from hardcoded value in `config.h` to `constants.h`
  - `DEFAULT_LOOP_COUNT` (1): Moved from hardcoded value in `config.h` to `constants.h`
- **Help text now references constants**: Updated help output in `output_printer.cpp` to dynamically reference constants instead of hardcoded default values, ensuring help text stays synchronized with actual defaults.
- **Enhanced license information display**: Added full GPL v3 license text including warranty disclaimer to both help output (`-h`/`--help`) and normal program execution output.
- **Test updates**: Updated unit tests to use constants instead of hardcoded values for better maintainability.
- Output examples for 0.46 version in 'README.md'

## [0.45] - 2025-12-13

### Added
- **Unit test suite with Google Test framework**: Added unit tests to improve code quality.
  - `tests/test_config.cpp`: 12 tests covering configuration parsing, validation, buffer size calculations, and access count calculations.
  - `tests/test_buffer_manager.cpp`: 4 tests for buffer allocation, initialization, and helper methods.
  - `tests/test_memory_manager.cpp`: 9 tests for memory allocation, read/write operations, and automatic cleanup.
  - `tests/test_benchmark_runner.cpp`: 6 tests for statistics initialization, result structures, benchmark execution, and result validation.
  - Total of 31 tests across 4 test suites
- **Makefile test support**: Added `test` target to build and run all unit tests, and `clean-test` target for test cleanup.
  - Test files compile with debug symbols (`-g`) for faster compilation during development.
  - Google Test framework integrated via Homebrew installation.

### Changed
- **Register allocation cleanup in `memory_latency.s`**: Removed unused register allocations (x2, x3) that were saving parameters but never used, improving code clarity and reducing unnecessary register operations.
- **Warmup function refactoring**: Created generic `warmup_parallel()` and `warmup_single()` template functions to eliminate duplicate code across warmup implementations. All parallel warmup functions (`warmup_read`, `warmup_write`, `warmup_copy`) now share the same thread management logic via `warmup_parallel()`, and all single-threaded warmup functions (`warmup_cache_read`, `warmup_cache_write`, `warmup_cache_copy`) use `warmup_single()`. Eliminated duplicate implementation between `warmup_latency()` and `warmup_cache_latency()` by having `warmup_cache_latency()` call `warmup_latency()`. This reduces code duplication and improves maintainability while preserving the existing public API.
- **Benchmark test function refactoring**: Created generic `run_parallel_test()` template function that handles the common threading pattern (thread creation, synchronization, chunk calculation, QoS setting, and timer coordination). Refactored `run_read_test()`, `run_write_test()`, and `run_copy_test()` to use this framework, reducing code duplication from ~60 lines per function to ~10-20 lines. Each test now defines only its specific work function (lambda), while the generic framework handles all thread management.
- **QoS class consistency**: Ensured `QOS_CLASS_USER_INTERACTIVE` (highest priority) is set consistently across all threads and operations. Updated `warmup_read()` to set QoS (previously disabled), and added QoS setting to `warmup_single()` template for all single-threaded warmup operations. All benchmark worker threads, warmup threads, and single-threaded operations now use the highest QoS class by default for optimal performance.
- **Benchmark runner refactoring**: Eliminated code duplication in cache testing functions by creating helper functions:
  - `calculate_single_bandwidth()`: Unified bandwidth calculation logic for main memory, L1, L2, and custom cache levels, reducing `calculate_bandwidth_results()` from ~70 lines to ~30 lines.
  - `run_single_cache_latency_test()`: Unified cache latency test execution for L1, L2, and custom cache levels, reducing `run_cache_latency_tests()` from ~28 lines to ~18 lines.
  - `run_single_cache_bandwidth_test()`: Unified cache bandwidth test execution (read, write, copy) for L1, L2, and custom cache levels, reducing `run_cache_bandwidth_tests()` from ~48 lines to ~28 lines.
  - Overall code reduction: ~146 lines of duplicated code reduced to ~60-70 lines with shared helper functions, improving maintainability and reducing bug risk.
- **Enhanced test validation**: Added `ResultsValidation` test that verifies benchmark results are mathematically valid (non-negative bandwidth, reasonable latency ranges, no NaN/Inf values), ensuring refactored calculation functions produce correct results.
- **Statistics printing refactoring**: Refactored monolithic `print_statistics()` function in `src/utils.cpp` to eliminate massive code duplication across cache levels (L1, L2, Custom). Created reusable helper functions:
  - `Statistics` struct and `calculate_statistics()`: Unified statistics calculation (average, min, max) from vectors.
  - `print_metric_statistics()`: Generic function for printing single metric statistics.
  - `print_cache_bandwidth_statistics()`: Unified bandwidth statistics printing for any cache level.
  - `print_cache_latency_statistics()`: Unified latency statistics printing for any cache level.
  - Reduced `print_statistics()` from ~200 lines to ~50 lines (75% reduction), eliminating duplicate code patterns for L1, L2, and Custom cache levels while maintaining identical output format.
- **Thread utility function extraction**: Extracted duplicated `join_threads` lambda from `benchmark_tests.cpp` and `warmup.cpp` into a common utility function in `utils.cpp`. The function is now declared in `benchmark.h` and shared across both files, eliminating code duplication and improving maintainability.

## [0.44] - 2025-12-13

### Changed
- README.md updated.
- **Refactoring of main.cpp**: Refactored  `main.cpp`:
  - `src/memory_manager.h/cpp`: Encapsulates mmap-based memory allocation with RAII (`MmapPtr`, `MmapDeleter`).
  - `src/config.h/cpp`: Centralizes configuration management (`BenchmarkConfig` struct, CLI argument parsing, validation, and buffer size calculations).
  - `src/buffer_manager.h/cpp`: Manages all benchmark buffer lifecycle (`BenchmarkBuffers` struct, allocation, and initialization).
  - `src/benchmark_runner.h/cpp`: Handles benchmark execution and result collection (`BenchmarkResults`, `BenchmarkStatistics` structs).
  - `main.cpp` now serves as a clean orchestration function, improving maintainability, testability, and extensibility.
-Extracted all magic numbers to named constants in `src/constants.h` (buffer size factors, memory limits, cache size limits, latency stride, access counts, etc.), improving maintainability and making configuration values explicit. Refactored `run_single_benchmark_loop()` into smaller, focused functions: `run_main_memory_bandwidth_tests()`, `run_cache_bandwidth_tests()`, `run_cache_latency_tests()`, `run_main_memory_latency_test()`, and `calculate_bandwidth_results()`.
- Optimized checksum accumulation in read benchmark: moved `checksum.fetch_xor()` out of the inner loop to reduce atomic operations from one per iteration to one per thread, significantly reducing contention and improving performance.
- Latency warmup changed from pointer chasing to page prefaulting: `warmup_latency()` and `warmup_cache_latency()` now perform linear page prefaulting (touching each page with 1-byte read/write at page-size intervals) instead of chasing pointers. This ensures pages are mapped and page faults are removed without building the pointer chain, providing more accurate "cold-ish" latency measurements, especially with small buffers or buffers near cache boundaries (L1/L2).

## [0.43] - 2025-12-12

### Added
- Cache-size parameter to set up custom cache testing size in KB.

### Changed
- split loop.s to separate files; memory_read.s, memory_copy.s, memory_write.s and memory_latency.s.
- moved assembly sources to src/asm -directory.
- main memory latency is printed after main memory bandwidth results and before cache results.

## [0.42] - 2025-12-12

### Changed
- L2 cache test buffer now 10% of reported L2 size (for example targets ~1–2 MB on detected 16MB) to better match observed cache behavior. 

## [0.41] - 2025-12-11

### Added
- L1 and L2 cache latency testing using pointer chasing methodology.
- Cache latency results displayed with buffer size information.
- Cache latency statistics (average, min, max) across multiple loops.
- L1 and L2 cache bandwidth testing (read, write, copy) using cache-sized buffers.
- Cache bandwidth results displayed with buffer size information.
- Cache bandwidth statistics (average, min, max) across multiple loops.

### Changed
- Updated result printing functions to include cache latency results.
- Simplified output: replaced verbose progress messages with a lightweight spinner indicator.
- Removed detailed allocation, initialization, warmup, and test progress messages to reduce output clutter.
- Added progress indicator that shows test execution status without affecting benchmark performance.

## [0.4] - 2025-12-10

### Fixed
- **ABI compatibility fix**: Fixed register corruption issue where all assembly functions (`memory_copy_loop_asm`, `memory_read_loop_asm`, `memory_write_loop_asm`) were using callee-saved registers q8-q15 (v8-v15) without preserving them, corrupting the calling C++ function's variables. All functions now use only caller-saved registers (q0-q7 and q16-q31) per AAPCS64, ensuring ABI compliance without requiring stack operations (maintaining leaf function optimization).
  - `memory_copy_loop_asm`: Replaced q8-q15 with q16-q23 and q0-q7 (reusing q0-q7 for remaining pairs).
  - `memory_read_loop_asm`: Moved accumulators from v8-v11 to v0-v3, and replaced all q8-q15 usage with q4-q7 and q16-q31 (data loaded first, then accumulated to avoid overwriting accumulators).
  - `memory_write_loop_asm`: Replaced q8-q15 with q16-q23 and q0-q7 (reusing q0-q7 since all values are zeros).

### Changed
- Help output (`-h`/`--help`) now includes author name, email, license, and link to GitHub page.


## [0.39] - 2025-12-07

### Fixed
- **Critical bug fix in `memory_read_loop_asm` main loop**: Fixed double accumulation bug where registers q0-q3 were loaded twice (at offsets 0-96 and 448-480) and accumulated twice, causing incorrect XOR checksum calculation. Reordered code to accumulate q0-q7 immediately after first load, then safely reuse q0-q3 for final loads, ensuring each data block is accumulated exactly once.

### Changed
- Bandwidth timing now starts after worker threads are created and waiting, and stops after join; added a start gate so setup/QoS overhead is excluded from measurements.
- Latency warm-up now runs immediately before each latency test loop to keep cache state consistent with the measurement that follows.
- Warmups for read/write/copy now run immediately before their respective benchmarks each loop to align cache/TLB state with the measurement.


## [0.38] - 2025-12-07

### Fixed
- **Critical bug fix in `memory_read_loop_asm` cleanup section**: Fixed register conflict in 256B chunk handling where accumulator registers `v8-v11` were being overwritten by loads into `q8-q11`, causing 128 bytes of data to be excluded from the XOR checksum calculation. Changed to use temporary registers `q16-q19` instead, ensuring all data is properly accumulated.

### Changed
- Standardized comments in `loops.s`: unified style, consistent alignment, added missing documentation.


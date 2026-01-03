# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.52.5] - NON-RELEASED

Summary: Fixes cache warmup to honor `num_threads` and adds null/size guards.

### Fixed
- **Cache warmup ignoring num_threads parameter**: Fixed issue where cache warmup functions (`warmup_cache_read`, `warmup_cache_write`, `warmup_cache_copy`) accepted a `num_threads` parameter but always ran single-threaded, ignoring the parameter.
- **Missing null buffer guards in cache warmup functions**: Added null buffer and size checks to cache warmup functions (`warmup_cache_read`, `warmup_cache_write`, `warmup_cache_copy`) to prevent null pointer dereferencing. The functions now validate buffers before passing them to `warmup_parallel()`, matching the defensive programming pattern used in `warmup_latency()`.

## [0.52.4] - 2026-01-02

### Added
- **Technical Specification document**: Added  [TECHNICAL_SPECIFICATION.md](TECHNICAL_SPECIFICATION.md) documenting system architecture, memory management, low-level ARM64 assembly implementation, benchmark execution flow, system integration, and technical specifications with links to relevant source files and directories.

### Changed
- **Refactored configuration module**: Split `src/core/config/config.cpp`:
  - `src/core/config/argument_parser.cpp` - Command-line argument parsing logic
  - `src/core/config/config_validator.cpp` - Configuration validation logic
  - `src/core/config/buffer_calculator.cpp` - Buffer size and access count calculations
- **Refactored buffer management module**: Split `src/core/memory/buffer_manager.cpp`:
  - `src/core/memory/buffer_allocator.cpp` - Memory allocation logic with validation and overflow checks
  - `src/core/memory/buffer_initializer.cpp` - Buffer initialization and latency chain setup
  - `buffer_manager.h` remains as the main interface, including both new headers for backward compatibility
- **Refactored benchmark runner module**: Split `src/benchmark/benchmark_runner.cpp`:
  - `src/benchmark/benchmark_executor.cpp` - Single benchmark loop execution (`run_single_benchmark_loop`)
  - `src/benchmark/benchmark_statistics_collector.cpp` - Statistics collection and aggregation (`initialize_statistics`, `collect_loop_results`)
  - `benchmark_runner.cpp` simplified to orchestration only (~73 lines), using the new modules for execution and statistics collection
- **Refactored pattern benchmark execution module**: Split `src/pattern_benchmark/execution.cpp`:
  - `src/pattern_benchmark/pattern_coordinator.cpp` - High-level pattern coordination (`run_pattern_benchmarks`) orchestrating different pattern types (forward, reverse, strided, random)
  - `src/pattern_benchmark/pattern_statistics_manager.cpp` - Loop management and statistics collection (`run_all_pattern_benchmarks`) handling result aggregation across multiple loops
- **Standardized JSON output structure**: All JSON output files now follow a consistent field ordering across all benchmark types (latency/bandwidth and pattern benchmarks).
- **Increased maximum cache size limit**: Raised `MAX_CACHE_SIZE_KB` from 512 MB to 1 GB to support larger memory working sets and to help identify bandwidth and latency changes when moving beyond on-chip cache regions into main memory.
- **README.md documentation**: Replaced detailed technical implementation section with a concise reference to the new Technical Specification document, reducing duplication and centralizing technical documentation.

### Fixed
- **Mach host port leak**: Fixed memory leak in `get_available_memory_mb()` where Mach host port acquired via `mach_host_self()` was never deallocated.
- **Unnecessary cache buffer allocation for pattern-only runs**: Fixed issue where pattern benchmarks (`-patterns` flag) were allocating and initializing cache buffers (L1/L2/custom latency and bandwidth) even though pattern benchmarks only use src/dst buffers.
- **Missing total memory guard**: Fixed issue where total memory requirement calculation did not validate against the 80% availability limit calculated in `validate_config()`.
- **JSON save logic inversion**: Fixed inverted logic in standard benchmark JSON save path (`main.cpp`) that caused the program to exit with failure (`EXIT_FAILURE`) when JSON output was successfully saved.
- **Division by zero in warmup_parallel**: Fixed bug in `warmup_parallel()` where `chunk_base_size` and `chunk_remainder` were computed before validating `num_threads`, causing division by zero and undefined behavior when `num_threads` is 0 or negative. 
- **Volatile write optimization in latency warmup**: Fixed issue in `warmup_latency()` where `buf[offset] = buf[offset];` on a non-volatile pointer could be optimized away by the compiler as a no-op, preventing write permissions from being faulted in.
- **Negative pointer difference in warmup offset calculation**: Fixed bug in `warmup_parallel()` where rounding `chunk_end` down to cache line boundary could result in `aligned_end_ptr` being before `buffer` when `chunk_end` was within the first cache line. The negative pointer difference would wrap to a huge `size_t` value, causing the next iteration to access memory far beyond the buffer.
- **Incomplete buffer coverage in warmup**: Fixed issue where unaligned prefix bytes between `buffer` and the first cache-line boundary were never touched during warmup, leaving those regions cold. 
- **Alignment gaps creating cold regions in warmup**: Fixed bug where bytes between `unaligned_start` and `chunk_start` (up to 63 bytes per thread) were never touched, creating unwarmed gaps throughout the buffer, especially with high thread counts or small chunks.
- **Tiny buffers completely skipped in warmup**: Fixed issue where buffers smaller than the cache line alignment gap (e.g., < 64 bytes) were completely skipped during warmup when `prefix_size >= effective_size`.
- **Small chunks completely skipped in warmup**: Fixed bug where chunks smaller than the alignment offset (`original_chunk_size <= alignment_offset`) were completely skipped without any processing, creating unwarmed "holes" in the buffer. 
- **Copy bandwidth underreported in pattern benchmarks**: Fixed issue where forward, reverse, and random copy operations in pattern benchmarks (`execution_patterns.cpp`) were underreported by 2x because they didn't account for copy operations moving data twice (read from source + write to destination).

## [0.52.2] - 2025-12-31

### Added
- **Selective test execution parameters**: Added `-only-bandwidth` and `-only-latency` parameters to allow running only specific test types:
  - `-only-bandwidth`: Runs only bandwidth tests (read/write/copy for main memory and cache), skipping all latency tests. 
  - `-only-latency`: Runs only latency tests (main memory and cache latency), skipping all bandwidth tests. 

### Fixed
- **Pattern benchmark memory allocation**: Fixed issue where pattern benchmarks (`-patterns` flag) were allocating 3x buffer size (src + dst + lat) when they only need 2x (src + dst). Pattern benchmarks are bandwidth-only tests and do not use the latency buffer, so the latency buffer allocation and initialization are now skipped when pattern benchmarks are enabled. The total allocation size display now correctly shows 2x instead of 3x for pattern benchmarks.

## [0.52.1] - 2025-12-30

### Added
- **Test coverage for memory allocation failures**: Added tests to verify error handling when memory allocation fails, including tests for zero-size buffers and allocation cleanup on failure.
- **Test coverage for buffer size edge cases**: Added tests for buffer size boundary conditions, including buffers smaller than stride, equal to stride, and minimum valid sizes for latency chain setup.

### Fixed
- **Strided access pattern benchmark accuracy**: Fixed bug in strided access pattern benchmarks where loops terminated incorrectly, causing unreliable and incorrect results.
- **Optimized reverse read byte loop**: Improved efficiency of byte cleanup loop in `memory_read_reverse.s`.
- **Standard deviation calculation consistency**: Fixed bug where console output used population standard deviation (÷n) while JSON output used sample standard deviation (÷n-1), causing 5-22% difference in reported stddev values. Console and JSON outputs now show identical stddev values using the statistically correct sample standard deviation formula.
- **Centralized latency precision**: Replaced ~36 hardcoded precision values across output formatting code with a single `LATENCY_PRECISION` constant for easier maintenance and consistency.
- **Better error messages**: Error messages for invalid parameters now show the actual invalid value and valid ranges (e.g., "iterations invalid (must be between 1 and 2147483647, got -5)").
- **Timer reliability**: Timer initialization no longer causes the program to exit unexpectedly. Errors are now reported gracefully with clear messages.
- **Consistent warning format**: All warning messages now use a centralized prefix system for consistent formatting throughout the application.
- **Test reliability**: Removed arbitrary test limits (10000 GB/s bandwidth and 1ms latency) that could cause false test failures on different hardware configurations. Tests now validate that results are positive and valid without overly restrictive upper bounds.

### Changed
- **Version numbering system**: Transitioned from simple incrementing version numbers (0.52, 0.53...) to semantic versioning format (0.52.1, 0.52.2...) to prevent the version number from growing too rapidly with small fixes and patches. The version is now stored as a string literal instead of a float, supporting proper semantic versioning with patch-level increments.
- **Test organization**: Separated integration tests from unit tests using Google Test filters. Tests that perform actual system operations (benchmark execution) are now marked as integration tests and can be run separately. The Makefile now provides three test targets: `make test` (unit tests only, faster), `make test-integration` (integration tests only), and `make test-all` (all tests). This improves test independence and allows faster unit test runs during development.

## [0.52] - 2025-12-29

### Fixed
- **Pattern benchmarks now respect loop count**: Fixed pattern benchmarks to properly respect the `-count` parameter. Pattern benchmarks now run the specified number of loops and collect aggregated statistics (average, min, max, median/P50, P90, P95, P99, stddev) across all loops.
- **Improved measurement accuracy**: Fixed issues with small buffer sizes and edge cases that could produce misleading results.
- **Better error handling**: Enhanced error reporting throughout the application.
- **Timer reliability**: Fixed timer issues that could cause crashes or incorrect measurements on long-running benchmarks or edge cases.
- **Bandwidth calculation fixes**: Fixed copy bandwidth calculation that was reporting inflated values.
- **Multi-threading improvements**: Fixed thread synchronization issues that could cause incomplete benchmark coverage or incorrect results when using multiple threads.
- **Memory safety**: Added comprehensive validation to prevent buffer overruns and memory allocation issues, especially when working with large memory regions or near buffer boundaries.

## [0.51] - 2025-12-27

### Added
- **Thread count control (`-threads` parameter)**: Added `-threads <count>` command-line parameter to allow users to specify the number of threads used for benchmarks. The parameter applies to all non-latency benchmarks. If the specified value exceeds the available CPU cores, it is automatically capped to the maximum number of cores. If not specified, the default behavior uses the detected CPU core count. This provides fine-grained control over benchmark parallelism for testing different thread configurations.
- **Example scripts**: Added script-examples folder containing example bash and Python scripts to automate benchmark execution and result analysis.

## [0.50] - 2025-12-21

### Added
- **Non-cacheable memory hints (`-non-cacheable` flag)**: Added support for applying hints to discourage aggressive memory caching during benchmarks. This provides hints to the memory system but does not provide true non-cacheable memory on macOS. The feature applies hints to source buffers (for read/copy operations) and destination buffers (for write/copy operations) when enabled. If hints cannot be applied, a warning is shown but the benchmark continues.
- **Improved warmup accuracy**: Enhanced warmup functions to better match the actual access patterns being tested, providing more accurate benchmark results. Warmup now adapts to the test type (strided, random, or sequential) and uses intelligent size limits to prevent excessive warmup time on very large buffers.
- **Better error messages**: Added validation and clearer error messages for invalid stride values and random access patterns.

### Changed
- **Faster warmup on large buffers**: Optimized warmup to use less time on very large buffers while still ensuring accurate results.
- **Improved error reporting**: All error messages now use consistent formatting and provide clearer information about what went wrong.

## [0.49] - 2025-12-20

### Added
- **Pattern benchmarks (`-patterns` parameter)**: Added new memory access pattern benchmarking to measure performance across different access patterns:
  - **Sequential Forward**: Baseline sequential access pattern (forward direction)
  - **Sequential Reverse**: Sequential access in reverse order (backwards)
  - **Strided (Cache Line - 64B)**: Strided access with 64-byte stride (cache line size)
  - **Strided (Page - 4096B)**: Strided access with 4096-byte stride (page size)
  - **Random Uniform**: Completely random access pattern (worst case for prefetch)
  - Each pattern measures read, write, and copy bandwidth
  - Results include percentage differences from sequential forward baseline
  - Pattern Efficiency Analysis provides insights into:
    - Sequential coherence (reverse vs forward performance)
    - Prefetcher effectiveness (strided 64B vs sequential)
    - Cache thrashing potential (strided 4096B performance)
    - TLB pressure (random vs strided access)
  - When `-patterns` is used, only pattern benchmarks are executed (standard bandwidth/latency tests are skipped)

### Fixed
- **Cache latency display bug**: Fixed cache latency tests displaying "inf ns" instead of valid latency values. Tests now show "N/A ns" when values are invalid, providing clearer error indication.

## [0.48] - 2025-12-20

### Added
- **JSON output support**: Added `-output <file>` parameter to save benchmark results to JSON format. Results include configuration, all benchmark metrics (bandwidth and latency for main memory and cache levels), per-loop values, aggregated statistics (average, min, max, median, P90, P95, P99, stddev), and execution timestamp. The output file path can be relative or absolute, and parent directories are automatically created if needed.

### Fixed
- **Timing display bug**: Fixed main memory bandwidth test times displaying as "0.000 s" for very fast tests. Times now display with sufficient precision to show accurate values.
- **Buffer safety**: Fixed potential buffer overflow issue in memory management code.

## [0.47] - 2025-12-15

### Added
- **Enhanced statistics**: Added percentile statistics (P50/median, P90, P95, P99) and standard deviation for all metrics:
  - Main memory bandwidth (read, write, copy)
  - Cache bandwidth (L1, L2, Custom - read, write, copy)
  - Cache latency (L1, L2, Custom)
  - Main memory latency
- **Configurable latency sampling**: Added `-latency-samples <count>` parameter (default: 1000) to control the number of samples collected per latency test for more accurate percentile calculations.
- **Improved statistics display**: Statistics output now clearly shows when percentiles are calculated from full sample distributions vs. loop averages, including sample count information.

## [0.46] - 2025-12-14

### Added
- Testing section added to README.md

### Changed
- **Help text accuracy**: Help output now dynamically shows the correct default values, ensuring help text stays synchronized with actual defaults.
- **License information**: Added full GPL v3 license text including warranty disclaimer to both help output (`-h`/`--help`) and normal program execution output.

## [0.45] - 2025-12-13

### Added
- **Unit test suite**: Added comprehensive unit tests to improve code quality and reliability.

### Changed
- **Performance improvements**: Optimized benchmark execution for better performance and accuracy.
- **Improved warmup accuracy**: Enhanced warmup procedures to better prepare memory state before measurements.

## [0.44] - 2025-12-13

### Changed
- **Performance improvements**: Optimized read benchmark to reduce contention and improve performance.
- **More accurate latency measurements**: Improved latency warmup to provide more accurate "cold-ish" latency measurements, especially with small buffers or buffers near cache boundaries.

## [0.43] - 2025-12-12

### Added
- **Custom cache size parameter**: Added ability to set custom cache testing size in KB.

### Changed
- **Improved output organization**: Main memory latency results are now printed after main memory bandwidth results and before cache results for better readability.

## [0.42] - 2025-12-12

### Changed
- **Better L2 cache testing**: L2 cache test buffer size adjusted to 10% of reported L2 size (targets ~1–2 MB on detected 16MB) to better match observed cache behavior.

## [0.41] - 2025-12-11

### Added
- **Cache latency testing**: Added L1 and L2 cache latency testing using pointer chasing methodology.
- **Cache bandwidth testing**: Added L1 and L2 cache bandwidth testing (read, write, copy) using cache-sized buffers.
- **Cache statistics**: Cache latency and bandwidth results now include statistics (average, min, max) across multiple loops.
- **Simplified output**: Replaced verbose progress messages with a lightweight spinner indicator that shows test execution status without cluttering the output.

## [0.4] - 2025-12-10

### Fixed
- **Critical bug fix**: Fixed register corruption issue that could cause incorrect benchmark results. All memory operations now correctly preserve register state.

### Changed
- **Enhanced help output**: Help output (`-h`/`--help`) now includes author name, email, license, and link to GitHub page.

## [0.39] - 2025-12-07

### Fixed
- **Critical bug fix**: Fixed double accumulation bug in read benchmark that caused incorrect checksum calculation. Read benchmarks now produce accurate results.

### Changed
- **More accurate timing**: Bandwidth timing now excludes thread setup overhead for more accurate measurements.
- **Better cache state consistency**: Latency and bandwidth warmups now run immediately before each test to ensure consistent cache state.

## [0.38] - 2025-12-07

### Fixed
- **Critical bug fix**: Fixed register conflict in read benchmark that caused some data to be excluded from checksum calculation. All data is now properly processed.

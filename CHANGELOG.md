# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

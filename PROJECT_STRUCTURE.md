# Project Structure — membenchmark

**Version:** 0.53.8
**Platform:** ARM64 / AArch64 (Apple Silicon macOS)
**License:** GNU General Public License v3.0

This document describes the layout of project files, organized by purpose. It is intended as a navigation aid for contributors and reviewers.

---

## Table of Contents

1. [Root-level files](#1-root-level-files)
2. [src/ — Source code](#2-src--source-code)
   - [src/asm/](#21-srcasm--arm64-assembly-kernels)
   - [src/benchmark/](#22-srcbenchmark--benchmarking-infrastructure)
   - [src/core/](#23-srccore--core-utilities)
   - [src/output/](#24-srcoutput--output-layer)
   - [src/pattern_benchmark/](#25-srcpattern_benchmark--pattern-access-benchmarks)
   - [src/warmup/](#26-srcwarmup--pre-benchmark-warm-up)
   - [src/utils/](#27-srcutils--shared-utilities)
   - [src/third_party/](#28-srcthird_party--vendored-dependencies)
3. [tests/ — Unit tests](#3-tests--unit-tests)
4. [results/ — Benchmark result data](#4-results--benchmark-result-data)
5. [pictures/ — Documentation images](#5-pictures--documentation-images)
6. [.github/ — GitHub integration](#6-github--github-integration)

---

## 1. Root-level files

### Entry point

| File | Purpose |
|---|---|
| `main.cpp` | Program entry point; parses configuration, dispatches to the appropriate benchmark or analysis mode |

### Build and tooling

| File | Purpose |
|---|---|
| `Makefile` | Primary build system; produces the `membenchmark` release binary and the `test_runner` test binary |
| `.clang-format` | Clang-Format style configuration enforced across all C++ sources |

### User-facing documentation

| File | Purpose |
|---|---|
| `README.md` | Project overview, quick-start instructions, and feature summary |
| `MANUAL.md` | Complete user manual: all CLI flags, modes, output formats, and usage examples |
| `TECHNICAL_SPECIFICATION.md` | Internal architecture, data structures, and implementation decisions |
| `CHANGELOG.md` | Version history and release notes |
| `CONTRIBUTING.md` | Contribution guidelines, coding standards, and pull request process |
| `CODE_OF_CONDUCT.md` | Community standards |
| `SECURITY.md` | Vulnerability disclosure policy |
| `TLB_ANALYSIS_WHITEPAPER.md` | Whitepaper: TLB analysis methodology and Apple Silicon results |
| `LATENCY_WHITEPAPER.md` | Whitepaper: cache and memory latency measurement methodology |
| `CORE_TO_CORE_WHITEPAPER.md` | Whitepaper: core-to-core cache-line handoff latency methodology and results |
| `PROJECT_STRUCTURE.md` | This file |

---

## 2. src/ — Source code

All production C++ and ARM64 assembly lives under `src/`. Headers use include paths relative to `src/` (e.g., `#include "core/config/config.h"`).

---

### 2.1 src/asm/ — ARM64 assembly kernels

Hand-written AArch64 assembly implementing the hot inner loops that must not be rewritten by the compiler. Each `.s` file corresponds to one access pattern or operation type. The public C-linkage declarations live in `asm_functions.h`.

| File | Operation |
|---|---|
| `asm_functions.h` | `extern "C"` declarations for all assembly functions |
| `memory_copy.s` | Sequential forward memory copy |
| `memory_copy_random.s` | Random-order memory copy |
| `memory_copy_reverse.s` | Reverse-order memory copy |
| `memory_copy_strided.s` | Strided memory copy |
| `memory_read.s` | Sequential forward memory read |
| `memory_read_random.s` | Random-order memory read |
| `memory_read_reverse.s` | Reverse-order memory read |
| `memory_read_strided.s` | Strided memory read |
| `memory_write.s` | Sequential forward memory write |
| `memory_write_random.s` | Random-order memory write |
| `memory_write_reverse.s` | Reverse-order memory write |
| `memory_write_strided.s` | Strided memory write |
| `memory_latency.s` | Pointer-chase latency measurement loop |
| `core_to_core_latency.s` | Cache-line handoff ping-pong loop for core-to-core latency |

---

### 2.2 src/benchmark/ — Benchmarking infrastructure

The benchmark subsystem owns all measurement logic: executing tests, collecting per-iteration samples, computing statistics, and coordinating TLB and core-to-core analysis modes.

#### Bandwidth and latency tests

| File | Purpose |
|---|---|
| `bandwidth_tests.cpp` | Bandwidth measurement: read, write, and copy for main memory and cache levels |
| `latency_tests.cpp` | Latency measurement: pointer-chase patterns across buffer sizes |
| `benchmark_tests.h` | Shared declarations for bandwidth and latency test functions |

#### Execution pipeline

| File | Purpose |
|---|---|
| `benchmark_executor.h` / `.cpp` | Executes individual benchmark passes (bandwidth and latency) at each cache/memory level; drives the parallel test framework |
| `benchmark_runner.h` / `.cpp` | Outer loop: runs multiple benchmark iterations, collects result vectors, and invokes the statistics collector |
| `benchmark_results.h` / `.cpp` | Data structures holding per-run and aggregated benchmark results |
| `benchmark_statistics_collector.h` / `.cpp` | Accumulates per-iteration samples and computes summary statistics (min, max, mean, median, percentiles, stddev) |
| `parallel_test_framework.h` | Template-based framework for dispatching multi-threaded benchmark work with synchronized start, cache-line-aligned per-thread state, and macOS QoS thread attributes |

#### TLB analysis mode

| File | Purpose |
|---|---|
| `tlb_analysis.h` / `.cpp` | Standalone `-analyze-tlb` mode: sweeps buffer sizes and strides to locate TLB capacity boundaries |
| `tlb_analysis_json.h` / `.cpp` | Serializes TLB analysis results to JSON |
| `tlb_boundary_detector.cpp` | Heuristic that identifies TLB miss inflection points in the latency-vs-buffer-size curve |

#### Core-to-core latency mode

| File | Purpose |
|---|---|
| `core_to_core_latency.h` | Public interface for the `-analyze-core2core` mode |
| `core_to_core_latency_internal.h` | Internal runner interfaces not exposed outside the module |
| `core_to_core_latency_runner.cpp` | Measurement loop: coordinates two threads on selected CPU cores, runs the assembly ping-pong hot loop, and collects round-trip latency samples |
| `core_to_core_latency_cli.cpp` | CLI argument parsing and entry point for the core-to-core mode |
| `core_to_core_latency_json.h` / `.cpp` | Serializes core-to-core results to JSON |

---

### 2.3 src/core/ — Core utilities

Platform-independent infrastructure: configuration, memory management, system introspection, and high-resolution timing.

#### src/core/config/

| File | Purpose |
|---|---|
| `config.h` | `BenchmarkConfig` structure; aggregates all run-time settings parsed from the command line |
| `constants.h` | Named constants for memory limits, cache size bounds, stride values, buffer sizing factors, and latency access counts |
| `version.h` | `SOFTVERSION` macro (semantic version string, currently `"0.53.8"`) |
| `argument_parser.cpp` | Parses `argv` into a `BenchmarkConfig`; implements all flag definitions |
| `config_validator.cpp` | Validates the parsed configuration; emits errors for out-of-range or conflicting settings |
| `buffer_calculator.cpp` | Derives buffer sizes for each cache/memory level from the validated configuration and detected system parameters |

#### src/core/memory/

| File | Purpose |
|---|---|
| `memory_manager.h` / `.cpp` | Top-level RAII memory manager; allocates and owns benchmark buffers via `mmap`; supports normal and cache-discouraging (`MAP_NOCACHE`) allocation |
| `buffer_manager.h` | Manages the set of named buffers handed to benchmark passes |
| `buffer_allocator.h` / `.cpp` | Low-level `mmap`/`munmap` wrapper with alignment support |
| `buffer_initializer.h` / `.cpp` | Initializes buffer contents (sequential fill, random fill, pointer-chase chain construction) before benchmark runs |
| `memory_utils.h` / `.cpp` | Miscellaneous memory helpers: cache-line flushing, pointer-chase chain verification |

#### src/core/system/

| File | Purpose |
|---|---|
| `system_info.h` / `.cpp` | Queries the OS for physical CPU core count, performance/efficiency core topology, cache sizes (L1/L2/L3), page size, and total physical memory via `sysctlbyname` |

#### src/core/timing/

| File | Purpose |
|---|---|
| `timer.h` / `.cpp` | High-resolution timer wrapping `clock_gettime(CLOCK_MONOTONIC_RAW)` on macOS; provides nanosecond-resolution elapsed-time measurement |

---

### 2.4 src/output/ — Output layer

All user-visible output is isolated in this subsystem. The split between console messages and JSON output allows each to evolve independently.

#### src/output/console/

| File | Purpose |
|---|---|
| `output_printer.h` / `.cpp` | Formats and prints benchmark results and statistics to stdout in human-readable form |
| `statistics.h` / `.cpp` | Computes and prints statistical summaries (average, median, P90, P95, P99, stddev, min, max) for a result vector |

##### src/output/console/messages/

All user-facing text strings are centralized here. Each `.cpp` file implements a specific category of messages declared in the shared API header.

| File | Purpose |
|---|---|
| `messages_api.h` | Declares the `Messages` namespace containing all string-returning functions |
| `cache_messages.cpp` | Cache-level result labels and headings |
| `config_messages.cpp` | Configuration echo and validation error text |
| `core_to_core_messages.cpp` | Core-to-core mode status and result messages |
| `error_messages.cpp` | Fatal error messages |
| `info_messages.cpp` | General informational messages |
| `pattern_messages.cpp` | Pattern benchmark descriptive labels |
| `program_messages.cpp` | Program banner, usage, and version text |
| `results_messages.cpp` | Benchmark result formatting strings |
| `statistics_messages.cpp` | Statistical output labels (e.g., `"Average:"`, `"Median (P50):"`, `"Stddev:"`) |
| `warning_messages.cpp` | Non-fatal warning messages |

#### src/output/json/json_output/

| File | Purpose |
|---|---|
| `json_output_api.h` | Public API for writing benchmark results to JSON files |
| `json_output.cpp` | Top-level orchestrator: selects the appropriate serializer and triggers file output |
| `builder.cpp` | Constructs the root JSON object and populates metadata (timestamp, version, platform) |
| `cache.cpp` | Serializes cache-level benchmark results |
| `main_memory.cpp` | Serializes main-memory benchmark results |
| `patterns.cpp` | Serializes pattern benchmark results |
| `file_writer.cpp` | Writes the completed JSON document to disk with ISO 8601 timestamp in the filename |

---

### 2.5 src/pattern_benchmark/ — Pattern access benchmarks

Benchmarks characterizing memory access patterns: sequential forward, sequential reverse, strided, and random. Results expose how access regularity and stride distance affect bandwidth and latency.

| File | Purpose |
|---|---|
| `pattern_benchmark.h` | Public interface for the pattern benchmark mode |
| `pattern_coordinator.cpp` | Orchestrates pattern selection, buffer preparation, and result collection across all configured patterns |
| `execution_patterns.cpp` | Implements sequential forward and reverse pattern execution |
| `execution_strided.cpp` | Implements strided pattern execution |
| `execution_utils.cpp` | Shared execution helpers (timing wrappers, iteration control) |
| `helpers.cpp` | Miscellaneous internal helpers (pattern name formatting, range validation) |
| `validation.cpp` | Validates pattern benchmark configuration parameters |
| `pattern_statistics_manager.cpp` | Manages per-pattern statistical accumulators |
| `output.cpp` | Formats pattern benchmark results for console output |

---

### 2.6 src/warmup/ — Pre-benchmark warm-up

Warm-up passes eliminate cold-start effects from page faults, TLB misses, and instruction-cache misses before timing begins.

| File | Purpose |
|---|---|
| `warmup.h` | Public warm-up API |
| `warmup_internal.h` | Internal warm-up helpers not exposed outside the module |
| `basic_warmup.cpp` | Simple sequential read/write pass to page in all benchmark buffers |
| `cache_warmup.cpp` | Targeted warm-up designed to fill a specific cache level before a cache-level benchmark |
| `latency_warmup.cpp` | Warm-up for latency tests: traverses the pointer-chase chain to bring it into the target cache level |
| `pattern_warmup.cpp` | Warm-up pass tailored to the configured access pattern |

---

### 2.7 src/utils/ — Shared utilities

| File | Purpose |
|---|---|
| `benchmark.h` | Convenience umbrella header that includes all benchmark-related headers |
| `utils.h` / `.cpp` | General-purpose helpers: size formatting, human-readable unit conversion, and string utilities |
| `json_utils.h` / `.cpp` | JSON helper functions shared between the TLB, core-to-core, and standard output serializers |

---

### 2.8 src/third_party/ — Vendored dependencies

| File | Purpose |
|---|---|
| `nlohmann/json.hpp` | nlohmann/json single-header library (MIT license); used throughout the JSON output subsystem |

---

## 3. tests/ — Unit tests

GoogleTest-based unit test suite. All files are picked up automatically by the Makefile. Tests named `*Integration*` are excluded from `make test` (unit-only) and must be run explicitly.

| File | Suite name | Coverage focus |
|---|---|---|
| `test_config.cpp` | `ConfigTest` | CLI argument parsing, config validation, buffer size calculation |
| `test_messages.cpp` | `Messages*Test` | Console message string functions across all categories |
| `test_memory_utils.cpp` | `MemoryUtilsTest` | Memory helpers: pointer-chase chain construction and verification |
| `test_memory_manager.cpp` | `MemoryManagerTest` | `mmap`-based allocation, RAII cleanup, alignment |
| `test_buffer_manager.cpp` | `BufferManagerTest` | Named buffer set management |
| `test_benchmark_executor.cpp` | `BenchmarkExecutorTest` | Per-level executor dispatch and result structure |
| `test_benchmark_runner.cpp` | `BenchmarkRunnerTest` | Multi-iteration outer loop and statistics hand-off |
| `test_analysis.cpp` | `AnalysisTest` | TLB boundary detection logic |
| `test_json_schema.cpp` | `JsonSchemaTest` | JSON output structure and field presence |
| `test_pattern_validation.cpp` | `PatternValidationTest` | Pattern benchmark parameter validation |
| `test_pattern_benchmark.cpp` | `PatternBenchmarkTest` | Pattern execution and statistics |
| `test_core_to_core_messages.cpp` | `CoreToCoreMessagesTest` | Core-to-core console message strings |
| `test_core_to_core_cli.cpp` | `CoreToCoreCliTest` | Core-to-core CLI argument parsing |
| `test_core_to_core_runner.cpp` | `CoreToCoreRunnerTest` | Core-to-core runner logic and JSON serialization |
| `test_statistics.cpp` | `StatisticsTest` | Statistical computations: median, percentiles, stddev, min, max |
| `test_timer.cpp` | `HighResTimerTest` | Timer resolution, monotonicity, and elapsed-time accuracy |
| `test_system_info.cpp` | `SystemInfoTest` | `sysctlbyname`-based system queries on macOS |

**Total unit tests (non-integration):** approximately 335 across 16 suites as of 2026-03-15.

---

## 4. results/ — Benchmark result data

Reference JSON (and legacy CSV/text) output from benchmark runs on specific hardware. Organized by software version subdirectory. These files are used for regression comparison and whitepaper data.

```
results/
  0.53.7/
    MacMiniM4_analyzetlb.json       — TLB analysis run, Mac Mini M4
    MacMiniM4_benchmark.json        — Standard benchmark run, Mac Mini M4
    MacMiniM4_core2core.json        — Core-to-core latency run, Mac Mini M4
    MacMiniM4_patterns.json         — Pattern benchmark run, Mac Mini M4
  0.53.8/
    MacMiniM4_analyze-tlb-*.json    — TLB analysis variants (chain mode, random-in-box)
    MacbookAirM5_analyze-tlb.json   — TLB analysis run, MacBook Air M5
    MacbookAirM5_benchmark.json     — Standard benchmark run, MacBook Air M5
    MacbookAirM5_latency.json       — Latency run, MacBook Air M5
  old/
    *.json / *.csv / *.txt          — Pre-versioned historical results
```

---

## 5. pictures/ — Documentation images

PNG charts generated from benchmark result data, used in the whitepapers and README.

| File | Content |
|---|---|
| `MacMiniM4_memory_hierarchy_v0_53_5.png` | Full memory hierarchy bandwidth/latency overview, Mac Mini M4 |
| `MacMiniM4_cache_latency_with_TLB.png` | Cache latency curve with TLB miss inflection, Mac Mini M4 |
| `MacMiniM4_cache_latency_with_TLB_5loops_p50_values.png` | P50 latency across 5 measurement loops, Mac Mini M4 |
| `MacMiniM4_cache_latency_with_TLB_5loops_p50_values_2.png` | Alternate P50 latency chart, Mac Mini M4 |
| `MacMiniM4_cache_latency_with_STRIDE_TLB.png` | Latency vs. stride size showing TLB effects, Mac Mini M4 |
| `MacMiniM4_TLB_analysis_with_64_STRIDE.png` | TLB analysis at 64-byte stride, Mac Mini M4 |
| `MacMiniM4_TLB_analysis_with_128_STRIDE.png` | TLB analysis at 128-byte stride, Mac Mini M4 |
| `MacMiniM4_cache_latency_TLB_16KB.png` | Cache latency detail at 16 KB (L1 TLB boundary), Mac Mini M4 |
| `MacMiniM4_cache_latency_2_TLB_16KB.png` | Second chart at the 16 KB boundary, Mac Mini M4 |
| `MacBookAirM5_latency_memory_hierarchy.png` | Memory hierarchy latency, MacBook Air M5 |
| `MacBookAirM5_latency_vs_cache-stride-tlb.png` | Latency vs. cache/stride/TLB interactions, MacBook Air M5 |

---

## 6. .github/ — GitHub integration

| File | Purpose |
|---|---|
| `.github/pull_request_template.md` | Default pull request description template |
| `.github/ISSUE_TEMPLATE/bug_report.md` | Structured bug report template |

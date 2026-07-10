# Project Structure — macOS-memory-benchmark

**Version:** 0.59.0
**Platform:** ARM64 / AArch64 (Apple Silicon macOS)
**License:** GNU General Public License v3.0

This document describes the layout of project files, organized by purpose. It is intended as a navigation aid for contributors and reviewers.

---

## Table of Contents

1. [Root-level files](#1-root-level-files)
2. [src/ — Source code](#2-src--source-code)
   - [src/asm/](#21-srcasm--arm64-assembly-kernels)
   - [src/benchmark/](#22-srcbenchmark---benchmarking-infrastructure)
   - [src/core/](#23-srccore--core-utilities)
   - [src/output/](#24-srcoutput---output-layer)
   - [src/pattern_benchmark/](#25-srcpattern_benchmark--pattern-access--benchmarks)
   - [src/warmup/](#26-srcwarmup--pre--benchmark-warm-up)
   - [src/utils/](#27-srcutils--shared-utilities)
   - [src/third_party/](#28-srcthird_party--vendored-dependencies)
3. [tests/ — Unit tests](#3-tests--unit-tests)
4. [results/ — Benchmark result data](#4-results---benchmark-result-data)
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
| `Makefile` | Primary build system; produces the `memory_benchmark` release binary and the `test_runner` test binary |
| `coverage.sh` | Runs isolated LLVM unit/all-test source coverage builds under `/tmp` without replacing normal workspace binaries |
| `.clang-format` | Clang-Format style baseline for C++ sources |

### User-facing documentation

| File | Purpose |
|---|---|
| `README.md` | Project overview, quick-start instructions, and feature summary |
| `CAPABILITIES.md` | Measurement capability overview and interpretation notes |
| `MANUAL.md` | Complete user manual: all CLI flags, modes, output formats, and usage examples |
| `PARAMETER_MATRIX.md` | Mode/flag compatibility, sweep support, and incompatible-option matrix |
| `TECHNICAL_SPECIFICATION.md` | Internal architecture, data structures, and implementation decisions |
| `CHANGELOG.md` | Version history and release notes |
| `CONTRIBUTING.md` | Contribution guidelines, coding standards, and pull request process |
| `CODE_OF_CONDUCT.md` | Community standards |
| `SECURITY.md` | Vulnerability disclosure policy |
| `TLB_ANALYSIS_WHITEPAPER.md` | Whitepaper: current TLB analysis methodology, schema, and interpretation limits |
| `LATENCY_WHITEPAPER.md` | Whitepaper: cache and memory latency measurement methodology |
| `CORE_TO_CORE_WHITEPAPER.md` | Whitepaper: calibrated core-to-core methodology, audit schema, and interpretation limits |
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
| `memory_copy.s` | Sequential forward memory copy (main-memory, non-temporal stores) |
| `memory_copy_cache.s` | Sequential forward memory copy (cache-focused) |
| `memory_copy_random.s` | Random-order memory copy |
| `memory_copy_reverse.s` | Reverse-order memory copy |
| `memory_copy_strided.s` | Phase-rotating strided memory copy (generic stride parameter) |
| `memory_read.s` | Sequential forward memory read (main-memory) |
| `memory_read_cache.s` | Sequential forward memory read (cache-focused) |
| `memory_read_random.s` | Random-order memory read |
| `memory_read_reverse.s` | Reverse-order memory read |
| `memory_read_strided.s` | Phase-rotating strided memory read (generic stride parameter) |
| `memory_write.s` | Sequential forward memory write (main-memory, non-temporal stores) |
| `memory_write_cache.s` | Sequential forward memory write (cache-focused) |
| `memory_write_random.s` | Random-order memory write |
| `memory_write_reverse.s` | Reverse-order memory write |
| `memory_write_strided.s` | Phase-rotating strided memory write (generic stride parameter) |
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
| `benchmark_measurement.h` | Shared status-bearing measurement type and value/unavailable helpers |

#### Execution pipeline

| File | Purpose |
|---|---|
| `benchmark_executor.h` / `.cpp` | Executes individual benchmark passes (bandwidth and latency) at each cache/memory level; drives the parallel test framework |
| `benchmark_runner.h` / `.cpp` | Outer loop: runs multiple benchmark iterations, collects result vectors, and invokes the statistics collector |
| `benchmark_statistics_collector.h` / `.cpp` | Initializes/preallocates statistics storage and accumulates measured per-loop values and latency samples for later aggregation |
| `benchmark_work_plan.h` / `.cpp` | Pure standard-benchmark work planning, calibration arithmetic, seed derivation, and cyclic scheduling helpers |
| `parallel_test_framework.h` | Template-based framework for dispatching multi-threaded benchmark work with synchronized start, cache-line-aligned per-thread state, and macOS QoS thread attributes |
| `sweep_runner.h` / `.cpp` | Common Cartesian sweeps for standard, pattern, and standalone TLB modes, including validation and checkpoints |

#### TLB analysis mode

| File | Purpose |
|---|---|
| `tlb_analysis.h` / `.cpp` | Standalone `--analyze-tlb` mode: sweeps locality windows, supports stride/chain-mode/density controls, and infers TLB capacity boundaries |
| `tlb_analysis_json.h` / `.cpp` | Serializes TLB analysis results to JSON |
| `tlb_boundary_detector.cpp` | Robust paired-delta boundary detection, bootstrap confidence intervals, persistence gates, and independent validation |
| `tlb_chain.h` / `.cpp` | Builds and verifies page-native spread/packed pointer-chain controls |
| `tlb_measurement_scheduler.h` / `.cpp` | Seeded cyclic-Latin round scheduling, task-seed derivation, and pass execution accounting |
| `tlb_runtime_policy.h` / `.cpp` | Runtime profiles, access calibration, convergence evaluation, memory budgeting, and work estimates |
| `tlb_sweep_planner.h` / `.cpp` | Page-consistent base/refinement locality planning and source tracking |

#### Core-to-core latency mode

| File | Purpose |
|---|---|
| `core_to_core_latency.h` | Public interface for the `--analyze-core2core` mode |
| `core_to_core_latency_internal.h` | Internal runner interfaces not exposed outside the module |
| `core_to_core_latency_runner.cpp` | Per-scenario calibration, balanced loop scheduling, unpinned two-thread ping-pong execution, and robust summaries |
| `core_to_core_latency_cli.cpp` | CLI argument parsing and entry point for the core-to-core mode |
| `core_to_core_latency_json.h` / `.cpp` | Serializes schema-2 work plans, loop audit records, completion state, and results |
| `core_to_core_sweep_runner.h` / `.cpp` | Core-to-core Cartesian sweeps and atomic per-run checkpoints |

---

### 2.3 src/core/ — Core utilities

Core infrastructure for configuration, memory management, macOS system introspection, signal handling, and Mach timing.

#### src/core/config/

| File | Purpose |
|---|---|
| `config.h` | `BenchmarkConfig` structure for standard, pattern, standalone TLB, and their common sweep settings; core-to-core uses a separate config type |
| `constants.h` | Named constants for memory limits, cache size bounds, stride values, buffer sizing factors, and latency access counts |
| `version.h` | `SOFTVERSION` macro (semantic version string, currently `"0.59.0"`) |
| `argument_parser.cpp` | Parses standard, pattern, and standalone TLB options into `BenchmarkConfig`; core-to-core is pre-routed to its dedicated parser |
| `config_validator.cpp` | Validates the parsed configuration; emits errors for out-of-range or conflicting settings |
| `buffer_calculator.cpp` | Derives buffer sizes for each cache/memory level from the validated configuration and detected system parameters |

#### src/core/signal/

| File | Purpose |
|---|---|
| `signal_handler.h` / `.cpp` | Installs SIGINT/SIGTERM handling and coordinates benchmark interruption between main and worker threads |

#### src/core/memory/

| File | Purpose |
|---|---|
| `memory_manager.h` / `.cpp` | Low-level normal and cache-discouraging `mmap` allocation; returns RAII `MmapPtr` owners and exposes a test-only mmap-family provider seam |
| `buffer_manager.h` | Defines the phase-local `BenchmarkBuffers` and command-local `PatternBuffers` RAII owners |
| `buffer_allocator.h` / `.cpp` | Atomically allocates the pattern source/destination pair and performs overflow-safe peak-concurrent memory accounting |
| `buffer_initializer.h` / `.cpp` | Initializes deterministic pattern source data and the zeroed pattern destination before execution |
| `memory_utils.h` / `.cpp` | Cache-line alignment helpers, deterministic/unseeded latency-chain construction and diagnostics, and basic buffer initialization |

#### src/core/system/

| File | Purpose |
|---|---|
| `system_info.h` / `.cpp` | Queries CPU/cache/OS/memory data through a macOS provider and a deterministic provider seam for fallback/error tests |

#### src/core/timing/

| File | Purpose |
|---|---|
| `timer.h` / `.cpp` | High-resolution Mach timer with exact tick conversion plus a deterministic clock/timebase provider seam |

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
| `json_output.cpp` | Builds standard/pattern root payloads, adds timestamp/version metadata, and triggers file output |
| `builder.cpp` | Builds common mode configuration metadata, including resolved chain, seed, calibration, scheduling, and worker policies |
| `standard.cpp` | Active standard schema-2 serializer for completion state, loop measurements, and main/cache aggregates |
| `patterns.cpp` | Serializes pattern benchmark results |
| `file_writer.cpp` | Atomically writes the completed JSON document to the requested output path, creating parent directories as needed |

---

### 2.5 src/pattern_benchmark/ — Pattern access benchmarks

Benchmarks characterizing memory access patterns: sequential forward, sequential reverse, strided, and random. Results expose how access regularity and stride distance affect effective payload bandwidth.

| File | Purpose |
|---|---|
| `pattern_benchmark.h` | Public interface for the pattern benchmark mode |
| `pattern_coordinator.cpp` | Orchestrates per-loop pattern selection, balanced order, timer setup, and status-bearing measurements |
| `pattern_work_plan.h` / `.cpp` | Finalizes strided/random worker ranges, phase-aware access/payload accounting, calibration arithmetic, and deterministic random offsets |
| `execution_patterns.cpp` | Orchestrates sequential forward/reverse and random read/write/copy warmup, calibration, measurement, and result construction |
| `execution_strided.cpp` | Orchestrates finalized phase-aware strided read/write/copy measurements and unavailable states |
| `execution_utils.cpp` | Calculates effective bandwidth and creates deterministic aligned random-offset sets/access counts |
| `helpers.cpp` | Connects sequential, strided, and random ARM64 kernels to the parallel framework and validates finalized worker plans |
| `validation.cpp` | Validates pattern benchmark configuration parameters |
| `pattern_statistics_manager.cpp` | Owns command-local pattern buffers and manages loop execution plus per-pattern statistical accumulators |
| `output.cpp` | Formats pattern benchmark results for console output |

---

### 2.6 src/warmup/ — Pre--benchmark warm-up

Warm-up passes reduce selected cold-start effects before timing. Main-memory bandwidth warm-up is bounded, cache
bandwidth warm-up covers the full target buffer, and latency warm-up page-touches without pre-traversing the chain.

| File | Purpose |
|---|---|
| `warmup.h` | Public warm-up API |
| `warmup_internal.h` | Internal warm-up helpers not exposed outside the module; provides template functions and shared inline chunk operations (`warmup_read_chunk_op`, `warmup_write_chunk_op`, `warmup_copy_chunk_op`) |
| `basic_warmup.cpp` | Parallel read/write/copy warm-ups over a bounded main-memory prefix (full buffer when small, otherwise `max(64 MiB, 10%)`) |
| `cache_warmup.cpp` | Targeted warm-up designed to fill a specific cache level before a cache-level benchmark |
| `latency_warmup.cpp` | Single-threaded page-touch warm-up for latency buffers; faults pages in without traversing the pointer chain |
| `pattern_warmup.cpp` | Warm-up pass tailored to the configured access pattern |

---

### 2.7 src/utils/ — Shared utilities

| File | Purpose |
|---|---|
| `benchmark.h` | Convenience umbrella header that includes all benchmark-related headers |
| `utils.h` / `.cpp` | Shared thread-joining and progress-indicator helpers |
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
| `test_config.cpp` | `ConfigTest` | Strict whole-token CLI/sweep parsing, validation, defaults, and derived buffer/access math |
| `test_messages.cpp` | `Messages*Test` | Console message string functions across all categories |
| `test_memory_utils.cpp` | `MemoryUtilsTest` | Memory helpers: pointer-chase chain construction and verification |
| `test_memory_manager.cpp` | `MemoryManagerTest` | Injected mmap/madvise policy, failures, and exact RAII unmapping |
| `test_buffer_manager.cpp` | `BufferManagerTest` | Pattern mapping policy, atomic allocation cleanup, initialized content, validation, and peak accounting |
| `test_benchmark_executor.cpp` | `BenchmarkExecutorTest` | Injected phase/chain failures, continuous latency sampling, and hardware executor contracts |
| `test_benchmark_runner.cpp` | `BenchmarkStatisticsCollectorTest`, `BenchmarkRunnerTest` | Status-bearing collection, reset/reserve contracts, checkpointing, and runner failure seams |
| `test_benchmark_work_plan.cpp` | `BenchmarkWorkPlanTest` | Exact payload/access planning, calibration, cyclic order, seed derivation, and duration classification |
| `test_analysis.cpp` | `AnalysisTest` | Injected TLB coordination, counters/status, boundary detection, validation, and paired analysis |
| `test_json_schema.cpp` | `JsonSchemaTest` | JSON output structure and field presence |
| `test_json_utils.cpp` | `JsonUtilsTest`, `JsonFileWriterTest` | JSON parse/statistics and atomic writer success/failure contracts |
| `test_output_printer.cpp` | `OutputPrinterTest` | Status-aware partial output, mode/cache composition, and size-unit boundaries |
| `test_sweep_runner.cpp` | `SweepRunnerTest` | Complete/partial/interrupted/failed attempt accounting and checkpoint behavior |
| `test_pattern_validation.cpp` | `PatternValidationTest` | Pattern benchmark parameter validation |
| `test_pattern_benchmark.cpp` | `PatternBenchmarkTest` | Pattern execution and statistics |
| `test_pattern_work_plan.cpp` | `PatternWorkPlanTest` | Strided phases, worker reduction, random partitions, exact payload work, and calibration |
| `test_core_to_core_messages.cpp` | `CoreToCoreMessagesTest` | Core-to-core console message strings |
| `test_core_to_core_cli.cpp` | `CoreToCoreCliTest` | Core-to-core CLI argument parsing |
| `test_core_to_core_runner.cpp` | `CoreToCoreRunnerTest` | Calibration, work planning, cyclic scenario order, deterministic failure seams, and real ARM64 integration paths |
| `test_executable_cli.cpp` | `ExecutableCliIntegrationTest` | Executable-level CLI routing, invalid config, JSON output, and pattern orchestration smoke coverage |
| `test_standard_kernels.cpp` | `StandardKernelIntegrationTest` | Real ARM64 standard-kernel ABI, tails, boundaries, checksums, and multi-worker execution |
| `test_statistics.cpp` | `StatisticsTest` | Statistical computations: median, percentiles, stddev, min, max |
| `test_timer.cpp` | `HighResTimerTest`, `HighResTimerIntegrationTest` | Exact conversion/failure seams plus one real monotonic smoke |
| `test_system_info.cpp` | `SystemInfoTest`, `SystemInfoIntegrationTest` | Deterministic fallbacks/errors plus four coherent hardware contracts |
| `test_tlb_chain.cpp` | `TlbChainTest` | Spread/packed planning, every traversal policy, explicit corruption statuses, and one ASM smoke |
| `test_tlb_measurement_scheduler.cpp` | `TlbMeasurementSchedulerTest` | Seeded balance, stop/error boundaries, callback contracts, convergence, and exact pass accounting |
| `test_tlb_runtime_policy.cpp` | `TlbRuntimePolicyTest` | Runtime profiles, calibration, convergence, memory budgets, and work estimates |
| `test_tlb_sweep_planner.cpp` | `TlbSweepPlannerTest` | Page-aligned base/refinement planning, stride bounds, deduplication, and source tracking |

**Current source inventory:** 28 test translation units contain 563 `TEST`/`TEST_F`/`TEST_P` declarations across 44
suite names as of 2026-07-10. Parameterized instantiations expand the executable to 584 runtime tests.

---

### Shared test helper headers

Three shared helper headers provide functionality reused across multiple test suites:

| File | Purpose |
|---|---|
| `test_config_helpers.h` | Provides `initialize_system_info(BenchmarkConfig&)` and `allocate_and_initialize_pattern_buffers(const BenchmarkConfig&, PatternBuffers&)` for shared system and pattern-buffer setup |
| `test_statistics_helpers.h` | Provides `capture_bw()`, `capture_lat()`, `capture_auto_tlb_breakdown()` helpers in `namespace test_statistics_helpers` — used by `StatisticsTest` to capture statistics output |
| `test_memory_system_calls.h` | Provides deterministic mmap/madvise/munmap state and a resetting fixture for allocation tests |

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
| `MacMiniM4vsMacbookAirM5_benchmark_comparison.png` | Historical Mac Mini M4 and MacBook Air M5 benchmark comparison |

---

## 6. .github/ — GitHub integration

| File | Purpose |
|---|---|
| `.github/pull_request_template.md` | Default pull request description template |
| `.github/ISSUE_TEMPLATE/bug_report.md` | Structured bug report template |

# Project Structure — macOS-memory-benchmark

**Version:** 0.63.0
**Platform:** ARM64 / AArch64 (Apple Silicon, macOS 26 or later)
**License:** GNU General Public License v3.0 or later

This document describes the layout of project files, organized by purpose. It is intended as a navigation aid for contributors and reviewers.

---

## Table of Contents

1. [Root-level files](#1-root-level-files)
   - [documents/ — Project documentation](#documents--project-documentation)
2. [src/ — Source code](#2-src--source-code)
   - [src/asm/](#21-srcasm--arm64-assembly-kernels)
   - [src/benchmark/](#22-srcbenchmark---benchmarking-infrastructure)
   - [src/core/](#23-srccore--core-utilities)
   - [src/output/](#24-srcoutput---output-layer)
   - [src/gpu_bandwidth/](#25-srcgpu_bandwidth--metal-gpu-bandwidth)
   - [src/llm_memory/](#26-srcllm_memory--synthetic-llm-memory-profile)
   - [src/pattern_benchmark/](#27-srcpattern_benchmark--pattern-access-benchmarks)
   - [src/warmup/](#28-srcwarmup--pre-benchmark-warm-up)
   - [src/utils/](#29-srcutils--shared-utilities)
   - [src/third_party/](#210-srcthird_party--vendored-dependencies)
3. [tests/ — Test suite](#3-tests--test-suite)
4. [results/ — Benchmark result data](#4-results---benchmark-result-data)
5. [pictures/ — Documentation images](#5-pictures--documentation-images)
6. [script-examples/ — Run and plotting helpers](#6-script-examples--run-and-plotting-helpers)
7. [.github/ — GitHub integration](#7-github--github-integration)

---

## 1. Root-level files

### Entry point

| File | Purpose |
|---|---|
| `main.cpp` | Program entry point; performs primary-mode conflict scan, dispatches dedicated LLM/GPU/core-to-core paths before the general parser, and runs the general standard/pattern/TLB and sweep pipelines; each standalone command boundary owns its output-session lifetime |

### Build and tooling

| File | Purpose |
|---|---|
| `Makefile` | Primary build system; discovers C++/Objective-C++/assembly, targets sources and final links at macOS 26.0, accepts a configurable `GTEST_DIR`, compiles `.mm` with ARC, links Metal/Foundation, and produces release/test binaries |
| `coverage.sh` | Runs isolated LLVM unit/all-test C++/Objective-C++ source coverage builds under `/tmp` without replacing normal workspace binaries |
| `.clang-format` | Clang-Format style baseline for C++ sources |

### Root-level documentation

| File | Purpose |
|---|---|
| `README.md` | Project overview, quick-start instructions, and feature summary |
| `CHANGELOG.md` | Version history and release notes |
| `LICENSE` | GNU General Public License v3.0 license text; source headers grant version 3 or any later version |

### documents/ — Project documentation

| File | Purpose |
|---|---|
| `documents/API.md` | Supported process-level machine interface: output targets, stream separation, current schema identities, and result acceptance |
| `documents/CAPABILITIES.md` | Measurement capability overview and interpretation notes |
| `documents/MANUAL.md` | Complete user manual: all CLI flags, modes, output formats, and usage examples |
| `documents/PARAMETER_MATRIX.md` | Mode/flag compatibility, sweep support, and incompatible-option matrix |
| `documents/TECHNICAL_SPECIFICATION.md` | Internal architecture, data structures, and implementation decisions |
| `documents/CONTRIBUTING.md` | Contribution guidelines, coding standards, and pull request process |
| `documents/CODE_OF_CONDUCT.md` | Community standards |
| `documents/SECURITY.md` | Vulnerability disclosure policy |
| `documents/TLB_ANALYSIS_WHITEPAPER.md` | Whitepaper: current TLB analysis methodology, schema, and interpretation limits |
| `documents/LATENCY_WHITEPAPER.md` | Whitepaper: cache and memory latency measurement methodology |
| `documents/CORE_TO_CORE_WHITEPAPER.md` | Whitepaper: calibrated two-thread token-handoff methodology, audit schema, and interpretation limits |
| `documents/GPU_BANDWIDTH_WHITEPAPER.md` | Whitepaper: Metal compute bandwidth methodology, GPU schema 1, validation, capability limits, and maintenance policy |
| `documents/LLM_MEMORY_PROFILE_WHITEPAPER.md` | Whitepaper: CPU and capability-gated Metal decode/prefill contiguous/paged methodologies, with exact payload formulas, resources, validation, schema 1, and interpretation limits |
| `documents/PROJECT_STRUCTURE.md` | This file |

---

## 2. src/ — Source code

All production C++, Objective-C++, and ARM64 assembly lives under `src/`. Headers use include paths relative to `src/`
(e.g., `#include "core/config/config.h"`). Metal/Objective-C types are confined to private `.mm` platform boundaries.

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
| `core_to_core_latency.s` | Acquire/release token-exchange ping-pong loop for core-to-core protocol latency |
| `llm_decode_memory.s` | Descriptor-driven weight/KV read and temporal current-token K/V append kernel for the synthetic LLM profile |
| `llm_decode_memory_paged.s` | Paged-KV decode kernel with explicit timed block-table loads, paired append lookup, separate K/V scan lookups, exact tails, and versioned checksum evidence |
| `llm_prefill_memory.s` | Contiguous-KV prefill kernel with full-prompt writes, tiled causal-prefix K/V scans, and operation-ordinal checksum evidence |
| `llm_prefill_memory_paged.s` | Paged-KV prefill kernel with block-exclusive prompt writes, exact partial-prefix K/V scans, timed uint32 table loads, and versioned checksum evidence |

---

### 2.2 src/benchmark/ — Benchmarking infrastructure

The benchmark subsystem owns the standard CPU pipeline, standalone TLB analysis, core-to-core analysis, and shared sweep execution. Pattern and GPU measurement pipelines live in dedicated subtrees, while reusable statistics and output helpers live under `src/utils/` and `src/output/`.

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
| `sweep_runner.h` / `.cpp` | Shared deterministic sweep executor, completion classification, retained terminal envelopes, and file/lazy-stdout checkpoint dispatch; provides standard/pattern/TLB entry-point orchestration and is called directly by the core-to-core sweep runner |

#### TLB analysis mode

| File | Purpose |
|---|---|
| `tlb_analysis.h` / `.cpp` | Standalone `--analyze-tlb` mode: compares spread/packed chains across locality sweeps, supports stride/chain-mode/density controls, and returns execution status separately from its in-memory result payload |
| `tlb_analysis_json.h` / `.cpp` | Builds schema-4 TLB payloads in memory and retains the shared atomic-file save adapter |
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
| `core_to_core_latency_runner.cpp` | Per-scenario calibration, 128-byte shared-state isolation, balanced loop scheduling, unpinned two-thread ping-pong execution, robust summaries, and in-memory result collection |
| `core_to_core_latency_cli.cpp` | CLI argument parsing, direct/sweep output-session ownership, and entry point for the core-to-core mode |
| `core_to_core_latency_json.h` / `.cpp` | Serializes schema-2 work plans, loop audit records, completion state, and results |
| `core_to_core_sweep_runner.h` / `.cpp` | Core-to-core Cartesian sweeps, retained terminal envelopes, atomic file checkpoints, and lazy stdout checkpoints |

---

### 2.3 src/core/ — Core utilities

Core infrastructure for configuration, memory management, macOS system introspection, signal handling, and Mach timing.

#### src/core/config/

| File | Purpose |
|---|---|
| `config.h` | `BenchmarkConfig` structure for standard, pattern, standalone TLB, and their common sweep settings; core-to-core, GPU, and LLM use separate config types |
| `constants.h` | Named constants for CPU/GPU/LLM memory limits, calibration, grid/dispatch/payload guardrails, buffer sizing, and latency access counts |
| `version.h` | `SOFTVERSION` macro (semantic version string, currently `"0.63.0"`) |
| `mode_selector.h` / `.cpp` | Pure primary-mode scan and conflict detection before mode-specific parsing; routes standard, pattern, TLB, core-to-core, GPU, and LLM deterministically while skipping one opaque output value |
| `argument_parser.cpp` | Parses standard, pattern, and standalone TLB options into `BenchmarkConfig`; core-to-core, GPU, and LLM are pre-routed to dedicated parsers |
| `config_validator.cpp` | Validates the parsed configuration; emits errors for out-of-range or conflicting settings |
| `buffer_calculator.cpp` | Derives buffer sizes for each cache/memory level from the validated configuration and detected system parameters |
| `sweep_utils.h` / `.cpp` | Shared structural sweep parsing and overflow-safe Cartesian run counting used by standard and core-to-core sweep parsers |

#### src/core/signal/

| File | Purpose |
|---|---|
| `signal_handler.h` / `.cpp` | Installs SIGINT/SIGTERM handling, exposes an exact-mask-restoring RAII guard for worker lifetimes, and reports pending interruption between phases |

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
| `benchmark_qos.h` / `.cpp` | Shared best-effort main-thread QoS preparation with requested/applied/code audit result |
| `page_size.h` / `.cpp` | Shared native page-size query boundary for accounting, validation, metadata, and page-prefault operations |

#### src/core/timing/

| File | Purpose |
|---|---|
| `timer.h` / `.cpp` | High-resolution Mach timer with exact tick conversion plus a deterministic clock/timebase provider seam |

---

### 2.4 src/output/ — Output layer

This subsystem contains shared console messages, standard-benchmark formatters, descriptive-statistics rendering, and JSON serialization. Mode-specific output orchestration can remain beside the corresponding benchmark mode, while user-facing message text is centralized under `messages/`.

#### src/output/console/

| File | Purpose |
|---|---|
| `output_printer.h` / `.cpp` | Prints standard CLI help/banner text, configuration and cache information, and status-aware per-loop standard benchmark results |
| `statistics.h` / `.cpp` | Coordinates standard multi-loop summary selection, diagnostics, and console output using the shared calculator and renderer |
| `statistics_renderer.h` / `.cpp` | Renders descriptive-statistics summaries in the canonical field order with shared precision, indentation, and diagnostic controls |

##### src/output/console/messages/

All user-facing text strings are centralized here. Each `.cpp` file implements a specific category of messages declared in the shared API header.

| File | Purpose |
|---|---|
| `messages_api.h` | Declares the `Messages` namespace containing all string-returning functions |
| `cache_messages.cpp` | Cache-level result labels and headings |
| `config_messages.cpp` | Configuration echo and validation error text |
| `core_to_core_messages.cpp` | Core-to-core mode status and result messages |
| `gpu_bandwidth_messages.cpp` | GPU help, status, result, interpretation, warning, and validation messages |
| `llm_memory_messages.cpp` | LLM help, validation, lifecycle, headline, interpretation, and quality-warning messages |
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
| `json_output_api.h` | Public standard/pattern payload builders and the shared atomic-file writer interface; this C++ API is distinct from the process contract in `API.md` |
| `json_output.cpp` | Builds standard/pattern root payloads, adds timestamp/version metadata, and retains file-save adapters; direct command dispatch can reuse the prebuilt object through `JsonOutputSession` |
| `json_output_session.h` / `.cpp` | Classifies raw output targets, applies mode-specific file-path policy, lazily dispatches checkpoints, routes command-scoped human stdout, and emits checked final JSON through the retained original stdout buffer; result-producing direct modes and CPU sweeps use it, including the LLM file-checkpoint/final-stdout lifecycle |
| `builder.cpp` | Builds common mode configuration metadata, including resolved chain, seed, calibration, scheduling, and worker policies |
| `standard.cpp` | Active standard schema-3 serializer for completion state, loop measurements, and main/cache aggregates |
| `patterns.cpp` | Serializes pattern benchmark results |
| `file_writer.cpp` | Atomically writes the completed JSON document to the requested output path, creating parent directories as needed |

---

### 2.5 src/gpu_bandwidth/ — Metal GPU bandwidth

Standalone GPU schema 1 implementation. The pure C++ planner/runner/result model is isolated from the private
Objective-C++ Metal backend so deterministic unit tests do not require GPU work.

| File | Purpose |
|---|---|
| `gpu_bandwidth.h` / `.cpp` | Dedicated `GpuBandwidthConfig`, exact option-whitelist parser, GPU help, standalone output-session boundary, QoS/signal scope, console handoff, and terminal stdout emission |
| `gpu_work_plan.h` / `.cpp` | Pure read/write/copy pass limits, exact payload, seed derivation, cyclic order, calibration arithmetic, frozen 8192-threadgroup grid cap/geometry, and `gpu-work-plan-v1` identity |
| `gpu_backend.h` / `.cpp` | Objective-C-free synchronous/noexcept backend contract, device/resource/phase/validation metadata, factory declaration, and status string mappings |
| `gpu_runner.h` / `.cpp` | Backend-independent calibration, warmup/precondition/timing/validation orchestration, completion-wins interruption, counters, aggregates, resource lifecycle, and transport-neutral logical checkpoints with a file-scoped post-release replacement |
| `gpu_json.h` / `.cpp` | GPU schema 1 builder and shared atomic-writer adapter; the prebuilt payload can also use the command's stdout session and preserves exact integer strings, nullable state, errors, provenance, and nested audit records |
| `gpu_kernels_source.h` | Private canonical embedded MSL 2.3 source, kernel revision, integer pattern/checksum contract, and exact source bytes hashed at runtime |
| `metal_gpu_backend.mm` | Only Objective-C++/Metal boundary: ARC/autorelease pools, capability checks, runtime compilation, private/tracked buffers, shared/tracked status, serial command buffers, timestamps, validation, and test readback |

---

### 2.6 src/llm_memory/ — Synthetic LLM memory profile

Standalone generic schema-1 vocabulary with eight active backend/phase/layout profiles: four CPU profiles and four
capability-gated Metal profiles spanning decode/prefill and contiguous/paged KV. Pure logical phase planning,
deterministic paged geometry/permutation, backend-specific execution planning/evidence, the Objective-C-free backend
contract, CPU mapping and ARM64 execution, the Objective-C++ Metal boundary, environment capture, console composition,
and serialization are separated so deterministic unit tests can inject platform and allocation behavior without running
hot kernels. Capability failures remain explicit, and no unsupported Metal request falls back to CPU.

| File | Purpose |
|---|---|
| `llm_memory.h` / `.cpp` | Separate config/status foundation, strict exact-whitelist parser, required/default worker/seed resolution, complete command boundary, backend factory ownership, peak-memory admission, QoS/signal scope, console output, and file/stdout transport orchestration |
| `llm_kv_layout.h` / `.cpp` | Pure checked paged-KV geometry, block ownership, SplitMix64/Fisher–Yates permutation and little-endian table hash, exact physical/logical/padding/table/lookup/accounting math, and transient preparation estimates |
| `llm_prefill.h` / `.cpp` | Checked prefill tile, causal-pair, payload, paged floor-sum lookup, token/block cost partition, versioned ownership evidence, write/checksum oracle, and semantic event-trace planning |
| `llm_work_plan.h` / `.cpp` | Checked phase-applicable weight/KV geometry, payload/metadata/accounted math, memory budget, scenario limits, exact CPU ownership evidence, Metal segment/argument-buffer/resource/grid plans, and frozen methodology/component/plan identities |
| `llm_backend.h` / `.cpp` | Objective-C-free synchronous backend lifecycle and task boundary, generic task identity/timing/completion/validation, tagged CPU/Metal task and command evidence, auxiliary-memory contract, stable statuses/reasons, checked evidence accessors, and backend factory |
| `llm_cpu_backend.h` / `.cpp` | Active CPU adapter that owns the timer and prepared contiguous or paged resources, delegates allocation/execution to the executor, validates CPU worker/QoS/timer/checksum invariants, and converts executor evidence into the generic task result |
| `llm_metal_backend.h` | Objective-C-free planning, capability, contiguous/paged resource and memory admission, cyclic block-owner grids with per-threadgroup accounting, decode/prefill parameter ABIs, timestamp/checksum mapping, diagnostics, and active-profile test seams |
| `llm_metal_backend.mm` | Objective-C++ Metal boundary for capability probing, selected-profile runtime compilation, Tier 2 argument encoding, private/shared allocation, table upload/validation, exact-tail dispatch, paged-prefill cyclic ownership, GPU timestamps, layout-aware dual-mod32 checksum, full-prompt K/V-write and padding validation, diagnostics, and idempotent cleanup |
| `llm_metal_kernels_source.h` | Canonical embedded MSL 2.3 source with profile-specific decode/prefill contiguous/paged parameter ABIs, common foundation entrypoints, scenario-specialized workload entrypoints, named-lane volatile paged-table publication, and phase-specific K/V-write/padding validation; exact selected source bytes are hashed at runtime |
| `llm_executor.h` / `.cpp` | CPU-specific full-size mappings, paged-table preparation, deterministic initialization/pre-touch, phase/layout descriptor materialization, independent checksum oracles, padding canaries, synchronized worker team, timer boundary, and ARM64 adapters retained behind `LlmCpuBackend` |
| `llm_runner.h` / `.cpp` | Backend-independent lifecycle, per-scenario automatic calibration or exact-work planning, generic task acceptance, frozen plans, cyclic loop order, status/counters, task-boundary interruption, aggregates, warnings, and logical checkpoints; resources are released before the command-terminal checkpoint |
| `llm_json.h` / `.cpp` | Ordered generic LLM schema-1 builder plus conservative output-peak estimator, with backend/phase/layout identity, resolved Metal segmentation, capability/resource/task evidence, decimal-string exact integers, nullable non-applicability, traffic diagnostics, environment evidence, and interpretation contract |
| `llm_output.h` / `.cpp` | Human-readable work-unit/model-payload/accounted geometry, Metal device/resource/task evidence, scenario headlines, interpretation limits, and evidence-backed quality warnings through centralized message helpers |
| `llm_environment.h` / `.mm` | Objective-C-free snapshot type plus macOS thermal-state and Low Power Mode capture through Foundation |

---

### 2.7 src/pattern_benchmark/ — Pattern access benchmarks

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

### 2.8 src/warmup/ — Pre-benchmark warm-up

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

### 2.9 src/utils/ — Shared utilities

| File | Purpose |
|---|---|
| `benchmark.h` | Convenience umbrella header for commonly used standard-benchmark, timing, memory, output, assembly, and warm-up interfaces |
| `utils.h` / `.cpp` | Shared thread-joining helpers and the TTY-aware stderr progress spinner used by benchmark modes |
| `json_utils.h` / `.cpp` | JSON helper functions shared between the TLB, core-to-core, and standard output serializers |
| `cyclic_order.h` / `.cpp` | Shared deterministic cyclic ordering used by CPU and GPU planners |
| `seed_utils.h` / `.cpp` | Shared SplitMix64 derivation and generate-once seed helper with deterministic provider seam |
| `hash_utils.h` / `.cpp` | CommonCrypto-based SHA-256 helper used for exact embedded MSL source provenance |
| `numeric_utils.h` / `.cpp` | Overflow-safe size arithmetic plus bounded pilot-count and duration-calibration helpers |
| `descriptive_statistics.h` / `.cpp` | Canonical average, percentile, sample-deviation, coefficient-of-variation, and median-absolute-deviation calculations |

---

### 2.10 src/third_party/ — Vendored dependencies

| File | Purpose |
|---|---|
| `nlohmann/json.hpp` | nlohmann/json single-header library (MIT license); used throughout the JSON output subsystem |

---

## 3. tests/ — Test suite

GoogleTest-based unit and integration tests are supplemented by a focused Python script-example entry test.
`make test-script-examples` exercises the current JSON-reading examples directly, and `make test-all` runs it after all
GTest cases pass. Python 3 is required; `jq` is optional, and only its dedicated test branch is skipped when it is not
installed. All `.cpp` files are picked up automatically by the Makefile. Tests named `*Integration*` are excluded from
`make test` (unit-only) and run through the integration or all-test targets.

| File | Suite name | Coverage focus |
|---|---|---|
| `test_config.cpp` | `ConfigTest` | Strict whole-token CLI/sweep parsing, validation, defaults, and derived buffer/access math |
| `test_signal_handler.cpp` | `BenchmarkSignalMaskGuardTest` | Exact thread-mask restoration, caller-preserved blocked signals, and nested scope ownership |
| `test_messages.cpp` | `Messages*Test` | Console message string functions across all categories |
| `test_memory_utils.cpp` | `MemoryUtilsTest` | Memory helpers: pointer-chase chain construction and verification |
| `test_memory_manager.cpp` | `MemoryManagerTest` | Injected mmap/madvise policy, failures, and exact RAII unmapping |
| `test_numeric_utils.cpp` | `NumericUtilsTest` | Overflow-safe arithmetic, duration calibration, pilot counts, and quantization boundaries |
| `test_llm_memory_contract.cpp` | `LlmMemoryContractTest` | Independent executable specification for all eight decode/prefill backend/layout methodology identities, payloads, paged geometry/lookup goldens, append and full-prompt writes, checksum/event-trace identities, descriptor ABI layouts, and schema acceptance; it does not exercise production LLM code |
| `test_llm_memory_config.cpp` | `LlmMemoryConfigTest` | Config/status defaults, exact standalone decode/prefill parsing including paged prefill, phase/layout rules and stable reasons, strict decimal errors, raw output values, help isolation, worker/seed resolution, geometry/work-limit preflight, and incompatible options |
| `test_llm_memory_work_plan.cpp` | `LlmMemoryWorkPlanTest` | Production checked decode/prefill geometry and payloads, physical/padding/table/lookup/accounted math including paged-prefill `N+2*M`, deterministic permutation/hash, CPU token/block ownership, Metal cyclic block-owner scheduling and component identities, memory budget, descriptor/range/layout invariants, worker reduction, scenario caps/calibration, and cyclic order |
| `test_llm_memory_executor.cpp` | `LlmMemoryExecutorTest` | CPU phase/layout dispatch, atomic mapping/table-preparation failure seams, contiguous and paged decode/prefill descriptor materialization, independent checksums, padding canaries, synchronized timing, QoS, cancellation, and fake-kernel validation |
| `test_llm_memory_runner.cpp` | `LlmMemoryRunnerTest` | Fake-backend lifecycle and generic decode/prefill task seams, lifecycle unsupported/failure handling, common identity/timing/completion/validation acceptance, exact calibration/single-unit/freeze/frozen-warmup order, cyclic measurements, status/counter/aggregate semantics, interruption, release/checkpoint precedence, auxiliary budgeting, and runner exception boundaries |
| `test_llm_memory_json.cpp` | `LlmMemoryJsonTest` | Schema-1 identity for all eight CPU/Metal profiles, exact nullable geometry/workers, paged layout/permutation, prefill geometry, CPU ownership, Metal cyclic owner/per-threadgroup cost evidence, segmentation, capability/resource/task K/V-write/lookup/padding evidence, output-peak, status, interpretation, environment, and checkpoints |
| `test_llm_memory_output.cpp` | `LlmMemoryOutputTest` | Exact decode/prefill, contiguous/paged, and Metal device/resource/table/task K/V-write/lookup/padding plus cyclic owner/per-threadgroup cost formatting, interpretation text, and deduplicated warnings |
| `test_llm_memory_kernels.cpp` | `LlmMemoryKernelIntegrationTest` | Real contiguous/paged decode/prefill ARM64 descriptor/kernel scenarios, paged tails/lookups, full-prompt writes, exact partial tiled scans, checksum, padding, worker-count, and AAPCS64 coverage |
| `test_llm_metal_backend.cpp` | `LlmMetalBackendTest`, `LlmMetalBackendIntegrationTest`, `LlmMetalBackendFailureInjectionIntegrationTest` | Pure capability, contiguous/paged segmentation, argument-buffer, decode/prefill ABIs, source-hash and prefill loop-order audit, phase-aware checksum/reason mapping, paged-prefill `N+2*M`, cyclic no-duplication ownership and per-threadgroup accounted-byte evidence, plus real-device selected-profile compilation, Tier-2 slots, private table/K/V initialization, exact-tail decode, full-prompt/tiled-prefix prefill, final-ordinal write/padding detection, interruption/failure cleanup, and idempotent release |
| `test_buffer_manager.cpp` | `BufferManagerTest` | Pattern mapping policy, atomic allocation cleanup, initialized content, validation, and peak accounting |
| `test_benchmark_executor.cpp` | `BenchmarkExecutorTest` | Injected phase/chain failures, continuous latency sampling, and hardware executor contracts |
| `test_benchmark_runner.cpp` | `BenchmarkStatisticsCollectorTest`, `BenchmarkRunnerTest` | Status-bearing aggregation, schema-3 retained snapshots, checkpointing, interruption, and runner exception/failure seams |
| `test_benchmark_work_plan.cpp` | `BenchmarkWorkPlanTest` | Exact payload/access planning, calibration, cyclic order, seed derivation, and duration classification |
| `test_gpu_bandwidth.cpp` | `GpuBandwidthParserTest`, `GpuMemoryBudgetTest`, `GpuRunnerTest`, `GpuJsonTest` | Strict standalone parsing and raw output spelling, memory budgets, fake-backend calibration/execution/failure/interruption and file/stdout checkpoint semantics, counters, and schema-1 serialization |
| `test_gpu_work_plan.cpp` | `GpuWorkPlanTest`, `GpuTimedAccumulatorOracleTest` | GPU constants, cyclic order, seed domains, pass/payload caps, calibration, vector/tail/grid geometry, frozen identities, and timed-accumulator oracle behavior |
| `test_gpu_metal_backend.cpp` | `GpuMetalBackendIntegrationTest` | Real-Metal capability/runtime compile, private/shared tracked resources, read/write/copy/tail correctness, timestamps, validation, and byte readback |
| `test_mode_selector.cpp` | `ModeSelectorTest` | All six primary modes and aliases, complete pairwise conflict order, duplicate ownership, and opaque flag-shaped output targets |
| `test_hash_utils.cpp` | `HashUtilsTest` | CommonCrypto SHA-256 standard vectors and source-provenance helper behavior |
| `test_analysis.cpp` | `AnalysisTest` | Injected TLB coordination, counters/status, boundary detection, validation, and paired analysis |
| `test_json_schema.cpp` | `JsonSchemaTest` | Current standard schema-3 output structure, completion fields, and other mode schema contracts |
| `test_script_examples.py` | Python script-example entry test | Current schema-3 JSON entry paths, local sanity checks, and metric extraction for the bundled standard-memory examples, including the optional jq-backed shell path when jq is installed |
| `test_json_output_session.cpp` | `JsonOutputTargetTest`, `JsonOutputSessionTest` | Exact sentinel/path classification, lazy checkpoint dispatch, atomic-file parity, stdout routing/restoration, and stream failure containment |
| `test_json_utils.cpp` | `JsonUtilsTest`, `JsonFileWriterTest` | JSON parse/statistics and atomic writer success/failure contracts |
| `test_output_printer.cpp` | `OutputPrinterTest`, `OutputPrinterCustomCacheUnitsTest` | Status-aware partial output, mode/cache composition, and custom-cache size-unit boundaries |
| `test_sweep_runner.cpp` | `SweepRunnerTest` | Current-only nested standard schema-3 completion classification, complete/partial/interrupted/failed attempt accounting, schema-1 envelopes, and file/lazy-stdout checkpoint behavior |
| `test_sweep_utils.cpp` | `SweepUtilsTest` | Shared sweep parsing, empty-dimension behavior, and overflow-safe Cartesian counts |
| `test_pattern_validation.cpp` | `PatternValidationTest` | Pattern benchmark parameter validation |
| `test_pattern_benchmark.cpp` | `PatternBenchmarkTest` | Pattern execution and statistics |
| `test_pattern_work_plan.cpp` | `PatternWorkPlanTest` | Strided phases, worker reduction, random partitions, exact payload work, and calibration |
| `test_core_to_core_messages.cpp` | `CoreToCoreMessagesTest` | Core-to-core console message strings |
| `test_core_to_core_cli.cpp` | `CoreToCoreCliTest` | Core-to-core CLI argument parsing |
| `test_core_to_core_runner.cpp` | `CoreToCoreRunnerTest` | Calibration, work planning, cyclic scenario order, deterministic failure seams, and real ARM64 integration paths |
| `test_executable_cli.cpp` | `ExecutableCliIntegrationTest` | Executable-level CLI routing, current standard schema-3 direct/sweep payloads, split stdout/stderr capture, bounded child execution, sentinel artifact rules, unsupported GPU evidence, bounded contiguous/paged LLM file/stdout/checkpoint contracts, and pattern orchestration smoke coverage |
| `test_standard_kernels.cpp` | `StandardKernelIntegrationTest`, `PatternKernelIntegrationTest` | Real ARM64 standard/pattern kernel ABI, tails, boundaries, checksums, and multi-worker execution |
| `test_statistics.cpp` | `StatisticsTest` | Standard multi-loop summary composition, mode filtering, loop/sample population separation, and rendered values |
| `test_descriptive_statistics.cpp` | `DescriptiveStatisticsTest` | Canonical shared percentiles, deviation, CV, and MAD contracts |
| `test_statistics_renderer.cpp` | `StatisticsRendererTest` | Shared console-summary ordering, precision, indentation, and diagnostics |
| `test_timer.cpp` | `HighResTimerTest` | Deterministic Mach-time conversion, creation failures, seconds/nanoseconds results, and tick wraparound |
| `test_system_info.cpp` | `BenchmarkQosTest`, `SystemInfoTest`, `SystemInfoIntegrationTest` | Deterministic QoS and system-information success/failure contracts plus one real-provider integration smoke |
| `test_tlb_chain.cpp` | `TlbChainTest` | Spread/packed planning, every traversal policy, explicit corruption statuses, and one ASM smoke |
| `test_tlb_measurement_scheduler.cpp` | `TlbMeasurementSchedulerTest` | Seeded balance, stop/error boundaries, callback contracts, convergence, and exact pass accounting |
| `test_tlb_runtime_policy.cpp` | `TlbRuntimePolicyTest` | Runtime profiles, calibration, convergence, memory budgets, and work estimates |
| `test_tlb_sweep_planner.cpp` | `TlbSweepPlannerTest` | Page-aligned base/refinement planning, stride bounds, deduplication, and source tracking |
| `test_utils.cpp` | `SeedUtilsTest`, `ProgressSpinnerTest`, `UtilsTest` | Seed-provider behavior, TTY-gated spinner rendering/cleanup, and worker-thread joining |

### tests/fixtures/ — Script-example input fixtures

| File | Purpose |
|---|---|
| `standard-schema-v3-complete-current.json` | Minimized structurally faithful current full standard result for the two plotter JSON entry paths |
| `standard-schema-v3-custom-complete-current.json` | Minimized structurally faithful current custom-cache standard result for shell jq/Python extraction and stride/TLB summaries |
| `README.md` | Records source HEAD, capture commands, reduction policy, example ownership, and the current-only fixture refresh policy |

Volatile source/test counts and the authoritative generated inventory are maintained in `DRY_CHECK.md`.

---

### Shared test helper headers

Four shared helper headers support deterministic setup and output capture across the test suite:

| File | Purpose |
|---|---|
| `test_config_helpers.h` | Provides `initialize_system_info(BenchmarkConfig&)` and `allocate_and_initialize_pattern_buffers(const BenchmarkConfig&, PatternBuffers&)` for shared system and pattern-buffer setup |
| `test_statistics_helpers.h` | Provides `empty_values()`, `capture_main_bandwidth()`, and `capture_auto_tlb_breakdown()` in `namespace test_statistics_helpers` for `StatisticsTest` output capture |
| `test_memory_system_calls.h` | Provides deterministic mmap/madvise/munmap state and a resetting fixture for allocation tests |
| `test_timer_system_calls.h` | Provides deterministic Mach absolute-time hooks and a scope-bound reset guard for timer-dependent unit tests |

---

## 4. results/ — Benchmark result data

Historical JSON, CSV, and text output from benchmark runs on specific hardware, organized by software-version
subdirectory. The files are retained as examples and historical schema evidence; they are not current 0.63.0
methodology baselines unless explicitly identified as such. Historical standard JSON is not an input to current
bundled standard-result consumers; separately governed TLB tooling retains its documented legacy-input policy.

```
results/
  0.63.0/
    AppleM5_LLM_working_set_scaling.md  — Apple M5 CPU decode working-set comparison note
    M4MacMiniVSM5MacbookAir_LLM.md  — Informal LLM CPU decode comparison
    apple_m5_cpu_decode_contiguous_weights_256mib_context_32768.json  — Current LLM schema-1 sample
    apple_m5_cpu_decode_contiguous_weights_1024mib_context_131072.json  — Current LLM schema-1 sample
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

The current-version result notes include an
[Apple M5 CPU decode working-set comparison](../results/0.63.0/AppleM5_LLM_working_set_scaling.md) with links to its
complete schema-1 JSON records, and an
[informal M4 Mac mini versus M5 MacBook Air LLM CPU decode comparison](../results/0.63.0/M4MacMiniVSM5MacbookAir_LLM.md).
The Apple M5 pair illustrates scaling between two model geometries.
The M4 run had a Docker container active in the background, so the comparison is an illustrative result rather than a
controlled methodology baseline.

---

## 5. pictures/ — Documentation images

Tracked PNG chart archive generated from benchmark result data. Several files are historical and are not referenced by current documentation; their filenames preserve historical labels rather than establishing current causal interpretations.

| File | Content |
|---|---|
| `MacMiniM4_memory_hierarchy_v0_53_5.png` | Historical memory hierarchy bandwidth/latency overview, Mac Mini M4 |
| `MacMiniM4_cache_latency_with_TLB.png` | Historical cache-latency/locality chart, Mac Mini M4 |
| `MacMiniM4_cache_latency_with_TLB_5loops_p50_values.png` | P50 latency across 5 measurement loops, Mac Mini M4 |
| `MacMiniM4_cache_latency_with_TLB_5loops_p50_values_2.png` | Alternate P50 latency chart, Mac Mini M4 |
| `MacMiniM4_cache_latency_with_STRIDE_TLB.png` | Historical latency-by-stride/locality chart, Mac Mini M4 |
| `MacMiniM4_TLB_analysis_with_64_STRIDE.png` | Historical analyze-TLB trend at 64-byte stride, Mac Mini M4 |
| `MacMiniM4_TLB_analysis_with_128_STRIDE.png` | Historical analyze-TLB trend at 128-byte stride, Mac Mini M4 |
| `MacMiniM4_cache_latency_TLB_16KB.png` | Historical cache-latency detail around 16 KiB locality, Mac Mini M4 |
| `MacMiniM4_cache_latency_2_TLB_16KB.png` | Second historical cache-latency chart around 16 KiB locality, Mac Mini M4 |
| `MacBookAirM5_latency_memory_hierarchy.png` | Memory hierarchy latency, MacBook Air M5 |
| `MacBookAirM5_latency_vs_cache-stride-tlb.png` | Latency across cache-size, stride, and configured locality settings, MacBook Air M5 |
| `MacMiniM4vsMacbookAirM5_benchmark_comparison.png` | Historical Mac Mini M4 and MacBook Air M5 benchmark comparison |

---

## 6. script-examples/ — Run and plotting helpers

Example shell workflows and Python/Matplotlib plotters for tracked benchmark outputs. The four standard-memory examples
are maintained in lockstep with the current producer rather than as a compatibility library. Each of those JSON-reading
entry points performs a small local release-version/schema-3/completion/field sanity check and then reads the current
metric paths it needs. Version 0.63.0 therefore requires exact top-level `version: "0.63.0"`. Standard schema 2,
unversioned historical layouts, alternative modes, and other explicit standard versions are unsupported by those four
examples. Separately governed mode-specific tools retain only the history policy documented in their own row below.

| File | Purpose |
|---|---|
| `final_output.txt` | Small bundled sample of pooled latency statistics for `plot_cache_percentiles.py`; regenerated by `latency_test_script.sh` |
| `latency_test_script.sh` | Sweeps custom cache size and configured latency-locality windows, writes per-run JSON, and extracts current pooled sample statistics into `final_output.txt` using jq when available or Python 3 otherwise |
| `latency_test_script_stride_tlb.sh` | Sweeps cache size, configured locality, and latency stride; uses embedded Python 3 to read current pooled sample statistics, retains per-run JSON, and requires one complete CSV row per planned run |
| `plot_M4vsM5_benchmark_comparison.py` | Compares effective payload bandwidth and latency from two explicitly supplied current standard schema-3 JSON files |
| `plot_analyzetlb.py` | Plots standalone TLB locality trends, including the paired spread/packed delta in current schemas and supported legacy data |
| `plot_benchmark-memory-latency-hierarchy.py` | Plots memory-hierarchy latency from an explicit current standard schema-3 JSON or console-text statistics input using the current producer's labels |
| `plot_cache_percentiles.py` | Plots a selected pooled latency statistic by cache size and configured locality from `final_output.txt` |
| `plot_cache_percentiles_stride_tlb.py` | Plots a selected pooled latency statistic from the stride/locality sweep CSV, with optional stride and locality filters |

---

## 7. .github/ — GitHub integration

| File | Purpose |
|---|---|
| `.github/pull_request_template.md` | Default pull request description template |
| `.github/ISSUE_TEMPLATE/bug_report.md` | Structured bug report template |

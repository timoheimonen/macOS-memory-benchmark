# Project Structure — macOS-memory-benchmark

**Platform:** ARM64 / AArch64 (Apple Silicon, macOS 26 or later)
**License:** GNU General Public License v3.0 or later

This document describes the layout of project files, organized by purpose. It is intended as a navigation aid for contributors and reviewers.

---

## Table of Contents

1. [Root-level files](#1-root-level-files)
   - [documents/ — Project documentation](#documents--project-documentation)
2. [src/ — Source code](#2-src--source-code)
   - [src/asm/](#21-srcasm--arm64-assembly-kernels)
   - [src/benchmark/](#22-srcbenchmark--core-infrastructure)
   - [src/core/](#23-srccore--core-utilities)
   - [src/cpu_bandwidth/](#24-srccpu_bandwidth--standalone-cpu-bandwidth-mode)
   - [src/cpu_latency/](#25-srccpu_latency--standalone-cpu-latency-mode)
   - [src/output/](#26-srcoutput--output-layer)
   - [src/gpu_bandwidth/](#27-srcgpu_bandwidth--metal-gpu-bandwidth)
   - [src/llm_memory/](#28-srcllm_memory--synthetic-llm-memory-profile)
   - [src/pattern_benchmark/](#29-srcpattern_benchmark--pattern-access-benchmarks)
   - [src/warmup/](#210-srcwarmup--pre-benchmark-warm-up)
   - [src/utils/](#211-srcutils--shared-utilities)
   - [src/third_party/](#212-srcthird_party--vendored-dependencies)
3. [tests/ — Test suite](#3-tests--test-suite)
4. [results/ — Result data](#4-results--result-data)
5. [pictures/ — Documentation images](#5-pictures--documentation-images)
6. [script-examples/ — Run and plotting helpers](#6-script-examples--run-and-plotting-helpers)
7. [.github/ — GitHub integration](#7-github--github-integration)

---

## 1. Root-level files

### Entry point

| File | Purpose |
|---|---|
| `main.cpp` | Program entry point; handles bare zero-argument help early before timer creation/mode scan, wraps mode selection and argument parsing in exception boundaries, dispatches standalone CPU bandwidth (`-B`), CPU latency (`-L`), patterns (`-P`), LLM (`-M`), GPU (`-G`), and core-to-core (`-C`) paths before the general parser, rejects modifier-only invocations with no primary mode before output-session setup, and runs the TLB (`-T`) and TLB sweep pipelines; each standalone command boundary owns its output-session lifetime |

### Build and tooling

| File | Purpose |
|---|---|
| `Makefile` | Primary build system; discovers C++/Objective-C++/assembly, targets sources and final links at macOS 26.0, accepts a configurable `GTEST_DIR`, compiles `.mm` with ARC, links Metal/Foundation, and produces release/test binaries |
| `coverage.sh` | Runs isolated LLVM unit/all-test C++/Objective-C++ source coverage builds under `/tmp` without replacing normal workspace binaries |
| `.clang-format` | Clang-Format style baseline for C++ sources |
| `build-support/generate_provenance.py` | Python 3 build-time generator for `.build-provenance.h`; freezes Git, compiler, flags, target, SDK, and deployment-target metadata and updates the generated header only when inputs change |

### Root-level documentation

| File | Purpose |
|---|---|
| `README.md` | Project overview, quick-start instructions, and feature summary |
| `CHANGELOG.md` | Version history and release notes |
| `CITATION.cff` | Standardized Citation File Format (CFF) metadata for citing the software release |
| `CONTRIBUTING.md` | Contribution guidelines, coding standards, and pull request process |
| `CODE_OF_CONDUCT.md` | Community standards |
| `SECURITY.md` | Vulnerability disclosure policy |
| `LICENSE` | GNU General Public License v3.0 license text; source headers grant version 3 or any later version |

### documents/ — Project documentation

| File | Purpose |
|---|---|
| `documents/API.md` | Supported process-level machine interface: output targets, stream separation, current schema identities, and result acceptance |
| `documents/CAPABILITIES.md` | Measurement capability overview and interpretation notes |
| `documents/MANUAL.md` | Complete user manual: all CLI flags, modes, output formats, and usage examples |
| `documents/PARAMETER_MATRIX.md` | Mode/flag compatibility, sweep support, and incompatible-option matrix |
| `documents/TECHNICAL_SPECIFICATION.md` | Internal architecture, data structures, and implementation decisions |
| `documents/CPU_BANDWIDTH_WHITEPAPER.md` | Whitepaper: audited standalone CPU bandwidth methodology (`-B`/`--cpu-bandwidth`), O(1) GF(2) read checksums, full-buffer validation, schema 1, and independent verifier |
| `documents/LATENCY_WHITEPAPER.md` | Whitepaper: audited standalone CPU pointer-chase latency methodology (`-L`/`--cpu-latency`), canonical chain hashing, sample windows, `--compare-locality`, schema 1, and independent verifier |
| `documents/PATTERNS_WHITEPAPER.md` | Whitepaper: audited standalone CPU access-patterns methodology (`-P`/`--patterns`), slot-tagged GF(2) read checksums, 2D partial-cycle strided oracles, cold touched/untouched-sentinel post-validation, schema 1, and independent verifier |
| `documents/TLB_ANALYSIS_WHITEPAPER.md` | Whitepaper: current TLB analysis methodology, schema, and interpretation limits |
| `documents/CORE_TO_CORE_WHITEPAPER.md` | Whitepaper: calibrated two-thread token-handoff methodology, audit schema, and interpretation limits |
| `documents/GPU_BANDWIDTH_WHITEPAPER.md` | Whitepaper: Metal compute bandwidth methodology, GPU schema 1, validation, capability limits, and maintenance policy |
| `documents/LLM_MEMORY_PROFILE_WHITEPAPER.md` | Whitepaper: CPU and capability-gated Metal decode/prefill contiguous/paged methodologies, with exact payload formulas, resources, validation, schema 2, checksum evidence limits, and interpretation limits |
| `documents/PROJECT_STRUCTURE.md` | This file |

---

## 2. src/ — Source code

Production C++, Objective-C++, and ARM64 assembly lives under `src/`, except for the root entry point `main.cpp`.
Headers use include paths relative to `src/`
(e.g., `#include "core/config/config.h"`). Metal/Objective-C types are confined to private `.mm` platform boundaries.

---

### 2.1 src/asm/ — ARM64 assembly kernels

Hand-written AArch64 assembly implementing the hot inner loops that must not be rewritten by the compiler. Each `.s` file corresponds to one access pattern or operation type. CPU bandwidth, cache, latency, pattern, and core-to-core `extern "C"` declarations live in `asm_functions.h`, while the four LLM assembly entrypoints are declared in `src/llm_memory/llm_executor.h`.

| File | Operation |
|---|---|
| `asm_functions.h` | `extern "C"` declarations for bandwidth, cache, latency, pattern, and core-to-core assembly functions |
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
| `memory_write_pattern.s` | Sequential forward 64-bit arbitrary-pattern memory write (`stnp` 512B main loop + little-endian scalar tail for `--cpu-bandwidth` and `--patterns`) |
| `memory_write_cache_pattern.s` | Sequential forward 64-bit arbitrary-pattern memory write (`stp` 512B cache loop + little-endian scalar tail for `--cpu-bandwidth`) |
| `memory_write_random_pattern.s` | Random-order 64-bit arbitrary-pattern memory write (`stnp` 32B stores for `--patterns`) |
| `memory_write_reverse_pattern.s` | Reverse-order 64-bit arbitrary-pattern memory write (`stnp` 512B main loop + aligned reverse byte tail for `--patterns`) |
| `memory_write_strided_pattern.s` | Phase-rotating 64-bit arbitrary-pattern strided memory write (`stnp` 32B phased stores for `--patterns`) |
| `memory_latency.s` | Pointer-chase latency measurement loop |
| `core_to_core_latency.s` | Acquire/release token-exchange ping-pong loop for core-to-core protocol latency |
| `llm_decode_memory.s` | Descriptor-driven weight/KV read and temporal current-token K/V append kernel for the synthetic LLM profile |
| `llm_decode_memory_paged.s` | Paged-KV decode kernel with explicit timed block-table loads, paired append lookup, separate K/V scan lookups, exact tails, and versioned checksum evidence |
| `llm_prefill_memory.s` | Contiguous-KV prefill kernel with full-prompt writes, tiled causal-prefix K/V scans, and operation-ordinal checksum evidence |
| `llm_prefill_memory_paged.s` | Paged-KV prefill kernel with block-exclusive prompt writes, exact partial-prefix K/V scans, timed uint32 table loads, and versioned checksum evidence |

---

### 2.2 src/benchmark/ — Core infrastructure

The benchmark subsystem owns shared latency test helpers, standalone TLB analysis, core-to-core analysis, and shared sweep execution. Standalone CPU bandwidth, CPU latency, pattern, GPU, and LLM pipelines live in their dedicated subtrees, while reusable statistics and output helpers live under `src/utils/` and `src/output/`.

#### Shared kernel dispatch and execution infrastructure

| File | Purpose |
|---|---|
| `latency_tests.cpp` | Pointer-chase kernel dispatch and sample-window execution helpers |
| `benchmark_tests.h` | Shared declarations for latency test functions and `LatencyMeasurementTestHooks` |
| `sweep_runner.h` / `.cpp` | Shared deterministic sweep executor, completion classification (`SweepNestedMode::{Patterns,TlbAnalysis,CoreToCore,LlmMemory,CpuBandwidth,CpuLatency}`), retained terminal envelopes, and file/lazy-stdout checkpoint dispatch; provides TLB entry-point orchestration and is called directly by the CPU bandwidth, CPU latency, patterns, core-to-core, and LLM sweep runners |

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
| `core_to_core_latency_cli.cpp` | Descriptor-driven CLI argument parsing (`cli_parser.h`), unified duplicate-sweep and direct-vs-sweep conflict validation, exception containment, direct/sweep output-session ownership, and entry point for the core-to-core mode |
| `core_to_core_latency_json.h` / `.cpp` | Serializes schema-2 work plans, loop audit records, completion state, and results |
| `core_to_core_sweep_runner.h` / `.cpp` | Core-to-core Cartesian sweeps, retained terminal envelopes, atomic file checkpoints, and lazy stdout checkpoints |

---

### 2.3 src/core/ — Core utilities

Core infrastructure for shared audit provenance/RNG/checkpoints, configuration, memory management, macOS system introspection, signal handling, and Mach timing.

#### src/core/audit/

| File | Purpose |
|---|---|
| `build_manifest.h` / `.cpp` | Builds versioned runtime provenance manifests (`build_runtime_provenance_manifest_json`) from embedded build metadata and `binary_sha256` |
| `checkpoint_policy.h` | Shared bounded loop checkpoint cadence (`K=max(1,ceil(N/8))`) and `checkpoint_lifecycle` JSON builder used by CPU bandwidth, CPU latency, patterns, and LLM modes |
| `deterministic_rng.h` | Shared `splitmix64_next`, `splitmix64_mix`, and descending `splitmix64-fisher-yates-rejection-v1` permutation helpers |
| `host_environment.h` / `.mm` | Objective-C++ `NSProcessInfo` thermal/Low Power Mode/physical-memory snapshot (`capture_host_environment()`) and streaming executable SHA-256 (`capture_binary_sha256()`) |

#### src/core/config/

| File | Purpose |
|---|---|
| `config.h` | `BenchmarkConfig` structure for standalone TLB and its common sweep settings; CPU bandwidth, CPU latency, patterns, core-to-core, GPU, and LLM use separate config types |
| `constants.h` | Named constants for CPU/GPU/LLM memory limits, calibration, grid/dispatch/payload guardrails, and pattern/TLB parameters |
| `version.h` | `SOFTVERSION` macro containing the semantic software-release provenance string |
| `cli_parser.h` / `.cpp` | Shared descriptor-driven CLI option parser (`parse_cli_options`), centralized error formatter (`format_cli_parse_error`), `cli_program_name`, `extract_long_option_name`, `is_known_cli_value_option`, and LP64-safe `parse_strict_nonnegative_ulong` shared by all seven primary modes |
| `mode_selector.h` / `.cpp` | Pure primary-mode scan and conflict detection before mode-specific parsing; routes all seven primary modes (`CpuBandwidth`, `CpuLatency`, `Patterns`, `AnalyzeTlb`, `AnalyzeCoreToCore`, `GpuBandwidth`, `LlmMemory`, including `--mode=value`) while skipping one opaque value token for every known space-separated value-taking option across all seven modes |
| `argument_parser.cpp` | Single-pass descriptor-driven parser for standalone TLB (`-T`/`--analyze-tlb`) and general help/no-mode fallback into `BenchmarkConfig` (resets config on entry, supports `--option=value`, accepts `-h`/`--help` in TLB mode, defers OS queries until after syntactic parsing/help, and honors `ConfigTestHooks` including `get_config_page_size_bytes()`); CPU bandwidth, CPU latency, patterns, core-to-core, GPU, and LLM are pre-routed to dedicated parsers |
| `config_validator.cpp` | Validates parsed TLB and missing-mode configurations; enforces required primary mode, direct-vs-`--sweep` conflict checks, and TLB memory limits |
| `sweep_utils.h` / `.cpp` | Shared structural sweep parsing (using centralized `Messages::` reasons) and overflow-safe Cartesian run counting used by CPU bandwidth, CPU latency, patterns, TLB, core-to-core, and LLM sweep parsers and runners |

#### src/core/signal/

| File | Purpose |
|---|---|
| `signal_handler.h` / `.cpp` | Installs SIGINT/SIGTERM handling, exposes an exact-mask-restoring RAII guard for worker lifetimes, and reports pending interruption between phases |

#### src/core/memory/

| File | Purpose |
|---|---|
| `memory_manager.h` / `.cpp` | Low-level normal and cache-discouraging `mmap` allocation, `AllocatedBufferEvidence` (`madvise`/`mprotect` errno/status), `allocate_buffer_with_evidence`, `protect_buffer_read_only_with_evidence`, RAII `MmapPtr` owners, and a test-only mmap-family provider seam |
| `memory_utils.h` / `.cpp` | Latency-chain mode helpers and deterministic/unseeded pointer-chasing latency-chain construction |

#### src/core/system/

| File | Purpose |
|---|---|
| `system_info.h` / `.cpp` | Queries CPU/cache/OS/memory data through a macOS provider and a deterministic provider seam for fallback/error tests |
| `benchmark_qos.h` / `.cpp` | Shared best-effort main-thread QoS preparation with requested/applied/code audit result |
| `page_size.h` / `.cpp` | Shared native page-size query boundary for accounting, validation, metadata, and page-prefault operations |

#### src/core/timing/

| File | Purpose |
|---|---|
| `timer.h` / `.cpp` | High-resolution Mach timer with tick/timebase conversion, retained original start/stop/delta `MachTimingSnapshot` records, and a deterministic clock/timebase provider seam |

---

### 2.4 src/cpu_bandwidth/ — Standalone CPU bandwidth mode

Standalone `-B`/`--cpu-bandwidth` schema-1 implementation (`mode: "cpu_bandwidth"`, `methodology_version: "cpu-bandwidth-v1-arm64-calibrated-seeded-balanced"`), completely decoupled from legacy `BenchmarkConfig`.

| File | Purpose |
|---|---|
| `cpu_bandwidth.h` / `.cpp` | Dedicated `CpuBandwidthConfig`, exact 12-option whitelist parser (`cli_parser.h`), target-omission/sweep validation, QoS/signal scope, output-session ownership, and command boundary (`run_cpu_bandwidth_mode`) |
| `cpu_bandwidth_work_plan.h` / `.cpp` | Target activation (`main_memory`, auto `l1`/`l2`, `custom`), 64-byte worker slice partitioning (`partition_cpu_bandwidth_64b`), SplitMix64 seed derivation, calibration arithmetic, strict `memory_budget` preflight, and `cpu-bandwidth-work-plan-v1` identity |
| `cpu_bandwidth_oracle.h` / `.cpp` | O(1) GF(2) `src` buffer generator/oracle (`cpu-bw-buffer-pattern-v1`), non-cancelling per-pass read checksum oracle (`cpu-bw-read-checksum-v1`), worker checksum fold, and unique odd `write_pattern` / `pre_copy_sentinel` derivations |
| `cpu_bandwidth_executor.h` / `.cpp` | Persistent `src`/`dst` mappings, `mprotect(PROT_READ)` on `src`, synchronized worker team barrier + `MachTimingSnapshot`, `memory_write{_cache}_pattern_loop_asm` dispatch, and cold parallel full-buffer `write`/`copy` post-validation (`validation.checks[]`) |
| `cpu_bandwidth_runner.h` / `.cpp` | Excluded calibration + atomic plan freeze, cyclic Latin-square target/operation schedule, `operation_order_balance_complete` / `target_order_balance_complete`, `K=max(1,ceil(N/8))` bounded checkpoints, and descriptive statistics |
| `cpu_bandwidth_json.h` / `.cpp` | Schema-1 `cpu_bandwidth` JSON builder (`build_cpu_bandwidth_json`), conservative output-peak estimator, and quality warning collector |
| `cpu_bandwidth_output.h` / `.cpp` | `-B`/`--cpu-bandwidth` CLI help, configuration banner, and human-readable console summary (`P50`, `min`, `max`, `CV`, `MAD`, `n`) |
| `cpu_bandwidth_sweep_runner.h` / `.cpp` | Cartesian sweeps over `buffer-size`, `cache-size`, and `threads` using the schema-1 sweep envelope (`SweepNestedMode::CpuBandwidth`) |

---

### 2.5 src/cpu_latency/ — Standalone CPU latency mode

Standalone `-L`/`--cpu-latency` schema-1 implementation (`mode: "cpu_latency"`, `methodology_version: "cpu-latency-v1-arm64-calibrated-seeded-balanced"`), completely decoupled from legacy `BenchmarkConfig`.

| File | Purpose |
|---|---|
| `cpu_latency.h` / `.cpp` | Dedicated `CpuLatencyConfig`, exact 16-option whitelist parser (`cli_parser.h`), stride/locality/`--compare-locality`/sweep validation, QoS/signal scope, output-session ownership, and command boundary (`run_cpu_latency_mode`) |
| `cpu_latency_work_plan.h` / `.cpp` | Target activation (`l1`, `l2`, `custom`, `main_memory`), `--compare-locality` eligibility, SplitMix64 domain seeds, calibration arithmetic, strict `memory_budget` preflight, and `cpu-latency-work-plan-v1` identity |
| `cpu_latency_chain.h` / `.cpp` | `splitmix64-fisher-yates-rejection-v1` pointer-chain generator (`auto`, `global-random`, `random-box`, `same-random-in-box`, `diff-random-in-box`), canonical `indices[0] == 0` rotation, `chain_permutation_sha256`, and single Hamiltonian-cycle prevalidation |
| `cpu_latency_executor.h` / `.cpp` | Buffer mapping, chain population + `mprotect(PROT_READ)`, `memory_latency_chase_asm` execution with `MachTimingSnapshot`, `main-final-pointer` verification, and separate continuous sample-window pass with per-window end-pointer checks (`validation.checks[]`) |
| `cpu_latency_runner.h` / `.cpp` | Excluded calibration + atomic plan freeze, optional 3-round/loop paired `--compare-locality` (`locality_16k` vs `locality_global`), cyclic target rotation, `pooled_sample_distribution`, `K=max(1,ceil(N/8))` bounded checkpoints, and descriptive statistics |
| `cpu_latency_json.h` / `.cpp` | Schema-1 `cpu_latency` JSON builder (`build_cpu_latency_json`), conservative output-peak estimator, and quality warning collector |
| `cpu_latency_output.h` / `.cpp` | `-L`/`--cpu-latency` CLI help, configuration banner, and human-readable console summary (`P50`, `min`, `max`, `CV`, `MAD`, `n`, `PooledSamples`, `Locality Comparison`) |
| `cpu_latency_sweep_runner.h` / `.cpp` | Cartesian sweeps over `buffer-size`, `cache-size`, `latency-tlb-locality-kb`, `latency-stride-bytes`, and `latency-chain-mode` using the schema-1 sweep envelope (`SweepNestedMode::CpuLatency`) |

---

### 2.6 src/output/ — Output layer

This subsystem contains shared console messages, shared CLI help/banner formatters, descriptive-statistics rendering, and JSON serialization for TLB and shared session output. Mode-specific output orchestration resides inside each standalone mode (`src/cpu_bandwidth/`, `src/cpu_latency/`, `src/pattern_benchmark/`, `src/gpu_bandwidth/`, `src/llm_memory/`), while user-facing message text is centralized under `messages/`.

#### src/output/console/

| File | Purpose |
|---|---|
| `output_printer.h` / `.cpp` | Prints general CLI help and shared runtime banner text |
| `statistics_renderer.h` / `.cpp` | Renders descriptive-statistics summaries in the canonical field order with shared precision, indentation, and diagnostic controls |

##### src/output/console/messages/

All user-facing text strings are centralized here. Each `.cpp` file implements a specific category of messages declared in the shared API header.

| File | Purpose |
|---|---|
| `messages_api.h` | Declares the `Messages` namespace containing all string-returning functions |
| `config_messages.cpp` | Runtime banner header, copyright, and license text |
| `core_to_core_messages.cpp` | Core-to-core mode status and result messages |
| `gpu_bandwidth_messages.cpp` | GPU help, status, result, interpretation, warning, and validation messages |
| `llm_memory_messages.cpp` | LLM help, validation, lifecycle, headline, interpretation, and quality-warning messages |
| `error_messages.cpp` | Fatal error messages |
| `program_messages.cpp` | Program banner, usage, and version text |
| `statistics_messages.cpp` | Statistical output labels (e.g., `"Average:"`, `"Median (P50):"`, `"Stddev:"`) |
| `warning_messages.cpp` | Non-fatal warning messages |

#### src/output/json/json_output/

| File | Purpose |
|---|---|
| `json_output_api.h` | Shared `JsonKeys` constants and atomic-file writer API; this C++ API is distinct from the process contract in `API.md` |
| `json_output_session.h` / `.cpp` | Classifies raw output targets, applies mode-specific file-path policy, lazily dispatches checkpoints, routes command-scoped human stdout, and emits checked final JSON through the retained original stdout buffer; all result-producing direct modes and supported sweeps use it |
| `file_writer.cpp` | Atomically writes the completed JSON document to the requested output path, creating parent directories as needed |

---

### 2.7 src/gpu_bandwidth/ — Metal GPU bandwidth

Standalone GPU schema 1 implementation. The pure C++ planner/runner/result model is isolated from the private
Objective-C++ Metal backend so deterministic unit tests do not require GPU work.

| File | Purpose |
|---|---|
| `gpu_bandwidth.h` / `.cpp` | Dedicated `GpuBandwidthConfig`, shared descriptor-driven exact option-whitelist parser (`cli_parser.h`) with LP64-safe `--buffer-size` range handling, GPU help, standalone output-session boundary, QoS/signal scope, console handoff, and terminal stdout emission |
| `gpu_work_plan.h` / `.cpp` | Pure read/write/copy pass limits, exact payload, seed derivation, cyclic order, calibration arithmetic, frozen 8192-threadgroup grid cap/geometry, and `gpu-work-plan-v1` identity |
| `gpu_backend.h` / `.cpp` | Objective-C-free synchronous/noexcept backend contract, device/resource/phase/validation metadata, factory declaration, and status string mappings |
| `gpu_runner.h` / `.cpp` | Backend-independent calibration, warmup/precondition/timing/validation orchestration, completion-wins interruption, counters, aggregates, resource lifecycle, and transport-neutral logical checkpoints with a file-scoped post-release replacement |
| `gpu_json.h` / `.cpp` | GPU schema 1 builder and shared atomic-writer adapter; the prebuilt payload can also use the command's stdout session and preserves exact integer strings, nullable state, errors, provenance, and nested audit records |
| `gpu_kernels_source.h` | Private canonical embedded MSL 2.3 source, kernel revision, integer pattern/checksum contract, and exact source bytes hashed at runtime |
| `metal_gpu_backend.mm` | Only Objective-C++/Metal boundary: ARC/autorelease pools, capability checks, runtime compilation, private/tracked buffers, shared/tracked status, serial command buffers, timestamps, validation, and test readback |

---

### 2.8 src/llm_memory/ — Synthetic LLM memory profile

Standalone schema-2 implementation with eight active backend/phase/layout profiles: four CPU profiles and four
capability-gated Metal profiles spanning decode/prefill and contiguous/paged KV. Pure logical phase planning,
deterministic paged geometry/permutation, backend-specific execution planning/evidence, the Objective-C-free backend
contract, CPU mapping and ARM64 execution, the Objective-C++ Metal boundary, environment capture, console composition,
and serialization are separated so deterministic unit tests can inject platform and allocation behavior without running
hot kernels. Capability failures remain explicit, and no unsupported Metal request falls back to CPU.

| File | Purpose |
|---|---|
| `llm_memory.h` / `.cpp` | Separate config/status foundation, shared descriptor-driven exact-whitelist parser (`cli_parser.h`, including repeatable `--sweep` dimensions, conflict/combination preflight, and `--sweep-max-runs`), required/default worker/seed resolution, complete command boundary, direct and sweep dispatch (`run_llm_memory_collect` / `run_llm_memory_sweep`), backend factory ownership, peak-memory admission, QoS/signal scope, console output, and file/stdout transport orchestration |
| `llm_kv_layout.h` / `.cpp` | Pure checked paged-KV geometry, block ownership, SplitMix64/Fisher–Yates permutation and little-endian table hash, exact physical/logical/padding/table/lookup/accounting math, and transient preparation estimates |
| `llm_prefill.h` / `.cpp` | Checked prefill tile, causal-pair, payload, paged floor-sum lookup, token/block cost partition, versioned ownership evidence, write/checksum oracle, and semantic event-trace planning |
| `llm_work_plan.h` / `.cpp` | Checked phase-applicable weight/KV geometry, payload/metadata/accounted math, memory budget, scenario limits, exact CPU ownership evidence, Metal segment/argument-buffer/resource/grid plans, and frozen methodology/component/plan identities |
| `llm_backend.h` / `.cpp` | Objective-C-free synchronous backend lifecycle and task boundary, generic task identity/timing/completion/validation, tagged CPU/Metal task and command evidence, auxiliary-memory contract, stable statuses/reasons, checked evidence accessors, and backend factory |
| `llm_cpu_backend.h` / `.cpp` | Active CPU adapter that owns the timer and prepared contiguous or paged resources, delegates allocation/execution to the executor, validates CPU worker/QoS/timer/checksum invariants, and converts executor evidence into the generic task result |
| `llm_metal_backend.h` | Objective-C-free planning, capability, contiguous/paged resource and memory admission, cyclic block-owner grids with per-threadgroup accounting, decode/prefill parameter ABIs, timestamp/checksum mapping, diagnostics, and active-profile test seams |
| `llm_metal_backend.mm` | Objective-C++ Metal boundary for capability probing, selected-profile runtime compilation, Tier 2 argument encoding, private/shared allocation, table upload/validation, exact-tail dispatch, paged-prefill cyclic ownership, GPU timestamps, layout-aware dual-mod32 checksum, full-prompt K/V-write and padding validation, diagnostics, and idempotent cleanup |
| `llm_metal_kernels_source.h` | Canonical embedded MSL 2.3 source with profile-specific decode/prefill contiguous/paged parameter ABIs, common foundation entrypoints, scenario-specialized workload entrypoints, named-lane volatile paged-table publication, and phase-specific K/V-write/padding validation; exact selected source bytes are hashed at runtime |
| `llm_executor.h` / `.cpp` | CPU-specific full-size mappings, paged-table preparation, deterministic initialization/pre-touch, phase/layout descriptor materialization, independent checksum oracles, padding canaries, synchronized worker team, timer boundary, and ARM64 adapters retained behind `LlmCpuBackend` |
| `llm_runner.h` / `.cpp` | Backend-independent lifecycle, per-scenario calibration or exact-work planning, canonical/frozen plans, named-check task acceptance, cyclic position balance, status/counters, interruption, accepted measurement populations, snapshot-time statistics, and bounded loop/terminal checkpoints; correctness acceptance is separate from comparison quality, and resources are released before the command-terminal checkpoint |
| `llm_sweep_runner.h` / `.cpp` | Standalone LLM Cartesian sweeps over `context-tokens`, `kv-block-tokens`, `prompt-tokens`, and `attention-query-tile-tokens` using the shared schema-1 sweep envelope with per-run resource cleanup, atomic file checkpoints, and lazy stdout checkpoints |
| `llm_json.h` / `.cpp` | Ordered LLM schema-2 builder and conservative output-peak estimator (including nested `collect_json_for_sweep` runs); projects canonical plans and document-local references, expected versus observed checksums, named validation checks, accepted measurement IDs, raw timing, build provenance, backend/resource evidence, nullable states, and checkpoint writer observations |
| `llm_validation.h` | Fixed, allocation-free structure/write/padding observation slots with independent applicability, evaluation, validity, and stable reasons for existing cold validation walks |
| `llm_output.h` / `.cpp` | Human-readable work-unit/model-payload/accounted geometry, Metal device/resource/task evidence, scenario headlines, interpretation limits, and evidence-backed quality warnings through centralized message helpers |
| `llm_environment.h` | Thin compatibility forwarding header delegating host environment and executable SHA-256 capture to `src/core/audit/host_environment.h` |

---

### 2.9 src/pattern_benchmark/ — Standalone patterns mode

Standalone `-P`/`--patterns` schema-1 implementation (`mode: "patterns"`, `methodology_version: "patterns-v1-arm64-phase-calibrated-seeded-balanced"`), completely decoupled from legacy `BenchmarkConfig`.

| File | Purpose |
|---|---|
| `patterns.h` / `.cpp` | Dedicated `PatternsConfig`, exact 11-option whitelist parser (`cli_parser.h`), sweep validation, QoS/signal scope, output-session ownership, and command boundary (`run_patterns_mode`, `run_patterns_collect`) |
| `patterns_work_plan.h` / `.cpp` | 7 canonical patterns, 3 operations, 64-byte worker slice partitioning (`partition_patterns_64b`), strided worker reduction & `strided_2mb` skip (`buffer-too-small-for-stride-transition`), random clamping/split, `SplitMix64` seed derivation, strict `memory_budget` preflight, and `patterns-work-plan-v1` identity |
| `patterns_oracle.h` / `.cpp` | Slot-tagged GF(2) `src` buffer generator/oracle (`patterns-buffer-pattern-v1`), O(1) sequential & 2D partial-cycle strided `patterns-read-checksum-v1` oracle, `splitmix64-coprime-fisher-yates-v1` + `random_indices_sha256`, unique odd `write_pattern`/`pre_op_sentinel`, and parallel cold 100% touched/untouched-sentinel post-validation |
| `patterns_executor.h` / `.cpp` | Persistent `src`/`dst` mappings, `mprotect(PROT_READ)` on `src`, synchronized worker team barrier + `MachTimingSnapshot`, 7×3 ARM64 kernel dispatch (including `memory_write_{reverse,strided_phased,random}_pattern_loop_asm`), and 6-entry `validation.checks[]` (`PatternsExecutorTestControl`) |
| `patterns_runner.h` / `.cpp` | Excluded calibration + atomic plan freeze (`frozen_plan_refs` / `frozen_measurement_warmup`), 2-axis cyclic Latin-square pattern/operation schedule, `operation_order_balance_complete` / `pattern_order_balance_complete`, `K=max(1,ceil(N/8))` bounded checkpoints, task-boundary `SIGINT`/`SIGTERM`, and descriptive statistics (`5.0%` streaming / `10.0%` sparse CV) |
| `patterns_json.h` / `.cpp` | Schema-1 `patterns` JSON builder (`build_patterns_json`, 26 top-level keys), conservative DOM peak estimator, and quality warning collector |
| `patterns_sweep_runner.h` / `.cpp` | Cartesian sweeps over `buffer-size` and `threads` using the schema-1 sweep envelope (`SweepNestedMode::Patterns`) |
| `patterns_output.h` / `.cpp` | `-P`/`--patterns` CLI help, configuration banner, and human-readable console summary (`Patterns Mode (schema v1)`, `P50`, `% vs forward`, `CV`, `MAD`, `n`) |

---

### 2.10 src/warmup/ — Pre-benchmark warm-up

Warm-up passes reduce selected cold-start effects before timing. Standalone TLB analysis (`-T`) uses single-threaded latency page-prefaulting without pre-traversing the pointer chain.

| File | Purpose |
|---|---|
| `warmup.h` | Public warm-up API (`warmup_latency`) |
| `latency_warmup.cpp` | Single-threaded page-touch warm-up for latency buffers; faults pages in without traversing the pointer chain |

---

### 2.11 src/utils/ — Shared utilities

| File | Purpose |
|---|---|
| `utils.h` / `.cpp` | Shared thread-joining helpers and the TTY-aware stderr progress spinner (`ProgressSpinner`) used by benchmark modes |
| `json_utils.h` / `.cpp` | JSON helper functions shared across `cpu_bandwidth`, `cpu_latency`, `patterns`, `llm_memory`, `gpu_bandwidth`, TLB, and core-to-core output serializers |
| `cyclic_order.h` / `.cpp` | Shared deterministic cyclic ordering used by CPU and GPU planners |
| `seed_utils.h` / `.cpp` | Shared SplitMix64 derivation and generate-once seed helper with deterministic provider seam |
| `hash_utils.h` / `.cpp` | CommonCrypto-based one-shot, incremental, and allocation-free SHA-256 helpers for MSL source, layout/identity evidence, and executable-file provenance |
| `numeric_utils.h` / `.cpp` | Overflow-safe size arithmetic plus bounded pilot-count and duration-calibration helpers |
| `descriptive_statistics.h` / `.cpp` | Canonical average, percentile, sample-deviation, coefficient-of-variation, and median-absolute-deviation calculations |

---

### 2.12 src/third_party/ — Vendored dependencies

| File | Purpose |
|---|---|
| `nlohmann/json.hpp` | nlohmann/json single-header library (MIT license); used throughout the JSON output subsystem |

---

## 3. tests/ — Test suite

GoogleTest-based unit and integration tests are supplemented by Python example, LLM verifier, CPU verifier, and E2E acceptance tests.
`make test-script-examples` exercises the current JSON-reading examples directly. `make test-llm-verifier` checks the
independent LLM verifier across schema-2 artifacts and schema-1 LLM sweep envelopes, numeric goldens, semantic mutations,
and build provenance. `make test-patterns-verifier` checks `tests/test_patterns_verifier.py`, and `make test-cpu-verifiers`
checks the independent CPU bandwidth, CPU latency, and patterns verifiers across schema-1 fixtures, O(1) vs brute-force
differential tests, all chain policies, 2D partial-cycle strided/random oracles, sweep envelopes, and semantic mutations.
`make test-all` runs all three Python gates (`test-script-examples`, `test-llm-verifier`, `test-cpu-verifiers`) after all
GTest cases pass, while `tests/test_cpu_e2e_acceptance.py` and `tests/test_patterns_e2e_acceptance.py` provide standalone
Tier 1–4 executable acceptance coverage.
Python 3 is required; `jq` is optional, and only its dedicated test branch is skipped when it is not installed. All test
`.cpp` and `.mm` files are picked up automatically by the Makefile. Tests named `*Integration*` are excluded from
`make test` (unit-only) and run through the integration or all-test targets.

| File | Suite name | Coverage focus |
|---|---|---|
| `test_audit_primitives.cpp` | `AuditPrimitivesTest` | Shared `src/core/audit/*` primitives: `SplitMix64` / `splitmix64-fisher-yates-rejection-v1`, bounded loop checkpoint cadence (`K=max(1,ceil(N/8))`), host environment, and `build_manifest` provenance |
| `test_cli_parser.cpp` | `CliParserTest` | Shared descriptor-driven CLI option parsing (`--option=value`, opaque values, arity, missing/unexpected/unknown errors, single-dash rejection), `cli_program_name`, `format_cli_parse_error`, `extract_long_option_name`, `is_known_cli_value_option`, and LP64-safe `parse_strict_nonnegative_ulong` |
| `test_config.cpp` | `ConfigTest` | Standalone TLB and general `BenchmarkConfig` descriptor-driven CLI/sweep parsing (`--option=value`, duplicate flags, `--analyze-tlb --help`, LP64 `--buffer-size`, config reset, TLB `ConfigTestHooks`), no-mode validation errors, direct+sweep conflicts, and defaults |
| `test_signal_handler.cpp` | `BenchmarkSignalMaskGuardTest` | Exact thread-mask restoration, caller-preserved blocked signals, and nested scope ownership |
| `test_messages.cpp` | `Messages*Test` | Exact help, CLI error/reason, and formatted console message contracts |
| `test_memory_utils.cpp` | `MemoryUtilsTest` | Memory helpers: pointer-chase chain construction and verification |
| `test_memory_manager.cpp` | `MemoryManagerTest` | Injected mmap/madvise/mprotect policy, `AllocatedBufferEvidence`, failures, and exact RAII unmapping |
| `test_numeric_utils.cpp` | `NumericUtilsTest` | Overflow-safe arithmetic, duration calibration, pilot counts, and quantization boundaries |
| `test_cpu_bandwidth_config.cpp` | `CpuBandwidthConfigTest` | Standalone `-B`/`--cpu-bandwidth` exact 12-option whitelist parsing, strict ranges (`-i 1..1000000000`, `-b 0` / `-k 0` selective omission), `--sweep` key validation/conflicts, and `-h` help isolation |
| `test_cpu_bandwidth_work_plan.cpp` | `CpuBandwidthWorkPlanTest` | Target activation (`main_memory`, `l1`, `l2`, `custom`), 64-byte worker slice partitioning, SplitMix64 domain/op seeds, calibration math, strict `memory_budget` preflight, and guardrails |
| `test_cpu_bandwidth_oracle.cpp` | `CpuBandwidthOracleTest`, `CpuBandwidthOracleIntegrationTest` | O(1) GF(2) `src` buffer pattern, O(1) vs byte/pass reference `pass_xor` and `cpu-bw-read-checksum-v1` differential checks, worker fold, and unique odd write/sentinel seeds |
| `test_cpu_bandwidth_executor_runner.cpp` | `CpuBandwidthExecutorRunnerTest`, `CpuBandwidthModeIntegrationTest` | Persistent `PROT_READ` `src` mappings, parallel team sync + `MachTimingSnapshot`, cold full-buffer `write`/`copy` validation (`validation.checks[]`), cyclic target/op order, bounded `K=max(1,ceil(N/8))` checkpoints, and `run_accepted` |
| `test_cpu_bandwidth_json_and_sweep.cpp` | `CpuBandwidthJsonAndSweepTest` | Schema-1 `cpu_bandwidth` JSON structure, decimal-string integers, `build_manifest`, `checkpoint_lifecycle`, and schema-1 `buffer-size`/`cache-size`/`threads` sweep envelopes |
| `test_cpu_latency_config.cpp` | `CpuLatencyConfigTest` | Standalone `-L`/`--cpu-latency` exact 16-option whitelist parsing, `-n 0`, `-i 1..1000000000`, stride/locality/`--compare-locality` rules, `--sweep` key validation/conflicts, and `-h` help isolation |
| `test_cpu_latency_work_plan.cpp` | `CpuLatencyWorkPlanTest` | Target activation (`l1`, `l2`, `custom`, `main_memory`), `--compare-locality` eligibility, SplitMix64 domain seeds, calibration math, and strict `memory_budget` preflight |
| `test_cpu_latency_chain.cpp` | `CpuLatencyChainTest` | All four chain modes (`global-random`, `random-box`, `same-random-in-box`, `diff-random-in-box`), canonical `indices[0] == 0` rotation, `chain_permutation_sha256`, and single Hamiltonian-cycle prevalidation |
| `test_cpu_latency_executor_runner.cpp` | `CpuLatencyExecutorRunnerTest`, `CpuLatencyModeIntegrationTest` | `mprotect(PROT_READ)` chain protection, `memory_latency_chase_asm` execution with `MachTimingSnapshot`, `main-final-pointer` and `sample-window-end-pointers`, paired `--compare-locality` rounds, `pooled_sample_distribution`, and `run_accepted` |
| `test_cpu_latency_json_and_sweep.cpp` | `CpuLatencyJsonAndSweepTest` | Schema-1 `cpu_latency` JSON structure, `locality_comparison` aggregates, `pooled_sample_distribution`, and schema-1 `buffer-size`/`cache-size`/`latency-tlb-locality-kb`/`latency-stride-bytes`/`latency-chain-mode` sweep envelopes |
| `test_llm_memory_config.cpp` | `LlmMemoryConfigTest`, `LlmMemoryConfigIntegrationTest` | Exact standalone decode/prefill parsing (including `--option=value`, paged prefill, phase/layout rules and stable reasons), strict decimal errors, raw output values, help isolation, worker/seed resolution, geometry/work-limit preflight, incompatible options, repeatable LLM `--sweep` parsing/expansion/conflicts/preflight, and `run_llm_memory_collect`/sweep runner behavior |
| `test_llm_memory_work_plan.cpp` | `LlmMemoryWorkPlanTest`, `LlmMemoryWorkPlanSystemCallsTest` | Production checked decode/prefill geometry and payloads, physical/padding/table/lookup/accounted math including paged-prefill `N+2*M`, deterministic permutation/hash, CPU token/block ownership, Metal cyclic block-owner scheduling and component identities, memory budget, descriptor/range/layout invariants, worker reduction, scenario caps/calibration, and cyclic order |
| `test_llm_memory_executor.cpp` | `LlmMemoryExecutorTest` | CPU phase/layout dispatch, atomic mapping/table-preparation failure seams, contiguous and paged decode/prefill descriptor materialization, independent checksums, padding canaries, synchronized timing, QoS, cancellation, and fake-kernel validation |
| `test_llm_memory_runner.cpp` | `LlmMemoryRunnerTest` | Fake-backend lifecycle and generic decode/prefill task seams, lifecycle unsupported/failure handling, common identity/timing/completion/validation acceptance, exact calibration/single-unit/freeze/frozen-warmup order, cyclic measurements, status/counter/aggregate semantics, interruption, release/checkpoint precedence, auxiliary budgeting, and runner exception boundaries |
| `test_llm_memory_json.cpp` | `LlmMemoryJsonTest` | Schema-2 identities for all eight profiles, canonical plan references, expected/actual checksums, named validation, accepted populations, raw Mach timing, build manifest, nullable geometry/workers, paged and Metal resource/grid evidence, output-peak estimates (including `collect_json_for_sweep`), acceptance, environment, and bounded checkpoint observations |
| `test_llm_metal_checksum.mm` | `LlmMetalChecksumHelperIntegrationTest` | Real-device checks of shared MSL affine checksum collision boundaries and six-component modulo reduction across partial SIMD/threadgroups |
| `test_llm_memory_output.cpp` | `LlmMemoryOutputTest` | Exact decode/prefill, contiguous/paged, and Metal device/resource/table/task K/V-write/lookup/padding plus cyclic owner/per-threadgroup cost formatting, interpretation text, and deduplicated warnings |
| `test_llm_memory_kernels.cpp` | `LlmMemoryKernelIntegrationTest` | Real contiguous/paged decode/prefill ARM64 descriptor/kernel scenarios, paged tails/lookups, full-prompt writes, exact partial tiled scans, checksum, padding, worker-count, and AAPCS64 coverage |
| `test_llm_metal_backend.cpp` | `LlmMetalBackendTest`, `LlmMetalBackendIntegrationTest`, `LlmMetalBackendFailureInjectionIntegrationTest` | Pure capability, contiguous/paged segmentation, argument-buffer, decode/prefill ABIs, source provenance, phase-aware checksum/reason mapping, paged-prefill `N+2*M`, cyclic no-duplication ownership and per-threadgroup accounted-byte evidence, plus real-device selected-profile compilation, Tier-2 slots, private table/K/V initialization, exact-tail decode, full-prompt/tiled-prefix prefill, final-ordinal write/padding detection, interruption/failure cleanup, and idempotent release |
| `test_gpu_bandwidth.cpp` | `GpuBandwidthParserTest`, `GpuMemoryBudgetTest`, `GpuRunnerTest`, `GpuJsonTest` | Strict standalone parsing (`--option=value`, LP64 `--buffer-size`, explicit offending-token diagnostics), memory budgets, fake-backend calibration/execution/failure/interruption and file/stdout checkpoint semantics, counters, and schema-1 serialization |
| `test_gpu_work_plan.cpp` | `GpuWorkPlanTest`, `GpuTimedAccumulatorOracleTest` | Cyclic order, seed domains, pass/payload caps, calibration, vector/tail/grid geometry, frozen identities, and timed-accumulator oracle behavior |
| `test_gpu_metal_backend.cpp` | `GpuMetalBackendIntegrationTest` | Real-Metal capability/runtime compile, private/shared tracked resources, read/write/copy/tail correctness, timestamps, validation, and byte readback |
| `test_mode_selector.cpp` | `ModeSelectorTest` | All seven primary modes (`-B`, `-L`, `-P`, `-T`, `-C`, `-G`, `-M`) and aliases, complete pairwise conflict order, `--mode=value` routing, null/`argc` safety, and opaque flag-shaped values skipped across all known space-separated value-taking options |
| `test_hash_utils.cpp` | `HashUtilsTest` | CommonCrypto SHA-256 standard vectors and source-provenance helper behavior |
| `test_analysis.cpp` | `AnalysisTest` | Injected TLB coordination, counters/status, boundary detection, validation, and paired analysis |
| `test_json_schema.cpp` | `JsonSchemaTest` | TLB schema-4, core-to-core schema-2, and sweep schema-1 output contracts |
| `test_cpu_bandwidth_verifier.py` | Independent CPU bandwidth verifier | O(1) vs brute-force differential goldens, schema-1 fixture verification, sweep envelope checks, `--binary` / `--require-raw-timing`, and semantic/checksum/timing mutations |
| `test_cpu_latency_verifier.py` | Independent CPU latency verifier | Chain-oracle goldens (all 4 policies + canonical rotation), schema-1 fixture verification, `--compare-locality`, `--max-chain-nodes`, sweep envelope checks, and semantic mutations |
| `test_cpu_e2e_acceptance.py` | Tier 1–4 CPU E2E acceptance | Live-binary Tier 1–4 acceptance tests for `-B`/`--cpu-bandwidth` and `-L`/`--cpu-latency` verified with `verify_cpu_bandwidth_result.py` and `verify_cpu_latency_result.py` |
| `test_patterns_config.cpp` | `PatternsConfigTest` | Standalone `-P`/`--patterns` exact 11-option whitelist parsing, strict bounds (`-b 1..2^53-1` rejecting `0`, `-i 1..10^9`, `-r`/`-X 1..2^53-1`, `-t 1..UINT32_MAX`), `--sweep` key validation/conflicts, and `-h` help isolation |
| `test_patterns_work_plan.cpp` | `PatternsWorkPlanTest` | 7 canonical patterns, 64B partitioning, strided worker reduction & `strided_2mb` skip (`buffer-too-small-for-stride-transition`), 2D phase rotation metrics, `random` clamping/split, `SplitMix64` seeds, strict `memory_budget`, and guardrails |
| `test_patterns_oracle.cpp` | `PatternsOracleTest`, `PatternsOracleIntegrationTest` | Slot-tagged GF(2) `patterns-buffer-pattern-v1`, O(1) vs brute-force sequential & 2D partial-cycle strided `patterns-read-checksum-v1`, `splitmix64-coprime-fisher-yates-v1` + `random_indices_sha256`, `base_seed = 424242` golden vectors, and parallel cold 100% touched/untouched-sentinel validation |
| `test_patterns_executor_runner.cpp` | `PatternsExecutorRunnerTest`, `PatternsExecutorRunnerIntegrationTest` | Persistent `PROT_READ` `src`, synchronized worker gate + Mach `cpu_raw`, 6-entry `validation.checks[]`, `PatternsExecutorTestControl` fault/corruption injection, excluded calibration/freeze, 2-axis Latin-square rotation, bounded `K=max(1,ceil(N/8))` checkpoints, and `SIGINT`/`SIGTERM` semantics |
| `test_patterns_json_and_sweep.cpp` | `PatternsJsonAndSweepTest`, `PatternsModeIntegrationTest` | Schema-1 `patterns` 26-key JSON contract, `!work_plan.valid` unavailable normalization, `build_manifest`, `checkpoint_lifecycle`, and schema-1 `buffer-size`/`threads` sweep envelopes |
| `test_patterns_verifier.py` | Independent patterns verifier | `base_seed = 424242` numeric goldens, O(1) vs brute-force differential tests, schema-1 fixtures (`complete` & `skipped-2mb`), sweep envelope checks, `--binary` / `--require-raw-timing` / `--max-random-indices`, and 86 mutation/rejection assertions |
| `test_patterns_e2e_acceptance.py` | Tier 1–4 patterns E2E acceptance | Live-binary Tier 1–4 acceptance tests (99 tests) for `-P`/`--patterns` covering CLI, skipped `strided_2mb`, cold validation, signal/checkpoint behavior, and sweeps verified with `verify_patterns_result.py` |
| `test_llm_result_verifier.py` | Independent LLM verifier | Eight current schema-2 profiles, schema-1 LLM sweep envelope verification and mutations, arithmetic goldens, status/population and semantic mutations, bounded input, and build provenance |
| `test_script_examples.py` | Python script-example entry test | Compatible `cpu_bandwidth` / `cpu_latency` schema-1 inputs, `run_accepted` and provenance-version handling, derived `--compare-locality` plotting tests, and metric extraction for the bundled `script-examples/` workflows, including the optional jq-backed shell path when jq is installed |
| `test_json_output_session.cpp` | `JsonOutputTargetTest`, `JsonOutputSessionTest` | Exact sentinel/path classification, lazy checkpoint dispatch, atomic-file parity, stdout routing/restoration, and stream failure containment |
| `test_json_utils.cpp` | `JsonUtilsTest`, `JsonFileWriterTest` | UTC timestamp/statistics and atomic writer success/failure contracts |
| `test_sweep_runner.cpp` | `SweepRunnerTest` | Current-only nested `cpu_bandwidth`, `cpu_latency`, patterns schema-1, TLB, core-to-core, and LLM schema-2 completion classification, pre-init failure and preparation-interruption handling, complete/partial/interrupted/failed attempt accounting, schema-1 envelopes, and checkpoint cadence/failure handling |
| `test_sweep_utils.cpp` | `SweepUtilsTest` | Shared sweep parsing, empty-dimension behavior, and overflow-safe Cartesian counts |
| `test_pattern_write_kernels.cpp` | `PatternWriteKernelTest` | ARM64 `memory_write_pattern_loop_asm`, `memory_write_cache_pattern_loop_asm`, `memory_write_reverse_pattern_loop_asm`, `memory_write_strided_phased_pattern_loop_asm`, and `memory_write_random_pattern_loop_asm` arbitrary 64-bit pattern fill, tails, AAPCS64 preservation, and guard-canary preservation |
| `test_core_to_core_messages.cpp` | `CoreToCoreMessagesTest` | Core-to-core console message strings |
| `test_core_to_core_cli.cpp` | `CoreToCoreCliTest` | Core-to-core CLI argument parsing (`--option=value`, duplicate flags, direct-vs-sweep conflicts, unified duplicate sweep-key diagnostics, config reset, and `argc == 0` safety) |
| `test_core_to_core_runner.cpp` | `CoreToCoreRunnerTest` | Calibration, work planning, cyclic scenario order, pooled sample ranges, and initiator-startup failure cleanup |
| `test_executable_cli.cpp` | `ExecutableCliIntegrationTest` | Executable-level CLI routing, `-B`/`--cpu-bandwidth`, `-L`/`--cpu-latency`, and `-P`/`--patterns` schema-1 direct/sweep payloads, split stdout/stderr capture, bounded child execution, sentinel artifact rules, explicit and automatic-calibration GPU contracts, bounded contiguous/paged direct and sweep LLM file/stdout/checkpoint contracts (CPU and Metal), and TLB/core-to-core end-to-end coverage |
| `test_standard_kernels.cpp` | `StandardKernelIntegrationTest`, `PatternKernelIntegrationTest` | Real ARM64 read/copy/latency/pattern kernel ABI, tails, boundaries, checksums, and latency sampling |
| `test_descriptive_statistics.cpp` | `DescriptiveStatisticsTest` | Canonical shared percentiles, deviation, CV, and MAD contracts |
| `test_statistics_renderer.cpp` | `StatisticsRendererTest` | Shared console-summary ordering, precision, indentation, and diagnostics |
| `test_timer.cpp` | `HighResTimerTest` | Deterministic Mach-time conversion, original boundary snapshots, creation failures, seconds/nanoseconds results, and tick wraparound |
| `test_system_info.cpp` | `BenchmarkQosTest`, `SystemInfoTest`, `SystemInfoIntegrationTest` | Deterministic QoS and system-information success/failure contracts plus one real-provider integration smoke |
| `test_tlb_chain.cpp` | `TlbChainTest` | Spread/packed planning, every traversal policy, and explicit corruption statuses |
| `test_tlb_measurement_scheduler.cpp` | `TlbMeasurementSchedulerTest` | Seeded balance, stop/error boundaries, callback contracts, convergence, and exact pass accounting |
| `test_tlb_runtime_policy.cpp` | `TlbRuntimePolicyTest` | Runtime profiles, calibration, convergence, memory budgets, and work estimates |
| `test_tlb_sweep_planner.cpp` | `TlbSweepPlannerTest` | Page-aligned base/refinement planning, stride bounds, deduplication, and source tracking |
| `test_utils.cpp` | `SeedUtilsTest`, `ProgressSpinnerTest` | Zero-seed fallback and enabled/disabled spinner rendering/cleanup |

### tests/fixtures/ — Script-example and verifier input fixtures

| File | Purpose |
|---|---|
| `cpu-bandwidth-schema-v1-complete-current.json` | Complete `-B`/`--cpu-bandwidth` schema-1 fixture covering main memory + custom cache bandwidth, GF(2) oracle/write-pattern checks, and raw Mach timing (`test_cpu_bandwidth_verifier.py` and `test_script_examples.py`) |
| `cpu-latency-schema-v1-complete-current.json` | Complete `-L`/`--cpu-latency` schema-1 fixture covering custom cache + main memory latency, canonical pointer-chain permutation hashes, final-pointer checks, and `pooled_sample_distribution` (`test_cpu_latency_verifier.py` and `test_script_examples.py`) |
| `patterns-schema-v1-complete-current.json` | Complete `-P`/`--patterns` schema-1 fixture (`-P -b 3 -t 2 -i 2 -r 3 --seed 424242`) with all 7 patterns active, slot-tagged GF(2) read checksums, 6-check cold post-validation, and raw Mach timing (`test_patterns_verifier.py`) |
| `patterns-schema-v1-skipped-2mb-current.json` | Complete `-P`/`--patterns` schema-1 fixture (`-P -b 1 -t 2 -i 2 -r 3 --seed 424242`) with `strided_2mb` explicitly `status: "skipped"` (`buffer-too-small-for-stride-transition`) (`test_patterns_verifier.py`) |
| `llm-schema-v2/` | Eight captured CPU/Metal × decode/prefill × contiguous/paged schema-2 profiles plus `unsupported.json` for independent verifier coverage |
| `README.md` | Records source HEAD, capture commands, example/verifier ownership, and fixture refresh policy |

Volatile source/test counts and the authoritative generated inventory are maintained in `DRY_CHECK.md`.

---

### Shared test helper headers

Two shared helper headers support deterministic setup and syscall hooks across the test suite:

| File | Purpose |
|---|---|
| `test_memory_system_calls.h` | Provides deterministic mmap/madvise/mprotect/munmap state and a resetting fixture for allocation tests |
| `test_timer_system_calls.h` | Provides deterministic Mach absolute-time hooks and a scope-bound reset guard for timer-dependent unit tests |

---

## 4. results/ — Result data

Historical JSON, CSV, and text output from benchmark runs on specific hardware, organized by software-version
subdirectory. The files are retained as examples and historical schema evidence; they are not baselines for the current
methodology unless explicitly identified as such. Historical standard JSON is not an input to current bundled
CPU result consumers; separately governed TLB tooling retains its documented legacy-input policy.

```
results/
  0.63.0/
    AppleM5_LLM_working_set_scaling.md  — Apple M5 CPU decode working-set comparison note
    M4MacMiniVSM5MacbookAir_LLM.md  — Informal LLM CPU decode comparison
    apple_m5_cpu_decode_contiguous_weights_256mib_context_32768.json  — Recorded LLM schema-1 sample
    apple_m5_cpu_decode_contiguous_weights_1024mib_context_131072.json  — Recorded LLM schema-1 sample
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

The 0.63.0 result notes include an
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

Example shell workflows, independent Python verifiers, and Python/Matplotlib plotters for tracked benchmark outputs.
The four CPU benchmark script-examples accept compatible producer releases rather than requiring an exact software
version. Each JSON-reading entry point checks for a non-empty `software.version` provenance string, `schema_version == 1`,
`mode` (`"cpu_bandwidth"` or `"cpu_latency"`), the exact methodology identity
(`cpu-bandwidth-v1-arm64-calibrated-seeded-balanced` or `cpu-latency-v1-arm64-calibrated-seeded-balanced`),
`status == "complete"`, `results_complete == true`, `run_accepted == true`, and the shapes of the fields it consumes
before reading those metric paths. Legacy standard schema 3/2, unversioned historical layouts, alternative modes, and
other methodology identities are not translated by those four examples. Separately governed mode-specific tools retain
only the history policy documented in their own row below.

| File | Purpose |
|---|---|
| `cpu_verify_common.py` | Shared strict JSON, SplitMix64/Fisher–Yates RNG, Mach timing, statistics, audit, memory-budget, and sweep envelope primitives (Python 3 stdlib only) for the CPU bandwidth, CPU latency, and patterns verifiers |
| `verify_cpu_bandwidth_result.py` | Bounded standalone verifier for `-B`/`--cpu-bandwidth` schema-1 artifacts and schema-1 `cpu_bandwidth` sweep envelopes; checks O(1) GF(2) read checksums, write/copy validation, raw Mach timing, and `run_accepted` |
| `verify_cpu_latency_result.py` | Bounded standalone verifier for `-L`/`--cpu-latency` schema-1 artifacts and schema-1 `cpu_latency` sweep envelopes; checks canonical chain permutation SHA-256 digests, final-pointer/sample-window end nodes, `--compare-locality`, raw Mach timing, and `run_accepted` |
| `verify_patterns_result.py` | Bounded standalone verifier for `-P`/`--patterns` schema-1 artifacts and schema-1 `patterns` sweep envelopes; checks 7-pattern geometry/`strided_2mb` skips, slot-tagged GF(2) read checksums, 2D partial-cycle strided/random oracles, 6-check cold post-validation, raw Mach timing, and `run_accepted` |
| `verify_llm_result.py` | Bounded standalone verifier for single LLM schema-2 artifacts and schema-1 LLM sweep envelopes; separate consistency, acceptance, and timing/build evidence levels |
| `llm_verify_oracles.py` | Independent CPU and Metal arithmetic/checksum reconstruction for the verifier |
| `llm_verify_profiles.py` | Supported LLM component identities for the eight verifier profiles |
| `final_output.txt` | Small bundled sample of pooled latency statistics for `plot_cache_percentiles.py`; regenerated by `latency_test_script.sh` |
| `latency_test_script.sh` | Runs `--cpu-latency` across custom cache sizes and configured latency-locality windows, writes per-run JSON, and extracts `pooled_sample_distribution.statistics` into `final_output.txt` using jq when available or Python 3 otherwise |
| `latency_test_script_stride_tlb.sh` | Runs `--cpu-latency` across cache size, configured locality, and latency stride; uses embedded Python 3 to read `pooled_sample_distribution.statistics`, `resolved_plan.chain_identities.custom.node_count`, and `environment.page_size_bytes`, retains per-run JSON, and requires one complete CSV row per planned run |
| `plot_M4vsM5_benchmark_comparison.py` | Compares effective payload bandwidth and latency from four explicitly supplied schema-1 JSON files (`--m4-bandwidth-file`, `--m4-latency-file`, `--m5-bandwidth-file`, `--m5-latency-file`; default `--metric median`) |
| `plot_analyzetlb.py` | Plots standalone TLB locality trends, including the paired spread/packed delta in current schemas and supported legacy data |
| `plot_benchmark-memory-latency-hierarchy.py` | Plots memory-hierarchy latency (`--metric` defaults to `median`) from an explicit `cpu_latency` schema-1 JSON generated with `--compare-locality`, or from live `-L --compare-locality` console text (`--metric median` only) |
| `plot_cache_percentiles.py` | Plots a selected pooled latency statistic by cache size and configured locality from `final_output.txt` |
| `plot_cache_percentiles_stride_tlb.py` | Plots a selected pooled latency statistic from the stride/locality sweep CSV, with optional stride and locality filters |

---

## 7. .github/ — GitHub integration

| File | Purpose |
|---|---|
| `.github/pull_request_template.md` | Default pull request description template |
| `.github/ISSUE_TEMPLATE/bug_report.md` | Structured bug report template |
| `.github/workflows/pr-tests.yml` | Runs `make test-all` on `macos-26` after installing GoogleTest, for pushes to `development` and pull requests targeting `main` or `development` |

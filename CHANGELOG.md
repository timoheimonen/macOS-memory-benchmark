# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.55.0] - 2026-X-X

### Changed
  - **`-benchmark` flag required for standard benchmarks**: Running with no arguments now prints help instead of starting the benchmark. Use `-benchmark` to run the standard memory benchmark (replicates old default behavior). `-benchmark` and `-patterns` are mutually exclusive. Standalone modes (`-analyze-tlb`, `-analyze-core2core`) are unaffected.
  - **`BenchmarkConfig` extended**: Added `bool run_benchmark` field to `BenchmarkConfig` struct (`config.h`) to track whether the user requested standard benchmark execution.
  - **Modifier flags require a mode flag**: Running with only modifier flags (e.g., `-only-latency`, `-only-bandwidth`, `-buffersize`) without `-benchmark` or `-patterns` now prints help and exits.
  - **Help text updated**: `-benchmark` documented as the first option; mutual exclusivity with `-patterns` noted in both option descriptions.
  - **Documentation and scripts updated**: `README.md`, `MANUAL.md`, and `script-examples/` updated to reflect `-benchmark` requirement across all examples and workflow descriptions.
  - Main-memory read/write/copy ASM kernels converted from offset-add addressing to pointer-bump addressing (`src/asm/memory_read.s`, `src/asm/memory_write.s`, `src/asm/memory_copy.s`). The hot loop no longer recomputes `base + offset` each iteration; instead a running pointer advances directly, eliminating one `add` per 512B block.
  - Reduced `memory_write.s` zero-register setup from 24 to 8 `movi` instructions (the stnp pairs reuse the same zero registers).
  - Reduced `memory_write_reverse.s` zero-register setup from 24 to 8 `movi` instructions, matching the forward write kernel. The 512B loop and 256B cleanup now reuse `q0–q7` for all 16 stnp pairs instead of using separate `q16–q31` registers.
  - Updated clobber documentation to reflect actual register usage.
  - **Improved `-analyze-tlb` boundary detection stability**: Replaced single-point persistence check with 3-point majority persistence, recency-weighted baselines, and IQR-overlap rejection. These changes reduce false boundary detections from transient noise and improve L1/L2 TLB classification accuracy.
  - **L2 detection uses offset baseline and specific guard**: L2 scanning now starts 2 points past the L1 boundary (instead of at the L1 boundary index) and uses a locality guard of `max(tlb_guard, L1_boundary_locality)`, preventing L1 transition noise from contaminating the L2 baseline.
  - **Last-point strong-step compensation**: Final sweep points with very large steps (>=8 ns or >=25%) now get effective persistence, preventing massive last-point steps from being downgraded to Low confidence.

### Added
  - **`-benchmark` CLI flag**: New explicit mode flag to run the standard memory benchmark. Required for standard, `-only-bandwidth`, and `-only-latency` modes. Mutually exclusive with `-patterns`. When omitted (and no other mode flag is set), the program prints help and exits.
  - **Mutual exclusion validation**: Added runtime check that rejects `-benchmark` and `-patterns` together with a clear error message.
  - **New error message `error_mutually_exclusive_modes()`**: Centralized mutual-exclusion error string in the Messages system.
  - **IQR-overlap rejection in TLB boundary detection**: When raw per-point loop latencies are available, the detector compares baseline Q3 with candidate Q1 and rejects boundaries where the step falls within measurement noise overlap.
  - **Multi-point persistence window**: Boundary persistence is now evaluated over a 3-point future window using majority rule (2 of 3), replacing the previous single-point check.
  - **New test cases for improved TLB detection**: Added `DetectBoundaryMultiPointPersistenceSurvivesNoiseDip`, `DetectBoundaryRejectsNoisyStepByIQR`, `DetectBoundaryAcceptsClearStepWithIQR`, and `DetectBoundaryLastPointStrongStepGetsMediumConfidence` tests.
  - **New tests for `-benchmark` flag**: Added `ConfigTest.ParseBenchmarkFlag`, `ConfigTest.ParseBenchmarkWithModifiers`, `ConfigTest.ParsePatternsSetsRunPatterns`, and `ConfigTest.BenchmarkPatternsMutuallyExclusive` tests. Updated existing validation tests to set `run_benchmark = true` where required.
  - **Parameter compatibility matrix**: Added `PARAMETER_MATRIX.md` documenting all CLI flag combinations and their validity.
  - **`-analyze-tlb` sweep progress now shows measured latency**: Each sweep step now prints the measured P50 latency alongside the locality label (e.g., `[1/15] Locality 16 KB — 10.23 ns`), giving real-time per-step feedback during the analysis sweep.
  - **Signal handler test coverage**: Added `MessagesFormattingTest.MsgInterruptedByUser` test verifying interrupt message content and static reference stability.
  - **Test count updated**: Unit test suite now contains 326 tests (was 325).

### Fixed
  - **Compiler warning in buffer overflow checks**: Fixed `-Wimplicit-const-int-float-conversion` warnings in `buffer_calculator.cpp` where `std::numeric_limits<size_t>::max()` was implicitly converted to `double` during L1/L2 buffer size factor division. Overflow checks now use integer arithmetic via `static_cast<size_t>(factor)`.
  - **Test correctness and cleanup**: Reviewed all 17 test files (~140 test cases) for correctness and necessity. Fixed 7 `PatternValidationTest` cases that claimed to test specific buffer-size boundaries but all used identical `buffer_size=512` without varying the boundary. Tests now use correct buffer sizes targeting the effective stride boundary (`stride + PATTERN_ACCESS_SIZE_BYTES`), matching each test name to actual behavior. Removed duplicate `AllPatternTypesComplete` test (identical to `RunPatternBenchmarksMinimalIntegration`). Removed 3 trivial struct-default tests (`StatisticsInitialization`, `BenchmarkResultsDefaults`, `StatisticsStructure`) that only tested C++ language behavior. Fixed stale comments claiming L1/L2 buffer size factors are 75%/10% when constants are 100%. Removed redundant `MAP_FAILED` assertions from `MemoryManagerTest` (allocator returns valid pointer or `nullptr`, never `MAP_FAILED`). Net result: 326/326 unit tests pass.
  - **DRY cleanup in `main.cpp`**: Removed 54-line duplicated pattern results reconstruction block (two identical 24-field copy-and-assign sections for single-loop vs multi-loop display). Replaced with `extract_pattern_results_at()` helper in `src/pattern_benchmark/pattern_statistics_manager.cpp` that takes a `PatternStatistics` and an index, returning a populated `PatternResults`. The two call sites now differ only in the index parameter (`0` vs `last_idx`). Added bounds clamping and empty-vector guard.
  - **Graceful Ctrl+C shutdown**: Added signal handling infrastructure so that pressing Ctrl+C during a long benchmark run (`-count`, `-loops`) stops the program between test phases or between loops instead of killing it mid-flight. Worker threads inherit a blocked SIGINT/SIGTERM mask so only the main thread handles interruption. The main thread checks for pending signals via `sigpending()` between test phases in `run_single_benchmark_loop()`, between individual pattern types in `run_pattern_benchmarks()`, and between loops in `run_all_benchmarks()` / `run_all_pattern_benchmarks()`. On interrupt, partial results from completed phases/loops are printed with an "Interrupted by user. Partial results shown." message, then the program exits cleanly with `EXIT_SUCCESS`. New module `src/core/signal/` with `signal_handler.h` and `signal_handler.cpp`.

## [0.54.0] - 2026-03-19

Changes in benchmark ASM kernels.
From this version onwards bandwidth results are not compatible against older(<0.54.0) version results.
Latency kernels unchanged.

### Changed
  - Main change: split pattern ASM kernels into separate files for each case (main-memory, cache, forward, reverse, strided, random).
  - Some kernels currently behave the same, but they are intentionally kept separate to make future tuning easier.
  - Tuned cache ASM kernels with pointer+remaining loop state and bit-test based tail handling.
  - Pattern strided benchmarks now scale effective iterations so each stride variant targets at least `256 MB` of touched data (for more stable results on small buffers).
  - Added `PATTERN_STRIDED_MIN_TOUCHED_BYTES` constant (`256 * 1024 * 1024`) for strided pattern iteration scaling.
  - Simplified and improved inline comments in cache ASM kernels:
    - `src/asm/memory_write_cache.s`
    - `src/asm/memory_read_cache.s`
    - `src/asm/memory_copy_cache.s`
  - Added clearer ABI/register-preservation notes, tail-bit (`tbz`) mapping notes, and short control-flow maps for easier future tuning.

## [0.53.9] - 2026-03-18

### Changed
  - **Centralized user-facing message strings**: Moved three remaining hardcoded user-facing strings to the Messages system:
    - Timer creation failure error (`main.cpp:95`) now uses `Messages::error_timer_creation_failed()`
    - JSON output success message (`file_writer.cpp:143`) now uses `Messages::msg_results_saved_to()`
    - Pattern benchmark loop progress (`pattern_statistics_manager.cpp:134`) now uses `Messages::msg_pattern_benchmark_loop_completed()`
  - **Test coverage for new message functions**: Added test cases for all three new message functions in `tests/test_messages.cpp`.
  - **DRY cleanup across benchmarks, warmup, and tests**: Extracted shared helpers to remove duplicated latency, warmup, pattern, and test setup code.

## [0.53.8] - 2026-03-14

### Added
  - **Configurable latency chain-construction policies (`-latency-chain-mode`)**: Added selectable pointer-chain modes for latency paths and standalone TLB analysis: `auto` (default), `global-random`, `random-box`, `same-random-in-box`, and `diff-random-in-box`.
  - **Latency chain mode surfaced in outputs**: Configuration output now prints the effective latency chain mode, and benchmark JSON now includes `configuration.latency_chain_mode`.

### Changed
  - **Standard benchmark memory lifecycle moved to phase-local allocation**: Standard mode now allocates and initializes buffers immediately before each benchmark phase (main bandwidth, cache bandwidth, cache latency, main latency) and releases them after the phase instead of keeping all mode buffers resident for the full run.
  - **Memory estimate now reports peak concurrent footprint**: Configuration output/help text now describes the memory requirement as peak concurrent allocation for the active mode/phase model.
  - **Validation defaults aligned with phased execution**: Main-buffer cap logic now validates against phased peak needs (for example, 2x main buffer during copy bandwidth, 1x in latency-only main path).
  - **Latency-chain validation and `-analyze-tlb` option contract extended**: Box-based chain modes now require non-zero `-latency-tlb-locality-kb`, and standalone TLB mode now allows optional `-latency-chain-mode <mode>` in addition to `-output` and `-latency-stride-bytes`.
  - **Documentation updated for chain modes**: Updated `README.md` and `MANUAL.md` to document latency-chain modes, examples, and troubleshooting guidance.

### Fixed
  - **Over-reservation for standard benchmark runs**: Large `-buffersize` runs no longer reserve all standard-mode buffers up-front when only one phase is active at a time.

## [0.53.7] - 2026-03-13

### Added
  - **Schema contract tests for JSON exports**: Added `tests/test_json_schema.cpp` coverage to verify mode tagging, canonical latency sample container shape, and shared-key consistency across benchmark, pattern, TLB-analysis, and core-to-core JSON exporters.

### Changed
  - **Unified JSON mode tagging across exporters**: Standard benchmark and pattern exports now include `configuration.mode` (`benchmark` / `patterns`), and standalone exporters use the same shared key convention (`mode`) for `analyze_tlb` and `analyze_core2core`.
  - **Canonicalized latency sample JSON structure**: Latency sample payloads now use `samples_ns.values` with optional `samples_ns.statistics` (legacy `samples_statistics` field removed).
  - **Aligned TLB-analysis configuration metadata with shared JSON keys**: Standalone TLB analysis JSON now includes `performance_cores` and `efficiency_cores` in `configuration` for parity with other exporters.
  - **Omitted empty result sections instead of emitting nulls**: Standard and pattern JSON writers now skip absent payload sections rather than serializing `null` blocks.

### Fixed
  - **Documentation/script schema drift after sample-structure normalization**: Updated docs and script examples to read `samples_ns.statistics` consistently (`README.md`, `MANUAL.md`, `TECHNICAL_SPECIFICATION.md`, and latency helper scripts).

## [0.53.6] - 2026-03-13

### Added
  - **Standalone core-to-core latency analysis mode (`-analyze-core2core`)**: Added a dedicated two-thread cache-line handoff benchmark mode that runs outside standard benchmark orchestration and supports optional `-output <file>`, `-count <count>`, and `-latency-samples <count>`.
  - **Core-to-core scenario reporting with scheduler hints**: New mode executes `no_affinity_hint`, `same_affinity_tag`, and `different_affinity_tags` scenarios, reports round-trip latency, one-way estimate, sample distribution statistics (P50/P90/P95/P99/stddev/min/max), and per-thread QoS/affinity hint status.
  - **Core-to-core JSON export**: Added mode-specific JSON payload under `core_to_core_latency` including loop values, sample distributions, computed statistics, and thread-hint application results.

### Changed
  - **CLI help text extended for standalone core-to-core mode**: `-h/--help` now documents `-analyze-core2core` and its allowed option set.
  - **Core-to-core handoff loop uses dedicated ARM64 assembly kernels**: Standalone `-analyze-core2core` now executes initiator/responder token ping-pong rounds through dedicated assembly routines, while keeping the existing mode interface, reporting, and JSON output structure.
  - **Refactored standalone TLB analysis module into smaller units**: Split previous `src/benchmark/analysis.cpp` into focused files: `src/benchmark/tlb_analysis.cpp` (orchestration), `src/benchmark/tlb_boundary_detector.cpp` (boundary detection and confidence logic), and `src/benchmark/tlb_analysis_json.cpp` (JSON serialization).
  - **Renamed TLB analysis interface/header**: Renamed `src/benchmark/analysis.h` to `src/benchmark/tlb_analysis.h` and updated includes/references accordingly (`main.cpp`, tests, and docs).
  - **Build wiring updated for new TLB analysis files**: Updated benchmark source list in `Makefile` to compile/link the new `tlb_analysis*` and `tlb_boundary_detector` units.

## [0.53.5] - 2026-03-12

### Added
  - **Aggregate statistics for automatic DRAM TLB breakdown metrics**: When auto TLB comparison is active (regular mode without explicitly setting `-latency-tlb-locality-kb`), end-of-run statistics now include full sections for `TLB Hit Latency (ns)`, `TLB Miss Latency (ns)`, and `Estimated Page-Walk Penalty (ns)` with Average/P50/P90/P95/P99/Stddev/Min/Max.
  - **JSON export for automatic DRAM TLB breakdown metrics**: Standard benchmark JSON now includes `main_memory.latency.auto_tlb_breakdown` with per-loop `values` (and multi-loop `statistics`) for `tlb_hit_ns`, `tlb_miss_ns`, and `page_walk_penalty_ns`.

### Changed
  - **Loop aggregation plumbing extended for auto TLB metrics**: Benchmark statistics collection now persists per-loop auto TLB hit/miss/penalty values so console summary statistics and JSON output remain consistent with per-loop console results.

## [0.53.4] - 2026-03-12

### Added
  - **Automatic DRAM TLB breakdown in regular benchmark mode**: When `-latency-tlb-locality-kb` is not explicitly set by the user, main-memory latency output now includes additional lines for `TLB hit latency` (16 KB locality), `TLB miss latency` (global random locality, `0`), and an `Estimated page-walk penalty` (`miss - hit`).
  - **Result plumbing for per-loop TLB breakdown metrics**: Added loop-level result fields for auto TLB hit/miss/page-walk values so they can be emitted in standard results output.

### Changed
  - **Main-memory latency execution path now performs an auto comparative pass**: In auto mode, the regular DRAM latency path keeps the default 16 KB locality measurement and adds a second global-random pass for miss-biased timing.
  - **Main-memory latency report now restores default locality chain after auto miss pass**: The latency chain is rebuilt back to configured/default locality after the global-random measurement so subsequent loops keep expected baseline behavior.

## [0.53.3] - 2026-03-12

### Added
  - **`-analyze-tlb` JSON export support via `-output <file>`**: Standalone TLB analysis mode now supports optional JSON output for verification workflows. Export includes full sweep locality points, raw per-loop latencies, per-point P50 values, inferred L1/L2 boundaries (with confidence and inferred entries), guard parameters, and page-walk penalty details.
  - **TLB analysis whitepaper**: Added `TLB_ANALYSIS_WHITEPAPER.md` documenting standalone TLB-analysis methodology, guard/boundary logic, confidence classification, and JSON output verification structure.
  - **`-analyze-tlb` optional argument support**: Added support for optional `-output <file>` and optional `-latency-stride-bytes <bytes>` with `-analyze-tlb`, while keeping other options disallowed in this mode.
  - **`-analyze-tlb` configurable stride source**: Added usage of configured `-latency-stride-bytes` in standalone TLB analysis (defaulting to the same stride default as standard latency mode).
  - **Stride-aware TLB sweep and page-walk baseline**: Added effective sweep start at `max(16KB, 2*stride)` (inserting that point when needed) to avoid invalid pointer-chain locality windows for large stride values. Page-walk penalty baseline now follows the same effective first sweep locality.

## [0.53.2] - 2026-03-12

### Changed
  - **Latency chain sizing now follows the configured stride**: Main/cache latency buffer sizing and pointer-chain setup now use the active `latency_stride_bytes` value instead of a fixed internal stride constant, keeping chain geometry and access-count calculations consistent when stride is customized.
  - **Help text `-cache-size` range now dynamically sourced from constants**: Updated `src/output/console/messages/program_messages.cpp` to generate the `-cache-size` range bounds from `Constants::MIN_CACHE_SIZE_KB` and `Constants::MAX_CACHE_SIZE_KB` instead of hardcoding values. This ensures help output (`-h`) and parser/validation limits stay automatically synchronized and prevents future drift.
  - Default Stride changed from 134 to 64.

### Fixed
  - **Help text showed stale 524288 KB max for `-cache-size` while parser allowed 1048576 KB**: The hardcoded max in help text generation is now replaced by a reference to the runtime constant, eliminating the drift and preventing regression.

### Added
  - **Configurable latency pointer-chase stride (`-latency-stride-bytes`)**: Added a new CLI option to control pointer-chain stride in bytes (default `136`). The value must be `> 0`, pointer-size aligned, and compatible with cache/locality constraints.
  - **Latency chain diagnostics in JSON output**: Added optional `chain_diagnostics` blocks for latency results (main/cache) with `pointer_count`, `unique_pages_touched`, `page_size_bytes`, and `stride_bytes` to support page-touch and TLB-vs-cache analysis.
  - **Coverage for stride parsing/validation and diagnostics**: Added targeted tests for stride defaults/parsing/validation messaging and latency-chain diagnostics serialization paths.
  - **Regression test for `-cache-size` help bounds**: Added check in `tests/test_messages.cpp` to verify that `UsageOptions` message includes both `MIN_CACHE_SIZE_KB` and `MAX_CACHE_SIZE_KB` constants in the cache-size option line.
  - **Standalone TLB analysis mode (`-analyze-tlb`)**: Added a dedicated mode that must be used alone, allocates a fallback analysis buffer (`1024 MB`, `512 MB`, then `256 MB`), sweeps TLB-locality windows (`16 KB` to `256 MB`) with fixed `64B` stride over 30 loops per point, and reports inferred L1/L2 TLB boundaries and confidence. Page-table-walk penalty is reported separately as `P50(512 MB) - P50(16 KB)` when analysis buffer size is at least `512 MB`.

## [0.53.1] - 2026-03-09

### Added
  - Two benchmark json files from Mac Mini M4 24GB machine.

### Changed
  - Benchmarks now default to 100% use of detected L1/L2 cache sizes.

## [0.53.0] - 2026-03-08

### Added
- **Selective latency-target execution in `-only-latency` mode**: You can now disable one latency path at a time:
  - `-buffersize 0` disables main memory latency
  - `-cache-size 0` disables cache latency
  Disabled latency targets are also skipped in allocation and execution paths.
- **TLB-locality mode for latency pointer-chase setup**: Added `-latency-tlb-locality-kb <size_kb>` to build latency chains with randomized access inside locality windows (instead of fully global randomization), helping reduce TLB-refill contamination when analyzing cache-level latency behavior.

### Changed
- **Clear zero-value rules for latency-only mode**: `-buffersize 0` and `-cache-size 0` are valid only with `-only-latency`; using both as zero at the same time is rejected as an invalid no-target configuration.
- **Cleaner latency-only output and docs**: Help/docs now describe zero-disable behavior, and console output omits disabled latency sections while showing a clear cache-disabled status when applicable.
- **More accurate total allocation reporting**: Displayed total allocation now reflects active benchmark mode and enabled buffers.
- **Strict validation and defaults for TLB-locality option**: `-latency-tlb-locality-kb` now defaults to `16` (KB); setting `0` explicitly disables locality mode. Non-zero values must be exact multiples of the system page size, and configuration/JSON output include explicit TLB-locality mode and size fields.

### Fixed
- **Main latency disable path in config validation**: `-only-latency -buffersize 0` no longer gets reset to default buffer size.
- **Custom cache zero handling in buffer sizing**: `-only-latency -cache-size 0` no longer rounds up to a minimum custom cache buffer.
- **Disabled-target statistics noise**: Statistics and loop output no longer show empty or irrelevant latency blocks for disabled targets.

## [0.52.9] - 2026-03-08

### Added
- **Extended `-patterns` strided coverage for page-size analysis**: Added two new strided pattern measurements to complement existing 64B and 4096B tests:
  - `strided_16384` (16 KB stride; Apple Silicon page-size candidate)
  - `strided_2mb` (2 MB stride; superpage-scale stride candidate)
  These are now executed, aggregated, and reported in both console and JSON outputs.

### Changed
- **Pattern output and stats now include new stride variants**: Console pattern results/statistics and JSON pattern payloads now include `strided_16384` and `strided_2mb` fields alongside existing pattern metrics.

### Fixed
- **Pattern strided skip behavior messaging**: Removed noisy error-style prints for expected "buffer too small for this stride" skip paths in pattern execution, while preserving graceful skip behavior and zero-valued results when a stride cannot be run.

## [0.52.8] - 2026-03-07

### Fixed
- **Zero-access latency chase safety**: Fixed `memory_latency_chase_asm` to return immediately when `count == 0`, preventing an unnecessary dereference when no pointer-chasing accesses are requested.
- **Unsigned loop termination in latency chase**: Fixed loop branch conditions in `memory_latency_chase_asm` to use counter-based `b.ne` termination, ensuring correct behavior for full `size_t` ranges.
- **Latency measurement overhead in pointer chase kernel**: Removed pre-touch and barrier instructions (`ldr` preload, `dsb/isb`, and `dmb` fences) from `memory_latency_chase_asm` to avoid adding non-chase overhead to latency measurements.
- **`size_t`-safe branch/cleanup fixes across ARM64 bandwidth kernels**: Unified loop/cleanup/byte-tail control flow in forward, reverse, strided, and random read/write/copy kernels to use unsigned or zero/non-zero termination semantics (`b.hs`/`b.lo`/`b.ls`, `cbz`, `b.ne`), and fixed reverse cleanup exit direction plus reverse copy byte-tail pointer initialization for correct full-range behavior.
- **Mode-aware memory cap validation for `-only-bandwidth` / `-only-latency` / `-patterns`**: Updated `validate_config()` to compute per-buffer limits using the active main-buffer count (1/2/3) instead of always dividing by 3, preventing unnecessary buffer-size capping in reduced-buffer modes while keeping total-memory safeguards intact.
- **Main-buffer total-memory overflow guard in allocator**: Added missing checked-add overflow validation when including the main latency buffer in `allocate_all_buffers()` total-memory accounting, and added a targeted unit test to lock in the overflow failure path.

## [0.52.7] - 2026-01-05

### Added
- **macOS version to JSON output**: JSON output files now include macOS version in the configuration section, example "macos_version": "26.2".

## [0.52.6] - 2026-01-04

### Fixed
- **Pattern benchmarks ignoring `-threads` parameter**: Fixed issue where pattern benchmarks (forward, reverse, strided, and random access patterns) were running single-threaded even when the `-threads` parameter was specified.

## [0.52.5] - 2026-01-04

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

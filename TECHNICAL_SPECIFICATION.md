# macOS Memory Benchmark - Technical Specification

## 1. Scope and Status

This document specifies the current implementation in this repository (version series `0.54.x`) for `memory_benchmark` on macOS Apple Silicon.

It is intentionally implementation-driven and reflects real behavior in code paths under `main.cpp`, `src/core`, `src/benchmark`, `src/pattern_benchmark`, `src/output`, and `src/asm`.

Primary goals:

- Define runtime architecture and execution flow.
- Define command/config semantics and validation rules.
- Define memory allocation, initialization, and benchmark execution contracts.
- Define output contracts (console and JSON).
- Capture current constraints, known drift, and measurement caveats.

Out of scope:

- Generic memory-performance theory (see [LATENCY_WHITEPAPER.md](LATENCY_WHITEPAPER.md)).
- `-analyze-tlb` methodology details (see [TLB_ANALYSIS_WHITEPAPER.md](TLB_ANALYSIS_WHITEPAPER.md)).
- `-analyze-core2core` methodology details (see [CORE_TO_CORE_WHITEPAPER.md](CORE_TO_CORE_WHITEPAPER.md)).
- Historical behavior from older releases.

## 2. Platform and Build Constraints

- Target OS: macOS.
- Target CPU architecture: ARM64 Apple Silicon.
- Language: C++17 with ARM64 Apple Silicon assembly kernels.
- Build: `Makefile` (`clang++`, `as`).
- Test framework: GoogleTest (`test_runner`).

The tool is designed and tuned for Apple Silicon execution characteristics (cache hierarchy, page behavior, QoS, and assembly kernels).

## 3. Hardware Limitations

### 3.1 Thermal constraints on fanless systems

Fanless Apple Silicon systems (e.g., MacBook Air M5) have limited thermal capacity and cannot sustain heavy benchmarking for extended durations.

Observed behavior:

- MacBook Air M5 typically enters Heavy thermal state after approximately 2 pattern benchmark runs.
- Thermal state transitions depend significantly on ambient environment temperature.
- Thermal throttling during benchmarking invalidates measurement reliability and comparability.

Impact:

- Repeated back-to-back benchmark executions may encounter thermal limiting.
- Measurements from throttled runs should not be compared with baseline measurements from thermal-normal conditions.
- For consistent results on fanless systems, allow sufficient idle time between consecutive benchmark runs to permit thermal cool-down.

Recommendation:

- For reliable baseline measurement on fanless systems, run single benchmark iterations with thermal cool-down intervals between runs.
- Use `caffeinate -i -d memory_benchmark ...` to prevent system sleep during longer measurement sessions on cooler systems.

## 4. High-Level Runtime Architecture

Main orchestration (`main.cpp`) follows this pipeline:

Standalone modes (`-analyze-tlb`, `-analyze-core2core`) are dispatched early and use dedicated runners.
The pipeline below applies to standard/pattern benchmark execution.

1. Create high-resolution total-execution timer.
2. Parse CLI arguments into `BenchmarkConfig` (`parse_arguments`).
3. Validate configuration (`validate_config`).
4. Calculate derived sizes and counts:
   - `calculate_buffer_sizes`
   - `calculate_access_counts`
   - `calculate_total_allocation_bytes`
5. Print resolved configuration and cache info.
6. Raise main thread QoS (`QOS_CLASS_USER_INTERACTIVE`) best-effort.
7. Execute one mode:
   - Standard benchmark mode (`run_all_benchmarks`), which allocates/initializes buffers per phase and releases them after the phase.
   - Pattern benchmark mode (`run_pattern_benchmarks`), which pre-allocates/initializes required buffers (`allocate_all_buffers` + `initialize_all_buffers`).
8. Print loop results and aggregate statistics.
9. Optionally serialize JSON output.
10. Print total elapsed runtime.

Memory cleanup is RAII-based through `MmapPtr` custom deleters (`munmap` on scope exit).

## 5. Configuration Model

Configuration state is represented by `BenchmarkConfig` (`src/core/config/config.h`).

### 5.1 User-facing control fields

- Main options: buffer size MB, iterations, loop count, output path, threads.
- Mode flags: `run_patterns`, `only_bandwidth`, `only_latency`.
- Standalone analysis flags are handled outside `BenchmarkConfig` orchestration flow:
  - `-analyze-tlb`
  - `-analyze-core2core`
- Cache behavior: auto L1/L2 or user-provided `-cache-size`.
- Latency sampling: `latency_sample_count`.
- Latency-chain construction mode:
  - `latency_chain_mode` (type `LatencyChainMode`, CLI flag `-latency-chain-mode`)
  - `user_specified_latency_chain_mode` flag
- TLB-locality control for latency chain construction:
  - `latency_tlb_locality_bytes` (default 16 KB)
  - `0` means global random chain.
- Best-effort cache-discouraging mode: `use_non_cacheable`.

### 5.2 Derived fields

- Resolved byte sizes for main and cache buffers.
- Access counts for latency paths.
- System metadata (CPU name, macOS version, core counts).
- Max memory limits and bookkeeping flags.

## 6. CLI Parsing and Validation

### 6.1 Parsing behavior (`argument_parser.cpp`)

- Two-pass parse:
  - First pass extracts `-cache-size` early.
  - Second pass parses remaining options.
- Parser may throw internally (`std::stoll`/validation) but converts to return-code failures at function boundary.
- Help (`-h`, `--help`) prints usage and exits successfully.
- `-latency-chain-mode` accepts string values and resolves to `LatencyChainMode` enum.
- `-analyze-core2core` uses dedicated mode parsing (outside `argument_parser.cpp`) and only allows optional `-output`, `-count`, and `-latency-samples`. Full methodology and JSON contract: [CORE_TO_CORE_WHITEPAPER.md](CORE_TO_CORE_WHITEPAPER.md).

### 6.2 Validation behavior (`config_validator.cpp`)

Validation rejects incompatible flag combinations and invalid value states:

- `-only-bandwidth` and `-only-latency` are mutually exclusive.
- `-patterns` cannot be combined with `-only-bandwidth` or `-only-latency`.
- `-only-bandwidth` cannot be combined with `-cache-size` and cannot use latency-sample overrides.
- `-only-latency` cannot be combined with `-iterations` override.

Zero-disabling semantics (supported only in `-only-latency`):

- `-buffersize 0` disables main-memory latency path.
- `-cache-size 0` disables cache-latency path.
- Both cannot be zero simultaneously in `-only-latency`.

TLB-locality constraints:

- Non-zero `-latency-tlb-locality-kb` must be a multiple of system page size.
- Non-zero locality window must span at least two latency-stride steps.

Latency stride constraints:

- `-latency-stride-bytes` must be greater than zero.
- Stride must be pointer-size aligned.

Memory-limit model:

- System available memory is queried.
- Global cap uses `MEMORY_LIMIT_FACTOR` (80%).
- Per-main-buffer cap is mode-aware (1 or 2 main buffers, depending on active mode/phase needs).
- A second peak-concurrent allocation check validates the highest active phase footprint (main + cache paths).

## 7. Size and Access Derivation

### 7.1 Buffer sizing (`buffer_calculator.cpp`)

- Main buffer size is derived from `buffer_size_mb`.
- L1/L2/custom cache test buffers use factor constants currently set to `1.0`.
- Cache buffers are rounded to active latency stride granularity (`latency_stride_bytes`, default `256`) and minimum constraints.
- Minimum practical lower bound includes page-size enforcement.
- `-cache-size 0` (in allowed mode) produces zero custom cache buffer.

### 7.2 Latency access counts

- Main-memory latency accesses scale from base count relative to default buffer size.
- Cache latency access counts use fixed constants (`L1`, `L2`, `CUSTOM`).
- `-buffersize 0` (in allowed mode) sets main latency accesses to zero.

## 8. Memory Allocation and Initialization

### 8.1 Allocation strategy

Allocation entrypoints:

- Standard benchmark mode: per-phase allocators in `src/benchmark/benchmark_executor.cpp` (`prepare_*_buffers` helpers).
- Pattern benchmark mode: `allocate_all_buffers` (`src/core/memory/buffer_allocator.cpp`).

Shared allocation behavior:

- Uses `mmap` anonymous private mappings (macOS-specific behavior and limits apply).
- Uses mode-aware conditional allocation to avoid unused buffers.
- Performs overflow-safe byte arithmetic before allocation.
- Enforces global memory-limit checks from peak-concurrent requirements.

Allocated buffer families (conditional):

- Main bandwidth: `src`, `dst`.
- Main latency: `lat`.
- Cache latency: `l1/l2` or `custom`.
- Cache bandwidth: `l1_bw_src/dst`, `l2_bw_src/dst` or `custom_bw_src/dst`.

Pattern mode intentionally allocates and uses only main source/destination buffers.

### 8.2 Best-effort non-cacheable mode

- `allocate_buffer_non_cacheable` still uses normal user-space mappings.
- Applies `madvise(MADV_RANDOM)` hints on macOS.
- This is best-effort cache discouragement only; not true uncached memory.

### 8.3 Initialization strategy

Initialization entrypoints:

- Standard benchmark mode: per-phase initialization in `run_single_benchmark_loop` before each measured phase.
- Pattern benchmark mode: `initialize_all_buffers` (`src/core/memory/buffer_initializer.cpp`).

Initialization semantics:

- Bandwidth buffers: deterministic source pattern + zeroed destination.
- Latency buffers: randomized pointer-chasing circular chain via `setup_latency_chain`.
- Allocation/initialization happen before phase timing starts and are excluded from measured benchmark durations.

## 9. Latency-Chain Construction Contract

`setup_latency_chain` (`src/core/memory/memory_utils.cpp`) builds pointer chains used by main/cache latency tests.

### 9.1 Chain Construction Modes

The `LatencyChainMode` enum (from `src/core/memory/memory_utils.h`) defines four explicit modes plus `Auto`:

- `Auto` (0, default): Resolves to effective mode based on `tlb_locality_bytes`:
  - If `tlb_locality_bytes == 0`, behaves as `GlobalRandom`.
  - If `tlb_locality_bytes > 0`, behaves as `RandomInBoxRandomBox`.
- `GlobalRandom`: Global random permutation across entire buffer (ignores locality).
- `RandomInBoxRandomBox`: Randomize within locality windows, then randomize window order.
- `SameRandomInBoxIncreasingBox`: Locality windows grow (doubling each step), randomization within each box.
- `DiffRandomInBoxIncreasingBox`: Randomization offset by locality window step.

### 9.2 Key Properties

- Uses stride-spaced pointer slots across buffer.
- Requires at least two pointers.
- Produces a circular linked structure.
- Collects chain diagnostics (pointer count, unique pages touched, page size, stride).

### 9.3 Randomization Behavior

- `tlb_locality_bytes == 0`: global random permutation (mode-independent).
- `tlb_locality_bytes > 0`: randomization within locality windows, then randomized window order (specific behavior mode-dependent).

### 9.4 Purpose

- Preserve dependent load-to-use semantics.
- Reduce prefetch predictability.
- Allow controlled locality pressure experiments.

## 10. Warmup Subsystem

Warmup functions (`src/warmup`) run before measured tests.

- Bandwidth warmups are multi-threaded and adaptive in size.
- Latency warmups are single-threaded page-touch prefault style, avoiding chain execution.
- Worker and/or single-thread warmups attempt high QoS class best-effort.

Adaptive warmup size (`warmup_internal.h`):

- `min(buffer_size, max(64MB, 10% of buffer_size))`.

## 10.1 Auto TLB Breakdown (when `latency_chain_mode == Auto`)

When the user selects or defaults to `latency_chain_mode=Auto`, the implementation runs two latency-chain passes to decompose memory latency into TLB-related components:

1. **TLB-hit pass**: Chain with 16 KB locality window (within typical TLB entry footprint).
2. **Global-random pass**: Chain with global randomization (no locality).

Results from both passes are stored in `BenchmarkResults` and `BenchmarkStatistics`:

- `tlb_hit_latency_ns`: Average latency from 16 KB locality pass.
- `tlb_miss_latency_ns`: Average latency from global-random pass.
- `page_walk_penalty_ns`: `tlb_miss_latency_ns - tlb_hit_latency_ns` (approximate TLB miss cost).

These breakdown metrics are optionally serialized to JSON under `main_memory.latency.auto_tlb_breakdown`.

## 11. Standard Benchmark Execution

Standard mode coordinator: `run_all_benchmarks` -> `run_single_benchmark_loop`.

Per loop, conditional by flags:

1. Main-memory bandwidth tests (read/write/copy).
2. Cache bandwidth tests (L1/L2 or custom).
3. Cache latency tests (L1/L2 or custom).
4. Main-memory latency test.

Important execution semantics:

- Phase-local buffers are allocated and initialized immediately before each phase and released after the phase, reducing standard-mode peak footprint.
- Cache bandwidth uses `iterations * CACHE_ITERATIONS_MULTIPLIER` (saturated) for stability.
- Cache bandwidth defaults to single-thread unless user explicitly provides `-threads`.
- Main-memory latency headline is computed from one continuous chase pass.
- If latency samples are enabled, sample collection runs in a separate pass.

## 12. Pattern Benchmark Execution

Pattern mode coordinator: `run_pattern_benchmarks` (`src/pattern_benchmark/pattern_coordinator.cpp`).

Executed pattern families:

- Sequential forward.
- Sequential reverse.
- Strided 64B.
- Strided 4096B.
- Strided 16384B.
- Strided 2MB.
- Random uniform.

Each pattern reports read/write/copy bandwidth metrics.

Implementation notes:

- Random indices generated once per pattern loop for random benchmarks.
- Large-stride patterns can be skipped when constraints invalidate execution.
- Pattern statistics are aggregated across outer loop count.

## 13. Assembly Kernel Layer

Assembly entrypoints are declared in `src/asm/asm_functions.h` and used by benchmark/warmup code:

- Main-memory kernels: read, write, copy (non-temporal stores).
- Cache kernels: read, write, copy (cache-focused variants).
- Latency kernel: pointer-chase loop.
- Pattern kernels: reverse (read/write/copy), strided generic (read/write/copy), strided fixed-stride variants (64B, 4096B, 16384B, 2MB — each as read/write/copy), random (read/write/copy).
- Core-to-core kernels: initiator and responder round-trip loops.

Design intent:

- Keep hot loops in ARM64 Apple Silicon assembly for predictable overhead and high throughput.
- Follow AAPCS64 conventions for register preservation and call boundaries.
- Use checksum sinks in read paths to keep loads architecturally meaningful.

Latency kernel (`memory_latency_chase_asm`) performs strictly dependent pointer chasing and returns terminal pointer to prevent dead-code elimination.

Core-to-core kernels:

- `core_to_core_initiator_round_trips_asm`: Waits for responder token, sends initiator token; repeats for specified round trips.
- `core_to_core_responder_round_trips_asm`: Mirrors initiator; coordinates via shared token word.

## 14. Timing Model

Timing API: `HighResTimer` (`src/core/timing/timer.*`).

- Provides second and nanosecond stop methods.
- Used for both macro (test durations) and micro (latency sample windows) timing.
- Factory creation returns optional; failure is treated as fatal at call sites.

## 15. Statistics and Aggregation

`BenchmarkStatistics` and pattern statistics collect vectors per metric across outer loops.

- Single loop: direct value reporting.
- Multiple loops: aggregate statistics printed and optionally serialized.
- Statistics include central tendency and percentile-oriented summaries.

For contention-prone environments, percentiles (P50/P95/P99) are more informative than mean-only interpretation.

### 15.1 Statistics Fields

Computed from collected value vectors:

- **Average**: Mean of all values.
- **Median (P50)**: 50th percentile value.
- **P90**: 90th percentile value.
- **P95**: 95th percentile value.
- **P99**: 99th percentile value.
- **Stddev**: Standard deviation.
- **Min**: Minimum observed value.
- **Max**: Maximum observed value.

## 16. Console Output Contract

Console rendering is centralized in `src/output/console` and message helpers in `src/output/console/messages`.

Contract highlights:

- Configuration and cache info printed before execution.
- Per-loop results are printed in standard mode.
- Pattern mode prints pattern table-style sections and derived efficiency indicators.
- Aggregate statistics printed when loop count > 1.
- Errors and warnings use `Messages::error_prefix()` / `Messages::warning_prefix()` conventions.

## 17. JSON Output Contract

JSON writer API (`src/output/json/json_output/json_output.cpp`):

- Standard mode: `configuration`, `execution_time_sec`, `main_memory`, `cache`, `timestamp`, `version`.
- Pattern mode: `configuration`, `execution_time_sec`, `patterns`, `timestamp`, `version`.

### 17.1 Configuration keys

In addition to standard fields (buffer size, iterations, loop count, thread count, CPU/OS info):

- `latency_chain_mode` (string): Resolved pointer-chain construction mode.
- `use_latency_tlb_locality` (boolean): Whether TLB-locality window is in use.
- `latency_tlb_locality_bytes` (number): TLB-locality window size in bytes.
- `latency_tlb_locality_kb` (number): TLB-locality window size in KB.

### 17.2 Main-memory latency keys

- `main_memory.latency.average_ns.values` (array): Latency sample values per loop.
- `main_memory.latency.average_ns.statistics` (object, optional): Aggregate stats (average, median, P90, P95, P99, stddev, min, max).
- `main_memory.latency.samples_ns.values` (array, optional): Per-sample latency values.
- `main_memory.latency.samples_ns.statistics` (object, optional): Per-sample aggregate statistics.
- `main_memory.latency.chain_diagnostics` (object): Pointer chain metadata (pointer_count, unique_pages_touched, page_size_bytes, stride_bytes).
- `main_memory.latency.auto_tlb_breakdown` (object, optional): TLB breakdown when `latency_chain_mode==Auto`:
  - `tlb_hit_ns.values` (array): TLB-hit latency per loop.
  - `tlb_miss_ns.values` (array): Global-random latency per loop.
  - `page_walk_penalty_ns.values` (array): Approximate TLB miss cost.

### 17.3 Structure conventions

- Ordered JSON is used for stable key order.
- Bandwidth metrics are nested as arrays in `values` with optional `statistics`.
- Latency is nested as described in §17.2.

### 17.4 Path behavior

- Relative `-output` paths are resolved against current working directory.

## 18. Error-Handling Model

This codebase uses boundary-aware mixed error handling:

- Program orchestration and most modules: return codes (`EXIT_SUCCESS`/`EXIT_FAILURE`).
- Argument parsing internals: exceptions for parsing/validation, converted to return codes at API boundary.
- Allocation internals: null smart-pointer returns for allocation failure.

Principle: no uncaught exceptions should escape to `main()` control flow.

## 19. Concurrency Model

- Bandwidth and pattern bandwidth paths are parallelized by thread-count configuration.
- Latency tests are intentionally single-threaded pointer-chase measurements.
- Cache tests default to single-thread unless user overrides thread count.
- Threaded work partitioning attempts cache-line-aware chunk handling to reduce false sharing effects.

## 20. Measurement Caveats and Interpretation Under Load

The tool can be run on active systems with concurrent workloads, but interpretation must account for scheduler and memory-system contention.

Practical caveats:

- Heavy concurrent activity can inflate tail latency and depress bandwidth.
- Tail metrics (`P95`, `P99`) usually reveal contention more clearly than averages.
- Comparing runs across time requires similar background-load conditions.
- Small buffers can become cache-dominated and hide DRAM behavior.

For high-confidence baselines, run repeated loops and analyze distributions rather than single-point values.

## 21. Verification and Test Expectations

Recommended validation commands:

- Build: `make`
- Unit tests (non-integration): `make test`
- Integration-only: `make test-integration`
- Full test set: `make test-all`
- CLI help smoke check: `memory_benchmark -h`

For narrow changes, prefer targeted `gtest` filters via `./test_runner --gtest_filter=...`.

## 22. Source Map (Primary Entry Points)

- Program entry: `main.cpp`
- Config parse/validate/derive:
  - `src/core/config/argument_parser.cpp`
  - `src/core/config/config_validator.cpp`
  - `src/core/config/buffer_calculator.cpp`
- Memory allocation/init:
  - `src/core/memory/buffer_allocator.cpp`
  - `src/core/memory/buffer_initializer.cpp`
  - `src/core/memory/memory_manager.cpp`
  - `src/core/memory/memory_utils.cpp`
- Standard benchmark:
  - `src/benchmark/benchmark_runner.cpp`
  - `src/benchmark/benchmark_executor.cpp`
  - `src/benchmark/bandwidth_tests.cpp`
  - `src/benchmark/latency_tests.cpp`
- Pattern benchmark:
  - `src/pattern_benchmark/pattern_coordinator.cpp`
  - `src/pattern_benchmark/output.cpp`
- Output:
  - `src/output/console/output_printer.cpp`
  - `src/output/json/json_output/*.cpp`
- Assembly kernels:
  - `src/asm/*.s`
  - `src/asm/core_to_core_latency.s` (core-to-core latency measurements; see [CORE_TO_CORE_WHITEPAPER.md](CORE_TO_CORE_WHITEPAPER.md))

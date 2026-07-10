# macOS Memory Benchmark - Technical Specification

## 1. Scope and Status

This document specifies the current implementation in this repository (version series `0.58.x`) for `memory_benchmark` on macOS Apple Silicon.

It is intentionally implementation-driven and reflects real behavior in code paths under `main.cpp`, `src/core`, `src/benchmark`, `src/pattern_benchmark`, `src/output`, and `src/asm`.

Primary goals:

- Define runtime architecture and execution flow.
- Define command/config semantics and validation rules.
- Define memory allocation, initialization, and benchmark execution contracts.
- Define output contracts (console and JSON).
- Capture current constraints, known drift, and measurement caveats.

Out of scope:

- Generic memory-performance theory (see [LATENCY_WHITEPAPER.md](LATENCY_WHITEPAPER.md)).
- `--analyze-tlb` methodology details (see [TLB_ANALYSIS_WHITEPAPER.md](TLB_ANALYSIS_WHITEPAPER.md)).
- `--analyze-core2core` methodology details (see [CORE_TO_CORE_WHITEPAPER.md](CORE_TO_CORE_WHITEPAPER.md)).
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

Standalone modes (`--analyze-tlb`, `--analyze-core2core`) are dispatched early and use dedicated runners.
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
  - `--analyze-tlb`
  - `--analyze-core2core`
- Cache behavior: auto L1/L2 or user-provided `--cache-size`.
- Latency sampling: `latency_sample_count`.
- Latency-chain construction mode:
  - `latency_chain_mode` (type `LatencyChainMode`, CLI flag `--latency-chain-mode`)
  - `user_specified_latency_chain_mode` flag
- TLB-locality control for latency chain construction:
  - `latency_tlb_locality_bytes` (default 1024 KB)
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
  - First pass extracts `--cache-size` early.
  - Second pass parses remaining options.
- Parser may throw internally (`std::stoll`/validation) but converts to return-code failures at function boundary.
- Help (`-h`, `--help`) prints usage and exits successfully.
- `--latency-chain-mode` accepts string values and resolves to `LatencyChainMode` enum.
- `--analyze-tlb` uses an early dedicated parse branch in `argument_parser.cpp`. It only allows optional `--output`, `--latency-stride-bytes`, `--latency-chain-mode`, `--tlb-density`, `--seed`, `--sweep`, and `--sweep-max-runs`. TLB sweep supports `latency-stride-bytes`, `latency-chain-mode`, and `tlb-density`; its default run guard is `16`, and `global-random` chain mode is rejected. One generated or user-provided seed drives the pure sweep planner, seeded cyclic Latin round scheduler, derived task seeds, layout-specific page-native chain permutations, and deterministic convergence bootstrap. Each task measures a verified one-node-per-page spread chain and an equal-cache-line packed control in the same round. A pilot calibrates whole-chain accesses toward the quick/standard/exhaustive target duration; rounds stop at the per-point CI-width target or profile maximum. Candidate buffers are admitted only when their predicted buffer-plus-scratch peak fits the available-memory budget. Full methodology and JSON contract: [TLB_ANALYSIS_WHITEPAPER.md](TLB_ANALYSIS_WHITEPAPER.md).
- `--analyze-core2core` uses dedicated mode parsing (outside `argument_parser.cpp`) and only allows optional `--output`, `--count`, `--latency-samples`, `--sweep`, and `--sweep-max-runs`. Its mode-specific loop default is `3`; the general loop default remains `1`. Core-to-core sweep supports `count` and `latency-samples`, rejects duplicate sweep keys, and atomically checkpoints the combined output after every run. Direct execution prepares/restores the benchmark signal mask before creating workers. Each scheduler-hint scenario runs an excluded pilot after a 1,000,000-round-trip calibration warmup, reuses its duration-calibrated plan across measured loops, and participates in a cyclic Latin-square scenario schedule. Full methodology and JSON schema 2 contract: [CORE_TO_CORE_WHITEPAPER.md](CORE_TO_CORE_WHITEPAPER.md).

### 6.2 Validation behavior (`config_validator.cpp`)

Validation rejects incompatible flag combinations and invalid value states:

- `--only-bandwidth` and `--only-latency` are mutually exclusive.
- `--patterns` cannot be combined with `--only-bandwidth` or `--only-latency`.
- `--only-bandwidth` cannot be combined with `--cache-size` and cannot use latency-sample overrides.
- `--only-latency` cannot be combined with `--iterations` override.

Zero-disabling semantics (supported only in `--only-latency`):

- `--buffer-size 0` disables main-memory latency path.
- `--cache-size 0` disables cache-latency path.
- Both cannot be zero simultaneously in `--only-latency`.

TLB-locality constraints:

- Non-zero `--latency-tlb-locality-kb` must be a multiple of system page size.
- Non-zero locality window must span at least two latency-stride steps.

Latency stride constraints:

- `--latency-stride-bytes` must be greater than zero.
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
- `--cache-size 0` (in allowed mode) produces zero custom cache buffer.

### 7.2 Latency access counts

- Main-memory latency accesses scale from base count relative to default buffer size.
- Cache latency access counts use fixed constants (`L1`, `L2`, `CUSTOM`).
- `--buffer-size 0` (in allowed mode) sets main latency accesses to zero.

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

## 10.1 Automatic Locality Comparison

When standard main-memory locality is not explicitly supplied, the executor runs three paired rounds comparing a
16 KiB-locality chain with a global-random chain. The first-measured layout alternates by round. Both chains use recorded,
domain-separated seeds and the calibrated complete-chain access count. Results retain both raw point values and the
same-round `global - locality` deltas; the delta headline is the median of paired deltas.

Schema 2 serializes these as `locality_16k_latency_ns`, `global_random_latency_ns`, and
`locality_latency_delta_ns` under `main_memory.latency.automatic_locality_comparison`. The comparison does not isolate
page-table walks and must not be used as a substitute for `--analyze-tlb`.

## 11. Standard Benchmark Execution

Standard mode coordinator: `run_all_benchmarks` -> `run_single_benchmark_loop`.

Enabled phase groups are main bandwidth, cache bandwidth, cache latency, and main latency. Their order rotates by outer
loop index using a deterministic cyclic Latin schedule. Read/write/copy order rotates independently by loop. Each
measurement records its phase and operation position.

Important execution semantics:

- Phase-local buffers are allocated and initialized immediately before each phase and released after the phase, reducing standard-mode peak footprint.
- `benchmark_work_plan` finalizes cache-line-aligned worker boundaries, effective workers, passes/accesses, and exact
  payload before execution. Executors consume those boundaries unchanged; copy payload counts both read and write.
- Omitted `--iterations` uses an excluded same-shape pilot to target 150 ms, with a 100–250 ms intended window and at
  most two corrections. Explicit iterations are exact. Resolved per-target/per-operation work is reused across loops.
- Cache bandwidth defaults to single-thread unless user explicitly provides `--threads`.
- All latency targets use one continuous headline pass calibrated toward 250 ms, evaluated against a 100–300 ms
  window, and rounded to at least 16 complete chain cycles. A cycle-minimum-limited overrun is classified explicitly;
  samples run separately and continue between windows.
- One command-level seed derives target/layout seeds; repeated loops rebuild equivalent logical chains.
- Each operation has explicit measurement status and optional value. Interrupted/incomplete work is excluded from
  aggregate vectors. Standard JSON is atomically checkpointed after completed loops.

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
- **CV**: Sample standard deviation divided by the absolute mean, as a percentage when the mean is valid.
- **MAD**: Median absolute deviation from the median.
- **Min**: Minimum observed value.
- **Max**: Maximum observed value.

Standard repeated-loop aggregates set a diagnostic quality warning above 7.5% CV. Values are retained without outlier
filtering, and the warning does not by itself invalidate the result.

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

- Standard mode schema 2: `configuration`, `execution_time_sec`, completion counters/status, `results_complete`,
  per-loop `loops`, `main_memory`, `cache`, `timestamp`, and `version`.
- Pattern mode: `configuration`, `execution_time_sec`, `patterns`, `timestamp`, `version`.
- TLB analysis mode: `configuration`, `execution_time_sec`, `tlb_analysis`, `timestamp`, `version`.
- Core-to-core schema 2: calibrated methodology configuration, `core_to_core_latency` command completion metadata, scenario work plans, nullable aggregate values, per-loop order/status/duration/hint/sample-boundary records, and affinity-comparison interpretability metadata.
- Sweep mode: `configuration.mode = "sweep"`, `configuration.base_mode`, `configuration.sweep_parameters`, top-level status/completion/conclusion fields, `runs[].result`, `execution_time_sec`, `timestamp`, `version`. For `--analyze-tlb --sweep`, `base_mode` is `analyze_tlb` and each `runs[].result` contains a TLB analysis payload. Core-to-core sweeps use `base_mode: "analyze_core2core"` and checkpoint after every run.

### 17.1 Configuration keys

In addition to standard fields (buffer size, iterations, loop count, thread count, CPU/OS info):

- `latency_chain_mode` (string): Resolved pointer-chain construction mode.
- `use_latency_tlb_locality` (boolean): Whether TLB-locality window is in use.
- `latency_tlb_locality_bytes` (number): TLB-locality window size in bytes.
- `latency_tlb_locality_kb` (number): TLB-locality window size in KB.
- `benchmark_schema_version` (number): `2`.
- `methodology_version` (string): `benchmark-v2-calibrated-seeded-balanced`.
- `benchmark_seed` (string): exact uint64 decimal string plus source/encoding fields.
- Calibration targets/windows and phase/operation schedule policies.

### 17.2 Main-memory latency keys

- `main_memory.latency.headline_ns`: status, median-or-single headline, loop values, robust statistics, measurement
  records, quality, and optional pooled separate-sample distribution with loop boundaries.
- `main_memory.latency.automatic_locality_comparison.locality_16k_latency_ns`.
- `main_memory.latency.automatic_locality_comparison.global_random_latency_ns`.
- `main_memory.latency.automatic_locality_comparison.locality_latency_delta_ns`.
- `loops[].measurements`: nullable per-loop status/value plus exact work, worker, seed, timing, calibration, and order
  metadata.

### 17.3 Structure conventions

- Ordered JSON is used for stable key order.
- Aggregate `value` is a single measured loop or median P50; `values` contains only measured loop headlines.
- Statistics include average, median, P90/P95/P99, sample stddev, CV, MAD, min, and max.
- Unavailable measurements use `null` plus status/reason, never numeric zero.

### 17.4 Core-to-core schema 2

- `configuration.schema_version` (number): `2`.
- `configuration.methodology_version` (string): `core2core-v2-calibrated-balanced-auditable`.
- Calibration metadata: excluded 100,000-round-trip pilot after a 1,000,000-round-trip calibration warmup; 25 ms
  final warmup target, 250 ms continuous headline target with a 100-300 ms intended window, and 1 ms sample-window
  target. Minimum work is 20,000/1,000,000/2,000 round trips respectively.
- `core_to_core_latency.status`, `planned_measurements`, `completed_measurements`, and `measurements_complete` describe
  command completion.
- Each scenario contains `status`, `status_reason`, planned/completed loops, a calibrated `work_plan`, continuous
  headline values/statistics, a nullable median headline, a distinct pooled `samples_ns` distribution, and
  `loop_records`.
- Loop records retain cyclic schedule position, status/reason, nullable round-trip and one-way estimates, headline
  duration/quality, pooled-sample index range, and both workers' observed QoS/affinity hint outcomes.
- `affinity_hint_comparison_interpretable` is true only for a complete command when every requested hint was applied in
  every measured record. It does not imply hard core pinning.
- Invalid, failed, interrupted, and not-run measurements never become numeric zeroes.

### 17.5 Path behavior

- Relative `--output` paths are resolved against current working directory.

## 18. Error-Handling Model

This codebase uses boundary-aware mixed error handling:

- Program orchestration and most modules: return codes (`EXIT_SUCCESS`/`EXIT_FAILURE`).
- Argument parsing internals: exceptions for parsing/validation, converted to return codes at API boundary.
- Allocation internals: null smart-pointer returns for allocation failure.

Principle: no uncaught exceptions should escape to `main()` control flow.

## 19. Concurrency Model

- Bandwidth and pattern bandwidth paths are parallelized by thread--count configuration.
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
- Standalone TLB planning and scheduling:
  - `src/benchmark/tlb_sweep_planner.cpp`
  - `src/benchmark/tlb_measurement_scheduler.cpp`
  - `src/benchmark/tlb_analysis.cpp`
- Pattern benchmark:
  - `src/pattern_benchmark/pattern_coordinator.cpp`
  - `src/pattern_benchmark/output.cpp`
- Output:
  - `src/output/console/output_printer.cpp`
  - `src/output/json/json_output/*.cpp`
- Assembly kernels:
  - `src/asm/*.s`
  - `src/asm/core_to_core_latency.s` (core-to-core latency measurements; see [CORE_TO_CORE_WHITEPAPER.md](CORE_TO_CORE_WHITEPAPER.md))

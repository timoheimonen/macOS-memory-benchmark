# macOS Memory Benchmark - Technical Specification

## 1. Scope and Status

This document specifies the current implementation in this repository (version `0.63.0`) for `memory_benchmark` on macOS Apple Silicon.

It is intentionally implementation-driven and reflects real behavior in code paths under `main.cpp`, `src/core`,
`src/benchmark`, `src/pattern_benchmark`, `src/gpu_bandwidth`, `src/llm_memory`, `src/output`, and `src/asm`.

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
- Detailed `--gpu-bandwidth` methodology and schema field catalog (see
  [GPU_BANDWIDTH_WHITEPAPER.md](GPU_BANDWIDTH_WHITEPAPER.md)).
- Detailed `--llm-memory` generic schema-v1 vocabulary, active decode/prefill methodologies, and field catalog (see
  [LLM_MEMORY_PROFILE_WHITEPAPER.md](LLM_MEMORY_PROFILE_WHITEPAPER.md)).
- Historical behavior from older releases.

## 2. Platform and Build Constraints

- Target OS: macOS.
- Target CPU architecture: ARM64 Apple Silicon.
- Language: C++17 with ARM64 Apple Silicon assembly kernels and one Objective-C++ Metal backend.
- Build: `Makefile` (`clang++`, `as`).
- Test framework: GoogleTest (`test_runner`).
- Deployment target: macOS 11.0 for production, tests, assembly, and links.
- First-party link dependencies: `-framework Metal -framework Foundation`; no new third-party production dependency.
- Objective-C++ sources are auto-discovered and compiled with ARC. Production/test links share one framework list, and
  coverage includes production `.mm` beside `.cpp`.
- GPU MSL is embedded canonical source compiled at runtime as MSL 2.3. The build does not require an offline `.metallib`,
  binary archive, persistent pipeline cache, or the optional Metal Toolchain component.

The tool is designed and tuned for Apple Silicon execution characteristics (cache hierarchy, page behavior, unified
memory, QoS, ARM64 assembly, and Metal compute). GPU mode requires a default Metal device with unified memory and
`supportsFamily(MTLGPUFamilyApple7)`. This is a capability boundary, not a throughput baseline.

## 3. Hardware Limitations

### 3.1 Thermal constraints on fanless systems

Fanless Apple Silicon systems have limited thermal headroom and may enter elevated thermal states during sustained
benchmarking. The transition depends on the device, starting temperature, ambient conditions, background activity, and
workload; this repository does not define a model-specific time-to-throttle or run-count threshold.

Impact:

- Repeated back-to-back benchmark executions may encounter thermal limiting.
- Measurements from throttled runs should not be compared with baseline measurements from thermal-normal conditions.
- For consistent results on fanless systems, allow sufficient idle time between consecutive benchmark runs to permit thermal cool-down.

Recommendation:

- For reliable baseline measurement on fanless systems, run single benchmark iterations with thermal cool-down intervals between runs.
- Use `caffeinate -i -d ./memory_benchmark ...` to prevent system sleep during longer measurement sessions on cooler systems.

## 4. High-Level Runtime Architecture

Main orchestration (`main.cpp`) follows this pipeline:

`select_primary_benchmark_mode` scans all mode flags before mode-specific parsing, so a command containing multiple
primary modes
fails deterministically instead of being routed by the first token. Core-to-core uses `CoreToCoreLatencyConfig`; GPU uses
`GpuBandwidthConfig`; LLM-memory uses `LlmMemoryConfig`; TLB uses a dedicated branch that populates `BenchmarkConfig`.
GPU and LLM-memory are dispatched before the general timer/parser pipeline and never call general CPU config validation
or buffer/access derivation. The numbered pipeline below
applies to standard/pattern execution; TLB and general sweeps share its parse, output-session, validation, and
final-output boundaries before entering their dedicated collectors/coordinators.

1. Create high-resolution total-execution timer.
2. Parse CLI arguments into `BenchmarkConfig` (`parse_arguments`) and finish human-facing help/no-mode handling.
3. Classify the raw output target. Standard/pattern/TLB commands and sweeps using `-` create a command-scoped
   `JsonOutputSession` that routes subsequent human `std::cout` output to stderr while retaining the original stdout
   buffer for final JSON.
4. Validate configuration (`validate_config`) while the supported machine-output routing is active.
5. Calculate derived sizes and counts:
   - `calculate_buffer_sizes`
   - `calculate_access_counts` (fallback latency access counts)
   - `calculate_total_allocation_bytes`
6. Print resolved configuration and cache info.
7. Raise main thread QoS (`QOS_CLASS_USER_INTERACTIVE`) best-effort.
8. Execute one mode:
   - Standard benchmark mode (`run_all_benchmarks`), which allocates/initializes buffers per phase and releases them after the phase.
   - Pattern benchmark mode (`run_all_pattern_benchmarks`), whose command-level orchestration in
     `pattern_statistics_manager.cpp` owns and prepares one shared `PatternBuffers` source/destination pair for every
     pattern and loop; `pattern_coordinator.cpp` executes one outer loop.
9. Print loop results and aggregate statistics.
10. Build any representable terminal payload, dispatch it according to the mode-aware file/stdout lifecycle, and preserve
    execution status separately from output status.
11. Print total elapsed runtime, restore the previous stdout buffer at scope exit, and return the combined status.

Memory cleanup is RAII-based through `MmapPtr` custom deleters (`munmap` on scope exit).

Core-to-core keeps its dedicated parser and configuration. After successful combined parsing/preflight and human help
handling, its command boundary creates the same output session, then either collects one schema-2 direct result or enters
the core-to-core sweep runner. Human output is routed before sweep execution, the runtime banner, and worker threads are
created.

GPU mode follows its own synchronous pipeline:

1. Parse and validate the exact GPU option whitelist, derive buffer bytes/seed, and finish human-facing help handling.
2. Classify the raw output target with preserve-raw file semantics and create the command-scoped output session.
3. Apply shared best-effort main-thread QoS and enter `BenchmarkSignalMaskGuard`.
4. Create the pure-C++ `GpuBackend` factory product and initialize the private Metal backend.
5. Verify default device, unified memory, Apple7 family, runtime MSL compilation/pipelines, `maxBufferLength`, and
   two-buffer memory budget.
6. Allocate two private/tracked data buffers plus one shared/tracked status buffer for the suite lifetime.
7. Resolve excluded automatic calibration or exact explicit work, then freeze read/write/copy plans.
8. Execute cyclic operation tasks. Each logical task is warmup → precondition → one timed command buffer → required
   validation, followed by status/counter/aggregate update and a logical checkpoint. File sessions persist it; stdout
   sessions keep it lazy while preserving the following stop observation.
9. Release Metal resources and record the final allocation/environment snapshot. File sessions perform the existing
   post-release replacement; stdout sessions serialize one terminal schema-1 payload at the command boundary after
   console rendering, then return success/failure according to explicit run and output status.

LLM-memory follows its own synchronous pipeline. CPU decode and prefill are active with contiguous or deterministic
paged KV:

1. Scan the complete standalone whitelist, parsing supplied values and rejecting unknown, duplicate, or missing-value
   input even when help is present. Human help returns after that scan but before required/default/platform work.
2. Require the five common model inputs plus phase-specific decode context or prefill prompt/tile geometry; resolve
   decode-by-default, contiguous-by-default or explicitly paged layout, requested/detected
   workers, explicit or automatic work, and one user/generated base seed; run checked geometry and per-scenario
   work-limit preflight. Paged layout requires exactly one valid block-token size, while contiguous rejects it.
3. Classify the raw output target and install one command-scoped `JsonOutputSession` before the runtime banner.
4. Request best-effort main-thread QoS, enter `BenchmarkSignalMaskGuard`, and capture CPU/OS/page/cache/available-memory
   plus initial thermal/Low Power Mode evidence.
5. Create the selected pure-C++ `LlmBackend` factory product. Build a preliminary pointer-free logical work plan and ask
   the backend and runner for their exact auxiliary backing, then add the conservative JSON DOM/serialization peak for a
   non-empty output target. Its checked variable-string term includes component/layout identities and all prefill
   execution/scenario/scope identities. Frozen identities are charged once, while scenario-plan identities scale with
   the maximum retained calibration attempts and planned measurement loops. Preliminary and finalized plans use the
   same canonical identity-size formula. Rebuild and admit
   the final plan with the full page-rounded weight/physical-K/physical-V
   mappings, paged table and validation transient when applicable, descriptor/planner storage, checksum storage, and
   orchestration storage included in one peak budget.
6. Initialize the backend, resolve the plan's exactly one tagged backend execution-plan variant, and prepare resources.
   The CPU adapter creates `HighResTimer`, atomically allocates and initializes/pre-touches the layout-specific mappings,
   materializes and validates the paged table when applicable, materializes the worker-major layout-specific descriptor
   ABI, and binds the matching production ARM64 executor. An initialized lifecycle
   `Unsupported` result remains distinct from a runtime failure and never falls back to another backend.
7. Resolve excluded scenario-specific calibration in canonical weights-only/KV-only/mixed order, or validate exact
   explicit work. Automatic calibration warms the initial pilot shape and adds only the conditional one-work-unit
   confirmation warmup; explicit work does not run a pilot. Atomically freeze all three resolved plans, run their
   same-shape frozen warmups in canonical order, and only then execute cyclic measured tasks through whole synchronous
   backend calls. Each call owns reset, timed work, correctness validation, and generic task evidence. Each terminal
   measurement is offered to the logical checkpoint hook while resources remain live.
8. Release backend resources exactly once before offering the command-terminal checkpoint, then capture final
   environment state, render the centralized console report, and either retain the runner-owned atomic file cadence or
   emit one final schema-1 stdout document. Execution status and output status remain separate.

Pre-run failures before runner-result initialization leave stdout empty. Once initialized,
partial/interrupted/unsupported/failed evidence is representable; a file checkpoint failure is terminal and is not
retried at an outer final-write boundary.

## 5. Configuration Model

Configuration state is represented by `BenchmarkConfig` (`src/core/config/config.h`).

### 5.1 User-facing control fields

- Main options: buffer size MB, iterations, loop count, raw output target, threads. The existing configuration fields are
  named `output_file`, but command adapters classify exact `-` before interpreting every other non-empty value as a
  path. An empty raw value disables JSON for direct commands and is missing/invalid for sweeps.
- General mode/config flags: `run_benchmark`, `run_patterns`, `analyze_tlb`, `only_bandwidth`, and `only_latency`.
- Standalone TLB state remains in `BenchmarkConfig` (`analyze_tlb`, density, seed, stride/chain settings, and common sweep
  fields) and is populated by the dedicated `--analyze-tlb` branch in `argument_parser.cpp`.
- `--analyze-core2core` is pre-routed in `main.cpp` and uses the separate `CoreToCoreLatencyConfig` parser/runner path.
- `--gpu-bandwidth` is pre-routed in `main.cpp` and uses separate `GpuBandwidthConfig`: per-buffer MB/bytes, optional
  explicit passes, loop count, raw output target, base seed/source, help state, and exact argv.
- `--llm-memory` is pre-routed in `main.cpp` and uses separate `LlmMemoryConfig`: resolved backend/phase/layout enums,
  required common and phase-specific model geometry, KV width/batch defaults, requested and detected worker counts,
  optional exact scenario work
  units, loop count, raw output target, base seed/source, help state, and exact argv. The immutable
  `LlmMemoryWorkPlan` separately records common phase-applicable geometry, exact model payload/accounted work, memory
  admission, layout metadata, domain seeds, methodology, exact component identities, and exactly one tagged
  backend-execution-plan variant. The active CPU alternative owns requested/available/effective workers plus its
  descriptor templates and storage accounting; the Metal alternative is reserved for later activation.
- Cache behavior: auto L1/L2 or user-provided `--cache-size`.
- Latency sampling: `latency_sample_count`.
- Latency-chain construction mode: `latency_chain_mode` (type `LatencyChainMode`,
  CLI flag `--latency-chain-mode`).
- TLB-locality control for latency chain construction:
  - `latency_tlb_locality_bytes` (default 1024 KB)
  - `0` means global random chain.
- Best-effort cache-discouraging mode: `use_non_cacheable`.

### 5.2 Derived fields

- Resolved byte sizes for main and cache buffers.
- Fallback access counts for latency paths; normal measured work is resolved by pilot calibration.
- System metadata (CPU name, macOS version, core counts).
- Max memory limits and bookkeeping flags.

## 6. CLI Parsing and Validation

### 6.1 Parsing behavior (`argument_parser.cpp`)

- Two-pass parse:
  - First pass extracts `--cache-size` early.
  - Second pass parses remaining options.
- Strict integer parsing uses `std::from_chars`; parser validation may throw internally, but the function boundary
  converts those failures to return codes.
- Help (`-h`, `--help`) prints usage and exits successfully.
- Standard/pattern/TLB output-target classification occurs only after successful parsing and help/no-mode handling.
  Core-to-core classifies its target after its dedicated combined parse/preflight and help handling; GPU and LLM do the
  same after their dedicated parser and help handling. For every result-producing direct command and supported CPU
  sweep, exact `-` is reserved for final stdout JSON. An empty value disables JSON for a direct command and is
  missing/invalid for a sweep. Every
  other non-empty value is a file target, including `./-` and flag-shaped names such as `-G`.
- `--latency-chain-mode` accepts string values and resolves to `LatencyChainMode` enum.
- `--analyze-tlb` uses an early dedicated parse branch in `argument_parser.cpp`. It only allows optional `--output`, `--latency-stride-bytes`, `--latency-chain-mode`, `--tlb-density`, `--seed`, `--sweep`, and `--sweep-max-runs`. TLB sweep supports `latency-stride-bytes`, `latency-chain-mode`, and `tlb-density`; its default run guard is `16`, and `global-random` chain mode is rejected. One generated or user-provided seed drives the pure sweep planner, seeded cyclic Latin round scheduler, derived task seeds, layout-specific page-native chain permutations, and deterministic convergence bootstrap. Each task measures a verified one-node-per-page spread chain and an equal-cache-line packed control in the same round. A pilot calibrates whole-chain accesses toward the quick/standard/exhaustive target duration; rounds stop at the per-point CI-width target or profile maximum. Candidate buffers are admitted only when their predicted buffer-plus-scratch peak fits the available-memory budget. Full methodology and JSON contract: [TLB_ANALYSIS_WHITEPAPER.md](TLB_ANALYSIS_WHITEPAPER.md).
- `--analyze-core2core` uses dedicated mode parsing (outside `argument_parser.cpp`) and only allows optional `--output`,
  `--count`, `--latency-samples`, `--sweep`, `--sweep-max-runs`, and `--help`. Its mode-specific loop default is `3`;
  the general loop default remains `1`. Core-to-core sweep supports `count` and `latency-samples`, rejects duplicate
  sweep keys, and atomically checkpoints a real-file combined output after every attempted run; an empty run plan or a
  stop observed before a run checkpoints a terminal zero-attempt envelope. Stdout keeps the same state transitions but
  emits only the terminal envelope. Only a nested `status: "complete"` result with `measurements_complete: true`
  increments `completed_runs`.
  Direct and sweep execution use the shared scope-bound signal guard before creating workers and restore the calling
  thread's exact previous mask on every return path. Each scheduler-hint scenario runs an excluded pilot after a
  1,000,000-round-trip warmup intended to reduce pilot startup transients, reuses its duration-calibrated plan across
  measured loops, and participates in a cyclic Latin-square scenario schedule. The result is effective acquire/release
  token-protocol round-trip time, not an isolated physical cache-line migration or coherence-fabric latency. Full
  methodology and JSON schema 2 contract: [CORE_TO_CORE_WHITEPAPER.md](CORE_TO_CORE_WHITEPAPER.md).
- `--gpu-bandwidth` uses a dedicated parser outside `argument_parser.cpp`. It accepts only `-G`/`--gpu-bandwidth`,
  `-b`/`--buffer-size`, `-i`/`--iterations`, `-r`/`--count`, `--seed`, `-o`/`--output`, and
  `-h`/`--help`. Duplicates, unknown/incompatible options, missing values, partial numeric tokens, non-positive
  iterations/count, and signed seeds fail before Metal work. Defaults are 512 MB per buffer, three loops, generated
  seed, and automatic passes. Buffer size must be at least 64 MB; checked MB→bytes overflow and explicit work guardrails
  are resolved before backend creation. Copy's 2× payload makes its pass limit the strict CLI ceiling shared by all three
  operations. GPU schema 1 rejects sweep options.
- `--llm-memory` uses a dedicated parser outside `argument_parser.cpp`. It accepts only `-M`/`--llm-memory`, the five
  required weight/layer/head inputs, phase and its context or prompt/tile geometry, optional KV width, batch,
  `--kv-layout`, `--kv-block-tokens`, threads, iterations, count, seed, output, and help.
  Duplicate/unknown/incompatible options, missing values, non-complete decimal tokens, zero size/count values, invalid
  KV widths, invalid query/KV sharing, checked geometry overflow, impossible one-step payloads, and explicit scenario
  work above the strictest limit fail before output-session creation. Output consumes exactly one opaque raw token.
  Help still completes the whitelist pass but returns before required/default/preparation work. LLM rejects sweeps and
  all cache, latency, TLB, pattern, GPU, and non-cacheable modifiers. Defaults are two-byte KV elements, batch one,
  three loops, detected workers, automatic per-scenario calibration, one generated nonzero seed, decode, and
  contiguous KV. Decode requires positive `--context-tokens` and rejects prefill geometry. Prefill requires positive
  `--prompt-tokens P` and `--attention-query-tile-tokens Q`, requires `Q <= P`, and rejects decode context.
  `--kv-layout` accepts `contiguous|paged`. Paged requires exactly one `--kv-block-tokens <G>`; contiguous rejects that
  option. `G` must be a positive power of two no greater than `UINT32_MAX`, and it may exceed the phase length. The
  cross-option rules are order-independent. All four CPU phase/layout combinations are public. Metal remains
  unavailable, and no request receives fallback execution. Explicit work must fit both the
  one-billion-work-unit and 64 GiB task-accounted-byte ceilings for all three scenarios.

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

- When the effective chain mode uses locality, `--latency-tlb-locality-kb` must be a non-zero multiple of system page
  size and span at least two latency-stride steps. Explicit `GlobalRandom` ignores the configured locality value.

Latency stride constraints:

- `--latency-stride-bytes` must be greater than zero.
- Stride must be pointer-size aligned.

Memory-limit model:

- System available memory is queried.
- Global cap uses `MEMORY_LIMIT_FACTOR` (80%).
- Per-main-buffer cap is mode-aware (1 or 2 main buffers, depending on active mode/phase needs).
- A second peak-concurrent allocation check validates the highest active phase footprint (main + cache paths).

GPU validation is separate from `config_validator.cpp`:

- The requested buffer is never silently reduced. Each data buffer must fit `MTLDevice.maxBufferLength`, and
  `2 × buffer + 4096 auxiliary bytes` must fit 80% of the project's available-memory estimate. When that estimate is
  zero, the existing 2048 MiB fallback total budget applies.
- `recommendedMaxWorkingSetSize` is advisory and serialized with signed byte/relative headroom plus an exceeded flag; it
  is not the hard allocation limit.
- Each measured plan is bounded by 16,384 dispatches and 64 GiB exact payload. Read/write count one buffer per pass;
  copy counts two.
- Device initialization is unsupported without a default Metal device, unified memory, or Apple7-family capability.
  Unknown future device names are not rejected when required capabilities succeed.

## 7. Size and Access Derivation

### 7.1 Buffer sizing (`buffer_calculator.cpp`)

- Main buffer size is derived from `buffer_size_mb`.
- L1/L2/custom cache test buffers use factor constants currently set to `1.0`.
- Cache buffers are rounded to active latency stride granularity (`latency_stride_bytes`, default `256`) and minimum constraints.
- Minimum practical lower bound includes page-size enforcement.
- `--cache-size 0` (in allowed mode) produces zero custom cache buffer.

### 7.2 Latency access counts

- `calculate_access_counts` populates fallback desired counts; it does not normally determine the measured latency work.
- Standard latency execution runs an excluded whole-chain pilot, calibrates a duration-targeted
  `BenchmarkLatencyWorkPlan`, rounds to complete chain cycles, and reuses the resolved plan across outer loops.
- The main-memory fallback scales from a base count relative to the default buffer size. Cache fallbacks use fixed
  `L1`, `L2`, and `CUSTOM` constants.
- `--buffer-size 0` (in allowed mode) sets the main-memory fallback to zero and disables that latency target.

## 8. Memory Allocation and Initialization

### 8.1 Allocation strategy

Allocation entrypoints:

- Standard benchmark mode: per-phase allocators in `src/benchmark/benchmark_executor.cpp` (`prepare_*_buffers` helpers).
- Pattern benchmark mode: `run_all_pattern_benchmarks` in `pattern_statistics_manager.cpp` calls
  `allocate_pattern_buffers` (`src/core/memory/buffer_allocator.cpp`), retains the pair for the full command, and
  releases it on return.

Shared allocation behavior:

- Uses `mmap` anonymous private mappings (macOS-specific behavior and limits apply).
- Uses phase- or mode-specific allocation to avoid unused buffers.
- Performs overflow-safe byte arithmetic before allocation.
- Enforces global memory-limit checks from peak-concurrent requirements.

Allocated buffer families (conditional):

- Main bandwidth: `src`, `dst`.
- Main latency: `lat`.
- Cache latency: `l1/l2` or `custom`.
- Cache bandwidth: `l1_bw_src/dst`, `l2_bw_src/dst` or `custom_bw_src/dst`.

Pattern mode intentionally allocates and uses only main source/destination buffers.

GPU mode does not use `mmap` buffers. The Metal backend allocates once before calibration and retains for the full suite:

- `buffer_a` and `buffer_b`: each exactly the requested size with
  `MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeTracked`.
- `status_buffer`: 4096 bytes with
  `MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked` for timed/validation checksums.

The backend reads back and serializes actual storage, CPU-cache, hazard-tracking, resource-option, label, and length
metadata. On Apple Silicon, private resources remain allocations in unified system memory and must not be documented as
separate VRAM. Partial allocation is released, and `currentAllocatedSize` is captured before allocation, at suite peak,
and after release.

LLM mode allocates suite-lifetime anonymous private `mmap` regions through its executor: active weights and either
logical contiguous K/V or full physical paged K/V pools. Paged execution also owns one shared row-major `uint32_t`
block table. Each mapping is rounded separately to native page granularity for committed-byte accounting, while hot
descriptors cover exact requested physical lengths. Logical bytes, layout padding, page rounding, table bytes, and
transient preparation storage remain separate accounting classes. The mappings are regular cacheable memory; the
standalone whitelist rejects `--non-cacheable`.

Allocation is atomic from the caller's perspective. The finalized pointer-free layout is revalidated and the exact
executor/runner auxiliary backing is re-admitted before the first mapping. Any partial candidate is released and the
command emits no new result payload on failure. The budget includes mappings, descriptor arrays, retained planner
templates, paged table plus its bounded bijection-validation bitset, static-reference and expected/actual checksum
arrays, thread/worker state, result/calibration records, statistics, and orchestration storage. The bitset and other
preparation transients are released before timed work. With non-empty file/stdout output, orchestration storage includes
a conservative estimate for one live schema DOM plus serialized transport text; disabled output contributes a valid
zero estimate.
Workload sizes are rejected rather than silently reduced.

### 8.2 Best-effort non-cacheable mode

- `allocate_buffer_non_cacheable` still uses normal user-space mappings.
- Applies `madvise(MADV_RANDOM)` hints on macOS.
- This is best-effort cache discouragement only; not true uncached memory.

### 8.3 Initialization strategy

Initialization entrypoints:

- Standard benchmark mode: per-phase initialization in `run_single_benchmark_loop` before each measured phase.
- Pattern benchmark mode: `initialize_pattern_buffers` (`src/core/memory/buffer_initializer.cpp`).

Initialization semantics:

- Bandwidth buffers: deterministic source pattern + zeroed destination.
- Latency buffers: deterministically seeded, randomized pointer-chasing circular chain via `setup_latency_chain`.
- Allocation/initialization happen before phase timing starts and are excluded from measured benchmark durations.

GPU initialization/precondition is compute-based and deterministic. Read fills A with a seed-derived source pattern;
write poisons A before the timed kernel writes a pass-derived pattern; copy fills A with source data and B with poison.
Every excluded calibration attempt and measured task runs a same-shape warmup and then restores this deterministic state
before timing. Timed/final dual checksum words are reset outside the primary duration. GPU caches are not flushed, so the
contract is steady-state warm-memory.

LLM preparation writes every requested weight byte and every physical K/V byte exactly once with the applicable
versioned pattern, pre-touching complete mappings including paged suffix padding. The paged physical pattern depends on
pool, physical block ID, and physical offset. Static checksum references are accumulated during those writes; there is
no separate reference-reading pass. Paged preparation derives one deterministic block-table permutation, validates its
range and bijection, hashes its explicit little-endian entries, and makes the table read-only before worker-major
layout-specific descriptors are materialized from immutable pointer-free ranges. Initialization, table preparation,
descriptor construction/validation, canary setup, and resulting page faults precede calibration and measurement and are
excluded from elapsed scenario time. Before each paged task, only the mutable current-token K/V append slots are
restored to the physical initialization pattern; this allocation-free reset is excluded from timing and leaves history
blocks and suffix-padding canaries unchanged.

## 9. Latency-Chain Construction Contract

`setup_latency_chain` (`src/core/memory/memory_utils.cpp`) builds pointer chains used by main/cache latency tests.

### 9.1 Chain Construction Modes

The `LatencyChainMode` enum (from `src/core/memory/memory_utils.h`) defines four explicit modes plus `Auto`:

- `Auto` (0, default): Resolves to effective mode based on `tlb_locality_bytes`:
  - If `tlb_locality_bytes == 0`, behaves as `GlobalRandom`.
  - If `tlb_locality_bytes > 0`, behaves as `RandomInBoxRandomBox`.
- `GlobalRandom`: Global random permutation across entire buffer (ignores locality).
- `RandomInBoxRandomBox`: Randomize within locality windows, then randomize window order.
- `SameRandomInBoxIncreasingBox`: Reuses one random within-window permutation for every locality window and visits
  windows in increasing address order.
- `DiffRandomInBoxIncreasingBox`: Generates an independent random within-window permutation for each locality window
  and visits windows in increasing address order.

### 9.2 Key Properties

- Uses stride-spaced pointer slots across buffer.
- Requires at least two pointers.
- Produces a circular linked structure.
- Supports optional chain diagnostics (pointer count, unique pages touched, page size, stride)
  when a caller supplies an output object; production benchmark setup does not request them.

### 9.3 Randomization Behavior

- `tlb_locality_bytes == 0`: `Auto` resolves to `GlobalRandom`, and explicit `GlobalRandom` is valid. Explicit
  locality-using box modes are rejected because they require a non-zero locality window.
- `tlb_locality_bytes > 0`: mode-specific randomization within locality windows; `RandomInBoxRandomBox` also shuffles
  window order, while the two increasing-box modes visit windows in increasing address order.

### 9.4 Purpose

- Preserve dependent load-to-use semantics.
- Reduce prefetch predictability.
- Allow controlled locality pressure experiments.

## 10. Warmup Subsystem

Warmup functions (`src/warmup`) run before measured tests.

- Main-memory bandwidth warmups are multi-threaded and use the adaptive byte limit
  `min(buffer_size, max(64 MiB, 10% of buffer_size))` from `warmup_internal.h`.
- Cache-bandwidth warmups are multi-threaded and touch the full cache-sized buffer.
- Latency warmups are single-threaded page-prefault passes that read and write one byte per native page; they do not
  execute the pointer chain.
- Sequential pattern warmups use the main-memory adaptive policy. Strided warmups execute one complete 32-byte phase
  period over the finalized worker partitions, while random warmups traverse each finalized worker's index list once.
- Worker and/or single-thread warmups attempt high QoS class best-effort.

## 10.1 Automatic Locality Comparison

When standard main-memory locality is not explicitly supplied, the executor runs three paired rounds comparing a
16 KiB-locality chain with a global-random chain. The first-measured layout alternates by round. Both chains use recorded,
domain-separated seeds and the calibrated complete-chain access count. Results retain both raw point values and the
same-round `global - locality` deltas; the delta headline is the median of paired deltas.

This auxiliary fixed-window comparison requires at least two stride-spaced nodes inside 16 KiB (stride at most 8192
bytes). It is not part of config-validator target eligibility: if an otherwise valid standard configuration uses a larger
stride, the comparison records unavailable measurements while the configured target measurements can still run. Because
the three comparison measurements remain part of the loop's plan, that loop is `partial` and top-level
`results_complete` remains `false`.

Current standard schema 3 serializes these as `locality_16k_latency_ns`, `global_random_latency_ns`, and
`locality_latency_delta_ns` under
`main_memory.latency.automatic_locality_comparison`. The comparison does not isolate page-table walks and must not be
used as a substitute for `--analyze-tlb`.

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
  the CLI then runs a separate sample pass with 1,000 windows by default. `--latency-samples` selects a positive window
  count, and sampling continues from one window's terminal pointer to the next without defining the headline.
- One command-level seed derives target/layout seeds; repeated loops rebuild equivalent logical chains.
- Each operation has explicit measurement status and optional value. Interrupted/incomplete work is excluded from
  aggregate vectors. A standard file target is atomically checkpointed after completed loop-state changes. Stdout
  checkpoint calls are lazy no-ops, and direct `--output -` receives one final schema-3 payload, including representable
  interrupted or failed evidence, before the execution status is returned.

## 12. Pattern Benchmark Execution

Command-level pattern orchestration is `run_all_pattern_benchmarks`
(`src/pattern_benchmark/pattern_statistics_manager.cpp`): it owns allocation, initialization, outer-loop control,
interruption handling, and aggregation. `run_pattern_benchmarks` (`src/pattern_benchmark/pattern_coordinator.cpp`)
executes one outer loop and coordinates its seven pattern families.

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

- Random offsets are generated once per pattern loop as a deterministic, aligned, no-replacement permutation prefix
  from the command-level pattern seed. The same finalized offsets are reused by read, write, and copy; repeated
  `--count` loops reconstruct the same workload.
- Omitted `--iterations` gives every read/write/copy operation an excluded same-shape pilot and automatic calibration
  toward 150 ms (100-250 ms intended window). An explicit value is the exact measured pass count.
- Pattern order rotates across outer loops. Strided plans also rotate the 32-byte starting phase by pass and record
  phase-aware access and payload totals.
- Requested workers may be reduced when a small buffer or large stride cannot give every worker a genuine stride
  transition. JSON records requested and effective workers.
- Large-stride patterns can be skipped when constraints invalidate execution.
- Pattern statistics are aggregated across completed outer-loop measurements; unavailable operations are excluded.
- Direct pattern output is built before propagating the run status. A file target keeps the existing final atomic write;
  exact `--output -` receives the same schema-3 object once after orchestration reaches its terminal state.

## 13. Standalone GPU and LLM Execution

GPU mode is implemented in `src/gpu_bandwidth/` behind a pure C++ `GpuBackend` interface. The concrete Objective-C++
backend is synchronous, uses bounded autorelease pools, retains suite resources until explicit release/destruction, and
converts nil/NSError/command failures into status-bearing C++ results. No Objective-C type crosses the public boundary.

### 13.1 Runtime compilation and capability contract

`MTLCreateSystemDefaultDevice` must succeed, `hasUnifiedMemory` must be true, and
`supportsFamily(MTLGPUFamilyApple7)` must succeed. Supported Apple family names plus availability-guarded Metal3/Metal4
families are diagnostic metadata; a future GPU is accepted by capability rather than a device-name allowlist.

The embedded canonical source is compiled once outside measured work with `newLibraryWithSource` using:

- `MTLLanguageVersion2_3`
- Integer-only kernels (`floating_point_math: "not_applicable_integer_only"`)
- Empty preprocessor macros
- Kernel revision `gpu-linear-word-mod32-tg-reduce-v2`
- SHA-256 over the exact canonical UTF-8 source bytes

Compilation metadata includes the source hash, compiler identifier, build SDK, deployment target, macOS product/build,
and compiler diagnostics. A non-nil `MTLLibrary` is runtime success even if NSError carries a warning; a nil library is
failure. The source hash identifies input bytes, not generated GPU machine code, so macOS build remains part of a strict
comparison cohort.

### 13.2 Work plan and compute geometry

Read, write, and copy derive domain-separated operation seeds with SplitMix64. Each immutable plan records requested and
effective bytes (identical; no silent reduction), pass and exact-payload accounting, guardrail ceilings, vector/tail
shape, pipeline limits, dispatch geometry, and a canonical `gpu-work-plan-v1` identity.

The grid contract is deterministic:

1. Process full regions as consecutive 16-byte `uint4` values; a direct backend path safely handles a 0–15 byte tail.
2. Choose the largest `threadExecutionWidth` multiple not exceeding
   `min(256, maxTotalThreadsPerThreadgroup)`.
3. Choose `min(ceil(vector_count / threads_per_threadgroup), 8192)` threadgroups, at least one.
4. Cover all vectors with a grid-stride loop.
5. Encode one full-buffer dispatch per pass. A measured attempt records exactly one command buffer and one serial compute
   encoder; maximum dispatch count is 16,384 and maximum exact payload is 64 GiB.

Read/write bytes per pass equal the buffer size. Copy bytes per pass equal twice the buffer size and alternates A→B/B→A
by pass parity. Every operation's `gpu-dual-mod32-v2` timed kernel contributes an observable dual modulo-2^32
accumulator. Each word affects the data/index reduction; after its loop, each GPU thread multiplies its two local lanes
once by pass-specific odd domain weights. Global thread zero then contributes one nonzero token per lane and dispatch.
The versioned weight/token keys mix operation seed, 64-bit buffer size, pass, operation, and copy direction, avoiding the
power-of-two population collapse of per-word pass terms. The independent CPU oracle uses closed-form pattern summaries
and O(passes) work. Reduction/status traffic is inside GPU time but outside exact payload. Write/copy retain the separate
`gpu-dual-mod32-v1` final-buffer checksum.

### 13.3 Calibration, warmup, ordering, and aggregation

Automatic mode resolves each operation once before loop 0:

1. Pick a pilot that covers at least 8 MiB when guardrails permit.
2. Run excluded warmup, deterministic precondition, timed pilot, and mandatory validation.
3. Scale toward 150 ms, run an excluded duration trial, and make at most two excluded corrections when outside the
   inclusive 100–250 ms window.
4. Freeze the last valid plan and reuse it unchanged in all measured loops.

Explicit `--iterations` skips pilot/trial/corrections but not measured-task warmup/preconditioning. All excluded attempts
remain in JSON. A later out-of-window measurement is retained/classified, not retried or recalibrated. Duration quality
distinguishes within/below/above window, single-pass overrun, and dispatch/payload-cap-limited short work.

Loop order rotates read→write→copy, write→copy→read, and copy→read→write. Only `measured` plus passed validation values
enter aggregates. A single value is its own headline; multiple values use median P50 and shared descriptive statistics.
Fewer than three values is `insufficient-samples`; CV above 5% is `noisy`; otherwise `stable`. No outlier filtering or
winsorization occurs. Order balance is complete only when the whole run is complete and completed loops are divisible by
three.

### 13.4 Timing and correctness boundaries

The primary duration is read after completion as:

```text
gpu_elapsed_seconds = command_buffer.GPUEndTime - command_buffer.GPUStartTime
value_gb_s = exact_payload_bytes / gpu_elapsed_seconds / 1e9
```

The timestamp delta must be finite, positive, ordered, and consistent with the stored elapsed value. Host submit,
wait-end, and wall duration are diagnostic and never the GB/s denominator. The timed command buffer contains only one
operation's frozen dispatches. Pipeline creation, fill/poison, warmup, precondition, final checksum, and test readback are
excluded command buffers.

Read correctness is established by comparing the timed dual accumulator against the independent CPU formula, without a
new GPU validation command. Write/copy require the same timed accumulator plus one excluded full-buffer final-checksum
dispatch. The direct Metal integration test additionally blits the private output to a shared staging buffer and compares
bytes against an independent CPU oracle, including multi-pass copy parity and tail sizes. A checksum mismatch is
`invalid`; a Metal/validation command error is `failed`; an invalid timestamp is `invalid` with validation
`not-run-timer-invalid`. Phase lifecycle metadata records zero validation status resets for read and one host reset for
write/copy before the final-checksum dispatch. Only a passed terminal attempt gets numeric bandwidth.

### 13.5 Interruption and checkpoint linearization

The runner uses task-level completion-wins. Stop is checked before a logical task and after its terminal result, not
between warmup, precondition, timed command, and required validation. Once started, that sequence finishes subject to
normal error short-circuiting. A valid current result stays measured even if SIGINT/SIGTERM arrived during it; no next
task starts. A genuine command/timer/validation/checkpoint failure has priority over interruption.

Every planned loop has three pre-created measurement slots. Interruption finalization preserves terminal slots and turns
every remaining `not-run` slot into `interrupted`, `value_gb_s: null`, reason `interruption-before-task`. Such slots do
not increment attempted/completed counters. `results_complete` and `conclusions_valid` are true only for top-level
complete with every planned measurement validated. A graceful interrupt returns `EXIT_SUCCESS` but serializes
`status: "interrupted"` and false conclusions.

With output enabled, the runner offers a logical checkpoint after each terminal measurement. It reads stop once before
and once immediately after that boundary; if the second read first observes the signal, it offers at most one additional
interruption checkpoint. File sessions persist those checkpoints atomically, and valid post-parse initialized backend/
capability/compile/allocation/work-plan failures also produce one file checkpoint. Stdout sessions invoke no
intermediate payload builder or serializer but preserve every state transition and stop read, then emit one terminal
schema 1 payload at the command boundary. CLI/config errors, including a buffer below 64 MB, and backend-factory failure
before result initialization do not produce JSON. A file checkpoint failure stops the run as failed; the file may remain
at the preceding successful checkpoint.

### 13.6 Interpretation and validation boundary

The result is effective versioned-kernel payload bandwidth at the Metal command-buffer timing boundary. JSON fixes
`dram_residency` to `unverified`. Private storage, buffer size, or a positive result cannot prove DRAM traffic; GPU cache,
dispatch processing, checksum reduction, other GPU load, thermals, Low Power Mode, compiler, and driver remain factors.
Copy is aggregate read-plus-write and may numerically exceed a one-direction bus figure. CPU/GPU values are not directly
comparable despite sharing decimal GB/s and copy 2× accounting.

The M4 Instruments audit exposed no usable memory-traffic counter, so it could not isolate or quantify timed accumulator
reduction/status-atomic traffic. That auxiliary traffic is included in GPU elapsed time but excluded from the logical
payload numerator; the separate final checksum remains outside primary timing.

Apple7 plus unified memory defines capability support. It does not validate performance. M4 is the schema-1 release
reference cohort: the completed 0.61.0 automatic and fixed-work populations establish a stable effective-payload baseline
for their exact hardware, OS, compiler, kernel, and methodology identity. The frozen pre-remediation validation identity uses
`gpu-linear-word-mod32-tg-reduce-v2`, the frozen 8192-threadgroup cap, canonical MSL SHA-256
`b9a242d2b959c9c11f6f130a52afd66f111d6761be2193beec1f051baa094296`, and the exact executable identity retained
with the local validation record. The current canonical source SHA-256 retained unchanged in 0.63.0 (introduced in
0.61.2) is
`21def2d75d3545dba31aa4897ea57ec2fd0e4481cd86ce21725338ab0f322ac5` after removing three unread shared-parameter
fields; runtime Metal integration revalidates compilation and correctness, while the performance population remains
tied to the frozen pre-remediation identity. Automatic read/write/copy
median-of-process-medians are 88.606742648049/74.383866793814/78.583784905446 GB/s with cross-process CV
0.221498348705/0.967311621904/0.310543092510%; fixed-24 values are
91.074797816490/75.240302989483/78.508461231110 GB/s with CV
0.506707339121/0.827667144983/0.326577301613%. The accepted grid gate retained 8192 because 4096 write was more than 2%
below the best candidate. The final Instruments audits did not expose a usable memory-traffic counter or a way to
isolate timed reduction overhead, so neither the baseline nor private storage proves DRAM throughput and
`dram_residency` remains `unverified`. Complete rejected populations were retained without cherry-picking. The large
raw validation record is local-only and intentionally excluded from Git. An admitted M1/Apple7 or newer GPU remains
capability-supported and performance-unvalidated until runtime compilation, exact correctness, timestamp smoke, and
the appropriate controlled performance campaign are recorded. Integration tests deliberately have no minimum-GB/s
assert.

### 13.7 LLM generic plan and active CPU decode/prefill execution

LLM mode is implemented under `src/llm_memory/` with a pure checked work planner, a pure paged-layout/permutation
module, an Objective-C-free synchronous `LlmBackend` boundary, a status-bearing backend-independent runner, a dedicated
CPU adapter, phase/layout-specific ARM64 kernels, an ordered schema builder, and a Foundation-backed environment
snapshot. Generic enums cover backend `cpu|metal`, phase `decode|prefill`, and KV layout `contiguous|paged`.
All four CPU phase/layout combinations are active with exact methodology identities
`llm-memory-v1-cpu-decode-contiguous`, `llm-memory-v1-cpu-decode-paged`,
`llm-memory-v1-cpu-prefill-contiguous`, and `llm-memory-v1-cpu-prefill-paged`. Metal remains unavailable and never
receives fallback execution.

`LlmMemoryWorkPlan` keeps logical geometry, resources, accounting, seeds, and identities common while
`LlmBackendExecutionPlan` carries exactly one `LlmCpuExecutionPlan` or `LlmMetalExecutionPlan` tag. The CPU alternative
owns layout-specific worker reduction, worker-major pointer-free ranges, descriptor counts, and retained planner storage.
Checked accessors require the declared backend and variant tag to agree.

The prefill planner resolves prompt length `P` and query-tile length `Q`. It computes `C = ceil(P/Q)`, prefix visits
`S = Q*triangular(P/Q) + (P%Q != 0 ? P : 0)`, causal pairs, logical
attention audit counts, one weight pass, full-prompt K/V writes, tiled prefix reads, and checked per-scenario payloads.
For paged geometry it computes prefix block visits and the exact `N + 2*M` semantic lookup count with a logarithmic
floor-sum. Its bounded semantic oracle freezes owner-local ascending-token writes with K then V for each token,
followed for every tile by the complete owned K prefix and then the complete owned V prefix. Its affine64 checksum
oracle covers every operation ordinal and every tile-read visit. Excluded post-validation checks deterministic
owner first/middle/last canonical-word samples, including bytes clipped to owner boundaries, against `T - 1`; it does
not reread every prompt record.

The pure CPU ownership API partitions contiguous tokens or whole logical blocks by exact scenario-accounted prefix cost.
Valid results publish a versioned canonical identity binding the prefill geometry, scenario, worker rotation, optional
weight shards, assignment boundaries, per-worker costs, and min/max/imbalance evidence. Any validation, overflow, or
allocation failure publishes no partial assignments, worker vectors, totals, or identity.

Contiguous prefill materializes scenario-specific cost-balanced token ownership, a dedicated descriptor ABI, full-size
K/V resources, and the `llm_prefill_memory_asm` executor. Paged prefill materializes block-exclusive weighted
ownership, full physical K/V pools, a read-only uint32 block table and seeded permutation, a distinct descriptor ABI,
with 48-byte layer and 112-byte block-assignment records, and the `llm_prefill_memory_paged_asm` executor. Both retain
exact per-scenario worker-cost minimum, maximum, imbalance,
scope, and execution identities.

For active weights `W`, layers `L`, KV heads `h_kv`, head dimension `d_h`, element bytes `s_kv`, batch `B`, and visible
context `A` including the current token, define one K or V token record as `R = h_kv*d_h*s_kv`, one layer's paired
record as `D = 2*R`, and the full model's paired record as `K = L*D`. One decode step has weight read `W`, KV read
`B*A*K`, and paired current-token append `B*K`. Weights-only payload is `W`; KV-only is `B*A*K + B*K`; mixed adds
both. Checked arithmetic covers every product, sum, division, round-up, offset, descriptor count, and task total.

Contiguous K and V use `[layer][batch_sequence][token][kv_head][head_dimension]`. For paged KV with explicit block size
`G`, the planner derives:

```text
N = A / G + (A % G != 0)
P_b = B * N
block_bytes = G * R
last_block_tokens = A - (N - 1) * G
last_block_valid_bytes = last_block_tokens * R
k_logical_bytes = L * B * A * R
k_physical_bytes = L * P_b * block_bytes
k_layout_padding_bytes = k_physical_bytes - k_logical_bytes
v_logical_bytes = k_logical_bytes
v_physical_bytes = k_physical_bytes
v_layout_padding_bytes = k_layout_padding_bytes
block_table_entries = B * N
block_table_bytes = block_table_entries * 4
```

There is no inter-block alignment padding. Each batch row's logical blocks map through one command-lifetime row-major
`uint32_t block_table[B][N]`. Its entries form a bijection over `0..P_b-1`; `UINT32_MAX` is the invalid sentinel, so
`P_b <= UINT32_MAX`. The same physical ID selects the per-layer block in both K and V pools. A token at logical block
`b` and offset `t` uses physical block `block_table[batch][b]` and byte offset
`(layer*P_b + physical_id)*block_bytes + t*R`.

The table starts as the identity and is shuffled with the versioned stateful SplitMix64 stream and unbiased descending
Fisher–Yates algorithm. The seed domain is `0x4C4C4D4B56504731`; the resolved permutation seed is derived from the base
seed, and table identity includes the algorithm version, domain, resolved seed, entry count, and SHA-256 of explicit
little-endian `uint32_t` entries. Preparation first admits the entire known-owned peak, then materializes the table,
validates range/bijection/sentinel constraints with a bounded transient bitset, hashes it incrementally, and makes the
CPU mapping read-only. The same table remains frozen across warmup, calibration, loops, and scenarios.

Paged K/V allocate full blocks. The suffix padding in each batch sequence's last block is initialized and pre-touched,
is never timed as model traffic, and is checked by post-task canaries. Physical initialization depends on pool,
physical block ID, and physical offset, so an incorrect table or address changes correctness evidence. Logical bytes,
physical lengths, layout padding, table bytes, mapping rounding, preparation transient, and admitted peak are distinct.

Per layer and batch sequence, paged decode executes one paired append lookup, `N` K-scan lookups, and `N` V-scan
lookups, in append → K → V order. Therefore KV-only and mixed use
`L*B*(2*N+1)` semantic lookups and `4*L*B*(2*N+1)` metadata bytes per work unit; weights-only uses zero. Each semantic
lookup is an explicit timed 32-bit table load, and the data address is formed only after the loaded ID. The host does
not pre-resolve IDs into pointer lists. K and V scans load the table separately; their paired append shares one lookup.
Metadata bytes are included in task-accounted guardrails but excluded from the primary effective-model-payload GB/s
numerator.

Worker partitioning assigns each logical paged block to exactly one worker and never splits a block. When blocks are
fewer than workers, ownership rotates deterministically by layer/batch ordinal. The active CPU planner caps the
admitted team so every effective worker owns KV work; lookup counts do not depend on worker count. Layout-specific
descriptors carry table-row and K/V layer-pool bases, logical block range, block geometry,
last-block valid bytes, append offset, and layer/batch identity. Contiguous and paged descriptor ABIs and assembly
entrypoints are distinct, so the hot loop has no layout branch.

`weights_only` reads each layer's weight shard. `kv_only` and `mixed` execute the layout-specific append/K/V order;
mixed reads the layer weight shard before that layer's KV work. Workers preserve local increasing layer/logical-block
order without a global per-layer barrier. Contiguous appends retain the prior affine pattern and checksum contract.
Paged execution uses a versioned checksum that non-separably mixes logical table index, loaded physical ID, semantic
visit kind, and work-unit ordinal. An independent bounded scalar oracle does not call the production kernel and does not
reread the full pool. Equal-multiplicity wrong-table swaps must mismatch. Timed checksum mismatch, append mismatch, or
padding-canary mismatch makes the measurement invalid without retry.

One command owns one backend. Its lifecycle is auxiliary estimation, initialization, execution-plan resolution,
resource preparation, zero or more atomic task calls, evidence readout, and one release. The CPU backend owns
`HighResTimer`, immutable prepared resources, synchronized worker start, best-effort QoS evidence, and exact checksum
validation. Mapping, initialization/pre-touch, table/permutation/hash work, descriptor construction, worker creation,
QoS, warmup, calibration, expected-oracle work, joins, canary/checksum validation, aggregation, console, JSON, and
checkpoints are outside the authoritative timed interval. Only worker kernel work from the all-ready release to
last-worker completion is timed.

For contiguous prefill, each operation writes owner-local prompt tokens in ascending order, K then V for each token,
before that owner's reads. Query tiles advance by remaining distance; each tile scans its complete owned K prefix and
then its complete owned V prefix. The write pattern and checksum include the operation ordinal, and the timed checksum
covers every tile-read visit. Excluded post-validation checks each owner's deterministic first/middle/last
canonical-word samples and clipped boundary bytes against final ordinal `T-1`. No cross-worker barrier is required
because an owner reads only records it owns.

Paged prefill preserves that semantic order through whole logical-block ownership. For prompt length `P`, tile ends
`e_j`, and block size `G`, it performs `N = ceil(P/G)` paired-write lookups and
`M = sum(ceil(e_j/G))` lookups for each of the separate K and V tile scans per layer/batch pair. Partial prefixes use
their exact valid byte count rather than rounding the visit to a full terminal block. Each semantic lookup executes an
`ldr w` before address formation, and checksum evidence non-separably binds logical index, physical ID, visit kind, and
operation ordinal. Total KV-active metadata is `L*B*(N+2*M)` lookups and four bytes per lookup; weights-only uses zero.

The common runner accepts a task only when status/reason, backend, phase, layout, methodology/component identity,
work-unit and KV-write kinds, scenario, frozen plan, authoritative finite positive timing, planned/completed work,
model-payload/metadata/accounted counters, and validation all match. Comparative acceptance additionally requires the
same `G`, paged geometry, physical/logical/padding/table resource identity, permutation identity/hash, schedule, checksum,
and timer identities. CPU worker/QoS/timer/checksum-vector invariants remain adapter-owned.

Omitted iterations calibrate the three scenarios independently outside measurement. The exact initial pilot shape is
warmed once; correction candidates do not receive general warmups, while the first irreducible one-work-unit candidate
receives one confirmation warmup when that shape has not already been warmed. Explicit iterations are exact. All three
plans freeze atomically before canonical same-shape frozen warmups, and loop order then cyclically rotates weights-only,
KV-only, and mixed. No measured task begins until every frozen warmup succeeds. Only measured and checksum-valid results
enter aggregates. Stop is checked between atomic backend tasks. File output checkpoints each terminal scenario;
resources are released before the command-terminal checkpoint. Stdout emits one final schema-1 object. See
[LLM_MEMORY_PROFILE_WHITEPAPER.md](LLM_MEMORY_PROFILE_WHITEPAPER.md) for the complete formulas, ABI, schema, and
interpretation contract.

## 14. Assembly Kernel Layer

Assembly entrypoints are declared in `src/asm/asm_functions.h` and used by benchmark/warmup code:

- Main-memory kernels: read, write, copy (non-temporal stores).
- Cache kernels: read, write, copy (cache-focused variants).
- Latency kernel: pointer-chase loop.
- Pattern kernels: reverse (read/write/copy), three parameterized phased strided entrypoints
  (`memory_{read,write,copy}_strided_phased_loop_asm`) that accept stride, pass count, and initial 32-byte phase, and
  random (read/write/copy). The 64 B, 4096 B, 16384 B, and 2 MiB patterns use the same phased strided API.
- Core-to-core kernels: initiator and responder round-trip loops.

Design intent:

- Keep hot loops in ARM64 Apple Silicon assembly for predictable overhead and high throughput.
- Follow AAPCS64 conventions for register preservation and call boundaries.
- Use checksum sinks in read paths to keep loads architecturally meaningful.

Latency kernel (`memory_latency_chase_asm`) performs strictly dependent pointer chasing and returns terminal pointer to prevent dead-code elimination.

Core-to-core kernels:

- `core_to_core_initiator_round_trips_asm`: Waits until the initiator owns the token, hands it to the responder, then
  waits for the responder to return it; repeats for the specified round trips.
- `core_to_core_responder_round_trips_asm`: Waits until the responder owns the token and hands it back to the
  initiator; coordinates via the shared token word.
- The timed token and startup/control flags occupy distinct 128-byte-aligned storage blocks. The elapsed result covers the
  complete assembly token protocol and system effects; it does not directly observe a physical coherence path.

The dedicated `llm_decode_memory_asm` entrypoint receives layer descriptors, sequence descriptors, layer count, decode
work-unit count, scenario bits, scenario seed, and one 96-byte worker checksum output. It follows AAPCS64, preserves used
callee-saved registers, handles exact 32-byte vector blocks plus bounded tails, uses `ldp q0,q1` reads and ordinary
temporal `stp`/`str` KV append stores, and performs no software prefetch. Weight, K, and V reads feed separate
`llm-read-checksum-v1` lanes with exact byte/span counts. Top-level zero/null inputs are safe for direct contract tests,
but zero work is not an admitted production plan.

The separate `llm_decode_memory_paged_asm` entrypoint receives the paged descriptor ABI. It performs one explicit
`ldr w` semantic lookup for the paired current-token K/V append, then independent table loads for every logical K block
and every logical V block. It forms physical addresses only after those loads, never crosses a block's valid-byte end,
uses exact scalar/vector tails without touching suffix padding, preserves AAPCS64 callee-saved registers, and returns
versioned non-separable logical-index/physical-ID/visit/work-unit checksum evidence. Layout dispatch happens before the
call; neither hot loop branches on layout.

The `llm_prefill_memory_asm` entrypoint receives contiguous-prefill layer and owner descriptors plus P/Q geometry,
work-unit count, scenario, and seed. For every operation ordinal it reads the applicable weight shard once, writes each
owner-local prompt token in ascending order with K before V, then scans each increasing query tile's complete owned K
prefix before its complete owned V prefix. Exact vector/scalar tails do not cross an owned range. Write and checksum
patterns bind the operation ordinal; layout/phase dispatch occurs before the assembly call.

The separate `llm_prefill_memory_paged_asm` entrypoint receives a paged-prefill descriptor containing the read-only
table row, physical K/V pools, owned first/count block range, prompt/block tail geometry, and prefill tile geometry.
Whole-block owners first populate their prompt span through paired K/V lookups, then perform a separate K lookup and V
lookup for every exact block fragment in each tile prefix. Address formation follows each `ldr w`; 32-byte bodies and
exact scalar tails never touch suffix padding. The kernel preserves AAPCS64 state and returns non-separable
logical-index/physical-ID/visit/operation checksum evidence without calls, prefetches, or layout branches.

## 15. Timing Model

Timing API: `HighResTimer` (`src/core/timing/timer.*`).

- Provides second and nanosecond stop methods.
- Used for both macro (test durations) and micro (latency sample windows) timing.
- Factory creation returns optional; failure is treated as fatal at call sites.

GPU primary timing does not use `HighResTimer`: it uses completed Metal command-buffer `GPUStartTime` and `GPUEndTime`.
Host steady-clock submit/wait/wall values are diagnostic. Total GPU-mode host execution time uses steady clock and is
separate from every operation denominator.

LLM scenario timing uses `HighResTimer` from a synchronized all-workers-ready gate to last-worker completion. Paged
block-table loads, address formation, model reads/writes, and timed checksum accumulation are inside. Mapping,
initialization/pre-touch, permutation/table generation and validation, table protection, descriptor validation and
materialization, task-local paged append-slot restoration, thread creation, QoS, same-shape warmup, calibration,
expected-oracle work, checksum/canary post-validation, joins, aggregation, console, JSON, and checkpoints are outside.
There is no cache flush between tasks; the timing semantics are warm, cacheable, and cache-inclusive.

## 16. Statistics and Aggregation

`BenchmarkStatistics` and pattern statistics collect vectors per metric across outer loops.

- Single loop: direct value reporting.
- Multiple loops: aggregate statistics printed and optionally serialized.
- Statistics include central tendency and percentile-oriented summaries.

For contention-prone environments, percentiles (P50/P95/P99) are more informative than mean-only interpretation.

### 16.1 Statistics Fields

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

GPU aggregates use the same descriptive-statistics implementation, a 5% CV threshold, and only validated measured
values. Multiple GPU loop values use median P50; fewer than three cannot receive a repeatability classification stronger
than `insufficient-samples`.

LLM maintains separate measured-only distributions for synthetic work-unit latency, synthetic memory work units per
second, and effective model-payload GB/s for weights-only, KV-only, and mixed. Multiple values use median P50; fewer than
three are `insufficient-samples`, and effective-model-payload-GB/s CV above 5% selects `noisy` and emits a high-CV
warning without filtering or retry. Comparative validity also requires complete three-position cyclic order balance.
Mixed weight/KV fractions are byte fractions, not independently timed component bandwidths.

## 17. Console Output Contract

Console rendering is centralized in `src/output/console` and message helpers in `src/output/console/messages`.

Contract highlights:

- Every successfully started direct mode and parameter sweep emits one shared version, copyright, and GPL banner
  before mode-specific configuration or status output. Nested sweep runs do not repeat it. Help output and usage
  diagnostics retain the separate usage preamble; preflight failures do not emit the runtime banner.
- Result-producing CPU benchmark paths print applicable configuration and cache information before execution.
- Per-loop results are printed in standard mode.
- Pattern mode prints pattern table-style sections and derived efficiency indicators.
- GPU mode prints a separate device/private-tracked header, read/write/copy effective-payload headlines, repeatability,
  and interpretation note. Copy is explicitly aggregate read + write and DRAM residency remains unverified.
- LLM mode prints a separate header with backend, phase/work unit, KV layout, phase geometry, cacheable semantics,
  exact weight/KV-read/KV-write payload bytes, a decode-only two-decimal crossover estimate, up to three measured
  scenario headlines, evidence-backed quality warnings, and a memory-only/non-DRAM interpretation note. Paged output
  also reports `G`, `N`, tail geometry, physical and padding bytes, table/permutation identity, lookup counts, metadata
  bytes, and checksum/canary status. Complete
  repeatability statistics remain in JSON; the console prints a CV value only when it exceeds the warning threshold.
  Phase-specific console wording maps to generic JSON `synthetic_work_unit_latency_seconds`,
  `synthetic_memory_work_units_per_second`, and `effective_model_payload_gb_s`; it never presents a value as inference
  tokens/s.
- Aggregate statistics printed when loop count > 1.
- Errors and warnings use `Messages::error_prefix()` / `Messages::warning_prefix()` conventions.
- Live progress uses the shared spinner on `stderr` only when it is a TTY; redirected standard and pattern output contains no carriage-return control sequences.
- For every result-producing direct command or CPU sweep using `--output -`, the output session is installed before
  runtime console
  rendering and worker-thread creation. The general standard/pattern/TLB branch installs it before sweep/direct
  validation; core-to-core installs it after its combined parse/preflight; GPU installs it after its dedicated parser
  and help handling but before the runtime banner, QoS, signal scope, and backend factory. LLM installs the same session
  before banner/QoS/signal preparation, mappings, and workers. Post-parse human output therefore goes to stderr, while
  final JSON bypasses the redirected stream through the retained original stdout buffer. Parse/preflight and other
  pre-result failures leave stdout empty; human help remains stdout.

## 18. JSON Output Contract

JSON writer API (`src/output/json/json_output/json_output.cpp`):

Every result-producing direct mode and the CPU modes' supported sweeps can send their existing payload either to an
atomic file target or, for exact raw `-`, to one final stdout document. This transport does not wrap or change the
measurement schema. [API.md](API.md) is the process-integration contract and support matrix.

- Standard mode schema 3: `configuration` with `mode = "benchmark"` and the raw `output_file` target as a required
  string,
  `execution_time_sec`, completion counters/status, `results_complete`, `conclusions_valid`, per-loop `loops`,
  `main_memory`, `cache`, `timestamp`, and `version`.
- Pattern mode schema 3: `configuration`, `execution_time_sec`, command status/reason, planned/completed loop and
  measurement counters, `results_complete`, optional retained `patterns` evidence, `timestamp`, and `version`.
- TLB analysis schema 4: `configuration`, `execution_time_sec`, `tlb_analysis`, `timestamp`, `version`; conclusions
  require nested `status: "complete"` and `conclusions_valid: true`.
- Core-to-core schema 2: calibrated methodology configuration, `core_to_core_latency` command completion metadata, scenario work plans, nullable aggregate values, per-loop order/status/duration/hint/sample-boundary records, and affinity-comparison interpretability metadata.
- GPU schema 1: top-level mode/schema/methodology/status, exact counters and completeness, effective/copy/DRAM semantics,
  config/argv, environment, backend device/compile/allocation, memory budget, frozen plans, excluded calibration,
  status-bearing measurements/loop records, aggregates, and warnings.
- LLM schema 1: top-level backend/phase/layout/methodology/status, `configuration`, generic `resolved_plan`, tagged
  `backend_evidence`, memory-budget/calibration evidence, status-bearing generic work-unit measurements, measured-only
  aggregates, and interpretation, plus additive exact traffic, checksum, loop/checkpoint, environment, and warning
  evidence.
- Sweep envelope schema 1: general and core-to-core producers record `configuration.mode = "sweep"`,
  `configuration.sweep_schema_version = 1`, `configuration.base_mode`, `configuration.sweep_parameters`, top-level
  `status`, `status_reason`, `planned_runs`, `attempted_runs`, `completed_runs`, and `conclusions_valid`, plus per-entry
  `runs[].status`, `status_reason`, and `result`. Every attempted run is retained and
  `attempted_runs == runs.size()`; a file target checkpoints each attempt and also checkpoints a terminal envelope when
  the run plan is empty or a stop is observed before a run, without incrementing `attempted_runs`. Stdout preserves that
  logical cadence but defers serialization until the terminal envelope. `completed_runs` requires current standard
  schema 3 to have nested `configuration.mode: "benchmark"`, `status: "complete"`, `results_complete: true`,
  `conclusions_valid: true`, and a string `configuration.output_file`. Nested standard schema 2 and every other
  standard version are unsupported. Pattern requires nested
  `status: "complete"` and `results_complete: true` for pattern; `tlb_analysis.status: "complete"` and
  `tlb_analysis.conclusions_valid: true` for TLB; or
  `core_to_core_latency.status: "complete"` and `measurements_complete: true` for core-to-core. Partial, interrupted,
  and failed attempts remain as evidence without incrementing the completed count. TLB's native `status: "error"` maps
  to a failed attempt while its schema-4 payload is retained without a fabricated nested `status_reason`. The
  authoritative schema-1 sweep acceptance predicate is exactly
  `status == "complete" && conclusions_valid == true`. Producers maintain `completed_runs == planned_runs` for an
  envelope satisfying that predicate; consumers may check the equality separately as a defensive consistency check,
  but it is not an additional completeness condition.

Pattern schema 3 plans 21 measurements per loop and treats numeric measured values plus intentional skips as terminal.
Only Complete loops feed aggregate vectors, medians, statistics, and console summaries. Partial, interrupted, and failed
loop measurements remain in JSON as evidence, while command status/counters keep `results_complete` false. Preparation
failure may omit `patterns`; main and sweep orchestration still build the completion payload before returning or
classifying the failure.

Command completeness for current standard schema 3 requires `configuration.mode == "benchmark"`,
`status == "complete" && results_complete == true && conclusions_valid == true`, with string
`configuration.output_file`. Bundled standard-memory examples track this current schema-3 producer, perform local
sanity checks (including exact top-level `version == "0.63.0"` for the current producer), and read the metric paths they
need directly. They do not support released standard schema 2, unversioned historical standard JSON layouts, or any
other explicit standard version. Pattern requires
`status == "complete" && results_complete == true`. TLB requires
`tlb_analysis.status == "complete" && tlb_analysis.conclusions_valid == true`; core-to-core requires
`core_to_core_latency.status == "complete" && core_to_core_latency.measurements_complete == true`. Metric consumers must
additionally require the selected measurement's status and non-null value; an intentionally skipped pattern measurement
can coexist with command completeness without being consumable as a measured value. Affinity-scenario interpretation
also requires `affinity_hint_comparison_interpretable == true`. GPU requires `status == "complete" &&
results_complete == true && conclusions_valid == true`; position-balanced comparisons additionally require
`operation_order_balance_complete == true`.
LLM requires `mode == "llm_memory" && schema_version == 1`, top-level backend/phase/layout equal to the request,
methodology equal to `llm-memory-v1-<backend>-<phase>-<layout>`, `status == "complete"`, `results_complete == true`,
`conclusions_valid == true`, and every planned measurement to be `measured`. A selected LLM metric additionally
requires non-null/checksum-accepted evidence; comparative validity already includes complete scenario-order balance.

### 18.1 Configuration keys

In addition to standard fields (buffer size, iterations, loop count, thread count, CPU/OS info):

- `latency_chain_mode` (string): Resolved pointer-chain construction mode.
- `use_latency_tlb_locality` (boolean): Whether the configured locality value is greater than zero; an explicit
  `global-random` mode can still ignore that value.
- `latency_tlb_locality_bytes` (number): TLB-locality window size in bytes.
- `latency_tlb_locality_kb` (number): TLB-locality window size in KB.
- `benchmark_schema_version` (number): `3` for current output.
- `output_file` (required string, standard schema 3): Raw direct output target; nested standard sweep results use an
  empty string because the envelope owns persistence.
- `methodology_version` (string): `benchmark-v2-calibrated-seeded-balanced`.
- `benchmark_seed` (string): exact uint64 decimal string plus source/encoding fields.
- Calibration targets/windows and phase/operation schedule policies.

Schema 3 makes `configuration.output_file` and top-level `conclusions_valid` mandatory without changing the standard
methodology version. Bundled standard-memory examples identify the current producer explicitly rather than inferring
standard identity from metric layout; schema 2 and unversioned historical standard JSON are unsupported inputs.

### 18.2 Main-memory latency keys

- `main_memory.latency.headline_ns`: status, median-or-single headline, loop values, robust statistics, measurement
  records, quality, and the pooled separate-sample distribution with loop boundaries.
- `main_memory.latency.automatic_locality_comparison.locality_16k_latency_ns`.
- `main_memory.latency.automatic_locality_comparison.global_random_latency_ns`.
- `main_memory.latency.automatic_locality_comparison.locality_latency_delta_ns`.
- `loops[].measurements`: nullable per-loop status/value plus exact work, worker, seed, timing, calibration, and order
  metadata.

### 18.3 Structure conventions

- Ordered JSON is used for stable key order.
- Aggregate `value` is a single measured loop or median P50; `values` contains only measured loop headlines.
- Statistics include average, median, P90/P95/P99, sample stddev, CV, MAD, min, and max.
- Unavailable measurements use `null` plus status/reason, never numeric zero.

### 18.4 Core-to-core schema 2

- `configuration.schema_version` (number): `2`.
- `configuration.methodology_version` (string): `core2core-v3-calibrated-balanced-auditable-128b-isolation`.
- Calibration metadata: excluded 100,000-round-trip pilot after a 1,000,000-round-trip calibration warmup; 25 ms
  final warmup target, 250 ms continuous headline target with a 100-300 ms intended window, and 1 ms sample-window
  target. Minimum work is 20,000/1,000,000/2,000 round trips respectively. Pilot warmup is intended to reduce startup
  transients; the calibration targets comparable durations but does not control environmental noise.
- `core_to_core_latency.status`, `planned_measurements`, `completed_measurements`, and `measurements_complete` describe
  command completion.
- Each scenario contains `status`, `status_reason`, planned/completed loops, a calibrated `work_plan`, continuous
  headline values/statistics, a nullable median headline, a distinct pooled `samples_ns` distribution, and
  `loop_records`.
- Loop records retain cyclic schedule position, status/reason, nullable round-trip and one-way estimates, headline
  duration/quality, the exact pooled-sample range contributed by measured loops, and both workers' observed QoS/affinity
  API outcomes. Invalid loops contribute no pooled samples and have a zero-length range.
- `affinity_hint_comparison_interpretable` is true only for a complete command when both workers' affinity API calls
  returned success in every measured affinity-tag record. It excludes QoS and calibration-pilot outcomes and does not
  imply that macOS honored a particular physical placement or hard core pinning.
- Invalid, failed, interrupted, and not-run measurements never become numeric zeroes.

### 18.5 GPU schema 1

- Top-level discriminator is `schema_version: 1`, `mode: "gpu_bandwidth"`, methodology
  `gpu-bandwidth-v1-private-runtime-single-cmdbuf-calibrated-balanced`; it is not nested under standard configuration.
- Run statuses: `not-started`, `complete`, `partial`, `interrupted`, `failed`, `unsupported`. Measurement statuses:
  `not-run`, `measured`, `interrupted`, `invalid`, `failed`. Only measured has finite positive `value_gb_s`; unavailable
  state is `null` plus reason.
- Exact counters distinguish planned/attempted/completed loops and planned/attempted/terminal/completed/validated
  measurements. Completeness/conclusions require every planned measurement validated; order balance additionally requires
  a multiple of three completed loops.
- Payload semantics, copy 2× semantics, `dram_residency: "unverified"`, exact argv/configuration/seed, environment,
  device/capabilities, compilation provenance, allocation/budget, work plans, excluded calibration attempts,
  status-bearing measurements, planned/realized loop records, aggregates, and warnings are all explicit.
- Seeds, buffer/payload/resource sizes, source-derived values that can exceed IEEE-754 exact-integer range, registry ID,
  and checksums use schema-named decimal strings. Non-finite numeric diagnostics serialize as null.
- Per-phase backend status, command status, command-buffer/encoder/dispatch counts, stable reason, and raw NSError
  domain/code/description remain separate. Measurement timing includes GPU and host diagnostics; validation includes
  separate `timed_accumulator_algorithm` and `final_checksum_algorithm` identities plus expected/actual checksums.
- See [GPU_BANDWIDTH_WHITEPAPER.md](GPU_BANDWIDTH_WHITEPAPER.md) for the complete consumer/maintenance contract.

### 18.6 LLM schema 1

- The prior CPU/step-specific schema-1 shape was unpublished. Generic v1 replaces it without compatibility aliases,
  fallback reading, or a schema-version increment.
- Required top-level fields are `schema_version`, `mode`, `backend`, `phase`, `kv_layout`, `methodology_version`,
  `software`, `configuration`, `resolved_plan`, `backend_evidence`, `memory_budget`, `calibration`, `measurements`,
  `aggregates`, `status`, `reason_code`, `results_complete`, `conclusions_valid`, and `interpretation`.
- Methodology is exactly `llm-memory-v1-<backend>-<phase>-<layout>`. Active identities are
  `llm-memory-v1-cpu-decode-contiguous`, `llm-memory-v1-cpu-decode-paged`,
  `llm-memory-v1-cpu-prefill-contiguous`, and `llm-memory-v1-cpu-prefill-paged`. Metal remains unavailable.
- `configuration` retains exact `argv` and `resolved_sources`. `resolved_plan` owns phase-applicable `geometry`,
  layout-applicable `layout`, immutable logical/physical `resources`, and exact `component_identities`.
  Exactly one of `geometry.decode` and `.prefill` is populated. Prefill records integer P/Q and decimal-string
  C/prefix-visit/causal-pair/logical-attention/FMA counts; decode-only crossover and weight/KV-read ratio fields are
  null. Paged results populate block geometry and permutation fields; contiguous results use null for paged-only
  values. Component identities include logical profile, KV layout, optional permutation, backend executor, resource ABI,
  schedule, timer, buffer, write, checksum, and nullable MSL identities under the canonical
  `llm-memory-components-v1` fixed-order serialization.
- CPU prefill `backend_evidence.cpu.prefill` records `cost_unit: "worker-cost"`, execution/scope identities,
  descriptors per scenario/worker, and scenario-specific decimal-string worker cost, minimum, maximum, and imbalance
  evidence. The object is null for decode; `backend_evidence.cpu.paged` is populated for either paged phase, so paged
  prefill has both objects.
- `backend_evidence` always has `cpu` and `metal` object-or-null branches; only CPU is populated now.
  `memory_budget` requires decimal-string `resource_rounding_bytes`, `transient_peak_bytes`, `known_owned_peak_bytes`,
  and `admitted_budget_bytes`.
- Run statuses are `not_started`, `complete`, `partial`, `interrupted`, `unsupported`, `failed`. Measurement statuses are `not_run`,
  `measured`, `interrupted`, `invalid`, `failed`. Unobserved rates and unavailable checksum validity are null; status and
  reason distinguish them from numeric zero. Multiword status tokens use underscores; multiword reason-code and
  duration-quality tokens use hyphens. Unsupported is terminal, non-acceptable performance evidence and never falls
  back to another backend.
- A measurement or excluded task whose backend call throws before returning evidence serializes nested
  `execution.status: "unavailable"`, retains the runner-exception reason, and uses null for absent lifecycle/QoS/checksum
  observations. A `not_run` measurement also uses null for its top-level successful/failed QoS-worker counts.
- Every potentially large byte, capacity, block/table/lookup, token-visit, causal-pair, FMA-term, seed, and checksum
  count is a canonical decimal string. Small control/configuration/work-unit counts are JSON integer numbers. Applicable
  zero traffic is `"0"` or `0` by fixed field type; non-applicable objects/scalars and non-finite floating observations
  are null.
- `resolved_plan.resources` records regular-cacheable full-size logical/physical weight/K/V layout, explicit layout
  padding, and paged table bytes, while
  allocation/preparation/admission peaks live in `memory_budget`. Pointer/range values are not exposed.
- `measurements[]` records frozen work identity, `work_unit_kind`, integer planned/completed work units,
  `kv_write_kind`, per-work-unit and planned/completed model payload, layout metadata, accounted bytes, nullable
  `synthetic_work_unit_latency_seconds`, nullable `synthetic_memory_work_units_per_second`, nullable
  `effective_model_payload_gb_s`, working set, CPU lifecycle, and expected/actual checksum evidence. Only
  measured/checksum-valid records populate `aggregates.scenarios.weights_only`, `.kv_only`, and `.mixed`.
- For paged KV, `kv_block_tokens` is an integer input, while potentially large block/table/lookup/byte counts remain
  canonical decimal strings. `permutation_seed_uint64_decimal` is a decimal string and `permutation_sha256` is exactly
  64 lowercase hexadecimal characters. KV-only and mixed report `L*B*(2*N+1)` lookups and four metadata bytes per
  lookup per work unit; weights-only reports applicable zero traffic.
- `traffic_diagnostics.classification_version` is `llm-exact-weight-vs-kv-read-payload-v1`; tokens are
  `weight_payload_dominant`, `near_crossover`, and `kv_read_payload_dominant`. Near means exact equality, and
  `classification_is_payload_only` prevents a hardware-bottleneck interpretation.
- The complete-result predicate also requires backend/phase/layout to match the request, exact derived methodology,
  `status == "complete"`, `results_complete == true`, `conclusions_valid == true`, and every planned measurement to be
  measured. Paged comparison/acceptance additionally matches `G`, `N`, tail, logical/physical/padding/table resources,
  permutation version/domain/seed/hash, lookup/accounting, schedule, timer, and checksum identities. A count-one result
  may be complete yet fail conclusions because scenario positions are not balanced.
- See [LLM_MEMORY_PROFILE_WHITEPAPER.md](LLM_MEMORY_PROFILE_WHITEPAPER.md) for the field groups and consumer boundary.

### 18.7 Command-output transport boundary

The cold-path `JsonOutputSession` classifies the raw output value before path handling. Exact `-` selects final stdout
JSON. An empty value disables JSON for a direct command and is missing/invalid for a sweep. Every other non-empty value
is a file target, including `./-` and flag-shaped names such as `-G`. CPU adapters can resolve relative file targets
against the current working directory, while modes whose schemas retain the original spelling can preserve the raw
path. The shared atomic file writer remains sentinel-agnostic.

For a stdout target, the single-owner session retains the original `std::cout` buffer and routes ordinary command output
to `std::cerr`. Final JSON uses a separate stream backed by the retained buffer, so benchmark formatting state cannot
affect serialization. Stdout checkpoints are lazy successful no-ops and do not build payloads; file checkpoints still
use atomic replacement. Final stdout serialization emits UTF-8 JSON with two-space indentation and one trailing newline,
checks write and flush state, contains exceptions, and restores the prior stream buffer through the scope-bound
destructor. An observable terminal serialization, write, or flush failure returns `EXIT_FAILURE` without changing the
already-computed measurement state.

Standard, pattern, and TLB command boundaries select this infrastructure after successful parse/help handling and before
direct or sweep validation. Core-to-core selects it after its dedicated combined parse/preflight and help handling, but
before sweep execution, the runtime banner, and worker creation. GPU selects it after its dedicated parser/help path and
before its banner, QoS, signal scope, and backend factory. LLM selects it before banner, QoS, signal scope, work-plan
admission, mappings, and workers. Direct pattern, TLB, and core-to-core files receive one atomic final write; standard
and sweep files retain their existing intermediate checkpoints. GPU files retain terminal-measurement/failure
checkpoints and their post-release replacement. LLM files checkpoint every terminal scenario measurement and, unless a
measurement checkpoint itself fails, one post-release command terminal; failure is terminal and is not retried. Stdout
boundaries perform lazy no-op persistence at every logical checkpoint and emit one terminal payload after orchestration;
GPU and LLM still perform their checkpoint-boundary stop reads. The supported process contract, including help and
pre-result-failure exceptions, is defined in [API.md](API.md).

### 18.8 Path behavior

- Relative file `--output` paths are resolved against current working directory by CPU-mode adapters.
- GPU and LLM preserve the raw output spelling in `configuration.output_file` and captured `argv`. Exact `-` therefore remains
  `"-"` in a stdout payload, while `./-`, `-G`, and every other non-empty non-sentinel value are real file targets.

## 19. Error-Handling Model

This codebase uses boundary-aware mixed error handling:

- Program orchestration and most modules: return codes (`EXIT_SUCCESS`/`EXIT_FAILURE`).
- Argument parsing internals: exceptions for parsing/validation, converted to return codes at API boundary.
- Allocation internals: null smart-pointer returns for allocation failure.
- Parse/preflight failures, the general pipeline's early timer failure, and TLB setup/allocation failures before result
  initialization have no schema-valid payload. A stdout target remains empty and the diagnostic/process failure are
  authoritative on those paths.
- After a mode or sweep initializes a representable result, execution status and payload persistence are handled
  separately so interrupted, unsupported, error, or failed evidence can still reach the selected target.
- Metal backend calls: synchronous/noexcept status results; nil/NSError/command failures become stable reason codes plus
  separate raw diagnostics. GPU correctness/timer invalidity fails the run, initialized unsupported capability remains
  distinct and is serializable, graceful interruption is process success with invalid conclusions, and a backend-factory
  failure before result initialization leaves stdout empty.
- LLM config/factory checks, checked final planning, JSON-output peak estimation, runner/backend auxiliary admission,
  backend initialization, tagged-plan resolution, and resource preparation precede runner-result initialization and
  have no schema payload on `Failed`. An `Unsupported` lifecycle result initializes a representable terminal result and
  is serialized distinctly from `Failed`; neither path falls back to another backend. The runner contains backend-task and checkpoint exceptions
  as stable status/reason evidence, finalizes untouched slots deterministically, gives real failures precedence over
  interruption, and keeps graceful task-boundary interruption separate from conclusion validity. Backend release is
  attempted once before the command-terminal checkpoint; release failure becomes terminal failure evidence. A file
  measurement-checkpoint failure is terminal, releases resources, and is not retried.

Principle: no uncaught exceptions should escape to `main()` control flow.

## 20. Concurrency Model

- Bandwidth and pattern bandwidth paths are parallelized by thread--count configuration.
- Latency tests are intentionally single-threaded pointer-chase measurements.
- Cache bandwidth tests default to one worker unless the user explicitly overrides `--threads`. Cache and
  main-memory latency tests remain single-threaded dependent pointer chases regardless of the bandwidth worker count.
- Threaded work partitioning attempts cache-line-aware chunk handling to reduce false sharing effects.
- GPU mode uses one Metal command queue and serial compute encoders. GPU grid threads are selected from pipeline
  execution width and do not map to CPU `--threads`; the CLI rejects that option. The host runner does not submit the next
  operation until the synchronous current task and required validation reach terminal state.
- The common LLM runner treats each synchronous backend invocation as one atomic task and checks stop only before and
  after it. The active CPU adapter creates one worker team per excluded or measured scenario task. Workers read immutable
  worker-major descriptors after an all-or-cancel ready gate; partial startup cancels before any kernel call. One
  synchronized timer spans gate release through last-worker completion. Worker-local layers are ordered without a
  per-layer global barrier, and neither the runner nor CPU adapter polls signal state inside the ARM64 hot loop.

## 21. Measurement Caveats and Interpretation Under Load

The tool can be run on active systems with concurrent workloads, but interpretation must account for scheduler and memory-system contention.

Practical caveats:

- Heavy concurrent activity can inflate tail latency and depress bandwidth.
- Tail metrics (`P95`, `P99`) usually reveal contention more clearly than averages.
- Comparing runs across time requires similar background-load conditions.
- Small buffers can become cache-dominated. Larger main-memory-sized working sets reduce that effect, but buffer size
  alone does not prove that physical DRAM served the measured traffic.
- GPU private buffers live in unified memory, not separate VRAM. GPU schema 1 cannot verify DRAM residency, and both
  small and large buffers can include cache/dispatch/reduction effects. Copy uses logical read+write payload.
- CPU and GPU GB/s should not be directly compared because timing, kernels, parallelism, resource modes, cache effects,
  dispatch processing, and observable validation work differ.
- LLM effective model-payload GB/s and synthetic memory work units/s describe exact logical memory-only payload, not
  inference tokens/s or a hardware counter. Full-size weight/K/V mappings prevent proxy-buffer scaling but remain
  cacheable and do not prove physical DRAM service.
- LLM weights-only and KV-only are separate component baselines; mixed is one layer-interleaved timing interval and
  cannot be decomposed into independent component bandwidths. Its exact weight-vs-KV-read classification is payload-only
  and does not locate a measured bottleneck.

For high-confidence baselines, run repeated loops and analyze distributions rather than single-point values.

## 22. Verification and Test Expectations

Recommended validation commands:

- Build: `make`
- Unit tests (non-integration): `make test`
- Script-example JSON entry paths: `make test-script-examples`
- Integration-only: `make test-integration`
- Full test set: `make test-all`
- CLI help smoke check: `./memory_benchmark -h`
- GPU help smoke check: `./memory_benchmark --gpu-bandwidth --help`
- LLM help smoke check: `./memory_benchmark --llm-memory --help`
- Deterministic LLM tests:
  `./test_runner '--gtest_filter=ModeSelectorTest.*:LlmMemoryContractTest.*:LlmMemoryConfigTest.*:LlmMemoryWorkPlanTest.*:LlmMemoryExecutorTest.*:LlmMemoryRunnerTest.*:LlmMemoryJsonTest.*:LlmMemoryOutputTest.*:MessagesTest.*'`
- Real LLM ARM64 and bounded executable transport contracts:
  `./test_runner '--gtest_filter=*LlmMemory*Integration*:*ExecutableCliIntegration*'`
- Deterministic GPU-focused tests:
  `./test_runner --gtest_filter='GpuBandwidthParserTest.*:GpuMemoryBudgetTest.*:GpuRunnerTest.*:GpuJsonTest.*:GpuWorkPlanTest.*:GpuTimedAccumulatorOracleTest.*:ModeSelectorTest.*:HashUtilsTest.*'`
- Real Metal contract: `./test_runner '--gtest_filter=GpuMetalBackendIntegrationTest.*'`; unsupported hardware may skip
  this integration suite but does not replace deterministic unsupported-path coverage.

For narrow changes, prefer targeted `gtest` filters via `./test_runner --gtest_filter=...`.

The aggregate `make test-all` gate requires Python 3; after all GTest cases pass, it runs the focused script-example
entry test. `jq` is not required by this gate.

## 23. Source Map (Primary Entry Points)

- Program entry: `main.cpp`
- LLM memory profile and active CPU backend:
  - `src/llm_memory/llm_memory.cpp`
  - `src/llm_memory/llm_work_plan.cpp`
  - `src/llm_memory/llm_kv_layout.cpp`
  - `src/llm_memory/llm_backend.cpp`
  - `src/llm_memory/llm_cpu_backend.cpp`
  - `src/llm_memory/llm_executor.cpp`
  - `src/llm_memory/llm_runner.cpp`
  - `src/llm_memory/llm_json.cpp`
  - `src/llm_memory/llm_output.cpp`
  - `src/llm_memory/llm_environment.mm`
- Config parse/validate/derive:
  - `src/core/config/mode_selector.cpp`
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
  - `src/benchmark/benchmark_work_plan.cpp`
  - `src/benchmark/benchmark_statistics_collector.cpp`
  - `src/benchmark/sweep_runner.cpp`
  - `src/benchmark/bandwidth_tests.cpp`
  - `src/benchmark/latency_tests.cpp`
- Standalone TLB planning and scheduling:
  - `src/benchmark/tlb_sweep_planner.cpp`
  - `src/benchmark/tlb_measurement_scheduler.cpp`
  - `src/benchmark/tlb_runtime_policy.cpp`
  - `src/benchmark/tlb_chain.cpp`
  - `src/benchmark/tlb_boundary_detector.cpp`
  - `src/benchmark/tlb_analysis.cpp`
  - `src/benchmark/tlb_analysis_json.cpp`
- Standalone core-to-core latency:
  - `src/benchmark/core_to_core_latency_cli.cpp`
  - `src/benchmark/core_to_core_latency_runner.cpp`
  - `src/benchmark/core_to_core_sweep_runner.cpp`
  - `src/benchmark/core_to_core_latency_json.cpp`
- Pattern benchmark:
  - `src/pattern_benchmark/pattern_statistics_manager.cpp`
  - `src/pattern_benchmark/pattern_coordinator.cpp`
  - `src/pattern_benchmark/pattern_work_plan.cpp`
  - `src/pattern_benchmark/output.cpp`
- Standalone GPU bandwidth:
  - `src/gpu_bandwidth/gpu_bandwidth.cpp`
  - `src/gpu_bandwidth/gpu_work_plan.cpp`
  - `src/gpu_bandwidth/gpu_runner.cpp`
  - `src/gpu_bandwidth/gpu_json.cpp`
  - `src/gpu_bandwidth/gpu_backend.h`
  - `src/gpu_bandwidth/metal_gpu_backend.mm`
  - `src/gpu_bandwidth/gpu_kernels_source.h`
- Output:
  - `src/output/console/output_printer.cpp`
  - `src/output/json/json_output/*.cpp`
- Assembly kernels:
  - `src/asm/*.s`
  - `src/asm/core_to_core_latency.s` (core-to-core latency measurements; see [CORE_TO_CORE_WHITEPAPER.md](CORE_TO_CORE_WHITEPAPER.md))
  - `src/asm/llm_decode_memory.s` (LLM weight/KV reads, temporal appends, and observable checksums; see
    [LLM_MEMORY_PROFILE_WHITEPAPER.md](LLM_MEMORY_PROFILE_WHITEPAPER.md))
  - `src/asm/llm_decode_memory_paged.s` (paged KV table lookups, physical block access, appends, and observable
    checksums; see [LLM_MEMORY_PROFILE_WHITEPAPER.md](LLM_MEMORY_PROFILE_WHITEPAPER.md))
  - `src/asm/llm_prefill_memory.s` (full-prompt writes, tiled causal-prefix reads, and operation-ordinal checksums; see
    [LLM_MEMORY_PROFILE_WHITEPAPER.md](LLM_MEMORY_PROFILE_WHITEPAPER.md))
  - `src/asm/llm_prefill_memory_paged.s` (block-exclusive physical prompt writes, timed table lookups, exact partial
    prefix scans, and permutation-bound checksums; see [LLM_MEMORY_PROFILE_WHITEPAPER.md](LLM_MEMORY_PROFILE_WHITEPAPER.md))

## 24. GPU Metal Maintenance Policy

- Every macOS major/minor/security build change defines a new GPU comparison cohort. Before baseline comparison, rerun
  runtime compilation, exact read/write/copy/tail correctness, valid timestamp checks, automatic calibration, and a
  fixed-seed/fixed-work repeatability campaign.
- Every Xcode/Command Line Tools/SDK change requires clean production/test/coverage builds and real-Metal integration.
- A new Apple GPU is admitted by capabilities, then starts a new hardware cohort. Capability success must not be
  presented as performance validation.
- Changes to MSL version/source/compile options, storage or hazard modes, checksum/reduction, grid cap/geometry, payload
  accounting, timed command-buffer boundary, or interruption/completion semantics require methodology-version review and
  normally a new validation population/schema compatibility decision.
- Runtime source compilation is intentional: it preserves Command Line Tools builds without an optional Metal Toolchain,
  but it does not freeze driver-generated machine code. The exact source SHA-256 and macOS build are both cohort keys.
- More than a 10% fixed-work operation-median shift across macOS builds stops automatic rebaselining even if the new
  cohort's CV is acceptable; compiler/driver, environment, and counter evidence must be investigated and documented.
- Optional APIs such as Low Power Mode metadata remain availability-guarded. Missing optional metadata is unavailable,
  not a crash or fabricated value. Historical files are never rewritten to resemble a newer methodology.
- Instruments/GPU counters are an auditable separate run with tool/counter/raw-trace provenance. Capture overhead can
  perturb timing, and counter traffic is not substituted for the production exact-payload/GPU-time value.

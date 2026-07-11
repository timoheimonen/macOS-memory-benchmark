# macOS Memory Benchmark - Technical Specification

## 1. Scope and Status

This document specifies the current implementation in this repository (version `0.61.0`) for `memory_benchmark` on macOS Apple Silicon.

It is intentionally implementation-driven and reflects real behavior in code paths under `main.cpp`, `src/core`,
`src/benchmark`, `src/pattern_benchmark`, `src/gpu_bandwidth`, `src/output`, and `src/asm`.

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
- Use `caffeinate -i -d ./memory_benchmark ...` to prevent system sleep during longer measurement sessions on cooler systems.

## 4. High-Level Runtime Architecture

Main orchestration (`main.cpp`) follows this pipeline:

`select_primary_benchmark_mode` scans all mode flags before mode-specific parsing, so a command containing multiple
primary modes
fails deterministically instead of being routed by the first token. Core-to-core uses `CoreToCoreLatencyConfig`; GPU uses
`GpuBandwidthConfig`; TLB uses a dedicated branch that populates `BenchmarkConfig`. GPU is dispatched before the general
timer/parser pipeline and never calls CPU config validation or buffer/access derivation. The numbered pipeline below
applies to standard/pattern execution.

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
   - Pattern benchmark mode (`run_all_pattern_benchmarks`), whose coordinator owns and prepares one shared
     `PatternBuffers` source/destination pair for every pattern and loop.
8. Print loop results and aggregate statistics.
9. Optionally serialize JSON output.
10. Print total elapsed runtime.

Memory cleanup is RAII-based through `MmapPtr` custom deleters (`munmap` on scope exit).

GPU mode follows its own synchronous pipeline:

1. Parse and validate the exact GPU option whitelist and derive buffer bytes/seed.
2. Apply shared best-effort main-thread QoS and enter `BenchmarkSignalMaskGuard`.
3. Create the pure-C++ `GpuBackend` factory product and initialize the private Metal backend.
4. Verify default device, unified memory, Apple7 family, runtime MSL compilation/pipelines, `maxBufferLength`, and
   two-buffer memory budget.
5. Allocate two private/tracked data buffers plus one shared/tracked status buffer for the suite lifetime.
6. Resolve excluded automatic calibration or exact explicit work, then freeze read/write/copy plans.
7. Execute cyclic operation tasks. Each logical task is warmup → precondition → one timed command buffer → required
   validation, followed by status/counter/aggregate update and optional atomic checkpoint.
8. Release Metal resources, record the final allocation/environment snapshot, replace the final production checkpoint,
   render console results, and return success/failure according to explicit run status.

## 5. Configuration Model

Configuration state is represented by `BenchmarkConfig` (`src/core/config/config.h`).

### 5.1 User-facing control fields

- Main options: buffer size MB, iterations, loop count, output path, threads.
- General mode/config flags: `run_benchmark`, `run_patterns`, `analyze_tlb`, `only_bandwidth`, and `only_latency`.
- Standalone TLB state remains in `BenchmarkConfig` (`analyze_tlb`, density, seed, stride/chain settings, and common sweep
  fields) and is populated by the dedicated `--analyze-tlb` branch in `argument_parser.cpp`.
- `--analyze-core2core` is pre-routed in `main.cpp` and uses the separate `CoreToCoreLatencyConfig` parser/runner path.
- `--gpu-bandwidth` is pre-routed in `main.cpp` and uses separate `GpuBandwidthConfig`: per-buffer MB/bytes, optional
  explicit passes, loop count, output path, base seed/source, help state, and exact argv.
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
- `--analyze-core2core` uses dedicated mode parsing (outside `argument_parser.cpp`) and only allows optional `--output`, `--count`, `--latency-samples`, `--sweep`, and `--sweep-max-runs`. Its mode-specific loop default is `3`; the general loop default remains `1`. Core-to-core sweep supports `count` and `latency-samples`, rejects duplicate sweep keys, and atomically checkpoints the combined output after every attempted run; only a nested `status: "complete"` result with `measurements_complete: true` increments `completed_runs`. Direct and sweep execution use the shared scope-bound signal guard before creating workers and restore the calling thread's exact previous mask on every return path. Each scheduler-hint scenario runs an excluded pilot after a 1,000,000-round-trip calibration warmup, reuses its duration-calibrated plan across measured loops, and participates in a cyclic Latin-square scenario schedule. Full methodology and JSON schema 2 contract: [CORE_TO_CORE_WHITEPAPER.md](CORE_TO_CORE_WHITEPAPER.md).
- `--gpu-bandwidth` uses a dedicated parser outside `argument_parser.cpp`. It accepts only `-G`/`--gpu-bandwidth`,
  `-b`/`--buffer-size`, `-i`/`--iterations`, `-r`/`--count`, `--seed`, `-o`/`--output`, and
  `-h`/`--help`. Duplicates, unknown/incompatible options, missing values, partial numeric tokens, non-positive
  iterations/count, and signed seeds fail before Metal work. Defaults are 512 MB per buffer, three loops, generated
  seed, and automatic passes. Buffer size must be at least 64 MB; checked MB→bytes overflow and explicit work guardrails
  are resolved before backend creation. Copy's 2× payload makes its pass limit the strict CLI ceiling shared by all three
  operations. GPU schema 1 rejects sweep options.

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

- Main-memory latency accesses scale from base count relative to default buffer size.
- Cache latency access counts use fixed constants (`L1`, `L2`, `CUSTOM`).
- `--buffer-size 0` (in allowed mode) sets main latency accesses to zero.

## 8. Memory Allocation and Initialization

### 8.1 Allocation strategy

Allocation entrypoints:

- Standard benchmark mode: per-phase allocators in `src/benchmark/benchmark_executor.cpp` (`prepare_*_buffers` helpers).
- Pattern benchmark mode: `allocate_pattern_buffers` (`src/core/memory/buffer_allocator.cpp`). The pattern
  coordinator retains the pair for the full command and releases it on return.

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
- Collects chain diagnostics (pointer count, unique pages touched, page size, stride).

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

This auxiliary fixed-window comparison requires at least two stride-spaced nodes inside 16 KiB (stride at most 8192
bytes). It is not part of config-validator target eligibility: if an otherwise valid standard configuration uses a larger
stride, the comparison records an unavailable status while the configured target measurements can still run.

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

## 13. GPU Bandwidth Execution

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

With output enabled, the shared atomic writer checkpoints after each terminal measurement. The runner reads stop once
before and once immediately after that checkpoint; if the second read first observes the signal, it writes at most one
additional interruption checkpoint. Valid post-parse pre-run unsupported/backend/compile/allocation/work-plan failures
also produce one checkpoint. CLI/config errors, including a buffer below 64 MB, do not. A checkpoint failure stops the
run as failed; the file may remain at the preceding successful checkpoint.

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
for their exact hardware, OS, compiler, kernel, and methodology identity. The release-validation identity uses
`gpu-linear-word-mod32-tg-reduce-v2`, the frozen 8192-threadgroup cap, and canonical MSL SHA-256
`b9a242d2b959c9c11f6f130a52afd66f111d6761be2193beec1f051baa094296` and frozen release-binary SHA-256
`31ce0285dc5fde382d40e6d7b769c20e6f3363754bb3c7c0afbb4f13cd71a6a7`. Automatic read/write/copy
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

## 14. Assembly Kernel Layer

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

- `core_to_core_initiator_round_trips_asm`: Waits until the initiator owns the token, hands it to the responder, then
  waits for the responder to return it; repeats for the specified round trips.
- `core_to_core_responder_round_trips_asm`: Waits until the responder owns the token and hands it back to the
  initiator; coordinates via the shared token word.

## 15. Timing Model

Timing API: `HighResTimer` (`src/core/timing/timer.*`).

- Provides second and nanosecond stop methods.
- Used for both macro (test durations) and micro (latency sample windows) timing.
- Factory creation returns optional; failure is treated as fatal at call sites.

GPU primary timing does not use `HighResTimer`: it uses completed Metal command-buffer `GPUStartTime` and `GPUEndTime`.
Host steady-clock submit/wait/wall values are diagnostic. Total GPU-mode host execution time uses steady clock and is
separate from every operation denominator.

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

## 17. Console Output Contract

Console rendering is centralized in `src/output/console` and message helpers in `src/output/console/messages`.

Contract highlights:

- Configuration and cache info printed before execution.
- Per-loop results are printed in standard mode.
- Pattern mode prints pattern table-style sections and derived efficiency indicators.
- GPU mode prints a separate device/private-tracked header, read/write/copy effective-payload headlines, repeatability,
  and interpretation note. Copy is explicitly aggregate read + write and DRAM residency remains unverified.
- Aggregate statistics printed when loop count > 1.
- Errors and warnings use `Messages::error_prefix()` / `Messages::warning_prefix()` conventions.
- Live progress uses the shared spinner on `stderr` only when it is a TTY; redirected standard and pattern output contains no carriage-return control sequences.

## 18. JSON Output Contract

JSON writer API (`src/output/json/json_output/json_output.cpp`):

- Standard mode schema 2: `configuration`, `execution_time_sec`, completion counters/status, `results_complete`,
  per-loop `loops`, `main_memory`, `cache`, `timestamp`, and `version`.
- Pattern mode schema 3: `configuration`, `execution_time_sec`, command status/reason, planned/completed loop and
  measurement counters, `results_complete`, optional retained `patterns` evidence, `timestamp`, and `version`.
- TLB analysis mode: `configuration`, `execution_time_sec`, `tlb_analysis`, `timestamp`, `version`.
- Core-to-core schema 2: calibrated methodology configuration, `core_to_core_latency` command completion metadata, scenario work plans, nullable aggregate values, per-loop order/status/duration/hint/sample-boundary records, and affinity-comparison interpretability metadata.
- GPU schema 1: top-level mode/schema/methodology/status, exact counters and completeness, effective/copy/DRAM semantics,
  config/argv, environment, backend device/compile/allocation, memory budget, frozen plans, excluded calibration,
  status-bearing measurements/loop records, aggregates, and warnings.
- Sweep mode: `configuration.mode = "sweep"`, `configuration.base_mode`, `configuration.sweep_parameters`, top-level `status`, `status_reason`, `planned_runs`, `attempted_runs`, `completed_runs`, and `conclusions_valid`, plus per-entry `runs[].status`, `status_reason`, and `result`. Every attempted run is checkpointed and `attempted_runs == runs.size()`. `completed_runs` requires nested `status: "complete"` and `results_complete: true` for standard/pattern, `tlb_analysis.status: "complete"` and `tlb_analysis.conclusions_valid: true` for TLB, or `core_to_core_latency.status: "complete"` and `measurements_complete: true` for core-to-core. Partial, interrupted, and failed attempts remain as evidence without incrementing the completed count. Top-level `conclusions_valid` is true only when top-level status is complete and `completed_runs == planned_runs`.

Pattern schema 3 plans 21 measurements per loop and treats numeric measured values plus intentional skips as terminal.
Only Complete loops feed aggregate vectors, medians, statistics, and console summaries. Partial, interrupted, and failed
loop measurements remain in JSON as evidence, while command status/counters keep `results_complete` false. Preparation
failure may omit `patterns`; main and sweep orchestration still build the completion payload before returning or
classifying the failure.

### 18.1 Configuration keys

In addition to standard fields (buffer size, iterations, loop count, thread count, CPU/OS info):

- `latency_chain_mode` (string): Resolved pointer-chain construction mode.
- `use_latency_tlb_locality` (boolean): Whether the configured locality value is greater than zero; an explicit
  `global-random` mode can still ignore that value.
- `latency_tlb_locality_bytes` (number): TLB-locality window size in bytes.
- `latency_tlb_locality_kb` (number): TLB-locality window size in KB.
- `benchmark_schema_version` (number): `2`.
- `methodology_version` (string): `benchmark-v2-calibrated-seeded-balanced`.
- `benchmark_seed` (string): exact uint64 decimal string plus source/encoding fields.
- Calibration targets/windows and phase/operation schedule policies.

### 18.2 Main-memory latency keys

- `main_memory.latency.headline_ns`: status, median-or-single headline, loop values, robust statistics, measurement
  records, quality, and optional pooled separate-sample distribution with loop boundaries.
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

### 18.6 Path behavior

- Relative `--output` paths are resolved against current working directory.

## 19. Error-Handling Model

This codebase uses boundary-aware mixed error handling:

- Program orchestration and most modules: return codes (`EXIT_SUCCESS`/`EXIT_FAILURE`).
- Argument parsing internals: exceptions for parsing/validation, converted to return codes at API boundary.
- Allocation internals: null smart-pointer returns for allocation failure.
- Metal backend calls: synchronous/noexcept status results; nil/NSError/command failures become stable reason codes plus
  separate raw diagnostics. GPU correctness/timer invalidity fails the run, unsupported capability remains distinct, and
  graceful interruption is process success with invalid conclusions.

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

## 21. Measurement Caveats and Interpretation Under Load

The tool can be run on active systems with concurrent workloads, but interpretation must account for scheduler and memory-system contention.

Practical caveats:

- Heavy concurrent activity can inflate tail latency and depress bandwidth.
- Tail metrics (`P95`, `P99`) usually reveal contention more clearly than averages.
- Comparing runs across time requires similar background-load conditions.
- Small buffers can become cache-dominated and hide DRAM behavior.
- GPU private buffers live in unified memory, not separate VRAM. GPU schema 1 cannot verify DRAM residency, and both
  small and large buffers can include cache/dispatch/reduction effects. Copy uses logical read+write payload.
- CPU and GPU GB/s should not be directly compared because timing, kernels, parallelism, resource modes, cache effects,
  dispatch processing, and observable validation work differ.

For high-confidence baselines, run repeated loops and analyze distributions rather than single-point values.

## 22. Verification and Test Expectations

Recommended validation commands:

- Build: `make`
- Unit tests (non-integration): `make test`
- Integration-only: `make test-integration`
- Full test set: `make test-all`
- CLI help smoke check: `./memory_benchmark -h`
- GPU help smoke check: `./memory_benchmark --gpu-bandwidth --help`
- GPU pure tests: `./test_runner --gtest_filter='GpuWorkPlanTest.*:ModeSelectorTest.*:HashUtilsTest.*'`
- Real Metal contract: `./test_runner '--gtest_filter=GpuMetalBackendIntegrationTest.*'`; unsupported hardware may skip
  this integration suite but does not replace deterministic unsupported-path coverage.

For narrow changes, prefer targeted `gtest` filters via `./test_runner --gtest_filter=...`.

## 23. Source Map (Primary Entry Points)

- Program entry: `main.cpp`
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
  - `src/benchmark/bandwidth_tests.cpp`
  - `src/benchmark/latency_tests.cpp`
- Standalone TLB planning and scheduling:
  - `src/benchmark/tlb_sweep_planner.cpp`
  - `src/benchmark/tlb_measurement_scheduler.cpp`
  - `src/benchmark/tlb_analysis.cpp`
- Pattern benchmark:
  - `src/pattern_benchmark/pattern_coordinator.cpp`
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

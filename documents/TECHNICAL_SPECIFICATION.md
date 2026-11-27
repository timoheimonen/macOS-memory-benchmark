# macOS Memory Benchmark - Technical Specification

## 1. Scope and Status

This document specifies the current `memory_benchmark` implementation in this repository for macOS Apple Silicon.

It is intentionally implementation-driven and reflects real behavior in code paths under `main.cpp`, `src/core`,
`src/cpu_bandwidth`, `src/cpu_latency`, `src/benchmark`, `src/pattern_benchmark`, `src/gpu_bandwidth`, `src/llm_memory`,
`src/output`, and `src/asm`.

Primary goals:

- Define runtime architecture and execution flow.
- Define command/config semantics and validation rules.
- Define memory allocation, initialization, and benchmark execution contracts.
- Define output contracts (console and JSON).
- Capture current constraints, known drift, and measurement caveats.

Out of scope:

- Detailed `--cpu-bandwidth` methodology and schema field catalog (see
  [CPU_BANDWIDTH_WHITEPAPER.md](CPU_BANDWIDTH_WHITEPAPER.md)).
- Detailed `--cpu-latency` pointer-chase methodology and schema field catalog (see
  [LATENCY_WHITEPAPER.md](LATENCY_WHITEPAPER.md)).
- Detailed `--patterns` 7-pattern methodology, GF(2)/random oracles, and schema field catalog (see
  [PATTERNS_WHITEPAPER.md](PATTERNS_WHITEPAPER.md)).
- `--analyze-tlb` methodology details (see [TLB_ANALYSIS_WHITEPAPER.md](TLB_ANALYSIS_WHITEPAPER.md)).
- `--analyze-core2core` methodology details (see [CORE_TO_CORE_WHITEPAPER.md](CORE_TO_CORE_WHITEPAPER.md)).
- Detailed `--gpu-bandwidth` methodology and schema field catalog (see
  [GPU_BANDWIDTH_WHITEPAPER.md](GPU_BANDWIDTH_WHITEPAPER.md)).
- Detailed `--llm-memory` schema-2 vocabulary, active decode/prefill methodologies, and field catalog (see
  [LLM_MEMORY_PROFILE_WHITEPAPER.md](LLM_MEMORY_PROFILE_WHITEPAPER.md)).
- Historical behavior from older releases.

## 2. Platform and Build Constraints

- Target OS: macOS.
- Target CPU architecture: ARM64 Apple Silicon.
- Language: C++17 with ARM64 Apple Silicon assembly kernels, two Objective-C++ Metal backends, and one Objective-C++
  environment-capture unit.
- Build: `Makefile` (`clang++`, `as`), with Python 3 required to generate embedded build provenance.
- Test frameworks: GoogleTest (`test_runner`) and Python `unittest` for script examples, the independent LLM
  artifact verifier (`test-llm-verifier`), the independent patterns artifact verifier (`test-patterns-verifier`), and
  the independent CPU bandwidth/latency/patterns artifact verifiers (`test-cpu-verifiers`). Objective-C++ test sources
  are compiled with test flags and ARC.
- Build provenance records Git revision/dirty state, compiler, target, SDK, deployment target, and build/link flags.
  Make regenerates the manifest and invalidates objects when these inputs change; unavailable Git metadata remains null.
- Deployment target: macOS 26.0 for production/test sources, assembly, and final links. External static archives must
  not require a newer OS. This is the oldest runtime family exercised by the current repeatable release gate.
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

`select_primary_benchmark_mode` scans all seven primary mode flags (`-B`/`--cpu-bandwidth`, `-L`/`--cpu-latency`,
`-P`/`--patterns`, `-T`/`--analyze-tlb`, `-C`/`--analyze-core2core`, `-G`/`--gpu-bandwidth`, `-M`/`--llm-memory`)
before mode-specific parsing, skipping one opaque value token for every known space-separated value-taking option across
all seven modes and recognizing `--mode=value`, so a command containing multiple primary modes fails deterministically
instead of being routed by the first token and flag-shaped option values (such as `--cpu-bandwidth --buffer-size -G` or
`--cpu-latency --sweep -T`) do not trigger false mode conflicts.
CPU bandwidth uses `CpuBandwidthConfig`; CPU latency uses `CpuLatencyConfig`; Patterns uses `PatternsConfig`; Core-to-core uses `CoreToCoreLatencyConfig`;
GPU uses `GpuBandwidthConfig`; LLM-memory uses `LlmMemoryConfig`; standalone TLB populates `BenchmarkConfig`. All seven
modes share the descriptor-driven CLI option scanner in `src/core/config/cli_parser.h` / `.cpp`.
Running with no arguments (`argc <= 1`) prints help to stdout and exits `0` before timer creation or mode selection.
`main.cpp` wraps mode selection and argument parsing to contain unexpected exceptions, pre-routes CPU bandwidth
(`run_cpu_bandwidth_mode`), CPU latency (`run_cpu_latency_mode`), Patterns (`run_patterns_mode`), LLM-memory (`run_llm_memory_mode`), GPU
(`run_gpu_bandwidth_mode`), and core-to-core (`run_core_to_core_latency_mode`) before the general `BenchmarkConfig`
pipeline, and uses the numbered pipeline below for `-T`/`--analyze-tlb` and TLB sweeps.

1. Parse CLI arguments into `BenchmarkConfig` (`parse_arguments`, single-pass via `parse_cli_options`) and finish
   human-facing `-h`/`--help` handling before OS sysctl queries, cache detection, seed generation, or timer creation.
2. Reject modifier-only invocations with no primary mode flag (`validate_config`) before creating `JsonOutputSession`,
   returning `EXIT_FAILURE` with empty stdout and a centralized stderr error.
3. Classify the raw output target. TLB commands and sweeps using `-` create a command-scoped
   `JsonOutputSession` that routes subsequent human `std::cout` output to stderr while retaining the original stdout
   buffer for final JSON.
4. Validate configuration (`validate_config`) while the supported machine-output routing is active (including
   direct-vs-`--sweep` conflict checks and memory limits).
5. Create high-resolution total-execution timer (when needed for execution).
6. Print the shared runtime banner and TLB run/resource summary.
7. Raise main thread QoS (`QOS_CLASS_USER_INTERACTIVE`) best-effort.
8. Execute standalone TLB analysis or TLB sweeps via `src/benchmark/tlb_*` and `sweep_runner.cpp`.
9. Print summary results.
10. Build any representable terminal payload, dispatch it according to the mode-aware file/stdout lifecycle, and preserve
    execution status separately from output status.
11. Print total elapsed runtime, restore the previous stdout buffer at scope exit, and return the combined status.

Memory cleanup is RAII-based through `MmapPtr` custom deleters (`munmap` on scope exit).

CPU Bandwidth (`run_cpu_bandwidth_mode`), CPU Latency (`run_cpu_latency_mode`), and Patterns (`run_patterns_mode`) each follow their own standalone
synchronous pipeline (mirroring LLM's audit and checkpoint discipline):

1. Parse the exact 12-option (`-B`), 16-option (`-L`), or 11-option (`-P`) whitelist via `parse_cli_options`, enforce strict unsigned ranges
   and sweep syntax/conflicts, and handle `-h`/`--help` before OS queries, target-omission/sweep checks, or file I/O.
2. Build `CpuBandwidthConfig` / `CpuLatencyConfig` / `PatternsConfig` (resolving seed, workers, and sweep preflight).
3. Classify the raw output target (`ResolveAgainstCurrentDirectory`) and install a command-scoped `JsonOutputSession`
   before the runtime banner. When `--sweep` is active, dispatch `run_cpu_bandwidth_sweep` / `run_cpu_latency_sweep` / `run_patterns_sweep`
   through `execute_sweep_plan(SweepNestedMode::CpuBandwidth | CpuLatency | Patterns, ...)`.
4. Capture host environment (`capture_host_environment()`) and executable SHA-256 (`capture_binary_sha256()`), build the
   runtime `build_manifest` (`src/core/audit/build_manifest.*`), apply best-effort `QOS_CLASS_USER_INTERACTIVE`, and
   enter `BenchmarkSignalMaskGuard`.
5. Build `CpuBandwidthWorkPlan` / `CpuLatencyWorkPlan` / `PatternsWorkPlan` with strict `memory_budget` preflight (no silent truncation).
6. Prepare persistent `mprotect(PROT_READ)` source buffers (`-B`, `-P`) or Hamiltonian-prevalidated `mprotect(PROT_READ)`
   pointer chains (`-L`), run excluded calibration and frozen-plan warmups, execute balanced cyclic loops with bounded
   `K=max(1,ceil(count/8))` atomic file checkpoints (`stdout` skips intermediate serialization), render console summary
   and quality warnings, and emit terminal JSON.

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

LLM-memory follows its own synchronous pipeline. CPU and Metal decode/prefill are active with contiguous
or deterministic paged KV:

1. Scan the complete standalone whitelist, parsing supplied values and rejecting unknown, duplicate, or missing-value
   input even when help is present. Human help returns after that scan but before required/default/platform work.
2. Require the five common model inputs plus phase-specific decode context or prefill prompt/tile geometry (either via
   direct flags or via compatible repeatable `--sweep` axes); resolve CPU-by-default, decode-by-default,
   contiguous-by-default or explicitly paged layout, CPU requested/detected workers when applicable, explicit or
   automatic work, and one user/generated base seed; enforce IEEE-754 `2^53 - 1` JSON safe-integer bounds; and run
   checked geometry and per-scenario work-limit preflight (for every expanded Cartesian combination when `--sweep` is
   set). Paged layout requires exactly one valid block-token size (direct or swept), while contiguous rejects it.
   Metal rejects explicit workers; both backends support both phases and layouts.
3. Classify the raw output target (resolving relative file targets against the current working directory via
   `JsonFilePathPolicy::ResolveAgainstCurrentDirectory`) and install one command-scoped `JsonOutputSession` before the
   runtime banner. When `--sweep` is active, dispatch `run_llm_memory_sweep`, which prints one runtime banner, enters
   one command-scoped `BenchmarkSignalMaskGuard`, creates one `HighResTimer`, and delegates to
   `execute_sweep_plan(SweepNestedMode::LlmMemory, ...)`. Each nested sweep point runs `run_llm_memory_collect` with
   sweep fields and `output_file` cleared and `collect_json_for_sweep = true`, creating and releasing its own backend
   and resources within the point, and checkpoints the schema-1 sweep envelope atomically after each attempted run (or
   emits one terminal envelope to stdout). Direct commands continue through steps 4–8 below.
4. Request best-effort main-thread QoS, enter `BenchmarkSignalMaskGuard`, and capture CPU/OS/page/cache/available-memory
   plus initial thermal/Low Power Mode evidence.
5. Create the selected pure-C++ `LlmBackend` factory product. Build a preliminary pointer-free logical work plan
   (`prepare_llm_memory_work_plan`). For Metal, initialize the backend here so runtime `maxBufferLength` and Tier-2
   argument-encoder sizing feed preflight execution-plan construction before auxiliary estimation. Ask the backend and
   runner for their exact auxiliary backing, then add the conservative JSON DOM/serialization peak whenever
   `!output_file.empty() || collect_json_for_sweep`. Its checked variable-string term includes component/layout
   identities and all prefill execution/scenario/scope identities. Canonical scenario identities and expected checksums
   are retained once per distinct plan; runtime evidence scales with planned measurements. Admission includes
   canonical-plan storage, expected reconstruction transients, accepted-ID maps, exact-statistics workspaces, and the
   simultaneous DOM/serialization peak. Bounded snapshot frequency does not reduce the single-snapshot peak allowance.
   Preliminary and finalized plans use the same canonical identity-size formula. Rebuild and admit the final plan (for
   Metal, rebuilding the runtime execution plan with `additional_owned_bytes` covering planner, checksum, and
   orchestration storage) with the full page-rounded weight/physical-K/physical-V mappings, paged table and validation
   transient when applicable, descriptor/planner storage, checksum storage, and orchestration storage included in one
   peak budget.
6. Initialize the backend (already performed in step 5 for Metal), resolve the plan's exactly one tagged backend
   execution-plan variant, and prepare resources.
   The CPU adapter creates `HighResTimer`, atomically allocates and initializes/pre-touches the layout-specific mappings,
   materializes and validates the paged table when applicable, materializes the worker-major layout-specific descriptor
   ABI, and binds the matching production ARM64 executor. The Metal adapter probes Apple7/unified-memory/Tier-2/
   maximum-buffer capability, compiles the common foundation, the selected phase/layout scenario pipelines, and the
   selected profile's K/V-write/padding-validation pipeline. It creates private W/K/V segments, an optional segmented
   private table, and shared argument/status/staging storage, then initializes and validates them outside task time.
   An initialized lifecycle
   `Unsupported` result remains distinct from a runtime failure and never falls back to another backend.
7. Resolve excluded scenario-specific calibration in canonical weights-only/KV-only/mixed order, or validate exact
   explicit work. Automatic calibration warms the initial pilot shape and adds only the conditional one-work-unit
   confirmation warmup; explicit work does not run a pilot. Atomically freeze all three resolved plans, run their
   same-shape frozen warmups in canonical order, and only then execute cyclic measured tasks through whole synchronous
   backend calls. Each call owns reset, timed work, correctness validation, and generic task evidence. A Metal call
   uses one reset command buffer, one timed command buffer/serial encoder/workload dispatch with its T loop in-kernel,
   authoritative `GPUStartTime`/`GPUEndTime`, dual-mod32 checksum comparison, and excluded phase-neutral K/V-write
   validation. Stop remains observable at each task boundary. File progress snapshots are offered only after every Kth
   completed loop, with `K=max(1,ceil(count/8))`; exact aggregate statistics are prepared at snapshots and terminal finalization.
8. Release backend resources exactly once before offering the command-terminal checkpoint, then capture final
   environment state, render the centralized console report, and either retain the runner-owned atomic file cadence or
   emit one final schema-2 stdout document. Execution status and output status remain separate.

Pre-run failures before runner-result initialization leave stdout empty. Once initialized,
partial/interrupted/unsupported/failed evidence is representable; a file checkpoint failure is terminal and is not
retried at an outer final-write boundary.

## 5. Configuration Model

Standalone TLB configuration state is represented by `BenchmarkConfig` (`src/core/config/config.h`); `-B`/`--cpu-bandwidth`, `-L`/`--cpu-latency`, `-P`/`--patterns`, `-C`/`--analyze-core2core`, `-G`/`--gpu-bandwidth`, and `-M`/`--llm-memory` each use independent, dedicated configuration and work-plan structures.

### 5.1 User-facing control fields

- Main options: buffer size MB, iterations, loop count, raw output target, threads. The configuration fields are
  named `output_file`, and command adapters classify exact `-` before interpreting every other non-empty value as a
  path. An empty raw value disables JSON for direct commands and is missing/invalid for sweeps.
- General mode/config flag in `BenchmarkConfig`: `analyze_tlb`.
- `--cpu-bandwidth` is pre-routed in `main.cpp` and uses `CpuBandwidthConfig`: buffer size MB (`0` to omit), optional custom
  cache KB (`0` to omit), requested workers, optional fixed passes (`1..1,000,000,000`), loop count (default `3`),
  non-cacheable hint, seed, raw output target, sweep controls (`buffer-size`, `cache-size`, `threads`), and exact `argv`.
- `--cpu-latency` is pre-routed in `main.cpp` and uses `CpuLatencyConfig`: buffer size MB (`0` to omit), optional custom
  cache KB (`0` to omit), optional fixed complete chain cycles (`1..1,000,000,000`), loop count (default `3`), sample
  window count (`0` to disable, default `1000`), 8-byte-aligned stride (default `256`), chain mode (`auto` default),
  TLB locality KB (default `1024`), explicit `--compare-locality` flag, non-cacheable hint, seed, raw output target,
  sweep controls (`buffer-size`, `cache-size`, `latency-tlb-locality-kb`, `latency-stride-bytes`, `latency-chain-mode`),
  and exact `argv`.
- `--patterns` is pre-routed in `main.cpp` and uses `PatternsConfig`: positive buffer size MB (`1..2^53 - 1`, default `512`),
  requested workers (`1..UINT32_MAX`, default logical cores, preserving `requested_workers` and clamping `effective_workers`
  per pattern), optional fixed passes (`1..1,000,000,000`), loop count (default `3`, max `2^53 - 1`), non-cacheable hint,
  seed, raw output target, sweep controls (`buffer-size`, `threads`, `--sweep-max-runs` default `256`, max `2^53 - 1`),
  and exact `argv`.
- Standalone TLB state remains in `BenchmarkConfig` (`analyze_tlb`, density, seed, stride/chain settings, and common sweep
  fields) and is populated by the dedicated `--analyze-tlb` branch in `argument_parser.cpp`.
- `--analyze-core2core` is pre-routed in `main.cpp` and uses the separate `CoreToCoreLatencyConfig` parser/runner path.
- `--gpu-bandwidth` is pre-routed in `main.cpp` and uses separate `GpuBandwidthConfig`: per-buffer MB/bytes, optional
  explicit passes, loop count, raw output target, base seed/source, help state, and exact argv.
- `--llm-memory` is pre-routed in `main.cpp` and uses separate `LlmMemoryConfig`: resolved backend/phase/layout enums,
  required common and phase-specific model geometry, KV width/batch defaults, nullable requested/detected worker counts,
  optional exact scenario work units, loop count, raw output target, sweep controls (`run_sweep`, `sweep_max_runs`,
  `sweep_specs`, and internal `collect_json_for_sweep` for nested sweep collection), base seed/source, help state, and
  exact argv. The immutable
  `LlmMemoryWorkPlan` separately records common phase-applicable geometry, exact model payload/accounted work, memory
  admission, layout metadata, domain seeds, methodology, exact component identities, and exactly one tagged
  backend-execution-plan variant. The CPU alternative owns requested/available/effective workers plus its descriptor
  templates and storage accounting. The active Metal alternative owns phase-specific contiguous or paged segments,
  Tier-2 argument-buffer and parameter ABIs, scenario pipeline/grid, resource, admission, GPU timing, lookup, and
  checksum/validation evidence.
- Cache behavior in `-B`/`-L`: auto L1/L2 detection or user-provided `--cache-size` (`0` to omit).
- Latency sampling in `-L`: `latency_samples` (default `1000`, `0` to disable).
- Latency-chain construction mode in `-L`/`-T`: `latency_chain_mode` (type `LatencyChainMode`, CLI flag `--latency-chain-mode`).
- TLB-locality control for latency chain construction:
  - `latency_tlb_locality_kb` / `latency_tlb_locality_bytes` (default `1024` KB)
  - `0` selects `global-random` when mode is `auto`.
- Best-effort cache-discouraging mode (`-B`, `-L`, `-P`): `non_cacheable` / `use_non_cacheable`.

### 5.2 Derived fields

- Resolved byte sizes for main and cache buffers.
- Fallback access counts for latency paths; normal measured work is resolved by pilot calibration.
- System metadata (CPU name, macOS version, core counts).
- Max memory limits and bookkeeping flags.

## 6. CLI Parsing and Validation

### 6.1 Parsing behavior (`cli_parser.cpp`, `mode_selector.cpp`, `argument_parser.cpp`)

- Shared descriptor-driven parsing (`src/core/config/cli_parser.h` / `.cpp`):
  - All seven primary modes (`--cpu-bandwidth`, `--cpu-latency`, `--patterns`, `--analyze-tlb`, `--analyze-core2core`,
    `--gpu-bandwidth`, and `--llm-memory`) share `parse_cli_options` over declarative `CliOptionSpec` tables.
  - Long options that take values accept both space-separated (`--buffer-size 512`, `--output -`) and inline
    `--option=value` forms (`--buffer-size=512`, `--output=-`, `--output=`, `--sweep=buffer-size=128,256`). Passing an
    inline value to a boolean or mode flag (such as `--cpu-bandwidth=1`, `--cpu-latency=1`, `--compare-locality=1`, or
    `--help=1`) is rejected as `UnexpectedValue` (`Option --... does not take a value`). Single-dash short options remain
    space-separated one-character flags; legacy single-dash long names (`-buffersize`) and single-dash inline forms
    (`-b=512`) are rejected as unknown options.
  - All single-occurrence options (including primary mode selectors `-B`, `-L`, `-P`, `-T`, `-C`, `-G`, `-M` and boolean
    flags `--compare-locality`, `-u`, `-h`) are uniformly rejected when duplicated across all seven modes. Only
    `-S`/`--sweep` is repeatable (once per distinct parameter key; duplicate keys fail with
    `Messages::error_duplicate_sweep_parameter`, i.e., `sweep parameter specified more than once: <key>`).
  - Mode pre-scan (`mode_selector.cpp`) uses `is_known_cli_value_option` and `extract_long_option_name` to skip one
    opaque following value token for every known space-separated value-taking option across all seven modes (so
    flag-shaped values such as `--cpu-bandwidth --buffer-size -G` or `--cpu-latency --sweep -T` route to the active mode
    and report a value error rather than a false primary-mode conflict) and to route `--mode=value` to that mode for a
    clean unexpected-value diagnostic.
  - Every mode parser resets its config struct on entry, guards `argc == 0` and null `argv` entries via
    `cli_program_name`, emits concise single-line `Error: ...` diagnostics on `stderr` through centralized `Messages::`
    helpers without dumping full usage text on parse errors, and contains exceptions at parser/command boundaries.
    Standalone whitelist modes (`--cpu-bandwidth`, `--cpu-latency`, `--patterns`, `--analyze-tlb`, `--analyze-core2core`,
    `--gpu-bandwidth`, `--llm-memory`) name the offending option explicitly (`Unsupported option '<token>': ...`).
- Single-pass TLB parse (`argument_parser.cpp`):
  - Options are scanned once via `parse_cli_options`.
  - Strict integer parsing uses `std::from_chars` with centralized `Messages::reason_*` helpers; `--buffer-size` uses
    LP64-safe `parse_strict_nonnegative_ulong` covering `[0, ULONG_MAX]` without `long long` truncation.
  - OS `sysctl` queries, cache detection, and seed generation are deferred until after syntactic parsing
    and `-h`/`--help` handling. Standalone TLB parsing honors `ConfigTestHooks` for system information, while `-P`
    uses `PatternsParserTestHooks` / `PatternsCommandTestHooks`.
- Uniform `-h`/`--help` and no-mode behavior:
  - All seven modes (including `--analyze-tlb --help`) accept `-h`/`--help`. Across every mode, `--help` still runs full
    syntactic, whitelist, duplicate, missing-value, and value-parse checks first, and once valid prints human-facing
    help to stdout and exits `0` before required-option checks, semantic validation, OS queries, timer creation, seed
    generation, or output-session setup (`--output - --help` and `--help --output -` remain human-facing help to stdout
    across all seven modes).
  - Running with no arguments (`argc <= 1`) prints help to stdout and exits `0`. Running with modifier options but no
    primary mode flag fails in validation before output-session setup with `EXIT_FAILURE`, empty stdout, and a
    centralized stderr error (`error_seed_requires_supported_mode` or `error_missing_primary_mode`).
- TLB output-target classification occurs only after successful parsing, `--help`, and primary-mode checks. CPU
  bandwidth, CPU latency, patterns, core-to-core, GPU, and LLM each classify their target after their dedicated parser and help
  handling. For every result-producing direct command and supported sweep, exact `-` is reserved for final stdout JSON.
  An empty value disables JSON for a direct command and is missing/invalid for a sweep. Every other non-empty value is
  a file target, including `./-` and flag-shaped names such as `-G`.
- `--cpu-bandwidth` (`src/cpu_bandwidth/cpu_bandwidth.cpp`) uses an exact 12-option whitelist (`-B/--cpu-bandwidth`,
  `-b/--buffer-size`, `-k/--cache-size`, `-t/--threads`, `-i/--iterations`, `-r/--count`, `--seed`, `-u/--non-cacheable`,
  `-o/--output`, `-S/--sweep`, `-X/--sweep-max-runs`, `-h/--help`). Strict unsigned parsing rejects negative/malformed
  tokens (`-i`: `1..1,000,000,000`; `-b`: `0..2^53 - 1`; `-k`: `0` or `16..1048576`; `-t`: `1..UINT32_MAX`; `-r`/`-X`:
  `1..2^53 - 1`), rejects `--buffer-size 0 --cache-size 0`, supports sweeps over `buffer-size`, `cache-size`, and
  `threads`, and rejects conflicting direct+sweep flags.
- `--cpu-latency` (`src/cpu_latency/cpu_latency.cpp`) uses an exact 16-option whitelist (`-L/--cpu-latency`,
  `-b/--buffer-size`, `-k/--cache-size`, `-i/--iterations`, `-r/--count`, `-n/--latency-samples`,
  `-s/--latency-stride-bytes`, `-m/--latency-chain-mode`, `-l/--latency-tlb-locality-kb`, `--compare-locality`, `--seed`,
  `-u/--non-cacheable`, `-o/--output`, `-S/--sweep`, `-X/--sweep-max-runs`, `-h/--help`). Enforces `-i`:
  `1..1,000,000,000`, `-n`: `0..INT_MAX`, `-s`: `8..UINT32_MAX` (8-byte aligned), `-l`: `0..2^53 - 1` (page-aligned
  when non-zero in locality modes), `--buffer-size 0 --cache-size 0` rejection, `--compare-locality` eligibility
  (`--buffer-size > 0` and `--latency-stride-bytes <= 8192`), and sweeps over `buffer-size`, `cache-size`,
  `latency-tlb-locality-kb`, `latency-stride-bytes`, and `latency-chain-mode`.
- `--patterns` (`src/pattern_benchmark/patterns.cpp`) uses an exact 11-option whitelist (`-P/--patterns`,
  `-b/--buffer-size`, `-t/--threads`, `-i/--iterations`, `-r/--count`, `--seed`, `-u/--non-cacheable`, `-o/--output`,
  `-S/--sweep`, `-X/--sweep-max-runs`, `-h/--help`). Enforces `-b`: `1..2^53 - 1` (`0` rejected), `-t`: `1..UINT32_MAX`,
  `-i`: `1..1,000,000,000`, `-r`/`-X`: `1..2^53 - 1` (defaults `-b 512`, `-r 3`, `-X 256`), `--seed` long-only
  `0..UINT64_MAX`, sweeps over `buffer-size` and `threads` with conflicting direct+sweep rejection and full
  `preflight_patterns_sweep` validation before run `0`.
- `--latency-chain-mode` accepts string values and resolves to `LatencyChainMode` enum.
- `--analyze-tlb` uses a dedicated descriptor-driven parse branch in `argument_parser.cpp`. It only allows optional
  `--output`, `--latency-stride-bytes`, `--latency-chain-mode`, `--tlb-density`, `--seed`, `--sweep`,
  `--sweep-max-runs`, and `-h`/`--help`. TLB sweep supports `latency-stride-bytes`, `latency-chain-mode`, and
  `tlb-density`, rejects combining a direct option with `--sweep` for the same parameter, uses default run guard `16`,
  and rejects `global-random` chain mode. One generated or user-provided seed drives the pure sweep planner, seeded
  cyclic Latin round scheduler, derived task seeds, layout-specific page-native chain permutations, and deterministic
  convergence bootstrap. Each task measures a verified one-node-per-page spread chain and an equal-cache-line packed
  control in the same round. A pilot calibrates whole-chain accesses toward the quick/standard/exhaustive target
  duration; rounds stop at the per-point CI-width target or profile maximum. Candidate buffers are admitted only when
  their predicted buffer-plus-scratch peak fits the available-memory budget. Full methodology and JSON contract:
  [TLB_ANALYSIS_WHITEPAPER.md](TLB_ANALYSIS_WHITEPAPER.md).
- `--analyze-core2core` uses dedicated mode parsing built on `cli_parser.h` (outside `argument_parser.cpp`) and only
  allows optional `--output`, `--count`, `--latency-samples`, `--sweep`, `--sweep-max-runs`, and `-h`/`--help`. Its
  mode-specific loop default is `3`; the general loop default remains `1`. Core-to-core sweep supports `count` and
  `latency-samples`, rejects duplicate sweep keys via `Messages::error_duplicate_sweep_parameter`, rejects combining a
  direct option (`--count` or `--latency-samples`) with `--sweep` for the same key via
  `Messages::error_sweep_option_conflict`, wraps parser execution in exception containment at the command boundary, and
  atomically checkpoints a real-file combined output after every attempted run; an empty run plan or a
  stop observed before a run checkpoints a terminal zero-attempt envelope. Stdout keeps the same state transitions but
  emits only the terminal envelope. Only a nested `status: "complete"` result with `measurements_complete: true`
  increments `completed_runs`.
  Direct and sweep execution use the shared scope-bound signal guard before creating workers and restore the calling
  thread's exact previous mask on every return path. Each scheduler-hint scenario runs an excluded pilot after a
  1,000,000-round-trip warmup intended to reduce pilot startup transients, reuses its duration-calibrated plan across
  measured loops, and participates in a cyclic Latin-square scenario schedule. The result is effective acquire/release
  token-protocol round-trip time, not an isolated physical cache-line migration or coherence-fabric latency. Full
  methodology and JSON schema 2 contract: [CORE_TO_CORE_WHITEPAPER.md](CORE_TO_CORE_WHITEPAPER.md).
- `--gpu-bandwidth` uses a dedicated parser built on `cli_parser.h` (outside `argument_parser.cpp`). It accepts only
  `-G`/`--gpu-bandwidth`, `-b`/`--buffer-size`, `-i`/`--iterations`, `-r`/`--count`, `--seed`, `-o`/`--output`, and
  `-h`/`--help`. Duplicates, unknown/incompatible options (named explicitly), unexpected inline values, missing values,
  partial numeric tokens, non-positive iterations/count, and signed seeds fail before Metal work. Defaults are 512 MB
  per buffer, three loops, generated seed, and automatic passes. `--buffer-size` uses LP64-safe `[0, ULONG_MAX]` range
  parsing before enforcing the 64 MB minimum, checked MB→bytes overflow, and explicit work guardrails before backend
  creation. Copy's 2× payload makes its pass limit the strict CLI ceiling shared by all three operations. GPU schema 1
  rejects sweep options.
- `--llm-memory` uses a dedicated parser built on `cli_parser.h` (outside `argument_parser.cpp`). It accepts only
  `-M`/`--llm-memory`, optional `--llm-memory-backend <cpu|metal>`, the five
  required weight/layer/head inputs, phase and its context or prompt/tile geometry, optional KV width, batch,
  `--kv-layout`, `--kv-block-tokens`, threads, iterations, count, seed, output, `-S`/`--sweep`, `-X`/`--sweep-max-runs`,
  and help.
  Duplicate/unknown/incompatible options (named explicitly), unexpected inline values, missing values, non-complete
  decimal tokens, zero size/count values, invalid KV widths, invalid query/KV sharing, checked geometry overflow,
  impossible one-step payloads, and explicit scenario work above the strictest limit fail before output-session
  creation. Output consumes exactly one opaque raw token (space-separated or inline `--output=...`).
  Help still completes the whitelist pass but returns before required/default/preparation work. All JSON integer
  configuration fields (`weight_size_mb`, `layer_count`, `query_head_count`, `kv_head_count`, `head_dimension`,
  `visible_context_tokens`, `prompt_tokens`, `attention_query_tile_tokens`, `kv_block_tokens`, `batch_size`,
  `requested_workers`, `available_workers`, `iterations`, `loop_count`, and `sweep_max_runs` when sweeping) must fit
  within IEEE-754 exact safe integers (`<= 2^53 - 1`), and `loop_count <= (2^53 - 1) / 3`. LLM supports repeatable
  Cartesian sweeps over `context-tokens` (decode), `prompt-tokens` and `attention-query-tile-tokens` (prefill), and
  `kv-block-tokens` (paged KV), rejecting duplicate sweep keys and direct-flag/sweep conflicts (a swept required
  parameter satisfies that requirement while forbidding its matching direct flag), requiring non-empty `--output`,
  enforcing `1 <= run_count <= sweep_max_runs` (default `256`), and preflight-validating every generated run
  configuration before execution. LLM rejects all cache, latency, TLB, pattern, GPU, and non-cacheable modifiers.
  Defaults are two-byte KV elements, batch one, three loops, automatic per-scenario calibration, one generated nonzero
  seed, CPU, decode, and contiguous KV. Decode requires positive `--context-tokens` (or swept `context-tokens`) and
  rejects prefill geometry. Prefill requires positive `--prompt-tokens P` and `--attention-query-tile-tokens Q` (direct
  or swept), requires `Q <= P`, and rejects decode context. `--kv-layout` accepts `contiguous|paged`. Paged requires
  exactly one `--kv-block-tokens <G>` (or swept `kv-block-tokens`); contiguous rejects that option. `G` must be a
  positive power of two no greater than `UINT32_MAX`, and it may exceed the phase length. The cross-option rules are
  order-independent. All eight backend/phase/layout combinations are public. CPU and Metal each support decode and
  prefill with contiguous or paged KV. Metal skips CPU worker detection, rejects explicit threads, and never receives
  fallback execution.
  Explicit work must fit both the common task guardrails and Metal's 65,536-work-unit dispatch cap for all three
  scenarios when Metal is selected. Metal paged KV-bearing scenarios (`kv_only`, `mixed`) additionally enforce the
  268,435,456 semantic-lookup-per-task cap (`LLM_METAL_MAX_PAGED_SEMANTIC_LOOKUPS_PER_TASK`). Metal prefill additionally
  rejects an overflowing lane-local serial range-helper count or a count above 1,048,576 before checksum-oracle
  enumeration or GPU dispatch: contiguous scenarios count all applicable ranges, while paged `weights_only` counts its
  weight-layer ranges and paged KV-bearing owner kernels report zero serial range visits.

### 6.2 Validation behavior (`config_validator.cpp` and standalone validators)

Validation rejects missing primary modes, incompatible flag combinations, direct-vs-sweep conflicts, and invalid value states:

- Missing primary mode (modifier options without `--cpu-bandwidth`, `--cpu-latency`, `--patterns`, `--analyze-tlb`,
  `--analyze-core2core`, `--gpu-bandwidth`, or `--llm-memory`): fails with `--seed` mode requirement or
  `A primary mode option is required (...)`.
- Removed pre-split combined-mode and subset flags are rejected as unknown/unsupported options.
- `--patterns` rejects options outside its exact 11-option whitelist in `src/pattern_benchmark/patterns.cpp`
  (`Unsupported option '<token>': ...`) rather than `config_validator.cpp`, preserves `requested_workers`, and clamps
  `effective_workers` per pattern in `patterns_work_plan.cpp`.
- Direct option + `--sweep` for the same parameter is rejected uniformly across all 6 sweep-capable modes
  (`--cpu-bandwidth`, `--cpu-latency`, `--patterns`, `--analyze-tlb`, `--analyze-core2core`, `--llm-memory`) via
  `Cannot specify both --<opt> and --sweep <key>`.

Zero-disabling semantics (supported in `--cpu-bandwidth` and `--cpu-latency`):

- `--buffer-size 0` omits `main_memory`.
- `--cache-size 0` omits cache targets (`l1`/`l2`/`custom`).
- Both cannot be zero simultaneously (`--buffer-size 0 --cache-size 0` is rejected).

TLB-locality constraints:

- When the effective chain mode uses locality, `--latency-tlb-locality-kb` must be a non-zero multiple of system page
  size and span at least two latency-stride steps. Explicit `global-random` ignores the configured locality value.

Latency stride constraints:

- `--latency-stride-bytes` must be greater than zero and pointer-size aligned (`8` bytes on Apple Silicon).

Memory-limit model:

- For `--analyze-tlb`, `config_validator.cpp` applies `MEMORY_LIMIT_FACTOR` (80% of available memory,
  or `FALLBACK_TOTAL_LIMIT_MB = 2048 MiB`).
- For `--cpu-bandwidth`, `--cpu-latency`, and `--patterns`, `cpu_bandwidth_work_plan`, `cpu_latency_work_plan`, and
  `patterns_work_plan` compute strict `memory_budget` admission
  (`known_owned_peak_bytes = committed_data_bytes + transient_peak_bytes <= admitted_budget_bytes`,
  where `admitted_budget_bytes = floor(0.80 * available_memory_bytes)` or `2048 MiB` fallback) and reject oversized
  requests (`memory-budget-exceeded`) without silent truncation. (`patterns_work_plan` also charges
  `index_structure_bytes = 2 * selected_random * sizeof(size_t)` for random worker indices and the sorted validation copy.)

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

## 7. Access Derivation

### 7.1 Latency access counts

- `cpu_latency_runner` runs an excluded whole-chain pilot (`kMinPilotAccesses = 16,384` minimum), calibrates complete
  chain cycles toward `0.250s` (`[0.100s, 0.300s]`, max `2` corrections, minimum `16` complete chain cycles when
  permitted by `kCpuLatencyMaxAccesses = 4,000,000,000`), freezes `target_plans[]`, and reuses the frozen plan across
  outer loops.
- `--buffer-size 0` omits `main_memory` in `--cpu-bandwidth` and `--cpu-latency`.

## 8. Memory Allocation and Initialization

### 8.1 Allocation strategy

Allocation entrypoints:

- CPU bandwidth (`-B`): persistent `src` + `dst` mappings per active target allocated and recorded via
  `allocate_buffer_with_evidence(...)` (`src/core/memory/memory_manager.cpp`) for the command lifetime (`src` is
  `mprotect(PROT_READ)` via `protect_buffer_read_only_with_evidence`).
- CPU latency (`-L`): persistent pointer-chain buffer per active target (plus optional `compare_buffer` when
  `--compare-locality` is enabled) allocated via `allocate_buffer_with_evidence(...)` and protected read-only via
  `mprotect(PROT_READ)`.
- Patterns (`-P`): `prepare_patterns_resources` (`src/pattern_benchmark/patterns_executor.cpp`) allocates persistent
  `src` + `dst` mappings via `allocate_buffer_with_evidence(...)` (`src/core/memory/memory_manager.cpp`), protects
  `src` via `protect_buffer_read_only_with_evidence` (`mprotect(PROT_READ)`), and builds per-worker `random` index
  tables plus an ascending `sorted_validation_indices` copy (`<= 8 MiB` at `1,000,000 * 8 B`).

Shared allocation behavior:

- Uses `mmap` anonymous private mappings (macOS-specific behavior and limits apply) and records `AllocatedBufferEvidence`
  (`madvise`/`mprotect` errno and status) for audited CPU modes.
- Performs overflow-safe byte arithmetic before allocation.
- Enforces strict `memory_budget` checks from peak-concurrent requirements.

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

CPU LLM mode allocates suite-lifetime anonymous private `mmap` regions through its executor: active weights and either
logical contiguous K/V or full physical paged K/V pools. Paged execution also owns one shared row-major `uint32_t`
block table. Each mapping is rounded separately to native page granularity for committed-byte accounting, while hot
descriptors cover exact requested physical lengths. Logical bytes, layout padding, page rounding, table bytes, and
transient preparation storage remain separate accounting classes. The mappings are regular cacheable memory; the
standalone whitelist rejects `--non-cacheable`. Metal LLM mode instead allocates private/shared tracked `MTLBuffer`
segments as described in Section 13.7. In LLM sweeps, each nested run creates and releases its own backend and
resources within `run_llm_memory_collect` so consecutive points do not accumulate mappings or Metal buffers.

Allocation is atomic from the caller's perspective. The finalized pointer-free layout is revalidated and the exact
executor/runner auxiliary backing is re-admitted before the first mapping. Any partial candidate is released and the
command emits no new result payload on failure. The budget includes mappings, descriptor arrays, retained planner
templates, paged table plus its bounded bijection-validation bitset, static-reference and expected/actual checksum
arrays, thread/worker state, result/calibration records, statistics, and orchestration storage. The bitset and other
preparation transients are released before timed work. With non-empty file/stdout output or nested sweep collection
(`collect_json_for_sweep`), orchestration storage includes a conservative estimate for one live schema DOM plus
serialized transport text; disabled direct output contributes a valid zero estimate.
Workload sizes are rejected rather than silently reduced.

### 8.2 Best-effort non-cacheable mode

- `allocate_buffer_non_cacheable` still uses normal user-space mappings.
- Applies `madvise(MADV_RANDOM)` hints on macOS.
- This is best-effort cache discouragement only; not true uncached memory.

### 8.3 Initialization strategy

Initialization entrypoints:

- CPU bandwidth (`-B`): O(1) GF(2) deterministic pattern fill of `src` (`cpu-bw-buffer-pattern-v1`) + `mprotect(PROT_READ)`
  at setup, and `pre_copy_sentinel` fill of `dst` before each `copy` attempt.
- CPU latency (`-L`): `build_cpu_latency_chain` (`splitmix64-fisher-yates-rejection-v1`, canonical `indices[0] == 0`),
  Hamiltonian prevalidation, and `mprotect(PROT_READ)` at setup (plus per-round rebuild/prevalidate/protect for
  `--compare-locality`).
- Patterns (`-P`): O(1) slot-tagged GF(2) deterministic pattern fill of `src` (`patterns-buffer-pattern-v1`) +
  `mprotect(PROT_READ)` at setup, initial `pre_op_sentinel` fill of `dst` at setup, and fresh odd `pre_op_sentinel`
  fill of `dst` before each `copy` attempt and each `write` attempt with `has_untouched_dst_bytes == true`.

Initialization semantics:

- Bandwidth buffers: deterministic GF(2) source pattern + sentinel-initialized destination.
- Latency buffers: deterministically seeded, randomized circular Hamiltonian pointer-chasing chain.
- Allocation/initialization happen before timing starts and are excluded from measured benchmark durations.

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
excluded from elapsed scenario time. Before each paged decode task, only the mutable current-token K/V append slots are
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
- Exposes both an unseeded overload (using `std::random_device` or a deterministic unit-test hook) and an explicitly
  seeded overload for reproducible benchmark and analysis chains.

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

Warmup functions (`src/warmup`) run before measured TLB latency tests.

- Latency warmups (`warmup_latency` in `src/warmup/latency_warmup.cpp`) are single-threaded page-prefault passes that
  read and write one byte per native page; they do not execute the pointer chain.
- `-B`, `-L`, and `-P` do not call `src/warmup/`; instead they run excluded `calibration_shape_warmup`, optional
  `single_unit_confirmation_warmup`, and `frozen_measurement_warmup` directly through their own executors before loop `0`.
- Single-thread latency warmup attempts high QoS class (`QOS_CLASS_USER_INTERACTIVE`) best-effort.

## 11. CPU Bandwidth Execution (`src/cpu_bandwidth/*`)

Standalone coordinator: `run_cpu_bandwidth_mode` -> `run_cpu_bandwidth_collect` (`src/cpu_bandwidth/cpu_bandwidth_runner.cpp`).

- Canonical active target set (subset of `("main_memory", "l1", "l2", "custom")`) and operations `("read", "write", "copy")`.
  Main memory defaults to all detected logical cores and cache targets default to `1` worker; explicit `--threads`
  applies to every target and is capped per target to `min(desired, available_workers, max(1, target_bytes / 64))` with
  64-byte-aligned worker slices (`partition_cpu_bandwidth_64b`).
- Persistent `src` mappings are seeded once via `cpu-bw-buffer-pattern-v1` (`T[k] = splitmix64_mix(target_src_seed ^ k)`,
  `W[i] = T[i & 255] ^ i ^ rotl64(i, 19) ^ rotl64(i, 41)`) and protected `PROT_READ`.
- Omitted `--iterations` runs excluded calibration (`8 MiB` min pilot, `0.150s` target in `[0.100s, 0.250s]`, max `2`
  corrections) + `calibration_shape_warmup` / optional `single_unit_confirmation_warmup`; plans freeze atomically and
  run `frozen_measurement_warmup` before loop `0`.
- Loop `i` rotates `active_targets` by `i % num_targets` and `("read", "write", "copy")` by `i % 3` in a cyclic
  Latin-square schedule (`operation_order_balance_complete` when `loop_count % 3 == 0`; `target_order_balance_complete`
  when `loop_count % num_targets == 0`).
- Read uses `memory_read*_loop_asm` with non-cancelling per-pass accumulators (`cpu-bw-read-checksum-v1`) verified in
  O(1); write uses `memory_write_pattern_loop_asm` (`stnp`) / `memory_write_cache_pattern_loop_asm` (`stp`) with unique
  odd `write_pattern` and cold parallel full-buffer byte check; copy fills `dst` with unique odd `pre_copy_sentinel`
  outside timing, runs `memory_copy*_loop_asm`, and validates every byte against `src` via parallel `std::memcmp`.
- Direct files checkpoint atomically every `K=max(1,ceil(loop_count/8))` completed loops and at terminal; stdout skips
  intermediate serialization and writes one schema-1 document.

## 11.1 CPU Latency Execution (`src/cpu_latency/*`)

Standalone coordinator: `run_cpu_latency_mode` -> `run_cpu_latency_collect` (`src/cpu_latency/cpu_latency_runner.cpp`).

- Canonical active target set (subset of `("l1", "l2", "custom", "main_memory")`), always single-threaded.
- Each target chain uses `chain_seed = splitmix64_mix(base_seed ^ domain_id)`, `splitmix64-fisher-yates-rejection-v1`,
  cyclic rotation so `indices[0] == 0`, `chain_permutation_sha256` over little-endian `uint64_t` indices, Hamiltonian
  cycle prevalidation (`validate_cpu_latency_hamiltonian_cycle`), and `mprotect(PROT_READ)`.
- Omitted `--iterations` calibrates complete chain cycles toward `0.250s` (`[0.100s, 0.300s]`, max `2` corrections,
  minimum `16` cycles when within `4,000,000,000` accesses), then freezes plans and runs `frozen_measurement_warmup`.
- Each target measurement executes `memory_latency_chase_asm` for `complete_chain_cycles * node_count` steps and verifies
  `main-final-pointer` (`expected_node = 0`). When `--latency-samples > 0`, a separate continuous sample pass runs
  `effective_samples = min(requested, access_count)` windows, verifies each window's `cpu_raw` timebase and exact
  terminal node `indices[cum_steps % N]` (`sample-window-end-pointers`), and populates `pooled_sample_distribution`.
- Explicit opt-in `--compare-locality` (requires `--buffer-size > 0` and `--latency-stride-bytes <= 8192`) executes `3`
  paired rounds per loop before target measurements (`locality_16k` `random-box` first, then `locality_global`
  `global-random`, using frozen `main_exec_plan.complete_chain_cycles`), each rebuilding, prevalidating, and protecting
  its chain; `aggregates.locality_comparison.locality_latency_delta_ns = median(global) - median(16k)`.
- Direct files checkpoint atomically every `K=max(1,ceil(loop_count/8))` completed loops and at terminal. See
  [LATENCY_WHITEPAPER.md](LATENCY_WHITEPAPER.md) for the complete specification.

## 12. Pattern Benchmark Execution (`src/pattern_benchmark/*`)

Standalone coordinator: `run_patterns_mode` -> `run_patterns_collect` (`src/pattern_benchmark/patterns.cpp` and `src/pattern_benchmark/patterns_runner.cpp`).

- Canonical 7-pattern order: `("sequential_forward", "sequential_reverse", "strided_64", "strided_4096", "strided_16384", "strided_2mb", "random")`, each with `read`, `write`, and `copy` (`32B` access width, `2 * total_accesses * 32` copy payload). `strided_2mb` is published as `status: "skipped"` (`buffer-too-small-for-stride-transition`) when `buffer_bytes < 2 MiB + 32 B` and omitted from `calibration`, `measurements[]`, and `loop_records[]`.
- `requested_workers` is preserved while `effective_workers` uses `partition_patterns_64b` (64B-aligned spans) and strided worker reduction (`>= 2` accesses per worker); `random` clamps selected accesses to `[1000, 1_000_000]` (`min(total_slots, clamped)`) and generates per-worker offsets via `splitmix64-coprime-fisher-yates-v1` (`random_indices_sha256` + `<= 8 MiB` sorted validation copy).
- Persistent `src` mappings are seeded once via slot-tagged GF(2) `patterns-buffer-pattern-v1` and protected `PROT_READ`.
- Omitted `--iterations` runs excluded calibration (`8 MiB` min pilot, `0.150s` target in `[0.100s, 0.250s]`, max `2` corrections) + `calibration_shape_warmup` / optional `single_unit_confirmation_warmup`; plans freeze atomically in `frozen_plan_refs` and run `frozen_measurement_warmup` before loop `0`.
- Loop `i` rotates active patterns (`i % active_patterns.size()`) and operations (`i % 3`) in a 2-axis cyclic Latin-square schedule (`operation_order_balance_complete` when `loop_count % 3 == 0`; `pattern_order_balance_complete` when `loop_count % active_patterns.size() == 0`).
- Read passes verify non-cancelling per-worker and folded `patterns-read-checksum-v1` values in O(1) (sequential + 2D partial-cycle strided); write (`memory_write_pattern_loop_asm`, `memory_write_reverse_pattern_loop_asm`, `memory_write_strided_phased_pattern_loop_asm`, `memory_write_random_pattern_loop_asm`) and copy fill `dst` with unique odd `pre_op_sentinel` (when `has_untouched_dst_bytes` or `copy`) and undergo parallel cold 100% touched-byte and 100% untouched-sentinel validation (`execution.validation.checks[]`, 6 entries).
- Direct files checkpoint atomically every `K=max(1,ceil(loop_count/8))` completed loops and at terminal; stdout skips intermediate serialization and emits one schema-1 document. See [PATTERNS_WHITEPAPER.md](PATTERNS_WHITEPAPER.md) for the complete specification.

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
with the local validation record. The canonical source SHA-256 introduced in 0.61.2 and retained by the current
implementation is
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

### 13.7 LLM generic plan and active CPU/Metal execution

LLM mode is implemented under `src/llm_memory/` with a pure checked work planner, a pure paged-layout/permutation
module, an Objective-C-free synchronous `LlmBackend` boundary, a status-bearing backend-independent runner, a dedicated
CPU adapter, an Objective-C++ Metal capability/resource/task boundary, phase/layout-specific ARM64 kernels, an
ordered schema builder, and a Foundation-backed environment snapshot. Generic enums cover backend `cpu|metal`, phase
`decode|prefill`, and KV layout `contiguous|paged`.
All eight backend/phase/layout combinations are active. The CPU identities are
`llm-memory-v2-cpu-decode-contiguous`, `llm-memory-v2-cpu-decode-paged`,
`llm-memory-v2-cpu-prefill-contiguous`, and `llm-memory-v2-cpu-prefill-paged`. The Metal identities are
`llm-memory-v2-metal-decode-contiguous`, `llm-memory-v2-metal-decode-paged`,
`llm-memory-v2-metal-prefill-contiguous`, and `llm-memory-v2-metal-prefill-paged`. Capability failure remains explicit;
no Metal request receives fallback CPU execution.

`LlmMemoryWorkPlan` keeps logical geometry, resources, accounting, seeds, and identities common while
`LlmBackendExecutionPlan` carries exactly one `LlmCpuExecutionPlan` or `LlmMetalExecutionPlan` tag. The CPU alternative
owns layout-specific worker reduction, worker-major pointer-free ranges, descriptor counts, and retained planner storage.
Checked accessors require the declared backend and variant tag to agree.

The Metal planner resolves canonical 256 MiB W/K/V and uint32 block-table segments, four bounded 256-slot
argument-buffer arrays plus one status/checksum slot, and checked resource lengths and transient memory. A separate pure
planner derives dispatch grids from runtime pipeline limits, while capability evidence records foundation pipeline
properties. The 1,025 resource IDs, exact segment lengths, MSL revision, and source hash are versioned execution
contracts; the plan does not treat `MTLArgumentEncoder.encodedLength` or resource-allocation rounding as a portable ABI
formula. Every contiguous final segment has its exact remainder, and paged K/V blocks remain whole within a segment.

Result evidence records the canonical embedded MSL revision and exact runtime source SHA-256. Decode workload labels use
`membenchmark.llm-metal.pipeline.decode-contiguous.*` or `membenchmark.llm-metal.pipeline.decode-paged.*`; contiguous
prefill uses `membenchmark.llm-metal.pipeline.prefill-contiguous.weights-only`,
`membenchmark.llm-metal.pipeline.prefill-contiguous.kv-only`, and
`membenchmark.llm-metal.pipeline.prefill-contiguous.mixed`, while paged prefill uses the corresponding
`membenchmark.llm-metal.pipeline.prefill-paged.*` labels. The JSON snapshot records each pipeline's runtime thread
execution width and maximum total threads per threadgroup. Contiguous prefill uses the 136-byte, 8-byte-aligned
`llm-metal-prefill-contiguous-parameters-v1` ABI, verified at runtime by
`llm_metal_probe_prefill_parameter_layout` under pipeline label
`membenchmark.llm-metal.pipeline.prefill-contiguous-layout-probe`. Paged prefill uses the distinct 184-byte,
8-byte-aligned `llm-metal-prefill-paged-parameters-v1` ABI, verified by
`llm_metal_probe_prefill_paged_parameter_layout` under
`membenchmark.llm-metal.pipeline.prefill-paged-layout-probe`. Its three scenario-specialized entrypoints use the
`llm_metal_prefill_paged_*` family, and excluded validation uses
`membenchmark.llm-metal.pipeline.validate-prefill-paged-writes-padding`. Paged decode likewise requires its selected
layout probe and padding-aware validator. Validation and probe pipelines are selected-profile requirements, not
workload-pipeline result identities.

The paged-prefill component identities are `llm-metal-executor-v1-prefill-paged`,
`llm-metal-prefill-paged-cyclic-block-owner-grid-stride-v1`, `metal-command-buffer-gpu-start-end-v1`,
`llm-paged-physical-buffer-pattern-v1`, `llm-metal-prefill-paged-full-prompt-affine32-v1`, and
`llm-metal-paged-prefill-dual-mod32-lookup-address-mix-v1` for executor, schedule, timer, buffer pattern, write pattern, and
checksum respectively. The selected profile's canonical embedded source revision and exact runtime SHA-256 complete
the provenance rather than being duplicated as a prose constant.

The Objective-C++ boundary requires a default Apple7-or-newer unified-memory device, Tier-2 argument buffers, and
`maxBufferLength` of at least one canonical segment. It runtime-compiles embedded MSL 2.3 common foundation entrypoints,
the selected profile's three scenario-specialized workload pipelines, and its K/V-write-validation pipeline. W/K/V and
block-table resources use private tracked storage; argument/status resources and bounded reusable staging use shared
tracked storage. Initialization and table upload are excluded from timing.

Every active Metal profile gives each warmup, calibration, and measurement task one reset command buffer, one timed
command buffer with one explicitly serial compute encoder and one workload dispatch, and one excluded post-validation
command buffer. The scenario pipeline loops over all `T` work units in-kernel and splits segment-boundary/vector tails
without changing logical traffic. The command buffer's `GPUStartTime` and `GPUEndTime` are authoritative after terminal
completion; host submit/wait is diagnostic and queue delay stays null across unmatched clock domains. Zero,
non-increasing, or non-finite GPU timestamps fail the task. The timed status uses the selected profile's versioned
dual-mod32 W/K/V lanes and a bounded independent CPU oracle. For a KV-bearing task, excluded
post-validation checks the phase-specific K/V write before the next task: decode checks its current-token record, while
prefill checks the full-prompt write contract through deterministic representative and boundary byte samples against
operation ordinal `T - 1`. Paged validation additionally checks terminal padding canaries. Checksum, K/V-write, or
padding mismatch invalidates the measurement without retry.

Contiguous prefill uses a lane-local vector-stripe schedule with no grid-wide barrier. Every lane executes
`T * ((weight_active ? L : 0) + (kv_active ? 2 * L * B * (P + C) : 0))` serial range-helper visits. Paged prefill
`weights_only` executes `T * L`; paged KV-bearing owner kernels report zero serial range visits. The grid evidence
publishes this checked count as `serial_range_visits_per_lane`; overflow and values above 1,048,576 fail with stable
planner reasons before the independent CPU oracle or timed command is started. Both Metal prefill methodologies publish
the cap, and scenario limits divide it by the exact per-operation count before explicit-work validation or automatic
calibration. Metal paged KV-bearing scenarios (`kv_only`, `mixed`) additionally enforce the 268,435,456 semantic-lookup
per-task cap (`LLM_METAL_MAX_PAGED_SEMANTIC_LOOKUPS_PER_TASK`), and Metal grid planning enforces the 65,536
owner-ordinals-per-threadgroup cap (`LLM_METAL_MAX_OWNER_ORDINALS_PER_THREADGROUP`) and 1,048,576
vector-iterations-per-lane-per-visit cap (`LLM_METAL_MAX_VECTOR_ITERATIONS_PER_LANE_PER_VISIT`).

For paged decode, each logical layer/batch/block owner belongs to exactly one threadgroup under a cyclic grid-stride
schedule. At each semantic visit one named lane loads the table through `device const volatile uint*`, publishes the
physical ID in threadgroup memory, and executes `threadgroup_barrier(mem_flags::mem_threadgroup)` before any lane
constructs a segment/block address. Each block owner performs one K-scan and one V-scan lookup, while the terminal block owner also
performs one paired append lookup, for exactly `L * B * (2 * N + 1)` uint32 loads per KV-active work unit. The
dual-mod32 mix binds logical table index, physical ID, append/K-read/V-read visit kind, and work-unit ordinal so an
equal-multiplicity wrong permutation changes the expected checksum. Whole-block K/V segmentation leaves only terminal
logical padding; excluded post-validation verifies its
canaries. Lookup bytes and padding are separately reported and never enter effective-model-payload GB/s.

Resource preparation is candidate-atomic. A checked pre-allocation admission uses requested lengths, runtime argument-
encoder metadata, host auxiliaries, and transient staging; paged host mappings are rounded with the queried native page
size, and plan resolution rejects a submitted host granularity that differs from that runtime value. A second admission
uses each owned resource's reported allocated size when available and otherwise its length. Layout padding and allocation
rounding remain separate evidence and are not double-counted. Any nil resource, Objective-C exception, failed command,
or post-allocation budget rejection releases the entire candidate, and explicit release is idempotent.

Paged-table generation and the staged upload/readback loop poll preparation interruption no less often than every
1,048,576 entries; upload commands are therefore capped at 4 MiB of uint32 entries. Interruption retains the stable
`preparation-interrupted` reason and publishes no resources. The runtime ABI probe reads the first and last active
Tier-2 argument-buffer slot in every segmented pool, plus the shared status slot, so the 256 MiB boundary exercises a
nonzero indirect slot. The pure resource planner accepts explicitly supplied `additional_owned_bytes`; the active
backend auxiliary estimate includes its checksum and orchestration backing in command-wide admission.

The prefill planner resolves prompt length `P` and query-tile length `Q`. It computes `C = ceil(P/Q)`, prefix visits
`S = Q*triangular(P/Q) + (P%Q != 0 ? P : 0)`, causal pairs, logical
attention context counts, one weight pass, full-prompt K/V writes, tiled prefix reads, and checked per-scenario payloads.
The theoretical causal-pair/attention/FMA context is nullable on arithmetic overflow; actual byte, prefix, lookup, and
allocation overflow still rejects the workload. Query heads affect theoretical context, while query tile Q affects
executed prefix reads. No attention or FMA computation is performed.
For paged geometry it computes prefix block visits and the exact `N + 2*M` semantic lookup count with a logarithmic
floor-sum. Its bounded semantic oracle freezes owner-local ascending-token writes with K then V for each token,
followed for every tile by the complete owned K prefix and then the complete owned V prefix. Its affine64 checksum
oracle covers every operation ordinal and every tile-read visit. Excluded post-validation checks deterministic
owner first/middle/last canonical-word samples, including bytes clipped to owner boundaries, against `T - 1`; it does
not reread every prompt record.

The contiguous Metal specialization preserves that exact logical contract without CPU ownership partitions or table
lookups. Within each operation, the weights path performs one weight pass. In KV-bearing scenarios each lane populates
its disjoint slices across every K/V prompt record before reading those slices; each tile then scans the complete K
prefix before the complete V prefix. No grid-wide barrier is required because a lane reads only the bytes it wrote. All
`T` operations remain inside one workload dispatch. The prefill dual-mod32 oracle is commutative, so a separate audit
binds the MSL source hash and entrypoints to the write-all → tile-K → tile-V loop nesting frozen by the shared
semantic trace.

The KV-active paged Metal specialization has `L*B*N` logical block-owner ordinals and maps ordinal `o` to threadgroup
`o % threadgroup_count`. Each threadgroup advances cyclically by `threadgroup_count`, so each owner executes exactly
once without duplicating its semantic table lookups. One named lane performs each volatile uint32 table load, publishes
the physical block ID through threadgroup memory, and barriers before any lane forms an address. Across a layer/batch
sequence, each operation performs exactly `N` paired-write lookups plus `M` K-scan and `M` V-scan lookups, for
`L*B*(N+2*M)` total. The final partial block and partial tile visits use only their valid bytes. Grid evidence records
the exact cyclically assigned per-threadgroup `actual-threadgroup-cost` accounted-byte vector plus its minimum, maximum,
and imbalance; this is auditable cyclic ownership, not weighted balancing. Excluded validation binds the full-prompt write to final ordinal
`T - 1` and checks terminal padding canaries. A source-hash-bound audit freezes the write-all → tile-K → tile-V
loop order and one-lookup-per-semantic-visit contract. Every Metal `weights_only` task instead uses the weight-vector
grid-stride schedule and publishes the same exact cost unit with zero metadata lookups. Contiguous KV-bearing grids do
not publish threadgroup-cost evidence.

The pure CPU ownership API partitions contiguous tokens or whole logical blocks by exact scenario-accounted prefix cost.
Valid results publish a versioned canonical identity binding the prefill geometry, scenario, worker rotation, optional
weight shards, assignment boundaries, per-worker costs, and min/max/imbalance evidence. Any validation, overflow, or
allocation failure publishes no partial assignments, worker vectors, totals, or identity.

Contiguous prefill materializes scenario-specific cost-balanced token ownership, a dedicated descriptor ABI (48-byte
layer and 80-byte sequence records), full-size K/V resources, and the `llm_prefill_memory_asm` executor. Paged prefill
materializes block-exclusive weighted ownership, full physical K/V pools, a read-only uint32 block table and seeded
permutation, a distinct descriptor ABI with 48-byte layer and 112-byte block-assignment records, and the
`llm_prefill_memory_paged_asm` executor. (Decode similarly uses 48-byte layer records with 80-byte contiguous sequence
records or 96-byte paged block-assignment records.) Both prefill profiles retain exact per-scenario worker-cost
minimum, maximum, imbalance, scope, and execution identities.

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
reread the full pool. Equal-multiplicity wrong-table swaps must mismatch. Timed checksum mismatch, phase-specific
K/V-write mismatch, or padding-canary mismatch makes the measurement invalid without retry.

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
their exact valid byte count rather than rounding the visit to a full terminal block. On CPU, each semantic lookup
executes an `ldr w` before address formation. On Metal, one named lane performs the equivalent volatile uint32 load,
publishes the physical ID, and synchronizes the threadgroup before address formation. Checksum evidence non-separably
binds logical index, physical ID, visit kind, and operation ordinal. Total KV-active metadata is
`L*B*(N+2*M)` lookups and four bytes per lookup; weights-only uses zero.

The common runner accepts a task only when status/reason, backend, phase, layout, methodology/component identity,
work-unit and KV-write kinds, scenario, frozen plan, authoritative finite positive timing, planned/completed work,
model-payload/metadata/accounted counters, and validation all match. Comparative acceptance additionally requires the
same `G`, paged geometry, physical/logical/padding/table resource identity, permutation identity/hash, schedule, checksum,
and timer identities. Available-memory and derived admitted-budget samples remain separately validated runtime evidence,
not resource/execution/model/frozen-plan identity inputs. CPU worker/QoS/timer/checksum-vector invariants remain
adapter-owned.

Omitted iterations calibrate the three scenarios independently outside measurement. The exact initial pilot shape is
warmed once; correction candidates do not receive general warmups, while the first irreducible one-work-unit candidate
receives one confirmation warmup when that shape has not already been warmed. Explicit iterations are exact. All three
plans freeze atomically before canonical same-shape frozen warmups, and loop order then cyclically rotates weights-only,
KV-only, and mixed. No measured task begins until every frozen warmup succeeds. Only measured results with accepted timing, checksum,
and required cold-check evidence enter aggregates. Stop is checked between atomic backend tasks. File progress snapshots occur after every Kth complete
loop, where `K=max(1,ceil(count/8))`; resources are released before the command-terminal snapshot. Stdout emits one
final schema-2 object. Scenario position balance is separate from run acceptance and does not balance directed
predecessor pairs across loop or repeated-block boundaries. Frozen warmups run once before the measured loops. See
[LLM_MEMORY_PROFILE_WHITEPAPER.md](LLM_MEMORY_PROFILE_WHITEPAPER.md) for the complete formulas, ABI, schema, and
interpretation contract.

## 14. Assembly Kernel Layer

CPU bandwidth, cache, latency, pattern, and core-to-core assembly entrypoints are declared in `src/asm/asm_functions.h` and
used by benchmark/warmup code, while the four LLM assembly entrypoints (`llm_decode_memory_asm`,
`llm_decode_memory_paged_asm`, `llm_prefill_memory_asm`, and `llm_prefill_memory_paged_asm`) are declared with
`extern "C"` in `src/llm_memory/llm_executor.h`:

- Main-memory kernels: `memory_read_loop_asm` (4 NEON XOR lanes folded to `x0` + `<32B` byte tail), `memory_write_pattern_loop_asm` (`stnp` 64-bit arbitrary pattern fill + little-endian scalar tail for `--cpu-bandwidth` and `--patterns`), and `memory_copy_loop_asm`.
- Cache kernels: `memory_read_cache_loop_asm`, `memory_write_cache_pattern_loop_asm` (`stp` 64-bit arbitrary pattern fill + little-endian scalar tail for `--cpu-bandwidth`), and `memory_copy_cache_loop_asm` (`stnp` `>=32B` + `str` `<32B`).
- Latency kernel: pointer-chase loop.
- Pattern kernels: sequential forward (`memory_read_loop_asm`, `memory_write_pattern_loop_asm`, `memory_copy_loop_asm`), reverse (`memory_read_reverse_loop_asm`, `memory_write_reverse_pattern_loop_asm`, `memory_copy_reverse_loop_asm`), parameterized phased strided entrypoints (`memory_read_strided_phased_loop_asm`, `memory_write_strided_phased_pattern_loop_asm`, `memory_copy_strided_phased_loop_asm`) accepting stride, pass count, and initial 32-byte phase, and random (`memory_read_random_loop_asm`, `memory_write_random_pattern_loop_asm`, `memory_copy_random_loop_asm`). The 64 B, 4096 B, 16384 B, and 2 MiB patterns use the same phased strided API.
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

CPU LLM scenario timing uses `HighResTimer` from a synchronized all-workers-ready gate to last-worker completion. Metal
LLM scenario timing uses the completed workload command buffer's `GPUStartTime`/`GPUEndTime`; host submit/wait values
are diagnostic. The selected phase's model reads/writes, applicable paged table loads and address formation, and timed
checksum accumulation are inside the backend-authoritative interval. Mapping, initialization/pre-touch,
permutation/table generation and validation, table protection, descriptor or parameter validation/materialization,
task-local paged-decode append-slot restoration, same-shape warmup, calibration, expected-oracle work,
checksum/K/V-write/canary post-validation, aggregation, console, JSON, and checkpoints are outside. CPU thread
creation, QoS and joins, plus Metal reset and post-validation command buffers, are also outside. There is no cache flush
between tasks; the timing semantics are warm, cacheable, and cache-inclusive.

## 16. Statistics and Aggregation

`DescriptiveStatistics` (`src/utils/descriptive_statistics.*`) and mode-specific runners collect accepted measurement vectors across outer loops.

- Single loop: direct value reporting (`repeatability_assessment_available: false`).
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
- **Stddev**: Sample standard deviation (`n - 1` denominator; `0` when `n == 1`).
- **CV**: Sample standard deviation divided by the absolute mean, as a percentage when the mean is valid.
- **MAD**: Median absolute deviation from the median.
- **Min**: Minimum observed value.
- **Max**: Maximum observed value.

CPU bandwidth (`-B`) flags CV above `5.0%` (`kCpuBandwidthCvWarningPct`), CPU latency (`-L`) flags CV above
`7.5%` (`kCpuLatencyCvWarningPct`), and patterns (`-P`) flags CV above `5.0%` for streaming (`sequential_forward`,
`sequential_reverse`, `strided_64`) and `10.0%` for sparse (`strided_4096`, `strided_16384`, `strided_2mb`, `random`).
Values are retained without outlier filtering, and the warning does not invalidate `run_accepted`.

GPU aggregates use the same descriptive-statistics implementation, a 5% CV threshold, and only validated measured
values. Multiple GPU loop values use median P50; fewer than three cannot receive a repeatability classification stronger
than `insufficient-samples`.

LLM maintains separate measured-only distributions for synthetic work-unit latency, synthetic memory work units per
second, and effective model-payload GB/s for weights-only, KV-only, and mixed. Multiple values use median P50; fewer than
three are `insufficient-samples`, and effective-model-payload-GB/s CV above 5% selects `noisy` and emits a high-CV
warning without filtering or retry. Comparative validity also requires complete three-position cyclic order balance.
Mixed weight/KV fractions are byte fractions, not independently timed component bandwidths.

## 17. Console Output Contract

Console rendering is centralized in `src/output/console` (with mode-specific renderers in `src/cpu_bandwidth/`,
`src/cpu_latency/`, `src/pattern_benchmark/`, and `src/llm_memory/`) and message helpers in `src/output/console/messages`.

Contract highlights:

- Every successfully started direct mode and parameter sweep emits one shared version, copyright, and GPL banner
  before mode-specific configuration or status output. Nested sweep runs do not repeat it. Help output retains the
  separate usage preamble; parse and preflight failures emit concise single-line `Error: ...` diagnostics on stderr
  without dumping full usage text or the runtime banner.
- Result-producing CPU benchmark paths print applicable configuration and cache/target information before execution.
- `-B`/`--cpu-bandwidth` prints `CPU Bandwidth Mode (schema v1)`, active targets/workers/seed, and a summary section
  with per-target/operation `P50=<v> GB/s (min=..., max=..., CV=..., MAD=..., n=...)` plus deduplicated warning tokens on stderr.
- `-L`/`--cpu-latency` prints `CPU Latency Mode (schema v1)`, page size/chain/seed/`compare_locality` settings, active
  targets/node counts, and a summary section with per-target `P50=<v> ns (min=..., max=..., CV=..., MAD=..., n=...)`,
  optional `| PooledSamples P50=... (P90=..., P99=..., n=...)`, and optional `Locality Comparison: 16KB P50=..., Global P50=..., Delta=...`.
- `-P`/`--patterns` prints `Patterns Mode (schema v1)`, buffer/work-policy/seed/non-cacheable metadata, per-pattern active or
  `skipped (<reason>)` lines, and `--- Patterns Benchmark Results (<status>, <measured>/<planned> measured) ---` with
  per-pattern/operation `P50=<v> GB/s (...% vs forward) (min=..., max=..., CV=..., MAD=..., n=...)` (or `skipped` /
  `unavailable`) plus deduplicated warning tokens on stderr.
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
- Live progress uses the shared spinner on `stderr` only when it is a TTY; redirected TLB output contains no carriage-return control sequences.
- For every result-producing direct command or supported sweep using `--output -`, the output session is installed before
  runtime console rendering and worker-thread creation. CPU bandwidth, CPU latency, patterns, TLB, core-to-core, GPU, and
  LLM each install `JsonOutputSession` before banner/QoS/signal preparation and workers. Post-parse human output therefore
  goes to stderr, while final JSON bypasses the redirected stream through the retained original stdout buffer.
  Parse/preflight and other pre-result failures leave stdout empty; human help remains stdout.

## 18. JSON Output Contract

Every result-producing direct mode and supported sweep can send its existing payload either to an
atomic file target or, for exact raw `-`, to one final stdout document. This transport does not wrap or change the
measurement schema. [API.md](API.md) is the process-integration contract and support matrix.

- CPU Bandwidth schema 1 (`src/cpu_bandwidth/cpu_bandwidth_json.cpp`): 26 top-level keys (`schema_version: 1`,
  `mode: "cpu_bandwidth"`, `methodology_version: "cpu-bandwidth-v1-arm64-calibrated-seeded-balanced"`, `software`,
  `build_manifest`, `configuration` with raw `output_file` and `argv`, `resolved_plan`, `memory_budget`, `calibration`,
  `measurements`, `aggregates`, `status`, `reason_code`, `results_complete`, `run_accepted`, `interpretation`,
  `diagnostic`, `interruption_requested`, `operation_order_balance_complete`, `target_order_balance_complete`, `seeds`,
  `counters`, `checkpoint_lifecycle`, `loop_records`, `environment`, `quality_warnings`).
- CPU Latency schema 1 (`src/cpu_latency/cpu_latency_json.cpp`): 26 top-level keys (`schema_version: 1`,
  `mode: "cpu_latency"`, `methodology_version: "cpu-latency-v1-arm64-calibrated-seeded-balanced"`, `software`,
  `build_manifest`, `configuration` with raw `output_file` and `argv`, `resolved_plan`, `memory_budget`, `calibration`,
  `measurements`, `locality_comparison`, `aggregates`, `status`, `reason_code`, `results_complete`, `run_accepted`,
  `interpretation`, `diagnostic`, `interruption_requested`, `target_order_balance_complete`, `seeds`, `counters`,
  `checkpoint_lifecycle`, `loop_records`, `environment`, `quality_warnings`).
- Patterns schema 1 (`src/pattern_benchmark/patterns_json.cpp`): 26 top-level keys (`schema_version: 1`,
  `mode: "patterns"`, `methodology_version: "patterns-v1-arm64-phase-calibrated-seeded-balanced"`, `software`,
  `build_manifest`, `configuration` with raw `output_file` and `argv`, `resolved_plan`, `memory_budget`, `calibration`,
  `measurements`, `aggregates`, `status`, `reason_code`, `results_complete`, `run_accepted`, `interpretation`,
  `diagnostic`, `interruption_requested`, `operation_order_balance_complete`, `pattern_order_balance_complete`, `seeds`,
  `counters`, `checkpoint_lifecycle`, `loop_records`, `environment`, `quality_warnings`).
- TLB analysis schema 4: `configuration`, `execution_time_sec`, `tlb_analysis`, `timestamp`, `version`; conclusions
  require nested `status: "complete"` and `conclusions_valid: true`.
- Core-to-core schema 2: calibrated methodology configuration, `core_to_core_latency` command completion metadata, scenario work plans, nullable aggregate values, per-loop order/status/duration/hint/sample-boundary records, and affinity-comparison interpretability metadata.
- GPU schema 1: top-level mode/schema/methodology/status, exact counters and completeness, effective/copy/DRAM semantics,
  config/argv, environment, backend device/compile/allocation, memory budget, frozen plans, excluded calibration,
  status-bearing measurements/loop records, aggregates, and warnings.
- LLM schema 2: top-level backend/phase/layout/methodology/status, `configuration`, generic `resolved_plan`, tagged
  `backend_evidence`, memory-budget/calibration evidence, status-bearing generic work-unit measurements, measured-only
  aggregates, and interpretation, plus additive exact traffic, checksum, loop/checkpoint, environment, and warning
  evidence.
- Sweep envelope schema 1: CPU bandwidth, CPU latency, patterns, TLB, core-to-core, and LLM-memory producers
  record `configuration.mode = "sweep"`, `configuration.sweep_schema_version = 1`, `configuration.base_mode`,
  `configuration.run_count`, `configuration.sweep_max_runs`, `configuration.sweep_parameters`, top-level `status`,
  `status_reason`, `planned_runs`, `attempted_runs`, `completed_runs`, and `conclusions_valid`, plus per-entry
  `runs[].index`, `parameters`, `status`, `status_reason`, and `result`. Every attempted run is retained and
  `attempted_runs == runs.size()`; a file target checkpoints each attempt and also checkpoints a terminal envelope when
  the run plan is empty or a stop is observed before a run, without incrementing `attempted_runs`. Stdout preserves that
  logical cadence but defers serialization until the terminal envelope. `completed_runs` requires nested CPU bandwidth
  (`SweepNestedMode::CpuBandwidth`) to have `mode: "cpu_bandwidth"`, `schema_version: 1`, `status: "complete"`,
  `results_complete: true`, `run_accepted: true`, `configuration.output_file: ""`, and every planned measurement
  `status: "measured"`; nested CPU latency (`SweepNestedMode::CpuLatency`) has the identical predicate with
  `mode: "cpu_latency"`; nested patterns (`SweepNestedMode::Patterns`) has the identical predicate with
  `mode: "patterns"`, `schema_version: 1`, `status: "complete"`, `results_complete: true`, `run_accepted: true`,
  `configuration.output_file: ""`, and non-empty `measurements[]` all `status: "measured"`. TLB requires
  `tlb_analysis.status: "complete"` and `tlb_analysis.conclusions_valid: true`; core-to-core requires
  `core_to_core_latency.status: "complete"` and `measurements_complete: true`; and LLM-memory requires nested
  `mode: "llm_memory"`, integer `schema_version: 2`, `status: "complete"`, `results_complete: true`, `run_accepted: true`,
  and a non-empty `measurements` array in which every entry has `status: "measured"`. Partial, interrupted, and failed
  attempts remain as evidence without incrementing the completed count. TLB's native `status: "error"`, CPU
  bandwidth/latency/patterns' native `status: "unsupported"`, and LLM's native `status: "unsupported"` map to a failed attempt.
  For CPU bandwidth, CPU latency, patterns, and LLM sweeps, a pre-initialization `preparation-interrupted` stop records an
  interrupted attempt with `result: null` and `EXIT_SUCCESS`, while a pre-initialization setup failure records a failed
  attempt with `result: null` and its stable failure reason. The authoritative schema-1 sweep acceptance predicate is
  exactly `status == "complete" && conclusions_valid == true`. Producers maintain `completed_runs == planned_runs` for an
  envelope satisfying that predicate; consumers may check the equality separately as a defensive consistency check,
  but it is not an additional completeness condition.

Command completeness for CPU bandwidth, CPU latency, and patterns schema 1 requires `schema_version == 1`, the expected `mode`
(`"cpu_bandwidth"`, `"cpu_latency"`, or `"patterns"`) and `methodology_version`, matching `component_identities`, `status == "complete"`,
`results_complete == true`, `run_accepted == true`, string `configuration.output_file`, and every scheduled measurement
`status == "measured"`. Bundled `script-examples/` scripts accept compatible producer releases when those schema-1
contracts and consumed field shapes match, retaining `software.version` as provenance only without translating legacy
standard schema-3/2, legacy pattern schema-3, or unversioned historical JSON layouts.
TLB requires `tlb_analysis.status == "complete" && tlb_analysis.conclusions_valid == true`; core-to-core requires
`core_to_core_latency.status: "complete" && core_to_core_latency.measurements_complete == true`. Metric consumers must
additionally require the selected aggregate/measurement's `status == "measured"` and non-null `value`; in patterns,
`strided_2mb` may be `status == "skipped"` (`buffer-too-small-for-stride-transition`) in `resolved_plan.patterns.strided_2mb`
and `aggregates.patterns.strided_2mb.<op>` (and omitted from `measurements[]`) while the command remains complete. Affinity-scenario interpretation
also requires `affinity_hint_comparison_interpretable == true`. GPU requires `status == "complete" &&
results_complete == true && conclusions_valid == true`; position-balanced comparisons additionally require
`operation_order_balance_complete == true`.
LLM requires `mode == "llm_memory" && schema_version == 2`, top-level backend/phase/layout equal to the request,
methodology equal to `llm-memory-v2-<backend>-<phase>-<layout>`, `status == "complete"`, `results_complete == true`,
`run_accepted == true`, the run-policy identity `bounded-loop-snapshots-v1` (or LLM policy token), and every planned
measurement to be `measured`. A selected metric additionally requires accepted timing, checksum, and required
post-validation evidence. Run acceptance is independent of position balance, sample count, CV, duration, and environment;
comparison policy must assess those separately and match frozen work, seeds, component identities, and output cadence.

### 18.1 CPU Bandwidth (`-B`) Schema 1 Highlights

- `schema_version: 1`, `mode: "cpu_bandwidth"`, `methodology_version: "cpu-bandwidth-v1-arm64-calibrated-seeded-balanced"`.
- `resolved_plan.component_identities`: `cpu-bandwidth-components-v1`, `cpu-bandwidth-work-plan-v1`,
  `cpu-bw-buffer-pattern-v1`, `cpu-bw-read-checksum-v1`, `bounded-loop-snapshots-v1`.
- `resolved_plan.targets.<target>` (`main_memory`, `l1`, `l2`, `custom`): `bytes`, `effective_workers`, `worker_spans[]`.
- `resolved_plan.operation_plans[]`: `plan_ref`, `target`, `operation`, `passes`, `exact_payload_bytes`, `op_seed_uint64_decimal`, `expected_checksum`.
- `measurements[]`: `plan_ref`, `bandwidth_gb_s`, `completed_passes`, `completed_exact_payload_bytes`, `execution.timing.cpu_raw`, `execution.write_pattern_uint64_decimal`, `execution.checksum`, `execution.validation.checks[]` (`post-validation-structure`, `read-checksum`, `write-final-full-buffer`, `copy-final-full-buffer`, `src-readonly-protection`).
- `aggregates.targets.<target>.<op>`: `accepted_measurement_ids`, `value` (median GB/s), `statistics` (11 keys), `quality`.

### 18.2 CPU Latency (`-L`) Schema 1 Highlights

- `schema_version: 1`, `mode: "cpu_latency"`, `methodology_version: "cpu-latency-v1-arm64-calibrated-seeded-balanced"`.
- `resolved_plan.component_identities`: `cpu-latency-components-v1`, `cpu-latency-work-plan-v1`,
  `splitmix64-fisher-yates-rejection-v1`, `cpu-latency-chain-v1`, `bounded-loop-snapshots-v1`.
- `resolved_plan.chain_identities.<target>`: `node_count`, `stride_bytes`, `chain_mode`, `tlb_locality_kb`, `chain_seed_uint64_decimal`, `chain_permutation_sha256`.
- `measurements[]`: `plan_ref`, `latency_ns`, `completed_accesses`, `completed_chain_cycles`, `final_pointer_verification` (`expected_node: 0`, `actual_node`, `valid`), `sample_windows` (`timebase`, `sample_count`, `sample_delta_ticks`, `samples_ns`, `validation.end_node_indices`), `execution.timing.cpu_raw`, `execution.validation.checks[]`.
- `locality_comparison`: `null` unless `--compare-locality`, else `rounds[]` (`locality_16k`, `locality_global` per round).
- `aggregates.targets.<target>`: `accepted_measurement_ids`, `value` (median ns), `statistics`, `quality`, `pooled_sample_distribution`; `aggregates.locality_comparison`: `locality_16k_latency_ns`, `global_random_latency_ns`, `locality_latency_delta_ns` (`median(global) - median(16k)`).

### 18.2.1 Patterns (`-P`) Schema 1 Highlights

- `schema_version: 1`, `mode: "patterns"`, `methodology_version: "patterns-v1-arm64-phase-calibrated-seeded-balanced"`.
- `resolved_plan.component_identities`: `patterns-components-v1`, `patterns-work-plan-v1`,
  `patterns-buffer-pattern-v1`, `patterns-read-checksum-v1`, `splitmix64-coprime-fisher-yates-v1`, `bounded-loop-snapshots-v1`.
- `resolved_plan.patterns.<kind>` (all 7 patterns): `enabled`, `status` (`active`/`skipped`/`unavailable`), `reason_code`, `access_size_bytes: 32`, `stride_bytes`, `phase_period_passes`, `native_page_size_bytes`, `stride_equals_native_page_size`, `large_page_backing_verified: false`, `large_page_backing_status`, `effective_workers`, `worker_spans[]`, plus `random_workload` (`random_indices_sha256`) on `random`.
- `resolved_plan.operation_plans[]` & `frozen_plan_refs`: `plan_ref`, `pattern`, `operation`, `passes`, `total_accesses`, `exact_payload_bytes`, `accesses_per_pass`, `min_accesses_per_pass`, `max_accesses_per_pass`, `distinct_address_count`, `logical_working_set_bytes`, `completed_phase_cycles`, `has_untouched_dst_bytes`, `op_seed_uint64_decimal`, `expected_checksum`.
- `measurements[]`: `plan_ref`, `bandwidth_gb_s`, `completed_passes`, `completed_exact_payload_bytes`, `execution.timing.cpu_raw`, `execution.write_pattern_uint64_decimal`, `execution.pre_op_sentinel_uint64_decimal`, `execution.validation.checks[]` (`post-validation-structure`, `read-checksum`, `write-final-touched-bytes`, `copy-final-touched-bytes`, `untouched-sentinel-isolation`, `src-readonly-protection`), `checksum`.
- `aggregates.patterns.<kind>.<op>`: `status`, `reason_code`, `accepted_measurement_ids`, `value` (median GB/s), `relative_to_forward_pct`, `statistics`, `quality` (`5.0%` streaming / `10.0%` sparse CV).

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

### 18.6 LLM schema 2

- Schema 2 replaces the unpublished schema-1 contract without compatibility aliases or fallback reading.
- Required top-level fields are `schema_version`, `mode`, `backend`, `phase`, `kv_layout`, `methodology_version`,
  `software`, `build_manifest`, `configuration`, `resolved_plan`, `backend_evidence`, `memory_budget`, `calibration`,
  `measurements`, `aggregates`, `status`, `reason_code`, `results_complete`, `run_accepted`, `interpretation`,
  `diagnostic`, `interruption_requested`, `scenario_order_balance_complete`, `seeds`, `counters`,
  `checkpoint_lifecycle`, `loop_records`, `environment`, and `quality_warnings`.
- Methodology is exactly `llm-memory-v2-<backend>-<phase>-<layout>`. Active identities are
  `llm-memory-v2-cpu-decode-contiguous`, `llm-memory-v2-cpu-decode-paged`,
  `llm-memory-v2-cpu-prefill-contiguous`, `llm-memory-v2-cpu-prefill-paged`,
  `llm-memory-v2-metal-decode-contiguous`, `llm-memory-v2-metal-decode-paged`,
  `llm-memory-v2-metal-prefill-contiguous`, and `llm-memory-v2-metal-prefill-paged`.
- `configuration` retains exact `argv` and `resolved_sources`. `resolved_plan` owns phase-applicable `geometry`,
  layout-applicable `layout`, immutable logical/physical `resources`, and exact `component_identities`.
  Exactly one of `geometry.decode` and `.prefill` is populated. Prefill records integer P/Q and decimal-string
  C/prefix-visit counts. Theoretical causal-pair/logical-attention/FMA counts live in `model_context.prefill` as
  decimal strings with individual `*_reason_code: "valid"`, or null with `arithmetic-overflow`. That object is null
  for decode; decode-only crossover and weight/KV-read ratio fields are null for prefill. Admitted paged plans populate
  block geometry, with `layout.decode_append_offset_in_last_block` serialized as a decimal string for paged decode and
  `null` for paged prefill (while `resolved_plan.plan_identity` encodes `0` for prefill and `((A - 1) % G) * R` for
  decode); runtime permutation and combined layout-identity fields remain null when Metal terminates before table
  preparation. Complete paged results populate them. Contiguous results use
  null for paged-only values. Component identities include logical profile, KV layout, optional permutation, backend
  executor, resource ABI, schedule, timer, buffer, write, checksum, and nullable MSL identities under the canonical
  `llm-memory-components-v1` fixed-order serialization.
- CPU prefill `backend_evidence.cpu.prefill` records `cost_unit: "worker-cost"`, execution/scope identities,
  descriptors per scenario/worker, and scenario-specific decimal-string worker cost, minimum, maximum, and imbalance
  evidence. The object is null for decode; `backend_evidence.cpu.paged` is populated for either paged phase, so paged
  prefill has both objects.
- `backend_evidence` always has `cpu` and `metal` object-or-null branches; exactly the selected branch is populated.
  Metal records worker/QoS applicability false, lifecycle status/reasons, capability/compiler/source/pipeline evidence,
  argument-buffer layout probe, actual buffer options/length/allocated-size evidence, and allocation/admission totals.
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
- `resolved_plan.resources` records full-size logical/physical weight/K/V layout, explicit layout padding, and paged
  table bytes. Metal additionally records 256 MiB segmentation, exact segment lengths/tails, Tier-2 argument-buffer
  slots/encoded length/alignment, status/staging sizes, and resource-plan admission, while
  allocation/preparation/admission peaks live in `memory_budget`. Pointer/range values are not exposed.
- `resolved_plan.scenario_plans[]` owns each canonical scenario/work-policy/work-unit plan, its payload/accounting,
  exact identity, and sole expected checksum witness. `resolved_plan.frozen_plan_refs` selects the frozen plans.
  Measurements have explicit zero-based `measurement_id` and `plan_ref`; excluded calibration attempts also use
  `plan_ref`. Runtime records retain actual observations rather than duplicate plan identities and expected checksums.
- `measurements[]` retains status, order, completed work, working set, actual checksum, and backend execution evidence.
  Accepted records expose `synthetic_work_unit_latency_seconds`, `synthetic_memory_work_units_per_second`, and
  `effective_model_payload_gb_s`; invalid or unavailable metrics are null. Metal evidence retains pipeline/grid,
  GPU timestamps, host envelope, and command/encoder/dispatch counts.
- Measurement and calibration `execution.validation.checks[]` records named checks with independent `applicable`,
  `evaluated`, `valid`, and `reason_code` fields. Names are `post-validation-structure`, `kv-append-final`,
  `kv-prefill-final-samples`, `kv-append-unchanged`, and `kv-padding-canary`, selected by backend/phase/layout/scenario.
  CPU decode validates the final append state outside timing. Inapplicable checks have null evaluated/valid values;
  missing required observations cannot become successful checks. Prefill content validation remains sampled.
  Checksum agreement is not exhaustive proof of contents, addresses, or visit multiplicity; a checksum-valid attempt
  with a failed required cold check remains invalid and excluded from aggregates.
- Each scenario aggregate owns `accepted_measurement_ids` shared by all three metrics. Rates are derived per
  measurement before exact statistics; raw metric vectors are not duplicated in aggregates. Exact median/MAD/percentiles
  are prepared only at snapshots or terminal, using reusable workspaces. `observed_cv_classification` is
  `insufficient-samples` for n<3, `undefined` for undefined CV, `above-threshold` for payload CV strictly above 5%,
  otherwise `below-threshold`. This is observed sample quality, not an acceptance or reproducibility guarantee.
- `build_manifest` binds reported build inputs and a once-per-command executable SHA-256; missing evidence remains
  null with an explicit unavailable/partial status. It is provenance, not signed runtime attestation.
  CPU measurement and excluded-attempt `execution.timing.cpu_raw` retain original start/stop/delta ticks as decimal
  strings and timebase numerator/denominator as integers. Metal has null CPU raw evidence and retains GPU timestamps.
- For paged KV, `kv_block_tokens` is an integer input, while potentially large block/table/lookup/byte counts remain
  canonical decimal strings. `permutation_seed_uint64_decimal` is a decimal string and `permutation_sha256` is exactly
  64 lowercase hexadecimal characters. Decode KV-only and mixed report `L*B*(2*N+1)` lookups; prefill KV-only and mixed
  report `L*B*(N+2*M)`. Both use four metadata bytes per lookup per work unit, while weights-only reports applicable
  zero traffic. Metal paged-prefill grid evidence also publishes `owner_ordinals_per_threadgroup`, the exact
  `threadgroup_accounted_bytes` vector, and its minimum, maximum, and imbalance as canonical decimal strings. Every
  Metal weights-only grid publishes exact weight-vector-grid costs in the same fields. Contiguous KV-bearing grids use
  null cost unit/summary fields and an empty vector.
- `traffic_diagnostics.classification_version` is `llm-exact-weight-vs-kv-read-payload-v1`; tokens are
  `weight_payload_dominant`, `near_crossover`, and `kv_read_payload_dominant`. Near means exact equality, and
  `classification_is_payload_only` prevents a hardware-bottleneck interpretation.
- The complete-result predicate also requires backend/phase/layout to match the request, exact derived methodology,
  `status == "complete"`, `results_complete == true`, `run_accepted == true`, the exact run policy, and every planned
  measurement to be measured. Paged comparison/acceptance additionally matches `G`, `N`, tail, logical/physical/padding/table
  resources, permutation version/domain/seed/hash, lookup/accounting, schedule, timer, and checksum identities.
  A correct count-one result may be accepted while `scenario_order_balance_complete` is false. Position balance requires complete loops and equal
  nonzero first/middle/last counts for every scenario; it does not establish predecessor-pair balance.
- See [LLM_MEMORY_PROFILE_WHITEPAPER.md](LLM_MEMORY_PROFILE_WHITEPAPER.md) for the field groups and consumer boundary.

### 18.7 Command-output transport boundary

The cold-path `JsonOutputSession` classifies the raw output value before path handling. Exact `-` selects final stdout
JSON. An empty value disables JSON for a direct command and is missing/invalid for a sweep. Every other non-empty value
is a file target, including `./-` and flag-shaped names such as `-G`. General CPU modes, core-to-core, and LLM resolve
relative file targets against the current working directory (`JsonFilePathPolicy::ResolveAgainstCurrentDirectory`),
while GPU retains the raw path (`JsonFilePathPolicy::PreserveRaw`). The shared atomic file writer remains
sentinel-agnostic.

For a stdout target, the single-owner session retains the original `std::cout` buffer and routes ordinary command output
to `std::cerr`. Final JSON uses a separate stream backed by the retained buffer, so benchmark formatting state cannot
affect serialization. Stdout checkpoints are lazy successful no-ops and do not build payloads; file checkpoints still
use atomic replacement. Final stdout serialization emits UTF-8 JSON with two-space indentation and one trailing newline,
checks write and flush state, contains exceptions, and restores the prior stream buffer through the scope-bound
destructor. An observable terminal serialization, write, or flush failure returns `EXIT_FAILURE` without changing the
already-computed measurement state.

TLB command boundaries select this infrastructure after successful parse/help handling and before
direct or sweep validation. CPU bandwidth (`-B`), CPU latency (`-L`), and patterns (`-P`) select it after dedicated parse/help/config-build
and before banner, QoS, signal scope, sweep dispatch or direct plan admission, mappings, and workers. Core-to-core
selects it after its dedicated combined parse/preflight and help handling, but before sweep execution, the runtime banner,
and worker creation. GPU selects it after its dedicated parser/help path and before its banner, QoS, signal scope, and
backend factory. LLM selects it before banner, QoS, signal scope, sweep dispatch or direct work-plan admission,
mappings, and workers. Direct TLB and core-to-core files receive one atomic final write; all sweep files
(CPU bandwidth, CPU latency, patterns, TLB, core-to-core, and LLM) retain per-attempt atomic checkpoints (without
intra-run snapshots). GPU files retain terminal-measurement/failure checkpoints and their post-release replacement.
Direct CPU bandwidth, CPU latency, patterns, and LLM files snapshot every Kth fully completed loop,
where `K=max(1,ceil(planned_loops/8))`, at most eight progress writes plus one post-release command terminal. An abrupt
exit can lose up to `3K` (or `K * active_patterns.size() * 3` for `-P`) completed attempts, including while replacement is in progress; no file may exist before the
first successful snapshot. Graceful interruption retains the finished prefix when terminal persistence succeeds.
A late command exception after successful terminal persistence may produce one corrective failure snapshot. A failed
checkpoint is terminal and is never retried. Atomic rename does not promise power-loss durability.
`checkpoint_lifecycle` reports prior actual file-writer attempts/successes before current snapshot preparation;
`current_persistence_success` remains null because the snapshot cannot attest to its own future write. Stdout emits one
terminal DOM, and disabled output builds none. Stop checks remain at task boundaries even when progress is skipped.
The supported process contract, including help and pre-result-failure exceptions, is defined in [API.md](API.md).

### 18.8 Path behavior

- Relative file `--output` paths are resolved against the current working directory
  (`JsonFilePathPolicy::ResolveAgainstCurrentDirectory`) by CPU bandwidth, CPU latency, patterns, TLB, core-to-core, and
  LLM command boundaries; standalone GPU uses `JsonFilePathPolicy::PreserveRaw`.
- CPU bandwidth schema 1, CPU latency schema 1, patterns schema 1, GPU schema 1, and LLM schema 2 preserve the raw output spelling in
  `configuration.output_file` (with GPU serializing `null` when empty, and CPU bandwidth/CPU latency/patterns/LLM serializing
  `""` for disabled direct output or nested sweep runs); all five also retain exact captured `argv`. Exact `-` therefore
  remains `"-"` in a stdout payload, while `./-`, `-G`, and every other non-empty non-sentinel value are real file
  targets.

## 19. Error-Handling Model

This codebase uses boundary-aware mixed error handling:

- Program orchestration and most modules: return codes (`EXIT_SUCCESS`/`EXIT_FAILURE`).
- Argument parsing internals: shared descriptor-driven syntax parsing returns structured `CliParseResult` errors, and
  value-validation exceptions are converted to return codes at parser and command boundaries.
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
- LLM configuration/logical-preflight, checked JSON-output peak estimation, and auxiliary admission that fail before
  runner-result initialization have no schema payload. After runner-result initialization, Metal capability absence is
  terminal `unsupported`; backend compilation, pipeline, tagged-plan, resource preparation, and task failures are
  terminal `failed`/`invalid` schema evidence. Neither path falls back to another backend. The runner contains
  backend-task and checkpoint exceptions as stable status/reason evidence, finalizes untouched slots deterministically,
  gives real failures precedence over interruption, and keeps graceful task-boundary interruption separate from
  run acceptance. Backend release is attempted once before the command-terminal checkpoint; release failure
  becomes terminal failure evidence. A file progress-snapshot failure is terminal, releases resources, and is not retried.

Principle: no uncaught exceptions should escape to `main()` control flow.

## 20. Concurrency Model

- Bandwidth and pattern bandwidth paths are parallelized by thread-count configuration.
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
  per-layer global barrier, and neither the runner nor CPU adapter polls signal state inside the ARM64 hot loop. The
  active Metal LLM adapter uses one Metal command queue and serial compute encoders with synchronous per-task reset,
  timed workload, and post-validation command buffers, and rejects CPU `--threads`.

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
- Independent patterns schema-1 contract and mutation tests: `make test-patterns-verifier`
- Independent CPU bandwidth, CPU latency, and patterns schema-1 contract and mutation tests: `make test-cpu-verifiers`
- Independent LLM schema-2 artifact and schema-1 sweep envelope contract and mutation tests: `make test-llm-verifier`
- Integration-only: `make test-integration`
- Full test set: `make test-all`
- CLI help smoke checks: `./memory_benchmark -h`, `./memory_benchmark -B -h`, `./memory_benchmark -L -h`, `./memory_benchmark -P -h`, `./memory_benchmark --analyze-tlb --help`
- GPU help smoke check: `./memory_benchmark --gpu-bandwidth --help`
- LLM help smoke check: `./memory_benchmark --llm-memory --help`
- Deterministic CLI parser tests:
  `./test_runner '--gtest_filter=CliParserTest.*:ModeSelectorTest.*:ConfigTest.*:CoreToCoreCliTest.*:GpuBandwidthParserTest.*:LlmMemoryConfigTest.*:Messages*Test.*'`
- Deterministic LLM tests:
  `./test_runner '--gtest_filter=CliParserTest.*:ModeSelectorTest.*:LlmMemoryConfigTest.*:LlmMemoryWorkPlan*.*:LlmMemoryExecutorTest.*:LlmMemoryRunnerTest.*:LlmMemoryJsonTest.*:LlmMemoryOutputTest.*:SweepRunnerTest.*:Messages*Test.*'`
- Real LLM ARM64 and bounded executable transport contracts:
  `./test_runner '--gtest_filter=*LlmMemory*Integration*:*ExecutableCliIntegration*'`
- LLM Metal decode and prefill for contiguous and paged KV:
  `./test_runner '--gtest_filter=*LlmMetalBackend*:*LlmMemoryRunner*:*LlmMemoryJson*:*ExecutableCli*'`; real-device
  cases compile each selected profile and cover all scenarios, GPU timestamps, exact tails or partial paged blocks,
  paged-prefill `N+2*M` lookups, cyclic block-owner threadgroups and their exact accounted-byte evidence, full-prompt
  writes, Q=1/Q=P/remainder tiled-prefix scans, source-hash provenance, permutation-sensitive lookups,
  K/V-write and padding validation, multi-threadgroup work, and a bounded run crossing the 256 MiB segment boundary.
- Deterministic GPU-focused tests:
  `./test_runner --gtest_filter='CliParserTest.*:GpuBandwidthParserTest.*:GpuMemoryBudgetTest.*:GpuRunnerTest.*:GpuJsonTest.*:GpuWorkPlanTest.*:GpuTimedAccumulatorOracleTest.*:ModeSelectorTest.*:HashUtilsTest.*'`
- Real Metal contract: `./test_runner '--gtest_filter=GpuMetalBackendIntegrationTest.*'`; unsupported hardware may skip
  this integration suite but does not replace deterministic unsupported-path coverage. On a supported host, sandbox
  `metal-device-unavailable` skips require narrowly scoped unsandboxed probes and applicable real-device gates, as
  specified in `AGENTS.md`; sandbox skips do not establish hardware unavailability.

For narrow changes, prefer targeted `gtest` filters via `./test_runner --gtest_filter=...`.

The aggregate `make test-all` gate requires Python 3; after all GTest cases pass, it runs the focused script-example
entry test (`test-script-examples`), independent LLM verifier tests (`test-llm-verifier`), and independent CPU
bandwidth/latency/patterns verifier tests (`test-cpu-verifiers`, including `test-patterns-verifier`). `jq` is not required by this gate.

`python3 script-examples/verify_cpu_bandwidth_result.py result.json`,
`python3 script-examples/verify_cpu_latency_result.py result.json`, and
`python3 script-examples/verify_patterns_result.py result.json` independently verify schema-1 `cpu_bandwidth`,
`cpu_latency`, and `patterns` direct artifacts or schema-1 sweep envelopes (shared library `script-examples/cpu_verify_common.py`), with
`--binary PATH` and `--require-raw-timing` (plus `--max-chain-nodes N` on latency and `--max-random-indices N` on patterns).
`python3 script-examples/verify_llm_result.py result.json` independently verifies either a single schema-2 LLM artifact
across the eight supported profiles or a schema-1 LLM sweep envelope (`configuration.mode == "sweep"`,
`configuration.base_mode == "llm_memory"`, `configuration.sweep_schema_version == 1`) without invoking producer
helpers. For single artifacts it reconstructs geometry, plan/checksum identities, timing, named validation, accepted
populations, and exact statistics. For sweeps it additionally verifies Cartesian parameter progression (independent
of JSON key order in `sweep_parameters`), base-configuration and build-manifest invariance across nested runs, empty
nested `output_file`, per-run status/result consistency, envelope completion counters, and every non-null nested
payload under one shared bounded work budget (hashing `--binary` once across the sweep), adding
`artifact_kind`, `planned_runs`, `completed_runs`, `verified_runs`, and `run_verdicts` to the verdict.
Verdicts separate `artifact_consistent` from `run_accepted`: exit 0 means consistent and accepted, exit 1 means
inconsistent or consistently unaccepted, and exit 2 means unsupported or bounded evidence is unavailable.
`--require-raw-timing` requires original CPU timing; `--binary PATH` checks the manifest hash without running that
file. Verification does not prove original process success, physical traffic, or runtime authenticity.

## 23. Source Map (Primary Entry Points)

- Program entry: `main.cpp`
- Build provenance: `build-support/generate_provenance.py`
- Shared audit primitives:
  - `src/core/audit/host_environment.h`, `src/core/audit/host_environment.mm`
  - `src/core/audit/build_manifest.h`, `src/core/audit/build_manifest.cpp`
  - `src/core/audit/checkpoint_policy.h`
  - `src/core/audit/deterministic_rng.h`
- Independent CPU verifiers: `script-examples/cpu_verify_common.py`, `script-examples/verify_cpu_bandwidth_result.py`,
  `script-examples/verify_cpu_latency_result.py`, and `script-examples/verify_patterns_result.py`.
- Independent LLM verifier: `script-examples/verify_llm_result.py`, `script-examples/llm_verify_profiles.py`,
  and `script-examples/llm_verify_oracles.py`.
- Standalone CPU bandwidth (`-B` / `--cpu-bandwidth`):
  - `src/cpu_bandwidth/cpu_bandwidth.cpp`
  - `src/cpu_bandwidth/cpu_bandwidth_work_plan.cpp`
  - `src/cpu_bandwidth/cpu_bandwidth_oracle.cpp`
  - `src/cpu_bandwidth/cpu_bandwidth_executor.cpp`
  - `src/cpu_bandwidth/cpu_bandwidth_runner.cpp`
  - `src/cpu_bandwidth/cpu_bandwidth_json.cpp`
  - `src/cpu_bandwidth/cpu_bandwidth_output.cpp`
  - `src/cpu_bandwidth/cpu_bandwidth_sweep_runner.cpp`
- Standalone CPU latency (`-L` / `--cpu-latency`):
  - `src/cpu_latency/cpu_latency.cpp`
  - `src/cpu_latency/cpu_latency_work_plan.cpp`
  - `src/cpu_latency/cpu_latency_chain.cpp`
  - `src/cpu_latency/cpu_latency_executor.cpp`
  - `src/cpu_latency/cpu_latency_runner.cpp`
  - `src/cpu_latency/cpu_latency_json.cpp`
  - `src/cpu_latency/cpu_latency_output.cpp`
  - `src/cpu_latency/cpu_latency_sweep_runner.cpp`
- LLM memory profile with all four CPU and all four Metal phase/layout profiles:
  - `src/llm_memory/llm_memory.cpp`
  - `src/llm_memory/llm_sweep_runner.cpp`
  - `src/llm_memory/llm_work_plan.cpp`
  - `src/llm_memory/llm_kv_layout.cpp`
  - `src/llm_memory/llm_prefill.cpp`
  - `src/llm_memory/llm_backend.cpp`
  - `src/llm_memory/llm_cpu_backend.cpp`
  - `src/llm_memory/llm_metal_backend.h`
  - `src/llm_memory/llm_metal_backend.mm`
  - `src/llm_memory/llm_metal_kernels_source.h`
  - `src/llm_memory/llm_executor.cpp`
  - `src/llm_memory/llm_runner.cpp`
  - `src/llm_memory/llm_json.cpp`
  - `src/llm_memory/llm_output.cpp`
  - `src/llm_memory/llm_environment.h`
- Config parse/validate/derive:
  - `src/core/config/cli_parser.h`
  - `src/core/config/cli_parser.cpp`
  - `src/core/config/mode_selector.cpp`
  - `src/core/config/argument_parser.cpp`
  - `src/core/config/config_validator.cpp`
  - `src/core/config/sweep_utils.cpp`
- Memory allocation/init:
  - `src/core/memory/memory_manager.cpp`
  - `src/core/memory/memory_utils.cpp`
- Shared benchmark infrastructure and sweeps:
  - `src/benchmark/sweep_runner.cpp`
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
- Standalone patterns (`-P` / `--patterns`):
  - `src/pattern_benchmark/patterns.cpp`
  - `src/pattern_benchmark/patterns_work_plan.cpp`
  - `src/pattern_benchmark/patterns_oracle.cpp`
  - `src/pattern_benchmark/patterns_executor.cpp`
  - `src/pattern_benchmark/patterns_runner.cpp`
  - `src/pattern_benchmark/patterns_json.cpp`
  - `src/pattern_benchmark/patterns_sweep_runner.cpp`
  - `src/pattern_benchmark/patterns_output.cpp`
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
  - `src/output/json/json_output/file_writer.cpp`
  - `src/output/json/json_output/json_output_session.cpp`
- Assembly kernels:
  - `src/asm/*.s`
  - `src/asm/memory_write_reverse_pattern.s`, `src/asm/memory_write_strided_pattern.s`, `src/asm/memory_write_random_pattern.s` (arbitrary 64-bit pattern writes for `-P`; see [PATTERNS_WHITEPAPER.md](PATTERNS_WHITEPAPER.md))
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

# macOS Memory Benchmark - User Manual

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Key Concepts](#key-concepts)
4. [Command-Line Options](#command-line-options)
5. [Mode Compatibility](#mode-compatibility)
6. [Common Workflows](#common-workflows)
7. [Understanding Console Output](#understanding-console-output)
8. [JSON Output Format](#json-output-format)
9. [Visualization Scripts](#visualization-scripts)
10. [Running Under Active System Load](#running-under-active-system-load)
11. [Best Practices and Pitfalls](#best-practices-and-pitfalls)
12. [Troubleshooting](#troubleshooting)
13. [Additional Resources](#additional-resources)

---

## Introduction

`macOS-memory-benchmark` is a low-level Apple Silicon benchmark tool for:

- Main memory (DRAM) bandwidth and latency
- Cache bandwidth and latency
- Memory access pattern analysis (sequential/strided/random)

Target platform is **macOS on Apple Silicon**.

This manual focuses on practical usage and interpretation. For implementation details and latency methodology internals, see:

- [README.md](README.md)
- [CAPABILITIES.md](CAPABILITIES.md)
- [TECHNICAL_SPECIFICATION.md](TECHNICAL_SPECIFICATION.md)
- [LATENCY_WHITEPAPER.md](LATENCY_WHITEPAPER.md)
- [CORE_TO_CORE_WHITEPAPER.md](CORE_TO_CORE_WHITEPAPER.md)

---

## Quick Start

### Prerequisites

- Apple Silicon Mac
- Xcode Command Line Tools

Install tools:

```bash
xcode-select --install
```

### Install

Homebrew:

```bash
brew install timoheimonen/macOS-memory-benchmark/memory-benchmark
```

Build from source:

```bash
git clone https://github.com/timoheimonen/macOS-memory-benchmark.git
cd macOS-memory-benchmark
make
```

### First run

Running with no arguments shows help:

```bash
memory_benchmark
```

To run the standard memory benchmark:

```bash
memory_benchmark --benchmark
```

If built from source, use `./memory_benchmark` instead.

All command examples in this manual use the installed/`PATH` form (`memory_benchmark ...`).
If running from an uninstalled local source build, prefix commands with `./`.

For longer runs, prevent sleep:

```bash
caffeinate -i -d memory_benchmark --benchmark --count 10 --buffer-size 1024
```

If running a local build, use `./memory_benchmark` instead of `memory_benchmark` (see note in "[First run](#first-run)").

---

## Key Concepts

### Bandwidth vs latency

- **Bandwidth (GB/s)**: throughput for moving large amounts of data
- **Latency (ns)**: per-access delay, measured using dependent pointer chasing

Both matter: some workloads are throughput-bound, others are access-latency-bound.

### Memory hierarchy behavior

- L1 and L2 are much faster than DRAM
- Small test buffers can become cache-dominated
- Large buffers are needed for DRAM-focused measurement
- Auto cache tests target full detected L1/L2 capacity, then apply stride/page alignment (so printed buffer sizes can be slightly smaller)

### Pointer-chase latency and TLB locality

Latency tests use dependent pointer-chase chains. `--latency-tlb-locality-kb` controls how the chain is constructed:

- `1024` (default): randomized within 1 MB windows, plus randomized window order
- `0`: fully global random chain
- non-zero values must be multiples of system page size

`--latency-chain-mode` controls pointer-chain ordering policy:

- `auto` (default): preserves current behavior (`random-box` when locality > 0, `global-random` when locality = 0)
- `global-random`: full-buffer random permutation
- `random-box`: random order within locality boxes and random box traversal order
- `same-random-in-box`: same in-box random pattern reused across boxes (increasing box order)
- `diff-random-in-box`: independently randomized in-box pattern per box (increasing box order)

Modes other than `global-random` require non-zero `--latency-tlb-locality-kb`.

`--latency-stride-bytes` controls spacing between chain nodes. Smaller stride biases toward same-page reuse;
larger stride increases page turnover pressure.

Use `0` when you explicitly want stronger translation effects in the measured path.

When `--latency-tlb-locality-kb` is omitted in regular benchmark mode, main-memory latency output also runs an
automatic comparison and prints:

- TLB hit latency (16 KB locality)
- TLB miss latency (global random locality, `0`)
- Estimated page-walk penalty (`miss - hit`)

Each automatic comparison point is measured as P50 over three complete pointer-chase passes. This reduces the impact of
a single IRQ-inflated timing pass while keeping every candidate pass continuous.

If you explicitly set `--latency-tlb-locality-kb` (including `16` or `0`), this auto comparison is skipped.

### Pattern benchmarks

Pattern mode (`--patterns`) measures bandwidth sensitivity across:

- Sequential Forward
- Sequential Reverse
- Strided (64-byte stride)
- Strided (4096-byte stride)
- Strided (16 KiB stride)
- Strided (2 MiB stride)
- Random Uniform

Pattern bandwidth is effective payload bandwidth, not inferred physical DRAM or cache-bus traffic. Each valid access
contributes the 32-byte payload actually processed by the pattern kernel; this is half of a 64-byte cache line, not a
complete cache-line payload. Copy counts both the read and write sides, for 64 payload bytes per logical copy access.
The bandwidth numerator is the exact planned payload completed by every worker and pass.

Strided results use a deterministic per-worker work plan. A worker's last candidate address is included when the complete
32-byte access fits within its cache-line-aligned chunk. Reported read/write bandwidth is calculated from the exact sum
of these worker payloads, while copy bandwidth counts both the read and write sides. On every pass,
the ARM64 kernel advances the starting phase by 32 bytes modulo the stride, so sparse tests do not repeatedly touch
only the phase-zero addresses. All passes for one worker execute inside one assembly call, and the bandwidth numerator
uses the exact phase-aware access count. Strided warmup executes one complete phase cycle (`stride / 32` passes).
Stride labels are byte distances, not page-size claims. For example, 4096 bytes is not one native page on a macOS
system with 16 KiB pages; JSON records the native page size and whether the configured stride equals it.

Phase rotation means a strided pass can contain a different number of valid accesses than the preceding pass. In JSON,
`accesses_per_pass` deliberately retains the phase-zero count and `accesses_per_pass_semantics` identifies it as
`"phase-zero-count"`; `min_accesses_per_pass`, `max_accesses_per_pass`, and `phase_period_passes` describe the variation.
Use `total_accesses` and `total_payload_bytes` as the authoritative exact totals. Do not calculate either total as
`accesses_per_pass * passes`. Non-strided measurements use `"constant-count"` semantics and equal minimum/maximum counts.

Every active strided worker must have at least two valid addresses and therefore make at least one genuine stride
transition. If the requested thread count cannot satisfy that rule, the benchmark automatically uses fewer workers for
that strided pattern. Sequential and random patterns continue to use the configured thread count. This distinction is
especially important for large strides and small buffers.

Parallel pattern timing begins only after every actual worker has completed its best-effort QoS setup attempt and
reached the ready gate. It stops when the last worker finishes its measured work; worker teardown and thread joining
remain outside the measured interval. Work planning, random-list creation and partitioning, thread creation, QoS setup,
and ready-gate waiting are also excluded. For the random pattern, the global access list is partitioned into per-worker
local index lists before any timed call, so index filtering and list allocation are not included in reported bandwidth.
QoS is a best-effort macOS scheduler hint; workers are not pinned to cores, and effective placement can still vary.

Unless `--iterations` is supplied explicitly, each sequential, strided, and random read/write/copy sample first runs an
excluded pilot and scales the measured pass count toward 150 ms; 100–250 ms is the intended measurement window. The
pilot uses the same operation, access pattern, and worker shape as the measured sample, so it also provides
preconditioning. If `--iterations` is explicit, that value is the measured pass count and the calibration pilot is not
run. In both modes, an operation-specific same-shape warmup runs before the measured operation: read warms read, write
warms write, copy warms copy, random warmup traverses the complete measured list, and strided warmup completes a phase
cycle.

Random offsets are the unique, no-replacement prefix of a deterministic seeded permutation of valid 32-byte-aligned
slots. `--seed` selects that permutation. If omitted, one seed is generated once for the command; regenerating the list
from the same resolved seed means every `--count` loop uses the same random workload. Random read, write, and copy
warmups traverse the full measured address list rather than a prefix.

Pattern results therefore describe steady-state, warm-memory bandwidth. They do not measure cold allocation, first
touch, or a cold-cache/cold-TLB start; warmup and, in automatic mode, the excluded pilot intentionally prepare the tested
access shape.

Across repeated `--count` loops, the seven pattern groups rotate in deterministic cyclic Latin-square order. This spreads
first/last-position and thermal-drift effects while preserving reproducibility. Operations inside each group remain in
fixed read, write, copy order, with operation-specific warmup before each one. The resolved random seed and workload are
identical across the repeated loops.

For `--count > 1`, the console headline is the median (P50), not the last loop or arithmetic mean. Pattern statistics
also report coefficient of variation (CV). The console warns when CV exceeds 5% for sequential or 64-byte-stride
operations and 10% for 4096-byte, 16 KiB, 2 MiB, or random operations. These thresholds flag a noisy measurement; they
are not correctness limits. If a workload cannot produce a valid measurement, console output shows `N/A` plus an
explicit status/reason, and JSON stores `value_gb_s: null` plus status metadata. Never interpret an unavailable value as
zero bandwidth.

---

## Command-Line Options

Options that take a value, such as `--buffer-size`, `--cache-size`, `--threads`, `--latency-samples`, and `--output`,
must be specified at most once per command.

Long options require a double dash (`--`). A single dash is reserved for one-character short options, so legacy
forms such as `-buffersize` or `-benchmark` are invalid.

| Short | Long |
|---|---|
| `-B` | `--benchmark` |
| `-P` | `--patterns` |
| `-W` | `--only-bandwidth` |
| `-L` | `--only-latency` |
| `-T` | `--analyze-tlb` |
| `-C` | `--analyze-core2core` |
| `-b` | `--buffer-size` |
| `-i` | `--iterations` |
| `-r` | `--count` |
| `-t` | `--threads` |
| `-k` | `--cache-size` |
| `-n` | `--latency-samples` |
| `-s` | `--latency-stride-bytes` |
| `-m` | `--latency-chain-mode` |
| `-l` | `--latency-tlb-locality-kb` |
| `-D` | `--tlb-density` |
| `-u` | `--non-cacheable` |
| `-o` | `--output` |
| `-S` | `--sweep` |
| `-X` | `--sweep-max-runs` |
| `-h` | `--help` |

### Core controls

#### `--buffer-size <MB>`

- Main buffer size in MB (per main buffer)
- Default: `512`
- Auto-capped to memory safety limit
- `--buffer-size 0` is valid only with `--only-latency` and disables main-memory latency path

#### `--iterations <count>`

- Bandwidth loop iterations
- Default: `1000`
- Positive integer
- Not allowed with `--only-latency`
- In `--patterns`, automatic calibration is used when this option is omitted; the default value is not forced as the
  measured pass count
- In `--patterns`, an explicitly supplied value is the measured pass count for each read/write/copy sample and bypasses
  the calibration pilot; operation-specific warmup still runs

#### `--count <count>`

- Full benchmark loop count
- Default: `1`
- Positive integer
- Use `5` to `10` for stable statistics

#### `--threads <count>`

- Thread count for bandwidth tests
- Default: detected core count in standard benchmark mode; detected performance-core count in `--patterns` mode
- If above available cores, it is capped
- An explicit `--threads` value may request more than the performance-core default, up to the detected cap
- In pattern mode this is a requested count; sparse strided work may use fewer effective workers so each worker performs
  at least one genuine stride transition
- Pattern workers request best-effort macOS QoS but are not pinned to specific cores
- Latency tests remain single-threaded

### Mode selection

#### `--benchmark`

- **Required** to run standard memory benchmark (bandwidth + latency)
- Mutually exclusive with `--patterns`
- Can be combined with `--only-bandwidth`, `--only-latency`, `--cache-size`, `--threads`, and other modifier flags
- Running without this flag (or `--patterns`) shows help and exits

#### `--patterns`

- Runs only access-pattern benchmarks
- Skips standard bandwidth/latency sections
- Automatically calibrates each read/write/copy sample toward 150 ms (intended window 100–250 ms) unless
  `--iterations` is explicitly supplied
- Uses detected performance-core count by default; explicit `--threads` can request a different detected-core count
- Rotates pattern groups across repeated loops; read/write/copy order within each group stays fixed

#### `--only-bandwidth`

- Runs bandwidth paths only
- **Requires `--benchmark`**
- Incompatible with: `--patterns`, `--cache-size` (any value including `0`), `--latency-samples`

#### `--only-latency`

- Runs latency paths only
- **Requires `--benchmark`**
- Incompatible with: `--patterns`, `--iterations`
- Supports selective target disabling:
  - `--buffer-size 0` disables main-memory latency
  - `--cache-size 0` disables cache latency
  - both zero is invalid

#### `--analyze-tlb`

- Runs standalone TLB analysis mode only
- Can be combined only with optional `--output <file>`, `--latency-stride-bytes <bytes>`, `--latency-chain-mode <mode>`, `--tlb-density <low|medium|high>`, `--seed <uint64>`, `--sweep <key=...>`, and `--sweep-max-runs <count>`
- Uses latency stride from `--latency-stride-bytes` (same default as standard latency mode). Analyze-TLB stride must be pointer-aligned and must not exceed the system page size; it does not need to divide the page size. The default standard profile performs a 15-point base locality sweep, stride-clamped to `max(16KB, 2*stride)` up to `256MB`, then inserts page-aligned refinement points near detected knees/boundaries
- Builds a page-native spread chain with exactly one pointer node per requested page and a cache-line-dense packed control with the same node and unique-cache-line counts. Each scheduler task measures both layouts in one round, alternates pair order, and stores the same-round `spread - packed` translation delta
- Detects likely private-cache knee candidates from spread latency as a separate diagnostic and reports whether the region may interfere with interpretation; accepted L1/L2 claims still require the paired translation-delta and validation gates
- Reports the validated bracket range (`inferred_entries_min`/`inferred_entries_max`) as the primary L1/L2 result; `inferred_entries` is an explicitly secondary midpoint estimate
- Builds a round-by-point matrix from same-round `spread - packed` deltas. Acceptance requires a paired median effect of at least `0.5ns`, a deterministic percentile-bootstrap 95% CI above the measured noise floor, persistence at both following points, and the same evidence in an independent validation pass
- Retains rejected boundary candidates, their confidence intervals, persistence counts, and rejection reasons in JSON
- Reports one 512 MiB paired comparison object when the analysis buffer is at least `512 MiB`: spread P50, packed P50, the median of same-round `spread - packed` deltas, spread/packed page counts, and active cache-line footprint. These are cache-hot translation-stress timings, not direct DRAM latency or an isolated page-table-walk cost
- Emits explicit `complete`, `interrupted`, or `partial` status. Boundary conclusions are suppressed unless the planned sweep completed
- Selects the largest `1024/512/256 MiB` buffer whose predicted buffer-plus-scratch peak fits the available-memory budget. The compact settings block reports the run identity, buffer-lock/QoS outcome, estimated peak versus budget, sweep plan, and rough duration. Full pointer-access and memory estimates remain in JSON
- Calibrates each spread and packed measurement from a timed pilot toward the active profile's target duration while requiring a minimum number of complete chain cycles
- Uses adaptive balanced rounds: every round measures each planned locality once in seeded cyclic-Latin order, and a pass stops after its minimum when every point's deterministic bootstrap median CI is narrow enough, or at the profile maximum
- Attempts `mlock()` as a best-effort noise reduction. Failure reports errno and its message, records the failure in JSON, and continues with the allocated buffer unlocked
- Requests `user-interactive` QoS for the main benchmark thread as a best-effort hint. Console and JSON report whether the request was applied and its return code; failure emits a warning and continues
- Rebuilds every standalone TLB pair from recorded task and layout seeds; pointer values are written in physical-slot order and every chain is verified to visit all nodes and return to its head. Latency-chain behavior outside standalone TLB analysis remains unchanged
- A user interrupt remains a successful graceful-shutdown return when partial JSON can be written; consumers must use `status` and `conclusions_valid` rather than the process code to accept conclusions
- Detailed methodology and JSON contract: `TLB_ANALYSIS_WHITEPAPER.md`

#### `--tlb-density <level>`

- Applies only to `--analyze-tlb`
- Default: `medium` (`standard`)
- Accepted values: `low`, `medium`, `high`
- `low` (`quick`): 15-point base sweep, no refinement pass, 7-12 rounds, 5 ms target per chain. Its console conclusions are screening estimates and explicitly advise confirmation with `medium` or `high`
- `medium` (`standard`): 15-point base sweep + refinement pass, 10-20 rounds, 10 ms target per chain
- `high` (`exhaustive`): 29-point base sweep + refinement pass, 15-30 rounds, 20 ms target per chain

#### `--seed <uint64>`

- Applies only to `--patterns` or `--analyze-tlb`
- In `--patterns`, selects the deterministic unique/no-replacement permutation prefix of valid 32-byte-aligned random
  offsets; the same resolved seed reproduces the workload, and every `--count` loop repeats it
- In `--patterns`, one unsigned 64-bit seed is generated once for the command when omitted
- In `--analyze-tlb`, controls base/refinement/validation/large-locality round order, pointer-chain construction, and
  deterministic bootstrap resampling
- In `--analyze-tlb`, the same seed reproduces planner order, derived task seeds, and chain permutations
- In `--analyze-tlb`, one unsigned 64-bit seed is generated for the command when omitted and reused by every TLB run in
  a Cartesian sweep
- Standalone TLB JSON stores the resolved seed, source (`user` or `generated`), schedule policy, and each task seed

#### `--analyze-core2core`

- Runs standalone two-thread cache-line handoff (ping-pong) mode only
- Can be combined only with optional `--output <file>`, `--count <count>`, `--latency-samples <count>`, `--sweep count=...`, `--sweep latency-samples=...`, and `--sweep-max-runs <count>`
- Executes three scheduler-hint scenarios: `no_affinity_hint`, `same_affinity_tag`, and `different_affinity_tags`
- Reports round-trip latency, one-way estimate (`round_trip / 2`), and sample distribution stats (P50/P90/P95/P99/stddev/min/max)
- Includes per-thread QoS/affinity hint status in console and JSON output
- Notes explicitly that macOS user-space cannot hard-pin exact core IDs
- Detailed methodology and JSON contract: [CORE_TO_CORE_WHITEPAPER.md](CORE_TO_CORE_WHITEPAPER.md)

### Latency-specific controls

#### `--latency-samples <count>`

- Sample count per latency test
- Default: `1000`
- Positive integer
- More samples improve percentile stability at the cost of run time

#### `--latency-stride-bytes <bytes>`

- Pointer-chain stride for latency tests
- Default: `256`
- Must be `> 0`
- Must be a multiple of 8 bytes (pointer size on Apple Silicon)
- With `--analyze-tlb`, must not exceed the system page size. The page-native spread builder rounds effective spacing up to a cache-line multiple, while the packed control uses one node per cache line; exact page-size divisibility is not required
- Use smaller values (for example `64`) to increase same-page cache-line activity and reduce TLB sensitivity

#### `--latency-chain-mode <mode>`

- Pointer-chain construction policy for latency paths
- Default: `auto`
- Accepted values: `auto`, `global-random`, `random-box`, `same-random-in-box`, `diff-random-in-box`
- `global-random` works with `--latency-tlb-locality-kb 0`
- `random-box`, `same-random-in-box`, and `diff-random-in-box` require `--latency-tlb-locality-kb > 0`
- In `--analyze-tlb` mode, `global-random` is rejected because it ignores locality windows and would make locality sweep boundaries misleading
- In `--analyze-tlb`, modes select page-native traversal policy: `random-box` randomizes page order and offsets, `same-random-in-box` uses increasing pages with a shared offset, and `diff-random-in-box` uses increasing pages with independently selected offsets. Compare results only when the effective chain mode is identical; increasing-page modes are intentionally sensitive to traversal order and hardware prefetch behavior

#### `--latency-tlb-locality-kb <size_kb>`

- Pointer-chain locality window for latency path
- Default: `1024`
- `0` disables locality mode (global random chain)
- Non-zero values must be exact multiples of system page size
- In regular benchmark mode, explicitly setting this option disables automatic TLB hit/miss comparison lines

### Cache and memory hint controls

#### `--cache-size <KB>`

- Enables custom cache test size
- Non-zero range: `16` to `1048576` KB (1 GB)
- `0` is accepted only with `--only-latency` and disables cache latency target
- When set to non-zero, auto L1/L2 detection is replaced by custom cache target

#### `--non-cacheable`

- Applies cache-discouraging `madvise()` hints
- Best effort only; this does **not** create truly uncached memory

### Output

#### `--output <file>`

- Saves JSON output
- Relative path writes under current working directory
- Parent directories are created automatically

#### `--sweep <key=value1,value2>`

- Runs a Cartesian parameter sweep and writes one combined JSON result
- Requires `--output <file>`
- Can be repeated to sweep multiple parameters
- Supported keys: `buffer-size`, `cache-size`, `threads`, `latency-tlb-locality-kb`, `latency-stride-bytes`, `latency-chain-mode`, `tlb-density`, `count`, `latency-samples`
- `tlb-density` applies only with `--analyze-tlb`
- `--patterns` supports `buffer-size` and `threads`
- In a `--patterns` thread sweep, each `threads` value is the requested count. A strided pattern may reduce its
  effective worker count to keep at least two valid strided addresses per active worker, so sparse-stride results must
  not be interpreted as requested-thread scaling when this reduction applies
- `--benchmark --only-bandwidth` supports `buffer-size` and `threads`
- `--benchmark --only-latency` supports `buffer-size`, `cache-size`, and latency chain/locality/stride keys
- `--analyze-tlb` supports `latency-stride-bytes`, `latency-chain-mode`, and `tlb-density`
- `--analyze-core2core` supports `count` and `latency-samples`

#### `--sweep-max-runs <count>`

- Maximum number of generated sweep combinations
- General default: `256`; default with `--analyze-tlb`: `16`
- An explicit `--sweep-max-runs` value overrides the mode-specific default
- Prevents accidental very large Cartesian sweeps
- Every generated configuration is validated before the first run
- Combined JSON is atomically checkpointed after each completed run and records `status`, `planned_runs`, `completed_runs`, and `conclusions_valid`
- A parameter key may appear only once in one sweep command

#### `-h`, `--help`

- Print help and exit

---

## Mode Compatibility

### Valid combinations

```bash
# Full benchmark
memory_benchmark --benchmark --count 10 --buffer-size 1024 --output full.json

# Pattern-only
memory_benchmark --patterns --count 5 --buffer-size 512 --output patterns.json

# Bandwidth-only
memory_benchmark --benchmark --only-bandwidth --threads 8 --count 5

# Latency-only (both main + cache)
memory_benchmark --benchmark --only-latency --latency-samples 5000 --count 10

# Latency-only (main memory only)
memory_benchmark --benchmark --only-latency --cache-size 0 --buffer-size 1024

# Latency-only (cache only)
memory_benchmark --benchmark --only-latency --buffer-size 0 --cache-size 2048

# Standalone TLB analysis
memory_benchmark --analyze-tlb

# Standalone TLB analysis with JSON export
memory_benchmark --analyze-tlb --output tlb_analysis.json

# Standalone TLB analysis with custom stride
memory_benchmark --analyze-tlb --latency-stride-bytes 128 --output tlb_analysis_stride128.json

# Standalone TLB analysis with explicit chain mode
memory_benchmark --analyze-tlb --latency-chain-mode same-random-in-box --output tlb_analysis_same_box.json

# Standalone TLB analysis with quick low-density sweep (no refinement)
memory_benchmark --analyze-tlb --tlb-density low --output tlb_analysis_low.json

# Reproducible standalone TLB analysis
memory_benchmark --analyze-tlb --seed 123456789 --output tlb_analysis_seeded.json

# Standalone TLB analysis parameter sweep
memory_benchmark --analyze-tlb --sweep latency-stride-bytes=64,128 --sweep tlb-density=medium,high --sweep-max-runs 4 --output tlb_stride_density_sweep.json

# Standalone core-to-core handoff analysis
memory_benchmark --analyze-core2core

# Standalone core-to-core analysis with deeper sampling + JSON
memory_benchmark --analyze-core2core --count 5 --latency-samples 2000 --output core2core.json

# Standalone core-to-core sample-depth sweep
memory_benchmark --analyze-core2core --count 3 --sweep latency-samples=500,1000,2000 --output core2core_sample_sweep.json

# Benchmark latency sweep over 3 buffer sizes and 3 locality windows (9 runs)
memory_benchmark --benchmark --only-latency --count 5 --sweep buffer-size=256,512,1024 --sweep latency-tlb-locality-kb=16,1024,0 --output latency_sweep.json

# Thread scaling sweep for bandwidth
memory_benchmark --benchmark --only-bandwidth --count 5 --sweep buffer-size=512,1024 --sweep threads=1,4,8 --output bandwidth_thread_sweep.json
```

### Invalid combinations

```bash
# invalid: --benchmark with --patterns (mutually exclusive)
memory_benchmark --benchmark --patterns

# invalid: pattern mode with only-bandwidth
memory_benchmark --patterns --only-bandwidth

# invalid: pattern mode with only-latency
memory_benchmark --patterns --only-latency

# invalid: latency samples with only-bandwidth
memory_benchmark --benchmark --only-bandwidth --latency-samples 5000

# invalid: iterations with only-latency
memory_benchmark --benchmark --only-latency --iterations 2000

# invalid: both latency targets disabled
memory_benchmark --benchmark --only-latency --buffer-size 0 --cache-size 0

# invalid: analyze-tlb with unsupported extra option
memory_benchmark --analyze-tlb --buffer-size 1024

# invalid: analyze-core2core with unsupported extra option
memory_benchmark --analyze-core2core --threads 4

# invalid: analyze-core2core sweep supports only count and latency-samples
memory_benchmark --analyze-core2core --sweep threads=1,2 --output core2core_sweep.json
```

---

## Common Workflows

### Quick baseline

```bash
memory_benchmark --benchmark
```

Good for a fast health check.

### Statistical baseline (recommended)

```bash
caffeinate -i -d memory_benchmark --benchmark --count 10 --buffer-size 1024 --output baseline.json
```

Use this for comparisons across machines or software versions.

### Pattern analysis

```bash
memory_benchmark --patterns --count 10 --buffer-size 512 --seed 123456789 --output patterns.json
```

Shows how effective payload bandwidth changes under different access patterns. Reuse the same explicit seed and command
line for comparisons; inspect the median, CV, requested/effective worker counts, and measurement status in the output.

### Manual pattern stability matrix

Run this matrix separately from routine unit/integration tests on an idle Apple Silicon system. These are test recipes,
not benchmark results; this manual does not claim that the matrix has been run on a particular machine.

| Buffer | Thread configurations | Count | Purpose |
|---|---|---:|---|
| 8 MiB | 1, detected P-core default, all detected cores | 10 | Cache-resident and sparse-stride regression |
| 64 MiB | 1, detected P-core default, all detected cores | 10 | Transition around cluster-cache-sized workloads |
| 512 MiB | 1, detected P-core default, all detected cores | 10 | DRAM-oriented pattern stability |
| 1 GiB | detected P-core default, all detected cores | 5 | Large working set and thermal/order behavior |

Use one explicit seed for the complete matrix. Omit `--threads` for the detected P-core default; use `--threads 1` for
the single-worker case and `--threads <all-detected-cores>` for the all-core request. For example:

```bash
# Repeat for buffer sizes/thread configurations in the table.
caffeinate -i -d ./memory_benchmark --patterns --buffer-size 64 --threads 1 \
  --count 10 --seed 123456789 --output pattern-stability-64m-t1.json

# P-core default: intentionally omit --threads.
caffeinate -i -d ./memory_benchmark --patterns --buffer-size 64 \
  --count 10 --seed 123456789 --output pattern-stability-64m-pcores.json
```

Record the environment before comparing files: hardware/CPU, macOS and benchmark version, power source/mode, thermal
state, active displays and notable background load, command, seed, and start time. Then record one row per
pattern/operation using this template:

| Pattern | Operation | Status | Median GB/s | CV % | Requested/effective threads | Median duration s | Phase-zero / min-max accesses | Phase period | Passes | Exact total accesses/payload bytes | Logical working set bytes | Notes |
|---|---|---|---:|---:|---|---:|---|---:|---:|---|---:|---|
| _example: strided_2mb_ | _read_ | _measured/skipped/..._ | _from JSON_ | _from JSON_ | _requested/effective_ | _from per-loop records_ | _from JSON_ | _from JSON_ | _from JSON_ | _from exact total fields_ | _from JSON_ | _noise/status reason_ |

Check that measured samples have positive duration and internally consistent exact payload accounting, sparse cases
report their effective worker reduction, and unavailable cases use explicit status rather than zero. As a stability
review target, investigate CV above 5% for sequential/64-byte stride or above 10% for sparse/random operations. Record
the observed result and conditions even when a target is missed; do not rewrite a noisy run as a pass.

### Latency analysis with TLB-locality control

```bash
# default locality mode (1 MB window)
memory_benchmark --benchmark --only-latency --buffer-size 1024 --latency-samples 5000 --count 10 --output lat_default_1mb.json

# global random chain
memory_benchmark --benchmark --only-latency --buffer-size 1024 --latency-samples 5000 --latency-tlb-locality-kb 0 --count 10 --output lat_global.json

# same in-box random pattern (good for prefetch-vs-TLB comparisons)
memory_benchmark --benchmark --only-latency --buffer-size 1024 --latency-samples 5000 --latency-tlb-locality-kb 16 --latency-chain-mode same-random-in-box --count 10 --output lat_same_box.json
```

### Regular benchmark with automatic DRAM TLB breakdown

```bash
memory_benchmark --benchmark --count 1
```

This prints `Average latency` plus auto-derived `TLB hit latency`, `TLB miss latency`, and
`Estimated page-walk penalty` when `--latency-tlb-locality-kb` is not explicitly set.

### Canonical standalone TLB analysis

```bash
memory_benchmark --analyze-tlb --output tlb_analysis.json
```

Quick first checks in the output file:

- `tlb_analysis.l1_tlb_detection.boundary_locality_kb`
- `tlb_analysis.l2_tlb_detection.boundary_locality_kb`
- `tlb_analysis.status` and `tlb_analysis.conclusions_valid`
- `tlb_analysis.large_locality_paired_comparison.translation_delta_p50_ns`
- `tlb_analysis.large_locality_paired_comparison.active_cache_line_footprint_bytes`

### Custom cache target

```bash
memory_benchmark --benchmark --cache-size 4096 --threads 1 --count 5 --output cache_4mb.json
```

### Cache-size sweep + trend plotting

```bash
./script-examples/latency_test_script.sh
python3 script-examples/plot_cache_percentiles.py script-examples/final_output.txt --metric median
```

### Built-in sweep JSON

```bash
memory_benchmark --benchmark --only-latency --count 5 --sweep buffer-size=256,512,1024 --sweep latency-stride-bytes=64,256 --output latency_sweep.json
```

The command above creates six runs and stores each run's normal benchmark JSON under `runs[].result`. TLB-analysis
sweeps use the same envelope with `base_mode: "analyze_tlb"` and support `latency-stride-bytes`,
`latency-chain-mode`, and `tlb-density`. Core-to-core sweeps use the same envelope with
`base_mode: "analyze_core2core"` and support only `count` and `latency-samples`.

---

## Understanding Console Output

### 1) Configuration section

Shows active settings and detected hardware.

Important fields:

- Buffer size and peak concurrent allocation estimate
- Loop/iteration/sample configuration
- Thread count
- Latency chain mode
- TLB locality setting
- Detected or custom cache sizes

### 2) Main memory bandwidth

Displayed as read/write/copy GB/s. Higher is better.

### 3) Main memory latency

Average latency in ns. Lower is better.

When `--latency-tlb-locality-kb` is not explicitly provided, this section also prints:

- `TLB hit latency (16 KB locality)`
- `TLB miss latency (global random locality)`
- `Estimated page-walk penalty (miss - hit)`

The auto-TLB hit/miss values are P50 values from three complete comparison passes per point. The main `Average latency`
headline remains one continuous pointer-chase pass.

### 4) Cache bandwidth and latency

L1/L2 or custom cache section, depending on `--cache-size` use.

### 5) Pattern benchmark output

Shows effective payload bandwidth for read, write, and copy, plus the relative percentage against sequential forward
when both values are measured. Copy includes its read and write payload. With `--count > 1`, the displayed headline is
the median of measured loops. An unavailable operation is shown as `N/A [status: reason]`, not `0 GB/s`.

Do not treat a low strided or random result as proof of prefetch behavior, cache thrashing, or a TLB boundary. Pattern
mode does not isolate those mechanisms, and the console intentionally does not emit those diagnoses. Use
`--analyze-tlb` for the controlled paired TLB analysis, and use matched buffer/stride/thread experiments for broader
cache or prefetch hypotheses.

### 6) Statistics (`--count > 1`)

Includes values such as:

- Average
- P50 (Median)
- P90, P95, P99
- Std Dev
- Coefficient of variation (CV)
- Min / Max

When automatic TLB comparison is active (you did not explicitly set `--latency-tlb-locality-kb`),
statistics also include dedicated sections for:

- `TLB Hit Latency (ns)`
- `TLB Miss Latency (ns)`
- `Estimated Page-Walk Penalty (ns)`

For noisy systems, prioritize median and P95/P99 rather than single fastest/slowest values.

Pattern statistics emit noise warnings above a pattern-specific CV threshold: 5% for sequential and 64-byte stride,
and 10% for 4096-byte, 16 KiB, 2 MiB, and random patterns. Treat a warning as a request to repeat under steadier
conditions or increase `--count`; it does not invalidate the sample automatically.

### 7) Standalone TLB analysis

`--analyze-tlb` keeps the default console report compact while retaining complete diagnostics in JSON:

- `Run` identifies CPU, page size, stride, profile, requested/effective chain mode, and seed on one line.
- `Resources` reports buffer-lock and QoS outcomes plus estimated peak versus the memory budget.
- `Sweep` reports the IEC-formatted locality range and whether the 512 MiB comparison is available.
- Each locality occupies one line: paired translation delta first, followed by spread/packed P50 controls and active cache-line footprint.
- A shared legend defines `*` as a below-64-node short-cycle diagnostic; per-point page counts and full chain diagnostics remain in JSON.
- Work estimates show points, adaptive round range, and rough duration. Pointer-access envelopes remain in JSON.
- The final report contains one compact run identifier, optional refinement count, completion status, L1/L2 conclusions, and the large-locality paired comparison.
- `quick` conclusions carry a visible screening-estimate note and should be confirmed with `medium` or `high` before being treated as hardware boundaries.

Displayed TLB sizes use `KiB`/`MiB`, and sub-resolution negative deltas are rendered as `0.00 ns` rather than `-0.00 ns`.
The large-locality result remains cache-hot paired translation stress, not DRAM latency or an isolated page-table-walk cost.

**Note:** Chain diagnostics (`pointer_count`, `unique_pages_touched`, etc.) appear in JSON output only when `--latency-stride-bytes` is explicitly set; they are not displayed in console output.

---

## JSON Output Format

### Standard benchmark JSON shape

```json
{
  "configuration": { ... },
  "execution_time_sec": 427.5,
  "main_memory": { ... },
  "cache": { ... },
  "timestamp": "2026-03-09T14:57:56Z",
  "version": "0.55.4"
}
```

Note: The `configuration` block includes fields such as `latency_chain_mode` (the resolved pointer-chain mode used), `latency_tlb_locality_kb`, `use_custom_cache_size`, and other runtime settings. For the complete configuration schema, inspect a sample output file or examine the configuration builder code.

### Pattern benchmark JSON shape

```json
{
  "configuration": {
    "mode": "patterns",
    "pattern_schema_version": 2,
    "methodology_version": "pattern-v2-phase-calibrated-seeded",
    "pattern_seed": "123456789",
    "pattern_seed_source": "user",
    "pattern_seed_encoding": "uint64-decimal-string",
    "pattern_pass_policy": "automatic-duration-calibration",
    "calibration_target_seconds": 0.15,
    "calibration_window_min_seconds": 0.1,
    "calibration_window_max_seconds": 0.25,
    "calibration_max_corrections": 2,
    "warmup_semantics": "steady-state-same-shape",
    "pattern_execution_order_policy": "cyclic-latin-square-across-count-loops",
    "operation_execution_order_policy": "fixed-read-write-copy-with-operation-specific-warmup",
    "thread_selection_policy": "performance-core-count-default",
    "qos_policy": "best-effort-scheduler-hint-no-core-pinning"
  },
  "patterns": {
    "strided_2mb": {
      "methodology_version": "pattern-v2-phase-calibrated-seeded",
      "access_size_bytes": 32,
      "stride_bytes": 2097152,
      "requested_threads": 4,
      "effective_threads": 2,
      "large_page_backing_verified": false,
      "large_page_backing_status": "not-verified",
      "bandwidth": {
        "read_gb_s": {
          "status": "measured",
          "headline": "median_p50",
          "value_gb_s": 0.222,
          "values_gb_s": [0.218, 0.222, 0.226],
          "statistics": {
            "median_p50": 0.222,
            "coefficient_of_variation_pct": 1.8
          },
          "measurements": [
            {
              "status": "measured",
              "value_gb_s": 0.222,
              "elapsed_seconds": 0.15,
              "access_size_bytes": 32,
              "requested_threads": 4,
              "effective_threads": 2,
              "accesses_per_pass": 256,
              "accesses_per_pass_semantics": "phase-zero-count",
              "min_accesses_per_pass": 252,
              "max_accesses_per_pass": 256,
              "passes": 4096,
              "total_accesses": 1040384,
              "total_payload_bytes": 33292288,
              "phase_period_passes": 65536,
              "benchmark_loop_index": 0,
              "pattern_order_index": 5
            }
          ]
        }
      }
    }
  },
  "execution_time_sec": 7.5,
  "timestamp": "...",
  "version": "..."
}
```

This is a structure example, not a recorded performance result; omitted patterns and operations use the same shape.
Schema 2 stores explicit measurement state. A skipped, invalid, or interrupted operation has `value_gb_s: null`, an empty
or partial value set as applicable, and a `status`/`reason`; it is not serialized as zero. Aggregate `value_gb_s` is the
single measurement for one loop or median P50 for multiple measured loops. Copy `total_payload_bytes` includes both read
and write payload. Per-loop records retain pilot/final timing, exact access/pass/payload accounting, working-set and phase
metadata, seed, requested/effective threads, native page comparison, and execution-order indexes.

For phase-rotated strided records, `accesses_per_pass` is the phase-zero count, not an average. The minimum/maximum and
phase-period fields describe pass-to-pass variation, while `total_accesses` and `total_payload_bytes` are computed from
the complete work plan. They can differ from `accesses_per_pass * passes` and must be consumed directly.

`strided_2mb` names a 2 MiB virtual address stride. It is not evidence that macOS supplied 2 MiB physical pages:
`large_page_backing_status: "not-verified"` and `large_page_backing_verified: false` must be interpreted literally.

### Sweep JSON shape

```json
{
  "status": "complete",
  "planned_runs": 6,
  "completed_runs": 6,
  "conclusions_valid": true,
  "configuration": {
    "mode": "sweep",
    "base_mode": "benchmark",
    "run_count": 6,
    "sweep_max_runs": 256,
    "sweep_parameters": {
      "buffer-size": [256, 512, 1024],
      "latency-stride-bytes": [64, 256]
    }
  },
  "runs": [
    {
      "index": 0,
      "parameters": {
        "buffer-size": 256,
        "latency-stride-bytes": 64
      },
      "result": { "...": "normal benchmark, pattern, TLB, or core-to-core JSON payload" }
    }
  ],
  "execution_time_sec": 123.4,
  "timestamp": "2026-04-29T12:00:00Z",
  "version": "0.57.0"
}
```

### Latency payload structure (current)

Latency values are structured objects, not scalars. The example below uses real values from a benchmark run.

When `--latency-stride-bytes` is explicitly set (with a non-default value), latency sections also include `chain_diagnostics`.

```json
"latency": {
  "auto_tlb_breakdown": {
    "page_walk_penalty_ns": {
      "statistics": {
        "average": 80.94071304166667,
        "max": 81.21026791666667,
        "median": 80.93019520833334,
        "min": 80.71570645833333,
        "p90": 81.10030966666667,
        "p95": 81.15528879166668,
        "p99": 81.19927209166667,
        "stddev": 0.1762455963793484
      },
      "values": [80.71570645833333, 80.91202333333334, 80.93019520833334, 81.21026791666667, 80.93537229166668]
    },
    "tlb_hit_ns": {
      "statistics": {
        "average": 15.289083916666666,
        "max": 16.014336875,
        "median": 15.122818125,
        "min": 15.072767291666667,
        "p90": 15.668637291666668,
        "p95": 15.841487083333334,
        "p99": 15.97976691666667,
        "stddev": 0.4065809440444531
      },
      "values": [15.072767291666667, 16.014336875, 15.085409375, 15.122818125, 15.150087916666667]
    },
    "tlb_miss_ns": {
      "statistics": {
        "average": 96.22979695833334,
        "max": 96.92636020833334,
        "median": 96.08546020833334,
        "min": 95.78847375,
        "p90": 96.68905054166666,
        "p95": 96.807705375,
        "p99": 96.90262924166667,
        "stddev": 0.4351283262178906
      },
      "values": [95.78847375, 96.92636020833334, 96.01560458333334, 96.33308604166666, 96.08546020833334]
    }
  },
  "average_ns": {
    "statistics": {
      "average": 15.289083916666666,
      "max": 16.014336875,
      "median": 15.122818125,
      "min": 15.072767291666667,
      "p90": 15.668637291666668,
      "p95": 15.841487083333334,
      "p99": 15.97976691666667,
      "stddev": 0.4065809440444531
    },
    "values": [15.072767291666667, 16.014336875, 15.085409375, 15.122818125, 15.150087916666667]
  },
  "chain_diagnostics": {
    "page_size_bytes": 16384,
    "pointer_count": 1057030,
    "stride_bytes": 128,
    "unique_pages_touched": 65536
  },
  "samples_ns": {
    "statistics": {
      "average": 15.289083916666666,
      "max": 16.014336875,
      "median": 15.122818125,
      "min": 15.072767291666667,
      "p90": 15.668637291666668,
      "p95": 15.841487083333334,
      "p99": 15.97976691666667,
      "stddev": 0.4065809440444531
    },
    "values": [15.072767291666667, 16.014336875, 15.085409375, 15.122818125, 15.150087916666667]
  }
}
```

**Source**: Real values extracted from `results/0.53.7/MacMiniM4_benchmark.json` (5-run sample)

### TLB analysis JSON (analyze mode)

When run with `--analyze-tlb --output tlb_analysis.json`, the payload includes a dedicated `tlb_analysis` block.
The following is a structure-focused schema-version-4 illustration. It is not presented as a hardware result; the current
serializer contract and concrete deterministic values are exercised by
`JsonSchemaTest.TlbAnalysisExporterIncludesModeAndCoreCounts`. New hardware baselines remain outside this release series by
project decision, so historical 0.53.x measurements are not relabeled as 0.57.0 results:

```json
{
  "configuration": {
    "mode": "analyze_tlb",
    "schema_version": 4,
    "methodology_version": "page-native-paired-adaptive-validated-v4",
    "runtime_profile": "standard",
    "adaptive_rounds": {
      "minimum": 10,
      "maximum": 20,
      "ci_width_target_ns": 0.3,
      "bootstrap_resamples": 600
    },
    "access_calibration": {
      "target_duration_ns": 10000000,
      "minimum_chain_cycles": 16,
      "profile_access_cap": 2000000
    },
    "memory_budget": {
      "available_memory_mb": 4096,
      "budget_mb": 1228,
      "estimated_peak_memory_bytes": 1082130432
    },
    "buffer_lock": {
      "locked": false,
      "errno": 12,
      "error": "Cannot allocate memory",
      "policy": "best-effort; continue unlocked on failure"
    },
    "seed": "123456789",
    "seed_encoding": "uint64-decimal-string",
    "seed_source": "user",
    "seed_derivation": {
      "measurement_task": "splitmix64(splitmix64(splitmix64(base_seed xor pass) xor round_index) xor point_index)",
      "chain_layout": "splitmix64(task_seed xor layout-domain-constant)"
    },
    "main_thread_qos": {
      "requested": true,
      "requested_class": "user-interactive",
      "applied": true,
      "code": 0,
      "policy": "best-effort; continue on failure"
    },
    "schedule_policy": "seeded-cyclic-latin",
    "chain_model": "one-node-per-spread-page-with-packed-control",
    "latency_interpretation": "cache-hot pointer-chain timings; virtual locality is not the active data footprint; values are not direct DRAM latency",
    "translation_delta_definition": "same-round spread_latency_ns - packed_latency_ns",
    "boundary_signal": "translation_delta_ns",
    "changepoint_method": "paired-point-median-bootstrap",
    "confidence_interval": "deterministic-percentile-bootstrap-95",
    "minimum_effect_ns": 0.5,
    "persistence_points_required": 2,
    "independent_validation_required": true,
    "latency_stride_bytes": 64,
    "buffer_size_mb": 1024
  },
  "tlb_analysis": {
    "status": "complete",
    "planned_points": 29,
    "measured_points": 29,
    "validation_planned_points": 8,
    "validation_measured_points": 8,
    "validation_required": true,
    "validation_status": "complete",
    "validation_complete": true,
    "conclusions_valid": true,
    "planned_base_validation_pairs": 740,
    "completed_base_validation_pairs": 740,
    "completed_large_locality_pairs": 20,
    "total_completed_measurement_pairs": 760,
    "total_completed_raw_measurements": 1520,
    "sweep": [
      {
        "locality_bytes": 16384,
        "locality_kb": 16,
        "requested_pages": 1,
        "effective_pages": 1,
        "actual_pages": 1,
        "packed_actual_pages": 1,
        "pointer_nodes": 1,
        "spread_pointers_per_page_max": 1,
        "packed_pointers_per_page_max": 1,
        "actual_unique_cache_lines": 1,
        "active_cache_line_footprint_bytes": 64,
        "short_cycle_diagnostic": true,
        "refinement_source": "base",
        "spread_loop_latencies_ns": [25.957278, 25.965990, 25.916902],
        "packed_loop_latencies_ns": [20.812301, 20.901100, 20.844002],
        "translation_deltas_ns": [5.144977, 5.064890, 5.072900],
        "spread_p50_latency_ns": 25.957278,
        "packed_p50_latency_ns": 20.844002,
        "translation_delta_p50_ns": 5.072900,
        "measurements": [
          {
            "pass": "base",
            "round_index": 0,
            "order_index": 4,
            "seed": "987654321",
            "paired_control": {
              "available": true,
              "pair_order": "spread-first",
              "spread": {"seed": "101", "latency_ns": 25.957278, "chain": {"actual_pages": 1, "pointer_nodes": 1, "unique_cache_lines": 1, "integrity_verified": true}},
              "packed": {"seed": "102", "latency_ns": 20.812301, "chain": {"actual_pages": 1, "pointer_nodes": 1, "unique_cache_lines": 1, "integrity_verified": true}},
              "translation_delta_ns": 5.144977
            }
          }
        ]
      }
    ],
    "l1_tlb_detection": {
      "detected": true,
      "segment_start_index": 0,
      "boundary_index": 7,
      "boundary_locality_bytes": 4194304,
      "boundary_locality_kb": 4096,
      "bracket_lower_bytes": 3145728,
      "bracket_upper_bytes": 4194304,
      "step_ns": 2.0137,
      "persistent_jump": true,
      "overlaps_private_cache_knee": false,
      "confidence": "High",
      "discovery": {"available": true, "passed": true, "effect_ns": 2.0137, "minimum_effect_ns": 0.5, "noise_floor_ns": 0.1, "effect_ci_95_ns": {"lower": 1.82, "upper": 2.21, "paired_sample_count": 30, "bootstrap_resamples": 2000}, "persistence_points_passed": 2, "persistence_points_required": 2, "rejection_reason": ""},
      "validation": {"available": true, "passed": true, "effect_ns": 1.97, "minimum_effect_ns": 0.5, "noise_floor_ns": 0.1, "effect_ci_95_ns": {"lower": 1.76, "upper": 2.16, "paired_sample_count": 30, "bootstrap_resamples": 2000}, "persistence_points_passed": 2, "persistence_points_required": 2, "rejection_reason": ""},
      "candidates": [{"accepted": true, "boundary_index": 7, "bracket_lower_bytes": 3145728, "bracket_upper_bytes": 4194304}],
      "inferred_entries": 224,
      "inferred_entries_method": "validated-bracket-range-midpoint-estimate",
      "inferred_entries_min": 192,
      "inferred_entries_max": 256
    },
    "l2_tlb_detection": {
      "detected": true,
      "segment_start_index": 7,
      "boundary_index": 8,
      "boundary_locality_bytes": 8388608,
      "boundary_locality_kb": 8192,
      "baseline_ns": 21.85142833333333,
      "boundary_latency_ns": 35.431156666666666,
      "step_ns": 13.579728333333335,
      "step_percent": 0.6214572395992117,
      "persistent_jump": true,
      "overlaps_private_cache_knee": false,
      "confidence": "High",
      "discovery": {"available": true, "passed": true, "effect_ns": 6.58, "effect_ci_95_ns": {"lower": 6.11, "upper": 7.02, "paired_sample_count": 30, "bootstrap_resamples": 2000}, "persistence_points_passed": 2, "persistence_points_required": 2},
      "validation": {"available": true, "passed": true, "effect_ns": 6.42, "effect_ci_95_ns": {"lower": 5.98, "upper": 6.88, "paired_sample_count": 30, "bootstrap_resamples": 2000}, "persistence_points_passed": 2, "persistence_points_required": 2},
      "inferred_entries": 448,
      "inferred_entries_method": "validated-bracket-range-midpoint-estimate",
      "inferred_entries_min": 384,
      "inferred_entries_max": 512
    },
    "large_locality_paired_comparison": {
      "available": true,
      "comparison_locality_mb": 512,
      "spread_p50_ns": 100.0,
      "packed_p50_ns": 94.0,
      "translation_delta_p50_ns": 1.0,
      "translation_delta_definition": "median of same-round (spread_latency_ns - packed_latency_ns)",
      "spread_actual_pages": 32768,
      "packed_actual_pages": 128,
      "unique_cache_lines": 32768,
      "active_cache_line_footprint_bytes": 2097152,
      "pointer_nodes": 32768,
      "interpretation": "cache-hot paired translation stress; not DRAM latency and not an isolated page-table-walk cost"
    }
  }
}
```

The example abbreviates the adaptive rounds and most chain-diagnostic fields. Actual output includes
`tlb_analysis.measurement_records` in execution order and per-point `measurements`; every complete record carries both raw
pair members, pilot duration/access count, calibrated access count, physical diagnostics, pair order, and same-round delta.
`minimum_planned_base_validation_pairs` and `maximum_planned_base_validation_pairs` bound the adaptive base/validation
scheduler tasks. `completed_base_validation_pairs`, `completed_large_locality_pairs`, and
`total_completed_measurement_pairs` state their pass scope explicitly; corresponding raw-measurement counters count the
spread and packed members. `pass_summaries` records rounds, convergence, and completion reason for each executed pass.
All configuration, task, and layout `seed` values are exact uint64 decimal strings. `active_cache_line_footprint_bytes` is
`unique_cache_lines * 64`, so a
512 MiB virtual locality on 16 KiB pages has 32,768 lines and a 2 MiB active footprint. Boundary inference uses the
round-matched translation-delta matrix. Accepted and rejected candidates retain separate discovery/validation evidence and rejection reasons.

If the 512 MiB comparison cannot run, `large_locality_paired_comparison.available` is `false` and paired values are omitted.
Schema 4 contains only the current fields. The bundled plotter independently reads historical schema 1-3 files but does not
accept their field names in a schema 4 document. Each measurement
contains calibrated `paired_control.spread.access_count` and `paired_control.packed.access_count` values. When analysis is
interrupted, `conclusions_valid` is `false`, boundary objects contain a suppression reason, and no delta is published.
`validation_status: "not-run"` and `validation_complete: false` distinguish an unexecuted validation pass from
`validation_status: "not-required"` after a complete run with no validation candidates. `validation_required` is true only
after candidate-specific validation points have been planned; an interruption during the base pass can therefore report
`validation_required: false`, `validation_status: "not-run"`, and zero planned validation points even though the methodology
requires independent validation before accepting a boundary.

### Pattern keys (current)

- `sequential_forward`
- `sequential_reverse`
- `strided_64`
- `strided_4096`
- `strided_16384`
- `strided_2mb`
- `random`

Each pattern key contains methodology and workload metadata plus a `bandwidth` object. Its `read_gb_s`, `write_gb_s`,
and `copy_gb_s` entries use the pattern-schema-v2 structure shown above: explicit status/reason, headline policy,
nullable aggregate value, measured values, statistics including CV, and detailed per-loop measurements. This is not the
same structure as the standard `main_memory.bandwidth` object.

### Useful JSON inspection commands

```bash
# Pretty print
python3 -m json.tool results.json

# Main memory read median
jq '.main_memory.bandwidth.read_gb_s.statistics.median' results.json

# Main memory latency P95 from sample distribution
jq '.main_memory.latency.samples_ns.statistics.p95' results.json

# Auto TLB miss latency median (when auto comparison is active)
jq '.main_memory.latency.auto_tlb_breakdown.tlb_miss_ns.statistics.median' results.json

# Pattern random read median and status
jq '{status: .patterns.random.bandwidth.read_gb_s.status, median: .patterns.random.bandwidth.read_gb_s.statistics.median_p50}' patterns.json

# Pattern phase-count semantics, requested/effective threads, and exact totals
jq '.patterns.strided_2mb.bandwidth.read_gb_s.measurements[] | {requested_threads, effective_threads, accesses_per_pass, accesses_per_pass_semantics, min_accesses_per_pass, max_accesses_per_pass, phase_period_passes, total_accesses, total_payload_bytes}' patterns.json

# TLB L1 boundary locality (KB)
jq '.tlb_analysis.l1_tlb_detection.boundary_locality_kb' tlb_analysis.json

# Standalone TLB large-locality paired translation delta P50 (ns)
jq '.tlb_analysis.large_locality_paired_comparison.translation_delta_p50_ns' tlb_analysis.json
```

---

## Visualization Scripts

### `script-examples/latency_test_script.sh`

What it does:

- Sweeps multiple custom cache sizes
- Sweeps multiple `--latency-tlb-locality-kb` values
- Writes per-run JSON files under `script-examples/tmp/`
- Extracts `.cache.custom.latency.samples_ns.statistics` into `script-examples/final_output.txt`
- Clears `tmp` after extraction

Important: the script currently invokes `memory_benchmark --benchmark` from `PATH`. If you only built locally as `./memory_benchmark`, either install it to `PATH` or update `BENCHMARK_CMD` in the script.

### `script-examples/plot_cache_percentiles.py`

Input format: `final_output.txt` blocks like:

```text
TLB Locality: 16 KB, Cache Size: 32 KB
----------------------------------------
{ ... statistics json ... }
```

Usage:

```bash
python3 script-examples/plot_cache_percentiles.py script-examples/final_output.txt --metric median
```

Supported metrics:

- `median`
- `p90`
- `p95`
- `p99`
- `average`
- `min`
- `max`
- `stddev`

---

## Running Under Active System Load

If you benchmark while other macOS apps are heavily active, treat results as **contention-influenced**, not hardware peak values.

Use this process:

1. Keep your background load profile as consistent as possible across comparison runs.
2. Increase statistical depth (`--count 10` or higher, larger `--latency-samples`).
3. Compare **median/P95/P99**, not single-loop min/max.
4. Keep exact command lines identical across systems/runs.
5. Record context (apps active, external displays, power mode) with the result files.

### Reference numbers from README (Mac mini M4 sample)

From repository examples under lighter conditions, typical values are around:

- Main memory read: ~116 GB/s
- Main memory write: ~66 GB/s
- Main memory copy: ~106 GB/s

Under heavy concurrent load, expect lower throughput and higher variance than these references.
Historical pattern files from earlier methodology versions are not a schema-2 stability baseline and should not be
compared numerically with `pattern-v2-phase-calibrated-seeded` results without accounting for the methodology change.

---

## Best Practices and Pitfalls

### Best practices

- Use `caffeinate -i -d` for long runs.
- Use larger buffers (`512 MB` to `1024 MB+`) when targeting DRAM behavior.
- Use `--count > 1` and inspect percentiles.
- For cache-focused runs, prefer `--threads 1` unless testing aggregate behavior.
- For pattern comparisons, keep seed, buffer, requested threads, count, and iteration/calibration policy identical.
- Prefer the pattern-mode P-core default for repeatability; request all cores explicitly only when that is the workload
  being studied.
- Inspect status, effective threads, exact payload/work metadata, median, and CV together.

### Common pitfalls

- **Small buffers for DRAM claims**: often cache-dominated.
- **Assuming `--non-cacheable` is true uncached memory**: it is only a hint.
- **Comparing runs with different parameters**: invalidates conclusions.
- **Interpreting global-random and locality-window latency as identical tests**: chain construction differs intentionally.
- **Treating effective payload GB/s as physical bus traffic**: copy counts logical read+write payload, and hardware may
  transfer or cache data differently.
- **Calling `strided_2mb` a superpage test**: the stride is 2 MiB, but physical large-page backing is not verified.
- **Inferring prefetch, cache-thrash, or TLB diagnoses from pattern ratios alone**: use controlled follow-up experiments;
  use `--analyze-tlb` for the supported TLB analysis.

---

## Troubleshooting

### "Incompatible flags" errors

Check mode combinations in [Mode Compatibility](#mode-compatibility).

### `--latency-tlb-locality-kb` rejected

Use `0` or a value that is an exact multiple of system page size.

### `--latency-chain-mode` rejected

Use one of: `auto`, `global-random`, `random-box`, `same-random-in-box`, `diff-random-in-box`.
If using a box mode (`random-box`, `same-random-in-box`, `diff-random-in-box`), also set `--latency-tlb-locality-kb` to a non-zero page-multiple value.

### Buffer size warnings/capping

The tool caps per-buffer size against memory safety limits. Use the printed configuration summary to see actual applied sizes.

### Script cannot find benchmark binary

`script-examples/latency_test_script.sh` calls `memory_benchmark` from `PATH`. Install the binary or adjust `BENCHMARK_CMD`.

### Plot script says no blocks found

Make sure you are passing `script-examples/final_output.txt` generated by the latency sweep script.

---

## Additional Resources

- [README.md](README.md) - project overview, install, examples
- [CAPABILITIES.md](CAPABILITIES.md) - measurement capability overview and interpretation notes
- [LATENCY_WHITEPAPER.md](LATENCY_WHITEPAPER.md) - pointer-chase latency methodology deep dive
- [TLB_ANALYSIS_WHITEPAPER.md](TLB_ANALYSIS_WHITEPAPER.md) - TLB analysis methodology and JSON schema
- [CORE_TO_CORE_WHITEPAPER.md](CORE_TO_CORE_WHITEPAPER.md) - Core-to-Core Cache-Line Handoff Latency Benchmark: methodology, assembly protocol, scheduler-hint scenarios, and JSON contract
- [TECHNICAL_SPECIFICATION.md](TECHNICAL_SPECIFICATION.md) - architecture and implementation details
- [CHANGELOG.md](CHANGELOG.md) - release history

Repository sample result files:

- `results/0.53.7/MacMiniM4_benchmark.json`
- `results/0.53.7/MacMiniM4_patterns.json`
- `results/0.53.7/MacMiniM4_core2core.json`
- `results/0.53.8/MacMiniM4_analyze-tlb-chain-mode-random-box.json`

Command help:

```bash
memory_benchmark -h
```

---

**Last Updated**: 2026-07-10

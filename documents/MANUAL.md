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

- Effective CPU read/write/copy payload bandwidth for cache-sized and main-memory-sized working sets
- Dependent pointer-chase latency for cache-sized and large working sets
- Memory access pattern analysis (sequential/strided/random)
- Standalone paired TLB analysis
- Standalone core-to-core cache-line handoff latency analysis
- Standalone Metal GPU memory read/write/copy bandwidth
- Standalone synthetic CPU LLM decode and full-prompt prefill memory profiling
- Cartesian parameter sweeps for supported benchmark modes

Target platform is **macOS on Apple Silicon**.

This manual focuses on practical usage and interpretation. For implementation details and latency methodology internals, see:

- [README.md](../README.md)
- [CAPABILITIES.md](CAPABILITIES.md)
- [TECHNICAL_SPECIFICATION.md](TECHNICAL_SPECIFICATION.md)
- [LATENCY_WHITEPAPER.md](LATENCY_WHITEPAPER.md)
- [CORE_TO_CORE_WHITEPAPER.md](CORE_TO_CORE_WHITEPAPER.md)
- [GPU_BANDWIDTH_WHITEPAPER.md](GPU_BANDWIDTH_WHITEPAPER.md)
- [LLM_MEMORY_PROFILE_WHITEPAPER.md](LLM_MEMORY_PROFILE_WHITEPAPER.md)

---

## Quick Start

### Prerequisites

- Apple Silicon Mac
- Xcode Command Line Tools
- GoogleTest for C++ tests; Python 3 for the script-example entry test in the aggregate `make test-all` gate. `jq` is
  optional for JSON inspection and the jq-backed latency-script path.
- For `--gpu-bandwidth`: a unified-memory Metal device supporting `MTLGPUFamilyApple7` or a compatible later family.
  Capability support is distinct from a controlled performance-validation cohort.

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

Test and coverage targets:

```bash
make test                 # deterministic unit suite
make test-script-examples # current script-example JSON entry paths
make test-integration     # real Apple Silicon/CLI workflows
make test-all             # all GTest cases, then the focused script-example entry test
make coverage-unit        # isolated LLVM report under /tmp
make coverage-all
```

Coverage reports are written to `/tmp/membenchmark-coverage-{unit,all}/report.txt`. The denominator contains
production C++ and Objective-C++ only and excludes tests, GoogleTest, the bundled JSON header, generated files, and
assembly. The macOS 11.0 build links the system Metal and Foundation frameworks. GPU kernels are embedded MSL 2.3
source compiled at runtime; the optional offline Metal Toolchain is not required.

`make test-all` requires Python 3 for its script-example entry test. It does not require `jq`.

### First run

Running with no arguments shows help:

```bash
memory_benchmark
```

To run the standard memory benchmark:

```bash
memory_benchmark --benchmark
```

To run the standalone GPU bandwidth suite:

```bash
memory_benchmark --gpu-bandwidth
```

To run a bounded fixed-work synthetic LLM decode-memory profile:

```bash
memory_benchmark --llm-memory --weight-size-mb 64 --layers 4 \
  --query-heads 8 --kv-heads 2 --head-dim 64 --context-tokens 512 \
  --iterations 1 --count 3
```

To run the same decode geometry with 16-token physical KV blocks:

```bash
memory_benchmark --llm-memory --weight-size-mb 64 --layers 4 \
  --query-heads 8 --kv-heads 2 --head-dim 64 --context-tokens 512 \
  --kv-layout paged --kv-block-tokens 16 --iterations 1 --count 3 --seed 42
```

To run contiguous full-prompt prefill with 64-token query tiles:

```bash
memory_benchmark --llm-memory --weight-size-mb 64 --layers 4 \
  --query-heads 8 --kv-heads 2 --head-dim 64 --phase prefill \
  --prompt-tokens 512 --attention-query-tile-tokens 64 \
  --iterations 1 --count 3
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

- **Bandwidth (GB/s)**: effective workload payload divided by measured time, not directly observed physical traffic
- **Latency (ns)**: per-access delay, measured using dependent pointer chasing

Both matter: some workloads are throughput-bound, others are access-latency-bound.

### GPU effective memory bandwidth

GPU mode measures the versioned Metal compute kernels' effective read, write, and copy payload rate. It uses two
suite-resident private/tracked data buffers in Apple Silicon unified memory plus a small shared/tracked status buffer.
`MTLStorageModePrivate` limits CPU access to the Metal resource; it does not mean a discrete VRAM allocation.

The exact numerator is:

| Operation | Exact payload per pass |
|---|---:|
| Read | `buffer_size_bytes` |
| Write | `buffer_size_bytes` |
| Copy | `2 × buffer_size_bytes` |

The reported decimal GB/s is `exact_payload_bytes / gpu_elapsed_seconds / 1e9`. Copy ping-pongs A→B and B→A and is
aggregate read-plus-write throughput. It is not a CPU↔GPU transfer measurement. The primary duration is one completed
Metal command buffer's `GPUEndTime - GPUStartTime`, not host wall time.

GPU samples have steady-state warm-memory semantics. Initialization, one same-shape warmup, deterministic
preconditioning, and validation are excluded from the primary time. GPU caches are not flushed, and the implementation
cannot prove physical traffic: JSON always records `dram_residency: "unverified"`. A 64 MB or larger private buffer may
still be cache- or dispatch-dominated. CPU and GPU GB/s values are not directly comparable because their kernels, timing
boundaries, parallelism, cache behavior, resource modes, and validation overhead differ.

### Synthetic LLM memory payload

LLM-memory schema 1 uses generic backend/phase/layout/work-unit vocabulary. This revision activates CPU/decode with
contiguous or paged KV and CPU/prefill with contiguous KV. Prefill uses `work_unit_kind: "prefill_operation"`;
decode uses `"decode_step"`. Metal and paged prefill remain unavailable and never fall back to another profile.

`--kv-layout` defaults to `contiguous`. Paged layout requires exactly one `--kv-block-tokens <G>` option. `G` must be
positive, a power of two, and no greater than `UINT32_MAX`; it may be larger than the active phase's sequence length.
Contiguous layout
rejects `--kv-block-tokens`, so a block size is never silently ignored or defaulted.

The active profile measures three memory-only scenarios for one explicitly supplied phase geometry. `weights_only`
reads all active weights once per work unit. `kv_only` performs the phase-specific K/V writes and reads. `mixed`
performs both kinds of work in worker-local layer order inside one synchronized timing interval. In decode, the K/V
schedule appends the current token's K and V records and reads the complete fixed visible history; prefill's
full-prompt population and tiled-prefix schedule is defined separately below.

For decode, let `W` be active-weight bytes per decode step, `L` the layer count, `h_kv` the KV-head count, `d_h` the
head dimension, `s_kv` the KV element width, `B` the batch size, and `A` the fixed visible context including the
current token. Define `R = h_kv * d_h * s_kv`, the size of one K or V token record in one layer, and:

```text
K = L * 2 * h_kv * d_h * s_kv
KV read / decode work unit   = B * A * K
KV append / decode work unit = B * K
```

The effective model payloads per work unit are `W`, `B*A*K + B*K`, and `W + B*A*K + B*K` for weights-only, KV-only,
and mixed. Reported GB/s divides those exact logical W/K/V bytes by synchronized CPU time. The traffic crossover
`W / (B*K)` and the
versioned current-context classification compare weight bytes with KV-read bytes only; exact equality is
`near_crossover`. They do not locate a measured hardware bottleneck.

Contiguous K/V uses `[layer][batch][token][kv_head][head_dimension]` order and has equal logical and physical lengths.
Paged K/V uses complete physical blocks and one shared uint32 block table. For block size `G`:

```text
N = ceil(A / G)
physical blocks per layer = B * N
block bytes = G * R
last block tokens = A - (N - 1) * G
last block valid bytes = last block tokens * R
K logical bytes = L * B * A * R
K physical bytes = L * B * N * block bytes
K layout padding bytes = K physical bytes - K logical bytes
block table entries = B * N
block table bytes = 4 * block table entries
```

V has the same byte counts. Each batch sequence's final physical block is allocated in full. Its unused suffix is
initialized, pre-touched, protected by a canary, and excluded from timed work and effective model payload. The table is
a bijection over physical IDs `0 .. B*N-1`. A versioned SplitMix64/Fisher-Yates rejection shuffle derives it once from
the command seed. Entries are four-byte uint32 values, `UINT32_MAX` is reserved as an invalid sentinel, and `B*N` may
not exceed `UINT32_MAX`. The table's domain, resolved seed, algorithm, entry count, and little-endian SHA-256 are frozen
for the whole command. Warmup, calibration, and every scenario/loop use the same table.

Paged `weights_only` does not touch the table. `kv_only` and `mixed` perform one paired append lookup, `N` K-scan
lookups, and `N` V-scan lookups for every layer/batch pair:

```text
lookups / decode work unit = L * B * (2 * N + 1)
metadata bytes / decode work unit = 4 * lookups
accounted bytes = effective model payload bytes + metadata bytes
```

Each lookup is an explicit uint32 load inside timed assembly, followed by physical-address calculation. Metadata bytes
count toward the 64 GiB task guardrail but not the GB/s numerator. The paged checksum binds logical table index, loaded
physical ID, append/K/V visit kind, and work-unit ordinal; physical initialization also depends on pool, physical ID,
and physical offset. Post-task validation checks current-token writes and padding canaries without adding to elapsed
time.

For prefill, prompt length `P` and query-tile length `Q` are explicit. Define `C = ceil(P/Q)`, tile ends
`e_j = min((j+1)*Q, P)`, and `S(P,Q) = sum(e_j)`. One full-prompt work unit has:

```text
KV write / prefill operation = B * P * K
KV read / prefill operation  = B * S(P,Q) * K
```

Prefill payload is `W`, `B*(P+S(P,Q))*K`, or `W+B*(P+S(P,Q))*K` for weights-only, KV-only, or mixed. Each owner writes
its prompt tokens in ascending order, K then V for each token, before that owner's reads; no global worker barrier is
implied. Each increasing tile then reads the complete owned K prefix followed by the complete owned V prefix. `Q=P`
scans one full prefix; `Q=1` gives `S=triangular(P)`. Reported causal-pair and logical attention/FMA counts are audit
metadata, not executed compute. The timed checksum covers every tile-read visit. Excluded post-validation checks each
owner's deterministic first/middle/last canonical-word samples, including bytes clipped to owner boundaries, against
the final operation ordinal `T-1`; it does not reread every prompt record.

The command uses full-size ordinary cacheable resources. Initialization/pre-touch, permutation preparation, worker
creation, same-shape warmup, calibration, JSON, expected-checksum construction, and post-validation are outside elapsed
time. Full-size resources prevent a small proxy buffer from masquerading as a larger model, but they do not prove
physical DRAM service. A synthetic decode step or full-prompt prefill operation is not an inference token. Prefill does
not predict TTFT. The profile excludes Transformer math, model/framework dispatch, GPU/ANE work, growing context,
runtime page allocation, prefix sharing,
sliding-window KV, and compute-memory overlap.

### Memory hierarchy behavior

- L1 and L2 normally serve accesses with much lower latency than a main-memory-sized working set
- Small test buffers can become cache-dominated
- Larger working sets reduce cache dominance but do not prove that every access was served by physical DRAM
- Auto cache tests target full detected L1/L2 capacity, then apply stride/page alignment (so printed buffer sizes can be slightly smaller)

### Pointer-chase latency and TLB locality

Latency tests use dependent pointer-chase chains. `--latency-tlb-locality-kb` controls how the chain is constructed:

- `1024` (default): randomized within 1 MB windows, plus randomized window order
- `0`: fully global random chain
- when the effective chain mode uses locality, non-zero values must be multiples of system page size

`--latency-chain-mode` controls pointer-chain ordering policy:

- `auto` (default): preserves current behavior (`random-box` when locality > 0, `global-random` when locality = 0)
- `global-random`: full-buffer random permutation
- `random-box`: random order within locality boxes and random box traversal order
- `same-random-in-box`: same in-box random pattern reused across boxes (increasing box order)
- `diff-random-in-box`: independently randomized in-box pattern per box (increasing box order)

Explicit box modes require non-zero `--latency-tlb-locality-kb`; `auto` accepts zero and resolves to `global-random`.

`--latency-stride-bytes` controls spacing between chain nodes. Smaller stride biases toward same-page reuse;
larger stride increases page turnover pressure.

Use `0` when you explicitly want a global-random chain. It can increase page turnover, but it does not isolate
translation effects.

When `--latency-tlb-locality-kb` is omitted in regular benchmark mode, main-memory latency output also runs an
automatic comparison and prints:

- 16 KiB locality latency
- Global-random latency
- Locality latency delta (`global - 16 KiB`)

The comparison uses three paired rounds. Each round rebuilds both layouts from recorded derived seeds, alternates which
layout is measured first, and retains the same-round delta. The reported delta is the median of those paired deltas, not
the difference of independently grouped medians. It combines locality, cache, and translation effects and is not an
isolated page-walk cost; use `--analyze-tlb` for controlled translation-boundary conclusions.

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
of these worker payloads, while copy bandwidth counts both the read and write sides. The executor receives the finalized
worker ranges directly and does not repartition them. On every pass,
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
local index lists and finalized worker boundaries before any timed call. The timed callback uses those lists directly;
worker lookup, index filtering, and list allocation are not included in reported bandwidth.
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
must be specified at most once per command. `--sweep` is the exception: repeat it once per swept parameter, while using
each parameter key at most once.

Long options require a double dash (`--`). A single dash is reserved for one-character short options, so legacy
forms such as `-buffersize` or `-benchmark` are invalid.

Numeric values must be complete decimal tokens. Leading/trailing whitespace,
leading `+`, trailing characters, and overflow are rejected; unsigned seeds
also reject either sign. Comma-separated sweep lists reject empty leading,
middle, and trailing items.

| Short | Long |
|---|---|
| `-B` | `--benchmark` |
| `-P` | `--patterns` |
| `-W` | `--only-bandwidth` |
| `-L` | `--only-latency` |
| `-T` | `--analyze-tlb` |
| `-C` | `--analyze-core2core` |
| `-G` | `--gpu-bandwidth` |
| `-M` | `--llm-memory` |
| — | `--weight-size-mb` |
| — | `--layers` |
| — | `--query-heads` |
| — | `--kv-heads` |
| — | `--head-dim` |
| — | `--kv-element-bytes` |
| — | `--phase` |
| — | `--context-tokens` |
| — | `--prompt-tokens` |
| — | `--attention-query-tile-tokens` |
| — | `--kv-layout` |
| — | `--kv-block-tokens` |
| — | `--batch-size` |
| `-b` | `--buffer-size` |
| `-i` | `--iterations` |
| `-r` | `--count` |
| — | `--seed` |
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
- Standard/pattern paths apply their documented memory safety size rules
- `--buffer-size 0` is valid only with `--only-latency` and disables main-memory latency path
- In GPU mode, this is the exact size of each of two private buffers, the hard minimum is `64` MB, and the requested
  value is rejected rather than silently reduced if it exceeds the Metal or suite memory budget

#### `--iterations <count>`

- Exact measured bandwidth pass/dispatch count, or exact LLM scenario work-unit count, when explicitly supplied
- Positive integer
- Not allowed with `--only-latency`
- In `--benchmark`, `--patterns`, `--gpu-bandwidth`, and `--llm-memory`, omission enables excluded
  operation- or scenario-specific work and automatic calibration toward 150 ms (100–250 ms intended window, at most
  two corrections)
- Standard calibration is target- and operation-specific and its resolved pass count is reused across `--count` loops
- An explicit value bypasses pilot/corrections but not the operation-specific warmup
- In GPU mode, the value is an exact full-buffer dispatch count. It must fit both the 16,384-dispatch limit and the
  64 GiB exact-payload limit; copy's 2× numerator defines the strict shared CLI ceiling
- In LLM mode, the value is an exact decode-step or full-prompt prefill-operation count per scenario. Paged KV
  table-read bytes are included in the 64 GiB task-accounted-byte ceiling even though they are excluded from
  effective-model-payload GB/s

#### `--count <count>`

- Full benchmark loop count
- General default: `1`
- Core-to-core default: `3`; this mode-specific default does not change standard or pattern execution
- GPU-bandwidth default: `3`; its operation order rotates so one complete three-loop block gives read/write/copy one
  first, middle, and last position each
- LLM-memory default: `3`; its planned weights-only/KV-only/mixed order rotates across one complete three-loop block
- Positive integer
- Use `5` to `10` for stable statistics

#### `--threads <count>`

- Thread count for bandwidth tests
- Main-memory bandwidth and `--patterns` default to the count of all detected CPU cores
- Standard cache bandwidth defaults to one worker when `--threads` is omitted; an explicit `--threads` value applies
  to both main-memory and cache bandwidth
- If above available cores, it is capped
- To reproduce a worker-count profile matching the detected P-core count, set that count explicitly with `--threads`;
  this does not pin those workers to P-cores
- In pattern mode this is a requested count; sparse strided work may use fewer effective workers so each worker performs
  at least one genuine stride transition
- Pattern workers request best-effort macOS QoS but are not pinned to specific cores
- LLM-memory defaults its requested count to detected CPU workers. An explicit request remains distinct from detected
  availability; its immutable work plan records requested, available, and effective counts, reducing effective workers
  only when detected availability or executable span size requires it
- Latency tests remain single-threaded

### Mode selection

#### `--benchmark`

- **Required** to run standard memory benchmark (bandwidth + latency)
- Mutually exclusive with `--patterns`
- Can be combined with `--only-bandwidth`, `--only-latency`, `--cache-size`, `--threads`, and other modifier flags
- Rotates enabled phase groups and read/write/copy order in deterministic cyclic schedules across `--count` loops
- Uses continuous latency headlines calibrated toward 250 ms, accepted in a 100–300 ms window, and rounded to at least
  16 complete pointer-chain cycles. If the cycle minimum itself exceeds 300 ms, metadata reports
  `minimum-complete-cycles-exceed-window` instead of treating it as an ordinary calibration miss
- Reuses calibrated work and seeded logical chains so repeated loops vary runtime conditions rather than workload shape
- A direct command accepts `--output -` as a final machine-readable stdout target. After parsing, the banner,
  configuration, progress, results, warnings, and errors are routed to stderr; stdout contains one JSON document
- Running without any primary mode flag shows help and exits

#### `--patterns`

- Runs only access-pattern benchmarks
- Skips standard bandwidth/latency sections
- Automatically calibrates each read/write/copy sample toward 150 ms (intended window 100–250 ms) unless
  `--iterations` is explicitly supplied
- Uses the historical all-detected-CPU-core default for comparison compatibility; a matching explicit worker-count
  profile can use `--threads <detected P-core count>`, but placement remains unpinned
- Rotates pattern groups across repeated loops; read/write/copy order within each group stays fixed
- A direct command accepts `--output -` with the same one-document stdout and human-stderr stream contract as direct
  standard mode

#### `--gpu-bandwidth`

- Runs only the standalone Metal GPU read/write/copy suite; it does not enter `BenchmarkConfig`, the CPU benchmark, or
  the sweep runner
- May be combined only with `-b`/`--buffer-size`, `-i`/`--iterations`, `-r`/`--count`, `--seed`,
  `-o`/`--output`, and `-h`/`--help`; every other option and every other mode flag is rejected
- Defaults to 512 MB per private buffer, three loops, generated seed, and automatic per-operation calibration
- Requires at least 64 MB per buffer, unified memory, and `MTLGPUFamilyApple7` capability. Capability admission is not a
  throughput promise
- Allocates two `private + tracked` data buffers for the full suite and one `shared + tracked` status buffer. On Apple
  Silicon all are resources in unified system memory; private storage is not separate VRAM
- Uses one excluded warmup and deterministic precondition before each attempt. Automatic mode runs an excluded pilot,
  duration trial, and at most two corrections per operation, then freezes each operation's plan for all loops
- Uses one measured command buffer and one `MTLDispatchTypeSerial` compute encoder per attempt. Each pass is one
  full-buffer dispatch; the deterministic grid-stride plan is capped at 8192 threadgroups, and copy alternates
  source/destination by pass parity
- Uses Metal `GPUStartTime`/`GPUEndTime` after command completion. Host wall/submit/wait data is diagnostic only
- Validates the `gpu-dual-mod32-v2` timed accumulator for every operation and the separate `gpu-dual-mod32-v1` final
  checksum for write/copy. V2 uses pass-specific odd lane weights plus one independently mixed nonzero token per lane
  and dispatch, including seed, buffer size, pass, operation, and copy direction. Only a completed, validly timed,
  passed-validation attempt becomes `measured`
- Rotates operation order `read/write/copy`, `write/copy/read`, `copy/read/write`; multiple values use median P50 and
  CV above 5% produces a warning without filtering samples
- Has warm-memory/cache-inclusive semantics and always reports DRAM residency as unverified. Copy GB/s counts aggregate
  read + write payload and is not CPU↔GPU transfer bandwidth
- A direct command accepts exact `--output -` for one terminal GPU schema 1 document on stdout and routes its runtime
  transcript to stderr. Initialized unsupported or failed state remains in that payload with a non-zero process status;
  a parser/config error or backend-factory failure before result initialization leaves stdout empty
- `--gpu-bandwidth --output - --help` and the reversed help/output order remain human-facing help commands: they emit
  help to stdout, perform no Metal work, and create no `-`/`-.tmp` artifact
- Detailed methodology and GPU schema 1 contract: [GPU_BANDWIDTH_WHITEPAPER.md](GPU_BANDWIDTH_WHITEPAPER.md)

#### `--llm-memory`

- Selects the standalone CPU decode/prefill memory mode. Active methodologies are
  `llm-memory-v1-cpu-decode-contiguous`, `llm-memory-v1-cpu-decode-paged`, and
  `llm-memory-v1-cpu-prefill-contiguous`. It never enters the general
  `BenchmarkConfig` parser or CPU sweep runner
- Requires each common model option `--weight-size-mb <MiB>`, `--layers <count>`, `--query-heads <count>`,
  `--kv-heads <count>`, and `--head-dim <count>` exactly once. Decode requires `--context-tokens <count>`; prefill
  requires `--prompt-tokens <P>` and `--attention-query-tile-tokens <Q>`. Cross-phase geometry is rejected
- Allows optional `--kv-element-bytes <1|2|4>`, `--batch-size <count>`,
  `--phase <decode|prefill>`, `--kv-layout <contiguous|paged>`, `-t`/`--threads <count>`,
  `-i`/`--iterations <count>`,
  `-r`/`--count <count>`, `--seed <uint64>`, `-o`/`--output <target>`, and `-h`/`--help`.
  `--kv-block-tokens <G>` is required exactly once with paged layout and rejected with contiguous layout. Every other
  option, including all other primary modes, cache/latency modifiers, `--non-cacheable`, `--sweep`, and
  `--sweep-max-runs`, is rejected
- Defaults to decode, contiguous layout, two-byte KV elements, batch size one, three planned cyclic loops, detected CPU
  workers, automatic scenario work, and one generated non-zero base seed. There is no default paged block size. An
  explicit seed may be any unsigned 64-bit value, including zero
- Parses every integer as one complete decimal token. Model counts, sizes, worker count, iterations, and loop count
  must be positive. Query heads must be at least the KV-head count and evenly divisible by it
- Requires `P >= 1` and `1 <= Q <= P` for prefill. Requires paged block size `G` to be positive, a power of two, and at
  most `UINT32_MAX`. `G > A` is valid for decode and produces one partially used physical block per batch sequence;
  unused capacity is reported rather than silently removed
- Treats `--context-tokens` as the fixed visible context including the current synthetic token. Checked preflight
  resolves weight/KV byte geometry and ensures all three scenarios can fit the one-billion-work-unit and 64 GiB
  accounted-byte task limits; explicit iterations must fit the strictest scenario limit
- Treats one prefill work unit as one complete prompt rewrite plus tiled causal-prefix scans for every batch sequence;
  weights are read once per full-prompt operation, not once per token or tile
- Preserves an explicit worker request even when it exceeds detected availability. Effective-worker reduction belongs
  to the executable work plan and remains separately reportable
- `--llm-memory --help` succeeds without required model inputs and returns before core detection, seed generation,
  output-session creation, QoS preparation, or signal masking. Unknown, duplicate, and missing-value errors are still
  diagnosed because help does not bypass the strict whitelist pass
- Builds page-rounded, budget-admitted full-size active-weight and K/V resources. Contiguous uses the logical K/V
  lengths. Paged uses complete physical K/V blocks plus one page-rounded uint32 block-table mapping, and admits the
  permutation-validation transient before materializing the table. All CPU mappings are regular cacheable anonymous
  memory; `--non-cacheable` is outside the whitelist. Deterministic initialization and pre-touch cover every requested
  physical byte before any measured task, and the validated table is read-only during execution. Before each paged
  task, an untimed allocation-free reset restores only the mutable current-token append slots; history blocks and
  suffix-padding canaries are not rewritten
- Runs `weights_only`, `kv_only`, and layer-interleaved `mixed` scenarios. Omitted iterations trigger excluded,
  scenario-specific calibration in that canonical order with an 8 MiB minimum pilot accounted-work floor when
  guardrails permit, a 150 ms target, a 100–250 ms intended window, and at most two corrections. The initial pilot has
  one same-shape warmup; later candidates add a warmup only for the first irreducible one-work-unit confirmation. After
  all three candidates resolve, their plans freeze atomically and each frozen plan receives one canonical-order warmup
  before loop zero. Explicit iterations freeze all three exact plans first and use the same frozen-plan warmup boundary.
  Cyclic measured order gives all scenarios each position once when count is three
- Uses task-boundary completion-wins interruption. A started scenario is not polled in its hot kernel; a completed and
  checksum-valid current task stays measured, while no next task starts after the stop is observed
- Output values retain the shared raw-target syntax: empty disables JSON, exact `-` emits one final schema 1 document,
  and every other non-empty value is a file target, including `./-` and flag-shaped names. File output atomically
  checkpoints each terminal scenario measurement and command terminal; stdout performs the same logical transitions
  without intermediate serialization
- CPU prefill with contiguous KV uses a dedicated ARM64 executor and reports scenario-specific cost-balanced owner
  identities plus minimum/maximum/imbalance worker-cost evidence. Valid paged prefill is rejected before execution with
  `cpu-prefill-paged-not-yet-supported`; its pure planner does not materialize resources. Metal remains unavailable,
  with no CPU fallback
- This profile is memory-only: it performs no Transformer mathematics and does not report inference tokens/s. Its
  `synthetic_memory_work_units_per_second` and `effective_model_payload_gb_s` must not be interpreted as model
  throughput or physical DRAM-counter traffic. Paged block-table reads are timed and separately accounted but are not
  included in effective-model-payload GB/s. See
  [LLM_MEMORY_PROFILE_WHITEPAPER.md](LLM_MEMORY_PROFILE_WHITEPAPER.md)

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
- Can be combined only with optional `--output <target>`, `--latency-stride-bytes <bytes>`, `--latency-chain-mode <mode>`, `--tlb-density <low|medium|high>`, `--seed <uint64>`, `--sweep <key=...>`, and `--sweep-max-runs <count>`
- Uses latency stride from `--latency-stride-bytes` (same default as standard latency mode). Analyze-TLB stride must be pointer-aligned and must not exceed the system page size; it does not need to divide the page size. The default standard profile performs a base locality sweep of up to 15 canonical points, stride-clamped to `max(16KB, 2*stride)` up to `256MB`, and may insert page-aligned refinement points near detected knees/boundaries
- Builds a page-native spread chain with exactly one pointer node per requested page and a cache-line-dense packed control with the same node and unique-cache-line counts. Each scheduler task measures both layouts in one round, alternates pair order, and stores the same-round `spread - packed` translation delta
- Detects likely private-cache knee candidates from spread latency as a separate diagnostic and reports whether the region may interfere with interpretation; accepted L1/L2 claims still require the paired translation-delta and validation gates
- Reports the validated bracket range (`inferred_entries_min`/`inferred_entries_max`) as the primary L1/L2 result; `inferred_entries` is an explicitly secondary midpoint estimate
- Builds a round-by-point matrix from same-round `spread - packed` deltas. Acceptance requires a paired median effect of at least `0.5ns`, a deterministic percentile-bootstrap 95% CI above the measured noise floor, persistence at both following points, and the same evidence in an independent validation pass
- Retains rejected boundary candidates, their confidence intervals, persistence counts, and rejection reasons in JSON
- Plans one 512 MiB paired comparison when the analysis buffer is at least `512 MiB` and the main sweep plus any required validation completed successfully. Its spread P50, packed P50, median same-round `spread - packed` delta, spread/packed virtual-page counts, and active cache-line footprint are available only after the separate large-locality pass completes successfully with a valid summary; otherwise the object is unavailable. These are cache-hot translation-stress timings, not direct DRAM latency or an isolated page-table-walk cost
- Emits explicit `complete`, `interrupted`, `partial`, or `error` status. Boundary conclusions are suppressed unless the planned sweep completed
- A direct command accepts exact `--output -` for one schema-4 JSON document on stdout while routing its runtime report
  to stderr. A real file target receives the same payload through one atomic terminal write
- Tries `1024/512/256 MiB` buffers in descending order, selecting the largest candidate whose predicted
  buffer-plus-scratch peak fits the available-memory budget and whose allocation succeeds. If allocation fails, it tries
  the next smaller budget-safe candidate. The compact settings block reports the run identity, buffer-lock/QoS outcome,
  estimated peak versus budget, sweep plan, and rough duration. Full pointer-access and memory estimates remain in JSON
- Calibrates each spread and packed measurement from a timed pilot toward the active profile's target duration while requiring a minimum number of complete chain cycles
- Uses adaptive balanced rounds: every round measures each planned locality once in seeded cyclic-Latin order, and a pass stops after its minimum when every point's deterministic bootstrap median CI is narrow enough, or at the profile maximum
- Attempts `mlock()` as a best-effort noise reduction. Failure reports errno and its message, records the failure in JSON, and continues with the allocated buffer unlocked
- Requests `user-interactive` QoS for the main benchmark thread as a best-effort hint. Console and JSON report whether the request was applied and its return code; failure emits a warning and continues
- Rebuilds every standalone TLB pair from recorded task and layout seeds; pointer values are written in buffer-offset order and every chain is verified to visit all nodes and return to its head. The recorded page and cache-line diagnostics are virtual-page and buffer-relative quantities; the tool does not translate virtual addresses to physical addresses. Latency-chain behavior outside standalone TLB analysis remains unchanged
- A user interrupt remains a successful graceful-shutdown return when partial JSON can be written; consumers must use `status` and `conclusions_valid` rather than the process code to accept conclusions
- Parse, preflight, and early TLB setup/allocation failures can occur before a schema-valid analysis payload exists; with
  `--output -` these paths leave stdout empty. A measurement error after result initialization emits the available
  `tlb_analysis.status: "error"` payload and returns failure
- Detailed methodology and JSON contract: `TLB_ANALYSIS_WHITEPAPER.md`

#### `--tlb-density <level>`

- Applies only to `--analyze-tlb`
- Default: `medium` (`standard`)
- Accepted values: `low`, `medium`, `high`
- `low` (`quick`): up to 15 base points, no refinement pass, 7-12 rounds, 5 ms target per chain. Its console conclusions are screening estimates and explicitly advise confirmation with `medium` or `high`
- `medium` (`standard`): up to 15 base points, with refinement points added only when detected targets produce them, 10-20 rounds, 10 ms target per chain
- `high` (`exhaustive`): up to 29 base points, with refinement points added only when detected targets produce them, 15-30 rounds, 20 ms target per chain

#### `--seed <uint64>`

- Applies to `--benchmark`, `--patterns`, `--analyze-tlb`, `--gpu-bandwidth`, or `--llm-memory`
- In `--benchmark`, derives domain-separated seeds for main, L1, L2, custom, sampling, and both automatic-locality
  layouts; repeated loops rebuild equivalent logical chains and schedules
- A standard seed reproduces workload/schedule metadata, not performance values or macOS thread placement
- In `--patterns`, selects the deterministic unique/no-replacement permutation prefix of valid 32-byte-aligned random
  offsets; the same resolved seed reproduces the workload, and every `--count` loop repeats it
- In `--patterns`, one unsigned 64-bit seed is generated once for the command when omitted
- In `--analyze-tlb`, controls base/refinement/validation/large-locality round order, pointer-chain construction, and
  deterministic bootstrap resampling
- In `--analyze-tlb`, the same seed reproduces planner order, derived task seeds, and chain permutations
- In `--analyze-tlb`, one unsigned 64-bit seed is generated for the command when omitted and reused by every TLB run in
  a Cartesian sweep
- Standalone TLB JSON stores the resolved seed, source (`user` or `generated`), schedule policy, and each task seed
- In GPU mode, the base seed is generated once when omitted, recorded as an exact decimal string, and used to derive
  stable domain-separated read/write/copy operation seeds. It reproduces data/work identity, not performance
- In LLM-memory mode, one base seed is generated when omitted. The work planner derives separate weight/K/V buffer
  seeds and weights-only/KV-only/mixed scenario seeds; the executor uses them for deterministic initialization, append
  values, and checksum-observable scenario identity. They reproduce workload identity, not performance

#### `--analyze-core2core`

- Runs standalone repeated two-thread acquire/release token-exchange (cache-line handoff/ping-pong) mode only
- Reports effective protocol round-trip time, including token-loop instructions, coherence behavior, and scheduler effects;
  it does not directly observe physical cache-line migration or isolate coherence-fabric latency
- Places the timed token and startup/control state in distinct 128-byte-aligned storage blocks. This is a conservative
  interference-isolation boundary for current Apple Silicon targets, not evidence of a particular physical handoff path
- Defaults to three measured loops per scenario, so bare `--analyze-core2core` reports a median headline and CV/MAD instead of only a single-loop value
- Can be combined only with optional `--output <target>`, `--count <count>`, `--latency-samples <count>`, `--sweep count=...`, `--sweep latency-samples=...`, `--sweep-max-runs <count>`, and `--help`
- Executes three scheduler-hint scenarios: `no_affinity_hint`, `same_affinity_tag`, and `different_affinity_tags`
- Calibrates each scenario independently with an excluded 100,000-round-trip pilot after a 1,000,000-round-trip warmup
  intended to reduce pilot startup transients; that scenario's resolved plan is reused across its measured `--count`
  loops. Pilots run in fixed scenario order, measured loops create new thread pairs, and pilot hint outcomes are not
  serialized
- Targets 25 ms for the final untimed warmup, 250 ms for the continuous headline (100-300 ms intended window), and 1 ms for each separate sample window. The work cannot fall below 20,000 warmup, 1,000,000 headline, or 2,000 sample-window round trips
- Rotates scenario order across loops in cyclic Latin-square style: loop starts advance through no-affinity-hint,
  same-tag, and different-tags to spread first/last-position effects
- Reports one completed loop's window mean directly, or the median (P50) of multiple completed continuous loop headlines,
  plus the corresponding derived one-way estimate (`round_trip / 2`)
- Reports loop repeatability separately from pooled sample-window-mean statistics. Both include average, P50/P90/P95/P99,
  sample stddev, CV, MAD, min, and max; headline CV above the project's 7.5% diagnostic threshold emits a warning
  without filtering or invalidating measured values
- With the default 1,000 windows targeting about 1 ms, sampling alone targets roughly 9 seconds across three scenarios and three loops; add continuous headlines, warmups, calibration, thread setup, and scheduler overhead when estimating the bare command's runtime
- Includes per-loop order, status, elapsed-duration quality, measured pooled-sample boundaries, and per-thread QoS/affinity
  API outcomes in JSON schema 2. Invalid loops contribute no pooled samples and serialize a zero-length sample range
- Missing, interrupted, invalid, or failed measurements are unavailable/`null`, never numeric zero. Command/scenario completion metadata states whether all planned measurements completed
- A direct command accepts exact `--output -` for one schema-2 JSON document on stdout while routing its runtime report
  to stderr. A real file target receives the same payload through one atomic terminal write. Once the result state is
  initialized, interruption or measurement failure retains and emits the available audit payload
- Notes explicitly that macOS user-space cannot hard-pin exact core IDs
- Sets `affinity_hint_comparison_interpretable` only when the command completed and both workers' affinity API calls
  returned success in every measured affinity-tag loop. The field excludes QoS and calibration-pilot outcomes and does
  not prove physical placement; otherwise affinity-scenario deltas must not be treated as an affinity-policy comparison
- Detailed methodology and JSON contract: [CORE_TO_CORE_WHITEPAPER.md](CORE_TO_CORE_WHITEPAPER.md)

### Latency-specific controls

#### `--latency-samples <count>`

- Sample count per latency test
- Default: `1000`
- Positive integer
- In standard pointer-chase latency, samples are collected in a separate pass whose windows continue from the preceding window's terminal pointer
- In standard pointer-chase latency, the effective sample count is capped to that measurement's access count so every
  sample window contains at least one access
- In core-to-core mode, each sample is a separately timed handoff window calibrated toward 1 ms with a 2,000-round-trip minimum
- In both modes, sample count/granularity does not define or change the separate continuous headline calculation

#### `--latency-stride-bytes <bytes>`

- Pointer-chain stride for latency tests
- Default: `256`
- Must be `> 0`
- Must be a multiple of 8 bytes (pointer size on Apple Silicon)
- In standard latency mode, every enabled main-memory, detected-cache, or custom-cache chain and the configured locality
  window must retain at least two pointer nodes after applying the stride; the effective upper bound therefore depends
  on the smallest enabled target or configured window
- When locality is omitted, the extra automatic 16 KiB comparison needs a stride of at most `8192` bytes to form two
  nodes. A larger otherwise-valid stride makes that comparison unavailable but does not invalidate the configured target
  chains
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
- When the effective chain mode uses locality, non-zero values must be exact multiples of system page size. Explicit
  `global-random` ignores the locality window and does not apply this page-multiple constraint
- For a chain mode that uses locality windows, a non-zero window must contain at least two stride-spaced pointer nodes
- In regular benchmark mode, explicitly setting this option disables the automatic paired locality comparison
- When omitted, the paired comparison uses conservative locality terminology and does not claim an isolated page walk

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

#### `--output <target>`

- An omitted output option or an empty value disables JSON output for a direct command. For a sweep, an empty value is
  a missing/invalid required output target
- For every result-producing direct mode and the CPU modes' supported sweeps, an exact raw value of `-` selects
  machine-readable stdout. The sentinel is classified before path normalization
- A supported stdout-target command emits exactly one two-space-indented UTF-8 JSON document followed by one newline
  after orchestration reaches its terminal state. Intermediate standard, sweep, GPU, and LLM checkpoint requests are
  lazy no-ops and do not emit documents
- Post-parse human output is routed to stderr while the stdout target is active. Parse, preflight, and other pre-result
  failures leave stdout empty; `--help` remains a human-facing stdout command when that mode accepts the combination
- For result-producing modes, every other non-empty value is a file target, including `./-` and flag-shaped names such
  as `-G`. Thus `--output ./-` writes an ordinary file named `-`. Relative paths write under the current working
  directory, and parent directories are created automatically. LLM-memory accepts and retains the same raw target
  syntax in `configuration.output_file`
- Standard and sweep file targets retain atomic intermediate checkpoints. Pattern, TLB, and core-to-core direct file
  targets each receive one final atomic write. GPU file output retains its terminal-measurement/failure checkpoints and
  post-release replacement. LLM file output checkpoints after every terminal scenario measurement and once at command
  terminal. A failed LLM checkpoint is terminal and is not retried as a final file write. Use a real file when
  crash-resilient intermediate checkpoints are required
- With GPU or LLM `--output -`, every logical checkpoint transition and post-checkpoint stop observation still runs, but its
  lazy payload builder is not invoked and no intermediate document is serialized. The command boundary writes one
  terminal schema 1 document. `configuration.output_file` remains the raw string `"-"`, and captured `argv` is not
  rewritten
- GPU syntax/config errors, including a buffer below 64 MB, fail before result JSON is created. A backend-factory failure
  also precedes result initialization. Initialized capability, compilation, allocation, or work-plan failures remain
  auditable in the selected file or stdout target
- LLM parser/preflight, plan, JSON-output peak-estimation, timer-creation, memory-budget, mapping, initialization, or
  descriptor-preparation failures occur before runner-result initialization and therefore leave stdout empty. Once
  initialized, runner, backend-task, checksum, interruption, and checkpoint states remain auditable through the selected
  non-empty output target
- An observable terminal stdout serialization, write, or flush failure returns failure without changing the
  already-computed measurement state; any bytes from that failed transfer must be rejected

#### `--sweep <key=value1,value2>`

- Runs a Cartesian parameter sweep and writes one combined JSON result
- Requires a non-empty `--output <target>`; exact `-` selects one final envelope on stdout, while every other non-empty
  value is a checkpointed file target, including `./-` and flag-shaped names such as `-G`. An empty value is
  missing/invalid for a sweep
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
- `--gpu-bandwidth` and `--llm-memory` do not support `--sweep` or `--sweep-max-runs` in their schema-1 modes

#### `--sweep-max-runs <count>`

- Maximum number of generated sweep combinations
- General default: `256`; default with `--analyze-tlb`: `16`
- An explicit `--sweep-max-runs` value overrides the mode-specific default
- Prevents accidental very large Cartesian sweeps
- Every generated configuration is validated before the first run
- General and core-to-core combined output use envelope schema
  `configuration.sweep_schema_version: 1`; each `runs[].result` retains its own mode schema version
- Standard, pattern, TLB, and core-to-core file output is atomically checkpointed after every attempted run. An empty
  run plan or a stop observed before a run also checkpoints a terminal envelope without adding a `runs[]` entry or
  incrementing `attempted_runs`. Each envelope records `status`, `status_reason`, `planned_runs`, `attempted_runs`,
  `completed_runs`, and `conclusions_valid`. Stdout runs retain the same logical checkpoint cadence without invoking the
  persistence payload builder or serializing an intermediate document, and emit only the final envelope
- For standard, pattern, and TLB sweeps, every attempted run is retained with its own `status` and `status_reason`.
  `attempted_runs` counts stored entries, while `completed_runs` counts only mode-specific nested results that are
  genuinely complete. Current standard schema 3 requires nested `configuration.mode: "benchmark"`,
  `status: "complete"`, `results_complete: true`, and `conclusions_valid: true`, plus a string
  `configuration.output_file`. Nested standard schema 2 and every other standard version are unsupported. Pattern
  requires nested `status: "complete"` with `results_complete: true`; TLB requires nested
  `tlb_analysis.status: "complete"` with
  `tlb_analysis.conclusions_valid: true`. Partial, interrupted, and failed nested results never increment it. TLB's
  native `tlb_analysis.status: "error"` is mapped to
  a failed sweep attempt, and the schema-4 payload is retained without adding a nested `tlb_analysis.status_reason`
- A parameter key may appear only once in one sweep command
- Core-to-core sweeps also append and retain the latest attempted run when it is interrupted or fails; a file target
  checkpoints that update. Each entry records `status` and `status_reason`; `attempted_runs` counts those entries, while
  `completed_runs` counts only nested core-to-core results with `status: "complete"` and
  `measurements_complete: true`. Therefore `runs` can contain more entries than `completed_runs`
- Any partial, interrupted, or failed attempt stops further attempts; a pre-run interruption or checkpoint failure can
  also stop execution without adding or completing another run. The authoritative schema-1 sweep acceptance predicate
  is exactly top-level `status == "complete" && conclusions_valid == true`. Producers make
  `completed_runs == planned_runs` an invariant of such an envelope. Consumers may check that equality separately as a
  defensive consistency check, but it is not an additional completeness predicate

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

# Direct standard automation: JSON stdout and human transcript stderr
memory_benchmark --benchmark --only-bandwidth --count 5 --buffer-size 512 --output - >benchmark.json 2>benchmark.log

# Direct pattern automation
memory_benchmark --patterns --count 5 --buffer-size 512 --output - >patterns.json 2>patterns.log

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

# Sweep automation: one final envelope on stdout and the human transcript on stderr
memory_benchmark --analyze-core2core --sweep latency-samples=500,1000 --output - >core2core_sweep.json 2>core2core_sweep.log

# Standalone GPU bandwidth, automatic calibrated work
memory_benchmark --gpu-bandwidth --output gpu_bandwidth.json

# Standalone GPU automation: one final schema 1 document on stdout
memory_benchmark --gpu-bandwidth --buffer-size 512 --count 3 --seed 42 --output - >gpu_bandwidth_stdout.json 2>gpu_bandwidth.log

# Standalone GPU bandwidth, reproducible fixed work
memory_benchmark --gpu-bandwidth --buffer-size 512 --iterations 24 --count 9 --seed 123456789 --output gpu_fixed.json

# Standalone LLM memory profile, automatic scenario-specific calibration and file checkpoints
memory_benchmark --llm-memory --weight-size-mb 4096 --layers 32 --query-heads 32 --kv-heads 8 \
  --head-dim 128 --context-tokens 8192 --count 3 --seed 123456789 --output llm_memory.json

# Standalone LLM automation: reproducible fixed work and one final schema 1 document on stdout
memory_benchmark --llm-memory --weight-size-mb 64 --layers 4 --query-heads 8 --kv-heads 2 \
  --head-dim 64 --context-tokens 512 --iterations 1 --count 3 --seed 42 --output - \
  >llm_memory_stdout.json 2>llm_memory.log

# Standalone CPU decode with deterministic paged KV storage
memory_benchmark --llm-memory --weight-size-mb 64 --layers 4 --query-heads 8 --kv-heads 2 \
  --head-dim 64 --context-tokens 512 --kv-layout paged --kv-block-tokens 16 \
  --iterations 1 --count 3 --seed 42 --output llm_memory_paged.json

# Standalone CPU prefill with contiguous KV and explicit prompt/tile geometry
memory_benchmark --llm-memory --weight-size-mb 64 --layers 4 --query-heads 8 --kv-heads 2 \
  --head-dim 64 --phase prefill --prompt-tokens 512 --attention-query-tile-tokens 64 \
  --iterations 1 --count 3 --seed 42 --output llm_memory_prefill.json

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

# invalid: GPU mode accepts only its standalone whitelist
memory_benchmark --gpu-bandwidth --threads 4

# invalid: GPU mode has no schema-v1 sweep support
memory_benchmark --gpu-bandwidth --sweep buffer-size=64,128 --output gpu_sweep.json

# invalid: LLM mode accepts only its standalone whitelist
memory_benchmark --llm-memory --weight-size-mb 64 --layers 4 --query-heads 8 --kv-heads 2 \
  --head-dim 64 --context-tokens 512 --non-cacheable

# invalid: LLM schema 1 has no sweep support
memory_benchmark --llm-memory --weight-size-mb 64 --layers 4 --query-heads 8 --kv-heads 2 \
  --head-dim 64 --context-tokens 512 --sweep context-tokens=512,1024 --output llm_sweep.json

# invalid: paged layout requires an explicit block size
memory_benchmark --llm-memory --weight-size-mb 64 --layers 4 --query-heads 8 --kv-heads 2 \
  --head-dim 64 --context-tokens 512 --kv-layout paged

# invalid: contiguous layout rejects a paged-only block size
memory_benchmark --llm-memory --weight-size-mb 64 --layers 4 --query-heads 8 --kv-heads 2 \
  --head-dim 64 --context-tokens 512 --kv-layout contiguous --kv-block-tokens 16

# invalid: paged block size must be a power of two
memory_benchmark --llm-memory --weight-size-mb 64 --layers 4 --query-heads 8 --kv-heads 2 \
  --head-dim 64 --context-tokens 512 --kv-layout paged --kv-block-tokens 24

# invalid: prefill rejects decode context and requires explicit P/Q
memory_benchmark --llm-memory --weight-size-mb 64 --layers 4 --query-heads 8 --kv-heads 2 \
  --head-dim 64 --phase prefill --context-tokens 512

# not yet supported: CPU prefill with paged KV never falls back to contiguous
memory_benchmark --llm-memory --weight-size-mb 64 --layers 4 --query-heads 8 --kv-heads 2 \
  --head-dim 64 --phase prefill --prompt-tokens 512 --attention-query-tile-tokens 64 \
  --kv-layout paged --kv-block-tokens 16

# invalid: multiple primary modes
memory_benchmark --gpu-bandwidth --benchmark
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

### GPU bandwidth characterization

Start with the user-facing automatic policy:

```bash
caffeinate -i -d memory_benchmark --gpu-bandwidth --buffer-size 512 --count 9 --seed 123456789 --output gpu_auto.json
```

For a strict same-work cohort, resolve and deliberately lock one valid pass count, then keep buffer size, iterations,
count, seed, hardware/GPU, macOS build, kernel source SHA-256, MSL/options, resource options, and frozen plan identity the
same. For example:

```bash
caffeinate -i -d memory_benchmark --gpu-bandwidth --buffer-size 512 --iterations 24 --count 9 --seed 123456789 --output gpu_fixed.json
```

The example is a protocol shape, not a published 24-pass performance baseline for every GPU. Require nominal thermal
state, Low Power Mode off, complete/valid schema state, and CV at or below 5% before calling one process stable. A new
macOS build or GPU is a new cohort. The repository does not infer verified DRAM traffic or a minimum GB/s from these
commands; a separate Instruments counter capture can provide audit evidence but is not the production value source.

### Synthetic LLM memory characterization

Supply active weights and attention geometry explicitly; the command has no built-in model preset. Start with automatic
scenario-specific calibration and the default balanced count:

```bash
caffeinate -i -d memory_benchmark --llm-memory --weight-size-mb 4096 --layers 32 \
  --query-heads 32 --kv-heads 8 --head-dim 128 --kv-element-bytes 2 \
  --context-tokens 8192 --batch-size 1 --count 3 --seed 123456789 --output llm_auto.json
```

For a strict same-work cohort, select an explicit decode work-unit count that fits all three scenario limits and keep
every model, worker, seed, software, hardware, and environment field matched:

```bash
caffeinate -i -d memory_benchmark --llm-memory --weight-size-mb 4096 --layers 32 \
  --query-heads 32 --kv-heads 8 --head-dim 128 --kv-element-bytes 2 \
  --context-tokens 8192 --batch-size 1 --threads 4 --iterations 1 --count 6 \
  --seed 123456789 --output llm_fixed.json
```

To characterize paged indirection, make the layout and block size explicit and retain the same seed:

```bash
caffeinate -i -d memory_benchmark --llm-memory --weight-size-mb 4096 --layers 32 \
  --query-heads 32 --kv-heads 8 --head-dim 128 --kv-element-bytes 2 \
  --context-tokens 8192 --batch-size 1 --kv-layout paged --kv-block-tokens 16 \
  --threads 4 --iterations 1 --count 6 --seed 123456789 --output llm_paged_fixed.json
```

The 4 GiB/32-layer/8-KV-head example derives 128 KiB of combined K+V per visible token. At context 8192, KV read is
1 GiB per decode work unit; the weight/KV-read payload crossover is 32768 visible tokens. Those are formula checks, not
expected performance values. Inspect the three scenario headlines separately, scenario order balance, exact work-plan
identity, checksums, CV, environment warnings, and the schema acceptance predicate before comparison. A paged cohort
must additionally match `G`, physical block geometry, permutation identity/hash, and paged component identities. Do not
combine contiguous and paged samples into one distribution.

For prefill, select P/Q explicitly and keep them fixed within a cohort:

```bash
caffeinate -i -d memory_benchmark --llm-memory --weight-size-mb 4096 --layers 32 \
  --query-heads 32 --kv-heads 8 --head-dim 128 --kv-element-bytes 2 \
  --phase prefill --prompt-tokens 8192 --attention-query-tile-tokens 128 \
  --batch-size 1 --threads 4 --iterations 1 --count 6 --seed 123456789 \
  --output llm_prefill_fixed.json
```

Do not compare different Q values as the same workload: Q changes exact prefix-read multiplicity. A prefill latency is
milliseconds per synthetic full-prompt operation, not TTFT.

### Pattern analysis

```bash
memory_benchmark --patterns --count 10 --buffer-size 512 --seed 123456789 --output patterns.json
```

Shows how effective payload bandwidth changes under different access patterns. Reuse the same explicit seed and command
line for comparisons; inspect the median, CV, requested/effective worker counts, and measurement status in the output.

### Manual pattern stability matrix

Run this matrix separately from routine unit/integration tests on an idle Apple Silicon system. Earlier matrix records
used a four-worker profile that was then called the "P-core default" because its worker count matched the four detected
P-cores on that machine. Treat it as an **explicit 4-worker profile with unpinned placement**, not as the current default
or proof that workers ran on P-cores. The commands below are recipes and do not claim new results for cells without a
separately recorded run.

| Buffer | Thread profiles | Count | Purpose |
|---|---|---:|---|
| 8 MiB | 1, explicit 4-worker profile, detected-core default | 10 | Cache-resident and sparse-stride regression |
| 64 MiB | 1, explicit 4-worker profile, detected-core default | 10 | Transition around cluster-cache-sized workloads |
| 512 MiB | 1, explicit 4-worker profile, detected-core default | 10 | Main-memory-focused pattern stability |
| 1 GiB | explicit 4-worker profile, detected-core default | 5 | Large working set and thermal/order behavior |

Use one explicit seed for the complete matrix. Omit `--threads` for the current detected-core default; use `--threads 1`
for the single-worker case and `--threads 4` for the recorded explicit 4-worker profile. The count happened to match the
detected P-core count on that machine; QoS remains best-effort and worker placement is not pinned. For example:

```bash
# Repeat for buffer sizes/thread configurations in the table.
caffeinate -i -d ./memory_benchmark --patterns --buffer-size 64 --threads 1 \
  --count 10 --seed 123456789 --output pattern-stability-64m-t1.json

# Current detected-core default: intentionally omit --threads.
caffeinate -i -d ./memory_benchmark --patterns --buffer-size 64 \
  --count 10 --seed 123456789 --output pattern-stability-64m-detected-default.json

# Explicit 4-worker profile; count matched detected P-cores, placement remained unpinned.
caffeinate -i -d ./memory_benchmark --patterns --buffer-size 64 --threads 4 \
  --count 10 --seed 123456789 --output pattern-stability-64m-workers-t4.json
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

### Regular benchmark with automatic locality comparison

```bash
memory_benchmark --benchmark --count 1
```

This prints the continuous `Average latency` headline plus the paired `16 KiB locality latency`, `Global-random
latency`, and `Locality latency delta (global - 16 KiB)` when `--latency-tlb-locality-kb` is not explicitly set.

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

Every successfully started direct mode and parameter sweep begins with one shared version, copyright, and GPL
banner before mode-specific configuration or status output. Nested runs within a sweep do not repeat it. Help and
usage diagnostics retain the separate usage preamble; preflight failures do not emit the runtime banner.

The numbered result sections below describe result-producing modes, including the standalone LLM profile.

### 1) Configuration section

Shows active settings and detected hardware.

Important fields:

- Buffer size and peak concurrent allocation estimate
- Loop and iteration configuration; standalone modes print their own sample- or round-work estimates
- Thread count
- Latency chain mode
- TLB locality setting
- Detected or custom cache sizes

### 2) Main memory bandwidth

Displayed as read/write/copy GB/s. Higher is better.

### 3) Main memory latency

Average latency in ns. Lower is better.

When `--latency-tlb-locality-kb` is not explicitly provided, this section also prints:

- `16 KiB locality latency`
- `Global-random latency`
- `Locality latency delta (global - 16 KiB)`

The locality values come from three alternating-order paired rounds and the delta is the median same-round difference.
The main `Average latency` headline remains a separate continuous pointer-chase pass. Sampling is another separate pass.

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

When the automatic locality comparison is active (you did not explicitly set `--latency-tlb-locality-kb`),
statistics also include dedicated sections for:

- `16 KiB Locality Latency (ns)`
- `Global-Random Latency (ns)`
- `Locality Latency Delta, Global - 16 KiB (ns)`

The locality delta is the median same-round difference from paired, alternating-order measurements. It is not an
isolated page-table-walk penalty; use `--analyze-tlb` for controlled translation analysis.

For noisy systems, prioritize median and P95/P99 rather than single fastest/slowest values.

Standard repeated-loop headline aggregates emit a warning when CV exceeds 7.5%. This hardware-validated threshold is
diagnostic only: the benchmark does not filter outliers or automatically invalidate the aggregate.

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
- The final report contains one compact run identifier, the refinement count when refinement points were inserted,
  completion status, L1/L2 conclusions, and a large-locality paired comparison only after its pass completed
  successfully with valid data.
- `quick` conclusions carry a visible screening-estimate note and should be confirmed with `medium` or `high` before being treated as hardware boundaries.

Displayed TLB sizes use `KiB`/`MiB`, and sub-resolution negative deltas are rendered as `0.00 ns` rather than `-0.00 ns`.
The large-locality result remains cache-hot paired translation stress, not DRAM latency or an isolated page-table-walk cost.

### 8) Standalone GPU bandwidth

`--gpu-bandwidth` prints a separate Metal section rather than the CPU main-memory table:

- Device name, private/tracked resource policy, loop count, and whether the headline is a single value or median
- Read and write effective compute payload GB/s
- Copy aggregate read-plus-write payload GB/s
- Read/write/copy CV when at least three measured values exist
- An interpretation note that DRAM residency is unverified and results can be cache/dispatch dominated

CV above 5%, non-nominal thermal/Low Power Mode state, incomplete three-position order balance, and an automatic/fixed
duration outside 100–250 ms are warnings. They do not cause performance-based retry or sample filtering. A missing or
invalid measurement is not printed as zero and remains status-bearing/null in JSON.

### 9) Synthetic LLM memory profile

`--llm-memory` prints a separate report for CPU decode (contiguous/paged) or CPU prefill (contiguous) containing:

- backend, phase/work unit, KV layout, cacheable semantics, and exact active-weight/KV-read/KV-write bytes;
- decode visible context and its two-decimal traffic crossover, or prefill P/Q/C, prefix visits, causal pairs, and
  logical attention/FMA audit counts; decode-only crossover fields are not repurposed for prefill;
- for paged layout, `G`, blocks per sequence, terminal valid bytes, physical K/V lengths, padding, table size,
  permutation identity, and timed lookup/metadata counts; the report states that metadata is excluded from the GB/s
  numerator;
- up to one measured headline per weights-only, KV-only, and mixed scenario, with phase-specific work-unit latency and
  effective model-payload GB/s; JSON uses the backend-neutral fields `synthetic_work_unit_latency_seconds`,
  `synthetic_memory_work_units_per_second`, and `effective_model_payload_gb_s`, without calling any value tokens/s;
- warnings for incomplete position balance, non-nominal environment, QoS failure, cache-dominant working sets, high
  effective-model-payload CV, or duration outside the intended window. JSON retains the complete repeatability statistics;
  the console prints the CV value only in a high-CV warning;
- an interpretation reminder that the payload is logical and memory-only, not physical DRAM traffic or model inference.

Paged suffix padding is capacity and validation evidence, not payload. A padding-canary or write/checksum mismatch makes
the affected measurement invalid and does not trigger a retry.

Missing, interrupted, invalid, or failed scenarios do not receive fabricated numeric console headlines; their exact
status, reason, and null observations remain in JSON, and command failure retains its diagnostic. The corresponding JSON
traffic classification uses version `llm-exact-weight-vs-kv-read-payload-v1`; `near_crossover` means exact equality
between weight and KV-read payload. It is not a tolerance band or a measured bottleneck diagnosis.

**Note:** Current standard schema-3 per-loop latency measurements record `chain_node_count` whether the stride was
explicit or defaulted. Standalone TLB schema 4 records buffer-relative chain diagnostics such as
requested/effective/actual
virtual-page counts, `pointer_nodes`, `unique_cache_lines`, `spread_chain`, and `packed_chain`. These fields do not
describe physical-page identities or prove physical placement. They are retained in JSON, not the compact console
report.

---

## JSON Output Format

Every result-producing direct mode and the CPU modes' supported sweeps can serialize their payload either to a real file
or, with the exact raw target `--output -`, once to stdout. The stdout transport does not wrap the result or change its
measurement schema; sweep stdout is one final envelope rather than a checkpoint stream. LLM file targets checkpoint
each terminal scenario measurement and command terminal, while LLM stdout emits one final schema 1 document. See
[API.md](API.md) for stream handling, process-status checks, current schema acceptance, and the transport support matrix.

### Standard benchmark JSON shape (schema 3)

```json
{
  "configuration": {
    "mode": "benchmark",
    "benchmark_schema_version": 3,
    "output_file": "results.json",
    "methodology_version": "benchmark-v2-calibrated-seeded-balanced",
    "benchmark_seed": "123456789",
    "bandwidth_work_policy": "automatic-duration-calibration"
  },
  "execution_time_sec": 427.5,
  "status": "complete",
  "planned_loops": 5,
  "completed_loops": 5,
  "planned_measurements": 75,
  "completed_measurements": 75,
  "results_complete": true,
  "conclusions_valid": true,
  "loops": [ ... ],
  "main_memory": { ... },
  "cache": { ... },
  "timestamp": "YYYY-MM-DDTHH:MM:SSZ",
  "version": "0.63.0"
}
```

Schema 3 stores exact uint64 seeds as decimal strings. Every per-loop measurement has status/reason, nullable value,
exact passes/accesses/payload, requested/effective workers, seed, pilot/final duration, calibration quality, and schedule
position. Only `measured` values enter aggregates. One measured loop is its own headline; multiple loop headlines use
median P50. Statistics include average, P90/P95/P99, sample standard deviation, CV, MAD, min, and max. A standard file
target is atomically checkpointed after completed loop-state changes. Schema 3 requires
`configuration.output_file` to be a string. Direct payloads preserve the raw output token there, including `"-"`, `./-`,
and flag-shaped file names; standard results nested in a sweep use an empty string because the envelope owns
persistence. With `--output -`, logical checkpoint boundaries remain active but persistence is a no-op, and the command
emits one final snapshot. Schema 3 also requires boolean `conclusions_valid`, which mirrors
`results_complete`; consumers must require `status: "complete"` and both booleans when completeness is mandatory.
Bandwidth QoS metadata includes created workers plus per-worker success/failure counts; latency carries the main-thread
outcome. These fields describe a best-effort scheduler hint, never hard core pinning.

The bundled standard-memory examples are kept compatible with the current producer. They sanity-check the current
standard result locally, including exact top-level `version: "0.63.0"` for the current producer, and read current
schema-3 metric paths directly; they are not a compatibility library. Released standard schema 2, unversioned
historical standard JSON layouts, and every other explicit standard version are intentionally unsupported inputs.

### Pattern benchmark JSON shape

```json
{
  "configuration": {
    "mode": "patterns",
    "pattern_schema_version": 3,
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
    "thread_selection_policy": "detected-core-count-default",
    "qos_policy": "best-effort-scheduler-hint-no-core-pinning"
  },
  "execution_time_sec": 7.5,
  "status": "complete",
  "status_reason": "",
  "planned_loops": 3,
  "completed_loops": 3,
  "planned_measurements": 63,
  "completed_measurements": 63,
  "results_complete": true,
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
  "timestamp": "...",
  "version": "..."
}
```

This is a structure example, not a recorded performance result; omitted patterns and operations use the same shape.
Schema 3 stores explicit measurement state. A skipped, invalid, or interrupted operation has `value_gb_s: null`, an empty
or partial value set as applicable, and a `status`/`reason`; it is not serialized as zero. Aggregate `value_gb_s` is the
single measurement for one loop or median P50 for multiple measured loops. Copy `total_payload_bytes` includes both read
and write payload. Per-loop records retain pilot/final timing, exact access/pass/payload accounting, working-set and phase
metadata, seed, requested/effective threads, native page comparison, and execution-order indexes.

Pattern schema 3 adds top-level `status`, `status_reason`, `planned_loops`, `completed_loops`, `planned_measurements`,
`completed_measurements`, and `results_complete`. Every requested loop plans 21 measurements: seven patterns times the
read, write, and copy operations. A `measured` record counts as complete only with a numeric value; an intentional
`skipped` record is also terminal. Invalid evidence or executor failure makes the loop failed. A fully completed loop is
not reclassified when interruption arrives after its final operation, but the command is interrupted if requested loops
remain. Partial, interrupted, and failed commands retain all produced loop evidence; allocation or initialization failure
can produce completion metadata without a `patterns` object. Consumers that require a complete pattern run must require
both `status: "complete"` and `results_complete: true`. Aggregate `values_gb_s`, statistics, median headlines, and
console summaries use only Complete loops. Measurements from partial, interrupted, and failed loops remain serialized
as raw evidence but cannot bias the cyclic-balanced aggregate.

`thread_selection_policy: "detected-core-count-default"` means `--threads` was omitted and the historical default
covering all detected CPU cores was used. An explicit request is recorded separately; use
`--threads <detected P-core count>` to reproduce a matching worker-count profile, not guaranteed P-core placement.
Compare pattern results only when thread-selection policy and requested/effective thread counts are compatible. In
particular, older files that called a four-worker run the "P-core default" represent that explicit four-worker profile
with unpinned placement, not the current detected-core default.

For phase-rotated strided records, `accesses_per_pass` is the phase-zero count, not an average. The minimum/maximum and
phase-period fields describe pass-to-pass variation, while `total_accesses` and `total_payload_bytes` are computed from
the complete work plan. They can differ from `accesses_per_pass * passes` and must be consumed directly.

`strided_2mb` names a 2 MiB virtual address stride. It is not evidence that macOS supplied 2 MiB physical pages:
`large_page_backing_status: "not-verified"` and `large_page_backing_verified: false` must be interpreted literally.

Pattern file and stdout transports serialize this same schema-3 payload. A file target receives the normal terminal
atomic write; `--output -` receives one terminal document after all representable complete, partial, interrupted, or
failed evidence has been retained. Command acceptance requires `status: "complete"` and `results_complete: true`, while
consuming one selected metric additionally requires that measurement's `status: "measured"` and a non-null value.

### GPU bandwidth JSON shape

GPU file and stdout output use the same separate top-level schema. It must not be sent to a standard-schema parser merely
because it contains read/write/copy values. The following valid JSON object is an abbreviated field-selection fragment
representing a complete automatic stdout run; it deliberately omits, rather than empties, the populated nested evidence
arrays:

```json
{
  "software_version": "0.63.0",
  "version": "0.63.0",
  "timestamp": "...",
  "schema_version": 1,
  "mode": "gpu_bandwidth",
  "methodology_version": "gpu-bandwidth-v1-private-runtime-single-cmdbuf-calibrated-balanced",
  "status": "complete",
  "reason_code": "complete",
  "interruption_requested": false,
  "results_complete": true,
  "conclusions_valid": true,
  "operation_order_balance_complete": true,
  "dram_residency": "unverified",
  "payload_semantics": "effective-kernel-payload-divided-by-metal-gpu-time",
  "copy_payload_semantics": "aggregate-read-plus-write",
  "configuration": {
    "buffer_size_mb": 512,
    "buffer_size_bytes": "536870912",
    "iterations": null,
    "work_policy": "automatic-calibration",
    "loop_count": 3,
    "base_seed_uint64_decimal": "123456789",
    "seed_source": "user",
    "output_file": "-",
    "argv": ["memory_benchmark", "--gpu-bandwidth", "--seed", "123456789", "--output", "-"]
  },
  "counters": {
    "planned_loops": 3,
    "attempted_loops": 3,
    "completed_loops": 3,
    "planned_measurements": 9,
    "attempted_measurements": 9,
    "terminal_measurements": 9,
    "completed_measurements": 9,
    "validated_measurements": 9
  }
}
```

An actual complete output matching those counters contains three populated operation work plans, non-empty excluded
calibration evidence for each automatically calibrated operation, nine measurement records, three loop records, and
populated read/write/copy aggregates. With explicit `--iterations`, the excluded calibration arrays are legitimately
empty. The authoritative rules are:

- Top-level run status vocabulary is `not-started`, `complete`, `partial`, `interrupted`, `failed`, `unsupported`.
  Consumers accepting a full result require `status: "complete"`, `results_complete: true`, and
  `conclusions_valid: true`.
- Measurement status vocabulary is `not-run`, `measured`, `interrupted`, `invalid`, `failed`. Only `measured` has a
  finite positive `value_gb_s`; every other status serializes it as `null` with a stable `reason_code`.
- `planned_loops` equals requested count. A loop is attempted when its first operation starts and completed only when all
  three operations are measured. `planned_measurements = planned_loops × 3`; attempted starts at operation warmup;
  terminal counts every non-`not-run` slot; completed requires a completed timed command and terminal validation;
  validated counts only measured values.
- `operation_order_balance_complete` requires all planned measurements to be validated and the completed-loop count to
  be divisible by three; it is computed independently of top-level `status`. A one- or two-loop run can have valid
  values while correctly reporting incomplete order balance. A stop first observed at the final checkpoint can leave
  order balance true even when top-level completeness is false.
- Each work plan records exact requested/effective bytes, passes, payload multiplier, bytes per pass, exact payload,
  dispatch/payload limits, seeds, 16-byte vector/tail geometry, the frozen 8192-threadgroup maximum and resolved grid
  geometry, one measured command buffer, one measured encoder, dispatch count, `gpu-dual-mod32-v2` timed identity,
  operation-specific final-checksum identity, and `gpu-work-plan-v1` identity. Large exact integers and seeds use
  decimal strings.
- `excluded_calibration_attempts` retains each pilot/trial/correction's purpose, passes, exact payload, phases, GPU time,
  validation, duration quality, and reason. Explicit iterations produce empty calibration arrays.
- Each measurement records warmup, precondition, timed and validation phase command/encoder/dispatch counts, stable
  backend errors separately from raw NSError domain/code/description, GPU start/end/elapsed and host timing diagnostics,
  expected/actual dual checksums, explicit `timed_accumulator_algorithm` and `final_checksum_algorithm` validation
  identities, resources, thermal/Low Power Mode/allocation snapshots, and kernel provenance.
- `aggregates` contains only measured values. One sample is its own headline; multiple samples use median P50. Fewer than
  three samples are `insufficient-samples`; CV above 5% is `noisy`; otherwise stability is `stable`. Values are not
  filtered or retried because of performance.
- `backend.device` records the Apple7 capability result, supported families, unified-memory flag, Metal limits, pipeline
  geometry, and available-memory source. `backend.allocation` records private/tracked A/B buffers, the shared/tracked
  status buffer, exact budget components, current-allocation snapshots, and the advisory recommended-working-set result.
- `backend.compilation` records runtime source mode, MSL 2.3, integer-only math, no preprocessor macros, kernel revision,
  exact source SHA-256, compiler identifier, SDK, deployment target, and any compiler diagnostic. A non-null runtime
  library succeeds even if a warning diagnostic exists; a nil library fails.

GPU file checkpoints use the shared atomic writer after every terminal measurement and once for auditable post-parse
pre-run failures; resource-held execution paths also retain the existing post-release replacement. Exact `--output -`
executes those logical checkpoint transitions lazily without building or serializing intermediate payloads, including
the same immediate post-checkpoint stop read, then emits one terminal schema 1 document at the command boundary. On
interruption, a started logical task is allowed to finish warmup/precondition/timing/required validation; a valid current
result remains measured. All not-started slots become `interrupted` with
`value_gb_s: null` and `reason_code: "interruption-before-task"`. A real command, timer, validation, or checkpoint error
wins over interruption. Graceful interruption returns success at the process boundary but has top-level
`status: "interrupted"` and false completeness/conclusions. A stop first observed after a terminal checkpoint may cause
at most one additional interruption checkpoint. Completion of the current task wins; completeness never does.

GPU retains the raw output token in `configuration.output_file` and captures exact `argv`; therefore a stdout payload
records `"-"` while `--output ./-` remains an ordinary file. Initialized `unsupported` or failed results are serialized
and retain their non-zero process status. A parser/config failure or backend-factory failure before result initialization
leaves stdout empty. An observable terminal stdout write or flush failure returns failure without rewriting the already
computed measurement status.

### LLM memory-profile JSON shape

LLM file and stdout output use the same separate top-level generic schema 1. It is not a standard benchmark payload and
must be classified by `mode` and `schema_version`, then by the exact backend/phase/layout/methodology identity. The old
unpublished CPU/step-specific schema-v1 shape has no compatibility aliases or fallback reader; schema 1 is intentionally
re-frozen in this generic form without a version bump.

This abbreviated structural selection shows CPU/decode/contiguous. CPU/decode/paged populates the paged-only fields;
CPU/prefill/contiguous instead populates `geometry.prefill`, uses null decode/crossover/weight-to-KV-read ratio fields,
and identifies
`llm-memory-v1-cpu-prefill-contiguous`. A real document contains complete
calibration, measurement, checksum, loop, checkpoint, environment, warning, and interpretation evidence in addition to
the required generic sections.

```json
{
  "schema_version": 1,
  "mode": "llm_memory",
  "backend": "cpu",
  "phase": "decode",
  "kv_layout": "contiguous",
  "methodology_version": "llm-memory-v1-cpu-decode-contiguous",
  "software": {
    "version": "0.63.0",
    "timestamp": "..."
  },
  "status": "complete",
  "reason_code": "complete",
  "results_complete": true,
  "conclusions_valid": true,
  "configuration": {
    "argv": ["memory_benchmark", "--llm-memory", "...", "--output", "-"],
    "resolved_sources": {
      "backend": "default",
      "phase": "default",
      "kv_layout": "default"
    }
  },
  "resolved_plan": {
    "geometry": {
      "decode": {"visible_context_tokens": 512},
      "prefill": null
    },
    "layout": {
      "kv_layout": "contiguous",
      "kv_block_tokens": null,
      "blocks_per_sequence": null,
      "physical_blocks_per_layer": null,
      "last_block_tokens": null,
      "last_block_valid_bytes": null,
      "block_table_entries": null,
      "block_table_bytes": null,
      "permutation_domain_uint64_hex": null,
      "permutation_seed_uint64_decimal": null,
      "permutation_algorithm_version": null,
      "permutation_sha256": null
    },
    "resources": {
      "weight_logical_bytes": "67108864",
      "k_logical_bytes": "524288",
      "v_logical_bytes": "524288",
      "k_physical_length_bytes": "524288",
      "v_physical_length_bytes": "524288",
      "k_layout_padding_bytes": "0",
      "v_layout_padding_bytes": "0",
      "block_table_bytes": null
    },
    "component_identities": {
      "logical_profile_version": "decode_steady_fixed_context",
      "kv_layout_version": "contiguous_layer_batch_token_head_dimension",
      "permutation_version": null,
      "backend_executor_version": "llm-cpu-executor-v1-arm64-decode-contiguous",
      "resource_abi_version": "llm-memory-descriptor-abi-v1",
      "schedule_version": "worker-local-layer-order-no-per-layer-global-barrier",
      "timer_policy_version": "synchronized-start-to-last-worker-completion-per-scenario-task",
      "buffer_pattern_version": "llm-buffer-pattern-v1",
      "write_pattern_version": "llm-kv-append-affine64-v1",
      "checksum_pattern_version": "llm-read-checksum-v1",
      "msl_revision": null,
      "msl_source_sha256": null
    }
  },
  "backend_evidence": {
    "cpu": {},
    "metal": null
  },
  "memory_budget": {},
  "calibration": {},
  "measurements": [],
  "aggregates": {},
  "interpretation": {}
}
```

For a paged result, `configuration.kv_layout` is `"paged"`, `configuration.kv_block_tokens` is the explicit integer
`G`, and `resolved_sources.kv_layout` records `"explicit"`. The `resolved_plan.layout` block populates `G`,
`blocks_per_sequence = ceil(A/G)`, `physical_blocks_per_layer = B*ceil(A/G)`, terminal-token/byte counts, uint32 table
entry/byte counts, permutation domain, resolved seed, algorithm version, and lowercase 64-hex SHA-256. The resource
block reports full physical K/V lengths and their difference from logical K/V as layout padding. These are fixed-schema
integers or canonical decimal strings according to the type rules below; they are not human-formatted sizes.

Schema 1 rules:

- Canonical selectors are `backend: cpu|metal`, `phase: decode|prefill`, `kv_layout: contiguous|paged`,
  `work_unit_kind: decode_step|prefill_operation`, and
  `kv_write_kind: none|current_token_append|full_prompt_population`. CPU/decode/contiguous, CPU/decode/paged, and
  CPU/prefill/contiguous are active; Metal and paged prefill remain unavailable.
- Methodology is always `llm-memory-v1-<backend>-<phase>-<layout>`. Active exact identities are
  `llm-memory-v1-cpu-decode-contiguous`, `llm-memory-v1-cpu-decode-paged`, and
  `llm-memory-v1-cpu-prefill-contiguous`.
- `backend_evidence.cpu.prefill` is null for decode. Contiguous prefill populates it with `cost_unit:
  "worker-cost"`, one execution identity, descriptors per scenario/worker, and scenario-specific partition/scope
  identities, decimal-string worker costs, minimum, maximum, and max-minus-min imbalance per work unit.
- Run statuses are `not_started`, `complete`, `partial`, `interrupted`, `unsupported`, and `failed`. Measurement statuses are
  `not_run`, `measured`, `interrupted`, `invalid`, and `failed`. Multiword status tokens use underscores; multiword
  reason-code and duration-quality tokens use hyphens. `unsupported` is terminal, invalid for performance acceptance,
  and never authorizes a backend fallback.
- `resolved_plan.geometry.decode` and `.prefill`, the layout-specific scalars, and `backend_evidence.cpu`/`.metal` use
  JSON null when not applicable. Applicable but absent scenario traffic is numeric zero or decimal-string `"0"`
  according to the field's fixed type; it is never the string `"not_applicable"`.
- Schema/control indexes, validated small configuration values, and planned/completed work units are integer numbers.
  Potentially large byte, capacity, block/table/lookup, token-visit, causal-pair, and FMA-term quantities are canonical
  decimal strings even when their value is zero. UInt64 seeds are canonical decimal strings. Unavailable elapsed,
  rate, ratio, and statistics values are null.
- When a backend call throws before returning evidence for a measurement or excluded task, `execution.status` is
  `unavailable`, the reason remains the runner-exception token, and missing lifecycle/QoS/checksum fields are null. A
  `not_run` measurement's top-level successful/failed QoS-worker counts are likewise null rather than zero.
- `memory_budget` reports canonical decimal-string `resource_rounding_bytes`, `transient_peak_bytes`,
  `known_owned_peak_bytes`, and `admitted_budget_bytes` separately. Immutable logical/physical resource geometry stays
  under `resolved_plan.resources`. Paged admission covers full physical K/V blocks, the table, page rounding, and the
  table-validation transient before materialization. A non-empty JSON target also reserves checked storage for all
  variable-length component/layout identities and the prefill execution, scenario, and scope identities; its preflight
  and finalized-plan estimates cover the same identity set.
- `measurements[]` preserves scenario/order/status/reason, frozen-plan identity, executor/checksum evidence, and these
  fixed generic work-accounting fields:

  ```text
  work_unit_kind, planned_work_units, completed_work_units,
  weight_read_bytes_per_work_unit, kv_read_bytes_per_work_unit,
  kv_write_bytes_per_work_unit, kv_write_kind,
  effective_model_payload_bytes_per_work_unit,
  layout_metadata_lookup_count_per_work_unit,
  layout_metadata_read_bytes_per_work_unit, accounted_bytes_per_work_unit,
  planned_effective_model_payload_bytes, completed_effective_model_payload_bytes,
  planned_layout_metadata_lookup_count, completed_layout_metadata_lookup_count,
  planned_layout_metadata_read_bytes, completed_layout_metadata_read_bytes,
  planned_task_accounted_bytes, completed_task_accounted_bytes,
  synthetic_work_unit_latency_seconds,
  synthetic_memory_work_units_per_second,
  effective_model_payload_gb_s
  ```

  Measurement checksum evidence uses phase-neutral `write_pattern_version` and `checksum_pattern_version` keys. The
  unpublished schema has no decode-specific append/read aliases.

  Contiguous records and paged `weights_only` report metadata lookup/read additions as decimal-string zero; non-weights
  decode scenarios use `kv_write_kind: "current_token_append"`; prefill KV-bearing scenarios use
  `"full_prompt_population"`. Paged KV-bearing scenarios report
  `L * B * (2 * N + 1)` table lookups and four bytes per lookup per work unit. These bytes contribute to
  `accounted_bytes_per_work_unit`, not to effective-model-payload GB/s. Derived rates are non-null only for a successful
  measured record.
- `aggregates` contains measured-only work-unit latency, work-unit rate, and effective model-payload GB/s values,
  headlines, statistics, and stability quality.
- `quality_warnings` combines runner high-CV/order tokens with evidence-backed environment, QoS, cache-dominance, and
  per-scenario duration-quality tokens; warning presence never causes measurement filtering or retry.
- Traffic classification tokens are `weight_payload_dominant`, `near_crossover`, and
  `kv_read_payload_dominant`. `near_crossover` means exact equality of weight and KV-read payload, not a tolerance band.
- The authoritative acceptance predicate requires `mode == "llm_memory"`, `schema_version == 1`, backend/phase/layout
  equal to the requested profile, `methodology_version` equal to the exact derived identity,
  `status == "complete"`, `results_complete == true`, `conclusions_valid == true`, and every planned measurement to be
  `measured`. Count one can be complete but has invalid comparative conclusions because its cyclic scenario positions
  are not balanced. Paged comparisons additionally require identical `kv_block_tokens`, physical geometry,
  permutation identity, and component identities.

LLM file output checkpoints after every terminal scenario measurement and at command terminal. Exact `--output -`
retains those logical transitions without building intermediate payloads and emits one final document. A started task
uses completion-wins interruption; remaining slots become interrupted/null. A real backend-task, timer, checksum, or
checkpoint failure wins over interruption, and a failed file checkpoint is not retried.

### Core-to-core JSON shape

Core-to-core file and stdout output use the same schema 2 payload and methodology
`core2core-v3-calibrated-balanced-auditable-128b-isolation`. This abbreviated structural excerpt shows one of three scenarios and only
its first loop record, while omitting detailed statistics/hint fields. Its counters describe the unabridged payload, and
the displayed continuous and sample arrays are complete for the illustrated three-loop scenario:

```json
{
  "configuration": {
    "mode": "analyze_core2core",
    "schema_version": 2,
    "methodology_version": "core2core-v3-calibrated-balanced-auditable-128b-isolation",
    "calibration_round_trips": 100000,
    "calibration_warmup_round_trips": 1000000,
    "warmup_target_seconds": 0.025,
    "headline_target_seconds": 0.25,
    "headline_duration_window_seconds": {"minimum": 0.1, "maximum": 0.3},
    "sample_window_target_seconds": 0.001,
    "scenario_schedule": "cyclic-latin-square-across-count-loops",
    "headline_aggregate": "median-p50",
    "repeatability_cv_warning_pct": 7.5
  },
  "core_to_core_latency": {
    "status": "complete",
    "planned_measurements": 9,
    "completed_measurements": 9,
    "measurements_complete": true,
    "scenarios": [
      {
        "name": "no_affinity_hint",
        "status": "measured",
        "planned_loops": 3,
        "completed_loops": 3,
        "work_plan": {
          "automatic_calibration": true,
          "calibration_excluded_from_results": true,
          "warmup_round_trips": 250000,
          "headline_round_trips": 2500000,
          "sample_window_round_trips": 10000
        },
        "headline_round_trip_ns": 100.0,
        "headline_statistic": "median-p50-across-completed-continuous-loops",
        "round_trip_ns": {
          "values": [99.0, 100.0, 101.0],
          "statistics": {
            "median": 100.0,
            "coefficient_of_variation_pct": 1.0,
            "median_absolute_deviation": 1.0
          }
        },
        "samples_ns": {
          "values": [100.2, 99.8, 100.1, 99.9, 100.3, 99.7],
          "statistics": {"median": 100.0}
        },
        "loop_records": [
          {
            "loop_index": 0,
            "order_position": 0,
            "status": "measured",
            "round_trip_ns": 99.0,
            "headline_elapsed_seconds": 0.2475,
            "duration_quality": "within-target-window",
            "sample_window_range": {"start_index": 0, "count": 2},
            "thread_hints": {"initiator": {}, "responder": {}}
          }
        ]
      }
    ],
    "hard_pinning_supported": false,
    "affinity_tags_are_hints": true,
    "affinity_hint_comparison_interpretable": false
  }
}
```

The numeric work-plan values above are illustrative calibrated outcomes; the actual values are scenario-specific and
bounded by the documented minimums. `round_trip_ns.values` contains only measured continuous headlines, while
`samples_ns` is a separate pooled sample-window-mean population. A measured loop's range covers exactly the values it
appended to that pool; a non-measured loop has a zero-length range and contributes neither headline nor sample values. A
loop that does not produce a valid measurement carries status/reason and `null` values instead of numeric zeros.
Consumers should require `core_to_core_latency.status: "complete"` and `measurements_complete: true` for a complete
comparison. Before interpreting affinity-tag
differences, additionally require `affinity_hint_comparison_interpretable: true`; that field covers measured affinity API
returns only, excludes QoS and pilot outcomes, and does not prove physical placement.

### Sweep JSON shape

```json
{
  "status": "complete",
  "status_reason": null,
  "planned_runs": 6,
  "attempted_runs": 6,
  "completed_runs": 6,
  "conclusions_valid": true,
  "configuration": {
    "mode": "sweep",
    "sweep_schema_version": 1,
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
      "status": "complete",
      "status_reason": null,
      "parameters": {
        "buffer-size": 256,
        "latency-stride-bytes": 64
      },
      "result": { "...": "normal benchmark, pattern, TLB, or core-to-core JSON payload" }
    }
  ],
  "execution_time_sec": 123.4,
  "timestamp": "YYYY-MM-DDTHH:MM:SSZ",
  "version": "0.63.0"
}
```

### Latency payload structure (current)

Current standard schema 3 separates per-loop continuous headlines, pooled sample-window distributions, and paired
locality comparisons. The values below illustrate structure only.

```json
"latency": {
  "headline_ns": {
    "status": "measured",
    "value": 84.2,
    "headline_semantics": "median-p50-across-loop-headlines",
    "values": [83.7, 84.2, 85.0],
    "statistics": {
      "median": 84.2,
      "coefficient_of_variation_pct": 0.8,
      "median_absolute_deviation": 0.5
    },
    "measurements": [
      {
        "status": "measured",
        "value": 83.7,
        "access_count": 33554432,
        "chain_node_count": 2097152,
        "complete_chain_cycles": 16,
        "seed": "987654321",
        "pilot_elapsed_seconds": 0.018,
        "elapsed_seconds": 0.25,
        "duration_quality": "within-target-window"
      }
    ],
    "pooled_sample_distribution": {
      "semantics": "pooled-separate-sample-window-distribution",
      "values_ns": [83.1, 84.0, 85.2],
      "loop_ranges": [
        {"benchmark_loop_index": 0, "start_index": 0, "sample_count": 3}
      ]
    }
  },
  "automatic_locality_comparison": {
    "locality_16k_latency_ns": {"value": 23.1, "measurements": []},
    "global_random_latency_ns": {"value": 91.4, "measurements": []},
    "locality_latency_delta_ns": {"value": 68.3, "measurements": []}
  }
}
```

An interrupted, skipped, invalid, or failed measurement has `value: null` and an explicit reason. The locality delta's
per-loop `samples_ns` are same-round `global - 16 KiB` differences; they must not be interpreted as isolated page walks.

### TLB analysis JSON (analyze mode)

When run directly with `--analyze-tlb --output <target>`, the payload includes a dedicated `tlb_analysis` block. A file
target and exact stdout target `-` serialize the same object.
The following is a structure-focused schema-version-4 illustration. It is not presented as a hardware result; the current
serializer contract and concrete deterministic values are exercised by
`JsonSchemaTest.TlbAnalysisExporterIncludesModeAndCoreCounts`. New hardware baselines remain outside this release series by
project decision, so historical 0.53.x measurements are not relabeled as current-release results:

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
      "estimated_peak_memory_bytes": 1091567616
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
pair members, pilot duration/access count, calibrated access count, verified virtual-page and buffer-relative cache-line
diagnostics, pair order, and same-round delta. These diagnostics do not identify physical pages.
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
Machine consumers accept conclusions only when `tlb_analysis.status` is `complete` and
`tlb_analysis.conclusions_valid` is `true`.
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
and `copy_gb_s` entries use the pattern-schema-v3 structure shown above: explicit status/reason, headline policy,
nullable aggregate value, measured values, statistics including CV, and detailed per-loop measurements. This is not the
same structure as the standard `main_memory.bandwidth` object.

### Useful JSON inspection commands

Capture any supported machine-output command before applying the schema-specific checks below:

```bash
memory_benchmark --benchmark --only-bandwidth --buffer-size 512 --count 5 --output - \
  >results.json 2>benchmark.log

memory_benchmark --gpu-bandwidth --buffer-size 512 --count 3 --seed 42 --output - \
  >gpu_bandwidth.json 2>gpu_bandwidth.log

memory_benchmark --llm-memory --weight-size-mb 64 --layers 4 \
  --query-heads 8 --kv-heads 2 --head-dim 64 --context-tokens 512 \
  --kv-layout paged --kv-block-tokens 16 \
  --iterations 1 --count 3 --seed 42 --output - \
  >llm_memory.json 2>llm_memory.log
```

```bash
# Pretty print
python3 -m json.tool results.json

# Main memory read median
jq '.main_memory.bandwidth.read_gb_s.statistics.median' results.json

# Main memory latency P95 from the pooled, separate sample pass
jq '.main_memory.latency.headline_ns.pooled_sample_distribution.statistics.p95' results.json

# Paired automatic locality delta median
jq '.main_memory.latency.automatic_locality_comparison.locality_latency_delta_ns.statistics.median' results.json

# Reject incomplete current standard output
jq -e 'select(.configuration.mode == "benchmark" and
              .configuration.benchmark_schema_version == 3 and
              (.configuration.output_file | type) == "string" and
              .status == "complete" and .results_complete == true and
              .conclusions_valid == true)' results.json

# Pattern random read median and status
jq '{status: .patterns.random.bandwidth.read_gb_s.status, median: .patterns.random.bandwidth.read_gb_s.statistics.median_p50}' patterns.json

# Reject incomplete pattern output
jq -e 'select(.configuration.pattern_schema_version == 3 and
              .status == "complete" and .results_complete == true)' patterns.json

# Pattern phase-count semantics, requested/effective threads, and exact totals
jq '.patterns.strided_2mb.bandwidth.read_gb_s.measurements[] | {requested_threads, effective_threads, accesses_per_pass, accesses_per_pass_semantics, min_accesses_per_pass, max_accesses_per_pass, phase_period_passes, total_accesses, total_payload_bytes}' patterns.json

# TLB L1 boundary locality (KB)
jq '.tlb_analysis.l1_tlb_detection.boundary_locality_kb' tlb_analysis.json

# Standalone TLB large-locality paired translation delta P50 (ns)
jq '.tlb_analysis.large_locality_paired_comparison.translation_delta_p50_ns' tlb_analysis.json

# Reject incomplete standalone TLB output
jq -e 'select(.configuration.schema_version == 4 and
              .tlb_analysis.status == "complete" and
              .tlb_analysis.conclusions_valid == true)' tlb_analysis.json

# Reject incomplete core-to-core output
jq -e 'select(.configuration.schema_version == 2 and
              .core_to_core_latency.status == "complete" and
              .core_to_core_latency.measurements_complete == true)' core2core.json

# Reject incomplete sweep envelope
jq -e 'select(.configuration.sweep_schema_version == 1 and
              .status == "complete" and .conclusions_valid == true)' sweep.json

# Optional producer-consistency check; not part of the sweep acceptance predicate
jq -e 'select(.completed_runs == .planned_runs)' sweep.json

# Reject incomplete GPU schema 1 output and inspect validated headlines
jq -e 'select(.mode == "gpu_bandwidth" and .schema_version == 1 and
              .status == "complete" and .results_complete == true and
              .conclusions_valid == true) |
       .aggregates |
       with_entries(.value = {headline_gb_s: .value.headline_gb_s,
                              sample_count: .value.sample_count,
                              stability_quality: .value.stability_quality})' gpu_bandwidth.json

# Inspect GPU exact payload, pass count, timing, and validation status
jq '.measurements[] | {operation, status, value_gb_s, passes: .work_plan.passes, exact_payload_bytes: .work_plan.exact_payload_bytes, gpu_elapsed_seconds: .timed.gpu_elapsed_seconds, validation_status: .validation.validation_status}' gpu_bandwidth.json

# Reject incomplete, identity-mismatched, or position-unbalanced active-profile LLM schema 1 output
jq -e 'select(.mode == "llm_memory" and .schema_version == 1 and
              .backend == "cpu" and .phase == "decode" and
              .kv_layout == "paged" and
              .methodology_version == "llm-memory-v1-cpu-decode-paged" and
              .resolved_plan.layout.kv_block_tokens == 16 and
              (.resolved_plan.layout.permutation_sha256 |
               type == "string" and test("^[0-9a-f]{64}$")) and
              .status == "complete" and .results_complete == true and
              .conclusions_valid == true and
              ([.measurements[] | select(.status != "measured")] | length) == 0)' llm_memory.json

# Inspect scenario status, generic work accounting, work-unit latency, and effective model-payload GB/s
jq '.measurements[] | {scenario, status, reason_code,
                      work_unit_kind,
                      planned_work_units,
                      planned_task_accounted_bytes,
                      layout_metadata_lookup_count_per_work_unit,
                      layout_metadata_read_bytes_per_work_unit,
                      synthetic_work_unit_latency_seconds,
                      effective_model_payload_gb_s,
                      checksum_valid: .checksum.checksum_valid}' llm_memory.json
```

---

## Visualization Scripts

Plotting requires Python 3 and `matplotlib`; the M4/M5 comparison script additionally requires `numpy`:

```bash
python3 -m pip install matplotlib numpy
```

The bundled standard-memory scripts are kept in lockstep with the current producer. Each performs only the local
version, completion, and field sanity checks needed by its current schema-3 metric paths. For version 0.63.0, the check
requires top-level `version: "0.63.0"`, standard mode/schema identity, complete/valid result state, and a string output
target before the selected metric path is read. These scripts are examples, not a versioned compatibility layer:
standard schema 2, unversioned historical standard JSON, other modes, and other standard versions are unsupported. The
separately governed `plot_analyzetlb.py` retains its own TLB-history policy. Standard-memory plotters do not accept GPU
schema 1.

### `script-examples/latency_test_script.sh`

What it does:

- Sweeps multiple custom cache sizes
- Sweeps multiple `--latency-tlb-locality-kb` values
- Writes per-run JSON files under `script-examples/tmp/`
- Extracts `.cache.custom.latency.headline_ns.pooled_sample_distribution.statistics` from complete current standard
  schema-3 files into `script-examples/final_output.txt`
- Uses `jq` for that local sanity check and extraction when available, with Python 3 as the fallback
- Clears `tmp` after extraction
- Returns a non-zero status after cleanup if any benchmark failed, an expected output is missing, or a current-schema
  result is incomplete or cannot be parsed

The script prefers the executable built at the repository root, falls back to `memory_benchmark` from `PATH` when that
file is unavailable, and honors an explicit `BENCHMARK_CMD=/path/to/memory_benchmark` override. Every selected producer
must emit complete current standard schema 3; incompatible output is rejected before metric extraction.

`latency_test_script_stride_tlb.sh` applies the same local-binary/override policy, uses embedded Python 3 for its local
current-result sanity check and metric extraction, retains its timestamped JSON files, and returns non-zero unless every
planned run produces one complete CSV row.

### Standard-result comparison and hierarchy plotters

Both JSON entry paths require explicit current standard schema-3 inputs:

```bash
python3 script-examples/plot_M4vsM5_benchmark_comparison.py \
  --m4-file current-m4.json --m5-file current-m5.json
python3 script-examples/plot_bechmark-memory-latency-hierarcy.py \
  --file current-standard.json
```

They read the current headline and automatic-locality `statistics[metric]` paths directly; there is no historical
`values` or `average` shape fallback.

There are no archived JSON defaults. The hierarchy plotter's explicit `--file` input may instead be a console-text
statistics file using the current producer's labels. That separate parser remains available but provides neither JSON
schema compatibility nor support for historical pre-schema label spellings.

### `script-examples/plot_cache_percentiles.py`

The repository includes a small valid `final_output.txt` example for an immediate smoke test.
`latency_test_script.sh` replaces that file with measured output.

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

### Historical Mac mini M4 sample (version 0.53.7)

The repository's historical `results/0.53.7/MacMiniM4_benchmark.json` sample reports approximately:

- Main memory read: ~116 GB/s
- Main memory write: ~66 GB/s
- Main memory copy: ~106 GB/s

Under heavy concurrent load, expect lower throughput and higher variance than this historical sample. It is an empirical
0.53.7 result, not a guaranteed current-version baseline. Historical pattern files from earlier methodology versions are
not a stability baseline for current pattern schema 3 and should not be compared numerically with
`pattern-v2-phase-calibrated-seeded` results without accounting for the methodology change.

---

## Best Practices and Pitfalls

### Best practices

- Use `caffeinate -i -d` for long runs.
- Use larger buffers (`512 MB` to `1024 MB+`) for main-memory-focused work. They reduce cache dominance but do not
  prove physical DRAM service.
- Use `--count > 1` and inspect percentiles.
- For cache-focused runs, prefer `--threads 1` unless testing aggregate behavior.
- For pattern comparisons, keep seed, buffer, requested threads, count, and iteration/calibration policy identical.
- Keep the detected-core default when preserving historical default-profile comparability. To match a prior worker count,
  request that count explicitly and label it as an unpinned worker-count profile, not a core-placement profile.
- Inspect status, effective threads, exact payload/work metadata, median, and CV together.
- For GPU comparisons, require identical mode/schema/methodology, hardware/GPU, macOS build, MSL/options, kernel source
  SHA-256, resource modes, and fixed work-plan identity. Treat automatic-policy and fixed-work cohorts separately.
- Keep GPU reference runs at nominal thermal state with Low Power Mode off and minimal competing GPU work. A separate
  counter capture is useful audit evidence, but its instrumented timing is not the production headline.
- For LLM comparisons, require identical schema plus exact backend/phase/layout/methodology and component identities,
  model and phase geometry, batch, requested/effective CPU workers, fixed or automatic policy, frozen work-plan
  identity, seed, hardware, software, and environment. Paged comparisons also require identical block size, physical
  geometry, table/permutation identity, and padding. Prefer a count divisible by three and require every planned
  measurement to be measured plus `conclusions_valid: true` for comparative conclusions.
- Size contiguous LLM experiments from `W + K_mapping + V_mapping`; size paged experiments from `W + K_physical +
  V_physical + block_table`, then include the recorded transient, auxiliary, and page-rounded peak before launch. A
  non-empty JSON target also reserves the recorded DOM/serialization peak in orchestration auxiliary bytes. Start with
  a bounded small command to verify the pipeline, then use the intended full model geometry; do not substitute a
  recycled small buffer for a large-model claim.

### Common pitfalls

- **Small buffers for main-memory claims**: often cache-dominated; buffer size alone does not establish physical DRAM
  service.
- **Assuming `--non-cacheable` is true uncached memory**: it is only a hint.
- **Comparing runs with different parameters**: invalidates conclusions.
- **Interpreting global-random and locality-window latency as identical tests**: chain construction differs intentionally.
- **Treating effective payload GB/s as physical bus traffic**: copy counts logical read+write payload, and hardware may
  transfer or cache data differently.
- **Calling `strided_2mb` a superpage test**: the stride is 2 MiB, but physical large-page backing is not verified.
- **Inferring prefetch, cache-thrash, or TLB diagnoses from pattern ratios alone**: use controlled follow-up experiments;
  use `--analyze-tlb` for the supported TLB analysis.
- **Calling GPU private storage VRAM or GPU GB/s verified DRAM bandwidth**: Apple Silicon private resources live in
  unified system memory, and schema 1 deliberately says `dram_residency: "unverified"`.
- **Treating GPU copy as one-way or CPU↔GPU bandwidth**: its exact numerator counts both the buffer read and write.
- **Comparing CPU and GPU GB/s directly**: the shared unit and 2× copy convention do not align timing, kernels, cache,
  resource mode, dispatch, or validation semantics.
- **Calling an LLM synthetic work unit a token or model throughput**: the work unit is a decode step or full-prompt
  prefill operation, while the
  profile omits Transformer compute, model/framework execution, GPU/ANE work, and compute-memory overlap.
- **Treating LLM crossover/classification as a hardware bottleneck**: it compares exact weight and KV-read logical
  payload only; equality is the complete `near_crossover` rule.
- **Adding paged table bytes to effective-model-payload GB/s**: table loads are timed and reported as metadata, but the
  numerator remains exact logical W/K/V payload.
- **Treating paged layout as a serving allocator**: the table is generated once and frozen; there is no runtime page
  allocation, prefix sharing, eviction, copy-on-write, or sliding window.
- **Splitting mixed time into independent weight and KV bandwidths**: mixed is one layer-interleaved timed workload;
  use weights-only and KV-only as the component baselines.
- **Assuming full-size LLM mappings prove DRAM traffic**: ordinary cacheable mappings and large working sets remain
  cache-inclusive, and the schema exposes no physical memory-traffic counter.

---

## Troubleshooting

### "Incompatible flags" errors

Check mode combinations in [Mode Compatibility](#mode-compatibility).

### `--latency-tlb-locality-kb` rejected

For `auto`/box modes, use `0` (which resolves `auto` to global random) or a non-zero exact multiple of system page size;
explicit box modes require the non-zero form. Explicit `global-random` ignores the locality value.

### `--latency-chain-mode` rejected

Use one of: `auto`, `global-random`, `random-box`, `same-random-in-box`, `diff-random-in-box`.
If using a box mode (`random-box`, `same-random-in-box`, `diff-random-in-box`), also set `--latency-tlb-locality-kb` to a non-zero page-multiple value.

### Buffer size warnings/capping

Standard/pattern paths apply their documented size safety rules. GPU mode never silently caps its requested private
buffer: below 64 MB, above Metal `maxBufferLength`, or above the two-buffer memory budget is rejected with a stable
reason code.

### GPU mode reports unsupported

GPU schema 1 requires `MTLCreateSystemDefaultDevice`, `hasUnifiedMemory`, and Apple7-family capability. With a real
`--output` file, a valid command writes an `unsupported` audit checkpoint even when no measurement starts; with exact
`--output -`, it emits that initialized payload once at the command boundary. Both return a non-zero process status. This
is different from a CLI validation or backend-factory error before result initialization, which does not create result
JSON and leaves stdout empty. Passing the capability check means the kernel contract is admitted; it does not make an
unvalidated device a performance baseline.

### GPU result is incomplete or interrupted

Check top-level `status`, `reason_code`, completeness booleans, counters, and each measurement's status. Graceful
interruption deliberately returns process success while schema conclusions remain false. A measurement already started
may remain valid and numeric because completion wins; not-started slots are interrupted/null. Never accept a result only
because the process exit code was zero.

### LLM mode fails before producing JSON

Parser/configuration errors, one-step payload-limit failures, work-plan/JSON-output-peak/memory-budget rejection, timer
creation, and weight/K/V mapping, initialization, or descriptor-preparation failures occur before runner-result
initialization. A stdout target is intentionally empty on these paths; use the process status and stderr reason. Check
that all six required model options are present, query heads are divisible by KV heads, and each scenario fits the
64 GiB task-accounted-byte limit. For paged layout, also require exactly one positive power-of-two
`--kv-block-tokens` value at or below `UINT32_MAX`; for contiguous layout, remove that option. Confirm that the
page-rounded weight, physical K/V, optional table, transient, and auxiliary peak fits the reported available-memory
policy.

### LLM result is incomplete, invalid, or not comparable

Check top-level backend/phase/layout/methodology, status/reason, `results_complete`, `conclusions_valid`, scenario-order
balance, counters, every
measurement's status/reason, checksum evidence, duration quality, QoS, and environment warnings. A started task may
finish after an interrupt, but remaining slots are interrupted/null. Count one may complete all three scenarios yet
remain unsuitable for balanced comparative conclusions. High CV, cache-dominant working sets, or non-nominal thermal/
power state do not turn a measurement into zero; they remain explicit quality warnings.

For paged results, also inspect the permutation identity/hash, physical K/V lengths, layout padding, table protection,
lookup/metadata counts, current-token post-validation, and padding-canary result. A different block size or permutation
is a different comparison cohort even when logical model geometry is unchanged.

If an otherwise identical command fails memory admission only after a non-empty `--output` target is added, include the
conservative JSON DOM/serialization peak in the diagnosis. It scales with planned measurement records, retained
calibration attempts, effective worker checksum trees, and their copied scenario-plan identities. For prefill it also
scales with the scenario-specific ownership-scope identity set, including layer/batch scopes; reduce the
model/count/worker demand or free memory rather than treating the reserve as measured payload.

### Script cannot find benchmark binary

The script first uses the repository-root `memory_benchmark` when it is executable, then tries `memory_benchmark` from
`PATH`. If neither is suitable, set `BENCHMARK_CMD` to an executable path. The selected binary must produce complete
current standard schema 3.

### Plot script says no blocks found

Make sure you are passing `script-examples/final_output.txt` generated by the latency sweep script.

---

## Additional Resources

- [README.md](../README.md) - project overview, install, examples
- [CAPABILITIES.md](CAPABILITIES.md) - measurement capability overview and interpretation notes
- [LATENCY_WHITEPAPER.md](LATENCY_WHITEPAPER.md) - pointer-chase latency methodology deep dive
- [TLB_ANALYSIS_WHITEPAPER.md](TLB_ANALYSIS_WHITEPAPER.md) - TLB analysis methodology and JSON schema
- [CORE_TO_CORE_WHITEPAPER.md](CORE_TO_CORE_WHITEPAPER.md) - Core-to-Core Cache-Line Handoff Latency Benchmark: methodology, assembly protocol, scheduler-hint scenarios, and JSON contract
- [GPU_BANDWIDTH_WHITEPAPER.md](GPU_BANDWIDTH_WHITEPAPER.md) - Metal GPU memory-bandwidth methodology, validation,
  schema 1, capability boundaries, and maintenance policy
- [LLM_MEMORY_PROFILE_WHITEPAPER.md](LLM_MEMORY_PROFILE_WHITEPAPER.md) - synthetic CPU decode/prefill memory formulas,
  layout, execution, schema 1, validation, and interpretation limits
- [TECHNICAL_SPECIFICATION.md](TECHNICAL_SPECIFICATION.md) - architecture and implementation details
- [CHANGELOG.md](../CHANGELOG.md) - release history

Repository sample result files:

These are retained historical records. Archived standard JSON among them is not accepted by the current bundled
standard-result consumers.

- `results/0.53.7/MacMiniM4_benchmark.json`
- `results/0.53.7/MacMiniM4_patterns.json`
- `results/0.53.7/MacMiniM4_core2core.json`
- `results/0.53.8/MacMiniM4_analyze-tlb-chain-mode-random-box.json`

Command help:

```bash
memory_benchmark -h
```

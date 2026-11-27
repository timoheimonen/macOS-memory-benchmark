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
- Standalone synthetic CPU and capability-gated Metal LLM decode/prefill profiling with both KV layouts
- Cartesian parameter sweeps for supported benchmark modes

Target platform is **macOS on Apple Silicon**.

This manual focuses on practical usage and interpretation. For implementation details and methodology internals, see:

- [README.md](../README.md)
- [CAPABILITIES.md](CAPABILITIES.md)
- [TECHNICAL_SPECIFICATION.md](TECHNICAL_SPECIFICATION.md)
- [CPU_BANDWIDTH_WHITEPAPER.md](CPU_BANDWIDTH_WHITEPAPER.md)
- [LATENCY_WHITEPAPER.md](LATENCY_WHITEPAPER.md)
- [PATTERNS_WHITEPAPER.md](PATTERNS_WHITEPAPER.md)
- [CORE_TO_CORE_WHITEPAPER.md](CORE_TO_CORE_WHITEPAPER.md)
- [GPU_BANDWIDTH_WHITEPAPER.md](GPU_BANDWIDTH_WHITEPAPER.md)
- [LLM_MEMORY_PROFILE_WHITEPAPER.md](LLM_MEMORY_PROFILE_WHITEPAPER.md)

---

## Quick Start

### Prerequisites

- Apple Silicon Mac running macOS 26 or later
- Xcode Command Line Tools
- GoogleTest for C++ tests; Python 3 for build provenance and the Python tests in the aggregate `make test-all` gate. `jq` is
  optional for JSON inspection and the jq-backed latency-script path.
- For `--gpu-bandwidth`: a unified-memory Metal device supporting `MTLGPUFamilyApple7` or a compatible later family.
  Capability support is distinct from a controlled performance-validation cohort.
- For LLM Metal: Apple7-or-later capability, unified memory, Tier 2 argument buffers, and
  `maxBufferLength >= 256 MiB`.

`GTEST_DIR` may select another GoogleTest installation. Its static archives must not require a newer macOS version than
the benchmark build's deployment target.

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
make test                  # deterministic unit suite
make test-script-examples  # current script-example JSON entry paths
make test-patterns-verifier # independent patterns artifact & mutation tests
make test-cpu-verifiers    # independent CPU bandwidth, CPU latency & patterns verifier tests
make test-llm-verifier     # independent LLM artifact contract and mutation tests
make test-integration      # real Apple Silicon/CLI workflows
make test-all              # all GTest cases, then script-example, LLM verifier, and CPU verifier tests
make coverage-unit        # isolated LLVM report under /tmp
make coverage-all
```

Coverage reports are written to `/tmp/membenchmark-coverage-{unit,all}/report.txt`. The denominator contains
production C++ and Objective-C++ only and excludes tests, GoogleTest, the bundled JSON header, generated files, and
assembly. The macOS 26.0 build links the system Metal and Foundation frameworks. GPU kernels are embedded MSL 2.3
source compiled at runtime; the optional offline Metal Toolchain is not required.

`make test-all` requires Python 3 for its script-example, LLM verifier, and CPU verifier tests. It does not require `jq`.

### First run

Running with no arguments shows help and exits `0` (passing modifier options without a primary mode flag fails with an
error instead of falling back to help):

```bash
memory_benchmark
```

To run the standalone audited CPU bandwidth benchmark:

```bash
memory_benchmark --cpu-bandwidth
```

To run the standalone audited CPU pointer-chasing latency benchmark:

```bash
memory_benchmark --cpu-latency
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
caffeinate -i -d memory_benchmark --cpu-bandwidth --count 10 --buffer-size 1024
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

LLM-memory schema 2 uses generic backend/phase/layout/work-unit vocabulary. Theoretical attention context
uses query-head count `h_q`; actual prefix-read geometry uses tile `Q`. They are different inputs. Theoretical
causal/attention/FMA overflow becomes null with `arithmetic-overflow` in `resolved_plan.model_context.prefill`;
actual byte/lookup/size overflow still rejects the workload. No FMA computation is performed. This revision activates all eight
backend/phase/layout profiles: CPU and Metal each support decode and prefill with contiguous or paged KV. Prefill uses
`work_unit_kind: "prefill_operation"`; decode uses `"decode_step"`. Metal never falls back to CPU.

All four active Metal profiles are governed by runtime capability checks.

`--kv-layout` defaults to `contiguous`. Paged layout requires `--kv-block-tokens <G>` (or `--sweep kv-block-tokens=...`,
not both). `G` must be positive, a power of two, and no greater than `UINT32_MAX`; it may be larger than the active
phase's sequence length. Contiguous layout rejects both `--kv-block-tokens` and `--sweep kv-block-tokens=...`, so a
block size is never silently ignored or defaulted.

Each selected profile measures three memory-only scenarios for one explicitly supplied phase geometry. `weights_only`
reads all active weights once per work unit. `kv_only` performs the phase-specific K/V writes and reads. `mixed`
performs both kinds of work in one timed scenario. CPU uses worker-local layer order inside one synchronized worker
interval; Metal uses its scenario-specialized grid-stride kernel inside one GPU timestamp interval. In decode, the K/V
schedule appends the current token's K and V records and reads the complete fixed visible history. In prefill, both
backends and layouts use the full-prompt population and tiled-prefix schedule defined separately below.

For decode, let `W` be active-weight bytes per decode step, `L` the layer count, `h_kv` the KV-head count, `d_h` the
head dimension, `s_kv` the KV element width, `B` the batch size, and `A` the fixed visible context including the
current token. Define `R = h_kv * d_h * s_kv`, the size of one K or V token record in one layer, and:

```text
K = L * 2 * h_kv * d_h * s_kv
KV read / decode work unit   = B * A * K
KV append / decode work unit = B * K
```

The effective model payloads per work unit are `W`, `B*A*K + B*K`, and `W + B*A*K + B*K` for weights-only, KV-only,
and mixed. Reported GB/s divides those exact logical W/K/V bytes by backend-authoritative time: synchronized worker time
for CPU or `GPUStartTime`/`GPUEndTime` for Metal. The traffic crossover
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

Paged `weights_only` does not touch the table. For paged decode, `kv_only` and `mixed` perform one paired append lookup,
`N` K-scan lookups, and `N` V-scan lookups for every layer/batch pair:

```text
lookups / decode work unit = L * B * (2 * N + 1)
metadata bytes / decode work unit = 4 * lookups
accounted bytes = effective model payload bytes + metadata bytes
```

Each lookup is an explicit uint32 load inside the timed CPU assembly or Metal MSL kernel, followed by physical-address
calculation. Metadata bytes count toward the 64 GiB task guardrail but not the GB/s numerator. The paged checksum binds
logical table index, loaded physical ID, append/K/V visit kind, and work-unit ordinal; physical initialization also
depends on pool, physical ID,
and physical offset. Post-task validation checks current-token writes and padding canaries without adding to elapsed
time.

CPU contiguous-decode checks every byte of both K/V append records for all layers and batches
against the final task-local ordinal `T-1`, after timer stop and all worker joins. A mismatch yields
`decode-post-validation-failed`: the invalid attempt is retained with null rates and excluded from
aggregates. The timed workload and checksum algorithm are unchanged. KV-write evidence applies only
to KV-bearing scenarios; schema 2 retains independent write, structure and applicable padding verdicts with the
existing sampling limits. No new memory scan or stronger fault-coverage claim follows from naming these checks. CPU paged decode weights-only checks its append slots and padding; CPU paged prefill weights-only checks padding,
not valid prompt contents.
See the [CPU final-state evidence contract](API.md#result-schemas-and-completion) for field and null semantics.


Treat a matching LLM checksum as a bounded consistency check. All four CPU weight paths have demonstrated
within-span parity collisions, and the shared Metal helper has two affine lanes that depend on the same content sum.
The Metal helper test is not a full-backend acceptance test. Separate final-state checks can reject a task whose
checksum still matches; such a task keeps null rates and stays outside aggregates. Prefill checks selected final-state
locations, not every prompt byte. CPU paged decode weights-only checks append slots and padding, while paged prefill weights-only checks padding only. Metal
weights-only does not perform those KV checks. Source hashes and generated event lists describe a build, not a GPU
runtime trace. Consult the [profile fault matrix](LLM_MEMORY_PROFILE_WHITEPAPER.md#checksum-fault-model-and-evidence-limits)
for detected mutations, conditional coverage, and blind spots; use the [API contract](API.md#result-schemas-and-completion)
for status/null/acceptance rules.

For prefill, prompt length `P` and query-tile length `Q` are explicit. Define `C = ceil(P/Q)`, tile ends
`e_j = min((j+1)*Q, P)`, and `S(P,Q) = sum(e_j)`. One full-prompt work unit has:

```text
KV write / prefill operation = B * P * K
KV read / prefill operation  = B * S(P,Q) * K
```

Prefill payload is `W`, `B*(P+S(P,Q))*K`, or `W+B*(P+S(P,Q))*K` for weights-only, KV-only, or mixed. Each owner writes
its prompt tokens in ascending order, K then V for each token, before that owner's reads; no global worker barrier is
implied. Each increasing tile then reads the complete owned K prefix followed by the complete owned V prefix. `Q=P`
scans one full prefix; `Q=1` gives `S=triangular(P)`. Reported causal-pair and logical attention/FMA counts are theoretical
model context, not executed compute or measured memory traffic. The intended checksum accumulation includes each
tile-read visit; equality alone does not establish every executed visit. For a KV-bearing CPU task, excluded
post-validation checks each owner's deterministic first/middle/last canonical-word samples, including bytes clipped to
owner boundaries, against the final operation ordinal `T-1`. For a KV-bearing Metal prefill task, it checks
representative/boundary byte locations per layer/batch sequence in both K and V against that ordinal; paged Metal
additionally validates applicable terminal padding canaries. Neither path rereads every prompt record.

For paged prefill, define `N = ceil(P/G)`, `m_j = ceil(e_j/G)`, and `M = sum(m_j)`. Each layer/batch pair performs `N`
paired K/V block lookups while populating the prompt, followed by `M` K-prefix and `M` V-prefix lookups across the
tiles:

```text
lookups / prefill work unit = L * B * (N + 2 * M)
metadata bytes / prefill work unit = 4 * lookups
accounted bytes = effective model payload bytes + metadata bytes
```

Partial tile prefixes visit only the exact valid bytes through `e_j`, including a partial terminal block. The timed
checksum binds logical table index, loaded physical ID, visit kind, and operation ordinal non-separably. Paged prefill
therefore preserves the same logical payload as contiguous prefill while reporting table-read metadata separately.
Post-validation checks final-ordinal prompt samples and the physical pools' terminal padding canaries. On Metal, the
row-major `L*B*N` owner ordinals use a deterministic cyclic threadgroup assignment for every work unit and semantic
visit. The grid reports the exact per-threadgroup accounted-byte vector, minimum, maximum, and imbalance without
claiming weighted balance; changing the threadgroup count does not duplicate lookups.

CPU uses full-size ordinary cacheable mappings; Metal uses exact-tail private/tracked segments on unified memory.
Initialization/pre-touch, same-shape warmup, calibration, JSON, expected-checksum construction, and post-validation are
outside elapsed time; permutation preparation and worker creation are additional CPU exclusions. Full-size resources
prevent a small proxy buffer from masquerading as a larger model, but they do not prove physical DRAM service. A
synthetic decode step or full-prompt prefill operation is not an inference token. Prefill does not predict TTFT. The
profile excludes Transformer math, model/framework dispatch, ANE work, GPU execution outside the defined Metal
kernels, growing context, runtime page allocation, prefix sharing, sliding-window KV, and compute-memory overlap.

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

In `--cpu-latency` mode, explicitly passing `--compare-locality` (which requires `--buffer-size > 0` and
`--latency-stride-bytes <= 8192`) runs `3` paired comparison rounds per loop before target measurements comparing a
16 KiB `random-box` chain against a `global-random` chain, and reports:

- `16KB P50` (`aggregates.locality_comparison.locality_16k_latency_ns`)
- `Global P50` (`aggregates.locality_comparison.global_random_latency_ns`)
- `Delta` (`aggregates.locality_comparison.locality_latency_delta_ns = median(global) - median(16k)`)

Each round rebuilds both layouts from recorded derived seeds and uses the frozen main-memory `complete_chain_cycles`.
The delta combines locality, cache, and translation effects and is not an isolated page-walk cost; use `--analyze-tlb`
for controlled translation-boundary conclusions. Without `--compare-locality`, `locality_comparison` is `null`.

### Pattern benchmarks

Standalone patterns mode (`-P`/`--patterns`, schema 1, `mode: "patterns"`, `methodology_version: "patterns-v1-arm64-phase-calibrated-seeded-balanced"`) measures audited effective `read`, `write`, and `copy` bandwidth across seven canonical access geometries:

- `sequential_forward`
- `sequential_reverse`
- `strided_64` (64-byte stride)
- `strided_4096` (4096-byte stride)
- `strided_16384` (16 KiB stride)
- `strided_2mb` (2 MiB virtual stride)
- `random` (32-byte-aligned per-worker unique permutation)

Pattern bandwidth is effective payload bandwidth, not inferred physical DRAM or cache-bus traffic. Each valid access processes a 32-byte payload (half of a 64-byte cache line, not a complete cache-line payload). Copy counts both the read and write sides, for 64 payload bytes per logical copy access (`exact_payload_bytes = 2 * total_accesses * 32`). Reported `bandwidth_gb_s` uses exact completed payload divided by Mach `cpu_raw` elapsed seconds (`10^9` bytes/s).

Before timing begins, persistent `src` and `dst` `mmap` buffers are allocated with `AllocatedBufferEvidence` (`madvise` `MADV_RANDOM` under `-u`, else `MADV_WILLNEED`). `src` is populated with the slot-tagged O(1) GF(2) pattern `patterns-buffer-pattern-v1` and protected read-only via `mprotect(PROT_READ)`. Before `copy` attempts (and `write` attempts where `has_untouched_dst_bytes` is true), `dst` is filled outside the timer with a unique odd `pre_op_sentinel`. Timed `read` passes compute non-cancelling per-worker and folded `patterns-read-checksum-v1` accumulators checked against O(1) sequential and 2D partial-cycle strided oracles; `write` (using `memory_write_pattern_loop_asm`, `memory_write_reverse_pattern_loop_asm`, `memory_write_strided_phased_pattern_loop_asm`, and `memory_write_random_pattern_loop_asm`) and `copy` attempts use unique odd `write_pattern` values and undergo parallel cold post-validation over 100% of touched bytes and 100% of untouched sentinel bytes (`execution.validation.checks[]` with 6 checks).

Strided plans use deterministic 64-byte-partitioned worker spans (`partition_patterns_64b`). On every pass, the ARM64 strided kernel advances the starting phase by 32 bytes modulo `stride_bytes`, so sparse tests do not repeatedly hit only phase-zero slots. Because pass-to-pass phase rotation varies per-pass access counts when `span_bytes` is not a full stride multiple, JSON `resolved_plan.operation_plans[]` records `accesses_per_pass` (phase 0), `min_accesses_per_pass`, `max_accesses_per_pass`, `completed_phase_cycles`, `total_accesses`, and `exact_payload_bytes` (while each `measurements[]` row records `completed_passes` and `completed_exact_payload_bytes`). Do not compute exact totals by multiplying `accesses_per_pass * passes`; use `total_accesses` and `exact_payload_bytes`. Stride labels are virtual byte distances, not page-size claims: `native_page_size_bytes` and `stride_equals_native_page_size` record whether a stride matches the OS page size (`16384` on Apple Silicon), and `strided_2mb` reports `large_page_backing_verified: false` / `large_page_backing_status: "not-verified"`.

Every active strided worker must perform at least two 32-byte accesses (one stride transition). If requested workers cannot satisfy that rule, `effective_workers` is reduced for that strided pattern. When `buffer_bytes < stride_bytes + 32` (for `strided_2mb`, `buffer_bytes < 2 MiB + 32 B`, such as 1 MiB or 2 MiB buffers), the pattern is published as `status: "skipped"` (`reason_code: "buffer-too-small-for-stride-transition"`) in `resolved_plan` and `aggregates` rather than failing the run, and omitted from `calibration`, `measurements[]`, and `loop_records[]`.

For `random`, selected accesses per pass are `min(total_slots, clamp(total_slots, 1000, 1_000_000))` where `total_slots = buffer_bytes / 32`, distributed proportionally across 64-byte worker spans. Each worker generates its own 32-byte-aligned offsets via `splitmix64-coprime-fisher-yates-v1` from `random_domain_seed` (with canonical `random_indices_sha256` and a `<= 8 MiB` sorted copy for untouched-sentinel checks) before calibration or timing starts.

Timing starts only after every worker completes best-effort `QOS_CLASS_USER_INTERACTIVE` setup and reaches the synchronized start gate, and stops after the last worker issues `dsb ish` and `HighResTimer::stop()` (`dsb ish; isb` + `mach_absolute_time()`). Unless `--iterations` is supplied, excluded auto-calibration runs once before loop `0` (`calibration_shape_warmup`, `pilot`, up to `2` `correction` steps targeting `0.150 s` in `[0.100 s, 0.250 s]`, plus optional `single_unit_confirmation_warmup`/`single_unit_confirmation`), freezes `frozen_plan_refs` atomically, and runs one `frozen_measurement_warmup` per active `(pattern, op)` before loop `0` (there are no per-loop warmups inside timed loops).

Across repeated `--count` loops (default `3`), both active patterns (`loop % active_patterns.size()`) and operations (`loop % 3`) rotate in a 2-axis cyclic Latin-square schedule (`pattern_order_balance_complete` and `operation_order_balance_complete`). Headline `aggregates.patterns.<kind>.<op>.value` is the P50 median of `measured` loops (alongside `relative_to_forward_pct` and full `statistics`); CV warnings trigger at `n >= 3` above `5.0%` for streaming patterns (`sequential_forward`, `sequential_reverse`, `strided_64`) and above `10.0%` for sparse patterns (`strided_4096`, `strided_16384`, `strided_2mb`, `random`). Unavailable or skipped metrics store `null` plus explicit `status`/`reason_code`, never `0`. See [PATTERNS_WHITEPAPER.md](PATTERNS_WHITEPAPER.md) for the complete specification.

---

## Command-Line Options

All seven primary modes share one descriptor-driven CLI option parser. Every single-occurrence option — including primary
mode selectors (`-B`, `-L`, `-P`, `-T`, `-C`, `-G`, `-M`), boolean flags (`--compare-locality`, `-u`, `-h`), and
single-value options such as `--buffer-size`, `--cache-size`, `--threads`, `--latency-samples`, and `--output` — must be
specified at most once per command across all seven modes. `-S`/`--sweep` is the sole repeatable option: repeat it once
per swept parameter, while using each parameter key at most once.

Long options require a double dash (`--`), and long value-taking options accept both space-separated (`--buffer-size 512`,
`--output -`) and inline (`--buffer-size=512`, `--output=-`, `--output=`, `--sweep=buffer-size=128,256`) syntax. Passing
an inline value to a boolean or primary-mode flag (such as `--cpu-bandwidth=1`, `--cpu-latency=1`,
`--compare-locality=1`, or `--help=1`) is rejected (`Option --... does not take a value`). A single dash is reserved for
space-separated one-character short options, so legacy single-dash long forms (`-buffersize`) and single-dash inline forms
(`-b=512`) are invalid.

All seven mode parsers emit concise single-line `Error: ...` diagnostics on `stderr` without dumping the full usage text
on parse errors, and standalone whitelist modes (`--cpu-bandwidth`, `--cpu-latency`, `--patterns`, `--analyze-tlb`,
`--analyze-core2core`, `--gpu-bandwidth`, `--llm-memory`) explicitly name the offending option
(`Unsupported option '<token>': ...`). Running `memory_benchmark` with no arguments prints help to stdout and exits `0`;
running with modifier options but no primary mode flag fails with `EXIT_FAILURE`, empty stdout, and a centralized error
on `stderr`.

Numeric values must be complete decimal tokens. Leading/trailing whitespace,
leading `+`, trailing characters, and overflow are rejected; unsigned seeds and standalone `--cpu-bandwidth`,
`--cpu-latency`, `--patterns`, and `--llm-memory` numeric options also reject either sign. Comma-separated sweep lists reject empty
leading, middle, and trailing items.

| Short | Long |
|---|---|
| `-B` | `--cpu-bandwidth` |
| `-L` | `--cpu-latency` |
| `-P` | `--patterns` |
| `-T` | `--analyze-tlb` |
| `-C` | `--analyze-core2core` |
| `-G` | `--gpu-bandwidth` |
| `-M` | `--llm-memory` |
| — | `--llm-memory-backend` |
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
| — | `--compare-locality` |
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
- In `--cpu-bandwidth` and `--cpu-latency`, `--buffer-size 0` (`0..2^53 - 1`) omits `main_memory` (provided at least one
  cache target remains active). Oversized requests are rejected at strict `memory_budget` preflight without silent truncation
- `--patterns` requires a positive value (`1..2^53 - 1`, `0` rejected) and enforces strict `memory_budget` preflight
  (`memory-budget-exceeded`) without silent truncation or capping
- In GPU mode, this is the exact size of each of two private buffers, the hard minimum is `64` MB, and the requested
  value is rejected rather than silently reduced if it exceeds the Metal or suite memory budget

#### `--iterations <count>`

- Exact measured R/W/Copy pass count (`--cpu-bandwidth`, `--patterns`, `--gpu-bandwidth`), complete pointer-chain cycles
  (`--cpu-latency`), or LLM scenario work-unit count when explicitly supplied
- Range: `1..1,000,000,000` in `--cpu-bandwidth`, `--cpu-latency`, and `--patterns`
- In `--cpu-bandwidth`, `--patterns`, `--gpu-bandwidth`, and `--llm-memory`, omission enables excluded
  operation- or scenario-specific work and automatic calibration toward 150 ms (100–250 ms intended window, at most
  two corrections). In `--cpu-latency`, omission calibrates complete chain cycles toward 250 ms (100–300 ms intended
  window, at most two corrections, floored at 16 complete cycles when guardrails permit)
- Resolved per-target/per-operation work is frozen once and reused across `--count` loops
- An explicit value bypasses pilot/corrections but not the frozen-plan warmup
- In GPU mode, the value is an exact full-buffer dispatch count. It must fit both the 16,384-dispatch limit and the
  64 GiB exact-payload limit; copy's 2× numerator defines the strict shared CLI ceiling
- In LLM mode, the value is an exact decode-step or full-prompt prefill-operation count per scenario. CPU caps one
  measurement at `1,000,000,000` work units, while Metal caps one dispatch at `65,536` work units (further lowered when
  applicable by Metal prefill's `1,048,576` lane-local serial range-helper visit cap or Metal paged KV's `268,435,456`
  semantic lookup cap). Paged KV table-read bytes are included in the 64 GiB task-accounted-byte ceiling even though
  they are excluded from effective-model-payload GB/s

#### `--count <count>`

- Full benchmark loop count
- `--cpu-bandwidth`, `--cpu-latency`, `--patterns`, `--analyze-core2core`, `--gpu-bandwidth`, and `--llm-memory` default:
  `3` (so repeated runs emit median P50 and repeatability metrics out of the box)
- In `--cpu-bandwidth`, cyclic Latin-square scheduling rotates active targets and `read`/`write`/`copy` operations
  (balancing both when `count % num_targets == 0` and `count % 3 == 0`). In `--patterns`, 2-axis cyclic Latin-square
  scheduling rotates active patterns (`loop % active_patterns.size()`) and operations (`loop % 3`). In `--cpu-latency`,
  cyclic scheduling rotates active targets (`target_order_balance_complete` when `count % num_targets == 0`)
- Positive integer: up to `2^53 - 1` (`9,007,199,254,740,991`) in `--cpu-bandwidth`, `--cpu-latency`, and `--patterns`,
  `INT_MAX` (`2,147,483,647`) for core-to-core and GPU modes, and `floor((2^53 - 1) / 3)` (`3,002,399,751,580,330`)
  for `--llm-memory`
- Use `5` to `10` for stable statistics

#### `--threads <count>`

- Thread count for bandwidth tests (`1..UINT32_MAX` in `--cpu-bandwidth` and `--patterns`)
- `--cpu-bandwidth` and `--patterns` default `main_memory` to the count of all detected logical CPU cores; `--cpu-bandwidth`
  defaults cache targets (`l1`/`l2`/`custom`) to `1` worker when `--threads` is omitted, while an explicit `--threads`
  value applies to every active target (capped per target to `min(desired, available_workers, max(1, target_bytes / 64))`)
- To reproduce a worker-count profile matching the detected P-core count, set that count explicitly with `--threads`;
  this does not pin those workers to P-cores
- In `--patterns`, `requested_workers` is preserved in `configuration` and `effective_workers` is clamped per pattern
  (`min(requested, available, max_aligned)` plus `>= 2` strided accesses per worker for strided patterns)
- Workers request best-effort macOS QoS (`QOS_CLASS_USER_INTERACTIVE`) but are not pinned to specific cores
- LLM-memory defaults its requested count to detected CPU workers. An explicit request remains distinct from detected
  availability; its immutable work plan records requested, available, and effective counts, reducing effective workers
  only when detected availability or executable span size requires it
- Metal LLM-memory does not detect CPU workers and rejects explicit `--threads`; its worker and worker-QoS JSON fields
  are null with applicability false
- Latency tests remain single-threaded; `--cpu-latency` rejects `--threads` (`Unsupported option ...`)

### Mode selection

#### `--cpu-bandwidth`

- Runs the standalone audited CPU memory bandwidth benchmark (`-B`/`--cpu-bandwidth`, schema 1, `mode: "cpu_bandwidth"`,
  methodology `cpu-bandwidth-v1-arm64-calibrated-seeded-balanced`)
- Exact 12-option whitelist: `-B/--cpu-bandwidth`, `-b/--buffer-size`, `-k/--cache-size`, `-t/--threads`, `-i/--iterations`,
  `-r/--count` (default `3`), `--seed`, `-u/--non-cacheable`, `-o/--output`, `-S/--sweep` (`buffer-size`, `cache-size`,
  `threads`), `-X/--sweep-max-runs` (default `256`), and `-h/--help`
- Measures `main_memory` + auto-detected `l1`/`l2` (or `--cache-size` `custom`), supporting `--buffer-size 0` or
  `--cache-size 0` selective omission (requiring at least one active target)
- Seeds persistent `src` once via O(1) GF(2) pattern (`cpu-bw-buffer-pattern-v1`) and protects it with `mprotect(PROT_READ)`;
  verifies read passes via non-cancelling checksum (`cpu-bw-read-checksum-v1`) and verifies all bytes of `dst` cold after
  `write` (`memory_write{_cache}_pattern_loop_asm`) and `copy` (`pre_copy_sentinel`)
- Rotates active targets and `read`/`write`/`copy` in a cyclic Latin-square schedule, captures `cpu_raw` Mach ticks, and
  writes bounded atomic file checkpoints every `K=max(1,ceil(count/8))` completed loops (stdout `--output -` writes one
  final JSON document and routes human output to stderr)
- Detailed methodology and schema contract: [CPU_BANDWIDTH_WHITEPAPER.md](CPU_BANDWIDTH_WHITEPAPER.md)

#### `--cpu-latency`

- Runs the standalone audited CPU pointer-chasing latency benchmark (`-L`/`--cpu-latency`, schema 1, `mode: "cpu_latency"`,
  methodology `cpu-latency-v1-arm64-calibrated-seeded-balanced`)
- Exact 16-option whitelist: `-L/--cpu-latency`, `-b/--buffer-size`, `-k/--cache-size`, `-i/--iterations`, `-r/--count`
  (default `3`), `-n/--latency-samples` (default `1000`, `0` to disable), `-s/--latency-stride-bytes` (default `256`),
  `-m/--latency-chain-mode` (`auto` default), `-l/--latency-tlb-locality-kb` (default `1024`), `--compare-locality`,
  `--seed`, `-u/--non-cacheable`, `-o/--output`, `-S/--sweep` (`buffer-size`, `cache-size`, `latency-tlb-locality-kb`,
  `latency-stride-bytes`, `latency-chain-mode`), `-X/--sweep-max-runs` (default `256`), and `-h/--help`
- Measures auto-detected `l1`/`l2` (or `--cache-size` `custom`) + `main_memory`, supporting `--buffer-size 0` or
  `--cache-size 0` selective omission (requiring at least one active target)
- Builds chains via `splitmix64-fisher-yates-rejection-v1`, cyclically rotates `indices` so `indices[0] == 0`, computes
  `chain_permutation_sha256`, prevalidates a single Hamiltonian cycle, and applies `mprotect(PROT_READ)`
- Uses continuous pointer-chase headlines calibrated toward 250 ms (`[100 ms, 300 ms]`, minimum 16 complete chain cycles
  when guardrails allow) returning to node `0` (`main-final-pointer`), optional continuous sample windows verifying every
  window end node (`sample-window-end-pointers`), and explicit opt-in `--compare-locality`
- Writes bounded atomic file checkpoints every `K=max(1,ceil(count/8))` completed loops (stdout `--output -` writes one
  final JSON document and routes human output to stderr)
- Detailed methodology and schema contract: [LATENCY_WHITEPAPER.md](LATENCY_WHITEPAPER.md)

#### `--patterns`

- Runs the standalone audited CPU access-pattern benchmark (`-P`/`--patterns`, schema 1, `mode: "patterns"`,
  methodology `patterns-v1-arm64-phase-calibrated-seeded-balanced`)
- Exact 11-option whitelist: `-P/--patterns`, `-b/--buffer-size` (`1..2^53 - 1`, `0` rejected), `-t/--threads`
  (`1..UINT32_MAX`), `-i/--iterations` (`1..1,000,000,000`), `-r/--count` (default `3`), `--seed` (long-only),
  `-u/--non-cacheable`, `-o/--output`, `-S/--sweep` (`buffer-size`, `threads`), `-X/--sweep-max-runs` (default `256`),
  and `-h/--help`
- Evaluates `sequential_forward`, `sequential_reverse`, `strided_64`, `strided_4096`, `strided_16384`, `strided_2mb`,
  and `random` (`strided_2mb` is `status: "skipped"` with `buffer-too-small-for-stride-transition` when
  `buffer_bytes < 2 MiB + 32 B`)
- Seeds persistent `src` once via slot-tagged O(1) GF(2) pattern (`patterns-buffer-pattern-v1`) and protects it with
  `mprotect(PROT_READ)`; verifies timed reads via `patterns-read-checksum-v1` (O(1) sequential & 2D partial-cycle
  strided), builds per-worker random indices via `splitmix64-coprime-fisher-yates-v1` (`random_indices_sha256`), and
  validates 100% of touched bytes and 100% of untouched `pre_op_sentinel` bytes cold after `write` and `copy`
  (`execution.validation.checks[]`)
- Automatically calibrates each active `(pattern, op)` toward 150 ms (`[0.100 s, 0.250 s]`, max `2` corrections) unless
  `--iterations` is supplied, freezes plans atomically, runs `frozen_measurement_warmup` before loop `0`, and rotates
  both active patterns and `read`/`write`/`copy` in a 2-axis cyclic Latin-square schedule
- Writes bounded atomic file checkpoints every `K=max(1,ceil(count/8))` completed loops (stdout `--output -` writes one
  final JSON document and routes human output to stderr)
- Detailed methodology and schema contract: [PATTERNS_WHITEPAPER.md](PATTERNS_WHITEPAPER.md)

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

- Selects the standalone CPU/Metal synthetic memory mode. Active methodologies are
  `llm-memory-v2-cpu-decode-contiguous`, `llm-memory-v2-cpu-decode-paged`,
  `llm-memory-v2-cpu-prefill-contiguous`, `llm-memory-v2-cpu-prefill-paged`,
  `llm-memory-v2-metal-decode-contiguous`, `llm-memory-v2-metal-decode-paged`,
  `llm-memory-v2-metal-prefill-contiguous`, and `llm-memory-v2-metal-prefill-paged`. It never enters the general
  `BenchmarkConfig` parser or CPU sweep runner
- The four active Metal methodologies are governed by runtime capability checks
- Requires each common model option `--weight-size-mb <MiB>`, `--layers <count>`, `--query-heads <count>`,
  `--kv-heads <count>`, and `--head-dim <count>` exactly once. Decode requires `--context-tokens <count>` (unless
  supplied via `--sweep context-tokens=...`); prefill requires `--prompt-tokens <P>` (unless supplied via
  `--sweep prompt-tokens=...`) and `--attention-query-tile-tokens <Q>` (unless supplied via
  `--sweep attention-query-tile-tokens=...`). Cross-phase geometry is rejected
- Allows optional `--llm-memory-backend <cpu|metal>`, `--kv-element-bytes <1|2|4>`, `--batch-size <count>`,
  `--phase <decode|prefill>`, `--kv-layout <contiguous|paged>`, CPU-only `-t`/`--threads <count>`,
  `-i`/`--iterations <count>`,
  `-r`/`--count <count>`, `--seed <uint64>`, `-o`/`--output <target>`,
  `-S`/`--sweep <key=a,b>` (repeatable for `context-tokens`, `kv-block-tokens`, `prompt-tokens`, and
  `attention-query-tile-tokens`), `-X`/`--sweep-max-runs <count>`, and `-h`/`--help`.
  `--kv-block-tokens <G>` is required exactly once with paged layout (unless supplied via `--sweep kv-block-tokens=...`)
  and rejected with contiguous layout. Combining a direct flag and `--sweep` for the same parameter is rejected.
  Every other option, including all other primary modes, cache/latency modifiers, and `--non-cacheable`, is rejected
- Defaults to CPU, decode, contiguous layout, two-byte KV elements, batch size one, three planned cyclic loops,
  automatic scenario work, and one generated non-zero base seed. CPU defaults to detected workers. Metal performs no
  worker detection and rejects explicit `--threads`. There is no default paged block size. An explicit seed may be any
  unsigned 64-bit value, including zero
- Parses every integer as one complete unsigned decimal token (rejecting signs, whitespace, and trailing characters).
  Model counts, sizes, worker count, iterations, and loop count must be positive. Every serialized integer configuration
  field (`--weight-size-mb`, `--layers`, `--query-heads`, `--kv-heads`, `--head-dim`, `--context-tokens`,
  `--prompt-tokens`, `--attention-query-tile-tokens`, `--kv-block-tokens`, `--batch-size`, `--threads`, `--iterations`,
  `--count`) and, when `--sweep` is active, `--sweep-max-runs` must not exceed the IEEE-754 exact safe integer ceiling
  `2^53 - 1` (`9,007,199,254,740,991`; `reason_code=json-integer-out-of-range`), and `--count` is further bounded by
  `floor((2^53 - 1) / 3)` (`3,002,399,751,580,330`). Without `--sweep`, `--sweep-max-runs` accepts any positive `size_t`
  with no effect and does not apply the `2^53 - 1` sweep-envelope ceiling. Query heads must be at least the KV-head
  count and evenly divisible by it
- Requires `P >= 1` and `1 <= Q <= P` for prefill. Requires paged block size `G` to be positive, a power of two, and at
  most `UINT32_MAX`. `G > A` is valid for decode and produces one partially used physical block per batch sequence;
  unused capacity is reported rather than silently removed
- Activates Metal for decode or prefill with contiguous or paged KV. Capability failure after
  output-session creation produces a terminal `unsupported` JSON result and non-zero exit. Runtime compiler, pipeline,
  resource, or task failure produces terminal `failed`/`invalid` schema evidence and a non-zero exit. The command never
  substitutes CPU
- Treats `--context-tokens` as the fixed visible context including the current synthetic token. Checked preflight
  resolves weight/KV byte geometry and ensures all three scenarios can fit the backend's work-unit cap (`1,000,000,000`
  on CPU; `65,536` on Metal, further lowered by Metal prefill's `1,048,576` lane-local serial range-helper visit cap or
  Metal paged KV's `268,435,456` semantic lookup cap where applicable) and 64 GiB accounted-byte task limits; explicit
  iterations must fit the strictest scenario limit. For `--llm-memory --sweep`, every generated Cartesian combination is
  preflight-validated before the first run starts
- Treats one prefill work unit as one complete prompt rewrite plus tiled causal-prefix scans for every batch sequence;
  weights are read once per full-prompt operation, not once per token or tile. With paged KV, full-prompt writes and
  exact partial-prefix scans perform timed uint32 table lookups
- On CPU, preserves an explicit worker request even when it exceeds detected availability. Effective-worker reduction
  belongs to the executable work plan and remains separately reportable
- `--llm-memory --help` succeeds without required model inputs and returns before core detection, seed generation,
  output-session creation, QoS preparation, or signal masking. Unknown, duplicate, and missing-value errors are still
  diagnosed because help does not bypass the strict whitelist pass
- Builds budget-admitted full-size active-weight and K/V resources. Contiguous uses the logical K/V lengths. Paged uses
  complete physical K/V blocks and one uint32 block table, and admits the permutation-validation transient before
  materializing the table. CPU uses page-rounded regular cacheable anonymous mappings and protects the validated table
  read-only. Metal uses block-aligned private K/V segments, segmented private table storage, and a transient shared
  staging segment; table upload and hash/readback validation precede publication. `--non-cacheable` is outside the
  whitelist. Deterministic initialization and pre-touch cover every requested physical byte before any measured task.
  Before each paged
  decode task, an untimed allocation-free reset restores only the mutable current-token append slots; history blocks
  and suffix-padding canaries are not rewritten. Paged prefill instead rewrites every owned prompt byte in timed work
- Runs `weights_only`, `kv_only`, and layer-interleaved `mixed` scenarios. Omitted iterations trigger excluded,
  scenario-specific calibration in that canonical order with an 8 MiB minimum pilot accounted-work floor when
  guardrails permit, a 150 ms target, a 100–250 ms intended window, and at most two corrections. The initial pilot has
  one same-shape warmup; later candidates add a warmup only for the first irreducible one-work-unit confirmation. After
  all three candidates resolve, their plans freeze atomically and each frozen plan receives one canonical-order warmup
  before loop zero. Explicit iterations freeze all three exact plans first and use the same frozen-plan warmup boundary.
  The measured WKM/KMW/MWK rotation balances first/middle/last positions per complete three-loop block, independently
  of seed. It does not balance directed predecessor pairs across loop/block boundaries. A partial final block keeps its
  actual order. There is no runner-level same-scenario conditioning immediately before each measurement; backend
  preparation and validation still occur outside its primary time. Changing conditioning or the six-permutation block
  order would change comparison policy and requires a versioned methodology/run-policy decision.
- Uses task-boundary completion-wins interruption. A started scenario is not polled in its hot kernel; a completed and
  checksum-valid current task stays measured, while no next task starts after the stop is observed
- Output values retain the shared raw-target syntax: empty disables JSON for direct commands (invalid for sweeps),
  exact `-` emits one final LLM schema 2 document (or one final schema 1 sweep envelope for `--sweep`), and every
  other non-empty value is a file target, including `./-` and flag-shaped names. Direct file output atomically
  snapshots every `K=max(1,ceil(count/8))` completed loops plus command terminal, with at most eight progress
  snapshots and an abrupt-loss bound of `3K` completed attempts. `--llm-memory --sweep` file targets instead checkpoint
  the schema-1 sweep envelope atomically after each attempted run and do not write intra-run loop snapshots. Stdout
  retains task stop observations without intermediate snapshot preparation
- CPU prefill uses layout-specific ARM64 executors and reports scenario-specific cost-balanced owner identities plus
  minimum/maximum/imbalance worker-cost evidence. Paged prefill additionally reports deterministic block-exclusive
  ownership, table/permutation identity, exact lookup metadata, and padding validation
- Every active Metal profile uses private/tracked W/K/V segments no larger than 256 MiB, with exact final tails and a
  Tier 2 argument buffer. Each task has one reset command buffer, one timed command buffer with one serial compute
  encoder and one scenario-specialized workload dispatch, and one excluded post-validation command buffer. The `T`
  work-unit loop is
  inside the kernel. `GPUStartTime`/`GPUEndTime` are authoritative; the host wait envelope is diagnostic only. A
  versioned profile-specific dual-mod32 W/K/V checksum and excluded phase-neutral `kv_write` validation must pass, with
  no performance retry. Decode validates its current-token write; prefill checks representative/boundary byte locations
  per layer/batch sequence in both K and V from the full prompt against the final operation ordinal `T - 1`. Both paged
  phases use cyclic grid-stride scheduling that maps each layer/batch/logical-block owner to exactly one threadgroup. A
  named lane performs each volatile uint32 table load, publishes the physical ID to its threadgroup, and crosses a
  required threadgroup barrier before address construction. Paged decode's paired append plus separate K/V scans
  account exactly
  `L * B * (2 * N + 1)` lookups per KV-active work unit. The paged checksum binds logical index, physical ID, visit
  kind, and work-unit ordinal; excluded validation also checks terminal-block padding canaries. Paged prefill instead
  performs `N` paired prompt-write lookups and `M` separate K-prefix plus `M` V-prefix lookups per layer/batch pair,
  giving exact
  `L * B * (N + 2 * M)` metadata loads. Its KV-active `L * B * N` row-major owners use a deterministic cyclic threadgroup map
  without lookup duplication. Grid evidence reports exact per-threadgroup accounted-byte minimum, maximum, imbalance,
  and the underlying vector with `cost_unit=actual-threadgroup-cost` without claiming weighted balance. Every Metal
  `weights_only` task uses a weight-vector grid stride and reports the same exact cost unit. Contiguous KV-bearing grids
  publish no threadgroup-cost evidence. Partial prefix visits stop at the exact tile or prompt end. KV-bearing tasks
  require final-operation representative/boundary prompt-write sample validation plus applicable padding-canary validity. In contiguous
  prefill, each lane writes its disjoint slices across all prompt K/V records before reading those slices in the tiled
  prefixes; it scans the complete K prefix before the V prefix at every tile end. A weight-bearing scenario performs
  one weight pass per `prefill_operation`. No
  grid-wide barrier is implied. Contiguous prefill's lane-local serial range-helper count is
  `T * ((weight_active ? L : 0) + (kv_active ? 2 * L * B * (P + C) : 0))`; paged `weights_only` counts `T * L`, while
  paged KV-bearing owner kernels report zero serial range visits. The task publishes the applicable count and rejects
  overflow or values above 1,048,576 before the CPU checksum oracle or GPU dispatch. The cap also lowers the effective
  scenario work-unit ceiling used by explicit-work validation and automatic calibration. Methodology evidence publishes
  the integer cap for both Metal prefill layouts and null for every other profile
- Runtime evidence records the canonical embedded MSL revision, its exact SHA-256, and the selected profile's workload
  and parameter-layout-probe identities. Validation pipelines are selected and compiled internally; runtime evidence
  records their outcomes, not separate labels. Metal paged prefill additionally records executor
  `llm-metal-executor-v1-prefill-paged`, schedule
  `llm-metal-prefill-paged-cyclic-block-owner-grid-stride-v1`, timer
  `metal-command-buffer-gpu-start-end-v1`, buffer pattern
  `llm-paged-physical-buffer-pattern-v1`, write pattern `llm-metal-prefill-paged-full-prompt-affine32-v1`, and checksum
  `llm-metal-paged-prefill-dual-mod32-lookup-address-mix-v1`
- This profile is memory-only: it performs no Transformer mathematics and does not report inference tokens/s. Its
  `synthetic_memory_work_units_per_second` and `effective_model_payload_gb_s` must not be interpreted as model
  throughput or physical DRAM-counter traffic. Paged block-table reads are timed and separately accounted but are not
  included in effective-model-payload GB/s. See
  [LLM_MEMORY_PROFILE_WHITEPAPER.md](LLM_MEMORY_PROFILE_WHITEPAPER.md)

#### `--analyze-tlb`

- Runs standalone TLB analysis mode only
- Can be combined only with optional `--output <target>`, `--latency-stride-bytes <bytes>`, `--latency-chain-mode <mode>`, `--tlb-density <low|medium|high>`, `--seed <uint64>`, `--sweep <key=...>`, `--sweep-max-runs <count>`, and `-h`/`--help`
- `--analyze-tlb --help` (including `--analyze-tlb --output - --help` and `--analyze-tlb --help --output -`) prints
  human-facing help to stdout and exits `0` after syntactic, whitelist, duplicate, missing-value, and value-parse
  checks, before semantic validation, OS queries, timer creation, seed generation, or output-session setup
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

- Applies to `--cpu-bandwidth`, `--cpu-latency`, `--patterns`, `--analyze-tlb`, `--gpu-bandwidth`, or `--llm-memory`
  (`0..UINT64_MAX`, `0` allowed)
- In `--cpu-bandwidth`, derives domain-separated target seeds (`main_memory`, `l1`, `l2`, `custom`, `pre_copy_sentinel`),
  operation seeds, write patterns, and per-worker checksum initial states; shared across all runs in a sweep
- In `--cpu-latency`, derives domain-separated chain seeds (`l1`, `l2`, `custom`, `main_memory`, `locality_16k`,
  `locality_global`); shared across all runs in a sweep
- A seed reproduces workload/schedule/chain identity, not performance values or macOS thread placement
- In `--patterns`, derives the `SplitMix64` domain seed hierarchy (`src_seed`, `random_domain_seed`, `pre_op_sentinel_seed`),
  per-op seeds, unique odd `write_pattern`/`pre_op_sentinel` sequences, and per-worker `splitmix64-coprime-fisher-yates-v1`
  random offsets; the same resolved seed reproduces all workloads across `--count` loops and sweeps
- In `--patterns`, one unsigned 64-bit nonzero seed is generated once for the command when omitted
- In `--analyze-tlb`, controls base/refinement/validation/large-locality round order, pointer-chain construction, and
  deterministic bootstrap resampling
- In `--analyze-tlb`, the same seed reproduces planner order, derived task seeds, and chain permutations
- In `--analyze-tlb`, one unsigned 64-bit seed is generated for the command when omitted and reused by every TLB run in
  a Cartesian sweep
- Standalone TLB JSON stores the resolved seed, source (`user` or `generated`), schedule policy, and each task seed
- In GPU mode, the base seed is generated once when omitted, recorded as an exact decimal string, and used to derive
  stable domain-separated read/write/copy operation seeds. It reproduces data/work identity, not performance
- In LLM-memory mode, one base seed is generated when omitted and reused across every generated run in an `--llm-memory`
  Cartesian sweep. The work planner derives separate weight/K/V buffer
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

- Sample count per latency test (`0..INT_MAX` in `--cpu-latency`, `1..INT_MAX` in `--analyze-core2core`)
- Default: `1000`
- In `--cpu-latency`, `0` disables sample windows; positive values run a separate pass whose windows continue from the
  preceding window's terminal pointer and verify each window's end pointer (`sample-window-end-pointers`)
- In `--cpu-latency`, the effective sample count is capped to that measurement's access count (`min(requested, access_count)`)
  so every sample window contains at least one access
- In core-to-core mode, each sample is a separately timed handoff window calibrated toward 1 ms with a 2,000-round-trip minimum
- In both modes, sample count/granularity does not define or change the separate continuous headline calculation

#### `--latency-stride-bytes <bytes>`

- Pointer-chain stride for latency tests
- Default: `256`
- Must be `> 0` (`8..UINT32_MAX` in `--cpu-latency`)
- Must be a multiple of 8 bytes (pointer size on Apple Silicon)
- In `--cpu-latency`, every enabled `l1`, `l2`, `custom`, or `main_memory` chain and the configured locality window must
  retain at least two pointer nodes after applying the stride
- When `--compare-locality` is passed, stride must be `<= 8192` bytes so a 16 KiB locality window retains at least two
  nodes (checked at preflight)
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

- Pointer-chain locality window for `--cpu-latency` (`0..2^53 - 1`)
- Default: `1024`
- `0` disables locality mode (resolves `auto` to `global-random`)
- When the effective chain mode uses locality, non-zero values must be exact multiples of system page size. Explicit
  `global-random` ignores the locality window and does not apply this page-multiple constraint
- For a chain mode that uses locality windows, a non-zero window must contain at least two stride-spaced pointer nodes

#### `--compare-locality`

- Explicit opt-in flag for `--cpu-latency` (not enabled automatically)
- Runs `3` paired comparison rounds per loop before target measurements comparing a `16 KiB` `random-box` chain
  (`locality_16k`) against a `global-random` chain (`locality_global`) using the frozen `main_memory` `complete_chain_cycles`
- Requires `--buffer-size > 0` and `--latency-stride-bytes <= 8192`
- Publishes `locality_comparison.rounds[]` and `aggregates.locality_comparison`
  (`locality_latency_delta_ns = median(global) - median(16k)`). Combines cache, locality, and translation effects; not
  an isolated page-walk cost

### Cache and memory hint controls

#### `--cache-size <KB>`

- Enables custom cache test size (`-B` and `-L`)
- Non-zero range: `16` to `1048576` KB (1 GB)
- `0` is accepted in `--cpu-bandwidth` and `--cpu-latency` to omit cache targets (requires `--buffer-size > 0`)
- When set to non-zero, auto L1/L2 detection is replaced by the single `custom` cache target

#### `--non-cacheable`

- Applies cache-discouraging `madvise(MADV_RANDOM)` hints (`-B`, `-L`, `-P`)
- Best effort only; this does **not** create truly uncached memory

### Output

#### `--output <target>`

- An omitted output option or an empty value disables JSON output for a direct command. For a sweep, an empty value is
  a missing/invalid required output target
- For every result-producing direct mode and supported sweep, an exact raw value of `-` selects
  machine-readable stdout. The sentinel is classified before path normalization
- A supported stdout-target command emits exactly one two-space-indented UTF-8 JSON document followed by one newline
  after orchestration reaches its terminal state. Intermediate sweep and GPU checkpoint requests are
  lazy no-ops; CPU bandwidth, CPU latency, and LLM skip intermediate snapshot preparation. None emits intermediate
  documents
- Post-parse human output is routed to stderr while the stdout target is active. Parse, preflight, and other pre-result
  failures leave stdout empty; `--help` remains a human-facing stdout command across all seven primary modes (including
  `--analyze-tlb --help`, `--output - --help`, and `--output=- --help`)
- For result-producing modes, every other non-empty value is a file target, including `./-` and flag-shaped names such
  as `-G`. Thus `--output ./-` writes an ordinary file named `-`. Relative paths write under the current working
  directory, and parent directories are created automatically. CPU bandwidth, CPU latency, patterns, GPU, and LLM-memory retain
  the raw target spelling in `configuration.output_file`
- Sweep file targets retain atomic per-attempt checkpoints (without intra-run loop snapshots). TLB and
  core-to-core direct file targets each receive one final atomic write. GPU file output retains its
  terminal-measurement/failure checkpoints and post-release replacement. Direct CPU bandwidth, CPU latency, patterns, and LLM files
  snapshot every `K=max(1,ceil(count/8))` completed loops plus command terminal (`checkpoint_lifecycle`, max 8 progress
  writes, abrupt loss bound `3K`—or `K * active_patterns.size() * 3` for `-P`—completed attempts). No file is promised before the first successful snapshot, and
  failed persistence is terminal without retry
- With GPU `--output -`, every logical checkpoint transition and post-checkpoint stop observation still runs, but its
  lazy payload builder is not invoked and no intermediate document is serialized. The command boundary writes one
  terminal GPU schema 1 document. CPU bandwidth, CPU latency, patterns, and LLM retain task stop observations and emit one
  terminal document with no intermediate DOM. `configuration.output_file` remains `"-"`, and captured `argv` is not
  rewritten
- GPU syntax/config errors, including a buffer below 64 MB, fail before result JSON is created. A backend-factory failure
  also precedes result initialization. Initialized capability, compilation, allocation, or work-plan failures remain
  auditable in the selected file or stdout target
- LLM parser/preflight, output-session initialization, backend-factory, work-plan/memory-budget, CPU paged block-table
  mapping/materialization/protection, JSON-output peak-estimation, auxiliary-bytes overflow, and CPU timer-creation
  (`timer-unavailable`) failures occur before runner-result initialization and therefore leave stdout empty (and write no
  direct JSON file). Once the work plan is finalized, backend capability/initialization (other than `timer-unavailable`),
  plan resolution, resource preparation (weight/K/V mapping, initialization, descriptor preparation, and Metal
  compilation/pipeline/allocation/upload/validation), and post-plan setup exceptions emit terminal schema-2 evidence
  through the selected non-empty output target
- An observable terminal stdout serialization, write, or flush failure returns failure without changing the
  already-computed measurement state; any bytes from that failed transfer must be rejected

#### `--sweep <key=value1,value2>`

- Runs a Cartesian parameter sweep and writes one combined JSON result
- Requires a non-empty `--output <target>`; exact `-` selects one final envelope on stdout, while every other non-empty
  value is a checkpointed file target, including `./-` and flag-shaped names such as `-G`. An empty value is
  missing/invalid for a sweep
- Can be repeated to sweep multiple parameters
- Supported keys: `buffer-size`, `cache-size`, `threads`, `latency-tlb-locality-kb`, `latency-stride-bytes`, `latency-chain-mode`, `tlb-density`, `count`, `latency-samples`, `context-tokens`, `kv-block-tokens`, `prompt-tokens`, `attention-query-tile-tokens`
- `tlb-density` applies only with `--analyze-tlb`
- `--cpu-bandwidth` supports `buffer-size`, `cache-size`, and `threads`
- `--cpu-latency` supports `buffer-size`, `cache-size`, `latency-tlb-locality-kb`, `latency-stride-bytes`, and `latency-chain-mode`
- `--patterns` supports `buffer-size` and `threads`
- In a `--patterns` thread sweep, each `threads` value is the requested count. A strided pattern may reduce its
  effective worker count to keep at least two valid strided addresses per active worker, so sparse-stride results must
  not be interpreted as requested-thread scaling when this reduction applies
- `--analyze-tlb` supports `latency-stride-bytes`, `latency-chain-mode`, and `tlb-density`
- `--analyze-core2core` supports `count` and `latency-samples`
- `--llm-memory` supports `context-tokens` (decode), `prompt-tokens` and `attention-query-tile-tokens` (prefill), and
  `kv-block-tokens` (paged KV), and preflight-validates every generated Cartesian configuration before execution
- Combining a direct option and `--sweep` for the same parameter is rejected uniformly across all sweep-capable modes
  (`--cpu-bandwidth`, `--cpu-latency`, `--patterns`, `--analyze-tlb`, `--analyze-core2core`, and `--llm-memory`) via
  `Cannot specify both --<opt> and --sweep <key>`
- `--gpu-bandwidth` does not support `--sweep` or `--sweep-max-runs` (GPU schema 1)

#### `--sweep-max-runs <count>`

- Maximum number of generated sweep combinations
- Default: `256` (`16` with `--analyze-tlb`)
- Mode-specific upper bounds: `1..(2^53 - 1)` for `--cpu-bandwidth`, `--cpu-latency`, `--patterns`, and active `--llm-memory --sweep`
  (`json-integer-out-of-range` for LLM); `1..LLONG_MAX` for `--analyze-tlb`; `1..INT_MAX` for
  `--analyze-core2core`. Accepted without `--sweep` (with no effect, and without enforcing the `2^53 - 1` bound in direct
  `--llm-memory`); rejected by `--gpu-bandwidth`
- An explicit `--sweep-max-runs` value overrides the mode-specific default
- Prevents accidental very large Cartesian sweeps
- Every generated configuration is validated before the first run
- Combined output uses envelope schema `configuration.sweep_schema_version: 1`; each `runs[].result` retains its own
  mode schema version
- CPU bandwidth, CPU latency, pattern, TLB, core-to-core, and LLM-memory sweep file output is atomically checkpointed
  after every attempted run (without intra-run loop snapshots). An empty run plan or a stop observed before a run also
  checkpoints a terminal envelope without adding a `runs[]` entry or incrementing `attempted_runs`. Each envelope
  records `status`, `status_reason`, `planned_runs`, `attempted_runs`, `completed_runs`, and `conclusions_valid`. Stdout
  runs retain the same logical checkpoint cadence without invoking the persistence payload builder or serializing an
  intermediate document, and emit only the final envelope
- Every attempted run is retained with its own `status` and `status_reason`. `attempted_runs` counts stored entries,
  while `completed_runs` counts only mode-specific nested results that are genuinely complete:
  - CPU bandwidth requires nested `mode: "cpu_bandwidth"`, `schema_version: 1`, `status: "complete"`,
    `results_complete: true`, `run_accepted: true`, `configuration.output_file: ""`, and every planned measurement `measured`
  - CPU latency requires nested `mode: "cpu_latency"`, `schema_version: 1`, `status: "complete"`,
    `results_complete: true`, `run_accepted: true`, `configuration.output_file: ""`, and every planned measurement `measured`
  - Pattern requires nested `mode: "patterns"`, `schema_version: 1`, `status: "complete"`, `results_complete: true`,
    `run_accepted: true`, `configuration.output_file: ""`, and a non-empty `measurements[]` array in which every scheduled
    entry has `status: "measured"`
  - TLB requires nested `tlb_analysis.status: "complete"` with `tlb_analysis.conclusions_valid: true`
  - Core-to-core requires nested `core_to_core_latency.status: "complete"` with `measurements_complete: true`
  - LLM-memory requires nested `mode: "llm_memory"`, `schema_version: 2`, `status: "complete"`, `results_complete: true`,
    `run_accepted: true`, and every planned `measurements[]` entry having `status: "measured"`
  Partial, interrupted, unsupported, and failed nested results never increment `completed_runs`.
  TLB's native `tlb_analysis.status: "error"`, CPU's native `status: "unsupported"`, and LLM-memory's native
  `status: "unsupported"` map to failed sweep attempts while retaining the nested payload; pre-initialization failures
  record `result: null` with the specific stable `status_reason`, and an interruption during pre-runner or backend
  resource preparation records `result: null` and `status: "interrupted"` with `status_reason: "preparation-interrupted"`
  and exit status `0`
- A parameter key may appear only once in one sweep command across all sweep-capable modes (duplicates fail with
  `sweep parameter specified more than once: <key>`)
- Any partial, interrupted, or failed attempt stops further attempts; a pre-run interruption or checkpoint failure can
  also stop execution without adding or completing another run. The authoritative schema-1 sweep acceptance predicate
  is exactly top-level `status == "complete" && conclusions_valid == true`. Producers make
  `completed_runs == planned_runs` an invariant of such an envelope. Consumers may check that equality separately as a
  defensive consistency check, but it is not an additional completeness predicate

#### `-h`, `--help`

- Accepted standalone and across all seven primary modes (`--cpu-bandwidth`, `--cpu-latency`, `--patterns`,
  `--analyze-tlb`, `--analyze-core2core`, `--gpu-bandwidth`, and `--llm-memory`)
- Runs full syntactic, whitelist, duplicate, missing-value, and value-parse validation first (for example,
  `--help --unknown`, `-h --help`, and `--help --iterations` fail)
- Once valid, prints human-facing help to stdout and exits `0` before required-option checks, semantic validation, OS
  queries, timer creation, seed generation, or output-session setup (including `--output - --help` and
  `--help --output -`)

---

## Mode Compatibility

### Valid combinations

```bash
# Audited CPU bandwidth (default 3 loops, main memory + auto L1/L2)
memory_benchmark --cpu-bandwidth --count 10 --buffer-size 1024 --output cpu_bw.json

# Audited CPU bandwidth with custom cache target and explicit threads
memory_benchmark --cpu-bandwidth --buffer-size 512 --cache-size 4096 --threads 4 --count 5 --output bw_custom.json

# Direct CPU bandwidth automation: JSON stdout and human transcript stderr
memory_benchmark --cpu-bandwidth --count 5 --buffer-size 512 --output - >cpu_bw.json 2>cpu_bw.log

# Audited CPU latency (both auto caches + main memory, 5000 sample windows)
memory_benchmark --cpu-latency --latency-samples 5000 --count 10 --output cpu_lat.json

# Audited CPU latency with explicit paired 16 KiB vs global-random comparison
memory_benchmark --cpu-latency --compare-locality --count 5 --output cpu_lat_locality.json

# CPU latency (main memory only, sample windows disabled)
memory_benchmark --cpu-latency --cache-size 0 --buffer-size 1024 --latency-samples 0

# CPU latency (custom cache only)
memory_benchmark --cpu-latency --buffer-size 0 --cache-size 2048

# Pattern-only
memory_benchmark --patterns --count 5 --buffer-size 512 --output patterns.json

# Direct pattern automation
memory_benchmark --patterns --count 5 --buffer-size 512 --output - >patterns.json 2>patterns.log

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

# Standalone LLM automation: reproducible fixed work and one final schema 2 document on stdout
memory_benchmark --llm-memory --weight-size-mb 64 --layers 4 --query-heads 8 --kv-heads 2 \
  --head-dim 64 --context-tokens 512 --iterations 1 --count 3 --seed 42 --output - \
  >llm_memory_stdout.json 2>llm_memory.log

# Standalone Metal contiguous decode with the same three synthetic scenarios
memory_benchmark --llm-memory --llm-memory-backend metal --weight-size-mb 64 --layers 4 \
  --query-heads 8 --kv-heads 2 --head-dim 64 --context-tokens 512 \
  --iterations 1 --count 3 --seed 42 --output llm_memory_metal.json

# Standalone Metal paged decode with a partial terminal block
memory_benchmark --llm-memory --llm-memory-backend metal --weight-size-mb 64 --layers 4 \
  --query-heads 8 --kv-heads 2 --head-dim 64 --context-tokens 513 \
  --kv-layout paged --kv-block-tokens 16 \
  --iterations 1 --count 3 --seed 42 --output llm_memory_metal_paged.json

# Standalone Metal contiguous prefill with one full-prompt write and tiled-prefix scans per operation
memory_benchmark --llm-memory --llm-memory-backend metal --weight-size-mb 64 --layers 4 \
  --query-heads 8 --kv-heads 2 --head-dim 64 --phase prefill --prompt-tokens 512 \
  --attention-query-tile-tokens 64 \
  --iterations 1 --count 3 --seed 42 --output llm_memory_metal_prefill.json

# Standalone Metal paged prefill with the same logical prompt and an explicit block size
memory_benchmark --llm-memory --llm-memory-backend metal --weight-size-mb 64 --layers 4 \
  --query-heads 8 --kv-heads 2 --head-dim 64 --phase prefill --prompt-tokens 512 \
  --attention-query-tile-tokens 64 --kv-layout paged --kv-block-tokens 16 \
  --iterations 1 --count 3 --seed 42 --output llm_memory_metal_prefill_paged.json

# Standalone CPU decode with deterministic paged KV storage
memory_benchmark --llm-memory --weight-size-mb 64 --layers 4 --query-heads 8 --kv-heads 2 \
  --head-dim 64 --context-tokens 512 --kv-layout paged --kv-block-tokens 16 \
  --iterations 1 --count 3 --seed 42 --output llm_memory_paged.json

# Standalone CPU prefill with contiguous KV and explicit prompt/tile geometry
memory_benchmark --llm-memory --weight-size-mb 64 --layers 4 --query-heads 8 --kv-heads 2 \
  --head-dim 64 --phase prefill --prompt-tokens 512 --attention-query-tile-tokens 64 \
  --iterations 1 --count 3 --seed 42 --output llm_memory_prefill.json

# Standalone CPU prefill with deterministic paged KV
memory_benchmark --llm-memory --weight-size-mb 64 --layers 4 --query-heads 8 --kv-heads 2 \
  --head-dim 64 --phase prefill --prompt-tokens 512 --attention-query-tile-tokens 64 \
  --kv-layout paged --kv-block-tokens 16 \
  --iterations 1 --count 3 --seed 42 --output llm_memory_prefill_paged.json

# Standalone LLM decode sweep across context length and paged block size (6 runs)
memory_benchmark --llm-memory --weight-size-mb 64 --layers 4 --query-heads 8 --kv-heads 2 \
  --head-dim 64 --kv-layout paged \
  --sweep context-tokens=512,2048,8192 --sweep kv-block-tokens=16,64 \
  --iterations 1 --count 3 --seed 42 --output llm_paged_sweep.json

# CPU latency sweep over 3 buffer sizes and 3 locality windows (9 runs)
memory_benchmark --cpu-latency --count 5 --sweep buffer-size=256,512,1024 --sweep latency-tlb-locality-kb=16,1024,0 --output latency_sweep.json

# Thread scaling sweep for CPU bandwidth
memory_benchmark --cpu-bandwidth --count 5 --sweep buffer-size=512,1024 --sweep threads=1,4,8 --output bandwidth_thread_sweep.json
```

### Invalid combinations

```bash
# invalid: modifier option(s) without a primary mode flag (bare memory_benchmark prints help)
memory_benchmark --threads 4
memory_benchmark --seed 42
memory_benchmark --buffer-size 512

# invalid: --cpu-bandwidth with --cpu-latency or --patterns (primary modes are mutually exclusive)
memory_benchmark --cpu-bandwidth --cpu-latency
memory_benchmark --cpu-bandwidth --patterns

# invalid: unrecognized option or misplaced mode flag (removed pre-split combined/subset flags are rejected as unknown/unsupported)
memory_benchmark --unknown-flag
memory_benchmark --cpu-bandwidth --tlb-density 4096

# invalid: pattern mode rejects unused cache and latency modifiers
memory_benchmark --patterns --cache-size 64
memory_benchmark --patterns --latency-stride-bytes 128

# invalid: latency modifiers with --cpu-bandwidth (outside exact whitelist)
memory_benchmark --cpu-bandwidth --latency-samples 5000
memory_benchmark --cpu-bandwidth --latency-stride-bytes 128

# invalid: --threads with --cpu-latency (latency tests are single-threaded)
memory_benchmark --cpu-latency --threads 4

# invalid: --compare-locality with --buffer-size 0 or --latency-stride-bytes > 8192
memory_benchmark --cpu-latency --buffer-size 0 --compare-locality
memory_benchmark --cpu-latency --latency-stride-bytes 16384 --compare-locality

# invalid: all targets omitted in --cpu-bandwidth or --cpu-latency
memory_benchmark --cpu-bandwidth --buffer-size 0 --cache-size 0
memory_benchmark --cpu-latency --buffer-size 0 --cache-size 0

# invalid: direct option combined with --sweep for the same parameter (all sweep-capable modes)
memory_benchmark --cpu-bandwidth --buffer-size 512 --sweep buffer-size=256,512 --output sweep.json
memory_benchmark --cpu-latency --latency-stride-bytes 256 --sweep latency-stride-bytes=64,256 --output lat_sweep.json
memory_benchmark --analyze-core2core --count 3 --sweep count=1,3 --output core2core_sweep.json

# invalid: inline value on a boolean/mode flag or single-dash inline value
memory_benchmark --cpu-bandwidth=1
memory_benchmark --cpu-bandwidth -b=512

# invalid: duplicated single-occurrence mode or boolean flag
memory_benchmark --cpu-bandwidth --cpu-bandwidth
memory_benchmark --analyze-tlb --help -h

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

# invalid: LLM sweep conflicts with the matching direct option
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

# invalid: multiple primary modes
memory_benchmark --gpu-bandwidth --cpu-bandwidth
```

---

## Common Workflows

### Quick baseline

```bash
memory_benchmark --cpu-bandwidth
memory_benchmark --cpu-latency
```

Good for a fast health check (both default to `--count 3`).

### Statistical baseline (recommended)

```bash
caffeinate -i -d memory_benchmark --cpu-bandwidth --count 10 --buffer-size 1024 --output bw_baseline.json
caffeinate -i -d memory_benchmark --cpu-latency --count 10 --buffer-size 1024 --output lat_baseline.json
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

For a Metal cohort, select the backend explicitly, omit `--threads`, and keep the Metal device/capability, MSL revision
and source hash, pipeline/grid, resource segmentation, work count, and seed evidence matched:

```bash
caffeinate -i -d memory_benchmark --llm-memory --llm-memory-backend metal \
  --weight-size-mb 4096 --layers 32 --query-heads 32 --kv-heads 8 \
  --head-dim 128 --kv-element-bytes 2 --context-tokens 8192 --batch-size 1 \
  --iterations 1 --count 6 --seed 123456789 --output llm_metal_fixed.json
```

Metal and CPU results are separate methodology cohorts. Do not combine their samples or compare their checksum values.
Metal private storage is unified-memory API policy, not evidence of separate VRAM or DRAM residency.

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

| Pattern | Operation | Status | Median GB/s | CV % | Requested/effective workers | Median duration s | Phase-zero / min-max accesses | Phase period | Passes | Exact `total_accesses` / `exact_payload_bytes` | `logical_working_set_bytes` | Notes |
|---|---|---|---:|---:|---|---:|---|---:|---:|---|---:|---|
| _example: strided_2mb_ | _read_ | _measured/skipped/..._ | _from `aggregates.patterns`_ | _from `statistics`_ | _requested/effective_ | _from `measurements[]`_ | _from `operation_plans[]`_ | _from `resolved_plan`_ | _from `operation_plans[]`_ | _from `operation_plans[]`_ | _from `operation_plans[]`_ | _noise/status reason_ |

Check that measured samples have positive duration and internally consistent exact payload accounting, sparse cases
report their effective worker reduction, and unavailable cases use explicit status rather than zero. As a stability
review target, investigate CV above 5% for sequential/64-byte stride or above 10% for sparse/random operations. Record
the observed result and conditions even when a target is missed; do not rewrite a noisy run as a pass.

### Latency analysis with TLB-locality control

```bash
# default locality mode (1 MB window)
memory_benchmark --cpu-latency --buffer-size 1024 --latency-samples 5000 --count 10 --output lat_default_1mb.json

# global random chain
memory_benchmark --cpu-latency --buffer-size 1024 --latency-samples 5000 --latency-tlb-locality-kb 0 --count 10 --output lat_global.json

# same in-box random pattern (good for prefetch-vs-TLB comparisons)
memory_benchmark --cpu-latency --buffer-size 1024 --latency-samples 5000 --latency-tlb-locality-kb 16 --latency-chain-mode same-random-in-box --count 10 --output lat_same_box.json
```

### Paired locality comparison (`--compare-locality`)

```bash
memory_benchmark --cpu-latency --compare-locality --count 3 --output lat_compare.json
```

This prints the target `P50` latency headlines plus the paired `Locality Comparison: 16KB P50=..., Global P50=..., Delta=...`
summary over `3` alternating-seed paired rounds per loop.

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
memory_benchmark --cpu-bandwidth --cache-size 4096 --threads 1 --count 5 --output bw_cache_4mb.json
memory_benchmark --cpu-latency --cache-size 4096 --count 5 --output lat_cache_4mb.json
```

### Cache-size sweep + trend plotting

```bash
./script-examples/latency_test_script.sh
python3 script-examples/plot_cache_percentiles.py script-examples/final_output.txt --metric median
```

### Built-in sweep JSON

```bash
memory_benchmark --cpu-latency --count 5 --sweep buffer-size=256,512,1024 --sweep latency-stride-bytes=64,256 --output latency_sweep.json
```

The command above creates six runs (`base_mode: "cpu_latency"`) and stores each run's CPU latency schema-1 JSON under
`runs[].result`. CPU bandwidth sweeps use `base_mode: "cpu_bandwidth"` and support `buffer-size`, `cache-size`, and
`threads`. TLB-analysis sweeps use the same envelope with `base_mode: "analyze_tlb"` and support `latency-stride-bytes`,
`latency-chain-mode`, and `tlb-density`. Core-to-core sweeps use the same envelope with
`base_mode: "analyze_core2core"` and support only `count` and `latency-samples`. LLM-memory sweeps use the same
schema-1 envelope with `base_mode: "llm_memory"`, embed each run's schema-2 result under `runs[].result`, and
support `context-tokens`, `kv-block-tokens`, `prompt-tokens`, and `attention-query-tile-tokens`.

---

## Understanding Console Output

Every successfully started direct mode and parameter sweep begins with one shared version, copyright, and GPL
banner before mode-specific configuration or status output. Nested runs within a sweep do not repeat it. Help output
retains the separate usage preamble; parse and preflight failures emit concise single-line `Error: ...` diagnostics on
stderr without dumping full usage text or the runtime banner.

The numbered result sections below describe result-producing modes, including the standalone CPU bandwidth, CPU latency,
patterns, GPU, and LLM profiles.

### 1) Configuration section

Shows active settings and detected hardware.

Important fields:

- Buffer size and active targets (`Target [Main Memory]`, `Target [L1 Cache]`, `Target [L2 Cache]`, or `Target [Custom Cache]`)
- Work policy (`automatic_calibration` or `explicit_fixed_work`), loop count, and sample/worker counts
- Base seed (`user` or `generated`) and non-cacheable hint status
- In `--cpu-latency`: page size, effective/requested chain mode, stride, TLB locality KB, and `compare_locality` (`enabled` or `disabled`)
- Detected or custom cache sizes

### 2) CPU bandwidth (`-B` / `--cpu-bandwidth`)

Prints `CPU Bandwidth Mode (schema v1)` followed by `--- CPU Bandwidth Results (<status>, <measured>/<planned> measured) ---`
and per-target (`Main Memory`, `L1 Cache`, `L2 Cache`, `Custom Cache`) lines for `Read`, `Write`, and `Copy`:

```text
  Main Memory:
    Read : P50=... GB/s (min=..., max=..., CV=..., MAD=..., n=...)
    Write: P50=... GB/s (min=..., max=..., CV=..., MAD=..., n=...)
    Copy : P50=... GB/s (min=..., max=..., CV=..., MAD=..., n=...)
```

Quality warning tokens (such as `environment-not-nominal`, `main-thread-qos-not-applied`, `worker-qos-not-applied`,
`madvise-random-not-applied`, `<target>-<op>-duration-<quality>`, `<target>-<op>-high-cv` when CV > 5.0%) are emitted
on stderr.

### 3) CPU latency (`-L` / `--cpu-latency`)

Prints `CPU Latency Mode (schema v1)` followed by `--- CPU Latency Results (<status>, <measured>/<planned> measured) ---`
and one line per active target (`L1 Cache`, `L2 Cache`, `Custom Cache`, `Main Memory`) plus optional `PooledSamples`
(when `--latency-samples > 0`) and optional `Locality Comparison` (when `--compare-locality` is passed):

```text
  L1 Cache    : P50=... ns (min=..., max=..., CV=..., MAD=..., n=...) | PooledSamples P50=... ns (P90=..., P99=..., n=...)
  L2 Cache    : P50=... ns (min=..., max=..., CV=..., MAD=..., n=...) | PooledSamples P50=... ns (P90=..., P99=..., n=...)
  Main Memory : P50=... ns (min=..., max=..., CV=..., MAD=..., n=...) | PooledSamples P50=... ns (P90=..., P99=..., n=...)
  Locality Comparison: 16KB P50=... ns, Global P50=... ns, Delta=... ns
```

The target `P50` headline comes from the continuous pointer-chase pass returning to node `0`. `PooledSamples` summarizes
the separate continuous sample-window pass and does not weight the headline. `Locality Comparison` summarizes `3 * loop_count`
paired rounds (`Delta = median(global) - median(16k)`), which is not an isolated page-table-walk penalty; use
`--analyze-tlb` for controlled translation analysis.

### 4) Cache targets

L1/L2 or custom cache sections are included automatically unless `--cache-size 0` omits them.

### 5) Pattern benchmark output (`-P` / `--patterns`)

Prints `Patterns Mode (schema v1)`, hardware/buffer/work-policy/seed/non-cacheable metadata, per-pattern active or
`skipped (<reason>)` geometry lines, and `--- Patterns Benchmark Results (<status>, <measured>/<planned> measured) ---`
with per-pattern `Read`, `Write`, and `Copy` lines:

```text
  Sequential Forward:
    Read : P50=... GB/s (min=..., max=..., CV=..., MAD=..., n=...)
    Write: P50=... GB/s (min=..., max=..., CV=..., MAD=..., n=...)
    Copy : P50=... GB/s (min=..., max=..., CV=..., MAD=..., n=...)
  Strided 2MiB:
    Read : P50=... GB/s (...% vs forward) (min=..., max=..., CV=..., MAD=..., n=...)   # or: skipped (buffer-too-small-for-stride-transition)
```

Deduplicated quality warning tokens (`environment-not-nominal`, `main-thread-qos-not-applied`, `worker-qos-not-applied`,
`madvise-random-not-applied`, `<pattern>-<op>-duration-<quality>`, `<pattern>-<op>-high-cv`) are emitted on `stderr`.
Unavailable operations print `unavailable (<reason>)`, never `0 GB/s`.

Do not treat a low strided or random result as proof of prefetch behavior, cache thrashing, or a TLB boundary. Pattern
mode does not isolate those mechanisms, and the console intentionally does not emit those diagnoses. Use
`--analyze-tlb` for controlled paired TLB analysis, and use matched buffer/stride/thread experiments for broader cache or
prefetch hypotheses.

### 6) Statistics and CV thresholds (`--count > 1`)

JSON `aggregates` include:

- `average`
- `median` (P50)
- `p90`, `p95`, `p99`
- `stddev` (sample standard deviation, `n - 1`)
- `coefficient_of_variation_pct` (CV)
- `median_absolute_deviation` (MAD)
- `min` / `max`

For noisy systems, prioritize median and P95/P99 rather than single fastest/slowest values.

`--cpu-bandwidth` emits a `high-cv` warning when CV exceeds `5.0%`, while `--cpu-latency` emits a `high-cv` warning
when CV exceeds `7.5%`. These thresholds are diagnostic only: the benchmark does not filter outliers or invalidate
`run_accepted`.

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

`--llm-memory` prints a separate report for CPU decode/prefill or an active Metal profile containing:

- backend, phase/work unit, KV layout, cacheable semantics, and exact active-weight/KV-read/KV-write bytes;
- for Metal, device/capability evidence, W/K/V segment counts, exact segment capacity, Tier 2 argument-buffer length,
  actual committed/known-peak/admitted memory, pipeline/grid geometry, authoritative GPU elapsed time, checksum,
  phase-neutral `kv_write` validation, and applicable canary status;
- decode visible context and its two-decimal traffic crossover, or prefill P/Q/C, prefix visits, causal pairs, and
  logical attention/FMA audit counts; decode-only crossover fields are not repurposed for prefill;
- for paged layout, `G`, blocks per sequence, terminal valid bytes, physical K/V lengths, padding, table size,
  permutation identity, and timed lookup/metadata counts; the report states that metadata is excluded from the GB/s
  numerator;
- up to one measured headline per weights-only, KV-only, and mixed scenario, with phase-specific work-unit latency and
  effective model-payload GB/s; JSON uses the backend-neutral fields `synthetic_work_unit_latency_seconds`,
  `synthetic_memory_work_units_per_second`, and `effective_model_payload_gb_s`, without calling any value tokens/s;
- warnings for incomplete position balance, non-nominal environment, QoS failure, cache-dominant working sets, high
  effective-model-payload CV, or duration outside the intended window. Console statistics include n, median, min/max,
  CV and MAD; JSON additionally retains descriptive P90/P95/P99, without treating them as confidence levels;
- an interpretation reminder that the payload is logical and memory-only, not physical DRAM traffic or model inference.

Paged suffix padding is capacity and validation evidence, not payload. A padding-canary or write/checksum mismatch makes
the affected measurement invalid and does not trigger a retry.

Missing, interrupted, invalid, or failed scenarios do not receive fabricated numeric console headlines; their exact
status, reason, and null observations remain in JSON, and command failure retains its diagnostic. The corresponding JSON
traffic classification uses version `llm-exact-weight-vs-kv-read-payload-v1`; `near_crossover` means exact equality
between weight and KV-read payload. It is not a tolerance band or a measured bottleneck diagnosis.

**Note:** CPU latency schema-1 `resolved_plan.chain_identities.<target>` records `node_count`, `stride_bytes`,
`chain_mode`, `tlb_locality_kb`, `chain_seed_uint64_decimal`, and `chain_permutation_sha256`. Standalone TLB schema 4
records buffer-relative chain diagnostics such as requested/effective/actual virtual-page counts, `pointer_nodes`,
`unique_cache_lines`, `spread_chain`, and `packed_chain`. These fields do not describe physical-page identities or
prove physical placement. They are retained in JSON, not the compact console report.

---

## JSON Output Format

Every result-producing direct mode and supported sweep can serialize their payload either to a real file
or, with the exact raw target `--output -`, once to stdout. The stdout transport does not wrap the result or change its
measurement schema; sweep stdout is one final envelope rather than a checkpoint stream. Direct CPU bandwidth, CPU latency, patterns,
and LLM files snapshot bounded completed-loop progress (`K=max(1,ceil(count/8))`) plus command terminal; sweep files
checkpoint the envelope after each attempted run without intra-run loop snapshots, and stdout emits one final document
or sweep envelope. See [API.md](API.md) for stream handling, process-status checks, current schema acceptance, and the
transport support matrix.

### CPU bandwidth JSON shape (schema 1)

```json
{
  "schema_version": 1,
  "mode": "cpu_bandwidth",
  "methodology_version": "cpu-bandwidth-v1-arm64-calibrated-seeded-balanced",
  "software": {
    "version": "<software-version>",
    "timestamp": "YYYY-MM-DDTHH:MM:SSZ"
  },
  "build_manifest": { "...": "provenance + binary_sha256" },
  "configuration": {
    "buffer_size_mb": 512,
    "custom_cache_size_kb": null,
    "requested_workers": 10,
    "available_workers": 10,
    "worker_source": "auto",
    "iterations": null,
    "work_policy": "automatic_calibration",
    "loop_count": 3,
    "non_cacheable": false,
    "base_seed_uint64_decimal": "424242",
    "seed_source": "user",
    "output_file": "cpu_bw.json",
    "argv": [ "./memory_benchmark", "--cpu-bandwidth", "..." ],
    "resolved_sources": { "...": "default|explicit" }
  },
  "resolved_plan": {
    "component_identities": {
      "version": "cpu-bandwidth-components-v1",
      "work_plan_version": "cpu-bandwidth-work-plan-v1",
      "buffer_pattern_version": "cpu-bw-buffer-pattern-v1",
      "read_checksum_version": "cpu-bw-read-checksum-v1",
      "run_policy_version": "bounded-loop-snapshots-v1"
    },
    "targets": { "main_memory": { "...": "..." } },
    "operation_plans": [ "..." ]
  },
  "memory_budget": { "...": "strict peak admission" },
  "calibration": { "excluded_from_results": true, "attempts": { "...": "..." } },
  "measurements": [ "..." ],
  "aggregates": {
    "targets": {
      "main_memory": {
        "read": { "accepted_measurement_ids": [0, 1, 2], "value": 105.2, "statistics": { "...": "..." }, "quality": { "...": "..." } },
        "write": { "...": "..." },
        "copy": { "...": "..." }
      }
    }
  },
  "status": "complete",
  "reason_code": "complete",
  "results_complete": true,
  "run_accepted": true,
  "interpretation": { "dram_residency": "unverified", "cache_residency": "unverified" },
  "operation_order_balance_complete": true,
  "target_order_balance_complete": true,
  "checkpoint_lifecycle": { "...": "..." }
}
```

### CPU latency JSON shape (schema 1)

```json
{
  "schema_version": 1,
  "mode": "cpu_latency",
  "methodology_version": "cpu-latency-v1-arm64-calibrated-seeded-balanced",
  "software": {
    "version": "<software-version>",
    "timestamp": "YYYY-MM-DDTHH:MM:SSZ"
  },
  "build_manifest": { "...": "provenance + binary_sha256" },
  "configuration": {
    "buffer_size_mb": 512,
    "custom_cache_size_kb": null,
    "iterations": null,
    "work_policy": "automatic_calibration",
    "loop_count": 3,
    "latency_samples": 1000,
    "latency_stride_bytes": 256,
    "latency_chain_mode": "auto",
    "latency_tlb_locality_kb": 1024,
    "compare_locality": false,
    "non_cacheable": false,
    "base_seed_uint64_decimal": "424242",
    "seed_source": "user",
    "output_file": "cpu_lat.json",
    "argv": [ "./memory_benchmark", "--cpu-latency", "..." ]
  },
  "resolved_plan": {
    "component_identities": {
      "version": "cpu-latency-components-v1",
      "work_plan_version": "cpu-latency-work-plan-v1",
      "chain_generator_version": "splitmix64-fisher-yates-rejection-v1",
      "chain_version": "cpu-latency-chain-v1",
      "run_policy_version": "bounded-loop-snapshots-v1"
    },
    "chain_identities": { "l1": { "...": "..." }, "l2": { "...": "..." }, "main_memory": { "...": "..." } },
    "target_plans": [ "..." ],
    "frozen_plan_refs": { "...": "..." }
  },
  "locality_comparison": null,
  "aggregates": {
    "targets": {
      "main_memory": {
        "accepted_measurement_ids": [ "..." ],
        "value": 84.2,
        "statistics": { "...": "..." },
        "quality": { "...": "..." },
        "pooled_sample_distribution": {
          "semantics": "pooled-separate-sample-window-distribution",
          "values_ns": [ "..." ],
          "statistics": { "...": "..." }
        }
      }
    },
    "locality_comparison": null
  },
  "status": "complete",
  "reason_code": "complete",
  "results_complete": true,
  "run_accepted": true,
  "interpretation": { "dram_residency": "unverified", "cache_residency": "unverified" },
  "target_order_balance_complete": true,
  "checkpoint_lifecycle": { "...": "..." }
}
```

Both schema-1 artifacts record canonical decimal-string `uint64` seeds, payloads, and `execution.timing.cpu_raw` Mach
tick boundaries, nullable unavailable metrics (`null`, never `0`), bounded `K=max(1,ceil(count/8))` file checkpoints,
raw `configuration.output_file` (`"-"` on stdout, `""` inside nested sweeps), and separate `results_complete` and
`run_accepted` predicates. Consumers must require `schema_version == 1`, the expected `mode` and `methodology_version`,
`status == "complete"`, `results_complete == true`, and `run_accepted == true`.

The bundled `script-examples/` scripts accept compatible producer releases. Before reading CPU bandwidth or CPU latency
metrics, they check `schema_version == 1`, `mode`, `methodology_version`, non-empty `software.version` (retained as
provenance only), string `configuration.output_file`, `status == "complete"`, `results_complete == true`,
`run_accepted == true`, and the expected types of consumed fields. Legacy standard schema-3/2 and unversioned historical
JSON layouts are unsupported and are not translated.

### Pattern benchmark JSON shape (schema 1)

```json
{
  "schema_version": 1,
  "mode": "patterns",
  "methodology_version": "patterns-v1-arm64-phase-calibrated-seeded-balanced",
  "software": {
    "version": "<software-version>",
    "timestamp": "YYYY-MM-DDTHH:MM:SSZ"
  },
  "build_manifest": { "...": "provenance + binary_sha256" },
  "configuration": {
    "buffer_size_mb": 512,
    "requested_workers": 10,
    "available_workers": 10,
    "worker_source": "auto",
    "iterations": null,
    "work_policy": "automatic_calibration",
    "loop_count": 3,
    "non_cacheable": false,
    "base_seed_uint64_decimal": "424242",
    "seed_source": "user",
    "output_file": "patterns.json",
    "argv": [ "./memory_benchmark", "--patterns", "..." ],
    "resolved_sources": { "buffer_size": "default|explicit|sweep", "workers": "auto|explicit|sweep", "iterations": "auto|explicit", "seed": "generated|explicit" }
  },
  "resolved_plan": {
    "valid": true,
    "reason_code": "valid",
    "plan_identity": "patterns-work-plan-v1|...",
    "component_identities": {
      "version": "patterns-components-v1",
      "work_plan_version": "patterns-work-plan-v1",
      "buffer_pattern_version": "patterns-buffer-pattern-v1",
      "read_checksum_version": "patterns-read-checksum-v1",
      "random_index_generator_version": "splitmix64-coprime-fisher-yates-v1",
      "run_policy_version": "bounded-loop-snapshots-v1"
    },
    "patterns": {
      "sequential_forward": { "...": "..." },
      "sequential_reverse": { "...": "..." },
      "strided_64": { "...": "..." },
      "strided_4096": { "...": "..." },
      "strided_16384": { "...": "..." },
      "strided_2mb": {
        "enabled": true,
        "status": "active",
        "reason_code": "active",
        "access_size_bytes": 32,
        "stride_bytes": 2097152,
        "phase_period_passes": 65536,
        "native_page_size_bytes": 16384,
        "stride_equals_native_page_size": false,
        "large_page_backing_verified": false,
        "large_page_backing_status": "not-verified",
        "effective_workers": 10,
        "worker_spans": [ "..." ]
      },
      "random": { "...": "includes random_workload.random_indices_sha256" }
    },
    "resources": { "src": { "...": "read_only_protected: true" }, "dst": { "...": "..." }, "random_index_table_bytes": "...", "random_validation_index_bytes": "..." },
    "operation_plans": [
      {
        "plan_ref": 0,
        "pattern": "strided_2mb",
        "operation": "read",
        "passes": 64,
        "explicit_iterations": false,
        "total_accesses": "...",
        "exact_payload_bytes": "...",
        "accesses_per_pass": "...",
        "min_accesses_per_pass": "...",
        "max_accesses_per_pass": "...",
        "distinct_address_count": "...",
        "logical_working_set_bytes": "...",
        "completed_phase_cycles": 0,
        "has_untouched_dst_bytes": true,
        "op_seed_uint64_decimal": "...",
        "expected_checksum": { "status": "available", "expected_worker_checksums": [ "..." ], "expected_run_checksum": { "...": "..." } }
      }
    ],
    "frozen_plan_refs": { "strided_2mb": { "read": 0, "write": 1, "copy": 2 } }
  },
  "memory_budget": { "...": "strict peak admission" },
  "calibration": { "excluded_from_results": true, "attempts": { "...": "..." } },
  "measurements": [
    {
      "measurement_id": 0,
      "plan_ref": 0,
      "pattern": "strided_2mb",
      "operation": "read",
      "loop_index": 0,
      "pattern_order_position": 0,
      "operation_order_position": 0,
      "status": "measured",
      "reason_code": "measured",
      "attempted": true,
      "completed_passes": 64,
      "completed_exact_payload_bytes": "...",
      "bandwidth_gb_s": 12.34,
      "execution": { "timing": { "cpu_raw": { "...": "..." } }, "validation": { "checks": [ "6 ordered checks" ] } },
      "checksum": { "status": "valid", "checksum_valid": true, "actual_worker_checksums": [ "..." ], "actual_run_checksum": { "...": "..." } }
    }
  ],
  "aggregates": {
    "patterns": {
      "strided_2mb": {
        "read": { "status": "measured", "reason_code": "measured", "accepted_measurement_ids": [ "..." ], "value": 12.34, "relative_to_forward_pct": 10.5, "statistics": { "...": "..." }, "quality": { "...": "..." } },
        "write": { "...": "..." },
        "copy": { "...": "..." }
      }
    }
  },
  "status": "complete",
  "reason_code": "complete",
  "results_complete": true,
  "run_accepted": true,
  "interpretation": { "dram_residency": "unverified", "cache_residency": "unverified", "large_page_backing_verified": false },
  "diagnostic": null,
  "interruption_requested": false,
  "operation_order_balance_complete": true,
  "pattern_order_balance_complete": false,
  "seeds": { "base_seed_uint64_decimal": "424242", "source": "user", "domain_seeds": { "...": "..." } },
  "counters": { "...": "..." },
  "checkpoint_lifecycle": { "...": "..." },
  "loop_records": [ "..." ],
  "environment": { "...": "..." },
  "quality_warnings": []
}
```

This is a structure illustration, not a recorded hardware measurement. All 26 top-level keys match the sibling `cpu_bandwidth` schema-1 layout (with `pattern_order_balance_complete` in place of `target_order_balance_complete`). All seven canonical patterns always appear in `resolved_plan.patterns.<kind>` and `aggregates.patterns.<kind>.<op>`. When `buffer_bytes < 2 MiB + 32 B` (for example 1 MiB or 2 MiB buffers), `strided_2mb` is `status: "skipped"` (`reason_code: "buffer-too-small-for-stride-transition"`, `frozen_plan_refs.strided_2mb.{read,write,copy}: null`) and omitted from `calibration.attempts`, `measurements[]`, and `loop_records[]`.

`requested_workers` preserves the CLI/default worker count (`worker_source: "auto"` or `"user"`), while `effective_workers` records the per-pattern clamped worker count. For phase-rotated strided records, `accesses_per_pass` is the phase-zero count while `total_accesses` and `exact_payload_bytes` (in `operation_plans[]`) and `completed_exact_payload_bytes` (in `measurements[]`) give exact totals. `strided_2mb` names a 2 MiB virtual-address stride (`large_page_backing_status: "not-verified"`, `large_page_backing_verified: false`).

Both file and stdout transports emit this same schema-1 payload (`configuration.output_file` preserves `"-"` on stdout and `""` in nested sweeps). Command acceptance requires `schema_version == 1`, `mode == "patterns"`, `methodology_version == "patterns-v1-arm64-phase-calibrated-seeded-balanced"`, `status == "complete"`, `results_complete == true`, and `run_accepted == true`, while consuming one aggregate metric additionally requires `aggregates.patterns.<kind>.<op>.status == "measured"` and a non-null `value`.

### GPU bandwidth JSON shape

GPU file and stdout output use the same separate top-level schema. It must not be sent to a standard-schema parser merely
because it contains read/write/copy values. The following valid JSON object is an abbreviated field-selection fragment
representing a complete automatic stdout run; it deliberately omits, rather than empties, the populated nested evidence
arrays:

```json
{
  "software_version": "<software-version>",
  "version": "<software-version>",
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

LLM file and stdout output use the same separate top-level schema 2. Select by `mode`, `schema_version`, and exact
backend/phase/layout/methodology. This replaces unpublished schema 1 with an explicit schema and methodology bump;
there is no compatibility alias or fallback reader. The [API field map](API.md#llm-schema-1-to-schema-2-field-map)
is the normative old-to-new mapping.

This structural projection comes from a complete one-loop CPU paged-decode run (1 MiB weights, one layer/head/worker,
head dimension 8, one-byte KV elements, context 32, block size 16, seed 42, one work unit). It keeps all three plan
reference targets but displays only measurement ID 1 and selected nested fields. Geometry, plan identities, expected
worker vectors, other measurements/calibration and full statistics are omitted. It is **not a complete acceptable
artifact** and must not be passed to a result-acceptance predicate. The displayed checksum states are observed fixture
data; they do not independently prove the implementation. The displayed unavailable `build_manifest` illustrates
the fallback for a caller supplying no manifest; normal CLI runs populate the available build and binary provenance.

```json
{
  "schema_version": 2,
  "mode": "llm_memory",
  "backend": "cpu",
  "phase": "decode",
  "kv_layout": "paged",
  "methodology_version": "llm-memory-v2-cpu-decode-paged",
  "status": "complete",
  "results_complete": true,
  "run_accepted": true,
  "scenario_order_balance_complete": false,
  "resolved_plan": {
    "component_identities": {
      "run_policy_version": "llm-run-policy-bounded-loop-snapshots-v1"
    },
    "scenario_plans": [
      {
        "scenario": "weights_only",
        "work_units": 1,
        "model_ref": "resolved_plan"
      },
      {
        "scenario": "kv_only",
        "work_units": 1,
        "model_ref": "resolved_plan",
        "expected_checksum": {
          "status": "available",
          "reason_code": "valid",
          "expected_run_checksum": {
            "state_a_uint64_decimal": "13746298198528704984",
            "state_b_uint64_decimal": "13236122334346434487"
          }
        }
      },
      {
        "scenario": "mixed",
        "work_units": 1,
        "model_ref": "resolved_plan"
      }
    ],
    "frozen_plan_refs": {
      "weights_only": 0,
      "kv_only": 1,
      "mixed": 2
    }
  },
  "measurements": [
    {
      "measurement_id": 1,
      "plan_ref": 1,
      "scenario": "kv_only",
      "status": "measured",
      "completed_work_units": 1,
      "completed_effective_model_payload_bytes": "528",
      "checksum": {
        "status": "valid",
        "checksum_valid": true,
        "actual_run_checksum": {
          "state_a_uint64_decimal": "13746298198528704984",
          "state_b_uint64_decimal": "13236122334346434487"
        }
      },
      "execution": {
        "validation": {
          "checks": [
            {
              "kind": "post-validation-structure",
              "applicable": true,
              "evaluated": true,
              "valid": true,
              "reason_code": "valid"
            },
            {
              "kind": "kv-append-final",
              "applicable": true,
              "evaluated": true,
              "valid": true,
              "reason_code": "valid"
            },
            {
              "kind": "kv-padding-canary",
              "applicable": false,
              "evaluated": null,
              "valid": null,
              "reason_code": "not-applicable"
            }
          ]
        }
      }
    }
  ],
  "aggregates": {
    "scenarios": {
      "kv_only": {
        "accepted_measurement_ids": [
          1
        ],
        "observed_cv_classification": "insufficient-samples",
        "cv_warning_threshold_pct": 5.0
      }
    }
  },
  "checkpoint_lifecycle": {
    "checkpoint_failed": false,
    "observation_point": "before-current-snapshot-preparation",
    "prior_file_writer_attempts": 0,
    "prior_successful_file_writes": 0,
    "current_request": "terminal",
    "current_persistence_success": null,
    "snapshot_interval_loops": 1,
    "checkpoint_policy": "bounded-loop-snapshots",
    "file_checkpoint_failure_is_terminal_and_not_retried": true,
    "stdout_intermediate_checkpoints_are_lazy": true
  },
  "build_manifest": {
    "manifest_version": 1,
    "status": "unavailable",
    "reason_code": "build-provenance-not-provided",
    "binary_sha256": null,
    "git_commit": null,
    "git_dirty": null,
    "compiler": null,
    "compile_flags": null,
    "link_flags": null,
    "target_arch": null,
    "sdk": null,
    "min_os": null
  }
}
```

Schema 2 ownership and interpretation:

- All eight exact selectors are `llm-memory-v2-{cpu|metal}-{decode|prefill}-{contiguous|paged}`, with the canonical
  `run_policy_version` shown above. These braces describe the eight combinations, not a prefix-matching acceptance rule.
- `resolved_plan` owns model geometry, resources, layout/permutation and component identities once.
  `scenario_plans[]` owns each unique scenario/T/explicit shape, its seed, work kind, per-work-unit byte/lookup values,
  planned totals and cold expected checksum once. Its order is scenario, increasing T, then false/true explicit policy.
  Frozen refs, measurements and calibration attempts reference that same document's array. Public indexes can change
  between snapshots; unknown, out-of-range or wrong-scenario references are rejected. Null is allowed for unresolved plans.
- Measurement `measurement_id` is the schedule-array index. `elapsed_seconds`, completed work/payload, actual checksums,
  timing/lifecycle observations and mutable quality remain measurement-owned. Work kind and planned totals come from
  `plan_ref`; total plan names have no `planned_` prefix. CPU `completion_derivation: "accepted-plan-derived"` is not a
  hardware work counter. Worker/QoS/calibration-index/working-set and mixed-fraction metadata remain available.
- Canonical `expected_checksum` has status/reason, `expected_worker_checksums`, and `expected_run_checksum`.
  CPU worker expectations have weight/K/V domains and exactly the effective worker count. Metal worker expectations
  are null and its run checksum contains W/K/V dual-mod32 lanes. Available expectation is not observed validation.
  Runtime actuals stay under measurement `checksum`; compact calibration retains only its existing actual-run witness.
- `execution.validation.checks[]` describes structure, phase/scenario write or unchanged-append, and padding separately.
  The five kinds are `post-validation-structure`, `kv-append-final`, `kv-prefill-final-samples`, `kv-append-unchanged`,
  `kv-padding-canary`. Inapplicable evaluated/valid are null; applicable unresolved valid is null; an observed mismatch
  remains false. A required unevaluated check prevents acceptance. Prefill remains sampled, and checksum agreement does
  not replace final-state checks. CPU paged decode weights-only checks unchanged append; CPU paged prefill weights-only
  checks structure/padding; Metal weights-only has no write/padding check. Padding applies only to a physical suffix.
- Run statuses are `not_started`, `complete`, `partial`, `interrupted`, `unsupported`, `failed`; measurement statuses
  are `not_run`, `measured`, `interrupted`, `invalid`, `failed`. Unsupported never authorizes backend fallback.
  Unavailable execution evidence has unavailable status and null worker/timing/checksum observations.
- Indexes, bounded configuration values and work units are integer JSON numbers; potentially large bytes, lookup,
  seed and checksum quantities are canonical unsigned decimal strings. Known zero is not null. Metal workers are null.
  Prefill uses `geometry.prefill`, decode uses `geometry.decode`; the other is null. Paged resources distinguish logical
  K/V from full physical blocks, suffix padding and table bytes; unsupported-before-preparation retains admitted geometry
  with null runtime permutation. CPU paged prefill has both backend `prefill` and `paged` evidence objects.
- Theoretical causal/attention/FMA values live in `resolved_plan.model_context.prefill` with individual `*_reason_code`:
  decimal value/`valid` or null/`arithmetic-overflow`. `h_q` is query-head count; tile `Q` determines executed prefix reads.
  No FMA is executed. Actual byte/lookup/size overflows still reject work. Decode model-context prefill is null.
- Memory admission covers full resources, rounded allocation, canonical identity/expected capacity, actual retained
  checksums, calibration, accepted-ID maps, exact-stat workspaces and the simultaneous DOM/serialized-string peak.
  Metal paged expected reconstruction also reserves its temporary table/validation/hash workspace. Available memory is
  runtime evidence, not part of immutable work identity. Reducing snapshot frequency does not remove peak admission.
- `run_accepted` requires complete status, every planned measurement measured, accepted required runtime evidence and
  no known command/checkpoint error. `results_complete` only describes that measured population. Correct count one can
  be accepted while balance is false. Quality, CV, duration and environment remain separate comparison criteria.
  Compare the same backend/profile, phase geometry, layout/permutation, frozen work, seed, component identities and
  run policy, including conditioning and file/stdout checkpoint cadence. Inspect accepted sample count and repeat
  alternating process comparisons; artifact acceptance alone does not establish a machine or methodology speed
  difference. Thermal-state/Low Power Mode start/end snapshots are instantaneous OS observations, not continuous
  temperatures, cache residency or guaranteed CPU affinity. Nominal endpoints cannot exclude intervening changes;
  unavailable observations remain missing evidence.
- One `accepted_measurement_ids` population feeds all three metrics. Derive each rate before statistics; exact median,
  MAD and linearly interpolated P90/P95/P99 are prepared at snapshots/terminal. Empty populations are null; n=1 stddev
  and positive-mean CV are zero. Classification is `insufficient-samples` for n<3, `undefined` for undefined CV,
  `above-threshold` for payload CV strictly over 5%, otherwise `below-threshold`; equality is below-threshold.
  These descriptive percentiles are not confidence levels. Default console shows n/median/min/max/CV/MAD.
- `build_manifest` records build-time Git state, compiler, compile/link flags, target architecture, SDK and deployment
  target. The command captures the executable's `binary_sha256` once before tasks. Missing build fields or a failed
  binary read leave partial provenance with explicit reasons; a caller supplying no manifest retains the unavailable
  fallback shown above. These fields bind available artifacts, not signed execution attestation.
- CPU measurements and excluded calibration attempts retain original Mach boundaries in `execution.timing.cpu_raw`:
  `start_ticks`, `stop_ticks` and `delta_ticks` are unsigned decimal strings; `timebase_numer` and `timebase_denom` are
  integer JSON numbers. A missing snapshot is null; older schema-2 artifacts may omit this optional field. Metal retains
  GPU timestamps and null CPU raw evidence. See the [API contract](API.md#result-schemas-and-completion) for duration
  reconstruction and provenance availability rules.

For N planned loops, files snapshot every `K=max(1,ceil(N/8))` completed loops, at most eight progress snapshots plus
terminal: maximum 4/7/9 normal snapshots for N=3/12/48. Abrupt termination can lose up to `3K` truly completed attempts,
including during replacement; unstarted placeholders do not count. No file is promised before the first successful
snapshot. Graceful SIGINT retains the completed prefix and interrupted tail if terminal writing succeeds; SIGKILL
cannot promise terminal output. Atomic replacement is not power-loss durability.

Prior writer counts are observed before the current snapshot preparation; its own persistence remains null, with
`current_request` distinguishing progress, terminal and `late-command-error-correction`. Builder failure is not a
writer attempt, and stdout/disabled no-ops are not file successes. A failed checkpoint is terminal with no final-write
retry. One later command exception after successful terminal persistence may write one corrective failure snapshot.
A previous file cannot attest to a later failure: retain process outcome and terminal context. Stdout builds one final
DOM; omitted/empty output disables JSON and builds none. Automatic calibration does not invent a default filename.

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
    "base_mode": "cpu_latency",
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
      "result": { "...": "CPU bandwidth, CPU latency, pattern, TLB, core-to-core, or LLM-memory JSON payload" }
    }
  ],
  "execution_time_sec": 123.4,
  "timestamp": "YYYY-MM-DDTHH:MM:SSZ",
  "version": "<software-version>"
}
```

### CPU latency sample-window & `--compare-locality` structure

In `cpu_latency` schema 1, each measured entry in `measurements[]` preserves `final_pointer_verification`, optional
`sample_windows` (`null` when `--latency-samples 0`), and `execution.validation.checks[]`. When `--compare-locality` is
enabled, top-level `locality_comparison.rounds[]` records the `3 * loop_count` paired rounds and
`aggregates.locality_comparison` summarizes the medians:

```json
"aggregates": {
  "targets": {
    "main_memory": {
      "accepted_measurement_ids": [1, 3, 5],
      "value": 84.2,
      "statistics": { "median": 84.2, "coefficient_of_variation_pct": 0.8, "median_absolute_deviation": 0.5 },
      "pooled_sample_distribution": {
        "semantics": "pooled-separate-sample-window-distribution",
        "values_ns": [83.1, 84.0, 85.2],
        "statistics": { "median": 84.0, "p95": 85.08 }
      }
    }
  },
  "locality_comparison": {
    "locality_16k_latency_ns": 23.1,
    "global_random_latency_ns": 91.4,
    "locality_latency_delta_ns": 68.3
  }
}
```

An interrupted, invalid, or failed measurement has `latency_ns: null` and an explicit `reason_code`. `locality_latency_delta_ns`
is `global_random_latency_ns - locality_16k_latency_ns` (`median(global) - median(16k)`) across the paired rounds and must
not be interpreted as an isolated page walk.

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

### Pattern keys (schema 1)

Canonical patterns under `resolved_plan.patterns.<kind>` and `aggregates.patterns.<kind>`:

- `sequential_forward`
- `sequential_reverse`
- `strided_64`
- `strided_4096`
- `strided_16384`
- `strided_2mb`
- `random`

Each `resolved_plan.patterns.<kind>` holds pattern geometry, `effective_workers`, `worker_spans[]`, and (for `random`) `random_workload`. Each `aggregates.patterns.<kind>` contains `read`, `write`, and `copy` entries with explicit `status`/`reason_code`, `accepted_measurement_ids`, nullable `value` (median P50 GB/s), `relative_to_forward_pct`, `statistics`, and `quality` (with `cv_warning_threshold_pct` `5.0` or `10.0`).

### Useful JSON inspection commands

Capture any supported machine-output command before applying the schema-specific checks below:

```bash
memory_benchmark --cpu-bandwidth --buffer-size 512 --count 5 --output - \
  >cpu_bw.json 2>cpu_bw.log

memory_benchmark --cpu-latency --buffer-size 512 --count 5 --compare-locality --output - \
  >cpu_lat.json 2>cpu_lat.log

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
python3 -m json.tool cpu_bw.json

# Main memory read median (GB/s)
jq '.aggregates.targets.main_memory.read.statistics.median' cpu_bw.json

# Reject incomplete or unaccepted CPU bandwidth output
jq -e 'select(.schema_version == 1 and .mode == "cpu_bandwidth" and
              .methodology_version == "cpu-bandwidth-v1-arm64-calibrated-seeded-balanced" and
              (.configuration.output_file | type) == "string" and
              .status == "complete" and .results_complete == true and
              .run_accepted == true)' cpu_bw.json

# Main memory latency P95 from the pooled, separate sample pass
jq '.aggregates.targets.main_memory.pooled_sample_distribution.statistics.p95' cpu_lat.json

# Paired locality delta median (requires --compare-locality)
jq '.aggregates.locality_comparison.locality_latency_delta_ns' cpu_lat.json

# Reject incomplete or unaccepted CPU latency output
jq -e 'select(.schema_version == 1 and .mode == "cpu_latency" and
              .methodology_version == "cpu-latency-v1-arm64-calibrated-seeded-balanced" and
              (.configuration.output_file | type) == "string" and
              .status == "complete" and .results_complete == true and
              .run_accepted == true)' cpu_lat.json

# Pattern random read median and status
jq '.aggregates.patterns.random.read | {status, median: .value, statistics}' patterns.json

# Reject incomplete or unaccepted pattern output
jq -e 'select(.schema_version == 1 and .mode == "patterns" and
              .methodology_version == "patterns-v1-arm64-phase-calibrated-seeded-balanced" and
              (.configuration.output_file | type) == "string" and
              .status == "complete" and .results_complete == true and
              .run_accepted == true)' patterns.json

# Pattern strided_2mb geometry, phase-count semantics, and exact totals
jq '{pattern: .resolved_plan.patterns.strided_2mb,
     plans: [.resolved_plan.operation_plans[] | select(.pattern == "strided_2mb")]}' patterns.json

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

# Consume producer acceptance for CPU paged-decode schema 2; quality is separate.
jq -e '. as $root | .mode == "llm_memory" and .schema_version == 2 and
       .backend == "cpu" and .phase == "decode" and .kv_layout == "paged" and
       .methodology_version == "llm-memory-v2-cpu-decode-paged" and
       .resolved_plan.layout.kv_block_tokens == 16 and
       (.resolved_plan.layout.permutation_sha256 |
        type == "string" and test("^[0-9a-f]{64}$")) and
       .resolved_plan.component_identities.run_policy_version ==
         "llm-run-policy-bounded-loop-snapshots-v1" and
       .status == "complete" and .results_complete == true and .run_accepted == true and
       (.resolved_plan.scenario_plans | type) == "array" and
       (.measurements | type) == "array" and
       .counters.planned_measurements > 0 and
       (.measurements | length) == .counters.planned_measurements and
       all(.measurements[];
           .status == "measured" and (.plan_ref | type) == "number" and
           .plan_ref >= 0 and .plan_ref == (.plan_ref | floor) and
           .plan_ref < ($root.resolved_plan.scenario_plans | length) and
           $root.resolved_plan.scenario_plans[.plan_ref].scenario == .scenario)' llm_memory.json

# Inspect the accepted document, joining each reference against the same root.
jq '. as $root | .measurements[] as $m |
  $m.plan_ref as $ref |
  if ($ref | type) != "number" then error("missing numeric plan_ref")
  elif $ref < 0 or $ref != ($ref | floor) or
       $ref >= ($root.resolved_plan.scenario_plans | length) then error("invalid plan_ref")
  elif $root.resolved_plan.scenario_plans[$ref].scenario != $m.scenario then error("scenario mismatch")
  else $root.resolved_plan.scenario_plans[$ref] as $plan |
    {scenario: $m.scenario, status: $m.status, reason_code: $m.reason_code,
     work_unit_kind: $plan.work_unit_kind, planned_work_units: $plan.work_units,
     planned_task_accounted_bytes: $plan.task_accounted_bytes,
     layout_metadata_lookup_count_per_work_unit: $plan.layout_metadata_lookup_count_per_work_unit,
     layout_metadata_read_bytes_per_work_unit: $plan.layout_metadata_read_bytes_per_work_unit,
     synthetic_work_unit_latency_seconds: $m.synthetic_work_unit_latency_seconds,
     effective_model_payload_gb_s: $m.effective_model_payload_gb_s,
     checksum_valid: $m.checksum.checksum_valid}
  end' llm_memory.json
```

---

## Visualization Scripts

Plotting requires Python 3 and `matplotlib`; the M4/M5 comparison script additionally requires `numpy`:

```bash
python3 -m pip install matplotlib numpy
```

The bundled CPU script-examples accept outputs from compatible software releases. Before reading a selected metric path,
each requires a non-empty `software.version` string as provenance, `schema_version == 1`, `mode` (`"cpu_bandwidth"` or
`"cpu_latency"`), the exact methodology identity (`cpu-bandwidth-v1-arm64-calibrated-seeded-balanced` or
`cpu-latency-v1-arm64-calibrated-seeded-balanced`), `status == "complete"`, `results_complete == true`, `run_accepted == true`,
a string `configuration.output_file`, and the expected types for the fields it consumes. Exact software-version equality
is not a compatibility condition. These scripts do not translate legacy standard schema-3/2, unversioned historical JSON,
other modes, or other methodology identities. The separately governed `plot_analyzetlb.py` retains its own TLB-history
policy. CPU plotters do not accept GPU schema 1.

### Independent CPU artifact verifiers (`verify_cpu_*_result.py`, `verify_patterns_result.py`)

Verify saved `-B` (`cpu_bandwidth`), `-L` (`cpu_latency`), or `-P` (`patterns`) schema-1 direct artifacts or schema-1 sweep envelopes
independently without executing the benchmark:

```bash
python3 script-examples/verify_cpu_bandwidth_result.py cpu_bw.json --binary ./memory_benchmark --require-raw-timing
python3 script-examples/verify_cpu_latency_result.py cpu_lat.json --binary ./memory_benchmark --require-raw-timing
python3 script-examples/verify_patterns_result.py patterns.json --binary ./memory_benchmark --require-raw-timing
make test-patterns-verifier
make test-cpu-verifiers
```

`--binary PATH` hashes the binary against `build_manifest.binary_sha256` without running it; `--require-raw-timing`
rejects missing `cpu_raw` Mach ticks; `verify_cpu_latency_result.py` also accepts `--max-chain-nodes N` (default
`8,388,608`); and `verify_patterns_result.py` also accepts `--max-random-indices N` (default `1,000,000`). Exit `0`
means consistent and accepted (`run_accepted: true`), exit `1` means inconsistent or consistently unaccepted, and exit `2`
means unsupported schema/methodology or bounded resource/evidence limit. See
[API.md](API.md#independent-cpu-artifact-verifiers) for full details.

### `script-examples/latency_test_script.sh`

What it does:

- Invokes `--cpu-latency` across multiple custom cache sizes and `--latency-tlb-locality-kb` values
- Writes per-run JSON files under `script-examples/tmp/`
- Extracts `.aggregates.targets.custom.pooled_sample_distribution.statistics` from complete, accepted `cpu_latency`
  schema-1 files into `script-examples/final_output.txt`
- Uses `jq` for that local sanity check and extraction when available, with Python 3 as the fallback
- Clears `tmp` after extraction
- Returns a non-zero status after cleanup if any benchmark failed, an expected output is missing, or a schema-1 result
  is incomplete, unaccepted, or cannot be parsed

The script prefers the executable built at the repository root, falls back to `memory_benchmark` from `PATH` when that
file is unavailable, and honors an explicit `BENCHMARK_CMD=/path/to/memory_benchmark` override. Every selected producer
must emit complete, accepted `cpu_latency` schema 1; incompatible output is rejected before metric extraction.

`latency_test_script_stride_tlb.sh` applies the same local-binary/override policy, runs `--cpu-latency` across cache size,
configured locality, and stride, uses embedded Python 3 to verify schema-1 contract acceptance and extract
`pooled_sample_distribution.statistics`, `resolved_plan.chain_identities.custom.node_count`, and
`environment.page_size_bytes`, retains its timestamped JSON files, and returns non-zero unless every planned run produces
one complete CSV row.

### CPU comparison and hierarchy plotters

Both plotters require explicit compatible schema-1 JSON inputs (`--metric` defaults to `median`):

```bash
python3 script-examples/plot_M4vsM5_benchmark_comparison.py \
  --m4-bandwidth-file m4-bw.json --m4-latency-file m4-lat.json \
  --m5-bandwidth-file m5-bw.json --m5-latency-file m5-lat.json
python3 script-examples/plot_benchmark-memory-latency-hierarchy.py \
  --file cpu-latency-compare-locality.json
```

`plot_M4vsM5_benchmark_comparison.py` takes four explicit inputs (also aliased as `--m4-bandwidth`, `--m4-latency`,
`--m5-bandwidth`, `--m5-latency`) so bandwidth (`mode: "cpu_bandwidth"`) and latency (`mode: "cpu_latency"` with
`--compare-locality`) are verified independently and matched by `environment.processor_name`.
`plot_benchmark-memory-latency-hierarchy.py` takes a `cpu_latency` schema-1 JSON generated with `--compare-locality`
(reading `aggregates.targets.{l1,l2}` and `aggregates.locality_comparison` for `median` or `locality_comparison.rounds`
for other `ALLOWED_METRICS`), or an explicit `-L --compare-locality` console-text file (for `--metric median` only).

There are no archived JSON defaults. The hierarchy plotter's separate console-text parser recognizes current `-L`
console labels only and provides neither JSON schema compatibility nor support for historical label spellings.

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
0.53.7 result, not a guaranteed current-methodology baseline. Historical pattern files from earlier methodology
versions are not a stability baseline for current patterns schema 1 and should not be compared numerically with
`patterns-v1-arm64-phase-calibrated-seeded-balanced` results without accounting for the methodology change.

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
  measurement to be measured plus `run_accepted: true`. Apply balance/CV/environment requirements as a separate
  comparison-quality policy; correct one-loop runs can pass producer acceptance.
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
  profile omits Transformer compute, model/framework execution, ANE work, GPU work outside the defined Metal
  kernels, and compute-memory overlap.
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

`--cpu-bandwidth`, `--cpu-latency`, `--patterns`, and `--gpu-bandwidth` enforce strict `memory_budget` preflight checks
and reject oversized requests (`memory-budget-exceeded`) without silently truncating the working set. Only legacy
`BenchmarkConfig`-based paths (`--analyze-tlb` candidate admission) apply `BenchmarkConfig` capping rules.

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

Parser/configuration errors, one-work-unit payload-limit failures, output-session initialization or backend-factory
errors, work-plan/JSON-output-peak/memory-budget rejection, CPU paged block-table mapping/materialization/protection
failures, and CPU timer creation (`timer-unavailable`) occur before runner-result initialization. A stdout target is
intentionally empty (and no direct JSON file is created) on these paths; use the process status and stderr reason. Once
the work plan is finalized, backend capability checks (`unsupported`), backend initialization (other than
`timer-unavailable`), plan resolution, resource preparation (weight/K/V mapping, initialization, descriptor preparation,
and Metal compilation/pipeline/allocation/upload/validation), and post-plan setup exceptions emit terminal schema-2 JSON
on non-empty file or stdout targets while returning a non-zero exit status. Check
that all common model options and the selected phase's geometry options are present (directly or via `--sweep`, never
both), all serialized integer configuration inputs (and active `--sweep-max-runs`) fit within `2^53 - 1` (with
`--count <= floor((2^53 - 1) / 3)`), query heads are divisible by KV
heads, and each scenario fits the backend work-unit cap and 64 GiB task-accounted-byte limit. For paged layout, also
require one positive power-of-two `--kv-block-tokens` value (or `--sweep kv-block-tokens=...`) at or below `UINT32_MAX`;
for contiguous layout, remove that option. Confirm that the
page-rounded weight, physical K/V, optional table, transient, and auxiliary peak fits the reported available-memory
policy.

CPU LLM results retain the original Mach tick boundaries and timebase for duration reconstruction.
The build embeds compiler, flags, SDK/deployment target and Git provenance; the command hashes the
executable once before tasks. These fields bind available artifacts and do not constitute signed
execution attestation. See the [LLM process contract](API.md) for availability and numeric rules.
Python 3 is required to generate build provenance.

A saved schema-2 artifact or schema-1 LLM sweep envelope can be checked without running the benchmark:

```bash
python3 script-examples/verify_llm_result.py llm-result.json --binary ./memory_benchmark --require-raw-timing
```

The optional binary is hashed, never executed. The verdict distinguishes artifact consistency, run acceptance,
independent checksum reconstruction, raw timing availability and binary binding (and, for schema-1 LLM sweeps, emits
`artifact_kind: "llm_sweep"` with per-run `run_verdicts` under the shared artifact budget). A consistent failed or
unaccepted run/sweep exits 1;
an unsupported schema or exceeded verifier work budget exits 2. Only a consistent accepted artifact exits 0.
See the [bounded verifier contract](API.md#independent-llm-artifact-verifier) for limits and float tolerances.

### LLM result is incomplete, invalid, or not comparable

Check top-level backend/phase/layout/methodology, status/reason, `results_complete`, `run_accepted`, scenario-order
balance, counters, every
measurement's status/reason, checksum evidence, duration quality, QoS, and environment warnings. A started task may
finish after an interrupt, but remaining slots are interrupted/null. Count one can be correct and accepted while
remaining unsuitable for a comparison policy requiring position balance. High CV, cache-dominant working sets, or non-nominal thermal/
power state do not turn a measurement into zero; they remain explicit quality warnings.

For paged results, also inspect the permutation identity/hash, physical K/V lengths, layout padding, table protection,
lookup/metadata counts, phase-specific K/V-write post-validation, and padding-canary result. For KV-bearing tasks,
decode validates the current-token write; prefill checks representative canonical-word/boundary samples against the
final operation ordinal, rather than all prompt contents. CPU paged decode weights-only checks append slots and padding,
CPU paged prefill weights-only checks padding only, and Metal weights-only performs neither KV-write nor padding checks. A different block
size or permutation is a different comparison cohort even when logical model geometry is unchanged.

If an otherwise identical command fails memory admission only after a non-empty `--output` target is added, include the
conservative JSON DOM/serialization peak in the diagnosis. It scales with planned measurement records, retained
calibration attempts, effective worker checksum trees and once-per-canonical-plan identities/expectations. For prefill it also
scales with the scenario-specific ownership-scope identity set, including layer/batch scopes; reduce the
model/count/worker demand or free memory rather than treating the reserve as measured payload.

### Script cannot find benchmark binary

The script first uses the repository-root `memory_benchmark` when it is executable, then tries `memory_benchmark` from
`PATH`. If neither is suitable, set `BENCHMARK_CMD` to an executable path. The selected binary must produce complete,
accepted `cpu_latency` schema 1.

### Plot script says no blocks found

Make sure you are passing `script-examples/final_output.txt` generated by the latency sweep script.

---

## Additional Resources

- [README.md](../README.md) - project overview, install, examples
- [API.md](API.md) - machine-readable process contract, stdout/stderr transport, schemas, and verifier contract
- [CAPABILITIES.md](CAPABILITIES.md) - measurement capability overview and interpretation notes
- [PARAMETER_MATRIX.md](PARAMETER_MATRIX.md) - complete flag list, mode compatibility tables, numeric limits, and sweep rules
- [CPU_BANDWIDTH_WHITEPAPER.md](CPU_BANDWIDTH_WHITEPAPER.md) - audited CPU bandwidth methodology deep dive
- [LATENCY_WHITEPAPER.md](LATENCY_WHITEPAPER.md) - audited pointer-chase latency methodology deep dive
- [PATTERNS_WHITEPAPER.md](PATTERNS_WHITEPAPER.md) - audited 7-pattern ARM64 methodology deep dive
- [TLB_ANALYSIS_WHITEPAPER.md](TLB_ANALYSIS_WHITEPAPER.md) - TLB analysis methodology and JSON schema
- [CORE_TO_CORE_WHITEPAPER.md](CORE_TO_CORE_WHITEPAPER.md) - Core-to-Core Cache-Line Handoff Latency Benchmark: methodology, assembly protocol, scheduler-hint scenarios, and JSON contract
- [GPU_BANDWIDTH_WHITEPAPER.md](GPU_BANDWIDTH_WHITEPAPER.md) - Metal GPU memory-bandwidth methodology, validation,
  schema 1, capability boundaries, and maintenance policy
- [LLM_MEMORY_PROFILE_WHITEPAPER.md](LLM_MEMORY_PROFILE_WHITEPAPER.md) - synthetic CPU and Metal
  decode/prefill methodology for both layouts, execution, schema 2, validation, and interpretation limits
- [TECHNICAL_SPECIFICATION.md](TECHNICAL_SPECIFICATION.md) - architecture and implementation details
- [CHANGELOG.md](../CHANGELOG.md) - release history

Repository sample result files:

Recorded 0.63.0 LLM schema-1 examples:

- [Apple M5 CPU-decode working-set comparison note](../results/0.63.0/AppleM5_LLM_working_set_scaling.md)
- [Apple M5, 256 MiB weights and 32,768-token context](../results/0.63.0/apple_m5_cpu_decode_contiguous_weights_256mib_context_32768.json)
- [Apple M5, 1,024 MiB weights and 131,072-token context](../results/0.63.0/apple_m5_cpu_decode_contiguous_weights_1024mib_context_131072.json)

The pair is an empirical working-set scaling example. The runs have different model geometry and frozen work plans,
so treat them as side-by-side examples rather than one pooled performance distribution.

The following are retained historical records. Archived standard JSON among them is not accepted by current
`script-examples/` consumers.

- `results/0.53.7/MacMiniM4_benchmark.json`
- `results/0.53.7/MacMiniM4_patterns.json`
- `results/0.53.7/MacMiniM4_core2core.json`
- `results/0.53.8/MacMiniM4_analyze-tlb-chain-mode-random-box.json`

Command help:

```bash
memory_benchmark -h
```

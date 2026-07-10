# macOS Apple Silicon Memory Benchmark

![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-000000?logo=apple) ![CLI](https://img.shields.io/badge/CLI-Tool-00A8CC?logo=terminal) ![License](https://img.shields.io/badge/license-GPL--3.0-blue) ![Assembly](https://img.shields.io/badge/Assembly-ARM64-6E4C13) ![C++](https://img.shields.io/badge/C++-00599C?logo=cplusplus&logoColor=white)

Copyright 2025-2026 Timo Heimonen <timo.heimonen@proton.me>  
License: GPL-3.0 license  
  
**Low-level tool** to measure memory read/write/copy bandwidth, cache/main memory latency, and access pattern performance on macOS Apple Silicon (ARM64).  
Application [measurement capabilities](CAPABILITIES.md) information.

![Macbook Air M5 Cache Latency from multiple JSON-files](pictures/MacBookAirM5_latency_vs_cache-stride-tlb.png)  
Macbook Air M5 Cache Latency by example-script provided. Using different size TLB locality and Stride size. 

## Star History

[![Star History Chart](https://api.star-history.com/chart?repos=timoheimonen/macOS-memory-benchmark&type=date&legend=top-left)](https://www.star-history.com/?repos=timoheimonen%2FmacOS-memory-benchmark&type=date&legend=top-left)

## Description

`memory_benchmark` measures memory behavior on macOS Apple Silicon with an implementation focused on practical low-level analysis:

1. Main memory bandwidth (read/write/copy).
2. Cache bandwidth (L1/L2 auto-detected, or custom cache target).
3. Main memory latency (dependent pointer chase).
4. Cache latency (dependent pointer chase).
5. Pattern bandwidth behavior (sequential, strided, random).

The benchmark uses ARM64 assembly kernels, warmup passes, and loop statistics to produce stable and comparable results.

## Why This Tool?

This project is built to characterize Apple Silicon memory behavior in a direct and reproducible way.

Compared to generic benchmark suites, it is intentionally focused on:

- Native ARM64 execution on macOS.
- Cache-aware and latency-aware test paths.
- Explicit mode control for bandwidth-only, latency-only, and pattern-only workflows.
- JSON output that is easy to automate and plot.

## Disclaimer

**Use this software at your own risk.** It performs sustained, intensive memory operations. The author is not responsible for instability, data loss, or hardware issues resulting from use.

## Target Platform

- macOS
- Apple Silicon (ARM64)

## Install with Homebrew

```bash
brew install timoheimonen/macOS-memory-benchmark/memory-benchmark
```

## Build from Source

Prerequisites:

- Xcode Command Line Tools (`xcode-select --install`)

Build:

```bash
git clone https://github.com/timoheimonen/macOS-memory-benchmark.git
cd macOS-memory-benchmark
make
```

Run from source tree (without installing to `PATH`):

```bash
./memory_benchmark -h
```

## Testing

Install test dependency:

```bash
brew install googletest
```

Run unit tests:

```bash
make test
```

## Quick Usage

Examples below use the Homebrew/`PATH` form (`memory_benchmark ...`).
If you are running directly from a local source build, use `./memory_benchmark ...`.

Help:

```bash
memory_benchmark -h
```

Standard benchmark:

```bash
memory_benchmark --benchmark
```

For longer runs, prevent sleep:

```bash
caffeinate -i -d memory_benchmark --benchmark --count 10 --buffer-size 1024
```

## Benchmark Modes

- **`--benchmark`**: Runs standard memory benchmark (main bandwidth + main latency + cache bandwidth + cache latency). **Required** to execute standard benchmarks. Mutually exclusive with `--patterns`.
- **`--patterns`**: Runs pattern bandwidth suite only (`sequential_forward`, `sequential_reverse`, `strided_64`, `strided_4096`, `strided_16384`, `strided_2mb`, `random`). Mutually exclusive with `--benchmark`. Unless `--iterations` is explicit, each read/write/copy sample uses an excluded same-shape pilot to calibrate toward 150 ms; 100–250 ms is the intended measurement window. An explicit `--iterations` value is the measured pass count and bypasses the calibration pilot. Every operation still receives its own same-shape warmup. Strided cases rotate their 32-byte starting phase on every pass inside one ARM64 assembly call, use exact phase-aware payload accounting, and may reduce the requested thread count so every active worker performs at least one real stride transition; consequently, `--threads` and pattern thread sweeps specify a requested count, not necessarily the effective count for every stride. Pattern timing starts only after every actual worker has completed its best-effort QoS setup attempt and reached the ready gate, and ends when the last worker finishes measured work; workload preparation, thread creation, teardown, and joining are excluded. Random per-worker index lists are also prepared before timing.
- **`--only-bandwidth`**: Runs bandwidth paths only. **Requires `--benchmark`**. Cannot be used with `--patterns`, `--cache-size`, or `--latency-samples`.
- **`--only-latency`**: Runs latency paths only. **Requires `--benchmark`**. Cannot be used with `--patterns` or `--iterations`.
- **`--analyze-tlb`**: Runs standalone TLB analysis with page-native spread/packed pairs, calibrated measurement windows, adaptive balanced rounds, paired bootstrap confidence intervals, and an independent boundary-validation pass; only optional `--output <file>`, `--latency-stride-bytes <bytes>`, `--latency-chain-mode <mode>`, `--tlb-density <low|medium|high>`, `--seed <uint64>`, `--sweep <key=...>`, and `--sweep-max-runs <count>` may be combined with it.
- **`--analyze-core2core`**: Runs standalone core-to-core cache-line handoff analysis mode; only optional `--output <file>`, `--count <count>`, `--latency-samples <count>`, `--sweep count=...`, `--sweep latency-samples=...`, and `--sweep-max-runs <count>` may be combined with it. See [CORE_TO_CORE_WHITEPAPER.md](CORE_TO_CORE_WHITEPAPER.md) for methodology and JSON contract.
- **`--sweep <key=a,b>`**: Runs a Cartesian parameter sweep for `--benchmark`, `--patterns`, `--analyze-tlb`, or `--analyze-core2core` and writes one combined JSON file. Repeat `--sweep` to sweep multiple parameters. Requires `--output <file>`.

Pattern samples have steady-state, warm-memory semantics rather than cold-start semantics. Random read/write/copy warmup
traverses the complete measured address list, and in automatic-calibration mode the excluded pilot further preconditions
the measured workload shape.

Pattern bandwidth means **effective payload bandwidth**, not an estimate of physical DRAM or cache-bus traffic. A read or
write access contributes its 32-byte kernel payload; copy contributes both sides (32-byte read + 32-byte write). The
numerator is the exact planned payload completed by all workers and passes. Sparse-stride metadata reports requested and
effective worker counts, phase-zero and minimum/maximum access counts per pass, phase period, pass count, total accesses,
exact payload bytes, and working-set details. Because phase rotation can change the access count between passes,
`total_accesses` and `total_payload_bytes` are authoritative exact totals; do not derive them by multiplying
`accesses_per_pass` by `passes`.

The executor consumes finalized strided worker ranges directly, without repartitioning them. Random worker boundaries
and per-worker index lists are built before timing; the timed callback performs no worker lookup or index filtering.

Across `--count` loops, pattern groups rotate in deterministic cyclic Latin-square order to spread first/last-position
effects. Operations within a group remain fixed read, write, copy, each with operation-specific warmup. Workers request a
macOS QoS class as a best-effort scheduler hint; the benchmark does not pin threads to cores. When `--count > 1`, the
headline is the median (P50). Statistics report coefficient of variation (CV) and warn above 5% for sequential and 64-byte
stride results or above 10% for sparse-stride and random results. A workload that cannot be measured is `N/A` with an
explicit status/reason in console output and `null` with status metadata in JSON; it is never represented as zero
bandwidth.

Latency-specific disable controls in `--only-latency`:

- `--buffer-size 0` disables main-memory latency target.
- `--cache-size 0` disables cache-latency target.
- Both disabled at once is invalid.

## Most-Used Options

Long options require `--`. A single dash is only valid for one-character short options such as `-B` or `-b`.

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

- `--benchmark`: Run standard memory benchmark. Mutually exclusive with `--patterns`. Required for standard, `--only-bandwidth`, and `--only-latency` modes.

- `--buffer-size <MB>`: Main buffer size (default `512`; auto-capped by memory safety rules).
- `--iterations <count>`: Bandwidth iterations per loop (default `1000`). In `--patterns` mode, the default does not force 1000 measured passes: samples are calibrated automatically unless this option is explicitly supplied. An explicit value bypasses the pattern calibration pilot but not the operation-specific warmup.
- `--count <count>`: Full benchmark repetitions (default `1`; use `5-10` for statistics).
- `--threads <count>`: Bandwidth thread count (latency tests remain single-threaded). In `--patterns`, the default is the
  detected performance-core count for repeatability; explicitly request a larger detected-core count to include more
  workers. This is a requested count, and sparse strides may reduce the effective count.
- `--cache-size <KB>`: Custom cache target. Non-zero range is `16` to `1048576` KB (1 GB).
- `--analyze-tlb`: Standalone TLB-boundary benchmark. It selects the largest `1024/512/256 MiB` candidate whose predicted buffer-plus-scratch peak fits a conservative available-memory budget. The compact settings block reports the run identity, buffer-lock/QoS outcome, estimated peak versus budget, sweep range, and rough duration; full access and memory estimates remain in JSON. Every scheduled point is a same-round page-native spread/packed pair. Each console point is one line containing cache-hot spread and packed P50 values, the primary paired translation delta, and the active cache-line footprint; detailed page/cache-line diagnostics remain in JSON. Virtual locality is not the active data footprint: with 16 KiB pages, the 512 MiB comparison has 32,768 one-line nodes, or a 2 MiB active cache-line footprint. Points below 64 nodes carry a compact `*` diagnostic marker explained once in the sweep legend. A pilot times each chain and calibrates the main measurement toward the profile target while retaining a minimum number of whole-chain cycles. Seeded cyclic-Latin rounds stop at the profile CI-width target or its maximum round count. Boundary inference operates on round-matched `spread - packed` deltas and requires independent validation. Stride must be pointer-aligned and no larger than the system page size; it need not divide the page size. Main-thread `user-interactive` QoS and `mlock()` are best-effort; their success/error status is reported in console/JSON and failures do not abort the analysis.
- `--tlb-density <low|medium|high>`: Selects the TLB runtime profile. `low`/`quick` uses a 15-point base sweep without refinement and 7-12 rounds; its console conclusions are explicitly labeled screening estimates that should be confirmed with `medium` or `high`. `medium`/`standard` is the default and uses a 15-point base sweep with refinement and 10-20 rounds; `high`/`exhaustive` uses a 29-point base sweep with refinement and 15-30 rounds.
- `--analyze-core2core`: Standalone two-thread cache-line ping-pong benchmark for coherence handoff latency, with three scheduler-hint scenarios (`no_affinity_hint`, `same_affinity_tag`, `different_affinity_tags`). Reports round-trip and one-way-estimate latency plus percentiles.
- `--latency-samples <count>`: Samples per latency test (default `1000`).
- `--latency-stride-bytes <bytes>`: Pointer-chain stride for latency tests (default `256`; must be > 0 and pointer-size aligned). With `--analyze-tlb`, it must also be no larger than the system page size. The page-native spread builder rounds spacing up to a cache-line multiple; the packed control uses one node per cache line. Page-size divisibility is not required.
- `--latency-chain-mode <mode>`: Pointer-chain construction policy. Modes: `auto` (default), `global-random`, `random-box`, `same-random-in-box`, `diff-random-in-box`. Analyze-TLB results are comparable only when the effective mode matches; the increasing-page `same-random-in-box` and `diff-random-in-box` modes are intentionally order/prefetch-sensitive.
- `--seed <uint64>`: Reproducible random-workload seed for `--patterns`, or planner, round-order, and pointer-chain seed for `--analyze-tlb`. It is supported only by those two modes. For patterns, the seed selects a unique, no-replacement permutation prefix of 32-byte-aligned offsets; when omitted, one seed is generated for the command, and the resolved workload is repeated across `--count` loops. For TLB analysis, one omitted seed is generated for the command and reused across all generated sweep runs.
- `--latency-tlb-locality-kb <KB>`: Pointer-chain locality window (default `1024`; `0` = global random chain; non-zero values must be page-size multiples). If omitted, regular main-memory latency output also includes an automatic TLB comparison (`16 KB` hit-biased vs `0` miss-biased) and estimated page-walk penalty. The automatic comparison uses P50 over three complete pointer-chase passes per point to reduce single-IRQ outlier impact.
- `--non-cacheable`: Best-effort cache-discouraging hints (not true uncached memory).
- `--output <file>`: Save JSON output.
- `--sweep <key=a,b>`: Sweep supported parameters. General benchmark keys: `buffer-size`, `cache-size`, `threads`, `latency-tlb-locality-kb`, `latency-stride-bytes`, `latency-chain-mode`; TLB analysis keys: `latency-stride-bytes`, `latency-chain-mode`, `tlb-density`; core-to-core keys: `count`, `latency-samples`.
- `--sweep-max-runs <count>`: Maximum generated sweep runs (general default `256`; `--analyze-tlb` default `16`). An explicit value overrides the mode default.

Sweep configurations are fully validated before the first run. Combined sweep JSON is atomically checkpointed after every completed run and includes `status`, `planned_runs`, `completed_runs`, and `conclusions_valid`.

## Typical Workflows

Statistical baseline:

```bash
caffeinate -i -d memory_benchmark --benchmark --count 10 --buffer-size 1024 --output baseline.json
```

Pattern analysis:

```bash
memory_benchmark --patterns --count 10 --buffer-size 512 --seed 123456789 --output patterns.json
```

Built-in parameter sweep:

```bash
memory_benchmark --benchmark --only-latency --count 5 --sweep buffer-size=256,512,1024 --sweep latency-tlb-locality-kb=16,1024,0 --output latency_sweep.json
```

Core-to-core sample-depth sweep:

```bash
memory_benchmark --analyze-core2core --count 3 --sweep latency-samples=500,1000,2000 --output core2core_sample_sweep.json
```

Latency locality comparison:

```bash
memory_benchmark --benchmark --only-latency --buffer-size 1024 --count 10 --latency-samples 5000 --latency-tlb-locality-kb 16 --output lat_tlb16.json
memory_benchmark --benchmark --only-latency --buffer-size 1024 --count 10 --latency-samples 5000 --latency-tlb-locality-kb 0 --output lat_global.json
```

Regular benchmark with default latency profile (`256B` stride, `1024KB` locality) and automatic TLB breakdown (omit `--latency-tlb-locality-kb`):

```bash
memory_benchmark --benchmark --count 1
```

TLB-vs-cache isolation (smaller stride within pages):

```bash
memory_benchmark --benchmark --only-latency --buffer-size 1024 --cache-size 4096 --latency-stride-bytes 64 --latency-tlb-locality-kb 16 --count 10 --latency-samples 5000 --output lat_stride64_tlb16.json
```

For a more pessimistic global-random DRAM stress profile, use:

```bash
memory_benchmark --benchmark --only-latency --buffer-size 1024 --latency-stride-bytes 512 --latency-tlb-locality-kb 0 --count 10 --latency-samples 5000 --output lat_stride512_global.json
```

Custom cache target:

```bash
memory_benchmark --benchmark --cache-size 4096 --threads 1 --count 5 --output cache_4mb.json
```

Standalone TLB analysis report:

```bash
memory_benchmark --analyze-tlb
```

Standalone TLB analysis with JSON export:

```bash
memory_benchmark --analyze-tlb --output tlb_analysis.json
```

Standalone TLB analysis with custom stride:

```bash
memory_benchmark --analyze-tlb --latency-stride-bytes 128 --output tlb_analysis_stride128.json
```

Reproducible TLB analysis:

```bash
memory_benchmark --analyze-tlb --seed 123456789 --output tlb_analysis_seeded.json
```

Standalone TLB analysis density/stride sweep:

```bash
memory_benchmark --analyze-tlb --sweep latency-stride-bytes=64,128 --sweep tlb-density=medium,high --sweep-max-runs 4 --output tlb_stride_density_sweep.json
```

Standalone core-to-core handoff analysis:

```bash
memory_benchmark --analyze-core2core
```

Standalone core-to-core handoff analysis with JSON export and custom sample depth:

```bash
memory_benchmark --analyze-core2core --count 5 --latency-samples 2000 --output core2core.json
```

## Output Overview

Console output includes:

- Resolved configuration and cache information.
- Per-loop benchmark results.
- Main-memory latency may include automatic TLB breakdown lines (`TLB hit latency`, `TLB miss latency`, and `Estimated page-walk penalty`) when `--latency-tlb-locality-kb` is not explicitly set. These auto-TLB hit/miss values are P50 values from three complete comparison passes per point, so a single IRQ-inflated pass is less likely to dominate the estimate.
- Aggregate statistics when `--count > 1` (including P50/P90/P95/P99 and stddev). In auto-TLB mode, statistics also include `TLB Hit Latency (ns)`, `TLB Miss Latency (ns)`, and `Estimated Page-Walk Penalty (ns)`.
- Standalone `--analyze-tlb` uses IEC units and one compact row per locality, reports `Analysis Status`, and suppresses boundary conclusions when the sweep is interrupted or incomplete. A `*` identifies a below-64-node diagnostic point, and the `quick` profile visibly advises confirmation with `medium` or `high`. Its primary 512 MiB result is the compact `Large-Locality Paired Comparison`: same-round spread, packed, and translation-delta P50 values plus the active cache-line footprint. These are cache-hot pointer-chain timings, not direct DRAM latency or an isolated page-table-walk cost.

Standalone TLB JSON uses `schema_version: 4` and `methodology_version: "page-native-paired-adaptive-validated-v4"`. It includes the runtime profile, adaptive-round bounds and completion reason, access-calibration data, memory budget, predicted peak, best-effort QoS/lock results, work estimate, exact uint64 decimal-string seeds and derivation policy, execution-ordered discovery/validation records, raw pair latencies, requested/actual pages, pointer-node density, active cache-line footprints, short-cycle diagnostics, effective chain-mode comparability guidance, verified physical diagnostics, and same-round `translation_delta_ns = spread - packed`. Explicit base+validation, large-locality, and total counters state their pass scope, while `validation_required`, `validation_status`, and `validation_complete` distinguish validation completion from not-run/not-required states. Boundary objects retain accepted and rejected candidates with discovery/validation evidence, deterministic bootstrap 95% CIs, noise floor, persistence counts, rejection reasons, and the final bracket. The only large-locality result object is `tlb_analysis.large_locality_paired_comparison`; its delta P50 is the median of same-round deltas, not the difference of independently aggregated medians. Schema 4 contains only the current fields. The bundled plotter separately retains read support for historical schema 1-3 files and does not accept their field names in a schema 4 document.

JSON output shape:

- Standard mode:

```json
{
  "configuration": {},
  "execution_time_sec": 0,
  "main_memory": {},
  "cache": {},
  "timestamp": "...",
  "version": "..."
}
```

- Pattern mode:

```json
{
  "configuration": {},
  "execution_time_sec": 0,
  "patterns": {},
  "timestamp": "...",
  "version": "..."
}
```

Pattern JSON configuration uses `pattern_schema_version: 2` and
`methodology_version: "pattern-v2-phase-calibrated-seeded"`. It records the exact decimal-string seed, pass/calibration
policy, warmup and execution-order policies, native page size, and best-effort QoS/no-pinning policy. Every operation has
an aggregate `status`, a median-or-single `value_gb_s`, raw `values_gb_s`, statistics (including CV), and per-loop
`measurements` with exact work and timing metadata. Unavailable values are `null`. `strided_2mb` describes a 2 MiB virtual
address stride only: `large_page_backing_verified` remains false unless backing is actually verified, so the label is not
a superpage claim.

Current latency payload is nested (not scalar):

```json
"latency": {
  "average_ns": {
    "values": [],
    "statistics": {}
  },
  "samples_ns": {
    "values": [],
    "statistics": {}
  },
  "auto_tlb_breakdown": {
    "tlb_hit_ns": {
      "values": [],
      "statistics": {}
    },
    "tlb_miss_ns": {
      "values": [],
      "statistics": {}
    },
    "page_walk_penalty_ns": {
      "values": [],
      "statistics": {}
    }
  }
}
```

## Visualization Workflow

Run cache/locality sweep script:

```bash
./script-examples/latency_test_script.sh
```

Plot extracted percentile trends from `script-examples/final_output.txt`:

```bash
python3 script-examples/plot_cache_percentiles.py script-examples/final_output.txt --metric median
```

Supported metrics: `median`, `p90`, `p95`, `p99`, `average`, `min`, `max`, `stddev`.

Note: `script-examples/latency_test_script.sh` invokes `memory_benchmark --benchmark` from `PATH`. If you only built locally as `./memory_benchmark`, either install it or update `BENCHMARK_CMD` in the script.

## Interpreting Results Under Active System Load

If other macOS programs are heavily active, treat results as contention-influenced.

Recommended interpretation approach:

1. Keep background load profile consistent across compared runs.
2. Use repeated loops (`--count 10` or more).
3. Prioritize median and tail percentiles (`P95`/`P99`) over single-run extremes.
4. Keep command lines identical when comparing machines or builds.

The measured bandwidth can be close to platform theoretical limits under favorable conditions. For example,
`results/0.53.7/MacMiniM4_benchmark.json` reports ~`115.87 GB/s` average main-memory read on Apple M4 versus
~`120 GB/s` theoretical peak (about `97%`). Treat this as an empirical reference, not a guaranteed ceiling.

Reference sample result files in this repository:

- `results/0.53.7/MacMiniM4_benchmark.json`
- `results/0.53.7/MacMiniM4_patterns.json`

The 0.53.7 pattern file predates pattern schema 2 and is not a direct numerical baseline for
`pattern-v2-phase-calibrated-seeded` results.

## Documentation

- **[Measurement Capabilities](CAPABILITIES.md)**: overview of what the tool can measure and how to interpret those measurements.
- **[User Manual](MANUAL.md)**: full usage guide, option reference, workflows, troubleshooting.
- **[Technical Specification](TECHNICAL_SPECIFICATION.md)**: architecture, execution flow, memory model, output contracts.
- **[Latency Whitepaper](LATENCY_WHITEPAPER.md)**: dependent pointer-chase design, chain construction, and sampling methodology.
- **[TLB Analysis Whitepaper](TLB_ANALYSIS_WHITEPAPER.md)**: standalone `--analyze-tlb` methodology, boundary/guard rules, confidence model, and JSON verification contract.
- **[Core-to-Core Cache-Line Handoff Latency Whitepaper](CORE_TO_CORE_WHITEPAPER.md)**: standalone `--analyze-core2core` methodology, LDAR/STLR assembly protocol, scheduler-hint scenarios, and JSON contract.

## Non-Goals

Here are some things what are not goals to this application.
- Support other than Apple Silicon systems
- GUI
- Server backend / top score tables

## Limitations and Caveats

- `--non-cacheable` is best effort only (`madvise` hints); it does not create true uncached mappings.
- Small buffers can be cache-dominated, so they may not represent DRAM behavior.
- Apple Silicon user space has no explicit data-cache flush primitive equivalent to x86 `CLFLUSH` for strict cold-cache control.
- TLB-locality mode controls pointer-chain construction policy; it does not directly control hardware TLB residency.
- Background activity, thermals, and scheduling can materially affect tails and variance.
- Pattern GB/s is effective kernel payload bandwidth, not observed physical memory-bus traffic.
- `strided_2mb` specifies a 2 MiB virtual-address stride; it does not establish physical superpage backing.
- Pattern ratios alone do not prove prefetch, cache-thrashing, or TLB mechanisms; use controlled follow-up tests and
  `--analyze-tlb` for supported TLB analysis.

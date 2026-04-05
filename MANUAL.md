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
memory_benchmark -benchmark
```

If built from source, use `./memory_benchmark` instead.

All command examples in this manual use the installed/`PATH` form (`memory_benchmark ...`).
If running from an uninstalled local source build, prefix commands with `./`.

For longer runs, prevent sleep:

```bash
caffeinate -i -d memory_benchmark -benchmark -count 10 -buffersize 1024
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

Latency tests use dependent pointer-chase chains. `-latency-tlb-locality-kb` controls how the chain is constructed:

- `16` (default): randomized within 16 KB windows, plus randomized window order
- `0`: fully global random chain
- non-zero values must be multiples of system page size

`-latency-chain-mode` controls pointer-chain ordering policy:

- `auto` (default): preserves current behavior (`random-box` when locality > 0, `global-random` when locality = 0)
- `global-random`: full-buffer random permutation
- `random-box`: random order within locality boxes and random box traversal order
- `same-random-in-box`: same in-box random pattern reused across boxes (increasing box order)
- `diff-random-in-box`: independently randomized in-box pattern per box (increasing box order)

Modes other than `global-random` require non-zero `-latency-tlb-locality-kb`.

`-latency-stride-bytes` controls spacing between chain nodes. Smaller stride biases toward same-page reuse;
larger stride increases page turnover pressure.

Use `0` when you explicitly want stronger translation effects in the measured path.

When `-latency-tlb-locality-kb` is omitted in regular benchmark mode, main-memory latency output also runs an
automatic comparison and prints:

- TLB hit latency (16 KB locality)
- TLB miss latency (global random locality, `0`)
- Estimated page-walk penalty (`miss - hit`)

If you explicitly set `-latency-tlb-locality-kb` (including `16` or `0`), this auto comparison is skipped.

### Pattern benchmarks

Pattern mode (`-patterns`) measures bandwidth sensitivity across:

- Sequential Forward
- Sequential Reverse
- Strided (Cache Line - 64B)
- Strided (Page - 4096B)
- Strided (Page - 16384B)
- Strided (Superpage - 2MB)
- Random Uniform

---

## Command-Line Options

### Core controls

#### `-buffersize <MB>`

- Main buffer size in MB (per main buffer)
- Default: `512`
- Auto-capped to memory safety limit
- `-buffersize 0` is valid only with `-only-latency` and disables main-memory latency path

#### `-iterations <count>`

- Bandwidth loop iterations
- Default: `1000`
- Positive integer
- Not allowed with `-only-latency`

#### `-count <count>`

- Full benchmark loop count
- Default: `1`
- Positive integer
- Use `5` to `10` for stable statistics

#### `-threads <count>`

- Thread count for bandwidth tests
- Default: detected core count
- If above available cores, it is capped
- Latency tests remain single-threaded

### Mode selection

#### `-benchmark`

- **Required** to run standard memory benchmark (bandwidth + latency)
- Mutually exclusive with `-patterns`
- Can be combined with `-only-bandwidth`, `-only-latency`, `-cache-size`, `-threads`, and other modifier flags
- Running without this flag (or `-patterns`) shows help and exits

#### `-patterns`

- Runs only access-pattern benchmarks
- Skips standard bandwidth/latency sections

#### `-only-bandwidth`

- Runs bandwidth paths only
- **Requires `-benchmark`**
- Incompatible with: `-patterns`, `-cache-size` (any value including `0`), `-latency-samples`

#### `-only-latency`

- Runs latency paths only
- **Requires `-benchmark`**
- Incompatible with: `-patterns`, `-iterations`
- Supports selective target disabling:
  - `-buffersize 0` disables main-memory latency
  - `-cache-size 0` disables cache latency
  - both zero is invalid

#### `-analyze-tlb`

- Runs standalone TLB analysis mode only
- Can be combined only with optional `-output <file>`, `-latency-stride-bytes <bytes>`, and `-latency-chain-mode <mode>`
- Uses latency stride from `-latency-stride-bytes` (same default as standard latency mode), sweeps locality windows `max(16KB, 2*stride)` to `256MB`, and reports inferred L1/L2 TLB boundaries and entry counts
- Separately computes page-walk penalty as `P50(512MB) - P50(effective baseline locality)` when analysis buffer is at least `512MB`
- Detailed methodology and JSON contract: `TLB_ANALYSIS_WHITEPAPER.md`

#### `-analyze-core2core`

- Runs standalone two-thread cache-line handoff (ping-pong) mode only
- Can be combined only with optional `-output <file>`, `-count <count>`, and `-latency-samples <count>`
- Executes three scheduler-hint scenarios: `no_affinity_hint`, `same_affinity_tag`, and `different_affinity_tags`
- Reports round-trip latency, one-way estimate (`round_trip / 2`), and sample distribution stats (P50/P90/P95/P99/stddev/min/max)
- Includes per-thread QoS/affinity hint status in console and JSON output
- Notes explicitly that macOS user-space cannot hard-pin exact core IDs
- Detailed methodology and JSON contract: [CORE_TO_CORE_WHITEPAPER.md](CORE_TO_CORE_WHITEPAPER.md)

### Latency-specific controls

#### `-latency-samples <count>`

- Sample count per latency test
- Default: `1000`
- Positive integer
- More samples improve percentile stability at the cost of run time

#### `-latency-stride-bytes <bytes>`

- Pointer-chain stride for latency tests
- Default: `64`
- Must be `> 0`
- Must be a multiple of 8 bytes (pointer size on Apple Silicon)
- Use smaller values (for example `64`) to increase same-page cache-line activity and reduce TLB sensitivity

#### `-latency-chain-mode <mode>`

- Pointer-chain construction policy for latency paths
- Default: `auto`
- Accepted values: `auto`, `global-random`, `random-box`, `same-random-in-box`, `diff-random-in-box`
- `global-random` works with `-latency-tlb-locality-kb 0`
- `random-box`, `same-random-in-box`, and `diff-random-in-box` require `-latency-tlb-locality-kb > 0`

#### `-latency-tlb-locality-kb <size_kb>`

- Pointer-chain locality window for latency path
- Default: `16`
- `0` disables locality mode (global random chain)
- Non-zero values must be exact multiples of system page size
- In regular benchmark mode, explicitly setting this option disables automatic TLB hit/miss comparison lines

### Cache and memory hint controls

#### `-cache-size <KB>`

- Enables custom cache test size
- Non-zero range: `16` to `1048576` KB (1 GB)
- `0` is accepted only with `-only-latency` and disables cache latency target
- When set to non-zero, auto L1/L2 detection is replaced by custom cache target

#### `-non-cacheable`

- Applies cache-discouraging `madvise()` hints
- Best effort only; this does **not** create truly uncached memory

### Output

#### `-output <file>`

- Saves JSON output
- Relative path writes under current working directory
- Parent directories are created automatically

#### `-h`, `--help`

- Print help and exit

---

## Mode Compatibility

### Valid combinations

```bash
# Full benchmark
memory_benchmark -benchmark -count 10 -buffersize 1024 -output full.json

# Pattern-only
memory_benchmark -patterns -count 5 -buffersize 512 -output patterns.json

# Bandwidth-only
memory_benchmark -benchmark -only-bandwidth -threads 8 -count 5

# Latency-only (both main + cache)
memory_benchmark -benchmark -only-latency -latency-samples 5000 -count 10

# Latency-only (main memory only)
memory_benchmark -benchmark -only-latency -cache-size 0 -buffersize 1024

# Latency-only (cache only)
memory_benchmark -benchmark -only-latency -buffersize 0 -cache-size 2048

# Standalone TLB analysis
memory_benchmark -analyze-tlb

# Standalone TLB analysis with JSON export
memory_benchmark -analyze-tlb -output tlb_analysis.json

# Standalone TLB analysis with custom stride
memory_benchmark -analyze-tlb -latency-stride-bytes 128 -output tlb_analysis_stride128.json

# Standalone TLB analysis with explicit chain mode
memory_benchmark -analyze-tlb -latency-chain-mode same-random-in-box -output tlb_analysis_same_box.json

# Standalone core-to-core handoff analysis
memory_benchmark -analyze-core2core

# Standalone core-to-core analysis with deeper sampling + JSON
memory_benchmark -analyze-core2core -count 5 -latency-samples 2000 -output core2core.json
```

### Invalid combinations

```bash
# invalid: -benchmark with -patterns (mutually exclusive)
memory_benchmark -benchmark -patterns

# invalid: pattern mode with only-bandwidth
memory_benchmark -patterns -only-bandwidth

# invalid: pattern mode with only-latency
memory_benchmark -patterns -only-latency

# invalid: latency samples with only-bandwidth
memory_benchmark -only-bandwidth -latency-samples 5000

# invalid: iterations with only-latency
memory_benchmark -only-latency -iterations 2000

# invalid: both latency targets disabled
memory_benchmark -only-latency -buffersize 0 -cache-size 0

# invalid: analyze-tlb with unsupported extra option
memory_benchmark -analyze-tlb -buffersize 1024

# invalid: analyze-core2core with unsupported extra option
memory_benchmark -analyze-core2core -threads 4
```

---

## Common Workflows

### Quick baseline

```bash
memory_benchmark -benchmark
```

Good for a fast health check.

### Statistical baseline (recommended)

```bash
caffeinate -i -d memory_benchmark -benchmark -count 10 -buffersize 1024 -output baseline.json
```

Use this for comparisons across machines or software versions.

### Pattern analysis

```bash
memory_benchmark -patterns -count 10 -buffersize 512 -output patterns.json
```

Shows how bandwidth changes under different access patterns.

### Latency analysis with TLB-locality control

```bash
# default locality mode
memory_benchmark -benchmark -only-latency -buffersize 1024 -latency-samples 5000 -count 10 -output lat_tlb16.json

# global random chain
memory_benchmark -benchmark -only-latency -buffersize 1024 -latency-samples 5000 -latency-tlb-locality-kb 0 -count 10 -output lat_global.json

# same in-box random pattern (good for prefetch-vs-TLB comparisons)
memory_benchmark -benchmark -only-latency -buffersize 1024 -latency-samples 5000 -latency-tlb-locality-kb 16 -latency-chain-mode same-random-in-box -count 10 -output lat_same_box.json
```

### Regular benchmark with automatic DRAM TLB breakdown

```bash
memory_benchmark -benchmark -latency-stride-bytes 128 -count 1
```

This prints `Average latency` plus auto-derived `TLB hit latency`, `TLB miss latency`, and
`Estimated page-walk penalty` when `-latency-tlb-locality-kb` is not explicitly set.

### Canonical standalone TLB analysis

```bash
memory_benchmark -analyze-tlb -output tlb_analysis.json
```

Quick first checks in the output file:

- `tlb_analysis.l1_tlb_detection.boundary_locality_kb`
- `tlb_analysis.l2_tlb_detection.boundary_locality_kb`
- `tlb_analysis.page_walk_penalty.penalty_ns`

### Custom cache target

```bash
memory_benchmark -benchmark -cache-size 4096 -threads 1 -count 5 -output cache_4mb.json
```

### Cache-size sweep + trend plotting

```bash
./script-examples/latency_test_script.sh
python3 script-examples/plot_cache_percentiles.py script-examples/final_output.txt --metric median
```

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

When `-latency-tlb-locality-kb` is not explicitly provided, this section also prints:

- `TLB hit latency (16 KB locality)`
- `TLB miss latency (global random locality)`
- `Estimated page-walk penalty (miss - hit)`

### 4) Cache bandwidth and latency

L1/L2 or custom cache section, depending on `-cache-size` use.

### 5) Pattern benchmark output

Shows each pattern, relative percentage vs sequential forward baseline, and efficiency analysis:

- Sequential coherence
- Prefetcher effectiveness
- Cache thrashing potential
- TLB pressure

### 6) Statistics (`-count > 1`)

Includes values such as:

- Average
- P50 (Median)
- P90, P95, P99
- Std Dev
- Min / Max

When automatic TLB comparison is active (you did not explicitly set `-latency-tlb-locality-kb`),
statistics also include dedicated sections for:

- `TLB Hit Latency (ns)`
- `TLB Miss Latency (ns)`
- `Estimated Page-Walk Penalty (ns)`

For noisy systems, prioritize median and P95/P99 rather than single fastest/slowest values.

**Note:** Chain diagnostics (`pointer_count`, `unique_pages_touched`, etc.) appear in JSON output only when `-latency-stride-bytes` is explicitly set; they are not displayed in console output.

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
  "version": "0.55.0"
}
```

Note: The `configuration` block includes fields such as `latency_chain_mode` (the resolved pointer-chain mode used), `latency_tlb_locality_kb`, `use_custom_cache_size`, and other runtime settings. For the complete configuration schema, inspect a sample output file or examine the configuration builder code.

### Pattern benchmark JSON shape

```json
{
  "configuration": { ... },
  "execution_time_sec": 705.6,
  "patterns": { ... },
  "timestamp": "2026-03-09T15:10:01Z",
  "version": "0.55.0"
}
```

### Latency payload structure (current)

Latency values are structured objects, not scalars. The example below uses real values from a benchmark run.

When `-latency-stride-bytes` is explicitly set (with a non-default value), latency sections also include `chain_diagnostics`.

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

When run with `-analyze-tlb -output tlb_analysis.json`, the payload includes a dedicated `tlb_analysis` block.
Example below uses real values extracted from `results/0.53.8/MacMiniM4_analyze-tlb-chain-mode-random-box.json`:

```json
{
  "configuration": {
    "mode": "analyze_tlb",
    "latency_stride_bytes": 64,
    "buffer_size_mb": 1024
  },
  "tlb_analysis": {
    "sweep": [
      {
        "locality_bytes": 16384,
        "locality_kb": 16,
        "loop_latencies_ns": [25.957278, 25.965990, 25.916902],
        "p50_latency_ns": 25.982678
      }
    ],
    "l1_tlb_detection": {
      "detected": true,
      "segment_start_index": 0,
      "boundary_index": 7,
      "boundary_locality_bytes": 4194304,
      "boundary_locality_kb": 4096,
      "baseline_ns": 17.837711547619048,
      "boundary_latency_ns": 21.85142833333333,
      "step_ns": 4.0137167857142835,
      "step_percent": 0.22501298863362484,
      "persistent_jump": true,
      "confidence": "High",
      "inferred_entries": 256
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
      "confidence": "High",
      "inferred_entries": 512
    },
    "page_walk_penalty": {
      "available": true,
      "baseline_locality_kb": 16,
      "baseline_p50_ns": 15.070645833333334,
      "comparison_locality_mb": 512,
      "comparison_loop_latencies_ns": [95.34058666666667, 95.126475, 95.072195],
      "comparison_p50_ns": 95.179255833333334,
      "penalty_ns": 80.10860949999999
    }
  }
}
```

### Pattern keys (current)

- `sequential_forward`
- `sequential_reverse`
- `strided_64`
- `strided_4096`
- `strided_16384`
- `strided_2mb`
- `random`

Each pattern key contains a `bandwidth` sub-object with the same structure as `main_memory.bandwidth` (i.e., `read_gb_s`, `write_gb_s`, `copy_gb_s`, each with `values` and `statistics`).

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

# Pattern random read average
jq '.patterns.random.bandwidth.read_gb_s.statistics.average' patterns.json

# TLB L1 boundary locality (KB)
jq '.tlb_analysis.l1_tlb_detection.boundary_locality_kb' tlb_analysis.json

# TLB page-walk penalty (ns)
jq '.tlb_analysis.page_walk_penalty.penalty_ns' tlb_analysis.json
```

---

## Visualization Scripts

### `script-examples/latency_test_script.sh`

What it does:

- Sweeps multiple custom cache sizes
- Sweeps multiple `-latency-tlb-locality-kb` values
- Writes per-run JSON files under `script-examples/tmp/`
- Extracts `.cache.custom.latency.samples_ns.statistics` into `script-examples/final_output.txt`
- Clears `tmp` after extraction

Important: the script currently invokes `memory_benchmark -benchmark` from `PATH`. If you only built locally as `./memory_benchmark`, either install it to `PATH` or update `BENCHMARK_CMD` in the script.

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
2. Increase statistical depth (`-count 10` or higher, larger `-latency-samples`).
3. Compare **median/P95/P99**, not single-loop min/max.
4. Keep exact command lines identical across systems/runs.
5. Record context (apps active, external displays, power mode) with the result files.

### Reference numbers from README (Mac mini M4 sample)

From repository examples under lighter conditions, typical values are around:

- Main memory read: ~116 GB/s
- Main memory write: ~66 GB/s
- Main memory copy: ~106 GB/s
- Pattern random read: ~26-27 GB/s

Under heavy concurrent load, expect lower throughput and higher variance than these references.

---

## Best Practices and Pitfalls

### Best practices

- Use `caffeinate -i -d` for long runs.
- Use larger buffers (`512 MB` to `1024 MB+`) when targeting DRAM behavior.
- Use `-count > 1` and inspect percentiles.
- For cache-focused runs, prefer `-threads 1` unless testing aggregate behavior.

### Common pitfalls

- **Small buffers for DRAM claims**: often cache-dominated.
- **Assuming `-non-cacheable` is true uncached memory**: it is only a hint.
- **Comparing runs with different parameters**: invalidates conclusions.
- **Interpreting global-random and locality-window latency as identical tests**: chain construction differs intentionally.

---

## Troubleshooting

### "Incompatible flags" errors

Check mode combinations in [Mode Compatibility](#mode-compatibility).

### `-latency-tlb-locality-kb` rejected

Use `0` or a value that is an exact multiple of system page size.

### `-latency-chain-mode` rejected

Use one of: `auto`, `global-random`, `random-box`, `same-random-in-box`, `diff-random-in-box`.
If using a box mode (`random-box`, `same-random-in-box`, `diff-random-in-box`), also set `-latency-tlb-locality-kb` to a non-zero page-multiple value.

### Buffer size warnings/capping

The tool caps per-buffer size against memory safety limits. Use the printed configuration summary to see actual applied sizes.

### Script cannot find benchmark binary

`script-examples/latency_test_script.sh` calls `memory_benchmark` from `PATH`. Install the binary or adjust `BENCHMARK_CMD`.

### Plot script says no blocks found

Make sure you are passing `script-examples/final_output.txt` generated by the latency sweep script.

---

## Additional Resources

- [README.md](README.md) - project overview, install, examples
- [LATENCY_WHITEPAPER.md](LATENCY_WHITEPAPER.md) - pointer-chase latency methodology deep dive
- [TLB_ANALYSIS_WHITEPAPER.md](TLB_ANALYSIS_WHITEPAPER.md) - TLB analysis methodology and JSON schema
- [CORE_TO_CORE_WHITEPAPER.md](CORE_TO_CORE_WHITEPAPER.md) - Core-to-Core Cache-Line Handoff Latency Benchmark: methodology, assembly protocol, scheduler-hint scenarios, and JSON contract
- [TECHNICAL_SPECIFICATION.md](TECHNICAL_SPECIFICATION.md) - architecture and implementation details
- [CHANGELOG.md](CHANGELOG.md) - release history

Repository sample result files:

- `results/0.53.8/MacMiniM4_benchmark.json`
- `results/0.53.8/MacMiniM4_patterns.json`
- `results/0.53.8/MacMiniM4_analyzetlb.json`
- `results/0.53.8/MacMiniM4_core2core.json`

Command help:

```bash
memory_benchmark -h
```

---

**Last Updated**: 2026-04-05

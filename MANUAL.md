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

If installed from Homebrew:

```bash
memory_benchmark
```

If built from source:

```bash
./memory_benchmark
```

For longer runs, prevent sleep:

```bash
caffeinate -i -d ./memory_benchmark -count 10 -buffersize 1024
```

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

#### `-patterns`

- Runs only access-pattern benchmarks
- Skips standard bandwidth/latency sections

#### `-only-bandwidth`

- Runs bandwidth paths only
- Incompatible with: `-patterns`, `-cache-size`, `-latency-samples`

#### `-only-latency`

- Runs latency paths only
- Incompatible with: `-patterns`, `-iterations`
- Supports selective target disabling:
  - `-buffersize 0` disables main-memory latency
  - `-cache-size 0` disables cache latency
  - both zero is invalid

#### `-analyze-tlb`

- Runs standalone TLB analysis mode only
- Can be combined only with optional `-output <file>` and `-latency-stride-bytes <bytes>`
- Uses latency stride from `-latency-stride-bytes` (same default as standard latency mode), sweeps locality windows `max(16KB, 2*stride)` to `256MB`, and reports inferred L1/L2 TLB boundaries and entry counts
- Separately computes page-walk penalty as `P50(512MB) - P50(effective baseline locality)` when analysis buffer is at least `512MB`
- Detailed methodology and JSON contract: `TLB_ANALYSIS_WHITEPAPER.md`

### Latency-specific controls

#### `-latency-samples <count>`

- Sample count per latency test
- Default: `1000`
- Positive integer
- More samples improve percentile stability at the cost of run time

#### `-latency-stride-bytes <bytes>`

- Pointer-chain stride for latency tests
- Default: `136`
- Must be `> 0`
- Must be a multiple of pointer size (`8` bytes on Apple Silicon)
- Use smaller values (for example `64`) to increase same-page cache-line activity and reduce TLB sensitivity

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
./memory_benchmark -count 10 -buffersize 1024 -output full.json

# Pattern-only
./memory_benchmark -patterns -count 5 -buffersize 512 -output patterns.json

# Bandwidth-only
./memory_benchmark -only-bandwidth -threads 8 -count 5

# Latency-only (both main + cache)
./memory_benchmark -only-latency -latency-samples 5000 -count 10

# Latency-only (main memory only)
./memory_benchmark -only-latency -cache-size 0 -buffersize 1024

# Latency-only (cache only)
./memory_benchmark -only-latency -buffersize 0 -cache-size 2048

# Standalone TLB analysis
./memory_benchmark -analyze-tlb

# Standalone TLB analysis with JSON export
./memory_benchmark -analyze-tlb -output tlb_analysis.json

# Standalone TLB analysis with custom stride
./memory_benchmark -analyze-tlb -latency-stride-bytes 128 -output tlb_analysis_stride128.json
```

### Invalid combinations

```bash
# invalid: pattern mode with only-bandwidth
./memory_benchmark -patterns -only-bandwidth

# invalid: pattern mode with only-latency
./memory_benchmark -patterns -only-latency

# invalid: latency samples with only-bandwidth
./memory_benchmark -only-bandwidth -latency-samples 5000

# invalid: iterations with only-latency
./memory_benchmark -only-latency -iterations 2000

# invalid: both latency targets disabled
./memory_benchmark -only-latency -buffersize 0 -cache-size 0

# invalid: analyze-tlb with unsupported extra option
./memory_benchmark -analyze-tlb -buffersize 1024
```

---

## Common Workflows

### Quick baseline

```bash
./memory_benchmark
```

Good for a fast health check.

### Statistical baseline (recommended)

```bash
caffeinate -i -d ./memory_benchmark -count 10 -buffersize 1024 -output baseline.json
```

Use this for comparisons across machines or software versions.

### Pattern analysis

```bash
./memory_benchmark -patterns -count 10 -buffersize 512 -output patterns.json
```

Shows how bandwidth changes under different access patterns.

### Latency analysis with TLB-locality control

```bash
# default locality mode
./memory_benchmark -only-latency -buffersize 1024 -latency-samples 5000 -count 10 -output lat_tlb16.json

# global random chain
./memory_benchmark -only-latency -buffersize 1024 -latency-samples 5000 -latency-tlb-locality-kb 0 -count 10 -output lat_global.json
```

### Regular benchmark with automatic DRAM TLB breakdown

```bash
./memory_benchmark -latency-stride-bytes 128 -count 1
```

This prints `Average latency` plus auto-derived `TLB hit latency`, `TLB miss latency`, and
`Estimated page-walk penalty` when `-latency-tlb-locality-kb` is not explicitly set.

### Canonical standalone TLB analysis

```bash
./memory_benchmark -analyze-tlb -output tlb_analysis.json
```

Quick first checks in the output file:

- `tlb_analysis.l1_tlb_detection.boundary_locality_kb`
- `tlb_analysis.l2_tlb_detection.boundary_locality_kb`
- `tlb_analysis.page_walk_penalty.penalty_ns`

### Custom cache target

```bash
./memory_benchmark -cache-size 4096 -threads 1 -count 5 -output cache_4mb.json
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

- Buffer size and total allocation estimate
- Loop/iteration/sample configuration
- Thread count
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
  "version": "0.53.1"
}
```

### Pattern benchmark JSON shape

```json
{
  "configuration": { ... },
  "execution_time_sec": 705.6,
  "patterns": { ... },
  "timestamp": "2026-03-09T15:10:01Z",
  "version": "0.53.1"
}
```

### Latency payload structure (current)

Latency values are structured objects, not scalars:

When `-latency-stride-bytes` is explicitly set, latency sections also include `chain_diagnostics`.

```json
"latency": {
  "average_ns": {
    "values": [26.80, 27.73, 26.62],
    "statistics": {
      "average": 26.75,
      "median": 26.71,
      "p90": 27.30,
      "p95": 27.51,
      "p99": 27.68,
      "stddev": 0.47,
      "min": 26.24,
      "max": 27.73
    }
  },
  "samples_ns": [26.86, 26.77, 26.69],
  "samples_statistics": {
    "average": 26.75,
    "median": 26.72,
    "p90": 27.30,
    "p95": 27.51,
    "p99": 27.68,
    "stddev": 0.47,
    "min": 26.24,
    "max": 27.73
  },
  "auto_tlb_breakdown": {
    "tlb_hit_ns": {
      "values": [26.80, 27.73, 26.62],
      "statistics": {
        "average": 26.75,
        "median": 26.71,
        "p90": 27.30,
        "p95": 27.51,
        "p99": 27.68,
        "stddev": 0.47,
        "min": 26.24,
        "max": 27.73
      }
    },
    "tlb_miss_ns": {
      "values": [89.10, 90.45, 91.02],
      "statistics": {
        "average": 90.19,
        "median": 90.45,
        "p90": 90.91,
        "p95": 90.96,
        "p99": 91.01,
        "stddev": 0.99,
        "min": 89.10,
        "max": 91.02
      }
    },
    "page_walk_penalty_ns": {
      "values": [62.30, 62.72, 64.40],
      "statistics": {
        "average": 63.14,
        "median": 62.72,
        "p90": 64.06,
        "p95": 64.23,
        "p99": 64.37,
        "stddev": 1.12,
        "min": 62.30,
        "max": 64.40
      }
    }
  },
  "chain_diagnostics": {
    "pointer_count": 1057030,
    "unique_pages_touched": 65536,
    "page_size_bytes": 16384,
    "stride_bytes": 64
  }
}
```

### TLB analysis JSON (analyze mode)

When run with `-analyze-tlb -output tlb_analysis.json`, the payload includes a dedicated `tlb_analysis` block.
Example below is modeled from `results/macminim4_analyte-tbl_stride_136B.json`:

```json
{
  "configuration": {
    "mode": "analyze_tlb",
    "latency_stride_bytes": 136,
    "buffer_size_mb": 1024
  },
  "tlb_analysis": {
    "sweep": [
      {
        "locality_kb": 16,
        "loop_latencies_ns": [25.957278, 25.965990, 25.916902],
        "p50_latency_ns": 25.982678
      }
    ],
    "l1_tlb_detection": {
      "detected": true,
      "boundary_locality_kb": 4096,
      "inferred_entries": 256,
      "confidence": "High"
    },
    "l2_tlb_detection": {
      "detected": true,
      "boundary_locality_kb": 8192,
      "inferred_entries": 512,
      "confidence": "High"
    },
    "page_walk_penalty": {
      "available": true,
      "baseline_locality_kb": 16,
      "comparison_locality_mb": 512,
      "baseline_p50_ns": 25.982678,
      "comparison_p50_ns": 98.832814,
      "penalty_ns": 72.850136
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

### Useful JSON inspection commands

```bash
# Pretty print
python3 -m json.tool results.json

# Main memory read median
jq '.main_memory.bandwidth.read_gb_s.statistics.median' results.json

# Main memory latency P95 from sample distribution
jq '.main_memory.latency.samples_statistics.p95' results.json

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
- Extracts `.cache.custom.latency.samples_statistics` into `script-examples/final_output.txt`
- Clears `tmp` after extraction

Important: the script currently invokes `memory_benchmark` from `PATH`. If you only built locally as `./memory_benchmark`, either install it to `PATH` or update `BENCHMARK_CMD` in the script.

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
- [TECHNICAL_SPECIFICATION.md](TECHNICAL_SPECIFICATION.md) - architecture and implementation details
- [CHANGELOG.md](CHANGELOG.md) - release history

Repository sample result files:

- `results/macminim4_benchmark_count10.json`
- `results/macminim4_patterns_count10.json`

Command help:

```bash
./memory_benchmark -h
```

---

**Last Updated**: 2026-03-12

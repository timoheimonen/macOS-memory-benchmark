# macOS Memory Benchmark - User Manual

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Understanding Key Concepts](#understanding-key-concepts)
4. [Command-Line Options Reference](#command-line-options-reference)
5. [Common Usage Workflows](#common-usage-workflows)
6. [Best Practices](#best-practices)
7. [Understanding Output Sections](#understanding-output-sections)
8. [JSON Output Format](#json-output-format)
9. [Advanced Usage](#advanced-usage)
10. [Additional Resources](#additional-resources)

---

## Introduction

This manual provides comprehensive guidance for using the macOS Apple Silicon Memory Benchmark tool. This low-level tool measures memory performance characteristics on Apple Silicon systems, including:

- Main memory and cache bandwidth (read, write, copy operations)
- Main memory and cache latency
- Performance across different memory access patterns

The tool is specifically designed for comparing memory performance across different generations of Apple Silicon chips (M1, M2, M3, M4, M5, etc.).

**Additional Documentation:**
- [README.md](README.md) - Overview, installation, and quick examples
- [TECHNICAL_SPECIFICATION.md](TECHNICAL_SPECIFICATION.md) - Detailed implementation and architecture
- [GitHub Repository](https://github.com/timoheimonen/macOS-memory-benchmark)

---

## Getting Started

### Prerequisites

- macOS running on Apple Silicon (M1, M2, M3, M4, M5, or later)
- Xcode Command Line Tools (includes `clang++` compiler and `as` assembler)

Install Command Line Tools:
```bash
xcode-select --install
```

### Installation

**Option 1: Install with Homebrew (Recommended)**
```bash
brew install timoheimonen/macOS-memory-benchmark/memory-benchmark
```

**Option 2: Build from Source**
```bash
# Clone the repository
git clone https://github.com/timoheimonen/macOS-memory-benchmark.git
cd macOS-memory-benchmark

# Build
make

# The executable will be created as ./memory_benchmark
```

### Your First Run

Run the benchmark with default settings:
```bash
./memory_benchmark
```

This will:
- Use 512 MB buffers
- Run 1000 iterations of read/write/copy tests
- Perform one complete benchmark loop
- Test both main memory and cache (L1 and L2)
- Display results in the terminal

The benchmark will take approximately 40-60 seconds on most Apple Silicon systems.

---

## Understanding Key Concepts

Before diving into advanced usage, it's helpful to understand the key concepts this tool measures.

### Memory Hierarchy

Modern CPUs use a hierarchical memory system for optimal performance:

```
┌──────────────────────────────────────────┐
│  CPU Core                                 │
│  ┌────────────────────────────────────┐  │
│  │  L1 Cache (fastest, smallest)      │  │  ~128 KB per core
│  │  Latency: ~1 ns                    │  │  Bandwidth: 100-200 GB/s
│  └────────────────────────────────────┘  │
│                                           │
│  ┌────────────────────────────────────┐  │
│  │  L2 Cache (fast, larger)           │  │  ~4-16 MB per cluster
│  │  Latency: ~5 ns                    │  │  Bandwidth: 50-150 GB/s
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────┐
│  Main Memory / DRAM (slower, largest)    │  8-128+ GB
│  Latency: ~80-120 ns                     │  Bandwidth: 50-120 GB/s
└──────────────────────────────────────────┘
```

**Key Points:**
- **L1 Cache**: Fastest but smallest, located on each CPU core
- **L2 Cache**: Larger but slightly slower, shared within core clusters
- **Main Memory (DRAM)**: Much larger but significantly slower

### Bandwidth vs Latency

These are two fundamental performance metrics:

**Bandwidth** measures **throughput** - how much data can be transferred per second:
- Measured in GB/s (gigabytes per second)
- Important for: large data transfers, video processing, scientific computing
- Think of it as: the width of a highway (how many cars can travel simultaneously)

**Latency** measures **access time** - how long it takes to retrieve a single piece of data:
- Measured in nanoseconds (ns)
- Important for: pointer chasing, random access, linked data structures
- Think of it as: the speed limit on a highway (how fast each car can go)

**Why measure both?** Some workloads are bandwidth-limited (video encoding), while others are latency-limited (database queries). Different applications benefit from different characteristics.

### Access Patterns

How you access memory significantly affects performance:

**Sequential Access**: Reading/writing memory in order
- **Pros**: Excellent for prefetching, cache-friendly, highest bandwidth
- **Example**: Processing an array from start to end

**Strided Access**: Accessing memory at fixed intervals (e.g., every 64 bytes)
- **Pros**: Can still benefit from prefetching if stride is predictable
- **Example**: Accessing column in a row-major matrix

**Random Access**: Accessing memory locations in unpredictable order
- **Cons**: Defeats prefetching, causes cache misses, lowest performance
- **Example**: Following pointers in a linked list

The `-patterns` option tests all these patterns to understand how your system's prefetcher and cache behave.

### Multi-threading

Different benchmarks use different threading strategies:

**Multi-threaded** (using all cores):
- Used for: Main memory bandwidth tests
- Why: Maximum throughput by saturating memory bandwidth
- Default behavior: Uses all available CPU cores

**Single-threaded**:
- Used for: Cache tests and latency measurements
- Why: More accurate per-core cache measurements
- Can be overridden with `-threads 1`

---

## Command-Line Options Reference

### Buffer and Iteration Control

#### `-buffersize <MB>`
Sets the size of each memory buffer in megabytes.

**Default**: 512 MB

**When to use different sizes:**
- **Small (64-256 MB)**: Faster tests, but may be cache-dominated for main memory tests
- **Medium (512-1024 MB)**: Good balance, recommended for most use cases
- **Large (2048+ MB)**: Most accurate main memory measurements (if RAM permits)

**Important**: The tool allocates 3 buffers, so total memory usage is `3 × buffersize`. Maximum is automatically capped at ~80% of available system memory.

**Example:**
```bash
./memory_benchmark -buffersize 1024
```

#### `-iterations <count>`
Number of times to repeat read/write/copy operations.

**Default**: 1000

**Impact:**
- **More iterations**: More stable measurements, longer execution time
- **Fewer iterations**: Faster tests, more variability

**Range**: 1 to 10000

**Example:**
```bash
./memory_benchmark -iterations 2000
```

#### `-count <count>`
Number of complete benchmark loops to run.

**Default**: 1

**Why use multiple loops:**
- Provides statistical data (min, max, mean, percentiles, standard deviation)
- Helps identify measurement variance
- Essential for reliable chip comparisons

**When to use:**
- Use `-count 5` or `-count 10` for statistical confidence
- Use `-count 1` for quick checks

**Example:**
```bash
./memory_benchmark -count 10
```

### Test Selection

#### `-patterns`
Run memory access pattern benchmarks instead of standard bandwidth/latency tests.

**What it tests:**
- Sequential forward and reverse
- Strided access (cache-line and page-level)
- Random access

**Use cases:**
- Understanding prefetcher effectiveness
- Analyzing cache thrashing
- Comparing pattern performance across chips

**Example:**
```bash
./memory_benchmark -patterns -buffersize 512
```

#### `-only-bandwidth`
Run only bandwidth tests, skip all latency tests.

**Use cases:**
- Faster execution when latency isn't needed
- Focus on throughput measurements

**Cannot be combined with**: `-patterns`, `-cache-size`, `-latency-samples`

**Example:**
```bash
./memory_benchmark -only-bandwidth
```

#### `-only-latency`
Run only latency tests, skip all bandwidth tests.

**Use cases:**
- Quick latency-focused measurements
- Analyzing access time characteristics

**Cannot be combined with**: `-patterns`, `-iterations`

**Example:**
```bash
./memory_benchmark -only-latency
```

### Cache Configuration

#### `-cache-size <KB>`
Specify a custom cache size to test (in kilobytes).

**Range**: 16 KB to 524288 KB (512 MB)

**Behavior:**
- Skips automatic L1/L2 detection
- Runs bandwidth and latency tests for the specified cache size only
- Uses 100% of specified size for actual buffer

**Use cases:**
- Testing specific cache sizes
- Comparing different cache configurations
- Custom cache level analysis

**Example:**
```bash
./memory_benchmark -cache-size 256 -threads 1
```

### Performance Options

#### `-threads <count>`
Specify number of threads to use for benchmarks.

**Default**: Automatically detected CPU core count

**Guidelines:**
- **All cores**: Maximum main memory bandwidth (default)
- **Single thread** (`-threads 1`): More accurate per-core cache measurements
- **Custom**: Test specific threading scenarios

**Note**: If specified value exceeds available cores, it will be capped automatically with a warning.

**Example:**
```bash
./memory_benchmark -threads 4
```

#### `-latency-samples <count>`
Number of samples to collect per latency test.

**Default**: 1000

**Impact:**
- More samples: Better percentile accuracy, longer test time
- Fewer samples: Faster tests, less statistical detail

**Range**: 1 to 1000000

**Example:**
```bash
./memory_benchmark -latency-samples 5000
```

#### `-non-cacheable`
Apply cache-discouraging hints to memory buffers.

**What it does:**
- Uses `madvise()` system call with `MADV_RANDOM` hint
- Suggests to OS that access pattern is random (discourages caching)

**Important Limitations:**
- This is a **best-effort hint**, not a guarantee
- User-space applications cannot create truly non-cacheable memory on macOS
- May reduce but not eliminate caching
- Actual behavior depends on CPU and kernel implementation

**Use cases:**
- Attempting to measure true DRAM performance
- Experimental cache behavior analysis

**Example:**
```bash
./memory_benchmark -non-cacheable
```

### Output Options

#### `-output <file>`
Save benchmark results to JSON file.

**Path handling:**
- Relative paths: Saved in current working directory
- Absolute paths: Saved to specified location
- Creates parent directories if needed

**Use cases:**
- Data visualization with Python scripts
- Comparing results across different systems
- Long-term performance tracking

**Example:**
```bash
./memory_benchmark -count 10 -output results.json
```

#### `-h` or `--help`
Display help message with all options and exit.

**Example:**
```bash
./memory_benchmark --help
```

---

## Common Usage Workflows

### Quick System Check

**Scenario**: You want a fast assessment of your system's memory performance.

**Command:**
```bash
./memory_benchmark
```

**Details:**
- Uses default settings (512 MB, 1000 iterations, 1 loop)
- Runs in ~40-60 seconds
- Tests main memory and cache bandwidth/latency
- Good for: Initial system assessment, quick comparisons

### Comprehensive Benchmark

**Scenario**: You want thorough, statistically significant results.

**Command:**
```bash
caffeinate -i -d ./memory_benchmark -count 10 -buffersize 1024 -output comprehensive_results.json
```

**Details:**
- **`-count 10`**: Provides statistical data (percentiles, standard deviation)
- **`-buffersize 1024`**: Larger buffer for accurate main memory measurement
- **`-output`**: Saves results for analysis
- **`caffeinate -i -d`**: Prevents system sleep during long test (~7-10 minutes)

**Best for**: Definitive system characterization, chip comparisons, publication-quality data

### Access Pattern Analysis

**Scenario**: You want to understand how different access patterns affect performance.

**Command:**
```bash
caffeinate -i -d ./memory_benchmark -patterns -buffersize 512 -output pattern_results.json
```

**Details:**
- **`-patterns`**: Runs 5 different access patterns (sequential, strided, random)
- Tests read, write, and copy for each pattern
- Shows percentage degradation from sequential baseline
- Reveals prefetcher effectiveness and cache behavior

**Best for**: Understanding memory subsystem characteristics, compiler optimization decisions

### Cache-Specific Testing

**Scenario**: You want to test a specific cache size in detail.

**Command:**
```bash
./memory_benchmark -cache-size 256 -threads 1 -count 5
```

**Details:**
- **`-cache-size 256`**: Tests 256 KB cache (might be L2 on some systems)
- **`-threads 1`**: Single-threaded for per-core accuracy
- **`-count 5`**: Multiple runs for statistical confidence
- Skips L1/L2 auto-detection

**Best for**: Detailed cache analysis, comparing cache sizes across chips

### Exporting for Visualization

**Scenario**: You want to create graphs and visualizations of benchmark results.

**Command:**
```bash
./memory_benchmark -count 10 -output results_$(date +%Y%m%d_%H%M%S).json
```

**Details:**
- Saves timestamped JSON file
- Contains all benchmark data, configuration, and statistics
- Can be visualized using Python scripts in `script-examples/` directory

**Follow-up**: Use provided visualization scripts
```bash
python3 script-examples/plot_cache_percentiles.py results_20250104_143022.json
```

**Best for**: Creating publication-quality graphs, comparing multiple systems

---

## Best Practices

### For Accurate Measurements

#### Choose Appropriate Buffer Sizes

**Main Memory Tests:**
- **Minimum**: 512 MB (recommended)
- **Optimal**: 1024 MB or larger
- **Why**: Small buffers (< 512 MB) can fit partially in cache, giving misleading "main memory" results that actually measure cache performance

**Cache Tests:**
- Automatically calculated by the tool (75% of L1, 10% of L2)
- No manual adjustment needed

#### Minimize System Load

Before running benchmarks:
1. Close unnecessary applications
2. Stop background processes (backups, indexing, etc.)
3. Disconnect external displays if testing memory bandwidth
4. Let system idle for 30 seconds before starting

#### Use `caffeinate` for Long Runs

For tests longer than 2 minutes, prevent system sleep:
```bash
caffeinate -i -d ./memory_benchmark -count 10 -buffersize 1024
```

**Flags explained:**
- `-i`: Prevent system idle sleep
- `-d`: Prevent display sleep

### Comparing Different Apple Silicon Chips

To fairly compare M1, M2, M3, M4, etc., use **identical test parameters** on all systems.

**Recommended comparison setup:**
```bash
caffeinate -i -d ./memory_benchmark \
  -count 10 \
  -buffersize 1024 \
  -iterations 1000 \
  -output m4_results.json
```

**Key guidelines:**
1. **Use same buffer size** across all systems (1024 MB recommended)
2. **Use same iteration count** (1000 is good)
3. **Run multiple loops** (`-count 10`) for statistical confidence
4. **Export to JSON** for side-by-side comparison
5. **Test under same conditions** (idle system, same macOS version if possible)

**Comparing results:**
- Compare mean values for bandwidth/latency
- Check standard deviation to ensure stable measurements
- Use percentiles (P90, P95, P99) to identify outliers
- Visualize with provided Python scripts

**Example comparison workflow:**
```bash
# On M1 Mac
./memory_benchmark -count 10 -buffersize 1024 -output m1_results.json

# On M2 Mac
./memory_benchmark -count 10 -buffersize 1024 -output m2_results.json

# On M4 Mac
./memory_benchmark -count 10 -buffersize 1024 -output m4_results.json

# Compare visually
python3 script-examples/plot_cache_percentiles.py m1_results.json m2_results.json m4_results.json
```

### Thread Count Guidelines

**Main Memory Bandwidth Tests:**
- **Use all cores** (default behavior)
- Maximizes memory bandwidth utilization
- Reflects real-world multi-threaded application performance

**Cache Bandwidth Tests:**
- **Consider single-threaded** (`-threads 1`) for per-core measurements
- Default multi-threaded tests show aggregate cache performance
- Single-threaded more accurately reflects per-core cache characteristics

**Latency Tests:**
- Always single-threaded internally (pointer chasing is inherently serial)
- `-threads` parameter doesn't affect latency measurements

**Example for single-threaded cache focus:**
```bash
./memory_benchmark -threads 1 -count 5
```

### Common Pitfalls to Avoid

#### 1. Small Buffers for Main Memory Tests

**Problem**: Using buffer sizes < 512 MB for main memory measurements

**Why it's wrong**: Data fits in large L2 cache, measuring cache performance instead of DRAM

**Solution**: Use `-buffersize 512` or larger (1024 recommended)

#### 2. Running with Active System Load

**Problem**: Running benchmarks while other applications are active

**Why it's wrong**: Competing memory access distorts measurements

**Solution**: Close applications, let system idle before testing

#### 3. Forgetting `caffeinate` for Long Runs

**Problem**: System sleeps during multi-loop benchmarks

**Why it's wrong**: Interrupted tests, incomplete data, system state changes on wake

**Solution**: Always use `caffeinate -i -d` for runs longer than 2 minutes

#### 4. Misunderstanding `-non-cacheable`

**Problem**: Expecting truly non-cached memory access

**Why it's wrong**: User-space applications cannot create non-cacheable memory on macOS

**Reality**: The flag provides cache-discouraging **hints**, not guarantees

**Solution**: Understand limitations, use large buffers and multiple iterations for more reliable measurements

#### 5. Inconsistent Test Parameters for Comparisons

**Problem**: Comparing results with different buffer sizes or iteration counts

**Why it's wrong**: Different test conditions make comparisons meaningless

**Solution**: Document test parameters, use identical settings for all comparison systems

---

## Understanding Output Sections

### Configuration Summary

Displayed before tests begin, shows all test parameters:

```text
Buffer Size (per buffer): 512.00 MiB (512 MB requested/capped)
Total Allocation Size: ~1536.00 MiB (for 3 buffers)
Iterations (per R/W/Copy test per loop): 1000
Loop Count (total benchmark repetitions): 1
Non-Cacheable Memory Hints: Enabled

Processor Name: Apple M4
  Performance Cores: 4
  Efficiency Cores: 6
  Total CPU Cores Detected: 10

Detected Cache Sizes:
  L1 Cache Size: 128.00 KB (per P-core)
  L2 Cache Size: 16.00 MB (per P-core cluster)
```

**Key information:**
- **Buffer sizes**: Verify requested vs actual (may be capped by available memory)
- **Total allocation**: 3× buffer size (src, dst, latency buffers)
- **CPU info**: Core counts affect multi-threaded test performance
- **Cache sizes**: Auto-detected, used for cache buffer sizing

### Main Memory Results

Shows DRAM performance (not cache):

```text
Main Memory Bandwidth Tests (multi-threaded, 10 threads):
  Read : 114.74709 GB/s (Total time: 4.67873 s)
  Write: 68.55840 GB/s (Total time: 7.83086 s)
  Copy : 105.60609 GB/s (Total time: 10.16742 s)

Main Memory Latency Test (single-threaded, pointer chase):
  Total time: 19.47850 s
  Average latency: 97.39 ns
```

**Understanding the numbers:**

**Bandwidth:**
- **Read**: Sequential read throughput (GB/s)
- **Write**: Sequential write throughput (GB/s)
- **Copy**: Memory-to-memory copy throughput (GB/s)
- Higher is better
- Multi-threaded by default for maximum throughput

**Latency:**
- Average time to access a single memory location (nanoseconds)
- Lower is better
- Single-threaded (pointer chasing is sequential)
- Typical range: 80-120 ns for modern Apple Silicon DRAM

### Cache Results

Shows L1 and L2 cache performance:

```text
Cache Bandwidth Tests (single-threaded):
  L1 Cache:
    Read : 139.59918 GB/s (Buffer size: 96.00 KB)
    Write: 72.70513 GB/s
    Copy : 163.58213 GB/s
  L2 Cache:
    Read : 120.46392 GB/s (Buffer size: 1.60 MB)
    Write: 45.08152 GB/s
    Copy : 126.15097 GB/s

Cache Latency Tests (single-threaded, pointer chase):
  L1 Cache: 0.69 ns (Buffer size: 96.00 KB)
  L2 Cache: 4.90 ns (Buffer size: 1.60 MB)
```

**Buffer sizes:**
- L1: 75% of detected L1 cache size
- L2: 10% of detected L2 cache size
- Ensures data fits in target cache level

**Expected characteristics:**
- L1 bandwidth higher than L2, L2 higher than main memory
- L1 latency much lower than L2, L2 lower than main memory
- Typical L1 latency: 0.5-1 ns
- Typical L2 latency: 3-7 ns

### Pattern Benchmark Results

Shows performance across different access patterns:

```text
Sequential Forward:
  Read : 113.385 GB/s
  Write: 105.412 GB/s
  Copy : 104.176 GB/s

Sequential Reverse:
  Read : 113.228 GB/s (-0.1%)
  Write: 104.662 GB/s (-0.7%)
  Copy : 103.399 GB/s (-0.7%)

Strided (Cache Line - 64B):
  Read : 56.387 GB/s (-50.3%)
  Write: 51.807 GB/s (-50.9%)
  Copy : 68.214 GB/s (-34.5%)

Strided (Page - 4096B):
  Read : 26.390 GB/s (-76.7%)
  Write: 52.138 GB/s (-50.5%)
  Copy : 33.388 GB/s (-68.0%)

Random Uniform:
  Read : 26.504 GB/s (-76.6%)
  Write: 43.328 GB/s (-58.9%)
  Copy : 31.687 GB/s (-69.6%)
```

**Patterns explained:**

- **Sequential Forward**: Baseline (optimal access pattern)
- **Sequential Reverse**: Backward iteration (tests prefetcher direction sensitivity)
- **Strided 64B**: One element per cache line (tests cache line utilization)
- **Strided 4096B**: One element per page (tests TLB pressure)
- **Random**: Unpredictable access (worst case for prefetcher)

**Percentage in parentheses**: Performance degradation compared to sequential forward baseline

**Pattern efficiency metrics:**
```text
Pattern Efficiency Analysis:
- Sequential coherence: 99.5%
- Prefetcher effectiveness: 54.6%
- Cache thrashing potential: High
- TLB pressure: Minimal
```

These derived metrics summarize access pattern characteristics.

### Statistics (Multiple Loops)

When using `-count > 1`, statistical analysis is displayed:

```text
Statistics for 10 loops:

Main Memory Bandwidth:
  Read:
    Min: 113.45 GB/s
    Max: 116.38 GB/s
    Mean: 115.12 GB/s
    P50 (Median): 115.18 GB/s
    P90: 115.89 GB/s
    P95: 116.02 GB/s
    P99: 116.28 GB/s
    Std Dev: 0.82 GB/s
```

**Metrics explained:**

- **Min/Max**: Fastest and slowest measurements
- **Mean**: Average across all loops
- **P50 (Median)**: Middle value (50th percentile)
- **P90**: 90% of measurements were at or below this value
- **P95/P99**: 95th and 99th percentiles (identify outliers)
- **Std Dev**: Standard deviation (measurement consistency)

**Low standard deviation** (< 1% of mean) indicates stable, reliable measurements.

---

## JSON Output Format

When using `-output <file>`, results are saved in JSON format for programmatic analysis.

### Structure Overview

```json
{
  "version": "1.0",
  "timestamp": "2025-01-04T14:30:22Z",
  "configuration": { ... },
  "main_memory": { ... },
  "cache": { ... },
  "patterns": { ... },
  "execution_time_sec": 217.86
}
```

### Key Fields

#### Configuration Section
```json
"configuration": {
  "buffer_size_mb": 512,
  "buffer_size_bytes": 536870912,
  "iterations": 1000,
  "loop_count": 10,
  "cpu_name": "Apple M4",
  "macos_version": "26.2",
  "performance_cores": 4,
  "efficiency_cores": 6,
  "total_threads": 10,
  "l1_cache_size_bytes": 131072,
  "l2_cache_size_bytes": 16777216,
  "use_non_cacheable": true,
  "latency_sample_count": 1000
}
```

#### Main Memory Section
```json
"main_memory": {
  "bandwidth": {
    "read_gb_s": {
      "values": [115.33, 116.10, 116.38, ...],
      "statistics": {
        "average": 116.01,
        "min": 115.33,
        "max": 116.38,
        "median": 116.10,
        "p90": 116.33,
        "p95": 116.36,
        "p99": 116.38,
        "stddev": 0.41
      }
    },
    "write_gb_s": { ... },
    "copy_gb_s": { ... }
  },
  "latency": {
    "average_ns": 97.39,
    "samples": [95.2, 98.3, 96.7, ...]
  }
}
```

#### Cache Section
```json
"cache": {
  "l1": {
    "bandwidth": {
      "read_gb_s": { ... },
      "write_gb_s": { ... },
      "copy_gb_s": { ... }
    },
    "latency_ns": {
      "values": [0.68, 0.69, 0.70, ...],
      "statistics": { ... }
    }
  },
  "l2": { ... }
}
```

### Using JSON Output

**View with command-line tools:**
```bash
# Pretty-print
cat results.json | python3 -m json.tool

# Extract specific value (using jq)
cat results.json | jq '.main_memory.bandwidth.read_gb_s.statistics.average'
```

**Visualize with provided scripts:**
```bash
# Plot cache latency percentiles
python3 script-examples/plot_cache_percentiles.py results.json

# Compare multiple systems
python3 script-examples/plot_cache_percentiles.py m1.json m2.json m4.json
```

**Example JSON files** are available in the `results/` directory:
- `benchmark_result_example_M4.json` - Standard benchmark
- `bandwidth_result_example_M4.json` - Bandwidth-only test
- `latency_results_example_M4.json` - Latency-only test
- `patterns_result_example_M4.json` - Pattern benchmarks

---

## Advanced Usage

### Statistical Analysis with Multiple Loops

For publication-quality data or definitive chip comparisons, run multiple loops to obtain statistical confidence.

**Recommended setup:**
```bash
caffeinate -i -d ./memory_benchmark \
  -count 10 \
  -buffersize 1024 \
  -iterations 2000 \
  -output statistical_results.json
```

**What you get:**
- Min, max, mean values
- Percentiles (P50, P90, P95, P99)
- Standard deviation
- All individual loop values for custom analysis

**Interpreting variance:**
- **Low std dev (< 1%)**: Stable, repeatable measurements
- **Medium std dev (1-3%)**: Some variance, acceptable for most purposes
- **High std dev (> 3%)**: Investigate system load, thermal throttling, background processes

**When to use:**
- Comparing different hardware generations
- Publishing benchmark results
- Validating system performance over time
- Detecting performance regressions

### Custom Cache Size Testing

Test specific cache sizes instead of auto-detected L1/L2.

**Use cases:**
1. **Testing hypothetical cache configurations**
2. **Comparing specific cache sizes across chips**
3. **Analyzing cache size vs performance relationship**

**Example - test 256 KB cache:**
```bash
./memory_benchmark -cache-size 256 -threads 1 -count 5
```

**Example - sweep multiple cache sizes:**
```bash
for size in 64 128 256 512 1024 2048; do
  ./memory_benchmark -cache-size $size -threads 1 -output cache_${size}kb.json
done
```

**Notes:**
- Skips L1/L2 auto-detection when `-cache-size` is specified
- Buffer size is 100% of specified cache size
- Single-threaded tests recommended for accuracy

### Combining Options

#### Valid Combinations

**Comprehensive statistical run:**
```bash
./memory_benchmark -count 10 -buffersize 1024 -iterations 2000 -output results.json
```

**Pattern analysis with statistics:**
```bash
./memory_benchmark -patterns -count 5 -buffersize 512 -output patterns.json
```

**Bandwidth-only multi-threaded:**
```bash
./memory_benchmark -only-bandwidth -threads 8 -count 5
```

**Latency-focused with many samples:**
```bash
./memory_benchmark -only-latency -latency-samples 10000 -count 10
```

**Custom cache with non-cacheable hints:**
```bash
./memory_benchmark -cache-size 512 -non-cacheable -threads 1
```

#### Invalid Combinations

These combinations will produce errors:

**Cannot combine `-patterns` with `-only-bandwidth` or `-only-latency`:**
```bash
# ERROR: Patterns is a separate test mode
./memory_benchmark -patterns -only-bandwidth
```

**Cannot use `-only-bandwidth` with `-latency-samples`:**
```bash
# ERROR: Bandwidth-only skips latency tests
./memory_benchmark -only-bandwidth -latency-samples 5000
```

**Cannot use `-only-latency` with `-iterations`:**
```bash
# ERROR: Latency tests don't use iterations parameter
./memory_benchmark -only-latency -iterations 2000
```

### Complex Example Commands

**Publication-quality comprehensive benchmark:**
```bash
caffeinate -i -d ./memory_benchmark \
  -count 20 \
  -buffersize 2048 \
  -iterations 2000 \
  -latency-samples 5000 \
  -threads 10 \
  -output publication_$(date +%Y%m%d)_$(uname -m).json
```

**Quick cache characterization:**
```bash
./memory_benchmark -only-bandwidth -threads 1 -count 3
```

**Memory subsystem stress test:**
```bash
caffeinate -i -d ./memory_benchmark \
  -buffersize 4096 \
  -iterations 5000 \
  -count 1 \
  -non-cacheable
```

---

## Additional Resources

### Documentation

- **[README.md](README.md)** - Quick start, installation, example output
- **[TECHNICAL_SPECIFICATION.md](TECHNICAL_SPECIFICATION.md)** - Architecture, implementation details, assembly code
- **This Manual** - Comprehensive usage guide

### Source Code Repository

- **GitHub**: [https://github.com/timoheimonen/macOS-memory-benchmark](https://github.com/timoheimonen/macOS-memory-benchmark)
- Report issues, contribute, view latest updates

### Example Scripts

The `script-examples/` directory contains Python scripts for visualizing JSON results:

- `plot_cache_percentiles.py` - Visualize cache latency percentiles from multiple benchmark runs

**Usage example:**
```bash
python3 script-examples/plot_cache_percentiles.py results.json
```

### Example Results

The `results/` directory contains example JSON output from Mac Mini M4:

- `benchmark_result_example_M4.json` - Complete benchmark with statistics
- `bandwidth_result_example_M4.json` - Bandwidth-only results
- `latency_results_example_M4.json` - Latency-only results
- `patterns_result_example_M4.json` - Access pattern analysis

Use these as references for expected JSON structure and typical M4 performance.

### Getting Help

- **Command-line help**: `./memory_benchmark -h`
- **GitHub Issues**: Report bugs or request features at repository
- **Technical details**: See TECHNICAL_SPECIFICATION.md for implementation questions

### License

This tool is licensed under GPL-3.0. See the LICENSE file for details.

Copyright 2025-2026 Timo Heimonen <timo.heimonen@proton.me>

---

**Last Updated**: 2026-01-04

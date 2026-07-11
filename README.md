# macOS Apple Silicon Memory Benchmark

![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-000000?logo=apple) ![CLI](https://img.shields.io/badge/CLI-Tool-00A8CC?logo=terminal) ![License](https://img.shields.io/badge/license-GPL--3.0-blue) ![Assembly](https://img.shields.io/badge/Assembly-ARM64-6E4C13) ![C++](https://img.shields.io/badge/C++-00599C?logo=cplusplus&logoColor=white)

`memory_benchmark` is a low-level command-line tool for measuring CPU and Metal GPU memory bandwidth, cache and main-memory latency, access-pattern performance, TLB behavior, and core-to-core cache-line handoff latency on Apple Silicon Macs.

It is designed for controlled microarchitectural investigation rather than a single synthetic score. CPU measurement paths use native ARM64 kernels; the standalone GPU mode uses runtime-compiled Metal compute kernels. Runs expose calibration, workload, completion, and repeatability metadata so results can be audited and compared.

![Cache latency on a MacBook Air M5 across working-set sizes, pointer strides, and TLB-locality configurations](pictures/MacBookAirM5_latency_vs_cache-stride-tlb.png)

*Cache latency on a MacBook Air M5 across working-set sizes, pointer strides, and TLB-locality configurations. Generated from multiple JSON result files using the included plotting tools.*

## Why This Tool?

- **Apple Silicon native:** C++17 and ARM64 assembly measurement paths for macOS.
- **Bandwidth and latency:** main-memory and cache read/write/copy throughput plus dependent pointer-chase latency.
- **Access-pattern analysis:** sequential, reverse, strided, and random workloads with exact effective-payload accounting.
- **Dedicated TLB analysis:** paired spread/packed chains, adaptive rounds, confidence intervals, and independent boundary validation.
- **Core-to-core analysis:** calibrated cache-line handoff measurements under scheduler-hint scenarios.
- **Metal GPU bandwidth:** standalone read/write/copy compute kernels with GPU timestamps and validation metadata.
- **Reproducible experiments:** explicit seeds, repeated loops, built-in Cartesian parameter sweeps, and checkpointed JSON output.

See [Measurement Capabilities](CAPABILITIES.md) for the full measurement scope and interpretation guidance.

## Platform Requirements

- macOS on Apple Silicon (ARM64)
- Xcode Command Line Tools for source builds
- GoogleTest from Homebrew for the test suite
- GPU mode: a unified-memory Metal device supporting `MTLGPUFamilyApple7` or a compatible later family

The build targets macOS 11.0 and links the system Metal and Foundation frameworks. GPU kernels are embedded MSL 2.3 source compiled at runtime, so the optional offline Metal Toolchain is not required. Passing the GPU capability check indicates compatibility; it does not mean performance has been validated on that device.

## Install

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

When using a source build without installing it to `PATH`, replace `memory_benchmark` in the examples with `./memory_benchmark`.

## Quick Start

Show the complete command-line reference:

```bash
memory_benchmark -h
```

Run the standard CPU bandwidth and latency benchmark:

```bash
memory_benchmark --benchmark
```

Run the standalone Metal GPU bandwidth suite:

```bash
memory_benchmark --gpu-bandwidth
```

For longer runs, prevent system sleep and collect repeated measurements:

```bash
caffeinate -i -d memory_benchmark --benchmark --count 10 --buffer-size 1024 --output baseline.json
```

## Benchmark Modes

| Mode | Purpose |
|---|---|
| `--benchmark` | Calibrated and balanced standard CPU benchmark for main-memory and cache bandwidth plus continuous-pass latency. Use `--only-bandwidth` or `--only-latency` to narrow the run. |
| `--patterns` | Effective read/write/copy bandwidth for sequential-forward, sequential-reverse, 64 B, 4096 B, 16384 B and 2 MiB virtual strides, and random access. |
| `--analyze-tlb` | Standalone paired spread/packed TLB analysis with adaptive measurement rounds, confidence intervals, and boundary validation. |
| `--analyze-core2core` | Calibrated two-thread cache-line handoff latency under best-effort macOS scheduler hints. |
| `--gpu-bandwidth` | Standalone Metal GPU read/write/copy effective compute-payload bandwidth. |
| `--sweep <key=a,b>` | Cartesian parameter sweep for supported CPU, pattern, TLB, and core-to-core modes; requires `--output`. GPU schema 1 does not support sweeps. |

Primary modes are intentionally separate and accept different option sets. Use `memory_benchmark -h` or the [User Manual](MANUAL.md) for defaults, valid combinations, and the complete option reference.

When `--iterations` is omitted, standard bandwidth, pattern, and GPU operations calibrate their work toward a bounded measurement duration. An explicit `--iterations` value selects fixed work. Standard latency headlines always come from a continuous dependent pointer-chase pass; optional sample windows are a separate distribution and do not define the headline.

## Representative Workflows

Repeated CPU baseline with JSON output:

```bash
caffeinate -i -d memory_benchmark --benchmark --count 10 --buffer-size 1024 --output baseline.json
```

Reproducible access-pattern comparison:

```bash
memory_benchmark --patterns --count 10 --buffer-size 512 --seed 123456789 --output patterns.json
```

Latency sweep over buffer size and pointer-chain locality:

```bash
memory_benchmark --benchmark --only-latency --count 5 \
  --sweep buffer-size=256,512,1024 \
  --sweep latency-tlb-locality-kb=16,1024,0 \
  --output latency_sweep.json
```

Standalone TLB analysis:

```bash
memory_benchmark --analyze-tlb --seed 123456789 --output tlb_analysis.json
```

Core-to-core handoff analysis with deeper sampling:

```bash
memory_benchmark --analyze-core2core --count 5 --latency-samples 2000 --output core2core.json
```

Reproducible fixed-work GPU run:

```bash
caffeinate -i -d memory_benchmark --gpu-bandwidth --buffer-size 512 \
  --iterations 24 --count 9 --seed 123456789 --output gpu_bandwidth.json
```

More workflows, including custom cache targets, latency-chain controls, density profiles, and sweep keys, are documented in the [User Manual](MANUAL.md).

## Interpreting Results

Treat benchmark values as measurements of the configured workload under the observed system conditions, not as immutable hardware specifications.

- Use identical commands, seeds, software versions, and system conditions when comparing runs.
- Prefer repeated loops. When `--count > 1`, the median (P50) is the headline; output also reports tail percentiles and variability metrics where applicable.
- Keep background load, power state, and thermal conditions consistent. Scheduling and other macOS activity can materially affect latency tails and variance.
- Use sufficiently large buffers for DRAM-focused CPU experiments. Small working sets can be cache-dominated.
- `--non-cacheable` applies best-effort cache-discouraging `madvise` hints; it does not create truly uncached memory.
- Pattern GB/s is exact **effective kernel payload bandwidth**, not observed physical cache-bus or DRAM traffic. `strided_2mb` describes a 2 MiB virtual-address stride and does not prove superpage backing.
- GPU GB/s is exact **effective compute-payload bandwidth** divided by Metal GPU time. Private storage is unified memory rather than separate VRAM, copy counts aggregate read plus write payload, and physical DRAM residency remains unverified.
- CPU and GPU GB/s values are not directly comparable: the kernels, timing boundaries, parallelism, resource modes, and validation work differ.
- TLB-locality controls pointer-chain construction, not hardware TLB residency. Standard locality comparisons combine cache, locality, and translation effects; use `--analyze-tlb` for controlled translation-boundary conclusions.
- Core-to-core results are scheduler-influenced handoff measurements. macOS user space cannot guarantee physical core pinning.

JSON output records completion and nullable measurement state instead of using zero for unavailable results. Consumers making conclusions should reject incomplete or interrupted runs according to the mode-specific status fields. Exact schema contracts and checkpoint behavior are documented in the [User Manual](MANUAL.md), [Technical Specification](TECHNICAL_SPECIFICATION.md), and mode whitepapers.

## Plotting Results

The repository includes scripts for cache/locality sweeps and percentile plots. Plotting requires Python 3 and `matplotlib`; the M4/M5 comparison script also uses `numpy`.

```bash
python3 -m pip install matplotlib numpy
./script-examples/latency_test_script.sh
python3 script-examples/plot_cache_percentiles.py \
  script-examples/final_output.txt --metric median
```

The sweep script invokes `memory_benchmark` from `PATH`. If you only built locally, install the binary or update `BENCHMARK_CMD` in the script. See the [User Manual](MANUAL.md#visualization-scripts) for supported inputs and metrics.

## Documentation

- [Measurement Capabilities](CAPABILITIES.md): what the tool measures and how those measurements should be interpreted.
- [User Manual](MANUAL.md): complete option reference, mode compatibility, workflows, output examples, and troubleshooting.
- [Technical Specification](TECHNICAL_SPECIFICATION.md): architecture, execution flow, memory model, and output contracts.
- [Latency Whitepaper](LATENCY_WHITEPAPER.md): dependent pointer-chase and sampling methodology.
- [TLB Analysis Whitepaper](TLB_ANALYSIS_WHITEPAPER.md): paired analysis, boundary rules, confidence model, and JSON verification contract.
- [Core-to-Core Whitepaper](CORE_TO_CORE_WHITEPAPER.md): LDAR/STLR handoff protocol, scheduler-hint scenarios, and JSON schema.
- [GPU Bandwidth Whitepaper](GPU_BANDWIDTH_WHITEPAPER.md): Metal methodology, timing, validation, resource model, and interpretation limits.

Runtime behavior and `memory_benchmark -h` are the authoritative sources when documentation differs.

## Development and Testing

Install GoogleTest and run the deterministic unit suite:

```bash
brew install googletest
make test
```

Run real Apple Silicon integration tests or the complete suite:

```bash
make test-integration
make test-all
```

Generate isolated LLVM production-source coverage reports under `/tmp`:

```bash
make coverage-unit
make coverage-all
```

See [AGENTS.md](AGENTS.md) for repository conventions and test taxonomy. API documentation can be generated with `make docs`.

## Scope and Safety

This project intentionally does not target Intel Macs or other operating systems, provide a GUI, or host a public leaderboard/backend.

The benchmark performs sustained, intensive memory operations. Use it at your own risk; the author is not responsible for instability, data loss, or hardware issues resulting from use.

## License

Copyright 2025-2026 Timo Heimonen \<timo.heimonen@proton.me\>

Licensed under the [GNU General Public License v3.0](LICENSE).

# TLB Analysis Whitepaper: `-analyze-tlb` Methodology

## 1. Purpose

This document specifies how `macOS-memory-benchmark` implements standalone TLB analysis mode (`-analyze-tlb`) in version `0.53.7`.

The goal is to provide a reproducible, implementation-accurate description of:

- measurement workflow,
- boundary detection logic,
- confidence scoring,
- derived metrics (entry reach and page-walk penalty),
- JSON verification payload.

This is an implementation whitepaper, not a generic architecture theory paper.

## 2. Scope and Source of Truth

Primary implementation references:

- `src/benchmark/tlb_analysis.cpp`
- `src/benchmark/tlb_analysis.h`
- `src/core/config/argument_parser.cpp`
- `tests/test_analysis.cpp`

Related user-facing docs:

- `README.md`
- `MANUAL.md`

## 3. CLI Contract

### 3.1 Accepted Forms

`-analyze-tlb` runs a dedicated analysis path and accepts only optional JSON output and optional latency stride override:

```bash
./memory_benchmark -analyze-tlb
./memory_benchmark -analyze-tlb -output tlb_analysis.json
./memory_benchmark -output tlb_analysis.json -analyze-tlb
./memory_benchmark -analyze-tlb -latency-stride-bytes 128 -output tlb_analysis_stride128.json
```

### 3.2 Rejected Combinations

All other options are rejected when `-analyze-tlb` is present.

Example (invalid):

```bash
./memory_benchmark -analyze-tlb -buffersize 1024
```

## 4. Measurement Workflow

### 4.1 Buffer Allocation Policy

The analysis tries these buffers in order:

1. `1024 MB`
2. `512 MB`
3. `256 MB`

If all fail, mode exits with an insufficient-memory error.

`mlock()` is attempted for the selected buffer; report shows whether lock succeeded.

### 4.2 Fixed Measurement Parameters

The mode uses fixed constants:

- `loops_per_point = 30`
- `accesses_per_loop = 25,000,000`

Stride behavior:

- Effective stride is taken from `-latency-stride-bytes`.
- Default stride is the same as standard latency mode default (`Constants::LATENCY_STRIDE_BYTES`).

Each loop rebuilds the latency chain for the target locality, performs warmup, runs pointer chase, and stores latency as `ns/access`.

Per-point central value is `P50` (median) over 30 loops.

### 4.3 Locality Sweep

Main sweep windows are selected from the canonical set:

- `16KB`, `64KB`, `128KB`, `256KB`, `512KB`
- `1MB`, `2MB`, `4MB`, `8MB`, `12MB`, `16MB`, `32MB`, `64MB`, `128MB`, `256MB`

Effective sweep start is stride-aware:

- `min_sweep_locality = max(16KB, 2 * stride)`
- points below `min_sweep_locality` are skipped
- if `min_sweep_locality` is not in the canonical set, it is inserted as the first sweep point

Separate page-walk comparison point:

- `512MB` (run only when selected buffer is at least `512MB`)

## 5. Boundary Detection Algorithm

Boundary detection is based on a running segment baseline.

For candidate index `i`:

- `baseline_ns = average(p50[segment_start .. i-1])`
- `step_ns = p50[i] - baseline_ns`
- `threshold_ns = max(2.0ns, baseline_ns * 0.10)`

Candidate passes threshold when:

- `step_ns >= threshold_ns`

### 5.1 TLB Guard (Cache-Transition Filter)

To avoid classifying early cache transitions as TLB boundaries, candidate locality must also satisfy:

`locality_bytes >= tlb_guard_bytes`

where:

`tlb_guard_bytes = max(2 * L1D_size_bytes, 64 * page_size_bytes)`

### 5.2 L1 and L2 Boundary Selection

- `L1 TLB boundary`: first candidate passing threshold + guard, scanning from sweep start.
- `L2 TLB boundary`: first candidate passing threshold + guard, scanning from the L1 segment start index.

Technical note (Apple Silicon):

- On Apple Silicon, the shared System Level Cache (SLC) can blur translation-only inflection points.
- In practice, the detected `L2 TLB boundary` should be interpreted as an inferred secondary translation-reach boundary, not a guaranteed pure architectural L2 TLB edge.

## 6. Confidence Model

Boundary confidence is classified by step strength and persistence:

- `strong_step = (step_ns >= 4.0) || (step_percent >= 0.15)`
- `persistent_jump = next_point_step >= threshold`

Confidence levels:

- **High**: `strong_step && persistent_jump`
- **Medium**: `strong_step || persistent_jump`
- **Low**: otherwise (still passed threshold)

## 7. Derived Metrics

### 7.1 Inferred Entry Reach

For detected boundaries:

`inferred_entries = boundary_locality_bytes / page_size_bytes`

This is reported separately for L1 and L2 sections.

### 7.2 Page-Walk Penalty

Page-walk penalty is intentionally computed independently from L1/L2 boundary step values:

`page_walk_penalty_ns = P50(512MB) - P50(effective baseline locality)`

where:

- `effective baseline locality = first point in the stride-aware sweep`

Availability rule:

- available only when selected analysis buffer is `>= 512MB`
- otherwise reported as unavailable (`N/A`) with reason.

## 8. Console Report Contract

Report always includes:

- CPU, page size, selected buffer, stride, loops/accesses config
- `[L1 TLB Detection]`
- `[L2 TLB / Page Walk]`

Detected boundary sections include:

- boundary locality,
- inferred entries,
- confidence string with step (`ns` and `%`).

Page-walk line uses explicit dynamic endpoints (`baseline -> 512MB`).

## 9. JSON Output Contract (`-output`)

When `-output <file>` is provided with `-analyze-tlb`, output includes:

- top-level metadata:
  - `configuration`
  - `execution_time_sec`
  - `tlb_analysis`
  - `timestamp`
  - `version`

- `configuration` contains mode and run constants (CPU/page/L1D/guard/stride/loops/accesses/buffer info).

- `tlb_analysis` contains:
  - `sweep[]` with raw `loop_latencies_ns` and per-point `p50_latency_ns`
  - `l1_tlb_detection`
  - `l2_tlb_detection`
  - `page_walk_penalty` block (`available`, baseline/comparison metadata, raw comparison loops, `penalty_ns` when available)

This payload is designed for full post-run verification and reproducibility checks.

## 10. Worked Example (Apple M4)

Example file:

- `results/0.53.7/MacMiniM4_analyzetlb.json`

Observed fields in this sample:

- `tlb_guard_bytes = 1048576` (`1MB`)
- `l1_tlb_detection.boundary_locality_kb = 4096` and `inferred_entries = 256`
- `l2_tlb_detection.boundary_locality_kb = 8192` and `inferred_entries = 512`
- `page_walk_penalty.penalty_ns ~= 81.93`

Interpretation note for Apple Silicon:

- The reported `l2_tlb_detection` value is an inferred boundary under this methodology and may reflect combined translation pressure and SLC/memory-hierarchy effects.

## 11. Interpretation Guidance and Limits

- This is a user-space benchmark; it does not directly read hardware PMU counters.
- Latency includes full dependent-load path effects (cache hierarchy, translation pressure, runtime noise).
- Guarding with `max(2*L1D, 64*page)` reduces but does not mathematically eliminate all non-TLB artifacts.
- On Apple Silicon specifically, SLC behavior can make strict architectural L2 TLB isolation difficult; treat `l2_tlb_detection` as an inferred secondary boundary.
- Prefer comparing trends and medians over single outlier points.

For comparable runs:

1. keep command line fixed,
2. keep thermal/load background stable,
3. use repeated runs,
4. verify with exported JSON raw loops and P50 values.

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
- `src/benchmark/tlb_boundary_detector.cpp`
- `src/benchmark/tlb_analysis_json.h`
- `src/core/config/argument_parser.cpp`
- `tests/test_analysis.cpp`

Related user-facing docs:

- `README.md`
- `MANUAL.md`

Related whitepaper for the other standalone analysis mode:

- [CORE_TO_CORE_WHITEPAPER.md](CORE_TO_CORE_WHITEPAPER.md) — Core-to-Core Cache-Line Handoff Latency Benchmark (`-analyze-core2core`)

## 3. CLI Contract

### 3.1 Accepted Forms

`-analyze-tlb` runs a dedicated analysis path and accepts only optional JSON output and optional latency stride override:

```bash
./memory_benchmark -analyze-tlb
./memory_benchmark -analyze-tlb -output tlb_analysis.json
./memory_benchmark -output tlb_analysis.json -analyze-tlb
./memory_benchmark -analyze-tlb -latency-stride-bytes 128 -output tlb_analysis_stride128.json
```

### 3.2 Stride Default

If `-latency-stride-bytes` is not provided, the default stride is **64 bytes**, which matches the standard latency mode default (`Constants::LATENCY_STRIDE_BYTES`).

### 3.3 Rejected Combinations

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

After allocation, the code validates that `pointer_count = buffer_size / stride >= 2`. If stride is very large relative to the selected buffer, mode exits with an error before any measurement begins.

### 4.2 Fixed Measurement Parameters

The mode uses fixed constants:

- `loops_per_point = 30`
- `accesses_per_loop = 25,000,000`

Stride behavior:

- Effective stride is taken from `-latency-stride-bytes` (default: **64 bytes**).

**Chain mode:** The latency chain mode is resolved via `resolve_latency_chain_mode(config.latency_chain_mode, page_walk_baseline_locality_bytes)`. This determines the method used to build the pointer-chase chain for each locality measurement. The resolved mode is reported in the console output and stored in the JSON metadata.

**Memory prefaulting:** After buffer allocation, the code calls `madvise(ptr, size_bytes, MADV_WILLNEED)` to prefault pages and reduce page-fault noise during early measurement.

**Measurement loop:** Each loop rebuilds the latency chain for the target locality, performs warmup, runs pointer chase, and stores latency as `ns/access`.

Per-point central value is `P50` (median) over 30 loops.

### 4.3 Locality Sweep

Main sweep windows are selected from the canonical set:

- `16KB`, `64KB`, `128KB`, `256KB`, `512KB`
- `1MB`, `2MB`, `4MB`, `8MB`, `12MB`, `16MB`, `32MB`, `64MB`, `128MB`, `256MB`

Effective sweep start is stride-aware:

- `min_sweep_locality = max(16KB, 2 * stride)`
- points below `min_sweep_locality` are skipped
- if `min_sweep_locality` is not in the canonical set, it is inserted as the first sweep point

The final sweep vector is deduplicated (via `std::sort` followed by `std::unique`). This prevents duplicates if `min_sweep_locality` exactly matches one of the canonical points.

Separate page-walk comparison point:

- `512MB` (run only when selected buffer is at least `512MB`)

## 5. Boundary Detection Algorithm

Boundary detection is based on a running segment baseline, using a first-match strategy.

For candidate index `i`:

- `baseline_ns = average(p50[segment_start, i))` — average of P50 latencies from `segment_start` (inclusive) to `i` (exclusive)
- `step_ns = p50[i] - baseline_ns`
- `threshold_ns = max(2.0ns, baseline_ns * 0.10)`

Candidate passes threshold when:

- `step_ns >= threshold_ns`

The algorithm scans from `segment_start_index + 1` and stops at the **first** candidate that satisfies both the threshold and guard conditions, returning it as the detected boundary. If no candidate passes, detection returns `detected = false`.

### 5.1 TLB Guard (Cache-Transition Filter)

To avoid classifying early cache transitions as TLB boundaries, candidate locality must also satisfy:

`locality_bytes[i] >= tlb_guard_bytes`

where:

`tlb_guard_bytes = max(2 * L1D_size_bytes, 64 * page_size_bytes)`

The guard acts as a hard lower-bound filter on the candidate index `i`; it does not shift the segment start or baseline calculation.

### 5.2 L1 and L2 Boundary Selection

- `L1 TLB boundary`: first candidate passing threshold + guard, scanning from sweep start.
- `L2 TLB boundary`: first candidate passing threshold + guard, scanning from the **L1 boundary index** (not segment start).

L2 detection only runs when L1 is detected and its boundary index is not at the last sweep point. If L1 is not detected, or if L1 is detected at the final sweep point, L2 detection is skipped.

Both L1 and L2 detections share data points from the L1 boundary index onward, and recompute their baselines independently.

Technical note (Apple Silicon):

- On Apple Silicon, the shared System Level Cache (SLC) can blur translation-only inflection points.
- In practice, the detected `L2 TLB boundary` should be interpreted as an inferred secondary translation-reach boundary, not a guaranteed pure architectural L2 TLB edge.

## 6. Confidence Model

Boundary confidence is classified by step strength and persistence:

- `strong_step = (step_ns >= 4.0) || (step_percent >= 0.15)`
- `persistent_jump = next_point_step >= threshold`, where `next_point_step = p50[i+1] - baseline_ns` (using the **same baseline** as the boundary point, not a new baseline)

The `persistent_jump` check is only evaluated when `i + 1 < p50_latency_ns.size()`. At the last sweep point, `persistent_jump` is forced to `false`, which can result in a Large boundary at the final point being classified as Medium or Low confidence even if the step is large.

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

- `effective baseline locality = P50 latency of the first point in the stride-aware sweep` (i.e., `p50_latency_ns[0]`; this is measured during the main run, not a separate reference measurement)

The penalty is computed as-is; a negative value indicates measurement noise (e.g., due to thermal variance or CPU frequency scaling) rather than a genuine negative penalty.

Availability rule:

- available only when selected analysis buffer is `>= 512MB`
- otherwise reported as unavailable (`N/A`) with reason.

## 8. Console Report Contract

Report always includes:

- CPU, page size, selected buffer, stride, loops/accesses config
- Resolved **latency chain mode**
- `[L1 TLB Detection]`
- `[L2 TLB / Page Walk]`

Boundary detection sections:

When a boundary is detected, the report shows:

- boundary locality,
- inferred entries,
- confidence string with step (`ns` and `%`).

When a boundary is **not detected**, the section reports "Not detected."

Page-walk section:

- When available: shows page-walk penalty in `ns`, with explicit dynamic endpoints (`baseline -> 512MB`)
- When unavailable: shows "N/A" with the reason (e.g., buffer < 512 MB)

## 9. JSON Output Contract (`-output`)

When `-output <file>` is provided with `-analyze-tlb`, output includes:

- top-level metadata:
  - `configuration`
  - `execution_time_sec`
  - `tlb_analysis`
  - `timestamp`
  - `version`

- `configuration` contains mode and run constants:
  - CPU info, page size, L1D size, TLB guard bytes, stride
  - `latency_sample_count` (fixed at 30 loops per point)
  - `accesses_per_sample` (fixed at 25,000,000)
  - **`latency_chain_mode`** (resolved chain mode used)
  - **`performance_cores`** and **`efficiency_cores`** (CPU core configuration)
  - selected buffer size and whether mlock succeeded

- `tlb_analysis` contains:
  - `sweep[]` with raw `loop_latencies_ns` and per-point `p50_latency_ns`
  - `l1_tlb_detection` (with `detected`, `boundary_locality_kb`, `inferred_entries`, `confidence`, and step metadata)
  - `l2_tlb_detection` (same structure as L1)
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

### Algorithm Walkthrough (M4 L1 Detection)

The sweep includes points: `[16KB, 64KB, 128KB, 256KB, 512KB, 1MB, 2MB, 4MB, 8MB, ...]`.

The L1 scan begins at `16KB` (min_sweep_locality with stride 64 bytes). It computes:

1. At `16KB`: `baseline = P50(16KB) = X ns`, `step = 0`, fails threshold.
2. At `64KB`: `baseline = avg(P50[16KB])`, `step = P50(64KB) - baseline`, still low.
3. ... (continuing through `1MB`, `2MB`)
4. At `4MB`: `baseline = avg(P50[16KB..2MB])`, `step = P50(4MB) - baseline ≥ threshold`, `locality(4MB) = 4096KB ≥ guard`, **first match** → detected at `4MB`.

The `inferred_entries = 4096KB / 16KB = 256`, which is typical for a first-level TLB on Apple M4.

### Limitations and Edge Cases

- If the entire sweep remained below threshold (e.g., very small buffer or very large stride), L1 would report "Not detected."
- If the 256 MB buffer fallback is used, `page_walk_penalty` will report `N/A` and explain why.
- On machines with 4 KB pages (instead of macOS's 16 KB), the same entry count would correspond to a 1 MB boundary.

Interpretation note for Apple Silicon:

- The reported `l2_tlb_detection` value is an inferred boundary under this methodology and may reflect combined translation pressure and SLC/memory-hierarchy effects.

## 11. Interpretation Guidance and Limits

### Measurement Scope

- This is a user-space benchmark; it does not directly read hardware PMU counters.
- Latency includes full dependent-load path effects (cache hierarchy, translation pressure, runtime noise).
- Guarding with `max(2*L1D, 64*page)` reduces but does not mathematically eliminate all non-TLB artifacts.
- On Apple Silicon specifically, SLC behavior can make strict architectural L2 TLB isolation difficult; treat `l2_tlb_detection` as an inferred secondary boundary.

### Comparing Results Across Apple Silicon Models

All Apple Silicon Macs use a fixed **16 KB page size**. When comparing `-analyze-tlb` results across different Apple Silicon generations (M1, M2, M3, M4, M5, etc.):

- `inferred_entries` is calculated as `boundary_locality_bytes / 16384`.
- The actual entry counts may differ between generations due to microarchitectural changes.
- Comparing `inferred_entries` directly across models is valid (same page size), but be aware that TLB capacity has evolved across generations.

### When L1 TLB Is Not Detected

If L1 detection reports "Not detected":

1. **Check the selected buffer size:** Is it large enough to sweep through the TLB capacity? The 16 KB sweep start may be insufficient if stride is large.
2. **Review the stride:** If `-latency-stride-bytes` is large (e.g., 512 bytes), `min_sweep_locality = max(16KB, 2 * stride)` may skip the actual L1 TLB boundary. Try re-running with a smaller stride.
3. **Check the raw JSON data:** Export with `-output` and inspect `sweep[].p50_latency_ns` to see if the expected inflection point is present in the raw data.
4. **Verify CPU/system state:** Thermal throttling or power-saving modes can obscure boundaries; run in a stable, idle state.

### Best Practices for Comparable Runs

1. Keep command line fixed (buffer size, stride, output file).
2. Keep thermal/load background stable.
3. Use repeated runs (3–5) to verify consistency.
4. Verify with exported JSON raw loops and P50 values.
5. Inspect `execution_time_sec` and `latency_chain_mode` to ensure measurement conditions were consistent.

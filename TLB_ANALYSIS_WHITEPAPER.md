# TLB Analysis Whitepaper: `--analyze-tlb` Methodology

## 1. Purpose

This document specifies how `macOS-memory-benchmark` implements standalone TLB analysis mode (`--analyze-tlb`) in version `0.57.0`.

The goal is to provide a reproducible, implementation-accurate description of:

- measurement workflow,
- boundary detection logic,
- confidence scoring,
- derived metrics (entry reach and a large-locality latency delta),
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

- [CORE_TO_CORE_WHITEPAPER.md](CORE_TO_CORE_WHITEPAPER.md) — Core-to-Core Cache-Line Handoff Latency Benchmark (`--analyze-core2core`)

## 3. CLI Contract

### 3.1 Accepted Forms

`--analyze-tlb` runs a dedicated analysis path and accepts only optional JSON output, optional latency stride override, optional chain-mode override, optional sweep density, optional reproducibility seed, optional parameter sweep specs, and an optional sweep run-count guardrail:

```bash
memory_benchmark --analyze-tlb
memory_benchmark --analyze-tlb --output tlb_analysis.json
memory_benchmark --output tlb_analysis.json --analyze-tlb
memory_benchmark --analyze-tlb --latency-stride-bytes 128 --output tlb_analysis_stride128.json
memory_benchmark --analyze-tlb --latency-chain-mode random-box --tlb-density medium --output tlb_analysis_medium.json
memory_benchmark --analyze-tlb --seed 123456789 --output tlb_analysis_seeded.json
memory_benchmark --analyze-tlb --sweep tlb-density=low,medium,high --output tlb_density_sweep.json
memory_benchmark --analyze-tlb --sweep latency-stride-bytes=64,128 --sweep tlb-density=medium,high --sweep-max-runs 4 --output tlb_stride_density_sweep.json
```

### 3.2 Stride Default

If `--latency-stride-bytes` is not provided, the default stride is **256 bytes**, which matches the standard latency mode default (`Constants::LATENCY_STRIDE_BYTES`).

For the page-native paired methodology, analyze-TLB stride must:

- be pointer-size aligned,
- be no larger than the system page size.

The builder rounds effective node spacing up to a cache-line multiple, so stride does not need to divide the system page
size. These rules apply to direct options and every stride value in a parameter sweep. The complete Cartesian sweep is
validated before its first benchmark run.

### 3.3 Sweep Density (`--tlb-density`)

Sweep density applies only to `--analyze-tlb`.

- `low`: 15-point base sweep, no refinement pass
- `medium`: 15-point base sweep + refinement pass
- `high` (default): 29-point base sweep + refinement pass

### 3.4 Reproducibility Seed (`--seed`)

`--seed <uint64>` fixes the planner order, derived per-task seeds, and standalone TLB pointer-chain permutations. If the
option is omitted, one 64-bit seed is generated when the command is parsed. A Cartesian parameter sweep reuses that
same base seed for every generated run, which makes parameter comparisons share the same deterministic schedule policy.

The base seed, whether it was user-provided or generated, and every derived measurement seed are stored in JSON.

### 3.5 Parameter Sweep (`--sweep`)

Sweep mode applies a Cartesian product over supported TLB-analysis parameters and writes one combined JSON file.

Allowed `--analyze-tlb` sweep keys:

- `latency-stride-bytes`
- `latency-chain-mode`
- `tlb-density`

Sweep mode requires `--output <file>`. `--sweep-max-runs <count>` limits the number of generated combinations; the default guardrail is `256`.

Each sweep parameter key may appear only once. The combined JSON is atomically checkpointed after each completed run
and records top-level `status`, `planned_runs`, `completed_runs`, and `conclusions_valid` fields.

`--sweep latency-chain-mode=...` follows the same chain-mode rule as direct `--latency-chain-mode`: `global-random` is rejected for `--analyze-tlb`.

### 3.6 Rejected Combinations

All options outside the accepted set are rejected when `--analyze-tlb` is present.
`--latency-chain-mode global-random` is also rejected for `--analyze-tlb`, because it ignores locality windows and would turn the locality sweep into repeated full-buffer random measurements with misleading boundary labels.

Example (invalid):

```bash
memory_benchmark --analyze-tlb --buffer-size 1024
memory_benchmark --analyze-tlb --latency-chain-mode global-random
```

## 4. Measurement Workflow

### 4.1 Buffer Allocation Policy

The analysis tries these buffers in order:

1. `1024 MB`
2. `512 MB`
3. `256 MB`

If all fail, mode exits with an insufficient-memory error.

`mlock()` is attempted for the selected buffer; report shows whether lock succeeded.

The page-native builder validates each requested spread footprint and packed footprint against the selected buffer before
writing any pointer slots.

### 4.2 Fixed Measurement Parameters

The mode uses fixed constants:

- `loops_per_point = 30`
- `accesses_per_loop = 25,000,000`

Stride behavior:

- Requested stride is taken from `--latency-stride-bytes` (default: **256 bytes**).
- Spread node-slot spacing is `cache_line_align_up(max(stride, 64 bytes))`; packed node spacing is one 64-byte cache line. Effective spacing is recorded per chain.

**Page-native chain pair:** Every task uses one logical node per requested page. The spread layout places exactly one node
on every requested page. The packed layout places the same number of nodes on consecutive distinct cache lines, minimizing
the page footprint while preserving node and unique-cache-line counts. Pointer
values are written in sorted physical-slot order only after traversal has been planned; setup writes therefore do not
replay the measured traversal order. An independent validation pass requires every node to stay in bounds, occur once,
use a unique cache line, and return to the chain head after the exact node count. Spread validation additionally requires
`actual_pages == requested_pages`.

**Chain mode mapping:** The latency chain mode is resolved through
`resolve_latency_chain_mode(config.latency_chain_mode, page_walk_baseline_locality_bytes)` and reported in console/JSON.
`auto` resolves to `random-box`; `global-random` remains invalid. In the page-native builder, `random-box` means randomized
page order plus randomized page-internal offsets, `same-random-in-box` means increasing page order with one shared offset,
and `diff-random-in-box` means increasing page order with independently selected offsets. Packed controls preserve the
corresponding randomized or increasing logical traversal policy.

**Memory prefaulting:** After buffer allocation, the code calls `madvise(ptr, size_bytes, MADV_WILLNEED)` to prefault pages and reduce page-fault noise during early measurement.

**Balanced round scheduler:** The pure scheduler creates 30 rounds. Every round contains every planned locality exactly
once. A seed-shuffled initial point order is cyclically rotated on subsequent rounds, so a point traverses different
order positions instead of always being early or late in the run. For a full point-count cycle, each point occupies every
order position once.

**Measurement task:** Each scheduled task derives a stable task seed from the base seed, pass, round, and point index,
then derives separate spread and packed layout seeds. Both chains are built, warmed, and measured inside that one task;
which layout runs first alternates by round/order parity and is recorded. The task stores both raw `ns/access` values and
`translation_delta_ns = spread_latency_ns - packed_latency_ns`. The regular benchmark path still uses its existing
latency-chain implementation and random-device behavior.

Per-point output contains spread, packed, and paired-delta P50 values over 30 rounds. The phase-3 boundary detector still
uses spread latency for compatibility with its positive-baseline thresholds; JSON identifies this with
`boundary_signal = "spread_latency_ns"`. The paired delta is a same-round measured metric, but boundary classification is
not switched to it until the phase-4 changepoint/confidence model is implemented.

### 4.3 Locality Sweep

Main sweep windows depend on `--tlb-density`:

- `low` / `medium` base set (15 points):
  - `16KB`, `64KB`, `128KB`, `256KB`, `512KB`
  - `1MB`, `2MB`, `4MB`, `8MB`, `12MB`, `16MB`, `32MB`, `64MB`, `128MB`, `256MB`

- `high` base set (29 points):
  - `16KB`, `32KB`, `64KB`, `96KB`, `128KB`, `192KB`, `256KB`, `384KB`, `512KB`, `768KB`
  - `1MB`, `1536KB`, `2MB`, `3MB`, `4MB`, `6MB`, `8MB`, `10MB`, `12MB`, `14MB`, `16MB`
  - `24MB`, `32MB`, `48MB`, `64MB`, `96MB`, `128MB`, `192MB`, `256MB`

Refinement policy by density:

- `low`: refinement disabled
- `medium` / `high`: refinement enabled around detected transitions

Effective sweep start is stride-aware:

- `min_sweep_locality = page_align_up(max(16KB, 2 * stride))`
- points below `min_sweep_locality` are skipped
- canonical points and an inserted minimum are aligned to the system page size and deduplicated

The final sweep vector is deduplicated (via `std::sort` followed by `std::unique`). This prevents duplicates if `min_sweep_locality` exactly matches one of the canonical points.

Refinement candidates are rounded down to system-page boundaries and deduplicated again. A narrow bracket can therefore
produce fewer than seven unique refinement points; this is intentional because sub-page refinement would invalidate the
entry-count interpretation under the current methodology.

Refinement points form a separate balanced round pass. Each point records whether it came from the private-cache, L1,
L2, or a merged refinement target and stores the bracket that produced it.

Separate large-locality comparison point:

- `512MB` (run only when selected buffer is at least `512MB`)

## 5. Boundary Detection Algorithm

Boundary detection uses a recency-weighted segment baseline with multi-point persistence, adaptive noise thresholding, and IQR-overlap rejection.

For candidate index `i`:

- `baseline_ns = recency_weighted_average(p50[segment_start, i))` — weighted average where point `j` receives weight `(j - segment_start + 1)`. Recent measurements carry more influence, reducing drag from early points that may have had different thermal or frequency conditions.
- `step_ns = p50[i] - baseline_ns`
- `noise_boost_ns = median(IQR of baseline loop-latency rows)` (when loop rows are available and baseline has at least 3 points)
- `threshold_ns = max(2.0ns, baseline_ns * 0.10, noise_boost_ns)`

Candidate passes threshold when:

- `step_ns >= threshold_ns`

The algorithm scans from `segment_start_index + 1` and returns the **first** candidate that satisfies threshold, guard, and IQR conditions. If no candidate passes, detection returns `detected = false`.

### 5.1 TLB Guard (Cache-Transition Filter)

To avoid classifying early cache transitions as TLB boundaries, candidate locality must also satisfy:

`locality_bytes[i] >= tlb_guard_bytes`

where:

`tlb_guard_bytes = max(2 * L1D_size_bytes, 64 * page_size_bytes)`

The guard acts as a hard lower-bound filter on the candidate index `i`; it does not shift the segment start or baseline calculation.

### 5.2 Adaptive Noise Floor and IQR-Overlap Rejection

When per-point raw loop latencies are available (the 30 individual loop measurements), the detector first estimates a baseline-noise floor from IQR values and raises the candidate threshold when needed:

- `IQR(point_j) = Q3_j - Q1_j`
- `noise_boost_ns = median(IQR(point_j))` for `j in [segment_start, i)`
- effective threshold includes this term: `max(2.0ns, 10% baseline, noise_boost_ns)`

Then it applies an IQR-overlap check to reject candidates whose step still falls inside baseline noise overlap:

- `avg_baseline_q3 = average(Q3 of raw loops for each point in [segment_start, i))`
- `candidate_q1 = Q1 of raw loops at point i`
- If `avg_baseline_q3 >= candidate_q1`, the baseline's upper noise band overlaps the candidate's lower band → candidate is **rejected** even if the P50 step exceeds threshold.

This prevents false positives from "lucky medians" where a noisy point happens to have a high P50 due to sampling variance.

### 5.3 Multi-Point Persistence

Instead of checking a single future point, the detector checks up to **3 future points**:

- `persistent_count = count of j in [i+1, min(i+4, size)) where p50[j] - baseline >= threshold`
- `persistent_jump = persistent_count >= 2` (majority of up to 3)

This makes detection robust against single-point noise dips after a genuine boundary. A boundary at index `i` followed by one noisy dip and two confirming points will still be classified as persistent.

### 5.4 Last-Point Strong-Step Compensation

When the candidate is at or near the last sweep point, there are few or no future points for persistence evaluation. In this case, if the step itself is very large:

- `strong_last_point = (step_ns >= 8.0) || (step_percent >= 0.25)`

Then `effective_persistent = persistent_jump || strong_last_point`. This prevents downgrading a massive final-point step to Low confidence purely due to lack of future data.

### 5.5 L1 and L2 Boundary Selection

- `L1 TLB boundary`: first candidate passing threshold + guard + IQR check, scanning from sweep start.
- `L2 TLB boundary`: first candidate passing threshold + guard + IQR check, scanning from an **offset segment start** past the L1 boundary.

Private-cache overlap handling:

- The analyzer first checks the direct L1 candidate from sweep start.
- If that candidate is the same locality as a detected private-cache knee, it is preserved as an ambiguous L1 TLB candidate and marked with `overlaps_private_cache_knee = true`.
- If the direct candidate does not overlap the private-cache knee, the analyzer also tries the cache-knee-offset scan used to avoid obvious cache-knee contamination.

L2 detection specifics:

- **Segment start offset**: L2 scanning starts at `min(L1_boundary_index + 2, size - 2)`, excluding the L1 boundary point and its immediate neighbour from the L2 baseline. This prevents L1 transition noise from contaminating the L2 baseline.
- **L2-specific guard**: `max(tlb_guard_bytes, L1_boundary_locality_bytes)`, preventing L2 from re-detecting at or below the L1 boundary locality.

L2 detection only runs when L1 is detected and its boundary index is not at the last two sweep points.

Technical note (Apple Silicon):

- On Apple Silicon, the shared System Level Cache (SLC) can blur translation-only inflection points.
- In practice, the detected `L2 TLB boundary` should be interpreted as an inferred secondary translation-reach boundary, not a guaranteed pure architectural L2 TLB edge.

## 6. Confidence Model

Boundary confidence is classified by step strength and multi-point persistence:

- `strong_step = (step_ns >= 4.0) || (step_percent >= 0.15)`
- `persistent_jump`: majority of up to 3 future points exceed threshold (see section 5.3). At the last sweep point, a strong step (`step_ns >= 8.0` or `step_percent >= 0.25`) compensates for the lack of future data.

Confidence levels:

- **High**: `strong_step && persistent_jump`
- **Medium**: `strong_step || persistent_jump`
- **Low**: otherwise (still passed threshold)

## 7. Derived Metrics

### 7.1 Inferred Entry Reach

For detected boundaries:

`inferred_entries_min = previous_locality_bytes / page_size_bytes`

`inferred_entries_max = boundary_locality_bytes / page_size_bytes`

`inferred_entries = midpoint(inferred_entries_min, inferred_entries_max)`

Point estimate and range are reported separately for L1 and L2 sections. The point estimate is intentionally the midpoint of the detected locality window, not the upper boundary edge, so the headline value does not imply more precision than the sweep grid supports.

### 7.2 Large-Locality Latency Delta

The compatibility methodology computes this delta independently from L1/L2 boundary step values:

`large_locality_latency_delta_ns = P50(512MB) - P50(effective baseline locality)`

where:

- `effective baseline locality = P50 latency of the first point in the stride-aware sweep` (i.e., `p50_latency_ns[0]`; this is measured during the main run, not a separate reference measurement)

The delta is computed as-is; a negative value indicates measurement noise (for example thermal variance or CPU
frequency scaling). This comparison changes locality and address-order behavior and therefore **does not isolate a
page-table-walk cost**. Console output uses `Large-Locality Latency Delta`, and JSON uses
`large_locality_latency_delta.delta_ns`.

The old `page_walk_penalty` object remains for one compatibility window with `deprecated: true` and
`replacement: "large_locality_latency_delta"`. If the comparison cannot run or analysis is interrupted, the delta is
unavailable and its numerical value is omitted.

Availability rule:

- available only when selected analysis buffer is `>= 512MB`
- otherwise reported as unavailable (`N/A`) with reason.

## 8. Console Report Contract

Report always includes:

- CPU, page size, selected buffer, stride, loops/accesses config
- Resolved **latency chain mode**
- Fine--sweep refinement summary (added points, total points)
- `[Private Cache Knee Detection]`
- `[L1 TLB Detection]`
- `[L2 TLB / Large-Locality Delta]`

Boundary detection sections:

When a boundary is detected, the report shows:

- boundary locality,
- inferred entries,
- confidence string with step (`ns` and `%`).

When a boundary is **not detected**, the section reports "Not detected."

Page-walk section:

- When available: shows the large-locality latency delta in `ns`, with explicit dynamic endpoints (`baseline -> 512MB`)
- When unavailable: shows "N/A" with the reason (e.g., buffer < 512 MB)
- Always shows analysis status, measured/planned point counts, and whether conclusions are valid.
- Interrupted or partial analyses suppress private-cache and L1/L2 conclusions.

A user interrupt uses the existing graceful-shutdown contract and may return process success after writing partial JSON.
Machine consumers must require `status == "complete"` and `conclusions_valid == true`; exit status alone is not a
completeness signal.

## 9. JSON Output Contract (`--output`)

### 9.1 Single-Run TLB Analysis JSON

When `--output <file>` is provided with `--analyze-tlb` without `--sweep`, output includes:

- top-level metadata:
  - `configuration`
  - `execution_time_sec`
  - `tlb_analysis`
  - `timestamp`
  - `version`

- `configuration` contains mode and run constants:
  - CPU info, page size, L1D size, TLB guard bytes, stride
  - `latency_sample_count` (fixed at 30 loops per point)
  - `accesses_per_loop` (fixed at 25,000,000)
  - **`latency_chain_mode`** (resolved chain mode used)
  - **`performance_cores`** and **`efficiency_cores`** (CPU core configuration)
  - selected buffer size and whether mlock succeeded
  - `schema_version = 3` and `methodology_version = "page-native-paired-v3"`
  - base `seed`, `seed_source`, `schedule_policy = "seeded-cyclic-latin"`, chain model, delta definition, and boundary signal

- `tlb_analysis` contains:
  - status, point/round counts, scheduled-pair counts, raw-measurement counts, and `conclusions_valid`
  - `measurement_records[]` in execution order with pass, point, locality, round, order, task seed, legacy spread `latency_ns`, and a `paired_control` object
  - each `paired_control` contains pair order, spread/packed seeds and raw latencies, verified chain diagnostics, and same-round `translation_delta_ns`
  - `sweep[]` contains requested/effective/actual pages, actual node and unique-cache-line counts, both chain diagnostics, raw spread/packed/delta arrays, their P50 values, refinement source/bracket, and per-task records
  - `private_cache_knee` (with `detected`, `boundary_locality_kb`, `confidence`, and `may_interfere_with_tlb`)
  - `l1_tlb_detection` (with `detected`, `boundary_locality_kb`, `inferred_entries`, `inferred_entries_method`, `inferred_entries_min`, `inferred_entries_max`, `overlaps_private_cache_knee`, `confidence`, and step metadata)
  - `l2_tlb_detection` (same structure as L1)
  - `large_locality_latency_delta` block (`available`, baseline/comparison metadata, raw comparison loops, `delta_ns` when the comparison completed, otherwise `reason`)
  - deprecated `page_walk_penalty` compatibility alias with an explicit replacement field

This payload is designed for full post-run verification and reproducibility checks.

### 9.2 TLB Sweep JSON

When `--sweep` is used with `--analyze-tlb`, output uses the common sweep envelope:

- top-level `configuration.mode = "sweep"`
- `configuration.base_mode = "analyze_tlb"`
- `configuration.run_count`
- `configuration.sweep_max_runs`
- `configuration.sweep_parameters`
- `runs[]`, where each run contains:
  - `index`
  - `parameters`
  - `result`

Each `runs[].result` entry is the same single-run TLB analysis JSON payload described in section 9.1.
The top-level sweep object also includes `status`, `planned_runs`, `completed_runs`, and `conclusions_valid`. It is
atomically rewritten after every completed run, so a later failure or interrupt leaves a readable checkpoint.

## 10. Historical Worked Example (Apple M4)

Example file:

- `results/0.53.7/MacMiniM4_analyzetlb.json`

This file predates schema version 3, page-native pairing, page-aligned refinement, completion metadata, and midpoint entry ranges. It remains
useful only as historical latency input; it is not a byte-for-byte example of current output. Observed legacy fields:

- `tlb_guard_bytes = 1048576` (`1MB`)
- `l1_tlb_detection.boundary_locality_kb = 4096` and `inferred_entries` near the midpoint of its detected entry range
- `l2_tlb_detection.boundary_locality_kb = 8192` and `inferred_entries` near the midpoint of its detected entry range
- legacy `page_walk_penalty.penalty_ns ~= 81.93` (now reported as a large-locality latency delta)

### Algorithm Walkthrough (M4 L1 Detection)

The sweep includes points: `[16KB, 32KB, 64KB, 96KB, 128KB, 192KB, 256KB, 384KB, 512KB, 768KB, 1MB, 1536KB, 2MB, 3MB, 4MB, 6MB, 8MB, ...]`.

The L1 scan begins at `16KB` (min_sweep_locality with stride 64 bytes). It computes:

1. At `16KB`: `baseline = P50(16KB) = X ns`, `step = 0`, fails threshold.
2. At `64KB`: `baseline = avg(P50[16KB])`, `step = P50(64KB) - baseline`, still low.
3. ... (continuing through `1MB`, `2MB`)
4. At `4MB`: `baseline = avg(P50[16KB..2MB])`, `step = P50(4MB) - baseline ≥ threshold`, `locality(4MB) = 4096KB ≥ guard`, **first match** → detected at `4MB`.

Under the current high-density grid, a `4MB` boundary following a `3MB` prior point would yield
`inferred_entries_min = 192`, `inferred_entries_max = 256`, and `inferred_entries = 224` as a midpoint estimate. This
calculation illustrates the current algorithm; those fields are not present in the historical 0.53.7 sample. The value
must be treated as a methodology-dependent candidate rather than an independently verified architectural entry count.

### Limitations and Edge Cases

- If the entire sweep remained below threshold (e.g., very small buffer or very large stride), L1 would report "Not detected."
- If the 256 MB buffer fallback is used, or if the 512MB comparison pass is interrupted, `large_locality_latency_delta` will report `N/A` and explain why.
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

All Apple Silicon Macs use a fixed **16 KB page size**. When comparing `--analyze-tlb` results across different Apple Silicon generations (M1, M2, M3, M4, M5, etc.):

- `inferred_entries` is calculated as the midpoint of the detected entry range; use `inferred_entries_min` and `inferred_entries_max` when comparing uncertainty windows.
- The actual entry counts may differ between generations due to microarchitectural changes.
- Comparing `inferred_entries` directly across models is valid (same page size), but be aware that TLB capacity has evolved across generations.

### When L1 TLB Is Not Detected

If L1 detection reports "Not detected":

1. **Check the selected buffer size:** Is it large enough to sweep through the TLB capacity? The 16 KB sweep start may be insufficient if stride is large.
2. **Review the stride:** If `--latency-stride-bytes` is large (e.g., 512 bytes), `min_sweep_locality = max(16KB, 2 * stride)` may skip the actual L1 TLB boundary. Try re-running with a smaller stride.
3. **Check the raw JSON data:** Export with `--output` and inspect `sweep[].p50_latency_ns` to see if the expected inflection point is present in the raw data.
4. **Verify CPU/system state:** Thermal throttling or power-saving modes can obscure boundaries; run in a stable, idle state.

### Best Practices for Comparable Runs

1. Keep command line fixed (buffer size, stride, output file).
2. Keep thermal/load background stable.
3. Use repeated runs (3–5) to verify consistency.
4. Verify with exported JSON raw loops and P50 values.
5. Inspect `execution_time_sec` and `latency_chain_mode` to ensure measurement conditions were consistent.

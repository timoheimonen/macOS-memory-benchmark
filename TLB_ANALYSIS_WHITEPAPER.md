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

- `low` (`quick`): 15-point base sweep, no refinement pass, 7-12 adaptive rounds
- `medium` (`standard`, default): 15-point base sweep + refinement pass, 10-20 adaptive rounds
- `high` (`exhaustive`): 29-point base sweep + refinement pass, 15-30 adaptive rounds

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

Sweep mode requires `--output <file>`. `--sweep-max-runs <count>` limits the number of generated combinations. Its default is `16` for `--analyze-tlb` and `256` for other modes; an explicit value overrides the mode default.

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

The analysis considers these buffers in order:

1. `1024 MB`
2. `512 MB`
3. `256 MB`

Before `mmap()`, each candidate receives a conservative peak estimate:

```text
peak = candidate buffer + 1 MB + 256 bytes * (candidate bytes / page size)
```

The memory budget is the smaller of 30% of currently available memory and the amount that preserves a 1 GB reserve.
On systems reporting at most 1 GB available, half is retained; if available-memory detection fails, the fallback budget is
384 MB. The first candidate whose predicted peak fits this budget is attempted. If no budget-safe candidate can be allocated,
the mode exits with an insufficient-memory error. An `mmap()` success alone therefore cannot select an unsafe candidate.

`mlock()` is best-effort. On failure, the mode reports errno and its message, records the failure and policy in JSON, and
continues unlocked. The settings block also reports available memory, the selected budget, and predicted peak use.

The page-native builder validates each requested spread footprint and packed footprint against the selected buffer before
writing any pointer slots.

### 4.2 Calibrated Measurement Parameters and Adaptive Rounds

The three runtime profiles are:

| Profile | Density | Rounds | Target per chain | Minimum chain cycles | Profile access cap | CI-width target |
|---|---|---:|---:|---:|---:|---:|
| quick | low | 7-12 | 5 ms | 8 | 1,000,000 | 0.50 ns |
| standard | medium | 10-20 | 10 ms | 16 | 2,000,000 | 0.30 ns |
| exhaustive | high | 15-30 | 20 ms | 32 | 5,000,000 | 0.15 ns |

Each chain first runs a pilot covering at least two whole cycles and 4,096 accesses. Pilot time determines the main access
count for the profile target duration. The result is rounded to a whole-chain multiple and cannot fall below the profile's
minimum cycle count. For a very large chain, that minimum takes precedence over the nominal profile access cap.

After the profile minimum round count, every completed round evaluates the deterministic bootstrap 95% median-CI width of
the translation delta at every point. The pass stops only when every width meets the profile target; otherwise it continues
to the maximum. Convergence bootstrap storage and chain-construction/validation scratch containers are reused between serial
measurements to avoid repeated inner-loop allocation.

Before every pass, console output reports point count, round range, conservative maximum pointer accesses, predicted peak
memory, and a rough duration based on target measurement time. JSON stores the base estimate, runtime policy, per-chain pilot
and calibrated access counts, and realized per-pass round/convergence summaries.

Stride behavior:

- Requested stride is taken from `--latency-stride-bytes` (default: **256 bytes**).
- Spread node-slot spacing is `cache_line_align_up(max(stride, 64 bytes))`; packed node spacing is one 64-byte cache line. Effective spacing is recorded per chain.

**Page-native chain pair:** Every task uses one logical node per requested page. The spread layout places exactly one node
on every requested page. The packed layout places the same number of nodes on consecutive distinct cache lines, minimizing
the page footprint while preserving node and unique-cache-line counts. Pointer
values are written in sorted physical-slot order only after traversal has been planned; setup writes therefore do not
replay the measured traversal order. An independent chain-integrity traversal requires every node to stay in bounds, occur once,
use a unique cache line, and return to the chain head after the exact node count. Spread validation additionally requires
`actual_pages == requested_pages`.

**Chain mode mapping:** The latency chain mode is resolved through
`resolve_latency_chain_mode(config.latency_chain_mode, page_walk_baseline_locality_bytes)` and reported in console/JSON.
`auto` resolves to `random-box`; `global-random` remains invalid. In the page-native builder, `random-box` means randomized
page order plus randomized page-internal offsets, `same-random-in-box` means increasing page order with one shared offset,
and `diff-random-in-box` means increasing page order with independently selected offsets. Packed controls preserve the
corresponding randomized or increasing logical traversal policy.

**Memory prefaulting:** After buffer allocation, the code calls `madvise(ptr, size_bytes, MADV_WILLNEED)` to prefault pages and reduce page-fault noise during early measurement.

**Balanced round scheduler:** The pure scheduler creates the active profile's maximum schedule and stops only after a whole
round. Every round contains every planned locality exactly once. A seed-shuffled initial point order is cyclically rotated on subsequent rounds, so a point traverses different
order positions instead of always being early or late in the run. For a full point-count cycle, each point occupies every
order position once.

**Measurement task:** Each scheduled task derives a stable task seed from the base seed, pass, round, and point index,
then derives separate spread and packed layout seeds. Both chains are built, warmed, and measured inside that one task;
which layout runs first alternates by round/order parity and is recorded. The task stores both raw `ns/access` values and
`translation_delta_ns = spread_latency_ns - packed_latency_ns`. The regular benchmark path still uses its existing
latency-chain implementation and random-device behavior.

Per-point output contains spread, packed, and paired-delta P50 values over the completed adaptive rounds. Boundary inference uses a
round-by-point matrix of `translation_delta_ns`, and JSON identifies this with
`boundary_signal = "translation_delta_ns"`. Spread latency remains available for compatibility plots and the separate
private-cache/refinement heuristics; it is not the accepted L1/L2 boundary signal.

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
L2, or a merged refinement target and stores the bracket that produced it. After refinement, discovery candidates that
pass the robust statistical gates cause their bracket baseline, candidate, and two following points to be measured again
in an independent validation pass.

Separate large-locality comparison point:

- `512MB` (run only when selected buffer is at least `512MB`)

## 5. Boundary Detection Algorithm

Accepted L1/L2 detection is based on same-round spread/packed differences, not pointwise spread P50 values. Let
`D[r,i]` be `translation_delta_ns` at round `r` and sorted locality point `i`. For a candidate index `i`, the detector
forms paired point-to-point effects:

`E[r,i] = D[r,i] - D[r,i-1]`

Only rounds containing both cells participate. At least seven paired samples are required. The candidate effect is
`median(E[:,i])`, and its uncertainty is a deterministic 2,000-resample percentile-bootstrap 95% interval derived from
the command seed and candidate index.

The predefined minimum effect is `0.5ns`. The adaptive noise floor is:

`max(0.1ns, 1.5 * median(abs(prior adjacent paired effects)))`

A discovery candidate must satisfy all of the following:

- median paired effect is at least `0.5ns`;
- the bootstrap 95% CI lower bound is above the adaptive noise floor;
- both of the next two locality points remain above the same pre-boundary point by at least `0.5ns`;
- both persistence effects also have bootstrap 95% CI lower bounds above the noise floor.

Material candidates rejected by CI, persistence, or validation are retained in JSON with their rejection reason.

### 5.1 TLB Guard (Cache-Transition Filter)

To avoid classifying early cache transitions as TLB boundaries, candidate locality must also satisfy:

`locality_bytes[i] >= tlb_guard_bytes`

where:

`tlb_guard_bytes = max(2 * L1D_size_bytes, 64 * page_size_bytes)`

The guard acts as a hard lower-bound filter on the candidate index `i`; it does not shift the segment start or baseline calculation.

### 5.2 Independent Validation Pass

Discovery evidence never creates an accepted boundary by itself. All points needed to check a discovery candidate are
deduplicated into a separate `validation` scheduler pass. This pass uses pass-specific task and chain seeds and repeats
the bracket baseline, candidate, and two persistence points. The same minimum-effect, noise-floor, paired-sample,
bootstrap-CI, and two-point persistence gates are applied to the validation matrix. `detected = true` requires both
`discovery.passed` and `validation.passed`.

### 5.3 Strict Two-Point Persistence

Both following locality points are mandatory. A final-point candidate or a candidate with only one following point is
rejected as `insufficient-persistence-points`; there is no strong-last-point exception. A single high point followed by a
return to baseline is rejected as `persistence-not-confirmed`.

### 5.4 L1 and L2 Boundary Selection

- `L1 TLB boundary`: first candidate above the TLB guard that passes discovery and validation.
- `L2 TLB boundary`: first independently validated candidate from the offset segment after L1.

Private-cache knee detection remains a separate spread-latency diagnostic. The packed control and translation-delta
signal are the primary protection against turning a data-cache step into an accepted translation boundary.

L2 detection specifics:

- **Segment start offset**: candidate scanning resumes after the L1 boundary and its immediate neighbour, preventing the L1 transition from becoming the next paired baseline.
- **L2-specific guard**: `max(tlb_guard_bytes, L1_boundary_locality_bytes)`, preventing L2 from re-detecting at or below the L1 boundary locality.

L2 detection only runs when L1 is detected and its boundary index is not at the last two sweep points.

Technical note (Apple Silicon):

- On Apple Silicon, the shared System Level Cache (SLC) can blur translation-only inflection points.
- In practice, the detected `L2 TLB boundary` should be interpreted as an inferred secondary translation-reach boundary, not a guaranteed pure architectural L2 TLB edge.

## 6. Confidence Model

Only candidates with complete discovery and validation evidence receive an accepted confidence label:

- **High**: both discovery and validation 95% CI lower bounds are at least `max(1.0ns, 2 * discovery_noise_floor)`.
- **Medium**: all acceptance gates pass, but the High-strength condition does not.

Rejected candidates receive no accepted confidence label. Their discovery and validation objects still expose effect,
95% CI, noise floor, paired sample count, persistence count, and rejection reason.

## 7. Derived Metrics

### 7.1 Inferred Entry Reach

For detected boundaries:

`inferred_entries_min = previous_locality_bytes / page_size_bytes`

`inferred_entries_max = boundary_locality_bytes / page_size_bytes`

`inferred_entries = midpoint(inferred_entries_min, inferred_entries_max)`

The range from the final refined and validated bracket is the primary result for L1 and L2. The midpoint remains an
explicitly secondary estimate and must not be interpreted as exact architectural capacity.

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
  - `runtime_profile`, adaptive minimum/maximum rounds, CI-width target, and convergence bootstrap count
  - access-calibration target, minimum whole-chain cycles, and profile access cap
  - **`latency_chain_mode`** (resolved chain mode used)
  - **`performance_cores`** and **`efficiency_cores`** (CPU core configuration)
  - selected buffer, available-memory budget, estimated peak, and best-effort `mlock()` status/errno/error
  - base-pass point/access/memory/duration work estimate
  - `schema_version = 4` and `methodology_version = "page-native-paired-adaptive-validated-v4"`
  - base `seed`, `seed_source`, `schedule_policy = "seeded-cyclic-latin"`, chain model, delta definition, and `boundary_signal = "translation_delta_ns"`
  - paired-bootstrap method, 2,000 resamples, `0.5ns` minimum effect, two-point persistence, and independent-validation requirement

- `tlb_analysis` contains:
  - status, discovery/validation point counts, adaptive round bounds, per-pass realized round/convergence summaries, scheduled-pair bounds, raw-measurement counts, and `conclusions_valid`
  - `measurement_records[]` in execution order with pass, point, locality, round, order, task seed, legacy spread `latency_ns`, and a `paired_control` object
  - each `paired_control` contains pair order, spread/packed seeds, pilot timing/accesses, calibrated accesses, raw latencies, verified chain diagnostics, and same-round `translation_delta_ns`
  - `sweep[]` contains requested/effective/actual pages, actual node and unique-cache-line counts, both chain diagnostics, raw spread/packed/delta arrays, their P50 values, refinement source/bracket, and per-task records
  - `private_cache_knee` (with `detected`, `boundary_locality_kb`, `confidence`, and `may_interfere_with_tlb`)
  - `l1_tlb_detection` (with `detected`, validated bracket, primary entry range, midpoint estimate, confidence, discovery/validation evidence, and accepted/rejected candidates)
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

This file predates schema version 4, page-native pairing, page-aligned refinement, independent validation, completion metadata, and bracket-first entry ranges. It remains
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

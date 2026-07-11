# TLB Analysis Whitepaper: `--analyze-tlb` Methodology

## 1. Purpose

This document specifies how `macOS-memory-benchmark` implements standalone TLB analysis mode (`--analyze-tlb`) in version `0.60.0`.

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

The base seed, whether it was user-provided or generated, and every derived measurement seed are stored in JSON. Task seeds
apply SplitMix64 successively to the base seed, pass, round index, and point index. Spread and packed layout seeds apply
SplitMix64 again with a layout-specific domain constant. The `seed_derivation` object records both rules.

### 3.5 Parameter Sweep (`--sweep`)

Sweep mode applies a Cartesian product over supported TLB-analysis parameters and writes one combined JSON file.

Allowed `--analyze-tlb` sweep keys:

- `latency-stride-bytes`
- `latency-chain-mode`
- `tlb-density`

Sweep mode requires `--output <file>`. `--sweep-max-runs <count>` limits the number of generated combinations. Its default is `16` for `--analyze-tlb` and `256` for other modes; an explicit value overrides the mode default.

Each sweep parameter key may appear only once. The combined JSON is atomically checkpointed after each attempted run and
records top-level `status`, `status_reason`, `planned_runs`, `attempted_runs`, `completed_runs`, and
`conclusions_valid` fields. Every attempted run is retained with its own `status` and `status_reason`, so
`attempted_runs` equals the length of `runs`. A TLB attempt increments `completed_runs` only when its nested
`tlb_analysis.status` is `complete` and its nested `tlb_analysis.conclusions_valid` is `true`; partial, interrupted, and
failed attempts remain auditable but do not increment the completed count. The first incomplete attempt stops further
runs.

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
384 MB. Budget-safe candidates are attempted in descending order; an allocation failure falls through to the next smaller
budget-safe candidate. If none can be allocated, the mode exits with an insufficient-memory error. An `mmap()` success
alone therefore cannot select an unsafe candidate.

`mlock()` is best-effort. On failure, the mode reports errno and its message, records the failure and policy in JSON, and
continues unlocked. The settings block also reports available memory, the selected budget, and predicted peak use.

The main thread also requests `user-interactive` QoS before measurement. This is a best-effort scheduler hint: console and
JSON record whether it was requested/applied and the API return code, and failure emits a warning without aborting the run.

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

Virtual locality is a translation working-set label, not the amount of active pointer data. Each logical node occupies one
64-byte cache line. With 16 KiB pages, the 512 MiB comparison therefore has `512 MiB / 16 KiB = 32,768` nodes and a
`32,768 * 64 B = 2 MiB` active cache-line footprint. Runtime profiles traverse every chain for multiple complete cycles, so
spread and packed timings are intentionally cache-hot and must not be interpreted as direct DRAM latency. Packed preserves
the same node and cache-line counts while reducing the translation footprint.

**Chain mode mapping:** The latency chain mode is resolved through
`resolve_latency_chain_mode(config.latency_chain_mode, page_walk_baseline_locality_bytes)` and reported in console/JSON.
`auto` resolves to `random-box`; `global-random` remains invalid. In the page-native builder, `random-box` means randomized
page order plus randomized page-internal offsets, `same-random-in-box` means increasing page order with one shared offset,
and `diff-random-in-box` means increasing page order with independently selected offsets. Packed controls preserve the
corresponding randomized or increasing logical traversal policy. Results are comparable only when the effective chain mode
matches. Increasing-page modes intentionally change address order and prefetch exposure, so their latency, detected brackets,
and runtime must not be treated as interchangeable with `random-box`.

**Memory residency hint:** After buffer allocation, the code calls `madvise(ptr, size_bytes, MADV_WILLNEED)`. This is a
best-effort advisory request intended to reduce early page-fault noise; it does not guarantee that every page is faulted
in immediately.

**Balanced round scheduler:** The pure scheduler creates the active profile's maximum schedule and stops only after a whole
round. Every round contains every planned locality exactly once. A seed-shuffled initial point order is cyclically rotated on subsequent rounds, so a point traverses different
order positions instead of always being early or late in the run. For a full point-count cycle, each point occupies every
order position once.

**Measurement task:** Each scheduled task derives a stable task seed from the base seed, pass, round, and point index,
then derives separate spread and packed layout seeds. Both chains are built, warmed, and measured inside that one task;
which layout runs first alternates by round/order parity and is recorded. The task stores both raw `ns/access` values and
`translation_delta_ns = spread_latency_ns - packed_latency_ns`. The regular benchmark path uses its separate
latency-chain implementation. It parses `--seed`, or generates one command-level seed with a random-device/clock
fallback when the option is omitted, then derives domain-separated target/layout seeds; chain construction does not
draw fresh random-device entropy and does not use the standalone page-native TLB chain builder.

The compact console uses one row per point: paired-delta P50 first, spread and packed P50 controls, active cache-line
footprint, and a `*` marker below 64 nodes. One shared legend explains the marker; spread/packed page counts, unique-line
counts, full chain diagnostics, and raw samples remain in JSON. The 64-node marker matches the existing minimum
`64 * page_size` TLB guard and is diagnostic only; it does not alter the grid or boundary acceptance. Sub-resolution
negative values are displayed as `0.00 ns`, while JSON retains the measured value. Boundary inference uses a
round-by-point matrix of `translation_delta_ns`, and JSON identifies this with
`boundary_signal = "translation_delta_ns"`. Spread latency remains available for compatibility plots and the separate
private-cache/refinement heuristics; it is not the accepted L1/L2 boundary signal.

The `quick` profile is explicitly labeled a screening estimate in the final console report because its coarse, fixed grid
can bracket a different boundary than the refined `standard` or `exhaustive` profiles. Hardware conclusions from `quick`
should be confirmed with `medium` or `high`.

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

Refinement candidates are rounded down to system-page boundaries, deduplicated across targets, and filtered against every
locality already present in the measured base grid. The planned, measured, and added-point counters therefore describe the
same unique locality set. A narrow bracket can produce fewer than seven unique refinement points; this is intentional because
sub-page refinement would invalidate the entry-count interpretation under the current methodology.

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

### 7.2 Large-Locality Paired Comparison

The primary 512 MiB result aggregates the same paired records used by the rest of the methodology:

`large_locality_translation_delta_p50_ns = median(spread_latency_ns[r] - packed_latency_ns[r])`

The median is taken over stored same-round deltas. It is deliberately not computed as
`median(spread_latency_ns) - median(packed_latency_ns)`, because those operations are not generally equivalent. Console and
JSON also expose the independent spread and packed P50 values, page counts, node/cache-line counts, and active cache-line
footprint. The result describes cache-hot paired translation stress; it is neither direct DRAM latency nor an isolated
page-table-walk cost.

Schema 4 publishes this only as `large_locality_paired_comparison`. It contains no raw-spread locality-delta or
page-walk alias, preventing a cache-hot spread difference from being mistaken for the primary paired signal.

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
- `[L2 TLB Detection]`
- `[Large-Locality Paired Comparison]`

Boundary detection sections:

When a boundary is detected, the report shows:

- boundary locality,
- inferred entries,
- confidence string with step (`ns` and `%`).

When a boundary is **not detected**, the section reports "Not detected."

Large-locality paired section:

- When available: shows virtual locality, active footprint, spread/packed page counts, spread P50, packed P50, and same-round translation-delta P50
- When unavailable: shows "N/A" with the reason (e.g., buffer < 512 MB)
- Always states that raw timings are cache-hot and are not direct DRAM or isolated page-table-walk latency
- Always shows analysis status, measured/planned point counts, and whether conclusions are valid.
- Interrupted or partial analyses suppress private-cache and L1/L2 conclusions.

A user interrupt uses the existing graceful-shutdown contract and may return process success after writing partial JSON.
Machine consumers must require `status == "complete"` and `conclusions_valid == true`; exit status alone is not a
completeness signal.

`validation_required` describes whether candidate-specific validation points were planned, not whether the methodology
generally requires independent validation. An interruption during the base pass can leave this field `false` with
`validation_status = "not-run"` and zero planned validation points. `validation_complete` remains `false` in that state.

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
  - exact uint64 decimal-string base `seed`, `seed_source`, explicit task/layout `seed_derivation`, `schedule_policy = "seeded-cyclic-latin"`, chain model, effective-mode comparability guidance, delta definition, and `boundary_signal = "translation_delta_ns"`
  - main-thread QoS request/applied/code metadata and its best-effort policy
  - paired-bootstrap method, 2,000 resamples, `0.5ns` minimum effect, two-point persistence, and independent-validation requirement

- `tlb_analysis` contains:
  - status, discovery/validation point counts, `validation_required`/`validation_status`/`validation_complete`, adaptive round bounds, per-pass realized round/convergence summaries, explicitly scoped base+validation/large-locality/total pair and raw-measurement counts, and `conclusions_valid`
  - `measurement_records[]` in execution order with pass, point, locality, round, order, exact decimal-string task seed, and a `paired_control` object
  - each `paired_control` contains pair order, exact decimal-string spread/packed seeds, pilot timing/accesses, calibrated accesses, raw latencies, verified chain diagnostics, and same-round `translation_delta_ns`
  - `sweep[]` contains requested/effective/actual pages, pointer-node and pointers-per-page counts, unique-cache-line and active-footprint counts, short-cycle diagnostics, both chain diagnostics, raw spread/packed/delta arrays, their P50 values, refinement source/bracket, and per-task records
  - `private_cache_knee` (with `detected`, `boundary_locality_kb`, `confidence`, and `may_interfere_with_tlb`)
  - `l1_tlb_detection` (with `detected`, validated bracket, primary entry range, midpoint estimate, confidence, discovery/validation evidence, and accepted/rejected candidates)
  - `l2_tlb_detection` (same structure as L1)
  - sole `large_locality_paired_comparison` block with same-round delta P50, spread/packed P50 values, physical page/cache-line diagnostics, active footprint, raw paired records, and explicit interpretation

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
  - `status`
  - `status_reason`
  - `result`

Each `runs[].result` entry is the same single-run TLB analysis JSON payload described in section 9.1.
The top-level sweep object also includes `status`, `status_reason`, `planned_runs`, `attempted_runs`, `completed_runs`,
and `conclusions_valid`. It is atomically rewritten after every attempted run, so a later failure or interrupt leaves a
readable checkpoint. `attempted_runs` equals the length of `runs`; `completed_runs` includes only entries whose nested
`tlb_analysis.status` is `complete` and nested `tlb_analysis.conclusions_valid` is `true`. Partial, interrupted, and
failed attempts stay in `runs` without validating the sweep conclusions.

## 10. Current Schema Worked Example (Deterministic Exporter Fixture)

The regression test `JsonSchemaTest.TlbAnalysisExporterIncludesModeAndCoreCounts` generates a current-version payload with
the production serializer. Its deliberately synthetic inputs make this a contract example, not an Apple hardware claim:

```json
{
  "configuration": {
    "mode": "analyze_tlb",
    "schema_version": 4,
    "methodology_version": "page-native-paired-adaptive-validated-v4",
    "seed": "18446744073709551615",
    "seed_encoding": "uint64-decimal-string",
    "main_thread_qos": {"requested": true, "applied": true, "code": 0},
    "seed_source": "user"
  },
  "tlb_analysis": {
    "status": "complete",
    "planned_points": 1,
    "measured_points": 1,
    "validation_required": true,
    "validation_status": "complete",
    "validation_complete": true,
    "conclusions_valid": true,
    "completed_base_validation_pairs": 4,
    "completed_large_locality_pairs": 3,
    "total_completed_measurement_pairs": 7,
    "total_completed_raw_measurements": 14,
    "sweep": [{
      "requested_pages": 1,
      "actual_pages": 1,
      "pointer_nodes": 1,
      "spread_pointers_per_page_max": 1,
      "packed_pointers_per_page_max": 1,
      "active_cache_line_footprint_bytes": 64,
      "short_cycle_diagnostic": true,
      "translation_delta_p50_ns": 5.0
    }],
    "large_locality_paired_comparison": {
      "available": true,
      "spread_p50_ns": 100.0,
      "packed_p50_ns": 94.0,
      "translation_delta_p50_ns": 1.0,
      "spread_actual_pages": 32768,
      "packed_actual_pages": 128,
      "active_cache_line_footprint_bytes": 2097152
    }
  },
  "version": "0.60.0"
}
```

New hardware baselines were explicitly excluded from this change series. Historical 0.53.x result files remain available
only for legacy-schema compatibility checks and are not presented as current-methodology measurements.

### Limitations and Edge Cases

- If the entire sweep remained below threshold (e.g., very small buffer or very large stride), L1 would report "Not detected."
- If the 256 MiB buffer fallback is used, or if the 512 MiB comparison pass is interrupted, `large_locality_paired_comparison` reports `available: false` and explains why.
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
3. **Check the paired JSON data:** Export with `--output` and inspect `sweep[].translation_delta_p50_ns` for the accepted signal. Use `spread_p50_latency_ns`, `packed_p50_latency_ns`, and `active_cache_line_footprint_bytes` only as explicitly named cache-hot diagnostics.
4. **Verify CPU/system state:** Thermal throttling or power-saving modes can obscure boundaries; run in a stable, idle state.

### Best Practices for Comparable Runs

1. Keep command line fixed (buffer size, stride, output file).
2. Keep thermal/load background stable.
3. Use repeated runs (3–5) to verify consistency.
4. Verify with exported JSON raw loops and P50 values.
5. Inspect `execution_time_sec` and `latency_chain_mode` to ensure measurement conditions were consistent.

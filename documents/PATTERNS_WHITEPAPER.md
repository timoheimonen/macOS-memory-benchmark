# CPU Memory Access Patterns Methodology

## Abstract

`memory_benchmark -P` (`--patterns`) is a standalone, versioned Apple Silicon user-space CPU memory access-pattern benchmark for macOS 26 or later. Across a persistent `mmap` buffer pair (`src`, `dst`), it evaluates seven canonical access patterns—`sequential_forward`, `sequential_reverse`, `strided_64`, `strided_4096`, `strided_16384`, `strided_2mb`, and `random`—across `read`, `write`, and `copy` operations using 32-byte logical accesses, 64-byte worker slice partitioning, and phase-aware strided accounting.

Each timed measurement executes a synchronized worker team using native AArch64 assembly kernels (`memory_read*_loop_asm`, `memory_write*_pattern_loop_asm`, `memory_copy*_loop_asm`) bracketed by `dsb ish; isb` barriers and Mach `cpu_raw` tick boundaries. Read operations accumulate a non-cancelling per-pass checksum (`patterns-read-checksum-v1`) over an O(1) slot-tagged GF(2) `src` buffer (`patterns-buffer-pattern-v1`) protected with `mprotect(PROT_READ)`. Write and copy operations use unique odd 64-bit write patterns and `pre_op_sentinel` values followed by parallel cold post-validation over 100% of touched bytes and, whenever sparse strides or clamped random selections leave destination gaps, 100% of untouched sentinel bytes outside the timed interval. The reported metric is exact completed user-space payload bytes divided by elapsed Mach time (`GB/s = exact_payload_bytes / elapsed_seconds / 1e9`). It is not a physical DRAM hardware-counter measurement, not proof of cache, DRAM, or superpage residency, and not an isolated prefetcher or TLB mechanism test.

## Methodology identity

The current contract is identified by:

| Property | Value | Source Citation |
|---|---|---|
| Software version | `SOFTVERSION` (`0.66.0` in current fixtures); release provenance only | `src/core/config/version.h`, `src/pattern_benchmark/patterns_json.cpp` |
| Mode | `patterns` (`kPatternsModeName`) | `src/pattern_benchmark/patterns_work_plan.h`, `src/core/config/constants.h` |
| JSON schema | `1` (`kPatternsSchemaVersion`) | `src/pattern_benchmark/patterns_work_plan.h` |
| Methodology | `patterns-v1-arm64-phase-calibrated-seeded-balanced` | `src/pattern_benchmark/patterns_work_plan.h` |
| Component identity (`component_identities.version`) | `patterns-components-v1` | `src/pattern_benchmark/patterns_work_plan.h`, `src/pattern_benchmark/patterns_json.cpp` |
| Work-plan identity (`component_identities.work_plan_version`) | `patterns-work-plan-v1` | `src/pattern_benchmark/patterns_work_plan.h`, `src/pattern_benchmark/patterns_json.cpp` |
| Buffer pattern (`component_identities.buffer_pattern_version`) | `patterns-buffer-pattern-v1` | `src/pattern_benchmark/patterns_work_plan.h`, `src/pattern_benchmark/patterns_json.cpp` |
| Read checksum (`component_identities.read_checksum_version`) | `patterns-read-checksum-v1` | `src/pattern_benchmark/patterns_work_plan.h`, `src/pattern_benchmark/patterns_json.cpp` |
| Random index generator (`component_identities.random_index_generator_version`) | `splitmix64-coprime-fisher-yates-v1` | `src/pattern_benchmark/patterns_work_plan.h`, `src/pattern_benchmark/patterns_json.cpp` |
| Run policy (`component_identities.run_policy_version`) | `bounded-loop-snapshots-v1` | `src/pattern_benchmark/patterns_work_plan.h`, `src/pattern_benchmark/patterns_json.cpp` |
| Canonical pattern order | `sequential_forward`, `sequential_reverse`, `strided_64`, `strided_4096`, `strided_16384`, `strided_2mb`, `random` | `src/pattern_benchmark/patterns_work_plan.h`, `src/pattern_benchmark/patterns_work_plan.cpp` |
| Canonical operation order | `read`, `write`, `copy` | `src/pattern_benchmark/patterns_work_plan.h`, `src/pattern_benchmark/patterns_work_plan.cpp` |
| Copy payload multiplier | `2` (`read + write`, `2 * total_accesses * 32`) | `src/pattern_benchmark/patterns_work_plan.cpp`, `src/core/config/constants.h` |
| Timing boundary | Synchronized worker start gate to last-worker `dsb ish` + `HighResTimer::stop()` (`dsb ish; isb` + `mach_absolute_time()`) | `src/pattern_benchmark/patterns_executor.cpp`, `src/core/timing/timer.cpp` |
| Write/Copy post-validation | Parallel cold 100% touched-byte check + 100% untouched-sentinel check (when `has_untouched_dst_bytes == true`) | `src/pattern_benchmark/patterns_oracle.cpp`, `src/pattern_benchmark/patterns_executor.cpp` |
| Calibration policy | Excluded auto-calibration toward `0.150 s` (`[0.100 s, 0.250 s]`), max `2` corrections, `8 MiB` min pilot | `src/core/config/constants.h`, `src/pattern_benchmark/patterns_runner.cpp` |
| Sweep envelope schema | `1` (`mode: "sweep"`, `base_mode: "patterns"`, `sweep_schema_version: 1`) | `src/pattern_benchmark/patterns_sweep_runner.cpp`, `src/benchmark/sweep_runner.cpp` |

A cross-run comparison requires matched `methodology_version`, complete `component_identities`, frozen `plan_identity` and per-pattern worker plans, base seed and explicit-vs-auto work policy, software version, hardware, thermal/power state, and output transport cadence (`comparability_requires`).

## Scope and non-goals

`--patterns` models effective user-space payload throughput across seven access geometries over a single configured main buffer (`--buffer-size`):

- **Effective payload accounting**:
  - `read`: `exact_payload_bytes = total_accesses * 32`
  - `write`: `exact_payload_bytes = total_accesses * 32`
  - `copy`: `exact_payload_bytes = 2 * total_accesses * 32` (counting both the logical 32-byte read from `src` and the logical 32-byte write to `dst` per access)
  - Rate: `bandwidth_gb_s = exact_payload_bytes / elapsed_seconds / 1e9` (decimal gigabytes per second, `10^9` bytes/s).
- **Physical and semantic non-claims** (published directly under `interpretation` in schema-1 JSON):
  - `dram_residency: "unverified"` and `cache_residency: "unverified"`. Buffer sizing and access strides do not prove which hardware cache, system-level cache (SLC), or DRAM controller serviced individual loads or stores.
  - Cache-line fills (each 32-byte logical access touches half of a 64-byte Apple Silicon cache line), write-allocate / read-for-ownership (RFO) traffic, dirty writebacks, hardware prefetcher speculation, TLB/page-table walks, and register checksum mixing are excluded from the payload numerator, though their physical cost appears in elapsed time.
  - `non_cacheable_requested` reflects `-u` / `--non-cacheable`, and `non_cacheable_is_best_effort_madvise: true`: `-u` calls `madvise(..., MADV_RANDOM)` rather than `MADV_WILLNEED`. User-space macOS cannot configure MAIR or page-table attributes to create uncacheable DRAM mappings.
  - `large_page_backing_verified: false` globally, and per-pattern `large_page_backing_status` is `"not-verified"` for `strided_2mb` (`"not-applicable"` for the other six patterns). `strided_2mb` denotes a 2 MiB virtual-address stride and does not establish physical 2 MiB superpage backing.
  - `single_pass_xor_collision_caveat`: `"Per-pass read XOR detects single-bit and slice-address errors, but even-parity identical flips within one pass can cancel in 64-bit XOR."`
  - `write_copy_post_validation_verifies_final_state_not_intermediate_passes: true`: cold post-validation inspects final `dst` state after `passes` iterations, not intermediate passes.
  - `comparability_requires`: `["same-methodology-version", "same-frozen-work-plan-and-patterns", "same-software-version", "sufficiently-similar-hardware", "sufficiently-similar-thermal-and-power-state"]`.
  - Pattern ratios must not be treated as isolated TLB or prefetcher measurements; use `-T` / `--analyze-tlb` for controlled paired translation-boundary analysis and `-L` / `--cpu-latency` for dependent pointer-chase latency.

## 1. CLI, transport boundary, and exit codes

The standalone primary mode is `-P` / `--patterns`, mutually exclusive with the six sibling modes in the seven-mode CLI (`-B`, `-L`, `-T`, `-C`, `-G`, `-M`). Its exact 11-option whitelist (`kPatternsOptionSpecs` in `src/pattern_benchmark/patterns.cpp`) is:

```text
-P, --patterns                Run standalone audited memory access pattern benchmark
-b, --buffer-size <MB>        Buffer size in MB (1-9007199254740991, default: 512)
-t, --threads <n>             Requested worker thread count (default: logical cores)
-i, --iterations <n>          Fixed pass count per operation (1-1000000000; default: auto-calibrate)
-r, --count <n>               Number of benchmark loops (default: 3)
    --seed <uint64>           Explicit 64-bit base seed (0 allowed)
-u, --non-cacheable           Best-effort madvise(MADV_RANDOM) allocation hint
-o, --output <target>         Write schema-1 JSON to file, or '-' for stdout JSON
-S, --sweep <key=v1,v2,...>   Parameter sweep (keys: buffer-size, threads)
-X, --sweep-max-runs <n>      Maximum generated sweep runs (default: 256)
-h, --help                    Display this help message and exit
```

### Parser, range, and conflict rules

1. **Syntax & uniqueness**: Built on `src/core/config/cli_parser.{h,cpp}`. Long value-taking options accept both `--option value` and `--option=value` (including `--output=-`, `--output=`, and `--sweep=buffer-size=64,128`). Inline values on `None` flags (`--patterns=1`, `--help=1`, `--non-cacheable=1`) and single-dash inline syntax (`-b=16`) are rejected. Every option except `-S` / `--sweep` may appear at most once; `-S` / `--sweep` is repeatable across distinct keys (`buffer-size`, `threads`), and duplicate keys are rejected. `--seed` is long-only (`short_name = nullptr`). Any option outside the 11-option whitelist (including `--cache-size`, `--latency-*`, `--compare-locality`, `--tlb-density`, `--gpu-bandwidth`, `--llm-memory`, or removed legacy flags) is rejected with `Unsupported option '<token>': ...` naming the allowed options.
2. **Numeric bounds**: All integer inputs must be strict unsigned decimal tokens (`[0-9]+`, no signs, whitespace, or prefixes):
   - `-b, --buffer-size`: `1 .. 9007199254740991` (`1 .. 2^53 - 1`), default `512` MiB. **`0` is explicitly rejected.**
   - `-t, --threads`: `1 .. 4294967295` (`1 .. UINT32_MAX`), default `available_workers` (detected logical CPU cores). Explicit values above detected cores are preserved in `configuration.requested_workers` and clamped per pattern in `resolved_plan.patterns.<kind>.effective_workers` (never globally truncated during CLI validation).
   - `-i, --iterations`: `1 .. 1000000000` (`1 .. 10^9`, `kPatternMaxPasses`), default omitted (automatic calibration).
   - `-r, --count`: `1 .. 9007199254740991` (`1 .. 2^53 - 1`), default `3`.
   - `--seed`: `0 .. 18446744073709551615` (`0 .. UINT64_MAX`), default omitted (generates one nonzero 64-bit seed once per command via `SeedUtils::generate_seed()`).
   - `-X, --sweep-max-runs`: `1 .. 9007199254740991` (`1 .. 2^53 - 1`), default `256`.
3. **Primary-mode conflict detection**: `select_primary_benchmark_mode` (`src/core/config/mode_selector.cpp`) pre-scans all seven primary mode flags while skipping one opaque value token for every known space-separated value-taking option (so `-o -G` or `--output -G` does not trigger a false conflict). Combining `-P` / `--patterns` with any sibling primary mode returns `PrimaryBenchmarkMode::Conflict` and exits `1`.
4. **Isolated help**: `-h` / `--help` runs after syntactic, whitelist, duplicate, and value/range checks and exits `0` before OS queries, sweep preflight, buffer allocation, or file creation (`--output - --help` remains human-facing stdout).
5. **Sweep rules**: `-S` / `--sweep` supports keys `buffer-size` (`1 .. 2^53 - 1`) and `threads` (`1 .. UINT32_MAX`). It rejects duplicate sweep keys, requires non-empty `--output <target>`, forbids combining `--sweep buffer-size=...` with `-b` / `--buffer-size` or `--sweep threads=...` with `-t` / `--threads`, bounds the Cartesian product to `<= sweep_max_runs`, and preflight-validates **every** generated run's work plan and `memory_budget` (`preflight_patterns_sweep`) before run `0` begins.

### Transport boundary and process exit codes

- **Empty/omitted `--output`**: Disables JSON serialization (`JsonOutputKind::Disabled`, console-only for direct commands; rejected for `--sweep`).
- **Exact `--output -`**: Selects `JsonOutputKind::Stdout`. `stdout` receives exclusively one final two-space-indented UTF-8 JSON document followed by one newline; all post-parse human output goes to `stderr`. Intermediate checkpoints are lazy no-ops (`stdout_intermediate_checkpoints_are_lazy: true`), and `configuration.output_file` preserves `"-"` verbatim.
- **Any other non-empty `--output <target>`**: Selects `JsonOutputKind::File`, resolved relative to CWD (`JsonFilePathPolicy::ResolveAgainstCurrentDirectory`, including `./-` and flag-shaped names). Direct runs write atomic `<target>.tmp` replacements every `K = max(1, ceil(loop_count / 8))` completed loops plus command terminal; sweep runs checkpoint the combined schema-1 envelope after each attempted run.
- **Exit status (`0` / `1`) vs `run_accepted`**: `EXIT_SUCCESS` (`0`) is returned for `-h`/`--help`, a complete run, or a graceful `SIGINT`/`SIGTERM` interruption (both preparation interruption and task-boundary interruption) whose terminal checkpoint succeeded. `EXIT_FAILURE` (`1`) is returned on CLI/config/preflight error, invalid work plan, resource preparation failure, timer/kernel/validation/checksum failure, uncaught exception, or checkpoint write failure. Consumers must never treat exit code `0` alone as a valid measurement: authoritative acceptance requires `schema_version == 1`, `mode == "patterns"`, `methodology_version == "patterns-v1-arm64-phase-calibrated-seeded-balanced"`, `status == "complete"`, `results_complete == true`, and `run_accepted == true` (plus `conclusions_valid == true` and `completed_runs == planned_runs` for sweeps).

## 2. 7-pattern geometry, partitioning, and guardrails

Every pattern operates on a single configured buffer size `buffer_bytes = buffer_size_mb * 1,048,576` with fixed logical access width `access_size_bytes = 32` (`kPatternsAccessSizeBytes`). In canonical order (`src/pattern_benchmark/patterns_work_plan.{h,cpp}`):

| Index | Canonical `PatternKind` | `stride_bytes` | `phase_period_passes` (`Q`) | Category (`cv_warning_threshold_pct`) |
|---:|---|---:|---:|---|
| 0 | `sequential_forward` | `0` | `0` | Streaming (`5.0%`) |
| 1 | `sequential_reverse` | `0` | `0` | Streaming (`5.0%`) |
| 2 | `strided_64` | `64` | `2` | Streaming (`5.0%`) |
| 3 | `strided_4096` | `4096` | `128` | Sparse (`10.0%`) |
| 4 | `strided_16384` | `16384` | `512` | Sparse (`10.0%`) |
| 5 | `strided_2mb` | `2097152` | `65536` | Sparse (`10.0%`) |
| 6 | `random` | `0` | `0` | Sparse (`10.0%`) |

For each pattern entry:
- `phase_period_passes = (stride_bytes > 0) ? (stride_bytes / 32) : 0`.
- `stride_equals_native_page_size = (stride_bytes > 0 && page_size_bytes > 0 && stride_bytes == page_size_bytes)` (on Apple Silicon 16 KiB pages, `true` for `strided_16384`).
- `large_page_backing_verified = false`; `large_page_backing_status = "not-verified"` for `strided_2mb` and `"not-applicable"` for all other patterns.

### 64-byte slice partitioning (`partition_patterns_64b`)

`partition_patterns_64b(total_bytes, workers)` divides `[0, total_bytes)` across `active = min(total_bytes, workers)` workers using proportional cuts snapped to 64-byte boundaries:

```text
boundaries[0] = 0
for rank = 1 .. active - 1:
  target     = (total_bytes / active) * rank + min(rank, total_bytes % active)
  lo         = boundaries[rank - 1] + 1
  hi         = total_bytes - (active - rank)
  target     = clamp(target, lo, hi)
  aligned_lo = ceil(lo / 64) * 64
  aligned_hi = floor(hi / 64) * 64
  if aligned_lo <= aligned_hi:
    down = clamp(floor(target / 64) * 64, aligned_lo, aligned_hi)
    up   = (down < target) ? min(aligned_hi, down + 64) : down
    target = (abs(target - down) <= abs(target - up)) ? down : up
  boundaries[rank] = target
boundaries[active] = total_bytes
```

Every interior cut `boundaries[rank]` (`rank > 0` when `aligned_lo <= aligned_hi`) is a 64-byte multiple, and ties break downward (`down`). Any remaining workers beyond `active` receive `{0, 0, 0, 0}`.

### Per-pattern worker selection and exact 2D phase rotation metrics

1. **Sequential (`sequential_forward`, `sequential_reverse`)**:
   - `effective_workers = min(requested_workers, available_workers, max(1, buffer_bytes / 64))`.
   - Per worker: `sp.accesses_per_pass = sp.span_bytes / 32`.
   - Across workers: `accesses_per_pass = sum(sp.accesses_per_pass)`, `min_accesses_per_pass = max_accesses_per_pass = accesses_per_pass`, `total_accesses = passes * accesses_per_pass`, `distinct_address_count = accesses_per_pass`, `logical_working_set_bytes = buffer_bytes`, `completed_phase_cycles = 0`, `has_untouched_dst_bytes = false`.

2. **Strided worker reduction and `strided_2mb` skip contract**:
   - Minimum worker span required for at least one stride transition (`>= 2` 32B accesses) is `min_worker_span = stride_bytes + 32`.
   - If `buffer_bytes < min_worker_span`—in particular, for `strided_2mb`, whenever `buffer_bytes < 2 MiB + 32 B = 2,097,184 B` (e.g., `1 MiB` and `2 MiB` buffers)—the pattern is **not** failed; it is published in `resolved_plan.patterns.<kind>` and `aggregates.patterns.<kind>.<op>` as:
     - `enabled: false`, `status: "skipped"`, `reason_code: "buffer-too-small-for-stride-transition"`
     - `effective_workers: 0`, `worker_spans: []`
     - `frozen_plan_refs.<kind>.{read, write, copy}: null`
     - Omitted from `calibration.attempts`, `measurements[]`, and `loop_records[]`.
   - Otherwise, `first_candidate = min(requested_workers, available_workers, buffer_bytes / min_worker_span)` counts down to `1` until every span produced by `partition_patterns_64b(buffer_bytes, candidate)` satisfies `span_bytes >= min_worker_span`, `offset_bytes % 32 == 0`, `span_bytes % 32 == 0`, and `sp.accesses_per_pass = 1 + (sp.span_bytes - 32) / stride_bytes >= 2`.
   - For `passes` passes with `Q = stride_bytes / 32`:
     ```text
     completed_phase_cycles = passes / Q
     remaining_phases       = passes % Q
     executed_phase_count   = min(passes, Q)
     last_executed_phase    = executed_phase_count - 1
     ```
     For each worker span with `aligned_slots = sp.span_bytes / 32`, `full_blocks = aligned_slots / Q`, and `tail_slots = aligned_slots % Q`:
     ```text
     worker_max_accesses_per_pass = 1 + (aligned_slots - 1) / Q
     worker_min_accesses_per_pass = (last_executed_phase < aligned_slots)
                                    ? (1 + (aligned_slots - 1 - last_executed_phase) / Q) : 0
     worker_total_accesses        = completed_phase_cycles * aligned_slots
                                    + full_blocks * remaining_phases
                                    + min(tail_slots, remaining_phases)
     worker_distinct_addresses    = full_blocks * executed_phase_count
                                    + min(tail_slots, executed_phase_count)
     ```
     Summing across all effective workers yields `total_accesses`, `min_accesses_per_pass`, `max_accesses_per_pass`, `distinct_address_count`, `logical_working_set_bytes = sum((last_selected_slot + 1) * 32)` over workers with at least one selected slot, and `has_untouched_dst_bytes = (distinct_address_count * 32 < buffer_bytes)`.

3. **Random clamping and worker split (`distribute_random_accesses`)**:
   - `total_slots = buffer_bytes / 32`.
   - `clamped = clamp(total_slots, 1000, 1_000_000)`; `selected = min(total_slots, clamped)`.
   - `effective_workers = min(requested_workers, available_workers, max(1, buffer_bytes / 64), selected)`.
   - `distribute_random_accesses` starts each worker with `1` access, distributes `remaining = selected - effective_workers` proportionally to spare slot capacity `cap_w = (span_bytes / 32) - 1` via 128-bit arithmetic `floor(remaining * cap_w / sum_capacity)`, and assigns any integer-division remainder round-robin across workers with `random_access_count < span_slots`.
   - `accesses_per_pass = min_accesses_per_pass = max_accesses_per_pass = selected`, `distinct_address_count = selected`, `total_accesses = passes * selected`, `logical_working_set_bytes = span_covering_selected_slots` (the populated distinct slots cover `buffer_bytes` whenever `selected == total_slots`, e.g., `<= 30 MiB`), and `has_untouched_dst_bytes = (selected * 32 < buffer_bytes)` (e.g., `true` once `buffer_bytes > 32,000,000 B`).

### Guardrails

| Guardrail | Exact Limit | Where Enforced |
|---|---|---|
| `kPatternMaxPasses` | `1,000,000,000` (`10^9`) | CLI `--iterations`, `build_patterns_operation_plan`, verifier |
| `kPatternMaxExactPayloadBytes` | `68,719,476,736` (`64 GiB`) per operation task | `build_patterns_operation_plan`, calibration cap, verifier |
| `kPatternMaxMeasurements` | `100,000` (`loop_count * active_patterns.size() * 3`) | `build_patterns_work_plan`, verifier |
| `LLM_JSON_MAX_SAFE_INTEGER` | `9,007,199,254,740,991` (`2^53 - 1`) | Numeric CLI options, sweep counts |
| Verifier `operation_plans` cap | `4,096` entries | `verify_patterns_result.py` |
| Verifier random-index cap | `1,000,000` (`--max-random-indices` default) | `verify_patterns_result.py` |
| Verifier shared input/DOM/work limits | `64 MiB` UTF-8 JSON, depth `64`, `1,000,000` nodes, `1,000,000` oracle work steps | `cpu_verify_common.py`, `verify_patterns_result.py` |
| Streaming / Sparse CV thresholds | `5.0%` streaming / `10.0%` sparse (`n >= 3`) | `patterns_work_plan.h`, `aggregates` quality warnings |

## 3. Golden mathematical test vectors

Every vector below was verified bit-for-bit across C++ (`src/pattern_benchmark/patterns_oracle.cpp`, `tests/test_patterns_oracle.cpp`), Python (`script-examples/verify_patterns_result.py`, `tests/test_patterns_verifier.py`), and the pinned schema-1 fixtures in `tests/fixtures/` for `base_seed = 424242`.

### 3.1 Domain seeds, write pattern, and pre-operation sentinel (`base_seed = 424242`)

| Quantity | Formula | Hex | Unsigned Decimal |
|---|---|---|---:|
| `src_seed` | `splitmix64_mix(424242 ^ 0x5041545F5352435F)` | `0x63329A0ADA1F7B42` | `7147944929990245186` |
| `random_domain_seed` | `splitmix64_mix(424242 ^ 0x5041545F52414E44)` | `0x8C0E47743450068D` | `10092082379403298445` |
| `pre_op_sentinel_seed` | `splitmix64_mix(424242 ^ 0x5041545F53454E54)` | `0x216631658CFD9063` | `2406665363102994531` |
| `write_op_seed` (`SequentialForward`, `Write`, `passes = 2`) | `splitmix64_mix(src_seed ^ (1 << 48) ^ (2 << 56) ^ 2)` | — | `10362866072134435624` |
| `write_attempt_seed` (`ordinal = 1`) | `splitmix64_mix(write_op_seed ^ (0x5752495400000000 \| 1))` | — | `13317374826786385297` |
| `write_pattern` (`ordinal = 1`) | `splitmix64_mix(write_attempt_seed) \| 1` | `0xD285D5C0A0D095DF` | `15169765943175845343` |
| `pre_op_sentinel` (`ordinal = 1`) | `splitmix64_mix(pre_op_sentinel_seed ^ 1) \| 1` | `0xB0FA22685008F2B1` | `12752543126267687601` |

In `tests/fixtures/patterns-schema-v1-complete-current.json` (`./memory_benchmark -P -b 3 -t 2 -i 2 -r 3 --seed 424242`), `ordinal = 1` for `write_pattern` (`15169765943175845343`, `attempt_seed = 13317374826786385297`) appears in `calibration.attempts.sequential_forward.write[0]` (`frozen_measurement_warmup`), and `ordinal = 1` for `pre_op_sentinel` (`12752543126267687601`) appears in `calibration.attempts.sequential_forward.copy[0]`. Because the command runs 7 `frozen_measurement_warmup` writes before loop `0`, `measurements[1]` (`sequential_forward`, `write`, `loop_index = 0`) uses `ordinal = 8` (`attempt_seed = 12170823724597505764`, `write_pattern = 15545434012381584747`), while `write_ordinal` and `sentinel_ordinal` continue monotonically across all calibration/warmup and measured loops.

### 3.2 3 MiB, 2-worker random digest and read checksum vectors (`base_seed = 424242`)

For `buffer_size_mb = 3` (`3,145,728 B`), `requested_workers = 2`:

- **`random` workload**: `effective_workers = 2`, spans `[(0, 1572864), (1572864, 1572864)]`, `selected_accesses_per_pass = 98304` (`49152` per worker), `logical_working_set_bytes = "3145728"`, and:
  - `random_indices_sha256`: `"c199eec93986c9390e43f9051fb3c7ccd139255c48590c5da11b86f75d46214e"`
- **`sequential_forward` read (`passes = 2`, `op_seed = 17788670734524640512`, `effective_workers = 2`)**:
  - Worker `0`: `state_a = 12333082077594647533`, `state_b = 327418441216473036`, `exact_bytes_read = 3145728`
  - Worker `1`: `state_a = 16990341082608790728`, `state_b = 12180571865171809690`, `exact_bytes_read = 3145728`
  - Folded `expected_run_checksum`: `state_a = 8998662144812030758` (`0x7CE1AB8B33469B26`), `state_b = 10912035358685781170` (`0x976F565822F7DCB2`)
- **`sequential_reverse` read (`passes = 2`, `op_seed = 7929026729967543888`, `effective_workers = 2`)**:
  - Worker `0`: `state_a = 14108980171579688653`, `state_b = 6592244529501121154`, `exact_bytes_read = 3145728`
  - Worker `1`: `state_a = 10858911041219858487`, `state_b = 8111634715484542969`, `exact_bytes_read = 3145728`
  - Folded `expected_run_checksum`: `state_a = 14622463275573373208` (`0xCAED6CE041BEE918`), `state_b = 2278219113277803231` (`0x1F9DDC36876DCADF`)
- **`strided_64` read (`passes = 3`, 2D partial-cycle unit golden in `test_patterns_oracle.cpp` & `test_patterns_verifier.py`, 1 full cycle `Q = 2` + 1 partial phase `rem = 1`)**:
  - Folded `expected_run_checksum`: `state_a = 1392249341305891027` (`0x13524366434F28D3`), `state_b = 13846455899336778268` (`0xC0287E475E14A21C`)
- **`strided_64` read (`passes = 2`, pinned `-i 2` fixture `patterns-schema-v1-complete-current.json`, `op_seed = 9764589247570031596`, `completed_phase_cycles = 1`)**:
  - Worker `0`: `state_a = 9632105972612422222`, `state_b = 13775808598636445219`, `exact_bytes_read = 1572864`
  - Worker `1`: `state_a = 14516050437196632752`, `state_b = 3868906133693776831`, `exact_bytes_read = 1572864`
  - Folded `expected_run_checksum`: `state_a = 16424153263904442687`, `state_b = 6370472496738990699`
- **`strided_2mb` read (`passes = 2`, `effective_workers = 1`, `span = (0, 3145728)`, `op_seed = 13589639886950992781`)**:
  - Worker `0`: `state_a = 10085333006801225084`, `state_b = 3135862343368555312`, `exact_bytes_read = 128`
  - Folded `expected_run_checksum`: `state_a = 12870340894742460177` (`0xB29CA2DA914EB711`), `state_b = 17290424146719303160` (`0xEFF3ED27210A65F8`)
- **`random` read (`passes = 2`, `op_seed = 15312585825716255573`, `effective_workers = 2`)**:
  - Worker `0`: `state_a = 6815219213458855127`, `state_b = 9009944095944866028`, `exact_bytes_read = 3145728`
  - Worker `1`: `state_a = 6345256335102328441`, `state_b = 14332238081747437155`, `exact_bytes_read = 3145728`
  - Folded `expected_run_checksum`: `state_a = 4299705137856029524` (`0x3BAB9F0AC4D4A354`), `state_b = 5187293785770800317` (`0x47FCF836F675B4BD`)

### 3.3 1 MiB, 2-worker skipped `strided_2mb` and random digest (`patterns-schema-v1-skipped-2mb-current.json`)

For `./memory_benchmark -P -b 1 -t 2 -i 2 -r 3 --seed 424242` (`1,048,576 B < 2,097,184 B`):

- **`resolved_plan.patterns.strided_2mb`**: `enabled: false`, `status: "skipped"`, `reason_code: "buffer-too-small-for-stride-transition"`, `access_size_bytes: 32`, `stride_bytes: 2097152`, `phase_period_passes: 65536`, `native_page_size_bytes: 16384`, `stride_equals_native_page_size: false`, `large_page_backing_verified: false`, `large_page_backing_status: "not-verified"`, `effective_workers: 0`, `worker_spans: []`.
- **`resolved_plan.frozen_plan_refs.strided_2mb`**: `{"read": null, "write": null, "copy": null}`.
- **`aggregates.patterns.strided_2mb.{read,write,copy}`**: `status: "skipped"`, `reason_code: "buffer-too-small-for-stride-transition"`, `accepted_measurement_ids: []`, `value: null`, `relative_to_forward_pct: null`, `statistics: null`, `quality: {"repeatability_assessment_available": false, "run_complete": false, "cv_warning_threshold_pct": 10.0, "cv_warning": false}`.
- **Scheduled counters**: `6` active patterns × `3` ops = `18` per loop, `planned_measurements: 54`, `planned_passes: "108"`, `planned_exact_payload_bytes: "88326144"`.
- **`random` workload at 1 MiB**: `selected_accesses_per_pass = 32768`, `logical_working_set_bytes = "1048576"`, `random_indices_sha256 = "aa26d6dbb9e8d149f233aad4980c9d3f3d1019b7a17e46e746ec677c1b8aee6d"`.

## 4. Physical memory mappings, strict memory budget, and resource evidence

Before allocating any buffers, `build_patterns_work_plan` evaluates strict memory admission against `available_memory_bytes`:

```text
if available_memory_bytes == 0:
  used_fallback        = true
  allowed_memory_bytes = 2048 * 1,048,576   // 2048 MiB fallback (Constants::FALLBACK_TOTAL_LIMIT_MB)
else:
  used_fallback        = false
  allowed_memory_bytes = floor(available_memory_bytes * 0.80)
admitted_budget_bytes  = allowed_memory_bytes
```

The command owns one persistent `MAP_PRIVATE | MAP_ANONYMOUS` buffer pair (`src` and `dst`) plus random index storage (`worker_indices` + `sorted_validation_indices`):

```text
requested_data_bytes    = 2 * buffer_bytes
committed_data_bytes    = 2 * round_up(buffer_bytes, page_size_bytes)
resource_rounding_bytes = committed_data_bytes - requested_data_bytes
index_structure_bytes   = 2 * selected_random * sizeof(size_t)   // 2 * selected_random * 8
setup_peak_bytes        = 4096 + max_effective_workers * 64 + index_structure_bytes
```

When JSON serialization is active (`!output_file.empty() || collect_json_for_sweep`), a conservative DOM reserve (`json_peak_estimate.total_bytes`) is added before admission (for console-only direct runs, `json_peak_estimate.total_bytes = 0`):

- `fixed_schema_bytes = 2 MiB` (`2,097,152 B`).
- `input_string_bytes = 16 * (output_file.size() + sum(argv[i].size()))`.
- `calibration_attempts_per_op = iterations.has_value() ? 1 : 6` (`4 + BANDWIDTH_CALIBRATION_MAX_CORRECTIONS`).
- `per_loop_measurements = active_patterns.size() * 3`.
- `total_op_plans = per_loop_measurements * (calibration_attempts_per_op + 1)`, `total_calibration_attempts = per_loop_measurements * calibration_attempts_per_op`.
- `measurement_record_bytes = 64 KiB * (planned_measurements + total_calibration_attempts + total_op_plans)`.
- `worker_checksum_bytes = 64 KiB * sum_{p in active} ((loop_count + calibration_attempts_per_op + 1) * effective_workers(p))`.
- `total_bytes = fixed_schema_bytes + input_string_bytes + measurement_record_bytes + worker_checksum_bytes`.

Total peak admission is:

```text
transient_peak_bytes   = planned_measurements * 512 + 4096 + index_structure_bytes + json_peak_estimate.total_bytes
runtime_peak_bytes     = committed_data_bytes + transient_peak_bytes
known_owned_peak_bytes = runtime_peak_bytes
admitted               = (known_owned_peak_bytes <= admitted_budget_bytes)
```

Workloads exceeding `admitted_budget_bytes` are rejected with `reason_code: "memory-budget-exceeded"` and are **never** silently capped or downsized.

### Buffer lifecycle and `AllocatedBufferEvidence`

`prepare_patterns_resources` allocates `src` and `dst` via `allocate_buffer_with_evidence()` (`src/core/memory/memory_manager.cpp`), recording `requested_bytes`, `mapped_bytes`, `non_cacheable_requested`, `madvise_advice` (`MADV_RANDOM = 1` when `-u`, else `MADV_WILLNEED = 3` on macOS), `madvise_applied`, `madvise_errno`, `read_only_protected`, and `mprotect_errno` under `resolved_plan.resources.{src,dst}`:

1. `src` is populated in parallel across `buffer_worker_spans` with `patterns-buffer-pattern-v1`.
2. `src` is protected read-only via `protect_buffer_read_only_with_evidence(resources.src, protect_size)` (`mprotect(ptr, mapped_bytes > 0 ? mapped_bytes : buffer_bytes, PROT_READ)`). Failure aborts preparation (`src-readonly-protection-failed`).
3. `dst` is pre-populated in parallel with `pre_op_sentinel(sentinel_ordinal = 0)` so all pages of `dst` are resident and faulted before timing starts.
4. When `random` is active, per-worker `random` index tables and ascending `sorted_validation_indices` (`<= 8 MiB` at `1,000,000 * 8 B`) are generated and recorded in `random_index_table_bytes` and `random_validation_index_bytes`.
5. Outside the timer before every `copy` attempt (and before every `write` attempt where `has_untouched_dst_bytes == true`), `dst` is reset in parallel with a fresh odd `pre_op_sentinel` (`sentinel_ordinal = 1, 2, ...`).

## 5. ARM64 kernels, QoS, and synchronized execution gate

### 7 × 3 kernel dispatch matrix

| Pattern Kind | `read` Kernel (`.s` source) | `write` Kernel (`.s` source) | `copy` Kernel (`.s` source) |
|---|---|---|---|
| `sequential_forward` | `memory_read_loop_asm` (`src/asm/memory_read.s`) | `memory_write_pattern_loop_asm` (`src/asm/memory_write_pattern.s`) | `memory_copy_loop_asm` (`src/asm/memory_copy.s`) |
| `sequential_reverse` | `memory_read_reverse_loop_asm` (`src/asm/memory_read_reverse.s`) | `memory_write_reverse_pattern_loop_asm` (`src/asm/memory_write_reverse_pattern.s`) | `memory_copy_reverse_loop_asm` (`src/asm/memory_copy_reverse.s`) |
| `strided_64` | `memory_read_strided_phased_loop_asm` (`src/asm/memory_read_strided.s`) | `memory_write_strided_phased_pattern_loop_asm` (`src/asm/memory_write_strided_pattern.s`) | `memory_copy_strided_phased_loop_asm` (`src/asm/memory_copy_strided.s`) |
| `strided_4096` | `memory_read_strided_phased_loop_asm` (`src/asm/memory_read_strided.s`) | `memory_write_strided_phased_pattern_loop_asm` (`src/asm/memory_write_strided_pattern.s`) | `memory_copy_strided_phased_loop_asm` (`src/asm/memory_copy_strided.s`) |
| `strided_16384` | `memory_read_strided_phased_loop_asm` (`src/asm/memory_read_strided.s`) | `memory_write_strided_phased_pattern_loop_asm` (`src/asm/memory_write_strided_pattern.s`) | `memory_copy_strided_phased_loop_asm` (`src/asm/memory_copy_strided.s`) |
| `strided_2mb` | `memory_read_strided_phased_loop_asm` (`src/asm/memory_read_strided.s`) | `memory_write_strided_phased_pattern_loop_asm` (`src/asm/memory_write_strided_pattern.s`) | `memory_copy_strided_phased_loop_asm` (`src/asm/memory_copy_strided.s`) |
| `random` | `memory_read_random_loop_asm` (`src/asm/memory_read_random.s`) | `memory_write_random_pattern_loop_asm` (`src/asm/memory_write_random_pattern.s`) | `memory_copy_random_loop_asm` (`src/asm/memory_copy_random.s`) |

All kernels adhere to AAPCS64 (preserving callee-saved `x19–x28`, `x29`, `x30`, `sp`, and `v8–v15`; verified by `tests/test_pattern_write_kernels.cpp` and `tests/test_standard_kernels.cpp`) and emit no internal memory barriers, relying on the caller's `dsb ish; isb` timing boundaries:

- **`memory_write_reverse_pattern_loop_asm(dst, byteCount, pattern64)`**: Rotates `pattern64` via `ror x6, x2, ((byteCount & 7) * 8)` so descending 512B/256B/128B/64B/32B `stnp` chunks and the byte-by-byte tail (`ror x6, x6, #56; strb w6, [x3, #-1]!`) store `(pattern64 >> ((i & 7) * 8)) & 0xFF` at every byte offset `i`.
- **`memory_write_strided_phased_pattern_loop_asm(dst, byteCount, stride, passes, initial_phase, pattern64)`**: Broadcasts `pattern64` into `v0.2d` (`dup v0.2d, x5`) and `mov v1.16b, v0.16b`, executing `passes` phase-rotated 32B `stnp q0, q1` passes advancing the 32-byte phase modulo `stride`.
- **`memory_write_random_pattern_loop_asm(dst, indices, num_accesses, pattern64)`**: Broadcasts `pattern64` into `v0.2d` (`dup v0.2d, x3`) and `mov v1.16b, v0.16b`, loads each relative byte offset `indices[i]` (`lsl #3`), and issues 32B `stnp q0, q1` stores.

### QoS and synchronized worker gate

1. Main thread calls `prepare_main_thread_benchmark_qos()` (`QOS_CLASS_USER_INTERACTIVE`, best-effort) and records the outcome under `environment.main_thread_qos`.
2. `execute_patterns_operation` spawns `worker_count = pattern_plan.effective_workers` threads. Each worker calls `pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)` (best-effort; tracked via `qos_successful_workers` / `qos_failed_workers`), increments `ready_workers`, and waits on `state_cv` for `start_workers || cancel_workers`. Partial spawn failure sets `cancel_workers = true`, joins created threads cleanly, and records `worker-startup-failed`.
3. Once `ready_workers == worker_count`, the coordinator calls `timer.start()` (`dsb ish; isb` + `mach_absolute_time()`), sets `start_workers = true`, and broadcasts `state_cv`.
4. Worker execution (`p = 1 .. passes`):
   - **Sequential / Random `read`**: executes `passes` kernel calls over the worker's span or `indices`, updating `(state_a, state_b)` after every pass.
   - **Strided `read`** (`Q = stride_bytes / 32`, `full_cycles = passes / Q`, `rem_phases = passes % Q`): executes `full_cycles` calls to `memory_read_strided_phased_loop_asm(src + offset, span, stride, Q, 0)` (updating `(state_a, state_b)` at steps `1 .. full_cycles`), followed by one call with `rem_phases` passes if `rem_phases > 0` (step `full_cycles + 1`).
   - **`write` / `copy`**: sequential and random loop `passes` times (`memory_write_pattern_loop_asm`, `memory_write_reverse_pattern_loop_asm`, `memory_write_random_pattern_loop_asm`, or `memory_copy*_loop_asm`); strided `write` and `copy` execute a single kernel call passing `op_plan.passes` with `initial_phase = 0`.
5. Each worker finishes with `asm volatile("dsb ish" ::: "memory")`. The last worker (`remaining_workers.fetch_sub(1, acq_rel) == 1`) calls `timer.stop()` (`dsb ish; isb` + `mach_absolute_time()`), captures `MachTimingSnapshot` (`cpu_raw`), and wakes the coordinator. Cold post-validation executes after worker join, outside the timed interval.

## 6. Deterministic seeding, oracles, and cold post-validation

### `SplitMix64` seed hierarchy

All mixing uses `splitmix64_mix(x)` (`src/core/audit/deterministic_rng.h`):

```text
z = (x + 0x9E3779B97F4A7C15) mod 2^64
z = ((z xor (z >> 30)) * 0xBF58476D1CE4E5B9) mod 2^64
z = ((z xor (z >> 27)) * 0x94D049BB133111EB) mod 2^64
return z xor (z >> 31)
```

With fixed 64-bit domain constants (`src/pattern_benchmark/patterns_work_plan.h`):

- `kPatternDomainSrc = 0x5041545F5352435F` (`"PAT_SRC_"`)
- `kPatternDomainRandom = 0x5041545F52414E44` (`"PAT_RAND"`)
- `kPatternDomainPreOpSentinel = 0x5041545F53454E54` (`"PAT_SENT"`)

Derivations:

```text
src_seed             = splitmix64_mix(base_seed xor kPatternDomainSrc)
random_domain_seed   = splitmix64_mix(base_seed xor kPatternDomainRandom)
pre_op_sentinel_seed = splitmix64_mix(base_seed xor kPatternDomainPreOpSentinel)
op_seed              = splitmix64_mix(src_seed xor ((kind_index + 1) << 48) xor ((op_index + 1) << 56) xor passes)
                       // kind_index in 0..6, op_index in {read: 0, write: 1, copy: 2}
write_attempt_seed   = splitmix64_mix(op_seed xor (0x5752495400000000 | write_ordinal))
write_pattern        = splitmix64_mix(write_attempt_seed) | 1
pre_op_sentinel      = splitmix64_mix(pre_op_sentinel_seed xor sentinel_ordinal) | 1
```

Each `write` attempt increments `write_ordinal` (`1, 2, ...`) until `write_pattern` is nonzero, odd, and unseen in `seen_write_patterns`; each `pre_op_sentinel` draw increments `sentinel_ordinal` (`1, 2, ...`) until nonzero, odd, and distinct from `write_pattern`.

### Slot-tagged GF(2) `src` buffer (`patterns-buffer-pattern-v1`)

In `-B`'s `cpu-bw-buffer-pattern-v1`, `W[i] = T[i & 255] ^ (i ^ rotl64(i, 19) ^ rotl64(i, 41))` cancels on 4 KiB+ 32B strides: for any 4-word 32B slot `s` (`words 4s, 4s+1, 4s+2, 4s+3`), the upper 62 bits `(4s)` appear four times (even multiplicity) in `W[4s] ^ W[4s+1] ^ W[4s+2] ^ W[4s+3]` and cancel out completely in GF(2), while for strides `>= 2048 B` (`4096 B`, `16384 B`, `2 MiB`, which are multiples of `256` words) `T[i & 255]` also repeats identically across every slot in a phase, causing every even-count strided pass to XOR to `0` and lose all slot-address sensitivity.

`patterns-buffer-pattern-v1` resolves this by tagging word `0` of every 32-byte slot (`i & 3 == 0`) with an odd-multiplicity slot polynomial:

```text
T[k]           = splitmix64_mix(src_seed xor k),       k = 0 .. 255
P_T[0] = 0,      P_T[k + 1] = P_T[k] xor T[k]
S_T[u]         = T[4u] xor T[4u + 1] xor T[4u + 2] xor T[4u + 3],  u = 0 .. 63
P_ST[0] = 0,     P_ST[u + 1] = P_ST[u] xor S_T[u]
P_ST_even[k+1] = P_ST_even[k] xor S_T[2k],             k = 0 .. 31
P_ST_odd[k+1]  = P_ST_odd[k]  xor S_T[2k + 1],         k = 0 .. 31

lane_mix(i) = i xor rotl64(i, 7) xor rotl64(i, 33)
slot_mix(s) = s xor rotl64(s, 19) xor rotl64(s, 41)
W[i]        = T[i & 255] xor lane_mix(i) xor (((i & 3) == 0) ? slot_mix(i >> 2) : 0)
```

Because `slot_mix(s)` enters only word `4s`, the 4-word XOR of 32-byte slot `s` is:

```text
slot_xor(s) = W[4s] xor W[4s + 1] xor W[4s + 2] xor W[4s + 3] = S_T[s & 63] xor slot_mix(s)
```

which preserves unique slot-address sensitivity across arbitrary strides and random indices while remaining O(1)-summable over intervals and 2D stride blocks.

### O(1) read checksum oracle (`patterns-read-checksum-v1`)

Fixed constants (`src/pattern_benchmark/patterns_oracle.h`):
- `kA = 0x9E3779B97F4A7C15`, `kB = 0xBF58476D1CE4E5B9`, `kC = 0x94D049BB133111EB`
- `kFoldInitA = 0x6A09E667F3BCC909`, `kFoldInitB = 0xBB67AE8584CAA73B`
- `init_a(w) = splitmix64_mix(op_seed xor (0xA100000000000000 | w))`
- `init_b(w) = splitmix64_mix(op_seed xor (0xB200000000000000 | w))`

Using `prefix_xor_0_to_n_minus_1(n)` (`0` if `n == 0`, else `{m, 1, m+1, 0}[(n - 1) & 3]` for `m = n - 1`):
- **Sequential forward / reverse**: `sequential_forward` XORs the 32B-aligned prefix `[offset, offset + vec_bytes)` via `xor_vec_bytes_32_aligned` and the `<32B` tail `[offset + vec_bytes, offset + span)` into bits `[0..7]`; `sequential_reverse` XORs the `<32B` low tail `[offset, offset + span % 32)` first and the remaining 32B-aligned vector range `[offset + tail_bytes, offset + span)` second.
- **Strided full cycle (`Q = stride_bytes / 32`)**: `fc_xor = xor_slots_range(base_slot, base_slot + num_slots)` using `prefix_st_mod64` and `slot_mix(prefix_xor(s_end) ^ prefix_xor(s_start))`.
- **Strided 2D partial cycle (`rem = passes % Q < Q`, `Q = 2^q`)**: Splits `base_slot = (high_0 << q) | low_0`, `rem_1 = min(rem, Q - low_0)`, `rem_2 = rem - rem_1`, sums up to two 2D rectangles `xor_rect(high_0, high_0 + full_blocks, low_0, low_0 + rem_1)` and (when `rem_2 > 0`) `xor_rect(high_0 + 1, high_0 + 1 + full_blocks, 0, rem_2)`—using O(1) `prefix_st_mod64` for `q >= 6` (`stride >= 2048 B`), O(1) `prefix_st_even_mod32` / `prefix_st_odd_mod32` for `q == 1` (`stride == 64 B`), and a bounded 64-periodic loop for `2 <= q <= 5`—plus `tail_count = min(tail_slots, rem)` slots at `base_slot + (full_blocks << q)`.
- **Closed-form step fold** (`S1(n) = (n even ? (n/2)*(n+1) : n*((n+1)/2)) mod 2^64`):
  - For sequential/random (`N = passes`, `px = pass_xor`): `state_a = init_a + N * px + kA * S1(N)`, `state_b = init_b + (px ^ kB) * S1(N) + kC * N`.
  - For strided (`N = full_cycles = passes / Q`, `px = fc_xor`, plus optional final partial step `step = full_cycles + 1` with `pc_xor` when `rem_phases > 0`):
    ```text
    state_a = (init_a + N * fc_xor + kA * S1(N) + (rem > 0 ? pc_xor + kA * (N + 1) : 0)) mod 2^64
    state_b = (init_b + (fc_xor ^ kB) * S1(N) + kC * N + (rem > 0 ? (pc_xor ^ kB) * (N + 1) + kC : 0)) mod 2^64
    ```
- **Cross-worker fold (`fold_patterns_worker_checksums`)**: Starting from `(a, b) = (kFoldInitA, kFoldInitB)`, for `ordinal = 0 .. W - 1`:
  ```text
  a = splitmix64_mix(a xor state_a xor (exact_bytes_read + kA * (ordinal + 1)))
  b = splitmix64_mix(b xor state_b xor (pass_count       + kB * (ordinal + 1)))
  ```

### `splitmix64-coprime-fisher-yates-v1` and `random_indices_sha256`

For `random` worker `w` with `span_slots_w = span_bytes_w / 32` and `count_w = random_access_count`:
1. Initialize `state = splitmix64_mix(random_domain_seed ^ w)`.
2. If `count_w == span_slots_w`, `local_slots = [0 .. count_w - 1]`. Otherwise draw `start = splitmix64_bounded_draw(state, span_slots_w)` and `step = splitmix64_bounded_draw(state, span_slots_w - 1) + 1`, incrementing `step` wrapping in `[1, span_slots_w - 1]` until `gcd(step, span_slots_w) == 1`, then set `local_slots[k] = (k * step + start) % span_slots_w`.
3. Shuffle `local_slots` in-place via descending `fisher_yates_rejection_shuffle_with_state(local_slots, state)`, scale by `32` to relative byte offsets `indices[k]`, and store an ascending sorted copy `sorted_validation_indices[w]` (`<= 8 MiB` total).
4. Compute `random_indices_sha256` as SHA-256 over the little-endian `uint64_t` stream `(w, count_w, indices[0], ..., indices[count_w - 1])` for `w = 0 .. W - 1`.

### Parallel cold post-validation

After timer stop and worker join, `validate_patterns_write_buffer` and `validate_patterns_copy_buffer` run in parallel across `worker_spans` (when `worker_spans.size() > 1 && bytes >= 64 KiB`), verifying:
- **100% of touched bytes** against `write_pattern` (`write`) or `src` via `std::memcmp` (`copy`), AND
- **100% of untouched gap and tail bytes** against `pre_op_sentinel` whenever `has_untouched_dst_bytes == true` (using `sorted_validation_indices` for `random` and exact phase intervals for strided runs with `rem_phases < Q` or `<32B` worker tails).

## 7. Check and validation fault model and evidence limits

Every calibration attempt and measurement publishes `execution.validation.checks[]` with **six entries in fixed canonical order**:

| Index | `kind` | Applicable When | What It Verifies |
|---|---|---|---|
| 0 | `post-validation-structure` | Always (`read`, `write`, `copy`) | `created_workers == completed_workers == effective_workers`, no startup failure, `kernel_succeeded`, `src.mapped_bytes >= buffer_bytes`, `dst.mapped_bytes >= buffer_bytes`, and `random` index/sorted-copy plan consistency |
| 1 | `read-checksum` | `op == "read"` | Exact equality of all per-worker `(state_a, state_b, exact_bytes_read, pass_count)` and folded `run_checksum` against expected oracle |
| 2 | `write-final-touched-bytes` | `op == "write"` | Cold parallel 100% touched-byte equality `dst_touched == write_pattern` |
| 3 | `copy-final-touched-bytes` | `op == "copy"` | Cold parallel 100% touched-byte equality `dst_touched == src_touched` |
| 4 | `untouched-sentinel-isolation` | `op in ("write", "copy") && has_untouched_dst_bytes` | Cold parallel 100% untouched gap/tail equality `dst_untouched == pre_op_sentinel` |
| 5 | `src-readonly-protection` | `op in ("read", "copy")` | `src.read_only_protected == true && src.mprotect_errno == 0` |

Inapplicable checks serialize `applicable: false`, `evaluated: null`, `valid: null`, `reason_code: "not-applicable"`. Fault injection via `PatternsExecutorTestControl` (`fail_before_worker_index`, `set_worker_qos`, `force_kernel_fail`, `force_zero_elapsed`, `corrupt_dst_after_kernel`) exercises every failure mode in `tests/test_patterns_executor_runner.cpp`:

| Fault / Mutation Class | Observable Detection | Scope / Caveat |
|---|---|---|
| Single-bit or odd-parity read corruption, wrong slice/phase/stride, or wrong pass count | **Detected** (`read-checksum` fails; non-cancelling `(state_a, state_b)` catches repeated-pass cancellation) | **Even-parity single-pass XOR caveat**: two identical bit flips within one pass in a worker's read range can cancel in 64-bit XOR (`single_pass_xor_collision_caveat`). |
| Unattempted/truncated `write`/`copy` final pass or stray write outside sparse/random slots | **Detected** (`write-final-touched-bytes` / `copy-final-touched-bytes` / `untouched-sentinel-isolation` fail) | **Final-state-only caveat**: cold post-validation checks final `dst` state after `passes` iterations (`write_copy_post_validation_verifies_final_state_not_intermediate_passes`). |
| Mutation of `src` during `read`/`copy` | **Prevented/Detected** (`src` is `mprotect(PROT_READ)` before timing; `src-readonly-protection` checks metadata) | Assumes no external concurrent `mprotect` override inside the process. |
| Worker spawn or kernel failure, or zero/missing Mach delta | **Detected** (`worker-startup-failed`, `kernel-failed`, `invalid-elapsed-time`, clean join of partial team) | — |

## 8. Timing boundary, excluded calibration, freeze, rotation, and statistics

- **Mach `cpu_raw` timing boundary**: `HighResTimer::create()` queries `mach_timebase_info`. `start()` and `stop()` execute `asm volatile("dsb ish\n\tisb" ::: "memory")` before `mach_absolute_time()`, recording `{start_ticks, stop_ticks, delta_ticks, timebase_numer, timebase_denom}` under `execution.timing.cpu_raw`. Thread spawn, QoS, `mmap`/page faults, `src` fill, `mprotect`, `pre_op_sentinel` resets, oracle computation, thread join, cold post-validation, statistics, and JSON serialization occur strictly **outside** the timed interval.
- **Excluded calibration and atomic freeze (`patterns_runner.cpp`)**:
  - When `--iterations P` is explicit, every active `(pattern, op)` registers its operation plan at `P` passes, freezes `frozen_plan_refs`, and runs one `frozen_measurement_warmup` before loop `0`.
  - When `--iterations` is omitted, each active `(pattern, op)` calibrates independently (`8 MiB` min pilot, target `0.150 s`, window `[0.100 s, 0.250 s]`, max `2` corrections): running `calibration_shape_warmup`, `pilot`, up to 2 `correction`s (or `single_unit_confirmation_warmup` + `single_unit_confirmation` when scaled `passes == 1`). All `frozen_plan_refs` then freeze atomically and run one `frozen_measurement_warmup` per active `(pattern, op)` before loop `0`.
  - All 6 calibration/warmup `purpose` tokens (`calibration_shape_warmup`, `pilot`, `correction`, `single_unit_confirmation_warmup`, `single_unit_confirmation`, `frozen_measurement_warmup`) are recorded under `calibration.attempts` with `excluded_from_results: true`.
- **Two-axis cyclic Latin-square rotation & balance flags**:
  - For loop `i` (`0 .. loop_count - 1`), `p_order = build_cyclic_order(active_patterns.size(), i)` (`active_patterns[(i + p_pos) % active_patterns.size()]`) and `o_order = build_cyclic_order(3, i)` (`OPERATIONS[(i + o_pos) % 3]`).
  - `operation_order_balance_complete = results_complete && (loop_count % 3 == 0)`.
  - `pattern_order_balance_complete = results_complete && !active_patterns.empty() && (loop_count % active_patterns.size() == 0)` (note: with default `--count 3`, `operation_order_balance_complete` is `true`, whereas `pattern_order_balance_complete` is `false` because `3 % 7 != 0` and `3 % 6 != 0`).
- **Statistics & CV thresholds**: Only `measured` rows enter `aggregates.patterns.<kind>.<op>`. Headline `value` is `statistics.median` (P50 linear interpolation; `P50/P90/P95/P99`, arithmetic mean, sample `stddev` with `n - 1`, `CV%`, `MAD`, `min`, `max`), and `relative_to_forward_pct = 100.0 * agg.value / sequential_forward_value` per op. `repeatability_assessment_available = (n >= 2)`; `cv_warning = (n >= 3 && cv_defined && cv > cv_warning_threshold_pct)` where the threshold is `5.0%` for streaming (`sequential_forward`, `sequential_reverse`, `strided_64`) and `10.0%` for sparse (`strided_4096`, `strided_16384`, `strided_2mb`, `random`).

## 9. Run statuses, interruption semantics, and bounded loop checkpoints

- **Run statuses**: `not_started`, `complete`, `partial`, `interrupted`, `unsupported`, `failed`.
- **Measurement statuses**: `not_run`, `measured`, `interrupted`, `invalid`, `failed`.
- **Task-boundary completion-wins `SIGINT`/`SIGTERM`**: Signal state is checked before each task and after each task when remaining work exists (never inside the ARM64 kernel). A started task completes normally and stays `measured` if valid; remaining unstarted measurements become `interrupted` (`interruption-before-task`). A genuine failure (`row.status != measured`) overrides simultaneous interruption (`status = "failed"`).
- **Bounded loop snapshots (`bounded-loop-snapshots-v1`)**: For `JsonOutputKind::File`, `snapshot_interval_loops = K = max(1, ceil(loop_count / 8))`. Intermediate `progress` snapshots write atomically via `<target>.tmp` when `(loop_index + 1) % K == 0 && (loop_index + 1 < loop_count)`, followed by one `terminal` snapshot at completion/interruption/failure (abrupt termination can lose at most `3K` active-pattern-scaled attempts, i.e., up to `K * active_patterns.size() * 3` completed tasks). `checkpoint_lifecycle` publishes the 10-key contract (`observation_point: "before-current-snapshot-preparation"`, `current_persistence_success: null`). Any checkpoint write failure sets `checkpoint_failed = true`, `status = "failed"`, `reason_code = "checkpoint-write-failed"`, `run_accepted = false`, and exits `1` without retry.
- **Acceptance predicate**: `run_accepted = (status == "complete") && results_complete && !checkpoint_failed`.

## 10. Patterns JSON schema 1, build manifest, and sweep envelope

Top-level keys (26, in exact `nlohmann::ordered_json` order in `src/pattern_benchmark/patterns_json.cpp`):

```text
schema_version, mode, methodology_version, software, build_manifest,
configuration, resolved_plan, memory_budget, calibration, measurements,
aggregates, status, reason_code, results_complete, run_accepted,
interpretation, diagnostic, interruption_requested,
operation_order_balance_complete, pattern_order_balance_complete,
seeds, counters, checkpoint_lifecycle, loop_records, environment,
quality_warnings
```

- **Unavailable work-plan normalization (`!work_plan.valid`)**: When `build_patterns_work_plan` fails preflight (e.g., `memory-budget-exceeded`), `resolved_plan_json` normalizes all 7 `resolved_plan.patterns.<kind>` entries to `enabled: false`, `status: "unavailable"`, `reason_code: work_plan.reason_code`, `plan_identity: "patterns-work-plan-v1|invalid"`, and `operation_plans: []` so unavailable entries never masquerade as `active` or `buffer-too-small-for-stride-transition`.
- **Integer encoding & `build_manifest` v1**: Bounded counts (`loop_count`, `passes`, `plan_ref`, `measurement_id`, `effective_workers`, `timebase_numer`, `timebase_denom`) are JSON integers `<= 2^53 - 1`; 64-bit byte counts, seeds, checksum states, Mach ticks, `planned_passes`, `completed_passes`, and payload totals are canonical unsigned decimal strings. `build_manifest` v1 records `git_commit`, `git_dirty`, `compiler`, `compile_flags`, `link_flags`, `target_arch`, `sdk`, `min_os`, and `binary_sha256` (`status`: `"available"`, `"partial"`, or `"unavailable"`).
- **Schema-1 Sweep Envelope**: `--sweep` (`buffer-size`, `threads`) emits `configuration.mode == "sweep"`, `configuration.base_mode == "patterns"`, `configuration.sweep_schema_version == 1`, `configuration.sweep_parameters`, `planned_runs`, `attempted_runs`, `completed_runs`, `status`, `status_reason`, `conclusions_valid`, `execution_time_sec`, `timestamp`, `version`, and `runs[]` (each embedding a full schema-1 artifact with `configuration.output_file == ""` or `null` if aborted in preflight/preparation). `classify_pattern_completion()` in `src/benchmark/sweep_runner.cpp` requires `schema_version == 1`, `mode == "patterns"`, `status == "complete"`, `results_complete == true`, `run_accepted == true`, non-empty `measurements[]`, and all measurements `status == "measured"`. The first incomplete nested run stops further sweep runs.

## 11. Independent Python artifact verifier and mutation testing

`script-examples/verify_patterns_result.py` (sharing `script-examples/cpu_verify_common.py`, executed by `make test-patterns-verifier`, `make test-cpu-verifiers`, and `make test-all`) independently verifies direct schema-1 artifacts and schema-1 sweep envelopes using Python 3 standard-library arithmetic only:

```bash
python3 script-examples/verify_patterns_result.py <result.json|-> [--binary <path>] [--require-raw-timing] [--max-random-indices N]
```

- **Strict input & budget bounds**: Rejects inputs `> 64 MiB`, JSON depth `> 64`, `> 1,000,000` JSON nodes, duplicate keys, non-finite numbers (`NaN`/`Infinity`), `> 100,000` measurements, `> 4,096` operation plans, `selected_accesses_per_pass > --max-random-indices` (default `1,000,000`), or shared `oracle_work_used > 1,000,000`. Never executes `--binary` (hashes in 64 KiB chunks).
- **Verified invariants**: Mode/schema/methodology/6-key `component_identities`, 7-pattern geometry (`strided_2mb` skip vs active eligibility, 64B partitioning, strided worker reduction, random clamping/distribution), `SplitMix64` domain/op seeds, O(1) `patterns-buffer-pattern-v1` & `patterns-read-checksum-v1` (sequential, strided full/2D-partial, random), `splitmix64-coprime-fisher-yates-v1` & `random_indices_sha256`, unique odd `write_pattern`/`attempt_seed`/`pre_op_sentinel`, `memory_budget` arithmetic, ordered 6-entry `validation.checks[]`, Mach `cpu_raw` reconciliation (`--require-raw-timing`), `bandwidth_gb_s`, 2-axis Latin-square `loop_records` & balance flags, statistics/CV warnings, counters, `checkpoint_lifecycle`, `run_accepted`, `build_manifest`/`--binary` SHA-256 binding, and sweep Cartesian/invariant-config/prefix-stop rules.
- **Exit codes & JSON verdict**: Emits one sorted JSON verdict (`artifact_consistent`, `run_accepted`, `reason_code`, `checks.{checksum,timing,build}`, `oracle_work_used`, plus `artifact_kind: "patterns_sweep"` summary keys for sweeps). Exit `0` = consistent and accepted; exit `1` = inconsistent, unaccepted, invalid JSON, or `--max-random-indices < 1`; exit `2` = unsupported (`artifact_consistent: null` for schema/mode/methodology mismatch, missing raw timing with `--require-raw-timing`, or budget/size cap exceeded) or `argparse` error.
- **Mutation suite**: `tests/test_patterns_verifier.py` enforces **86 mutation/rejection assertions** across single-run tampering (`55` cases), `strided_2mb` skip-forgery (`2`), sweep envelope/nested mutations (`9` + `1` work-limit), null/invalid nested containers (`6`), execution-prefix/raw-timing checks (`2`), strict-JSON/CLI-cap/binary-binding rejections (`8`), and `resolved_sources`/`madvise`/duration-warning rejections (`3`), alongside O(1)-vs-brute-force differential checks and pinned fixture verification.

## 12. Console contract, quality warnings, comparison protocol, and change control

- **Console contract (`patterns_output.cpp`)**: Prints `Patterns Mode (schema v1)`, CPU core counts (`P`, `E`, `Logical`), buffer size in MiB and bytes, work policy (`explicit_fixed_work` or `automatic_calibration (target=0.150s)`), loops, requested workers (`user` or `auto`), base seed (`user` or `generated`), non-cacheable hint state, and per-pattern active/skipped status. The summary prints `--- Patterns Benchmark Results (<status>, <measured>/<planned> measured) ---` and per-pattern `Read`, `Write`, `Copy` lines (`P50=... GB/s (...% vs forward) (min=..., max=..., CV=..., MAD=..., n=...)`, `skipped (<reason>)`, or `unavailable (<reason>)`).
- **Quality warning tokens (emitted deduplicated on `stderr` and in `quality_warnings[]`)**:
  ```text
  environment-not-nominal
  main-thread-qos-not-applied
  worker-qos-not-applied
  madvise-random-not-applied
  <pattern>-<operation>-duration-<below-target-window|above-target-window|unavailable>   (when --iterations omitted)
  <pattern>-<operation>-high-cv                                                          (n >= 3 and CV > threshold)
  ```
- **Validation, comparison protocol, and change control**:
  - Correctness is gated by C++ tests (`tests/test_pattern_write_kernels.cpp`, `tests/test_patterns_config.cpp`, `tests/test_patterns_work_plan.cpp`, `tests/test_patterns_oracle.cpp`, `tests/test_patterns_executor_runner.cpp`, `tests/test_patterns_json_and_sweep.cpp`), Python verifier mutation tests (`tests/test_patterns_verifier.py`), and E2E acceptance (`tests/test_patterns_e2e_acceptance.py`).
  - Valid A/B comparisons require matched `methodology_version`, `component_identities`, frozen work plan, `worker_spans`, base seed, `--non-cacheable`, macOS/hardware, and thermal/power state; prefer `loop_count` divisible by `3 * active_patterns.size()` (`21` when all 7 patterns are active, `18` when `strided_2mb` is skipped) for complete two-axis position balance. Neither `build_manifest` nor `binary_sha256` constitutes signed runtime execution attestation.
  - Any change to the 11-option whitelist, 7-pattern geometry, 64B partitioning, strided worker reduction, payload formulas, seed/GF(2)/checksum/random-index derivations, ARM64 kernels, cold post-validation coverage, timing boundaries, calibration targets, or JSON fields/semantics requires methodology/schema review and synchronized updates across `src/pattern_benchmark/*`, `script-examples/verify_patterns_result.py`, and this whitepaper; removing/renaming fields or altering semantics requires a `schema_version` bump.

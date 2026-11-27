# CPU Memory Bandwidth Methodology

## Abstract

`memory_benchmark -B` (`--cpu-bandwidth`) is a versioned Apple Silicon user-space CPU memory bandwidth benchmark for macOS 26 or later. It measures sustained sequential `read`, `write`, and `copy` payload throughput across up to three active target classes—auto-detected L1 data cache, auto-detected L2 cache (or a single `--cache-size` custom target), and main memory (`--buffer-size`)—with selective target omission via `--buffer-size 0` or `--cache-size 0`.

Each timed measurement executes a synchronized worker team over disjoint, 64-byte-partitioned spans of persistent `mmap` buffers using AArch64 NEON assembly kernels (`memory_read*_loop_asm`, `memory_write*_pattern_loop_asm`, `memory_copy*_loop_asm`) bracketed by `dsb ish; isb` barriers and Mach `cpu_raw` tick boundaries. Read operations accumulate a non-cancelling per-pass checksum (`cpu-bw-read-checksum-v1`) over an O(1) GF(2) `src` buffer (`cpu-bw-buffer-pattern-v1`) protected with `mprotect(PROT_READ)`. Write and copy operations use unique odd 64-bit patterns and pre-copy sentinels followed by full-buffer cold byte-exact post-validation outside the timed interval. The reported metric is effective logical user-space payload bytes divided by elapsed Mach time (`GB/s = exact_payload_bytes / elapsed_seconds / 1e9`). It is not a physical DRAM hardware-counter measurement, not proof of cache or DRAM residency, and not directly comparable to Metal GPU bandwidth.

## Methodology identity

The current contract is identified by:

| Property | Value |
|---|---|
| Software version | Release provenance (`software.version`); not a schema or methodology selector |
| Mode | `cpu_bandwidth` |
| JSON schema | `1` |
| Methodology | `cpu-bandwidth-v1-arm64-calibrated-seeded-balanced` |
| Component identity | `cpu-bandwidth-components-v1` |
| Work-plan identity prefix | `cpu-bandwidth-work-plan-v1` |
| Buffer pattern | `cpu-bw-buffer-pattern-v1` |
| Read checksum | `cpu-bw-read-checksum-v1` |
| Run policy | `bounded-loop-snapshots-v1` |
| Canonical target order | `main_memory`, `l1`, `l2`, `custom` |
| Canonical operation order | `read`, `write`, `copy` |
| Copy payload multiplier | `2` (`read + write`, `2 * passes * target_bytes`) |
| Timing boundary | Synchronized worker start gate to last-worker `dsb ish` + Mach `HighResTimer::stop()` |
| Write/Copy post-validation | Parallel cold full-buffer check (`dst == write_pattern`, `dst == src`) |
| Calibration policy | Excluded auto-calibration toward `0.150 s` (`[0.100 s, 0.250 s]`), max `2` corrections |
| Sweep envelope schema | `1` (`mode: "sweep"`, `base_mode: "cpu_bandwidth"`, `sweep_schema_version: 1`) |

A cross-run comparison requires matched `methodology_version`, complete `component_identities`, frozen `plan_identity` and target/worker plans, base seed / explicit-vs-auto work policy, software version, hardware, thermal/power state, and output transport cadence.

## Scope and non-goals

`--cpu-bandwidth` models sustained sequential streaming bandwidth in user space across main memory and cache-sized working sets:

- **Effective payload accounting**:
  - `read`: `exact_payload_bytes = passes * target_bytes`
  - `write`: `exact_payload_bytes = passes * target_bytes`
  - `copy`: `exact_payload_bytes = 2 * passes * target_bytes` (counting both the logical read stream from `src` and the logical write stream to `dst`)
  - Rate: `bandwidth_gb_s = exact_payload_bytes / elapsed_seconds / 1e9` (decimal gigabytes per second, `10^9` bytes/s).
- **Physical non-claims**:
  - `interpretation.dram_residency` is `"unverified"` and `interpretation.cache_residency` is `"unverified"`. Sizing a buffer to L1, L2, or main memory does not prove which hardware cache, system-level cache (SLC), or DRAM controller serviced individual loads or stores.
  - Cache-line fills, write-allocate / read-for-ownership (RFO) traffic, dirty writebacks, hardware prefetcher speculation, TLB/page-table walks, and register checksum mixing are not added to the payload numerator, though their physical execution cost is reflected in elapsed time.
  - `-u` / `--non-cacheable` calls `madvise(..., MADV_RANDOM)` as a best-effort prefetch/cache-management hint (`non_cacheable_is_best_effort_madvise: true`). User-space macOS cannot configure page-table memory attributes or MAIR bits to create truly uncacheable DRAM mappings.
  - CPU streaming rates and Metal GPU bandwidth (`--gpu-bandwidth`) exercise different memory fabrics, store semantics, and timing domains and must not be pooled or directly equated.
  - Single-pass read XOR checksums and cold post-validation on write/copy have explicit fault boundaries documented in [Check and validation fault model and evidence limits](#7-check-and-validation-fault-model-and-evidence-limits).

## 1. CLI, transport boundary, and exit codes

The standalone primary mode is `-B` / `--cpu-bandwidth`, mutually exclusive with the six sibling modes in the seven-mode CLI (`-L`, `-P`, `-T`, `-C`, `-G`, `-M`). Its exact 12-option whitelist is:

```text
-B, --cpu-bandwidth           Run standalone audited CPU memory bandwidth benchmark
-b, --buffer-size <MB>        Main memory buffer size in MB (default: 512, 0 to omit)
-k, --cache-size <KB>         Custom cache buffer size in KB (16-1048576, 0 to omit caches)
-t, --threads <n>             Requested worker thread count (default: logical cores for main, 1 for cache)
-i, --iterations <n>          Fixed pass count per operation (1-1000000000; default: auto-calibrate)
-r, --count <n>               Number of benchmark loops (default: 3)
    --seed <uint64>           Explicit 64-bit base seed (0 allowed)
-u, --non-cacheable           Best-effort madvise(MADV_RANDOM) allocation hint
-o, --output <target>         Write schema-1 JSON to file, or '-' for stdout JSON
-S, --sweep <key=v1,v2,...>   Parameter sweep (keys: buffer-size, cache-size, threads)
-X, --sweep-max-runs <n>      Maximum generated sweep runs (default: 256)
-h, --help                    Display this help message and exit
```

### Parser and target activation rules

1. **Syntax & uniqueness**: Built on `src/core/config/cli_parser.{h,cpp}`. Long value-taking options accept both `--option value` and `--option=value` (including `--output=-`, `--output=`, and `--sweep=buffer-size=64,128`). Inline values on boolean/mode flags (`--cpu-bandwidth=1`, `--help=1`, `--non-cacheable=1`) and single-dash inline syntax (`-r=3`) are rejected. Every option except `-S` / `--sweep` may appear at most once; `-S` / `--sweep` is repeatable across distinct keys, and duplicate keys are rejected. Any option outside the 12-option whitelist (including `--latency-*`, `--patterns`, `--gpu-bandwidth`, `--llm-memory`, or removed legacy flags) is rejected with a single-line stderr diagnostic listing the allowed options.
2. **Numeric bounds**: All integer inputs must be strict unsigned decimal tokens (`[0-9]+`, no signs, whitespace, or prefixes):
   - `--buffer-size`: `0 .. 2^53 - 1` (`9007199254740991`), default `512` MiB.
   - `--cache-size`: `0` or `16 .. 1048576` KiB (`16 KiB .. 1 GiB`).
   - `--threads`: `1 .. 4294967295` (`UINT32_MAX`).
   - `--iterations`: `1 .. 1000000000` (`10^9`).
   - `--count`: `1 .. 2^53 - 1`, default `3`.
   - `--seed`: `0 .. 18446744073709551615` (`UINT64_MAX`); omitted generates one nonzero 64-bit seed per command.
   - `--sweep-max-runs`: `1 .. 2^53 - 1`, default `256`.
3. **Target activation and selective omission**:
   - `main_memory` is active iff `buffer_size_mb > 0`. `--buffer-size 0` omits `main_memory`.
   - When `-k` / `--cache-size` is omitted, auto-detected `l1` (`l1_data_cache_bytes > 0`) and `l2` (`l2_data_cache_bytes > 0`) are activated.
   - When `-k` is `16..1048576`, `custom` is active (`l1` and `l2` are omitted).
   - When `-k 0` is supplied, all cache targets (`l1`, `l2`, `custom`) are omitted.
   - At least one target must be active: combining `--buffer-size 0` and `--cache-size 0` is rejected.
   - Active targets appear in canonical order `("main_memory", "l1", "l2", "custom")` regardless of CLI flag order.
4. **Isolated help**: `-h` / `--help` runs after option parsing/range validation and exits `0` without OS queries, buffer allocation, or file creation.
5. **Sweep rules**: `-S` / `--sweep` accepts keys `buffer-size`, `cache-size`, and `threads`. It requires a non-empty `--output <target>`, forbids combining a swept key with its corresponding direct flag (`--buffer-size`, `--cache-size`, `--threads`), caps Cartesian product size at `--sweep-max-runs`, and preflight-validates **every** generated run configuration (active targets + work plan + memory budget + explicit iteration caps) before run 0 begins.

### Transport boundary and process exit codes

- **Empty/omitted `--output`**: Disables JSON serialization (console-only for direct runs; rejected for sweeps).
- **Exact `--output -`**: Reserves `stdout` exclusively for one final indented UTF-8 JSON document with trailing newline (`schema_version: 1` for direct runs, or schema-1 sweep envelope for `--sweep`) and routes all post-parse human output to `stderr`. No intermediate stdout checkpoints are built (`stdout_intermediate_checkpoints_are_lazy: true`).
- **Any other non-empty `--output <target>`**: Resolved relative to CWD (`./-` and flag-shaped names are treated as files). Direct runs write atomic `<target>.tmp` replacements every `K = max(1, ceil(loop_count / 8))` completed loops plus command terminal. Sweep runs atomically checkpoint the envelope after each attempted run.
- **Exit status vs `run_accepted`**: `EXIT_SUCCESS` (`0`) is returned for `-h`, a complete run, or a graceful `SIGINT`/`SIGTERM` interruption whose terminal checkpoint succeeded. `EXIT_FAILURE` (`1`) is returned on CLI/preflight rejection, work-plan/resource-preparation failure, timer/kernel/validation failure, or checkpoint persistence failure. **Consumers must never treat exit code `0` alone as a valid measurement**: authoritative acceptance requires `schema_version == 1`, `mode == "cpu_bandwidth"`, `status == "complete"`, `results_complete == true`, and `run_accepted == true` (plus `conclusions_valid == true` and `completed_runs == planned_runs` for sweeps).

## 2. Exact geometry, 64-byte partitioning, and guardrails

For each active target $t \in \{\text{main\_memory}, \text{l1}, \text{l2}, \text{custom}\}$:

```text
target_bytes(main_memory) = buffer_size_mb * 1,048,576
target_bytes(l1)          = l1_data_cache_bytes
target_bytes(l2)          = l2_data_cache_bytes
target_bytes(custom)      = custom_cache_size_kb * 1,024
```

### Worker selection and 64-byte slice partitioning

Let `available_workers` be the detected logical CPU core count (`>= 1`). For each active target:

```text
desired_workers   = user_specified_workers ? requested_workers
                    : (target == main_memory ? requested_workers : 1)
max_aligned       = max(1, target_bytes / 64)
effective_workers = min(desired_workers, available_workers, max_aligned)
```

- When `--threads` is omitted, `main_memory` uses all logical cores (`available_workers`) while `l1` and `l2` default to `1` thread (single-core cache bandwidth).
- When `--threads <n>` is explicitly given, `desired_workers = n` applies to **all** active targets (still capped by `available_workers` and `max(1, target_bytes / 64)`).

`partition_cpu_bandwidth_64b(total_bytes, workers)` divides `[0, total_bytes)` across `active = min(total_bytes, workers)` workers using proportional cuts snapped to 64-byte boundaries:

```text
boundaries[0] = 0
for rank = 1 .. active - 1:
  target = (total_bytes / active) * rank + min(rank, total_bytes % active)
  lo = boundaries[rank - 1] + 1
  hi = total_bytes - (active - rank)
  target = clamp(target, lo, hi)
  aligned_lo = ceil(lo / 64) * 64
  aligned_hi = floor(hi / 64) * 64
  if aligned_lo <= aligned_hi:
    down = clamp(floor(target / 64) * 64, aligned_lo, aligned_hi)
    up   = (down < target) ? min(aligned_hi, down + 64) : down
    target = (abs(target - down) <= abs(target - up)) ? down : up
  boundaries[rank] = target
boundaries[active] = total_bytes
```

Every interior cut `boundaries[rank]` (`rank > 0` when `aligned_lo <= aligned_hi`) is an exact multiple of 64 bytes (hence 8-byte aligned for 64-bit word reads), and ties break downward (`down`). Any remaining workers beyond `active` receive `{0, 0}` spans.

### Guardrails

| Guardrail | Exact Limit | Where Enforced |
|---|---|---|
| `kCpuBandwidthMaxPasses` | `1,000,000,000` (`10^9`) | CLI `--iterations`, `build_cpu_bandwidth_operation_plan`, verifier |
| `kCpuBandwidthMaxExactPayloadBytes` | `68,719,476,736` (`64 GiB`) per operation task | `build_cpu_bandwidth_operation_plan`, calibration cap, verifier |
| `kCpuBandwidthMaxMeasurements` | `100,000` (`loop_count * active_targets.size() * 3`) | `build_cpu_bandwidth_work_plan`, verifier |
| `LLM_JSON_MAX_SAFE_INTEGER` | `9,007,199,254,740,991` (`2^53 - 1`) | Numeric CLI options, sweep counts |
| Verifier `operation_plans` cap | `4,096` entries | `verify_cpu_bandwidth_result.py` |
| `kCpuBandwidthCvWarningPct` | `5.0%` (`CV > 5.0` with `n >= 3`) | `aggregates` quality warning |

For a given `(target_bytes, op)` pair, the maximum admissible pass count is:

```text
bytes_per_pass = target_bytes * (op == copy ? 2 : 1)
max_passes     = min(1_000_000_000, floor(68_719_476_736 / bytes_per_pass))
```

## 3. Golden mathematical test vectors

Every value below was computed directly from `script-examples/verify_cpu_bandwidth_result.py` and verified bit-for-bit against live `./memory_benchmark -B` output and `tests/test_cpu_bandwidth_oracle.cpp`.

### 3.1 Domain seeds and fixture-matched 2-worker vector (`base_seed = 424242`)

For `base_seed = 424242` (`./memory_benchmark -B -b 1 -k 16 -t 2 -i 2 -r 3 --seed 424242`):

| Quantity | Formula | Exact Unsigned Decimal |
|---|---|---:|
| `buffer_domain_seeds.main_memory` | `splitmix64_mix(424242 ^ 0x4342575F4D41494E)` | `12853032350930100478` |
| `buffer_domain_seeds.l1` | `splitmix64_mix(424242 ^ 0x4342575F4C315F5F)` | `8052984286948143865` |
| `buffer_domain_seeds.l2` | `splitmix64_mix(424242 ^ 0x4342575F4C325F5F)` | `10238245925842745822` |
| `buffer_domain_seeds.custom` | `splitmix64_mix(424242 ^ 0x4342575F43555354)` | `5364175457813105886` |
| `pre_copy_sentinel_uint64_decimal` | `splitmix64_mix(424242 ^ 0x4342575F434F5059)` | `5953240118362669119` |
| `pre_copy_sentinel(copy_ordinal=1)` | `splitmix64_mix(5953240118362669119 ^ 1) \| 1` | `5703989422672563035` (`0x4F28A428ACFBCB5B`) |

For `main_memory` (`1,048,576 B`, `effective_workers = 2`), `partition_cpu_bandwidth_64b(1048576, 2)` yields spans `[(0, 524288), (524288, 524288)]`. The first two pattern-table words and buffer words are:

```text
T[0] = 0xB9108E72184E9D70   W[0] = 0xB9108E72184E9D70
T[1] = 0x49C9E9E58C7CA363   W[1] = 0x49C9EBE58C74A362
```

For `read` with `passes = 2` (`op_index = 0`):

- `op_seed = splitmix64_mix(12853032350930100478 ^ (1 << 56) ^ 2) = 5431363063597238049`
- Note that `524,288` bytes equals `65,536` 64-bit words (`256 * 256` words, an even multiple of 256 and a multiple of 4), so `pass_xor` evaluates to `0` on both workers, yet the non-cancelling `(state_a, state_b)` accumulators still evolve distinctly per worker and pass:
  - Worker `0`: `state_a = 1363154657124793970`, `state_b = 765718975942869604`, `exact_bytes_read = 1048576`, `pass_count = 2`
  - Worker `1`: `state_a = 7879840563061088676`, `state_b = 8349491997236135861`, `exact_bytes_read = 1048576`, `pass_count = 2`
  - Folded `expected_run_checksum`: `state_a = 9997130236859432985`, `state_b = 5602982515975408360`
- For `write` with `passes = 2` (`op_index = 1`):
  - `op_seed = 15770748730058986844`
  - `attempt_ordinal = 1`: `attempt_seed = 7587250918863781903`, `write_pattern = 181384587478128971` (`0x02846852405D594B`)

### 3.2 Non-zero `pass_xor`, `<32 B` scalar tail, and 64B partitioning goldens

Using the same `main_memory` seed (`12853032350930100478`):

1. **Non-256-multiple vector span (`offset = 0`, `span = 2048 B`, `passes = 3`, `workers = 2`)**:
   - `partition_cpu_bandwidth_64b(2048, 2) = [(0, 1024), (1024, 1024)]`
   - `op_seed = 11247853501389303978`
   - `pass_xor(0, 2048) = 10719348703578661018` (`0x94C2C6DA3459A09A`)
   - Worker `0`: `state_a = 14209785145809115956`, `state_b = 10556315701979402099`, `exact_bytes_read = 3072`
   - Worker `1`: `state_a = 10830142123713516723`, `state_b = 6336039750821930611`, `exact_bytes_read = 3072`
   - Folded `expected_run_checksum`: `state_a = 16079714545819116324`, `state_b = 1790011886326754563`
2. **Vector + `<32 B` byte-tail span (`offset = 0`, `span = 613 B`)**:
   - Vector range `[0, 608)` (`613 - (613 % 32) = 608 B = 76` words) plus 5-byte scalar tail `[608, 613)` folded into bits `[0..7]`:
   - `pass_xor(0, 613) = 2753644117688965028` (`0x2636E8BC50B1D7A4`)
3. **Unaligned-total 64B partition (`total_bytes = 1000`, `workers = 3`)**:
   - `partition_cpu_bandwidth_64b(1000, 3) = [(0, 320), (320, 320), (640, 360)]`

## 4. Physical memory mappings, strict memory budget, and resource evidence

Before allocating any buffers, `build_cpu_bandwidth_work_plan` evaluates strict memory admission against `available_memory_bytes`:

```text
if available_memory_bytes == 0:
  used_fallback        = true
  allowed_memory_bytes = 2048 * 1,048,576   // 2 GiB fallback
else:
  used_fallback        = false
  allowed_memory_bytes = floor(available_memory_bytes * 0.80)
admitted_budget_bytes  = allowed_memory_bytes
```

Every active target owns two persistent `MAP_PRIVATE | MAP_ANONYMOUS` mappings (`src` and `dst`):

```text
requested_data_bytes    = sum_{t in active} (2 * target_bytes(t))
committed_data_bytes    = sum_{t in active} (2 * round_up(target_bytes(t), page_size_bytes))
resource_rounding_bytes = committed_data_bytes - requested_data_bytes
setup_peak_bytes        = 4096 + max_effective_workers * 64
```

When JSON serialization is active (`--output` non-empty or `--sweep` collection), a conservative JSON DOM reserve (`json_peak_estimate.total_bytes`) is added before admission:

- `fixed_schema_bytes`: `2 MiB` (`2,097,152 B`)
- `input_string_bytes`: `16 * (output_file.size() + sum(argv[i].size()))`
- `measurement_record_bytes`: `64 KiB * (planned_measurements + total_calibration_attempts + total_op_plans)` where `calibration_attempts_per_op` is `1` for explicit `--iterations` or `4 + 2 = 6` for auto-calibration
- `worker_checksum_bytes`: `64 KiB * sum_{t in active} (effective_workers(t) * (loop_count + calibration_attempts_per_op + 1))`

For console-only direct runs (`output_file == ""` and `!collect_json_for_sweep`), `json_peak_estimate.total_bytes = 0`. Total peak admission is:

```text
transient_peak_bytes   = planned_measurements * 512 + 4096 + json_peak_estimate.total_bytes
runtime_peak_bytes     = committed_data_bytes + transient_peak_bytes
known_owned_peak_bytes = runtime_peak_bytes
admitted               = (known_owned_peak_bytes <= admitted_budget_bytes)
```

Workloads exceeding `admitted_budget_bytes` are rejected (`memory-budget-exceeded`) rather than silently downsized.

### Buffer lifecycle and `AllocatedBufferEvidence`

Each `src` and `dst` buffer is allocated via `allocate_buffer_with_evidence()`, recording `requested_bytes`, `mapped_bytes`, `non_cacheable_requested`, `madvise_advice` (`MADV_RANDOM` if `-u`, else `MADV_WILLNEED`), `madvise_applied`, `madvise_errno`, `read_only_protected`, and `mprotect_errno` under `resolved_plan.resources.<target>.{src,dst}`:

1. `src` is populated in parallel across `worker_spans` with `cpu-bw-buffer-pattern-v1`.
2. `src` is protected read-only via `mprotect(ptr, protect_size, PROT_READ)` where `protect_size = mapped_bytes > 0 ? mapped_bytes : target_bytes`. Failure aborts preparation (`src-readonly-failed`).
3. `dst` is pre-populated in parallel with `pre_copy_sentinel(copy_ordinal = 0)` using the target-appropriate pattern-write kernel so every page of `dst` is resident and faulted before timing begins.
4. Before every `copy` attempt (calibration or measured), `dst` is reset outside the timer with a fresh odd `pre_copy_sentinel` (`copy_ordinal = 1, 2, ...`).

## 5. ARM64 kernels, traversal, and synchronized execution gate

### Kernel dispatch matrix

| Target class | `read` kernel | `write` kernel | `copy` kernel |
|---|---|---|---|
| `main_memory` | `memory_read_loop_asm` | `memory_write_pattern_loop_asm` | `memory_copy_loop_asm` |
| `l1`, `l2`, `custom` | `memory_read_cache_loop_asm` | `memory_write_cache_pattern_loop_asm` | `memory_copy_cache_loop_asm` |

All six assembly kernels follow AAPCS64 (preserving callee-saved `x19–x28`, `v8–v15`), emit no internal memory barriers (relying on the caller's `dsb ish; isb` timer contract), and align their 512-byte main loops to 64 bytes (`.p2align 6`) using count-down `subs + b.ne`:

- **Read kernels (`memory_read_loop_asm`, `memory_read_cache_loop_asm`)**: Process 512B/iter via 16 `ldp q*, q*, [x6, #imm]` pairs XORed into four 128-bit accumulators `v0–v3`, followed by `tbz`-dispatched 256/128/64/32B vector tail chunks. Any `<32B` remainder is loaded byte-by-byte (`ldrb w13, [x6], #1`) and XORed into `x12` (bits `[0..7]`). Finally `v0..v3` fold into two 64-bit lanes XORed together with `x12` into `x0`.
- **Pattern-write kernels (`memory_write_pattern_loop_asm`, `memory_write_cache_pattern_loop_asm`)**: Prototype `(void* dst, size_t byteCount, uint64_t pattern64)`. Broadcast `dup v0.2d, x2` once before the loop. The main-memory kernel writes 512B/iter and 256/128/64/32B tails using non-temporal pair stores (`stnp q*, q*, [...]`), whereas the cache kernel uses ordinary temporal pair stores (`stp q0, q0, [...]`). Both share an exact little-endian `16/8/4/2/1` scalar tail (`str q0`, `str x2`, `str w2; lsr x2, x2, #32`, `strh w2; lsr x2, x2, #16`, `strb w2`) so every byte at offset `i` equals `(pattern64 >> ((i & 7) * 8)) & 0xFF`. (Existing `memory_write.s` and `memory_write_cache.s` remain untouched for `--patterns`.)
- **Copy kernels (`memory_copy_loop_asm`, `memory_copy_cache_loop_asm`)**: Process 512B/iter (`ldp` + `stnp`), followed by 256/128/64/32B `tbz` tiers (`stnp`) and 16/8/4/2/1 scalar `ldr`/`str` tiers.

### Synchronized worker execution gate

Every calibration attempt and measured operation executes inside `execute_cpu_bandwidth_operation`:

1. Spawn `worker_count = effective_workers` threads. Each worker calls `pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)` (best-effort; QoS failure counts `qos_failed_workers` without aborting execution) and increments `ready_workers` under `state_mutex`, waiting on `state_cv` for `start_workers || cancel_workers`. If spawning fails mid-team, `cancel_workers` is set, created threads are joined cleanly, and `worker-startup-failed` is recorded.
2. Once `ready_workers == worker_count`, the coordinator calls `timer.start()` (`dsb ish; isb`, then `mach_absolute_time()`), sets `start_workers = true`, and broadcasts `state_cv`.
3. Each worker loops `p = 1 .. passes` over its disjoint slice `[offset_bytes, offset_bytes + span_bytes)`:
   - For `read`: computes `pass_xor = kernel(src + offset, span)` and updates `(state_a, state_b)` per pass.
   - For `write`: calls the pattern-write kernel with `write_pattern` `passes` times.
   - For `copy`: calls the copy kernel `passes` times.
4. Each worker finishes with `asm volatile("dsb ish" ::: "memory")`. The last worker (`remaining_workers.fetch_sub(1, acq_rel) == 1`) calls `timer.stop()` (`dsb ish; isb`, then `mach_absolute_time()`), captures `timer.last_snapshot` (`cpu_raw`), and wakes the coordinator.
5. The coordinator joins all worker threads and performs cold post-validation outside the timed interval.

## 6. Deterministic seeds, O(1) GF(2) pattern, read oracle, and cold post-validation

### Seed hierarchy

All PRNG mixing uses `splitmix64_mix(x)`:

```text
z = (x + 0x9E3779B97F4A7C15) mod 2^64
z = ((z xor (z >> 30)) * 0xBF58476D1CE4E5B9) mod 2^64
z = ((z xor (z >> 27)) * 0x94D049BB133111EB) mod 2^64
return z xor (z >> 31)
```

With fixed 64-bit domain constants:

- `main_memory`: `0x4342575F4D41494E` (`"CBW_MAIN"`)
- `l1`: `0x4342575F4C315F5F` (`"CBW_L1__"`)
- `l2`: `0x4342575F4C325F5F` (`"CBW_L2__"`)
- `custom`: `0x4342575F43555354` (`"CBW_CUST"`)
- `pre_copy_sentinel_domain`: `0x4342575F434F5059` (`"CBW_COPY"`)

Derivations (all stored as unsigned decimal strings in JSON):

```text
target_src_seed        = splitmix64_mix(base_seed xor target_domain_id)
pre_copy_sentinel_seed = splitmix64_mix(base_seed xor 0x4342575F434F5059)
op_seed                = splitmix64_mix(target_src_seed xor ((op_index + 1) << 56) xor passes)
                         where op_index = {read: 0, write: 1, copy: 2}
write_attempt_seed     = splitmix64_mix(op_seed xor (0x5752495400000000 | attempt_ordinal))
write_pattern          = splitmix64_mix(write_attempt_seed) | 1
pre_copy_sentinel      = splitmix64_mix(pre_copy_sentinel_seed ^ copy_ordinal) | 1
```

Each `write` attempt increments `write_attempt_ordinal` until `write_pattern` is nonzero, odd, and unseen in `seen_write_patterns` across all prior calibration and measurement write attempts in the command, guaranteeing that every write attempt writes a pattern distinct from every prior buffer state.

### O(1) GF(2) `src` buffer (`cpu-bw-buffer-pattern-v1`)

For a target with seed `target_src_seed`, precompute the 256-word lookup table `T[0..255]` and prefix XOR table `P_T[0..256]`:

```text
T[k]     = splitmix64_mix(target_src_seed xor k),  k = 0 .. 255
P_T[0]   = 0
P_T[k+1] = P_T[k] xor T[k]
```

The 64-bit little-endian word at word index `i` (covering byte offsets `[8i, 8i + 8)`) is:

```text
W[i] = T[i & 255] xor (i xor rotl64(i, 19) xor rotl64(i, 41))
```

Any trailing 1–7 bytes at the end of a non-8-byte-aligned buffer take the low little-endian bytes of `W[floor(bytes / 8)]`. Because XOR and bitwise rotation distribute linearly over GF(2), the XOR of words `[start_word, end_word)` is computed in O(1) time without scanning the buffer:

```text
prefix_t(n) = (((n >> 8) & 1) ? P_T[256] : 0) xor P_T[n & 255]
prefix_xor_0_to_n_minus_1(n) = 0 if n == 0, else switch ((n - 1) & 3):
  0 -> n - 1,  1 -> 1,  2 -> n,  3 -> 0
```

For any worker span `[offset_bytes, offset_bytes + span_bytes)` with `offset_bytes % 8 == 0` and `vec_bytes = span_bytes - (span_bytes % 32)`:

```text
start_word = offset_bytes >> 3
end_word   = start_word + (vec_bytes >> 3)
range_t    = prefix_t(end_word) xor prefix_t(start_word)
x          = prefix_xor_0_to_n_minus_1(end_word) xor prefix_xor_0_to_n_minus_1(start_word)
vec_xor    = range_t xor (x xor rotl64(x, 19) xor rotl64(x, 41))
tail_xor   = XOR_{pos = offset_bytes + vec_bytes}^{offset_bytes + span_bytes - 1}
               ((W[pos >> 3] >> (8 * (pos & 7))) & 0xFF)
pass_xor   = vec_xor xor tail_xor
```

### Non-cancelling per-pass read checksum (`cpu-bw-read-checksum-v1`)

Fixed constants: `kA = 0x9E3779B97F4A7C15`, `kB = 0xBF58476D1CE4E5B9`, `kC = 0x94D049BB133111EB`.
For worker `w`:

```text
init_a(w) = splitmix64_mix(op_seed xor (0xA100000000000000 | w))
init_b(w) = splitmix64_mix(op_seed xor (0xB200000000000000 | w))
```

During the timed loop `p = 1 .. P` (where `P = passes`):

```text
state_a = (state_a + pass_xor + kA * p) mod 2^64
state_b = (state_b + (pass_xor xor kB) * p + kC) mod 2^64
```

Because `src` is `PROT_READ` and immutable, `pass_xor` is invariant across passes `1 .. P`, yielding the O(1) closed-form oracle (using even-half decomposition for `S1` so `P <= 10^9` never overflows intermediate 64-bit arithmetic):

```text
S1      = ((P % 2 == 0) ? (P / 2) * (P + 1) : P * ((P + 1) / 2)) mod 2^64
state_a = (init_a(w) + P * pass_xor + kA * S1) mod 2^64
state_b = (init_b(w) + (pass_xor xor kB) * S1 + kC * P) mod 2^64
```

Finally, `fold_cpu_bandwidth_worker_checksums` folds all `W` worker tuples `(state_a, state_b, exact_bytes_read, pass_count)` in worker order `ordinal = 0 .. W - 1`, starting from `(a = 0x6A09E667F3BCC909, b = 0xBB67AE8584CAA73B)`:

```text
tag_a = (exact_bytes_read + kA * (ordinal + 1)) mod 2^64
tag_b = (pass_count       + kB * (ordinal + 1)) mod 2^64
a = splitmix64_mix(a xor state_a xor tag_a)
b = splitmix64_mix(b xor state_b xor tag_b)
```

### Cold parallel write and copy post-validation

After timer stop and worker join, `validate_cpu_bandwidth_write_buffer` checks **every byte** of `dst[0..bytes)` in parallel across `worker_spans` against the little-endian repeating `write_pattern`, and `validate_cpu_bandwidth_copy_buffer` checks **every byte** of `dst[0..bytes) == src[0..bytes)` in parallel using `std::memcmp`. Because `dst` was seeded with an odd `pre_copy_sentinel` before `copy` and `write_pattern` is odd and unique across all prior writes, any unexecuted pass or stale untouched byte region fails cold validation.

## 7. Check and validation fault model and evidence limits

Every calibration attempt and measurement records `execution.validation.checks[]` with **five entries in fixed canonical order**:

| Index | `kind` | Applicable When | What It Verifies |
|---|---|---|---|
| 0 | `post-validation-structure` | Always (`read`, `write`, `copy`) | `created_workers == completed_workers == effective_workers`, no startup failure, `kernel_succeeded`, `src.mapped_bytes >= target_bytes`, `dst.mapped_bytes >= target_bytes` |
| 1 | `read-checksum` | `op == "read"` | Exact equality of all per-worker `(state_a, state_b, exact_bytes_read, pass_count)` and folded `run_checksum` against expected oracle |
| 2 | `write-final-full-buffer` | `op == "write"` | Cold full-buffer byte-exact equality `dst[0..bytes) == write_pattern` |
| 3 | `copy-final-full-buffer` | `op == "copy"` | Cold full-buffer byte-exact equality `dst[0..bytes) == src[0..bytes)` |
| 4 | `src-readonly-protection` | `op in ("read", "copy")` | `src.read_only_protected == true && src.mprotect_errno == 0` |

Inapplicable checks serialize `applicable: false`, `evaluated: null`, `valid: null`, `reason_code: "not-applicable"`.

### Fault-model matrix and caveats

| Fault / Mutation Class | Observable Detection | Scope / Caveat |
|---|---|---|
| Single-bit or odd-parity read corruption, wrong slice offset, or wrong pass count (`P != expected`) | **Detected** (`read-checksum` fails; non-cancelling `(state_a, state_b)` catches repeated-pass cancellation) | **Even-parity single-pass XOR caveat**: within one pass, two identical bit flips at the same bit position inside a worker's vector range (or inside its `<32B` tail) cancel in `pass_xor` (`single_pass_xor_collision_caveat`). |
| Unattempted or truncated `write` / `copy` final pass, or single-bit corruption in `dst` | **Detected** (`write-final-full-buffer` / `copy-final-full-buffer` fails via cold full-buffer scan, backed by `tests/test_cpu_bandwidth_oracle.cpp` and `tests/test_cpu_bandwidth_executor_runner.cpp`) | **Final-state-only caveat**: cold post-validation inspects the final buffer state after `passes` iterations, not intermediate passes (`write_copy_post_validation_verifies_final_state_not_intermediate_passes`). A transient corruption overwritten by a later pass is unobservable. |
| Accidental mutation of `src` during timed `read` or `copy` | **Prevented/Detected** (`src` is `mprotect(PROT_READ)` before timing; `src-readonly-protection` verifies status) | OS page protection is verified at preparation/metadata level; kernel/process-wide `mprotect` invariants assume no external concurrent `mprotect` override. |
| Worker thread creation failure or kernel error | **Detected** (`worker-startup-failed` / `kernel-failed` on `post-validation-structure`, zero-leak join of partial team) | — |
| Non-positive or missing Mach elapsed time | **Detected** (`invalid-elapsed-time`, raw timing snapshot absent or zero delta rejected) | — |

## 8. Timing boundary, calibration, cyclic balance, and statistics

- **Timing boundary (`cpu_raw`)**: `HighResTimer::create()` queries `mach_timebase_info`. `start()` and `stop()` execute `asm volatile("dsb ish\n\tisb" ::: "memory")` before `mach_absolute_time()`, recording `{start_ticks, stop_ticks, delta_ticks, timebase_numer, timebase_denom}` under `execution.timing.cpu_raw`. Worker creation, QoS calls, `mmap`/page faults, `src` initialization, `mprotect`, `pre_copy_sentinel` resets, expected oracle computation, thread joins, cold post-validation, statistics, and JSON serialization are strictly **outside** the timed interval. Normal cache state is not flushed between tasks, so runs are cache-inclusive.
- **Excluded calibration and atomic freeze**:
  - When `--iterations P` is explicit, every `(target, op)` operation plan is registered at `P` passes, frozen immediately in `frozen_plan_refs`, and warmed once (`frozen_measurement_warmup`) in canonical `(target, op)` order before loop `0`.
  - When `--iterations` is omitted, each active `(target, op)` calibrates independently in canonical order:
    1. Choose minimum pilot pass count covering at least `8 MiB` (`BANDWIDTH_CALIBRATION_MIN_PILOT_BYTES`), bounded by `[1, max_passes]`.
    2. Run `calibration_shape_warmup` (excluded), then `pilot` (excluded).
    3. If `elapsed` is outside `[0.100 s, 0.250 s]`, scale toward `0.150 s` (`calculate_duration_scaled_count`) for at most `2` `correction` attempts; if scaling reaches `passes == 1`, run `single_unit_confirmation_warmup` once (if not yet warmed at `1`) followed by `single_unit_confirmation`.
    4. Freeze all resolved `frozen_plan_refs` atomically and execute one `frozen_measurement_warmup` per `(target, op)` before timed loop `0`.
  - All calibration and warmup attempts are recorded under `calibration.attempts` with `excluded_from_results: true`.
- **Duration quality classification**: Measured durations are classified as `within-target-window` (`[0.100, 0.250]`), `below-target-window`, `above-target-window`, or `unavailable`.
- **Cyclic Latin-square schedule**: For loop `i` (`0 .. loop_count - 1`), targets execute in order `active_targets[(i + t_pos) % num_targets]` and operations inside each target execute in order `OPERATIONS[(i + o_pos) % 3]`.
  - `operation_order_balance_complete = results_complete && (loop_count % 3 == 0)`.
  - `target_order_balance_complete = results_complete && num_targets > 0 && (loop_count % num_targets == 0)`.
  - With the default `loop_count = 3`, operation order is always position-balanced across complete runs.
- **Statistics**: Only `measured` rows enter `aggregates.targets.<target>.<op>`. The headline `value` is `statistics.median` (P50 via linear interpolation, `P50/P90/P95/P99`, arithmetic mean, sample `stddev` with `n - 1` denominator, `CV% = |stddev / mean| * 100`, `MAD`, `min`, `max`). `repeatability_assessment_available = (n >= 2)`; `cv_warning = (n >= 3 && cv_defined && cv > 5.0)`.

## 9. Run status, interruption semantics, and bounded loop checkpoints

- **Run statuses**: `not_started`, `complete`, `partial`, `interrupted`, `unsupported`, `failed`.
- **Measurement statuses**: `not_run`, `measured`, `interrupted`, `invalid`, `failed`.
- **Task-boundary completion-wins**: `SIGINT`/`SIGTERM` is checked between tasks (never inside the ARM64 kernel). A started task completes normally; if valid, it stays `measured`, and all remaining unstarted measurements become `interrupted` (`interruption-before-task`). A genuine task/checksum/validation/checkpoint failure overrides simultaneous interruption (`failed`).
- **Bounded loop checkpoints**: File targets write an atomic `.tmp` replacement every `K = max(1, ceil(loop_count / 8))` completed loops plus command terminal (at most 8 intermediate snapshots; at abrupt termination at most `3K` completed tasks may lack persistence). `checkpoint_lifecycle` publishes the 10-key contract with `observation_point: "before-current-snapshot-preparation"` and `current_persistence_success: null`. Any file checkpoint write failure is terminal (`checkpoint-write-failed`, `run_accepted: false`) and is not retried.
- **Acceptance predicate**: `run_accepted = (status == "complete") && results_complete && !checkpoint_failed`.

## 10. CPU Bandwidth JSON schema 1 and build manifest

Top-level keys (26, in canonical `nlohmann::ordered_json` order):

```text
schema_version, mode, methodology_version, software, build_manifest,
configuration, resolved_plan, memory_budget, calibration, measurements,
aggregates, status, reason_code, results_complete, run_accepted,
interpretation, diagnostic, interruption_requested,
operation_order_balance_complete, target_order_balance_complete,
seeds, counters, checkpoint_lifecycle, loop_records, environment,
quality_warnings
```

- **Integer encoding**: Bounded small counts (`loop_count`, `passes`, `plan_ref`, `measurement_id`, `effective_workers`, `timebase_numer`, `timebase_denom`) are JSON integers `<= 2^53 - 1`; byte counts, 64-bit seeds, checksum states, Mach ticks, `planned_passes`, `completed_passes`, and payload totals are canonical unsigned decimal strings (`0` or `[1-9][0-9]*`).
- **`build_manifest` v1**: Captures `git_commit`, `git_dirty`, `compiler`, `compile_flags`, `link_flags`, `target_arch`, `sdk`, `min_os`, and once-per-command `binary_sha256` (`status` `"available"`, `"partial"`, or `"unavailable"`).
- **Schema-1 Sweep Envelope**: `--sweep` emits `configuration.mode == "sweep"`, `configuration.base_mode == "cpu_bandwidth"`, `configuration.sweep_schema_version == 1`, `configuration.sweep_parameters` (`buffer-size`, `cache-size`, `threads`), `planned_runs`, `attempted_runs`, `completed_runs`, `status`, `status_reason`, `conclusions_valid`, `execution_time_sec`, `timestamp`, `version`, and `runs[]` (each embedding a full schema-1 artifact with `configuration.output_file == ""` or `null` if aborted during preparation). The first incomplete nested run stops further sweep runs.

## 11. Independent Python artifact verifier

`script-examples/verify_cpu_bandwidth_result.py` (using `script-examples/cpu_verify_common.py`, exercised by `make test-cpu-verifiers` and `make test-all`) independently verifies both direct schema-1 artifacts and schema-1 sweep envelopes with zero third-party dependencies:

```bash
python3 script-examples/verify_cpu_bandwidth_result.py <result.json> [--binary <path>] [--require-raw-timing]
```

- **Strict input & budget bounds**: Rejects files `> 64 MiB`, JSON depth `> 64`, `> 1,000,000` JSON nodes, duplicate keys, non-finite numbers (`NaN`/`Infinity`), `> 100,000` measurements, `> 4,096` operation plans, or shared `oracle_work_used > 1,000,000`. Never executes `--binary` (streams SHA-256 in 64 KiB chunks).
- **Verified invariants**: Mode/schema/methodology/component identities, target geometry & 64B worker span partitioning, SplitMix64 domain seeds, O(1) GF(2) `pass_xor` & `cpu-bw-read-checksum-v1` worker/run fold oracle, unique odd `write_pattern` & `attempt_seed` checks, `memory_budget` arithmetic, ordered 5-entry `validation.checks[]`, Mach `cpu_raw` delta/timebase/elapsed reconciliation (`--require-raw-timing`), `bandwidth_gb_s` calculation, Latin-square `loop_records`, descriptive statistics & CV warnings, counters, `checkpoint_lifecycle`, `run_accepted`, `build_manifest` + `--binary` SHA-256 binding, and sweep Cartesian/invariant-config/prefix-stop rules.
- **Output & exit codes**: Emits one sorted JSON verdict (`artifact_consistent`, `run_accepted`, `reason_code`, `checks.{checksum,timing,build}`, `oracle_work_used`, plus sweep summary keys). Exit `0` = consistent and accepted; `1` = inconsistent, unaccepted, or invalid JSON; `2` = unsupported (`artifact_consistent: null` for schema/mode/methodology mismatch, missing raw timing with `--require-raw-timing`, or budget/size cap exceeded) or `argparse` error.

## 12. Console contract and quality warnings

Console configuration prints CPU core counts (`P`, `E`, `Logical`), work policy (`explicit_fixed_work` or `automatic_calibration (target=0.150s)`), loops, requested workers (`user` or `auto`), base seed, non-cacheable hint state, and per-target byte size + `effective_workers`. The summary prints per-target `Read`, `Write`, `Copy` medians (`P50=... GB/s (min=..., max=..., CV=...%, MAD=..., n=...)`) or `unavailable`, followed on `stderr` by any deduplicated `quality_warnings[]` tokens:

```text
environment-not-nominal
main-thread-qos-not-applied
worker-qos-not-applied
madvise-random-not-applied
<target>-<operation>-duration-<quality>   (only when --iterations is omitted)
<target>-<operation>-high-cv              (n >= 3 and CV > 5.0%)
```

## 13. Validation, comparison protocol, and proof boundaries

Correctness is gated by C++ unit/integration tests (`tests/test_cpu_bandwidth_config.cpp`, `tests/test_cpu_bandwidth_work_plan.cpp`, `tests/test_cpu_bandwidth_oracle.cpp`, `tests/test_cpu_bandwidth_executor_runner.cpp`, `tests/test_cpu_bandwidth_json_and_sweep.cpp`, `tests/test_pattern_write_kernels.cpp`), Python verifier mutation tests (`tests/test_cpu_bandwidth_verifier.py`), and E2E acceptance (`tests/test_cpu_e2e_acceptance.py`).

For valid A/B comparisons: keep `methodology_version`, `component_identities`, frozen work plan (`--iterations` or matched calibrated passes), `worker_spans`, base seed, `--non-cacheable`, macOS/hardware, and thermal/power state matched; prefer `loop_count` divisible by `3 * num_targets` for complete position balance; and inspect `environment.start`/`end` (noting these are instantaneous snapshots, not continuous thermal telemetry) and `quality_warnings`. Neither `build_manifest` nor `binary_sha256` constitutes signed runtime execution attestation.

## 14. Change control

Any change to CLI options, target activation rules, 64B partitioning, payload formulas, seed/pattern/checksum derivations, ARM64 kernel instruction classes (`stnp` vs `stp`), cold post-validation coverage, timing boundaries, calibration targets, or JSON field types/meanings requires methodology/schema compatibility review and synchronized updates across `src/cpu_bandwidth/*`, `script-examples/verify_cpu_bandwidth_result.py`, and this whitepaper. Removing/renaming fields or altering semantics requires a `schema_version` bump.

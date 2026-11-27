# CPU Memory Latency Methodology

## Abstract

`memory_benchmark -L` (`--cpu-latency`) is a versioned Apple Silicon single-threaded dependent pointer-chasing benchmark for macOS 26 or later. It measures observed load-to-use critical-path delay (`ns/access`) across up to three active target classes—auto-detected L1 data cache, auto-detected L2 cache (or a single `--cache-size` custom target), and main memory (`--buffer-size`)—with selective target omission via `--buffer-size 0` or `--cache-size 0`.

Each target buffer is constructed once per run as a single deterministic Hamiltonian cycle over `N = target_bytes / stride_bytes` pointer-sized nodes using SplitMix64 with rejection-sampled Fisher–Yates permutation (`splitmix64-fisher-yates-rejection-v1`), cyclically rotated so `indices[0] == 0`, hashed with SHA-256 (`chain_permutation_sha256`), prevalidated for single-cycle reachability, and locked read-only via `mprotect(PROT_READ)`. Timed execution runs the untouched 8x-unrolled AArch64 chase kernel (`memory_latency_chase_asm`) over `complete_chain_cycles * N` dependent dereferences (`dsb ish; isb` + Mach `cpu_raw` ticks), verifying exact return to `base_ptr` (node `0`). When `--latency-samples > 0`, a separate non-headline sample pass continues from the running pointer across `effective_samples` windows and verifies every window's end-node index. Optional `--compare-locality` runs three paired `16 KiB` `random-box` versus `global-random` rounds per loop before target measurements. Reported nanoseconds reflect total dependent-load path cost (cache hierarchy, address translation, and runtime effects) and do not isolate or prove residency in any single hardware structure.

## Methodology identity

The current contract is identified by:

| Property | Value |
|---|---|
| Software version | Release provenance (`software.version`); not a schema or methodology selector |
| Mode | `cpu_latency` |
| JSON schema | `1` |
| Methodology | `cpu-latency-v1-arm64-calibrated-seeded-balanced` |
| Component identity | `cpu-latency-components-v1` |
| Work-plan identity prefix | `cpu-latency-work-plan-v1` |
| Chain generator | `splitmix64-fisher-yates-rejection-v1` |
| Chain format | `cpu-latency-chain-v1` |
| Run policy | `bounded-loop-snapshots-v1` |
| Canonical target order | `l1`, `l2`, `custom`, `main_memory` |
| Chase kernel | `memory_latency_chase_asm` (`8x` unrolled `ldr x0, [x0]`, single-threaded) |
| Timing boundary | `dsb ish; isb` + Mach `HighResTimer` (`cpu_raw` ticks) |
| Calibration policy | Excluded auto-calibration toward `0.250 s` (`[0.100 s, 0.300 s]`), min `16` cycles, max `2` corrections |
| Sweep envelope schema | `1` (`mode: "sweep"`, `base_mode: "cpu_latency"`, `sweep_schema_version: 1`) |

A cross-run comparison requires matched `methodology_version`, complete `component_identities`, frozen `plan_identity` and chain parameters (`stride`, `chain_mode`, `tlb_locality_kb`, `compare_locality`), base seed / explicit-vs-auto cycle counts, software version, hardware, thermal/power state, and output cadence.

## Scope and non-goals

`--cpu-latency` measures serialized load-to-use pointer-chase latency (`latency_ns = elapsed_seconds * 1e9 / completed_accesses`):

- **Why pointer chasing**: Sequential loops are hidden by out-of-order (OoO) execution, multiple outstanding fills, and hardware stream prefetchers, measuring bandwidth rather than latency. A circular linked list `p = *p` introduces a loop-carried read-after-write (RAW) dependency so load `n + 1` cannot compute its effective address until load `n` returns.
- **Physical non-claims**:
  - `interpretation.dram_residency` is `"unverified"` and `interpretation.cache_residency` is `"unverified"`. A cache-sized buffer does not guarantee that every load hits that cache, and a main-memory-sized buffer does not isolate DRAM from cache or TLB effects.
  - Observed latency combines cache/DRAM access, TLB and page-walk delays when working sets span pages, and OS scheduling/thermal variance. For isolated translation-boundary detection with paired bootstrap confidence intervals, use standalone `-T` / `--analyze-tlb`.
  - `-u` / `--non-cacheable` calls `madvise(..., MADV_RANDOM)` (`non_cacheable_is_best_effort_madvise: true`); user-space macOS cannot create uncacheable page-table mappings.
  - This project defines no fixed canonical nanosecond ranges for Apple Silicon cache or DRAM latency.
  - Bulk streaming bandwidth is out of scope (use `-B` / `--cpu-bandwidth`).

## 1. CLI, transport boundary, and exit codes

The standalone primary mode is `-L` / `--cpu-latency`, mutually exclusive with the six sibling modes in the seven-mode CLI (`-B`, `-P`, `-T`, `-C`, `-G`, `-M`). Its exact 16-option whitelist is:

```text
-L, --cpu-latency                   Run standalone audited CPU memory latency benchmark
-b, --buffer-size <MB>              Main memory buffer size in MB (default: 512, 0 to omit)
-k, --cache-size <KB>               Custom cache buffer size in KB (16-1048576, 0 to omit caches)
-i, --iterations <n>                Fixed complete chain cycles (1-1000000000; default: auto-calibrate)
-r, --count <n>                     Number of benchmark loops (default: 3)
-n, --latency-samples <n>           Sample windows per measurement (default: 1000, 0 to disable)
-s, --latency-stride-bytes <B>      Pointer stride in bytes, 8-byte aligned (default: 256)
-m, --latency-chain-mode <mode>     Chain mode: auto, global-random, random-box, same-random-in-box, diff-random-in-box
-l, --latency-tlb-locality-kb <KB>  TLB locality window in KB, page-aligned or 0 (default: 1024)
    --compare-locality              Run paired 16 KiB random-box vs global-random comparison rounds
    --seed <uint64>                 Explicit 64-bit base seed (0 allowed)
-u, --non-cacheable                 Best-effort madvise(MADV_RANDOM) allocation hint
-o, --output <target>               Write schema-1 JSON to file, or '-' for stdout JSON
-S, --sweep <key=v1,v2,...>         Parameter sweep (keys: buffer-size, cache-size, latency-tlb-locality-kb, latency-stride-bytes, latency-chain-mode)
-X, --sweep-max-runs <n>            Maximum generated sweep runs (default: 256)
-h, --help                          Display this help message and exit
```

### Parser, bounds, and target activation rules

1. **Syntax & uniqueness**: Built on `src/core/config/cli_parser.{h,cpp}`. Long value-taking options accept `--option value` and `--option=value`; boolean/mode flags reject `=value` and short options reject `-r=3`. Every option except `-S` / `--sweep` may occur at most once; `-S` is repeatable across distinct keys, and duplicate keys are rejected. Options outside the 16-option whitelist (such as `--threads`, `--patterns`, `--gpu-bandwidth`, `--llm-memory`, or removed legacy flags) fail with a single-line stderr diagnostic listing the allowed options.
2. **Numeric and semantic bounds**:
   - `--buffer-size`: `0 .. 2^53 - 1` (`9007199254740991`), default `512` MiB (`0` omits `main_memory`).
   - `--cache-size`: `0` or `16 .. 1048576` KiB (`0` omits caches; omitted auto-detects `l1` and `l2`).
   - `--iterations`: `1 .. 1000000000` (`10^9`) complete chain cycles (each target must also satisfy `complete_chain_cycles * node_count <= 4,000,000,000`).
   - `--count`: `1 .. 2^53 - 1`, default `3`.
   - `--latency-samples`: `0 .. 2147483647` (`INT_MAX`), default `1000`; `0` disables the separate sample-window pass.
   - `--latency-stride-bytes`: `8 .. 4294967295` (`UINT32_MAX`), must be a multiple of `8` (`sizeof(uintptr_t)`), default `256`.
   - `--latency-chain-mode`: `auto` (default), `global-random`, `random-box`, `same-random-in-box`, `diff-random-in-box` (canonicalized case-insensitively across `-`/`_`/space aliases).
   - `--latency-tlb-locality-kb`: `0 .. 2^53 - 1`, default `1024`. When effective mode is not `global-random`, `tlb_locality_kb * 1024` must be `> 0`, an exact multiple of `page_size_bytes`, and contain at least `2` stride-spaced nodes (`locality_bytes / stride_bytes >= 2`). Explicit `global-random` ignores `--latency-tlb-locality-kb` for chain construction.
   - `--compare-locality`: Explicit opt-in flag. Requires `buffer_size_mb > 0`, `stride_bytes <= 8192` (so a `16 KiB` box holds `>= 2` nodes), and `16384 % page_size_bytes == 0`.
   - `--seed`: `0 .. 18446744073709551615` (`UINT64_MAX`).
   - `--sweep-max-runs`: `1 .. 2^53 - 1`, default `256`.
3. **Target activation and canonical order**: At least one target (`--buffer-size > 0` or `--cache-size != 0`) is required. Active targets always appear in canonical order `("l1", "l2", "custom", "main_memory")`.
4. **Isolated help**: `-h` / `--help` exits `0` after syntax/range parsing without OS queries, allocation, or file writes.
5. **Sweep rules**: `-S` accepts `buffer-size`, `cache-size`, `latency-tlb-locality-kb`, `latency-stride-bytes`, and `latency-chain-mode`. Requires non-empty `--output`, forbids colliding direct flags for any swept key, caps Cartesian runs at `--sweep-max-runs`, and preflight-validates every combination (active targets, chain/locality/compare-locality geometry, memory budget, and explicit iteration access cap) before run `0`.

### Transport boundary and exit codes

Output targets (`""` console-only, `"-"` single final stdout JSON with human output on `stderr`, or CWD-relative file with atomic `.tmp` checkpoints every `K = max(1, ceil(loop_count / 8))` loops + terminal) and exit codes (`0` for `-h`, complete run, or checkpointed graceful interrupt; `1` for parse/preflight/runtime/checkpoint error) match the shared process contract. Authoritative acceptance requires `schema_version == 1 && mode == "cpu_latency" && status == "complete" && results_complete == true && run_accepted == true` (plus `conclusions_valid == true` and `completed_runs == planned_runs` for sweeps).

## 2. Dependent pointer-chasing geometry and guardrails

For each active target $t \in \{\text{l1}, \text{l2}, \text{custom}, \text{main\_memory}\}$:

```text
target_bytes(l1)          = l1_data_cache_bytes
target_bytes(l2)          = l2_data_cache_bytes
target_bytes(custom)      = custom_cache_size_kb * 1,024
target_bytes(main_memory) = buffer_size_mb * 1,048,576

node_count(t)             = target_bytes(t) / stride_bytes   (must be >= 2)
access_count(t)           = complete_chain_cycles * node_count(t)
```

For box-based chain modes (`effective_mode != "global-random"`):

```text
locality_bytes = latency_tlb_locality_kb * 1,024
locality_nodes = locality_bytes / stride_bytes   (must be >= 2)
box_count      = ceil(node_count / locality_nodes)
```

### Guardrails

| Guardrail | Exact Limit | Where Enforced |
|---|---|---|
| `kCpuLatencyMaxCliIterations` | `1,000,000,000` (`10^9` cycles) | CLI `--iterations` |
| `kCpuLatencyMaxAccesses` | `4,000,000,000` (`4 * 10^9` dereferences per task) | `build_cpu_latency_target_execution_plan`, calibration cap, verifier |
| `kCpuLatencyMinCompleteCycles` | `16` (floored at `min(16, max_cycles)`) | Auto-calibration cycle floor |
| `kMinPilotAccesses` | `16,384` | Initial auto-calibration pilot floor |
| `kCpuLatencyMaxMeasurements` | `100,000` (`loop_count * active_targets.size()`) | `build_cpu_latency_work_plan`, verifier |
| `kCpuLatencyMaxTargetPlans` | `4,096` | Plan table cap / verifier |
| `kCpuLatencyCvWarningPct` | `7.5%` (`CV > 7.5` with `n >= 3`) | `aggregates` quality warning |

For a target with `node_count` nodes, `max_cycles = floor(4_000_000_000 / node_count)`.

## 3. Deterministic chain construction, canonicalization, and golden vectors

### Mode resolution

```text
resolve_latency_chain_mode(requested_mode, locality_bytes):
  if requested_mode == "auto":
    return (locality_bytes == 0) ? "global-random" : "random-box"
  return requested_mode
```

| CLI Mode Token | Effective Strategy | Uses `tlb_locality_kb` | Behavior |
|---|---|---|---|
| `auto` | `global-random` if `l == 0`, else `random-box` | If `l > 0` | Default mode; adapts cleanly when `-l 0` is passed |
| `global-random` | `GlobalRandom` | No (`locality_nodes = 0`) | Single Fisher–Yates shuffle across all `0 .. N - 1` nodes |
| `random-box` | `RandomInBoxRandomBox` | Yes | Shuffles box visit order `0 .. box_count - 1`, then shuffles nodes independently within each box |
| `same-random-in-box` | `SameRandomInBoxIncreasingBox` | Yes | Shuffles one shared offset permutation of length `locality_nodes`, visits boxes in increasing order `0 .. box_count - 1`, keeping offsets `< end` (handling a partial final box cleanly) |
| `diff-random-in-box` | `DiffRandomInBoxIncreasingBox` | Yes | Visits boxes in increasing order `0 .. box_count - 1`, shuffling each box's `[start, end)` slice independently |

### Domain seeds and `splitmix64-fisher-yates-rejection-v1`

Every target derives `chain_seed = splitmix64_mix(base_seed xor domain_id)` from fixed 64-bit domain constants:

| Domain | Hex Constant | ASCII Tag |
|---|---|---|
| `l1` | `0x434C41545F4C315F` | `"CLAT_L1_"` |
| `l2` | `0x434C41545F4C325F` | `"CLAT_L2_"` |
| `custom` | `0x434C41545F435553` | `"CLAT_CUS"` |
| `main_memory` | `0x434C41545F4D4149` | `"CLAT_MAI"` |
| `locality_16k` | `0x434C41545F4C3136` | `"CLAT_L16"` |
| `locality_global` | `0x434C41545F474C4F` | `"CLAT_GLO"` |

Starting from `state = chain_seed`, all shuffles use `fisher_yates_rejection_shuffle_with_state` with SplitMix64 (`A = 0x9E3779B97F4A7C15`, `B = 0xBF58476D1CE4E5B9`, `C = 0x94D049BB133111EB`):

```text
for bound = len(items) down to 2:
  threshold = (2^64) mod bound   // uint64_wrap(0 - bound) % bound
  loop:
    state += 0x9E3779B97F4A7C15  (mod 2^64)
    draw = splitmix64_mix_raw(state)  // i.e. splitmix64_next
  until draw >= threshold
  j = draw mod bound
  swap(items[bound - 1], items[j])
```

Note that `splitmix64_next(state)` advances `state += A` first and mixes the updated `state` (`splitmix64_mix(x)` mixes `x + A`). One continuous `state` stream drives both the box-order shuffle (when `random-box`) and the within-box shuffles.

### Canonical rotation, SHA-256 identity, and Hamiltonian prevalidation

1. **Canonical rotation (`indices[0] == 0`)**: Because a circular tour of `N` nodes has `N` equivalent starting representations, `generate_cpu_latency_chain_permutation` cyclically rotates `indices` so node `0` is at `indices[0]`. This canonicalizes the tour order without altering the directed cycle and ensures every complete cycle starts and ends at `base_ptr` (slot `0`).
2. **Permutation digest (`chain_permutation_sha256`)**: SHA-256 over the `8 * N` little-endian `uint64_t` bytes of the rotated `indices[0 .. N - 1]`.
3. **Buffer linkage**: `slot[indices[i]] = &slot[indices[(i + 1) % N]]`, where `slot[k] = base_ptr + k * stride_bytes`.
4. **Hamiltonian prevalidation**: `validate_cpu_latency_hamiltonian_cycle` walks `N` pointer hops from `base_ptr`, checking alignment, bounds, and a `visited[N]` bit/byte array to confirm every node in `[0, N)` is visited exactly once and step `N` returns to `base_ptr`.
5. **Read-only protection**: `mprotect(buffer, protect_size, PROT_READ)` locks the chain before timing (`chain-readonly-protection`).

### Golden mathematical test vectors

All values below were computed via `script-examples/verify_cpu_latency_result.py` and verified bit-for-bit against live `./memory_benchmark -L` output and `tests/test_cpu_latency_chain.cpp`:

1. **Fixture-matched 64-bit seeds and SHA-256 digests (`base_seed = 424242`, `stride = 256`, `random-box`, `tlb_locality_kb = 1024`)**:
   - `l1` (`domain = 0x434C41545F4C315F`): `chain_seed = 3383413817332500519`
   - `l2` (`domain = 0x434C41545F4C325F`): `chain_seed = 4656391254177041200`
   - `custom` (`16384 B`, `N = 64`): `chain_seed = 7436701275638583618`, `indices[0..7] = [0, 44, 43, 3, 26, 56, 42, 45]`, `chain_permutation_sha256 = "968dc7e052a2c3f576982910159bcde7225ba76697bc3eb7a5cbb999ee159ea9"`
   - `main_memory` (`1048576 B`, `N = 4096`): `chain_seed = 5661157505406129473`, `chain_permutation_sha256 = "7b2c27666a8b73e9af9989fd5e860feed9cd4943296534ea095ed25ed0eafb94"`
   - `locality_16k` (`domain = 0x434C41545F4C3136`): `chain_seed = 11206493890640466498`
   - `locality_global` (`domain = 0x434C41545F474C4F`): `chain_seed = 17277003159666147137`
2. **8-node full-permutation goldens (`base_seed = 42`, `domain = main_memory`, `N = 8`, `chain_seed = 3357413162514629927`, `locality_nodes = 4` for box modes)**:

| Mode | Rotated `indices` | `chain_permutation_sha256` |
|---|---|---|
| `global-random` | `[0, 2, 1, 5, 4, 6, 3, 7]` | `a5c785cd4f63404e7354777a3ba19cd39a061b334ee4592c62dbc5dc095dcfd6` |
| `random-box` | `[0, 6, 5, 7, 4, 2, 1, 3]` | `19fc8aa4f869e4b48aa019495354958db89461b12f55bb120069ab63583bd825` |
| `same-random-in-box` | `[0, 7, 5, 6, 4, 3, 1, 2]` | `fb3a3c0bf0d3c84bcee8d4980a177c5bd237c8489c868fbb959b4eec0054f642` |
| `diff-random-in-box` | `[0, 6, 5, 4, 7, 3, 1, 2]` | `2b2811d19996408be5485422af46e0f7638f96fdb8d890c136273d7f8fb807fb` |

## 4. Physical memory mappings, strict memory budget, and resource evidence

Each active target owns one `MAP_PRIVATE | MAP_ANONYMOUS` buffer (`allocate_buffer_with_evidence`), and `--compare-locality` allocates one additional dedicated `compare_buffer` of `main_memory` byte length (reused across both `locality_16k` and `locality_global` subruns):

```text
allowed_memory_bytes    = (available_memory_bytes == 0)
                            ? 2048 * 1,048,576
                            : floor(available_memory_bytes * 0.80)
admitted_budget_bytes   = allowed_memory_bytes

requested_data_bytes    = sum_{t in active} target_bytes(t) + (compare_locality ? main_bytes : 0)
committed_data_bytes    = sum_{t in active} round_up(target_bytes(t), page_size)
                          + (compare_locality ? round_up(main_bytes, page_size) : 0)
resource_rounding_bytes = committed_data_bytes - requested_data_bytes

max_nodes               = max(locality_nodes, max_{t in active} node_count(t))
setup_peak_bytes        = 4096 + max_nodes * 8
```

When JSON output is enabled (`--output` non-empty or `--sweep`), `json_peak_estimate.total_bytes` adds:
- `2 MiB` fixed schema reserve + `16 * (output_file.size() + sum(argv[i].size()))`
- `16 KiB * (planned_measurements + total_calibration_attempts + total_target_plans + (compare_locality ? loop_count * 3 : 0))`
- `128 B * planned_measurements * min(latency_samples, max_nodes * iterations)` (or `128 B * planned_measurements * latency_samples` when auto-calibrating)

Total admission check (rejecting with `memory-budget-exceeded` rather than downsizing):

```text
transient_peak_bytes   = planned_measurements * 512 + setup_peak_bytes + json_peak_estimate.total_bytes
known_owned_peak_bytes = committed_data_bytes + transient_peak_bytes <= admitted_budget_bytes
```

Every buffer records `AllocatedBufferEvidence` under `resolved_plan.resources.<target>.buffer` (and `resolved_plan.resources.compare_buffer` when `--compare-locality` is active).

## 5. ARM64 chase kernel (`memory_latency_chase_asm`), timing boundary, and sample windows

### Assembly kernel (`src/asm/memory_latency.s`)

- **Prototype**: `extern "C" uintptr_t* memory_latency_chase_asm(uintptr_t* start_pointer, size_t count)` (`x0 = start_pointer`, `x1 = count`, returns terminal pointer in `x0`).
- **Safeguards**:
  1. **Zero-access fast return**: `cbz x1, latency_end` returns immediately if `count == 0`.
  2. **64-byte aligned `8x` unroll**: `lsr x4, x1, #3` executes `count / 8` iterations of 8 sequential `ldr x0, [x0]` loads with `subs x4, x4, #1; b.ne` at `.p2align 6` to keep the hot loop on a consistent I-cache boundary and minimize branch overhead while strictly preserving the RAW address dependency.
  3. **Unsigned remainder loop**: `and x5, x1, #7` executes `count % 8` single `ldr x0, [x0]` loads using `subs`/`b.ne`.
  4. **No internal barriers or prefetches**: Emits no `DMB`/`DSB` or prefetch hints inside the hot path; relies on `HighResTimer::start()` and `HighResTimer::stop()` emitting `dsb ish; isb` outside the chase loop.

### Main continuous pass and end-pointer verification

Each measurement runs `memory_latency_chase_asm(base_ptr, complete_chain_cycles * N)`. Because `base_ptr` is node `0` (`indices[0] == 0`) and `access_count` is an exact integer multiple of the Hamiltonian cycle length `N`, an uncorrupted execution **must** return `actual_final_ptr == base_ptr` (`final_pointer_verification`: `expected_node = 0`, `actual_node = 0`, `valid = true`).

### Separate sample-window pass (`--latency-samples`)

When `latency_samples > 0` and the main headline pass succeeded, a **separate** post-headline pass executes `access_count` additional dereferences across `effective_samples = min(latency_samples, access_count)` windows (never replacing or weighting the continuous headline):

```text
base_step = access_count / effective_samples
rem       = access_count % effective_samples
current_ptr = base_ptr
cum_steps   = 0
for idx = 0 .. effective_samples - 1:
  accesses_i = base_step + (idx < rem ? 1 : 0)
  cum_steps += accesses_i
  timer.start()
  current_ptr = memory_latency_chase_asm(current_ptr, accesses_i)
  timer.stop()
  // verify timebase matches main cpu_raw and finite positive tick conversion
  // verify pointer_to_chain_node_index(current_ptr) == indices[cum_steps % N]
```

**Pointer continuity**: Passing `current_ptr` from window `idx` as the start pointer for window `idx + 1` ensures each window begins at an arbitrary point along the cycle rather than restarting at hot-cached node `0`. Each window records `sample_delta_ticks[]`, `samples_ns[]`, and `validation.end_node_indices[]`; completed measurements for a target are pooled into `aggregates.targets.<target>.pooled_sample_distribution` (`semantics: "pooled-separate-sample-window-distribution"`). Passing `--latency-samples 0` disables the sample pass (`sample_windows: null`, `pooled_sample_distribution: null`).

## 6. Paired `--compare-locality` protocol

When `--compare-locality` is explicitly passed:

- **Eligibility**: Requires `buffer_size_mb > 0`, `stride_bytes <= 8192`, and `16384 % page_size_bytes == 0`.
- **Execution order**: In **every** loop `0 .. loop_count - 1`, **before** measuring the loop's target sequence, the runner executes `3` paired rounds (`round_index = 0, 1, 2`). Each round measures:
  1. `locality_16k` (`random-box`, `tlb_locality_kb = 16`, `domain = 0x434C41545F4C3136`), then
  2. `locality_global` (`global-random`, `tlb_locality_kb = 0`, `domain = 0x434C41545F474C4F`).
- **Cycle count & buffer safety**: Both subruns use `main_exec_plan.complete_chain_cycles` on the dedicated `compare_buffer`. Before each subrun, `compare_buffer` is switched `PROT_READ | PROT_WRITE`, populated with that chain's deterministic pointers, verified via `validate_cpu_latency_hamiltonian_cycle`, locked `PROT_READ`, timed via `memory_latency_chase_asm`, and verified to return to `base_ptr`.
- **Aggregation**: Across all measured rounds (`3 * loop_count` pairs for a complete run):

```text
locality_16k_latency_ns   = median(measured locality_16k latencies)
global_random_latency_ns  = median(measured locality_global latencies)
locality_latency_delta_ns = global_random_latency_ns - locality_16k_latency_ns
```

A `complete` run requires both all planned target measurements and all `3 * loop_count` locality pairs measured. Because `16 KiB` versus `global-random` alters both cache spatial locality and TLB page turnover, `locality_latency_delta_ns` is a mixed cache+TLB contrast; use `--analyze-tlb` for isolated TLB boundary analysis.

## 7. Check and validation fault model and evidence limits

Every calibration attempt and target measurement records `execution.validation.checks[]` in **five fixed positions**:

| Index | `kind` | Applicable When | What It Verifies |
|---|---|---|---|
| 0 | `post-validation-structure` | Always | Enabled target, valid execution plan & buffer, `mapped_bytes >= target_bytes`, `node_count >= 2`, `indices.size() == node_count`, `access_count > 0` |
| 1 | `chain-hamiltonian-prevalidation` | Always | `buffer.hamiltonian_prevalidated == true` (full `N`-step single-cycle walk completed at preparation) |
| 2 | `chain-readonly-protection` | Always | `buffer.read_only_protected == true && buffer.mprotect_errno == 0` |
| 3 | `main-final-pointer` | Always | `actual_final_ptr == base_ptr` (`expected_node = 0`, `actual_node = 0`) |
| 4 | `sample-window-end-pointers` | `latency_samples > 0` | Every window's `cpu_raw` timebase valid & `end_node_indices[idx] == indices[cum_steps % N]` |

| Fault / Mutation Class | Observable Detection | Scope / Caveat |
|---|---|---|
| Disconnected/short cycle or out-of-bounds link during chain setup | **Detected** (`validate_cpu_latency_hamiltonian_cycle` fails preparation; `chain-hamiltonian-prevalidation` backed by `tests/test_cpu_latency_chain.cpp`) | Verifies chain before timing; assumes `mprotect(PROT_READ)` prevents in-run buffer mutation |
| Wrong dereference count or corrupted final pointer in main pass | **Detected** (`main-final-pointer` fails unless `actual_delta_steps % N == 0`) | Modular cycle wrap: an error shifting total steps by an exact nonzero multiple of `N` still lands on node `0` |
| Wrong step count or corrupted pointer in sample-window pass | **Detected** (`sample-window-end-pointers` fails at the first mismatched window `indices[cum_steps % N]`, backed by `tests/test_cpu_latency_executor_runner.cpp`) | Disabled (`not-applicable`) when `-n 0`; modular wrap by multiples of `N` per window remains unobservable |
| Missing/zero Mach timing or kernel failure | **Detected** (`invalid-elapsed-time` / `kernel-failed`) | — |

## 8. Calibration, cyclic target order, statistics, and bounded checkpoints

- **Excluded calibration & freeze**:
  - Explicit `--iterations C`: validates `C * node_count <= 4,000,000,000` for each active target, freezes `frozen_plan_refs`, and runs `frozen_measurement_warmup` (excluded, `requested_samples = 0`) per target before loop `0`.
  - Automatic calibration (`--iterations` omitted): calibrates each active target in canonical order (`l1, l2, custom, main_memory`). Starts at `min_pilot_cycles` covering `>= 16,384` accesses (`kMinPilotAccesses`), runs `calibration_shape_warmup` then `pilot`, and performs up to `2` `correction` steps toward `0.250 s` (`[0.100 s, 0.300 s]`), floored at `min_cycles = min(16, max_cycles)` (with `single_unit_confirmation_warmup` + `single_unit_confirmation` if `min_cycles` is reached). Freezes all `frozen_plan_refs` atomically and runs `frozen_measurement_warmup` before loop `0`.
- **Cyclic target order**: Loop `i` measures `active_targets[(i + pos) % num_targets]`. `target_order_balance_complete = results_complete && num_targets > 0 && (loop_count % num_targets == 0)`.
- **Statistics**: `aggregates.targets.<target>` aggregates `measured` continuous headlines (`value = median`, `average`, `min`, `max`, `P90/P95/P99`, `stddev`, `CV%`, `MAD`, `repeatability_assessment_available = (n >= 2)`, `cv_warning = (n >= 3 && cv > 7.5%)`). Pooled sample windows are aggregated separately under `pooled_sample_distribution`.
- **Interruption & checkpoints**: Task-boundary completion-wins `SIGINT`/`SIGTERM` handling; bounded atomic file snapshots every `K = max(1, ceil(loop_count / 8))` loops plus terminal (`3K` abrupt loss bound); `run_accepted = (status == "complete") && results_complete && !checkpoint_failed`.

## 9. CPU Latency JSON schema 1, sweep envelope, and build manifest

Top-level keys (26, in canonical `nlohmann::ordered_json` order):

```text
schema_version, mode, methodology_version, software, build_manifest,
configuration, resolved_plan, memory_budget, calibration, measurements,
locality_comparison, aggregates, status, reason_code, results_complete,
run_accepted, interpretation, diagnostic, interruption_requested,
target_order_balance_complete, seeds, counters, checkpoint_lifecycle,
loop_records, environment, quality_warnings
```

- **Encoding rules**: `completed_accesses` and `completed_chain_cycles` in `measurements[]` (and `locality_comparison.rounds[]`) are JSON integers `<= 2^53 - 1`, whereas `counters.planned_accesses`, `counters.completed_accesses`, `resolved_plan.targets.<t>.bytes`, `chain_seed_uint64_decimal`, Mach ticks, and `sample_delta_ticks[]` are canonical unsigned decimal strings (`0` or `[1-9][0-9]*`).
- **Sweep envelope**: `--sweep` writes schema-1 sweep JSON (`configuration.mode: "sweep"`, `base_mode: "cpu_latency"`, `sweep_schema_version: 1`) across `buffer-size`, `cache-size`, `latency-tlb-locality-kb`, `latency-stride-bytes`, and `latency-chain-mode`, embedding nested schema-1 documents with `configuration.output_file: ""`.

## 10. Independent Python artifact verifier

`script-examples/verify_cpu_latency_result.py` (using `script-examples/cpu_verify_common.py`, exercised by `make test-cpu-verifiers` and `make test-all`) verifies direct schema-1 artifacts and schema-1 sweep envelopes offline:

```bash
python3 script-examples/verify_cpu_latency_result.py <result.json> [--binary <path>] [--require-raw-timing] [--max-chain-nodes N]
```

- **Limits**: `64 MiB` max file, depth `64`, `1,000,000` JSON nodes, `100,000` measurements, `4,096` target plans, `1,000,000` shared `oracle_work_used`, and `--max-chain-nodes` (default `8,388,608` nodes per chain). Rejects duplicate keys and non-finite numbers.
- **Verified invariants**: Mode/schema/methodology/component identities; target geometry & `auto` chain mode resolution; full SplitMix64 + Fisher–Yates chain reconstruction, `indices[0] == 0` canonicalization, and `chain_permutation_sha256` verification; `memory_budget` arithmetic (including `compare_buffer`); ordered 5-entry `validation.checks[]`; `main-final-pointer`; sample-window timebase, `samples_ns`, `end_node_indices == indices[cum_steps % N]`, and `pooled_sample_distribution`; paired `locality_comparison` rounds and `locality_latency_delta_ns = median(global) - median(16k)`; cyclic `loop_records`; Mach `cpu_raw` reconciliation (`--require-raw-timing`); `build_manifest` + `--binary` SHA-256 binding; and sweep invariants.
- **Exit codes**: `0` = consistent & accepted; `1` = inconsistent, unaccepted, or invalid JSON; `2` = unsupported (`artifact_consistent: null` for schema/mode/methodology mismatch, chain node count exceeding `--max-chain-nodes`, work/size limit exceeded, or missing raw timing with `--require-raw-timing`) or `argparse` error.

## 11. Console contract, quality warnings, comparison protocol, and proof boundaries

Console output reports CPU topology, page size, work policy (`explicit_fixed_work` or `automatic_calibration (target=0.250s)`), loops, sample count, resolved/requested chain mode, stride, locality window, `compare_locality`, base seed, and per-target bytes/node counts. Summary lines print per-target `P50=... ns (min=..., max=..., CV=...%, MAD=..., n=...)`, optional `| PooledSamples P50=... ns (P90=..., P99=..., n=...)`, and optional `Locality Comparison: 16KB P50=... ns, Global P50=... ns, Delta=... ns`, followed on `stderr` by deduplicated `quality_warnings[]` tokens:

```text
environment-not-nominal
main-thread-qos-not-applied
madvise-random-not-applied
<target>-duration-<quality>   (only when --iterations is omitted)
<target>-high-cv              (n >= 3 and CV > 7.5%)
```

For cross-run comparisons, match `methodology_version`, `component_identities`, `stride`, `chain_mode`, `tlb_locality_kb`, `compare_locality`, frozen `complete_chain_cycles`, base seed, `--non-cacheable`, hardware/macOS, and thermal/power state (`environment.start`/`end` are instantaneous snapshots, not continuous telemetry). Prefer `loop_count` divisible by `num_targets` for complete target-order balance. Any change to CLI options, chain generation, canonicalization, kernel instructions, sample-window semantics, `--compare-locality` ordering, timing boundaries, or JSON fields requires methodology/schema compatibility review across `src/cpu_latency/*`, `script-examples/verify_cpu_latency_result.py`, and this whitepaper.

## Related Documents

- [CPU_BANDWIDTH_WHITEPAPER.md](CPU_BANDWIDTH_WHITEPAPER.md) — Standalone CPU memory bandwidth benchmark (`-B` / `--cpu-bandwidth`): 64B partitioning, O(1) GF(2) read checksum, cold full-buffer validation, and schema 1 contract.
- [TLB_ANALYSIS_WHITEPAPER.md](TLB_ANALYSIS_WHITEPAPER.md) — Standalone TLB boundary detection mode (`-T` / `--analyze-tlb`): paired bootstrap CI and boundary validation.
- [CORE_TO_CORE_WHITEPAPER.md](CORE_TO_CORE_WHITEPAPER.md) — Core-to-Core cache-line handoff latency benchmark (`-C` / `--analyze-core2core`).
- [TECHNICAL_SPECIFICATION.md](TECHNICAL_SPECIFICATION.md) — Runtime architecture, execution flow, and output contracts for all seven modes.

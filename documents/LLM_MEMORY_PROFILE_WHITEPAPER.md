# Synthetic LLM Memory Profile

## Abstract

`memory_benchmark --llm-memory` is a versioned Apple Silicon synthetic memory benchmark with a generic
backend/phase/KV-layout schema. CPU decode is active with contiguous or deterministic paged KV, and CPU prefill is
active with contiguous or deterministic paged KV. Metal decode is active with both layouts as an experimental preview.
Every selected profile executes three scenarios derived from the same explicit model geometry:

- `weights_only`: read the active weights once;
- `kv_only`: perform the phase-specific K/V write and reads;
- `mixed`: perform the weight and KV work in one timed scenario, using CPU worker-local layer order or the Metal
  scenario-specialized grid-stride kernel according to the selected backend.

The result is effective logical payload divided by backend-authoritative elapsed time. It is not a Transformer implementation, an
inference-engine benchmark, a physical DRAM counter, or a `tokens/s` claim.

## Methodology identity

The current contract is identified by:

| Property | Value |
|---|---|
| Software version | `0.63.0` |
| Mode | `llm_memory` |
| Backend | `cpu` or `metal` |
| JSON schema | `1` |
| Phase selector | `decode` or `prefill` |
| Work unit | `decode_step` or `prefill_operation` |
| Methodology | `llm-memory-v1-<backend>-<phase>-<layout>` |
| Model/scenario plan identity prefix | `llm-memory-work-plan-v1` |
| Component identity prefix | `llm-memory-components-v1` |
| Logical profile version | phase-specific decode or prefill profile identity |
| KV layout selector/version | `contiguous` / `contiguous_layer_batch_token_head_dimension`, or `paged` / `paged-uint32-block-table-full-blocks-v1` |
| Permutation version | null for contiguous; `splitmix64-fisher-yates-rejection-v1` for paged |
| Backend executor version | phase/layout-specific CPU ARM64 or Metal scenario-pipeline identity |
| Resource ABI | CPU descriptor identity or Metal Tier 2 argument-buffer ABI identity |
| Schedule version | CPU owner-local or Metal capped grid-stride schedule identity |
| Timer policy | CPU synchronized worker timer or Metal GPU command-buffer timestamp policy |
| Buffer pattern | `llm-buffer-pattern-v1` or `llm-paged-physical-buffer-pattern-v1` |
| Write pattern | phase-specific append or full-prompt affine64 identity |
| Checksum pattern | CPU phase/layout checksum or `llm-metal-dual-mod32-v1` |
| MSL revision/source SHA-256 | null / null for CPU; exact selected runtime source identity for Metal |
| Traffic classification | `llm-exact-weight-vs-kv-read-payload-v1` |

Component identities use a fixed-order, length-prefixed canonical serialization beginning with
`llm-memory-components-v1`. Always-applicable identities are strings; permutation and MSL identities are JSON null when
they do not apply. A workload comparison requires the selectors, methodology, complete component identity, frozen
work-plan, and environment evidence, not merely the same nominal model name.

## Scope and non-goals

The active profiles model warm-memory decode or full-prompt prefill traffic with one active weight pass per work unit.
They support MHA, GQA, and MQA geometry through explicit query- and KV-head counts, a positive batch
count, and 1-, 2-, or 4-byte KV elements. Paged KV adds timed table indirection, deterministic physical scatter,
block-granular ownership, and full-block suffix padding.

Schema-v1 vocabulary includes `cpu|metal`, `decode|prefill`, `contiguous|paged`,
`decode_step|prefill_operation`, and `none|current_token_append|full_prompt_population`. Contiguous and paged are public
for both CPU phases and for Metal decode. The experimental Metal preview never receives hidden fallback.
Current M4 evidence covers both contiguous and paged decode. Paged validation includes partial terminal blocks,
permutation and padding rejection, all scenarios, and multi-segment K/V execution. Required Apple7/M1 baseline
validation remains pending. Apple7-or-later runtime capability admission therefore does not establish cross-family
production-ready validation.

The prefill implementation resolves checked tile/prefix/payload formulas, versioned atomic CPU ownership evidence,
owner-local semantic traces, and operation-ordinal checksum oracles. Contiguous prefill uses token-range ownership;
paged prefill uses block-exclusive weighted ownership plus a read-only uint32 table, deterministic permutation,
full physical K/V pools, a dedicated descriptor ABI, and a separate ARM64 executor.

It intentionally excludes:

- GEMV/GEMM/FMA, dequantization, RoPE, softmax, layer normalization, activation, and scratch traffic;
- tokenizer, model loader, framework scheduler/dispatch, kernel fusion, and compute-memory overlap;
- MLX, llama.cpp, Core ML, ANE execution, and GPU execution outside the defined Metal decode kernels;
- chunked/prefix-reuse prefill or a context that grows during one measurement;
- runtime KV allocation/free lists, prefix sharing, copy-on-write, eviction, sliding windows, ragged batches, block
  swapping, fragmentation simulation, or KV compression;
- speculative decoding and model-specific control flow;
- built-in model presets, model-file introspection, and LLM parameter sweeps;
- recycling a small physical buffer to represent a larger logical model;
- hardware-counter claims about cache, SLC, TLB, memory-controller, or DRAM traffic.

For a mixture-of-experts model, `--weight-size-mb` must represent weights active for one synthetic work unit, not the
model's total stored weights unless all of them are active.

## CLI and output contract

The standalone primary mode is `-M` / `--llm-memory`. Its exact whitelist is:

```text
--llm-memory-backend <cpu|metal>
                            default cpu; Metal preview accepts decode with either layout
--weight-size-mb <MiB>      required
--layers <count>            required
--query-heads <count>       required
--kv-heads <count>          required
--head-dim <count>          required
--phase <decode|prefill>    default decode
--context-tokens <count>    required only for decode
--prompt-tokens <P>         required only for prefill
--attention-query-tile-tokens <Q>
                            required only for prefill; 1 <= Q <= P
--kv-element-bytes <1|2|4>  default 2
--batch-size <count>        default 1
--kv-layout <contiguous|paged>
                            default contiguous
--kv-block-tokens <G>       required exactly once for paged; rejected for contiguous
-t, --threads <count>       CPU only; default detected CPU worker count
-i, --iterations <count>    default automatic per-scenario calibration
-r, --count <count>         default 3
--seed <uint64>             default one generated nonzero base seed
-o, --output <target>       default console only
-h, --help
```

Every required common and phase-specific option must occur exactly once. Optional values and the mode/help selector may
occur at most once. Numeric input is a complete decimal token; counts are positive, while an explicit seed may be zero.
Query heads must be at least the KV-head count and divisible by it. `G` must be a positive power of two no greater than
`UINT32_MAX`; it may exceed the active phase length. Phase/layout validation is order-independent.
All other primary modes and all buffer/cache/latency/TLB/pattern/GPU,
`--non-cacheable`, `--sweep`, and `--sweep-max-runs` options are rejected.

Backend defaults to `cpu`; phase defaults to `decode`. Layout defaults to `contiguous`; explicit backend/phase/layout
and block-size sources are retained in `configuration.resolved_sources`. Decode and prefill inputs are mutually
exclusive. All four CPU phase/layout combinations are active. The Metal preview accepts decode with contiguous or
paged KV, rejects explicit `--threads`, performs no CPU worker detection, and never falls back to another profile.

Output targets follow the shared process contract:

- omitted or empty output means console only;
- exact `--output -` reserves stdout for one final schema 1 document and routes the post-parse human transcript to
  stderr;
- every other non-empty raw token is a file, including `./-` and flag-shaped names;
- file output uses atomic `<target>.tmp` replacement after each terminal scenario measurement and at command terminal;
- stdout performs the same logical checkpoint and stop transitions without intermediate serialization.

Parser/logical-preflight or JSON-output peak-estimation failure before runner-result initialization leaves stdout empty.
A Metal runtime capability failure after output-session creation emits terminal `unsupported` JSON when enabled and
returns nonzero without CPU fallback. Runtime compiler, pipeline, resource, or task failure emits terminal
`failed`/`invalid` schema evidence and returns nonzero. Once the runner initializes status-bearing evidence, normal
failure or graceful interruption is serializable. A file checkpoint failure is terminal and is not retried by a final
file write.

## Exact geometry and logical traffic

Let:

- `W` = active weight bytes per work unit;
- `L` = layer count;
- `h_q` = query-head count;
- `h_kv` = physical KV-head count;
- `d_h` = elements per K/V head vector;
- `s_kv` = bytes per KV element;
- `B` = batch-sequence count;
- `A` = decode visible context tokens, including the current synthetic token;
- `P` = prefill prompt tokens;
- `Q` = prefill attention query-tile tokens;
- `T` = work units in one scenario measurement.

The head-sharing ratio and classification are:

```text
query_heads_per_kv_head = h_q / h_kv
MHA: h_q == h_kv
GQA: h_kv > 1 and h_q > h_kv
MQA: h_kv == 1 and h_q > 1
```

The classification is metadata. KV bytes depend on `h_kv`, not on the number of query heads sharing each KV head.

One head vector and one combined K+V record are:

```text
kv_vector_bytes = d_h * s_kv
k_or_v_record_bytes_per_layer = h_kv * d_h * s_kv
kv_record_bytes_per_layer = 2 * h_kv * d_h * s_kv
```

Define the combined K+V bytes for one visible token across all layers:

```text
K = L * 2 * h_kv * d_h * s_kv
```

### Decode logical traffic

The full mapped KV capacity and per-work-unit work are:

```text
k_mapping_bytes = L * B * A * h_kv * d_h * s_kv
v_mapping_bytes = k_mapping_bytes
kv_capacity_bytes = k_mapping_bytes + v_mapping_bytes = B * A * K

weight read / decode work unit = W
KV read / decode work unit = B * A * K
KV append write / decode work unit = B * K
```

Scenario payloads are therefore:

```text
weights_only / work unit = W
kv_only / work unit      = B*A*K + B*K
mixed / work unit        = W + B*A*K + B*K
```

Every per-measurement exact byte count is its per-work-unit value multiplied by `T` with checked arithmetic. The active
weight read is not multiplied by batch: one batched decode work unit shares the same active-weight pass, while KV work
is per batch sequence.

The decimal effective rate is:

```text
effective_model_payload_gb_s = completed_effective_model_payload_bytes / elapsed_seconds / 1e9
```

The numerator excludes cache-line fills, write allocate/RFO, writeback, hardware prefetch, translation, page-table, and
checksum/control traffic. Those effects can influence elapsed time without being added to logical payload.

### Prefill logical traffic

One `prefill_operation` processes the complete P-token prompt for every batch sequence. Let:

```text
C = ceil(P / Q)
e_j = min((j + 1) * Q, P), j = 0 .. C - 1
S(P,Q) = sum(e_j)
       = Q * triangular(P / Q) + (P % Q != 0 ? P : 0)
```

The executor advances tile ends by `min(Q, P-current_end)` rather than computing an overflowing `(j+1)*Q` product.
`Q=P` gives one scan with `S=P`; `Q=1` gives `S=triangular(P)`. Each query tile reads the complete causal prefix at
its end, not only the tile itself. One operation has:

```text
k_mapping_bytes = L * B * P * h_kv * d_h * s_kv
v_mapping_bytes = k_mapping_bytes
weight read = W
KV write = B * P * K
KV read = B * S(P,Q) * K

weights_only = W
kv_only = B * (P + S(P,Q)) * K
mixed = W + B * (P + S(P,Q)) * K
```

The weight pass occurs once per full prompt, not once per token or tile. Audit metadata separately records
`causal_token_pairs_per_sequence = triangular(P)`, `logical_attention_pairs = L*B*h_q*triangular(P)`, and
`logical_attention_fma_terms = logical_attention_pairs*d_h`. These values are not payload and no FMA is executed.

### Paged physical geometry, table, and lookup traffic

For paged layout, let `A` be visible context for decode or `P` for prefill, let `R = h_kv*d_h*s_kv` be one K or V
token record per layer, and let `G` be the explicit block size in tokens. Checked geometry derives:

```text
N = A / G + (A % G != 0)
P_b = B * N
block_bytes = G * R
last_block_tokens = A - (N - 1) * G
last_block_valid_bytes = last_block_tokens * R
decode_append_offset_in_last_block = ((A - 1) % G) * R  # decode only; null for prefill

k_logical_bytes = L * B * A * R
k_physical_bytes = L * P_b * block_bytes
k_layout_padding_bytes = k_physical_bytes - k_logical_bytes
v_logical_bytes = k_logical_bytes
v_physical_bytes = k_physical_bytes
v_layout_padding_bytes = k_layout_padding_bytes
block_table_entries = B * N
block_table_bytes = block_table_entries * 4
```

Every physical block has exactly `G` records; there is no extra inter-block alignment padding. Each batch sequence's
last block has initialized suffix padding. Timed work never reads or writes that padding, and post-validation checks its
canary. Layout padding, CPU page rounding, table bytes, preparation transient, and admitted peak are reported
separately.

One row-major `uint32_t block_table[B][N]` contains a bijection over `0..P_b-1`. `UINT32_MAX` is reserved as an invalid
sentinel, so `P_b <= UINT32_MAX`. The same physical ID selects the corresponding block inside every layer's K pool and
V pool:

```text
logical_block = token / G
token_in_block = token % G
p = block_table[batch][logical_block]
offset = (layer * P_b + p) * block_bytes + token_in_block * R
```

The table is generated once per command and remains frozen across warmup, calibration, loops, and scenarios. Starting
from identity order, it uses the stateful SplitMix64 stream and descending Fisher–Yates with rejection sampling:

```text
state += 0x9E3779B97F4A7C15
z = state
z = (z xor (z >> 30)) * 0xBF58476D1CE4E5B9
z = (z xor (z >> 27)) * 0x94D049BB133111EB
value = z xor (z >> 31)

bound = i + 1
threshold = uint64_wrap(0 - bound) % bound
draw until value >= threshold
j = value % bound
swap(table[i], table[j])
```

All stream arithmetic is modulo `2^64`. The permutation domain is `0x4c4c4d4b56504731`, and the resolved state is
`splitmix64(base_seed xor domain)`. Identity records the algorithm version, domain, resolved seed, entry count, and the
SHA-256 of explicit row-major little-endian `uint32_t` entries. Generation and hashing do not depend on the C++ standard
library's random algorithms or host endianness. Range/bijection/sentinel validation completes before descriptors are
published, and the CPU table becomes read-only before warmup.

For base seed `42` and eight entries, the resolved seed is `8109369757063363730`, the permutation is
`[0, 6, 2, 3, 7, 1, 5, 4]`, and its little-endian SHA-256 is
`4032b29a855010d82199c15c3f3e2b94582b86e67b3add8cb86bebc425f9c2b4`.

Per layer and batch sequence, paged decode performs exactly one paired current-token K/V append lookup, `N` K-scan
lookups, and `N` V-scan lookups. KV-only and mixed therefore use:

```text
layout_metadata_lookup_count_per_work_unit = L * B * (2 * N + 1)
layout_metadata_read_bytes_per_work_unit = 4 * L * B * (2 * N + 1)
accounted_bytes_per_work_unit =
  effective_model_payload_bytes_per_work_unit + layout_metadata_read_bytes_per_work_unit
```

Weights-only uses zero lookups. Each semantic lookup is an explicit timed 32-bit table load; the loaded ID determines
the data address. The host does not pre-resolve physical IDs. The paired append shares one lookup, while K and V scans
load the table independently. Lookup metadata is included in task guardrails but excluded from the primary GB/s
numerator.

Paged prefill also defines `m_j = ceil(e_j/G)` and `M = sum(m_j)`. Per layer/batch pair it performs `N` paired K/V
write lookups to populate the prompt, followed by `M` K-prefix lookups and `M` V-prefix lookups. KV-only and mixed use:

```text
layout_metadata_lookup_count_per_work_unit = L * B * (N + 2 * M)
layout_metadata_read_bytes_per_work_unit = 4 * L * B * (N + 2 * M)
accounted_bytes_per_work_unit =
  effective_model_payload_bytes_per_work_unit + layout_metadata_read_bytes_per_work_unit
```

Each prefix visit is physically exact: when `e_j` ends inside a block, the kernel scans only the valid bytes through
that tile end. It does not round a partial logical visit up to a complete block. The total lookup count therefore
depends only on P/Q/G/L/B and is invariant under worker-count changes.

An independent geometry golden with `A=35`, `G=16`, `L=2`, `B=2`, and `R=32` has `N=3`, `P_b=6`,
`block_bytes=512`, `last_block_tokens=3`, `last_block_valid_bytes=96`, K logical/physical/padding bytes
`4480/6144/1664`, six table entries/24 table bytes, and 28 lookups/112 metadata bytes per KV-bearing work unit.

Independent paged-prefill lookup goldens are `P=5,Q=2,G=2`: `N=3,M=6,N+2M=15`; and
`P=7,Q=3,G=2`: `N=4,M=9,N+2M=22`. For `P=6,Q=2,G=4`, successive tile prefixes visit 2, 4, and 6 valid tokens, so
the final tile alone reaches logical block 1 and its partial terminal bytes.

## Formula golden vectors and decode crossover

For:

```text
W = 4 GiB
L = 32
h_q = 32
h_kv = 8
d_h = 128
s_kv = 2 bytes
B = 1
```

the exact derived values are:

| Field | Exact value |
|---|---:|
| `kv_vector_bytes` | 256 B |
| combined K+V record per layer/token | 4096 B |
| `K` | 131072 B/token (128 KiB/token) |
| KV read at `A=8192` | 1073741824 B/work unit (1 GiB/work unit) |
| KV append at `A=8192` | 131072 B/work unit |
| KV-only at `A=8192` | 1073872896 B/work unit |
| mixed at `A=8192` | 5368840192 B/work unit |

The exact logical weight/KV-read crossover is:

```text
traffic_crossover_context_tokens = W / (B * K)
```

Schema 1 preserves the numerator and denominator as exact decimal strings plus a floating estimate. For the vector
above, the ratio is `4294967296 / 131072`, exactly 32768 visible tokens.

The current-context classification is versioned as `llm-exact-weight-vs-kv-read-payload-v1`:

- weight bytes greater than KV-read bytes: `weight_payload_dominant`;
- exact equality: `near_crossover`;
- KV-read bytes greater than weight bytes: `kv_read_payload_dominant`.

There is no tolerance band. The classification compares bytes only; it does not establish which kernel or hardware
resource limits measured performance.

An independent contiguous-prefill golden with `W=1024`, `K=128`, `B=2`, `P=5`, and `Q=2` has tile ends
`[2,4,5]`, `C=3`, `S=11`, weight read 1024 bytes, KV read 2816 bytes, KV write 1280 bytes, KV-only payload 4096
bytes, and mixed payload 5120 bytes. Decode crossover fields are null for this profile.

## Mappings, budget, and layout

The command owns regular private anonymous cacheable mappings for the suite lifetime: active weights and either exact
logical contiguous K/V or full physical paged K/V pools. Paged execution additionally owns one cacheable, page-rounded
`uint32_t` table mapping. Contiguous prefill uses the same full logical K/V mapping shape with P in place of A. Each
mapping is rounded separately to native page granularity for committed-byte accounting.
Admission includes every page-rounded mapping, table bytes, descriptor arrays, retained pointer-free planner storage,
expected/actual checksum storage, worker/thread state, calibration/result records, statistics workspace, warnings, and
orchestration storage. It also includes paged permutation construction, the checked `ceil(P_b/8)` range/bijection
bitset,
and hashing as a transient peak. Table materialization occurs only after full known-owned admission; the transient is
released before timed work. The normal policy admits no more than 80% of the current available-memory estimate; the
existing fallback applies when that sample is unavailable and caps the admitted total at 2 GiB. The workload is
rejected rather than silently scaled, and `G`, context, workers, or work-unit geometry are never changed to fit.

A non-empty file or stdout JSON target also reserves a conservative peak for one live schema DOM and its serialized
transport text before final memory admission. The estimate covers fixed schema storage, captured input strings, every
planned measurement record, and both expected and actual worker-checksum trees. The input-string term also includes
the frozen model-plan, methodology, component/layout identities, and applicable prefill aggregate, scenario, execution,
and scope identities. Frozen identities are charged once. Scenario-plan identities scale with the maximum retained
calibration attempts and planned measurement loops. Each variable-length addition is checked, and preliminary and finalized-plan
estimates use the same canonical identity-size formula. Omitted or empty output adds no serialization reserve; normal
console-only execution does not serialize a schema document.

Schema ownership separates immutable resource geometry from allocation/admission evidence. `resolved_plan.resources`
contains canonical decimal-string logical weight/K/V lengths, physical K/V lengths, layout padding, and nullable
block-table bytes. Top-level `memory_budget` contains canonical decimal-string `resource_rounding_bytes`,
`transient_peak_bytes`, `known_owned_peak_bytes`, and `admitted_budget_bytes`, along with any additive detailed estimate
evidence. Active contiguous profiles have identical K/V logical and physical lengths, zero layout padding, and a null
block-table resource.

Metal decode owns private/tracked W/K/V buffers, plus shared/tracked Tier 2 argument-buffer and status/checksum storage.
Contiguous pools use exact canonical segments no larger than 256 MiB. Paged K/V segments contain whole blocks, and the
private uint32 table uses whole-entry segments; transient shared staging uploads one bounded table segment at a time.
No segment-capacity slack or paged suffix padding enters payload. Each pool has at most 256 argument-buffer slots.
Runtime admission uses actual encoder length/alignment, `maxBufferLength`, and each resource's `allocatedSize` when
available. Runtime admission requires Apple7-or-later capability, unified memory, Tier 2 argument buffers,
`maxBufferLength >= 256 MiB`, and successful MSL 2.3 compilation of the common foundation pipelines, the selected
profile's three scenario pipelines, and its layout-specific validation/probe pipelines.

The activated embedded source revision is `llm-metal-decode-contiguous-paged-msl23-v2`. Runtime evidence binds its
exact source SHA-256. Contiguous scenario labels use `membenchmark.llm-metal.pipeline.decode-contiguous.*`; paged labels
use `membenchmark.llm-metal.pipeline.decode-paged.*`. The paged layout probe is
`membenchmark.llm-metal.pipeline.decode-paged-layout-probe`, and excluded validation uses
`membenchmark.llm-metal.pipeline.validate-decode-paged-appends-padding`.

Contiguous K and V use `[layer][batch_sequence][token][kv_head][head_dimension]`. Paged K and V use
`[layer][physical_block][token_in_block][kv_head][head_dimension]`; their block table contains no pointers. The visible
decode context includes the current token, so its append record is the final logical token. Prefill instead owns all
prompt records. Separate K and V pools make the two streams explicit and prevent aliasing.

Active weights are divided across layers by quotient and remainder so every byte belongs to exactly one layer. Every
layer's weight span is partitioned into disjoint ranges. Contiguous K/V visible spans are similarly byte-partitioned.
Paged K/V partitions use exact accounted-cost prefix sums and assign complete logical blocks; no block is split between
workers. If `N` is smaller than the worker count, ownership rotates deterministically by layer/batch ordinal. The
active CPU planner caps the admitted team so every effective worker owns KV work. Effective
workers are the minimum of requested workers, detected availability, and executable span capacity; requested and
available counts remain separately recorded. Lookup count is invariant under worker-count changes.

Decode's contiguous 64-bit ARM64 ABI uses 16-byte-aligned 48-byte layer descriptors and 80-byte sequence descriptors.
The separate paged ABI uses 48-byte layer descriptors and 96-byte block-assignment descriptors carrying the table-row
pointer, K/V layer-pool bases, first logical block/count, block geometry, last valid bytes, append offset/size, and
layer/batch identity. Every layout has independent `sizeof`/`offsetof` assertions and a version identity; neither hot
kernel branches on layout. Empty worker spans are null/zero and skipped, and no admitted worker has an entirely empty
scenario plan.

Contiguous prefill has separate 48-byte layer and 80-byte owner descriptors. Each owner descriptor carries K/V bases,
first token/count, P, Q, record bytes, and layer/batch identity. Its scenario-specific exact-cost partition records a
canonical identity plus minimum, maximum, and imbalance bytes in `worker-cost` units.

Paged prefill has separate 48-byte layer and 112-byte block-assignment descriptors rather than overloading paged decode
fields. Each block-assignment descriptor carries the read-only table-row pointer, physical K/V layer-pool bases, first
logical block/count, P/N/G,
block and terminal valid-byte geometry, Q/tile geometry, and layer/batch identity. Whole logical blocks are owned by
exact scenario-weighted prefix cost; no block is split, and the union of assignments covers every logical block exactly
once. Independent `sizeof`/`offsetof` goldens freeze this ABI.

## Scenario traversal

`weights_only` sets the kernel's weight bit and, for every work unit and layer, reads that worker's layer weight shard
once.

Decode `kv_only` sets the KV bit. For every work unit, layer, and batch sequence, the worker:

1. writes its current-token K append subrange;
2. writes its current-token V append subrange;
3. reads its complete visible K subrange;
4. reads its complete visible V subrange.

The read includes the current token that was just written.

For contiguous KV those operations traverse the assigned byte ranges. For paged KV the owner processes complete
logical blocks in increasing order and the semantic order is exact:

1. load the final logical block's physical ID once and perform the paired current-token K/V append;
2. traverse logical blocks `0..N-1`, explicitly load each ID, and scan only the valid K bytes;
3. traverse logical blocks `0..N-1` again, explicitly load each ID, and scan only the valid V bytes.

The final partial block contributes only `last_block_valid_bytes` to each scan. Its suffix is neither model payload nor
timed access. The physical data order may be scattered, but logical traversal order and semantic lookup multiplicity are
frozen. Full blocks remain whole ownership units.

Decode `mixed` sets both bits. For every work unit, each worker processes layers in increasing order. Inside a layer it reads its
weight shard, then processes every batch sequence's K/V append and visible-history reads before moving to the next
layer. Workers share one synchronized task start, but there is no synthetic global barrier at every layer. Thus mixed is
one layer-interleaved workload, not a post-hoc sum of separately timed weight and KV passes.

Metal contiguous decode preserves the same logical scenario traffic and layer/batch/token/head/dimension byte order,
but each scenario has a separately compiled entrypoint. One grid-stride workload dispatch owns all `T` work units for a
task. Threads cooperatively visit exact 16-byte prefixes plus bounded scalar tails and split accesses at W/K/V segment
boundaries. Each thread reads the disjoint slice it wrote for current-token append, so no grid-wide or cross-threadgroup
barrier is introduced. Grid width derives from the selected pipeline's runtime `threadExecutionWidth` and
`maxTotalThreadsPerThreadgroup`, not from an SoC-specific constant.

Metal paged decode preserves the three semantic visits above and assigns each layer/batch owner to exactly one
threadgroup under a cyclic grid-stride schedule. At every visit one named lane loads the table through
`device const volatile uint*`, publishes the physical ID in threadgroup memory, and executes
`threadgroup_barrier(mem_flags::mem_threadgroup)` before address construction. The append lookup is shared by its paired
K/V writes; K and V scans issue separate lookups, giving exactly `L * B * (2 * N + 1)` lookup evidence for each
KV-active work unit independent of threadgroup count. Segment/block/address selection depends on the loaded physical
ID. The timed checksum non-separably mixes logical table index, physical ID, append/K-read/V-read kind, and work-unit
ordinal, so swapping equal-multiplicity non-tail IDs is detectable. Excluded post-validation checks both append bytes
and terminal-block padding canaries.

For contiguous prefill KV work, every operation/layer/batch owner first writes all of its token ranges in increasing
logical order. It then visits query tiles in increasing order; for each tile it scans every owned range intersecting the
causal prefix for K, then repeats those ranges for V. Only after all tiles does it advance to the next descriptor.
Write happens before every read by the same owner, and owners never read one another's records, so no global
worker-per-layer barrier is required. Mixed reads its applicable layer weight shard before that layer's prefill work.

Paged prefill uses the same logical order with whole-block assignments. For each operation/layer/batch owner it first
loads each owned logical block's physical ID and populates all valid prompt K/V bytes in that block. It then visits tiles
in increasing order, independently loading and scanning each owned K block fragment intersecting the exact causal
prefix, followed by the corresponding V fragments. A terminal fragment stops at the tile end or prompt end and never
touches suffix padding. Mixed reads the applicable layer weight shard before the paged-prefill block work.

## Initialization, append pattern, and checksums

Preparation writes every weight byte and every requested K/V physical byte exactly once, pre-touching the full mapped
working set. Contiguous resources use `llm-buffer-pattern-v1`; paged K/V use
`llm-paged-physical-buffer-pattern-v1`, whose bytes depend on pool, physical block ID, and physical offset. This makes a
wrong table entry or address observable even when it visits the same multiset of blocks. Paged suffix padding is filled
with a canary. Preparation accumulates static references while writing, avoiding a second full-pool read.
Initialization,
table construction/protection, mapping page faults, reference construction, and canary setup occur before timed work.
Before every paged decode task, the executor restores only the mutable current-token K/V append slots to their physical
initialization pattern. This allocation-free reset is outside the timed interval; it does not rewrite history blocks or
suffix-padding canaries. Paged prefill instead rewrites every owned prompt byte during each timed operation.
The planner uses shared SplitMix64 derivation with frozen domains to derive separate weight/K/V buffer seeds,
permutation seed, and scenario seeds from one base seed; schema 1 stores exact seeds as decimal strings.

The initialization pattern treats each mapping as a zero-based stream of little-endian 64-bit words. For mapping word
index `i` and that mapping's domain-separated buffer seed:

```text
word_i = buffer_seed + 0x9E3779B97F4A7C15 * (i + 1) mod 2^64
```

A final 1–7-byte mapping tail takes the low little-endian bytes of the next word. A worker span that begins inside a
word observes the corresponding canonical mapping bytes rather than restarting the generator. With seed zero, words
zero and one are `0x9E3779B97F4A7C15` and `0x3C6EF372FE94F82A`.

Decode's current-token K/V append uses `llm-kv-append-affine64-v1`. The deterministic 64-bit word depends on the
scenario seed, task-local step, layer index, batch-sequence index, record-word index, and K/V domain. The dedicated
ARM64 kernel
uses ordinary temporal `stp`/`str` stores, including a bounds-safe tail; it does not use a non-temporal `stnp` hint.

The exact little-endian append word is:

```text
word = scenario_seed
     + 0x9E3779B97F4A7C15 * (task_local_step + 1)
     + 0xBF58476D1CE4E5B9 * (layer + 1)
     + 0x94D049BB133111EB * (batch_sequence + 1)
     + 0xD6E8FEB86659FD93 * (record_word_index + 1)
     + buffer_domain
     mod 2^64

buffer_domain(K) = 0x4B4B4B4B4B4B4B4B
buffer_domain(V) = 0x5656565656565656
```

The task-local step restarts at zero for every warmup, calibration, and measurement task. A partial worker append span
uses the canonical record byte offset; every partial fragment writes only the corresponding little-endian bytes without
widening its bounds.

Every read contributes to an observable `llm-read-checksum-v1` state. For each worker, weight, K, and V components each
contain two 64-bit states plus exact bytes read and span count. Cold-path code independently derives the expected values
from frozen descriptors, initialization references, append formula, scenario, and step count. After the timer stops, it
compares every worker/component tuple and folds expected and actual results in stable worker/weight/K/V order.

For component domain `D`, its initial state is:

```text
state_a = 0x243F6A8885A308D3 xor D
state_b = 0x13198A2E03707344 + D mod 2^64

D(weight) = 0x5745494748545F31
D(K)      = 0x4B5F524541445F31
D(V)      = 0x565F524541445F31
```

Within each non-empty read span, 64-bit little-endian words are added alternately into `span_even` and `span_odd`; a
1–7-byte tail is zero-padded as that span's last word. Parity restarts at word zero for each span. With zero-based
component-local span ordinal `o`:

```text
state_a = rotl64(state_a + span_even
                 + 0x9E3779B97F4A7C15 * (o + 1), 17)
state_b = rotl64(state_b + span_odd + span_bytes
                 + 0xD6E8FEB86659FD93 * (o + 1), 29)
exact_bytes_read += span_bytes
span_count += 1
```

Span accumulation uses modular sums rather than XOR, so an even number of identical work units does not erase evidence.

The run fold begins at `(0x6A09E667F3BCC909, 0xBB67AE8584CAA73B)` and visits worker zero's weight/K/V tuples, then
worker one's, and so on. Empty components retain their versioned initial state and still participate. For tuple ordinal
`q`:

```text
run_a = rotl64(run_a + state_a
               + 0x9E3779B97F4A7C15 * (q + 1), 23)
run_b = rotl64(run_b + state_b + exact_bytes_read
               + 0xD6E8FEB86659FD93 * span_count
               + 0x94D049BB133111EB * (q + 1), 41)
```

All checksum arithmetic is modulo `2^64`; byte/span counters use checked arithmetic. Representative independent golden
vectors are:

| Contract | Input | Exact result |
|---|---|---|
| append K | seed/step/layer/batch/word all zero | `0x149454E56105BC97`, little-endian `97bc0561e5549414` |
| append V | seed/step/layer/batch/word all zero | `0x1F9F5FF06C10C7A2`, little-endian `a2c7106cf05f9f1f` |
| read span | bytes `00 01 ... 12` | even `0x0706050403141210`, odd `0x0F0E0D0C0B0A0908` |
| weight state | preceding 19-byte span | `a=0x451AA0ABCC0E316F`, `b=0x37A51B246A0ABBE7`, bytes 19, spans 1 |

This checksum is workload-liveness and bounds evidence, not a cryptographic integrity primitive.

Paged execution uses the separate `llm-paged-read-checksum-v1` contract. Its timed accumulator binds each semantic visit
to the logical table index, loaded physical ID, visit kind (paired write/append, K scan, or V scan), and work-unit ordinal in one
non-separable mix. Summing physical IDs alone is prohibited because every permutation has the same ID sum. A test that
swaps two non-tail blocks with equal read multiplicity must therefore produce a mismatch. An independent bounded scalar
oracle computes the cold-path expected value without calling the assembly helper or rereading a multi-GiB pool.
Post-validation checks the logical decode current-token K/V append or prefill final-ordinal samples at their resolved
physical locations and every last-block padding canary.

Prefill uses its versioned full-prompt affine64 write pattern. Every K/V word binds scenario seed, operation
ordinal, layer, batch, logical token, record-word index, and K/V domain. Timed checksum agreement proves the expected
contents and visit multiplicity across all T operations, including every tile-read visit; it does not by itself prove
event order. Source/disassembly audit establishes the locked owner-local order: ascending prompt population followed
for every increasing tile by the complete K prefix and then the complete V prefix. In paged prefill, population and
prefix fragments additionally bind the table index and loaded physical ID and stop at exact partial-block boundaries.
The independent oracle does not call the assembly helper. Excluded post-validation checks each owner's deterministic
first/middle/last canonical-word samples, including bytes clipped to owner boundaries, against final operation ordinal
`T-1`; it does not reread every prompt record.

Only exact checksum agreement, successful phase/layout-specific post-validation (decode current-token append and paged
padding canaries, or prefill final-ordinal representative/boundary samples), complete worker lifecycle, successful
kernel status, and a finite positive elapsed time permit `measured` status. A mismatch is terminal invalid evidence and
is not retried. Checksum and fold traffic is validation evidence, not part of the logical payload numerator.

Metal checksum uses `llm-metal-dual-mod32-v1`, with separate W/K/V lanes, 32-bit modulo arithmetic, domain separation
for backend/phase/layout/scenario/layer/batch/work-unit/logical offset, and commutative threadgroup reduction. Its
independent bounded CPU oracle derives the expected accumulator without rereading multi-GiB resources or calling the
production kernel helper. For each KV-bearing Metal task, excluded post-validation checks every byte of every final K/V
append record before accepting the task. A timed checksum or append mismatch invalidates the current task and prevents
the next task; there is no retry and no numeric comparison with the CPU checksum algorithm.

## Timing boundary and backend lifecycle

A CPU scenario task follows this boundary:

1. validate immutable descriptors and derive expected checksums;
2. create the complete worker team and prepare best-effort worker QoS;
3. wait until every worker reaches the start gate;
4. execute `dsb ish; isb`, start `HighResTimer`, and release the gate;
5. let each worker call the layout-specific ARM64 kernel once for all frozen work units;
6. stop the timer at last-worker completion;
7. join workers and validate/fold checksums outside the elapsed interval.

Paged table loads, ID-dependent address formation, model reads/writes, and timed checksum accumulation are inside the
primary elapsed interval. Thread creation, QoS calls, allocation, initialization/pre-touch, permutation generation,
table validation/hash/protection, preparation page faults, descriptor validation, expected-oracle generation,
task-local paged append-slot restoration, warmup/calibration, joins, checksum validation, decode append/padding
validation, prefill final-ordinal representative/boundary-sample validation, aggregation, console, JSON, and checkpoint
writes are outside. Normal cache state is not flushed between tasks or loops, so the methodology is explicitly
warm/cacheable and cache-inclusive.

For Metal, each warmup, calibration, and measurement task instead follows:

1. derive the bounded expected accumulator and commit one excluded reset command buffer;
2. encode one workload dispatch in one explicitly serial encoder and one timed command buffer;
3. commit, wait for terminal completion, and reject command-buffer error;
4. read `GPUStartTime` and `GPUEndTime` only after completion and require finite, positive, increasing timestamps;
5. compare timed dual-mod32 W/K/V accumulators;
6. commit one excluded post-validation command buffer and validate every byte of every final K/V append record before
   task acceptance for a KV-bearing scenario.

The authoritative duration is `GPUEndTime - GPUStartTime`. It includes GPU-side scheduling within the command-buffer
execution window, argument/segment indirection, one workload dispatch, kernel work, reduction, and timed status
atomics. It excludes host encoding, pre-commit time, queue wait before GPU start, post-GPU host wait, reset, oracle,
post-validation, initialization, warmup classification, aggregation, and serialization. A host steady-clock submit/wait
envelope is diagnostic. Queue delay remains null unless captured in the GPU timestamp clock domain. Metal worker and
worker-QoS fields are null with applicability false.

## Calibration, frozen work, order, and statistics

When `--iterations T` is present, the runner first validates and atomically freezes all three scenario plans with
exactly `T` work units. It then runs one excluded same-shape warmup for each frozen plan in canonical `weights_only`,
`kv_only`, `mixed` order. No measured loop begins until all three frozen warmups have succeeded.

When iterations are omitted, the runner resolves each scenario independently in canonical order. The initial pilot
count covers at least 8 MiB of accounted work when scenario guardrails permit. A same-shape warmup at that exact count
precedes the timed pilot. Each subsequent correction candidate is timed without a general extra warmup. If a candidate
first reaches the irreducible one-work-unit shape and no one-work-unit warmup has already run, the runner performs
exactly one same-shape confirmation warmup before the timed one-work-unit confirmation. It retains the last accepted
candidate without starting measurements and performs at most the configured two corrections outside the inclusive
100–250 ms intended window.

Only after all three scenarios have resolved does the runner atomically freeze their plans. It then runs one excluded
same-shape warmup for each frozen plan in canonical scenario order. Measured loops start only after all three warmups
succeed, and reuse those plans without recalibration.

The corresponding excluded-attempt purpose tokens are `calibration_shape_warmup`, `pilot`, `correction`,
`single_unit_confirmation_warmup`, `single_unit_confirmation`, and `frozen_measurement_warmup`. An explicit-work run
records only `frozen_measurement_warmup` for each scenario.

The one-billion-work-unit and 64 GiB accounted-byte ceilings apply per scenario task. For every scenario:

```text
accounted_bytes_per_work_unit =
  effective_model_payload_bytes_per_work_unit
  + layout_metadata_read_bytes_per_work_unit
planned_task_accounted_bytes = planned_work_units * accounted_bytes_per_work_unit
completed_task_accounted_bytes = completed_work_units * accounted_bytes_per_work_unit
```

The contiguous profile has zero layout-metadata bytes. Paged KV-only and mixed use the exact lookup metadata calculated
above, while weights-only remains zero; metadata constrains calibration and explicit-work admission without entering the
GB/s numerator. One exact work unit may legitimately exceed the duration window and is retained with an
`above-target-single-work-unit` quality token only after a real timed one-work-unit attempt. A later slow or fast
measurement remains evidence rather than triggering performance-based retry.

Measured duration-quality values are `within-target-window`, `above-target-single-work-unit`,
`guardrail-limited-below-target`, `below-target-window`, or `above-target-window`; an untouched slot begins as
`not-run`. Every measured non-window value produces the corresponding `<scenario>-duration-<quality>` warning.

The base order is `weights_only`, `kv_only`, `mixed`; loop `i` rotates it by `i mod 3`. A complete block of three loops
gives every scenario one first, middle, and last position. Count need not be divisible by three, but comparative
conclusions require exact position balance.

Only checksum-valid `measured` records enter aggregates. Each scenario stores work-unit latency,
`synthetic_memory_work_units_per_second`, and `effective_model_payload_gb_s` values. One value is its own headline;
multiple values use median P50. Statistics include average, median, P90/P95/P99, sample standard deviation, CV, MAD,
minimum, and maximum. Fewer than three measurements is `insufficient-samples`; the effective-model-payload-GB/s CV
above 5% selects `noisy` and the
scenario's diagnostic high-CV warning. Values are not removed, winsorized, or retried because of performance.

Mixed payload fractions are exact byte fractions for weight read, KV read, and KV append write. The single mixed elapsed
time is not used to publish separate independent weight and KV bandwidths.

## Status, interruption, and checkpoints

Run statuses are:

```text
not_started, complete, partial, interrupted, unsupported, failed
```

`unsupported` is a terminal non-performance result. It reports the requested identity and stable reason, returns a
nonzero process status, and never silently executes another backend/profile.

Measurement statuses are:

```text
not_run, measured, interrupted, invalid, failed
```

Multiword status tokens use underscores. Multiword stable reason codes and duration-quality tokens use hyphens. Each
unavailable record carries a reason code; unavailable observed metrics and checksum validity serialize as JSON null,
never numeric zero.

If the backend call throws before returning evidence for a measurement or excluded task, nested `execution.status` is
`unavailable`, its reason is the runner-exception token, and absent worker-lifecycle, QoS, elapsed-time, and checksum
values are null. For a `not_run` measurement, the top-level successful/failed QoS-worker counts are also null. Neither
case may be interpreted as zero workers, a zero duration, or a successful checksum.

Interruption uses task-level completion-wins semantics. Stop is checked between complete backend tasks, not inside the
ARM64 hot loop or between layer descriptors. A started task runs to normal completion or genuine failure; a valid current
measurement remains measured even if the signal arrived during it. Once stop is observed, no next task starts and all
remaining slots become interrupted/null. A real backend-task, timer, checksum, or checkpoint failure remains
authoritative over a simultaneous interruption.

The runner offers one logical checkpoint after every terminal scenario measurement and a distinct command-terminal
checkpoint. File targets serialize each with atomic replacement. Stdout targets preserve the same state transitions and
stop observations but build no intermediate payload and emit exactly one final document. A failed file checkpoint ends
the run, marks checkpoint failure, and is not retried at command terminal.

`results_complete` means every planned scenario measurement is measured. `scenario_order_balance_complete` requires
the complete realized position matrix to be balanced. `conclusions_valid` requires complete status/results, balanced
order, and no checkpoint failure. A count-one command can therefore finish every planned task and retain valid numeric
measurements while correctly reporting `conclusions_valid: false`.

## JSON schema 1

The old schema-1 producer shape was unpublished. The generic vocabulary below replaces its CPU/decode/step-specific
field names without compatibility aliases, a fallback reader, or a schema-version increment.

The required top-level schema keys are:

```text
schema_version, mode, backend, phase, kv_layout, methodology_version,
software, configuration, resolved_plan, backend_evidence, memory_budget,
calibration, measurements, aggregates, status, reason_code,
results_complete, conclusions_valid, interpretation
```

Top-level `backend`, `phase`, and `kv_layout` are canonical selectors. `methodology_version` is derived exactly as
`llm-memory-v1-<backend>-<phase>-<layout>`. `configuration` preserves exact argv plus `resolved_sources`; a default is
recorded as `default`, not as fabricated argv. Additional diagnostic, interruption, checkpoint, loop-order, checksum,
environment, warning, and traffic-classification evidence may be present.

`resolved_plan` has four required ownership groups:

```text
resolved_plan.geometry
resolved_plan.layout
resolved_plan.resources
resolved_plan.component_identities
```

`geometry.decode` and `geometry.prefill` are object-or-null. Active decode populates integer
`decode.visible_context_tokens` and uses null prefill; active prefill does the reverse and never reuses context fields.
Its object has integer `prompt_tokens` and
`attention_query_tile_tokens` plus decimal-string `tile_count`,
`attention_prefix_token_visits_per_sequence`, `causal_token_pairs_per_sequence`, `logical_attention_pairs`, and
`logical_attention_fma_terms`. Decode-only crossover numerator/denominator/context, weight/KV-read ratio, and
context-classification fields are null for prefill.
`layout.kv_layout` is a string. Paged populates integer `kv_block_tokens`; decimal-string
`blocks_per_sequence`, `physical_blocks_per_layer`, `last_block_tokens`, `last_block_valid_bytes`,
`block_table_entries`, and `block_table_bytes`; plus `permutation_domain_uint64_hex`,
`permutation_seed_uint64_decimal`, `permutation_algorithm_version`, and `permutation_sha256`. These paged-only fields
are null for contiguous.
`resources` stores canonical decimal-string `weight_logical_bytes`, `k_logical_bytes`, `v_logical_bytes`,
`k_physical_length_bytes`, `v_physical_length_bytes`, `k_layout_padding_bytes`, and `v_layout_padding_bytes`;
`block_table_bytes` is decimal-string-or-null.

`component_identities` contains, in canonical fixed order:

```text
logical_profile_version
kv_layout_version
permutation_version
backend_executor_version
resource_abi_version
schedule_version
timer_policy_version
buffer_pattern_version
write_pattern_version
checksum_pattern_version
msl_revision
msl_source_sha256
```

Always-applicable values are strings. `permutation_version` is null for contiguous; MSL fields are null for CPU. The
serialized identity begins `llm-memory-components-v1` and appends every field in that order as `|key=<length>:<value>`
or `|key=null`.

`backend_evidence` contains both `cpu` and `metal` object-or-null branches. Exactly the selected backend is populated;
CPU profiles have a CPU object and `metal: null`. Within the CPU object, `prefill` is null for decode and
populated for either prefill layout. The prefill evidence fixes `cost_unit: "worker-cost"`, execution and scope identities,
descriptors per scenario/worker, decimal-string worker cost vectors, and scenario-specific decimal-string minimum,
maximum, and max-minus-min imbalance per work unit. Its `paged` sibling is populated for either paged phase, so paged
prefill has both objects. Metal profiles use `cpu: null` and populate `metal` with worker/QoS applicability false;
lifecycle status/reasons; device/family/unified/Tier-2/compiler/MSL/source/pipeline evidence; argument-buffer layout
probe; actual resource options, lengths, optional allocated sizes, and memory totals. Paged Metal evidence additionally
publishes whole-entry table segments, upload/readback validation, materialized permutation identity, combined K/V
padding, exact grid lookup count, and append/padding-canary validity. Runtime unsupported/failure diagnostics remain
bounded and separate from stable reason codes.
`memory_budget` separates allocation-time evidence into required
canonical decimal-string `resource_rounding_bytes`, `transient_peak_bytes`, `known_owned_peak_bytes`, and
`admitted_budget_bytes`. `calibration` owns excluded work-resolution attempts. `aggregates` contains measured-only
scenario values.

Every measurement exposes this stable backend-neutral accounting vocabulary:

```text
work_unit_kind
planned_work_units
completed_work_units
weight_read_bytes_per_work_unit
kv_read_bytes_per_work_unit
kv_write_bytes_per_work_unit
kv_write_kind
effective_model_payload_bytes_per_work_unit
layout_metadata_lookup_count_per_work_unit
layout_metadata_read_bytes_per_work_unit
accounted_bytes_per_work_unit
planned_effective_model_payload_bytes
completed_effective_model_payload_bytes
planned_layout_metadata_lookup_count
completed_layout_metadata_lookup_count
planned_layout_metadata_read_bytes
completed_layout_metadata_read_bytes
planned_task_accounted_bytes
completed_task_accounted_bytes
synthetic_work_unit_latency_seconds
synthetic_memory_work_units_per_second
effective_model_payload_gb_s
```

Measurement checksum evidence uses generic `write_pattern_version` and `checksum_pattern_version` keys; it does not
reuse decode-specific append terminology for prefill. Metal measurement evidence additionally publishes pipeline/grid,
raw GPU start/end and authoritative elapsed, host envelope and nullable same-clock queue delay,
command-buffer/encoder/dispatch counts, dual-mod32 W/K/V expected/actual values, and excluded append-validation state.

Decode uses `decode_step` and KV-bearing `current_token_append`; prefill uses `prefill_operation` and KV-bearing
`full_prompt_population`. `weights_only` uses `kv_write_kind: "none"` in both phases. Planned/completed work units are
integer numbers. All listed byte, lookup, metadata, and accounted fields are canonical decimal strings even when the
applicable value is zero. Derived elapsed/rate/statistic
values are finite JSON numbers only for successful measured evidence and otherwise null.

The field type never changes by backend, phase, layout, scenario, or magnitude. Schema/control indexes and validated
small inputs such as worker count and visible-context tokens are JSON integers bounded to the exact IEEE-754 integer
range. Potentially large bytes, capacities, block/table/lookup counts, token visits, causal pairs, FMA terms, seeds, and
checksums are canonical decimal strings. A non-applicable object or scalar is null; an applicable count of zero is
number `0` or decimal string `"0"` according to the field's fixed type. The string `"not_applicable"` is never used.

The complete document additionally retains:

- exact raw argv/output plus requested/default configuration;
- methodology/component identities, fixed policies, geometry, MHA/GQA/MQA metadata, and exact traffic/crossover inputs;
- requested, page-rounded committed, transient, known-owned, allowed, and available memory evidence;
- full mapping/init/descriptor or Metal segment/argument-buffer evidence and backend-applicable worker fields;
- base, buffer-domain, and scenario-domain seeds;
- immutable model and per-scenario work-plan identities, limits, exact work units, model payload, metadata, and accounted
  totals;
- excluded warmup/pilot/trial/correction attempts;
- planned/attempted/completed loop, measurement, work-unit, payload, and checkpoint counters;
- planned/realized cyclic order, every status-bearing measurement, checksum evidence, and measured-only aggregates;
- CPU/OS/cache/page/QoS plus start/end thermal, Low Power Mode, and physical-memory snapshots;
- quality-warning tokens and the non-inference/non-DRAM interpretation boundary.

The authoritative consumer acceptance predicate is exactly:

```text
mode == "llm_memory"
schema_version == 1
backend == requested_backend
phase == requested_phase
kv_layout == requested_kv_layout
methodology_version ==
  "llm-memory-v1-" + backend + "-" + phase + "-" + kv_layout
status == "complete"
results_complete == true
conclusions_valid == true
every planned measurement has status == "measured"
```

For active commands the requested values are CPU decode or prefill with contiguous or paged KV, or Metal decode with
contiguous or paged KV. Paged
acceptance and comparison must additionally match `G`, `N`, tail geometry, logical/physical/padding/table resources,
permutation
version/domain/resolved seed/hash, lookup/accounted bytes, worker schedule, descriptor/executor, timer, and checksum
identities. `unsupported`, `partial`, `interrupted`, `invalid`, and `failed` evidence is never accepted as performance.
Metal comparison additionally requires matching device capability, MSL revision/source hash, pipeline/grid identity,
exact segment geometry, and timer/checksum contracts. CPU and Metal samples are never pooled and their checksums are
not numerically comparable.
After the predicate, a consumer must still require the selected scenario metric's non-null value plus checksum and
quality conditions relevant to its conclusion. Process exit success alone is insufficient because graceful
interruption is an established success-return path.

The `interpretation` object always preserves the generic boundary: the workload is synthetic and memory-only;
effective GB/s is exact logical W/K/V payload divided by authoritative elapsed time, not measured physical DRAM
traffic; timed paged-table metadata is excluded from that numerator; prefill profiles do not perform Transformer compute
or predict TTFT; private Metal storage on unified memory is not separate VRAM; cache/SLC/DRAM residency is unmeasured;
and results with differing backend, phase, layout, phase geometry, paged geometry, methodology, or component identity
must not be pooled as one performance distribution.

## Console contract and quality warnings

The console identifies backend, phase/work unit, KV layout, phase geometry, warm/cacheable semantics, exact
weight/KV-read/KV-write bytes, and up to one measured headline per scenario. Decode prints context and crossover;
prefill prints P/Q/C, prefix visits, causal pairs, and logical attention/FMA audit counts. It uses phase-specific labels
such as `ms/decode step` or `ms/prefill operation`; JSON remains backend-neutral with
`synthetic_work_unit_latency_seconds`, `synthetic_memory_work_units_per_second`, and
`effective_model_payload_gb_s`. The report never uses bare `tokens/s` and states that effective model payload is not a
physical DRAM counter. A scenario without a headline does not receive a fabricated numeric console value; its status,
reason, and null observations remain in JSON.

Metal console output additionally identifies device/capability limits, W/K/V segment counts and capacity, Tier 2
argument-buffer length, committed/known-peak/admitted memory, scenario pipeline/grid, authoritative GPU elapsed time,
and checksum/append/canary validation state. It does not print CPU workers or worker QoS for Metal.

Paged console output additionally identifies `G`, `N`, final-block tokens/valid bytes, logical versus physical K/V,
layout padding, table entries/bytes, permutation version/seed/hash, and per-work-unit lookup/metadata/accounted values.
It explicitly states that table loads are timed while their four-byte metadata traffic is excluded from the effective
model-payload numerator. Checksum, phase-specific write, and padding-canary failure produces invalid evidence without
retry.

Report-level warnings include non-nominal environment, requested-but-unapplied main-thread QoS, worker QoS failure,
weight or KV working sets that may be cache-dominant, scenario durations outside their intended quality class, high CV,
and incomplete scenario-order balance. Warnings preserve measurements; they do not silently rewrite, filter, or retry
values.

The schema exposes those conditions with stable tokens:

```text
environment-not-nominal
main-thread-qos-not-applied
worker-qos-not-applied
weight-working-set-cache-dominant
kv-working-set-cache-dominant
<scenario>-duration-<quality>
weights_only-high-cv
kv_only-high-cv
mixed-high-cv
scenario-order-not-balanced
```

The first six are composed from the final report metadata and terminal measurements. The high-CV and order tokens are
retained from the runner. A main-thread QoS warning appears only when QoS was requested but not applied.

## Validation and comparison protocol

Correctness gates cover:

- independent formula golden vectors and overflow boundaries;
- exact CLI whitelist, layout/block rules, defaults/incompatibilities, and help isolation;
- pointer-free contiguous/paged layouts, block ownership, memory admission, and both descriptor ABI offset sets;
- atomic layout-specific allocation, table protection, full physical initialization/pre-touch, and cleanup;
- paged geometry/lookup goldens, deterministic permutation/hash, K/V append, independent checksum oracle,
  equal-multiplicity wrong-table mismatch, padding canaries, tails, bounds, and multi-step behavior;
- synchronized worker timing, startup cancellation, QoS evidence, timer/error containment, and AAPCS64 preservation;
- scenario-specific calibration, frozen plans, cyclic balance, aggregate population, interruption, and checkpoint
  precedence;
- schema identity, decimal exact integers, status/null rules, classification, interpretation, file/stdout transport, and
  executable CLI behavior.

The M4 paged real-device gate passed its exact Phase 10 filter 239/239, `make test-integration` 137/137, and
`make test-all` 967/967 GoogleTests plus 8/8 script examples without skips. It covered partial terminal blocks,
permutation-sensitive lookup rejection, padding-canary rejection, all three scenarios, and K/V execution across the
256 MiB segment boundary in a greater-than-256 MiB multi-segment run. The required bounded Apple7/M1 baseline
validation remains pending, so an Apple7-or-later capability pass is not evidence that the full public hardware
baseline is complete.

For a performance comparison, keep the exact command/model geometry, layout, paged `G`/physical geometry/table and
permutation identity when applicable, explicit-versus-automatic policy, frozen plan, seed, worker counts,
software/methodology, hardware, macOS, power/thermal state, and background load matched. Prefer a count divisible by
three, inspect CV and warnings, and retain both stdout/file payload and stderr transcript. A separate
real inference-engine run can be useful correlation evidence, but it is not part of this benchmark's correctness or
acceptance predicate.

## Change control

Any change to traffic formulas, context semantics, buffer sizing/layout, temporal append behavior, worker/layer order,
output-serialization peak admission, timing boundary, checksum observability, calibration/frozen-plan rules,
interruption/checkpoint lifecycle, or meaning of a reported field requires methodology and schema compatibility review.
Removing or renaming a field, changing its type, or changing its meaning requires a schema-version bump. Additive
evidence may remain schema 1 only when existing consumers can safely ignore it.

Runtime paged allocation/free lists, prefix sharing, sliding windows, growing context, chunked prefill, Metal prefill or
Metal execution outside the active decode preview, ANE execution,
model presets, quantization metadata, multiple weight passes, or KV replay factors other than one are separate
methodology features. The generic schema vocabulary does not activate them: each requires its own end-to-end
implementation gate, exact selector-derived methodology and component identity, public CLI/documentation update, and
compatibility review before it becomes a supported profile.

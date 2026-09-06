# Synthetic LLM Memory Profile

## Abstract

`memory_benchmark --llm-memory` is a versioned Apple Silicon synthetic memory benchmark for macOS 26 or later with a
generic backend/phase/KV-layout schema. CPU decode is active with contiguous or deterministic paged KV, and CPU
prefill is active with contiguous or deterministic paged KV. Capability-gated Metal decode and prefill are also active
with both layouts. These are eight active profiles. Every selected profile executes three scenarios derived from the
same explicit model geometry:

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
| Software version | Release provenance; not a schema or methodology selector |
| Mode | `llm_memory` |
| Backend | `cpu` or `metal` |
| JSON schema | `2` |
| Phase selector | `decode` or `prefill` |
| Work unit | `decode_step` or `prefill_operation` |
| Methodology | `llm-memory-v2-<backend>-<phase>-<layout>` |
| Model/scenario plan identity prefix | `llm-memory-work-plan-v1` |
| Component identity prefix | `llm-memory-components-v1` |
| Logical profile version | phase-specific decode or prefill profile identity |
| KV layout selector/version | `contiguous` / `contiguous_layer_batch_token_head_dimension`, or `paged` / `paged-uint32-block-table-full-blocks-v1` |
| Permutation version | null for contiguous; `splitmix64-fisher-yates-rejection-v1` for paged |
| Backend executor version | phase/layout-specific CPU ARM64 or Metal scenario-pipeline identity |
| Resource ABI | CPU descriptor identity or Metal Tier 2 argument-buffer ABI identity |
| Metal resource-plan identity prefix | `llm-metal-resource-foundation-v1` |
| Schedule version | CPU owner-local or Metal capped grid-stride schedule identity |
| Timer policy | CPU synchronized worker timer or Metal GPU command-buffer timestamp policy |
| Buffer pattern | backend/layout-specific contiguous or paged buffer-pattern identity |
| Write pattern | backend/phase-specific append or full-prompt affine identity |
| Checksum pattern | Backend/phase/layout-specific checksum identity; Metal uses profile-specific dual-mod32 variants |
| MSL revision/source SHA-256 | null / null for CPU; exact selected runtime source identity for Metal |
| Run policy | `llm-run-policy-bounded-loop-snapshots-v1` |
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

Schema-2 vocabulary includes `cpu|metal`, `decode|prefill`, `contiguous|paged`,
`decode_step|prefill_operation`, and `none|current_token_append|full_prompt_population`. Contiguous and paged are public
for both phases on both backends. The four Metal profiles never receive hidden fallback.

The prefill implementation resolves checked tile/prefix/payload formulas, versioned atomic CPU ownership evidence,
generated owner-local semantic event lists, and operation-ordinal checksum oracles. CPU contiguous prefill uses token-range ownership;
paged prefill uses block-exclusive weighted ownership plus a read-only uint32 table, deterministic permutation,
full physical K/V pools, a dedicated descriptor ABI, and a separate ARM64 executor. Metal prefill uses
scenario-specialized MSL pipelines, one in-kernel weight pass per operation when weights are active, full-prompt K/V
writes, and tiled-prefix K-then-V scans. Its paged profile additionally uses exact `N + 2*M` lookup accounting,
cyclic one-threadgroup block ownership, segmented table/K/V addressing, and terminal-padding validation.

It intentionally excludes:

- GEMV/GEMM/FMA, dequantization, RoPE, softmax, layer normalization, activation, and scratch traffic;
- tokenizer, model loader, framework scheduler/dispatch, kernel fusion, and compute-memory overlap;
- MLX, llama.cpp, Core ML, ANE execution, and GPU execution outside the defined Metal kernels;
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
                            default cpu; both backends accept both phases and layouts
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
exclusive. All eight backend/phase/layout combinations are active. CPU and Metal each accept decode and
prefill with contiguous or paged KV. Metal rejects explicit `--threads`, performs no CPU worker detection, and never
falls back to another profile.

Output targets follow the shared process contract:

- omitted or empty output means console only;
- exact `--output -` reserves stdout for one final schema 2 document and routes the post-parse human transcript to
  stderr;
- every other non-empty raw token is a file, including `./-` and flag-shaped names;
- file output uses atomic `<target>.tmp` replacement every `K=max(1,ceil(count/8))` completed loops plus command terminal;
- stdout preserves task-boundary stop observations but performs no intermediate snapshot preparation or serialization.

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
`logical_attention_fma_terms = logical_attention_pairs*d_h`. These values are not payload and no FMA is executed. Schema 2 places them in `resolved_plan.model_context.prefill`,
with an individual `*_reason_code`: available decimal string/`valid`, or null/`arithmetic-overflow`. Theoretical overflow
does not reject otherwise representable byte work; byte/prefix/lookup/allocation overflow still rejects it. Available
identity values retain their numeric encoding; unavailable values use `unavailable:arithmetic-overflow`. Query heads
`h_q` affect theoretical context, whereas query tile `Q` changes actual prefix-read traffic.

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

Schema 2 preserves the numerator and denominator as exact decimal strings plus a floating estimate. For the vector
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
and scope identities. Frozen identities are charged once. Canonical scenario-plan identities scale with the bounded maximum retained calibration shapes plus frozen plans,
independently of measured loop count. Actual measurement records/checksums still scale with planned loops. Each variable-length addition is checked, and preliminary and finalized-plan
estimates use the same canonical identity-size formula. Omitted or empty output adds no serialization reserve; normal
console-only execution does not serialize a schema document.

Schema ownership separates immutable resource geometry from allocation/admission evidence. `resolved_plan.resources`
contains canonical decimal-string logical weight/K/V lengths, physical K/V lengths, layout padding, and nullable
block-table bytes. Top-level `memory_budget` contains canonical decimal-string `resource_rounding_bytes`,
`transient_peak_bytes`, `known_owned_peak_bytes`, and `admitted_budget_bytes`, along with any additive detailed estimate
evidence. Active contiguous profiles have identical K/V logical and physical lengths, zero layout padding, and a null
block-table resource. The sampled available-memory value and derived admitted budget remain runtime evidence rather
than resource, execution, model, or frozen-plan identity inputs. Identical fixed-seed work therefore retains its
identity when only the admission sample changes.

Every active Metal profile owns private/tracked W/K/V buffers, plus shared/tracked Tier 2 argument-buffer and
status/checksum storage.
Contiguous pools use exact canonical segments no larger than 256 MiB. Paged K/V segments contain whole blocks, and the
private uint32 table uses whole-entry segments; transient shared staging uploads one bounded table segment at a time.
No segment-capacity slack or paged suffix padding enters payload. Each pool has at most 256 argument-buffer slots.
Runtime admission uses actual encoder length/alignment, `maxBufferLength`, and each resource's `allocatedSize` when
available. Runtime admission requires Apple7-or-later capability, unified memory, Tier 2 argument buffers,
`maxBufferLength >= 256 MiB`, and successful MSL 2.3 compilation of the common foundation pipelines, the selected
profile's three scenario pipelines, and its layout-specific validation/probe pipelines.

Runtime evidence binds the canonical embedded MSL revision and its exact source SHA-256 without duplicating either
value in this document. Every selected profile records its scenario-pipeline, parameter-ABI, and layout-probe
identities. Validation pipelines are selected and compiled internally; evidence records their outcomes rather than a
separate pipeline label. Metal paged prefill records executor `llm-metal-executor-v1-prefill-paged`, schedule
`llm-metal-prefill-paged-cyclic-block-owner-grid-stride-v1`, timer
`metal-command-buffer-gpu-start-end-v1`, buffer pattern
`llm-paged-physical-buffer-pattern-v1`, write pattern `llm-metal-prefill-paged-full-prompt-affine32-v1`, and checksum
`llm-metal-paged-prefill-dual-mod32-lookup-address-mix-v1`.

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

Metal paged decode preserves the three semantic visits above and assigns each layer/batch/logical-block owner to
exactly one threadgroup under a cyclic grid-stride schedule. At every visit one named lane loads the table through
`device const volatile uint*`, publishes the physical ID in threadgroup memory, and executes
`threadgroup_barrier(mem_flags::mem_threadgroup)` before address construction. The append lookup is shared by its paired
K/V writes. Each block owner issues its separate K and V scan lookups, giving exactly `L * B * (2 * N + 1)` lookup
evidence for each KV-active work unit independent of threadgroup count. Segment/block/address selection depends on the
loaded physical ID. The timed checksum non-separably mixes logical table index, physical ID, append/K-read/V-read kind,
and work-unit ordinal; the selected equal-multiplicity non-tail ID swaps in the mutation tests are detectable.
For KV-bearing Metal tasks, excluded post-validation checks both append bytes and terminal-block padding canaries.

For contiguous prefill KV work, every operation/layer/batch owner first writes all of its token ranges in increasing
logical order. It then visits query tiles in increasing order; for each tile it scans every owned range intersecting the
causal prefix for K, then repeats those ranges for V. Only after all tiles does it advance to the next descriptor.
Write happens before every read by the same owner, and owners never read one another's records, so no global
worker-per-layer barrier is required. Mixed reads its applicable layer weight shard before that layer's prefill work.

Metal contiguous prefill preserves the same complete-population-before-prefix-read contract in one
scenario-specialized grid-stride dispatch. For each operation, the weights path makes one pass over the active weight
bytes, not one pass per prompt token or query tile. In KV-bearing entrypoints each lane populates its disjoint slices
across every prompt K/V record before reading those slices, then scans each tile's complete K prefix before its complete
V prefix. No grid-wide barrier is required because a lane reads only the bytes it wrote. The dispatch loops over all
`T` operations and uses exact vector prefixes, scalar tails, and segment-boundary splits. The commutative dual-mod32
checksum supplies a bounded consistency witness, with the collision limits described below. A separate source-bound
audit checks this loop nesting for the recorded MSL build; it is not a trace collected from GPU execution.

CPU paged prefill uses the same logical order with weighted whole-block assignments. For each operation/layer/batch
owner it first loads each owned logical block's physical ID and populates all valid prompt K/V bytes in that block. It
then visits tiles
in increasing order, independently loading and scanning each owned K block fragment intersecting the exact causal
prefix, followed by the corresponding V fragments. A terminal fragment stops at the tile end or prompt end and never
touches suffix padding. Mixed reads the applicable layer weight shard before the paged-prefill block work.

Metal paged prefill retains that order but assigns row-major `(layer,batch,logical_block)` owner ordinals to
threadgroups with a deterministic cyclic grid stride rather than CPU weighted repartitioning. One named lane performs
each volatile uint32 table load and publishes the physical ID before the owner threadgroup addresses the block. Prompt
population contributes `N` paired lookups and the tiled scans contribute `M` K-prefix plus `M` V-prefix lookups per
layer/batch pair, so each KV-active operation has exactly `L * B * (N + 2 * M)` loads independent of grid size. The
source-hash-bound semantic event enumeration checks full-prompt population before tile-ordered K-then-V reads
in the described implementation, rather than recording those events from GPU execution. Grid evidence
reports exact per-threadgroup accounted-byte vectors and their minimum, maximum, and imbalance without claiming
weighted balance. Every Metal `weights_only` grid reports the same exact weight-vector grid-stride cost evidence;
contiguous KV-bearing grids leave threadgroup-cost evidence unavailable. Excluded validation checks final-ordinal
full-prompt samples and applicable suffix-padding canaries.

## Initialization, K/V write patterns, and checksums

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
permutation seed, and scenario seeds from one base seed; schema 2 stores exact seeds as decimal strings.

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

The intended read path accumulates each read into an observable `llm-read-checksum-v1` state. For each worker, weight, K, and V components each
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
non-separable mix. Summing physical IDs alone is prohibited because every permutation has the same ID sum. The maintained tests
show a mismatch for selected two-entry non-tail swaps with equal read multiplicity; this is not a guarantee for all
possible table mutations or modulo-arithmetic collisions. An independent bounded scalar
oracle computes the cold-path expected value without calling the assembly helper or rereading a multi-GiB pool.
Post-validation checks the logical decode current-token K/V append or prefill final-ordinal samples at their resolved
physical locations and every last-block padding canary.

CPU prefill uses its versioned full-prompt affine64 write pattern. Metal contiguous prefill uses
`llm-metal-prefill-contiguous-full-prompt-affine32-v1`, while Metal paged prefill uses
`llm-metal-prefill-paged-full-prompt-affine32-v1` with checksum
`llm-metal-paged-prefill-dual-mod32-lookup-address-mix-v1`. Every K/V word binds scenario seed, operation ordinal, layer,
batch, logical token, record-word index, and K/V domain. Timed checksum agreement checks the expected accumulator
across T operations, including planned tile-read contributions. It does not establish every executed load, content value, visit multiplicity, or event order.
CPU source/disassembly and Metal entrypoint/source-hash audits qualify the implementation of the specified order: one
weight pass per operation when weights are active, complete prompt population for KV-bearing scenarios, then for every
increasing tile the complete K prefix followed by the complete V prefix. In paged prefill, population and
prefix fragments additionally bind the table index and loaded physical ID and stop at exact partial-block boundaries.
The independent oracle does not call the assembly helper. Excluded post-validation checks each owner's deterministic
first/middle/last byte samples, including samples at canonical word and owner boundaries, against final operation ordinal
`T-1`; it does not reread every prompt record.

CPU contiguous-decode checks every byte of both K/V append records for all layers and batches
against the final task-local ordinal `T-1`, after timer stop and all worker joins. A mismatch yields
`decode-post-validation-failed`: the invalid attempt is retained with null rates and excluded from
aggregates. The timed workload and checksum algorithm are unchanged. KV-write evidence applies only
to KV-bearing scenarios; schema 2 retains independent write, structure and applicable padding verdicts with the
existing sampling limits. No new memory scan or stronger fault-coverage claim follows from naming these checks. CPU paged decode weights-only checks its append slots and padding; CPU paged prefill weights-only checks padding,
not valid prompt contents.
See the [CPU final-state evidence contract](API.md#result-schemas-and-completion) for field and null semantics.


Only exact checksum agreement, successful phase/layout-specific post-validation (decode current-token append and paged
padding canaries, or prefill final-ordinal representative/boundary samples), complete worker lifecycle, successful
kernel status, and a finite positive elapsed time permit `measured` status. A mismatch is terminal invalid evidence and
is not retried. Checksum and fold traffic is validation evidence, not part of the logical payload numerator.

Metal checksum uses profile-specific dual-mod32 identities, with separate W/K/V lanes, 32-bit modulo arithmetic,
domain separation for backend/phase/layout/scenario/layer/batch/work-unit/logical offset, and commutative threadgroup
reduction. Its independent bounded CPU oracle derives the expected accumulator without rereading multi-GiB resources or calling the
production kernel helper. For each KV-bearing Metal task, excluded phase-neutral `kv_write` validation checks the
applicable write before accepting the task. Decode validates its current-token K/V record. Prefill validates
representative and boundary byte samples from the full-prompt population against the final operation ordinal `T-1`;
the paged profile also checks applicable suffix-padding canaries. It does not reread every prompt record. A timed
checksum, K/V-write, or applicable padding mismatch invalidates the current task and prevents the
next task; there is no retry and no numeric comparison with the CPU checksum algorithm.

## Checksum fault model and evidence limits

Three kinds of evidence have different owners: the immutable work plan describes intended geometry, payload, and
visits; a task records runtime checksum, timing, lifecycle, and post-validation observations; kernel qualification
checks a particular build using independent oracles, ABI/layout goldens, source/disassembly, and real-device tests.
A source hash or generated semantic event list is a build/semantic change guard, not a GPU execution trace. Exact-byte
and lookup counters are programmed observations or plan-derived completion quantities, not physical DRAM counters.
Checksum equality is necessary for acceptance, but is not an exhaustive content or address proof.

The following matrix covers CPU/Metal × decode/prefill × contiguous/paged. **O** means an observed, specifically
bounded test case; **E** means detection is conditional on the checksum/validator and has no corresponding injected
kernel-fault proof here; **S** identifies a known blind spot. **—** is inapplicable. Multiple marks distinguish a
selected detected mutation from other undetected mutations in the same broad class. They are not coverage percentages.

| Fault | CPU decode contiguous | CPU decode paged | CPU prefill contiguous | CPU prefill paged | Metal decode contiguous | Metal decode paged | Metal prefill contiguous | Metal prefill paged |
|---|---|---|---|---|---|---|---|---|
| Skipped/duplicated read or wrong work-unit count | O W; E other reads | O W; E KV | O W; E KV | O W; E KV | E | E | E | E |
| Wrong address or equal-valued substitution | S equal value | O T; S other substitutions | S equal value | O T; S other substitutions | S equal value | O T; S other substitutions | S equal value | O T; S other substitutions |
| CPU within-span 32-byte swap or same-parity +1/−1 | S W | S W | S W | S W | — | — | — | — |
| Metal content permutation or cancellation with the same visits | — | — | — | — | S H | S H | S H | S H |
| Wrong paged table ID / lookup | — | O T; E skipped lookup | — | O T; E skipped lookup | — | O T; E skipped lookup | — | O T; E skipped lookup |
| Wrong K/V append or prefill final state | O A | O A | O P; S unsampled post-checksum writes | O P; S unsampled post-checksum writes | E A | E A | E P; S unsampled post-checksum writes | E P; S unsampled post-checksum writes |
| Padding or other extra write | E outside checked append | O C; E elsewhere | E outside samples | O C; E elsewhere | E outside checked append | O C; E elsewhere | E outside samples | O C; E elsewhere |

- **W — shared CPU weight span only.** In all four real ASM entrypoints, the 64-byte word vector
  `11,23,37,43,59,67,79,83` retains its checksum after swapping its two 32-byte halves or adding one to word zero and
  subtracting one from word two, at T=1 and T=2. Both mutations preserve the even/odd word sums. A one-bit change is
  detected. Separate invocations against the original independent 64-byte × T=2 oracle detect a 32-byte range removal,
  a duplicated 32-byte range, and T=1/T=3. These are selected descriptor/work-count mutations, not tests of every
  possible skipped machine load. They do not establish a whole KV/backend acceptance collision. CPU prefill KV uses
  logical-word parity sums with its own visit/lookup rules; the weight-span test does not replace that separate oracle.
- **H — actual Metal shared helper, not full-backend acceptance.** The test adds a small test-only entrypoint to the
  unchanged embedded MSL and calls `mix_checksum_word` on four words. For fixed visits, modulo 2^32:
  `a=sum(value)+sum(domain)` and
  `b=0x9e3779b1*sum(value)+0x85ebca77*sum(word_index)+0x7feb352d*sum(domain)`.
  Thus both lanes depend on the same content sum; they are not an independent 64-bit content proof. The original
  `11,23,37,43`, its reversal, and `12,22,37,43` each produce `(49596,3847175070)` for domains `12345+17*i`.
  The one-bit control `10,23,37,43` produces `(49595,1192739309)`. This shared-helper result applies to its use by
  all four profiles with the same visit/domain contributions; additional lookup, write, and lifecycle checks can
  still reject a complete task. Content terms and separately added address terms do not establish arbitrary
  content-to-address binding.
- **T — selected real table mutations.** CPU decode/prefill tests swap two table IDs while preserving ID multiplicity
  and using identical initial physical block contents. The actual ASM K/V checksums differ from the independent
  correct-table oracle even when byte/span counts match. Metal decode/prefill tests swap equal-read-multiplicity
  nonterminal IDs after generating the canonical expected summary and before uploading the actual GPU table.
  The real workload dispatch returns a checksum mismatch. This does not prove detection of every table permutation,
  lookup omission, or modulo collision.
- **A/P — append versus sampled prefill state.** CPU contiguous decode checks all bytes of both final append records,
  across layers/batches after timer stop and worker joins. Its K/V first/last-byte corruptions retain a valid timed
  checksum but fail post-validation. CPU paged decode also has real append-corruption coverage. CPU prefill tests
  corrupt checked canonical-word/boundary samples after the real kernel; Metal prefill checks representative/boundary
  locations per layer/batch sequence. Neither prefill validator scans every prompt record or every intermediate
  operation. A mutation confined to unsampled prompt bytes after the timed checksum has been collected is a known blind spot
  of the cold sampler on both backends; it cannot change that already collected checksum. Metal's forced `kv_write` result test is an acceptance hook, not an actual shader write-corruption test;
  hence E rather than O for that fault in the matrix.
- **C — bounded canaries.** CPU paged decode/prefill tests alter real final-state/padding memory after the kernel.
  Metal paged tests use a real blit write into K padding before post-validation. CPU paged decode weights-only checks its append slots and padding. CPU paged prefill weights-only checks
  structure and padding only, not unexpected writes to valid prompt contents. Metal weights-only does not
  evaluate KV-write or padding validation. Tail/guard-page, layout, and ABI tests additionally qualify the intended
  implementation; none is an all-memory post-execution scan.

The maintained examples are in [CPU kernel tests](../tests/test_llm_memory_kernels.cpp), notably
`AllCpuWeightSpansExposeBoundedParityCollisions`, `AllCpuWeightSpansDetectBoundedReadAndWorkUnitMutations`,
`PagedWrongSameMultiplicityBlockTableDoesNotMatchOracle`, and
`PagedPrefillWrongSameMultiplicityBlockTableDoesNotMatchOracle`; in
[CPU executor tests](../tests/test_llm_memory_executor.cpp) for append/prefill/padding corruptions; and in
[Metal helper tests](../tests/test_llm_metal_checksum.mm) for
`SharedAffineLanesExposeBoundedContentCollisions`. The
[Metal backend tests](../tests/test_llm_metal_backend.cpp) contain
`DecodePagedPermutationAndPaddingHooksAreDetectedIntegration` and
`PrefillPagedPermutationAndPaddingHooksAreDetectedIntegration`. Their real table/blit mutations differ from
`force_timed_checksum_mismatch` (host readback altered after execution) and `force_kv_write_validation_mismatch`
(forced validation boolean), which test result handling. Pure independent checksum-oracle goldens test arithmetic;
they are not GPU execution evidence. Algorithms and their existing identities remain unchanged by this fault model.

CPU LLM results retain the original Mach tick boundaries and timebase for duration reconstruction.
The build embeds compiler, flags, SDK/deployment target and Git provenance; the command hashes the
executable once before tasks. These fields bind available artifacts and do not constitute signed
execution attestation. See the [LLM process contract](API.md) for availability and numeric rules.
Python 3 is required to generate build provenance.

## Timing boundary and backend lifecycle

A CPU scenario task follows this boundary:

1. validate immutable descriptors and derive expected checksums;
2. create the complete worker team and prepare best-effort worker QoS;
3. wait until every worker reaches the start gate;
4. execute `dsb ish; isb`, start `HighResTimer`, and release the gate;
5. let each worker call the layout-specific ARM64 kernel once for all frozen work units;
6. stop the timer at last-worker completion;
7. join workers and validate/fold checksums outside the elapsed interval.

Each executor task validates its borrowed plan and materialized resources before oracle generation. The private
oracle calculation consumes that same checked input synchronously, without an intervening callback or mutation;
it does not repeat the descriptor walk. A separate public oracle call validates its own inputs. No validation verdict
is retained across tasks: plan identities alone do not make caller-owned structures immutable. Mutable K/V reset and
all applicable post-execution checks still run for each task.

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
6. commit one excluded post-validation command buffer and validate the phase-specific K/V write before task acceptance
   for a KV-bearing scenario.

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

Contiguous Metal prefill checks the exact lane-local serial range-helper count
`T * ((weight_active ? L : 0) + (kv_active ? 2 * L * B * (P + C) : 0))`; paged `weights_only` checks `T * L`, while
paged KV-bearing owner kernels report zero serial range visits. Metal prefill publishes the applicable count in grid
evidence and the 1,048,576 cap in methodology evidence. Scenario planning divides the cap by the exact per-operation count, so
explicit work and automatic calibration remain within it; runtime overflow and cap checks still run before the bounded
CPU oracle or GPU dispatch.

The contiguous profile has zero layout-metadata bytes. Paged KV-only and mixed use the exact lookup metadata calculated
above, while weights-only remains zero; metadata constrains calibration and explicit-work admission without entering the
GB/s numerator. One exact work unit may legitimately exceed the duration window and is retained with an
`above-target-single-work-unit` quality token only after a real timed one-work-unit attempt. A later slow or fast
measurement remains evidence rather than triggering performance-based retry.

Measured duration-quality values are `within-target-window`, `above-target-single-work-unit`,
`guardrail-limited-below-target`, `below-target-window`, or `above-target-window`; an untouched slot begins as
`not-run`. Every measured non-window value produces the corresponding `<scenario>-duration-<quality>` warning.

The base order is `weights_only`, `kv_only`, `mixed`; loop `i` rotates it by `i mod 3`. A complete block of three loops
gives every scenario one first, middle, and last position. This is **position balance**, not directed predecessor-pair
balance. With W=weights_only, K=kv_only and M=mixed, the finite three-loop stream is `WKM KMW MWK`.
Including loop boundaries, its transition counts are W→K=2, K→M=2, M→W=2, M→K=1, W→M=1, K→W=0.
Repeating the block adds K→W at its boundary. Count need not be divisible by three; a partial final block retains its
actual positions and transitions. `scenario_order_balance_complete` checks complete measured loops and equal nonzero
first/middle/last counts per scenario, without asserting carryover balance. A comparison policy may require position
balance, but producer correctness acceptance does not. The order depends on loop index, not the resolved seed.

The canonical W/K/M frozen-plan warmups occur once before loop zero. There is no runner-level same-scenario warmup
immediately before every measurement. Backend task-local reset, precondition, expected-witness construction and
post-validation still apply as documented above; they can affect the state seen by the next kernel. The initial warmup
prefix ends in M before measured W. Measured-order records do not enumerate those excluded tasks, calibration, or
host/output work. File checkpoints and stdout can produce different intertask gaps even with the same measured order.

Using all six permutations balances within-loop directed pairs, but the order of those loops and repeated-block
boundaries still matters. For example, `WKM WMK KWM KMW MWK MKW` adds M→W, K→K, M→K, W→M and K→M once each
at its five loop boundaries, plus W→W when the block repeats. It does not balance the whole execution stream.
Per-measurement conditioning would also add a same-scenario predecessor and change the prepared-state phenomenon;
it requires an explicit run-policy/methodology review rather than treating a higher rate as equivalent work under the
current preparation policy. The production default remains three cyclic loops and canonical initial frozen warmups.

Only `measured` records with accepted required timing, checksum and cold-check evidence enter aggregates.
One retained authoritative duration/work/payload sample feeds all three metrics through one
`accepted_measurement_ids` population. Each rate is derived before statistics: median(work/duration) need not equal
work/median(duration). Exact statistics use the existing shared linear interpolation and sample standard deviation
helper only at snapshots/terminal, with reusable extraction/sort/MAD workspaces. Raw sample console output does not
sort earlier samples. One sample is its own headline; otherwise the headline is median P50.

JSON retains average, median, P90/P95/P99, sample stddev, CV, MAD, min/max. Empty populations have null statistics;
n=1 has stddev=0 and positive-mean CV=0. `observed_cv_classification` is `insufficient-samples` for n<3, `undefined`
for undefined CV, `above-threshold` for payload CV strictly greater than 5%, otherwise `below-threshold`. Equality is
below-threshold; `cv_warning_threshold_pct` states the bound. These describe observed samples, not confidence levels
or reproducibility guarantees. Default console uses n/median/min/max/CV/MAD. Quality does not remove, winsorize or retry
samples and does not change correctness acceptance.

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

Files receive a snapshot after every Kth fully completed loop, where `K=max(1,ceil(N/8))` for N planned loops.
The overflow-safe calculation is `max(1,N/8 + (N%8 != 0))`. At most eight progress snapshots and one terminal snapshot
are written (normal maxima 4/7/9 for N=3/12/48). A successful terminal may be followed by one corrective failure
snapshot for a late command exception. Failed persistence is terminal and never retried at the final-write boundary.

At abrupt termination, at most `3K` truly completed attempts may be missing after the last successful snapshot,
including while its replacement is in progress. Unstarted tail placeholders do not count. Before the first snapshot
there may be no file; SIGKILL/crash cannot promise terminal output. Graceful SIGINT retains the completed prefix and
interrupted tail if terminal persistence succeeds. Atomic rename does not imply power-loss durability. Task-boundary
stop observations remain active even when snapshots are skipped. Stdout builds one terminal DOM, disabled output none.

`checkpoint_lifecycle` distinguishes actual writer entry/return from snapshot construction and logical task transitions.
`prior_file_writer_attempts` and `prior_successful_file_writes` have observation point
`before-current-snapshot-preparation`. `current_request` is `progress`, `terminal` or `late-command-error-correction`;
`current_persistence_success` is null, never predicted. Builder failure is not writer entry; stdout/disabled no-ops
are not persistent successes. No extra snapshot is written to count itself. An older preserved file cannot encode
future write/cleanup failure, so final collection retains actual process outcome and terminal context.

`results_complete` describes the fully measured planned population. `run_accepted` requires complete run status,
accepted required timing/checksum/cold-check/backend lifecycle and completion evidence, and no known command or
checkpoint error. Balance, count, CV, duration and environment are independent quality context. Count one can be
correct and accepted while position balance is false and sample classification is insufficient. A late command error
may preserve `results_complete: true` while status is failed and `run_accepted: false`.

## JSON schema 2

Schema 2 deliberately replaces the unpublished schema-1 shape and adopts v2 methodology selectors. There is no
compatibility alias or fallback reader. The normative [API field map](API.md#llm-schema-1-to-schema-2-field-map)
lists each relocation and changed predicate.

The exact active methodology selectors are:

- `llm-memory-v2-cpu-decode-contiguous`, `llm-memory-v2-cpu-decode-paged`;
- `llm-memory-v2-cpu-prefill-contiguous`, `llm-memory-v2-cpu-prefill-paged`;
- `llm-memory-v2-metal-decode-contiguous`, `llm-memory-v2-metal-decode-paged`;
- `llm-memory-v2-metal-prefill-contiguous`, `llm-memory-v2-metal-prefill-paged`.

The required top-level fields are:

```text
schema_version, mode, backend, phase, kv_layout, methodology_version,
software, configuration, resolved_plan, backend_evidence, memory_budget,
calibration, measurements, aggregates, status, reason_code,
results_complete, run_accepted, interpretation, diagnostic, interruption_requested,
scenario_order_balance_complete, seeds, counters, checkpoint_lifecycle, loop_records,
environment, quality_warnings, build_manifest
```

`configuration` preserves exact argv/output target and resolved/default input sources. Omitted iterations means
automatic calibration; file output still needs a target. Omitted/empty output is disabled, with no automatic filename.

`resolved_plan` owns `plan_identity`, logical `geometry`, `model_context`, `layout`, `resources`,
`component_identities`, `methodology`, `model_work_plan`, `scenario_plans`, and `frozen_plan_refs`.
Exactly one geometry phase object is populated. Decode uses integer visible-context tokens; prefill has integer P/Q
and decimal-string tile/prefix visits. Decode-only crossover and classification values are null for prefill.
`model_context.prefill` contains nullable theoretical quantities as described above; it is null for decode.

Paged layout retains integer block size, decimal block/tail/table geometry and materialized permutation domain,
seed, version and hash. Before Metal table preparation, admitted geometry survives with null runtime permutation.
Resources retain exact logical/physical K/V, suffix padding, table and Metal segment/argument-buffer geometry.
Component identity retains fixed-order length-prefixed logical/layout/permutation/backend/ABI/schedule/timer/
buffer/write/checksum/MSL fields under `llm-memory-components-v1`; CPU MSL and contiguous permutation fields are null.
The separately published `run_policy_version` is `llm-run-policy-bounded-loop-snapshots-v1`.

Canonical `scenario_plans[]` stores unique scenario/T/explicit content once, including excluded calibration shapes,
ordered by weights/KV/mixed, increasing T, then false/true explicit policy. Each retains exact identity, scenario seed,
`model_ref: "resolved_plan"`, work kind, work units, limits and all planned byte/lookup quantities. `frozen_plan_refs`
has the three scenario keys and integer index or unresolved null. Measurement and calibration `plan_ref` refer to the
same document's array; public indexes may change between snapshots while internal insertion handles remain stable.
Unknown/out-of-range and wrong-scenario references reject publication. Canonical content is immutable after registration.

A measurement retains `measurement_id` (its scheduled array index), scenario/loop/order, status/reason, authoritative
accepted `elapsed_seconds`, completed work/payload/metadata/accounted bytes and actual runtime evidence.
Work kind, `work_units`, per-work-unit bytes and planned totals are read through `plan_ref`. Canonical total names are
`effective_model_payload_bytes`, `layout_metadata_lookup_count`, `layout_metadata_read_bytes`, `task_accounted_bytes`;
measurement totals keep `completed_`. CPU `completion_derivation: "accepted-plan-derived"` is not a hardware counter.
Derived workers/QoS, calibration indexes, working-set and mixed fractions remain available. Invalid attempts retain
observed diagnostic time and actual checksum evidence while accepted elapsed/rates are null.

Expected checksum lives once at `scenario_plans[].expected_checksum`, with exactly status/reason,
`expected_worker_checksums`, `expected_run_checksum`. Available means the canonical cold expectation was reconstructed,
not that execution passed; unavailable makes both expected fields null. CPU worker expectations are ordered by worker
and include W/K/V component state, exact bytes and span count; CPU run checksum has state A/B. Metal expected workers
are null; its run object contains W/K/V dual-mod32 pairs. Measurement checksum contains status/reason/validity and
actual worker/run witnesses. Calibration retains compact actual-run-only evidence at `execution.checksum`, with
expectations resolved via its plan reference; no actual-worker array is invented. Algorithm versions have one owner in
canonical components, and CPU/Metal numerical witnesses are not cross-backend comparisons.

Named checks under measurement/calibration `execution.validation.checks[]` preserve the independently completed
structure, applicable phase/scenario write/unchanged and padding observations. Their kind tokens are
`post-validation-structure`, `kv-append-final`, `kv-prefill-final-samples`, `kv-append-unchanged`, `kv-padding-canary`.
CPU paged decode weights-only checks unchanged append and applicable padding; CPU paged prefill weights-only checks
structure/padding, not valid prompt content. Metal weights-only has no write/padding check. Inapplicable evaluated/valid are null;
applicable unresolved valid is null; observed mismatch remains false. Checksum agreement may coexist with failed cold
validation and cannot override it. The [fault matrix](#checksum-fault-model-and-evidence-limits) retains its scope:
new names add no collision resistance, unsampled-byte coverage or runtime trace.

`backend_evidence` retains tagged CPU/Metal branches, lifecycle, capability, resources and bounded error diagnostics.
CPU prefill preserves execution/scope identities and exact worker-cost vectors; paged prefill also has paged evidence.
Metal has null workers and preserves MSL/layout-probe/pipeline, actual resource options/lengths, grid/owner costs,
GPU raw start/end, host envelope, command/encoder/dispatch observations and nullable same-clock queue delay.
`memory_budget` separates immutable geometry from rounded/committed/transient/known-owned/admitted runtime evidence.
Canonical plan/expected vector capacities, calibration, accepted-ID maps, exact-stat scratch and DOM/serialized-string
simultaneous peak are admitted together; paged Metal canonical expectations also reserve table/validation/hash scratch.
The available-memory sample is excluded from immutable identities. Snapshot count reduction does not reduce peak size.

Indexes, bounded small inputs and work units are JSON integers below 2^53, never booleans. Bytes, lookups, visits,
seeds and checksums are canonical unsigned decimal strings (`0` or `[1-9][0-9]*`), with the relevant uint64/uint32 bounds.
Known zero is not null. Unavailable/inapplicable values retain null plus applicability/status/reason semantics.

`build_manifest` version 1 records build-time Git provenance, compiler, compile/link flags, target architecture,
SDK and deployment target. The command captures `binary_sha256` once before tasks. Missing build fields or an
unavailable binary hash produce partial evidence; a caller supplying no manifest retains `status: "unavailable"`,
`reason_code: "build-provenance-not-provided"`, and null provenance fields.
CPU measurement and excluded-attempt `execution.timing.cpu_raw` retain the original Mach `start_ticks`, `stop_ticks`,
`delta_ticks` and timebase numerator/denominator. Missing snapshots and Metal CPU-timing evidence are null; older
schema-2 artifacts may omit this optional field. See the [API contract](API.md#result-schemas-and-completion) for
availability and numeric rules. Neither manifest/file binding nor software/MSL identity is signed execution attestation.

Consumer acceptance requires successful process outcome, exact mode/schema2/backend/phase/layout/v2 methodology,
`run_policy_version: "llm-run-policy-bounded-loop-snapshots-v1"`, complete status, `results_complete: true`,
`run_accepted: true`, every planned measurement measured, valid same-document references, and a non-null selected
metric. This consumes producer acceptance; it is not an independent re-execution of the checksum, timing or build proof.
A comparison policy separately checks matched geometry, physical layout/permutation, seeds, exact work, backend/device,
component/MSL/pipeline/timer identity and environment. Position balance and CV can inform that policy but never redefine
correctness or justify performance-based sample filtering.

The `interpretation` object preserves the memory-only boundary: effective GB/s is logical W/K/V divided by authoritative
elapsed time, not physical DRAM; timed table metadata is excluded; prefill does no Transformer math and predicts no TTFT;
private Metal memory is not separate VRAM; cache/SLC/DRAM residency is unmeasured. Distinct backend/phase/layout/model/
component cohorts must not be pooled as one distribution.

## Console contract and quality warnings

The console identifies backend, phase/work unit, KV layout, phase geometry, warm/cacheable semantics, exact
weight/KV-read/KV-write bytes, and up to one measured headline per scenario. Decode prints context and crossover;
prefill prints P/Q/C, prefix visits, causal pairs, and theoretical logical attention/FMA model-context counts.
It uses phase-specific labels such as `ms/decode step` or `ms/prefill operation`; JSON remains backend-neutral with
`synthetic_work_unit_latency_seconds`, `synthetic_memory_work_units_per_second`, and
`effective_model_payload_gb_s`. Metal task output uses `kv_write=valid|invalid|not-evaluated|not-applicable`, summarizing
the phase-specific named JSON checks. The report never uses bare `tokens/s` and
states that effective model payload is not a physical DRAM counter. A scenario without a headline does not receive a
fabricated numeric console value; its status, reason, and null observations remain in JSON.

Metal console output additionally identifies device/capability limits, W/K/V segment counts and capacity, Tier 2
argument-buffer length, committed/known-peak/admitted memory, scenario pipeline/grid, authoritative GPU elapsed time,
and checksum/K/V-write/canary validation state. It does not print CPU workers or worker QoS for Metal.

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
- Metal contiguous-prefill parameter ABI/layout probe, Q=1/Q=P/remainder tiles, multiple operations/layers/batches,
  exact payload and vector tails, final-ordinal K/V-write validation, and source-hash-bound full-write → tile-K →
  tile-V loop-order audit;
- Metal paged-prefill `N + 2*M` lookup accounting, cyclic owner-ordinal scheduling, partial terminal visits,
  per-threadgroup accounted-cost evidence, multi-segment table/K/V addressing, final-ordinal representative/boundary prompt-write samples and
  padding-canary validation, wrong-table detection, and source-hash-bound loop-order audit;
- synchronized worker timing, startup cancellation, QoS evidence, timer/error containment, and AAPCS64 preservation;
- scenario-specific calibration, frozen plans, cyclic balance, aggregate population, interruption, and checkpoint
  precedence;
- schema identity, decimal exact integers, status/null rules, classification, interpretation, file/stdout transport, and
  executable CLI behavior.

For a performance comparison, keep the exact command/model geometry, layout, paged `G`/physical geometry/table and
permutation identity when applicable, explicit-versus-automatic policy, frozen plan, seed, worker counts,
software/methodology, hardware, macOS, power/thermal state, and background load matched. Prefer a count divisible by
three, inspect each scenario's accepted n, CV, duration quality and warnings, and retain both stdout/file payload and
stderr transcript. Match the complete component identities and run policy, including ordering, conditioning and output
transport/checkpoint cadence. Freeze scenario-specific work across an A/B pair; separately calibrated work can differ.
Compare independent, alternating process pairs in repeated series; a single accepted artifact or lower median is not
sufficient evidence of a machine or methodology performance difference. If hardware or methodology is the experimental
variable, label that contrast and keep its populations separate instead of claiming matched-cohort reproducibility.

The `environment.start` and `environment.end` thermal-state and Low Power Mode fields are instantaneous operating-system
observations. They are not continuous temperature monitoring, cache-residency evidence, or guaranteed CPU affinity;
equal nominal endpoints do not exclude an intervening state change. Unavailable observations are missing evidence.
Intertask host gaps include orchestration and optional output work and are distinct from authoritative CPU/GPU task
times. Added excluded conditioning costs wall time even when the following measured kernel gets faster. A separate
real inference-engine run can be useful correlation evidence, but it is not part of this benchmark's correctness or
acceptance predicate.

## Change control

Any change to traffic formulas, context semantics, buffer sizing/layout, temporal append behavior, worker/layer order,
output-serialization peak admission, timing boundary, checksum observability, calibration/frozen-plan rules,
interruption/checkpoint lifecycle, or meaning of a reported field requires methodology and schema compatibility review.
Removing or renaming a field, changing its type, or changing its meaning requires a schema-version bump. Additive
evidence may remain schema 2 only when existing consumers can safely ignore it.

Runtime paged allocation/free lists, prefix sharing, sliding windows, growing context, chunked prefill, Metal execution
outside the active profiles, ANE execution, model presets, quantization metadata,
multiple weight passes, or KV replay factors other than one are separate methodology features. The generic schema
vocabulary does not activate them: each requires its own end-to-end implementation gate, exact selector-derived
methodology and component identity, public CLI/documentation update, and compatibility review before it becomes a
supported profile.

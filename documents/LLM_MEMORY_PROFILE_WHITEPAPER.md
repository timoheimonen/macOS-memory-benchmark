# Synthetic LLM Memory Profile

## Abstract

`memory_benchmark --llm-memory` is a versioned Apple Silicon synthetic memory benchmark with a generic
backend/phase/KV-layout schema. The only active profile in this revision is CPU/decode/contiguous: the memory side of
one autoregressive decode step at a fixed visible context. It executes three scenarios derived from the same explicit
model geometry:

- `weights_only`: read the active weights once;
- `kv_only`: append the current token's K/V records and read the complete visible K/V history;
- `mixed`: perform the weight and KV work in worker-local layer order inside one synchronized timing interval.

The result is effective logical payload divided by elapsed CPU time. It is not a Transformer implementation, an
inference-engine benchmark, a physical DRAM counter, or a `tokens/s` claim.

## Methodology identity

The current contract is identified by:

| Property | Value |
|---|---|
| Software version | `0.63.0` |
| Mode | `llm_memory` |
| Backend | `cpu` |
| JSON schema | `1` |
| Phase selector | `decode` |
| Work unit | `decode_step` |
| Methodology | `llm-memory-v1-cpu-decode-contiguous` |
| Model/scenario plan identity prefix | `llm-memory-work-plan-v1` |
| Component identity prefix | `llm-memory-components-v1` |
| Logical profile version | `decode_steady_fixed_context` |
| KV layout selector/version | `contiguous` / `contiguous_layer_batch_token_head_dimension` |
| Permutation version | null |
| Backend executor version | `llm-cpu-executor-v1-arm64-decode-contiguous` |
| Resource ABI | `llm-memory-descriptor-abi-v1` |
| Schedule version | `worker-local-layer-order-no-per-layer-global-barrier` |
| Timer policy | `synchronized-start-to-last-worker-completion-per-scenario-task` |
| Buffer pattern | `llm-buffer-pattern-v1` |
| Write pattern | `llm-kv-append-affine64-v1` |
| Checksum pattern | `llm-read-checksum-v1` |
| MSL revision/source SHA-256 | null / null |
| Traffic classification | `llm-exact-weight-vs-kv-read-payload-v1` |

Component identities use a fixed-order, length-prefixed canonical serialization beginning with
`llm-memory-components-v1`. Always-applicable identities are strings; permutation and MSL identities are JSON null when
they do not apply. A workload comparison requires the selectors, methodology, complete component identity, frozen
work-plan, and environment evidence, not merely the same nominal model name.

## Scope and non-goals

The active profile models decode-only, warm-memory, fixed-context traffic with one active weight pass and one KV replay
per work unit. It supports MHA, GQA, and MQA geometry through explicit query- and KV-head counts, a positive batch count,
and 1-, 2-, or 4-byte KV elements.

Schema-v1 vocabulary reserves the tokens `cpu|metal`, `decode|prefill`, `contiguous|paged`,
`decode_step|prefill_operation`, and `none|current_token_append|full_prompt_population`. Metal, prefill, and paged KV
are not selectable or supported in this revision, and no hidden fallback activates them. Their future objects and
scalars remain null in an active CPU/decode/contiguous result.

It intentionally excludes:

- GEMV/GEMM/FMA, dequantization, RoPE, softmax, layer normalization, activation, and scratch traffic;
- tokenizer, model loader, framework scheduler/dispatch, kernel fusion, and compute-memory overlap;
- MLX, llama.cpp, Core ML, active Metal/GPU execution, and ANE execution;
- prefill or a context that grows during one measurement;
- paged/block-table, sliding-window, head-strided, compressed, or shared-prefix KV policies;
- speculative decoding and model-specific control flow;
- built-in model presets, model-file introspection, and LLM parameter sweeps;
- recycling a small physical buffer to represent a larger logical model;
- hardware-counter claims about cache, SLC, TLB, memory-controller, or DRAM traffic.

For a mixture-of-experts model, `--weight-size-mb` must represent weights active for one synthetic decode step, not the
model's total stored weights unless all of them are active.

## CLI and output contract

The standalone primary mode is `-M` / `--llm-memory`. Its exact whitelist is:

```text
--weight-size-mb <MiB>      required
--layers <count>            required
--query-heads <count>       required
--kv-heads <count>          required
--head-dim <count>          required
--context-tokens <count>    required
--kv-element-bytes <1|2|4>  default 2
--batch-size <count>        default 1
-t, --threads <count>       default detected CPU worker count
-i, --iterations <count>    default automatic per-scenario calibration
-r, --count <count>         default 3
--seed <uint64>             default one generated nonzero base seed
-o, --output <target>       default console only
-h, --help
```

Every required model option must occur exactly once. Optional values and the mode/help selector may occur at most once.
Numeric input is a complete decimal token; counts are positive, while an explicit seed may be zero. Query heads must be
at least the KV-head count and divisible by it. All other primary modes and all buffer/cache/latency/TLB/pattern/GPU,
`--non-cacheable`, `--sweep`, and `--sweep-max-runs` options are rejected.

Backend, phase, and layout have no public selector flags in this phase. The resolved defaults are always `cpu`,
`decode`, and `contiguous`, and `configuration.resolved_sources` records each as `default`. Reserved future selectors
must not be supplied or inferred from the schema vocabulary.

Output targets follow the shared process contract:

- omitted or empty output means console only;
- exact `--output -` reserves stdout for one final schema 1 document and routes the post-parse human transcript to
  stderr;
- every other non-empty raw token is a file, including `./-` and flag-shaped names;
- file output uses atomic `<target>.tmp` replacement after each terminal scenario measurement and at command terminal;
- stdout performs the same logical checkpoint and stop transitions without intermediate serialization.

Parser/preflight, work-plan, JSON-output peak-estimation, timer-creation, memory-budget, mapping, initialization, or
descriptor-preparation failure before runner-result initialization leaves stdout empty. Once the runner initializes
status-bearing evidence, normal failure or graceful interruption is serializable. A file checkpoint failure is terminal
and is not retried by a final file write.

## Exact geometry and logical traffic

Let:

- `W` = active weight bytes per work unit;
- `L` = layer count;
- `h_q` = query-head count;
- `h_kv` = physical KV-head count;
- `d_h` = elements per K/V head vector;
- `s_kv` = bytes per KV element;
- `B` = batch-sequence count;
- `A` = visible context tokens, including the current synthetic token;
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

## Formula golden vector and crossover

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

## Mappings, budget, and layout

The command owns three regular private anonymous cacheable mappings for the suite lifetime:

1. active weights, exactly `W` requested bytes;
2. K cache, exactly `L * B * A * h_kv * d_h * s_kv` requested bytes;
3. V cache, the same size as K.

Each mapping is rounded separately to the native page granularity for committed-byte accounting. Admission includes the
three page-rounded mappings, descriptor arrays, retained pointer-free planner storage, expected/actual checksum storage,
worker/thread state, calibration/result records, statistics workspace, warnings, and orchestration storage. The normal
policy admits no more than 80% of the current available-memory estimate; the existing fallback applies when that sample
is unavailable and caps the admitted total at 2 GiB. The requested workload is rejected rather than silently reduced.

A non-empty file or stdout JSON target also reserves a conservative peak for one live schema DOM and its serialized
transport text before final memory admission. The estimate covers fixed schema storage, captured input strings, every
planned measurement record, and both expected and actual worker-checksum trees. The input-string term also includes
the frozen model-plan, methodology, and component identities. Omitted or empty output adds no serialization reserve;
normal console-only execution does not serialize a schema document.

Schema ownership separates immutable resource geometry from allocation/admission evidence. `resolved_plan.resources`
contains canonical decimal-string logical weight/K/V lengths, physical K/V lengths, layout padding, and nullable
block-table bytes. Top-level `memory_budget` contains canonical decimal-string `resource_rounding_bytes`,
`transient_peak_bytes`, `known_owned_peak_bytes`, and `admitted_budget_bytes`, along with any additive detailed estimate
evidence. The active contiguous profile has identical K/V logical and physical lengths, zero layout padding, and a null
block-table resource.

The K and V mappings each use contiguous `[layer][batch_sequence][token][kv_head][head_dimension]` order. The visible
context includes the current token, so its append record is the final token record in each layer/sequence region.
Separate K and V mappings make the two streams explicit and prevent accidental aliasing.

Active weights are divided across layers by quotient and remainder so every byte belongs to exactly one layer. Every
layer's weight span and each layer/sequence K/V visible span are partitioned across the effective workers into disjoint
contiguous ranges. Internal boundaries prefer 32-byte alignment without dropping the exact tail. Effective workers are
the minimum of requested workers, detected availability, and the smallest executable span capacity; requested and
available counts remain separately recorded.

Each worker owns exactly `L` layer descriptors and `L*B` sequence descriptors. The frozen 64-bit ARM64 ABI uses 16-byte
aligned, 48-byte `LlmLayerDescriptor` values and 80-byte `LlmKvSequenceDescriptor` values. Pointer/length pairs identify
the worker's weight, visible K/V, and current-token append intersections. Empty worker spans are null/zero and skipped;
the planner never creates a worker whose entire scenario is empty.

## Scenario traversal

`weights_only` sets the kernel's weight bit and, for every work unit and layer, reads that worker's layer weight shard
once.

`kv_only` sets the KV bit. For every work unit, layer, and batch sequence, the worker:

1. writes its current-token K append subrange;
2. writes its current-token V append subrange;
3. reads its complete visible K subrange;
4. reads its complete visible V subrange.

The read includes the current token that was just written.

`mixed` sets both bits. For every work unit, each worker processes layers in increasing order. Inside a layer it reads its
weight shard, then processes every batch sequence's K/V append and visible-history reads before moving to the next
layer. Workers share one synchronized task start, but there is no synthetic global barrier at every layer. Thus mixed is
one layer-interleaved workload, not a post-hoc sum of separately timed weight and KV passes.

## Initialization, append pattern, and checksums

Preparation writes every requested byte of all three mappings exactly once with `llm-buffer-pattern-v1`, pre-touching
the full requested mapped working set. It accumulates static per-span checksum references during those writes, avoiding
another reference-reading pass. Initialization, mapping page faults, and reference construction occur before timed work.
The planner uses the shared SplitMix64 derivation primitive with frozen domains to derive separate weight/K/V buffer
seeds and weights-only/KV-only/mixed scenario seeds from the one base seed; schema 1 stores each as an exact decimal
string.

The initialization pattern treats each mapping as a zero-based stream of little-endian 64-bit words. For mapping word
index `i` and that mapping's domain-separated buffer seed:

```text
word_i = buffer_seed + 0x9E3779B97F4A7C15 * (i + 1) mod 2^64
```

A final 1–7-byte mapping tail takes the low little-endian bytes of the next word. A worker span that begins inside a
word observes the corresponding canonical mapping bytes rather than restarting the generator. With seed zero, words
zero and one are `0x9E3779B97F4A7C15` and `0x3C6EF372FE94F82A`.

The current-token K/V append uses `llm-kv-append-affine64-v1`. The deterministic 64-bit word depends on the scenario
seed, task-local step, layer index, batch-sequence index, record-word index, and K/V domain. The dedicated ARM64 kernel
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

Only exact checksum agreement, complete worker lifecycle, successful kernel status, and a finite positive elapsed time
permit `measured` status. Checksum and fold traffic is validation evidence, not part of the logical payload numerator.

## Timing boundary and worker lifecycle

One scenario task follows this boundary:

1. validate immutable descriptors and derive expected checksums;
2. create the complete worker team and prepare best-effort worker QoS;
3. wait until every worker reaches the start gate;
4. execute `dsb ish; isb`, start `HighResTimer`, and release the gate;
5. let each worker call the dedicated ARM64 kernel once for all frozen work units;
6. stop the timer at last-worker completion;
7. join workers and validate/fold checksums outside the elapsed interval.

Thread creation, QoS calls, allocation, initialization, pre-touch, page faults caused by preparation, descriptor
validation, expected-checksum generation, warmup, calibration, aggregation, console, JSON, and checkpoint writes are
outside primary elapsed time. Normal cache state is not flushed between tasks or loops, so the methodology is explicitly
warm/cacheable and cache-inclusive.

## Calibration, frozen work, order, and statistics

When `--iterations T` is present, each scenario uses exactly `T` work units. It still runs one excluded same-shape
warmup.

When iterations are omitted, each scenario is calibrated independently:

1. excluded one-work-unit same-shape warmup;
2. excluded pilot covering at least 8 MiB of effective payload when scenario guardrails permit;
3. scale toward 150 ms;
4. excluded same-shape warmup and duration trial;
5. at most two excluded correction attempts when outside the inclusive 100–250 ms intended window.

The one-billion-work-unit and 64 GiB accounted-byte ceilings apply per scenario task. For every scenario:

```text
accounted_bytes_per_work_unit =
  effective_model_payload_bytes_per_work_unit
  + layout_metadata_read_bytes_per_work_unit
planned_task_accounted_bytes = planned_work_units * accounted_bytes_per_work_unit
completed_task_accounted_bytes = completed_work_units * accounted_bytes_per_work_unit
```

The active contiguous profile has zero layout-metadata bytes, so its admitted workload and exact payload remain
numerically unchanged. One exact work unit may legitimately exceed the duration window and is retained with an
`above-target-single-work-unit` quality token. The last valid plans for all
three scenarios are frozen before loop zero and reused without recalibration. A later slow or fast measurement remains
evidence rather than triggering performance-based retry.

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

If the executor throws before returning evidence for a measurement or excluded task, nested `execution.status` is
`unavailable`, its reason is the runner-exception token, and absent worker-lifecycle, QoS, elapsed-time, and checksum
values are null. For a `not_run` measurement, the top-level successful/failed QoS-worker counts are also null. Neither
case may be interpreted as zero workers, a zero duration, or a successful checksum.

Interruption uses task-level completion-wins semantics. Stop is checked between complete executor tasks, not inside the
ARM64 hot loop or between layer descriptors. A started task runs to normal completion or genuine failure; a valid current
measurement remains measured even if the signal arrived during it. Once stop is observed, no next task starts and all
remaining slots become interrupted/null. A real timer, executor, checksum, or checkpoint failure remains authoritative
over a simultaneous interruption.

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

`geometry.decode` and `geometry.prefill` are object-or-null. The active profile has
`decode.visible_context_tokens` as an integer and `prefill: null`; the context field is never reused for prefill.
When the prefill profile is eventually activated, its object has integer `prompt_tokens` and
`attention_query_tile_tokens` plus decimal-string `tile_count`,
`attention_prefix_token_visits_per_sequence`, `causal_token_pairs_per_sequence`, `logical_attention_pairs`, and
`logical_attention_fma_terms`.
`layout.kv_layout` is a string. Paged-only integer-or-null `kv_block_tokens`; decimal-string-or-null
`blocks_per_sequence`, `physical_blocks_per_layer`, `last_block_tokens`, `last_block_valid_bytes`,
`block_table_entries`, and `block_table_bytes`; and nullable `permutation_domain_uint64_hex`,
`permutation_seed_uint64_decimal`, `permutation_algorithm_version`, and `permutation_sha256` are null for contiguous.
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
the active profile has a CPU object and `metal: null`. `memory_budget` separates allocation-time evidence into required
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

The active profile uses `decode_step`; `weights_only` uses `kv_write_kind: "none"`, and KV-bearing scenarios use
`current_token_append`. Planned/completed work units are integer numbers. All listed byte, lookup, metadata, and
accounted fields are canonical decimal strings even when the applicable value is zero. Derived elapsed/rate/statistic
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
- full mapping/init/descriptor evidence and requested/available/effective workers;
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

For this revision the requested values are `cpu`, `decode`, and `contiguous`. `unsupported`, `partial`, `interrupted`,
`invalid`, and `failed` evidence is never accepted as performance. After the predicate, a consumer must still require
the selected scenario metric's non-null value plus checksum and quality conditions relevant to its conclusion. Process
exit success alone is insufficient because graceful interruption is an established success-return path.

The `interpretation` object always preserves the generic boundary: the workload is synthetic and memory-only;
effective GB/s is exact logical W/K/V payload divided by authoritative elapsed time, not measured physical DRAM
traffic; timed paged-table metadata is excluded from that numerator; prefill profiles do not perform Transformer compute
or predict TTFT; private Metal storage on unified memory is not separate VRAM; cache/SLC/DRAM residency is unmeasured;
and results with differing backend, phase, layout, phase geometry, paged geometry, methodology, or component identity
must not be pooled as one performance distribution.

## Console contract and quality warnings

The console identifies the backend, phase/decode-step work unit, KV layout, fixed-context warm/cacheable semantics,
exact weight/KV-read/KV-append bytes per work unit, crossover, and up to one measured headline for each scenario. It may
use decode-specific labels such as `ms/decode step` and `synthetic decode steps/s`; JSON remains backend-neutral with
`synthetic_work_unit_latency_seconds`, `synthetic_memory_work_units_per_second`, and
`effective_model_payload_gb_s`. The report never uses bare `tokens/s` and states that effective model payload is not a
physical DRAM counter. A scenario without a headline does not receive a fabricated numeric console value; its status,
reason, and null observations remain in JSON.

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
- exact CLI whitelist/defaults/incompatibilities and help isolation;
- pointer-free layout, disjoint coverage, memory admission, and descriptor ABI offsets;
- atomic three-mapping allocation, full-byte initialization/pre-touch, and cleanup;
- K/V append word, expected/read/run checksum, tail, bounds, and multi-step behavior;
- synchronized worker timing, startup cancellation, QoS evidence, timer/error containment, and AAPCS64 preservation;
- scenario-specific calibration, frozen plans, cyclic balance, aggregate population, interruption, and checkpoint
  precedence;
- schema identity, decimal exact integers, status/null rules, classification, interpretation, file/stdout transport, and
  executable CLI behavior.

For a performance comparison, keep the exact command/model geometry, explicit-versus-automatic policy, frozen plan,
seed, worker counts, software/methodology, hardware, macOS, power/thermal state, and background load matched. Prefer a
count divisible by three, inspect CV and warnings, and retain both stdout/file payload and stderr transcript. A separate
real inference-engine run can be useful correlation evidence, but it is not part of this benchmark's correctness or
acceptance predicate.

## Change control

Any change to traffic formulas, context semantics, buffer sizing/layout, temporal append behavior, worker/layer order,
output-serialization peak admission, timing boundary, checksum observability, calibration/frozen-plan rules,
interruption/checkpoint lifecycle, or meaning of a reported field requires methodology and schema compatibility review.
Removing or renaming a field, changing its type, or changing its meaning requires a schema-version bump. Additive
evidence may remain schema 1 only when existing consumers can safely ignore it.

Paged KV, growing context, prefill, Metal/ANE execution, model presets, quantization metadata, multiple weight passes,
or KV replay factors other than one are separate methodology features. The generic schema vocabulary does not activate
them: each requires its own end-to-end implementation gate, exact selector-derived methodology and component identity,
public CLI/documentation update, and compatibility review before it becomes a supported profile.

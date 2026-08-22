# Machine-Readable Command-Line API

This document defines the supported process-level integration contract for `memory_benchmark` 0.63.0. It describes how
software launches a benchmark, separates machine-readable output from the human transcript, and decides whether a JSON
result is safe to consume. The generated Doxygen pages document C++ internals; they are not this process API.

Runtime behavior and executable integration tests are authoritative if this document and the implementation differ.

## Transport support in this revision

All result-producing direct benchmark modes and the CPU modes' supported parameter sweeps provide the stdout JSON
transport. GPU schema 1 remains a direct-only mode and does not support sweeps.

| Command | Real JSON file | Exact `--output -` stdout transport |
|---|---:|---:|
| Direct `--benchmark` | Yes | Yes |
| Direct `--patterns` | Yes | Yes |
| Standard or pattern `--sweep` | Yes | Yes |
| Direct `--analyze-tlb` | Yes | Yes |
| `--analyze-tlb --sweep ...` | Yes | Yes |
| Direct `--analyze-core2core` | Yes | Yes |
| `--analyze-core2core --sweep ...` | Yes | Yes |
| Direct `--gpu-bandwidth` | Yes | Yes |
| Direct `--llm-memory` | Yes | Yes |

GPU schema 1 and LLM schema 1 are direct-only modes and do not support sweeps.

## Invocation and stream contract

For any result-producing direct benchmark command, or a supported CPU parameter sweep, an output value that is exactly
`-` selects stdout JSON:

```bash
memory_benchmark --benchmark --only-bandwidth --count 5 --buffer-size 512 --output -
```

For example, a sweep emits one final envelope rather than one document per attempted run:

```bash
memory_benchmark --benchmark --only-latency --sweep buffer-size=256,512 --output -
```

A direct GPU command uses the same stream transport with its existing top-level schema 1 payload:

```bash
memory_benchmark --gpu-bandwidth --buffer-size 512 --count 3 --seed 42 --output -
```

A direct LLM command likewise emits its top-level schema 1 payload. CPU decode and prefill each support contiguous or
paged KV:

```bash
memory_benchmark --llm-memory --weight-size-mb 64 --layers 4 \
  --query-heads 8 --kv-heads 2 --head-dim 64 --context-tokens 512 \
  --iterations 1 --count 3 --seed 42 --output -
```

The experimental Metal preview is selected explicitly and currently accepts decode with contiguous KV only:

```bash
memory_benchmark --llm-memory --llm-memory-backend metal \
  --weight-size-mb 64 --layers 4 --query-heads 8 --kv-heads 2 \
  --head-dim 64 --context-tokens 512 --iterations 1 --count 3 \
  --seed 42 --output -
```

The preview has passed its M4 validation gate; the required Apple7/M1 baseline validation remains pending. Runtime
capability admission is therefore not a cross-family production-readiness claim.

A prefill request supplies full-prompt and query-tile geometry instead of decode context:

```bash
memory_benchmark --llm-memory --weight-size-mb 64 --layers 4 \
  --query-heads 8 --kv-heads 2 --head-dim 64 --phase prefill \
  --prompt-tokens 512 --attention-query-tile-tokens 64 \
  --iterations 1 --count 3 --seed 42 --output -
```

A paged request selects the layout and supplies its required block size in tokens. The layout options may be added to
either a decode or prefill request:

```bash
memory_benchmark --llm-memory --weight-size-mb 64 --layers 4 \
  --query-heads 8 --kv-heads 2 --head-dim 64 --context-tokens 512 \
  --kv-layout paged --kv-block-tokens 16 \
  --iterations 1 --count 3 --seed 42 --output -
```

`--kv-layout` accepts `contiguous` or `paged` and defaults to `contiguous`. Paged requests require exactly one
`--kv-block-tokens <G>`; `G` must be positive, a power of two, and at most `UINT32_MAX`. A value larger than the active
phase length is valid. Contiguous requests reject `--kv-block-tokens`. Phase defaults to decode. Decode requires
`--context-tokens`; prefill requires `--prompt-tokens P` and `--attention-query-tile-tokens Q` with
`1 <= Q <= P`, and the phase-specific inputs are mutually exclusive. `--llm-memory-backend` defaults to `cpu`. All
four CPU phase/layout combinations are active. The experimental `metal` preview accepts only decode plus contiguous KV;
it rejects prefill, paged KV, and explicit `--threads`. No unsupported request receives a fallback.

The sentinel is classified from the raw option value before path normalization:

- `--output -` selects stdout JSON;
- an empty value disables JSON for a direct command, as does omitting `--output`;
- an empty value in a sweep is a missing/invalid required output target;
- every other non-empty value is a file target, including `./-` and flag-shaped names such as `-G`, `-T`,
  `--analyze-tlb`, `-k`, and `--cache-size`; therefore
  `--output ./-` writes an ordinary file named `-` in the current directory.

After successful parsing and mode selection, a supported machine-output command follows these stream rules:

- stdout contains exactly one UTF-8 JSON document, serialized with two-space indentation and followed by one newline;
- banners, configuration, progress, result tables, information, warnings, and runtime errors are written to stderr;
- no `-` or `-.tmp` transport file is created by the exact sentinel;
- the JSON payload is the mode's existing file-output payload, not a transport-specific wrapper;
- clients consume object keys by name and do not rely on textual key order.

Argument parsing and preflight validation can fail before a result state exists. Those failures return a non-zero process
status, write the centralized diagnostic to stderr, and leave stdout empty. Help is deliberately human-facing: `--help`
prints normal help to stdout and does not promise JSON when the selected parser accepts that combination. The standalone
TLB whitelist rejects `--analyze-tlb --help`; use `--help` without that mode flag.

Some runtime setup failures also occur before a schema-valid payload exists. The general command's early timer failure,
TLB setup, memory-budget, or allocation failures before analysis-state initialization, a GPU backend-factory failure,
and LLM logical preflight or JSON-output peak-estimation failure before runner-result initialization therefore produce
stderr plus a non-zero status and leave stdout empty. After the LLM runner has initialized a representable result, a
selected Metal backend that fails its runtime capability check emits one terminal schema-1 `unsupported` document;
runtime compiler, pipeline, resource, or task failures emit a terminal schema-1 `failed` run with failed or invalid
measurement evidence as applicable. All return non-zero, and none is retried on CPU. Once a mode has initialized a
representable result, graceful interruption or a normal runtime failure emits the available partial, interrupted,
error, failed, or unsupported payload. Core-to-core measurement failures, TLB measurement errors, GPU
post-initialization failures, and LLM runner, backend-task, checksum, or checkpoint failures fall on this status-bearing
path. Initialized failed payloads retain their non-zero process status; established graceful-interruption paths may
return zero but are not complete conclusions.

An observable final serialization, write, or flush failure returns `EXIT_FAILURE` and reports its diagnostic to stderr
without changing the already-computed measurement state. Any stdout bytes from that failed transfer are not an
acceptable result document.

Abrupt process termination, a crash, `SIGKILL`, or an unusable stdout pipe cannot guarantee a final document. Version 1
does not install a process-wide `SIGPIPE` policy.

## Checkpoints and final snapshots

Real file targets retain their existing persistence behavior:

- standard commands atomically checkpoint after completed loop-state changes and write their normal terminal result;
- pattern, TLB, and core-to-core commands write one terminal payload through the shared atomic file writer;
- parameter sweeps atomically checkpoint their combined envelope after each attempted run and also checkpoint a terminal
  zero-attempt envelope when the run plan is empty or interruption is observed before a run;
- GPU mode retains its mode-specific terminal-measurement and failure checkpoints;
- LLM mode atomically checkpoints after every terminal scenario measurement and once at command terminal;
- a temporary `<target>.tmp` file is replaced atomically, and a failed replacement preserves the preceding destination
  when possible.

Stdout is final-only. Intermediate standard, sweep, GPU, and LLM checkpoint requests are successful lazy no-ops: their
payload builders are not invoked, while all logical state changes, stop observations, counters, cleanup, and final result
construction still occur. In particular, GPU and LLM perform the same checkpoint-boundary stop reads as file output.
The command serializes one terminal snapshot after orchestration finishes. Stdout is not JSON Lines and never contains
a sequence of checkpoint documents.

File-output ownership is mode-aware. Standard and sweep file producers retain their existing checkpoints, GPU retains
its terminal-measurement, failure, and post-release replacement cadence, and LLM owns its scenario-terminal plus
command-terminal cadence. A failed LLM checkpoint is terminal and is not retried at a final file-write boundary; the
last successfully replaced destination remains the recoverable evidence when possible. The stdout command boundary
alone emits the retained final document.

Use a real file target when crash-resilient intermediate checkpoints are required.

## Result schemas and completion

The stdout transport reuses the current mode payload and does not add a transport-version field. Process transport
contract version 1 first appears in software version `0.62.0`.

| Payload | Schema authority | A command result is complete only when |
|---|---|---|
| Current standard | `configuration.mode == "benchmark" && configuration.benchmark_schema_version == 3` | `status == "complete" && results_complete == true && conclusions_valid == true && configuration.output_file is a string` |
| Patterns | `configuration.pattern_schema_version == 3` | `status == "complete" && results_complete == true` |
| TLB | `configuration.schema_version == 4` | `tlb_analysis.status == "complete" && tlb_analysis.conclusions_valid == true` |
| Core-to-core | `configuration.schema_version == 2` | `core_to_core_latency.status == "complete" && core_to_core_latency.measurements_complete == true` |
| GPU | `schema_version == 1` | `status == "complete" && results_complete == true && conclusions_valid == true` |
| LLM memory profile | `mode == "llm_memory" && schema_version == 1` | Requested backend/phase/layout match, methodology is the exact derived identity, `status == "complete"`, `results_complete == true`, `conclusions_valid == true`, and every planned measurement is `measured` |
| General CPU sweep | `configuration.sweep_schema_version == 1` | `status == "complete" && conclusions_valid == true` |
| Core-to-core sweep | `configuration.sweep_schema_version == 1` | `status == "complete" && conclusions_valid == true` |

For either schema-1 sweep envelope, the table contains the authoritative completeness predicate. Producers maintain
`completed_runs == planned_runs` whenever they emit a complete envelope with valid conclusions. A consumer may check
that equality separately as a defensive producer-consistency check, but it is not an additional schema-1 acceptance
predicate.

Each `runs[].result` in a sweep retains its nested mode's own schema-version field and completeness contract. Nested
standard classification recognizes only current schema 3 with `configuration.mode == "benchmark"` plus typed
`results_complete`, `conclusions_valid`, and `configuration.output_file` fields; standard schema 2 and every other
standard version are unsupported. Complete, partial, interrupted, and failed current schema-3 evidence remains
classifiable and retained rather than being discarded by the complete-result consumer boundary. A non-zero nested
execution that initialized a result remains in the envelope: its attempt is failed, but the payload is not replaced by
a generic diagnostic. In particular, nested TLB `tlb_analysis.status == "error"` maps to a failed sweep attempt without
adding a `tlb_analysis.status_reason` field.

Command completeness does not make every optional metric available. A selected standard measurement must have its
mode-specific measured/quality state and a non-null value. A pattern measurement may be intentionally `skipped` while
the command remains complete; consumers of a particular pattern metric must require `status == "measured"` and a
non-null value. TLB consumers must also honor the selected detection/evidence fields. Core-to-core affinity-scenario
conclusions additionally require `affinity_hint_comparison_interpretable == true`. A position-balanced GPU comparison
additionally requires `operation_order_balance_complete == true`; consumers of an operation also require a measured,
non-null value and its applicable validation/quality fields.

LLM schema 1 is an unpublished generic contract. It has top-level `mode: "llm_memory"`, `schema_version: 1`, and exact
`backend`, `phase`, `kv_layout`, and `methodology_version` selectors. Methodology is derived as
`llm-memory-v1-<backend>-<phase>-<layout>`. This revision activates `cpu`/`decode`/`contiguous`,
`cpu`/`decode`/`paged`, `cpu`/`prefill`/`contiguous`, and `cpu`/`prefill`/`paged`, with exact methodologies
`llm-memory-v1-cpu-decode-contiguous`, `llm-memory-v1-cpu-decode-paged`,
`llm-memory-v1-cpu-prefill-contiguous`, and `llm-memory-v1-cpu-prefill-paged`. It also activates
`metal`/`decode`/`contiguous` with methodology `llm-memory-v1-metal-decode-contiguous`. No selection is silently
replaced by another backend, phase, or layout.

Run statuses are `not_started`, `complete`, `partial`, `interrupted`, `unsupported`, and `failed`; measurement statuses
are `not_run`, `measured`, `interrupted`, `invalid`, and `failed`. `unsupported` is a terminal, non-acceptable result,
not permission to substitute CPU execution. Multiword status tokens use underscores, while multiword stable reason
codes and duration-quality tokens use hyphens. Only a `measured` record with accepted checksum evidence has non-null
elapsed/rate values and contributes to the matching aggregate. Unavailable metrics and unavailable checksum validity
are JSON null, never numeric zero.

If a measurement or excluded runner task never receives backend task evidence because the backend call throws, its
nested `execution.status` is `unavailable`, its reason remains the applicable runner-exception token, and unavailable
worker lifecycle, QoS, elapsed-time, and checksum fields are null. For a `not_run` measurement, top-level
`qos_successful_workers` and `qos_failed_workers` are also null. These are absence-of-evidence states, not zero-worker or
successful-checksum observations.

The required generic top-level field set is:

```text
schema_version, mode, backend, phase, kv_layout, methodology_version,
software, configuration, resolved_plan, backend_evidence, memory_budget,
calibration, measurements, aggregates, status, reason_code,
results_complete, conclusions_valid, interpretation
```

`configuration` preserves exact `argv`, the raw output target, requested/resolved inputs, and a `resolved_sources`
object. Defaults remain evidence as `default`; they are not fabricated as explicit argv. Backend defaults to CPU and
phase defaults to decode; an explicit prefill selection records `phase: "explicit"`. Decode configuration publishes
integer `visible_context_tokens` and null prompt/tile values. Prefill publishes integer prompt/tile values and null
`visible_context_tokens`. Layout resolves to `contiguous` by default or to the explicit `--kv-layout` value. The
`kv_block_tokens` input is an integer for paged requests and null for contiguous requests; paged requests record it as
explicit because no block-size default exists.
Metal configuration records null `requested_workers`, `available_workers`, `worker_source`, and resolved worker source.
Its backend evidence records `workers_applicable: false` and `worker_qos_applicable: false`; the command does not run
CPU worker detection. Metal measurement and nested execution worker/QoS fields are likewise null. CPU retains its
requested/available/effective worker contract.

`resolved_plan` owns immutable logical plan evidence:

- Exactly one of `geometry.decode` and `geometry.prefill` is an object. Phase-specific fields are never overloaded.
  Decode has integer `visible_context_tokens`. Prefill has integer
  `prompt_tokens`/`attention_query_tile_tokens` and decimal-string tile, prefix-visit, causal-pair, attention-pair, and
  FMA-term counts. Decode-only crossover numerator, denominator, and context, current visible context, weight/KV-read
  ratio, and `current_context_classification` are null for prefill. `classification_version` and
  `classification_is_payload_only` remain populated because they describe the schema field's semantics.
- `layout.kv_layout` is the selected `contiguous` or `paged` token. Contiguous results use null for paged-only fields
  and report applicable lookup/read counts as decimal-string zero. Paged results populate integer `kv_block_tokens`;
  decimal-string block, tail, table-entry, and table-byte counts; permutation domain/algorithm/hash strings; and the
  decimal-string resolved permutation seed.
- `resources` separates decimal-string `weight_logical_bytes`, `k_logical_bytes`, `v_logical_bytes`, physical K/V
  lengths, K/V layout padding, and nullable block-table bytes. For contiguous, physical lengths equal logical lengths,
  padding is zero, and block-table bytes are null. For paged, physical lengths cover complete blocks and may exceed
  logical lengths; suffix padding and the resident uint32 table are reported separately.
  Metal additionally publishes its 256 MiB segment capacity, exact per-segment lengths and counts, maximum addressable
  bytes, unused nominal capacity, Tier 2 argument-buffer slots/encoded length/alignment, status/staging lengths, and
  admitted resource-plan totals. A final segment has its exact logical remainder; it is not padded to 256 MiB.
- `component_identities` records logical profile, KV layout, optional permutation, backend executor, resource ABI,
  schedule, timer policy, buffer pattern, write pattern, checksum pattern, and nullable MSL revision/source SHA-256.
  Their canonical aggregate identity uses fixed field order and length-prefixed values under
  `llm-memory-components-v1`; CPU MSL fields and contiguous permutation fields are null. Paged CPU results bind the
  physical layout and permutation identity to their paged descriptor, schedule, pattern, and checksum identities.

For a paged profile, let `A` be `visible_context_tokens` for decode or `prompt_tokens` for prefill, `G` be
`kv_block_tokens`, `B` be batch size, `L` be layer count, and
`R = kv_head_count * head_dimension * kv_element_bytes`. The schema evidence is derived exactly as follows:

```text
N = A / G + (A % G != 0)
physical_blocks_per_layer = B * N
block_bytes = G * R
last_block_tokens = A - (N - 1) * G
last_block_valid_bytes = last_block_tokens * R
decode_append_offset_in_last_block = ((A - 1) % G) * R  # decode only; null for prefill
k_logical_bytes = L * B * A * R
k_physical_length_bytes = L * B * N * block_bytes
k_layout_padding_bytes = k_physical_length_bytes - k_logical_bytes
block_table_entries = B * N
block_table_bytes = block_table_entries * 4
```

V uses the same logical, physical, and padding counts. All blocks are physically complete; terminal suffix padding is
initialized and validated but is neither touched by timed work nor included in effective model payload. The single
row-major `block_table[B][N]` is a bijection over `0 .. B*N-1`, stored as uint32 entries. `UINT32_MAX` is the invalid
sentinel and the physical-block count cannot exceed it. The table is generated once with the versioned descending
Fisher–Yates rejection algorithm driven by stateful SplitMix64. Its domain is `0x4c4c4d4b56504731`, its resolved state
is `splitmix64(base_seed xor domain)`. Each draw uses `threshold = uint64_wrap(0-bound) % bound`; after rejection,
`j = value % bound`. The table is validated before execution, hashed from explicit row-major little-endian entries,
made read-only for CPU execution, and held constant across warmup, calibration, scenarios, and measured loops. Its exact
algorithm version, domain, resolved seed, entry count, and lowercase 64-hex SHA-256 are part of the result identity.

For prefill, let `P` be prompt length, `Q` query-tile length, and `K = L*2*R`. With
`C = ceil(P/Q)`, tile ends `e_j = min((j+1)*Q, P)`, and `S(P,Q) = sum(e_j)`, one `prefill_operation` has:

```text
weight_read_bytes = W
kv_write_bytes = B * P * K
kv_read_bytes = B * S(P,Q) * K
weights_only_payload = W
kv_only_payload = B * (P + S(P,Q)) * K
mixed_payload = W + B * (P + S(P,Q)) * K
```

Each operation writes owner-local prompt tokens in ascending order, K then V for each token. Those writes precede that
owner's reads; no global worker barrier is implied. Each tile reads its complete owned K prefix before its complete
owned V prefix. The operation ordinal is bound into write/checksum evidence, and the timed checksum covers every
tile-read visit. Excluded post-validation checks each owner's deterministic first/middle/last canonical-word samples,
including bytes clipped to owner boundaries, against the final operation ordinal `T-1`; it does not scan every prompt
record. CPU scenario partitions report a stable identity plus minimum, maximum, and imbalance `worker-cost` evidence.
Audit-only causal pairs are
`triangular(P)` per sequence; logical attention pairs/FMA terms are reported but never executed.

For paged prefill, additionally let `N = ceil(P/G)`, `m_j = ceil(e_j/G)`, and `M = sum(m_j)`. Each layer/batch pair
performs `N` paired K/V write lookups, `M` K-prefix lookups, and `M` V-prefix lookups. A lookup that reaches a partial
tile prefix visits only the exact valid token bytes required by `e_j`; it never expands the logical visit to a whole
terminal block. Thus a KV-active prefill work unit reports:

```text
layout_metadata_lookup_count_per_work_unit = L * B * (N + 2 * M)
layout_metadata_read_bytes_per_work_unit = 4 * layout_metadata_lookup_count_per_work_unit
accounted_bytes_per_work_unit =
  effective_model_payload_bytes_per_work_unit + layout_metadata_read_bytes_per_work_unit
```

`backend_evidence.cpu.prefill` is null for decode and an object for either prefill layout. It records `cost_unit:
"worker-cost"`, the execution identity, descriptors per scenario/worker, and a three-entry scenario array. Each scenario
records its identity, scope count/identities, decimal-string per-worker accounted costs, and decimal-string minimum,
maximum, and max-minus-min imbalance per work unit. `backend_evidence.cpu.paged` is populated for either paged phase,
so paged prefill has both CPU evidence objects populated.

`backend_evidence` always contains both tagged branches. Exactly the selected branch is populated. Metal evidence
contains backend lifecycle status/reasons; device name and registry ID; Apple-family, unified-memory, Tier 2, and
maximum-buffer capability evidence; MSL version/revision/source hash; foundation and workload pipeline limits;
argument-buffer layout-probe evidence; actual `MTLBuffer.length`, optional `allocatedSize`, resource options, storage
and hazard modes; committed/peak/budget totals; and bounded Metal error diagnostics. Worker and worker-QoS
applicability are false. Capability absence is `unsupported`; compile, pipeline, allocation, command-buffer,
timestamp, checksum, or validation errors are failed/invalid states with stable reason codes.

The activated embedded source revision is `llm-metal-decode-contiguous-msl23-v1`. The three reported workload labels
are `membenchmark.llm-metal.pipeline.decode-contiguous.weights-only`,
`membenchmark.llm-metal.pipeline.decode-contiguous.kv-only`, and
`membenchmark.llm-metal.pipeline.decode-contiguous.mixed`. These are output identities, not a promise that future
schema revisions retain the same kernel implementation.

`memory_budget` separates immutable resource geometry from allocation-time evidence and includes canonical
decimal-string `resource_rounding_bytes`, `transient_peak_bytes`, `known_owned_peak_bytes`, and
`admitted_budget_bytes`. A paged candidate admits full physical K/V resources, the resident block table, page rounding,
descriptor/planner/checksum/orchestration storage, and the permutation-validation transient before table
materialization. A non-empty JSON target's orchestration reserve uses checked arithmetic for every variable-length
component/layout identity and, for prefill, the aggregate execution identity plus all scenario and scope identities.
For Metal it also reserves a maximum-sized grid/task evidence tree for every retained measurement and excluded
calibration attempt. The preflight and finalized-plan estimates cover the same identity set. `calibration` contains
excluded work-resolution and post-freeze same-shape warmup evidence;
`aggregates` contains only accepted measured values. No measured loop begins until all three scenario plans have been
atomically frozen and their canonical-order frozen warmups have succeeded. Additional diagnostic, interruption,
checkpoint, loop-order, checksum, environment, warning, and resource-preparation evidence may be present without
changing those ownership boundaries.

Each `calibration.attempts.<scenario>` array preserves its execution order. Its exact `purpose` vocabulary is
`calibration_shape_warmup`, `pilot`, `correction`, `single_unit_confirmation_warmup`,
`single_unit_confirmation`, and `frozen_measurement_warmup`. Successful automatic preparation starts with the shape
warmup and pilot, may contain correction attempts and at most one conditional single-unit confirmation warmup/attempt
pair, and ends with the frozen-plan warmup. Successful explicit preparation has no pilot or correction attempts and
records only its frozen-plan warmup. A failed or interrupted preparation retains only the attempts reached before its
terminal boundary.

Every measurement has stable generic work accounting:

```text
work_unit_kind, planned_work_units, completed_work_units,
weight_read_bytes_per_work_unit, kv_read_bytes_per_work_unit,
kv_write_bytes_per_work_unit, kv_write_kind,
effective_model_payload_bytes_per_work_unit,
layout_metadata_lookup_count_per_work_unit,
layout_metadata_read_bytes_per_work_unit, accounted_bytes_per_work_unit,
planned_effective_model_payload_bytes, completed_effective_model_payload_bytes,
planned_layout_metadata_lookup_count, completed_layout_metadata_lookup_count,
planned_layout_metadata_read_bytes, completed_layout_metadata_read_bytes,
planned_task_accounted_bytes, completed_task_accounted_bytes,
synthetic_work_unit_latency_seconds,
synthetic_memory_work_units_per_second,
effective_model_payload_gb_s
```

Decode uses `work_unit_kind: "decode_step"` and `kv_write_kind: "current_token_append"` for KV-bearing scenarios.
Prefill uses `work_unit_kind: "prefill_operation"` and `kv_write_kind: "full_prompt_population"`.
`weights_only` uses `kv_write_kind: "none"` in both phases. `planned_work_units` and `completed_work_units` are
integers.
All listed byte, lookup, metadata, and accounted quantities are canonical decimal strings, including applicable zero.
The three derived metrics are finite numbers only for a successfully measured record and otherwise null. The effective
model numerator contains versioned logical W/K/V reads and writes; timed layout metadata is reported separately and
included in `accounted_bytes_per_work_unit`, not in `effective_model_payload_gb_s`.
Measurement checksum evidence uses the phase-neutral keys `write_pattern_version` and `checksum_pattern_version`;
schema 1 does not publish decode-specific `append_pattern_version` or `read_checksum_version` aliases. Metal uses
`llm-metal-dual-mod32-v1` with separate W/K/V lanes. Its measurement execution evidence records the selected pipeline,
threadgroup/grid geometry, raw `GPUStartTime` and `GPUEndTime`, authoritative GPU elapsed time, the host submit/wait
envelope, command/encoder/dispatch counts and statuses, timed checksum comparison, and excluded validation of every byte
in every final K/V append record for a KV-bearing scenario.
Queue delay is null unless it is measured in the same clock domain. CPU and Metal checksum values are backend-specific
and are never compared numerically.

Each Metal task uses one reset command buffer, one timed command buffer with one explicitly serial compute encoder and
one workload dispatch, and one excluded post-validation command buffer. The task's `T` work units loop inside the
kernel. Initialization, pre-touch, expected-checksum construction, reset, and post-validation are outside the timed
window. `GPUStartTime`/`GPUEndTime` are read only after completion; zero, non-finite, negative, or non-increasing values
invalidate the task. Exact vector/scalar tails and segment-boundary splits preserve the planned logical byte count.

Paged `weights_only` performs no block-table access and reports zero layout-metadata work. For decode `kv_only` and
`mixed`, each layer/batch pair performs one paired K/V append lookup, `N` K-scan lookups, and `N` V-scan lookups per
decode step:

```text
layout_metadata_lookup_count_per_work_unit = L * B * (2 * N + 1)
layout_metadata_read_bytes_per_work_unit = 4 * layout_metadata_lookup_count_per_work_unit
accounted_bytes_per_work_unit =
  effective_model_payload_bytes_per_work_unit + layout_metadata_read_bytes_per_work_unit
```

Task totals multiply these quantities by the exact work-unit count with checked arithmetic. The shared one-billion
work-unit ceiling and 64 GiB task-accounted-byte guardrail apply to model plus metadata work. The
effective-model-payload
numerator never includes table bytes.

Every paged semantic visit loads its uint32 physical ID inside the timed ARM64 kernel and calculates the block address
after that load; the host does not replace the table with a pre-resolved pointer list. The traversal is append, complete
logical K-block scan, then complete logical V-block scan for each layer/batch pair, with mixed reading the layer weight
span first. Paged prefill instead writes all owned prompt blocks first, then for each query tile scans the exact K prefix
followed by the exact V prefix. The paged checksum binds logical table index, loaded physical ID, semantic visit kind,
and work-unit ordinal non-separably. Physical data patterns depend on pool, physical ID, and physical offset. Post-task
validation checks decode current-token writes or prefill final-ordinal prompt samples plus terminal padding canaries.
Generation, validation, initialization, pre-touch, expected-checksum construction, and post-validation are outside the
authoritative synchronized CPU task time.

The traffic classification version is `llm-exact-weight-vs-kv-read-payload-v1`: it compares exact active-weight bytes
with exact KV-read bytes only. `near_crossover` means equality, not a tolerance band and not an observed hardware
bottleneck.

`results_complete` requires all planned scenario measurements to be terminal and measured. `conclusions_valid` also
requires valid geometry/checksums, no checkpoint failure, and a complete cyclic position balance. A complete one-loop
run is therefore inspectable with `results_complete: true` but has `conclusions_valid: false`. Consumers of comparative
LLM conclusions must apply the exact selector/methodology predicate, require every planned measurement to be
`measured`, and then check the selected scenario metric's non-null value plus relevant checksum, quality, and environment
evidence. Comparisons also require identical phase geometry and, for paged results, identical `kv_block_tokens`,
permutation identity, physical-resource geometry, and component identities. Contiguous and paged measurements are not
members of the same comparison cohort merely because their logical model geometry matches.

`quality_warnings` merges and deduplicates runner tokens `weights_only-high-cv`, `kv_only-high-cv`,
`mixed-high-cv`, and `scenario-order-not-balanced` with final-report tokens `environment-not-nominal`,
`main-thread-qos-not-applied`, `worker-qos-not-applied`, `weight-working-set-cache-dominant`,
`kv-working-set-cache-dominant`, and `<scenario>-duration-<quality>`. The main-thread token requires QoS to have been
requested and not applied; absence of a warning is not proof of DRAM residency or scheduler placement.

Current standard schema-3 payloads require `configuration.output_file` to be a string and preserve the raw target token:
stdout therefore records `"-"`, while file targets retain spellings such as `./-`, `-T`, or `--cache-size` rather than a
normalized path. A current standard result nested in a sweep records an empty string because the envelope owns
persistence and nested file writes are disabled. Schema 3 requires boolean `results_complete` and `conclusions_valid`;
the producer makes `conclusions_valid` true exactly when `results_complete` is true, while consumers must still check
the explicit status and both booleans shown in the table.

The bundled standard-memory example scripts are maintained in lockstep with the current producer. Each script performs
only the local version, completion, and field sanity checks needed before reading its current schema-3 metric paths. For
the current producer that includes exact top-level `version == "0.63.0"` in addition to the standard identity and
completeness fields above. The examples are not a versioned compatibility library. Released standard schema 2,
unversioned historical standard JSON layouts, and every other explicit standard version are unsupported inputs and are
not routed through a metric-shape fallback.

Graceful interruption or runtime failure after a representable result state has been initialized emits the available
partial, interrupted, error, or failed JSON snapshot. The execution status and payload are independent: a non-zero status
must not cause a caller to discard evidence without parsing it. Exit status zero is used for human help, complete
execution, and established graceful-interruption paths; it alone does not prove that JSON conclusions are complete.

## Consumer acceptance procedure

A caller accepts a benchmark conclusion only after all of the following checks succeed:

1. Launch the executable with an argv array; do not construct an unquoted shell command from external input.
2. Capture stdout and stderr separately by draining both simultaneously while the child runs, or use a
   platform/language `communicate`-equivalent that drains both pipes while the child runs; then wait for the process and
   both streams to finish.
3. Reject an empty stdout result with the process status and stderr diagnostic.
4. Parse stdout as exactly one JSON document with no trailing non-whitespace data.
5. Check the supported mode and schema-version field.
6. Require the expected successful process status before accepting conclusions.
7. Apply the mode-specific command-completeness predicate above.
8. Apply the selected metric's status, non-null, and quality predicates.

Do not wait for process exit before reading sequential pipe captures. If either stdout or stderr fills its pipe buffer,
the child can block before exit and the parent's wait can deadlock.

Language-neutral pseudocode:

```text
result = run_process(argv, capture_stdout=true, capture_stderr=true)
if result.stdout is empty:
    reject_with_diagnostic(result.exit_status, result.stderr)

document = parse_one_json_value(result.stdout)
require_only_whitespace_after_document()
require_supported_mode_and_schema(document)
require_process_completed_as_expected(result.exit_status)
require_mode_completion_predicate(document)
require_selected_metric_is_measured_and_non_null(document)
accept(document)
```

Shell capture example:

```bash
memory_benchmark --benchmark --only-bandwidth --buffer-size 512 --count 5 --seed 42 --output - \
  >benchmark.json 2>benchmark.log
jq -e '.configuration.mode == "benchmark" and
       .configuration.benchmark_schema_version == 3 and
       (.configuration.output_file | type) == "string" and
       .status == "complete" and .results_complete == true and
       .conclusions_valid == true' benchmark.json

memory_benchmark --patterns --buffer-size 512 --count 5 --seed 42 --output - \
  >patterns.json 2>patterns.log
jq -e '.configuration.pattern_schema_version == 3 and
       .status == "complete" and .results_complete == true' patterns.json
```

The corresponding TLB and core-to-core command predicates are:

```bash
jq -e '.configuration.schema_version == 4 and
       .tlb_analysis.status == "complete" and
       .tlb_analysis.conclusions_valid == true' tlb.json

jq -e '.configuration.schema_version == 2 and
       .core_to_core_latency.status == "complete" and
       .core_to_core_latency.measurements_complete == true' core2core.json

jq -e '.configuration.sweep_schema_version == 1 and
       .status == "complete" and .conclusions_valid == true' sweep.json

# Optional defensive producer-consistency check; not part of sweep acceptance.
jq -e '.completed_runs == .planned_runs' sweep.json

jq -e '.schema_version == 1 and .mode == "gpu_bandwidth" and
       .status == "complete" and .results_complete == true and
       .conclusions_valid == true' gpu.json

jq -e '.mode == "llm_memory" and .schema_version == 1 and
       .backend == "cpu" and .phase == "decode" and
       .kv_layout == "paged" and
       .methodology_version == "llm-memory-v1-cpu-decode-paged" and
       .resolved_plan.layout.kv_block_tokens == 16 and
       (.resolved_plan.layout.permutation_sha256 |
        type == "string" and test("^[0-9a-f]{64}$")) and
       .status == "complete" and .results_complete == true and
       .conclusions_valid == true and
       ([.measurements[] | select(.status != "measured")] | length) == 0' llm_memory.json

jq -e '.mode == "llm_memory" and .schema_version == 1 and
       .backend == "cpu" and .phase == "prefill" and
       .kv_layout == "contiguous" and
       .methodology_version == "llm-memory-v1-cpu-prefill-contiguous" and
       .resolved_plan.geometry.decode == null and
       .resolved_plan.geometry.prefill.prompt_tokens == 512 and
       .resolved_plan.geometry.prefill.attention_query_tile_tokens == 64 and
       .status == "complete" and .results_complete == true and
       .conclusions_valid == true and
       ([.measurements[] | select(.status != "measured")] | length) == 0' llm_prefill.json
```

## Compatibility policy

- `version`, the GPU `software_version` field, and LLM `software` identity identify the application release; none is a
  result schema version.
- Current standard schema 3, pattern schema 3, TLB schema 4, core-to-core schema 2, GPU schema 1, and LLM schema 1 remain
  authoritative at their existing locations. The schema field is intentionally not normalized across these established
  payloads.
- The prior LLM schema-1 producer shape was unpublished. This revision deliberately replaces its CPU/step-specific
  vocabulary with the generic v1 contract without a compatibility reader, field alias, or schema bump. LLM schema-1
  consumers must use the current generic fields, tolerate unknown additive evidence fields, and validate every known
  field they consume. Future removal/rename/type/meaning changes require schema-version review; software identity alone
  is not a compatibility substitute.
- Bundled standard-memory examples track the current producer and read current standard schema-3 metric paths directly
  after local sanity checks. They provide no compatibility layer for released standard schema 2, unversioned
  historical standard JSON layouts, or any other explicit standard version.
- Both general and core-to-core sweep envelopes use `configuration.sweep_schema_version == 1`; nested results keep their
  independent mode schema versions.
- Additive optional fields may remain within a schema version only when old consumers can safely ignore them.
- Removing or renaming a field, changing its type, or changing its meaning requires a mode schema-version bump.
- A methodology change that affects comparison requires the mode's methodology-version mechanism even when JSON shape
  is unchanged.
- A transport change alone does not change the measurement schema.
- Schema-location normalization belongs in client code; this API does not move existing version fields.
- Direct standard, GPU, and LLM payloads retain raw `configuration.output_file`. With stdout transport it is the original
  target token `"-"`, not a filesystem path; `./-`, flag-shaped names, and every other non-empty non-sentinel value retain
  their file meaning. GPU and LLM additionally retain their exact captured `argv`.

## Benchmark process policy

Benchmark commands should be serialized and run on an otherwise idle machine. Avoid overlapping benchmark processes,
keep power and thermal conditions controlled, and use `caffeinate -i -d` for long runs when sleep would invalidate the
experiment. Process isolation is intentional: signals, QoS, large mappings, ARM64 work, and Metal resource lifetime stay
inside the launched benchmark process.

# Machine-Readable Command-Line API

This document defines the supported process-level integration contract for the current `memory_benchmark`
implementation. It describes how software launches a benchmark, separates machine-readable output from the human
transcript, and decides whether a JSON result is safe to consume. The generated Doxygen pages document C++ internals;
they are not this process API.

Runtime behavior and executable integration tests are authoritative if this document and the implementation differ.
The documented process runtime baseline is macOS 26 or later on Apple Silicon (ARM64).

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

GPU schema 1 and LLM schema 2 are direct-only modes and do not support sweeps.

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

A direct LLM command likewise emits its top-level schema 2 payload. CPU decode and prefill each support contiguous or
paged KV:

```bash
memory_benchmark --llm-memory --weight-size-mb 64 --layers 4 \
  --query-heads 8 --kv-heads 2 --head-dim 64 --context-tokens 512 \
  --iterations 1 --count 3 --seed 42 --output -
```

The capability-gated Metal backend is selected explicitly and accepts decode or prefill with contiguous or paged KV:

```bash
memory_benchmark --llm-memory --llm-memory-backend metal \
  --weight-size-mb 64 --layers 4 --query-heads 8 --kv-heads 2 \
  --head-dim 64 --context-tokens 512 --iterations 1 --count 3 \
  --kv-layout paged --kv-block-tokens 16 --seed 42 --output -
```

Metal paged prefill combines explicit prompt/tile geometry with an explicit block size:

```bash
memory_benchmark --llm-memory --llm-memory-backend metal \
  --weight-size-mb 64 --layers 4 --query-heads 8 --kv-heads 2 \
  --head-dim 64 --phase prefill --prompt-tokens 512 \
  --attention-query-tile-tokens 64 --iterations 1 --count 3 \
  --kv-layout paged --kv-block-tokens 16 --seed 42 --output -
```

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
eight backend/phase/layout combinations are active: CPU and Metal each accept decode and
prefill with contiguous or paged KV. Metal rejects explicit `--threads`, and no request receives a backend fallback.
Metal capability admission requires a default unified-memory device with Apple7-or-later support, Tier 2 argument
buffers, and `maxBufferLength >= 256 MiB`. The selected MSL 2.3 source must compile, its pipelines must be created, and
the runtime layout probe must pass. Capability absence produces terminal `unsupported` evidence; compiler, pipeline,
resource, or task failures produce terminal `failed`/`invalid` evidence and never authorize CPU fallback.

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
selected Metal backend that fails its runtime capability check emits one terminal schema-2 `unsupported` document;
runtime compiler, pipeline, resource, or task failures emit a terminal schema-2 `failed` run with failed or invalid
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
- LLM mode atomically snapshots bounded completed-loop progress and command terminal as specified below;
- a temporary `<target>.tmp` file is replaced atomically, and a failed replacement preserves the preceding destination
  when possible.

Stdout is final-only. Intermediate standard, sweep and GPU checkpoint requests are successful lazy no-ops: their
payload builders are not invoked, while all logical state changes, stop observations, counters, cleanup, and final result
construction still occur. LLM skips intermediate snapshot preparation on stdout but preserves task-boundary stop observations.
The command serializes one terminal snapshot after orchestration finishes. Stdout is not JSON Lines and never contains
a sequence of checkpoint documents.

For LLM, let `N=planned_loops` and `K=max(1,ceil(N/8))`, calculated without addition overflow. File targets
write progress after every Kth fully completed loop, at most eight times, and one command-terminal snapshot on
success, graceful interruption or representable failure. The current cadence gives at most 4/7/9 normal
snapshots for N=3/12/48. One late command exception after successful terminal persistence may write one corrective
failure snapshot; a failed checkpoint is terminal and never retried or followed by a disguised final write.

An abrupt exit can lose up to `3K` truly completed measurement attempts after the last successful snapshot, including
while its replacement is in progress. Unstarted tail placeholders do not count toward this bound. Before the first
successful snapshot there may be no file. Graceful SIGINT retains the finished prefix and an interrupted/null tail
when terminal persistence succeeds; SIGKILL or a crash does not promise terminal output. Atomic rename does not imply
power-loss durability. Stop checks remain at each existing task/terminal boundary even when a snapshot is skipped.
Stdout builds one final DOM; disabled output builds none.

`checkpoint_lifecycle` publishes `checkpoint_failed`, `prior_file_writer_attempts`, `prior_successful_file_writes`,
`observation_point: "before-current-snapshot-preparation"`, `current_request` (`progress`, `terminal`, or
`late-command-error-correction`), `current_persistence_success: null`, `snapshot_interval_loops`,
`checkpoint_policy: "bounded-loop-snapshots"`, `file_checkpoint_failure_is_terminal_and_not_retried: true`, and
`stdout_intermediate_checkpoints_are_lazy: true`. The prior counts observe writer entry and successful return before
this snapshot, not the current write's eventual success. Builder failure is not writer entry; stdout/disabled no-ops
are not file writes. For a successful N=3 file command the terminal document normally reports three prior successes,
not four self-inclusive successes. No extra write is made just to count itself. A retained prior snapshot cannot attest
to a future write/cleanup failure; final collection must retain actual process outcome and terminal context.

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
| LLM memory profile | `mode == "llm_memory" && schema_version == 2` | Requested backend/phase/layout match, methodology and run-policy match the exact identities, `status == "complete"`, `results_complete == true`, `run_accepted == true`, and every planned measurement is `measured` |
| General CPU sweep | `configuration.sweep_schema_version == 1` | `status == "complete" && conclusions_valid == true` |
| Core-to-core sweep | `configuration.sweep_schema_version == 1` | `status == "complete" && conclusions_valid == true` |

For either schema-1 sweep envelope, the table contains the authoritative completeness predicate. Producers maintain
`completed_runs == planned_runs` whenever they emit a complete envelope with valid conclusions. A consumer may check
that equality separately as a defensive producer-consistency check, but it is not an additional schema-1 acceptance
predicate.

Each `runs[].result` in a sweep retains its nested mode's own schema-version field and completeness contract. Nested
standard classification recognizes only current schema 3 with `configuration.mode == "benchmark"` plus typed
`results_complete`, `conclusions_valid`, and `configuration.output_file` fields; standard schema 2 and every other
standard schema version are unsupported. Complete, partial, interrupted, and failed current schema-3 evidence remains
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

LLM schema 2 replaces the unpublished schema-1 contract without compatibility aliases or a fallback reader. It has top-level `mode: "llm_memory"`, `schema_version: 2`, and exact
`backend`, `phase`, `kv_layout`, and `methodology_version` selectors. Methodology is derived as
`llm-memory-v2-<backend>-<phase>-<layout>`. This revision activates `cpu`/`decode`/`contiguous`,
`cpu`/`decode`/`paged`, `cpu`/`prefill`/`contiguous`, and `cpu`/`prefill`/`paged`, with exact methodologies
`llm-memory-v2-cpu-decode-contiguous`, `llm-memory-v2-cpu-decode-paged`,
`llm-memory-v2-cpu-prefill-contiguous`, and `llm-memory-v2-cpu-prefill-paged`. It also activates
`metal`/`decode`/`contiguous`, `metal`/`decode`/`paged`, `metal`/`prefill`/`contiguous`, and
`metal`/`prefill`/`paged`, with methodologies `llm-memory-v2-metal-decode-contiguous`,
`llm-memory-v2-metal-decode-paged`, `llm-memory-v2-metal-prefill-contiguous`, and
`llm-memory-v2-metal-prefill-paged`. No selection is silently replaced by another backend, phase, or layout.

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

Checksum validity describes agreement with the versioned accumulator, not an exhaustive proof of all contents,
addresses, or visit multiplicities. A post-validation failure can retain `checksum_valid: true`; the attempt remains
invalid, its rates are null, and it does not enter aggregates. Missing checksum evidence remains null rather than false
or successful. CPU paged decode weights-only checks append slots and padding; CPU paged prefill weights-only checks padding only.
Metal weights-only
reports KV-write and padding inapplicable, so no successful KV-write observation may be inferred from its
combined lifecycle result. Prefill validation remains sampled on both backends. Programmed byte/lookup counts and
plan-derived completion do not measure physical DRAM traffic. Source hashes, ABI goldens, and generated semantic
lists qualify a specific implementation rather than adding runtime observations to the JSON. See the
[profile fault matrix and exact collision limits](LLM_MEMORY_PROFILE_WHITEPAPER.md#checksum-fault-model-and-evidence-limits).
Independent named checks retain these existing detection limits and checksum identities.

The required generic top-level field set is:

```text
schema_version, mode, backend, phase, kv_layout, methodology_version,
software, configuration, resolved_plan, backend_evidence, memory_budget,
calibration, measurements, aggregates, status, reason_code,
results_complete, run_accepted, interpretation, diagnostic, interruption_requested,
scenario_order_balance_complete, seeds, counters, checkpoint_lifecycle, loop_records,
environment, quality_warnings, build_manifest
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
  `prompt_tokens`/`attention_query_tile_tokens` and decimal-string tile and prefix-visit counts. Theoretical
  causal-pair, attention-pair and FMA-term values instead belong to `model_context.prefill`. Decode-only crossover numerator, denominator, and context, current visible context, weight/KV-read
  ratio, and `current_context_classification` are null for prefill. `classification_version` and
  `classification_is_payload_only` remain populated because they describe the schema field's semantics.
- `layout.kv_layout` is the selected `contiguous` or `paged` token. Contiguous results use null for paged-only fields
  and report applicable lookup/read counts as decimal-string zero. An admitted paged plan populates integer
  `kv_block_tokens` and decimal-string block, tail, table-entry, and table-byte counts. A materialized table additionally
  populates permutation domain/algorithm/hash strings and the decimal-string resolved permutation seed. If Metal exits
  unsupported or failed before table preparation, the admitted geometry remains populated while runtime permutation
  and combined layout-identity fields are null.
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
  `llm-memory-components-v1`; CPU MSL fields and contiguous permutation fields are null. Paged CPU and Metal results
  bind the physical layout and permutation identity to their paged resource, schedule, pattern, and checksum
  identities.

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

Paged decode performs one paired append lookup followed by `N` independent K-scan lookups and `N` independent V-scan
lookups for each layer/batch pair. CPU owners and Metal threadgroups use different schedules but preserve the same
semantic count:

```text
layout_metadata_lookup_count_per_work_unit = L * B * (2 * N + 1)
layout_metadata_read_bytes_per_work_unit = 4 * layout_metadata_lookup_count_per_work_unit
accounted_bytes_per_work_unit =
  effective_model_payload_bytes_per_work_unit + layout_metadata_read_bytes_per_work_unit
```

The Metal paged kernel assigns each layer/batch/logical-block owner to exactly one threadgroup through a cyclic
grid-stride schedule. A named lane loads each `device const volatile uint` table entry, publishes the physical ID in
threadgroup memory, and executes a `mem_threadgroup` barrier before address construction. The timed checksum
non-separably binds logical table index, physical ID, append/K-read/V-read visit kind, and work-unit ordinal;
the maintained real-device tests distinguish selected equal-multiplicity non-tail two-entry swaps, not all
possible table permutations. These four-byte lookups and physical suffix padding are reported evidence, not effective
model payload.


CPU final-state validation evidence:

CPU contiguous-decode checks every byte of both K and V append records for every layer and batch
against the cold affine oracle at final task-local work-unit ordinal `T - 1`, after the last worker
stops the timer and all workers join. Failure is `decode-post-validation-failed`, retained as an
invalid attempt with null rates and excluded from aggregate populations. The timed payload and
checksum algorithm are unchanged. A matching runtime checksum does not replace this final-state check.

Both measurements and excluded calibration attempts publish cold checks at `execution.validation.checks[]`.
Each entry contains exactly `kind`, `applicable`, `evaluated`, `valid`, and `reason_code`. The three slots are structure,
the scenario's write/unchanged check, and padding. The five kind tokens are `post-validation-structure`,
`kv-append-final`, `kv-prefill-final-samples`, `kv-append-unchanged`, and `kv-padding-canary`.
Structure describes existing checked prerequisites, not an exhaustive plan proof.

- Inapplicable: `applicable=false`, `evaluated=null`, `valid=null`, reason `not-applicable`.
- Applicable but unresolved: `evaluated=false`, `valid=null`; missing observed evidence cannot pass.
- Evaluated: `evaluated=true`, `valid` is a boolean. An observed mismatch stays false even if another check is unresolved.

KV-bearing decode checks final append state; prefill checks final-ordinal samples, retaining its sampling limits.
CPU paged decode weights-only checks unchanged append slots and applicable padding; CPU paged prefill weights-only
checks structure and applicable padding, not unexpected changes to valid prompt contents. Metal weights-only checks
neither writes nor padding. Padding applies only where terminal suffix padding exists. CPU contiguous decode
weights-only has no applicable structure/write/padding check. CPU prefill retains its structural prerequisite check.
The separate checksum verdict may remain true when a cold check fails.

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
owned V prefix. The operation ordinal is bound into write/checksum evidence, and the intended checksum accumulation includes each
tile-read visit; equality alone does not establish every executed visit. For a KV-bearing CPU task, excluded post-validation checks each owner's deterministic
first/middle/last canonical-word samples, including bytes clipped to owner boundaries, against the final operation
ordinal `T-1`. For a KV-bearing Metal prefill task, it checks representative and boundary byte locations per
layer/batch sequence in both K and V against that ordinal; paged Metal additionally validates applicable terminal
padding canaries. Neither path scans every prompt record. CPU scenario partitions report a stable identity plus
minimum, maximum, and imbalance `worker-cost` evidence.
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

For either Metal paged phase, `resolved_plan.layout` obtains geometry from the admitted Metal paged resource plan and
the materialized permutation identity from runtime table evidence. `resolved_plan.resources.metal.table_segments`
publishes whole-uint32 table segmentation. `backend_evidence.metal.resources` records private-table upload and
validation, the same permutation identity, and combined K/V layout padding. Each task grid reports
`paged_semantic_lookups` and `serial_range_visits_per_lane`; task validation reports the phase-neutral K/V-write result
and applicable padding-canary evaluation. A complete KV-active task requires the expected lookup count, valid checksum,
valid K/V-write validation and, when terminal padding exists, an evaluated-valid padding canary. The runtime checks are published at
`measurements[].execution.validation.checks` and the corresponding calibration execution path. Human-readable Metal
task output summarizes the write result as `kv_write=valid|invalid|not-evaluated|not-applicable`.

The canonical embedded MSL revision and its exact SHA-256 are computed from the selected runtime source and reported
rather than duplicated here. Every profile publishes its selected scenario-pipeline and parameter-layout-probe
identities. Layout-specific validation pipelines are selected and compiled internally; schema evidence publishes their
validation outcomes, but not a separate validation-pipeline label. Metal paged prefill records
executor `llm-metal-executor-v1-prefill-paged`, schedule
`llm-metal-prefill-paged-cyclic-block-owner-grid-stride-v1`, timer
`metal-command-buffer-gpu-start-end-v1`, buffer pattern
`llm-paged-physical-buffer-pattern-v1`, write pattern `llm-metal-prefill-paged-full-prompt-affine32-v1`, and checksum
`llm-metal-paged-prefill-dual-mod32-lookup-address-mix-v1`. Workload and layout-probe labels are output identities; validation
labels are internal selected-profile pipeline identities. None is a promise that future schema revisions retain the
same kernel implementation.

`memory_budget` separates immutable resource geometry from allocation-time evidence and includes canonical
decimal-string `resource_rounding_bytes`, `transient_peak_bytes`, `known_owned_peak_bytes`, and
`admitted_budget_bytes`. A paged candidate admits full physical K/V resources, the resident block table, page rounding,
descriptor/planner/checksum/orchestration storage, and the permutation-validation transient before table
materialization. The sampled available-memory value and its derived admitted budget remain runtime evidence and are
excluded from resource, execution, model, and frozen-plan identities, so identical fixed-seed work retains its identity
when only the admission sample changes. A non-empty JSON target's orchestration reserve uses checked arithmetic for
every variable-length component/layout identity and, for prefill, the aggregate execution identity plus all scenario
and scope identities.
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

Canonical work and runtime observations have separate owners:

- `resolved_plan.plan_identity`, `geometry`, `layout`, `resources`, `component_identities` and `model_work_plan`
  describe the model once. The root methodology selector is not repeated in nested objects.
- `resolved_plan.scenario_plans[]` contains unique scenario/T/explicit plans, including calibration shapes. Plans sort
  by scenario (`weights_only`, `kv_only`, `mixed`), increasing `work_units`, then `explicit_iterations`
  (`false` before `true`). Each has `model_ref: "resolved_plan"`, exact `plan_identity`, scenario seed, work kind,
  work policy, limits, per-work-unit and total payload/lookup/accounted quantities.
- `resolved_plan.frozen_plan_refs` has the three scenario keys, with a zero-based plan-array index or null when unresolved.
  Measurements and `calibration.attempts.<scenario>[]` use `plan_ref` into that same document. Unknown/out-of-range or
  wrong-scenario references are rejected. Internal insertion handles stay stable, but public indices can change between
  snapshots as calibration plans arrive; never join references across documents.
- `measurements[].measurement_id` is its zero-based position in the scheduled measurement array. A measurement retains
  scenario, loop/order, attempted/status/reason, authoritative duration, completed work and mutable runtime evidence.
  `completed_work_units` is an integer. `completed_effective_model_payload_bytes`, `completed_layout_metadata_lookup_count`,
  `completed_layout_metadata_read_bytes` and `completed_task_accounted_bytes` are decimal strings. CPU accepted completion
  has `completion_derivation: "accepted-plan-derived"`; this is not a hardware work counter.
- Work kind (`decode_step` or `prefill_operation`), `work_units`, `kv_write_kind`
  (`none`, `current_token_append`, `full_prompt_population`), per-work-unit quantities and planned totals belong to the
  referenced plan. Its total names are `effective_model_payload_bytes`, `layout_metadata_lookup_count`,
  `layout_metadata_read_bytes` and `task_accounted_bytes`, without a `planned_` prefix. Measurement workers, QoS,
  calibration attempt indexes, working-set description and mixed payload fractions remain derived metadata.

Each canonical plan has an `expected_checksum` object with exactly `status`, `reason_code`,
`expected_worker_checksums`, and `expected_run_checksum`. Status `available` means the cold expectation was
reconstructed once for that plan; it is not an observed runtime pass. Status `unavailable` makes both expected fields
null and retains the reason. CPU available worker arrays have increasing `worker_index` and exactly effective-workers
entries; each has `weight`, `k`, `v` objects with decimal-string `state_a_uint64_decimal`, `state_b_uint64_decimal`,
`exact_bytes_read`, and `span_count_uint64_decimal`. Zero-work domains retain actual algorithm initial values.
CPU run checksum has the two state fields. Metal expected workers are null; its run object has `weight`, `k`, `v`,
each with `a_uint32_decimal` and `b_uint32_decimal`. CPU and Metal checksums are never compared numerically.

Measurement `checksum` contains exactly `status`, `reason_code`, `checksum_valid`, `actual_worker_checksums`, and
`actual_run_checksum`. Unevaluated checksum has null validity and both actual fields null; evaluated invalid retains
actuals and false validity. Calibration keeps the compact actual-run-only counterpart at
`calibration.attempts.<scenario>[].execution.checksum`, with no actual-worker vector. All expectations resolve through
that record's plan reference. Algorithm identity comes from canonical `component_identities.checksum_pattern_version`.

The one accepted `elapsed_seconds` and completed work/payload derive `synthetic_work_unit_latency_seconds`,
`synthetic_memory_work_units_per_second`, and `effective_model_payload_gb_s`. Invalid/excluded measurements have null
rates and accepted elapsed; any observed time is retained as `execution.timing.diagnostic_elapsed_seconds`. Metal retains
raw GPU start/end, host submit/wait envelope, nullable same-clock queue delay, pipeline/grid and command/encoder/dispatch
observations. Queue delay is null unless measured in the same clock domain. Effective payload includes logical W/K/V
bytes; timed table metadata stays outside that numerator.

Each Metal task uses one reset command buffer, one timed command buffer with one explicitly serial compute encoder and
one workload dispatch, and one excluded post-validation command buffer. The task's `T` work units loop inside the
kernel. Initialization, pre-touch, expected-checksum construction, reset, and post-validation are outside the timed
window. `GPUStartTime`/`GPUEndTime` are read only after completion; zero, non-finite, negative, or non-increasing values
invalidate the task. Exact vector/scalar tails and segment-boundary splits preserve the planned logical byte count.
For contiguous prefill, every lane performs exactly
`T * ((weight_active ? L : 0) + (kv_active ? 2 * L * B * (P + C) : 0))` serial range-helper visits. Paged prefill
`weights_only` performs `T * L`; its KV-bearing owner kernels report zero serial range visits and use the semantic-lookup
guardrail below. The checked count is published as the decimal-string `serial_range_visits_per_lane` grid field and may
not exceed 1,048,576. `resolved_plan.methodology.maximum_serial_range_visits_per_lane_per_task` publishes that integer
cap for both Metal prefill layouts and is null for other profiles. Scenario planning reduces
the effective work-unit ceiling so explicit work and automatic calibration cannot cross it. The runtime grid retains
`serial-range-visit-count-overflow` and `serial-range-visit-cap-exceeded` safeguards before expected-checksum
enumeration and GPU dispatch.

For KV-active paged prefill, each layer/batch pair performs `N` paired full-prompt-write lookups followed by `M` K-prefix and `M`
V-prefix lookups, where `M = sum(ceil(e_j/G))` across query-tile ends. Thus every KV-active operation reports exactly
`L * B * (N + 2 * M)` timed uint32 loads. Its row-major owner count is `L * B * N`; each owner ordinal is assigned to
one threadgroup by a deterministic cyclic grid stride for all work units and semantic visits. Grid evidence publishes
the owner count, actual threadgroups, owner ordinals per threadgroup, exact per-threadgroup accounted-byte vector,
minimum, maximum, and max-minus-min imbalance with `cost_unit: "actual-threadgroup-cost"`. Paged `weights_only` uses
the weight-vector grid-stride schedule and reports zero metadata work. The same exact weight-grid cost evidence is
published for every Metal `weights_only` task. Contiguous KV-bearing grids retain the grid geometry but publish null
cost unit/minimum/maximum/imbalance fields and an empty cost vector. The cyclic KV schedule is deliberately not
described as weighted-balanced.
Partial prefix visits stop at the exact tile or prompt end. KV-bearing prefill tasks require final-operation
representative/boundary sample validation of the prompt writes, plus applicable suffix-padding validation.

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

Every paged semantic visit loads its uint32 physical ID inside the timed backend path and calculates the block address
after that load; the host does not replace the table with a pre-resolved pointer list. CPU uses the ARM64 kernel path;
Metal uses the named-lane volatile-load and threadgroup-publication path described above. The traversal is append,
complete logical K-block scan, then complete logical V-block scan for each layer/batch pair, with mixed reading the
layer weight span first. Paged prefill instead writes all owned prompt blocks first, then for each query tile scans the
exact K prefix followed by the exact V prefix. The paged checksum binds logical table index, loaded physical ID,
semantic visit kind, and work-unit ordinal non-separably. Physical data patterns depend on pool, physical ID, and
physical offset. Post-task validation checks decode current-token writes or prefill final-ordinal prompt samples plus
applicable terminal padding canaries. Generation, validation, initialization, pre-touch, expected-checksum construction,
and post-validation are outside the authoritative synchronized CPU interval or Metal GPU interval.

Metal prefill with contiguous KV uses the same logical prefill traffic contract without block-table lookups. Within each
`prefill_operation`, weight-bearing scenarios read each active layer's weights once, KV-bearing scenarios populate all
`P` K/V records, and each lane scans only its own written slices of the complete K prefix followed by the complete V
prefix at each tile end. No grid-wide barrier is implied. All `T` operations execute inside the single timed workload
dispatch. The commutative checksum binds scenario, layer, batch, byte domain, and work-unit ordinal; the
source-hash-bound entrypoint audit separately checks the build’s write-before-tiled-read loop nesting, rather than
recording a runtime GPU trace.

The traffic classification version is `llm-exact-weight-vs-kv-read-payload-v1`: it compares exact active-weight bytes
with exact KV-read bytes only. `near_crossover` means equality, not a tolerance band and not an observed hardware
bottleneck.

`results_complete` means all planned measurements are terminal and measured. `run_accepted` additionally requires
complete run status, required accepted timing/checksum/cold-check and backend completion/lifecycle evidence, and no
known checkpoint or command failure. It is independent of position balance, sample count, CV, duration and environment.
A correct count-one run can have both booleans true and `scenario_order_balance_complete: false`. A late command error
can leave the measured population complete while run status is failed and `run_accepted` is false. Consumers retain
process exit status as well as the snapshot; an older preserved file cannot report a later failure.

`scenario_order_balance_complete` describes positions only: all planned rows must be measured, each loop complete, and
each scenario's first/middle/last counts equal and nonzero. Loop `i` rotates `weights_only`/`kv_only`/`mixed` by `i
mod 3`, independently of seed. This does not balance directed predecessor pairs, including loop and repeated-block
boundaries. Partial final blocks retain their realized order. Canonical frozen-plan warmups run once before loop zero,
not immediately before each measured scenario; backend task-local preparation and validation remain outside the
primary timer.

Each `aggregates.scenarios.<scenario>.accepted_measurement_ids` defines one population shared by all three metrics.
Excluded calibration never enters it. Derive each rate before applying the shared linear percentile interpolation;
for even n, median(work/duration) generally differs from work/median(duration). Exact median/MAD/percentiles are
prepared only at snapshots or terminal, with reusable workspaces; adding or printing a raw measurement does not sort
all earlier samples. Statistics retain average, min/max, median, P90/P95/P99, sample stddev, CV percent and MAD.
Empty populations have null statistics/headlines; n=1 has stddev=0 and positive-mean CV=0, while classification remains
`insufficient-samples`. `observed_cv_classification` is `insufficient-samples` for n<3, `undefined` for undefined CV,
`above-threshold` for payload CV strictly greater than `cv_warning_threshold_pct` (5), otherwise `below-threshold`.
Equality is below-threshold. These describe observed samples, not reproducibility guarantees; percentiles are not
confidence levels. Default LLM console output shows n, median, min/max, CV and MAD, without tail percentiles.

A comparison may impose stricter quality criteria, but those are separate policy and do not delete or retry noisy
measurements. Match phase/model geometry and, for paged results, block size, permutation, physical-resource geometry
and component identities. Contiguous and paged results are distinct cohorts even with identical logical geometry.
Match backend/profile, frozen work, seeds, full component identities and run policy, including conditioning and
output/checkpoint cadence. Inspect accepted n, CV, duration and environment separately from acceptance.
`environment.start`/`end` thermal-state and Low Power Mode values are instantaneous OS observations, not continuous
temperature, cache-residency or CPU-affinity evidence; unavailable is not nominal. Accepted artifacts alone do not
establish a machine or methodology performance difference.

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

The bundled standard-memory example scripts accept compatible producer releases. Before reading their standard
schema-3 metric paths, they require `configuration.mode == "benchmark"`, schema 3, methodology
`benchmark-v2-calibrated-seeded-balanced`, the completion fields above, and the expected types for every consumed
field. They retain a non-empty top-level `version` string as release provenance, but exact software-version equality is
not an acceptance condition. The examples do not translate released standard schema 2, unversioned historical
standard JSON layouts, or other methodology identities through a metric-shape fallback.

Graceful interruption or runtime failure after a representable result state has been initialized emits the available
partial, interrupted, error, or failed JSON snapshot. The execution status and payload are independent: a non-zero status
must not cause a caller to discard evidence without parsing it. Exit status zero is used for human help, complete
execution, and established graceful-interruption paths; it alone does not prove that JSON conclusions are complete.

## LLM schema-1 to schema-2 field map

This table describes the intentional cutover, not an automatic reader. Unlisted supported geometry, resource,
seed, runtime and interpretation fields retain their meaning. Schema/version checks must precede field consumption.

| Schema 1 path or behavior | Schema 2 owner or behavior |
|---|---|
| `schema_version: 1`, `llm-memory-v1-*` selectors | `schema_version: 2`, the eight exact v2 selectors above |
| `conclusions_valid` | `run_accepted`; correctness is independent of balance/quality |
| `resolved_plan.methodology_version`, `resolved_plan.model_work_plan.methodology_version`, `resolved_plan.methodology.methodology_version` | root `methodology_version` |
| methodology component-version copies | `resolved_plan.component_identities` only; `run_policy_version` is `llm-run-policy-bounded-loop-snapshots-v1` |
| `resolved_plan.model_work_plan.plan_identity` | `resolved_plan.plan_identity` |
| `resolved_plan.model_work_plan.component_identity` | `resolved_plan.component_identities.identity` |
| `backend_evidence.cpu.resources.model_plan_identity` | `model_ref: "resolved_plan"` when resolved, otherwise null |
| `resolved_plan.frozen_scenario_work_plans` wrapper and `scenarios[]` | `resolved_plan.scenario_plans[]` plus `resolved_plan.frozen_plan_refs` |
| frozen wrapper/scenario `model_plan_identity` | canonical scenario `model_ref: "resolved_plan"`; wrapper removed |
| `measurements[].frozen_plan_index`, `frozen_work_plan_identity`, execution model/scenario identity copies | measurement `plan_ref`; canonical plan carries its exact identity and `model_ref` |
| `measurements[].scenario_seed_uint64_decimal`, `explicit_iterations`, `work_policy`, `work_unit_kind`, `kv_write_kind`, `*_bytes_per_work_unit`, `planned_*` | referenced canonical plan, with `work_units` and total names without `planned_` |
| implicit measurement array position | explicit `measurement_id`, still the zero-based schedule index |
| `measurements[].checksum.expected_*`, `execution.metal.checksum` | referenced plan's `expected_checksum`; runtime actuals remain in measurement `checksum` |
| `calibration.attempts.<scenario>[].work_plan_identity`, planned payload and execution expected run checksum | `calibration.attempts.<scenario>[].plan_ref`; compact runtime actual stays under `execution.checksum` |
| runtime checksum algorithm/write-version copies | canonical component checksum/write versions |
| `execution.post_validation_evaluated/valid`, CPU `kv_write_validation_*`, Metal `execution.metal.validation` | measurement/calibration `execution.validation.checks[]` independent observations |
| duplicate accepted elapsed values and retained metric sample vectors | authoritative measurement elapsed/work/payload; derived metrics and shared `accepted_measurement_ids` |
| `aggregates.scenarios.<scenario>.stability_quality` | `observed_cv_classification` plus explicit threshold and sample counts |
| `checkpoint_lifecycle.logical_checkpoint_attempts`, `successful_logical_checkpoints`, terminal checkpoint flags | prior actual writer attempts/successes at the named observation point; current persistence unresolved |
| `geometry.prefill` causal-pair, logical-attention-pair and FMA context | nullable `resolved_plan.model_context.prefill` scalars with individual `*_reason_code` |
| no build reservation | always-present `build_manifest` object; unavailable fields remain null |

`model_context.prefill` is null for decode. For prefill its exact children are
`causal_token_pairs_per_sequence`, `logical_attention_pairs`, `logical_attention_fma_terms` and each one's
`<name>_reason_code`. Available values are decimal strings with reason `valid`; theoretical arithmetic overflow gives
null with `arithmetic-overflow`. Available values retain numeric exact-identity encoding; unavailable identity fields
use `unavailable:arithmetic-overflow`. Actual byte/prefix/lookup/allocation overflows still reject the workload.
`h_q` (`--query-heads`) controls theoretical attention context; query tile `Q` (`--attention-query-tile-tokens`)
controls executed prefix-read geometry. No Transformer or FMA computation is performed.

`build_manifest` version 1 records build-time `git_commit` and `git_dirty`, compiler version,
`compile_flags` (`cxxflags`, `test_cxxflags`, `asflags`), effective link flags including frameworks,
compiler target triple (`target_arch`), SDK version and `min_os`. Make regenerates the embedded manifest
when those inputs change and invalidates the object graph. Python 3 is required for this build step.
A source archive without Git metadata records null Git fields and status `partial` with
`build-fields-unavailable`; it never consults the run directory's Git state. The software version remains
separate provenance. A caller supplying no manifest retains the explicit `unavailable` reservation.
The command streams `binary_sha256` once before tasks, from the executable path. Read failure leaves
null with status `partial` and `binary-hash-unavailable`. This is a file binding, not signed attestation
of executed machine code; replacement of the executable during capture is outside the claim.

CPU measurement `execution.timing.cpu_raw` and excluded-attempt `execution.timing.cpu_raw` contain
`start_ticks`, `stop_ticks`, `delta_ticks` as uint64 decimal strings and `timebase_numer` and
`timebase_denom` as uint32 JSON integers. Null means no captured snapshot; older schema-2 artifacts may
omit this optional field. The shared timer captures the original boundaries with one clock read each,
then computes `delta=(stop-start) mod 2^64` (at most one wrap assumed) and
`elapsed_seconds=(double(delta)*double(numer)/double(denom))/1e9`. Floating multiplication avoids
integer overflow. Zero delta or numerator produces zero and cannot admit an LLM measurement; a zero
denominator rejects timer creation, or defensively returns zero if externally corrupted afterwards.
A new start clears the prior snapshot. Invalid measurements keep diagnostic elapsed and any original
raw snapshot; they retain null headline/rate fields. Metal has null CPU raw evidence and retains its
GPU timestamps. CPU `completion_derivation=accepted-plan-derived` refers to accepted planned work;
the kernel's exact-byte checksum counter is programmed evidence, not measured hardware traffic.

All byte, lookup, seed and checksum values follow canonical unsigned decimal-string encoding (`0` or
`[1-9][0-9]*`); Metal lanes retain uint32 bounds. Indexes, work units and validated small inputs remain bounded JSON
integers below 2^53, never booleans or coerced strings. Null separates unavailable/inapplicable from known zero.
Canonical plan/expected vector capacities, transient expected reconstruction, accepted-ID maps, exact-stat workspaces,
DOM and stdout serialized strings are charged together at their simultaneous peak; bounded snapshot count does not
reduce the single-snapshot peak allowance.

## Independent LLM artifact verifier

`python3 script-examples/verify_llm_result.py result.json` reads exactly one schema-2 `llm_memory` document.
It supports the eight current CPU/Metal × decode/prefill × contiguous/paged component sets. Other schemas,
methodologies and profiles are unsupported; there is no migration or historical field fallback. The independent
Python arithmetic never imports or invokes producer helpers or executes the optional binary.

The JSON verdict separates `artifact_consistent` from `run_accepted`. Exit 0 means a consistent accepted artifact;
exit 1 means inconsistency or a consistently described unaccepted run; exit 2 means unsupported schema/profile,
a resource bound or a requested evidence level that is unavailable. `artifact_consistent` is null on unsupported
cases. An artifact alone cannot establish the original process exit status; the process acceptance procedure below
still applies to callers launching the producer.

Checks reconstruct geometry, frozen work, payload/layout/lookup quantities, seed derivation and the canonical
Fisher–Yates table digest; bind component, model, scenario, layout and nested identity evidence; derive independent
CPU worker/run and Metal dual-mod32 expectations once per canonical plan; check original timing, rates, named
validation, ordered measurement prefix, accepted populations, exact statistics, counters and run acceptance.
The CPU oracle reconstructs owner ranges, including scenario-specific prefill prefix-cost partitions. Affine range
sums avoid materializing weight or KV pools. The verifier does not reproduce allocator capacity, host admission
samples or GPU traces; those remain reported implementation/environment evidence. Identity/hash agreement binds
reported content and never authenticates a build or a runtime observation.

`checks.checksum` is `independent-oracle` when expectations were available, otherwise `unavailable`.
`checks.timing` is `cpu-raw-ticks`, `gpu-timestamps`, `elapsed-only`, or `unavailable`. Optional CPU raw evidence
may be absent from earlier schema-2 artifacts; rates are still checked but duration reconstruction is limited.
`--require-raw-timing` rejects missing CPU raw evidence with `unsupported-missing-raw-timing`.
`checks.build` is `manifest-only`, `unavailable`, or `binary-bound`. `--binary PATH` streams the supplied file's
SHA-256 and requires equality with the manifest. It does not run the file. A correctly described failed,
partial, interrupted or unsupported run can be consistent while `run_accepted` is false.

Counts use exact integers and the producer's canonical decimal representation. Derived floats use relative
tolerance `1e-10` and absolute tolerance `1e-15`; integer equality never uses float tolerance. CPU conversion
uses the documented double evaluation order. Zero/nonfinite accepted durations and invalid rates are rejected.
Statistics use sample standard deviation (n−1; singleton zero), linearly interpolated quantiles at `(n−1)p`,
median absolute deviation and the accepted measurement IDs only. Empty populations require null statistics.

Input limits are 64 MiB UTF-8 JSON, depth 64, one million JSON nodes, 100,000 measurements, 1,024 canonical
plans, 1 MiB per identity and one million charged oracle/partition steps across the artifact. Decimal uint64
strings have at most 20 digits. Duplicate JSON keys, booleans used as integers and nonfinite JSON numbers
are rejected. A limit produces an explicit unsupported verdict, never successful verification; large calibrated
work can exceed this verifier's budget even when the benchmark itself supports it. Parsing errors do not become
partial success. Exact acceptance claims remain subject to the checksum collision boundaries documented in the
[LLM whitepaper](LLM_MEMORY_PROFILE_WHITEPAPER.md#checksum-fault-model-and-evidence-limits), and the verifier
cannot replace kernel qualification tests, prove executed instructions or establish physical GPU/DRAM traffic.

`make test-llm-verifier` runs independent small arithmetic goldens, all eight captured current profile fixtures,
status fixtures, malformed input and specific artifact mutations. `make test-all` includes this separate gate
alongside the existing standard-memory script-example gate.

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
8. Apply the selected metric's mode-specific status, non-null, and quality predicates described above.
   For LLM, balance, observed CV, duration quality, and environment are separate comparison-quality context;
   they do not add correctness predicates to `run_accepted`.

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
       .configuration.methodology_version == "benchmark-v2-calibrated-seeded-balanced" and
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

jq -e '. as $root | .mode == "llm_memory" and .schema_version == 2 and
       .backend == "cpu" and .phase == "decode" and .kv_layout == "paged" and
       .methodology_version == "llm-memory-v2-cpu-decode-paged" and
       .resolved_plan.layout.kv_block_tokens == 16 and
       (.resolved_plan.layout.permutation_sha256 |
        type == "string" and test("^[0-9a-f]{64}$")) and
       .resolved_plan.component_identities.run_policy_version ==
         "llm-run-policy-bounded-loop-snapshots-v1" and
       .status == "complete" and .results_complete == true and .run_accepted == true and
       (.resolved_plan.scenario_plans | type) == "array" and
       (.measurements | type) == "array" and
       .counters.planned_measurements > 0 and
       (.measurements | length) == .counters.planned_measurements and
       all(.measurements[];
           .status == "measured" and (.plan_ref | type) == "number" and
           .plan_ref >= 0 and .plan_ref == (.plan_ref | floor) and
           .plan_ref < ($root.resolved_plan.scenario_plans | length) and
           $root.resolved_plan.scenario_plans[.plan_ref].scenario == .scenario)' llm_memory.json

jq -e '. as $root | .mode == "llm_memory" and .schema_version == 2 and
       .backend == "cpu" and .phase == "prefill" and .kv_layout == "contiguous" and
       .methodology_version == "llm-memory-v2-cpu-prefill-contiguous" and
       .resolved_plan.geometry.decode == null and
       .resolved_plan.geometry.prefill.prompt_tokens == 512 and
       .resolved_plan.geometry.prefill.attention_query_tile_tokens == 64 and
       .resolved_plan.component_identities.run_policy_version ==
         "llm-run-policy-bounded-loop-snapshots-v1" and
       .status == "complete" and .results_complete == true and .run_accepted == true and
       (.resolved_plan.scenario_plans | type) == "array" and
       (.measurements | type) == "array" and
       .counters.planned_measurements > 0 and
       (.measurements | length) == .counters.planned_measurements and
       all(.measurements[];
           .status == "measured" and (.plan_ref | type) == "number" and
           .plan_ref >= 0 and .plan_ref == (.plan_ref | floor) and
           .plan_ref < ($root.resolved_plan.scenario_plans | length) and
           $root.resolved_plan.scenario_plans[.plan_ref].scenario == .scenario)' llm_prefill.json
```

The two LLM examples consume the producer's acceptance and validate same-document plan joins; they do not independently
re-execute checksum/timing or attest to the build. Supply your requested geometry and cohort checks in addition to the
shown block-size or prompt/tile requirements. A noisy or one-loop correct result passes; comparison-quality policy is
separate. Require the successful producer process outcome before applying `jq -e`.

## Compatibility policy

- `version`, the GPU `software_version` field, and LLM `software` identity identify the application release; none is a
  result schema version or compatibility selector. Consumers may retain and display this provenance independently of
  schema and methodology acceptance.
- Current standard schema 3, pattern schema 3, TLB schema 4, core-to-core schema 2, GPU schema 1, and LLM schema 2 remain
  authoritative at their existing locations. The schema field is intentionally not normalized across these established
  payloads.
- LLM schema 2 deliberately replaces the unpublished schema-1 shape and changes methodology selectors. The concrete
  field map describes relocation and changed acceptance; no automatic v1 reader, alias or migration layer is provided.
  Future removal/rename/type/meaning changes require schema-version review.
- Bundled standard-memory examples accept compatible software releases only when standard mode, schema 3, the exact
  methodology identity, completion state, and consumed field shapes match. They retain software-version provenance but
  provide no translation layer for released standard schema 2, unversioned historical standard JSON layouts, or other
  methodology identities.
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

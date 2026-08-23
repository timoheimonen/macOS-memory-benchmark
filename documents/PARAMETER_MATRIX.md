# Parameter Compatibility Matrix

Working version `0.63.0`

## All Flags

| Short alias | Long option | Value | Description |
|-------------|-------------|-------|-------------|
| `-B` | `--benchmark` | — | Run the standard CPU bandwidth/latency benchmark |
| `-P` | `--patterns` | — | Run the CPU access-pattern benchmark |
| `-T` | `--analyze-tlb` | — | Run standalone TLB analysis |
| `-C` | `--analyze-core2core` | — | Run standalone two-thread acquire/release token-protocol handoff analysis |
| `-G` | `--gpu-bandwidth` | — | Run standalone Metal GPU memory bandwidth |
| `-M` | `--llm-memory` | — | Run the standalone CPU/Metal synthetic LLM memory profile |
| — | `--llm-memory-backend` | `cpu\|metal` | LLM execution backend; default `cpu`; CPU and the experimental Metal preview support decode/prefill with contiguous or paged KV |
| — | `--weight-size-mb` | `<MiB>` | Required positive active weight size for LLM-memory mode |
| — | `--layers` | `<count>` | Required positive LLM layer count |
| — | `--query-heads` | `<count>` | Required positive query-head count; at least the KV-head count and divisible by it |
| — | `--kv-heads` | `<count>` | Required positive physical KV-head count |
| — | `--head-dim` | `<count>` | Required positive K/V head-vector element count |
| — | `--kv-element-bytes` | `1\|2\|4` | LLM KV element width; default `2` |
| — | `--phase` | `decode\|prefill` | LLM phase; default `decode` |
| — | `--context-tokens` | `<count>` | Required only for decode; fixed visible context including the current synthetic token |
| — | `--prompt-tokens` | `<P>` | Required only for prefill; positive full-prompt token count |
| — | `--attention-query-tile-tokens` | `<Q>` | Required only for prefill; query-tile size in `1..P` |
| — | `--batch-size` | `<count>` | Positive LLM batch-sequence count; default `1` |
| — | `--kv-layout` | `contiguous\|paged` | LLM KV layout; default `contiguous` |
| — | `--kv-block-tokens` | `<G>` | Required only for paged KV: a positive power of two no greater than `UINT32_MAX`; may exceed the active phase length |
| `-i` | `--iterations` | `<count>` | Positive exact R/W/Copy pass count or LLM scenario work-unit count; CPU standard maximum is `INT_MAX`, while GPU and LLM apply work-dependent ceilings. Omission enables automatic calibration in the applicable mode |
| `-b` | `--buffer-size` | `<MB>` | Default `512` MB. Standard mode permits `0` only with `--only-latency`; pattern mode requires a positive value; GPU minimum is `64` MB |
| `-r` | `--count` | `<count>` | Positive loop count; default `1` for benchmark/pattern modes and `3` for core-to-core/GPU/LLM modes |
| — | `--seed` | `<uint64>` | Unsigned 64-bit reproducibility seed for benchmark, pattern, TLB, GPU, or LLM mode; generated once when omitted |
| `-n` | `--latency-samples` | `<count>` | Positive sample-window count up to `INT_MAX`; default `1000` in benchmark and core-to-core modes |
| `-s` | `--latency-stride-bytes` | `<bytes>` | Positive, pointer-aligned latency-chain stride; default `256` bytes |
| `-m` | `--latency-chain-mode` | `<mode>` | Chain policy: `auto` (default), `global-random`, `random-box`, `same-random-in-box`, or `diff-random-in-box` |
| `-l` | `--latency-tlb-locality-kb` | `<KB>` | Latency-chain locality window; default `1024` KB. With `auto`, `0` selects global random |
| `-D` | `--tlb-density` | `low\|medium\|high` | Standalone TLB runtime profile; default `medium` |
| `-t` | `--threads` | `<count>` | Positive requested bandwidth worker count. General main-memory/pattern execution caps above detected cores; CPU LLM preserves requested, detected, and executable effective counts separately, while Metal LLM rejects this option |
| `-k` | `--cache-size` | `<KB>` | Custom cache target: `16..1048576` KB, or `0` only with `--benchmark --only-latency` |
| `-W` | `--only-bandwidth` | — | Run only standard benchmark bandwidth tests; requires `--benchmark` |
| `-L` | `--only-latency` | — | Run only standard benchmark latency tests; requires `--benchmark` |
| `-u` | `--non-cacheable` | — | Apply best-effort cache-discouraging allocation hints; does not create truly uncached memory |
| `-o` | `--output` | `<target>` | Result target syntax. Exact `-` selects one final stdout document for result-producing modes. An empty direct value disables JSON; every other non-empty value is a file. LLM file output checkpoints each terminal scenario measurement and command terminal |
| `-S` | `--sweep` | `<key=a,b>` | Add a Cartesian sweep parameter; repeat once per distinct key and use with `--output` |
| `-X` | `--sweep-max-runs` | `<count>` | Positive generated-run limit; default `256`, or `16` with `--analyze-tlb`; effective only with `--sweep` |
| `-h` | `--help` | — | Show help; the standalone `--analyze-tlb` whitelist is the exception and rejects this combination |

Short and long forms are equivalent. The compatibility tables below use long forms as canonical names; the GPU and LLM
tables also repeat their exact whitelist aliases. LLM model-geometry options and `--seed` have no short aliases. Long options require two
dashes, short options are exactly one character, and short options cannot be bundled. The parser does not support
`--option=value` syntax. Options that take one value may appear at most once, except that `--sweep` may be repeated for
distinct parameter keys. Numeric values must be complete decimal tokens without whitespace, a leading `+`, or trailing
characters; unsigned seeds reject either sign.

The chain-mode names in the table are the canonical forms written to output. Parsing is case-insensitive, treats hyphens,
spaces, and underscores as equivalent separators, and also accepts the legacy aliases `global`,
`random-in-box-random-box`, `same-random-in-box-increasing-box`, and `diff-random-in-box-increasing-box`. TLB analysis
rejects every spelling that resolves to `global-random`. In contrast, TLB density accepts only the exact lowercase input
values `low`, `medium`, and `high`; quick/standard/exhaustive are profile descriptions, not input values.

## Compatibility Matrix

### Mode Flags (exactly one distinct primary mode required for benchmark execution)

| | `--benchmark` | `--patterns` | `--analyze-tlb` | `--analyze-core2core` | `--gpu-bandwidth` | `--llm-memory` |
|---|---|---|---|---|---|---|
| `--benchmark` | ✅ | ❌ mutually exclusive | ❌ | ❌ | ❌ | ❌ |
| `--patterns` | ❌ mutually exclusive | ✅ | ❌ | ❌ | ❌ | ❌ |
| `--analyze-tlb` | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| `--analyze-core2core` | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| `--gpu-bandwidth` | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| `--llm-memory` | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

### Modifiers with `--benchmark`

| Modifier | Compatible | Notes |
|----------|------------|-------|
| `--iterations <n>` | ✅ | Positive exact pass override; omission enables automatic duration calibration |
| `--buffer-size <MB>` | ✅ | Default `512`; must be positive except for the `--only-latency` disabling case below. Oversized requests can be reduced to the memory-safety cap with a warning |
| `--count <n>` | ✅ | Positive integer; default `1` |
| `--seed <uint64>` | ✅ | Reproduces workload/schedule metadata across count loops; generated once when omitted |
| `--latency-samples <n>` | ✅ | Positive integer; default `1000`. Samples use a separate pass and do not define the continuous headline; the effective count is capped to the measurement access count |
| `--latency-stride-bytes <n>` | ✅ | Positive and pointer-aligned; each enabled target/configured locality window needs two nodes. The optional fixed 16 KiB comparison is unavailable above an 8192 B stride. Accepted but unused by `--only-bandwidth` after validation |
| `--latency-chain-mode <mode>` | ✅ | Box modes require `--latency-tlb-locality-kb > 0`; `global-random` works with locality `0`. Accepted but unused by `--only-bandwidth` after validation |
| `--latency-tlb-locality-kb <n>` | ✅ | Locality-using modes require a non-zero page-size multiple; explicit `global-random` ignores the value. Any explicit value disables the automatic 16 KiB vs global-random comparison. Accepted but unused by `--only-bandwidth` after validation |
| `--threads <n>` | ✅ | Omitted main-memory bandwidth uses detected cores and omitted cache bandwidth uses one worker; an explicit value applies to both. Accepted but unused by `--only-latency`, whose latency paths are single-threaded |
| `--cache-size <KB>` | ✅ | Replaces auto L1/L2 cache tests with one custom cache target |
| `--only-bandwidth` | ✅ | ❌ with `--cache-size`, ❌ with `--latency-samples` |
| `--only-latency` | ✅ | ❌ with `--iterations`. At least one latency target must remain enabled; `--buffer-size 0 --cache-size 0` is invalid |
| `--non-cacheable` | ✅ | |
| `--output <target>` | ✅ | Exact `-` selects final JSON stdout; an empty direct value disables JSON and an empty sweep value is invalid; every other non-empty value is a file, including `./-` and `-G` |
| `--sweep <key=a,b>` | ✅ | Requires `--output`; supported keys depend on benchmark subtype, see [Sweep Compatibility](#sweep-compatibility) |
| `--sweep-max-runs <n>` | ✅ | Default `256`; accepted without `--sweep` but has no effect then |
| `--tlb-density <low\|medium\|high>` | ❌ | Parsed only by standalone `--analyze-tlb` |
| `--help` | ✅ | Prints general help and exits without running a benchmark |
| `--buffer-size 0` | ✅ only with `--only-latency` | Disables main memory latency |
| `--cache-size 0` | ✅ only with `--only-latency` | Disables cache latency |

### Modifiers with `--patterns`

| Modifier | Compatible | Notes |
|----------|------------|-------|
| `--iterations <n>` | ✅ | Positive exact pass override; omission enables automatic duration calibration |
| `--buffer-size <MB>` | ✅ | Default `512`; must be positive because pattern mode cannot use the latency-only disabling case. Oversized requests can be reduced to the memory-safety cap with a warning |
| `--count <n>` | ✅ | Positive integer; default `1` |
| `--seed <uint64>` | ✅ | Reproduces random workload; generated once when omitted |
| `--latency-samples <n>` | Accepted, ignored | Positive value must parse; pattern mode has no latency path |
| `--latency-stride-bytes <n>` | Accepted, ignored | Value must validate; pattern mode has no latency pointer chain |
| `--latency-chain-mode <mode>` | Accepted, ignored | Mode/locality combination must validate; pattern mode has no latency pointer chain |
| `--latency-tlb-locality-kb <n>` | Accepted, ignored | Value must validate; pattern mode has no latency pointer chain |
| `--threads <n>` | ✅ | Omission uses the detected CPU core count; sparse-stride work may reduce the effective worker count |
| `--cache-size <KB>` | Accepted, ignored | Value must be `16..1048576`; `0` is rejected because it is reserved for `--benchmark --only-latency` |
| `--only-bandwidth` | ❌ | Separate execution mode |
| `--only-latency` | ❌ | Separate execution mode |
| `--non-cacheable` | ✅ | |
| `--output <target>` | ✅ | Exact `-` selects final JSON stdout; an empty direct value disables JSON and an empty sweep value is invalid; every other non-empty value is a file, including `./-` and `-G` |
| `--sweep <key=a,b>` | ✅ | Requires `--output`; supported keys: `buffer-size`, `threads` |
| `--sweep-max-runs <n>` | ✅ | Default `256`; accepted without `--sweep` but has no effect then |
| `--tlb-density <low\|medium\|high>` | ❌ | Parsed only by standalone `--analyze-tlb` |
| `--help` | ✅ | Prints general help and exits without running a benchmark |

### Modifiers with `--analyze-tlb` (standalone mode)

| Modifier | Compatible | Notes |
|----------|------------|-------|
| `--output <target>` | ✅ | Exact `-` selects final JSON stdout; an empty direct value disables JSON and an empty sweep value is invalid; every other non-empty value is a file, including `./-` and `-G` |
| `--latency-stride-bytes <n>` | ✅ | Must be positive, pointer-aligned (8 bytes on Apple Silicon), and no larger than the system page size; exact page-size divisibility is not required |
| `--latency-chain-mode <mode>` | ✅ | `global-random` is rejected with `--analyze-tlb` |
| `--tlb-density <low\|medium\|high>` | ✅ | Default `medium`/standard; low=quick, high=exhaustive |
| `--seed <uint64>` | ✅ | Fixed reproducibility seed; generated once when omitted |
| `--sweep <key=a,b>` | ✅ | Requires `--output`; supported keys: `latency-stride-bytes`, `latency-chain-mode`, `tlb-density` |
| `--sweep-max-runs <n>` | ✅ | Default `16`; accepted without `--sweep` but has no effect then |
| `--help` | ❌ | The standalone TLB parser has an exact whitelist and rejects `--analyze-tlb --help`; use `--help` without this mode flag |
| All others | ❌ | Rejected by the standalone whitelist |

### Modifiers with `--analyze-core2core` (standalone mode)

| Modifier | Compatible | Notes |
|----------|------------|-------|
| `--output <target>` | ✅ | Exact `-` selects final JSON stdout; an empty direct value disables JSON and an empty sweep value is invalid; every other non-empty value is a file, including `./-` and `-G` |
| `--count <n>` | ✅ | Core-to-core default `3` (general default remains `1`); scenario order rotates and the headline is the loop median P50 |
| `--latency-samples <n>` | ✅ | Positive integer; default `1000`. Separate calibrated sample windows per scenario/loop do not define the continuous headline |
| `--sweep <key=a,b>` | ✅ | Requires `--output`; supported keys: `count`, `latency-samples` |
| `--sweep-max-runs <n>` | ✅ | Default `256`; accepted without `--sweep` but has no effect then |
| `--help` | ✅ | Prints general help and exits without running core-to-core analysis |
| All others | ❌ | Rejected by the standalone whitelist |

### Modifiers with `--gpu-bandwidth` (standalone mode)

GPU schema 1 has an exact whitelist. Short and long aliases are equivalent, and duplicate occurrences are rejected.

| Modifier | Compatible | Notes |
|----------|------------|-------|
| `-b, --buffer-size <MB>` | ✅ | Default `512` MB per private buffer; hard minimum `64` MB; requested size is never silently reduced |
| `-i, --iterations <n>` | ✅ | Exact full-buffer pass/dispatch count. Omission calibrates each operation toward 150 ms; explicit values must fit 16,384 dispatches and 64 GiB exact payload, with copy 2× defining the strict shared ceiling |
| `-r, --count <n>` | ✅ | GPU-local default `3`; order rotates read/write/copy and balances only in complete multiples of three |
| `--seed <uint64>` | ✅ | Exact base seed; generated once when omitted; domain-separated operation seeds are recorded |
| `-o, --output <target>` | ✅ | Exact `-` emits one final schema 1 document on stdout; an empty value disables direct JSON; `./-`, `-G`, and every other non-empty value are files with the existing atomic terminal-measurement and failure checkpoints |
| `-h, --help` | ✅ | Prints GPU-mode help and exits without Metal work |
| `--sweep`, `--sweep-max-runs` | ❌ | No GPU sweep support in schema 1 |
| `--threads`, cache/latency/pattern/TLB/core-to-core modifiers | ❌ | Outside the standalone whitelist |
| Any other primary mode | ❌ | Primary modes are mutually exclusive |

GPU config validation, including the 64 MB minimum and strict number parsing, happens before Metal initialization and
does not write result JSON. After valid parsing, initialized backend/capability and compilation/allocation/work-plan
failures are status-bearing GPU schema 1 results when a non-empty `--output` target enables JSON; a backend-factory
failure before result initialization leaves stdout empty. For exact `-`, intermediate checkpoints remain logical lazy
transitions and the terminal payload retains raw `configuration.output_file: "-"`; a real file retains the atomic
checkpoint cadence.
Grid geometry is not a CLI parameter: schema 1 uses the frozen 8192-threadgroup maximum and records both that maximum and
the resolved grid in each work plan.

### Modifiers with `--llm-memory` (standalone mode)

The LLM parser has an exact whitelist. The five common model options and phase-specific geometry must occur once; every
optional value and the mode or help selector may also occur at most once. Backend defaults to CPU; Metal is selected
with `--llm-memory-backend metal`. Both backends support decode and prefill with either KV layout, for eight active
backend/phase/layout profiles. The four Metal profiles are an experimental preview. Phase defaults to decode.
Decode requires exactly one `--context-tokens`; prefill requires exactly one `--prompt-tokens P` and
`--attention-query-tile-tokens Q`, with `P >= 1` and `1 <= Q <= P`. Cross-phase geometry is rejected. `--kv-layout`
defaults to `contiguous`. Paged layout requires exactly one
`--kv-block-tokens <G>`; contiguous layout rejects that option. `G` must be a positive power of two no greater than
`UINT32_MAX`, and `G` may exceed the active phase length. These rules are order-independent. After checked configuration
and memory-budget preflight, the command allocates the layout-specific resources and runs the three schema-1 scenarios.

| Modifier | Compatible | Notes |
|----------|------------|-------|
| `--llm-memory-backend <cpu\|metal>` | ✅ | Default `cpu`. Both backends support decode/prefill with contiguous or paged KV. The experimental Metal preview performs no CPU fallback, reports capability absence as terminal `unsupported`, and reports runtime compiler/pipeline/resource/task failure as terminal `failed`/`invalid` evidence |
| `--weight-size-mb <MiB>` | ✅ required | Positive active weight size; checked MiB-to-byte conversion |
| `--layers <n>` | ✅ required | Positive layer count |
| `--query-heads <n>` | ✅ required | Positive; must be at least and evenly divisible by KV heads |
| `--kv-heads <n>` | ✅ required | Positive physical KV-head count |
| `--head-dim <n>` | ✅ required | Positive K/V head-vector element count |
| `--phase <decode\|prefill>` | ✅ | Default `decode`; selects a versioned work-unit contract |
| `--context-tokens <n>` | ✅ decode only | Positive fixed visible context including the current synthetic token |
| `--prompt-tokens <P>` | ✅ prefill only | Positive full-prompt length; no default |
| `--attention-query-tile-tokens <Q>` | ✅ prefill only | Required with prefill; `1 <= Q <= P`, with no default |
| `--kv-element-bytes <1\|2\|4>` | ✅ | Default `2`; every other width is rejected |
| `--batch-size <n>` | ✅ | Positive; default `1` |
| `--kv-layout <contiguous\|paged>` | ✅ | Default `contiguous`; both layouts are executable for decode and prefill on CPU and Metal |
| `--kv-block-tokens <G>` | ✅ paged only | Required exactly once with paged; rejected with contiguous. Positive power of two, at most `UINT32_MAX`; may exceed the active phase length |
| `-t, --threads <n>` | ✅ CPU only | Positive requested workers; omission uses detected workers. Metal rejects the option and does not perform worker detection; worker/QoS evidence is null with applicability false |
| `-i, --iterations <n>` | ✅ | Positive exact work units per scenario; omission selects excluded per-scenario calibration toward 150 ms. CPU values fit the common work/task guardrails; Metal additionally caps one dispatch at 65,536 work units. Metal prefill caps lane-local serial range-helper visits at 1,048,576 per task, including `T*L` for paged `weights_only`; paged Metal profiles also enforce semantic-lookup, owner-ordinal, threadgroup, and per-visit vector-iteration caps |
| `-r, --count <n>` | ✅ | Positive cyclic loop count; default `3` |
| `--seed <uint64>` | ✅ | Exact base seed including zero; a non-zero seed is generated once when omitted |
| `-o, --output <target>` | ✅ | Empty disables JSON; exact `-` emits one final schema 1 document; `./-`, flag-shaped values, and every other non-empty non-sentinel value are atomic file targets. A non-empty target adds the conservative JSON output peak to memory admission |
| `-h, --help` | ✅ | Prints dedicated help before enforcing required inputs or resolving worker/seed/session/QoS state; malformed or duplicate tokens still fail |
| `--sweep`, `--sweep-max-runs`, `--non-cacheable` | ❌ | Outside the frozen v1 whitelist |
| Buffer/cache/latency/TLB/pattern/core-to-core/GPU modifiers | ❌ | Outside the standalone whitelist |
| Any other primary mode | ❌ | Primary modes are mutually exclusive |

The parser rejects checked weight/KV geometry that overflows or makes even one scenario work unit exceed the 64 GiB
task-accounted-byte ceiling. For paged KV, task-accounted bytes include logical model bytes plus timed block-table lookup
traffic, while throughput remains model bytes divided by timed seconds. Before allocation, the planner separately admits
page-rounded full-size weight and physical K/V mappings, the block table, descriptors, retained planner/transient storage,
checksum storage, and orchestration storage against the current memory budget. All four CPU and all four experimental
Metal phase/layout profiles are active. Metal uses Tier 2 argument-buffer indirection, GPU command-buffer timestamps,
dual-mod32 checksum validation, and excluded phase-neutral `kv_write` validation. Contiguous uses exact-tail W/K/V
segments. Metal prefill allows at most 1,048,576 serial range-helper visits per lane and rejects a larger or overflowing
task before checksum-oracle work or GPU dispatch. Contiguous tasks count all applicable serial ranges; paged
`weights_only` counts `T*L`, while paged KV-bearing tasks use their separately capped owner schedule. Contiguous prefill
otherwise logically visits
all prompt K/V records before its tiled-prefix reads; a weight-bearing scenario performs one weight pass per operation.
Both paged Metal phases use whole-block K/V segmentation, segmented private table storage, cyclic one-threadgroup
ownership, and terminal-block padding canaries. Decode records exact `L*B*(2*N+1)` lookup evidence. Prefill records
exact `L*B*(N+2*M)` lookup evidence, where `M` sums the blocks reached by every query-tile prefix, and reports exact
per-threadgroup `actual-threadgroup-cost` accounted-byte vector, minimum, maximum, and imbalance. Every Metal
`weights_only` task reports the same cost unit for its weight-vector grid-stride schedule; contiguous KV-bearing grids
publish no threadgroup-cost evidence. Full-prompt write samples and applicable padding canaries must validate after each
KV-active task. This is a cyclic assignment, not a weighted balance. The command does not fall back to another backend,
phase, or layout.

### Sweep Compatibility

`--sweep` runs a Cartesian product over one or more parameter lists. It always requires a non-empty
`--output <target>`; an empty value is missing/invalid. Exact `-` emits one final combined JSON document to stdout.
Every other non-empty target is an atomically checkpointed file, including `./-` and flag-shaped names such as `-G`.
The envelope uses `configuration.mode: "sweep"`, with per-run payloads under `runs[].result`. General and core-to-core
envelopes use `configuration.sweep_schema_version: 1`; each nested result keeps its own mode schema version.

| Base mode | Supported sweep keys | Not supported |
|-----------|----------------------|---------------|
| `--benchmark` | `buffer-size`, `cache-size`, `threads`, `latency-tlb-locality-kb`, `latency-stride-bytes`, `latency-chain-mode` | `tlb-density` |
| `--benchmark --only-bandwidth` | `buffer-size`, `threads` | `cache-size`, latency keys, `tlb-density` |
| `--benchmark --only-latency` | `buffer-size`, `cache-size`, `latency-tlb-locality-kb`, `latency-stride-bytes`, `latency-chain-mode` | `threads`, `tlb-density` |
| `--patterns` | `buffer-size`, `threads` | `cache-size`, latency keys, `tlb-density` |
| `--analyze-tlb` | `latency-stride-bytes`, `latency-chain-mode`, `tlb-density` | `buffer-size`, `cache-size`, `threads`, `latency-tlb-locality-kb` |
| `--analyze-core2core` | `count`, `latency-samples` | `buffer-size`, `cache-size`, `threads`, latency chain/locality/stride keys, `tlb-density` |
| `--gpu-bandwidth` | none | GPU schema 1 rejects all sweep keys and `--sweep-max-runs` |
| `--llm-memory` | none | The standalone LLM whitelist rejects all sweep keys |

Additional sweep rules:

- `--sweep-max-runs <n>` limits the generated Cartesian product; default is `16` with `--analyze-tlb` and `256` otherwise.
- In general CPU and core-to-core modes, `--sweep-max-runs` is accepted without `--sweep` but has no effect in that
  case. GPU and LLM standalone parsers reject it.
- A sweep parameter key may appear only once; duplicate keys are rejected before execution.
- `--sweep latency-chain-mode=global-random` is invalid with `--analyze-tlb`.
- Direct options outside `--sweep` are used as fixed values for every generated run.
- If the same parameter is provided both directly and through `--sweep`, the sweep value is applied per run.
- A real-file sweep is atomically checkpointed after every attempted run without an additional terminal rewrite. An
  empty run plan or a stop observed before a run also checkpoints a terminal envelope without adding a `runs[]` entry or
  incrementing `attempted_runs`. A stdout sweep still performs every corresponding logical checkpoint transition without
  invoking the persistence payload builder or serializing an intermediate document; one final envelope is emitted after
  orchestration. `attempted_runs` equals stored `runs` entries; partial, interrupted, and failed attempts remain in that
  array but stop further execution and do not increment `completed_runs`. A current standard schema-3 attempt is
  complete only with nested `configuration.mode: "benchmark"`, `status: "complete"`, `results_complete: true`,
  `conclusions_valid: true`, and a string `configuration.output_file`. Nested standard schema 2 and every other
  standard version are unsupported. A pattern attempt requires nested
  `status: "complete"` and `results_complete: true`; TLB requires nested `tlb_analysis.status: "complete"` and
  `tlb_analysis.conclusions_valid: true`; core-to-core requires nested `core_to_core_latency.status: "complete"` and
  `measurements_complete: true`. The authoritative schema-1 sweep acceptance predicate is exactly
  `status == "complete" && conclusions_valid == true`. Producers maintain `completed_runs == planned_runs` for an
  envelope satisfying that predicate; consumers may check the equality separately as a defensive consistency check,
  but it is not another completeness condition. A nested TLB `status: "error"` is retained in `runs[].result` and maps
  the attempt to failed without adding a TLB `status_reason` field.

### Compatibility and Incompatibility Cases

| Pair | Reason |
|------|--------|
| `--only-bandwidth` + `--only-latency` | Mutually exclusive |
| `--only-bandwidth` + `--cache-size` | cache-size only for latency |
| `--only-bandwidth` + `--latency-samples` | latency-samples only for latency |
| `--only-latency` + `--iterations` | iterations only for bandwidth |
| `--only-latency --buffer-size 0 --cache-size 0` | At least one latency target must remain enabled |
| `--only-bandwidth` + `--patterns` | Separate modes |
| `--only-latency` + `--patterns` | Separate modes |
| `--benchmark` + `--patterns` | Mutually exclusive |
| `--tlb-density` without `--analyze-tlb` | TLB density is parsed only by the standalone TLB mode |
| `--analyze-tlb` + `--help` | The standalone TLB whitelist does not include help; use `--help` without `--analyze-tlb` |
| `--gpu-bandwidth` + any other primary mode | GPU is a standalone primary mode |
| `--gpu-bandwidth` + any option outside `buffer-size`, `iterations`, `count`, `seed`, `output`, `help` | GPU schema 1 exact whitelist |
| `--llm-memory` + any other primary mode | LLM-memory is a standalone primary mode |
| `--llm-memory` + any option outside `llm-memory-backend`, its model/phase geometry, `kv-element-bytes`, `batch-size`, `kv-layout`, `kv-block-tokens`, `threads`, `iterations`, `count`, `seed`, `output`, `help` | LLM exact whitelist |
| `--llm-memory --kv-layout paged` without exactly one valid `--kv-block-tokens` | Paged KV requires an explicit positive power-of-two block size no greater than `UINT32_MAX` |
| `--llm-memory --kv-layout contiguous --kv-block-tokens <G>` | Block size has no meaning for contiguous KV and is rejected |
| `--llm-memory --phase decode` with prefill geometry, or prefill with `--context-tokens` | Phase-specific geometry is not interchangeable |
| `--llm-memory --llm-memory-backend metal --phase prefill` | Valid experimental Metal contiguous-prefill profile when paged options are absent; requires normal `P`/`Q` geometry |
| `--llm-memory --llm-memory-backend metal --phase prefill --kv-layout paged --kv-block-tokens <G>` | Valid experimental Metal paged-prefill profile; requires normal `P`/`Q` geometry and explicit valid `G` |
| `--llm-memory --llm-memory-backend metal --threads <n>` | Metal has no CPU-worker contract and rejects explicit threads |
| `--llm-memory --phase prefill --kv-layout paged --kv-block-tokens <G>` | Valid CPU paged-prefill profile; requires the normal prefill `P`/`Q` geometry and explicit valid `G` |
| `--llm-memory --llm-memory-backend metal --phase decode --kv-layout paged --kv-block-tokens <G>` | Valid experimental Metal paged-decode profile; requires normal decode context and explicit valid `G` |
| `--sweep` without `--output`, or with an empty output value | Sweep mode requires a non-empty combined JSON output target |
| `--sweep` generated runs > `--sweep-max-runs` | Guardrail against accidental large Cartesian sweeps |

### No Mode Flag (shows help)

Running with syntactically valid general modifiers but no primary mode flag (`--benchmark`, `--patterns`,
`--analyze-tlb`, `--analyze-core2core`, `--gpu-bandwidth`, or `--llm-memory`) shows help and exits without semantic validation. Parser
errors still fail before this fallback: for example, missing/malformed values and unknown options are errors, and
`--tlb-density` is unknown unless `--analyze-tlb` selects the standalone TLB parser.

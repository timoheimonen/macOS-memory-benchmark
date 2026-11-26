# Parameter Compatibility Matrix

Applies to the current repository implementation.

Runtime platform: macOS 26 or later on Apple Silicon (ARM64).

## All Flags

| Short alias | Long option | Value | Description |
|-------------|-------------|-------|-------------|
| `-B` | `--benchmark` | — | Run the standard CPU bandwidth/latency benchmark |
| `-P` | `--patterns` | — | Run the CPU access-pattern benchmark |
| `-T` | `--analyze-tlb` | — | Run standalone TLB analysis |
| `-C` | `--analyze-core2core` | — | Run standalone two-thread acquire/release token-protocol handoff analysis |
| `-G` | `--gpu-bandwidth` | — | Run standalone Metal GPU memory bandwidth |
| `-M` | `--llm-memory` | — | Run the standalone CPU/Metal synthetic LLM memory profile |
| — | `--llm-memory-backend` | `cpu\|metal` | LLM execution backend; default `cpu`; CPU and capability-gated Metal support decode/prefill with contiguous or paged KV |
| — | `--weight-size-mb` | `<MiB>` | Required positive active weight size for LLM-memory mode |
| — | `--layers` | `<count>` | Required positive LLM layer count |
| — | `--query-heads` | `<count>` | Required positive query-head count; at least the KV-head count and divisible by it; defines theoretical attention context |
| — | `--kv-heads` | `<count>` | Required positive physical KV-head count |
| — | `--head-dim` | `<count>` | Required positive K/V head-vector element count |
| — | `--kv-element-bytes` | `1\|2\|4` | LLM KV element width; default `2` |
| — | `--phase` | `decode\|prefill` | LLM phase; default `decode` |
| — | `--context-tokens` | `<count>` | Required only for decode; fixed visible context including the current synthetic token |
| — | `--prompt-tokens` | `<P>` | Required only for prefill; positive full-prompt token count |
| — | `--attention-query-tile-tokens` | `<Q>` | Required only for prefill; query-tile size in `1..P` controls synthetic prefix rereads |
| — | `--batch-size` | `<count>` | Positive LLM batch-sequence count; default `1` |
| — | `--kv-layout` | `contiguous\|paged` | LLM KV layout; default `contiguous` |
| — | `--kv-block-tokens` | `<G>` | Required only for paged KV: a positive power of two no greater than `UINT32_MAX`; may exceed the active phase length |
| `-i` | `--iterations` | `<count>` | Positive exact R/W/Copy pass count or LLM scenario work-unit count; CPU `--benchmark`/`--patterns` maximum is `INT_MAX`, while GPU and LLM apply work-dependent ceilings. Omission enables automatic calibration in the applicable mode |
| `-b` | `--buffer-size` | `<MB>` | Default `512` MB. Standard mode permits `0` only with `--only-latency`; pattern mode requires a positive value; GPU minimum is `64` MB |
| `-r` | `--count` | `<count>` | Positive loop count; default `1` for benchmark/pattern modes and `3` for core-to-core/GPU/LLM modes. Maximum is `INT_MAX` for general/core-to-core/GPU modes and `floor((2^53 - 1) / 3)` (`3,002,399,751,580,330`) for `--llm-memory` |
| — | `--seed` | `<uint64>` | Unsigned 64-bit reproducibility seed for benchmark, pattern, TLB, GPU, or LLM mode; generated once when omitted |
| `-n` | `--latency-samples` | `<count>` | Positive sample-window count up to `INT_MAX`; default `1000` in benchmark and core-to-core modes |
| `-s` | `--latency-stride-bytes` | `<bytes>` | Positive, pointer-aligned latency-chain stride; default `256` bytes |
| `-m` | `--latency-chain-mode` | `<mode>` | Chain policy: `auto` (default), `global-random`, `random-box`, `same-random-in-box`, or `diff-random-in-box` |
| `-l` | `--latency-tlb-locality-kb` | `<KB>` | Latency-chain locality window; default `1024` KB. With `auto`, `0` selects global random |
| `-D` | `--tlb-density` | `low\|medium\|high` | Standalone TLB runtime profile; default `medium` |
| `-t` | `--threads` | `<count>` | Positive requested bandwidth worker count. General main-memory/pattern execution accepts `1..INT_MAX` and caps above detected cores; CPU LLM accepts `1..(2^53 - 1)` and preserves requested, detected, and executable effective counts separately, while Metal LLM rejects this option |
| `-k` | `--cache-size` | `<KB>` | Custom cache target: `16..1048576` KB, or `0` only with `--benchmark --only-latency` |
| `-W` | `--only-bandwidth` | — | Run only standard benchmark bandwidth tests; requires `--benchmark` |
| `-L` | `--only-latency` | — | Run only standard benchmark latency tests; requires `--benchmark` |
| `-u` | `--non-cacheable` | — | Apply best-effort cache-discouraging allocation hints; does not create truly uncached memory |
| `-o` | `--output` | `<target>` | Result target syntax. Exact `-` selects one final stdout document for result-producing modes. An empty direct value disables JSON; every other non-empty value is a file. Direct LLM file output snapshots every `K=max(1,ceil(count/8))` completed loops and at command terminal; sweep files (including LLM sweeps) checkpoint after each attempted run without intra-run snapshots |
| `-S` | `--sweep` | `<key=a,b>` | Add a Cartesian sweep parameter; repeat once per distinct key and use with `--output` |
| `-X` | `--sweep-max-runs` | `<count>` | Positive generated-run limit; default `256`, or `16` with `--analyze-tlb`; effective only with `--sweep`. Upper bound is `LLONG_MAX` for general/TLB, `INT_MAX` for core-to-core, and `2^53 - 1` (`9,007,199,254,740,991`) when active with `--llm-memory --sweep` (any positive `size_t` is accepted without effect on direct `--llm-memory`); rejected by `--gpu-bandwidth` |
| `-h` | `--help` | — | Show help across all six primary modes (including `--analyze-tlb`) after syntactic, whitelist, duplicate, and value-parse validation |

Short and long forms are equivalent. The compatibility tables below use long forms as canonical names; the GPU and LLM
tables also repeat their exact whitelist aliases. LLM model-geometry options and `--seed` have no short aliases. Long
options require two dashes, short options are exactly one character, and short options cannot be bundled. Value-taking
long options accept both space-separated (`--buffer-size 512`, `--output -`) and inline (`--buffer-size=512`,
`--output=-`, `--output=`, `--sweep=buffer-size=128,256`) syntax across all six modes; passing an inline value to a
boolean or mode flag (such as `--benchmark=1` or `--help=1`) is rejected (`Option --... does not take a value`), and
legacy single-dash long names (`-buffersize`) or single-dash inline forms (`-b=512`) remain rejected. Every
single-occurrence option — including primary mode selectors (`-B`, `-P`, `-T`, `-C`, `-G`, `-M`), boolean flags
(`-W`, `-L`, `-u`, `-h`), and single-value options — may appear at most once across all six modes; only `-S`/`--sweep`
may be repeated for distinct parameter keys. All six mode parsers emit concise single-line `Error: ...` diagnostics on
stderr without dumping full usage text, and standalone whitelist modes explicitly name the offending option
(`Unsupported option '<token>': ...`). Numeric values must be complete decimal tokens without whitespace, a leading `+`, or trailing
characters; unsigned seeds and `--llm-memory` numeric options reject either sign. Serialized `--llm-memory` integer
configuration fields (and `--sweep-max-runs` when `--sweep` is active) must not exceed the IEEE-754 exact safe integer
bound `2^53 - 1` (`9,007,199,254,740,991`; `reason_code=json-integer-out-of-range`), and `--count` is further bounded
by `floor((2^53 - 1) / 3)` (`3,002,399,751,580,330`).

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
| `--latency-stride-bytes <n>` | ✅ | Positive and pointer-aligned; each enabled target/configured locality window needs two nodes. The optional fixed 16 KiB comparison is unavailable above an 8192 B stride. Rejected with `--only-bandwidth` |
| `--latency-chain-mode <mode>` | ✅ | Box modes require `--latency-tlb-locality-kb > 0`; `global-random` works with locality `0`. Rejected with `--only-bandwidth` |
| `--latency-tlb-locality-kb <n>` | ✅ | Locality-using modes require a non-zero page-size multiple; explicit `global-random` ignores the value. Any explicit value disables the automatic 16 KiB vs global-random comparison. Rejected with `--only-bandwidth` |
| `--threads <n>` | ✅ | Omitted main-memory bandwidth uses detected cores and omitted cache bandwidth uses one worker; an explicit value applies to both (capped in validation if above available cores). Rejected with `--only-latency` (latency tests are single-threaded) |
| `--cache-size <KB>` | ✅ | Replaces auto L1/L2 cache tests with one custom cache target; rejected with `--only-bandwidth` |
| `--only-bandwidth` | ✅ | ❌ with `--cache-size`, `--latency-samples`, `--latency-stride-bytes`, `--latency-chain-mode`, `--latency-tlb-locality-kb` |
| `--only-latency` | ✅ | ❌ with `--iterations`, ❌ with `--threads`. At least one latency target must remain enabled; `--buffer-size 0 --cache-size 0` is invalid |
| `--non-cacheable` | ✅ | |
| `--output <target>` | ✅ | Exact `-` selects final JSON stdout; an empty direct value disables JSON and an empty sweep value is invalid; every other non-empty value is a file, including `./-` and `-G` |
| `--sweep <key=a,b>` | ✅ | Requires `--output`; supported keys depend on benchmark subtype, see [Sweep Compatibility](#sweep-compatibility); cannot be combined with the matching direct option |
| `--sweep-max-runs <n>` | ✅ | Default `256`; accepted without `--sweep` but has no effect then |
| `--tlb-density <low\|medium\|high>` | ❌ | Parsed only by standalone `--analyze-tlb` |
| `--help` | ✅ | Prints general help to stdout and exits `0` after syntactic/value checks, before semantic validation, OS queries, timer creation, seed generation, or output-session setup (including with `--output -`) |
| `--buffer-size 0` | ✅ only with `--only-latency` | Disables main memory latency |
| `--cache-size 0` | ✅ only with `--only-latency` | Disables cache latency |

### Modifiers with `--patterns`

| Modifier | Compatible | Notes |
|----------|------------|-------|
| `--iterations <n>` | ✅ | Positive exact pass override; omission enables automatic duration calibration |
| `--buffer-size <MB>` | ✅ | Default `512`; must be positive because pattern mode cannot use the latency-only disabling case. Oversized requests can be reduced to the memory-safety cap with a warning |
| `--count <n>` | ✅ | Positive integer; default `1` |
| `--seed <uint64>` | ✅ | Reproduces random workload; generated once when omitted |
| `--latency-samples <n>` | ❌ | Rejected; pattern benchmarks do not run cache or latency tests |
| `--latency-stride-bytes <n>` | ❌ | Rejected; pattern benchmarks do not run cache or latency tests |
| `--latency-chain-mode <mode>` | ❌ | Rejected; pattern benchmarks do not run cache or latency tests |
| `--latency-tlb-locality-kb <n>` | ❌ | Rejected; pattern benchmarks do not run cache or latency tests |
| `--threads <n>` | ✅ | Omission uses the detected CPU core count; sparse-stride work may reduce the effective worker count |
| `--cache-size <KB>` | ❌ | Rejected; pattern benchmarks do not run cache or latency tests |
| `--only-bandwidth` | ❌ | Separate execution mode |
| `--only-latency` | ❌ | Separate execution mode |
| `--non-cacheable` | ✅ | |
| `--output <target>` | ✅ | Exact `-` selects final JSON stdout; an empty direct value disables JSON and an empty sweep value is invalid; every other non-empty value is a file, including `./-` and `-G` |
| `--sweep <key=a,b>` | ✅ | Requires `--output`; supported keys: `buffer-size`, `threads`; cannot be combined with the matching direct option |
| `--sweep-max-runs <n>` | ✅ | Default `256`; accepted without `--sweep` but has no effect then |
| `--tlb-density <low\|medium\|high>` | ❌ | Parsed only by standalone `--analyze-tlb` |
| `--help` | ✅ | Prints general help to stdout and exits `0` after syntactic/value checks, before semantic validation, OS queries, timer creation, seed generation, or output-session setup (including with `--output -`) |

### Modifiers with `--analyze-tlb` (standalone mode)

| Modifier | Compatible | Notes |
|----------|------------|-------|
| `--output <target>` | ✅ | Exact `-` selects final JSON stdout; an empty direct value disables JSON and an empty sweep value is invalid; every other non-empty value is a file, including `./-` and `-G` |
| `--latency-stride-bytes <n>` | ✅ | Must be positive, pointer-aligned (8 bytes on Apple Silicon), and no larger than the system page size; exact page-size divisibility is not required |
| `--latency-chain-mode <mode>` | ✅ | `global-random` is rejected with `--analyze-tlb` |
| `--tlb-density <low\|medium\|high>` | ✅ | Default `medium`/standard; low=quick, high=exhaustive |
| `--seed <uint64>` | ✅ | Fixed reproducibility seed; generated once when omitted |
| `--sweep <key=a,b>` | ✅ | Requires `--output`; supported keys: `latency-stride-bytes`, `latency-chain-mode`, `tlb-density`; cannot be combined with the matching direct option |
| `--sweep-max-runs <n>` | ✅ | Default `16`; accepted without `--sweep` but has no effect then |
| `--help` | ✅ | Prints general help to stdout and exits `0` after syntactic/whitelist/duplicate/value checks, before semantic validation, OS queries, timer creation, seed generation, or output-session setup (including with `--output -`) |
| All others | ❌ | Rejected by the standalone whitelist (`Unsupported option '<token>': ...`) |

### Modifiers with `--analyze-core2core` (standalone mode)

| Modifier | Compatible | Notes |
|----------|------------|-------|
| `--output <target>` | ✅ | Exact `-` selects final JSON stdout; an empty direct value disables JSON and an empty sweep value is invalid; every other non-empty value is a file, including `./-` and `-G` |
| `--count <n>` | ✅ | Core-to-core default `3` (general default remains `1`); scenario order rotates and the headline is the loop median P50 |
| `--latency-samples <n>` | ✅ | Positive integer; default `1000`. Separate calibrated sample windows per scenario/loop do not define the continuous headline |
| `--sweep <key=a,b>` | ✅ | Requires `--output`; supported keys: `count`, `latency-samples`; cannot be combined with the matching direct option |
| `--sweep-max-runs <n>` | ✅ | Default `256`; accepted without `--sweep` but has no effect then |
| `--help` | ✅ | Prints general help to stdout and exits `0` after syntactic/whitelist/duplicate/value checks, without running core-to-core analysis or opening output sessions (including with `--output -`) |
| All others | ❌ | Rejected by the standalone whitelist (`Unsupported option '<token>': ...`) |

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

The LLM parser has an exact whitelist. The five common model options must occur once; every optional single-value flag
and the mode or help selector may occur at most once, while `-S`/`--sweep` may be repeated once per distinct sweep key.
Backend defaults to CPU; Metal is selected with `--llm-memory-backend metal`. Both backends support decode and prefill
with either KV layout, for eight active backend/phase/layout profiles. The four Metal profiles are capability-gated.
Phase defaults to decode. Decode requires `--context-tokens` (unless supplied via `--sweep context-tokens=...`);
prefill requires `--prompt-tokens P` (unless supplied via `--sweep prompt-tokens=...`) and
`--attention-query-tile-tokens Q` (unless supplied via `--sweep attention-query-tile-tokens=...`), with `P >= 1` and
`1 <= Q <= P`. Cross-phase geometry (direct or swept) is rejected. `--kv-layout` defaults to `contiguous`. Paged layout
requires `--kv-block-tokens <G>` (unless supplied via `--sweep kv-block-tokens=...`); contiguous layout rejects both.
`G` must be a positive power of two no greater than `UINT32_MAX`, and `G` may exceed the active phase length. Combining
a direct flag and `--sweep` for the same parameter is rejected. These rules are order-independent. Every serialized
integer configuration field (and `--sweep-max-runs` when `--sweep` is active) must not exceed `2^53 - 1`
(`9,007,199,254,740,991`; `json-integer-out-of-range`), and `--count` is further bounded by
`floor((2^53 - 1) / 3)` (`3,002,399,751,580,330`). After checked configuration preflight (across every generated
combination in a sweep) and memory-budget admission, the command allocates the layout-specific resources and runs the
three scenarios. Direct LLM output uses schema 2 and `llm-memory-v2-<backend>-<phase>-<layout>` methodology identifiers;
`--llm-memory --sweep` emits the shared schema-1 sweep envelope (`base_mode: "llm_memory"`) embedding each run's
schema-2 result under `runs[].result`.

| Modifier | Compatible | Notes |
|----------|------------|-------|
| `--llm-memory-backend <cpu\|metal>` | ✅ | Default `cpu`. Both backends support decode/prefill with contiguous or paged KV. Metal performs no CPU fallback, reports capability absence as terminal `unsupported`, and reports runtime compiler/pipeline/resource/task failure as terminal `failed`/`invalid` evidence |
| `--weight-size-mb <MiB>` | ✅ required | Positive active weight size (`<= 2^53 - 1`); checked MiB-to-byte conversion |
| `--layers <n>` | ✅ required | Positive layer count (`<= 2^53 - 1`) |
| `--query-heads <n>` | ✅ required | Positive (`<= 2^53 - 1`); must be at least and evenly divisible by KV heads. Defines theoretical attention context; no attention computation is executed |
| `--kv-heads <n>` | ✅ required | Positive physical KV-head count (`<= 2^53 - 1`) |
| `--head-dim <n>` | ✅ required | Positive K/V head-vector element count (`<= 2^53 - 1`) |
| `--phase <decode\|prefill>` | ✅ | Default `decode`; selects a versioned work-unit contract |
| `--context-tokens <n>` | ✅ decode only | Positive fixed visible context (`<= 2^53 - 1`) including the current synthetic token; required for decode unless supplied via `--sweep context-tokens=...`; cannot be combined with `--sweep context-tokens=...` |
| `--prompt-tokens <P>` | ✅ prefill only | Positive full-prompt length (`<= 2^53 - 1`); no default; required for prefill unless supplied via `--sweep prompt-tokens=...`; cannot be combined with `--sweep prompt-tokens=...` |
| `--attention-query-tile-tokens <Q>` | ✅ prefill only | Required with prefill unless supplied via `--sweep attention-query-tile-tokens=...`; `1 <= Q <= P` (`<= 2^53 - 1`), with no default; cannot be combined with `--sweep attention-query-tile-tokens=...`. Controls executed synthetic prefix rereads; it does not specify inference-kernel tiling |
| `--kv-element-bytes <1\|2\|4>` | ✅ | Default `2`; every other width is rejected |
| `--batch-size <n>` | ✅ | Positive (`<= 2^53 - 1`); default `1` |
| `--kv-layout <contiguous\|paged>` | ✅ | Default `contiguous`; both layouts are executable for decode and prefill on CPU and Metal |
| `--kv-block-tokens <G>` | ✅ paged only | Required with paged unless supplied via `--sweep kv-block-tokens=...`; cannot be combined with `--sweep kv-block-tokens=...`; contiguous rejects both. Positive power of two, at most `UINT32_MAX`; may exceed the active phase length |
| `-t, --threads <n>` | ✅ CPU only | Positive requested workers (`<= 2^53 - 1`); omission uses detected workers. Metal rejects the option and does not perform worker detection; worker/QoS evidence is null with applicability false |
| `-i, --iterations <n>` | ✅ | Positive exact work units per scenario; omission selects excluded per-scenario calibration toward 150 ms. CPU values fit the common `1,000,000,000`-work-unit and `64 GiB`-accounted-byte task guardrails; Metal additionally caps one dispatch at `65,536` work units. Metal prefill caps lane-local serial range-helper visits at `1,048,576` per task, including `T*L` for paged `weights_only`; paged Metal profiles also enforce semantic-lookup (`268,435,456`/task), owner-ordinal (`65,536`/threadgroup), threadgroup (`8,192`), and per-visit vector-iteration (`1,048,576`) caps |
| `-r, --count <n>` | ✅ | Positive cyclic loop count (`<= floor((2^53 - 1) / 3) = 3,002,399,751,580,330`); default `3` |
| `--seed <uint64>` | ✅ | Exact base seed including zero; a non-zero seed is generated once when omitted and shared across all runs in a sweep |
| `-o, --output <target>` | ✅ | Empty disables JSON for direct commands (invalid for `--sweep`); exact `-` emits one final schema 2 document (or one final schema 1 sweep envelope with `--sweep`); `./-`, flag-shaped values, and every other non-empty non-sentinel value are atomic file targets. A non-empty direct target or any sweep nested run adds the conservative JSON output peak to memory admission |
| `-S, --sweep <key=a,b>` | ✅ | Repeatable Cartesian sweep over distinct keys: `context-tokens` (decode), `prompt-tokens` and `attention-query-tile-tokens` (prefill), and `kv-block-tokens` (paged KV). Cannot be combined with the matching direct flag; requires non-empty `--output`; every generated combination is preflight-validated before execution |
| `-X, --sweep-max-runs <n>` | ✅ | Positive generated-run limit; default `256`. When `--sweep` is active, must also be `<= 2^53 - 1` (`json-integer-out-of-range`). Accepted without `--sweep` (any positive `size_t`, without enforcing `2^53 - 1`) with no effect |
| `-h, --help` | ✅ | Prints dedicated help before enforcing required inputs or resolving worker/seed/session/QoS state; malformed, unknown, or duplicate tokens still fail |
| `--non-cacheable` | ❌ | Outside the standalone whitelist |
| Buffer/cache/latency/TLB/pattern/core-to-core/GPU modifiers | ❌ | Outside the standalone whitelist |
| Any other primary mode | ❌ | Primary modes are mutually exclusive |

Direct LLM file output writes an atomic progress snapshot after every `K=max(1,ceil(count/8))` fully completed loops
(at most eight progress snapshots), plus a command-terminal snapshot on success, graceful interruption, or representable
failure. An abrupt exit can lose up to `3K` completed measurement attempts since the last successful snapshot. A failed
checkpoint stops the command without retry; a late command exception after successful terminal persistence may write
one corrective failure snapshot. `--llm-memory --sweep` file targets instead checkpoint the schema-1 sweep envelope
atomically after each attempted run and do not write intra-run loop snapshots. Stdout builds only the final document or
sweep envelope, and disabled direct output builds no JSON document.
See [API output checkpoints](API.md#checkpoints-and-final-snapshots) for the full persistence contract.

The parser rejects checked weight/KV geometry that overflows or makes even one scenario work unit exceed the 64 GiB
task-accounted-byte ceiling. For paged KV, task-accounted bytes include logical model bytes plus timed block-table lookup
traffic, while throughput remains model bytes divided by timed seconds. Before allocation, the planner separately admits
page-rounded full-size weight and physical K/V mappings, the block table, descriptors, retained planner/transient storage,
checksum storage, and orchestration storage against the current memory budget. All four CPU and all four
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

Prefill theoretical causal-pair, attention-pair, and FMA counts are descriptive model context. Overflow in those
calculations produces JSON `null` with reason `arithmetic-overflow` and does not by itself reject an otherwise valid
workload. Actual byte, prefix-read, lookup, and allocation arithmetic must still pass checked validation.

### Sweep Compatibility

`--sweep` runs a Cartesian product over one or more parameter lists. It always requires a non-empty
`--output <target>`; an empty value is missing/invalid. Exact `-` emits one final combined JSON document to stdout.
Every other non-empty target is an atomically checkpointed file, including `./-` and flag-shaped names such as `-G`.
The envelope uses `configuration.mode: "sweep"`, with per-run payloads under `runs[].result`. General, core-to-core,
and LLM-memory envelopes use `configuration.sweep_schema_version: 1`; each nested result keeps its own mode schema version.

| Base mode | Supported sweep keys | Not supported |
|-----------|----------------------|---------------|
| `--benchmark` | `buffer-size`, `cache-size`, `threads`, `latency-tlb-locality-kb`, `latency-stride-bytes`, `latency-chain-mode` | `tlb-density`, `count`, `latency-samples`, LLM sweep keys |
| `--benchmark --only-bandwidth` | `buffer-size`, `threads` | `cache-size`, latency keys, `tlb-density`, `count`, `latency-samples`, LLM sweep keys |
| `--benchmark --only-latency` | `buffer-size`, `cache-size`, `latency-tlb-locality-kb`, `latency-stride-bytes`, `latency-chain-mode` | `threads`, `tlb-density`, `count`, `latency-samples`, LLM sweep keys |
| `--patterns` | `buffer-size`, `threads` | `cache-size`, latency keys, `tlb-density`, `count`, `latency-samples`, LLM sweep keys |
| `--analyze-tlb` | `latency-stride-bytes`, `latency-chain-mode`, `tlb-density` | `buffer-size`, `cache-size`, `threads`, `latency-tlb-locality-kb`, `count`, `latency-samples`, LLM sweep keys |
| `--analyze-core2core` | `count`, `latency-samples` | `buffer-size`, `cache-size`, `threads`, latency chain/locality/stride keys, `tlb-density`, LLM sweep keys |
| `--gpu-bandwidth` | none | GPU schema 1 rejects all sweep keys and `--sweep-max-runs` |
| `--llm-memory` | `context-tokens` (decode), `prompt-tokens` and `attention-query-tile-tokens` (prefill), `kv-block-tokens` (paged KV); applicable keys combine in Cartesian products | All other sweep keys and non-applicable phase/layout keys; combining a direct flag and `--sweep` for the same parameter is rejected |

Additional sweep rules:

- `--sweep-max-runs <n>` limits the generated Cartesian product; default is `16` with `--analyze-tlb` and `256` otherwise.
  Upper bounds are `LLONG_MAX` (general CPU and `--analyze-tlb`), `INT_MAX` (`--analyze-core2core`), and `2^53 - 1`
  (`9,007,199,254,740,991`, `json-integer-out-of-range`) when `--sweep` is active in `--llm-memory`.
- In general CPU, core-to-core, and LLM modes, `--sweep-max-runs` is accepted without `--sweep` but has no effect in that
  case (in direct `--llm-memory` without `--sweep`, any positive `size_t` is accepted without enforcing `2^53 - 1`). The
  GPU standalone parser rejects it.
- A sweep parameter key may appear only once; duplicate keys are rejected before execution with the centralized `sweep parameter specified more than once: <key>` diagnostic across all sweep-capable modes.
- `--sweep latency-chain-mode=global-random` is invalid with `--analyze-tlb`.
- Direct options outside `--sweep` are used as fixed values for every generated run, and every generated combination is
  preflight-validated before the first run starts (including `1 <= Q <= P`, power-of-two `G <= UINT32_MAX`, `2^53 - 1`
  safe integer bounds, and scenario work/guardrail limits for `--llm-memory`). The resolved command seed is shared across
  all runs in the sweep.
- Across all sweep-capable modes (`--benchmark`, `--patterns`, `--analyze-tlb`, `--analyze-core2core`, and `--llm-memory`), providing both a direct flag and `--sweep` for the same parameter is rejected as a conflict (`Cannot specify both --<opt> and --sweep <key>`).
- A real-file sweep is atomically checkpointed after every attempted run without an additional terminal rewrite (and
  without intra-run loop snapshots). An
  empty run plan or a stop observed before a run also checkpoints a terminal envelope without adding a `runs[]` entry or
  incrementing `attempted_runs`. A stdout sweep still performs every corresponding logical checkpoint transition without
  invoking the persistence payload builder or serializing an intermediate document; one final envelope is emitted after
  orchestration. `attempted_runs` equals stored `runs` entries; partial, interrupted, and failed attempts remain in that
  array but stop further execution and do not increment `completed_runs`. A current standard schema-3 attempt is
  complete only with nested `configuration.mode: "benchmark"`, `status: "complete"`, `results_complete: true`,
  `conclusions_valid: true`, and a string `configuration.output_file`. Nested standard schema 2 and every other
  standard schema version are unsupported. A pattern attempt requires nested
  `status: "complete"` and `results_complete: true`; TLB requires nested `tlb_analysis.status: "complete"` and
  `tlb_analysis.conclusions_valid: true`; core-to-core requires nested `core_to_core_latency.status: "complete"` and
  `measurements_complete: true`; LLM-memory requires nested `mode: "llm_memory"`, `schema_version: 2`,
  `status: "complete"`, `results_complete: true`, `run_accepted: true`, and every planned `measurements[]` entry having
  `status: "measured"`. The authoritative schema-1 sweep acceptance predicate is exactly
  `status == "complete" && conclusions_valid == true`. Producers maintain `completed_runs == planned_runs` for an
  envelope satisfying that predicate; consumers may check the equality separately as a defensive consistency check,
  but it is not another completeness condition. A nested TLB `status: "error"` is retained in `runs[].result` and maps
  the attempt to failed without adding a TLB `status_reason` field. A nested LLM-memory `status: "unsupported"` maps to
  a failed sweep attempt while retaining the schema-2 payload; pre-initialization failures record `result: null` with
  the specific stable `status_reason`, and an interruption during pre-runner or backend resource preparation records
  `result: null`, `status: "interrupted"`, `status_reason: "preparation-interrupted"`, and exit status `0`.

### Compatibility and Incompatibility Cases

| Pair | Reason |
|------|--------|
| `--only-bandwidth` + `--only-latency` | Mutually exclusive |
| `--only-bandwidth` + `--cache-size` | cache-size only for latency |
| `--only-bandwidth` + `--latency-samples` | latency-samples only for latency |
| `--only-bandwidth` + `--latency-stride-bytes` / `--latency-chain-mode` / `--latency-tlb-locality-kb` | Latency pointer-chain options are only relevant for latency tests |
| `--only-latency` + `--iterations` | iterations only for bandwidth |
| `--only-latency` + `--threads` | Latency tests are single-threaded |
| `--only-latency --buffer-size 0 --cache-size 0` | At least one latency target must remain enabled |
| `--only-bandwidth` + `--patterns` | Separate modes |
| `--only-latency` + `--patterns` | Separate modes |
| `--patterns` + `--cache-size` / `--latency-samples` / `--latency-stride-bytes` / `--latency-chain-mode` / `--latency-tlb-locality-kb` | Pattern benchmarks do not run cache or latency tests |
| `--benchmark` + `--patterns` | Mutually exclusive |
| `--tlb-density` without `--analyze-tlb` | TLB density is parsed only by the standalone TLB mode |
| `--gpu-bandwidth` + any other primary mode | GPU is a standalone primary mode |
| `--gpu-bandwidth` + any option outside `buffer-size`, `iterations`, `count`, `seed`, `output`, `help` | GPU schema 1 exact whitelist |
| `--llm-memory` + any other primary mode | LLM-memory is a standalone primary mode |
| `--llm-memory` + any option outside `llm-memory-backend`, its model/phase geometry, `kv-element-bytes`, `batch-size`, `kv-layout`, `kv-block-tokens`, `threads`, `iterations`, `count`, `seed`, `output`, `sweep`, `sweep-max-runs`, `help` | LLM exact whitelist |
| `--llm-memory --kv-layout paged` without `--kv-block-tokens` or `--sweep kv-block-tokens=...` | Paged KV requires an explicit positive power-of-two block size no greater than `UINT32_MAX` |
| `--llm-memory --kv-layout contiguous` with `--kv-block-tokens <G>` or `--sweep kv-block-tokens=...` | Block size has no meaning for contiguous KV and is rejected |
| `--llm-memory --phase decode` with prefill geometry (direct or swept), or prefill with `--context-tokens` (direct or swept) | Phase-specific geometry is not interchangeable |
| Direct option + `--sweep` for the same parameter (all sweep-capable modes: `--benchmark`, `--patterns`, `--analyze-tlb`, `--analyze-core2core`, `--llm-memory`) | Conflicting direct and sweep specification (`Cannot specify both --<opt> and --sweep <key>`) |
| `--llm-memory --sweep` with `--sweep-max-runs > 2^53 - 1` (`9,007,199,254,740,991`) | Exceeds the exact IEEE-754 JSON safe integer range (`json-integer-out-of-range`) |
| `--llm-memory --llm-memory-backend metal --phase prefill` | Valid Metal contiguous-prefill profile when paged options are absent; requires normal `P`/`Q` geometry |
| `--llm-memory --llm-memory-backend metal --phase prefill --kv-layout paged --kv-block-tokens <G>` | Valid Metal paged-prefill profile; requires normal `P`/`Q` geometry and explicit valid `G` |
| `--llm-memory --llm-memory-backend metal --threads <n>` | Metal has no CPU-worker contract and rejects explicit threads |
| `--llm-memory --phase prefill --kv-layout paged --kv-block-tokens <G>` | Valid CPU paged-prefill profile; requires the normal prefill `P`/`Q` geometry and explicit valid `G` |
| `--llm-memory --llm-memory-backend metal --phase decode --kv-layout paged --kv-block-tokens <G>` | Valid Metal paged-decode profile; requires normal decode context and explicit valid `G` |
| `--sweep` without `--output`, or with an empty output value | Sweep mode requires a non-empty combined JSON output target |
| `--sweep` generated runs > `--sweep-max-runs` | Guardrail against accidental large Cartesian sweeps |
| Inline value on boolean/mode flag (e.g. `--benchmark=1`, `--help=1`) or single-dash inline value (`-b=512`) | Boolean/mode flags do not take a value (`Option --... does not take a value`); single-dash inline forms are rejected |
| Duplicated single-occurrence flag (including `-B`, `-P`, `-T`, `-C`, `-G`, `-M`, `-W`, `-L`, `-u`, `-h`) | Single-occurrence options may appear at most once across all six modes |
| Modifier option(s) without a primary mode flag (e.g. `--only-bandwidth`, `--seed 42`, `--threads 1`) | Primary mode required (bare `memory_benchmark` with no arguments prints help) |

### No Arguments vs. No Mode Flag

Running `memory_benchmark` with **no arguments** prints general help to stdout and exits `0`.

Running with one or more modifier options but **no primary mode flag** (`--benchmark`, `--patterns`,
`--analyze-tlb`, `--analyze-core2core`, `--gpu-bandwidth`, or `--llm-memory`) — and without `-h`/`--help` — fails
with `EXIT_FAILURE`, empty stdout, and a centralized error on stderr (`--only-bandwidth and --only-latency require
--benchmark flag`, `--seed requires --benchmark, --patterns, --analyze-tlb, --gpu-bandwidth, or --llm-memory`, or
`A primary mode option is required (--benchmark, --patterns, --analyze-tlb, --analyze-core2core, --gpu-bandwidth, or
--llm-memory)`). Syntactic parser errors (unknown options, duplicates, missing/unexpected values, malformed numbers)
still fail first during parsing: for example, `--tlb-density` is unknown unless `--analyze-tlb` selects the standalone
TLB parser.

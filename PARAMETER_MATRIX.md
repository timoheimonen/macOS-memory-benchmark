# Parameter Compatibility Matrix

Working version `0.61.1`

## All Flags

| Short alias | Long option | Value | Description |
|-------------|-------------|-------|-------------|
| `-B` | `--benchmark` | — | Run the standard CPU bandwidth/latency benchmark |
| `-P` | `--patterns` | — | Run the CPU access-pattern benchmark |
| `-T` | `--analyze-tlb` | — | Run standalone TLB analysis |
| `-C` | `--analyze-core2core` | — | Run standalone two-thread acquire/release token-protocol handoff analysis |
| `-G` | `--gpu-bandwidth` | — | Run standalone Metal GPU memory bandwidth |
| `-i` | `--iterations` | `<count>` | Positive exact R/W/Copy pass count; CPU maximum is `INT_MAX`, while GPU mode applies a smaller work-dependent ceiling. Omission enables automatic calibration in benchmark, pattern, and GPU modes |
| `-b` | `--buffer-size` | `<MB>` | Default `512` MB. Standard mode permits `0` only with `--only-latency`; pattern mode requires a positive value; GPU minimum is `64` MB |
| `-r` | `--count` | `<count>` | Positive loop count up to `INT_MAX`; default `1` for benchmark/pattern modes and `3` for core-to-core/GPU modes |
| — | `--seed` | `<uint64>` | Unsigned 64-bit reproducibility seed for benchmark, pattern, TLB, or GPU mode; generated once when omitted |
| `-n` | `--latency-samples` | `<count>` | Positive sample-window count up to `INT_MAX`; default `1000` in benchmark and core-to-core modes |
| `-s` | `--latency-stride-bytes` | `<bytes>` | Positive, pointer-aligned latency-chain stride; default `256` bytes |
| `-m` | `--latency-chain-mode` | `<mode>` | Chain policy: `auto` (default), `global-random`, `random-box`, `same-random-in-box`, or `diff-random-in-box` |
| `-l` | `--latency-tlb-locality-kb` | `<KB>` | Latency-chain locality window; default `1024` KB. With `auto`, `0` selects global random |
| `-D` | `--tlb-density` | `low\|medium\|high` | Standalone TLB runtime profile; default `medium` |
| `-t` | `--threads` | `<count>` | Positive requested bandwidth worker count up to `INT_MAX`; omitted main-memory/pattern work uses detected cores, omitted cache bandwidth uses one worker, and requests above available cores are capped |
| `-k` | `--cache-size` | `<KB>` | Custom cache target: `16..1048576` KB, or `0` only with `--benchmark --only-latency` |
| `-W` | `--only-bandwidth` | — | Run only standard benchmark bandwidth tests; requires `--benchmark` |
| `-L` | `--only-latency` | — | Run only standard benchmark latency tests; requires `--benchmark` |
| `-u` | `--non-cacheable` | — | Apply best-effort cache-discouraging allocation hints; does not create truly uncached memory |
| `-o` | `--output` | `<file>` | Write JSON output |
| `-S` | `--sweep` | `<key=a,b>` | Add a Cartesian sweep parameter; repeat once per distinct key and use with `--output` |
| `-X` | `--sweep-max-runs` | `<count>` | Positive generated-run limit; default `256`, or `16` with `--analyze-tlb`; effective only with `--sweep` |
| `-h` | `--help` | — | Show help; the standalone `--analyze-tlb` whitelist is the exception and rejects this combination |

Short and long forms are equivalent. The compatibility tables below use long forms as canonical names; the GPU table
also repeats its exact whitelist aliases. `--seed` is the only option without a short alias. Long options require two
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

| | `--benchmark` | `--patterns` | `--analyze-tlb` | `--analyze-core2core` | `--gpu-bandwidth` |
|---|---|---|---|---|---|
| `--benchmark` | ✅ | ❌ mutually exclusive | ❌ | ❌ | ❌ |
| `--patterns` | ❌ mutually exclusive | ✅ | ❌ | ❌ | ❌ |
| `--analyze-tlb` | ❌ | ❌ | ✅ | ❌ | ❌ |
| `--analyze-core2core` | ❌ | ❌ | ❌ | ✅ | ❌ |
| `--gpu-bandwidth` | ❌ | ❌ | ❌ | ❌ | ✅ |

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
| `--output <file>` | ✅ | |
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
| `--output <file>` | ✅ | |
| `--sweep <key=a,b>` | ✅ | Requires `--output`; supported keys: `buffer-size`, `threads` |
| `--sweep-max-runs <n>` | ✅ | Default `256`; accepted without `--sweep` but has no effect then |
| `--tlb-density <low\|medium\|high>` | ❌ | Parsed only by standalone `--analyze-tlb` |
| `--help` | ✅ | Prints general help and exits without running a benchmark |

### Modifiers with `--analyze-tlb` (standalone mode)

| Modifier | Compatible | Notes |
|----------|------------|-------|
| `--output <file>` | ✅ | |
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
| `--output <file>` | ✅ | |
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
| `-o, --output <file>` | ✅ | Atomically checkpoints GPU schema 1 after each terminal measurement and for valid post-parse pre-run failures |
| `-h, --help` | ✅ | Prints GPU-mode help and exits without Metal work |
| `--sweep`, `--sweep-max-runs` | ❌ | No GPU sweep support in schema 1 |
| `--threads`, cache/latency/pattern/TLB/core-to-core modifiers | ❌ | Outside the standalone whitelist |
| Any other primary mode | ❌ | Primary modes are mutually exclusive |

GPU config validation, including the 64 MB minimum and strict number parsing, happens before Metal initialization and
does not write result JSON. After valid parsing, unsupported device/capability and backend/allocation failures are
status-bearing GPU schema 1 checkpoints when `--output` is present. Grid geometry is not a CLI parameter: schema 1 uses
the frozen 8192-threadgroup maximum and records both that maximum and the resolved grid in each work plan.

### Sweep Compatibility

`--sweep` runs a Cartesian product over one or more parameter lists. It always requires `--output <file>` because sweep
results are written as one combined JSON document with `configuration.mode: "sweep"` and per-run payloads under
`runs[].result`.

| Base mode | Supported sweep keys | Not supported |
|-----------|----------------------|---------------|
| `--benchmark` | `buffer-size`, `cache-size`, `threads`, `latency-tlb-locality-kb`, `latency-stride-bytes`, `latency-chain-mode` | `tlb-density` |
| `--benchmark --only-bandwidth` | `buffer-size`, `threads` | `cache-size`, latency keys, `tlb-density` |
| `--benchmark --only-latency` | `buffer-size`, `cache-size`, `latency-tlb-locality-kb`, `latency-stride-bytes`, `latency-chain-mode` | `threads`, `tlb-density` |
| `--patterns` | `buffer-size`, `threads` | `cache-size`, latency keys, `tlb-density` |
| `--analyze-tlb` | `latency-stride-bytes`, `latency-chain-mode`, `tlb-density` | `buffer-size`, `cache-size`, `threads`, `latency-tlb-locality-kb` |
| `--analyze-core2core` | `count`, `latency-samples` | `buffer-size`, `cache-size`, `threads`, latency chain/locality/stride keys, `tlb-density` |
| `--gpu-bandwidth` | none | GPU schema 1 rejects all sweep keys and `--sweep-max-runs` |

Additional sweep rules:

- `--sweep-max-runs <n>` limits the generated Cartesian product; default is `16` with `--analyze-tlb` and `256` otherwise.
- Outside GPU mode, `--sweep-max-runs` is accepted without `--sweep` but has no effect in that case.
- A sweep parameter key may appear only once; duplicate keys are rejected before execution.
- `--sweep latency-chain-mode=global-random` is invalid with `--analyze-tlb`.
- Direct options outside `--sweep` are used as fixed values for every generated run.
- If the same parameter is provided both directly and through `--sweep`, the sweep value is applied per run.
- Combined sweep JSON is atomically checkpointed after every attempted run. `attempted_runs` equals stored `runs`
  entries; partial, interrupted, and failed attempts remain in that array but stop further execution and do not
  increment `completed_runs`. A standard or pattern attempt is complete only with nested `status: "complete"` and
  `results_complete: true`; TLB requires nested `tlb_analysis.status: "complete"` and
  `tlb_analysis.conclusions_valid: true`; core-to-core requires nested `core_to_core_latency.status: "complete"` and
  `measurements_complete: true`. Top-level `conclusions_valid` is true only when the sweep status is complete and
  `completed_runs == planned_runs`.

### Incompatible Modifier Combinations

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
| `--sweep` without `--output` | Sweep mode requires a combined JSON output file |
| `--sweep` generated runs > `--sweep-max-runs` | Guardrail against accidental large Cartesian sweeps |

### No Mode Flag (shows help)

Running with syntactically valid general modifiers but no primary mode flag (`--benchmark`, `--patterns`,
`--analyze-tlb`, `--analyze-core2core`, or `--gpu-bandwidth`) shows help and exits without semantic validation. Parser
errors still fail before this fallback: for example, missing/malformed values and unknown options are errors, and
`--tlb-density` is unknown unless `--analyze-tlb` selects the standalone TLB parser.

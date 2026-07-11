# Parameter Compatibility Matrix

Working version 0.60.0

## All Flags

| Flag | Description |
|------|-------------|
| `--benchmark` | Run standard benchmark |
| `--patterns` | Run pattern benchmark |
| `--iterations <n>` | Iterations for R/W/Copy tests |
| `--buffer-size <MB>` | Main memory buffer size |
| `--count <n>` | Number of benchmark loops |
| `--seed <uint64>` | Reproducible benchmark/pattern/TLB workload seed |
| `--latency-samples <n>` | Latency samples per test |
| `--latency-stride-bytes <n>` | Latency pointer chain stride |
| `--latency-chain-mode <mode>` | Chain construction policy |
| `--latency-tlb-locality-kb <n>` | TLB-locality window |
| `-D, --tlb-density <low\|medium\|high>` | Standalone TLB runtime profile |
| `--threads <n>` | Thread count |
| `--cache-size <KB>` | Custom cache size |
| `--only-bandwidth` | Bandwidth tests only |
| `--only-latency` | Latency tests only |
| `--non-cacheable` | Cache-discouraging hints |
| `--output <file>` | JSON output file |
| `--sweep <key=a,b>` | Cartesian parameter sweep |
| `--sweep-max-runs <n>` | Maximum generated sweep runs |
| `--analyze-tlb` | Standalone TLB analysis |
| `--analyze-core2core` | Core-to-core analysis |
| `-h, --help` | Show help |

## Compatibility Matrix

### Mode Flags (exactly one required for execution)

| | `--benchmark` | `--patterns` | `--analyze-tlb` | `--analyze-core2core` |
|---|---|---|---|---|
| `--benchmark` | ✅ | ❌ mutually exclusive | ❌ | ❌ |
| `--patterns` | ❌ mutually exclusive | ✅ | ❌ | ❌ |
| `--analyze-tlb` | ❌ | ❌ | ✅ | ❌ |
| `--analyze-core2core` | ❌ | ❌ | ❌ | ✅ |

### Modifiers with `--benchmark`

| Modifier | Compatible | Notes |
|----------|------------|-------|
| `--iterations <n>` | ✅ | Exact pass override; omission enables automatic duration calibration |
| `--buffer-size <MB>` | ✅ | |
| `--count <n>` | ✅ | |
| `--seed <uint64>` | ✅ | Reproduces workload/schedule metadata across count loops; generated once when omitted |
| `--latency-samples <n>` | ✅ | Effective standard sample count is capped to the measurement access count |
| `--latency-stride-bytes <n>` | ✅ | Positive and pointer-aligned; each enabled target/configured locality window needs two nodes. The optional fixed 16 KiB comparison is unavailable above an 8192 B stride |
| `--latency-chain-mode <mode>` | ✅ | Box modes require `--latency-tlb-locality-kb > 0`; `global-random` works with locality `0` |
| `--latency-tlb-locality-kb <n>` | ✅ | Locality-using modes require a non-zero page-size multiple; explicit `global-random` ignores the value |
| `--threads <n>` | ✅ | Explicit value applies to main-memory and cache bandwidth; omitted cache bandwidth uses one worker |
| `--cache-size <KB>` | ✅ | Replaces auto L1/L2 cache tests with one custom cache target |
| `--only-bandwidth` | ✅ | ❌ with `--cache-size`, ❌ with `--latency-samples` |
| `--only-latency` | ✅ | ❌ with `--iterations`, needs `--buffer-size > 0` or `--cache-size > 0` |
| `--non-cacheable` | ✅ | |
| `--output <file>` | ✅ | |
| `--sweep <key=a,b>` | ✅ | Requires `--output`; supported keys depend on benchmark subtype, see [Sweep Compatibility](#sweep-compatibility) |
| `--sweep-max-runs <n>` | ✅ with `--sweep` | Default `256` |
| `--buffer-size 0` | ✅ only with `--only-latency` | Disables main memory latency |
| `--cache-size 0` | ✅ only with `--only-latency` | Disables cache latency |

### Modifiers with `--patterns`

| Modifier | Compatible | Notes |
|----------|------------|-------|
| `--buffer-size <MB>` | ✅ | |
| `--count <n>` | ✅ | |
| `--seed <uint64>` | ✅ | Reproduces random workload; generated once when omitted |
| `--latency-samples <n>` | Accepted, ignored | Value must parse; pattern mode has no latency path |
| `--latency-stride-bytes <n>` | Accepted, ignored | Value must validate; pattern mode has no latency pointer chain |
| `--latency-chain-mode <mode>` | Accepted, ignored | Mode/locality combination must validate; pattern mode has no latency pointer chain |
| `--latency-tlb-locality-kb <n>` | Accepted, ignored | Value must validate; pattern mode has no latency pointer chain |
| `--threads <n>` | ✅ | |
| `--cache-size <KB>` | Accepted, ignored | Non-zero value must validate; pattern mode does not run cache tests |
| `--only-bandwidth` | ❌ | Separate execution mode |
| `--only-latency` | ❌ | Separate execution mode |
| `--non-cacheable` | ✅ | |
| `--output <file>` | ✅ | |
| `--sweep <key=a,b>` | ✅ | Requires `--output`; supported keys: `buffer-size`, `threads` |
| `--sweep-max-runs <n>` | ✅ with `--sweep` | Default `256` |
| `--iterations <n>` | ✅ | Exact pass override; omission enables automatic duration calibration |

### Modifiers with `--analyze-tlb` (standalone mode)

| Modifier | Compatible | Notes |
|----------|------------|-------|
| `--output <file>` | ✅ | |
| `--latency-stride-bytes <n>` | ✅ | |
| `--latency-chain-mode <mode>` | ✅ | `global-random` is rejected with `--analyze-tlb` |
| `--tlb-density <low\|medium\|high>` | ✅ | Default `medium`/standard; low=quick, high=exhaustive |
| `--seed <uint64>` | ✅ | Fixed reproducibility seed; generated once when omitted |
| `--sweep <key=a,b>` | ✅ | Requires `--output`; supported keys: `latency-stride-bytes`, `latency-chain-mode`, `tlb-density` |
| `--sweep-max-runs <n>` | ✅ with `--sweep` | Default `16`; explicit value overrides |
| All others | ❌ | Must be used alone |

### Modifiers with `--analyze-core2core` (standalone mode)

| Modifier | Compatible | Notes |
|----------|------------|-------|
| `--output <file>` | ✅ | |
| `--count <n>` | ✅ | Core-to-core default `3` (general default remains `1`); scenario order rotates and the headline is the loop median P50 |
| `--latency-samples <n>` | ✅ | Separate calibrated sample windows per scenario/loop; does not define the continuous headline |
| `--sweep <key=a,b>` | ✅ | Requires `--output`; supported keys: `count`, `latency-samples` |
| `--sweep-max-runs <n>` | ✅ with `--sweep` | Default `256` |
| All others | ❌ | Must be used alone |

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

Additional sweep rules:

- `--sweep-max-runs <n>` limits the generated Cartesian product; default is `16` with `--analyze-tlb` and `256` otherwise.
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
| `--only-bandwidth` + `--patterns` | Separate modes |
| `--only-latency` + `--patterns` | Separate modes |
| `--benchmark` + `--patterns` | Mutually exclusive |
| `--sweep` without `--output` | Sweep mode requires a combined JSON output file |
| `--sweep` generated runs > `--sweep-max-runs` | Guardrail against accidental large Cartesian sweeps |

### No Mode Flag (shows help)

Running with only modifier flags and no mode flag (`--benchmark`, `--patterns`, `--analyze-tlb`, `--analyze-core2core`) shows help and exits.

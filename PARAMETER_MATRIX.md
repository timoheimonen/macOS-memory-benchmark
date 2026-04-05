# Parameter Compatibility Matrix

Last updated 2026-04-05
Version 0.55.0

## All Flags

| Flag | Description |
|------|-------------|
| `-benchmark` | Run standard benchmark |
| `-patterns` | Run pattern benchmark |
| `-iterations <n>` | Iterations for R/W/Copy tests |
| `-buffersize <MB>` | Main memory buffer size |
| `-count <n>` | Number of benchmark loops |
| `-latency-samples <n>` | Latency samples per test |
| `-latency-stride-bytes <n>` | Latency pointer chain stride |
| `-latency-chain-mode <mode>` | Chain construction policy |
| `-latency-tlb-locality-kb <n>` | TLB-locality window |
| `-threads <n>` | Thread count |
| `-cache-size <KB>` | Custom cache size |
| `-only-bandwidth` | Bandwidth tests only |
| `-only-latency` | Latency tests only |
| `-non-cacheable` | Cache-discouraging hints |
| `-output <file>` | JSON output file |
| `-analyze-tlb` | Standalone TLB analysis |
| `-analyze-core2core` | Core-to-core analysis |
| `-h, --help` | Show help |

## Compatibility Matrix

### Mode Flags (exactly one required for execution)

| | `-benchmark` | `-patterns` | `-analyze-tlb` | `-analyze-core2core` |
|---|---|---|---|---|
| `-benchmark` | ✅ | ❌ mutually exclusive | ❌ | ❌ |
| `-patterns` | ❌ mutually exclusive | ✅ | ❌ | ❌ |
| `-analyze-tlb` | ❌ | ❌ | ✅ | ❌ |
| `-analyze-core2core` | ❌ | ❌ | ❌ | ✅ |

### Modifiers with `-benchmark`

| Modifier | Compatible | Notes |
|----------|------------|-------|
| `-iterations <n>` | ✅ | |
| `-buffersize <MB>` | ✅ | |
| `-count <n>` | ✅ | |
| `-latency-samples <n>` | ✅ | |
| `-latency-stride-bytes <n>` | ✅ | |
| `-latency-chain-mode <mode>` | ✅ | Requires `-latency-tlb-locality-kb > 0` for non-auto modes |
| `-latency-tlb-locality-kb <n>` | ✅ | |
| `-threads <n>` | ✅ | |
| `-cache-size <KB>` | ✅ | Only relevant for latency tests |
| `-only-bandwidth` | ✅ | ❌ with `-cache-size`, ❌ with `-latency-samples` |
| `-only-latency` | ✅ | ❌ with `-iterations`, needs `-buffersize > 0` or `-cache-size > 0` |
| `-non-cacheable` | ✅ | |
| `-output <file>` | ✅ | |
| `-buffersize 0` | ✅ only with `-only-latency` | Disables main memory latency |
| `-cache-size 0` | ✅ only with `-only-latency` | Disables cache latency |

### Modifiers with `-patterns`

| Modifier | Compatible | Notes |
|----------|------------|-------|
| `-buffersize <MB>` | ✅ | |
| `-count <n>` | ✅ | |
| `-latency-samples <n>` | ❌ | Not applicable |
| `-latency-stride-bytes <n>` | ❌ | Not applicable |
| `-latency-chain-mode <mode>` | ❌ | Not applicable |
| `-latency-tlb-locality-kb <n>` | ❌ | Not applicable |
| `-threads <n>` | ✅ | |
| `-cache-size <KB>` | ❌ | Not applicable |
| `-only-bandwidth` | ❌ | Separate execution mode |
| `-only-latency` | ❌ | Separate execution mode |
| `-non-cacheable` | ✅ | |
| `-output <file>` | ✅ | |
| `-iterations <n>` | ❌ | Not applicable |

### Modifiers with `-analyze-tlb` (standalone mode)

| Modifier | Compatible | Notes |
|----------|------------|-------|
| `-output <file>` | ✅ | |
| `-latency-stride-bytes <n>` | ✅ | |
| `-latency-chain-mode <mode>` | ✅ | |
| All others | ❌ | Must be used alone |

### Modifiers with `-analyze-core2core` (standalone mode)

| Modifier | Compatible | Notes |
|----------|------------|-------|
| `-output <file>` | ✅ | |
| `-count <n>` | ✅ | |
| `-latency-samples <n>` | ✅ | |
| All others | ❌ | Must be used alone |

### Incompatible Modifier Combinations

| Pair | Reason |
|------|--------|
| `-only-bandwidth` + `-only-latency` | Mutually exclusive |
| `-only-bandwidth` + `-cache-size` | cache-size only for latency |
| `-only-bandwidth` + `-latency-samples` | latency-samples only for latency |
| `-only-latency` + `-iterations` | iterations only for bandwidth |
| `-only-bandwidth` + `-patterns` | Separate modes |
| `-only-latency` + `-patterns` | Separate modes |
| `-benchmark` + `-patterns` | Mutually exclusive |

### No Mode Flag (shows help)

Running with only modifier flags and no mode flag (`-benchmark`, `-patterns`, `-analyze-tlb`, `-analyze-core2core`) shows help and exits.

# Script-example fixtures

## Supported schema-3 fixtures

`standard-schema-v3-complete-current.json` and `standard-schema-v3-custom-complete-current.json` are captured complete
results for the bundled standard-memory script examples. They cover the supported schema and methodology,
configuration, work metadata, completion fields, metric blocks, and raw output targets consumed by those examples.

To refresh the fixtures, build `memory_benchmark` and run:

```bash
./memory_benchmark --benchmark --iterations 1 --buffer-size 1 --count 1 --latency-samples 1 --output /tmp/standard-schema-v3-complete-current.capture.json
./memory_benchmark --benchmark --only-latency --buffer-size 0 --cache-size 16 --count 1 --latency-samples 1 --latency-tlb-locality-kb 16 --latency-stride-bytes 256 --output /tmp/standard-schema-v3-custom-complete-current.capture.json
```

Replace the corresponding fixture files with the captured outputs. The full run supplies the bandwidth, L1/L2 headline
latency, and automatic-locality paths used by both plotters. The custom-cache run supplies the pooled sample distribution
and work metadata used by the shell examples. Refresh the fixtures, examples, and entry-path assertions together when
the consumed schema, methodology, or result shape changes. A `SOFTVERSION`-only change does not require a fixture
refresh. Run `make test-script-examples` after updating them.

## LLM schema 2 verifier fixtures

`llm-schema-v2/` contains exact producer captures from the Phase 7A binary (build Git provenance
`870b175927d747fb1d90fadefebe8728da38407a`, dirty tree; executable SHA-256 retained in each artifact).
The eight admitted profiles used W=1 MiB, layers=2, query heads=2, KV heads=1, head dimension=3,
element bytes=1, batch=2, iterations=2, count=3 and seed=424242. CPU requested two workers.
Decode used context=5; prefill used P=5/Q=2; paged layouts used block tokens=2. Explicit file output
preserves the terminal snapshot. The unsupported fixture used Metal decode contiguous, layers/heads/dimension/
context/iterations/count=1, weight=1 MiB, seed=0 and default element width, under seatbelt without device access.
These captured timestamps/build values are immutable provenance data, not documentation release dates.

`test_llm_result_verifier.py` separately authors partial/interrupted/failed/invalid states with exact populations.
Its independent one-byte goldens cover every profile without treating these producer captures as the oracle.
The multi-byte affine arithmetic is also checked against direct byte enumeration, including unaligned boundaries.
The fixture gate has no device or producer executable dependency; real-device readback remains a separate gate.

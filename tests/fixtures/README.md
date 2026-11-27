# Test fixtures

## CPU Bandwidth & CPU Latency schema-1 verifier fixtures

`cpu-bandwidth-schema-v1-complete-current.json` and `cpu-latency-schema-v1-complete-current.json` are schema-1 fixtures for the standalone `-B`/`--cpu-bandwidth` (`mode: "cpu_bandwidth"`) and `-L`/`--cpu-latency` (`mode: "cpu_latency"`) modes, exercised by `tests/test_cpu_bandwidth_verifier.py` and `tests/test_cpu_latency_verifier.py` (`make test-cpu-verifiers`) as well as `tests/test_script_examples.py` (`make test-script-examples`).

- `cpu-bandwidth-schema-v1-complete-current.json`: complete `-B` fixture covering main memory + custom cache bandwidth measurements, GF(2) oracle / write-pattern post-validation checks, and raw Mach timing evidence.
- `cpu-latency-schema-v1-complete-current.json`: complete `-L` fixture covering custom cache + main memory latency measurements, canonical pointer-chain permutation hashes, final-pointer verification, and `pooled_sample_distribution` sample-window evidence.

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

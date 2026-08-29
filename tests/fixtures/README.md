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

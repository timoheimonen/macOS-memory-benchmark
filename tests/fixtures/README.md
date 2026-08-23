# Current script-example fixtures

## Current schema-3 producer captures

`standard-schema-v3-complete-current.json` and `standard-schema-v3-custom-complete-current.json` are exact captures from
the repository's real 0.63.0 standard producer in the Phase 0 working tree based on source HEAD
`0fbd050312c29c40859ffd89c7fb61bcee5d3b9e`. The commands minimize volatile populations at capture time to one loop and
one latency sample while preserving the producer's configuration, work metadata, completion fields, metric blocks, and
raw output target:

```bash
./memory_benchmark --benchmark --iterations 1 --buffer-size 1 --count 1 --latency-samples 1 --output /tmp/standard-schema-v3-complete-current.capture.json
./memory_benchmark --benchmark --only-latency --buffer-size 0 --cache-size 16 --count 1 --latency-samples 1 --latency-tlb-locality-kb 16 --latency-stride-bytes 256 --output /tmp/standard-schema-v3-custom-complete-current.capture.json
```

The captures have SHA-256 digests
`81a35dbf70c60c03d72039e9bd0c6b00d71d38f41180b6c0e78e1b293c5019f6` and
`06afb3b7b35c642b039446fef6acaa4e2ffd4fb4acf11aa4042fd567ace07a94`, respectively. The full run supplies the
bandwidth, L1/L2 headline latency, and automatic-locality paths used by both plotters. The custom-cache run supplies the
pooled sample distribution and work metadata used by the shell examples. The fixtures intentionally cover only the
current producer shape and are refreshed together with the examples when that shape changes.

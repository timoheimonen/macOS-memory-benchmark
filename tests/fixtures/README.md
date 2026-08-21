# Current script-example fixtures

## Current schema-3 producer captures

`standard-schema-v3-complete-current.json` and `standard-schema-v3-custom-complete-current.json` are exact captures from
the repository's real 0.62.0 standard producer at source HEAD
`d73c7e71ebe167e1d90f64e3cb3d223aec380d3e`. The commands minimize volatile populations at capture time to one loop and
one latency sample while preserving the producer's configuration, work metadata, completion fields, metric blocks, and
raw output target:

```bash
./memory_benchmark --benchmark --iterations 1 --buffer-size 1 --count 1 --latency-samples 1 --output /tmp/standard-schema-v3-complete-current.capture.json
./memory_benchmark --benchmark --only-latency --buffer-size 0 --cache-size 16 --count 1 --latency-samples 1 --latency-tlb-locality-kb 16 --latency-stride-bytes 256 --output /tmp/standard-schema-v3-custom-complete-current.capture.json
```

The committed captures have SHA-256 digests
`39fa5db050bb00cabe997bd6800557819f5fe5aa1710d869bab243f91d37d436` and
`a49448a887f4755e0233fe9b1d8fbd5090721dbf1a8aaea48a4974539fbcb519`, respectively. The full run supplies the
bandwidth, L1/L2 headline latency, and automatic-locality paths used by both plotters. The custom-cache run supplies the
pooled sample distribution and work metadata used by the shell examples. The fixtures intentionally cover only the
current producer shape and are refreshed together with the examples when that shape changes.

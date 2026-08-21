# Strict current standard-result fixtures

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
pooled sample distribution and work metadata used by the jq, Python, and stride/TLB summary paths. These mutually
exclusive producer modes remain separate; neither fixture combines metric blocks from another run or promotes older
data by changing its schema number.

## Released schema-2 negative evidence

`standard-schema-v2-complete-released.json` preserves the standard schema-2 shape emitted by released version 0.58.0.
Current bundled readers intentionally reject it as unsupported. It omits both `conclusions_valid` and
`configuration.output_file`, because neither field was part of that released contract.

The fixture is a minimized composition of two real Apple M4 validation results produced by version 0.58.0 (release tag
`v.0.58.0`, commit `4cff40649883f1ff4c1eb0f3d7cd0a1d0a13e3e2`):

- `plans/benchmark-stability-validation/full-1m-t4-c5-auto.json` (SHA-256
  `2db160a0eebf02584cd4e79df9db03a331b5b487087b39ff98210ed3bb669da9`) supplies the schema and methodology identities,
  completion fields, L1/L2 headline latency, main-memory bandwidth, and automatic-locality metric blocks.
- `plans/benchmark-stability-validation/lat-64m-custom4m-c1-s1000-global.json` (SHA-256
  `2d2aaf54b43f3460b4c3e82e723e26a0596450558ce4bf0b06abd77b8c16e86b`) supplies the custom-cache pooled latency block
  needed by the bundled shell consumers.

The historical producer recorded built-in-cache and custom-cache runs separately. This older fixture combines those
released-shape metric blocks only to preserve compact negative evidence from the former compatibility suite. Metric
values are copied from the source results; large measurement and sample arrays, unrelated configuration fields, and
statistics unused by that evidence are removed. It must not be used as a base for a current schema-3 positive case.

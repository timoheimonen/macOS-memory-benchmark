# Standard result compatibility fixtures

`standard-schema-v2-complete-released.json` preserves the standard schema-2 shape emitted by released version 0.58.0.
It intentionally omits both `conclusions_valid` and `configuration.output_file`, because neither field was part of that
released contract.

The fixture is a minimized composition of two real Apple M4 validation results produced by version 0.58.0 (release tag
`v.0.58.0`, commit `4cff40649883f1ff4c1eb0f3d7cd0a1d0a13e3e2`):

- `plans/benchmark-stability-validation/full-1m-t4-c5-auto.json` (SHA-256
  `2db160a0eebf02584cd4e79df9db03a331b5b487087b39ff98210ed3bb669da9`) supplies the schema and methodology identities,
  completion fields, L1/L2 headline latency, main-memory bandwidth, and automatic-locality metric blocks.
- `plans/benchmark-stability-validation/lat-64m-custom4m-c1-s1000-global.json` (SHA-256
  `2d2aaf54b43f3460b4c3e82e723e26a0596450558ce4bf0b06abd77b8c16e86b`) supplies the custom-cache pooled latency block
  needed by the bundled shell consumers.

The producer records built-in-cache and custom-cache runs separately, so the two released-shape metric blocks are
combined here to keep one small fixture usable by all four bundled consumers. Metric values are copied from the source
results; large measurement and sample arrays, unrelated configuration fields, and statistics unused by the consumers
are removed.

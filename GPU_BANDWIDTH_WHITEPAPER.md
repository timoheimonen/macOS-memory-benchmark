# Metal GPU Memory Bandwidth Whitepaper

- **Software version:** 0.61.0
- **JSON schema:** 1
- **Methodology:** `gpu-bandwidth-v1-private-runtime-single-cmdbuf-calibrated-balanced`
- **Platform:** macOS on Apple Silicon

This document defines the current `memory_benchmark --gpu-bandwidth` measurement and consumer contract. It describes
what the number means, what is validated, what remains unknown, and which implementation changes require a methodology
review. Source code and tests remain authoritative if this document and runtime behavior ever diverge.

## 1. Scope

GPU schema 1 measures three Metal compute workloads over unified-memory resources:

| Operation | Logical data flow | Exact effective payload |
|---|---|---:|
| Read | Private buffer A → read/reduction kernel | `buffer_size × passes` |
| Write | Generated values → private buffer A | `buffer_size × passes` |
| Copy | Private A ↔ private B, alternating by pass | `2 × buffer_size × passes` |

The unit is decimal GB/s:

```text
value_gb_s = exact_payload_bytes / gpu_elapsed_seconds / 1e9
```

The following are outside schema 1:

- CPU↔GPU transfer or handoff bandwidth
- Simultaneous CPU/GPU contention
- GPU memory latency
- A separate Metal blit benchmark
- Shared-storage comparison
- Textures, tile memory, or render-pipeline traffic
- Discrete GPU and Intel Mac support
- GPU parameter sweeps
- Offline `.metallib`, Metal binary archives, or persistent pipeline caches

Copy's 2× numerator is aggregate read-plus-write payload, matching the project's logical copy convention. It does not
make the result a one-direction bus rate, and it is not a CPU↔GPU transfer number.

## 2. Interpretation Boundary

Apple Silicon uses unified system memory. `MTLStorageModePrivate` means that the Metal resource is GPU-private from the
CPU access/API perspective; it does not mean that a separate VRAM device exists. The benchmark therefore calls the
results GPU memory read/write/copy bandwidth.

The result is effective payload bandwidth for the exact versioned compute kernel at one Metal command-buffer timing
boundary. It does not prove that every byte reached physical DRAM. GPU caches, command processing, the observable
reduction used for correctness, other GPU work, thermals, Low Power Mode, power management, the runtime Metal compiler,
and the driver can affect the value.

Every schema 1 file therefore records:

```json
"dram_residency": "unverified"
```

The 64 MB CLI minimum excludes extremely small routine runs; it is not an SLC-size claim and does not verify DRAM
traffic. A large buffer may reduce cache dominance but cannot establish residency by itself. A separately instrumented
GPU-counter capture can provide supporting evidence for the same command, but its counters and perturbed timing are not
the production GB/s source.

CPU and GPU results are not directly comparable even though both use decimal GB/s and logical 2× copy accounting. Their
timing boundaries, parallelism, kernels, caches, resource modes, dispatch overhead, and correctness mechanisms differ.

## 3. CLI Contract

The primary mode selector is:

```text
-G, --gpu-bandwidth
```

The exact schema 1 whitelist is:

- `-b, --buffer-size <MB>`
- `-i, --iterations <count>`
- `-r, --count <count>`
- `--seed <uint64>`
- `-o, --output <file>`
- `-h, --help`

All other flags, including every other primary mode, CPU thread/cache/latency modifiers, non-cacheable mode, and sweep
options are rejected. Duplicate options, unknown options, missing values, partial decimal tokens, signed seeds, overflow,
and non-positive iterations/count are rejected by the dedicated parser.

Defaults and limits:

| Setting | Schema 1 behavior |
|---|---|
| Buffer size | 512 MB per private data buffer |
| Minimum buffer | 64 MB per private data buffer |
| Data-buffer count | Two, retained for the whole suite |
| Count | Three loops |
| Iterations omitted | Automatic per-operation duration calibration |
| Seed omitted | One generated base seed for the command |
| Output omitted | Console only; no checkpoint file |

`--buffer-size` uses the project MB convention of 1,048,576 bytes. The requested value is never silently reduced. A
value below 64 MB fails before Metal initialization and produces no result JSON. After valid parsing, Metal
`maxBufferLength`, suite memory budget, and actual allocation can still fail with an auditable schema checkpoint when
`--output` is present.

An explicit `--iterations` is the exact number of full-buffer passes and therefore the exact number of timed dispatches.
It bypasses pilot/correction calibration but not measured-task warmup or preconditioning. It is rejected before GPU work
if copy would exceed either the 16,384-dispatch guardrail or the 64 GiB exact-payload guardrail; the value is never capped.

The base seed is stored as an exact uint64 decimal string with source `user` or `generated`. Stable read/write/copy
operation seeds are derived with SplitMix64 over separate versioned domains. Repeating a seed reproduces data and work
identity; it does not reproduce performance or macOS scheduling.

## 4. Platform, Build, and Capability Boundary

The production/test build has one deployment target:

```text
MACOSX_DEPLOYMENT_TARGET=11.0
```

The Objective-C++ backend is auto-discovered, compiled with `-fobjc-arc`, and linked for both release and tests with:

```text
-framework Metal -framework Foundation
```

No third-party production dependency is added. CommonCrypto SHA-256 comes from the platform/libSystem API and needs no
additional framework link. Coverage includes production `.mm` beside `.cpp`.

Runtime admission requires:

1. `MTLCreateSystemDefaultDevice()` returns a device.
2. `device.hasUnifiedMemory` is true.
3. `device supportsFamily:MTLGPUFamilyApple7` is true.

Failure of step 1–3 is top-level `unsupported`, not a numeric measurement. Future devices are not rejected by name or by
absence from a latest-known-family allowlist if the required capability succeeds. Supported family names, including
availability-guarded Metal3/Metal4 where available, remain diagnostic metadata.

Apple7 is a capability minimum, not a performance-validation claim. A device can be capability-supported while its
performance is unvalidated.

## 5. Backend and Resource Model

`GpuBackend` is a pure C++17 synchronous/noexcept interface. The private concrete `MetalGpuBackend` is the only file that
imports Objective-C/Metal. Every entry point uses a bounded autorelease pool and converts exceptions, nil returns,
NSError, and command-buffer failures into explicit status/reason/diagnostic objects.

One backend instance owns:

- One `MTLDevice`
- One `MTLCommandQueue`
- One runtime-compiled `MTLLibrary`
- Initialization, read, write, copy, and validation compute pipelines
- Two suite-resident private data buffers
- One shared checksum/status buffer

Data resources are created with:

```text
MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeTracked
```

The status resource uses:

```text
MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked
```

The backend reads back and records each resource's actual storage mode, CPU cache mode, hazard mode, raw resource
options, label, and exact length. Defaults are not inferred from the request alone.

Before allocation, the runner checks overflow and computes:

```text
required_total_bytes = 2 × requested_buffer_bytes + 4096
```

The hard suite budget is 80% of the project's available-memory estimate. If that estimate is zero, the existing 2048
MiB fallback total budget is used. Each data buffer must also fit `maxBufferLength`. `recommendedMaxWorkingSetSize` is an
advisory value. Signed byte headroom is `recommended - current_allocated_before - required_total`; relative headroom
divides that signed value by the recommendation. Both are serialized, a negative value sets
`exceeds_recommended_working_set`, and exceeding the recommendation is diagnostic rather than an unsupported/failure
decision by itself. Current allocated size is captured before allocation, at the suite-resident peak, and after release.

## 6. Runtime MSL and Provenance

The canonical kernel source is one embedded raw string in `gpu_kernels_source.h`; no generated `.metal` duplicate exists.
The backend compiles it once during initialization with `newLibraryWithSource:options:error:` and:

- `MTLLanguageVersion2_3`
- Empty preprocessor macros
- Integer-only kernels
- No fast-math/math-mode API setting

Schema fields include:

- `compilation_mode: "runtime-source"`
- `msl_language_version: "2.3"`
- `floating_point_math: "not_applicable_integer_only"`
- `preprocessor_macros: "none"`
- `kernel_revision: "gpu-linear-word-mod32-tg-reduce-v2"`
- `kernel_source_sha256`: SHA-256 of the exact canonical UTF-8 source bytes
- Compiler identifier, build SDK, deployment target, and compiler diagnostics

A non-nil runtime library is success even if NSError contains a warning; diagnostics are preserved. A nil library is
failure. The source hash proves input-byte identity, not identical generated GPU machine code. Hardware/GPU, macOS build,
source hash, MSL/options, resource options, and frozen work plan are all required strict-cohort keys.

Runtime source compilation keeps Command Line Tools builds independent of the optional offline Metal Toolchain. Moving
to `.metallib`, changing MSL/options, or adding a binary archive is a methodology decision, not an invisible build tweak.

## 7. Data, Kernels, and Correctness

Data is a deterministic affine sequence of little-endian 32-bit words derived from operation seed, pattern tag, index,
and (for write) pass. Full data is processed as consecutive `uint4` values. A direct backend request also handles an
exact 0–15 byte suffix without overrun; normal CLI MB sizes are 16-byte aligned, while integration tests exercise tails.

Before every excluded or measured timed command:

- Read restores A to its read-source pattern.
- Write poisons A; each timed pass writes a pass-specific pattern.
- Copy restores A to the copy-source pattern and B to poison.
- Timed and final checksum words are reset outside the primary time.

### Read

Each pass reads the full logical buffer. The `gpu-dual-mod32-v2` timed kernel first reduces every data word and logical
word index into two thread-local modulo-2^32 lanes. It then multiplies each local lane once by a pass-specific odd
domain weight and has `global_id == 0` add one nonzero dispatch token per lane. The independently mixed domains include
the operation seed, full 64-bit buffer size, pass, operation, and copy direction. SIMD/threadgroup reduction limits
global atomic updates to two per threadgroup. The CPU computes the same expected result independently in O(passes)
from closed-form affine-pattern summaries; it neither allocates the logical buffer nor consumes the GPU result. Read
validation needs no additional GPU command because every planned dispatch contributes its own token and every word
still contributes data/index evidence.

### Write

Each pass writes a deterministic pass-specific full-buffer pattern without reading the destination. The timed v2 dual
accumulator proves that each planned pass contributed. An excluded `gpu-dual-mod32-v1` final checksum command
additionally verifies the last written buffer.

### Copy

Passes alternate A→B, B→A. Each pass therefore consumes the preceding pass's output, and final-buffer identity follows
pass parity. The timed v2 dual accumulator includes direction/pass information. An excluded v1 final-checksum command
verifies the parity-selected final buffer. Exact payload is 2× buffer bytes per pass.

For write/copy, the real-Metal integration suite also blits the private final buffer to a shared staging buffer outside
timing and compares every byte with an independent CPU oracle. This catches common-mode GPU initialization/checksum
errors. The production dual checksum is a strong error detector, not a mathematical collision-free proof.

Correctness outcome rules:

- Completed command + valid GPU timestamps + expected timed accumulator + passed required validation → `measured`
- Invalid/non-finite timestamp → measurement `invalid`, validation `not-run-timer-invalid`
- Timed accumulator or final checksum mismatch → measurement `invalid`
- Metal command/validation error → measurement `failed`
- Only `measured` receives finite positive `value_gb_s`

## 8. Deterministic Work Plan

Each valid `GpuWorkPlan` records:

- Operation, requested/effective buffer bytes, base seed, and operation seed
- Passes, payload multiplier, bytes per pass, exact payload, and pass ceilings
- 16-byte vector width, vector count, and tail bytes
- Pipeline execution width and maximum threads
- Threads per threadgroup, required/capped threadgroups, and grid threads
- Dispatch count
- `measured_command_buffer_count = 1`
- `measured_compute_encoder_count = 1`
- Actual data/status resource-option values and storage/hazard modes
- Kernel revision, exact source SHA-256, MSL language, and methodology version
- Timing, pass mapping, warmup, calibration, duration-quality, `gpu-dual-mod32-v2` timed-accumulator, and
  operation-specific `not-applicable`/`gpu-dual-mod32-v1` final-checksum policies
- Canonical `gpu-work-plan-v1` identity

Grid geometry is frozen as follows:

1. `threads_per_threadgroup` is the largest pipeline execution-width multiple no greater than
   `min(256, maxTotalThreadsPerThreadgroup)`.
2. `threadgroups_per_grid` is at least one and no greater than
   `min(ceil(vector_count / threads_per_threadgroup), 8192)`.
3. Threads cover all vectors with a grid-stride loop.

The 8192-threadgroup cap, 16,384-dispatch cap, 64 GiB payload cap, reduction contract, and geometry algorithm are
methodology constants. Runtime does not tune the grid by device name.

## 9. Warmup and Calibration

Every attempt has warm-memory semantics: a same-shape full-buffer warmup and deterministic precondition complete before
the timed command. GPU caches are not flushed.

When iterations are omitted, read, write, and copy are calibrated independently before loop 0:

1. Select a pilot with at least 8 MiB payload when the operation guardrails permit.
2. Run excluded warmup, precondition, timed pilot, and required validation.
3. Scale pass count toward 150 ms.
4. Run an excluded duration trial.
5. If outside the inclusive 100–250 ms window, run at most two excluded correction trials when a guardrail permits a
   different count.
6. Freeze the last valid plan and reuse it unchanged across all measured loops.

Every excluded attempt is retained in `excluded_calibration_attempts` with purpose, passes, exact payload, phase
statuses/counts, GPU timing, validation, duration quality, and stable reason. An invalid excluded attempt fails
calibration; it is not replaced with a performance retry.

Automatic duration quality is one of the exact classifications emitted by the planner, including
`within-target-window`, `below-target-window`, `above-target-window`, `single-pass-exceeds-window`,
`dispatch-cap-below-target`, and `payload-cap-below-target`. A measured loop outside the window remains evidence and does
not trigger recalibration.

## 10. Command-Buffer Timing Boundary

One measured attempt contains exactly one command buffer and one explicitly serial compute encoder. It contains only one
operation's frozen full-buffer dispatches. Serial encoding orders write/copy side effects, so schema 1 adds no redundant
barrier between passes.

After `waitUntilCompleted` returns, the backend records:

- `GPUStartTime`
- `GPUEndTime`
- Their exact positive finite difference
- Host submit, wait-end, and wall envelope
- Queue delay only when a compatible value is available

Only GPU elapsed time is the bandwidth denominator. GPU-side command processing within the command-buffer timestamp is
included. Pipeline creation, initialization/poison, warmup, precondition, final checksum, and test readback are excluded.
Schema 1 does not sum several timed command buffers.

## 11. Order, Statistics, and Quality

The planned loop order rotates:

1. read → write → copy
2. write → copy → read
3. copy → read → write

Every loop records planned and realized order; every measurement records its order position. Order balance is true only
when results are complete and completed loop count is divisible by three.

Aggregates use only `measured` values:

- One value: direct single-measurement headline
- Multiple values: median/P50 headline
- Statistics: average, median, P90/P95/P99, sample standard deviation, CV, MAD, minimum, maximum
- Fewer than three values: `insufficient-samples`
- At least three and CV > 5%: `noisy`
- At least three and CV ≤ 5%: `stable`

The 5% threshold is diagnostic. No sample is removed, winsorized, or retried because of performance. Non-nominal thermal
state or Low Power Mode produces a separate environment warning and makes the run unsuitable for a reference baseline
without erasing the raw value.

## 12. Status, Counters, and Null Semantics

Top-level run statuses are:

```text
not-started, complete, partial, interrupted, failed, unsupported
```

Measurement statuses are:

```text
not-run, measured, interrupted, invalid, failed
```

Only `measured` can have numeric `value_gb_s`. Unavailable/non-finite/failed/interrupted values are `null` with a stable
reason. Numeric zero is never an unavailable sentinel.

Counter definitions:

| Counter | Definition |
|---|---|
| `planned_loops` | Requested count |
| `attempted_loops` | Loops whose first operation warmup started |
| `completed_loops` | Loops whose three operations are all measured |
| `planned_measurements` | `planned_loops × 3` |
| `attempted_measurements` | Operations whose warmup started |
| `terminal_measurements` | Slots whose status is no longer `not-run` |
| `completed_measurements` | Attempts with completed timed command and terminal validation, even if later invalid |
| `validated_measurements` | Measurements whose final status is `measured` |

`results_complete` and `conclusions_valid` are true only when top-level status is complete and every planned measurement
is validated. Full completed/validated counters can coexist with false completeness if interruption is first observed at
the final checkpoint boundary. Consumers must use the booleans and status, not infer completeness from counts or process
exit code alone.

## 13. Interruption and Atomic Checkpoints

The benchmark cannot reliably cancel a committed Metal command buffer. Schema 1 therefore uses task-level
completion-wins. One logical measured task begins at operation warmup and ends after deterministic precondition, timed
command, and required validation reach a terminal result.

Stop is checked before the task and after its terminal result, not between its internal phases. Once a task starts:

- A signal does not cancel or null the current valid result.
- Normal error short-circuiting still applies; a failed warmup/precondition prevents timed submission, a timed command
  error prevents validation submission, and invalid timestamps prevent checksum submission.
- A successfully timed task performs its required validation exactly once even when a signal is pending.
- No next calibration attempt, measured operation, or loop starts after the signal is observed.

After terminal current status is resolved, all remaining not-started slots become:

```json
{
  "status": "interrupted",
  "reason_code": "interruption-before-task",
  "value_gb_s": null
}
```

Existing measured/invalid/failed slots are not rewritten. A real command, timer, validation, or checkpoint failure has
priority over interruption. Graceful interruption returns `EXIT_SUCCESS` for orderly shutdown but serializes
`status: "interrupted"`, `results_complete: false`, and `conclusions_valid: false`.

Without a pending interruption, a runtime failure finalizes every later not-started planned slot as `failed`/null with
`reason_code: "not-run-after-runtime-failure"`; the current real failure and top-level status remain failed. If the same
failure coincides with a pending signal, the current real failure still wins, `interruption_requested` remains true, and
the not-started tail uses the interruption finalization above. This distinction preserves both failure precedence and an
accurate record of why later work did not start.

With `--output`, the runner uses the shared atomic JSON writer after each terminal measurement. It checks stop once before
and once immediately after that checkpoint. If the post-checkpoint read first sees the signal, it writes at most one
additional interruption checkpoint, then stops. Synthetic interruption finalization does not write one file per slot.

Valid post-parse device/capability/runtime-compile/allocation/work-plan failures write one auditable checkpoint. Parser
and config failures, including a buffer below 64 MB, produce only the error/exit failure. A checkpoint write failure makes
the run failed; disk may still contain the previous successful atomic snapshot.

## 14. GPU JSON Schema 1

The top-level discriminator is independent of standard schema 2:

```json
{
  "schema_version": 1,
  "mode": "gpu_bandwidth",
  "methodology_version": "gpu-bandwidth-v1-private-runtime-single-cmdbuf-calibrated-balanced"
}
```

A standard/TLB plotter must reject this discriminator unless it explicitly implements GPU schema 1.

### 14.1 Top-level identity and semantics

- `software_version` and compatibility `version`
- UTC `timestamp`
- `schema_version`, `mode`, `methodology_version`
- `status`, stable `reason_code`, `interruption_requested`
- `results_complete`, `conclusions_valid`, `operation_order_balance_complete`
- Total host `execution_time_sec`
- `dram_residency: "unverified"`
- `payload_semantics: "effective-kernel-payload-divided-by-metal-gpu-time"`
- `copy_payload_semantics: "aggregate-read-plus-write"`

### 14.2 Configuration and counters

`configuration` records per-buffer MB and exact bytes, explicit iterations or null, automatic/fixed work policy, loop
count, exact base seed/source, output path or null, and exact argv. `counters` uses the definitions in Section 12.

### 14.3 Environment, device, compilation, and allocation

`environment` records macOS product/build, hardware model, physical memory, main-thread QoS requested/applied/code, and
start/end thermal/Low Power Mode/current-allocation snapshots.

`backend` records initialization status/reason/error, device identity/capabilities/limits/families/pipeline geometry,
runtime compilation provenance, and allocation/resource metadata. `memory_budget` separately records exact requested,
auxiliary, required, available, budget, fallback, validity, and reason fields.

Stable `reason_code` is the semantic key. Raw NSError is separate as domain, numeric code, and free description; localized
description is never a decision key.

### 14.4 Plans, calibration, measurements, and loops

- `work_plans`: frozen operation plans and guardrails
- `excluded_calibration_attempts`: arrays keyed by read/write/copy
- `measurements`: flat planned slots, including not-run/pre-run failure state
- `loop_records`: loop index, planned/realized order, and the loop's measurement audit records

Every measurement contains status/reason/value/units, operation/loop/order, work policy and duration quality, full work
plan, warmup/precondition/timed/validation records, before/after environment, resource metadata, and kernel provenance.
Timed diagnostics include GPU/host timestamps and expected/actual accumulators. Validation includes status,
`timed_accumulator_algorithm`, `final_checksum_algorithm`, expected/actual final checksum, and phase counts. Every
warmup, precondition, timed, and validation phase record exposes
`data_initialization_dispatch_count`, `benchmark_operation_dispatch_count`, and `validation_dispatch_count`; their sum
is the phase's total `dispatch_count`. `status_reset_count` is a separate non-dispatch lifecycle count: read validation
records zero, while write/copy validation records the one host reset of its final-checksum status lanes. Read validation
can therefore pass with zero validation command buffers/dispatches because it compares the timed shared accumulator on
the CPU, while write/copy final checksum validation records its separate GPU command and validation dispatch.

### 14.5 Aggregates and exact representation

`aggregates.read`, `.write`, and `.copy` contain aggregate status, sample count, headline semantics/value, raw measured
values, descriptive statistics, stability quality, and the serialized 5% threshold. `quality_warnings` contains stable
warning tokens.

Seeds, checksums, registry ID, resource options, and byte sizes/payloads that require exact integer representation use
schema-named decimal strings. Non-finite optional numerics become null; they are never emitted as JSON NaN/Infinity.

## 15. Capability Support vs Performance Validation

Three levels must not be conflated:

1. **Capability-supported:** default Metal device, unified memory, and Apple7-family capability pass.
2. **Correctness-smoke validated:** runtime compilation, exact read/write/copy/tail behavior, timestamps, and schema
   contract pass on that hardware/OS.
3. **Performance-validated cohort:** a controlled, retained automatic/fixed-work population satisfies the release gates,
   and a separate counter audit is performed and retained with any tool/device limitations stated explicitly.

M4 is the defined schema 1 reference cohort. The controlled protocol uses nominal thermals, Low Power Mode off,
`caffeinate`, minimized competing GPU activity, exact commands/environment, a geometric 64/128/256/512 MB series (and
1024 MB when practical), automatic 512 MB/fixed-seed/count-9 runs, and a separate fixed-work 512 MB/
`--iterations 24`/same-seed/count-9 cohort. Five predeclared acceptance processes follow one excluded preconditioning
process for each policy. No noisy or ineligible run is silently replaced.

The frozen 8192-threadgroup methodology has a test-only 2048/4096/8192 comparison protocol using the same fixed work.
The production candidate is acceptable only when its per-operation count-9 median is within 2% of the best candidate;
runtime never auto-tunes this geometry. That audit is release evidence, not a user-facing runtime option.

The acceptance gate requires exact correctness/payload/counters/order, valid duration or exact cap reason, per-process
and cross-process operation CV at or below 5%, nominal thermal state, Low Power Mode off, and console/JSON agreement.
There is deliberately no minimum GB/s and no required read/write/copy ranking.

The controlled 0.61.0 M4 campaign was retained and audited locally; its large raw JSON, console, XML, and trace files
are intentionally excluded from Git. Its frozen pre-remediation identity is kernel revision
`gpu-linear-word-mod32-tg-reduce-v2`, canonical MSL SHA-256
`b9a242d2b959c9c11f6f130a52afd66f111d6761be2193beec1f051baa094296`. The exact executable SHA-256 remains with
the local validation record and machine-readable artifacts rather than being duplicated in general documentation.
Removing three unread `KernelParams` fields later in the same 0.61.0 development cycle changed the current canonical
source SHA-256 to `21def2d75d3545dba31aa4897ea57ec2fd0e4481cd86ce21725338ab0f322ac5` without changing any kernel operation or
payload count. Runtime Metal integration covers the current source's compilation and correctness; the performance
figures below remain evidence for the exact frozen pre-remediation identity.

Both final five-process acceptance populations passed the 5% gate. Automatic cross-process CV was
0.221498348705% read, 0.967311621904% write, and 0.310543092510% copy; fixed-24 cross-process CV was
0.506707339121%, 0.827667144983%, and 0.326577301613%, respectively. Median-of-process-medians values were
88.606742648049/74.383866793814/78.583784905446 GB/s for automatic read/write/copy and
91.074797816490/75.240302989483/78.508461231110 GB/s for fixed-24. Every final acceptance process was complete,
balanced, warning-free at runtime compilation, nominal-thermal, Low-Power-Mode-off, and fully validated. The
exploratory geometric series is retained locally and is not part of these acceptance gates.

The final fixed-work 2048/4096/8192 accumulator-v2 grid audit selected 8192. Its read medians were
92.169070/92.720094/93.738203 GB/s, write medians were 71.985490/75.521718/77.733721 GB/s, and copy medians were
79.830931/80.401743/81.523719 GB/s. Their 2048/4096/8192 CVs were 2.678187/0.537302/0.612318% for read,
0.818006/2.135930/2.827338% for write, and 0.742460/0.482440/0.626672% for copy. The 4096 write candidate was more than
2% below the 8192 best, while 8192 met the frozen rule for every operation. The first final3 grid attempt is retained
whole after its 2048 write CV exceeded 5%; no row was cherry-picked into the accepted rerun.

The final direct Ctrl+C audit preserved 11 completed, validated measurements—three full loops plus two operations—with
both accumulator-v2 lanes nonzero. It finalized the other 16 slots as interrupted/null without numeric-zero
placeholders. Read validation recorded zero status resets and write/copy validation recorded one, matching execution.

Rejected local cohorts are retained alongside the final population: the 4096-cap cohort records the forced grid-choice
rejection and its CV attempts; earlier 8192 automatic and fixed cohorts record predeclared CV failures; and the complete
fixed24-final3 population is rejected because process 4 write CV was 6.669560665481%. The accepted fixed24-final4
population was rerun in full. None of those records was silently dropped or cherry-picked into a final population.

The final3 counter audits used Xcode `xctrace` 16.0 (`17F113`) against the exact frozen binary; their local record
includes benchmark JSON, XML exports, and raw trace ZIP archives. The `Metal GPU Counters` profile was not supported on
the target device, while the `Game Performance` template exposed only `RT Unit Active`, not a memory-traffic counter.
Consequently the audit documents the available instrument evidence but does not establish physical DRAM residency.
Capture can alter timing, and counter traffic need not equal logical payload; counter evidence never replaces production
values. The available profiles also could not isolate or quantify timed accumulator reduction/status-atomic traffic.
That traffic remains inside the GPU-time denominator but outside the exact logical-payload numerator; the separate
final-checksum pass remains outside primary timing. Both reduction overhead and DRAM residency remain unverified. M4 is
performance-validated only for this versioned effective-payload methodology and exact hardware/OS/compiler cohort, and
`dram_residency` remains `"unverified"`. Other Apple7-capable devices remain capability-supported and
performance-unvalidated until equivalent controlled evidence is retained.

## 16. Maintenance and Revalidation Policy

- After every macOS major/minor/security build change: rerun runtime compilation, exact read/write/copy/tail correctness,
  finite timestamps, automatic calibration, and fixed-seed/count-9 repeatability before baseline comparison.
- After every Xcode/Command Line Tools/SDK change: run clean release/test/coverage builds and Metal integration tests.
- For every new Apple GPU: admit by capabilities, record exact geometry/provenance, and create a new hardware cohort.
- A change to MSL version/source/options, compilation mode, storage/hazard mode, checksum/reduction, grid cap/geometry,
  payload semantics, timing boundary, or completion/checkpoint semantics requires methodology-version review and a new
  validation population/schema compatibility decision.
- A fixed-work operation-median shift greater than 10% across macOS builds stops automatic rebaselining even if the new
  cohort CV is acceptable. Compiler/driver, environment, and counter evidence must be investigated and documented.
- Optional runtime APIs remain availability-guarded. Missing optional metadata is unavailable, not fabricated or fatal.
- Historical files are never rewritten to look like a newer methodology.

Changing to shared storage, a blit benchmark, concurrent encoders, several timed command buffers, CPU/GPU contention,
sweeps, `.metallib`, or binary archives requires an explicit methodology and schema compatibility decision.

## 17. Consumer Acceptance Checklist

Before using a GPU schema 1 result as a complete measurement, require:

1. `mode == "gpu_bandwidth"`, `schema_version == 1`, and the exact expected methodology.
2. `status == "complete"`, `results_complete == true`, and `conclusions_valid == true`.
3. Every required measurement is `measured`, finite/positive, and validation-passed.
4. Exact payload equals buffer × passes × operation multiplier.
5. Timed command/encoder counts are one and dispatch count equals passes.
6. Device has unified memory and Apple7 capability; resources are private/tracked A/B and shared/tracked status.
7. Hardware/GPU, macOS build, MSL/options, source SHA-256, resource modes, seed, and frozen plan match the comparison
   cohort.
8. Thermal/Low Power Mode are eligible and stability quality/CV meet the intended use.
9. Interpret copy as aggregate read+write and DRAM residency as unverified.

For a position-balanced reference cohort, additionally require `operation_order_balance_complete == true` and a complete
loop count divisible by three.

## 18. Implementation and Verification Map

- CLI/entry: `src/gpu_bandwidth/gpu_bandwidth.cpp`
- Pure work planning: `src/gpu_bandwidth/gpu_work_plan.cpp`
- Backend-independent runner: `src/gpu_bandwidth/gpu_runner.cpp`
- GPU schema 1: `src/gpu_bandwidth/gpu_json.cpp`
- Pure backend contract: `src/gpu_bandwidth/gpu_backend.h`
- Metal backend: `src/gpu_bandwidth/metal_gpu_backend.mm`
- Canonical MSL: `src/gpu_bandwidth/gpu_kernels_source.h`
- Capability/correctness integration: `tests/test_gpu_metal_backend.cpp`
- Planning/guardrails: `tests/test_gpu_work_plan.cpp`
- Mode routing/hash helpers: `tests/test_mode_selector.cpp`, `tests/test_hash_utils.cpp`

Recommended checks:

```bash
make
make test
make test-integration
make test-all
./memory_benchmark -h
./memory_benchmark --gpu-bandwidth --help
```

Real Metal tests may skip on an unsupported/no-device execution environment. That skip does not replace deterministic
unsupported-path tests and does not create a performance-validation claim.

## 19. Apple API References

- [Choosing a resource storage mode for Apple GPUs](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus)
- [MTLDevice unified-memory and resource properties](https://developer.apple.com/documentation/metal/mtldevice/hasunifiedmemory)
- [MTLDevice maximum buffer length](https://developer.apple.com/documentation/metal/mtldevice/maxbufferlength)
- [MTLCompileOptions](https://developer.apple.com/documentation/metal/mtlcompileoptions)
- [Runtime Metal library compilation](https://developer.apple.com/documentation/metal/mtldevice/makelibrary(source:options:))
- [MTLCommandBuffer GPU start time](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpustarttime)
- [Metal dispatch types](https://developer.apple.com/documentation/metal/mtldispatchtype)
- [Metal capabilities](https://developer.apple.com/metal/capabilities/)
- [GPU memory-bandwidth counter analysis](https://developer.apple.com/documentation/xcode/measuring-the-gpus-use-of-memory-bandwidth)

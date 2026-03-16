# Core-to-Core Cache-Line Handoff Latency Benchmark

**membenchmark — Technical Whitepaper**
*Revision 2026-03-16 — Applies to release 0.53.7 and later*

---

## TL;DR

The `-analyze-core2core` mode measures the round-trip latency of a single
cache-line bouncing between two POSIX threads on an ARM64 system. Two threads
exchange a 32-bit turn token using `LDAR`/`STLR` acquire-release instructions
in a tight ping-pong loop. The mode runs three scenarios that vary the macOS
QoS/affinity scheduler hints applied to the threads. Results are reported in
nanoseconds and optionally serialized to JSON for later analysis.

---

## 1. Overview

Cache-line handoff latency — the time for a modified cache line to travel from
one CPU core's L1 cache to another's — is a foundational metric for
multi-threaded software on shared-memory processors. It sets a hard lower
bound on the cost of every lock acquisition, producer-consumer queue handoff,
and fine-grained atomic flag.

The `membenchmark` tool provides a dedicated standalone mode,
`-analyze-core2core`, for measuring this latency on Apple Silicon and other
ARM64 targets running macOS. The mode deliberately excludes all standard
benchmark orchestration (buffer sizing, iteration counts, memory allocation)
so the measurement path is as short and reproducible as possible.

### What is being measured

A single 64-byte-aligned `uint32_t` turn token resides in a shared memory
location. Thread A (the *initiator*) spins until the token indicates it is the
initiator's turn, flips the token to the responder's value, then spins again
until the token is returned. Thread B (the *responder*) mirrors this: spin
until the responder's token value is visible, then flip it back. One *round
trip* completes when the initiator has both sent the token and received it back.

The measured quantity is therefore:

```
round_trip_latency_ns = total_timed_ns / headline_round_trips
one_way_estimate_ns   = round_trip_latency_ns / 2
```

The one-way estimate assumes symmetric latency, which holds on a homogeneous
ring interconnect (e.g., Apple M-series die-level ring bus) but is not
guaranteed when the scheduler places threads on cores of different types
(P-core vs. E-core).

---

## 2. Architecture and Design Decisions

### 2.1 Standalone mode isolation

The `-analyze-core2core` flag is detected before the standard argument parser
runs. This design choice ensures:

- None of the standard benchmark flags (`-buffersize`, `-iterations`, etc.) are
  accepted or silently ignored.
- The configuration struct `CoreToCoreLatencyConfig` remains small and
  orthogonal to `BenchmarkConfig`.
- The mode can evolve its own flags independently without risk of collisions.

The early-exit detection is in `main()`:

```cpp
for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "-analyze-core2core") {
        return run_core_to_core_latency_mode(argc, argv);
    }
}
```

Any unknown flag passed alongside `-analyze-core2core` returns `EXIT_FAILURE`
with a descriptive error message listing the permitted options.

### 2.2 Assembly hot loop

The ping-pong loop is implemented in AArch64 assembly
(`src/asm/core_to_core_latency.s`) rather than C++ to guarantee the compiler
cannot reorder, fuse, or optimize away the acquire-load / release-store
sequence that drives the handoff.

The instructions used are:

| Instruction | Semantics |
|---|---|
| `LDAR Wt, [Xn]` | Load-Acquire 32-bit — reads token with acquire ordering; no later load or store may be reordered before this |
| `STLR Wt, [Xn]` | Store-Release 32-bit — writes token with release ordering; no earlier store may be reordered after this |

This pairing is the minimum-overhead mechanism on AArch64 that guarantees
visibility across cores without issuing a full `DMB SY` barrier. On Apple M
silicon the effect is equivalent to a sequentially consistent atomic exchange
over the ring interconnect.

The inner loop body for the initiator is:

```asm
c2c_initiator_wait_turn:
    ldar w8, [x0]           // Acquire: spin until token == initiator_turn
    cmp  w8, w2
    b.ne c2c_initiator_wait_turn

    stlr w3, [x0]           // Release: hand token to responder

c2c_initiator_wait_return:
    ldar w8, [x0]           // Acquire: spin until token == initiator_turn again
    cmp  w8, w2
    b.ne c2c_initiator_wait_return

    subs x1, x1, #1
    b.ne c2c_initiator_loop
```

The responder loop omits the "wait for return" phase because from its
perspective each iteration is a single acquire-spin then release-store.

### 2.3 Cache-line alignment

The shared turn token struct `SharedTurn` is aligned to the cache line size:

```cpp
struct alignas(Constants::CACHE_LINE_SIZE_BYTES) SharedTurn {
    uint32_t value = Constants::CORE_TO_CORE_INITIATOR_TURN_VALUE;
};
```

This prevents false sharing: the token occupies its own 64-byte cache line so
no unrelated data can cause spurious invalidations that would inflate the
measured latency.

The `SharedFlags` struct (start signal and ready counter) is similarly
cache-line aligned and placed in a separate struct to avoid false sharing with
the hot `turn` variable during the timed measurement phase.

### 2.4 Barrier-free start synchronization

Thread creation time is excluded from the timed region by a two-phase barrier:

1. Each worker thread atomically increments `ready_threads` with release
   ordering once it has applied its scheduler hints.
2. The main thread spins with acquire loads until `ready_threads == 2`, then
   stores `start = true` with release ordering.
3. Worker threads spin on `start` with acquire loads before entering the
   handoff loop.

This busy-wait approach avoids `pthread_barrier_wait()` and the associated
kernel-mode transitions, which would introduce timing noise and potentially
cause the scheduler to park one thread before the benchmark begins.

### 2.5 Scheduler hints — not hard pinning

macOS user-space does not expose a `sched_setaffinity(2)` equivalent. The
`THREAD_AFFINITY_POLICY` Mach call accepts an integer *tag*; the scheduler
groups threads with the same tag onto the same physical core cluster when
possible, but this is advisory only. The JSON output explicitly records:

```json
"hard_pinning_supported": false,
"affinity_tags_are_hints": true
```

Empirical results (see Section 6) show that the affinity hints have **no
measurable effect** on the round-trip latency on current Apple M-series
hardware, and the Mach call returns error code 46
(`KERN_NOT_SUPPORTED` / `THREAD_AFFINITY_POLICY` not honoured) on those
systems. The hint status is recorded per-thread and printed in the report so
the user can observe this directly.

### 2.6 QoS elevation

Both worker threads call `pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)`
before signalling readiness. This elevates thread priority to the highest
user-space QoS class, reducing the probability that the OS schedules other
work on the same core during the measurement window.

---

## 3. Key Components and Modules

| File | Role |
|---|---|
| `src/benchmark/core_to_core_latency.h` | Public API: config structs, function declarations |
| `src/benchmark/core_to_core_latency_runner.cpp` | Scenario execution, statistics, console output |
| `src/benchmark/core_to_core_latency_cli.cpp` | CLI argument parsing and validation |
| `src/benchmark/core_to_core_latency_json.cpp` | JSON serialization of results |
| `src/benchmark/core_to_core_latency_json.h` | JSON serialization context struct |
| `src/asm/core_to_core_latency.s` | AArch64 assembly hot loops |
| `src/asm/asm_functions.h` | `extern "C"` declarations for assembly functions |
| `src/output/console/messages/core_to_core_messages.cpp` | User-facing string constants |
| `tests/test_core_to_core_cli.cpp` | Unit tests: CLI parsing |
| `tests/test_core_to_core_messages.cpp` | Unit tests: message content contracts |

### 3.1 Data structures

**`CoreToCoreLatencyConfig`** — parsed from CLI, drives the benchmark run:

| Field | Type | Default | Description |
|---|---|---|---|
| `loop_count` | `int` | 1 | Number of full scenario passes |
| `latency_sample_count` | `int` | 1000 | Sampled windows per loop |
| `output_file` | `std::string` | `""` | JSON output path (empty = no file) |
| `help_requested` | `bool` | `false` | Short-circuit flag for `-h`/`--help` |

**`CoreToCoreLatencyScenarioResult`** — collects all measurements for one scenario:

| Field | Type | Description |
|---|---|---|
| `scenario_name` | `std::string` | One of `no_affinity_hint`, `same_affinity_tag`, `different_affinity_tags` |
| `loop_round_trip_ns` | `vector<double>` | Per-loop headline latency (one value per `loop_count`) |
| `sample_round_trip_ns` | `vector<double>` | All per-window samples (size = `loop_count * latency_sample_count`) |
| `initiator_hint` / `responder_hint` | `ThreadHintStatus` | QoS and affinity application status from loop 0 |

**`ThreadHintStatus`** — documents what the OS actually accepted:

| Field | Meaning |
|---|---|
| `qos_applied` | `pthread_set_qos_class_self_np` returned `KERN_SUCCESS` |
| `qos_code` | Raw return code for diagnostics |
| `affinity_requested` | Whether the scenario requested an affinity tag |
| `affinity_applied` | Whether `thread_policy_set` returned `KERN_SUCCESS` |
| `affinity_code` | Raw Mach return code (46 = not supported on this system) |
| `affinity_tag` | The tag value that was requested |

---

## 4. Implementation Details

### 4.1 Measurement phases per scenario execution

Each call to `execute_single_scenario()` performs three sequential phases
inside the initiator thread:

| Phase | Round trips | Timed? | Purpose |
|---|---|---|---|
| Warmup | 20,000 | No | Bring the cache line to steady state and allow CPU frequency scaling to settle |
| Headline | 1,000,000 | Yes | Single timing window; yields the per-loop `round_trip_ns` value |
| Sampling | `latency_sample_count × 2,000` | Yes (per window) | Fine-grained distribution; each window is timed independently |

The warmup phase is important on Apple M-series because the first few hundred
round trips may see elevated latency as the cache line migrates from its
initial allocation core and the branch predictor learns the spin pattern.

### 4.2 Statistics computation

Two statistics sets are computed:

1. **Loop-level** (`loop_round_trip_ns`): one headline latency value per loop
   iteration. For `loop_count > 1` this captures run-to-run variance across
   independent executions of the 1M-trip timed window.

2. **Sample-level** (`sample_round_trip_ns`): one value per 2,000-trip sampling
   window. With default settings (`loop_count=1`, `latency_sample_count=1000`)
   this yields 1,000 samples, each averaging 2,000 round trips.

Both sets are summarized with: average, min, max, median (P50), P90, P95, P99,
and sample standard deviation (Bessel-corrected, N-1 denominator).

Percentile interpolation uses linear interpolation between adjacent sorted
values:

```
idx   = p * (n - 1)
lower = floor(idx)
upper = lower + 1
result = sorted[lower] * (1 - frac) + sorted[upper] * frac
```

### 4.3 Scenario execution order

Scenarios are always run in the same fixed order per loop:

1. `no_affinity_hint` — no affinity tag; QoS elevated
2. `same_affinity_tag` — both threads tag=1; QoS elevated
3. `different_affinity_tags` — initiator tag=1, responder tag=2; QoS elevated

All scenarios in loop N complete before loop N+1 begins. Results accumulate
across loops: `loop_round_trip_ns` grows by one element per loop, and
`sample_round_trip_ns` grows by `latency_sample_count` elements per loop.

Thread hint status is captured from loop 0 only. This is sufficient because
the Mach call result is deterministic across identical calls on the same
kernel version.

### 4.4 Timing infrastructure

Timing uses `HighResTimer`, the project's Mach timebase timer wrapper
(`src/core/timing/timer.h`). It queries `mach_timebase_info` once at
construction and applies the numer/denom conversion factor to convert
`mach_absolute_time()` ticks to nanoseconds. The timer is created via
`HighResTimer::create()` which returns `std::optional<HighResTimer>`; failure
to obtain the timebase is treated as a fatal error.

---

## 5. Usage

### 5.1 Command-line interface

```
memory_benchmark -analyze-core2core [options]

Options:
  -count <n>             Number of benchmark loops (default: 1)
  -latency-samples <n>   Sampled windows per loop (default: 1000)
  -output <file>         Write JSON results to <file>
  -h, --help             Print usage and exit
```

All options are optional. Only `-analyze-core2core` is required. Any standard
benchmark flag (e.g., `-buffersize`) causes immediate failure with an error
message listing the permitted options.

### 5.2 Typical invocations

**Quick single run with JSON output:**
```sh
./memory_benchmark -analyze-core2core -output results.json
```

**Extended run for stable statistics:**
```sh
./memory_benchmark -analyze-core2core -count 10 -latency-samples 5000 -output results.json
```

**Verbose mode for manual inspection (no JSON):**
```sh
./memory_benchmark -analyze-core2core -count 3
```

### 5.3 Example console output

```
membenchmark 0.53.7

Running standalone core-to-core latency analysis...
  [Loop 1/1] Scenario: no_affinity_hint
  [Loop 1/1] Scenario: same_affinity_tag
  [Loop 1/1] Scenario: different_affinity_tags

--- Core-to-Core Cache-Line Handoff Report ---
Scheduler note: macOS user-space cannot hard-pin threads to exact core IDs; QoS/affinity are best-effort hints.
CPU: Apple M4
Detected Cores: 4 P, 6 E
Config: loops=1, samples/loop=1000, headline_round_trips=1000000, sample_window_round_trips=2000

[Scenario] no_affinity_hint
  Round-trip latency: 66.01 ns
  One-way estimate: 33.01 ns
  Samples: 1000
    Median (P50): 58.37 ns
    P90: 58.54 ns
    P95: 58.58 ns
    P99: 63.77 ns
    Stddev: 0.77 ns
    Min: 58.35 ns
    Max: 90.79 ns
  Initiator hints: qos=ok, affinity=not requested
  Responder hints: qos=ok, affinity=not requested

[Scenario] same_affinity_tag
  ...

[Scenario] different_affinity_tags
  ...
```

### 5.4 JSON output schema

```json
{
  "configuration": {
    "mode": "analyze_core2core",
    "cpu_name": "Apple M4",
    "performance_cores": 4,
    "efficiency_cores": 6,
    "loop_count": 1,
    "latency_sample_count": 1000,
    "warmup_round_trips": 20000,
    "headline_round_trips": 1000000,
    "sample_window_round_trips": 2000
  },
  "execution_time_sec": 43.45,
  "core_to_core_latency": {
    "scenarios": [
      {
        "name": "no_affinity_hint",
        "round_trip_ns": {
          "values": [ ... ],
          "statistics": { "average": ..., "min": ..., "max": ..., "median": ...,
                          "p90": ..., "p95": ..., "p99": ..., "stddev": ... }
        },
        "one_way_estimate_ns": { "values": [ ... ], "statistics": { ... } },
        "samples_ns": { "values": [ ... ], "statistics": { ... } },
        "thread_hints": {
          "initiator": { "qos_applied": true, "qos_code": 0,
                         "affinity_requested": false, "affinity_applied": false,
                         "affinity_code": 0, "affinity_tag": 0 },
          "responder": { ... }
        }
      }
    ],
    "hard_pinning_supported": false,
    "affinity_tags_are_hints": true
  },
  "timestamp": "2026-03-16T10:00:00Z",
  "version": "0.53.7"
}
```

The `statistics` sub-object is only emitted when there is more than one value
to summarize (i.e., omitted when `loop_count == 1` for `round_trip_ns` and
`one_way_estimate_ns`).

---

## 6. Performance Characteristics

### 6.1 Apple M4 — Mac Mini (observed results)

The following data is from `results/old/macminim4_core_to_core_v0_53_6.json`
(50 loops, 10,000 samples/loop) and
`results/0.53.7/MacMiniM4_core2core.json` (20 loops, 5,000 samples/loop).

**Round-trip latency by scenario (headline average, ns):**

| Scenario | Mean | Median | Min | P99 | Stddev |
|---|---|---|---|---|---|
| `no_affinity_hint` | ~65–68 ns | ~65–66 ns | ~58.5 ns | ~83–87 ns | ~5.4–6.0 ns |
| `same_affinity_tag` | ~64–66 ns | ~65 ns | ~58.4 ns | ~80–84 ns | ~4.8–5.5 ns |
| `different_affinity_tags` | ~64–66 ns | ~65 ns | ~58.4 ns | ~80–83 ns | ~4.8–5.3 ns |

**One-way latency estimate (half round-trip):**

| Scenario | Mean | Median | Min |
|---|---|---|---|
| All three | ~32–34 ns | ~32–33 ns | ~29.2 ns |

The minimum observable round-trip (~58.4 ns, ~29.2 ns one-way) represents the
best-case path: both threads resident on P-cores, no scheduler preemption, and
the cache line travelling over the ring interconnect without contention. The
higher tail values (P99 ~83–87 ns) correspond to loops where the OS scheduler
rescheduled one thread — visible as a spike roughly 1.5–1.7× the median.

**Key observation:** The three scenarios show nearly identical distributions.
The affinity tag hints (`same_affinity_tag`, `different_affinity_tags`) return
Mach error code 46 (`KERN_NOT_SUPPORTED`) on Apple M4, confirming that
`THREAD_AFFINITY_POLICY` is not honoured on current Apple Silicon. The
scheduler appears to place both threads on P-cores based on the
`QOS_CLASS_USER_INTERACTIVE` hint alone.

### 6.2 Interpretation guide

| Observation | Likely cause |
|---|---|
| Median ~58–66 ns consistently | Two P-cores communicating over the Apple M ring bus |
| P99 1.5–2× median | Scheduler preemption; one thread was descheduled |
| First-loop round-trip notably higher than subsequent loops | CPU frequency scaling settling; always present in loop 0 |
| `affinity_applied: false`, code 46 | `THREAD_AFFINITY_POLICY` not supported on this Apple Silicon kernel version |
| All three scenarios statistically equivalent | Affinity hints have no effect; QoS drives placement |

### 6.3 Execution time

At default settings (1 loop, 1,000 samples), execution is approximately
1–2 seconds. At 50 loops and 10,000 samples (the extended run in the archived
result), total execution time is ~207 seconds. The benchmark is single-run
blocking — no background threads other than the two worker threads exist during
the timed phases.

---

## 7. Testing Strategy and Coverage

### 7.1 Test suites

Three dedicated GoogleTest suites cover this module:

| Suite | File | Tests | Focus |
|---|---|---|---|
| `CoreToCoreCliTest` | `tests/test_core_to_core_cli.cpp` | 5 | CLI parsing, validation, flag rejection |
| `CoreToCoreMessagesTest` | `tests/test_core_to_core_messages.cpp` | 3 | Message content contracts |
| `CoreToCoreRunnerTest` | `tests/test_core_to_core_runner.cpp` | 2 | Measurement loop correctness and thread-hint propagation |

### 7.2 CLI test coverage (`CoreToCoreCliTest`)

| Test | Behaviour verified |
|---|---|
| `ParsesDefaultStandaloneModeValues` | Bare `-analyze-core2core` produces default `loop_count` and `latency_sample_count` from centralized constants |
| `ParsesOptionalModeArguments` | `-count`, `-latency-samples`, `-output` are parsed and stored correctly |
| `RejectsUnknownOptionsInStandaloneMode` | Standard benchmark flag `-buffersize` returns `EXIT_FAILURE` |
| `RejectsInvalidCountValues` | `-count 0` returns `EXIT_FAILURE` (must be ≥ 1) |
| `HelpFlagReturnsSuccessAndSetsHelpRequested` | `-h` returns `EXIT_SUCCESS` and sets `config.help_requested = true` |

### 7.3 Messages test coverage (`CoreToCoreMessagesTest`)

| Test | Behaviour verified |
|---|---|
| `UsageMentionsStandaloneMode` | The usage string produced by `Messages::usage_options()` contains `-analyze-core2core` |
| `StandaloneModeErrorMessageExists` | The isolation error message names all three permitted options: `-output`, `-count`, `-latency-samples` |
| `HintStatusMessageContainsRoleAndCodes` | `report_core_to_core_hint_status()` embeds the thread role name, QoS failure code, affinity tag, and affinity failure code |

### 7.4 Runner test coverage (`CoreToCoreRunnerTest`)

| Test | Behaviour verified |
|---|---|
| `ExecuteSingleScenarioProducesHeadlineAndSamples` | `execute_single_scenario` returns `true`; `round_trip_ns` is positive, finite, and non-NaN; `samples_ns` contains exactly the requested number of entries (1 in this test) and each entry is positive |
| `ExecuteSingleScenarioSupportsZeroSamples` | Passing `latency_sample_count = 0` returns `true` with a positive `round_trip_ns` and an empty `samples_ns` vector; `initiator_hint.affinity_requested`, `responder_hint.affinity_requested`, `initiator_hint.affinity_tag`, and `responder_hint.affinity_tag` all match the values in the `ScenarioDescriptor` that was passed in |

The `ExecuteSingleScenarioProducesHeadlineAndSamples` test uses the
`no_affinity_hint` scenario descriptor (`CORE_TO_CORE_AFFINITY_HINT_DISABLED`,
both tags `CORE_TO_CORE_AFFINITY_TAG_NONE`). The
`ExecuteSingleScenarioSupportsZeroSamples` test uses the
`different_affinity_tags` scenario descriptor (`CORE_TO_CORE_AFFINITY_HINT_ENABLED`,
initiator tag `CORE_TO_CORE_AFFINITY_TAG_PRIMARY`, responder tag
`CORE_TO_CORE_AFFINITY_TAG_SECONDARY`), which exercises the hint-propagation
path even when sampling is disabled.

Both tests exercise the internal `execute_single_scenario` function via the
`core_to_core_latency_internal.h` header, which is the only
translation-unit-private interface exposed exclusively for unit testing.

### 7.5 Additional unit-test coverage

The JSON serialization path (`save_core_to_core_latency_to_json`) is covered by
`tests/test_json_schema.cpp` with mode/schema assertions, one-way estimate
value checks, thread-hint field serialization checks, single-value statistics
omission checks, and an invalid-path failure case.

The statistics computation (`calculate_summary_stats`) is a file-scope static
function private to the runner translation unit. It is validated indirectly
through the printed console output rather than through a dedicated unit test.
The analogous `calculate_statistics()` in `statistics.cpp` has its own
22-test suite (`StatisticsTest`) which provides independent confidence that
the percentile algorithm is correct.

---

## 8. Known Limitations and Caveats

1. **macOS-only**: The implementation uses `mach/thread_policy.h`,
   `pthread/qos.h`, and `mach_absolute_time()` — APIs that are specific to
   Apple's XNU kernel. The code will not compile on Linux or other POSIX
   systems without a porting layer.

2. **No hard core pinning**: On current Apple Silicon under macOS user-space,
   it is not possible to bind a thread to a specific core index. The benchmark
   measures what the scheduler arranges, not a controlled P-P or P-E pair.
   Results therefore reflect average-case scheduler placement, not a
   deterministic hardware topology.

3. **Heterogeneous core latency is not isolated**: If the scheduler places one
   thread on a P-core and one on an E-core, the measured latency will be
   significantly higher. This event is statistically rare at
   `QOS_CLASS_USER_INTERACTIVE` but is visible in P99 tails. The benchmark
   does not currently detect or annotate which core type each thread ran on.

4. **One-way estimate assumes symmetry**: `one_way_ns = round_trip_ns / 2` is
   an approximation. On a heterogeneous topology (P+E) or under asymmetric
   scheduler load, the true initiator→responder and responder→initiator
   latencies may differ.

5. **No NUMA topology awareness**: The benchmark does not query cluster
   membership or NUMA node distance. On a multi-die Apple Silicon system
   (e.g., M1 Ultra with two M1 Max dies) the measured latency could reflect
   either on-die or cross-die communication depending on scheduler placement.

6. **Statistics are round-trip only**: The per-window sample (2,000 round
   trips) averages over many handoffs. Instantaneous single-handoff latency
   is not directly observable through this interface.

---

## Appendix A — Constant Reference

| Constant | Value | Description |
|---|---|---|
| `CORE_TO_CORE_WARMUP_ROUND_TRIPS` | 20,000 | Untimed warmup before each headline window |
| `CORE_TO_CORE_HEADLINE_ROUND_TRIPS` | 1,000,000 | Trips per timed headline window |
| `CORE_TO_CORE_SAMPLE_WINDOW_ROUND_TRIPS` | 2,000 | Trips per sampled distribution window |
| `CORE_TO_CORE_DEFAULT_LOOP_COUNT` | 1 | Default `-count` value |
| `CORE_TO_CORE_DEFAULT_LATENCY_SAMPLE_COUNT` | 1,000 | Default `-latency-samples` value |
| `CORE_TO_CORE_INITIATOR_TURN_VALUE` | 0 | Token value for initiator ownership |
| `CORE_TO_CORE_RESPONDER_TURN_VALUE` | 1 | Token value for responder ownership |
| `CORE_TO_CORE_AFFINITY_TAG_NONE` | 0 | No affinity tag |
| `CORE_TO_CORE_AFFINITY_TAG_PRIMARY` | 1 | Affinity tag shared by both threads in `same_affinity_tag` |
| `CORE_TO_CORE_AFFINITY_TAG_SECONDARY` | 2 | Affinity tag for responder in `different_affinity_tags` |
| `CORE_TO_CORE_JSON_MODE_NAME` | `"analyze_core2core"` | JSON mode identifier |
| `CORE_TO_CORE_JSON_HARD_PINNING_SUPPORTED` | `false` | Emitted as JSON metadata |
| `CORE_TO_CORE_JSON_AFFINITY_TAGS_ARE_HINTS` | `true` | Emitted as JSON metadata |

---

## Appendix B — ARM64 Memory Ordering Notes

The `LDAR`/`STLR` instructions used in the hot loop provide the following
guarantees under the ARMv8-A memory model:

- **`LDAR`** (Load-Acquire): All memory accesses that appear after `LDAR` in
  program order observe the effects of all stores that were visible before the
  corresponding `STLR` on another thread. This is the acquire side of the
  acquire-release synchronization pair.

- **`STLR`** (Store-Release): All memory accesses that appear before `STLR` in
  program order complete before any load or store that appears after the
  corresponding `LDAR` on another thread observes the `STLR`'s written value.

Together they implement sequentially consistent atomic operations on the
flagged location without requiring an explicit `DMB SY` barrier. On Apple M
silicon this is backed by the ring bus coherence protocol, which invalidates
the remote core's L1 cache line and issues a new ownership grant before the
acquire load returns the updated value.

The clobber register `w8` (callee-saved in the AArch64 PCS when used as `x8`
for indirect result, but `w8` is generally caller-saved in practice for simple
integer temporaries) is used as the scratch register for the loaded value. The
function clobbers only `w8` and `x1` (the loop counter via `SUBS`); all other
registers are preserved.

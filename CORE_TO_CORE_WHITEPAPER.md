# Core-to-Core Cache-Line Handoff Latency Benchmark

**memory_benchmark — Technical Whitepaper**

*Revision 2026-07-11 — Applies to version 0.61.1; archived examples may use older methodologies*

---

## TL;DR

`-C` / `--analyze-core2core` measures the effective round-trip latency of a repeated ARM64 acquire/release token
protocol between two POSIX threads. The result includes the complete polling and handoff loop plus coherence and
scheduler effects; it is not an isolated measurement of one physical cache-line migration or the coherence fabric.
Software version 0.58.0 introduced the change from fixed work in a fixed scenario order to the auditable
v2 calibrated/balanced design. Version 0.61.1 retains that design, isolates token and control state in distinct
128-byte blocks, and identifies the current methodology as
`core2core-v3-calibrated-balanced-auditable-128b-isolation`:

- every scheduler-hint scenario receives its own excluded pilot after a long calibration warmup;
- final warmup, continuous headline, and separate sample windows use pilot-derived duration targets with fixed minimum
  work;
- scenario order rotates across repeated loops;
- core-to-core mode defaults to three measured loops, without changing other modes' one-loop default;
- the scenario headline is the median P50 of completed continuous loop measurements;
- CV and MAD describe repeatability, with a diagnostic warning above 7.5% headline CV;
- JSON schema 2 records the work plan, every attempted loop, completion state, and observed thread hints;
- unavailable measurements are status-bearing `null` values, never numeric latency zeroes.

The benchmark still cannot hard-pin threads to exact macOS core IDs. Its results describe a scheduler-influenced token
handoff protocol, not a guaranteed physical-core topology.

---

## 1. Measurement Scope

The shared-state object uses 128-byte alignment and isolation slots. A 32-bit turn token occupies the first slot,
while start and control flags occupy a separate slot at least 128 bytes away. The initiator waits for its token, stores
the responder token, and waits for the responder to return ownership. The responder mirrors the handoff. One round trip
ends when the initiator receives the returned token.

For one continuous headline window:

```text
round_trip_latency_ns = total_timed_ns / headline_round_trips
one_way_estimate_ns   = round_trip_latency_ns / 2
```

The one-way value is an estimate, not an independently timed direction. It assumes symmetry and can be misleading if
the scheduler places the workers on different core types or migrates them during a window.

The round-trip value is the effective cost of the complete token protocol. It includes `LDAR` polling, `STLR` stores,
comparisons, branches, loop bookkeeping, cache-coherence activity, and any scheduler influence during the timed window.
It is not a direct measurement of one physical cache-line transfer, raw load-to-use latency, or coherence-interconnect
latency.

Three scenarios exercise the same handoff kernel with different scheduler hints:

1. `no_affinity_hint`
2. `same_affinity_tag`
3. `different_affinity_tags`

Both workers request `QOS_CLASS_USER_INTERACTIVE`. The latter two scenarios additionally request Mach affinity tags.
Those tags are advisory and do not identify or pin a physical core.

---

## 2. ARM64 Handoff Kernel

The hot loop is implemented as hand-written assembly in `src/asm/core_to_core_latency.s`, so a C++ compiler cannot
remove or transform it. `LDAR` and `STLR` provide the ARM architectural acquire/release ordering used by the token
protocol:

```asm
c2c_initiator_wait_turn:
    ldar w8, [x0]
    cmp  w8, w2
    b.ne c2c_initiator_wait_turn

    stlr w3, [x0]

c2c_initiator_wait_return:
    ldar w8, [x0]
    cmp  w8, w2
    b.ne c2c_initiator_wait_return

    subs x1, x1, #1
    b.ne c2c_initiator_loop
```

The responder performs one acquire-spin and one release-store per iteration. The turn token and the start/control flags
begin in separate 128-byte-aligned isolation blocks, preventing them from occupying the same 128-byte block. This is a
conservative false-sharing boundary for current Apple Silicon targets, not a runtime cache-line-size measurement.

Timing uses `HighResTimer`, whose start/stop path applies the project's ARM64 timing fences before reading
`mach_absolute_time()`. Thread creation, scheduler-hint calls, readiness synchronization, and thread joining remain
outside every timed window.

---

## 3. Calibrated Measurement Methodology

### 3.1 Per-scenario excluded pilot

Before any measured `--count` loop, each scenario is calibrated independently. Its pilot execution contains:

| Pilot phase | Round trips | Timed? | Included in reported values? |
|---|---:|---|---|
| Calibration warmup | 1,000,000 | No | No |
| Calibration pilot | 100,000 | Yes | No |
| Sample windows | 0 | — | No |

The million-round-trip calibration warmup is intended to reduce startup and transient effects before the pilot estimates
round-trip cost; it does not prove that the system reached a steady state. Pilot latency is excluded from both the
continuous headline population and the sample-window population.

The three pilots run once in fixed scenario order before the balanced measured schedule. Each pilot uses a newly created
worker pair, and every later measured scenario execution creates another pair. Pilot affinity/QoS application outcomes
are not serialized; only the excluded pilot timing and the resolved work plan are retained.

One scenario's pilot resolves one `CoreToCoreWorkPlan`, and the same plan is reused for that scenario in every measured
loop. This avoids changing workload size in response to normal loop-to-loop noise.

### 3.2 Work-plan calculation

For each target, round trips are scaled from the pilot and rounded upward:

```text
resolved_round_trips = ceil(pilot_round_trips * target_seconds / pilot_elapsed_seconds)
resolved_round_trips = clamp(resolved_round_trips, minimum_round_trips, 100,000,000)
```

| Final phase | Duration target | Minimum round trips | Timing semantics |
|---|---:|---:|---|
| Warmup | 25 ms | 20,000 | Untimed, immediately before headline |
| Continuous headline | 250 ms | 1,000,000 | One continuous timed window per scenario/loop |
| Sample window | 1 ms | 2,000 | Each requested window timed separately |

The 20,000/1,000,000/2,000 values are minimum work, not fixed work. Slow or fast systems may retain a minimum or receive
a larger calibrated count. Calibration targets each phase's nominal duration from the excluded pilot; it neither
normalizes nor guarantees the observed duration. The 100,000,000 guardrail applies independently to the resolved final
warmup count, headline count, and per-sample-window count, not to their combined work.

The continuous headline duration is classified as:

- `within-target-window` for 100-300 ms;
- `below-target-window` below 100 ms;
- `above-target-window` above 300 ms;
- `invalid-elapsed` for a non-positive or non-finite elapsed value.

Duration quality is audit metadata. A finite positive measurement outside the intended window remains measured; the
classification tells consumers that its timing envelope was not ideal.

### 3.3 Measured execution

For one scenario in one loop:

1. create responder and initiator threads;
2. apply QoS and optional affinity hints inside each worker;
3. wait until both workers reach the readiness gate;
4. release the start gate;
5. execute the calibrated final warmup without timing;
6. time one continuous calibrated headline window;
7. time `--latency-samples` separate calibrated sample windows;
8. join both workers and validate the headline and all requested sample-window elapsed values as one record.

The responder is given the exact combined number of warmup, headline, and sample-window handoffs, with overflow checks.
The initiator and responder therefore finish the same token exchange without a partially consumed batch.

Record validity is all-or-nothing. A record is `measured` only when its headline and every requested sample window are
valid. Otherwise it contributes neither a continuous headline nor sample-window values to scenario aggregates.

### 3.4 Safe worker startup

The responder is started first. If responder creation fails, no worker needs cleanup. If initiator creation fails after
the responder is waiting, the main thread sets a cancellation flag, releases the start gate, and joins the responder.
The waiting worker observes cancellation and returns instead of remaining stranded indefinitely.

Direct core-to-core execution blocks the benchmark termination signals before creating workers and restores the previous
signal mask afterward. This keeps the signal-handling policy consistent with the rest of the benchmark and lets the main
thread observe graceful interruption between completed measurements.

---

## 4. Balanced Scenario Schedule

The old fixed order could confound a scenario with first/last-position, frequency, thermal, or background-load effects.
Software version 0.58.0 introduced rotation of the starting scenario by loop:

| Loop index | Scenario order |
|---:|---|
| 0 | no affinity hint → same tag → different tags |
| 1 | same tag → different tags → no affinity hint |
| 2 | different tags → no affinity hint → same tag |
| 3 | repeats loop 0 |

This cyclic Latin-square-style schedule gives each scenario each order position once per three loops. For counts that are
not multiples of three, the schedule remains deterministic but position counts cannot be perfectly equal.

Calibration pilots precede this measured schedule and do not enter its aggregates. JSON records `loop_index` and
`order_position` for every attempted scenario measurement.

---

## 5. Headline and Distribution Statistics

The benchmark maintains two distinct populations.

### 5.1 Continuous loop headlines

Each completed scenario/loop contributes one average round-trip value from its single continuous headline window. The
scenario headline is:

```text
median P50 across completed continuous loop headlines
```

One completed loop is its own median. Multiple completed loops additionally receive average, median, P90, P95, P99,
sample standard deviation, coefficient of variation (CV), median absolute deviation (MAD), minimum, and maximum. Values
are not filtered as outliers.

If headline CV exceeds 7.5%, the console prints a diagnostic warning. The warning does not discard values or by itself
change a measured scenario to invalid.

### 5.2 Separate sample-window-mean distribution

Every requested sample window contributes its average round-trip latency to `samples_ns`. These window means are pooled
only for a separate distribution with the same statistical fields. They do not replace, weight, or modify the continuous
loop headline, and they do not expose the distribution or tail latency of individual handoffs within a window.

Each measured loop record stores the start index and count of the sample-window means it contributed, allowing a
consumer to reconstruct loop boundaries within the pooled array. A non-measured record contributes no sample-window
values and has a serialized range count of zero.

### 5.3 One-way estimate

Every one-way estimate is exactly half of its corresponding round-trip value. The aggregate one-way distribution is
therefore derived from measured round-trip headlines; it is not an additional benchmark phase.

---

## 6. Scheduler-Hint Interpretation

macOS user space cannot guarantee exact physical-core pinning. The output therefore preserves both the requested policy
and what each API call reported:

- `qos_applied` and raw `qos_code`;
- `affinity_requested`;
- `affinity_applied` and raw `affinity_code`;
- `affinity_tag`.

Schema 2 stores these fields in every loop record, because repeated thread creation can produce different observed
outcomes. The scenario-level `thread_hints` object retains the first measured loop's values as a compact compatibility
summary; loop records are authoritative for audit.

`affinity_hint_comparison_interpretable` is true only when:

1. the command status is `complete`;
2. at least one measured record requested affinity; and
3. both workers reported successful affinity-tag API calls in every measured record that requested affinity.

Even when true, this field means only that the requested affinity API calls in measured records succeeded. It does not
incorporate QoS success, pilot hint outcomes, or observed physical placement, and it does not prove hard pinning. When
false, differences between affinity scenarios must not be presented as evidence of the requested affinity policy.

---

## 7. Completion and Failure Semantics

Measurement status values are `not-run`, `measured`, `interrupted`, `invalid`, and `failed`. A status reason accompanies
non-successful paths. Examples include timer creation failure, invalid/overflowing work, allocation failure, worker
startup failure, and invalid elapsed time.

The JSON command payload records:

- `status`: `complete`, `interrupted`, or `failed`;
- `planned_measurements`: scenarios multiplied by requested loops;
- `completed_measurements`: valid measured loops;
- `measurements_complete`: true only when the complete command realized every planned measurement.

Each scenario separately records planned/completed loops and status/reason. Aggregate value arrays include only measured
values. When no valid continuous headline exists, `headline_round_trip_ns` is `null`. Failed, invalid, interrupted, and
not-run measurements are never synthesized as numeric zeroes.

The aggregate contribution is all-or-nothing per scenario/loop record: an invalid headline or requested sample window
makes the record non-measured, leaves its sample range count at zero, and contributes neither headline nor sample
values.

When `--output` is set, direct execution writes the in-memory audit payload even after a measurement failure or graceful
interruption, provided a payload was built. Consumers should use completion fields rather than treating file existence or
process success alone as proof of a complete comparison.

---

## 8. JSON Schema 2

The mode identifier remains `analyze_core2core`. The methodology contract is identified independently. The following is
an abbreviated structural excerpt: scenario entries, loop records, sample-array members, and statistical/hint fields are
intentionally omitted. Count fields describe the corresponding unabridged payload; the displayed continuous round-trip
and one-way arrays are complete for the illustrated scenario.

```json
{
  "configuration": {
    "mode": "analyze_core2core",
    "schema_version": 2,
    "methodology_version": "core2core-v3-calibrated-balanced-auditable-128b-isolation",
    "loop_count": 3,
    "latency_sample_count": 2,
    "minimum_warmup_round_trips": 20000,
    "minimum_headline_round_trips": 1000000,
    "minimum_sample_window_round_trips": 2000,
    "calibration_round_trips": 100000,
    "calibration_warmup_round_trips": 1000000,
    "warmup_target_seconds": 0.025,
    "headline_target_seconds": 0.25,
    "headline_duration_window_seconds": {
      "minimum": 0.1,
      "maximum": 0.3
    },
    "sample_window_target_seconds": 0.001,
    "scenario_schedule": "cyclic-latin-square-across-count-loops",
    "headline_aggregate": "median-p50",
    "repeatability_cv_warning_pct": 7.5
  },
  "core_to_core_latency": {
    "status": "complete",
    "planned_measurements": 9,
    "completed_measurements": 9,
    "measurements_complete": true,
    "scenarios": [
      {
        "name": "no_affinity_hint",
        "status": "measured",
        "status_reason": null,
        "planned_loops": 3,
        "completed_loops": 3,
        "work_plan": {
          "automatic_calibration": true,
          "calibration_excluded_from_results": true,
          "calibration_round_trips": 100000,
          "calibration_elapsed_seconds": 0.01,
          "calibration_round_trip_ns": 100.0,
          "warmup_round_trips": 250000,
          "headline_round_trips": 2500000,
          "sample_window_round_trips": 10000
        },
        "round_trip_ns": {
          "values": [99.0, 100.0, 101.0],
          "statistics": {
            "median": 100.0,
            "coefficient_of_variation_pct": 1.0,
            "median_absolute_deviation": 1.0
          }
        },
        "headline_round_trip_ns": 100.0,
        "headline_statistic": "median-p50-across-completed-continuous-loops",
        "one_way_estimate_ns": {
          "values": [49.5, 50.0, 50.5]
        },
        "samples_ns": {
          "values": [100.2, 99.8],
          "statistics": {"median": 100.0}
        },
        "loop_records": [
          {
            "loop_index": 0,
            "order_position": 0,
            "status": "measured",
            "status_reason": null,
            "round_trip_ns": 99.0,
            "one_way_estimate_ns": 49.5,
            "headline_elapsed_seconds": 0.2475,
            "duration_quality": "within-target-window",
            "sample_window_range": {"start_index": 0, "count": 2},
            "thread_hints": {
              "initiator": {"qos_applied": true, "affinity_requested": false},
              "responder": {"qos_applied": true, "affinity_requested": false}
            }
          }
        ]
      }
    ],
    "hard_pinning_supported": false,
    "affinity_tags_are_hints": true,
    "affinity_hint_comparison_interpretable": false
  },
  "execution_time_sec": 4.2,
  "timestamp": "2026-07-10T12:00:00Z",
  "version": "0.61.1"
}
```

The numbers are structural examples, not recorded performance results. Only the first of three scenario entries and its
first loop record are shown, and the pooled sample array is truncated. Statistical and hint sub-objects are abbreviated
as well. Actual `thread_hints` objects also contain raw QoS/affinity codes, the `*_applied` booleans, and the affinity tag.
Statistics contain the full common statistics set when emitted.

### 8.1 Work plan

`work_plan` makes calibration auditable per scenario. `calibration_elapsed_seconds` and
`calibration_round_trip_ns` describe the excluded pilot; the three resolved round-trip counts describe actual final
work. A consumer should not assume all scenarios received equal work counts.

### 8.2 Loop records

`loop_records` are the authoritative link between scenario schedule, measurement status, duration quality, pooled sample
boundaries, and observed worker hints. A measured record's range identifies all sample-window means it contributed; a
non-measured record contributes none and has range count zero. Nullable values distinguish unavailable results from real
measured latency.

### 8.3 Completion metadata

For comparisons, require `measurements_complete: true`. For affinity-scenario interpretation, additionally require
`affinity_hint_comparison_interpretable: true` and retain the caveat that no physical core IDs are controlled.

---

## 9. CLI and Sweep Behavior

```text
memory_benchmark -C [options]
memory_benchmark --analyze-core2core [options]

Options:
  -r, --count <n>             Measured loop count (core-to-core default: 3)
  -n, --latency-samples <n>   Separate sample windows per scenario/loop (default: 1000)
  -o, --output <file>         JSON output path
  -S, --sweep <key=a,b>       Sweep count or latency-samples
  -X, --sweep-max-runs <n>    Generated-run guard (default: 256)
  -h, --help                  Print help
```

Standard benchmark options are rejected in this standalone mode. Sweep mode requires `--output`, supports only `count`
and `latency-samples`, and forms their Cartesian product. Each sweep parameter key may appear only once; for example, two
`--sweep count=...` specifications are rejected before execution.

The general benchmark loop default remains one; only core-to-core mode defaults to three. At the default 1,000 sample
windows per scenario and loop, the 1 ms target represents about 9 seconds of sampling across three scenarios and three
loops. Continuous headlines add about 2.25 seconds, and final warmups, per-scenario calibration, thread startup, and
scheduler effects add further time. Bare `-C` therefore intentionally runs materially longer than the old single-loop
default in exchange for an immediate loop median and CV/MAD repeatability evidence.

Examples:

```bash
memory_benchmark -C --count 5 --output core2core.json
memory_benchmark --analyze-core2core --count 3 --latency-samples 2000 --output core2core_deep.json
memory_benchmark -C --count 3 --sweep latency-samples=500,1000,2000 --output core2core_sweep.json
```

Core-to-core sweep output is written through the atomic temporary-file-and-rename path after every attempted run. Its
envelope records `status`, `status_reason`, `planned_runs`, `attempted_runs`, `completed_runs`, and
`conclusions_valid`. Every run entry has its own `status` and `status_reason`. Only entries whose nested
`core_to_core_latency.status` is `complete` and `measurements_complete` is `true` increment `completed_runs`; partial,
interrupted, and failed entries remain auditable but do not. An interruption or later failure therefore does not erase
previously checkpointed runs. `conclusions_valid` is true only when top-level status is `complete` and
`completed_runs == planned_runs`.

---

## 10. Comparability With Older Results

Archived 0.53.x/0.57.x core-to-core files used fixed 20,000/1,000,000/2,000 work and fixed scenario order. Versions
0.58.0 through 0.61.0 used the calibrated/balanced v2 methodology with 64-byte token/control separation. Version 0.61.1
advances the methodology identity to v3 for the corrected 128-byte isolation boundary while retaining JSON schema 2.
The current methodology also uses duration-targeted work, a balanced schedule, a loop-median headline, and explicit
completion semantics.

JSON schema version and measurement methodology are separate contracts: schema 2 serializes the audit data but does not
itself calibrate work, identify the isolation layout, or define the measurement procedure. Use `methodology_version`,
not schema version alone, to decide comparability, and do not merge populations with different methodology identifiers
without explicitly labeling and justifying that difference.

Older result files remain useful as historical observations. Their hardware-specific values should not be read as
validation of current affinity behavior, current scheduler placement, or current-methodology duration quality.

---

## 11. Testing and Verification

Relevant deterministic coverage includes:

- calibration scaling, rounding, validation, and min/max clamping;
- work-plan targets derived from an excluded pilot;
- scenario-order rotation;
- CLI short/long options, supported sweeps, duplicate sweep-key rejection, and run-count guards;
- calibrated console/audit messages;
- schema-2 work plans, loop records, completion fields, nullable values, and affinity interpretability.

The runner also has Apple Silicon integration coverage for a headline plus sample window and for a zero-sample internal
execution path. Those tests exercise the real two-thread ARM64 handoff and are hardware/runtime-sensitive.

For manual validation, prefer several loops and inspect:

- all three scenario work plans;
- `duration_quality` for every completed headline;
- headline CV/MAD rather than only one median;
- per-loop hint outcomes;
- command/sweep completion metadata;
- affinity interpretability before comparing hint scenarios.

---

## 12. Limitations

1. **No hard core pinning:** macOS may place or migrate either worker; exact P-P, P-E, E-E, cluster, or die paths are not
   selected by this tool.
2. **Affinity tags are advisory:** an applied call is not proof of a physical placement, and a rejected call makes the
   affinity-scenario comparison uninterpretable.
3. **One-way latency is derived:** half round trip assumes symmetric directions.
4. **Sample windows average many handoffs:** their statistics describe window means, not instantaneous single-handoff
   values or fine-grained handoff tails.
5. **System state matters:** thermal state, frequency policy, background load, and scheduler activity can move both
   headline and sample-window means. Calibration targets a duration but does not normalize or guarantee it, and it does
   not normalize environmental noise.
6. **No topology annotation:** the mode does not record the actual core type, cluster, or die used during a window.
7. **macOS/Apple Silicon implementation:** Mach thread policy, pthread QoS, Mach timing, and ARM64 assembly make the
   current implementation platform-specific.

---

## Appendix A — Constants

| Constant | Value | Meaning |
|---|---:|---|
| `CORE_TO_CORE_SHARED_STATE_ISOLATION_BYTES` | 128 | Token/control alignment and minimum slot separation |
| `CORE_TO_CORE_DEFAULT_LOOP_COUNT` | 3 | Core-to-core-only default measured loop count |
| `CORE_TO_CORE_DEFAULT_LATENCY_SAMPLE_COUNT` | 1,000 | Default sample windows per scenario/loop |
| `CORE_TO_CORE_CALIBRATION_WARMUP_ROUND_TRIPS` | 1,000,000 | Excluded warmup before each scenario pilot |
| `CORE_TO_CORE_CALIBRATION_ROUND_TRIPS` | 100,000 | Excluded timed pilot work |
| `CORE_TO_CORE_WARMUP_TARGET_SECONDS` | 0.025 | Final warmup duration target |
| `CORE_TO_CORE_HEADLINE_TARGET_SECONDS` | 0.250 | Continuous headline duration target |
| `CORE_TO_CORE_HEADLINE_MIN_SECONDS` | 0.100 | Intended headline lower bound |
| `CORE_TO_CORE_HEADLINE_MAX_SECONDS` | 0.300 | Intended headline upper bound |
| `CORE_TO_CORE_SAMPLE_TARGET_SECONDS` | 0.001 | Separate sample-window target |
| `CORE_TO_CORE_WARMUP_ROUND_TRIPS` | 20,000 | Final warmup minimum |
| `CORE_TO_CORE_HEADLINE_ROUND_TRIPS` | 1,000,000 | Continuous headline minimum |
| `CORE_TO_CORE_SAMPLE_WINDOW_ROUND_TRIPS` | 2,000 | Sample-window minimum |
| `CORE_TO_CORE_MAX_ROUND_TRIPS` | 100,000,000 | Independent cap for each resolved phase count |
| `CORE_TO_CORE_CV_WARNING_PCT` | 7.5 | Headline repeatability warning threshold |
| `CORE_TO_CORE_JSON_SCHEMA_VERSION` | 2 | Core-to-core schema version |
| `CORE_TO_CORE_METHODOLOGY_VERSION` | `core2core-v3-calibrated-balanced-auditable-128b-isolation` | Methodology identifier |

---

## Appendix B — Source Map

| File | Role |
|---|---|
| `src/benchmark/core_to_core_latency.h` | Public configuration, status, work-plan, loop-record, and result types |
| `src/benchmark/core_to_core_latency_internal.h` | Planner/scheduler/runner testable interface |
| `src/benchmark/core_to_core_latency_runner.cpp` | Calibration, worker execution, schedule, statistics, and console report |
| `src/benchmark/core_to_core_latency_cli.cpp` | Standalone parsing, sweep validation, and direct signal-mask setup |
| `src/benchmark/core_to_core_latency_json.cpp` | Schema 2 serialization and affinity interpretability |
| `src/benchmark/core_to_core_sweep_runner.cpp` | Cartesian sweep execution and atomic checkpoints |
| `src/core/config/constants.h` | Shared-state isolation and core-to-core methodology constants |
| `src/core/config/sweep_utils.cpp` | Shared sweep-value parsing and Cartesian run-count validation |
| `src/asm/core_to_core_latency.s` | ARM64 initiator/responder hot loops |
| `src/output/console/messages/core_to_core_messages.cpp` | User-facing core-to-core messages |
| `src/output/json/json_output/file_writer.cpp` | Shared atomic temporary-file-and-rename writer |
| `tests/test_core_to_core_cli.cpp` | CLI and sweep validation coverage |
| `tests/test_core_to_core_messages.cpp` | Console message contract coverage |
| `tests/test_core_to_core_runner.cpp` | Calibration, schedule, and hardware integration coverage |
| `tests/test_json_schema.cpp` | Core-to-core schema and audit-trail coverage |

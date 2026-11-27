#!/usr/bin/env python3
"""Shared strict JSON, RNG, Mach-timing, statistics, audit, and sweep primitives
for independent CPU bandwidth and CPU latency artifact verifiers.

Zero third-party dependencies (Python 3 standard library only).
"""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sys

U64 = (1 << 64) - 1
U32 = (1 << 32) - 1
MAX_SAFE_INT = (1 << 53) - 1
MAX_INPUT = 64 * 1024 * 1024
DEFAULT_MAX_WORK = 1_000_000

SPLITMIX_A = 0x9E3779B97F4A7C15
SPLITMIX_B = 0xBF58476D1CE4E5B9
SPLITMIX_C = 0x94D049BB133111EB

FOLD_INIT_A = 0x6A09E667F3BCC909
FOLD_INIT_B = 0xBB67AE8584CAA73B

VALID_RUN_STATUSES = (
    "complete",
    "partial",
    "interrupted",
    "failed",
    "unsupported",
    "not_started",
)
VALID_MEASUREMENT_STATUSES = (
    "measured",
    "invalid",
    "failed",
    "interrupted",
    "not_run",
)
VALID_THERMAL_STATES = ("nominal", "fair", "serious", "critical", "unavailable")

_DECIMAL_UINT_RE = re.compile(r"0|[1-9][0-9]*")
_HEX64_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40,64}")


class Rejected(Exception):
    """Typed verification failure mapped to verifier exit codes 1 or 2."""

    def __init__(self, reason, unsupported=False):
        super().__init__(reason)
        self.reason = reason
        self.unsupported = unsupported


def require(condition, reason):
    """Raise Rejected(reason) when condition is falsy."""
    if not condition:
        raise Rejected(reason)


def equal(actual, expected, reason):
    """Strict type-preserving recursive equality check."""
    require(type(actual) is type(expected), reason)
    if isinstance(expected, dict):
        require(actual.keys() == expected.keys(), reason)
        for key in expected:
            equal(actual[key], expected[key], reason)
    elif isinstance(expected, (list, tuple)):
        require(len(actual) == len(expected), reason)
        for a, b in zip(actual, expected):
            equal(a, b, reason)
    else:
        require(actual == expected, reason)


def integer(value, decimal=False, maximum=U64):
    """Validate an unsigned integer encoded either natively or as a decimal string."""
    if decimal:
        require(
            isinstance(value, str)
            and len(value) <= 20
            and _DECIMAL_UINT_RE.fullmatch(value) is not None,
            "integer-encoding",
        )
        number = int(value)
    else:
        require(type(value) is int, "integer-encoding")
        number = value
    require(0 <= number <= maximum, "integer-range")
    return number


def close(actual, expected, reason="rate-mismatch", rel_tol=1e-10, abs_tol=1e-15):
    """Validate finite numeric value within strict relative/absolute tolerance."""
    require(
        type(actual) in (int, float)
        and math.isfinite(actual)
        and math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol),
        reason,
    )


class Budget:
    """Shared bounded-work oracle budget across single runs and sweeps."""

    def __init__(self, limit=DEFAULT_MAX_WORK):
        self.limit = limit
        self.remaining = limit

    def __call__(self, amount):
        self.remaining -= amount
        if self.remaining < 0:
            raise Rejected("unsupported-work-limit", True)

    @property
    def used(self):
        return self.limit - self.remaining


def load_strict_json(path):
    """Load untrusted JSON enforcing 64 MiB cap, UTF-8, no duplicate keys,
    no non-finite numbers, depth <= 64, and node count <= 1,000,000.
    """
    if str(path) == "-":
        data = sys.stdin.buffer.read(MAX_INPUT + 1)
    else:
        with open(path, "rb") as f:
            data = f.read(MAX_INPUT + 1)
    if len(data) > MAX_INPUT:
        raise Rejected("unsupported-input-limit", True)

    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, "duplicate-json-key")
            result[k] = v
        return result

    def reject_constant(_token):
        raise Rejected("nonfinite-json-number")

    doc = json.loads(
        data.decode("utf-8"),
        object_pairs_hook=pairs,
        parse_constant=reject_constant,
    )
    stack = [(doc, 0)]
    count = 0
    while stack:
        obj, depth = stack.pop()
        count += 1
        if depth > 64 or count > 1_000_000:
            raise Rejected("unsupported-structure-limit", True)
        if isinstance(obj, dict):
            stack.extend((v, depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            stack.extend((v, depth + 1) for v in obj)
    return doc


def rotl64(value, shift):
    """64-bit left circular rotation."""
    value &= U64
    shift &= 63
    if shift == 0:
        return value
    return ((value << shift) | (value >> (64 - shift))) & U64


def splitmix64_mix(x):
    """Stateless SplitMix64 mixing function (exact match to C++ & llm_verify_oracles)."""
    x = (x + SPLITMIX_A) & U64
    x = ((x ^ (x >> 30)) * SPLITMIX_B) & U64
    x = ((x ^ (x >> 27)) * SPLITMIX_C) & U64
    return x ^ (x >> 31)


def splitmix64_next(state):
    """Advance SplitMix64 state and return (draw, next_state)."""
    draw = splitmix64_mix(state)
    next_state = (state + SPLITMIX_A) & U64
    return draw, next_state


def splitmix64_bounded_draw(state, bound, budget=None):
    """Draw uniform integer in [0, bound) via SplitMix64 rejection sampling."""
    if bound <= 1:
        return 0, state
    threshold = (1 << 64) % bound
    while True:
        draw, state = splitmix64_next(state)
        if draw >= threshold:
            return draw % bound, state
        if budget is not None:
            budget(1)


def fisher_yates_rejection_shuffle(items, state, budget=None):
    """In-place descending Fisher-Yates shuffle using SplitMix64 rejection sampling
    (`splitmix64-fisher-yates-rejection-v1`), returning updated SplitMix64 state.
    """
    n = len(items)
    for bound in range(n, 1, -1):
        j, state = splitmix64_bounded_draw(state, bound, budget=budget)
        items[bound - 1], items[j] = items[j], items[bound - 1]
    return state


def partition_64b(total_bytes, workers):
    """Canonical 64-byte aligned partition across workers."""
    if workers <= 0:
        return []
    active = min(total_bytes, workers)
    if active <= 0:
        return [(0, 0)] * workers
    boundaries = [0]
    for rank in range(1, active):
        target = total_bytes // active * rank + min(rank, total_bytes % active)
        lo = boundaries[-1] + 1
        hi = total_bytes - (active - rank)
        target = max(lo, min(hi, target))
        aligned_lo = ((lo + 63) // 64) * 64
        aligned_hi = (hi // 64) * 64
        if aligned_lo <= aligned_hi:
            down = max(aligned_lo, min(aligned_hi, (target // 64) * 64))
            up = min(aligned_hi, down + 64) if down < target else down
            target = min((down, up), key=lambda x: (abs(x - target), x))
        boundaries.append(target)
    boundaries.append(total_bytes)
    return [(a, b - a) for a, b in zip(boundaries, boundaries[1:])] + [(0, 0)] * (
        workers - active
    )


def verify_cpu_raw_timing(raw, elapsed=None):
    """Validate optional Mach raw-tick snapshot and reconcile with elapsed_seconds.

    Returns "elapsed-only" when raw is None, else "cpu-raw-ticks".
    """
    if raw is None:
        return "elapsed-only"
    require(isinstance(raw, dict), "timing-raw-invalid")
    start = integer(raw.get("start_ticks"), decimal=True)
    stop = integer(raw.get("stop_ticks"), decimal=True)
    delta = integer(raw.get("delta_ticks"), decimal=True)
    numer = integer(raw.get("timebase_numer"), maximum=U32)
    denom = integer(raw.get("timebase_denom"), maximum=U32)
    require(numer > 0 and denom > 0, "timebase-invalid")
    equal(delta, (stop - start) & U64, "tick-delta-mismatch")
    reconstructed = float(delta) * float(numer) / float(denom) / 1e9
    if elapsed is not None:
        close(elapsed, reconstructed, "elapsed-mismatch")
    return "cpu-raw-ticks"


def enforce_require_raw_timing(require_raw, levels):
    """Enforce --require-raw-timing across attempted CPU measurements/calibrations."""
    if require_raw and ("elapsed-only" in levels or not levels):
        raise Rejected("unsupported-missing-raw-timing", True)


def _linear_percentile(sorted_values, percentile):
    n = len(sorted_values)
    if n == 1:
        return float(sorted_values[0])
    index = percentile * (n - 1)
    lower = int(index)
    upper = lower + 1
    if upper >= n:
        return float(sorted_values[-1])
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def compute_statistics(values):
    """Compute exact descriptive statistics matching src/utils/descriptive_statistics.cpp."""
    n = len(values)
    if n == 0:
        return None
    for v in values:
        require(type(v) in (int, float) and math.isfinite(v), "statistic-value-invalid")
    mean = sum(values) / n
    s = sorted(values)
    median = _linear_percentile(s, 0.50)
    sq_sum = 0.0
    for v in values:
        d = v - mean
        sq_sum += d * d
    sd = math.sqrt(sq_sum / (n - 1)) if n > 1 else 0.0
    cv_pct = (
        abs(sd / mean) * 100.0
        if (math.isfinite(mean) and math.isfinite(sd) and mean != 0.0)
        else None
    )
    devs = sorted(abs(v - median) for v in values)
    mad = _linear_percentile(devs, 0.50)
    return dict(
        sample_count=n,
        average=mean,
        min=s[0],
        max=s[-1],
        median=median,
        p90=_linear_percentile(s, 0.90),
        p95=_linear_percentile(s, 0.95),
        p99=_linear_percentile(s, 0.99),
        stddev=sd,
        coefficient_of_variation_pct=cv_pct,
        median_absolute_deviation=mad,
    )


def hash_binary(path):
    """Stream SHA-256 hex digest of a binary file in 64 KiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def check_manifest(manifest, binary_digest=None):
    """Validate build_manifest v1 and optional binary SHA-256 binding."""
    require(isinstance(manifest, dict), "manifest-invalid")
    equal(manifest.get("manifest_version"), 1, "manifest-version")
    status = manifest.get("status")
    require(status in ("available", "partial", "unavailable"), "manifest-status")
    bhash = manifest.get("binary_sha256")
    if bhash is not None:
        require(
            isinstance(bhash, str) and _HEX64_RE.fullmatch(bhash) is not None,
            "manifest-hash-encoding",
        )
    commit = manifest.get("git_commit")
    if commit is not None:
        require(
            isinstance(commit, str) and _COMMIT_RE.fullmatch(commit) is not None,
            "manifest-commit-encoding",
        )
    require(
        manifest.get("git_dirty") is None or type(manifest.get("git_dirty")) is bool,
        "manifest-dirty-encoding",
    )
    if status == "available":
        require(
            all(
                manifest.get(k) is not None
                for k in (
                    "binary_sha256",
                    "git_commit",
                    "git_dirty",
                    "compiler",
                    "compile_flags",
                    "link_flags",
                    "target_arch",
                    "sdk",
                    "min_os",
                )
            ),
            "manifest-availability",
        )
    build = "manifest-only" if status != "unavailable" else "unavailable"
    if binary_digest is not None:
        require(bhash is not None, "binary-hash-unavailable")
        equal(binary_digest, bhash, "binary-hash-mismatch")
        build = "binary-bound"
    return build


def check_checkpoint_lifecycle(life, planned_loops, run_status):
    """Verify bounded-loop-snapshots checkpoint lifecycle invariants."""
    require(isinstance(life, dict), "checkpoint-invalid")
    equal(
        life.get("snapshot_interval_loops"),
        max(1, (planned_loops + 7) // 8),
        "checkpoint-policy",
    )
    equal(life.get("checkpoint_policy"), "bounded-loop-snapshots", "checkpoint-policy")
    equal(
        life.get("observation_point"),
        "before-current-snapshot-preparation",
        "checkpoint-policy",
    )
    require(
        life.get("current_request")
        in ("terminal", "progress", "late-command-error-correction"),
        "checkpoint-policy",
    )
    require(life.get("current_persistence_success") is None, "checkpoint-current-persistence")
    attempts = integer(life.get("prior_file_writer_attempts"), maximum=MAX_SAFE_INT)
    successes = integer(life.get("prior_successful_file_writes"), maximum=MAX_SAFE_INT)
    require(successes <= attempts, "checkpoint-counts")
    require(type(life.get("checkpoint_failed")) is bool, "checkpoint-failure")
    equal(
        life.get("file_checkpoint_failure_is_terminal_and_not_retried"),
        True,
        "checkpoint-policy",
    )
    equal(life.get("stdout_intermediate_checkpoints_are_lazy"), True, "checkpoint-policy")
    if life["checkpoint_failed"]:
        require(run_status == "failed", "run-status-mismatch")


def check_environment(env):
    """Validate static hardware fields and start/end thermal/power snapshots."""
    require(isinstance(env, dict), "environment-invalid")
    require(
        isinstance(env.get("processor_name"), str) and len(env["processor_name"]) > 0,
        "environment-invalid",
    )
    require(
        isinstance(env.get("macos_version"), str) and len(env["macos_version"]) > 0,
        "environment-invalid",
    )
    integer(env.get("performance_core_count"), maximum=U32)
    integer(env.get("efficiency_core_count"), maximum=U32)
    logical = integer(env.get("logical_core_count"), maximum=U32)
    require(logical > 0, "environment-invalid")
    page_size = integer(env.get("page_size_bytes"), decimal=True)
    require(page_size > 0, "environment-invalid")
    integer(env.get("l1_data_cache_bytes"), decimal=True)
    integer(env.get("l2_data_cache_bytes"), decimal=True)
    avail = env.get("available_memory_bytes")
    if avail is not None:
        integer(avail, decimal=True)
    require(isinstance(env.get("available_memory_source"), str), "environment-invalid")
    qos = env.get("main_thread_qos")
    require(isinstance(qos, dict), "environment-invalid")
    require(type(qos.get("requested")) is bool, "environment-invalid")
    require(type(qos.get("applied")) is bool, "environment-invalid")
    require(type(qos.get("code")) is int, "environment-invalid")
    for snap_key in ("start", "end"):
        snap = env.get(snap_key)
        require(isinstance(snap, dict), "environment-invalid")
        require(snap.get("thermal_state") in VALID_THERMAL_STATES, "environment-thermal-state")
        require(type(snap.get("low_power_mode_available")) is bool, "environment-invalid")
        lpm_enabled = snap.get("low_power_mode_enabled")
        require(lpm_enabled is None or type(lpm_enabled) is bool, "environment-invalid")
        if snap["low_power_mode_available"]:
            require(type(lpm_enabled) is bool, "environment-invalid")
        phys = snap.get("physical_memory_bytes")
        if phys is not None:
            integer(phys, decimal=True)


def check_validation_checks(checks, expected_specs, measured):
    """Validate ordered execution.validation.checks[] list."""
    require(isinstance(checks, list), "validation-invalid")
    equal(len(checks), len(expected_specs), "validation-count")
    for check, (kind, applicable) in zip(checks, expected_specs):
        require(isinstance(check, dict), "validation-invalid")
        equal(check.get("kind"), kind, "validation-kind")
        equal(check.get("applicable"), applicable, "validation-applicability")
        require(
            isinstance(check.get("reason_code"), str) and len(check["reason_code"]) > 0,
            "validation-reason",
        )
        if not applicable:
            require(
                check.get("evaluated") is None and check.get("valid") is None,
                "validation-null",
            )
        else:
            require(type(check.get("evaluated")) is bool, "validation-evaluated")
            if check["evaluated"]:
                require(type(check.get("valid")) is bool, "validation-valid")
            else:
                require(check.get("valid") is None, "validation-null")
            if measured:
                require(check["evaluated"] and check["valid"], "required-validation-failed")


def check_run_acceptance(doc, measured_count, total_planned, attempted_count, rows):
    """Validate run status state machine, results_complete, and run_accepted."""
    status = doc.get("status")
    require(status in VALID_RUN_STATUSES, "run-status")
    require(type(doc.get("interruption_requested")) is bool, "interruption-flag")
    if status == "complete":
        require(
            measured_count == total_planned
            and total_planned > 0
            and not doc["interruption_requested"],
            "run-status-mismatch",
        )
    elif status == "partial":
        require(
            0 < attempted_count
            and measured_count < total_planned
            and not doc["interruption_requested"],
            "run-status-mismatch",
        )
    elif status == "interrupted":
        require(
            doc["interruption_requested"] and measured_count < total_planned,
            "run-status-mismatch",
        )
    elif status in ("not_started", "unsupported"):
        require(attempted_count == 0, "run-status-mismatch")
    if any(r.get("status") == "invalid" for r in rows):
        require(status == "failed", "run-status-mismatch")
    for r in rows:
        if r.get("status") == "interrupted":
            require(doc["interruption_requested"], "interruption-flag")
    complete = total_planned > 0 and measured_count == total_planned
    equal(doc.get("results_complete"), complete, "results-complete-mismatch")
    checkpoint_failed = doc["checkpoint_lifecycle"]["checkpoint_failed"]
    accepted_run = status == "complete" and complete and not checkpoint_failed
    equal(doc.get("run_accepted"), accepted_run, "run-acceptance-mismatch")
    return accepted_run


def check_statistics_block(got_stats, expected_stats, count_reason="aggregate-count-mismatch", stat_reason="aggregate-statistic-mismatch"):
    """Compare serialized statistics object against computed DescriptiveStatistics."""
    if expected_stats is None:
        require(got_stats is None, "aggregate-null")
        return
    require(isinstance(got_stats, dict), stat_reason)
    equal(got_stats.get("sample_count"), expected_stats["sample_count"], count_reason)
    for key, value in expected_stats.items():
        if key == "sample_count":
            continue
        if value is None:
            require(got_stats.get(key) is None, stat_reason)
        else:
            close(got_stats.get(key), value, stat_reason)


def check_memory_budget_block(mb, min_committed_bytes=0, require_valid_when_active=True):
    """Validate memory_budget arithmetic invariants."""
    require(isinstance(mb, dict), "memory-budget-invalid")
    require(type(mb.get("valid")) is bool, "memory-budget-invalid")
    require(
        isinstance(mb.get("reason_code"), str) and len(mb["reason_code"]) > 0,
        "memory-budget-invalid",
    )
    avail = integer(mb.get("available_memory_bytes"), decimal=True)
    allowed = integer(mb.get("allowed_memory_bytes"), decimal=True)
    admitted = integer(mb.get("admitted_budget_bytes"), decimal=True)
    known_peak = integer(mb.get("known_owned_peak_bytes"), decimal=True)
    runtime_peak = integer(mb.get("runtime_peak_bytes"), decimal=True)
    setup_peak = integer(mb.get("setup_peak_bytes"), decimal=True)
    transient_peak = integer(mb.get("transient_peak_bytes"), decimal=True)
    rounding_bytes = integer(mb.get("resource_rounding_bytes"), decimal=True)
    require(type(mb.get("used_fallback")) is bool, "memory-budget-invalid")
    equal(runtime_peak, known_peak, "memory-budget-arithmetic")
    require(admitted <= allowed, "memory-budget-arithmetic")
    if mb["valid"]:
        require(known_peak <= admitted, "memory-budget-arithmetic")
        require(known_peak >= min_committed_bytes, "memory-budget-arithmetic")
    elif require_valid_when_active:
        raise Rejected("memory-budget-invalid")
    require(avail >= 0 and setup_peak >= 0 and transient_peak >= 0 and rounding_bytes >= 0, "memory-budget-arithmetic")


_VALUE_OPTIONS = frozenset({
    "-b", "--buffer-size",
    "-k", "--cache-size",
    "-t", "--threads",
    "-i", "--iterations",
    "-r", "--count",
    "--seed",
    "-o", "--output",
    "-S", "--sweep",
    "-X", "--sweep-max-runs",
    "-n", "--latency-samples",
    "-s", "--latency-stride-bytes",
    "-m", "--latency-chain-mode",
    "-l", "--latency-tlb-locality-kb",
})


def argv_present_flags(argv):
    """Extract canonical option flags from argv, skipping option values."""
    flags = set()
    idx = 1
    while idx < len(argv):
        tok = argv[idx]
        if tok.startswith("--") and "=" in tok:
            flags.add(tok.split("=", 1)[0])
            idx += 1
            continue
        flags.add(tok)
        if tok in _VALUE_OPTIONS:
            idx += 2
        else:
            idx += 1
    return flags


def check_buffer_evidence(
    ev,
    expected_bytes,
    non_cacheable,
    page_size,
    *,
    require_ro=False,
    allow_unallocated=False,
):
    """Validate serialized AllocatedBufferEvidence entry."""
    require(isinstance(ev, dict), "resource-evidence-invalid")
    req = integer(ev.get("requested_bytes"), decimal=True)
    mapped = integer(ev.get("mapped_bytes"), decimal=True)
    nc_req = ev.get("non_cacheable_requested")
    madv_advice = integer(ev.get("madvise_advice"), maximum=U32)
    madv_applied = ev.get("madvise_applied")
    madv_errno = integer(ev.get("madvise_errno"), maximum=U32)
    ro = ev.get("read_only_protected")
    mprot_errno = integer(ev.get("mprotect_errno"), maximum=U32)
    require(
        type(nc_req) is bool and type(madv_applied) is bool and type(ro) is bool,
        "resource-evidence-invalid",
    )
    if req == 0 and allow_unallocated:
        require(
            mapped == 0 and ro is False and mprot_errno == 0 and madv_applied is False,
            "resource-evidence-invalid",
        )
        return
    equal(req, expected_bytes, "resource-evidence-invalid")
    require(mapped >= req and mapped % page_size == 0, "resource-evidence-invalid")
    equal(nc_req, non_cacheable, "resource-evidence-invalid")
    equal(madv_advice, 1 if non_cacheable else 3, "resource-evidence-invalid")
    if madv_applied:
        equal(madv_errno, 0, "resource-evidence-invalid")
    elif req > 0:
        require(madv_errno != 0, "resource-evidence-invalid")
    if require_ro:
        require(ro is True and mprot_errno == 0, "resource-evidence-invalid")


def check_resolved_sources(resolved_sources, cfg, expected_specs, flag_map):
    """Validate configuration.resolved_sources tokens and cross-field consistency."""
    require(isinstance(resolved_sources, dict), "resolved-sources-invalid")
    equal(set(resolved_sources.keys()), set(expected_specs.keys()), "resolved-sources-invalid")
    for key, allowed in expected_specs.items():
        require(resolved_sources.get(key) in allowed, "resolved-sources-invalid")

    equal(
        resolved_sources["iterations"],
        "explicit" if cfg.get("iterations") is not None else "auto",
        "resolved-sources-invalid",
    )
    equal(
        resolved_sources["seed"],
        "explicit" if cfg.get("seed_source") == "user" else "generated",
        "resolved-sources-invalid",
    )
    if "workers" in expected_specs:
        if cfg.get("worker_source") == "auto":
            equal(resolved_sources["workers"], "auto", "resolved-sources-invalid")
        else:
            require(resolved_sources["workers"] in ("explicit", "sweep"), "resolved-sources-invalid")

    argv = cfg.get("argv") or []
    if argv:
        flags = argv_present_flags(argv)
        has_sweep_flag = ("-S" in flags or "--sweep" in flags)
        for key, cli_flags in flag_map.items():
            val = resolved_sources[key]
            if any(f in flags for f in cli_flags):
                equal(val, "explicit", "resolved-sources-invalid")
            elif val == "explicit":
                raise Rejected("resolved-sources-invalid")
            if val == "sweep" and not has_sweep_flag:
                raise Rejected("resolved-sources-invalid")


def _env_snapshot_non_nominal(snap):
    thermal_bad = snap["thermal_state"] not in ("nominal", "unavailable")
    lpm_bad = snap["low_power_mode_available"] and snap["low_power_mode_enabled"]
    return thermal_bad or lpm_bad


def check_quality_warnings(
    qw,
    *,
    env,
    cfg,
    buffer_evidences,
    all_targets_have_resources,
    active_targets,
    operations,
    agg_targets,
    attempted_any,
    iterations,
    rows,
    duration_window,
    target_key="target",
):
    """Validate quality_warnings types, uniqueness, vocabulary, and consistency."""
    require(
        isinstance(qw, list) and all(isinstance(w, str) and len(w) > 0 for w in qw),
        "quality-warnings-invalid",
    )
    require(len(qw) == len(set(qw)), "quality-warnings-invalid")

    allowed = {"environment-not-nominal", "main-thread-qos-not-applied", "madvise-random-not-applied"}
    if operations is not None:
        allowed.add("worker-qos-not-applied")
    for tname in active_targets:
        if operations is not None:
            for op in operations:
                allowed.add(f"{tname}-{op}-high-cv")
                if iterations is None:
                    for q in ("below-target-window", "above-target-window", "unavailable"):
                        allowed.add(f"{tname}-{op}-duration-{q}")
        else:
            allowed.add(f"{tname}-high-cv")
            if iterations is None:
                for q in ("below-target-window", "above-target-window", "unavailable"):
                    allowed.add(f"{tname}-duration-{q}")

    require(all(w in allowed for w in qw), "quality-warnings-invalid")

    env_bad = _env_snapshot_non_nominal(env["start"]) or _env_snapshot_non_nominal(env["end"])
    equal("environment-not-nominal" in qw, env_bad, "quality-warnings-invalid")

    qos = env["main_thread_qos"]
    qos_bad = bool(qos["requested"] and not qos["applied"])
    equal("main-thread-qos-not-applied" in qw, qos_bad, "quality-warnings-invalid")

    if operations is not None and not attempted_any:
        require("worker-qos-not-applied" not in qw, "quality-warnings-invalid")

    if not cfg["non_cacheable"]:
        require("madvise-random-not-applied" not in qw, "quality-warnings-invalid")
    else:
        any_madvise_missed = any(ev.get("madvise_applied") is False for ev in buffer_evidences)
        if any_madvise_missed:
            require("madvise-random-not-applied" in qw, "quality-warnings-invalid")
        elif all_targets_have_resources:
            require("madvise-random-not-applied" not in qw, "quality-warnings-invalid")

    for tname in active_targets:
        if operations is not None:
            for op in operations:
                cv_warn = agg_targets[tname][op]["quality"]["cv_warning"]
                equal(f"{tname}-{op}-high-cv" in qw, cv_warn, "quality-warnings-invalid")
        else:
            cv_warn = agg_targets[tname]["quality"]["cv_warning"]
            equal(f"{tname}-high-cv" in qw, cv_warn, "quality-warnings-invalid")

    if iterations is None:
        min_sec, max_sec = duration_window
        expected_duration_warnings = set()
        for row in rows:
            if row.get("status") != "measured":
                continue
            elapsed = row["execution"]["timing"]["elapsed_seconds"]
            if elapsed < min_sec:
                dq = "below-target-window"
            elif elapsed > max_sec:
                dq = "above-target-window"
            else:
                continue
            tname = row[target_key]
            if operations is not None:
                expected_duration_warnings.add(f"{tname}-{row['operation']}-duration-{dq}")
            else:
                expected_duration_warnings.add(f"{tname}-duration-{dq}")

        for tname in active_targets:
            if operations is not None:
                for op in operations:
                    for q in ("below-target-window", "above-target-window", "unavailable"):
                        tok = f"{tname}-{op}-duration-{q}"
                        equal(tok in qw, tok in expected_duration_warnings, "quality-warnings-invalid")
            else:
                for q in ("below-target-window", "above-target-window", "unavailable"):
                    tok = f"{tname}-duration-{q}"
                    equal(tok in qw, tok in expected_duration_warnings, "quality-warnings-invalid")


def verify_sweep_envelope(
    doc,
    base_mode,
    param_specs,
    verify_single_fn,
    fixed_config_keys,
    binary=None,
    require_raw=False,
    budget=None,
    extra_verify_kwargs=None,
    cross_param_validator=None,
    sweep_resolved_source_map=None,
):
    """Verify a schema-1 sweep envelope independently of JSON key ordering.

    param_specs: dict mapping cli_key -> (config_key, parse_and_validate_fn)
    """
    extra_verify_kwargs = extra_verify_kwargs or {}
    cfg = doc.get("configuration")
    require(isinstance(cfg, dict), "sweep-configuration-invalid")
    if cfg.get("mode") != "sweep" or cfg.get("base_mode") != base_mode:
        raise Rejected("unsupported-mode", True)
    if type(cfg.get("sweep_schema_version")) is not int or cfg["sweep_schema_version"] != 1:
        raise Rejected("unsupported-sweep-schema", True)

    run_count = integer(cfg.get("run_count"), maximum=MAX_SAFE_INT)
    sweep_max_runs = integer(cfg.get("sweep_max_runs"), maximum=MAX_SAFE_INT)
    planned_runs = integer(doc.get("planned_runs"), maximum=MAX_SAFE_INT)
    attempted_runs = integer(doc.get("attempted_runs"), maximum=MAX_SAFE_INT)
    completed_runs = integer(doc.get("completed_runs"), maximum=MAX_SAFE_INT)
    require(
        run_count > 0 and run_count == planned_runs and run_count <= sweep_max_runs,
        "sweep-plan-count-mismatch",
    )

    sweep_params = cfg.get("sweep_parameters")
    require(isinstance(sweep_params, dict) and len(sweep_params) > 0, "sweep-parameters-invalid")
    param_keys = []
    parsed_params = {}
    product = 1
    for key, values in sweep_params.items():
        require(key in param_specs, "sweep-parameter-unsupported")
        require(isinstance(values, list) and len(values) > 0, "sweep-parameters-invalid")
        _, validator = param_specs[key]
        parsed_values = [validator(v) for v in values]
        product *= len(parsed_values)
        if product > planned_runs or product > sweep_max_runs:
            raise Rejected("sweep-plan-count-mismatch")
        param_keys.append(key)
        parsed_params[key] = parsed_values
    require(product == planned_runs, "sweep-plan-count-mismatch")
    if cross_param_validator is not None:
        cross_param_validator(parsed_params)

    status = doc.get("status")
    require(status in ("complete", "partial", "interrupted", "failed"), "sweep-status-invalid")
    status_reason = doc.get("status_reason")
    require(
        status_reason is None or (isinstance(status_reason, str) and len(status_reason) > 0),
        "sweep-status-invalid",
    )
    conclusions_valid = doc.get("conclusions_valid")
    require(type(conclusions_valid) is bool, "sweep-conclusions-invalid")

    elapsed_sec = doc.get("execution_time_sec")
    require(
        type(elapsed_sec) in (int, float) and math.isfinite(elapsed_sec) and elapsed_sec >= 0,
        "sweep-metadata-invalid",
    )
    timestamp = doc.get("timestamp")
    require(isinstance(timestamp, str) and len(timestamp) > 0, "sweep-metadata-invalid")
    version = doc.get("version")
    require(isinstance(version, str) and len(version) > 0, "sweep-metadata-invalid")

    runs = doc.get("runs")
    require(isinstance(runs, list) and len(runs) <= planned_runs, "sweep-runs-invalid")
    equal(attempted_runs, len(runs), "sweep-attempted-runs-mismatch")

    if budget is None:
        budget = Budget()
    budget(len(runs))
    binary_digest = hash_binary(binary) if binary is not None else None

    swept_config_keys = {param_specs[k][0] for k in param_keys}
    invariant_keys = tuple(fixed_config_keys) + tuple(
        cfg_key for _, (cfg_key, _) in param_specs.items() if cfg_key not in swept_config_keys
    )
    expected_swept_rs_keys = set()
    sweepable_rs_keys = ()
    if sweep_resolved_source_map is not None:
        expected_swept_rs_keys = {sweep_resolved_source_map[k] for k in param_keys}
        sweepable_rs_keys = tuple(sweep_resolved_source_map.values())

    multi_keys = [k for k in param_keys if len(parsed_params[k]) > 1]
    fixed_keys = tuple(k for k in param_keys if len(parsed_params[k]) == 1)
    candidate_orders = [tuple(order) + fixed_keys for order in itertools.permutations(multi_keys)]
    active_candidates = [
        (order, itertools.islice(itertools.product(*(parsed_params[k] for k in order)), len(runs)))
        for order in candidate_orders
    ]
    param_key_set = set(param_keys)

    run_verdicts = []
    observed_completed = 0
    observed_methodology = None
    observed_base_config = None
    observed_manifest = None
    checksum_levels = set()
    timing_levels = set()
    build_levels = set()
    stopped = False

    for index, run in enumerate(runs):
        require(not stopped, "sweep-invalid-execution-prefix")
        require(isinstance(run, dict), "sweep-run-invalid")
        equal(run.get("index"), index, "sweep-run-index-mismatch")
        run_params = run.get("parameters")
        require(
            isinstance(run_params, dict) and run_params.keys() == param_key_set,
            "sweep-parameters-mismatch",
        )
        normalized_run_params = {}
        for k, v in run_params.items():
            _, validator = param_specs[k]
            try:
                normalized_run_params[k] = validator(v)
            except Rejected as err:
                if err.reason in ("integer-encoding", "integer-range", "sweep-parameter-value-invalid"):
                    raise Rejected("sweep-parameters-mismatch") from err
                raise

        budget(len(active_candidates))
        next_candidates = []
        for order, iterator in active_candidates:
            combo = next(iterator, None)
            if combo is not None and all(normalized_run_params[k] == v for k, v in zip(order, combo)):
                next_candidates.append((order, iterator))
        active_candidates = next_candidates
        require(len(active_candidates) > 0, "sweep-parameters-mismatch")

        run_status = run.get("status")
        require(run_status in ("complete", "partial", "interrupted", "failed"), "sweep-run-status-invalid")
        run_reason = run.get("status_reason")
        if run_status == "complete":
            require(run_reason is None, "sweep-run-status-mismatch")
        else:
            require(isinstance(run_reason, str) and len(run_reason) > 0, "sweep-run-status-mismatch")
            stopped = True

        nested = run.get("result")
        if nested is None:
            require(run_status != "complete", "sweep-run-status-mismatch")
            run_verdicts.append(
                dict(
                    run_index=index,
                    status=run_status,
                    artifact_consistent=True,
                    run_accepted=False,
                    reason_code="nested-result-null",
                )
            )
            continue

        require(isinstance(nested, dict), "sweep-run-result-invalid")
        nested_verdict = verify_single_fn(
            nested,
            require_raw=require_raw,
            budget=budget,
            binary_digest=binary_digest,
            **extra_verify_kwargs,
        )
        nested_methodology = (nested.get("mode"), nested.get("methodology_version"))
        if observed_methodology is None:
            observed_methodology = nested_methodology
        else:
            equal(nested_methodology, observed_methodology, "sweep-profile-mismatch")

        nested_cfg = nested["configuration"]
        if sweep_resolved_source_map is not None:
            nested_rs = nested_cfg.get("resolved_sources")
            require(isinstance(nested_rs, dict), "resolved-sources-invalid")
            for rs_key in sweepable_rs_keys:
                equal(
                    nested_rs.get(rs_key) == "sweep",
                    rs_key in expected_swept_rs_keys,
                    "resolved-sources-invalid",
                )
        equal(nested_cfg.get("output_file"), "", "sweep-nested-output-file-mismatch")
        base_config = {k: nested_cfg.get(k) for k in invariant_keys}
        if observed_base_config is None:
            observed_base_config = base_config
            observed_manifest = nested["build_manifest"]
        else:
            equal(base_config, observed_base_config, "sweep-base-config-mismatch")
            equal(nested["build_manifest"], observed_manifest, "sweep-base-config-mismatch")

        for key, expected_value in normalized_run_params.items():
            cfg_key = param_specs[key][0]
            actual_cfg_val = nested_cfg.get(cfg_key)
            if isinstance(expected_value, str):
                _, validator = param_specs[key]
                actual_cfg_val = validator(actual_cfg_val)
            equal(actual_cfg_val, expected_value, "sweep-nested-parameter-mismatch")

        nested_status = nested["status"]
        if run_status == "complete":
            require(nested_verdict["run_accepted"] is True, "sweep-run-status-mismatch")
            observed_completed += 1
        elif run_status == "interrupted":
            require(nested_status == "interrupted", "sweep-run-status-mismatch")
        elif run_status == "failed":
            require(
                nested_status in ("failed", "unsupported") or not nested_verdict["run_accepted"],
                "sweep-run-status-mismatch",
            )
        elif run_status == "partial":
            require(not nested_verdict["run_accepted"], "sweep-run-status-mismatch")

        checksum_levels.add(nested_verdict["checks"].get("checksum", "unavailable"))
        timing_levels.add(nested_verdict["checks"].get("timing", "unavailable"))
        build_levels.add(nested_verdict["checks"].get("build", "unavailable"))
        run_verdicts.append(
            dict(
                run_index=index,
                status=run_status,
                artifact_consistent=nested_verdict["artifact_consistent"],
                run_accepted=nested_verdict["run_accepted"],
                reason_code=nested_verdict["reason_code"],
            )
        )

    equal(completed_runs, observed_completed, "sweep-completed-runs-mismatch")
    all_complete = (
        planned_runs > 0 and len(runs) == planned_runs and completed_runs == planned_runs
    )
    last_status = runs[-1]["status"] if runs else None
    if status == "complete":
        require(
            all_complete and conclusions_valid is True and status_reason is None,
            "sweep-status-mismatch",
        )
    else:
        require(
            conclusions_valid is False
            and isinstance(status_reason, str)
            and len(status_reason) > 0,
            "sweep-status-mismatch",
        )
        if status == "failed":
            if status_reason != "checkpoint-write-failed":
                require(last_status == "failed", "sweep-status-mismatch")
        elif status == "interrupted":
            require(last_status != "failed", "sweep-status-mismatch")
        elif status == "partial":
            require(
                len(runs) > 0 and last_status in ("partial", "complete"),
                "sweep-status-mismatch",
            )
            require(
                last_status == "partial" or len(runs) < planned_runs,
                "sweep-status-mismatch",
            )

    envelope_accepted = status == "complete" and conclusions_valid is True and all_complete

    def summarize_level(levels, preferred_order):
        for candidate in preferred_order:
            if candidate in levels:
                return candidate
        return "unavailable"

    return dict(
        artifact_consistent=True,
        run_accepted=envelope_accepted,
        reason_code="valid" if envelope_accepted else "sweep-not-accepted",
        artifact_kind=f"{base_mode}_sweep",
        planned_runs=planned_runs,
        completed_runs=completed_runs,
        verified_runs=len(runs),
        run_verdicts=run_verdicts,
        checks=dict(
            checksum=summarize_level(checksum_levels, ("unavailable", "independent-oracle")),
            timing=summarize_level(timing_levels, ("elapsed-only", "unavailable", "cpu-raw-ticks")),
            build=summarize_level(build_levels, ("unavailable", "manifest-only", "binary-bound")),
        ),
        oracle_work_used=budget.used,
    )


def run_verifier_cli(verify_fn, description, extra_args_fn=None):
    """Standard CLI entrypoint handling exit codes 0/1/2 and single sorted JSON line."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("result", nargs="?", default=None)
    parser.add_argument("--binary", type=Path)
    parser.add_argument(
        "--require-raw-timing",
        action="store_true",
        help="Require original raw Mach timing evidence for CPU measurements",
    )
    if extra_args_fn is not None:
        extra_args_fn(parser)
    args = parser.parse_args()
    if args.result is None:
        if not sys.stdin.isatty():
            args.result = "-"
        else:
            parser.error("the following arguments are required: result")
    try:
        doc = load_strict_json(args.result)
        verdict = verify_fn(doc, args)
        code = 0 if verdict["run_accepted"] else 1
    except Rejected as error:
        verdict = dict(
            artifact_consistent=None if error.unsupported else False,
            run_accepted=False,
            reason_code=error.reason,
        )
        code = 2 if error.unsupported else 1
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        RecursionError,
        ArithmeticError,
        AttributeError,
    ) as error:
        verdict = dict(
            artifact_consistent=False,
            run_accepted=False,
            reason_code="invalid-input",
            detail=str(error)[:200],
        )
        code = 1
    print(json.dumps(verdict, sort_keys=True))
    return code

#!/usr/bin/env python3
"""Verify one CPU bandwidth schema-1 artifact or schema-1 sweep envelope
independently of the C++ producer.

Usage:
  python3 script-examples/verify_cpu_bandwidth_result.py result.json [--binary memory_benchmark] [--require-raw-timing]
Exit 0: consistent accepted artifact; 1: inconsistent or unaccepted; 2: unsupported.
"""

import math
import sys

from cpu_verify_common import (
    DEFAULT_MAX_WORK,
    FOLD_INIT_A,
    FOLD_INIT_B,
    MAX_SAFE_INT,
    SPLITMIX_A,
    SPLITMIX_B,
    SPLITMIX_C,
    U32,
    U64,
    Budget,
    Rejected,
    check_checkpoint_lifecycle,
    check_buffer_evidence,
    check_environment,
    check_manifest,
    check_memory_budget_block,
    check_quality_warnings,
    check_resolved_sources,
    check_run_acceptance,
    check_statistics_block,
    check_validation_checks,
    close,
    compute_statistics,
    enforce_require_raw_timing,
    equal,
    hash_binary,
    integer,
    require,
    rotl64,
    run_verifier_cli,
    splitmix64_mix,
    verify_cpu_raw_timing,
    verify_sweep_envelope,
)

SCHEMA_VERSION = 1
MODE_NAME = "cpu_bandwidth"
METHODOLOGY_VERSION = "cpu-bandwidth-v1-arm64-calibrated-seeded-balanced"

EXPECTED_COMPONENTS = {
    "version": "cpu-bandwidth-components-v1",
    "work_plan_version": "cpu-bandwidth-work-plan-v1",
    "buffer_pattern_version": "cpu-bw-buffer-pattern-v1",
    "read_checksum_version": "cpu-bw-read-checksum-v1",
    "run_policy_version": "bounded-loop-snapshots-v1",
}

EXPECTED_RESOLVED_SOURCE_TOKENS = {
    "buffer_size": ("explicit", "default", "sweep"),
    "cache_size": ("explicit", "default", "sweep"),
    "workers": ("explicit", "auto", "sweep"),
    "iterations": ("explicit", "auto"),
    "seed": ("explicit", "generated"),
}

BW_FLAG_MAP = {
    "buffer_size": ("-b", "--buffer-size"),
    "cache_size": ("-k", "--cache-size"),
    "workers": ("-t", "--threads"),
    "iterations": ("-i", "--iterations"),
    "seed": ("--seed",),
}

EXPECTED_COMPARABILITY = [
    "same-methodology-version",
    "same-frozen-work-plan-and-targets",
    "same-software-version",
    "sufficiently-similar-hardware",
    "sufficiently-similar-thermal-and-power-state",
]

TARGET_DOMAINS = {
    "main_memory": 0x4342575F4D41494E,
    "l1": 0x4342575F4C315F5F,
    "l2": 0x4342575F4C325F5F,
    "custom": 0x4342575F43555354,
}
PRE_COPY_SENTINEL_DOMAIN = 0x4342575F434F5059

CANONICAL_TARGETS = ("main_memory", "l1", "l2", "custom")
OPERATIONS = ("read", "write", "copy")

MAX_PASSES = 1_000_000_000
MAX_EXACT_PAYLOAD_BYTES = 64 * 1024 * 1024 * 1024
CV_WARNING_THRESHOLD_PCT = 5.0


def build_bw_table(target_src_seed):
    """Build 256-entry SplitMix64 table T[k] and 257-entry prefix XOR table P_T."""
    table = [splitmix64_mix(target_src_seed ^ k) for k in range(256)]
    prefix = [0] * 257
    for k in range(256):
        prefix[k + 1] = prefix[k] ^ table[k]
    return table, prefix


def prefix_xor_0_to_n_minus_1(n):
    """Exact O(1) XOR of integers [0, n)."""
    if n <= 0:
        return 0
    m = n - 1
    r = m & 3
    if r == 0:
        return m & U64
    if r == 1:
        return 1
    if r == 2:
        return (m + 1) & U64
    return 0


def bw_pass_xor(target_src_seed, offset_bytes, span_bytes, table_cache=None):
    """Exact O(1) ARM64 read-kernel XOR oracle for [offset_bytes, offset_bytes + span_bytes).

    Vector chunks [0, span - (span % 32)) XOR 64-bit LE words into 4 SIMD lanes folded to x0;
    remainder < 32B tail XORs each byte into bits [0..7] of x0.
    """
    if span_bytes == 0:
        return 0
    table, prefix = table_cache if table_cache is not None else build_bw_table(target_src_seed)

    def prefix_t(n):
        return (prefix[256] if ((n >> 8) & 1) else 0) ^ prefix[n & 255]

    vec_bytes = span_bytes - (span_bytes % 32)
    vec_xor = 0
    if vec_bytes > 0:
        require(offset_bytes % 8 == 0, "worker-alignment")
        start_word = offset_bytes >> 3
        end_word = start_word + (vec_bytes >> 3)
        range_t = prefix_t(end_word) ^ prefix_t(start_word)
        x = prefix_xor_0_to_n_minus_1(end_word) ^ prefix_xor_0_to_n_minus_1(start_word)
        range_mix = x ^ rotl64(x, 19) ^ rotl64(x, 41)
        vec_xor = range_t ^ range_mix

    tail_xor = 0
    for pos in range(offset_bytes + vec_bytes, offset_bytes + span_bytes):
        w_idx = pos >> 3
        w_val = table[w_idx & 255] ^ (w_idx ^ rotl64(w_idx, 19) ^ rotl64(w_idx, 41))
        tail_xor ^= (w_val >> (8 * (pos & 7))) & 0xFF

    return (vec_xor ^ tail_xor) & U64


def partition_64b(total_bytes, workers):
    """Canonical 64-byte aligned partition across workers."""
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


def fold_bw_workers(worker_tuples):
    """Deterministic SplitMix64 fold across worker tuples (state_a, state_b, exact_bytes_read, pass_count)."""
    a = FOLD_INIT_A
    b = FOLD_INIT_B
    for ordinal, (sa, sb, exact_bytes, passes) in enumerate(worker_tuples):
        a = splitmix64_mix(a ^ sa ^ ((exact_bytes + SPLITMIX_A * (ordinal + 1)) & U64))
        b = splitmix64_mix(b ^ sb ^ ((passes + SPLITMIX_B * (ordinal + 1)) & U64))
    return {
        "state_a_uint64_decimal": str(a),
        "state_b_uint64_decimal": str(b),
    }


def expected_read_checksum(target_src_seed, op_seed, passes, worker_spans, table_cache=None, budget=None):
    """Closed-form O(1) cpu-bw-read-checksum-v1 oracle per worker and run fold."""
    if budget is not None:
        budget(len(worker_spans))
    if table_cache is None:
        table_cache = build_bw_table(target_src_seed)
    s1 = ((passes * (passes + 1)) // 2) & U64
    workers_out = []
    tuples = []
    for w, (offset, span) in enumerate(worker_spans):
        px = bw_pass_xor(target_src_seed, offset, span, table_cache)
        init_a = splitmix64_mix(op_seed ^ (0xA100000000000000 | w))
        init_b = splitmix64_mix(op_seed ^ (0xB200000000000000 | w))
        state_a = (init_a + passes * px + SPLITMIX_A * s1) & U64
        state_b = (init_b + ((px ^ SPLITMIX_B) * s1) + SPLITMIX_C * passes) & U64
        exact_bytes = passes * span
        workers_out.append(
            {
                "worker_index": w,
                "state_a_uint64_decimal": str(state_a),
                "state_b_uint64_decimal": str(state_b),
                "exact_bytes_read": str(exact_bytes),
                "pass_count_uint64_decimal": str(passes),
            }
        )
        tuples.append((state_a, state_b, exact_bytes, passes))
    return workers_out, fold_bw_workers(tuples)


def _validation_specs_for_op(op):
    return [
        ("post-validation-structure", True),
        ("read-checksum", op == "read"),
        ("write-final-full-buffer", op == "write"),
        ("copy-final-full-buffer", op == "copy"),
        ("src-readonly-protection", op in ("read", "copy")),
    ]


def _check_bw_checksum(observed, expected, op, compact=False):
    require(isinstance(observed, dict), "checksum-invalid")
    status = observed.get("status")
    if op != "read":
        require(status in ("not_applicable", "not_evaluated"), "checksum-status")
        require(
            observed.get("checksum_valid") is None
            and observed.get("actual_run_checksum") is None
            and observed.get("actual_worker_checksums") is None,
            "checksum-null",
        )
        return True
    if status in ("unavailable", "not_evaluated"):
        require(
            observed.get("checksum_valid") is None
            and observed.get("actual_run_checksum") is None,
            "checksum-null",
        )
        if not compact:
            require(observed.get("actual_worker_checksums") is None, "checksum-null")
        return False
    require(status in ("valid", "invalid"), "checksum-status")
    require(expected.get("status") == "available", "expected-checksum-unavailable")
    matches = observed.get("actual_run_checksum") == expected.get("expected_run_checksum")
    if not compact:
        actual_workers = observed.get("actual_worker_checksums")
        require(isinstance(actual_workers, list), "actual-worker-checksums-invalid")
        matches = matches and actual_workers == expected.get("expected_worker_checksums")
        tuples = []
        for i, worker in enumerate(actual_workers):
            require(isinstance(worker, dict), "actual-worker-checksums-invalid")
            equal(worker.get("worker_index"), i, "actual-worker-index")
            tuples.append(
                (
                    integer(worker.get("state_a_uint64_decimal"), decimal=True),
                    integer(worker.get("state_b_uint64_decimal"), decimal=True),
                    integer(worker.get("exact_bytes_read"), decimal=True),
                    integer(worker.get("pass_count_uint64_decimal"), decimal=True),
                )
            )
        equal(observed.get("actual_run_checksum"), fold_bw_workers(tuples), "actual-run-fold-mismatch")
    equal(observed.get("checksum_valid"), matches, "checksum-valid-mismatch")
    equal(status, "valid" if matches else "invalid", "checksum-status")
    return matches


def _check_write_pattern(ex, op, attempted, seen_patterns):
    wp = ex.get("write_pattern_uint64_decimal")
    if op != "write" or not attempted:
        require(wp is None, "write-pattern-unexpected")
        return
    try:
        pattern = integer(wp, decimal=True)
    except Rejected as err:
        raise Rejected("write-pattern-invalid") from err
    require(pattern != 0 and (pattern & 1) == 1, "write-pattern-invalid")
    attempt_seed = ex.get("attempt_seed_uint64_decimal")
    if attempt_seed is not None:
        seed_val = integer(attempt_seed, decimal=True)
        equal(pattern, splitmix64_mix(seed_val) | 1, "write-pattern-invalid")
    require(pattern not in seen_patterns, "write-pattern-invalid")
    seen_patterns.add(pattern)


def verify_single(doc, binary=None, require_raw=False, budget=None, binary_digest=None):
    """Verify a single cpu_bandwidth schema-1 artifact."""
    require(isinstance(doc, dict), "invalid-input")
    if type(doc.get("schema_version")) is not int or doc["schema_version"] != SCHEMA_VERSION:
        raise Rejected("unsupported-schema", True)
    if doc.get("mode") != MODE_NAME:
        raise Rejected("unsupported-mode", True)
    if doc.get("methodology_version") != METHODOLOGY_VERSION:
        raise Rejected("unsupported-methodology", True)

    if budget is None:
        budget = Budget()
    work_before = budget.remaining

    cfg = doc.get("configuration")
    require(isinstance(cfg, dict), "configuration-invalid")
    buf_mb = integer(cfg.get("buffer_size_mb"), maximum=MAX_SAFE_INT)
    cache_kb = cfg.get("custom_cache_size_kb")
    if cache_kb is not None:
        cache_kb = integer(cache_kb, maximum=1048576)
        require(cache_kb == 0 or cache_kb >= 16, "configuration-invalid")
    req_workers = integer(cfg.get("requested_workers"), maximum=U32)
    avail_workers = integer(cfg.get("available_workers"), maximum=U32)
    require(req_workers >= 1 and avail_workers >= 1, "configuration-invalid")
    require(cfg.get("worker_source") in ("auto", "user"), "configuration-invalid")
    iterations = cfg.get("iterations")
    if iterations is not None:
        iterations = integer(iterations, maximum=MAX_PASSES)
        require(iterations >= 1, "configuration-invalid")
    equal(
        cfg.get("work_policy"),
        "explicit_fixed_work" if iterations is not None else "automatic_calibration",
        "configuration-invalid",
    )
    loop_count = integer(cfg.get("loop_count"), maximum=MAX_SAFE_INT)
    require(loop_count >= 1, "configuration-invalid")
    require(type(cfg.get("non_cacheable")) is bool, "configuration-invalid")
    require(cfg.get("seed_source") in ("user", "generated"), "seed-mismatch")
    require(isinstance(cfg.get("output_file"), str), "configuration-invalid")
    require(isinstance(cfg.get("argv"), list) and all(isinstance(a, str) for a in cfg["argv"]), "configuration-invalid")
    check_resolved_sources(
        cfg.get("resolved_sources"),
        cfg,
        EXPECTED_RESOLVED_SOURCE_TOKENS,
        BW_FLAG_MAP,
    )

    env = doc.get("environment")
    check_environment(env)
    page_size = integer(env["page_size_bytes"], decimal=True)

    seeds = doc.get("seeds")
    require(isinstance(seeds, dict), "seed-mismatch")
    base_seed = integer(seeds.get("base_seed_uint64_decimal"), decimal=True)
    equal(cfg.get("base_seed_uint64_decimal"), str(base_seed), "seed-mismatch")
    equal(seeds.get("source"), cfg.get("seed_source"), "seed-mismatch")
    domain_seeds = seeds.get("buffer_domain_seeds")
    require(isinstance(domain_seeds, dict), "seed-mismatch")

    rp = doc.get("resolved_plan")
    require(isinstance(rp, dict), "plan-invalid")
    rp_valid = rp.get("valid")
    require(type(rp_valid) is bool, "plan-invalid")
    rp_reason = rp.get("reason_code")
    require(isinstance(rp_reason, str) and len(rp_reason) > 0, "plan-invalid")
    if rp_valid:
        equal(rp_reason, "valid", "plan-invalid")
    equal(rp.get("component_identities"), EXPECTED_COMPONENTS, "component-mismatch")
    require(
        isinstance(rp.get("plan_identity"), str)
        and rp["plan_identity"].startswith("cpu-bandwidth-work-plan-v1|"),
        "plan-identity-mismatch",
    )

    rows = doc.get("measurements")
    require(isinstance(rows, list), "measurement-invalid")
    if len(rows) > 100_000:
        raise Rejected("unsupported-measurement-limit", True)
    require(all(isinstance(r, dict) for r in rows), "measurement-invalid")
    attempted_any = any(r.get("attempted") is True for r in rows)
    doc_status = doc.get("status")
    if not rp_valid:
        require(
            doc_status == "failed" and not attempted_any and rp_reason != "valid",
            "plan-invalid",
        )
        equal(doc.get("reason_code"), rp_reason, "plan-invalid")

    targets_obj = rp.get("targets")
    require(isinstance(targets_obj, dict), "target-geometry-mismatch")
    active_targets = []
    target_info = {}
    table_cache = {}
    min_committed_bytes = 0

    for tname in CANONICAL_TARGETS:
        tdata = targets_obj.get(tname)
        if tdata is None:
            continue
        require(isinstance(tdata, dict) and type(tdata.get("enabled")) is bool, "target-geometry-mismatch")
        if not tdata["enabled"]:
            continue
        tbytes = integer(tdata.get("bytes"), decimal=True)
        require(tbytes > 0, "target-geometry-mismatch")
        if tname == "main_memory":
            require(buf_mb > 0, "target-geometry-mismatch")
            equal(tbytes, buf_mb * 1048576, "target-geometry-mismatch")
        elif tname == "custom":
            require(cache_kb is not None and cache_kb > 0, "target-geometry-mismatch")
            equal(tbytes, cache_kb * 1024, "target-geometry-mismatch")
        else:
            require(cache_kb is None, "target-geometry-mismatch")

        eff_workers = integer(tdata.get("effective_workers"), maximum=U32)
        require(1 <= eff_workers <= min(req_workers, avail_workers), "worker-plan-mismatch")
        if "worker_spans" in tdata and tdata["worker_spans"] is not None:
            raw_spans = tdata["worker_spans"]
            require(isinstance(raw_spans, list) and len(raw_spans) == eff_workers, "worker-plan-mismatch")
            spans = []
            cursor = 0
            for idx, sp in enumerate(raw_spans):
                require(isinstance(sp, dict), "worker-plan-mismatch")
                off = integer(sp.get("offset_bytes"), decimal=True)
                ln = integer(sp.get("span_bytes"), decimal=True)
                equal(off, cursor, "worker-plan-mismatch")
                if idx > 0 and ln > 0:
                    require(off % 64 == 0, "worker-alignment")
                spans.append((off, ln))
                cursor += ln
            equal(cursor, tbytes, "worker-plan-mismatch")
        else:
            spans = partition_64b(tbytes, eff_workers)

        expected_tseed = splitmix64_mix(base_seed ^ TARGET_DOMAINS[tname])
        equal(domain_seeds.get(tname), str(expected_tseed), "seed-mismatch")
        table_cache[tname] = build_bw_table(expected_tseed)
        active_targets.append(tname)
        target_info[tname] = {
            "bytes": tbytes,
            "effective_workers": eff_workers,
            "spans": spans,
            "src_seed": expected_tseed,
        }
        rounded = ((tbytes + page_size - 1) // page_size) * page_size
        min_committed_bytes += 2 * rounded

    if rp_valid:
        require(len(active_targets) > 0, "target-omission")
        if buf_mb > 0:
            require("main_memory" in target_info, "target-geometry-mismatch")
        else:
            require("main_memory" not in target_info, "target-geometry-mismatch")
        if cache_kb == 0:
            require(all(t not in target_info for t in ("l1", "l2", "custom")), "target-geometry-mismatch")
        elif cache_kb is not None and cache_kb > 0:
            require("custom" in target_info and "l1" not in target_info and "l2" not in target_info, "target-geometry-mismatch")
        else:
            require(("l1" in target_info or "l2" in target_info) and "custom" not in target_info, "target-geometry-mismatch")

    resources_obj = rp.get("resources")
    require(isinstance(resources_obj, dict), "resource-evidence-invalid")
    bw_buffer_evidences = []
    for tname, t_res in resources_obj.items():
        require(tname in target_info and isinstance(t_res, dict), "resource-evidence-invalid")
        require("src" in t_res and "dst" in t_res, "resource-evidence-invalid")
        expected_bytes = target_info[tname]["bytes"]
        require_src_ro = (doc_status == "complete" or attempted_any)
        check_buffer_evidence(
            t_res["src"],
            expected_bytes,
            cfg["non_cacheable"],
            page_size,
            require_ro=require_src_ro,
        )
        check_buffer_evidence(
            t_res["dst"],
            expected_bytes,
            cfg["non_cacheable"],
            page_size,
            require_ro=False,
        )
        bw_buffer_evidences.extend((t_res["src"], t_res["dst"]))
    if doc_status == "complete" or attempted_any:
        equal(set(resources_obj.keys()), set(active_targets), "resource-evidence-invalid")

    if "pre_copy_sentinel_uint64_decimal" in domain_seeds:
        equal(
            domain_seeds["pre_copy_sentinel_uint64_decimal"],
            str(splitmix64_mix(base_seed ^ PRE_COPY_SENTINEL_DOMAIN)),
            "seed-mismatch",
        )

    check_memory_budget_block(
        doc.get("memory_budget"),
        min_committed_bytes=min_committed_bytes,
        require_valid_when_active=attempted_any,
    )

    op_plans = rp.get("operation_plans")
    require(isinstance(op_plans, list), "plan-invalid")
    if len(op_plans) > 4096:
        raise Rejected("unsupported-plan-limit", True)

    for idx, plan in enumerate(op_plans):
        require(isinstance(plan, dict), "plan-invalid")
        equal(plan.get("plan_ref"), idx, "plan-reference")
        tname = plan.get("target")
        op = plan.get("operation")
        require(tname in target_info and op in OPERATIONS, "plan-invalid")
        passes = integer(plan.get("passes"), maximum=MAX_PASSES)
        require(passes >= 1, "work-mismatch")
        require(type(plan.get("explicit_iterations")) is bool, "work-mismatch")
        mult = 2 if op == "copy" else 1
        exact_payload = passes * target_info[tname]["bytes"] * mult
        require(exact_payload <= MAX_EXACT_PAYLOAD_BYTES, "work-cap-mismatch")
        equal(plan.get("exact_payload_bytes"), str(exact_payload), "plan-bytes-mismatch")
        op_seed = integer(plan.get("op_seed_uint64_decimal"), decimal=True)
        exp_ck = plan.get("expected_checksum")
        require(isinstance(exp_ck, dict), "checksum-invalid")
        if op == "read":
            equal(exp_ck.get("status"), "available", "checksum-status")
            exp_workers, exp_run = expected_read_checksum(
                target_info[tname]["src_seed"],
                op_seed,
                passes,
                target_info[tname]["spans"],
                table_cache=table_cache[tname],
                budget=budget,
            )
            equal(exp_ck.get("expected_worker_checksums"), exp_workers, "expected-worker-checksum-mismatch")
            equal(exp_ck.get("expected_run_checksum"), exp_run, "expected-run-checksum-mismatch")
        else:
            equal(exp_ck.get("status"), "not_applicable", "checksum-status")
            require(
                exp_ck.get("expected_worker_checksums") is None
                and exp_ck.get("expected_run_checksum") is None,
                "checksum-null",
            )

    frozen_refs = rp.get("frozen_plan_refs")
    require(isinstance(frozen_refs, dict), "plan-reference")
    for tname in active_targets:
        trefs = frozen_refs.get(tname)
        require(isinstance(trefs, dict), "plan-reference")
        for op in OPERATIONS:
            ref = trefs.get(op)
            if ref is not None:
                integer(ref, maximum=MAX_SAFE_INT)
                require(
                    ref < len(op_plans)
                    and op_plans[ref]["target"] == tname
                    and op_plans[ref]["operation"] == op,
                    "plan-reference",
                )
                if iterations is not None:
                    equal(op_plans[ref]["passes"], iterations, "frozen-work-mismatch")

    seen_write_patterns = set()
    levels = []

    calibration = doc.get("calibration")
    require(isinstance(calibration, dict), "calibration-invalid")
    equal(calibration.get("excluded_from_results"), True, "calibration-excluded-mismatch")
    cal_attempts = calibration.get("attempts")
    require(isinstance(cal_attempts, dict), "calibration-invalid")
    for tname, per_op in cal_attempts.items():
        require(tname in target_info and isinstance(per_op, dict), "calibration-invalid")
        for op, attempts in per_op.items():
            require(op in OPERATIONS and isinstance(attempts, list), "calibration-invalid")
            for idx, attempt in enumerate(attempts):
                require(isinstance(attempt, dict), "calibration-invalid")
                equal(attempt.get("attempt_index"), idx, "calibration-index")
                ref = attempt.get("plan_ref")
                ex = attempt.get("execution")
                require(isinstance(ex, dict), "calibration-invalid")
                if ref is None:
                    require(
                        not attempt.get("terminal")
                        and attempt.get("valid") is False
                        and ex.get("valid") is None
                        and ex.get("elapsed_seconds") is None,
                        "calibration-placeholder",
                    )
                    continue
                integer(ref, maximum=MAX_SAFE_INT)
                require(
                    ref < len(op_plans)
                    and op_plans[ref]["target"] == tname
                    and op_plans[ref]["operation"] == op,
                    "plan-reference",
                )
                require(type(attempt.get("valid")) is bool, "calibration-invalid")
                ok = _check_bw_checksum(
                    ex.get("checksum"),
                    op_plans[ref]["expected_checksum"],
                    op,
                    compact=True,
                )
                if attempt.get("valid"):
                    require(
                        ok and ex.get("valid") is True and ex.get("elapsed_seconds") is not None,
                        "calibration-valid-mismatch",
                    )
                _check_write_pattern(ex, op, True, seen_write_patterns)
                val_obj = ex.get("validation")
                require(isinstance(val_obj, dict), "validation-invalid")
                check_validation_checks(
                    val_obj.get("checks"),
                    _validation_specs_for_op(op),
                    bool(attempt.get("valid")),
                )
                cal_timing = ex.get("timing")
                require(isinstance(cal_timing, dict), "timing-invalid")
                levels.append(verify_cpu_raw_timing(cal_timing.get("cpu_raw"), ex.get("elapsed_seconds")))

    per_loop_count = len(active_targets) * len(OPERATIONS)
    equal(len(rows), loop_count * per_loop_count, "measurement-count-mismatch")

    loop_records = doc.get("loop_records")
    require(isinstance(loop_records, list) and len(loop_records) == loop_count, "loop-count-mismatch")

    accepted = {t: {op: [] for op in OPERATIONS} for t in active_targets}
    stopped = False
    num_targets = len(active_targets)

    for i, row in enumerate(rows):
        budget(1)
        require(isinstance(row, dict), "measurement-invalid")
        integer(row.get("measurement_id"), maximum=MAX_SAFE_INT)
        equal(row.get("measurement_id"), i, "measurement-id-mismatch")
        loop, within_loop = divmod(i, per_loop_count)
        t_pos, o_pos = divmod(within_loop, len(OPERATIONS))
        expected_target = active_targets[(loop + t_pos) % num_targets]
        expected_op = OPERATIONS[(loop + o_pos) % len(OPERATIONS)]
        equal(row.get("loop_index"), loop, "measurement-order-mismatch")
        equal(row.get("target_order_position"), t_pos, "measurement-order-mismatch")
        equal(row.get("operation_order_position"), o_pos, "measurement-order-mismatch")
        equal(row.get("target"), expected_target, "measurement-order-mismatch")
        equal(row.get("operation"), expected_op, "measurement-order-mismatch")

        status = row.get("status")
        require(
            status in ("measured", "invalid", "failed", "interrupted", "not_run"),
            "measurement-status",
        )
        attempted = row.get("attempted")
        require(type(attempted) is bool, "measurement-attempted")
        if attempted:
            require(not stopped, "invalid-execution-prefix")
        if status != "measured":
            stopped = True

        ref = row.get("plan_ref")
        plan = None
        if ref is not None:
            integer(ref, maximum=MAX_SAFE_INT)
            require(
                ref < len(op_plans)
                and op_plans[ref]["target"] == expected_target
                and op_plans[ref]["operation"] == expected_op,
                "plan-reference",
            )
            plan = op_plans[ref]
            equal(ref, frozen_refs[expected_target][expected_op], "frozen-plan-reference")

        ex = row.get("execution")
        require(isinstance(ex, dict), "execution-invalid")
        timing_obj = ex.get("timing")
        require(isinstance(timing_obj, dict), "timing-invalid")
        elapsed = (
            timing_obj.get("elapsed_seconds")
            if status == "measured"
            else timing_obj.get("diagnostic_elapsed_seconds")
        )

        if attempted:
            require(plan is not None, "plan-reference")
            checksum_ok = _check_bw_checksum(
                row.get("checksum"),
                plan["expected_checksum"],
                expected_op,
                compact=False,
            )
            levels.append(verify_cpu_raw_timing(timing_obj.get("cpu_raw"), elapsed))
            _check_write_pattern(ex, expected_op, True, seen_write_patterns)
        else:
            require(timing_obj.get("elapsed_seconds") is None, "metric-null")
            require(ex.get("valid") is None, "execution-null")
            require(row.get("bandwidth_gb_s") is None, "metric-null")
            equal(row.get("completed_passes"), 0, "completion-mismatch")
            equal(row.get("completed_exact_payload_bytes"), "0", "completion-mismatch")
            ck = row.get("checksum")
            require(isinstance(ck, dict), "checksum-invalid")
            require(
                ck.get("status") in ("not_evaluated", "not_applicable")
                and ck.get("checksum_valid") is None
                and ck.get("actual_run_checksum") is None
                and ck.get("actual_worker_checksums") is None,
                "checksum-null",
            )
            _check_write_pattern(ex, expected_op, False, seen_write_patterns)
            checksum_ok = False

        val_obj = ex.get("validation")
        require(isinstance(val_obj, dict), "validation-invalid")
        check_validation_checks(
            val_obj.get("checks"),
            _validation_specs_for_op(expected_op),
            status == "measured",
        )

        if status == "measured":
            require(
                attempted
                and checksum_ok
                and ex.get("valid") is True
                and timing_obj.get("evaluated") is True
                and timing_obj.get("valid") is True,
                "measurement-valid-mismatch",
            )
            require(
                type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed > 0,
                "elapsed-invalid",
            )
            equal(row.get("completed_passes"), plan["passes"], "completion-mismatch")
            equal(
                row.get("completed_exact_payload_bytes"),
                plan["exact_payload_bytes"],
                "completion-mismatch",
            )
            exact_bytes = integer(plan["exact_payload_bytes"], decimal=True)
            close(row.get("bandwidth_gb_s"), exact_bytes / elapsed / 1e9, "rate-mismatch")
            exp_workers = target_info[expected_target]["effective_workers"]
            equal(ex.get("requested_workers"), cfg["requested_workers"], "worker-lifecycle")
            equal(ex.get("created_workers"), exp_workers, "worker-lifecycle")
            equal(ex.get("completed_workers"), exp_workers, "worker-lifecycle")
            accepted[expected_target][expected_op].append(i)
        else:
            require(row.get("bandwidth_gb_s") is None, "metric-null")

    for loop, record in enumerate(loop_records):
        require(isinstance(record, dict), "loop-order-mismatch")
        equal(record.get("loop_index"), loop, "loop-order-mismatch")
        planned_targets = [active_targets[(loop + k) % num_targets] for k in range(num_targets)]
        planned_ops = [OPERATIONS[(loop + k) % len(OPERATIONS)] for k in range(len(OPERATIONS))]
        planned_order = [
            {"target": t, "operation": o} for t in planned_targets for o in planned_ops
        ]
        equal(record.get("planned_order"), planned_order, "loop-order-mismatch")
        start_idx = loop * per_loop_count
        attempted_order = [
            {"target": r["target"], "operation": r["operation"]}
            for r in rows[start_idx : start_idx + per_loop_count]
            if r["attempted"]
        ]
        equal(record.get("realized_order"), attempted_order, "loop-prefix-mismatch")
        equal(record.get("realized_order_count"), len(attempted_order), "loop-prefix-mismatch")
        equal(
            record.get("measurement_indexes"),
            list(range(start_idx, start_idx + per_loop_count)),
            "loop-measurement-reference",
        )

    aggs = doc.get("aggregates")
    require(isinstance(aggs, dict) and isinstance(aggs.get("targets"), dict), "aggregate-invalid")
    equal(set(aggs["targets"].keys()), set(active_targets), "aggregate-population-mismatch")
    for tname in active_targets:
        t_agg = aggs["targets"][tname]
        require(isinstance(t_agg, dict) and set(t_agg.keys()) == set(OPERATIONS), "aggregate-population-mismatch")
        for op in OPERATIONS:
            a = t_agg[op]
            require(isinstance(a, dict), "aggregate-invalid")
            ids = accepted[tname][op]
            equal(a.get("accepted_measurement_ids"), ids, "aggregate-population-mismatch")
            values = [rows[idx]["bandwidth_gb_s"] for idx in ids]
            expected_stats = compute_statistics(values)
            n = len(ids)
            if expected_stats is None:
                require(a.get("value") is None and a.get("statistics") is None, "aggregate-null")
            else:
                close(a.get("value"), expected_stats["median"], "aggregate-headline-mismatch")
                check_statistics_block(a.get("statistics"), expected_stats)
            q = a.get("quality")
            require(isinstance(q, dict), "aggregate-quality-invalid")
            equal(q.get("repeatability_assessment_available"), n >= 2, "aggregate-quality-mismatch")
            equal(q.get("run_complete"), n == loop_count, "aggregate-quality-mismatch")
            close(q.get("cv_warning_threshold_pct"), CV_WARNING_THRESHOLD_PCT, "aggregate-cv-mismatch")
            cv_warn = (
                n >= 3
                and expected_stats is not None
                and expected_stats["coefficient_of_variation_pct"] is not None
                and expected_stats["coefficient_of_variation_pct"] > CV_WARNING_THRESHOLD_PCT
            )
            equal(q.get("cv_warning"), cv_warn, "aggregate-cv-mismatch")

    counters = doc.get("counters")
    require(isinstance(counters, dict), "counter-mismatch")
    measured_total = sum(
        len(accepted[t][op]) for t in active_targets for op in OPERATIONS
    )
    attempted_total = sum(1 for r in rows if r["attempted"])
    completed_loops = (
        sum(
            all(r["status"] == "measured" for r in rows[k : k + per_loop_count])
            for k in range(0, len(rows), per_loop_count)
        )
        if per_loop_count > 0
        else 0
    )
    expected_counters = {
        "planned_loops": loop_count,
        "attempted_loops": len({r["loop_index"] for r in rows if r["attempted"]}),
        "completed_loops": completed_loops,
        "planned_measurements": len(rows),
        "attempted_measurements": attempted_total,
        "terminal_measurements": sum(1 for r in rows if r["status"] != "not_run"),
        "measured_measurements": measured_total,
    }
    for key, val in expected_counters.items():
        equal(counters.get(key), val, "counter-mismatch")

    completed_passes = sum(integer(r.get("completed_passes"), maximum=MAX_SAFE_INT) for r in rows)
    completed_bytes = sum(integer(r.get("completed_exact_payload_bytes"), decimal=True) for r in rows)
    equal(counters.get("completed_passes"), str(completed_passes), "counter-mismatch")
    equal(counters.get("completed_exact_payload_bytes"), str(completed_bytes), "counter-mismatch")
    if all(frozen_refs[t][op] is not None for t in active_targets for op in OPERATIONS):
        planned_passes = sum(op_plans[frozen_refs[t][op]]["passes"] for t in active_targets for op in OPERATIONS) * loop_count
        planned_bytes = sum(
            integer(op_plans[frozen_refs[t][op]]["exact_payload_bytes"], decimal=True)
            for t in active_targets
            for op in OPERATIONS
        ) * loop_count
    else:
        planned_passes = 0
        planned_bytes = 0
    equal(counters.get("planned_passes"), str(planned_passes), "counter-mismatch")
    equal(counters.get("planned_exact_payload_bytes"), str(planned_bytes), "counter-mismatch")

    check_checkpoint_lifecycle(doc.get("checkpoint_lifecycle"), loop_count, doc.get("status"))
    accepted_run = check_run_acceptance(doc, measured_total, len(rows), attempted_total, rows)
    complete = len(rows) > 0 and measured_total == len(rows)
    equal(
        doc.get("operation_order_balance_complete"),
        complete and loop_count % len(OPERATIONS) == 0,
        "order-balance-mismatch",
    )
    equal(
        doc.get("target_order_balance_complete"),
        complete and num_targets > 0 and loop_count % num_targets == 0,
        "order-balance-mismatch",
    )

    expected_interp = {
        "dram_residency": "unverified",
        "cache_residency": "unverified",
        "non_cacheable_requested": cfg["non_cacheable"],
        "non_cacheable_is_best_effort_madvise": True,
        "single_pass_xor_collision_caveat": (
            "Per-pass read XOR detects single-bit and slice-address errors, "
            "but even-parity identical flips within one pass can cancel in 64-bit XOR."
        ),
        "write_copy_post_validation_verifies_final_state_not_intermediate_passes": True,
        "comparability_requires": EXPECTED_COMPARABILITY,
    }
    equal(doc.get("interpretation"), expected_interp, "interpretation-invalid")

    check_quality_warnings(
        doc.get("quality_warnings"),
        env=env,
        cfg=cfg,
        buffer_evidences=bw_buffer_evidences,
        all_targets_have_resources=(set(resources_obj.keys()) == set(active_targets) and len(active_targets) > 0),
        active_targets=active_targets,
        operations=OPERATIONS,
        agg_targets=aggs["targets"],
        attempted_any=attempted_any,
        iterations=iterations,
        rows=rows,
        duration_window=(0.100, 0.250),
    )

    enforce_require_raw_timing(require_raw, levels)
    manifest = doc.get("build_manifest")
    if binary_digest is None and binary is not None:
        binary_digest = hash_binary(binary)
    build_level = check_manifest(manifest, binary_digest=binary_digest)

    return dict(
        artifact_consistent=True,
        run_accepted=accepted_run,
        reason_code="valid" if accepted_run else "run-not-accepted",
        checks=dict(
            checksum="independent-oracle",
            timing=(
                "elapsed-only"
                if "elapsed-only" in levels
                else ("cpu-raw-ticks" if levels else "unavailable")
            ),
            build=build_level,
        ),
        oracle_work_used=work_before - budget.remaining,
    )


def _validate_bw_buffer_size(v):
    return integer(v, maximum=MAX_SAFE_INT)


def _validate_bw_cache_size(v):
    n = integer(v, maximum=1048576)
    require(n == 0 or n >= 16, "sweep-parameter-value-invalid")
    return n


def _validate_bw_threads(v):
    n = integer(v, maximum=U32)
    require(n >= 1, "sweep-parameter-value-invalid")
    return n


SWEEP_PARAM_SPECS = {
    "buffer-size": ("buffer_size_mb", _validate_bw_buffer_size),
    "cache-size": ("custom_cache_size_kb", _validate_bw_cache_size),
    "threads": ("requested_workers", _validate_bw_threads),
}

SWEEP_RESOLVED_SOURCE_MAP = {
    "buffer-size": "buffer_size",
    "cache-size": "cache_size",
    "threads": "workers",
}

FIXED_CONFIG_KEYS = (
    "available_workers",
    "worker_source",
    "iterations",
    "work_policy",
    "loop_count",
    "non_cacheable",
    "base_seed_uint64_decimal",
    "seed_source",
    "argv",
    "resolved_sources",
)


def verify_sweep(doc, binary=None, require_raw=False, budget=None):
    """Verify a cpu_bandwidth schema-1 sweep envelope."""
    return verify_sweep_envelope(
        doc,
        base_mode=MODE_NAME,
        param_specs=SWEEP_PARAM_SPECS,
        verify_single_fn=verify_single,
        fixed_config_keys=FIXED_CONFIG_KEYS,
        binary=binary,
        require_raw=require_raw,
        budget=budget,
        sweep_resolved_source_map=SWEEP_RESOLVED_SOURCE_MAP,
    )


def verify(doc, binary=None, require_raw=False, budget=None):
    """Verify either a direct cpu_bandwidth artifact or a cpu_bandwidth sweep envelope."""
    require(isinstance(doc, dict), "invalid-input")
    cfg = doc.get("configuration")
    if isinstance(cfg, dict) and cfg.get("mode") == "sweep" and cfg.get("base_mode") == MODE_NAME:
        return verify_sweep(doc, binary=binary, require_raw=require_raw, budget=budget)
    return verify_single(doc, binary=binary, require_raw=require_raw, budget=budget)


def _verify_cli(doc, args):
    return verify(doc, binary=args.binary, require_raw=args.require_raw_timing)


def main():
    return run_verifier_cli(_verify_cli, __doc__)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Verify one patterns schema-1 artifact or schema-1 sweep envelope
independently of the C++ producer.

Usage:
  python3 script-examples/verify_patterns_result.py [result.json|-] [--binary memory_benchmark] [--require-raw-timing] [--max-random-indices N]
Exit 0: consistent accepted artifact; 1: inconsistent or unaccepted; 2: unsupported.
"""

import hashlib
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
    check_buffer_evidence,
    check_checkpoint_lifecycle,
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
    fisher_yates_rejection_shuffle,
    hash_binary,
    integer,
    partition_64b,
    require,
    rotl64,
    run_verifier_cli,
    splitmix64_bounded_draw,
    splitmix64_mix,
    verify_cpu_raw_timing,
    verify_sweep_envelope,
)

SCHEMA_VERSION = 1
MODE_NAME = "patterns"
METHODOLOGY_VERSION = "patterns-v1-arm64-phase-calibrated-seeded-balanced"

EXPECTED_COMPONENTS = {
    "version": "patterns-components-v1",
    "work_plan_version": "patterns-work-plan-v1",
    "buffer_pattern_version": "patterns-buffer-pattern-v1",
    "read_checksum_version": "patterns-read-checksum-v1",
    "random_index_generator_version": "splitmix64-coprime-fisher-yates-v1",
    "run_policy_version": "bounded-loop-snapshots-v1",
}

EXPECTED_RESOLVED_SOURCE_TOKENS = {
    "buffer_size": ("explicit", "default", "sweep"),
    "workers": ("explicit", "auto", "sweep"),
    "iterations": ("explicit", "auto"),
    "seed": ("explicit", "generated"),
}

PATTERNS_FLAG_MAP = {
    "buffer_size": ("-b", "--buffer-size"),
    "workers": ("-t", "--threads"),
    "iterations": ("-i", "--iterations"),
    "seed": ("--seed",),
}

EXPECTED_COMPARABILITY = [
    "same-methodology-version",
    "same-frozen-work-plan-and-patterns",
    "same-software-version",
    "sufficiently-similar-hardware",
    "sufficiently-similar-thermal-and-power-state",
]

CANONICAL_PATTERNS = (
    "sequential_forward",
    "sequential_reverse",
    "strided_64",
    "strided_4096",
    "strided_16384",
    "strided_2mb",
    "random",
)
OPERATIONS = ("read", "write", "copy")
STRIDED_PATTERNS = ("strided_64", "strided_4096", "strided_16384", "strided_2mb")
STREAMING_PATTERNS = ("sequential_forward", "sequential_reverse", "strided_64")

STRIDE_BYTES = {
    "sequential_forward": 0,
    "sequential_reverse": 0,
    "strided_64": 64,
    "strided_4096": 4096,
    "strided_16384": 16384,
    "strided_2mb": 2097152,
    "random": 0,
}

DOMAIN_SRC = 0x5041545F5352435F
DOMAIN_RANDOM = 0x5041545F52414E44
DOMAIN_PRE_OP_SENTINEL = 0x5041545F53454E54

ACCESS_SIZE_BYTES = 32
MIN_RANDOM_ACCESSES = 1000
MAX_RANDOM_ACCESSES = 1_000_000
DEFAULT_MAX_RANDOM_INDICES = 1_000_000
MAX_PASSES = 1_000_000_000
MAX_EXACT_PAYLOAD_BYTES = 64 * 1024 * 1024 * 1024
SKIPPED_BUFFER_TOO_SMALL = "buffer-too-small-for-stride-transition"

ALLOWED_CALIBRATION_PURPOSES = (
    "calibration_shape_warmup",
    "pilot",
    "correction",
    "single_unit_confirmation_warmup",
    "single_unit_confirmation",
    "frozen_measurement_warmup",
)
CALIBRATION_MIN_SECONDS = 0.100
CALIBRATION_MAX_SECONDS = 0.250

STREAMING_CV_WARNING_PCT = 5.0
SPARSE_CV_WARNING_PCT = 10.0

EXPECTED_TOP_LEVEL_KEYS = (
    "schema_version",
    "mode",
    "methodology_version",
    "software",
    "build_manifest",
    "configuration",
    "resolved_plan",
    "memory_budget",
    "calibration",
    "measurements",
    "aggregates",
    "status",
    "reason_code",
    "results_complete",
    "run_accepted",
    "interpretation",
    "diagnostic",
    "interruption_requested",
    "operation_order_balance_complete",
    "pattern_order_balance_complete",
    "seeds",
    "counters",
    "checkpoint_lifecycle",
    "loop_records",
    "environment",
    "quality_warnings",
)


def _even_half_s1(passes):
    if (passes & 1) == 0:
        return ((passes >> 1) * (passes + 1)) & U64
    return (passes * ((passes + 1) >> 1)) & U64


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


def lane_mix(idx):
    return (idx ^ rotl64(idx, 7) ^ rotl64(idx, 33)) & U64


def slot_mix(slot_idx):
    return (slot_idx ^ rotl64(slot_idx, 19) ^ rotl64(slot_idx, 41)) & U64


class PatternsTable:
    """Slot-tagged GF(2) pattern table (`patterns-buffer-pattern-v1`)."""

    def __init__(self, src_seed):
        self.table = [splitmix64_mix(src_seed ^ k) for k in range(256)]
        self.prefix_t = [0] * 257
        for k in range(256):
            self.prefix_t[k + 1] = self.prefix_t[k] ^ self.table[k]

        self.s_t = [
            self.table[4 * u]
            ^ self.table[4 * u + 1]
            ^ self.table[4 * u + 2]
            ^ self.table[4 * u + 3]
            for u in range(64)
        ]
        self.prefix_st = [0] * 65
        for u in range(64):
            self.prefix_st[u + 1] = self.prefix_st[u] ^ self.s_t[u]

        self.prefix_st_even = [0] * 33
        self.prefix_st_odd = [0] * 33
        for k in range(32):
            self.prefix_st_even[k + 1] = self.prefix_st_even[k] ^ self.s_t[2 * k]
            self.prefix_st_odd[k + 1] = self.prefix_st_odd[k] ^ self.s_t[2 * k + 1]

    def word(self, word_idx):
        val = self.table[word_idx & 255] ^ lane_mix(word_idx)
        if (word_idx & 3) == 0:
            val ^= slot_mix(word_idx >> 2)
        return val & U64

    def slot_xor(self, slot_idx):
        return (self.s_t[slot_idx & 63] ^ slot_mix(slot_idx)) & U64

    def prefix_t_mod256(self, n):
        return (self.prefix_t[256] if ((n >> 8) & 1) else 0) ^ self.prefix_t[n & 255]

    def prefix_st_mod64(self, n):
        return (self.prefix_st[64] if ((n >> 6) & 1) else 0) ^ self.prefix_st[n & 63]

    def prefix_st_even_mod32(self, n):
        return (self.prefix_st_even[32] if ((n >> 5) & 1) else 0) ^ self.prefix_st_even[n & 31]

    def prefix_st_odd_mod32(self, n):
        return (self.prefix_st_odd[32] if ((n >> 5) & 1) else 0) ^ self.prefix_st_odd[n & 31]


def _xor_words_range(table, w_start, w_end):
    if w_end <= w_start:
        return 0
    range_t = table.prefix_t_mod256(w_end) ^ table.prefix_t_mod256(w_start)
    lx = prefix_xor_0_to_n_minus_1(w_end) ^ prefix_xor_0_to_n_minus_1(w_start)
    range_lane = lx ^ rotl64(lx, 7) ^ rotl64(lx, 33)
    s_start = (w_start + 3) >> 2
    s_end = (w_end + 3) >> 2
    sx = prefix_xor_0_to_n_minus_1(s_end) ^ prefix_xor_0_to_n_minus_1(s_start)
    range_slot = slot_mix(sx)
    return (range_t ^ range_lane ^ range_slot) & U64


def _xor_slots_range(table, s_start, s_end):
    if s_end <= s_start:
        return 0
    range_st = table.prefix_st_mod64(s_end) ^ table.prefix_st_mod64(s_start)
    sx = prefix_xor_0_to_n_minus_1(s_end) ^ prefix_xor_0_to_n_minus_1(s_start)
    return (range_st ^ slot_mix(sx)) & U64


def _xor_vec_bytes_32_aligned(table, byte_start, vec_bytes):
    if vec_bytes == 0:
        return 0
    k = vec_bytes >> 3
    w0 = byte_start >> 3
    r = byte_start & 7
    x0 = _xor_words_range(table, w0, w0 + k)
    if r == 0:
        return x0
    x1 = x0 ^ table.word(w0) ^ table.word(w0 + k)
    return ((x0 >> (8 * r)) | (x1 << (64 - 8 * r))) & U64


def pass_xor_sequential_forward(table, offset_bytes, span_bytes):
    """Exact O(1) sequential_forward pass XOR matching ARM64 read kernel."""
    if span_bytes == 0:
        return 0
    vec_bytes = span_bytes - (span_bytes % ACCESS_SIZE_BYTES)
    vec_xor = _xor_vec_bytes_32_aligned(table, offset_bytes, vec_bytes)
    tail_xor = 0
    for pos in range(offset_bytes + vec_bytes, offset_bytes + span_bytes):
        w_val = table.word(pos >> 3)
        tail_xor ^= (w_val >> (8 * (pos & 7))) & 0xFF
    return (vec_xor ^ tail_xor) & U64


def pass_xor_sequential_reverse(table, offset_bytes, span_bytes):
    """Exact O(1) sequential_reverse pass XOR matching ARM64 read kernel."""
    if span_bytes == 0:
        return 0
    tail_bytes = span_bytes % ACCESS_SIZE_BYTES
    vec_bytes = span_bytes - tail_bytes
    tail_end = offset_bytes + tail_bytes
    tail_xor = 0
    for pos in range(offset_bytes, tail_end):
        w_val = table.word(pos >> 3)
        tail_xor ^= (w_val >> (8 * (pos & 7))) & 0xFF
    vec_xor = _xor_vec_bytes_32_aligned(table, tail_end, vec_bytes)
    return (vec_xor ^ tail_xor) & U64


def strided_full_cycle_xor(table, offset_bytes, span_bytes):
    """Exact O(1) full Q-phase strided XOR across [offset_bytes, offset_bytes + span_bytes)."""
    if span_bytes < ACCESS_SIZE_BYTES or (offset_bytes % ACCESS_SIZE_BYTES) != 0:
        return 0
    base_slot = offset_bytes >> 5
    num_slots = span_bytes >> 5
    return _xor_slots_range(table, base_slot, base_slot + num_slots)


def strided_partial_cycle_xor(table, offset_bytes, span_bytes, stride_bytes, rem_phases):
    """Exact O(1) 2D carry-split partial-cycle strided XOR."""
    if (
        span_bytes < ACCESS_SIZE_BYTES
        or (offset_bytes % ACCESS_SIZE_BYTES) != 0
        or stride_bytes < 64
        or (stride_bytes % ACCESS_SIZE_BYTES) != 0
        or rem_phases == 0
    ):
        return 0
    q_period = stride_bytes >> 5
    if rem_phases >= q_period:
        return strided_full_cycle_xor(table, offset_bytes, span_bytes)
    if (q_period & (q_period - 1)) != 0:
        return 0
    q_shift = (q_period & -q_period).bit_length() - 1

    base_slot = offset_bytes >> 5
    num_slots = span_bytes >> 5
    full_blocks = num_slots >> q_shift
    tail_slots = num_slots & (q_period - 1)
    rem = rem_phases

    high_0 = base_slot >> q_shift
    low_0 = base_slot & (q_period - 1)
    rem_1 = min(rem, q_period - low_0)
    rem_2 = rem - rem_1

    def xor_rect(h_start, h_end, l_start, l_end):
        if h_end <= h_start or l_end <= l_start:
            return 0
        n_h = h_end - h_start
        n_l = l_end - l_start
        out = 0
        if (n_l & 1) != 0:
            hx = prefix_xor_0_to_n_minus_1(h_end) ^ prefix_xor_0_to_n_minus_1(h_start)
            out ^= slot_mix(hx << q_shift)
        if (n_h & 1) != 0:
            lx = prefix_xor_0_to_n_minus_1(l_end) ^ prefix_xor_0_to_n_minus_1(l_start)
            out ^= slot_mix(lx)
        if q_shift >= 6:
            if (n_h & 1) != 0:
                out ^= table.prefix_st_mod64(l_end) ^ table.prefix_st_mod64(l_start)
        elif q_shift == 1:
            if l_start == 0:
                out ^= table.prefix_st_even_mod32(h_end) ^ table.prefix_st_even_mod32(h_start)
            else:
                out ^= table.prefix_st_odd_mod32(h_end) ^ table.prefix_st_odd_mod32(h_start)
        else:
            for h in range(h_start, h_end):
                for l_val in range(l_start, l_end):
                    out ^= table.s_t[((h << q_shift) | l_val) & 63]
        return out & U64

    acc = 0
    if full_blocks > 0:
        acc ^= xor_rect(high_0, high_0 + full_blocks, low_0, low_0 + rem_1)
        if rem_2 > 0:
            acc ^= xor_rect(high_0 + 1, high_0 + 1 + full_blocks, 0, rem_2)
    tail_count = min(tail_slots, rem)
    if tail_count > 0:
        tail_base = base_slot + (full_blocks << q_shift)
        acc ^= _xor_slots_range(table, tail_base, tail_base + tail_count)
    return acc & U64


def distribute_random_accesses(worker_spans, total_selected):
    """Exact proportional random access distribution matching C++ build_patterns_work_plan."""
    workers = len(worker_spans)
    if workers == 0 or total_selected < workers:
        return [0] * workers
    span_slots_list = [sp_len // ACCESS_SIZE_BYTES for _, sp_len in worker_spans]
    total_slots = sum(span_slots_list)
    if total_slots < total_selected:
        return [0] * workers

    remaining = total_selected - workers
    sum_capacity = total_slots - workers
    counts = []
    distributed = 0
    for span_slots in span_slots_list:
        cap = (span_slots - 1) if span_slots > 0 else 0
        add = 0
        if remaining > 0 and sum_capacity > 0:
            add = min(cap, (remaining * cap) // sum_capacity)
        cnt = 1 + add
        counts.append(cnt)
        distributed += cnt

    leftover = total_selected - distributed
    while leftover > 0:
        progressed = False
        for i, span_slots in enumerate(span_slots_list):
            if leftover == 0:
                break
            if counts[i] < span_slots:
                counts[i] += 1
                leftover -= 1
                progressed = True
        if not progressed:
            break
    return counts


def generate_random_workload(random_domain_seed, table, worker_spans, worker_access_counts, budget=None):
    """Reconstruct splitmix64-coprime-fisher-yates-v1 indices, SHA-256, LWS, and pass XORs."""
    hasher = hashlib.sha256()
    worker_pass_xors = []
    has_any = False
    min_global = None
    max_global = 0

    for w, ((off_bytes, span_bytes), count_w) in enumerate(zip(worker_spans, worker_access_counts)):
        if budget is not None:
            budget(1)
        require(off_bytes % ACCESS_SIZE_BYTES == 0, "worker-alignment")
        span_slots_w = span_bytes // ACCESS_SIZE_BYTES
        require(count_w <= span_slots_w, "random-workload-mismatch")

        state = splitmix64_mix(random_domain_seed ^ w)
        if count_w > 0:
            if count_w == span_slots_w:
                local_slots = list(range(count_w))
            else:
                start, state = splitmix64_bounded_draw(state, span_slots_w, budget=budget)
                step_draw, state = splitmix64_bounded_draw(state, span_slots_w - 1, budget=budget)
                step = step_draw + 1
                while math.gcd(step, span_slots_w) != 1:
                    step = 1 if step + 1 >= span_slots_w else step + 1
                local_slots = [(k * step + start) % span_slots_w for k in range(count_w)]
            state = fisher_yates_rejection_shuffle(local_slots, state, budget=budget)
            indices = [slot * ACCESS_SIZE_BYTES for slot in local_slots]
            min_local = min(indices)
            max_local = max(indices)
            g_min = off_bytes + min_local
            g_max = off_bytes + max_local
            min_global = g_min if min_global is None else min(min_global, g_min)
            max_global = max(max_global, g_max)
            has_any = True
        else:
            indices = []

        hasher.update(w.to_bytes(8, "little"))
        hasher.update(count_w.to_bytes(8, "little"))
        base_slot = off_bytes >> 5
        px = 0
        for rel_off in indices:
            hasher.update(rel_off.to_bytes(8, "little"))
            px ^= table.slot_xor(base_slot + (rel_off >> 5))
        worker_pass_xors.append(px & U64)

    logical_ws = (max_global - min_global + ACCESS_SIZE_BYTES) if has_any else 0
    return {
        "random_indices_sha256": hasher.hexdigest(),
        "logical_working_set_bytes": logical_ws,
        "worker_pass_xors": worker_pass_xors,
    }


def fold_patterns_workers(worker_tuples):
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


def expected_read_checksum(
    table,
    op_seed,
    pattern_name,
    stride_bytes,
    passes,
    worker_spans,
    random_worker_info=None,
    budget=None,
):
    """Closed-form O(1) patterns-read-checksum-v1 oracle per worker and run fold."""
    if budget is not None:
        budget(len(worker_spans))
    s1_passes = _even_half_s1(passes)
    workers_out = []
    tuples = []

    q_period = (stride_bytes // ACCESS_SIZE_BYTES) if stride_bytes > 0 else 0
    for w, (offset, span) in enumerate(worker_spans):
        init_a = splitmix64_mix(op_seed ^ (0xA100000000000000 | w))
        init_b = splitmix64_mix(op_seed ^ (0xB200000000000000 | w))

        if pattern_name in ("sequential_forward", "sequential_reverse"):
            px = (
                pass_xor_sequential_forward(table, offset, span)
                if pattern_name == "sequential_forward"
                else pass_xor_sequential_reverse(table, offset, span)
            )
            state_a = (init_a + passes * px + SPLITMIX_A * s1_passes) & U64
            state_b = (init_b + ((px ^ SPLITMIX_B) * s1_passes) + SPLITMIX_C * passes) & U64
            exact_bytes = passes * span
        elif pattern_name == "random":
            require(random_worker_info is not None, "missing-random-workload")
            px = random_worker_info["worker_pass_xors"][w]
            count_w = random_worker_info["worker_counts"][w]
            state_a = (init_a + passes * px + SPLITMIX_A * s1_passes) & U64
            state_b = (init_b + ((px ^ SPLITMIX_B) * s1_passes) + SPLITMIX_C * passes) & U64
            exact_bytes = passes * count_w * ACCESS_SIZE_BYTES
        else:
            full_cycles = passes // q_period
            rem_phases = passes % q_period
            fc_xor = strided_full_cycle_xor(table, offset, span)
            s1_fc = _even_half_s1(full_cycles)
            state_a = init_a + full_cycles * fc_xor + SPLITMIX_A * s1_fc
            state_b = init_b + ((fc_xor ^ SPLITMIX_B) * s1_fc) + SPLITMIX_C * full_cycles
            if rem_phases > 0:
                pc_xor = strided_partial_cycle_xor(table, offset, span, stride_bytes, rem_phases)
                step = full_cycles + 1
                state_a += pc_xor + SPLITMIX_A * step
                state_b += ((pc_xor ^ SPLITMIX_B) * step) + SPLITMIX_C
            state_a &= U64
            state_b &= U64
            num_slots = span // ACCESS_SIZE_BYTES
            full_blocks = num_slots // q_period
            tail_slots = num_slots % q_period
            total_slots_read = (
                full_cycles * num_slots
                + full_blocks * rem_phases
                + min(tail_slots, rem_phases)
            )
            exact_bytes = total_slots_read * ACCESS_SIZE_BYTES

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

    return workers_out, fold_patterns_workers(tuples)


def _select_strided_workers(buffer_bytes, stride_bytes, req_workers, avail_workers):
    min_worker_span = stride_bytes + ACCESS_SIZE_BYTES
    if buffer_bytes < min_worker_span:
        return 0, []
    max_by_span = buffer_bytes // min_worker_span
    first_candidate = min(req_workers, avail_workers, max_by_span)
    for candidate in range(first_candidate, 0, -1):
        spans = partition_64b(buffer_bytes, candidate)
        if len(spans) != candidate:
            continue
        ok = True
        for off, sp_len in spans:
            if (
                sp_len < min_worker_span
                or (off % ACCESS_SIZE_BYTES) != 0
                or (sp_len % ACCESS_SIZE_BYTES) != 0
            ):
                ok = False
                break
            if 1 + (sp_len - ACCESS_SIZE_BYTES) // stride_bytes < 2:
                ok = False
                break
        if ok:
            return candidate, spans
    return 0, []


def _validation_specs_for_op(op, has_untouched_dst_bytes):
    return [
        ("post-validation-structure", True),
        ("read-checksum", op == "read"),
        ("write-final-touched-bytes", op == "write"),
        ("copy-final-touched-bytes", op == "copy"),
        (
            "untouched-sentinel-isolation",
            op in ("write", "copy") and has_untouched_dst_bytes,
        ),
        ("src-readonly-protection", op in ("read", "copy")),
    ]


def _check_patterns_checksum(observed, expected, op, compact=False):
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
        equal(observed.get("actual_run_checksum"), fold_patterns_workers(tuples), "actual-run-fold-mismatch")
    equal(observed.get("checksum_valid"), matches, "checksum-valid-mismatch")
    equal(status, "valid" if matches else "invalid", "checksum-status")
    return matches


def _check_write_pattern_and_sentinel(
    ex,
    op,
    attempted,
    has_untouched_dst_bytes,
    seen_patterns,
    require_sentinel_when_needed=False,
):
    wp = ex.get("write_pattern_uint64_decimal")
    write_pattern = None
    if op != "write" or not attempted:
        require(wp is None, "write-pattern-unexpected")
        require(ex.get("attempt_seed_uint64_decimal") is None, "write-pattern-unexpected")
    else:
        try:
            write_pattern = integer(wp, decimal=True)
        except Rejected as err:
            raise Rejected("write-pattern-invalid") from err
        require(write_pattern != 0 and (write_pattern & 1) == 1, "write-pattern-invalid")
        attempt_seed = ex.get("attempt_seed_uint64_decimal")
        if attempt_seed is not None:
            seed_val = integer(attempt_seed, decimal=True)
            equal(write_pattern, splitmix64_mix(seed_val) | 1, "write-pattern-invalid")
        require(write_pattern not in seen_patterns, "write-pattern-invalid")
        seen_patterns.add(write_pattern)

    needs_sentinel = attempted and (
        op == "copy" or (op == "write" and has_untouched_dst_bytes)
    )
    sent_raw = ex.get("pre_op_sentinel_uint64_decimal")
    if not needs_sentinel:
        require(sent_raw is None, "pre-op-sentinel-unexpected")
        return
    if sent_raw is None:
        require(not require_sentinel_when_needed, "pre-op-sentinel-invalid")
        return
    try:
        sentinel = integer(sent_raw, decimal=True)
    except Rejected as err:
        raise Rejected("pre-op-sentinel-invalid") from err
    require(sentinel != 0 and (sentinel & 1) == 1, "pre-op-sentinel-invalid")
    if write_pattern is not None:
        require(sentinel != write_pattern, "pre-op-sentinel-invalid")


def _compute_operation_plan_expected(pattern_name, pattern_entry, op, passes, buffer_bytes, random_lws):
    if pattern_name in ("sequential_forward", "sequential_reverse"):
        accesses_per_pass = pattern_entry["accesses_per_pass"]
        min_app = accesses_per_pass
        max_app = accesses_per_pass
        total_accesses = passes * accesses_per_pass
        distinct_addrs = accesses_per_pass
        logical_ws = buffer_bytes
        completed_phase_cycles = 0
    elif pattern_name == "random":
        accesses_per_pass = pattern_entry["accesses_per_pass"]
        min_app = accesses_per_pass
        max_app = accesses_per_pass
        total_accesses = passes * accesses_per_pass
        distinct_addrs = accesses_per_pass
        logical_ws = random_lws
        completed_phase_cycles = 0
    else:
        stride = STRIDE_BYTES[pattern_name]
        q_period = stride // ACCESS_SIZE_BYTES
        completed_phase_cycles = passes // q_period
        remaining_phases = passes % q_period
        executed_phase_count = min(passes, q_period)
        last_executed_phase = executed_phase_count - 1
        accesses_per_pass = pattern_entry["accesses_per_pass"]
        min_app = 0
        max_app = 0
        total_accesses = 0
        distinct_addrs = 0
        logical_ws = 0
        for _, sp_len in pattern_entry["spans"]:
            aligned_slots = sp_len // ACCESS_SIZE_BYTES
            full_blocks = aligned_slots // q_period
            tail_slots = aligned_slots % q_period
            max_app += 1 + (aligned_slots - 1) // q_period
            min_app += (
                1 + (aligned_slots - 1 - last_executed_phase) // q_period
                if last_executed_phase < aligned_slots
                else 0
            )
            total_accesses += (
                completed_phase_cycles * aligned_slots
                + full_blocks * remaining_phases
                + min(tail_slots, remaining_phases)
            )
            distinct_addrs += (
                full_blocks * executed_phase_count
                + min(tail_slots, executed_phase_count)
            )
            selected_tail = min(tail_slots, executed_phase_count)
            if selected_tail > 0:
                last_slot = full_blocks * q_period + selected_tail - 1
                logical_ws += (last_slot + 1) * ACCESS_SIZE_BYTES
            elif full_blocks > 0 and executed_phase_count > 0:
                last_slot = (full_blocks - 1) * q_period + executed_phase_count - 1
                logical_ws += (last_slot + 1) * ACCESS_SIZE_BYTES

    has_untouched = (distinct_addrs * ACCESS_SIZE_BYTES) < buffer_bytes
    mult = 2 if op == "copy" else 1
    exact_payload = total_accesses * ACCESS_SIZE_BYTES * mult
    return {
        "total_accesses": total_accesses,
        "exact_payload_bytes": exact_payload,
        "accesses_per_pass": accesses_per_pass,
        "min_accesses_per_pass": min_app,
        "max_accesses_per_pass": max_app,
        "distinct_address_count": distinct_addrs,
        "logical_working_set_bytes": logical_ws,
        "completed_phase_cycles": completed_phase_cycles,
        "has_untouched_dst_bytes": has_untouched,
    }


def verify_single(
    doc,
    binary=None,
    require_raw=False,
    budget=None,
    binary_digest=None,
    max_random_indices=DEFAULT_MAX_RANDOM_INDICES,
):
    """Verify a single patterns schema-1 artifact."""
    require(isinstance(doc, dict), "invalid-input")
    if type(doc.get("schema_version")) is not int or doc["schema_version"] != SCHEMA_VERSION:
        raise Rejected("unsupported-schema", True)
    if doc.get("mode") != MODE_NAME:
        raise Rejected("unsupported-mode", True)
    if doc.get("methodology_version") != METHODOLOGY_VERSION:
        raise Rejected("unsupported-methodology", True)

    require(
        all(k in doc for k in EXPECTED_TOP_LEVEL_KEYS),
        "missing-top-level-key",
    )

    if budget is None:
        budget = Budget()
    work_before = budget.remaining

    sw = doc.get("software")
    require(
        isinstance(sw, dict)
        and isinstance(sw.get("version"), str)
        and len(sw["version"]) > 0
        and isinstance(sw.get("timestamp"), str)
        and len(sw["timestamp"]) > 0,
        "software-invalid",
    )
    diag = doc.get("diagnostic")
    require(diag is None or (isinstance(diag, str) and len(diag) > 0), "diagnostic-invalid")

    cfg = doc.get("configuration")
    require(isinstance(cfg, dict), "configuration-invalid")
    buf_mb = integer(cfg.get("buffer_size_mb"), maximum=MAX_SAFE_INT)
    require(buf_mb >= 1, "configuration-invalid")
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
    require(
        isinstance(cfg.get("argv"), list) and all(isinstance(a, str) for a in cfg["argv"]),
        "configuration-invalid",
    )
    check_resolved_sources(
        cfg.get("resolved_sources"),
        cfg,
        EXPECTED_RESOLVED_SOURCE_TOKENS,
        PATTERNS_FLAG_MAP,
    )

    env = doc.get("environment")
    check_environment(env)
    page_size = integer(env["page_size_bytes"], decimal=True)

    seeds = doc.get("seeds")
    require(isinstance(seeds, dict), "seed-mismatch")
    base_seed = integer(seeds.get("base_seed_uint64_decimal"), decimal=True)
    equal(cfg.get("base_seed_uint64_decimal"), str(base_seed), "seed-mismatch")
    equal(seeds.get("source"), cfg.get("seed_source"), "seed-mismatch")
    domain_seeds = seeds.get("domain_seeds")
    require(isinstance(domain_seeds, dict), "seed-mismatch")
    expected_src_seed = splitmix64_mix(base_seed ^ DOMAIN_SRC)
    expected_random_seed = splitmix64_mix(base_seed ^ DOMAIN_RANDOM)
    expected_sentinel_seed = splitmix64_mix(base_seed ^ DOMAIN_PRE_OP_SENTINEL)
    equal(
        set(domain_seeds.keys()),
        {"src_uint64_decimal", "random_uint64_decimal", "pre_op_sentinel_uint64_decimal"},
        "seed-mismatch",
    )
    equal(domain_seeds.get("src_uint64_decimal"), str(expected_src_seed), "seed-mismatch")
    equal(domain_seeds.get("random_uint64_decimal"), str(expected_random_seed), "seed-mismatch")
    equal(domain_seeds.get("pre_op_sentinel_uint64_decimal"), str(expected_sentinel_seed), "seed-mismatch")

    table = PatternsTable(expected_src_seed)

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
        and rp["plan_identity"].startswith("patterns-work-plan-v1|"),
        "plan-identity-mismatch",
    )

    rows = doc.get("measurements")
    require(isinstance(rows, list), "measurement-invalid")
    if len(rows) > 100_000:
        raise Rejected("unsupported-measurement-limit", True)
    require(all(isinstance(r, dict) for r in rows), "measurement-invalid")
    attempted_any = any(r.get("attempted") is True for r in rows)
    doc_status = doc.get("status")
    require(
        isinstance(doc.get("reason_code"), str) and len(doc["reason_code"]) > 0,
        "run-status",
    )
    if not rp_valid:
        require(
            doc_status == "failed" and not attempted_any and rp_reason != "valid",
            "plan-invalid",
        )
        equal(doc.get("reason_code"), rp_reason, "plan-invalid")

    patterns_obj = rp.get("patterns")
    require(isinstance(patterns_obj, dict), "pattern-geometry-mismatch")
    equal(set(patterns_obj.keys()), set(CANONICAL_PATTERNS), "pattern-geometry-mismatch")

    resources_obj = rp.get("resources")
    require(isinstance(resources_obj, dict), "resource-evidence-invalid")

    buffer_bytes = buf_mb * 1048576
    page_rounded = ((buffer_bytes + page_size - 1) // page_size) * page_size

    active_patterns = []
    pattern_info = {}
    selected_random = 0
    random_lws = 0
    random_worker_info = None

    for pname in CANONICAL_PATTERNS:
        pdata = patterns_obj[pname]
        require(isinstance(pdata, dict) and type(pdata.get("enabled")) is bool, "pattern-geometry-mismatch")
        stride = STRIDE_BYTES[pname]
        expected_phase_period = (stride // ACCESS_SIZE_BYTES) if stride > 0 else 0
        expected_stride_eq_page = stride > 0 and page_size > 0 and stride == page_size
        expected_lp_status = "not-verified" if pname == "strided_2mb" else "not-applicable"

        equal(pdata.get("access_size_bytes"), ACCESS_SIZE_BYTES, "pattern-geometry-mismatch")
        equal(pdata.get("stride_bytes"), stride, "pattern-geometry-mismatch")
        equal(pdata.get("phase_period_passes"), expected_phase_period, "pattern-geometry-mismatch")
        equal(pdata.get("native_page_size_bytes"), page_size, "pattern-geometry-mismatch")
        equal(pdata.get("stride_equals_native_page_size"), expected_stride_eq_page, "pattern-geometry-mismatch")
        equal(pdata.get("large_page_backing_verified"), False, "pattern-geometry-mismatch")
        equal(pdata.get("large_page_backing_status"), expected_lp_status, "pattern-geometry-mismatch")

        if not rp_valid:
            equal(pdata.get("enabled"), False, "pattern-geometry-mismatch")
            equal(pdata.get("status"), "unavailable", "pattern-geometry-mismatch")
            equal(pdata.get("reason_code"), rp_reason, "pattern-geometry-mismatch")
            continue

        if pname in ("sequential_forward", "sequential_reverse"):
            require(
                pdata.get("enabled") is True
                and pdata.get("status") == "active"
                and pdata.get("reason_code") == "active",
                "pattern-geometry-mismatch",
            )
            max_aligned = max(1, buffer_bytes // 64)
            expected_eff = min(req_workers, avail_workers, max_aligned)
            expected_spans = partition_64b(buffer_bytes, expected_eff)
            expected_app_per_worker = [sp_len // ACCESS_SIZE_BYTES for _, sp_len in expected_spans]
            expected_rand_counts = [0] * expected_eff
        elif pname in STRIDED_PATTERNS:
            expected_eff, expected_spans = _select_strided_workers(
                buffer_bytes, stride, req_workers, avail_workers
            )
            if expected_eff == 0:
                require(
                    pdata.get("enabled") is False
                    and pdata.get("status") == "skipped"
                    and pdata.get("reason_code") == SKIPPED_BUFFER_TOO_SMALL
                    and pdata.get("effective_workers") == 0
                    and pdata.get("worker_spans") == [],
                    "pattern-geometry-mismatch",
                )
                continue
            require(
                pdata.get("enabled") is True
                and pdata.get("status") == "active"
                and pdata.get("reason_code") == "active",
                "pattern-geometry-mismatch",
            )
            expected_app_per_worker = [
                1 + (sp_len - ACCESS_SIZE_BYTES) // stride for _, sp_len in expected_spans
            ]
            expected_rand_counts = [0] * expected_eff
        else:
            # random
            require(
                pdata.get("enabled") is True
                and pdata.get("status") == "active"
                and pdata.get("reason_code") == "active",
                "pattern-geometry-mismatch",
            )
            total_slots = buffer_bytes // ACCESS_SIZE_BYTES
            clamped = min(MAX_RANDOM_ACCESSES, max(MIN_RANDOM_ACCESSES, total_slots))
            selected_random = min(total_slots, clamped)
            if selected_random > max_random_indices:
                raise Rejected("unsupported-random-index-limit", True)
            max_aligned = max(1, buffer_bytes // 64)
            expected_eff = min(req_workers, avail_workers, max_aligned, selected_random)
            expected_spans = partition_64b(buffer_bytes, expected_eff)
            expected_rand_counts = distribute_random_accesses(expected_spans, selected_random)
            expected_app_per_worker = list(expected_rand_counts)

        equal(pdata.get("effective_workers"), expected_eff, "worker-plan-mismatch")
        raw_spans = pdata.get("worker_spans")
        require(isinstance(raw_spans, list) and len(raw_spans) == expected_eff, "worker-plan-mismatch")
        cursor = 0
        for idx, (sp, (exp_off, exp_len), exp_app, exp_rc) in enumerate(
            zip(raw_spans, expected_spans, expected_app_per_worker, expected_rand_counts)
        ):
            require(isinstance(sp, dict), "worker-plan-mismatch")
            off = integer(sp.get("offset_bytes"), decimal=True)
            ln = integer(sp.get("span_bytes"), decimal=True)
            app = integer(sp.get("accesses_per_pass"), decimal=True)
            rc = integer(sp.get("random_access_count"), maximum=MAX_RANDOM_ACCESSES)
            equal(off, exp_off, "worker-plan-mismatch")
            equal(ln, exp_len, "worker-plan-mismatch")
            equal(off, cursor, "worker-plan-mismatch")
            if idx > 0 and ln > 0:
                require(off % 64 == 0, "worker-alignment")
            equal(app, exp_app, "worker-plan-mismatch")
            equal(rc, exp_rc, "worker-plan-mismatch")
            cursor += ln
        equal(cursor, buffer_bytes, "worker-plan-mismatch")

        if pname == "random":
            rw = pdata.get("random_workload")
            require(isinstance(rw, dict), "random-workload-mismatch")
            equal(rw.get("random_seed_uint64_decimal"), str(expected_random_seed), "seed-mismatch")
            equal(rw.get("selected_accesses_per_pass"), selected_random, "random-workload-mismatch")
            equal(rw.get("distinct_address_count"), selected_random, "random-workload-mismatch")
            reported_lws = integer(rw.get("logical_working_set_bytes"), decimal=True)
            reported_sha = rw.get("random_indices_sha256")
            if reported_sha is not None:
                require(
                    isinstance(reported_sha, str) and len(reported_sha) == 64,
                    "random-indices-sha256-mismatch",
                )
            need_random_oracle = (
                doc_status == "complete"
                or attempted_any
                or reported_sha is not None
                or reported_lws > 0
            )
            if need_random_oracle:
                generated = generate_random_workload(
                    expected_random_seed,
                    table,
                    expected_spans,
                    expected_rand_counts,
                    budget=budget,
                )
                equal(reported_sha, generated["random_indices_sha256"], "random-indices-sha256-mismatch")
                equal(reported_lws, generated["logical_working_set_bytes"], "random-workload-mismatch")
                random_lws = generated["logical_working_set_bytes"]
                random_worker_info = {
                    "worker_pass_xors": generated["worker_pass_xors"],
                    "worker_counts": expected_rand_counts,
                }
            else:
                equal(reported_lws, 0, "random-workload-mismatch")
                require(reported_sha is None, "random-indices-sha256-mismatch")

        active_patterns.append(pname)
        pattern_info[pname] = {
            "effective_workers": expected_eff,
            "spans": expected_spans,
            "accesses_per_pass": sum(expected_app_per_worker),
        }

    if rp_valid:
        require(len(active_patterns) > 0, "pattern-omission")
        fwd_eff = pattern_info["sequential_forward"]["effective_workers"]
        expected_identity = (
            f"patterns-work-plan-v1|buffer_size_mb={buf_mb}|workers={fwd_eff}"
            f"|worker_source={cfg['worker_source']}"
            f"|iterations={iterations if iterations is not None else 'auto'}"
            f"|loop_count={loop_count}|non_cacheable={1 if cfg['non_cacheable'] else 0}"
        )
        equal(rp.get("plan_identity"), expected_identity, "plan-identity-mismatch")
    else:
        fwd_eff = patterns_obj["sequential_forward"].get("effective_workers")
        valid_identities = {"patterns-work-plan-v1|invalid"}
        if type(fwd_eff) is int and fwd_eff > 0:
            valid_identities.add(
                f"patterns-work-plan-v1|buffer_size_mb={buf_mb}|workers={fwd_eff}"
                f"|worker_source={cfg['worker_source']}"
                f"|iterations={iterations if iterations is not None else 'auto'}"
                f"|loop_count={loop_count}|non_cacheable={1 if cfg['non_cacheable'] else 0}"
            )
        require(rp.get("plan_identity") in valid_identities, "plan-identity-mismatch")

    patterns_buffer_evidences = []
    if len(resources_obj) == 0:
        require(not attempted_any and doc_status != "complete", "resource-evidence-invalid")
    else:
        equal(
            set(resources_obj.keys()),
            {"src", "dst", "random_index_table_bytes", "random_validation_index_bytes"},
            "resource-evidence-invalid",
        )
        require_src_ro = (doc_status == "complete" or attempted_any)
        check_buffer_evidence(
            resources_obj["src"],
            buffer_bytes,
            cfg["non_cacheable"],
            page_size,
            require_ro=require_src_ro,
        )
        check_buffer_evidence(
            resources_obj["dst"],
            buffer_bytes,
            cfg["non_cacheable"],
            page_size,
            require_ro=False,
        )
        patterns_buffer_evidences.extend((resources_obj["src"], resources_obj["dst"]))
        rand_tbl_bytes = integer(resources_obj["random_index_table_bytes"], decimal=True)
        rand_val_bytes = integer(resources_obj["random_validation_index_bytes"], decimal=True)
        equal(rand_tbl_bytes, rand_val_bytes, "resource-evidence-invalid")
        if require_src_ro:
            equal(rand_tbl_bytes, selected_random * 8, "resource-evidence-invalid")
        else:
            require(rand_tbl_bytes in (0, selected_random * 8), "resource-evidence-invalid")

    min_committed_bytes = (2 * page_rounded) if rp_valid else 0
    mb = doc.get("memory_budget")
    check_memory_budget_block(
        mb,
        min_committed_bytes=min_committed_bytes,
        require_valid_when_active=attempted_any,
    )
    equal(mb["valid"], mb["reason_code"] == "within-memory-budget", "memory-budget-invalid")
    if rp_valid or mb.get("reason_code") in ("within-memory-budget", "memory-budget-exceeded"):
        env_avail_raw = env.get("available_memory_bytes")
        env_avail = integer(env_avail_raw, decimal=True) if env_avail_raw is not None else 0
        equal(mb.get("available_memory_bytes"), str(env_avail), "memory-budget-arithmetic")
        equal(mb.get("used_fallback"), env_avail == 0, "memory-budget-arithmetic")
        equal(mb.get("admitted_budget_bytes"), mb.get("allowed_memory_bytes"), "memory-budget-arithmetic")
        allowed_i = integer(mb.get("allowed_memory_bytes"), decimal=True)
        if env_avail == 0:
            equal(allowed_i, 2048 * 1048576, "memory-budget-arithmetic")
        else:
            expected_allowed = int(env_avail * 0.80)
            require(abs(allowed_i - expected_allowed) <= 1, "memory-budget-arithmetic")
        equal(mb.get("requested_data_bytes"), str(2 * buffer_bytes), "memory-budget-arithmetic")
        equal(mb.get("committed_data_bytes"), str(2 * page_rounded), "memory-budget-arithmetic")
        equal(
            mb.get("resource_rounding_bytes"),
            str(2 * page_rounded - 2 * buffer_bytes),
            "memory-budget-arithmetic",
        )
        if rp_valid:
            equal(mb.get("index_structure_bytes"), str(2 * selected_random * 8), "memory-budget-arithmetic")
            max_workers = max(pattern_info[p]["effective_workers"] for p in active_patterns)
            equal(
                mb.get("setup_peak_bytes"),
                str(4096 + 64 * max_workers + 2 * selected_random * 8),
                "memory-budget-arithmetic",
            )
        committed_i = integer(mb.get("committed_data_bytes"), decimal=True)
        transient_i = integer(mb.get("transient_peak_bytes"), decimal=True)
        known_peak_i = integer(mb.get("known_owned_peak_bytes"), decimal=True)
        equal(known_peak_i, committed_i + transient_i, "memory-budget-arithmetic")
        equal(mb["valid"], known_peak_i <= allowed_i, "memory-budget-arithmetic")

    op_plans = rp.get("operation_plans")
    require(isinstance(op_plans, list), "plan-invalid")
    if len(op_plans) > 4096:
        raise Rejected("unsupported-plan-limit", True)
    if not rp_valid:
        equal(op_plans, [], "plan-invalid")

    for idx, plan in enumerate(op_plans):
        budget(1)
        require(isinstance(plan, dict), "plan-invalid")
        equal(plan.get("plan_ref"), idx, "plan-reference")
        pname = plan.get("pattern")
        op = plan.get("operation")
        require(pname in pattern_info and op in OPERATIONS, "plan-invalid")
        passes = integer(plan.get("passes"), maximum=MAX_PASSES)
        require(passes >= 1, "work-mismatch")
        require(type(plan.get("explicit_iterations")) is bool, "work-mismatch")
        equal(plan.get("explicit_iterations"), iterations is not None, "work-mismatch")

        exp_geom = _compute_operation_plan_expected(
            pname, pattern_info[pname], op, passes, buffer_bytes, random_lws
        )
        require(exp_geom["exact_payload_bytes"] <= MAX_EXACT_PAYLOAD_BYTES, "work-cap-mismatch")
        equal(plan.get("total_accesses"), str(exp_geom["total_accesses"]), "plan-bytes-mismatch")
        equal(plan.get("exact_payload_bytes"), str(exp_geom["exact_payload_bytes"]), "plan-bytes-mismatch")
        equal(plan.get("accesses_per_pass"), str(exp_geom["accesses_per_pass"]), "plan-bytes-mismatch")
        equal(plan.get("min_accesses_per_pass"), str(exp_geom["min_accesses_per_pass"]), "plan-bytes-mismatch")
        equal(plan.get("max_accesses_per_pass"), str(exp_geom["max_accesses_per_pass"]), "plan-bytes-mismatch")
        equal(plan.get("distinct_address_count"), str(exp_geom["distinct_address_count"]), "plan-bytes-mismatch")
        equal(plan.get("logical_working_set_bytes"), str(exp_geom["logical_working_set_bytes"]), "plan-bytes-mismatch")
        equal(plan.get("completed_phase_cycles"), exp_geom["completed_phase_cycles"], "plan-bytes-mismatch")
        equal(plan.get("has_untouched_dst_bytes"), exp_geom["has_untouched_dst_bytes"], "plan-bytes-mismatch")

        kind_idx = CANONICAL_PATTERNS.index(pname)
        op_idx = OPERATIONS.index(op)
        expected_op_seed = splitmix64_mix(
            expected_src_seed ^ ((kind_idx + 1) << 48) ^ ((op_idx + 1) << 56) ^ passes
        )
        op_seed = integer(plan.get("op_seed_uint64_decimal"), decimal=True)
        equal(op_seed, expected_op_seed, "seed-mismatch")

        exp_ck = plan.get("expected_checksum")
        require(isinstance(exp_ck, dict), "checksum-invalid")
        if op == "read":
            equal(exp_ck.get("status"), "available", "checksum-status")
            exp_workers, exp_run = expected_read_checksum(
                table,
                op_seed,
                pname,
                STRIDE_BYTES[pname],
                passes,
                pattern_info[pname]["spans"],
                random_worker_info=(random_worker_info if pname == "random" else None),
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
    equal(set(frozen_refs.keys()), set(CANONICAL_PATTERNS), "plan-reference")
    for pname in CANONICAL_PATTERNS:
        prefs = frozen_refs.get(pname)
        require(isinstance(prefs, dict) and set(prefs.keys()) == set(OPERATIONS), "plan-reference")
        for op in OPERATIONS:
            ref = prefs.get(op)
            if pname not in pattern_info:
                require(ref is None, "plan-reference")
            elif ref is not None:
                integer(ref, maximum=MAX_SAFE_INT)
                require(
                    ref < len(op_plans)
                    and op_plans[ref]["pattern"] == pname
                    and op_plans[ref]["operation"] == op,
                    "plan-reference",
                )
                if iterations is not None:
                    equal(op_plans[ref]["passes"], iterations, "frozen-work-mismatch")

    seen_write_patterns = set()
    levels = []
    attempted_pairs = {
        (r.get("pattern"), r.get("operation")) for r in rows if r.get("attempted") is True
    }

    calibration = doc.get("calibration")
    require(isinstance(calibration, dict), "calibration-invalid")
    equal(calibration.get("excluded_from_results"), True, "calibration-excluded-mismatch")
    cal_attempts = calibration.get("attempts")
    require(isinstance(cal_attempts, dict), "calibration-invalid")
    if not rp_valid:
        equal(cal_attempts, {}, "calibration-invalid")
    else:
        equal(set(cal_attempts.keys()), set(active_patterns), "calibration-invalid")
        for pname, per_op in cal_attempts.items():
            require(pname in pattern_info and isinstance(per_op, dict), "calibration-invalid")
            equal(set(per_op.keys()), set(OPERATIONS), "calibration-invalid")
            exp_workers = pattern_info[pname]["effective_workers"]
            for op, attempts in per_op.items():
                require(op in OPERATIONS and isinstance(attempts, list), "calibration-invalid")
                frozen_ref = frozen_refs[pname][op]
                pair_attempted = (pname, op) in attempted_pairs
                require_complete_cal = (
                    doc_status == "complete"
                    or pair_attempted
                    or attempted_any
                    or (frozen_ref is not None and doc_status not in ("interrupted", "failed"))
                )
                if iterations is not None:
                    require(len(attempts) <= 1, "calibration-invalid")
                else:
                    require(len(attempts) <= 6, "calibration-invalid")
                if require_complete_cal:
                    require(len(attempts) > 0, "calibration-incomplete")
                    require(frozen_ref is not None, "plan-reference")
                elif doc_status == "interrupted" and iterations is None and frozen_ref is not None:
                    require(len(attempts) >= 2, "calibration-incomplete")
                for idx, attempt in enumerate(attempts):
                    budget(1)
                    require(isinstance(attempt, dict), "calibration-invalid")
                    equal(attempt.get("attempt_index"), idx, "calibration-index")
                    equal(attempt.get("pattern"), pname, "calibration-invalid")
                    equal(attempt.get("operation"), op, "calibration-invalid")
                    ref = attempt.get("plan_ref")
                    ex = attempt.get("execution")
                    require(isinstance(ex, dict), "calibration-invalid")
                    require(
                        isinstance(ex.get("status"), str)
                        and isinstance(ex.get("reason_code"), str)
                        and len(ex["reason_code"]) > 0,
                        "execution-invalid",
                    )
                    if ref is None:
                        raise Rejected("calibration-placeholder")
                    integer(ref, maximum=MAX_SAFE_INT)
                    require(
                        ref < len(op_plans)
                        and op_plans[ref]["pattern"] == pname
                        and op_plans[ref]["operation"] == op,
                        "plan-reference",
                    )
                    plan = op_plans[ref]
                    require(attempt.get("terminal") is True, "calibration-invalid")
                    require(type(attempt.get("valid")) is bool, "calibration-invalid")
                    if require_complete_cal:
                        require(attempt["valid"] is True, "calibration-valid-mismatch")
                    purpose = attempt.get("purpose")
                    require(purpose in ALLOWED_CALIBRATION_PURPOSES, "calibration-purpose-invalid")
                    if iterations is not None:
                        equal(purpose, "frozen_measurement_warmup", "calibration-purpose-invalid")
                        if frozen_ref is not None:
                            equal(ref, frozen_ref, "frozen-plan-reference")
                    else:
                        if idx == 0:
                            equal(purpose, "calibration_shape_warmup", "calibration-purpose-invalid")
                        elif idx == 1:
                            equal(purpose, "pilot", "calibration-purpose-invalid")
                        elif purpose == "frozen_measurement_warmup":
                            require(idx == len(attempts) - 1, "calibration-purpose-invalid")
                        else:
                            require(
                                purpose in ("correction", "single_unit_confirmation_warmup", "single_unit_confirmation"),
                                "calibration-purpose-invalid",
                            )
                            if purpose in ("single_unit_confirmation_warmup", "single_unit_confirmation"):
                                equal(plan["passes"], 1, "calibration-invalid")
                    cal_elapsed = ex.get("elapsed_seconds")
                    if cal_elapsed is None:
                        expected_dq = "unavailable"
                    else:
                        require(
                            type(cal_elapsed) in (int, float)
                            and math.isfinite(cal_elapsed)
                            and cal_elapsed > 0.0,
                            "elapsed-invalid",
                        )
                        if cal_elapsed < CALIBRATION_MIN_SECONDS:
                            expected_dq = "below-target-window"
                        elif cal_elapsed > CALIBRATION_MAX_SECONDS:
                            expected_dq = "above-target-window"
                        else:
                            expected_dq = "within-target-window"
                    equal(attempt.get("duration_quality"), expected_dq, "calibration-duration-quality")

                    req_w = integer(ex.get("requested_workers"), maximum=U32)
                    crt_w = integer(ex.get("created_workers"), maximum=U32)
                    cmp_w = integer(ex.get("completed_workers"), maximum=U32)
                    equal(req_w, cfg["requested_workers"], "worker-lifecycle")
                    if attempt["valid"]:
                        equal(crt_w, exp_workers, "worker-lifecycle")
                        equal(cmp_w, exp_workers, "worker-lifecycle")
                    else:
                        require(cmp_w <= crt_w <= exp_workers, "worker-lifecycle")

                    ok = _check_patterns_checksum(
                        ex.get("checksum"),
                        plan["expected_checksum"],
                        op,
                        compact=True,
                    )
                    if attempt["valid"]:
                        require(
                            ok and ex.get("valid") is True and ex.get("elapsed_seconds") is not None,
                            "calibration-valid-mismatch",
                        )
                    else:
                        require(ex.get("valid") is not True, "calibration-valid-mismatch")
                    _check_write_pattern_and_sentinel(
                        ex,
                        op,
                        True,
                        plan["has_untouched_dst_bytes"],
                        seen_write_patterns,
                        require_sentinel_when_needed=bool(attempt["valid"]),
                    )
                    val_obj = ex.get("validation")
                    require(isinstance(val_obj, dict), "validation-invalid")
                    check_validation_checks(
                        val_obj.get("checks"),
                        _validation_specs_for_op(op, plan["has_untouched_dst_bytes"]),
                        bool(attempt["valid"]),
                    )
                    cal_timing = ex.get("timing")
                    require(isinstance(cal_timing, dict), "timing-invalid")
                    levels.append(verify_cpu_raw_timing(cal_timing.get("cpu_raw"), ex.get("elapsed_seconds")))

                if iterations is None:
                    if (
                        len(attempts) >= 2
                        and attempts[0].get("plan_ref") is not None
                        and attempts[1].get("plan_ref") is not None
                    ):
                        equal(attempts[0]["plan_ref"], attempts[1]["plan_ref"], "frozen-plan-reference")
                    if (
                        len(attempts) >= 2
                        and attempts[-1].get("purpose") == "frozen_measurement_warmup"
                        and frozen_ref is not None
                    ):
                        equal(attempts[-2].get("plan_ref"), frozen_ref, "frozen-plan-reference")

                if require_complete_cal:
                    last_attempt = attempts[-1]
                    equal(last_attempt.get("purpose"), "frozen_measurement_warmup", "calibration-purpose-invalid")
                    equal(last_attempt.get("plan_ref"), frozen_ref, "frozen-plan-reference")
                elif doc_status == "interrupted" and frozen_ref is not None and attempts:
                    last_attempt = attempts[-1]
                    require(last_attempt.get("valid") is True, "calibration-valid-mismatch")
                    equal(last_attempt.get("plan_ref"), frozen_ref, "frozen-plan-reference")

    per_loop_count = len(active_patterns) * len(OPERATIONS)
    equal(len(rows), loop_count * per_loop_count, "measurement-count-mismatch")

    loop_records = doc.get("loop_records")
    require(isinstance(loop_records, list) and len(loop_records) == loop_count, "loop-count-mismatch")

    accepted = {p: {op: [] for op in OPERATIONS} for p in active_patterns}
    stopped = False
    num_patterns = len(active_patterns)

    for i, row in enumerate(rows):
        budget(1)
        require(isinstance(row, dict), "measurement-invalid")
        integer(row.get("measurement_id"), maximum=MAX_SAFE_INT)
        equal(row.get("measurement_id"), i, "measurement-id-mismatch")
        loop, within_loop = divmod(i, per_loop_count)
        p_pos, o_pos = divmod(within_loop, len(OPERATIONS))
        expected_pattern = active_patterns[(loop + p_pos) % num_patterns]
        expected_op = OPERATIONS[(loop + o_pos) % len(OPERATIONS)]
        equal(row.get("loop_index"), loop, "measurement-order-mismatch")
        equal(row.get("pattern_order_position"), p_pos, "measurement-order-mismatch")
        equal(row.get("operation_order_position"), o_pos, "measurement-order-mismatch")
        equal(row.get("pattern"), expected_pattern, "measurement-order-mismatch")
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
                and op_plans[ref]["pattern"] == expected_pattern
                and op_plans[ref]["operation"] == expected_op,
                "plan-reference",
            )
            plan = op_plans[ref]
            equal(ref, frozen_refs[expected_pattern][expected_op], "frozen-plan-reference")

        if plan is not None:
            has_untouched = plan["has_untouched_dst_bytes"]
        elif expected_pattern in STRIDED_PATTERNS:
            has_untouched = True
        elif expected_pattern == "random":
            has_untouched = (selected_random * ACCESS_SIZE_BYTES) < buffer_bytes
        else:
            has_untouched = False

        ex = row.get("execution")
        require(isinstance(ex, dict), "execution-invalid")
        require(
            isinstance(ex.get("status"), str)
            and isinstance(ex.get("reason_code"), str)
            and len(ex["reason_code"]) > 0,
            "execution-invalid",
        )
        timing_obj = ex.get("timing")
        require(isinstance(timing_obj, dict), "timing-invalid")
        require(type(timing_obj.get("evaluated")) is bool, "timing-invalid")
        if timing_obj["evaluated"]:
            require(type(timing_obj.get("valid")) is bool, "timing-invalid")
        else:
            require(timing_obj.get("valid") is None, "timing-invalid")
        if status == "measured":
            require(timing_obj.get("diagnostic_elapsed_seconds") is None, "metric-null")
            elapsed = timing_obj.get("elapsed_seconds")
        else:
            require(timing_obj.get("elapsed_seconds") is None, "metric-null")
            if not attempted or not timing_obj["evaluated"]:
                require(timing_obj.get("diagnostic_elapsed_seconds") is None, "metric-null")
            elapsed = timing_obj.get("diagnostic_elapsed_seconds")

        if attempted:
            require(plan is not None, "plan-reference")
            checksum_ok = _check_patterns_checksum(
                row.get("checksum"),
                plan["expected_checksum"],
                expected_op,
                compact=False,
            )
            levels.append(verify_cpu_raw_timing(timing_obj.get("cpu_raw"), elapsed))
            _check_write_pattern_and_sentinel(
                ex,
                expected_op,
                True,
                has_untouched,
                seen_write_patterns,
                require_sentinel_when_needed=(status == "measured"),
            )
        else:
            equal(ex.get("status"), "not_run", "execution-invalid")
            require(
                timing_obj.get("elapsed_seconds") is None
                and timing_obj.get("diagnostic_elapsed_seconds") is None
                and timing_obj.get("cpu_raw") is None,
                "metric-null",
            )
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
            _check_write_pattern_and_sentinel(
                ex,
                expected_op,
                False,
                has_untouched,
                seen_write_patterns,
                require_sentinel_when_needed=False,
            )
            checksum_ok = False

        val_obj = ex.get("validation")
        require(isinstance(val_obj, dict), "validation-invalid")
        check_validation_checks(
            val_obj.get("checks"),
            _validation_specs_for_op(expected_op, has_untouched),
            status == "measured",
        )

        if status == "measured":
            require(
                attempted
                and checksum_ok
                and ex.get("status") == "valid"
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
            exp_workers = pattern_info[expected_pattern]["effective_workers"]
            equal(ex.get("requested_workers"), cfg["requested_workers"], "worker-lifecycle")
            equal(ex.get("created_workers"), exp_workers, "worker-lifecycle")
            equal(ex.get("completed_workers"), exp_workers, "worker-lifecycle")
            accepted[expected_pattern][expected_op].append(i)
        else:
            require(row.get("bandwidth_gb_s") is None, "metric-null")

    for loop, record in enumerate(loop_records):
        require(isinstance(record, dict), "loop-order-mismatch")
        equal(record.get("loop_index"), loop, "loop-order-mismatch")
        if not rp_valid:
            equal(record.get("planned_order"), [], "loop-order-mismatch")
            equal(record.get("realized_order"), [], "loop-prefix-mismatch")
            equal(record.get("realized_order_count"), 0, "loop-prefix-mismatch")
            equal(record.get("measurement_indexes"), [], "loop-measurement-reference")
            continue
        planned_patterns = [active_patterns[(loop + k) % num_patterns] for k in range(num_patterns)]
        planned_ops = [OPERATIONS[(loop + k) % len(OPERATIONS)] for k in range(len(OPERATIONS))]
        planned_order = [
            {"pattern": p, "operation": o} for p in planned_patterns for o in planned_ops
        ]
        equal(record.get("planned_order"), planned_order, "loop-order-mismatch")
        start_idx = loop * per_loop_count
        attempted_order = [
            {"pattern": r["pattern"], "operation": r["operation"]}
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
    require(isinstance(aggs, dict) and isinstance(aggs.get("patterns"), dict), "aggregate-invalid")
    equal(set(aggs["patterns"].keys()), set(CANONICAL_PATTERNS), "aggregate-population-mismatch")
    for pname in CANONICAL_PATTERNS:
        p_agg = aggs["patterns"][pname]
        require(
            isinstance(p_agg, dict) and set(p_agg.keys()) == set(OPERATIONS),
            "aggregate-population-mismatch",
        )
        cv_threshold = (
            STREAMING_CV_WARNING_PCT
            if pname in STREAMING_PATTERNS
            else SPARSE_CV_WARNING_PCT
        )
        for op in OPERATIONS:
            a = p_agg[op]
            require(isinstance(a, dict), "aggregate-invalid")
            q = a.get("quality")
            require(isinstance(q, dict), "aggregate-quality-invalid")
            close(q.get("cv_warning_threshold_pct"), cv_threshold, "aggregate-cv-mismatch")

            if not rp_valid:
                equal(a.get("status"), "unavailable", "aggregate-status-mismatch")
                equal(a.get("reason_code"), rp_reason, "aggregate-status-mismatch")
                equal(a.get("accepted_measurement_ids"), [], "aggregate-population-mismatch")
                require(
                    a.get("value") is None
                    and a.get("relative_to_forward_pct") is None
                    and a.get("statistics") is None,
                    "aggregate-null",
                )
                equal(q.get("repeatability_assessment_available"), False, "aggregate-quality-mismatch")
                equal(q.get("run_complete"), False, "aggregate-quality-mismatch")
                equal(q.get("cv_warning"), False, "aggregate-cv-mismatch")
                continue

            if pname not in pattern_info:
                equal(a.get("status"), "skipped", "aggregate-status-mismatch")
                equal(a.get("reason_code"), SKIPPED_BUFFER_TOO_SMALL, "aggregate-status-mismatch")
                equal(a.get("accepted_measurement_ids"), [], "aggregate-population-mismatch")
                require(
                    a.get("value") is None
                    and a.get("relative_to_forward_pct") is None
                    and a.get("statistics") is None,
                    "aggregate-null",
                )
                equal(q.get("repeatability_assessment_available"), False, "aggregate-quality-mismatch")
                equal(q.get("run_complete"), False, "aggregate-quality-mismatch")
                equal(q.get("cv_warning"), False, "aggregate-cv-mismatch")
                continue

            ids = accepted[pname][op]
            equal(a.get("accepted_measurement_ids"), ids, "aggregate-population-mismatch")
            values = [rows[idx]["bandwidth_gb_s"] for idx in ids]
            expected_stats = compute_statistics(values)
            n = len(ids)
            if expected_stats is None:
                equal(a.get("status"), "unavailable", "aggregate-status-mismatch")
                require(
                    isinstance(a.get("reason_code"), str) and len(a["reason_code"]) > 0,
                    "aggregate-status-mismatch",
                )
                require(a.get("value") is None and a.get("statistics") is None, "aggregate-null")
            else:
                equal(a.get("status"), "measured", "aggregate-status-mismatch")
                equal(a.get("reason_code"), "measured", "aggregate-status-mismatch")
                close(a.get("value"), expected_stats["median"], "aggregate-headline-mismatch")
                check_statistics_block(a.get("statistics"), expected_stats)
            equal(q.get("repeatability_assessment_available"), n >= 2, "aggregate-quality-mismatch")
            equal(q.get("run_complete"), n == loop_count, "aggregate-quality-mismatch")
            cv_warn = (
                n >= 3
                and expected_stats is not None
                and expected_stats["coefficient_of_variation_pct"] is not None
                and expected_stats["coefficient_of_variation_pct"] > cv_threshold
            )
            equal(q.get("cv_warning"), cv_warn, "aggregate-cv-mismatch")

    for op in OPERATIONS:
        fwd_val = aggs["patterns"]["sequential_forward"][op].get("value")
        fwd_ok = (
            fwd_val is not None
            and type(fwd_val) in (int, float)
            and math.isfinite(fwd_val)
            and fwd_val > 0.0
        )
        for pname in CANONICAL_PATTERNS:
            a = aggs["patterns"][pname][op]
            val = a.get("value")
            if fwd_ok and val is not None:
                close(
                    a.get("relative_to_forward_pct"),
                    100.0 * val / fwd_val,
                    "aggregate-relative-mismatch",
                )
            else:
                require(a.get("relative_to_forward_pct") is None, "aggregate-relative-mismatch")

    counters = doc.get("counters")
    require(isinstance(counters, dict), "counter-mismatch")
    measured_total = sum(len(accepted[p][op]) for p in active_patterns for op in OPERATIONS)
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
    if active_patterns and all(frozen_refs[p][op] is not None for p in active_patterns for op in OPERATIONS):
        planned_passes = (
            sum(op_plans[frozen_refs[p][op]]["passes"] for p in active_patterns for op in OPERATIONS)
            * loop_count
        )
        planned_bytes = (
            sum(
                integer(op_plans[frozen_refs[p][op]]["exact_payload_bytes"], decimal=True)
                for p in active_patterns
                for op in OPERATIONS
            )
            * loop_count
        )
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
        doc.get("pattern_order_balance_complete"),
        complete and num_patterns > 0 and loop_count % num_patterns == 0,
        "order-balance-mismatch",
    )

    expected_interp = {
        "dram_residency": "unverified",
        "cache_residency": "unverified",
        "non_cacheable_requested": cfg["non_cacheable"],
        "non_cacheable_is_best_effort_madvise": True,
        "large_page_backing_verified": False,
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
        buffer_evidences=patterns_buffer_evidences,
        all_targets_have_resources=(len(resources_obj) > 0),
        active_targets=active_patterns,
        operations=OPERATIONS,
        agg_targets=aggs["patterns"],
        attempted_any=attempted_any,
        iterations=iterations,
        rows=rows,
        duration_window=(0.100, 0.250),
        target_key="pattern",
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


def _validate_patterns_buffer_size(v):
    n = integer(v, maximum=MAX_SAFE_INT)
    require(n >= 1, "sweep-parameter-value-invalid")
    return n


def _validate_patterns_threads(v):
    n = integer(v, maximum=U32)
    require(n >= 1, "sweep-parameter-value-invalid")
    return n


SWEEP_PARAM_SPECS = {
    "buffer-size": ("buffer_size_mb", _validate_patterns_buffer_size),
    "threads": ("requested_workers", _validate_patterns_threads),
}

SWEEP_RESOLVED_SOURCE_MAP = {
    "buffer-size": "buffer_size",
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


def verify_sweep(
    doc,
    binary=None,
    require_raw=False,
    budget=None,
    max_random_indices=DEFAULT_MAX_RANDOM_INDICES,
):
    """Verify a patterns schema-1 sweep envelope."""
    return verify_sweep_envelope(
        doc,
        base_mode=MODE_NAME,
        param_specs=SWEEP_PARAM_SPECS,
        verify_single_fn=verify_single,
        fixed_config_keys=FIXED_CONFIG_KEYS,
        binary=binary,
        require_raw=require_raw,
        budget=budget,
        extra_verify_kwargs={"max_random_indices": max_random_indices},
        sweep_resolved_source_map=SWEEP_RESOLVED_SOURCE_MAP,
    )


def verify(
    doc,
    binary=None,
    require_raw=False,
    budget=None,
    max_random_indices=DEFAULT_MAX_RANDOM_INDICES,
):
    """Verify either a direct patterns artifact or a patterns sweep envelope."""
    require(isinstance(doc, dict), "invalid-input")
    cfg = doc.get("configuration")
    if isinstance(cfg, dict) and cfg.get("mode") == "sweep" and cfg.get("base_mode") == MODE_NAME:
        return verify_sweep(
            doc,
            binary=binary,
            require_raw=require_raw,
            budget=budget,
            max_random_indices=max_random_indices,
        )
    return verify_single(
        doc,
        binary=binary,
        require_raw=require_raw,
        budget=budget,
        max_random_indices=max_random_indices,
    )


def _extra_args(parser):
    parser.add_argument(
        "--max-random-indices",
        type=int,
        default=DEFAULT_MAX_RANDOM_INDICES,
        help=f"Maximum allowed random accesses per pass (default: {DEFAULT_MAX_RANDOM_INDICES})",
    )


def _verify_cli(doc, args):
    require(args.max_random_indices >= 1, "invalid-input")
    return verify(
        doc,
        binary=args.binary,
        require_raw=args.require_raw_timing,
        max_random_indices=args.max_random_indices,
    )


def main():
    return run_verifier_cli(_verify_cli, __doc__, extra_args_fn=_extra_args)


if __name__ == "__main__":
    sys.exit(main())

"""Independent numeric goldens, O(1) vs brute-force differential checks,
schema-v1 fixture verification, sweep envelope checks, and semantic mutations
for verify_patterns_result.py.
"""

import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "script-examples"))

import cpu_verify_common as common
import verify_patterns_result as verifier

COMPLETE_FIXTURE = ROOT / "tests/fixtures/patterns-schema-v1-complete-current.json"
SKIPPED_2MB_FIXTURE = ROOT / "tests/fixtures/patterns-schema-v1-skipped-2mb-current.json"


def load_fixture(path=COMPLETE_FIXTURE):
    return json.loads(path.read_text())


def make_truncated(status, fixture_path=COMPLETE_FIXTURE):
    """Build a consistent partial/interrupted/failed artifact from a valid fixture."""
    doc = load_fixture(fixture_path)
    measured_keep = 2
    for idx, row in enumerate(doc["measurements"]):
        if idx < measured_keep:
            continue
        row["attempted"] = False
        row["status"] = (
            "not_run"
            if status == "partial"
            else "interrupted" if status == "interrupted" else "failed"
        )
        row["reason_code"] = (
            "not-run"
            if status == "partial"
            else "interruption-before-task" if status == "interrupted" else "runtime-failure"
        )
        row["bandwidth_gb_s"] = None
        row["completed_passes"] = 0
        row["completed_exact_payload_bytes"] = "0"
        ex = row["execution"]
        ex["status"] = "not_run"
        ex["reason_code"] = row["reason_code"]
        ex["valid"] = None
        ex["write_pattern_uint64_decimal"] = None
        ex["pre_op_sentinel_uint64_decimal"] = None
        ex.pop("attempt_seed_uint64_decimal", None)
        ex["timing"] = {
            "evaluated": False,
            "valid": None,
            "elapsed_seconds": None,
            "diagnostic_elapsed_seconds": None,
            "cpu_raw": None,
        }
        for ck in ex["validation"]["checks"]:
            ck["evaluated"] = False if ck["applicable"] else None
            ck["valid"] = None
            ck["reason_code"] = "not-evaluated" if ck["applicable"] else "not-applicable"
        row["checksum"] = {
            "status": "not_evaluated" if row["operation"] == "read" else "not_applicable",
            "reason_code": row["reason_code"] if row["operation"] == "read" else "not-applicable",
            "checksum_valid": None,
            "actual_worker_checksums": None,
            "actual_run_checksum": None,
        }

    doc.update(
        status=status,
        reason_code=(
            "partial-results"
            if status == "partial"
            else "interruption-requested" if status == "interrupted" else "runtime-failure"
        ),
        results_complete=False,
        run_accepted=False,
        interruption_requested=(status == "interrupted"),
        operation_order_balance_complete=False,
        pattern_order_balance_complete=False,
    )
    rows = doc["measurements"]
    active_count = sum(1 for p in doc["resolved_plan"]["patterns"].values() if p["enabled"])
    per_loop = active_count * 3
    for loop_idx, rec in enumerate(doc["loop_records"]):
        slice_rows = rows[loop_idx * per_loop : (loop_idx + 1) * per_loop]
        realized = [
            {"pattern": r["pattern"], "operation": r["operation"]}
            for r in slice_rows
            if r["attempted"]
        ]
        rec["realized_order"] = realized
        rec["realized_order_count"] = len(realized)

    c = doc["counters"]
    c.update(
        attempted_loops=len({r["loop_index"] for r in rows if r["attempted"]}),
        completed_loops=sum(
            all(r["status"] == "measured" for r in rows[i : i + per_loop])
            for i in range(0, len(rows), per_loop)
        ),
        attempted_measurements=sum(1 for r in rows if r["attempted"]),
        terminal_measurements=sum(1 for r in rows if r["status"] != "not_run"),
        measured_measurements=sum(1 for r in rows if r["status"] == "measured"),
        completed_passes=str(sum(r["completed_passes"] for r in rows)),
        completed_exact_payload_bytes=str(
            sum(int(r["completed_exact_payload_bytes"]) for r in rows)
        ),
    )

    for pname, p_agg in doc["aggregates"]["patterns"].items():
        if not doc["resolved_plan"]["patterns"][pname]["enabled"]:
            continue
        for op, a in p_agg.items():
            ids = [
                r["measurement_id"]
                for r in rows
                if r["pattern"] == pname and r["operation"] == op and r["status"] == "measured"
            ]
            vals = [rows[idx]["bandwidth_gb_s"] for idx in ids]
            st = common.compute_statistics(vals)
            a["accepted_measurement_ids"] = ids
            a["status"] = "measured" if st is not None else "unavailable"
            a["reason_code"] = "measured" if st is not None else doc["reason_code"]
            a["value"] = st["median"] if st is not None else None
            a["statistics"] = st
            a["quality"]["repeatability_assessment_available"] = len(ids) >= 2
            a["quality"]["run_complete"] = len(ids) == doc["configuration"]["loop_count"]
            a["quality"]["cv_warning"] = False

    for op in verifier.OPERATIONS:
        fwd_val = doc["aggregates"]["patterns"]["sequential_forward"][op]["value"]
        for pname in verifier.CANONICAL_PATTERNS:
            a = doc["aggregates"]["patterns"][pname][op]
            if fwd_val is not None and fwd_val > 0 and a["value"] is not None:
                a["relative_to_forward_pct"] = 100.0 * a["value"] / fwd_val
            else:
                a["relative_to_forward_pct"] = None
    doc["quality_warnings"] = [w for w in doc["quality_warnings"] if not w.endswith("-high-cv")]
    return doc


def brute_force_src_bytes(src_seed, total_bytes):
    """Explicit byte-array reference matching GF(2) patterns-buffer-pattern-v1."""
    total_words = (total_bytes + 7) // 8
    t = [common.splitmix64_mix(src_seed ^ k) for k in range(256)]
    buf = bytearray(total_words * 8)
    for wi in range(total_words):
        lane = wi ^ common.rotl64(wi, 7) ^ common.rotl64(wi, 33)
        slot = ((wi >> 2) ^ common.rotl64(wi >> 2, 19) ^ common.rotl64(wi >> 2, 41)) if (wi & 3) == 0 else 0
        w = (t[wi & 255] ^ lane ^ slot) & common.U64
        buf[wi * 8 : (wi + 1) * 8] = w.to_bytes(8, "little")
    return buf[:total_bytes]


def brute_force_seq_pass_xor(buf, offset_bytes, span_bytes, reverse=False):
    if span_bytes == 0:
        return 0
    tail_bytes = span_bytes % 32
    vec_bytes = span_bytes - tail_bytes
    vec_start = offset_bytes + (tail_bytes if reverse else 0)
    tail_start = offset_bytes + (0 if reverse else vec_bytes)
    vec_xor = 0
    for pos in range(vec_start, vec_start + vec_bytes, 8):
        vec_xor ^= int.from_bytes(buf[pos : pos + 8], "little")
    tail_xor = 0
    for pos in range(tail_start, tail_start + tail_bytes):
        tail_xor ^= buf[pos]
    return (vec_xor ^ tail_xor) & common.U64


def brute_force_strided_phase_xor(buf, offset_bytes, span_bytes, stride_bytes, phase):
    acc = 0
    pos = offset_bytes + phase * 32
    end = offset_bytes + span_bytes
    while pos + 32 <= end:
        for lane in range(4):
            p = pos + lane * 8
            acc ^= int.from_bytes(buf[p : p + 8], "little")
        pos += stride_bytes
    return acc & common.U64


class PatternsVerifierTests(unittest.TestCase):
    def assert_reason(self, doc, reason, unsupported=None, **kwargs):
        with self.assertRaises(common.Rejected) as caught:
            verifier.verify(doc, **kwargs)
        self.assertEqual(caught.exception.reason, reason)
        if unsupported is not None:
            self.assertEqual(caught.exception.unsupported, unsupported)

    def test_valid_fixtures_and_cli_raw_timing_and_stdin(self):
        for fixture_path, expected_2mb_status in (
            (COMPLETE_FIXTURE, "active"),
            (SKIPPED_2MB_FIXTURE, "skipped"),
        ):
            with self.subTest(fixture=fixture_path.name):
                doc = load_fixture(fixture_path)
                self.assertEqual(
                    doc["resolved_plan"]["patterns"]["strided_2mb"]["status"],
                    expected_2mb_status,
                )
                verdict = verifier.verify(doc, require_raw=True)
                self.assertTrue(verdict["artifact_consistent"])
                self.assertTrue(verdict["run_accepted"])
                self.assertEqual(verdict["checks"]["checksum"], "independent-oracle")
                self.assertEqual(verdict["checks"]["timing"], "cpu-raw-ticks")
                self.assertEqual(verdict["checks"]["build"], "manifest-only")

                proc = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "script-examples/verify_patterns_result.py"),
                        str(fixture_path),
                        "--require-raw-timing",
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                parsed = json.loads(proc.stdout)
                self.assertTrue(parsed["run_accepted"])

        # Explicit "-" and implicit piped stdin both work cleanly.
        raw_json = SKIPPED_2MB_FIXTURE.read_text()
        for cli_args in (
            [sys.executable, str(ROOT / "script-examples/verify_patterns_result.py"), "-", "--require-raw-timing"],
            [sys.executable, str(ROOT / "script-examples/verify_patterns_result.py"), "--require-raw-timing"],
        ):
            with self.subTest(cli_args=cli_args):
                proc = subprocess.run(
                    cli_args,
                    input=raw_json,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                self.assertTrue(json.loads(proc.stdout)["run_accepted"])

    def test_pinned_424242_golden_oracle_vectors(self):
        # Pinned golden vectors from M2 / test_patterns_oracle.cpp (base_seed = 424242, 3 MiB buffer, 2 workers)
        doc = load_fixture(COMPLETE_FIXTURE)
        self.assertEqual(doc["seeds"]["domain_seeds"]["src_uint64_decimal"], str(0x63329A0ADA1F7B42))
        self.assertEqual(doc["seeds"]["domain_seeds"]["random_uint64_decimal"], str(0x8C0E47743450068D))
        self.assertEqual(doc["seeds"]["domain_seeds"]["pre_op_sentinel_uint64_decimal"], str(0x216631658CFD9063))
        self.assertEqual(
            doc["resolved_plan"]["patterns"]["random"]["random_workload"]["random_indices_sha256"],
            "c199eec93986c9390e43f9051fb3c7ccd139255c48590c5da11b86f75d46214e",
        )

        # Verify pinned read expected_run_checksums for passes=2 (matching -i 2 in fixture)
        pinned_read = {
            "sequential_forward": (str(0x7CE1AB8B33469B26), str(0x976F565822F7DCB2)),
            "sequential_reverse": (str(0xCAED6CE041BEE918), str(0x1F9DDC36876DCADF)),
            "strided_2mb": (str(0xB29CA2DA914EB711), str(0xEFF3ED27210A65F8)),
            "random": (str(0x3BAB9F0AC4D4A354), str(0x47FCF836F675B4BD)),
        }
        plans = doc["resolved_plan"]["operation_plans"]
        refs = doc["resolved_plan"]["frozen_plan_refs"]
        for pname, (exp_a, exp_b) in pinned_read.items():
            rc = plans[refs[pname]["read"]]["expected_checksum"]["expected_run_checksum"]
            self.assertEqual(rc["state_a_uint64_decimal"], exp_a)
            self.assertEqual(rc["state_b_uint64_decimal"], exp_b)

        # Pinned strided_64 (passes=3) and write_pattern / pre_op_sentinel from test_patterns_oracle.cpp
        src_seed = int(doc["seeds"]["domain_seeds"]["src_uint64_decimal"])
        table = verifier.PatternsTable(src_seed)
        s64_op_seed = common.splitmix64_mix(src_seed ^ (3 << 48) ^ (1 << 56) ^ 3)
        _, s64_rc = verifier.expected_read_checksum(
            table, s64_op_seed, "strided_64", 64, 3,
            [(0, 1572864), (1572864, 1572864)],
        )
        self.assertEqual(s64_rc["state_a_uint64_decimal"], str(0x13524366434F28D3))
        self.assertEqual(s64_rc["state_b_uint64_decimal"], str(0xC0287E475E14A21C))

        fwd_write_op_seed = common.splitmix64_mix(src_seed ^ (1 << 48) ^ (2 << 56) ^ 2)
        write_attempt_seed = common.splitmix64_mix(fwd_write_op_seed ^ (0x5752495400000000 | 1))
        self.assertEqual(common.splitmix64_mix(write_attempt_seed) | 1, 0xD285D5C0A0D095DF)
        sent_seed = int(doc["seeds"]["domain_seeds"]["pre_op_sentinel_uint64_decimal"])
        self.assertEqual(common.splitmix64_mix(sent_seed ^ 1) | 1, 0xB0FA22685008F2B1)

    def test_o1_sequential_and_strided_oracles_match_brute_force(self):
        src_seed = common.splitmix64_mix(424242 ^ verifier.DOMAIN_SRC)
        table = verifier.PatternsTable(src_seed)
        buf = brute_force_src_bytes(src_seed, 65536)

        # Sequential forward & reverse including unaligned offsets and <32B tails
        spans = (0, 1, 7, 8, 9, 15, 16, 31, 32, 33, 63, 64, 65, 255, 256, 257, 511, 512, 1023, 2048 + 17)
        for offset in (0, 3, 7, 8, 13, 32, 64, 129, 256):
            for span in spans:
                with self.subTest(mode="seq", offset=offset, span=span):
                    self.assertEqual(
                        verifier.pass_xor_sequential_forward(table, offset, span),
                        brute_force_seq_pass_xor(buf, offset, span, reverse=False),
                    )
                    self.assertEqual(
                        verifier.pass_xor_sequential_reverse(table, offset, span),
                        brute_force_seq_pass_xor(buf, offset, span, reverse=True),
                    )

        # Strided 2D partial-cycle carry-split (low_0 + rem > Q), q==1, 2<=q<=5, q>=6, and tails
        for stride in (64, 256, 2048, 4096):
            q_period = stride // 32
            for offset in (0, 32, (q_period - 1) * 32, q_period * 32 + 32):
                for span_slots in (2, q_period, q_period + 1, 2 * q_period + 3, 3 * q_period - 1):
                    span = span_slots * 32
                    if offset + span > len(buf):
                        continue
                    # Full Q-phase XOR
                    full_bf = 0
                    for ph in range(q_period):
                        full_bf ^= brute_force_strided_phase_xor(buf, offset, span, stride, ph)
                    self.assertEqual(
                        verifier.strided_full_cycle_xor(table, offset, span),
                        full_bf,
                    )
                    for rem in range(1, q_period):
                        if q_period >= 64 and rem not in (1, 2, 17, q_period - 1):
                            continue
                        with self.subTest(stride=stride, offset=offset, span_slots=span_slots, rem=rem):
                            partial_bf = 0
                            for ph in range(rem):
                                partial_bf ^= brute_force_strided_phase_xor(buf, offset, span, stride, ph)
                            self.assertEqual(
                                verifier.strided_partial_cycle_xor(table, offset, span, stride, rem),
                                partial_bf,
                            )

    def test_closed_form_read_checksum_and_random_oracle_match_iterative(self):
        src_seed = common.splitmix64_mix(99 ^ verifier.DOMAIN_SRC)
        rand_seed = common.splitmix64_mix(99 ^ verifier.DOMAIN_RANDOM)
        table = verifier.PatternsTable(src_seed)
        op_seed = 0x123456789ABCDEF0
        buf_bytes = 32768
        buf = brute_force_src_bytes(src_seed, buf_bytes)
        worker_spans = common.partition_64b(buf_bytes, 3)

        # Random coprime + Fisher-Yates (both partial and full-span count_w == span_slots_w)
        rand_counts = verifier.distribute_random_accesses(worker_spans, 250)
        rand_counts_full = [sp // 32 for _, sp in worker_spans]
        for counts in (rand_counts, rand_counts_full):
            rw = verifier.generate_random_workload(rand_seed, table, worker_spans, counts)
            self.assertEqual(len(rw["random_indices_sha256"]), 64)
            self.assertGreater(rw["logical_working_set_bytes"], 0)
            r_info = {"worker_pass_xors": rw["worker_pass_xors"], "worker_counts": counts}
            for passes in (1, 2, 5, 17):
                workers, run_ck = verifier.expected_read_checksum(
                    table, op_seed, "random", 0, passes, worker_spans, random_worker_info=r_info
                )
                tuples = []
                for w, ((off, _), px, cnt) in enumerate(zip(worker_spans, rw["worker_pass_xors"], counts)):
                    sa = common.splitmix64_mix(op_seed ^ (0xA100000000000000 | w))
                    sb = common.splitmix64_mix(op_seed ^ (0xB200000000000000 | w))
                    for p in range(1, passes + 1):
                        sa = (sa + px + common.SPLITMIX_A * p) & common.U64
                        sb = (sb + ((px ^ common.SPLITMIX_B) * p) + common.SPLITMIX_C) & common.U64
                    self.assertEqual(workers[w]["state_a_uint64_decimal"], str(sa))
                    self.assertEqual(workers[w]["state_b_uint64_decimal"], str(sb))
                    tuples.append((sa, sb, passes * cnt * 32, passes))
                self.assertEqual(run_ck, verifier.fold_patterns_workers(tuples))

        # Strided closed-form fold across phase cycles (passes < Q, == Q, > Q, multiple cycles)
        for stride in (64, 4096):
            q_period = stride // 32
            for passes in (1, 2, 3, q_period - 1, q_period, q_period + 1, 2 * q_period + 3):
                if passes <= 0:
                    continue
                with self.subTest(stride=stride, passes=passes):
                    workers, run_ck = verifier.expected_read_checksum(
                        table, op_seed, "strided_64" if stride == 64 else "strided_4096",
                        stride, passes, worker_spans
                    )
                    tuples = []
                    for w, (off, sp) in enumerate(worker_spans):
                        sa = common.splitmix64_mix(op_seed ^ (0xA100000000000000 | w))
                        sb = common.splitmix64_mix(op_seed ^ (0xB200000000000000 | w))
                        cycles = passes // q_period
                        rem = passes % q_period
                        fc_bf = 0
                        for ph in range(q_period):
                            fc_bf ^= brute_force_strided_phase_xor(buf, off, sp, stride, ph)
                        for p in range(1, cycles + 1):
                            sa = (sa + fc_bf + common.SPLITMIX_A * p) & common.U64
                            sb = (sb + ((fc_bf ^ common.SPLITMIX_B) * p) + common.SPLITMIX_C) & common.U64
                        if rem > 0:
                            pc_bf = 0
                            for ph in range(rem):
                                pc_bf ^= brute_force_strided_phase_xor(buf, off, sp, stride, ph)
                            p = cycles + 1
                            sa = (sa + pc_bf + common.SPLITMIX_A * p) & common.U64
                            sb = (sb + ((pc_bf ^ common.SPLITMIX_B) * p) + common.SPLITMIX_C) & common.U64
                        self.assertEqual(workers[w]["state_a_uint64_decimal"], str(sa))
                        self.assertEqual(workers[w]["state_b_uint64_decimal"], str(sb))
                        tuples.append((sa, sb, int(workers[w]["exact_bytes_read"]), passes))
                    self.assertEqual(run_ck, verifier.fold_patterns_workers(tuples))

    def test_lifecycle_states_consistent_but_not_accepted(self):
        for status in ("partial", "interrupted", "failed"):
            with self.subTest(status=status):
                doc = make_truncated(status, COMPLETE_FIXTURE)
                verdict = verifier.verify(doc)
                self.assertTrue(verdict["artifact_consistent"])
                self.assertFalse(verdict["run_accepted"])

        # Unsupported run with 0 attempted measurements
        unsupported_doc = make_truncated("partial", SKIPPED_2MB_FIXTURE)
        for row in unsupported_doc["measurements"]:
            row["attempted"] = False
            row["status"] = "not_run"
            row["reason_code"] = "unsupported"
            row["bandwidth_gb_s"] = None
            row["completed_passes"] = 0
            row["completed_exact_payload_bytes"] = "0"
            row["execution"]["status"] = "not_run"
            row["execution"]["reason_code"] = "unsupported"
            row["execution"]["valid"] = None
            row["execution"]["write_pattern_uint64_decimal"] = None
            row["execution"]["pre_op_sentinel_uint64_decimal"] = None
            row["execution"].pop("attempt_seed_uint64_decimal", None)
            row["execution"]["timing"] = {
                "evaluated": False,
                "valid": None,
                "elapsed_seconds": None,
                "diagnostic_elapsed_seconds": None,
                "cpu_raw": None,
            }
            for ck in row["execution"]["validation"]["checks"]:
                ck["evaluated"] = False if ck["applicable"] else None
                ck["valid"] = None
                ck["reason_code"] = "not-evaluated" if ck["applicable"] else "not-applicable"
            row["checksum"] = {
                "status": "not_evaluated" if row["operation"] == "read" else "not_applicable",
                "reason_code": "unsupported" if row["operation"] == "read" else "not-applicable",
                "checksum_valid": None,
                "actual_worker_checksums": None,
                "actual_run_checksum": None,
            }
        unsupported_doc["status"] = "unsupported"
        unsupported_doc["reason_code"] = "unsupported"
        unsupported_doc["counters"].update(
            attempted_loops=0,
            completed_loops=0,
            attempted_measurements=0,
            terminal_measurements=0,
            measured_measurements=0,
            completed_passes="0",
            completed_exact_payload_bytes="0",
        )
        for rec in unsupported_doc["loop_records"]:
            rec["realized_order"] = []
            rec["realized_order_count"] = 0
        for pname, p_agg in unsupported_doc["aggregates"]["patterns"].items():
            if not unsupported_doc["resolved_plan"]["patterns"][pname]["enabled"]:
                continue
            for a in p_agg.values():
                a["status"] = "unavailable"
                a["reason_code"] = "unsupported"
                a["accepted_measurement_ids"] = []
                a["value"] = None
                a["relative_to_forward_pct"] = None
                a["statistics"] = None
                a["quality"]["repeatability_assessment_available"] = False
                a["quality"]["run_complete"] = False
        verdict = verifier.verify(unsupported_doc)
        self.assertTrue(verdict["artifact_consistent"])
        self.assertFalse(verdict["run_accepted"])

        # Preflight !work_plan.valid (both pre-geometry "|invalid" and post-geometry memory-budget-exceeded)
        for post_geometry in (False, True):
            with self.subTest(post_geometry=post_geometry):
                pf = load_fixture(COMPLETE_FIXTURE)
                if not post_geometry:
                    pf["resolved_plan"]["plan_identity"] = "patterns-work-plan-v1|invalid"
                    for p in pf["resolved_plan"]["patterns"].values():
                        p["effective_workers"] = 0
                        p["worker_spans"] = []
                    pf["resolved_plan"]["patterns"]["random"]["random_workload"].update(
                        selected_accesses_per_pass=0,
                        distinct_address_count=0,
                        logical_working_set_bytes="0",
                        random_indices_sha256=None,
                    )
                    pf["memory_budget"]["reason_code"] = "memory-budget-overflow"
                else:
                    pf["environment"]["available_memory_bytes"] = "1048576"
                    allowed = str(int(1048576 * 0.80))
                    pf["memory_budget"].update(
                        available_memory_bytes="1048576",
                        allowed_memory_bytes=allowed,
                        admitted_budget_bytes=allowed,
                        used_fallback=False,
                        reason_code="memory-budget-exceeded",
                    )
                pf["resolved_plan"].update(
                    valid=False,
                    reason_code="memory-budget-exceeded",
                    resources={},
                    operation_plans=[],
                )
                for p in pf["resolved_plan"]["patterns"].values():
                    p.update(enabled=False, status="unavailable", reason_code="memory-budget-exceeded")
                for prefs in pf["resolved_plan"]["frozen_plan_refs"].values():
                    for op in verifier.OPERATIONS:
                        prefs[op] = None
                pf["memory_budget"]["valid"] = False
                pf["calibration"]["attempts"] = {}
                pf["measurements"] = []
                for p_agg in pf["aggregates"]["patterns"].values():
                    for a in p_agg.values():
                        a.update(
                            status="unavailable",
                            reason_code="memory-budget-exceeded",
                            accepted_measurement_ids=[],
                            value=None,
                            relative_to_forward_pct=None,
                            statistics=None,
                        )
                        a["quality"].update(
                            repeatability_assessment_available=False,
                            run_complete=False,
                            cv_warning=False,
                        )
                pf.update(
                    status="failed",
                    reason_code="memory-budget-exceeded",
                    results_complete=False,
                    run_accepted=False,
                    operation_order_balance_complete=False,
                    pattern_order_balance_complete=False,
                    quality_warnings=[w for w in pf["quality_warnings"] if not w.endswith("-high-cv")],
                )
                pf["counters"].update(
                    attempted_loops=0,
                    completed_loops=0,
                    planned_measurements=0,
                    attempted_measurements=0,
                    terminal_measurements=0,
                    measured_measurements=0,
                    planned_passes="0",
                    completed_passes="0",
                    planned_exact_payload_bytes="0",
                    completed_exact_payload_bytes="0",
                )
                for rec in pf["loop_records"]:
                    rec.update(planned_order=[], realized_order=[], realized_order_count=0, measurement_indexes=[])
                pf_verdict = verifier.verify(pf)
                self.assertTrue(pf_verdict["artifact_consistent"])
                self.assertFalse(pf_verdict["run_accepted"])

    def test_execution_prefix_and_missing_raw_timing(self):
        doc = make_truncated("interrupted", COMPLETE_FIXTURE)
        doc["measurements"][4] = load_fixture(COMPLETE_FIXTURE)["measurements"][4]
        self.assert_reason(doc, "invalid-execution-prefix")

        no_raw = load_fixture(COMPLETE_FIXTURE)
        for row in no_raw["measurements"]:
            row["execution"]["timing"].pop("cpu_raw")
        for p_cal in no_raw["calibration"]["attempts"].values():
            for op_attempts in p_cal.values():
                for att in op_attempts:
                    att["execution"]["timing"].pop("cpu_raw")
        v = verifier.verify(no_raw)
        self.assertTrue(v["artifact_consistent"])
        self.assertEqual(v["checks"]["timing"], "elapsed-only")
        with self.assertRaises(common.Rejected) as caught:
            verifier.verify(no_raw, require_raw=True)
        self.assertEqual(caught.exception.reason, "unsupported-missing-raw-timing")
        self.assertTrue(caught.exception.unsupported)

    def test_mutation_reasons(self):
        base = load_fixture(COMPLETE_FIXTURE)
        read_idx = next(r["measurement_id"] for r in base["measurements"] if r["operation"] == "read")
        write_idx = next(r["measurement_id"] for r in base["measurements"] if r["operation"] == "write")
        copy_idx = next(r["measurement_id"] for r in base["measurements"] if r["operation"] == "copy")
        sparse_write_idx = next(
            r["measurement_id"]
            for r in base["measurements"]
            if r["pattern"] == "strided_4096" and r["operation"] == "write"
        )
        seq_write_idx = next(
            r["measurement_id"]
            for r in base["measurements"]
            if r["pattern"] == "sequential_forward" and r["operation"] == "write"
        )

        def mutate_write_dup(d):
            w_rows = [r for r in d["measurements"] if r["operation"] == "write"]
            w_rows[1]["execution"].pop("attempt_seed_uint64_decimal", None)
            w_rows[1]["execution"]["write_pattern_uint64_decimal"] = w_rows[0]["execution"]["write_pattern_uint64_decimal"]

        def mutate_write_even(d):
            d["measurements"][write_idx]["execution"].pop("attempt_seed_uint64_decimal", None)
            d["measurements"][write_idx]["execution"]["write_pattern_uint64_decimal"] = "2"

        def mutate_sentinel_eq_write(d):
            ex = d["measurements"][sparse_write_idx]["execution"]
            ex["pre_op_sentinel_uint64_decimal"] = ex["write_pattern_uint64_decimal"]

        def mutate_actual_fold(d):
            r = d["measurements"][read_idx]
            r["checksum"]["actual_worker_checksums"][0]["state_a_uint64_decimal"] = "1"
            r["checksum"]["actual_run_checksum"]["state_a_uint64_decimal"] = "2"
            r["checksum"]["checksum_valid"] = False
            r["checksum"]["status"] = "invalid"

        def mutate_worker_alignment(d):
            spans = d["resolved_plan"]["patterns"]["sequential_forward"]["worker_spans"]
            spans[0].update(span_bytes="1572865", accesses_per_pass="49152")
            spans[1].update(offset_bytes="1572865", span_bytes="1572863", accesses_per_pass="49151")

        def mutate_cv_threshold_streaming_vs_sparse(d):
            # Strided 64 is streaming (5.0%); random is sparse (10.0%).
            # Mutating cv_warning_threshold_pct on random to 5.0 must be rejected.
            d["aggregates"]["patterns"]["random"]["read"]["quality"]["cv_warning_threshold_pct"] = 5.0

        def mutate_cal_empty_arrays(d):
            for p_ops in d["calibration"]["attempts"].values():
                for op in p_ops:
                    p_ops[op] = []

        def mutate_cal_placeholder_in_complete(d):
            att = d["calibration"]["attempts"]["sequential_forward"]["read"][0]
            att["plan_ref"] = None
            att["terminal"] = False
            att["valid"] = False
            att["reason_code"] = "not-evaluated"
            att["execution"]["status"] = "unavailable"
            att["execution"]["valid"] = None
            att["execution"]["elapsed_seconds"] = None

        def mutate_cal_failed_attempt_in_complete(d):
            att = d["calibration"]["attempts"]["sequential_forward"]["read"][0]
            att["valid"] = False
            att["reason_code"] = "checksum-mismatch"
            att["execution"]["status"] = "invalid"
            att["execution"]["reason_code"] = "checksum-mismatch"
            att["execution"]["valid"] = False
            att["execution"]["checksum"]["status"] = "invalid"
            att["execution"]["checksum"]["checksum_valid"] = False
            att["execution"]["validation"]["checks"][1]["valid"] = False

        def mutate_cal_bogus_purpose_in_complete(d):
            d["calibration"]["attempts"]["sequential_forward"]["read"][0]["purpose"] = "bogus_purpose"

        def mutate_cal_wrong_passes_in_explicit(d):
            src_seed = int(d["seeds"]["domain_seeds"]["src_uint64_decimal"])
            table = verifier.PatternsTable(src_seed)
            op_seed = common.splitmix64_mix(src_seed ^ (1 << 48) ^ (1 << 56) ^ 1)
            spans = [(int(s["offset_bytes"]), int(s["span_bytes"])) for s in d["resolved_plan"]["patterns"]["sequential_forward"]["worker_spans"]]
            exp_workers, exp_run = verifier.expected_read_checksum(table, op_seed, "sequential_forward", 0, 1, spans)
            plans = d["resolved_plan"]["operation_plans"]
            new_ref = len(plans)
            plans.append({
                "plan_ref": new_ref,
                "pattern": "sequential_forward",
                "operation": "read",
                "passes": 1,
                "explicit_iterations": True,
                "total_accesses": "98304",
                "exact_payload_bytes": "3145728",
                "accesses_per_pass": "98304",
                "min_accesses_per_pass": "98304",
                "max_accesses_per_pass": "98304",
                "distinct_address_count": "98304",
                "logical_working_set_bytes": "3145728",
                "completed_phase_cycles": 0,
                "has_untouched_dst_bytes": False,
                "op_seed_uint64_decimal": str(op_seed),
                "expected_checksum": {
                    "status": "available",
                    "reason_code": "valid",
                    "expected_worker_checksums": exp_workers,
                    "expected_run_checksum": exp_run,
                },
            })
            att = d["calibration"]["attempts"]["sequential_forward"]["read"][0]
            att["plan_ref"] = new_ref
            att["execution"]["checksum"]["actual_run_checksum"] = exp_run

        def mutate_cal_duration_quality(d):
            att = d["calibration"]["attempts"]["sequential_forward"]["read"][0]
            att["duration_quality"] = "within-target-window"

        def mutate_cal_worker_count(d):
            att = d["calibration"]["attempts"]["sequential_forward"]["read"][0]
            att["execution"]["created_workers"] = 1

        def mutate_explicit_multi_warmup(d):
            atts = d["calibration"]["attempts"]["sequential_forward"]["read"]
            extra = copy.deepcopy(atts[0])
            extra["attempt_index"] = 0
            atts[0]["attempt_index"] = 1
            atts.insert(0, extra)

        def mutate_budget_allowed_and_fallback(d):
            d["memory_budget"]["allowed_memory_bytes"] = "18446744073709551615"
            d["memory_budget"]["admitted_budget_bytes"] = "18446744073709551615"
            d["memory_budget"]["used_fallback"] = True

        def mutate_budget_admitted_mismatch(d):
            allowed = int(d["memory_budget"]["allowed_memory_bytes"])
            d["memory_budget"]["admitted_budget_bytes"] = str(allowed - 1)

        cases = [
            ("unsupported-schema", lambda d: d.update(schema_version=2), True),
            ("unsupported-mode", lambda d: d.update(mode="cpu_bandwidth"), True),
            ("unsupported-methodology", lambda d: d.update(methodology_version="bad"), True),
            ("missing-top-level-key", lambda d: d.pop("software"), False),
            ("component-mismatch", lambda d: d["resolved_plan"]["component_identities"].update(version="bad"), False),
            ("seed-mismatch", lambda d: d["seeds"]["domain_seeds"].update(src_uint64_decimal="1"), False),
            ("seed-mismatch", lambda d: d["seeds"]["domain_seeds"].update(random_uint64_decimal="1"), False),
            ("seed-mismatch", lambda d: d["seeds"]["domain_seeds"].update(pre_op_sentinel_uint64_decimal="1"), False),
            ("seed-mismatch", lambda d: d["resolved_plan"]["operation_plans"][0].update(op_seed_uint64_decimal="1"), False),
            ("pattern-geometry-mismatch", lambda d: d["resolved_plan"]["patterns"]["strided_2mb"].update(status="skipped", enabled=False), False),
            ("pattern-geometry-mismatch", lambda d: d["resolved_plan"]["patterns"]["strided_2mb"].update(large_page_backing_verified=True), False),
            ("worker-plan-mismatch", mutate_worker_alignment, False),
            ("random-indices-sha256-mismatch", lambda d: d["resolved_plan"]["patterns"]["random"]["random_workload"].update(random_indices_sha256="0" * 64), False),
            ("random-workload-mismatch", lambda d: d["resolved_plan"]["patterns"]["random"]["random_workload"].update(logical_working_set_bytes="1"), False),
            ("plan-reference", lambda d: d["measurements"][0].update(plan_ref=999), False),
            ("plan-bytes-mismatch", lambda d: d["resolved_plan"]["operation_plans"][0].update(exact_payload_bytes="1"), False),
            ("expected-run-checksum-mismatch", lambda d: d["resolved_plan"]["operation_plans"][0]["expected_checksum"]["expected_run_checksum"].update(state_a_uint64_decimal="0"), False),
            ("expected-worker-checksum-mismatch", lambda d: d["resolved_plan"]["operation_plans"][0]["expected_checksum"]["expected_worker_checksums"][0].update(state_a_uint64_decimal="0"), False),
            ("checksum-valid-mismatch", lambda d: d["measurements"][read_idx]["checksum"].update(checksum_valid=False), False),
            ("actual-run-fold-mismatch", mutate_actual_fold, False),
            ("write-pattern-invalid", mutate_write_dup, False),
            ("write-pattern-invalid", mutate_write_even, False),
            ("pre-op-sentinel-invalid", mutate_sentinel_eq_write, False),
            ("pre-op-sentinel-invalid", lambda d: d["measurements"][sparse_write_idx]["execution"].update(pre_op_sentinel_uint64_decimal=None), False),
            ("pre-op-sentinel-unexpected", lambda d: d["measurements"][seq_write_idx]["execution"].update(pre_op_sentinel_uint64_decimal="3"), False),
            ("required-validation-failed", lambda d: d["measurements"][read_idx]["execution"]["validation"]["checks"][1].update(valid=False), False),
            ("required-validation-failed", lambda d: d["measurements"][write_idx]["execution"]["validation"]["checks"][2].update(valid=False), False),
            ("required-validation-failed", lambda d: d["measurements"][copy_idx]["execution"]["validation"]["checks"][3].update(valid=False), False),
            ("required-validation-failed", lambda d: d["measurements"][sparse_write_idx]["execution"]["validation"]["checks"][4].update(valid=False), False),
            ("required-validation-failed", lambda d: d["measurements"][read_idx]["execution"]["validation"]["checks"][5].update(valid=False), False),
            ("validation-applicability", lambda d: d["measurements"][sparse_write_idx]["execution"]["validation"]["checks"][4].update(applicable=False, evaluated=None, valid=None), False),
            ("tick-delta-mismatch", lambda d: d["measurements"][0]["execution"]["timing"]["cpu_raw"].update(stop_ticks="1"), False),
            ("elapsed-mismatch", lambda d: d["measurements"][0]["execution"]["timing"]["cpu_raw"].update(timebase_numer=126), False),
            ("rate-mismatch", lambda d: d["measurements"][0].update(bandwidth_gb_s=1.0), False),
            ("measurement-order-mismatch", lambda d: d["measurements"][0].update(pattern="random"), False),
            ("order-balance-mismatch", lambda d: d.update(pattern_order_balance_complete=not d["pattern_order_balance_complete"]), False),
            ("order-balance-mismatch", lambda d: d.update(operation_order_balance_complete=False), False),
            ("aggregate-headline-mismatch", lambda d: d["aggregates"]["patterns"]["sequential_forward"]["read"].update(value=1.0), False),
            ("aggregate-relative-mismatch", lambda d: d["aggregates"]["patterns"]["random"]["read"].update(relative_to_forward_pct=999.0), False),
            ("aggregate-statistic-mismatch", lambda d: d["aggregates"]["patterns"]["sequential_forward"]["read"]["statistics"].update(average=1.0), False),
            ("aggregate-cv-mismatch", mutate_cv_threshold_streaming_vs_sparse, False),
            ("counter-mismatch", lambda d: d["counters"].update(completed_passes="0"), False),
            ("checkpoint-policy", lambda d: d["checkpoint_lifecycle"].update(snapshot_interval_loops=99), False),
            ("checkpoint-current-persistence", lambda d: d["checkpoint_lifecycle"].update(current_persistence_success=True), False),
            ("run-acceptance-mismatch", lambda d: d.update(run_accepted=False), False),
            ("integer-encoding", lambda d: d["measurements"][0].update(plan_ref=True), False),
            ("integer-encoding", lambda d: d["seeds"].update(base_seed_uint64_decimal="01"), False),
            ("run-status-mismatch", lambda d: d.update(interruption_requested=True), False),
            ("resource-evidence-invalid", lambda d: d["resolved_plan"]["resources"]["src"].update(read_only_protected=False, mprotect_errno=13), False),
            ("resource-evidence-invalid", lambda d: d["resolved_plan"]["resources"].update(random_index_table_bytes="0"), False),
            ("resolved-sources-invalid", lambda d: d["configuration"]["resolved_sources"].update(buffer_size="bogus"), False),
            ("quality-warnings-invalid", lambda d: d.update(quality_warnings=["completely-fabricated-warning"]), False),
            ("interpretation-invalid", lambda d: d["interpretation"].update(large_page_backing_verified=True), False),
            ("memory-budget-arithmetic", lambda d: d["memory_budget"].update(index_structure_bytes="0"), False),
            ("memory-budget-arithmetic", mutate_budget_allowed_and_fallback, False),
            ("memory-budget-arithmetic", lambda d: d["memory_budget"].update(used_fallback=True), False),
            ("memory-budget-arithmetic", lambda d: d["memory_budget"].update(available_memory_bytes="1048576"), False),
            ("memory-budget-arithmetic", mutate_budget_admitted_mismatch, False),
            ("calibration-incomplete", mutate_cal_empty_arrays, False),
            ("calibration-placeholder", mutate_cal_placeholder_in_complete, False),
            ("calibration-valid-mismatch", mutate_cal_failed_attempt_in_complete, False),
            ("calibration-purpose-invalid", mutate_cal_bogus_purpose_in_complete, False),
            ("frozen-plan-reference", mutate_cal_wrong_passes_in_explicit, False),
            ("calibration-duration-quality", mutate_cal_duration_quality, False),
            ("worker-lifecycle", mutate_cal_worker_count, False),
            ("calibration-invalid", mutate_explicit_multi_warmup, False),
            ("metric-null", lambda d: d["measurements"][0]["execution"]["timing"].update(diagnostic_elapsed_seconds=0.123), False),
            ("plan-invalid", lambda d: d["resolved_plan"].update(valid=False), False),
        ]
        for idx, (reason, mutate, unsup) in enumerate(cases):
            with self.subTest(idx=idx, reason=reason):
                doc = load_fixture(COMPLETE_FIXTURE)
                mutate(doc)
                self.assert_reason(doc, reason, unsup)

        # Also check strided_2mb skip forgery on the 1 MiB fixture
        skipped_doc = load_fixture(SKIPPED_2MB_FIXTURE)
        skipped_doc["resolved_plan"]["patterns"]["strided_2mb"].update(enabled=True, status="active", reason_code="active")
        self.assert_reason(skipped_doc, "pattern-geometry-mismatch", False)

        skipped_agg = load_fixture(SKIPPED_2MB_FIXTURE)
        skipped_agg["aggregates"]["patterns"]["strided_2mb"]["read"]["status"] = "unavailable"
        self.assert_reason(skipped_agg, "aggregate-status-mismatch", False)

    def test_strict_json_cli_limits_random_cap_and_binary_binding(self):
        with tempfile.TemporaryDirectory(prefix="patterns-verifier-") as directory:
            d = Path(directory)
            p = d / "input.json"
            p.write_text('{"schema_version":1,"schema_version":1}')
            with self.assertRaises(common.Rejected) as caught:
                common.load_strict_json(p)
            self.assertEqual(caught.exception.reason, "duplicate-json-key")

            p.write_text('{"schema_version":1,"x":NaN}')
            with self.assertRaises(common.Rejected) as caught:
                common.load_strict_json(p)
            self.assertEqual(caught.exception.reason, "nonfinite-json-number")

            p.write_text("[" * 66 + "0" + "]" * 66)
            with self.assertRaises(common.Rejected) as caught:
                common.load_strict_json(p)
            self.assertEqual(caught.exception.reason, "unsupported-structure-limit")
            self.assertTrue(caught.exception.unsupported)

            p.write_bytes(b"\xff\xfe\xfd")
            proc = subprocess.run(
                [sys.executable, str(ROOT / "script-examples/verify_patterns_result.py"), str(p)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 1)
            self.assertEqual(json.loads(proc.stdout)["reason_code"], "invalid-input")

            # --max-random-indices boundary (exit 2 when exceeded, exit 1 when < 1)
            complete_path = COMPLETE_FIXTURE
            res_cap = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "script-examples/verify_patterns_result.py"),
                    str(complete_path),
                    "--max-random-indices",
                    "98303",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(res_cap.returncode, 2, res_cap.stdout + res_cap.stderr)
            verdict_cap = json.loads(res_cap.stdout)
            self.assertEqual(verdict_cap["reason_code"], "unsupported-random-index-limit")
            self.assertIsNone(verdict_cap["artifact_consistent"])

            res_bad_cap = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "script-examples/verify_patterns_result.py"),
                    str(complete_path),
                    "--max-random-indices",
                    "0",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(res_bad_cap.returncode, 1)
            self.assertEqual(json.loads(res_bad_cap.stdout)["reason_code"], "invalid-input")

            # --binary binding
            bin_path = d / "memory_benchmark"
            bin_path.write_bytes(b"known patterns binary")
            doc = load_fixture(COMPLETE_FIXTURE)
            doc["build_manifest"]["binary_sha256"] = hashlib.sha256(
                b"known patterns binary"
            ).hexdigest()
            self.assertEqual(verifier.verify(doc, binary=bin_path)["checks"]["build"], "binary-bound")
            doc["build_manifest"]["binary_sha256"] = "0" * 64
            self.assert_reason(doc, "binary-hash-mismatch", binary=bin_path)
            doc["build_manifest"]["binary_sha256"] = None
            doc["build_manifest"]["status"] = "partial"
            self.assert_reason(doc, "binary-hash-unavailable", binary=bin_path)

    @staticmethod
    def _strip_cli_flag(argv, flag):
        out = list(argv)
        if flag in out:
            idx = out.index(flag)
            del out[idx : idx + 2]
        return out

    def make_sweep(self, runs_count=2):
        base_doc = load_fixture(SKIPPED_2MB_FIXTURE)
        base_doc["configuration"]["output_file"] = ""
        argv = self._strip_cli_flag(base_doc["configuration"]["argv"], "-b")
        argv = self._strip_cli_flag(argv, "-t")
        argv.extend(["-S", f"buffer-size={','.join(['1'] * runs_count)}", "-S", "threads=2"])
        base_doc["configuration"]["argv"] = argv
        base_doc["configuration"]["resolved_sources"]["buffer_size"] = "sweep"
        base_doc["configuration"]["resolved_sources"]["workers"] = "sweep"
        runs = []
        for idx in range(runs_count):
            runs.append(
                {
                    "index": idx,
                    "parameters": {"buffer-size": 1, "threads": 2},
                    "status": "complete",
                    "status_reason": None,
                    "result": copy.deepcopy(base_doc),
                }
            )
        return {
            "configuration": {
                "mode": "sweep",
                "base_mode": "patterns",
                "sweep_schema_version": 1,
                "run_count": runs_count,
                "sweep_max_runs": 16,
                "sweep_parameters": {"buffer-size": [1] * runs_count, "threads": [2]},
            },
            "planned_runs": runs_count,
            "attempted_runs": runs_count,
            "completed_runs": runs_count,
            "status": "complete",
            "status_reason": None,
            "conclusions_valid": True,
            "execution_time_sec": 0.5,
            "timestamp": "2026-11-27T15:41:00Z",
            "version": "0.66.0",
            "runs": runs,
        }

    def test_sweep_verification_mutations_single_hash_budget_and_resolved_sources(self):
        sweep = self.make_sweep(runs_count=2)
        sorted_sweep = json.loads(json.dumps(sweep, sort_keys=True))
        verdict = verifier.verify(sorted_sweep, require_raw=True)
        self.assertTrue(verdict["artifact_consistent"])
        self.assertTrue(verdict["run_accepted"])
        self.assertEqual(verdict["artifact_kind"], "patterns_sweep")

        with tempfile.NamedTemporaryFile() as bin_file:
            bin_file.write(b"patterns-sweep-binary")
            bin_file.flush()
            digest = hashlib.sha256(b"patterns-sweep-binary").hexdigest()
            for r in sweep["runs"]:
                r["result"]["build_manifest"]["binary_sha256"] = digest
            with mock.patch.object(common, "hash_binary", wraps=common.hash_binary) as mocked:
                v = verifier.verify(sweep, binary=Path(bin_file.name))
                self.assertEqual(v["checks"]["build"], "binary-bound")
                self.assertEqual(mocked.call_count, 1)

        single_work = verifier.verify(self.make_sweep(1))["oracle_work_used"]
        two_run = self.make_sweep(2)
        two_work = verifier.verify(two_run)["oracle_work_used"]
        self.assertGreater(two_work, single_work)
        with self.assertRaises(common.Rejected) as caught:
            verifier.verify(two_run, budget=common.Budget(limit=single_work + 1))
        self.assertEqual(caught.exception.reason, "unsupported-work-limit")
        self.assertTrue(caught.exception.unsupported)

        sweep_mutations = [
            ("unsupported-sweep-schema", lambda e: e["configuration"].update(sweep_schema_version=2), True),
            ("sweep-plan-count-mismatch", lambda e: e.update(planned_runs=99), False),
            ("sweep-nested-output-file-mismatch", lambda e: e["runs"][0]["result"]["configuration"].update(output_file="bad.json"), False),
            ("sweep-base-config-mismatch", lambda e: e["runs"][1]["result"]["build_manifest"].update(compiler="gcc"), False),
            ("sweep-invalid-execution-prefix", lambda e: e["runs"][0].update(status="failed", status_reason="err", result=None), False),
            ("resolved-sources-invalid", lambda e: e["runs"][0]["result"]["configuration"]["resolved_sources"].update(buffer_size="default"), False),
            ("resolved-sources-invalid", lambda e: e["runs"][0]["result"]["configuration"]["resolved_sources"].update(workers="auto"), False),
            (
                "resolved-sources-invalid",
                lambda e: (
                    e["runs"][0]["result"]["configuration"]["resolved_sources"].update(workers="auto"),
                    e["runs"][0]["result"]["configuration"].update(worker_source="auto"),
                    e["runs"][0]["result"]["resolved_plan"].update(
                        plan_identity=e["runs"][0]["result"]["resolved_plan"]["plan_identity"].replace(
                            "worker_source=user", "worker_source=auto"
                        )
                    ),
                ),
                False,
            ),
            ("resolved-sources-invalid", lambda e: e["runs"][0]["result"]["configuration"]["resolved_sources"].update(buffer_size="explicit") or e["runs"][0]["result"]["configuration"]["argv"].extend(["-b", "1"]), False),
        ]
        for idx, (reason, mutate, unsup) in enumerate(sweep_mutations):
            with self.subTest(idx=idx, reason=reason):
                env = self.make_sweep(2)
                mutate(env)
                self.assert_reason(env, reason, unsup)

    def test_null_and_invalid_nested_containers_raise_typed_rejections(self):
        cases = [
            ("measurement-invalid", lambda d: d.update(measurements=None)),
            ("measurement-invalid", lambda d: d.update(measurements=[123])),
            (
                "validation-invalid",
                lambda d: d["calibration"]["attempts"]["sequential_forward"]["read"][0]["execution"].update(validation=None),
            ),
            (
                "timing-invalid",
                lambda d: d["calibration"]["attempts"]["sequential_forward"]["read"][0]["execution"].update(timing=None),
            ),
            (
                "validation-invalid",
                lambda d: d["measurements"][0]["execution"].update(validation=None),
            ),
            (
                "timing-invalid",
                lambda d: d["measurements"][0]["execution"].update(timing=None),
            ),
        ]
        for reason, mutate in cases:
            with self.subTest(reason=reason):
                doc = load_fixture(COMPLETE_FIXTURE)
                mutate(doc)
                self.assert_reason(doc, reason, unsupported=False)

    @staticmethod
    def _make_auto_calibrated_doc(elapsed_by_pattern_op=None):
        doc = load_fixture(SKIPPED_2MB_FIXTURE)
        cfg = doc["configuration"]
        argv = list(cfg["argv"])
        idx = argv.index("-i")
        del argv[idx : idx + 2]
        cfg["argv"] = argv
        cfg["iterations"] = None
        cfg["work_policy"] = "automatic_calibration"
        cfg["resolved_sources"]["iterations"] = "auto"
        fwd_eff = doc["resolved_plan"]["patterns"]["sequential_forward"]["effective_workers"]
        doc["resolved_plan"]["plan_identity"] = (
            f"patterns-work-plan-v1|buffer_size_mb=1|workers={fwd_eff}"
            f"|worker_source=user|iterations=auto|loop_count=3|non_cacheable=0"
        )

        plans = doc["resolved_plan"]["operation_plans"]
        for p in plans:
            p["explicit_iterations"] = False

        # Real auto-calibrated runs execute calibration_shape_warmup + pilot (+ optional corrections)
        # before frozen_measurement_warmup.
        ordinal = 1000
        for pname, per_op in doc["calibration"]["attempts"].items():
            for op, atts in per_op.items():
                frozen_att = copy.deepcopy(atts[0])
                op_seed = int(plans[frozen_att["plan_ref"]]["op_seed_uint64_decimal"])
                sent_raw = frozen_att["execution"].get("pre_op_sentinel_uint64_decimal")
                sent_val = int(sent_raw) if sent_raw is not None else None
                new_atts = []
                for idx_att, purpose in enumerate(("calibration_shape_warmup", "pilot")):
                    att = copy.deepcopy(frozen_att)
                    att["attempt_index"] = idx_att
                    att["purpose"] = purpose
                    if op == "write":
                        while True:
                            ordinal += 1
                            attempt_seed = common.splitmix64_mix(op_seed ^ (0x5752495400000000 | ordinal))
                            wp = common.splitmix64_mix(attempt_seed) | 1
                            if wp != 0 and wp != sent_val:
                                break
                        att["execution"]["attempt_seed_uint64_decimal"] = str(attempt_seed)
                        att["execution"]["write_pattern_uint64_decimal"] = str(wp)
                    new_atts.append(att)
                frozen_att["attempt_index"] = 2
                new_atts.append(frozen_att)
                per_op[op] = new_atts

        elapsed_by_pattern_op = elapsed_by_pattern_op or {}
        for row in doc["measurements"]:
            if row["status"] != "measured":
                continue
            target_sec = elapsed_by_pattern_op.get((row["pattern"], row["operation"]), 0.150)
            if callable(target_sec):
                target_sec = target_sec(row["loop_index"])
            raw = row["execution"]["timing"]["cpu_raw"]
            numer = raw["timebase_numer"]
            denom = raw["timebase_denom"]
            dt = round(target_sec * 1e9 * denom / numer)
            start = int(raw["start_ticks"])
            raw["stop_ticks"] = str(start + dt)
            raw["delta_ticks"] = str(dt)
            elapsed = dt * numer / denom / 1e9
            row["execution"]["timing"]["elapsed_seconds"] = elapsed
            exact_bytes = int(plans[row["plan_ref"]]["exact_payload_bytes"])
            row["bandwidth_gb_s"] = exact_bytes / elapsed / 1e9

        rows = doc["measurements"]
        for pname, p_agg in doc["aggregates"]["patterns"].items():
            if not doc["resolved_plan"]["patterns"][pname]["enabled"]:
                continue
            threshold = 5.0 if pname in verifier.STREAMING_PATTERNS else 10.0
            for op, a in p_agg.items():
                vals = [
                    r["bandwidth_gb_s"]
                    for r in rows
                    if r["pattern"] == pname and r["operation"] == op and r["status"] == "measured"
                ]
                st = common.compute_statistics(vals)
                a["value"] = st["median"]
                a["statistics"] = st
                cv_pct = st["coefficient_of_variation_pct"]
                a["quality"]["cv_warning"] = (
                    len(vals) >= 3 and cv_pct is not None and cv_pct > threshold
                )
        for op in verifier.OPERATIONS:
            fwd_val = doc["aggregates"]["patterns"]["sequential_forward"][op]["value"]
            for pname in verifier.CANONICAL_PATTERNS:
                a = doc["aggregates"]["patterns"][pname][op]
                if a["value"] is not None:
                    a["relative_to_forward_pct"] = 100.0 * a["value"] / fwd_val
        return doc

    def test_resolved_sources_two_way_quality_duration_window_cv_and_madvise_errno(self):
        # Two-way check_resolved_sources
        no_sweep_flag = load_fixture(SKIPPED_2MB_FIXTURE)
        no_sweep_flag["configuration"]["resolved_sources"]["buffer_size"] = "sweep"
        self.assert_reason(no_sweep_flag, "resolved-sources-invalid")

        omitted_explicit = load_fixture(SKIPPED_2MB_FIXTURE)
        argv = list(omitted_explicit["configuration"]["argv"])
        idx = argv.index("-b")
        del argv[idx : idx + 2]
        omitted_explicit["configuration"]["argv"] = argv
        omitted_explicit["configuration"]["resolved_sources"]["buffer_size"] = "default"
        self.assertTrue(verifier.verify(omitted_explicit, require_raw=True)["run_accepted"])
        omitted_explicit["configuration"]["resolved_sources"]["buffer_size"] = "explicit"
        self.assert_reason(omitted_explicit, "resolved-sources-invalid")

        # Auto-calibration duration warnings ([0.100, 0.250]) with target_key="pattern"
        within_doc = self._make_auto_calibrated_doc(
            {("sequential_forward", "read"): 0.100, ("random", "write"): 0.250}
        )
        within_doc["quality_warnings"] = []
        self.assertTrue(verifier.verify(within_doc, require_raw=True)["run_accepted"])

        out_of_window = self._make_auto_calibrated_doc(
            {("sequential_forward", "read"): 0.050, ("random", "write"): 0.300}
        )
        out_of_window["quality_warnings"] = []
        self.assert_reason(out_of_window, "quality-warnings-invalid")

        exact_warn = copy.deepcopy(out_of_window)
        exact_warn["quality_warnings"] = [
            "sequential_forward-read-duration-below-target-window",
            "random-write-duration-above-target-window",
        ]
        self.assertTrue(verifier.verify(exact_warn, require_raw=True)["run_accepted"])

        # Differential CV threshold: ~7% CV triggers high-cv on streaming strided_64 (5%) but NOT sparse random (10%)
        cv_doc = self._make_auto_calibrated_doc(
            {
                ("strided_64", "read"): lambda loop_i: (0.140, 0.150, 0.161)[loop_i],
                ("random", "read"): lambda loop_i: (0.140, 0.150, 0.161)[loop_i],
            }
        )
        self.assertTrue(cv_doc["aggregates"]["patterns"]["strided_64"]["read"]["quality"]["cv_warning"])
        self.assertFalse(cv_doc["aggregates"]["patterns"]["random"]["read"]["quality"]["cv_warning"])
        cv_doc["quality_warnings"] = ["strided_64-read-high-cv"]
        self.assertTrue(verifier.verify(cv_doc, require_raw=True)["run_accepted"])

        # check_buffer_evidence madvise_applied=False requires madvise_errno != 0
        bad_madv = load_fixture(SKIPPED_2MB_FIXTURE)
        bad_madv["resolved_plan"]["resources"]["src"]["madvise_applied"] = False
        self.assert_reason(bad_madv, "resource-evidence-invalid")

        valid_madv_fail = load_fixture(SKIPPED_2MB_FIXTURE)
        valid_madv_fail["configuration"]["non_cacheable"] = True
        valid_madv_fail["interpretation"]["non_cacheable_requested"] = True
        fwd_eff = valid_madv_fail["resolved_plan"]["patterns"]["sequential_forward"]["effective_workers"]
        valid_madv_fail["resolved_plan"]["plan_identity"] = (
            f"patterns-work-plan-v1|buffer_size_mb=1|workers={fwd_eff}"
            f"|worker_source=user|iterations=2|loop_count=3|non_cacheable=1"
        )
        for buf in (
            valid_madv_fail["resolved_plan"]["resources"]["src"],
            valid_madv_fail["resolved_plan"]["resources"]["dst"],
        ):
            buf["non_cacheable_requested"] = True
            buf["madvise_advice"] = 1
        valid_madv_fail["resolved_plan"]["resources"]["src"].update(
            madvise_applied=False, madvise_errno=22
        )
        # Preserve only existing non-madvise warnings if any, or recompute cleanly:
        non_cv_warnings = [w for w in valid_madv_fail["quality_warnings"] if w != "madvise-random-not-applied"]
        valid_madv_fail["quality_warnings"] = non_cv_warnings + ["madvise-random-not-applied"]
        self.assertTrue(verifier.verify(valid_madv_fail, require_raw=True)["run_accepted"])

    @staticmethod
    def _make_pre_freeze_interrupted_doc(buffer_size_mb):
        fixture = COMPLETE_FIXTURE if buffer_size_mb >= 3 else SKIPPED_2MB_FIXTURE
        doc = load_fixture(fixture)
        cfg = doc["configuration"]
        cfg["buffer_size_mb"] = buffer_size_mb
        cfg["iterations"] = None
        cfg["work_policy"] = "automatic_calibration"
        cfg["resolved_sources"]["iterations"] = "auto"
        argv = list(cfg["argv"])
        if "-i" in argv:
            idx = argv.index("-i")
            del argv[idx : idx + 2]
        if "-b" in argv:
            b_idx = argv.index("-b")
            argv[b_idx + 1] = str(buffer_size_mb)
        cfg["argv"] = argv

        buffer_bytes = buffer_size_mb * 1048576
        page_size = int(doc["environment"]["page_size_bytes"])
        page_rounded = ((buffer_bytes + page_size - 1) // page_size) * page_size
        eff_workers = cfg["requested_workers"]

        rp = doc["resolved_plan"]
        rp["plan_identity"] = (
            f"patterns-work-plan-v1|buffer_size_mb={buffer_size_mb}|workers={eff_workers}"
            f"|worker_source=user|iterations=auto|loop_count={cfg['loop_count']}|non_cacheable=0"
        )
        rp["resources"] = {}
        rp["operation_plans"] = []
        for pname in verifier.CANONICAL_PATTERNS:
            for op in verifier.OPERATIONS:
                rp["frozen_plan_refs"][pname][op] = None

        spans = common.partition_64b(buffer_bytes, eff_workers)
        total_slots = buffer_bytes // verifier.ACCESS_SIZE_BYTES
        selected_random = min(
            total_slots,
            min(verifier.MAX_RANDOM_ACCESSES, max(verifier.MIN_RANDOM_ACCESSES, total_slots)),
        )
        rand_counts = verifier.distribute_random_accesses(spans, selected_random)
        active_patterns = []
        for pname in verifier.CANONICAL_PATTERNS:
            pdata = rp["patterns"][pname]
            if not pdata["enabled"]:
                continue
            active_patterns.append(pname)
            stride = verifier.STRIDE_BYTES[pname]
            if pname in ("sequential_forward", "sequential_reverse"):
                apps = [sp_len // verifier.ACCESS_SIZE_BYTES for _, sp_len in spans]
                rcs = [0] * eff_workers
            elif pname in verifier.STRIDED_PATTERNS:
                apps = [1 + (sp_len - verifier.ACCESS_SIZE_BYTES) // stride for _, sp_len in spans]
                rcs = [0] * eff_workers
            else:
                apps = list(rand_counts)
                rcs = list(rand_counts)
            pdata["effective_workers"] = eff_workers
            pdata["worker_spans"] = [
                {
                    "offset_bytes": str(off),
                    "span_bytes": str(sp_len),
                    "accesses_per_pass": str(app),
                    "random_access_count": rc,
                }
                for (off, sp_len), app, rc in zip(spans, apps, rcs)
            ]
            if pname == "random":
                pdata["random_workload"].update(
                    selected_accesses_per_pass=selected_random,
                    distinct_address_count=selected_random,
                    logical_working_set_bytes="0",
                    random_indices_sha256=None,
                )

        mb = doc["memory_budget"]
        known_peak = 2 * page_rounded + int(mb["transient_peak_bytes"])
        mb.update(
            requested_data_bytes=str(2 * buffer_bytes),
            committed_data_bytes=str(2 * page_rounded),
            resource_rounding_bytes=str(2 * page_rounded - 2 * buffer_bytes),
            index_structure_bytes=str(2 * selected_random * 8),
            setup_peak_bytes=str(4096 + 64 * eff_workers + 2 * selected_random * 8),
            known_owned_peak_bytes=str(known_peak),
            runtime_peak_bytes=str(known_peak),
        )

        doc["calibration"]["attempts"] = {
            pname: {op: [] for op in verifier.OPERATIONS}
            for pname in active_patterns
        }

        random_sparse = (selected_random * verifier.ACCESS_SIZE_BYTES) < buffer_bytes
        for row in doc["measurements"]:
            pname = row["pattern"]
            op = row["operation"]
            has_untouched = (
                pname in verifier.STRIDED_PATTERNS
                or (pname == "random" and random_sparse)
            )
            row["plan_ref"] = None
            row["attempted"] = False
            row["status"] = "interrupted"
            row["reason_code"] = "interruption-before-task"
            row["bandwidth_gb_s"] = None
            row["completed_passes"] = 0
            row["completed_exact_payload_bytes"] = "0"
            ex = row["execution"]
            ex["status"] = "not_run"
            ex["reason_code"] = "interruption-before-task"
            ex["valid"] = None
            ex["write_pattern_uint64_decimal"] = None
            ex["pre_op_sentinel_uint64_decimal"] = None
            ex.pop("attempt_seed_uint64_decimal", None)
            ex["timing"] = {
                "evaluated": False,
                "valid": None,
                "elapsed_seconds": None,
                "diagnostic_elapsed_seconds": None,
                "cpu_raw": None,
            }
            ex["validation"]["checks"] = [
                {
                    "kind": kind,
                    "applicable": applicable,
                    "evaluated": False if applicable else None,
                    "valid": None,
                    "reason_code": "not-evaluated" if applicable else "not-applicable",
                }
                for kind, applicable in verifier._validation_specs_for_op(op, has_untouched)
            ]
            row["checksum"] = {
                "status": "not_evaluated" if op == "read" else "not_applicable",
                "reason_code": "interruption-before-task" if op == "read" else "not-applicable",
                "checksum_valid": None,
                "actual_worker_checksums": None,
                "actual_run_checksum": None,
            }

        for rec in doc["loop_records"]:
            rec["realized_order"] = []
            rec["realized_order_count"] = 0

        doc.update(
            status="interrupted",
            reason_code="interruption-requested",
            results_complete=False,
            run_accepted=False,
            interruption_requested=True,
            operation_order_balance_complete=False,
            pattern_order_balance_complete=False,
            quality_warnings=[w for w in doc["quality_warnings"] if not w.endswith("-high-cv")],
        )
        doc["counters"].update(
            attempted_loops=0,
            completed_loops=0,
            attempted_measurements=0,
            terminal_measurements=len(doc["measurements"]),
            measured_measurements=0,
            planned_passes="0",
            completed_passes="0",
            planned_exact_payload_bytes="0",
            completed_exact_payload_bytes="0",
        )
        for pname, p_agg in doc["aggregates"]["patterns"].items():
            if not rp["patterns"][pname]["enabled"]:
                continue
            for a in p_agg.values():
                a.update(
                    status="unavailable",
                    reason_code="interruption-requested",
                    accepted_measurement_ids=[],
                    value=None,
                    relative_to_forward_pct=None,
                    statistics=None,
                )
                a["quality"].update(
                    repeatability_assessment_available=False,
                    run_complete=False,
                    cv_warning=False,
                )
        return doc

    def test_pre_freeze_interrupted_null_plan_ref_validation_applicability(self):
        # 1 MiB: random is dense (applicable=False), strided_64..strided_16384 are True.
        doc_1mb = self._make_pre_freeze_interrupted_doc(1)
        verdict_1mb = verifier.verify(doc_1mb)
        self.assertTrue(verdict_1mb["artifact_consistent"])
        self.assertFalse(verdict_1mb["run_accepted"])

        s64_write_1mb = next(
            r["measurement_id"]
            for r in doc_1mb["measurements"]
            if r["pattern"] == "strided_64" and r["operation"] == "write"
        )
        bad_s64 = copy.deepcopy(doc_1mb)
        bad_s64["measurements"][s64_write_1mb]["execution"]["validation"]["checks"][4].update(
            applicable=False, evaluated=None, reason_code="not-applicable"
        )
        self.assert_reason(bad_s64, "validation-applicability", unsupported=False)

        rand_write_1mb = next(
            r["measurement_id"]
            for r in doc_1mb["measurements"]
            if r["pattern"] == "random" and r["operation"] == "write"
        )
        bad_rand_1mb = copy.deepcopy(doc_1mb)
        bad_rand_1mb["measurements"][rand_write_1mb]["execution"]["validation"]["checks"][4].update(
            applicable=True, evaluated=False, reason_code="not-evaluated"
        )
        self.assert_reason(bad_rand_1mb, "validation-applicability", unsupported=False)

        # 32 MiB: random is sparse (1_000_000 * 32 < 32 MiB -> applicable=True).
        doc_32mb = self._make_pre_freeze_interrupted_doc(32)
        verdict_32mb = verifier.verify(doc_32mb)
        self.assertTrue(verdict_32mb["artifact_consistent"])
        self.assertFalse(verdict_32mb["run_accepted"])

        rand_write_32mb = next(
            r["measurement_id"]
            for r in doc_32mb["measurements"]
            if r["pattern"] == "random" and r["operation"] == "write"
        )
        bad_rand_32mb = copy.deepcopy(doc_32mb)
        bad_rand_32mb["measurements"][rand_write_32mb]["execution"]["validation"]["checks"][4].update(
            applicable=False, evaluated=None, reason_code="not-applicable"
        )
        self.assert_reason(bad_rand_32mb, "validation-applicability", unsupported=False)

    def test_progress_checkpoint_fallback_budget_and_preflight_rejections(self):
        # 1. Mid-run "progress" checkpoint snapshot after loop 0 (18 measured, 36 not_run).
        progress_doc = load_fixture(SKIPPED_2MB_FIXTURE)
        keep = 18  # 6 active patterns * 3 ops = loop 0 complete
        for idx, row in enumerate(progress_doc["measurements"]):
            if idx < keep:
                continue
            row["attempted"] = False
            row["status"] = "not_run"
            row["reason_code"] = "not-run"
            row["bandwidth_gb_s"] = None
            row["completed_passes"] = 0
            row["completed_exact_payload_bytes"] = "0"
            ex = row["execution"]
            ex["status"] = "not_run"
            ex["reason_code"] = "not-run"
            ex["valid"] = None
            ex["write_pattern_uint64_decimal"] = None
            ex["pre_op_sentinel_uint64_decimal"] = None
            ex.pop("attempt_seed_uint64_decimal", None)
            ex["timing"] = {
                "evaluated": False,
                "valid": None,
                "elapsed_seconds": None,
                "diagnostic_elapsed_seconds": None,
                "cpu_raw": None,
            }
            for ck in ex["validation"]["checks"]:
                ck["evaluated"] = False if ck["applicable"] else None
                ck["valid"] = None
                ck["reason_code"] = "not-evaluated" if ck["applicable"] else "not-applicable"
            row["checksum"] = {
                "status": "not_evaluated" if row["operation"] == "read" else "not_applicable",
                "reason_code": "not-run" if row["operation"] == "read" else "not-applicable",
                "checksum_valid": None,
                "actual_worker_checksums": None,
                "actual_run_checksum": None,
            }
        rows = progress_doc["measurements"]
        for loop_idx, rec in enumerate(progress_doc["loop_records"]):
            slice_rows = rows[loop_idx * keep : (loop_idx + 1) * keep]
            realized = [
                {"pattern": r["pattern"], "operation": r["operation"]}
                for r in slice_rows
                if r["attempted"]
            ]
            rec["realized_order"] = realized
            rec["realized_order_count"] = len(realized)
        progress_doc["counters"].update(
            attempted_loops=1,
            completed_loops=1,
            attempted_measurements=keep,
            terminal_measurements=keep,
            measured_measurements=keep,
            completed_passes=str(sum(r["completed_passes"] for r in rows)),
            completed_exact_payload_bytes=str(sum(int(r["completed_exact_payload_bytes"]) for r in rows)),
        )
        for pname, p_agg in progress_doc["aggregates"]["patterns"].items():
            if not progress_doc["resolved_plan"]["patterns"][pname]["enabled"]:
                continue
            for op, a in p_agg.items():
                ids = [r["measurement_id"] for r in rows if r["pattern"] == pname and r["operation"] == op and r["status"] == "measured"]
                vals = [rows[idx]["bandwidth_gb_s"] for idx in ids]
                st = common.compute_statistics(vals)
                a.update(accepted_measurement_ids=ids, status="measured", reason_code="measured", value=st["median"], statistics=st)
                a["quality"].update(repeatability_assessment_available=False, run_complete=False, cv_warning=False)
        for op in verifier.OPERATIONS:
            fwd_val = progress_doc["aggregates"]["patterns"]["sequential_forward"][op]["value"]
            for pname in verifier.CANONICAL_PATTERNS:
                a = progress_doc["aggregates"]["patterns"][pname][op]
                if a["value"] is not None:
                    a["relative_to_forward_pct"] = 100.0 * a["value"] / fwd_val
        progress_doc.update(
            status="partial",
            reason_code="partial-results",
            results_complete=False,
            run_accepted=False,
            operation_order_balance_complete=False,
            pattern_order_balance_complete=False,
            quality_warnings=[w for w in progress_doc["quality_warnings"] if not w.endswith("-high-cv")],
        )
        progress_doc["checkpoint_lifecycle"].update(
            current_request="progress",
            prior_file_writer_attempts=0,
            prior_successful_file_writes=0,
        )
        prog_verdict = verifier.verify(progress_doc, require_raw=True)
        self.assertTrue(prog_verdict["artifact_consistent"])
        self.assertFalse(prog_verdict["run_accepted"])

        # 2. Fallback 2048 MiB memory budget when environment.available_memory_bytes is None.
        fb_doc = load_fixture(SKIPPED_2MB_FIXTURE)
        fb_doc["environment"]["available_memory_bytes"] = None
        fb_doc["environment"]["available_memory_source"] = "unavailable"
        fb_limit = str(2048 * 1048576)
        fb_doc["memory_budget"].update(
            available_memory_bytes="0",
            used_fallback=True,
            allowed_memory_bytes=fb_limit,
            admitted_budget_bytes=fb_limit,
        )
        self.assertTrue(verifier.verify(fb_doc, require_raw=True)["run_accepted"])
        fb_bad = copy.deepcopy(fb_doc)
        fb_bad["memory_budget"]["allowed_memory_bytes"] = str(1024 * 1048576)
        fb_bad["memory_budget"]["admitted_budget_bytes"] = str(1024 * 1048576)
        self.assert_reason(fb_bad, "memory-budget-arithmetic")

        # 3. Preflight work-cap-exceeded, measurement-limit-exceeded, and buffer-bytes-overflow
        #    artifacts (always tested synthetically, plus live against ./memory_benchmark when built).
        for reason in ("work-cap-exceeded", "measurement-limit-exceeded", "buffer-bytes-overflow"):
            with self.subTest(synthetic_preflight=reason):
                pf = load_fixture(SKIPPED_2MB_FIXTURE)
                if reason == "work-cap-exceeded":
                    pf["configuration"]["iterations"] = 10000
                    pf["configuration"]["buffer_size_mb"] = 10
                    pf["resolved_plan"]["plan_identity"] = "patterns-work-plan-v1|invalid"
                else:
                    pf["resolved_plan"]["plan_identity"] = "patterns-work-plan-v1|invalid"
                    for p in pf["resolved_plan"]["patterns"].values():
                        p["effective_workers"] = 0
                        p["worker_spans"] = []
                    pf["resolved_plan"]["patterns"]["random"]["random_workload"].update(
                        selected_accesses_per_pass=0,
                        distinct_address_count=0,
                        logical_working_set_bytes="0",
                        random_indices_sha256=None,
                    )
                    pf["memory_budget"].update(
                        requested_data_bytes="0",
                        committed_data_bytes="0",
                        resource_rounding_bytes="0",
                        index_structure_bytes="0",
                        transient_peak_bytes="0",
                        setup_peak_bytes="0",
                        runtime_peak_bytes="0",
                        known_owned_peak_bytes="0",
                        admitted_budget_bytes="0",
                        available_memory_bytes="0",
                        allowed_memory_bytes="0",
                        used_fallback=False,
                        valid=False,
                        reason_code="memory-budget-overflow",
                    )
                    if reason == "measurement-limit-exceeded":
                        pf["configuration"]["loop_count"] = 20000
                        pf["counters"]["planned_loops"] = 20000
                        pf["loop_records"] = [
                            {"loop_index": i, "planned_order": [], "realized_order": [], "realized_order_count": 0, "measurement_indexes": []}
                            for i in range(20000)
                        ]
                    else:
                        pf["configuration"]["buffer_size_mb"] = 9007199254740991
                if reason == "work-cap-exceeded":
                    pf["configuration"]["buffer_size_mb"] = 1
                    pf["configuration"]["iterations"] = 100000
                    pf["resolved_plan"]["plan_identity"] = (
                        f"patterns-work-plan-v1|buffer_size_mb=1"
                        f"|workers={pf['resolved_plan']['patterns']['sequential_forward']['effective_workers']}"
                        f"|worker_source=user|iterations=100000|loop_count=3|non_cacheable=0"
                    )
                pf["resolved_plan"].update(
                    valid=False,
                    reason_code=reason,
                    resources={},
                    operation_plans=[],
                )
                for p in pf["resolved_plan"]["patterns"].values():
                    p.update(enabled=False, status="unavailable", reason_code=reason)
                for prefs in pf["resolved_plan"]["frozen_plan_refs"].values():
                    for op in verifier.OPERATIONS:
                        prefs[op] = None
                pf["calibration"]["attempts"] = {}
                pf["measurements"] = []
                for p_agg in pf["aggregates"]["patterns"].values():
                    for a in p_agg.values():
                        a.update(
                            status="unavailable",
                            reason_code=reason,
                            accepted_measurement_ids=[],
                            value=None,
                            relative_to_forward_pct=None,
                            statistics=None,
                        )
                        a["quality"].update(
                            repeatability_assessment_available=False,
                            run_complete=False,
                            cv_warning=False,
                        )
                pf.update(
                    status="failed",
                    reason_code=reason,
                    results_complete=False,
                    run_accepted=False,
                    operation_order_balance_complete=False,
                    pattern_order_balance_complete=False,
                    quality_warnings=[w for w in pf["quality_warnings"] if not w.endswith("-high-cv")],
                )
                pf["counters"].update(
                    attempted_loops=0,
                    completed_loops=0,
                    planned_measurements=0,
                    attempted_measurements=0,
                    terminal_measurements=0,
                    measured_measurements=0,
                    planned_passes="0",
                    completed_passes="0",
                    planned_exact_payload_bytes="0",
                    completed_exact_payload_bytes="0",
                )
                for rec in pf["loop_records"]:
                    rec.update(planned_order=[], realized_order=[], realized_order_count=0, measurement_indexes=[])
                if reason != "measurement-limit-exceeded":
                    pf["checkpoint_lifecycle"].update(snapshot_interval_loops=1)
                else:
                    pf["checkpoint_lifecycle"].update(snapshot_interval_loops=2500)
                verdict = verifier.verify(pf)
                self.assertTrue(verdict["artifact_consistent"])
                self.assertFalse(verdict["run_accepted"])

        bin_path = ROOT / "memory_benchmark"
        if bin_path.is_file():
            with tempfile.TemporaryDirectory(prefix="patterns-preflight-") as directory:
                d = Path(directory)
                for name, cli_args, expected_reason in (
                    ("work_cap", ["-P", "-b", "10", "-t", "1", "-i", "10000", "-r", "1"], "work-cap-exceeded"),
                    ("meas_limit", ["-P", "-b", "1", "-t", "1", "-i", "1", "-r", "20000"], "measurement-limit-exceeded"),
                    ("buf_overflow", ["-P", "-b", "9007199254740991"], "buffer-bytes-overflow"),
                ):
                    with self.subTest(preflight=name):
                        out_p = d / f"{name}.json"
                        proc = subprocess.run(
                            [str(bin_path), *cli_args, "-o", str(out_p)],
                            capture_output=True,
                            text=True,
                        )
                        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
                        raw_doc = json.loads(out_p.read_text())
                        self.assertEqual(raw_doc["reason_code"], expected_reason)
                        verdict = verifier.verify(raw_doc, binary=bin_path)
                        self.assertTrue(verdict["artifact_consistent"])
                        self.assertFalse(verdict["run_accepted"])

    def test_calibration_provenance_timing_nullability_and_budget_consistency(self):
        def inject_seq_fwd_read_plan(doc, passes):
            src_seed = int(doc["seeds"]["domain_seeds"]["src_uint64_decimal"])
            table = verifier.PatternsTable(src_seed)
            op_seed = common.splitmix64_mix(src_seed ^ (1 << 48) ^ (1 << 56) ^ passes)
            spans = [
                (int(s["offset_bytes"]), int(s["span_bytes"]))
                for s in doc["resolved_plan"]["patterns"]["sequential_forward"]["worker_spans"]
            ]
            exp_workers, exp_run = verifier.expected_read_checksum(
                table, op_seed, "sequential_forward", 0, passes, spans
            )
            plans = doc["resolved_plan"]["operation_plans"]
            new_ref = len(plans)
            total_acc = sum(sp_len // 32 for _, sp_len in spans) * passes
            exact_bytes = total_acc * 32
            plans.append(
                {
                    "plan_ref": new_ref,
                    "pattern": "sequential_forward",
                    "operation": "read",
                    "passes": passes,
                    "explicit_iterations": False,
                    "total_accesses": str(total_acc),
                    "exact_payload_bytes": str(exact_bytes),
                    "accesses_per_pass": str(sum(sp_len // 32 for _, sp_len in spans)),
                    "min_accesses_per_pass": str(sum(sp_len // 32 for _, sp_len in spans)),
                    "max_accesses_per_pass": str(sum(sp_len // 32 for _, sp_len in spans)),
                    "distinct_address_count": str(sum(sp_len // 32 for _, sp_len in spans)),
                    "logical_working_set_bytes": str(doc["configuration"]["buffer_size_mb"] * 1048576),
                    "completed_phase_cycles": 0,
                    "has_untouched_dst_bytes": False,
                    "op_seed_uint64_decimal": str(op_seed),
                    "expected_checksum": {
                        "status": "available",
                        "reason_code": "valid",
                        "expected_worker_checksums": exp_workers,
                        "expected_run_checksum": exp_run,
                    },
                }
            )
            return new_ref, exp_workers, exp_run

        # probeA: auto-cal attempts[-2].plan_ref != frozen_ref (even when frozen_measurement_warmup & measurements match frozen_ref)
        probe_a = self._make_auto_calibrated_doc()
        new_ref, exp_workers, exp_run = inject_seq_fwd_read_plan(probe_a, passes=1)
        probe_a["resolved_plan"]["frozen_plan_refs"]["sequential_forward"]["read"] = new_ref
        atts_a = probe_a["calibration"]["attempts"]["sequential_forward"]["read"]
        atts_a[-1]["plan_ref"] = new_ref
        atts_a[-1]["execution"]["checksum"]["actual_run_checksum"] = exp_run
        for r in probe_a["measurements"]:
            if r["pattern"] == "sequential_forward" and r["operation"] == "read":
                r["plan_ref"] = new_ref
                r["completed_passes"] = 1
                exact_b = int(probe_a["resolved_plan"]["operation_plans"][new_ref]["exact_payload_bytes"])
                r["completed_exact_payload_bytes"] = str(exact_b)
                r["bandwidth_gb_s"] = exact_b / r["execution"]["timing"]["elapsed_seconds"] / 1e9
                r["checksum"]["actual_worker_checksums"] = exp_workers
                r["checksum"]["actual_run_checksum"] = exp_run
        self.assert_reason(probe_a, "frozen-plan-reference", unsupported=False)

        # probeH: auto-cal calibration_shape_warmup (attempts[0]) vs pilot (attempts[1]) plan_ref mismatch
        probe_h = self._make_auto_calibrated_doc()
        alt_ref, _, alt_run = inject_seq_fwd_read_plan(probe_h, passes=1)
        att0 = probe_h["calibration"]["attempts"]["sequential_forward"]["read"][0]
        att0["plan_ref"] = alt_ref
        att0["execution"]["checksum"]["actual_run_checksum"] = alt_run
        self.assert_reason(probe_h, "frozen-plan-reference", unsupported=False)

        # Auto-cal > 6 attempts rejected
        too_many_cal = self._make_auto_calibrated_doc()
        atts_many = too_many_cal["calibration"]["attempts"]["sequential_forward"]["read"]
        for _ in range(4):
            corr = copy.deepcopy(atts_many[1])
            corr["purpose"] = "correction"
            atts_many.insert(-1, corr)
        for i, a in enumerate(atts_many):
            a["attempt_index"] = i
        self.assert_reason(too_many_cal, "calibration-invalid", unsupported=False)

        # single_unit_confirmation with passes != 1 rejected
        bad_single_unit = self._make_auto_calibrated_doc()
        atts_su = bad_single_unit["calibration"]["attempts"]["sequential_forward"]["read"]
        su = copy.deepcopy(atts_su[1])
        su["purpose"] = "single_unit_confirmation"
        atts_su.insert(-1, su)
        for i, a in enumerate(atts_su):
            a["attempt_index"] = i
        self.assert_reason(bad_single_unit, "calibration-invalid", unsupported=False)

        # probeC1: plan_ref=None placeholder in interrupted calibration.attempts rejected
        probe_c1 = make_truncated("interrupted", COMPLETE_FIXTURE)
        att_c1 = probe_c1["calibration"]["attempts"]["sequential_forward"]["read"][0]
        att_c1["plan_ref"] = None
        att_c1["terminal"] = False
        att_c1["valid"] = False
        att_c1["reason_code"] = "not-evaluated"
        att_c1["execution"]["status"] = "unavailable"
        att_c1["execution"]["valid"] = None
        att_c1["execution"]["elapsed_seconds"] = None
        self.assert_reason(probe_c1, "calibration-placeholder", unsupported=False)

        # Not-attempted row with non-null diagnostic_elapsed_seconds or cpu_raw rejected
        not_att_diag = make_truncated("interrupted", COMPLETE_FIXTURE)
        not_att_diag["measurements"][-1]["execution"]["timing"]["diagnostic_elapsed_seconds"] = 0.01
        self.assert_reason(not_att_diag, "metric-null", unsupported=False)

        not_att_raw = make_truncated("interrupted", COMPLETE_FIXTURE)
        not_att_raw["measurements"][-1]["execution"]["timing"]["cpu_raw"] = copy.deepcopy(
            not_att_raw["measurements"][0]["execution"]["timing"]["cpu_raw"]
        )
        self.assert_reason(not_att_raw, "metric-null", unsupported=False)

        # probeG1 & probeG2: memory_budget.valid vs reason_code mismatch on preflight work-cap-exceeded
        work_cap_base = load_fixture(SKIPPED_2MB_FIXTURE)
        work_cap_base["configuration"]["buffer_size_mb"] = 1
        work_cap_base["configuration"]["iterations"] = 100000
        work_cap_base["resolved_plan"].update(
            valid=False,
            reason_code="work-cap-exceeded",
            plan_identity=(
                f"patterns-work-plan-v1|buffer_size_mb=1"
                f"|workers={work_cap_base['resolved_plan']['patterns']['sequential_forward']['effective_workers']}"
                f"|worker_source=user|iterations=100000|loop_count=3|non_cacheable=0"
            ),
            resources={},
            operation_plans=[],
        )
        for p in work_cap_base["resolved_plan"]["patterns"].values():
            p.update(enabled=False, status="unavailable", reason_code="work-cap-exceeded")
        for prefs in work_cap_base["resolved_plan"]["frozen_plan_refs"].values():
            for op in verifier.OPERATIONS:
                prefs[op] = None
        work_cap_base["calibration"]["attempts"] = {}
        work_cap_base["measurements"] = []
        for p_agg in work_cap_base["aggregates"]["patterns"].values():
            for a in p_agg.values():
                a.update(
                    status="unavailable",
                    reason_code="work-cap-exceeded",
                    accepted_measurement_ids=[],
                    value=None,
                    relative_to_forward_pct=None,
                    statistics=None,
                )
                a["quality"].update(repeatability_assessment_available=False, run_complete=False, cv_warning=False)
        work_cap_base.update(
            status="failed",
            reason_code="work-cap-exceeded",
            results_complete=False,
            run_accepted=False,
            operation_order_balance_complete=False,
            pattern_order_balance_complete=False,
            quality_warnings=[],
        )
        work_cap_base["counters"].update(
            attempted_loops=0,
            completed_loops=0,
            planned_measurements=0,
            attempted_measurements=0,
            terminal_measurements=0,
            measured_measurements=0,
            planned_passes="0",
            completed_passes="0",
            planned_exact_payload_bytes="0",
            completed_exact_payload_bytes="0",
        )
        for rec in work_cap_base["loop_records"]:
            rec.update(planned_order=[], realized_order=[], realized_order_count=0, measurement_indexes=[])
        work_cap_base["checkpoint_lifecycle"].update(snapshot_interval_loops=1)
        self.assertTrue(verifier.verify(copy.deepcopy(work_cap_base))["artifact_consistent"])

        probe_g1 = copy.deepcopy(work_cap_base)
        probe_g1["memory_budget"]["valid"] = False
        probe_g1["memory_budget"]["reason_code"] = "within-memory-budget"
        self.assert_reason(probe_g1, "memory-budget-invalid", unsupported=False)

        probe_g2 = copy.deepcopy(work_cap_base)
        probe_g2["memory_budget"]["valid"] = True
        probe_g2["memory_budget"]["reason_code"] = "memory-budget-exceeded"
        self.assert_reason(probe_g2, "memory-budget-invalid", unsupported=False)

        probe_g3 = copy.deepcopy(work_cap_base)
        probe_g3["memory_budget"]["valid"] = False
        probe_g3["memory_budget"]["reason_code"] = "memory-budget-exceeded"
        self.assert_reason(probe_g3, "memory-budget-arithmetic", unsupported=False)


if __name__ == "__main__":
    unittest.main()

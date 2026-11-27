"""Independent numeric goldens, O(1) vs brute-force differential checks,
schema-v1 fixture verification, sweep envelope checks, and semantic mutations
for verify_cpu_bandwidth_result.py.
"""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "script-examples"))

import cpu_verify_common as common
import verify_cpu_bandwidth_result as verifier

FIXTURE_PATH = ROOT / "tests/fixtures/cpu-bandwidth-schema-v1-complete-current.json"


def load_fixture():
    return json.loads(FIXTURE_PATH.read_text())


def make_truncated(status):
    """Build a consistent partial/interrupted/failed artifact from the complete fixture."""
    doc = load_fixture()
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
        ex["valid"] = None
        ex["write_pattern_uint64_decimal"] = None
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
        row["checksum"] = {
            "status": "not_evaluated" if row["operation"] == "read" else "not_applicable",
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
        target_order_balance_complete=False,
    )
    rows = doc["measurements"]
    per_loop = 6
    for loop_idx, rec in enumerate(doc["loop_records"]):
        slice_rows = rows[loop_idx * per_loop : (loop_idx + 1) * per_loop]
        realized = [
            {"target": r["target"], "operation": r["operation"]}
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
    for tname, t_agg in doc["aggregates"]["targets"].items():
        for op, a in t_agg.items():
            ids = [
                r["measurement_id"]
                for r in rows
                if r["target"] == tname and r["operation"] == op and r["status"] == "measured"
            ]
            vals = [rows[idx]["bandwidth_gb_s"] for idx in ids]
            st = common.compute_statistics(vals)
            a["accepted_measurement_ids"] = ids
            a["value"] = st["median"] if st is not None else None
            a["statistics"] = st
            a["quality"]["repeatability_assessment_available"] = len(ids) >= 2
            a["quality"]["run_complete"] = len(ids) == doc["configuration"]["loop_count"]
            a["quality"]["cv_warning"] = False
    return doc


def brute_force_pass_xor(target_src_seed, offset_bytes, span_bytes):
    """Explicit byte-array reference matching ARM64 memory_read*_loop_asm."""
    if span_bytes == 0:
        return 0
    total_end = offset_bytes + span_bytes
    total_words = (total_end + 7) // 8
    table = [common.splitmix64_mix(target_src_seed ^ k) for k in range(256)]
    buf = bytearray(total_words * 8)
    for wi in range(total_words):
        w = table[wi & 255] ^ (
            wi ^ common.rotl64(wi, 19) ^ common.rotl64(wi, 41)
        )
        buf[wi * 8 : (wi + 1) * 8] = w.to_bytes(8, "little")

    vec_bytes = span_bytes - (span_bytes % 32)
    vec_xor = 0
    for pos in range(offset_bytes, offset_bytes + vec_bytes, 8):
        vec_xor ^= int.from_bytes(buf[pos : pos + 8], "little")
    tail_xor = 0
    for pos in range(offset_bytes + vec_bytes, offset_bytes + span_bytes):
        tail_xor ^= buf[pos]
    return (vec_xor ^ tail_xor) & common.U64


class CpuBandwidthVerifierTests(unittest.TestCase):
    def assert_reason(self, doc, reason, unsupported=None, **kwargs):
        with self.assertRaises(common.Rejected) as caught:
            verifier.verify(doc, **kwargs)
        self.assertEqual(caught.exception.reason, reason)
        if unsupported is not None:
            self.assertEqual(caught.exception.unsupported, unsupported)

    def test_valid_fixture_and_cli_raw_timing(self):
        doc = load_fixture()
        verdict = verifier.verify(doc, require_raw=True)
        self.assertTrue(verdict["artifact_consistent"])
        self.assertTrue(verdict["run_accepted"])
        self.assertEqual(verdict["checks"]["checksum"], "independent-oracle")
        self.assertEqual(verdict["checks"]["timing"], "cpu-raw-ticks")
        self.assertEqual(verdict["checks"]["build"], "manifest-only")

        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "script-examples/verify_cpu_bandwidth_result.py"),
                str(FIXTURE_PATH),
                "--require-raw-timing",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        parsed = json.loads(proc.stdout)
        self.assertTrue(parsed["run_accepted"])

    def test_o1_pass_xor_matches_brute_force_including_tails(self):
        seed = common.splitmix64_mix(424242 ^ verifier.TARGET_DOMAINS["main_memory"])
        table = verifier.build_bw_table(seed)
        spans = (0, 1, 7, 8, 9, 15, 16, 31, 32, 33, 63, 64, 65, 255, 256, 257, 511, 512, 513, 1023, 2048 + 17)
        for offset in (0, 64, 128, 256):
            for span in spans:
                with self.subTest(offset=offset, span=span):
                    expected = brute_force_pass_xor(seed, offset, span)
                    actual = verifier.bw_pass_xor(seed, offset, span, table)
                    self.assertEqual(actual, expected)

    def test_closed_form_read_checksum_matches_iterative_passes(self):
        target_seed = common.splitmix64_mix(99 ^ verifier.TARGET_DOMAINS["custom"])
        op_seed = 0x123456789ABCDEF0
        worker_spans = [(0, 4096), (4096, 4096 + 19)]
        table = verifier.build_bw_table(target_seed)
        for passes in (1, 2, 3, 10, 100, 10000):
            with self.subTest(passes=passes):
                workers, run_ck = verifier.expected_read_checksum(
                    target_seed, op_seed, passes, worker_spans, table_cache=table
                )
                tuples = []
                for w, (off, sp) in enumerate(worker_spans):
                    px = brute_force_pass_xor(target_seed, off, sp)
                    sa = common.splitmix64_mix(op_seed ^ (0xA100000000000000 | w))
                    sb = common.splitmix64_mix(op_seed ^ (0xB200000000000000 | w))
                    for p in range(1, passes + 1):
                        sa = (sa + px + common.SPLITMIX_A * p) & common.U64
                        sb = (sb + ((px ^ common.SPLITMIX_B) * p) + common.SPLITMIX_C) & common.U64
                    self.assertEqual(workers[w]["state_a_uint64_decimal"], str(sa))
                    self.assertEqual(workers[w]["state_b_uint64_decimal"], str(sb))
                    tuples.append((sa, sb, passes * sp, passes))
                self.assertEqual(run_ck, verifier.fold_bw_workers(tuples))

    def test_compute_statistics_edge_cases(self):
        self.assertIsNone(common.compute_statistics([]))
        one = common.compute_statistics([5.0])
        self.assertEqual(one["stddev"], 0.0)
        self.assertEqual(one["coefficient_of_variation_pct"], 0.0)
        zeros = common.compute_statistics([0.0, 0.0])
        self.assertIsNone(zeros["coefficient_of_variation_pct"])
        neg = common.compute_statistics([-4.0, -2.0])
        self.assertGreater(neg["coefficient_of_variation_pct"], 0.0)

    def test_lifecycle_states_consistent_but_not_accepted(self):
        for status in ("partial", "interrupted", "failed"):
            with self.subTest(status=status):
                doc = make_truncated(status)
                verdict = verifier.verify(doc)
                self.assertTrue(verdict["artifact_consistent"])
                self.assertFalse(verdict["run_accepted"])

        unsupported_doc = make_truncated("partial")
        for row in unsupported_doc["measurements"]:
            row["attempted"] = False
            row["status"] = "not_run"
            row["reason_code"] = "unsupported"
            row["bandwidth_gb_s"] = None
            row["completed_passes"] = 0
            row["completed_exact_payload_bytes"] = "0"
            row["execution"]["valid"] = None
            row["execution"]["write_pattern_uint64_decimal"] = None
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
            row["checksum"] = {
                "status": "not_evaluated" if row["operation"] == "read" else "not_applicable",
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
        for t_agg in unsupported_doc["aggregates"]["targets"].values():
            for a in t_agg.values():
                a["accepted_measurement_ids"] = []
                a["value"] = None
                a["statistics"] = None
                a["quality"]["repeatability_assessment_available"] = False
                a["quality"]["run_complete"] = False
        verdict = verifier.verify(unsupported_doc)
        self.assertTrue(verdict["artifact_consistent"])
        self.assertFalse(verdict["run_accepted"])

        # Preflight !work_plan.valid failed artifact (empty targets/resources) is consistent & not accepted.
        preflight_fail = load_fixture()
        preflight_fail["resolved_plan"].update(
            valid=False,
            reason_code="target-bytes-overflow",
            targets={},
            resources={},
            operation_plans=[],
            frozen_plan_refs={},
        )
        preflight_fail["memory_budget"]["valid"] = False
        preflight_fail["memory_budget"]["reason_code"] = "memory-budget-overflow"
        preflight_fail["calibration"]["attempts"] = {}
        preflight_fail["measurements"] = []
        preflight_fail["aggregates"]["targets"] = {}
        preflight_fail["status"] = "failed"
        preflight_fail["reason_code"] = "target-bytes-overflow"
        preflight_fail["results_complete"] = False
        preflight_fail["run_accepted"] = False
        preflight_fail["operation_order_balance_complete"] = False
        preflight_fail["target_order_balance_complete"] = False
        preflight_fail["counters"].update(
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
        for rec in preflight_fail["loop_records"]:
            rec["planned_order"] = []
            rec["realized_order"] = []
            rec["realized_order_count"] = 0
            rec["measurement_indexes"] = []
        pf_verdict = verifier.verify(preflight_fail)
        self.assertTrue(pf_verdict["artifact_consistent"])
        self.assertFalse(pf_verdict["run_accepted"])

    def test_execution_prefix_and_missing_raw_timing(self):
        doc = make_truncated("interrupted")
        doc["measurements"][4] = load_fixture()["measurements"][4]
        self.assert_reason(doc, "invalid-execution-prefix")

        no_raw = load_fixture()
        for row in no_raw["measurements"]:
            row["execution"]["timing"].pop("cpu_raw")
        for t_cal in no_raw["calibration"]["attempts"].values():
            for op_attempts in t_cal.values():
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
        # Find indices for read, write, copy in loop 0
        base = load_fixture()
        read_idx = next(r["measurement_id"] for r in base["measurements"] if r["operation"] == "read")
        write_idx = next(r["measurement_id"] for r in base["measurements"] if r["operation"] == "write")
        copy_idx = next(r["measurement_id"] for r in base["measurements"] if r["operation"] == "copy")

        def mutate_write_dup(d):
            w_rows = [r for r in d["measurements"] if r["operation"] == "write"]
            w_rows[1]["execution"].pop("attempt_seed_uint64_decimal", None)
            w_rows[1]["execution"]["write_pattern_uint64_decimal"] = w_rows[0]["execution"]["write_pattern_uint64_decimal"]

        def mutate_write_even(d):
            d["measurements"][write_idx]["execution"].pop("attempt_seed_uint64_decimal", None)
            d["measurements"][write_idx]["execution"]["write_pattern_uint64_decimal"] = "2"

        def mutate_actual_fold(d):
            r = d["measurements"][read_idx]
            r["checksum"]["actual_worker_checksums"][0]["state_a_uint64_decimal"] = "1"
            r["checksum"]["actual_run_checksum"]["state_a_uint64_decimal"] = "2"
            r["checksum"]["checksum_valid"] = False
            r["checksum"]["status"] = "invalid"

        def mutate_worker_alignment(d):
            spans = d["resolved_plan"]["targets"]["main_memory"]["worker_spans"]
            spans[0]["span_bytes"] = "524289"
            spans[1].update(offset_bytes="524289", span_bytes="524287")

        cases = [
            ("unsupported-schema", lambda d: d.update(schema_version=2), True),
            ("unsupported-mode", lambda d: d.update(mode="llm_memory"), True),
            ("unsupported-methodology", lambda d: d.update(methodology_version="bad"), True),
            ("component-mismatch", lambda d: d["resolved_plan"]["component_identities"].update(version="bad"), False),
            ("seed-mismatch", lambda d: d["seeds"]["buffer_domain_seeds"].update(main_memory="1"), False),
            ("target-geometry-mismatch", lambda d: d["resolved_plan"]["targets"]["main_memory"].update(bytes="1024"), False),
            ("worker-alignment", mutate_worker_alignment, False),
            ("plan-reference", lambda d: d["measurements"][0].update(plan_ref=999), False),
            ("plan-bytes-mismatch", lambda d: d["resolved_plan"]["operation_plans"][0].update(exact_payload_bytes="1"), False),
            (
                "expected-run-checksum-mismatch",
                lambda d: d["resolved_plan"]["operation_plans"][0]["expected_checksum"]["expected_run_checksum"].update(state_a_uint64_decimal="0"),
                False,
            ),
            (
                "expected-worker-checksum-mismatch",
                lambda d: d["resolved_plan"]["operation_plans"][0]["expected_checksum"]["expected_worker_checksums"][0].update(state_a_uint64_decimal="0"),
                False,
            ),
            (
                "checksum-valid-mismatch",
                lambda d: d["measurements"][read_idx]["checksum"].update(checksum_valid=False),
                False,
            ),
            ("actual-run-fold-mismatch", mutate_actual_fold, False),
            ("write-pattern-invalid", mutate_write_dup, False),
            ("write-pattern-invalid", mutate_write_even, False),
            (
                "required-validation-failed",
                lambda d: d["measurements"][read_idx]["execution"]["validation"]["checks"][1].update(valid=False),
                False,
            ),
            (
                "required-validation-failed",
                lambda d: d["measurements"][write_idx]["execution"]["validation"]["checks"][2].update(valid=False),
                False,
            ),
            (
                "required-validation-failed",
                lambda d: d["measurements"][copy_idx]["execution"]["validation"]["checks"][3].update(valid=False),
                False,
            ),
            (
                "validation-applicability",
                lambda d: d["measurements"][read_idx]["execution"]["validation"]["checks"][4].update(applicable=False, evaluated=None, valid=None),
                False,
            ),
            (
                "tick-delta-mismatch",
                lambda d: d["measurements"][0]["execution"]["timing"]["cpu_raw"].update(stop_ticks="1"),
                False,
            ),
            (
                "elapsed-mismatch",
                lambda d: d["measurements"][0]["execution"]["timing"]["cpu_raw"].update(timebase_numer=126),
                False,
            ),
            ("rate-mismatch", lambda d: d["measurements"][0].update(bandwidth_gb_s=1.0), False),
            (
                "aggregate-headline-mismatch",
                lambda d: d["aggregates"]["targets"]["main_memory"]["read"].update(value=1.0),
                False,
            ),
            (
                "aggregate-statistic-mismatch",
                lambda d: d["aggregates"]["targets"]["main_memory"]["read"]["statistics"].update(average=1.0),
                False,
            ),
            ("counter-mismatch", lambda d: d["counters"].update(completed_passes="0"), False),
            ("checkpoint-policy", lambda d: d["checkpoint_lifecycle"].update(snapshot_interval_loops=99), False),
            ("checkpoint-current-persistence", lambda d: d["checkpoint_lifecycle"].update(current_persistence_success=True), False),
            ("run-acceptance-mismatch", lambda d: d.update(run_accepted=False), False),
            ("integer-encoding", lambda d: d["measurements"][0].update(plan_ref=True), False),
            ("integer-encoding", lambda d: d["seeds"].update(base_seed_uint64_decimal="01"), False),
            ("integer-encoding", lambda d: d["seeds"].update(base_seed_uint64_decimal="-1"), False),
            ("run-status-mismatch", lambda d: d.update(interruption_requested=True), False),
            ("resource-evidence-invalid", lambda d: d["resolved_plan"]["resources"]["main_memory"]["src"].update(read_only_protected=False, mprotect_errno=13), False),
            ("resource-evidence-invalid", lambda d: d["resolved_plan"]["resources"]["main_memory"]["src"].update(requested_bytes="0"), False),
            ("resource-evidence-invalid", lambda d: d["resolved_plan"]["resources"]["main_memory"]["dst"].update(non_cacheable_requested=True), False),
            ("resource-evidence-invalid", lambda d: d["resolved_plan"]["resources"].pop("custom"), False),
            ("resolved-sources-invalid", lambda d: d["configuration"]["resolved_sources"].update(buffer_size="default", cache_size="default", workers="auto", iterations="auto", seed="generated"), False),
            ("resolved-sources-invalid", lambda d: d["configuration"]["resolved_sources"].update(buffer_size="bogus"), False),
            ("quality-warnings-invalid", lambda d: d.update(quality_warnings=["completely-fabricated-warning"]), False),
            ("quality-warnings-invalid", lambda d: d.update(quality_warnings=["environment-not-nominal"]), False),
            ("interpretation-invalid", lambda d: d.update(interpretation="corrupted interpretation"), False),
            ("interpretation-invalid", lambda d: d["interpretation"].update(non_cacheable_requested=True), False),
            ("plan-invalid", lambda d: d["resolved_plan"].update(valid=False), False),
        ]
        for reason, mutate, unsup in cases:
            with self.subTest(reason=reason):
                doc = load_fixture()
                mutate(doc)
                self.assert_reason(doc, reason, unsup)

    def test_strict_json_cli_limits_and_binary_binding(self):
        with tempfile.TemporaryDirectory(prefix="cpu-bw-verifier-") as directory:
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
                [sys.executable, str(ROOT / "script-examples/verify_cpu_bandwidth_result.py"), str(p)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 1)
            self.assertEqual(json.loads(proc.stdout)["reason_code"], "invalid-input")

            p.write_text(json.dumps(load_fixture()))
            with mock.patch.object(common, "MAX_INPUT", 100):
                with self.assertRaises(common.Rejected) as caught:
                    common.load_strict_json(p)
                self.assertEqual(caught.exception.reason, "unsupported-input-limit")
                self.assertTrue(caught.exception.unsupported)

            bin_path = d / "memory_benchmark"
            bin_path.write_bytes(b"known cpu bandwidth binary")
            doc = load_fixture()
            doc["build_manifest"]["binary_sha256"] = hashlib.sha256(
                b"known cpu bandwidth binary"
            ).hexdigest()
            self.assertEqual(verifier.verify(doc, binary=bin_path)["checks"]["build"], "binary-bound")
            doc["build_manifest"]["binary_sha256"] = "0" * 64
            self.assert_reason(doc, "binary-hash-mismatch", binary=bin_path)
            doc["build_manifest"]["binary_sha256"] = None
            doc["build_manifest"]["status"] = "partial"
            self.assert_reason(doc, "binary-hash-unavailable", binary=bin_path)

            for payload, expected in (
                (load_fixture(), 0),
                (make_truncated("failed"), 1),
                ({"schema_version": 99}, 2),
            ):
                p.write_text(json.dumps(payload))
                res = subprocess.run(
                    [sys.executable, str(ROOT / "script-examples/verify_cpu_bandwidth_result.py"), str(p)],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(res.returncode, expected, res.stdout + res.stderr)
                json.loads(res.stdout)

    @staticmethod
    def _strip_cli_flag(argv, flag):
        out = list(argv)
        if flag in out:
            idx = out.index(flag)
            del out[idx : idx + 2]
        return out

    def make_sweep(self, runs_count=2):
        doc0 = load_fixture()
        doc0["configuration"]["output_file"] = ""
        argv = self._strip_cli_flag(doc0["configuration"]["argv"], "-b")
        argv = self._strip_cli_flag(argv, "-t")
        argv.extend(["-S", f"buffer-size={','.join(['1'] * runs_count)}", "-S", "threads=2"])
        doc0["configuration"]["argv"] = argv
        doc0["configuration"]["resolved_sources"]["buffer_size"] = "sweep"
        doc0["configuration"]["resolved_sources"]["workers"] = "sweep"
        runs = []
        for idx in range(runs_count):
            runs.append(
                {
                    "index": idx,
                    "parameters": {"buffer-size": 1, "threads": 2},
                    "status": "complete",
                    "status_reason": None,
                    "result": copy.deepcopy(doc0),
                }
            )
        return {
            "configuration": {
                "mode": "sweep",
                "base_mode": "cpu_bandwidth",
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
            "timestamp": "2026-11-27T07:00:00Z",
            "version": "0.65.0",
            "runs": runs,
        }

    def test_sweep_verification_mutations_single_hash_and_budget(self):
        sweep = self.make_sweep(runs_count=2)
        sorted_sweep = json.loads(json.dumps(sweep, sort_keys=True))
        verdict = verifier.verify(sorted_sweep, require_raw=True)
        self.assertTrue(verdict["artifact_consistent"])
        self.assertTrue(verdict["run_accepted"])
        self.assertEqual(verdict["artifact_kind"], "cpu_bandwidth_sweep")

        with tempfile.NamedTemporaryFile() as bin_file:
            bin_file.write(b"sweep-binary")
            bin_file.flush()
            digest = hashlib.sha256(b"sweep-binary").hexdigest()
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
        ]
        for reason, mutate, unsup in sweep_mutations:
            with self.subTest(reason=reason):
                env = self.make_sweep(2)
                mutate(env)
                self.assert_reason(env, reason, unsup)

    def test_sweep_resolved_sources_provenance_tamper_rejected(self):
        def mutate_all_runs(env, fn):
            for r in env["runs"]:
                fn(r["result"]["configuration"])

        def swap_buffer_and_cache(cfg):
            cfg["argv"] = self._strip_cli_flag(cfg["argv"], "-k")
            cfg["resolved_sources"]["buffer_size"] = "default"
            cfg["resolved_sources"]["cache_size"] = "sweep"

        def unswept_cache_marked_sweep(cfg):
            cfg["argv"] = self._strip_cli_flag(cfg["argv"], "-k")
            cfg["resolved_sources"]["cache_size"] = "sweep"

        def swept_buffer_marked_default(cfg):
            cfg["resolved_sources"]["buffer_size"] = "default"

        def swept_workers_marked_auto(cfg):
            cfg["worker_source"] = "auto"
            cfg["resolved_sources"]["workers"] = "auto"

        def swept_buffer_marked_explicit_with_flag(cfg):
            cfg["argv"] = list(cfg["argv"]) + ["-b", "1"]
            cfg["resolved_sources"]["buffer_size"] = "explicit"

        cases = [
            ("swap_sweep_default_axes", swap_buffer_and_cache),
            ("unswept_key_marked_sweep", unswept_cache_marked_sweep),
            ("swept_buffer_marked_default", swept_buffer_marked_default),
            ("swept_workers_marked_auto", swept_workers_marked_auto),
            ("swept_buffer_marked_explicit_with_flag", swept_buffer_marked_explicit_with_flag),
        ]
        with tempfile.TemporaryDirectory(prefix="cpu-bw-sweep-rs-") as directory:
            p = Path(directory) / "tamper.json"
            for name, cfg_mut in cases:
                with self.subTest(case=name):
                    env = self.make_sweep(2)
                    mutate_all_runs(env, cfg_mut)
                    self.assert_reason(env, "resolved-sources-invalid", unsupported=False)
                    p.write_text(json.dumps(env))
                    res = subprocess.run(
                        [sys.executable, str(ROOT / "script-examples/verify_cpu_bandwidth_result.py"), str(p)],
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(res.returncode, 1, res.stdout + res.stderr)
                    verdict = json.loads(res.stdout)
                    self.assertEqual(verdict["reason_code"], "resolved-sources-invalid")
                    self.assertFalse(verdict["run_accepted"])

    def test_null_and_invalid_nested_containers_raise_typed_rejections(self):
        cases = [
            ("measurement-invalid", lambda d: d.update(measurements=None)),
            ("measurement-invalid", lambda d: d.update(measurements=[123])),
            (
                "validation-invalid",
                lambda d: d["calibration"]["attempts"]["main_memory"]["read"][0]["execution"].update(validation=None),
            ),
            (
                "timing-invalid",
                lambda d: d["calibration"]["attempts"]["main_memory"]["read"][0]["execution"].update(timing=None),
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
                doc = load_fixture()
                mutate(doc)
                self.assert_reason(doc, reason, unsupported=False)

        with tempfile.TemporaryDirectory(prefix="cpu-bw-null-containers-") as directory:
            p = Path(directory) / "bad.json"
            for expected_reason, mutate in cases[:3]:
                doc = load_fixture()
                mutate(doc)
                p.write_text(json.dumps(doc))
                res = subprocess.run(
                    [sys.executable, str(ROOT / "script-examples/verify_cpu_bandwidth_result.py"), str(p)],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(res.returncode, 1, res.stdout + res.stderr)
                verdict = json.loads(res.stdout)
                self.assertEqual(verdict["reason_code"], expected_reason)
                self.assertFalse(verdict["run_accepted"])

    @staticmethod
    def _make_auto_calibrated_doc(elapsed_by_target_op=None):
        doc = load_fixture()
        cfg = doc["configuration"]
        argv = list(cfg["argv"])
        idx = argv.index("-i")
        del argv[idx : idx + 2]
        cfg["argv"] = argv
        cfg["iterations"] = None
        cfg["work_policy"] = "automatic_calibration"
        cfg["resolved_sources"]["iterations"] = "auto"

        plans = doc["resolved_plan"]["operation_plans"]
        for p in plans:
            p["explicit_iterations"] = False

        elapsed_by_target_op = elapsed_by_target_op or {}
        for row in doc["measurements"]:
            if row["status"] != "measured":
                continue
            target_sec = elapsed_by_target_op.get((row["target"], row["operation"]), 0.150)
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
        for tname, t_agg in doc["aggregates"]["targets"].items():
            for op, a in t_agg.items():
                vals = [
                    r["bandwidth_gb_s"]
                    for r in rows
                    if r["target"] == tname and r["operation"] == op and r["status"] == "measured"
                ]
                st = common.compute_statistics(vals)
                a["value"] = st["median"]
                a["statistics"] = st
                a["quality"]["cv_warning"] = False
        return doc

    def test_resolved_sources_two_way_quality_duration_window_and_madvise_errno(self):
        # (2a) Two-way check_resolved_sources
        no_sweep_flag = load_fixture()
        no_sweep_flag["configuration"]["resolved_sources"]["buffer_size"] = "sweep"
        self.assert_reason(no_sweep_flag, "resolved-sources-invalid")

        omitted_explicit = load_fixture()
        argv = list(omitted_explicit["configuration"]["argv"])
        idx = argv.index("-b")
        del argv[idx : idx + 2]
        omitted_explicit["configuration"]["argv"] = argv
        omitted_explicit["configuration"]["resolved_sources"]["buffer_size"] = "default"
        self.assertTrue(verifier.verify(omitted_explicit, require_raw=True)["run_accepted"])
        omitted_explicit["configuration"]["resolved_sources"]["buffer_size"] = "explicit"
        self.assert_reason(omitted_explicit, "resolved-sources-invalid")

        swept_in_argv = load_fixture()
        argv = list(swept_in_argv["configuration"]["argv"])
        idx = argv.index("-b")
        del argv[idx : idx + 2]
        argv.extend(["-S", "buffer-size=1"])
        swept_in_argv["configuration"]["argv"] = argv
        swept_in_argv["configuration"]["resolved_sources"]["buffer_size"] = "sweep"
        self.assertTrue(verifier.verify(swept_in_argv, require_raw=True)["run_accepted"])

        # (2b) Cross-checked auto-calibration duration warnings ([0.100, 0.250])
        within_doc = self._make_auto_calibrated_doc(
            {("main_memory", "read"): 0.100, ("custom", "write"): 0.250}
        )
        within_doc["quality_warnings"] = []
        self.assertTrue(verifier.verify(within_doc, require_raw=True)["run_accepted"])

        fabricated_warn = copy.deepcopy(within_doc)
        fabricated_warn["quality_warnings"] = ["main_memory-read-duration-above-target-window"]
        self.assert_reason(fabricated_warn, "quality-warnings-invalid")

        out_of_window = self._make_auto_calibrated_doc(
            {("main_memory", "read"): 0.050, ("custom", "write"): 0.300}
        )
        out_of_window["quality_warnings"] = []
        self.assert_reason(out_of_window, "quality-warnings-invalid")

        wrong_dir = copy.deepcopy(out_of_window)
        wrong_dir["quality_warnings"] = [
            "main_memory-read-duration-above-target-window",
            "custom-write-duration-above-target-window",
        ]
        self.assert_reason(wrong_dir, "quality-warnings-invalid")

        exact_warn = copy.deepcopy(out_of_window)
        exact_warn["quality_warnings"] = [
            "main_memory-read-duration-below-target-window",
            "custom-write-duration-above-target-window",
        ]
        self.assertTrue(verifier.verify(exact_warn, require_raw=True)["run_accepted"])

        # (2c) check_buffer_evidence madvise_applied=False requires madvise_errno != 0
        bad_madv = load_fixture()
        bad_madv["resolved_plan"]["resources"]["main_memory"]["src"]["madvise_applied"] = False
        self.assert_reason(bad_madv, "resource-evidence-invalid")

        valid_madv_fail = load_fixture()
        valid_madv_fail["configuration"]["non_cacheable"] = True
        valid_madv_fail["interpretation"]["non_cacheable_requested"] = True
        for t_res in valid_madv_fail["resolved_plan"]["resources"].values():
            for buf in (t_res["src"], t_res["dst"]):
                buf["non_cacheable_requested"] = True
                buf["madvise_advice"] = 1
        valid_madv_fail["resolved_plan"]["resources"]["main_memory"]["src"].update(
            madvise_applied=False, madvise_errno=22
        )
        valid_madv_fail["quality_warnings"] = ["madvise-random-not-applied"]
        self.assertTrue(verifier.verify(valid_madv_fail, require_raw=True)["run_accepted"])


if __name__ == "__main__":
    unittest.main()

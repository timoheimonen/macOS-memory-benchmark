"""Independent numeric goldens, current producer artifacts, and semantic mutations."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "script-examples"))
import verify_llm_result as verifier
import llm_verify_oracles as oracle

FIXTURES = ROOT / "tests/fixtures/llm-schema-v2"


def fixture(name="cpu-decode-contiguous"):
    return json.loads((FIXTURES / (name + ".json")).read_text())


def truncated(status):
    """Author the status contract explicitly around a captured two-row prefix."""
    doc = fixture()
    measured = 2
    for index, row in enumerate(doc["measurements"]):
        if index < measured:
            continue
        row["attempted"] = False
        row["status"] = (
            "not_run" if status == "partial" else "interrupted" if status == "interrupted" else "failed"
        )
        row["reason_code"] = (
            "not-run"
            if status == "partial"
            else "interruption-before-task" if status == "interrupted" else "not-run-after-runtime-failure"
        )
        for key in verifier.METRICS + ("elapsed_seconds",):
            row[key] = None
        row["completed_work_units"] = 0
        for key in (
            "effective_model_payload_bytes",
            "layout_metadata_lookup_count",
            "layout_metadata_read_bytes",
            "task_accounted_bytes",
        ):
            row["completed_" + key] = "0"
        row["completion_derivation"] = None
        ex = row["execution"]
        ex.update(status="not_run", valid=None)
        ex["timing"] = dict(evaluated=False, valid=None, diagnostic_elapsed_seconds=None, cpu_raw=None)
        for check in ex["validation"]["checks"]:
            check["evaluated"] = False if check["applicable"] else None
            check["valid"] = None
        row["checksum"].update(
            status="not_evaluated",
            checksum_valid=None,
            actual_worker_checksums=None,
            actual_run_checksum=None,
        )
    doc.update(
        status=status,
        reason_code=(
            "partial-results"
            if status == "partial"
            else "interruption-requested" if status == "interrupted" else "runtime-failure"
        ),
        results_complete=False,
        run_accepted=False,
        scenario_order_balance_complete=False,
        interruption_requested=status == "interrupted",
    )
    for i, loop in enumerate(doc["loop_records"]):
        loop["realized_order"] = loop["planned_order"][:2] if i == 0 else []
        loop["realized_order_count"] = len(loop["realized_order"])
    refresh(doc)
    return doc


def refresh(doc):
    """Fixture authoring: rebuild only population/counter metadata after a status edit."""
    rows = doc["measurements"]
    c = doc["counters"]
    c.update(
        attempted_loops=len({r["loop_index"] for r in rows if r["attempted"]}),
        completed_loops=sum(
            all(r["status"] == "measured" for r in rows[i : i + 3]) for i in range(0, len(rows), 3)
        ),
        attempted_measurements=sum(r["attempted"] for r in rows),
        terminal_measurements=sum(r["status"] != "not_run" for r in rows),
        measured_measurements=sum(r["status"] == "measured" for r in rows),
    )
    for key in (
        "work_units",
        "effective_model_payload_bytes",
        "layout_metadata_lookup_count",
        "layout_metadata_read_bytes",
        "task_accounted_bytes",
    ):
        c["completed_" + key] = str(sum(int(r["completed_" + key]) for r in rows))
    for scenario, a in doc["aggregates"]["scenarios"].items():
        ids = [i for i, r in enumerate(rows) if r["status"] == "measured" and r["scenario"] == scenario]
        a.update(
            accepted_measurement_ids=ids,
            status="partial" if ids else "unavailable",
            observed_cv_classification="insufficient-samples",
        )
        for metric in verifier.METRICS:
            block = a[metric]
            values = [rows[i][metric] for i in ids]
            # These fixtures deliberately have at most one member per scenario.
            assert len(values) <= 1
            v = values[0] if values else None
            block.update(
                sample_count=len(values),
                headline=v,
                headline_semantics="single_measurement" if values else "unavailable",
            )
            block["statistics"] = (
                None
                if not values
                else dict(
                    sample_count=1,
                    average=v,
                    min=v,
                    max=v,
                    median=v,
                    p90=v,
                    p95=v,
                    p99=v,
                    stddev=0.0,
                    coefficient_of_variation_pct=0.0,
                    median_absolute_deviation=0.0,
                )
            )


class VerifierTests(unittest.TestCase):
    def assert_reason(self, doc, reason):
        with self.assertRaises(verifier.Rejected) as caught:
            verifier.verify(doc)
        self.assertEqual(caught.exception.reason, reason)

    def test_all_eight_real_profiles(self):
        for backend in ("cpu", "metal"):
            for phase in ("decode", "prefill"):
                for layout in ("contiguous", "paged"):
                    with self.subTest(backend=backend, phase=phase, layout=layout):
                        self.assertTrue(
                            verifier.verify(fixture("-".join((backend, phase, layout))))["run_accepted"]
                        )

    def test_status_artifacts_are_consistent_but_not_accepted(self):
        for status in ("partial", "interrupted", "failed"):
            with self.subTest(status=status):
                verdict = verifier.verify(truncated(status))
                self.assertTrue(verdict["artifact_consistent"])
                self.assertFalse(verdict["run_accepted"])
        verdict = verifier.verify(fixture("unsupported"))
        self.assertTrue(verdict["artifact_consistent"])
        self.assertFalse(verdict["run_accepted"])

    def test_invalid_checksum_keeps_actual_evidence_out_of_population(self):
        doc = truncated("failed")
        row = doc["measurements"][1]
        row["status"] = "invalid"
        row["execution"]["valid"] = False
        row["execution"]["status"] = "invalid"
        row["checksum"]["actual_worker_checksums"][0]["k"]["state_a_uint64_decimal"] = "0"
        states = [
            [
                [
                    int(worker[pool][key])
                    for key in (
                        "state_a_uint64_decimal",
                        "state_b_uint64_decimal",
                        "exact_bytes_read",
                        "span_count_uint64_decimal",
                    )
                ]
                for pool in ("weight", "k", "v")
            ]
            for worker in row["checksum"]["actual_worker_checksums"]
        ]
        row["checksum"]["actual_run_checksum"] = oracle.fold(states)
        row["checksum"].update(status="invalid", checksum_valid=False)
        row["execution"]["timing"]["diagnostic_elapsed_seconds"] = row["elapsed_seconds"]
        for key in verifier.METRICS + ("elapsed_seconds",):
            row[key] = None
        row["completed_work_units"] = 0
        for key in (
            "effective_model_payload_bytes",
            "layout_metadata_lookup_count",
            "layout_metadata_read_bytes",
            "task_accounted_bytes",
        ):
            row["completed_" + key] = "0"
        refresh(doc)
        verdict = verifier.verify(doc)
        self.assertTrue(verdict["artifact_consistent"])
        self.assertFalse(verdict["run_accepted"])
        doc["aggregates"]["scenarios"]["kv_only"]["accepted_measurement_ids"] = [1]
        self.assert_reason(doc, "aggregate-population-mismatch")

    def test_mutation_reasons(self):
        cases = [
            ("plan-reference", lambda d: d["measurements"][0].update(plan_ref=999)),
            (
                "plan-identity-mismatch",
                lambda d: d["resolved_plan"]["scenario_plans"][0].update(
                    plan_identity=d["resolved_plan"]["scenario_plans"][0]["plan_identity"] + "|extra=1"
                ),
            ),
            (
                "model-identity-mismatch",
                lambda d: d["resolved_plan"].update(
                    plan_identity=d["resolved_plan"]["plan_identity"].replace(
                        "|base_seed=424242", "|base_seed=424243"
                    )
                ),
            ),
            ("seed-mismatch", lambda d: d["seeds"]["buffer_domain_seeds"].update(k_uint64_decimal="1")),
            (
                "configuration-geometry-mismatch",
                lambda d: d["resolved_plan"]["geometry"].update(layer_count=3),
            ),
            ("plan-bytes-mismatch", lambda d: d["resolved_plan"]["scenario_plans"][0].update(work_units=3)),
            (
                "plan-bytes-mismatch",
                lambda d: d["resolved_plan"]["scenario_plans"][0].update(effective_model_payload_bytes="1"),
            ),
            (
                "tick-delta-mismatch",
                lambda d: d["measurements"][0]["execution"]["timing"]["cpu_raw"].update(stop_ticks="1"),
            ),
            (
                "elapsed-mismatch",
                lambda d: d["measurements"][0]["execution"]["timing"]["cpu_raw"].update(timebase_numer=126),
            ),
            ("rate-mismatch", lambda d: d["measurements"][0].update(effective_model_payload_gb_s=1.0)),
            (
                "expected-run-checksum-mismatch",
                lambda d: d["resolved_plan"]["scenario_plans"][0]["expected_checksum"][
                    "expected_run_checksum"
                ].update(state_a_uint64_decimal="0"),
            ),
            (
                "checksum-valid-mismatch",
                lambda d: d["measurements"][0]["checksum"].update(checksum_valid=False),
            ),
            ("measurement-valid-mismatch", lambda d: d["measurements"][0]["execution"].update(valid=False)),
            ("measurement-id-mismatch", lambda d: d["measurements"][1].update(measurement_id=0)),
            ("measurement-count-mismatch", lambda d: d["measurements"].pop()),
            (
                "aggregate-population-mismatch",
                lambda d: d["aggregates"]["scenarios"]["weights_only"]["accepted_measurement_ids"].append(1),
            ),
            (
                "aggregate-statistic-mismatch",
                lambda d: d["aggregates"]["scenarios"]["weights_only"]["effective_model_payload_gb_s"][
                    "statistics"
                ].update(average=1.0),
            ),
            (
                "required-validation-failed",
                lambda d: d["measurements"][1]["execution"]["validation"]["checks"][1].update(valid=False),
            ),
            ("unsupported-schema", lambda d: d.update(schema_version=1)),
            ("unsupported-methodology", lambda d: d.update(methodology_version="unknown")),
            ("run-acceptance-mismatch", lambda d: d.update(run_accepted=False)),
            ("counter-mismatch", lambda d: d["counters"].update(completed_work_units="0")),
        ]
        for reason, mutate in cases:
            with self.subTest(reason=reason):
                doc = fixture()
                mutate(doc)
                self.assert_reason(doc, reason)
        doc = fixture("metal-prefill-paged")
        doc["resolved_plan"]["layout"]["permutation_sha256"] = "0" * 64
        self.assert_reason(doc, "permutation-digest-mismatch")
        doc = fixture("cpu-decode-paged")
        doc["measurements"][1]["completed_layout_metadata_lookup_count"] = "0"
        self.assert_reason(doc, "completion-mismatch")

    def test_interruption_prefix_rejects_resumption(self):
        doc = truncated("interrupted")
        doc["measurements"][3] = fixture()["measurements"][3]
        self.assert_reason(doc, "invalid-execution-prefix")

    def test_missing_raw_timing_is_explicitly_limited(self):
        doc = fixture()
        for row in doc["measurements"]:
            row["execution"]["timing"].pop("cpu_raw")
        verdict = verifier.verify(doc)
        self.assertTrue(verdict["artifact_consistent"])
        self.assertEqual(verdict["checks"]["timing"], "elapsed-only")

    def test_work_budget_never_means_success(self):
        budget = verifier.Budget()
        with self.assertRaises(verifier.Rejected) as caught:
            budget(verifier.MAX_WORK + 1)
        self.assertEqual(caught.exception.reason, "unsupported-work-limit")
        self.assertTrue(caught.exception.unsupported)

    def test_strict_json_and_cli_exit_contract(self):
        with tempfile.TemporaryDirectory(prefix="llm-verifier-", dir="/tmp") as directory:
            path = Path(directory) / "input.json"
            path.write_text('{"schema_version":2,"schema_version":2}')
            with self.assertRaises(verifier.Rejected) as caught:
                verifier.load(path)
            self.assertEqual(caught.exception.reason, "duplicate-json-key")
            for doc, expected in ((fixture(), 0), (truncated("failed"), 1), (dict(schema_version=1), 2)):
                path.write_text(json.dumps(doc))
                result = subprocess.run(
                    [sys.executable, str(ROOT / "script-examples/verify_llm_result.py"), str(path)],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                json.loads(result.stdout)

    def test_integer_encoding_and_binary_binding(self):
        doc = fixture()
        doc["measurements"][0]["plan_ref"] = True
        self.assert_reason(doc, "integer-encoding")
        with tempfile.NamedTemporaryFile(dir="/tmp") as binary:
            binary.write(b"known binary bytes")
            binary.flush()
            import hashlib

            doc = fixture()
            doc["build_manifest"]["binary_sha256"] = hashlib.sha256(b"known binary bytes").hexdigest()
            self.assertEqual(verifier.verify(doc, binary.name)["checks"]["build"], "binary-bound")
            doc["build_manifest"]["binary_sha256"] = "0" * 64
            with self.assertRaises(verifier.Rejected) as caught:
                verifier.verify(doc, binary.name)
            self.assertEqual(caught.exception.reason, "binary-hash-mismatch")


class ArithmeticGoldens(unittest.TestCase):
    def test_published_cpu_golden_and_wrap(self):
        # Independent published 19-byte vector 00..12; parity restarts in the span.
        state = [
            0x243F6A8885A308D3 ^ oracle.DOMAINS[0],
            (0x13198A2E03707344 + oracle.DOMAINS[0]) & oracle.U64,
            0,
            0,
        ]
        oracle.absorb(state, 0x0706050403141210, 0x0F0E0D0C0B0A0908, 19)
        self.assertEqual(state, [0x451AA0ABCC0E316F, 0x37A51B246A0ABBE7, 19, 1])
        self.assertEqual(
            (oracle.A + oracle.B + oracle.C + oracle.D + 0x4B4B4B4B4B4B4B4B) & oracle.U64, 0x149454E56105BC97
        )
        self.assertEqual(oracle.splitmix(0), 0xE220A8397B1DCDAF)

    def test_hand_derived_one_byte_goldens_for_all_eight_profiles(self):
        # W=K=V=one byte, seed=0, one layer/batch/operation, P=Q=block=1.
        # W low byte is 0x15 (CPU) / 0xb9 (Metal). CPU decode append K/V
        # are 0x97/0xa2; prefill adds phase-domain low byte 0x31 -> 0xc8/0xd3.
        # Paged lookup has logical=physical=0; its pair term is exactly one.
        cpu_runs = {
            (False, False): (8604485311877256335, 11711443974649127621),
            (False, True): (12314371692794693905, 18049950523347819890),
            (True, False): (4538172094021051140, 2453338796290460398),
            (True, True): (2839643602375233441, 6185486332617838998),
        }
        metal_runs = {
            (False, False): [(3495835427, 20476811), (1723508762, 3471944986), (2092076570, 539171610)],
            (False, True): [(3495838755, 4075535499), (1694068630, 572221926), (2431925142, 225265126)],
            (True, False): [(3496621859, 2136241035), (1200551007, 2254724667), (1569118815, 3616918587)],
            (True, True): [(3496625187, 1896332427), (3181621096, 3137674192), (3918759528, 2024859344)],
        }
        for prefill, paged in cpu_runs:
            model = dict(
                prefill=prefill,
                paged=paged,
                workers=1,
                layers=1,
                batch=1,
                record=1,
                tokens=1,
                q=1,
                tile_ends=[1],
                blocks=1 if paged else 0,
                block_tokens=1 if paged else 0,
                block_bytes=1 if paged else 0,
                weight=1,
                buffer_seeds=[0, 0, 0],
                scenario_seeds={"mixed": 0},
                ownership={"mixed": [[(0, 1)]]},
                table=[0],
            )
            with self.subTest(backend="cpu", prefill=prefill, paged=paged):
                workers, run = oracle.cpu_checksum(model, "mixed", 1, verifier.Budget())
                self.assertEqual(tuple(map(int, run.values())), cpu_runs[prefill, paged])
                for pool in ("weight", "k", "v"):
                    self.assertEqual(workers[0][pool]["exact_bytes_read"], "1")
                    self.assertEqual(workers[0][pool]["span_count_uint64_decimal"], "1")
            with self.subTest(backend="metal", prefill=prefill, paged=paged):
                _, run = oracle.metal_checksum(model, "mixed", 1, verifier.Budget())
                self.assertEqual(
                    [tuple(map(int, run[p].values())) for p in ("weight", "k", "v")],
                    metal_runs[prefill, paged],
                )

    def test_affine_closed_form_matches_independent_byte_enumeration(self):
        for bits in (32, 64):
            width = bits // 8
            mask = (1 << bits) - 1
            data = b"".join(
                ((mask - 3 + 0x13579 * (i + 1)) & mask).to_bytes(width, "little") for i in range(10)
            )
            for first in range(2 * width):
                for length in range(1, 3 * width):
                    for relative in (True, False):
                        expected = [0, 0]
                        if relative:
                            for i in range(0, length, width):
                                expected[(i // width) % 2] += int.from_bytes(
                                    data[first + i : first + min(length, i + width)], "little"
                                )
                        else:
                            for i in range(first, first + length):
                                expected[(i // width) % 2] += data[i] << (8 * (i % width))
                        self.assertEqual(
                            oracle.affine_slice(
                                mask - 3, 0x13579, first, length, bits, first if relative else None
                            ),
                            [v & mask for v in expected],
                        )


class BuildProvenanceTests(unittest.TestCase):
    def test_tarball_does_not_inherit_enclosing_checkout_and_header_is_stable(self):
        spec = importlib.util.spec_from_file_location(
            "build_provenance", ROOT / "build-support/generate_provenance.py"
        )
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        environment = {
            "PROVENANCE_" + key: "test" for key in ("CXX", "CXXFLAGS", "TEST_CXXFLAGS", "ASFLAGS", "LDFLAGS")
        }
        with tempfile.TemporaryDirectory() as directory:
            header = Path(directory) / "manifest.h"

            def command(args):
                if args[:2] == ["git", "rev-parse"]:
                    self.assertEqual(args, ["git", "rev-parse", "--show-toplevel"])
                    return str(ROOT.parent)
                return None

            with mock.patch.object(generator, "command", side_effect=command), mock.patch.dict(
                os.environ, environment, clear=True
            ), mock.patch.object(sys, "argv", ["generator", str(header)]):
                generator.main()
                manifest = json.loads(
                    json.loads(header.read_text().split("#define LLM_BUILD_MANIFEST_JSON ")[1])
                )
                self.assertIsNone(manifest["git_commit"])
                self.assertIsNone(manifest["git_dirty"])
                self.assertEqual(manifest["status"], "partial")
                before = header.stat().st_mtime_ns
                generator.main()
                self.assertEqual(header.stat().st_mtime_ns, before)

    def test_reader_enforces_structure_and_utf8_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text("[" * 66 + "0" + "]" * 66)
            with self.assertRaises(verifier.Rejected) as caught:
                verifier.load(path)
            self.assertEqual(caught.exception.reason, "unsupported-structure-limit")
            path.write_bytes("{}".encode("utf-16"))
            with self.assertRaises(UnicodeError):
                verifier.load(path)
            path.write_bytes(b" " * 101)
            with mock.patch.object(verifier, "MAX_INPUT", 100), self.assertRaises(
                verifier.Rejected
            ) as caught:
                verifier.load(path)
            self.assertEqual(caught.exception.reason, "unsupported-input-limit")


if __name__ == "__main__":
    unittest.main()

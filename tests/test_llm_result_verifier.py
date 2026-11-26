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

    def make_sweep_envelope(self, fixture_name="cpu-decode-contiguous", param_key="context-tokens"):
        doc = fixture(fixture_name)
        doc["configuration"]["output_file"] = ""
        config_key = verifier.SWEEP_PARAM_CONFIG_KEYS[param_key]
        value = doc["configuration"][config_key]
        return dict(
            configuration=dict(
                mode="sweep",
                base_mode="llm_memory",
                sweep_schema_version=1,
                run_count=1,
                sweep_max_runs=16,
                sweep_parameters={param_key: [value]},
            ),
            planned_runs=1,
            attempted_runs=1,
            completed_runs=1,
            status="complete",
            status_reason=None,
            conclusions_valid=True,
            execution_time_sec=0.125,
            timestamp="2026-01-01T00:00:00Z",
            version="0.64.1",
            runs=[
                dict(
                    index=0,
                    parameters={param_key: value},
                    status="complete",
                    status_reason=None,
                    result=doc,
                )
            ],
        )

    def make_two_run_sweep_envelope(self, fixture_name="cpu-decode-contiguous", param_key="context-tokens"):
        doc0 = fixture(fixture_name)
        doc0["configuration"]["output_file"] = ""
        doc1 = copy.deepcopy(doc0)
        config_key = verifier.SWEEP_PARAM_CONFIG_KEYS[param_key]
        value = doc0["configuration"][config_key]
        return dict(
            configuration=dict(
                mode="sweep",
                base_mode="llm_memory",
                sweep_schema_version=1,
                run_count=2,
                sweep_max_runs=16,
                sweep_parameters={param_key: [value, value]},
            ),
            planned_runs=2,
            attempted_runs=2,
            completed_runs=2,
            status="complete",
            status_reason=None,
            conclusions_valid=True,
            execution_time_sec=0.25,
            timestamp="2026-01-01T00:00:00Z",
            version="0.64.1",
            runs=[
                dict(
                    index=0,
                    parameters={param_key: value},
                    status="complete",
                    status_reason=None,
                    result=doc0,
                ),
                dict(
                    index=1,
                    parameters={param_key: value},
                    status="complete",
                    status_reason=None,
                    result=doc1,
                ),
            ],
        )

    def test_llm_sweep_envelope_verification_and_mutations(self):
        for fixture_name, param_key in (
            ("cpu-decode-contiguous", "context-tokens"),
            ("cpu-decode-paged", "kv-block-tokens"),
            ("cpu-prefill-contiguous", "prompt-tokens"),
            ("metal-prefill-paged", "attention-query-tile-tokens"),
        ):
            with self.subTest(fixture=fixture_name, key=param_key):
                envelope = self.make_sweep_envelope(fixture_name, param_key)
                verdict = verifier.verify(envelope)
                self.assertTrue(verdict["artifact_consistent"])
                self.assertTrue(verdict["run_accepted"])
                self.assertEqual(verdict["artifact_kind"], "llm_sweep")
                self.assertEqual(verdict["completed_runs"], 1)

        def nested_truncated(status):
            doc = truncated(status)
            doc["configuration"]["output_file"] = ""
            return doc

        failed_envelope = self.make_sweep_envelope()
        failed_envelope["runs"][0]["status"] = "failed"
        failed_envelope["runs"][0]["status_reason"] = "runtime-failure"
        failed_envelope["runs"][0]["result"] = nested_truncated("failed")
        failed_envelope["completed_runs"] = 0
        failed_envelope["status"] = "failed"
        failed_envelope["status_reason"] = "runtime-failure"
        failed_envelope["conclusions_valid"] = False
        verdict = verifier.verify(failed_envelope)
        self.assertTrue(verdict["artifact_consistent"])
        self.assertFalse(verdict["run_accepted"])
        self.assertEqual(verdict["reason_code"], "sweep-not-accepted")

        null_failed_envelope = self.make_sweep_envelope()
        null_failed_envelope["runs"][0]["status"] = "failed"
        null_failed_envelope["runs"][0]["status_reason"] = "memory-budget-exceeded"
        null_failed_envelope["runs"][0]["result"] = None
        null_failed_envelope["completed_runs"] = 0
        null_failed_envelope["status"] = "failed"
        null_failed_envelope["status_reason"] = "memory-budget-exceeded"
        null_failed_envelope["conclusions_valid"] = False
        verdict = verifier.verify(null_failed_envelope)
        self.assertTrue(verdict["artifact_consistent"])
        self.assertFalse(verdict["run_accepted"])

        null_interrupted_envelope = self.make_sweep_envelope()
        null_interrupted_envelope["runs"][0]["status"] = "interrupted"
        null_interrupted_envelope["runs"][0]["status_reason"] = "preparation-interrupted"
        null_interrupted_envelope["runs"][0]["result"] = None
        null_interrupted_envelope["completed_runs"] = 0
        null_interrupted_envelope["status"] = "interrupted"
        null_interrupted_envelope["status_reason"] = "preparation-interrupted"
        null_interrupted_envelope["conclusions_valid"] = False
        verdict = verifier.verify(null_interrupted_envelope)
        self.assertTrue(verdict["artifact_consistent"])
        self.assertFalse(verdict["run_accepted"])

        mutation_cases = [
            (
                "unsupported-sweep-schema",
                lambda e: e["configuration"].update(sweep_schema_version=2),
            ),
            (
                "sweep-plan-count-mismatch",
                lambda e: e.update(planned_runs=2),
            ),
            (
                "sweep-plan-count-mismatch",
                lambda e: e["configuration"].update(
                    sweep_max_runs=1000,
                    sweep_parameters={
                        "context-tokens": list(range(1, 101)),
                        "kv-block-tokens": list(range(1, 101)),
                        "prompt-tokens": list(range(1, 101)),
                        "attention-query-tile-tokens": list(range(1, 101)),
                    },
                ),
            ),
            (
                "sweep-attempted-runs-mismatch",
                lambda e: e.update(attempted_runs=0),
            ),
            (
                "sweep-nested-output-file-mismatch",
                lambda e: e["runs"][0]["result"]["configuration"].update(output_file="out.json"),
            ),
            (
                "sweep-parameters-mismatch",
                lambda e: e["runs"][0]["parameters"].update({"context-tokens": 999}),
            ),
            (
                "sweep-nested-parameter-mismatch",
                lambda e: (
                    e["configuration"]["sweep_parameters"].update({"context-tokens": [999]}),
                    e["runs"][0]["parameters"].update({"context-tokens": 999}),
                ),
            ),
            (
                "sweep-run-status-mismatch",
                lambda e: e["runs"][0].update(result=nested_truncated("failed")),
            ),
            (
                "sweep-completed-runs-mismatch",
                lambda e: e.update(completed_runs=0),
            ),
            (
                "sweep-status-mismatch",
                lambda e: e.update(conclusions_valid=False),
            ),
            (
                "counter-mismatch",
                lambda e: e["runs"][0]["result"]["counters"].update(completed_work_units="0"),
            ),
        ]
        for reason, mutate in mutation_cases:
            with self.subTest(reason=reason):
                envelope = self.make_sweep_envelope()
                mutate(envelope)
                self.assert_reason(envelope, reason)

        # Splicing runs from different base configurations or build manifests must fail.
        for splice_name, mutate in (
            ("manifest", lambda e: e["runs"][1]["result"]["build_manifest"].update(compiler="other-cc")),
            ("argv", lambda e: e["runs"][1]["result"]["configuration"].update(argv=["other"])),
        ):
            with self.subTest(splice=splice_name):
                spliced = self.make_two_run_sweep_envelope()
                mutate(spliced)
                self.assert_reason(spliced, "sweep-base-config-mismatch")

        # Shared work budget must apply across the entire sweep artifact, not reset per run.
        single_verdict = verifier.verify(self.make_sweep_envelope())
        single_work = single_verdict["oracle_work_used"]
        two_run = self.make_two_run_sweep_envelope()
        two_verdict = verifier.verify(two_run)
        self.assertGreater(two_verdict["oracle_work_used"], single_work)
        with mock.patch.object(verifier, "MAX_WORK", single_work + 1), self.assertRaises(
            verifier.Rejected
        ) as caught:
            verifier.verify(two_run)
        self.assertEqual(caught.exception.reason, "unsupported-work-limit")
        self.assertTrue(caught.exception.unsupported)

        with tempfile.TemporaryDirectory(prefix="llm-sweep-verifier-", dir="/tmp") as directory:
            path = Path(directory) / "sweep.json"
            for envelope, expected in ((self.make_sweep_envelope(), 0), (failed_envelope, 1)):
                path.write_text(json.dumps(envelope))
                result = subprocess.run(
                    [sys.executable, str(ROOT / "script-examples/verify_llm_result.py"), str(path)],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                parsed = json.loads(result.stdout)
                self.assertEqual(parsed["artifact_kind"], "llm_sweep")

    def test_llm_sweep_order_independent_parameters_and_envelope_invariants(self):
        # Re-serializing a multi-axis sweep with sorted JSON keys must remain verifiable.
        doc0 = fixture("cpu-prefill-paged")
        doc0["configuration"]["output_file"] = ""
        p0 = doc0["configuration"]["prompt_tokens"]
        q0 = doc0["configuration"]["attention_query_tile_tokens"]
        two_dim = dict(
            configuration=dict(
                mode="sweep",
                base_mode="llm_memory",
                sweep_schema_version=1,
                run_count=4,
                sweep_max_runs=16,
                sweep_parameters={
                    "prompt-tokens": [p0, p0 * 2],
                    "attention-query-tile-tokens": [q0, q0 + 1],
                },
            ),
            planned_runs=4,
            attempted_runs=2,
            completed_runs=1,
            status="failed",
            status_reason="memory-budget-exceeded",
            conclusions_valid=False,
            execution_time_sec=0.25,
            timestamp="2026-01-01T00:00:00Z",
            version="0.64.1",
            runs=[
                dict(
                    index=0,
                    parameters={"prompt-tokens": p0, "attention-query-tile-tokens": q0},
                    status="complete",
                    status_reason=None,
                    result=doc0,
                ),
                dict(
                    index=1,
                    parameters={"prompt-tokens": p0, "attention-query-tile-tokens": q0 + 1},
                    status="failed",
                    status_reason="memory-budget-exceeded",
                    result=None,
                ),
            ],
        )
        reserialized = json.loads(json.dumps(two_dim, sort_keys=True))
        self.assertEqual(
            list(reserialized["configuration"]["sweep_parameters"].keys()),
            ["attention-query-tile-tokens", "prompt-tokens"],
        )
        verdict = verifier.verify(reserialized)
        self.assertTrue(verdict["artifact_consistent"])
        self.assertFalse(verdict["run_accepted"])

        # A parameter tuple inconsistent with every Cartesian axis ordering must be rejected.
        bad_order = copy.deepcopy(reserialized)
        bad_order["runs"][1]["parameters"] = {
            "prompt-tokens": p0 * 2,
            "attention-query-tile-tokens": q0 + 1,
        }
        self.assert_reason(bad_order, "sweep-parameters-mismatch")

        # Valid checkpoint-write failure after a completed run and valid zero-run interruption.
        checkpoint_failed = self.make_sweep_envelope()
        checkpoint_failed["status"] = "failed"
        checkpoint_failed["status_reason"] = "checkpoint-write-failed"
        checkpoint_failed["conclusions_valid"] = False
        verdict = verifier.verify(checkpoint_failed)
        self.assertTrue(verdict["artifact_consistent"])
        self.assertFalse(verdict["run_accepted"])

        zero_run_interrupted = self.make_two_run_sweep_envelope()
        zero_run_interrupted.update(
            attempted_runs=0,
            completed_runs=0,
            status="interrupted",
            status_reason="interruption-requested-before-run",
            conclusions_valid=False,
            runs=[],
        )
        verdict = verifier.verify(zero_run_interrupted)
        self.assertTrue(verdict["artifact_consistent"])
        self.assertFalse(verdict["run_accepted"])

        # Forged execution resumption after a failed run must fail.
        resumed = self.make_two_run_sweep_envelope()
        resumed["runs"][0].update(status="failed", status_reason="memory-budget-exceeded", result=None)
        resumed.update(
            completed_runs=1,
            status="interrupted",
            status_reason="interruption-requested",
            conclusions_valid=False,
        )
        self.assert_reason(resumed, "sweep-invalid-execution-prefix")

        # Envelope status, per-run status_reason, and metadata invariants.
        invariant_cases = [
            (
                "failed-run-in-interrupted-envelope",
                "sweep-status-mismatch",
                lambda e: (
                    e["runs"][0].update(status="failed", status_reason="runtime-failure", result=None),
                    e.update(
                        completed_runs=0,
                        status="interrupted",
                        status_reason="interruption-requested",
                        conclusions_valid=False,
                    ),
                ),
            ),
            (
                "complete-run-in-failed-envelope-without-checkpoint-reason",
                "sweep-status-mismatch",
                lambda e: e.update(status="failed", status_reason="runtime-failure", conclusions_valid=False),
            ),
            (
                "zero-run-partial-envelope",
                "sweep-status-mismatch",
                lambda e: e.update(
                    attempted_runs=0,
                    completed_runs=0,
                    status="partial",
                    status_reason="partial-results",
                    conclusions_valid=False,
                    runs=[],
                ),
            ),
            (
                "all-complete-partial-envelope",
                "sweep-status-mismatch",
                lambda e: e.update(status="partial", status_reason="partial-results", conclusions_valid=False),
            ),
            (
                "complete-envelope-with-status-reason",
                "sweep-status-mismatch",
                lambda e: e.update(status_reason="unexpected"),
            ),
            (
                "complete-run-with-status-reason",
                "sweep-run-status-mismatch",
                lambda e: e["runs"][0].update(status_reason="unexpected"),
            ),
            (
                "failed-run-without-status-reason",
                "sweep-run-status-mismatch",
                lambda e: (
                    e["runs"][0].update(status="failed", status_reason=None, result=None),
                    e.update(
                        completed_runs=0,
                        status="failed",
                        status_reason="runtime-failure",
                        conclusions_valid=False,
                    ),
                ),
            ),
            ("negative-execution-time", "sweep-metadata-invalid", lambda e: e.update(execution_time_sec=-0.1)),
            ("bool-execution-time", "sweep-metadata-invalid", lambda e: e.update(execution_time_sec=True)),
            ("empty-timestamp", "sweep-metadata-invalid", lambda e: e.update(timestamp="")),
            ("empty-version", "sweep-metadata-invalid", lambda e: e.update(version="")),
        ]
        for label, reason, mutate in invariant_cases:
            with self.subTest(case=label):
                envelope = self.make_sweep_envelope()
                mutate(envelope)
                self.assert_reason(envelope, reason)

        # Illegal sweep_parameters must be rejected even when nested runs have result=None.
        null_param_cases = [
            (
                "mixed-decode-and-prefill",
                "sweep-parameters-invalid",
                dict(
                    sweep_parameters={"context-tokens": [16], "prompt-tokens": [16]},
                    run_count=1,
                ),
                {"context-tokens": 16, "prompt-tokens": 16},
                1,
            ),
            (
                "non-power-of-two-kv-block",
                "sweep-parameter-value-invalid",
                dict(sweep_parameters={"kv-block-tokens": [24]}, run_count=1),
                {"kv-block-tokens": 24},
                1,
            ),
            (
                "kv-block-above-u32",
                "sweep-parameter-value-invalid",
                dict(sweep_parameters={"kv-block-tokens": [1 << 32]}, run_count=1),
                {"kv-block-tokens": 1 << 32},
                1,
            ),
            (
                "tile-exceeds-prompt",
                "sweep-parameter-value-invalid",
                dict(
                    sweep_parameters={"prompt-tokens": [32, 64], "attention-query-tile-tokens": [16, 48]},
                    run_count=4,
                ),
                {"prompt-tokens": 32, "attention-query-tile-tokens": 16},
                4,
            ),
        ]
        for label, reason, cfg_patch, run_params, planned in null_param_cases:
            with self.subTest(null_param_case=label):
                envelope = self.make_sweep_envelope()
                envelope["configuration"].update(cfg_patch)
                envelope.update(
                    planned_runs=planned,
                    attempted_runs=1,
                    completed_runs=0,
                    status="failed",
                    status_reason="memory-budget-exceeded",
                    conclusions_valid=False,
                )
                envelope["runs"][0].update(
                    parameters=run_params,
                    status="failed",
                    status_reason="memory-budget-exceeded",
                    result=None,
                )
                self.assert_reason(envelope, reason)

    def test_llm_sweep_hashes_binary_once_across_runs(self):
        import hashlib

        with tempfile.NamedTemporaryFile(dir="/tmp") as binary:
            binary.write(b"known sweep binary bytes")
            binary.flush()
            digest = hashlib.sha256(b"known sweep binary bytes").hexdigest()
            two_run = self.make_two_run_sweep_envelope()
            for run in two_run["runs"]:
                run["result"]["build_manifest"]["binary_sha256"] = digest
            with mock.patch.object(verifier, "hash_binary", wraps=verifier.hash_binary) as mock_hash:
                verdict = verifier.verify(two_run, binary.name)
            self.assertTrue(verdict["run_accepted"])
            self.assertEqual(verdict["checks"]["build"], "binary-bound")
            self.assertEqual(mock_hash.call_count, 1)

    def test_unmaterialized_metal_paged_failure_and_prefill_append_offset(self):
        # Prefill paged layout requires null decode_append_offset_in_last_block; decode requires decimal string.
        bad_prefill_layout = fixture("cpu-prefill-paged")
        bad_prefill_layout["resolved_plan"]["layout"]["decode_append_offset_in_last_block"] = "0"
        self.assert_reason(bad_prefill_layout, "layout-mismatch")
        bad_decode_layout = fixture("cpu-decode-paged")
        bad_decode_layout["resolved_plan"]["layout"]["decode_append_offset_in_last_block"] = None
        self.assert_reason(bad_decode_layout, "layout-mismatch")

        # Complete Metal paged artifacts must still have materialized permutation fields.
        bad_complete_metal = fixture("metal-prefill-paged")
        bad_complete_metal["resolved_plan"]["layout"]["permutation_sha256"] = None
        self.assert_reason(bad_complete_metal, "permutation-digest-mismatch")

        # Build an unattempted Metal paged unsupported artifact with non-unit P % G (P=6, G=4, record=3).
        import hashlib

        doc = fixture("metal-prefill-paged")
        c = doc["configuration"]
        r = doc["resolved_plan"]
        g = r["geometry"]
        layout = r["layout"]
        c.update(prompt_tokens=6, kv_block_tokens=4)
        # Recompute exact geometry/resources for P=6, Q=2, G=4, L=2, B=2, R=3, W=1048576.
        p, q, block, layers, batch, record, weight = 6, 2, 4, 2, 2, 3, 1048576
        ends = [2, 4, 6]
        blocks = 2
        entries = batch * blocks
        logical = layers * batch * p * record
        physical = layers * batch * blocks * block * record
        read = 2 * layers * batch * record * sum(ends)
        write = 2 * layers * batch * record * p
        g["prefill"].update(
            prompt_tokens=p,
            attention_query_tile_tokens=q,
            tile_count=str(len(ends)),
            attention_prefix_token_visits_per_sequence=str(sum(ends)),
        )
        g.update(
            k_or_v_sequence_visible_bytes=str(p * record),
            k_mapping_bytes=str(physical),
            v_mapping_bytes=str(physical),
            kv_capacity_bytes=str(2 * physical),
            kv_read_bytes_per_work_unit=str(read),
            kv_write_bytes_per_work_unit=str(write),
            kv_only_effective_model_payload_bytes_per_work_unit=str(read + write),
            mixed_effective_model_payload_bytes_per_work_unit=str(weight + read + write),
            total_data_mapping_bytes=str(weight + 2 * physical),
        )
        pairs = p * (p + 1) // 2
        attn = pairs * layers * batch * g["query_head_count"]
        r["model_context"]["prefill"].update(
            causal_token_pairs_per_sequence=str(pairs),
            logical_attention_pairs=str(attn),
            logical_attention_fma_terms=str(attn * g["head_dimension"]),
        )
        r["resources"].update(
            k_logical_bytes=str(logical),
            v_logical_bytes=str(logical),
            k_physical_length_bytes=str(physical),
            v_physical_length_bytes=str(physical),
            k_layout_padding_bytes=str(physical - logical),
            v_layout_padding_bytes=str(physical - logical),
        )
        r["resources"]["metal"]["valid"] = False
        r["resources"]["metal"]["execution_identity"] = None
        r["resources"]["metal"]["resource_identity"] = None
        r["resources"]["metal"]["msl_revision"] = None
        r["resources"]["metal"]["msl_source_sha256"] = None
        r["component_identities"].update(msl_revision=None, msl_source_sha256=None)
        profile_components = dict(verifier.PROFILES["metal-prefill-paged"])
        profile_components.update(msl_revision=None, msl_source_sha256=None)
        comp_id = "llm-memory-components-v1" + "".join(
            "|" + k + "=" + ("null" if v is None else str(len(v)) + ":" + v)
            for k, v in profile_components.items()
            if k != "run_policy_version"
        )
        r["component_identities"]["identity"] = comp_id
        prefix_block_visits = sum((t + block - 1) // block for t in ends)
        lookups_per_ls = blocks + 2 * prefix_block_visits
        plan_updates = dict(
            component_identity_size=str(len(comp_id)),
            component_identity=comp_id,
            k_or_v_sequence_visible_bytes=str(p * record),
            kv_block_tokens=str(block),
            kv_blocks_per_sequence=str(blocks),
            physical_blocks_per_layer=str(entries),
            total_physical_blocks=str(layers * entries),
            kv_block_bytes=str(block * record),
            last_block_tokens=str(p - (blocks - 1) * block),
            last_block_valid_bytes=str((p - (blocks - 1) * block) * record),
            decode_append_offset_in_last_block="0",
            k_logical_bytes=str(logical),
            v_logical_bytes=str(logical),
            k_layout_padding_bytes=str(physical - logical),
            v_layout_padding_bytes=str(physical - logical),
            block_table_entries=str(entries),
            block_table_bytes=str(4 * entries),
            layout_metadata_lookups_per_layer_sequence_per_work_unit=str(lookups_per_ls),
            k_mapping_bytes=str(physical),
            v_mapping_bytes=str(physical),
            kv_capacity_bytes=str(2 * physical),
            kv_read_bytes_per_work_unit=str(read),
            kv_write_bytes_per_work_unit=str(write),
            kv_only_payload_bytes_per_work_unit=str(read + write),
            mixed_payload_bytes_per_work_unit=str(weight + read + write),
            total_data_mapping_bytes=str(weight + 2 * physical),
            prefill_prompt_tokens=str(p),
            prefill_tile_count=str(len(ends)),
            prefill_prefix_token_visits_per_sequence=str(sum(ends)),
            prefill_causal_token_pairs=str(pairs),
            prefill_logical_attention_pairs=str(attn),
            prefill_logical_attention_fma_terms=str(attn * g["head_dimension"]),
            prefill_prefix_block_visits_per_sequence=str(prefix_block_visits),
            metal_execution_resolved="0",
            metal_execution_identity_size="0",
            metal_execution_identity_sha256="0" * 64,
        )
        _, plan_pairs = verifier.identity_fields(r["plan_identity"])
        r["plan_identity"] = verifier.pipe(
            "llm-memory-work-plan-v1", [(k, plan_updates.get(k, v)) for k, v in plan_pairs]
        )
        r["scenario_plans"] = []
        r["frozen_plan_refs"] = {s: None for s in verifier.SCENARIOS}
        doc["calibration"]["attempts"] = {s: [] for s in verifier.SCENARIOS}
        doc["backend_evidence"]["metal"]["workload_pipelines"] = []
        doc["backend_evidence"]["metal"]["resources"]["table_permutation"] = None
        for row in doc["measurements"]:
            row.update(
                plan_ref=None,
                status="not_run",
                reason_code="not-run",
                attempted=False,
                completed_work_units=0,
                completed_effective_model_payload_bytes="0",
                completed_layout_metadata_lookup_count="0",
                completed_layout_metadata_read_bytes="0",
                completed_task_accounted_bytes="0",
                completion_derivation=None,
                elapsed_seconds=None,
                synthetic_work_unit_latency_seconds=None,
                synthetic_memory_work_units_per_second=None,
                effective_model_payload_gb_s=None,
            )
            row["execution"].update(status="not_run", valid=None)
            row["execution"]["timing"] = dict(evaluated=False, valid=None, diagnostic_elapsed_seconds=None, cpu_raw=None)
            row["execution"].pop("metal", None)
            for check in row["execution"]["validation"]["checks"]:
                check["evaluated"] = False if check["applicable"] else None
                check["valid"] = None
            row["checksum"].update(
                status="not_evaluated",
                checksum_valid=None,
                actual_worker_checksums=None,
                actual_run_checksum=None,
            )
        for loop in doc["loop_records"]:
            loop["realized_order"] = []
            loop["realized_order_count"] = 0
        doc.update(
            status="unsupported",
            reason_code="metal-device-unavailable",
            results_complete=False,
            run_accepted=False,
            scenario_order_balance_complete=False,
            interruption_requested=False,
        )
        for k in (
            "planned_work_units",
            "planned_effective_model_payload_bytes",
            "planned_layout_metadata_lookup_count",
            "planned_layout_metadata_read_bytes",
            "planned_task_accounted_bytes",
        ):
            doc["counters"][k] = "0"
        refresh(doc)

        # Subcase A: layout geometry present with non-zero layout decode_append_offset_in_last_block (P=6, G=4 -> 3),
        # while plan_identity keeps 0, JSON layout field is None, and permutation fields are unmaterialized (None).
        geom_only = copy.deepcopy(doc)
        table_bytes = 4 * entries
        bits = (entries + 7) // 8
        resident = 2 * physical + table_bytes
        layout_append = ((p - 1) % block) * record
        self.assertEqual(layout_append, 3)
        kv_layout_ver = r["component_identities"]["kv_layout_version"]
        geom_only["resolved_plan"]["layout"].update(
            kv_block_tokens=block,
            blocks_per_sequence=str(blocks),
            physical_blocks_per_layer=str(entries),
            total_physical_blocks=str(layers * entries),
            block_bytes=str(block * record),
            last_block_tokens=str(p - (blocks - 1) * block),
            last_block_valid_bytes=str((p - (blocks - 1) * block) * record),
            decode_append_offset_in_last_block=None,
            block_table_entries=str(entries),
            block_table_bytes=str(table_bytes),
            layout_geometry_identity=verifier.pipe(
                "llm-kv-layout-geometry-v1",
                [
                    ("kv_layout_version", f"{len(kv_layout_ver)}:{kv_layout_ver}"),
                    ("sequence_tokens", p),
                    ("kv_block_tokens", block),
                    ("layer_count", layers),
                    ("batch_size", batch),
                    ("k_or_v_record_bytes_per_layer", record),
                    ("blocks_per_sequence", blocks),
                    ("physical_blocks_per_layer", entries),
                    ("total_physical_blocks", layers * entries),
                    ("block_bytes", block * record),
                    ("last_block_tokens", p - (blocks - 1) * block),
                    ("last_block_valid_bytes", (p - (blocks - 1) * block) * record),
                    ("decode_append_offset_in_last_block", layout_append),
                    ("k_logical_bytes", logical),
                    ("v_logical_bytes", logical),
                    ("k_physical_bytes", physical),
                    ("v_physical_bytes", physical),
                    ("k_layout_padding_bytes", physical - logical),
                    ("v_layout_padding_bytes", physical - logical),
                    ("block_table_entries", entries),
                    ("block_table_bytes", table_bytes),
                    ("validation_bitset_bytes", bits),
                    ("transient_peak_bytes", table_bytes + bits),
                    ("resident_layout_bytes", resident),
                    ("known_owned_peak_bytes", resident + bits),
                    ("permutation_iterations", entries - 1),
                    ("validation_entries", entries),
                    ("hash_entries", entries),
                    ("upload_bytes", table_bytes),
                ],
            ),
            layout_identity=None,
            permutation_domain_uint64_hex=None,
            permutation_seed_uint64_decimal=None,
            permutation_algorithm_version=None,
            permutation_entry_count=None,
            permutation_sha256=None,
            permutation_identity=None,
        )
        verdict = verifier.verify(geom_only)
        self.assertTrue(verdict["artifact_consistent"])
        self.assertFalse(verdict["run_accepted"])

        # Replacing layout_geometry_identity's append offset with 0 must fail for P=6, G=4.
        wrong_geom = copy.deepcopy(geom_only)
        wrong_geom["resolved_plan"]["layout"]["layout_geometry_identity"] = wrong_geom["resolved_plan"]["layout"][
            "layout_geometry_identity"
        ].replace("decode_append_offset_in_last_block=3", "decode_append_offset_in_last_block=0")
        self.assert_reason(wrong_geom, "layout-identity-mismatch")

        # Subcase B: all paged layout geometry and permutation fields are None on unresolved Metal failure.
        all_null_layout = copy.deepcopy(geom_only)
        for key in (
            "kv_block_tokens",
            "blocks_per_sequence",
            "physical_blocks_per_layer",
            "total_physical_blocks",
            "block_bytes",
            "last_block_tokens",
            "last_block_valid_bytes",
            "decode_append_offset_in_last_block",
            "block_table_entries",
            "block_table_bytes",
            "layout_geometry_identity",
        ):
            all_null_layout["resolved_plan"]["layout"][key] = None
        verdict = verifier.verify(all_null_layout)
        self.assertTrue(verdict["artifact_consistent"])
        self.assertFalse(verdict["run_accepted"])
        del hashlib


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

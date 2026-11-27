"""Independent chain-oracle goldens (all 4 chain modes + canonical rotation),
schema-v1 fixture verification, --compare-locality, --max-chain-nodes,
sweep envelope checks, and semantic mutations for verify_cpu_latency_result.py.
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
import verify_cpu_latency_result as verifier

FIXTURE_PATH = ROOT / "tests/fixtures/cpu-latency-schema-v1-complete-current.json"


def load_fixture():
    return json.loads(FIXTURE_PATH.read_text())


def make_truncated(status):
    """Build a consistent partial/interrupted/failed latency artifact."""
    doc = load_fixture()
    measured_keep = 1
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
        row["latency_ns"] = None
        row["completed_accesses"] = 0
        row["completed_chain_cycles"] = 0
        row["final_pointer_verification"] = {"expected_node": 0, "actual_node": None, "valid": None}
        row["sample_windows"] = None
        ex = row["execution"]
        ex["status"] = "not_run"
        ex["valid"] = None
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
        target_order_balance_complete=False,
    )
    rows = doc["measurements"]
    per_loop = 2
    for loop_idx, rec in enumerate(doc["loop_records"]):
        slice_rows = rows[loop_idx * per_loop : (loop_idx + 1) * per_loop]
        realized = [r["target"] for r in slice_rows if r["attempted"]]
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
        completed_accesses=str(sum(r["completed_accesses"] for r in rows)),
    )
    for tname, a in doc["aggregates"]["targets"].items():
        ids = [
            r["measurement_id"]
            for r in rows
            if r["target"] == tname and r["status"] == "measured"
        ]
        vals = [rows[idx]["latency_ns"] for idx in ids]
        st = common.compute_statistics(vals)
        a["accepted_measurement_ids"] = ids
        a["value"] = st["median"] if st is not None else None
        a["statistics"] = st
        a["quality"]["repeatability_assessment_available"] = len(ids) >= 2
        a["quality"]["run_complete"] = len(ids) == doc["configuration"]["loop_count"]
        a["quality"]["cv_warning"] = False
        pooled = []
        for idx in ids:
            pooled.extend(rows[idx]["sample_windows"]["samples_ns"])
        if ids:
            a["pooled_sample_distribution"]["values_ns"] = pooled
            a["pooled_sample_distribution"]["statistics"] = common.compute_statistics(pooled)
        else:
            a["pooled_sample_distribution"]["values_ns"] = []
            a["pooled_sample_distribution"]["statistics"] = None
    return doc


def build_locality_enabled_doc():
    """Augment a single-target main_memory fixture with paired --compare-locality."""
    doc = load_fixture()
    base_seed = int(doc["seeds"]["base_seed_uint64_decimal"])
    stride = doc["configuration"]["latency_stride_bytes"]
    main_bytes = int(doc["resolved_plan"]["targets"]["main_memory"]["bytes"])
    node_count = main_bytes // stride
    doc["configuration"]["compare_locality"] = True
    new_peak = int(doc["memory_budget"]["known_owned_peak_bytes"]) + main_bytes
    doc["memory_budget"]["known_owned_peak_bytes"] = str(new_peak)
    doc["memory_budget"]["runtime_peak_bytes"] = str(new_peak)

    s16, _idx16, d16 = verifier.build_latency_chain(
        base_seed, verifier.LOCALITY_16K_DOMAIN, node_count, "random-box", 16384 // stride
    )
    sg, _idxg, dg = verifier.build_latency_chain(
        base_seed, verifier.LOCALITY_GLOBAL_DOMAIN, node_count, "global-random", 0
    )
    doc["seeds"]["target_domain_seeds"]["locality_16k"] = str(s16)
    doc["seeds"]["target_domain_seeds"]["locality_global"] = str(sg)
    doc["resolved_plan"]["resources"]["compare_buffer"] = {
        "requested_bytes": str(main_bytes),
        "mapped_bytes": str(main_bytes),
        "non_cacheable_requested": False,
        "madvise_advice": 3,
        "madvise_applied": True,
        "madvise_errno": 0,
        "read_only_protected": True,
        "mprotect_errno": 0,
    }
    doc["resolved_plan"]["chain_identities"]["locality_16k"] = {
        "node_count": node_count,
        "stride_bytes": stride,
        "chain_mode": "random-box",
        "tlb_locality_kb": 16,
        "chain_seed_uint64_decimal": str(s16),
        "chain_permutation_sha256": d16,
    }
    doc["resolved_plan"]["chain_identities"]["locality_global"] = {
        "node_count": node_count,
        "stride_bytes": stride,
        "chain_mode": "global-random",
        "tlb_locality_kb": 0,
        "chain_seed_uint64_decimal": str(sg),
        "chain_permutation_sha256": dg,
    }

    rounds = []
    v16_list = []
    vg_list = []
    tick = 9000000
    for loop_idx in range(doc["configuration"]["loop_count"]):
        for r_idx in range(3):
            dt16 = 300 + loop_idx + r_idx
            dtg = 900 + loop_idx * 2 + r_idx
            el16 = dt16 * 125.0 / 3.0 / 1e9
            elg = dtg * 125.0 / 3.0 / 1e9
            acc = node_count * 2
            l16 = (el16 * 1e9) / acc
            lg = (elg * 1e9) / acc
            v16_list.append(l16)
            vg_list.append(lg)
            rounds.append(
                {
                    "loop_index": loop_idx,
                    "round_index": r_idx,
                    "locality_16k": {
                        "status": "measured",
                        "attempted": True,
                        "completed_accesses": acc,
                        "latency_ns": l16,
                        "final_pointer_verification": {"expected_node": 0, "actual_node": 0, "valid": True},
                        "execution": {
                            "timing": {
                                "elapsed_seconds": el16,
                                "cpu_raw": {"start_ticks": str(tick), "stop_ticks": str(tick + dt16), "delta_ticks": str(dt16), "timebase_numer": 125, "timebase_denom": 3},
                            }
                        },
                    },
                    "locality_global": {
                        "status": "measured",
                        "attempted": True,
                        "completed_accesses": acc,
                        "latency_ns": lg,
                        "final_pointer_verification": {"expected_node": 0, "actual_node": 0, "valid": True},
                        "execution": {
                            "timing": {
                                "elapsed_seconds": elg,
                                "cpu_raw": {"start_ticks": str(tick + 2000), "stop_ticks": str(tick + 2000 + dtg), "delta_ticks": str(dtg), "timebase_numer": 125, "timebase_denom": 3},
                            }
                        },
                    },
                }
            )
            tick += 5000
    doc["locality_comparison"] = {"rounds": rounds}
    m16 = common.compute_statistics(v16_list)["median"]
    mg = common.compute_statistics(vg_list)["median"]
    doc["aggregates"]["locality_comparison"] = {
        "locality_16k_latency_ns": m16,
        "global_random_latency_ns": mg,
        "locality_latency_delta_ns": mg - m16,
    }
    return doc


class CpuLatencyVerifierTests(unittest.TestCase):
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

        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "script-examples/verify_cpu_latency_result.py"),
                str(FIXTURE_PATH),
                "--require-raw-timing",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertTrue(json.loads(proc.stdout)["run_accepted"])

    def test_all_four_chain_modes_canonical_rotation_and_aliases(self):
        # Verify canonical rotation indices[0] == 0, Hamiltonian single cycle, and SHA-256 across all 4 modes
        for mode_token, expected_mode in (
            ("global", "global-random"),
            ("global-random", "global-random"),
            ("random-box", "random-box"),
            ("random_in_box_random_box", "random-box"),
            ("same-random-in-box", "same-random-in-box"),
            ("same_random_in_box_increasing_box", "same-random-in-box"),
            ("diff-random-in-box", "diff-random-in-box"),
            ("diff_random_in_box_increasing_box", "diff-random-in-box"),
        ):
            with self.subTest(mode_token=mode_token):
                eff = verifier.resolve_chain_mode(mode_token, 4096)
                self.assertEqual(eff, expected_mode)
                # Use non-divisible node_count=23 and locality_nodes=6 to test partial final box too
                cseed, indices, digest = verifier.build_latency_chain(
                    424242, verifier.TARGET_DOMAINS["custom"], 23, eff, 6
                )
                self.assertEqual(indices[0], 0)
                self.assertEqual(len(indices), 23)
                self.assertEqual(sorted(indices), list(range(23)))
                self.assertEqual(len(digest), 64)
                self.assertEqual(cseed, common.splitmix64_mix(424242 ^ verifier.TARGET_DOMAINS["custom"]))

        # Verify 'auto' resolves to global-random when locality=0, random-box when locality>0
        self.assertEqual(verifier.resolve_chain_mode("auto", 0), "global-random")
        self.assertEqual(verifier.resolve_chain_mode("auto", 1024 * 1024), "random-box")

        # Full artifact verification across all 4 modes + digest mismatch rejection
        for mode in ("global-random", "random-box", "same-random-in-box", "diff-random-in-box"):
            with self.subTest(artifact_mode=mode):
                doc = load_fixture()
                doc["configuration"]["latency_chain_mode"] = mode
                base_seed = int(doc["seeds"]["base_seed_uint64_decimal"])
                stride = doc["configuration"]["latency_stride_bytes"]
                loc_nodes = (doc["configuration"]["latency_tlb_locality_kb"] * 1024) // stride
                for t in ("custom", "main_memory"):
                    ci = doc["resolved_plan"]["chain_identities"][t]
                    cseed, indices, digest = verifier.build_latency_chain(
                        base_seed, verifier.TARGET_DOMAINS[t], ci["node_count"], mode, loc_nodes
                    )
                    ci["chain_mode"] = mode
                    ci["chain_seed_uint64_decimal"] = str(cseed)
                    ci["chain_permutation_sha256"] = digest
                for row in doc["measurements"]:
                    t = row["target"]
                    ci = doc["resolved_plan"]["chain_identities"][t]
                    _, indices, _ = verifier.build_latency_chain(
                        base_seed, verifier.TARGET_DOMAINS[t], ci["node_count"], mode, loc_nodes
                    )
                    acc = row["completed_accesses"]
                    sw = row["sample_windows"]
                    base_step, rem = divmod(acc, sw["sample_count"])
                    cum = 0
                    end_nodes = []
                    for i in range(sw["sample_count"]):
                        cum += base_step + (1 if i < rem else 0)
                        end_nodes.append(indices[cum % ci["node_count"]])
                    sw["validation"]["end_node_indices"] = end_nodes
                self.assertTrue(verifier.verify(doc, require_raw=True)["run_accepted"])

                doc["resolved_plan"]["chain_identities"]["custom"]["chain_permutation_sha256"] = "0" * 64
                self.assert_reason(doc, "chain-digest-mismatch")

    def test_compare_locality_and_zero_samples(self):
        loc_doc = build_locality_enabled_doc()
        verdict = verifier.verify(loc_doc, require_raw=True)
        self.assertTrue(verdict["artifact_consistent"])
        self.assertTrue(verdict["run_accepted"])

        bad_delta = copy.deepcopy(loc_doc)
        bad_delta["aggregates"]["locality_comparison"]["locality_latency_delta_ns"] += 1.0
        self.assert_reason(bad_delta, "locality-delta-mismatch")

        zero_samples = load_fixture()
        zero_samples["configuration"]["latency_samples"] = 0
        for row in zero_samples["measurements"]:
            row["sample_windows"] = None
            row["execution"]["validation"]["checks"][4].update(
                applicable=False, evaluated=None, valid=None, reason_code="not-applicable"
            )
        for a in zero_samples["aggregates"]["targets"].values():
            a["pooled_sample_distribution"] = None
        self.assertTrue(verifier.verify(zero_samples, require_raw=True)["run_accepted"])

    def test_max_chain_nodes_limit_and_exit_2(self):
        doc = load_fixture()  # main_memory has 4096 nodes
        self.assert_reason(doc, "unsupported-chain-node-limit", unsupported=True, max_chain_nodes=1024)

        with tempfile.TemporaryDirectory(prefix="cpu-lat-maxnodes-") as directory:
            p = Path(directory) / "lat.json"
            p.write_text(json.dumps(doc))
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "script-examples/verify_cpu_latency_result.py"),
                    str(p),
                    "--max-chain-nodes",
                    "1024",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
            parsed = json.loads(proc.stdout)
            self.assertIsNone(parsed["artifact_consistent"])
            self.assertEqual(parsed["reason_code"], "unsupported-chain-node-limit")

    def test_lifecycle_states_and_missing_raw_timing(self):
        for status in ("partial", "interrupted", "failed"):
            with self.subTest(status=status):
                doc = make_truncated(status)
                verdict = verifier.verify(doc)
                self.assertTrue(verdict["artifact_consistent"])
                self.assertFalse(verdict["run_accepted"])

        # Preflight !work_plan.valid failed artifact (work-cap-exceeded, empty targets/chains).
        preflight_fail = load_fixture()
        preflight_fail["resolved_plan"].update(
            valid=False,
            reason_code="work-cap-exceeded",
            targets={},
            resources={},
            chain_identities={},
            target_plans=[],
            frozen_plan_refs={},
        )
        preflight_fail["memory_budget"]["valid"] = False
        preflight_fail["memory_budget"]["reason_code"] = "memory-budget-overflow"
        preflight_fail["calibration"]["attempts"] = {}
        preflight_fail["measurements"] = []
        preflight_fail["aggregates"]["targets"] = {}
        preflight_fail["status"] = "failed"
        preflight_fail["reason_code"] = "work-cap-exceeded"
        preflight_fail["results_complete"] = False
        preflight_fail["run_accepted"] = False
        preflight_fail["target_order_balance_complete"] = False
        preflight_fail["counters"].update(
            attempted_loops=0,
            completed_loops=0,
            planned_measurements=0,
            attempted_measurements=0,
            terminal_measurements=0,
            measured_measurements=0,
            planned_accesses="0",
            completed_accesses="0",
        )
        for rec in preflight_fail["loop_records"]:
            rec["planned_order"] = []
            rec["realized_order"] = []
            rec["realized_order_count"] = 0
            rec["measurement_indexes"] = []
        pf_verdict = verifier.verify(preflight_fail)
        self.assertTrue(pf_verdict["artifact_consistent"])
        self.assertFalse(pf_verdict["run_accepted"])

        no_raw = load_fixture()
        for row in no_raw["measurements"]:
            row["execution"]["timing"].pop("cpu_raw")
        for t_cal in no_raw["calibration"]["attempts"].values():
            for att in t_cal:
                att["execution"]["timing"].pop("cpu_raw")
        v = verifier.verify(no_raw)
        self.assertEqual(v["checks"]["timing"], "elapsed-only")
        self.assert_reason(no_raw, "unsupported-missing-raw-timing", unsupported=True, require_raw=True)

    def test_mutation_reasons(self):
        cases = [
            ("unsupported-schema", lambda d: d.update(schema_version=2), True),
            ("unsupported-mode", lambda d: d.update(mode="cpu_bandwidth"), True),
            ("unsupported-methodology", lambda d: d.update(methodology_version="bad"), True),
            ("component-mismatch", lambda d: d["resolved_plan"]["component_identities"].update(chain_version="bad"), False),
            ("seed-mismatch", lambda d: d["seeds"]["target_domain_seeds"].update(custom="1"), False),
            ("chain-digest-mismatch", lambda d: d["resolved_plan"]["chain_identities"]["custom"].update(chain_permutation_sha256="f" * 64), False),
            ("final-pointer-mismatch", lambda d: d["measurements"][0]["final_pointer_verification"].update(actual_node=1), False),
            ("sample-timing-mismatch", lambda d: d["measurements"][0]["sample_windows"]["sample_delta_ticks"].__setitem__(0, "999999"), False),
            ("sample-window-validation-failed", lambda d: d["measurements"][0]["sample_windows"]["validation"].update(all_end_pointers_valid=False), False),
            ("sample-window-end-pointer-mismatch", lambda d: d["measurements"][0]["sample_windows"]["validation"]["end_node_indices"].__setitem__(0, 999999), False),
            ("pooled-sample-statistic-mismatch", lambda d: d["aggregates"]["targets"]["custom"]["pooled_sample_distribution"]["statistics"].update(average=999.0), False),
            ("tick-delta-mismatch", lambda d: d["measurements"][0]["execution"]["timing"]["cpu_raw"].update(stop_ticks="1"), False),
            ("elapsed-mismatch", lambda d: d["measurements"][0]["execution"]["timing"]["cpu_raw"].update(timebase_numer=126), False),
            ("latency-mismatch", lambda d: d["measurements"][0].update(latency_ns=999.0), False),
            ("aggregate-headline-mismatch", lambda d: d["aggregates"]["targets"]["custom"].update(value=999.0), False),
            ("aggregate-statistic-mismatch", lambda d: d["aggregates"]["targets"]["custom"]["statistics"].update(average=999.0), False),
            ("aggregate-population-mismatch", lambda d: d["aggregates"]["targets"]["custom"]["accepted_measurement_ids"].append(1), False),
            ("counter-mismatch", lambda d: d["counters"].update(completed_accesses="0"), False),
            ("checkpoint-policy", lambda d: d["checkpoint_lifecycle"].update(snapshot_interval_loops=9), False),
            ("checkpoint-current-persistence", lambda d: d["checkpoint_lifecycle"].update(current_persistence_success=False), False),
            ("run-acceptance-mismatch", lambda d: d.update(run_accepted=False), False),
            ("required-validation-failed", lambda d: d["measurements"][0]["execution"]["validation"]["checks"][3].update(valid=False), False),
            ("integer-encoding", lambda d: d["measurements"][0].update(plan_ref=True), False),
            ("run-status-mismatch", lambda d: d.update(interruption_requested=True), False),
            ("resource-evidence-invalid", lambda d: d["resolved_plan"]["resources"]["main_memory"]["buffer"].update(read_only_protected=False, mprotect_errno=13), False),
            ("resource-evidence-invalid", lambda d: d["resolved_plan"]["resources"]["main_memory"]["buffer"].update(requested_bytes="0"), False),
            ("resource-evidence-invalid", lambda d: d["resolved_plan"]["resources"]["main_memory"]["buffer"].update(non_cacheable_requested=True), False),
            ("resource-evidence-invalid", lambda d: d["resolved_plan"]["resources"].pop("custom"), False),
            ("resolved-sources-invalid", lambda d: d["configuration"]["resolved_sources"].update(buffer_size="default", cache_size="default", iterations="auto", seed="generated"), False),
            ("resolved-sources-invalid", lambda d: d["configuration"]["resolved_sources"].update(latency_chain_mode="bogus"), False),
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

        # Forged non-page-multiple TLB locality (20 KB on 16384 B pages) with recomputed chain hashes must still be rejected.
        forged_loc = load_fixture()
        forged_loc["configuration"]["latency_tlb_locality_kb"] = 20
        forged_loc["configuration"]["latency_chain_mode"] = "random-box"
        base_seed = int(forged_loc["seeds"]["base_seed_uint64_decimal"])
        stride = forged_loc["configuration"]["latency_stride_bytes"]
        loc_nodes = (20 * 1024) // stride
        for t in ("custom", "main_memory"):
            ci = forged_loc["resolved_plan"]["chain_identities"][t]
            cseed, indices, digest = verifier.build_latency_chain(
                base_seed, verifier.TARGET_DOMAINS[t], ci["node_count"], "random-box", loc_nodes
            )
            ci["chain_mode"] = "random-box"
            ci["tlb_locality_kb"] = 20
            ci["chain_seed_uint64_decimal"] = str(cseed)
            ci["chain_permutation_sha256"] = digest
        for row in forged_loc["measurements"]:
            t = row["target"]
            ci = forged_loc["resolved_plan"]["chain_identities"][t]
            _, indices, _ = verifier.build_latency_chain(
                base_seed, verifier.TARGET_DOMAINS[t], ci["node_count"], "random-box", loc_nodes
            )
            acc = row["completed_accesses"]
            sw = row["sample_windows"]
            base_step, rem = divmod(acc, sw["sample_count"])
            cum = 0
            end_nodes = []
            for i in range(sw["sample_count"]):
                cum += base_step + (1 if i < rem else 0)
                end_nodes.append(indices[cum % ci["node_count"]])
            sw["validation"]["end_node_indices"] = end_nodes
        self.assert_reason(forged_loc, "locality-not-page-multiple", unsupported=False)

        prefix_doc = make_truncated("interrupted")
        prefix_doc["measurements"][2] = load_fixture()["measurements"][2]
        self.assert_reason(prefix_doc, "invalid-execution-prefix")

    def test_binary_binding_and_strict_json_cli(self):
        with tempfile.TemporaryDirectory(prefix="cpu-lat-verifier-") as directory:
            d = Path(directory)
            bin_path = d / "memory_benchmark"
            bin_path.write_bytes(b"known latency binary")
            doc = load_fixture()
            doc["build_manifest"]["binary_sha256"] = hashlib.sha256(
                b"known latency binary"
            ).hexdigest()
            self.assertEqual(verifier.verify(doc, binary=bin_path)["checks"]["build"], "binary-bound")
            doc["build_manifest"]["binary_sha256"] = "0" * 64
            self.assert_reason(doc, "binary-hash-mismatch", binary=bin_path)

            p = d / "input.json"
            for payload, expected in (
                (load_fixture(), 0),
                (make_truncated("failed"), 1),
                ({"schema_version": 2}, 2),
            ):
                p.write_text(json.dumps(payload))
                res = subprocess.run(
                    [sys.executable, str(ROOT / "script-examples/verify_cpu_latency_result.py"), str(p)],
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

    def make_sweep(self, modes=("global-random", "random-box")):
        base_doc = load_fixture()
        base_doc["configuration"]["output_file"] = ""
        argv = self._strip_cli_flag(base_doc["configuration"]["argv"], "-b")
        argv.extend(["-S", "buffer-size=1", "-S", f"latency-chain-mode={','.join(modes)}"])
        base_doc["configuration"]["argv"] = argv
        base_doc["configuration"]["resolved_sources"]["buffer_size"] = "sweep"
        base_doc["configuration"]["resolved_sources"]["latency_chain_mode"] = "sweep"
        runs = []
        for idx, mode in enumerate(modes):
            doc = copy.deepcopy(base_doc)
            doc["configuration"]["latency_chain_mode"] = mode
            base_seed = int(doc["seeds"]["base_seed_uint64_decimal"])
            stride = doc["configuration"]["latency_stride_bytes"]
            loc_nodes = (doc["configuration"]["latency_tlb_locality_kb"] * 1024) // stride
            for t in ("custom", "main_memory"):
                ci = doc["resolved_plan"]["chain_identities"][t]
                cseed, indices, digest = verifier.build_latency_chain(
                    base_seed, verifier.TARGET_DOMAINS[t], ci["node_count"], mode, loc_nodes
                )
                ci["chain_mode"] = mode
                ci["chain_seed_uint64_decimal"] = str(cseed)
                ci["chain_permutation_sha256"] = digest
            for row in doc["measurements"]:
                t = row["target"]
                ci = doc["resolved_plan"]["chain_identities"][t]
                _, indices, _ = verifier.build_latency_chain(
                    base_seed, verifier.TARGET_DOMAINS[t], ci["node_count"], mode, loc_nodes
                )
                acc = row["completed_accesses"]
                sw = row["sample_windows"]
                base_step, rem = divmod(acc, sw["sample_count"])
                cum = 0
                end_nodes = []
                for s_idx in range(sw["sample_count"]):
                    cum += base_step + (1 if s_idx < rem else 0)
                    end_nodes.append(indices[cum % ci["node_count"]])
                sw["validation"]["end_node_indices"] = end_nodes
            runs.append(
                {
                    "index": idx,
                    "parameters": {"buffer-size": 1, "latency-chain-mode": mode},
                    "status": "complete",
                    "status_reason": None,
                    "result": doc,
                }
            )
        return {
            "configuration": {
                "mode": "sweep",
                "base_mode": "cpu_latency",
                "sweep_schema_version": 1,
                "run_count": len(modes),
                "sweep_max_runs": 16,
                "sweep_parameters": {"buffer-size": [1], "latency-chain-mode": list(modes)},
            },
            "planned_runs": len(modes),
            "attempted_runs": len(modes),
            "completed_runs": len(modes),
            "status": "complete",
            "status_reason": None,
            "conclusions_valid": True,
            "execution_time_sec": 0.4,
            "timestamp": "2026-11-27T07:00:00Z",
            "version": "0.65.0",
            "runs": runs,
        }

    def test_sweep_verification_single_hash_budget_and_mutations(self):
        sweep = self.make_sweep(("global-random", "random-box"))
        sorted_sweep = json.loads(json.dumps(sweep, sort_keys=True))
        verdict = verifier.verify(sorted_sweep, require_raw=True)
        self.assertTrue(verdict["artifact_consistent"])
        self.assertTrue(verdict["run_accepted"])
        self.assertEqual(verdict["artifact_kind"], "cpu_latency_sweep")

        with tempfile.NamedTemporaryFile() as bin_file:
            bin_file.write(b"latency-sweep-binary")
            bin_file.flush()
            digest = hashlib.sha256(b"latency-sweep-binary").hexdigest()
            for r in sweep["runs"]:
                r["result"]["build_manifest"]["binary_sha256"] = digest
            with mock.patch.object(common, "hash_binary", wraps=common.hash_binary) as mocked:
                v = verifier.verify(sweep, binary=Path(bin_file.name))
                self.assertEqual(v["checks"]["build"], "binary-bound")
                self.assertEqual(mocked.call_count, 1)

        single_work = verifier.verify(self.make_sweep(("global-random",)))["oracle_work_used"]
        two_run = self.make_sweep(("global-random", "random-box"))
        two_work = verifier.verify(two_run)["oracle_work_used"]
        self.assertGreater(two_work, single_work)
        self.assert_reason(
            two_run,
            "unsupported-work-limit",
            unsupported=True,
            budget=common.Budget(limit=single_work + 1),
        )

        sweep_mutations = [
            ("unsupported-sweep-schema", lambda e: e["configuration"].update(sweep_schema_version=2), True),
            ("sweep-parameter-value-invalid", lambda e: e["configuration"]["sweep_parameters"].update({"latency-chain-mode": ["bad-mode"]}), False),
            ("sweep-nested-output-file-mismatch", lambda e: e["runs"][0]["result"]["configuration"].update(output_file="out.json"), False),
            ("sweep-base-config-mismatch", lambda e: e["runs"][1]["result"]["build_manifest"].update(compiler="other"), False),
            ("sweep-invalid-execution-prefix", lambda e: e["runs"][0].update(status="failed", status_reason="err", result=None), False),
        ]
        for reason, mutate, unsup in sweep_mutations:
            with self.subTest(reason=reason):
                env = self.make_sweep(("global-random", "random-box"))
                mutate(env)
                self.assert_reason(env, reason, unsup)

    def test_sweep_resolved_sources_provenance_tamper_rejected(self):
        def mutate_all_runs(env, fn):
            for r in env["runs"]:
                fn(r["result"]["configuration"])

        def swap_buffer_and_chain_mode_to_locality(cfg):
            cfg["resolved_sources"]["buffer_size"] = "default"
            cfg["resolved_sources"]["latency_tlb_locality_kb"] = "sweep"

        def swap_chain_mode_and_stride(cfg):
            cfg["resolved_sources"]["latency_chain_mode"] = "default"
            cfg["resolved_sources"]["latency_stride_bytes"] = "sweep"

        def unswept_cache_marked_sweep(cfg):
            cfg["argv"] = self._strip_cli_flag(cfg["argv"], "-k")
            cfg["resolved_sources"]["cache_size"] = "sweep"

        def unswept_locality_marked_sweep(cfg):
            cfg["resolved_sources"]["latency_tlb_locality_kb"] = "sweep"

        def swept_chain_mode_marked_default(cfg):
            cfg["resolved_sources"]["latency_chain_mode"] = "default"

        def swept_buffer_marked_explicit_with_flag(cfg):
            cfg["argv"] = list(cfg["argv"]) + ["-b", "1"]
            cfg["resolved_sources"]["buffer_size"] = "explicit"

        cases = [
            ("swap_buffer_and_locality", swap_buffer_and_chain_mode_to_locality),
            ("swap_chain_mode_and_stride", swap_chain_mode_and_stride),
            ("unswept_cache_marked_sweep", unswept_cache_marked_sweep),
            ("unswept_locality_marked_sweep", unswept_locality_marked_sweep),
            ("swept_chain_mode_marked_default", swept_chain_mode_marked_default),
            ("swept_buffer_marked_explicit_with_flag", swept_buffer_marked_explicit_with_flag),
        ]
        with tempfile.TemporaryDirectory(prefix="cpu-lat-sweep-rs-") as directory:
            p = Path(directory) / "tamper.json"
            for name, cfg_mut in cases:
                with self.subTest(case=name):
                    env = self.make_sweep(("global-random", "random-box"))
                    mutate_all_runs(env, cfg_mut)
                    self.assert_reason(env, "resolved-sources-invalid", unsupported=False)
                    p.write_text(json.dumps(env))
                    res = subprocess.run(
                        [sys.executable, str(ROOT / "script-examples/verify_cpu_latency_result.py"), str(p)],
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(res.returncode, 1, res.stdout + res.stderr)
                    verdict = json.loads(res.stdout)
                    self.assertEqual(verdict["reason_code"], "resolved-sources-invalid")
                    self.assertFalse(verdict["run_accepted"])

    def test_locality_nodes_exceeding_max_chain_nodes_rejected_before_allocation(self):
        for mode in ("same-random-in-box", "random-box", "diff-random-in-box"):
            with self.subTest(mode=mode):
                with self.assertRaises(common.Rejected) as caught:
                    verifier.build_latency_chain(
                        424242,
                        verifier.TARGET_DOMAINS["custom"],
                        64,
                        mode,
                        10**12,
                        max_chain_nodes=1000,
                    )
                self.assertEqual(caught.exception.reason, "unsupported-chain-node-limit")
                self.assertTrue(caught.exception.unsupported)

        with tempfile.TemporaryDirectory(prefix="cpu-lat-loc-limit-") as directory:
            p = Path(directory) / "loc_limit.json"
            for mode in ("same-random-in-box", "random-box", "diff-random-in-box"):
                with self.subTest(cli_mode=mode):
                    doc = load_fixture()
                    doc["configuration"]["latency_chain_mode"] = mode
                    doc["configuration"]["latency_tlb_locality_kb"] = 262144
                    for t in ("custom", "main_memory"):
                        ci = doc["resolved_plan"]["chain_identities"][t]
                        ci["chain_mode"] = mode
                        ci["tlb_locality_kb"] = 262144
                    self.assert_reason(
                        doc,
                        "unsupported-chain-node-limit",
                        unsupported=True,
                        max_chain_nodes=1000,
                    )
                    p.write_text(json.dumps(doc))
                    res = subprocess.run(
                        [
                            sys.executable,
                            str(ROOT / "script-examples/verify_cpu_latency_result.py"),
                            str(p),
                            "--max-chain-nodes",
                            "1000",
                        ],
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(res.returncode, 2, res.stdout + res.stderr)
                    verdict = json.loads(res.stdout)
                    self.assertIsNone(verdict["artifact_consistent"])
                    self.assertEqual(verdict["reason_code"], "unsupported-chain-node-limit")

    def test_null_and_invalid_nested_containers_raise_typed_rejections(self):
        cases = [
            ("measurement-invalid", lambda d: d.update(measurements=None)),
            ("measurement-invalid", lambda d: d.update(measurements=[123])),
            (
                "validation-invalid",
                lambda d: d["calibration"]["attempts"]["custom"][0]["execution"].update(validation=None),
            ),
            (
                "timing-invalid",
                lambda d: d["calibration"]["attempts"]["custom"][0]["execution"].update(timing=None),
            ),
            (
                "validation-invalid",
                lambda d: d["measurements"][0]["execution"].update(validation=None),
            ),
            (
                "timing-invalid",
                lambda d: d["measurements"][0]["execution"].update(timing=None),
            ),
            (
                "final-pointer-mismatch",
                lambda d: d["measurements"][0].update(final_pointer_verification=None),
            ),
        ]
        for reason, mutate in cases:
            with self.subTest(reason=reason):
                doc = load_fixture()
                mutate(doc)
                self.assert_reason(doc, reason, unsupported=False)

        loc_cases = [
            (
                "timing-invalid",
                lambda d: d["locality_comparison"]["rounds"][0]["locality_16k"]["execution"].update(timing=None),
            ),
            (
                "final-pointer-mismatch",
                lambda d: d["locality_comparison"]["rounds"][0]["locality_16k"].update(final_pointer_verification=None),
            ),
        ]
        for reason, mutate in loc_cases:
            with self.subTest(loc_reason=reason):
                doc = build_locality_enabled_doc()
                mutate(doc)
                self.assert_reason(doc, reason, unsupported=False)

        with tempfile.TemporaryDirectory(prefix="cpu-lat-null-containers-") as directory:
            p = Path(directory) / "bad.json"
            for expected_reason, mutate in (cases[0], cases[1], cases[4], cases[6]):
                doc = load_fixture()
                mutate(doc)
                p.write_text(json.dumps(doc))
                res = subprocess.run(
                    [sys.executable, str(ROOT / "script-examples/verify_cpu_latency_result.py"), str(p)],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(res.returncode, 1, res.stdout + res.stderr)
                verdict = json.loads(res.stdout)
                self.assertEqual(verdict["reason_code"], expected_reason)
                self.assertFalse(verdict["run_accepted"])

    def test_compare_locality_complete_requires_all_measured_rounds_and_non_null_aggregates(self):
        empty_rounds = build_locality_enabled_doc()
        empty_rounds["locality_comparison"] = {"rounds": []}
        empty_rounds["aggregates"]["locality_comparison"] = {
            "locality_16k_latency_ns": None,
            "global_random_latency_ns": None,
            "locality_latency_delta_ns": None,
        }
        self.assert_reason(empty_rounds, "locality-rounds-invalid", unsupported=False)

        with tempfile.TemporaryDirectory(prefix="cpu-lat-empty-loc-") as directory:
            p = Path(directory) / "empty_loc.json"
            p.write_text(json.dumps(empty_rounds))
            res = subprocess.run(
                [sys.executable, str(ROOT / "script-examples/verify_cpu_latency_result.py"), str(p)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(res.returncode, 1, res.stdout + res.stderr)
            verdict = json.loads(res.stdout)
            self.assertEqual(verdict["reason_code"], "locality-rounds-invalid")
            self.assertFalse(verdict["run_accepted"])

        partial_rounds = build_locality_enabled_doc()
        partial_rounds["locality_comparison"]["rounds"] = partial_rounds["locality_comparison"]["rounds"][:1]
        self.assert_reason(partial_rounds, "locality-rounds-invalid", unsupported=False)

        failed_sub = build_locality_enabled_doc()
        failed_sub["locality_comparison"]["rounds"][0]["locality_16k"].update(
            status="failed", attempted=False, latency_ns=None
        )
        self.assert_reason(failed_sub, "locality-rounds-invalid", unsupported=False)

        null_agg = build_locality_enabled_doc()
        null_agg["aggregates"]["locality_comparison"]["locality_latency_delta_ns"] = None
        self.assert_reason(null_agg, "locality-delta-mismatch", unsupported=False)

    @staticmethod
    def _make_preflight_failed_doc(reason_code):
        doc = load_fixture()
        doc["resolved_plan"].update(
            valid=False,
            reason_code=reason_code,
            plan_identity="cpu-latency-work-plan-v1|invalid",
            targets={},
            resources={},
            chain_identities={},
            target_plans=[],
            frozen_plan_refs={},
        )
        doc["memory_budget"].update(
            resource_rounding_bytes="0",
            transient_peak_bytes="0",
            setup_peak_bytes="0",
            runtime_peak_bytes="0",
            known_owned_peak_bytes="0",
            valid=False,
            reason_code="memory-budget-overflow",
        )
        doc["calibration"]["attempts"] = {}
        doc["measurements"] = []
        doc["aggregates"]["targets"] = {}
        doc["status"] = "failed"
        doc["reason_code"] = reason_code
        doc["results_complete"] = False
        doc["run_accepted"] = False
        doc["target_order_balance_complete"] = False
        doc["counters"].update(
            attempted_loops=0,
            completed_loops=0,
            planned_measurements=0,
            attempted_measurements=0,
            terminal_measurements=0,
            measured_measurements=0,
            planned_accesses="0",
            completed_accesses="0",
        )
        for rec in doc["loop_records"]:
            rec["planned_order"] = []
            rec["realized_order"] = []
            rec["realized_order_count"] = 0
            rec["measurement_indexes"] = []
        return doc

    def test_preflight_invalid_plan_locality_and_compare_locality_consistent_not_accepted(self):
        with tempfile.TemporaryDirectory(prefix="cpu-lat-preflight-") as directory:
            d = Path(directory)

            # 1. !work_plan.valid due to locality-not-page-multiple (20 KiB on 16 KiB pages)
            loc_fail = self._make_preflight_failed_doc("locality-not-page-multiple")
            loc_fail["configuration"]["latency_tlb_locality_kb"] = 20
            loc_fail["configuration"]["argv"].extend(["-l", "20"])
            loc_fail["configuration"]["resolved_sources"]["latency_tlb_locality_kb"] = "explicit"
            v_loc = verifier.verify(loc_fail)
            self.assertTrue(v_loc["artifact_consistent"])
            self.assertFalse(v_loc["run_accepted"])
            self.assertEqual(v_loc["reason_code"], "run-not-accepted")

            p_loc = d / "loc_fail.json"
            p_loc.write_text(json.dumps(loc_fail))
            res_loc = subprocess.run(
                [sys.executable, str(ROOT / "script-examples/verify_cpu_latency_result.py"), str(p_loc)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(res_loc.returncode, 1, res_loc.stdout + res_loc.stderr)
            parsed_loc = json.loads(res_loc.stdout)
            self.assertTrue(parsed_loc["artifact_consistent"])
            self.assertFalse(parsed_loc["run_accepted"])
            self.assertEqual(parsed_loc["reason_code"], "run-not-accepted")

            # Spoofed locality-not-page-multiple when locality IS page-aligned must still reject
            spoof_loc = self._make_preflight_failed_doc("locality-not-page-multiple")
            self.assert_reason(spoof_loc, "plan-invalid")

            # 2. !work_plan.valid due to compare-locality-invalid (compare_locality=True, buffer_size_mb=0)
            cmp_fail = self._make_preflight_failed_doc("compare-locality-invalid")
            cmp_fail["configuration"]["compare_locality"] = True
            cmp_fail["configuration"]["buffer_size_mb"] = 0
            argv = list(cmp_fail["configuration"]["argv"])
            idx = argv.index("-b")
            argv[idx + 1] = "0"
            argv.append("--compare-locality")
            cmp_fail["configuration"]["argv"] = argv
            cmp_fail["locality_comparison"] = {"rounds": []}
            cmp_fail["aggregates"]["locality_comparison"] = {
                "locality_16k_latency_ns": None,
                "global_random_latency_ns": None,
                "locality_latency_delta_ns": None,
            }
            v_cmp = verifier.verify(cmp_fail)
            self.assertTrue(v_cmp["artifact_consistent"])
            self.assertFalse(v_cmp["run_accepted"])
            self.assertEqual(v_cmp["reason_code"], "run-not-accepted")

            p_cmp = d / "cmp_fail.json"
            p_cmp.write_text(json.dumps(cmp_fail))
            res_cmp = subprocess.run(
                [sys.executable, str(ROOT / "script-examples/verify_cpu_latency_result.py"), str(p_cmp)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(res_cmp.returncode, 1, res_cmp.stdout + res_cmp.stderr)
            parsed_cmp = json.loads(res_cmp.stdout)
            self.assertTrue(parsed_cmp["artifact_consistent"])
            self.assertFalse(parsed_cmp["run_accepted"])
            self.assertEqual(parsed_cmp["reason_code"], "run-not-accepted")

            # Spoofed rp_valid=True with compare_locality=True, buffer_size_mb=0 must reject as configuration-invalid
            forged_cmp = load_fixture()
            forged_cmp["configuration"]["compare_locality"] = True
            forged_cmp["configuration"]["buffer_size_mb"] = 0
            self.assert_reason(forged_cmp, "configuration-invalid")

    @staticmethod
    def _make_auto_calibrated_doc(elapsed_by_target=None):
        doc = load_fixture()
        cfg = doc["configuration"]
        argv = list(cfg["argv"])
        idx = argv.index("-i")
        del argv[idx : idx + 2]
        cfg["argv"] = argv
        cfg["iterations"] = None
        cfg["work_policy"] = "automatic_calibration"
        cfg["resolved_sources"]["iterations"] = "auto"

        plans = doc["resolved_plan"]["target_plans"]
        for p in plans:
            p["explicit_iterations"] = False

        elapsed_by_target = elapsed_by_target or {}
        for row in doc["measurements"]:
            if row["status"] != "measured":
                continue
            target_sec = elapsed_by_target.get(row["target"], 0.150)
            raw = row["execution"]["timing"]["cpu_raw"]
            numer = raw["timebase_numer"]
            denom = raw["timebase_denom"]
            dt = round(target_sec * 1e9 * denom / numer)
            start = int(raw["start_ticks"])
            raw["stop_ticks"] = str(start + dt)
            raw["delta_ticks"] = str(dt)
            elapsed = dt * numer / denom / 1e9
            row["execution"]["timing"]["elapsed_seconds"] = elapsed
            accesses = plans[row["plan_ref"]]["access_count"]
            row["latency_ns"] = (elapsed * 1e9) / float(accesses)

        rows = doc["measurements"]
        for tname, a in doc["aggregates"]["targets"].items():
            vals = [r["latency_ns"] for r in rows if r["target"] == tname and r["status"] == "measured"]
            st = common.compute_statistics(vals)
            a["value"] = st["median"]
            a["statistics"] = st
            a["quality"]["cv_warning"] = False
        return doc

    def test_resolved_sources_two_way_quality_duration_window_and_madvise_errno(self):
        # (2a) Two-way check_resolved_sources
        no_sweep = load_fixture()
        no_sweep["configuration"]["resolved_sources"]["latency_chain_mode"] = "sweep"
        self.assert_reason(no_sweep, "resolved-sources-invalid")

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
        swept_in_argv["configuration"]["argv"].extend(["-S", "latency-chain-mode=auto"])
        swept_in_argv["configuration"]["resolved_sources"]["latency_chain_mode"] = "sweep"
        self.assertTrue(verifier.verify(swept_in_argv, require_raw=True)["run_accepted"])

        # (2b) Cross-checked auto-calibration duration warnings ([0.100, 0.300])
        within_doc = self._make_auto_calibrated_doc({"custom": 0.100, "main_memory": 0.300})
        within_doc["quality_warnings"] = []
        self.assertTrue(verifier.verify(within_doc, require_raw=True)["run_accepted"])

        fabricated = copy.deepcopy(within_doc)
        fabricated["quality_warnings"] = ["main_memory-duration-above-target-window"]
        self.assert_reason(fabricated, "quality-warnings-invalid")

        out_of_window = self._make_auto_calibrated_doc({"custom": 0.050, "main_memory": 0.350})
        out_of_window["quality_warnings"] = []
        self.assert_reason(out_of_window, "quality-warnings-invalid")

        exact_warn = copy.deepcopy(out_of_window)
        exact_warn["quality_warnings"] = [
            "custom-duration-below-target-window",
            "main_memory-duration-above-target-window",
        ]
        self.assertTrue(verifier.verify(exact_warn, require_raw=True)["run_accepted"])

        # (2c) check_buffer_evidence madvise_applied=False requires madvise_errno != 0
        bad_madv = load_fixture()
        bad_madv["resolved_plan"]["resources"]["main_memory"]["buffer"]["madvise_applied"] = False
        self.assert_reason(bad_madv, "resource-evidence-invalid")


if __name__ == "__main__":
    unittest.main()

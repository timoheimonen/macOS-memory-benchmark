#!/usr/bin/env python3
# Copyright 2026 Timo Heimonen
# SPDX-License-Identifier: GPL-3.0-or-later

"""Entry-path tests for the bundled CPU memory script examples."""

import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = REPO_ROOT / "script-examples"
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"
CPU_BANDWIDTH_FIXTURE = FIXTURE_DIR / "cpu-bandwidth-schema-v1-complete-current.json"
CPU_LATENCY_FIXTURE = FIXTURE_DIR / "cpu-latency-schema-v1-complete-current.json"


def copy_script(root: Path, filename: str) -> tuple[Path, Path]:
    """Copy one example in isolation so undeclared helper dependencies fail."""

    copied_script_dir = root / "script-examples"
    copied_script_dir.mkdir(parents=True, exist_ok=True)
    script = copied_script_dir / filename
    shutil.copy2(SCRIPT_DIR / filename, script)
    return copied_script_dir, script


def load_plotter(path: Path, module_name: str):
    """Load a plotter's data boundary without matplotlib or a display."""

    pyplot = types.ModuleType("matplotlib.pyplot")
    matplotlib = types.ModuleType("matplotlib")
    matplotlib.pyplot = pyplot
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load plotter: {path}")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"matplotlib": matplotlib, "matplotlib.pyplot": pyplot}):
        spec.loader.exec_module(module)
    return module


def make_full_latency_doc(*, cpu_name: str = "Apple M4", version: str = "0.66.0") -> dict:
    """Build a complete cpu_latency schema-1 document with L1, L2, and locality comparison."""

    doc = json.loads(CPU_LATENCY_FIXTURE.read_text(encoding="utf-8"))
    doc["software"]["version"] = version
    doc["environment"]["processor_name"] = cpu_name
    doc["configuration"]["custom_cache_size_kb"] = None
    doc["configuration"]["compare_locality"] = True

    doc["aggregates"]["targets"] = {
        "l1": {
            "accepted_measurement_ids": [0, 3, 6],
            "value": 1.15,
            "statistics": {
                "sample_count": 3,
                "average": 1.20,
                "min": 1.10,
                "max": 1.35,
                "median": 1.15,
                "p90": 1.31,
                "p95": 1.33,
                "p99": 1.346,
                "stddev": 0.13228756555322954,
                "coefficient_of_variation_pct": 11.02,
                "median_absolute_deviation": 0.05,
            },
            "quality": {
                "repeatability_assessment_available": True,
                "run_complete": True,
                "cv_warning_threshold_pct": 7.5,
                "cv_warning": False,
            },
            "pooled_sample_distribution": None,
        },
        "l2": {
            "accepted_measurement_ids": [1, 4, 7],
            "value": 4.20,
            "statistics": {
                "sample_count": 3,
                "average": 4.30,
                "min": 4.00,
                "max": 4.70,
                "median": 4.20,
                "p90": 4.60,
                "p95": 4.65,
                "p99": 4.69,
                "stddev": 0.36055512754639896,
                "coefficient_of_variation_pct": 8.38,
                "median_absolute_deviation": 0.20,
            },
            "quality": {
                "repeatability_assessment_available": True,
                "run_complete": True,
                "cv_warning_threshold_pct": 7.5,
                "cv_warning": False,
            },
            "pooled_sample_distribution": None,
        },
        "main_memory": doc["aggregates"]["targets"]["main_memory"],
    }

    # Three measured locality rounds:
    # 16K:    [80.0, 82.0, 87.0] -> median 82.0, avg 83.0, p90 86.0
    # Global: [90.0, 94.0, 98.0] -> median 94.0, avg 94.0, p90 97.2
    # Paired deltas: [10.0, 12.0, 11.0] -> sample stddev = 1.0
    doc["locality_comparison"] = {
        "rounds": [
            {
                "loop_index": 0,
                "round_index": 0,
                "locality_16k": {"status": "measured", "reason_code": "measured", "attempted": True, "completed_accesses": 1024, "latency_ns": 80.0},
                "locality_global": {"status": "measured", "reason_code": "measured", "attempted": True, "completed_accesses": 1024, "latency_ns": 90.0},
            },
            {
                "loop_index": 0,
                "round_index": 1,
                "locality_16k": {"status": "measured", "reason_code": "measured", "attempted": True, "completed_accesses": 1024, "latency_ns": 82.0},
                "locality_global": {"status": "measured", "reason_code": "measured", "attempted": True, "completed_accesses": 1024, "latency_ns": 94.0},
            },
            {
                "loop_index": 0,
                "round_index": 2,
                "locality_16k": {"status": "measured", "reason_code": "measured", "attempted": True, "completed_accesses": 1024, "latency_ns": 87.0},
                "locality_global": {"status": "measured", "reason_code": "measured", "attempted": True, "completed_accesses": 1024, "latency_ns": 98.0},
            },
        ]
    }
    doc["aggregates"]["locality_comparison"] = {
        "locality_16k_latency_ns": 82.0,
        "global_random_latency_ns": 94.0,
        "locality_latency_delta_ns": 12.0,
    }
    return doc


class PlotterEntryPathTest(unittest.TestCase):
    def test_comparison_loader_accepts_supported_bandwidth_and_latency_results(self):
        full_latency = make_full_latency_doc()
        with tempfile.TemporaryDirectory(prefix="script-example-plotter-") as temporary:
            root = Path(temporary)
            _, script = copy_script(root, "plot_M4vsM5_benchmark_comparison.py")
            lat_path = root / "full-latency.json"
            lat_path.write_text(json.dumps(full_latency), encoding="utf-8")

            plotter = load_plotter(script, "current_comparison_plotter")
            bw_median = plotter.load_bandwidth_data(CPU_BANDWIDTH_FIXTURE, "median")
            bw_avg = plotter.load_bandwidth_data(CPU_BANDWIDTH_FIXTURE, "average")
            lat_median = plotter.load_latency_data(lat_path, "median")
            lat_avg = plotter.load_latency_data(lat_path, "average")
            machine_median = plotter.load_machine_data(CPU_BANDWIDTH_FIXTURE, lat_path, "median")
            machine_avg = plotter.load_machine_data(CPU_BANDWIDTH_FIXTURE, lat_path, "average")

        self.assertEqual(bw_median["cpu_name"], "Apple M4")
        self.assertEqual(bw_median["version"], "0.66.0")
        self.assertAlmostEqual(bw_median["bw_copy"], 100.663296)
        self.assertAlmostEqual(bw_median["bw_read"], 100.663296)
        self.assertAlmostEqual(bw_median["bw_write"], 100.663296)
        self.assertAlmostEqual(bw_avg["bw_copy"], 100.05752524960754)

        self.assertAlmostEqual(lat_median["l1_lat"], 1.15)
        self.assertAlmostEqual(lat_median["l2_lat"], 4.20)
        self.assertAlmostEqual(lat_median["locality_16k"], 82.0)
        self.assertAlmostEqual(lat_median["global_random"], 94.0)
        self.assertAlmostEqual(lat_avg["l1_lat"], 1.20)
        self.assertAlmostEqual(lat_avg["locality_16k"], 83.0)
        self.assertAlmostEqual(lat_avg["global_random"], 94.0)

        self.assertEqual(machine_median["cpu_name"], "Apple M4")
        self.assertEqual(machine_median["version"], "0.66.0")
        self.assertAlmostEqual(machine_median["bw_read"], 100.663296)
        self.assertAlmostEqual(machine_median["global_random"], 94.0)
        self.assertAlmostEqual(machine_avg["l2_lat"], 4.30)

    def test_comparison_cli_accepts_primary_and_alias_flags_with_median_default(self):
        with tempfile.TemporaryDirectory(prefix="script-example-cmp-cli-") as temporary:
            _, script = copy_script(Path(temporary), "plot_M4vsM5_benchmark_comparison.py")
            plotter = load_plotter(script, "cli_comparison_plotter")

            with mock.patch.object(
                sys,
                "argv",
                [
                    str(script),
                    "--m4-bandwidth-file", "m4_bw.json",
                    "--m4-latency-file", "m4_lat.json",
                    "--m5-bandwidth-file", "m5_bw.json",
                    "--m5-latency-file", "m5_lat.json",
                ],
            ):
                args_primary = plotter.parse_args()
            self.assertEqual(args_primary.m4_bandwidth_file, "m4_bw.json")
            self.assertEqual(args_primary.m4_latency_file, "m4_lat.json")
            self.assertEqual(args_primary.m5_bandwidth_file, "m5_bw.json")
            self.assertEqual(args_primary.m5_latency_file, "m5_lat.json")
            self.assertEqual(args_primary.metric, "median")

            with mock.patch.object(
                sys,
                "argv",
                [
                    str(script),
                    "--m4-bandwidth", "a.json",
                    "--m4-latency", "b.json",
                    "--m5-bandwidth", "c.json",
                    "--m5-latency", "d.json",
                    "--metric", "p99",
                ],
            ):
                args_alias = plotter.parse_args()
            self.assertEqual(args_alias.m4_bandwidth_file, "a.json")
            self.assertEqual(args_alias.m4_latency_file, "b.json")
            self.assertEqual(args_alias.m5_bandwidth_file, "c.json")
            self.assertEqual(args_alias.m5_latency_file, "d.json")
            self.assertEqual(args_alias.metric, "p99")

    def test_comparison_loader_rejects_cpu_mismatch_and_missing_locality(self):
        mismatched_lat = make_full_latency_doc(cpu_name="Apple M5")
        with tempfile.TemporaryDirectory(prefix="script-example-cmp-err-") as temporary:
            root = Path(temporary)
            _, script = copy_script(root, "plot_M4vsM5_benchmark_comparison.py")
            mismatched_path = root / "mismatched-lat.json"
            mismatched_path.write_text(json.dumps(mismatched_lat), encoding="utf-8")
            plotter = load_plotter(script, "err_comparison_plotter")

            with self.assertRaisesRegex(RuntimeError, "CPU mismatch"):
                plotter.load_machine_data(CPU_BANDWIDTH_FIXTURE, mismatched_path, "median")
            with self.assertRaises(RuntimeError):
                plotter.load_latency_data(CPU_LATENCY_FIXTURE, "median")

    def test_hierarchy_loader_accepts_supported_full_result_and_metrics(self):
        full_latency = make_full_latency_doc()
        with tempfile.TemporaryDirectory(prefix="script-example-plotter-") as temporary:
            root = Path(temporary)
            _, script = copy_script(root, "plot_benchmark-memory-latency-hierarchy.py")
            lat_path = root / "full-latency.json"
            lat_path.write_text(json.dumps(full_latency), encoding="utf-8")
            plotter = load_plotter(script, "current_hierarchy_plotter")

            with mock.patch.object(sys, "argv", [str(script), "-f", str(lat_path)]):
                self.assertEqual(plotter.parse_args().metric, "median")

            med = plotter.load_latency_data(lat_path, "median")
            avg = plotter.load_latency_data(lat_path, "average")
            p90 = plotter.load_latency_data(lat_path, "p90")
            sd = plotter.load_latency_data(lat_path, "stddev")

        cpu_name, categories, latencies, locality_delta, version = med
        self.assertEqual(cpu_name, "Apple M4")
        self.assertEqual(len(categories), 4)
        self.assertEqual(latencies, [1.15, 4.20, 82.0, 94.0])
        self.assertAlmostEqual(locality_delta, 12.0)
        self.assertEqual(version, "0.66.0")

        self.assertEqual(avg[2], [1.20, 4.30, 83.0, 94.0])
        self.assertAlmostEqual(avg[3], 11.0)

        self.assertAlmostEqual(p90[2][0], 1.31)
        self.assertAlmostEqual(p90[2][1], 4.60)
        self.assertAlmostEqual(p90[2][2], 86.0)
        self.assertAlmostEqual(p90[2][3], 97.2)
        self.assertAlmostEqual(p90[3], 11.2)

        # Paired per-round deltas [10.0, 12.0, 11.0] have sample stddev 1.0
        self.assertAlmostEqual(sd[2][2], math.sqrt(((80.0 - 83.0) ** 2 + (82.0 - 83.0) ** 2 + (87.0 - 83.0) ** 2) / 2.0))
        self.assertAlmostEqual(sd[2][3], 4.0)
        self.assertAlmostEqual(sd[3], 1.0)

    def test_hierarchy_loader_accepts_supported_console_statistics_and_rejects_non_median(self):
        console_output = """\
CPU Latency Mode (schema v1)
  CPU: Apple M4 (P=4, E=6, Logical=10, Page=16384 B)
  Work Policy: explicit_fixed_work (cycles=4), Loops: 2, Samples: 4
  Chain: mode=random-box (requested=auto), stride=256 B, locality=1024 KB, compare_locality=enabled
  Base Seed: 42 (user), Non-cacheable hint: disabled
  Target [L1 Cache]: 131072 bytes (512 nodes)
  Target [L2 Cache]: 16777216 bytes (65536 nodes)
  Target [Main Memory]: 1048576 bytes (4096 nodes)

--- CPU Latency Results (complete, 6/6 measured) ---
  L1 Cache    : P50=1.25 ns (min=1.10, max=1.40, CV=10.0%, MAD=0.10, n=2) | PooledSamples P50=1.20 ns (P90=1.35, P99=1.40, n=8)
  L2 Cache    : P50=4.50 ns (min=4.20, max=4.80, CV=5.0%, MAD=0.20, n=2) | PooledSamples P50=4.45 ns (P90=4.70, P99=4.80, n=8)
  Main Memory : P50=21.73 ns (min=14.83, max=28.64, CV=44.9%, MAD=6.91, n=2) | PooledSamples P50=4.88 ns (P90=6.06, P99=6.45, n=8)
  Locality Comparison: 16KB P50=80.00 ns, Global P50=105.25 ns, Delta=25.25 ns
"""
        with tempfile.TemporaryDirectory(prefix="script-example-text-") as temporary:
            root = Path(temporary)
            _, script = copy_script(root, "plot_benchmark-memory-latency-hierarchy.py")
            input_path = root / "cpu_latency_console.txt"
            input_path.write_text(console_output, encoding="utf-8")
            plotter = load_plotter(script, "current_text_hierarchy_plotter")

            cpu_name, _, latencies, locality_delta, version = plotter.load_latency_data(
                input_path, "median"
            )
            with self.assertRaisesRegex(RuntimeError, "P50"):
                plotter.load_latency_data(input_path, "p99")
            with self.assertRaisesRegex(RuntimeError, "P50"):
                plotter.load_latency_data(input_path, "average")

        self.assertEqual(cpu_name, "Apple M4")
        self.assertEqual(latencies, [1.25, 4.50, 80.00, 105.25])
        self.assertEqual(locality_delta, 25.25)
        self.assertIsNone(version)

    def test_plotter_loaders_enforce_supported_contract(self):
        bw_supported = json.loads(CPU_BANDWIDTH_FIXTURE.read_text(encoding="utf-8"))
        lat_supported = make_full_latency_doc()

        def make_mutations(doc: dict, wrong_mode_name: str):
            accepted = json.loads(json.dumps(doc))
            accepted["software"]["version"] = "0.67.0"

            wrong_schema = json.loads(json.dumps(doc))
            wrong_schema["schema_version"] = 2

            bool_schema = json.loads(json.dumps(doc))
            bool_schema["schema_version"] = True

            wrong_mode = json.loads(json.dumps(doc))
            wrong_mode["mode"] = wrong_mode_name

            wrong_methodology = json.loads(json.dumps(doc))
            wrong_methodology["methodology_version"] = "unexpected-methodology-v0"

            not_accepted = json.loads(json.dumps(doc))
            not_accepted["run_accepted"] = False

            incomplete = json.loads(json.dumps(doc))
            incomplete["results_complete"] = False

            non_complete_status = json.loads(json.dumps(doc))
            non_complete_status["status"] = "partial"

            wrong_shape = json.loads(json.dumps(doc))
            wrong_shape["configuration"]["output_file"] = None

            missing_provenance = json.loads(json.dumps(doc))
            missing_provenance["software"]["version"] = ""

            return accepted, {
                "schema 2": wrong_schema,
                "boolean schema": bool_schema,
                "wrong mode": wrong_mode,
                "wrong methodology": wrong_methodology,
                "run_accepted false": not_accepted,
                "results_complete false": incomplete,
                "status partial": non_complete_status,
                "non-string output_file": wrong_shape,
                "empty provenance version": missing_provenance,
            }

        with tempfile.TemporaryDirectory(prefix="script-example-contract-") as temporary:
            root = Path(temporary)
            _, cmp_script = copy_script(root / "cmp", "plot_M4vsM5_benchmark_comparison.py")
            _, hier_script = copy_script(root / "hier", "plot_benchmark-memory-latency-hierarchy.py")
            cmp_plotter = load_plotter(cmp_script, "contract_comparison")
            hier_plotter = load_plotter(hier_script, "contract_hierarchy")

            bw_accepted, bw_invalid = make_mutations(bw_supported, "cpu_latency")
            lat_accepted, lat_invalid = make_mutations(lat_supported, "cpu_bandwidth")

            bw_acc_path = root / "bw-accepted.json"
            bw_acc_path.write_text(json.dumps(bw_accepted), encoding="utf-8")
            lat_acc_path = root / "lat-accepted.json"
            lat_acc_path.write_text(json.dumps(lat_accepted), encoding="utf-8")

            self.assertEqual(cmp_plotter.load_bandwidth_data(bw_acc_path, "median")["version"], "0.67.0")
            self.assertEqual(cmp_plotter.load_latency_data(lat_acc_path, "median")["version"], "0.67.0")
            self.assertEqual(hier_plotter.load_latency_data(lat_acc_path, "median")[-1], "0.67.0")

            for name, data in bw_invalid.items():
                path = root / f"bw-{name.replace(' ', '-')}.json"
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.subTest(loader="comparison-bandwidth", input=name):
                    with self.assertRaisesRegex(RuntimeError, "complete supported CPU bandwidth schema-1"):
                        cmp_plotter.load_bandwidth_data(path, "median")

            for name, data in lat_invalid.items():
                path = root / f"lat-{name.replace(' ', '-')}.json"
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.subTest(loader="comparison-latency", input=name):
                    with self.assertRaisesRegex(RuntimeError, "complete supported CPU latency schema-1"):
                        cmp_plotter.load_latency_data(path, "median")
                with self.subTest(loader="hierarchy-latency", input=name):
                    with self.assertRaisesRegex(RuntimeError, "complete supported CPU latency schema-1"):
                        hier_plotter.load_latency_data(path, "median")

            missing_locality = json.loads(json.dumps(lat_supported))
            missing_locality["aggregates"]["locality_comparison"] = None
            missing_loc_path = root / "lat-missing-locality.json"
            missing_loc_path.write_text(json.dumps(missing_locality), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "--compare-locality"):
                hier_plotter.load_latency_data(missing_loc_path, "median")


class ShellWorkflowEntryPathTest(unittest.TestCase):
    def make_fake_benchmark(self, root: Path) -> Path:
        fake = root / "fake-memory-benchmark"
        fake.write_text(
            """#!/bin/bash
if [ "${1:-}" = "-h" ]; then
  echo "--cpu-latency"
  echo "--latency-stride-bytes"
  exit 0
fi
saw_cpu_latency=0
output_file=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --cpu-latency)
      saw_cpu_latency=1
      shift
      ;;
    --non-cacheable)
      shift
      ;;
    --buffer-size|--cache-size|--latency-samples|--count|--latency-tlb-locality-kb|--latency-stride-bytes)
      if [ "$#" -lt 2 ]; then
        echo "missing value for $1" >&2
        exit 1
      fi
      shift 2
      ;;
    --output)
      output_file="$2"
      shift 2
      ;;
    *)
      echo "unexpected argument: $1" >&2
      exit 1
      ;;
  esac
done
if [ "${saw_cpu_latency}" -ne 1 ]; then
  echo "missing --cpu-latency" >&2
  exit 1
fi
if [ -z "${output_file}" ]; then
  echo "missing --output" >&2
  exit 1
fi
/bin/cp "${SCRIPT_EXAMPLE_FIXTURE}" "${output_file}"
""",
            encoding="utf-8",
        )
        fake.chmod(0o755)
        return fake

    def python_only_path(self, root: Path) -> Path:
        isolated_bin = root / "path-without-jq"
        isolated_bin.mkdir()
        for command in ("cat", "dirname", "mkdir", "rm"):
            resolved = shutil.which(command)
            if resolved is None:
                self.fail(f"required shell utility not found: {command}")
            (isolated_bin / command).symlink_to(resolved)
        (isolated_bin / "python3").symlink_to(sys.executable)
        return isolated_bin

    def run_workflow(
        self,
        root: Path,
        script_name: str,
        fixture: Path,
        *,
        force_python: bool = False,
    ) -> tuple[Path, subprocess.CompletedProcess[str]]:
        copied_script_dir, script = copy_script(root, script_name)
        environment = os.environ.copy()
        environment["BENCHMARK_CMD"] = str(self.make_fake_benchmark(root))
        environment["SCRIPT_EXAMPLE_FIXTURE"] = str(fixture)
        if force_python:
            environment["PATH"] = str(self.python_only_path(root))

        completed = subprocess.run(
            ["/bin/bash", str(script)],
            cwd=root,
            env=environment,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        return copied_script_dir, completed

    def assert_success(self, script_name: str, completed: subprocess.CompletedProcess[str]):
        if completed.returncode != 0:
            self.fail(
                f"{script_name} failed with {completed.returncode}\n"
                f"stdout tail:\n{completed.stdout[-4000:]}\n"
                f"stderr tail:\n{completed.stderr[-4000:]}"
            )

    def test_latency_workflow_accepts_supported_result(self):
        for parser, force_python in (("Python", True), ("jq", False)):
            with self.subTest(parser=parser):
                if not force_python and not shutil.which("jq"):
                    self.skipTest("jq is optional and is not installed")
                with tempfile.TemporaryDirectory(prefix="script-example-latency-") as temporary:
                    script_dir, completed = self.run_workflow(
                        Path(temporary),
                        "latency_test_script.sh",
                        CPU_LATENCY_FIXTURE,
                        force_python=force_python,
                    )
                    self.assert_success("latency_test_script.sh", completed)
                    final_output = (script_dir / "final_output.txt").read_text(encoding="utf-8")

                self.assertIn(f"Using {parser} to extract latency sample statistics", completed.stdout)
                self.assertEqual(final_output.count("TLB Locality:"), 120)
                self.assertIn('"average": 8.463541666666666', final_output)
                self.assertIn('"median": 8.463541666666666', final_output)

    def test_stride_tlb_workflow_accepts_supported_result(self):
        with tempfile.TemporaryDirectory(prefix="script-example-stride-tlb-") as temporary:
            script_dir, completed = self.run_workflow(
                Path(temporary), "latency_test_script_stride_tlb.sh", CPU_LATENCY_FIXTURE
            )
            self.assert_success("latency_test_script_stride_tlb.sh", completed)
            summaries = {
                path.resolve()
                for path in (script_dir / "latency-results").glob("*/latency_summary.csv")
            }
            self.assertEqual(len(summaries), 1)
            with next(iter(summaries)).open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 315)
        self.assertEqual(rows[0]["average"], "8.463541666666666")
        self.assertEqual(rows[0]["median"], "8.463541666666666")
        self.assertEqual(rows[0]["chain_node_count"], "64")
        self.assertEqual(rows[0]["page_size_bytes"], "16384")

    def test_shell_workflows_reject_incompatible_contracts(self):
        supported = json.loads(CPU_LATENCY_FIXTURE.read_text(encoding="utf-8"))
        wrong_schema = json.loads(json.dumps(supported))
        wrong_schema["schema_version"] = 2
        wrong_mode = json.loads(json.dumps(supported))
        wrong_mode["mode"] = "cpu_bandwidth"
        wrong_methodology = json.loads(json.dumps(supported))
        wrong_methodology["methodology_version"] = "unexpected-methodology-v0"
        not_accepted = json.loads(json.dumps(supported))
        not_accepted["run_accepted"] = False

        cases = {
            "wrong_schema": wrong_schema,
            "wrong_mode": wrong_mode,
            "wrong_methodology": wrong_methodology,
            "not_accepted": not_accepted,
        }

        with tempfile.TemporaryDirectory(prefix="script-example-shell-contract-") as temporary:
            root = Path(temporary)
            for name, payload in cases.items():
                case_root = root / name
                case_root.mkdir(parents=True, exist_ok=True)
                bad_fixture = case_root / "bad.json"
                bad_fixture.write_text(json.dumps(payload), encoding="utf-8")

                for parser, force_python in (("Python", True), ("jq", False)):
                    if not force_python and not shutil.which("jq"):
                        continue
                    with self.subTest(script="latency_test_script.sh", parser=parser, case=name):
                        latency_dir, latency = self.run_workflow(
                            case_root / f"latency-{parser}",
                            "latency_test_script.sh",
                            bad_fixture,
                            force_python=force_python,
                        )
                        final_output = (latency_dir / "final_output.txt").read_text(encoding="utf-8")
                        self.assertNotEqual(latency.returncode, 0)
                        self.assertNotIn("TLB Locality:", final_output)

                with self.subTest(script="latency_test_script_stride_tlb.sh", case=name):
                    stride_dir, stride = self.run_workflow(
                        case_root / "stride",
                        "latency_test_script_stride_tlb.sh",
                        bad_fixture,
                    )
                    summaries = {
                        path.resolve()
                        for path in (stride_dir / "latency-results").glob("*/latency_summary.csv")
                    }
                    self.assertEqual(len(summaries), 1)
                    with next(iter(summaries)).open(newline="", encoding="utf-8") as handle:
                        rows = list(csv.DictReader(handle))
                    self.assertNotEqual(stride.returncode, 0)
                    self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)

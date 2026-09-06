#!/usr/bin/env python3
# Copyright 2026 Timo Heimonen
# SPDX-License-Identifier: GPL-3.0-or-later

"""Entry-path tests for the bundled standard-memory script examples."""

import csv
import importlib.util
import json
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
CURRENT_FULL_FIXTURE = FIXTURE_DIR / "standard-schema-v3-complete-current.json"
CURRENT_CUSTOM_FIXTURE = FIXTURE_DIR / "standard-schema-v3-custom-complete-current.json"


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


class PlotterEntryPathTest(unittest.TestCase):
    def test_comparison_loader_accepts_supported_full_result(self):
        with tempfile.TemporaryDirectory(prefix="script-example-plotter-") as temporary:
            _, script = copy_script(Path(temporary), "plot_M4vsM5_benchmark_comparison.py")
            plotter = load_plotter(script, "current_comparison_plotter")
            result = plotter.load_data(CURRENT_FULL_FIXTURE, "average")

        self.assertEqual(result["cpu_name"], "Apple M4")
        self.assertEqual(result["version"], "0.63.0")
        self.assertAlmostEqual(result["bw_copy"], 29.520028152492667)
        self.assertAlmostEqual(result["l1_lat"], 1.1562554254729616)
        self.assertAlmostEqual(result["global_random"], 5.622018794239533)

    def test_hierarchy_loader_accepts_supported_full_result(self):
        with tempfile.TemporaryDirectory(prefix="script-example-plotter-") as temporary:
            _, script = copy_script(Path(temporary), "plot_benchmark-memory-latency-hierarchy.py")
            plotter = load_plotter(script, "current_hierarchy_plotter")
            result = plotter.load_latency_data(CURRENT_FULL_FIXTURE, "average")

        cpu_name, categories, latencies, locality_delta, version = result
        self.assertEqual(cpu_name, "Apple M4")
        self.assertEqual(len(categories), 4)
        self.assertEqual(len(latencies), 4)
        self.assertAlmostEqual(locality_delta, 0.6151600647670126)
        self.assertEqual(version, "0.63.0")

    def test_hierarchy_loader_accepts_supported_console_statistics(self):
        statistics = """\
L1 Cache:
  Average: 1.25
L2 Cache:
  Average: 4.50
16 KiB Locality Latency (ns):
  Average: 80.00
Global-Random Latency (ns):
  Average: 105.25
Locality Latency Delta, Global - 16 KiB (ns):
  Average: 25.25
"""
        with tempfile.TemporaryDirectory(prefix="script-example-text-") as temporary:
            root = Path(temporary)
            _, script = copy_script(root, "plot_benchmark-memory-latency-hierarchy.py")
            input_path = root / "Apple_M4_statistics.txt"
            input_path.write_text(statistics, encoding="utf-8")
            plotter = load_plotter(script, "current_text_hierarchy_plotter")
            result = plotter.load_latency_data(input_path, "average")

        cpu_name, _, latencies, locality_delta, version = result
        self.assertEqual(cpu_name, "Apple M4")
        self.assertEqual(latencies, [1.25, 4.5, 80.0, 105.25])
        self.assertEqual(locality_delta, 25.25)
        self.assertIsNone(version)

    def test_plotter_loaders_enforce_supported_contract(self):
        supported = json.loads(CURRENT_FULL_FIXTURE.read_text(encoding="utf-8"))
        other_version = json.loads(json.dumps(supported))
        other_version["version"] = "0.64.0"
        wrong_schema = json.loads(json.dumps(supported))
        wrong_schema["configuration"]["benchmark_schema_version"] = 2
        wrong_methodology = json.loads(json.dumps(supported))
        wrong_methodology["configuration"]["methodology_version"] = "benchmark-v1"
        wrong_shape = json.loads(json.dumps(supported))
        wrong_shape["configuration"]["output_file"] = None
        missing_provenance = json.loads(json.dumps(supported))
        missing_provenance["version"] = ""

        with tempfile.TemporaryDirectory(prefix="script-example-contract-") as temporary:
            root = Path(temporary)
            accepted_path = root / "other-version.json"
            accepted_path.write_text(json.dumps(other_version), encoding="utf-8")
            incompatible_inputs = {
                "schema 2": wrong_schema,
                "methodology benchmark-v1": wrong_methodology,
                "non-string output_file": wrong_shape,
                "empty provenance version": missing_provenance,
            }
            loaders = (
                ("plot_M4vsM5_benchmark_comparison.py", "contract_comparison", "load_data"),
                (
                    "plot_benchmark-memory-latency-hierarchy.py",
                    "contract_hierarchy",
                    "load_latency_data",
                ),
            )
            for filename, module_name, loader_name in loaders:
                _, script = copy_script(root / module_name, filename)
                plotter = load_plotter(script, module_name)
                accepted = getattr(plotter, loader_name)(accepted_path, "average")
                accepted_version = accepted["version"] if isinstance(accepted, dict) else accepted[-1]
                self.assertEqual(accepted_version, "0.64.0")
                for name, data in incompatible_inputs.items():
                    input_path = root / f"{module_name}-{name.replace(' ', '-')}.json"
                    input_path.write_text(json.dumps(data), encoding="utf-8")
                    with self.subTest(plotter=filename, input=name):
                        with self.assertRaisesRegex(RuntimeError, "complete supported standard schema-3"):
                            getattr(plotter, loader_name)(input_path, "average")


class ShellWorkflowEntryPathTest(unittest.TestCase):
    def make_fake_benchmark(self, root: Path) -> Path:
        fake = root / "fake-memory-benchmark"
        fake.write_text(
            """#!/bin/bash
if [ "${1:-}" = "-h" ]; then
  echo "--latency-stride-bytes"
  exit 0
fi
output_file=""
while [ "$#" -gt 0 ]; do
  if [ "$1" = "--output" ]; then
    output_file="$2"
    shift 2
  else
    shift
  fi
done
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

    @unittest.skipUnless(shutil.which("jq"), "jq is optional and is not installed")
    def test_latency_workflow_accepts_supported_result_with_jq(self):
        with tempfile.TemporaryDirectory(prefix="script-example-latency-jq-") as temporary:
            script_dir, completed = self.run_workflow(
                Path(temporary), "latency_test_script.sh", CURRENT_CUSTOM_FIXTURE
            )
            self.assert_success("latency_test_script.sh", completed)
            final_output = (script_dir / "final_output.txt").read_text(encoding="utf-8")

        self.assertIn("Using jq to extract latency sample statistics", completed.stdout)
        self.assertEqual(final_output.count("TLB Locality:"), 120)
        self.assertIn('"average": 0.6893617567937856', final_output)

    def test_latency_workflow_accepts_supported_result_with_python(self):
        with tempfile.TemporaryDirectory(prefix="script-example-latency-python-") as temporary:
            script_dir, completed = self.run_workflow(
                Path(temporary),
                "latency_test_script.sh",
                CURRENT_CUSTOM_FIXTURE,
                force_python=True,
            )
            self.assert_success("latency_test_script.sh", completed)
            final_output = (script_dir / "final_output.txt").read_text(encoding="utf-8")

        self.assertIn("Using Python to extract latency sample statistics", completed.stdout)
        self.assertEqual(final_output.count("TLB Locality:"), 120)
        self.assertIn('"average": 0.6893617567937856', final_output)

    def test_stride_tlb_workflow_accepts_supported_result(self):
        with tempfile.TemporaryDirectory(prefix="script-example-stride-tlb-") as temporary:
            script_dir, completed = self.run_workflow(
                Path(temporary), "latency_test_script_stride_tlb.sh", CURRENT_CUSTOM_FIXTURE
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
        self.assertEqual(rows[0]["average"], "0.6893617567937856")
        self.assertEqual(rows[0]["chain_node_count"], "64")

    def test_shell_workflows_reject_incompatible_contracts(self):
        supported = json.loads(CURRENT_CUSTOM_FIXTURE.read_text(encoding="utf-8"))
        wrong_schema_data = json.loads(json.dumps(supported))
        wrong_schema_data["configuration"]["benchmark_schema_version"] = 2
        wrong_methodology_data = json.loads(json.dumps(supported))
        wrong_methodology_data["configuration"]["methodology_version"] = "benchmark-v1"

        with tempfile.TemporaryDirectory(prefix="script-example-shell-contract-") as temporary:
            root = Path(temporary)
            wrong_schema = root / "wrong-schema.json"
            wrong_schema.write_text(json.dumps(wrong_schema_data), encoding="utf-8")
            wrong_methodology = root / "wrong-methodology.json"
            wrong_methodology.write_text(json.dumps(wrong_methodology_data), encoding="utf-8")

            latency_dir, latency = self.run_workflow(
                root / "latency",
                "latency_test_script.sh",
                wrong_schema,
                force_python=True,
            )
            final_output = (latency_dir / "final_output.txt").read_text(encoding="utf-8")

            stride_dir, stride = self.run_workflow(
                root / "stride", "latency_test_script_stride_tlb.sh", wrong_methodology
            )
            summaries = {
                path.resolve()
                for path in (stride_dir / "latency-results").glob("*/latency_summary.csv")
            }
            self.assertEqual(len(summaries), 1)
            with next(iter(summaries)).open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertNotEqual(latency.returncode, 0)
        self.assertNotIn("TLB Locality:", final_output)
        self.assertNotEqual(stride.returncode, 0)
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)

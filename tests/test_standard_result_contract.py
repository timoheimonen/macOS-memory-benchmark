#!/usr/bin/env python3
# Copyright 2026 Timo Heimonen
# SPDX-License-Identifier: GPL-3.0-or-later

"""Dependency-free compatibility tests for bundled standard-result readers."""

import copy
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
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "standard-schema-v2-complete-released.json"
HISTORICAL_FIXTURE = REPO_ROOT / "results" / "0.53.8" / "MacbookAirM5_benchmark.json"

sys.path.insert(0, str(SCRIPT_DIR))
from standard_result_contract import (  # noqa: E402
    HEADLINE_LOCALITY_LAYOUT,
    HISTORICAL_AVERAGE_TLB_LAYOUT,
    StandardResultContractError,
    require_standard_result,
    validate_standard_result,
)


def load_fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def make_schema_3():
    data = load_fixture()
    data["configuration"]["benchmark_schema_version"] = 3
    data["configuration"]["output_file"] = "-"
    data["conclusions_valid"] = True
    return data


def make_historical_result():
    statistics = {
        "average": 1.0,
        "median": 1.0,
        "p90": 1.0,
        "p95": 1.0,
        "p99": 1.0,
        "min": 1.0,
        "max": 1.0,
        "stddev": 0.0,
    }
    return {
        "version": "0.53.8",
        "configuration": {"cpu_name": "Apple M4"},
        "cache": {"custom": {"latency": {"samples_ns": {"statistics": statistics}}}},
    }


def load_plotter(filename, module_name):
    """Load a plotter boundary without requiring matplotlib or a display."""

    pyplot = types.ModuleType("matplotlib.pyplot")
    matplotlib = types.ModuleType("matplotlib")
    matplotlib.pyplot = pyplot
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_DIR / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load consumer module: {filename}")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"matplotlib": matplotlib, "matplotlib.pyplot": pyplot}):
        spec.loader.exec_module(module)
    return module


class StandardResultContractMatrixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jq = shutil.which("jq")
        if cls.jq is None:
            raise RuntimeError("jq is required to verify standard_result_contract.jq")

    def assert_contract(self, data, accepted, layout, schema_version, diagnostic):
        python_result = validate_standard_result(data)
        self.assertEqual(python_result.accepted, accepted)
        self.assertEqual(python_result.metric_layout, layout)
        self.assertEqual(python_result.schema_version, schema_version)
        self.assertEqual(python_result.diagnostic, diagnostic)

        jq_filter = 'include "standard_result_contract"; standard_result_contract'
        completed = subprocess.run(
            [self.jq, "-c", "-L", str(SCRIPT_DIR), jq_filter],
            input=json.dumps(data),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        jq_result = json.loads(completed.stdout)
        self.assertEqual(
            jq_result,
            {
                "accepted": accepted,
                "metric_layout": layout,
                "schema_version": schema_version,
                "diagnostic": diagnostic,
            },
        )

    def test_python_and_jq_contract_matrix(self):
        released = load_fixture()
        schema_2_conclusions_true = copy.deepcopy(released)
        schema_2_conclusions_true["conclusions_valid"] = True
        schema_2_conclusions_true["configuration"]["output_file"] = "released.json"
        schema_2_conclusions_false = copy.deepcopy(released)
        schema_2_conclusions_false["conclusions_valid"] = False
        schema_2_incomplete_status = copy.deepcopy(released)
        schema_2_incomplete_status["status"] = "interrupted"
        schema_2_incomplete_results = copy.deepcopy(released)
        schema_2_incomplete_results["results_complete"] = False
        schema_2_non_boolean_results = copy.deepcopy(released)
        schema_2_non_boolean_results["results_complete"] = 1
        schema_2_non_boolean_conclusions = copy.deepcopy(released)
        schema_2_non_boolean_conclusions["conclusions_valid"] = "true"
        schema_2_bad_output = copy.deepcopy(released)
        schema_2_bad_output["configuration"]["output_file"] = None

        schema_3 = make_schema_3()
        schema_3_missing_conclusions = copy.deepcopy(schema_3)
        del schema_3_missing_conclusions["conclusions_valid"]
        schema_3_false_conclusions = copy.deepcopy(schema_3)
        schema_3_false_conclusions["conclusions_valid"] = False
        schema_3_bad_conclusions = copy.deepcopy(schema_3)
        schema_3_bad_conclusions["conclusions_valid"] = "true"
        schema_3_missing_output = copy.deepcopy(schema_3)
        del schema_3_missing_output["configuration"]["output_file"]
        schema_3_bad_output = copy.deepcopy(schema_3)
        schema_3_bad_output["configuration"]["output_file"] = False

        unknown_schema = copy.deepcopy(released)
        unknown_schema["configuration"]["benchmark_schema_version"] = 4
        bad_schema_type = copy.deepcopy(released)
        bad_schema_type["configuration"]["benchmark_schema_version"] = "2"
        integral_float_schema = copy.deepcopy(released)
        integral_float_schema["configuration"]["benchmark_schema_version"] = 2.0
        integral_float_schema_3 = make_schema_3()
        integral_float_schema_3["configuration"]["benchmark_schema_version"] = 3.0
        fractional_schema = copy.deepcopy(released)
        fractional_schema["configuration"]["benchmark_schema_version"] = 2.5
        historical_without_configuration = make_historical_result()
        del historical_without_configuration["configuration"]
        bad_configuration_type = copy.deepcopy(released)
        bad_configuration_type["configuration"] = None

        cases = [
            (
                "released schema 2",
                released,
                True,
                HEADLINE_LOCALITY_LAYOUT,
                2,
                None,
            ),
            (
                "schema 2 optional fields have valid types",
                schema_2_conclusions_true,
                True,
                HEADLINE_LOCALITY_LAYOUT,
                2,
                None,
            ),
            (
                "schema 2 conclusions false",
                schema_2_conclusions_false,
                False,
                HEADLINE_LOCALITY_LAYOUT,
                2,
                "standard schema 2 field 'conclusions_valid' must be true when present",
            ),
            (
                "schema 2 incomplete status",
                schema_2_incomplete_status,
                False,
                HEADLINE_LOCALITY_LAYOUT,
                2,
                "standard schema 2 field 'status' must equal 'complete'",
            ),
            (
                "schema 2 incomplete results",
                schema_2_incomplete_results,
                False,
                HEADLINE_LOCALITY_LAYOUT,
                2,
                "standard schema 2 field 'results_complete' must be true",
            ),
            (
                "schema 2 non-boolean results",
                schema_2_non_boolean_results,
                False,
                HEADLINE_LOCALITY_LAYOUT,
                2,
                "standard schema 2 field 'results_complete' must be a boolean",
            ),
            (
                "schema 2 non-boolean conclusions",
                schema_2_non_boolean_conclusions,
                False,
                HEADLINE_LOCALITY_LAYOUT,
                2,
                "standard schema 2 field 'conclusions_valid' must be a boolean when present",
            ),
            (
                "schema 2 non-string output",
                schema_2_bad_output,
                False,
                HEADLINE_LOCALITY_LAYOUT,
                2,
                "standard schema 2 field 'configuration.output_file' must be a string when present",
            ),
            (
                "complete schema 3",
                schema_3,
                True,
                HEADLINE_LOCALITY_LAYOUT,
                3,
                None,
            ),
            (
                "schema 3 missing conclusions",
                schema_3_missing_conclusions,
                False,
                HEADLINE_LOCALITY_LAYOUT,
                3,
                "standard schema 3 field 'conclusions_valid' is required",
            ),
            (
                "schema 3 false conclusions",
                schema_3_false_conclusions,
                False,
                HEADLINE_LOCALITY_LAYOUT,
                3,
                "standard schema 3 field 'conclusions_valid' must be true",
            ),
            (
                "schema 3 non-boolean conclusions",
                schema_3_bad_conclusions,
                False,
                HEADLINE_LOCALITY_LAYOUT,
                3,
                "standard schema 3 field 'conclusions_valid' must be a boolean",
            ),
            (
                "schema 3 missing output",
                schema_3_missing_output,
                False,
                HEADLINE_LOCALITY_LAYOUT,
                3,
                "standard schema 3 field 'configuration.output_file' is required",
            ),
            (
                "schema 3 non-string output",
                schema_3_bad_output,
                False,
                HEADLINE_LOCALITY_LAYOUT,
                3,
                "standard schema 3 field 'configuration.output_file' must be a string",
            ),
            (
                "historical unversioned",
                make_historical_result(),
                True,
                HISTORICAL_AVERAGE_TLB_LAYOUT,
                None,
                None,
            ),
            (
                "historical with missing configuration",
                historical_without_configuration,
                True,
                HISTORICAL_AVERAGE_TLB_LAYOUT,
                None,
                None,
            ),
            (
                "unknown explicit schema",
                unknown_schema,
                False,
                None,
                4,
                "unsupported standard benchmark schema version: 4",
            ),
            (
                "integral numeric schema",
                integral_float_schema,
                True,
                HEADLINE_LOCALITY_LAYOUT,
                2,
                None,
            ),
            (
                "fractional numeric schema",
                fractional_schema,
                False,
                None,
                None,
                "standard benchmark schema version must be integer 2 or 3",
            ),
            (
                "integral numeric schema 3",
                integral_float_schema_3,
                True,
                HEADLINE_LOCALITY_LAYOUT,
                3,
                None,
            ),
            (
                "wrong schema type",
                bad_schema_type,
                False,
                None,
                None,
                "standard benchmark schema version must be integer 2 or 3",
            ),
            (
                "wrong configuration type",
                bad_configuration_type,
                False,
                None,
                None,
                "standard benchmark field 'configuration' must be a JSON object",
            ),
            (
                "non-object result",
                [],
                False,
                None,
                None,
                "standard benchmark result must be a JSON object",
            ),
        ]

        for name, data, accepted, layout, schema_version, diagnostic in cases:
            with self.subTest(name=name):
                self.assert_contract(data, accepted, layout, schema_version, diagnostic)

        self.assertIs(type(validate_standard_result(integral_float_schema).schema_version), int)
        self.assertIs(type(validate_standard_result(integral_float_schema_3).schema_version), int)

    def test_python_rejects_non_finite_schema_numbers(self):
        diagnostic = "standard benchmark schema version must be integer 2 or 3"
        for schema_version in (float("nan"), float("inf"), float("-inf")):
            data = load_fixture()
            data["configuration"]["benchmark_schema_version"] = schema_version
            with self.subTest(schema_version=schema_version):
                validation = validate_standard_result(data)
                self.assertIs(validation.accepted, False)
                self.assertIsNone(validation.metric_layout)
                self.assertIsNone(validation.schema_version)
                self.assertEqual(validation.diagnostic, diagnostic)

    def test_released_fixture_preserves_published_absence_and_identity(self):
        data = load_fixture()
        self.assertEqual(data["version"], "0.58.0")
        self.assertEqual(data["configuration"]["benchmark_schema_version"], 2)
        self.assertEqual(
            data["configuration"]["methodology_version"],
            "benchmark-v2-calibrated-seeded-balanced",
        )
        self.assertEqual(data["status"], "complete")
        self.assertIs(data["results_complete"], True)
        self.assertNotIn("conclusions_valid", data)
        self.assertNotIn("output_file", data["configuration"])

    def test_diagnostics_only_include_a_source_when_supplied(self):
        data = load_fixture()
        data["conclusions_valid"] = False
        diagnostic = "standard schema 2 field 'conclusions_valid' must be true when present"
        with self.assertRaises(StandardResultContractError) as without_source:
            require_standard_result(data)
        self.assertEqual(str(without_source.exception), diagnostic)
        with self.assertRaises(StandardResultContractError) as with_source:
            require_standard_result(data, "released.json")
        self.assertEqual(str(with_source.exception), f"{diagnostic}: released.json")

        completed = subprocess.run(
            [
                self.jq,
                "-c",
                "-L",
                str(SCRIPT_DIR),
                'include "standard_result_contract"; require_standard_result_contract',
            ],
            input=json.dumps(data),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(diagnostic, completed.stderr)
        self.assertNotIn("released.json", completed.stderr)


class BundledConsumerEntryPathTest(unittest.TestCase):
    def test_comparison_plotter_loader_accepts_released_fixture(self):
        plotter = load_plotter(
            "plot_M4vsM5_benchmark_comparison.py",
            "standard_contract_comparison_plotter",
        )
        result = plotter.load_data(FIXTURE, "average")
        self.assertEqual(result["cpu_name"], "Apple M4")
        self.assertAlmostEqual(result["bw_copy"], 163.97677637870817)
        self.assertAlmostEqual(result["l1_lat"], 0.7554867382098841)

    def test_hierarchy_plotter_loader_accepts_released_fixture(self):
        plotter = load_plotter(
            "plot_bechmark-memory-latency-hierarcy.py",
            "standard_contract_hierarchy_plotter",
        )
        cpu_name, categories, latencies, locality_delta, version = plotter.load_latency_data(
            FIXTURE, "average"
        )
        self.assertEqual(cpu_name, "Apple M4")
        self.assertEqual(len(categories), 4)
        self.assertEqual(len(latencies), 4)
        self.assertAlmostEqual(locality_delta, 0.6325197103262529)
        self.assertEqual(version, "0.58.0")

    def test_plotter_loaders_preserve_historical_metric_path(self):
        comparison = load_plotter(
            "plot_M4vsM5_benchmark_comparison.py",
            "standard_contract_historical_comparison_plotter",
        )
        comparison_result = comparison.load_data(HISTORICAL_FIXTURE, "average")
        self.assertEqual(comparison_result["version"], "0.53.8")
        self.assertGreater(comparison_result["global_random"], comparison_result["locality_16k"])

        hierarchy = load_plotter(
            "plot_bechmark-memory-latency-hierarcy.py",
            "standard_contract_historical_hierarchy_plotter",
        )
        _, categories, latencies, locality_delta, version = hierarchy.load_latency_data(
            HISTORICAL_FIXTURE, "average"
        )
        self.assertEqual(len(categories), 4)
        self.assertEqual(len(latencies), 4)
        self.assertGreater(locality_delta, 0.0)
        self.assertEqual(version, "0.53.8")

    def _make_fake_benchmark(self, root):
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
/bin/cp "${CONTRACT_FIXTURE_PATH}" "${output_file}"
""",
            encoding="utf-8",
        )
        fake.chmod(0o755)
        return fake

    def _copy_consumer(self, root, script_name):
        copied_script_dir = root / "script-examples"
        copied_script_dir.mkdir()
        for filename in (
            script_name,
            "standard_result_contract.py",
            "standard_result_contract.jq",
        ):
            shutil.copy2(SCRIPT_DIR / filename, copied_script_dir / filename)
        return copied_script_dir, copied_script_dir / script_name

    def _run_script(self, script_name, force_python=False):
        temporary = tempfile.TemporaryDirectory(prefix="standard-result-consumer-")
        root = Path(temporary.name)
        copied_script_dir, script = self._copy_consumer(root, script_name)
        fake_benchmark = self._make_fake_benchmark(root)
        environment = os.environ.copy()
        environment["BENCHMARK_CMD"] = str(fake_benchmark)
        environment["CONTRACT_FIXTURE_PATH"] = str(FIXTURE)

        if force_python:
            isolated_bin = root / "path-without-jq"
            isolated_bin.mkdir()
            for command in ("cat", "dirname", "mkdir", "rm"):
                resolved = shutil.which(command)
                if resolved is None:
                    self.fail(f"required shell utility not found: {command}")
                (isolated_bin / command).symlink_to(resolved)
            (isolated_bin / "python3").symlink_to(sys.executable)
            environment["PATH"] = str(isolated_bin)

        completed = subprocess.run(
            ["/bin/bash", str(script)],
            cwd=root,
            env=environment,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        if completed.returncode != 0:
            temporary.cleanup()
            self.fail(
                f"{script_name} failed with {completed.returncode}\n"
                f"stdout tail:\n{completed.stdout[-4000:]}\n"
                f"stderr tail:\n{completed.stderr[-4000:]}"
            )
        return temporary, copied_script_dir, completed

    def test_latency_script_jq_entry_path_accepts_released_fixture(self):
        temporary, copied_script_dir, completed = self._run_script("latency_test_script.sh")
        try:
            self.assertIn("Using jq to extract latency sample statistics", completed.stdout)
            final_output = (copied_script_dir / "final_output.txt").read_text(encoding="utf-8")
            self.assertEqual(final_output.count("TLB Locality:"), 120)
            self.assertIn('"average": 6.722050883633584', final_output)
        finally:
            temporary.cleanup()

    def test_latency_script_python_entry_path_accepts_released_fixture(self):
        temporary, copied_script_dir, completed = self._run_script(
            "latency_test_script.sh", force_python=True
        )
        try:
            self.assertIn("Using Python to extract latency sample statistics", completed.stdout)
            final_output = (copied_script_dir / "final_output.txt").read_text(encoding="utf-8")
            self.assertEqual(final_output.count("TLB Locality:"), 120)
            self.assertIn('"average": 6.722050883633584', final_output)
        finally:
            temporary.cleanup()

    def test_stride_tlb_script_entry_path_accepts_released_fixture(self):
        temporary, copied_script_dir, _ = self._run_script(
            "latency_test_script_stride_tlb.sh"
        )
        try:
            summaries = {
                path.resolve()
                for path in (copied_script_dir / "latency-results").glob("*/latency_summary.csv")
            }
            self.assertEqual(len(summaries), 1)
            with next(iter(summaries)).open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 315)
            self.assertEqual(rows[0]["average"], "6.722050883633584")
            self.assertEqual(rows[0]["chain_node_count"], "16384")
        finally:
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)

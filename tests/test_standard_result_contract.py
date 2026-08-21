#!/usr/bin/env python3
# Copyright 2026 Timo Heimonen
# SPDX-License-Identifier: GPL-3.0-or-later

"""Dependency-free strict schema-3 tests for bundled standard-result readers."""

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
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"
RELEASED_FIXTURE = FIXTURE_DIR / "standard-schema-v2-complete-released.json"
CURRENT_FULL_FIXTURE = FIXTURE_DIR / "standard-schema-v3-complete-current.json"
CURRENT_CUSTOM_FIXTURE = FIXTURE_DIR / "standard-schema-v3-custom-complete-current.json"
HISTORICAL_FIXTURE = REPO_ROOT / "results" / "0.53.8" / "MacbookAirM5_benchmark.json"

sys.path.insert(0, str(SCRIPT_DIR))
from standard_result_contract import (  # noqa: E402
    HEADLINE_LOCALITY_LAYOUT,
    StandardResultContractError,
    require_standard_result,
    validate_standard_result,
)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_full_fixture():
    return load_json(CURRENT_FULL_FIXTURE)


def load_custom_fixture():
    return load_json(CURRENT_CUSTOM_FIXTURE)


def variant(
    source,
    *,
    top_values=None,
    remove_top=(),
    configuration_values=None,
    remove_configuration=(),
):
    data = copy.deepcopy(source)
    for key in remove_top:
        data.pop(key, None)
    if top_values:
        data.update(top_values)
    for key in remove_configuration:
        data["configuration"].pop(key, None)
    if configuration_values:
        data["configuration"].update(configuration_values)
    return data


def statistics(value=1.0):
    return {
        "average": value,
        "median": value,
        "p90": value,
        "p95": value,
        "p99": value,
        "min": value,
        "max": value,
        "stddev": 0.0,
    }


def make_historical_full_result():
    metric_block = {"statistics": statistics()}
    return {
        "version": "0.53.8",
        "configuration": {
            "mode": "benchmark",
            "cpu_name": "Apple M4",
        },
        "cache": {
            "l1": {"latency": {"average_ns": copy.deepcopy(metric_block)}},
            "l2": {"latency": {"average_ns": copy.deepcopy(metric_block)}},
        },
        "main_memory": {
            "bandwidth": {
                "copy_gb_s": copy.deepcopy(metric_block),
                "read_gb_s": copy.deepcopy(metric_block),
                "write_gb_s": copy.deepcopy(metric_block),
            },
            "latency": {
                "average_ns": copy.deepcopy(metric_block),
                "auto_tlb_breakdown": {
                    "tlb_hit_ns": copy.deepcopy(metric_block),
                    "tlb_miss_ns": copy.deepcopy(metric_block),
                    "page_walk_penalty_ns": copy.deepcopy(metric_block),
                },
            }
        },
    }


def make_historical_custom_result():
    return {
        "version": "0.53.8",
        "configuration": {
            "mode": "benchmark",
            "cpu_name": "Apple M4",
        },
        "cache": {
            "custom": {
                "latency": {
                    "samples_ns": {
                        "statistics": statistics(),
                    }
                }
            }
        },
    }


def accepted(schema_version=3):
    return {
        "accepted": True,
        "metric_layout": HEADLINE_LOCALITY_LAYOUT,
        "schema_version": schema_version,
        "diagnostic": None,
    }


def rejected(diagnostic, schema_version=None, metric_layout=None):
    return {
        "accepted": False,
        "metric_layout": metric_layout,
        "schema_version": schema_version,
        "diagnostic": diagnostic,
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

    def assert_contract(self, data, expected):
        python_result = validate_standard_result(data)
        self.assertEqual(
            {
                "accepted": python_result.accepted,
                "metric_layout": python_result.metric_layout,
                "schema_version": python_result.schema_version,
                "diagnostic": python_result.diagnostic,
            },
            expected,
        )

        completed = subprocess.run(
            [
                self.jq,
                "-c",
                "-L",
                str(SCRIPT_DIR),
                'include "standard_result_contract"; standard_result_contract',
            ],
            input=json.dumps(data, allow_nan=False),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout), expected)

    def test_python_and_jq_contract_matrix(self):
        current = load_full_fixture()
        custom = load_custom_fixture()
        released = load_json(RELEASED_FIXTURE)

        released_with_schema_3_fields = variant(
            released,
            top_values={"conclusions_valid": True},
            configuration_values={
                "mode": "benchmark",
                "output_file": "released.json",
            },
        )

        historical_full = make_historical_full_result()
        historical_full_without_configuration = copy.deepcopy(historical_full)
        del historical_full_without_configuration["configuration"]
        historical_custom = make_historical_custom_result()
        historical_custom_without_configuration = copy.deepcopy(historical_custom)
        del historical_custom_without_configuration["configuration"]

        required_configuration = "standard benchmark field 'configuration' is required"
        required_schema = (
            "standard benchmark field "
            "'configuration.benchmark_schema_version' is required"
        )
        exact_mode = "standard benchmark field 'configuration.mode' must equal 'benchmark'"
        conflicting_identity = (
            "standard benchmark result contains a conflicting mode/schema identity"
        )
        invalid_schema_type = "standard benchmark schema version must be integer 3"
        schema_3_layout = HEADLINE_LOCALITY_LAYOUT

        cases = [
            ("current full fixture", current, accepted()),
            ("current custom fixture", custom, accepted()),
            (
                "direct stdout output",
                variant(current, configuration_values={"output_file": "-"}),
                accepted(),
            ),
            (
                "ordinary output token",
                variant(current, configuration_values={"output_file": "./result.json"}),
                accepted(),
            ),
            (
                "nested empty output",
                variant(current, configuration_values={"output_file": ""}),
                accepted(),
            ),
            (
                "integral numeric schema 3.0",
                variant(current, configuration_values={"benchmark_schema_version": 3.0}),
                accepted(),
            ),
            (
                "null top-level result",
                None,
                rejected("standard benchmark result must be a JSON object"),
            ),
            (
                "array top-level result",
                [],
                rejected("standard benchmark result must be a JSON object"),
            ),
            (
                "string top-level result",
                "benchmark",
                rejected("standard benchmark result must be a JSON object"),
            ),
            (
                "boolean top-level result",
                True,
                rejected("standard benchmark result must be a JSON object"),
            ),
            (
                "empty object",
                {},
                rejected(required_configuration),
            ),
            (
                "arbitrary release-looking object",
                {"version": "0.62.0", "status": "complete"},
                rejected(required_configuration),
            ),
            (
                "empty configuration",
                {"configuration": {}},
                rejected(required_schema),
            ),
            (
                "benchmark mode without schema",
                {"configuration": {"mode": "benchmark"}},
                rejected(required_schema),
            ),
            (
                "wrong mode without standard schema",
                {"configuration": {"mode": "patterns"}},
                rejected(exact_mode),
            ),
            (
                "null configuration",
                {"configuration": None},
                rejected("standard benchmark field 'configuration' must be a JSON object"),
            ),
            (
                "array configuration",
                {"configuration": []},
                rejected("standard benchmark field 'configuration' must be a JSON object"),
            ),
            (
                "string configuration",
                {"configuration": "benchmark"},
                rejected("standard benchmark field 'configuration' must be a JSON object"),
            ),
            (
                "schema 3 without mode",
                variant(current, remove_configuration=("mode",)),
                rejected(exact_mode, schema_version=3),
            ),
            (
                "wrong benchmark mode",
                variant(current, configuration_values={"mode": "patterns"}),
                rejected(exact_mode),
            ),
            (
                "null benchmark mode",
                variant(current, configuration_values={"mode": None}),
                rejected(exact_mode),
            ),
            (
                "pattern schema identity",
                {
                    "configuration": {
                        "mode": "patterns",
                        "pattern_schema_version": 3,
                    }
                },
                rejected(conflicting_identity),
            ),
            (
                "TLB schema identity",
                {
                    "configuration": {
                        "mode": "analyze-tlb",
                        "schema_version": 4,
                    }
                },
                rejected(conflicting_identity),
            ),
            (
                "core-to-core schema identity",
                {
                    "configuration": {
                        "mode": "core-to-core",
                        "schema_version": 2,
                    }
                },
                rejected(conflicting_identity),
            ),
            (
                "sweep schema identity",
                {
                    "configuration": {
                        "mode": "benchmark",
                        "sweep_schema_version": 1,
                    }
                },
                rejected(conflicting_identity),
            ),
            (
                "mixed pattern and standard identity",
                variant(current, configuration_values={"pattern_schema_version": 3}),
                rejected(conflicting_identity),
            ),
            (
                "mixed TLB and standard identity",
                variant(current, configuration_values={"schema_version": 4}),
                rejected(conflicting_identity),
            ),
            (
                "mixed sweep and standard identity",
                variant(current, configuration_values={"sweep_schema_version": 1}),
                rejected(conflicting_identity),
            ),
            (
                "alternative schema precedes wrong mode",
                variant(
                    current,
                    configuration_values={
                        "mode": "patterns",
                        "pattern_schema_version": 3,
                    },
                ),
                rejected(conflicting_identity),
            ),
            (
                "conflicting top-level schema identity",
                variant(current, top_values={"schema_version": 1}),
                rejected(conflicting_identity),
            ),
            (
                "current GPU schema 1",
                {"mode": "gpu_bandwidth", "schema_version": 1},
                rejected(
                    "GPU bandwidth schema 1 is not supported by this standard CPU result consumer"
                ),
            ),
            (
                "GPU schema boolean is not numeric schema 1",
                {"mode": "gpu_bandwidth", "schema_version": True},
                rejected(required_configuration),
            ),
            (
                "released schema 2",
                released,
                rejected(
                    "unsupported standard benchmark schema version: 2",
                    schema_version=2,
                ),
            ),
            (
                "released schema 2 with later fields",
                released_with_schema_3_fields,
                rejected(
                    "unsupported standard benchmark schema version: 2",
                    schema_version=2,
                ),
            ),
            (
                "historical full hierarchy",
                historical_full,
                rejected(required_schema),
            ),
            (
                "historical full hierarchy without configuration",
                historical_full_without_configuration,
                rejected(required_configuration),
            ),
            (
                "historical custom cache",
                historical_custom,
                rejected(required_schema),
            ),
            (
                "historical custom cache without configuration",
                historical_custom_without_configuration,
                rejected(required_configuration),
            ),
            (
                "partial historical-looking object",
                {
                    "configuration": {"mode": "benchmark"},
                    "cache": {"custom": {"latency": {}}},
                },
                rejected(required_schema),
            ),
        ]

        for schema_version in (-1, 0, 1, 2, 4):
            cases.append(
                (
                    f"unsupported integral schema {schema_version}",
                    variant(
                        current,
                        configuration_values={
                            "benchmark_schema_version": schema_version,
                        },
                    ),
                    rejected(
                        f"unsupported standard benchmark schema version: {schema_version}",
                        schema_version=schema_version,
                    ),
                )
            )

        for name, schema_value, normalized_schema_version in (
            ("unsupported integral schema 2.0", 2.0, 2),
            ("unsupported integral schema 0.0", 0.0, 0),
            ("unsupported integral schema -0.0", -0.0, 0),
            ("unsupported integral schema 4.0", 4.0, 4),
        ):
            cases.append(
                (
                    name,
                    variant(
                        current,
                        configuration_values={
                            "benchmark_schema_version": schema_value,
                        },
                    ),
                    rejected(
                        "unsupported standard benchmark schema version: "
                        f"{normalized_schema_version}",
                        schema_version=normalized_schema_version,
                    ),
                )
            )

        for schema_version in (
            9007199254740993,
            9223372036854775807,
            -9223372036854775809,
        ):
            cases.append(
                (
                    f"unsupported exact large integer schema {schema_version}",
                    variant(
                        current,
                        configuration_values={
                            "benchmark_schema_version": schema_version,
                        },
                    ),
                    rejected(
                        f"unsupported standard benchmark schema version: {schema_version}",
                        schema_version=schema_version,
                    ),
                )
            )

        for name, schema_value in (
            ("boolean", True),
            ("string", "3"),
            ("fractional", 3.5),
            ("null", None),
            ("array", [3]),
            ("object", {"version": 3}),
        ):
            cases.append(
                (
                    f"{name} schema value",
                    variant(
                        current,
                        configuration_values={
                            "benchmark_schema_version": schema_value,
                        },
                    ),
                    rejected(invalid_schema_type),
                )
            )

        schema_3_completion_cases = [
            (
                "missing status",
                variant(current, remove_top=("status",)),
                "standard schema 3 field 'status' must equal 'complete'",
            ),
            (
                "incomplete status",
                variant(current, top_values={"status": "interrupted"}),
                "standard schema 3 field 'status' must equal 'complete'",
            ),
            (
                "wrong-type status",
                variant(current, top_values={"status": True}),
                "standard schema 3 field 'status' must equal 'complete'",
            ),
            (
                "missing results_complete",
                variant(current, remove_top=("results_complete",)),
                "standard schema 3 field 'results_complete' is required",
            ),
            (
                "false results_complete",
                variant(current, top_values={"results_complete": False}),
                "standard schema 3 field 'results_complete' must be true",
            ),
            (
                "wrong-type results_complete",
                variant(current, top_values={"results_complete": 1}),
                "standard schema 3 field 'results_complete' must be a boolean",
            ),
            (
                "missing conclusions_valid",
                variant(current, remove_top=("conclusions_valid",)),
                "standard schema 3 field 'conclusions_valid' is required",
            ),
            (
                "false conclusions_valid",
                variant(current, top_values={"conclusions_valid": False}),
                "standard schema 3 field 'conclusions_valid' must be true",
            ),
            (
                "wrong-type conclusions_valid",
                variant(current, top_values={"conclusions_valid": "true"}),
                "standard schema 3 field 'conclusions_valid' must be a boolean",
            ),
            (
                "missing output_file",
                variant(current, remove_configuration=("output_file",)),
                "standard schema 3 field 'configuration.output_file' is required",
            ),
            (
                "null output_file",
                variant(current, configuration_values={"output_file": None}),
                "standard schema 3 field 'configuration.output_file' must be a string",
            ),
            (
                "boolean output_file",
                variant(current, configuration_values={"output_file": False}),
                "standard schema 3 field 'configuration.output_file' must be a string",
            ),
            (
                "array output_file",
                variant(current, configuration_values={"output_file": []}),
                "standard schema 3 field 'configuration.output_file' must be a string",
            ),
        ]
        for name, data, diagnostic in schema_3_completion_cases:
            cases.append(
                (
                    name,
                    data,
                    rejected(
                        diagnostic,
                        schema_version=3,
                        metric_layout=schema_3_layout,
                    ),
                )
            )

        for name, data, expected in cases:
            with self.subTest(name=name):
                self.assert_contract(data, expected)

        schema_float_result = validate_standard_result(
            variant(current, configuration_values={"benchmark_schema_version": 3.0})
        )
        self.assertIs(type(schema_float_result.schema_version), int)

    def test_python_rejects_non_finite_schema_numbers_without_raising(self):
        diagnostic = "standard benchmark schema version must be integer 3"
        current = load_full_fixture()
        for schema_version in (float("nan"), float("inf"), float("-inf")):
            data = variant(
                current,
                configuration_values={"benchmark_schema_version": schema_version},
            )
            with self.subTest(schema_version=schema_version):
                validation = validate_standard_result(data)
                self.assertEqual(
                    {
                        "accepted": validation.accepted,
                        "metric_layout": validation.metric_layout,
                        "schema_version": validation.schema_version,
                        "diagnostic": validation.diagnostic,
                    },
                    rejected(diagnostic),
                )

    def test_python_and_jq_handle_raw_numeric_representations_identically(self):
        cases = (
            (
                "1e999",
                rejected("standard benchmark schema version must be integer 3"),
            ),
            (
                "-1e999",
                rejected("standard benchmark schema version must be integer 3"),
            ),
            (
                "2e0",
                rejected(
                    "unsupported standard benchmark schema version: 2",
                    schema_version=2,
                ),
            ),
            (
                "1e20",
                rejected(
                    "unsupported standard benchmark schema version: "
                    "100000000000000000000",
                    schema_version=100000000000000000000,
                ),
            ),
            (
                "-1e20",
                rejected(
                    "unsupported standard benchmark schema version: "
                    "-100000000000000000000",
                    schema_version=-100000000000000000000,
                ),
            ),
            (
                "9223372036854775807.0",
                rejected(
                    "unsupported standard benchmark schema version: "
                    "9223372036854776000",
                    schema_version=9223372036854776000,
                ),
            ),
            (
                "9007199254740993.0",
                rejected(
                    "unsupported standard benchmark schema version: "
                    "9007199254740992",
                    schema_version=9007199254740992,
                ),
            ),
        )
        for schema_token, expected in cases:
            payload = (
                '{"configuration":{"mode":"benchmark",'
                f'"benchmark_schema_version":{schema_token},"output_file":"-"}},'
                '"status":"complete","results_complete":true,'
                '"conclusions_valid":true}'
            )
            with self.subTest(schema_token=schema_token, boundary="python"):
                python_result = validate_standard_result(json.loads(payload))
                self.assertEqual(
                    {
                        "accepted": python_result.accepted,
                        "metric_layout": python_result.metric_layout,
                        "schema_version": python_result.schema_version,
                        "diagnostic": python_result.diagnostic,
                    },
                    expected,
                )

            with self.subTest(schema_token=schema_token, boundary="jq"):
                completed = subprocess.run(
                    [
                        self.jq,
                        "-c",
                        "-L",
                        str(SCRIPT_DIR),
                        'include "standard_result_contract"; standard_result_contract',
                    ],
                    input=payload,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(json.loads(completed.stdout), expected)

    def test_canonical_fixtures_preserve_current_producer_shapes(self):
        full = load_full_fixture()
        custom = load_custom_fixture()

        for name, data, output_file in (
            (
                "full",
                full,
                "/tmp/standard-schema-v3-complete-current.capture.json",
            ),
            (
                "custom",
                custom,
                "/tmp/standard-schema-v3-custom-complete-current.capture.json",
            ),
        ):
            with self.subTest(name=name):
                self.assertEqual(data["version"], "0.62.0")
                self.assertEqual(data["configuration"]["mode"], "benchmark")
                self.assertEqual(data["configuration"]["benchmark_schema_version"], 3)
                self.assertEqual(data["configuration"]["output_file"], output_file)
                self.assertEqual(
                    data["configuration"]["methodology_version"],
                    "benchmark-v2-calibrated-seeded-balanced",
                )
                self.assertEqual(
                    data["configuration"]["latency_headline_semantics"],
                    "one-continuous-pointer-chase-pass",
                )
                self.assertEqual(
                    data["configuration"]["latency_sample_semantics"],
                    "separate-continuing-window-pass",
                )
                self.assertEqual(data["status"], "complete")
                self.assertIs(data["results_complete"], True)
                self.assertIs(data["conclusions_valid"], True)
                self.assertEqual(data["planned_loops"], data["completed_loops"])
                self.assertEqual(
                    data["planned_measurements"],
                    data["completed_measurements"],
                )
                self.assertEqual(validate_standard_result(data), (True, HEADLINE_LOCALITY_LAYOUT, 3, None))

        self.assertIs(full["configuration"]["use_custom_cache_size"], False)
        self.assertEqual(set(full["cache"]), {"l1", "l2"})
        self.assertIs(custom["configuration"]["use_custom_cache_size"], True)
        self.assertEqual(set(custom["cache"]), {"custom"})

        full_metric_paths = (
            full["main_memory"]["bandwidth"]["copy_gb_s"]["statistics"],
            full["main_memory"]["bandwidth"]["read_gb_s"]["statistics"],
            full["main_memory"]["bandwidth"]["write_gb_s"]["statistics"],
            full["cache"]["l1"]["latency"]["headline_ns"]["statistics"],
            full["cache"]["l2"]["latency"]["headline_ns"]["statistics"],
            full["main_memory"]["latency"]["automatic_locality_comparison"][
                "locality_16k_latency_ns"
            ]["statistics"],
            full["main_memory"]["latency"]["automatic_locality_comparison"][
                "global_random_latency_ns"
            ]["statistics"],
            full["main_memory"]["latency"]["automatic_locality_comparison"][
                "locality_latency_delta_ns"
            ]["statistics"],
        )
        required_statistics = {
            "average",
            "median",
            "p90",
            "p95",
            "p99",
            "min",
            "max",
            "stddev",
        }
        for metric in full_metric_paths:
            self.assertTrue(required_statistics.issubset(metric))

        custom_headline = custom["cache"]["custom"]["latency"]["headline_ns"]
        self.assertTrue(
            required_statistics.issubset(
                custom_headline["pooled_sample_distribution"]["statistics"]
            )
        )
        self.assertEqual(custom_headline["measurements"][0]["chain_node_count"], 64)
        self.assertEqual(custom["configuration"]["custom_cache_size_kb"], 16)
        self.assertEqual(custom["configuration"]["latency_tlb_locality_kb"], 16)
        self.assertEqual(custom["configuration"]["latency_stride_bytes"], 256)
        self.assertEqual(custom["configuration"]["native_page_size_bytes"], 16384)

    def test_released_fixture_is_preserved_only_as_negative_evidence(self):
        data = load_json(RELEASED_FIXTURE)
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
        self.assertEqual(
            validate_standard_result(data),
            (
                False,
                None,
                2,
                "unsupported standard benchmark schema version: 2",
            ),
        )

    def test_raising_boundaries_reject_noncurrent_inputs(self):
        released = load_json(RELEASED_FIXTURE)
        historical = load_json(HISTORICAL_FIXTURE)
        wrong_mode = variant(
            load_full_fixture(),
            configuration_values={"mode": "patterns"},
        )
        cases = (
            (
                "released schema 2",
                released,
                "unsupported standard benchmark schema version: 2",
            ),
            (
                "archived 0.53.x",
                historical,
                "standard benchmark field 'configuration.benchmark_schema_version' is required",
            ),
            (
                "arbitrary JSON",
                {},
                "standard benchmark field 'configuration' is required",
            ),
            (
                "wrong mode",
                wrong_mode,
                "standard benchmark field 'configuration.mode' must equal 'benchmark'",
            ),
        )

        for name, data, diagnostic in cases:
            with self.subTest(name=name, boundary="python"):
                with self.assertRaises(StandardResultContractError) as raised:
                    require_standard_result(data)
                self.assertEqual(str(raised.exception), diagnostic)

            with self.subTest(name=name, boundary="jq"):
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

    def test_raising_boundaries_accept_both_current_fixtures(self):
        for fixture in (CURRENT_FULL_FIXTURE, CURRENT_CUSTOM_FIXTURE):
            data = load_json(fixture)
            with self.subTest(fixture=fixture.name, boundary="python"):
                self.assertEqual(
                    require_standard_result(data),
                    (True, HEADLINE_LOCALITY_LAYOUT, 3, None),
                )

            with self.subTest(fixture=fixture.name, boundary="jq"):
                completed = subprocess.run(
                    [
                        self.jq,
                        "-c",
                        "-L",
                        str(SCRIPT_DIR),
                        'include "standard_result_contract"; '
                        "require_standard_result_contract",
                    ],
                    input=json.dumps(data),
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(json.loads(completed.stdout), accepted())

    def test_diagnostics_only_include_a_source_when_supplied(self):
        data = load_json(RELEASED_FIXTURE)
        diagnostic = "unsupported standard benchmark schema version: 2"
        with self.assertRaises(StandardResultContractError) as without_source:
            require_standard_result(data)
        self.assertEqual(str(without_source.exception), diagnostic)
        with self.assertRaises(StandardResultContractError) as with_source:
            require_standard_result(data, "released.json")
        self.assertEqual(str(with_source.exception), f"{diagnostic}: released.json")


class BundledConsumerEntryPathTest(unittest.TestCase):
    def test_comparison_plotter_loader_accepts_current_full_fixture(self):
        plotter = load_plotter(
            "plot_M4vsM5_benchmark_comparison.py",
            "standard_contract_comparison_plotter",
        )
        result = plotter.load_data(CURRENT_FULL_FIXTURE, "average")
        self.assertEqual(result["cpu_name"], "Apple M4")
        self.assertEqual(result["version"], "0.62.0")
        self.assertAlmostEqual(result["bw_copy"], 54.82750326797385)
        self.assertAlmostEqual(result["l1_lat"], 0.7860867955690964)
        self.assertAlmostEqual(result["l2_lat"], 26.12450366883766)
        self.assertAlmostEqual(result["locality_16k"], 4.727339302165334)
        self.assertAlmostEqual(result["global_random"], 5.333158211283286)

    def test_hierarchy_plotter_loader_accepts_current_full_fixture(self):
        plotter = load_plotter(
            "plot_bechmark-memory-latency-hierarcy.py",
            "standard_contract_hierarchy_plotter",
        )
        cpu_name, categories, latencies, locality_delta, version = plotter.load_latency_data(
            CURRENT_FULL_FIXTURE,
            "average",
        )
        self.assertEqual(cpu_name, "Apple M4")
        self.assertEqual(len(categories), 4)
        self.assertEqual(len(latencies), 4)
        self.assertAlmostEqual(locality_delta, 0.5273043967303268)
        self.assertEqual(version, "0.62.0")

    def test_plotter_loaders_reject_released_and_archived_json(self):
        comparison = load_plotter(
            "plot_M4vsM5_benchmark_comparison.py",
            "standard_contract_negative_comparison_plotter",
        )
        hierarchy = load_plotter(
            "plot_bechmark-memory-latency-hierarcy.py",
            "standard_contract_negative_hierarchy_plotter",
        )
        cases = (
            (
                RELEASED_FIXTURE,
                "unsupported standard benchmark schema version: 2",
            ),
            (
                HISTORICAL_FIXTURE,
                "standard benchmark field 'configuration.benchmark_schema_version' is required",
            ),
        )
        for path, diagnostic in cases:
            for name, loader in (
                ("comparison", comparison.load_data),
                ("hierarchy", hierarchy.load_latency_data),
            ):
                with self.subTest(path=path.name, consumer=name):
                    with self.assertRaises(StandardResultContractError) as raised:
                        loader(path, "average")
                    self.assertIn(diagnostic, str(raised.exception))

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

    def _run_script(self, script_name, fixture, force_python=False):
        temporary = tempfile.TemporaryDirectory(prefix="standard-result-consumer-")
        root = Path(temporary.name)
        copied_script_dir, script = self._copy_consumer(root, script_name)
        fake_benchmark = self._make_fake_benchmark(root)
        environment = os.environ.copy()
        environment["BENCHMARK_CMD"] = str(fake_benchmark)
        environment["CONTRACT_FIXTURE_PATH"] = str(fixture)

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

    def test_latency_script_jq_entry_path_accepts_current_custom_fixture(self):
        temporary, copied_script_dir, completed = self._run_script(
            "latency_test_script.sh",
            CURRENT_CUSTOM_FIXTURE,
        )
        try:
            self.assertIn("Using jq to extract latency sample statistics", completed.stdout)
            final_output = (copied_script_dir / "final_output.txt").read_text(encoding="utf-8")
            self.assertEqual(final_output.count("TLB Locality:"), 120)
            self.assertIn('"average": 0.696527076913686', final_output)
        finally:
            temporary.cleanup()

    def test_latency_script_python_entry_path_accepts_current_custom_fixture(self):
        temporary, copied_script_dir, completed = self._run_script(
            "latency_test_script.sh",
            CURRENT_CUSTOM_FIXTURE,
            force_python=True,
        )
        try:
            self.assertIn("Using Python to extract latency sample statistics", completed.stdout)
            final_output = (copied_script_dir / "final_output.txt").read_text(encoding="utf-8")
            self.assertEqual(final_output.count("TLB Locality:"), 120)
            self.assertIn('"average": 0.696527076913686', final_output)
        finally:
            temporary.cleanup()

    def test_stride_tlb_script_entry_path_accepts_current_custom_fixture(self):
        temporary, copied_script_dir, _ = self._run_script(
            "latency_test_script_stride_tlb.sh",
            CURRENT_CUSTOM_FIXTURE,
        )
        try:
            summaries = {
                path.resolve()
                for path in (copied_script_dir / "latency-results").glob(
                    "*/latency_summary.csv"
                )
            }
            self.assertEqual(len(summaries), 1)
            with next(iter(summaries)).open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 315)
            self.assertEqual(rows[0]["average"], "0.696527076913686")
            self.assertEqual(rows[0]["chain_node_count"], "64")
        finally:
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)

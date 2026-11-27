#!/usr/bin/env python3
# Copyright 2026 Timo Heimonen <timo.heimonen@proton.me>
# SPDX-License-Identifier: GPL-3.0-or-later

"""End-to-end Tier 1-4 acceptance test suite for --cpu-bandwidth (-B) and --cpu-latency (-L).

Usage:
  python3 tests/test_cpu_e2e_acceptance.py [--binary ./memory_benchmark] [--list] [-v]
  MEMORY_BENCHMARK_BIN=./memory_benchmark python3 -m unittest -v tests/test_cpu_e2e_acceptance.py
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import tempfile
import time
import unittest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = REPO_ROOT / "script-examples"
BANDWIDTH_VERIFIER = SCRIPT_DIR / "verify_cpu_bandwidth_result.py"
LATENCY_VERIFIER = SCRIPT_DIR / "verify_cpu_latency_result.py"
LLM_VERIFIER = SCRIPT_DIR / "verify_llm_result.py"

BANDWIDTH_METHODOLOGY = "cpu-bandwidth-v1-arm64-calibrated-seeded-balanced"
LATENCY_METHODOLOGY = "cpu-latency-v1-arm64-calibrated-seeded-balanced"

DEFAULT_RUN_TIMEOUT_SEC = 120
TLB_SMOKE_TIMEOUT_SEC = 300
INTERRUPT_TIMEOUT_SEC = 30


def resolve_binary_path(raw_path: str | None = None) -> Path:
    """Resolve the target binary path without touching or building anything."""
    candidate = raw_path if raw_path is not None else os.environ.get("MEMORY_BENCHMARK_BIN", "./memory_benchmark")
    path = Path(candidate)
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate.resolve()
    return (REPO_ROOT / path).resolve()


def validate_executable_binary(binary_path: Path) -> str | None:
    """Return an error string if binary_path is missing or not executable, else None."""
    if not binary_path.exists():
        return f"Target binary does not exist: {binary_path}"
    if not binary_path.is_file():
        return f"Target binary path is not a regular file: {binary_path}"
    if not os.access(binary_path, os.X_OK):
        return f"Target binary is not executable: {binary_path}"
    return None


class E2EAcceptanceTestCase(unittest.TestCase):
    """Base TestCase providing binary validation, subprocess execution, and verifier oracles."""

    binary_path: Path

    def setUp(self) -> None:
        self.binary_path = resolve_binary_path()
        err = validate_executable_binary(self.binary_path)
        if err is not None:
            self.fail(err)

    def run_binary(
        self,
        args: list[str],
        *,
        cwd: Path,
        timeout: int = DEFAULT_RUN_TIMEOUT_SEC,
    ) -> subprocess.CompletedProcess[str]:
        err = validate_executable_binary(self.binary_path)
        if err is not None:
            self.fail(err)
        return subprocess.run(
            [str(self.binary_path), *args],
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )

    def run_binary_and_interrupt(
        self,
        args: list[str],
        *,
        cwd: Path,
        sig: int,
        wait_for_output_timeout: float = 10.0,
        post_signal_delay: float = 0.25,
    ) -> subprocess.CompletedProcess[str]:
        """Launch binary, wait until runtime output begins, deliver sig, and capture clean shutdown."""
        err = validate_executable_binary(self.binary_path)
        if err is not None:
            self.fail(err)

        proc = subprocess.Popen(
            [str(self.binary_path), *args],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        sel = selectors.DefaultSelector()
        assert proc.stdout is not None and proc.stderr is not None
        sel.register(proc.stdout, selectors.EVENT_READ, data="stdout")
        sel.register(proc.stderr, selectors.EVENT_READ, data="stderr")

        saw_runtime_output = False
        deadline = time.monotonic() + wait_for_output_timeout
        try:
            while time.monotonic() < deadline and proc.poll() is None:
                events = sel.select(timeout=0.05)
                for key, _ in events:
                    chunk = os.read(key.fd, 4096).decode("utf-8", errors="replace")
                    if not chunk:
                        try:
                            sel.unregister(key.fileobj)
                        except KeyError:
                            pass
                        continue
                    if key.data == "stdout":
                        stdout_chunks.append(chunk)
                    else:
                        stderr_chunks.append(chunk)
                    saw_runtime_output = True
                if saw_runtime_output:
                    time.sleep(post_signal_delay)
                    break

            if proc.poll() is None:
                proc.send_signal(sig)
            else:
                self.fail(
                    "Process exited before interruption signal could be delivered "
                    f"(rc={proc.returncode}, args={args})\n"
                    f"stdout:\n{''.join(stdout_chunks)}\nstderr:\n{''.join(stderr_chunks)}"
                )

            try:
                rem_out, rem_err = proc.communicate(timeout=INTERRUPT_TIMEOUT_SEC)
                if rem_out:
                    stdout_chunks.append(rem_out)
                if rem_err:
                    stderr_chunks.append(rem_err)
            except subprocess.TimeoutExpired:
                proc.kill()
                rem_out, rem_err = proc.communicate(timeout=5)
                if rem_out:
                    stdout_chunks.append(rem_out)
                if rem_err:
                    stderr_chunks.append(rem_err)
                self.fail(f"Process hung after receiving signal {sig}: {args}")
        finally:
            sel.close()

        return subprocess.CompletedProcess(
            args=[str(self.binary_path), *args],
            returncode=proc.returncode,
            stdout="".join(stdout_chunks),
            stderr="".join(stderr_chunks),
        )

    def assert_run_succeeded(self, proc: subprocess.CompletedProcess[str], context: str = "") -> None:
        if proc.returncode != 0:
            self.fail(
                f"Expected exit 0 ({context}), got {proc.returncode}\n"
                f"cmd: {proc.args}\n"
                f"stdout tail:\n{proc.stdout[-3000:]}\n"
                f"stderr tail:\n{proc.stderr[-3000:]}"
            )

    def assert_run_rejected(self, proc: subprocess.CompletedProcess[str], context: str = "") -> None:
        self.assertGreater(
            proc.returncode,
            0,
            f"Expected clean non-zero exit without signal crash ({context}), got {proc.returncode}\n"
            f"cmd: {proc.args}\nstdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}",
        )
        self.assertTrue(
            bool(proc.stderr.strip()),
            f"Expected diagnostic error on stderr ({context}), got empty stderr\n"
            f"cmd: {proc.args}\nstdout:\n{proc.stdout[-2000:]}",
        )

    def assert_graceful_interruption(self, proc: subprocess.CompletedProcess[str], context: str = "") -> None:
        self.assertGreaterEqual(
            proc.returncode,
            0,
            f"Expected clean graceful-interruption exit without uncaught signal ({context}), got {proc.returncode}\n"
            f"cmd: {proc.args}\nstdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}",
        )

    def assert_atomic_json_file(self, json_path: Path) -> dict:
        self.assertTrue(json_path.is_file(), f"Expected JSON output file: {json_path}")
        tmp_path = Path(f"{json_path}.tmp")
        self.assertFalse(tmp_path.exists(), f"Leftover atomic temp file exists: {tmp_path}")
        return json.loads(json_path.read_text(encoding="utf-8"))

    def assert_stdout_json_transport(self, proc: subprocess.CompletedProcess[str], cwd: Path) -> dict:
        self.assertFalse((cwd / "-").exists(), "Unexpected literal '-' file created in cwd")
        self.assertFalse((cwd / "-.tmp").exists(), "Unexpected '-.tmp' file created in cwd")
        stdout_text = proc.stdout.strip()
        self.assertTrue(stdout_text, "Expected JSON on stdout for -o -, got empty stdout")
        try:
            doc = json.loads(stdout_text)
        except json.JSONDecodeError as exc:
            self.fail(
                f"Expected pure single JSON document on stdout for -o -, got decode error: {exc}\n"
                f"stdout:\n{proc.stdout[:2000]}\nstderr:\n{proc.stderr[:2000]}"
            )
        self.assertIsInstance(doc, dict)
        self.assertTrue(proc.stderr.strip(), "Expected human log output on stderr for -o -")
        return doc

    def invoke_verifier(
        self,
        verifier_script: Path,
        json_path: Path,
        *,
        expected_exit: int = 0,
        binary_path: Path | None = None,
        include_binary: bool = True,
        require_raw_timing: bool = True,
        extra_args: list[str] | None = None,
    ) -> dict:
        self.assertTrue(verifier_script.is_file(), f"Missing verifier script: {verifier_script}")
        cmd = [sys.executable, str(verifier_script), str(json_path)]
        if include_binary:
            target_bin = binary_path if binary_path is not None else self.binary_path
            cmd.extend(["--binary", str(target_bin)])
        if require_raw_timing:
            cmd.append("--require-raw-timing")
        if extra_args:
            cmd.extend(extra_args)

        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
            timeout=DEFAULT_RUN_TIMEOUT_SEC,
            check=False,
        )
        self.assertEqual(
            proc.returncode,
            expected_exit,
            f"Verifier {verifier_script.name} returned {proc.returncode}, expected {expected_exit}\n"
            f"cmd: {cmd}\nstdout: {proc.stdout}\nstderr: {proc.stderr}",
        )
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        self.assertEqual(
            len(lines),
            1,
            f"Expected single sorted JSON line on verifier stdout, got {len(lines)} lines:\n{proc.stdout}",
        )
        verdict = json.loads(lines[0])
        self.assertIsInstance(verdict, dict)
        for required_key in ("artifact_consistent", "run_accepted", "reason_code"):
            self.assertIn(required_key, verdict, f"Missing verifier key {required_key}: {verdict}")
        if verdict.get("artifact_consistent") is True:
            for detail_key in ("checks", "oracle_work_used"):
                self.assertIn(detail_key, verdict, f"Missing verifier key {detail_key}: {verdict}")
        return verdict

    def verify_bandwidth_json(
        self,
        json_path: Path,
        *,
        expected_exit: int = 0,
        expected_consistent: bool | None = True,
        expected_accepted: bool = True,
        binary_path: Path | None = None,
        include_binary: bool = True,
        require_raw_timing: bool = True,
        extra_args: list[str] | None = None,
    ) -> dict:
        verdict = self.invoke_verifier(
            BANDWIDTH_VERIFIER,
            json_path,
            expected_exit=expected_exit,
            binary_path=binary_path,
            include_binary=include_binary,
            require_raw_timing=require_raw_timing,
            extra_args=extra_args,
        )
        self.assertEqual(verdict["artifact_consistent"], expected_consistent, verdict)
        self.assertEqual(verdict["run_accepted"], expected_accepted, verdict)
        return verdict

    def verify_latency_json(
        self,
        json_path: Path,
        *,
        expected_exit: int = 0,
        expected_consistent: bool | None = True,
        expected_accepted: bool = True,
        binary_path: Path | None = None,
        include_binary: bool = True,
        require_raw_timing: bool = True,
        extra_args: list[str] | None = None,
    ) -> dict:
        verdict = self.invoke_verifier(
            LATENCY_VERIFIER,
            json_path,
            expected_exit=expected_exit,
            binary_path=binary_path,
            include_binary=include_binary,
            require_raw_timing=require_raw_timing,
            extra_args=extra_args,
        )
        self.assertEqual(verdict["artifact_consistent"], expected_consistent, verdict)
        self.assertEqual(verdict["run_accepted"], expected_accepted, verdict)
        return verdict


# =============================================================================
# Tier 1: Feature Coverage (-B, -L, Help, Targets, Chain Modes, Transport)
# =============================================================================


class Tier1CpuBandwidthTests(E2EAcceptanceTestCase):
    """Tier 1 feature coverage for -B / --cpu-bandwidth."""

    def test_t1_bw_help_variants_exit_zero_without_runtime_banner(self) -> None:
        cases = [
            ["-B", "-h"],
            ["--cpu-bandwidth", "--help"],
            ["-B", "-h", "-o", "-"],
            ["--cpu-bandwidth", "--output=-", "--help"],
        ]
        with tempfile.TemporaryDirectory(prefix="e2e_bw_help_") as tmp:
            cwd = Path(tmp)
            for args in cases:
                with self.subTest(args=args):
                    proc = self.run_binary(args, cwd=cwd)
                    self.assert_run_succeeded(proc, f"help {args}")
                    self.assertEqual(proc.stderr.strip(), "", f"Expected empty stderr on help: {proc.stderr}")
                    self.assertIn("Usage:", proc.stdout)
                    self.assertIn("--cpu-bandwidth", proc.stdout)
                    self.assertNotIn("Running", proc.stdout)
                    with self.assertRaises(json.JSONDecodeError):
                        json.loads(proc.stdout)
                    self.assertFalse((cwd / "-").exists())

    def test_t1_bw_auto_calibrated_default_loop_count(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_bw_autocal_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "bw_autocal.json"
            proc = self.run_binary(["-B", "-b", "16", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "bw auto-calibrated")
            doc = self.assert_atomic_json_file(out_path)

            self.assertEqual(doc["schema_version"], 1)
            self.assertEqual(doc["mode"], "cpu_bandwidth")
            self.assertEqual(doc["methodology_version"], BANDWIDTH_METHODOLOGY)
            self.assertEqual(doc["configuration"]["loop_count"], 3)
            self.assertTrue(doc["calibration"]["excluded_from_results"])
            self.assertTrue(doc["calibration"]["attempts"])
            self.assertEqual(doc["status"], "complete")
            self.assertTrue(doc["run_accepted"])
            self.verify_bandwidth_json(out_path)

    def test_t1_bw_explicit_iterations_and_latin_square_order(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_bw_explicit_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "bw_explicit.json"
            proc = self.run_binary(["-B", "-b", "16", "-i", "4", "-r", "3", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "bw explicit")
            doc = self.assert_atomic_json_file(out_path)

            self.assertEqual(doc["configuration"]["iterations"], 4)
            self.assertEqual(doc["configuration"]["loop_count"], 3)
            self.assertTrue(doc["operation_order_balance_complete"])

            comp = doc["resolved_plan"]["component_identities"]
            self.assertEqual(comp["version"], "cpu-bandwidth-components-v1")
            self.assertEqual(comp["work_plan_version"], "cpu-bandwidth-work-plan-v1")
            self.assertEqual(comp["buffer_pattern_version"], "cpu-bw-buffer-pattern-v1")
            self.assertEqual(comp["read_checksum_version"], "cpu-bw-read-checksum-v1")

            # Verify balanced 3-op Latin-square order per target across 3 loops.
            by_target_loop: dict[str, dict[int, dict[int, str]]] = {}
            for row in doc["measurements"]:
                by_target_loop.setdefault(row["target"], {}).setdefault(row["loop_index"], {})[
                    row["operation_order_position"]
                ] = row["operation"]
            for target, loops in by_target_loop.items():
                self.assertEqual(len(loops), 3, f"Expected 3 loops for {target}")
                for pos in range(3):
                    pos_ops = {loops[loop_idx][pos] for loop_idx in range(3)}
                    self.assertEqual(
                        pos_ops,
                        {"read", "write", "copy"},
                        f"Latin square violated at pos {pos} for {target}: {loops}",
                    )
            self.verify_bandwidth_json(out_path)

    def test_t1_bw_target_toggles_custom_and_zero_omissions(self) -> None:
        cases = [
            ("main_and_custom", ["-B", "-b", "16", "-k", "64", "-i", "2", "-r", "2"], {"main_memory", "custom"}),
            ("custom_only", ["-B", "-b", "0", "-k", "64", "-i", "2", "-r", "2"], {"custom"}),
            ("main_only", ["-B", "-b", "16", "-k", "0", "-i", "2", "-r", "2"], {"main_memory"}),
        ]
        with tempfile.TemporaryDirectory(prefix="e2e_bw_targets_") as tmp:
            cwd = Path(tmp)
            for name, base_args, expected_targets in cases:
                with self.subTest(case=name):
                    out_path = cwd / f"{name}.json"
                    proc = self.run_binary([*base_args, "-o", str(out_path)], cwd=cwd)
                    self.assert_run_succeeded(proc, name)
                    doc = self.assert_atomic_json_file(out_path)
                    enabled_targets = {
                        t for t, info in doc["resolved_plan"]["targets"].items() if info.get("enabled")
                    }
                    self.assertEqual(enabled_targets, expected_targets)
                    self.verify_bandwidth_json(out_path)

    def test_t1_bw_threads_non_cacheable_and_seeds(self) -> None:
        cases = [
            ("threads_2", ["-B", "-b", "16", "-k", "0", "-i", "2", "-r", "1", "-t", "2"]),
            ("non_cacheable", ["-B", "-b", "16", "-k", "0", "-i", "2", "-r", "1", "-u"]),
            ("seed_zero", ["-B", "-b", "16", "-k", "0", "-i", "2", "-r", "1", "--seed", "0"]),
            ("seed_424242", ["-B", "-b", "16", "-k", "0", "-i", "2", "-r", "1", "--seed", "424242"]),
        ]
        with tempfile.TemporaryDirectory(prefix="e2e_bw_flags_") as tmp:
            cwd = Path(tmp)
            for name, base_args in cases:
                with self.subTest(case=name):
                    out_path = cwd / f"{name}.json"
                    proc = self.run_binary([*base_args, "-o", str(out_path)], cwd=cwd)
                    self.assert_run_succeeded(proc, name)
                    doc = self.assert_atomic_json_file(out_path)
                    if name == "threads_2":
                        self.assertEqual(doc["configuration"]["requested_workers"], 2)
                    elif name == "non_cacheable":
                        self.assertTrue(doc["configuration"]["non_cacheable"])
                    elif name == "seed_zero":
                        self.assertEqual(doc["configuration"]["base_seed_uint64_decimal"], "0")
                        self.assertEqual(doc["seeds"]["source"], "user")
                    elif name == "seed_424242":
                        self.assertEqual(doc["configuration"]["base_seed_uint64_decimal"], "424242")
                        self.assertEqual(doc["seeds"]["source"], "user")
                    self.verify_bandwidth_json(out_path)

    def test_t1_bw_transports_stdout_console_only_and_inline_equals(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_bw_transport_") as tmp:
            cwd = Path(tmp)

            # 1. -o - pure JSON stdout + human stderr
            proc_stdout = self.run_binary(["-B", "-b", "8", "-k", "0", "-i", "2", "-r", "1", "-o", "-"], cwd=cwd)
            self.assert_run_succeeded(proc_stdout, "bw -o -")
            doc_stdout = self.assert_stdout_json_transport(proc_stdout, cwd)
            self.assertEqual(doc_stdout["configuration"]["output_file"], "-")
            stdout_copy = cwd / "captured_bw_stdout.json"
            stdout_copy.write_text(proc_stdout.stdout, encoding="utf-8")
            self.verify_bandwidth_json(stdout_copy)

            # 2. Omitted -o (console-only)
            proc_console = self.run_binary(["-B", "-b", "8", "-k", "0", "-i", "2", "-r", "1"], cwd=cwd)
            self.assert_run_succeeded(proc_console, "bw console-only")
            self.assertTrue((proc_console.stdout + proc_console.stderr).strip())

            # 3. --option=value inline syntax
            inline_out = cwd / "bw_inline.json"
            proc_inline = self.run_binary(
                [
                    "--cpu-bandwidth",
                    "--buffer-size=8",
                    "--cache-size=0",
                    "--iterations=2",
                    "--count=1",
                    "--threads=2",
                    "--seed=424242",
                    f"--output={inline_out}",
                ],
                cwd=cwd,
            )
            self.assert_run_succeeded(proc_inline, "bw inline equals")
            doc_inline = self.assert_atomic_json_file(inline_out)
            self.assertEqual(doc_inline["configuration"]["base_seed_uint64_decimal"], "424242")
            self.assertEqual(doc_inline["configuration"]["requested_workers"], 2)
            self.verify_bandwidth_json(inline_out)


class Tier1CpuLatencyTests(E2EAcceptanceTestCase):
    """Tier 1 feature coverage for -L / --cpu-latency."""

    def test_t1_lat_help_variants_exit_zero_without_runtime_banner(self) -> None:
        cases = [
            ["-L", "-h"],
            ["--cpu-latency", "--help"],
            ["-L", "-h", "-o", "-"],
            ["--cpu-latency", "--output=-", "--help"],
        ]
        with tempfile.TemporaryDirectory(prefix="e2e_lat_help_") as tmp:
            cwd = Path(tmp)
            for args in cases:
                with self.subTest(args=args):
                    proc = self.run_binary(args, cwd=cwd)
                    self.assert_run_succeeded(proc, f"help {args}")
                    self.assertEqual(proc.stderr.strip(), "", f"Expected empty stderr on help: {proc.stderr}")
                    self.assertIn("Usage:", proc.stdout)
                    self.assertIn("--cpu-latency", proc.stdout)
                    self.assertNotIn("Running", proc.stdout)
                    with self.assertRaises(json.JSONDecodeError):
                        json.loads(proc.stdout)
                    self.assertFalse((cwd / "-").exists())

    def test_t1_lat_auto_calibrated_and_explicit(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_lat_cal_") as tmp:
            cwd = Path(tmp)

            # Auto-calibrated
            auto_out = cwd / "lat_auto.json"
            proc_auto = self.run_binary(["-L", "-b", "16", "-o", str(auto_out)], cwd=cwd)
            self.assert_run_succeeded(proc_auto, "lat auto-calibrated")
            doc_auto = self.assert_atomic_json_file(auto_out)
            self.assertEqual(doc_auto["schema_version"], 1)
            self.assertEqual(doc_auto["mode"], "cpu_latency")
            self.assertEqual(doc_auto["methodology_version"], LATENCY_METHODOLOGY)
            self.assertEqual(doc_auto["configuration"]["loop_count"], 3)
            self.assertTrue(doc_auto["calibration"]["excluded_from_results"])
            self.assertTrue(doc_auto["calibration"]["attempts"])
            comp = doc_auto["resolved_plan"]["component_identities"]
            self.assertEqual(comp["version"], "cpu-latency-components-v1")
            self.assertEqual(comp["work_plan_version"], "cpu-latency-work-plan-v1")
            self.assertEqual(comp["chain_generator_version"], "splitmix64-fisher-yates-rejection-v1")
            self.assertEqual(comp["chain_version"], "cpu-latency-chain-v1")
            self.verify_latency_json(auto_out)

            # Explicit iterations & count
            exp_out = cwd / "lat_explicit.json"
            proc_exp = self.run_binary(["-L", "-b", "16", "-i", "2", "-r", "3", "-n", "20", "-o", str(exp_out)], cwd=cwd)
            self.assert_run_succeeded(proc_exp, "lat explicit")
            doc_exp = self.assert_atomic_json_file(exp_out)
            self.assertEqual(doc_exp["configuration"]["iterations"], 2)
            self.assertEqual(doc_exp["configuration"]["loop_count"], 3)
            self.verify_latency_json(exp_out)

    def test_t1_lat_sample_counts_zero_and_fifty(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_lat_samples_") as tmp:
            cwd = Path(tmp)

            # -n 0 -> sample_windows is None and pooled_sample_distribution is None
            n0_out = cwd / "lat_n0.json"
            proc_n0 = self.run_binary(["-L", "-b", "16", "-k", "0", "-i", "2", "-r", "2", "-n", "0", "-o", str(n0_out)], cwd=cwd)
            self.assert_run_succeeded(proc_n0, "lat -n 0")
            doc_n0 = self.assert_atomic_json_file(n0_out)
            self.assertEqual(doc_n0["configuration"]["latency_samples"], 0)
            for row in doc_n0["measurements"]:
                self.assertIsNone(row["sample_windows"])
            for target_agg in doc_n0["aggregates"]["targets"].values():
                self.assertIsNone(target_agg["pooled_sample_distribution"])
            self.verify_latency_json(n0_out)

            # -n 50 -> sample_windows and pooled_sample_distribution populated
            n50_out = cwd / "lat_n50.json"
            proc_n50 = self.run_binary(["-L", "-b", "16", "-k", "0", "-i", "2", "-r", "2", "-n", "50", "-o", str(n50_out)], cwd=cwd)
            self.assert_run_succeeded(proc_n50, "lat -n 50")
            doc_n50 = self.assert_atomic_json_file(n50_out)
            self.assertEqual(doc_n50["configuration"]["latency_samples"], 50)
            for row in doc_n50["measurements"]:
                self.assertIsInstance(row["sample_windows"], dict)
                self.assertEqual(len(row["sample_windows"]["samples_ns"]), 50)
                self.assertEqual(len(row["sample_windows"]["sample_delta_ticks"]), 50)
            for target_agg in doc_n50["aggregates"]["targets"].values():
                self.assertIsInstance(target_agg["pooled_sample_distribution"], dict)
            self.verify_latency_json(n50_out)

    def test_t1_lat_target_toggles_non_cacheable_and_seed_zero(self) -> None:
        cases = [
            ("main_and_custom", ["-L", "-b", "16", "-k", "64", "-i", "2", "-r", "2", "-n", "10"], {"main_memory", "custom"}),
            ("custom_only", ["-L", "-b", "0", "-k", "64", "-i", "2", "-r", "2", "-n", "10"], {"custom"}),
            ("main_only", ["-L", "-b", "16", "-k", "0", "-i", "2", "-r", "2", "-n", "10"], {"main_memory"}),
            ("non_cacheable_seed0", ["-L", "-b", "16", "-k", "0", "-i", "2", "-r", "1", "-n", "10", "-u", "--seed", "0"], {"main_memory"}),
        ]
        with tempfile.TemporaryDirectory(prefix="e2e_lat_targets_") as tmp:
            cwd = Path(tmp)
            for name, base_args, expected_targets in cases:
                with self.subTest(case=name):
                    out_path = cwd / f"{name}.json"
                    proc = self.run_binary([*base_args, "-o", str(out_path)], cwd=cwd)
                    self.assert_run_succeeded(proc, name)
                    doc = self.assert_atomic_json_file(out_path)
                    enabled = {t for t, info in doc["resolved_plan"]["targets"].items() if info.get("enabled")}
                    self.assertEqual(enabled, expected_targets)
                    if name == "non_cacheable_seed0":
                        self.assertTrue(doc["configuration"]["non_cacheable"])
                        self.assertEqual(doc["configuration"]["base_seed_uint64_decimal"], "0")
                        self.assertEqual(doc["seeds"]["source"], "user")
                    self.verify_latency_json(out_path)

    def test_t1_lat_all_chain_modes_and_normalized_aliases(self) -> None:
        mode_tokens = [
            "auto",
            "global-random",
            "random-box",
            "same-random-in-box",
            "diff-random-in-box",
            "global",
            "random_in_box_random_box",
            "same_random_in_box_increasing_box",
            "diff_random_in_box_increasing_box",
        ]
        with tempfile.TemporaryDirectory(prefix="e2e_lat_modes_") as tmp:
            cwd = Path(tmp)
            for token in mode_tokens:
                with self.subTest(chain_mode=token):
                    out_path = cwd / f"lat_mode_{token}.json"
                    proc = self.run_binary(
                        [
                            "-L",
                            "-b",
                            "0",
                            "-k",
                            "64",
                            "-i",
                            "2",
                            "-r",
                            "1",
                            "-n",
                            "10",
                            "-s",
                            "64",
                            "-l",
                            "16",
                            "-m",
                            token,
                            "--seed",
                            "424242",
                            "-o",
                            str(out_path),
                        ],
                        cwd=cwd,
                    )
                    self.assert_run_succeeded(proc, f"chain mode {token}")
                    self.assert_atomic_json_file(out_path)
                    self.verify_latency_json(out_path)

    def test_t1_lat_locality_zero_and_stride_variants(self) -> None:
        cases = [
            ("locality_zero", ["-L", "-b", "0", "-k", "64", "-i", "2", "-r", "1", "-n", "10", "-l", "0"]),
            ("stride_64", ["-L", "-b", "0", "-k", "64", "-i", "2", "-r", "1", "-n", "10", "-s", "64"]),
            ("stride_512", ["-L", "-b", "0", "-k", "64", "-i", "2", "-r", "1", "-n", "10", "-s", "512"]),
        ]
        with tempfile.TemporaryDirectory(prefix="e2e_lat_stride_") as tmp:
            cwd = Path(tmp)
            for name, base_args in cases:
                with self.subTest(case=name):
                    out_path = cwd / f"{name}.json"
                    proc = self.run_binary([*base_args, "-o", str(out_path)], cwd=cwd)
                    self.assert_run_succeeded(proc, name)
                    doc = self.assert_atomic_json_file(out_path)
                    if name == "locality_zero":
                        self.assertEqual(doc["configuration"]["latency_tlb_locality_kb"], 0)
                    elif name == "stride_64":
                        self.assertEqual(doc["configuration"]["latency_stride_bytes"], 64)
                    elif name == "stride_512":
                        self.assertEqual(doc["configuration"]["latency_stride_bytes"], 512)
                    self.verify_latency_json(out_path)

    def test_t1_lat_compare_locality_paired_rounds(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_lat_compare_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "lat_compare_locality.json"
            proc = self.run_binary(
                ["-L", "-b", "16", "-k", "0", "-i", "2", "-r", "3", "-n", "10", "--compare-locality", "-o", str(out_path)],
                cwd=cwd,
            )
            self.assert_run_succeeded(proc, "lat --compare-locality")
            doc = self.assert_atomic_json_file(out_path)
            self.assertTrue(doc["configuration"]["compare_locality"])
            self.assertIsNotNone(doc["aggregates"]["locality_comparison"])
            self.verify_latency_json(out_path)

    def test_t1_lat_transports_stdout_console_only_and_inline_equals(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_lat_transport_") as tmp:
            cwd = Path(tmp)

            # 1. -o - pure JSON stdout + human stderr
            proc_stdout = self.run_binary(["-L", "-b", "0", "-k", "64", "-i", "2", "-r", "1", "-n", "10", "-o", "-"], cwd=cwd)
            self.assert_run_succeeded(proc_stdout, "lat -o -")
            doc_stdout = self.assert_stdout_json_transport(proc_stdout, cwd)
            self.assertEqual(doc_stdout["configuration"]["output_file"], "-")
            stdout_copy = cwd / "captured_lat_stdout.json"
            stdout_copy.write_text(proc_stdout.stdout, encoding="utf-8")
            self.verify_latency_json(stdout_copy)

            # 2. Omitted -o (console-only)
            proc_console = self.run_binary(["-L", "-b", "0", "-k", "64", "-i", "2", "-r", "1", "-n", "10"], cwd=cwd)
            self.assert_run_succeeded(proc_console, "lat console-only")
            self.assertTrue((proc_console.stdout + proc_console.stderr).strip())

            # 3. --option=value inline syntax
            inline_out = cwd / "lat_inline.json"
            proc_inline = self.run_binary(
                [
                    "--cpu-latency",
                    "--buffer-size=0",
                    "--cache-size=64",
                    "--iterations=2",
                    "--count=1",
                    "--latency-samples=10",
                    "--latency-stride-bytes=256",
                    "--latency-chain-mode=random-box",
                    "--latency-tlb-locality-kb=16",
                    "--seed=424242",
                    f"--output={inline_out}",
                ],
                cwd=cwd,
            )
            self.assert_run_succeeded(proc_inline, "lat inline equals")
            doc_inline = self.assert_atomic_json_file(inline_out)
            self.assertEqual(doc_inline["configuration"]["base_seed_uint64_decimal"], "424242")
            self.verify_latency_json(inline_out)


# =============================================================================
# Tier 2: Boundary & Sweep Envelope
# =============================================================================


class Tier2BoundaryAndSweepTests(E2EAcceptanceTestCase):
    """Tier 2 boundary values and sweep envelope coverage for -B and -L."""

    def test_t2_bw_single_axis_sweeps(self) -> None:
        cases = [
            ("buffer_size", ["-B", "-k", "0", "-i", "2", "-r", "1", "-S", "buffer-size=8,16"]),
            ("cache_size", ["-B", "-b", "0", "-i", "2", "-r", "1", "-S", "cache-size=32,64"]),
            ("threads", ["-B", "-b", "8", "-k", "0", "-i", "2", "-r", "1", "-S", "threads=1,2"]),
        ]
        with tempfile.TemporaryDirectory(prefix="e2e_t2_bw_single_") as tmp:
            cwd = Path(tmp)
            for name, base_args in cases:
                with self.subTest(axis=name):
                    out_path = cwd / f"bw_sweep_{name}.json"
                    proc = self.run_binary([*base_args, "-o", str(out_path)], cwd=cwd)
                    self.assert_run_succeeded(proc, f"bw sweep {name}")
                    doc = self.assert_atomic_json_file(out_path)
                    self.assertEqual(doc["configuration"]["mode"], "sweep")
                    self.assertEqual(doc["configuration"]["base_mode"], "cpu_bandwidth")
                    self.assertEqual(doc["configuration"]["sweep_schema_version"], 1)
                    self.assertEqual(doc["configuration"]["run_count"], 2)
                    self.assertEqual(doc["planned_runs"], 2)
                    self.assertEqual(doc["completed_runs"], 2)
                    self.verify_bandwidth_json(out_path)

    def test_t2_bw_multi_axis_cartesian_sweep_and_stdout_transport(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_t2_bw_cartesian_") as tmp:
            cwd = Path(tmp)

            # File checkpointing multi-axis Cartesian sweep
            file_out = cwd / "bw_cartesian.json"
            proc_file = self.run_binary(
                [
                    "-B",
                    "-k",
                    "0",
                    "-i",
                    "2",
                    "-r",
                    "1",
                    "-S",
                    "buffer-size=8,16",
                    "-S",
                    "threads=1,2",
                    "-X",
                    "4",
                    "-o",
                    str(file_out),
                ],
                cwd=cwd,
            )
            self.assert_run_succeeded(proc_file, "bw cartesian file")
            doc = self.assert_atomic_json_file(file_out)
            self.assertEqual(doc["configuration"]["run_count"], 4)
            self.assertEqual(doc["planned_runs"], 4)
            self.assertEqual(doc["completed_runs"], 4)
            self.assertEqual(len(doc["runs"]), 4)
            self.assertEqual(doc["status"], "complete")
            self.assertTrue(doc["conclusions_valid"])
            self.verify_bandwidth_json(file_out)

            # Stdout final-only JSON sweep transport
            proc_stdout = self.run_binary(
                ["-B", "-k", "0", "-i", "2", "-r", "1", "-S", "buffer-size=8,16", "-X", "2", "-o", "-"],
                cwd=cwd,
            )
            self.assert_run_succeeded(proc_stdout, "bw sweep -o -")
            doc_stdout = self.assert_stdout_json_transport(proc_stdout, cwd)
            self.assertEqual(doc_stdout["configuration"]["mode"], "sweep")
            self.assertEqual(doc_stdout["completed_runs"], 2)
            stdout_copy = cwd / "bw_sweep_stdout.json"
            stdout_copy.write_text(proc_stdout.stdout, encoding="utf-8")
            self.verify_bandwidth_json(stdout_copy)

    def test_t2_lat_all_five_sweep_keys(self) -> None:
        cases = [
            ("buffer-size", ["-L", "-k", "0", "-i", "2", "-r", "1", "-n", "5", "-S", "buffer-size=8,16"]),
            ("cache-size", ["-L", "-b", "0", "-i", "2", "-r", "1", "-n", "5", "-S", "cache-size=32,64"]),
            (
                "latency-tlb-locality-kb",
                ["-L", "-b", "16", "-k", "0", "-i", "2", "-r", "1", "-n", "5", "-S", "latency-tlb-locality-kb=16,64"],
            ),
            (
                "latency-stride-bytes",
                ["-L", "-b", "0", "-k", "64", "-i", "2", "-r", "1", "-n", "5", "-S", "latency-stride-bytes=64,256"],
            ),
            (
                "latency-chain-mode",
                [
                    "-L",
                    "-b",
                    "0",
                    "-k",
                    "64",
                    "-l",
                    "16",
                    "-i",
                    "2",
                    "-r",
                    "1",
                    "-n",
                    "5",
                    "-S",
                    "latency-chain-mode=global-random,random-box",
                ],
            ),
        ]
        with tempfile.TemporaryDirectory(prefix="e2e_t2_lat_sweeps_") as tmp:
            cwd = Path(tmp)
            for key_name, base_args in cases:
                with self.subTest(sweep_key=key_name):
                    out_path = cwd / f"lat_sweep_{key_name}.json"
                    proc = self.run_binary([*base_args, "-o", str(out_path)], cwd=cwd)
                    self.assert_run_succeeded(proc, f"lat sweep {key_name}")
                    doc = self.assert_atomic_json_file(out_path)
                    self.assertEqual(doc["configuration"]["mode"], "sweep")
                    self.assertEqual(doc["configuration"]["base_mode"], "cpu_latency")
                    self.assertEqual(doc["configuration"]["sweep_schema_version"], 1)
                    self.assertEqual(doc["configuration"]["run_count"], 2)
                    self.assertEqual(doc["planned_runs"], 2)
                    self.assertEqual(doc["completed_runs"], 2)
                    self.verify_latency_json(out_path)

    def test_t2_lat_multi_axis_sweep_and_stdout_transport(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_t2_lat_cartesian_") as tmp:
            cwd = Path(tmp)

            file_out = cwd / "lat_cartesian.json"
            proc_file = self.run_binary(
                [
                    "-L",
                    "-k",
                    "0",
                    "-i",
                    "2",
                    "-r",
                    "1",
                    "-n",
                    "5",
                    "-S",
                    "buffer-size=8,16",
                    "-S",
                    "latency-stride-bytes=64,256",
                    "-X",
                    "4",
                    "-o",
                    str(file_out),
                ],
                cwd=cwd,
            )
            self.assert_run_succeeded(proc_file, "lat cartesian file")
            doc = self.assert_atomic_json_file(file_out)
            self.assertEqual(doc["configuration"]["run_count"], 4)
            self.assertEqual(doc["planned_runs"], 4)
            self.assertEqual(doc["completed_runs"], 4)
            self.verify_latency_json(file_out)

            proc_stdout = self.run_binary(
                ["-L", "-b", "0", "-i", "2", "-r", "1", "-n", "5", "-S", "cache-size=32,64", "-X", "2", "-o", "-"],
                cwd=cwd,
            )
            self.assert_run_succeeded(proc_stdout, "lat sweep -o -")
            doc_stdout = self.assert_stdout_json_transport(proc_stdout, cwd)
            self.assertEqual(doc_stdout["configuration"]["mode"], "sweep")
            self.assertEqual(doc_stdout["completed_runs"], 2)
            stdout_copy = cwd / "lat_sweep_stdout.json"
            stdout_copy.write_text(proc_stdout.stdout, encoding="utf-8")
            self.verify_latency_json(stdout_copy)

    def test_t2_bw_boundary_values_and_thread_clamping(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_t2_bw_bounds_") as tmp:
            cwd = Path(tmp)

            # Min cache size 16 KB, min iterations 1, min count 1, seed 0
            min_out = cwd / "bw_min.json"
            proc_min = self.run_binary(
                ["-B", "-b", "0", "-k", "16", "-i", "1", "-r", "1", "--seed", "0", "-o", str(min_out)],
                cwd=cwd,
            )
            self.assert_run_succeeded(proc_min, "bw min bounds")
            doc_min = self.assert_atomic_json_file(min_out)
            self.assertEqual(doc_min["configuration"]["custom_cache_size_kb"], 16)
            self.assertEqual(doc_min["configuration"]["iterations"], 1)
            self.assertEqual(doc_min["configuration"]["loop_count"], 1)
            self.verify_bandwidth_json(min_out)

            # Excessive threads -t 9999 preserves requested_workers != effective_workers
            clamp_out = cwd / "bw_clamp_threads.json"
            proc_clamp = self.run_binary(
                ["-B", "-b", "16", "-k", "0", "-i", "1", "-r", "1", "-t", "9999", "-o", str(clamp_out)],
                cwd=cwd,
            )
            self.assert_run_succeeded(proc_clamp, "bw -t 9999")
            doc_clamp = self.assert_atomic_json_file(clamp_out)
            self.assertEqual(doc_clamp["configuration"]["requested_workers"], 9999)
            effective_workers = [
                info["effective_workers"]
                for info in doc_clamp["resolved_plan"]["targets"].values()
                if info.get("enabled")
            ]
            self.assertTrue(effective_workers)
            for ew in effective_workers:
                self.assertNotEqual(ew, 9999)
                self.assertGreater(ew, 0)
            self.verify_bandwidth_json(clamp_out)

    def test_t2_lat_boundary_values(self) -> None:
        cases = [
            ("min_k16_i1_r1_n0_s8_l0_seed0", ["-L", "-b", "0", "-k", "16", "-i", "1", "-r", "1", "-n", "0", "-s", "8", "-l", "0", "--seed", "0"]),
            ("min_n1_s8", ["-L", "-b", "0", "-k", "16", "-i", "1", "-r", "1", "-n", "1", "-s", "8", "-l", "0"]),
            ("min_locality_k16_l16_s8", ["-L", "-b", "0", "-k", "16", "-i", "1", "-r", "1", "-n", "5", "-s", "8", "-l", "16", "-m", "random-box"]),
        ]
        with tempfile.TemporaryDirectory(prefix="e2e_t2_lat_bounds_") as tmp:
            cwd = Path(tmp)
            for name, base_args in cases:
                with self.subTest(case=name):
                    out_path = cwd / f"{name}.json"
                    proc = self.run_binary([*base_args, "-o", str(out_path)], cwd=cwd)
                    self.assert_run_succeeded(proc, name)
                    self.assert_atomic_json_file(out_path)
                    self.verify_latency_json(out_path)


# =============================================================================
# Tier 3: Negative / Invalid-Input & Budget Rejections
# =============================================================================


class Tier3NegativeAndBudgetTests(E2EAcceptanceTestCase):
    """Tier 3 negative, mutual-exclusion, legacy-removal, bounds, sweep, and budget rejections."""

    def test_t3_mutual_exclusion_and_removed_legacy_flags(self) -> None:
        cases = [
            # Primary mode mutual exclusions
            ["-B", "-L"],
            ["--cpu-bandwidth", "--cpu-latency"],
            ["-B", "-M"],
            ["-L", "-G"],
            ["-B", "-P"],
            ["-L", "-T"],
            ["-B", "-C"],
            # Legacy removed flags
            ["--benchmark"],
            ["-W"],
            ["--only-bandwidth"],
            ["--only-latency"],
            ["-B", "-W"],
            ["-B", "--only-bandwidth"],
            ["-L", "--only-latency"],
            ["--benchmark", "-b", "16"],
        ]
        with tempfile.TemporaryDirectory(prefix="e2e_t3_mutex_") as tmp:
            cwd = Path(tmp)
            for args in cases:
                with self.subTest(args=args):
                    proc = self.run_binary(args, cwd=cwd)
                    self.assert_run_rejected(proc, f"mutex/legacy {args}")

    def test_t3_cross_mode_options_and_target_omission(self) -> None:
        cases = [
            # Cross-mode options rejected on -B
            ["-B", "-n", "10"],
            ["-B", "--latency-samples", "10"],
            ["-B", "-s", "64"],
            ["-B", "--latency-stride-bytes", "64"],
            ["-B", "-m", "auto"],
            ["-B", "--latency-chain-mode", "global-random"],
            ["-B", "-l", "64"],
            ["-B", "--latency-tlb-locality-kb", "64"],
            ["-B", "--compare-locality"],
            # Cross-mode options rejected on -L
            ["-L", "-t", "2"],
            ["-L", "--threads", "2"],
            # Target omission (both buffer-size and cache-size disabled)
            ["-B", "-b", "0", "-k", "0"],
            ["-L", "-b", "0", "-k", "0"],
        ]
        with tempfile.TemporaryDirectory(prefix="e2e_t3_cross_") as tmp:
            cwd = Path(tmp)
            for args in cases:
                with self.subTest(args=args):
                    proc = self.run_binary(args, cwd=cwd)
                    self.assert_run_rejected(proc, f"cross/omission {args}")

            # Sweep yielding both buffer-size=0 and cache-size=0
            for mode_flag, sweep_args in (
                ("-B", ["-k", "0", "-S", "buffer-size=0,16"]),
                ("-L", ["-b", "0", "-S", "cache-size=0,64"]),
            ):
                with self.subTest(sweep_omission=mode_flag):
                    out_path = cwd / f"omit_{mode_flag.strip('-')}.json"
                    proc = self.run_binary([mode_flag, *sweep_args, "-o", str(out_path)], cwd=cwd)
                    self.assert_run_rejected(proc, f"sweep target omission {mode_flag}")

    def test_t3_invalid_bounds_and_cli_syntax(self) -> None:
        cases = [
            # Numeric bounds on -B
            ["-B", "-b", "-1"],
            ["-B", "-b", "abc"],
            ["-B", "-k", "15"],
            ["-B", "-k", "1048577"],
            ["-B", "-i", "0"],
            ["-B", "-i", "1000000001"],
            ["-B", "-r", "0"],
            ["-B", "-t", "0"],
            ["-B", "--seed", "-1"],
            ["-B", "--seed", "+1"],
            # Numeric bounds & domain rules on -L
            ["-L", "-b", "-1"],
            ["-L", "-b", "abc"],
            ["-L", "-k", "15"],
            ["-L", "-k", "1048577"],
            ["-L", "-i", "0"],
            ["-L", "-i", "1000000001"],
            ["-L", "-r", "0"],
            ["-L", "--seed", "-1"],
            ["-L", "--seed", "+1"],
            ["-L", "-n", "-1"],
            ["-L", "-s", "0"],
            ["-L", "-s", "65"],
            ["-L", "-l", "1"],
            ["-L", "-m", "random-box", "-l", "0"],
            ["-L", "-l", "16", "-s", "16384"],
            ["-L", "--compare-locality", "-b", "0", "-k", "64"],
            ["-L", "--compare-locality", "-b", "16", "-s", "16384"],
            # CLI syntax errors (single-dash inline, duplicates, unknowns, missing value)
            ["-B", "-b=16"],
            ["-L", "-b=16"],
            ["-B", "-b", "16", "-b", "16"],
            ["-L", "-n", "10", "-n", "20"],
            ["-B", "--unknown-option"],
            ["-L", "--unknown-option"],
            ["-B", "-b"],
            ["-L", "-s"],
            ["-B", "-o"],
        ]
        with tempfile.TemporaryDirectory(prefix="e2e_t3_bounds_") as tmp:
            cwd = Path(tmp)
            for args in cases:
                with self.subTest(args=args):
                    proc = self.run_binary(args, cwd=cwd)
                    self.assert_run_rejected(proc, f"invalid bounds/syntax {args}")

    def test_t3_sweep_validation_errors(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_t3_sweep_err_") as tmp:
            cwd = Path(tmp)
            dummy_out = cwd / "should_not_succeed.json"
            cases = [
                # Missing -o
                ["-B", "-S", "buffer-size=8,16"],
                ["-L", "-S", "buffer-size=8,16"],
                # Direct option conflict
                ["-B", "-b", "8", "-S", "buffer-size=8,16", "-o", str(dummy_out)],
                ["-L", "-s", "64", "-S", "latency-stride-bytes=64,256", "-o", str(dummy_out)],
                # Duplicate -S key
                ["-B", "-S", "buffer-size=8", "-S", "buffer-size=16", "-o", str(dummy_out)],
                ["-L", "-S", "cache-size=32", "-S", "cache-size=64", "-o", str(dummy_out)],
                # Exceeds -X
                ["-B", "-S", "buffer-size=8,16", "-S", "threads=1,2", "-X", "3", "-o", str(dummy_out)],
                ["-L", "-S", "buffer-size=8,16", "-S", "cache-size=32,64", "-X", "2", "-o", str(dummy_out)],
                # Empty value
                ["-B", "-S", "buffer-size=", "-o", str(dummy_out)],
                ["-B", "-S", "buffer-size=8,,16", "-o", str(dummy_out)],
                ["-L", "-S", "cache-size=,32", "-o", str(dummy_out)],
                # Disallowed sweep key
                ["-B", "-S", "latency-stride-bytes=64,256", "-o", str(dummy_out)],
                ["-B", "-S", "tlb-density=low,high", "-o", str(dummy_out)],
                ["-L", "-S", "threads=1,2", "-o", str(dummy_out)],
            ]
            for args in cases:
                with self.subTest(args=args):
                    if dummy_out.exists():
                        dummy_out.unlink()
                    proc = self.run_binary(args, cwd=cwd)
                    self.assert_run_rejected(proc, f"sweep error {args}")

    def test_t3_memory_budget_overflow_fails_without_silent_capping(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_t3_budget_") as tmp:
            cwd = Path(tmp)
            for mode_flag, verifier_fn in (("-B", self.verify_bandwidth_json), ("-L", self.verify_latency_json)):
                with self.subTest(mode=mode_flag):
                    out_path = cwd / f"budget_overflow_{mode_flag.strip('-')}.json"
                    proc = self.run_binary(
                        [mode_flag, "-b", "999999999", "-k", "0", "-i", "1", "-r", "1", "-o", str(out_path)],
                        cwd=cwd,
                    )
                    self.assert_run_rejected(proc, f"memory_budget overflow {mode_flag}")
                    self.assertTrue(out_path.exists(), f"{mode_flag} preflight rejection must emit JSON to -o file")
                    doc = self.assert_atomic_json_file(out_path)
                    self.assertEqual(doc["status"], "failed")
                    self.assertFalse(doc["run_accepted"])
                    self.assertFalse(doc["memory_budget"]["valid"])
                    verifier_fn(
                        out_path,
                        expected_exit=1,
                        expected_consistent=True,
                        expected_accepted=False,
                        require_raw_timing=False,
                    )

                    stdout_proc = self.run_binary(
                        [mode_flag, "-b", "999999999", "-k", "0", "-i", "1", "-r", "1", "-o", "-"],
                        cwd=cwd,
                    )
                    self.assert_run_rejected(stdout_proc, f"memory_budget overflow stdout {mode_flag}")
                    stdout_doc = json.loads(stdout_proc.stdout)
                    self.assertEqual(stdout_doc["status"], "failed")
                    self.assertFalse(stdout_doc["run_accepted"])
                    self.assertFalse(stdout_doc["memory_budget"]["valid"])


# =============================================================================
# Tier 4: Fault-Tolerance, Tamper/Exit-Code, Determinism & Sibling Regression
# =============================================================================


class Tier4FaultToleranceDeterminismAndRegressionTests(E2EAcceptanceTestCase):
    """Tier 4 signal interruption, live tamper/exit-code checks, determinism, and sibling modes."""

    def test_t4_sigint_and_sigterm_mid_run_interruption(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_t4_signals_") as tmp:
            cwd = Path(tmp)
            cases = [
                (
                    "bw_direct_sigint",
                    ["-B", "-b", "16", "-k", "0", "-r", "50"],
                    signal.SIGINT,
                    self.verify_bandwidth_json,
                ),
                (
                    "lat_direct_sigterm",
                    ["-L", "-b", "16", "-k", "0", "-r", "50"],
                    signal.SIGTERM,
                    self.verify_latency_json,
                ),
                (
                    "bw_sweep_sigint",
                    ["-B", "-k", "0", "-r", "15", "-S", "buffer-size=8,16,32,64", "-X", "4"],
                    signal.SIGINT,
                    self.verify_bandwidth_json,
                ),
                (
                    "lat_sweep_sigterm",
                    ["-L", "-k", "0", "-r", "15", "-S", "buffer-size=8,16,32,64", "-X", "4"],
                    signal.SIGTERM,
                    self.verify_latency_json,
                ),
            ]
            for name, base_args, sig, verifier_fn in cases:
                with self.subTest(case=name):
                    out_path = cwd / f"{name}.json"
                    proc = self.run_binary_and_interrupt(
                        [*base_args, "-o", str(out_path)],
                        cwd=cwd,
                        sig=sig,
                        post_signal_delay=1.0,
                    )
                    self.assert_graceful_interruption(proc, f"interrupted {name}")
                    doc = self.assert_atomic_json_file(out_path)
                    self.assertEqual(doc["status"], "interrupted")
                    if doc["configuration"].get("mode") == "sweep":
                        self.assertFalse(doc["conclusions_valid"])
                    else:
                        self.assertTrue(doc["interruption_requested"])
                        self.assertFalse(doc["run_accepted"])
                    verifier_fn(
                        out_path,
                        expected_exit=1,
                        expected_consistent=True,
                        expected_accepted=False,
                        require_raw_timing=False,
                    )

    def test_t4_live_bandwidth_artifact_tamper_and_verifier_gates(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_t4_tamper_bw_") as tmp:
            cwd = Path(tmp)
            live_out = cwd / "live_bw.json"
            proc = self.run_binary(["-B", "-b", "16", "-k", "0", "-i", "2", "-r", "2", "-o", str(live_out)], cwd=cwd)
            self.assert_run_succeeded(proc, "live bw for tamper")
            base_doc = self.assert_atomic_json_file(live_out)
            self.verify_bandwidth_json(live_out)

            # 1. --binary mismatch -> exit 1
            mismatch_verdict = self.verify_bandwidth_json(
                live_out,
                expected_exit=1,
                expected_consistent=False,
                expected_accepted=False,
                binary_path=Path(sys.executable),
            )
            self.assertEqual(mismatch_verdict["reason_code"], "binary-hash-mismatch")

            # 2. --require-raw-timing on stripped cpu_raw -> exit 2
            stripped = copy.deepcopy(base_doc)
            self._strip_cpu_raw(stripped)
            stripped_path = cwd / "bw_stripped_raw.json"
            stripped_path.write_text(json.dumps(stripped), encoding="utf-8")
            ok_without_raw = self.verify_bandwidth_json(stripped_path, require_raw_timing=False)
            self.assertEqual(ok_without_raw["checks"]["timing"], "elapsed-only")
            reject_with_raw = self.verify_bandwidth_json(
                stripped_path,
                expected_exit=2,
                expected_consistent=None,
                expected_accepted=False,
                require_raw_timing=True,
            )
            self.assertEqual(reject_with_raw["reason_code"], "unsupported-missing-raw-timing")

            # 3. Tamper run_checksum on a read measurement
            tamper_checksum = copy.deepcopy(base_doc)
            read_row = next(r for r in tamper_checksum["measurements"] if r["operation"] == "read")
            actual_rc = read_row["checksum"]["actual_run_checksum"]
            if isinstance(actual_rc, dict):
                key = next(iter(actual_rc))
                actual_rc[key] = "0" if actual_rc[key] != "0" else "1"
            else:
                read_row["checksum"]["actual_run_checksum"] = "0" if actual_rc != "0" else "1"
            self._assert_bw_tamper_rejected(cwd, "tamper_checksum.json", tamper_checksum)

            # 4. Tamper write_pattern_uint64_decimal on a write measurement
            tamper_wp = copy.deepcopy(base_doc)
            write_row = next(r for r in tamper_wp["measurements"] if r["operation"] == "write")
            write_row["execution"]["write_pattern_uint64_decimal"] = "0"
            self._assert_bw_tamper_rejected(cwd, "tamper_write_pattern.json", tamper_wp)

            # 5. Tamper delta_ticks
            tamper_ticks = copy.deepcopy(base_doc)
            raw = tamper_ticks["measurements"][0]["execution"]["timing"]["cpu_raw"]
            raw["delta_ticks"] = str(int(raw["delta_ticks"]) + 1)
            self._assert_bw_tamper_rejected(cwd, "tamper_ticks.json", tamper_ticks)

            # 6. Tamper validation.checks[]
            tamper_checks = copy.deepcopy(base_doc)
            tamper_checks["measurements"][0]["execution"]["validation"]["checks"][0]["valid"] = False
            self._assert_bw_tamper_rejected(cwd, "tamper_checks.json", tamper_checks)

            # 7. Tamper run_accepted
            tamper_accepted = copy.deepcopy(base_doc)
            tamper_accepted["run_accepted"] = False
            self._assert_bw_tamper_rejected(cwd, "tamper_accepted.json", tamper_accepted)

            # 8. Duplicate JSON key
            dup_path = cwd / "bw_dup_key.json"
            raw_text = live_out.read_text(encoding="utf-8")
            dup_text = raw_text.replace('"mode":', '"schema_version": 1, "mode":', 1)
            self.assertNotEqual(dup_text, raw_text)
            dup_path.write_text(dup_text, encoding="utf-8")
            dup_verdict = self.verify_bandwidth_json(
                dup_path,
                expected_exit=1,
                expected_consistent=False,
                expected_accepted=False,
            )
            self.assertEqual(dup_verdict["reason_code"], "duplicate-json-key")

    def test_t4_live_latency_artifact_tamper_and_verifier_gates(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_t4_tamper_lat_") as tmp:
            cwd = Path(tmp)
            live_out = cwd / "live_lat.json"
            proc = self.run_binary(
                ["-L", "-b", "0", "-k", "64", "-i", "2", "-r", "2", "-n", "10", "-s", "256", "-o", str(live_out)],
                cwd=cwd,
            )
            self.assert_run_succeeded(proc, "live lat for tamper")
            base_doc = self.assert_atomic_json_file(live_out)
            self.verify_latency_json(live_out)

            # 1. --binary mismatch -> exit 1
            mismatch_verdict = self.verify_latency_json(
                live_out,
                expected_exit=1,
                expected_consistent=False,
                expected_accepted=False,
                binary_path=Path(sys.executable),
            )
            self.assertEqual(mismatch_verdict["reason_code"], "binary-hash-mismatch")

            # 2. --require-raw-timing on stripped cpu_raw -> exit 2
            stripped = copy.deepcopy(base_doc)
            self._strip_cpu_raw(stripped)
            stripped_path = cwd / "lat_stripped_raw.json"
            stripped_path.write_text(json.dumps(stripped), encoding="utf-8")
            ok_without_raw = self.verify_latency_json(stripped_path, require_raw_timing=False)
            self.assertEqual(ok_without_raw["checks"]["timing"], "elapsed-only")
            reject_with_raw = self.verify_latency_json(
                stripped_path,
                expected_exit=2,
                expected_consistent=None,
                expected_accepted=False,
                require_raw_timing=True,
            )
            self.assertEqual(reject_with_raw["reason_code"], "unsupported-missing-raw-timing")

            # 3. --max-chain-nodes 100 on 256-node chain -> exit 2
            self.verify_latency_json(
                live_out,
                expected_exit=2,
                expected_consistent=None,
                expected_accepted=False,
                extra_args=["--max-chain-nodes", "100"],
            )

            # 4. Tamper chain_permutation_sha256
            tamper_chain = copy.deepcopy(base_doc)
            first_cid = next(iter(tamper_chain["resolved_plan"]["chain_identities"].values()))
            orig_sha = first_cid["chain_permutation_sha256"]
            first_cid["chain_permutation_sha256"] = ("1" if orig_sha[0] == "0" else "0") + orig_sha[1:]
            self._assert_lat_tamper_rejected(cwd, "tamper_chain_sha.json", tamper_chain)

            # 5. Tamper sample_delta_ticks[0]
            tamper_sample = copy.deepcopy(base_doc)
            sw = tamper_sample["measurements"][0]["sample_windows"]
            sw["sample_delta_ticks"][0] = str(int(sw["sample_delta_ticks"][0]) + 1)
            self._assert_lat_tamper_rejected(cwd, "tamper_sample_ticks.json", tamper_sample)

            # 6. Tamper main delta_ticks
            tamper_ticks = copy.deepcopy(base_doc)
            raw = tamper_ticks["measurements"][0]["execution"]["timing"]["cpu_raw"]
            raw["delta_ticks"] = str(int(raw["delta_ticks"]) + 1)
            self._assert_lat_tamper_rejected(cwd, "tamper_lat_ticks.json", tamper_ticks)

            # 7. Tamper final_pointer_verification
            tamper_ptr = copy.deepcopy(base_doc)
            tamper_ptr["measurements"][0]["final_pointer_verification"]["actual_node"] = 1
            self._assert_lat_tamper_rejected(cwd, "tamper_final_ptr.json", tamper_ptr)

            # 8. Tamper run_accepted
            tamper_accepted = copy.deepcopy(base_doc)
            tamper_accepted["run_accepted"] = False
            self._assert_lat_tamper_rejected(cwd, "tamper_lat_accepted.json", tamper_accepted)

            # 9. Duplicate JSON key
            dup_path = cwd / "lat_dup_key.json"
            raw_text = live_out.read_text(encoding="utf-8")
            dup_text = raw_text.replace('"mode":', '"schema_version": 1, "mode":', 1)
            self.assertNotEqual(dup_text, raw_text)
            dup_path.write_text(dup_text, encoding="utf-8")
            dup_verdict = self.verify_latency_json(
                dup_path,
                expected_exit=1,
                expected_consistent=False,
                expected_accepted=False,
            )
            self.assertEqual(dup_verdict["reason_code"], "duplicate-json-key")

    def test_t4_seed_determinism_bandwidth_and_latency(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_t4_seed_") as tmp:
            cwd = Path(tmp)

            # Bandwidth determinism
            bw1, bw2, bw3 = cwd / "bw1.json", cwd / "bw2.json", cwd / "bw3.json"
            for out_path, seed in ((bw1, "123456"), (bw2, "123456"), (bw3, "654321")):
                proc = self.run_binary(["-B", "-b", "8", "-k", "0", "-i", "3", "-r", "1", "--seed", seed, "-o", str(out_path)], cwd=cwd)
                self.assert_run_succeeded(proc, f"bw seed {seed}")
                self.verify_bandwidth_json(out_path)

            d_bw1 = json.loads(bw1.read_text(encoding="utf-8"))
            d_bw2 = json.loads(bw2.read_text(encoding="utf-8"))
            d_bw3 = json.loads(bw3.read_text(encoding="utf-8"))

            self.assertEqual(d_bw1["seeds"]["buffer_domain_seeds"], d_bw2["seeds"]["buffer_domain_seeds"])
            self.assertNotEqual(d_bw1["seeds"]["buffer_domain_seeds"], d_bw3["seeds"]["buffer_domain_seeds"])
            self.assertEqual(
                [p.get("expected_checksum") for p in d_bw1["resolved_plan"]["operation_plans"]],
                [p.get("expected_checksum") for p in d_bw2["resolved_plan"]["operation_plans"]],
            )
            self.assertNotEqual(
                [p.get("expected_checksum") for p in d_bw1["resolved_plan"]["operation_plans"]],
                [p.get("expected_checksum") for p in d_bw3["resolved_plan"]["operation_plans"]],
            )
            self.assertEqual(
                [m["execution"]["write_pattern_uint64_decimal"] for m in d_bw1["measurements"]],
                [m["execution"]["write_pattern_uint64_decimal"] for m in d_bw2["measurements"]],
            )
            self.assertNotEqual(
                [m["execution"]["write_pattern_uint64_decimal"] for m in d_bw1["measurements"]],
                [m["execution"]["write_pattern_uint64_decimal"] for m in d_bw3["measurements"]],
            )

            # Latency determinism
            lat1, lat2, lat3 = cwd / "lat1.json", cwd / "lat2.json", cwd / "lat3.json"
            for out_path, seed in ((lat1, "123456"), (lat2, "123456"), (lat3, "654321")):
                proc = self.run_binary(["-L", "-b", "0", "-k", "64", "-i", "3", "-r", "1", "-n", "10", "--seed", seed, "-o", str(out_path)], cwd=cwd)
                self.assert_run_succeeded(proc, f"lat seed {seed}")
                self.verify_latency_json(out_path)

            d_lat1 = json.loads(lat1.read_text(encoding="utf-8"))
            d_lat2 = json.loads(lat2.read_text(encoding="utf-8"))
            d_lat3 = json.loads(lat3.read_text(encoding="utf-8"))
            self.assertEqual(d_lat1["resolved_plan"]["chain_identities"], d_lat2["resolved_plan"]["chain_identities"])
            self.assertNotEqual(d_lat1["resolved_plan"]["chain_identities"], d_lat3["resolved_plan"]["chain_identities"])

    def test_t4_sibling_modes_help_and_smoke_non_regression(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_t4_siblings_") as tmp:
            cwd = Path(tmp)

            for flag in ("-P", "-T", "-C", "-G", "-M"):
                with self.subTest(help_flag=flag):
                    proc = self.run_binary([flag, "-h"], cwd=cwd)
                    self.assert_run_succeeded(proc, f"sibling help {flag}")
                    self.assertEqual(proc.stderr.strip(), "")
                    self.assertIn("Usage:", proc.stdout)

            # -P patterns smoke
            p_out = cwd / "patterns.json"
            proc_p = self.run_binary(["-P", "-b", "1", "-i", "1", "-r", "1", "-t", "1", "--seed", "42", "-o", str(p_out)], cwd=cwd)
            self.assert_run_succeeded(proc_p, "sibling -P")
            doc_p = self.assert_atomic_json_file(p_out)
            self.assertEqual(doc_p["schema_version"], 1)
            self.assertEqual(doc_p["mode"], "patterns")
            self.assertTrue(doc_p["run_accepted"])

            # -C core-to-core smoke
            c_out = cwd / "core2core.json"
            proc_c = self.run_binary(["-C", "-r", "1", "-n", "1", "-o", str(c_out)], cwd=cwd)
            self.assert_run_succeeded(proc_c, "sibling -C")
            doc_c = self.assert_atomic_json_file(c_out)
            self.assertEqual(doc_c["configuration"]["mode"], "analyze_core2core")

            # -G GPU bandwidth smoke (accept complete or unsupported if Metal unavailable)
            g_out = cwd / "gpu.json"
            proc_g = self.run_binary(["-G", "-b", "64", "-i", "1", "-r", "1", "--seed", "42", "-o", str(g_out)], cwd=cwd)
            doc_g = self.assert_atomic_json_file(g_out)
            self.assertEqual(doc_g["mode"], "gpu_bandwidth")
            if doc_g["status"] == "complete":
                self.assert_run_succeeded(proc_g, "sibling -G")
            else:
                self.assertEqual(doc_g["status"], "unsupported")

            # -M LLM memory smoke + verify_llm_result.py
            m_out = cwd / "llm.json"
            proc_m = self.run_binary(
                [
                    "-M",
                    "--weight-size-mb",
                    "1",
                    "--layers",
                    "1",
                    "--query-heads",
                    "1",
                    "--kv-heads",
                    "1",
                    "--head-dim",
                    "8",
                    "--kv-element-bytes",
                    "1",
                    "--context-tokens",
                    "2",
                    "--batch-size",
                    "1",
                    "--threads",
                    "1",
                    "--iterations",
                    "1",
                    "--count",
                    "1",
                    "--seed",
                    "42",
                    "-o",
                    str(m_out),
                ],
                cwd=cwd,
            )
            self.assert_run_succeeded(proc_m, "sibling -M")
            doc_m = self.assert_atomic_json_file(m_out)
            self.assertEqual(doc_m["mode"], "llm_memory")
            self.invoke_verifier(LLM_VERIFIER, m_out, expected_exit=0)

            # -T analyze-tlb smoke
            t_out = cwd / "tlb.json"
            proc_t = self.run_binary(["-T", "-D", "low", "--seed", "42", "-o", str(t_out)], cwd=cwd, timeout=TLB_SMOKE_TIMEOUT_SEC)
            self.assert_run_succeeded(proc_t, "sibling -T")
            doc_t = self.assert_atomic_json_file(t_out)
            self.assertEqual(doc_t["configuration"]["mode"], "analyze_tlb")

    @staticmethod
    def _strip_cpu_raw(node: object) -> None:
        if isinstance(node, dict):
            if "cpu_raw" in node:
                node["cpu_raw"] = None
            for val in node.values():
                Tier4FaultToleranceDeterminismAndRegressionTests._strip_cpu_raw(val)
        elif isinstance(node, list):
            for item in node:
                Tier4FaultToleranceDeterminismAndRegressionTests._strip_cpu_raw(item)

    def _assert_bw_tamper_rejected(self, cwd: Path, filename: str, mutated_doc: dict) -> None:
        path = cwd / filename
        path.write_text(json.dumps(mutated_doc), encoding="utf-8")
        self.verify_bandwidth_json(
            path,
            expected_exit=1,
            expected_consistent=False,
            expected_accepted=False,
        )

    def _assert_lat_tamper_rejected(self, cwd: Path, filename: str, mutated_doc: dict) -> None:
        path = cwd / filename
        path.write_text(json.dumps(mutated_doc), encoding="utf-8")
        self.verify_latency_json(
            path,
            expected_exit=1,
            expected_consistent=False,
            expected_accepted=False,
        )


def iter_test_ids(suite: unittest.TestSuite) -> list[str]:
    ids: list[str] = []
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            ids.extend(iter_test_ids(item))
        else:
            ids.append(item.id())
    return ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Tier 1-4 E2E acceptance tests for --cpu-bandwidth (-B) and --cpu-latency (-L)."
    )
    parser.add_argument(
        "--binary",
        default=None,
        help="Path to memory_benchmark binary (default: $MEMORY_BENCHMARK_BIN or ./memory_benchmark)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all discovered Tier 1-4 test IDs and exit 0",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose unittest output",
    )
    args, remaining = parser.parse_known_args(argv)

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])

    if args.list:
        for test_id in iter_test_ids(suite):
            print(test_id)
        return 0

    binary_path = resolve_binary_path(args.binary)
    err = validate_executable_binary(binary_path)
    if err is not None:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1

    os.environ["MEMORY_BENCHMARK_BIN"] = str(binary_path)

    unittest_argv = [sys.argv[0]]
    if args.verbose:
        unittest_argv.append("-v")
    unittest_argv.extend(remaining)
    program = unittest.main(module=__name__, argv=unittest_argv, exit=False)
    return 0 if program.result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

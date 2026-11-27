#!/usr/bin/env python3
# Copyright 2026 Timo Heimonen <timo.heimonen@proton.me>
# SPDX-License-Identifier: GPL-3.0-or-later

"""End-to-end Tier 1-4 acceptance test suite for --patterns (-P).

Usage:
  python3 tests/test_patterns_e2e_acceptance.py [--binary ./memory_benchmark] [--list] [-v]
  MEMORY_BENCHMARK_BIN=./memory_benchmark python3 -m unittest -v tests/test_patterns_e2e_acceptance.py
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import sys
import tempfile
import time
import unittest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = REPO_ROOT / "script-examples"
PATTERNS_VERIFIER = SCRIPT_DIR / "verify_patterns_result.py"
BANDWIDTH_VERIFIER = SCRIPT_DIR / "verify_cpu_bandwidth_result.py"
LATENCY_VERIFIER = SCRIPT_DIR / "verify_cpu_latency_result.py"

PATTERNS_METHODOLOGY = "patterns-v1-arm64-phase-calibrated-seeded-balanced"
SKIP_REASON_STRIDE_TRANSITION = "buffer-too-small-for-stride-transition"

CANONICAL_PATTERNS = (
    "sequential_forward",
    "sequential_reverse",
    "strided_64",
    "strided_4096",
    "strided_16384",
    "strided_2mb",
    "random",
)
STREAMING_PATTERNS = ("sequential_forward", "sequential_reverse", "strided_64")
SPARSE_PATTERNS = ("strided_4096", "strided_16384", "strided_2mb", "random")
CANONICAL_OPERATIONS = ("read", "write", "copy")

EXPECTED_TOP_LEVEL_KEYS = (
    "schema_version",
    "mode",
    "methodology_version",
    "software",
    "build_manifest",
    "configuration",
    "resolved_plan",
    "memory_budget",
    "calibration",
    "measurements",
    "aggregates",
    "status",
    "reason_code",
    "results_complete",
    "run_accepted",
    "interpretation",
    "diagnostic",
    "interruption_requested",
    "operation_order_balance_complete",
    "pattern_order_balance_complete",
    "seeds",
    "counters",
    "checkpoint_lifecycle",
    "loop_records",
    "environment",
    "quality_warnings",
)

EXPECTED_VALIDATION_CHECKS = (
    "post-validation-structure",
    "read-checksum",
    "write-final-touched-bytes",
    "copy-final-touched-bytes",
    "untouched-sentinel-isolation",
    "src-readonly-protection",
)

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
DECIMAL_UINT_RE = re.compile(r"^(0|[1-9][0-9]*)$")

DEFAULT_RUN_TIMEOUT_SEC = 120
INTERRUPT_TIMEOUT_SEC = 30


def resolve_binary_path(raw_path: str | None = None) -> Path:
    """Resolve the target binary path without touching or building anything."""
    candidate = (
        raw_path
        if raw_path is not None
        else (
            os.environ.get("MEMORY_BENCHMARK_BIN")
            or os.environ.get("MEMBENCH_BINARY")
            or "./memory_benchmark"
        )
    )
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


def _expected_validation_specs(op: str, has_untouched_dst_bytes: bool) -> list[tuple[str, bool]]:
    return [
        ("post-validation-structure", True),
        ("read-checksum", op == "read"),
        ("write-final-touched-bytes", op == "write"),
        ("copy-final-touched-bytes", op == "copy"),
        ("untouched-sentinel-isolation", op in ("write", "copy") and has_untouched_dst_bytes),
        ("src-readonly-protection", op in ("read", "copy")),
    ]


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

    def verify_patterns_json(
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
            PATTERNS_VERIFIER,
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
# Tier 1: Feature Coverage (35 tests: 5 per feature F1..F7)
# =============================================================================


class Tier1PatternsFeatureTests(E2EAcceptanceTestCase):
    """Tier 1 feature coverage for -P / --patterns across F1..F7."""

    # --- F1: CLI 11-Option Whitelist & Help ---

    def test_t1_f1_help_short_flag(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t1_f1_h_") as tmp:
            cwd = Path(tmp)
            proc = self.run_binary(["-P", "-h"], cwd=cwd)
            self.assert_run_succeeded(proc, "-P -h")
            self.assertEqual(proc.stderr.strip(), "", f"Expected empty stderr on help: {proc.stderr}")
            self.assertIn("Usage:", proc.stdout)
            self.assertIn("--patterns", proc.stdout)
            self.assertNotIn("Running", proc.stdout)
            with self.assertRaises(json.JSONDecodeError):
                json.loads(proc.stdout)

    def test_t1_f1_help_long_flag_with_dash_output(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t1_f1_hl_") as tmp:
            cwd = Path(tmp)
            for args in (["--patterns", "--help"], ["-P", "-h", "-o", "-"], ["--patterns", "--output=-", "--help"]):
                proc = self.run_binary(args, cwd=cwd)
                self.assert_run_succeeded(proc, f"help {args}")
                self.assertEqual(proc.stderr.strip(), "")
                self.assertIn("Usage:", proc.stdout)
                self.assertIn("--patterns", proc.stdout)
                self.assertFalse((cwd / "-").exists())
                self.assertFalse((cwd / "-.tmp").exists())

    def test_t1_f1_inline_long_options(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t1_f1_inline_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "inline.json"
            proc = self.run_binary(
                [
                    "--patterns",
                    "--buffer-size=3",
                    "--threads=2",
                    "--iterations=2",
                    "--count=1",
                    "--seed=424242",
                    f"--output={out_path}",
                ],
                cwd=cwd,
            )
            self.assert_run_succeeded(proc, "inline long options")
            doc = self.assert_atomic_json_file(out_path)
            self.assertEqual(doc["configuration"]["buffer_size_mb"], 3)
            self.assertEqual(doc["configuration"]["requested_workers"], 2)
            self.assertEqual(doc["configuration"]["iterations"], 2)
            self.assertEqual(doc["configuration"]["loop_count"], 1)
            self.assertEqual(doc["configuration"]["base_seed_uint64_decimal"], "424242")
            self.verify_patterns_json(out_path)

    def test_t1_f1_stdout_transport_dash(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t1_f1_stdout_") as tmp:
            cwd = Path(tmp)
            proc = self.run_binary(["-P", "-b", "3", "-t", "2", "-i", "2", "-r", "1", "-o", "-"], cwd=cwd)
            self.assert_run_succeeded(proc, "-P -o -")
            doc = self.assert_stdout_json_transport(proc, cwd)
            self.assertEqual(doc["configuration"]["output_file"], "-")
            self.assertEqual(doc["mode"], "patterns")
            captured = cwd / "captured_stdout.json"
            captured.write_text(proc.stdout, encoding="utf-8")
            self.verify_patterns_json(captured)

    def test_t1_f1_console_only_omitted_output(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t1_f1_console_") as tmp:
            cwd = Path(tmp)
            proc = self.run_binary(["-P", "-b", "1", "-t", "2", "-i", "2", "-r", "1"], cwd=cwd)
            self.assert_run_succeeded(proc, "-P console only")
            self.assertTrue((proc.stdout + proc.stderr).strip())
            self.assertEqual(list(cwd.iterdir()), [])

    # --- F2: 7 Patterns & 3 Operations ---

    def test_t1_f2_all_seven_patterns_published_in_plan_and_aggregates(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t1_f2_all7_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "all7.json"
            proc = self.run_binary(
                ["-P", "-b", "3", "-t", "2", "-i", "2", "-r", "3", "--seed", "424242", "-o", str(out_path)],
                cwd=cwd,
            )
            self.assert_run_succeeded(proc, "all 7 patterns")
            doc = self.assert_atomic_json_file(out_path)
            self.assertEqual(tuple(doc["resolved_plan"]["patterns"].keys()), CANONICAL_PATTERNS)
            self.assertEqual(tuple(doc["resolved_plan"]["frozen_plan_refs"].keys()), CANONICAL_PATTERNS)
            self.assertEqual(tuple(doc["aggregates"]["patterns"].keys()), CANONICAL_PATTERNS)
            for pat in CANONICAL_PATTERNS:
                self.assertTrue(doc["resolved_plan"]["patterns"][pat]["enabled"], pat)
                self.assertEqual(doc["resolved_plan"]["patterns"][pat]["status"], "active", pat)
                for op in CANONICAL_OPERATIONS:
                    self.assertIsInstance(doc["resolved_plan"]["frozen_plan_refs"][pat][op], int)
                    self.assertEqual(doc["aggregates"]["patterns"][pat][op]["status"], "measured")
            self.verify_patterns_json(out_path)

    def test_t1_f2_sequential_forward_and_reverse_read_write_copy(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t1_f2_seq_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "seq.json"
            proc = self.run_binary(
                ["-P", "-b", "1", "-t", "2", "-i", "2", "-r", "2", "--seed", "424242", "-o", str(out_path)],
                cwd=cwd,
            )
            self.assert_run_succeeded(proc, "seq forward & reverse")
            doc = self.assert_atomic_json_file(out_path)
            for pat in ("sequential_forward", "sequential_reverse"):
                pinfo = doc["resolved_plan"]["patterns"][pat]
                self.assertEqual(pinfo["access_size_bytes"], 32)
                self.assertEqual(pinfo["phase_period_passes"], 0)
                for op in CANONICAL_OPERATIONS:
                    agg = doc["aggregates"]["patterns"][pat][op]
                    self.assertEqual(agg["status"], "measured")
                    self.assertEqual(len(agg["accepted_measurement_ids"]), 2)
                    self.assertGreater(agg["value"], 0.0)
                    if pat == "sequential_forward":
                        self.assertAlmostEqual(agg["relative_to_forward_pct"], 100.0, places=9)
                    else:
                        self.assertGreater(agg["relative_to_forward_pct"], 0.0)
            self.verify_patterns_json(out_path)

    def test_t1_f2_strided_64_4096_16384_phase_metrics(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t1_f2_strided_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "strided.json"
            proc = self.run_binary(["-P", "-b", "1", "-t", "2", "-i", "2", "-r", "1", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "strided 64/4096/16384")
            doc = self.assert_atomic_json_file(out_path)
            page_size = int(doc["environment"]["page_size_bytes"])
            expected_strides = {"strided_64": 64, "strided_4096": 4096, "strided_16384": 16384}
            for pat, stride in expected_strides.items():
                pinfo = doc["resolved_plan"]["patterns"][pat]
                self.assertEqual(pinfo["access_size_bytes"], 32)
                self.assertEqual(pinfo["stride_bytes"], stride)
                self.assertEqual(pinfo["phase_period_passes"], stride // 32)
                self.assertEqual(pinfo["native_page_size_bytes"], page_size)
                self.assertEqual(pinfo["stride_equals_native_page_size"], stride == page_size)
                self.assertFalse(pinfo["large_page_backing_verified"])
            self.verify_patterns_json(out_path)

    def test_t1_f2_strided_2mb_active_with_3mb_buffer(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t1_f2_2mb_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "strided_2mb.json"
            proc = self.run_binary(["-P", "-b", "3", "-t", "2", "-i", "2", "-r", "1", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "strided_2mb active at 3 MiB")
            doc = self.assert_atomic_json_file(out_path)
            pinfo = doc["resolved_plan"]["patterns"]["strided_2mb"]
            self.assertTrue(pinfo["enabled"])
            self.assertEqual(pinfo["status"], "active")
            self.assertEqual(pinfo["stride_bytes"], 2 * 1024 * 1024)
            self.assertEqual(pinfo["phase_period_passes"], (2 * 1024 * 1024) // 32)
            self.assertEqual(pinfo["effective_workers"], 1)
            for op in CANONICAL_OPERATIONS:
                self.assertEqual(doc["aggregates"]["patterns"]["strided_2mb"][op]["status"], "measured")
            self.verify_patterns_json(out_path)

    def test_t1_f2_random_workload_metadata_and_sha256(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t1_f2_random_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "random.json"
            proc = self.run_binary(
                ["-P", "-b", "3", "-t", "2", "-i", "2", "-r", "1", "--seed", "424242", "-o", str(out_path)],
                cwd=cwd,
            )
            self.assert_run_succeeded(proc, "random workload metadata")
            doc = self.assert_atomic_json_file(out_path)
            pinfo = doc["resolved_plan"]["patterns"]["random"]
            rw = pinfo["random_workload"]
            self.assertRegex(str(rw["random_seed_uint64_decimal"]), DECIMAL_UINT_RE)
            self.assertRegex(rw["random_indices_sha256"], HEX64_RE)
            selected = int(rw["selected_accesses_per_pass"])
            self.assertEqual(selected, (3 * 1024 * 1024) // 32)
            self.assertEqual(int(rw["distinct_address_count"]), selected)
            self.assertEqual(int(rw["logical_working_set_bytes"]), selected * 32)
            res = doc["resolved_plan"]["resources"]
            self.assertEqual(int(res["random_index_table_bytes"]), selected * 8)
            self.assertEqual(int(res["random_validation_index_bytes"]), selected * 8)
            self.verify_patterns_json(out_path)

    # --- F3: Buffer Size, Memory Budget & strided_2mb Skip ---

    def test_t1_f3_buffer_1mb_skips_strided_2mb(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t1_f3_b1_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "b1.json"
            proc = self.run_binary(
                ["-P", "-b", "1", "-t", "2", "-i", "2", "-r", "3", "--seed", "424242", "-o", str(out_path)],
                cwd=cwd,
            )
            self.assert_run_succeeded(proc, "-b 1 skips strided_2mb")
            doc = self.assert_atomic_json_file(out_path)
            s2m = doc["resolved_plan"]["patterns"]["strided_2mb"]
            self.assertFalse(s2m["enabled"])
            self.assertEqual(s2m["status"], "skipped")
            self.assertEqual(s2m["reason_code"], SKIP_REASON_STRIDE_TRANSITION)
            self.assertEqual(s2m["effective_workers"], 0)
            self.assertEqual(len(doc["measurements"]), 3 * 6 * 3)
            self.assertNotIn("strided_2mb", doc["calibration"]["attempts"])
            self.assertTrue(all(row["pattern"] != "strided_2mb" for row in doc["measurements"]))
            for op in CANONICAL_OPERATIONS:
                self.assertIsNone(doc["resolved_plan"]["frozen_plan_refs"]["strided_2mb"][op])
                agg = doc["aggregates"]["patterns"]["strided_2mb"][op]
                self.assertEqual(agg["status"], "skipped")
                self.assertEqual(agg["reason_code"], SKIP_REASON_STRIDE_TRANSITION)
            self.verify_patterns_json(out_path)

    def test_t1_f3_buffer_2mb_skips_strided_2mb(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t1_f3_b2_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "b2.json"
            proc = self.run_binary(["-P", "-b", "2", "-t", "2", "-i", "2", "-r", "1", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "-b 2 skips strided_2mb")
            doc = self.assert_atomic_json_file(out_path)
            s2m = doc["resolved_plan"]["patterns"]["strided_2mb"]
            self.assertFalse(s2m["enabled"])
            self.assertEqual(s2m["status"], "skipped")
            self.assertEqual(s2m["reason_code"], SKIP_REASON_STRIDE_TRANSITION)
            self.assertEqual(len(doc["measurements"]), 6 * 3)
            self.verify_patterns_json(out_path)

    def test_t1_f3_buffer_3mb_activates_all_seven_patterns(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t1_f3_b3_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "b3.json"
            proc = self.run_binary(
                ["-P", "-b", "3", "-t", "2", "-i", "2", "-r", "3", "--seed", "424242", "-o", str(out_path)],
                cwd=cwd,
            )
            self.assert_run_succeeded(proc, "-b 3 activates all 7")
            doc = self.assert_atomic_json_file(out_path)
            self.assertEqual(len(doc["measurements"]), 3 * 7 * 3)
            self.assertEqual(doc["counters"]["measured_measurements"], 63)
            self.assertEqual(doc["resolved_plan"]["patterns"]["strided_2mb"]["status"], "active")
            self.verify_patterns_json(out_path)

    def test_t1_f3_non_cacheable_flag_madvise_evidence(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t1_f3_u_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "non_cacheable.json"
            proc = self.run_binary(["-P", "-b", "3", "-t", "1", "-i", "2", "-r", "1", "-u", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "-u non-cacheable")
            doc = self.assert_atomic_json_file(out_path)
            self.assertTrue(doc["configuration"]["non_cacheable"])
            self.assertTrue(doc["interpretation"]["non_cacheable_requested"])
            self.assertTrue(doc["interpretation"]["non_cacheable_is_best_effort_madvise"])
            for buf_name in ("src", "dst"):
                ev = doc["resolved_plan"]["resources"][buf_name]
                self.assertTrue(ev["non_cacheable_requested"])
                self.assertEqual(ev["madvise_advice"], 1)
            self.verify_patterns_json(out_path)

    def test_t1_f3_memory_budget_fields_within_budget(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t1_f3_budget_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "budget.json"
            proc = self.run_binary(["-P", "-b", "3", "-t", "2", "-i", "2", "-r", "1", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "memory_budget fields")
            doc = self.assert_atomic_json_file(out_path)
            mb = doc["memory_budget"]
            self.assertTrue(mb["valid"])
            self.assertEqual(mb["reason_code"], "within-memory-budget")
            known_peak = int(mb["known_owned_peak_bytes"])
            admitted = int(mb["admitted_budget_bytes"])
            allowed = int(mb["allowed_memory_bytes"])
            self.assertLessEqual(known_peak, admitted)
            self.assertLessEqual(admitted, allowed)
            self.assertGreaterEqual(known_peak, 2 * 3 * 1024 * 1024)
            self.verify_patterns_json(out_path)

    # --- F4: Calibration, Determinism & Latin Square ---

    def test_t1_f4_auto_calibration_excluded_from_results(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t1_f4_autocal_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "autocal.json"
            proc = self.run_binary(["-P", "-b", "3", "-t", "2", "-r", "1", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "auto calibration")
            doc = self.assert_atomic_json_file(out_path)
            self.assertIsNone(doc["configuration"]["iterations"])
            self.assertEqual(doc["configuration"]["work_policy"], "automatic_calibration")
            self.assertEqual(doc["configuration"]["resolved_sources"]["iterations"], "auto")
            self.assertTrue(doc["calibration"]["excluded_from_results"])
            attempts = doc["calibration"]["attempts"]
            self.assertEqual(set(attempts.keys()), set(CANONICAL_PATTERNS))
            for pat in CANONICAL_PATTERNS:
                for op in CANONICAL_OPERATIONS:
                    att_list = attempts[pat][op]
                    self.assertTrue(att_list, f"Empty calibration attempts for {pat}/{op}")
                    purposes = {a["purpose"] for a in att_list}
                    self.assertIn("pilot", purposes)
            self.verify_patterns_json(out_path)

    def test_t1_f4_explicit_iterations_skips_pilot_scaling(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t1_f4_explicit_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "explicit.json"
            proc = self.run_binary(["-P", "-b", "3", "-t", "2", "-i", "2", "-r", "3", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "explicit iterations")
            doc = self.assert_atomic_json_file(out_path)
            self.assertEqual(doc["configuration"]["iterations"], 2)
            self.assertEqual(doc["configuration"]["work_policy"], "explicit_fixed_work")
            self.assertEqual(doc["configuration"]["resolved_sources"]["iterations"], "explicit")
            for pat in CANONICAL_PATTERNS:
                for op in CANONICAL_OPERATIONS:
                    att_list = doc["calibration"]["attempts"][pat][op]
                    self.assertEqual(len(att_list), 1)
                    self.assertEqual(att_list[0]["purpose"], "frozen_measurement_warmup")
            for plan in doc["resolved_plan"]["operation_plans"]:
                self.assertEqual(plan["passes"], 2)
                self.assertTrue(plan["explicit_iterations"])
            self.verify_patterns_json(out_path)

    def test_t1_f4_default_count_is_three(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t1_f4_defcount_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "default_count.json"
            proc = self.run_binary(["-P", "-b", "1", "-t", "2", "-i", "2", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "default loop count 3")
            doc = self.assert_atomic_json_file(out_path)
            self.assertEqual(doc["configuration"]["loop_count"], 3)
            self.assertEqual(doc["counters"]["planned_loops"], 3)
            self.assertEqual(doc["counters"]["completed_loops"], 3)
            self.assertTrue(doc["operation_order_balance_complete"])
            self.assertFalse(doc["pattern_order_balance_complete"])
            self.verify_patterns_json(out_path)

    def test_t1_f4_two_axis_latin_square_rotation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t1_f4_latin_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "latin7.json"
            proc = self.run_binary(["-P", "-b", "3", "-t", "2", "-i", "1", "-r", "7", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "2-axis latin square")
            doc = self.assert_atomic_json_file(out_path)
            self.assertTrue(doc["pattern_order_balance_complete"])
            self.assertFalse(doc["operation_order_balance_complete"])
            self.assertEqual(len(doc["loop_records"]), 7)
            for loop_idx, rec in enumerate(doc["loop_records"]):
                self.assertEqual(rec["loop_index"], loop_idx)
                exp_pats = [CANONICAL_PATTERNS[(loop_idx + k) % 7] for k in range(7)]
                exp_ops = [CANONICAL_OPERATIONS[(loop_idx + k) % 3] for k in range(3)]
                expected_order = [{"pattern": p, "operation": o} for p in exp_pats for o in exp_ops]
                self.assertEqual(rec["planned_order"], expected_order)
                self.assertEqual(rec["realized_order"], expected_order)
            self.verify_patterns_json(out_path)

    def test_t1_f4_explicit_seed_zero_accepted(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t1_f4_seed0_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "seed0.json"
            proc = self.run_binary(["-P", "-b", "1", "-t", "1", "-i", "1", "-r", "1", "--seed", "0", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "--seed 0")
            doc = self.assert_atomic_json_file(out_path)
            self.assertEqual(doc["configuration"]["base_seed_uint64_decimal"], "0")
            self.assertEqual(doc["configuration"]["seed_source"], "user")
            self.assertEqual(doc["configuration"]["resolved_sources"]["seed"], "explicit")
            self.assertEqual(doc["seeds"]["base_seed_uint64_decimal"], "0")
            self.assertEqual(doc["seeds"]["source"], "user")
            self.verify_patterns_json(out_path)

    # --- F5: Sweep Matrix & Guardrails ---

    def test_t1_f5_sweep_buffer_size_single_axis(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t1_f5_sb_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "sweep_buf.json"
            proc = self.run_binary(
                ["-P", "-t", "2", "-i", "1", "-r", "1", "-S", "buffer-size=1,3", "-o", str(out_path)],
                cwd=cwd,
            )
            self.assert_run_succeeded(proc, "sweep buffer-size=1,3")
            doc = self.assert_atomic_json_file(out_path)
            self.assertEqual(doc["configuration"]["mode"], "sweep")
            self.assertEqual(doc["configuration"]["base_mode"], "patterns")
            self.assertEqual(doc["configuration"]["sweep_schema_version"], 1)
            self.assertEqual(doc["configuration"]["run_count"], 2)
            self.assertEqual(doc["completed_runs"], 2)
            self.assertEqual(doc["runs"][0]["result"]["resolved_plan"]["patterns"]["strided_2mb"]["status"], "skipped")
            self.assertEqual(doc["runs"][1]["result"]["resolved_plan"]["patterns"]["strided_2mb"]["status"], "active")
            self.verify_patterns_json(out_path)

    def test_t1_f5_sweep_threads_single_axis(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t1_f5_st_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "sweep_threads.json"
            proc = self.run_binary(
                ["-P", "-b", "3", "-i", "1", "-r", "1", "-S", "threads=1,2", "-o", str(out_path)],
                cwd=cwd,
            )
            self.assert_run_succeeded(proc, "sweep threads=1,2")
            doc = self.assert_atomic_json_file(out_path)
            self.assertEqual(doc["configuration"]["run_count"], 2)
            self.assertEqual(doc["completed_runs"], 2)
            for idx, req_w in enumerate((1, 2)):
                nested = doc["runs"][idx]["result"]
                self.assertEqual(nested["configuration"]["requested_workers"], req_w)
                self.assertEqual(nested["configuration"]["resolved_sources"]["workers"], "sweep")
            self.verify_patterns_json(out_path)

    def test_t1_f5_sweep_cartesian_two_axis(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t1_f5_cart_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "sweep_cartesian.json"
            proc = self.run_binary(
                [
                    "-P",
                    "-i",
                    "1",
                    "-r",
                    "1",
                    "-S",
                    "buffer-size=1,3",
                    "-S",
                    "threads=1,2",
                    "-X",
                    "4",
                    "-o",
                    str(out_path),
                ],
                cwd=cwd,
            )
            self.assert_run_succeeded(proc, "2-axis cartesian sweep")
            doc = self.assert_atomic_json_file(out_path)
            self.assertEqual(doc["configuration"]["run_count"], 4)
            self.assertEqual(doc["planned_runs"], 4)
            self.assertEqual(doc["completed_runs"], 4)
            self.assertTrue(doc["conclusions_valid"])
            self.verify_patterns_json(out_path)

    def test_t1_f5_sweep_stdout_transport(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t1_f5_sout_") as tmp:
            cwd = Path(tmp)
            proc = self.run_binary(["-P", "-i", "1", "-r", "1", "-S", "buffer-size=1,3", "-o", "-"], cwd=cwd)
            self.assert_run_succeeded(proc, "sweep -o -")
            doc = self.assert_stdout_json_transport(proc, cwd)
            self.assertEqual(doc["configuration"]["mode"], "sweep")
            self.assertEqual(doc["configuration"]["base_mode"], "patterns")
            self.assertEqual(doc["completed_runs"], 2)
            captured = cwd / "sweep_stdout.json"
            captured.write_text(proc.stdout, encoding="utf-8")
            self.verify_patterns_json(captured)

    def test_t1_f5_sweep_shared_generated_seed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t1_f5_sseed_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "sweep_shared_seed.json"
            proc = self.run_binary(
                ["-P", "-i", "1", "-r", "1", "-S", "buffer-size=1,3", "-o", str(out_path)],
                cwd=cwd,
            )
            self.assert_run_succeeded(proc, "sweep shared generated seed")
            doc = self.assert_atomic_json_file(out_path)
            s0 = doc["runs"][0]["result"]["seeds"]
            s1 = doc["runs"][1]["result"]["seeds"]
            self.assertEqual(s0["source"], "generated")
            self.assertEqual(s1["source"], "generated")
            self.assertEqual(s0["base_seed_uint64_decimal"], s1["base_seed_uint64_decimal"])
            self.verify_patterns_json(out_path)

    # --- F6: Schema v1 Output, Audit Evidence & Checkpoints ---

    def test_t1_f6_schema_v1_26_canonical_keys_and_identities(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t1_f6_schema_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "schema.json"
            proc = self.run_binary(["-P", "-b", "3", "-t", "2", "-i", "2", "-r", "1", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "schema v1 26 keys")
            doc = self.assert_atomic_json_file(out_path)
            self.assertEqual(tuple(doc.keys()), EXPECTED_TOP_LEVEL_KEYS)
            self.assertEqual(doc["schema_version"], 1)
            self.assertEqual(doc["mode"], "patterns")
            self.assertEqual(doc["methodology_version"], PATTERNS_METHODOLOGY)
            comp = doc["resolved_plan"]["component_identities"]
            self.assertEqual(comp["version"], "patterns-components-v1")
            self.assertEqual(comp["work_plan_version"], "patterns-work-plan-v1")
            self.assertEqual(comp["buffer_pattern_version"], "patterns-buffer-pattern-v1")
            self.assertEqual(comp["read_checksum_version"], "patterns-read-checksum-v1")
            self.assertEqual(comp["random_index_generator_version"], "splitmix64-coprime-fisher-yates-v1")
            self.assertEqual(comp["run_policy_version"], "bounded-loop-snapshots-v1")
            self.verify_patterns_json(out_path)

    def test_t1_f6_build_manifest_and_environment_snapshots(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t1_f6_env_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "env.json"
            proc = self.run_binary(["-P", "-b", "1", "-t", "2", "-i", "1", "-r", "1", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "manifest and environment")
            doc = self.assert_atomic_json_file(out_path)
            bm = doc["build_manifest"]
            self.assertEqual(bm["manifest_version"], 1)
            self.assertIn(bm["status"], ("available", "partial", "unavailable"))
            if bm["binary_sha256"] is not None:
                self.assertRegex(bm["binary_sha256"], HEX64_RE)
            env = doc["environment"]
            self.assertGreater(int(env["logical_core_count"]), 0)
            self.assertGreater(int(env["page_size_bytes"]), 0)
            self.assertIn("main_thread_qos", env)
            for snap_key in ("start", "end"):
                self.assertIn(env[snap_key]["thermal_state"], ("nominal", "fair", "serious", "critical", "unavailable"))
            self.verify_patterns_json(out_path)

    def test_t1_f6_six_validation_checks_per_operation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t1_f6_checks_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "checks.json"
            proc = self.run_binary(["-P", "-b", "3", "-t", "2", "-i", "2", "-r", "1", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "6 validation checks")
            doc = self.assert_atomic_json_file(out_path)
            plan_by_ref = {p["plan_ref"]: p for p in doc["resolved_plan"]["operation_plans"]}
            for row in doc["measurements"]:
                plan = plan_by_ref[row["plan_ref"]]
                checks = row["execution"]["validation"]["checks"]
                self.assertEqual([c["kind"] for c in checks], list(EXPECTED_VALIDATION_CHECKS))
                expected_specs = _expected_validation_specs(row["operation"], bool(plan["has_untouched_dst_bytes"]))
                for ck, (kind, applicable) in zip(checks, expected_specs):
                    self.assertEqual(ck["kind"], kind)
                    self.assertEqual(ck["applicable"], applicable)
                    if applicable:
                        self.assertTrue(ck["evaluated"])
                        self.assertTrue(ck["valid"])
                    else:
                        self.assertIsNone(ck["evaluated"])
                        self.assertIsNone(ck["valid"])
            self.verify_patterns_json(out_path)

    def test_t1_f6_write_patterns_unique_odd_and_sentinel_isolated(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t1_f6_write_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "write_sentinel.json"
            proc = self.run_binary(["-P", "-b", "3", "-t", "2", "-i", "2", "-r", "2", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "write patterns unique odd & sentinel isolated")
            doc = self.assert_atomic_json_file(out_path)
            seen_write_patterns: set[str] = set()
            sparse_checked = 0
            plan_by_ref = {p["plan_ref"]: p for p in doc["resolved_plan"]["operation_plans"]}
            for row in doc["measurements"]:
                ex = row["execution"]
                if row["operation"] == "write":
                    wp_str = ex["write_pattern_uint64_decimal"]
                    self.assertRegex(str(wp_str), DECIMAL_UINT_RE)
                    wp_val = int(wp_str)
                    self.assertGreater(wp_val, 0)
                    self.assertEqual(wp_val & 1, 1)
                    self.assertNotIn(wp_str, seen_write_patterns)
                    seen_write_patterns.add(wp_str)
                else:
                    self.assertIsNone(ex["write_pattern_uint64_decimal"])
                if row["operation"] in ("write", "copy") and plan_by_ref[row["plan_ref"]]["has_untouched_dst_bytes"]:
                    sentinel_ck = ex["validation"]["checks"][4]
                    self.assertEqual(sentinel_ck["kind"], "untouched-sentinel-isolation")
                    self.assertTrue(sentinel_ck["applicable"])
                    self.assertTrue(sentinel_ck["valid"])
                    sparse_checked += 1
            self.assertGreater(sparse_checked, 0)
            self.verify_patterns_json(out_path)

    def test_t1_f6_checkpoint_lifecycle_file_vs_stdout(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t1_f6_ckpt_") as tmp:
            cwd = Path(tmp)
            file_out = cwd / "ckpt.json"
            proc_file = self.run_binary(["-P", "-b", "1", "-t", "1", "-i", "1", "-r", "3", "-o", str(file_out)], cwd=cwd)
            self.assert_run_succeeded(proc_file, "checkpoint file")
            doc_file = self.assert_atomic_json_file(file_out)
            cl_file = doc_file["checkpoint_lifecycle"]
            self.assertEqual(cl_file["checkpoint_policy"], "bounded-loop-snapshots")
            self.assertEqual(cl_file["snapshot_interval_loops"], 1)
            self.assertFalse(cl_file["checkpoint_failed"])
            self.assertTrue(cl_file["stdout_intermediate_checkpoints_are_lazy"])
            self.assertGreaterEqual(cl_file["prior_successful_file_writes"], 1)

            proc_stdout = self.run_binary(["-P", "-b", "1", "-t", "1", "-i", "1", "-r", "3", "-o", "-"], cwd=cwd)
            self.assert_run_succeeded(proc_stdout, "checkpoint stdout")
            doc_stdout = self.assert_stdout_json_transport(proc_stdout, cwd)
            cl_stdout = doc_stdout["checkpoint_lifecycle"]
            self.assertEqual(cl_stdout["prior_file_writer_attempts"], 0)
            self.assertEqual(cl_stdout["prior_successful_file_writes"], 0)
            self.verify_patterns_json(file_out)

    # --- F7: Independent Verifier, Interruption & Non-Regression ---

    def test_t1_f7_verifier_accepts_3mb_complete_with_binary_and_raw(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t1_f7_v3_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "complete_3mb.json"
            proc = self.run_binary(
                ["-P", "-b", "3", "-t", "2", "-i", "2", "-r", "3", "--seed", "424242", "-o", str(out_path)],
                cwd=cwd,
            )
            self.assert_run_succeeded(proc, "3 MiB complete")
            doc = self.assert_atomic_json_file(out_path)
            self.assertEqual(doc["status"], "complete")
            self.assertTrue(doc["run_accepted"])
            verdict = self.verify_patterns_json(out_path, include_binary=True, require_raw_timing=True)
            self.assertEqual(verdict["checks"]["timing"], "cpu-raw-ticks")
            self.assertEqual(verdict["checks"]["checksum"], "independent-oracle")

    def test_t1_f7_verifier_accepts_1mb_skipped_2mb_with_binary_and_raw(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t1_f7_v1_") as tmp:
            cwd = Path(tmp)
            proc = self.run_binary(["-P", "-b", "1", "-t", "2", "-i", "2", "-r", "3", "--seed", "424242", "-o", "-"], cwd=cwd)
            self.assert_run_succeeded(proc, "1 MiB skipped strided_2mb stdout")
            doc = self.assert_stdout_json_transport(proc, cwd)
            self.assertEqual(doc["resolved_plan"]["patterns"]["strided_2mb"]["status"], "skipped")
            self.assertEqual(doc["resolved_plan"]["patterns"]["strided_2mb"]["reason_code"], SKIP_REASON_STRIDE_TRANSITION)
            captured = cwd / "skipped_2mb_stdout.json"
            captured.write_text(proc.stdout, encoding="utf-8")
            verdict = self.verify_patterns_json(captured, include_binary=True, require_raw_timing=True)
            self.assertEqual(verdict["checks"]["timing"], "cpu-raw-ticks")

    def test_t1_f7_cv_warning_thresholds_streaming_vs_sparse(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t1_f7_cv_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "cv_thresholds.json"
            proc = self.run_binary(["-P", "-b", "3", "-t", "2", "-i", "2", "-r", "3", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "cv warning thresholds")
            doc = self.assert_atomic_json_file(out_path)
            for pat in STREAMING_PATTERNS:
                for op in CANONICAL_OPERATIONS:
                    self.assertEqual(doc["aggregates"]["patterns"][pat][op]["quality"]["cv_warning_threshold_pct"], 5.0)
            for pat in SPARSE_PATTERNS:
                for op in CANONICAL_OPERATIONS:
                    self.assertEqual(doc["aggregates"]["patterns"][pat][op]["quality"]["cv_warning_threshold_pct"], 10.0)
            self.verify_patterns_json(out_path)

    def test_t1_f7_counters_and_statistics_consistency(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t1_f7_stats_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "counters_stats.json"
            proc = self.run_binary(["-P", "-b", "1", "-t", "2", "-i", "2", "-r", "3", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "counters and statistics")
            doc = self.assert_atomic_json_file(out_path)
            c = doc["counters"]
            self.assertEqual(c["planned_loops"], 3)
            self.assertEqual(c["attempted_loops"], 3)
            self.assertEqual(c["completed_loops"], 3)
            self.assertEqual(c["planned_measurements"], 54)
            self.assertEqual(c["attempted_measurements"], 54)
            self.assertEqual(c["measured_measurements"], 54)
            for pat in CANONICAL_PATTERNS:
                if pat == "strided_2mb":
                    continue
                for op in CANONICAL_OPERATIONS:
                    agg = doc["aggregates"]["patterns"][pat][op]
                    st = agg["statistics"]
                    self.assertEqual(st["sample_count"], 3)
                    self.assertAlmostEqual(agg["value"], st["median"], places=12)
                    self.assertTrue(agg["quality"]["repeatability_assessment_available"])
                    self.assertTrue(agg["quality"]["run_complete"])
            self.verify_patterns_json(out_path)

    def test_t1_f7_worker_spans_64b_alignment_and_src_readonly(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t1_f7_spans_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "spans_ro.json"
            proc = self.run_binary(["-P", "-b", "3", "-t", "2", "-i", "2", "-r", "1", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "worker spans 64B & src readonly")
            doc = self.assert_atomic_json_file(out_path)
            src_ev = doc["resolved_plan"]["resources"]["src"]
            self.assertTrue(src_ev["read_only_protected"])
            self.assertEqual(src_ev["mprotect_errno"], 0)
            total_bytes = 3 * 1024 * 1024
            for pat, pinfo in doc["resolved_plan"]["patterns"].items():
                spans = pinfo["worker_spans"]
                self.assertEqual(len(spans), pinfo["effective_workers"])
                offset = 0
                for idx, sp in enumerate(spans):
                    sp_off = int(sp["offset_bytes"])
                    sp_len = int(sp["span_bytes"])
                    self.assertEqual(sp_off, offset, f"{pat} span {idx} contiguity")
                    if idx > 0:
                        self.assertEqual(sp_off % 64, 0, f"{pat} span {idx} 64B alignment")
                    offset += sp_len
                self.assertEqual(offset, total_bytes, f"{pat} total span coverage")
            self.verify_patterns_json(out_path)


# =============================================================================
# Tier 2: Boundary & Scale (35 tests: 5 per feature F1..F7)
# =============================================================================


class Tier2PatternsBoundaryAndScaleTests(E2EAcceptanceTestCase):
    """Tier 2 boundary values, clamping, scaling, and sweep envelopes across F1..F7."""

    # --- F1 boundaries ---

    def test_t2_f1_help_precedes_invalid_semantic_args(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t2_f1_hp_") as tmp:
            cwd = Path(tmp)
            for args in (
                ["-P", "-h", "-S", "buffer-size=1,3"],
                ["-P", "-h", "-b", "999999999"],
                ["-P", "-b", "3", "-h", "-S", "buffer-size=1,3"],
            ):
                proc = self.run_binary(args, cwd=cwd)
                self.assert_run_succeeded(proc, f"isolated help {args}")
                self.assertEqual(proc.stderr.strip(), "")
                self.assertIn("Usage:", proc.stdout)

    def test_t2_f1_short_vs_long_flag_equivalence(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t2_f1_eq_") as tmp:
            cwd = Path(tmp)
            short_out = cwd / "short.json"
            long_out = cwd / "long.json"
            p_short = self.run_binary(
                ["-P", "-b", "3", "-t", "1", "-i", "1", "-r", "1", "-u", "--seed", "424242", "-o", str(short_out)],
                cwd=cwd,
            )
            p_long = self.run_binary(
                [
                    "--patterns",
                    "--buffer-size",
                    "3",
                    "--threads",
                    "1",
                    "--iterations",
                    "1",
                    "--count",
                    "1",
                    "--non-cacheable",
                    "--seed",
                    "424242",
                    "--output",
                    str(long_out),
                ],
                cwd=cwd,
            )
            self.assert_run_succeeded(p_short, "short flags")
            self.assert_run_succeeded(p_long, "long flags")
            d_short = self.assert_atomic_json_file(short_out)
            d_long = self.assert_atomic_json_file(long_out)
            self.assertEqual(d_short["seeds"], d_long["seeds"])
            self.assertEqual(d_short["resolved_plan"]["plan_identity"], d_long["resolved_plan"]["plan_identity"])
            self.assertEqual(
                d_short["resolved_plan"]["patterns"]["random"]["random_workload"]["random_indices_sha256"],
                d_long["resolved_plan"]["patterns"]["random"]["random_workload"]["random_indices_sha256"],
            )
            self.verify_patterns_json(short_out)
            self.verify_patterns_json(long_out)

    def test_t2_f1_seed_max_uint64(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t2_f1_umax_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "seed_max.json"
            max_u64 = "18446744073709551615"
            proc = self.run_binary(
                ["-P", "-b", "1", "-t", "1", "-i", "1", "-r", "1", "--seed", max_u64, "-o", str(out_path)],
                cwd=cwd,
            )
            self.assert_run_succeeded(proc, "seed UINT64_MAX")
            doc = self.assert_atomic_json_file(out_path)
            self.assertEqual(doc["configuration"]["base_seed_uint64_decimal"], max_u64)
            self.assertEqual(doc["seeds"]["base_seed_uint64_decimal"], max_u64)
            self.verify_patterns_json(out_path)

    def test_t2_f1_count_one_and_three_balance_flags(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t2_f1_r13_") as tmp:
            cwd = Path(tmp)
            r1_out = cwd / "r1.json"
            r3_out = cwd / "r3.json"
            self.assert_run_succeeded(
                self.run_binary(["-P", "-b", "1", "-t", "1", "-i", "1", "-r", "1", "-o", str(r1_out)], cwd=cwd),
                "-r 1",
            )
            self.assert_run_succeeded(
                self.run_binary(["-P", "-b", "1", "-t", "1", "-i", "1", "-r", "3", "-o", str(r3_out)], cwd=cwd),
                "-r 3",
            )
            d1 = self.assert_atomic_json_file(r1_out)
            d3 = self.assert_atomic_json_file(r3_out)
            self.assertFalse(d1["operation_order_balance_complete"])
            self.assertFalse(d1["pattern_order_balance_complete"])
            self.assertTrue(d3["operation_order_balance_complete"])
            self.assertFalse(d3["pattern_order_balance_complete"])
            self.verify_patterns_json(r1_out)
            self.verify_patterns_json(r3_out)

    def test_t2_f1_auto_threads_default_vs_explicit(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t2_f1_thr_") as tmp:
            cwd = Path(tmp)
            auto_out = cwd / "auto_threads.json"
            exp_out = cwd / "explicit_threads.json"
            self.assert_run_succeeded(
                self.run_binary(["-P", "-b", "1", "-i", "1", "-r", "1", "-o", str(auto_out)], cwd=cwd),
                "auto threads",
            )
            d_auto = self.assert_atomic_json_file(auto_out)
            self.assertEqual(d_auto["configuration"]["worker_source"], "auto")
            self.assertEqual(d_auto["configuration"]["resolved_sources"]["workers"], "auto")
            avail = d_auto["configuration"]["available_workers"]
            self.assertEqual(d_auto["configuration"]["requested_workers"], avail)

            self.assert_run_succeeded(
                self.run_binary(["-P", "-b", "1", "-t", str(avail), "-i", "1", "-r", "1", "-o", str(exp_out)], cwd=cwd),
                "explicit threads",
            )
            d_exp = self.assert_atomic_json_file(exp_out)
            self.assertEqual(d_exp["configuration"]["worker_source"], "user")
            self.assertEqual(d_exp["configuration"]["resolved_sources"]["workers"], "explicit")
            self.verify_patterns_json(auto_out)
            self.verify_patterns_json(exp_out)

    # --- F2 boundaries ---

    def test_t2_f2_strided_2mb_worker_reduction_on_3mb_with_8_threads(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t2_f2_red3_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "b3_t8.json"
            proc = self.run_binary(["-P", "-b", "3", "-t", "8", "-i", "1", "-r", "1", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "-b 3 -t 8 worker reduction")
            doc = self.assert_atomic_json_file(out_path)
            self.assertEqual(doc["configuration"]["requested_workers"], 8)
            self.assertEqual(doc["resolved_plan"]["patterns"]["strided_2mb"]["effective_workers"], 1)
            self.assertGreaterEqual(doc["resolved_plan"]["patterns"]["sequential_forward"]["effective_workers"], 1)
            self.verify_patterns_json(out_path)

    def test_t2_f2_strided_2mb_two_workers_on_5mb(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t2_f2_5mb_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "b5_t2.json"
            proc = self.run_binary(["-P", "-b", "5", "-t", "2", "-i", "1", "-r", "1", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "-b 5 -t 2")
            doc = self.assert_atomic_json_file(out_path)
            expected_workers = min(2, int(doc["configuration"]["available_workers"]))
            self.assertEqual(doc["resolved_plan"]["patterns"]["strided_2mb"]["effective_workers"], expected_workers)
            self.verify_patterns_json(out_path)

    def test_t2_f2_strided_partial_cycle_passes_lt_phase_period(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t2_f2_part_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "partial_cycle.json"
            proc = self.run_binary(["-P", "-b", "3", "-t", "1", "-i", "2", "-r", "1", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "partial phase cycles")
            doc = self.assert_atomic_json_file(out_path)
            plans = {
                (p["pattern"], p["operation"]): p
                for p in doc["resolved_plan"]["operation_plans"]
            }
            for pat in ("strided_4096", "strided_16384", "strided_2mb"):
                for op in CANONICAL_OPERATIONS:
                    self.assertEqual(plans[(pat, op)]["completed_phase_cycles"], 0)
                    self.assertTrue(plans[(pat, op)]["has_untouched_dst_bytes"])
            self.verify_patterns_json(out_path)

    def test_t2_f2_strided_64_exact_full_cycle_passes_2(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t2_f2_full64_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "strided64_p2.json"
            proc = self.run_binary(["-P", "-b", "1", "-t", "1", "-i", "2", "-r", "1", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "strided_64 passes=2 full cycle")
            doc = self.assert_atomic_json_file(out_path)
            for p in doc["resolved_plan"]["operation_plans"]:
                if p["pattern"] == "strided_64":
                    self.assertEqual(p["completed_phase_cycles"], 1)
                    self.assertFalse(p["has_untouched_dst_bytes"])
            self.verify_patterns_json(out_path)

    def test_t2_f2_strided_64_odd_passes_3_partial_cycle(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t2_f2_odd64_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "strided64_p3.json"
            proc = self.run_binary(["-P", "-b", "1", "-t", "1", "-i", "3", "-r", "1", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "strided_64 passes=3")
            doc = self.assert_atomic_json_file(out_path)
            for p in doc["resolved_plan"]["operation_plans"]:
                if p["pattern"] == "strided_64":
                    self.assertEqual(p["completed_phase_cycles"], 1)
                    self.assertFalse(p["has_untouched_dst_bytes"])
            self.verify_patterns_json(out_path)

    # --- F3 boundaries ---

    def test_t2_f3_stride_transition_boundary_2mb_vs_3mb(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t2_f3_b23_") as tmp:
            cwd = Path(tmp)
            b2_out = cwd / "b2.json"
            b3_out = cwd / "b3.json"
            self.assert_run_succeeded(
                self.run_binary(["-P", "-b", "2", "-t", "1", "-i", "1", "-r", "1", "-o", str(b2_out)], cwd=cwd),
                "-b 2",
            )
            self.assert_run_succeeded(
                self.run_binary(["-P", "-b", "3", "-t", "1", "-i", "1", "-r", "1", "-o", str(b3_out)], cwd=cwd),
                "-b 3",
            )
            d2 = self.assert_atomic_json_file(b2_out)
            d3 = self.assert_atomic_json_file(b3_out)
            self.assertEqual(d2["resolved_plan"]["patterns"]["strided_2mb"]["status"], "skipped")
            self.assertEqual(d2["resolved_plan"]["patterns"]["strided_2mb"]["reason_code"], SKIP_REASON_STRIDE_TRANSITION)
            self.assertEqual(d3["resolved_plan"]["patterns"]["strided_2mb"]["status"], "active")
            self.verify_patterns_json(b2_out)
            self.verify_patterns_json(b3_out)

    def test_t2_f3_random_selected_accesses_clamped_to_1m_at_33mb(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t2_f3_r33_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "b33.json"
            proc = self.run_binary(["-P", "-b", "33", "-t", "2", "-i", "1", "-r", "1", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "-b 33 random clamp 1M")
            doc = self.assert_atomic_json_file(out_path)
            rw = doc["resolved_plan"]["patterns"]["random"]["random_workload"]
            self.assertEqual(int(rw["selected_accesses_per_pass"]), 1_000_000)
            self.assertEqual(int(rw["distinct_address_count"]), 1_000_000)
            for p in doc["resolved_plan"]["operation_plans"]:
                if p["pattern"] == "random":
                    self.assertTrue(p["has_untouched_dst_bytes"])
            self.verify_patterns_json(out_path)

    def test_t2_f3_random_full_coverage_on_1mb(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t2_f3_r1_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "b1_random.json"
            proc = self.run_binary(["-P", "-b", "1", "-t", "2", "-i", "1", "-r", "1", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "-b 1 random full coverage")
            doc = self.assert_atomic_json_file(out_path)
            rw = doc["resolved_plan"]["patterns"]["random"]["random_workload"]
            expected_slots = (1 * 1024 * 1024) // 32
            self.assertEqual(int(rw["selected_accesses_per_pass"]), expected_slots)
            self.assertEqual(int(rw["distinct_address_count"]), expected_slots)
            for p in doc["resolved_plan"]["operation_plans"]:
                if p["pattern"] == "random":
                    self.assertFalse(p["has_untouched_dst_bytes"])
            self.verify_patterns_json(out_path)

    def test_t2_f3_excessive_threads_9999_clamped_per_pattern(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t2_f3_t9999_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "t9999.json"
            proc = self.run_binary(["-P", "-b", "3", "-t", "9999", "-i", "1", "-r", "1", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "-t 9999")
            doc = self.assert_atomic_json_file(out_path)
            self.assertEqual(doc["configuration"]["requested_workers"], 9999)
            avail = int(doc["configuration"]["available_workers"])
            for pat in CANONICAL_PATTERNS:
                ew = int(doc["resolved_plan"]["patterns"][pat]["effective_workers"])
                self.assertGreater(ew, 0)
                self.assertLessEqual(ew, avail)
                self.assertNotEqual(ew, 9999)
            self.verify_patterns_json(out_path)

    def test_t2_f3_buffer_1mb_strided_4096_worker_count_boundary(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t2_f3_w16_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "b1_t16.json"
            proc = self.run_binary(["-P", "-b", "1", "-t", "16", "-i", "1", "-r", "1", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "-b 1 -t 16 strided worker reduction")
            doc = self.assert_atomic_json_file(out_path)
            for pat in ("strided_64", "strided_4096", "strided_16384"):
                pinfo = doc["resolved_plan"]["patterns"][pat]
                stride = int(pinfo["stride_bytes"])
                for sp in pinfo["worker_spans"]:
                    self.assertGreaterEqual(int(sp["span_bytes"]), stride + 32)
            self.verify_patterns_json(out_path)

    # --- F4 boundaries ---

    def test_t2_f4_auto_calibration_with_1mb_small_buffer(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t2_f4_cal1_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "cal_1mb.json"
            proc = self.run_binary(["-P", "-b", "1", "-t", "2", "-r", "1", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "auto calibration 1 MiB")
            doc = self.assert_atomic_json_file(out_path)
            self.assertEqual(doc["status"], "complete")
            self.assertTrue(doc["run_accepted"])
            self.assertEqual(doc["configuration"]["work_policy"], "automatic_calibration")
            for p in doc["resolved_plan"]["operation_plans"]:
                self.assertGreaterEqual(p["passes"], 1)
            self.verify_patterns_json(out_path)

    def test_t2_f4_full_balance_6_loops_on_1mb(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t2_f4_bal6_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "bal6.json"
            proc = self.run_binary(["-P", "-b", "1", "-t", "1", "-i", "1", "-r", "6", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "6 loops on 1 MiB full balance")
            doc = self.assert_atomic_json_file(out_path)
            self.assertTrue(doc["pattern_order_balance_complete"])
            self.assertTrue(doc["operation_order_balance_complete"])
            self.assertEqual(len(doc["measurements"]), 6 * 6 * 3)
            self.verify_patterns_json(out_path)

    def test_t2_f4_full_balance_21_loops_on_3mb(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t2_f4_bal21_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "bal21.json"
            proc = self.run_binary(["-P", "-b", "3", "-t", "1", "-i", "1", "-r", "21", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "21 loops on 3 MiB full balance")
            doc = self.assert_atomic_json_file(out_path)
            self.assertTrue(doc["pattern_order_balance_complete"])
            self.assertTrue(doc["operation_order_balance_complete"])
            self.assertEqual(len(doc["measurements"]), 21 * 7 * 3)
            self.verify_patterns_json(out_path)

    def test_t2_f4_iterations_boundary_1(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t2_f4_i1_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "i1.json"
            proc = self.run_binary(["-P", "-b", "1", "-t", "1", "-i", "1", "-r", "1", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "-i 1 boundary")
            doc = self.assert_atomic_json_file(out_path)
            self.assertEqual(doc["configuration"]["iterations"], 1)
            for row in doc["measurements"]:
                self.assertEqual(row["completed_passes"], 1)
            self.verify_patterns_json(out_path)

    def test_t2_f4_different_seeds_produce_different_random_and_write_patterns(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t2_f4_diffseed_") as tmp:
            cwd = Path(tmp)
            out_a = cwd / "s111.json"
            out_b = cwd / "s222.json"
            self.assert_run_succeeded(
                self.run_binary(["-P", "-b", "3", "-t", "1", "-i", "1", "-r", "1", "--seed", "111", "-o", str(out_a)], cwd=cwd),
                "seed 111",
            )
            self.assert_run_succeeded(
                self.run_binary(["-P", "-b", "3", "-t", "1", "-i", "1", "-r", "1", "--seed", "222", "-o", str(out_b)], cwd=cwd),
                "seed 222",
            )
            da = self.assert_atomic_json_file(out_a)
            db = self.assert_atomic_json_file(out_b)
            self.assertNotEqual(
                da["resolved_plan"]["patterns"]["random"]["random_workload"]["random_indices_sha256"],
                db["resolved_plan"]["patterns"]["random"]["random_workload"]["random_indices_sha256"],
            )
            wp_a = [m["execution"]["write_pattern_uint64_decimal"] for m in da["measurements"] if m["operation"] == "write"]
            wp_b = [m["execution"]["write_pattern_uint64_decimal"] for m in db["measurements"] if m["operation"] == "write"]
            self.assertNotEqual(wp_a, wp_b)
            self.verify_patterns_json(out_a)
            self.verify_patterns_json(out_b)

    # --- F5 boundaries ---

    def test_t2_f5_sweep_max_runs_exact_boundary_pass(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t2_f5_xmax_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "sweep_exact_x4.json"
            proc = self.run_binary(
                [
                    "-P",
                    "-i",
                    "1",
                    "-r",
                    "1",
                    "-S",
                    "buffer-size=1,3",
                    "-S",
                    "threads=1,2",
                    "-X",
                    "4",
                    "-o",
                    str(out_path),
                ],
                cwd=cwd,
            )
            self.assert_run_succeeded(proc, "sweep exact -X 4")
            doc = self.assert_atomic_json_file(out_path)
            self.assertEqual(doc["configuration"]["sweep_max_runs"], 4)
            self.assertEqual(doc["completed_runs"], 4)
            self.verify_patterns_json(out_path)

    def test_t2_f5_sweep_single_value_axis(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t2_f5_s1_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "sweep_single.json"
            proc = self.run_binary(["-P", "-i", "1", "-r", "1", "-S", "buffer-size=3", "-X", "1", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "sweep single value -X 1")
            doc = self.assert_atomic_json_file(out_path)
            self.assertEqual(doc["configuration"]["run_count"], 1)
            self.assertEqual(doc["completed_runs"], 1)
            self.verify_patterns_json(out_path)

    def test_t2_f5_sweep_reverse_axis_order(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t2_f5_rev_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "sweep_rev.json"
            proc = self.run_binary(
                [
                    "-P",
                    "-i",
                    "1",
                    "-r",
                    "1",
                    "-S",
                    "threads=1,2",
                    "-S",
                    "buffer-size=1,3",
                    "-o",
                    str(out_path),
                ],
                cwd=cwd,
            )
            self.assert_run_succeeded(proc, "sweep reverse axis order")
            doc = self.assert_atomic_json_file(out_path)
            self.assertEqual(doc["completed_runs"], 4)
            self.verify_patterns_json(out_path)

    def test_t2_f5_sweep_mixed_skipped_and_active_2mb(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t2_f5_mix2mb_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "sweep_123.json"
            proc = self.run_binary(["-P", "-t", "1", "-i", "1", "-r", "1", "-S", "buffer-size=1,2,3", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "sweep buffer-size=1,2,3")
            doc = self.assert_atomic_json_file(out_path)
            self.assertEqual(doc["runs"][0]["result"]["resolved_plan"]["patterns"]["strided_2mb"]["status"], "skipped")
            self.assertEqual(doc["runs"][1]["result"]["resolved_plan"]["patterns"]["strided_2mb"]["status"], "skipped")
            self.assertEqual(doc["runs"][2]["result"]["resolved_plan"]["patterns"]["strided_2mb"]["status"], "active")
            self.verify_patterns_json(out_path)

    def test_t2_f5_sweep_threads_exceeding_2mb_worker_cap(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t2_f5_cap2mb_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "sweep_t14_b3.json"
            proc = self.run_binary(["-P", "-b", "3", "-i", "1", "-r", "1", "-S", "threads=1,4", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "sweep threads=1,4 on -b 3")
            doc = self.assert_atomic_json_file(out_path)
            self.assertEqual(doc["runs"][0]["result"]["resolved_plan"]["patterns"]["strided_2mb"]["effective_workers"], 1)
            self.assertEqual(doc["runs"][1]["result"]["resolved_plan"]["patterns"]["strided_2mb"]["effective_workers"], 1)
            self.assertEqual(doc["runs"][1]["result"]["configuration"]["requested_workers"], 4)
            self.verify_patterns_json(out_path)

    # --- F6 boundaries ---

    def test_t2_f6_checkpoint_interval_1_for_loops_1_to_8(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t2_f6_k1_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "r8.json"
            proc = self.run_binary(["-P", "-b", "1", "-t", "2", "-i", "1", "-r", "8", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "-r 8 checkpoint interval")
            doc = self.assert_atomic_json_file(out_path)
            self.assertEqual(doc["checkpoint_lifecycle"]["snapshot_interval_loops"], 1)
            self.assertEqual(doc["checkpoint_lifecycle"]["prior_successful_file_writes"], 7)
            self.verify_patterns_json(out_path)

    def test_t2_f6_checkpoint_interval_2_for_loops_9_to_16(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t2_f6_k2_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "r9.json"
            proc = self.run_binary(["-P", "-b", "1", "-t", "2", "-i", "1", "-r", "9", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "-r 9 checkpoint interval")
            doc = self.assert_atomic_json_file(out_path)
            self.assertEqual(doc["checkpoint_lifecycle"]["snapshot_interval_loops"], math.ceil(9 / 8))
            self.assertGreaterEqual(doc["checkpoint_lifecycle"]["prior_successful_file_writes"], 4)
            self.verify_patterns_json(out_path)

    def test_t2_f6_copy_and_write_validation_check_applicability_matrix(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t2_f6_valmat_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "val_matrix.json"
            proc = self.run_binary(["-P", "-b", "3", "-t", "2", "-i", "2", "-r", "1", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "validation check applicability matrix")
            doc = self.assert_atomic_json_file(out_path)
            plan_by_ref = {p["plan_ref"]: p for p in doc["resolved_plan"]["operation_plans"]}
            saw_dense_write = False
            saw_sparse_write = False
            saw_dense_copy = False
            saw_sparse_copy = False
            for row in doc["measurements"]:
                untouched = bool(plan_by_ref[row["plan_ref"]]["has_untouched_dst_bytes"])
                checks = {c["kind"]: c for c in row["execution"]["validation"]["checks"]}
                if row["operation"] == "write":
                    if untouched:
                        saw_sparse_write = True
                        self.assertTrue(checks["untouched-sentinel-isolation"]["applicable"])
                    else:
                        saw_dense_write = True
                        self.assertFalse(checks["untouched-sentinel-isolation"]["applicable"])
                elif row["operation"] == "copy":
                    self.assertTrue(checks["src-readonly-protection"]["applicable"])
                    if untouched:
                        saw_sparse_copy = True
                        self.assertTrue(checks["untouched-sentinel-isolation"]["applicable"])
                    else:
                        saw_dense_copy = True
                        self.assertFalse(checks["untouched-sentinel-isolation"]["applicable"])
            self.assertTrue(saw_dense_write and saw_sparse_write and saw_dense_copy and saw_sparse_copy)
            self.verify_patterns_json(out_path)

    def test_t2_f6_json_key_ordering_top_level_and_patterns(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t2_f6_order_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "key_order.json"
            proc = self.run_binary(["-P", "-b", "3", "-t", "1", "-i", "1", "-r", "1", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "json key ordering")
            doc = self.assert_atomic_json_file(out_path)
            self.assertEqual(tuple(doc.keys()), EXPECTED_TOP_LEVEL_KEYS)
            self.assertEqual(tuple(doc["resolved_plan"]["patterns"].keys()), CANONICAL_PATTERNS)
            self.assertEqual(tuple(doc["aggregates"]["patterns"].keys()), CANONICAL_PATTERNS)
            self.verify_patterns_json(out_path)

    def test_t2_f6_stdout_checkpoint_lazy_policy(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t2_f6_lazy_") as tmp:
            cwd = Path(tmp)
            proc = self.run_binary(["-P", "-b", "1", "-t", "2", "-i", "1", "-r", "9", "-o", "-"], cwd=cwd)
            self.assert_run_succeeded(proc, "stdout lazy checkpoint with -r 9")
            doc = self.assert_stdout_json_transport(proc, cwd)
            cl = doc["checkpoint_lifecycle"]
            self.assertEqual(cl["snapshot_interval_loops"], 2)
            self.assertTrue(cl["stdout_intermediate_checkpoints_are_lazy"])
            self.assertEqual(cl["prior_file_writer_attempts"], 0)
            self.assertEqual(cl["prior_successful_file_writes"], 0)
            captured = cwd / "lazy_stdout.json"
            captured.write_text(proc.stdout, encoding="utf-8")
            self.verify_patterns_json(captured)

    # --- F7 boundaries ---

    def test_t2_f7_verifier_accepts_large_random_index_boundary_under_default_cap(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t2_f7_cap_ok_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "b33_cap_ok.json"
            proc = self.run_binary(["-P", "-b", "33", "-t", "2", "-i", "1", "-r", "1", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "-b 33 under --max-random-indices 1000000")
            self.verify_patterns_json(out_path, extra_args=["--max-random-indices", "1000000"])

    def test_t2_f7_verifier_rejects_under_tight_random_index_cap(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t2_f7_cap_fail_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "b33_cap_fail.json"
            proc = self.run_binary(["-P", "-b", "33", "-t", "2", "-i", "1", "-r", "1", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "-b 33 under --max-random-indices 999999")
            self.verify_patterns_json(
                out_path,
                expected_exit=2,
                expected_consistent=None,
                expected_accepted=False,
                extra_args=["--max-random-indices", "999999"],
            )

    def test_t2_f7_verifier_accepts_sweep_with_binary_and_raw_timing(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t2_f7_swver_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "sweep_verify.json"
            proc = self.run_binary(
                ["-P", "-i", "1", "-r", "1", "-S", "buffer-size=1,3", "-S", "threads=1,2", "-o", str(out_path)],
                cwd=cwd,
            )
            self.assert_run_succeeded(proc, "sweep verify with binary & raw")
            verdict = self.verify_patterns_json(out_path, include_binary=True, require_raw_timing=True)
            self.assertEqual(verdict["checks"]["timing"], "cpu-raw-ticks")
            self.assertEqual(verdict["checks"]["build"], "binary-bound")

    def test_t2_f7_raw_timing_timebase_and_delta_ticks_exact(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t2_f7_raw_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "raw_ticks.json"
            proc = self.run_binary(["-P", "-b", "3", "-t", "2", "-i", "2", "-r", "1", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "raw timing ticks")
            doc = self.assert_atomic_json_file(out_path)
            for row in doc["measurements"]:
                timing = row["execution"]["timing"]
                raw = timing["cpu_raw"]
                self.assertIsNotNone(raw)
                start = int(raw["start_ticks"])
                stop = int(raw["stop_ticks"])
                delta = int(raw["delta_ticks"])
                numer = int(raw["timebase_numer"])
                denom = int(raw["timebase_denom"])
                self.assertEqual(delta, (stop - start) & ((1 << 64) - 1))
                reconstructed = float(delta) * float(numer) / float(denom) / 1e9
                self.assertTrue(math.isclose(timing["elapsed_seconds"], reconstructed, rel_tol=1e-10, abs_tol=1e-15))
            self.verify_patterns_json(out_path)

    def test_t2_f7_relative_to_forward_pct_for_all_patterns(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t2_f7_rel_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "rel_pct.json"
            proc = self.run_binary(["-P", "-b", "3", "-t", "2", "-i", "2", "-r", "2", "-o", str(out_path)], cwd=cwd)
            self.assert_run_succeeded(proc, "relative_to_forward_pct")
            doc = self.assert_atomic_json_file(out_path)
            aggs = doc["aggregates"]["patterns"]
            for op in CANONICAL_OPERATIONS:
                fwd_val = aggs["sequential_forward"][op]["value"]
                self.assertGreater(fwd_val, 0.0)
                for pat in CANONICAL_PATTERNS:
                    agg = aggs[pat][op]
                    expected_pct = 100.0 * agg["value"] / fwd_val
                    self.assertTrue(
                        math.isclose(agg["relative_to_forward_pct"], expected_pct, rel_tol=1e-9, abs_tol=1e-12),
                        f"{pat}/{op}: {agg['relative_to_forward_pct']} != {expected_pct}",
                    )
            self.verify_patterns_json(out_path)


# =============================================================================
# Tier 3: Pairwise Negative & Guardrail Rejections (21 tests: all C(7,2) pairs)
# =============================================================================


class Tier3PatternsPairwiseNegativeTests(E2EAcceptanceTestCase):
    """Tier 3 pairwise negative tests covering all 21 pairs of features F1..F7."""

    def test_t3_f1_f2_unsupported_cpu_mode_options_rejected_with_patterns(self) -> None:
        cases = [
            ["-P", "-k", "64"],
            ["-P", "--cache-size", "64"],
            ["-P", "-n", "10"],
            ["-P", "-s", "64"],
            ["-P", "-m", "auto"],
            ["-P", "-l", "16"],
            ["-P", "--compare-locality"],
            ["-P", "-D", "low"],
            ["-P", "--weight-size-mb", "1"],
            ["-P", "--unknown-option"],
        ]
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t3_f1f2_") as tmp:
            cwd = Path(tmp)
            for args in cases:
                with self.subTest(args=args):
                    proc = self.run_binary(args, cwd=cwd)
                    self.assert_run_rejected(proc, f"unsupported option {args}")

    def test_t3_f1_f3_buffer_zero_and_negative_rejected_and_syntax(self) -> None:
        cases = [
            ["-P", "-b", "0"],
            ["-P", "--buffer-size", "0"],
            ["-P", "-b", "-1"],
            ["-P", "-b", "abc"],
            ["-P", "-b=3"],
            ["-P", "-b", "9007199254740992"],
        ]
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t3_f1f3_") as tmp:
            cwd = Path(tmp)
            for args in cases:
                with self.subTest(args=args):
                    proc = self.run_binary(args, cwd=cwd)
                    self.assert_run_rejected(proc, f"invalid buffer syntax/bound {args}")

    def test_t3_f1_f4_invalid_iterations_count_and_seed_syntax(self) -> None:
        cases = [
            ["-P", "-i", "0"],
            ["-P", "-i", "1000000001"],
            ["-P", "-r", "0"],
            ["-P", "--seed", "-1"],
            ["-P", "--seed", "+1"],
            ["-P", "--seed", "18446744073709551616"],
        ]
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t3_f1f4_") as tmp:
            cwd = Path(tmp)
            for args in cases:
                with self.subTest(args=args):
                    proc = self.run_binary(args, cwd=cwd)
                    self.assert_run_rejected(proc, f"invalid i/r/seed {args}")

    def test_t3_f1_f5_sweep_missing_output_and_invalid_max_runs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t3_f1f5_") as tmp:
            cwd = Path(tmp)
            dummy_out = cwd / "unused.json"
            cases = [
                ["-P", "-S", "buffer-size=1,3"],
                ["-P", "-S", "buffer-size=1,3", "-X", "0", "-o", str(dummy_out)],
                ["-P", "-S", "threads=1,2", "-X", "abc", "-o", str(dummy_out)],
                ["-P", "-S", "buffer-size=1,3", "-X", "9007199254740992", "-o", str(dummy_out)],
            ]
            for args in cases:
                with self.subTest(args=args):
                    proc = self.run_binary(args, cwd=cwd)
                    self.assert_run_rejected(proc, f"sweep missing output / bad -X {args}")

    def test_t3_f1_f6_missing_option_value_and_duplicate_single_options(self) -> None:
        cases = [
            ["-P", "-o"],
            ["-P", "-b"],
            ["-P", "-t"],
            ["-P", "-b", "1", "-b", "3"],
            ["-P", "-P"],
            ["-P=1"],
            ["-P", "-u=1"],
        ]
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t3_f1f6_") as tmp:
            cwd = Path(tmp)
            for args in cases:
                with self.subTest(args=args):
                    proc = self.run_binary(args, cwd=cwd)
                    self.assert_run_rejected(proc, f"missing/duplicate option {args}")

    def test_t3_f1_f7_primary_mode_mutual_exclusion_and_legacy_flags(self) -> None:
        cases = [
            ["-P", "-B"],
            ["-P", "-L"],
            ["-P", "-T"],
            ["-P", "-C"],
            ["-P", "-G"],
            ["-P", "-M"],
            ["--patterns", "--cpu-bandwidth"],
            ["--benchmark"],
            ["-W"],
            ["--only-bandwidth"],
            ["--only-latency"],
            ["-P", "-W"],
        ]
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t3_f1f7_") as tmp:
            cwd = Path(tmp)
            for args in cases:
                with self.subTest(args=args):
                    proc = self.run_binary(args, cwd=cwd)
                    self.assert_run_rejected(proc, f"mutex/legacy flag {args}")

    def test_t3_f2_f3_oversized_buffer_budget_exceeded_emits_failed_json(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t3_f2f3_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "budget_overflow.json"
            proc_file = self.run_binary(["-P", "-b", "999999999", "-i", "1", "-r", "1", "-o", str(out_path)], cwd=cwd)
            self.assert_run_rejected(proc_file, "budget overflow file")
            doc = self.assert_atomic_json_file(out_path)
            self.assertEqual(doc["status"], "failed")
            self.assertEqual(doc["reason_code"], "memory-budget-exceeded")
            self.assertFalse(doc["run_accepted"])
            self.assertFalse(doc["memory_budget"]["valid"])
            self.verify_patterns_json(
                out_path,
                expected_exit=1,
                expected_consistent=True,
                expected_accepted=False,
                require_raw_timing=False,
            )

            proc_stdout = self.run_binary(["-P", "-b", "999999999", "-i", "1", "-r", "1", "-o", "-"], cwd=cwd)
            self.assert_run_rejected(proc_stdout, "budget overflow stdout")
            stdout_doc = json.loads(proc_stdout.stdout)
            self.assertEqual(stdout_doc["status"], "failed")
            self.assertEqual(stdout_doc["reason_code"], "memory-budget-exceeded")
            self.assertFalse(stdout_doc["run_accepted"])
            self.assertFalse(stdout_doc["memory_budget"]["valid"])

    def test_t3_f2_f4_work_cap_64gib_exceeded_explicit_iterations(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t3_f2f4_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "work_cap.json"
            # 10 MiB * 2 (copy) * 10000 = ~195 GiB > 64 GiB work cap, well within RAM budget.
            proc = self.run_binary(["-P", "-b", "10", "-t", "1", "-i", "10000", "-r", "1", "-o", str(out_path)], cwd=cwd)
            self.assert_run_rejected(proc, "64 GiB work cap exceeded")
            if out_path.exists():
                doc = self.assert_atomic_json_file(out_path)
                self.assertEqual(doc["status"], "failed")
                self.assertFalse(doc["run_accepted"])

    def test_t3_f2_f5_disallowed_sweep_keys_from_other_modes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t3_f2f5_") as tmp:
            cwd = Path(tmp)
            dummy_out = cwd / "bad_sweep_key.json"
            for spec in ("cache-size=32,64", "latency-stride-bytes=64,256", "tlb-density=low,high", "unknown=1,2"):
                with self.subTest(spec=spec):
                    proc = self.run_binary(["-P", "-S", spec, "-o", str(dummy_out)], cwd=cwd)
                    self.assert_run_rejected(proc, f"disallowed sweep key {spec}")

    def test_t3_f2_f6_measurement_limit_100k_exceeded(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t3_f2f6_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "meas_limit.json"
            # 20000 loops * 6 active patterns * 3 ops = 360,000 > 100,000 kPatternMaxMeasurements.
            proc = self.run_binary(["-P", "-b", "1", "-t", "1", "-i", "1", "-r", "20000", "-o", str(out_path)], cwd=cwd)
            self.assert_run_rejected(proc, "100k measurement limit exceeded")

    def test_t3_f2_f7_verifier_rejects_wrong_mode_or_methodology(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t3_f2f7_") as tmp:
            cwd = Path(tmp)
            live_out = cwd / "live.json"
            self.assert_run_succeeded(
                self.run_binary(["-P", "-b", "1", "-t", "1", "-i", "1", "-r", "1", "-o", str(live_out)], cwd=cwd),
                "live baseline",
            )
            base_doc = self.assert_atomic_json_file(live_out)

            bad_mode = copy.deepcopy(base_doc)
            bad_mode["mode"] = "cpu_bandwidth"
            bad_mode_path = cwd / "bad_mode.json"
            bad_mode_path.write_text(json.dumps(bad_mode), encoding="utf-8")
            self.verify_patterns_json(
                bad_mode_path,
                expected_exit=2,
                expected_consistent=None,
                expected_accepted=False,
            )

            bad_meth = copy.deepcopy(base_doc)
            bad_meth["methodology_version"] = "wrong-methodology-v0"
            bad_meth_path = cwd / "bad_meth.json"
            bad_meth_path.write_text(json.dumps(bad_meth), encoding="utf-8")
            self.verify_patterns_json(
                bad_meth_path,
                expected_exit=2,
                expected_consistent=None,
                expected_accepted=False,
            )

    def test_t3_f3_f4_invalid_threads_zero_and_u32_overflow(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t3_f3f4_") as tmp:
            cwd = Path(tmp)
            for args in (["-P", "-b", "3", "-t", "0"], ["-P", "-b", "3", "-t", "4294967296"]):
                with self.subTest(args=args):
                    proc = self.run_binary(args, cwd=cwd)
                    self.assert_run_rejected(proc, f"invalid threads {args}")

    def test_t3_f3_f5_sweep_buffer_zero_and_budget_preflight_rejection(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t3_f3f5_") as tmp:
            cwd = Path(tmp)
            for spec in ("buffer-size=0,3", "buffer-size=1,999999999"):
                with self.subTest(spec=spec):
                    out_path = cwd / "sweep_preflight_fail.json"
                    if out_path.exists():
                        out_path.unlink()
                    proc = self.run_binary(["-P", "-i", "1", "-r", "1", "-S", spec, "-o", str(out_path)], cwd=cwd)
                    self.assert_run_rejected(proc, f"sweep preflight rejection {spec}")

    def test_t3_f3_f6_unwritable_output_path_rejected_cleanly(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t3_f3f6_") as tmp:
            cwd = Path(tmp)
            blocking_file = cwd / "blocking_file"
            blocking_file.write_text("not a directory\n", encoding="utf-8")
            bad_path = blocking_file / "out.json"
            proc = self.run_binary(["-P", "-b", "1", "-t", "1", "-i", "1", "-r", "1", "-o", str(bad_path)], cwd=cwd)
            self.assert_run_rejected(proc, "unwritable output path")

    def test_t3_f3_f7_verifier_rejects_forged_strided_2mb_skip_or_active(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t3_f3f7_") as tmp:
            cwd = Path(tmp)
            b1_out = cwd / "b1.json"
            b3_out = cwd / "b3.json"
            self.assert_run_succeeded(
                self.run_binary(["-P", "-b", "1", "-t", "1", "-i", "1", "-r", "1", "-o", str(b1_out)], cwd=cwd),
                "-b 1",
            )
            self.assert_run_succeeded(
                self.run_binary(["-P", "-b", "3", "-t", "1", "-i", "1", "-r", "1", "-o", str(b3_out)], cwd=cwd),
                "-b 3",
            )

            forged_active = self.assert_atomic_json_file(b1_out)
            forged_active["resolved_plan"]["patterns"]["strided_2mb"]["status"] = "active"
            forged_active["resolved_plan"]["patterns"]["strided_2mb"]["enabled"] = True
            forged_active_path = cwd / "forged_active.json"
            forged_active_path.write_text(json.dumps(forged_active), encoding="utf-8")
            self.verify_patterns_json(
                forged_active_path,
                expected_exit=1,
                expected_consistent=False,
                expected_accepted=False,
            )

            forged_skip = self.assert_atomic_json_file(b3_out)
            forged_skip["resolved_plan"]["patterns"]["strided_2mb"]["status"] = "skipped"
            forged_skip["resolved_plan"]["patterns"]["strided_2mb"]["enabled"] = False
            forged_skip_path = cwd / "forged_skip.json"
            forged_skip_path.write_text(json.dumps(forged_skip), encoding="utf-8")
            self.verify_patterns_json(
                forged_skip_path,
                expected_exit=1,
                expected_consistent=False,
                expected_accepted=False,
            )

    def test_t3_f4_f5_sweep_conflict_with_direct_buffer_or_threads(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t3_f4f5_") as tmp:
            cwd = Path(tmp)
            dummy_out = cwd / "conflict.json"
            for args in (
                ["-P", "-b", "3", "-S", "buffer-size=1,3", "-o", str(dummy_out)],
                ["-P", "-t", "2", "-S", "threads=1,2", "-o", str(dummy_out)],
            ):
                with self.subTest(args=args):
                    proc = self.run_binary(args, cwd=cwd)
                    self.assert_run_rejected(proc, f"sweep conflict {args}")

    def test_t3_f4_f6_verifier_rejects_latin_square_order_or_balance_tamper(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t3_f4f6_") as tmp:
            cwd = Path(tmp)
            live_out = cwd / "latin_base.json"
            self.assert_run_succeeded(
                self.run_binary(["-P", "-b", "1", "-t", "1", "-i", "1", "-r", "3", "-o", str(live_out)], cwd=cwd),
                "latin base",
            )
            base_doc = self.assert_atomic_json_file(live_out)

            swapped = copy.deepcopy(base_doc)
            swapped["measurements"][0], swapped["measurements"][1] = (
                swapped["measurements"][1],
                swapped["measurements"][0],
            )
            swapped_path = cwd / "swapped_rows.json"
            swapped_path.write_text(json.dumps(swapped), encoding="utf-8")
            self.verify_patterns_json(
                swapped_path,
                expected_exit=1,
                expected_consistent=False,
                expected_accepted=False,
            )

            flipped = copy.deepcopy(base_doc)
            flipped["pattern_order_balance_complete"] = not flipped["pattern_order_balance_complete"]
            flipped_path = cwd / "flipped_balance.json"
            flipped_path.write_text(json.dumps(flipped), encoding="utf-8")
            self.verify_patterns_json(
                flipped_path,
                expected_exit=1,
                expected_consistent=False,
                expected_accepted=False,
            )

    def test_t3_f4_f7_verifier_rejects_forged_seed_or_random_sha256(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t3_f4f7_") as tmp:
            cwd = Path(tmp)
            live_out = cwd / "seed_base.json"
            self.assert_run_succeeded(
                self.run_binary(["-P", "-b", "3", "-t", "1", "-i", "1", "-r", "1", "--seed", "424242", "-o", str(live_out)], cwd=cwd),
                "seed base",
            )
            base_doc = self.assert_atomic_json_file(live_out)

            forged_seed = copy.deepcopy(base_doc)
            forged_seed["seeds"]["base_seed_uint64_decimal"] = "424243"
            forged_seed_path = cwd / "forged_seed.json"
            forged_seed_path.write_text(json.dumps(forged_seed), encoding="utf-8")
            self.verify_patterns_json(
                forged_seed_path,
                expected_exit=1,
                expected_consistent=False,
                expected_accepted=False,
            )

            forged_sha = copy.deepcopy(base_doc)
            rw = forged_sha["resolved_plan"]["patterns"]["random"]["random_workload"]
            orig_sha = rw["random_indices_sha256"]
            rw["random_indices_sha256"] = ("1" if orig_sha[0] == "0" else "0") + orig_sha[1:]
            forged_sha_path = cwd / "forged_sha.json"
            forged_sha_path.write_text(json.dumps(forged_sha), encoding="utf-8")
            self.verify_patterns_json(
                forged_sha_path,
                expected_exit=1,
                expected_consistent=False,
                expected_accepted=False,
            )

    def test_t3_f5_f6_sweep_duplicate_key_empty_value_and_exceeds_max_runs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t3_f5f6_") as tmp:
            cwd = Path(tmp)
            dummy_out = cwd / "bad_sweep.json"
            cases = [
                ["-P", "-S", "buffer-size=1", "-S", "buffer-size=3", "-o", str(dummy_out)],
                ["-P", "-S", "threads=1", "-S", "threads=2", "-o", str(dummy_out)],
                ["-P", "-S", "buffer-size=", "-o", str(dummy_out)],
                ["-P", "-S", "buffer-size=1,,3", "-o", str(dummy_out)],
                ["-P", "-S", "buffer-size=1,3", "-S", "threads=1,2", "-X", "3", "-o", str(dummy_out)],
            ]
            for args in cases:
                with self.subTest(args=args):
                    proc = self.run_binary(args, cwd=cwd)
                    self.assert_run_rejected(proc, f"bad sweep syntax/limits {args}")

    def test_t3_f5_f7_verifier_rejects_tampered_sweep_envelope_invariants(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t3_f5f7_") as tmp:
            cwd = Path(tmp)
            sweep_out = cwd / "sweep_base.json"
            self.assert_run_succeeded(
                self.run_binary(["-P", "-i", "1", "-r", "1", "-S", "buffer-size=1,3", "-o", str(sweep_out)], cwd=cwd),
                "sweep base",
            )
            base_doc = self.assert_atomic_json_file(sweep_out)

            bad_rs = copy.deepcopy(base_doc)
            bad_rs["runs"][0]["result"]["configuration"]["resolved_sources"]["buffer_size"] = "explicit"
            bad_rs_path = cwd / "bad_sweep_rs.json"
            bad_rs_path.write_text(json.dumps(bad_rs), encoding="utf-8")
            self.verify_patterns_json(
                bad_rs_path,
                expected_exit=1,
                expected_consistent=False,
                expected_accepted=False,
            )

    def test_t3_f6_f7_verifier_rejects_oversized_or_deep_json_and_nonfinite(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t3_f6f7_") as tmp:
            cwd = Path(tmp)

            deep_obj: dict = {"leaf": 1}
            for _ in range(70):
                deep_obj = {"nest": deep_obj}
            deep_path = cwd / "deep.json"
            deep_path.write_text(json.dumps(deep_obj), encoding="utf-8")
            deep_verdict = self.verify_patterns_json(
                deep_path,
                expected_exit=2,
                expected_consistent=None,
                expected_accepted=False,
                include_binary=False,
                require_raw_timing=False,
            )
            self.assertEqual(deep_verdict["reason_code"], "unsupported-structure-limit")

            nan_path = cwd / "nonfinite.json"
            nan_path.write_text('{"schema_version": 1, "mode": "patterns", "bad": NaN}', encoding="utf-8")
            nan_verdict = self.verify_patterns_json(
                nan_path,
                expected_exit=1,
                expected_consistent=False,
                expected_accepted=False,
                include_binary=False,
                require_raw_timing=False,
            )
            self.assertEqual(nan_verdict["reason_code"], "nonfinite-json-number")


# =============================================================================
# Tier 4: Fault-Tolerance, Determinism, Tamper & Sibling Regression (8 tests)
# =============================================================================


class Tier4PatternsApplicationAndFaultTests(E2EAcceptanceTestCase):
    """Tier 4 signal interruption, determinism, live tamper gates, and sibling non-regression."""

    def test_t4_sigint_mid_direct_run_graceful_interruption(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t4_sigint_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "interrupted_direct.json"
            proc = self.run_binary_and_interrupt(
                ["-P", "-b", "16", "-t", "2", "-r", "40", "-o", str(out_path)],
                cwd=cwd,
                sig=signal.SIGINT,
                post_signal_delay=1.0,
            )
            self.assert_graceful_interruption(proc, "SIGINT direct -P")
            doc = self.assert_atomic_json_file(out_path)
            self.assertEqual(doc["mode"], "patterns")
            self.assertEqual(doc["status"], "interrupted")
            self.assertTrue(doc["interruption_requested"])
            self.assertFalse(doc["run_accepted"])
            self.verify_patterns_json(
                out_path,
                expected_exit=1,
                expected_consistent=True,
                expected_accepted=False,
                require_raw_timing=False,
            )

    def test_t4_sigterm_mid_sweep_graceful_interruption(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t4_sigterm_") as tmp:
            cwd = Path(tmp)
            out_path = cwd / "interrupted_sweep.json"
            proc = self.run_binary_and_interrupt(
                ["-P", "-t", "2", "-r", "15", "-S", "buffer-size=8,16,32,64", "-X", "4", "-o", str(out_path)],
                cwd=cwd,
                sig=signal.SIGTERM,
                post_signal_delay=1.0,
            )
            self.assert_graceful_interruption(proc, "SIGTERM sweep -P")
            doc = self.assert_atomic_json_file(out_path)
            self.assertEqual(doc["configuration"]["mode"], "sweep")
            self.assertEqual(doc["configuration"]["base_mode"], "patterns")
            self.assertEqual(doc["status"], "interrupted")
            self.assertFalse(doc["conclusions_valid"])
            self.verify_patterns_json(
                out_path,
                expected_exit=1,
                expected_consistent=True,
                expected_accepted=False,
                require_raw_timing=False,
            )

    def test_t4_seed_determinism_across_repeated_runs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t4_det_") as tmp:
            cwd = Path(tmp)
            p1, p2, p3 = cwd / "s1.json", cwd / "s2.json", cwd / "s3.json"
            for out_path, seed in ((p1, "123456"), (p2, "123456"), (p3, "654321")):
                proc = self.run_binary(
                    ["-P", "-b", "3", "-t", "2", "-i", "2", "-r", "1", "--seed", seed, "-o", str(out_path)],
                    cwd=cwd,
                )
                self.assert_run_succeeded(proc, f"determinism seed {seed}")
                self.verify_patterns_json(out_path)

            d1 = self.assert_atomic_json_file(p1)
            d2 = self.assert_atomic_json_file(p2)
            d3 = self.assert_atomic_json_file(p3)

            self.assertEqual(d1["seeds"], d2["seeds"])
            self.assertNotEqual(d1["seeds"], d3["seeds"])
            self.assertEqual(
                d1["resolved_plan"]["patterns"]["random"]["random_workload"]["random_indices_sha256"],
                d2["resolved_plan"]["patterns"]["random"]["random_workload"]["random_indices_sha256"],
            )
            self.assertNotEqual(
                d1["resolved_plan"]["patterns"]["random"]["random_workload"]["random_indices_sha256"],
                d3["resolved_plan"]["patterns"]["random"]["random_workload"]["random_indices_sha256"],
            )
            self.assertEqual(
                [op.get("expected_checksum") for op in d1["resolved_plan"]["operation_plans"]],
                [op.get("expected_checksum") for op in d2["resolved_plan"]["operation_plans"]],
            )
            self.assertNotEqual(
                [op.get("expected_checksum") for op in d1["resolved_plan"]["operation_plans"]],
                [op.get("expected_checksum") for op in d3["resolved_plan"]["operation_plans"]],
            )
            self.assertEqual(
                [m["execution"]["write_pattern_uint64_decimal"] for m in d1["measurements"]],
                [m["execution"]["write_pattern_uint64_decimal"] for m in d2["measurements"]],
            )
            self.assertNotEqual(
                [m["execution"]["write_pattern_uint64_decimal"] for m in d1["measurements"]],
                [m["execution"]["write_pattern_uint64_decimal"] for m in d3["measurements"]],
            )

    def test_t4_live_artifact_tamper_binary_and_raw_timing_gates(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t4_binraw_") as tmp:
            cwd = Path(tmp)
            live_out = cwd / "live.json"
            self.assert_run_succeeded(
                self.run_binary(["-P", "-b", "3", "-t", "2", "-i", "2", "-r", "1", "-o", str(live_out)], cwd=cwd),
                "live binary & raw timing",
            )
            base_doc = self.assert_atomic_json_file(live_out)
            self.verify_patterns_json(live_out)

            mismatch = self.verify_patterns_json(
                live_out,
                expected_exit=1,
                expected_consistent=False,
                expected_accepted=False,
                binary_path=Path(sys.executable),
            )
            self.assertEqual(mismatch["reason_code"], "binary-hash-mismatch")

            stripped = copy.deepcopy(base_doc)
            self._strip_cpu_raw(stripped)
            stripped_path = cwd / "stripped_raw.json"
            stripped_path.write_text(json.dumps(stripped), encoding="utf-8")
            ok_no_raw = self.verify_patterns_json(stripped_path, require_raw_timing=False)
            self.assertEqual(ok_no_raw["checks"]["timing"], "elapsed-only")
            reject_raw = self.verify_patterns_json(
                stripped_path,
                expected_exit=2,
                expected_consistent=None,
                expected_accepted=False,
                require_raw_timing=True,
            )
            self.assertEqual(reject_raw["reason_code"], "unsupported-missing-raw-timing")

    def test_t4_live_artifact_tamper_oracle_checksums_and_random_sha256(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t4_oracle_") as tmp:
            cwd = Path(tmp)
            live_out = cwd / "live_oracle.json"
            self.assert_run_succeeded(
                self.run_binary(["-P", "-b", "3", "-t", "2", "-i", "2", "-r", "2", "-o", str(live_out)], cwd=cwd),
                "live oracle tamper",
            )
            base_doc = self.assert_atomic_json_file(live_out)
            self.verify_patterns_json(live_out)

            # 1. Tamper read actual_run_checksum
            t_ck = copy.deepcopy(base_doc)
            read_row = next(r for r in t_ck["measurements"] if r["operation"] == "read")
            actual_rc = read_row["checksum"]["actual_run_checksum"]
            if isinstance(actual_rc, dict):
                k = next(iter(actual_rc))
                actual_rc[k] = "0" if actual_rc[k] != "0" else "1"
            else:
                read_row["checksum"]["actual_run_checksum"] = "0" if actual_rc != "0" else "1"
            self._assert_pat_tamper_rejected(cwd, "tamper_read_checksum.json", t_ck)

            # 2. Tamper random_indices_sha256
            t_sha = copy.deepcopy(base_doc)
            rw = t_sha["resolved_plan"]["patterns"]["random"]["random_workload"]
            orig_sha = rw["random_indices_sha256"]
            rw["random_indices_sha256"] = ("a" if orig_sha[0] != "a" else "b") + orig_sha[1:]
            self._assert_pat_tamper_rejected(cwd, "tamper_random_sha.json", t_sha)

            # 3. Tamper write_pattern_uint64_decimal (even/zero)
            t_wp = copy.deepcopy(base_doc)
            write_row = next(r for r in t_wp["measurements"] if r["operation"] == "write")
            write_row["execution"]["write_pattern_uint64_decimal"] = "2"
            self._assert_pat_tamper_rejected(cwd, "tamper_write_pattern.json", t_wp)

            # 4. Tamper Mach delta_ticks
            t_ticks = copy.deepcopy(base_doc)
            raw = t_ticks["measurements"][0]["execution"]["timing"]["cpu_raw"]
            raw["delta_ticks"] = str(int(raw["delta_ticks"]) + 1)
            self._assert_pat_tamper_rejected(cwd, "tamper_delta_ticks.json", t_ticks)

            # 5. Tamper relative_to_forward_pct
            t_rel = copy.deepcopy(base_doc)
            t_rel["aggregates"]["patterns"]["random"]["read"]["relative_to_forward_pct"] = 999.999
            self._assert_pat_tamper_rejected(cwd, "tamper_rel_pct.json", t_rel)

    def test_t4_live_artifact_tamper_validation_checks_and_sentinel_isolation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t4_val_") as tmp:
            cwd = Path(tmp)
            live_out = cwd / "live_val.json"
            self.assert_run_succeeded(
                self.run_binary(["-P", "-b", "3", "-t", "2", "-i", "2", "-r", "1", "-o", str(live_out)], cwd=cwd),
                "live validation tamper",
            )
            base_doc = self.assert_atomic_json_file(live_out)
            self.verify_patterns_json(live_out)

            # 1. Tamper post-validation-structure
            t_struct = copy.deepcopy(base_doc)
            t_struct["measurements"][0]["execution"]["validation"]["checks"][0]["valid"] = False
            self._assert_pat_tamper_rejected(cwd, "tamper_struct_check.json", t_struct)

            # 2. Tamper untouched-sentinel-isolation on sparse write/copy
            t_sent = copy.deepcopy(base_doc)
            sparse_row = next(
                r
                for r in t_sent["measurements"]
                if r["pattern"] == "strided_4096" and r["operation"] in ("write", "copy")
            )
            sparse_row["execution"]["validation"]["checks"][4]["applicable"] = False
            sparse_row["execution"]["validation"]["checks"][4]["evaluated"] = None
            sparse_row["execution"]["validation"]["checks"][4]["valid"] = None
            self._assert_pat_tamper_rejected(cwd, "tamper_sentinel_check.json", t_sent)

            # 3. Tamper run_accepted
            t_acc = copy.deepcopy(base_doc)
            t_acc["run_accepted"] = False
            self._assert_pat_tamper_rejected(cwd, "tamper_run_accepted.json", t_acc)

            # 4. Duplicate JSON key
            dup_path = cwd / "dup_key.json"
            raw_text = live_out.read_text(encoding="utf-8")
            dup_text = raw_text.replace('"mode":', '"schema_version": 1, "mode":', 1)
            self.assertNotEqual(dup_text, raw_text)
            dup_path.write_text(dup_text, encoding="utf-8")
            dup_verdict = self.verify_patterns_json(
                dup_path,
                expected_exit=1,
                expected_consistent=False,
                expected_accepted=False,
            )
            self.assertEqual(dup_verdict["reason_code"], "duplicate-json-key")

    def test_t4_sweep_live_artifact_tamper_gates(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t4_swtamper_") as tmp:
            cwd = Path(tmp)
            sweep_out = cwd / "live_sweep.json"
            self.assert_run_succeeded(
                self.run_binary(["-P", "-i", "1", "-r", "1", "-S", "buffer-size=1,3", "-o", str(sweep_out)], cwd=cwd),
                "live sweep for tamper",
            )
            base_doc = self.assert_atomic_json_file(sweep_out)
            self.verify_patterns_json(sweep_out)

            t_conc = copy.deepcopy(base_doc)
            t_conc["conclusions_valid"] = False
            self._assert_pat_tamper_rejected(cwd, "tamper_sw_conclusions.json", t_conc)

            t_cnt = copy.deepcopy(base_doc)
            t_cnt["completed_runs"] = 1
            self._assert_pat_tamper_rejected(cwd, "tamper_sw_completed.json", t_cnt)

            t_nested = copy.deepcopy(base_doc)
            t_nested["runs"][1]["result"]["run_accepted"] = False
            self._assert_pat_tamper_rejected(cwd, "tamper_sw_nested.json", t_nested)

    def test_t4_sibling_modes_non_regression(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e2e_pat_t4_siblings_") as tmp:
            cwd = Path(tmp)
            for flag in ("-B", "-L", "-T", "-C", "-G", "-M"):
                with self.subTest(help_flag=flag):
                    proc = self.run_binary([flag, "-h"], cwd=cwd)
                    self.assert_run_succeeded(proc, f"sibling help {flag}")
                    self.assertEqual(proc.stderr.strip(), "")
                    self.assertIn("Usage:", proc.stdout)

            bw_out = cwd / "bw_smoke.json"
            self.assert_run_succeeded(
                self.run_binary(["-B", "-b", "8", "-k", "0", "-i", "2", "-r", "1", "-o", str(bw_out)], cwd=cwd),
                "sibling -B smoke",
            )
            self.assert_atomic_json_file(bw_out)
            self.verify_bandwidth_json(bw_out)

            lat_out = cwd / "lat_smoke.json"
            self.assert_run_succeeded(
                self.run_binary(["-L", "-b", "0", "-k", "64", "-i", "2", "-r", "1", "-n", "5", "-o", str(lat_out)], cwd=cwd),
                "sibling -L smoke",
            )
            self.assert_atomic_json_file(lat_out)
            self.verify_latency_json(lat_out)

    @staticmethod
    def _strip_cpu_raw(node: object) -> None:
        if isinstance(node, dict):
            if "cpu_raw" in node:
                node["cpu_raw"] = None
            for val in node.values():
                Tier4PatternsApplicationAndFaultTests._strip_cpu_raw(val)
        elif isinstance(node, list):
            for item in node:
                Tier4PatternsApplicationAndFaultTests._strip_cpu_raw(item)

    def _assert_pat_tamper_rejected(self, cwd: Path, filename: str, mutated_doc: dict) -> None:
        path = cwd / filename
        path.write_text(json.dumps(mutated_doc), encoding="utf-8")
        self.verify_patterns_json(
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
        description="Run Tier 1-4 E2E acceptance tests for --patterns (-P)."
    )
    parser.add_argument(
        "--binary",
        default=None,
        help="Path to memory_benchmark binary (default: $MEMORY_BENCHMARK_BIN, $MEMBENCH_BINARY, or ./memory_benchmark)",
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

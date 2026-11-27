#!/usr/bin/env python3
# Copyright 2026 Timo Heimonen
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
@file plot_M4vsM5_benchmark_comparison.py
@brief Side-by-side memory bandwidth and latency comparison: Apple M4 vs M5.

Reads four schema-1 JSON files produced by macos-memory-benchmark (--cpu-bandwidth
and --cpu-latency --compare-locality for each machine) and renders a two-subplot
figure:
  - Top:    Memory bandwidth (Copy / Read / Write, GB/s)
  - Bottom: Memory latency  (L1 / L2 / 16 KiB locality / global random, ns)
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

VALID_METRICS = ("average", "median", "p90", "p95", "p99", "min", "max", "stddev")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Apple M4 vs M5 memory bandwidth and latency.")
    parser.add_argument(
        "--m4-bandwidth-file", "--m4-bandwidth",
        dest="m4_bandwidth_file",
        required=True,
        help="Path to a supported CPU bandwidth schema-1 JSON for M4")
    parser.add_argument(
        "--m4-latency-file", "--m4-latency",
        dest="m4_latency_file",
        required=True,
        help="Path to a supported CPU latency schema-1 JSON (with --compare-locality) for M4")
    parser.add_argument(
        "--m5-bandwidth-file", "--m5-bandwidth",
        dest="m5_bandwidth_file",
        required=True,
        help="Path to a supported CPU bandwidth schema-1 JSON for M5")
    parser.add_argument(
        "--m5-latency-file", "--m5-latency",
        dest="m5_latency_file",
        required=True,
        help="Path to a supported CPU latency schema-1 JSON (with --compare-locality) for M5")
    parser.add_argument(
        "--metric",
        default="median",
        choices=VALID_METRICS,
        help="Statistic to plot (default: median)")
    parser.add_argument(
        "--save",
        default=None,
        help="Save plot to this file path (e.g. /tmp/m4vsm5.png)")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive plot window")
    return parser.parse_args()


def resolve_input_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path

    script_relative = Path(__file__).resolve().parent / path
    if script_relative.exists():
        return script_relative

    repo_relative = Path(__file__).resolve().parent.parent / path
    if repo_relative.exists():
        return repo_relative

    raise RuntimeError(f"Input JSON file not found: {raw_path}")


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _linear_percentile(sorted_values: list[float], percentile: float) -> float:
    n = len(sorted_values)
    if n == 1:
        return float(sorted_values[0])
    index = percentile * (n - 1)
    lower = int(index)
    upper = lower + 1
    if upper >= n:
        return float(sorted_values[-1])
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _compute_metric(values: list[float], metric: str, label: str) -> float:
    if not values:
        raise RuntimeError(f"Missing measured locality rounds for '{label}'.")
    sorted_vals = sorted(float(v) for v in values)
    if metric == "average":
        return sum(sorted_vals) / len(sorted_vals)
    if metric == "median":
        return _linear_percentile(sorted_vals, 0.50)
    if metric == "p90":
        return _linear_percentile(sorted_vals, 0.90)
    if metric == "p95":
        return _linear_percentile(sorted_vals, 0.95)
    if metric == "p99":
        return _linear_percentile(sorted_vals, 0.99)
    if metric == "min":
        return sorted_vals[0]
    if metric == "max":
        return sorted_vals[-1]
    if metric == "stddev":
        if len(sorted_vals) <= 1:
            return 0.0
        mean = sum(sorted_vals) / len(sorted_vals)
        variance = sum((v - mean) ** 2 for v in sorted_vals) / (len(sorted_vals) - 1)
        return math.sqrt(variance)
    raise RuntimeError(f"Unsupported metric '{metric}'.")


def require_supported_cpu_result(
    data: object,
    source: str,
    *,
    expected_mode: str,
    expected_methodology: str,
    label: str,
) -> tuple[dict, dict, dict]:
    """Require the supported CPU schema-1 and methodology contract."""
    software = data.get("software") if isinstance(data, dict) else None
    configuration = data.get("configuration") if isinstance(data, dict) else None
    environment = data.get("environment") if isinstance(data, dict) else None
    if not (
        isinstance(data, dict)
        and type(data.get("schema_version")) is int
        and data["schema_version"] == 1
        and data.get("mode") == expected_mode
        and data.get("methodology_version") == expected_methodology
        and isinstance(software, dict)
        and isinstance(software.get("version"), str)
        and bool(software["version"])
        and isinstance(configuration, dict)
        and isinstance(configuration.get("output_file"), str)
        and isinstance(environment, dict)
        and data.get("status") == "complete"
        and data.get("results_complete") is True
        and data.get("run_accepted") is True
    ):
        raise RuntimeError(
            f"{source} is not a complete supported {label} schema-1 result")
    return software, configuration, environment


def _stat(obj: dict, *keys: str, metric: str) -> float:
    """Navigate nested dicts and return the requested statistic value."""
    node: object = obj
    for key in keys:
        if not isinstance(node, dict):
            raise RuntimeError(
                f"Metric '{metric}' not found at path {' -> '.join(keys)}")
        node = node.get(key)
    if not isinstance(node, dict):
        raise RuntimeError(
            f"Metric '{metric}' not found at path {' -> '.join(keys)}")
    stats = node.get("statistics")
    if not isinstance(stats, dict):
        raise RuntimeError(
            f"Metric '{metric}' not found at path {' -> '.join(keys)}")
    value = stats.get(metric)
    if not _is_number(value):
        raise RuntimeError(
            f"Metric '{metric}' not found at path {' -> '.join(keys)}")
    return float(value)


def load_bandwidth_data(path: Path, metric: str) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    software, _, environment = require_supported_cpu_result(
        data,
        str(path),
        expected_mode="cpu_bandwidth",
        expected_methodology="cpu-bandwidth-v1-arm64-calibrated-seeded-balanced",
        label="CPU bandwidth",
    )
    cpu_name = str(environment.get("processor_name") or "Unknown CPU")
    version = software["version"]

    aggregates = data.get("aggregates")
    targets = aggregates.get("targets") if isinstance(aggregates, dict) else None
    mm = targets.get("main_memory") if isinstance(targets, dict) else None
    if not isinstance(mm, dict):
        raise RuntimeError(f"{path} is missing main_memory bandwidth aggregates")

    return {
        "cpu_name": cpu_name,
        "version": version,
        "bw_copy": _stat(mm, "copy", metric=metric),
        "bw_read": _stat(mm, "read", metric=metric),
        "bw_write": _stat(mm, "write", metric=metric),
    }


def _extract_locality_values(data: dict, metric: str, source: Path) -> tuple[float, float]:
    aggregates = data.get("aggregates")
    loc_agg = aggregates.get("locality_comparison") if isinstance(aggregates, dict) else None
    if not isinstance(loc_agg, dict):
        raise RuntimeError(
            f"{source} does not include locality comparison results; re-run --cpu-latency with --compare-locality"
        )

    if metric == "median":
        loc_16k = loc_agg.get("locality_16k_latency_ns")
        loc_global = loc_agg.get("global_random_latency_ns")
        if _is_number(loc_16k) and _is_number(loc_global):
            return float(loc_16k), float(loc_global)
        raise RuntimeError(f"{source} is missing median locality comparison values")

    loc_raw = data.get("locality_comparison")
    rounds = loc_raw.get("rounds") if isinstance(loc_raw, dict) else None
    if not isinstance(rounds, list) or not rounds:
        raise RuntimeError(
            f"{source} is missing locality_comparison.rounds needed for metric '{metric}'"
        )

    vals_16k: list[float] = []
    vals_global: list[float] = []
    for rnd in rounds:
        if not isinstance(rnd, dict):
            continue
        sub16 = rnd.get("locality_16k")
        if isinstance(sub16, dict) and sub16.get("status") == "measured" and _is_number(sub16.get("latency_ns")):
            vals_16k.append(float(sub16["latency_ns"]))
        subg = rnd.get("locality_global")
        if isinstance(subg, dict) and subg.get("status") == "measured" and _is_number(subg.get("latency_ns")):
            vals_global.append(float(subg["latency_ns"]))

    return (
        _compute_metric(vals_16k, metric, "16 KiB locality latency"),
        _compute_metric(vals_global, metric, "Global-random latency"),
    )


def load_latency_data(path: Path, metric: str) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    software, _, environment = require_supported_cpu_result(
        data,
        str(path),
        expected_mode="cpu_latency",
        expected_methodology="cpu-latency-v1-arm64-calibrated-seeded-balanced",
        label="CPU latency",
    )
    cpu_name = str(environment.get("processor_name") or "Unknown CPU")
    version = software["version"]

    aggregates = data.get("aggregates")
    targets = aggregates.get("targets") if isinstance(aggregates, dict) else None
    if not isinstance(targets, dict):
        raise RuntimeError(f"{path} is missing latency target aggregates")

    l1_lat = _stat(targets, "l1", metric=metric)
    l2_lat = _stat(targets, "l2", metric=metric)
    locality_16k, global_random = _extract_locality_values(data, metric, path)

    return {
        "cpu_name": cpu_name,
        "version": version,
        "l1_lat": l1_lat,
        "l2_lat": l2_lat,
        "locality_16k": locality_16k,
        "global_random": global_random,
    }


def load_machine_data(bw_path: Path, lat_path: Path, metric: str) -> dict:
    bw = load_bandwidth_data(bw_path, metric)
    lat = load_latency_data(lat_path, metric)
    if bw["cpu_name"] != lat["cpu_name"]:
        raise RuntimeError(
            f"CPU mismatch between bandwidth ({bw['cpu_name']}) and latency ({lat['cpu_name']}) files"
        )
    version = bw["version"] if bw["version"] == lat["version"] else f"{bw['version']}/{lat['version']}"
    return {
        "cpu_name": bw["cpu_name"],
        "version": version,
        "bw_copy": bw["bw_copy"],
        "bw_read": bw["bw_read"],
        "bw_write": bw["bw_write"],
        "l1_lat": lat["l1_lat"],
        "l2_lat": lat["l2_lat"],
        "locality_16k": lat["locality_16k"],
        "global_random": lat["global_random"],
    }


def _bar_labels(ax, bars):
    """Annotate each bar with its numeric value."""
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height * 1.01,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8)


def main():
    args = parse_args()

    m4_bw_path = resolve_input_path(args.m4_bandwidth_file)
    m4_lat_path = resolve_input_path(args.m4_latency_file)
    m5_bw_path = resolve_input_path(args.m5_bandwidth_file)
    m5_lat_path = resolve_input_path(args.m5_latency_file)

    m4 = load_machine_data(m4_bw_path, m4_lat_path, args.metric)
    m5 = load_machine_data(m5_bw_path, m5_lat_path, args.metric)

    metric_label = args.metric.capitalize()
    m4_label = f"{m4['cpu_name']} (v{m4['version']})"
    m5_label = f"{m5['cpu_name']} (v{m5['version']})"

    color_m4 = "#1f77b4"  # blue
    color_m5 = "#ff7f0e"  # orange

    fig, (ax_bw, ax_lat) = plt.subplots(2, 1, figsize=(10, 9))
    fig.suptitle(
        f"Memory Performance: {m4['cpu_name']} vs {m5['cpu_name']}  [{metric_label}]",
        fontsize=13,
        fontweight="bold")

    # --- Bandwidth subplot ---
    bw_labels = ["Copy", "Read", "Write"]
    m4_bw = [m4["bw_copy"], m4["bw_read"], m4["bw_write"]]
    m5_bw = [m5["bw_copy"], m5["bw_read"], m5["bw_write"]]

    import numpy as np
    x = np.arange(len(bw_labels))
    width = 0.35

    bars_m4 = ax_bw.bar(x - width / 2, m4_bw, width, label=m4_label, color=color_m4)
    bars_m5 = ax_bw.bar(x + width / 2, m5_bw, width, label=m5_label, color=color_m5)
    _bar_labels(ax_bw, bars_m4)
    _bar_labels(ax_bw, bars_m5)

    ax_bw.set_title("Memory Bandwidth (GB/s)")
    ax_bw.set_ylabel("GB/s")
    ax_bw.set_xticks(x)
    ax_bw.set_xticklabels(bw_labels)
    ax_bw.legend()
    ax_bw.grid(axis="y", alpha=0.3)
    ax_bw.set_ylim(0, max(m4_bw + m5_bw) * 1.18)

    # --- Latency subplot ---
    lat_labels = ["L1 Cache", "L2 Cache", "16 KiB Locality", "Global Random"]
    m4_lat = [m4["l1_lat"], m4["l2_lat"], m4["locality_16k"], m4["global_random"]]
    m5_lat = [m5["l1_lat"], m5["l2_lat"], m5["locality_16k"], m5["global_random"]]

    x2 = np.arange(len(lat_labels))

    bars_m4_lat = ax_lat.bar(x2 - width / 2, m4_lat, width, label=m4_label, color=color_m4)
    bars_m5_lat = ax_lat.bar(x2 + width / 2, m5_lat, width, label=m5_label, color=color_m5)
    _bar_labels(ax_lat, bars_m4_lat)
    _bar_labels(ax_lat, bars_m5_lat)

    ax_lat.set_title("Memory Latency (ns)")
    ax_lat.set_ylabel("ns")
    ax_lat.set_xticks(x2)
    ax_lat.set_xticklabels(lat_labels)
    ax_lat.legend()
    ax_lat.grid(axis="y", alpha=0.3)
    ax_lat.set_ylim(0, max(m4_lat + m5_lat) * 1.18)

    fig.text(
        0.5,
        0.01,
        "https://github.com/timoheimonen/macOS-memory-benchmark",
        ha="center",
        fontsize=9)
    plt.tight_layout(rect=(0.01, 0.03, 0.99, 0.97))

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot: {save_path}")

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()

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
@file plot_M4vsM5.py
@brief Side-by-side memory bandwidth and latency comparison: Apple M4 vs M5.

Reads two benchmark JSON files produced by macos-memory-benchmark and renders
a two-subplot figure:
  - Top:    Memory bandwidth (Copy / Read / Write, GB/s)
  - Bottom: Memory latency  (L1 / L2 / TLB Hit / TLB Miss, ns)
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

VALID_METRICS = ("average", "median", "p90", "p95", "p99", "min", "max", "stddev")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Apple M4 vs M5 memory bandwidth and latency.")
    parser.add_argument(
        "-m4", "--m4-file",
        default="results/0.53.7/MacMiniM4_benchmark.json",
        help="Path to M4 benchmark JSON (default: results/0.53.7/MacMiniM4_benchmark.json)")
    parser.add_argument(
        "-m5", "--m5-file",
        default="results/0.53.8/MacbookAirM5_benchmark.json",
        help="Path to M5 benchmark JSON (default: results/0.53.8/MacbookAirM5_benchmark.json)")
    parser.add_argument(
        "--metric",
        default="average",
        choices=VALID_METRICS,
        help="Statistic to plot (default: average)")
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


def _stat(obj: dict, *keys: str, metric: str) -> float:
    """Navigate nested dicts and return the requested statistic value."""
    node = obj
    for key in keys:
        node = node.get(key, {})
    stats = node.get("statistics", {})
    value = stats.get(metric)
    if value is None:
        raise RuntimeError(
            f"Metric '{metric}' not found at path {' -> '.join(keys)}")
    return float(value)


def load_data(path: Path, metric: str) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    config = data.get("configuration", {})
    cpu_name = config.get("cpu_name") or data.get("cpu_name") or "Unknown CPU"
    version = data.get("version", "?")

    mm = data.get("main_memory", {})
    bw = mm.get("bandwidth", {})
    cache = data.get("cache", {})
    tlb = mm.get("latency", {}).get("auto_tlb_breakdown", {})

    return {
        "cpu_name": cpu_name,
        "version": version,
        "bw_copy":   _stat(bw,    "copy_gb_s",  metric=metric),
        "bw_read":   _stat(bw,    "read_gb_s",  metric=metric),
        "bw_write":  _stat(bw,    "write_gb_s", metric=metric),
        "l1_lat":    _stat(cache, "l1", "latency", "average_ns", metric=metric),
        "l2_lat":    _stat(cache, "l2", "latency", "average_ns", metric=metric),
        "tlb_hit":   _stat(tlb,   "tlb_hit_ns",  metric=metric),
        "tlb_miss":  _stat(tlb,   "tlb_miss_ns", metric=metric),
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

    m4_path = resolve_input_path(args.m4_file)
    m5_path = resolve_input_path(args.m5_file)

    m4 = load_data(m4_path, args.metric)
    m5 = load_data(m5_path, args.metric)

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
    lat_labels = ["L1 Cache", "L2 Cache", "TLB Hit", "TLB Miss"]
    m4_lat = [m4["l1_lat"], m4["l2_lat"], m4["tlb_hit"], m4["tlb_miss"]]
    m5_lat = [m5["l1_lat"], m5["l2_lat"], m5["tlb_hit"], m5["tlb_miss"]]

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
        "github.com/timoheimonen/macOS-memory-benchmark",
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

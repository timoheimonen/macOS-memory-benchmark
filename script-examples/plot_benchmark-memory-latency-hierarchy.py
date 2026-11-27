#!/usr/bin/env python3
import argparse
import json
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource]

# Uses JSON output from, for example:
# memory_benchmark --cpu-latency --compare-locality --count 5 --output results/<file>.json

ALLOWED_METRICS = {
    "average": "Average",
    "median": "Median",
    "p90": "P90",
    "p95": "P95",
    "p99": "P99",
    "min": "Min",
    "max": "Max",
    "stddev": "Stddev",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot memory hierarchy latencies from benchmark JSON or text statistics output."
    )
    parser.add_argument(
        "-f",
        "--file",
        required=True,
        help="Path to a supported CPU latency schema-1 JSON (run with --compare-locality) or console-text statistics file",
    )
    parser.add_argument(
        "--metric",
        choices=sorted(ALLOWED_METRICS.keys()),
        default="median",
        help="Statistic to plot (default: median)",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Save plot to file (for example: memory_hierarchy.png)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive plot window",
    )
    parser.add_argument(
        "--cpu-name",
        default=None,
        help="Override CPU name shown in the title",
    )
    for arg in sys.argv[1:]:
        if arg.startswith("-file"):
            parser.error("legacy option '-file' is no longer supported; use '-f' or '--file'")
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

    raise RuntimeError(f"Input file not found: {raw_path}")


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def linear_percentile(sorted_values: list[float], percentile: float) -> float:
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


def compute_metric(values: list[float], metric: str, label: str) -> float:
    if not values:
        raise RuntimeError(f"Missing measured locality rounds for '{label}'.")
    sorted_vals = sorted(float(v) for v in values)
    if metric == "average":
        return sum(sorted_vals) / len(sorted_vals)
    if metric == "median":
        return linear_percentile(sorted_vals, 0.50)
    if metric == "p90":
        return linear_percentile(sorted_vals, 0.90)
    if metric == "p95":
        return linear_percentile(sorted_vals, 0.95)
    if metric == "p99":
        return linear_percentile(sorted_vals, 0.99)
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


def require_supported_cpu_latency_result(data: object, source: str) -> tuple[dict, dict, dict]:
    """Require the supported CPU latency schema-1 and methodology contract."""
    software = data.get("software") if isinstance(data, dict) else None
    configuration = data.get("configuration") if isinstance(data, dict) else None
    environment = data.get("environment") if isinstance(data, dict) else None
    if not (
        isinstance(data, dict)
        and type(data.get("schema_version")) is int
        and data["schema_version"] == 1
        and data.get("mode") == "cpu_latency"
        and data.get("methodology_version") == "cpu-latency-v1-arm64-calibrated-seeded-balanced"
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
            f"{source} is not a complete supported CPU latency schema-1 result")
    return software, configuration, environment


def pick_stat(metric_block: dict, metric: str, label: str) -> float:
    try:
        val = metric_block["statistics"][metric]
        if not is_number(val):
            raise ValueError("non-numeric")
        return float(val)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Missing supported '{metric}' statistic for '{label}'.") from exc


def infer_cpu_name_from_path(path: Path) -> str:
    lower_name = path.name.lower()
    match = re.search(r"\bm([1-9])\b", lower_name)
    if not match:
        match = re.search(r"m([1-9])", lower_name)
    if match:
        return f"Apple M{match.group(1)}"
    return "Unknown CPU"


def parse_text_statistics(path: Path, metric: str):
    if metric != "median":
        raise RuntimeError(
            f"Console text output from --cpu-latency only reports P50 (median) hierarchy and locality values; "
            f"cannot plot '--metric {metric}'. Re-run with -o <file>.json for '{metric}'."
        )

    number = r"([-+]?[0-9]+(?:\.[0-9]+)?)"
    cpu_re = re.compile(r"^\s*CPU:\s*(.+?)(?:\s*\(|$)")
    l1_re = re.compile(rf"^\s*L1 Cache\s*:\s*P50={number}\s*ns\b")
    l2_re = re.compile(rf"^\s*L2 Cache\s*:\s*P50={number}\s*ns\b")
    loc_re = re.compile(
        rf"^\s*Locality Comparison:\s*16KB\s+P50={number}\s*ns,\s*Global\s+P50={number}\s*ns,\s*Delta={number}\s*ns"
    )

    cpu_name = None
    l1_val = None
    l2_val = None
    loc_16k = None
    loc_global = None
    loc_delta = None

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if cpu_name is None:
            m_cpu = cpu_re.match(raw_line)
            if m_cpu and m_cpu.group(1).strip():
                cpu_name = m_cpu.group(1).strip()
        m_l1 = l1_re.match(raw_line)
        if m_l1:
            l1_val = float(m_l1.group(1))
            continue
        m_l2 = l2_re.match(raw_line)
        if m_l2:
            l2_val = float(m_l2.group(1))
            continue
        m_loc = loc_re.match(raw_line)
        if m_loc:
            loc_16k = float(m_loc.group(1))
            loc_global = float(m_loc.group(2))
            loc_delta = float(m_loc.group(3))
            continue

    missing = []
    if l1_val is None:
        missing.append("L1 Cache P50")
    if l2_val is None:
        missing.append("L2 Cache P50")
    if loc_16k is None or loc_global is None or loc_delta is None:
        missing.append("Locality Comparison (16KB P50, Global P50, Delta)")
    if missing:
        raise RuntimeError(
            "Console text file is missing required '--cpu-latency --compare-locality' fields: "
            + ", ".join(missing)
        )

    categories = [
        "L1 Cache",
        "L2 Cache",
        "Main Memory\n(16 KiB locality)",
        "Main Memory\n(Global random)",
    ]
    latencies = [l1_val, l2_val, loc_16k, loc_global]
    return cpu_name or infer_cpu_name_from_path(path), categories, latencies, loc_delta, None


def extract_locality_metrics(data: dict, metric: str) -> tuple[float, float, float]:
    aggregates = data.get("aggregates")
    loc_agg = aggregates.get("locality_comparison") if isinstance(aggregates, dict) else None
    if not isinstance(loc_agg, dict):
        raise RuntimeError(
            "JSON is missing locality comparison results. "
            "Re-run --cpu-latency with --compare-locality."
        )

    if metric == "median":
        loc_16k = loc_agg.get("locality_16k_latency_ns")
        loc_global = loc_agg.get("global_random_latency_ns")
        delta = loc_agg.get("locality_latency_delta_ns")
        if is_number(loc_16k) and is_number(loc_global) and is_number(delta):
            return float(loc_16k), float(loc_global), float(delta)
        raise RuntimeError("JSON is missing median locality comparison values.")

    loc_raw = data.get("locality_comparison")
    rounds = loc_raw.get("rounds") if isinstance(loc_raw, dict) else None
    if not isinstance(rounds, list) or not rounds:
        raise RuntimeError(
            f"JSON is missing locality_comparison.rounds required to compute '{metric}'."
        )

    vals_16k: list[float] = []
    vals_global: list[float] = []
    paired_deltas: list[float] = []
    for rnd in rounds:
        if not isinstance(rnd, dict):
            continue
        sub16 = rnd.get("locality_16k")
        subg = rnd.get("locality_global")
        v16 = (
            float(sub16["latency_ns"])
            if isinstance(sub16, dict)
            and sub16.get("status") == "measured"
            and is_number(sub16.get("latency_ns"))
            else None
        )
        vg = (
            float(subg["latency_ns"])
            if isinstance(subg, dict)
            and subg.get("status") == "measured"
            and is_number(subg.get("latency_ns"))
            else None
        )
        if v16 is not None:
            vals_16k.append(v16)
        if vg is not None:
            vals_global.append(vg)
        if v16 is not None and vg is not None:
            paired_deltas.append(vg - v16)

    loc_16k_val = compute_metric(vals_16k, metric, "16 KiB locality latency")
    loc_global_val = compute_metric(vals_global, metric, "Global-random latency")
    if metric == "stddev":
        delta_val = compute_metric(paired_deltas, "stddev", "Locality latency delta")
    else:
        delta_val = loc_global_val - loc_16k_val
    return loc_16k_val, loc_global_val, delta_val


def load_latency_data(path: Path, metric: str):
    raw_text = path.read_text(encoding="utf-8", errors="replace")
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        return parse_text_statistics(path, metric)

    software, _, environment = require_supported_cpu_latency_result(data, str(path))
    cpu_name = str(environment.get("processor_name") or "Unknown CPU")
    version = software["version"]

    try:
        targets = data["aggregates"]["targets"]
        l1_block = targets["l1"]
        l2_block = targets["l2"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            "JSON is missing required L1/L2 target latency aggregates for memory hierarchy plot. "
            "Expected supported CPU latency schema-1 output (without --cache-size override)."
        ) from exc

    loc_16k, loc_global, delta = extract_locality_metrics(data, metric)

    categories = [
        "L1 Cache",
        "L2 Cache",
        "Main Memory\n(16 KiB locality)",
        "Main Memory\n(Global random)",
    ]
    latencies = [
        pick_stat(l1_block, metric, "L1 latency"),
        pick_stat(l2_block, metric, "L2 latency"),
        loc_16k,
        loc_global,
    ]
    return cpu_name, categories, latencies, delta, version


def main():
    args = parse_args()
    input_path = resolve_input_path(args.file)
    cpu_name, categories, latencies, locality_delta, version = load_latency_data(input_path, args.metric)
    if args.cpu_name:
        cpu_name = args.cpu_name

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(categories, latencies, color=colors, edgecolor="black", alpha=0.8)

    metric_name = ALLOWED_METRICS[args.metric]
    title_suffix = f" v{version}" if version else ""
    ax.set_ylabel("Latency (ns)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"{cpu_name} Memory Latency Hierarchy ({metric_name})\n"
        f"https://github.com/timoheimonen/macOS-memory-benchmark{title_suffix}",
        fontsize=10,
        fontweight="bold",
        pad=20,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    ymax = max(latencies) * 1.18 if latencies else 1.0
    ax.set_ylim(0, ymax)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + (ymax * 0.015),
            f"{height:.2f} ns",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    miss_x = 3
    miss_y = latencies[3]
    ax.annotate(
        f"Locality Delta\n~{locality_delta:.2f} ns",
        xy=(miss_x, miss_y),
        xycoords="data",
        xytext=(2.35, ymax * 0.58),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "connectionstyle": "arc3,rad=.2", "color": "black"},
        bbox={"boxstyle": "round", "fc": "w", "ec": "0.5", "alpha": 0.9},
        fontsize=10,
        fontweight="bold",
    )

    plt.tight_layout()

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

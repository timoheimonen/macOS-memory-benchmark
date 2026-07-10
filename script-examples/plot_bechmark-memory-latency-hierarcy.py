#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource]

# Uses JSON output from, for example:
# memory_benchmark --benchmark --count 5 --only-latency --output results/<file>.json

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
        default="results/old/macminim4_count5_latency.json",
        help="Path to benchmark JSON/text file (default: results/old/macminim4_count5_latency.json)",
    )
    parser.add_argument(
        "--metric",
        choices=sorted(ALLOWED_METRICS.keys()),
        default="average",
        help="Statistic to plot (default: average)",
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


def pick_stat_or_values(metric_block: dict, metric: str, label: str) -> float:
    statistics = metric_block.get("statistics")
    if isinstance(statistics, dict):
        if metric in statistics:
            return float(statistics[metric])
        if "average" in statistics:
            return float(statistics["average"])

    values = metric_block.get("values")
    if isinstance(values, list) and values:
        return float(sum(values) / len(values))

    raise RuntimeError(f"Missing usable metric data for '{label}'.")


def infer_cpu_name_from_path(path: Path) -> str:
    lower_name = path.name.lower()
    match = re.search(r"\bm([1-9])\b", lower_name)
    if not match:
        match = re.search(r"m([1-9])", lower_name)
    if match:
        return f"Apple M{match.group(1)}"
    return "Unknown CPU"


def parse_text_statistics(path: Path, metric: str):
    metric_aliases = {
        "average": "average",
        "median": "median",
        "p90": "p90",
        "p95": "p95",
        "p99": "p99",
        "min": "min",
        "max": "max",
        "stddev": "stddev",
    }

    stat_patterns = {
        "average": re.compile(r"^Average:\s*([0-9]+(?:\.[0-9]+)?)$"),
        "median": re.compile(r"^Median\s*\(P50\):\s*([0-9]+(?:\.[0-9]+)?)"),
        "p90": re.compile(r"^P90:\s*([0-9]+(?:\.[0-9]+)?)$"),
        "p95": re.compile(r"^P95:\s*([0-9]+(?:\.[0-9]+)?)$"),
        "p99": re.compile(r"^P99:\s*([0-9]+(?:\.[0-9]+)?)$"),
        "stddev": re.compile(r"^Stddev:\s*([0-9]+(?:\.[0-9]+)?)$"),
        "min": re.compile(r"^Min:\s*([0-9]+(?:\.[0-9]+)?)$"),
        "max": re.compile(r"^Max:\s*([0-9]+(?:\.[0-9]+)?)$"),
    }

    section_aliases = {
        "L1 Cache": "l1",
        "L2 Cache": "l2",
        "16 KiB Locality Latency (ns)": "locality_16k",
        "Global-Random Latency (ns)": "global_random",
        "Locality Latency Delta, Global - 16 KiB (ns)": "locality_delta",
        # Historical pre-schema-2 console labels remain readable.
        "TLB Hit Latency (ns)": "locality_16k",
        "TLB Miss Latency (ns)": "global_random",
        "Estimated Page-Walk Penalty (ns)": "locality_delta",
    }

    results = {"l1": {}, "l2": {}, "locality_16k": {}, "global_random": {}, "locality_delta": {}}
    current_section = None

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue

        if stripped.endswith(":"):
            title = stripped[:-1]
            current_section = section_aliases.get(title)
            continue

        if current_section is None:
            continue

        for stat_name, pattern in stat_patterns.items():
            match = pattern.match(stripped)
            if match:
                results[current_section][stat_name] = float(match.group(1))
                break

    def require_metric(section_key: str, label: str):
        values = results.get(section_key, {})
        wanted_key = metric_aliases[metric]
        if wanted_key in values:
            return values[wanted_key]
        if "average" in values:
            return values["average"]
        raise RuntimeError(
            f"Missing '{metric}' and fallback 'average' for '{label}' in text statistics file."
        )

    categories = [
        "L1 Cache",
        "L2 Cache",
        "Main Memory\n(16 KiB locality)",
        "Main Memory\n(Global random)",
    ]
    latencies = [
        require_metric("l1", "L1 Cache"),
        require_metric("l2", "L2 Cache"),
        require_metric("locality_16k", "16 KiB locality latency"),
        require_metric("global_random", "Global-random latency"),
    ]
    delta = require_metric("locality_delta", "Locality latency delta")
    return infer_cpu_name_from_path(path), categories, latencies, delta, None


def load_latency_data(path: Path, metric: str):
    raw_text = path.read_text(encoding="utf-8", errors="replace")
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        return parse_text_statistics(path, metric)

    config = data.get("configuration", {})
    cpu_name = str(config.get("cpu_name", "Unknown CPU"))
    version = data.get("version")
    if version is not None:
        version = str(version)

    schema_version = config.get("benchmark_schema_version")
    try:
        if schema_version == 2:
            if not data.get("results_complete", False):
                raise RuntimeError("Standard benchmark JSON is incomplete (results_complete is false).")
            l1_block = data["cache"]["l1"]["latency"]["headline_ns"]
            l2_block = data["cache"]["l2"]["latency"]["headline_ns"]
            locality = data["main_memory"]["latency"]["automatic_locality_comparison"]
            hit_block = locality["locality_16k_latency_ns"]
            miss_block = locality["global_random_latency_ns"]
            penalty_block = locality["locality_latency_delta_ns"]
        else:
            l1_block = data["cache"]["l1"]["latency"]["average_ns"]
            l2_block = data["cache"]["l2"]["latency"]["average_ns"]
            tlb = data["main_memory"]["latency"]["auto_tlb_breakdown"]
            hit_block = tlb["tlb_hit_ns"]
            miss_block = tlb["tlb_miss_ns"]
            penalty_block = tlb["page_walk_penalty_ns"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            "JSON is missing required fields for memory hierarchy plot. "
            "Expected schema-2 headline/locality fields or historical benchmark latency fields."
        ) from exc

    categories = [
        "L1 Cache",
        "L2 Cache",
        "Main Memory\n(16 KiB locality)",
        "Main Memory\n(Global random)",
    ]
    latencies = [
        pick_stat_or_values(l1_block, metric, "L1 latency"),
        pick_stat_or_values(l2_block, metric, "L2 latency"),
        pick_stat_or_values(hit_block, metric, "16 KiB locality latency"),
        pick_stat_or_values(miss_block, metric, "Global-random latency"),
    ]
    delta = pick_stat_or_values(penalty_block, metric, "Locality latency delta")
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

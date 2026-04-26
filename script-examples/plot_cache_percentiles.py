#!/usr/bin/env python3
import re
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource]

#run ./latency_test_script.sh to get final_output.txt

ALLOWED_METRICS = {
    "median": "Median",
    "p90": "P90",
    "p95": "P95",
    "p99": "P99",
    "average": "Average",
    "min": "Min",
    "max": "Max",
    "stddev": "Stddev",
}


def print_usage(script_name: str):
    allowed = ", ".join(ALLOWED_METRICS.keys())
    print(f"Usage: {script_name} [final_output_path] [--metric <name>]")
    print(f"Allowed metrics: {allowed}")
    print(f"Example: {script_name} script-examples/final_output.txt --metric p99")


def parse_args(argv):
    src = Path("final_output.txt")
    metric = "median"

    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg in ("-h", "--help"):
            print_usage(argv[0])
            sys.exit(0)
        if arg == "--metric":
            if i + 1 >= len(argv):
                raise RuntimeError("Missing value for --metric")
            metric = argv[i + 1].lower()
            i += 2
            continue
        if arg.startswith("-"):
            raise RuntimeError(f"Unknown option: {arg}")
        src = Path(arg)
        i += 1

    if metric not in ALLOWED_METRICS:
        allowed = ", ".join(ALLOWED_METRICS.keys())
        raise RuntimeError(f"Invalid --metric '{metric}'. Allowed: {allowed}")

    return src, metric


def parse_final_output(path: Path, metric: str):
    """Parse final_output.txt blocks keyed by TLB locality and cache size."""
    text = path.read_text(encoding="utf-8", errors="replace")

    pattern = re.compile(
        r"TLB\s+Locality:\s*(\d+)\s*KB\s*,\s*Cache\s+Size:\s*(\d+)\s*KB\s*.*?\n\s*(\{.*?\})",
        re.DOTALL | re.IGNORECASE,
    )

    rows = []
    for m in pattern.finditer(text):
        tlb_kb = int(m.group(1))
        cache_kb = int(m.group(2))
        stats = json.loads(m.group(3))

        if metric not in stats:
            raise RuntimeError(
                f"Missing '{metric}' in stats block for tlb={tlb_kb}KB cache={cache_kb}KB"
            )

        rows.append((tlb_kb, cache_kb, float(stats[metric])))

    if not rows:
        raise RuntimeError(
            "No TLB/cache blocks found. Expected lines like 'TLB Locality: 16 KB, Cache Size: 32 KB'."
        )

    grouped = {}
    for tlb_kb, cache_kb, median in rows:
        grouped.setdefault(tlb_kb, []).append((cache_kb, median))

    for tlb_kb in grouped:
        grouped[tlb_kb].sort(key=lambda x: x[0])

    return grouped


def main():
    src, metric = parse_args(sys.argv)
    grouped = parse_final_output(src, metric)

    plt.figure(figsize=(11, 6))

    for tlb_kb in sorted(grouped.keys()):
        cache_sizes = [p[0] for p in grouped[tlb_kb]]
        medians = [p[1] for p in grouped[tlb_kb]]
        plt.plot(cache_sizes, medians, marker="o", linewidth=2.0, label=f"TLB {tlb_kb} KB")

    all_cache_sizes = sorted({cache_kb for points in grouped.values() for cache_kb, _ in points})

    plt.xscale("log", base=2)
    plt.xlabel("Cache Size (KB)\nhttps://github.com/timoheimonen/macOS-memory-benchmark")
    plt.ylabel(f"{ALLOWED_METRICS[metric]} Latency (ns)")
    plt.title(f"{ALLOWED_METRICS[metric]} Latency Trend vs Cache Size by TLB Locality")
    plt.xticks(all_cache_sizes, [f"{size} KB" for size in all_cache_sizes], rotation=45, ha="right")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

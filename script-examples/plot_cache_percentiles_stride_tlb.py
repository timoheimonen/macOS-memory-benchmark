#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


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

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    default_csv = SCRIPT_DIR / "latency-results" / "latest" / "latency_summary.csv"
    parser = argparse.ArgumentParser(
        description="Plot cache-latency metric vs cache size from latency_summary.csv",
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default=str(default_csv),
        help="Path to latency_summary.csv",
    )
    parser.add_argument(
        "--metric",
        default="median",
        choices=sorted(ALLOWED_METRICS.keys()),
        help="Which samples_statistics metric to plot",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Filter to one stride in bytes (for example 64 or 136)",
    )
    parser.add_argument(
        "--tlb",
        type=int,
        default=None,
        help="Filter to one TLB locality in KB",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Save figure to file instead of only showing it",
    )
    return parser.parse_args()


def load_rows(path: Path):
    if not path.exists():
        raise RuntimeError(f"Input file not found: {path}")

    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append(
                    {
                        "cache_kb": int(float(row["cache_kb"])),
                        "tlb_kb": int(float(row["tlb_kb"])),
                        "stride_bytes": int(float(row["stride_bytes"])),
                        "average": float(row["average"]),
                        "median": float(row["median"]),
                        "p90": float(row["p90"]),
                        "p95": float(row["p95"]),
                        "p99": float(row["p99"]),
                        "min": float(row["min"]),
                        "max": float(row["max"]),
                        "stddev": float(row["stddev"]),
                    }
                )
            except (ValueError, KeyError):
                continue

    if not rows:
        raise RuntimeError("No valid rows in CSV.")

    return rows


def group_rows(rows, metric):
    grouped = {}
    for row in rows:
        key = (row["stride_bytes"], row["tlb_kb"])
        grouped.setdefault(key, []).append((row["cache_kb"], row[metric]))

    for key in grouped:
        grouped[key].sort(key=lambda x: x[0])

    return grouped


def main():
    args = parse_args()
    csv_path = Path(args.input_csv)
    rows = load_rows(csv_path)

    if args.stride is not None:
        rows = [r for r in rows if r["stride_bytes"] == args.stride]
    if args.tlb is not None:
        rows = [r for r in rows if r["tlb_kb"] == args.tlb]

    if not rows:
        raise RuntimeError("No rows match the selected filters.")

    grouped = group_rows(rows, args.metric)
    all_cache_sizes = sorted({r["cache_kb"] for r in rows})

    plt.figure(figsize=(12, 6))

    for (stride_bytes, tlb_kb) in sorted(grouped.keys()):
        points = grouped[(stride_bytes, tlb_kb)]
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        label = f"stride={stride_bytes}B tlb={tlb_kb}KB"
        plt.plot(x, y, marker="o", linewidth=1.8, label=label)

    plt.xscale("log", base=2)
    plt.xticks(all_cache_sizes, [f"{v} KB" for v in all_cache_sizes], rotation=45, ha="right")
    plt.xlabel("Cache Size (KB)\ngithub.com/timoheimonen/macos-memory-benchmark")
    plt.ylabel(f"{ALLOWED_METRICS[args.metric]} Latency (ns)")
    plt.title(f"{ALLOWED_METRICS[args.metric]} Cache Latency vs Cache Size/Stride/TLB")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    if args.save:
        output_path = Path(args.save)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()

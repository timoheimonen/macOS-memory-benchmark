#!/usr/bin/env python3
import re
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def parse_final_output(path: Path):
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

        if "median" not in stats:
            raise RuntimeError(
                f"Missing 'median' in stats block for tlb={tlb_kb}KB cache={cache_kb}KB"
            )

        rows.append((tlb_kb, cache_kb, float(stats["median"])))

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
    src = Path("final_output.txt")
    if len(sys.argv) > 1:
        src = Path(sys.argv[1])

    grouped = parse_final_output(src)

    plt.figure(figsize=(11, 6))

    for tlb_kb in sorted(grouped.keys()):
        cache_sizes = [p[0] for p in grouped[tlb_kb]]
        medians = [p[1] for p in grouped[tlb_kb]]
        plt.plot(cache_sizes, medians, marker="o", linewidth=2.0, label=f"TLB {tlb_kb} KB")

    all_cache_sizes = sorted({cache_kb for points in grouped.values() for cache_kb, _ in points})

    plt.xscale("log", base=2)
    plt.xlabel(f"Cache Size (KB)\ngithub.com/timoheimonen/macos-memory-benchmark")
    plt.ylabel("Median Latency (ns)")
    plt.title("Latency Trend vs Cache Size by TLB Locality")
    plt.xticks(all_cache_sizes, [f"{size} KB" for size in all_cache_sizes], rotation=45, ha="right")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

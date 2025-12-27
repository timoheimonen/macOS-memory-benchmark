#!/usr/bin/env python3
import re
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def parse_final_output(path: Path):
    """
    Parses blocks like:

    Cache Size: 32 KB
    ----------------------------------------
    {
      "median": ...,
      "p90": ...,
      "p95": ...,
      "p99": ...
    }

    Returns:
      labels, p50, p90, p95, p99
    """
    text = path.read_text(encoding="utf-8", errors="replace")

    pattern = re.compile(
        r"Cache Size:\s*(\d+)\s*KB\s*.*?\n\s*(\{.*?\})",
        re.DOTALL | re.IGNORECASE,
    )

    rows = []
    for m in pattern.finditer(text):
        size_kb = int(m.group(1))
        stats = json.loads(m.group(2))

        rows.append((
            size_kb,
            float(stats["median"]),  # P50
            float(stats["p90"]),
            float(stats["p95"]),
            float(stats["p99"]),
        ))

    if not rows:
        raise RuntimeError("No cache blocks found in final_output.txt")

    # sort by cache size
    rows.sort(key=lambda x: x[0])

    labels = [f"{r[0]} KB" for r in rows]
    p50 = [r[1] for r in rows]
    p90 = [r[2] for r in rows]
    p95 = [r[3] for r in rows]
    p99 = [r[4] for r in rows]

    return labels, p50, p90, p95, p99


def main():
    src = Path("final_output.txt")
    if len(sys.argv) > 1:
        src = Path(sys.argv[1])

    labels, p50, p90, p95, p99 = parse_final_output(src)

    plt.figure()
    plt.plot(labels, p50, marker="o", label="P50")
    plt.plot(labels, p90, marker="o", label="P90")
    plt.plot(labels, p95, marker="o", label="P95")
    plt.plot(labels, p99, marker="o", label="P99")

    plt.xlabel("Cache Size")
    plt.ylabel("Latency (ns)")
    plt.title("Cache Latency as a Function of Cache Size (P50 / P90 / P95 / P99)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
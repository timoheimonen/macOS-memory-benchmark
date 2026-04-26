#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
  parser = argparse.ArgumentParser(
      description="Plot locality-vs-latency trend from -analyze-tlb JSON output.")
  parser.add_argument(
      "-file",
      "--file",
      default="macminim4_analyte-tbl.json",
      help="Path to -analyze-tlb JSON file (default: macminim4_analyte-tbl.json)")
  parser.add_argument(
      "--save",
      default=None,
      help="Save plot to file (for example: tlb_trend.png)")
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

  raise RuntimeError(f"Input JSON file not found: {raw_path}")


def percentile(sorted_values, fraction: float) -> float:
  if not sorted_values:
    return 0.0

  if len(sorted_values) == 1:
    return sorted_values[0]

  position = (len(sorted_values) - 1) * fraction
  low = int(position)
  high = min(low + 1, len(sorted_values) - 1)
  weight = position - low
  return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def format_kb(value_kb: int) -> str:
  if value_kb >= 1024 and (value_kb % 1024) == 0:
    return f"{value_kb // 1024} MB"
  if value_kb >= 1024:
    return f"{value_kb / 1024.0:.1f} MB"
  return f"{value_kb} KB"


def load_tlb_json(path: Path):
  with path.open("r", encoding="utf-8") as handle:
    data = json.load(handle)

  config = data.get("configuration", {})
  mode = config.get("mode")
  if mode != "analyze_tlb":
    raise RuntimeError(
        f"Unsupported configuration.mode '{mode}'. Expected 'analyze_tlb'.")

  tlb = data.get("tlb_analysis")
  if not isinstance(tlb, dict):
    raise RuntimeError("Missing 'tlb_analysis' object in JSON.")

  sweep = tlb.get("sweep")
  if not isinstance(sweep, list) or len(sweep) == 0:
    raise RuntimeError("Missing or empty 'tlb_analysis.sweep' array.")

  points = []
  for entry in sweep:
    locality_kb = int(entry["locality_kb"])
    p50_latency_ns = float(entry["p50_latency_ns"])
    loop_values = entry.get("loop_latencies_ns", [])
    if isinstance(loop_values, list) and loop_values:
      loops = sorted(float(v) for v in loop_values)
      p10 = percentile(loops, 0.10)
      p90 = percentile(loops, 0.90)
    else:
      p10 = None
      p90 = None

    points.append({
        "locality_kb": locality_kb,
        "p50_latency_ns": p50_latency_ns,
        "p10_latency_ns": p10,
        "p90_latency_ns": p90,
    })

  points.sort(key=lambda item: item["locality_kb"])
  return data, config, tlb, points


def draw_vertical_marker(ax, locality_kb: int, label: str, color: str, linestyle: str):
  if locality_kb is None or locality_kb <= 0:
    return
  ax.axvline(locality_kb, color=color, linestyle=linestyle, linewidth=1.4, alpha=0.9, label=label)


def main():
  args = parse_args()
  input_path = resolve_input_path(args.file)
  data, config, tlb, points = load_tlb_json(input_path)

  x_kb = [p["locality_kb"] for p in points]
  y_p50 = [p["p50_latency_ns"] for p in points]

  p10_values = [p["p10_latency_ns"] for p in points]
  p90_values = [p["p90_latency_ns"] for p in points]
  has_spread_band = all(v is not None for v in p10_values) and all(v is not None for v in p90_values)

  fig, ax = plt.subplots(figsize=(12, 6.5))
  ax.plot(x_kb, y_p50, marker="o", linewidth=2.2, color="#0b5d8f", label="P50 latency")

  if has_spread_band:
    ax.fill_between(x_kb, p10_values, p90_values, alpha=0.20, color="#6aaed6", label="P10-P90 band")

  tlb_guard_bytes = config.get("tlb_guard_bytes")
  guard_kb = None
  if isinstance(tlb_guard_bytes, (int, float)) and tlb_guard_bytes > 0:
    guard_kb = int(round(float(tlb_guard_bytes) / 1024.0))
    draw_vertical_marker(ax, guard_kb, f"TLB guard ({format_kb(guard_kb)})", "#6c757d", ":")

  l1 = tlb.get("l1_tlb_detection", {})
  l2 = tlb.get("l2_tlb_detection", {})

  if l1.get("detected"):
    l1_kb = int(l1.get("boundary_locality_kb", 0))
    draw_vertical_marker(ax, l1_kb, f"L1 boundary ({format_kb(l1_kb)})", "#2e8b57", "--")

  if l2.get("detected"):
    l2_kb = int(l2.get("boundary_locality_kb", 0))
    draw_vertical_marker(ax, l2_kb, f"L2 boundary ({format_kb(l2_kb)})", "#d97706", "--")

  cpu_name = config.get("cpu_name", "Unknown CPU")
  stride_bytes = config.get("latency_stride_bytes", "?")
  page_size_bytes = config.get("page_size_bytes", 0)
  page_size_kb = int(round(float(page_size_bytes) / 1024.0)) if isinstance(page_size_bytes, (int, float)) else 0

  ax.set_xscale("log", base=2)
  ax.set_xticks(x_kb)
  ax.set_xticklabels([format_kb(v) for v in x_kb], rotation=45, ha="right")
  ax.set_xlabel("TLB Locality Window")
  ax.set_ylabel("Latency (ns/access)")
  ax.set_title(
      f"TLB Analysis Trend (P50) - {cpu_name}\n"
      f"Stride: {stride_bytes} B, Page Size: {page_size_kb} KB")
  ax.grid(True, which="both", alpha=0.3)

  note_lines = []
  if l1.get("detected"):
    note_lines.append(
        f"L1: {format_kb(int(l1.get('boundary_locality_kb', 0)))}"
        f", {int(l1.get('inferred_entries', 0))} entries")
  if l2.get("detected"):
    note_lines.append(
        f"L2: {format_kb(int(l2.get('boundary_locality_kb', 0)))}"
        f", {int(l2.get('inferred_entries', 0))} entries")

  page_walk = tlb.get("page_walk_penalty", {})
  if page_walk.get("available") and "penalty_ns" in page_walk:
    note_lines.append(f"Page walk 16KB->512MB: {float(page_walk['penalty_ns']):.2f} ns")
  else:
    note_lines.append("Page walk 16KB->512MB: N/A")

  if note_lines:
    ax.text(0.02,
            0.98,
            "\n".join(note_lines),
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85})

  ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.74), fontsize=9)
  fig.text(0.5,
           0.01,
           "https://github.com/timoheimonen/macOS-memory-benchmark",
           ha="center",
           fontsize=9)
  plt.tight_layout(rect=(0.01, 0.03, 0.99, 0.99))

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

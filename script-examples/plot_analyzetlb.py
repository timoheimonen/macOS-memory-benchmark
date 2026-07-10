#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
  parser = argparse.ArgumentParser(
      description="Plot locality-vs-latency trend from --analyze-tlb JSON output.")
  parser.add_argument(
      "-f",
      "--file",
      default="macminim4_analyte-tbl.json",
      help="Path to --analyze-tlb JSON file (default: macminim4_analyte-tbl.json)")
  parser.add_argument(
      "--save",
      default=None,
      help="Save plot to file (for example: tlb_trend.png)")
  parser.add_argument(
      "--no-show",
      action="store_true",
      help="Do not open an interactive plot window")
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

  schema_version = config.get("schema_version", 1)
  if not isinstance(schema_version, int) or isinstance(schema_version, bool):
    raise RuntimeError("configuration.schema_version must be an integer when present.")
  if schema_version < 1 or schema_version > 4:
    raise RuntimeError(
        f"Unsupported analyze-TLB schema version {schema_version}; supported versions are 1-4.")

  tlb = data.get("tlb_analysis")
  if not isinstance(tlb, dict):
    raise RuntimeError("Missing 'tlb_analysis' object in JSON.")

  sweep = tlb.get("sweep")
  if not isinstance(sweep, list) or len(sweep) == 0:
    raise RuntimeError("Missing or empty 'tlb_analysis.sweep' array.")

  points = []
  for entry in sweep:
    locality_kb = int(entry["locality_kb"])
    if schema_version == 4:
      spread_value = entry.get("spread_p50_latency_ns")
      loop_values = entry.get("spread_loop_latencies_ns", [])
    else:
      spread_value = entry.get("spread_p50_latency_ns", entry.get("p50_latency_ns"))
      loop_values = entry.get("spread_loop_latencies_ns", entry.get("loop_latencies_ns", []))
    if spread_value is None:
      required_field = "spread_p50_latency_ns" if schema_version == 4 else "a supported spread P50 field"
      raise RuntimeError(f"Sweep point is missing {required_field}.")
    spread_latency_ns = float(spread_value)
    packed_latency = entry.get("packed_p50_latency_ns")
    translation_delta = entry.get("translation_delta_p50_ns")
    if isinstance(loop_values, list) and loop_values:
      loops = sorted(float(v) for v in loop_values)
      p10 = percentile(loops, 0.10)
      p90 = percentile(loops, 0.90)
    else:
      p10 = None
      p90 = None

    points.append({
        "locality_kb": locality_kb,
        "spread_latency_ns": spread_latency_ns,
        "packed_latency_ns": (
            float(packed_latency) if packed_latency is not None else None),
        "p10_latency_ns": p10,
        "p90_latency_ns": p90,
        "translation_delta_ns": (
            float(translation_delta) if translation_delta is not None else None),
        "active_cache_line_footprint_bytes": entry.get(
            "active_cache_line_footprint_bytes"),
        "short_cycle_diagnostic": bool(entry.get("short_cycle_diagnostic", False)),
    })

  points.sort(key=lambda item: item["locality_kb"])
  return data, config, tlb, points


def draw_vertical_marker(ax, locality_kb: int, label: str, color: str, linestyle: str):
  if locality_kb is None or locality_kb <= 0:
    return
  ax.axvline(locality_kb, color=color, linestyle=linestyle, linewidth=1.4, alpha=0.9, label=label)


def format_entry_result(detection: dict) -> str:
  minimum = detection.get("inferred_entries_min")
  maximum = detection.get("inferred_entries_max")
  estimate = int(detection.get("inferred_entries", 0))
  if isinstance(minimum, int) and isinstance(maximum, int):
    return f"{minimum}-{maximum} entries (midpoint estimate {estimate})"
  return f"midpoint estimate {estimate} entries"


def main():
  args = parse_args()
  input_path = resolve_input_path(args.file)
  data, config, tlb, points = load_tlb_json(input_path)

  x_kb = [p["locality_kb"] for p in points]
  y_spread = [p["spread_latency_ns"] for p in points]
  y_packed = [p["packed_latency_ns"] for p in points]
  y_delta = [p["translation_delta_ns"] for p in points]

  p10_values = [p["p10_latency_ns"] for p in points]
  p90_values = [p["p90_latency_ns"] for p in points]
  has_spread_band = all(v is not None for v in p10_values) and all(v is not None for v in p90_values)
  has_paired_series = all(value is not None for value in y_delta)
  has_packed_series = all(value is not None for value in y_packed)

  fig, ax = plt.subplots(figsize=(12, 6.5))
  if has_paired_series:
    ax.plot(x_kb,
            y_delta,
            marker="s",
            linewidth=2.4,
            color="#0b5d8f",
            label="Primary paired translation delta (spread - packed)")
    ax.plot(x_kb,
            y_spread,
            marker="o",
            linewidth=1.5,
            linestyle="--",
            color="#6c757d",
            label="Cache-hot spread P50 (raw diagnostic)")
    if has_packed_series:
      ax.plot(x_kb,
              y_packed,
              marker="^",
              linewidth=1.4,
              linestyle=":",
              color="#7b5ea7",
              label="Cache-hot packed P50 (control)")
  else:
    ax.plot(x_kb,
            y_spread,
            marker="o",
            linewidth=2.2,
            color="#0b5d8f",
            label="Spread P50 latency (legacy schema)")

  if has_spread_band:
    ax.fill_between(x_kb,
                    p10_values,
                    p90_values,
                    alpha=0.15,
                    color="#9aa0a6",
                    label="Spread P10-P90 raw band")

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
  power_of_two_ticks = [value for value in x_kb if value > 0 and (value & (value - 1)) == 0]
  visible_ticks = power_of_two_ticks if len(power_of_two_ticks) >= 2 else x_kb
  ax.set_xticks(visible_ticks)
  ax.set_xticklabels([format_kb(v) for v in visible_ticks], rotation=45, ha="right")
  ax.set_xlabel("TLB Locality Window")
  ax.set_ylabel("Latency or paired effect (ns/access)")
  ax.set_title(
      f"TLB Analysis Trend (paired delta primary; cache-hot raw controls) - {cpu_name}\n"
      f"Stride: {stride_bytes} B, Page Size: {page_size_kb} KB")
  ax.grid(True, which="both", alpha=0.3)

  note_lines = []
  analysis_status = str(tlb.get("status", "legacy/unknown"))
  conclusions_valid = bool(tlb.get("conclusions_valid", True))
  note_lines.append(f"Status: {analysis_status}")
  if not conclusions_valid:
    note_lines.append("Boundary conclusions suppressed")
  if l1.get("detected"):
    note_lines.append(
        f"L1: {format_kb(int(l1.get('boundary_locality_kb', 0)))}"
        f", {format_entry_result(l1)}")
  if l2.get("detected"):
    note_lines.append(
        f"L2: {format_kb(int(l2.get('boundary_locality_kb', 0)))}"
        f", {format_entry_result(l2)}")
  rejected_candidates = sum(
      1
      for detection in (l1, l2)
      for candidate in detection.get("candidates", [])
      if not candidate.get("accepted", False))
  if rejected_candidates:
    note_lines.append(f"Rejected boundary candidates: {rejected_candidates}")
  note_lines.append("Raw spread/packed curves are cache-hot, not direct DRAM latency")

  schema_version = int(config.get("schema_version", 1))
  paired_large = tlb.get("large_locality_paired_comparison", {})
  if paired_large.get("available") and "translation_delta_p50_ns" in paired_large:
    footprint_bytes = paired_large.get("active_cache_line_footprint_bytes")
    footprint_text = ""
    if isinstance(footprint_bytes, (int, float)) and footprint_bytes >= 0:
      footprint_text = f", {format_kb(int(footprint_bytes) // 1024)} active footprint"
    note_lines.append(
        "Large-locality paired delta: "
        f"{float(paired_large['translation_delta_p50_ns']):.2f} ns"
        f"{footprint_text}")
  elif schema_version <= 3:
    raw_large = tlb.get("large_locality_latency_delta", {})
    if raw_large.get("available") and "delta_ns" in raw_large:
      note_lines.append(
          "Large-locality raw spread delta (legacy schema): "
          f"{float(raw_large['delta_ns']):.2f} ns")
    else:
      legacy_page_walk = tlb.get("page_walk_penalty", {})
      if legacy_page_walk.get("available") and "penalty_ns" in legacy_page_walk:
        note_lines.append(
            "Large-locality raw spread delta (legacy): "
            f"{float(legacy_page_walk['penalty_ns']):.2f} ns")
      else:
        note_lines.append("Large-locality paired comparison: N/A")
  else:
    note_lines.append("Large-locality paired comparison: N/A")

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

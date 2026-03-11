#!/bin/bash
# Latency sweep helper for memory_benchmark
# - Sweeps cache size, TLB locality, and latency stride
# - Writes one JSON per run
# - Builds a CSV summary for plotting

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_ROOT="${SCRIPT_DIR}/latency-results"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${RESULTS_ROOT}/${TIMESTAMP}"
JSON_DIR="${RUN_DIR}/json"
SUMMARY_CSV="${RUN_DIR}/latency_summary.csv"

DEFAULT_BENCHMARK="${SCRIPT_DIR}/../memory_benchmark"
if [ -x "${DEFAULT_BENCHMARK}" ]; then
  BENCHMARK_CMD="${BENCHMARK_CMD:-${DEFAULT_BENCHMARK}}"
else
  BENCHMARK_CMD="${BENCHMARK_CMD:-memory_benchmark}"
fi
LATENCY_SAMPLES="${LATENCY_SAMPLES:-5000}"
LOOP_COUNT="${LOOP_COUNT:-5}"

# Keep main-memory latency disabled for cache-focused sweeps.
MAIN_BUFFER_MB="${MAIN_BUFFER_MB:-0}"

# Set to "1" to add -non-cacheable.
USE_NON_CACHEABLE="${USE_NON_CACHEABLE:-0}"

# Default sweep knobs. Override by editing arrays below.
cache_sizes_kb=(32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288)
tlb_locality_kb=(0 16 256 1024 4096 16384 32768)
latency_strides_bytes=(32 64 136)

mkdir -p "${JSON_DIR}"
ln -sfn "${RUN_DIR}" "${RESULTS_ROOT}/latest"

if [ -x "${BENCHMARK_CMD}" ]; then
  :
elif ! command -v "${BENCHMARK_CMD}" >/dev/null 2>&1; then
  echo "Error: ${BENCHMARK_CMD} not found"
  echo "Tip: build the local binary (make) or export BENCHMARK_CMD=/path/to/memory_benchmark"
  exit 1
fi

help_text="$("${BENCHMARK_CMD}" -h 2>&1 || true)"
if [[ "${help_text}" != *"-latency-stride-bytes"* ]]; then
  echo "Error: selected benchmark does not support -latency-stride-bytes"
  echo "Selected command: ${BENCHMARK_CMD}"
  echo "Tip: use the newer local binary, e.g. BENCHMARK_CMD=${SCRIPT_DIR}/../memory_benchmark"
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 is required to build summary CSV"
  exit 1
fi

total_runs=$(( ${#cache_sizes_kb[@]} * ${#tlb_locality_kb[@]} * ${#latency_strides_bytes[@]} ))
current_run=0
fail_count=0

echo "Starting latency sweep"
echo "  Output directory: ${RUN_DIR}"
echo "  Cache sizes (KB): ${cache_sizes_kb[*]}"
echo "  TLB locality (KB): ${tlb_locality_kb[*]}"
echo "  Strides (bytes): ${latency_strides_bytes[*]}"
echo "  -buffersize: ${MAIN_BUFFER_MB} MB"
echo "  -latency-samples: ${LATENCY_SAMPLES}"
echo "  -count: ${LOOP_COUNT}"
echo "  -non-cacheable: ${USE_NON_CACHEABLE}"

for stride in "${latency_strides_bytes[@]}"; do
  for tlb_kb in "${tlb_locality_kb[@]}"; do
    for cache_kb in "${cache_sizes_kb[@]}"; do
      current_run=$((current_run + 1))
      progress=$(( current_run * 100 / total_runs ))

      output_file="${JSON_DIR}/latency_stride_${stride}_tlb_${tlb_kb}_cache_${cache_kb}.json"

      cmd=(
        "${BENCHMARK_CMD}"
        -only-latency
        -buffersize "${MAIN_BUFFER_MB}"
        -cache-size "${cache_kb}"
        -latency-samples "${LATENCY_SAMPLES}"
        -count "${LOOP_COUNT}"
        -latency-tlb-locality-kb "${tlb_kb}"
        -latency-stride-bytes "${stride}"
        -output "${output_file}"
      )

      if [ "${USE_NON_CACHEABLE}" = "1" ]; then
        cmd+=("-non-cacheable")
      fi

      echo ""
      echo "[${current_run}/${total_runs}] ${progress}% | stride=${stride}B tlb=${tlb_kb}KB cache=${cache_kb}KB"
      printf "Command:"
      printf " %q" "${cmd[@]}"
      printf "\n"

      if "${cmd[@]}"; then
        echo "  OK: ${output_file}"
      else
        echo "  FAIL: stride=${stride} tlb=${tlb_kb} cache=${cache_kb}"
        fail_count=$((fail_count + 1))
      fi
    done
  done
done

echo ""
echo "Building CSV summary: ${SUMMARY_CSV}"
python3 - <<'PY' "${JSON_DIR}" "${SUMMARY_CSV}"
import csv
import json
import sys
from pathlib import Path

json_dir = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])

rows = []
for path in sorted(json_dir.glob("*.json")):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        cfg = data.get("configuration", {})
        cache = data.get("cache", {}) or {}
        custom = cache.get("custom", {}) or {}
        latency = custom.get("latency", {}) or {}
        stats = latency.get("samples_statistics", {}) or {}
        diag = latency.get("chain_diagnostics", {}) or {}

        rows.append({
            "file": path.name,
            "cache_kb": cfg.get("custom_cache_size_kb", 0),
            "tlb_kb": cfg.get("latency_tlb_locality_kb", 0),
            "stride_bytes": cfg.get("latency_stride_bytes", 0),
            "average": stats.get("average", ""),
            "median": stats.get("median", ""),
            "p90": stats.get("p90", ""),
            "p95": stats.get("p95", ""),
            "p99": stats.get("p99", ""),
            "min": stats.get("min", ""),
            "max": stats.get("max", ""),
            "stddev": stats.get("stddev", ""),
            "pointer_count": diag.get("pointer_count", ""),
            "unique_pages_touched": diag.get("unique_pages_touched", ""),
            "page_size_bytes": diag.get("page_size_bytes", ""),
        })
    except Exception as exc:
        print(f"Warning: failed to parse {path.name}: {exc}")

summary_csv.parent.mkdir(parents=True, exist_ok=True)
fieldnames = [
    "file",
    "cache_kb",
    "tlb_kb",
    "stride_bytes",
    "average",
    "median",
    "p90",
    "p95",
    "p99",
    "min",
    "max",
    "stddev",
    "pointer_count",
    "unique_pages_touched",
    "page_size_bytes",
]

with summary_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {summary_csv}")
PY

echo ""
echo "Sweep done: ${current_run}/${total_runs} runs"
if [ "${fail_count}" -gt 0 ]; then
  echo "Failed runs: ${fail_count}"
fi
echo "JSON files: ${JSON_DIR}"
echo "CSV summary: ${SUMMARY_CSV}"
echo ""
echo "Plot example:"
echo "  python3 ${SCRIPT_DIR}/plot_cache_percentiles_stride_tlb.py ${SUMMARY_CSV} --metric p99 --stride 64"

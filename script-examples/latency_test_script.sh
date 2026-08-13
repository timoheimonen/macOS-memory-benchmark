#!/bin/bash
# Latency test script for memory_benchmark
# Sweeps custom cache sizes and extracts latency percentiles

set -u

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_DIR="${SCRIPT_DIR}/tmp"

# Create tmp directory if it doesn't exist
mkdir -p "${TMP_DIR}"

# Prefer the binary built in the repository; allow an explicit override or an
# installed memory_benchmark from PATH when the local binary is unavailable.
DEFAULT_BENCHMARK="${SCRIPT_DIR}/../memory_benchmark"
if [ -x "${DEFAULT_BENCHMARK}" ]; then
    BENCHMARK_CMD="${BENCHMARK_CMD:-${DEFAULT_BENCHMARK}}"
else
    BENCHMARK_CMD="${BENCHMARK_CMD:-memory_benchmark}"
fi
BUFFER_SIZE_MB=0
LATENCY_SAMPLES=5000
LOOP_COUNT=5
ONLY_LATENCY=true
tlb_locality_sizes_kb=(16 512 1024 2048 4096 8192 16384 32768)

# Leave empty by default for cleaner cache-hierarchy latency runs.
# Set to "--non-cacheable" to request best-effort cache-discouraging madvise()
# hints. This does not create truly uncached memory.
NON_CACHEABLE=""

# Cache sizes in KB, choose list from the following or make your own
cache_sizes=(32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288)
#cache_sizes=(32 64 128 256 512 1024 2048 3072 4096 5120 6144 7168 8192 9216 10240 11264 12288 13312 14336 15360 16384 17408 18432 19456 20480 21504 22528 23552 24576 25600 26624 27648 28672 29696 30720 31744 32768 33792 34816 35840 36864 37888 38912 39936 40960 41984 43008 44032 45056 46080 47104 48128 49152 50176 51200 52224 53248 54272 55296 56320 57344 58368 59392 60416 61440 62464 63488 65536)


echo "Starting latency tests for cache sizes: ${cache_sizes[*]} KB"
echo "TLB locality sizes: ${tlb_locality_sizes_kb[*]} KB"
echo "=========================================="

if [ -x "${BENCHMARK_CMD}" ]; then
    :
elif ! command -v "${BENCHMARK_CMD}" > /dev/null 2>&1; then
    echo "Error: ${BENCHMARK_CMD} not found"
    echo "Tip: build the local binary (make) or export BENCHMARK_CMD=/path/to/memory_benchmark"
    exit 1
fi

echo "Configuration:"
echo "  --only-latency: ${ONLY_LATENCY}"
echo "  --buffer-size: ${BUFFER_SIZE_MB} MB"
echo "  --latency-samples: ${LATENCY_SAMPLES}"
echo "  --count: ${LOOP_COUNT}"
echo "  --latency-tlb-locality-kb values: ${tlb_locality_sizes_kb[*]}"
if [ -n "${NON_CACHEABLE}" ]; then
    echo "  --non-cacheable: enabled"
else
    echo "  --non-cacheable: disabled"
fi

fail_count=0
extraction_fail_count=0
total_runs=$(( ${#tlb_locality_sizes_kb[@]} * ${#cache_sizes[@]} ))
current_run=0

for tlb_kb in "${tlb_locality_sizes_kb[@]}"; do
    echo ""
    echo "--- Running sweep for --latency-tlb-locality-kb ${tlb_kb} ---"

    for cache_size in "${cache_sizes[@]}"; do
        current_run=$((current_run + 1))
        progress_percent=$(( current_run * 100 / total_runs ))
        output_file="${TMP_DIR}/output_tlb_${tlb_kb}_cache_${cache_size}.json"

        cmd=(
            "${BENCHMARK_CMD}"
            --benchmark
            --latency-tlb-locality-kb "${tlb_kb}"
            --cache-size "${cache_size}"
            --buffer-size "${BUFFER_SIZE_MB}"
            --output "${output_file}"
            --latency-samples "${LATENCY_SAMPLES}"
            --count "${LOOP_COUNT}"
        )

        if [ "${ONLY_LATENCY}" = true ]; then
            cmd+=("--only-latency")
        fi

        if [ -n "${NON_CACHEABLE}" ]; then
            cmd+=("${NON_CACHEABLE}")
        fi

        echo ""
        echo "Progress: [${current_run}/${total_runs}] ${progress_percent}% | cache=${cache_size}KB tlb=${tlb_kb}KB"
        echo "Running test for cache size: ${cache_size} KB, TLB locality: ${tlb_kb} KB"
        echo "Output file: ${output_file}"
        printf "Command:"
        printf " %q" "${cmd[@]}"
        printf "\n"

        if "${cmd[@]}"; then
            echo "✓ Successfully completed test for ${cache_size} KB cache size at ${tlb_kb} KB locality"
        else
            echo "✗ Failed test for ${cache_size} KB cache size at ${tlb_kb} KB locality"
            fail_count=$((fail_count + 1))
        fi
    done
done

echo ""
echo "=========================================="
echo "Latency benchmark phase completed."
echo "Completed ${current_run}/${total_runs} runs"
if [ "${fail_count}" -gt 0 ]; then
    echo "Warning: ${fail_count} test(s) failed"
fi
echo "Output files created:"
for tlb_kb in "${tlb_locality_sizes_kb[@]}"; do
    for cache_size in "${cache_sizes[@]}"; do
        output_file="${TMP_DIR}/output_tlb_${tlb_kb}_cache_${cache_size}.json"
        if [ -f "${output_file}" ]; then
            echo "  - ${output_file}"
        fi
    done
done

echo ""
echo "=========================================="
echo "Extracting separate latency-sample statistics from output files..."
echo ""

# Output file for aggregated statistics (in script directory)
final_output="${SCRIPT_DIR}/final_output.txt"

# Clear/create the final output file
> "${final_output}"

# Function to extract schema-2 pooled sample statistics (with historical fallback)
extract_with_jq() {
    local json_file=$1
    local cache_size=$2
    local tlb_kb=$3
    local extracted_file="${TMP_DIR}/extracted_tlb_${tlb_kb}_cache_${cache_size}.json"
    if ! jq 'if .mode == "gpu_bandwidth" and .schema_version == 1 then
               error("GPU bandwidth schema 1 is not supported by this standard CPU latency extractor")
             elif .configuration.benchmark_schema_version == 2 then
               if .results_complete == true then
                 .cache.custom.latency.headline_ns.pooled_sample_distribution.statistics
               else
                 error("incomplete benchmark result")
               end
             else
               .cache.custom.latency.samples_ns.statistics
             end
             | if type == "object"
                  and .average != null and .median != null
                  and .p90 != null and .p95 != null and .p99 != null
                  and .min != null and .max != null and .stddev != null
               then .
               else error("missing latency statistics")
               end' \
        "${json_file}" > "${extracted_file}"
    then
        rm -f "${extracted_file}"
        return 1
    fi
    echo "TLB Locality: ${tlb_kb} KB, Cache Size: ${cache_size} KB" >> "${final_output}"
    echo "----------------------------------------" >> "${final_output}"
    cat "${extracted_file}" >> "${final_output}"
    echo "" >> "${final_output}"
    rm -f "${extracted_file}"
}

# Function to extract schema-2 pooled sample statistics (with historical fallback)
extract_with_python() {
    local json_file=$1
    local cache_size=$2
    local tlb_kb=$3
    local extracted_file="${TMP_DIR}/extracted_tlb_${tlb_kb}_cache_${cache_size}.json"
    if ! python3 - "${json_file}" <<'PY' > "${extracted_file}"
import json
import sys
try:
    with open(sys.argv[1], 'r') as f:
        data = json.load(f)
        if data.get('mode') == 'gpu_bandwidth' and data.get('schema_version') == 1:
            raise RuntimeError(
                'GPU bandwidth schema 1 is not supported by this standard CPU latency extractor'
            )
        latency = data['cache']['custom']['latency']
        if data.get('configuration', {}).get('benchmark_schema_version') == 2:
            if not data.get('results_complete', False):
                raise RuntimeError('incomplete benchmark result')
            stats = latency['headline_ns']['pooled_sample_distribution']['statistics']
        else:
            stats = latency['samples_ns']['statistics']
        required_statistics = ('average', 'median', 'p90', 'p95', 'p99', 'min', 'max', 'stddev')
        missing_statistics = [name for name in required_statistics if stats.get(name) is None]
        if missing_statistics:
            raise RuntimeError('missing latency statistics: ' + ', '.join(missing_statistics))
        print(json.dumps(stats, indent=2))
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    sys.exit(1)
PY
    then
        rm -f "${extracted_file}"
        return 1
    fi
    echo "TLB Locality: ${tlb_kb} KB, Cache Size: ${cache_size} KB" >> "${final_output}"
    echo "----------------------------------------" >> "${final_output}"
    cat "${extracted_file}" >> "${final_output}"
    echo "" >> "${final_output}"
    rm -f "${extracted_file}"
}

# Check if jq is available, otherwise use Python
if command -v jq &> /dev/null; then
    echo "Using jq to extract latency sample statistics..."
    USE_JQ=true
elif command -v python3 &> /dev/null; then
    echo "Using Python to extract latency sample statistics..."
    USE_JQ=false
else
    echo "Error: Neither jq nor python3 is available. Cannot extract latency sample statistics."
    exit 1
fi

# Extract latency sample statistics from each output file
for tlb_kb in "${tlb_locality_sizes_kb[@]}"; do
    for cache_size in "${cache_sizes[@]}"; do
        output_file="${TMP_DIR}/output_tlb_${tlb_kb}_cache_${cache_size}.json"
        if [ -f "${output_file}" ]; then
            echo "Extracting from ${output_file}..."
            if [ "$USE_JQ" = true ]; then
                if ! extract_with_jq "${output_file}" "${cache_size}" "${tlb_kb}"; then
                    echo "Error: failed to extract ${output_file}"
                    extraction_fail_count=$((extraction_fail_count + 1))
                fi
            else
                if ! extract_with_python "${output_file}" "${cache_size}" "${tlb_kb}"; then
                    echo "Error: failed to extract ${output_file}"
                    extraction_fail_count=$((extraction_fail_count + 1))
                fi
            fi
        else
            echo "Warning: ${output_file} not found, skipping..."
            extraction_fail_count=$((extraction_fail_count + 1))
        fi
    done
done

echo "=========================================="
if [ "${extraction_fail_count}" -eq 0 ]; then
    echo "All latency sample statistics extracted to ${final_output}"
else
    echo "Warning: ${extraction_fail_count} output file(s) could not be extracted"
fi

# Clear tmp folder after final_output.txt is created
echo ""
echo "=========================================="
echo "Clearing tmp folder..."
if [ -d "${TMP_DIR}" ]; then
    rm -rf "${TMP_DIR}"/*
    echo "✓ Tmp folder cleared: ${TMP_DIR}"
else
    echo "Warning: Tmp folder does not exist: ${TMP_DIR}"
fi

if [ "${fail_count}" -gt 0 ] || [ "${extraction_fail_count}" -gt 0 ]; then
    exit 1
fi

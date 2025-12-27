#!/bin/bash
# Latency test script for memory_benchmark
# Tests cache sizes from 32 KB to 256 MB in doubling steps

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_DIR="${SCRIPT_DIR}/tmp"

# Create tmp directory if it doesn't exist
mkdir -p "${TMP_DIR}"

# Base command parameters
BENCHMARK_CMD="memory_benchmark"
BUFFER_SIZE=1
NON_CACHEABLE="-non-cacheable"
LATENCY_SAMPLES=5000

# Cache sizes in KB: 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144
# Starting from 32 KB, doubling each time up to 256 MB (262144 KB)
cache_sizes=(32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144)

echo "Starting latency tests for cache sizes: ${cache_sizes[*]} KB"
echo "=========================================="

for cache_size in "${cache_sizes[@]}"; do
    output_file="${TMP_DIR}/output_${cache_size}.json"
    
    echo ""
    echo "Running test for cache size: ${cache_size} KB"
    echo "Output file: ${output_file}"
    echo "Command: ${BENCHMARK_CMD} -cache-size ${cache_size} -buffersize ${BUFFER_SIZE} ${NON_CACHEABLE} -output ${output_file} -latency-samples ${LATENCY_SAMPLES}"
    
    ${BENCHMARK_CMD} \
        -cache-size ${cache_size} \
        -buffersize ${BUFFER_SIZE} \
        ${NON_CACHEABLE} \
        -output ${output_file} \
        -latency-samples ${LATENCY_SAMPLES}
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed test for ${cache_size} KB cache size"
    else
        echo "✗ Failed test for ${cache_size} KB cache size"
    fi
done

echo ""
echo "=========================================="
echo "All latency tests completed!"
echo "Output files created:"
for cache_size in "${cache_sizes[@]}"; do
    output_file="${TMP_DIR}/output_${cache_size}.json"
    if [ -f "${output_file}" ]; then
        echo "  - ${output_file}"
    fi
done

echo ""
echo "=========================================="
echo "Extracting samples_statistics from output files..."
echo ""

# Output file for aggregated statistics (in script directory)
final_output="${SCRIPT_DIR}/final_output.txt"

# Clear/create the final output file
> "${final_output}"

# Function to extract samples_statistics using jq
extract_with_jq() {
    local json_file=$1
    local cache_size=$2
    echo "Cache Size: ${cache_size} KB" >> "${final_output}"
    echo "----------------------------------------" >> "${final_output}"
    jq '.cache.custom.latency.samples_statistics' "${json_file}" >> "${final_output}"
    echo "" >> "${final_output}"
}

# Function to extract samples_statistics using Python
extract_with_python() {
    local json_file=$1
    local cache_size=$2
    echo "Cache Size: ${cache_size} KB" >> "${final_output}"
    echo "----------------------------------------" >> "${final_output}"
    python3 <<EOF >> "${final_output}"
import json
import sys
try:
    with open('${json_file}', 'r') as f:
        data = json.load(f)
        stats = data['cache']['custom']['latency']['samples_statistics']
        print(json.dumps(stats, indent=2))
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    sys.exit(1)
EOF
    echo "" >> "${final_output}"
}

# Check if jq is available, otherwise use Python
if command -v jq &> /dev/null; then
    echo "Using jq to extract samples_statistics..."
    USE_JQ=true
elif command -v python3 &> /dev/null; then
    echo "Using Python to extract samples_statistics..."
    USE_JQ=false
else
    echo "Error: Neither jq nor python3 is available. Cannot extract samples_statistics."
    exit 1
fi

# Extract samples_statistics from each output file
for cache_size in "${cache_sizes[@]}"; do
    output_file="${TMP_DIR}/output_${cache_size}.json"
    if [ -f "${output_file}" ]; then
        echo "Extracting from ${output_file}..."
        if [ "$USE_JQ" = true ]; then
            extract_with_jq "${output_file}" "${cache_size}"
        else
            extract_with_python "${output_file}" "${cache_size}"
        fi
    else
        echo "Warning: ${output_file} not found, skipping..."
    fi
done

echo "=========================================="
echo "All samples_statistics extracted to ${final_output}"

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


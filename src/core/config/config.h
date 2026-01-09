// Copyright 2026 Timo Heimonen <timo.heimonen@proton.me>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
//
/**
 * @file config.h
 * @brief Benchmark configuration structure and parsing functions
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2025
 *
 * This header defines the BenchmarkConfig structure and provides functions
 * to parse command-line arguments and validate configuration settings.
 *
 * This module uses various constants from constants.h including:
 * - Memory limits (MEMORY_LIMIT_FACTOR, FALLBACK_TOTAL_LIMIT_MB, MINIMUM_LIMIT_MB_PER_BUFFER)
 * - Cache size validation (MIN_CACHE_SIZE_KB, MAX_CACHE_SIZE_KB)
 * - Size conversion (BYTES_PER_KB, BYTES_PER_MB)
 * - Buffer sizing factors (L1_BUFFER_SIZE_FACTOR, L2_BUFFER_SIZE_FACTOR)
 * - Latency test parameters (LATENCY_STRIDE_BYTES, MIN_LATENCY_BUFFER_SIZE, BASE_LATENCY_ACCESSES, L1_LATENCY_ACCESSES, L2_LATENCY_ACCESSES, CUSTOM_LATENCY_ACCESSES)
 *
 * @see constants.h for detailed constant definitions and values.
 */
#ifndef CONFIG_H
#define CONFIG_H

#include <cstddef>  // size_t
#include <string>
#include <cstdlib>  // EXIT_SUCCESS, EXIT_FAILURE
#include "core/config/constants.h"

/**
 * @struct BenchmarkConfig
 * @brief Configuration structure containing all benchmark settings
 *
 * This structure holds all configuration parameters for running benchmarks,
 * including user-provided settings, calculated buffer sizes, system information,
 * and various flags.
 */
struct BenchmarkConfig {
  // User-provided settings
  unsigned long buffer_size_mb = Constants::DEFAULT_BUFFER_SIZE_MB;  ///< Buffer size in megabytes
  int iterations = Constants::DEFAULT_ITERATIONS;  ///< Number of test iterations per benchmark
  int loop_count = Constants::DEFAULT_LOOP_COUNT;  ///< Number of benchmark loops to run
  long long custom_cache_size_kb_ll = -1;  ///< User-requested custom cache size in KB (-1 = none)
  int latency_sample_count = Constants::DEFAULT_LATENCY_SAMPLE_COUNT;  ///< Number of latency samples to collect per test
  
  // Calculated sizes
  size_t buffer_size = 0;        ///< Final buffer size in bytes (calculated from buffer_size_mb)
  size_t l1_buffer_size = 0;     ///< L1 cache buffer size in bytes
  size_t l2_buffer_size = 0;     ///< L2 cache buffer size in bytes
  size_t custom_buffer_size = 0; ///< Custom cache buffer size in bytes
  
  // Access counts
  size_t lat_num_accesses = 0;    ///< Main memory latency test access count (calculated)
  size_t l1_num_accesses = 0;     ///< L1 cache latency test access count (set from constants)
  size_t l2_num_accesses = 0;     ///< L2 cache latency test access count (set from constants)
  size_t custom_num_accesses = 0; ///< Custom cache latency test access count (set from constants)
  
  // System info
  std::string cpu_name;          ///< CPU model name
  std::string macos_version;     ///< macOS version string (e.g., "14.2.1")
  int perf_cores = 0;            ///< Number of performance cores
  int eff_cores = 0;             ///< Number of efficiency cores
  int num_threads = 0;           ///< Total number of threads to use
  size_t l1_cache_size = 0;      ///< L1 cache size in bytes
  size_t l2_cache_size = 0;      ///< L2 cache size in bytes
  size_t custom_cache_size_bytes = 0;  ///< Custom cache size in bytes
  unsigned long max_total_allowed_mb = 0;  ///< Maximum total memory allowed in MB (80% of available)
  
  // Flags
  bool use_custom_cache_size = false;  ///< Whether to use custom cache size
  bool run_patterns = false;           ///< Whether to run pattern benchmarks
  bool use_non_cacheable = false;      ///< Use cache-discouraging hints (best-effort, not true non-cacheable)
  bool user_specified_threads = false; ///< Whether user explicitly set -threads parameter
  bool only_bandwidth = false;         ///< When true, run only bandwidth tests
  bool only_latency = false;           ///< When true, run only latency tests
  
  // Tracking flags for user-specified parameters
  bool user_specified_buffersize = false;      ///< Whether user explicitly set -buffersize
  bool user_specified_iterations = false;      ///< Whether user explicitly set -iterations
  bool user_specified_latency_samples = false; ///< Whether user explicitly set -latency-samples
  
  // Output file
  std::string output_file;  ///< JSON output file path (empty = no JSON output)
};

/**
 * @brief Parse command line arguments and populate config
 * @param argc Number of command-line arguments
 * @param argv Array of command-line argument strings
 * @param[out] config Reference to BenchmarkConfig structure to populate
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on error (and prints error message)
 * @see Constants::MIN_CACHE_SIZE_KB
 * @see Constants::MAX_CACHE_SIZE_KB
 * @see Constants::BYTES_PER_KB
 * @see Constants::L1_LATENCY_ACCESSES
 * @see Constants::L2_LATENCY_ACCESSES
 * @see Constants::CUSTOM_LATENCY_ACCESSES
 */
int parse_arguments(int argc, char* argv[], BenchmarkConfig& config);

/**
 * @brief Validate configuration values
 * @param[in,out] config Reference to BenchmarkConfig structure to validate
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on error
 *
 * Validates that all configuration values are within acceptable ranges and
 * performs necessary adjustments (e.g., rounding buffer sizes).
 * @see Constants::MEMORY_LIMIT_FACTOR
 * @see Constants::FALLBACK_TOTAL_LIMIT_MB
 * @see Constants::MINIMUM_LIMIT_MB_PER_BUFFER
 * @see Constants::BYTES_PER_MB
 * @see Constants::MIN_LATENCY_BUFFER_SIZE
 */
int validate_config(BenchmarkConfig& config);

/**
 * @brief Calculate cache buffer sizes based on cache sizes and constraints
 * @param[in,out] config Reference to BenchmarkConfig structure
 *
 * Calculates appropriate buffer sizes for L1, L2, and custom cache tests
 * based on detected cache sizes and configuration constraints.
 * @see Constants::L1_BUFFER_SIZE_FACTOR
 * @see Constants::L2_BUFFER_SIZE_FACTOR
 * @see Constants::LATENCY_STRIDE_BYTES
 * @see Constants::MIN_LATENCY_BUFFER_SIZE
 * @see Constants::BYTES_PER_KB
 */
void calculate_buffer_sizes(BenchmarkConfig& config);

/**
 * @brief Calculate latency test access counts based on buffer sizes
 * @param[in,out] config Reference to BenchmarkConfig structure
 *
 * Calculates the number of pointer-chasing accesses to perform for each
 * latency test based on the corresponding buffer sizes.
 * @see Constants::BASE_LATENCY_ACCESSES
 * @see Constants::DEFAULT_BUFFER_SIZE_MB
 */
void calculate_access_counts(BenchmarkConfig& config);

#endif // CONFIG_H


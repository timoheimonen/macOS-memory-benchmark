// Copyright 2025 Timo Heimonen <timo.heimonen@proton.me>
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
 * @file pattern_benchmark.h
 * @brief Pattern-based memory access benchmark functions
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2025
 *
 * This header provides functions to benchmark memory access patterns including
 * sequential forward, sequential reverse, strided, and random access patterns.
 */
#ifndef PATTERN_BENCHMARK_H
#define PATTERN_BENCHMARK_H

#include <array>
#include <cstddef>  // size_t
#include <cstdint>
#include <cstdlib>  // EXIT_SUCCESS, EXIT_FAILURE
#include <optional>
#include <string>
#include <vector>

#include "pattern_benchmark/pattern_work_plan.h"

// Forward declarations
struct BenchmarkConfig;
struct BenchmarkBuffers;
struct HighResTimer;

enum class PatternKind {
  SequentialForward = 0,
  SequentialReverse,
  Strided64,
  Strided4096,
  Strided16384,
  Strided2MiB,
  Random,
  Count,
};

enum class PatternOperation {
  Read = 0,
  Write,
  Copy,
  Count,
};

struct PatternMeasurement {
  PatternMeasurementStatus status = PatternMeasurementStatus::Invalid;
  std::string status_reason;
  std::optional<double> bandwidth_gb_s;
  double elapsed_seconds = 0.0;
  double pilot_elapsed_seconds = 0.0;
  size_t access_size_bytes = 0;
  size_t stride_bytes = 0;
  int requested_threads = 0;
  int effective_threads = 0;
  size_t accesses_per_pass = 0;  ///< Constant count or phase-zero count
  size_t min_accesses_per_pass = 0;
  size_t max_accesses_per_pass = 0;
  size_t passes = 0;
  size_t total_accesses = 0;
  size_t total_payload_bytes = 0;
  size_t distinct_address_count = 0;
  size_t logical_working_set_bytes = 0;
  size_t completed_phase_cycles = 0;
  size_t phase_period_passes = 0;
  uint64_t seed = 0;
  bool has_seed = false;
  bool automatic_calibration = false;
  size_t native_page_size_bytes = 0;
  bool stride_equals_native_page_size = false;
  bool large_page_backing_verified = false;
  size_t benchmark_loop_index = 0;
  size_t pattern_order_index = 0;
};

constexpr size_t pattern_measurement_index(PatternKind kind,
                                           PatternOperation operation) {
  return static_cast<size_t>(kind) * static_cast<size_t>(PatternOperation::Count) +
         static_cast<size_t>(operation);
}

/**
 * @struct PatternResults
 * @brief Structure to hold pattern benchmark results
 *
 * Stores bandwidth measurements (in GB/s) for various memory access patterns.
 * All bandwidth values are in gigabytes per second.
 */
struct PatternResults {
  // Sequential Forward (baseline)
  double forward_read_bw = 0.0;   ///< Sequential forward read bandwidth (GB/s)
  double forward_write_bw = 0.0;  ///< Sequential forward write bandwidth (GB/s)
  double forward_copy_bw = 0.0;   ///< Sequential forward copy bandwidth (GB/s)
  
  // Sequential Reverse
  double reverse_read_bw = 0.0;   ///< Sequential reverse read bandwidth (GB/s)
  double reverse_write_bw = 0.0;  ///< Sequential reverse write bandwidth (GB/s)
  double reverse_copy_bw = 0.0;   ///< Sequential reverse copy bandwidth (GB/s)
  
  // Strided (Cache Line - 64B)
  double strided_64_read_bw = 0.0;   ///< Strided (64B stride) read bandwidth (GB/s)
  double strided_64_write_bw = 0.0; ///< Strided (64B stride) write bandwidth (GB/s)
  double strided_64_copy_bw = 0.0;  ///< Strided (64B stride) copy bandwidth (GB/s)
  
  // Strided (Page - 4096B)
  double strided_4096_read_bw = 0.0;   ///< Strided (4096B stride) read bandwidth (GB/s)
  double strided_4096_write_bw = 0.0;  ///< Strided (4096B stride) write bandwidth (GB/s)
  double strided_4096_copy_bw = 0.0;   ///< Strided (4096B stride) copy bandwidth (GB/s)

  // Strided (Page - 16384B)
  double strided_16384_read_bw = 0.0;   ///< Strided (16384B stride) read bandwidth (GB/s)
  double strided_16384_write_bw = 0.0;  ///< Strided (16384B stride) write bandwidth (GB/s)
  double strided_16384_copy_bw = 0.0;   ///< Strided (16384B stride) copy bandwidth (GB/s)

  // Strided (Superpage - 2MB)
  double strided_2mb_read_bw = 0.0;   ///< Strided (2MB stride) read bandwidth (GB/s)
  double strided_2mb_write_bw = 0.0;  ///< Strided (2MB stride) write bandwidth (GB/s)
  double strided_2mb_copy_bw = 0.0;   ///< Strided (2MB stride) copy bandwidth (GB/s)
  
  // Random Uniform
  double random_read_bw = 0.0;   ///< Random access read bandwidth (GB/s)
  double random_write_bw = 0.0; ///< Random access write bandwidth (GB/s)
  double random_copy_bw = 0.0;  ///< Random access copy bandwidth (GB/s)

  std::array<PatternMeasurement,
             static_cast<size_t>(PatternKind::Count) *
                 static_cast<size_t>(PatternOperation::Count)>
      measurements;
};

/**
 * @struct PatternStatistics
 * @brief Structure containing aggregated statistics across all pattern benchmark loops
 *
 * Collects results from multiple pattern benchmark loops to enable statistical analysis,
 * including mean, min, max, and percentile calculations.
 */
struct PatternStatistics {
  std::vector<PatternResults> loop_results;  ///< Status-bearing per-loop results and metadata
  // Vectors storing results from each loop
  std::vector<double> all_forward_read_bw;        ///< Forward read bandwidth from each loop (GB/s)
  std::vector<double> all_forward_write_bw;       ///< Forward write bandwidth from each loop (GB/s)
  std::vector<double> all_forward_copy_bw;        ///< Forward copy bandwidth from each loop (GB/s)
  std::vector<double> all_reverse_read_bw;         ///< Reverse read bandwidth from each loop (GB/s)
  std::vector<double> all_reverse_write_bw;       ///< Reverse write bandwidth from each loop (GB/s)
  std::vector<double> all_reverse_copy_bw;        ///< Reverse copy bandwidth from each loop (GB/s)
  std::vector<double> all_strided_64_read_bw;     ///< Strided 64B read bandwidth from each loop (GB/s)
  std::vector<double> all_strided_64_write_bw;    ///< Strided 64B write bandwidth from each loop (GB/s)
  std::vector<double> all_strided_64_copy_bw;     ///< Strided 64B copy bandwidth from each loop (GB/s)
  std::vector<double> all_strided_4096_read_bw;   ///< Strided 4096B read bandwidth from each loop (GB/s)
  std::vector<double> all_strided_4096_write_bw;  ///< Strided 4096B write bandwidth from each loop (GB/s)
  std::vector<double> all_strided_4096_copy_bw;   ///< Strided 4096B copy bandwidth from each loop (GB/s)
  std::vector<double> all_strided_16384_read_bw;   ///< Strided 16384B read bandwidth from each loop (GB/s)
  std::vector<double> all_strided_16384_write_bw;  ///< Strided 16384B write bandwidth from each loop (GB/s)
  std::vector<double> all_strided_16384_copy_bw;   ///< Strided 16384B copy bandwidth from each loop (GB/s)
  std::vector<double> all_strided_2mb_read_bw;   ///< Strided 2MB read bandwidth from each loop (GB/s)
  std::vector<double> all_strided_2mb_write_bw;  ///< Strided 2MB write bandwidth from each loop (GB/s)
  std::vector<double> all_strided_2mb_copy_bw;   ///< Strided 2MB copy bandwidth from each loop (GB/s)
  std::vector<double> all_random_read_bw;         ///< Random read bandwidth from each loop (GB/s)
  std::vector<double> all_random_write_bw;        ///< Random write bandwidth from each loop (GB/s)
  std::vector<double> all_random_copy_bw;         ///< Random copy bandwidth from each loop (GB/s)
};

struct PatternStatisticsData {
  double average = 0.0;
  double min = 0.0;
  double max = 0.0;
  double median = 0.0;
  double p90 = 0.0;
  double p95 = 0.0;
  double p99 = 0.0;
  double stddev = 0.0;
  double coefficient_of_variation_pct = 0.0;
};

PatternMeasurement& get_pattern_measurement(PatternResults& results, PatternKind kind,
                                            PatternOperation operation);
const PatternMeasurement& get_pattern_measurement(const PatternResults& results,
                                                  PatternKind kind,
                                                  PatternOperation operation);
void set_pattern_measurement(PatternResults& results, PatternKind kind,
                             PatternOperation operation,
                             PatternMeasurement measurement);
PatternStatisticsData calculate_pattern_statistics(const std::vector<double>& values);
std::array<PatternKind, static_cast<size_t>(PatternKind::Count)>
build_pattern_execution_order(size_t loop_index);

/**
 * @brief Run pattern benchmarks for various memory access patterns
 * @param buffers Reference to benchmark buffers structure
 * @param config Reference to benchmark configuration
 * @param results Reference to PatternResults structure to store results
 * @param loop_index Zero-based outer-loop index used to balance pattern order
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on error
 *
 * Executes benchmarks for sequential forward, sequential reverse, strided (64B and 4096B),
 * and random access patterns. Results are stored in the PatternResults structure.
 */
int run_pattern_benchmarks(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
                           PatternResults& results, size_t loop_index = 0);

/**
 * @brief Run all pattern benchmark loops and collect statistics
 * @param buffers Reference to benchmark buffers structure
 * @param config Reference to benchmark configuration
 * @param[out] stats Reference to PatternStatistics structure to store aggregated results
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on error
 *
 * Executes multiple pattern benchmark loops as specified by config.loop_count and
 * aggregates all results into the PatternStatistics structure for statistical analysis.
 */
int run_all_pattern_benchmarks(const BenchmarkBuffers& buffers, const BenchmarkConfig& config, PatternStatistics& stats);

/**
 * @brief Print pattern benchmark results to console
 * @param results Reference to PatternResults structure containing results to print
 */
void print_pattern_results(const PatternResults& results);

/**
 * @brief Print pattern benchmark statistics to console
 * @param loop_count Number of loops executed
 * @param stats Reference to PatternStatistics structure containing aggregated results
 *
 * Calculates and displays summary statistics (average, min, max, percentiles) across
 * all pattern benchmark loops. Only prints if loop_count > 1.
 */
void print_pattern_statistics(int loop_count, const PatternStatistics& stats);

/**
 * @brief Extract a PatternResults snapshot at a specific loop index from PatternStatistics
 * @param stats Reference to PatternStatistics structure containing aggregated results
 * @param index Loop index to extract from (must be < vector sizes)
 * @return PatternResults populated from the given index, or default PatternResults if vectors are empty
 *
 * Convenience function that extracts a single loop's results from the multi-loop statistics
 * vectors. Returns a default-constructed PatternResults (all zeros) when the statistics are empty.
 */
PatternResults extract_pattern_results_at(const PatternStatistics& stats, size_t index);

/** @brief Build median headline results while preserving representative metadata. */
PatternResults extract_pattern_median_results(const PatternStatistics& stats);

#endif // PATTERN_BENCHMARK_H

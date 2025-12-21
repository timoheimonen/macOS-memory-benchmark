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
 * @file benchmark_runner.h
 * @brief Benchmark runner and results structures
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2025
 *
 * This header defines structures for storing benchmark results and provides
 * functions to run all benchmark loops and collect statistics.
 */
#ifndef BENCHMARK_RUNNER_H
#define BENCHMARK_RUNNER_H

#include <vector>
#include <cstddef>  // size_t
#include <cstdlib>  // EXIT_SUCCESS, EXIT_FAILURE

// Forward declarations to avoid including headers
struct BenchmarkConfig;
struct BenchmarkBuffers;

/**
 * @struct BenchmarkResults
 * @brief Structure containing results from a single benchmark loop
 *
 * Stores all performance measurements from one complete benchmark execution,
 * including bandwidth (GB/s) and latency (nanoseconds) for main memory and caches.
 */
struct BenchmarkResults {
  // Main memory bandwidth results
  double read_bw_gb_s = 0.0;      ///< Main memory read bandwidth (GB/s)
  double write_bw_gb_s = 0.0;     ///< Main memory write bandwidth (GB/s)
  double copy_bw_gb_s = 0.0;      ///< Main memory copy bandwidth (GB/s)
  double total_read_time = 0.0;   ///< Total main memory read time (seconds)
  double total_write_time = 0.0;  ///< Total main memory write time (seconds)
  double total_copy_time = 0.0;   ///< Total main memory copy time (seconds)
  
  // Main memory latency results
  double average_latency_ns = 0.0;  ///< Average main memory latency (nanoseconds)
  double total_lat_time_ns = 0.0;   ///< Total main memory latency time (nanoseconds)
  std::vector<double> latency_samples;  ///< Per-sample latencies for main memory (nanoseconds)
  
  // Cache latency results
  double l1_latency_ns = 0.0;  ///< L1 cache latency (nanoseconds)
  double l2_latency_ns = 0.0;  ///< L2 cache latency (nanoseconds)
  double custom_latency_ns = 0.0;  ///< Custom cache latency (nanoseconds)
  std::vector<double> l1_latency_samples;  ///< Per-sample L1 cache latencies (nanoseconds)
  std::vector<double> l2_latency_samples;  ///< Per-sample L2 cache latencies (nanoseconds)
  std::vector<double> custom_latency_samples;  ///< Per-sample custom cache latencies (nanoseconds)
  
  // Cache bandwidth results
  double l1_read_bw_gb_s = 0.0;   ///< L1 cache read bandwidth (GB/s)
  double l1_write_bw_gb_s = 0.0;  ///< L1 cache write bandwidth (GB/s)
  double l1_copy_bw_gb_s = 0.0;   ///< L1 cache copy bandwidth (GB/s)
  double l2_read_bw_gb_s = 0.0;   ///< L2 cache read bandwidth (GB/s)
  double l2_write_bw_gb_s = 0.0;  ///< L2 cache write bandwidth (GB/s)
  double l2_copy_bw_gb_s = 0.0;   ///< L2 cache copy bandwidth (GB/s)
  double custom_read_bw_gb_s = 0.0;   ///< Custom cache read bandwidth (GB/s)
  double custom_write_bw_gb_s = 0.0;  ///< Custom cache write bandwidth (GB/s)
  double custom_copy_bw_gb_s = 0.0;   ///< Custom cache copy bandwidth (GB/s)
};

/**
 * @struct BenchmarkStatistics
 * @brief Structure containing aggregated statistics across all benchmark loops
 *
 * Collects results from multiple benchmark loops to enable statistical analysis,
 * including mean, min, max, and percentile calculations.
 */
struct BenchmarkStatistics {
  // Vectors storing results from each loop
  std::vector<double> all_read_bw_gb_s;        ///< Read bandwidth from each loop (GB/s)
  std::vector<double> all_write_bw_gb_s;      ///< Write bandwidth from each loop (GB/s)
  std::vector<double> all_copy_bw_gb_s;       ///< Copy bandwidth from each loop (GB/s)
  std::vector<double> all_l1_latency_ns;      ///< L1 latency from each loop (nanoseconds)
  std::vector<double> all_l2_latency_ns;      ///< L2 latency from each loop (nanoseconds)
  std::vector<double> all_average_latency_ns; ///< Main memory latency from each loop (nanoseconds)
  std::vector<double> all_l1_read_bw_gb_s;    ///< L1 read bandwidth from each loop (GB/s)
  std::vector<double> all_l1_write_bw_gb_s;   ///< L1 write bandwidth from each loop (GB/s)
  std::vector<double> all_l1_copy_bw_gb_s;   ///< L1 copy bandwidth from each loop (GB/s)
  std::vector<double> all_l2_read_bw_gb_s;   ///< L2 read bandwidth from each loop (GB/s)
  std::vector<double> all_l2_write_bw_gb_s;  ///< L2 write bandwidth from each loop (GB/s)
  std::vector<double> all_l2_copy_bw_gb_s;   ///< L2 copy bandwidth from each loop (GB/s)
  std::vector<double> all_custom_latency_ns;  ///< Custom cache latency from each loop (nanoseconds)
  std::vector<double> all_custom_read_bw_gb_s;   ///< Custom cache read bandwidth from each loop (GB/s)
  std::vector<double> all_custom_write_bw_gb_s;  ///< Custom cache write bandwidth from each loop (GB/s)
  std::vector<double> all_custom_copy_bw_gb_s;  ///< Custom cache copy bandwidth from each loop (GB/s)
  
  // Full sample distributions for percentile calculations
  std::vector<double> all_main_mem_latency_samples;  ///< Concatenated latency samples from all loops (nanoseconds)
  std::vector<double> all_l1_latency_samples;  ///< Concatenated L1 latency samples from all loops (nanoseconds)
  std::vector<double> all_l2_latency_samples;  ///< Concatenated L2 latency samples from all loops (nanoseconds)
  std::vector<double> all_custom_latency_samples;  ///< Concatenated custom cache latency samples from all loops (nanoseconds)
};

/**
 * @brief Run all benchmark loops and collect statistics
 * @param buffers Reference to benchmark buffers structure
 * @param config Reference to benchmark configuration
 * @param[out] stats Reference to BenchmarkStatistics structure to store aggregated results
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on error
 *
 * Executes multiple benchmark loops as specified by config.loop_count and
 * aggregates all results into the BenchmarkStatistics structure for statistical analysis.
 */
int run_all_benchmarks(const BenchmarkBuffers& buffers, const BenchmarkConfig& config, BenchmarkStatistics& stats);

#endif // BENCHMARK_RUNNER_H


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

#include <cstddef>  // size_t
#include <vector>
#include <cstdlib>  // EXIT_SUCCESS, EXIT_FAILURE

// Forward declarations
struct BenchmarkConfig;
struct BenchmarkBuffers;
struct HighResTimer;

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
  
  // Random Uniform
  double random_read_bw = 0.0;   ///< Random access read bandwidth (GB/s)
  double random_write_bw = 0.0; ///< Random access write bandwidth (GB/s)
  double random_copy_bw = 0.0;  ///< Random access copy bandwidth (GB/s)
};

/**
 * @brief Run pattern benchmarks for various memory access patterns
 * @param buffers Reference to benchmark buffers structure
 * @param config Reference to benchmark configuration
 * @param results Reference to PatternResults structure to store results
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on error
 *
 * Executes benchmarks for sequential forward, sequential reverse, strided (64B and 4096B),
 * and random access patterns. Results are stored in the PatternResults structure.
 */
int run_pattern_benchmarks(const BenchmarkBuffers& buffers, const BenchmarkConfig& config, PatternResults& results);

/**
 * @brief Print pattern benchmark results to console
 * @param results Reference to PatternResults structure containing results to print
 */
void print_pattern_results(const PatternResults& results);

#endif // PATTERN_BENCHMARK_H


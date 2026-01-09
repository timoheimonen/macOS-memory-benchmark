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
 * @file benchmark_results.h
 * @brief Benchmark result calculation functions
 *
 * This file provides utility functions for calculating performance metrics from
 * raw timing measurements. It handles the conversion of execution times into
 * meaningful bandwidth measurements in GB/s for different memory operations:
 * - Read bandwidth: Sequential read operations
 * - Write bandwidth: Sequential write operations
 * - Copy bandwidth: Memory-to-memory copy operations
 *
 * The calculations account for:
 * - Buffer sizes (main memory, L1, L2, custom cache sizes)
 * - Iteration counts for statistical stability
 * - Different memory operation types (read/write/copy)
 *
 * @note Bandwidth is calculated as: (buffer_size * iterations) / time_seconds / 1e9
 * @note Results are returned in GB/s (gigabytes per second)
 */

#ifndef BENCHMARK_RESULTS_H
#define BENCHMARK_RESULTS_H

#include <cstddef>  // size_t

// Forward declarations
struct BenchmarkConfig;
struct BenchmarkResults;
struct TimingResults;

/**
 * @brief Calculate bandwidth metrics for a single cache level
 *
 * Computes read, write, and copy bandwidth from timing measurements for a specific
 * buffer size and iteration count. Bandwidth is calculated as total data transferred
 * divided by time in seconds, expressed in GB/s.
 *
 * @param buffer_size Size of the buffer in bytes
 * @param iterations Number of iterations performed
 * @param read_time Total time for read operations in seconds
 * @param write_time Total time for write operations in seconds
 * @param copy_time Total time for copy operations in seconds
 * @param read_bw_gb_s Output parameter for calculated read bandwidth in GB/s
 * @param write_bw_gb_s Output parameter for calculated write bandwidth in GB/s
 * @param copy_bw_gb_s Output parameter for calculated copy bandwidth in GB/s
 */
void calculate_single_bandwidth(size_t buffer_size, int iterations,
                               double read_time, double write_time, double copy_time,
                               double& read_bw_gb_s, double& write_bw_gb_s, double& copy_bw_gb_s);

/**
 * @brief Calculate all bandwidth results from timing data
 *
 * Processes complete timing results for all cache levels (main memory, L1, L2, custom)
 * and populates the BenchmarkResults structure with calculated bandwidth metrics.
 *
 * @param config Benchmark configuration containing buffer sizes and iteration counts
 * @param timings Timing measurements for all benchmark operations
 * @param results Output parameter to store calculated bandwidth results
 */
void calculate_bandwidth_results(const BenchmarkConfig& config, const TimingResults& timings,
                                 BenchmarkResults& results);

#endif // BENCHMARK_RESULTS_H

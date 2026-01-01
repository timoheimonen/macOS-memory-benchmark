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
 * @file benchmark_executor.h
 * @brief Benchmark execution functions for memory and cache tests
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 *
 * This header provides functions to execute various memory and cache benchmarks,
 * including bandwidth and latency tests for main memory and different cache levels.
 */
#ifndef BENCHMARK_EXECUTOR_H
#define BENCHMARK_EXECUTOR_H

#include <cstddef>  // size_t
#include <vector>   // std::vector
#include <atomic>
#include <cstdint>

// Forward declarations
struct BenchmarkBuffers;
struct BenchmarkConfig;
struct BenchmarkResults;
struct HighResTimer;

/**
 * @struct TimingResults
 * @brief Structure to hold timing results during benchmark execution
 *
 * Accumulates timing measurements and checksums for all benchmark tests.
 * Times are in seconds unless otherwise noted (ns = nanoseconds).
 */
struct TimingResults {
  double total_read_time = 0.0;      ///< Total main memory read time (seconds)
  double total_write_time = 0.0;    ///< Total main memory write time (seconds)
  double total_copy_time = 0.0;     ///< Total main memory copy time (seconds)
  double total_lat_time_ns = 0.0;   ///< Total main memory latency time (nanoseconds)
  double l1_lat_time_ns = 0.0;      ///< L1 cache latency time (nanoseconds)
  double l2_lat_time_ns = 0.0;      ///< L2 cache latency time (nanoseconds)
  double custom_lat_time_ns = 0.0;  ///< Custom cache latency time (nanoseconds)
  double l1_read_time = 0.0;        ///< L1 cache read time (seconds)
  double l1_write_time = 0.0;       ///< L1 cache write time (seconds)
  double l1_copy_time = 0.0;        ///< L1 cache copy time (seconds)
  double l2_read_time = 0.0;        ///< L2 cache read time (seconds)
  double l2_write_time = 0.0;       ///< L2 cache write time (seconds)
  double l2_copy_time = 0.0;        ///< L2 cache copy time (seconds)
  double custom_read_time = 0.0;    ///< Custom cache read time (seconds)
  double custom_write_time = 0.0;   ///< Custom cache write time (seconds)
  double custom_copy_time = 0.0;    ///< Custom cache copy time (seconds)
  std::atomic<uint64_t> total_read_checksum{0};   ///< Main memory read checksum
  std::atomic<uint64_t> l1_read_checksum{0};     ///< L1 cache read checksum
  std::atomic<uint64_t> l2_read_checksum{0};      ///< L2 cache read checksum
  std::atomic<uint64_t> custom_read_checksum{0};   ///< Custom cache read checksum
};

/**
 * @brief Run main memory bandwidth tests (read, write, copy)
 * @param buffers Reference to benchmark buffers structure
 * @param config Reference to benchmark configuration
 * @param timings Reference to TimingResults structure to accumulate timing data
 * @param test_timer Reference to high-resolution timer for measurements
 */
void run_main_memory_bandwidth_tests(const BenchmarkBuffers& buffers, const BenchmarkConfig& config, 
                                     TimingResults& timings, HighResTimer& test_timer);

/**
 * @brief Helper function to run a single cache bandwidth test (read, write, copy)
 * @param src_buffer Pointer to source buffer
 * @param dst_buffer Pointer to destination buffer
 * @param buffer_size Size of buffers in bytes
 * @param cache_iterations Number of iterations for the cache test
 * @param num_threads Number of threads to use for the test
 * @param test_timer Reference to high-resolution timer
 * @param[out] read_time Output parameter for read time (seconds)
 * @param[out] write_time Output parameter for write time (seconds)
 * @param[out] copy_time Output parameter for copy time (seconds)
 * @param[out] read_checksum Output parameter for read checksum
 */
void run_single_cache_bandwidth_test(void* src_buffer, void* dst_buffer, size_t buffer_size,
                                     int cache_iterations, int num_threads, HighResTimer& test_timer,
                                     double& read_time, double& write_time, double& copy_time,
                                     std::atomic<uint64_t>& read_checksum);

/**
 * @brief Run cache bandwidth tests (L1, L2, or custom)
 * @param buffers Reference to benchmark buffers structure
 * @param config Reference to benchmark configuration
 * @param timings Reference to TimingResults structure to accumulate timing data
 * @param test_timer Reference to high-resolution timer for measurements
 */
void run_cache_bandwidth_tests(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
                               TimingResults& timings, HighResTimer& test_timer);

/**
 * @brief Helper function to run a single cache latency test
 * @param buffer Pointer to latency test buffer
 * @param buffer_size Size of buffer in bytes
 * @param num_accesses Number of pointer-chasing accesses to perform
 * @param test_timer Reference to high-resolution timer
 * @param[out] lat_time_ns Output parameter for total latency time (nanoseconds)
 * @param[out] latency_ns Output parameter for average latency per access (nanoseconds)
 * @param[out] latency_samples Optional pointer to vector to store individual latency samples
 * @param sample_count Number of samples to collect (0 = collect all)
 */
void run_single_cache_latency_test(void* buffer, size_t buffer_size, size_t num_accesses,
                                   HighResTimer& test_timer, double& lat_time_ns, double& latency_ns,
                                   std::vector<double>* latency_samples = nullptr, int sample_count = 0);

/**
 * @brief Run cache latency tests (L1, L2, or custom)
 * @param buffers Reference to benchmark buffers structure
 * @param config Reference to benchmark configuration
 * @param timings Reference to TimingResults structure to accumulate timing data
 * @param results Reference to BenchmarkResults structure to store latency results
 * @param test_timer Reference to high-resolution timer for measurements
 */
void run_cache_latency_tests(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
                             TimingResults& timings, BenchmarkResults& results, HighResTimer& test_timer);

/**
 * @brief Run main memory latency test
 * @param buffers Reference to benchmark buffers structure
 * @param config Reference to benchmark configuration
 * @param timings Reference to TimingResults structure to accumulate timing data
 * @param results Reference to BenchmarkResults structure to store latency results
 * @param test_timer Reference to high-resolution timer for measurements
 */
void run_main_memory_latency_test(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
                                  TimingResults& timings, BenchmarkResults& results, HighResTimer& test_timer);

/**
 * @brief Run a single benchmark loop and return results
 * @param buffers Reference to benchmark buffers structure
 * @param config Reference to benchmark configuration
 * @param loop Loop number (for display purposes)
 * @param test_timer Reference to high-resolution timer for measurements
 * @return BenchmarkResults structure containing all results from the loop
 *
 * Executes one complete benchmark loop, running all configured tests
 * (bandwidth and/or latency) and calculating results.
 */
BenchmarkResults run_single_benchmark_loop(const BenchmarkBuffers& buffers, const BenchmarkConfig& config, int loop, HighResTimer& test_timer);

#endif // BENCHMARK_EXECUTOR_H

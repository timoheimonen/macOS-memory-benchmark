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
 * @file benchmark_executor.cpp
 * @brief Single benchmark loop execution
 *
 * Implements the execution logic for a single benchmark loop, coordinating all test
 * types (bandwidth and latency for main memory and cache). Handles warmup operations,
 * test execution, and result calculation.
 *
 * Key features:
 * - Modular test execution (main memory, cache bandwidth, cache latency)
 * - Automatic warmup before each test to stabilize caches
 * - Conditional execution based on configuration flags
 * - Progress indication for user feedback
 * - Exception handling with re-throw for caller handling
 *
 * Test execution order:
 * 1. Main memory bandwidth (read, write, copy)
 * 2. Cache bandwidth (L1/L2 or custom)
 * 3. Cache latency tests
 * 4. Main memory latency test
 *
 * Conditional execution:
 * - only_bandwidth: Skip latency tests
 * - only_latency: Skip bandwidth tests
 * - use_custom_cache_size: Use custom cache instead of L1/L2
 */
 
#include "benchmark/benchmark_executor.h"
#include "core/memory/buffer_manager.h"  // BenchmarkBuffers
#include "core/config/config.h"           // BenchmarkConfig
#include "utils/benchmark.h"        // All benchmark functions and print functions
#include "benchmark/benchmark_runner.h" // BenchmarkResults
#include "benchmark/benchmark_results.h"   // Results calculation functions
#include "output/console/messages.h"             // Centralized messages
#include "core/config/constants.h"
#include <atomic>
#include <iostream>
#include <stdexcept>

/**
 * @brief Run main memory bandwidth tests (read, write, copy).
 *
 * Executes bandwidth tests for main memory using the configured buffer size and thread count.
 * Each test is preceded by a warmup operation to stabilize cache state.
 *
 * Test sequence:
 * 1. Warmup read → Measure read bandwidth
 * 2. Warmup write → Measure write bandwidth
 * 3. Warmup copy → Measure copy bandwidth
 *
 * @param[in]     buffers     Benchmark buffers (src, dst)
 * @param[in]     config      Configuration (buffer_size, iterations, num_threads)
 * @param[out]    timings     Timing results structure to populate
 * @param[in,out] test_timer  High-resolution timer for measurements
 *
 * @note Progress indicator is shown before each test.
 * @note Read test accumulates checksum to prevent optimization.
 *
 * @see run_cache_bandwidth_tests() for cache-specific bandwidth
 */
void run_main_memory_bandwidth_tests(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
                                     TimingResults& timings, HighResTimer& test_timer) {
  show_progress();
  std::atomic<uint64_t> warmup_read_checksum{0};
  warmup_read(buffers.src_buffer(), config.buffer_size, config.num_threads, warmup_read_checksum);
  timings.total_read_time = run_read_test(buffers.src_buffer(), config.buffer_size, config.iterations, 
                                          config.num_threads, timings.total_read_checksum, test_timer);
  
  show_progress();
  warmup_write(buffers.dst_buffer(), config.buffer_size, config.num_threads);
  timings.total_write_time = run_write_test(buffers.dst_buffer(), config.buffer_size, config.iterations, 
                                            config.num_threads, test_timer);
  
  show_progress();
  warmup_copy(buffers.dst_buffer(), buffers.src_buffer(), config.buffer_size, config.num_threads);
  timings.total_copy_time = run_copy_test(buffers.dst_buffer(), buffers.src_buffer(), config.buffer_size, 
                                          config.iterations, config.num_threads, test_timer);
}

/**
 * @brief Helper function to run a single cache bandwidth test (read, write, copy).
 *
 * Executes bandwidth tests for a specific cache level (L1, L2, or custom).
 * Uses cache-specific warmup and iteration counts.
 *
 * @param[in]     src_buffer       Source buffer for cache tests
 * @param[in]     dst_buffer       Destination buffer for cache tests
 * @param[in]     buffer_size      Size of cache buffer
 * @param[in]     cache_iterations Iteration count (typically higher than main memory)
 * @param[in]     num_threads      Thread count for parallel execution
 * @param[in,out] test_timer       High-resolution timer
 * @param[out]    read_time        Read timing result
 * @param[out]    write_time       Write timing result
 * @param[out]    copy_time        Copy timing result
 * @param[out]    read_checksum    Checksum accumulator for read validation
 *
 * @note Uses cache-specific warmup functions.
 */
void run_single_cache_bandwidth_test(void* src_buffer, void* dst_buffer, size_t buffer_size,
                                     int cache_iterations, int num_threads, HighResTimer& test_timer,
                                     double& read_time, double& write_time, double& copy_time,
                                     std::atomic<uint64_t>& read_checksum) {
  show_progress();
  std::atomic<uint64_t> warmup_read_checksum{0};
  warmup_cache_read(src_buffer, buffer_size, num_threads, warmup_read_checksum);
  read_time = run_read_test(src_buffer, buffer_size, cache_iterations, 
                           num_threads, read_checksum, test_timer);
  
  warmup_cache_write(dst_buffer, buffer_size, num_threads);
  write_time = run_write_test(dst_buffer, buffer_size, cache_iterations, 
                             num_threads, test_timer);
  
  warmup_cache_copy(dst_buffer, src_buffer, buffer_size, num_threads);
  copy_time = run_copy_test(dst_buffer, src_buffer, buffer_size, 
                           cache_iterations, num_threads, test_timer);
}

/**
 * @brief Run cache bandwidth tests (L1, L2, or custom).
 *
 * Executes bandwidth tests for configured cache levels. Uses higher iteration count
 * and optionally single-threaded execution for cache-specific measurements.
 *
 * Cache iteration multiplier:
 * - Cache tests use iterations * CACHE_ITERATIONS_MULTIPLIER
 * - Needed because cache access is faster, requiring more iterations for accuracy
 *
 * Thread count:
 * - Uses user-specified thread count if set
 * - Otherwise defaults to single-threaded for cache tests
 *
 * @param[in]     buffers     Benchmark buffers (cache-specific buffers)
 * @param[in]     config      Configuration (cache sizes, flags, iterations)
 * @param[out]    timings     Timing results structure to populate
 * @param[in,out] test_timer  High-resolution timer for measurements
 *
 * @note Conditionally executes based on use_custom_cache_size flag.
 * @note Progress indicator is shown by run_single_cache_bandwidth_test().
 *
 * @see run_main_memory_bandwidth_tests() for main memory bandwidth
 */
void run_cache_bandwidth_tests(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
                               TimingResults& timings, HighResTimer& test_timer) {
  int cache_iterations = config.iterations * Constants::CACHE_ITERATIONS_MULTIPLIER;
  // Use user-specified threads if set, otherwise default to single-threaded for cache tests
  int cache_threads = config.user_specified_threads ? config.num_threads : Constants::SINGLE_THREAD;
  
  if (config.use_custom_cache_size) {
    if (config.custom_buffer_size > 0 && buffers.custom_bw_src() != nullptr && buffers.custom_bw_dst() != nullptr) {
      run_single_cache_bandwidth_test(buffers.custom_bw_src(), buffers.custom_bw_dst(), config.custom_buffer_size,
                                      cache_iterations, cache_threads, test_timer,
                                      timings.custom_read_time, timings.custom_write_time, timings.custom_copy_time,
                                      timings.custom_read_checksum);
    }
  } else {
    if (config.l1_buffer_size > 0 && buffers.l1_bw_src() != nullptr && buffers.l1_bw_dst() != nullptr) {
      run_single_cache_bandwidth_test(buffers.l1_bw_src(), buffers.l1_bw_dst(), config.l1_buffer_size,
                                      cache_iterations, cache_threads, test_timer,
                                      timings.l1_read_time, timings.l1_write_time, timings.l1_copy_time,
                                      timings.l1_read_checksum);
    }
    
    if (config.l2_buffer_size > 0 && buffers.l2_bw_src() != nullptr && buffers.l2_bw_dst() != nullptr) {
      run_single_cache_bandwidth_test(buffers.l2_bw_src(), buffers.l2_bw_dst(), config.l2_buffer_size,
                                      cache_iterations, cache_threads, test_timer,
                                      timings.l2_read_time, timings.l2_write_time, timings.l2_copy_time,
                                      timings.l2_read_checksum);
    }
  }
}

/**
 * @brief Helper function to run a single cache latency test.
 *
 * Executes latency test for a specific cache level with optional sample collection.
 *
 * @param[in]     buffer           Cache buffer with pointer chain
 * @param[in]     buffer_size      Size of cache buffer
 * @param[in]     num_accesses     Number of pointer dereferences
 * @param[in,out] test_timer       High-resolution timer
 * @param[out]    lat_time_ns      Total latency time in nanoseconds
 * @param[out]    latency_ns       Average latency per access in nanoseconds
 * @param[out]    latency_samples  Optional sample collection vector
 * @param[in]     sample_count     Number of samples to collect
 *
 * @note Calculates average latency as total_time / num_accesses.
 * @note Handles zero num_accesses by setting latency to 0.0.
 */
void run_single_cache_latency_test(void* buffer, size_t buffer_size, size_t num_accesses,
                                   HighResTimer& test_timer, double& lat_time_ns, double& latency_ns,
                                   std::vector<double>* latency_samples, int sample_count) {
  show_progress();
  warmup_cache_latency(buffer, buffer_size);
  lat_time_ns = run_cache_latency_test(buffer, buffer_size, num_accesses, test_timer, latency_samples, sample_count);
  if (num_accesses > 0) {
    latency_ns = lat_time_ns / static_cast<double>(num_accesses);
  } else {
    latency_ns = 0.0;  // Avoid division by zero
  }
}

/**
 * @brief Run cache latency tests (L1, L2, or custom).
 *
 * Executes latency tests for configured cache levels with optional sample collection.
 *
 * @param[in]     buffers     Benchmark buffers (cache-specific latency buffers)
 * @param[in]     config      Configuration (cache sizes, num_accesses, sample_count)
 * @param[out]    timings     Timing results structure to populate
 * @param[out]    results     Results structure for latency values and samples
 * @param[in,out] test_timer  High-resolution timer for measurements
 *
 * @note Conditionally executes based on use_custom_cache_size flag.
 * @note Sample collection controlled by latency_sample_count in config.
 */
void run_cache_latency_tests(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
                             TimingResults& timings, BenchmarkResults& results, HighResTimer& test_timer) {
  if (config.use_custom_cache_size) {
    if (config.custom_buffer_size > 0 && buffers.custom_buffer() != nullptr && config.custom_num_accesses > 0) {
      run_single_cache_latency_test(buffers.custom_buffer(), config.custom_buffer_size, config.custom_num_accesses,
                                    test_timer, timings.custom_lat_time_ns, results.custom_latency_ns,
                                    &results.custom_latency_samples, config.latency_sample_count);
    }
  } else {
    if (config.l1_buffer_size > 0 && buffers.l1_buffer() != nullptr && config.l1_num_accesses > 0) {
      run_single_cache_latency_test(buffers.l1_buffer(), config.l1_buffer_size, config.l1_num_accesses,
                                    test_timer, timings.l1_lat_time_ns, results.l1_latency_ns,
                                    &results.l1_latency_samples, config.latency_sample_count);
    }
    
    if (config.l2_buffer_size > 0 && buffers.l2_buffer() != nullptr && config.l2_num_accesses > 0) {
      run_single_cache_latency_test(buffers.l2_buffer(), config.l2_buffer_size, config.l2_num_accesses,
                                    test_timer, timings.l2_lat_time_ns, results.l2_latency_ns,
                                    &results.l2_latency_samples, config.latency_sample_count);
    }
  }
}

/**
 * @brief Run main memory latency test.
 *
 * Executes latency test for main memory using pointer-chasing methodology.
 *
 * @param[in]     buffers     Benchmark buffers (lat_buffer)
 * @param[in]     config      Configuration (lat_num_accesses)
 * @param[out]    timings     Timing results structure to populate
 * @param[out]    results     Results structure (not used for samples in main memory)
 * @param[in,out] test_timer  High-resolution timer for measurements
 *
 * @note No sample collection for main memory latency (nullptr, 0 passed).
 * @note Progress indicator shown before test execution.
 */
void run_main_memory_latency_test(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
                                  TimingResults& timings, BenchmarkResults& results, HighResTimer& test_timer) {
  show_progress();
  warmup_latency(buffers.lat_buffer(), config.buffer_size);
  timings.total_lat_time_ns = run_latency_test(buffers.lat_buffer(), config.lat_num_accesses, test_timer,
                                                nullptr, 0);
}

/**
 * @brief Run a single benchmark loop and return results.
 *
 * Orchestrates execution of all configured tests for one benchmark loop:
 * 1. Executes tests based on configuration flags
 * 2. Calculates bandwidth results from timing data
 * 3. Calculates main memory latency if applicable
 * 4. Returns complete results for the loop
 *
 * Conditional execution modes:
 * - only_bandwidth: Execute only bandwidth tests (main + cache)
 * - only_latency: Execute only latency tests (cache + main)
 * - Default: Execute all tests
 *
 * Exception handling:
 * - Catches std::exception during test execution
 * - Logs error message to stderr
 * - Re-throws exception for caller to handle
 *
 * @param[in]     buffers     Pre-allocated and initialized benchmark buffers
 * @param[in]     config      Benchmark configuration (sizes, counts, flags)
 * @param[in]     loop        Loop number (for error reporting, not used internally)
 * @param[in,out] test_timer  High-resolution timer for measurements
 *
 * @return BenchmarkResults structure with all calculated results for this loop
 *
 * @throws std::exception Re-thrown from test execution failures
 *
 * @note Results include both raw timing data and calculated bandwidth values.
 * @note Main memory latency is calculated as total_time / num_accesses.
 *
 * @see run_main_memory_bandwidth_tests() for main memory bandwidth execution
 * @see run_cache_bandwidth_tests() for cache bandwidth execution
 * @see run_cache_latency_tests() for cache latency execution
 * @see run_main_memory_latency_test() for main memory latency execution
 * @see calculate_bandwidth_results() for bandwidth calculation
 */
BenchmarkResults run_single_benchmark_loop(const BenchmarkBuffers& buffers, const BenchmarkConfig& config, int loop, HighResTimer& test_timer) {
  BenchmarkResults results;
  TimingResults timings;

  try {
    // Run benchmark tests based on flags
    if (config.only_bandwidth) {
      // Run only bandwidth tests
      run_main_memory_bandwidth_tests(buffers, config, timings, test_timer);
      run_cache_bandwidth_tests(buffers, config, timings, test_timer);
    } else if (config.only_latency) {
      // Run only latency tests
      run_cache_latency_tests(buffers, config, timings, results, test_timer);
      run_main_memory_latency_test(buffers, config, timings, results, test_timer);
    } else {
      // Run all tests (default behavior)
      run_main_memory_bandwidth_tests(buffers, config, timings, test_timer);
      run_cache_bandwidth_tests(buffers, config, timings, test_timer);
      run_cache_latency_tests(buffers, config, timings, results, test_timer);
      run_main_memory_latency_test(buffers, config, timings, results, test_timer);
    }
  } catch (const std::exception &e) {
    std::cerr << Messages::error_benchmark_tests(e.what()) << std::endl;
    throw;  // Re-throw to be handled by caller
  }

  // Calculate all results from timing data
  calculate_bandwidth_results(config, timings, results);
  
  // Store timing results
  results.total_read_time = timings.total_read_time;
  results.total_write_time = timings.total_write_time;
  results.total_copy_time = timings.total_copy_time;
  results.total_lat_time_ns = timings.total_lat_time_ns;
  
  // Calculate main memory latency
  if (config.lat_num_accesses > 0) {
    results.average_latency_ns = timings.total_lat_time_ns / static_cast<double>(config.lat_num_accesses);
  }

  return results;
}

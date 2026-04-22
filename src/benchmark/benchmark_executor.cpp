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
#include "core/memory/memory_manager.h"   // allocate_buffer, allocate_buffer_non_cacheable
#include "core/memory/memory_utils.h"     // initialize_buffers, setup_latency_chain
#include "utils/benchmark.h"        // All benchmark functions and print functions
#include "benchmark/benchmark_runner.h" // BenchmarkResults
#include "benchmark/benchmark_results.h"   // Results calculation functions
#include "output/console/messages/messages_api.h"             // Centralized messages
#include "core/config/constants.h"
#include "core/signal/signal_handler.h"
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

/**
 * @brief Scales main iteration count for cache tests with saturation.
 *
 * Cache bandwidth paths intentionally run more iterations than main-memory paths
 * to stabilize measurements at much lower per-access latency. Saturation avoids
 * signed integer overflow when the caller already uses a very large iteration
 * value.
 */
int calculate_cache_iterations_saturated(int iterations) {
  if (iterations <= 0) {
    return 0;
  }

  if (iterations > std::numeric_limits<int>::max() / Constants::CACHE_ITERATIONS_MULTIPLIER) {
    return std::numeric_limits<int>::max();
  }

  return iterations * Constants::CACHE_ITERATIONS_MULTIPLIER;
}

/**
 * @brief Allocates a phase-local buffer using the active cacheability policy.
 *
 * Standard benchmark execution allocates buffers immediately before a phase and
 * releases them when the local `BenchmarkBuffers` owner goes out of scope.
 * This helper centralizes the mode switch between regular mappings and the
 * best-effort cache-discouraging allocation path.
 */
MmapPtr allocate_phase_buffer(const BenchmarkConfig& config, size_t size, const char* buffer_name) {
  if (config.use_non_cacheable) {
    return allocate_buffer_non_cacheable(size, buffer_name);
  }
  return allocate_buffer(size, buffer_name);
}

/**
 * @brief Clears latency-chain diagnostics before each benchmark loop.
 *
 * Diagnostics are re-populated when latency buffers are prepared. Resetting at
 * loop start prevents stale values from a previously executed path.
 */
void reset_latency_chain_diagnostics(BenchmarkConfig& config) {
  config.main_latency_chain_diagnostics = {};
  config.l1_latency_chain_diagnostics = {};
  config.l2_latency_chain_diagnostics = {};
  config.custom_latency_chain_diagnostics = {};
}

/**
 * @brief Prepares main-memory bandwidth buffers for one phase.
 *
 * Allocates source and destination buffers and initializes deterministic data
 * before timing starts. Copy/read/write kernels depend on both buffers being
 * present at the same time, so this phase intentionally uses a 2x main-buffer
 * footprint.
 */
int prepare_main_memory_bandwidth_buffers(const BenchmarkConfig& config, BenchmarkBuffers& buffers) {
  if (config.buffer_size == 0) {
    return EXIT_SUCCESS;
  }

  buffers.src_buffer_ptr = allocate_phase_buffer(config, config.buffer_size, "src_buffer");
  if (!buffers.src_buffer_ptr) {
    return EXIT_FAILURE;
  }

  buffers.dst_buffer_ptr = allocate_phase_buffer(config, config.buffer_size, "dst_buffer");
  if (!buffers.dst_buffer_ptr) {
    return EXIT_FAILURE;
  }

  return initialize_buffers(buffers.src_buffer(), buffers.dst_buffer(), config.buffer_size);
}

/**
 * @brief Prepares cache-bandwidth buffers for one phase.
 *
 * In custom-cache mode this prepares one src/dst pair. In auto-cache mode this
 * prepares L1 and L2 src/dst pairs. All initialization happens before measured
 * cache bandwidth kernels run.
 */
int prepare_cache_bandwidth_buffers(const BenchmarkConfig& config, BenchmarkBuffers& buffers) {
  if (config.use_custom_cache_size) {
    if (config.custom_buffer_size == 0) {
      return EXIT_SUCCESS;
    }

    buffers.custom_bw_src_ptr =
        allocate_phase_buffer(config, config.custom_buffer_size, "custom_bw_src_buffer");
    if (!buffers.custom_bw_src_ptr) {
      return EXIT_FAILURE;
    }

    buffers.custom_bw_dst_ptr =
        allocate_phase_buffer(config, config.custom_buffer_size, "custom_bw_dst_buffer");
    if (!buffers.custom_bw_dst_ptr) {
      return EXIT_FAILURE;
    }

    return initialize_buffers(buffers.custom_bw_src(), buffers.custom_bw_dst(),
                              config.custom_buffer_size);
  }

  if (config.l1_buffer_size > 0) {
    buffers.l1_bw_src_ptr = allocate_phase_buffer(config, config.l1_buffer_size, "l1_bw_src_buffer");
    if (!buffers.l1_bw_src_ptr) {
      return EXIT_FAILURE;
    }

    buffers.l1_bw_dst_ptr = allocate_phase_buffer(config, config.l1_buffer_size, "l1_bw_dst_buffer");
    if (!buffers.l1_bw_dst_ptr) {
      return EXIT_FAILURE;
    }

    if (initialize_buffers(buffers.l1_bw_src(), buffers.l1_bw_dst(), config.l1_buffer_size) !=
        EXIT_SUCCESS) {
      return EXIT_FAILURE;
    }
  }

  if (config.l2_buffer_size > 0) {
    buffers.l2_bw_src_ptr = allocate_phase_buffer(config, config.l2_buffer_size, "l2_bw_src_buffer");
    if (!buffers.l2_bw_src_ptr) {
      return EXIT_FAILURE;
    }

    buffers.l2_bw_dst_ptr = allocate_phase_buffer(config, config.l2_buffer_size, "l2_bw_dst_buffer");
    if (!buffers.l2_bw_dst_ptr) {
      return EXIT_FAILURE;
    }

    if (initialize_buffers(buffers.l2_bw_src(), buffers.l2_bw_dst(), config.l2_buffer_size) !=
        EXIT_SUCCESS) {
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}

/**
 * @brief Prepares cache-latency buffers and pointer chains for one phase.
 *
 * Builds latency chains using current stride/locality settings and optionally
 * records chain diagnostics when the user explicitly sets latency stride.
 */
int prepare_cache_latency_buffers(BenchmarkConfig& config, BenchmarkBuffers& buffers) {
  const bool collect_chain_diagnostics = config.user_specified_latency_stride;

  if (config.use_custom_cache_size) {
    if (config.custom_buffer_size == 0) {
      return EXIT_SUCCESS;
    }

    buffers.custom_buffer_ptr = allocate_phase_buffer(config, config.custom_buffer_size, "custom_buffer");
    if (!buffers.custom_buffer_ptr) {
      return EXIT_FAILURE;
    }

    return setup_latency_chain(buffers.custom_buffer(),
                               config.custom_buffer_size,
                               config.latency_stride_bytes,
                               config.latency_tlb_locality_bytes,
                               collect_chain_diagnostics ? &config.custom_latency_chain_diagnostics
                                                          : nullptr,
                               config.latency_chain_mode);
  }

  if (config.l1_buffer_size > 0) {
    buffers.l1_buffer_ptr = allocate_phase_buffer(config, config.l1_buffer_size, "l1_buffer");
    if (!buffers.l1_buffer_ptr) {
      return EXIT_FAILURE;
    }

    if (setup_latency_chain(buffers.l1_buffer(),
                            config.l1_buffer_size,
                            config.latency_stride_bytes,
                            config.latency_tlb_locality_bytes,
                            collect_chain_diagnostics ? &config.l1_latency_chain_diagnostics
                                                      : nullptr,
                            config.latency_chain_mode) != EXIT_SUCCESS) {
      return EXIT_FAILURE;
    }
  }

  if (config.l2_buffer_size > 0) {
    buffers.l2_buffer_ptr = allocate_phase_buffer(config, config.l2_buffer_size, "l2_buffer");
    if (!buffers.l2_buffer_ptr) {
      return EXIT_FAILURE;
    }

    if (setup_latency_chain(buffers.l2_buffer(),
                            config.l2_buffer_size,
                            config.latency_stride_bytes,
                            config.latency_tlb_locality_bytes,
                            collect_chain_diagnostics ? &config.l2_latency_chain_diagnostics
                                                      : nullptr,
                            config.latency_chain_mode) != EXIT_SUCCESS) {
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}

/**
 * @brief Prepares the main-memory latency buffer and pointer chain.
 *
 * This path is skipped when main latency is disabled (zero main buffer or zero
 * configured accesses). Chain setup is completed before timing starts.
 */
int prepare_main_memory_latency_buffer(BenchmarkConfig& config, BenchmarkBuffers& buffers) {
  if (config.buffer_size == 0 || config.lat_num_accesses == 0) {
    return EXIT_SUCCESS;
  }

  buffers.lat_buffer_ptr = allocate_phase_buffer(config, config.buffer_size, "lat_buffer");
  if (!buffers.lat_buffer_ptr) {
    return EXIT_FAILURE;
  }

  const bool collect_chain_diagnostics = config.user_specified_latency_stride;
  return setup_latency_chain(buffers.lat_buffer(),
                             config.buffer_size,
                             config.latency_stride_bytes,
                             config.latency_tlb_locality_bytes,
                             collect_chain_diagnostics ? &config.main_latency_chain_diagnostics : nullptr,
                             config.latency_chain_mode);
}

}  // namespace

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
                                     uint64_t (*read_kernel)(const void*, size_t),
                                     void (*write_kernel)(void*, size_t),
                                     void (*copy_kernel)(void*, const void*, size_t),
                                     double& read_time, double& write_time, double& copy_time,
                                     std::atomic<uint64_t>& read_checksum) {
  show_progress();
  std::atomic<uint64_t> warmup_read_checksum{0};
  warmup_cache_read(src_buffer, buffer_size, num_threads, warmup_read_checksum);
  read_time = run_read_test_with_kernel(src_buffer, buffer_size, cache_iterations,
                                        num_threads, read_checksum, test_timer, read_kernel);
  
  warmup_cache_write(dst_buffer, buffer_size, num_threads);
  write_time = run_write_test_with_kernel(dst_buffer, buffer_size, cache_iterations,
                                          num_threads, test_timer, write_kernel);
  
  warmup_cache_copy(dst_buffer, src_buffer, buffer_size, num_threads);
  copy_time = run_copy_test_with_kernel(dst_buffer, src_buffer, buffer_size,
                                        cache_iterations, num_threads, test_timer, copy_kernel);
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
  int cache_iterations = calculate_cache_iterations_saturated(config.iterations);
  // Use user-specified threads if set, otherwise default to single-threaded for cache tests
  int cache_threads = config.user_specified_threads ? config.num_threads : Constants::SINGLE_THREAD;
  auto cache_read_kernel = memory_read_cache_loop_asm;
  auto cache_write_kernel = memory_write_cache_loop_asm;
  auto cache_copy_kernel = memory_copy_cache_loop_asm;
  
  if (config.use_custom_cache_size) {
    if (config.custom_buffer_size > 0 && buffers.custom_bw_src() != nullptr && buffers.custom_bw_dst() != nullptr) {
      run_single_cache_bandwidth_test(buffers.custom_bw_src(), buffers.custom_bw_dst(), config.custom_buffer_size,
                                      cache_iterations, cache_threads, test_timer,
                                      cache_read_kernel, cache_write_kernel, cache_copy_kernel,
                                      timings.custom_read_time, timings.custom_write_time, timings.custom_copy_time,
                                      timings.custom_read_checksum);
    }
  } else {
    if (config.l1_buffer_size > 0 && buffers.l1_bw_src() != nullptr && buffers.l1_bw_dst() != nullptr) {
      run_single_cache_bandwidth_test(buffers.l1_bw_src(), buffers.l1_bw_dst(), config.l1_buffer_size,
                                      cache_iterations, cache_threads, test_timer,
                                      cache_read_kernel, cache_write_kernel, cache_copy_kernel,
                                      timings.l1_read_time, timings.l1_write_time, timings.l1_copy_time,
                                      timings.l1_read_checksum);
    }
    
    if (config.l2_buffer_size > 0 && buffers.l2_bw_src() != nullptr && buffers.l2_bw_dst() != nullptr) {
      run_single_cache_bandwidth_test(buffers.l2_bw_src(), buffers.l2_bw_dst(), config.l2_buffer_size,
                                      cache_iterations, cache_threads, test_timer,
                                      cache_read_kernel, cache_write_kernel, cache_copy_kernel,
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
 * @param[out]    results     Results structure for optional latency samples
 * @param[in,out] test_timer  High-resolution timer for measurements
 *
 * @note Main memory average latency is always computed from a single continuous chase.
 * @note When sample collection is enabled, samples are collected in a separate pass.
 * @note Progress indicator shown before test execution.
 */
void run_main_memory_latency_test(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
                                     TimingResults& timings, BenchmarkResults& results, HighResTimer& test_timer) {
  if (buffers.lat_buffer() == nullptr || config.buffer_size == 0 || config.lat_num_accesses == 0) {
    return;
  }

  results.has_auto_tlb_breakdown = false;
  results.tlb_hit_latency_ns = 0.0;
  results.tlb_miss_latency_ns = 0.0;
  results.page_walk_penalty_ns = 0.0;

  show_progress();
  warmup_latency(buffers.lat_buffer(), config.buffer_size);
  timings.total_lat_time_ns = run_latency_test(buffers.lat_buffer(), config.lat_num_accesses, test_timer,
                                                nullptr, 0);

  if (config.latency_sample_count > 0) {
    (void)run_latency_test(buffers.lat_buffer(), config.lat_num_accesses, test_timer,
                           &results.latency_samples, config.latency_sample_count);
  }

  if (!config.user_specified_latency_tlb_locality) {
    constexpr size_t kAutoTlbHitLocalityBytes = 16 * Constants::BYTES_PER_KB;
    bool hit_measured = false;
    bool miss_measured = false;
    double tlb_hit_latency_ns = 0.0;
    double tlb_miss_latency_ns = 0.0;

    if (setup_latency_chain(buffers.lat_buffer(),
                            config.buffer_size,
                            config.latency_stride_bytes,
                            kAutoTlbHitLocalityBytes,
                            nullptr,
                            LatencyChainMode::RandomInBoxRandomBox) == EXIT_SUCCESS) {
      show_progress();
      warmup_latency(buffers.lat_buffer(), config.buffer_size);
      const double hit_total_lat_time_ns =
          run_latency_test(buffers.lat_buffer(), config.lat_num_accesses, test_timer, nullptr, 0);
      tlb_hit_latency_ns = hit_total_lat_time_ns / static_cast<double>(config.lat_num_accesses);
      hit_measured = true;
    }

    if (setup_latency_chain(buffers.lat_buffer(),
                            config.buffer_size,
                            config.latency_stride_bytes,
                            0,
                            nullptr,
                            LatencyChainMode::GlobalRandom) == EXIT_SUCCESS) {
      show_progress();
      warmup_latency(buffers.lat_buffer(), config.buffer_size);
      const double miss_total_lat_time_ns =
          run_latency_test(buffers.lat_buffer(), config.lat_num_accesses, test_timer, nullptr, 0);
      tlb_miss_latency_ns = miss_total_lat_time_ns / static_cast<double>(config.lat_num_accesses);
      miss_measured = true;
    }

    if (hit_measured && miss_measured) {
      results.has_auto_tlb_breakdown = true;
      results.tlb_hit_latency_ns = tlb_hit_latency_ns;
      results.tlb_miss_latency_ns = tlb_miss_latency_ns;
      results.page_walk_penalty_ns = tlb_miss_latency_ns - tlb_hit_latency_ns;
    }

    (void)setup_latency_chain(buffers.lat_buffer(),
                              config.buffer_size,
                              config.latency_stride_bytes,
                              config.latency_tlb_locality_bytes,
                              nullptr,
                              config.latency_chain_mode);
  }
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
 * @param[in]     buffers     Benchmark buffers (unused in phase-local allocation mode)
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
 * @note Phase-local `BenchmarkBuffers` objects free mappings on scope exit (RAII).
 *
 * @see run_main_memory_bandwidth_tests() for main memory bandwidth execution
 * @see run_cache_bandwidth_tests() for cache bandwidth execution
 * @see run_cache_latency_tests() for cache latency execution
 * @see run_main_memory_latency_test() for main memory latency execution
 * @see calculate_bandwidth_results() for bandwidth calculation
 */
BenchmarkResults run_single_benchmark_loop(const BenchmarkBuffers& buffers, BenchmarkConfig& config, int loop, HighResTimer& test_timer) {
  BenchmarkResults results;
  TimingResults timings;

  (void)buffers;
  (void)loop;

  reset_latency_chain_diagnostics(config);

  try {
    // Run benchmark tests based on flags
    if (config.only_bandwidth) {
      // Run only bandwidth tests
      BenchmarkBuffers main_bandwidth_buffers;
      if (prepare_main_memory_bandwidth_buffers(config, main_bandwidth_buffers) != EXIT_SUCCESS) {
        throw std::runtime_error("Failed to prepare main memory bandwidth buffers");
      }
      run_main_memory_bandwidth_tests(main_bandwidth_buffers, config, timings, test_timer);
      if (signal_received()) return results;

      BenchmarkBuffers cache_bandwidth_buffers;
      if (prepare_cache_bandwidth_buffers(config, cache_bandwidth_buffers) != EXIT_SUCCESS) {
        throw std::runtime_error("Failed to prepare cache bandwidth buffers");
      }
      run_cache_bandwidth_tests(cache_bandwidth_buffers, config, timings, test_timer);
    } else if (config.only_latency) {
      // Run only latency tests. Cache and main latency buffers are prepared in separate
      // local owners; both remain alive until the branch scope ends.
      BenchmarkBuffers cache_latency_buffers;
      if (prepare_cache_latency_buffers(config, cache_latency_buffers) != EXIT_SUCCESS) {
        throw std::runtime_error("Failed to prepare cache latency buffers");
      }
      run_cache_latency_tests(cache_latency_buffers, config, timings, results, test_timer);
      if (signal_received()) return results;

      BenchmarkBuffers main_latency_buffers;
      if (prepare_main_memory_latency_buffer(config, main_latency_buffers) != EXIT_SUCCESS) {
        throw std::runtime_error("Failed to prepare main memory latency buffer");
      }
      run_main_memory_latency_test(main_latency_buffers, config, timings, results, test_timer);
    } else {
      // Run all tests (default behavior)
      BenchmarkBuffers main_bandwidth_buffers;
      if (prepare_main_memory_bandwidth_buffers(config, main_bandwidth_buffers) != EXIT_SUCCESS) {
        throw std::runtime_error("Failed to prepare main memory bandwidth buffers");
      }
      run_main_memory_bandwidth_tests(main_bandwidth_buffers, config, timings, test_timer);
      if (signal_received()) return results;

      BenchmarkBuffers cache_bandwidth_buffers;
      if (prepare_cache_bandwidth_buffers(config, cache_bandwidth_buffers) != EXIT_SUCCESS) {
        throw std::runtime_error("Failed to prepare cache bandwidth buffers");
      }
      run_cache_bandwidth_tests(cache_bandwidth_buffers, config, timings, test_timer);
      if (signal_received()) return results;

      BenchmarkBuffers cache_latency_buffers;
      if (prepare_cache_latency_buffers(config, cache_latency_buffers) != EXIT_SUCCESS) {
        throw std::runtime_error("Failed to prepare cache latency buffers");
      }
      run_cache_latency_tests(cache_latency_buffers, config, timings, results, test_timer);
      if (signal_received()) return results;

      BenchmarkBuffers main_latency_buffers;
      if (prepare_main_memory_latency_buffer(config, main_latency_buffers) != EXIT_SUCCESS) {
        throw std::runtime_error("Failed to prepare main memory latency buffer");
      }
      run_main_memory_latency_test(main_latency_buffers, config, timings, results, test_timer);
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

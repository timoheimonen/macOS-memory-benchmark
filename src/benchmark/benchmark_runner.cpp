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
 * @file benchmark_runner.cpp
 * @brief Benchmark execution coordinator
 *
 * Orchestrates the main benchmark loop, coordinating test execution, result collection,
 * and output printing. Handles multiple benchmark loops for statistical analysis and
 * exception handling for robust execution.
 *
 * Key features:
 * - Multi-loop execution for statistical reliability
 * - Exception handling with error propagation
 * - Progress indication during benchmark execution
 * - Integration with statistics collection system
 * - Real-time result printing for each loop
 *
 * Execution flow:
 * 1. Initialize statistics structure
 * 2. Create high-resolution timer
 * 3. For each loop:
 *    a. Run single benchmark loop (all tests)
 *    b. Collect results into statistics
 *    c. Print loop results
 * 4. Return success/failure status
 */

#include "benchmark/benchmark_runner.h"
#include "benchmark/benchmark_executor.h"  // run_single_benchmark_loop
#include "benchmark/benchmark_statistics_collector.h"  // initialize_statistics, collect_loop_results
#include "core/memory/buffer_manager.h"      // BenchmarkBuffers
#include "core/config/config.h"               // BenchmarkConfig
#include "core/timing/timer.h"                // HighResTimer
#include "utils/benchmark.h"            // All benchmark functions and print functions
#include "output/console/messages.h"             // Centralized messages
#include <iostream>
#include <cstdlib>  // EXIT_SUCCESS, EXIT_FAILURE

/**
 * @brief Runs all benchmarks for the specified number of loops and collects statistics.
 *
 * Orchestrates the complete benchmark execution:
 * 1. Initializes statistics collection structure
 * 2. Creates high-resolution timer for measurements
 * 3. Executes config.loop_count benchmark loops
 * 4. Collects results from each loop into statistics
 * 5. Prints results for each loop in real-time
 * 6. Handles exceptions and reports errors
 *
 * Loop execution:
 * - Each loop runs all configured tests (bandwidth, latency, cache)
 * - Results are collected for statistical analysis
 * - Progress indicator is shown during execution
 * - Results are printed immediately after each loop
 *
 * Error handling:
 * - Timer creation failure returns EXIT_FAILURE
 * - Exceptions during benchmark loops are caught and reported
 * - Loop number and error details are included in error messages
 *
 * @param[in]  buffers  Pre-allocated and initialized benchmark buffers
 * @param[in]  config   Benchmark configuration (buffer sizes, threads, loops, flags)
 * @param[out] stats    Statistics structure to populate with all loop results
 *
 * @return EXIT_SUCCESS (0) if all loops complete successfully
 * @return EXIT_FAILURE (1) if timer creation fails or any loop throws exception
 *
 * @note Statistics structure is used for final aggregate calculations.
 * @note Progress indicators are cleared before printing each loop's results.
 * @note Exception messages include loop number for debugging.
 *
 * @see run_single_benchmark_loop() for individual loop execution
 * @see initialize_statistics() for statistics setup
 * @see collect_loop_results() for result collection
 * @see print_results() for output formatting
 */
int run_all_benchmarks(const BenchmarkBuffers& buffers, const BenchmarkConfig& config, BenchmarkStatistics& stats) {
  // Initialize statistics structure
  initialize_statistics(stats, config);

  // Create timer for benchmark execution
  auto test_timer_opt = HighResTimer::create();
  if (!test_timer_opt) {
    std::cerr << Messages::error_prefix()
              << "Failed to create benchmark timer."
              << std::endl;
    return EXIT_FAILURE;
  }
  auto& test_timer = *test_timer_opt;

  // Main benchmark loop
  for (int loop = 0; loop < config.loop_count; ++loop) {
    try {
      // Run single benchmark loop
      BenchmarkResults loop_results = run_single_benchmark_loop(buffers, config, loop, test_timer);

      // Collect results into statistics
      collect_loop_results(stats, loop_results, config);

      // Print results for this loop
      std::cout << '\r' << std::flush;  // Clear progress indicator
      print_results(loop, config.buffer_size, config.buffer_size_mb, config.iterations, config.num_threads, 
                    loop_results.read_bw_gb_s, loop_results.total_read_time,
                    loop_results.write_bw_gb_s, loop_results.total_write_time, 
                    loop_results.copy_bw_gb_s, loop_results.total_copy_time,
                    loop_results.l1_latency_ns, loop_results.l2_latency_ns,
                    config.l1_buffer_size, config.l2_buffer_size,
                    loop_results.l1_read_bw_gb_s, loop_results.l1_write_bw_gb_s, loop_results.l1_copy_bw_gb_s,
                    loop_results.l2_read_bw_gb_s, loop_results.l2_write_bw_gb_s, loop_results.l2_copy_bw_gb_s,
                    loop_results.average_latency_ns, loop_results.total_lat_time_ns,
                    config.use_custom_cache_size, loop_results.custom_latency_ns, config.custom_buffer_size,
                    loop_results.custom_read_bw_gb_s, loop_results.custom_write_bw_gb_s, loop_results.custom_copy_bw_gb_s,
                    config.user_specified_threads, config.only_bandwidth, config.only_latency);
    } catch (const std::exception &e) {
      std::cerr << Messages::error_benchmark_loop(loop, e.what()) << std::endl;
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}


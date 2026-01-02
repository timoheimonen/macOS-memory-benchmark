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


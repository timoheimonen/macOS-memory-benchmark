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
#include "benchmark/benchmark_runner.h"
#include "benchmark/benchmark_executor.h"  // TimingResults and test execution functions
#include "benchmark/benchmark_results.h"   // Results calculation functions
#include "core/memory/buffer_manager.h"      // BenchmarkBuffers
#include "core/config/config.h"               // BenchmarkConfig
#include "utils/benchmark.h"            // All benchmark functions and print functions
#include "output/console/messages.h"             // Centralized messages
#include <iostream>
#include <stdexcept>
#include <sstream>

// Run a single benchmark loop and return results
static BenchmarkResults run_single_benchmark_loop(const BenchmarkBuffers& buffers, const BenchmarkConfig& config, int loop, HighResTimer& test_timer) {
  BenchmarkResults results;
  TimingResults timings;

  try {
    // Run all benchmark tests
    run_main_memory_bandwidth_tests(buffers, config, timings, test_timer);
    run_cache_bandwidth_tests(buffers, config, timings, test_timer);
    run_cache_latency_tests(buffers, config, timings, results, test_timer);
    run_main_memory_latency_test(buffers, config, timings, results, test_timer);
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

int run_all_benchmarks(const BenchmarkBuffers& buffers, const BenchmarkConfig& config, BenchmarkStatistics& stats) {
  // Initialize result vectors
  stats.all_read_bw_gb_s.clear();
  stats.all_write_bw_gb_s.clear();
  stats.all_copy_bw_gb_s.clear();
  stats.all_l1_latency_ns.clear();
  stats.all_l2_latency_ns.clear();
  stats.all_average_latency_ns.clear();
  stats.all_l1_read_bw_gb_s.clear();
  stats.all_l1_write_bw_gb_s.clear();
  stats.all_l1_copy_bw_gb_s.clear();
  stats.all_l2_read_bw_gb_s.clear();
  stats.all_l2_write_bw_gb_s.clear();
  stats.all_l2_copy_bw_gb_s.clear();
  stats.all_custom_latency_ns.clear();
  stats.all_custom_read_bw_gb_s.clear();
  stats.all_custom_write_bw_gb_s.clear();
  stats.all_custom_copy_bw_gb_s.clear();
  
  // Initialize sample vectors
  stats.all_main_mem_latency_samples.clear();
  stats.all_l1_latency_samples.clear();
  stats.all_l2_latency_samples.clear();
  stats.all_custom_latency_samples.clear();

  // Pre-allocate vector space if needed
  if (config.loop_count > 0) {
    stats.all_read_bw_gb_s.reserve(config.loop_count);
    stats.all_write_bw_gb_s.reserve(config.loop_count);
    stats.all_copy_bw_gb_s.reserve(config.loop_count);
    stats.all_average_latency_ns.reserve(config.loop_count);
    if (config.use_custom_cache_size) {
      if (config.custom_buffer_size > 0) {
        stats.all_custom_latency_ns.reserve(config.loop_count);
        stats.all_custom_read_bw_gb_s.reserve(config.loop_count);
        stats.all_custom_write_bw_gb_s.reserve(config.loop_count);
        stats.all_custom_copy_bw_gb_s.reserve(config.loop_count);
        // Pre-allocate sample vectors (latency_sample_count samples per loop)
        stats.all_custom_latency_samples.reserve(config.loop_count * config.latency_sample_count);
      }
    } else {
      stats.all_l1_latency_ns.reserve(config.loop_count);
      stats.all_l2_latency_ns.reserve(config.loop_count);
      if (config.l1_buffer_size > 0) {
        stats.all_l1_read_bw_gb_s.reserve(config.loop_count);
        stats.all_l1_write_bw_gb_s.reserve(config.loop_count);
        stats.all_l1_copy_bw_gb_s.reserve(config.loop_count);
        stats.all_l1_latency_samples.reserve(config.loop_count * config.latency_sample_count);
      }
      if (config.l2_buffer_size > 0) {
        stats.all_l2_read_bw_gb_s.reserve(config.loop_count);
        stats.all_l2_write_bw_gb_s.reserve(config.loop_count);
        stats.all_l2_copy_bw_gb_s.reserve(config.loop_count);
        stats.all_l2_latency_samples.reserve(config.loop_count * config.latency_sample_count);
      }
    }
    // Pre-allocate main memory sample vector
    stats.all_main_mem_latency_samples.reserve(config.loop_count * config.latency_sample_count);
  }

  HighResTimer test_timer;  // Timer for individual tests

  // Main benchmark loop
  for (int loop = 0; loop < config.loop_count; ++loop) {
    try {
      BenchmarkResults loop_results = run_single_benchmark_loop(buffers, config, loop, test_timer);

      // Store results for this loop
      stats.all_read_bw_gb_s.push_back(loop_results.read_bw_gb_s);
      stats.all_write_bw_gb_s.push_back(loop_results.write_bw_gb_s);
      stats.all_copy_bw_gb_s.push_back(loop_results.copy_bw_gb_s);
      if (config.use_custom_cache_size) {
        if (config.custom_buffer_size > 0) {
          stats.all_custom_latency_ns.push_back(loop_results.custom_latency_ns);
          stats.all_custom_read_bw_gb_s.push_back(loop_results.custom_read_bw_gb_s);
          stats.all_custom_write_bw_gb_s.push_back(loop_results.custom_write_bw_gb_s);
          stats.all_custom_copy_bw_gb_s.push_back(loop_results.custom_copy_bw_gb_s);
        }
      } else {
        if (config.l1_buffer_size > 0) {
          stats.all_l1_latency_ns.push_back(loop_results.l1_latency_ns);
          stats.all_l1_read_bw_gb_s.push_back(loop_results.l1_read_bw_gb_s);
          stats.all_l1_write_bw_gb_s.push_back(loop_results.l1_write_bw_gb_s);
          stats.all_l1_copy_bw_gb_s.push_back(loop_results.l1_copy_bw_gb_s);
        }
        if (config.l2_buffer_size > 0) {
          stats.all_l2_latency_ns.push_back(loop_results.l2_latency_ns);
          stats.all_l2_read_bw_gb_s.push_back(loop_results.l2_read_bw_gb_s);
          stats.all_l2_write_bw_gb_s.push_back(loop_results.l2_write_bw_gb_s);
          stats.all_l2_copy_bw_gb_s.push_back(loop_results.l2_copy_bw_gb_s);
        }
      }
      stats.all_average_latency_ns.push_back(loop_results.average_latency_ns);
      
      // Collect latency samples from this loop
      if (!loop_results.latency_samples.empty()) {
        stats.all_main_mem_latency_samples.insert(stats.all_main_mem_latency_samples.end(),
                                                   loop_results.latency_samples.begin(),
                                                   loop_results.latency_samples.end());
      }
      
      if (config.use_custom_cache_size) {
        if (config.custom_buffer_size > 0 && !loop_results.custom_latency_samples.empty()) {
          stats.all_custom_latency_samples.insert(stats.all_custom_latency_samples.end(),
                                                  loop_results.custom_latency_samples.begin(),
                                                  loop_results.custom_latency_samples.end());
        }
      } else {
        if (config.l1_buffer_size > 0 && !loop_results.l1_latency_samples.empty()) {
          stats.all_l1_latency_samples.insert(stats.all_l1_latency_samples.end(),
                                               loop_results.l1_latency_samples.begin(),
                                               loop_results.l1_latency_samples.end());
        }
        if (config.l2_buffer_size > 0 && !loop_results.l2_latency_samples.empty()) {
          stats.all_l2_latency_samples.insert(stats.all_l2_latency_samples.end(),
                                               loop_results.l2_latency_samples.begin(),
                                               loop_results.l2_latency_samples.end());
        }
      }

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
                    loop_results.custom_read_bw_gb_s, loop_results.custom_write_bw_gb_s, loop_results.custom_copy_bw_gb_s);
    } catch (const std::exception &e) {
      std::cerr << Messages::error_benchmark_loop(loop, e.what()) << std::endl;
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}


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
#include <cstdlib>  // Exit codes
#include <iomanip>  // Output formatting
#include <iostream>
#include <string>

#include "utils/benchmark.h"
#include "core/memory/memory_manager.h"
#include "core/config/config.h"
#include "core/memory/buffer_manager.h"
#include "benchmark/benchmark_runner.h"
#include "output/console/messages.h"
#include "core/config/constants.h"
#include "output/json/json_output.h"
#include "pattern_benchmark/pattern_benchmark.h"

// macOS specific memory management
#include <mach/mach.h>  // kern_return_t
#include <pthread/qos.h>

// Main program entry
int main(int argc, char *argv[]) {
  // Start total execution timer
  auto timer_opt = HighResTimer::create();
  if (!timer_opt) {
    std::cerr << Messages::error_prefix()
              << "Failed to create high-resolution timer. Exiting."
              << std::endl;
    return EXIT_FAILURE;
  }
  auto& total_execution_timer = *timer_opt;
  total_execution_timer.start();

  // --- Parse and Validate Configuration ---
  BenchmarkConfig config;
  int parse_result = parse_arguments(argc, argv, config);
  if (parse_result != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  
  // Check if help was requested (simple check: if only help flag, exit after printing)
  if (argc == 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
    return EXIT_SUCCESS;  // Help already printed by parse_arguments
  }
  
  if (validate_config(config) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  
  calculate_buffer_sizes(config);
  calculate_access_counts(config);

  // --- Print Config ---
  print_configuration(config.buffer_size, config.buffer_size_mb, config.iterations, config.loop_count,
                      config.use_non_cacheable, config.cpu_name, config.perf_cores, config.eff_cores, config.num_threads,
                      config.only_bandwidth, config.only_latency, config.run_patterns);
  print_cache_info(config.l1_cache_size, config.l2_cache_size, config.use_custom_cache_size, config.custom_cache_size_bytes);

  // --- Set QoS for the main thread (affects latency tests) ---
  kern_return_t qos_ret = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  if (qos_ret != KERN_SUCCESS) {
    // Non-critical error, just print a warning
    std::cerr << Messages::warning_prefix() << Messages::warning_qos_failed(qos_ret) << std::endl;
  }

  // --- Allocate and Initialize Buffers ---
  BenchmarkBuffers buffers;
  if (allocate_all_buffers(config, buffers) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  
  if (initialize_all_buffers(buffers, config) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }

  // --- Run Benchmarks ---
  if (config.run_patterns) {
    // Run pattern benchmarks only
    PatternStatistics pattern_stats;
    if (run_all_pattern_benchmarks(buffers, config, pattern_stats) != EXIT_SUCCESS) {
      return EXIT_FAILURE;
    }
    
    // Print results - if single loop, print detailed results; if multiple loops, print last loop's results
    if (config.loop_count == 1) {
      // For single loop, reconstruct PatternResults from statistics for display
      PatternResults single_result;
      if (!pattern_stats.all_forward_read_bw.empty()) {
        single_result.forward_read_bw = pattern_stats.all_forward_read_bw[0];
        single_result.forward_write_bw = pattern_stats.all_forward_write_bw[0];
        single_result.forward_copy_bw = pattern_stats.all_forward_copy_bw[0];
        single_result.reverse_read_bw = pattern_stats.all_reverse_read_bw[0];
        single_result.reverse_write_bw = pattern_stats.all_reverse_write_bw[0];
        single_result.reverse_copy_bw = pattern_stats.all_reverse_copy_bw[0];
        single_result.strided_64_read_bw = pattern_stats.all_strided_64_read_bw[0];
        single_result.strided_64_write_bw = pattern_stats.all_strided_64_write_bw[0];
        single_result.strided_64_copy_bw = pattern_stats.all_strided_64_copy_bw[0];
        single_result.strided_4096_read_bw = pattern_stats.all_strided_4096_read_bw[0];
        single_result.strided_4096_write_bw = pattern_stats.all_strided_4096_write_bw[0];
        single_result.strided_4096_copy_bw = pattern_stats.all_strided_4096_copy_bw[0];
        single_result.random_read_bw = pattern_stats.all_random_read_bw[0];
        single_result.random_write_bw = pattern_stats.all_random_write_bw[0];
        single_result.random_copy_bw = pattern_stats.all_random_copy_bw[0];
      }
      print_pattern_results(single_result);
    } else {
      // For multiple loops, print last loop's results and then statistics
      PatternResults last_result;
      if (!pattern_stats.all_forward_read_bw.empty()) {
        size_t last_idx = pattern_stats.all_forward_read_bw.size() - 1;
        last_result.forward_read_bw = pattern_stats.all_forward_read_bw[last_idx];
        last_result.forward_write_bw = pattern_stats.all_forward_write_bw[last_idx];
        last_result.forward_copy_bw = pattern_stats.all_forward_copy_bw[last_idx];
        last_result.reverse_read_bw = pattern_stats.all_reverse_read_bw[last_idx];
        last_result.reverse_write_bw = pattern_stats.all_reverse_write_bw[last_idx];
        last_result.reverse_copy_bw = pattern_stats.all_reverse_copy_bw[last_idx];
        last_result.strided_64_read_bw = pattern_stats.all_strided_64_read_bw[last_idx];
        last_result.strided_64_write_bw = pattern_stats.all_strided_64_write_bw[last_idx];
        last_result.strided_64_copy_bw = pattern_stats.all_strided_64_copy_bw[last_idx];
        last_result.strided_4096_read_bw = pattern_stats.all_strided_4096_read_bw[last_idx];
        last_result.strided_4096_write_bw = pattern_stats.all_strided_4096_write_bw[last_idx];
        last_result.strided_4096_copy_bw = pattern_stats.all_strided_4096_copy_bw[last_idx];
        last_result.random_read_bw = pattern_stats.all_random_read_bw[last_idx];
        last_result.random_write_bw = pattern_stats.all_random_write_bw[last_idx];
        last_result.random_copy_bw = pattern_stats.all_random_copy_bw[last_idx];
      }
      print_pattern_results(last_result);
      
      // Print summary statistics
      print_pattern_statistics(config.loop_count, pattern_stats);
    }
    
    // --- Save JSON Output if requested ---
    if (!config.output_file.empty()) {
      double total_elapsed_time_sec = total_execution_timer.stop();
      if (save_pattern_results_to_json(config, pattern_stats, total_elapsed_time_sec) != EXIT_SUCCESS) {
        return EXIT_FAILURE;
      }
    }
  } else {
    // Run standard benchmarks
    std::cout << Messages::msg_running_benchmarks() << std::endl;
    
    BenchmarkStatistics stats;
    if (run_all_benchmarks(buffers, config, stats) != EXIT_SUCCESS) {
      return EXIT_FAILURE;
    }

    // --- Print Stats ---
    // Print summary statistics if more than one loop was run
    print_statistics(config.loop_count, stats.all_read_bw_gb_s, stats.all_write_bw_gb_s, stats.all_copy_bw_gb_s,
                     stats.all_l1_latency_ns, stats.all_l2_latency_ns,
                     stats.all_l1_read_bw_gb_s, stats.all_l1_write_bw_gb_s, stats.all_l1_copy_bw_gb_s,
                     stats.all_l2_read_bw_gb_s, stats.all_l2_write_bw_gb_s, stats.all_l2_copy_bw_gb_s,
                     stats.all_average_latency_ns,
                     config.use_custom_cache_size,
                     stats.all_custom_latency_ns, stats.all_custom_read_bw_gb_s,
                     stats.all_custom_write_bw_gb_s, stats.all_custom_copy_bw_gb_s,
                     stats.all_main_mem_latency_samples,
                     stats.all_l1_latency_samples,
                     stats.all_l2_latency_samples,
                     stats.all_custom_latency_samples,
                     config.only_bandwidth,
                     config.only_latency);

    // --- Save JSON Output if requested ---
    if (!config.output_file.empty()) {
      double total_elapsed_time_sec = total_execution_timer.stop();
      if (save_results_to_json(config, stats, total_elapsed_time_sec) != EXIT_SUCCESS) {
        return EXIT_FAILURE;
      }
    }
  }

  // --- Free Memory ---
  // std::cout << "\nFreeing memory..." << std::endl;
  // Memory is freed automatically when src_buffer_ptr, dst_buffer_ptr,
  // and lat_buffer_ptr go out of scope. No manual munmap needed.

  // --- Print Total Time ---
  double total_elapsed_time_sec = total_execution_timer.stop();                                  // Stop overall timer
  std::cout << Messages::msg_done_total_time(total_elapsed_time_sec) << std::endl;  // Print duration

  return EXIT_SUCCESS;  // Indicate success
}
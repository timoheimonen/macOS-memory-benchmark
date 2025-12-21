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

#include "benchmark.h"
#include "memory_manager.h"
#include "config.h"
#include "buffer_manager.h"
#include "benchmark_runner.h"
#include "messages.h"
#include "constants.h"
#include "json_output.h"
#include "pattern_benchmark/pattern_benchmark.h"

// macOS specific memory management
#include <mach/mach.h>  // kern_return_t
#include <pthread/qos.h>

// Main program entry
int main(int argc, char *argv[]) {
  // Start total execution timer
  HighResTimer total_execution_timer;
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
                      config.use_non_cacheable, config.cpu_name, config.perf_cores, config.eff_cores, config.num_threads);
  print_cache_info(config.l1_cache_size, config.l2_cache_size, config.use_custom_cache_size, config.custom_cache_size_bytes);

  // --- Set QoS for the main thread (affects latency tests) ---
  kern_return_t qos_ret = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  if (qos_ret != KERN_SUCCESS) {
    // Non-critical error, just print a warning
    std::cerr << Messages::warning_qos_failed(qos_ret) << std::endl;
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
    PatternResults pattern_results;
    if (run_pattern_benchmarks(buffers, config, pattern_results) != EXIT_SUCCESS) {
      return EXIT_FAILURE;
    }
    print_pattern_results(pattern_results);
    
    // --- Save JSON Output if requested ---
    if (!config.output_file.empty()) {
      double total_elapsed_time_sec = total_execution_timer.stop();
      if (save_pattern_results_to_json(config, pattern_results, total_elapsed_time_sec) != EXIT_SUCCESS) {
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
                     stats.all_custom_latency_samples);

    // --- Save JSON Output if requested ---
    if (!config.output_file.empty()) {
      double total_elapsed_time_sec = total_execution_timer.stop();
      if (save_results_to_json(config, stats, total_elapsed_time_sec) != EXIT_FAILURE) {
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
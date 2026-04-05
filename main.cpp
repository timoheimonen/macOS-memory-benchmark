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
 * @file main.cpp
 * @brief Main entry point for the memory benchmark application
 *
 * This file contains the main program logic that orchestrates the execution
 * of memory benchmarks. It handles configuration parsing, mode-specific buffer
 * preparation, benchmark execution, and results output in both console and JSON formats.
 *
 * The program supports two primary benchmark modes:
 * - Standard benchmarks: Memory bandwidth and latency tests for different cache levels
 * - Pattern benchmarks: Access pattern-specific tests (forward, reverse, strided, random)
 *
 * @author Timo Heimonen
 * @date 2026
 */

#include <cstdlib>  // Exit codes
#include <iomanip>  // Output formatting
#include <iostream>
#include <string>

#include "utils/benchmark.h"
#include "core/memory/memory_manager.h"
#include "core/config/config.h"
#include "core/memory/buffer_manager.h"
#include "benchmark/benchmark_runner.h"
#include "benchmark/core_to_core_latency.h"
#include "benchmark/tlb_analysis.h"
#include "output/console/messages/messages_api.h"
#include "core/config/constants.h"
#include "output/json/json_output/json_output_api.h"
#include "pattern_benchmark/pattern_benchmark.h"
#include "core/signal/signal_handler.h"

// macOS specific memory management
#include <mach/mach.h>  // kern_return_t
#include <pthread/qos.h>

/**
 * @brief Main entry point for the memory benchmark application
 *
 * This function orchestrates the complete benchmark workflow:
 * 1. Parses and validates command-line arguments
 * 2. Configures system settings (QoS, cache parameters)
 * 3. Prepares benchmark buffers using mode-appropriate strategy
 * 4. Executes requested benchmarks (standard or pattern-based)
 * 5. Outputs results to console and optionally to JSON file
 *
 * The program supports multiple execution modes:
 * - Bandwidth-only measurements (--bandwidth-only)
 * - Latency-only measurements (--latency-only)
 * - Pattern-specific benchmarks (--patterns)
 * - Multiple loop iterations for statistical analysis (--loops)
 *
 * @param argc Number of command-line arguments
 * @param argv Array of command-line argument strings
 *
 * @return EXIT_SUCCESS (0) on successful completion
 * @return EXIT_FAILURE (1) on error (configuration, allocation, or benchmark failure)
 *
 * @note The main thread is set to QOS_CLASS_USER_INTERACTIVE for optimal latency test performance
 * @note All allocated buffers are automatically freed when going out of scope
 * @note Standard mode uses per-phase allocation; pattern mode uses pre-allocated buffers
 *
 * @see parse_arguments() for command-line argument details
 * @see run_all_benchmarks() for standard benchmark execution
 * @see run_all_pattern_benchmarks() for pattern benchmark execution
 */
int main(int argc, char *argv[]) {
  // Install signal handlers early (before any benchmark logic)
  install_signal_handlers();

  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "-analyze-core2core") {
      return run_core_to_core_latency_mode(argc, argv);
    }
  }

  // Start total execution timer
  auto timer_opt = HighResTimer::create();
  if (!timer_opt) {
    std::cerr << Messages::error_prefix()
              << Messages::error_timer_creation_failed()
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

  if (config.analyze_tlb) {
    return run_tlb_analysis(config);
  }
  
  // If no arguments provided, show help
  if (argc == 1) {
    print_usage(argv[0]);
    return EXIT_SUCCESS;
  }
  
  // If no mode flag is set (neither -benchmark nor -patterns), show help
  if (!config.run_benchmark && !config.run_patterns && !config.help_printed) {
    print_usage(argv[0]);
    return EXIT_SUCCESS;
  }
  
  if (validate_config(config) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  
  calculate_buffer_sizes(config);
  calculate_access_counts(config);

  size_t peak_allocation_bytes = 0;
  if (calculate_total_allocation_bytes(config, peak_allocation_bytes) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }

  // --- Print Config ---
  print_configuration(config.buffer_size, config.buffer_size_mb, peak_allocation_bytes,
                      config.iterations, config.loop_count,
                      config.use_non_cacheable, config.latency_stride_bytes,
                      latency_chain_mode_to_string(resolve_latency_chain_mode(
                          config.latency_chain_mode, config.latency_tlb_locality_bytes)),
                      config.latency_tlb_locality_bytes,
                      config.cpu_name, config.perf_cores, config.eff_cores, config.num_threads,
                      config.only_bandwidth, config.only_latency, config.run_patterns);
  print_cache_info(config.l1_cache_size, config.l2_cache_size, config.use_custom_cache_size, config.custom_cache_size_bytes);

  // --- Set QoS for the main thread (affects latency tests) ---
  kern_return_t qos_ret = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  if (qos_ret != KERN_SUCCESS) {
    // Non-critical error, just print a warning
    std::cerr << Messages::warning_prefix() << Messages::warning_qos_failed(qos_ret) << std::endl;
  }

  // --- Run Benchmarks ---
  // Block SIGINT/SIGTERM so worker threads inherit the blocked mask.
  // Only the main thread will receive Ctrl+C — worker threads finish their
  // current kernel before the main thread checks signal_received().
  block_benchmark_signals();

  BenchmarkBuffers buffers;
  if (config.run_patterns) {
    // Pattern mode uses pre-allocated src/dst buffers across pattern tests.
    if (allocate_all_buffers(config, buffers) != EXIT_SUCCESS) {
      return EXIT_FAILURE;
    }

    if (initialize_all_buffers(buffers, config) != EXIT_SUCCESS) {
      return EXIT_FAILURE;
    }

    // Run pattern benchmarks only
    PatternStatistics pattern_stats;
    if (run_all_pattern_benchmarks(buffers, config, pattern_stats) != EXIT_SUCCESS) {
      return EXIT_FAILURE;
    }
    
    // Print results - if single loop, print detailed results; if multiple loops, print last loop's results
    if (config.loop_count == 1) {
      print_pattern_results(extract_pattern_results_at(pattern_stats, 0));
    } else {
      size_t last_idx = pattern_stats.all_forward_read_bw.size() - 1;
      print_pattern_results(extract_pattern_results_at(pattern_stats, last_idx));
      
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
                     stats.all_tlb_hit_latency_ns,
                     stats.all_tlb_miss_latency_ns,
                     stats.all_page_walk_penalty_ns,
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

  // --- Restore Signal Mask ---
  restore_signal_mask();

  // --- Print Total Time ---
  double total_elapsed_time_sec = total_execution_timer.stop();                                  // Stop overall timer
  std::cout << Messages::msg_done_total_time(total_elapsed_time_sec) << std::endl;  // Print duration

  return EXIT_SUCCESS;  // Indicate success
}

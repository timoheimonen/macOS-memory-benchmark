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
 * The program supports five benchmark modes:
 * - Standard benchmarks: Memory bandwidth and latency tests for different cache levels
 * - Pattern benchmarks: Access pattern-specific tests (forward, reverse, strided, random)
 * - TLB analysis: Page-native paired locality measurements and boundary analysis
 * - Core-to-core analysis: Best-effort inter-core round-trip latency measurements
 * - GPU bandwidth: Standalone Metal GPU memory read/write/copy measurements
 *
 * Standard, pattern, TLB, and core-to-core modes also support validated parameter sweeps.
 * GPU bandwidth is intentionally standalone and does not participate in sweeps.
 *
 * @author Timo Heimonen
 * @date 2026
 */

#include <cstdlib>  // Exit codes
#include <iomanip>  // Output formatting
#include <iostream>
#include <string>

#include "utils/benchmark.h"
#include "core/config/config.h"
#include "core/config/mode_selector.h"
#include "core/memory/buffer_allocator.h"
#include "core/memory/buffer_manager.h"
#include "benchmark/benchmark_runner.h"
#include "benchmark/core_to_core_latency.h"
#include "benchmark/sweep_runner.h"
#include "benchmark/tlb_analysis.h"
#include "output/console/messages/messages_api.h"
#include "core/config/constants.h"
#include "output/json/json_output/json_output_api.h"
#include "pattern_benchmark/pattern_benchmark.h"
#include "core/signal/signal_handler.h"
#include "core/system/benchmark_qos.h"
#include "gpu_bandwidth/gpu_bandwidth.h"

namespace {

void set_benchmark_qos(BenchmarkConfig& config) {
  const MainThreadQosResult qos_result = prepare_main_thread_benchmark_qos();
  config.main_thread_qos_requested = qos_result.requested;
  config.main_thread_qos_applied = qos_result.applied;
  config.main_thread_qos_code = qos_result.code;
}

template <typename Fn>
int run_with_benchmark_preparation(BenchmarkConfig& config, Fn&& fn) {
  set_benchmark_qos(config);
  BenchmarkSignalMaskGuard signal_guard;
  return fn();
}

}  // namespace

/**
 * @brief Main entry point for the memory benchmark application
 *
 * This function orchestrates the complete benchmark workflow:
 * 1. Parses and validates command-line arguments
 * 2. Configures system settings (QoS, cache parameters)
 * 3. Prepares benchmark buffers using mode-appropriate strategy
 * 4. Executes the requested standard, pattern, TLB, core-to-core, or GPU mode
 * 5. Outputs results to console and optionally to JSON file
 *
 * The program supports multiple execution modes:
 * - Bandwidth-only measurements (--only-bandwidth)
 * - Latency-only measurements (--only-latency)
 * - Pattern-specific benchmarks (--patterns)
 * - Standalone TLB analysis (--analyze-tlb)
 * - Standalone core-to-core analysis (--analyze-core2core)
 * - Standalone GPU memory bandwidth (--gpu-bandwidth)
 * - Validated multi-configuration runs (--sweep)
 * - Multiple loop iterations for statistical analysis (--count)
 *
 * @param argc Number of command-line arguments
 * @param argv Array of command-line argument strings
 *
 * @return EXIT_SUCCESS (0) on successful completion
 * @return EXIT_FAILURE (1) on error (configuration, allocation, or benchmark failure)
 *
 * @note The main thread is set to QOS_CLASS_USER_INTERACTIVE for optimal latency test performance
 * @note All allocated buffers are automatically freed when going out of scope
 * @note Standard mode uses per-phase allocation; pattern mode owns one shared
 *       source/destination pair for the command lifetime.
 *
 * @see parse_arguments() for command-line argument details
 * @see run_all_benchmarks() for standard benchmark execution
 * @see run_all_pattern_benchmarks() for pattern benchmark execution
 */
int main(int argc, char *argv[]) {
  // Install signal handlers early (before any benchmark logic)
  install_signal_handlers();

  const PrimaryModeSelection mode_selection =
      select_primary_benchmark_mode(argc, argv);
  if (mode_selection.mode == PrimaryBenchmarkMode::Conflict) {
    std::cerr << Messages::error_prefix()
              << Messages::error_mutually_exclusive_modes(
                     mode_selection.selected_options[0],
                     mode_selection.selected_options[1])
              << std::endl;
    return EXIT_FAILURE;
  }
  if (mode_selection.mode == PrimaryBenchmarkMode::GpuBandwidth) {
    return run_gpu_bandwidth_mode(argc, argv);
  }
  if (mode_selection.mode == PrimaryBenchmarkMode::AnalyzeCoreToCore) {
    return run_core_to_core_latency_mode(argc, argv);
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

  // If -h/--help was handled, usage already printed — exit now
  if (config.help_printed) {
    return EXIT_SUCCESS;
  }

  // If no arguments provided, show help
  if (argc == 1) {
    print_help(argv[0]);
    return EXIT_SUCCESS;
  }

  // If no mode flag is set (neither --benchmark nor --patterns nor --analyze-tlb), show help
  if (!config.analyze_tlb && !config.run_benchmark && !config.run_patterns && !config.help_printed) {
    print_help(argv[0]);
    return EXIT_SUCCESS;
  }

  if (config.run_sweep) {
    if (validate_config(config) != EXIT_SUCCESS) {
      return EXIT_FAILURE;
    }
    return run_sweep_mode(config);
  }

  if (config.analyze_tlb) {
    if (validate_config(config) != EXIT_SUCCESS) {
      return EXIT_FAILURE;
    }
    return run_with_benchmark_preparation(config, [&]() {
      return run_tlb_analysis(config);
    });
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
                      config.only_bandwidth, config.only_latency, config.run_patterns,
                      config.user_specified_iterations);
  print_cache_info(config.l1_cache_size, config.l2_cache_size, config.use_custom_cache_size,
                   config.custom_cache_size_bytes);

  // --- Run Benchmarks ---
  const int benchmark_result = run_with_benchmark_preparation(config, [&]() {
    if (config.run_patterns) {
      // The pattern coordinator owns its shared src/dst mappings.
      PatternStatistics pattern_stats;
      const int pattern_run_status =
          run_all_pattern_benchmarks(config, pattern_stats);

      // Print detailed single-loop results or robust median headlines for repeated loops.
      if (config.loop_count == 1 && !pattern_stats.loop_results.empty()) {
        print_pattern_results(extract_pattern_results_at(pattern_stats, 0));
      } else if (!pattern_stats.loop_results.empty()) {
        print_pattern_results(extract_pattern_median_results(pattern_stats));

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

      if (pattern_run_status != EXIT_SUCCESS) {
        return pattern_run_status;
      }
    } else {
      // Run standard benchmarks
      std::cout << Messages::msg_running_benchmarks() << std::endl;

      BenchmarkBuffers buffers;
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

    return EXIT_SUCCESS;
  });
  if (benchmark_result != EXIT_SUCCESS) {
    return benchmark_result;
  }

  // --- Free Memory ---
  // std::cout << "\nFreeing memory..." << std::endl;
  // Memory is freed automatically when src_buffer_ptr, dst_buffer_ptr,
  // and lat_buffer_ptr go out of scope. No manual munmap needed.

  // --- Print Total Time ---
  double total_elapsed_time_sec = total_execution_timer.stop();                      // Stop overall timer
  std::cout << Messages::msg_done_total_time(total_elapsed_time_sec) << std::endl;  // Print duration

  return EXIT_SUCCESS;  // Indicate success
}

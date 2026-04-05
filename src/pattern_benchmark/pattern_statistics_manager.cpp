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
 * @file pattern_statistics_manager.cpp
 * @brief Statistics collection and management for pattern benchmarks
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 *
 * This file implements the top-level coordinator for running multiple
 * pattern benchmark loops and collecting aggregated statistics. It manages
 * the execution of all pattern types across multiple iterations and stores
 * results for statistical analysis.
 *
 * Coordinates execution of:
 * - Multiple benchmark loops (user-configurable loop count)
 * - All pattern types (forward, reverse, strided, random)
 * - Result aggregation into PatternStatistics structure
 */
#include "pattern_benchmark/pattern_benchmark.h"
#include "core/config/config.h"
#include "core/config/constants.h"
#include "core/signal/signal_handler.h"
#include "output/console/messages/messages_api.h"
#include <iostream>
#include <vector>
#include <cstdlib>

// ============================================================================
// Public API Functions
// ============================================================================

int run_all_pattern_benchmarks(const BenchmarkBuffers& buffers, const BenchmarkConfig& config, PatternStatistics& stats) {
  using namespace Constants;
  
  // Initialize result vectors
  stats.all_forward_read_bw.clear();
  stats.all_forward_write_bw.clear();
  stats.all_forward_copy_bw.clear();
  stats.all_reverse_read_bw.clear();
  stats.all_reverse_write_bw.clear();
  stats.all_reverse_copy_bw.clear();
  stats.all_strided_64_read_bw.clear();
  stats.all_strided_64_write_bw.clear();
  stats.all_strided_64_copy_bw.clear();
  stats.all_strided_4096_read_bw.clear();
  stats.all_strided_4096_write_bw.clear();
  stats.all_strided_4096_copy_bw.clear();
  stats.all_strided_16384_read_bw.clear();
  stats.all_strided_16384_write_bw.clear();
  stats.all_strided_16384_copy_bw.clear();
  stats.all_strided_2mb_read_bw.clear();
  stats.all_strided_2mb_write_bw.clear();
  stats.all_strided_2mb_copy_bw.clear();
  stats.all_random_read_bw.clear();
  stats.all_random_write_bw.clear();
  stats.all_random_copy_bw.clear();
  
  // Pre-allocate vector space if needed
  if (config.loop_count > 0) {
    stats.all_forward_read_bw.reserve(config.loop_count);
    stats.all_forward_write_bw.reserve(config.loop_count);
    stats.all_forward_copy_bw.reserve(config.loop_count);
    stats.all_reverse_read_bw.reserve(config.loop_count);
    stats.all_reverse_write_bw.reserve(config.loop_count);
    stats.all_reverse_copy_bw.reserve(config.loop_count);
    stats.all_strided_64_read_bw.reserve(config.loop_count);
    stats.all_strided_64_write_bw.reserve(config.loop_count);
    stats.all_strided_64_copy_bw.reserve(config.loop_count);
    stats.all_strided_4096_read_bw.reserve(config.loop_count);
    stats.all_strided_4096_write_bw.reserve(config.loop_count);
    stats.all_strided_4096_copy_bw.reserve(config.loop_count);
    stats.all_strided_16384_read_bw.reserve(config.loop_count);
    stats.all_strided_16384_write_bw.reserve(config.loop_count);
    stats.all_strided_16384_copy_bw.reserve(config.loop_count);
    stats.all_strided_2mb_read_bw.reserve(config.loop_count);
    stats.all_strided_2mb_write_bw.reserve(config.loop_count);
    stats.all_strided_2mb_copy_bw.reserve(config.loop_count);
    stats.all_random_read_bw.reserve(config.loop_count);
    stats.all_random_write_bw.reserve(config.loop_count);
    stats.all_random_copy_bw.reserve(config.loop_count);
  }
  
  std::cout << Messages::msg_running_pattern_benchmarks() << std::flush;
  
  // Main pattern benchmark loop
  for (int loop = 0; loop < config.loop_count; ++loop) {
    // Check for Ctrl+C between pattern loops
    if (signal_received()) {
      std::cout << std::endl << Messages::msg_interrupted_by_user() << std::endl;
      return EXIT_SUCCESS;
    }

    try {
      PatternResults loop_results;
      
      // Run pattern benchmarks for this loop
      int status = run_pattern_benchmarks(buffers, config, loop_results);
      if (status != EXIT_SUCCESS) {
        return status;
      }
      
      // Store results for this loop
      stats.all_forward_read_bw.push_back(loop_results.forward_read_bw);
      stats.all_forward_write_bw.push_back(loop_results.forward_write_bw);
      stats.all_forward_copy_bw.push_back(loop_results.forward_copy_bw);
      stats.all_reverse_read_bw.push_back(loop_results.reverse_read_bw);
      stats.all_reverse_write_bw.push_back(loop_results.reverse_write_bw);
      stats.all_reverse_copy_bw.push_back(loop_results.reverse_copy_bw);
      stats.all_strided_64_read_bw.push_back(loop_results.strided_64_read_bw);
      stats.all_strided_64_write_bw.push_back(loop_results.strided_64_write_bw);
      stats.all_strided_64_copy_bw.push_back(loop_results.strided_64_copy_bw);
      stats.all_strided_4096_read_bw.push_back(loop_results.strided_4096_read_bw);
      stats.all_strided_4096_write_bw.push_back(loop_results.strided_4096_write_bw);
      stats.all_strided_4096_copy_bw.push_back(loop_results.strided_4096_copy_bw);
      stats.all_strided_16384_read_bw.push_back(loop_results.strided_16384_read_bw);
      stats.all_strided_16384_write_bw.push_back(loop_results.strided_16384_write_bw);
      stats.all_strided_16384_copy_bw.push_back(loop_results.strided_16384_copy_bw);
      stats.all_strided_2mb_read_bw.push_back(loop_results.strided_2mb_read_bw);
      stats.all_strided_2mb_write_bw.push_back(loop_results.strided_2mb_write_bw);
      stats.all_strided_2mb_copy_bw.push_back(loop_results.strided_2mb_copy_bw);
      stats.all_random_read_bw.push_back(loop_results.random_read_bw);
      stats.all_random_write_bw.push_back(loop_results.random_write_bw);
      stats.all_random_copy_bw.push_back(loop_results.random_copy_bw);
      
      // Print simple progress message for each loop
      if (config.loop_count > 1) {
        std::cout << '\r' << std::flush;  // Clear progress indicator
        std::cout << Messages::msg_pattern_benchmark_loop_completed(loop + 1, config.loop_count) << std::endl;
      }
    } catch (const std::exception &e) {
      std::cerr << Messages::error_benchmark_loop(loop, e.what()) << std::endl;
      return EXIT_FAILURE;
    }
  }
  
  return EXIT_SUCCESS;
}

PatternResults extract_pattern_results_at(const PatternStatistics& stats, size_t index) {
  PatternResults result;

  if (stats.all_forward_read_bw.empty()) {
    return result;
  }

  if (index >= stats.all_forward_read_bw.size()) {
    index = stats.all_forward_read_bw.size() - 1;
  }

  result.forward_read_bw = stats.all_forward_read_bw[index];
  result.forward_write_bw = stats.all_forward_write_bw[index];
  result.forward_copy_bw = stats.all_forward_copy_bw[index];
  result.reverse_read_bw = stats.all_reverse_read_bw[index];
  result.reverse_write_bw = stats.all_reverse_write_bw[index];
  result.reverse_copy_bw = stats.all_reverse_copy_bw[index];
  result.strided_64_read_bw = stats.all_strided_64_read_bw[index];
  result.strided_64_write_bw = stats.all_strided_64_write_bw[index];
  result.strided_64_copy_bw = stats.all_strided_64_copy_bw[index];
  result.strided_4096_read_bw = stats.all_strided_4096_read_bw[index];
  result.strided_4096_write_bw = stats.all_strided_4096_write_bw[index];
  result.strided_4096_copy_bw = stats.all_strided_4096_copy_bw[index];
  result.strided_16384_read_bw = stats.all_strided_16384_read_bw[index];
  result.strided_16384_write_bw = stats.all_strided_16384_write_bw[index];
  result.strided_16384_copy_bw = stats.all_strided_16384_copy_bw[index];
  result.strided_2mb_read_bw = stats.all_strided_2mb_read_bw[index];
  result.strided_2mb_write_bw = stats.all_strided_2mb_write_bw[index];
  result.strided_2mb_copy_bw = stats.all_strided_2mb_copy_bw[index];
  result.random_read_bw = stats.all_random_read_bw[index];
  result.random_write_bw = stats.all_random_write_bw[index];
  result.random_copy_bw = stats.all_random_copy_bw[index];

  return result;
}

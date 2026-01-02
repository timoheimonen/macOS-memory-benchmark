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
#include "pattern_benchmark/pattern_benchmark.h"
#include "core/config/config.h"
#include "core/config/constants.h"
#include "output/console/messages.h"
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
    stats.all_random_read_bw.reserve(config.loop_count);
    stats.all_random_write_bw.reserve(config.loop_count);
    stats.all_random_copy_bw.reserve(config.loop_count);
  }
  
  std::cout << Messages::msg_running_pattern_benchmarks() << std::flush;
  
  // Main pattern benchmark loop
  for (int loop = 0; loop < config.loop_count; ++loop) {
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
      stats.all_random_read_bw.push_back(loop_results.random_read_bw);
      stats.all_random_write_bw.push_back(loop_results.random_write_bw);
      stats.all_random_copy_bw.push_back(loop_results.random_copy_bw);
      
      // Print simple progress message for each loop
      if (config.loop_count > 1) {
        std::cout << '\r' << std::flush;  // Clear progress indicator
        std::cout << "Pattern benchmarks - Loop " << (loop + 1) << "/" << config.loop_count << " completed" << std::endl;
      }
    } catch (const std::exception &e) {
      std::cerr << Messages::error_benchmark_loop(loop, e.what()) << std::endl;
      return EXIT_FAILURE;
    }
  }
  
  return EXIT_SUCCESS;
}


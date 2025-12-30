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
#include "pattern_benchmark/pattern_benchmark.h"
#include "utils/benchmark.h"
#include "core/memory/buffer_manager.h"
#include "core/config/config.h"
#include "core/config/constants.h"
#include "output/console/messages.h"
#include <iostream>
#include <vector>
#include <atomic>
#include <cstdlib>

// Forward declarations from helpers.cpp
double run_pattern_read_test(void* buffer, size_t size, int iterations, 
                             uint64_t (*read_func)(const void*, size_t),
                             std::atomic<uint64_t>& checksum, HighResTimer& timer);
double run_pattern_write_test(void* buffer, size_t size, int iterations,
                              void (*write_func)(void*, size_t),
                              HighResTimer& timer);
double run_pattern_copy_test(void* dst, void* src, size_t size, int iterations,
                             void (*copy_func)(void*, const void*, size_t),
                             HighResTimer& timer);
double run_pattern_read_strided_test(void* buffer, size_t size, size_t stride, int iterations,
                                     std::atomic<uint64_t>& checksum, HighResTimer& timer);
double run_pattern_write_strided_test(void* buffer, size_t size, size_t stride, int iterations,
                                      HighResTimer& timer);
double run_pattern_copy_strided_test(void* dst, void* src, size_t size, size_t stride, int iterations,
                                     HighResTimer& timer);
double run_pattern_read_random_test(void* buffer, const std::vector<size_t>& indices, int iterations,
                                    std::atomic<uint64_t>& checksum, HighResTimer& timer);
double run_pattern_write_random_test(void* buffer, const std::vector<size_t>& indices, int iterations,
                                     HighResTimer& timer);
double run_pattern_copy_random_test(void* dst, void* src, const std::vector<size_t>& indices, int iterations,
                                    HighResTimer& timer);

// Forward declarations from validation.cpp
bool validate_stride(size_t stride, size_t buffer_size);
bool validate_random_indices(const std::vector<size_t>& indices, size_t buffer_size);

// Forward declarations from execution_utils.cpp
double calculate_bandwidth(size_t data_size, int iterations, double elapsed_time_ns);
std::vector<size_t> generate_random_indices(size_t buffer_size, size_t num_accesses);
size_t calculate_num_random_accesses(size_t buffer_size);

// Forward declarations from execution_strided.cpp
int run_strided_pattern_benchmarks(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
                                     size_t stride, double& read_bw, double& write_bw, double& copy_bw,
                                     HighResTimer& timer);

// Forward declarations from execution_patterns.cpp
void run_forward_pattern_benchmarks(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
                                    PatternResults& results, HighResTimer& timer);
void run_reverse_pattern_benchmarks(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
                                   PatternResults& results, HighResTimer& timer);
int run_random_pattern_benchmarks(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
                                   const std::vector<size_t>& random_indices, size_t num_accesses,
                                   PatternResults& results, HighResTimer& timer);

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

int run_pattern_benchmarks(const BenchmarkBuffers& buffers, const BenchmarkConfig& config, PatternResults& results) {
  using namespace Constants;

  auto timer_opt = HighResTimer::create();
  if (!timer_opt) {
    std::cerr << Messages::error_prefix()
              << "Failed to create pattern benchmark timer."
              << std::endl;
    return EXIT_FAILURE;
  }
  auto& timer = *timer_opt;

  // Calculate number of accesses for random pattern
  size_t num_random_accesses = calculate_num_random_accesses(config.buffer_size);
  
  // Generate random indices once
  std::vector<size_t> random_indices = generate_random_indices(config.buffer_size, num_random_accesses);
  
  std::cout << Messages::msg_running_pattern_benchmarks() << std::flush;
  
  // Sequential Forward (baseline)
  run_forward_pattern_benchmarks(buffers, config, results, timer);
  
  // Sequential Reverse
  run_reverse_pattern_benchmarks(buffers, config, results, timer);
  
  // Strided (Cache Line)
  int status = run_strided_pattern_benchmarks(buffers, config, PATTERN_STRIDE_CACHE_LINE, 
                                               results.strided_64_read_bw,
                                               results.strided_64_write_bw, 
                                               results.strided_64_copy_bw, timer);
  if (status != EXIT_SUCCESS) {
    return status;
  }
  
  // Strided (Page) - may be skipped if buffer too small
  status = run_strided_pattern_benchmarks(buffers, config, PATTERN_STRIDE_PAGE, 
                                          results.strided_4096_read_bw,
                                          results.strided_4096_write_bw, 
                                          results.strided_4096_copy_bw, timer);
  if (status != EXIT_SUCCESS) {
    return status;
  }
  
  // Random Uniform - may be skipped if buffer too small or no valid indices
  status = run_random_pattern_benchmarks(buffers, config, random_indices, num_random_accesses, results, timer);
  if (status != EXIT_SUCCESS) {
    return status;
  }
  
  return EXIT_SUCCESS;
}

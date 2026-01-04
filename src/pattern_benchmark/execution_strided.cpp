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
#include "utils/benchmark.h"
#include "core/memory/buffer_manager.h"
#include "core/config/config.h"
#include "core/config/constants.h"
#include "output/console/messages.h"
#include "warmup/warmup.h"
#include <iostream>
#include <atomic>
#include <cstdlib>

// Forward declarations from helpers.cpp
double run_pattern_read_strided_test(void* buffer, size_t size, size_t stride, int iterations,
                                     std::atomic<uint64_t>& checksum, HighResTimer& timer,
                                     int num_threads);
double run_pattern_write_strided_test(void* buffer, size_t size, size_t stride, int iterations,
                                      HighResTimer& timer, int num_threads);
double run_pattern_copy_strided_test(void* dst, void* src, size_t size, size_t stride, int iterations,
                                     HighResTimer& timer, int num_threads);

// Forward declarations from validation.cpp
bool validate_stride(size_t stride, size_t buffer_size);

// Forward declarations from execution_utils.cpp
double calculate_bandwidth(size_t data_size, int iterations, double elapsed_time_ns);

// ============================================================================
// Strided Pattern Helper Functions
// ============================================================================

// Calculate effective buffer size for strided pattern
static bool calculate_strided_params(size_t buffer_size, size_t stride, 
                                      size_t& effective_buffer_size, 
                                      size_t& num_iterations, 
                                      size_t& actual_data_accessed) {
  using namespace Constants;
  
  // The strided assembly functions use modulo arithmetic: addr = base + (offset % byteCount)
  // Each access loads/stores PATTERN_ACCESS_SIZE_BYTES bytes, so we need to ensure 
  // addr + PATTERN_ACCESS_SIZE_BYTES <= buffer_size
  // Therefore, we pass (buffer_size - PATTERN_ACCESS_SIZE_BYTES) as the effective buffer size
  effective_buffer_size = buffer_size - PATTERN_ACCESS_SIZE_BYTES;
  
  // Ensure effective buffer size is at least as large as stride
  if (effective_buffer_size < stride) {
    std::cerr << Messages::error_prefix() 
              << Messages::error_stride_too_large(stride, buffer_size) << std::endl;
    return false;
  }
  
  // Calculate actual data accessed for strided pattern
  // The strided loop accesses PATTERN_ACCESS_SIZE_BYTES bytes per iteration, advancing by stride each time
  // Number of iterations = ceil(effective_buffer_size / stride)
  num_iterations = (effective_buffer_size + stride - 1) / stride;  // Ceiling division
  
  // Ensure we have at least one iteration
  if (num_iterations == 0) {
    std::cerr << Messages::error_prefix() 
              << Messages::error_no_iterations_strided() << std::endl;
    return false;
  }
  
  actual_data_accessed = num_iterations * PATTERN_ACCESS_SIZE_BYTES;
  return true;
}

// ============================================================================
// Strided Pattern Execution Functions
// ============================================================================

// Run strided pattern benchmarks (access with specified stride)
// Returns EXIT_SUCCESS on success, EXIT_FAILURE on error, or skips pattern if buffer too small
int run_strided_pattern_benchmarks(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
                                     size_t stride, double& read_bw, double& write_bw, double& copy_bw,
                                     HighResTimer& timer) {
  using namespace Constants;
  
  // Initialize results to 0 in case we skip this pattern
  read_bw = 0.0;
  write_bw = 0.0;
  copy_bw = 0.0;
  
  // Validate stride - if validation fails due to buffer size, skip pattern gracefully
  if (!validate_stride(stride, config.buffer_size)) {
    // Buffer too small for this stride - skip pattern (not an error)
    return EXIT_SUCCESS;
  }
  
  // Calculate strided parameters - if calculation fails, skip pattern gracefully
  size_t effective_buffer_size;
  size_t num_iterations;
  size_t actual_data_accessed;
  if (!calculate_strided_params(config.buffer_size, stride, effective_buffer_size, 
                                 num_iterations, actual_data_accessed)) {
    // Buffer too small for this stride - skip pattern (not an error)
    return EXIT_SUCCESS;
  }
  
  // Execute read benchmark
  show_progress();
  std::atomic<uint64_t> checksum{0};
  warmup_read_strided(buffers.src_buffer(), effective_buffer_size, stride, config.num_threads, checksum);
  double read_time = run_pattern_read_strided_test(buffers.src_buffer(), effective_buffer_size, stride,
                                                     config.iterations, checksum, timer, config.num_threads);
  read_bw = calculate_bandwidth(actual_data_accessed, config.iterations, read_time);

  // Execute write benchmark
  show_progress();
  warmup_write_strided(buffers.dst_buffer(), effective_buffer_size, stride, config.num_threads);
  double write_time = run_pattern_write_strided_test(buffers.dst_buffer(), effective_buffer_size, stride,
                                                      config.iterations, timer, config.num_threads);
  write_bw = calculate_bandwidth(actual_data_accessed, config.iterations, write_time);

  // Execute copy benchmark
  show_progress();
  warmup_copy_strided(buffers.dst_buffer(), buffers.src_buffer(), effective_buffer_size, stride, config.num_threads);
  double copy_time = run_pattern_copy_strided_test(buffers.dst_buffer(), buffers.src_buffer(),
                                                    effective_buffer_size, stride, config.iterations, timer,
                                                    config.num_threads);
  // For copy: actual_data_accessed bytes are read + actual_data_accessed bytes are written per iteration
  copy_bw = calculate_bandwidth(actual_data_accessed * Constants::COPY_OPERATION_MULTIPLIER,
                                config.iterations, copy_time);
  
  return EXIT_SUCCESS;
}


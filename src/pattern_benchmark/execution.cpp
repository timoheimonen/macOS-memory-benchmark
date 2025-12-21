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
#include "benchmark.h"
#include "buffer_manager.h"
#include "config.h"
#include "constants.h"
#include "messages.h"
#include <iostream>
#include <random>
#include <algorithm>
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

// ============================================================================
// Utility Functions
// ============================================================================

// Helper function to calculate bandwidth from time and data size
static double calculate_bandwidth(size_t data_size, int iterations, double elapsed_time_ns) {
  // Avoid division by zero - use minimum time if time is too small
  // This handles cases where very fast operations complete in < 1ns due to timer resolution
  double effective_time_ns = (elapsed_time_ns < Constants::PATTERN_MIN_TIME_NS) 
                              ? Constants::PATTERN_MIN_TIME_NS 
                              : elapsed_time_ns;
  return (static_cast<double>(data_size) * iterations) / 
         (effective_time_ns * Constants::NANOSECONDS_PER_SECOND);
}

// Helper function to calculate maximum valid aligned offset
static size_t calculate_max_aligned_offset(size_t buffer_size) {
  using namespace Constants;
  if (buffer_size < PATTERN_MIN_BUFFER_SIZE_BYTES) {
    return 0;
  }
  // Round down (buffer_size - PATTERN_ACCESS_SIZE_BYTES) to alignment boundary
  return ((buffer_size - PATTERN_ACCESS_SIZE_BYTES) / PATTERN_ACCESS_SIZE_BYTES) * PATTERN_ACCESS_SIZE_BYTES;
}

// Helper function to generate random indices for random access pattern
static std::vector<size_t> generate_random_indices(size_t buffer_size, size_t num_accesses) {
  using namespace Constants;
  std::vector<size_t> indices;
  indices.reserve(num_accesses);
  
  // Generate random offsets that are aligned (for aligned loads)
  // Each access loads PATTERN_ACCESS_SIZE_BYTES bytes, so we need offset + PATTERN_ACCESS_SIZE_BYTES <= buffer_size
  size_t max_offset = calculate_max_aligned_offset(buffer_size);
  if (max_offset == 0) {
    return indices;  // Return empty vector if buffer too small
  }
  
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> dis(0, max_offset / PATTERN_ACCESS_SIZE_BYTES);  // Index in units
  
  for (size_t i = 0; i < num_accesses; ++i) {
    size_t offset = dis(gen) * PATTERN_ACCESS_SIZE_BYTES;  // Convert to byte offset
    indices.push_back(offset);
  }
  
  return indices;
}

// Helper function to calculate number of random accesses based on buffer size
static size_t calculate_num_random_accesses(size_t buffer_size) {
  using namespace Constants;
  size_t num_accesses = buffer_size / PATTERN_ACCESS_SIZE_BYTES;
  if (num_accesses < PATTERN_RANDOM_ACCESS_MIN) {
    num_accesses = PATTERN_RANDOM_ACCESS_MIN;
  }
  if (num_accesses > PATTERN_RANDOM_ACCESS_MAX) {
    num_accesses = PATTERN_RANDOM_ACCESS_MAX;
  }
  return num_accesses;
}

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
// Pattern Benchmark Execution Functions
// ============================================================================

// Run forward pattern benchmarks (baseline sequential access)
static void run_forward_pattern_benchmarks(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
                                            PatternResults& results, HighResTimer& timer) {
  show_progress();
  std::atomic<uint64_t> checksum{0};
  warmup_read(buffers.src_buffer(), config.buffer_size, config.num_threads, checksum);
  double read_time = run_read_test(buffers.src_buffer(), config.buffer_size, config.iterations,
                                    config.num_threads, checksum, timer);
  results.forward_read_bw = calculate_bandwidth(config.buffer_size, config.iterations, read_time);
  
  show_progress();
  warmup_write(buffers.dst_buffer(), config.buffer_size, config.num_threads);
  double write_time = run_write_test(buffers.dst_buffer(), config.buffer_size, config.iterations,
                                      config.num_threads, timer);
  results.forward_write_bw = calculate_bandwidth(config.buffer_size, config.iterations, write_time);
  
  show_progress();
  warmup_copy(buffers.dst_buffer(), buffers.src_buffer(), config.buffer_size, config.num_threads);
  double copy_time = run_copy_test(buffers.dst_buffer(), buffers.src_buffer(), config.buffer_size,
                                   config.iterations, config.num_threads, timer);
  results.forward_copy_bw = calculate_bandwidth(config.buffer_size, config.iterations, copy_time);
}

// Run reverse pattern benchmarks (backward sequential access)
static void run_reverse_pattern_benchmarks(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
                                           PatternResults& results, HighResTimer& timer) {
  show_progress();
  std::atomic<uint64_t> checksum{0};
  warmup_read(buffers.src_buffer(), config.buffer_size, config.num_threads, checksum);
  double read_time = run_pattern_read_test(buffers.src_buffer(), config.buffer_size, config.iterations,
                                            memory_read_reverse_loop_asm, checksum, timer);
  results.reverse_read_bw = calculate_bandwidth(config.buffer_size, config.iterations, read_time);
  
  show_progress();
  warmup_write(buffers.dst_buffer(), config.buffer_size, config.num_threads);
  double write_time = run_pattern_write_test(buffers.dst_buffer(), config.buffer_size, config.iterations,
                                             memory_write_reverse_loop_asm, timer);
  results.reverse_write_bw = calculate_bandwidth(config.buffer_size, config.iterations, write_time);
  
  show_progress();
  warmup_copy(buffers.dst_buffer(), buffers.src_buffer(), config.buffer_size, config.num_threads);
  double copy_time = run_pattern_copy_test(buffers.dst_buffer(), buffers.src_buffer(), config.buffer_size,
                                            config.iterations, memory_copy_reverse_loop_asm, timer);
  results.reverse_copy_bw = calculate_bandwidth(config.buffer_size, config.iterations, copy_time);
}

// Run strided pattern benchmarks (access with specified stride)
// Returns EXIT_SUCCESS on success, EXIT_FAILURE on error, or skips pattern if buffer too small
static int run_strided_pattern_benchmarks(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
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
                                                     config.iterations, checksum, timer);
  read_bw = calculate_bandwidth(actual_data_accessed, config.iterations, read_time);
  
  // Execute write benchmark
  show_progress();
  warmup_write_strided(buffers.dst_buffer(), effective_buffer_size, stride, config.num_threads);
  double write_time = run_pattern_write_strided_test(buffers.dst_buffer(), effective_buffer_size, stride,
                                                      config.iterations, timer);
  write_bw = calculate_bandwidth(actual_data_accessed, config.iterations, write_time);
  
  // Execute copy benchmark
  show_progress();
  warmup_copy_strided(buffers.dst_buffer(), buffers.src_buffer(), effective_buffer_size, stride, config.num_threads);
  double copy_time = run_pattern_copy_strided_test(buffers.dst_buffer(), buffers.src_buffer(),
                                                    effective_buffer_size, stride, config.iterations, timer);
  // For copy: actual_data_accessed bytes are read + actual_data_accessed bytes are written per iteration
  copy_bw = calculate_bandwidth(actual_data_accessed * Constants::COPY_OPERATION_MULTIPLIER, 
                                config.iterations, copy_time);
  
  return EXIT_SUCCESS;
}

// Prepare warmup indices for random pattern
static std::vector<size_t> prepare_warmup_indices(const std::vector<size_t>& random_indices) {
  using namespace Constants;
  size_t warmup_indices_count = std::min(static_cast<size_t>(PATTERN_WARMUP_INDICES_MAX), 
                                         random_indices.size() / PATTERN_WARMUP_INDICES_FRACTION);
  if (warmup_indices_count == 0) {
    warmup_indices_count = 1;  // At least one index
  }
  return std::vector<size_t>(random_indices.begin(), random_indices.begin() + warmup_indices_count);
}

// Run random pattern benchmarks (uniform random access)
// Returns EXIT_SUCCESS on success, EXIT_FAILURE on error, or skips pattern if buffer too small
static int run_random_pattern_benchmarks(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
                                           const std::vector<size_t>& random_indices, size_t num_accesses,
                                           PatternResults& results, HighResTimer& timer) {
  using namespace Constants;
  
  // Initialize results to 0 in case we skip this pattern
  results.random_read_bw = 0.0;
  results.random_write_bw = 0.0;
  results.random_copy_bw = 0.0;
  
  // Validate indices - if validation fails due to buffer size, skip pattern gracefully
  if (!validate_random_indices(random_indices, config.buffer_size)) {
    // No valid indices or buffer too small - skip pattern (not an error)
    return EXIT_SUCCESS;
  }
  
  // Prepare warmup indices
  std::vector<size_t> warmup_indices = prepare_warmup_indices(random_indices);
  
  // Execute read benchmark
  show_progress();
  std::atomic<uint64_t> checksum{0};
  warmup_read_random(buffers.src_buffer(), warmup_indices, config.num_threads, checksum);
  double read_time = run_pattern_read_random_test(buffers.src_buffer(), random_indices,
                                                   config.iterations, checksum, timer);
  // For random, we use num_accesses * PATTERN_ACCESS_SIZE_BYTES instead of buffer_size
  results.random_read_bw = calculate_bandwidth(num_accesses * PATTERN_ACCESS_SIZE_BYTES, 
                                               config.iterations, read_time);
  
  // Execute write benchmark
  show_progress();
  warmup_write_random(buffers.dst_buffer(), warmup_indices, config.num_threads);
  double write_time = run_pattern_write_random_test(buffers.dst_buffer(), random_indices,
                                                      config.iterations, timer);
  results.random_write_bw = calculate_bandwidth(num_accesses * PATTERN_ACCESS_SIZE_BYTES, 
                                                config.iterations, write_time);
  
  // Execute copy benchmark
  show_progress();
  warmup_copy_random(buffers.dst_buffer(), buffers.src_buffer(), warmup_indices, config.num_threads);
  double copy_time = run_pattern_copy_random_test(buffers.dst_buffer(), buffers.src_buffer(), random_indices,
                                                   config.iterations, timer);
  results.random_copy_bw = calculate_bandwidth(num_accesses * PATTERN_ACCESS_SIZE_BYTES, 
                                              config.iterations, copy_time);
  
  return EXIT_SUCCESS;
}

// ============================================================================
// Public API Functions
// ============================================================================

int run_pattern_benchmarks(const BenchmarkBuffers& buffers, const BenchmarkConfig& config, PatternResults& results) {
  using namespace Constants;
  
  HighResTimer timer;
  
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


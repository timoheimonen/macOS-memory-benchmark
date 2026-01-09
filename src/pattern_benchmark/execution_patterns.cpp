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
 * @file execution_patterns.cpp
 * @brief Sequential and random pattern benchmark execution
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 *
 * This file implements the execution of sequential (forward/reverse) and
 * random memory access pattern benchmarks. It coordinates warmup routines,
 * test execution, timing, and bandwidth calculations for these patterns.
 *
 * Implemented patterns:
 * - Sequential forward: Standard linear memory access (baseline)
 * - Sequential reverse: Backward linear memory access
 * - Random uniform: Pseudo-random memory access at cache-line-aligned offsets
 */
#include "pattern_benchmark/pattern_benchmark.h"
#include "utils/benchmark.h"
#include "core/memory/buffer_manager.h"
#include "core/config/config.h"
#include "core/config/constants.h"
#include "warmup/warmup.h"
#include <atomic>
#include <vector>
#include <algorithm>
#include <cstdlib>

// Forward declarations from helpers.cpp
double run_pattern_read_test(void* buffer, size_t size, int iterations,
                             uint64_t (*read_func)(const void*, size_t),
                             std::atomic<uint64_t>& checksum, HighResTimer& timer,
                             int num_threads);
double run_pattern_write_test(void* buffer, size_t size, int iterations,
                              void (*write_func)(void*, size_t),
                              HighResTimer& timer, int num_threads);
double run_pattern_copy_test(void* dst, void* src, size_t size, int iterations,
                             void (*copy_func)(void*, const void*, size_t),
                             HighResTimer& timer, int num_threads);
double run_pattern_read_random_test(void* buffer, const std::vector<size_t>& indices, int iterations,
                                    std::atomic<uint64_t>& checksum, HighResTimer& timer,
                                    int num_threads, size_t buffer_size);
double run_pattern_write_random_test(void* buffer, const std::vector<size_t>& indices, int iterations,
                                     HighResTimer& timer, int num_threads, size_t buffer_size);
double run_pattern_copy_random_test(void* dst, void* src, const std::vector<size_t>& indices, int iterations,
                                    HighResTimer& timer, int num_threads, size_t buffer_size);

// Forward declarations from validation.cpp
bool validate_random_indices(const std::vector<size_t>& indices, size_t buffer_size);

// Forward declarations from execution_utils.cpp
double calculate_bandwidth(size_t data_size, int iterations, double elapsed_time_ns);

// ============================================================================
// Pattern Benchmark Execution Functions
// ============================================================================

// Run forward pattern benchmarks (baseline sequential access)
void run_forward_pattern_benchmarks(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
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
  results.forward_copy_bw = calculate_bandwidth(config.buffer_size * Constants::COPY_OPERATION_MULTIPLIER, 
                                                config.iterations, copy_time);
}

// Run reverse pattern benchmarks (backward sequential access)
void run_reverse_pattern_benchmarks(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
                                   PatternResults& results, HighResTimer& timer) {
  show_progress();
  std::atomic<uint64_t> checksum{0};
  warmup_read(buffers.src_buffer(), config.buffer_size, config.num_threads, checksum);
  double read_time = run_pattern_read_test(buffers.src_buffer(), config.buffer_size, config.iterations,
                                            memory_read_reverse_loop_asm, checksum, timer, config.num_threads);
  results.reverse_read_bw = calculate_bandwidth(config.buffer_size, config.iterations, read_time);

  show_progress();
  warmup_write(buffers.dst_buffer(), config.buffer_size, config.num_threads);
  double write_time = run_pattern_write_test(buffers.dst_buffer(), config.buffer_size, config.iterations,
                                             memory_write_reverse_loop_asm, timer, config.num_threads);
  results.reverse_write_bw = calculate_bandwidth(config.buffer_size, config.iterations, write_time);

  show_progress();
  warmup_copy(buffers.dst_buffer(), buffers.src_buffer(), config.buffer_size, config.num_threads);
  double copy_time = run_pattern_copy_test(buffers.dst_buffer(), buffers.src_buffer(), config.buffer_size,
                                            config.iterations, memory_copy_reverse_loop_asm, timer, config.num_threads);
  results.reverse_copy_bw = calculate_bandwidth(config.buffer_size * Constants::COPY_OPERATION_MULTIPLIER,
                                                config.iterations, copy_time);
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
int run_random_pattern_benchmarks(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
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
                                                   config.iterations, checksum, timer,
                                                   config.num_threads, config.buffer_size);
  // For random, we use num_accesses * PATTERN_ACCESS_SIZE_BYTES instead of buffer_size
  results.random_read_bw = calculate_bandwidth(num_accesses * PATTERN_ACCESS_SIZE_BYTES,
                                               config.iterations, read_time);

  // Execute write benchmark
  show_progress();
  warmup_write_random(buffers.dst_buffer(), warmup_indices, config.num_threads);
  double write_time = run_pattern_write_random_test(buffers.dst_buffer(), random_indices,
                                                      config.iterations, timer,
                                                      config.num_threads, config.buffer_size);
  results.random_write_bw = calculate_bandwidth(num_accesses * PATTERN_ACCESS_SIZE_BYTES,
                                                config.iterations, write_time);

  // Execute copy benchmark
  show_progress();
  warmup_copy_random(buffers.dst_buffer(), buffers.src_buffer(), warmup_indices, config.num_threads);
  double copy_time = run_pattern_copy_random_test(buffers.dst_buffer(), buffers.src_buffer(), random_indices,
                                                   config.iterations, timer,
                                                   config.num_threads, config.buffer_size);
  results.random_copy_bw = calculate_bandwidth(num_accesses * PATTERN_ACCESS_SIZE_BYTES * Constants::COPY_OPERATION_MULTIPLIER,
                                              config.iterations, copy_time);
  
  return EXIT_SUCCESS;
}


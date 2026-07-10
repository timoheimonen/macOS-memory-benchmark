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
 * @file execution_strided.cpp
 * @brief Strided pattern benchmark execution
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 *
 * This file implements the execution of strided memory access pattern
 * benchmarks. Strided access patterns skip fixed numbers of bytes between
 * successive accesses, used to evaluate cache line utilization and TLB
 * behavior.
 *
 * Supported stride patterns:
 * - Cache line stride (64 bytes): Tests cache line prefetcher effectiveness
 * - Page stride (4096 bytes): Tests TLB behavior and page boundary crossing
 */
#include "pattern_benchmark/pattern_benchmark.h"
#include "pattern_benchmark/pattern_work_plan.h"
#include "utils/benchmark.h"
#include "core/memory/buffer_manager.h"
#include "core/config/config.h"
#include "core/config/constants.h"
#include "warmup/warmup.h"
#include <iostream>
#include <atomic>
#include <cstdlib>
#include <algorithm>
#include <limits>

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

  const PatternWorkPlan plan =
      build_strided_pattern_work_plan(config.buffer_size, stride, PATTERN_ACCESS_SIZE_BYTES,
                                      config.num_threads, config.iterations,
                                      PATTERN_STRIDED_MIN_TOUCHED_BYTES);
  if (plan.status == PatternMeasurementStatus::Skipped) {
    return EXIT_SUCCESS;
  }
  if (plan.status != PatternMeasurementStatus::Measured) {
    return EXIT_FAILURE;
  }

  const int effective_iterations = static_cast<int>(plan.passes);
  
  // Execute read benchmark
  show_progress();
  std::atomic<uint64_t> checksum{0};
  warmup_read_strided(buffers.src_buffer(), config.buffer_size, stride,
                      plan.effective_threads, checksum);
  double read_time = run_pattern_read_strided_test(buffers.src_buffer(), config.buffer_size,
                                                   stride, effective_iterations, checksum, timer,
                                                   plan.effective_threads);
  read_bw = calculate_bandwidth(plan.payload_bytes_per_pass, effective_iterations, read_time);

  // Execute write benchmark
  show_progress();
  warmup_write_strided(buffers.dst_buffer(), config.buffer_size, stride, plan.effective_threads);
  double write_time = run_pattern_write_strided_test(buffers.dst_buffer(), config.buffer_size,
                                                     stride, effective_iterations, timer,
                                                     plan.effective_threads);
  write_bw = calculate_bandwidth(plan.payload_bytes_per_pass, effective_iterations, write_time);

  // Execute copy benchmark
  show_progress();
  warmup_copy_strided(buffers.dst_buffer(), buffers.src_buffer(), config.buffer_size, stride,
                      plan.effective_threads);
  double copy_time = run_pattern_copy_strided_test(buffers.dst_buffer(), buffers.src_buffer(),
                                                   config.buffer_size, stride, effective_iterations,
                                                   timer, plan.effective_threads);
  if (plan.payload_bytes_per_pass >
      std::numeric_limits<size_t>::max() / Constants::COPY_OPERATION_MULTIPLIER) {
    return EXIT_FAILURE;
  }
  copy_bw = calculate_bandwidth(plan.payload_bytes_per_pass * Constants::COPY_OPERATION_MULTIPLIER,
                                effective_iterations, copy_time);
  
  return EXIT_SUCCESS;
}

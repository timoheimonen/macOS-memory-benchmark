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
 * @file pattern_coordinator.cpp
 * @brief Main coordinator for pattern benchmark execution
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 *
 * This file provides the main public API function for running pattern
 * benchmarks. It orchestrates the execution of all pattern types (sequential,
 * strided, random) within a single benchmark loop and generates random
 * indices for random access patterns.
 *
 * Primary responsibilities:
 * - Coordinate execution of all pattern types in sequence
 * - Generate random access indices
 * - Create and manage high-resolution timer
 * - Handle errors and gracefully skip patterns when buffer constraints prevent execution
 */
#include "pattern_benchmark/pattern_benchmark.h"
#include "pattern_benchmark/pattern_work_plan.h"
#include "utils/benchmark.h"
#include "core/memory/buffer_manager.h"
#include "core/config/config.h"
#include "core/config/constants.h"
#include "core/signal/signal_handler.h"
#include "output/console/messages/messages_api.h"
#include <array>
#include <iostream>
#include <vector>
#include <atomic>
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
double run_pattern_read_strided_test(void* buffer, size_t size, size_t stride, int iterations,
                                     std::atomic<uint64_t>& checksum, HighResTimer& timer,
                                     int num_threads);
double run_pattern_write_strided_test(void* buffer, size_t size, size_t stride, int iterations,
                                      HighResTimer& timer, int num_threads);
double run_pattern_copy_strided_test(void* dst, void* src, size_t size, size_t stride, int iterations,
                                     HighResTimer& timer, int num_threads);
// Forward declarations from validation.cpp
bool validate_stride(size_t stride, size_t buffer_size);
bool validate_random_indices(const std::vector<size_t>& indices, size_t buffer_size);

// Forward declarations from execution_utils.cpp
double calculate_bandwidth(size_t data_size, int iterations, double elapsed_time_ns);
size_t calculate_num_random_accesses(size_t buffer_size);

// Forward declarations from execution_strided.cpp
int run_strided_pattern_benchmarks(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
                                   size_t stride, PatternResults& results,
                                   HighResTimer& timer);

// Forward declarations from execution_patterns.cpp
void run_forward_pattern_benchmarks(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
                                    PatternResults& results, HighResTimer& timer);
void run_reverse_pattern_benchmarks(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
                                   PatternResults& results, HighResTimer& timer);
int run_random_pattern_benchmarks(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
                                   const std::vector<size_t>& random_indices,
                                   const std::vector<PatternRandomWorkerIndices>& worker_indices,
                                   PatternResults& results, HighResTimer& timer);

// ============================================================================
// Public API Functions
// ============================================================================

std::array<PatternKind, static_cast<size_t>(PatternKind::Count)>
build_pattern_execution_order(size_t loop_index) {
  constexpr std::array<PatternKind, static_cast<size_t>(PatternKind::Count)>
      base_order = {PatternKind::SequentialForward,
                    PatternKind::SequentialReverse,
                    PatternKind::Strided64,
                    PatternKind::Strided4096,
                    PatternKind::Strided16384,
                    PatternKind::Strided2MiB,
                    PatternKind::Random};
  std::array<PatternKind, static_cast<size_t>(PatternKind::Count)> order{};
  const size_t rotation = loop_index % base_order.size();
  for (size_t position = 0; position < order.size(); ++position) {
    order[position] = base_order[(position + rotation) % base_order.size()];
  }
  return order;
}

int run_pattern_benchmarks(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
                           PatternResults& results, size_t loop_index) {
  using namespace Constants;

  const auto execution_order = build_pattern_execution_order(loop_index);
  std::array<size_t, static_cast<size_t>(PatternKind::Count)> order_positions{};
  for (size_t position = 0; position < execution_order.size(); ++position) {
    order_positions[static_cast<size_t>(execution_order[position])] = position;
  }

  for (size_t kind_index = 0;
       kind_index < static_cast<size_t>(PatternKind::Count); ++kind_index) {
    for (size_t operation_index = 0;
         operation_index < static_cast<size_t>(PatternOperation::Count);
         ++operation_index) {
      PatternMeasurement& measurement = get_pattern_measurement(
          results, static_cast<PatternKind>(kind_index),
          static_cast<PatternOperation>(operation_index));
      measurement.status = PatternMeasurementStatus::Interrupted;
      measurement.status_reason = Messages::pattern_reason_measurement_not_completed();
      measurement.bandwidth_gb_s.reset();
      measurement.benchmark_loop_index = loop_index;
      measurement.pattern_order_index = order_positions[kind_index];
    }
  }

  auto timer_opt = HighResTimer::create();
  if (!timer_opt) {
    for (PatternMeasurement& measurement : results.measurements) {
      measurement.status = PatternMeasurementStatus::Invalid;
      measurement.status_reason = Messages::pattern_reason_timer_creation_failed();
    }
    std::cerr << Messages::error_prefix()
              << Messages::pattern_reason_timer_creation_failed() << std::endl;
    return EXIT_FAILURE;
  }
  auto& timer = *timer_opt;

  // Calculate number of accesses for random pattern
  size_t num_random_accesses = calculate_num_random_accesses(config.buffer_size);
  
  // Generate random indices once
  std::vector<size_t> random_indices =
      generate_random_indices(config.buffer_size, num_random_accesses, config.pattern_seed);
  std::vector<PatternRandomWorkerIndices> random_worker_indices = build_random_worker_indices(
      config.buffer_size, PATTERN_ACCESS_SIZE_BYTES, config.num_threads, random_indices);
  
  for (size_t position = 0; position < execution_order.size(); ++position) {
    const PatternKind kind = execution_order[position];
    int status = EXIT_SUCCESS;
    switch (kind) {
      case PatternKind::SequentialForward:
        run_forward_pattern_benchmarks(buffers, config, results, timer);
        break;
      case PatternKind::SequentialReverse:
        run_reverse_pattern_benchmarks(buffers, config, results, timer);
        break;
      case PatternKind::Strided64:
        status = run_strided_pattern_benchmarks(
            buffers, config, PATTERN_STRIDE_CACHE_LINE, results, timer);
        break;
      case PatternKind::Strided4096:
        status = run_strided_pattern_benchmarks(
            buffers, config, PATTERN_STRIDE_PAGE, results, timer);
        break;
      case PatternKind::Strided16384:
        status = run_strided_pattern_benchmarks(
            buffers, config, PATTERN_STRIDE_PAGE_16K, results, timer);
        break;
      case PatternKind::Strided2MiB:
        status = run_strided_pattern_benchmarks(
            buffers, config, PATTERN_STRIDE_SUPERPAGE_2MB, results, timer);
        break;
      case PatternKind::Random:
        status = run_random_pattern_benchmarks(
            buffers, config, random_indices, random_worker_indices, results, timer);
        break;
      case PatternKind::Count:
        status = EXIT_FAILURE;
        break;
    }
    for (PatternOperation operation : {PatternOperation::Read,
                                       PatternOperation::Write,
                                       PatternOperation::Copy}) {
      PatternMeasurement& measurement =
          get_pattern_measurement(results, kind, operation);
      measurement.benchmark_loop_index = loop_index;
      measurement.pattern_order_index = position;
    }
    if (status != EXIT_SUCCESS) return status;
    if (signal_received()) return EXIT_SUCCESS;
  }
  
  return EXIT_SUCCESS;
}

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
#include "pattern_benchmark/pattern_work_plan.h"
#include "utils/benchmark.h"
#include "core/memory/buffer_manager.h"
#include "core/config/config.h"
#include "core/config/constants.h"
#include "warmup/warmup.h"
#include <atomic>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <limits>

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
double run_pattern_read_random_test(
    void* buffer, const std::vector<PatternRandomWorkerIndices>& worker_indices, int iterations,
    std::atomic<uint64_t>& checksum, HighResTimer& timer, int num_threads, size_t buffer_size);
double run_pattern_write_random_test(
    void* buffer, const std::vector<PatternRandomWorkerIndices>& worker_indices, int iterations,
    HighResTimer& timer, int num_threads, size_t buffer_size);
double run_pattern_copy_random_test(
    void* dst, void* src, const std::vector<PatternRandomWorkerIndices>& worker_indices,
    int iterations, HighResTimer& timer, int num_threads, size_t buffer_size);

// Forward declarations from validation.cpp
bool validate_random_indices(const std::vector<size_t>& indices, size_t buffer_size);

// Forward declarations from execution_utils.cpp
double calculate_bandwidth(size_t data_size, int iterations, double elapsed_time_ns);

namespace {

template <typename PilotRunner>
int resolve_pattern_passes(const BenchmarkConfig& config, size_t payload_bytes_per_pass,
                           PilotRunner pilot_runner) {
  using namespace Constants;
  const size_t pilot_passes = calculate_pattern_pilot_passes(
      payload_bytes_per_pass, PATTERN_CALIBRATION_MIN_PILOT_BYTES,
      PATTERN_CALIBRATION_MAX_PASSES);
  if (pilot_passes == 0 || pilot_passes > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return 0;
  }

  const double pilot_duration = pilot_runner(static_cast<int>(pilot_passes));
  if (config.user_specified_iterations) {
    return config.iterations;
  }

  const size_t calibrated_passes = calculate_pattern_calibrated_passes(
      pilot_duration, pilot_passes, PATTERN_CALIBRATION_TARGET_SECONDS, 1,
      PATTERN_CALIBRATION_MAX_PASSES);
  if (calibrated_passes == 0 ||
      calibrated_passes > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return 0;
  }
  return static_cast<int>(calibrated_passes);
}

}  // namespace

// ============================================================================
// Pattern Benchmark Execution Functions
// ============================================================================

// Run forward pattern benchmarks (baseline sequential access)
void run_forward_pattern_benchmarks(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
                                    PatternResults& results, HighResTimer& timer) {
  show_progress();
  std::atomic<uint64_t> checksum{0};
  warmup_read(buffers.src_buffer(), config.buffer_size, config.num_threads, checksum);
  auto run_read = [&](int passes) {
    return run_pattern_read_test(buffers.src_buffer(), config.buffer_size, passes,
                                 memory_read_loop_asm, checksum, timer,
                                 config.num_threads);
  };
  const int read_passes = resolve_pattern_passes(config, config.buffer_size, run_read);
  const double read_time = run_read(read_passes);
  results.forward_read_bw = calculate_bandwidth(config.buffer_size, read_passes, read_time);
  
  show_progress();
  warmup_write(buffers.dst_buffer(), config.buffer_size, config.num_threads);
  auto run_write = [&](int passes) {
    return run_write_test(buffers.dst_buffer(), config.buffer_size, passes,
                          config.num_threads, timer);
  };
  const int write_passes = resolve_pattern_passes(config, config.buffer_size, run_write);
  const double write_time = run_write(write_passes);
  results.forward_write_bw = calculate_bandwidth(config.buffer_size, write_passes, write_time);
  
  show_progress();
  warmup_copy(buffers.dst_buffer(), buffers.src_buffer(), config.buffer_size, config.num_threads);
  auto run_copy = [&](int passes) {
    return run_copy_test(buffers.dst_buffer(), buffers.src_buffer(), config.buffer_size,
                         passes, config.num_threads, timer);
  };
  const int copy_passes = resolve_pattern_passes(
      config, config.buffer_size * Constants::COPY_OPERATION_MULTIPLIER, run_copy);
  const double copy_time = run_copy(copy_passes);
  results.forward_copy_bw = calculate_bandwidth(config.buffer_size * Constants::COPY_OPERATION_MULTIPLIER, 
                                                copy_passes, copy_time);
}

// Run reverse pattern benchmarks (backward sequential access)
void run_reverse_pattern_benchmarks(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
                                   PatternResults& results, HighResTimer& timer) {
  show_progress();
  std::atomic<uint64_t> checksum{0};
  warmup_read(buffers.src_buffer(), config.buffer_size, config.num_threads, checksum);
  auto run_read = [&](int passes) {
    return run_pattern_read_test(buffers.src_buffer(), config.buffer_size, passes,
                                 memory_read_reverse_loop_asm, checksum, timer,
                                 config.num_threads);
  };
  const int read_passes = resolve_pattern_passes(config, config.buffer_size, run_read);
  const double read_time = run_read(read_passes);
  results.reverse_read_bw = calculate_bandwidth(config.buffer_size, read_passes, read_time);

  show_progress();
  warmup_write(buffers.dst_buffer(), config.buffer_size, config.num_threads);
  auto run_write = [&](int passes) {
    return run_pattern_write_test(buffers.dst_buffer(), config.buffer_size, passes,
                                  memory_write_reverse_loop_asm, timer, config.num_threads);
  };
  const int write_passes = resolve_pattern_passes(config, config.buffer_size, run_write);
  const double write_time = run_write(write_passes);
  results.reverse_write_bw = calculate_bandwidth(config.buffer_size, write_passes, write_time);

  show_progress();
  warmup_copy(buffers.dst_buffer(), buffers.src_buffer(), config.buffer_size, config.num_threads);
  auto run_copy = [&](int passes) {
    return run_pattern_copy_test(buffers.dst_buffer(), buffers.src_buffer(), config.buffer_size,
                                 passes, memory_copy_reverse_loop_asm, timer,
                                 config.num_threads);
  };
  const int copy_passes = resolve_pattern_passes(
      config, config.buffer_size * Constants::COPY_OPERATION_MULTIPLIER, run_copy);
  const double copy_time = run_copy(copy_passes);
  results.reverse_copy_bw = calculate_bandwidth(config.buffer_size * Constants::COPY_OPERATION_MULTIPLIER,
                                                copy_passes, copy_time);
}

// Run random pattern benchmarks (uniform random access)
// Returns EXIT_SUCCESS on success, EXIT_FAILURE on error, or skips pattern if buffer too small
int run_random_pattern_benchmarks(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
                                   const std::vector<size_t>& random_indices,
                                   const std::vector<PatternRandomWorkerIndices>& worker_indices,
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

  size_t num_accesses = 0;
  for (const PatternRandomWorkerIndices& worker : worker_indices) {
    if (worker.indices.size() > std::numeric_limits<size_t>::max() - num_accesses) {
      return EXIT_FAILURE;
    }
    num_accesses += worker.indices.size();
  }
  if (num_accesses == 0 ||
      num_accesses > std::numeric_limits<size_t>::max() / PATTERN_ACCESS_SIZE_BYTES) {
    return EXIT_FAILURE;
  }
  
  // Execute read benchmark
  show_progress();
  std::atomic<uint64_t> checksum{0};
  warmup_read_random(buffers.src_buffer(), random_indices, config.num_threads, checksum);
  auto run_read = [&](int passes) {
    return run_pattern_read_random_test(buffers.src_buffer(), worker_indices, passes,
                                        checksum, timer, config.num_threads,
                                        config.buffer_size);
  };
  const size_t payload_bytes_per_pass = num_accesses * PATTERN_ACCESS_SIZE_BYTES;
  const int read_passes = resolve_pattern_passes(config, payload_bytes_per_pass, run_read);
  const double read_time = run_read(read_passes);
  // For random, we use num_accesses * PATTERN_ACCESS_SIZE_BYTES instead of buffer_size
  results.random_read_bw = calculate_bandwidth(payload_bytes_per_pass, read_passes, read_time);

  // Execute write benchmark
  show_progress();
  warmup_write_random(buffers.dst_buffer(), random_indices, config.num_threads);
  auto run_write = [&](int passes) {
    return run_pattern_write_random_test(buffers.dst_buffer(), worker_indices, passes,
                                         timer, config.num_threads, config.buffer_size);
  };
  const int write_passes = resolve_pattern_passes(config, payload_bytes_per_pass, run_write);
  const double write_time = run_write(write_passes);
  results.random_write_bw = calculate_bandwidth(payload_bytes_per_pass, write_passes, write_time);

  // Execute copy benchmark
  show_progress();
  warmup_copy_random(buffers.dst_buffer(), buffers.src_buffer(), random_indices,
                     config.num_threads);
  auto run_copy = [&](int passes) {
    return run_pattern_copy_random_test(buffers.dst_buffer(), buffers.src_buffer(),
                                        worker_indices, passes, timer, config.num_threads,
                                        config.buffer_size);
  };
  const size_t copy_payload_bytes_per_pass =
      payload_bytes_per_pass * Constants::COPY_OPERATION_MULTIPLIER;
  const int copy_passes = resolve_pattern_passes(config, copy_payload_bytes_per_pass, run_copy);
  const double copy_time = run_copy(copy_passes);
  results.random_copy_bw =
      calculate_bandwidth(copy_payload_bytes_per_pass, copy_passes, copy_time);
  
  return EXIT_SUCCESS;
}

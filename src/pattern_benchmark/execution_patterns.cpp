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
#include "core/system/page_size.h"
#include "output/console/messages/messages_api.h"
#include "utils/numeric_utils.h"
#include "warmup/warmup.h"
#include <atomic>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <utility>

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
double run_pattern_read_random_test(void* buffer, const std::vector<PatternRandomWorkerIndices>& worker_indices,
                                    int iterations, std::atomic<uint64_t>& checksum, HighResTimer& timer);
double run_pattern_write_random_test(void* buffer, const std::vector<PatternRandomWorkerIndices>& worker_indices,
                                     int iterations, HighResTimer& timer);
double run_pattern_copy_random_test(void* dst, void* src, const std::vector<PatternRandomWorkerIndices>& worker_indices,
                                    int iterations, HighResTimer& timer);

// Forward declarations from execution_utils.cpp
double calculate_bandwidth(size_t data_size, int iterations, double elapsed_time_ns);

namespace {

struct PatternCalibrationDecision {
  int passes = 0;
  double pilot_elapsed_seconds = 0.0;
  bool automatic = false;
};

template <typename PilotRunner>
PatternCalibrationDecision resolve_pattern_passes(
    const BenchmarkConfig& config, size_t payload_bytes_per_pass,
    PilotRunner pilot_runner) {
  using namespace Constants;
  PatternCalibrationDecision decision;
  decision.automatic = !config.user_specified_iterations;
  if (config.user_specified_iterations) {
    decision.passes = config.iterations;
    return decision;
  }
  const size_t pilot_passes = calculate_pattern_pilot_passes(
      payload_bytes_per_pass, PATTERN_CALIBRATION_MIN_PILOT_BYTES,
      PATTERN_CALIBRATION_MAX_PASSES);
  if (pilot_passes == 0 || pilot_passes > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return decision;
  }

  decision.pilot_elapsed_seconds = pilot_runner(static_cast<int>(pilot_passes));
  const size_t calibrated_passes = calculate_pattern_calibrated_passes(
      decision.pilot_elapsed_seconds, pilot_passes,
      PATTERN_CALIBRATION_TARGET_SECONDS, 1,
      PATTERN_CALIBRATION_MAX_PASSES);
  if (calibrated_passes == 0 ||
      calibrated_passes > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return decision;
  }
  decision.passes = static_cast<int>(calibrated_passes);
  return decision;
}

template <typename Runner>
double run_pattern_sample(const BenchmarkConfig& config, Runner runner,
                          PatternCalibrationDecision& decision) {
  using namespace Constants;
  double elapsed_seconds = runner(decision.passes);
  for (size_t correction = 0;
       decision.automatic && correction < PATTERN_CALIBRATION_MAX_CORRECTIONS;
       ++correction) {
    if (elapsed_seconds <= 0.0 || !std::isfinite(elapsed_seconds) ||
        (elapsed_seconds >= PATTERN_CALIBRATION_MIN_SECONDS &&
         elapsed_seconds <= PATTERN_CALIBRATION_MAX_SECONDS)) {
      break;
    }
    const size_t corrected_passes = calculate_pattern_calibrated_passes(
        elapsed_seconds, static_cast<size_t>(decision.passes),
        PATTERN_CALIBRATION_TARGET_SECONDS, 1, PATTERN_CALIBRATION_MAX_PASSES);
    if (corrected_passes == 0 ||
        corrected_passes > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        corrected_passes == static_cast<size_t>(decision.passes)) {
      break;
    }
    decision.passes = static_cast<int>(corrected_passes);
    elapsed_seconds = runner(decision.passes);
  }
  return elapsed_seconds;
}

PatternMeasurement build_pattern_measurement(
    const BenchmarkConfig& config, double bandwidth_gb_s, double elapsed_seconds,
    const PatternCalibrationDecision& calibration, size_t payload_bytes_per_pass,
    size_t accesses_per_pass, size_t distinct_address_count,
    size_t logical_working_set_bytes, size_t stride_bytes = 0,
    bool has_seed = false) {
  PatternMeasurement measurement;
  measurement.access_size_bytes = Constants::PATTERN_ACCESS_SIZE_BYTES;
  measurement.stride_bytes = stride_bytes;
  measurement.requested_threads = config.num_threads;
  measurement.effective_threads = config.num_threads;
  measurement.accesses_per_pass = accesses_per_pass;
  measurement.min_accesses_per_pass = accesses_per_pass;
  measurement.max_accesses_per_pass = accesses_per_pass;
  measurement.passes = calibration.passes > 0
                           ? static_cast<size_t>(calibration.passes)
                           : 0;
  measurement.distinct_address_count = distinct_address_count;
  measurement.logical_working_set_bytes = logical_working_set_bytes;
  measurement.elapsed_seconds = elapsed_seconds;
  measurement.pilot_elapsed_seconds = calibration.pilot_elapsed_seconds;
  measurement.automatic_calibration = calibration.automatic;
  measurement.native_page_size_bytes = get_system_page_size_bytes();
  measurement.stride_equals_native_page_size =
      stride_bytes != 0 && stride_bytes == measurement.native_page_size_bytes;
  measurement.has_seed = has_seed;
  measurement.seed = has_seed ? config.pattern_seed : 0;

  if (measurement.passes == 0 || elapsed_seconds <= 0.0 ||
      !std::isfinite(elapsed_seconds) || !std::isfinite(bandwidth_gb_s) ||
      bandwidth_gb_s <= 0.0 ||
      !NumericUtils::checked_multiply(accesses_per_pass, measurement.passes,
                                      measurement.total_accesses) ||
      !NumericUtils::checked_multiply(payload_bytes_per_pass,
                                      measurement.passes,
                                      measurement.total_payload_bytes)) {
    measurement.status = PatternMeasurementStatus::Invalid;
    measurement.status_reason =
        Messages::pattern_reason_calibration_or_accounting_failed();
    measurement.bandwidth_gb_s.reset();
    return measurement;
  }

  measurement.status = PatternMeasurementStatus::Measured;
  measurement.status_reason.clear();
  measurement.bandwidth_gb_s = bandwidth_gb_s;
  return measurement;
}

void set_triplet_status(PatternResults& results, PatternKind kind,
                        PatternMeasurementStatus status,
                        const std::string& reason, const BenchmarkConfig& config,
                        size_t stride_bytes = 0, bool has_seed = false) {
  for (PatternOperation operation : {PatternOperation::Read, PatternOperation::Write,
                                     PatternOperation::Copy}) {
    PatternMeasurement measurement;
    measurement.status = status;
    measurement.status_reason = reason;
    measurement.access_size_bytes = Constants::PATTERN_ACCESS_SIZE_BYTES;
    measurement.stride_bytes = stride_bytes;
    measurement.requested_threads = config.num_threads;
    measurement.effective_threads = config.num_threads;
    measurement.native_page_size_bytes = get_system_page_size_bytes();
    measurement.stride_equals_native_page_size =
        stride_bytes != 0 && stride_bytes == measurement.native_page_size_bytes;
    measurement.has_seed = has_seed;
    measurement.seed = has_seed ? config.pattern_seed : 0;
    set_pattern_measurement(results, kind, operation, std::move(measurement));
  }
}

}  // namespace

// ============================================================================
// Pattern Benchmark Execution Functions
// ============================================================================

// Run forward pattern benchmarks (baseline sequential access)
void run_forward_pattern_benchmarks(const PatternBuffers& buffers, const BenchmarkConfig& config,
                                    PatternResults& results, HighResTimer& timer) {
  show_progress();
  std::atomic<uint64_t> checksum{0};
  warmup_read(buffers.src_buffer(), config.buffer_size, config.num_threads, checksum);
  auto run_read = [&](int passes) {
    return run_pattern_read_test(buffers.src_buffer(), config.buffer_size, passes,
                                 memory_read_loop_asm, checksum, timer,
                                 config.num_threads);
  };
  PatternCalibrationDecision read_calibration =
      resolve_pattern_passes(config, config.buffer_size, run_read);
  const double read_time = run_pattern_sample(config, run_read, read_calibration);
  const double read_bandwidth =
      calculate_bandwidth(config.buffer_size, read_calibration.passes, read_time);
  set_pattern_measurement(
      results, PatternKind::SequentialForward, PatternOperation::Read,
      build_pattern_measurement(
          config, read_bandwidth, read_time, read_calibration, config.buffer_size,
          (config.buffer_size + Constants::PATTERN_ACCESS_SIZE_BYTES - 1) /
              Constants::PATTERN_ACCESS_SIZE_BYTES,
          (config.buffer_size + Constants::PATTERN_ACCESS_SIZE_BYTES - 1) /
              Constants::PATTERN_ACCESS_SIZE_BYTES,
          config.buffer_size));
  
  show_progress();
  warmup_write(buffers.dst_buffer(), config.buffer_size, config.num_threads);
  auto run_write = [&](int passes) {
    return run_write_test(buffers.dst_buffer(), config.buffer_size, passes,
                          config.num_threads, timer);
  };
  PatternCalibrationDecision write_calibration =
      resolve_pattern_passes(config, config.buffer_size, run_write);
  const double write_time = run_pattern_sample(config, run_write, write_calibration);
  const double write_bandwidth =
      calculate_bandwidth(config.buffer_size, write_calibration.passes, write_time);
  set_pattern_measurement(
      results, PatternKind::SequentialForward, PatternOperation::Write,
      build_pattern_measurement(
          config, write_bandwidth, write_time, write_calibration, config.buffer_size,
          (config.buffer_size + Constants::PATTERN_ACCESS_SIZE_BYTES - 1) /
              Constants::PATTERN_ACCESS_SIZE_BYTES,
          (config.buffer_size + Constants::PATTERN_ACCESS_SIZE_BYTES - 1) /
              Constants::PATTERN_ACCESS_SIZE_BYTES,
          config.buffer_size));
  
  show_progress();
  warmup_copy(buffers.dst_buffer(), buffers.src_buffer(), config.buffer_size, config.num_threads);
  auto run_copy = [&](int passes) {
    return run_copy_test(buffers.dst_buffer(), buffers.src_buffer(), config.buffer_size,
                         passes, config.num_threads, timer);
  };
  PatternCalibrationDecision copy_calibration = resolve_pattern_passes(
      config, config.buffer_size * Constants::COPY_OPERATION_MULTIPLIER, run_copy);
  const double copy_time = run_pattern_sample(config, run_copy, copy_calibration);
  const size_t copy_payload_per_pass =
      config.buffer_size * Constants::COPY_OPERATION_MULTIPLIER;
  const double copy_bandwidth = calculate_bandwidth(
      copy_payload_per_pass, copy_calibration.passes, copy_time);
  set_pattern_measurement(
      results, PatternKind::SequentialForward, PatternOperation::Copy,
      build_pattern_measurement(
          config, copy_bandwidth, copy_time, copy_calibration,
          copy_payload_per_pass,
          (config.buffer_size + Constants::PATTERN_ACCESS_SIZE_BYTES - 1) /
              Constants::PATTERN_ACCESS_SIZE_BYTES,
          (config.buffer_size + Constants::PATTERN_ACCESS_SIZE_BYTES - 1) /
              Constants::PATTERN_ACCESS_SIZE_BYTES,
          config.buffer_size));
}

// Run reverse pattern benchmarks (backward sequential access)
void run_reverse_pattern_benchmarks(const PatternBuffers& buffers, const BenchmarkConfig& config,
                                   PatternResults& results, HighResTimer& timer) {
  show_progress();
  std::atomic<uint64_t> checksum{0};
  warmup_read(buffers.src_buffer(), config.buffer_size, config.num_threads, checksum);
  auto run_read = [&](int passes) {
    return run_pattern_read_test(buffers.src_buffer(), config.buffer_size, passes,
                                 memory_read_reverse_loop_asm, checksum, timer,
                                 config.num_threads);
  };
  PatternCalibrationDecision read_calibration =
      resolve_pattern_passes(config, config.buffer_size, run_read);
  const double read_time = run_pattern_sample(config, run_read, read_calibration);
  const double read_bandwidth =
      calculate_bandwidth(config.buffer_size, read_calibration.passes, read_time);
  set_pattern_measurement(
      results, PatternKind::SequentialReverse, PatternOperation::Read,
      build_pattern_measurement(
          config, read_bandwidth, read_time, read_calibration, config.buffer_size,
          (config.buffer_size + Constants::PATTERN_ACCESS_SIZE_BYTES - 1) /
              Constants::PATTERN_ACCESS_SIZE_BYTES,
          (config.buffer_size + Constants::PATTERN_ACCESS_SIZE_BYTES - 1) /
              Constants::PATTERN_ACCESS_SIZE_BYTES,
          config.buffer_size));

  show_progress();
  warmup_write(buffers.dst_buffer(), config.buffer_size, config.num_threads);
  auto run_write = [&](int passes) {
    return run_pattern_write_test(buffers.dst_buffer(), config.buffer_size, passes,
                                  memory_write_reverse_loop_asm, timer, config.num_threads);
  };
  PatternCalibrationDecision write_calibration =
      resolve_pattern_passes(config, config.buffer_size, run_write);
  const double write_time = run_pattern_sample(config, run_write, write_calibration);
  const double write_bandwidth =
      calculate_bandwidth(config.buffer_size, write_calibration.passes, write_time);
  set_pattern_measurement(
      results, PatternKind::SequentialReverse, PatternOperation::Write,
      build_pattern_measurement(
          config, write_bandwidth, write_time, write_calibration, config.buffer_size,
          (config.buffer_size + Constants::PATTERN_ACCESS_SIZE_BYTES - 1) /
              Constants::PATTERN_ACCESS_SIZE_BYTES,
          (config.buffer_size + Constants::PATTERN_ACCESS_SIZE_BYTES - 1) /
              Constants::PATTERN_ACCESS_SIZE_BYTES,
          config.buffer_size));

  show_progress();
  warmup_copy(buffers.dst_buffer(), buffers.src_buffer(), config.buffer_size, config.num_threads);
  auto run_copy = [&](int passes) {
    return run_pattern_copy_test(buffers.dst_buffer(), buffers.src_buffer(), config.buffer_size,
                                 passes, memory_copy_reverse_loop_asm, timer,
                                 config.num_threads);
  };
  PatternCalibrationDecision copy_calibration = resolve_pattern_passes(
      config, config.buffer_size * Constants::COPY_OPERATION_MULTIPLIER, run_copy);
  const double copy_time = run_pattern_sample(config, run_copy, copy_calibration);
  const size_t copy_payload_per_pass =
      config.buffer_size * Constants::COPY_OPERATION_MULTIPLIER;
  const double copy_bandwidth = calculate_bandwidth(
      copy_payload_per_pass, copy_calibration.passes, copy_time);
  set_pattern_measurement(
      results, PatternKind::SequentialReverse, PatternOperation::Copy,
      build_pattern_measurement(
          config, copy_bandwidth, copy_time, copy_calibration,
          copy_payload_per_pass,
          (config.buffer_size + Constants::PATTERN_ACCESS_SIZE_BYTES - 1) /
              Constants::PATTERN_ACCESS_SIZE_BYTES,
          (config.buffer_size + Constants::PATTERN_ACCESS_SIZE_BYTES - 1) /
              Constants::PATTERN_ACCESS_SIZE_BYTES,
          config.buffer_size));
}

// Run random pattern benchmarks (uniform random access)
// Returns EXIT_SUCCESS on success, EXIT_FAILURE on error, or skips pattern if buffer too small
int run_random_pattern_benchmarks(const PatternBuffers& buffers, const BenchmarkConfig& config,
                                   const std::vector<size_t>& random_indices,
                                   const std::vector<PatternRandomWorkerIndices>& worker_indices,
                                   PatternResults& results, HighResTimer& timer) {
  using namespace Constants;
  
  // Validate indices - if validation fails due to buffer size, skip pattern gracefully
  if (!validate_random_indices(random_indices, config.buffer_size)) {
    // No valid indices or buffer too small - skip pattern (not an error)
    set_triplet_status(results, PatternKind::Random,
                       PatternMeasurementStatus::Skipped,
                       Messages::pattern_reason_no_valid_random_workload(),
                       config, 0, true);
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
  warmup_read_random(buffers.src_buffer(), worker_indices, checksum);
  auto run_read = [&](int passes) {
    return run_pattern_read_random_test(buffers.src_buffer(), worker_indices, passes, checksum, timer);
  };
  const size_t payload_bytes_per_pass = num_accesses * PATTERN_ACCESS_SIZE_BYTES;
  PatternCalibrationDecision read_calibration =
      resolve_pattern_passes(config, payload_bytes_per_pass, run_read);
  const double read_time = run_pattern_sample(config, run_read, read_calibration);
  // For random, we use num_accesses * PATTERN_ACCESS_SIZE_BYTES instead of buffer_size
  const double read_bandwidth = calculate_bandwidth(
      payload_bytes_per_pass, read_calibration.passes, read_time);
  const auto [minimum_index, maximum_index] =
      std::minmax_element(random_indices.begin(), random_indices.end());
  const size_t logical_working_set_bytes =
      *maximum_index - *minimum_index + PATTERN_ACCESS_SIZE_BYTES;
  set_pattern_measurement(
      results, PatternKind::Random, PatternOperation::Read,
      build_pattern_measurement(config, read_bandwidth, read_time, read_calibration,
                                payload_bytes_per_pass, num_accesses, num_accesses,
                                logical_working_set_bytes, 0, true));

  // Execute write benchmark
  show_progress();
  warmup_write_random(buffers.dst_buffer(), worker_indices);
  auto run_write = [&](int passes) {
    return run_pattern_write_random_test(buffers.dst_buffer(), worker_indices, passes, timer);
  };
  PatternCalibrationDecision write_calibration =
      resolve_pattern_passes(config, payload_bytes_per_pass, run_write);
  const double write_time = run_pattern_sample(config, run_write, write_calibration);
  const double write_bandwidth = calculate_bandwidth(
      payload_bytes_per_pass, write_calibration.passes, write_time);
  set_pattern_measurement(
      results, PatternKind::Random, PatternOperation::Write,
      build_pattern_measurement(config, write_bandwidth, write_time,
                                write_calibration, payload_bytes_per_pass,
                                num_accesses, num_accesses,
                                logical_working_set_bytes, 0, true));

  // Execute copy benchmark
  show_progress();
  warmup_copy_random(buffers.dst_buffer(), buffers.src_buffer(), worker_indices);
  auto run_copy = [&](int passes) {
    return run_pattern_copy_random_test(buffers.dst_buffer(), buffers.src_buffer(), worker_indices, passes, timer);
  };
  const size_t copy_payload_bytes_per_pass =
      payload_bytes_per_pass * Constants::COPY_OPERATION_MULTIPLIER;
  PatternCalibrationDecision copy_calibration = resolve_pattern_passes(
      config, copy_payload_bytes_per_pass, run_copy);
  const double copy_time = run_pattern_sample(config, run_copy, copy_calibration);
  const double copy_bandwidth = calculate_bandwidth(
      copy_payload_bytes_per_pass, copy_calibration.passes, copy_time);
  set_pattern_measurement(
      results, PatternKind::Random, PatternOperation::Copy,
      build_pattern_measurement(config, copy_bandwidth, copy_time,
                                copy_calibration, copy_payload_bytes_per_pass,
                                num_accesses, num_accesses,
                                logical_working_set_bytes, 0, true));
  
  return EXIT_SUCCESS;
}

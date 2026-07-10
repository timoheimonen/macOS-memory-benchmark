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
#include "output/console/messages/messages_api.h"
#include "warmup/warmup.h"
#include <iostream>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <unistd.h>
#include <utility>

// Forward declarations from helpers.cpp
double run_pattern_read_strided_test(void* buffer, const PatternWorkPlan& plan, std::atomic<uint64_t>& checksum,
                                     HighResTimer& timer);
double run_pattern_write_strided_test(void* buffer, const PatternWorkPlan& plan, HighResTimer& timer);
double run_pattern_copy_strided_test(void* dst, void* src, const PatternWorkPlan& plan, HighResTimer& timer);

// Forward declarations from execution_utils.cpp
double calculate_bandwidth(size_t data_size, int iterations, double elapsed_time_ns);

namespace {

template <typename PilotRunner>
bool calibrate_strided_plan(const BenchmarkConfig& config,
                            const PatternWorkPlan& pilot_plan,
                            PilotRunner pilot_runner,
                            PatternWorkPlan& measured_plan,
                            double& pilot_elapsed_seconds) {
  using namespace Constants;
  if (config.user_specified_iterations) {
    pilot_elapsed_seconds = 0.0;
    measured_plan = pilot_plan;
    return config.iterations > 0 &&
           set_strided_pattern_passes(measured_plan,
                                      static_cast<size_t>(config.iterations));
  }
  pilot_elapsed_seconds = pilot_runner(pilot_plan);
  const size_t measured_passes = calculate_pattern_calibrated_passes(
      pilot_elapsed_seconds, pilot_plan.passes, PATTERN_CALIBRATION_TARGET_SECONDS, 1,
      PATTERN_CALIBRATION_MAX_PASSES);
  measured_plan = pilot_plan;
  return measured_passes > 0 && set_strided_pattern_passes(measured_plan, measured_passes);
}

template <typename Runner>
double run_strided_sample(const BenchmarkConfig& config, Runner runner,
                          PatternWorkPlan& plan) {
  using namespace Constants;
  double elapsed_seconds = runner(plan);
  for (size_t correction = 0;
       !config.user_specified_iterations &&
       correction < PATTERN_CALIBRATION_MAX_CORRECTIONS;
       ++correction) {
    if (elapsed_seconds <= 0.0 || !std::isfinite(elapsed_seconds) ||
        (elapsed_seconds >= PATTERN_CALIBRATION_MIN_SECONDS &&
         elapsed_seconds <= PATTERN_CALIBRATION_MAX_SECONDS)) {
      break;
    }
    const size_t corrected_passes = calculate_pattern_calibrated_passes(
        elapsed_seconds, plan.passes, PATTERN_CALIBRATION_TARGET_SECONDS, 1,
        PATTERN_CALIBRATION_MAX_PASSES);
    if (corrected_passes == 0 || corrected_passes == plan.passes ||
        !set_strided_pattern_passes(plan, corrected_passes)) {
      break;
    }
    elapsed_seconds = runner(plan);
  }
  return elapsed_seconds;
}

PatternKind pattern_kind_for_stride(size_t stride) {
  using namespace Constants;
  if (stride == PATTERN_STRIDE_CACHE_LINE) return PatternKind::Strided64;
  if (stride == PATTERN_STRIDE_PAGE) return PatternKind::Strided4096;
  if (stride == PATTERN_STRIDE_PAGE_16K) return PatternKind::Strided16384;
  return PatternKind::Strided2MiB;
}

void set_strided_triplet_status(PatternResults& results, PatternKind kind,
                                PatternMeasurementStatus status,
                                const std::string& reason,
                                const BenchmarkConfig& config, size_t stride,
                                int effective_threads) {
  for (PatternOperation operation : {PatternOperation::Read, PatternOperation::Write,
                                     PatternOperation::Copy}) {
    PatternMeasurement measurement;
    measurement.status = status;
    measurement.status_reason = reason;
    measurement.access_size_bytes = Constants::PATTERN_ACCESS_SIZE_BYTES;
    measurement.stride_bytes = stride;
    measurement.phase_period_passes =
        stride % Constants::PATTERN_ACCESS_SIZE_BYTES == 0
            ? stride / Constants::PATTERN_ACCESS_SIZE_BYTES
            : 0;
    measurement.requested_threads = config.num_threads;
    measurement.effective_threads = effective_threads;
    measurement.native_page_size_bytes = static_cast<size_t>(getpagesize());
    measurement.stride_equals_native_page_size =
        stride == measurement.native_page_size_bytes;
    set_pattern_measurement(results, kind, operation, std::move(measurement));
  }
}

PatternMeasurement build_strided_measurement(
    const BenchmarkConfig& config, const PatternWorkPlan& plan,
    double bandwidth_gb_s, double elapsed_seconds, double pilot_elapsed_seconds,
    bool copy_operation) {
  PatternMeasurement measurement;
  measurement.access_size_bytes = plan.access_size_bytes;
  measurement.stride_bytes = plan.stride_bytes;
  measurement.requested_threads = plan.requested_threads;
  measurement.effective_threads = plan.effective_threads;
  measurement.accesses_per_pass = plan.accesses_per_pass;
  measurement.min_accesses_per_pass = plan.min_accesses_per_pass;
  measurement.max_accesses_per_pass = plan.max_accesses_per_pass;
  measurement.passes = plan.passes;
  measurement.total_accesses = plan.total_accesses;
  measurement.total_payload_bytes = plan.total_payload_bytes;
  measurement.distinct_address_count = plan.distinct_address_count;
  measurement.logical_working_set_bytes = plan.logical_working_set_bytes;
  measurement.completed_phase_cycles = plan.completed_phase_cycles;
  measurement.phase_period_passes = plan.phase_period_passes;
  measurement.elapsed_seconds = elapsed_seconds;
  measurement.pilot_elapsed_seconds = pilot_elapsed_seconds;
  measurement.automatic_calibration = !config.user_specified_iterations;
  measurement.native_page_size_bytes = static_cast<size_t>(getpagesize());
  measurement.stride_equals_native_page_size =
      plan.stride_bytes == measurement.native_page_size_bytes;

  if (copy_operation) {
    if (measurement.total_payload_bytes >
        std::numeric_limits<size_t>::max() / Constants::COPY_OPERATION_MULTIPLIER) {
      measurement.status = PatternMeasurementStatus::Invalid;
      measurement.status_reason =
          Messages::pattern_reason_copy_accounting_overflow();
      return measurement;
    }
    measurement.total_payload_bytes *= Constants::COPY_OPERATION_MULTIPLIER;
  }

  if (elapsed_seconds <= 0.0 || !std::isfinite(elapsed_seconds) ||
      bandwidth_gb_s <= 0.0 || !std::isfinite(bandwidth_gb_s)) {
    measurement.status = PatternMeasurementStatus::Invalid;
    measurement.status_reason = Messages::pattern_reason_invalid_strided_timing();
    return measurement;
  }
  measurement.status = PatternMeasurementStatus::Measured;
  measurement.status_reason.clear();
  measurement.bandwidth_gb_s = bandwidth_gb_s;
  return measurement;
}

}  // namespace

// ============================================================================
// Strided Pattern Execution Functions
// ============================================================================

// Run strided pattern benchmarks (access with specified stride)
// Returns EXIT_SUCCESS on success, EXIT_FAILURE on error, or skips pattern if buffer too small
int run_strided_pattern_benchmarks(const BenchmarkBuffers& buffers, const BenchmarkConfig& config,
                                   size_t stride, PatternResults& results,
                                   HighResTimer& timer) {
  using namespace Constants;
  const PatternKind pattern_kind = pattern_kind_for_stride(stride);
  
  // Validate stride - if validation fails due to buffer size, skip pattern gracefully
  if (!validate_stride(stride, config.buffer_size)) {
    // Buffer too small for this stride - skip pattern (not an error)
    set_strided_triplet_status(results, pattern_kind,
                               PatternMeasurementStatus::Skipped,
                               Messages::pattern_reason_stride_transition_unavailable(),
                               config, stride, 0);
    return EXIT_SUCCESS;
  }

  const PatternWorkPlan pilot_plan =
      build_strided_pattern_work_plan(config.buffer_size, stride, PATTERN_ACCESS_SIZE_BYTES,
                                      config.num_threads, 1,
                                      PATTERN_CALIBRATION_MIN_PILOT_BYTES);
  if (pilot_plan.status == PatternMeasurementStatus::Skipped) {
    set_strided_triplet_status(results, pattern_kind,
                               PatternMeasurementStatus::Skipped,
                               pilot_plan.status_reason, config, stride,
                               pilot_plan.effective_threads);
    return EXIT_SUCCESS;
  }
  if (pilot_plan.status != PatternMeasurementStatus::Measured) {
    set_strided_triplet_status(results, pattern_kind,
                               PatternMeasurementStatus::Invalid,
                               pilot_plan.status_reason, config, stride,
                               pilot_plan.effective_threads);
    return EXIT_FAILURE;
  }
  
  // Execute read benchmark
  show_progress();
  std::atomic<uint64_t> checksum{0};
  warmup_read_strided(buffers.src_buffer(), config.buffer_size, stride,
                      pilot_plan.effective_threads, checksum);
  auto run_read = [&](const PatternWorkPlan& plan) {
    return run_pattern_read_strided_test(buffers.src_buffer(), plan, checksum, timer);
  };
  PatternWorkPlan read_plan;
  double read_pilot_time = 0.0;
  if (!calibrate_strided_plan(config, pilot_plan, run_read, read_plan,
                              read_pilot_time)) {
    return EXIT_FAILURE;
  }
  const double read_time = run_strided_sample(config, run_read, read_plan);
  const double read_bw =
      calculate_bandwidth(read_plan.total_payload_bytes, 1, read_time);
  set_pattern_measurement(
      results, pattern_kind, PatternOperation::Read,
      build_strided_measurement(config, read_plan, read_bw, read_time,
                                read_pilot_time, false));

  // Execute write benchmark
  show_progress();
  warmup_write_strided(buffers.dst_buffer(), config.buffer_size, stride,
                       pilot_plan.effective_threads);
  auto run_write = [&](const PatternWorkPlan& plan) {
    return run_pattern_write_strided_test(buffers.dst_buffer(), plan, timer);
  };
  PatternWorkPlan write_plan;
  double write_pilot_time = 0.0;
  if (!calibrate_strided_plan(config, pilot_plan, run_write, write_plan,
                              write_pilot_time)) {
    return EXIT_FAILURE;
  }
  const double write_time = run_strided_sample(config, run_write, write_plan);
  const double write_bw =
      calculate_bandwidth(write_plan.total_payload_bytes, 1, write_time);
  set_pattern_measurement(
      results, pattern_kind, PatternOperation::Write,
      build_strided_measurement(config, write_plan, write_bw, write_time,
                                write_pilot_time, false));

  // Execute copy benchmark
  show_progress();
  warmup_copy_strided(buffers.dst_buffer(), buffers.src_buffer(), config.buffer_size, stride,
                      pilot_plan.effective_threads);
  auto run_copy = [&](const PatternWorkPlan& plan) {
    return run_pattern_copy_strided_test(buffers.dst_buffer(), buffers.src_buffer(), plan, timer);
  };
  PatternWorkPlan copy_plan;
  double copy_pilot_time = 0.0;
  if (!calibrate_strided_plan(config, pilot_plan, run_copy, copy_plan,
                              copy_pilot_time)) {
    return EXIT_FAILURE;
  }
  const double copy_time = run_strided_sample(config, run_copy, copy_plan);
  if (copy_plan.total_payload_bytes >
      std::numeric_limits<size_t>::max() / Constants::COPY_OPERATION_MULTIPLIER) {
    return EXIT_FAILURE;
  }
  const double copy_bw = calculate_bandwidth(
      copy_plan.total_payload_bytes * Constants::COPY_OPERATION_MULTIPLIER, 1, copy_time);
  set_pattern_measurement(
      results, pattern_kind, PatternOperation::Copy,
      build_strided_measurement(config, copy_plan, copy_bw, copy_time,
                                copy_pilot_time, true));
  
  return EXIT_SUCCESS;
}

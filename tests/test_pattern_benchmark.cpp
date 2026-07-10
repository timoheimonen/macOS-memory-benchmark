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
#include <gtest/gtest.h>
#include "asm/asm_functions.h"
#include "benchmark/parallel_test_framework.h"
#include "pattern_benchmark/pattern_benchmark.h"
#include "pattern_benchmark/pattern_work_plan.h"
#include "core/memory/buffer_manager.h"
#include "core/config/config.h"
#include "core/config/constants.h"
#include "output/console/messages/messages_api.h"
#include "output/json/json_output/json_output_api.h"
#include "utils/benchmark.h"  // Declares system_info functions
#include "warmup/warmup.h"
#include "test_config_helpers.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
extern "C" uint64_t verify_pattern_callee_saved_registers_asm(
    uintptr_t function_address, uintptr_t arg0, uintptr_t arg1, uintptr_t arg2,
    uintptr_t arg3, uintptr_t arg4, uintptr_t arg5);

// Test-only AAPCS64 probe. It seeds x19-x29 and the preserved low 64 bits of
// d8-d15, calls a pattern kernel with up to six integer arguments, and returns
// one only when every callee-saved value survives.
__asm__(R"ASM(
.text
.p2align 4
.global _verify_pattern_callee_saved_registers_asm
_verify_pattern_callee_saved_registers_asm:
    stp x29, x30, [sp, #-160]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    stp x27, x28, [sp, #80]
    stp d8, d9, [sp, #96]
    stp d10, d11, [sp, #112]
    stp d12, d13, [sp, #128]
    stp d14, d15, [sp, #144]

    mov x16, x0
    mov x0, x1
    mov x1, x2
    mov x2, x3
    mov x3, x4
    mov x4, x5
    mov x5, x6

    mov x19, #0x1919
    mov x20, #0x2020
    mov x21, #0x2121
    mov x22, #0x2222
    mov x23, #0x2323
    mov x24, #0x2424
    mov x25, #0x2525
    mov x26, #0x2626
    mov x27, #0x2727
    mov x28, #0x2828
    mov x9, #0xd8
    fmov d8, x9
    mov x9, #0xd9
    fmov d9, x9
    mov x9, #0xda
    fmov d10, x9
    mov x9, #0xdb
    fmov d11, x9
    mov x9, #0xdc
    fmov d12, x9
    mov x9, #0xdd
    fmov d13, x9
    mov x9, #0xde
    fmov d14, x9
    mov x9, #0xdf
    fmov d15, x9

    blr x16

    mov x17, #1
    mov x9, sp
    cmp x29, x9
    csel x17, x17, xzr, eq
    mov x9, #0x1919
    cmp x19, x9
    csel x17, x17, xzr, eq
    mov x9, #0x2020
    cmp x20, x9
    csel x17, x17, xzr, eq
    mov x9, #0x2121
    cmp x21, x9
    csel x17, x17, xzr, eq
    mov x9, #0x2222
    cmp x22, x9
    csel x17, x17, xzr, eq
    mov x9, #0x2323
    cmp x23, x9
    csel x17, x17, xzr, eq
    mov x9, #0x2424
    cmp x24, x9
    csel x17, x17, xzr, eq
    mov x9, #0x2525
    cmp x25, x9
    csel x17, x17, xzr, eq
    mov x9, #0x2626
    cmp x26, x9
    csel x17, x17, xzr, eq
    mov x9, #0x2727
    cmp x27, x9
    csel x17, x17, xzr, eq
    mov x9, #0x2828
    cmp x28, x9
    csel x17, x17, xzr, eq
    mov x9, #0xd8
    fmov x10, d8
    cmp x10, x9
    csel x17, x17, xzr, eq
    mov x9, #0xd9
    fmov x10, d9
    cmp x10, x9
    csel x17, x17, xzr, eq
    mov x9, #0xda
    fmov x10, d10
    cmp x10, x9
    csel x17, x17, xzr, eq
    mov x9, #0xdb
    fmov x10, d11
    cmp x10, x9
    csel x17, x17, xzr, eq
    mov x9, #0xdc
    fmov x10, d12
    cmp x10, x9
    csel x17, x17, xzr, eq
    mov x9, #0xdd
    fmov x10, d13
    cmp x10, x9
    csel x17, x17, xzr, eq
    mov x9, #0xde
    fmov x10, d14
    cmp x10, x9
    csel x17, x17, xzr, eq
    mov x9, #0xdf
    fmov x10, d15
    cmp x10, x9
    csel x17, x17, xzr, eq

    ldp d14, d15, [sp, #144]
    ldp d12, d13, [sp, #128]
    ldp d10, d11, [sp, #112]
    ldp d8, d9, [sp, #96]
    ldp x27, x28, [sp, #80]
    ldp x25, x26, [sp, #64]
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #160
    mov x0, x17
    ret
)ASM");

namespace {

BenchmarkConfig make_pattern_config(size_t buffer_size, int iterations, int num_threads = 1) {
  BenchmarkConfig config;
  config.buffer_size = buffer_size;
  config.iterations = iterations;
  config.user_specified_iterations = true;
  initialize_system_info(config);
  config.num_threads = num_threads;
  return config;
}

PatternMeasurement make_test_measurement(
    PatternMeasurementStatus status, const std::string& reason,
    std::optional<double> bandwidth_gb_s = std::nullopt) {
  PatternMeasurement measurement;
  measurement.status = status;
  measurement.status_reason = reason;
  measurement.bandwidth_gb_s = bandwidth_gb_s;
  return measurement;
}

PatternResults make_complete_pattern_loop(double bandwidth = 1.0) {
  PatternResults results;
  for (PatternMeasurement& measurement : results.measurements) {
    measurement.status = PatternMeasurementStatus::Measured;
    measurement.bandwidth_gb_s = bandwidth;
  }
  return results;
}

PatternResults make_skipped_pattern_loop(
    const std::string& reason = "intentional test skip") {
  PatternResults results;
  for (PatternMeasurement& measurement : results.measurements) {
    measurement.status = PatternMeasurementStatus::Skipped;
    measurement.status_reason = reason;
    measurement.bandwidth_gb_s.reset();
  }
  return results;
}

PatternRunnerTestHooks make_pattern_runner_hooks() {
  PatternRunnerTestHooks hooks;
  hooks.allocate_buffers = [](const BenchmarkConfig&, PatternBuffers&) {
    return EXIT_SUCCESS;
  };
  hooks.initialize_buffers = [](const PatternBuffers&, size_t) {
    return EXIT_SUCCESS;
  };
  hooks.stop_requested = []() { return false; };
  return hooks;
}

unsigned char* align_to_cache_line(unsigned char* pointer) {
  const uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
  const uintptr_t aligned =
      (address + Constants::CACHE_LINE_SIZE_BYTES - 1) &
      ~(static_cast<uintptr_t>(Constants::CACHE_LINE_SIZE_BYTES) - 1);
  return reinterpret_cast<unsigned char*>(aligned);
}

::testing::AssertionResult run_pattern_benchmarks_with_fresh_buffers(BenchmarkConfig& config,
                                                                      PatternResults& results) {
  PatternBuffers buffers;
  const ::testing::AssertionResult alloc_init_result =
      allocate_and_initialize_pattern_buffers(config, buffers);
  if (!alloc_init_result) {
    return alloc_init_result;
  }

  const int run_result = run_pattern_benchmarks(buffers, config, results);
  if (run_result != EXIT_SUCCESS) {
    return ::testing::AssertionFailure()
           << "run_pattern_benchmarks(buffers, config, results) failed with code " << run_result;
  }

  return ::testing::AssertionSuccess();
}

void expect_core_pattern_bandwidths_positive(const PatternResults& results) {
  const std::array<PatternKind, 6> core_kinds = {
      PatternKind::SequentialForward, PatternKind::SequentialReverse,
      PatternKind::Strided64, PatternKind::Strided4096,
      PatternKind::Strided16384, PatternKind::Random};
  for (PatternKind kind : core_kinds) {
    for (PatternOperation operation : {PatternOperation::Read,
                                       PatternOperation::Write,
                                       PatternOperation::Copy}) {
      const PatternMeasurement& measurement =
          get_pattern_measurement(results, kind, operation);
      EXPECT_EQ(measurement.status, PatternMeasurementStatus::Measured);
      ASSERT_TRUE(measurement.bandwidth_gb_s.has_value());
      EXPECT_GT(*measurement.bandwidth_gb_s, 0.0);
      EXPECT_GT(measurement.elapsed_seconds, 0.0);
      EXPECT_GT(measurement.total_payload_bytes, 0u);
    }
  }
}

void expect_2mb_pattern_bandwidths_zero(const PatternResults& results) {
  for (PatternOperation operation : {PatternOperation::Read, PatternOperation::Write,
                                     PatternOperation::Copy}) {
    const PatternMeasurement& measurement =
        get_pattern_measurement(results, PatternKind::Strided2MiB, operation);
    EXPECT_EQ(measurement.status, PatternMeasurementStatus::Skipped);
    EXPECT_FALSE(measurement.bandwidth_gb_s.has_value());
    EXPECT_EQ(measurement.status_reason,
              Messages::pattern_reason_stride_transition_unavailable());
  }
}

void expect_2mb_pattern_bandwidths_positive(const PatternResults& results) {
  for (PatternOperation operation : {PatternOperation::Read, PatternOperation::Write,
                                     PatternOperation::Copy}) {
    const PatternMeasurement& measurement =
        get_pattern_measurement(results, PatternKind::Strided2MiB, operation);
    EXPECT_EQ(measurement.status, PatternMeasurementStatus::Measured);
    ASSERT_TRUE(measurement.bandwidth_gb_s.has_value());
    EXPECT_GT(measurement.elapsed_seconds, 0.0);
    EXPECT_GT(measurement.total_payload_bytes, 0u);
    EXPECT_GT(measurement.total_accesses, 0u);
    EXPECT_EQ(measurement.stride_bytes,
              Constants::PATTERN_STRIDE_SUPERPAGE_2MB);
    EXPECT_FALSE(measurement.large_page_backing_verified);
  }
}

}  // namespace

TEST(PatternBenchmarkTest, PatternResultsDefaultValues) {
  PatternResults results;

  EXPECT_EQ(results.status, PatternRunStatus::NotStarted);
  EXPECT_EQ(results.planned_measurements, kPatternMeasurementsPerLoop);
  EXPECT_EQ(results.completed_measurements, 0u);

  for (const PatternMeasurement& measurement : results.measurements) {
    EXPECT_EQ(measurement.status, PatternMeasurementStatus::Invalid);
    EXPECT_FALSE(measurement.bandwidth_gb_s.has_value());
  }
}

TEST(PatternBenchmarkTest, PatternRunStatusStringsAreStable) {
  EXPECT_STREQ(pattern_run_status_to_string(PatternRunStatus::NotStarted),
               "not-started");
  EXPECT_STREQ(pattern_run_status_to_string(PatternRunStatus::Complete),
               "complete");
  EXPECT_STREQ(pattern_run_status_to_string(PatternRunStatus::Partial),
               "partial");
  EXPECT_STREQ(pattern_run_status_to_string(PatternRunStatus::Interrupted),
               "interrupted");
  EXPECT_STREQ(pattern_run_status_to_string(PatternRunStatus::Failed),
               "failed");
}

TEST(PatternBenchmarkTest, LoopSummaryAcceptsMeasuredValuesAndSkippedMeasurements) {
  PatternResults results = make_complete_pattern_loop();
  results.measurements[4].status = PatternMeasurementStatus::Skipped;
  results.measurements[4].status_reason = "unsupported by this buffer";
  results.measurements[4].bandwidth_gb_s.reset();

  const PatternLoopSummary summary = summarize_pattern_loop(results);
  EXPECT_EQ(summary.status, PatternRunStatus::Complete);
  EXPECT_TRUE(summary.status_reason.empty());
  EXPECT_EQ(summary.planned_measurements, 21u);
  EXPECT_EQ(summary.completed_measurements, 21u);
}

TEST(PatternBenchmarkTest, LoopSummaryClassifiesIncompleteInterruptedInvalidAndExecutionFailure) {
  PatternResults incomplete = make_complete_pattern_loop();
  incomplete.measurements[3].bandwidth_gb_s.reset();
  EXPECT_EQ(summarize_pattern_loop(incomplete).status,
            PatternRunStatus::Partial);

  PatternResults interrupted = incomplete;
  interrupted.measurements[3].status = PatternMeasurementStatus::Interrupted;
  interrupted.measurements[3].status_reason = "stop observed";
  const PatternLoopSummary interrupted_summary =
      summarize_pattern_loop(interrupted);
  EXPECT_EQ(interrupted_summary.status, PatternRunStatus::Interrupted);
  EXPECT_EQ(interrupted_summary.status_reason, "stop observed");

  PatternResults invalid = incomplete;
  invalid.measurements[3].status = PatternMeasurementStatus::Invalid;
  invalid.measurements[3].status_reason = "invalid duration";
  const PatternLoopSummary invalid_summary = summarize_pattern_loop(invalid);
  EXPECT_EQ(invalid_summary.status, PatternRunStatus::Failed);
  EXPECT_EQ(invalid_summary.status_reason, "invalid duration");

  const PatternLoopSummary execution_failure = summarize_pattern_loop(
      make_complete_pattern_loop(), true, true, "executor failed");
  EXPECT_EQ(execution_failure.status, PatternRunStatus::Failed);
  EXPECT_EQ(execution_failure.status_reason, "executor failed");
}

TEST(PatternBenchmarkTest, CompleteLoopWinsLateInterruptionFlag) {
  const PatternLoopSummary summary =
      summarize_pattern_loop(make_complete_pattern_loop(), false, true);
  EXPECT_EQ(summary.status, PatternRunStatus::Complete);
  EXPECT_EQ(summary.completed_measurements, 21u);
}

TEST(PatternBenchmarkTest, CollectorSumsExactCompletionCounters) {
  PatternStatistics statistics;
  initialize_pattern_statistics(statistics, 2);

  collect_pattern_loop_result(statistics, make_complete_pattern_loop());
  EXPECT_EQ(statistics.status, PatternRunStatus::Partial);
  EXPECT_EQ(statistics.planned_loops, 2u);
  EXPECT_EQ(statistics.completed_loops, 1u);
  EXPECT_EQ(statistics.planned_measurements, 42u);
  EXPECT_EQ(statistics.completed_measurements, 21u);

  PatternResults partial = make_complete_pattern_loop();
  partial.measurements.back().bandwidth_gb_s.reset();
  collect_pattern_loop_result(statistics, std::move(partial));
  EXPECT_EQ(statistics.status, PatternRunStatus::Partial);
  EXPECT_EQ(statistics.completed_loops, 1u);
  EXPECT_EQ(statistics.completed_measurements, 41u);
  ASSERT_EQ(statistics.loop_results.size(), 2u);
  EXPECT_EQ(statistics.loop_results[1].status, PatternRunStatus::Partial);
}

TEST(PatternBenchmarkTest, LaterCompleteLoopPreservesEarlierPartialReason) {
  PatternStatistics statistics;
  initialize_pattern_statistics(statistics, 2);
  PatternResults partial = make_complete_pattern_loop();
  partial.measurements.back().bandwidth_gb_s.reset();
  collect_pattern_loop_result(statistics, std::move(partial));
  const std::string partial_reason = statistics.status_reason;

  collect_pattern_loop_result(statistics, make_complete_pattern_loop());
  EXPECT_EQ(statistics.status, PatternRunStatus::Partial);
  EXPECT_EQ(statistics.status_reason, partial_reason);
  EXPECT_EQ(statistics.completed_loops, 1u);
  EXPECT_EQ(statistics.planned_loops, 2u);
  EXPECT_EQ(statistics.completed_measurements, 41u);
  EXPECT_EQ(statistics.planned_measurements, 42u);
}

TEST(PatternBenchmarkTest, CoordinatorReportsBufferPreparationFailuresWithPlannedCounts) {
  BenchmarkConfig config;
  config.loop_count = 2;
  config.buffer_size = 4096;

  PatternRunnerTestHooks allocation_failure = make_pattern_runner_hooks();
  allocation_failure.allocate_buffers =
      [](const BenchmarkConfig&, PatternBuffers&) { return EXIT_FAILURE; };
  PatternStatistics statistics;
  EXPECT_EQ(run_all_pattern_benchmarks(config, statistics,
                                       &allocation_failure),
            EXIT_FAILURE);
  EXPECT_EQ(statistics.status, PatternRunStatus::Failed);
  EXPECT_EQ(statistics.status_reason,
            Messages::pattern_reason_buffers_allocation_failed());
  EXPECT_EQ(statistics.planned_loops, 2u);
  EXPECT_EQ(statistics.completed_loops, 0u);
  EXPECT_EQ(statistics.planned_measurements, 42u);
  EXPECT_EQ(statistics.completed_measurements, 0u);
  EXPECT_TRUE(statistics.loop_results.empty());

  PatternRunnerTestHooks initialization_failure = make_pattern_runner_hooks();
  initialization_failure.initialize_buffers =
      [](const PatternBuffers&, size_t) { return EXIT_FAILURE; };
  EXPECT_EQ(run_all_pattern_benchmarks(config, statistics,
                                       &initialization_failure),
            EXIT_FAILURE);
  EXPECT_EQ(statistics.status, PatternRunStatus::Failed);
  EXPECT_EQ(statistics.status_reason,
            Messages::pattern_reason_buffers_initialization_failed());
  EXPECT_EQ(statistics.planned_measurements, 42u);
  EXPECT_EQ(statistics.completed_measurements, 0u);
  EXPECT_TRUE(statistics.loop_results.empty());
}

TEST(PatternBenchmarkTest, CoordinatorStopsBeforeFirstLoopWithPlannedCounts) {
  BenchmarkConfig config;
  config.loop_count = 2;
  config.buffer_size = 4096;
  PatternRunnerTestHooks hooks = make_pattern_runner_hooks();
  hooks.stop_requested = []() { return true; };
  hooks.execute_loop = [](const PatternBuffers&, const BenchmarkConfig&,
                          PatternResults&, size_t) {
    ADD_FAILURE() << "executor must not run after a pre-loop stop";
    return EXIT_FAILURE;
  };

  PatternStatistics statistics;
  EXPECT_EQ(run_all_pattern_benchmarks(config, statistics, &hooks),
            EXIT_SUCCESS);
  EXPECT_EQ(statistics.status, PatternRunStatus::Interrupted);
  EXPECT_EQ(statistics.status_reason,
            Messages::pattern_reason_loop_interrupted());
  EXPECT_EQ(statistics.planned_loops, 2u);
  EXPECT_EQ(statistics.completed_loops, 0u);
  EXPECT_EQ(statistics.planned_measurements, 42u);
  EXPECT_EQ(statistics.completed_measurements, 0u);
  EXPECT_TRUE(statistics.loop_results.empty());
}

TEST(PatternBenchmarkTest, CoordinatorConvertsLoopExceptionToFailedEvidence) {
  BenchmarkConfig config;
  config.loop_count = 1;
  config.buffer_size = 4096;
  PatternRunnerTestHooks hooks = make_pattern_runner_hooks();
  hooks.execute_loop = [](const PatternBuffers&, const BenchmarkConfig&,
                          PatternResults&, size_t) -> int {
    throw std::runtime_error("injected exception");
  };

  PatternStatistics statistics;
  testing::internal::CaptureStderr();
  const int status = run_all_pattern_benchmarks(config, statistics, &hooks);
  const std::string error_output = testing::internal::GetCapturedStderr();
  EXPECT_EQ(status, EXIT_FAILURE);
  ASSERT_EQ(statistics.loop_results.size(), 1u);
  EXPECT_EQ(statistics.status, PatternRunStatus::Failed);
  EXPECT_EQ(statistics.status_reason,
            Messages::pattern_reason_loop_exception("injected exception"));
  EXPECT_EQ(statistics.completed_measurements, 0u);
  EXPECT_NE(error_output.find("injected exception"), std::string::npos);

  hooks.execute_loop = [](const PatternBuffers&, const BenchmarkConfig&,
                          PatternResults&, size_t) -> int { throw 7; };
  testing::internal::CaptureStderr();
  EXPECT_EQ(run_all_pattern_benchmarks(config, statistics, &hooks),
            EXIT_FAILURE);
  (void)testing::internal::GetCapturedStderr();
  ASSERT_EQ(statistics.loop_results.size(), 1u);
  EXPECT_EQ(statistics.status_reason,
            Messages::pattern_reason_unknown_loop_exception());
}

TEST(PatternBenchmarkTest, CoordinatorContainsPreparationHookExceptions) {
  BenchmarkConfig config;
  config.loop_count = 2;
  config.buffer_size = 4096;
  PatternStatistics statistics;

  PatternRunnerTestHooks hooks = make_pattern_runner_hooks();
  hooks.allocate_buffers = [](const BenchmarkConfig&, PatternBuffers&) -> int {
    throw std::runtime_error("allocation hook exception");
  };
  testing::internal::CaptureStderr();
  EXPECT_EQ(run_all_pattern_benchmarks(config, statistics, &hooks),
            EXIT_FAILURE);
  (void)testing::internal::GetCapturedStderr();
  EXPECT_EQ(statistics.status, PatternRunStatus::Failed);
  EXPECT_EQ(
      statistics.status_reason,
      Messages::pattern_reason_coordinator_exception(
          "allocation hook exception"));
  EXPECT_EQ(statistics.planned_measurements, 42u);
  EXPECT_TRUE(statistics.loop_results.empty());

  hooks = make_pattern_runner_hooks();
  hooks.initialize_buffers = [](const PatternBuffers&, size_t) -> int {
    throw 9;
  };
  testing::internal::CaptureStderr();
  EXPECT_EQ(run_all_pattern_benchmarks(config, statistics, &hooks),
            EXIT_FAILURE);
  (void)testing::internal::GetCapturedStderr();
  EXPECT_EQ(statistics.status_reason,
            Messages::pattern_reason_unknown_coordinator_exception());
  EXPECT_EQ(statistics.planned_measurements, 42u);
  EXPECT_TRUE(statistics.loop_results.empty());
}

TEST(PatternBenchmarkTest, CoordinatorPreservesFailedAndPartialLoopEvidence) {
  BenchmarkConfig config;
  config.loop_count = 1;
  config.buffer_size = 4096;

  PatternRunnerTestHooks failure = make_pattern_runner_hooks();
  failure.execute_loop = [](const PatternBuffers&, const BenchmarkConfig&,
                            PatternResults& results, size_t) {
    results = make_complete_pattern_loop(4.0);
    return EXIT_FAILURE;
  };
  PatternStatistics statistics;
  EXPECT_EQ(run_all_pattern_benchmarks(config, statistics, &failure),
            EXIT_FAILURE);
  ASSERT_EQ(statistics.loop_results.size(), 1u);
  EXPECT_EQ(statistics.status, PatternRunStatus::Failed);
  EXPECT_EQ(statistics.loop_results[0].status, PatternRunStatus::Failed);
  EXPECT_EQ(statistics.completed_loops, 0u);
  EXPECT_EQ(statistics.completed_measurements, 21u);

  PatternRunnerTestHooks partial = make_pattern_runner_hooks();
  partial.execute_loop = [](const PatternBuffers&, const BenchmarkConfig&,
                            PatternResults& results, size_t) {
    results = make_complete_pattern_loop(5.0);
    results.measurements.back().bandwidth_gb_s.reset();
    return EXIT_SUCCESS;
  };
  EXPECT_EQ(run_all_pattern_benchmarks(config, statistics, &partial),
            EXIT_SUCCESS);
  ASSERT_EQ(statistics.loop_results.size(), 1u);
  EXPECT_EQ(statistics.status, PatternRunStatus::Partial);
  EXPECT_EQ(statistics.status_reason,
            Messages::pattern_reason_loop_incomplete());
  EXPECT_EQ(statistics.loop_results[0].status, PatternRunStatus::Partial);
  EXPECT_EQ(statistics.completed_loops, 0u);
  EXPECT_EQ(statistics.completed_measurements, 20u);
}

TEST(PatternBenchmarkTest, InvalidEvidenceFailsEvenWhenExecutorReturnsSuccess) {
  BenchmarkConfig config;
  config.loop_count = 1;
  config.buffer_size = 4096;
  PatternRunnerTestHooks hooks = make_pattern_runner_hooks();
  hooks.execute_loop = [](const PatternBuffers&, const BenchmarkConfig&,
                          PatternResults& results, size_t) {
    results = make_complete_pattern_loop();
    results.measurements.back().status = PatternMeasurementStatus::Invalid;
    results.measurements.back().status_reason = "invalid injected timing";
    results.measurements.back().bandwidth_gb_s.reset();
    return EXIT_SUCCESS;
  };

  PatternStatistics statistics;
  EXPECT_EQ(run_all_pattern_benchmarks(config, statistics, &hooks),
            EXIT_FAILURE);
  ASSERT_EQ(statistics.loop_results.size(), 1u);
  EXPECT_EQ(statistics.status, PatternRunStatus::Failed);
  EXPECT_EQ(statistics.status_reason, "invalid injected timing");
  EXPECT_EQ(statistics.completed_loops, 0u);
  EXPECT_EQ(statistics.completed_measurements, 20u);
}

TEST(PatternBenchmarkTest, FailedLoopEvidenceIsExcludedFromJsonAggregate) {
  BenchmarkConfig config;
  config.loop_count = 2;
  config.buffer_size = 4096;
  PatternRunnerTestHooks hooks = make_pattern_runner_hooks();
  hooks.execute_loop = [](const PatternBuffers&, const BenchmarkConfig&,
                          PatternResults& results, size_t loop_index) {
    results = make_skipped_pattern_loop();
    set_pattern_measurement(
        results, PatternKind::SequentialForward, PatternOperation::Read,
        make_test_measurement(PatternMeasurementStatus::Measured, "",
                              loop_index == 0 ? 10.0 : 100.0));
    if (loop_index == 0) {
      return EXIT_SUCCESS;
    }
    results.measurements.back().status = PatternMeasurementStatus::Measured;
    results.measurements.back().status_reason = "missing injected value";
    results.measurements.back().bandwidth_gb_s.reset();
    return EXIT_FAILURE;
  };

  PatternStatistics statistics;
  EXPECT_EQ(run_all_pattern_benchmarks(config, statistics, &hooks),
            EXIT_FAILURE);
  ASSERT_EQ(statistics.all_forward_read_bw.size(), 1u);
  EXPECT_DOUBLE_EQ(statistics.all_forward_read_bw.front(), 10.0);
  const PatternResults headline = extract_pattern_median_results(statistics);
  const PatternMeasurement& headline_read = get_pattern_measurement(
      headline, PatternKind::SequentialForward, PatternOperation::Read);
  ASSERT_TRUE(headline_read.bandwidth_gb_s.has_value());
  EXPECT_DOUBLE_EQ(*headline_read.bandwidth_gb_s, 10.0);

  const nlohmann::ordered_json output =
      build_pattern_results_json(config, statistics, 0.5);
  EXPECT_EQ(output["status"], "failed");
  EXPECT_EQ(output["completed_loops"], 1u);
  EXPECT_EQ(output["planned_loops"], 2u);
  EXPECT_EQ(output["completed_measurements"], 41u);
  EXPECT_EQ(output["planned_measurements"], 42u);
  EXPECT_FALSE(output["results_complete"].get<bool>());
  const nlohmann::ordered_json& read =
      output[JsonKeys::PATTERNS][JsonKeys::SEQUENTIAL_FORWARD]
            [JsonKeys::BANDWIDTH][JsonKeys::READ_GB_S];
  ASSERT_EQ(read["values_gb_s"].size(), 1u);
  EXPECT_DOUBLE_EQ(read["values_gb_s"][0].get<double>(), 10.0);
  EXPECT_DOUBLE_EQ(read["value_gb_s"].get<double>(), 10.0);
  EXPECT_DOUBLE_EQ(read["statistics"]["median_p50"].get<double>(), 10.0);
  ASSERT_EQ(read["measurements"].size(), 2u);
  EXPECT_DOUBLE_EQ(read["measurements"][0]["value_gb_s"].get<double>(),
                   10.0);
  EXPECT_DOUBLE_EQ(read["measurements"][1]["value_gb_s"].get<double>(),
                   100.0);
}

TEST(PatternBenchmarkTest, CoordinatorPreservesInterruptedLoopEvidence) {
  BenchmarkConfig config;
  config.loop_count = 1;
  config.buffer_size = 4096;
  PatternRunnerTestHooks hooks = make_pattern_runner_hooks();
  hooks.execute_loop = [](const PatternBuffers&, const BenchmarkConfig&,
                          PatternResults& results, size_t) {
    results = make_complete_pattern_loop();
    results.measurements.back().status = PatternMeasurementStatus::Interrupted;
    results.measurements.back().status_reason = "injected interruption";
    results.measurements.back().bandwidth_gb_s.reset();
    return EXIT_SUCCESS;
  };

  PatternStatistics statistics;
  EXPECT_EQ(run_all_pattern_benchmarks(config, statistics, &hooks),
            EXIT_SUCCESS);
  ASSERT_EQ(statistics.loop_results.size(), 1u);
  EXPECT_EQ(statistics.status, PatternRunStatus::Interrupted);
  EXPECT_EQ(statistics.status_reason, "injected interruption");
  EXPECT_EQ(statistics.completed_measurements, 20u);
}

TEST(PatternBenchmarkTest, OneShotLateStopInterruptsOnlyWhenLoopsRemain) {
  auto run_with_loop_count = [](int loop_count) {
    BenchmarkConfig config;
    config.loop_count = loop_count;
    config.buffer_size = 4096;
    PatternRunnerTestHooks hooks = make_pattern_runner_hooks();
    hooks.execute_loop = [](const PatternBuffers&, const BenchmarkConfig&,
                            PatternResults& results, size_t) {
      results = make_complete_pattern_loop();
      return EXIT_SUCCESS;
    };
    size_t stop_checks = 0;
    hooks.stop_requested = [&stop_checks]() {
      ++stop_checks;
      return stop_checks == 2;
    };
    PatternStatistics statistics;
    EXPECT_EQ(run_all_pattern_benchmarks(config, statistics, &hooks),
              EXIT_SUCCESS);
    return statistics;
  };

  const PatternStatistics unfinished = run_with_loop_count(2);
  EXPECT_EQ(unfinished.status, PatternRunStatus::Interrupted);
  EXPECT_EQ(unfinished.completed_loops, 1u);
  EXPECT_EQ(unfinished.completed_measurements, 21u);

  const PatternStatistics finished = run_with_loop_count(1);
  EXPECT_EQ(finished.status, PatternRunStatus::Complete);
  EXPECT_TRUE(finished.status_reason.empty());
  EXPECT_EQ(finished.completed_loops, 1u);
  EXPECT_EQ(finished.completed_measurements, 21u);
}

TEST(PatternBenchmarkTest, ExecutionOrderIsDeterministicAndRotatesAcrossLoops) {
  const auto first = build_pattern_execution_order(0);
  const auto repeated = build_pattern_execution_order(0);
  const auto next = build_pattern_execution_order(1);

  EXPECT_EQ(first, repeated);
  EXPECT_NE(first, next);
  for (size_t position = 0; position < first.size(); ++position) {
    EXPECT_EQ(next[position], first[(position + 1) % first.size()]);
  }
}

TEST(PatternBenchmarkTest, ExecutionOrderBalancesEveryPatternAcrossPositions) {
  constexpr size_t pattern_count = static_cast<size_t>(PatternKind::Count);
  std::array<std::array<size_t, pattern_count>, pattern_count> positions{};

  for (size_t loop = 0; loop < pattern_count; ++loop) {
    const auto order = build_pattern_execution_order(loop);
    for (size_t position = 0; position < pattern_count; ++position) {
      ++positions[static_cast<size_t>(order[position])][position];
    }
  }

  for (const auto& pattern_positions : positions) {
    for (size_t count : pattern_positions) EXPECT_EQ(count, 1u);
  }
}

// Integration test: Test that the representative pattern benchmark run produces every core pattern result.
// NOTE: This is an integration test that performs actual system operations.
// It runs real pattern benchmarks which may be slower and can fail on slow systems or under load.
// Use 'make test-integration' to run integration tests, or 'make test' for unit tests only.
TEST(PatternBenchmarkTest, RunPatternBenchmarksCorePatternsIntegration) {
  BenchmarkConfig config = make_pattern_config(512 * 1024, 1);
  PatternResults results;

  ASSERT_TRUE(run_pattern_benchmarks_with_fresh_buffers(config, results));

  expect_core_pattern_bandwidths_positive(results);
  expect_2mb_pattern_bandwidths_zero(results);
  const PatternMeasurement& forward_read = get_pattern_measurement(
      results, PatternKind::SequentialForward, PatternOperation::Read);
  EXPECT_EQ(forward_read.status, PatternMeasurementStatus::Measured);
  EXPECT_EQ(forward_read.passes, 1u);
  EXPECT_GT(forward_read.total_payload_bytes, 0u);
  const PatternMeasurement& random_read = get_pattern_measurement(
      results, PatternKind::Random, PatternOperation::Read);
  EXPECT_EQ(random_read.status, PatternMeasurementStatus::Measured);
  EXPECT_TRUE(random_read.has_seed);
  EXPECT_EQ(random_read.seed, config.pattern_seed);
}

TEST(PatternBenchmarkTest, Strided2MiBRouteAndKernelIntegration) {
  BenchmarkConfig config = make_pattern_config(8 * 1024 * 1024, 1);

  PatternResults results;
  ASSERT_TRUE(run_pattern_benchmarks_with_fresh_buffers(config, results));

  expect_2mb_pattern_bandwidths_positive(results);
}

TEST(PatternBenchmarkTest, PhasedStridedKernelsRespectAccessBoundariesIntegration) {
  const std::vector<size_t> strides = {
      Constants::PATTERN_STRIDE_CACHE_LINE,
      Constants::PATTERN_STRIDE_PAGE,
      Constants::PATTERN_STRIDE_PAGE_16K,
      Constants::PATTERN_STRIDE_SUPERPAGE_2MB,
  };

  for (size_t stride : strides) {
    SCOPED_TRACE(stride);
    const size_t span = stride + Constants::PATTERN_ACCESS_SIZE_BYTES;
    std::vector<unsigned char> source_storage(span + 256, 0xA5);
    std::vector<unsigned char> destination_storage(span + 256, 0x5A);
    unsigned char* source = align_to_cache_line(source_storage.data() + 64);
    unsigned char* destination = align_to_cache_line(destination_storage.data() + 64);

    std::fill(source, source + span, 0x00);
    const uint64_t values[] = {1, 2, 4};
    std::memcpy(source, &values[0], sizeof(values[0]));
    std::memcpy(source + stride, &values[1], sizeof(values[1]));
    std::memcpy(source + Constants::PATTERN_ACCESS_SIZE_BYTES, &values[2],
                sizeof(values[2]));

    EXPECT_EQ(memory_read_strided_phased_loop_asm(source, span, stride, 2, 0), 7u);

    memory_copy_strided_phased_loop_asm(destination, source, span, stride, 2, 0);
    for (size_t offset = 0; offset < span; ++offset) {
      const bool copied = offset < Constants::PATTERN_ACCESS_SIZE_BYTES ||
                          (offset >= Constants::PATTERN_ACCESS_SIZE_BYTES &&
                           offset < 2 * Constants::PATTERN_ACCESS_SIZE_BYTES) ||
                          offset >= stride;
      EXPECT_EQ(destination[offset], copied ? source[offset] : 0x5A);
    }

    std::fill(destination, destination + span, 0x5A);
    memory_write_strided_phased_loop_asm(destination, span, stride, 2, 0);
    for (size_t offset = 0; offset < span; ++offset) {
      const bool written = offset < 2 * Constants::PATTERN_ACCESS_SIZE_BYTES ||
                           offset >= stride;
      EXPECT_EQ(destination[offset], written ? 0x00 : 0x5A);
    }

    for (size_t guard = 1; guard <= 64; ++guard) {
      EXPECT_EQ(destination[-static_cast<ptrdiff_t>(guard)], 0x5A);
      EXPECT_EQ(destination[span + guard - 1], 0x5A);
    }
  }
}

TEST(PatternBenchmarkTest, FinalizedPatternPlansDriveWarmupKernelsIntegration) {
  constexpr size_t buffer_size = 256;
  constexpr uint64_t initial_checksum = 0x123456789abcdef0ULL;
  std::vector<unsigned char> source_storage(
      buffer_size + Constants::CACHE_LINE_SIZE_BYTES);
  std::vector<unsigned char> destination_storage(
      buffer_size + Constants::CACHE_LINE_SIZE_BYTES);
  unsigned char* source = align_to_cache_line(source_storage.data());
  unsigned char* destination =
      align_to_cache_line(destination_storage.data());
  for (size_t offset = 0; offset < buffer_size; ++offset) {
    source[offset] = static_cast<unsigned char>((offset * 29 + 17) & 0xff);
  }

  const PatternWorkPlan strided_plan = build_strided_pattern_work_plan(
      buffer_size, Constants::PATTERN_STRIDE_CACHE_LINE,
      Constants::PATTERN_ACCESS_SIZE_BYTES, 2, 1, 0);
  ASSERT_EQ(strided_plan.status, PatternMeasurementStatus::Measured);
  ASSERT_EQ(strided_plan.workers.size(), 2u);

  uint64_t expected_checksum = initial_checksum;
  for (const PatternWorkerRange& worker : strided_plan.workers) {
    expected_checksum ^= memory_read_strided_phased_loop_asm(
        source + worker.offset_bytes, worker.span_bytes,
        strided_plan.stride_bytes, strided_plan.phase_period_passes, 0);
  }
  std::atomic<uint64_t> checksum{initial_checksum};
  warmup_read_strided(source, strided_plan, checksum);
  EXPECT_EQ(checksum.load(std::memory_order_acquire), expected_checksum);

  std::fill(destination, destination + buffer_size, 0xa5);
  warmup_write_strided(destination, strided_plan);
  EXPECT_TRUE(std::all_of(destination, destination + buffer_size,
                          [](unsigned char value) { return value == 0; }));

  std::fill(destination, destination + buffer_size, 0xa5);
  warmup_copy_strided(destination, source, strided_plan);
  EXPECT_TRUE(std::equal(source, source + buffer_size, destination));

  const std::vector<size_t> global_indices = {0, 96, 128, 224};
  const std::vector<PatternRandomWorkerIndices> random_workers =
      build_random_worker_indices(buffer_size,
                                  Constants::PATTERN_ACCESS_SIZE_BYTES, 2,
                                  global_indices);
  ASSERT_EQ(random_workers.size(), 2u);

  expected_checksum = initial_checksum;
  for (const PatternRandomWorkerIndices& worker : random_workers) {
    expected_checksum ^= memory_read_random_loop_asm(
        source + worker.offset_bytes, worker.indices.data(),
        worker.indices.size());
  }
  checksum.store(initial_checksum, std::memory_order_relaxed);
  warmup_read_random(source, random_workers, checksum);
  EXPECT_EQ(checksum.load(std::memory_order_acquire), expected_checksum);
}

TEST(PatternBenchmarkTest, CacheReadWarmupUsesCacheKernelIntegration) {
  constexpr size_t buffer_size = 1024;
  std::vector<unsigned char> storage(
      buffer_size + Constants::CACHE_LINE_SIZE_BYTES);
  unsigned char* source = align_to_cache_line(storage.data());
  for (size_t offset = 0; offset < buffer_size; ++offset) {
    source[offset] = static_cast<unsigned char>((offset * 37 + 11) & 0xff);
  }

  const uint64_t expected_checksum =
      memory_read_cache_loop_asm(source, buffer_size);
  std::atomic<uint64_t> checksum{0};
  warmup_cache_read(source, buffer_size, 1, checksum);

  EXPECT_EQ(checksum.load(std::memory_order_acquire), expected_checksum);
}

TEST(PatternBenchmarkTest, RandomWarmupUsesChunkRelativeWorkerOffsetsIntegration) {
  constexpr size_t buffer_size = 256;
  std::vector<unsigned char> source_storage(
      buffer_size + Constants::CACHE_LINE_SIZE_BYTES);
  std::vector<unsigned char> destination_storage(
      buffer_size + Constants::CACHE_LINE_SIZE_BYTES);
  unsigned char* source = align_to_cache_line(source_storage.data());
  unsigned char* destination =
      align_to_cache_line(destination_storage.data());
  for (size_t offset = 0; offset < buffer_size; ++offset) {
    source[offset] = static_cast<unsigned char>((offset * 31 + 7) & 0xff);
  }

  const std::vector<size_t> global_indices = {0, 96, 128, 224};
  const std::vector<PatternRandomWorkerIndices> workers =
      build_random_worker_indices(buffer_size,
                                  Constants::PATTERN_ACCESS_SIZE_BYTES, 2,
                                  global_indices);
  ASSERT_EQ(workers.size(), 2u);
  EXPECT_EQ(workers[0].indices, (std::vector<size_t>{0, 96}));
  EXPECT_EQ(workers[1].indices, (std::vector<size_t>{0, 96}));

  std::fill(destination, destination + buffer_size, 0xa5);
  warmup_copy_random(destination, source, workers);
  for (size_t offset = 0; offset < buffer_size; ++offset) {
    const bool selected = offset < 32 || (offset >= 96 && offset < 160) ||
                          offset >= 224;
    EXPECT_EQ(destination[offset], selected ? source[offset] : 0xa5)
        << "offset=" << offset;
  }

  std::fill(destination, destination + buffer_size, 0xa5);
  warmup_write_random(destination, workers);
  for (size_t offset = 0; offset < buffer_size; ++offset) {
    const bool selected = offset < 32 || (offset >= 96 && offset < 160) ||
                          offset >= 224;
    EXPECT_EQ(destination[offset], selected ? 0u : 0xa5u)
        << "offset=" << offset;
  }
}

TEST(PatternBenchmarkTest, PhasedStridedKernelsPreserveAapcs64RegistersIntegration) {
  const std::vector<size_t> strides = {
      Constants::PATTERN_STRIDE_CACHE_LINE,
      Constants::PATTERN_STRIDE_PAGE,
      Constants::PATTERN_STRIDE_PAGE_16K,
      Constants::PATTERN_STRIDE_SUPERPAGE_2MB,
  };

  for (size_t stride : strides) {
    SCOPED_TRACE(stride);
    const size_t span = stride + Constants::PATTERN_ACCESS_SIZE_BYTES;
    std::vector<unsigned char> source(span, 0xA5);
    std::vector<unsigned char> destination(span, 0x5A);

    EXPECT_EQ(verify_pattern_callee_saved_registers_asm(
                  reinterpret_cast<uintptr_t>(
                      &memory_read_strided_phased_loop_asm),
                  reinterpret_cast<uintptr_t>(source.data()), span, stride, 2,
                  0, 0),
              1u);
    EXPECT_EQ(verify_pattern_callee_saved_registers_asm(
                  reinterpret_cast<uintptr_t>(
                      &memory_write_strided_phased_loop_asm),
                  reinterpret_cast<uintptr_t>(destination.data()), span, stride,
                  2, 0, 0),
              1u);
    EXPECT_EQ(verify_pattern_callee_saved_registers_asm(
                  reinterpret_cast<uintptr_t>(
                      &memory_copy_strided_phased_loop_asm),
                  reinterpret_cast<uintptr_t>(destination.data()),
                  reinterpret_cast<uintptr_t>(source.data()), span, stride, 2,
                  0),
              1u);
  }
}

TEST(PatternBenchmarkTest, FinalizedWorkPlanDrivesInstrumentedExecutorIntegration) {
  constexpr size_t buffer_size = 8 * 1024 * 1024;
  const PatternWorkPlan pilot_plan = build_strided_pattern_work_plan(
      buffer_size, Constants::PATTERN_STRIDE_SUPERPAGE_2MB, Constants::PATTERN_ACCESS_SIZE_BYTES, 10, 1, 0);
  ASSERT_EQ(pilot_plan.status, PatternMeasurementStatus::Measured);

  PatternWorkPlan measured_plan = pilot_plan;
  ASSERT_TRUE(set_strided_pattern_passes(measured_plan, measured_plan.phase_period_passes + 3));

  std::vector<size_t> boundaries;
  boundaries.reserve(measured_plan.workers.size() + 1);
  boundaries.push_back(0);
  for (const PatternWorkerRange& worker : measured_plan.workers) {
    ASSERT_EQ(worker.offset_bytes, boundaries.back());
    boundaries.push_back(worker.offset_bytes + worker.span_bytes);
  }
  ASSERT_EQ(boundaries.back(), buffer_size);

  std::vector<unsigned char> buffer(buffer_size, 0);
  auto timer = HighResTimer::create();
  ASSERT_TRUE(timer.has_value());
  std::atomic<size_t> executed_accesses{0};
  std::atomic<size_t> range_mismatches{0};

  const double duration = run_parallel_test_indexed_with_boundaries(
      buffer.data(), buffer.size(), static_cast<int>(measured_plan.passes), *timer, boundaries,
      [&](char* chunk_start, size_t chunk_size, int passes, size_t worker_index) {
        const PatternWorkerRange& worker = measured_plan.workers[worker_index];
        if (chunk_start != reinterpret_cast<char*>(buffer.data()) + worker.offset_bytes ||
            chunk_size != worker.span_bytes) {
          range_mismatches.fetch_add(1, std::memory_order_relaxed);
          return;
        }

        size_t local_accesses = 0;
        for (int pass = 0; pass < passes; ++pass) {
          const size_t phase =
              (static_cast<size_t>(pass) * measured_plan.access_size_bytes) % measured_plan.stride_bytes;
          for (size_t offset = phase; offset <= chunk_size - measured_plan.access_size_bytes;
               offset += measured_plan.stride_bytes) {
            ++local_accesses;
          }
        }
        executed_accesses.fetch_add(local_accesses, std::memory_order_relaxed);
      },
      "instrumented_pattern");

  EXPECT_GT(duration, 0.0);
  EXPECT_EQ(range_mismatches.load(std::memory_order_relaxed), 0u);
  EXPECT_EQ(executed_accesses.load(std::memory_order_relaxed), measured_plan.total_accesses);
  EXPECT_EQ(executed_accesses.load(std::memory_order_relaxed) * measured_plan.access_size_bytes,
            measured_plan.total_payload_bytes);
}

TEST(PatternBenchmarkTest, PureCollectionFiltersUnavailableValuesAndPreservesEvidence) {
  PatternStatistics statistics;
  statistics.all_forward_read_bw = {999.0};
  statistics.all_reverse_write_bw = {999.0};
  statistics.all_random_copy_bw = {999.0};
  statistics.loop_results.emplace_back();
  initialize_pattern_statistics(statistics, 4);

  EXPECT_TRUE(statistics.loop_results.empty());
  EXPECT_TRUE(statistics.all_forward_read_bw.empty());
  EXPECT_TRUE(statistics.all_reverse_write_bw.empty());
  EXPECT_TRUE(statistics.all_random_copy_bw.empty());

  PatternResults measured_loop = make_skipped_pattern_loop();
  set_pattern_measurement(
      measured_loop, PatternKind::SequentialForward, PatternOperation::Read,
      make_test_measurement(PatternMeasurementStatus::Measured, "", 12.5));
  set_pattern_measurement(
      measured_loop, PatternKind::SequentialReverse, PatternOperation::Write,
      make_test_measurement(PatternMeasurementStatus::Measured, "", 7.5));
  set_pattern_measurement(
      measured_loop, PatternKind::Random, PatternOperation::Copy,
      make_test_measurement(PatternMeasurementStatus::Measured, "", 3.25));
  collect_pattern_loop_result(statistics, std::move(measured_loop));

  const std::array<PatternMeasurementStatus, 3> unavailable_statuses = {
      PatternMeasurementStatus::Skipped,
      PatternMeasurementStatus::Interrupted,
      PatternMeasurementStatus::Invalid};
  const std::array<std::string, 3> unavailable_reasons = {
      "unsupported test pattern", "interrupted after preparation",
      "invalid measured duration"};
  for (size_t index = 0; index < unavailable_statuses.size(); ++index) {
    PatternResults loop = make_skipped_pattern_loop();
    set_pattern_measurement(
        loop, PatternKind::SequentialForward, PatternOperation::Read,
        make_test_measurement(unavailable_statuses[index],
                              unavailable_reasons[index]));
    collect_pattern_loop_result(statistics, std::move(loop));
  }

  ASSERT_EQ(statistics.all_forward_read_bw.size(), 1u);
  EXPECT_DOUBLE_EQ(statistics.all_forward_read_bw.front(), 12.5);
  ASSERT_EQ(statistics.all_reverse_write_bw.size(), 1u);
  EXPECT_DOUBLE_EQ(statistics.all_reverse_write_bw.front(), 7.5);
  ASSERT_EQ(statistics.all_random_copy_bw.size(), 1u);
  EXPECT_DOUBLE_EQ(statistics.all_random_copy_bw.front(), 3.25);
  EXPECT_TRUE(statistics.all_forward_write_bw.empty());
  EXPECT_TRUE(statistics.all_strided_2mb_read_bw.empty());

  ASSERT_EQ(statistics.loop_results.size(), 4u);
  const PatternMeasurement& measured = get_pattern_measurement(
      statistics.loop_results[0], PatternKind::SequentialForward,
      PatternOperation::Read);
  EXPECT_EQ(measured.status, PatternMeasurementStatus::Measured);
  EXPECT_TRUE(measured.status_reason.empty());
  ASSERT_TRUE(measured.bandwidth_gb_s.has_value());
  EXPECT_DOUBLE_EQ(*measured.bandwidth_gb_s, 12.5);
  for (size_t index = 0; index < unavailable_statuses.size(); ++index) {
    const PatternMeasurement& unavailable = get_pattern_measurement(
        statistics.loop_results[index + 1], PatternKind::SequentialForward,
        PatternOperation::Read);
    EXPECT_EQ(unavailable.status, unavailable_statuses[index]);
    EXPECT_EQ(unavailable.status_reason, unavailable_reasons[index]);
    EXPECT_FALSE(unavailable.bandwidth_gb_s.has_value());
  }
}

TEST(PatternBenchmarkTest,
     MedianHeadlineUsesMeasuredPopulationAndPreservesUnavailableEvidence) {
  PatternStatistics statistics;
  initialize_pattern_statistics(statistics, 4);
  const std::array<std::optional<double>, 4> read_values = {
      10.0, 30.0, 20.0, std::nullopt};
  for (size_t index = 0; index < read_values.size(); ++index) {
    const std::optional<double>& read_value = read_values[index];
    PatternResults loop = make_skipped_pattern_loop();
    set_pattern_measurement(
        loop, PatternKind::SequentialForward, PatternOperation::Read,
        read_value.has_value()
            ? make_test_measurement(PatternMeasurementStatus::Measured, "",
                                    read_value)
            : make_test_measurement(PatternMeasurementStatus::Skipped,
                                    "read unavailable"));
    set_pattern_measurement(
        loop, PatternKind::SequentialForward, PatternOperation::Write,
        make_test_measurement(PatternMeasurementStatus::Skipped,
                              "write unsupported"));
    set_pattern_measurement(
        loop, PatternKind::SequentialForward, PatternOperation::Copy,
        make_test_measurement(
            index + 1 == read_values.size()
                ? PatternMeasurementStatus::Interrupted
                : PatternMeasurementStatus::Skipped,
            index + 1 == read_values.size() ? "copy interrupted"
                                            : "copy unsupported"));
    set_pattern_measurement(
        loop, PatternKind::SequentialReverse, PatternOperation::Read,
        make_test_measurement(
            index + 1 == read_values.size()
                ? PatternMeasurementStatus::Invalid
                : PatternMeasurementStatus::Skipped,
            index + 1 == read_values.size() ? "reverse timing invalid"
                                            : "reverse unsupported"));
    collect_pattern_loop_result(statistics, std::move(loop));
  }

  ASSERT_EQ(statistics.all_forward_read_bw.size(), 3u);
  EXPECT_DOUBLE_EQ(statistics.all_forward_read_bw[0], 10.0);
  EXPECT_DOUBLE_EQ(statistics.all_forward_read_bw[1], 30.0);
  EXPECT_DOUBLE_EQ(statistics.all_forward_read_bw[2], 20.0);
  EXPECT_TRUE(statistics.all_forward_write_bw.empty());
  EXPECT_TRUE(statistics.all_forward_copy_bw.empty());
  EXPECT_TRUE(statistics.all_reverse_read_bw.empty());

  const PatternResults headline = extract_pattern_median_results(statistics);
  const PatternMeasurement& read = get_pattern_measurement(
      headline, PatternKind::SequentialForward, PatternOperation::Read);
  EXPECT_EQ(read.status, PatternMeasurementStatus::Measured);
  EXPECT_TRUE(read.status_reason.empty());
  ASSERT_TRUE(read.bandwidth_gb_s.has_value());
  EXPECT_DOUBLE_EQ(*read.bandwidth_gb_s, 20.0);

  const PatternMeasurement& write = get_pattern_measurement(
      headline, PatternKind::SequentialForward, PatternOperation::Write);
  EXPECT_EQ(write.status, PatternMeasurementStatus::Skipped);
  EXPECT_EQ(write.status_reason, "write unsupported");
  EXPECT_FALSE(write.bandwidth_gb_s.has_value());

  const PatternMeasurement& copy = get_pattern_measurement(
      headline, PatternKind::SequentialForward, PatternOperation::Copy);
  EXPECT_EQ(copy.status, PatternMeasurementStatus::Skipped);
  EXPECT_EQ(copy.status_reason, "copy unsupported");
  EXPECT_FALSE(copy.bandwidth_gb_s.has_value());

  const PatternMeasurement& invalid = get_pattern_measurement(
      headline, PatternKind::SequentialReverse, PatternOperation::Read);
  EXPECT_EQ(invalid.status, PatternMeasurementStatus::Skipped);
  EXPECT_EQ(invalid.status_reason, "reverse unsupported");
  EXPECT_FALSE(invalid.bandwidth_gb_s.has_value());

  const PatternMeasurement& retained_interruption = get_pattern_measurement(
      statistics.loop_results.back(), PatternKind::SequentialForward,
      PatternOperation::Copy);
  EXPECT_EQ(retained_interruption.status,
            PatternMeasurementStatus::Interrupted);
  const PatternMeasurement& retained_invalid = get_pattern_measurement(
      statistics.loop_results.back(), PatternKind::SequentialReverse,
      PatternOperation::Read);
  EXPECT_EQ(retained_invalid.status, PatternMeasurementStatus::Invalid);
}

TEST(PatternBenchmarkTest, StatisticsUseExactMedianCvAndMad) {
  const PatternStatisticsData statistics =
      calculate_pattern_statistics({10.0, 20.0, 30.0});
  EXPECT_DOUBLE_EQ(statistics.average, 20.0);
  EXPECT_DOUBLE_EQ(statistics.min, 10.0);
  EXPECT_DOUBLE_EQ(statistics.max, 30.0);
  EXPECT_DOUBLE_EQ(statistics.median, 20.0);
  EXPECT_DOUBLE_EQ(statistics.p90, 28.0);
  EXPECT_DOUBLE_EQ(statistics.p95, 29.0);
  EXPECT_DOUBLE_EQ(statistics.p99, 29.8);
  EXPECT_DOUBLE_EQ(statistics.stddev, 10.0);
  EXPECT_DOUBLE_EQ(statistics.coefficient_of_variation_pct, 50.0);
  EXPECT_DOUBLE_EQ(statistics.median_absolute_deviation, 10.0);

  const PatternStatisticsData even =
      calculate_pattern_statistics({1.0, 2.0, 100.0, 101.0});
  EXPECT_DOUBLE_EQ(even.median, 51.0);
  EXPECT_DOUBLE_EQ(even.median_absolute_deviation, 49.5);

  const PatternStatisticsData zero_mean =
      calculate_pattern_statistics({-1.0, 1.0});
  EXPECT_DOUBLE_EQ(zero_mean.average, 0.0);
  EXPECT_DOUBLE_EQ(zero_mean.median, 0.0);
  EXPECT_DOUBLE_EQ(zero_mean.coefficient_of_variation_pct, 0.0);
  EXPECT_DOUBLE_EQ(zero_mean.median_absolute_deviation, 1.0);
}

TEST(PatternBenchmarkTest, ConsoleRendersUnavailableMeasurementsAsStatusNotZero) {
  PatternResults results;
  for (size_t kind_index = 0;
       kind_index < static_cast<size_t>(PatternKind::Count); ++kind_index) {
    for (size_t operation_index = 0;
         operation_index < static_cast<size_t>(PatternOperation::Count);
         ++operation_index) {
      PatternMeasurement measurement;
      measurement.status = PatternMeasurementStatus::Skipped;
      measurement.status_reason = "test skip";
      set_pattern_measurement(results, static_cast<PatternKind>(kind_index),
                              static_cast<PatternOperation>(operation_index),
                              std::move(measurement));
    }
  }

  testing::internal::CaptureStdout();
  print_pattern_results(results);
  const std::string output = testing::internal::GetCapturedStdout();
  EXPECT_NE(output.find("N/A [skipped: test skip]"), std::string::npos);
  EXPECT_EQ(output.find("0.000 GB/s"), std::string::npos);
  EXPECT_EQ(output.find("Pattern Efficiency Analysis"), std::string::npos);
  EXPECT_NE(output.find("2 MiB stride"), std::string::npos);
}

TEST(PatternBenchmarkTest, StatisticsExposeCvAndEmitNoiseWarning) {
  PatternStatistics statistics;
  statistics.all_forward_read_bw = {10.0, 30.0};

  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  print_pattern_statistics(2, statistics);
  const std::string error_output = testing::internal::GetCapturedStderr();
  const std::string standard_output = testing::internal::GetCapturedStdout();

  EXPECT_NE(standard_output.find("Pattern Bandwidth"), std::string::npos);
  EXPECT_NE(standard_output.find("Median (P50):"), std::string::npos);
  EXPECT_NE(standard_output.find("CV:"), std::string::npos);
  EXPECT_NE(standard_output.find("Median absolute deviation:"),
            std::string::npos);
  EXPECT_NE(error_output.find("Noisy pattern measurement"), std::string::npos);
}

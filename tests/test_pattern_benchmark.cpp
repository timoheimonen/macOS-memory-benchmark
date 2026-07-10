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
#include "pattern_benchmark/pattern_benchmark.h"
#include "core/memory/buffer_manager.h"
#include "core/config/config.h"
#include "utils/benchmark.h"  // Declares system_info functions
#include "core/config/constants.h"
#include "test_config_helpers.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdlib>
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

unsigned char* align_to_cache_line(unsigned char* pointer) {
  const uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
  const uintptr_t aligned =
      (address + Constants::CACHE_LINE_SIZE_BYTES - 1) &
      ~(static_cast<uintptr_t>(Constants::CACHE_LINE_SIZE_BYTES) - 1);
  return reinterpret_cast<unsigned char*>(aligned);
}

::testing::AssertionResult run_pattern_benchmarks_with_fresh_buffers(BenchmarkConfig& config,
                                                                      PatternResults& results) {
  BenchmarkBuffers buffers;
  const ::testing::AssertionResult alloc_init_result = allocate_and_initialize_buffers(config, buffers);
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
  EXPECT_GT(results.forward_read_bw, 0.0);
  EXPECT_GT(results.forward_write_bw, 0.0);
  EXPECT_GT(results.forward_copy_bw, 0.0);

  EXPECT_GT(results.reverse_read_bw, 0.0);
  EXPECT_GT(results.reverse_write_bw, 0.0);
  EXPECT_GT(results.reverse_copy_bw, 0.0);

  EXPECT_GT(results.strided_64_read_bw, 0.0);
  EXPECT_GT(results.strided_64_write_bw, 0.0);
  EXPECT_GT(results.strided_64_copy_bw, 0.0);

  EXPECT_GT(results.strided_4096_read_bw, 0.0);
  EXPECT_GT(results.strided_4096_write_bw, 0.0);
  EXPECT_GT(results.strided_4096_copy_bw, 0.0);

  EXPECT_GT(results.strided_16384_read_bw, 0.0);
  EXPECT_GT(results.strided_16384_write_bw, 0.0);
  EXPECT_GT(results.strided_16384_copy_bw, 0.0);

  EXPECT_GT(results.random_read_bw, 0.0);
  EXPECT_GT(results.random_write_bw, 0.0);
  EXPECT_GT(results.random_copy_bw, 0.0);
}

void expect_2mb_pattern_bandwidths_zero(const PatternResults& results) {
  EXPECT_EQ(results.strided_2mb_read_bw, 0.0);
  EXPECT_EQ(results.strided_2mb_write_bw, 0.0);
  EXPECT_EQ(results.strided_2mb_copy_bw, 0.0);
  for (PatternOperation operation : {PatternOperation::Read, PatternOperation::Write,
                                     PatternOperation::Copy}) {
    const PatternMeasurement& measurement =
        get_pattern_measurement(results, PatternKind::Strided2MiB, operation);
    EXPECT_EQ(measurement.status, PatternMeasurementStatus::Skipped);
    EXPECT_FALSE(measurement.bandwidth_gb_s.has_value());
  }
}

void expect_2mb_pattern_bandwidths_positive(const PatternResults& results) {
  EXPECT_GT(results.strided_2mb_read_bw, 0.0);
  EXPECT_GT(results.strided_2mb_write_bw, 0.0);
  EXPECT_GT(results.strided_2mb_copy_bw, 0.0);
  for (PatternOperation operation : {PatternOperation::Read, PatternOperation::Write,
                                     PatternOperation::Copy}) {
    const PatternMeasurement& measurement =
        get_pattern_measurement(results, PatternKind::Strided2MiB, operation);
    EXPECT_EQ(measurement.status, PatternMeasurementStatus::Measured);
    ASSERT_TRUE(measurement.bandwidth_gb_s.has_value());
    EXPECT_GT(measurement.elapsed_seconds, 0.0);
    EXPECT_GT(measurement.total_payload_bytes, 0u);
    EXPECT_EQ(measurement.stride_bytes,
              Constants::PATTERN_STRIDE_SUPERPAGE_2MB);
    EXPECT_FALSE(measurement.large_page_backing_verified);
  }
}

}  // namespace

// Test PatternResults default initialization
TEST(PatternBenchmarkTest, PatternResultsDefaultValues) {
  PatternResults results;
  
  // All values should be initialized to 0.0
  EXPECT_EQ(results.forward_read_bw, 0.0);
  EXPECT_EQ(results.forward_write_bw, 0.0);
  EXPECT_EQ(results.forward_copy_bw, 0.0);
  EXPECT_EQ(results.reverse_read_bw, 0.0);
  EXPECT_EQ(results.reverse_write_bw, 0.0);
  EXPECT_EQ(results.reverse_copy_bw, 0.0);
  EXPECT_EQ(results.strided_64_read_bw, 0.0);
  EXPECT_EQ(results.strided_64_write_bw, 0.0);
  EXPECT_EQ(results.strided_64_copy_bw, 0.0);
  EXPECT_EQ(results.strided_4096_read_bw, 0.0);
  EXPECT_EQ(results.strided_4096_write_bw, 0.0);
  EXPECT_EQ(results.strided_4096_copy_bw, 0.0);
  EXPECT_EQ(results.strided_16384_read_bw, 0.0);
  EXPECT_EQ(results.strided_16384_write_bw, 0.0);
  EXPECT_EQ(results.strided_16384_copy_bw, 0.0);
  EXPECT_EQ(results.strided_2mb_read_bw, 0.0);
  EXPECT_EQ(results.strided_2mb_write_bw, 0.0);
  EXPECT_EQ(results.strided_2mb_copy_bw, 0.0);
  EXPECT_EQ(results.random_read_bw, 0.0);
  EXPECT_EQ(results.random_write_bw, 0.0);
  EXPECT_EQ(results.random_copy_bw, 0.0);
  for (const PatternMeasurement& measurement : results.measurements) {
    EXPECT_EQ(measurement.status, PatternMeasurementStatus::Invalid);
    EXPECT_FALSE(measurement.bandwidth_gb_s.has_value());
  }
}

// Test PatternResults can be set and read
TEST(PatternBenchmarkTest, PatternResultsSetValues) {
  PatternResults results;
  
  // Set some values
  results.forward_read_bw = 10.5;
  results.reverse_write_bw = 8.3;
  results.strided_64_copy_bw = 7.2;
  results.random_read_bw = 5.1;
  
  // Verify they can be read back
  EXPECT_DOUBLE_EQ(results.forward_read_bw, 10.5);
  EXPECT_DOUBLE_EQ(results.reverse_write_bw, 8.3);
  EXPECT_DOUBLE_EQ(results.strided_64_copy_bw, 7.2);
  EXPECT_DOUBLE_EQ(results.random_read_bw, 5.1);
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

// Test large-page stride variants with a larger buffer
TEST(PatternBenchmarkTest, StridedLargePagePatternsIntegration) {
  BenchmarkConfig config = make_pattern_config(8 * 1024 * 1024, 1);

  PatternResults results;
  ASSERT_TRUE(run_pattern_benchmarks_with_fresh_buffers(config, results));

  EXPECT_GT(results.strided_16384_read_bw, 0.0);
  EXPECT_GT(results.strided_16384_write_bw, 0.0);
  EXPECT_GT(results.strided_16384_copy_bw, 0.0);

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

TEST(PatternBenchmarkTest, StatisticsUseMedianAndCoefficientOfVariation) {
  const PatternStatisticsData statistics =
      calculate_pattern_statistics({10.0, 20.0, 30.0});
  EXPECT_DOUBLE_EQ(statistics.average, 20.0);
  EXPECT_DOUBLE_EQ(statistics.median, 20.0);
  EXPECT_DOUBLE_EQ(statistics.stddev, 10.0);
  EXPECT_DOUBLE_EQ(statistics.coefficient_of_variation_pct, 50.0);
}

TEST(PatternBenchmarkTest, MedianHeadlineExcludesSkippedMeasurements) {
  PatternStatistics statistics;
  for (double value : {10.0, 30.0, 20.0}) {
    PatternResults loop;
    PatternMeasurement measurement;
    measurement.status = PatternMeasurementStatus::Measured;
    measurement.status_reason.clear();
    measurement.bandwidth_gb_s = value;
    set_pattern_measurement(loop, PatternKind::SequentialForward,
                            PatternOperation::Read, std::move(measurement));
    statistics.loop_results.push_back(loop);
  }
  PatternResults skipped_loop;
  PatternMeasurement skipped;
  skipped.status = PatternMeasurementStatus::Skipped;
  skipped.status_reason = "not supported";
  set_pattern_measurement(skipped_loop, PatternKind::SequentialForward,
                          PatternOperation::Read, std::move(skipped));
  statistics.loop_results.push_back(skipped_loop);

  const PatternResults headline = extract_pattern_median_results(statistics);
  const PatternMeasurement& measurement = get_pattern_measurement(
      headline, PatternKind::SequentialForward, PatternOperation::Read);
  ASSERT_TRUE(measurement.bandwidth_gb_s.has_value());
  EXPECT_DOUBLE_EQ(*measurement.bandwidth_gb_s, 20.0);
  EXPECT_DOUBLE_EQ(headline.forward_read_bw, 20.0);
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
  EXPECT_NE(error_output.find("Noisy pattern measurement"), std::string::npos);
}

TEST(PatternBenchmarkTest, SkippedPatternsDoNotEnterStatisticsIntegration) {
  BenchmarkConfig config = make_pattern_config(512 * 1024, 1);
  BenchmarkBuffers buffers;
  ASSERT_TRUE(allocate_and_initialize_buffers(config, buffers));
  PatternStatistics statistics;

  ASSERT_EQ(run_all_pattern_benchmarks(buffers, config, statistics), EXIT_SUCCESS);
  ASSERT_EQ(statistics.loop_results.size(), 1u);
  EXPECT_TRUE(statistics.all_strided_2mb_read_bw.empty());
  EXPECT_TRUE(statistics.all_strided_2mb_write_bw.empty());
  EXPECT_TRUE(statistics.all_strided_2mb_copy_bw.empty());
  EXPECT_EQ(get_pattern_measurement(statistics.loop_results.front(),
                                    PatternKind::Strided2MiB,
                                    PatternOperation::Read)
                .status,
            PatternMeasurementStatus::Skipped);
}

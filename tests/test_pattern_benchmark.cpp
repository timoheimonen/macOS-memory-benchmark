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
#include <gtest/gtest.h>
#include "pattern_benchmark/pattern_benchmark.h"
#include "core/memory/buffer_manager.h"
#include "core/config/config.h"
#include "utils/benchmark.h"  // Declares system_info functions
#include "core/config/constants.h"
#include "test_config_helpers.h"
#include <cstdlib>
#include <cstring>
#include <algorithm>

namespace {

BenchmarkConfig make_pattern_config(size_t buffer_size, int iterations, int num_threads = 1) {
  BenchmarkConfig config;
  config.buffer_size = buffer_size;
  config.iterations = iterations;
  initialize_system_info(config);
  config.num_threads = num_threads;
  return config;
}

::testing::AssertionResult run_pattern_benchmarks_checked(BenchmarkBuffers& buffers,
                                                          const BenchmarkConfig& config,
                                                          PatternResults& results) {
  const int run_result = run_pattern_benchmarks(buffers, config, results);
  if (run_result != EXIT_SUCCESS) {
    return ::testing::AssertionFailure()
           << "run_pattern_benchmarks(buffers, config, results) failed with code " << run_result;
  }

  return ::testing::AssertionSuccess();
}

::testing::AssertionResult run_pattern_benchmarks_with_fresh_buffers(BenchmarkConfig& config,
                                                                      PatternResults& results) {
  BenchmarkBuffers buffers;
  const ::testing::AssertionResult alloc_init_result = allocate_and_initialize_buffers(config, buffers);
  if (!alloc_init_result) {
    return alloc_init_result;
  }

  return run_pattern_benchmarks_checked(buffers, config, results);
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
}

void expect_2mb_pattern_bandwidths_positive(const PatternResults& results) {
  EXPECT_GT(results.strided_2mb_read_bw, 0.0);
  EXPECT_GT(results.strided_2mb_write_bw, 0.0);
  EXPECT_GT(results.strided_2mb_copy_bw, 0.0);
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

// Integration test: Test pattern benchmarks with minimal configuration
// NOTE: This is an integration test that performs actual system operations.
// It runs real pattern benchmarks which may be slower and can fail on slow systems or under load.
// Use 'make test-integration' to run integration tests, or 'make test' for unit tests only.
TEST(PatternBenchmarkTest, RunPatternBenchmarksMinimalIntegration) {
  BenchmarkConfig config = make_pattern_config(512 * 1024, 1);
  PatternResults results;

  ASSERT_TRUE(run_pattern_benchmarks_with_fresh_buffers(config, results));

  expect_core_pattern_bandwidths_positive(results);
  expect_2mb_pattern_bandwidths_zero(results);
}

// Test pattern benchmarks with multiple iterations
TEST(PatternBenchmarkTest, RunPatternBenchmarksMultipleIterations) {
  BenchmarkConfig config = make_pattern_config(128 * 1024, 3);
  PatternResults results;
  ASSERT_TRUE(run_pattern_benchmarks_with_fresh_buffers(config, results));
  
  // All results should be positive
  EXPECT_GT(results.forward_read_bw, 0.0);
  EXPECT_GT(results.random_copy_bw, 0.0);
}

// Test that forward pattern is baseline (typically fastest)
TEST(PatternBenchmarkTest, ForwardPatternBaseline) {
  BenchmarkConfig config = make_pattern_config(256 * 1024, 2);
  PatternResults results;
  ASSERT_TRUE(run_pattern_benchmarks_with_fresh_buffers(config, results));
  
  // All patterns should produce valid results
  EXPECT_GT(results.forward_read_bw, 0.0);
  EXPECT_GT(results.reverse_read_bw, 0.0);
  EXPECT_GT(results.strided_64_read_bw, 0.0);
  EXPECT_GT(results.strided_4096_read_bw, 0.0);
  EXPECT_GT(results.strided_16384_read_bw, 0.0);
  EXPECT_GT(results.random_read_bw, 0.0);
}

// Test pattern benchmarks with different buffer sizes
TEST(PatternBenchmarkTest, RunPatternBenchmarksDifferentSizes) {
  BenchmarkConfig config1 = make_pattern_config(64 * 1024, 1);
  BenchmarkConfig config2 = make_pattern_config(512 * 1024, 1);

  PatternResults results1, results2;
  ASSERT_TRUE(run_pattern_benchmarks_with_fresh_buffers(config1, results1));
  ASSERT_TRUE(run_pattern_benchmarks_with_fresh_buffers(config2, results2));
  
  // Both should produce valid results
  EXPECT_GT(results1.forward_read_bw, 0.0);
  EXPECT_GT(results2.forward_read_bw, 0.0);
}

// Test that random indices are generated correctly (indirectly through benchmarks)
// This tests that random pattern benchmarks complete successfully, which implies
// random indices were generated within valid bounds
TEST(PatternBenchmarkTest, RandomIndicesGeneration) {
  BenchmarkConfig config = make_pattern_config(1024 * 1024, 1);
  PatternResults results;
  ASSERT_TRUE(run_pattern_benchmarks_with_fresh_buffers(config, results));
  
  // Random pattern should complete successfully, implying indices were valid
  EXPECT_GT(results.random_read_bw, 0.0);
  EXPECT_GT(results.random_write_bw, 0.0);
  EXPECT_GT(results.random_copy_bw, 0.0);
  
  // Random bandwidth should be reasonable (not zero, not unreasonably high)
  // With small buffers and fast memory, bandwidth can be high
  EXPECT_LT(results.random_read_bw, 10000.0);  // Less than 10000 GB/s
  EXPECT_LT(results.random_write_bw, 10000.0);
  EXPECT_LT(results.random_copy_bw, 10000.0);
}

// Test pattern benchmarks with very small buffer (edge case)
TEST(PatternBenchmarkTest, RunPatternBenchmarksSmallBuffer) {
  BenchmarkConfig config = make_pattern_config(1024, 1);
  PatternResults results;
  ASSERT_TRUE(run_pattern_benchmarks_with_fresh_buffers(config, results));
  
  // Results should be valid (may be small but should be > 0)
  EXPECT_GT(results.forward_read_bw, 0.0);
}

// Test that strided patterns use correct stride values
// 64B stride should access cache lines, 4096B stride should access pages
TEST(PatternBenchmarkTest, StridedPatternStrides) {
  BenchmarkConfig config = make_pattern_config(512 * 1024, 2);
  PatternResults results;
  ASSERT_TRUE(run_pattern_benchmarks_with_fresh_buffers(config, results));
  
  // Both strided patterns should complete successfully
  EXPECT_GT(results.strided_64_read_bw, 0.0);
  EXPECT_GT(results.strided_4096_read_bw, 0.0);
  EXPECT_GT(results.strided_16384_read_bw, 0.0);
  
  // Verify both are reasonable (not zero, not unreasonably high)
  // With small buffers and fast memory, bandwidth can be very high
  EXPECT_LT(results.strided_64_read_bw, 10000.0);  // Less than 10000 GB/s
  EXPECT_LT(results.strided_4096_read_bw, 10000.0);
  EXPECT_LT(results.strided_16384_read_bw, 10000.0);
}

// Test large-page stride variants with a larger buffer
TEST(PatternBenchmarkTest, StridedLargePagePatterns) {
  BenchmarkConfig config = make_pattern_config(8 * 1024 * 1024, 1);

  PatternResults results;
  ASSERT_TRUE(run_pattern_benchmarks_with_fresh_buffers(config, results));

  EXPECT_GT(results.strided_16384_read_bw, 0.0);
  EXPECT_GT(results.strided_16384_write_bw, 0.0);
  EXPECT_GT(results.strided_16384_copy_bw, 0.0);

  expect_2mb_pattern_bandwidths_positive(results);
}

// Test that all pattern types produce results
TEST(PatternBenchmarkTest, AllPatternTypesComplete) {
  BenchmarkConfig config = make_pattern_config(512 * 1024, 1);
  PatternResults results;
  ASSERT_TRUE(run_pattern_benchmarks_with_fresh_buffers(config, results));

  expect_core_pattern_bandwidths_positive(results);
  expect_2mb_pattern_bandwidths_zero(results);
}

// Test that pattern results are consistent across multiple runs
// (Results may vary but should be in reasonable range)
TEST(PatternBenchmarkTest, PatternResultsConsistency) {
  BenchmarkConfig config = make_pattern_config(256 * 1024, 2);
  
  BenchmarkBuffers buffers;
  ASSERT_TRUE(allocate_and_initialize_buffers(config, buffers));
  
  PatternResults results1, results2;

  ASSERT_TRUE(run_pattern_benchmarks_checked(buffers, config, results1));
  ASSERT_TRUE(run_pattern_benchmarks_checked(buffers, config, results2));
  
  // Both runs should produce valid results
  EXPECT_GT(results1.forward_read_bw, 0.0);
  EXPECT_GT(results2.forward_read_bw, 0.0);
  
  // Results should be in reasonable range (within 10x of each other)
  // This accounts for system variability while ensuring they're not wildly different
  double ratio = results1.forward_read_bw / results2.forward_read_bw;
  EXPECT_GT(ratio, 0.1);
  EXPECT_LT(ratio, 10.0);
}

// Test that copy operations use both source and destination buffers
TEST(PatternBenchmarkTest, CopyOperationsUseBothBuffers) {
  BenchmarkConfig config = make_pattern_config(512 * 1024, 1);
  
  BenchmarkBuffers buffers;
  ASSERT_TRUE(allocate_and_initialize_buffers(config, buffers));
  
  // Verify buffers are different
  EXPECT_NE(buffers.src_buffer(), buffers.dst_buffer());
  
  PatternResults results;
  ASSERT_TRUE(run_pattern_benchmarks_checked(buffers, config, results));
  
  // Copy operations should complete successfully
  EXPECT_GT(results.forward_copy_bw, 0.0);
  EXPECT_GT(results.reverse_copy_bw, 0.0);
  EXPECT_GT(results.strided_64_copy_bw, 0.0);
  EXPECT_GT(results.strided_4096_copy_bw, 0.0);
  EXPECT_GT(results.strided_16384_copy_bw, 0.0);
  EXPECT_EQ(results.strided_2mb_copy_bw, 0.0);
  EXPECT_GT(results.random_copy_bw, 0.0);
}

// Test strided pattern with buffer smaller than stride - should skip gracefully
TEST(PatternBenchmarkTest, StridedPatternSkippedWhenBufferTooSmall) {
  using namespace Constants;
  
  BenchmarkConfig config = make_pattern_config(PATTERN_STRIDE_PAGE - 1, 1);
  
  PatternResults results;
  ASSERT_TRUE(run_pattern_benchmarks_with_fresh_buffers(config, results));
  
  // Strided 4096B should be skipped (buffer too small)
  // When skipped, bandwidth remains 0.0
  EXPECT_EQ(results.strided_4096_read_bw, 0.0);
  EXPECT_EQ(results.strided_4096_write_bw, 0.0);
  EXPECT_EQ(results.strided_4096_copy_bw, 0.0);
  EXPECT_EQ(results.strided_16384_read_bw, 0.0);
  EXPECT_EQ(results.strided_16384_write_bw, 0.0);
  EXPECT_EQ(results.strided_16384_copy_bw, 0.0);
  EXPECT_EQ(results.strided_2mb_read_bw, 0.0);
  EXPECT_EQ(results.strided_2mb_write_bw, 0.0);
  EXPECT_EQ(results.strided_2mb_copy_bw, 0.0);
  
  // Strided 64B may work or be skipped depending on effective buffer size
  // The important thing is that the test completes successfully
  SUCCEED();
}

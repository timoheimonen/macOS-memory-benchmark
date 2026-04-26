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
#include "pattern_benchmark/pattern_benchmark.h"
#include "core/memory/buffer_manager.h"
#include "core/config/config.h"
#include "utils/benchmark.h"  // Declares system_info functions
#include "core/config/constants.h"
#include "test_config_helpers.h"
#include <cstdlib>

namespace {

BenchmarkConfig make_pattern_config(size_t buffer_size, int iterations, int num_threads = 1) {
  BenchmarkConfig config;
  config.buffer_size = buffer_size;
  config.iterations = iterations;
  initialize_system_info(config);
  config.num_threads = num_threads;
  return config;
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

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
#include "core/config/constants.h"
#include "pattern_benchmark/pattern_benchmark.h"
#include "core/memory/buffer_manager.h"
#include "core/config/config.h"
#include "utils/benchmark.h"
#include "test_config_helpers.h"
#include <iostream>
#include <cstdlib>

// Shared helper: configure for pattern-only mode with minimal buffers
// run_patterns=true skips latency/cache buffer allocation and initialization,
// allowing tests to use buffer sizes below LATENCY_STRIDE_BYTES * 2.
static BenchmarkConfig make_pattern_validation_config(size_t buffer_size) {
  using namespace Constants;
  BenchmarkConfig config;
  config.buffer_size = buffer_size;
  config.l1_buffer_size = 0;
  config.l2_buffer_size = 0;
  config.use_custom_cache_size = false;
  config.iterations = 1;
  config.num_threads = 1;
  config.run_patterns = true;
  initialize_system_info(config);
  return config;
}

// ============================================================================
// 64B (cache line) stride boundary tests
//
// calculate_strided_params subtracts PATTERN_ACCESS_SIZE_BYTES (32) from
// buffer_size before comparing with stride. So strided 64B pattern produces
// results only when buffer_size >= 96 (= 64 + 32). Below 96, the pattern is
// gracefully skipped (bw=0, EXIT_SUCCESS).
// ============================================================================

// buffer_size=95 = stride+31: 64B strided pattern skipped, other patterns succeed
TEST(PatternValidationTest, ValidateStride64SkippedBelowEffectiveBoundary) {
  using namespace Constants;
  auto config = make_pattern_validation_config(PATTERN_STRIDE_CACHE_LINE + PATTERN_ACCESS_SIZE_BYTES - 1);

  BenchmarkBuffers buffers;
  ASSERT_TRUE(allocate_and_initialize_buffers(config, buffers));

  PatternResults results;
  EXPECT_EQ(run_pattern_benchmarks(buffers, config, results), EXIT_SUCCESS);
  EXPECT_GT(results.forward_read_bw, 0.0);
  EXPECT_EQ(results.strided_64_read_bw, 0.0);
  EXPECT_EQ(results.strided_4096_read_bw, 0.0);
}

// buffer_size=96 = stride+32: 64B strided pattern produces results
TEST(PatternValidationTest, ValidateStride64AtEffectiveBoundary) {
  using namespace Constants;
  auto config = make_pattern_validation_config(PATTERN_STRIDE_CACHE_LINE + PATTERN_ACCESS_SIZE_BYTES);

  BenchmarkBuffers buffers;
  ASSERT_TRUE(allocate_and_initialize_buffers(config, buffers));

  PatternResults results;
  EXPECT_EQ(run_pattern_benchmarks(buffers, config, results), EXIT_SUCCESS);
  EXPECT_GT(results.strided_64_read_bw, 0.0);
  EXPECT_EQ(results.strided_4096_read_bw, 0.0);
}

// ============================================================================
// Page (4096B) stride boundary tests
//
// Strided 4096B pattern produces results only when buffer_size >= 4128 (= 4096 + 32).
// ============================================================================

// buffer_size=4095 < stride: validate_stride fails (silently skips)
TEST(PatternValidationTest, ValidateStridePageBelowStrideBoundary) {
  using namespace Constants;
  auto config = make_pattern_validation_config(PATTERN_STRIDE_PAGE - 1);

  BenchmarkBuffers buffers;
  ASSERT_TRUE(allocate_and_initialize_buffers(config, buffers));

  PatternResults results;
  EXPECT_EQ(run_pattern_benchmarks(buffers, config, results), EXIT_SUCCESS);
  EXPECT_EQ(results.strided_4096_read_bw, 0.0);
  EXPECT_EQ(results.strided_16384_read_bw, 0.0);
  EXPECT_EQ(results.strided_2mb_read_bw, 0.0);
}

// buffer_size=4096 == stride: validate passes but calculate_strided_params skips (effective < stride)
TEST(PatternValidationTest, ValidateStridePageAtStrideBoundary) {
  using namespace Constants;
  auto config = make_pattern_validation_config(PATTERN_STRIDE_PAGE);

  BenchmarkBuffers buffers;
  ASSERT_TRUE(allocate_and_initialize_buffers(config, buffers));

  PatternResults results;
  EXPECT_EQ(run_pattern_benchmarks(buffers, config, results), EXIT_SUCCESS);
  EXPECT_EQ(results.strided_4096_read_bw, 0.0);
  EXPECT_EQ(results.strided_16384_read_bw, 0.0);
  EXPECT_EQ(results.strided_2mb_read_bw, 0.0);
}

// buffer_size=4128 = stride+32: 4096B strided pattern produces results
TEST(PatternValidationTest, ValidateStridePageAtEffectiveBoundary) {
  using namespace Constants;
  auto config = make_pattern_validation_config(PATTERN_STRIDE_PAGE + PATTERN_ACCESS_SIZE_BYTES);

  BenchmarkBuffers buffers;
  ASSERT_TRUE(allocate_and_initialize_buffers(config, buffers));

  PatternResults results;
  EXPECT_EQ(run_pattern_benchmarks(buffers, config, results), EXIT_SUCCESS);
  EXPECT_GT(results.strided_4096_read_bw, 0.0);
  EXPECT_EQ(results.strided_16384_read_bw, 0.0);
  EXPECT_EQ(results.strided_2mb_read_bw, 0.0);
}

// ============================================================================
// 16KB page stride boundary tests
// ============================================================================

// buffer_size=16416 = 16K+32: 16K strided pattern produces results
TEST(PatternValidationTest, ValidateStride16KAtEffectiveBoundary) {
  using namespace Constants;
  auto config = make_pattern_validation_config(PATTERN_STRIDE_PAGE_16K + PATTERN_ACCESS_SIZE_BYTES);

  BenchmarkBuffers buffers;
  ASSERT_TRUE(allocate_and_initialize_buffers(config, buffers));

  PatternResults results;
  EXPECT_EQ(run_pattern_benchmarks(buffers, config, results), EXIT_SUCCESS);
  EXPECT_GT(results.strided_4096_read_bw, 0.0);
  EXPECT_GT(results.strided_16384_read_bw, 0.0);
  EXPECT_EQ(results.strided_2mb_read_bw, 0.0);
}

// ============================================================================
// 2MB superpage stride boundary test
// ============================================================================

// buffer_size=2MB+32: 2MB strided pattern produces results
TEST(PatternValidationTest, ValidateStrideSuperpage2MBAtEffectiveBoundary) {
  using namespace Constants;
  auto config = make_pattern_validation_config(PATTERN_STRIDE_SUPERPAGE_2MB + PATTERN_ACCESS_SIZE_BYTES);

  BenchmarkBuffers buffers;
  ASSERT_TRUE(allocate_and_initialize_buffers(config, buffers));

  PatternResults results;
  EXPECT_EQ(run_pattern_benchmarks(buffers, config, results), EXIT_SUCCESS);
  EXPECT_GT(results.strided_64_read_bw, 0.0);
  EXPECT_GT(results.strided_4096_read_bw, 0.0);
  EXPECT_GT(results.strided_16384_read_bw, 0.0);
  EXPECT_GT(results.strided_2mb_read_bw, 0.0);
}

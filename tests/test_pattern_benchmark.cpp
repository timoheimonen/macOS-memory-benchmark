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
#include <vector>

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

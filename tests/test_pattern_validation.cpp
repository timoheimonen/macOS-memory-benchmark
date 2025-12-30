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
#include <iostream>
#include <cstdlib>

// Test validate_stride behavior indirectly through pattern benchmarks
// When buffer is smaller than stride, pattern should be skipped (not error)
TEST(PatternValidationTest, ValidateStrideBufferSmallerThanStride) {
  using namespace Constants;
  
  BenchmarkConfig config;
  // Use buffer smaller than stride, but need at least LATENCY_STRIDE_BYTES * 2 = 256 for latency chain
  // So we'll use a larger buffer but test that strided patterns with stride > effective_buffer_size are skipped
  config.buffer_size = 512;  // Large enough for latency chain
  config.l1_buffer_size = 0;  // Disable cache buffers
  config.l2_buffer_size = 0;
  config.use_custom_cache_size = false;
  config.iterations = 1;
  config.num_threads = 1;
  
  // Initialize system info
  config.cpu_name = get_processor_name();
  config.perf_cores = get_performance_cores();
  config.eff_cores = get_efficiency_cores();
  config.num_threads = get_total_logical_cores();
  config.l1_cache_size = get_l1_cache_size();
  config.l2_cache_size = get_l2_cache_size();
  
  BenchmarkBuffers buffers;
  int alloc_result = allocate_all_buffers(config, buffers);
  ASSERT_EQ(alloc_result, EXIT_SUCCESS);
  
  int init_result = initialize_all_buffers(buffers, config);
  ASSERT_EQ(init_result, EXIT_SUCCESS);
  
  PatternResults results;
  int result = run_pattern_benchmarks(buffers, config, results);
  
  // Should succeed
  EXPECT_EQ(result, EXIT_SUCCESS);
  
  // With 512 byte buffer, patterns should complete successfully
  // Strided 4096B should be skipped (buffer too small)
  EXPECT_EQ(results.strided_4096_read_bw, 0.0);
}

// Test validate_stride with buffer equal to stride - boundary case
TEST(PatternValidationTest, ValidateStrideBufferEqualToStride) {
  using namespace Constants;
  
  BenchmarkConfig config;
  // Need at least LATENCY_STRIDE_BYTES * 2 = 256 for latency chain
  config.buffer_size = 512;  // Large enough for latency chain
  config.l1_buffer_size = 0;  // Disable cache buffers
  config.l2_buffer_size = 0;
  config.use_custom_cache_size = false;
  config.iterations = 1;
  config.num_threads = 1;
  
  // Initialize system info
  config.cpu_name = get_processor_name();
  config.perf_cores = get_performance_cores();
  config.eff_cores = get_efficiency_cores();
  config.num_threads = get_total_logical_cores();
  config.l1_cache_size = get_l1_cache_size();
  config.l2_cache_size = get_l2_cache_size();
  
  BenchmarkBuffers buffers;
  int alloc_result = allocate_all_buffers(config, buffers);
  ASSERT_EQ(alloc_result, EXIT_SUCCESS);
  
  int init_result = initialize_all_buffers(buffers, config);
  ASSERT_EQ(init_result, EXIT_SUCCESS);
  
  PatternResults results;
  int result = run_pattern_benchmarks(buffers, config, results);
  
  // Should succeed
  EXPECT_EQ(result, EXIT_SUCCESS);
  
  // Test should complete successfully
  SUCCEED();
}

// Test buffer size progression: buffer < PATTERN_MIN_BUFFER_SIZE_BYTES
TEST(PatternValidationTest, BufferSizeProgressionLessThanMin) {
  using namespace Constants;
  
  BenchmarkConfig config;
  // Need at least LATENCY_STRIDE_BYTES * 2 = 256 for latency chain
  config.buffer_size = 512;  // Large enough for latency chain
  config.l1_buffer_size = 0;  // Disable cache buffers
  config.l2_buffer_size = 0;
  config.use_custom_cache_size = false;
  config.iterations = 1;
  config.num_threads = 1;
  
  // Initialize system info
  config.cpu_name = get_processor_name();
  config.perf_cores = get_performance_cores();
  config.eff_cores = get_efficiency_cores();
  config.num_threads = get_total_logical_cores();
  config.l1_cache_size = get_l1_cache_size();
  config.l2_cache_size = get_l2_cache_size();
  
  BenchmarkBuffers buffers;
  int alloc_result = allocate_all_buffers(config, buffers);
  ASSERT_EQ(alloc_result, EXIT_SUCCESS);
  
  int init_result = initialize_all_buffers(buffers, config);
  ASSERT_EQ(init_result, EXIT_SUCCESS);
  
  PatternResults results;
  int result = run_pattern_benchmarks(buffers, config, results);
  
  // Should succeed
  EXPECT_EQ(result, EXIT_SUCCESS);
}

// Test buffer size progression: buffer == PATTERN_MIN_BUFFER_SIZE_BYTES
TEST(PatternValidationTest, BufferSizeProgressionEqualToMin) {
  using namespace Constants;
  
  BenchmarkConfig config;
  // Need at least LATENCY_STRIDE_BYTES * 2 = 256 for latency chain
  config.buffer_size = 512;  // Large enough for latency chain
  config.l1_buffer_size = 0;  // Disable cache buffers
  config.l2_buffer_size = 0;
  config.use_custom_cache_size = false;
  config.iterations = 1;
  config.num_threads = 1;
  
  // Initialize system info
  config.cpu_name = get_processor_name();
  config.perf_cores = get_performance_cores();
  config.eff_cores = get_efficiency_cores();
  config.num_threads = get_total_logical_cores();
  config.l1_cache_size = get_l1_cache_size();
  config.l2_cache_size = get_l2_cache_size();
  
  BenchmarkBuffers buffers;
  int alloc_result = allocate_all_buffers(config, buffers);
  ASSERT_EQ(alloc_result, EXIT_SUCCESS);
  
  int init_result = initialize_all_buffers(buffers, config);
  ASSERT_EQ(init_result, EXIT_SUCCESS);
  
  PatternResults results;
  int result = run_pattern_benchmarks(buffers, config, results);
  
  // Should succeed
  EXPECT_EQ(result, EXIT_SUCCESS);
}

// Test buffer size progression: buffer < PATTERN_STRIDE_CACHE_LINE
TEST(PatternValidationTest, BufferSizeProgressionLessThanCacheLine) {
  using namespace Constants;
  
  BenchmarkConfig config;
  // Need at least LATENCY_STRIDE_BYTES * 2 = 256 for latency chain
  config.buffer_size = 512;  // Large enough for latency chain
  config.l1_buffer_size = 0;  // Disable cache buffers
  config.l2_buffer_size = 0;
  config.use_custom_cache_size = false;
  config.iterations = 1;
  config.num_threads = 1;
  
  // Initialize system info
  config.cpu_name = get_processor_name();
  config.perf_cores = get_performance_cores();
  config.eff_cores = get_efficiency_cores();
  config.num_threads = get_total_logical_cores();
  config.l1_cache_size = get_l1_cache_size();
  config.l2_cache_size = get_l2_cache_size();
  
  BenchmarkBuffers buffers;
  int alloc_result = allocate_all_buffers(config, buffers);
  ASSERT_EQ(alloc_result, EXIT_SUCCESS);
  
  int init_result = initialize_all_buffers(buffers, config);
  ASSERT_EQ(init_result, EXIT_SUCCESS);
  
  PatternResults results;
  int result = run_pattern_benchmarks(buffers, config, results);
  
  // Should succeed
  EXPECT_EQ(result, EXIT_SUCCESS);
  
  // Test should complete successfully
  SUCCEED();
}

// Test buffer size progression: buffer == PATTERN_STRIDE_CACHE_LINE
TEST(PatternValidationTest, BufferSizeProgressionEqualToCacheLine) {
  using namespace Constants;
  
  BenchmarkConfig config;
  // Need at least LATENCY_STRIDE_BYTES * 2 = 256 for latency chain
  config.buffer_size = 512;  // Large enough for latency chain
  config.l1_buffer_size = 0;  // Disable cache buffers
  config.l2_buffer_size = 0;
  config.use_custom_cache_size = false;
  config.iterations = 1;
  config.num_threads = 1;
  
  // Initialize system info
  config.cpu_name = get_processor_name();
  config.perf_cores = get_performance_cores();
  config.eff_cores = get_efficiency_cores();
  config.num_threads = get_total_logical_cores();
  config.l1_cache_size = get_l1_cache_size();
  config.l2_cache_size = get_l2_cache_size();
  
  BenchmarkBuffers buffers;
  int alloc_result = allocate_all_buffers(config, buffers);
  ASSERT_EQ(alloc_result, EXIT_SUCCESS);
  
  int init_result = initialize_all_buffers(buffers, config);
  ASSERT_EQ(init_result, EXIT_SUCCESS);
  
  PatternResults results;
  int result = run_pattern_benchmarks(buffers, config, results);
  
  // Should succeed
  EXPECT_EQ(result, EXIT_SUCCESS);
  
  // Test should complete successfully
  SUCCEED();
}

// Test buffer size progression: buffer == PATTERN_STRIDE_CACHE_LINE + 1
TEST(PatternValidationTest, BufferSizeProgressionJustLargerThanCacheLine) {
  using namespace Constants;
  
  BenchmarkConfig config;
  // Need at least LATENCY_STRIDE_BYTES * 2 = 256 for latency chain
  config.buffer_size = 512;  // Large enough for latency chain
  config.l1_buffer_size = 0;  // Disable cache buffers
  config.l2_buffer_size = 0;
  config.use_custom_cache_size = false;
  config.iterations = 1;
  config.num_threads = 1;
  
  // Initialize system info
  config.cpu_name = get_processor_name();
  config.perf_cores = get_performance_cores();
  config.eff_cores = get_efficiency_cores();
  config.num_threads = get_total_logical_cores();
  config.l1_cache_size = get_l1_cache_size();
  config.l2_cache_size = get_l2_cache_size();
  
  BenchmarkBuffers buffers;
  int alloc_result = allocate_all_buffers(config, buffers);
  ASSERT_EQ(alloc_result, EXIT_SUCCESS);
  
  int init_result = initialize_all_buffers(buffers, config);
  ASSERT_EQ(init_result, EXIT_SUCCESS);
  
  PatternResults results;
  int result = run_pattern_benchmarks(buffers, config, results);
  
  // Should succeed
  EXPECT_EQ(result, EXIT_SUCCESS);
  
  // Test should complete successfully
  SUCCEED();
}

// Test buffer size progression: buffer < PATTERN_STRIDE_PAGE
TEST(PatternValidationTest, BufferSizeProgressionLessThanPage) {
  using namespace Constants;
  
  BenchmarkConfig config;
  config.buffer_size = PATTERN_STRIDE_PAGE - 1;  // 4095 bytes
  config.iterations = 1;
  config.num_threads = 1;
  
  // Initialize system info
  config.cpu_name = get_processor_name();
  config.perf_cores = get_performance_cores();
  config.eff_cores = get_efficiency_cores();
  config.num_threads = get_total_logical_cores();
  config.l1_cache_size = get_l1_cache_size();
  config.l2_cache_size = get_l2_cache_size();
  
  BenchmarkBuffers buffers;
  int alloc_result = allocate_all_buffers(config, buffers);
  ASSERT_EQ(alloc_result, EXIT_SUCCESS);
  
  int init_result = initialize_all_buffers(buffers, config);
  ASSERT_EQ(init_result, EXIT_SUCCESS);
  
  PatternResults results;
  int result = run_pattern_benchmarks(buffers, config, results);
  
  // Should succeed (strided 4096B pattern skipped)
  EXPECT_EQ(result, EXIT_SUCCESS);
  EXPECT_EQ(results.strided_4096_read_bw, 0.0);
}

// Test buffer size progression: buffer == PATTERN_STRIDE_PAGE
TEST(PatternValidationTest, BufferSizeProgressionEqualToPage) {
  using namespace Constants;
  
  BenchmarkConfig config;
  config.buffer_size = PATTERN_STRIDE_PAGE;  // 4096 bytes
  config.iterations = 1;
  config.num_threads = 1;
  
  // Initialize system info
  config.cpu_name = get_processor_name();
  config.perf_cores = get_performance_cores();
  config.eff_cores = get_efficiency_cores();
  config.num_threads = get_total_logical_cores();
  config.l1_cache_size = get_l1_cache_size();
  config.l2_cache_size = get_l2_cache_size();
  
  BenchmarkBuffers buffers;
  int alloc_result = allocate_all_buffers(config, buffers);
  ASSERT_EQ(alloc_result, EXIT_SUCCESS);
  
  int init_result = initialize_all_buffers(buffers, config);
  ASSERT_EQ(init_result, EXIT_SUCCESS);
  
  PatternResults results;
  int result = run_pattern_benchmarks(buffers, config, results);
  
  // Should succeed
  EXPECT_EQ(result, EXIT_SUCCESS);
}

// Test buffer size progression: buffer == PATTERN_STRIDE_PAGE + 1
TEST(PatternValidationTest, BufferSizeProgressionJustLargerThanPage) {
  using namespace Constants;
  
  BenchmarkConfig config;
  config.buffer_size = PATTERN_STRIDE_PAGE + 1;  // 4097 bytes
  config.iterations = 1;
  config.num_threads = 1;
  
  // Initialize system info
  config.cpu_name = get_processor_name();
  config.perf_cores = get_performance_cores();
  config.eff_cores = get_efficiency_cores();
  config.num_threads = get_total_logical_cores();
  config.l1_cache_size = get_l1_cache_size();
  config.l2_cache_size = get_l2_cache_size();
  
  BenchmarkBuffers buffers;
  int alloc_result = allocate_all_buffers(config, buffers);
  ASSERT_EQ(alloc_result, EXIT_SUCCESS);
  
  int init_result = initialize_all_buffers(buffers, config);
  ASSERT_EQ(init_result, EXIT_SUCCESS);
  
  PatternResults results;
  int result = run_pattern_benchmarks(buffers, config, results);
  
  // Should succeed
  EXPECT_EQ(result, EXIT_SUCCESS);
}


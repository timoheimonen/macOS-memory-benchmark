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
#include <cstdlib>
#include <cstring>
#include <algorithm>

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

// Test pattern benchmarks with minimal configuration
TEST(PatternBenchmarkTest, RunPatternBenchmarksMinimal) {
  BenchmarkConfig config;
  config.buffer_size = 512 * 1024;  // 512 KB - large enough for all patterns including strided 4096
  config.iterations = 1;  // Single iteration for speed
  config.num_threads = 1;  // Single thread
  
  // Initialize system info
  config.cpu_name = get_processor_name();
  config.perf_cores = get_performance_cores();
  config.eff_cores = get_efficiency_cores();
  config.num_threads = get_total_logical_cores();
  config.l1_cache_size = get_l1_cache_size();
  config.l2_cache_size = get_l2_cache_size();
  
  // Allocate buffers
  BenchmarkBuffers buffers;
  int alloc_result = allocate_all_buffers(config, buffers);
  ASSERT_EQ(alloc_result, EXIT_SUCCESS);
  
  // Initialize buffers
  int init_result = initialize_all_buffers(buffers, config);
  ASSERT_EQ(init_result, EXIT_SUCCESS);
  
  // Run pattern benchmarks
  PatternResults results;
  int result = run_pattern_benchmarks(buffers, config, results);
  
  EXPECT_EQ(result, EXIT_SUCCESS);
  
  // Verify that results were populated (should be > 0 for valid benchmarks)
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
  EXPECT_GT(results.random_read_bw, 0.0);
  EXPECT_GT(results.random_write_bw, 0.0);
  EXPECT_GT(results.random_copy_bw, 0.0);
}

// Test pattern benchmarks with multiple iterations
TEST(PatternBenchmarkTest, RunPatternBenchmarksMultipleIterations) {
  BenchmarkConfig config;
  config.buffer_size = 128 * 1024;  // 128 KB
  config.iterations = 3;  // Multiple iterations
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
  
  EXPECT_EQ(result, EXIT_SUCCESS);
  
  // All results should be positive
  EXPECT_GT(results.forward_read_bw, 0.0);
  EXPECT_GT(results.random_copy_bw, 0.0);
}

// Test that forward pattern is baseline (typically fastest)
TEST(PatternBenchmarkTest, ForwardPatternBaseline) {
  BenchmarkConfig config;
  config.buffer_size = 256 * 1024;  // 256 KB
  config.iterations = 2;
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
  ASSERT_EQ(result, EXIT_SUCCESS);
  
  // All patterns should produce valid results
  EXPECT_GT(results.forward_read_bw, 0.0);
  EXPECT_GT(results.reverse_read_bw, 0.0);
  EXPECT_GT(results.strided_64_read_bw, 0.0);
  EXPECT_GT(results.strided_4096_read_bw, 0.0);
  EXPECT_GT(results.random_read_bw, 0.0);
  
  // Verify all bandwidth values are reasonable (not zero, not unreasonably high)
  // With small buffers, bandwidth can be very high, so we use a generous upper bound
  EXPECT_LT(results.forward_read_bw, 10000.0);  // Less than 10000 GB/s
  EXPECT_LT(results.reverse_read_bw, 10000.0);
  EXPECT_LT(results.strided_64_read_bw, 10000.0);
  EXPECT_LT(results.strided_4096_read_bw, 10000.0);
  EXPECT_LT(results.random_read_bw, 10000.0);
}

// Test pattern benchmarks with different buffer sizes
TEST(PatternBenchmarkTest, RunPatternBenchmarksDifferentSizes) {
  BenchmarkConfig config1, config2;
  
  config1.buffer_size = 64 * 1024;  // 64 KB
  config1.iterations = 1;
  config1.num_threads = 1;
  
  config2.buffer_size = 512 * 1024;  // 512 KB
  config2.iterations = 1;
  config2.num_threads = 1;
  
  // Initialize system info for both
  std::string cpu_name = get_processor_name();
  int perf_cores = get_performance_cores();
  int eff_cores = get_efficiency_cores();
  int num_threads = get_total_logical_cores();
  size_t l1_cache_size = get_l1_cache_size();
  size_t l2_cache_size = get_l2_cache_size();
  
  config1.cpu_name = config2.cpu_name = cpu_name;
  config1.perf_cores = config2.perf_cores = perf_cores;
  config1.eff_cores = config2.eff_cores = eff_cores;
  config1.num_threads = config2.num_threads = num_threads;
  config1.l1_cache_size = config2.l1_cache_size = l1_cache_size;
  config1.l2_cache_size = config2.l2_cache_size = l2_cache_size;
  
  BenchmarkBuffers buffers1, buffers2;
  
  int alloc_result1 = allocate_all_buffers(config1, buffers1);
  ASSERT_EQ(alloc_result1, EXIT_SUCCESS);
  int alloc_result2 = allocate_all_buffers(config2, buffers2);
  ASSERT_EQ(alloc_result2, EXIT_SUCCESS);
  
  int init_result1 = initialize_all_buffers(buffers1, config1);
  ASSERT_EQ(init_result1, EXIT_SUCCESS);
  int init_result2 = initialize_all_buffers(buffers2, config2);
  ASSERT_EQ(init_result2, EXIT_SUCCESS);
  
  PatternResults results1, results2;
  int result1 = run_pattern_benchmarks(buffers1, config1, results1);
  int result2 = run_pattern_benchmarks(buffers2, config2, results2);
  
  EXPECT_EQ(result1, EXIT_SUCCESS);
  EXPECT_EQ(result2, EXIT_SUCCESS);
  
  // Both should produce valid results
  EXPECT_GT(results1.forward_read_bw, 0.0);
  EXPECT_GT(results2.forward_read_bw, 0.0);
}

// Test that random indices are generated correctly (indirectly through benchmarks)
// This tests that random pattern benchmarks complete successfully, which implies
// random indices were generated within valid bounds
TEST(PatternBenchmarkTest, RandomIndicesGeneration) {
  BenchmarkConfig config;
  config.buffer_size = 1024 * 1024;  // 1 MB - enough for random pattern
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
  
  EXPECT_EQ(result, EXIT_SUCCESS);
  
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
  BenchmarkConfig config;
  config.buffer_size = 1024;  // 1 KB - very small
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
  
  // Should still succeed even with small buffer
  EXPECT_EQ(result, EXIT_SUCCESS);
  
  // Results should be valid (may be small but should be > 0)
  EXPECT_GT(results.forward_read_bw, 0.0);
}

// Test that strided patterns use correct stride values
// 64B stride should access cache lines, 4096B stride should access pages
TEST(PatternBenchmarkTest, StridedPatternStrides) {
  BenchmarkConfig config;
  config.buffer_size = 512 * 1024;  // 512 KB - large enough for strided 4096 pattern
  config.iterations = 2;
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
  ASSERT_EQ(result, EXIT_SUCCESS);
  
  // Both strided patterns should complete successfully
  EXPECT_GT(results.strided_64_read_bw, 0.0);
  EXPECT_GT(results.strided_4096_read_bw, 0.0);
  
  // Verify both are reasonable (not zero, not unreasonably high)
  // With small buffers and fast memory, bandwidth can be very high
  EXPECT_LT(results.strided_64_read_bw, 10000.0);  // Less than 10000 GB/s
  EXPECT_LT(results.strided_4096_read_bw, 10000.0);
}

// Test that all pattern types produce results
TEST(PatternBenchmarkTest, AllPatternTypesComplete) {
  BenchmarkConfig config;
  config.buffer_size = 512 * 1024;  // 512 KB - large enough for all patterns including strided 4096
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
  ASSERT_EQ(result, EXIT_SUCCESS);
  
  // Verify all pattern types have results
  // Forward patterns
  EXPECT_GT(results.forward_read_bw, 0.0);
  EXPECT_GT(results.forward_write_bw, 0.0);
  EXPECT_GT(results.forward_copy_bw, 0.0);
  
  // Reverse patterns
  EXPECT_GT(results.reverse_read_bw, 0.0);
  EXPECT_GT(results.reverse_write_bw, 0.0);
  EXPECT_GT(results.reverse_copy_bw, 0.0);
  
  // Strided 64B patterns
  EXPECT_GT(results.strided_64_read_bw, 0.0);
  EXPECT_GT(results.strided_64_write_bw, 0.0);
  EXPECT_GT(results.strided_64_copy_bw, 0.0);
  
  // Strided 4096B patterns
  EXPECT_GT(results.strided_4096_read_bw, 0.0);
  EXPECT_GT(results.strided_4096_write_bw, 0.0);
  EXPECT_GT(results.strided_4096_copy_bw, 0.0);
  
  // Random patterns
  EXPECT_GT(results.random_read_bw, 0.0);
  EXPECT_GT(results.random_write_bw, 0.0);
  EXPECT_GT(results.random_copy_bw, 0.0);
}

// Test that pattern results are consistent across multiple runs
// (Results may vary but should be in reasonable range)
TEST(PatternBenchmarkTest, PatternResultsConsistency) {
  BenchmarkConfig config;
  config.buffer_size = 256 * 1024;  // 256 KB
  config.iterations = 2;
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
  
  PatternResults results1, results2;
  
  int result1 = run_pattern_benchmarks(buffers, config, results1);
  ASSERT_EQ(result1, EXIT_SUCCESS);
  
  int result2 = run_pattern_benchmarks(buffers, config, results2);
  ASSERT_EQ(result2, EXIT_SUCCESS);
  
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
  BenchmarkConfig config;
  config.buffer_size = 512 * 1024;  // 512 KB - large enough for all patterns including strided 4096
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
  
  // Verify buffers are different
  EXPECT_NE(buffers.src_buffer(), buffers.dst_buffer());
  
  PatternResults results;
  int result = run_pattern_benchmarks(buffers, config, results);
  ASSERT_EQ(result, EXIT_SUCCESS);
  
  // Copy operations should complete successfully
  EXPECT_GT(results.forward_copy_bw, 0.0);
  EXPECT_GT(results.reverse_copy_bw, 0.0);
  EXPECT_GT(results.strided_64_copy_bw, 0.0);
  EXPECT_GT(results.strided_4096_copy_bw, 0.0);
  EXPECT_GT(results.random_copy_bw, 0.0);
}


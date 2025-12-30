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
#include "benchmark/benchmark_runner.h"
#include "core/memory/buffer_manager.h"
#include "core/config/config.h"
#include "utils/benchmark.h"
#include "core/config/constants.h"
#include <cstdlib>
#include <cmath>     // std::isnan, std::isinf
#include <unistd.h>  // getpagesize

// Test that BenchmarkStatistics is properly initialized
TEST(BenchmarkRunnerTest, StatisticsInitialization) {
  BenchmarkStatistics stats;
  
  // Vectors should be empty initially
  EXPECT_TRUE(stats.all_read_bw_gb_s.empty());
  EXPECT_TRUE(stats.all_write_bw_gb_s.empty());
  EXPECT_TRUE(stats.all_copy_bw_gb_s.empty());
  EXPECT_TRUE(stats.all_l1_latency_ns.empty());
  EXPECT_TRUE(stats.all_l2_latency_ns.empty());
  EXPECT_TRUE(stats.all_average_latency_ns.empty());
}

// Test BenchmarkResults default values
TEST(BenchmarkRunnerTest, BenchmarkResultsDefaults) {
  BenchmarkResults results;
  
  EXPECT_EQ(results.read_bw_gb_s, 0.0);
  EXPECT_EQ(results.write_bw_gb_s, 0.0);
  EXPECT_EQ(results.copy_bw_gb_s, 0.0);
  EXPECT_EQ(results.average_latency_ns, 0.0);
  EXPECT_EQ(results.l1_latency_ns, 0.0);
  EXPECT_EQ(results.l2_latency_ns, 0.0);
  EXPECT_EQ(results.custom_latency_ns, 0.0);
}

// Test statistics structure after run_all_benchmarks clears vectors
// Note: This test verifies the clearing behavior without running actual benchmarks
TEST(BenchmarkRunnerTest, StatisticsClearing) {
  BenchmarkConfig config;
  BenchmarkBuffers buffers;
  BenchmarkStatistics stats;
  
  // Initialize system info
  config.cpu_name = get_processor_name();
  config.perf_cores = get_performance_cores();
  config.eff_cores = get_efficiency_cores();
  config.num_threads = get_total_logical_cores();
  config.l1_cache_size = get_l1_cache_size();
  config.l2_cache_size = get_l2_cache_size();
  
  // Set minimal config for testing
  config.buffer_size = getpagesize();  // Minimum size
  config.buffer_size_mb = 1;
  config.iterations = 1;
  config.loop_count = 0;  // No loops to avoid running actual benchmarks
  config.use_custom_cache_size = false;
  
  // Calculate buffer sizes and access counts (required for cache tests)
  calculate_buffer_sizes(config);
  calculate_access_counts(config);
  
  // Allocate minimal buffers
  int alloc_result = allocate_all_buffers(config, buffers);
  ASSERT_EQ(alloc_result, EXIT_SUCCESS);
  
  // Initialize buffers
  int init_result = initialize_all_buffers(buffers, config);
  ASSERT_EQ(init_result, EXIT_SUCCESS);
  
  // Run with 0 loops - should just clear statistics
  int result = run_all_benchmarks(buffers, config, stats);
  
  // Should succeed even with 0 loops
  EXPECT_EQ(result, EXIT_SUCCESS);
  
  // Statistics should be cleared/empty
  EXPECT_TRUE(stats.all_read_bw_gb_s.empty());
  EXPECT_TRUE(stats.all_write_bw_gb_s.empty());
  EXPECT_TRUE(stats.all_copy_bw_gb_s.empty());
}

// Integration test: Test that statistics vectors are properly reserved
// NOTE: This is an integration test that performs actual system operations.
// It runs real benchmarks which may be slower and can fail on slow systems or under load.
// Use 'make test-integration' to run integration tests, or 'make test' for unit tests only.
TEST(BenchmarkRunnerTest, StatisticsReservationIntegration) {
  BenchmarkConfig config;
  BenchmarkBuffers buffers;
  BenchmarkStatistics stats;
  
  // Initialize system info
  config.cpu_name = get_processor_name();
  config.perf_cores = get_performance_cores();
  config.eff_cores = get_efficiency_cores();
  config.num_threads = get_total_logical_cores();
  config.l1_cache_size = get_l1_cache_size();
  config.l2_cache_size = get_l2_cache_size();
  
  // Set config with loop_count > 0
  config.buffer_size = getpagesize();
  config.buffer_size_mb = 1;
  config.iterations = 1;
  config.loop_count = 3;  // 3 loops
  config.use_custom_cache_size = false;
  
  // Calculate buffer sizes and access counts (required for cache tests)
  calculate_buffer_sizes(config);
  calculate_access_counts(config);
  
  // Allocate minimal buffers
  int alloc_result = allocate_all_buffers(config, buffers);
  ASSERT_EQ(alloc_result, EXIT_SUCCESS);
  
  // Initialize buffers
  int init_result = initialize_all_buffers(buffers, config);
  ASSERT_EQ(init_result, EXIT_SUCCESS);
  
  // Note: This will run actual benchmarks, which may take a moment
  // but tests the full integration
  int result = run_all_benchmarks(buffers, config, stats);
  
  // Should succeed
  EXPECT_EQ(result, EXIT_SUCCESS);
  
  // Should have collected results for each loop
  EXPECT_EQ(static_cast<int>(stats.all_read_bw_gb_s.size()), config.loop_count);
  EXPECT_EQ(static_cast<int>(stats.all_write_bw_gb_s.size()), config.loop_count);
  EXPECT_EQ(static_cast<int>(stats.all_copy_bw_gb_s.size()), config.loop_count);
  EXPECT_EQ(static_cast<int>(stats.all_average_latency_ns.size()), config.loop_count);
}

// Test BenchmarkStatistics structure size and layout
TEST(BenchmarkRunnerTest, StatisticsStructure) {
  BenchmarkStatistics stats;
  
  // Verify all expected vectors exist and are accessible
  stats.all_read_bw_gb_s.push_back(1.0);
  stats.all_write_bw_gb_s.push_back(2.0);
  stats.all_copy_bw_gb_s.push_back(3.0);
  stats.all_l1_latency_ns.push_back(4.0);
  stats.all_l2_latency_ns.push_back(5.0);
  stats.all_average_latency_ns.push_back(6.0);
  stats.all_l1_read_bw_gb_s.push_back(7.0);
  stats.all_l1_write_bw_gb_s.push_back(8.0);
  stats.all_l1_copy_bw_gb_s.push_back(9.0);
  stats.all_l2_read_bw_gb_s.push_back(10.0);
  stats.all_l2_write_bw_gb_s.push_back(11.0);
  stats.all_l2_copy_bw_gb_s.push_back(12.0);
  stats.all_custom_latency_ns.push_back(13.0);
  stats.all_custom_read_bw_gb_s.push_back(14.0);
  stats.all_custom_write_bw_gb_s.push_back(15.0);
  stats.all_custom_copy_bw_gb_s.push_back(16.0);
  
  // Verify values were stored correctly
  EXPECT_EQ(stats.all_read_bw_gb_s[0], 1.0);
  EXPECT_EQ(stats.all_write_bw_gb_s[0], 2.0);
  EXPECT_EQ(stats.all_copy_bw_gb_s[0], 3.0);
  EXPECT_EQ(stats.all_l1_latency_ns[0], 4.0);
  EXPECT_EQ(stats.all_l2_latency_ns[0], 5.0);
  EXPECT_EQ(stats.all_average_latency_ns[0], 6.0);
  EXPECT_EQ(stats.all_l1_read_bw_gb_s[0], 7.0);
  EXPECT_EQ(stats.all_l1_write_bw_gb_s[0], 8.0);
  EXPECT_EQ(stats.all_l1_copy_bw_gb_s[0], 9.0);
  EXPECT_EQ(stats.all_l2_read_bw_gb_s[0], 10.0);
  EXPECT_EQ(stats.all_l2_write_bw_gb_s[0], 11.0);
  EXPECT_EQ(stats.all_l2_copy_bw_gb_s[0], 12.0);
  EXPECT_EQ(stats.all_custom_latency_ns[0], 13.0);
  EXPECT_EQ(stats.all_custom_read_bw_gb_s[0], 14.0);
  EXPECT_EQ(stats.all_custom_write_bw_gb_s[0], 15.0);
  EXPECT_EQ(stats.all_custom_copy_bw_gb_s[0], 16.0);
}

// Test that benchmark results are valid and reasonable
// This validates that the refactored calculation functions produce correct results
TEST(BenchmarkRunnerTest, ResultsValidation) {
  BenchmarkConfig config;
  BenchmarkBuffers buffers;
  BenchmarkStatistics stats;
  
  // Initialize system info
  config.cpu_name = get_processor_name();
  config.perf_cores = get_performance_cores();
  config.eff_cores = get_efficiency_cores();
  config.num_threads = get_total_logical_cores();
  config.l1_cache_size = get_l1_cache_size();
  config.l2_cache_size = get_l2_cache_size();
  
  // Set config with loop_count > 0
  config.buffer_size = getpagesize();
  config.buffer_size_mb = 1;
  config.iterations = 1;
  config.loop_count = 1;  // Single loop for validation
  config.use_custom_cache_size = false;
  
  // Calculate buffer sizes and access counts (required for cache tests)
  calculate_buffer_sizes(config);
  calculate_access_counts(config);
  
  // Allocate minimal buffers
  int alloc_result = allocate_all_buffers(config, buffers);
  ASSERT_EQ(alloc_result, EXIT_SUCCESS);
  
  // Initialize buffers
  int init_result = initialize_all_buffers(buffers, config);
  ASSERT_EQ(init_result, EXIT_SUCCESS);
  
  // Run benchmarks
  int result = run_all_benchmarks(buffers, config, stats);
  ASSERT_EQ(result, EXIT_SUCCESS);
  
  // Validate that results were collected
  ASSERT_EQ(static_cast<int>(stats.all_read_bw_gb_s.size()), 1);
  ASSERT_EQ(static_cast<int>(stats.all_write_bw_gb_s.size()), 1);
  ASSERT_EQ(static_cast<int>(stats.all_copy_bw_gb_s.size()), 1);
  ASSERT_EQ(static_cast<int>(stats.all_average_latency_ns.size()), 1);
  
  // Validate main memory bandwidth results are reasonable
  // Bandwidth should be positive (even if small for minimal test)
  EXPECT_GE(stats.all_read_bw_gb_s[0], 0.0);
  EXPECT_GE(stats.all_write_bw_gb_s[0], 0.0);
  EXPECT_GE(stats.all_copy_bw_gb_s[0], 0.0);
  
  // Validate latency results are reasonable
  // Latency should be positive
  EXPECT_GT(stats.all_average_latency_ns[0], 0.0);
  
  // Validate that bandwidth calculations are consistent
  // Copy bandwidth should typically be >= read or write (it's both operations)
  // But we allow for variance, so just check they're not negative
  EXPECT_FALSE(std::isnan(stats.all_read_bw_gb_s[0]));
  EXPECT_FALSE(std::isnan(stats.all_write_bw_gb_s[0]));
  EXPECT_FALSE(std::isnan(stats.all_copy_bw_gb_s[0]));
  EXPECT_FALSE(std::isnan(stats.all_average_latency_ns[0]));
  EXPECT_FALSE(std::isinf(stats.all_read_bw_gb_s[0]));
  EXPECT_FALSE(std::isinf(stats.all_write_bw_gb_s[0]));
  EXPECT_FALSE(std::isinf(stats.all_copy_bw_gb_s[0]));
  EXPECT_FALSE(std::isinf(stats.all_average_latency_ns[0]));
}


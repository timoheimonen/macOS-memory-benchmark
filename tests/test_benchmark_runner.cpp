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
#include "benchmark/benchmark_executor.h"
#include "benchmark/benchmark_results.h"
#include "benchmark/benchmark_runner.h"
#include "benchmark/benchmark_statistics_collector.h"
#include "core/memory/buffer_manager.h"
#include "core/config/config.h"
#include "output/json/json_output/json_output_api.h"
#include "utils/benchmark.h"
#include "core/config/constants.h"
#include "test_config_helpers.h"
#include "test_statistics_helpers.h"
#include <cstdlib>
#include <cmath>     // std::isnan, std::isinf
#include <limits>
#include <vector>
#include <unistd.h>  // getpagesize

namespace {

constexpr size_t kDecimalGbBytes = 1000ULL * 1000ULL * 1000ULL;
constexpr size_t kHundredMbBytes = 100ULL * 1000ULL * 1000ULL;
constexpr size_t kTwoHundredMbBytes = 200ULL * 1000ULL * 1000ULL;

void expect_double_vector_eq(const std::vector<double>& actual, const std::vector<double>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_DOUBLE_EQ(actual[i], expected[i]) << "index " << i;
  }
}

BenchmarkConfig make_collector_config() {
  BenchmarkConfig config;
  config.loop_count = 2;
  config.latency_sample_count = 3;
  config.buffer_size = 4096;
  config.lat_num_accesses = 16;
  config.l1_buffer_size = 1024;
  config.l2_buffer_size = 2048;
  return config;
}

BenchmarkResults make_collector_results() {
  BenchmarkResults results;
  results.read_bw_gb_s = 10.0;
  results.write_bw_gb_s = 20.0;
  results.copy_bw_gb_s = 30.0;
  results.average_latency_ns = 40.0;
  results.has_auto_tlb_breakdown = true;
  results.tlb_hit_latency_ns = 41.0;
  results.tlb_miss_latency_ns = 90.0;
  results.page_walk_penalty_ns = 49.0;
  results.latency_samples = {40.1, 40.2};

  results.l1_latency_ns = 5.0;
  results.l1_read_bw_gb_s = 50.0;
  results.l1_write_bw_gb_s = 60.0;
  results.l1_copy_bw_gb_s = 70.0;
  results.l1_latency_samples = {5.1, 5.2};

  results.l2_latency_ns = 8.0;
  results.l2_read_bw_gb_s = 80.0;
  results.l2_write_bw_gb_s = 90.0;
  results.l2_copy_bw_gb_s = 100.0;
  results.l2_latency_samples = {8.1, 8.2};

  results.custom_latency_ns = 12.0;
  results.custom_read_bw_gb_s = 110.0;
  results.custom_write_bw_gb_s = 120.0;
  results.custom_copy_bw_gb_s = 130.0;
  results.custom_latency_samples = {12.1, 12.2};

  return results;
}

}  // namespace

TEST(BenchmarkResultsTest, CalculateSingleBandwidthUsesDecimalGbAndCopyMultiplier) {
  double read_bw = 0.0;
  double write_bw = 0.0;
  double copy_bw = 0.0;

  calculate_single_bandwidth(kDecimalGbBytes, 2, 2.0, 4.0, 1.0, read_bw, write_bw, copy_bw);

  EXPECT_DOUBLE_EQ(read_bw, 1.0);
  EXPECT_DOUBLE_EQ(write_bw, 0.5);
  EXPECT_DOUBLE_EQ(copy_bw, 4.0);
}

TEST(BenchmarkResultsTest, CalculateSingleBandwidthRejectsInvalidInputs) {
  double read_bw = 1.0;
  double write_bw = 1.0;
  double copy_bw = 1.0;

  calculate_single_bandwidth(kDecimalGbBytes, 1, 0.0, -1.0, std::numeric_limits<double>::infinity(),
                             read_bw, write_bw, copy_bw);

  EXPECT_EQ(read_bw, 0.0);
  EXPECT_EQ(write_bw, 0.0);
  EXPECT_EQ(copy_bw, 0.0);

  calculate_single_bandwidth(kDecimalGbBytes, 0, 1.0, 1.0, 1.0, read_bw, write_bw, copy_bw);

  EXPECT_EQ(read_bw, 0.0);
  EXPECT_EQ(write_bw, 0.0);
  EXPECT_EQ(copy_bw, 0.0);
}

TEST(BenchmarkResultsTest, CalculateSingleBandwidthHandlesCopyByteCountOverflow) {
  const size_t large_buffer = (std::numeric_limits<size_t>::max() / Constants::COPY_OPERATION_MULTIPLIER) + 1;
  double read_bw = 0.0;
  double write_bw = 0.0;
  double copy_bw = 0.0;

  calculate_single_bandwidth(large_buffer, 1, 1.0, 1.0, 1.0, read_bw, write_bw, copy_bw);

  ASSERT_GT(read_bw, 0.0);
  EXPECT_DOUBLE_EQ(write_bw, read_bw);
  EXPECT_NEAR(copy_bw, read_bw * Constants::COPY_OPERATION_MULTIPLIER, read_bw * 1e-12);
}

TEST(BenchmarkResultsTest, CalculateBandwidthResultsPopulatesMainAndDetectedCacheLevels) {
  BenchmarkConfig config;
  config.buffer_size = kDecimalGbBytes;
  config.iterations = 2;
  config.use_custom_cache_size = false;
  config.l1_buffer_size = kHundredMbBytes;
  config.l2_buffer_size = kTwoHundredMbBytes;

  TimingResults timings;
  timings.total_read_time = 2.0;
  timings.total_write_time = 4.0;
  timings.total_copy_time = 1.0;
  timings.l1_read_time = 1.0;
  timings.l1_write_time = 2.0;
  timings.l1_copy_time = 1.0;
  timings.l2_read_time = 1.0;
  timings.l2_write_time = 2.0;
  timings.l2_copy_time = 1.0;

  BenchmarkResults results;
  calculate_bandwidth_results(config, timings, results);

  EXPECT_DOUBLE_EQ(results.read_bw_gb_s, 1.0);
  EXPECT_DOUBLE_EQ(results.write_bw_gb_s, 0.5);
  EXPECT_DOUBLE_EQ(results.copy_bw_gb_s, 4.0);

  EXPECT_DOUBLE_EQ(results.l1_read_bw_gb_s, 2.0);
  EXPECT_DOUBLE_EQ(results.l1_write_bw_gb_s, 1.0);
  EXPECT_DOUBLE_EQ(results.l1_copy_bw_gb_s, 4.0);

  EXPECT_DOUBLE_EQ(results.l2_read_bw_gb_s, 4.0);
  EXPECT_DOUBLE_EQ(results.l2_write_bw_gb_s, 2.0);
  EXPECT_DOUBLE_EQ(results.l2_copy_bw_gb_s, 8.0);

  EXPECT_EQ(results.custom_read_bw_gb_s, 0.0);
  EXPECT_EQ(results.custom_write_bw_gb_s, 0.0);
  EXPECT_EQ(results.custom_copy_bw_gb_s, 0.0);
}

TEST(BenchmarkResultsTest, CalculateBandwidthResultsUsesCustomCacheInsteadOfDetectedCaches) {
  BenchmarkConfig config;
  config.buffer_size = kDecimalGbBytes;
  config.iterations = 2;
  config.use_custom_cache_size = true;
  config.custom_buffer_size = kHundredMbBytes;
  config.l1_buffer_size = kHundredMbBytes;
  config.l2_buffer_size = kTwoHundredMbBytes;

  TimingResults timings;
  timings.custom_read_time = 1.0;
  timings.custom_write_time = 2.0;
  timings.custom_copy_time = 1.0;
  timings.l1_read_time = 1.0;
  timings.l2_read_time = 1.0;

  BenchmarkResults results;
  calculate_bandwidth_results(config, timings, results);

  EXPECT_DOUBLE_EQ(results.custom_read_bw_gb_s, 2.0);
  EXPECT_DOUBLE_EQ(results.custom_write_bw_gb_s, 1.0);
  EXPECT_DOUBLE_EQ(results.custom_copy_bw_gb_s, 4.0);
  EXPECT_EQ(results.l1_read_bw_gb_s, 0.0);
  EXPECT_EQ(results.l2_read_bw_gb_s, 0.0);
}

TEST(BenchmarkStatisticsCollectorTest, CollectLoopResultsAggregatesMainAndDetectedCacheMetrics) {
  BenchmarkConfig config = make_collector_config();
  BenchmarkStatistics stats;
  stats.all_read_bw_gb_s = {999.0};
  initialize_statistics(stats, config);

  collect_loop_results(stats, make_collector_results(), config);

  expect_double_vector_eq(stats.all_read_bw_gb_s, {10.0});
  expect_double_vector_eq(stats.all_write_bw_gb_s, {20.0});
  expect_double_vector_eq(stats.all_copy_bw_gb_s, {30.0});
  expect_double_vector_eq(stats.all_average_latency_ns, {40.0});
  expect_double_vector_eq(stats.all_tlb_hit_latency_ns, {41.0});
  expect_double_vector_eq(stats.all_tlb_miss_latency_ns, {90.0});
  expect_double_vector_eq(stats.all_page_walk_penalty_ns, {49.0});
  expect_double_vector_eq(stats.all_main_mem_latency_samples, {40.1, 40.2});

  expect_double_vector_eq(stats.all_l1_latency_ns, {5.0});
  expect_double_vector_eq(stats.all_l1_read_bw_gb_s, {50.0});
  expect_double_vector_eq(stats.all_l1_write_bw_gb_s, {60.0});
  expect_double_vector_eq(stats.all_l1_copy_bw_gb_s, {70.0});
  expect_double_vector_eq(stats.all_l1_latency_samples, {5.1, 5.2});

  expect_double_vector_eq(stats.all_l2_latency_ns, {8.0});
  expect_double_vector_eq(stats.all_l2_read_bw_gb_s, {80.0});
  expect_double_vector_eq(stats.all_l2_write_bw_gb_s, {90.0});
  expect_double_vector_eq(stats.all_l2_copy_bw_gb_s, {100.0});
  expect_double_vector_eq(stats.all_l2_latency_samples, {8.1, 8.2});
}

TEST(BenchmarkStatisticsCollectorTest, CollectLoopResultsUsesCustomCacheInsteadOfDetectedCaches) {
  BenchmarkConfig config = make_collector_config();
  config.use_custom_cache_size = true;
  config.custom_buffer_size = 4096;

  BenchmarkStatistics stats;
  initialize_statistics(stats, config);

  collect_loop_results(stats, make_collector_results(), config);

  expect_double_vector_eq(stats.all_custom_latency_ns, {12.0});
  expect_double_vector_eq(stats.all_custom_read_bw_gb_s, {110.0});
  expect_double_vector_eq(stats.all_custom_write_bw_gb_s, {120.0});
  expect_double_vector_eq(stats.all_custom_copy_bw_gb_s, {130.0});
  expect_double_vector_eq(stats.all_custom_latency_samples, {12.1, 12.2});

  EXPECT_TRUE(stats.all_l1_latency_ns.empty());
  EXPECT_TRUE(stats.all_l1_read_bw_gb_s.empty());
  EXPECT_TRUE(stats.all_l2_latency_ns.empty());
  EXPECT_TRUE(stats.all_l2_read_bw_gb_s.empty());
}

TEST(BenchmarkStatisticsCollectorTest, CollectLoopResultsSkipsMainLatencyWhenDisabled) {
  BenchmarkConfig config = make_collector_config();
  config.only_bandwidth = true;

  BenchmarkStatistics stats;
  initialize_statistics(stats, config);

  collect_loop_results(stats, make_collector_results(), config);

  expect_double_vector_eq(stats.all_read_bw_gb_s, {10.0});
  EXPECT_TRUE(stats.all_average_latency_ns.empty());
  EXPECT_TRUE(stats.all_tlb_hit_latency_ns.empty());
  EXPECT_TRUE(stats.all_tlb_miss_latency_ns.empty());
  EXPECT_TRUE(stats.all_page_walk_penalty_ns.empty());
  EXPECT_TRUE(stats.all_main_mem_latency_samples.empty());
}

// Test statistics structure after run_all_benchmarks clears vectors
// Note: This test verifies the clearing behavior without running actual benchmarks
TEST(BenchmarkRunnerTest, StatisticsClearing) {
  BenchmarkConfig config;
  BenchmarkBuffers buffers;
  BenchmarkStatistics stats;
  
  initialize_system_info(config);
  
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
  ASSERT_TRUE(allocate_and_initialize_buffers(config, buffers));
  
  // Run with 0 loops - should just clear statistics
  int result = run_all_benchmarks(buffers, config, stats);
  
  // Should succeed even with 0 loops
  EXPECT_EQ(result, EXIT_SUCCESS);
  
  // Statistics should be cleared/empty
  EXPECT_TRUE(stats.all_read_bw_gb_s.empty());
  EXPECT_TRUE(stats.all_write_bw_gb_s.empty());
  EXPECT_TRUE(stats.all_copy_bw_gb_s.empty());
  EXPECT_TRUE(stats.all_tlb_hit_latency_ns.empty());
  EXPECT_TRUE(stats.all_tlb_miss_latency_ns.empty());
  EXPECT_TRUE(stats.all_page_walk_penalty_ns.empty());
}

// Integration test: Test that statistics vectors are properly reserved
// NOTE: This is an integration test that performs actual system operations.
// It runs real benchmarks which may be slower and can fail on slow systems or under load.
// Use 'make test-integration' to run integration tests, or 'make test' for unit tests only.
TEST(BenchmarkRunnerTest, StatisticsReservationIntegration) {
  BenchmarkConfig config;
  BenchmarkBuffers buffers;
  BenchmarkStatistics stats;
  
  initialize_system_info(config);
  
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
  ASSERT_TRUE(allocate_and_initialize_buffers(config, buffers));
  
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
  EXPECT_EQ(static_cast<int>(stats.all_tlb_hit_latency_ns.size()), config.loop_count);
  EXPECT_EQ(static_cast<int>(stats.all_tlb_miss_latency_ns.size()), config.loop_count);
  EXPECT_EQ(static_cast<int>(stats.all_page_walk_penalty_ns.size()), config.loop_count);
}

// Test that benchmark results are valid and reasonable
// This validates that the refactored calculation functions produce correct results
TEST(BenchmarkRunnerTest, ResultsValidationIntegration) {
  BenchmarkConfig config;
  BenchmarkBuffers buffers;
  BenchmarkStatistics stats;
  
  initialize_system_info(config);
  
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
  ASSERT_TRUE(allocate_and_initialize_buffers(config, buffers));
  
  // Run benchmarks
  int result = run_all_benchmarks(buffers, config, stats);
  ASSERT_EQ(result, EXIT_SUCCESS);
  
  // Validate that results were collected
  ASSERT_EQ(static_cast<int>(stats.all_read_bw_gb_s.size()), 1);
  ASSERT_EQ(static_cast<int>(stats.all_write_bw_gb_s.size()), 1);
  ASSERT_EQ(static_cast<int>(stats.all_copy_bw_gb_s.size()), 1);
  ASSERT_EQ(static_cast<int>(stats.all_average_latency_ns.size()), 1);
  ASSERT_EQ(static_cast<int>(stats.all_tlb_hit_latency_ns.size()), 1);
  ASSERT_EQ(static_cast<int>(stats.all_tlb_miss_latency_ns.size()), 1);
  ASSERT_EQ(static_cast<int>(stats.all_page_walk_penalty_ns.size()), 1);
  
  // Validate main memory bandwidth results are reasonable
  // Bandwidth should be positive (even if small for minimal test)
  EXPECT_GE(stats.all_read_bw_gb_s[0], 0.0);
  EXPECT_GE(stats.all_write_bw_gb_s[0], 0.0);
  EXPECT_GE(stats.all_copy_bw_gb_s[0], 0.0);
  
  // Validate latency results are reasonable
  // Latency should be positive
  EXPECT_GT(stats.all_average_latency_ns[0], 0.0);
  EXPECT_GT(stats.all_tlb_hit_latency_ns[0], 0.0);
  EXPECT_GT(stats.all_tlb_miss_latency_ns[0], 0.0);
  
  // Validate that bandwidth calculations are consistent
  // Copy bandwidth should typically be >= read or write (it's both operations)
  // But we allow for variance, so just check they're not negative
  EXPECT_FALSE(std::isnan(stats.all_read_bw_gb_s[0]));
  EXPECT_FALSE(std::isnan(stats.all_write_bw_gb_s[0]));
  EXPECT_FALSE(std::isnan(stats.all_copy_bw_gb_s[0]));
  EXPECT_FALSE(std::isnan(stats.all_average_latency_ns[0]));
  EXPECT_FALSE(std::isnan(stats.all_tlb_hit_latency_ns[0]));
  EXPECT_FALSE(std::isnan(stats.all_tlb_miss_latency_ns[0]));
  EXPECT_FALSE(std::isnan(stats.all_page_walk_penalty_ns[0]));
  EXPECT_FALSE(std::isinf(stats.all_read_bw_gb_s[0]));
  EXPECT_FALSE(std::isinf(stats.all_write_bw_gb_s[0]));
  EXPECT_FALSE(std::isinf(stats.all_copy_bw_gb_s[0]));
  EXPECT_FALSE(std::isinf(stats.all_average_latency_ns[0]));
  EXPECT_FALSE(std::isinf(stats.all_tlb_hit_latency_ns[0]));
  EXPECT_FALSE(std::isinf(stats.all_tlb_miss_latency_ns[0]));
  EXPECT_FALSE(std::isinf(stats.all_page_walk_penalty_ns[0]));
}

TEST(BenchmarkRunnerTest, StatisticsPrintsAutoTlbBreakdownMetrics) {
  const std::vector<double> all_main_mem_latency = {15.0, 16.0};
  const std::vector<double> all_tlb_hit_latency = {14.0, 15.0};
  const std::vector<double> all_tlb_miss_latency = {90.0, 92.0};
  const std::vector<double> all_page_walk_penalty = {76.0, 77.0};

  const std::string output = test_statistics_helpers::capture_auto_tlb_breakdown(
      all_main_mem_latency, all_tlb_hit_latency, all_tlb_miss_latency, all_page_walk_penalty);

  EXPECT_NE(output.find("TLB Hit Latency (ns):"), std::string::npos);
  EXPECT_NE(output.find("TLB Miss Latency (ns):"), std::string::npos);
  EXPECT_NE(output.find("Estimated Page-Walk Penalty (ns):"), std::string::npos);
}

TEST(BenchmarkRunnerTest, MainMemoryJsonIncludesAutoTlbBreakdownWhenAvailable) {
  BenchmarkConfig config;
  config.only_bandwidth = false;
  config.only_latency = true;

  BenchmarkStatistics stats;
  stats.all_average_latency_ns = {15.10, 15.90};
  stats.all_main_mem_latency_samples = {15.20, 15.80, 16.00, 15.60};
  stats.all_tlb_hit_latency_ns = {15.10, 15.90};
  stats.all_tlb_miss_latency_ns = {95.10, 97.90};
  stats.all_page_walk_penalty_ns = {80.00, 82.00};

  const nlohmann::json main_memory_json = build_main_memory_json(config, stats);
  ASSERT_TRUE(main_memory_json.contains(JsonKeys::LATENCY));
  ASSERT_TRUE(main_memory_json[JsonKeys::LATENCY].contains(JsonKeys::AUTO_TLB_BREAKDOWN));

  const nlohmann::json auto_tlb_json = main_memory_json[JsonKeys::LATENCY][JsonKeys::AUTO_TLB_BREAKDOWN];
  ASSERT_TRUE(auto_tlb_json.contains(JsonKeys::TLB_HIT_NS));
  ASSERT_TRUE(auto_tlb_json.contains(JsonKeys::TLB_MISS_NS));
  ASSERT_TRUE(auto_tlb_json.contains(JsonKeys::PAGE_WALK_PENALTY_NS));

  ASSERT_EQ(auto_tlb_json[JsonKeys::TLB_HIT_NS][JsonKeys::VALUES].size(), 2u);
  ASSERT_EQ(auto_tlb_json[JsonKeys::TLB_MISS_NS][JsonKeys::VALUES].size(), 2u);
  ASSERT_EQ(auto_tlb_json[JsonKeys::PAGE_WALK_PENALTY_NS][JsonKeys::VALUES].size(), 2u);

  EXPECT_TRUE(auto_tlb_json[JsonKeys::TLB_HIT_NS].contains(JsonKeys::STATISTICS));
  EXPECT_TRUE(auto_tlb_json[JsonKeys::TLB_MISS_NS].contains(JsonKeys::STATISTICS));
  EXPECT_TRUE(auto_tlb_json[JsonKeys::PAGE_WALK_PENALTY_NS].contains(JsonKeys::STATISTICS));
}

TEST(BenchmarkRunnerTest, MainMemoryJsonOmitsAutoTlbBreakdownWhenUnavailable) {
  BenchmarkConfig config;
  config.only_bandwidth = false;
  config.only_latency = true;

  BenchmarkStatistics stats;
  stats.all_average_latency_ns = {15.10, 15.90};
  stats.all_main_mem_latency_samples = {15.20, 15.80, 16.00, 15.60};

  const nlohmann::json main_memory_json = build_main_memory_json(config, stats);
  ASSERT_TRUE(main_memory_json.contains(JsonKeys::LATENCY));
  EXPECT_FALSE(main_memory_json[JsonKeys::LATENCY].contains(JsonKeys::AUTO_TLB_BREAKDOWN));
}

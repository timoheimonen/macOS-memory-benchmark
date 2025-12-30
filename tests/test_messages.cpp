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
#include "output/console/messages.h"
#include "core/config/constants.h"
#include <string>
#include <sstream>
#include <tuple>

// ============================================================================
// Test Fixtures for Organization
// ============================================================================

// --- Error Messages Test Fixture ---
class MessagesErrorTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Common setup for error message tests if needed
  }
  
  void TearDown() override {
    // Common teardown for error message tests if needed
  }
};

// --- Warning Messages Test Fixture ---
class MessagesWarningTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Common setup for warning message tests if needed
  }
};

// --- Info Messages Test Fixture ---
class MessagesInfoTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Common setup for info message tests if needed
  }
};

// --- Formatting Test Fixture ---
class MessagesFormattingTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Common setup for formatting tests if needed
  }
};

// ============================================================================
// Parameterized Tests
// ============================================================================

// Parameterized test for error_missing_value
class MessagesErrorMissingValueTest : public ::testing::TestWithParam<std::string> {};

TEST_P(MessagesErrorMissingValueTest, ErrorMissingValue) {
  std::string option = GetParam();
  std::string msg = Messages::error_missing_value(option);
  std::string expected = "Missing value for " + option;
  EXPECT_EQ(msg, expected);
}

INSTANTIATE_TEST_SUITE_P(
  ErrorMissingValueVariants,
  MessagesErrorMissingValueTest,
  ::testing::Values("-iterations", "-buffersize", "-count", "-cache-size")
);

// Parameterized test for error_unknown_option
class MessagesErrorUnknownOptionTest : public ::testing::TestWithParam<std::string> {};

TEST_P(MessagesErrorUnknownOptionTest, ErrorUnknownOption) {
  std::string option = GetParam();
  std::string msg = Messages::error_unknown_option(option);
  std::string expected = "Unknown option: " + option;
  EXPECT_EQ(msg, expected);
}

INSTANTIATE_TEST_SUITE_P(
  ErrorUnknownOptionVariants,
  MessagesErrorUnknownOptionTest,
  ::testing::Values("-unknown", "--invalid", "-bad-flag")
);

// Parameterized test for error_invalid_value
class MessagesErrorInvalidValueTest : public ::testing::TestWithParam<std::tuple<std::string, std::string, std::string>> {};

TEST_P(MessagesErrorInvalidValueTest, ErrorInvalidValue) {
  auto [option, value, reason] = GetParam();
  std::string msg = Messages::error_invalid_value(option, value, reason);
  std::string expected = "Invalid value for " + option + ": " + value + " (" + reason + ")";
  EXPECT_EQ(msg, expected);
}

INSTANTIATE_TEST_SUITE_P(
  ErrorInvalidValueVariants,
  MessagesErrorInvalidValueTest,
  ::testing::Values(
    std::make_tuple("-iterations", "abc", "must be a number"),
    std::make_tuple("-cache-size", "-1", "must be positive"),
    std::make_tuple("-buffersize", "0", "must be greater than zero")
  )
);

// Parameterized test for error_mmap_failed
class MessagesErrorMmapFailedTest : public ::testing::TestWithParam<std::string> {};

TEST_P(MessagesErrorMmapFailedTest, ErrorMmapFailed) {
  std::string buffer_name = GetParam();
  std::string msg = Messages::error_mmap_failed(buffer_name);
  std::string expected = "mmap failed for " + buffer_name;
  EXPECT_EQ(msg, expected);
}

INSTANTIATE_TEST_SUITE_P(
  ErrorMmapFailedVariants,
  MessagesErrorMmapFailedTest,
  ::testing::Values("src_buffer", "dst_buffer", "lat_buffer")
);

// Parameterized test for error_benchmark_loop
class MessagesErrorBenchmarkLoopTest : public ::testing::TestWithParam<std::tuple<int, std::string>> {};

TEST_P(MessagesErrorBenchmarkLoopTest, ErrorBenchmarkLoop) {
  auto [loop, error] = GetParam();
  std::string msg = Messages::error_benchmark_loop(loop, error);
  EXPECT_NE(msg.find(std::to_string(loop)), std::string::npos);
  EXPECT_NE(msg.find(error), std::string::npos);
}

INSTANTIATE_TEST_SUITE_P(
  ErrorBenchmarkLoopVariants,
  MessagesErrorBenchmarkLoopTest,
  ::testing::Values(
    std::make_tuple(1, "memory error"),
    std::make_tuple(5, "timeout"),
    std::make_tuple(10, "allocation failed")
  )
);

// Parameterized test for warning_qos_failed
class MessagesWarningQosFailedTest : public ::testing::TestWithParam<int> {};

TEST_P(MessagesWarningQosFailedTest, WarningQosFailed) {
  int code = GetParam();
  std::string msg = Messages::warning_qos_failed(code);
  EXPECT_NE(msg.find("Failed to set QoS"), std::string::npos);
  EXPECT_NE(msg.find(std::to_string(code)), std::string::npos);
}

INSTANTIATE_TEST_SUITE_P(
  WarningQosFailedVariants,
  MessagesWarningQosFailedTest,
  ::testing::Values(1, 42, 100, -1)
);

// Parameterized test for statistics_metric_name
class MessagesStatisticsMetricNameTest : public ::testing::TestWithParam<std::string> {};

TEST_P(MessagesStatisticsMetricNameTest, StatisticsMetricName) {
  std::string metric_name = GetParam();
  std::string msg = Messages::statistics_metric_name(metric_name);
  std::string expected = metric_name + ":";
  EXPECT_EQ(msg, expected);
}

INSTANTIATE_TEST_SUITE_P(
  StatisticsMetricNameVariants,
  MessagesStatisticsMetricNameTest,
  ::testing::Values("Read Bandwidth", "Latency", "Write Bandwidth")
);

// ============================================================================
// Error Messages Tests (using fixture)
// ============================================================================

TEST_F(MessagesErrorTest, ErrorPrefix) {
  const std::string& prefix = Messages::error_prefix();
  EXPECT_EQ(prefix, "Error: ");
  // Test that it returns the same reference (static)
  EXPECT_EQ(&prefix, &Messages::error_prefix());
}

TEST_F(MessagesErrorTest, ErrorBufferSizeCalculation) {
  std::string msg = Messages::error_buffer_size_calculation(1024);
  EXPECT_NE(msg.find("1024"), std::string::npos);
  EXPECT_NE(msg.find("MB"), std::string::npos);
}

TEST_F(MessagesErrorTest, ErrorBufferSizeTooSmall) {
  std::string msg = Messages::error_buffer_size_too_small(1024);
  EXPECT_NE(msg.find("1024"), std::string::npos);
  EXPECT_NE(msg.find("bytes"), std::string::npos);
}

TEST_F(MessagesErrorTest, ErrorCacheSizeInvalid) {
  std::string msg = Messages::error_cache_size_invalid(16, 524288, 512);
  EXPECT_NE(msg.find("16"), std::string::npos);
  EXPECT_NE(msg.find("524288"), std::string::npos);
  EXPECT_NE(msg.find("512"), std::string::npos);
}

TEST_F(MessagesErrorTest, ErrorIterationsInvalid) {
  std::string msg = Messages::error_iterations_invalid(-5, 1, 2147483647);
  EXPECT_EQ(msg, "iterations invalid (must be between 1 and 2147483647, got -5)");
  // Test with different values
  std::string msg2 = Messages::error_iterations_invalid(0, 1, 100);
  EXPECT_EQ(msg2, "iterations invalid (must be between 1 and 100, got 0)");
}

TEST_F(MessagesErrorTest, ErrorBuffersizeInvalid) {
  std::string msg = Messages::error_buffersize_invalid(-100, 18446744073709551615UL);
  EXPECT_TRUE(msg.find("buffersize invalid") != std::string::npos);
  EXPECT_TRUE(msg.find("got -100") != std::string::npos);
  // Test with zero
  std::string msg2 = Messages::error_buffersize_invalid(0, 1000);
  EXPECT_EQ(msg2, "buffersize invalid (must be > 0 and <= 1000, got 0)");
}

TEST_F(MessagesErrorTest, ErrorCountInvalid) {
  std::string msg = Messages::error_count_invalid(0, 1, 2147483647);
  EXPECT_EQ(msg, "count invalid (must be between 1 and 2147483647, got 0)");
  // Test with negative value
  std::string msg2 = Messages::error_count_invalid(-10, 1, 1000);
  EXPECT_EQ(msg2, "count invalid (must be between 1 and 1000, got -10)");
}

TEST_F(MessagesErrorTest, ErrorLatencySamplesInvalid) {
  std::string msg = Messages::error_latency_samples_invalid(0, 1, 2147483647);
  EXPECT_EQ(msg, "latency-samples invalid (must be between 1 and 2147483647, got 0)");
  // Test with negative value
  std::string msg2 = Messages::error_latency_samples_invalid(-1, 1, 5000);
  EXPECT_EQ(msg2, "latency-samples invalid (must be between 1 and 5000, got -1)");
}

TEST_F(MessagesErrorTest, ErrorMadviseFailed) {
  std::string msg = Messages::error_madvise_failed("lat_buffer");
  EXPECT_EQ(msg, "madvise failed for lat_buffer");
}

TEST_F(MessagesErrorTest, ErrorBenchmarkTests) {
  std::string msg = Messages::error_benchmark_tests("test failure");
  EXPECT_EQ(msg, "Error during benchmark tests: test failure");
}

// ============================================================================
// Warning Messages Tests (using fixture)
// ============================================================================

TEST_F(MessagesWarningTest, WarningCannotGetMemory) {
  const std::string& msg = Messages::warning_cannot_get_memory();
  EXPECT_NE(msg.find("Cannot get"), std::string::npos);
  EXPECT_NE(msg.find("memory"), std::string::npos);
  EXPECT_EQ(&msg, &Messages::warning_cannot_get_memory());
}

TEST_F(MessagesWarningTest, WarningBufferSizeExceedsLimit) {
  std::string msg = Messages::warning_buffer_size_exceeds_limit(2048, 1024);
  EXPECT_NE(msg.find("2048"), std::string::npos);
  EXPECT_NE(msg.find("1024"), std::string::npos);
  EXPECT_NE(msg.find("Requested buffer size"), std::string::npos);
}

// ============================================================================
// Info Messages Tests (using fixture)
// ============================================================================

TEST_F(MessagesInfoTest, InfoSettingMaxFallback) {
  std::string msg = Messages::info_setting_max_fallback(2048);
  EXPECT_NE(msg.find("2048"), std::string::npos);
  EXPECT_NE(msg.find("MB"), std::string::npos);
  EXPECT_NE(msg.find("Info"), std::string::npos);
}

TEST_F(MessagesInfoTest, InfoCalculatedMaxLessThanMin) {
  std::string msg = Messages::info_calculated_max_less_than_min(512, 1024);
  EXPECT_NE(msg.find("512"), std::string::npos);
  EXPECT_NE(msg.find("1024"), std::string::npos);
}

TEST_F(MessagesInfoTest, InfoCustomCacheRoundedUp) {
  std::string msg = Messages::info_custom_cache_rounded_up(250, 256);
  EXPECT_NE(msg.find("250"), std::string::npos);
  EXPECT_NE(msg.find("256"), std::string::npos);
  EXPECT_NE(msg.find("KB"), std::string::npos);
}

// ============================================================================
// Main Program Messages Tests (using formatting fixture)
// ============================================================================

TEST_F(MessagesFormattingTest, MsgRunningBenchmarks) {
  const std::string& msg = Messages::msg_running_benchmarks();
  EXPECT_NE(msg.find("Running benchmarks"), std::string::npos);
  EXPECT_EQ(&msg, &Messages::msg_running_benchmarks());
}

TEST_F(MessagesFormattingTest, MsgDoneTotalTime) {
  std::string msg = Messages::msg_done_total_time(123.456);
  EXPECT_NE(msg.find("123.456"), std::string::npos);
  EXPECT_NE(msg.find("s"), std::string::npos);
  
  msg = Messages::msg_done_total_time(0.001);
  EXPECT_NE(msg.find("0.001"), std::string::npos);
}

// ============================================================================
// Usage/Help Messages Tests (using formatting fixture)
// ============================================================================

TEST_F(MessagesFormattingTest, UsageHeader) {
  std::string msg = Messages::usage_header("1.0.0");
  EXPECT_NE(msg.find("1.0.0"), std::string::npos);
  EXPECT_NE(msg.find("Timo Heimonen"), std::string::npos);
  EXPECT_NE(msg.find("GNU GPL"), std::string::npos);
  EXPECT_NE(msg.find("github.com"), std::string::npos);
}

TEST_F(MessagesFormattingTest, UsageOptions) {
  std::string msg = Messages::usage_options("memory_benchmark");
  EXPECT_NE(msg.find("memory_benchmark"), std::string::npos);
  EXPECT_NE(msg.find("-iterations"), std::string::npos);
  EXPECT_NE(msg.find("-buffersize"), std::string::npos);
  EXPECT_NE(msg.find("-count"), std::string::npos);
  EXPECT_NE(msg.find("-latency-samples"), std::string::npos);
  EXPECT_NE(msg.find("-cache-size"), std::string::npos);
  EXPECT_NE(msg.find("-h"), std::string::npos);
  // Check that default values are included
  EXPECT_NE(msg.find(std::to_string(Constants::DEFAULT_ITERATIONS)), std::string::npos);
  EXPECT_NE(msg.find(std::to_string(Constants::DEFAULT_BUFFER_SIZE_MB)), std::string::npos);
  EXPECT_NE(msg.find(std::to_string(Constants::DEFAULT_LOOP_COUNT)), std::string::npos);
  EXPECT_NE(msg.find(std::to_string(Constants::DEFAULT_LATENCY_SAMPLE_COUNT)), std::string::npos);
}

TEST_F(MessagesFormattingTest, UsageExample) {
  std::string msg = Messages::usage_example("memory_benchmark");
  EXPECT_NE(msg.find("memory_benchmark"), std::string::npos);
  EXPECT_NE(msg.find("-iterations"), std::string::npos);
  EXPECT_NE(msg.find("-buffersize"), std::string::npos);
  EXPECT_NE(msg.find("-output"), std::string::npos);
}

// ============================================================================
// Configuration Output Messages Tests (using formatting fixture)
// ============================================================================

TEST_F(MessagesFormattingTest, ConfigHeader) {
  std::string msg = Messages::config_header("1.0.0");
  EXPECT_NE(msg.find("1.0.0"), std::string::npos);
  EXPECT_NE(msg.find("macOS-memory-benchmark"), std::string::npos);
}

TEST_F(MessagesFormattingTest, ConfigCopyright) {
  std::string msg = Messages::config_copyright();
  EXPECT_NE(msg.find("2025"), std::string::npos);
  EXPECT_NE(msg.find("Timo Heimonen"), std::string::npos);
}

TEST_F(MessagesFormattingTest, ConfigLicense) {
  std::string msg = Messages::config_license();
  EXPECT_NE(msg.find("free software"), std::string::npos);
  EXPECT_NE(msg.find("GNU General Public License"), std::string::npos);
  EXPECT_NE(msg.find("version 3"), std::string::npos);
}

TEST_F(MessagesFormattingTest, ConfigBufferSize) {
  std::string msg = Messages::config_buffer_size(1024.5, 1024);
  EXPECT_NE(msg.find("1024.50"), std::string::npos);
  EXPECT_NE(msg.find("1024"), std::string::npos);
  EXPECT_NE(msg.find("MiB"), std::string::npos);
  EXPECT_NE(msg.find("MB"), std::string::npos);
}

TEST_F(MessagesFormattingTest, ConfigTotalAllocation) {
  std::string msg = Messages::config_total_allocation(3072.75);
  EXPECT_NE(msg.find("3072.75"), std::string::npos);
  EXPECT_NE(msg.find("MiB"), std::string::npos);
  EXPECT_NE(msg.find("3 buffers"), std::string::npos);
}

TEST_F(MessagesFormattingTest, ConfigIterations) {
  std::string msg = Messages::config_iterations(1000);
  EXPECT_NE(msg.find("1000"), std::string::npos);
  EXPECT_NE(msg.find("Iterations"), std::string::npos);
}

TEST_F(MessagesFormattingTest, ConfigLoopCount) {
  std::string msg = Messages::config_loop_count(5);
  EXPECT_NE(msg.find("5"), std::string::npos);
  EXPECT_NE(msg.find("Loop Count"), std::string::npos);
}

TEST_F(MessagesFormattingTest, ConfigNonCacheable) {
  // Test enabled
  std::string msg = Messages::config_non_cacheable(true);
  EXPECT_NE(msg.find("Non-Cacheable Memory Hints"), std::string::npos);
  EXPECT_NE(msg.find("Enabled"), std::string::npos);
  
  // Test disabled
  msg = Messages::config_non_cacheable(false);
  EXPECT_NE(msg.find("Non-Cacheable Memory Hints"), std::string::npos);
  EXPECT_NE(msg.find("Disabled"), std::string::npos);
}

TEST_F(MessagesFormattingTest, ConfigProcessorName) {
  std::string msg = Messages::config_processor_name("Apple M1");
  EXPECT_NE(msg.find("Apple M1"), std::string::npos);
  EXPECT_NE(msg.find("Processor Name"), std::string::npos);
}

TEST_F(MessagesFormattingTest, ConfigProcessorNameError) {
  std::string msg = Messages::config_processor_name_error();
  EXPECT_NE(msg.find("Could not retrieve"), std::string::npos);
}

TEST_F(MessagesFormattingTest, ConfigPerformanceCores) {
  std::string msg = Messages::config_performance_cores(8);
  EXPECT_NE(msg.find("8"), std::string::npos);
  EXPECT_NE(msg.find("Performance Cores"), std::string::npos);
}

TEST_F(MessagesFormattingTest, ConfigEfficiencyCores) {
  std::string msg = Messages::config_efficiency_cores(2);
  EXPECT_NE(msg.find("2"), std::string::npos);
  EXPECT_NE(msg.find("Efficiency Cores"), std::string::npos);
}

TEST_F(MessagesFormattingTest, ConfigTotalCores) {
  std::string msg = Messages::config_total_cores(10);
  EXPECT_NE(msg.find("10"), std::string::npos);
  EXPECT_NE(msg.find("Total CPU Cores"), std::string::npos);
}

// ============================================================================
// Cache Info Messages Tests (using formatting fixture)
// ============================================================================

TEST_F(MessagesFormattingTest, CacheInfoHeader) {
  std::string msg = Messages::cache_info_header();
  EXPECT_NE(msg.find("Detected Cache Sizes"), std::string::npos);
}

TEST_F(MessagesFormattingTest, CacheSizeCustom) {
  // Test bytes
  std::string msg = Messages::cache_size_custom(512);
  EXPECT_NE(msg.find("512"), std::string::npos);
  EXPECT_NE(msg.find("B"), std::string::npos);
  
  // Test KB
  msg = Messages::cache_size_custom(256 * 1024);
  EXPECT_NE(msg.find("256"), std::string::npos);
  EXPECT_NE(msg.find("KB"), std::string::npos);
  
  // Test MB
  msg = Messages::cache_size_custom(2 * 1024 * 1024);
  EXPECT_NE(msg.find("2"), std::string::npos);
  EXPECT_NE(msg.find("MB"), std::string::npos);
}

TEST_F(MessagesFormattingTest, CacheSizeL1) {
  // Test bytes
  std::string msg = Messages::cache_size_l1(128 * 1024);
  EXPECT_NE(msg.find("128"), std::string::npos);
  EXPECT_NE(msg.find("KB"), std::string::npos);
  EXPECT_NE(msg.find("per P-core"), std::string::npos);
  
  // Test MB
  msg = Messages::cache_size_l1(1 * 1024 * 1024);
  EXPECT_NE(msg.find("1"), std::string::npos);
  EXPECT_NE(msg.find("MB"), std::string::npos);
}

TEST_F(MessagesFormattingTest, CacheSizeL2) {
  // Test KB
  std::string msg = Messages::cache_size_l2(4 * 1024 * 1024);
  EXPECT_NE(msg.find("4"), std::string::npos);
  EXPECT_NE(msg.find("MB"), std::string::npos);
  EXPECT_NE(msg.find("per P-core cluster"), std::string::npos);
}

TEST_F(MessagesFormattingTest, CacheSizePerPcore) {
  std::string msg = Messages::cache_size_per_pcore();
  EXPECT_EQ(msg, " (per P-core)");
}

TEST_F(MessagesFormattingTest, CacheSizePerPcoreCluster) {
  std::string msg = Messages::cache_size_per_pcore_cluster();
  EXPECT_EQ(msg, " (per P-core cluster)");
}

// ============================================================================
// Results Output Messages Tests (using formatting fixture)
// ============================================================================

TEST_F(MessagesFormattingTest, ResultsLoopHeader) {
  std::string msg = Messages::results_loop_header(0);
  EXPECT_NE(msg.find("1"), std::string::npos);  // Loop 0 displays as "Loop 1"
  EXPECT_NE(msg.find("Loop"), std::string::npos);
  
  msg = Messages::results_loop_header(4);
  EXPECT_NE(msg.find("5"), std::string::npos);  // Loop 4 displays as "Loop 5"
}

TEST_F(MessagesFormattingTest, ResultsMainMemoryBandwidth) {
  std::string msg = Messages::results_main_memory_bandwidth(8);
  EXPECT_NE(msg.find("8"), std::string::npos);
  EXPECT_NE(msg.find("threads"), std::string::npos);
  EXPECT_NE(msg.find("Main Memory Bandwidth"), std::string::npos);
}

TEST_F(MessagesFormattingTest, ResultsReadBandwidth) {
  std::string msg = Messages::results_read_bandwidth(25.123, 1.456);
  EXPECT_NE(msg.find("25.123"), std::string::npos);
  EXPECT_NE(msg.find("GB/s"), std::string::npos);
  EXPECT_NE(msg.find("1.456"), std::string::npos);
  EXPECT_NE(msg.find("s)"), std::string::npos);
}

TEST_F(MessagesFormattingTest, ResultsWriteBandwidth) {
  std::string msg = Messages::results_write_bandwidth(30.789, 2.345);
  EXPECT_NE(msg.find("30.789"), std::string::npos);
  EXPECT_NE(msg.find("GB/s"), std::string::npos);
  EXPECT_NE(msg.find("2.345"), std::string::npos);
}

TEST_F(MessagesFormattingTest, ResultsCopyBandwidth) {
  std::string msg = Messages::results_copy_bandwidth(20.456, 3.789);
  EXPECT_NE(msg.find("20.456"), std::string::npos);
  EXPECT_NE(msg.find("GB/s"), std::string::npos);
  EXPECT_NE(msg.find("3.789"), std::string::npos);
}

TEST_F(MessagesFormattingTest, ResultsMainMemoryLatency) {
  std::string msg = Messages::results_main_memory_latency();
  EXPECT_NE(msg.find("Main Memory Latency"), std::string::npos);
  EXPECT_NE(msg.find("single-threaded"), std::string::npos);
}

TEST_F(MessagesFormattingTest, ResultsLatencyTotalTime) {
  std::string msg = Messages::results_latency_total_time(5.678);
  EXPECT_NE(msg.find("5.678"), std::string::npos);
  EXPECT_NE(msg.find("s"), std::string::npos);
}

TEST_F(MessagesFormattingTest, ResultsLatencyAverage) {
  std::string msg = Messages::results_latency_average(123.45);
  EXPECT_NE(msg.find("123.45"), std::string::npos);
  EXPECT_NE(msg.find("ns"), std::string::npos);
}

TEST_F(MessagesFormattingTest, ResultsCacheBandwidth) {
  std::string msg = Messages::results_cache_bandwidth(1);
  EXPECT_NE(msg.find("Cache Bandwidth"), std::string::npos);
  EXPECT_NE(msg.find("single-threaded"), std::string::npos);
}

TEST_F(MessagesFormattingTest, ResultsCacheLatency) {
  std::string msg = Messages::results_cache_latency();
  EXPECT_NE(msg.find("Cache Latency"), std::string::npos);
  EXPECT_NE(msg.find("pointer chase"), std::string::npos);
}

TEST_F(MessagesFormattingTest, ResultsCustomCache) {
  std::string msg = Messages::results_custom_cache();
  EXPECT_EQ(msg, "  Custom Cache:");
}

TEST_F(MessagesFormattingTest, ResultsL1Cache) {
  std::string msg = Messages::results_l1_cache();
  EXPECT_EQ(msg, "  L1 Cache:");
}

TEST_F(MessagesFormattingTest, ResultsL2Cache) {
  std::string msg = Messages::results_l2_cache();
  EXPECT_EQ(msg, "  L2 Cache:");
}

TEST_F(MessagesFormattingTest, ResultsCacheReadBandwidth) {
  std::string msg = Messages::results_cache_read_bandwidth(150.789);
  EXPECT_NE(msg.find("150.789"), std::string::npos);
  EXPECT_NE(msg.find("GB/s"), std::string::npos);
  EXPECT_NE(msg.find("Read"), std::string::npos);
}

TEST_F(MessagesFormattingTest, ResultsCacheWriteBandwidth) {
  std::string msg = Messages::results_cache_write_bandwidth(200.123);
  EXPECT_NE(msg.find("200.123"), std::string::npos);
  EXPECT_NE(msg.find("GB/s"), std::string::npos);
  EXPECT_NE(msg.find("Write"), std::string::npos);
}

TEST_F(MessagesFormattingTest, ResultsCacheCopyBandwidth) {
  std::string msg = Messages::results_cache_copy_bandwidth(175.456);
  EXPECT_NE(msg.find("175.456"), std::string::npos);
  EXPECT_NE(msg.find("GB/s"), std::string::npos);
  EXPECT_NE(msg.find("Copy"), std::string::npos);
}

TEST_F(MessagesFormattingTest, ResultsBufferSizeBytes) {
  std::string msg = Messages::results_buffer_size_bytes(1024);
  EXPECT_NE(msg.find("1024"), std::string::npos);
  EXPECT_NE(msg.find("B)"), std::string::npos);
}

TEST_F(MessagesFormattingTest, ResultsBufferSizeKb) {
  std::string msg = Messages::results_buffer_size_kb(256.5);
  EXPECT_NE(msg.find("256.50"), std::string::npos);
  EXPECT_NE(msg.find("KB)"), std::string::npos);
}

TEST_F(MessagesFormattingTest, ResultsBufferSizeMb) {
  std::string msg = Messages::results_buffer_size_mb(1.25);
  EXPECT_NE(msg.find("1.25"), std::string::npos);
  EXPECT_NE(msg.find("MB)"), std::string::npos);
}

TEST_F(MessagesFormattingTest, ResultsCacheLatencyNs) {
  std::string msg = Messages::results_cache_latency_ns(1.23);
  EXPECT_NE(msg.find("1.23"), std::string::npos);
  EXPECT_NE(msg.find("ns"), std::string::npos);
}

TEST_F(MessagesFormattingTest, ResultsSeparator) {
  std::string msg = Messages::results_separator();
  EXPECT_EQ(msg, "--------------");
}

TEST_F(MessagesFormattingTest, ResultsCacheLatencyCustomNs) {
  std::string msg = Messages::results_cache_latency_custom_ns(2.5, 256 * 1024);
  EXPECT_NE(msg.find("2.50"), std::string::npos);
  EXPECT_NE(msg.find("ns"), std::string::npos);
  EXPECT_NE(msg.find("262144"), std::string::npos);
  EXPECT_NE(msg.find("B)"), std::string::npos);
}

TEST_F(MessagesFormattingTest, ResultsCacheLatencyCustomNsKb) {
  std::string msg = Messages::results_cache_latency_custom_ns_kb(3.75, 128.5);
  EXPECT_NE(msg.find("3.75"), std::string::npos);
  EXPECT_NE(msg.find("128.50"), std::string::npos);
  EXPECT_NE(msg.find("KB)"), std::string::npos);
}

TEST_F(MessagesFormattingTest, ResultsCacheLatencyCustomNsMb) {
  std::string msg = Messages::results_cache_latency_custom_ns_mb(4.25, 0.5);
  EXPECT_NE(msg.find("4.25"), std::string::npos);
  EXPECT_NE(msg.find("0.50"), std::string::npos);
  EXPECT_NE(msg.find("MB)"), std::string::npos);
}

TEST_F(MessagesFormattingTest, ResultsCacheLatencyL1Ns) {
  std::string msg = Messages::results_cache_latency_l1_ns(0.5, 64 * 1024);
  EXPECT_NE(msg.find("0.50"), std::string::npos);
  EXPECT_NE(msg.find("65536"), std::string::npos);
  EXPECT_NE(msg.find("L1 Cache"), std::string::npos);
}

TEST_F(MessagesFormattingTest, ResultsCacheLatencyL1NsKb) {
  std::string msg = Messages::results_cache_latency_l1_ns_kb(0.75, 32.25);
  EXPECT_NE(msg.find("0.75"), std::string::npos);
  EXPECT_NE(msg.find("32.25"), std::string::npos);
  EXPECT_NE(msg.find("L1 Cache"), std::string::npos);
}

TEST_F(MessagesFormattingTest, ResultsCacheLatencyL1NsMb) {
  std::string msg = Messages::results_cache_latency_l1_ns_mb(1.0, 0.064);
  EXPECT_NE(msg.find("1.00"), std::string::npos);
  EXPECT_NE(msg.find("0.06"), std::string::npos);
  EXPECT_NE(msg.find("L1 Cache"), std::string::npos);
}

TEST_F(MessagesFormattingTest, ResultsCacheLatencyL2Ns) {
  std::string msg = Messages::results_cache_latency_l2_ns(2.5, 4 * 1024 * 1024);
  EXPECT_NE(msg.find("2.50"), std::string::npos);
  EXPECT_NE(msg.find("4194304"), std::string::npos);
  EXPECT_NE(msg.find("L2 Cache"), std::string::npos);
}

TEST_F(MessagesFormattingTest, ResultsCacheLatencyL2NsKb) {
  std::string msg = Messages::results_cache_latency_l2_ns_kb(3.0, 4096.5);
  EXPECT_NE(msg.find("3.00"), std::string::npos);
  EXPECT_NE(msg.find("4096.50"), std::string::npos);
  EXPECT_NE(msg.find("L2 Cache"), std::string::npos);
}

TEST_F(MessagesFormattingTest, ResultsCacheLatencyL2NsMb) {
  std::string msg = Messages::results_cache_latency_l2_ns_mb(4.5, 4.0);
  EXPECT_NE(msg.find("4.50"), std::string::npos);
  EXPECT_NE(msg.find("4.00"), std::string::npos);
  EXPECT_NE(msg.find("L2 Cache"), std::string::npos);
}

// ============================================================================
// Statistics Messages Tests (using formatting fixture)
// ============================================================================

TEST_F(MessagesFormattingTest, StatisticsHeader) {
  std::string msg = Messages::statistics_header(5);
  EXPECT_NE(msg.find("5"), std::string::npos);
  EXPECT_NE(msg.find("Loops"), std::string::npos);
  EXPECT_NE(msg.find("Statistics"), std::string::npos);
}

TEST_F(MessagesFormattingTest, StatisticsAverage) {
  std::string msg = Messages::statistics_average(25.123, 3);
  EXPECT_NE(msg.find("25.123"), std::string::npos);
  EXPECT_NE(msg.find("Average"), std::string::npos);
  
  msg = Messages::statistics_average(100.5, 1);
  EXPECT_NE(msg.find("100.5"), std::string::npos);
}

TEST_F(MessagesFormattingTest, StatisticsMedianP50) {
  std::string msg = Messages::statistics_median_p50(24.567, 3);
  EXPECT_NE(msg.find("24.567"), std::string::npos);
  EXPECT_NE(msg.find("Median"), std::string::npos);
  EXPECT_NE(msg.find("P50"), std::string::npos);
}

TEST_F(MessagesFormattingTest, StatisticsP90) {
  std::string msg = Messages::statistics_p90(30.123, 3);
  EXPECT_NE(msg.find("30.123"), std::string::npos);
  EXPECT_NE(msg.find("P90"), std::string::npos);
}

TEST_F(MessagesFormattingTest, StatisticsP95) {
  std::string msg = Messages::statistics_p95(32.456, 3);
  EXPECT_NE(msg.find("32.456"), std::string::npos);
  EXPECT_NE(msg.find("P95"), std::string::npos);
}

TEST_F(MessagesFormattingTest, StatisticsP99) {
  std::string msg = Messages::statistics_p99(35.789, 3);
  EXPECT_NE(msg.find("35.789"), std::string::npos);
  EXPECT_NE(msg.find("P99"), std::string::npos);
}

TEST_F(MessagesFormattingTest, StatisticsStddev) {
  std::string msg = Messages::statistics_stddev(2.345, 3);
  EXPECT_NE(msg.find("2.345"), std::string::npos);
  EXPECT_NE(msg.find("Stddev"), std::string::npos);
}

TEST_F(MessagesFormattingTest, StatisticsMin) {
  std::string msg = Messages::statistics_min(20.0, 1);
  EXPECT_NE(msg.find("20.0"), std::string::npos);
  EXPECT_NE(msg.find("Min"), std::string::npos);
}

TEST_F(MessagesFormattingTest, StatisticsMax) {
  std::string msg = Messages::statistics_max(40.0, 1);
  EXPECT_NE(msg.find("40.0"), std::string::npos);
  EXPECT_NE(msg.find("Max"), std::string::npos);
}

TEST_F(MessagesFormattingTest, StatisticsCacheBandwidthHeader) {
  std::string msg = Messages::statistics_cache_bandwidth_header("L1");
  EXPECT_NE(msg.find("L1"), std::string::npos);
  EXPECT_NE(msg.find("Cache Bandwidth"), std::string::npos);
  EXPECT_NE(msg.find("GB/s"), std::string::npos);
}

TEST_F(MessagesFormattingTest, StatisticsCacheRead) {
  std::string msg = Messages::statistics_cache_read();
  EXPECT_EQ(msg, "  Read:");
}

TEST_F(MessagesFormattingTest, StatisticsCacheWrite) {
  std::string msg = Messages::statistics_cache_write();
  EXPECT_EQ(msg, "  Write:");
}

TEST_F(MessagesFormattingTest, StatisticsCacheCopy) {
  std::string msg = Messages::statistics_cache_copy();
  EXPECT_EQ(msg, "  Copy:");
}

TEST_F(MessagesFormattingTest, StatisticsCacheLatencyHeader) {
  std::string msg = Messages::statistics_cache_latency_header();
  EXPECT_NE(msg.find("Cache Latency"), std::string::npos);
  EXPECT_NE(msg.find("ns"), std::string::npos);
}

TEST_F(MessagesFormattingTest, StatisticsCacheLatencyName) {
  std::string msg = Messages::statistics_cache_latency_name("L1");
  EXPECT_EQ(msg, "  L1 Cache:");
  
  msg = Messages::statistics_cache_latency_name("Custom");
  EXPECT_EQ(msg, "  Custom Cache:");
}

TEST_F(MessagesFormattingTest, StatisticsMedianP50FromSamples) {
  std::string msg = Messages::statistics_median_p50_from_samples(1.5, 1000, 2);
  EXPECT_NE(msg.find("1.50"), std::string::npos);
  EXPECT_NE(msg.find("1000"), std::string::npos);
  EXPECT_NE(msg.find("samples"), std::string::npos);
  EXPECT_NE(msg.find("Median"), std::string::npos);
  EXPECT_NE(msg.find("P50"), std::string::npos);
}

TEST_F(MessagesFormattingTest, StatisticsMainMemoryLatencyHeader) {
  std::string msg = Messages::statistics_main_memory_latency_header();
  EXPECT_NE(msg.find("Main Memory Latency"), std::string::npos);
  EXPECT_NE(msg.find("ns"), std::string::npos);
}

TEST_F(MessagesFormattingTest, StatisticsFooter) {
  std::string msg = Messages::statistics_footer();
  EXPECT_EQ(msg, "----------------------------------");
}


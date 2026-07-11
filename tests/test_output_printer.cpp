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

#include <gtest/gtest.h>

#include <string>

#include "benchmark/benchmark_runner.h"
#include "core/config/config.h"
#include "core/config/constants.h"
#include "core/config/version.h"
#include "output/console/messages/messages_api.h"
#include "output/console/output_printer.h"

namespace {

std::string capture_results(const BenchmarkConfig& config,
                            const BenchmarkResults& results) {
  testing::internal::CaptureStdout();
  print_results(0, config, results);
  return testing::internal::GetCapturedStdout();
}

}  // namespace

TEST(OutputPrinterTest, RuntimeBannerUsesSharedContract) {
  testing::internal::CaptureStdout();
  print_runtime_banner();
  const std::string output = testing::internal::GetCapturedStdout();

  const std::string expected =
      Messages::config_header(SOFTVERSION) + "\n" +
      Messages::config_copyright() + "\n" + Messages::config_license() + "\n";
  EXPECT_EQ(output, expected);
}

TEST(OutputPrinterTest, PartialBandwidthPrintsEachMetricWithStatusInsteadOfZero) {
  BenchmarkConfig config;
  config.only_bandwidth = true;
  config.num_threads = 2;
  BenchmarkResults results;
  set_measurement_value(results.main_read_bandwidth, 12.5, 0.25);
  set_measurement_unavailable(results.main_write_bandwidth,
                              BenchmarkMeasurementStatus::Interrupted,
                              "interrupted during write");
  set_measurement_unavailable(results.main_copy_bandwidth,
                              BenchmarkMeasurementStatus::Failed,
                              "copy worker failed");

  const std::string output = capture_results(config, results);

  EXPECT_NE(output.find(Messages::results_read_bandwidth(12.5, 0.25)),
            std::string::npos);
  EXPECT_NE(output.find(Messages::results_measurement_unavailable(
                "  Write", "interrupted", "interrupted during write")),
            std::string::npos);
  EXPECT_NE(output.find(Messages::results_measurement_unavailable(
                "  Copy ", "failed", "copy worker failed")),
            std::string::npos);
  EXPECT_EQ(output.find("Write Bandwidth: 0"), std::string::npos);
  EXPECT_EQ(output.find("Copy Bandwidth: 0"), std::string::npos);
}

TEST(OutputPrinterTest, PartialLocalityComparisonRetainsMeasuredAndUnavailableEvidence) {
  BenchmarkConfig config;
  config.only_latency = true;
  config.buffer_size = 4096;
  BenchmarkResults results;
  set_measurement_value(results.main_latency, 80.0, 0.2);
  set_measurement_value(results.locality_16k_latency, 20.0, 0.01);
  set_measurement_unavailable(results.global_random_latency,
                              BenchmarkMeasurementStatus::Interrupted,
                              "stop requested");
  set_measurement_unavailable(results.locality_latency_delta,
                              BenchmarkMeasurementStatus::Invalid,
                              "paired measurement incomplete");

  const std::string output = capture_results(config, results);

  EXPECT_NE(output.find(Messages::results_latency_tlb_hit(20.0)),
            std::string::npos);
  EXPECT_NE(output.find(Messages::results_measurement_unavailable(
                "  Global-random latency", "interrupted", "stop requested")),
            std::string::npos);
  EXPECT_NE(output.find(Messages::results_measurement_unavailable(
                "  Locality latency delta (global - 16 KiB)", "invalid",
                "paired measurement incomplete")),
            std::string::npos);
}

class OutputPrinterCustomCacheUnitsTest
    : public testing::TestWithParam<size_t> {};

TEST_P(OutputPrinterCustomCacheUnitsTest, UsesMatchingByteKiBOrMiBFormatter) {
  const size_t buffer_size = GetParam();
  BenchmarkConfig config;
  config.only_latency = true;
  config.use_custom_cache_size = true;
  config.custom_buffer_size = buffer_size;
  BenchmarkResults results;
  set_measurement_value(results.custom_latency, 3.25, 0.01);

  const std::string output = capture_results(config, results);
  std::string expected;
  if (buffer_size < 1024) {
    expected = Messages::results_cache_latency_custom_ns(3.25, buffer_size);
  } else if (buffer_size < 1024 * 1024) {
    expected = Messages::results_cache_latency_custom_ns_kb(
        3.25, buffer_size / 1024.0);
  } else {
    expected = Messages::results_cache_latency_custom_ns_mb(
        3.25, buffer_size / (1024.0 * 1024.0));
  }
  EXPECT_NE(output.find(expected), std::string::npos);
  EXPECT_EQ(output.find(Messages::results_l1_cache()), std::string::npos);
  EXPECT_EQ(output.find(Messages::results_l2_cache()), std::string::npos);
}

INSTANTIATE_TEST_SUITE_P(ByteKiBMiBBoundaries,
                         OutputPrinterCustomCacheUnitsTest,
                         testing::Values(1023u, 1024u,
                                         1024u * 1024u - 1u,
                                         1024u * 1024u));

TEST(OutputPrinterTest, DetectedCachesPrintIndependentlyFromCustomCache) {
  BenchmarkConfig config;
  config.only_latency = true;
  config.l1_buffer_size = 64 * 1024;
  config.l2_buffer_size = 4 * 1024 * 1024;
  BenchmarkResults results;
  set_measurement_value(results.l1_latency, 1.5, 0.01);
  set_measurement_unavailable(results.l2_latency,
                              BenchmarkMeasurementStatus::Skipped,
                              "cache target disabled by planner");

  const std::string output = capture_results(config, results);

  EXPECT_NE(output.find(Messages::results_cache_latency_l1_ns_kb(1.5, 64.0)),
            std::string::npos);
  EXPECT_NE(output.find(Messages::results_measurement_unavailable(
                "  L2 Cache", "skipped", "cache target disabled by planner")),
            std::string::npos);
  EXPECT_EQ(output.find(Messages::results_custom_cache()), std::string::npos);
}

TEST(OutputPrinterTest, ConfigurationSelectsPatternAutomaticWorkWithoutLatencySection) {
  testing::internal::CaptureStdout();
  print_configuration(1024 * 1024, 1, 2 * 1024 * 1024, 7, 3, false,
                      256, "auto", 0, "Apple Test", 4, 2, 6, false,
                      false, true, false);
  const std::string output = testing::internal::GetCapturedStdout();

  EXPECT_NE(output.find(Messages::config_pattern_iterations_auto(
                Constants::PATTERN_CALIBRATION_TARGET_SECONDS,
                Constants::PATTERN_CALIBRATION_MIN_SECONDS,
                Constants::PATTERN_CALIBRATION_MAX_SECONDS)),
            std::string::npos);
  EXPECT_EQ(output.find(Messages::config_benchmark_iterations_auto(
                Constants::BENCHMARK_CALIBRATION_TARGET_SECONDS,
                Constants::BENCHMARK_CALIBRATION_MIN_SECONDS,
                Constants::BENCHMARK_CALIBRATION_MAX_SECONDS)),
            std::string::npos);
  EXPECT_EQ(output.find("Continuous latency calibration"), std::string::npos);
}

TEST(OutputPrinterTest, CacheInfoChoosesCustomOrDetectedComposition) {
  testing::internal::CaptureStdout();
  print_cache_info(128 * 1024, 16 * 1024 * 1024, true, 512 * 1024);
  const std::string custom = testing::internal::GetCapturedStdout();
  EXPECT_NE(custom.find(Messages::cache_size_custom(512 * 1024)),
            std::string::npos);
  EXPECT_EQ(custom.find(Messages::cache_size_l1(128 * 1024)),
            std::string::npos);

  testing::internal::CaptureStdout();
  print_cache_info(128 * 1024, 16 * 1024 * 1024, false, 0);
  const std::string detected = testing::internal::GetCapturedStdout();
  EXPECT_NE(detected.find(Messages::cache_size_l1(128 * 1024)),
            std::string::npos);
  EXPECT_NE(detected.find(Messages::cache_size_l2(16 * 1024 * 1024)),
            std::string::npos);
  EXPECT_EQ(detected.find(Messages::cache_size_custom(512 * 1024)),
            std::string::npos);
}

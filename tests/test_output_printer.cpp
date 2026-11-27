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

#include "core/config/constants.h"
#include "output/console/messages/messages_api.h"
#include "output/console/output_printer.h"

TEST(OutputPrinterTest, ConfigurationSelectsPatternAutomaticWorkWithoutLatencySection) {
  testing::internal::CaptureStdout();
  print_configuration(1024 * 1024, 1, 2 * 1024 * 1024, 7, 3, false, 256, "auto", 0, "Apple Test", 4, 2, 6,
                      true, false);
  const std::string output = testing::internal::GetCapturedStdout();

  const std::string expected =
      Messages::config_buffer_size(1.0, 1) + "\n" + Messages::config_total_allocation(2.0) + "\n" +
      Messages::config_pattern_iterations_auto(Constants::PATTERN_CALIBRATION_TARGET_SECONDS,
                                               Constants::PATTERN_CALIBRATION_MIN_SECONDS,
                                               Constants::PATTERN_CALIBRATION_MAX_SECONDS) +
      "\n" + Messages::config_loop_count(3) + "\n" + Messages::config_non_cacheable(false) + "\n" +
      Messages::config_latency_stride(256) + "\n" + Messages::config_latency_chain_mode("auto") + "\n" +
      Messages::config_latency_tlb_locality(0) + "\n" + Messages::config_processor_name("Apple Test") + "\n" +
      Messages::config_performance_cores(4) + "\n" + Messages::config_efficiency_cores(2) + "\n" +
      Messages::config_total_cores(6) + "\n" + Messages::config_benchmark_threads(6) + "\n";
  EXPECT_EQ(output, expected);
}

TEST(OutputPrinterTest, CacheInfoChoosesCustomOrDetectedComposition) {
  testing::internal::CaptureStdout();
  print_cache_info(128 * 1024, 16 * 1024 * 1024, true, 512 * 1024);
  const std::string custom = testing::internal::GetCapturedStdout();
  EXPECT_EQ(custom, Messages::cache_info_header() + "\n" + Messages::cache_size_custom(512 * 1024) + "\n");

  testing::internal::CaptureStdout();
  print_cache_info(128 * 1024, 16 * 1024 * 1024, false, 0);
  const std::string detected = testing::internal::GetCapturedStdout();
  EXPECT_EQ(detected, Messages::cache_info_header() + "\n" + Messages::cache_size_l1(128 * 1024) + "\n" +
                          Messages::cache_size_l2(16 * 1024 * 1024) + "\n");
}

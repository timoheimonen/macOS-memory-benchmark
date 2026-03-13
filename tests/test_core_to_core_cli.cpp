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

/**
 * @file test_core_to_core_cli.cpp
 * @brief Unit tests for standalone core-to-core CLI parsing
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 */

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "benchmark/core_to_core_latency.h"
#include "core/config/constants.h"

namespace {

// Builds argc/argv-style input and forwards to the standalone mode parser.
int parse_with_args(const std::vector<std::string>& args, CoreToCoreLatencyConfig& config) {
  std::vector<std::string> mutable_args = args;
  std::vector<char*> argv;
  argv.reserve(mutable_args.size());
  for (std::string& arg : mutable_args) {
    argv.push_back(const_cast<char*>(arg.c_str()));
  }
  return parse_core_to_core_mode_arguments(static_cast<int>(argv.size()), argv.data(), config);
}

}  // namespace

TEST(CoreToCoreCliTest, ParsesDefaultStandaloneModeValues) {
  // Baseline parse should inherit defaults directly from centralized constants.
  CoreToCoreLatencyConfig config;
  const int parse_result = parse_with_args({"memory_benchmark", "-analyze-core2core"}, config);

  EXPECT_EQ(parse_result, EXIT_SUCCESS);
  EXPECT_FALSE(config.help_requested);
  EXPECT_EQ(config.loop_count, Constants::CORE_TO_CORE_DEFAULT_LOOP_COUNT);
  EXPECT_EQ(config.latency_sample_count, Constants::CORE_TO_CORE_DEFAULT_LATENCY_SAMPLE_COUNT);
  EXPECT_TRUE(config.output_file.empty());
}

TEST(CoreToCoreCliTest, ParsesOptionalModeArguments) {
  // Optional standalone flags should override defaults without enabling help mode.
  CoreToCoreLatencyConfig config;
  const int parse_result =
      parse_with_args({"memory_benchmark",
                       "-analyze-core2core",
                       "-count",
                       "5",
                       "-latency-samples",
                       "128",
                       "-output",
                       "core2core.json"},
                      config);

  EXPECT_EQ(parse_result, EXIT_SUCCESS);
  EXPECT_FALSE(config.help_requested);
  EXPECT_EQ(config.loop_count, 5);
  EXPECT_EQ(config.latency_sample_count, 128);
  EXPECT_EQ(config.output_file, "core2core.json");
}

TEST(CoreToCoreCliTest, RejectsUnknownOptionsInStandaloneMode) {
  // Standalone mode must reject standard benchmark options.
  CoreToCoreLatencyConfig config;
  const int parse_result = parse_with_args(
      {"memory_benchmark", "-analyze-core2core", "-buffersize", "256"},
      config);

  EXPECT_EQ(parse_result, EXIT_FAILURE);
}

TEST(CoreToCoreCliTest, RejectsInvalidCountValues) {
  // Count must be a positive integer.
  CoreToCoreLatencyConfig config;
  const int parse_result =
      parse_with_args({"memory_benchmark", "-analyze-core2core", "-count", "0"}, config);

  EXPECT_EQ(parse_result, EXIT_FAILURE);
}

TEST(CoreToCoreCliTest, HelpFlagReturnsSuccessAndSetsHelpRequested) {
  // Help should short-circuit with success and mark help flag in config.
  CoreToCoreLatencyConfig config;
  const int parse_result = parse_with_args(
      {"memory_benchmark", "-analyze-core2core", "-h"},
      config);

  EXPECT_EQ(parse_result, EXIT_SUCCESS);
  EXPECT_TRUE(config.help_requested);
}

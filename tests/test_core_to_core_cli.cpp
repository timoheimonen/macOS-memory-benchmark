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

#include <limits>
#include <string>
#include <vector>

#include "benchmark/core_to_core_latency.h"
#include "core/config/constants.h"
#include "output/console/messages/messages_api.h"

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

struct CapturedCoreCliParse {
  int result = EXIT_FAILURE;
  std::string stderr_output;
};

CapturedCoreCliParse parse_capturing_stderr(
    const std::vector<std::string>& args,
    CoreToCoreLatencyConfig& config) {
  testing::internal::CaptureStderr();
  const int result = parse_with_args(args, config);
  return {result, testing::internal::GetCapturedStderr()};
}

std::string expected_invalid_value(const std::string& option,
                                   const std::string& value,
                                   const std::string& reason) {
  return Messages::error_prefix() +
         Messages::error_invalid_value(option, value, reason);
}

std::string first_output_line(const std::string& output) {
  return output.substr(0, output.find('\n'));
}

}  // namespace

TEST(CoreToCoreCliTest, ParsesDefaultsOptionalAliasesBoundariesAndHelp) {
  {
    SCOPED_TRACE("defaults");
    CoreToCoreLatencyConfig config;
    EXPECT_EQ(parse_with_args({"memory_benchmark", "--analyze-core2core"}, config), EXIT_SUCCESS);
    EXPECT_FALSE(config.help_requested);
    EXPECT_EQ(config.loop_count, Constants::CORE_TO_CORE_DEFAULT_LOOP_COUNT);
    EXPECT_EQ(config.loop_count, 3);
    EXPECT_EQ(config.latency_sample_count, Constants::CORE_TO_CORE_DEFAULT_LATENCY_SAMPLE_COUNT);
    EXPECT_TRUE(config.output_file.empty());
  }

  for (bool short_options : {false, true}) {
    for (const char* output : {"core2core.json", "-", "./-"}) {
      SCOPED_TRACE(short_options);
      SCOPED_TRACE(output);
      CoreToCoreLatencyConfig config;
      const CapturedCoreCliParse parsed = parse_capturing_stderr(
          {"memory_benchmark", short_options ? "-C" : "--analyze-core2core", short_options ? "-r" : "--count", "5",
           short_options ? "-n" : "--latency-samples", "128", short_options ? "-o" : "--output", output},
          config);
      ASSERT_EQ(parsed.result, EXIT_SUCCESS) << parsed.stderr_output;
      EXPECT_TRUE(parsed.stderr_output.empty());
      EXPECT_FALSE(config.help_requested);
      EXPECT_EQ(config.loop_count, 5);
      EXPECT_EQ(config.latency_sample_count, 128);
      EXPECT_EQ(config.output_file, output);
    }
  }

  {
    SCOPED_TRACE("int-max-boundaries");
    CoreToCoreLatencyConfig config;
    const std::string int_max = std::to_string(std::numeric_limits<int>::max());
    EXPECT_EQ(parse_with_args({"memory_benchmark", "--analyze-core2core", "--count", int_max, "--latency-samples",
                               int_max, "--sweep-max-runs", int_max},
                              config),
              EXIT_SUCCESS);
    EXPECT_EQ(config.loop_count, std::numeric_limits<int>::max());
    EXPECT_EQ(config.latency_sample_count, std::numeric_limits<int>::max());
    EXPECT_EQ(config.sweep_max_runs, static_cast<size_t>(std::numeric_limits<int>::max()));
  }

  {
    SCOPED_TRACE("help");
    CoreToCoreLatencyConfig config;
    EXPECT_EQ(parse_with_args({"memory_benchmark", "--analyze-core2core", "-h"}, config), EXIT_SUCCESS);
    EXPECT_TRUE(config.help_requested);
  }
}

TEST(CoreToCoreCliTest, ParsesSweepArguments) {
  CoreToCoreLatencyConfig config;
  const int parse_result =
      parse_with_args({"memory_benchmark", "--analyze-core2core", "--output", "core2core_sweep.json", "--sweep",
                       "count=1,2", "--sweep", "latency-samples=4,8", "--sweep-max-runs", "4"},
                      config);

  EXPECT_EQ(parse_result, EXIT_SUCCESS);
  EXPECT_TRUE(config.run_sweep);
  EXPECT_EQ(config.output_file, "core2core_sweep.json");
  EXPECT_EQ(config.sweep_max_runs, 4u);
  ASSERT_EQ(config.sweep_specs.size(), 2u);
  EXPECT_EQ(config.sweep_specs[0].parameter, CoreToCoreSweepParameter::Count);
  EXPECT_EQ(config.sweep_specs[0].values[0].integer_value, 1);
  EXPECT_EQ(config.sweep_specs[0].values[1].integer_value, 2);
  EXPECT_EQ(config.sweep_specs[1].parameter, CoreToCoreSweepParameter::LatencySamples);
  EXPECT_EQ(config.sweep_specs[1].values[0].integer_value, 4);
  EXPECT_EQ(config.sweep_specs[1].values[1].integer_value, 8);
}

TEST(CoreToCoreCliTest, RejectsMissingDuplicateAndUnknownOptionsWithCentralizedDiagnostics) {
  struct OptionErrorCase {
    std::vector<std::string> arguments;
    std::string diagnostic;
  };

  std::vector<OptionErrorCase> cases = {
      {{"memory_benchmark", "--analyze-core2core", "--output"}, Messages::error_missing_value("--output")},
      {{"memory_benchmark", "--analyze-core2core", "--output", "first", "--output", "second"},
       Messages::error_duplicate_option("--output")},
      {{"memory_benchmark", "--analyze-core2core", "-o", "first", "--output", "second"},
       Messages::error_duplicate_option("--output")},
      {{"memory_benchmark", "--analyze-core2core", "--output", "first", "-o", "second"},
       Messages::error_duplicate_option("--output")},
      {{"memory_benchmark", "--analyze-core2core", "--buffer-size", "256"}, ""},
  };
  for (const char* option : {"--count", "--latency-samples", "--sweep-max-runs"}) {
    cases.push_back({{"memory_benchmark", "--analyze-core2core", option}, Messages::error_missing_value(option)});
    cases.push_back({{"memory_benchmark", "--analyze-core2core", option, "1", option, "2"},
                     Messages::error_duplicate_option(option)});
  }

  for (const OptionErrorCase& test_case : cases) {
    SCOPED_TRACE(::testing::PrintToString(test_case.arguments));
    CoreToCoreLatencyConfig config;
    const CapturedCoreCliParse parsed = parse_capturing_stderr(test_case.arguments, config);
    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    if (!test_case.diagnostic.empty()) {
      EXPECT_EQ(first_output_line(parsed.stderr_output), Messages::error_prefix() + test_case.diagnostic);
    }
  }
}

TEST(CoreToCoreCliTest, RejectsMalformedNumericTokensWithCentralizedErrors) {
  struct InvalidNumericCase {
    const char* option;
    const char* value;
    const char* reason;
  };

  constexpr const char* kInvalidReason =
      "must be an integer without whitespace, a plus sign, or trailing characters";
  const std::string positive_range_reason =
      "must be between 1 and " + std::to_string(std::numeric_limits<int>::max());
  const InvalidNumericCase cases[] = {
      {"--count", "5junk", kInvalidReason},
      {"--latency-samples", "8junk", kInvalidReason},
      {"--sweep-max-runs", "4junk", kInvalidReason},
      {"--count", "0", positive_range_reason.c_str()},
      {"--count", "-1", positive_range_reason.c_str()},
  };

  for (const InvalidNumericCase& test_case : cases) {
    SCOPED_TRACE(test_case.option);
    SCOPED_TRACE(test_case.value);
    CoreToCoreLatencyConfig config;
    const CapturedCoreCliParse parsed =
        parse_capturing_stderr({"memory_benchmark", "--analyze-core2core", test_case.option, test_case.value}, config);
    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    EXPECT_NE(parsed.stderr_output.find(expected_invalid_value(test_case.option, test_case.value, test_case.reason)),
              std::string::npos);
  }
}

TEST(CoreToCoreCliTest, RejectsInvalidAndMalformedSweepConfigurations) {
  struct SweepErrorCase {
    std::vector<std::string> options;
    std::string expected_first_line;
  };

  constexpr const char* kInvalidReason =
      "must be an integer without whitespace, a plus sign, or trailing characters";
  const SweepErrorCase cases[] = {
      {{"--sweep", "count=1,2"}, Messages::error_prefix() + Messages::error_sweep_requires_output()},
      {{"--output", "sweep.json", "--sweep", "count=1,2", "--sweep", "latency-samples=4,8", "--sweep-max-runs", "3"},
       Messages::error_prefix() + Messages::error_sweep_too_many_runs(4, 3)},
      {{"--output", "sweep.json", "--sweep", "count=1,2", "--sweep", "count=3,4"},
       Messages::error_prefix() + Messages::error_duplicate_option("--sweep count")},
      {{"--output", "core2core_sweep.json", "--sweep", "threads=1,2"},
       expected_invalid_value("--sweep", "threads=1,2", "unsupported core-to-core sweep parameter: threads")},
      {{"--output", "core2core_sweep.json", "--sweep", "count"},
       expected_invalid_value("--sweep", "count", "sweep must use key=value1,value2 syntax")},
      {{"--output", "core2core_sweep.json", "--sweep", "count=,1"},
       expected_invalid_value("--sweep", "count=,1", "sweep value list cannot contain empty values")},
      {{"--output", "core2core_sweep.json", "--sweep", "count=1,"},
       expected_invalid_value("--sweep", "count=1,", "sweep value list cannot contain empty values")},
      {{"--output", "core2core_sweep.json", "--sweep", "count=1,,2"},
       expected_invalid_value("--sweep", "count=1,,2", "sweep value list cannot contain empty values")},
      {{"--output", "core2core_sweep.json", "--sweep", "count=1x"},
       expected_invalid_value("--count", "1x", kInvalidReason)},
      {{"--output", "core2core_sweep.json", "--sweep", "latency-samples=4x"},
       expected_invalid_value("--latency-samples", "4x", kInvalidReason)},
  };

  for (const SweepErrorCase& test_case : cases) {
    SCOPED_TRACE(::testing::PrintToString(test_case.options));
    std::vector<std::string> arguments = {"memory_benchmark", "--analyze-core2core"};
    arguments.insert(arguments.end(), test_case.options.begin(), test_case.options.end());
    CoreToCoreLatencyConfig config;
    const CapturedCoreCliParse parsed = parse_capturing_stderr(arguments, config);
    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    EXPECT_EQ(first_output_line(parsed.stderr_output), test_case.expected_first_line);
  }
}

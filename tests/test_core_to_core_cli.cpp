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
  std::string stdout_output;
  std::string stderr_output;
};

CapturedCoreCliParse parse_capturing_stderr(
    const std::vector<std::string>& args,
    CoreToCoreLatencyConfig& config) {
  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  const int result = parse_with_args(args, config);
  const std::string stderr_output = testing::internal::GetCapturedStderr();
  const std::string stdout_output = testing::internal::GetCapturedStdout();
  return {result, stdout_output, stderr_output};
}

std::string expected_invalid_value(const std::string& option,
                                   const std::string& value,
                                   const std::string& reason) {
  return Messages::error_prefix() +
         Messages::error_invalid_value(option, value, reason);
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
    const CapturedCoreCliParse parsed =
        parse_capturing_stderr({"memory_benchmark", "--analyze-core2core", "-h"}, config);
    EXPECT_EQ(parsed.result, EXIT_SUCCESS);
    EXPECT_TRUE(parsed.stderr_output.empty());
    EXPECT_FALSE(parsed.stdout_output.empty());
    EXPECT_TRUE(config.help_requested);
  }

  {
    SCOPED_TRACE("inline-equals-options");
    CoreToCoreLatencyConfig config;
    const CapturedCoreCliParse parsed = parse_capturing_stderr(
        {"memory_benchmark", "--analyze-core2core", "--count=5", "--latency-samples=128", "--output=-"}, config);
    ASSERT_EQ(parsed.result, EXIT_SUCCESS) << parsed.stderr_output;
    EXPECT_TRUE(parsed.stderr_output.empty());
    EXPECT_EQ(config.loop_count, 5);
    EXPECT_EQ(config.latency_sample_count, 128);
    EXPECT_EQ(config.output_file, "-");

    const CapturedCoreCliParse empty_output =
        parse_capturing_stderr({"memory_benchmark", "--analyze-core2core", "--output="}, config);
    ASSERT_EQ(empty_output.result, EXIT_SUCCESS) << empty_output.stderr_output;
    EXPECT_TRUE(config.output_file.empty());
  }

  {
    SCOPED_TRACE("config-reset-on-reuse");
    CoreToCoreLatencyConfig config;
    config.loop_count = 99;
    config.latency_sample_count = 777;
    config.output_file = "stale.json";
    config.help_requested = true;
    config.run_sweep = true;
    config.sweep_max_runs = 2u;
    config.sweep_specs.push_back(
        CoreToCoreSweepSpec{CoreToCoreSweepParameter::Count, "count", {CoreToCoreSweepValue{"1", 1}}});

    const CapturedCoreCliParse parsed = parse_capturing_stderr({"memory_benchmark", "--analyze-core2core"}, config);
    ASSERT_EQ(parsed.result, EXIT_SUCCESS) << parsed.stderr_output;
    EXPECT_FALSE(config.help_requested);
    EXPECT_FALSE(config.run_sweep);
    EXPECT_TRUE(config.output_file.empty());
    EXPECT_TRUE(config.sweep_specs.empty());
    EXPECT_EQ(config.loop_count, Constants::CORE_TO_CORE_DEFAULT_LOOP_COUNT);
    EXPECT_EQ(config.latency_sample_count, Constants::CORE_TO_CORE_DEFAULT_LATENCY_SAMPLE_COUNT);
    EXPECT_EQ(config.sweep_max_runs, Constants::DEFAULT_SWEEP_MAX_RUNS);
  }

  {
    SCOPED_TRACE("argc-zero-and-null-program-name-safety");
    CoreToCoreLatencyConfig config;
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    const int empty_status = parse_core_to_core_mode_arguments(0, nullptr, config);
    const std::string empty_stderr = testing::internal::GetCapturedStderr();
    const std::string empty_stdout = testing::internal::GetCapturedStdout();
    EXPECT_EQ(empty_status, EXIT_FAILURE);
    EXPECT_TRUE(empty_stdout.empty());
    EXPECT_EQ(empty_stderr,
              Messages::error_prefix() + Messages::error_analyze_core_to_core_must_be_used_alone() + "\n");

    std::string mode_flag = "--analyze-core2core";
    std::string help_flag = "--help";
    char* null_prog_argv[] = {nullptr, mode_flag.data(), help_flag.data()};
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    const int help_status = parse_core_to_core_mode_arguments(3, null_prog_argv, config);
    const std::string help_stderr = testing::internal::GetCapturedStderr();
    const std::string help_stdout = testing::internal::GetCapturedStdout();
    EXPECT_EQ(help_status, EXIT_SUCCESS);
    EXPECT_TRUE(help_stderr.empty());
    EXPECT_TRUE(config.help_requested);
    EXPECT_NE(help_stdout.find("memory_benchmark"), std::string::npos);
  }
}

TEST(CoreToCoreCliTest, ParsesSweepArguments) {
  {
    SCOPED_TRACE("space-separated-sweep");
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

  {
    SCOPED_TRACE("inline-equals-sweep");
    CoreToCoreLatencyConfig config;
    const CapturedCoreCliParse parsed = parse_capturing_stderr(
        {"memory_benchmark", "--analyze-core2core", "--output=core2core_sweep.json", "--sweep=count=1,2",
         "--sweep-max-runs=4"},
        config);
    ASSERT_EQ(parsed.result, EXIT_SUCCESS) << parsed.stderr_output;
    EXPECT_TRUE(parsed.stderr_output.empty());
    EXPECT_TRUE(config.run_sweep);
    EXPECT_EQ(config.output_file, "core2core_sweep.json");
    EXPECT_EQ(config.sweep_max_runs, 4u);
    ASSERT_EQ(config.sweep_specs.size(), 1u);
    EXPECT_EQ(config.sweep_specs[0].parameter, CoreToCoreSweepParameter::Count);
    ASSERT_EQ(config.sweep_specs[0].values.size(), 2u);
    EXPECT_EQ(config.sweep_specs[0].values[0].integer_value, 1);
    EXPECT_EQ(config.sweep_specs[0].values[1].integer_value, 2);
  }
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
      {{"memory_benchmark", "-C", "-C"}, Messages::error_duplicate_option("--analyze-core2core")},
      {{"memory_benchmark", "-C", "--analyze-core2core"}, Messages::error_duplicate_option("--analyze-core2core")},
      {{"memory_benchmark", "-C", "-h", "--help"}, Messages::error_duplicate_option("--help")},
      {{"memory_benchmark", "--analyze-core2core=1"},
       Messages::error_option_does_not_take_value("--analyze-core2core")},
      {{"memory_benchmark", "--analyze-core2core", "--help=1"}, Messages::error_option_does_not_take_value("--help")},
      {{"memory_benchmark", "--analyze-core2core", "--buffer-size", "256"},
       Messages::error_option_not_allowed_in_mode("--buffer-size",
                                                  Messages::error_analyze_core_to_core_must_be_used_alone())},
      {{"memory_benchmark", "--analyze-core2core", "--help", "--unknown"},
       Messages::error_option_not_allowed_in_mode("--unknown",
                                                  Messages::error_analyze_core_to_core_must_be_used_alone())},
      {{"memory_benchmark", "--analyze-core2core", "--help", "--count", "5junk"},
       Messages::error_invalid_value("--count", "5junk", Messages::reason_signed_integer_invalid())},
      {{"memory_benchmark", "--help"}, Messages::error_analyze_core_to_core_must_be_used_alone()},
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
    EXPECT_TRUE(parsed.stdout_output.empty());
    EXPECT_EQ(parsed.stderr_output, Messages::error_prefix() + test_case.diagnostic + "\n");
  }
}

TEST(CoreToCoreCliTest, RejectsMalformedNumericTokensWithCentralizedErrors) {
  struct InvalidNumericCase {
    const char* option;
    const char* value;
    std::string reason;
  };

  const std::string invalid_reason = Messages::reason_signed_integer_invalid();
  const std::string positive_range_reason = Messages::reason_positive_int_range(std::numeric_limits<int>::max());
  const InvalidNumericCase cases[] = {
      {"--count", "5junk", invalid_reason},
      {"--latency-samples", "8junk", invalid_reason},
      {"--sweep-max-runs", "4junk", invalid_reason},
      {"--count", "0", positive_range_reason},
      {"--count", "-1", positive_range_reason},
  };

  for (const InvalidNumericCase& test_case : cases) {
    SCOPED_TRACE(test_case.option);
    SCOPED_TRACE(test_case.value);
    CoreToCoreLatencyConfig config;
    const CapturedCoreCliParse parsed =
        parse_capturing_stderr({"memory_benchmark", "--analyze-core2core", test_case.option, test_case.value}, config);
    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    EXPECT_TRUE(parsed.stdout_output.empty());
    EXPECT_EQ(parsed.stderr_output, expected_invalid_value(test_case.option, test_case.value, test_case.reason) + "\n");
  }
}

TEST(CoreToCoreCliTest, RejectsInvalidAndMalformedSweepConfigurations) {
  struct SweepErrorCase {
    std::vector<std::string> options;
    std::string expected_first_line;
  };

  const std::string invalid_reason = Messages::reason_signed_integer_invalid();
  const SweepErrorCase cases[] = {
      {{"--sweep", "count=1,2"}, Messages::error_prefix() + Messages::error_sweep_requires_output()},
      {{"--output", "sweep.json", "--sweep", "count=1,2", "--sweep", "latency-samples=4,8", "--sweep-max-runs", "3"},
       Messages::error_prefix() + Messages::error_sweep_too_many_runs(4, 3)},
      {{"--output", "sweep.json", "--sweep", "count=1,2", "--sweep", "count=3,4"},
       Messages::error_prefix() + Messages::error_duplicate_sweep_parameter("count")},
      {{"--count", "5", "--sweep", "count=1,2"},
       Messages::error_prefix() + Messages::error_sweep_option_conflict("--count", "count")},
      {{"--output", "sweep.json", "-r", "5", "--sweep=count=1,2"},
       Messages::error_prefix() + Messages::error_sweep_option_conflict("--count", "count")},
      {{"--latency-samples", "10", "--sweep", "latency-samples=4,8"},
       Messages::error_prefix() + Messages::error_sweep_option_conflict("--latency-samples", "latency-samples")},
      {{"--output", "sweep.json", "-n", "10", "--sweep=latency-samples=4,8"},
       Messages::error_prefix() + Messages::error_sweep_option_conflict("--latency-samples", "latency-samples")},
      {{"--output", "core2core_sweep.json", "--sweep", "threads=1,2"},
       expected_invalid_value("--sweep", "threads=1,2",
                              Messages::reason_unsupported_core_to_core_sweep_parameter("threads"))},
      {{"--output", "core2core_sweep.json", "--sweep", "count"},
       expected_invalid_value("--sweep", "count", Messages::reason_sweep_syntax())},
      {{"--output", "core2core_sweep.json", "--sweep", "count=,1"},
       expected_invalid_value("--sweep", "count=,1", Messages::reason_sweep_empty_value())},
      {{"--output", "core2core_sweep.json", "--sweep", "count=1,"},
       expected_invalid_value("--sweep", "count=1,", Messages::reason_sweep_empty_value())},
      {{"--output", "core2core_sweep.json", "--sweep", "count=1,,2"},
       expected_invalid_value("--sweep", "count=1,,2", Messages::reason_sweep_empty_value())},
      {{"--output", "core2core_sweep.json", "--sweep", "count=1x"},
       expected_invalid_value("--count", "1x", invalid_reason)},
      {{"--output", "core2core_sweep.json", "--sweep", "latency-samples=4x"},
       expected_invalid_value("--latency-samples", "4x", invalid_reason)},
  };

  for (const SweepErrorCase& test_case : cases) {
    SCOPED_TRACE(::testing::PrintToString(test_case.options));
    std::vector<std::string> arguments = {"memory_benchmark", "--analyze-core2core"};
    arguments.insert(arguments.end(), test_case.options.begin(), test_case.options.end());
    CoreToCoreLatencyConfig config;
    const CapturedCoreCliParse parsed = parse_capturing_stderr(arguments, config);
    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    EXPECT_TRUE(parsed.stdout_output.empty());
    EXPECT_EQ(parsed.stderr_output, test_case.expected_first_line + "\n");
  }
}

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

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "core/config/version.h"
#include "llm_memory/llm_memory.h"
#include "output/console/messages/messages_api.h"

namespace {

LlmMemoryConfig valid_config() {
  LlmMemoryConfig config;
  config.weight_size_mb = 4096;
  config.layer_count = 32;
  config.query_head_count = 32;
  config.kv_head_count = 8;
  config.head_dimension = 128;
  config.visible_context_tokens = 8192;
  config.requested_workers = 8;
  return config;
}

void expect_invalid(const LlmMemoryConfig& config,
                    const std::string& expected_reason) {
  const LlmMemoryConfigValidation validation =
      validate_llm_memory_config(config);
  EXPECT_FALSE(validation.valid);
  EXPECT_EQ(validation.reason_code, expected_reason);
  EXPECT_EQ(validation.active_weight_bytes, 0u);
}

std::vector<std::string> valid_llm_arguments() {
  return {"memory_benchmark", "--llm-memory",      "--weight-size-mb",
          "1",                "--layers",         "2",
          "--query-heads",    "4",                "--kv-heads",
          "2",                "--head-dim",       "8",
          "--context-tokens", "3"};
}

int parse_llm_arguments(std::vector<std::string> arguments,
                        LlmMemoryConfig& config) {
  std::vector<char*> argv;
  argv.reserve(arguments.size());
  for (std::string& argument : arguments) {
    argv.push_back(argument.data());
  }
  return parse_llm_memory_arguments(static_cast<int>(argv.size()),
                                    argv.data(), config);
}

struct CapturedLlmParse {
  int result = EXIT_FAILURE;
  std::string stdout_output;
  std::string stderr_output;
};

CapturedLlmParse parse_llm_arguments_capturing(
    std::vector<std::string> arguments, LlmMemoryConfig& config) {
  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  const int status = parse_llm_arguments(std::move(arguments), config);
  const std::string stderr_output = testing::internal::GetCapturedStderr();
  const std::string stdout_output = testing::internal::GetCapturedStdout();
  return {status, stdout_output, stderr_output};
}

int parse_llm_arguments_silently(std::vector<std::string> arguments,
                                 LlmMemoryConfig& config) {
  return parse_llm_arguments_capturing(std::move(arguments), config).result;
}

std::string first_output_line(const std::string& output) {
  return output.substr(0, output.find('\n'));
}

class LlmParserHooksScope {
 public:
  explicit LlmParserHooksScope(size_t detected_workers = 8,
                               uint64_t generated_seed = 0x123456789abcdef0ULL) {
    hooks_.detected_workers = detected_workers;
    hooks_.generated_seed = generated_seed;
    set_llm_memory_parser_test_hooks(&hooks_);
  }

  ~LlmParserHooksScope() { set_llm_memory_parser_test_hooks(nullptr); }

  LlmParserHooksScope(const LlmParserHooksScope&) = delete;
  LlmParserHooksScope& operator=(const LlmParserHooksScope&) = delete;

 private:
  LlmMemoryParserTestHooks hooks_;
};

void replace_option_value(std::vector<std::string>& arguments,
                          const std::string& option,
                          const std::string& value) {
  const auto position = std::find(arguments.begin(), arguments.end(), option);
  ASSERT_NE(position, arguments.end()) << option;
  ASSERT_NE(position + 1, arguments.end()) << option;
  *(position + 1) = value;
}

}  // namespace

TEST(LlmMemoryConfigTest, DefaultsMatchFrozenStandaloneContract) {
  const LlmMemoryConfig config;
  EXPECT_EQ(config.weight_size_mb, 0u);
  EXPECT_EQ(config.layer_count, 0u);
  EXPECT_EQ(config.query_head_count, 0u);
  EXPECT_EQ(config.kv_head_count, 0u);
  EXPECT_EQ(config.head_dimension, 0u);
  EXPECT_EQ(config.kv_element_bytes, 2u);
  EXPECT_EQ(config.visible_context_tokens, 0u);
  EXPECT_EQ(config.batch_size, 1u);
  EXPECT_EQ(config.requested_workers, 0u);
  EXPECT_EQ(config.available_workers, 0u);
  EXPECT_EQ(config.iterations, 0u);
  EXPECT_EQ(config.loop_count, 3u);
  EXPECT_EQ(config.seed, 0u);
  EXPECT_FALSE(config.user_specified_iterations);
  EXPECT_FALSE(config.user_specified_seed);
  EXPECT_FALSE(config.user_specified_workers);
  EXPECT_FALSE(config.help_printed);
  EXPECT_TRUE(config.output_file.empty());
  EXPECT_TRUE(config.argv.empty());

  const LlmMemoryConfigValidation validation =
      validate_llm_memory_config(config);
  EXPECT_FALSE(validation.valid);
  EXPECT_EQ(validation.reason_code,
            LlmMemoryConfigReason::WEIGHT_SIZE_REQUIRED);
}

TEST(LlmMemoryConfigTest,
     ParserResolvesRequiredInputsAndDeterministicDefaults) {
  LlmParserHooksScope hooks;
  const std::vector<std::string> arguments = valid_llm_arguments();
  LlmMemoryConfig config;

  ASSERT_EQ(parse_llm_arguments(arguments, config), EXIT_SUCCESS);
  EXPECT_EQ(config.weight_size_mb, 1u);
  EXPECT_EQ(config.layer_count, 2u);
  EXPECT_EQ(config.query_head_count, 4u);
  EXPECT_EQ(config.kv_head_count, 2u);
  EXPECT_EQ(config.head_dimension, 8u);
  EXPECT_EQ(config.kv_element_bytes, 2u);
  EXPECT_EQ(config.visible_context_tokens, 3u);
  EXPECT_EQ(config.batch_size, 1u);
  EXPECT_EQ(config.requested_workers, 8u);
  EXPECT_EQ(config.available_workers, 8u);
  EXPECT_EQ(config.iterations, 0u);
  EXPECT_EQ(config.loop_count, 3u);
  EXPECT_EQ(config.seed, 0x123456789abcdef0ULL);
  EXPECT_FALSE(config.user_specified_iterations);
  EXPECT_FALSE(config.user_specified_seed);
  EXPECT_FALSE(config.user_specified_workers);
  EXPECT_FALSE(config.help_printed);
  EXPECT_TRUE(config.output_file.empty());
  EXPECT_EQ(config.argv, arguments);
}

TEST(LlmMemoryConfigTest, ParserPreservesEveryExplicitFieldAndAlias) {
  LlmParserHooksScope hooks;
  const std::vector<std::string> arguments = {
      "memory_benchmark",      "-M",
      "--weight-size-mb",      "4096",
      "--layers",              "32",
      "--query-heads",         "32",
      "--kv-heads",            "8",
      "--head-dim",            "128",
      "--kv-element-bytes",    "4",
      "--context-tokens",      "8192",
      "--batch-size",          "2",
      "-t",                    "3",
      "-i",                    "4",
      "-r",                    "5",
      "--seed",                "18446744073709551615",
      "-o",                    "./-",
  };
  LlmMemoryConfig config;

  ASSERT_EQ(parse_llm_arguments(arguments, config), EXIT_SUCCESS);
  EXPECT_EQ(config.weight_size_mb, 4096u);
  EXPECT_EQ(config.layer_count, 32u);
  EXPECT_EQ(config.query_head_count, 32u);
  EXPECT_EQ(config.kv_head_count, 8u);
  EXPECT_EQ(config.head_dimension, 128u);
  EXPECT_EQ(config.kv_element_bytes, 4u);
  EXPECT_EQ(config.visible_context_tokens, 8192u);
  EXPECT_EQ(config.batch_size, 2u);
  EXPECT_EQ(config.requested_workers, 3u);
  EXPECT_EQ(config.available_workers, 8u);
  EXPECT_EQ(config.iterations, 4u);
  EXPECT_EQ(config.loop_count, 5u);
  EXPECT_EQ(config.seed, std::numeric_limits<uint64_t>::max());
  EXPECT_TRUE(config.user_specified_iterations);
  EXPECT_TRUE(config.user_specified_seed);
  EXPECT_TRUE(config.user_specified_workers);
  EXPECT_EQ(config.output_file, "./-");
  EXPECT_EQ(config.argv, arguments);
}

TEST(LlmMemoryConfigTest,
     ParserRetainsExplicitWorkerRequestAboveDetectedAvailability) {
  LlmParserHooksScope hooks(4, 9);
  std::vector<std::string> arguments = valid_llm_arguments();
  arguments.insert(arguments.end(), {"--threads", "12", "--seed", "7"});
  LlmMemoryConfig config;

  ASSERT_EQ(parse_llm_arguments(arguments, config), EXIT_SUCCESS);
  EXPECT_EQ(config.requested_workers, 12u);
  EXPECT_EQ(config.available_workers, 4u);
  EXPECT_TRUE(config.user_specified_workers);
}

TEST(LlmMemoryConfigTest,
     ParserRejectsZeroDetectedWorkersBeforeGeneratingTheSeed) {
  LlmParserHooksScope hooks(0, 9);
  LlmMemoryConfig config;
  const CapturedLlmParse parsed =
      parse_llm_arguments_capturing(valid_llm_arguments(), config);

  EXPECT_EQ(parsed.result, EXIT_FAILURE);
  EXPECT_TRUE(parsed.stdout_output.empty());
  EXPECT_EQ(first_output_line(parsed.stderr_output),
            Messages::error_prefix() +
                Messages::error_llm_memory_config_invalid(
                    LlmMemoryConfigReason::AVAILABLE_WORKER_COUNT_REQUIRED));
  EXPECT_EQ(config.seed, 0u);
}

TEST(LlmMemoryConfigTest, ParserRetainsRawOutputTargetAndArgv) {
  LlmParserHooksScope hooks;
  for (const std::string& output_option : {"-o", "--output"}) {
    for (const std::string& target : {
             "", "-", "./-", "-M", "--llm-memory", "--gpu-bandwidth",
             "-o", "--output", "--help"}) {
      SCOPED_TRACE(output_option + "=" + target);
      std::vector<std::string> arguments = valid_llm_arguments();
      arguments.insert(arguments.end(), {output_option, target});
      LlmMemoryConfig config;

      ASSERT_EQ(parse_llm_arguments(arguments, config), EXIT_SUCCESS);
      EXPECT_EQ(config.output_file, target);
      EXPECT_EQ(config.argv, arguments);
    }
  }
}

TEST(LlmMemoryConfigTest, ParserRequiresEveryModelGeometryOption) {
  LlmParserHooksScope hooks;
  for (const std::string& required_option : {
           "--weight-size-mb", "--layers", "--query-heads", "--kv-heads",
           "--head-dim", "--context-tokens"}) {
    SCOPED_TRACE(required_option);
    std::vector<std::string> arguments = valid_llm_arguments();
    const auto position =
        std::find(arguments.begin(), arguments.end(), required_option);
    ASSERT_NE(position, arguments.end());
    arguments.erase(position, position + 2);
    LlmMemoryConfig config;
    const CapturedLlmParse parsed =
        parse_llm_arguments_capturing(arguments, config);

    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    EXPECT_TRUE(parsed.stdout_output.empty());
    EXPECT_EQ(first_output_line(parsed.stderr_output),
              Messages::error_prefix() +
                  Messages::error_llm_memory_missing_required_option(
                      required_option));
  }
}

TEST(LlmMemoryConfigTest,
     ParserRejectsMissingValuesForEveryValueTakingOption) {
  LlmParserHooksScope hooks;
  for (const std::string& option : {
           "--weight-size-mb", "--layers", "--query-heads", "--kv-heads",
           "--head-dim", "--kv-element-bytes", "--context-tokens",
           "--batch-size", "--threads", "--iterations", "--count",
           "--seed", "--output"}) {
    SCOPED_TRACE(option);
    LlmMemoryConfig config;
    const CapturedLlmParse parsed = parse_llm_arguments_capturing(
        {"memory_benchmark", "--llm-memory", option}, config);

    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    EXPECT_EQ(first_output_line(parsed.stderr_output),
              Messages::error_prefix() +
                  Messages::error_missing_value(option));
  }
}

TEST(LlmMemoryConfigTest,
     ParserRejectsDuplicateCanonicalAndMixedAliasOptions) {
  LlmParserHooksScope hooks;
  struct DuplicateCase {
    std::vector<std::string> suffix;
    std::string canonical_option;
  };
  const std::vector<DuplicateCase> cases = {
      {{"--llm-memory"}, "--llm-memory"},
      {{"--weight-size-mb", "1"}, "--weight-size-mb"},
      {{"--layers", "2"}, "--layers"},
      {{"--query-heads", "4"}, "--query-heads"},
      {{"--kv-heads", "2"}, "--kv-heads"},
      {{"--head-dim", "8"}, "--head-dim"},
      {{"--kv-element-bytes", "2", "--kv-element-bytes", "2"},
       "--kv-element-bytes"},
      {{"--context-tokens", "3"}, "--context-tokens"},
      {{"--batch-size", "1", "--batch-size", "1"}, "--batch-size"},
      {{"-t", "2", "--threads", "2"}, "--threads"},
      {{"-i", "1", "--iterations", "1"}, "--iterations"},
      {{"-r", "3", "--count", "3"}, "--count"},
      {{"--seed", "1", "--seed", "2"}, "--seed"},
      {{"-o", "first", "--output", "second"}, "--output"},
      {{"-h", "--help"}, "--help"},
  };

  for (const DuplicateCase& test_case : cases) {
    SCOPED_TRACE(test_case.canonical_option);
    std::vector<std::string> arguments = valid_llm_arguments();
    arguments.insert(arguments.end(), test_case.suffix.begin(),
                     test_case.suffix.end());
    LlmMemoryConfig config;
    const CapturedLlmParse parsed =
        parse_llm_arguments_capturing(arguments, config);

    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    EXPECT_EQ(first_output_line(parsed.stderr_output),
              Messages::error_prefix() +
                  Messages::error_duplicate_option(
                      test_case.canonical_option));
  }
}

TEST(LlmMemoryConfigTest,
     ParserUsesStrictPositiveDecimalTokensForAllSizeFields) {
  LlmParserHooksScope hooks;
  const std::vector<std::string> invalid_tokens = {
      "0",  "-1", "+1", " 1", "1 ", "1x", "1.0", "0x10",
      "18446744073709551616"};
  const std::vector<std::string> options = {
      "--weight-size-mb", "--layers", "--query-heads", "--kv-heads",
      "--head-dim",       "--kv-element-bytes", "--context-tokens",
      "--batch-size",     "--threads",          "--iterations",
      "--count"};

  for (const std::string& option : options) {
    for (const std::string& token : invalid_tokens) {
      SCOPED_TRACE(option + "=" + token);
      std::vector<std::string> arguments = valid_llm_arguments();
      const auto position =
          std::find(arguments.begin(), arguments.end(), option);
      if (position == arguments.end()) {
        arguments.insert(arguments.end(), {option, token});
      } else {
        *(position + 1) = token;
      }
      LlmMemoryConfig config;
      EXPECT_EQ(parse_llm_arguments_silently(arguments, config),
                EXIT_FAILURE);
    }
  }
}

TEST(LlmMemoryConfigTest, ParserUsesStrictUnsignedSeedTokens) {
  LlmParserHooksScope hooks;
  for (const std::string& token : {
           "-1", "+1", " 1", "1 ", "1x", "1.0", "0x10",
           "18446744073709551616"}) {
    SCOPED_TRACE(token);
    std::vector<std::string> arguments = valid_llm_arguments();
    arguments.insert(arguments.end(), {"--seed", token});
    LlmMemoryConfig config;
    EXPECT_EQ(parse_llm_arguments_silently(arguments, config),
              EXIT_FAILURE);
  }

  for (const std::string& token : {"0", "18446744073709551615"}) {
    SCOPED_TRACE(token);
    std::vector<std::string> arguments = valid_llm_arguments();
    arguments.insert(arguments.end(), {"--seed", token});
    LlmMemoryConfig config;
    ASSERT_EQ(parse_llm_arguments(arguments, config), EXIT_SUCCESS);
    EXPECT_TRUE(config.user_specified_seed);
    EXPECT_EQ(config.seed, token == "0"
                               ? 0u
                               : std::numeric_limits<uint64_t>::max());
  }
}

TEST(LlmMemoryConfigTest,
     ParserAcceptsOnlyFrozenKvElementWidthsAndHeadSharing) {
  LlmParserHooksScope hooks;
  for (const std::string& width : {"1", "2", "4"}) {
    SCOPED_TRACE(width);
    std::vector<std::string> arguments = valid_llm_arguments();
    arguments.insert(arguments.end(), {"--kv-element-bytes", width});
    LlmMemoryConfig config;
    ASSERT_EQ(parse_llm_arguments(arguments, config), EXIT_SUCCESS);
    EXPECT_EQ(config.kv_element_bytes,
              static_cast<size_t>(std::stoul(width)));
  }

  for (const std::string& width : {"3", "5", "8"}) {
    SCOPED_TRACE(width);
    std::vector<std::string> arguments = valid_llm_arguments();
    arguments.insert(arguments.end(), {"--kv-element-bytes", width});
    LlmMemoryConfig config;
    EXPECT_EQ(parse_llm_arguments_silently(arguments, config),
              EXIT_FAILURE);
  }

  std::vector<std::string> arguments = valid_llm_arguments();
  replace_option_value(arguments, "--query-heads", "1");
  LlmMemoryConfig config;
  EXPECT_EQ(parse_llm_arguments_silently(arguments, config), EXIT_FAILURE);

  arguments = valid_llm_arguments();
  replace_option_value(arguments, "--query-heads", "3");
  EXPECT_EQ(parse_llm_arguments_silently(arguments, config), EXIT_FAILURE);
}

TEST(LlmMemoryConfigTest,
     ParserRejectsGeometryAndExplicitWorkBeyondFrozenGuardrails) {
  LlmParserHooksScope hooks;
  LlmMemoryConfig config;

  std::vector<std::string> arguments = valid_llm_arguments();
  replace_option_value(arguments, "--weight-size-mb", "65537");
  EXPECT_EQ(parse_llm_arguments_silently(arguments, config), EXIT_FAILURE);

  arguments = valid_llm_arguments();
  replace_option_value(arguments, "--layers", "18446744073709551615");
  EXPECT_EQ(parse_llm_arguments_silently(arguments, config), EXIT_FAILURE);

  arguments = valid_llm_arguments();
  arguments.insert(arguments.end(), {"--iterations", "1000000001"});
  EXPECT_EQ(parse_llm_arguments_silently(arguments, config), EXIT_FAILURE);
}

TEST(LlmMemoryConfigTest,
     ParserBracketsExactWeightAndScenarioWorkGuardrails) {
  LlmParserHooksScope hooks;
  LlmMemoryConfig config;

  std::vector<std::string> arguments = valid_llm_arguments();
  replace_option_value(arguments, "--weight-size-mb", "65535");
  EXPECT_EQ(parse_llm_arguments(arguments, config), EXIT_SUCCESS);

  replace_option_value(arguments, "--weight-size-mb", "65536");
  EXPECT_EQ(parse_llm_arguments_silently(arguments, config), EXIT_FAILURE);

  arguments = valid_llm_arguments();
  arguments.insert(arguments.end(), {"--iterations", "65504"});
  EXPECT_EQ(parse_llm_arguments(arguments, config), EXIT_SUCCESS);

  replace_option_value(arguments, "--iterations", "65505");
  const CapturedLlmParse parsed =
      parse_llm_arguments_capturing(arguments, config);
  EXPECT_EQ(parsed.result, EXIT_FAILURE);
  EXPECT_TRUE(parsed.stdout_output.empty());
  EXPECT_EQ(first_output_line(parsed.stderr_output),
            Messages::error_prefix() +
                Messages::error_llm_memory_iterations_exceed_limit(
                    65505, 65504));
}

TEST(LlmMemoryConfigTest, ParserRejectsEveryIncompatibleOption) {
  LlmParserHooksScope hooks;
  const std::vector<std::vector<std::string>> suffixes = {
      {"--benchmark"},
      {"-B"},
      {"--patterns"},
      {"-P"},
      {"--analyze-tlb"},
      {"-T"},
      {"--analyze-core2core"},
      {"-C"},
      {"--gpu-bandwidth"},
      {"-G"},
      {"--only-bandwidth"},
      {"-W"},
      {"--only-latency"},
      {"-L"},
      {"--buffer-size", "1"},
      {"-b", "1"},
      {"--cache-size", "16"},
      {"-k", "16"},
      {"--latency-samples", "1"},
      {"-n", "1"},
      {"--latency-stride-bytes", "256"},
      {"-s", "256"},
      {"--latency-chain-mode", "auto"},
      {"-m", "auto"},
      {"--latency-tlb-locality-kb", "1024"},
      {"-l", "1024"},
      {"--tlb-density", "medium"},
      {"-D", "medium"},
      {"--non-cacheable"},
      {"-u"},
      {"--sweep", "threads=1"},
      {"-S", "threads=1"},
      {"--sweep-max-runs", "1"},
      {"-X", "1"},
      {"--unknown"},
  };
  for (const std::vector<std::string>& suffix : suffixes) {
    SCOPED_TRACE(::testing::PrintToString(suffix));
    std::vector<std::string> arguments = valid_llm_arguments();
    arguments.insert(arguments.end(), suffix.begin(), suffix.end());
    LlmMemoryConfig config;
    EXPECT_EQ(parse_llm_arguments_silently(arguments, config),
              EXIT_FAILURE);
  }
}

TEST(LlmMemoryConfigTest,
     ParserHelpSkipsRequiredInputsWorkerDetectionAndSeedGeneration) {
  LlmParserHooksScope hooks(0, 0);
  LlmMemoryConfig config;
  const CapturedLlmParse parsed = parse_llm_arguments_capturing(
      {"memory_benchmark", "-M", "--help"}, config);

  EXPECT_EQ(parsed.result, EXIT_SUCCESS);
  EXPECT_TRUE(parsed.stderr_output.empty()) << parsed.stderr_output;
  EXPECT_NE(parsed.stdout_output.find(
                "Usage: memory_benchmark --llm-memory [options]"),
            std::string::npos);
  EXPECT_NE(parsed.stdout_output.find("memory traffic only"),
            std::string::npos);
  EXPECT_EQ(parsed.stdout_output.find(Messages::config_header(SOFTVERSION)),
            std::string::npos);
  EXPECT_TRUE(config.help_printed);
  EXPECT_EQ(config.available_workers, 0u);
  EXPECT_EQ(config.requested_workers, 0u);
  EXPECT_EQ(config.seed, 0u);
}

TEST(LlmMemoryConfigTest,
     ParserHelpDoesNotHideUnknownDuplicateOrMissingValueErrors) {
  LlmParserHooksScope hooks;
  for (const std::vector<std::string>& arguments : {
           std::vector<std::string>{"memory_benchmark", "-M", "--help",
                                    "--unknown"},
           std::vector<std::string>{"memory_benchmark", "-M", "--help",
                                    "-h"},
           std::vector<std::string>{"memory_benchmark", "-M", "--help",
                                    "--layers"},
       }) {
    SCOPED_TRACE(::testing::PrintToString(arguments));
    LlmMemoryConfig config;
    const CapturedLlmParse parsed =
        parse_llm_arguments_capturing(arguments, config);
    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    EXPECT_TRUE(parsed.stdout_output.empty());
    EXPECT_FALSE(parsed.stderr_output.empty());
  }
}

TEST(LlmMemoryConfigTest, ParserRequiresTheOwningPrimaryMode) {
  LlmParserHooksScope hooks;
  LlmMemoryConfig config;
  const CapturedLlmParse parsed = parse_llm_arguments_capturing(
      {"memory_benchmark", "--help"}, config);

  EXPECT_EQ(parsed.result, EXIT_FAILURE);
  EXPECT_TRUE(parsed.stdout_output.empty());
  EXPECT_EQ(first_output_line(parsed.stderr_output),
            Messages::error_prefix() +
                Messages::error_llm_memory_must_be_used_alone());
}

TEST(LlmMemoryConfigTest, StableScenarioAttentionAndStatusTokens) {
  EXPECT_STREQ(llm_scenario_to_string(LlmScenario::WeightsOnly),
               "weights_only");
  EXPECT_STREQ(llm_scenario_to_string(LlmScenario::KvOnly), "kv_only");
  EXPECT_STREQ(llm_scenario_to_string(LlmScenario::Mixed), "mixed");
  EXPECT_STREQ(llm_scenario_to_string(static_cast<LlmScenario>(99)),
               "unknown");

  EXPECT_STREQ(llm_attention_kind_to_string(LlmAttentionKind::Mha), "mha");
  EXPECT_STREQ(llm_attention_kind_to_string(LlmAttentionKind::Gqa), "gqa");
  EXPECT_STREQ(llm_attention_kind_to_string(LlmAttentionKind::Mqa), "mqa");
  EXPECT_STREQ(
      llm_attention_kind_to_string(static_cast<LlmAttentionKind>(99)),
      "unknown");

  EXPECT_STREQ(
      llm_measurement_status_to_string(LlmMeasurementStatus::NotRun),
      "not_run");
  EXPECT_STREQ(
      llm_measurement_status_to_string(LlmMeasurementStatus::Measured),
      "measured");
  EXPECT_STREQ(
      llm_measurement_status_to_string(LlmMeasurementStatus::Interrupted),
      "interrupted");
  EXPECT_STREQ(
      llm_measurement_status_to_string(LlmMeasurementStatus::Invalid),
      "invalid");
  EXPECT_STREQ(
      llm_measurement_status_to_string(LlmMeasurementStatus::Failed),
      "failed");
  EXPECT_STREQ(llm_measurement_status_to_string(
                   static_cast<LlmMeasurementStatus>(99)),
               "invalid");

  EXPECT_STREQ(llm_run_status_to_string(LlmRunStatus::NotStarted),
               "not_started");
  EXPECT_STREQ(llm_run_status_to_string(LlmRunStatus::Complete), "complete");
  EXPECT_STREQ(llm_run_status_to_string(LlmRunStatus::Partial), "partial");
  EXPECT_STREQ(llm_run_status_to_string(LlmRunStatus::Interrupted),
               "interrupted");
  EXPECT_STREQ(llm_run_status_to_string(LlmRunStatus::Failed), "failed");
  EXPECT_STREQ(llm_run_status_to_string(static_cast<LlmRunStatus>(99)),
               "failed");
}

TEST(LlmMemoryConfigTest, ValidatesPositiveGeometryAndHeadSharing) {
  LlmMemoryConfig config = valid_config();
  LlmMemoryConfigValidation validation = validate_llm_memory_config(config);
  ASSERT_TRUE(validation.valid) << validation.reason_code;
  EXPECT_EQ(validation.reason_code, LlmMemoryConfigReason::VALID);
  EXPECT_EQ(validation.active_weight_bytes, 4ULL * 1024ULL * 1024ULL * 1024ULL);

  config = valid_config();
  config.weight_size_mb = 0;
  expect_invalid(config, LlmMemoryConfigReason::WEIGHT_SIZE_REQUIRED);
  config = valid_config();
  config.layer_count = 0;
  expect_invalid(config, LlmMemoryConfigReason::LAYER_COUNT_REQUIRED);
  config = valid_config();
  config.query_head_count = 0;
  expect_invalid(config, LlmMemoryConfigReason::QUERY_HEAD_COUNT_REQUIRED);
  config = valid_config();
  config.kv_head_count = 0;
  expect_invalid(config, LlmMemoryConfigReason::KV_HEAD_COUNT_REQUIRED);
  config = valid_config();
  config.head_dimension = 0;
  expect_invalid(config, LlmMemoryConfigReason::HEAD_DIMENSION_REQUIRED);
  for (size_t invalid_width : {0u, 3u, 5u, 8u}) {
    config = valid_config();
    config.kv_element_bytes = invalid_width;
    expect_invalid(config, LlmMemoryConfigReason::INVALID_KV_ELEMENT_BYTES);
  }
  config = valid_config();
  config.visible_context_tokens = 0;
  expect_invalid(config, LlmMemoryConfigReason::CONTEXT_TOKENS_REQUIRED);
  config = valid_config();
  config.batch_size = 0;
  expect_invalid(config, LlmMemoryConfigReason::BATCH_SIZE_REQUIRED);
  config = valid_config();
  config.requested_workers = 0;
  expect_invalid(config, LlmMemoryConfigReason::WORKER_COUNT_REQUIRED);
  config = valid_config();
  config.loop_count = 0;
  expect_invalid(config, LlmMemoryConfigReason::LOOP_COUNT_REQUIRED);
  config = valid_config();
  config.query_head_count = 4;
  config.kv_head_count = 8;
  expect_invalid(config, LlmMemoryConfigReason::QUERY_HEADS_BELOW_KV_HEADS);
  config = valid_config();
  config.query_head_count = 10;
  config.kv_head_count = 4;
  expect_invalid(
      config,
      LlmMemoryConfigReason::QUERY_HEADS_NOT_DIVISIBLE_BY_KV_HEADS);
  config = valid_config();
  config.user_specified_iterations = true;
  config.iterations = 0;
  expect_invalid(config, LlmMemoryConfigReason::EXPLICIT_ITERATIONS_REQUIRED);
  config = valid_config();
  config.iterations = 4;
  expect_invalid(
      config, LlmMemoryConfigReason::AUTOMATIC_ITERATIONS_MUST_BE_ZERO);

  config = valid_config();
  config.user_specified_iterations = true;
  config.iterations = 4;
  EXPECT_TRUE(validate_llm_memory_config(config).valid);
  config = valid_config();
  config.query_head_count = config.kv_head_count;
  EXPECT_TRUE(validate_llm_memory_config(config).valid);
  config = valid_config();
  config.query_head_count = 8;
  config.kv_head_count = 1;
  EXPECT_TRUE(validate_llm_memory_config(config).valid);
}

TEST(LlmMemoryConfigTest, ConvertsWeightMiBWithCheckedArithmetic) {
  LlmMemoryConfig config = valid_config();
  config.weight_size_mb = 4096;
  LlmMemoryConfigValidation validation = validate_llm_memory_config(config);
  ASSERT_TRUE(validation.valid);
  EXPECT_EQ(validation.active_weight_bytes, 4294967296ULL);

  const size_t maximum_mb =
      std::numeric_limits<size_t>::max() / Constants::BYTES_PER_MB;
  config.weight_size_mb = maximum_mb;
  validation = validate_llm_memory_config(config);
  ASSERT_TRUE(validation.valid);
  EXPECT_EQ(validation.active_weight_bytes,
            maximum_mb * Constants::BYTES_PER_MB);

  config.weight_size_mb = maximum_mb + 1;
  expect_invalid(config, LlmMemoryConfigReason::WEIGHT_SIZE_BYTES_OVERFLOW);
}

TEST(LlmMemoryConfigTest, ResultFoundationKeepsUnavailableValuesAbsent) {
  const LlmMeasurementState measurement;
  EXPECT_EQ(measurement.status, LlmMeasurementStatus::NotRun);
  EXPECT_EQ(measurement.reason_code, "not-run");
  EXPECT_EQ(measurement.planned_steps, 0u);
  EXPECT_EQ(measurement.completed_steps, 0u);
  EXPECT_FALSE(measurement.elapsed_seconds.has_value());
  EXPECT_FALSE(measurement.effective_payload_gb_s.has_value());
  EXPECT_FALSE(measurement.checksum_valid);

  const LlmMemoryResult result;
  EXPECT_EQ(result.status, LlmRunStatus::NotStarted);
  EXPECT_EQ(result.reason_code, "not-started");
  EXPECT_FALSE(result.interruption_requested);
  EXPECT_FALSE(result.results_complete);
  EXPECT_FALSE(result.conclusions_valid);
  EXPECT_FALSE(result.scenario_order_balance_complete);
  EXPECT_TRUE(result.measurements.empty());
  EXPECT_EQ(result.counters.planned_measurements, 0u);
  EXPECT_EQ(result.counters.completed_exact_payload_bytes, 0u);
}

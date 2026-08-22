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
#include <mach/mach_error.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

#include "core/config/version.h"
#include "core/timing/timer.h"
#include "llm_memory/llm_memory.h"
#include "llm_memory/llm_runner.h"
#include "output/console/messages/messages_api.h"
#include "third_party/nlohmann/json.hpp"

namespace {

uint64_t unused_timer_ticks() { return 0; }

kern_return_t failing_timebase_info(mach_timebase_info_t) {
  return KERN_FAILURE;
}

class ScopedFailingTimerSystemCalls {
 public:
  ScopedFailingTimerSystemCalls() {
    set_timer_system_calls_for_testing(
        {unused_timer_ticks, failing_timebase_info});
  }

  ScopedFailingTimerSystemCalls(const ScopedFailingTimerSystemCalls&) =
      delete;
  ScopedFailingTimerSystemCalls& operator=(
      const ScopedFailingTimerSystemCalls&) = delete;

  ~ScopedFailingTimerSystemCalls() { reset_timer_system_calls_for_testing(); }
};

LlmMemoryConfig valid_config() {
  LlmMemoryConfig config;
  config.weight_size_mb = 4096;
  config.layer_count = 32;
  config.query_head_count = 32;
  config.kv_head_count = 8;
  config.head_dimension = 128;
  config.visible_context_tokens = 8192;
  config.user_specified_context_tokens = true;
  config.requested_workers = 8;
  return config;
}

LlmMemoryConfig valid_prefill_config() {
  LlmMemoryConfig config = valid_config();
  config.phase = LlmPhase::Prefill;
  config.visible_context_tokens = 0;
  config.prompt_tokens = 33;
  config.attention_query_tile_tokens = 16;
  config.user_specified_context_tokens = false;
  config.user_specified_prompt_tokens = true;
  config.user_specified_attention_query_tile_tokens = true;
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

std::vector<std::string> valid_prefill_arguments() {
  return {"memory_benchmark",
          "--llm-memory",
          "--weight-size-mb",
          "1",
          "--layers",
          "2",
          "--query-heads",
          "4",
          "--kv-heads",
          "2",
          "--head-dim",
          "8",
          "--phase",
          "prefill",
          "--prompt-tokens",
          "5",
          "--attention-query-tile-tokens",
          "2"};
}

std::vector<std::string> valid_metal_arguments() {
  std::vector<std::string> arguments = valid_llm_arguments();
  arguments.insert(arguments.begin() + 2,
                   {"--llm-memory-backend", "metal"});
  return arguments;
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

class LlmCommandHooksScope {
 public:
  explicit LlmCommandHooksScope(std::function<void()> after_runner) {
    hooks_.after_initialized_runner = std::move(after_runner);
    set_llm_memory_command_test_hooks(&hooks_);
  }

  ~LlmCommandHooksScope() { set_llm_memory_command_test_hooks(nullptr); }

  LlmCommandHooksScope(const LlmCommandHooksScope&) = delete;
  LlmCommandHooksScope& operator=(const LlmCommandHooksScope&) = delete;

 private:
  LlmMemoryCommandTestHooks hooks_;
};

class LlmCommandOutputFileScope {
 public:
  LlmCommandOutputFileScope()
      : path_(std::filesystem::path("/tmp") /
              ("membenchmark_llm_post_run_exception_" +
               std::to_string(::getpid()) + ".json")) {
    cleanup();
  }

  ~LlmCommandOutputFileScope() { cleanup(); }

  LlmCommandOutputFileScope(const LlmCommandOutputFileScope&) = delete;
  LlmCommandOutputFileScope& operator=(
      const LlmCommandOutputFileScope&) = delete;

  const std::filesystem::path& path() const { return path_; }

 private:
  void cleanup() noexcept {
    std::error_code ignored;
    std::filesystem::remove(path_, ignored);
    ignored.clear();
    std::filesystem::remove(path_.string() + ".tmp", ignored);
  }

  std::filesystem::path path_;
};

int run_llm_command(std::vector<std::string> arguments) {
  std::vector<char*> argv;
  argv.reserve(arguments.size());
  for (std::string& argument : arguments) {
    argv.push_back(argument.data());
  }
  return run_llm_memory_mode(static_cast<int>(argv.size()), argv.data());
}

std::string read_llm_command_output_file(
    const std::filesystem::path& path) {
  std::ifstream file(path);
  std::ostringstream contents;
  contents << file.rdbuf();
  return contents.str();
}

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
  EXPECT_EQ(config.backend, LlmMemoryBackend::Cpu);
  EXPECT_EQ(config.phase, LlmPhase::Decode);
  EXPECT_EQ(config.kv_layout, LlmKvLayout::Contiguous);
  EXPECT_EQ(config.weight_size_mb, 0u);
  EXPECT_EQ(config.layer_count, 0u);
  EXPECT_EQ(config.query_head_count, 0u);
  EXPECT_EQ(config.kv_head_count, 0u);
  EXPECT_EQ(config.head_dimension, 0u);
  EXPECT_EQ(config.kv_element_bytes, 2u);
  EXPECT_EQ(config.visible_context_tokens, 0u);
  EXPECT_EQ(config.prompt_tokens, 0u);
  EXPECT_EQ(config.attention_query_tile_tokens, 0u);
  EXPECT_EQ(config.kv_block_tokens, 0u);
  EXPECT_EQ(config.batch_size, 1u);
  EXPECT_EQ(config.requested_workers, 0u);
  EXPECT_EQ(config.available_workers, 0u);
  EXPECT_EQ(config.iterations, 0u);
  EXPECT_EQ(config.loop_count, 3u);
  EXPECT_EQ(config.seed, 0u);
  EXPECT_FALSE(config.user_specified_backend);
  EXPECT_FALSE(config.user_specified_iterations);
  EXPECT_FALSE(config.user_specified_seed);
  EXPECT_FALSE(config.user_specified_workers);
  EXPECT_FALSE(config.user_specified_phase);
  EXPECT_FALSE(config.user_specified_context_tokens);
  EXPECT_FALSE(config.user_specified_prompt_tokens);
  EXPECT_FALSE(config.user_specified_attention_query_tile_tokens);
  EXPECT_FALSE(config.user_specified_kv_layout);
  EXPECT_FALSE(config.user_specified_kv_block_tokens);
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
  EXPECT_EQ(config.backend, LlmMemoryBackend::Cpu);
  EXPECT_EQ(config.weight_size_mb, 1u);
  EXPECT_EQ(config.layer_count, 2u);
  EXPECT_EQ(config.query_head_count, 4u);
  EXPECT_EQ(config.kv_head_count, 2u);
  EXPECT_EQ(config.head_dimension, 8u);
  EXPECT_EQ(config.kv_element_bytes, 2u);
  EXPECT_EQ(config.visible_context_tokens, 3u);
  EXPECT_EQ(config.prompt_tokens, 0u);
  EXPECT_EQ(config.attention_query_tile_tokens, 0u);
  EXPECT_EQ(config.kv_layout, LlmKvLayout::Contiguous);
  EXPECT_EQ(config.kv_block_tokens, 0u);
  EXPECT_EQ(config.batch_size, 1u);
  EXPECT_EQ(config.requested_workers, 8u);
  EXPECT_EQ(config.available_workers, 8u);
  EXPECT_EQ(config.iterations, 0u);
  EXPECT_EQ(config.loop_count, 3u);
  EXPECT_EQ(config.seed, 0x123456789abcdef0ULL);
  EXPECT_FALSE(config.user_specified_backend);
  EXPECT_FALSE(config.user_specified_iterations);
  EXPECT_FALSE(config.user_specified_seed);
  EXPECT_FALSE(config.user_specified_workers);
  EXPECT_FALSE(config.user_specified_phase);
  EXPECT_TRUE(config.user_specified_context_tokens);
  EXPECT_FALSE(config.user_specified_prompt_tokens);
  EXPECT_FALSE(config.user_specified_attention_query_tile_tokens);
  EXPECT_FALSE(config.user_specified_kv_layout);
  EXPECT_FALSE(config.user_specified_kv_block_tokens);
  EXPECT_FALSE(config.help_printed);
  EXPECT_TRUE(config.output_file.empty());
  EXPECT_EQ(config.argv, arguments);
}

TEST(LlmMemoryConfigTest,
     ParserActivatesMetalDecodeContiguousWithoutCpuWorkerDetection) {
  LlmParserHooksScope hooks(0, 9);
  const std::vector<std::string> arguments = valid_metal_arguments();
  LlmMemoryConfig config;

  const CapturedLlmParse parsed =
      parse_llm_arguments_capturing(arguments, config);

  ASSERT_EQ(parsed.result, EXIT_SUCCESS) << parsed.stderr_output;
  EXPECT_TRUE(parsed.stdout_output.empty());
  EXPECT_TRUE(parsed.stderr_output.empty());
  EXPECT_EQ(config.backend, LlmMemoryBackend::Metal);
  EXPECT_EQ(config.phase, LlmPhase::Decode);
  EXPECT_EQ(config.kv_layout, LlmKvLayout::Contiguous);
  EXPECT_TRUE(config.user_specified_backend);
  EXPECT_FALSE(config.user_specified_workers);
  EXPECT_EQ(config.requested_workers, 0u);
  EXPECT_EQ(config.available_workers, 0u);
  EXPECT_EQ(config.seed, 9u);
  EXPECT_EQ(config.argv, arguments);
}

TEST(LlmMemoryConfigTest,
     ParserRejectsUnknownBackendWithCentralizedReason) {
  LlmParserHooksScope hooks;
  std::vector<std::string> arguments = valid_llm_arguments();
  arguments.insert(arguments.begin() + 2,
                   {"--llm-memory-backend", "gpu"});
  LlmMemoryConfig config;

  const CapturedLlmParse parsed =
      parse_llm_arguments_capturing(arguments, config);

  EXPECT_EQ(parsed.result, EXIT_FAILURE);
  EXPECT_TRUE(parsed.stdout_output.empty());
  EXPECT_EQ(first_output_line(parsed.stderr_output),
            Messages::error_prefix() +
                Messages::error_invalid_value(
                    "--llm-memory-backend", "gpu",
                    Messages::llm_memory_reason_backend()));
}

TEST(LlmMemoryConfigTest,
     ParserRejectsUnactivatedMetalProfilesWithStableOrderIndependentReasons) {
  LlmParserHooksScope hooks(0, 9);
  struct InvalidCase {
    std::vector<std::string> arguments;
    std::string reason_code;
  };

  std::vector<std::string> prefill_backend_first =
      valid_prefill_arguments();
  prefill_backend_first.insert(prefill_backend_first.begin() + 2,
                               {"--llm-memory-backend", "metal"});
  std::vector<std::string> prefill_backend_last =
      valid_prefill_arguments();
  prefill_backend_last.insert(prefill_backend_last.end(),
                              {"--llm-memory-backend", "metal"});
  std::vector<std::string> paged_backend_first = valid_metal_arguments();
  paged_backend_first.insert(paged_backend_first.end(),
                             {"--kv-layout", "paged",
                              "--kv-block-tokens", "4"});
  std::vector<std::string> paged_backend_last = valid_llm_arguments();
  paged_backend_last.insert(paged_backend_last.end(),
                            {"--kv-layout", "paged",
                             "--kv-block-tokens", "4",
                             "--llm-memory-backend", "metal"});
  std::vector<std::string> prefill_paged = prefill_backend_first;
  prefill_paged.insert(prefill_paged.end(),
                       {"--kv-layout", "paged", "--kv-block-tokens", "4"});

  const std::vector<InvalidCase> cases = {
      {prefill_backend_first,
       LlmMemoryConfigReason::PHASE_NOT_ACTIVATED},
      {prefill_backend_last,
       LlmMemoryConfigReason::PHASE_NOT_ACTIVATED},
      {paged_backend_first,
       LlmMemoryConfigReason::KV_LAYOUT_NOT_ACTIVATED},
      {paged_backend_last,
       LlmMemoryConfigReason::KV_LAYOUT_NOT_ACTIVATED},
      {prefill_paged, LlmMemoryConfigReason::PHASE_NOT_ACTIVATED},
  };

  for (const InvalidCase& test_case : cases) {
    SCOPED_TRACE(::testing::PrintToString(test_case.arguments));
    LlmMemoryConfig config;
    const CapturedLlmParse parsed =
        parse_llm_arguments_capturing(test_case.arguments, config);
    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    EXPECT_TRUE(parsed.stdout_output.empty());
    EXPECT_EQ(first_output_line(parsed.stderr_output),
              Messages::error_prefix() +
                  Messages::error_llm_memory_config_invalid(
                      test_case.reason_code));
    EXPECT_EQ(config.requested_workers, 0u);
    EXPECT_EQ(config.available_workers, 0u);
  }
}

TEST(LlmMemoryConfigTest,
     ParserRejectsMetalThreadsInEitherArgumentOrder) {
  LlmParserHooksScope hooks(0, 9);
  std::vector<std::string> backend_first = valid_metal_arguments();
  backend_first.insert(backend_first.end(), {"--threads", "1"});
  std::vector<std::string> threads_first = valid_llm_arguments();
  threads_first.insert(threads_first.begin() + 2,
                       {"--threads", "1"});
  threads_first.insert(threads_first.end(),
                       {"--llm-memory-backend", "metal"});

  for (const std::vector<std::string>& arguments :
       {backend_first, threads_first}) {
    SCOPED_TRACE(::testing::PrintToString(arguments));
    LlmMemoryConfig config;
    const CapturedLlmParse parsed =
        parse_llm_arguments_capturing(arguments, config);
    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    EXPECT_TRUE(parsed.stdout_output.empty());
    EXPECT_EQ(first_output_line(parsed.stderr_output),
              Messages::error_prefix() +
                  Messages::error_llm_memory_config_invalid(
                      LlmMemoryConfigReason::THREADS_NOT_APPLICABLE));
    EXPECT_TRUE(config.user_specified_workers);
    EXPECT_EQ(config.available_workers, 0u);
  }
}

TEST(LlmMemoryConfigTest, ParserPreservesEveryExplicitFieldAndAlias) {
  LlmParserHooksScope hooks;
  const std::vector<std::string> arguments = {
      "memory_benchmark",      "-M",
      "--llm-memory-backend",  "cpu",
      "--weight-size-mb",      "4096",
      "--layers",              "32",
      "--query-heads",         "32",
      "--kv-heads",            "8",
      "--head-dim",            "128",
      "--kv-element-bytes",    "4",
      "--phase",               "decode",
      "--context-tokens",      "8192",
      "--kv-layout",           "paged",
      "--kv-block-tokens",     "16",
      "--batch-size",          "2",
      "-t",                    "3",
      "-i",                    "4",
      "-r",                    "5",
      "--seed",                "18446744073709551615",
      "-o",                    "./-",
  };
  LlmMemoryConfig config;

  ASSERT_EQ(parse_llm_arguments(arguments, config), EXIT_SUCCESS);
  EXPECT_EQ(config.backend, LlmMemoryBackend::Cpu);
  EXPECT_EQ(config.weight_size_mb, 4096u);
  EXPECT_EQ(config.layer_count, 32u);
  EXPECT_EQ(config.query_head_count, 32u);
  EXPECT_EQ(config.kv_head_count, 8u);
  EXPECT_EQ(config.head_dimension, 128u);
  EXPECT_EQ(config.kv_element_bytes, 4u);
  EXPECT_EQ(config.visible_context_tokens, 8192u);
  EXPECT_EQ(config.kv_layout, LlmKvLayout::Paged);
  EXPECT_EQ(config.kv_block_tokens, 16u);
  EXPECT_EQ(config.batch_size, 2u);
  EXPECT_EQ(config.requested_workers, 3u);
  EXPECT_EQ(config.available_workers, 8u);
  EXPECT_EQ(config.iterations, 4u);
  EXPECT_EQ(config.loop_count, 5u);
  EXPECT_EQ(config.seed, std::numeric_limits<uint64_t>::max());
  EXPECT_TRUE(config.user_specified_backend);
  EXPECT_TRUE(config.user_specified_iterations);
  EXPECT_TRUE(config.user_specified_seed);
  EXPECT_TRUE(config.user_specified_workers);
  EXPECT_TRUE(config.user_specified_phase);
  EXPECT_TRUE(config.user_specified_context_tokens);
  EXPECT_FALSE(config.user_specified_prompt_tokens);
  EXPECT_FALSE(config.user_specified_attention_query_tile_tokens);
  EXPECT_TRUE(config.user_specified_kv_layout);
  EXPECT_TRUE(config.user_specified_kv_block_tokens);
  EXPECT_EQ(config.output_file, "./-");
  EXPECT_EQ(config.argv, arguments);
}

TEST(LlmMemoryConfigTest, ParserActivatesContiguousCpuPrefillWithExactPhaseGeometry) {
  LlmParserHooksScope hooks;
  const std::vector<std::string> arguments = valid_prefill_arguments();
  LlmMemoryConfig config;

  ASSERT_EQ(parse_llm_arguments(arguments, config), EXIT_SUCCESS);
  EXPECT_EQ(config.backend, LlmMemoryBackend::Cpu);
  EXPECT_EQ(config.phase, LlmPhase::Prefill);
  EXPECT_EQ(config.kv_layout, LlmKvLayout::Contiguous);
  EXPECT_EQ(config.visible_context_tokens, 0u);
  EXPECT_EQ(config.prompt_tokens, 5u);
  EXPECT_EQ(config.attention_query_tile_tokens, 2u);
  EXPECT_TRUE(config.user_specified_phase);
  EXPECT_FALSE(config.user_specified_context_tokens);
  EXPECT_TRUE(config.user_specified_prompt_tokens);
  EXPECT_TRUE(config.user_specified_attention_query_tile_tokens);
  EXPECT_EQ(config.argv, arguments);
}

TEST(LlmMemoryConfigTest, ParserUsesStableOrderIndependentPrefillAndPagedLayoutRules) {
  LlmParserHooksScope hooks;
  struct InvalidCase {
    std::vector<std::string> arguments;
    std::string reason_code;
  };
  std::vector<std::string> common = valid_prefill_arguments();
  const auto erase_option = [](std::vector<std::string> arguments, const std::string& option) {
    const auto position = std::find(arguments.begin(), arguments.end(), option);
    EXPECT_NE(position, arguments.end());
    if (position != arguments.end()) {
      arguments.erase(position, position + 2);
    }
    return arguments;
  };

  std::vector<InvalidCase> cases;
  const auto with_valid_paged_layout = [](std::vector<std::string> arguments) {
    arguments.insert(arguments.begin() + 2, {"--kv-layout", "paged", "--kv-block-tokens", "4"});
    return arguments;
  };
  const auto add_phase_error_cases = [&cases, &with_valid_paged_layout](const std::vector<std::string>& arguments,
                                                                        const std::string& reason_code) {
    cases.push_back({arguments, reason_code});
    cases.push_back({with_valid_paged_layout(arguments), reason_code});
  };
  add_phase_error_cases(erase_option(common, "--prompt-tokens"), LlmMemoryConfigReason::PROMPT_TOKENS_REQUIRED);
  add_phase_error_cases(erase_option(common, "--attention-query-tile-tokens"),
                        LlmMemoryConfigReason::ATTENTION_QUERY_TILE_TOKENS_REQUIRED);

  std::vector<std::string> prompt_zero = common;
  replace_option_value(prompt_zero, "--prompt-tokens", "0");
  add_phase_error_cases(prompt_zero, LlmMemoryConfigReason::PROMPT_TOKENS_MUST_BE_POSITIVE);
  std::vector<std::string> query_zero = common;
  replace_option_value(query_zero, "--attention-query-tile-tokens", "0");
  add_phase_error_cases(query_zero, LlmMemoryConfigReason::ATTENTION_QUERY_TILE_TOKENS_MUST_BE_POSITIVE);
  std::vector<std::string> query_too_large = common;
  replace_option_value(query_too_large, "--attention-query-tile-tokens", "6");
  add_phase_error_cases(query_too_large, LlmMemoryConfigReason::ATTENTION_QUERY_TILE_TOKENS_EXCEEDS_PROMPT);

  for (const InvalidCase& test_case : cases) {
    SCOPED_TRACE(::testing::PrintToString(test_case.arguments));
    LlmMemoryConfig config;
    const CapturedLlmParse parsed = parse_llm_arguments_capturing(test_case.arguments, config);
    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    EXPECT_TRUE(parsed.stdout_output.empty());
    EXPECT_EQ(first_output_line(parsed.stderr_output),
              Messages::error_prefix() + Messages::error_llm_memory_config_invalid(test_case.reason_code));
  }

  std::vector<std::string> invalid_paged = common;
  invalid_paged.insert(invalid_paged.end(), {"--kv-layout", "paged", "--kv-block-tokens", "3"});
  LlmMemoryConfig config;
  const CapturedLlmParse invalid_paged_result = parse_llm_arguments_capturing(invalid_paged, config);
  EXPECT_EQ(first_output_line(invalid_paged_result.stderr_output),
            Messages::error_prefix() +
                Messages::error_llm_memory_config_invalid(LlmMemoryConfigReason::KV_BLOCK_TOKENS_NOT_POWER_OF_TWO));

  for (const std::vector<std::string>& paged_suffix :
       {std::vector<std::string>{"--kv-layout", "paged", "--kv-block-tokens", "4"},
        std::vector<std::string>{"--kv-block-tokens", "4", "--kv-layout", "paged"}}) {
    std::vector<std::string> paged = common;
    paged.insert(paged.end(), paged_suffix.begin(), paged_suffix.end());
    LlmMemoryConfig paged_config;
    const CapturedLlmParse parsed = parse_llm_arguments_capturing(paged, paged_config);
    SCOPED_TRACE(::testing::PrintToString(paged));
    EXPECT_EQ(parsed.result, EXIT_SUCCESS);
    EXPECT_TRUE(parsed.stdout_output.empty());
    EXPECT_TRUE(parsed.stderr_output.empty());
    EXPECT_EQ(paged_config.phase, LlmPhase::Prefill);
    EXPECT_EQ(paged_config.kv_layout, LlmKvLayout::Paged);
    EXPECT_EQ(paged_config.prompt_tokens, 5u);
    EXPECT_EQ(paged_config.attention_query_tile_tokens, 2u);
    EXPECT_EQ(paged_config.kv_block_tokens, 4u);
    EXPECT_TRUE(paged_config.user_specified_phase);
    EXPECT_TRUE(paged_config.user_specified_kv_layout);
    EXPECT_TRUE(paged_config.user_specified_kv_block_tokens);
    EXPECT_EQ(paged_config.argv, paged);
  }
}

TEST(LlmMemoryConfigTest, ParserRejectsCrossPhaseTokenOptionsAndUnknownPhase) {
  LlmParserHooksScope hooks;
  struct InvalidCase {
    std::vector<std::string> arguments;
    std::string reason_code;
  };
  std::vector<std::string> decode_prompt = valid_llm_arguments();
  decode_prompt.insert(decode_prompt.end(), {"--prompt-tokens", "5"});
  std::vector<std::string> decode_query = valid_llm_arguments();
  decode_query.insert(decode_query.end(), {"--attention-query-tile-tokens", "2"});
  std::vector<std::string> prefill_context = valid_prefill_arguments();
  prefill_context.insert(prefill_context.end(), {"--context-tokens", "5"});
  const std::vector<InvalidCase> cases = {
      {decode_prompt, LlmMemoryConfigReason::PROMPT_TOKENS_NOT_APPLICABLE},
      {decode_query, LlmMemoryConfigReason::ATTENTION_QUERY_TILE_TOKENS_NOT_APPLICABLE},
      {prefill_context, LlmMemoryConfigReason::CONTEXT_TOKENS_NOT_APPLICABLE},
  };
  for (const InvalidCase& test_case : cases) {
    LlmMemoryConfig config;
    const CapturedLlmParse parsed = parse_llm_arguments_capturing(test_case.arguments, config);
    EXPECT_EQ(first_output_line(parsed.stderr_output),
              Messages::error_prefix() + Messages::error_llm_memory_config_invalid(test_case.reason_code));
  }

  std::vector<std::string> invalid_phase = valid_llm_arguments();
  invalid_phase.insert(invalid_phase.end(), {"--phase", "training"});
  LlmMemoryConfig config;
  const CapturedLlmParse parsed = parse_llm_arguments_capturing(invalid_phase, config);
  EXPECT_EQ(first_output_line(parsed.stderr_output),
            Messages::error_prefix() +
                Messages::error_invalid_value("--phase", "training", Messages::llm_memory_reason_phase()));
}

TEST(LlmMemoryConfigTest,
     ParserResolvesContiguousAndPagedLayoutSourcesExactly) {
  LlmParserHooksScope hooks;

  std::vector<std::string> arguments = valid_llm_arguments();
  arguments.insert(arguments.end(), {"--kv-layout", "contiguous"});
  LlmMemoryConfig config;
  ASSERT_EQ(parse_llm_arguments(arguments, config), EXIT_SUCCESS);
  EXPECT_EQ(config.kv_layout, LlmKvLayout::Contiguous);
  EXPECT_EQ(config.kv_block_tokens, 0u);
  EXPECT_TRUE(config.user_specified_kv_layout);
  EXPECT_FALSE(config.user_specified_kv_block_tokens);

  arguments = valid_llm_arguments();
  arguments.insert(arguments.end(), {"--kv-block-tokens", "16",
                                     "--kv-layout", "paged"});
  ASSERT_EQ(parse_llm_arguments(arguments, config), EXIT_SUCCESS);
  EXPECT_EQ(config.kv_layout, LlmKvLayout::Paged);
  EXPECT_EQ(config.kv_block_tokens, 16u);
  EXPECT_TRUE(config.user_specified_kv_layout);
  EXPECT_TRUE(config.user_specified_kv_block_tokens);
}

TEST(LlmMemoryConfigTest,
     ParserUsesStableOrderIndependentPagedLayoutReasons) {
  LlmParserHooksScope hooks;
  struct InvalidCase {
    std::vector<std::string> suffix;
    std::string reason_code;
  };
  const std::vector<InvalidCase> cases = {
      {{"--kv-layout", "contiguous", "--kv-block-tokens", "16"},
       LlmMemoryConfigReason::KV_BLOCK_TOKENS_NOT_APPLICABLE},
      {{"--kv-block-tokens", "16", "--kv-layout", "contiguous"},
       LlmMemoryConfigReason::KV_BLOCK_TOKENS_NOT_APPLICABLE},
      {{"--kv-layout", "paged", "--kv-block-tokens", "0"},
       LlmMemoryConfigReason::KV_BLOCK_TOKENS_ZERO},
      {{"--kv-block-tokens", "0", "--kv-layout", "paged"},
       LlmMemoryConfigReason::KV_BLOCK_TOKENS_ZERO},
      {{"--kv-layout", "paged", "--kv-block-tokens", "3"},
       LlmMemoryConfigReason::KV_BLOCK_TOKENS_NOT_POWER_OF_TWO},
      {{"--kv-block-tokens", "3", "--kv-layout", "paged"},
       LlmMemoryConfigReason::KV_BLOCK_TOKENS_NOT_POWER_OF_TWO},
      {{"--kv-layout", "paged", "--kv-block-tokens", "4294967296"},
       LlmMemoryConfigReason::KV_BLOCK_TOKENS_EXCEEDS_UINT32},
      {{"--kv-block-tokens", "4294967296", "--kv-layout", "paged"},
       LlmMemoryConfigReason::KV_BLOCK_TOKENS_EXCEEDS_UINT32},
  };

  for (const InvalidCase& test_case : cases) {
    SCOPED_TRACE(::testing::PrintToString(test_case.suffix));
    std::vector<std::string> arguments = valid_llm_arguments();
    arguments.insert(arguments.end(), test_case.suffix.begin(),
                     test_case.suffix.end());
    LlmMemoryConfig config;
    const CapturedLlmParse parsed =
        parse_llm_arguments_capturing(arguments, config);
    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    EXPECT_TRUE(parsed.stdout_output.empty());
    EXPECT_EQ(first_output_line(parsed.stderr_output),
              Messages::error_prefix() +
                  Messages::error_llm_memory_config_invalid(
                      test_case.reason_code));
  }
}

TEST(LlmMemoryConfigTest,
     ParserRequiresPagedBlockTokensAndRejectsUnknownLayout) {
  LlmParserHooksScope hooks;
  LlmMemoryConfig config;

  std::vector<std::string> arguments = valid_llm_arguments();
  arguments.insert(arguments.end(), {"--kv-layout", "paged"});
  CapturedLlmParse parsed =
      parse_llm_arguments_capturing(arguments, config);
  EXPECT_EQ(parsed.result, EXIT_FAILURE);
  EXPECT_TRUE(parsed.stdout_output.empty());
  EXPECT_EQ(first_output_line(parsed.stderr_output),
            Messages::error_prefix() +
                Messages::error_llm_memory_missing_required_option(
                    "--kv-block-tokens"));

  arguments = valid_llm_arguments();
  arguments.insert(arguments.end(), {"--kv-layout", "sparse"});
  parsed = parse_llm_arguments_capturing(arguments, config);
  EXPECT_EQ(parsed.result, EXIT_FAILURE);
  EXPECT_TRUE(parsed.stdout_output.empty());
  EXPECT_EQ(first_output_line(parsed.stderr_output),
            Messages::error_prefix() +
                Messages::error_invalid_value(
                    "--kv-layout", "sparse",
                    Messages::llm_memory_reason_kv_layout()));
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
           "--llm-memory-backend", "--weight-size-mb", "--layers",
           "--query-heads", "--kv-heads", "--head-dim",
           "--kv-element-bytes", "--phase",
           "--context-tokens", "--prompt-tokens",
           "--attention-query-tile-tokens", "--kv-layout",
           "--kv-block-tokens", "--batch-size", "--threads",
           "--iterations", "--count", "--seed", "--output"}) {
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
      {{"--llm-memory-backend", "cpu", "--llm-memory-backend", "metal"},
       "--llm-memory-backend"},
      {{"--weight-size-mb", "1"}, "--weight-size-mb"},
      {{"--layers", "2"}, "--layers"},
      {{"--query-heads", "4"}, "--query-heads"},
      {{"--kv-heads", "2"}, "--kv-heads"},
      {{"--head-dim", "8"}, "--head-dim"},
      {{"--kv-element-bytes", "2", "--kv-element-bytes", "2"},
       "--kv-element-bytes"},
      {{"--phase", "decode", "--phase", "prefill"}, "--phase"},
      {{"--context-tokens", "3"}, "--context-tokens"},
      {{"--prompt-tokens", "3", "--prompt-tokens", "4"}, "--prompt-tokens"},
      {{"--attention-query-tile-tokens", "2",
        "--attention-query-tile-tokens", "3"},
       "--attention-query-tile-tokens"},
      {{"--kv-layout", "contiguous", "--kv-layout", "paged"},
       "--kv-layout"},
      {{"--kv-block-tokens", "16", "--kv-block-tokens", "32"},
       "--kv-block-tokens"},
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
      "--kv-block-tokens", "--batch-size",      "--threads",
      "--iterations",     "--count"};

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
      {"memory_benchmark", "-M", "--llm-memory-backend", "metal",
       "--help"},
      config);

  EXPECT_EQ(parsed.result, EXIT_SUCCESS);
  EXPECT_TRUE(parsed.stderr_output.empty()) << parsed.stderr_output;
  EXPECT_NE(parsed.stdout_output.find(
                "Usage: memory_benchmark --llm-memory [options]"),
            std::string::npos);
  EXPECT_NE(parsed.stdout_output.find("memory traffic only"),
            std::string::npos);
  EXPECT_NE(parsed.stdout_output.find(
                "--llm-memory-backend <cpu|metal>"),
            std::string::npos);
  EXPECT_NE(parsed.stdout_output.find("--phase <decode|prefill>"),
            std::string::npos);
  EXPECT_NE(parsed.stdout_output.find("--prompt-tokens <count>"),
            std::string::npos);
  EXPECT_NE(
      parsed.stdout_output.find("--attention-query-tile-tokens <count>"),
      std::string::npos);
  EXPECT_EQ(parsed.stdout_output.find(Messages::config_header(SOFTVERSION)),
            std::string::npos);
  EXPECT_TRUE(config.help_printed);
  EXPECT_EQ(config.backend, LlmMemoryBackend::Metal);
  EXPECT_TRUE(config.user_specified_backend);
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

TEST(LlmMemoryConfigTest, StableVocabularyAndStatusTokens) {
  EXPECT_STREQ(llm_memory_backend_to_string(LlmMemoryBackend::Cpu), "cpu");
  EXPECT_STREQ(llm_memory_backend_to_string(LlmMemoryBackend::Metal),
               "metal");
  EXPECT_STREQ(llm_memory_backend_to_string(
                   static_cast<LlmMemoryBackend>(99)),
               "unknown");

  EXPECT_STREQ(llm_phase_to_string(LlmPhase::Decode), "decode");
  EXPECT_STREQ(llm_phase_to_string(LlmPhase::Prefill), "prefill");
  EXPECT_STREQ(llm_phase_to_string(static_cast<LlmPhase>(99)), "unknown");

  EXPECT_STREQ(llm_kv_layout_to_string(LlmKvLayout::Contiguous),
               "contiguous");
  EXPECT_STREQ(llm_kv_layout_to_string(LlmKvLayout::Paged), "paged");
  EXPECT_STREQ(llm_kv_layout_to_string(static_cast<LlmKvLayout>(99)),
               "unknown");

  EXPECT_EQ(llm_work_unit_kind_for_phase(LlmPhase::Decode),
            LlmWorkUnitKind::DecodeStep);
  EXPECT_EQ(llm_work_unit_kind_for_phase(LlmPhase::Prefill),
            LlmWorkUnitKind::PrefillOperation);
  EXPECT_STREQ(llm_work_unit_kind_to_string(LlmWorkUnitKind::DecodeStep),
               "decode_step");
  EXPECT_STREQ(
      llm_work_unit_kind_to_string(LlmWorkUnitKind::PrefillOperation),
      "prefill_operation");
  EXPECT_STREQ(llm_work_unit_kind_to_string(
                   static_cast<LlmWorkUnitKind>(99)),
               "unknown");

  EXPECT_EQ(llm_kv_write_kind_for(LlmPhase::Decode,
                                  LlmScenario::WeightsOnly),
            LlmKvWriteKind::None);
  EXPECT_EQ(llm_kv_write_kind_for(LlmPhase::Decode, LlmScenario::KvOnly),
            LlmKvWriteKind::CurrentTokenAppend);
  EXPECT_EQ(llm_kv_write_kind_for(LlmPhase::Prefill, LlmScenario::Mixed),
            LlmKvWriteKind::FullPromptPopulation);
  EXPECT_STREQ(llm_kv_write_kind_to_string(LlmKvWriteKind::None), "none");
  EXPECT_STREQ(
      llm_kv_write_kind_to_string(LlmKvWriteKind::CurrentTokenAppend),
      "current_token_append");
  EXPECT_STREQ(
      llm_kv_write_kind_to_string(LlmKvWriteKind::FullPromptPopulation),
      "full_prompt_population");
  EXPECT_STREQ(llm_kv_write_kind_to_string(
                   static_cast<LlmKvWriteKind>(99)),
               "unknown");

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
  EXPECT_STREQ(llm_run_status_to_string(LlmRunStatus::Unsupported),
               "unsupported");
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
  config.kv_layout = LlmKvLayout::Paged;
  expect_invalid(config, LlmMemoryConfigReason::KV_BLOCK_TOKENS_REQUIRED);
  config = valid_config();
  config.kv_layout = LlmKvLayout::Paged;
  config.user_specified_kv_block_tokens = true;
  expect_invalid(config, LlmMemoryConfigReason::KV_BLOCK_TOKENS_ZERO);
  config = valid_config();
  config.kv_layout = LlmKvLayout::Paged;
  config.kv_block_tokens = 3;
  config.user_specified_kv_block_tokens = true;
  expect_invalid(
      config, LlmMemoryConfigReason::KV_BLOCK_TOKENS_NOT_POWER_OF_TWO);
  config = valid_config();
  config.kv_layout = LlmKvLayout::Paged;
  config.kv_block_tokens =
      static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 1;
  config.user_specified_kv_block_tokens = true;
  expect_invalid(config,
                 LlmMemoryConfigReason::KV_BLOCK_TOKENS_EXCEEDS_UINT32);
  config = valid_config();
  config.kv_block_tokens = 16;
  config.user_specified_kv_block_tokens = true;
  expect_invalid(config,
                 LlmMemoryConfigReason::KV_BLOCK_TOKENS_NOT_APPLICABLE);
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
  config = valid_config();
  config.kv_layout = LlmKvLayout::Paged;
  config.kv_block_tokens = 16;
  config.user_specified_kv_layout = true;
  config.user_specified_kv_block_tokens = true;
  EXPECT_TRUE(validate_llm_memory_config(config).valid);
  config.kv_block_tokens = static_cast<size_t>(1) << 31;
  EXPECT_TRUE(validate_llm_memory_config(config).valid);
}

TEST(LlmMemoryConfigTest,
     ValidatesPhaseSpecificTokenInputsThroughPureConfigSeam) {
  LlmMemoryConfig config = valid_config();
  ASSERT_TRUE(validate_llm_memory_config(config).valid);

  config.phase = static_cast<LlmPhase>(99);
  expect_invalid(config, LlmMemoryConfigReason::INVALID_PHASE);

  config = valid_prefill_config();
  LlmMemoryConfigValidation validation = validate_llm_memory_config(config);
  ASSERT_TRUE(validation.valid) << validation.reason_code;
  EXPECT_EQ(validation.reason_code, LlmMemoryConfigReason::VALID);
  EXPECT_EQ(validation.active_weight_bytes,
            4ULL * 1024ULL * 1024ULL * 1024ULL);

  config.prompt_tokens = 1;
  config.attention_query_tile_tokens = 1;
  EXPECT_TRUE(validate_llm_memory_config(config).valid);

  config = valid_prefill_config();
  config.attention_query_tile_tokens = config.prompt_tokens;
  EXPECT_TRUE(validate_llm_memory_config(config).valid);

  config = valid_prefill_config();
  config.user_specified_prompt_tokens = false;
  config.user_specified_attention_query_tile_tokens = false;
  EXPECT_TRUE(validate_llm_memory_config(config).valid);

  config = valid_config();
  config.visible_context_tokens = 0;
  expect_invalid(config, LlmMemoryConfigReason::CONTEXT_TOKENS_REQUIRED);

  config = valid_config();
  config.user_specified_prompt_tokens = true;
  expect_invalid(config, LlmMemoryConfigReason::PROMPT_TOKENS_NOT_APPLICABLE);
  config = valid_config();
  config.prompt_tokens = 1;
  expect_invalid(config, LlmMemoryConfigReason::PROMPT_TOKENS_NOT_APPLICABLE);
  config = valid_config();
  config.user_specified_attention_query_tile_tokens = true;
  expect_invalid(
      config,
      LlmMemoryConfigReason::ATTENTION_QUERY_TILE_TOKENS_NOT_APPLICABLE);
  config = valid_config();
  config.attention_query_tile_tokens = 1;
  expect_invalid(
      config,
      LlmMemoryConfigReason::ATTENTION_QUERY_TILE_TOKENS_NOT_APPLICABLE);

  config = valid_prefill_config();
  config.user_specified_context_tokens = true;
  expect_invalid(config,
                 LlmMemoryConfigReason::CONTEXT_TOKENS_NOT_APPLICABLE);
  config = valid_prefill_config();
  config.visible_context_tokens = 1;
  expect_invalid(config,
                 LlmMemoryConfigReason::CONTEXT_TOKENS_NOT_APPLICABLE);

  config = valid_prefill_config();
  config.prompt_tokens = 0;
  config.user_specified_prompt_tokens = false;
  expect_invalid(config, LlmMemoryConfigReason::PROMPT_TOKENS_REQUIRED);
  config = valid_prefill_config();
  config.prompt_tokens = 0;
  expect_invalid(config,
                 LlmMemoryConfigReason::PROMPT_TOKENS_MUST_BE_POSITIVE);

  config = valid_prefill_config();
  config.attention_query_tile_tokens = 0;
  config.user_specified_attention_query_tile_tokens = false;
  expect_invalid(
      config,
      LlmMemoryConfigReason::ATTENTION_QUERY_TILE_TOKENS_REQUIRED);
  config = valid_prefill_config();
  config.attention_query_tile_tokens = 0;
  expect_invalid(
      config,
      LlmMemoryConfigReason::ATTENTION_QUERY_TILE_TOKENS_MUST_BE_POSITIVE);
  config = valid_prefill_config();
  config.attention_query_tile_tokens = config.prompt_tokens + 1;
  expect_invalid(
      config,
      LlmMemoryConfigReason::ATTENTION_QUERY_TILE_TOKENS_EXCEEDS_PROMPT);
}

TEST(LlmMemoryConfigTest,
     PureValidationKeepsMetalWorkersNonApplicableWithoutApplyingActivationMatrix) {
  LlmMemoryConfig config = valid_config();
  config.backend = LlmMemoryBackend::Metal;
  config.requested_workers = 0;
  config.available_workers = 0;
  LlmMemoryConfigValidation validation =
      validate_llm_memory_config(config);
  ASSERT_TRUE(validation.valid) << validation.reason_code;

  config.user_specified_workers = true;
  config.requested_workers = 1;
  expect_invalid(config, LlmMemoryConfigReason::THREADS_NOT_APPLICABLE);

  config = valid_config();
  config.backend = LlmMemoryBackend::Metal;
  config.requested_workers = 0;
  config.available_workers = 1;
  expect_invalid(config, LlmMemoryConfigReason::THREADS_NOT_APPLICABLE);

  config = valid_prefill_config();
  config.backend = LlmMemoryBackend::Metal;
  config.requested_workers = 0;
  config.available_workers = 0;
  validation = validate_llm_memory_config(config);
  EXPECT_TRUE(validation.valid) << validation.reason_code;

  config = valid_config();
  config.backend = static_cast<LlmMemoryBackend>(99);
  expect_invalid(config, LlmMemoryConfigReason::INVALID_BACKEND);
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

TEST(LlmMemoryConfigTest,
     TimerSetupFailureKeepsStdoutEmptyAndPreservesLegacyDiagnosticBoundary) {
  std::vector<std::string> arguments = valid_llm_arguments();
  arguments.insert(arguments.end(), {"--iterations", "1", "--count", "1",
                                     "--output", "-"});
  std::vector<char*> argv;
  argv.reserve(arguments.size());
  for (std::string& argument : arguments) {
    argv.push_back(argument.data());
  }
  const ScopedFailingTimerSystemCalls timer_system_calls;

  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  const int status =
      run_llm_memory_mode(static_cast<int>(argv.size()), argv.data());
  const std::string stderr_output = testing::internal::GetCapturedStderr();
  const std::string stdout_output = testing::internal::GetCapturedStdout();

  EXPECT_EQ(status, EXIT_FAILURE);
  EXPECT_TRUE(stdout_output.empty());
  EXPECT_NE(stderr_output.find(Messages::error_mach_timebase_info_failed(
                mach_error_string(KERN_FAILURE))),
            std::string::npos);
  EXPECT_EQ(stderr_output.find(Messages::error_llm_memory_run_failed(
                LlmBackendReason::TIMER_UNAVAILABLE)),
            std::string::npos);
}

TEST(LlmMemoryConfigIntegrationTest,
     PostRunExceptionWritesExactlyOneFailedStdoutDocumentIntegration) {
  std::vector<std::string> arguments = valid_llm_arguments();
  arguments.insert(arguments.end(),
                   {"--threads", "1", "--iterations", "1", "--count", "1",
                    "--seed", "1", "--output", "-"});
  const LlmParserHooksScope parser_hooks(1, 1);
  size_t hook_calls = 0;
  const LlmCommandHooksScope command_hooks([&]() {
    ++hook_calls;
    throw std::runtime_error("injected post-run command exception");
  });

  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  const int status = run_llm_command(std::move(arguments));
  const std::string stderr_output = testing::internal::GetCapturedStderr();
  const std::string stdout_output = testing::internal::GetCapturedStdout();

  EXPECT_EQ(status, EXIT_FAILURE);
  EXPECT_EQ(hook_calls, 1u);
  ASSERT_FALSE(stdout_output.empty());
  EXPECT_EQ(stdout_output.back(), '\n');
  nlohmann::ordered_json document;
  ASSERT_NO_THROW(document = nlohmann::ordered_json::parse(stdout_output));
  EXPECT_EQ(document.at("status"), "failed");
  EXPECT_EQ(document.at("reason_code"), LlmRunnerReason::RUNNER_EXCEPTION);
  EXPECT_FALSE(document.at("results_complete").get<bool>());
  EXPECT_FALSE(document.at("conclusions_valid").get<bool>());
  EXPECT_NE(stderr_output.find("injected post-run command exception"),
            std::string::npos);
}

TEST(LlmMemoryConfigIntegrationTest,
     PostRunExceptionAtomicallyReplacesTerminalFileDocumentIntegration) {
  const LlmCommandOutputFileScope output;
  std::vector<std::string> arguments = valid_llm_arguments();
  arguments.insert(
      arguments.end(),
      {"--threads", "1", "--iterations", "1", "--count", "1", "--seed",
       "1", "--output", output.path().string()});
  const LlmParserHooksScope parser_hooks(1, 1);
  size_t hook_calls = 0;
  bool observed_complete_terminal_file = false;
  const LlmCommandHooksScope command_hooks([&]() {
    ++hook_calls;
    const nlohmann::ordered_json before_exception =
        nlohmann::ordered_json::parse(
            read_llm_command_output_file(output.path()));
    observed_complete_terminal_file =
        before_exception.at("status") == "complete";
    throw std::runtime_error("injected post-run command exception");
  });

  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  const int status = run_llm_command(std::move(arguments));
  const std::string stderr_output = testing::internal::GetCapturedStderr();
  const std::string stdout_output = testing::internal::GetCapturedStdout();

  EXPECT_EQ(status, EXIT_FAILURE);
  EXPECT_EQ(hook_calls, 1u);
  EXPECT_TRUE(observed_complete_terminal_file);
  ASSERT_TRUE(std::filesystem::exists(output.path()));
  EXPECT_FALSE(std::filesystem::exists(output.path().string() + ".tmp"));
  const nlohmann::ordered_json document = nlohmann::ordered_json::parse(
      read_llm_command_output_file(output.path()));
  EXPECT_EQ(document.at("status"), "failed");
  EXPECT_EQ(document.at("reason_code"), LlmRunnerReason::RUNNER_EXCEPTION);
  EXPECT_FALSE(document.at("results_complete").get<bool>());
  EXPECT_FALSE(document.at("conclusions_valid").get<bool>());
  EXPECT_EQ(stdout_output.find("Results saved"), std::string::npos);
  EXPECT_NE(stderr_output.find("injected post-run command exception"),
            std::string::npos);
}

TEST(LlmMemoryConfigTest,
     RejectsJsonIntegerFieldsOutsideTheExactIeee754Range) {
  static_assert(Constants::LLM_JSON_MAX_SAFE_INTEGER <
                std::numeric_limits<size_t>::max());
  constexpr size_t kUnsafeInteger =
      Constants::LLM_JSON_MAX_SAFE_INTEGER + 1;
  constexpr std::array<size_t LlmMemoryConfig::*, 9> kFields = {
      &LlmMemoryConfig::weight_size_mb,
      &LlmMemoryConfig::layer_count,
      &LlmMemoryConfig::query_head_count,
      &LlmMemoryConfig::kv_head_count,
      &LlmMemoryConfig::head_dimension,
      &LlmMemoryConfig::visible_context_tokens,
      &LlmMemoryConfig::batch_size,
      &LlmMemoryConfig::requested_workers,
      &LlmMemoryConfig::available_workers,
  };

  for (size_t LlmMemoryConfig::*field : kFields) {
    LlmMemoryConfig config = valid_config();
    config.*field = kUnsafeInteger;
    expect_invalid(config,
                   LlmMemoryConfigReason::JSON_INTEGER_OUT_OF_RANGE);
  }

  LlmMemoryConfig config = valid_config();
  config.user_specified_iterations = true;
  config.iterations = kUnsafeInteger;
  expect_invalid(config, LlmMemoryConfigReason::JSON_INTEGER_OUT_OF_RANGE);

  config = valid_config();
  config.loop_count =
      Constants::LLM_JSON_MAX_SAFE_INTEGER / kLlmScenarioCount;
  EXPECT_TRUE(validate_llm_memory_config(config).valid);
  ++config.loop_count;
  expect_invalid(config, LlmMemoryConfigReason::JSON_INTEGER_OUT_OF_RANGE);
}

TEST(LlmMemoryConfigTest, ResultFoundationKeepsUnavailableValuesAbsent) {
  const LlmMeasurementState measurement;
  EXPECT_EQ(measurement.status, LlmMeasurementStatus::NotRun);
  EXPECT_EQ(measurement.reason_code, "not-run");
  EXPECT_EQ(measurement.planned_work_units, 0u);
  EXPECT_EQ(measurement.completed_work_units, 0u);
  EXPECT_FALSE(measurement.elapsed_seconds.has_value());
  EXPECT_FALSE(measurement.effective_model_payload_gb_s.has_value());
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
  EXPECT_EQ(result.counters.completed_effective_model_payload_bytes, 0u);
}

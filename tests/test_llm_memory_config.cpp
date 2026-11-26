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
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

#include "core/config/version.h"
#include "core/timing/timer.h"
#include "llm_memory/llm_backend.h"
#include "llm_memory/llm_memory.h"
#include "llm_memory/llm_runner.h"
#include "llm_memory/llm_sweep_runner.h"
#include "output/console/messages/messages_api.h"
#include "third_party/nlohmann/json.hpp"

namespace {

uint64_t unused_timer_ticks() { return 0; }
kern_return_t failing_timebase_info(mach_timebase_info_t) { return KERN_FAILURE; }

class ScopedFailingTimerSystemCalls {
 public:
  ScopedFailingTimerSystemCalls() { set_timer_system_calls_for_testing({unused_timer_ticks, failing_timebase_info}); }
  ScopedFailingTimerSystemCalls(const ScopedFailingTimerSystemCalls&) = delete;
  ScopedFailingTimerSystemCalls& operator=(const ScopedFailingTimerSystemCalls&) = delete;
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

void expect_invalid(const LlmMemoryConfig& config, const std::string& expected_reason) {
  const LlmMemoryConfigValidation validation = validate_llm_memory_config(config);
  EXPECT_FALSE(validation.valid);
  EXPECT_EQ(validation.reason_code, expected_reason);
  EXPECT_EQ(validation.active_weight_bytes, 0u);
}

std::vector<std::string> valid_llm_arguments() {
  return {"memory_benchmark", "--llm-memory", "--weight-size-mb", "1", "--layers", "2", "--query-heads", "4",
          "--kv-heads",       "2",            "--head-dim",       "8", "--context-tokens", "3"};
}

std::vector<std::string> valid_prefill_arguments() {
  return {"memory_benchmark", "--llm-memory", "--weight-size-mb", "1", "--layers", "2", "--query-heads", "4",
          "--kv-heads",       "2",            "--head-dim",       "8", "--phase",  "prefill",
          "--prompt-tokens",  "5",            "--attention-query-tile-tokens", "2"};
}

std::vector<std::string> valid_metal_arguments() {
  std::vector<std::string> arguments = valid_llm_arguments();
  arguments.insert(arguments.begin() + 2, {"--llm-memory-backend", "metal"});
  return arguments;
}

int parse_llm_arguments(std::vector<std::string> arguments, LlmMemoryConfig& config) {
  std::vector<char*> argv;
  argv.reserve(arguments.size());
  for (std::string& argument : arguments) argv.push_back(argument.data());
  return parse_llm_memory_arguments(static_cast<int>(argv.size()), argv.data(), config);
}

struct CapturedLlmParse {
  int result = EXIT_FAILURE;
  std::string stdout_output;
  std::string stderr_output;
};

CapturedLlmParse parse_llm_arguments_capturing(std::vector<std::string> arguments, LlmMemoryConfig& config) {
  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  const int status = parse_llm_arguments(std::move(arguments), config);
  const std::string stderr_output = testing::internal::GetCapturedStderr();
  const std::string stdout_output = testing::internal::GetCapturedStdout();
  return {status, stdout_output, stderr_output};
}

int parse_llm_arguments_silently(std::vector<std::string> arguments, LlmMemoryConfig& config) {
  return parse_llm_arguments_capturing(std::move(arguments), config).result;
}

std::string first_output_line(const std::string& output) { return output.substr(0, output.find('\n')); }

class LlmParserHooksScope {
 public:
  explicit LlmParserHooksScope(size_t detected_workers = 8, uint64_t generated_seed = 0x123456789abcdef0ULL) {
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
  explicit LlmCommandHooksScope(LlmMemoryCommandTestHooks hooks) : hooks_(std::move(hooks)) {
    set_llm_memory_command_test_hooks(&hooks_);
  }
  ~LlmCommandHooksScope() { set_llm_memory_command_test_hooks(nullptr); }
  LlmCommandHooksScope(const LlmCommandHooksScope&) = delete;
  LlmCommandHooksScope& operator=(const LlmCommandHooksScope&) = delete;

 private:
  LlmMemoryCommandTestHooks hooks_;
};

class LlmBackendFactoryHookScope {
 public:
  explicit LlmBackendFactoryHookScope(LlmBackendFactoryTestHook hook) : hook_(std::move(hook)) {
    set_llm_backend_factory_test_hook(&hook_);
  }
  ~LlmBackendFactoryHookScope() { set_llm_backend_factory_test_hook(nullptr); }
  LlmBackendFactoryHookScope(const LlmBackendFactoryHookScope&) = delete;
  LlmBackendFactoryHookScope& operator=(const LlmBackendFactoryHookScope&) = delete;

 private:
  LlmBackendFactoryTestHook hook_;
};

class PreparationInterruptedFakeBackend final : public LlmBackend {
 public:
  PreparationInterruptedFakeBackend() { evidence_.backend = LlmMemoryBackend::Cpu; }
  LlmMemoryBackend kind() const noexcept override { return LlmMemoryBackend::Cpu; }

  LlmBackendAuxiliaryEstimate calculate_auxiliary_estimate(
      const LlmMemoryWorkPlan& model_plan) const noexcept override {
    const LlmExecutorAuxiliaryEstimate cpu = calculate_llm_executor_auxiliary_estimate(model_plan);
    LlmBackendAuxiliaryEstimate estimate;
    estimate.valid = cpu.valid;
    estimate.reason_code = cpu.reason_code;
    estimate.checksum_auxiliary_bytes = cpu.checksum_auxiliary_bytes;
    estimate.orchestration_auxiliary_bytes = cpu.orchestration_auxiliary_bytes;
    estimate.total_auxiliary_bytes = cpu.checksum_auxiliary_bytes + cpu.orchestration_auxiliary_bytes;
    estimate.backend_evidence = cpu;
    return estimate;
  }

  LlmBackendLifecycleResult initialize(const LlmMemoryConfig&) noexcept override {
    return evidence_.initialization = {LlmBackendStatus::Ready, LlmBackendReason::VALID};
  }
  LlmBackendLifecycleResult resolve_execution_plan(const LlmMemoryWorkPlan&) noexcept override {
    return evidence_.plan_resolution = {LlmBackendStatus::Ready, LlmBackendReason::VALID};
  }
  LlmBackendLifecycleResult prepare_resources(const LlmMemoryWorkPlan&) override {
    return evidence_.preparation = {LlmBackendStatus::Failed, LlmBackendReason::PREPARATION_INTERRUPTED};
  }
  LlmTaskExecutionResult execute_task(const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&,
                                      const LlmRunnerTaskContext&) override {
    return LlmTaskExecutionResult{};
  }
  const LlmBackendEvidence& evidence() const noexcept override { return evidence_; }
  LlmBackendLifecycleResult release_resources() noexcept override {
    return evidence_.release = {LlmBackendStatus::Ready, LlmBackendReason::VALID};
  }

 private:
  LlmBackendEvidence evidence_;
};

class LlmCommandOutputFileScope {
 public:
  LlmCommandOutputFileScope()
      : path_(std::filesystem::path("/tmp") /
              ("membenchmark_llm_post_run_exception_" + std::to_string(::getpid()) + ".json")) {
    cleanup();
  }
  ~LlmCommandOutputFileScope() { cleanup(); }
  LlmCommandOutputFileScope(const LlmCommandOutputFileScope&) = delete;
  LlmCommandOutputFileScope& operator=(const LlmCommandOutputFileScope&) = delete;
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
  for (std::string& argument : arguments) argv.push_back(argument.data());
  return run_llm_memory_mode(static_cast<int>(argv.size()), argv.data());
}

std::string read_llm_command_output_file(const std::filesystem::path& path) {
  std::ifstream file(path);
  std::ostringstream contents;
  contents << file.rdbuf();
  return contents.str();
}

void replace_option_value(std::vector<std::string>& arguments, const std::string& option, const std::string& value) {
  const auto position = std::find(arguments.begin(), arguments.end(), option);
  ASSERT_NE(position, arguments.end()) << option;
  ASSERT_NE(position + 1, arguments.end()) << option;
  *(position + 1) = value;
}

}  // namespace

TEST(LlmMemoryConfigTest, ParserResolvesDefaultsExplicitAliasesMetalAndRawOutput) {
  {
    SCOPED_TRACE("cpu-defaults");
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

  {
    SCOPED_TRACE("metal-decode-contiguous");
    LlmParserHooksScope hooks(0, 9);
    const std::vector<std::string> arguments = valid_metal_arguments();
    LlmMemoryConfig config;

    const CapturedLlmParse parsed = parse_llm_arguments_capturing(arguments, config);
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

  {
    SCOPED_TRACE("explicit-fields-and-aliases");
    LlmParserHooksScope hooks;
    const std::vector<std::string> arguments = {
        "memory_benchmark", "-M", "--llm-memory-backend", "cpu", "--weight-size-mb", "4096", "--layers", "32",
        "--query-heads", "32", "--kv-heads", "8", "--head-dim", "128", "--kv-element-bytes", "4", "--phase", "decode",
        "--context-tokens", "8192", "--kv-layout", "paged", "--kv-block-tokens", "16", "--batch-size", "2", "-t", "12",
        "-i", "4", "-r", "5", "--seed", "18446744073709551615", "-o", "./-",
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
    EXPECT_EQ(config.requested_workers, 12u);
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

  {
    SCOPED_TRACE("raw-output-targets");
    LlmParserHooksScope hooks;
    for (const std::string& output_option : {"-o", "--output"}) {
      for (const std::string& target :
           {"", "-", "./-", "-M", "--llm-memory", "--gpu-bandwidth", "-o", "--output", "--help"}) {
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

  {
    SCOPED_TRACE("inline-equals-options-metal-and-empty-output");
    LlmParserHooksScope hooks(8, 9);
    const std::vector<std::string> metal_arguments = {
        "memory_benchmark", "--llm-memory", "--llm-memory-backend=metal", "--weight-size-mb=1", "--layers=2",
        "--query-heads=4", "--kv-heads=2", "--head-dim=8", "--kv-element-bytes=4", "--phase=decode",
        "--context-tokens=3", "--kv-layout=paged", "--kv-block-tokens=16", "--batch-size=2", "--iterations=4",
        "--count=5", "--seed=42", "--output=-",
    };
    LlmMemoryConfig metal_config;
    const CapturedLlmParse parsed_metal = parse_llm_arguments_capturing(metal_arguments, metal_config);
    ASSERT_EQ(parsed_metal.result, EXIT_SUCCESS) << parsed_metal.stderr_output;
    EXPECT_TRUE(parsed_metal.stdout_output.empty());
    EXPECT_TRUE(parsed_metal.stderr_output.empty());
    EXPECT_EQ(metal_config.backend, LlmMemoryBackend::Metal);
    EXPECT_EQ(metal_config.weight_size_mb, 1u);
    EXPECT_EQ(metal_config.layer_count, 2u);
    EXPECT_EQ(metal_config.query_head_count, 4u);
    EXPECT_EQ(metal_config.kv_head_count, 2u);
    EXPECT_EQ(metal_config.head_dimension, 8u);
    EXPECT_EQ(metal_config.kv_element_bytes, 4u);
    EXPECT_EQ(metal_config.phase, LlmPhase::Decode);
    EXPECT_EQ(metal_config.visible_context_tokens, 3u);
    EXPECT_EQ(metal_config.kv_layout, LlmKvLayout::Paged);
    EXPECT_EQ(metal_config.kv_block_tokens, 16u);
    EXPECT_EQ(metal_config.batch_size, 2u);
    EXPECT_EQ(metal_config.iterations, 4u);
    EXPECT_EQ(metal_config.loop_count, 5u);
    EXPECT_EQ(metal_config.seed, 42u);
    EXPECT_EQ(metal_config.output_file, "-");

    const std::vector<std::string> prefill_arguments = {
        "memory_benchmark", "--llm-memory", "--weight-size-mb=1", "--layers=2", "--query-heads=4", "--kv-heads=2",
        "--head-dim=8", "--phase=prefill", "--prompt-tokens=8", "--attention-query-tile-tokens=4", "--threads=4",
        "--output=",
    };
    LlmMemoryConfig prefill_config;
    prefill_config.output_file = "stale.json";
    const CapturedLlmParse parsed_prefill = parse_llm_arguments_capturing(prefill_arguments, prefill_config);
    ASSERT_EQ(parsed_prefill.result, EXIT_SUCCESS) << parsed_prefill.stderr_output;
    EXPECT_EQ(prefill_config.phase, LlmPhase::Prefill);
    EXPECT_EQ(prefill_config.prompt_tokens, 8u);
    EXPECT_EQ(prefill_config.attention_query_tile_tokens, 4u);
    EXPECT_EQ(prefill_config.requested_workers, 4u);
    EXPECT_TRUE(prefill_config.output_file.empty());

    const std::vector<std::string> sweep_arguments = {
        "memory_benchmark", "--llm-memory", "--weight-size-mb=1", "--layers=2", "--query-heads=4", "--kv-heads=2",
        "--head-dim=8", "--sweep=context-tokens=2,4", "--sweep-max-runs=4", "--output=llm_inline_sweep.json",
    };
    LlmMemoryConfig sweep_config;
    const CapturedLlmParse parsed_sweep = parse_llm_arguments_capturing(sweep_arguments, sweep_config);
    ASSERT_EQ(parsed_sweep.result, EXIT_SUCCESS) << parsed_sweep.stderr_output;
    EXPECT_TRUE(sweep_config.run_sweep);
    EXPECT_EQ(sweep_config.sweep_max_runs, 4u);
    EXPECT_EQ(sweep_config.output_file, "llm_inline_sweep.json");
    ASSERT_EQ(sweep_config.sweep_specs.size(), 1u);
    EXPECT_EQ(sweep_config.sweep_specs[0].parameter, LlmSweepParameter::ContextTokens);
    ASSERT_EQ(sweep_config.sweep_specs[0].values.size(), 2u);
    EXPECT_EQ(sweep_config.sweep_specs[0].values[0].integer_value, 2u);
    EXPECT_EQ(sweep_config.sweep_specs[0].values[1].integer_value, 4u);
  }
}

TEST(LlmMemoryConfigTest, ParserActivatesExplicitBackendPhaseLayoutCompatibilityMatrix) {
  LlmParserHooksScope hooks(8, 9);
  for (LlmMemoryBackend backend : {LlmMemoryBackend::Cpu, LlmMemoryBackend::Metal}) {
    for (LlmPhase phase : {LlmPhase::Decode, LlmPhase::Prefill}) {
      for (LlmKvLayout kv_layout : {LlmKvLayout::Contiguous, LlmKvLayout::Paged}) {
        SCOPED_TRACE(::testing::Message() << llm_memory_backend_to_string(backend) << "/"
                                          << llm_phase_to_string(phase) << "/"
                                          << llm_kv_layout_to_string(kv_layout));
        std::vector<std::string> arguments =
            phase == LlmPhase::Prefill ? valid_prefill_arguments() : valid_llm_arguments();
        arguments.insert(arguments.begin() + 2,
                         {"--llm-memory-backend", backend == LlmMemoryBackend::Metal ? "metal" : "cpu"});
        if (phase == LlmPhase::Decode) arguments.insert(arguments.end(), {"--phase", "decode"});
        arguments.insert(arguments.end(), {"--kv-layout", kv_layout == LlmKvLayout::Paged ? "paged" : "contiguous"});
        if (kv_layout == LlmKvLayout::Paged) arguments.insert(arguments.end(), {"--kv-block-tokens", "4"});

        LlmMemoryConfig config;
        const CapturedLlmParse parsed = parse_llm_arguments_capturing(arguments, config);
        EXPECT_EQ(parsed.result, EXIT_SUCCESS) << parsed.stderr_output;
        EXPECT_TRUE(parsed.stdout_output.empty());
        EXPECT_TRUE(parsed.stderr_output.empty());
        EXPECT_EQ(config.backend, backend);
        EXPECT_EQ(config.phase, phase);
        EXPECT_EQ(config.kv_layout, kv_layout);
        EXPECT_EQ(config.visible_context_tokens, phase == LlmPhase::Decode ? 3u : 0u);
        EXPECT_EQ(config.prompt_tokens, phase == LlmPhase::Prefill ? 5u : 0u);
        EXPECT_EQ(config.attention_query_tile_tokens, phase == LlmPhase::Prefill ? 2u : 0u);
        EXPECT_EQ(config.kv_block_tokens, kv_layout == LlmKvLayout::Paged ? 4u : 0u);
        EXPECT_EQ(config.requested_workers, backend == LlmMemoryBackend::Cpu ? 8u : 0u);
        EXPECT_EQ(config.available_workers, backend == LlmMemoryBackend::Cpu ? 8u : 0u);
        EXPECT_TRUE(config.user_specified_backend);
        EXPECT_TRUE(config.user_specified_phase);
        EXPECT_TRUE(config.user_specified_kv_layout);
        EXPECT_EQ(config.user_specified_kv_block_tokens, kv_layout == LlmKvLayout::Paged);
        EXPECT_EQ(config.argv, arguments);
      }
    }
  }
}

TEST(LlmMemoryConfigTest, ParserRejectsInvalidBackendPhaseLayoutAndMetalThreads) {
  {
    SCOPED_TRACE("metal-threads-either-order");
    LlmParserHooksScope hooks(0, 9);
    std::vector<std::string> backend_first = valid_metal_arguments();
    backend_first.insert(backend_first.end(), {"--threads", "1"});
    std::vector<std::string> threads_first = valid_llm_arguments();
    threads_first.insert(threads_first.begin() + 2, {"--threads", "1"});
    threads_first.insert(threads_first.end(), {"--llm-memory-backend", "metal"});
    for (const std::vector<std::string>& arguments : {backend_first, threads_first}) {
      SCOPED_TRACE(::testing::PrintToString(arguments));
      LlmMemoryConfig config;
      const CapturedLlmParse parsed = parse_llm_arguments_capturing(arguments, config);
      EXPECT_EQ(parsed.result, EXIT_FAILURE);
      EXPECT_TRUE(parsed.stdout_output.empty());
      EXPECT_EQ(first_output_line(parsed.stderr_output),
                Messages::error_prefix() +
                    Messages::error_llm_memory_config_invalid(LlmMemoryConfigReason::THREADS_NOT_APPLICABLE));
      EXPECT_TRUE(config.user_specified_workers);
      EXPECT_EQ(config.available_workers, 0u);
    }
  }

  {
    SCOPED_TRACE("invalid-phase-backend-and-paged-layout");
    LlmParserHooksScope hooks;
    struct InvalidCase {
      bool prefill;
      std::vector<std::string> suffix;
      std::string diagnostic;
    };
    const std::vector<InvalidCase> cases = {
        {false, {"--prompt-tokens", "5"},
         Messages::error_llm_memory_config_invalid(LlmMemoryConfigReason::PROMPT_TOKENS_NOT_APPLICABLE)},
        {false, {"--attention-query-tile-tokens", "2"},
         Messages::error_llm_memory_config_invalid(
             LlmMemoryConfigReason::ATTENTION_QUERY_TILE_TOKENS_NOT_APPLICABLE)},
        {true, {"--context-tokens", "5"},
         Messages::error_llm_memory_config_invalid(LlmMemoryConfigReason::CONTEXT_TOKENS_NOT_APPLICABLE)},
        {false, {"--phase", "training"},
         Messages::error_invalid_value("--phase", "training", Messages::llm_memory_reason_phase())},
        {false, {"--llm-memory-backend", "gpu"},
         Messages::error_invalid_value("--llm-memory-backend", "gpu", Messages::llm_memory_reason_backend())},
        {false, {"--kv-layout", "paged"}, Messages::error_llm_memory_missing_required_option("--kv-block-tokens")},
        {false, {"--kv-layout", "sparse"},
         Messages::error_invalid_value("--kv-layout", "sparse", Messages::llm_memory_reason_kv_layout())},
        {false, {"--kv-layout", "contiguous", "--kv-block-tokens", "16"},
         Messages::error_llm_memory_config_invalid(LlmMemoryConfigReason::KV_BLOCK_TOKENS_NOT_APPLICABLE)},
        {false, {"--kv-block-tokens", "16", "--kv-layout", "contiguous"},
         Messages::error_llm_memory_config_invalid(LlmMemoryConfigReason::KV_BLOCK_TOKENS_NOT_APPLICABLE)},
        {false, {"--kv-layout", "paged", "--kv-block-tokens", "0"},
         Messages::error_llm_memory_config_invalid(LlmMemoryConfigReason::KV_BLOCK_TOKENS_ZERO)},
        {false, {"--kv-block-tokens", "0", "--kv-layout", "paged"},
         Messages::error_llm_memory_config_invalid(LlmMemoryConfigReason::KV_BLOCK_TOKENS_ZERO)},
        {false, {"--kv-layout", "paged", "--kv-block-tokens", "3"},
         Messages::error_llm_memory_config_invalid(LlmMemoryConfigReason::KV_BLOCK_TOKENS_NOT_POWER_OF_TWO)},
        {false, {"--kv-block-tokens", "3", "--kv-layout", "paged"},
         Messages::error_llm_memory_config_invalid(LlmMemoryConfigReason::KV_BLOCK_TOKENS_NOT_POWER_OF_TWO)},
        {false, {"--kv-layout", "paged", "--kv-block-tokens", "4294967296"},
         Messages::error_llm_memory_config_invalid(LlmMemoryConfigReason::KV_BLOCK_TOKENS_EXCEEDS_UINT32)},
        {false, {"--kv-block-tokens", "4294967296", "--kv-layout", "paged"},
         Messages::error_llm_memory_config_invalid(LlmMemoryConfigReason::KV_BLOCK_TOKENS_EXCEEDS_UINT32)},
    };
    for (const InvalidCase& test_case : cases) {
      SCOPED_TRACE(::testing::PrintToString(test_case.suffix));
      std::vector<std::string> arguments = test_case.prefill ? valid_prefill_arguments() : valid_llm_arguments();
      arguments.insert(arguments.end(), test_case.suffix.begin(), test_case.suffix.end());
      LlmMemoryConfig config;
      const CapturedLlmParse parsed = parse_llm_arguments_capturing(arguments, config);
      EXPECT_EQ(parsed.result, EXIT_FAILURE);
      EXPECT_TRUE(parsed.stdout_output.empty());
      EXPECT_EQ(first_output_line(parsed.stderr_output), Messages::error_prefix() + test_case.diagnostic);
    }
  }
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
    EXPECT_TRUE(paged_config.user_specified_phase);
    EXPECT_TRUE(paged_config.user_specified_kv_layout);
    EXPECT_TRUE(paged_config.user_specified_kv_block_tokens);
    EXPECT_EQ(paged_config.argv, paged);
  }
}

TEST(LlmMemoryConfigTest, ParserRejectsMissingRequiredMissingValuesAndIncompatibleOptions) {
  LlmParserHooksScope hooks;

  for (const std::string& required_option :
       {"--weight-size-mb", "--layers", "--query-heads", "--kv-heads", "--head-dim", "--context-tokens"}) {
    SCOPED_TRACE("missing-required:" + required_option);
    std::vector<std::string> arguments = valid_llm_arguments();
    const auto position = std::find(arguments.begin(), arguments.end(), required_option);
    ASSERT_NE(position, arguments.end());
    arguments.erase(position, position + 2);
    LlmMemoryConfig config;
    const CapturedLlmParse parsed = parse_llm_arguments_capturing(arguments, config);
    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    EXPECT_TRUE(parsed.stdout_output.empty());
    EXPECT_EQ(first_output_line(parsed.stderr_output),
              Messages::error_prefix() + Messages::error_llm_memory_missing_required_option(required_option));
  }

  for (const std::string& option :
       {"--llm-memory-backend", "--weight-size-mb", "--layers", "--query-heads", "--kv-heads", "--head-dim",
        "--kv-element-bytes", "--phase", "--context-tokens", "--prompt-tokens", "--attention-query-tile-tokens",
        "--kv-layout", "--kv-block-tokens", "--batch-size", "--threads", "--iterations", "--count", "--seed",
        "--output"}) {
    SCOPED_TRACE("missing-value:" + option);
    LlmMemoryConfig config;
    const CapturedLlmParse parsed =
        parse_llm_arguments_capturing({"memory_benchmark", "--llm-memory", option}, config);
    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    EXPECT_EQ(first_output_line(parsed.stderr_output),
              Messages::error_prefix() + Messages::error_missing_value(option));
  }

  // All non-LLM options share the same unknown-option branch and name the offending token.
  for (const std::vector<std::string>& suffix :
       {std::vector<std::string>{"--benchmark"}, {"-P"}, {"--buffer-size", "1"}, {"--unknown"}}) {
    SCOPED_TRACE(::testing::PrintToString(suffix));
    auto arguments = valid_llm_arguments();
    arguments.insert(arguments.end(), suffix.begin(), suffix.end());
    LlmMemoryConfig config;
    const CapturedLlmParse parsed = parse_llm_arguments_capturing(arguments, config);
    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    EXPECT_TRUE(parsed.stdout_output.empty());
    EXPECT_EQ(first_output_line(parsed.stderr_output),
              Messages::error_prefix() +
                  Messages::error_option_not_allowed_in_mode(suffix.front(),
                                                             Messages::error_llm_memory_must_be_used_alone()));
  }

  for (const std::pair<std::vector<std::string>, std::string>& unexpected_case : {
           std::pair<std::vector<std::string>, std::string>{{"memory_benchmark", "--llm-memory=1"}, "--llm-memory"},
           std::pair<std::vector<std::string>, std::string>{{"memory_benchmark", "--llm-memory", "--help=1"}, "--help"},
       }) {
    SCOPED_TRACE(::testing::PrintToString(unexpected_case.first));
    LlmMemoryConfig config;
    const CapturedLlmParse parsed = parse_llm_arguments_capturing(unexpected_case.first, config);
    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    EXPECT_TRUE(parsed.stdout_output.empty());
    EXPECT_EQ(first_output_line(parsed.stderr_output),
              Messages::error_prefix() + Messages::error_option_does_not_take_value(unexpected_case.second));
  }

  {
    SCOPED_TRACE("missing-primary-mode");
    LlmMemoryConfig config;
    const CapturedLlmParse parsed = parse_llm_arguments_capturing({"memory_benchmark", "--help"}, config);
    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    EXPECT_TRUE(parsed.stdout_output.empty());
    EXPECT_EQ(first_output_line(parsed.stderr_output),
              Messages::error_prefix() + Messages::error_llm_memory_must_be_used_alone());
  }

  {
    SCOPED_TRACE("argc-zero-and-null-argv-safe");
    LlmMemoryConfig config;
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    const int status = parse_llm_memory_arguments(0, nullptr, config);
    const std::string stderr_output = testing::internal::GetCapturedStderr();
    const std::string stdout_output = testing::internal::GetCapturedStdout();
    EXPECT_EQ(status, EXIT_FAILURE);
    EXPECT_TRUE(stdout_output.empty());
    EXPECT_TRUE(config.argv.empty());
    EXPECT_EQ(first_output_line(stderr_output),
              Messages::error_prefix() + Messages::error_llm_memory_must_be_used_alone());
  }
}

TEST(LlmMemoryConfigTest, ParserRejectsDuplicateOptionsAndMalformedNumericTokens) {
  LlmParserHooksScope hooks;

  struct DuplicateCase {
    std::vector<std::string> suffix;
    std::string canonical_option;
  };
  const std::vector<DuplicateCase> duplicate_cases = {
      {{"--llm-memory"}, "--llm-memory"},
      {{"--llm-memory-backend", "cpu", "--llm-memory-backend", "metal"}, "--llm-memory-backend"},
      {{"--weight-size-mb", "1"}, "--weight-size-mb"},
      {{"--layers", "2"}, "--layers"},
      {{"--query-heads", "4"}, "--query-heads"},
      {{"--kv-heads", "2"}, "--kv-heads"},
      {{"--head-dim", "8"}, "--head-dim"},
      {{"--kv-element-bytes", "2", "--kv-element-bytes", "2"}, "--kv-element-bytes"},
      {{"--phase", "decode", "--phase", "prefill"}, "--phase"},
      {{"--context-tokens", "3"}, "--context-tokens"},
      {{"--prompt-tokens", "3", "--prompt-tokens", "4"}, "--prompt-tokens"},
      {{"--attention-query-tile-tokens", "2", "--attention-query-tile-tokens", "3"}, "--attention-query-tile-tokens"},
      {{"--kv-layout", "contiguous", "--kv-layout", "paged"}, "--kv-layout"},
      {{"--kv-block-tokens", "16", "--kv-block-tokens", "32"}, "--kv-block-tokens"},
      {{"--batch-size", "1", "--batch-size", "1"}, "--batch-size"},
      {{"-t", "2", "--threads", "2"}, "--threads"},
      {{"-i", "1", "--iterations", "1"}, "--iterations"},
      {{"-r", "3", "--count", "3"}, "--count"},
      {{"--seed", "1", "--seed", "2"}, "--seed"},
      {{"-o", "first", "--output", "second"}, "--output"},
      {{"-h", "--help"}, "--help"},
  };
  for (const DuplicateCase& test_case : duplicate_cases) {
    SCOPED_TRACE(test_case.canonical_option);
    std::vector<std::string> arguments = valid_llm_arguments();
    arguments.insert(arguments.end(), test_case.suffix.begin(), test_case.suffix.end());
    LlmMemoryConfig config;
    const CapturedLlmParse parsed = parse_llm_arguments_capturing(arguments, config);
    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    EXPECT_EQ(first_output_line(parsed.stderr_output),
              Messages::error_prefix() + Messages::error_duplicate_option(test_case.canonical_option));
  }

  // Full decimal-token grammar is owned by ConfigTest; retain each LLM dispatch and the seed's zero boundary.
  const std::vector<std::string> numeric_options = {
      "--weight-size-mb", "--layers", "--query-heads", "--kv-heads", "--head-dim", "--kv-element-bytes",
      "--context-tokens", "--kv-block-tokens", "--batch-size", "--threads", "--iterations", "--count", "--seed",
  };
  for (const std::string& option : numeric_options) {
    for (const std::string token : {"0", "1x", "18446744073709551616"}) {
      SCOPED_TRACE(option + "=" + token);
      auto arguments = valid_llm_arguments();
      const auto position = std::find(arguments.begin(), arguments.end(), option);
      if (position == arguments.end()) {
        arguments.insert(arguments.end(), {option, token});
      } else {
        *(position + 1) = token;
      }
      LlmMemoryConfig config;
      const auto parsed = parse_llm_arguments_capturing(arguments, config);
      const bool valid = option == "--seed" && token == "0";
      EXPECT_EQ(parsed.result, valid ? EXIT_SUCCESS : EXIT_FAILURE);
      if (valid) {
        EXPECT_TRUE(config.user_specified_seed);
        EXPECT_EQ(config.seed, 0u);
      } else {
        EXPECT_FALSE(parsed.stderr_output.empty());
      }
    }
  }
}

TEST(LlmMemoryConfigTest, ParserHelpAndZeroDetectedWorkerBoundaries) {
  {
    SCOPED_TRACE("help-skips-required-workers-and-seed");
    LlmParserHooksScope hooks(0, 0);
    LlmMemoryConfig config;
    const CapturedLlmParse parsed =
        parse_llm_arguments_capturing({"memory_benchmark", "-M", "--llm-memory-backend", "metal", "--help"}, config);
    EXPECT_EQ(parsed.result, EXIT_SUCCESS);
    EXPECT_TRUE(parsed.stderr_output.empty()) << parsed.stderr_output;
    EXPECT_NE(parsed.stdout_output.find("Usage: memory_benchmark --llm-memory [options]"), std::string::npos);
    EXPECT_EQ(parsed.stdout_output.find(Messages::config_header(SOFTVERSION)), std::string::npos);
    EXPECT_TRUE(config.help_printed);
    EXPECT_EQ(config.backend, LlmMemoryBackend::Metal);
    EXPECT_TRUE(config.user_specified_backend);
    EXPECT_EQ(config.available_workers, 0u);
    EXPECT_EQ(config.requested_workers, 0u);
    EXPECT_EQ(config.seed, 0u);
  }

  {
    SCOPED_TRACE("help-does-not-hide-errors");
    LlmParserHooksScope hooks;
    for (const std::vector<std::string>& arguments : {
             std::vector<std::string>{"memory_benchmark", "-M", "--help", "--unknown"},
             std::vector<std::string>{"memory_benchmark", "-M", "--help", "-h"},
             std::vector<std::string>{"memory_benchmark", "-M", "--help", "--layers"},
         }) {
      SCOPED_TRACE(::testing::PrintToString(arguments));
      LlmMemoryConfig config;
      const CapturedLlmParse parsed = parse_llm_arguments_capturing(arguments, config);
      EXPECT_EQ(parsed.result, EXIT_FAILURE);
      EXPECT_TRUE(parsed.stdout_output.empty());
      EXPECT_FALSE(parsed.stderr_output.empty());
    }
  }

  {
    SCOPED_TRACE("zero-detected-workers");
    LlmParserHooksScope hooks(0, 9);
    LlmMemoryConfig config;
    const CapturedLlmParse parsed = parse_llm_arguments_capturing(valid_llm_arguments(), config);
    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    EXPECT_TRUE(parsed.stdout_output.empty());
    EXPECT_EQ(first_output_line(parsed.stderr_output),
              Messages::error_prefix() +
                  Messages::error_llm_memory_config_invalid(LlmMemoryConfigReason::AVAILABLE_WORKER_COUNT_REQUIRED));
    EXPECT_EQ(config.seed, 0u);
  }
}

TEST(LlmMemoryConfigTest, StableVocabularyAndStatusTokens) {
  EXPECT_STREQ(llm_memory_backend_to_string(
                   static_cast<LlmMemoryBackend>(99)),
               "unknown");

  EXPECT_STREQ(llm_phase_to_string(static_cast<LlmPhase>(99)), "unknown");

  EXPECT_STREQ(llm_kv_layout_to_string(static_cast<LlmKvLayout>(99)),
               "unknown");

  EXPECT_EQ(llm_work_unit_kind_for_phase(LlmPhase::Decode),
            LlmWorkUnitKind::DecodeStep);
  EXPECT_EQ(llm_work_unit_kind_for_phase(LlmPhase::Prefill),
            LlmWorkUnitKind::PrefillOperation);
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
  EXPECT_STREQ(llm_kv_write_kind_to_string(
                   static_cast<LlmKvWriteKind>(99)),
               "unknown");

  EXPECT_STREQ(llm_scenario_to_string(static_cast<LlmScenario>(99)),
               "unknown");

  EXPECT_STREQ(
      llm_attention_kind_to_string(static_cast<LlmAttentionKind>(99)),
      "unknown");

  EXPECT_STREQ(llm_measurement_status_to_string(
                   static_cast<LlmMeasurementStatus>(99)),
               "invalid");

  EXPECT_STREQ(llm_run_status_to_string(static_cast<LlmRunStatus>(99)),
               "failed");
}

TEST(LlmMemoryConfigTest, PureConfigValidationCoversGeometryPhasesLayoutsAndMetalWorkers) {
  for (bool prefill : {false, true}) {
    const LlmMemoryConfigValidation v = validate_llm_memory_config(prefill ? valid_prefill_config() : valid_config());
    ASSERT_TRUE(v.valid) << v.reason_code;
    EXPECT_EQ(v.reason_code, LlmMemoryConfigReason::VALID);
    EXPECT_EQ(v.active_weight_bytes, 4ULL * 1024ULL * 1024ULL * 1024ULL);
  }

  using Mutate = void (*)(LlmMemoryConfig&);
  struct ValidCase {
    const char* name;
    bool prefill;
    Mutate mutate;
  };
  const std::vector<ValidCase> valid_cases = {
      {"explicit-iterations", false, [](LlmMemoryConfig& c) { c.user_specified_iterations = true; c.iterations = 4; }},
      {"mha-heads", false, [](LlmMemoryConfig& c) { c.query_head_count = c.kv_head_count; }},
      {"mqa-heads", false, [](LlmMemoryConfig& c) { c.query_head_count = 8; c.kv_head_count = 1; }},
      {"paged-block-16", false, [](LlmMemoryConfig& c) {
        c.kv_layout = LlmKvLayout::Paged; c.kv_block_tokens = 16;
        c.user_specified_kv_layout = true; c.user_specified_kv_block_tokens = true;
      }},
      {"paged-block-uint32-pow2", false, [](LlmMemoryConfig& c) {
        c.kv_layout = LlmKvLayout::Paged; c.kv_block_tokens = static_cast<size_t>(1) << 31;
        c.user_specified_kv_layout = true; c.user_specified_kv_block_tokens = true;
      }},
      {"prefill-unit", true, [](LlmMemoryConfig& c) { c.prompt_tokens = 1; c.attention_query_tile_tokens = 1; }},
      {"prefill-tile-eq-prompt", true, [](LlmMemoryConfig& c) { c.attention_query_tile_tokens = c.prompt_tokens; }},
      {"prefill-unflagged", true, [](LlmMemoryConfig& c) {
        c.user_specified_prompt_tokens = false; c.user_specified_attention_query_tile_tokens = false;
      }},
      {"metal-decode-zero-workers", false, [](LlmMemoryConfig& c) {
        c.backend = LlmMemoryBackend::Metal; c.requested_workers = 0; c.available_workers = 0;
      }},
      {"metal-prefill-zero-workers", true, [](LlmMemoryConfig& c) {
        c.backend = LlmMemoryBackend::Metal; c.requested_workers = 0; c.available_workers = 0;
      }},
  };
  for (const ValidCase& tc : valid_cases) {
    SCOPED_TRACE(tc.name);
    LlmMemoryConfig config = tc.prefill ? valid_prefill_config() : valid_config();
    tc.mutate(config);
    const LlmMemoryConfigValidation validation = validate_llm_memory_config(config);
    EXPECT_TRUE(validation.valid) << validation.reason_code;
  }

  for (size_t invalid_width : {0u, 3u, 5u, 8u}) {
    SCOPED_TRACE(::testing::Message() << "kv_element_bytes=" << invalid_width);
    LlmMemoryConfig config = valid_config();
    config.kv_element_bytes = invalid_width;
    expect_invalid(config, LlmMemoryConfigReason::INVALID_KV_ELEMENT_BYTES);
  }

  struct InvalidCase {
    const char* name;
    bool prefill;
    Mutate mutate;
    const char* reason_code;
  };
  const std::vector<InvalidCase> invalid_cases = {
      {"zero-weight", false, [](LlmMemoryConfig& c) { c.weight_size_mb = 0; },
       LlmMemoryConfigReason::WEIGHT_SIZE_REQUIRED},
      {"zero-layers", false, [](LlmMemoryConfig& c) { c.layer_count = 0; },
       LlmMemoryConfigReason::LAYER_COUNT_REQUIRED},
      {"zero-query-heads", false, [](LlmMemoryConfig& c) { c.query_head_count = 0; },
       LlmMemoryConfigReason::QUERY_HEAD_COUNT_REQUIRED},
      {"zero-kv-heads", false, [](LlmMemoryConfig& c) { c.kv_head_count = 0; },
       LlmMemoryConfigReason::KV_HEAD_COUNT_REQUIRED},
      {"zero-head-dim", false, [](LlmMemoryConfig& c) { c.head_dimension = 0; },
       LlmMemoryConfigReason::HEAD_DIMENSION_REQUIRED},
      {"zero-context", false, [](LlmMemoryConfig& c) { c.visible_context_tokens = 0; },
       LlmMemoryConfigReason::CONTEXT_TOKENS_REQUIRED},
      {"paged-block-req", false, [](LlmMemoryConfig& c) { c.kv_layout = LlmKvLayout::Paged; },
       LlmMemoryConfigReason::KV_BLOCK_TOKENS_REQUIRED},
      {"paged-block-zero", false,
       [](LlmMemoryConfig& c) { c.kv_layout = LlmKvLayout::Paged; c.user_specified_kv_block_tokens = true; },
       LlmMemoryConfigReason::KV_BLOCK_TOKENS_ZERO},
      {"paged-block-non-pow2", false,
       [](LlmMemoryConfig& c) {
         c.kv_layout = LlmKvLayout::Paged; c.kv_block_tokens = 3; c.user_specified_kv_block_tokens = true;
       },
       LlmMemoryConfigReason::KV_BLOCK_TOKENS_NOT_POWER_OF_TWO},
      {"paged-block-u32", false,
       [](LlmMemoryConfig& c) {
         c.kv_layout = LlmKvLayout::Paged; c.user_specified_kv_block_tokens = true;
         c.kv_block_tokens = static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 1;
       },
       LlmMemoryConfigReason::KV_BLOCK_TOKENS_EXCEEDS_UINT32},
      {"contiguous-block-na", false,
       [](LlmMemoryConfig& c) { c.kv_block_tokens = 16; c.user_specified_kv_block_tokens = true; },
       LlmMemoryConfigReason::KV_BLOCK_TOKENS_NOT_APPLICABLE},
      {"zero-batch", false, [](LlmMemoryConfig& c) { c.batch_size = 0; }, LlmMemoryConfigReason::BATCH_SIZE_REQUIRED},
      {"zero-workers", false, [](LlmMemoryConfig& c) { c.requested_workers = 0; },
       LlmMemoryConfigReason::WORKER_COUNT_REQUIRED},
      {"zero-loops", false, [](LlmMemoryConfig& c) { c.loop_count = 0; }, LlmMemoryConfigReason::LOOP_COUNT_REQUIRED},
      {"query-below-kv", false, [](LlmMemoryConfig& c) { c.query_head_count = 4; c.kv_head_count = 8; },
       LlmMemoryConfigReason::QUERY_HEADS_BELOW_KV_HEADS},
      {"query-not-div", false, [](LlmMemoryConfig& c) { c.query_head_count = 10; c.kv_head_count = 4; },
       LlmMemoryConfigReason::QUERY_HEADS_NOT_DIVISIBLE_BY_KV_HEADS},
      {"explicit-iter-zero", false, [](LlmMemoryConfig& c) { c.user_specified_iterations = true; c.iterations = 0; },
       LlmMemoryConfigReason::EXPLICIT_ITERATIONS_REQUIRED},
      {"auto-iter-nonzero", false, [](LlmMemoryConfig& c) { c.iterations = 4; },
       LlmMemoryConfigReason::AUTOMATIC_ITERATIONS_MUST_BE_ZERO},
      {"invalid-phase", false, [](LlmMemoryConfig& c) { c.phase = static_cast<LlmPhase>(99); },
       LlmMemoryConfigReason::INVALID_PHASE},
      {"decode-prompt-flag", false, [](LlmMemoryConfig& c) { c.user_specified_prompt_tokens = true; },
       LlmMemoryConfigReason::PROMPT_TOKENS_NOT_APPLICABLE},
      {"decode-prompt-val", false, [](LlmMemoryConfig& c) { c.prompt_tokens = 1; },
       LlmMemoryConfigReason::PROMPT_TOKENS_NOT_APPLICABLE},
      {"decode-tile-flag", false, [](LlmMemoryConfig& c) { c.user_specified_attention_query_tile_tokens = true; },
       LlmMemoryConfigReason::ATTENTION_QUERY_TILE_TOKENS_NOT_APPLICABLE},
      {"decode-tile-val", false, [](LlmMemoryConfig& c) { c.attention_query_tile_tokens = 1; },
       LlmMemoryConfigReason::ATTENTION_QUERY_TILE_TOKENS_NOT_APPLICABLE},
      {"prefill-ctx-flag", true, [](LlmMemoryConfig& c) { c.user_specified_context_tokens = true; },
       LlmMemoryConfigReason::CONTEXT_TOKENS_NOT_APPLICABLE},
      {"prefill-ctx-val", true, [](LlmMemoryConfig& c) { c.visible_context_tokens = 1; },
       LlmMemoryConfigReason::CONTEXT_TOKENS_NOT_APPLICABLE},
      {"prefill-prompt-req", true,
       [](LlmMemoryConfig& c) { c.prompt_tokens = 0; c.user_specified_prompt_tokens = false; },
       LlmMemoryConfigReason::PROMPT_TOKENS_REQUIRED},
      {"prefill-prompt-zero", true, [](LlmMemoryConfig& c) { c.prompt_tokens = 0; },
       LlmMemoryConfigReason::PROMPT_TOKENS_MUST_BE_POSITIVE},
      {"prefill-tile-req", true,
       [](LlmMemoryConfig& c) {
         c.attention_query_tile_tokens = 0; c.user_specified_attention_query_tile_tokens = false;
       },
       LlmMemoryConfigReason::ATTENTION_QUERY_TILE_TOKENS_REQUIRED},
      {"prefill-tile-zero", true, [](LlmMemoryConfig& c) { c.attention_query_tile_tokens = 0; },
       LlmMemoryConfigReason::ATTENTION_QUERY_TILE_TOKENS_MUST_BE_POSITIVE},
      {"prefill-tile-gt-prompt", true, [](LlmMemoryConfig& c) { c.attention_query_tile_tokens = c.prompt_tokens + 1; },
       LlmMemoryConfigReason::ATTENTION_QUERY_TILE_TOKENS_EXCEEDS_PROMPT},
      {"metal-req-workers", false,
       [](LlmMemoryConfig& c) {
         c.backend = LlmMemoryBackend::Metal; c.user_specified_workers = true;
         c.requested_workers = 1; c.available_workers = 0;
       },
       LlmMemoryConfigReason::THREADS_NOT_APPLICABLE},
      {"metal-avail-workers", false,
       [](LlmMemoryConfig& c) {
         c.backend = LlmMemoryBackend::Metal; c.requested_workers = 0; c.available_workers = 1;
       },
       LlmMemoryConfigReason::THREADS_NOT_APPLICABLE},
      {"invalid-backend", false, [](LlmMemoryConfig& c) { c.backend = static_cast<LlmMemoryBackend>(99); },
       LlmMemoryConfigReason::INVALID_BACKEND},
  };
  for (const InvalidCase& tc : invalid_cases) {
    SCOPED_TRACE(tc.name);
    LlmMemoryConfig config = tc.prefill ? valid_prefill_config() : valid_config();
    tc.mutate(config);
    expect_invalid(config, tc.reason_code);
  }
}

TEST(LlmMemoryConfigTest, ConfigGuardrailsWeightOverflowAndSafeJsonIntegerBounds) {
  {
    SCOPED_TRACE("weight-mib-checked-arithmetic");
    LlmMemoryConfig config = valid_config();
    config.weight_size_mb = 4096;
    LlmMemoryConfigValidation validation = validate_llm_memory_config(config);
    ASSERT_TRUE(validation.valid);
    EXPECT_EQ(validation.active_weight_bytes, 4294967296ULL);

    const size_t maximum_mb = std::numeric_limits<size_t>::max() / Constants::BYTES_PER_MB;
    config.weight_size_mb = maximum_mb;
    validation = validate_llm_memory_config(config);
    ASSERT_TRUE(validation.valid);
    EXPECT_EQ(validation.active_weight_bytes, maximum_mb * Constants::BYTES_PER_MB);

    config.weight_size_mb = maximum_mb + 1;
    expect_invalid(config, LlmMemoryConfigReason::WEIGHT_SIZE_BYTES_OVERFLOW);
  }

  {
    SCOPED_TRACE("ieee754-safe-json-integer-bounds");
    static_assert(Constants::LLM_JSON_MAX_SAFE_INTEGER < std::numeric_limits<size_t>::max());
    constexpr size_t kUnsafeInteger = Constants::LLM_JSON_MAX_SAFE_INTEGER + 1;
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
      expect_invalid(config, LlmMemoryConfigReason::JSON_INTEGER_OUT_OF_RANGE);
    }

    LlmMemoryConfig config = valid_config();
    config.user_specified_iterations = true;
    config.iterations = kUnsafeInteger;
    expect_invalid(config, LlmMemoryConfigReason::JSON_INTEGER_OUT_OF_RANGE);

    // sweep_max_runs is emitted only for sweep envelopes, so it must not reject direct runs.
    config = valid_config();
    config.run_sweep = false;
    config.sweep_max_runs = kUnsafeInteger;
    EXPECT_TRUE(validate_llm_memory_config(config).valid);
    config.run_sweep = true;
    expect_invalid(config, LlmMemoryConfigReason::JSON_INTEGER_OUT_OF_RANGE);

    config = valid_config();
    config.loop_count = Constants::LLM_JSON_MAX_SAFE_INTEGER / kLlmScenarioCount;
    EXPECT_TRUE(validate_llm_memory_config(config).valid);
    ++config.loop_count;
    expect_invalid(config, LlmMemoryConfigReason::JSON_INTEGER_OUT_OF_RANGE);
  }

  {
    SCOPED_TRACE("parser-weight-and-iteration-guardrails");
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
    const CapturedLlmParse parsed = parse_llm_arguments_capturing(arguments, config);
    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    EXPECT_TRUE(parsed.stdout_output.empty());
    EXPECT_EQ(first_output_line(parsed.stderr_output),
              Messages::error_prefix() + Messages::error_llm_memory_iterations_exceed_limit(65505, 65504));
  }
}

TEST(LlmMemoryConfigTest,
     TimerSetupFailureKeepsStdoutEmptyAndPreservesLegacyDiagnosticBoundaryIntegration) {
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

TEST(LlmMemoryConfigIntegrationTest, PostRunExceptionWritesFailedStdoutAndAtomicallyReplacesFileIntegration) {
  for (bool to_file : {false, true}) {
    SCOPED_TRACE(to_file ? "file" : "stdout");
    const LlmCommandOutputFileScope output;
    std::vector<std::string> arguments = valid_llm_arguments();
    arguments.insert(arguments.end(), {"--threads", "1", "--iterations", "1", "--count", "1", "--seed", "1",
                                       "--output", to_file ? output.path().string() : "-"});
    const LlmParserHooksScope parser_hooks(1, 1);
    size_t hook_calls = 0;
    bool observed_complete_terminal_file = false;
    const LlmCommandHooksScope command_hooks([&]() {
      ++hook_calls;
      if (to_file) {
        const nlohmann::ordered_json before_exception =
            nlohmann::ordered_json::parse(read_llm_command_output_file(output.path()));
        observed_complete_terminal_file = before_exception.at("status") == "complete";
      }
      throw std::runtime_error("injected post-run command exception");
    });

    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    const int status = run_llm_command(std::move(arguments));
    const std::string stderr_output = testing::internal::GetCapturedStderr();
    const std::string stdout_output = testing::internal::GetCapturedStdout();

    EXPECT_EQ(status, EXIT_FAILURE);
    EXPECT_EQ(hook_calls, 1u);
    std::string payload;
    if (to_file) {
      EXPECT_TRUE(observed_complete_terminal_file);
      ASSERT_TRUE(std::filesystem::exists(output.path()));
      EXPECT_FALSE(std::filesystem::exists(output.path().string() + ".tmp"));
      EXPECT_EQ(stdout_output.find("Results saved"), std::string::npos);
      payload = read_llm_command_output_file(output.path());
    } else {
      ASSERT_FALSE(stdout_output.empty());
      EXPECT_EQ(stdout_output.back(), '\n');
      payload = stdout_output;
    }

    nlohmann::ordered_json document;
    ASSERT_NO_THROW(document = nlohmann::ordered_json::parse(payload));
    EXPECT_EQ(document.at("status"), "failed");
    EXPECT_EQ(document.at("reason_code"), LlmRunnerReason::RUNNER_EXCEPTION);
    EXPECT_TRUE(document.at("results_complete").get<bool>());
    EXPECT_FALSE(document.at("run_accepted").get<bool>());
    const auto& lifecycle = document.at("checkpoint_lifecycle");
    EXPECT_EQ(lifecycle.at("current_request"), "late-command-error-correction");
    EXPECT_EQ(lifecycle.at("prior_file_writer_attempts"), to_file ? 2 : 0);
    EXPECT_EQ(lifecycle.at("prior_successful_file_writes"), to_file ? 2 : 0);
    EXPECT_TRUE(lifecycle.at("current_persistence_success").is_null());
    EXPECT_NE(stderr_output.find("injected post-run command exception"), std::string::npos);
  }
}

TEST(LlmMemoryConfigTest, ParsesDecodePagedPrefillAndMetalCartesianSweeps) {
  {
    SCOPED_TRACE("decode-paged-sweep");
    LlmParserHooksScope hooks(4, 987654321ULL);
    LlmMemoryConfig config;
    const CapturedLlmParse parsed = parse_llm_arguments_capturing(
        {"memory_benchmark", "--llm-memory", "--weight-size-mb", "64", "--layers", "4", "--query-heads", "8",
         "--kv-heads", "2", "--head-dim", "64", "--kv-layout", "paged", "--sweep", "context-tokens=32,64", "--sweep",
         "kv-block-tokens=16,32", "--sweep-max-runs", "4", "--output", "llm_sweep.json"},
        config);

    ASSERT_EQ(parsed.result, EXIT_SUCCESS) << parsed.stderr_output;
    EXPECT_TRUE(config.run_sweep);
    EXPECT_EQ(config.sweep_max_runs, 4u);
    EXPECT_EQ(calculate_llm_sweep_run_count(config), 4u);
    EXPECT_EQ(config.seed, 987654321ULL);
    EXPECT_FALSE(config.user_specified_seed);

    const std::vector<LlmMemoryConfig> run_configs = build_llm_sweep_run_configs(config);
    ASSERT_EQ(run_configs.size(), 4u);
    const std::array<std::pair<size_t, size_t>, 4> expected_pairs = {{{32u, 16u}, {32u, 32u}, {64u, 16u}, {64u, 32u}}};
    for (size_t i = 0; i < expected_pairs.size(); ++i) {
      EXPECT_FALSE(run_configs[i].run_sweep);
      EXPECT_TRUE(run_configs[i].sweep_specs.empty());
      EXPECT_TRUE(run_configs[i].output_file.empty());
      EXPECT_TRUE(run_configs[i].collect_json_for_sweep);
      EXPECT_EQ(run_configs[i].visible_context_tokens, expected_pairs[i].first);
      EXPECT_TRUE(run_configs[i].user_specified_context_tokens);
      EXPECT_EQ(run_configs[i].kv_block_tokens, expected_pairs[i].second);
      EXPECT_TRUE(run_configs[i].user_specified_kv_block_tokens);
      EXPECT_EQ(run_configs[i].seed, config.seed);
    }
  }

  LlmParserHooksScope hooks(4, 12345ULL);
  {
    SCOPED_TRACE("prefill-paged-sweep");
    LlmMemoryConfig config;
    const CapturedLlmParse parsed = parse_llm_arguments_capturing(
        {"memory_benchmark", "--llm-memory", "--phase", "prefill", "--weight-size-mb", "64", "--layers", "4",
         "--query-heads", "8", "--kv-heads", "2", "--head-dim", "64", "--kv-layout", "paged", "--sweep",
         "prompt-tokens=64,128", "--sweep", "attention-query-tile-tokens=16,32", "--sweep", "kv-block-tokens=16,64",
         "--seed", "42", "--output", "-"},
        config);
    ASSERT_EQ(parsed.result, EXIT_SUCCESS) << parsed.stderr_output;
    EXPECT_TRUE(config.run_sweep);
    EXPECT_EQ(calculate_llm_sweep_run_count(config), 8u);
    const std::vector<LlmMemoryConfig> runs = build_llm_sweep_run_configs(config);
    ASSERT_EQ(runs.size(), 8u);
    EXPECT_EQ(runs[0].prompt_tokens, 64u);
    EXPECT_EQ(runs[0].attention_query_tile_tokens, 16u);
    EXPECT_EQ(runs[0].kv_block_tokens, 16u);
    EXPECT_EQ(runs[7].prompt_tokens, 128u);
    EXPECT_EQ(runs[7].attention_query_tile_tokens, 32u);
    EXPECT_EQ(runs[7].kv_block_tokens, 64u);
  }

  {
    SCOPED_TRACE("metal-decode-sweep");
    LlmMemoryConfig config;
    const CapturedLlmParse parsed = parse_llm_arguments_capturing(
        {"memory_benchmark", "--llm-memory", "--llm-memory-backend", "metal", "--weight-size-mb", "64", "--layers", "4",
         "--query-heads", "8", "--kv-heads", "2", "--head-dim", "64", "--sweep", "context-tokens=32,64", "--output",
         "metal_sweep.json"},
        config);
    ASSERT_EQ(parsed.result, EXIT_SUCCESS) << parsed.stderr_output;
    EXPECT_EQ(config.backend, LlmMemoryBackend::Metal);
    EXPECT_EQ(config.available_workers, 0u);
    EXPECT_EQ(config.requested_workers, 0u);
  }
}

TEST(LlmMemoryConfigTest, RejectsInvalidAndConflictingLlmSweepConfigurations) {
  LlmParserHooksScope hooks(4, 12345ULL);

  const auto sweep_args = [](std::initializer_list<const char*> extra, bool prefill = false,
                             bool with_output = true) {
    std::vector<std::string> args = {"memory_benchmark", "--llm-memory"};
    if (prefill) args.insert(args.end(), {"--phase", "prefill"});
    args.insert(args.end(), {"--weight-size-mb", "64", "--layers", "4", "--query-heads", "8", "--kv-heads", "2",
                             "--head-dim", "64"});
    for (const char* token : extra) args.emplace_back(token);
    if (with_output) args.insert(args.end(), {"--output", "out.json"});
    return args;
  };

  struct InvalidSweepCase {
    const char* name;
    std::vector<std::string> args;
    std::string expected_substring;
  };
  const std::vector<InvalidSweepCase> cases = {
      {"direct context conflict", sweep_args({"--context-tokens", "32", "--sweep", "context-tokens=32,64"}),
       Messages::error_sweep_option_conflict("--context-tokens", "context-tokens")},
      {"direct block conflict",
       sweep_args({"--context-tokens", "64", "--kv-layout", "paged", "--kv-block-tokens", "16", "--sweep",
                   "kv-block-tokens=16,32"}),
       Messages::error_sweep_option_conflict("--kv-block-tokens", "kv-block-tokens")},
      {"direct prompt conflict",
       sweep_args({"--prompt-tokens", "64", "--attention-query-tile-tokens", "16", "--sweep", "prompt-tokens=64,128"},
                  true),
       Messages::error_sweep_option_conflict("--prompt-tokens", "prompt-tokens")},
      {"direct query tile conflict",
       sweep_args({"--prompt-tokens", "64", "--attention-query-tile-tokens", "16", "--sweep",
                   "attention-query-tile-tokens=16,32"},
                  true),
       Messages::error_sweep_option_conflict("--attention-query-tile-tokens", "attention-query-tile-tokens")},
      {"context sweep in prefill",
       sweep_args({"--prompt-tokens", "64", "--attention-query-tile-tokens", "16", "--sweep", "context-tokens=32,64"},
                  true),
       Messages::error_sweep_parameter_not_allowed("context-tokens", "--llm-memory --phase prefill")},
      {"prompt sweep in decode", sweep_args({"--context-tokens", "64", "--sweep", "prompt-tokens=32,64"}),
       Messages::error_sweep_parameter_not_allowed("prompt-tokens", "--llm-memory --phase decode")},
      {"tile sweep in decode", sweep_args({"--context-tokens", "64", "--sweep", "attention-query-tile-tokens=16,32"}),
       Messages::error_sweep_parameter_not_allowed("attention-query-tile-tokens", "--llm-memory --phase decode")},
      {"block sweep in contiguous", sweep_args({"--context-tokens", "64", "--sweep", "kv-block-tokens=16,32"}),
       Messages::error_sweep_parameter_not_allowed("kv-block-tokens", "--llm-memory --kv-layout contiguous")},
      {"unsupported key", sweep_args({"--context-tokens", "64", "--sweep", "layers=4,8"}),
       Messages::error_sweep_parameter_not_allowed("layers", "--llm-memory")},
      {"duplicate sweep key", sweep_args({"--sweep", "context-tokens=32,64", "--sweep", "context-tokens=128"}),
       Messages::error_duplicate_sweep_parameter("context-tokens")},
      {"missing output", sweep_args({"--sweep", "context-tokens=32,64"}, false, false),
       Messages::error_sweep_requires_output()},
      {"exceeds sweep-max-runs", sweep_args({"--sweep", "context-tokens=32,64,128", "--sweep-max-runs", "2"}),
       Messages::error_sweep_too_many_runs(3u, 2u)},
      {"sweep-max-runs exceeds exact JSON integer range",
       sweep_args({"--sweep", "context-tokens=32,64", "--sweep-max-runs", "9007199254740992"}),
       LlmMemoryConfigReason::JSON_INTEGER_OUT_OF_RANGE},
      {"non-power-of-two block in sweep preflight",
       sweep_args({"--context-tokens", "64", "--kv-layout", "paged", "--sweep", "kv-block-tokens=16,24"}),
       LlmMemoryConfigReason::KV_BLOCK_TOKENS_NOT_POWER_OF_TWO},
      {"prefill tile exceeds prompt in Cartesian product",
       sweep_args({"--sweep", "prompt-tokens=32,64", "--sweep", "attention-query-tile-tokens=16,48"}, true),
       LlmMemoryConfigReason::ATTENTION_QUERY_TILE_TOKENS_EXCEEDS_PROMPT},
      {"metal sweep rejects threads",
       sweep_args({"--llm-memory-backend", "metal", "--threads", "2", "--sweep", "context-tokens=32,64"}),
       LlmMemoryConfigReason::THREADS_NOT_APPLICABLE},
  };

  for (const InvalidSweepCase& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    LlmMemoryConfig config;
    const CapturedLlmParse parsed = parse_llm_arguments_capturing(test_case.args, config);
    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    EXPECT_TRUE(parsed.stdout_output.empty());
    EXPECT_NE(parsed.stderr_output.find(test_case.expected_substring), std::string::npos) << parsed.stderr_output;
  }

  // Preflight failures must identify the exact failing sweep combination.
  {
    LlmMemoryConfig config;
    const CapturedLlmParse parsed = parse_llm_arguments_capturing(
        sweep_args({"--sweep", "prompt-tokens=32,64", "--sweep", "attention-query-tile-tokens=16,48"}, true), config);
    EXPECT_EQ(parsed.result, EXIT_FAILURE);
    EXPECT_NE(parsed.stderr_output.find(Messages::error_llm_memory_sweep_combination_invalid(
                  2u, 4u, "prompt-tokens=32, attention-query-tile-tokens=48")),
              std::string::npos)
        << parsed.stderr_output;
  }

  // Direct commands without --sweep must remain unaffected by an out-of-JSON-range --sweep-max-runs.
  {
    LlmMemoryConfig config;
    const CapturedLlmParse parsed = parse_llm_arguments_capturing(
        sweep_args({"--context-tokens", "32", "--sweep-max-runs", "9007199254740992"}, false, false), config);
    EXPECT_EQ(parsed.result, EXIT_SUCCESS) << parsed.stderr_output;
    EXPECT_FALSE(config.run_sweep);
  }
}

TEST(LlmMemoryConfigIntegrationTest, DirectSetupFailureWritesSchema2DocumentForStdoutAndFileIntegration) {
  auto verify_setup_failure_document = [](const nlohmann::ordered_json& doc, size_t expected_loops,
                                          const std::string& expected_reason) {
    EXPECT_EQ(doc.at("schema_version"), Constants::LLM_JSON_SCHEMA_VERSION);
    EXPECT_EQ(doc.at("mode"), Constants::LLM_JSON_MODE_NAME);
    EXPECT_EQ(doc.at("status"), "failed");
    EXPECT_EQ(doc.at("reason_code"), expected_reason);
    EXPECT_FALSE(doc.at("results_complete").get<bool>());
    EXPECT_FALSE(doc.at("run_accepted").get<bool>());
    EXPECT_FALSE(doc.at("scenario_order_balance_complete").get<bool>());
    ASSERT_EQ(doc.at("loop_records").size(), expected_loops);
    for (size_t loop_idx = 0; loop_idx < expected_loops; ++loop_idx) {
      const auto& loop = doc.at("loop_records")[loop_idx];
      EXPECT_EQ(loop.at("loop_index"), loop_idx);
      EXPECT_EQ(loop.at("realized_order").size(), 0u);
      EXPECT_EQ(loop.at("realized_order_count"), 0u);
      ASSERT_EQ(loop.at("measurement_indexes").size(), kLlmScenarioCount);
      for (size_t pos = 0; pos < kLlmScenarioCount; ++pos) {
        EXPECT_EQ(loop.at("measurement_indexes")[pos], loop_idx * kLlmScenarioCount + pos);
      }
    }
    const size_t expected_measurements = expected_loops * kLlmScenarioCount;
    ASSERT_EQ(doc.at("measurements").size(), expected_measurements);
    for (size_t i = 0; i < expected_measurements; ++i) {
      const auto& row = doc.at("measurements")[i];
      EXPECT_EQ(row.at("measurement_id"), i);
      EXPECT_TRUE(row.at("plan_ref").is_null());
      EXPECT_EQ(row.at("status"), "failed");
      EXPECT_EQ(row.at("reason_code"), LlmRunnerReason::NOT_RUN_AFTER_RUNTIME_FAILURE);
      EXPECT_FALSE(row.at("attempted").get<bool>());
    }
    const auto& counters = doc.at("counters");
    EXPECT_EQ(counters.at("planned_loops"), expected_loops);
    EXPECT_EQ(counters.at("attempted_loops"), 0u);
    EXPECT_EQ(counters.at("completed_loops"), 0u);
    EXPECT_EQ(counters.at("planned_measurements"), expected_measurements);
    EXPECT_EQ(counters.at("attempted_measurements"), 0u);
    EXPECT_EQ(counters.at("terminal_measurements"), expected_measurements);
    EXPECT_EQ(counters.at("measured_measurements"), 0u);
    const auto& lifecycle = doc.at("checkpoint_lifecycle");
    EXPECT_EQ(lifecycle.at("current_request"), "terminal");
    EXPECT_EQ(lifecycle.at("snapshot_interval_loops"),
              std::max<size_t>(1, expected_loops / 8 + (expected_loops % 8 != 0)));
  };

  const LlmParserHooksScope parser_hooks(1, 1);
  for (bool throw_exception : {false, true}) {
    for (bool to_file : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "throw_exception=" << throw_exception << " to_file=" << to_file);
      const LlmCommandOutputFileScope output;
      std::vector<std::string> arguments = valid_llm_arguments();
      arguments.insert(arguments.end(), {"--threads", "1", "--iterations", "1", "--count", "9", "--seed", "1",
                                         "--output", to_file ? output.path().string() : "-"});
      LlmMemoryCommandTestHooks hooks;
      hooks.before_runner = [throw_exception](LlmMemoryWorkPlan& plan) {
        if (throw_exception) throw std::runtime_error("injected pre-init command exception");
        plan.memory_budget.request.orchestration_auxiliary_bytes = 0;
      };
      const LlmCommandHooksScope command_hooks(std::move(hooks));

      testing::internal::CaptureStdout();
      testing::internal::CaptureStderr();
      const int status = run_llm_command(std::move(arguments));
      const std::string stderr_output = testing::internal::GetCapturedStderr();
      const std::string stdout_output = testing::internal::GetCapturedStdout();

      EXPECT_EQ(status, EXIT_FAILURE) << stderr_output;
      const std::string payload = to_file ? read_llm_command_output_file(output.path()) : stdout_output;
      ASSERT_FALSE(payload.empty()) << stderr_output;
      nlohmann::ordered_json document;
      ASSERT_NO_THROW(document = nlohmann::ordered_json::parse(payload));
      verify_setup_failure_document(document, 9u,
                                    throw_exception ? LlmRunnerReason::RUNNER_EXCEPTION
                                                    : LlmRunnerReason::AUXILIARY_BUDGET_INSUFFICIENT);
    }
  }
}

TEST(LlmMemoryConfigIntegrationTest, LlmSweepCollectCoversPreInitErrorAndInterruptionBranchesIntegration) {
  struct CollectOutcome {
    int status = EXIT_FAILURE;
    nlohmann::ordered_json collected;
    std::string reason;
    bool prep_interrupted = false;
    std::string stdout_text;
    std::string stderr_text;
  };
  const auto parse_collect_config = [](std::vector<std::string> args) {
    LlmMemoryConfig config;
    const CapturedLlmParse parsed = parse_llm_arguments_capturing(std::move(args), config);
    EXPECT_EQ(parsed.result, EXIT_SUCCESS) << parsed.stderr_output;
    config.collect_json_for_sweep = true;
    return config;
  };
  const auto run_collect = [](const LlmMemoryConfig& config, bool initial_prep_interrupted) {
    CollectOutcome outcome;
    outcome.prep_interrupted = initial_prep_interrupted;
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    outcome.status =
        run_llm_memory_collect(config, outcome.collected, &outcome.reason, &outcome.prep_interrupted);
    outcome.stderr_text = testing::internal::GetCapturedStderr();
    outcome.stdout_text = testing::internal::GetCapturedStdout();
    return outcome;
  };
  const std::vector<std::string> base_decode_args = {
      "memory_benchmark", "--llm-memory", "--weight-size-mb", "1", "--layers", "2", "--query-heads", "4", "--kv-heads",
      "2", "--head-dim", "16", "--context-tokens", "16", "--threads", "1", "--iterations", "1", "--count", "1",
      "--seed", "42"};

  // 0. Failing timer pre-init preserves TIMER_UNAVAILABLE reason and legacy diagnostic with empty JSON.
  {
    SCOPED_TRACE("preinit-timer-failure");
    const LlmMemoryConfig config = parse_collect_config(base_decode_args);
    const ScopedFailingTimerSystemCalls timer_system_calls;
    const CollectOutcome out = run_collect(config, false);
    EXPECT_EQ(out.status, EXIT_FAILURE) << out.stdout_text << out.stderr_text;
    EXPECT_TRUE(out.collected.empty());
    EXPECT_FALSE(out.prep_interrupted);
    EXPECT_EQ(out.reason, LlmBackendReason::TIMER_UNAVAILABLE);
    EXPECT_NE(out.stderr_text.find(Messages::error_mach_timebase_info_failed(mach_error_string(KERN_FAILURE))),
              std::string::npos);
  }

  // 1. Invalid backend fails before planning with empty JSON and preserved reason.
  {
    SCOPED_TRACE("invalid-backend");
    LlmMemoryConfig config = parse_collect_config(base_decode_args);
    config.backend = static_cast<LlmMemoryBackend>(99);
    const CollectOutcome out = run_collect(config, true);
    EXPECT_EQ(out.status, EXIT_FAILURE);
    EXPECT_TRUE(out.collected.empty());
    EXPECT_FALSE(out.prep_interrupted);
    EXPECT_EQ(out.reason, LlmBackendReason::INVALID_BACKEND);
  }

  // 2. Interruption during prepare_llm_memory_work_plan (paged decode ownership).
  {
    SCOPED_TRACE("prepare-plan-interrupted");
    std::vector<std::string> args = base_decode_args;
    args.insert(args.end(), {"--kv-layout", "paged", "--kv-block-tokens", "8"});
    const LlmMemoryConfig config = parse_collect_config(std::move(args));
    LlmMemoryCommandTestHooks hooks;
    hooks.stop_requested = []() { return true; };
    const LlmCommandHooksScope command_hooks(std::move(hooks));
    const CollectOutcome out = run_collect(config, false);
    EXPECT_EQ(out.status, EXIT_SUCCESS);
    EXPECT_TRUE(out.collected.empty());
    EXPECT_TRUE(out.prep_interrupted);
    EXPECT_EQ(out.reason, LlmKvLayoutReason::PREPARATION_INTERRUPTED);
  }

  // 3. Interruption during finalize_llm_memory_work_plan (paged prefill table materialization).
  {
    SCOPED_TRACE("finalize-plan-interrupted");
    const LlmMemoryConfig config = parse_collect_config(
        {"memory_benchmark", "--llm-memory", "--phase", "prefill", "--kv-layout", "paged", "--weight-size-mb", "1",
         "--layers", "2", "--query-heads", "4", "--kv-heads", "2", "--head-dim", "16", "--prompt-tokens", "16",
         "--attention-query-tile-tokens", "8", "--kv-block-tokens", "8", "--threads", "1", "--iterations", "1",
         "--count", "1", "--seed", "42"});
    LlmMemoryCommandTestHooks hooks;
    hooks.stop_requested = []() { return true; };
    const LlmCommandHooksScope command_hooks(std::move(hooks));
    const CollectOutcome out = run_collect(config, false);
    EXPECT_EQ(out.status, EXIT_SUCCESS);
    EXPECT_TRUE(out.collected.empty());
    EXPECT_TRUE(out.prep_interrupted);
    EXPECT_EQ(out.reason, LlmKvLayoutReason::PREPARATION_INTERRUPTED);
  }

  // 4. Interruption reported by backend->prepare_resources().
  {
    SCOPED_TRACE("backend-prepare-interrupted");
    const LlmMemoryConfig config = parse_collect_config(base_decode_args);
    const LlmBackendFactoryHookScope backend_hook(
        [](LlmMemoryBackend) { return std::make_unique<PreparationInterruptedFakeBackend>(); });
    const CollectOutcome out = run_collect(config, false);
    EXPECT_EQ(out.status, EXIT_SUCCESS);
    EXPECT_TRUE(out.collected.empty());
    EXPECT_TRUE(out.prep_interrupted);
    EXPECT_EQ(out.reason, LlmBackendReason::PREPARATION_INTERRUPTED);
  }

  // 5. Non-timer !result.initialized setup failure populates schema-2 failed JSON.
  {
    SCOPED_TRACE("setup-failure-aux-budget");
    const LlmMemoryConfig config = parse_collect_config(base_decode_args);
    LlmMemoryCommandTestHooks hooks;
    hooks.before_runner = [](LlmMemoryWorkPlan& plan) {
      plan.memory_budget.request.orchestration_auxiliary_bytes = 0;
    };
    const LlmCommandHooksScope command_hooks(std::move(hooks));
    const CollectOutcome out = run_collect(config, true);
    EXPECT_EQ(out.status, EXIT_FAILURE);
    EXPECT_FALSE(out.prep_interrupted);
    EXPECT_EQ(out.reason, LlmRunnerReason::AUXILIARY_BUDGET_INSUFFICIENT);
    ASSERT_FALSE(out.collected.empty());
    EXPECT_EQ(out.collected.at("status"), "failed");
    EXPECT_EQ(out.collected.at("reason_code"), LlmRunnerReason::AUXILIARY_BUDGET_INSUFFICIENT);
    EXPECT_EQ(out.collected.at("measurements").size(), kLlmScenarioCount);
    EXPECT_EQ(out.collected.at("counters").at("terminal_measurements"), kLlmScenarioCount);
  }

  // 6. Exception before result initialization populates schema-2 setup-failure JSON.
  {
    SCOPED_TRACE("preinit-exception");
    const LlmMemoryConfig config = parse_collect_config(base_decode_args);
    LlmMemoryCommandTestHooks hooks;
    hooks.before_runner = [](LlmMemoryWorkPlan&) { throw std::runtime_error("injected collect pre-init exception"); };
    const LlmCommandHooksScope command_hooks(std::move(hooks));
    const CollectOutcome out = run_collect(config, true);
    EXPECT_EQ(out.status, EXIT_FAILURE);
    EXPECT_FALSE(out.prep_interrupted);
    EXPECT_EQ(out.reason, LlmRunnerReason::RUNNER_EXCEPTION);
    ASSERT_FALSE(out.collected.empty());
    EXPECT_EQ(out.collected.at("status"), "failed");
    EXPECT_EQ(out.collected.at("reason_code"), LlmRunnerReason::RUNNER_EXCEPTION);
    EXPECT_EQ(out.collected.at("measurements").size(), kLlmScenarioCount);
  }

  // 7. Post-run exception in collect mode rewrites collected JSON with late-command-error-correction.
  {
    SCOPED_TRACE("post-run-exception");
    const LlmMemoryConfig config = parse_collect_config(base_decode_args);
    const LlmCommandHooksScope command_hooks(
        []() { throw std::runtime_error("injected collect post-run exception"); });
    const CollectOutcome out = run_collect(config, true);
    EXPECT_EQ(out.status, EXIT_FAILURE);
    EXPECT_FALSE(out.prep_interrupted);
    EXPECT_EQ(out.reason, LlmRunnerReason::RUNNER_EXCEPTION);
    ASSERT_FALSE(out.collected.empty());
    EXPECT_EQ(out.collected.at("status"), "failed");
    EXPECT_EQ(out.collected.at("reason_code"), LlmRunnerReason::RUNNER_EXCEPTION);
    EXPECT_TRUE(out.collected.at("results_complete").get<bool>());
    EXPECT_FALSE(out.collected.at("run_accepted").get<bool>());
    EXPECT_EQ(out.collected.at("checkpoint_lifecycle").at("current_request"), "late-command-error-correction");
  }

  // 8. Graceful task-boundary interruption returns EXIT_SUCCESS with interrupted schema-2 JSON.
  {
    SCOPED_TRACE("graceful-task-interruption");
    const LlmMemoryConfig config = parse_collect_config(base_decode_args);
    LlmMemoryCommandTestHooks hooks;
    hooks.stop_requested = []() { return true; };
    const LlmCommandHooksScope command_hooks(std::move(hooks));
    const CollectOutcome out = run_collect(config, true);
    EXPECT_EQ(out.status, EXIT_SUCCESS);
    EXPECT_FALSE(out.prep_interrupted);
    ASSERT_FALSE(out.collected.empty());
    EXPECT_EQ(out.collected.at("status"), "interrupted");
    EXPECT_TRUE(out.collected.at("interruption_requested").get<bool>());
    EXPECT_FALSE(out.collected.at("run_accepted").get<bool>());
  }
}

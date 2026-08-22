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

/**
 * @file llm_memory.cpp
 * @brief Standalone parsing, command boundary, validation, and status tokens
 */

#include "llm_memory/llm_memory.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#include "core/config/config.h"
#include "core/config/version.h"
#include "core/signal/signal_handler.h"
#include "core/system/benchmark_qos.h"
#include "core/system/page_size.h"
#include "core/system/system_info.h"
#include "core/timing/timer.h"
#include "llm_memory/llm_environment.h"
#include "llm_memory/llm_executor.h"
#include "llm_memory/llm_json.h"
#include "llm_memory/llm_output.h"
#include "llm_memory/llm_runner.h"
#include "llm_memory/llm_work_plan.h"
#include "output/console/messages/messages_api.h"
#include "output/console/output_printer.h"
#include "output/json/json_output/json_output_session.h"
#include "utils/json_utils.h"
#include "utils/numeric_utils.h"
#include "utils/seed_utils.h"

namespace {

LlmMemoryParserTestHooks active_llm_parser_hooks;
bool llm_parser_hooks_active = false;

bool is_option(const std::string& argument, const char* short_option,
               const char* long_option) {
  return argument == short_option || argument == long_option;
}

bool report_duplicate(bool& seen, const char* canonical_option) {
  if (!seen) {
    seen = true;
    return false;
  }
  std::cerr << Messages::error_prefix()
            << Messages::error_duplicate_option(canonical_option)
            << std::endl;
  return true;
}

bool take_value(int argc, char* argv[], int& index,
                const char* canonical_option, std::string& value) {
  if (++index >= argc) {
    std::cerr << Messages::error_prefix()
              << Messages::error_missing_value(canonical_option)
              << std::endl;
    return false;
  }
  value = argv[index];
  return true;
}

bool parse_positive_size(const std::string& option,
                         const std::string& token, size_t& value) {
  uint64_t parsed = 0;
  const StrictIntegerParseStatus status =
      parse_strict_unsigned_decimal(token, parsed);
  if (status != StrictIntegerParseStatus::Success || parsed == 0 ||
      parsed > std::numeric_limits<size_t>::max()) {
    std::string reason;
    if (status != StrictIntegerParseStatus::Success) {
      reason = strict_unsigned_decimal_error_reason(status);
    } else if (parsed == 0) {
      reason = Messages::llm_memory_reason_positive_integer();
    } else {
      reason = Messages::llm_memory_reason_platform_size_range();
    }
    std::cerr << Messages::error_prefix()
              << Messages::error_invalid_value(option, token, reason)
              << std::endl;
    return false;
  }
  value = static_cast<size_t>(parsed);
  return true;
}

void print_llm_memory_help(const char* program_name) {
  std::cout << Messages::usage_header(SOFTVERSION)
            << Messages::llm_memory_usage_options(program_name);
}

void report_llm_config_failure(const std::string& reason_code) {
  std::cerr << Messages::error_prefix()
            << Messages::error_llm_memory_config_invalid(reason_code)
            << std::endl;
}

void report_json_output_boundary_failure(const std::string& raw_output,
                                         const std::string& details) noexcept {
  try {
    std::cerr << Messages::error_prefix();
    if (raw_output == "-") {
      std::cerr << Messages::error_json_stdout_write_failed(details);
    } else {
      std::cerr << Messages::error_file_write_failed(raw_output, details);
    }
    std::cerr << std::endl;
  } catch (...) {
    // A secondary diagnostic failure must not escape the command boundary.
  }
}

void report_llm_command_exception(const std::string& details) noexcept {
  try {
    std::cerr << Messages::error_prefix()
              << Messages::error_command_execution_exception(
                     Messages::llm_memory_command_name(), details)
              << std::endl;
  } catch (...) {
    // A secondary diagnostic failure must not escape the command boundary.
  }
}

int write_llm_stdout_final(
    JsonOutputSession& output_session, const LlmMemoryConfig& config,
    const LlmMemoryWorkPlan& model_plan,
    const LlmResourcePreparationResult& preparation,
    const LlmResultMetadata& metadata,
    const LlmMemoryResult& result) noexcept {
  try {
    return output_session.write_final(build_llm_memory_json(config, model_plan, preparation, metadata, result));
  } catch (const std::exception& error) {
    report_json_output_boundary_failure(
        config.output_file,
        Messages::error_json_payload_construction_failed(error.what()));
  } catch (...) {
    report_json_output_boundary_failure(
        config.output_file,
        Messages::error_json_payload_construction_failed(""));
  }
  return EXIT_FAILURE;
}

size_t detect_available_llm_workers() {
  if (llm_parser_hooks_active) {
    return active_llm_parser_hooks.detected_workers;
  }
  const int detected = get_total_logical_cores();
  return detected > 0 ? static_cast<size_t>(detected) : 0;
}

uint64_t generate_llm_seed() {
  if (llm_parser_hooks_active) {
    const uint64_t injected_seed = active_llm_parser_hooks.generated_seed;
    return SeedUtils::generate_seed(
        [injected_seed]() { return injected_seed; });
  }
  return SeedUtils::generate_seed();
}

}  // namespace

void set_llm_memory_parser_test_hooks(
    const LlmMemoryParserTestHooks* hooks) {
  if (hooks == nullptr) {
    active_llm_parser_hooks = LlmMemoryParserTestHooks{};
    llm_parser_hooks_active = false;
    return;
  }
  active_llm_parser_hooks = *hooks;
  llm_parser_hooks_active = true;
}

int parse_llm_memory_arguments(int argc, char* argv[],
                               LlmMemoryConfig& config) {
  config = LlmMemoryConfig{};
  config.argv.reserve(static_cast<size_t>(std::max(argc, 0)));
  for (int index = 0; index < argc; ++index) {
    config.argv.emplace_back(argv[index]);
  }

  bool mode_seen = false;
  bool weight_seen = false;
  bool layers_seen = false;
  bool query_heads_seen = false;
  bool kv_heads_seen = false;
  bool head_dimension_seen = false;
  bool kv_element_bytes_seen = false;
  bool context_tokens_seen = false;
  bool batch_size_seen = false;
  bool threads_seen = false;
  bool iterations_seen = false;
  bool count_seen = false;
  bool seed_seen = false;
  bool output_seen = false;
  bool help_seen = false;

  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (is_option(argument, "-M", "--llm-memory")) {
      if (report_duplicate(mode_seen, "--llm-memory")) {
        return EXIT_FAILURE;
      }
      continue;
    }
    if (is_option(argument, "-h", "--help")) {
      if (report_duplicate(help_seen, "--help")) {
        return EXIT_FAILURE;
      }
      config.help_printed = true;
      continue;
    }
    if (argument == "--weight-size-mb") {
      if (report_duplicate(weight_seen, "--weight-size-mb")) {
        return EXIT_FAILURE;
      }
      std::string token;
      if (!take_value(argc, argv, index, "--weight-size-mb", token) ||
          !parse_positive_size(argument, token, config.weight_size_mb)) {
        return EXIT_FAILURE;
      }
      continue;
    }
    if (argument == "--layers") {
      if (report_duplicate(layers_seen, "--layers")) {
        return EXIT_FAILURE;
      }
      std::string token;
      if (!take_value(argc, argv, index, "--layers", token) ||
          !parse_positive_size(argument, token, config.layer_count)) {
        return EXIT_FAILURE;
      }
      continue;
    }
    if (argument == "--query-heads") {
      if (report_duplicate(query_heads_seen, "--query-heads")) {
        return EXIT_FAILURE;
      }
      std::string token;
      if (!take_value(argc, argv, index, "--query-heads", token) ||
          !parse_positive_size(argument, token,
                               config.query_head_count)) {
        return EXIT_FAILURE;
      }
      continue;
    }
    if (argument == "--kv-heads") {
      if (report_duplicate(kv_heads_seen, "--kv-heads")) {
        return EXIT_FAILURE;
      }
      std::string token;
      if (!take_value(argc, argv, index, "--kv-heads", token) ||
          !parse_positive_size(argument, token, config.kv_head_count)) {
        return EXIT_FAILURE;
      }
      continue;
    }
    if (argument == "--head-dim") {
      if (report_duplicate(head_dimension_seen, "--head-dim")) {
        return EXIT_FAILURE;
      }
      std::string token;
      if (!take_value(argc, argv, index, "--head-dim", token) ||
          !parse_positive_size(argument, token, config.head_dimension)) {
        return EXIT_FAILURE;
      }
      continue;
    }
    if (argument == "--kv-element-bytes") {
      if (report_duplicate(kv_element_bytes_seen, "--kv-element-bytes")) {
        return EXIT_FAILURE;
      }
      std::string token;
      if (!take_value(argc, argv, index, "--kv-element-bytes", token) ||
          !parse_positive_size(argument, token, config.kv_element_bytes)) {
        return EXIT_FAILURE;
      }
      if (config.kv_element_bytes != 1 && config.kv_element_bytes != 2 &&
          config.kv_element_bytes != 4) {
        std::cerr << Messages::error_prefix()
                  << Messages::error_invalid_value(
                         argument, token,
                         Messages::llm_memory_reason_kv_element_bytes())
                  << std::endl;
        return EXIT_FAILURE;
      }
      continue;
    }
    if (argument == "--context-tokens") {
      if (report_duplicate(context_tokens_seen, "--context-tokens")) {
        return EXIT_FAILURE;
      }
      std::string token;
      if (!take_value(argc, argv, index, "--context-tokens", token) ||
          !parse_positive_size(argument, token,
                               config.visible_context_tokens)) {
        return EXIT_FAILURE;
      }
      continue;
    }
    if (argument == "--batch-size") {
      if (report_duplicate(batch_size_seen, "--batch-size")) {
        return EXIT_FAILURE;
      }
      std::string token;
      if (!take_value(argc, argv, index, "--batch-size", token) ||
          !parse_positive_size(argument, token, config.batch_size)) {
        return EXIT_FAILURE;
      }
      continue;
    }
    if (is_option(argument, "-t", "--threads")) {
      if (report_duplicate(threads_seen, "--threads")) {
        return EXIT_FAILURE;
      }
      std::string token;
      if (!take_value(argc, argv, index, "--threads", token) ||
          !parse_positive_size(argument, token,
                               config.requested_workers)) {
        return EXIT_FAILURE;
      }
      config.user_specified_workers = true;
      continue;
    }
    if (is_option(argument, "-i", "--iterations")) {
      if (report_duplicate(iterations_seen, "--iterations")) {
        return EXIT_FAILURE;
      }
      std::string token;
      if (!take_value(argc, argv, index, "--iterations", token) ||
          !parse_positive_size(argument, token, config.iterations)) {
        return EXIT_FAILURE;
      }
      config.user_specified_iterations = true;
      continue;
    }
    if (is_option(argument, "-r", "--count")) {
      if (report_duplicate(count_seen, "--count")) {
        return EXIT_FAILURE;
      }
      std::string token;
      if (!take_value(argc, argv, index, "--count", token) ||
          !parse_positive_size(argument, token, config.loop_count)) {
        return EXIT_FAILURE;
      }
      continue;
    }
    if (argument == "--seed") {
      if (report_duplicate(seed_seen, "--seed")) {
        return EXIT_FAILURE;
      }
      std::string token;
      if (!take_value(argc, argv, index, "--seed", token)) {
        return EXIT_FAILURE;
      }
      const StrictIntegerParseStatus status =
          parse_strict_unsigned_decimal(token, config.seed);
      if (status != StrictIntegerParseStatus::Success) {
        std::cerr << Messages::error_prefix()
                  << Messages::error_invalid_value(
                         argument, token,
                         strict_unsigned_decimal_error_reason(status))
                  << std::endl;
        return EXIT_FAILURE;
      }
      config.user_specified_seed = true;
      continue;
    }
    if (is_option(argument, "-o", "--output")) {
      if (report_duplicate(output_seen, "--output")) {
        return EXIT_FAILURE;
      }
      if (!take_value(argc, argv, index, "--output", config.output_file)) {
        return EXIT_FAILURE;
      }
      continue;
    }

    std::cerr << Messages::error_prefix()
              << Messages::error_llm_memory_must_be_used_alone()
              << std::endl;
    return EXIT_FAILURE;
  }

  if (!mode_seen) {
    std::cerr << Messages::error_prefix()
              << Messages::error_llm_memory_must_be_used_alone()
              << std::endl;
    return EXIT_FAILURE;
  }
  if (config.help_printed) {
    print_llm_memory_help(argc > 0 ? argv[0] : "memory_benchmark");
    return EXIT_SUCCESS;
  }

  const std::array<std::pair<bool, const char*>, 6> required_options = {{
      {weight_seen, "--weight-size-mb"},
      {layers_seen, "--layers"},
      {query_heads_seen, "--query-heads"},
      {kv_heads_seen, "--kv-heads"},
      {head_dimension_seen, "--head-dim"},
      {context_tokens_seen, "--context-tokens"},
  }};
  for (const auto& [seen, option] : required_options) {
    if (!seen) {
      std::cerr << Messages::error_prefix()
                << Messages::error_llm_memory_missing_required_option(option)
                << std::endl;
      return EXIT_FAILURE;
    }
  }

  config.available_workers = detect_available_llm_workers();
  if (config.available_workers == 0) {
    report_llm_config_failure(
        LlmMemoryConfigReason::AVAILABLE_WORKER_COUNT_REQUIRED);
    return EXIT_FAILURE;
  }
  if (!config.user_specified_workers) {
    config.requested_workers = config.available_workers;
  }

  const LlmMemoryConfigValidation validation =
      validate_llm_memory_config(config);
  if (!validation.valid) {
    report_llm_config_failure(validation.reason_code);
    return EXIT_FAILURE;
  }

  const LlmGeometry geometry = resolve_llm_geometry(
      {validation.active_weight_bytes,
       config.layer_count,
       config.query_head_count,
       config.kv_head_count,
       config.head_dimension,
       config.kv_element_bytes,
       config.visible_context_tokens,
       config.batch_size});
  if (!geometry.valid) {
    report_llm_config_failure(geometry.reason_code);
    return EXIT_FAILURE;
  }

  constexpr std::array<LlmScenario, kLlmScenarioCount> kScenarios = {
      LlmScenario::WeightsOnly, LlmScenario::KvOnly, LlmScenario::Mixed};
  size_t maximum_explicit_iterations =
      std::numeric_limits<size_t>::max();
  for (LlmScenario scenario : kScenarios) {
    const LlmScenarioLimits limits =
        calculate_llm_scenario_limits(geometry, scenario);
    if (!limits.valid) {
      report_llm_config_failure(limits.reason_code);
      return EXIT_FAILURE;
    }
    maximum_explicit_iterations =
        std::min(maximum_explicit_iterations,
                 limits.effective_maximum_work_units);
  }
  if (config.user_specified_iterations &&
      config.iterations > maximum_explicit_iterations) {
    std::cerr << Messages::error_prefix()
              << Messages::error_llm_memory_iterations_exceed_limit(
                     config.iterations, maximum_explicit_iterations)
              << std::endl;
    return EXIT_FAILURE;
  }

  if (!config.user_specified_seed) {
    config.seed = generate_llm_seed();
  }
  return EXIT_SUCCESS;
}

int run_llm_memory_mode(int argc, char* argv[]) {
  LlmMemoryConfig config;
  int parse_status = EXIT_FAILURE;
  try {
    parse_status = parse_llm_memory_arguments(argc, argv, config);
  } catch (const std::exception& error) {
    report_llm_command_exception(error.what());
    return EXIT_FAILURE;
  } catch (...) {
    report_llm_command_exception("");
    return EXIT_FAILURE;
  }
  if (parse_status != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (config.help_printed) {
    return EXIT_SUCCESS;
  }

  std::optional<JsonOutputSession> output_session;
  try {
    output_session.emplace(make_json_output_target(
        config.output_file,
        JsonFilePathPolicy::ResolveAgainstCurrentDirectory));
  } catch (const std::exception& error) {
    report_json_output_boundary_failure(
        config.output_file,
        Messages::error_json_output_initialization_failed(error.what()));
    return EXIT_FAILURE;
  } catch (...) {
    report_json_output_boundary_failure(
        config.output_file,
        Messages::error_json_output_initialization_failed(""));
    return EXIT_FAILURE;
  }

  int run_status = EXIT_FAILURE;
  LlmMemoryWorkPlan model_plan;
  LlmExecutionResources resources;
  LlmResourcePreparationResult preparation;
  LlmResultMetadata metadata;
  LlmMemoryResult result;
  try {
    print_runtime_banner();
    metadata.timestamp = build_utc_timestamp();
    metadata.main_thread_qos = prepare_main_thread_benchmark_qos(MainThreadQosSetter{}, false);
    BenchmarkSignalMaskGuard signal_guard;

    metadata.environment_start = capture_llm_host_environment();
    metadata.environment_end = metadata.environment_start;
    metadata.processor_name = get_processor_name();
    metadata.macos_version = get_macos_version();
    metadata.performance_core_count = get_performance_cores();
    metadata.efficiency_core_count = get_efficiency_cores();
    metadata.logical_core_count = get_total_logical_cores();
    metadata.page_size_bytes = get_system_page_size_bytes();
    metadata.l1_data_cache_bytes = get_l1_cache_size();
    metadata.l2_data_cache_bytes = get_l2_cache_size();

    const unsigned long available_memory_mb = get_available_memory_mb();
    if (available_memory_mb == 0) {
      metadata.available_memory_source = "unavailable";
    } else {
      if (!NumericUtils::checked_multiply(static_cast<size_t>(available_memory_mb), Constants::BYTES_PER_MB,
                                          metadata.available_memory_bytes)) {
        std::cerr << Messages::error_prefix()
                  << Messages::error_llm_memory_run_failed(LlmWorkPlanReason::MEMORY_BUDGET_OVERFLOW) << std::endl;
        return EXIT_FAILURE;
      }
    }

    size_t checksum_auxiliary_bytes = 0;
    size_t orchestration_auxiliary_bytes = 0;
    {
      const LlmMemoryWorkPlan preliminary = build_llm_memory_work_plan(
          config, config.available_workers, metadata.available_memory_bytes, metadata.page_size_bytes, 0, 0);
      if (!preliminary.valid) {
        std::cerr << Messages::error_prefix() << Messages::error_llm_memory_run_failed(preliminary.reason_code)
                  << std::endl;
        return EXIT_FAILURE;
      }
      const LlmExecutorAuxiliaryEstimate executor_auxiliary = calculate_llm_executor_auxiliary_estimate(preliminary);
      const LlmRunnerAuxiliaryEstimate runner_auxiliary = calculate_llm_runner_auxiliary_estimate(config, preliminary);
      metadata.json_peak_estimate = calculate_llm_json_peak_estimate(config, preliminary);
      if (!metadata.json_peak_estimate.valid) {
        std::cerr << Messages::error_prefix()
                  << Messages::error_llm_memory_run_failed(std::string(metadata.json_peak_estimate.reason_code))
                  << std::endl;
        return EXIT_FAILURE;
      }
      size_t runner_and_executor_orchestration_bytes = 0;
      if (!executor_auxiliary.valid || !runner_auxiliary.valid ||
          !NumericUtils::checked_add(executor_auxiliary.checksum_auxiliary_bytes,
                                     runner_auxiliary.checksum_auxiliary_bytes, checksum_auxiliary_bytes) ||
          !NumericUtils::checked_add(executor_auxiliary.orchestration_auxiliary_bytes,
                                     runner_auxiliary.orchestration_auxiliary_bytes,
                                     runner_and_executor_orchestration_bytes)) {
        const std::string reason_code =
            !executor_auxiliary.valid
                ? std::string(executor_auxiliary.reason_code)
                : (!runner_auxiliary.valid ? std::string(runner_auxiliary.reason_code)
                                           : std::string(LlmRunnerReason::AUXILIARY_BYTES_OVERFLOW));
        std::cerr << Messages::error_prefix() << Messages::error_llm_memory_run_failed(reason_code) << std::endl;
        return EXIT_FAILURE;
      }
      if (!NumericUtils::checked_add(runner_and_executor_orchestration_bytes, metadata.json_peak_estimate.total_bytes,
                                     orchestration_auxiliary_bytes)) {
        std::cerr << Messages::error_prefix()
                  << Messages::error_llm_memory_run_failed(LlmJsonReason::PEAK_BYTES_OVERFLOW) << std::endl;
        return EXIT_FAILURE;
      }
    }

    model_plan =
        build_llm_memory_work_plan(config, config.available_workers, metadata.available_memory_bytes,
                                   metadata.page_size_bytes, checksum_auxiliary_bytes, orchestration_auxiliary_bytes);
    if (!model_plan.valid) {
      std::cerr << Messages::error_prefix() << Messages::error_llm_memory_run_failed(model_plan.reason_code)
                << std::endl;
      return EXIT_FAILURE;
    }

    std::optional<HighResTimer> timer = HighResTimer::create();
    if (!timer.has_value()) {
      return EXIT_FAILURE;
    }
    preparation = prepare_llm_execution_resources(model_plan, resources);
    if (!preparation.valid) {
      std::cerr << Messages::error_prefix() << Messages::error_llm_memory_run_failed(preparation.reason_code)
                << std::endl;
      return EXIT_FAILURE;
    }

    LlmRunnerHooks hooks;
    hooks.checkpoint = [&](const LlmMemoryResult& snapshot, LlmCheckpointKind kind) {
      static_cast<void>(kind);
      return output_session->checkpoint([&]() {
        metadata.environment_end = capture_llm_host_environment();
        return build_llm_memory_json(config, model_plan, preparation, metadata, snapshot);
      });
    };
    const LlmTaskExecutor executor = [&](const LlmMemoryWorkPlan& plan, const LlmScenarioWorkPlan& scenario_plan,
                                         const LlmRunnerTaskContext& context) {
      static_cast<void>(context);
      return execute_llm_scenario(plan, scenario_plan, resources, *timer);
    };

    run_status = run_llm_memory_suite(config, model_plan, executor, result, hooks);
    if (output_session->kind() != JsonOutputKind::File || !result.terminal_checkpoint_completed) {
      metadata.environment_end = capture_llm_host_environment();
    }
    // File checkpoints need the live mappings, but final stdout serialization
    // does not. Release them once all benchmark evidence has been captured.
    resources = LlmExecutionResources{};
    print_llm_memory_console_report(model_plan, metadata, result);

    if (output_session->kind() == JsonOutputKind::File && result.terminal_checkpoint_completed &&
        !result.checkpoint_failed) {
      std::cout << Messages::msg_results_saved_to(config.output_file) << std::endl;
    }
    if (result.status == LlmRunStatus::Interrupted) {
      std::cout << Messages::msg_interrupted_by_user() << std::endl;
    } else if (run_status != EXIT_SUCCESS) {
      std::cerr << Messages::error_prefix() << Messages::error_llm_memory_run_failed(result.reason_code) << std::endl;
    }
  } catch (const std::exception& error) {
    report_llm_command_exception(error.what());
    run_status = EXIT_FAILURE;
  } catch (...) {
    report_llm_command_exception("");
    run_status = EXIT_FAILURE;
  }

  if (output_session->kind() == JsonOutputKind::Stdout && result.initialized &&
      write_llm_stdout_final(*output_session, config, model_plan, preparation, metadata, result) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  return run_status;
}

LlmMemoryConfigValidation validate_llm_memory_config(
    const LlmMemoryConfig& config) {
  LlmMemoryConfigValidation validation;
  if (config.weight_size_mb == 0) {
    validation.reason_code = LlmMemoryConfigReason::WEIGHT_SIZE_REQUIRED;
    return validation;
  }
  if (config.layer_count == 0) {
    validation.reason_code = LlmMemoryConfigReason::LAYER_COUNT_REQUIRED;
    return validation;
  }
  if (config.query_head_count == 0) {
    validation.reason_code =
        LlmMemoryConfigReason::QUERY_HEAD_COUNT_REQUIRED;
    return validation;
  }
  if (config.kv_head_count == 0) {
    validation.reason_code = LlmMemoryConfigReason::KV_HEAD_COUNT_REQUIRED;
    return validation;
  }
  if (config.head_dimension == 0) {
    validation.reason_code = LlmMemoryConfigReason::HEAD_DIMENSION_REQUIRED;
    return validation;
  }
  if (config.kv_element_bytes != 1 && config.kv_element_bytes != 2 &&
      config.kv_element_bytes != 4) {
    validation.reason_code =
        LlmMemoryConfigReason::INVALID_KV_ELEMENT_BYTES;
    return validation;
  }
  if (config.visible_context_tokens == 0) {
    validation.reason_code =
        LlmMemoryConfigReason::CONTEXT_TOKENS_REQUIRED;
    return validation;
  }
  if (config.batch_size == 0) {
    validation.reason_code = LlmMemoryConfigReason::BATCH_SIZE_REQUIRED;
    return validation;
  }
  if (config.requested_workers == 0) {
    validation.reason_code = LlmMemoryConfigReason::WORKER_COUNT_REQUIRED;
    return validation;
  }
  if (config.loop_count == 0) {
    validation.reason_code = LlmMemoryConfigReason::LOOP_COUNT_REQUIRED;
    return validation;
  }
  const std::array<size_t, 11> json_integer_fields = {
      config.weight_size_mb,
      config.layer_count,
      config.query_head_count,
      config.kv_head_count,
      config.head_dimension,
      config.visible_context_tokens,
      config.batch_size,
      config.requested_workers,
      config.available_workers,
      config.iterations,
      config.loop_count,
  };
  if (std::any_of(json_integer_fields.begin(), json_integer_fields.end(),
                  [](size_t value) {
                    return value > Constants::LLM_JSON_MAX_SAFE_INTEGER;
                  }) ||
      config.loop_count >
          Constants::LLM_JSON_MAX_SAFE_INTEGER / kLlmScenarioCount) {
    validation.reason_code =
        LlmMemoryConfigReason::JSON_INTEGER_OUT_OF_RANGE;
    return validation;
  }
  if (config.query_head_count < config.kv_head_count) {
    validation.reason_code =
        LlmMemoryConfigReason::QUERY_HEADS_BELOW_KV_HEADS;
    return validation;
  }
  if (config.query_head_count % config.kv_head_count != 0) {
    validation.reason_code =
        LlmMemoryConfigReason::QUERY_HEADS_NOT_DIVISIBLE_BY_KV_HEADS;
    return validation;
  }
  if (config.user_specified_iterations && config.iterations == 0) {
    validation.reason_code =
        LlmMemoryConfigReason::EXPLICIT_ITERATIONS_REQUIRED;
    return validation;
  }
  if (!config.user_specified_iterations && config.iterations != 0) {
    validation.reason_code =
        LlmMemoryConfigReason::AUTOMATIC_ITERATIONS_MUST_BE_ZERO;
    return validation;
  }
  if (!NumericUtils::checked_multiply(config.weight_size_mb,
                                      Constants::BYTES_PER_MB,
                                      validation.active_weight_bytes)) {
    validation.reason_code =
        LlmMemoryConfigReason::WEIGHT_SIZE_BYTES_OVERFLOW;
    return validation;
  }

  validation.valid = true;
  validation.reason_code = LlmMemoryConfigReason::VALID;
  return validation;
}

const char* llm_scenario_to_string(LlmScenario scenario) {
  switch (scenario) {
    case LlmScenario::WeightsOnly:
      return "weights_only";
    case LlmScenario::KvOnly:
      return "kv_only";
    case LlmScenario::Mixed:
      return "mixed";
  }
  return "unknown";
}

const char* llm_memory_backend_to_string(LlmMemoryBackend backend) {
  switch (backend) {
    case LlmMemoryBackend::Cpu:
      return "cpu";
    case LlmMemoryBackend::Metal:
      return "metal";
  }
  return "unknown";
}

const char* llm_phase_to_string(LlmPhase phase) {
  switch (phase) {
    case LlmPhase::Decode:
      return "decode";
    case LlmPhase::Prefill:
      return "prefill";
  }
  return "unknown";
}

const char* llm_kv_layout_to_string(LlmKvLayout layout) {
  switch (layout) {
    case LlmKvLayout::Contiguous:
      return "contiguous";
    case LlmKvLayout::Paged:
      return "paged";
  }
  return "unknown";
}

LlmWorkUnitKind llm_work_unit_kind_for_phase(LlmPhase phase) {
  switch (phase) {
    case LlmPhase::Decode:
      return LlmWorkUnitKind::DecodeStep;
    case LlmPhase::Prefill:
      return LlmWorkUnitKind::PrefillOperation;
  }
  return static_cast<LlmWorkUnitKind>(UINT8_MAX);
}

const char* llm_work_unit_kind_to_string(LlmWorkUnitKind kind) {
  switch (kind) {
    case LlmWorkUnitKind::DecodeStep:
      return "decode_step";
    case LlmWorkUnitKind::PrefillOperation:
      return "prefill_operation";
  }
  return "unknown";
}

LlmKvWriteKind llm_kv_write_kind_for(LlmPhase phase,
                                     LlmScenario scenario) {
  if (scenario == LlmScenario::WeightsOnly) {
    return LlmKvWriteKind::None;
  }
  if (scenario != LlmScenario::KvOnly && scenario != LlmScenario::Mixed) {
    return static_cast<LlmKvWriteKind>(UINT8_MAX);
  }
  switch (phase) {
    case LlmPhase::Decode:
      return LlmKvWriteKind::CurrentTokenAppend;
    case LlmPhase::Prefill:
      return LlmKvWriteKind::FullPromptPopulation;
  }
  return static_cast<LlmKvWriteKind>(UINT8_MAX);
}

const char* llm_kv_write_kind_to_string(LlmKvWriteKind kind) {
  switch (kind) {
    case LlmKvWriteKind::None:
      return "none";
    case LlmKvWriteKind::CurrentTokenAppend:
      return "current_token_append";
    case LlmKvWriteKind::FullPromptPopulation:
      return "full_prompt_population";
  }
  return "unknown";
}

const char* llm_attention_kind_to_string(LlmAttentionKind kind) {
  switch (kind) {
    case LlmAttentionKind::Mha:
      return "mha";
    case LlmAttentionKind::Gqa:
      return "gqa";
    case LlmAttentionKind::Mqa:
      return "mqa";
  }
  return "unknown";
}

const char* llm_measurement_status_to_string(LlmMeasurementStatus status) {
  switch (status) {
    case LlmMeasurementStatus::NotRun:
      return "not_run";
    case LlmMeasurementStatus::Measured:
      return "measured";
    case LlmMeasurementStatus::Interrupted:
      return "interrupted";
    case LlmMeasurementStatus::Invalid:
      return "invalid";
    case LlmMeasurementStatus::Failed:
      return "failed";
  }
  return "invalid";
}

const char* llm_run_status_to_string(LlmRunStatus status) {
  switch (status) {
    case LlmRunStatus::NotStarted:
      return "not_started";
    case LlmRunStatus::Complete:
      return "complete";
    case LlmRunStatus::Partial:
      return "partial";
    case LlmRunStatus::Interrupted:
      return "interrupted";
    case LlmRunStatus::Unsupported:
      return "unsupported";
    case LlmRunStatus::Failed:
      return "failed";
  }
  return "failed";
}

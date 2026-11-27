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
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "core/audit/build_manifest.h"
#include "core/audit/checkpoint_policy.h"

#include "core/config/cli_parser.h"
#include "core/config/config.h"
#include "core/config/sweep_utils.h"
#include "core/config/version.h"
#include "core/signal/signal_handler.h"
#include "core/system/benchmark_qos.h"
#include "core/system/page_size.h"
#include "core/system/system_info.h"
#include "llm_memory/llm_backend.h"
#include "llm_memory/llm_environment.h"
#include "llm_memory/llm_json.h"
#include "llm_memory/llm_metal_backend.h"
#include "llm_memory/llm_output.h"
#include "llm_memory/llm_runner.h"
#include "llm_memory/llm_sweep_runner.h"
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
LlmMemoryCommandTestHooks active_llm_command_hooks;
bool llm_command_hooks_active = false;

enum LlmOptionId {
  OPT_ID_LLM_MEMORY = 1,
  OPT_ID_HELP,
  OPT_ID_BACKEND,
  OPT_ID_KV_LAYOUT,
  OPT_ID_PHASE,
  OPT_ID_KV_BLOCK_TOKENS,
  OPT_ID_WEIGHT_SIZE_MB,
  OPT_ID_LAYERS,
  OPT_ID_QUERY_HEADS,
  OPT_ID_KV_HEADS,
  OPT_ID_HEAD_DIM,
  OPT_ID_KV_ELEMENT_BYTES,
  OPT_ID_CONTEXT_TOKENS,
  OPT_ID_PROMPT_TOKENS,
  OPT_ID_ATTENTION_QUERY_TILE_TOKENS,
  OPT_ID_BATCH_SIZE,
  OPT_ID_THREADS,
  OPT_ID_ITERATIONS,
  OPT_ID_COUNT,
  OPT_ID_SEED,
  OPT_ID_OUTPUT,
  OPT_ID_SWEEP,
  OPT_ID_SWEEP_MAX_RUNS,
};

constexpr std::array<CliOptionSpec, 23> kLlmOptionSpecs{{
    {OPT_ID_LLM_MEMORY, "-M", "--llm-memory", CliOptionValueKind::None,
     CliOptionArity::Single},
    {OPT_ID_HELP, "-h", "--help", CliOptionValueKind::None,
     CliOptionArity::Single},
    {OPT_ID_BACKEND, nullptr, "--llm-memory-backend",
     CliOptionValueKind::RequiredValue, CliOptionArity::Single},
    {OPT_ID_KV_LAYOUT, nullptr, "--kv-layout",
     CliOptionValueKind::RequiredValue, CliOptionArity::Single},
    {OPT_ID_PHASE, nullptr, "--phase", CliOptionValueKind::RequiredValue,
     CliOptionArity::Single},
    {OPT_ID_KV_BLOCK_TOKENS, nullptr, "--kv-block-tokens",
     CliOptionValueKind::RequiredValue, CliOptionArity::Single},
    {OPT_ID_WEIGHT_SIZE_MB, nullptr, "--weight-size-mb",
     CliOptionValueKind::RequiredValue, CliOptionArity::Single},
    {OPT_ID_LAYERS, nullptr, "--layers", CliOptionValueKind::RequiredValue,
     CliOptionArity::Single},
    {OPT_ID_QUERY_HEADS, nullptr, "--query-heads",
     CliOptionValueKind::RequiredValue, CliOptionArity::Single},
    {OPT_ID_KV_HEADS, nullptr, "--kv-heads", CliOptionValueKind::RequiredValue,
     CliOptionArity::Single},
    {OPT_ID_HEAD_DIM, nullptr, "--head-dim", CliOptionValueKind::RequiredValue,
     CliOptionArity::Single},
    {OPT_ID_KV_ELEMENT_BYTES, nullptr, "--kv-element-bytes",
     CliOptionValueKind::RequiredValue, CliOptionArity::Single},
    {OPT_ID_CONTEXT_TOKENS, nullptr, "--context-tokens",
     CliOptionValueKind::RequiredValue, CliOptionArity::Single},
    {OPT_ID_PROMPT_TOKENS, nullptr, "--prompt-tokens",
     CliOptionValueKind::RequiredValue, CliOptionArity::Single},
    {OPT_ID_ATTENTION_QUERY_TILE_TOKENS, nullptr,
     "--attention-query-tile-tokens", CliOptionValueKind::RequiredValue,
     CliOptionArity::Single},
    {OPT_ID_BATCH_SIZE, nullptr, "--batch-size",
     CliOptionValueKind::RequiredValue, CliOptionArity::Single},
    {OPT_ID_THREADS, "-t", "--threads", CliOptionValueKind::RequiredValue,
     CliOptionArity::Single},
    {OPT_ID_ITERATIONS, "-i", "--iterations", CliOptionValueKind::RequiredValue,
     CliOptionArity::Single},
    {OPT_ID_COUNT, "-r", "--count", CliOptionValueKind::RequiredValue,
     CliOptionArity::Single},
    {OPT_ID_SEED, nullptr, "--seed", CliOptionValueKind::RequiredValue,
     CliOptionArity::Single},
    {OPT_ID_OUTPUT, "-o", "--output", CliOptionValueKind::RequiredValue,
     CliOptionArity::Single},
    {OPT_ID_SWEEP, "-S", "--sweep", CliOptionValueKind::RequiredValue,
     CliOptionArity::Repeatable},
    {OPT_ID_SWEEP_MAX_RUNS, "-X", "--sweep-max-runs",
     CliOptionValueKind::RequiredValue, CliOptionArity::Single},
}};

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

bool parse_nonnegative_size(const std::string& option,
                            const std::string& token, size_t& value) {
  uint64_t parsed = 0;
  const StrictIntegerParseStatus status =
      parse_strict_unsigned_decimal(token, parsed);
  if (status != StrictIntegerParseStatus::Success ||
      parsed > std::numeric_limits<size_t>::max()) {
    const std::string reason =
        status != StrictIntegerParseStatus::Success
            ? strict_unsigned_decimal_error_reason(status)
            : Messages::llm_memory_reason_platform_size_range();
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
    const LlmBackendEvidence& backend_evidence,
    const LlmResultMetadata& metadata,
    const LlmMemoryResult& result) noexcept {
  try {
    return output_session.write_final(
        build_llm_memory_json(config, model_plan, backend_evidence, metadata,
                              result));
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

void initialize_llm_setup_failure_result(
    const LlmMemoryConfig& config, std::string_view reason_code,
    LlmMemoryResult& result) {
  result = LlmMemoryResult{};
  result.status = LlmRunStatus::Failed;
  result.reason_code = canonicalize_llm_result_reason_code(reason_code);
  result.counters.planned_loops = config.loop_count;
  result.snapshot_interval_loops =
      calculate_bounded_snapshot_interval_loops(config.loop_count);
  constexpr std::array<LlmScenario, kLlmScenarioCount> kScenarios = {
      LlmScenario::WeightsOnly, LlmScenario::KvOnly,
      LlmScenario::Mixed};
  for (size_t index = 0; index < kScenarios.size(); ++index) {
    result.aggregates[index].scenario = kScenarios[index];
  }
  if (NumericUtils::checked_multiply(
          config.loop_count, kLlmScenarioCount,
          result.counters.planned_measurements)) {
    result.loops.reserve(config.loop_count);
    result.measurements.reserve(result.counters.planned_measurements);
    for (size_t loop_index = 0; loop_index < config.loop_count; ++loop_index) {
      LlmLoopRecord loop;
      loop.loop_index = loop_index;
      loop.planned_order = build_llm_scenario_order(loop_index);
      for (size_t position = 0; position < kLlmScenarioCount; ++position) {
        LlmMeasurementState measurement;
        measurement.scenario = loop.planned_order[position];
        measurement.status = LlmMeasurementStatus::Failed;
        measurement.reason_code = LlmRunnerReason::NOT_RUN_AFTER_RUNTIME_FAILURE;
        measurement.loop_index = loop_index;
        measurement.order_position = position;
        loop.measurement_indexes[position] = result.measurements.size();
        result.measurements.push_back(std::move(measurement));
      }
      result.loops.push_back(std::move(loop));
    }
    result.counters.terminal_measurements = result.measurements.size();
  }
  result.snapshot_request = "terminal";
  result.terminal_checkpoint_attempted = true;
  result.initialized = true;
}

int write_llm_setup_failure_final(
    JsonOutputSession& output_session, const LlmMemoryConfig& config,
    const LlmMemoryWorkPlan& model_plan,
    const LlmBackendEvidence& backend_evidence,
    LlmResultMetadata& metadata, std::string_view reason_code,
    LlmMemoryResult& result) noexcept {
  try {
    initialize_llm_setup_failure_result(config, reason_code, result);
    if (output_session.kind() == JsonOutputKind::Disabled) return EXIT_SUCCESS;
    metadata.environment_end = capture_llm_host_environment();
    return output_session.write_final(
        build_llm_memory_json(config, model_plan, backend_evidence, metadata,
                              result),
        false);
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

int write_llm_post_run_exception_final(
    JsonOutputSession& output_session, const LlmMemoryConfig& config,
    const LlmMemoryWorkPlan& model_plan,
    const LlmBackendEvidence& backend_evidence,
    LlmResultMetadata& metadata, LlmMemoryResult& result) noexcept {
  try {
    if (result.checkpoint_failed || (output_session.kind() == JsonOutputKind::File &&
        !result.terminal_checkpoint_completed)) return EXIT_FAILURE;
    result.prior_file_writer_attempts = output_session.file_writer_attempts();
    result.prior_successful_file_writes = output_session.successful_file_writes();
    result.snapshot_request = "late-command-error-correction";
    result.terminal_checkpoint_completed = false;
    result.status = LlmRunStatus::Failed;
    result.reason_code = LlmRunnerReason::RUNNER_EXCEPTION;
    result.run_accepted = false;
    result.diagnostic.clear();
    if (output_session.kind() == JsonOutputKind::Disabled) return EXIT_SUCCESS;
    metadata.environment_end = capture_llm_host_environment();
    return output_session.write_final(
        build_llm_memory_json(config, model_plan, backend_evidence, metadata,
                              result),
        false);
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

const char* validate_llm_kv_layout_options(
    const LlmMemoryConfig& config) noexcept {
  if (config.kv_layout == LlmKvLayout::Contiguous) {
    if (config.user_specified_kv_block_tokens ||
        config.kv_block_tokens != 0) {
      return LlmMemoryConfigReason::KV_BLOCK_TOKENS_NOT_APPLICABLE;
    }
    return nullptr;
  }
  if (config.kv_layout != LlmKvLayout::Paged) {
    return LlmMemoryConfigReason::KV_BLOCK_TOKENS_NOT_APPLICABLE;
  }
  if (!config.user_specified_kv_block_tokens) {
    return LlmMemoryConfigReason::KV_BLOCK_TOKENS_REQUIRED;
  }
  if (config.kv_block_tokens == 0) {
    return LlmMemoryConfigReason::KV_BLOCK_TOKENS_ZERO;
  }
  if (config.kv_block_tokens >
      static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    return LlmMemoryConfigReason::KV_BLOCK_TOKENS_EXCEEDS_UINT32;
  }
  if ((config.kv_block_tokens & (config.kv_block_tokens - 1)) != 0) {
    return LlmMemoryConfigReason::KV_BLOCK_TOKENS_NOT_POWER_OF_TWO;
  }
  return nullptr;
}

const char* validate_llm_activated_profile(
    const LlmMemoryConfig& config) noexcept {
  switch (config.backend) {
    case LlmMemoryBackend::Cpu:
      return nullptr;
    case LlmMemoryBackend::Metal:
      if ((config.phase == LlmPhase::Decode &&
           config.kv_layout != LlmKvLayout::Contiguous &&
           config.kv_layout != LlmKvLayout::Paged) ||
          (config.phase == LlmPhase::Prefill &&
           config.kv_layout != LlmKvLayout::Contiguous &&
           config.kv_layout != LlmKvLayout::Paged)) {
        return LlmMemoryConfigReason::INVALID_KV_LAYOUT;
      }
      if (config.phase != LlmPhase::Decode &&
          config.phase != LlmPhase::Prefill) {
        return LlmMemoryConfigReason::INVALID_PHASE;
      }
      return nullptr;
  }
  return LlmMemoryConfigReason::INVALID_BACKEND;
}

std::optional<LlmMetalResourcePlanRequest>
build_runtime_metal_resource_request(
    const LlmMemoryWorkPlan& logical_plan,
    const LlmBackendEvidence& backend_evidence,
    size_t available_memory_bytes, size_t mapping_granularity_bytes,
    size_t additional_owned_bytes) {
  const LlmMetalBackendEvidence* const metal =
      get_llm_metal_backend_evidence(backend_evidence);
  if (backend_evidence.backend != LlmMemoryBackend::Metal ||
      backend_evidence.initialization.status != LlmBackendStatus::Ready ||
      metal == nullptr || !logical_plan.geometry.valid ||
      logical_plan.backend != LlmMemoryBackend::Metal ||
      (logical_plan.phase == LlmPhase::Decode &&
       logical_plan.kv_layout != LlmKvLayout::Contiguous &&
       logical_plan.kv_layout != LlmKvLayout::Paged) ||
      (logical_plan.phase == LlmPhase::Prefill &&
       logical_plan.kv_layout != LlmKvLayout::Contiguous &&
       logical_plan.kv_layout != LlmKvLayout::Paged) ||
      (logical_plan.phase != LlmPhase::Decode &&
       logical_plan.phase != LlmPhase::Prefill)) {
    return std::nullopt;
  }
  LlmMetalResourcePlanRequest request;
  request.geometry = logical_plan.geometry;
  if (logical_plan.kv_layout == LlmKvLayout::Paged) {
    const size_t sequence_tokens =
        logical_plan.phase == LlmPhase::Prefill
            ? logical_plan.geometry.prefill.has_value()
                  ? logical_plan.geometry.prefill->prompt_tokens
                  : 0
            : logical_plan.geometry.decode.has_value()
                  ? logical_plan.geometry.decode->visible_context_tokens
                  : 0;
    if (sequence_tokens == 0) {
      return std::nullopt;
    }
    request.paged_layout = build_llm_kv_layout_plan(
        {sequence_tokens,
         logical_plan.geometry.kv_block_tokens,
         logical_plan.geometry.layer_count,
         logical_plan.geometry.batch_size,
         logical_plan.geometry.k_or_v_record_bytes_per_layer});
    if (!request.paged_layout->valid) {
      return std::nullopt;
    }
  }
  request.argument_buffer_encoded_length =
      metal->capability.argument_buffer_encoded_length;
  request.argument_buffer_alignment =
      metal->capability.argument_buffer_alignment;
  request.max_buffer_length = metal->capability.max_buffer_length;
  request.available_memory_bytes = available_memory_bytes;
  request.host_mapping_granularity_bytes = mapping_granularity_bytes;
  request.additional_owned_bytes = additional_owned_bytes;
  return request;
}

/** Map a raw CLI `--sweep` parameter key to its typed LLM sweep enum. */
bool llm_sweep_parameter_from_string(const std::string& key,
                                     LlmSweepParameter& out_parameter,
                                     std::string& out_name) {
  if (key == "context-tokens") {
    out_parameter = LlmSweepParameter::ContextTokens;
    out_name = "context-tokens";
    return true;
  }
  if (key == "kv-block-tokens") {
    out_parameter = LlmSweepParameter::KvBlockTokens;
    out_name = "kv-block-tokens";
    return true;
  }
  if (key == "prompt-tokens") {
    out_parameter = LlmSweepParameter::PromptTokens;
    out_name = "prompt-tokens";
    return true;
  }
  if (key == "attention-query-tile-tokens") {
    out_parameter = LlmSweepParameter::AttentionQueryTileTokens;
    out_name = "attention-query-tile-tokens";
    return true;
  }
  return false;
}

/**
 * Parse and type-check one `--sweep key=v1,v2,...` specification for LLM mode.
 *
 * Structural syntax and non-empty tokens are checked by `parse_sweep_text`;
 * key admissibility and integer bounds are validated against LLM CLI rules.
 */
bool parse_llm_sweep_spec(const std::string& spec_text,
                          LlmSweepSpec& out_spec) {
  ParsedSweepText parsed_text;
  try {
    parsed_text = parse_sweep_text(spec_text);
  } catch (const std::invalid_argument& error) {
    std::cerr << Messages::error_prefix()
              << Messages::error_invalid_value("--sweep", spec_text,
                                               error.what())
              << std::endl;
    return false;
  }

  LlmSweepSpec spec;
  if (!llm_sweep_parameter_from_string(parsed_text.key, spec.parameter,
                                       spec.parameter_name)) {
    std::cerr << Messages::error_prefix()
              << Messages::error_sweep_parameter_not_allowed(parsed_text.key,
                                                             "--llm-memory")
              << std::endl;
    return false;
  }

  const std::string option_name = "--" + spec.parameter_name;
  spec.values.reserve(parsed_text.values.size());
  for (const std::string& raw_value : parsed_text.values) {
    size_t parsed = 0;
    const bool ok =
        (spec.parameter == LlmSweepParameter::ContextTokens)
            ? parse_positive_size(option_name, raw_value, parsed)
            : parse_nonnegative_size(option_name, raw_value, parsed);
    if (!ok) {
      return false;
    }
    spec.values.push_back(LlmSweepValue{raw_value, parsed});
  }

  out_spec = std::move(spec);
  return true;
}

/** Format the swept parameter values of one expanded run for preflight diagnostics. */
std::string format_llm_sweep_combination(const LlmMemoryConfig& base_config,
                                         const LlmMemoryConfig& run_config) {
  std::string description;
  for (size_t i = 0; i < base_config.sweep_specs.size(); ++i) {
    if (i > 0) {
      description += ", ";
    }
    const LlmSweepSpec& spec = base_config.sweep_specs[i];
    size_t value = 0;
    switch (spec.parameter) {
      case LlmSweepParameter::ContextTokens:
        value = run_config.visible_context_tokens;
        break;
      case LlmSweepParameter::KvBlockTokens:
        value = run_config.kv_block_tokens;
        break;
      case LlmSweepParameter::PromptTokens:
        value = run_config.prompt_tokens;
        break;
      case LlmSweepParameter::AttentionQueryTileTokens:
        value = run_config.attention_query_tile_tokens;
        break;
    }
    description += spec.parameter_name + "=" + std::to_string(value);
  }
  return description;
}

/**
 * Run pure logical, profile, geometry, and explicit-iteration preflight checks
 * for one concrete LLM configuration without allocating resources.
 */
bool preflight_llm_single_config(const LlmMemoryConfig& config) {
  const LlmMemoryConfigValidation validation =
      validate_llm_memory_config(config);
  if (!validation.valid) {
    report_llm_config_failure(validation.reason_code);
    return false;
  }
  if (const char* profile_reason =
          validate_llm_activated_profile(config)) {
    report_llm_config_failure(profile_reason);
    return false;
  }
  LlmGeometryRequest geometry_request{
      validation.active_weight_bytes,
      config.layer_count,
      config.query_head_count,
      config.kv_head_count,
      config.head_dimension,
      config.kv_element_bytes,
      config.visible_context_tokens,
      config.batch_size,
      config.kv_block_tokens,
      config.phase,
      config.kv_layout};
  geometry_request.prompt_tokens = config.prompt_tokens;
  geometry_request.attention_query_tile_tokens =
      config.attention_query_tile_tokens;
  const LlmGeometry geometry = resolve_llm_geometry(geometry_request);
  if (!geometry.valid) {
    report_llm_config_failure(geometry.reason_code);
    return false;
  }

  constexpr std::array<LlmScenario, kLlmScenarioCount> kScenarios = {
      LlmScenario::WeightsOnly, LlmScenario::KvOnly, LlmScenario::Mixed};
  size_t maximum_explicit_iterations =
      std::numeric_limits<size_t>::max();
  for (LlmScenario scenario : kScenarios) {
    const LlmScenarioLimits limits = calculate_llm_scenario_limits(
        geometry, scenario, config.backend);
    if (!limits.valid) {
      report_llm_config_failure(limits.reason_code);
      return false;
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
    return false;
  }
  return true;
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

void set_llm_memory_command_test_hooks(
    const LlmMemoryCommandTestHooks* hooks) {
  if (hooks == nullptr) {
    active_llm_command_hooks = LlmMemoryCommandTestHooks{};
    llm_command_hooks_active = false;
    return;
  }
  active_llm_command_hooks = *hooks;
  llm_command_hooks_active = true;
}

int parse_llm_memory_arguments(int argc, char* argv[],
                               LlmMemoryConfig& config) {
  config = LlmMemoryConfig{};
  if (argc > 0 && argv != nullptr) {
    config.argv.reserve(static_cast<size_t>(argc));
    for (int index = 0; index < argc; ++index) {
      if (argv[index] != nullptr) {
        config.argv.emplace_back(argv[index]);
      }
    }
  }

  const CliParseResult parsed = parse_cli_options(argc, argv, kLlmOptionSpecs);
  if (!parsed.ok) {
    std::cerr << Messages::error_prefix()
              << format_cli_parse_error(
                     parsed, &Messages::error_llm_memory_must_be_used_alone())
              << std::endl;
    return EXIT_FAILURE;
  }

  bool mode_seen = false;
  bool weight_seen = false;
  bool layers_seen = false;
  bool query_heads_seen = false;
  bool kv_heads_seen = false;
  bool head_dimension_seen = false;
  bool context_tokens_seen = false;
  bool prompt_tokens_seen = false;
  bool attention_query_tile_tokens_seen = false;
  bool kv_block_tokens_seen = false;

  for (const CliParsedOption& option : parsed.options) {
    switch (option.id) {
      case OPT_ID_LLM_MEMORY:
        mode_seen = true;
        break;
      case OPT_ID_HELP:
        config.help_printed = true;
        break;
      case OPT_ID_BACKEND:
        if (option.value == "cpu") {
          config.backend = LlmMemoryBackend::Cpu;
        } else if (option.value == "metal") {
          config.backend = LlmMemoryBackend::Metal;
        } else {
          std::cerr << Messages::error_prefix()
                    << Messages::error_invalid_value(
                           option.matched_name, option.value,
                           Messages::llm_memory_reason_backend())
                    << std::endl;
          return EXIT_FAILURE;
        }
        config.user_specified_backend = true;
        break;
      case OPT_ID_KV_LAYOUT:
        if (option.value == "contiguous") {
          config.kv_layout = LlmKvLayout::Contiguous;
        } else if (option.value == "paged") {
          config.kv_layout = LlmKvLayout::Paged;
        } else {
          std::cerr << Messages::error_prefix()
                    << Messages::error_invalid_value(
                           option.matched_name, option.value,
                           Messages::llm_memory_reason_kv_layout())
                    << std::endl;
          return EXIT_FAILURE;
        }
        config.user_specified_kv_layout = true;
        break;
      case OPT_ID_PHASE:
        if (option.value == "decode") {
          config.phase = LlmPhase::Decode;
        } else if (option.value == "prefill") {
          config.phase = LlmPhase::Prefill;
        } else {
          std::cerr << Messages::error_prefix()
                    << Messages::error_invalid_value(
                           option.matched_name, option.value,
                           Messages::llm_memory_reason_phase())
                    << std::endl;
          return EXIT_FAILURE;
        }
        config.user_specified_phase = true;
        break;
      case OPT_ID_KV_BLOCK_TOKENS:
        if (!parse_nonnegative_size(option.matched_name, option.value,
                                    config.kv_block_tokens)) {
          return EXIT_FAILURE;
        }
        config.user_specified_kv_block_tokens = true;
        kv_block_tokens_seen = true;
        break;
      case OPT_ID_WEIGHT_SIZE_MB:
        if (!parse_positive_size(option.matched_name, option.value,
                                 config.weight_size_mb)) {
          return EXIT_FAILURE;
        }
        weight_seen = true;
        break;
      case OPT_ID_LAYERS:
        if (!parse_positive_size(option.matched_name, option.value,
                                 config.layer_count)) {
          return EXIT_FAILURE;
        }
        layers_seen = true;
        break;
      case OPT_ID_QUERY_HEADS:
        if (!parse_positive_size(option.matched_name, option.value,
                                 config.query_head_count)) {
          return EXIT_FAILURE;
        }
        query_heads_seen = true;
        break;
      case OPT_ID_KV_HEADS:
        if (!parse_positive_size(option.matched_name, option.value,
                                 config.kv_head_count)) {
          return EXIT_FAILURE;
        }
        kv_heads_seen = true;
        break;
      case OPT_ID_HEAD_DIM:
        if (!parse_positive_size(option.matched_name, option.value,
                                 config.head_dimension)) {
          return EXIT_FAILURE;
        }
        head_dimension_seen = true;
        break;
      case OPT_ID_KV_ELEMENT_BYTES:
        if (!parse_positive_size(option.matched_name, option.value,
                                 config.kv_element_bytes)) {
          return EXIT_FAILURE;
        }
        if (config.kv_element_bytes != 1 && config.kv_element_bytes != 2 &&
            config.kv_element_bytes != 4) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_invalid_value(
                           option.matched_name, option.value,
                           Messages::llm_memory_reason_kv_element_bytes())
                    << std::endl;
          return EXIT_FAILURE;
        }
        break;
      case OPT_ID_CONTEXT_TOKENS:
        if (!parse_positive_size(option.matched_name, option.value,
                                 config.visible_context_tokens)) {
          return EXIT_FAILURE;
        }
        config.user_specified_context_tokens = true;
        context_tokens_seen = true;
        break;
      case OPT_ID_PROMPT_TOKENS:
        if (!parse_nonnegative_size(option.matched_name, option.value,
                                    config.prompt_tokens)) {
          return EXIT_FAILURE;
        }
        config.user_specified_prompt_tokens = true;
        prompt_tokens_seen = true;
        break;
      case OPT_ID_ATTENTION_QUERY_TILE_TOKENS:
        if (!parse_nonnegative_size(option.matched_name, option.value,
                                    config.attention_query_tile_tokens)) {
          return EXIT_FAILURE;
        }
        config.user_specified_attention_query_tile_tokens = true;
        attention_query_tile_tokens_seen = true;
        break;
      case OPT_ID_BATCH_SIZE:
        if (!parse_positive_size(option.matched_name, option.value,
                                 config.batch_size)) {
          return EXIT_FAILURE;
        }
        break;
      case OPT_ID_THREADS:
        if (!parse_positive_size(option.matched_name, option.value,
                                 config.requested_workers)) {
          return EXIT_FAILURE;
        }
        config.user_specified_workers = true;
        break;
      case OPT_ID_ITERATIONS:
        if (!parse_positive_size(option.matched_name, option.value,
                                 config.iterations)) {
          return EXIT_FAILURE;
        }
        config.user_specified_iterations = true;
        break;
      case OPT_ID_COUNT:
        if (!parse_positive_size(option.matched_name, option.value,
                                 config.loop_count)) {
          return EXIT_FAILURE;
        }
        break;
      case OPT_ID_SEED: {
        const StrictIntegerParseStatus status =
            parse_strict_unsigned_decimal(option.value, config.seed);
        if (status != StrictIntegerParseStatus::Success) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_invalid_value(
                           option.matched_name, option.value,
                           strict_unsigned_decimal_error_reason(status))
                    << std::endl;
          return EXIT_FAILURE;
        }
        config.user_specified_seed = true;
        break;
      }
      case OPT_ID_OUTPUT:
        config.output_file = option.value;
        break;
      case OPT_ID_SWEEP: {
        LlmSweepSpec spec;
        if (!parse_llm_sweep_spec(option.value, spec)) {
          return EXIT_FAILURE;
        }
        for (const LlmSweepSpec& existing : config.sweep_specs) {
          if (existing.parameter == spec.parameter) {
            std::cerr << Messages::error_prefix()
                      << Messages::error_duplicate_sweep_parameter(
                             spec.parameter_name)
                      << std::endl;
            return EXIT_FAILURE;
          }
        }
        config.sweep_specs.push_back(std::move(spec));
        config.run_sweep = true;
        break;
      }
      case OPT_ID_SWEEP_MAX_RUNS:
        if (!parse_positive_size(option.matched_name, option.value,
                                 config.sweep_max_runs)) {
          return EXIT_FAILURE;
        }
        break;
    }
  }

  if (!mode_seen) {
    std::cerr << Messages::error_prefix()
              << Messages::error_llm_memory_must_be_used_alone()
              << std::endl;
    return EXIT_FAILURE;
  }
  if (config.help_printed) {
    print_llm_memory_help(cli_program_name(argc, argv));
    return EXIT_SUCCESS;
  }

  bool sweeps_context_tokens = false;
  bool sweeps_kv_block_tokens = false;
  bool sweeps_prompt_tokens = false;
  bool sweeps_attention_query_tile_tokens = false;
  for (const LlmSweepSpec& spec : config.sweep_specs) {
    switch (spec.parameter) {
      case LlmSweepParameter::ContextTokens:
        sweeps_context_tokens = true;
        break;
      case LlmSweepParameter::KvBlockTokens:
        sweeps_kv_block_tokens = true;
        break;
      case LlmSweepParameter::PromptTokens:
        sweeps_prompt_tokens = true;
        break;
      case LlmSweepParameter::AttentionQueryTileTokens:
        sweeps_attention_query_tile_tokens = true;
        break;
    }
  }

  if (config.run_sweep) {
    if (sweeps_context_tokens && context_tokens_seen) {
      std::cerr << Messages::error_prefix()
                << Messages::error_sweep_option_conflict("--context-tokens",
                                                         "context-tokens")
                << std::endl;
      return EXIT_FAILURE;
    }
    if (sweeps_kv_block_tokens && kv_block_tokens_seen) {
      std::cerr << Messages::error_prefix()
                << Messages::error_sweep_option_conflict("--kv-block-tokens",
                                                         "kv-block-tokens")
                << std::endl;
      return EXIT_FAILURE;
    }
    if (sweeps_prompt_tokens && prompt_tokens_seen) {
      std::cerr << Messages::error_prefix()
                << Messages::error_sweep_option_conflict("--prompt-tokens",
                                                         "prompt-tokens")
                << std::endl;
      return EXIT_FAILURE;
    }
    if (sweeps_attention_query_tile_tokens &&
        attention_query_tile_tokens_seen) {
      std::cerr << Messages::error_prefix()
                << Messages::error_sweep_option_conflict(
                       "--attention-query-tile-tokens",
                       "attention-query-tile-tokens")
                << std::endl;
      return EXIT_FAILURE;
    }
    if (sweeps_context_tokens && config.phase != LlmPhase::Decode) {
      std::cerr << Messages::error_prefix()
                << Messages::error_sweep_parameter_not_allowed(
                       "context-tokens", "--llm-memory --phase prefill")
                << std::endl;
      return EXIT_FAILURE;
    }
    if (sweeps_prompt_tokens && config.phase != LlmPhase::Prefill) {
      std::cerr << Messages::error_prefix()
                << Messages::error_sweep_parameter_not_allowed(
                       "prompt-tokens", "--llm-memory --phase decode")
                << std::endl;
      return EXIT_FAILURE;
    }
    if (sweeps_attention_query_tile_tokens &&
        config.phase != LlmPhase::Prefill) {
      std::cerr << Messages::error_prefix()
                << Messages::error_sweep_parameter_not_allowed(
                       "attention-query-tile-tokens",
                       "--llm-memory --phase decode")
                << std::endl;
      return EXIT_FAILURE;
    }
    if (sweeps_kv_block_tokens && config.kv_layout != LlmKvLayout::Paged) {
      std::cerr << Messages::error_prefix()
                << Messages::error_sweep_parameter_not_allowed(
                       "kv-block-tokens",
                       "--llm-memory --kv-layout contiguous")
                << std::endl;
      return EXIT_FAILURE;
    }
    if (config.output_file.empty()) {
      std::cerr << Messages::error_prefix()
                << Messages::error_sweep_requires_output() << std::endl;
      return EXIT_FAILURE;
    }
    if (config.sweep_max_runs > Constants::LLM_JSON_MAX_SAFE_INTEGER) {
      report_llm_config_failure(
          LlmMemoryConfigReason::JSON_INTEGER_OUT_OF_RANGE);
      return EXIT_FAILURE;
    }
    const size_t run_count = calculate_llm_sweep_run_count(config);
    if (run_count == 0 || run_count > config.sweep_max_runs) {
      std::cerr << Messages::error_prefix()
                << Messages::error_sweep_too_many_runs(run_count,
                                                       config.sweep_max_runs)
                << std::endl;
      return EXIT_FAILURE;
    }
  }

  const std::array<std::pair<bool, const char*>, 5> required_options = {{
      {weight_seen, "--weight-size-mb"},
      {layers_seen, "--layers"},
      {query_heads_seen, "--query-heads"},
      {kv_heads_seen, "--kv-heads"},
      {head_dimension_seen, "--head-dim"},
  }};
  for (const auto& [seen, option] : required_options) {
    if (!seen) {
      std::cerr << Messages::error_prefix()
                << Messages::error_llm_memory_missing_required_option(option)
                << std::endl;
      return EXIT_FAILURE;
    }
  }
  if (config.phase == LlmPhase::Decode && !context_tokens_seen &&
      !sweeps_context_tokens) {
    std::cerr << Messages::error_prefix()
              << Messages::error_llm_memory_missing_required_option(
                     "--context-tokens")
              << std::endl;
    return EXIT_FAILURE;
  }
  if (config.kv_layout == LlmKvLayout::Paged && !kv_block_tokens_seen &&
      !sweeps_kv_block_tokens) {
    std::cerr << Messages::error_prefix()
              << Messages::error_llm_memory_missing_required_option(
                     "--kv-block-tokens")
              << std::endl;
    return EXIT_FAILURE;
  }
  if (!sweeps_kv_block_tokens) {
    if (const char* layout_reason = validate_llm_kv_layout_options(config)) {
      report_llm_config_failure(layout_reason);
      return EXIT_FAILURE;
    }
  }
  if (config.backend == LlmMemoryBackend::Metal &&
      config.user_specified_workers) {
    report_llm_config_failure(
        LlmMemoryConfigReason::THREADS_NOT_APPLICABLE);
    return EXIT_FAILURE;
  }
  if (config.backend == LlmMemoryBackend::Cpu) {
    config.available_workers = detect_available_llm_workers();
    if (config.available_workers == 0) {
      report_llm_config_failure(
          LlmMemoryConfigReason::AVAILABLE_WORKER_COUNT_REQUIRED);
      return EXIT_FAILURE;
    }
    if (!config.user_specified_workers) {
      config.requested_workers = config.available_workers;
    }
  }

  if (config.run_sweep) {
    const std::vector<LlmMemoryConfig> run_configs =
        build_llm_sweep_run_configs(config);
    for (size_t i = 0; i < run_configs.size(); ++i) {
      if (!preflight_llm_single_config(run_configs[i])) {
        std::cerr << Messages::error_prefix()
                  << Messages::error_llm_memory_sweep_combination_invalid(
                         i + 1, run_configs.size(),
                         format_llm_sweep_combination(config, run_configs[i]))
                  << std::endl;
        return EXIT_FAILURE;
      }
    }
  } else {
    if (!preflight_llm_single_config(config)) {
      return EXIT_FAILURE;
    }
  }

  if (!config.user_specified_seed) {
    config.seed = generate_llm_seed();
  }
  return EXIT_SUCCESS;
}

namespace {

/**
 * Execute one concrete LLM memory profile run for direct or sweep-collect mode.
 *
 * When `output_session` is non-null, persists direct file checkpoints or final
 * stdout JSON through the caller's session. When `collected_json` is non-null,
 * populates the in-memory schema-2 payload (or records `failure_reason` /
 * `preparation_interrupted` when stopping before result initialization) and
 * releases all backend resources before returning.
 */
int execute_llm_memory_run(const LlmMemoryConfig& config,
                           JsonOutputSession* output_session,
                           bool install_signal_mask, bool print_banner,
                           nlohmann::ordered_json* collected_json,
                           std::string* failure_reason,
                           bool* preparation_interrupted) noexcept {
  const auto stop_check = []() {
    if (llm_command_hooks_active && active_llm_command_hooks.stop_requested) {
      return active_llm_command_hooks.stop_requested();
    }
    return signal_received();
  };
  const auto record_failure = [&](std::string_view reason) {
    if (failure_reason != nullptr && !reason.empty()) {
      *failure_reason = std::string(reason);
    }
  };
  if (collected_json != nullptr) {
    *collected_json = nlohmann::ordered_json();
  }
  if (failure_reason != nullptr) {
    failure_reason->clear();
  }
  if (preparation_interrupted != nullptr) {
    *preparation_interrupted = false;
  }
  int run_status = EXIT_FAILURE;
  LlmMemoryWorkPlan model_plan;
  std::unique_ptr<LlmBackend> backend;
  LlmResultMetadata metadata;
  LlmMemoryResult result;
  bool post_run_exception = false;
  try {
    if (print_banner) {
      print_runtime_banner();
    }
    const std::string binary_hash = capture_llm_binary_sha256();
    metadata.build_manifest = build_runtime_provenance_manifest_json(binary_hash);
    metadata.timestamp = build_utc_timestamp();
    metadata.main_thread_qos = prepare_main_thread_benchmark_qos(MainThreadQosSetter{}, false);
    std::optional<BenchmarkSignalMaskGuard> signal_guard;
    if (install_signal_mask) {
      signal_guard.emplace();
    }

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

    backend = create_llm_backend(config.backend);
    if (!backend) {
      record_failure(LlmBackendReason::INVALID_BACKEND);
      std::cerr << Messages::error_prefix()
                << Messages::error_llm_memory_run_failed(
                       LlmBackendReason::INVALID_BACKEND)
                << std::endl;
      return EXIT_FAILURE;
    }

    const unsigned long available_memory_mb = get_available_memory_mb();
    if (available_memory_mb == 0) {
      metadata.available_memory_source = "unavailable";
    } else {
      if (!NumericUtils::checked_multiply(static_cast<size_t>(available_memory_mb), Constants::BYTES_PER_MB,
                                          metadata.available_memory_bytes)) {
        record_failure(LlmWorkPlanReason::MEMORY_BUDGET_OVERFLOW);
        std::cerr << Messages::error_prefix()
                  << Messages::error_llm_memory_run_failed(LlmWorkPlanReason::MEMORY_BUDGET_OVERFLOW) << std::endl;
        return EXIT_FAILURE;
      }
    }

    LlmMemoryWorkPlanDraft model_draft = prepare_llm_memory_work_plan(
        config, config.available_workers, metadata.available_memory_bytes,
        metadata.page_size_bytes, stop_check);
    if (!model_draft.valid) {
      record_failure(model_draft.reason_code);
      if (collected_json != nullptr &&
          model_draft.reason_code == LlmKvLayoutReason::PREPARATION_INTERRUPTED) {
        if (preparation_interrupted != nullptr) {
          *preparation_interrupted = true;
        }
        std::cout << Messages::msg_interrupted_by_user() << std::endl;
        return EXIT_SUCCESS;
      }
      std::cerr << Messages::error_prefix()
                << Messages::error_llm_memory_run_failed(
                       model_draft.reason_code)
                << std::endl;
      return EXIT_FAILURE;
    }

    if (config.backend == LlmMemoryBackend::Metal) {
      const LlmBackendLifecycleResult initialization =
          backend->initialize(config);
      if (initialization.status == LlmBackendStatus::Ready) {
        const std::optional<LlmMetalResourcePlanRequest> request =
            build_runtime_metal_resource_request(
                model_draft.candidate, backend->evidence(),
                metadata.available_memory_bytes, metadata.page_size_bytes,
                model_draft.candidate.memory_budget.request
                    .planner_storage_bytes);
        LlmMetalExecutionPlan preflight_execution;
        if (request.has_value()) {
          preflight_execution =
              build_llm_metal_execution_plan(*request);
        } else {
          preflight_execution.reason_code =
              LlmBackendReason::EXECUTION_PLAN_MISMATCH;
        }
        const std::string preflight_reason =
            preflight_execution.reason_code;
        if (!attach_llm_metal_execution_plan(
                model_draft, std::move(preflight_execution))) {
          const std::string effective_reason =
              preflight_reason.empty()
                  ? std::string(LlmBackendReason::EXECUTION_PLAN_MISMATCH)
                  : preflight_reason;
          record_failure(effective_reason);
          std::cerr << Messages::error_prefix()
                    << Messages::error_llm_memory_run_failed(effective_reason)
                    << std::endl;
          return EXIT_FAILURE;
        }
      }
    }

    size_t checksum_auxiliary_bytes = 0;
    size_t orchestration_auxiliary_bytes = 0;
    {
      const LlmBackendAuxiliaryEstimate backend_auxiliary =
          backend->calculate_auxiliary_estimate(
              model_draft.auxiliary_preflight);
      const LlmRunnerAuxiliaryEstimate runner_auxiliary =
          calculate_llm_runner_auxiliary_estimate(
              config, model_draft.auxiliary_preflight);
      metadata.json_peak_estimate =
          calculate_llm_json_peak_estimate(
              config, model_draft.auxiliary_preflight);
      if (!metadata.json_peak_estimate.valid) {
        record_failure(metadata.json_peak_estimate.reason_code);
        std::cerr << Messages::error_prefix()
                  << Messages::error_llm_memory_run_failed(std::string(metadata.json_peak_estimate.reason_code))
                  << std::endl;
        return EXIT_FAILURE;
      }
      size_t runner_and_backend_orchestration_bytes = 0;
      if (!backend_auxiliary.valid || !runner_auxiliary.valid ||
          !NumericUtils::checked_add(backend_auxiliary.checksum_auxiliary_bytes,
                                     runner_auxiliary.checksum_auxiliary_bytes, checksum_auxiliary_bytes) ||
          !NumericUtils::checked_add(backend_auxiliary.orchestration_auxiliary_bytes,
                                     runner_auxiliary.orchestration_auxiliary_bytes,
                                     runner_and_backend_orchestration_bytes)) {
        const std::string reason_code =
            !backend_auxiliary.valid
                ? std::string(backend_auxiliary.reason_code)
                : (!runner_auxiliary.valid ? std::string(runner_auxiliary.reason_code)
                                           : std::string(LlmRunnerReason::AUXILIARY_BYTES_OVERFLOW));
        record_failure(reason_code);
        std::cerr << Messages::error_prefix() << Messages::error_llm_memory_run_failed(reason_code) << std::endl;
        return EXIT_FAILURE;
      }
      if (!NumericUtils::checked_add(runner_and_backend_orchestration_bytes,
                                     metadata.json_peak_estimate.total_bytes,
                                     orchestration_auxiliary_bytes)) {
        record_failure(LlmJsonReason::PEAK_BYTES_OVERFLOW);
        std::cerr << Messages::error_prefix()
                  << Messages::error_llm_memory_run_failed(LlmJsonReason::PEAK_BYTES_OVERFLOW) << std::endl;
        return EXIT_FAILURE;
      }
    }
    if (config.backend == LlmMemoryBackend::Metal &&
        backend->evidence().initialization.status ==
            LlmBackendStatus::Ready) {
      size_t additional_owned_bytes = 0;
      size_t command_auxiliary_bytes = 0;
      if (!NumericUtils::checked_add(
              checksum_auxiliary_bytes, orchestration_auxiliary_bytes,
              command_auxiliary_bytes) ||
          !NumericUtils::checked_add(
              model_draft.candidate.memory_budget.request
                  .planner_storage_bytes,
              command_auxiliary_bytes,
              additional_owned_bytes)) {
        record_failure(LlmRunnerReason::AUXILIARY_BYTES_OVERFLOW);
        std::cerr << Messages::error_prefix()
                  << Messages::error_llm_memory_run_failed(
                         LlmRunnerReason::AUXILIARY_BYTES_OVERFLOW)
                  << std::endl;
        return EXIT_FAILURE;
      }
      const std::optional<LlmMetalResourcePlanRequest> request =
          build_runtime_metal_resource_request(
              model_draft.candidate, backend->evidence(),
              metadata.available_memory_bytes, metadata.page_size_bytes,
              additional_owned_bytes);
      LlmMetalExecutionPlan runtime_execution;
      if (request.has_value()) {
        runtime_execution = build_llm_metal_execution_plan(*request);
      } else {
        runtime_execution.reason_code =
            LlmBackendReason::EXECUTION_PLAN_MISMATCH;
      }
      model_plan = finalize_llm_memory_work_plan(
          std::move(model_draft),
          std::move(runtime_execution),
          checksum_auxiliary_bytes, orchestration_auxiliary_bytes,
          stop_check);
    } else {
      model_plan = finalize_llm_memory_work_plan(
          std::move(model_draft), checksum_auxiliary_bytes,
          orchestration_auxiliary_bytes, stop_check);
    }
    if (!model_plan.valid) {
      record_failure(model_plan.reason_code);
      if (collected_json != nullptr &&
          model_plan.reason_code == LlmKvLayoutReason::PREPARATION_INTERRUPTED) {
        if (preparation_interrupted != nullptr) {
          *preparation_interrupted = true;
        }
        std::cout << Messages::msg_interrupted_by_user() << std::endl;
        return EXIT_SUCCESS;
      }
      if (model_plan.reason_code ==
          LlmWorkPlanReason::BLOCK_TABLE_PROTECTION_FAILED) {
        std::cerr << Messages::error_prefix()
                  << Messages::error_llm_paged_table_protection_failed()
                  << std::endl;
      } else {
        std::cerr << Messages::error_prefix()
                  << Messages::error_llm_memory_run_failed(
                         model_plan.reason_code)
                  << std::endl;
      }
      return EXIT_FAILURE;
    }

    LlmRunnerHooks hooks;
    hooks.stop_requested = stop_check;
    if (output_session != nullptr) {
      hooks.progress_snapshots = output_session->persists_checkpoints();
      hooks.observe_file_writes = [&]() {
        return std::make_pair(output_session->file_writer_attempts(),
                              output_session->successful_file_writes());
      };
      hooks.checkpoint = [&](const LlmMemoryResult& snapshot,
                             LlmCheckpointKind kind) {
        static_cast<void>(kind);
        return output_session->checkpoint([&]() {
          metadata.environment_end = capture_llm_host_environment();
          return build_llm_memory_json(config, model_plan,
                                       backend->evidence(), metadata,
                                       snapshot);
        });
      };
    } else {
      hooks.progress_snapshots = false;
      hooks.observe_file_writes = []() { return std::make_pair(0U, 0U); };
      hooks.checkpoint = [](const LlmMemoryResult&, LlmCheckpointKind) {
        return EXIT_SUCCESS;
      };
    }

    if (llm_command_hooks_active && active_llm_command_hooks.before_runner) {
      active_llm_command_hooks.before_runner(model_plan);
    }
    run_status =
        run_llm_memory_suite(config, model_plan, *backend, result, hooks);
    if (!result.initialized) {
      record_failure(result.reason_code);
      if (result.reason_code == LlmBackendReason::TIMER_UNAVAILABLE) {
        return EXIT_FAILURE;
      }
      std::cerr << Messages::error_prefix()
                << Messages::error_llm_memory_run_failed(result.reason_code)
                << std::endl;
      const std::string reason_code = result.reason_code;
      if (output_session != nullptr) {
        if (write_llm_setup_failure_final(
                *output_session, config, model_plan, backend->evidence(),
                metadata, reason_code, result) != EXIT_SUCCESS) {
          return EXIT_FAILURE;
        }
      } else if (collected_json != nullptr) {
        initialize_llm_setup_failure_result(config, reason_code, result);
        metadata.environment_end = capture_llm_host_environment();
        *collected_json = build_llm_memory_json(
            config, model_plan, backend->evidence(), metadata, result);
      }
      return EXIT_FAILURE;
    }
    if (collected_json != nullptr &&
        result.status == LlmRunStatus::Failed &&
        result.reason_code == LlmBackendReason::PREPARATION_INTERRUPTED) {
      record_failure(result.reason_code);
      if (preparation_interrupted != nullptr) {
        *preparation_interrupted = true;
      }
      std::cout << Messages::msg_interrupted_by_user() << std::endl;
      return EXIT_SUCCESS;
    }
    if (llm_command_hooks_active &&
        active_llm_command_hooks.after_initialized_runner) {
      active_llm_command_hooks.after_initialized_runner();
    }
    if (output_session == nullptr ||
        output_session->kind() != JsonOutputKind::File ||
        !result.terminal_checkpoint_completed) {
      metadata.environment_end = capture_llm_host_environment();
    }
    print_llm_memory_console_report(model_plan, backend->evidence(), metadata, result);

    if (output_session != nullptr &&
        output_session->kind() == JsonOutputKind::File &&
        result.terminal_checkpoint_completed && !result.checkpoint_failed) {
      std::cout << Messages::msg_results_saved_to(config.output_file) << std::endl;
    }
    if (result.status == LlmRunStatus::Interrupted) {
      std::cout << Messages::msg_interrupted_by_user() << std::endl;
    } else if (run_status != EXIT_SUCCESS) {
      record_failure(result.reason_code);
      std::cerr << Messages::error_prefix() << Messages::error_llm_memory_run_failed(result.reason_code) << std::endl;
    }
  } catch (const std::exception& error) {
    record_failure(LlmRunnerReason::RUNNER_EXCEPTION);
    report_llm_command_exception(error.what());
    run_status = EXIT_FAILURE;
    post_run_exception = result.initialized;
  } catch (...) {
    record_failure(LlmRunnerReason::RUNNER_UNKNOWN_EXCEPTION);
    report_llm_command_exception("");
    run_status = EXIT_FAILURE;
    post_run_exception = result.initialized;
  }

  if (model_plan.valid && backend && !result.initialized) {
    if (output_session != nullptr) {
      if (write_llm_setup_failure_final(
              *output_session, config, model_plan, backend->evidence(),
              metadata, LlmRunnerReason::RUNNER_EXCEPTION,
              result) != EXIT_SUCCESS) {
        return EXIT_FAILURE;
      }
    } else if (collected_json != nullptr) {
      try {
        initialize_llm_setup_failure_result(
            config, LlmRunnerReason::RUNNER_EXCEPTION, result);
        metadata.environment_end = capture_llm_host_environment();
        *collected_json = build_llm_memory_json(
            config, model_plan, backend->evidence(), metadata, result);
      } catch (...) {
        *collected_json = nlohmann::ordered_json();
      }
    }
    return EXIT_FAILURE;
  }

  if (post_run_exception && model_plan.valid && backend) {
    if (output_session != nullptr) {
      if (write_llm_post_run_exception_final(
              *output_session, config, model_plan, backend->evidence(),
              metadata, result) != EXIT_SUCCESS) {
        return EXIT_FAILURE;
      }
    } else if (collected_json != nullptr) {
      try {
        result.snapshot_request = "late-command-error-correction";
        result.terminal_checkpoint_completed = false;
        result.status = LlmRunStatus::Failed;
        result.reason_code = LlmRunnerReason::RUNNER_EXCEPTION;
        result.run_accepted = false;
        result.diagnostic.clear();
        metadata.environment_end = capture_llm_host_environment();
        *collected_json = build_llm_memory_json(
            config, model_plan, backend->evidence(), metadata, result);
      } catch (...) {
        *collected_json = nlohmann::ordered_json();
      }
    }
    return EXIT_FAILURE;
  }

  if (collected_json != nullptr && result.initialized && backend) {
    try {
      *collected_json = build_llm_memory_json(
          config, model_plan, backend->evidence(), metadata, result);
    } catch (const std::exception& error) {
      record_failure(LlmRunnerReason::RUNNER_EXCEPTION);
      report_llm_command_exception(error.what());
      *collected_json = nlohmann::ordered_json();
      return EXIT_FAILURE;
    } catch (...) {
      record_failure(LlmRunnerReason::RUNNER_UNKNOWN_EXCEPTION);
      report_llm_command_exception("");
      *collected_json = nlohmann::ordered_json();
      return EXIT_FAILURE;
    }
  }

  if (output_session != nullptr &&
      output_session->kind() == JsonOutputKind::Stdout &&
      result.initialized && backend &&
      write_llm_stdout_final(*output_session, config, model_plan,
                             backend->evidence(), metadata,
                             result) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  return run_status;
}

}  // namespace

int run_llm_memory_collect(const LlmMemoryConfig& config,
                           nlohmann::ordered_json& result_json,
                           std::string* failure_reason,
                           bool* preparation_interrupted) {
  return execute_llm_memory_run(config, /*output_session=*/nullptr,
                                /*install_signal_mask=*/false,
                                /*print_banner=*/false, &result_json,
                                failure_reason, preparation_interrupted);
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

  JsonOutputTarget output_target;
  try {
    output_target = make_json_output_target(
        config.output_file,
        JsonFilePathPolicy::ResolveAgainstCurrentDirectory);
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

  std::optional<JsonOutputSession> output_session;
  try {
    output_session.emplace(std::move(output_target));
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

  if (config.run_sweep) {
    try {
      const SweepExecutionResult execution =
          run_llm_memory_sweep(config, *output_session);
      if (output_session->kind() == JsonOutputKind::Stdout &&
          !execution.output_json.empty() &&
          output_session->write_final(execution.output_json) != EXIT_SUCCESS) {
        return EXIT_FAILURE;
      }
      return execution.exit_code;
    } catch (const std::exception& error) {
      report_llm_command_exception(error.what());
      return EXIT_FAILURE;
    } catch (...) {
      report_llm_command_exception("");
      return EXIT_FAILURE;
    }
  }

  return execute_llm_memory_run(config, &*output_session,
                                /*install_signal_mask=*/true,
                                /*print_banner=*/true,
                                /*collected_json=*/nullptr,
                                /*failure_reason=*/nullptr,
                                /*preparation_interrupted=*/nullptr);
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
  switch (config.phase) {
    case LlmPhase::Decode:
      if (config.visible_context_tokens == 0) {
        validation.reason_code =
            LlmMemoryConfigReason::CONTEXT_TOKENS_REQUIRED;
        return validation;
      }
      if (config.user_specified_prompt_tokens || config.prompt_tokens != 0) {
        validation.reason_code =
            LlmMemoryConfigReason::PROMPT_TOKENS_NOT_APPLICABLE;
        return validation;
      }
      if (config.user_specified_attention_query_tile_tokens ||
          config.attention_query_tile_tokens != 0) {
        validation.reason_code = LlmMemoryConfigReason::
            ATTENTION_QUERY_TILE_TOKENS_NOT_APPLICABLE;
        return validation;
      }
      break;
    case LlmPhase::Prefill:
      if (config.user_specified_context_tokens ||
          config.visible_context_tokens != 0) {
        validation.reason_code =
            LlmMemoryConfigReason::CONTEXT_TOKENS_NOT_APPLICABLE;
        return validation;
      }
      if (config.prompt_tokens == 0) {
        validation.reason_code =
            config.user_specified_prompt_tokens
                ? LlmMemoryConfigReason::PROMPT_TOKENS_MUST_BE_POSITIVE
                : LlmMemoryConfigReason::PROMPT_TOKENS_REQUIRED;
        return validation;
      }
      if (config.attention_query_tile_tokens == 0) {
        validation.reason_code =
            config.user_specified_attention_query_tile_tokens
                ? LlmMemoryConfigReason::
                      ATTENTION_QUERY_TILE_TOKENS_MUST_BE_POSITIVE
                : LlmMemoryConfigReason::
                      ATTENTION_QUERY_TILE_TOKENS_REQUIRED;
        return validation;
      }
      if (config.attention_query_tile_tokens > config.prompt_tokens) {
        validation.reason_code = LlmMemoryConfigReason::
            ATTENTION_QUERY_TILE_TOKENS_EXCEEDS_PROMPT;
        return validation;
      }
      break;
    default:
      validation.reason_code = LlmMemoryConfigReason::INVALID_PHASE;
      return validation;
  }
  if (const char* layout_reason = validate_llm_kv_layout_options(config)) {
    validation.reason_code = layout_reason;
    return validation;
  }
  if (config.batch_size == 0) {
    validation.reason_code = LlmMemoryConfigReason::BATCH_SIZE_REQUIRED;
    return validation;
  }
  switch (config.backend) {
    case LlmMemoryBackend::Cpu:
      if (config.requested_workers == 0) {
        validation.reason_code =
            LlmMemoryConfigReason::WORKER_COUNT_REQUIRED;
        return validation;
      }
      break;
    case LlmMemoryBackend::Metal:
      if (config.user_specified_workers || config.requested_workers != 0 ||
          config.available_workers != 0) {
        validation.reason_code =
            LlmMemoryConfigReason::THREADS_NOT_APPLICABLE;
        return validation;
      }
      break;
    default:
      validation.reason_code = LlmMemoryConfigReason::INVALID_BACKEND;
      return validation;
  }
  if (config.loop_count == 0) {
    validation.reason_code = LlmMemoryConfigReason::LOOP_COUNT_REQUIRED;
    return validation;
  }
  const std::array<size_t, 14> json_integer_fields = {
      config.weight_size_mb,
      config.layer_count,
      config.query_head_count,
      config.kv_head_count,
      config.head_dimension,
      config.visible_context_tokens,
      config.prompt_tokens,
      config.attention_query_tile_tokens,
      config.kv_block_tokens,
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
      (config.run_sweep &&
       config.sweep_max_runs > Constants::LLM_JSON_MAX_SAFE_INTEGER) ||
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

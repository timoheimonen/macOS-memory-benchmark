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
 * @file llm_sweep_runner.cpp
 * @brief Standalone LLM memory-profile parameter sweep runner implementation.
 */

#include "llm_memory/llm_sweep_runner.h"

#include <iostream>
#include <utility>
#include <vector>

#include "benchmark/sweep_runner.h"
#include "core/config/constants.h"
#include "core/config/sweep_utils.h"
#include "core/signal/signal_handler.h"
#include "core/timing/timer.h"
#include "llm_memory/llm_memory.h"
#include "output/console/messages/messages_api.h"
#include "output/console/output_printer.h"
#include "output/json/json_output/json_output_api.h"
#include "output/json/json_output/json_output_session.h"
#include "third_party/nlohmann/json.hpp"

namespace {

struct LlmSweepAssignment {
  const LlmSweepSpec* spec = nullptr;
  const LlmSweepValue* value = nullptr;
};

nlohmann::ordered_json build_sweep_parameters_json(
    const LlmMemoryConfig& config) {
  nlohmann::ordered_json params = nlohmann::ordered_json::object();
  for (const LlmSweepSpec& spec : config.sweep_specs) {
    nlohmann::ordered_json values = nlohmann::ordered_json::array();
    for (const LlmSweepValue& value : spec.values) {
      values.push_back(value.integer_value);
    }
    params[spec.parameter_name] = std::move(values);
  }
  return params;
}

nlohmann::ordered_json build_assignment_json(
    const std::vector<LlmSweepAssignment>& assignments) {
  nlohmann::ordered_json params = nlohmann::ordered_json::object();
  for (const LlmSweepAssignment& assignment : assignments) {
    params[assignment.spec->parameter_name] = assignment.value->integer_value;
  }
  return params;
}

void apply_assignment(LlmMemoryConfig& config,
                      const LlmSweepAssignment& assignment) {
  switch (assignment.spec->parameter) {
    case LlmSweepParameter::ContextTokens:
      config.visible_context_tokens = assignment.value->integer_value;
      config.user_specified_context_tokens = true;
      break;
    case LlmSweepParameter::KvBlockTokens:
      config.kv_block_tokens = assignment.value->integer_value;
      config.user_specified_kv_block_tokens = true;
      break;
    case LlmSweepParameter::PromptTokens:
      config.prompt_tokens = assignment.value->integer_value;
      config.user_specified_prompt_tokens = true;
      break;
    case LlmSweepParameter::AttentionQueryTileTokens:
      config.attention_query_tile_tokens = assignment.value->integer_value;
      config.user_specified_attention_query_tile_tokens = true;
      break;
  }
}

LlmMemoryConfig build_run_config(
    const LlmMemoryConfig& base_config,
    const std::vector<LlmSweepAssignment>& assignments) {
  LlmMemoryConfig run_config = base_config;
  run_config.run_sweep = false;
  run_config.sweep_specs.clear();
  run_config.output_file.clear();
  run_config.collect_json_for_sweep = true;

  for (const LlmSweepAssignment& assignment : assignments) {
    apply_assignment(run_config, assignment);
  }
  return run_config;
}

void append_assignments_recursive(
    const LlmMemoryConfig& config, size_t spec_index,
    std::vector<LlmSweepAssignment>& current,
    std::vector<std::vector<LlmSweepAssignment>>& out_assignments) {
  if (spec_index == config.sweep_specs.size()) {
    out_assignments.push_back(current);
    return;
  }

  const LlmSweepSpec& spec = config.sweep_specs[spec_index];
  for (const LlmSweepValue& value : spec.values) {
    current.push_back(LlmSweepAssignment{&spec, &value});
    append_assignments_recursive(config, spec_index + 1, current,
                                 out_assignments);
    current.pop_back();
  }
}

std::vector<std::vector<LlmSweepAssignment>> build_assignments(
    const LlmMemoryConfig& config) {
  std::vector<std::vector<LlmSweepAssignment>> assignments;
  assignments.reserve(calculate_llm_sweep_run_count(config));
  std::vector<LlmSweepAssignment> current;
  append_assignments_recursive(config, 0, current, assignments);
  return assignments;
}

}  // namespace

size_t calculate_llm_sweep_run_count(const LlmMemoryConfig& config) {
  return calculate_sweep_run_count_from_specs(config.sweep_specs);
}

std::vector<LlmMemoryConfig> build_llm_sweep_run_configs(
    const LlmMemoryConfig& base_config) {
  const std::vector<std::vector<LlmSweepAssignment>> assignments =
      build_assignments(base_config);
  std::vector<LlmMemoryConfig> run_configs;
  run_configs.reserve(assignments.size());
  for (const std::vector<LlmSweepAssignment>& assignment : assignments) {
    run_configs.push_back(build_run_config(base_config, assignment));
  }
  return run_configs;
}

SweepExecutionResult run_llm_memory_sweep(
    const LlmMemoryConfig& base_config,
    JsonOutputSession& output_session) {
  const size_t run_count = calculate_llm_sweep_run_count(base_config);
  print_runtime_banner();
  std::cout << Messages::msg_running_sweep(run_count) << std::endl;

  BenchmarkSignalMaskGuard signal_guard;

  auto total_timer_opt = HighResTimer::create();
  if (!total_timer_opt) {
    std::cerr << Messages::error_prefix()
              << Messages::error_timer_creation_failed() << std::endl;
    return {};
  }
  auto& total_timer = *total_timer_opt;
  total_timer.start();

  nlohmann::ordered_json output_json;
  output_json[JsonKeys::CONFIGURATION] = {
      {JsonKeys::MODE, Constants::SWEEP_JSON_MODE_NAME},
      {"base_mode", Constants::LLM_JSON_MODE_NAME},
      {"run_count", run_count},
      {"sweep_max_runs", base_config.sweep_max_runs},
      {"sweep_parameters", build_sweep_parameters_json(base_config)}};

  const std::vector<std::vector<LlmSweepAssignment>> assignments =
      build_assignments(base_config);
  std::vector<nlohmann::ordered_json> run_parameters;
  run_parameters.reserve(assignments.size());
  for (const std::vector<LlmSweepAssignment>& assignment : assignments) {
    run_parameters.push_back(build_assignment_json(assignment));
  }

  SweepExecutionHooks hooks;
  hooks.execute_run = [&](size_t run_index) {
    std::cout << Messages::msg_sweep_run_progress(run_index + 1,
                                                  assignments.size())
              << std::endl;
    const LlmMemoryConfig run_config =
        build_run_config(base_config, assignments[run_index]);
    SweepRunOutcome outcome;
    outcome.exit_code = run_llm_memory_collect(run_config, outcome.result_json);
    if (outcome.exit_code != EXIT_SUCCESS && outcome.result_json.empty()) {
      outcome.failure_reason = "nested-llm-run-failed";
    }
    return outcome;
  };
  hooks.stop_requested = []() { return signal_received(); };
  hooks.elapsed_seconds = [&]() { return total_timer.stop(); };
  hooks.write_checkpoint = [&](const nlohmann::ordered_json& checkpoint,
                               bool announce_success) {
    return output_session.checkpoint([&checkpoint]() { return checkpoint; },
                                     announce_success);
  };

  const SweepExecutionResult execution = execute_sweep_plan(
      SweepNestedMode::LlmMemory, run_parameters, std::move(output_json),
      hooks);

  if (execution.output_json.value("status", "failed") == "interrupted") {
    const nlohmann::ordered_json& runs = execution.output_json["runs"];
    if (!runs.is_array() || runs.empty() ||
        runs.back().value("status", "failed") == "complete") {
      std::cout << Messages::msg_interrupted_by_user() << std::endl;
    }
  }
  if (execution.exit_code == EXIT_SUCCESS) {
    const double total_elapsed_sec =
        execution.output_json.value(JsonKeys::EXECUTION_TIME_SEC, 0.0);
    std::cout << Messages::msg_done_total_time(total_elapsed_sec) << std::endl;
  }
  return execution;
}

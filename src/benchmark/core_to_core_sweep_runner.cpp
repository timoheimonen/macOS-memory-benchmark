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
 * @file core_to_core_sweep_runner.cpp
 * @brief Core-to-core parameter sweep runner implementation.
 */

#include "benchmark/core_to_core_sweep_runner.h"

#include <filesystem>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

#include "benchmark/core_to_core_latency.h"
#include "core/config/constants.h"
#include "core/config/version.h"
#include "core/signal/signal_handler.h"
#include "core/timing/timer.h"
#include "output/console/messages/messages_api.h"
#include "output/json/json_output/json_output_api.h"
#include "third_party/nlohmann/json.hpp"

namespace {

struct CoreToCoreSweepAssignment {
  const CoreToCoreSweepSpec* spec = nullptr;
  const CoreToCoreSweepValue* value = nullptr;
};

nlohmann::ordered_json build_sweep_parameters_json(const CoreToCoreLatencyConfig& config) {
  nlohmann::ordered_json params;
  for (const CoreToCoreSweepSpec& spec : config.sweep_specs) {
    nlohmann::ordered_json values = nlohmann::ordered_json::array();
    for (const CoreToCoreSweepValue& value : spec.values) {
      values.push_back(value.integer_value);
    }
    params[spec.parameter_name] = values;
  }
  return params;
}

nlohmann::ordered_json build_assignment_json(const std::vector<CoreToCoreSweepAssignment>& assignments) {
  nlohmann::ordered_json params;
  for (const CoreToCoreSweepAssignment& assignment : assignments) {
    params[assignment.spec->parameter_name] = assignment.value->integer_value;
  }
  return params;
}

void apply_assignment(CoreToCoreLatencyConfig& config, const CoreToCoreSweepAssignment& assignment) {
  switch (assignment.spec->parameter) {
    case CoreToCoreSweepParameter::Count:
      config.loop_count = assignment.value->integer_value;
      break;
    case CoreToCoreSweepParameter::LatencySamples:
      config.latency_sample_count = assignment.value->integer_value;
      break;
  }
}

CoreToCoreLatencyConfig build_run_config(const CoreToCoreLatencyConfig& base_config,
                                         const std::vector<CoreToCoreSweepAssignment>& assignments) {
  CoreToCoreLatencyConfig run_config = base_config;
  run_config.run_sweep = false;
  run_config.sweep_specs.clear();
  run_config.output_file.clear();

  for (const CoreToCoreSweepAssignment& assignment : assignments) {
    apply_assignment(run_config, assignment);
  }
  return run_config;
}

void append_assignments_recursive(const CoreToCoreLatencyConfig& config, size_t spec_index,
                                  std::vector<CoreToCoreSweepAssignment>& current,
                                  std::vector<std::vector<CoreToCoreSweepAssignment>>& out_assignments) {
  if (spec_index == config.sweep_specs.size()) {
    out_assignments.push_back(current);
    return;
  }

  const CoreToCoreSweepSpec& spec = config.sweep_specs[spec_index];
  for (const CoreToCoreSweepValue& value : spec.values) {
    current.push_back(CoreToCoreSweepAssignment{&spec, &value});
    append_assignments_recursive(config, spec_index + 1, current, out_assignments);
    current.pop_back();
  }
}

std::vector<std::vector<CoreToCoreSweepAssignment>> build_assignments(const CoreToCoreLatencyConfig& config) {
  std::vector<std::vector<CoreToCoreSweepAssignment>> assignments;
  assignments.reserve(calculate_core_to_core_sweep_run_count(config));
  std::vector<CoreToCoreSweepAssignment> current;
  append_assignments_recursive(config, 0, current, assignments);
  return assignments;
}

}  // namespace

size_t calculate_core_to_core_sweep_run_count(const CoreToCoreLatencyConfig& config) {
  size_t run_count = 1;
  for (const CoreToCoreSweepSpec& spec : config.sweep_specs) {
    if (spec.values.empty()) {
      return 0;
    }
    if (run_count > std::numeric_limits<size_t>::max() / spec.values.size()) {
      return std::numeric_limits<size_t>::max();
    }
    run_count *= spec.values.size();
  }
  return run_count;
}

SweepExecutionResult execute_core_to_core_sweep_plan(const std::vector<nlohmann::ordered_json>& run_parameters,
                                                     nlohmann::ordered_json initial_output,
                                                     const SweepExecutionHooks& hooks) {
  return execute_sweep_plan(SweepNestedMode::CoreToCore, run_parameters, std::move(initial_output), hooks);
}

int run_core_to_core_latency_sweep(const CoreToCoreLatencyConfig& base_config) {
  const size_t run_count = calculate_core_to_core_sweep_run_count(base_config);
  std::cout << Messages::usage_header(SOFTVERSION);
  std::cout << Messages::msg_running_sweep(run_count) << std::endl;

  block_benchmark_signals();

  auto total_timer_opt = HighResTimer::create();
  if (!total_timer_opt) {
    std::cerr << Messages::error_prefix() << Messages::error_timer_creation_failed() << std::endl;
    restore_signal_mask();
    return EXIT_FAILURE;
  }
  auto& total_timer = *total_timer_opt;
  total_timer.start();

  nlohmann::ordered_json output_json;
  output_json[JsonKeys::CONFIGURATION] = {{JsonKeys::MODE, Constants::SWEEP_JSON_MODE_NAME},
                                          {"base_mode", Constants::CORE_TO_CORE_JSON_MODE_NAME},
                                          {"run_count", run_count},
                                          {"sweep_max_runs", base_config.sweep_max_runs},
                                          {"sweep_parameters", build_sweep_parameters_json(base_config)}};

  const std::vector<std::vector<CoreToCoreSweepAssignment>> assignments = build_assignments(base_config);
  std::filesystem::path file_path(base_config.output_file);
  if (file_path.is_relative()) {
    file_path = std::filesystem::current_path() / file_path;
  }

  std::vector<nlohmann::ordered_json> run_parameters;
  run_parameters.reserve(assignments.size());
  for (const std::vector<CoreToCoreSweepAssignment>& assignment : assignments) {
    run_parameters.push_back(build_assignment_json(assignment));
  }

  SweepExecutionHooks hooks;
  hooks.execute_run = [&](size_t run_index) {
    std::cout << Messages::msg_sweep_run_progress(run_index + 1, assignments.size()) << std::endl;
    CoreToCoreLatencyConfig run_config = build_run_config(base_config, assignments[run_index]);
    SweepRunOutcome outcome;
    outcome.exit_code = run_core_to_core_latency_collect(run_config, outcome.result_json);
    if (outcome.exit_code != EXIT_SUCCESS) {
      outcome.failure_reason = "nested-core-to-core-run-failed";
    }
    return outcome;
  };
  hooks.stop_requested = []() { return signal_received(); };
  hooks.elapsed_seconds = [&]() { return total_timer.stop(); };
  hooks.write_checkpoint = [&](const nlohmann::ordered_json& checkpoint, bool announce_success) {
    return write_json_to_file(file_path, checkpoint, announce_success);
  };

  const SweepExecutionResult execution = execute_core_to_core_sweep_plan(run_parameters, std::move(output_json), hooks);

  restore_signal_mask();
  if (execution.output_json.value("status", "failed") == "interrupted") {
    const nlohmann::ordered_json& runs = execution.output_json["runs"];
    if (!runs.is_array() || runs.empty() || runs.back().value("status", "failed") == "complete") {
      std::cout << Messages::msg_interrupted_by_user() << std::endl;
    }
  }
  if (execution.exit_code == EXIT_SUCCESS) {
    const double total_elapsed_sec = execution.output_json.value(JsonKeys::EXECUTION_TIME_SEC, 0.0);
    std::cout << Messages::msg_done_total_time(total_elapsed_sec) << std::endl;
  }
  return execution.exit_code;
}

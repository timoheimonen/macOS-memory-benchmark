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
 * @file cpu_latency_sweep_runner.cpp
 * @brief Cartesian sweep planner, preflight, and schema-1 envelope runner
 */

#include "cpu_latency/cpu_latency_sweep_runner.h"

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "core/config/constants.h"
#include "core/config/sweep_utils.h"
#include "core/signal/signal_handler.h"
#include "core/timing/timer.h"
#include "output/console/messages/messages_api.h"
#include "output/console/output_printer.h"
#include "output/json/json_output/json_output_api.h"
#include "third_party/nlohmann/json.hpp"

namespace {

struct CpuLatencySweepAssignment {
  const CpuLatencySweepSpec* spec = nullptr;
  const CpuLatencySweepValue* value = nullptr;
};

nlohmann::ordered_json sweep_value_to_json(
    CpuLatencySweepParameter parameter,
    const CpuLatencySweepValue& value) {
  if (parameter == CpuLatencySweepParameter::LatencyChainMode) {
    return nlohmann::ordered_json(
        latency_chain_mode_to_string(value.chain_mode));
  }
  return nlohmann::ordered_json(value.integer_value);
}

nlohmann::ordered_json build_sweep_parameters_json(
    const CpuLatencyConfig& config) {
  nlohmann::ordered_json params = nlohmann::ordered_json::object();
  for (const CpuLatencySweepSpec& spec : config.sweep_specs) {
    nlohmann::ordered_json values = nlohmann::ordered_json::array();
    for (const CpuLatencySweepValue& v : spec.values) {
      values.push_back(sweep_value_to_json(spec.parameter, v));
    }
    params[spec.parameter_name] = std::move(values);
  }
  return params;
}

nlohmann::ordered_json build_assignment_json(
    const std::vector<CpuLatencySweepAssignment>& assignments) {
  nlohmann::ordered_json params = nlohmann::ordered_json::object();
  for (const auto& assignment : assignments) {
    params[assignment.spec->parameter_name] =
        sweep_value_to_json(assignment.spec->parameter, *assignment.value);
  }
  return params;
}

void apply_assignment(CpuLatencyConfig& config,
                      const CpuLatencySweepAssignment& assignment) {
  switch (assignment.spec->parameter) {
    case CpuLatencySweepParameter::BufferSizeMb:
      config.buffer_size_mb = assignment.value->integer_value;
      break;
    case CpuLatencySweepParameter::CacheSizeKb:
      config.custom_cache_size_kb = assignment.value->integer_value;
      break;
    case CpuLatencySweepParameter::LatencyTlbLocalityKb:
      config.latency_tlb_locality_kb = assignment.value->integer_value;
      break;
    case CpuLatencySweepParameter::LatencyStrideBytes:
      config.latency_stride_bytes = assignment.value->integer_value;
      break;
    case CpuLatencySweepParameter::LatencyChainMode:
      config.latency_chain_mode = assignment.value->chain_mode;
      break;
  }
}

CpuLatencyConfig build_run_config(
    const CpuLatencyConfig& base_config,
    const std::vector<CpuLatencySweepAssignment>& assignments) {
  CpuLatencyConfig run_config = base_config;
  run_config.run_sweep = false;
  run_config.sweep_specs.clear();
  run_config.output_file.clear();
  run_config.collect_json_for_sweep = true;

  for (const auto& assignment : assignments) {
    apply_assignment(run_config, assignment);
  }
  return run_config;
}

void append_assignments_recursive(
    const CpuLatencyConfig& config,
    size_t spec_index,
    std::vector<CpuLatencySweepAssignment>& current,
    std::vector<std::vector<CpuLatencySweepAssignment>>& out) {
  if (spec_index == config.sweep_specs.size()) {
    out.push_back(current);
    return;
  }
  const auto& spec = config.sweep_specs[spec_index];
  for (const auto& val : spec.values) {
    current.push_back({&spec, &val});
    append_assignments_recursive(config, spec_index + 1, current, out);
    current.pop_back();
  }
}

std::vector<std::vector<CpuLatencySweepAssignment>> build_assignments(
    const CpuLatencyConfig& config) {
  std::vector<std::vector<CpuLatencySweepAssignment>> assignments;
  assignments.reserve(calculate_cpu_latency_sweep_run_count(config));
  std::vector<CpuLatencySweepAssignment> current;
  append_assignments_recursive(config, 0, current, assignments);
  return assignments;
}

bool single_run_has_active_targets(const CpuLatencyConfig& config,
                                   const CpuLatencySystemInfo& sys_info) {
  if (config.buffer_size_mb > 0) {
    return true;
  }
  if (!config.custom_cache_size_kb.has_value()) {
    return sys_info.l1_cache_bytes > 0 || sys_info.l2_cache_bytes > 0;
  }
  return *config.custom_cache_size_kb > 0;
}

}  // namespace

size_t calculate_cpu_latency_sweep_run_count(
    const CpuLatencyConfig& config) noexcept {
  return calculate_sweep_run_count_from_specs(config.sweep_specs);
}

std::vector<CpuLatencyConfig> build_cpu_latency_sweep_run_configs(
    const CpuLatencyConfig& base_config) {
  const auto assignments = build_assignments(base_config);
  std::vector<CpuLatencyConfig> configs;
  configs.reserve(assignments.size());
  for (const auto& assignment : assignments) {
    configs.push_back(build_run_config(base_config, assignment));
  }
  return configs;
}

bool preflight_cpu_latency_sweep(const CpuLatencyConfig& base_config,
                                 const CpuLatencySystemInfo& sys_info) {
  if (!base_config.run_sweep || base_config.sweep_specs.empty()) {
    return false;
  }
  const size_t run_count = calculate_cpu_latency_sweep_run_count(base_config);
  if (run_count == 0 || run_count > base_config.sweep_max_runs) {
    return false;
  }
  const std::vector<CpuLatencyConfig> run_configs =
      build_cpu_latency_sweep_run_configs(base_config);
  for (const auto& rc : run_configs) {
    if (!single_run_has_active_targets(rc, sys_info)) {
      std::cerr << Messages::error_prefix()
                << "Invalid sweep combination: at least one target (--buffer-size > 0 or --cache-size != 0) is required."
                << std::endl;
      return false;
    }
    const CpuLatencyWorkPlan wp = build_cpu_latency_work_plan(rc, sys_info);
    if (!wp.valid) {
      std::cerr << Messages::error_prefix()
                << "Invalid sweep combination (" << wp.reason_code << ")."
                << std::endl;
      return false;
    }
  }
  return true;
}

SweepExecutionResult run_cpu_latency_sweep(
    const CpuLatencyConfig& base_config,
    JsonOutputSession& output_session) {
  const size_t run_count = calculate_cpu_latency_sweep_run_count(base_config);
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
      {"base_mode", kCpuLatencyJsonModeName},
      {"run_count", run_count},
      {"sweep_max_runs", base_config.sweep_max_runs},
      {"sweep_parameters", build_sweep_parameters_json(base_config)}};

  const auto assignments = build_assignments(base_config);
  std::vector<nlohmann::ordered_json> run_parameters;
  run_parameters.reserve(assignments.size());
  for (const auto& assignment : assignments) {
    run_parameters.push_back(build_assignment_json(assignment));
  }

  SweepExecutionHooks hooks;
  hooks.execute_run = [&](size_t run_index) {
    std::cout << Messages::msg_sweep_run_progress(run_index + 1,
                                                  assignments.size())
              << std::endl;
    const CpuLatencyConfig run_config =
        build_run_config(base_config, assignments[run_index]);
    SweepRunOutcome outcome;
    std::string failure_reason;
    bool preparation_interrupted = false;
    outcome.exit_code = run_cpu_latency_collect(
        run_config, outcome.result_json, &failure_reason,
        &preparation_interrupted);
    outcome.interrupted = preparation_interrupted;
    if (preparation_interrupted) {
      outcome.failure_reason = failure_reason.empty()
                                   ? "preparation-interrupted"
                                   : failure_reason;
    } else if (outcome.exit_code != EXIT_SUCCESS &&
               outcome.result_json.empty()) {
      outcome.failure_reason = failure_reason.empty()
                                   ? "nested-cpu-latency-run-failed"
                                   : failure_reason;
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
      SweepNestedMode::CpuLatency, run_parameters, std::move(output_json),
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

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

#include <chrono>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
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

std::string build_utc_timestamp() {
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);
  std::tm utc_time;
  gmtime_r(&time_t, &utc_time);
  std::ostringstream timestamp_str;
  timestamp_str << std::put_time(&utc_time, "%Y-%m-%dT%H:%M:%SZ");
  return timestamp_str.str();
}

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

void append_assignments_recursive(const CoreToCoreLatencyConfig& config,
                                  size_t spec_index,
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
  output_json[JsonKeys::CONFIGURATION] = {
      {JsonKeys::MODE, Constants::SWEEP_JSON_MODE_NAME},
      {"base_mode", Constants::CORE_TO_CORE_JSON_MODE_NAME},
      {"run_count", run_count},
      {"sweep_max_runs", base_config.sweep_max_runs},
      {"sweep_parameters", build_sweep_parameters_json(base_config)}};

  nlohmann::ordered_json runs_json = nlohmann::ordered_json::array();
  const std::vector<std::vector<CoreToCoreSweepAssignment>> assignments = build_assignments(base_config);
  for (size_t i = 0; i < assignments.size(); ++i) {
    if (signal_received()) {
      std::cout << Messages::msg_interrupted_by_user() << std::endl;
      break;
    }

    std::cout << Messages::msg_sweep_run_progress(i + 1, assignments.size()) << std::endl;
    CoreToCoreLatencyConfig run_config = build_run_config(base_config, assignments[i]);
    nlohmann::ordered_json result_json;
    if (run_core_to_core_latency_collect(run_config, result_json) != EXIT_SUCCESS) {
      restore_signal_mask();
      return EXIT_FAILURE;
    }

    nlohmann::ordered_json run_json;
    run_json["index"] = i;
    run_json["parameters"] = build_assignment_json(assignments[i]);
    run_json["result"] = result_json;
    runs_json.push_back(run_json);
  }

  const double total_elapsed_sec = total_timer.stop();
  output_json["runs"] = runs_json;
  output_json[JsonKeys::EXECUTION_TIME_SEC] = total_elapsed_sec;
  output_json[JsonKeys::TIMESTAMP] = build_utc_timestamp();
  output_json[JsonKeys::VERSION] = SOFTVERSION;

  std::filesystem::path file_path(base_config.output_file);
  if (file_path.is_relative()) {
    file_path = std::filesystem::current_path() / file_path;
  }

  restore_signal_mask();
  const int write_result = write_json_to_file(file_path, output_json);
  if (write_result == EXIT_SUCCESS) {
    std::cout << Messages::msg_done_total_time(total_elapsed_sec) << std::endl;
  }
  return write_result;
}

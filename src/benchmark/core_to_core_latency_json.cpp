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
 * @file core_to_core_latency_json.cpp
 * @brief JSON serialization for standalone core-to-core latency mode
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 *
 * Builds mode-specific JSON output for cache-line handoff measurements,
 * scenario metadata, and scheduler-hint status used during each scenario run.
 */

#include "benchmark/core_to_core_latency_json.h"

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <vector>

#include "core/config/version.h"
#include "output/json/json_output.h"
#include "utils/json_utils.h"

namespace {

nlohmann::ordered_json build_thread_hint_json(const ThreadHintStatus& hint_status) {
  nlohmann::ordered_json hint_json;
  hint_json["qos_applied"] = hint_status.qos_applied;
  hint_json["qos_code"] = hint_status.qos_code;
  hint_json["affinity_requested"] = hint_status.affinity_requested;
  hint_json["affinity_applied"] = hint_status.affinity_applied;
  hint_json["affinity_code"] = hint_status.affinity_code;
  hint_json["affinity_tag"] = hint_status.affinity_tag;
  return hint_json;
}

std::string build_utc_timestamp() {
  const auto now = std::chrono::system_clock::now();
  const auto time_t_now = std::chrono::system_clock::to_time_t(now);
  std::tm utc_time;
  gmtime_r(&time_t_now, &utc_time);

  std::ostringstream timestamp_str;
  timestamp_str << std::put_time(&utc_time, "%Y-%m-%dT%H:%M:%SZ");
  return timestamp_str.str();
}

nlohmann::ordered_json build_scenario_json(const CoreToCoreLatencyScenarioResult& scenario_result) {
  nlohmann::ordered_json scenario_json;
  scenario_json["name"] = scenario_result.scenario_name;

  scenario_json["round_trip_ns"][JsonKeys::VALUES] = scenario_result.loop_round_trip_ns;
  if (scenario_result.loop_round_trip_ns.size() > 1) {
    scenario_json["round_trip_ns"][JsonKeys::STATISTICS] =
        calculate_json_statistics(scenario_result.loop_round_trip_ns);
  }

  std::vector<double> one_way_estimate_ns;
  one_way_estimate_ns.reserve(scenario_result.loop_round_trip_ns.size());
  for (double round_trip_ns : scenario_result.loop_round_trip_ns) {
    one_way_estimate_ns.push_back(round_trip_ns * 0.5);
  }
  scenario_json["one_way_estimate_ns"][JsonKeys::VALUES] = one_way_estimate_ns;
  if (one_way_estimate_ns.size() > 1) {
    scenario_json["one_way_estimate_ns"][JsonKeys::STATISTICS] =
        calculate_json_statistics(one_way_estimate_ns);
  }

  scenario_json[JsonKeys::SAMPLES_NS][JsonKeys::VALUES] = scenario_result.sample_round_trip_ns;
  if (scenario_result.sample_round_trip_ns.size() > 1) {
    scenario_json[JsonKeys::SAMPLES_NS][JsonKeys::STATISTICS] =
        calculate_json_statistics(scenario_result.sample_round_trip_ns);
  }

  scenario_json["thread_hints"] = {
      {"initiator", build_thread_hint_json(scenario_result.initiator_hint)},
      {"responder", build_thread_hint_json(scenario_result.responder_hint)},
  };

  return scenario_json;
}

}  // namespace

int save_core_to_core_latency_to_json(const CoreToCoreLatencyJsonContext& context) {
  if (context.config.output_file.empty()) {
    return EXIT_SUCCESS;
  }

  nlohmann::ordered_json json_output;
  json_output[JsonKeys::CONFIGURATION] = {
      {"mode", Constants::CORE_TO_CORE_JSON_MODE_NAME},
      {JsonKeys::CPU_NAME, context.cpu_name},
      {JsonKeys::PERFORMANCE_CORES, context.perf_cores},
      {JsonKeys::EFFICIENCY_CORES, context.eff_cores},
      {JsonKeys::LOOP_COUNT, context.config.loop_count},
      {JsonKeys::LATENCY_SAMPLE_COUNT, context.config.latency_sample_count},
      {"warmup_round_trips", context.warmup_round_trips},
      {"headline_round_trips", context.headline_round_trips},
      {"sample_window_round_trips", context.sample_window_round_trips},
  };
  json_output[JsonKeys::EXECUTION_TIME_SEC] = context.total_execution_time_sec;

  nlohmann::ordered_json scenario_array = nlohmann::ordered_json::array();
  for (const CoreToCoreLatencyScenarioResult& scenario_result : context.scenario_results) {
    scenario_array.push_back(build_scenario_json(scenario_result));
  }

  json_output["core_to_core_latency"] = {
      {"scenarios", scenario_array},
      {"hard_pinning_supported", Constants::CORE_TO_CORE_JSON_HARD_PINNING_SUPPORTED},
      {"affinity_tags_are_hints", Constants::CORE_TO_CORE_JSON_AFFINITY_TAGS_ARE_HINTS},
  };

  json_output[JsonKeys::TIMESTAMP] = build_utc_timestamp();
  json_output[JsonKeys::VERSION] = SOFTVERSION;

  std::filesystem::path file_path(context.config.output_file);
  if (file_path.is_relative()) {
    file_path = std::filesystem::current_path() / file_path;
  }

  return write_json_to_file(file_path, json_output);
}

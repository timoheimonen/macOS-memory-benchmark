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

#include <cstdlib>
#include <filesystem>
#include <vector>

#include "core/config/version.h"
#include "output/json/json_output/json_output_api.h"
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

nlohmann::ordered_json build_work_plan_json(const CoreToCoreWorkPlan& work_plan) {
  return {
      {"automatic_calibration", work_plan.calibrated},
      {"calibration_excluded_from_results", true},
      {"calibration_round_trips", work_plan.calibration_round_trips},
      {"calibration_elapsed_seconds", work_plan.calibrated
                                          ? nlohmann::ordered_json(work_plan.calibration_elapsed_seconds)
                                          : nlohmann::ordered_json(nullptr)},
      {"calibration_round_trip_ns", work_plan.calibrated ? nlohmann::ordered_json(work_plan.calibration_round_trip_ns)
                                                         : nlohmann::ordered_json(nullptr)},
      {"warmup_round_trips", work_plan.warmup_round_trips},
      {"headline_round_trips", work_plan.headline_round_trips},
      {"sample_window_round_trips", work_plan.sample_window_round_trips},
  };
}

nlohmann::ordered_json build_loop_record_json(const CoreToCoreLoopRecord& record) {
  nlohmann::ordered_json loop_json;
  loop_json["loop_index"] = record.loop_index;
  loop_json["order_position"] = record.order_position;
  loop_json["status"] = core_to_core_measurement_status_to_string(record.status);
  loop_json["status_reason"] =
      record.status_reason.empty() ? nlohmann::ordered_json(nullptr) : nlohmann::ordered_json(record.status_reason);
  loop_json["round_trip_ns"] = record.round_trip_ns.has_value() ? nlohmann::ordered_json(*record.round_trip_ns)
                                                                : nlohmann::ordered_json(nullptr);
  loop_json["one_way_estimate_ns"] = record.round_trip_ns.has_value()
                                         ? nlohmann::ordered_json(*record.round_trip_ns * 0.5)
                                         : nlohmann::ordered_json(nullptr);
  loop_json["headline_elapsed_seconds"] = record.headline_elapsed_seconds.has_value()
                                              ? nlohmann::ordered_json(*record.headline_elapsed_seconds)
                                              : nlohmann::ordered_json(nullptr);
  loop_json["duration_quality"] = record.duration_quality.empty() ? nlohmann::ordered_json(nullptr)
                                                                  : nlohmann::ordered_json(record.duration_quality);
  loop_json["sample_window_range"] = {
      {"start_index", record.sample_start_index},
      {"count", record.completed_sample_windows},
  };
  loop_json["thread_hints"] = {
      {"initiator", build_thread_hint_json(record.initiator_hint)},
      {"responder", build_thread_hint_json(record.responder_hint)},
  };
  return loop_json;
}

nlohmann::ordered_json build_scenario_json(const CoreToCoreLatencyScenarioResult& scenario_result) {
  nlohmann::ordered_json scenario_json;
  scenario_json["name"] = scenario_result.scenario_name;
  scenario_json["status"] = core_to_core_measurement_status_to_string(scenario_result.status);
  scenario_json["status_reason"] = scenario_result.status_reason.empty()
                                       ? nlohmann::ordered_json(nullptr)
                                       : nlohmann::ordered_json(scenario_result.status_reason);
  scenario_json["planned_loops"] = scenario_result.planned_loops;
  scenario_json["completed_loops"] = scenario_result.completed_loops;
  scenario_json["work_plan"] = build_work_plan_json(scenario_result.work_plan);

  scenario_json["round_trip_ns"][JsonKeys::VALUES] = scenario_result.loop_round_trip_ns;
  if (!scenario_result.loop_round_trip_ns.empty()) {
    const nlohmann::ordered_json headline_statistics = calculate_json_statistics(scenario_result.loop_round_trip_ns);
    if (scenario_result.loop_round_trip_ns.size() > 1) {
      scenario_json["round_trip_ns"][JsonKeys::STATISTICS] = headline_statistics;
    }
    scenario_json["headline_round_trip_ns"] = headline_statistics["median"];
    scenario_json["headline_statistic"] = "median-p50-across-completed-continuous-loops";
  } else {
    scenario_json["headline_round_trip_ns"] = nullptr;
    scenario_json["headline_statistic"] = "median-p50-across-completed-continuous-loops";
  }

  std::vector<double> one_way_estimate_ns;
  one_way_estimate_ns.reserve(scenario_result.loop_round_trip_ns.size());
  for (double round_trip_ns : scenario_result.loop_round_trip_ns) {
    one_way_estimate_ns.push_back(round_trip_ns * 0.5);
  }
  scenario_json["one_way_estimate_ns"][JsonKeys::VALUES] = one_way_estimate_ns;
  if (one_way_estimate_ns.size() > 1) {
    scenario_json["one_way_estimate_ns"][JsonKeys::STATISTICS] = calculate_json_statistics(one_way_estimate_ns);
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

  nlohmann::ordered_json loop_records = nlohmann::ordered_json::array();
  for (const CoreToCoreLoopRecord& record : scenario_result.loop_records) {
    loop_records.push_back(build_loop_record_json(record));
  }
  scenario_json["loop_records"] = loop_records;

  return scenario_json;
}

bool affinity_hint_comparison_interpretable(const std::vector<CoreToCoreLatencyScenarioResult>& scenario_results) {
  bool observed_requested_hint = false;
  for (const CoreToCoreLatencyScenarioResult& scenario : scenario_results) {
    for (const CoreToCoreLoopRecord& record : scenario.loop_records) {
      if (record.status != CoreToCoreMeasurementStatus::Measured) {
        continue;
      }
      if (!record.initiator_hint.affinity_requested && !record.responder_hint.affinity_requested) {
        continue;
      }
      observed_requested_hint = true;
      if (!record.initiator_hint.affinity_applied || !record.responder_hint.affinity_applied) {
        return false;
      }
    }
  }
  return observed_requested_hint;
}

}  // namespace

nlohmann::ordered_json build_core_to_core_latency_json(const CoreToCoreLatencyJsonContext& context) {
  nlohmann::ordered_json json_output;
  json_output[JsonKeys::CONFIGURATION] = {
      {JsonKeys::MODE, Constants::CORE_TO_CORE_JSON_MODE_NAME},
      {"schema_version", Constants::CORE_TO_CORE_JSON_SCHEMA_VERSION},
      {"methodology_version", Constants::CORE_TO_CORE_METHODOLOGY_VERSION},
      {JsonKeys::CPU_NAME, context.cpu_name},
      {JsonKeys::PERFORMANCE_CORES, context.perf_cores},
      {JsonKeys::EFFICIENCY_CORES, context.eff_cores},
      {JsonKeys::LOOP_COUNT, context.config.loop_count},
      {JsonKeys::LATENCY_SAMPLE_COUNT, context.config.latency_sample_count},
      {"minimum_warmup_round_trips", context.warmup_round_trips},
      {"minimum_headline_round_trips", context.headline_round_trips},
      {"minimum_sample_window_round_trips", context.sample_window_round_trips},
      {"calibration_round_trips", Constants::CORE_TO_CORE_CALIBRATION_ROUND_TRIPS},
      {"calibration_warmup_round_trips", Constants::CORE_TO_CORE_CALIBRATION_WARMUP_ROUND_TRIPS},
      {"warmup_target_seconds", Constants::CORE_TO_CORE_WARMUP_TARGET_SECONDS},
      {"headline_target_seconds", Constants::CORE_TO_CORE_HEADLINE_TARGET_SECONDS},
      {"headline_duration_window_seconds",
       {{"minimum", Constants::CORE_TO_CORE_HEADLINE_MIN_SECONDS},
        {"maximum", Constants::CORE_TO_CORE_HEADLINE_MAX_SECONDS}}},
      {"sample_window_target_seconds", Constants::CORE_TO_CORE_SAMPLE_TARGET_SECONDS},
      {"scenario_schedule", "cyclic-latin-square-across-count-loops"},
      {"headline_aggregate", "median-p50"},
      {"repeatability_cv_warning_pct", Constants::CORE_TO_CORE_CV_WARNING_PCT},
  };
  json_output[JsonKeys::EXECUTION_TIME_SEC] = context.total_execution_time_sec;

  nlohmann::ordered_json scenario_array = nlohmann::ordered_json::array();
  for (const CoreToCoreLatencyScenarioResult& scenario_result : context.scenario_results) {
    scenario_array.push_back(build_scenario_json(scenario_result));
  }

  json_output["core_to_core_latency"] = {
      {"status", context.status},
      {"planned_measurements", context.planned_measurements},
      {"completed_measurements", context.completed_measurements},
      {"measurements_complete",
       context.status == "complete" && context.completed_measurements == context.planned_measurements},
      {"scenarios", scenario_array},
      {"hard_pinning_supported", Constants::CORE_TO_CORE_JSON_HARD_PINNING_SUPPORTED},
      {"affinity_tags_are_hints", Constants::CORE_TO_CORE_JSON_AFFINITY_TAGS_ARE_HINTS},
      {"affinity_hint_comparison_interpretable",
       context.status == "complete" && affinity_hint_comparison_interpretable(context.scenario_results)},
  };

  json_output[JsonKeys::TIMESTAMP] = build_utc_timestamp();
  json_output[JsonKeys::VERSION] = SOFTVERSION;

  return json_output;
}

int save_core_to_core_latency_to_json(const CoreToCoreLatencyJsonContext& context) {
  if (context.config.output_file.empty()) {
    return EXIT_SUCCESS;
  }

  nlohmann::ordered_json json_output = build_core_to_core_latency_json(context);

  std::filesystem::path file_path(context.config.output_file);
  if (file_path.is_relative()) {
    file_path = std::filesystem::current_path() / file_path;
  }

  return write_json_to_file(file_path, json_output);
}

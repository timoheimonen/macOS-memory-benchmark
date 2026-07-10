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
 * @file core_to_core_latency.h
 * @brief Standalone core-to-core cache-line handoff benchmark mode interfaces
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 */

#ifndef CORE_TO_CORE_LATENCY_H
#define CORE_TO_CORE_LATENCY_H

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "core/config/constants.h"
#include "third_party/nlohmann/json.hpp"

enum class CoreToCoreSweepParameter {
  Count = 0,
  LatencySamples,
};

struct CoreToCoreSweepValue {
  std::string raw_value;
  int integer_value = 0;
};

struct CoreToCoreSweepSpec {
  CoreToCoreSweepParameter parameter = CoreToCoreSweepParameter::Count;
  std::string parameter_name;
  std::vector<CoreToCoreSweepValue> values;
};

struct CoreToCoreLatencyConfig {
  int loop_count = Constants::CORE_TO_CORE_DEFAULT_LOOP_COUNT;
  int latency_sample_count = Constants::CORE_TO_CORE_DEFAULT_LATENCY_SAMPLE_COUNT;
  std::string output_file;
  bool help_requested = false;
  bool run_sweep = false;
  size_t sweep_max_runs = Constants::DEFAULT_SWEEP_MAX_RUNS;
  std::vector<CoreToCoreSweepSpec> sweep_specs;
};

struct ThreadHintStatus {
  bool qos_applied = false;
  int qos_code = 0;
  bool affinity_requested = false;
  bool affinity_applied = false;
  int affinity_code = 0;
  int affinity_tag = 0;
};

enum class CoreToCoreMeasurementStatus {
  NotRun = 0,
  Measured,
  Interrupted,
  Invalid,
  Failed,
};

inline const char* core_to_core_measurement_status_to_string(CoreToCoreMeasurementStatus status) {
  switch (status) {
    case CoreToCoreMeasurementStatus::NotRun:
      return "not-run";
    case CoreToCoreMeasurementStatus::Measured:
      return "measured";
    case CoreToCoreMeasurementStatus::Interrupted:
      return "interrupted";
    case CoreToCoreMeasurementStatus::Invalid:
      return "invalid";
    case CoreToCoreMeasurementStatus::Failed:
      return "failed";
  }
  return "unknown";
}

struct CoreToCoreWorkPlan {
  bool calibrated = false;
  size_t calibration_round_trips = 0;
  double calibration_elapsed_seconds = 0.0;
  double calibration_round_trip_ns = 0.0;
  size_t warmup_round_trips = 0;
  size_t headline_round_trips = 0;
  size_t sample_window_round_trips = 0;
};

struct CoreToCoreLoopRecord {
  size_t loop_index = 0;
  size_t order_position = 0;
  CoreToCoreMeasurementStatus status = CoreToCoreMeasurementStatus::NotRun;
  std::string status_reason;
  std::optional<double> round_trip_ns;
  std::optional<double> headline_elapsed_seconds;
  std::string duration_quality;
  size_t sample_start_index = 0;
  size_t completed_sample_windows = 0;
  ThreadHintStatus initiator_hint;
  ThreadHintStatus responder_hint;
};

struct CoreToCoreLatencyScenarioResult {
  std::string scenario_name;
  std::vector<double> loop_round_trip_ns;
  std::vector<double> sample_round_trip_ns;
  ThreadHintStatus initiator_hint;
  ThreadHintStatus responder_hint;
  CoreToCoreWorkPlan work_plan;
  std::vector<CoreToCoreLoopRecord> loop_records;
  CoreToCoreMeasurementStatus status = CoreToCoreMeasurementStatus::NotRun;
  std::string status_reason;
  size_t planned_loops = 0;
  size_t completed_loops = 0;
};

/**
 * @brief Parse CLI args for standalone core-to-core mode.
 * @param argc Argument count.
 * @param argv Argument vector.
 * @param config Output mode configuration.
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on parse/validation error.
 */
int parse_core_to_core_mode_arguments(int argc, char* argv[], CoreToCoreLatencyConfig& config);

/**
 * @brief Run standalone core-to-core cache-line handoff benchmark.
 * @param config Parsed mode configuration.
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on runtime/IO error.
 */
int run_core_to_core_latency(const CoreToCoreLatencyConfig& config);

/**
 * @brief Run standalone core-to-core benchmark and return its JSON payload in memory.
 * @param config Parsed mode configuration.
 * @param[out] result_json JSON payload with the normal core-to-core schema.
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on runtime error.
 */
int run_core_to_core_latency_collect(const CoreToCoreLatencyConfig& config, nlohmann::ordered_json& result_json);

/**
 * @brief Parse and run standalone core-to-core mode from main().
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on error.
 */
int run_core_to_core_latency_mode(int argc, char* argv[]);

#endif  // CORE_TO_CORE_LATENCY_H

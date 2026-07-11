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
 * @file core_to_core_latency_internal.h
 * @brief Internal runner interfaces for standalone core-to-core latency mode
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 */

#ifndef CORE_TO_CORE_LATENCY_INTERNAL_H
#define CORE_TO_CORE_LATENCY_INTERNAL_H

#include <string>
#include <vector>

#include "benchmark/core_to_core_latency.h"
#include "utils/descriptive_statistics.h"

struct ScenarioDescriptor {
  std::string name;
  bool apply_affinity = false;
  int initiator_affinity_tag = 0;
  int responder_affinity_tag = 0;
};

struct ScenarioMeasurement {
  CoreToCoreMeasurementStatus status = CoreToCoreMeasurementStatus::NotRun;
  std::string status_reason;
  double round_trip_ns = 0.0;
  double headline_elapsed_seconds = 0.0;
  std::string duration_quality;
  std::vector<double> samples_ns;
  ThreadHintStatus initiator_hint;
  ThreadHintStatus responder_hint;
};

struct CoreToCoreFailureInjection {
  bool fail_timer_creation = false;
  bool fail_responder_startup = false;
  bool fail_initiator_startup = false;
};

/** Shared summary values used by core-to-core console reporting. */
using CoreToCoreSummaryStats = DescriptiveStatistics;

CoreToCoreSummaryStats calculate_core_to_core_summary_stats(const std::vector<double>& values);

std::string classify_core_to_core_duration_quality(double elapsed_seconds);

size_t calculate_core_to_core_calibrated_round_trips(double pilot_elapsed_seconds, size_t pilot_round_trips,
                                                     double target_duration_seconds, size_t minimum_round_trips,
                                                     size_t maximum_round_trips);

std::vector<size_t> build_core_to_core_scenario_order(size_t scenario_count, size_t loop_index);

bool build_core_to_core_work_plan(double pilot_elapsed_seconds, CoreToCoreWorkPlan& out_plan);

bool execute_single_scenario(const ScenarioDescriptor& scenario, const CoreToCoreWorkPlan& work_plan, int sample_count,
                             ScenarioMeasurement& out_measurement,
                             const CoreToCoreFailureInjection* failure_injection = nullptr);

#endif  // CORE_TO_CORE_LATENCY_INTERNAL_H

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
 * @file core_to_core_latency_json.h
 * @brief JSON serialization helpers for standalone core-to-core latency mode
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 */

#ifndef CORE_TO_CORE_LATENCY_JSON_H
#define CORE_TO_CORE_LATENCY_JSON_H

#include <cstddef>
#include <string>
#include <vector>

#include "benchmark/core_to_core_latency.h"
#include "third_party/nlohmann/json.hpp"

/**
 * Borrowed inputs and terminal counters for one core-to-core schema-v2 build.
 *
 * The referenced configuration, CPU name, and scenario vector must outlive the
 * context and remain immutable while it is passed to the builder. The context
 * does not own or copy those inputs.
 */
struct CoreToCoreLatencyJsonContext {
  const CoreToCoreLatencyConfig& config;
  const std::string& cpu_name;
  int perf_cores;
  int eff_cores;
  size_t warmup_round_trips;
  size_t headline_round_trips;
  size_t sample_window_round_trips;
  const std::vector<CoreToCoreLatencyScenarioResult>& scenario_results;
  double total_execution_time_sec;
  std::string status = "complete";
  size_t planned_measurements = 0;
  size_t completed_measurements = 0;
};

/**
 * Build one core-to-core schema-v2 document without performing I/O.
 *
 * @param context Immutable borrowed inputs, terminal status, and counters.
 * @return An ordered document containing configuration, execution time,
 *         scenario evidence, completion metadata, timestamp, and version.
 * @throws std::exception If timestamp, string, vector, or JSON construction
 *         fails.
 * @pre Every reference held by @p context is valid and immutable for the call;
 *      its status and counters describe the supplied scenario results.
 * @note The builder retains no references. It is safe to call concurrently
 *       when each caller keeps the referenced inputs immutable.
 */
nlohmann::ordered_json build_core_to_core_latency_json(
    const CoreToCoreLatencyJsonContext& context);

#endif  // CORE_TO_CORE_LATENCY_JSON_H

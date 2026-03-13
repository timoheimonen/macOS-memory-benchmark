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
};

int save_core_to_core_latency_to_json(const CoreToCoreLatencyJsonContext& context);

#endif  // CORE_TO_CORE_LATENCY_JSON_H

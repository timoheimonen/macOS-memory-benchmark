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
 * @file core_to_core_latency_runner_internal.h
 * @brief Internal runner interfaces for standalone core-to-core latency mode
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 */

#ifndef CORE_TO_CORE_LATENCY_RUNNER_INTERNAL_H
#define CORE_TO_CORE_LATENCY_RUNNER_INTERNAL_H

#include <string>
#include <vector>

#include "benchmark/core_to_core_latency.h"

struct ScenarioDescriptor {
  std::string name;
  bool apply_affinity = false;
  int initiator_affinity_tag = 0;
  int responder_affinity_tag = 0;
};

struct ScenarioMeasurement {
  double round_trip_ns = 0.0;
  std::vector<double> samples_ns;
  ThreadHintStatus initiator_hint;
  ThreadHintStatus responder_hint;
};

bool execute_single_scenario(const ScenarioDescriptor& scenario,
                             int sample_count,
                             ScenarioMeasurement& out_measurement);

#endif  // CORE_TO_CORE_LATENCY_RUNNER_INTERNAL_H

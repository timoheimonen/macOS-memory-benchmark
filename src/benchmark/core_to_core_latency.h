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

#include <string>
#include <vector>

#include "core/config/constants.h"

struct CoreToCoreLatencyConfig {
  int loop_count = Constants::CORE_TO_CORE_DEFAULT_LOOP_COUNT;
  int latency_sample_count = Constants::CORE_TO_CORE_DEFAULT_LATENCY_SAMPLE_COUNT;
  std::string output_file;
  bool help_requested = false;
};

struct ThreadHintStatus {
  bool qos_applied = false;
  int qos_code = 0;
  bool affinity_requested = false;
  bool affinity_applied = false;
  int affinity_code = 0;
  int affinity_tag = 0;
};

struct CoreToCoreLatencyScenarioResult {
  std::string scenario_name;
  std::vector<double> loop_round_trip_ns;
  std::vector<double> sample_round_trip_ns;
  ThreadHintStatus initiator_hint;
  ThreadHintStatus responder_hint;
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
 * @brief Parse and run standalone core-to-core mode from main().
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on error.
 */
int run_core_to_core_latency_mode(int argc, char* argv[]);

#endif  // CORE_TO_CORE_LATENCY_H

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
 * @file core_to_core_messages.cpp
 * @brief Message helpers for standalone core-to-core benchmark mode
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 *
 * Provides centralized user-facing strings for core-to-core mode so benchmark
 * logic can format output without hardcoded text literals.
 */

#include "messages.h"

#include <iomanip>
#include <sstream>

#include "core/config/constants.h"

namespace Messages {

const std::string& error_analyze_core_to_core_must_be_used_alone() {
  static const std::string msg =
      "-analyze-core2core allows only optional -output <file>, -count <count>, and "
      "-latency-samples <count> (no other options allowed)";
  return msg;
}

const std::string& error_core_to_core_timer_creation_failed() {
  static const std::string msg = "Failed to create timer for core-to-core analysis";
  return msg;
}

const std::string& msg_running_core_to_core_analysis() {
  static const std::string msg = "\nRunning standalone core-to-core latency analysis...";
  return msg;
}

std::string msg_core_to_core_scenario_progress(size_t current_loop,
                                               size_t total_loops,
                                               const std::string& scenario_name) {
  std::ostringstream oss;
  oss << "  [Loop " << current_loop << "/" << total_loops << "] Scenario: " << scenario_name;
  return oss.str();
}

const std::string& report_core_to_core_header() {
  static const std::string msg = "--- Core-to-Core Cache-Line Handoff Report ---";
  return msg;
}

const std::string& report_core_to_core_scheduler_note() {
  static const std::string msg =
      "Scheduler note: macOS user-space cannot hard-pin threads to exact core IDs; "
      "QoS/affinity are best-effort hints.";
  return msg;
}

std::string report_core_to_core_cpu(const std::string& cpu_name) {
  if (cpu_name.empty()) {
    return "CPU: Unknown";
  }
  return "CPU: " + cpu_name;
}

std::string report_core_to_core_cores(int perf_cores, int eff_cores) {
  std::ostringstream oss;
  oss << "Detected Cores: " << perf_cores << " P, " << eff_cores << " E";
  return oss.str();
}

std::string report_core_to_core_loop_config(int loop_count,
                                            int sample_count,
                                            size_t headline_round_trips,
                                            size_t sample_window_round_trips) {
  std::ostringstream oss;
  oss << "Config: loops=" << loop_count << ", samples/loop=" << sample_count
      << ", headline_round_trips=" << headline_round_trips
      << ", sample_window_round_trips=" << sample_window_round_trips;
  return oss.str();
}

std::string report_core_to_core_scenario_title(const std::string& scenario_name) {
  return "[Scenario] " + scenario_name;
}

std::string report_core_to_core_round_trip(double round_trip_ns) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(Constants::LATENCY_PRECISION);
  oss << "  Round-trip latency: " << round_trip_ns << " ns";
  return oss.str();
}

std::string report_core_to_core_one_way_estimate(double one_way_ns) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(Constants::LATENCY_PRECISION);
  oss << "  One-way estimate: " << one_way_ns << " ns";
  return oss.str();
}

std::string report_core_to_core_samples(size_t sample_count) {
  std::ostringstream oss;
  oss << "  Samples: " << sample_count;
  return oss.str();
}

std::string report_core_to_core_hint_status(const std::string& thread_role,
                                            bool qos_applied,
                                            int qos_code,
                                            bool affinity_requested,
                                            bool affinity_applied,
                                            int affinity_code,
                                            int affinity_tag) {
  std::ostringstream oss;
  oss << "  " << thread_role
      << " hints: qos=" << (qos_applied ? "ok" : "failed(" + std::to_string(qos_code) + ")");
  if (affinity_requested) {
    oss << ", affinity(tag=" << affinity_tag << ")="
        << (affinity_applied ? "ok" : "failed(" + std::to_string(affinity_code) + ")");
  } else {
    oss << ", affinity=not requested";
  }
  return oss.str();
}

}  // namespace Messages

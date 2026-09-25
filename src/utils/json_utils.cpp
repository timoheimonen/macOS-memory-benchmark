// Copyright 2025 Timo Heimonen <timo.heimonen@proton.me>
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
// This file uses the nlohmann/json library for JSON parsing and generation.
// Library: https://github.com/nlohmann/json
// License: MIT License
//

/**
 * @file json_utils.cpp
 * @brief JSON utility function implementations
 *
 * Provides implementations for JSON-related utility functions including
 * statistical calculations and UTC timestamp formatting.
 */

#include "json_utils.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

#include "utils/descriptive_statistics.h"

std::string build_utc_timestamp(
    std::chrono::system_clock::time_point time_point) {
  const std::time_t time = std::chrono::system_clock::to_time_t(time_point);
  std::tm utc_time{};
  gmtime_r(&time, &utc_time);

  std::ostringstream timestamp;
  timestamp << std::put_time(&utc_time, "%Y-%m-%dT%H:%M:%SZ");
  return timestamp.str();
}

// Calculate the canonical descriptive-statistics object declared in json_utils.h.
nlohmann::json calculate_json_statistics(const std::vector<double>& values) {
  if (values.empty()) {
    return nullptr;
  }

  const DescriptiveStatistics statistics =
      calculate_descriptive_statistics(values);
  nlohmann::json json_statistics = nlohmann::json::object();
  json_statistics["average"] = statistics.average;
  json_statistics["min"] = statistics.min;
  json_statistics["max"] = statistics.max;
  json_statistics["median"] = statistics.median;
  json_statistics["p90"] = statistics.p90;
  json_statistics["p95"] = statistics.p95;
  json_statistics["p99"] = statistics.p99;
  json_statistics["stddev"] = statistics.stddev;
  if (statistics.coefficient_of_variation_defined) {
    json_statistics["coefficient_of_variation_pct"] =
        statistics.coefficient_of_variation_pct;
  } else {
    json_statistics["coefficient_of_variation_pct"] = nullptr;
  }
  json_statistics["median_absolute_deviation"] =
      statistics.median_absolute_deviation;

  return json_statistics;
}

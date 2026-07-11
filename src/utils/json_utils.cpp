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
 * statistical calculations, JSON parsing from strings and files with
 * comprehensive error handling and validation.
 */

#include "json_utils.h"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
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

// Calculate statistics (average, min, max, percentiles, stddev) from a vector of values
// Returns a JSON object containing the calculated statistics
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

// Parse JSON from a string with validation
// Returns true on success, false on error
// On error, error_message is populated with a descriptive error message
bool parse_json_from_string(const std::string& json_string, nlohmann::json& result, std::string& error_message) {
  error_message.clear();
  if (json_string.empty()) {
    error_message = "Empty JSON string";
    return false;
  }
  
  try {
    result = nlohmann::json::parse(json_string);
    return true;
  } catch (const nlohmann::json::parse_error& e) {
    std::ostringstream oss;
    oss << "JSON parse error at position " << e.byte << ": " << e.what();
    error_message = oss.str();
    return false;
  } catch (const nlohmann::json::exception& e) {
    std::ostringstream oss;
    oss << "JSON exception: " << e.what();
    error_message = oss.str();
    return false;
  } catch (const std::exception& e) {
    std::ostringstream oss;
    oss << "Unexpected error during JSON parsing: " << e.what();
    error_message = oss.str();
    return false;
  }
}

// Parse JSON from a file with validation
// Returns true on success, false on error
// On error, error_message is populated with a descriptive error message
bool parse_json_from_file(const std::string& file_path, nlohmann::json& result, std::string& error_message) {
  error_message.clear();
  std::filesystem::path path(file_path);
  
  // Check if file exists
  if (!std::filesystem::exists(path)) {
    error_message = "File does not exist: " + file_path;
    return false;
  }
  
  // Check if it's a regular file
  if (!std::filesystem::is_regular_file(path)) {
    error_message = "Path is not a regular file: " + file_path;
    return false;
  }
  
  // Try to open the file
  std::ifstream file(path);
  if (!file.is_open()) {
    std::ostringstream oss;
    oss << "Failed to open file: " << file_path;
    error_message = oss.str();
    return false;
  }
  
  // Read file content
  std::ostringstream content;
  content << file.rdbuf();
  file.close();
  
  if (content.str().empty()) {
    error_message = "File is empty: " + file_path;
    return false;
  }
  
  // Parse JSON from file content
  return parse_json_from_string(content.str(), result, error_message);
}

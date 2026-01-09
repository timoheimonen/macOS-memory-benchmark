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
#include <algorithm>  // Required for std::sort
#include <cmath>      // Required for std::sqrt
#include <fstream>    // Required for std::ifstream
#include <filesystem> // Required for std::filesystem
#include <sstream>    // Required for std::ostringstream

// Calculate statistics (average, min, max, percentiles, stddev) from a vector of values
// Returns a JSON object containing the calculated statistics
nlohmann::json calculate_json_statistics(const std::vector<double>& values) {
  nlohmann::json stats;
  if (values.empty()) {
    stats["average"] = 0.0;
    stats["min"] = 0.0;
    stats["max"] = 0.0;
    stats["median"] = 0.0;
    stats["p90"] = 0.0;
    stats["p95"] = 0.0;
    stats["p99"] = 0.0;
    stats["stddev"] = 0.0;
    return stats;
  }

  // Calculate basic statistics
  double sum = 0.0;
  for (double v : values) sum += v;
  double avg = sum / values.size();
  
  std::vector<double> sorted = values;
  std::sort(sorted.begin(), sorted.end());
  size_t n = sorted.size();
  
  auto percentile = [&sorted, n](double p) -> double {
    if (n == 0) return 0.0;
    if (n == 1) return sorted[0];
    double index = p * (n - 1);
    size_t lower = static_cast<size_t>(index);
    size_t upper = lower + 1;
    if (upper >= n) return sorted[n - 1];
    double weight = index - lower;
    return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
  };
  
  double variance = 0.0;
  for (double v : values) variance += (v - avg) * (v - avg);
  // Use n-1 for sample standard deviation (Bessel's correction)
  // Handle n == 1 case to avoid division by zero (stddev is 0 for single value)
  double stddev = (n > 1) ? std::sqrt(variance / (n - 1)) : 0.0;
  
  stats["average"] = avg;
  stats["min"] = sorted[0];
  stats["max"] = sorted[n - 1];
  stats["median"] = percentile(0.50);
  stats["p90"] = percentile(0.90);
  stats["p95"] = percentile(0.95);
  stats["p99"] = percentile(0.99);
  stats["stddev"] = stddev;
  
  return stats;
}

// Parse JSON from a string with validation
// Returns true on success, false on error
// On error, error_message is populated with a descriptive error message
bool parse_json_from_string(const std::string& json_string, nlohmann::json& result, std::string& error_message) {
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


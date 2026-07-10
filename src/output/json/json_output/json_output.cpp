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
/**
 * @file json_output.cpp
 * @brief Main JSON output public API implementation
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2025
 *
 * This file implements the public API for saving benchmark results to JSON
 * files. Coordinates the building of JSON structures from configuration,
 * statistics, and metadata, then writes to the specified output file.
 *
 * Public functions:
 * - save_results_to_json(): Save main memory and cache benchmark results
 * - save_pattern_results_to_json(): Save pattern benchmark results
 *
 * Features:
 * - ISO 8601 UTC timestamp generation
 * - Ordered JSON output for readability
 * - Path resolution (relative/absolute)
 * - Version information inclusion
 */
// This file uses the nlohmann/json library for JSON parsing and generation.
// Library: https://github.com/nlohmann/json
// License: MIT License
//
#include <sstream>    // Required for std::ostringstream
#include <filesystem> // Required for std::filesystem::path
#include <chrono>     // Required for std::chrono
#include <ctime>      // Required for std::time, std::localtime, std::strftime
#include <iomanip>    // Required for std::put_time

#include "output/json/json_output/json_output_api.h"
#include "core/config/version.h"  // SOFTVERSION
#include "core/config/constants.h"
#include "core/config/config.h"     // For BenchmarkConfig
#include "benchmark/benchmark_runner.h"  // For BenchmarkStatistics
#include "pattern_benchmark/pattern_benchmark.h" // For PatternStatistics
#include "third_party/nlohmann/json.hpp"   // JSON library

namespace {

std::string build_utc_timestamp() {
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);
  std::tm utc_time;
  gmtime_r(&time_t, &utc_time);
  std::ostringstream timestamp_str;
  timestamp_str << std::put_time(&utc_time, "%Y-%m-%dT%H:%M:%SZ");
  return timestamp_str.str();
}

}  // namespace

nlohmann::ordered_json build_results_json(const BenchmarkConfig& config,
                                          const BenchmarkStatistics& stats,
                                          double total_execution_time_sec) {
  nlohmann::ordered_json json_output;

  // Add fields in the correct order - nlohmann::json preserves insertion order
  // Add configuration (first)
  json_output[JsonKeys::CONFIGURATION] =
      build_config_json(config, Constants::BENCHMARK_JSON_MODE_NAME);
  
  // Add execution time (second)
  json_output[JsonKeys::EXECUTION_TIME_SEC] = total_execution_time_sec;
  
  // Add completion, per-loop measurements, and aggregate schema-v2 sections.
  add_standard_benchmark_results(json_output, config, stats);
  
  // Add timestamp (fifth)
  json_output[JsonKeys::TIMESTAMP] = build_utc_timestamp();
  
  // Add version (last)
  json_output[JsonKeys::VERSION] = SOFTVERSION;

  return json_output;
}

// Save benchmark results to JSON file
// Returns EXIT_SUCCESS on success, EXIT_FAILURE on error
int save_results_to_json(const BenchmarkConfig& config,
                         const BenchmarkStatistics& stats,
                         double total_execution_time_sec,
                         bool announce_success) {
  if (config.output_file.empty()) {
    return EXIT_SUCCESS;  // No output file specified, nothing to do
  }

  nlohmann::ordered_json json_output = build_results_json(config, stats, total_execution_time_sec);
  
  // Resolve file path (handle relative paths)
  std::filesystem::path file_path(config.output_file);
  if (file_path.is_relative()) {
    // Relative path - use current working directory
    file_path = std::filesystem::current_path() / file_path;
  }
  
  // Write JSON to file
  return write_json_to_file(file_path, json_output, announce_success);
}

nlohmann::ordered_json build_pattern_results_json(const BenchmarkConfig& config,
                                                  const PatternStatistics& stats,
                                                  double total_execution_time_sec) {
  nlohmann::ordered_json json_output;

  // Add fields in the correct order - nlohmann::json preserves insertion order
  // Add configuration (first)
  json_output[JsonKeys::CONFIGURATION] =
      build_config_json(config, Constants::PATTERNS_JSON_MODE_NAME);
  
  // Add execution time (second)
  json_output[JsonKeys::EXECUTION_TIME_SEC] = total_execution_time_sec;
  
  // Add patterns results (third)
  const nlohmann::json patterns_json = build_patterns_json(stats);
  if (!patterns_json.is_null()) {
    json_output[JsonKeys::PATTERNS] = patterns_json;
  }
  
  // Add timestamp (fourth)
  json_output[JsonKeys::TIMESTAMP] = build_utc_timestamp();
  
  // Add version (last)
  json_output[JsonKeys::VERSION] = SOFTVERSION;

  return json_output;
}

// Save pattern benchmark results to JSON file
// Returns EXIT_SUCCESS on success, EXIT_FAILURE on error
int save_pattern_results_to_json(const BenchmarkConfig& config, const PatternStatistics& stats, double total_execution_time_sec) {
  if (config.output_file.empty()) {
    return EXIT_SUCCESS;  // No output file specified, nothing to do
  }

  nlohmann::ordered_json json_output = build_pattern_results_json(config, stats, total_execution_time_sec);
  
  // Resolve file path (handle relative paths)
  std::filesystem::path file_path(config.output_file);
  if (file_path.is_relative()) {
    // Relative path - use current working directory
    file_path = std::filesystem::current_path() / file_path;
  }
  
  // Write JSON to file
  return write_json_to_file(file_path, json_output);
}

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
#include <sstream>    // Required for std::ostringstream
#include <filesystem> // Required for std::filesystem::path
#include <chrono>     // Required for std::chrono
#include <ctime>      // Required for std::time, std::localtime, std::strftime
#include <iomanip>    // Required for std::put_time

#include "output/json/json_output.h"
#include "utils/benchmark.h"  // Include common definitions/constants (e.g., SOFTVERSION)
#include "core/config/config.h"     // For BenchmarkConfig
#include "benchmark/benchmark_runner.h"  // For BenchmarkStatistics
#include "pattern_benchmark/pattern_benchmark.h" // For PatternResults
#include "third_party/nlohmann/json.hpp"   // JSON library

// Save benchmark results to JSON file
// Returns EXIT_SUCCESS on success, EXIT_FAILURE on error
int save_results_to_json(const BenchmarkConfig& config, const BenchmarkStatistics& stats, double total_execution_time_sec) {
  if (config.output_file.empty()) {
    return EXIT_SUCCESS;  // No output file specified, nothing to do
  }
  
  nlohmann::json json_output;
  
  // Add version
  std::ostringstream version_str;
  version_str << SOFTVERSION;
  json_output[JsonKeys::VERSION] = version_str.str();
  
  // Add timestamp (ISO 8601 UTC)
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);
  std::tm utc_time;
  gmtime_r(&time_t, &utc_time);
  std::ostringstream timestamp_str;
  timestamp_str << std::put_time(&utc_time, "%Y-%m-%dT%H:%M:%SZ");
  json_output[JsonKeys::TIMESTAMP] = timestamp_str.str();
  
  // Add configuration
  json_output[JsonKeys::CONFIGURATION] = build_config_json(config);
  
  // Add main memory results
  json_output[JsonKeys::MAIN_MEMORY] = build_main_memory_json(stats);
  
  // Add cache results
  json_output[JsonKeys::CACHE] = build_cache_json(config, stats);
  
  // Add execution time
  json_output[JsonKeys::EXECUTION_TIME_SEC] = total_execution_time_sec;
  
  // Resolve file path (handle relative paths)
  std::filesystem::path file_path(config.output_file);
  if (file_path.is_relative()) {
    // Relative path - use current working directory
    file_path = std::filesystem::current_path() / file_path;
  }
  
  // Write JSON to file
  return write_json_to_file(file_path, json_output);
}

// Save pattern benchmark results to JSON file
// Returns EXIT_SUCCESS on success, EXIT_FAILURE on error
int save_pattern_results_to_json(const BenchmarkConfig& config, const PatternResults& results, double total_execution_time_sec) {
  if (config.output_file.empty()) {
    return EXIT_SUCCESS;  // No output file specified, nothing to do
  }
  
  nlohmann::json json_output;
  
  // Add version
  std::ostringstream version_str;
  version_str << SOFTVERSION;
  json_output[JsonKeys::VERSION] = version_str.str();
  
  // Add timestamp (ISO 8601 UTC)
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);
  std::tm utc_time;
  gmtime_r(&time_t, &utc_time);
  std::ostringstream timestamp_str;
  timestamp_str << std::put_time(&utc_time, "%Y-%m-%dT%H:%M:%SZ");
  json_output[JsonKeys::TIMESTAMP] = timestamp_str.str();
  
  // Add configuration
  json_output[JsonKeys::CONFIGURATION] = build_config_json(config);
  
  // Add patterns results
  json_output[JsonKeys::PATTERNS] = build_patterns_json(results);
  
  // Add execution time
  json_output[JsonKeys::EXECUTION_TIME_SEC] = total_execution_time_sec;
  
  // Resolve file path (handle relative paths)
  std::filesystem::path file_path(config.output_file);
  if (file_path.is_relative()) {
    // Relative path - use current working directory
    file_path = std::filesystem::current_path() / file_path;
  }
  
  // Write JSON to file
  return write_json_to_file(file_path, json_output);
}


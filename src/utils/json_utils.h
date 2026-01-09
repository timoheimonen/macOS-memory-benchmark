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
 * @file json_utils.h
 * @brief JSON utility functions for parsing, generation, and statistical calculations
 *
 * This file provides utility functions for JSON operations used throughout the benchmark application:
 * - Statistical calculations (average, median, percentiles, standard deviation) with JSON output
 * - Validated JSON parsing from strings and files with comprehensive error handling
 * - JSON generation helpers for structured benchmark result output
 *
 * These utilities wrap the nlohmann/json library to provide consistent error handling
 * and statistical analysis for benchmark results exported to JSON format.
 *
 * @note Uses nlohmann/json library (MIT License): https://github.com/nlohmann/json
 */

#ifndef JSON_UTILS_H
#define JSON_UTILS_H

#include <vector>
#include <string>
#include "third_party/nlohmann/json.hpp"

/**
 * @brief Calculate statistical measures from a dataset and return as JSON
 *
 * Computes comprehensive statistics including average, min, max, median (p50),
 * percentiles (p90, p95, p99), and standard deviation.
 *
 * @param values Vector of numerical values to analyze
 * @return JSON object containing all calculated statistics
 *
 * @note Returns empty JSON object if input vector is empty
 */
nlohmann::json calculate_json_statistics(const std::vector<double>& values);

/**
 * @brief Parse JSON from a string with validation
 *
 * @param json_string Input string containing JSON data
 * @param result Output parameter to store parsed JSON object
 * @param error_message Output parameter for error description on failure
 * @return true on successful parsing, false on error
 *
 * @note On error, error_message contains detailed parsing error information
 */
bool parse_json_from_string(const std::string& json_string, nlohmann::json& result, std::string& error_message);

/**
 * @brief Parse JSON from a file with validation
 *
 * @param file_path Path to the JSON file to parse
 * @param result Output parameter to store parsed JSON object
 * @param error_message Output parameter for error description on failure
 * @return true on successful parsing, false on error
 *
 * @note On error, error_message contains detailed file I/O or parsing error information
 */
bool parse_json_from_file(const std::string& file_path, nlohmann::json& result, std::string& error_message);

#endif // JSON_UTILS_H


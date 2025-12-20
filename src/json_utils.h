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
#ifndef JSON_UTILS_H
#define JSON_UTILS_H

#include <vector>
#include <string>
#include "nlohmann/json.hpp"

// Calculate statistics (average, min, max, percentiles, stddev) from a vector of values
// Returns a JSON object containing the calculated statistics
nlohmann::json calculate_json_statistics(const std::vector<double>& values);

// Parse JSON from a string with validation
// Returns true on success, false on error
// On error, error_message is populated with a descriptive error message
bool parse_json_from_string(const std::string& json_string, nlohmann::json& result, std::string& error_message);

// Parse JSON from a file with validation
// Returns true on success, false on error
// On error, error_message is populated with a descriptive error message
bool parse_json_from_file(const std::string& file_path, nlohmann::json& result, std::string& error_message);

#endif // JSON_UTILS_H


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
#include "json_output.h"
#include "pattern_benchmark/pattern_benchmark.h" // For PatternResults
#include "nlohmann/json.hpp"   // JSON library

// Build patterns JSON object from PatternResults
nlohmann::json build_patterns_json(const PatternResults& results) {
  nlohmann::json patterns;
  
  // Sequential Forward (baseline)
  patterns[JsonKeys::SEQUENTIAL_FORWARD] = nlohmann::json::object();
  patterns[JsonKeys::SEQUENTIAL_FORWARD][JsonKeys::READ_GB_S] = results.forward_read_bw;
  patterns[JsonKeys::SEQUENTIAL_FORWARD][JsonKeys::WRITE_GB_S] = results.forward_write_bw;
  patterns[JsonKeys::SEQUENTIAL_FORWARD][JsonKeys::COPY_GB_S] = results.forward_copy_bw;
  
  // Sequential Reverse
  patterns[JsonKeys::SEQUENTIAL_REVERSE] = nlohmann::json::object();
  patterns[JsonKeys::SEQUENTIAL_REVERSE][JsonKeys::READ_GB_S] = results.reverse_read_bw;
  patterns[JsonKeys::SEQUENTIAL_REVERSE][JsonKeys::WRITE_GB_S] = results.reverse_write_bw;
  patterns[JsonKeys::SEQUENTIAL_REVERSE][JsonKeys::COPY_GB_S] = results.reverse_copy_bw;
  
  // Strided (Cache Line - 64B)
  patterns[JsonKeys::STRIDED_64] = nlohmann::json::object();
  patterns[JsonKeys::STRIDED_64][JsonKeys::READ_GB_S] = results.strided_64_read_bw;
  patterns[JsonKeys::STRIDED_64][JsonKeys::WRITE_GB_S] = results.strided_64_write_bw;
  patterns[JsonKeys::STRIDED_64][JsonKeys::COPY_GB_S] = results.strided_64_copy_bw;
  
  // Strided (Page - 4096B)
  patterns[JsonKeys::STRIDED_4096] = nlohmann::json::object();
  patterns[JsonKeys::STRIDED_4096][JsonKeys::READ_GB_S] = results.strided_4096_read_bw;
  patterns[JsonKeys::STRIDED_4096][JsonKeys::WRITE_GB_S] = results.strided_4096_write_bw;
  patterns[JsonKeys::STRIDED_4096][JsonKeys::COPY_GB_S] = results.strided_4096_copy_bw;
  
  // Random Uniform
  patterns[JsonKeys::RANDOM] = nlohmann::json::object();
  patterns[JsonKeys::RANDOM][JsonKeys::READ_GB_S] = results.random_read_bw;
  patterns[JsonKeys::RANDOM][JsonKeys::WRITE_GB_S] = results.random_write_bw;
  patterns[JsonKeys::RANDOM][JsonKeys::COPY_GB_S] = results.random_copy_bw;
  
  return patterns;
}


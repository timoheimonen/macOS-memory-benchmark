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
#include "output/json/json_output.h"
#include "pattern_benchmark/pattern_benchmark.h" // For PatternStatistics
#include "third_party/nlohmann/json.hpp"   // JSON library

// Build patterns JSON object from PatternStatistics
nlohmann::json build_patterns_json(const PatternStatistics& stats) {
  nlohmann::json patterns;
  
  // Sequential Forward (baseline)
  if (!stats.all_forward_read_bw.empty()) {
    patterns[JsonKeys::SEQUENTIAL_FORWARD] = nlohmann::json::object();
    add_bandwidth_results(patterns[JsonKeys::SEQUENTIAL_FORWARD],
                          stats.all_forward_read_bw,
                          stats.all_forward_write_bw,
                          stats.all_forward_copy_bw);
  }
  
  // Sequential Reverse
  if (!stats.all_reverse_read_bw.empty()) {
    patterns[JsonKeys::SEQUENTIAL_REVERSE] = nlohmann::json::object();
    add_bandwidth_results(patterns[JsonKeys::SEQUENTIAL_REVERSE],
                          stats.all_reverse_read_bw,
                          stats.all_reverse_write_bw,
                          stats.all_reverse_copy_bw);
  }
  
  // Strided (Cache Line - 64B)
  if (!stats.all_strided_64_read_bw.empty()) {
    patterns[JsonKeys::STRIDED_64] = nlohmann::json::object();
    add_bandwidth_results(patterns[JsonKeys::STRIDED_64],
                          stats.all_strided_64_read_bw,
                          stats.all_strided_64_write_bw,
                          stats.all_strided_64_copy_bw);
  }
  
  // Strided (Page - 4096B)
  if (!stats.all_strided_4096_read_bw.empty()) {
    patterns[JsonKeys::STRIDED_4096] = nlohmann::json::object();
    add_bandwidth_results(patterns[JsonKeys::STRIDED_4096],
                          stats.all_strided_4096_read_bw,
                          stats.all_strided_4096_write_bw,
                          stats.all_strided_4096_copy_bw);
  }
  
  // Random Uniform
  if (!stats.all_random_read_bw.empty()) {
    patterns[JsonKeys::RANDOM] = nlohmann::json::object();
    add_bandwidth_results(patterns[JsonKeys::RANDOM],
                          stats.all_random_read_bw,
                          stats.all_random_write_bw,
                          stats.all_random_copy_bw);
  }
  
  return patterns;
}


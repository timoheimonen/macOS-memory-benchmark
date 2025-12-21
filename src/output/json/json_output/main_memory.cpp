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
#include "benchmark/benchmark_runner.h"  // For BenchmarkStatistics
#include "third_party/nlohmann/json.hpp"   // JSON library

// Build main memory results JSON object
nlohmann::json build_main_memory_json(const BenchmarkStatistics& stats) {
  nlohmann::json main_memory;
  
  // Add bandwidth results
  add_bandwidth_results(main_memory, 
                        stats.all_read_bw_gb_s,
                        stats.all_write_bw_gb_s,
                        stats.all_copy_bw_gb_s);
  
  // Add latency results
  add_latency_results(main_memory,
                      stats.all_average_latency_ns,
                      stats.all_main_mem_latency_samples);
  
  return main_memory;
}


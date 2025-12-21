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
#include "core/config/config.h"     // For BenchmarkConfig
#include "benchmark/benchmark_runner.h"  // For BenchmarkStatistics
#include "third_party/nlohmann/json.hpp"   // JSON library

// Build cache bandwidth JSON for a specific cache level
// cache_key: "l1", "l2", or "custom"
void add_cache_bandwidth_json(nlohmann::json& cache_json,
                               const std::string& cache_key,
                               const std::vector<double>& read_values,
                               const std::vector<double>& write_values,
                               const std::vector<double>& copy_values) {
  if (read_values.empty()) {
    return;
  }
  
  if (!cache_json.contains(cache_key)) {
    cache_json[cache_key] = nlohmann::json::object();
  }
  
  add_bandwidth_results(cache_json[cache_key], read_values, write_values, copy_values);
}

// Build cache latency JSON for a specific cache level
// cache_key: "l1", "l2", or "custom"
void add_cache_latency_json(nlohmann::json& cache_json,
                             const std::string& cache_key,
                             const std::vector<double>& average_values,
                             const std::vector<double>& samples) {
  if (average_values.empty()) {
    return;
  }
  
  if (!cache_json.contains(cache_key)) {
    cache_json[cache_key] = nlohmann::json::object();
  }
  
  add_latency_results(cache_json[cache_key], average_values, samples);
}

// Build cache results JSON object
nlohmann::json build_cache_json(const BenchmarkConfig& config, const BenchmarkStatistics& stats) {
  nlohmann::json cache;
  
  if (config.use_custom_cache_size) {
    // Custom cache bandwidth
    add_cache_bandwidth_json(cache,
                             JsonKeys::CUSTOM,
                             stats.all_custom_read_bw_gb_s,
                             stats.all_custom_write_bw_gb_s,
                             stats.all_custom_copy_bw_gb_s);
    
    // Custom cache latency
    add_cache_latency_json(cache,
                           JsonKeys::CUSTOM,
                           stats.all_custom_latency_ns,
                           stats.all_custom_latency_samples);
  } else {
    // L1 cache bandwidth
    add_cache_bandwidth_json(cache,
                             JsonKeys::L1,
                             stats.all_l1_read_bw_gb_s,
                             stats.all_l1_write_bw_gb_s,
                             stats.all_l1_copy_bw_gb_s);
    
    // L1 cache latency
    add_cache_latency_json(cache,
                          JsonKeys::L1,
                          stats.all_l1_latency_ns,
                          stats.all_l1_latency_samples);
    
    // L2 cache bandwidth
    add_cache_bandwidth_json(cache,
                             JsonKeys::L2,
                             stats.all_l2_read_bw_gb_s,
                             stats.all_l2_write_bw_gb_s,
                             stats.all_l2_copy_bw_gb_s);
    
    // L2 cache latency
    add_cache_latency_json(cache,
                          JsonKeys::L2,
                          stats.all_l2_latency_ns,
                          stats.all_l2_latency_samples);
  }
  
  return cache;
}


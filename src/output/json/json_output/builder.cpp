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
 * @file builder.cpp
 * @brief JSON builder helper functions for bandwidth and latency results
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2025
 *
 * This file provides helper functions for building JSON objects from benchmark
 * results. Handles the structure for bandwidth (read/write/copy) and latency
 * results with optional statistical aggregation.
 *
 * Functions:
 * - add_bandwidth_results(): Adds bandwidth measurements with statistics
 * - add_latency_results(): Adds latency measurements and sample distributions
 * - build_config_json(): Creates configuration section with system info
 */
// This file uses the nlohmann/json library for JSON parsing and generation.
// Library: https://github.com/nlohmann/json
// License: MIT License
//
#include "output/json/json_output.h"
#include "utils/json_utils.h" // JSON utility functions
#include "core/config/config.h"     // For BenchmarkConfig
#include "third_party/nlohmann/json.hpp"   // JSON library

// Helper function to add bandwidth results to JSON
// Adds read, write, and copy bandwidth with values and statistics if applicable
void add_bandwidth_results(nlohmann::json& json_obj, 
                           const std::vector<double>& read_values,
                           const std::vector<double>& write_values,
                           const std::vector<double>& copy_values) {
  if (read_values.empty()) {
    return;
  }
  
  json_obj[JsonKeys::BANDWIDTH] = nlohmann::json::object();
  
  // Read bandwidth
  json_obj[JsonKeys::BANDWIDTH][JsonKeys::READ_GB_S] = nlohmann::json::object();
  json_obj[JsonKeys::BANDWIDTH][JsonKeys::READ_GB_S][JsonKeys::VALUES] = read_values;
  if (read_values.size() > 1) {
    json_obj[JsonKeys::BANDWIDTH][JsonKeys::READ_GB_S][JsonKeys::STATISTICS] = 
        calculate_json_statistics(read_values);
  }
  
  // Write bandwidth
  json_obj[JsonKeys::BANDWIDTH][JsonKeys::WRITE_GB_S] = nlohmann::json::object();
  json_obj[JsonKeys::BANDWIDTH][JsonKeys::WRITE_GB_S][JsonKeys::VALUES] = write_values;
  if (write_values.size() > 1) {
    json_obj[JsonKeys::BANDWIDTH][JsonKeys::WRITE_GB_S][JsonKeys::STATISTICS] = 
        calculate_json_statistics(write_values);
  }
  
  // Copy bandwidth
  json_obj[JsonKeys::BANDWIDTH][JsonKeys::COPY_GB_S] = nlohmann::json::object();
  json_obj[JsonKeys::BANDWIDTH][JsonKeys::COPY_GB_S][JsonKeys::VALUES] = copy_values;
  if (copy_values.size() > 1) {
    json_obj[JsonKeys::BANDWIDTH][JsonKeys::COPY_GB_S][JsonKeys::STATISTICS] = 
        calculate_json_statistics(copy_values);
  }
}

// Helper function to add latency results to JSON
// Adds average latency and optionally sample distribution with statistics
void add_latency_results(nlohmann::json& json_obj,
                         const std::vector<double>& average_values,
                         const std::vector<double>& samples) {
  if (average_values.empty()) {
    return;
  }
  
  json_obj[JsonKeys::LATENCY] = nlohmann::json::object();
  json_obj[JsonKeys::LATENCY][JsonKeys::AVERAGE_NS] = nlohmann::json::object();
  json_obj[JsonKeys::LATENCY][JsonKeys::AVERAGE_NS][JsonKeys::VALUES] = average_values;
  if (average_values.size() > 1) {
    json_obj[JsonKeys::LATENCY][JsonKeys::AVERAGE_NS][JsonKeys::STATISTICS] = 
        calculate_json_statistics(average_values);
  }
  
  // Add sample distribution if available
  if (!samples.empty()) {
    json_obj[JsonKeys::LATENCY][JsonKeys::SAMPLES_NS] = samples;
    if (samples.size() > 1) {
      json_obj[JsonKeys::LATENCY][JsonKeys::SAMPLES_STATISTICS] = 
          calculate_json_statistics(samples);
    }
  }
}

// Build configuration JSON object
nlohmann::json build_config_json(const BenchmarkConfig& config) {
  nlohmann::json config_json;
  config_json[JsonKeys::BUFFER_SIZE_MB] = config.buffer_size_mb;
  config_json[JsonKeys::BUFFER_SIZE_BYTES] = config.buffer_size;
  config_json[JsonKeys::ITERATIONS] = config.iterations;
  config_json[JsonKeys::LOOP_COUNT] = config.loop_count;
  config_json[JsonKeys::LATENCY_SAMPLE_COUNT] = config.latency_sample_count;
  config_json[JsonKeys::CPU_NAME] = config.cpu_name;
  config_json[JsonKeys::MACOS_VERSION] = config.macos_version;
  config_json[JsonKeys::PERFORMANCE_CORES] = config.perf_cores;
  config_json[JsonKeys::EFFICIENCY_CORES] = config.eff_cores;
  config_json[JsonKeys::TOTAL_THREADS] = config.num_threads;
  config_json[JsonKeys::USE_CUSTOM_CACHE_SIZE] = config.use_custom_cache_size;
  config_json[JsonKeys::USE_NON_CACHEABLE] = config.use_non_cacheable;
  
  if (config.use_custom_cache_size) {
    config_json[JsonKeys::CUSTOM_CACHE_SIZE_BYTES] = config.custom_cache_size_bytes;
    config_json[JsonKeys::CUSTOM_CACHE_SIZE_KB] = config.custom_cache_size_bytes / 1024;
    config_json[JsonKeys::CUSTOM_BUFFER_SIZE_BYTES] = config.custom_buffer_size;
  } else {
    config_json[JsonKeys::L1_CACHE_SIZE_BYTES] = config.l1_cache_size;
    config_json[JsonKeys::L2_CACHE_SIZE_BYTES] = config.l2_cache_size;
    config_json[JsonKeys::L1_BUFFER_SIZE_BYTES] = config.l1_buffer_size;
    config_json[JsonKeys::L2_BUFFER_SIZE_BYTES] = config.l2_buffer_size;
  }
  
  return config_json;
}


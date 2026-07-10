// Copyright 2026 Timo Heimonen <timo.heimonen@proton.me>
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
#include "output/json/json_output/json_output_api.h"
#include "utils/json_utils.h" // JSON utility functions
#include "core/config/config.h"     // For BenchmarkConfig
#include "core/config/constants.h"
#include "third_party/nlohmann/json.hpp"   // JSON library

#include <string>
#include <unistd.h>

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
    json_obj[JsonKeys::LATENCY][JsonKeys::SAMPLES_NS] = nlohmann::json::object();
    json_obj[JsonKeys::LATENCY][JsonKeys::SAMPLES_NS][JsonKeys::VALUES] = samples;
    if (samples.size() > 1) {
      json_obj[JsonKeys::LATENCY][JsonKeys::SAMPLES_NS][JsonKeys::STATISTICS] =
          calculate_json_statistics(samples);
    }
  }
}

// Build configuration JSON object
nlohmann::json build_config_json(const BenchmarkConfig& config, const char* mode_name) {
  nlohmann::json config_json;
  config_json[JsonKeys::MODE] = mode_name;
  config_json[JsonKeys::BUFFER_SIZE_MB] = config.buffer_size_mb;
  config_json[JsonKeys::BUFFER_SIZE_BYTES] = config.buffer_size;
  config_json[JsonKeys::ITERATIONS] = config.iterations;
  config_json[JsonKeys::LOOP_COUNT] = config.loop_count;
  config_json[JsonKeys::LATENCY_SAMPLE_COUNT] = config.latency_sample_count;
  config_json[JsonKeys::LATENCY_STRIDE_BYTES] = config.latency_stride_bytes;
  config_json[JsonKeys::LATENCY_CHAIN_MODE] =
      latency_chain_mode_to_string(resolve_latency_chain_mode(config.latency_chain_mode,
                                                              config.latency_tlb_locality_bytes));
  config_json[JsonKeys::USE_LATENCY_TLB_LOCALITY] = (config.latency_tlb_locality_bytes > 0);
  config_json[JsonKeys::LATENCY_TLB_LOCALITY_BYTES] = config.latency_tlb_locality_bytes;
  config_json[JsonKeys::LATENCY_TLB_LOCALITY_KB] = config.latency_tlb_locality_bytes / 1024;
  config_json[JsonKeys::CPU_NAME] = config.cpu_name;
  config_json[JsonKeys::MACOS_VERSION] = config.macos_version;
  config_json[JsonKeys::PERFORMANCE_CORES] = config.perf_cores;
  config_json[JsonKeys::EFFICIENCY_CORES] = config.eff_cores;
  config_json[JsonKeys::TOTAL_THREADS] = config.num_threads;
  config_json[JsonKeys::USE_CUSTOM_CACHE_SIZE] = config.use_custom_cache_size;
  config_json[JsonKeys::USE_NON_CACHEABLE] = config.use_non_cacheable;

  if (std::string(mode_name) == Constants::PATTERNS_JSON_MODE_NAME) {
    config_json["pattern_schema_version"] =
        Constants::PATTERN_JSON_SCHEMA_VERSION;
    config_json["methodology_version"] =
        Constants::PATTERN_METHODOLOGY_VERSION;
    config_json["pattern_seed"] = std::to_string(config.pattern_seed);
    config_json["pattern_seed_source"] =
        config.user_specified_pattern_seed ? "user" : "generated";
    config_json["pattern_seed_encoding"] = "uint64-decimal-string";
    config_json["pattern_pass_policy"] = config.user_specified_iterations
                                              ? "explicit-iterations"
                                              : "automatic-duration-calibration";
    config_json["calibration_target_seconds"] =
        Constants::PATTERN_CALIBRATION_TARGET_SECONDS;
    config_json["calibration_window_min_seconds"] =
        Constants::PATTERN_CALIBRATION_MIN_SECONDS;
    config_json["calibration_window_max_seconds"] =
        Constants::PATTERN_CALIBRATION_MAX_SECONDS;
    config_json["calibration_max_corrections"] =
        Constants::PATTERN_CALIBRATION_MAX_CORRECTIONS;
    config_json["warmup_semantics"] = "steady-state-same-shape";
    config_json["pattern_execution_order_policy"] =
        "cyclic-latin-square-across-count-loops";
    config_json["operation_execution_order_policy"] =
        "fixed-read-write-copy-with-operation-specific-warmup";
    config_json["native_page_size_bytes"] = static_cast<size_t>(getpagesize());
    config_json["qos_policy"] = "best-effort-scheduler-hint-no-core-pinning";
    config_json["thread_selection_policy"] =
        config.user_specified_threads ? "explicit-request-capped-to-detected-cores"
                                      : "detected-core-count-default";
  } else if (std::string(mode_name) == Constants::BENCHMARK_JSON_MODE_NAME) {
    config_json["benchmark_schema_version"] =
        Constants::BENCHMARK_JSON_SCHEMA_VERSION;
    config_json["methodology_version"] =
        Constants::BENCHMARK_METHODOLOGY_VERSION;
    config_json["benchmark_seed"] = std::to_string(config.benchmark_seed);
    config_json["benchmark_seed_source"] =
        config.user_specified_benchmark_seed ? "user" : "generated";
    config_json["benchmark_seed_encoding"] = "uint64-decimal-string";
    config_json["bandwidth_work_policy"] =
        config.user_specified_iterations ? "explicit-iterations"
                                         : "automatic-duration-calibration";
    config_json["bandwidth_calibration_target_seconds"] =
        Constants::BENCHMARK_CALIBRATION_TARGET_SECONDS;
    config_json["bandwidth_calibration_window_min_seconds"] =
        Constants::BENCHMARK_CALIBRATION_MIN_SECONDS;
    config_json["bandwidth_calibration_window_max_seconds"] =
        Constants::BENCHMARK_CALIBRATION_MAX_SECONDS;
    config_json["calibration_max_corrections"] =
        Constants::BENCHMARK_CALIBRATION_MAX_CORRECTIONS;
    config_json["latency_calibration_target_seconds"] =
        Constants::BENCHMARK_LATENCY_TARGET_SECONDS;
    config_json["latency_calibration_window_min_seconds"] =
        Constants::BENCHMARK_LATENCY_CALIBRATION_MIN_SECONDS;
    config_json["latency_calibration_window_max_seconds"] =
        Constants::BENCHMARK_LATENCY_CALIBRATION_MAX_SECONDS;
    config_json["latency_minimum_complete_chain_cycles"] =
        Constants::BENCHMARK_LATENCY_MIN_COMPLETE_CYCLES;
    config_json["phase_execution_order_policy"] =
        "cyclic-latin-square-across-count-loops";
    config_json["operation_execution_order_policy"] =
        "cyclic-read-write-copy-with-operation-specific-warmup";
    config_json["latency_headline_semantics"] =
        "one-continuous-pointer-chase-pass";
    config_json["latency_sample_semantics"] =
        "separate-continuing-window-pass";
    config_json["locality_comparison_semantics"] =
        "paired-alternating-rounds-global-minus-locality-16k";
    config_json["warmup_semantics"] = "steady-state-warm-memory";
    config_json["native_page_size_bytes"] =
        static_cast<size_t>(getpagesize());
    config_json["qos_policy"] =
        "best-effort-scheduler-hint-no-core-pinning";
    config_json["main_thread_qos"] = {
        {"requested", config.main_thread_qos_requested},
        {"applied", config.main_thread_qos_applied},
        {"code", config.main_thread_qos_code}};
    config_json["thread_selection_policy"] =
        config.user_specified_threads ? "explicit-request-capped-to-detected-cores"
                                      : "detected-core-count-default";
  }
  
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

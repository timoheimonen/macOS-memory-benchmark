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
 * This file provides the shared configuration builder used by the active
 * status-bearing pattern JSON producer.
 *
 * Functions:
 * - build_config_json(): Creates configuration section with system info
 */
// This file uses the nlohmann/json library for JSON parsing and generation.
// Library: https://github.com/nlohmann/json
// License: MIT License
//
#include "output/json/json_output/json_output_api.h"
#include "core/config/config.h"     // For BenchmarkConfig
#include "core/config/constants.h"
#include "core/system/page_size.h"
#include "third_party/nlohmann/json.hpp"   // JSON library

#include <string>

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
    config_json["native_page_size_bytes"] = get_system_page_size_bytes();
    config_json["qos_policy"] = "best-effort-scheduler-hint-no-core-pinning";
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

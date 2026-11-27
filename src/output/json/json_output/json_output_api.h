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
 * @file json_output_api.h
 * @brief Shared JSON key constants and atomic file writer API
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 */
#ifndef JSON_OUTPUT_JSON_OUTPUT_API_H
#define JSON_OUTPUT_JSON_OUTPUT_API_H

#include <filesystem>
#include "third_party/nlohmann/json.hpp"

// JSON structure constants
namespace JsonKeys {
  // Top-level keys
  constexpr const char* VERSION = "version";
  constexpr const char* TIMESTAMP = "timestamp";
  constexpr const char* CONFIGURATION = "configuration";
  constexpr const char* EXECUTION_TIME_SEC = "execution_time_sec";
  
  // Configuration keys
  constexpr const char* MODE = "mode";
  constexpr const char* LOOP_COUNT = "loop_count";
  constexpr const char* LATENCY_SAMPLE_COUNT = "latency_sample_count";
  constexpr const char* LATENCY_STRIDE_BYTES = "latency_stride_bytes";
  constexpr const char* LATENCY_CHAIN_MODE = "latency_chain_mode";
  constexpr const char* TLB_DENSITY = "tlb_density";
  constexpr const char* CPU_NAME = "cpu_name";
  constexpr const char* MACOS_VERSION = "macos_version";
  constexpr const char* PERFORMANCE_CORES = "performance_cores";
  constexpr const char* EFFICIENCY_CORES = "efficiency_cores";
  constexpr const char* L1_CACHE_SIZE_BYTES = "l1_cache_size_bytes";
  
  // Main memory keys
  constexpr const char* SAMPLES_NS = "samples_ns";
  constexpr const char* PAGE_SIZE_BYTES = "page_size_bytes";
  constexpr const char* VALUES = "values";
  constexpr const char* STATISTICS = "statistics";
  
  // Cache keys
  constexpr const char* L1 = "l1";
  constexpr const char* L2 = "l2";
}

int write_json_to_file(const std::filesystem::path& file_path,
                       const nlohmann::ordered_json& json_output,
                       bool announce_success = true);

#endif // JSON_OUTPUT_JSON_OUTPUT_API_H

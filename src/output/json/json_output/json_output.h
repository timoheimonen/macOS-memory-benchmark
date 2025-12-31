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
#ifndef JSON_OUTPUT_JSON_OUTPUT_H
#define JSON_OUTPUT_JSON_OUTPUT_H

#include <cstdlib>  // EXIT_SUCCESS, EXIT_FAILURE
#include <vector>
#include <string>
#include <filesystem>
#include "third_party/nlohmann/json.hpp"

// Forward declarations
struct BenchmarkConfig;
struct BenchmarkStatistics;
struct PatternResults;
struct PatternStatistics;

// JSON structure constants
namespace JsonKeys {
  // Top-level keys
  constexpr const char* VERSION = "version";
  constexpr const char* TIMESTAMP = "timestamp";
  constexpr const char* CONFIGURATION = "configuration";
  constexpr const char* MAIN_MEMORY = "main_memory";
  constexpr const char* CACHE = "cache";
  constexpr const char* PATTERNS = "patterns";
  constexpr const char* EXECUTION_TIME_SEC = "execution_time_sec";
  
  // Configuration keys
  constexpr const char* BUFFER_SIZE_MB = "buffer_size_mb";
  constexpr const char* BUFFER_SIZE_BYTES = "buffer_size_bytes";
  constexpr const char* ITERATIONS = "iterations";
  constexpr const char* LOOP_COUNT = "loop_count";
  constexpr const char* LATENCY_SAMPLE_COUNT = "latency_sample_count";
  constexpr const char* CPU_NAME = "cpu_name";
  constexpr const char* PERFORMANCE_CORES = "performance_cores";
  constexpr const char* EFFICIENCY_CORES = "efficiency_cores";
  constexpr const char* TOTAL_THREADS = "total_threads";
  constexpr const char* USE_CUSTOM_CACHE_SIZE = "use_custom_cache_size";
  constexpr const char* USE_NON_CACHEABLE = "use_non_cacheable";
  constexpr const char* CUSTOM_CACHE_SIZE_BYTES = "custom_cache_size_bytes";
  constexpr const char* CUSTOM_CACHE_SIZE_KB = "custom_cache_size_kb";
  constexpr const char* CUSTOM_BUFFER_SIZE_BYTES = "custom_buffer_size_bytes";
  constexpr const char* L1_CACHE_SIZE_BYTES = "l1_cache_size_bytes";
  constexpr const char* L2_CACHE_SIZE_BYTES = "l2_cache_size_bytes";
  constexpr const char* L1_BUFFER_SIZE_BYTES = "l1_buffer_size_bytes";
  constexpr const char* L2_BUFFER_SIZE_BYTES = "l2_buffer_size_bytes";
  
  // Main memory keys
  constexpr const char* BANDWIDTH = "bandwidth";
  constexpr const char* READ_GB_S = "read_gb_s";
  constexpr const char* WRITE_GB_S = "write_gb_s";
  constexpr const char* COPY_GB_S = "copy_gb_s";
  constexpr const char* LATENCY = "latency";
  constexpr const char* AVERAGE_NS = "average_ns";
  constexpr const char* SAMPLES_NS = "samples_ns";
  constexpr const char* SAMPLES_STATISTICS = "samples_statistics";
  constexpr const char* VALUES = "values";
  constexpr const char* STATISTICS = "statistics";
  
  // Cache keys
  constexpr const char* L1 = "l1";
  constexpr const char* L2 = "l2";
  constexpr const char* CUSTOM = "custom";
  
  // Pattern keys
  constexpr const char* SEQUENTIAL_FORWARD = "sequential_forward";
  constexpr const char* SEQUENTIAL_REVERSE = "sequential_reverse";
  constexpr const char* STRIDED_64 = "strided_64";
  constexpr const char* STRIDED_4096 = "strided_4096";
  constexpr const char* RANDOM = "random";
}

// Internal helper functions (used by implementation files)
void add_bandwidth_results(nlohmann::json& json_obj, 
                           const std::vector<double>& read_values,
                           const std::vector<double>& write_values,
                           const std::vector<double>& copy_values);

void add_latency_results(nlohmann::json& json_obj,
                         const std::vector<double>& average_values,
                         const std::vector<double>& samples);

nlohmann::json build_config_json(const BenchmarkConfig& config);
nlohmann::json build_main_memory_json(const BenchmarkConfig& config, const BenchmarkStatistics& stats);
nlohmann::json build_cache_json(const BenchmarkConfig& config, const BenchmarkStatistics& stats);
nlohmann::json build_patterns_json(const PatternStatistics& stats);
int write_json_to_file(const std::filesystem::path& file_path, const nlohmann::json& json_output);

// Public API functions
// Save benchmark results to JSON file
// Returns EXIT_SUCCESS on success, EXIT_FAILURE on error
int save_results_to_json(const BenchmarkConfig& config, const BenchmarkStatistics& stats, double total_execution_time_sec);

// Save pattern benchmark results to JSON file
// Returns EXIT_SUCCESS on success, EXIT_FAILURE on error
int save_pattern_results_to_json(const BenchmarkConfig& config, const PatternStatistics& stats, double total_execution_time_sec);

#endif // JSON_OUTPUT_JSON_OUTPUT_H


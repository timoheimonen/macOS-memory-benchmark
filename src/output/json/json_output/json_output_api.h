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
 * @brief JSON output generation for benchmark results
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2025
 *
 * This header provides the public API and internal functions for generating
 * JSON output files from benchmark results. Uses the nlohmann/json library
 * for JSON manipulation.
 *
 * Features:
 * - Structured JSON output with configuration, results, and metadata
 * - Statistical aggregation (average, percentiles, stddev) for multi-loop runs
 * - Support for main memory, cache, and pattern benchmark results
 * - ISO 8601 timestamp generation
 * - File path resolution and output
 *
 * JSON Structure:
 * - configuration: System info and benchmark parameters
 * - execution_time_sec: Total benchmark duration
 * - main_memory/cache/patterns: Bandwidth and latency results with statistics
 * - timestamp: ISO 8601 UTC timestamp
 * - version: Software version
 */
#ifndef JSON_OUTPUT_JSON_OUTPUT_API_H
#define JSON_OUTPUT_JSON_OUTPUT_API_H

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
  constexpr const char* MODE = "mode";
  constexpr const char* BUFFER_SIZE_MB = "buffer_size_mb";
  constexpr const char* BUFFER_SIZE_BYTES = "buffer_size_bytes";
  constexpr const char* ITERATIONS = "iterations";
  constexpr const char* LOOP_COUNT = "loop_count";
  constexpr const char* LATENCY_SAMPLE_COUNT = "latency_sample_count";
  constexpr const char* LATENCY_STRIDE_BYTES = "latency_stride_bytes";
  constexpr const char* LATENCY_CHAIN_MODE = "latency_chain_mode";
  constexpr const char* TLB_DENSITY = "tlb_density";
  constexpr const char* USE_LATENCY_TLB_LOCALITY = "use_latency_tlb_locality";
  constexpr const char* LATENCY_TLB_LOCALITY_BYTES = "latency_tlb_locality_bytes";
  constexpr const char* LATENCY_TLB_LOCALITY_KB = "latency_tlb_locality_kb";
  constexpr const char* CPU_NAME = "cpu_name";
  constexpr const char* MACOS_VERSION = "macos_version";
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
  constexpr const char* SAMPLES_NS = "samples_ns";
  constexpr const char* PAGE_SIZE_BYTES = "page_size_bytes";
  constexpr const char* VALUES = "values";
  constexpr const char* STATISTICS = "statistics";
  
  // Cache keys
  constexpr const char* L1 = "l1";
  constexpr const char* L2 = "l2";
  
  // Pattern keys
  constexpr const char* SEQUENTIAL_FORWARD = "sequential_forward";
  constexpr const char* SEQUENTIAL_REVERSE = "sequential_reverse";
  constexpr const char* STRIDED_64 = "strided_64";
  constexpr const char* STRIDED_4096 = "strided_4096";
  constexpr const char* STRIDED_16384 = "strided_16384";
  constexpr const char* STRIDED_2MB = "strided_2mb";
  constexpr const char* RANDOM = "random";
}

nlohmann::json build_config_json(const BenchmarkConfig& config, const char* mode_name);
nlohmann::json build_patterns_json(const PatternStatistics& stats);
void add_standard_benchmark_results(nlohmann::ordered_json& output,
                                    const BenchmarkConfig& config,
                                    const BenchmarkStatistics& stats);
int write_json_to_file(const std::filesystem::path& file_path,
                       const nlohmann::ordered_json& json_output,
                       bool announce_success = true);

/**
 * @brief Build the standard benchmark schema-2 payload in memory.
 *
 * The document shape is governed by
 * `configuration.benchmark_schema_version`; this builder performs no output
 * target classification or I/O.
 *
 * @param config Immutable command configuration recorded in the payload.
 * @param stats Immutable terminal or intermediate statistics snapshot,
 *        including retained partial evidence and completion counters.
 * @param total_execution_time_sec Command elapsed time in seconds.
 * @return A caller-owned ordered JSON value containing the complete schema-2
 *         snapshot. The returned value retains no references to the inputs.
 * @throws std::exception If allocation, timestamp creation, string handling,
 *         or JSON construction fails.
 * @note Concurrent calls are safe when each caller keeps its referenced inputs
 *       immutable for the duration of the call.
 */
nlohmann::ordered_json build_results_json(const BenchmarkConfig& config,
                                          const BenchmarkStatistics& stats,
                                          double total_execution_time_sec);

/**
 * @brief Build the pattern benchmark schema-3 payload in memory.
 *
 * The document shape is governed by
 * `configuration.pattern_schema_version`; this builder performs no output
 * target classification or I/O.
 *
 * @param config Immutable command configuration recorded in the payload.
 * @param stats Immutable pattern statistics and retained per-loop evidence.
 * @param total_execution_time_sec Command elapsed time in seconds.
 * @return A caller-owned ordered JSON value containing the complete schema-3
 *         snapshot. The returned value retains no references to the inputs.
 * @throws std::exception If allocation, timestamp creation, string handling,
 *         or JSON construction fails.
 * @note Concurrent calls are safe when each caller keeps its referenced inputs
 *       immutable for the duration of the call.
 */
nlohmann::ordered_json build_pattern_results_json(const BenchmarkConfig& config,
                                                  const PatternStatistics& stats,
                                                  double total_execution_time_sec);

/**
 * @brief Build and atomically replace a file with standard schema-2 JSON.
 *
 * This is a legacy file-only adapter. An empty target is a successful no-op.
 * Command code that accepts stdout must first classify the raw target and use
 * `JsonOutputSession` for the exact `-` sentinel.
 *
 * @param config Immutable configuration whose non-empty `output_file` names a
 *        real file target.
 * @param stats Immutable statistics snapshot to serialize.
 * @param total_execution_time_sec Command elapsed time in seconds.
 * @param announce_success Whether a successful replacement prints the
 *        centralized save announcement.
 * @return `EXIT_SUCCESS` for a disabled target or successful atomic replace;
 *         `EXIT_FAILURE` for a contained file-output failure.
 * @throws std::exception If payload construction or output-path resolution
 *         fails before the atomic writer's return-code boundary.
 * @pre A non-empty `config.output_file` is not the exact stdout sentinel `-`.
 * @note The function retains no references and is synchronous. Callers must
 *       not mutate its inputs or target path concurrently.
 */
int save_results_to_json(const BenchmarkConfig& config,
                         const BenchmarkStatistics& stats,
                         double total_execution_time_sec,
                         bool announce_success = true);

/**
 * @brief Build and atomically replace a file with pattern schema-3 JSON.
 *
 * This is a legacy file-only adapter. An empty target is a successful no-op.
 * Command code that accepts stdout must first classify the raw target and use
 * `JsonOutputSession` for the exact `-` sentinel.
 *
 * @param config Immutable configuration whose non-empty `output_file` names a
 *        real file target.
 * @param stats Immutable pattern statistics snapshot to serialize.
 * @param total_execution_time_sec Command elapsed time in seconds.
 * @return `EXIT_SUCCESS` for a disabled target or successful atomic replace;
 *         `EXIT_FAILURE` for a contained file-output failure.
 * @throws std::exception If payload construction or output-path resolution
 *         fails before the atomic writer's return-code boundary.
 * @pre A non-empty `config.output_file` is not the exact stdout sentinel `-`.
 * @note The function retains no references and is synchronous. Callers must
 *       not mutate its inputs or target path concurrently.
 */
int save_pattern_results_to_json(const BenchmarkConfig& config,
                                 const PatternStatistics& stats,
                                 double total_execution_time_sec);

#endif // JSON_OUTPUT_JSON_OUTPUT_API_H

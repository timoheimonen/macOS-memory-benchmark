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
 * @file config.h
 * @brief Benchmark configuration structure and parsing functions
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 *
 * This header defines the BenchmarkConfig structure and provides functions
 * to parse command-line arguments and validate configuration settings.
 *
 * This module uses various constants from constants.h including:
 * - Cache size validation (MIN_CACHE_SIZE_KB, MAX_CACHE_SIZE_KB)
 * - Size conversion (BYTES_PER_KB, BYTES_PER_MB)
 * - Latency test parameters (LATENCY_STRIDE_BYTES, DEFAULT_LATENCY_TLB_LOCALITY_KB)
 *
 * @see constants.h for detailed constant definitions and values.
 */
#ifndef CONFIG_H
#define CONFIG_H

#include <cstddef>  // size_t
#include <cstdint>
#include <string>
#include <vector>
#include <cstdlib>  // EXIT_SUCCESS, EXIT_FAILURE
#include "core/config/constants.h"
#include "core/memory/memory_utils.h"

/**
 * @enum TlbSweepDensity
 * @brief Sweep density profile for standalone `--analyze-tlb` mode
 */
enum class TlbSweepDensity {
  Low = 0,    ///< Quick: 15 points, no refinement, 7-12 rounds
  Medium,     ///< Standard: 15 points + refinement, 10-20 rounds
  High,       ///< Exhaustive: 29 points + refinement, 15-30 rounds
};

/**
 * @enum SweepParameter
 * @brief Parameter names supported by `--sweep key=value1,value2`.
 */
enum class SweepParameter {
  BufferSizeMb = 0,
  CacheSizeKb,
  Threads,
  LatencyTlbLocalityKb,
  LatencyStrideBytes,
  LatencyChainMode,
  TlbDensity,
};

/**
 * @struct SweepValue
 * @brief Parsed typed value for one sweep parameter point.
 */
struct SweepValue {
  std::string raw_value;
  long long integer_value = 0;
  unsigned long unsigned_value = 0;
  LatencyChainMode latency_chain_mode = LatencyChainMode::Auto;
  TlbSweepDensity tlb_sweep_density = TlbSweepDensity::Medium;
};

/**
 * @struct SweepSpec
 * @brief One `--sweep` option with a parameter and candidate values.
 */
struct SweepSpec {
  SweepParameter parameter = SweepParameter::BufferSizeMb;
  std::string parameter_name;
  std::vector<SweepValue> values;
};

/** Deterministic parser/platform values used only by unit tests. */
struct ConfigTestHooks {
  bool use_system_info = false;
  std::string cpu_name;
  std::string macos_version;
  int performance_cores = 0;
  int efficiency_cores = 0;
  int total_logical_cores = 1;
  size_t l1_cache_size = 0;
  size_t l2_cache_size = 0;
  uint64_t generated_seed = 0;
  size_t page_size_bytes = 0;
};

void set_config_test_hooks(const ConfigTestHooks* hooks);
const ConfigTestHooks* get_config_test_hooks();

/**
 * @brief Return the effective OS page size in bytes for config validation.
 * @return Injected test-hook page size when active and non-zero, otherwise the host system page size.
 */
size_t get_config_page_size_bytes();

/**
 * @enum StrictIntegerParseStatus
 * @brief Result of parsing one complete base-10 CLI integer token.
 */
enum class StrictIntegerParseStatus {
  Success = 0,
  Invalid,
  OutOfRange,
};

/**
 * @brief Parse a complete signed base-10 token without whitespace or a leading plus sign.
 */
StrictIntegerParseStatus parse_strict_signed_decimal(const std::string& value,
                                                     long long& out_value);

/**
 * @brief Parse a complete unsigned base-10 token without whitespace or any sign.
 */
StrictIntegerParseStatus parse_strict_unsigned_decimal(const std::string& value,
                                                       uint64_t& out_value);

/** @brief Stable user-facing reason for a failed signed-token parse. */
const char* strict_signed_decimal_error_reason(StrictIntegerParseStatus status);

/** @brief Stable user-facing reason for a failed unsigned-token parse. */
const char* strict_unsigned_decimal_error_reason(StrictIntegerParseStatus status);

/**
 * @struct BenchmarkConfig
 * @brief Configuration structure containing standalone TLB and preflight settings
 *
 * This structure holds configuration parameters for standalone TLB analysis
 * and general preflight parsing, including user-provided settings, detected
 * system information, and various flags.
 */
struct BenchmarkConfig {
  // User-provided settings
  unsigned long buffer_size_mb = Constants::DEFAULT_BUFFER_SIZE_MB;  ///< Buffer size in megabytes
  /// Fixed iteration count when explicitly set; otherwise the calibration pilot/fallback value.
  int iterations = Constants::DEFAULT_ITERATIONS;
  int loop_count = Constants::DEFAULT_LOOP_COUNT;  ///< Number of benchmark loops to run
  long long custom_cache_size_kb_ll = -1;  ///< User-requested custom cache size in KB (-1 = none)
  int latency_sample_count = Constants::DEFAULT_LATENCY_SAMPLE_COUNT;  ///< Number of latency samples to collect per test
  size_t latency_stride_bytes = Constants::LATENCY_STRIDE_BYTES;  ///< Stride used for latency pointer chains (bytes)
  LatencyChainMode latency_chain_mode = LatencyChainMode::Auto;  ///< Pointer-chain construction policy (auto preserves default behavior)
  size_t latency_tlb_locality_bytes = Constants::DEFAULT_LATENCY_TLB_LOCALITY_KB * Constants::BYTES_PER_KB;  ///< TLB-locality window for latency chains (default 1 MB; 0 = global random)
  TlbSweepDensity tlb_sweep_density = TlbSweepDensity::Medium;  ///< Standard profile for standalone --analyze-tlb
  uint64_t tlb_seed = 0;  ///< Reproducible standalone TLB planner/chain seed
  
  // System info
  std::string cpu_name;          ///< CPU model name
  std::string macos_version;     ///< macOS version string (e.g., "14.2.1")
  int perf_cores = 0;            ///< Number of performance cores
  int eff_cores = 0;             ///< Number of efficiency cores
  int num_threads = 0;           ///< Total number of threads to use
  size_t l1_cache_size = 0;      ///< L1 cache size in bytes
  size_t l2_cache_size = 0;      ///< L2 cache size in bytes
  size_t custom_cache_size_bytes = 0;  ///< Custom cache size in bytes
  
  // Flags
  bool use_custom_cache_size = false;  ///< Whether to use custom cache size
  bool use_non_cacheable = false;      ///< Use cache-discouraging hints (best-effort, not true non-cacheable)
  bool user_specified_threads = false; ///< Whether user explicitly set --threads parameter
  bool analyze_tlb = false;            ///< When true, run standalone TLB analysis mode
  bool run_sweep = false;              ///< Whether to execute a multi-configuration sweep
  bool help_printed = false;           ///< Whether -h/--help was invoked (usage already printed)
  size_t sweep_max_runs = Constants::DEFAULT_SWEEP_MAX_RUNS;  ///< Maximum allowed sweep combinations

  // Best-effort benchmark preparation status
  bool main_thread_qos_requested = false;  ///< Whether USER_INTERACTIVE QoS was requested
  bool main_thread_qos_applied = false;    ///< Whether the QoS request succeeded
  int main_thread_qos_code = 0;            ///< Return code from pthread_set_qos_class_self_np()

  // Tracking flags for user-specified parameters
  bool user_specified_buffer_size = false;         ///< Whether user explicitly set --buffer-size
  bool user_specified_iterations = false;          ///< Whether user explicitly set --iterations
  bool user_specified_latency_samples = false;     ///< Whether user explicitly set --latency-samples
  bool user_specified_latency_stride = false;      ///< Whether user explicitly set --latency-stride-bytes
  bool user_specified_latency_chain_mode = false;  ///< Whether user explicitly set --latency-chain-mode
  bool user_specified_latency_tlb_locality = false; ///< Whether user explicitly set --latency-tlb-locality-kb
  bool user_specified_tlb_density = false;         ///< Whether user explicitly set --tlb-density
  bool user_specified_tlb_seed = false;  ///< Whether user explicitly set --seed for --analyze-tlb
  bool user_specified_seed = false;  ///< Whether user explicitly set --seed in general preflight
  
  /**
   * Raw command-level JSON output target. Empty disables JSON for a direct
   * command, exact `-` selects one final stdout document, and every other
   * non-empty value is a file target, including `./-` and flag-shaped names.
   * Sweep validation separately rejects an empty target. Command code must
   * classify this raw value before passing only real file targets to legacy
   * file-only adapters.
   */
  std::string output_file;

  // Sweep configuration
  std::vector<SweepSpec> sweep_specs;  ///< Parsed `--sweep` parameter/value lists
};

/**
 * @brief Parse command line arguments and populate config
 * @param argc Number of command-line arguments
 * @param argv Array of command-line argument strings
 * @param[out] config Reference to BenchmarkConfig structure to populate
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on error (and prints error message)
 * @see Constants::MIN_CACHE_SIZE_KB
 * @see Constants::MAX_CACHE_SIZE_KB
 * @see Constants::BYTES_PER_KB
 */
int parse_arguments(int argc, char* argv[], BenchmarkConfig& config);

/**
 * @brief Validate configuration values
 * @param[in,out] config Reference to BenchmarkConfig structure to validate
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on error
 *
 * Validates that all configuration values are within acceptable ranges and
 * performs necessary adjustments (e.g., rounding buffer sizes).
 * @see Constants::MEMORY_LIMIT_FACTOR
 * @see Constants::FALLBACK_TOTAL_LIMIT_MB
 * @see Constants::DEFAULT_BUFFER_SIZE_MB
 * @see Constants::BYTES_PER_MB
 */
int validate_config(BenchmarkConfig& config);

#endif // CONFIG_H

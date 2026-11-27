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

/**
 * @file cpu_latency.h
 * @brief Standalone `-L` / `--cpu-latency` CLI, configuration, and entrypoint
 */

#ifndef CPU_LATENCY_H
#define CPU_LATENCY_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "core/config/constants.h"
#include "core/memory/memory_utils.h"
#include "third_party/nlohmann/json.hpp"

class JsonOutputSession;
struct CpuLatencyWorkPlan;

/** Supported sweepable axes for standalone `-L` / `--cpu-latency`. */
enum class CpuLatencySweepParameter : uint8_t {
  BufferSizeMb = 0,
  CacheSizeKb,
  LatencyTlbLocalityKb,
  LatencyStrideBytes,
  LatencyChainMode,
};

/** One parsed typed sweep point for `--cpu-latency`. */
struct CpuLatencySweepValue {
  std::string raw_value;
  size_t integer_value = 0;
  LatencyChainMode chain_mode = LatencyChainMode::Auto;
};

/** One validated `--sweep key=v1,v2,...` dimension for `--cpu-latency`. */
struct CpuLatencySweepSpec {
  CpuLatencySweepParameter parameter = CpuLatencySweepParameter::BufferSizeMb;
  std::string parameter_name;
  std::vector<CpuLatencySweepValue> values;
};

/** Raw parsed CLI options for `-L` / `--cpu-latency`. */
struct CpuLatencyCliOptions {
  size_t buffer_size_mb = Constants::DEFAULT_BUFFER_SIZE_MB;
  std::optional<size_t> custom_cache_size_kb;
  std::optional<size_t> iterations;
  size_t loop_count = 3;
  size_t latency_samples =
      static_cast<size_t>(Constants::DEFAULT_LATENCY_SAMPLE_COUNT);
  size_t latency_stride_bytes = Constants::LATENCY_STRIDE_BYTES;
  LatencyChainMode latency_chain_mode = LatencyChainMode::Auto;
  size_t latency_tlb_locality_kb = Constants::DEFAULT_LATENCY_TLB_LOCALITY_KB;
  bool compare_locality = false;
  bool non_cacheable = false;
  std::optional<uint64_t> seed;
  std::string output_file;
  bool run_sweep = false;
  size_t sweep_max_runs = Constants::DEFAULT_SWEEP_MAX_RUNS;
  std::vector<CpuLatencySweepSpec> sweep_specs;
  bool help_requested = false;
  bool user_specified_buffer_size = false;
  bool user_specified_cache_size = false;
  bool user_specified_iterations = false;
  bool user_specified_latency_samples = false;
  bool user_specified_stride_bytes = false;
  bool user_specified_chain_mode = false;
  bool user_specified_tlb_locality_kb = false;
  bool user_specified_seed = false;
  std::vector<std::string> argv;
};

/** Resolved command configuration for `-L` / `--cpu-latency`. */
struct CpuLatencyConfig {
  size_t buffer_size_mb = Constants::DEFAULT_BUFFER_SIZE_MB;
  std::optional<size_t> custom_cache_size_kb;
  std::optional<size_t> iterations;
  size_t loop_count = 3;
  size_t latency_samples =
      static_cast<size_t>(Constants::DEFAULT_LATENCY_SAMPLE_COUNT);
  size_t latency_stride_bytes = Constants::LATENCY_STRIDE_BYTES;
  LatencyChainMode latency_chain_mode = LatencyChainMode::Auto;
  size_t latency_tlb_locality_kb = Constants::DEFAULT_LATENCY_TLB_LOCALITY_KB;
  bool compare_locality = false;
  bool non_cacheable = false;
  uint64_t seed = 0;
  bool user_specified_seed = false;
  bool user_specified_buffer_size = false;
  bool user_specified_cache_size = false;
  bool user_specified_iterations = false;
  bool user_specified_latency_samples = false;
  bool user_specified_stride_bytes = false;
  bool user_specified_chain_mode = false;
  bool user_specified_tlb_locality_kb = false;
  bool swept_buffer_size = false;
  bool swept_cache_size = false;
  bool swept_tlb_locality_kb = false;
  bool swept_stride_bytes = false;
  bool swept_chain_mode = false;
  bool help_printed = false;
  bool run_sweep = false;
  size_t sweep_max_runs = Constants::DEFAULT_SWEEP_MAX_RUNS;
  std::vector<CpuLatencySweepSpec> sweep_specs;
  bool collect_json_for_sweep = false;
  std::string output_file;
  std::vector<std::string> argv;
};

/** Deterministic platform and entropy inputs for parser unit tests. */
struct CpuLatencyParserTestHooks {
  size_t detected_workers = 1;
  uint64_t generated_seed = 1;
  size_t l1_cache_bytes = Constants::L1_CACHE_FALLBACK_SIZE_BYTES;
  size_t l2_cache_bytes = Constants::L2_CACHE_M2_M3_M4_M5_FALLBACK_SIZE_BYTES;
  size_t page_size_bytes = 16384;
  size_t available_memory_bytes = 0;
  bool use_memory_override = false;
};

/** Install or clear deterministic parser/system test hooks (not thread-safe). */
void set_cpu_latency_parser_test_hooks(const CpuLatencyParserTestHooks* hooks);

/** Deterministic command-boundary seams for interruption/fault tests. */
struct CpuLatencyCommandTestHooks {
  std::function<bool()> stop_requested;
  std::function<void(CpuLatencyWorkPlan&)> before_runner;
  std::function<void()> after_initialized_runner;
};

/** Install or clear deterministic command-boundary test hooks (not thread-safe). */
void set_cpu_latency_command_test_hooks(
    const CpuLatencyCommandTestHooks* hooks);

/**
 * Parse and validate `-L` / `--cpu-latency` CLI arguments.
 *
 * Performs exact-whitelist syntax checking, value/range parsing, help handling
 * (returning before OS queries or file creation), sweep validation, and
 * semantic target-omission / locality / compare-locality checks.
 */
int parse_cpu_latency_cli(int argc, char* argv[], CpuLatencyCliOptions& options);

/**
 * Resolve CLI options into a concrete `CpuLatencyConfig` (with generated seed
 * when omitted, page-size locality validation, and sweep preflight).
 */
int build_cpu_latency_config(const CpuLatencyCliOptions& options,
                             CpuLatencyConfig& config);

/** Execute one concrete `CpuLatencyConfig` directly via a caller session. */
int execute_cpu_latency_run(const CpuLatencyConfig& config,
                            JsonOutputSession* output_session,
                            bool install_signal_mask,
                            bool print_banner,
                            nlohmann::ordered_json* collected_json = nullptr,
                            std::string* failure_reason = nullptr,
                            bool* preparation_interrupted = nullptr) noexcept;

/** Execute one concrete `CpuLatencyConfig` and return its JSON document. */
int run_cpu_latency_collect(const CpuLatencyConfig& config,
                            nlohmann::ordered_json& result_json,
                            std::string* failure_reason = nullptr,
                            bool* preparation_interrupted = nullptr);

/** Standalone command entrypoint for `-L` / `--cpu-latency`. */
int run_cpu_latency_mode(int argc, char* argv[]);

#endif  // CPU_LATENCY_H

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
 * @file patterns.h
 * @brief Standalone `-P` / `--patterns` CLI, configuration, and entrypoint
 */

#ifndef PATTERNS_H
#define PATTERNS_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "core/config/constants.h"
#include "pattern_benchmark/patterns_work_plan.h"
#include "third_party/nlohmann/json.hpp"

class JsonOutputSession;

/** Supported sweepable axes for standalone `-P` / `--patterns`. */
enum class PatternsSweepParameter : uint8_t {
  BufferSizeMb = 0,
  Threads,
};

/** One parsed typed sweep point for `--patterns`. */
struct PatternsSweepValue {
  std::string raw_value;
  size_t integer_value = 0;
};

/** One validated `--sweep key=v1,v2,...` dimension for `--patterns`. */
struct PatternsSweepSpec {
  PatternsSweepParameter parameter = PatternsSweepParameter::BufferSizeMb;
  std::string parameter_name;
  std::vector<PatternsSweepValue> values;
};

/** Raw parsed CLI options for `-P` / `--patterns`. */
struct PatternsCliOptions {
  size_t buffer_size_mb = Constants::DEFAULT_BUFFER_SIZE_MB;
  std::optional<size_t> threads;
  std::optional<size_t> iterations;
  size_t loop_count = 3;
  bool non_cacheable = false;
  std::optional<uint64_t> seed;
  std::string output_file;
  bool run_sweep = false;
  size_t sweep_max_runs = Constants::DEFAULT_SWEEP_MAX_RUNS;
  std::vector<PatternsSweepSpec> sweep_specs;
  bool help_requested = false;
  bool user_specified_buffer_size = false;
  bool user_specified_threads = false;
  bool user_specified_iterations = false;
  bool user_specified_seed = false;
  std::vector<std::string> argv;
};

/** Resolved command configuration for `-P` / `--patterns`. */
struct PatternsConfig {
  size_t buffer_size_mb = Constants::DEFAULT_BUFFER_SIZE_MB;
  size_t requested_workers = 1;
  size_t available_workers = 1;
  bool user_specified_workers = false;
  std::optional<size_t> iterations;
  size_t loop_count = 3;
  bool non_cacheable = false;
  uint64_t seed = 0;
  bool user_specified_seed = false;
  bool user_specified_buffer_size = false;
  bool user_specified_iterations = false;
  bool swept_buffer_size = false;
  bool swept_threads = false;
  bool help_printed = false;
  bool run_sweep = false;
  size_t sweep_max_runs = Constants::DEFAULT_SWEEP_MAX_RUNS;
  std::vector<PatternsSweepSpec> sweep_specs;
  bool collect_json_for_sweep = false;
  std::string output_file;
  std::vector<std::string> argv;

  PatternsWorkPlanConfig to_work_plan_config() const;
};

/** Deterministic platform and entropy inputs for parser unit tests. */
struct PatternsParserTestHooks {
  size_t detected_workers = 1;
  uint64_t generated_seed = 1;
  size_t l1_cache_bytes = Constants::L1_CACHE_FALLBACK_SIZE_BYTES;
  size_t l2_cache_bytes = Constants::L2_CACHE_M2_M3_M4_M5_FALLBACK_SIZE_BYTES;
  size_t page_size_bytes = 16384;
  size_t available_memory_bytes = 0;
  bool use_memory_override = false;
};

/** Install or clear deterministic parser/system test hooks (not thread-safe). */
void set_patterns_parser_test_hooks(const PatternsParserTestHooks* hooks);

/** Deterministic command-boundary seams for interruption/fault tests. */
struct PatternsCommandTestHooks {
  std::function<bool()> stop_requested;
  std::function<void(PatternsWorkPlan&)> before_runner;
  std::function<void()> after_initialized_runner;
};

/** Install or clear deterministic command-boundary test hooks (not thread-safe). */
void set_patterns_command_test_hooks(const PatternsCommandTestHooks* hooks);

/**
 * Parse and validate `-P` / `--patterns` CLI arguments.
 *
 * Performs exact-whitelist syntax checking, value/range parsing, help handling
 * (returning before OS queries or file creation), and sweep validation.
 */
int parse_patterns_cli(int argc, char* argv[], PatternsCliOptions& options);

/**
 * Resolve CLI options into a concrete `PatternsConfig` (with detected logical
 * cores, generated seed when omitted, and sweep combination preflight).
 */
int build_patterns_config(const PatternsCliOptions& options,
                          PatternsConfig& config);

/** Execute one concrete `PatternsConfig` directly via a caller session. */
int execute_patterns_run(const PatternsConfig& config,
                         JsonOutputSession* output_session,
                         bool install_signal_mask,
                         bool print_banner,
                         nlohmann::ordered_json* collected_json = nullptr,
                         std::string* failure_reason = nullptr,
                         bool* preparation_interrupted = nullptr) noexcept;

/** Execute one concrete `PatternsConfig` and return its JSON document. */
int run_patterns_collect(const PatternsConfig& config,
                         nlohmann::ordered_json& result_json,
                         std::string* failure_reason = nullptr,
                         bool* preparation_interrupted = nullptr);

/** Standalone command entrypoint for `-P` / `--patterns`. */
int run_patterns_mode(int argc, char* argv[]);

#endif  // PATTERNS_H

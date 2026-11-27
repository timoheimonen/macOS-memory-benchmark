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
 * @file patterns.cpp
 * @brief Standalone `-P` / `--patterns` CLI, configuration, and entrypoint
 */

#include "pattern_benchmark/patterns.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "core/audit/build_manifest.h"
#include "core/audit/host_environment.h"
#include "core/config/cli_parser.h"
#include "core/config/config.h"
#include "core/config/constants.h"
#include "core/config/sweep_utils.h"
#include "core/signal/signal_handler.h"
#include "core/system/benchmark_qos.h"
#include "core/system/page_size.h"
#include "core/system/system_info.h"
#include "output/console/messages/messages_api.h"
#include "output/console/output_printer.h"
#include "output/json/json_output/json_output_session.h"
#include "pattern_benchmark/patterns_executor.h"
#include "pattern_benchmark/patterns_json.h"
#include "pattern_benchmark/patterns_output.h"
#include "pattern_benchmark/patterns_runner.h"
#include "pattern_benchmark/patterns_sweep_runner.h"
#include "pattern_benchmark/patterns_work_plan.h"
#include "utils/json_utils.h"
#include "utils/numeric_utils.h"
#include "utils/seed_utils.h"

namespace {

PatternsParserTestHooks active_parser_hooks;
bool parser_hooks_active = false;
PatternsCommandTestHooks active_command_hooks;
bool command_hooks_active = false;

enum PatternsOptionId {
  OPT_PATTERNS = 1,
  OPT_HELP,
  OPT_BUFFER_SIZE,
  OPT_THREADS,
  OPT_ITERATIONS,
  OPT_COUNT,
  OPT_SEED,
  OPT_NON_CACHEABLE,
  OPT_OUTPUT,
  OPT_SWEEP,
  OPT_SWEEP_MAX_RUNS,
};

constexpr std::array<CliOptionSpec, 11> kPatternsOptionSpecs{{
    {OPT_PATTERNS, "-P", "--patterns", CliOptionValueKind::None,
     CliOptionArity::Single},
    {OPT_HELP, "-h", "--help", CliOptionValueKind::None,
     CliOptionArity::Single},
    {OPT_BUFFER_SIZE, "-b", "--buffer-size", CliOptionValueKind::RequiredValue,
     CliOptionArity::Single},
    {OPT_THREADS, "-t", "--threads", CliOptionValueKind::RequiredValue,
     CliOptionArity::Single},
    {OPT_ITERATIONS, "-i", "--iterations", CliOptionValueKind::RequiredValue,
     CliOptionArity::Single},
    {OPT_COUNT, "-r", "--count", CliOptionValueKind::RequiredValue,
     CliOptionArity::Single},
    {OPT_SEED, nullptr, "--seed", CliOptionValueKind::RequiredValue,
     CliOptionArity::Single},
    {OPT_NON_CACHEABLE, "-u", "--non-cacheable", CliOptionValueKind::None,
     CliOptionArity::Single},
    {OPT_OUTPUT, "-o", "--output", CliOptionValueKind::RequiredValue,
     CliOptionArity::Single},
    {OPT_SWEEP, "-S", "--sweep", CliOptionValueKind::RequiredValue,
     CliOptionArity::Repeatable},
    {OPT_SWEEP_MAX_RUNS, "-X", "--sweep-max-runs",
     CliOptionValueKind::RequiredValue, CliOptionArity::Single},
}};

const std::string& allowed_options_hint() {
  static const std::string hint =
      "-P/--patterns, -b/--buffer-size, -t/--threads, -i/--iterations, "
      "-r/--count, --seed, -u/--non-cacheable, -o/--output, -S/--sweep, "
      "-X/--sweep-max-runs, -h/--help";
  return hint;
}

bool parse_strict_uint64_bounded(const std::string& option,
                                 const std::string& token,
                                 uint64_t min_val,
                                 uint64_t max_val,
                                 const char* range_reason,
                                 uint64_t& out_value) {
  uint64_t parsed = 0;
  const StrictIntegerParseStatus status =
      parse_strict_unsigned_decimal(token, parsed);
  if (status != StrictIntegerParseStatus::Success) {
    std::cerr << Messages::error_prefix()
              << Messages::error_invalid_value(
                     option, token,
                     strict_unsigned_decimal_error_reason(status))
              << std::endl;
    return false;
  }
  if (parsed < min_val || parsed > max_val) {
    std::cerr << Messages::error_prefix()
              << Messages::error_invalid_value(option, token, range_reason)
              << std::endl;
    return false;
  }
  out_value = parsed;
  return true;
}

bool parse_sweep_spec(const std::string& raw_spec,
                      PatternsSweepSpec& out_spec) {
  ParsedSweepText parsed;
  try {
    parsed = parse_sweep_text(raw_spec);
  } catch (const std::invalid_argument& err) {
    std::cerr << Messages::error_prefix()
              << Messages::error_invalid_value("--sweep", raw_spec, err.what())
              << std::endl;
    return false;
  }

  out_spec = PatternsSweepSpec{};
  out_spec.parameter_name = parsed.key;
  if (parsed.key == "buffer-size") {
    out_spec.parameter = PatternsSweepParameter::BufferSizeMb;
  } else if (parsed.key == "threads") {
    out_spec.parameter = PatternsSweepParameter::Threads;
  } else {
    std::cerr << Messages::error_prefix()
              << Messages::error_sweep_parameter_not_allowed(parsed.key,
                                                             kPatternsModeName)
              << std::endl;
    return false;
  }

  out_spec.values.reserve(parsed.values.size());
  for (const std::string& raw_val : parsed.values) {
    PatternsSweepValue typed;
    typed.raw_value = raw_val;
    if (out_spec.parameter == PatternsSweepParameter::BufferSizeMb) {
      uint64_t v = 0;
      if (!parse_strict_uint64_bounded(
              "--sweep", raw_val, 1, Constants::LLM_JSON_MAX_SAFE_INTEGER,
              "must be between 1 and 9007199254740991", v)) {
        return false;
      }
      typed.integer_value = static_cast<size_t>(v);
    } else {
      uint64_t v = 0;
      if (!parse_strict_uint64_bounded(
              "--sweep", raw_val, 1,
              static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()),
              "must be between 1 and 4294967295", v)) {
        return false;
      }
      typed.integer_value = static_cast<size_t>(v);
    }
    out_spec.values.push_back(std::move(typed));
  }
  return true;
}

size_t detect_available_workers() {
  if (parser_hooks_active) {
    return std::max<size_t>(1, active_parser_hooks.detected_workers);
  }
  const int cores = get_total_logical_cores();
  return cores > 0 ? static_cast<size_t>(cores) : 1;
}

uint64_t generate_base_seed() {
  if (parser_hooks_active) {
    const uint64_t s = active_parser_hooks.generated_seed;
    return SeedUtils::generate_seed([s]() { return s; });
  }
  return SeedUtils::generate_seed();
}

PatternsSystemInfo capture_system_info(PatternsResultMetadata& metadata) {
  PatternsSystemInfo info;
  if (parser_hooks_active) {
    metadata.processor_name = "Test Apple Silicon";
    metadata.macos_version = "26.0";
    metadata.performance_core_count =
        static_cast<int>(active_parser_hooks.detected_workers);
    metadata.efficiency_core_count = 0;
    metadata.logical_core_count =
        static_cast<int>(active_parser_hooks.detected_workers);
    metadata.page_size_bytes = active_parser_hooks.page_size_bytes;
    metadata.l1_data_cache_bytes = active_parser_hooks.l1_cache_bytes;
    metadata.l2_data_cache_bytes = active_parser_hooks.l2_cache_bytes;
    if (active_parser_hooks.use_memory_override) {
      metadata.available_memory_bytes =
          active_parser_hooks.available_memory_bytes;
      metadata.available_memory_source =
          metadata.available_memory_bytes == 0
              ? "unavailable"
              : "mach-free-plus-inactive-rounded-mib";
    } else {
      const unsigned long avail_mb = get_available_memory_mb();
      if (avail_mb > 0) {
        NumericUtils::checked_multiply(static_cast<size_t>(avail_mb),
                                       Constants::BYTES_PER_MB,
                                       metadata.available_memory_bytes);
      } else {
        metadata.available_memory_source = "unavailable";
      }
    }
  } else {
    metadata.processor_name = get_processor_name();
    metadata.macos_version = get_macos_version();
    metadata.performance_core_count = get_performance_cores();
    metadata.efficiency_core_count = get_efficiency_cores();
    metadata.logical_core_count = get_total_logical_cores();
    metadata.page_size_bytes = get_system_page_size_bytes();
    metadata.l1_data_cache_bytes = get_l1_cache_size();
    metadata.l2_data_cache_bytes = get_l2_cache_size();
    const unsigned long avail_mb = get_available_memory_mb();
    if (avail_mb == 0) {
      metadata.available_memory_source = "unavailable";
    } else if (!NumericUtils::checked_multiply(static_cast<size_t>(avail_mb),
                                               Constants::BYTES_PER_MB,
                                               metadata.available_memory_bytes)) {
      metadata.available_memory_bytes = 0;
      metadata.available_memory_source = "unavailable";
    }
  }
  info.page_size_bytes = metadata.page_size_bytes;
  info.available_memory_bytes = metadata.available_memory_bytes;
  return info;
}

void report_json_boundary_failure(const std::string& raw_output,
                                  const std::string& details) noexcept {
  try {
    std::cerr << Messages::error_prefix();
    if (raw_output == "-") {
      std::cerr << Messages::error_json_stdout_write_failed(details);
    } else {
      std::cerr << Messages::error_file_write_failed(raw_output, details);
    }
    std::cerr << std::endl;
  } catch (...) {
  }
}

void report_command_exception(const std::string& details) noexcept {
  try {
    std::cerr << Messages::error_prefix()
              << Messages::error_command_execution_exception("patterns", details)
              << std::endl;
  } catch (...) {
  }
}

}  // namespace

PatternsWorkPlanConfig PatternsConfig::to_work_plan_config() const {
  PatternsWorkPlanConfig wp;
  wp.buffer_size_mb = buffer_size_mb;
  wp.requested_workers = requested_workers;
  wp.available_workers = available_workers;
  wp.user_specified_workers = user_specified_workers;
  wp.iterations = iterations;
  wp.loop_count = loop_count;
  wp.non_cacheable = non_cacheable;
  wp.seed = seed;
  wp.collect_json_for_sweep = collect_json_for_sweep;
  wp.output_file = output_file;
  wp.argv = argv;
  return wp;
}

void set_patterns_parser_test_hooks(const PatternsParserTestHooks* hooks) {
  if (hooks == nullptr) {
    active_parser_hooks = PatternsParserTestHooks{};
    parser_hooks_active = false;
  } else {
    active_parser_hooks = *hooks;
    parser_hooks_active = true;
  }
}

void set_patterns_command_test_hooks(const PatternsCommandTestHooks* hooks) {
  if (hooks == nullptr) {
    active_command_hooks = PatternsCommandTestHooks{};
    command_hooks_active = false;
  } else {
    active_command_hooks = *hooks;
    command_hooks_active = true;
  }
}

int parse_patterns_cli(int argc, char* argv[], PatternsCliOptions& options) {
  options = PatternsCliOptions{};
  if (argc > 0 && argv != nullptr) {
    options.argv.reserve(static_cast<size_t>(argc));
    for (int i = 0; i < argc; ++i) {
      if (argv[i] != nullptr) {
        options.argv.emplace_back(argv[i]);
      }
    }
  }

  // Stage 1: Exact-whitelist syntax parsing.
  const CliParseResult parsed =
      parse_cli_options(argc, argv, kPatternsOptionSpecs);
  if (!parsed.ok) {
    std::cerr << Messages::error_prefix()
              << format_cli_parse_error(parsed, &allowed_options_hint())
              << std::endl;
    return EXIT_FAILURE;
  }

  // Stage 2: Value parsing & range bounds before help check.
  for (const CliParsedOption& opt : parsed.options) {
    switch (opt.id) {
      case OPT_PATTERNS:
        break;
      case OPT_HELP:
        options.help_requested = true;
        break;
      case OPT_BUFFER_SIZE: {
        uint64_t val = 0;
        if (!parse_strict_uint64_bounded(
                opt.canonical_name, opt.value, 1,
                Constants::LLM_JSON_MAX_SAFE_INTEGER,
                "must be between 1 and 9007199254740991", val)) {
          return EXIT_FAILURE;
        }
        options.buffer_size_mb = static_cast<size_t>(val);
        options.user_specified_buffer_size = true;
        break;
      }
      case OPT_THREADS: {
        uint64_t val = 0;
        if (!parse_strict_uint64_bounded(
                opt.canonical_name, opt.value, 1,
                static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()),
                "must be between 1 and 4294967295", val)) {
          return EXIT_FAILURE;
        }
        options.threads = static_cast<size_t>(val);
        options.user_specified_threads = true;
        break;
      }
      case OPT_ITERATIONS: {
        uint64_t val = 0;
        if (!parse_strict_uint64_bounded(
                opt.canonical_name, opt.value, 1,
                static_cast<uint64_t>(kPatternMaxPasses),
                "must be between 1 and 1000000000", val)) {
          return EXIT_FAILURE;
        }
        options.iterations = static_cast<size_t>(val);
        options.user_specified_iterations = true;
        break;
      }
      case OPT_COUNT: {
        uint64_t val = 0;
        if (!parse_strict_uint64_bounded(
                opt.canonical_name, opt.value, 1,
                Constants::LLM_JSON_MAX_SAFE_INTEGER,
                "must be between 1 and 9007199254740991", val)) {
          return EXIT_FAILURE;
        }
        options.loop_count = static_cast<size_t>(val);
        break;
      }
      case OPT_SEED: {
        uint64_t val = 0;
        if (!parse_strict_uint64_bounded(
                opt.canonical_name, opt.value, 0,
                std::numeric_limits<uint64_t>::max(),
                "must be a valid unsigned 64-bit integer", val)) {
          return EXIT_FAILURE;
        }
        options.seed = val;
        options.user_specified_seed = true;
        break;
      }
      case OPT_NON_CACHEABLE:
        options.non_cacheable = true;
        break;
      case OPT_OUTPUT:
        options.output_file = opt.value;
        break;
      case OPT_SWEEP: {
        PatternsSweepSpec spec;
        if (!parse_sweep_spec(opt.value, spec)) {
          return EXIT_FAILURE;
        }
        for (const auto& existing : options.sweep_specs) {
          if (existing.parameter == spec.parameter) {
            std::cerr << Messages::error_prefix()
                      << Messages::error_duplicate_sweep_parameter(
                             spec.parameter_name)
                      << std::endl;
            return EXIT_FAILURE;
          }
        }
        options.sweep_specs.push_back(std::move(spec));
        options.run_sweep = true;
        break;
      }
      case OPT_SWEEP_MAX_RUNS: {
        uint64_t val = 0;
        if (!parse_strict_uint64_bounded(
                opt.canonical_name, opt.value, 1,
                Constants::LLM_JSON_MAX_SAFE_INTEGER,
                "must be between 1 and 9007199254740991", val)) {
          return EXIT_FAILURE;
        }
        options.sweep_max_runs = static_cast<size_t>(val);
        break;
      }
    }
  }

  // Stage 3: Isolated help (before OS queries, sweep checks, or file I/O).
  if (options.help_requested) {
    print_patterns_help(cli_program_name(argc, argv), std::cout);
    return EXIT_SUCCESS;
  }

  // Stage 4: Sweep validation.
  if (options.run_sweep) {
    if (options.output_file.empty()) {
      std::cerr << Messages::error_prefix()
                << Messages::error_sweep_requires_output() << std::endl;
      return EXIT_FAILURE;
    }
    for (const auto& spec : options.sweep_specs) {
      if (spec.parameter == PatternsSweepParameter::BufferSizeMb &&
          options.user_specified_buffer_size) {
        std::cerr << Messages::error_prefix()
                  << Messages::error_sweep_option_conflict("--buffer-size",
                                                           spec.parameter_name)
                  << std::endl;
        return EXIT_FAILURE;
      }
      if (spec.parameter == PatternsSweepParameter::Threads &&
          options.user_specified_threads) {
        std::cerr << Messages::error_prefix()
                  << Messages::error_sweep_option_conflict("--threads",
                                                           spec.parameter_name)
                  << std::endl;
        return EXIT_FAILURE;
      }
    }
    const size_t run_count =
        calculate_sweep_run_count_from_specs(options.sweep_specs);
    if (run_count == 0) {
      std::cerr << Messages::error_prefix()
                << Messages::error_sweep_requires_parameter() << std::endl;
      return EXIT_FAILURE;
    }
    if (run_count > options.sweep_max_runs) {
      std::cerr << Messages::error_prefix()
                << Messages::error_sweep_too_many_runs(run_count,
                                                       options.sweep_max_runs)
                << std::endl;
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}

int build_patterns_config(const PatternsCliOptions& options,
                          PatternsConfig& config) {
  config = PatternsConfig{};
  config.help_printed = options.help_requested;
  config.argv = options.argv;
  if (options.help_requested) {
    return EXIT_SUCCESS;
  }

  config.buffer_size_mb = options.buffer_size_mb;
  config.iterations = options.iterations;
  config.loop_count = options.loop_count;
  config.non_cacheable = options.non_cacheable;
  config.output_file = options.output_file;
  config.run_sweep = options.run_sweep;
  config.sweep_max_runs = options.sweep_max_runs;
  config.sweep_specs = options.sweep_specs;
  config.user_specified_buffer_size = options.user_specified_buffer_size;
  config.user_specified_iterations = options.user_specified_iterations;
  config.user_specified_seed = options.user_specified_seed;
  config.seed = options.user_specified_seed ? *options.seed : generate_base_seed();

  config.available_workers = detect_available_workers();
  if (options.user_specified_threads) {
    config.user_specified_workers = true;
    config.requested_workers = *options.threads;
  } else {
    config.user_specified_workers = false;
    config.requested_workers = config.available_workers;
  }

  for (const auto& spec : config.sweep_specs) {
    if (spec.parameter == PatternsSweepParameter::BufferSizeMb) {
      config.swept_buffer_size = true;
    } else if (spec.parameter == PatternsSweepParameter::Threads) {
      config.swept_threads = true;
      config.user_specified_workers = true;
    }
  }

  if (config.run_sweep) {
    PatternsResultMetadata dummy_meta;
    const PatternsSystemInfo sys_info = capture_system_info(dummy_meta);
    if (!preflight_patterns_sweep(config, sys_info)) {
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}

int execute_patterns_run(const PatternsConfig& config,
                         JsonOutputSession* output_session,
                         bool install_signal_mask,
                         bool print_banner,
                         nlohmann::ordered_json* collected_json,
                         std::string* failure_reason,
                         bool* preparation_interrupted) noexcept {
  const auto stop_check = []() {
    if (command_hooks_active && active_command_hooks.stop_requested) {
      return active_command_hooks.stop_requested();
    }
    return signal_received();
  };
  const auto record_failure = [&](std::string_view reason) {
    if (failure_reason != nullptr && !reason.empty()) {
      *failure_reason = std::string(reason);
    }
  };
  if (collected_json != nullptr) {
    *collected_json = nlohmann::ordered_json();
  }
  if (failure_reason != nullptr) {
    failure_reason->clear();
  }
  if (preparation_interrupted != nullptr) {
    *preparation_interrupted = false;
  }

  int run_status = EXIT_FAILURE;
  const PatternsWorkPlanConfig wp_cfg = config.to_work_plan_config();
  PatternsWorkPlan work_plan;
  PatternsResourceSet resources;
  PatternsResultMetadata metadata;
  PatternsResult result;

  try {
    if (print_banner) {
      print_runtime_banner();
    }
    const std::string binary_sha256 = capture_binary_sha256();
    metadata.build_manifest =
        build_runtime_provenance_manifest_json(binary_sha256);
    metadata.timestamp = build_utc_timestamp();
    metadata.user_specified_seed = config.user_specified_seed;
    metadata.main_thread_qos =
        prepare_main_thread_benchmark_qos(MainThreadQosSetter{}, false);

    std::optional<BenchmarkSignalMaskGuard> signal_guard;
    if (install_signal_mask) {
      signal_guard.emplace();
    }

    metadata.environment_start = capture_host_environment();
    metadata.environment_end = metadata.environment_start;
    const PatternsSystemInfo sys_info = capture_system_info(metadata);

    work_plan = build_patterns_work_plan(wp_cfg, sys_info);
    if (!work_plan.valid) {
      record_failure(work_plan.reason_code);
      std::cerr << Messages::error_prefix()
                << "Patterns work plan rejected: " << work_plan.reason_code
                << std::endl;
      initialize_patterns_setup_failure_result(
          wp_cfg, work_plan, PatternsRunStatus::Failed, work_plan.reason_code,
          false, result);
      metadata.environment_end = capture_host_environment();
      if (output_session != nullptr &&
          output_session->kind() != JsonOutputKind::Disabled) {
        const nlohmann::ordered_json payload = build_patterns_json(
            config, work_plan, resources, metadata, result);
        if (output_session->write_final(payload, false) != EXIT_SUCCESS) {
          release_patterns_resources(resources);
          return EXIT_FAILURE;
        }
      } else if (collected_json != nullptr) {
        *collected_json = build_patterns_json(
            config, work_plan, resources, metadata, result);
      }
      release_patterns_resources(resources);
      return EXIT_FAILURE;
    }

    if (!prepare_patterns_resources(work_plan, config.non_cacheable, resources,
                                    stop_check)) {
      record_failure(resources.reason_code);
      const bool interrupted_in_prep =
          resources.reason_code ==
          PatternsExecutorReason::PREPARATION_INTERRUPTED;
      if (collected_json != nullptr && interrupted_in_prep) {
        if (preparation_interrupted != nullptr) {
          *preparation_interrupted = true;
        }
        std::cout << Messages::msg_interrupted_by_user() << std::endl;
        release_patterns_resources(resources);
        return EXIT_SUCCESS;
      }
      if (interrupted_in_prep) {
        std::cout << Messages::msg_interrupted_by_user() << std::endl;
        initialize_patterns_setup_failure_result(
            wp_cfg, work_plan, PatternsRunStatus::Interrupted,
            "interruption-requested", true, result);
      } else {
        std::cerr << Messages::error_prefix()
                  << "Patterns resource preparation failed: "
                  << resources.reason_code << std::endl;
        initialize_patterns_setup_failure_result(
            wp_cfg, work_plan, PatternsRunStatus::Failed, resources.reason_code,
            false, result);
      }
      metadata.environment_end = capture_host_environment();
      if (output_session != nullptr &&
          output_session->kind() != JsonOutputKind::Disabled) {
        const nlohmann::ordered_json payload = build_patterns_json(
            config, work_plan, resources, metadata, result);
        if (output_session->write_final(payload, false) != EXIT_SUCCESS) {
          release_patterns_resources(resources);
          return EXIT_FAILURE;
        }
      } else if (collected_json != nullptr) {
        *collected_json = build_patterns_json(
            config, work_plan, resources, metadata, result);
      }
      release_patterns_resources(resources);
      return interrupted_in_prep ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    if (print_banner) {
      print_patterns_configuration(wp_cfg, work_plan, metadata);
    }

    PatternsRunnerHooks hooks;
    hooks.stop_requested = stop_check;
    if (output_session != nullptr) {
      hooks.progress_snapshots = output_session->persists_checkpoints();
      hooks.observe_file_writes = [&]() {
        return std::make_pair(output_session->file_writer_attempts(),
                              output_session->successful_file_writes());
      };
      hooks.checkpoint = [&](const PatternsResult& snapshot,
                             PatternsCheckpointKind kind) {
        if (kind == PatternsCheckpointKind::Progress) {
          return output_session->checkpoint([&]() {
            metadata.environment_end = capture_host_environment();
            return build_patterns_json(config, work_plan, resources, metadata,
                                       snapshot);
          });
        }
        if (output_session->kind() == JsonOutputKind::File) {
          return output_session->checkpoint([&]() {
            metadata.environment_end = capture_host_environment();
            return build_patterns_json(config, work_plan, resources, metadata,
                                       snapshot);
          });
        }
        return EXIT_SUCCESS;
      };
    }

    if (command_hooks_active && active_command_hooks.before_runner) {
      active_command_hooks.before_runner(work_plan);
    }

    run_status =
        run_patterns_suite(wp_cfg, work_plan, resources, result, hooks);
    if (command_hooks_active && active_command_hooks.after_initialized_runner) {
      active_command_hooks.after_initialized_runner();
    }

    if (output_session == nullptr ||
        output_session->kind() != JsonOutputKind::File ||
        !result.terminal_checkpoint_completed) {
      metadata.environment_end = capture_host_environment();
    }

    if (print_banner || output_session != nullptr) {
      print_patterns_console_summary(wp_cfg, work_plan, resources, metadata,
                                     result);
    }

    if (output_session != nullptr &&
        output_session->kind() == JsonOutputKind::File &&
        result.terminal_checkpoint_completed && !result.checkpoint_failed) {
      std::cout << Messages::msg_results_saved_to(config.output_file)
                << std::endl;
    }

    if (result.status == PatternsRunStatus::Interrupted) {
      std::cout << Messages::msg_interrupted_by_user() << std::endl;
    } else if (run_status != EXIT_SUCCESS) {
      record_failure(result.reason_code);
      std::cerr << Messages::error_prefix()
                << "Patterns run failed (" << result.reason_code << ")."
                << std::endl;
    }
  } catch (const std::exception& err) {
    record_failure("runner-exception");
    report_command_exception(err.what());
    release_patterns_resources(resources);
    return EXIT_FAILURE;
  } catch (...) {
    record_failure("runner-unknown-exception");
    report_command_exception("");
    release_patterns_resources(resources);
    return EXIT_FAILURE;
  }

  nlohmann::ordered_json built_json;
  bool json_built = false;
  if (collected_json != nullptr ||
      (output_session != nullptr &&
       output_session->kind() == JsonOutputKind::Stdout &&
       result.initialized)) {
    try {
      built_json = build_patterns_json(config, work_plan, resources, metadata,
                                       result);
      json_built = true;
    } catch (const std::exception& err) {
      report_json_boundary_failure(
          config.output_file,
          Messages::error_json_payload_construction_failed(err.what()));
      release_patterns_resources(resources);
      return EXIT_FAILURE;
    } catch (...) {
      report_json_boundary_failure(
          config.output_file,
          Messages::error_json_payload_construction_failed(""));
      release_patterns_resources(resources);
      return EXIT_FAILURE;
    }
  }

  release_patterns_resources(resources);

  if (collected_json != nullptr && json_built) {
    *collected_json = std::move(built_json);
  }
  if (output_session != nullptr &&
      output_session->kind() == JsonOutputKind::Stdout && json_built) {
    if (output_session->write_final(built_json) != EXIT_SUCCESS) {
      return EXIT_FAILURE;
    }
  }
  return run_status;
}

int run_patterns_collect(const PatternsConfig& config,
                         nlohmann::ordered_json& result_json,
                         std::string* failure_reason,
                         bool* preparation_interrupted) {
  return execute_patterns_run(config, /*output_session=*/nullptr,
                              /*install_signal_mask=*/false,
                              /*print_banner=*/false, &result_json,
                              failure_reason, preparation_interrupted);
}

int run_patterns_mode(int argc, char* argv[]) {
  install_signal_handlers();

  PatternsCliOptions options;
  if (parse_patterns_cli(argc, argv, options) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (options.help_requested) {
    return EXIT_SUCCESS;
  }

  PatternsConfig config;
  if (build_patterns_config(options, config) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }

  JsonOutputTarget output_target;
  try {
    output_target = make_json_output_target(
        config.output_file,
        JsonFilePathPolicy::ResolveAgainstCurrentDirectory);
  } catch (const std::exception& err) {
    report_json_boundary_failure(
        config.output_file,
        Messages::error_json_output_initialization_failed(err.what()));
    return EXIT_FAILURE;
  } catch (...) {
    report_json_boundary_failure(
        config.output_file,
        Messages::error_json_output_initialization_failed(""));
    return EXIT_FAILURE;
  }

  std::optional<JsonOutputSession> output_session;
  try {
    output_session.emplace(std::move(output_target));
  } catch (const std::exception& err) {
    report_json_boundary_failure(
        config.output_file,
        Messages::error_json_output_initialization_failed(err.what()));
    return EXIT_FAILURE;
  } catch (...) {
    report_json_boundary_failure(
        config.output_file,
        Messages::error_json_output_initialization_failed(""));
    return EXIT_FAILURE;
  }

  if (config.run_sweep) {
    try {
      const SweepExecutionResult execution =
          run_patterns_sweep(config, *output_session);
      if (output_session->kind() == JsonOutputKind::Stdout &&
          !execution.output_json.empty() &&
          output_session->write_final(execution.output_json) != EXIT_SUCCESS) {
        return EXIT_FAILURE;
      }
      return execution.exit_code;
    } catch (const std::exception& err) {
      report_command_exception(err.what());
      return EXIT_FAILURE;
    } catch (...) {
      report_command_exception("");
      return EXIT_FAILURE;
    }
  }

  return execute_patterns_run(config, &*output_session,
                              /*install_signal_mask=*/true,
                              /*print_banner=*/true);
}

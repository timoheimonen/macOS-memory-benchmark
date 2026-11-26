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
 * @file core_to_core_latency_cli.cpp
 * @brief CLI parsing for standalone core-to-core latency mode
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 *
 * Parses and validates mode-specific command line options for `-C, --analyze-core2core`.
 * This parser intentionally accepts only a small, explicit option set so the
 * standalone mode remains isolated from standard benchmark orchestration flags.
 */

#include "benchmark/core_to_core_latency.h"
#include "benchmark/core_to_core_sweep_runner.h"

#include <array>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "core/config/cli_parser.h"
#include "core/config/config.h"
#include "core/config/constants.h"
#include "core/config/sweep_utils.h"
#include "core/signal/signal_handler.h"
#include "output/console/messages/messages_api.h"
#include "output/console/output_printer.h"
#include "output/json/json_output/json_output_session.h"

namespace {

constexpr const char* OPT_ANALYZE_CORE_TO_CORE_SHORT = "-C";
constexpr const char* OPT_ANALYZE_CORE_TO_CORE_LONG = "--analyze-core2core";
constexpr const char* OPT_COUNT_SHORT = "-r";
constexpr const char* OPT_COUNT_LONG = "--count";
constexpr const char* OPT_HELP_SHORT = "-h";
constexpr const char* OPT_HELP_LONG = "--help";
constexpr const char* OPT_LATENCY_SAMPLES_SHORT = "-n";
constexpr const char* OPT_LATENCY_SAMPLES_LONG = "--latency-samples";
constexpr const char* OPT_OUTPUT_SHORT = "-o";
constexpr const char* OPT_OUTPUT_LONG = "--output";
constexpr const char* OPT_SWEEP_SHORT = "-S";
constexpr const char* OPT_SWEEP_LONG = "--sweep";
constexpr const char* OPT_SWEEP_MAX_RUNS_SHORT = "-X";
constexpr const char* OPT_SWEEP_MAX_RUNS_LONG = "--sweep-max-runs";

enum CoreToCoreOptionId {
  OPT_ID_ANALYZE_CORE_TO_CORE = 1,
  OPT_ID_HELP,
  OPT_ID_OUTPUT,
  OPT_ID_SWEEP,
  OPT_ID_SWEEP_MAX_RUNS,
  OPT_ID_COUNT,
  OPT_ID_LATENCY_SAMPLES,
};

constexpr std::array<CliOptionSpec, 7> kCoreToCoreOptionSpecs{{
    {OPT_ID_ANALYZE_CORE_TO_CORE, OPT_ANALYZE_CORE_TO_CORE_SHORT, OPT_ANALYZE_CORE_TO_CORE_LONG,
     CliOptionValueKind::None, CliOptionArity::Single},
    {OPT_ID_HELP, OPT_HELP_SHORT, OPT_HELP_LONG, CliOptionValueKind::None, CliOptionArity::Single},
    {OPT_ID_OUTPUT, OPT_OUTPUT_SHORT, OPT_OUTPUT_LONG, CliOptionValueKind::RequiredValue, CliOptionArity::Single},
    {OPT_ID_SWEEP, OPT_SWEEP_SHORT, OPT_SWEEP_LONG, CliOptionValueKind::RequiredValue, CliOptionArity::Repeatable},
    {OPT_ID_SWEEP_MAX_RUNS, OPT_SWEEP_MAX_RUNS_SHORT, OPT_SWEEP_MAX_RUNS_LONG,
     CliOptionValueKind::RequiredValue, CliOptionArity::Single},
    {OPT_ID_COUNT, OPT_COUNT_SHORT, OPT_COUNT_LONG, CliOptionValueKind::RequiredValue, CliOptionArity::Single},
    {OPT_ID_LATENCY_SAMPLES, OPT_LATENCY_SAMPLES_SHORT, OPT_LATENCY_SAMPLES_LONG,
     CliOptionValueKind::RequiredValue, CliOptionArity::Single},
}};

void report_core_to_core_command_exception(const std::string& details) noexcept {
  try {
    std::cerr << Messages::error_prefix()
              << Messages::error_command_execution_exception("Core-to-core analysis", details)
              << std::endl;
  } catch (...) {
    // A secondary diagnostic failure must not escape the command boundary.
  }
}

void report_json_output_initialization_failure(const std::string& raw_output,
                                               const std::string& details) noexcept {
  try {
    std::cerr << Messages::error_prefix();
    if (raw_output == "-") {
      std::cerr << Messages::error_json_stdout_write_failed(
          Messages::error_json_output_initialization_failed(details));
    } else {
      std::cerr << Messages::error_file_write_failed(
          raw_output,
          Messages::error_json_output_initialization_failed(details));
    }
    std::cerr << std::endl;
  } catch (...) {
    // A secondary diagnostic failure must not escape the command boundary.
  }
}

// Shared validator for positive integer options used by standalone mode flags.
bool parse_positive_int_option(const std::string& option,
                               const std::string& value,
                               int& out_value) {
  long long parsed = 0;
  const StrictIntegerParseStatus parse_status =
      parse_strict_signed_decimal(value, parsed);
  if (parse_status != StrictIntegerParseStatus::Success) {
    std::cerr << Messages::error_prefix()
              << Messages::error_invalid_value(
                     option, value,
                     strict_signed_decimal_error_reason(parse_status))
              << std::endl;
    return false;
  }

  if (parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
    std::cerr << Messages::error_prefix()
              << Messages::error_invalid_value(
                     option,
                     value,
                     Messages::reason_positive_int_range(std::numeric_limits<int>::max()))
              << std::endl;
    return false;
  }

  out_value = static_cast<int>(parsed);
  return true;
}

bool core_to_core_sweep_parameter_from_string(const std::string& key,
                                              CoreToCoreSweepParameter& out_parameter,
                                              std::string& out_name) {
  if (key == "count") {
    out_parameter = CoreToCoreSweepParameter::Count;
    out_name = "count";
    return true;
  }
  if (key == "latency-samples") {
    out_parameter = CoreToCoreSweepParameter::LatencySamples;
    out_name = "latency-samples";
    return true;
  }
  return false;
}

bool parse_core_to_core_sweep_spec(const std::string& spec_text,
                                   CoreToCoreSweepSpec& out_spec) {
  ParsedSweepText parsed_text;
  try {
    parsed_text = parse_sweep_text(spec_text);
  } catch (const std::invalid_argument& error) {
    std::cerr << Messages::error_prefix()
              << Messages::error_invalid_value(OPT_SWEEP_LONG, spec_text, error.what())
              << std::endl;
    return false;
  }

  CoreToCoreSweepSpec spec;
  if (!core_to_core_sweep_parameter_from_string(parsed_text.key, spec.parameter, spec.parameter_name)) {
    std::cerr << Messages::error_prefix()
              << Messages::error_invalid_value(
                     OPT_SWEEP_LONG, spec_text,
                     Messages::reason_unsupported_core_to_core_sweep_parameter(parsed_text.key))
              << std::endl;
    return false;
  }

  for (const std::string& raw_value : parsed_text.values) {
    int parsed = 0;
    const std::string option_name =
        (spec.parameter == CoreToCoreSweepParameter::Count) ? OPT_COUNT_LONG : OPT_LATENCY_SAMPLES_LONG;
    if (!parse_positive_int_option(option_name, raw_value, parsed)) {
      return false;
    }
    spec.values.push_back(CoreToCoreSweepValue{raw_value, parsed});
  }

  out_spec = std::move(spec);
  return true;
}

}  // namespace

int parse_core_to_core_mode_arguments(int argc, char* argv[], CoreToCoreLatencyConfig& config) {
  // Always start from centralized core-to-core defaults.
  config = CoreToCoreLatencyConfig{};

  const CliParseResult parsed = parse_cli_options(argc, argv, kCoreToCoreOptionSpecs);
  if (!parsed.ok) {
    std::cerr << Messages::error_prefix()
              << format_cli_parse_error(parsed, &Messages::error_analyze_core_to_core_must_be_used_alone())
              << std::endl;
    return EXIT_FAILURE;
  }

  bool mode_seen = false;
  bool count_seen = false;
  bool samples_seen = false;

  for (const CliParsedOption& option : parsed.options) {
    switch (option.id) {
      case OPT_ID_ANALYZE_CORE_TO_CORE:
        mode_seen = true;
        break;
      case OPT_ID_HELP:
        config.help_requested = true;
        break;
      case OPT_ID_OUTPUT:
        config.output_file = option.value;
        break;
      case OPT_ID_SWEEP: {
        CoreToCoreSweepSpec spec;
        if (!parse_core_to_core_sweep_spec(option.value, spec)) {
          return EXIT_FAILURE;
        }
        for (const CoreToCoreSweepSpec& existing_spec : config.sweep_specs) {
          if (existing_spec.parameter == spec.parameter) {
            std::cerr << Messages::error_prefix()
                      << Messages::error_duplicate_sweep_parameter(spec.parameter_name)
                      << std::endl;
            return EXIT_FAILURE;
          }
        }
        config.sweep_specs.push_back(std::move(spec));
        config.run_sweep = true;
        break;
      }
      case OPT_ID_SWEEP_MAX_RUNS: {
        int parsed_value = 0;
        if (!parse_positive_int_option(option.canonical_name, option.value, parsed_value)) {
          return EXIT_FAILURE;
        }
        config.sweep_max_runs = static_cast<size_t>(parsed_value);
        break;
      }
      case OPT_ID_COUNT:
        if (!parse_positive_int_option(option.canonical_name, option.value, config.loop_count)) {
          return EXIT_FAILURE;
        }
        count_seen = true;
        break;
      case OPT_ID_LATENCY_SAMPLES:
        if (!parse_positive_int_option(option.canonical_name, option.value, config.latency_sample_count)) {
          return EXIT_FAILURE;
        }
        samples_seen = true;
        break;
    }
  }

  if (!mode_seen) {
    std::cerr << Messages::error_prefix()
              << Messages::error_analyze_core_to_core_must_be_used_alone()
              << std::endl;
    return EXIT_FAILURE;
  }

  if (config.help_requested) {
    print_help(cli_program_name(argc, argv));
    return EXIT_SUCCESS;
  }

  if (config.run_sweep) {
    if (config.sweep_specs.empty()) {
      std::cerr << Messages::error_prefix()
                << Messages::error_sweep_requires_parameter()
                << std::endl;
      return EXIT_FAILURE;
    }
    for (const CoreToCoreSweepSpec& spec : config.sweep_specs) {
      if (spec.parameter == CoreToCoreSweepParameter::Count && count_seen) {
        std::cerr << Messages::error_prefix()
                  << Messages::error_sweep_option_conflict(OPT_COUNT_LONG, "count")
                  << std::endl;
        return EXIT_FAILURE;
      }
      if (spec.parameter == CoreToCoreSweepParameter::LatencySamples && samples_seen) {
        std::cerr << Messages::error_prefix()
                  << Messages::error_sweep_option_conflict(OPT_LATENCY_SAMPLES_LONG, "latency-samples")
                  << std::endl;
        return EXIT_FAILURE;
      }
    }
    if (config.output_file.empty()) {
      std::cerr << Messages::error_prefix()
                << Messages::error_sweep_requires_output()
                << std::endl;
      return EXIT_FAILURE;
    }
    const size_t run_count = calculate_core_to_core_sweep_run_count(config);
    if (run_count == 0 || run_count > config.sweep_max_runs) {
      std::cerr << Messages::error_prefix()
                << Messages::error_sweep_too_many_runs(run_count, config.sweep_max_runs)
                << std::endl;
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}

int run_core_to_core_latency_mode(int argc, char* argv[]) {
  CoreToCoreLatencyConfig config;
  int parse_result = EXIT_FAILURE;
  try {
    parse_result = parse_core_to_core_mode_arguments(argc, argv, config);
  } catch (const std::exception& error) {
    report_core_to_core_command_exception(error.what());
    return EXIT_FAILURE;
  } catch (...) {
    report_core_to_core_command_exception("");
    return EXIT_FAILURE;
  }
  if (parse_result != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }

  if (config.help_requested) {
    return EXIT_SUCCESS;
  }

  std::optional<JsonOutputSession> output_session;
  try {
    output_session.emplace(make_json_output_target(
        config.output_file,
        JsonFilePathPolicy::ResolveAgainstCurrentDirectory));
  } catch (const std::exception& error) {
    report_json_output_initialization_failure(config.output_file,
                                              error.what());
    return EXIT_FAILURE;
  } catch (...) {
    report_json_output_initialization_failure(config.output_file, "");
    return EXIT_FAILURE;
  }

  // Execute benchmark only after successful parse and non-help path.
  try {
    if (config.run_sweep) {
      const SweepExecutionResult execution =
          run_core_to_core_latency_sweep(config, *output_session);
      if (output_session->kind() == JsonOutputKind::Stdout &&
          !execution.output_json.empty() &&
          output_session->write_final(execution.output_json) != EXIT_SUCCESS) {
        return EXIT_FAILURE;
      }
      return execution.exit_code;
    }

    BenchmarkSignalMaskGuard signal_guard;
    return run_core_to_core_latency(config, *output_session);
  } catch (const std::exception& error) {
    report_core_to_core_command_exception(error.what());
  } catch (...) {
    report_core_to_core_command_exception("");
  }
  return EXIT_FAILURE;
}

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
 * Parses and validates mode-specific command line options for `-analyze-core2core`.
 * This parser intentionally accepts only a small, explicit option set so the
 * standalone mode remains isolated from standard benchmark orchestration flags.
 */

#include "benchmark/core_to_core_latency.h"
#include "benchmark/core_to_core_sweep_runner.h"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/config/constants.h"
#include "output/console/messages/messages_api.h"
#include "output/console/output_printer.h"

namespace {

// Shared validator for positive integer options used by standalone mode flags.
bool parse_positive_int_option(const std::string& option,
                               const std::string& value,
                               int& out_value,
                               const char* prog_name) {
  long long parsed = 0;
  try {
    parsed = std::stoll(value);
  } catch (const std::invalid_argument&) {
    std::cerr << Messages::error_prefix()
              << Messages::error_invalid_value(option, value, "must be an integer")
              << std::endl;
    print_usage(prog_name);
    return false;
  } catch (const std::out_of_range&) {
    std::cerr << Messages::error_prefix()
              << Messages::error_invalid_value(option, value, "out of range")
              << std::endl;
    print_usage(prog_name);
    return false;
  }

  if (parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
    std::cerr << Messages::error_prefix()
              << Messages::error_invalid_value(
                     option,
                     value,
                     "must be between 1 and " + std::to_string(std::numeric_limits<int>::max()))
              << std::endl;
    print_usage(prog_name);
    return false;
  }

  out_value = static_cast<int>(parsed);
  return true;
}

std::vector<std::string> split_comma_values(const std::string& input) {
  std::vector<std::string> values;
  std::stringstream stream(input);
  std::string item;
  while (std::getline(stream, item, ',')) {
    values.push_back(item);
  }
  return values;
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
                                   CoreToCoreSweepSpec& out_spec,
                                   const char* prog_name) {
  const size_t equals_pos = spec_text.find('=');
  if (equals_pos == std::string::npos || equals_pos == 0 || equals_pos == spec_text.size() - 1) {
    std::cerr << Messages::error_prefix()
              << Messages::error_invalid_value("-sweep", spec_text, "sweep must use key=value1,value2 syntax")
              << std::endl;
    print_usage(prog_name);
    return false;
  }

  const std::string key = spec_text.substr(0, equals_pos);
  const std::string value_text = spec_text.substr(equals_pos + 1);

  CoreToCoreSweepSpec spec;
  if (!core_to_core_sweep_parameter_from_string(key, spec.parameter, spec.parameter_name)) {
    std::cerr << Messages::error_prefix()
              << Messages::error_invalid_value("-sweep", spec_text, "unsupported core-to-core sweep parameter: " + key)
              << std::endl;
    print_usage(prog_name);
    return false;
  }

  const std::vector<std::string> raw_values = split_comma_values(value_text);
  if (raw_values.empty()) {
    std::cerr << Messages::error_prefix()
              << Messages::error_invalid_value("-sweep", spec_text, "sweep value list cannot be empty")
              << std::endl;
    print_usage(prog_name);
    return false;
  }

  for (const std::string& raw_value : raw_values) {
    if (raw_value.empty()) {
      std::cerr << Messages::error_prefix()
                << Messages::error_invalid_value("-sweep", spec_text, "sweep value list cannot contain empty values")
                << std::endl;
      print_usage(prog_name);
      return false;
    }

    int parsed = 0;
    const std::string option_name =
        (spec.parameter == CoreToCoreSweepParameter::Count) ? "-count" : "-latency-samples";
    if (!parse_positive_int_option(option_name, raw_value, parsed, prog_name)) {
      return false;
    }
    spec.values.push_back(CoreToCoreSweepValue{raw_value, parsed});
  }

  out_spec = spec;
  return true;
}

}  // namespace

int parse_core_to_core_mode_arguments(int argc, char* argv[], CoreToCoreLatencyConfig& config) {
  // Always start from centralized core-to-core defaults.
  config.loop_count = Constants::CORE_TO_CORE_DEFAULT_LOOP_COUNT;
  config.latency_sample_count = Constants::CORE_TO_CORE_DEFAULT_LATENCY_SAMPLE_COUNT;

  bool mode_seen = false;
  bool output_seen = false;
  bool count_seen = false;
  bool samples_seen = false;
  bool sweep_max_runs_seen = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];

    // Mode flag is required but does not consume a value.
    if (arg == "-analyze-core2core") {
      mode_seen = true;
      continue;
    }

    // Help is an early-exit path for this standalone parser.
    if (arg == "-h" || arg == "--help") {
      print_help(argv[0]);
      config.help_requested = true;
      return EXIT_SUCCESS;
    }

    // Optional JSON output target.
    if (arg == "-output") {
      if (output_seen) {
        std::cerr << Messages::error_prefix()
                  << Messages::error_duplicate_option("-output")
                  << std::endl;
        print_usage(argv[0]);
        return EXIT_FAILURE;
      }
      if (++i >= argc) {
        std::cerr << Messages::error_prefix()
                  << Messages::error_missing_value("-output")
                  << std::endl;
        print_usage(argv[0]);
        return EXIT_FAILURE;
      }
      config.output_file = argv[i];
      output_seen = true;
      continue;
    }

    if (arg == "-sweep") {
      if (++i >= argc) {
        std::cerr << Messages::error_prefix()
                  << Messages::error_missing_value("-sweep")
                  << std::endl;
        print_usage(argv[0]);
        return EXIT_FAILURE;
      }

      CoreToCoreSweepSpec spec;
      if (!parse_core_to_core_sweep_spec(argv[i], spec, argv[0])) {
        return EXIT_FAILURE;
      }
      config.sweep_specs.push_back(std::move(spec));
      config.run_sweep = true;
      continue;
    }

    if (arg == "-sweep-max-runs") {
      if (sweep_max_runs_seen) {
        std::cerr << Messages::error_prefix()
                  << Messages::error_duplicate_option("-sweep-max-runs")
                  << std::endl;
        print_usage(argv[0]);
        return EXIT_FAILURE;
      }
      if (++i >= argc) {
        std::cerr << Messages::error_prefix()
                  << Messages::error_missing_value("-sweep-max-runs")
                  << std::endl;
        print_usage(argv[0]);
        return EXIT_FAILURE;
      }
      int parsed = 0;
      if (!parse_positive_int_option("-sweep-max-runs", argv[i], parsed, argv[0])) {
        return EXIT_FAILURE;
      }
      config.sweep_max_runs = static_cast<size_t>(parsed);
      sweep_max_runs_seen = true;
      continue;
    }

    // Optional benchmark loop count override.
    if (arg == "-count") {
      if (count_seen) {
        std::cerr << Messages::error_prefix()
                  << Messages::error_duplicate_option("-count")
                  << std::endl;
        print_usage(argv[0]);
        return EXIT_FAILURE;
      }
      if (++i >= argc) {
        std::cerr << Messages::error_prefix()
                  << Messages::error_missing_value("-count")
                  << std::endl;
        print_usage(argv[0]);
        return EXIT_FAILURE;
      }
      if (!parse_positive_int_option("-count", argv[i], config.loop_count, argv[0])) {
        return EXIT_FAILURE;
      }
      count_seen = true;
      continue;
    }

    // Optional per-loop latency sample count override.
    if (arg == "-latency-samples") {
      if (samples_seen) {
        std::cerr << Messages::error_prefix()
                  << Messages::error_duplicate_option("-latency-samples")
                  << std::endl;
        print_usage(argv[0]);
        return EXIT_FAILURE;
      }
      if (++i >= argc) {
        std::cerr << Messages::error_prefix()
                  << Messages::error_missing_value("-latency-samples")
                  << std::endl;
        print_usage(argv[0]);
        return EXIT_FAILURE;
      }
      if (!parse_positive_int_option("-latency-samples",
                                     argv[i],
                                     config.latency_sample_count,
                                     argv[0])) {
        return EXIT_FAILURE;
      }
      samples_seen = true;
      continue;
    }

    // Any unknown or standard-mode option is rejected in standalone mode.
    std::cerr << Messages::error_prefix()
              << Messages::error_analyze_core_to_core_must_be_used_alone()
              << std::endl;
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  if (!mode_seen) {
    std::cerr << Messages::error_prefix()
              << Messages::error_analyze_core_to_core_must_be_used_alone()
              << std::endl;
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  if (config.run_sweep) {
    if (config.sweep_specs.empty()) {
      std::cerr << Messages::error_prefix()
                << Messages::error_sweep_requires_parameter()
                << std::endl;
      print_usage(argv[0]);
      return EXIT_FAILURE;
    }
    if (config.output_file.empty()) {
      std::cerr << Messages::error_prefix()
                << Messages::error_sweep_requires_output()
                << std::endl;
      print_usage(argv[0]);
      return EXIT_FAILURE;
    }
    const size_t run_count = calculate_core_to_core_sweep_run_count(config);
    if (run_count == 0 || run_count > config.sweep_max_runs) {
      std::cerr << Messages::error_prefix()
                << Messages::error_sweep_too_many_runs(run_count, config.sweep_max_runs)
                << std::endl;
      print_usage(argv[0]);
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}

int run_core_to_core_latency_mode(int argc, char* argv[]) {
  CoreToCoreLatencyConfig config;
  const int parse_result = parse_core_to_core_mode_arguments(argc, argv, config);
  if (parse_result != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }

  if (config.help_requested) {
    return EXIT_SUCCESS;
  }

  if (config.run_sweep) {
    return run_core_to_core_latency_sweep(config);
  }

  // Execute benchmark only after successful parse and non-help path.
  return run_core_to_core_latency(config);
}

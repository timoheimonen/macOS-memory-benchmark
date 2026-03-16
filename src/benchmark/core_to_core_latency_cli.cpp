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

#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

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

}  // namespace

int parse_core_to_core_mode_arguments(int argc, char* argv[], CoreToCoreLatencyConfig& config) {
  // Always start from centralized core-to-core defaults.
  config.loop_count = Constants::CORE_TO_CORE_DEFAULT_LOOP_COUNT;
  config.latency_sample_count = Constants::CORE_TO_CORE_DEFAULT_LATENCY_SAMPLE_COUNT;

  bool mode_seen = false;
  bool output_seen = false;
  bool count_seen = false;
  bool samples_seen = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];

    // Mode flag is required but does not consume a value.
    if (arg == "-analyze-core2core") {
      mode_seen = true;
      continue;
    }

    // Help is an early-exit path for this standalone parser.
    if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
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

  // Execute benchmark only after successful parse and non-help path.
  return run_core_to_core_latency(config);
}

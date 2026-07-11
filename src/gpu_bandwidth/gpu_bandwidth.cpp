// Copyright 2026 Timo Heimonen <timo.heimonen@proton.me>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

/**
 * @file gpu_bandwidth.cpp
 * @brief Standalone GPU memory-bandwidth CLI parsing and entry point
 */

#include "gpu_bandwidth/gpu_bandwidth.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include "core/config/config.h"
#include "core/config/version.h"
#include "core/signal/signal_handler.h"
#include "core/system/benchmark_qos.h"
#include "gpu_bandwidth/gpu_backend.h"
#include "gpu_bandwidth/gpu_runner.h"
#include "output/console/messages/messages_api.h"
#include "utils/numeric_utils.h"
#include "utils/seed_utils.h"

namespace {

GpuBandwidthParserTestHooks active_gpu_parser_hooks;
bool gpu_parser_hooks_active = false;

bool is_option(const std::string& argument, const char* short_option,
               const char* long_option) {
  return argument == short_option || argument == long_option;
}

bool report_duplicate(bool& seen, const char* canonical_option) {
  if (!seen) {
    seen = true;
    return false;
  }
  std::cerr << Messages::error_prefix()
            << Messages::error_duplicate_option(canonical_option) << std::endl;
  return true;
}

bool take_value(int argc, char* argv[], int& index,
                const char* canonical_option, std::string& value) {
  if (++index >= argc) {
    std::cerr << Messages::error_prefix()
              << Messages::error_missing_value(canonical_option) << std::endl;
    return false;
  }
  value = argv[index];
  return true;
}

bool parse_positive_size(const std::string& option, const std::string& token,
                         size_t& value) {
  long long parsed = 0;
  const StrictIntegerParseStatus status =
      parse_strict_signed_decimal(token, parsed);
  if (status != StrictIntegerParseStatus::Success || parsed <= 0) {
    const std::string reason = status == StrictIntegerParseStatus::Success
                                   ? Messages::gpu_reason_positive_integer()
                                   : strict_signed_decimal_error_reason(status);
    std::cerr << Messages::error_prefix()
              << Messages::error_invalid_value(option, token, reason)
              << std::endl;
    return false;
  }
  value = static_cast<size_t>(parsed);
  return true;
}

size_t maximum_explicit_gpu_passes(size_t buffer_size_bytes) {
  const GpuPassLimits copy_limits =
      calculate_gpu_pass_limits(buffer_size_bytes, GpuOperation::Copy);
  return copy_limits.valid ? copy_limits.effective_maximum_passes : 0;
}

void print_gpu_help(const char* program_name) {
  std::cout << Messages::usage_header(SOFTVERSION)
            << Messages::gpu_usage_options(program_name);
}

}  // namespace

void set_gpu_bandwidth_parser_test_hooks(
    const GpuBandwidthParserTestHooks* hooks) {
  if (hooks == nullptr) {
    active_gpu_parser_hooks = GpuBandwidthParserTestHooks{};
    gpu_parser_hooks_active = false;
    return;
  }
  active_gpu_parser_hooks = *hooks;
  gpu_parser_hooks_active = true;
}

int parse_gpu_bandwidth_arguments(int argc, char* argv[],
                                  GpuBandwidthConfig& config) {
  config = GpuBandwidthConfig{};
  config.argv.reserve(static_cast<size_t>(std::max(argc, 0)));
  for (int index = 0; index < argc; ++index) {
    config.argv.emplace_back(argv[index]);
  }

  bool mode_seen = false;
  bool buffer_seen = false;
  bool iterations_seen = false;
  bool count_seen = false;
  bool output_seen = false;
  bool seed_seen = false;
  bool help_seen = false;

  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (is_option(argument, "-G", "--gpu-bandwidth")) {
      if (report_duplicate(mode_seen, "--gpu-bandwidth")) {
        return EXIT_FAILURE;
      }
      continue;
    }
    if (is_option(argument, "-h", "--help")) {
      if (report_duplicate(help_seen, "--help")) {
        return EXIT_FAILURE;
      }
      config.help_printed = true;
      continue;
    }
    if (is_option(argument, "-b", "--buffer-size")) {
      if (report_duplicate(buffer_seen, "--buffer-size")) {
        return EXIT_FAILURE;
      }
      std::string token;
      if (!take_value(argc, argv, index, "--buffer-size", token)) {
        return EXIT_FAILURE;
      }
      long long parsed = 0;
      const StrictIntegerParseStatus status =
          parse_strict_signed_decimal(token, parsed);
      if (status != StrictIntegerParseStatus::Success || parsed < 0 ||
          static_cast<unsigned long long>(parsed) >
              std::numeric_limits<unsigned long>::max()) {
        const std::string reason =
            status == StrictIntegerParseStatus::Success
                ? Messages::gpu_reason_nonnegative_unsigned_long()
                : strict_signed_decimal_error_reason(status);
        std::cerr << Messages::error_prefix()
                  << Messages::error_invalid_value(argument, token, reason)
                  << std::endl;
        return EXIT_FAILURE;
      }
      config.buffer_size_mb = static_cast<unsigned long>(parsed);
      continue;
    }
    if (is_option(argument, "-i", "--iterations")) {
      if (report_duplicate(iterations_seen, "--iterations")) {
        return EXIT_FAILURE;
      }
      std::string token;
      if (!take_value(argc, argv, index, "--iterations", token) ||
          !parse_positive_size(argument, token, config.iterations)) {
        return EXIT_FAILURE;
      }
      config.user_specified_iterations = true;
      continue;
    }
    if (is_option(argument, "-r", "--count")) {
      if (report_duplicate(count_seen, "--count")) {
        return EXIT_FAILURE;
      }
      std::string token;
      if (!take_value(argc, argv, index, "--count", token) ||
          !parse_positive_size(argument, token, config.loop_count)) {
        return EXIT_FAILURE;
      }
      if (config.loop_count >
          static_cast<size_t>(std::numeric_limits<int>::max())) {
        std::cerr << Messages::error_prefix()
                  << Messages::error_invalid_value(
                         argument, token,
                         Messages::gpu_reason_loop_count_out_of_range())
                  << std::endl;
        return EXIT_FAILURE;
      }
      continue;
    }
    if (is_option(argument, "-o", "--output")) {
      if (report_duplicate(output_seen, "--output")) {
        return EXIT_FAILURE;
      }
      if (!take_value(argc, argv, index, "--output", config.output_file)) {
        return EXIT_FAILURE;
      }
      continue;
    }
    if (argument == "--seed") {
      if (report_duplicate(seed_seen, "--seed")) {
        return EXIT_FAILURE;
      }
      std::string token;
      if (!take_value(argc, argv, index, "--seed", token)) {
        return EXIT_FAILURE;
      }
      const StrictIntegerParseStatus status =
          parse_strict_unsigned_decimal(token, config.seed);
      if (status != StrictIntegerParseStatus::Success) {
        std::cerr << Messages::error_prefix()
                  << Messages::error_invalid_value(
                         argument, token,
                         strict_unsigned_decimal_error_reason(status))
                  << std::endl;
        return EXIT_FAILURE;
      }
      config.user_specified_seed = true;
      continue;
    }

    std::cerr << Messages::error_prefix()
              << Messages::error_gpu_bandwidth_must_be_used_alone()
              << std::endl;
    return EXIT_FAILURE;
  }

  if (!mode_seen) {
    std::cerr << Messages::error_prefix()
              << Messages::error_gpu_bandwidth_must_be_used_alone()
              << std::endl;
    return EXIT_FAILURE;
  }
  if (config.help_printed) {
    print_gpu_help(argc > 0 ? argv[0] : "memory_benchmark");
    return EXIT_SUCCESS;
  }
  if (config.buffer_size_mb < Constants::GPU_MIN_BUFFER_SIZE_MB) {
    std::cerr << Messages::error_prefix()
              << Messages::error_gpu_buffer_size_below_minimum(
                     config.buffer_size_mb,
                     Constants::GPU_MIN_BUFFER_SIZE_MB)
              << std::endl;
    return EXIT_FAILURE;
  }
  if (!NumericUtils::checked_multiply(
          static_cast<size_t>(config.buffer_size_mb),
          Constants::BYTES_PER_MB, config.buffer_size_bytes)) {
    std::cerr << Messages::error_prefix()
              << Messages::error_buffer_size_calculation(config.buffer_size_mb)
              << std::endl;
    return EXIT_FAILURE;
  }
  if (config.user_specified_iterations) {
    const size_t maximum_passes =
        maximum_explicit_gpu_passes(config.buffer_size_bytes);
    if (config.iterations > maximum_passes) {
      std::cerr << Messages::error_prefix()
                << Messages::error_gpu_iterations_exceed_limit(
                       config.iterations, maximum_passes)
                << std::endl;
      return EXIT_FAILURE;
    }
  }
  if (!config.user_specified_seed) {
    if (gpu_parser_hooks_active &&
        active_gpu_parser_hooks.generated_seed != 0) {
      const uint64_t injected_seed = active_gpu_parser_hooks.generated_seed;
      config.seed = SeedUtils::generate_seed(
          [injected_seed]() { return injected_seed; });
    } else {
      config.seed = SeedUtils::generate_seed();
    }
  }

  return EXIT_SUCCESS;
}

int run_gpu_bandwidth_mode(int argc, char* argv[]) {
  GpuBandwidthConfig config;
  if (parse_gpu_bandwidth_arguments(argc, argv, config) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (config.help_printed) {
    return EXIT_SUCCESS;
  }

  GpuRunResult result;
  result.main_thread_qos = prepare_main_thread_benchmark_qos();
  BenchmarkSignalMaskGuard signal_guard;
  std::unique_ptr<GpuBackend> backend = create_metal_gpu_backend();
  if (!backend) {
    std::cerr << Messages::error_prefix()
              << Messages::error_gpu_run_failed("backend-factory-failed")
              << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << Messages::msg_running_gpu_bandwidth() << std::endl;
  const int run_status =
      run_gpu_bandwidth_suite(config, *backend, result);

  bool any_result = false;
  for (const GpuOperationAggregate& aggregate : result.aggregates) {
    any_result = any_result || aggregate.headline_gb_s.has_value();
  }
  if (any_result) {
    const std::string device_name =
        result.backend_initialization.device.device_name.empty()
            ? Messages::gpu_unknown_device_name()
            : result.backend_initialization.device.device_name;
    std::cout << Messages::report_gpu_bandwidth_header(
                     device_name, config.loop_count, config.loop_count > 1)
              << std::endl;
    for (const GpuOperationAggregate& aggregate : result.aggregates) {
      if (!aggregate.headline_gb_s.has_value()) {
        continue;
      }
      const bool is_copy = aggregate.operation == GpuOperation::Copy;
      std::string operation = gpu_operation_to_string(aggregate.operation);
      operation[0] = static_cast<char>(std::toupper(operation[0]));
      std::cout << Messages::report_gpu_bandwidth_value(
                       operation, *aggregate.headline_gb_s, is_copy)
                << std::endl;
    }
    const bool repeatability_available =
        result.aggregates[0].values_gb_s.size() >= 3 &&
        result.aggregates[1].values_gb_s.size() >= 3 &&
        result.aggregates[2].values_gb_s.size() >= 3;
    std::cout << Messages::report_gpu_bandwidth_repeatability(
                     result.aggregates[0]
                         .statistics.coefficient_of_variation_pct,
                     result.aggregates[1]
                         .statistics.coefficient_of_variation_pct,
                     result.aggregates[2]
                         .statistics.coefficient_of_variation_pct,
                     repeatability_available)
              << std::endl;
    std::cout << Messages::report_gpu_bandwidth_interpretation_note()
              << std::endl;
    std::cout.flush();

    for (const GpuOperationAggregate& aggregate : result.aggregates) {
      if (aggregate.stability_quality == "noisy") {
        std::cerr << Messages::warning_prefix()
                  << Messages::warning_gpu_high_cv(
                         gpu_operation_to_string(aggregate.operation),
                         aggregate.statistics.coefficient_of_variation_pct,
                         Constants::GPU_STREAMING_CV_WARNING_PCT)
                  << std::endl;
      }
      const GpuWorkPlan& plan =
          result.work_plans[static_cast<size_t>(aggregate.operation)];
      if (plan.valid) {
        std::string warning_quality;
        for (const GpuMeasurement& measurement : result.measurements) {
          if (measurement.operation != aggregate.operation ||
              measurement.status != GpuMeasurementStatus::Measured ||
              measurement.duration_quality == "within-target-window") {
            continue;
          }
          warning_quality = measurement.duration_quality;
          break;
        }
        if (!warning_quality.empty()) {
          std::cerr << Messages::warning_prefix()
                    << Messages::warning_gpu_duration_quality(
                           gpu_operation_to_string(aggregate.operation),
                           warning_quality)
                    << std::endl;
        }
      }
    }
    if (!result.operation_order_balance_complete) {
      std::cerr << Messages::warning_prefix()
                << Messages::warning_gpu_order_not_balanced() << std::endl;
    }
    const bool environment_warning =
        std::find(result.quality_warnings.begin(),
                  result.quality_warnings.end(),
                  "environment-not-nominal") !=
        result.quality_warnings.end();
    if (environment_warning) {
      std::cerr << Messages::warning_prefix()
                << Messages::warning_gpu_environment_not_nominal()
                << std::endl;
    }
    if (result.allocation.exceeds_recommended_working_set) {
      std::cerr << Messages::warning_prefix()
                << Messages::warning_gpu_recommended_working_set_exceeded()
                << std::endl;
    }
  }

  if (!config.output_file.empty() &&
      result.reason_code != "checkpoint-write-failed") {
    std::cout << Messages::msg_results_saved_to(config.output_file)
              << std::endl;
  }
  if (result.status == GpuRunStatus::Interrupted) {
    std::cout << Messages::msg_interrupted_by_user() << std::endl;
  } else if (run_status != EXIT_SUCCESS) {
    std::cerr << Messages::error_prefix()
              << Messages::error_gpu_run_failed(result.reason_code)
              << std::endl;
  }
  return run_status;
}

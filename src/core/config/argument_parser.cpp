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
 * @file argument_parser.cpp
 * @brief Command-line argument parsing implementation
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 *
 * This file implements the descriptor-driven command-line argument parser for
 * standalone TLB analysis (`-T/--analyze-tlb`) and general help/modifier-only
 * preflight diagnostics.
 *
 * Syntactic scanning (short flags, long flags, `--option=value`, duplicate
 * detection, missing values, unexpected values, and unknown options) is handled
 * uniformly by `parse_cli_options()`. Option values are validated in a single
 * pass before any help printing, OS `sysctl` queries, cache detection, or seed
 * generation occur. Unexpected exceptions are contained at the `parse_arguments`
 * boundary and converted to `EXIT_FAILURE`.
 */

#include "core/config/cli_parser.h"
#include "core/config/config.h"
#include "core/config/constants.h"
#include "core/config/mode_selector.h"
#include "core/config/sweep_utils.h"
#include "core/system/page_size.h"
#include "core/system/system_info.h"
#include "output/console/messages/messages_api.h"
#include "output/console/output_printer.h"
#include "utils/seed_utils.h"

#include <array>
#include <charconv>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

StrictIntegerParseStatus parse_strict_signed_decimal(const std::string& value,
                                                     long long& out_value) {
  if (value.empty()) {
    return StrictIntegerParseStatus::Invalid;
  }

  const char* const begin = value.data();
  const char* const end = begin + value.size();
  const std::from_chars_result result = std::from_chars(begin, end, out_value, 10);
  if (result.ec == std::errc::result_out_of_range) {
    return StrictIntegerParseStatus::OutOfRange;
  }
  if (result.ec != std::errc{} || result.ptr != end) {
    return StrictIntegerParseStatus::Invalid;
  }
  return StrictIntegerParseStatus::Success;
}

StrictIntegerParseStatus parse_strict_unsigned_decimal(const std::string& value,
                                                       uint64_t& out_value) {
  if (value.empty()) {
    return StrictIntegerParseStatus::Invalid;
  }

  const char* const begin = value.data();
  const char* const end = begin + value.size();
  const std::from_chars_result result = std::from_chars(begin, end, out_value, 10);
  if (result.ec == std::errc::result_out_of_range) {
    return StrictIntegerParseStatus::OutOfRange;
  }
  if (result.ec != std::errc{} || result.ptr != end) {
    return StrictIntegerParseStatus::Invalid;
  }
  return StrictIntegerParseStatus::Success;
}

const char* strict_signed_decimal_error_reason(StrictIntegerParseStatus status) {
  return status == StrictIntegerParseStatus::OutOfRange
             ? Messages::reason_signed_integer_out_of_range().c_str()
             : Messages::reason_signed_integer_invalid().c_str();
}

const char* strict_unsigned_decimal_error_reason(StrictIntegerParseStatus status) {
  return status == StrictIntegerParseStatus::OutOfRange
             ? Messages::reason_unsigned_integer_out_of_range().c_str()
             : Messages::reason_unsigned_integer_invalid().c_str();
}

namespace {

struct ConfigTestHookState {
  ConfigTestHooks hooks;
  bool active = false;
};

ConfigTestHookState& config_test_hook_state() {
  static ConfigTestHookState state;
  return state;
}

}  // namespace

void set_config_test_hooks(const ConfigTestHooks* hooks) {
  ConfigTestHookState& state = config_test_hook_state();
  if (hooks == nullptr) {
    state.hooks = ConfigTestHooks{};
    state.active = false;
    return;
  }
  state.hooks = *hooks;
  state.active = true;
}

const ConfigTestHooks* get_config_test_hooks() {
  const ConfigTestHookState& state = config_test_hook_state();
  return state.active ? &state.hooks : nullptr;
}

size_t get_config_page_size_bytes() {
  const ConfigTestHooks* hooks = get_config_test_hooks();
  if (hooks != nullptr && hooks->page_size_bytes != 0) {
    return hooks->page_size_bytes;
  }
  return get_system_page_size_bytes();
}

namespace {

constexpr const char* OPT_ANALYZE_TLB_SHORT = "-T";
constexpr const char* OPT_ANALYZE_TLB_LONG = "--analyze-tlb";
constexpr const char* OPT_BUFFER_SIZE_SHORT = "-b";
constexpr const char* OPT_BUFFER_SIZE_LONG = "--buffer-size";
constexpr const char* OPT_CACHE_SIZE_SHORT = "-k";
constexpr const char* OPT_CACHE_SIZE_LONG = "--cache-size";
constexpr const char* OPT_COUNT_SHORT = "-r";
constexpr const char* OPT_COUNT_LONG = "--count";
constexpr const char* OPT_HELP_SHORT = "-h";
constexpr const char* OPT_HELP_LONG = "--help";
constexpr const char* OPT_ITERATIONS_SHORT = "-i";
constexpr const char* OPT_ITERATIONS_LONG = "--iterations";
constexpr const char* OPT_LATENCY_CHAIN_MODE_SHORT = "-m";
constexpr const char* OPT_LATENCY_CHAIN_MODE_LONG = "--latency-chain-mode";
constexpr const char* OPT_LATENCY_SAMPLES_SHORT = "-n";
constexpr const char* OPT_LATENCY_SAMPLES_LONG = "--latency-samples";
constexpr const char* OPT_LATENCY_STRIDE_SHORT = "-s";
constexpr const char* OPT_LATENCY_STRIDE_LONG = "--latency-stride-bytes";
constexpr const char* OPT_LATENCY_TLB_LOCALITY_SHORT = "-l";
constexpr const char* OPT_LATENCY_TLB_LOCALITY_LONG = "--latency-tlb-locality-kb";
constexpr const char* OPT_NON_CACHEABLE_SHORT = "-u";
constexpr const char* OPT_NON_CACHEABLE_LONG = "--non-cacheable";
constexpr const char* OPT_OUTPUT_SHORT = "-o";
constexpr const char* OPT_OUTPUT_LONG = "--output";
constexpr const char* OPT_SEED_LONG = "--seed";
constexpr const char* OPT_SWEEP_SHORT = "-S";
constexpr const char* OPT_SWEEP_LONG = "--sweep";
constexpr const char* OPT_SWEEP_MAX_RUNS_SHORT = "-X";
constexpr const char* OPT_SWEEP_MAX_RUNS_LONG = "--sweep-max-runs";
constexpr const char* OPT_THREADS_SHORT = "-t";
constexpr const char* OPT_THREADS_LONG = "--threads";
constexpr const char* OPT_TLB_DENSITY_SHORT = "-D";
constexpr const char* OPT_TLB_DENSITY_LONG = "--tlb-density";

enum TlbOptionId {
  TLB_OPT_ANALYZE_TLB = 1,
  TLB_OPT_HELP,
  TLB_OPT_OUTPUT,
  TLB_OPT_SWEEP,
  TLB_OPT_SWEEP_MAX_RUNS,
  TLB_OPT_LATENCY_STRIDE,
  TLB_OPT_LATENCY_CHAIN_MODE,
  TLB_OPT_TLB_DENSITY,
  TLB_OPT_SEED,
};

constexpr std::array<CliOptionSpec, 9> kTlbOptionSpecs{{
    {TLB_OPT_ANALYZE_TLB, OPT_ANALYZE_TLB_SHORT, OPT_ANALYZE_TLB_LONG,
     CliOptionValueKind::None, CliOptionArity::Single},
    {TLB_OPT_HELP, OPT_HELP_SHORT, OPT_HELP_LONG,
     CliOptionValueKind::None, CliOptionArity::Single},
    {TLB_OPT_OUTPUT, OPT_OUTPUT_SHORT, OPT_OUTPUT_LONG,
     CliOptionValueKind::RequiredValue, CliOptionArity::Single},
    {TLB_OPT_SWEEP, OPT_SWEEP_SHORT, OPT_SWEEP_LONG,
     CliOptionValueKind::RequiredValue, CliOptionArity::Repeatable},
    {TLB_OPT_SWEEP_MAX_RUNS, OPT_SWEEP_MAX_RUNS_SHORT, OPT_SWEEP_MAX_RUNS_LONG,
     CliOptionValueKind::RequiredValue, CliOptionArity::Single},
    {TLB_OPT_LATENCY_STRIDE, OPT_LATENCY_STRIDE_SHORT, OPT_LATENCY_STRIDE_LONG,
     CliOptionValueKind::RequiredValue, CliOptionArity::Single},
    {TLB_OPT_LATENCY_CHAIN_MODE, OPT_LATENCY_CHAIN_MODE_SHORT, OPT_LATENCY_CHAIN_MODE_LONG,
     CliOptionValueKind::RequiredValue, CliOptionArity::Single},
    {TLB_OPT_TLB_DENSITY, OPT_TLB_DENSITY_SHORT, OPT_TLB_DENSITY_LONG,
     CliOptionValueKind::RequiredValue, CliOptionArity::Single},
    {TLB_OPT_SEED, nullptr, OPT_SEED_LONG,
     CliOptionValueKind::RequiredValue, CliOptionArity::Single},
}};

enum GeneralOptionId {
  GEN_OPT_NON_CACHEABLE = 1,
  GEN_OPT_HELP,
  GEN_OPT_ITERATIONS,
  GEN_OPT_BUFFER_SIZE,
  GEN_OPT_COUNT,
  GEN_OPT_LATENCY_SAMPLES,
  GEN_OPT_LATENCY_STRIDE,
  GEN_OPT_LATENCY_CHAIN_MODE,
  GEN_OPT_LATENCY_TLB_LOCALITY,
  GEN_OPT_CACHE_SIZE,
  GEN_OPT_THREADS,
  GEN_OPT_OUTPUT,
  GEN_OPT_SEED,
  GEN_OPT_SWEEP,
  GEN_OPT_SWEEP_MAX_RUNS,
};

constexpr std::array<CliOptionSpec, 15> kGeneralOptionSpecs{{
    {GEN_OPT_NON_CACHEABLE, OPT_NON_CACHEABLE_SHORT, OPT_NON_CACHEABLE_LONG,
     CliOptionValueKind::None, CliOptionArity::Single},
    {GEN_OPT_HELP, OPT_HELP_SHORT, OPT_HELP_LONG,
     CliOptionValueKind::None, CliOptionArity::Single},
    {GEN_OPT_ITERATIONS, OPT_ITERATIONS_SHORT, OPT_ITERATIONS_LONG,
     CliOptionValueKind::RequiredValue, CliOptionArity::Single},
    {GEN_OPT_BUFFER_SIZE, OPT_BUFFER_SIZE_SHORT, OPT_BUFFER_SIZE_LONG,
     CliOptionValueKind::RequiredValue, CliOptionArity::Single},
    {GEN_OPT_COUNT, OPT_COUNT_SHORT, OPT_COUNT_LONG,
     CliOptionValueKind::RequiredValue, CliOptionArity::Single},
    {GEN_OPT_LATENCY_SAMPLES, OPT_LATENCY_SAMPLES_SHORT, OPT_LATENCY_SAMPLES_LONG,
     CliOptionValueKind::RequiredValue, CliOptionArity::Single},
    {GEN_OPT_LATENCY_STRIDE, OPT_LATENCY_STRIDE_SHORT, OPT_LATENCY_STRIDE_LONG,
     CliOptionValueKind::RequiredValue, CliOptionArity::Single},
    {GEN_OPT_LATENCY_CHAIN_MODE, OPT_LATENCY_CHAIN_MODE_SHORT, OPT_LATENCY_CHAIN_MODE_LONG,
     CliOptionValueKind::RequiredValue, CliOptionArity::Single},
    {GEN_OPT_LATENCY_TLB_LOCALITY, OPT_LATENCY_TLB_LOCALITY_SHORT, OPT_LATENCY_TLB_LOCALITY_LONG,
     CliOptionValueKind::RequiredValue, CliOptionArity::Single},
    {GEN_OPT_CACHE_SIZE, OPT_CACHE_SIZE_SHORT, OPT_CACHE_SIZE_LONG,
     CliOptionValueKind::RequiredValue, CliOptionArity::Single},
    {GEN_OPT_THREADS, OPT_THREADS_SHORT, OPT_THREADS_LONG,
     CliOptionValueKind::RequiredValue, CliOptionArity::Single},
    {GEN_OPT_OUTPUT, OPT_OUTPUT_SHORT, OPT_OUTPUT_LONG,
     CliOptionValueKind::RequiredValue, CliOptionArity::Single},
    {GEN_OPT_SEED, nullptr, OPT_SEED_LONG,
     CliOptionValueKind::RequiredValue, CliOptionArity::Single},
    {GEN_OPT_SWEEP, OPT_SWEEP_SHORT, OPT_SWEEP_LONG,
     CliOptionValueKind::RequiredValue, CliOptionArity::Repeatable},
    {GEN_OPT_SWEEP_MAX_RUNS, OPT_SWEEP_MAX_RUNS_SHORT, OPT_SWEEP_MAX_RUNS_LONG,
     CliOptionValueKind::RequiredValue, CliOptionArity::Single},
}};

long long parse_signed_decimal_or_throw(const std::string& value) {
  long long parsed = 0;
  const StrictIntegerParseStatus status = parse_strict_signed_decimal(value, parsed);
  if (status != StrictIntegerParseStatus::Success) {
    throw std::out_of_range(strict_signed_decimal_error_reason(status));
  }
  return parsed;
}

uint64_t parse_unsigned_decimal_or_throw(const std::string& value) {
  uint64_t parsed = 0;
  const StrictIntegerParseStatus status = parse_strict_unsigned_decimal(value, parsed);
  if (status != StrictIntegerParseStatus::Success) {
    throw std::out_of_range(strict_unsigned_decimal_error_reason(status));
  }
  return parsed;
}

unsigned long parse_buffer_size_mb_or_throw(const std::string& value) {
  unsigned long parsed = 0;
  long long negative_value = 0;
  const StrictNonNegativeUlongStatus status =
      parse_strict_nonnegative_ulong(value, parsed, negative_value);
  switch (status) {
    case StrictNonNegativeUlongStatus::Success:
      return parsed;
    case StrictNonNegativeUlongStatus::Negative:
      throw std::out_of_range(Messages::error_buffersize_invalid(
          negative_value, std::numeric_limits<unsigned long>::max()));
    case StrictNonNegativeUlongStatus::OutOfRange:
      throw std::out_of_range(Messages::reason_unsigned_integer_out_of_range());
    case StrictNonNegativeUlongStatus::Invalid:
      throw std::out_of_range(Messages::reason_signed_integer_invalid());
  }
  throw std::out_of_range(Messages::reason_signed_integer_invalid());
}

uint64_t generate_config_seed() {
  const ConfigTestHooks* hooks = get_config_test_hooks();
  if (hooks != nullptr && hooks->generated_seed != 0) {
    const uint64_t injected_seed = hooks->generated_seed;
    return SeedUtils::generate_seed(
        [injected_seed]() { return injected_seed; });
  }
  return SeedUtils::generate_seed();
}

void populate_common_system_info(BenchmarkConfig& config, int* out_max_cores = nullptr) {
  const ConfigTestHooks* test_hooks = get_config_test_hooks();
  const bool use_injected_system_info = test_hooks != nullptr && test_hooks->use_system_info;
  config.cpu_name = use_injected_system_info ? test_hooks->cpu_name : get_processor_name();
  config.macos_version = use_injected_system_info ? test_hooks->macos_version : get_macos_version();
  config.perf_cores = use_injected_system_info ? test_hooks->performance_cores : get_performance_cores();
  config.eff_cores = use_injected_system_info ? test_hooks->efficiency_cores : get_efficiency_cores();
  if (out_max_cores != nullptr) {
    *out_max_cores = use_injected_system_info ? test_hooks->total_logical_cores : get_total_logical_cores();
  }
}

bool tlb_sweep_density_from_string(const std::string& value, TlbSweepDensity& out_density) {
  if (value == "low") {
    out_density = TlbSweepDensity::Low;
    return true;
  }
  if (value == "medium") {
    out_density = TlbSweepDensity::Medium;
    return true;
  }
  if (value == "high") {
    out_density = TlbSweepDensity::High;
    return true;
  }
  return false;
}

bool sweep_parameter_from_string(const std::string& value,
                                 SweepParameter& out_parameter,
                                 std::string& out_name) {
  if (value == "buffer-size") {
    out_parameter = SweepParameter::BufferSizeMb;
    out_name = "buffer-size";
    return true;
  }
  if (value == "cache-size") {
    out_parameter = SweepParameter::CacheSizeKb;
    out_name = "cache-size";
    return true;
  }
  if (value == "threads") {
    out_parameter = SweepParameter::Threads;
    out_name = "threads";
    return true;
  }
  if (value == "latency-tlb-locality-kb") {
    out_parameter = SweepParameter::LatencyTlbLocalityKb;
    out_name = "latency-tlb-locality-kb";
    return true;
  }
  if (value == "latency-stride-bytes") {
    out_parameter = SweepParameter::LatencyStrideBytes;
    out_name = "latency-stride-bytes";
    return true;
  }
  if (value == "latency-chain-mode") {
    out_parameter = SweepParameter::LatencyChainMode;
    out_name = "latency-chain-mode";
    return true;
  }
  if (value == "tlb-density") {
    out_parameter = SweepParameter::TlbDensity;
    out_name = "tlb-density";
    return true;
  }
  return false;
}

SweepSpec parse_sweep_spec(const std::string& spec_text) {
  const ParsedSweepText parsed_text = parse_sweep_text(spec_text);

  SweepSpec spec;
  if (!sweep_parameter_from_string(parsed_text.key, spec.parameter, spec.parameter_name)) {
    throw std::invalid_argument(Messages::reason_unsupported_sweep_parameter(parsed_text.key));
  }

  for (const std::string& raw_value : parsed_text.values) {
    SweepValue value;
    value.raw_value = raw_value;

    if (spec.parameter == SweepParameter::LatencyChainMode) {
      if (!latency_chain_mode_from_string(raw_value, value.latency_chain_mode)) {
        throw std::out_of_range(Messages::error_latency_chain_mode_invalid());
      }
    } else if (spec.parameter == SweepParameter::TlbDensity) {
      if (!tlb_sweep_density_from_string(raw_value, value.tlb_sweep_density)) {
        throw std::out_of_range(Messages::reason_tlb_density_invalid());
      }
    } else if (spec.parameter == SweepParameter::BufferSizeMb) {
      const unsigned long parsed_mb = parse_buffer_size_mb_or_throw(raw_value);
      value.unsigned_value = parsed_mb;
      if (parsed_mb <= static_cast<unsigned long>(std::numeric_limits<long long>::max())) {
        value.integer_value = static_cast<long long>(parsed_mb);
      }
    } else {
      const long long parsed = parse_signed_decimal_or_throw(raw_value);
      if (spec.parameter == SweepParameter::CacheSizeKb) {
        if (parsed < 0 || parsed > Constants::MAX_CACHE_SIZE_KB ||
            (parsed > 0 && parsed < Constants::MIN_CACHE_SIZE_KB)) {
          throw std::out_of_range(Messages::error_cache_size_invalid(
              Constants::MIN_CACHE_SIZE_KB,
              Constants::MAX_CACHE_SIZE_KB,
              Constants::MAX_CACHE_SIZE_KB / 1024));
        }
      } else if (spec.parameter == SweepParameter::Threads) {
        if (parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
          throw std::out_of_range(Messages::error_threads_invalid(
              parsed, 1, std::numeric_limits<int>::max()));
        }
      } else if (spec.parameter == SweepParameter::LatencyTlbLocalityKb) {
        const long long max_locality_kb =
            static_cast<long long>(std::numeric_limits<size_t>::max() / Constants::BYTES_PER_KB);
        if (parsed < 0 || parsed > max_locality_kb) {
          throw std::out_of_range(Messages::error_latency_tlb_locality_invalid(parsed, max_locality_kb));
        }
      } else if (spec.parameter == SweepParameter::LatencyStrideBytes && parsed <= 0) {
        throw std::out_of_range(Messages::error_latency_stride_invalid(
            parsed, 1, std::numeric_limits<long long>::max()));
      }
      value.integer_value = parsed;
    }

    spec.values.push_back(value);
  }

  return spec;
}

int parse_analyze_tlb_arguments(int argc, char* argv[], BenchmarkConfig& config) {
  config.analyze_tlb = true;
  config.tlb_sweep_density = TlbSweepDensity::Medium;
  config.sweep_max_runs = Constants::DEFAULT_ANALYZE_TLB_SWEEP_MAX_RUNS;

  const CliParseResult parsed = parse_cli_options(argc, argv, kTlbOptionSpecs);
  if (!parsed.ok) {
    std::cerr << Messages::error_prefix()
              << format_cli_parse_error(parsed, &Messages::error_analyze_tlb_must_be_used_alone())
              << std::endl;
    return EXIT_FAILURE;
  }

  for (const CliParsedOption& opt : parsed.options) {
    switch (opt.id) {
      case TLB_OPT_ANALYZE_TLB:
        config.analyze_tlb = true;
        break;
      case TLB_OPT_HELP:
        config.help_printed = true;
        break;
      case TLB_OPT_OUTPUT:
        config.output_file = opt.value;
        break;
      case TLB_OPT_SWEEP:
        try {
          config.sweep_specs.push_back(parse_sweep_spec(opt.value));
        } catch (const std::invalid_argument& e) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_invalid_value(opt.matched_name, opt.value, e.what())
                    << std::endl;
          return EXIT_FAILURE;
        } catch (const std::out_of_range& e) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_invalid_value(opt.matched_name, opt.value, e.what())
                    << std::endl;
          return EXIT_FAILURE;
        }
        config.run_sweep = true;
        break;
      case TLB_OPT_SWEEP_MAX_RUNS:
        try {
          const long long val_ll = parse_signed_decimal_or_throw(opt.value);
          if (val_ll <= 0) {
            throw std::out_of_range(Messages::reason_positive_integer());
          }
          config.sweep_max_runs = static_cast<size_t>(val_ll);
        } catch (const std::out_of_range& e) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_invalid_value(opt.matched_name, opt.value, e.what())
                    << std::endl;
          return EXIT_FAILURE;
        }
        break;
      case TLB_OPT_LATENCY_STRIDE: {
        long long val_ll = 0;
        try {
          val_ll = parse_signed_decimal_or_throw(opt.value);
        } catch (const std::out_of_range& e) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_invalid_value(opt.matched_name, opt.value, e.what())
                    << std::endl;
          return EXIT_FAILURE;
        }
        if (val_ll <= 0) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_invalid_value(
                           opt.matched_name, opt.value,
                           Messages::error_latency_stride_invalid(
                               val_ll, 1, std::numeric_limits<long long>::max()))
                    << std::endl;
          return EXIT_FAILURE;
        }
        config.latency_stride_bytes = static_cast<size_t>(val_ll);
        if ((config.latency_stride_bytes % sizeof(void*)) != 0) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_latency_stride_alignment(
                           config.latency_stride_bytes, sizeof(void*))
                    << std::endl;
          return EXIT_FAILURE;
        }
        config.user_specified_latency_stride = true;
        break;
      }
      case TLB_OPT_LATENCY_CHAIN_MODE: {
        LatencyChainMode parsed_mode = LatencyChainMode::Auto;
        if (!latency_chain_mode_from_string(opt.value, parsed_mode)) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_invalid_value(
                           opt.matched_name, opt.value,
                           Messages::error_latency_chain_mode_invalid())
                    << std::endl;
          return EXIT_FAILURE;
        }
        if (parsed_mode == LatencyChainMode::GlobalRandom) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_analyze_tlb_global_random_unsupported()
                    << std::endl;
          return EXIT_FAILURE;
        }
        config.latency_chain_mode = parsed_mode;
        config.user_specified_latency_chain_mode = true;
        break;
      }
      case TLB_OPT_TLB_DENSITY: {
        TlbSweepDensity parsed_density = TlbSweepDensity::Medium;
        if (!tlb_sweep_density_from_string(opt.value, parsed_density)) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_invalid_value(
                           opt.matched_name, opt.value,
                           Messages::reason_tlb_density_invalid())
                    << std::endl;
          return EXIT_FAILURE;
        }
        config.tlb_sweep_density = parsed_density;
        config.user_specified_tlb_density = true;
        break;
      }
      case TLB_OPT_SEED:
        try {
          config.tlb_seed = parse_unsigned_decimal_or_throw(opt.value);
        } catch (const std::out_of_range& e) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_invalid_value(opt.matched_name, opt.value, e.what())
                    << std::endl;
          return EXIT_FAILURE;
        }
        config.user_specified_tlb_seed = true;
        break;
      default:
        break;
    }
  }

  if (config.help_printed) {
    print_help(cli_program_name(argc, argv));
    return EXIT_SUCCESS;
  }

  if (!config.user_specified_tlb_seed) {
    config.tlb_seed = generate_config_seed();
  }
  populate_common_system_info(config);

  return EXIT_SUCCESS;
}

int parse_general_arguments(int argc, char* argv[], BenchmarkConfig& config) {
  const CliParseResult parsed = parse_cli_options(argc, argv, kGeneralOptionSpecs);
  if (!parsed.ok) {
    std::cerr << Messages::error_prefix() << format_cli_parse_error(parsed) << std::endl;
    return EXIT_FAILURE;
  }

  int requested_threads = 0;
  bool has_requested_threads = false;
  long long requested_latency_tlb_locality_kb_ll = -1;
  bool seed_seen = false;

  for (const CliParsedOption& opt : parsed.options) {
    try {
      switch (opt.id) {
        case GEN_OPT_NON_CACHEABLE:
          config.use_non_cacheable = true;
          break;
        case GEN_OPT_HELP:
          config.help_printed = true;
          break;
        case GEN_OPT_ITERATIONS: {
          const long long val_ll = parse_signed_decimal_or_throw(opt.value);
          if (val_ll <= 0 || val_ll > std::numeric_limits<int>::max()) {
            throw std::out_of_range(
                Messages::error_iterations_invalid(val_ll, 1, std::numeric_limits<int>::max()));
          }
          config.iterations = static_cast<int>(val_ll);
          config.user_specified_iterations = true;
          break;
        }
        case GEN_OPT_BUFFER_SIZE:
          config.buffer_size_mb = parse_buffer_size_mb_or_throw(opt.value);
          config.user_specified_buffer_size = true;
          break;
        case GEN_OPT_COUNT: {
          const long long val_ll = parse_signed_decimal_or_throw(opt.value);
          if (val_ll <= 0 || val_ll > std::numeric_limits<int>::max()) {
            throw std::out_of_range(
                Messages::error_count_invalid(val_ll, 1, std::numeric_limits<int>::max()));
          }
          config.loop_count = static_cast<int>(val_ll);
          break;
        }
        case GEN_OPT_LATENCY_SAMPLES: {
          const long long val_ll = parse_signed_decimal_or_throw(opt.value);
          if (val_ll <= 0 || val_ll > std::numeric_limits<int>::max()) {
            throw std::out_of_range(
                Messages::error_latency_samples_invalid(val_ll, 1, std::numeric_limits<int>::max()));
          }
          config.latency_sample_count = static_cast<int>(val_ll);
          config.user_specified_latency_samples = true;
          break;
        }
        case GEN_OPT_LATENCY_STRIDE: {
          const long long val_ll = parse_signed_decimal_or_throw(opt.value);
          if (val_ll <= 0) {
            throw std::out_of_range(
                Messages::error_latency_stride_invalid(val_ll, 1, std::numeric_limits<long long>::max()));
          }
          config.latency_stride_bytes = static_cast<size_t>(val_ll);
          config.user_specified_latency_stride = true;
          break;
        }
        case GEN_OPT_LATENCY_CHAIN_MODE: {
          LatencyChainMode parsed_mode = LatencyChainMode::Auto;
          if (!latency_chain_mode_from_string(opt.value, parsed_mode)) {
            throw std::out_of_range(Messages::error_latency_chain_mode_invalid());
          }
          config.latency_chain_mode = parsed_mode;
          config.user_specified_latency_chain_mode = true;
          break;
        }
        case GEN_OPT_LATENCY_TLB_LOCALITY: {
          const long long val_ll = parse_signed_decimal_or_throw(opt.value);
          const long long max_locality_kb =
              static_cast<long long>(std::numeric_limits<size_t>::max() / Constants::BYTES_PER_KB);
          if (val_ll < 0 || val_ll > max_locality_kb) {
            throw std::out_of_range(
                Messages::error_latency_tlb_locality_invalid(val_ll, max_locality_kb));
          }
          requested_latency_tlb_locality_kb_ll = val_ll;
          config.user_specified_latency_tlb_locality = true;
          break;
        }
        case GEN_OPT_CACHE_SIZE: {
          const long long val_ll = parse_signed_decimal_or_throw(opt.value);
          // 0 is accepted here and validated later.
          if (val_ll < 0 || val_ll > Constants::MAX_CACHE_SIZE_KB ||
              (val_ll > 0 && val_ll < Constants::MIN_CACHE_SIZE_KB)) {
            throw std::out_of_range(Messages::error_cache_size_invalid(
                Constants::MIN_CACHE_SIZE_KB,
                Constants::MAX_CACHE_SIZE_KB,
                Constants::MAX_CACHE_SIZE_KB / 1024));
          }
          config.custom_cache_size_kb_ll = val_ll;
          break;
        }
        case GEN_OPT_THREADS: {
          const long long val_ll = parse_signed_decimal_or_throw(opt.value);
          if (val_ll <= 0 || val_ll > std::numeric_limits<int>::max()) {
            throw std::out_of_range(
                Messages::error_threads_invalid(val_ll, 1, std::numeric_limits<int>::max()));
          }
          requested_threads = static_cast<int>(val_ll);
          has_requested_threads = true;
          break;
        }
        case GEN_OPT_OUTPUT:
          config.output_file = opt.value;
          break;
        case GEN_OPT_SEED:
          (void)parse_unsigned_decimal_or_throw(opt.value);
          seed_seen = true;
          break;
        case GEN_OPT_SWEEP:
          config.sweep_specs.push_back(parse_sweep_spec(opt.value));
          config.run_sweep = true;
          break;
        case GEN_OPT_SWEEP_MAX_RUNS: {
          const long long val_ll = parse_signed_decimal_or_throw(opt.value);
          if (val_ll <= 0) {
            throw std::out_of_range(Messages::reason_positive_integer());
          }
          config.sweep_max_runs = static_cast<size_t>(val_ll);
          break;
        }
        default:
          break;
      }
    } catch (const std::invalid_argument& e) {
      std::cerr << Messages::error_prefix()
                << Messages::error_invalid_value(opt.matched_name, opt.value, e.what())
                << std::endl;
      return EXIT_FAILURE;
    } catch (const std::out_of_range& e) {
      std::cerr << Messages::error_prefix()
                << Messages::error_invalid_value(opt.matched_name, opt.value, e.what())
                << std::endl;
      return EXIT_FAILURE;
    }
  }

  if (config.help_printed) {
    print_help(cli_program_name(argc, argv));
    return EXIT_SUCCESS;
  }

  int max_cores = 0;
  populate_common_system_info(config, &max_cores);

  const ConfigTestHooks* test_hooks = get_config_test_hooks();
  const bool use_injected_system_info = test_hooks != nullptr && test_hooks->use_system_info;

  config.use_custom_cache_size = (config.custom_cache_size_kb_ll != -1);
  if (config.use_custom_cache_size) {
    config.custom_cache_size_bytes =
        static_cast<size_t>(config.custom_cache_size_kb_ll) * Constants::BYTES_PER_KB;
  } else {
    config.l1_cache_size = use_injected_system_info ? test_hooks->l1_cache_size : get_l1_cache_size();
    config.l2_cache_size = use_injected_system_info ? test_hooks->l2_cache_size : get_l2_cache_size();
  }

  // Preserve requested thread count without warning; validate_config() handles capping.
  if (has_requested_threads) {
    config.num_threads = requested_threads;
    config.user_specified_threads = true;
  } else {
    config.num_threads = max_cores;
  }

  if (requested_latency_tlb_locality_kb_ll != -1) {
    config.latency_tlb_locality_bytes =
        static_cast<size_t>(requested_latency_tlb_locality_kb_ll) * Constants::BYTES_PER_KB;
  }

  if (seed_seen) {
    config.user_specified_seed = true;
  }

  return EXIT_SUCCESS;
}

int parse_arguments_impl(int argc, char* argv[], BenchmarkConfig& config) {
  config = BenchmarkConfig{};

  const PrimaryModeSelection mode_selection = select_primary_benchmark_mode(argc, argv);
  if (mode_selection.mode == PrimaryBenchmarkMode::Conflict) {
    std::cerr << Messages::error_prefix()
              << Messages::error_mutually_exclusive_modes(
                     mode_selection.selected_options[0],
                     mode_selection.selected_options[1])
              << std::endl;
    return EXIT_FAILURE;
  }

  if (mode_selection.mode == PrimaryBenchmarkMode::AnalyzeTlb) {
    return parse_analyze_tlb_arguments(argc, argv, config);
  }

  return parse_general_arguments(argc, argv, config);
}

}  // namespace

/**
 * @brief Parse command-line arguments and populate a benchmark configuration.
 *
 * Uses exception handling internally for strict numeric and sweep validation,
 * converting all errors and unexpected exceptions to `EXIT_FAILURE` at the
 * boundary so no exception escapes to callers.
 */
int parse_arguments(int argc, char* argv[], BenchmarkConfig& config) {
  try {
    return parse_arguments_impl(argc, argv, config);
  } catch (const std::exception& error) {
    std::cerr << Messages::error_prefix()
              << Messages::error_command_execution_exception("Argument parsing", error.what())
              << std::endl;
  } catch (...) {
    std::cerr << Messages::error_prefix()
              << Messages::error_command_execution_exception("Argument parsing", "")
              << std::endl;
  }
  return EXIT_FAILURE;
}

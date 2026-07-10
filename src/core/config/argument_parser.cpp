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
 *
 * This file implements the argument parser for the memory benchmark application.
 * It handles parsing and validation of all command-line options including:
 * - Buffer size configuration (-b, --buffer-size)
 * - Iteration counts (-i, --iterations; -r, --count)
 * - Latency sampling and pointer-chain controls (--latency-samples, -s/--latency-stride-bytes,
 *   --latency-chain-mode, --latency-tlb-locality-kb)
 * - Cache size specification (-k, --cache-size)
 * - Thread count configuration (-t, --threads)
 * - Test mode selection (-B/--benchmark, -P/--patterns, --analyze-tlb,
 *   -W/--only-bandwidth, -L/--only-latency)
 * - Reproducible workload selection (--seed) and TLB density (--tlb-density)
 * - Multi-configuration sweeps (--sweep, --sweep-max-runs)
 * - Best-effort cache-discouraging allocation hints (--non-cacheable)
 * - Output options (-o, --output)
 * - Help display (-h, --help)
 *
 * The parser first discovers the selected mode and cache size, then parses the
 * complete standard/pattern/TLB option set. Core-to-core mode is routed to its
 * dedicated parser before this function. It employs exception
 * handling internally for validation, converting exceptions to return codes
 * at the function boundary.
 *
 * @note Uses exception handling internally but presents a return code interface
 * @note Performs comprehensive range validation for all numeric parameters
 */

#include "core/config/config.h"
#include "core/config/constants.h"
#include "core/system/system_info.h"
#include "output/console/messages/messages_api.h"
#include "utils/benchmark.h"
#include <charconv>
#include <chrono>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <cmath>
#include <cstdlib>
#include <vector>

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
             ? "out of range"
             : "must be an integer without whitespace, a plus sign, or trailing characters";
}

const char* strict_unsigned_decimal_error_reason(StrictIntegerParseStatus status) {
  return status == StrictIntegerParseStatus::OutOfRange
             ? "out of range for an unsigned 64-bit integer"
             : "must be an unsigned 64-bit integer without whitespace, a sign, or trailing characters";
}

namespace {

ConfigTestHooks active_test_hooks;
bool test_hooks_active = false;

}  // namespace

void set_config_test_hooks(const ConfigTestHooks* hooks) {
  if (hooks == nullptr) {
    active_test_hooks = ConfigTestHooks{};
    test_hooks_active = false;
    return;
  }
  active_test_hooks = *hooks;
  test_hooks_active = true;
}

const ConfigTestHooks* get_config_test_hooks() {
  return test_hooks_active ? &active_test_hooks : nullptr;
}

namespace {

constexpr const char* OPT_ANALYZE_TLB_SHORT = "-T";
constexpr const char* OPT_ANALYZE_TLB_LONG = "--analyze-tlb";
constexpr const char* OPT_BENCHMARK_SHORT = "-B";
constexpr const char* OPT_BENCHMARK_LONG = "--benchmark";
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
constexpr const char* OPT_ONLY_BANDWIDTH_SHORT = "-W";
constexpr const char* OPT_ONLY_BANDWIDTH_LONG = "--only-bandwidth";
constexpr const char* OPT_ONLY_LATENCY_SHORT = "-L";
constexpr const char* OPT_ONLY_LATENCY_LONG = "--only-latency";
constexpr const char* OPT_OUTPUT_SHORT = "-o";
constexpr const char* OPT_OUTPUT_LONG = "--output";
constexpr const char* OPT_PATTERNS_SHORT = "-P";
constexpr const char* OPT_PATTERNS_LONG = "--patterns";
constexpr const char* OPT_SEED_LONG = "--seed";
constexpr const char* OPT_SWEEP_SHORT = "-S";
constexpr const char* OPT_SWEEP_LONG = "--sweep";
constexpr const char* OPT_SWEEP_MAX_RUNS_SHORT = "-X";
constexpr const char* OPT_SWEEP_MAX_RUNS_LONG = "--sweep-max-runs";
constexpr const char* OPT_THREADS_SHORT = "-t";
constexpr const char* OPT_THREADS_LONG = "--threads";
constexpr const char* OPT_TLB_DENSITY_SHORT = "-D";
constexpr const char* OPT_TLB_DENSITY_LONG = "--tlb-density";

bool is_option(const std::string& arg, const char* short_option, const char* long_option) {
  return arg == short_option || arg == long_option;
}

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

uint64_t generate_seed() {
  const ConfigTestHooks* hooks = get_config_test_hooks();
  if (hooks != nullptr && hooks->generated_seed != 0) {
    return hooks->generated_seed;
  }
  try {
    std::random_device random_device;
    const uint64_t high = static_cast<uint64_t>(random_device()) << 32U;
    const uint64_t seed = high ^ static_cast<uint64_t>(random_device());
    if (seed != 0) {
      return seed;
    }
  } catch (...) {
    // Fall through to a local monotonic-clock seed if random_device is unavailable.
  }
  const uint64_t clock_seed = static_cast<uint64_t>(
      std::chrono::steady_clock::now().time_since_epoch().count());
  return clock_seed != 0 ? clock_seed : 0x9e3779b97f4a7c15ULL;
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

std::vector<std::string> split_comma_values(const std::string& input) {
  std::vector<std::string> values;
  size_t value_start = 0;
  while (value_start <= input.size()) {
    const size_t comma = input.find(',', value_start);
    if (comma == std::string::npos) {
      values.push_back(input.substr(value_start));
      break;
    }
    values.push_back(input.substr(value_start, comma - value_start));
    value_start = comma + 1;
  }
  return values;
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
  const size_t equals_pos = spec_text.find('=');
  if (equals_pos == std::string::npos || equals_pos == 0 || equals_pos == spec_text.size() - 1) {
    throw std::invalid_argument("sweep must use key=value1,value2 syntax");
  }

  const std::string key = spec_text.substr(0, equals_pos);
  const std::string value_text = spec_text.substr(equals_pos + 1);

  SweepSpec spec;
  if (!sweep_parameter_from_string(key, spec.parameter, spec.parameter_name)) {
    throw std::invalid_argument("unsupported sweep parameter: " + key);
  }

  const std::vector<std::string> raw_values = split_comma_values(value_text);
  if (raw_values.empty()) {
    throw std::invalid_argument("sweep value list cannot be empty");
  }

  for (const std::string& raw_value : raw_values) {
    if (raw_value.empty()) {
      throw std::invalid_argument("sweep value list cannot contain empty values");
    }

    SweepValue value;
    value.raw_value = raw_value;

    if (spec.parameter == SweepParameter::LatencyChainMode) {
      if (!latency_chain_mode_from_string(raw_value, value.latency_chain_mode)) {
        throw std::out_of_range(Messages::error_latency_chain_mode_invalid());
      }
    } else if (spec.parameter == SweepParameter::TlbDensity) {
      if (!tlb_sweep_density_from_string(raw_value, value.tlb_sweep_density)) {
        throw std::out_of_range("must be one of: low, medium, high");
      }
    } else {
      const long long parsed = parse_signed_decimal_or_throw(raw_value);
      switch (spec.parameter) {
        case SweepParameter::BufferSizeMb:
          if (parsed < 0 || parsed > std::numeric_limits<unsigned long>::max()) {
            throw std::out_of_range(Messages::error_buffersize_invalid(
                parsed, std::numeric_limits<unsigned long>::max()));
          }
          break;
        case SweepParameter::CacheSizeKb:
          if (parsed < 0 || parsed > Constants::MAX_CACHE_SIZE_KB ||
              (parsed > 0 && parsed < Constants::MIN_CACHE_SIZE_KB)) {
            throw std::out_of_range(Messages::error_cache_size_invalid(
                Constants::MIN_CACHE_SIZE_KB,
                Constants::MAX_CACHE_SIZE_KB,
                Constants::MAX_CACHE_SIZE_KB / 1024));
          }
          break;
        case SweepParameter::Threads:
          if (parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
            throw std::out_of_range(Messages::error_threads_invalid(
                parsed, 1, std::numeric_limits<int>::max()));
          }
          break;
        case SweepParameter::LatencyTlbLocalityKb: {
          const long long max_locality_kb =
              static_cast<long long>(std::numeric_limits<size_t>::max() / Constants::BYTES_PER_KB);
          if (parsed < 0 || parsed > max_locality_kb) {
            throw std::out_of_range(Messages::error_latency_tlb_locality_invalid(parsed, max_locality_kb));
          }
          break;
        }
        case SweepParameter::LatencyStrideBytes:
          if (parsed <= 0) {
            throw std::out_of_range(Messages::error_latency_stride_invalid(
                parsed, 1, std::numeric_limits<long long>::max()));
          }
          break;
        case SweepParameter::LatencyChainMode:
        case SweepParameter::TlbDensity:
          break;
      }
      value.integer_value = parsed;
    }

    spec.values.push_back(value);
  }

  return spec;
}

}  // namespace

/**
 * @brief Error Handling Strategy for this module:
 * 
 * This function uses EXCEPTIONS internally, which are caught and converted to RETURN CODES.
 * 
 * Rationale:
 * - Argument parsing involves multiple validation points with complex logic
 * - Strict numeric conversion reports malformed and out-of-range tokens through exceptions
 * - Exceptions allow early termination without deeply nested conditionals
 * - Cleaner code flow for validation logic
 * - BUT: Must convert to return codes at boundary (called from main())
 * 
 * Exception types used:
 * - std::invalid_argument: Missing values, unknown options, malformed arguments
 * - std::out_of_range: Value validation errors (out of valid range)
 * 
 * Error conversion:
 * - All exceptions are caught and converted to EXIT_FAILURE
 * - Error messages are printed to std::cerr
 * - Usage information is printed before returning
 * - Returns EXIT_SUCCESS on success, EXIT_FAILURE on any error
 * 
 * Note: This is a "boundary function" - it converts exceptions to return codes
 *       to integrate with the main program flow (which uses exit codes).
 */
int parse_arguments(int argc, char* argv[], BenchmarkConfig& config) {
  long long requested_buffer_size_mb_ll = -1;  // User requested size (-1 = none)
  long long requested_threads_ll = -1;  // User requested threads (-1 = none)
  long long requested_latency_tlb_locality_kb_ll = -1;  // User requested TLB-locality window in KB (-1 = none)

  bool analyze_tlb_present = false;
  for (int i = 1; i < argc; ++i) {
    if (is_option(std::string(argv[i]), OPT_ANALYZE_TLB_SHORT, OPT_ANALYZE_TLB_LONG)) {
      analyze_tlb_present = true;
      break;
    }
  }

  if (analyze_tlb_present) {
    config.analyze_tlb = true;
    config.tlb_sweep_density = TlbSweepDensity::Medium;
    config.sweep_max_runs = Constants::DEFAULT_ANALYZE_TLB_SWEEP_MAX_RUNS;
    bool output_seen = false;
    bool latency_stride_seen = false;
    bool latency_chain_mode_seen = false;
    bool tlb_density_seen = false;
    bool seed_seen = false;
    bool sweep_max_runs_seen = false;

    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (is_option(arg, OPT_ANALYZE_TLB_SHORT, OPT_ANALYZE_TLB_LONG)) {
        continue;
      }

      if (is_option(arg, OPT_OUTPUT_SHORT, OPT_OUTPUT_LONG)) {
        if (output_seen) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_duplicate_option(OPT_OUTPUT_LONG)
                    << std::endl;
          print_usage(argv[0]);
          return EXIT_FAILURE;
        }
        if (++i >= argc) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_missing_value(OPT_OUTPUT_LONG)
                    << std::endl;
          print_usage(argv[0]);
          return EXIT_FAILURE;
        }
        config.output_file = argv[i];
        output_seen = true;
        continue;
      }

      if (is_option(arg, OPT_SWEEP_SHORT, OPT_SWEEP_LONG)) {
        if (++i >= argc) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_missing_value(OPT_SWEEP_LONG)
                    << std::endl;
          print_usage(argv[0]);
          return EXIT_FAILURE;
        }
        try {
          config.sweep_specs.push_back(parse_sweep_spec(argv[i]));
        } catch (const std::invalid_argument& e) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_invalid_value(arg, argv[i], e.what())
                    << std::endl;
          print_usage(argv[0]);
          return EXIT_FAILURE;
        } catch (const std::out_of_range& e) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_invalid_value(arg, argv[i], e.what())
                    << std::endl;
          print_usage(argv[0]);
          return EXIT_FAILURE;
        }
        config.run_sweep = true;
        continue;
      }

      if (is_option(arg, OPT_SWEEP_MAX_RUNS_SHORT, OPT_SWEEP_MAX_RUNS_LONG)) {
        if (sweep_max_runs_seen) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_duplicate_option(OPT_SWEEP_MAX_RUNS_LONG)
                    << std::endl;
          print_usage(argv[0]);
          return EXIT_FAILURE;
        }
        if (++i >= argc) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_missing_value(OPT_SWEEP_MAX_RUNS_LONG)
                    << std::endl;
          print_usage(argv[0]);
          return EXIT_FAILURE;
        }
        try {
          const long long val_ll = parse_signed_decimal_or_throw(argv[i]);
          if (val_ll <= 0) {
            throw std::out_of_range("must be a positive integer");
          }
          config.sweep_max_runs = static_cast<size_t>(val_ll);
        } catch (const std::invalid_argument& e) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_invalid_value(arg, argv[i], e.what())
                    << std::endl;
          print_usage(argv[0]);
          return EXIT_FAILURE;
        } catch (const std::out_of_range& e) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_invalid_value(arg, argv[i], e.what())
                    << std::endl;
          print_usage(argv[0]);
          return EXIT_FAILURE;
        }
        sweep_max_runs_seen = true;
        continue;
      }

      if (is_option(arg, OPT_LATENCY_STRIDE_SHORT, OPT_LATENCY_STRIDE_LONG)) {
        if (latency_stride_seen) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_duplicate_option(OPT_LATENCY_STRIDE_LONG)
                    << std::endl;
          print_usage(argv[0]);
          return EXIT_FAILURE;
        }
        if (++i >= argc) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_missing_value(OPT_LATENCY_STRIDE_LONG)
                    << std::endl;
          print_usage(argv[0]);
          return EXIT_FAILURE;
        }

        const std::string stride_value = argv[i];
        long long val_ll = 0;
        try {
          val_ll = parse_signed_decimal_or_throw(stride_value);
        } catch (const std::out_of_range& e) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_invalid_value(arg, stride_value, e.what())
                    << std::endl;
          print_usage(argv[0]);
          return EXIT_FAILURE;
        }

        if (val_ll <= 0) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_invalid_value(
                           arg,
                           stride_value,
                           Messages::error_latency_stride_invalid(
                               val_ll, 1, std::numeric_limits<long long>::max()))
                    << std::endl;
          print_usage(argv[0]);
          return EXIT_FAILURE;
        }

        config.latency_stride_bytes = static_cast<size_t>(val_ll);
        if ((config.latency_stride_bytes % sizeof(void*)) != 0) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_latency_stride_alignment(
                           config.latency_stride_bytes,
                           sizeof(void*))
                    << std::endl;
          print_usage(argv[0]);
          return EXIT_FAILURE;
        }

        config.user_specified_latency_stride = true;
        latency_stride_seen = true;
        continue;
      }

      if (is_option(arg, OPT_LATENCY_CHAIN_MODE_SHORT, OPT_LATENCY_CHAIN_MODE_LONG)) {
        if (latency_chain_mode_seen) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_duplicate_option(OPT_LATENCY_CHAIN_MODE_LONG)
                    << std::endl;
          print_usage(argv[0]);
          return EXIT_FAILURE;
        }
        if (++i >= argc) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_missing_value(OPT_LATENCY_CHAIN_MODE_LONG)
                    << std::endl;
          print_usage(argv[0]);
          return EXIT_FAILURE;
        }

        const std::string mode_value = argv[i];
        LatencyChainMode parsed_mode = LatencyChainMode::Auto;
        if (!latency_chain_mode_from_string(mode_value, parsed_mode)) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_invalid_value(arg, mode_value,
                                                     Messages::error_latency_chain_mode_invalid())
                    << std::endl;
          print_usage(argv[0]);
          return EXIT_FAILURE;
        }
        if (parsed_mode == LatencyChainMode::GlobalRandom) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_analyze_tlb_global_random_unsupported()
                    << std::endl;
          print_usage(argv[0]);
          return EXIT_FAILURE;
        }

        config.latency_chain_mode = parsed_mode;
        config.user_specified_latency_chain_mode = true;
        latency_chain_mode_seen = true;
        continue;
      }

      if (is_option(arg, OPT_TLB_DENSITY_SHORT, OPT_TLB_DENSITY_LONG)) {
        if (tlb_density_seen) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_duplicate_option(OPT_TLB_DENSITY_LONG)
                    << std::endl;
          print_usage(argv[0]);
          return EXIT_FAILURE;
        }
        if (++i >= argc) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_missing_value(OPT_TLB_DENSITY_LONG)
                    << std::endl;
          print_usage(argv[0]);
          return EXIT_FAILURE;
        }

        const std::string density_value = argv[i];
        TlbSweepDensity parsed_density = TlbSweepDensity::Medium;
        if (!tlb_sweep_density_from_string(density_value, parsed_density)) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_invalid_value(
                           arg, density_value, "must be one of: low, medium, high")
                    << std::endl;
          print_usage(argv[0]);
          return EXIT_FAILURE;
        }

        config.tlb_sweep_density = parsed_density;
        tlb_density_seen = true;
        continue;
      }

      if (arg == OPT_SEED_LONG) {
        if (seed_seen) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_duplicate_option(OPT_SEED_LONG)
                    << std::endl;
          print_usage(argv[0]);
          return EXIT_FAILURE;
        }
        if (++i >= argc) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_missing_value(OPT_SEED_LONG)
                    << std::endl;
          print_usage(argv[0]);
          return EXIT_FAILURE;
        }

        const std::string seed_value = argv[i];
        try {
          config.tlb_seed = parse_unsigned_decimal_or_throw(seed_value);
        } catch (const std::out_of_range& e) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_invalid_value(arg, seed_value, e.what())
                    << std::endl;
          print_usage(argv[0]);
          return EXIT_FAILURE;
        }
        config.user_specified_tlb_seed = true;
        seed_seen = true;
        continue;
      }

      std::cerr << Messages::error_prefix()
                << Messages::error_analyze_tlb_must_be_used_alone()
                << std::endl;
      print_usage(argv[0]);
      return EXIT_FAILURE;
    }

    if (!seed_seen) {
      config.tlb_seed = generate_seed();
    }
    config.cpu_name = get_processor_name();
    config.macos_version = get_macos_version();
    config.perf_cores = get_performance_cores();
    config.eff_cores = get_efficiency_cores();

    return EXIT_SUCCESS;
  }
  
  // First pass: parse --cache-size early (needed for cache size detection)
  bool cache_size_seen = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    try {
      if (is_option(arg, OPT_CACHE_SIZE_SHORT, OPT_CACHE_SIZE_LONG)) {
        if (cache_size_seen)
          throw std::invalid_argument(Messages::error_duplicate_option(OPT_CACHE_SIZE_LONG));
        if (++i < argc) {
          long long val_ll = parse_signed_decimal_or_throw(argv[i]);
          // Error: Value validation - out of valid range
          // Note: 0 is accepted here and validated later (allowed only with --only-latency)
          if (val_ll < 0 || val_ll > Constants::MAX_CACHE_SIZE_KB ||
              (val_ll > 0 && val_ll < Constants::MIN_CACHE_SIZE_KB))
            throw std::out_of_range(Messages::error_cache_size_invalid(Constants::MIN_CACHE_SIZE_KB,
                                                                       Constants::MAX_CACHE_SIZE_KB,
                                                                       Constants::MAX_CACHE_SIZE_KB / 1024));
          config.custom_cache_size_kb_ll = val_ll;
          cache_size_seen = true;
        } else
          // Error: Missing required value for option
          throw std::invalid_argument(Messages::error_missing_value(OPT_CACHE_SIZE_LONG));
      }
    } catch (const std::invalid_argument &e) {
      // Exception caught: Convert to return code and print error
      std::cerr << Messages::error_prefix() << e.what() << std::endl;
      print_usage(argv[0]);
      return EXIT_FAILURE;  // Convert exception to return code
    } catch (const std::out_of_range &e) {
      // Exception caught: Convert to return code and print error
      std::cerr << Messages::error_prefix() << Messages::error_invalid_value(arg, argv[i], e.what()) << std::endl;
      print_usage(argv[0]);
      return EXIT_FAILURE;  // Convert exception to return code
    }
  }

  // Get system info. Unit tests may provide a deterministic platform snapshot.
  const ConfigTestHooks* test_hooks = get_config_test_hooks();
  const bool use_injected_system_info = test_hooks != nullptr && test_hooks->use_system_info;
  config.cpu_name = use_injected_system_info ? test_hooks->cpu_name : get_processor_name();
  config.macos_version = use_injected_system_info ? test_hooks->macos_version : get_macos_version();
  config.perf_cores = use_injected_system_info ? test_hooks->performance_cores : get_performance_cores();
  config.eff_cores = use_injected_system_info ? test_hooks->efficiency_cores : get_efficiency_cores();
  int max_cores = use_injected_system_info ? test_hooks->total_logical_cores : get_total_logical_cores();
  config.num_threads = max_cores;  // Default: use all available cores
  
  // Determine if custom cache size is being used
  config.use_custom_cache_size = (config.custom_cache_size_kb_ll != -1);
  
  // Get cache sizes
  if (config.use_custom_cache_size) {
    config.custom_cache_size_bytes = static_cast<size_t>(config.custom_cache_size_kb_ll) * Constants::BYTES_PER_KB;
  } else {
    config.l1_cache_size = use_injected_system_info ? test_hooks->l1_cache_size : get_l1_cache_size();
    config.l2_cache_size = use_injected_system_info ? test_hooks->l2_cache_size : get_l2_cache_size();
  }
  
  // Set default access counts from constants
  config.l1_num_accesses = Constants::L1_LATENCY_ACCESSES;
  config.l2_num_accesses = Constants::L2_LATENCY_ACCESSES;
  config.custom_num_accesses = Constants::CUSTOM_LATENCY_ACCESSES;

  // Second pass: parse all other arguments
  // Error handling: Uses try-catch to convert exceptions to return codes
  bool iterations_seen = false;
  bool buffersize_seen = false;
  bool count_seen = false;
  bool latency_samples_seen = false;
  bool latency_stride_seen = false;
  bool latency_chain_mode_seen = false;
  bool latency_tlb_locality_seen = false;
  bool threads_seen = false;
  bool output_seen = false;
  bool seed_seen = false;
  uint64_t parsed_general_seed = 0;
  bool sweep_max_runs_seen = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    try {
      if (is_option(arg, OPT_ITERATIONS_SHORT, OPT_ITERATIONS_LONG)) {
        if (iterations_seen)
          throw std::invalid_argument(Messages::error_duplicate_option(OPT_ITERATIONS_LONG));
        if (++i < argc) {
          long long val_ll = parse_signed_decimal_or_throw(argv[i]);
          // Error: Value validation - out of valid range
          if (val_ll <= 0 || val_ll > std::numeric_limits<int>::max())
            throw std::out_of_range(Messages::error_iterations_invalid(val_ll, 1, std::numeric_limits<int>::max()));
          config.iterations = static_cast<int>(val_ll);
          config.user_specified_iterations = true;
        } else
          // Error: Missing required value
          throw std::invalid_argument(Messages::error_missing_value(OPT_ITERATIONS_LONG));
        iterations_seen = true;
      } else if (is_option(arg, OPT_BUFFER_SIZE_SHORT, OPT_BUFFER_SIZE_LONG)) {
        if (buffersize_seen)
          throw std::invalid_argument(Messages::error_duplicate_option(OPT_BUFFER_SIZE_LONG));
        if (++i < argc) {
          long long val_ll = parse_signed_decimal_or_throw(argv[i]);
          // Error: Value validation - out of valid range
          // Note: 0 is accepted here and validated later (allowed only with --only-latency)
          if (val_ll < 0 || val_ll > std::numeric_limits<unsigned long>::max())
            throw std::out_of_range(Messages::error_buffersize_invalid(val_ll, std::numeric_limits<unsigned long>::max()));
          requested_buffer_size_mb_ll = val_ll;
          config.user_specified_buffersize = true;
        } else
          // Error: Missing required value
          throw std::invalid_argument(Messages::error_missing_value(OPT_BUFFER_SIZE_LONG));
        buffersize_seen = true;
      } else if (is_option(arg, OPT_COUNT_SHORT, OPT_COUNT_LONG)) {
        if (count_seen)
          throw std::invalid_argument(Messages::error_duplicate_option(OPT_COUNT_LONG));
        if (++i < argc) {
          long long val_ll = parse_signed_decimal_or_throw(argv[i]);
          // Error: Value validation - out of valid range
          if (val_ll <= 0 || val_ll > std::numeric_limits<int>::max())
            throw std::out_of_range(Messages::error_count_invalid(val_ll, 1, std::numeric_limits<int>::max()));
          config.loop_count = static_cast<int>(val_ll);
        } else
          // Error: Missing required value
          throw std::invalid_argument(Messages::error_missing_value(OPT_COUNT_LONG));
        count_seen = true;
      } else if (is_option(arg, OPT_LATENCY_SAMPLES_SHORT, OPT_LATENCY_SAMPLES_LONG)) {
        if (latency_samples_seen)
          throw std::invalid_argument(Messages::error_duplicate_option(OPT_LATENCY_SAMPLES_LONG));
        if (++i < argc) {
          long long val_ll = parse_signed_decimal_or_throw(argv[i]);
          // Error: Value validation - out of valid range
          if (val_ll <= 0 || val_ll > std::numeric_limits<int>::max())
            throw std::out_of_range(Messages::error_latency_samples_invalid(val_ll, 1, std::numeric_limits<int>::max()));
          config.latency_sample_count = static_cast<int>(val_ll);
          config.user_specified_latency_samples = true;
        } else
          // Error: Missing required value
          throw std::invalid_argument(Messages::error_missing_value(OPT_LATENCY_SAMPLES_LONG));
        latency_samples_seen = true;
      } else if (is_option(arg, OPT_LATENCY_STRIDE_SHORT, OPT_LATENCY_STRIDE_LONG)) {
        if (latency_stride_seen)
          throw std::invalid_argument(Messages::error_duplicate_option(OPT_LATENCY_STRIDE_LONG));
        if (++i < argc) {
          long long val_ll = parse_signed_decimal_or_throw(argv[i]);
          if (val_ll <= 0) {
            throw std::out_of_range(Messages::error_latency_stride_invalid(
                val_ll, 1, std::numeric_limits<long long>::max()));
          }
          config.latency_stride_bytes = static_cast<size_t>(val_ll);
          config.user_specified_latency_stride = true;
        } else {
          throw std::invalid_argument(Messages::error_missing_value(OPT_LATENCY_STRIDE_LONG));
        }
        latency_stride_seen = true;
      } else if (is_option(arg, OPT_LATENCY_CHAIN_MODE_SHORT, OPT_LATENCY_CHAIN_MODE_LONG)) {
        if (latency_chain_mode_seen)
          throw std::invalid_argument(Messages::error_duplicate_option(OPT_LATENCY_CHAIN_MODE_LONG));
        if (++i < argc) {
          LatencyChainMode parsed_mode = LatencyChainMode::Auto;
          if (!latency_chain_mode_from_string(argv[i], parsed_mode)) {
            throw std::out_of_range(Messages::error_latency_chain_mode_invalid());
          }
          config.latency_chain_mode = parsed_mode;
          config.user_specified_latency_chain_mode = true;
        } else {
          throw std::invalid_argument(Messages::error_missing_value(OPT_LATENCY_CHAIN_MODE_LONG));
        }
        latency_chain_mode_seen = true;
      } else if (is_option(arg, OPT_LATENCY_TLB_LOCALITY_SHORT, OPT_LATENCY_TLB_LOCALITY_LONG)) {
        if (latency_tlb_locality_seen)
          throw std::invalid_argument(Messages::error_duplicate_option(OPT_LATENCY_TLB_LOCALITY_LONG));
        if (++i < argc) {
          long long val_ll = parse_signed_decimal_or_throw(argv[i]);
          const long long max_locality_kb =
              static_cast<long long>(std::numeric_limits<size_t>::max() / Constants::BYTES_PER_KB);
          if (val_ll < 0 || val_ll > max_locality_kb) {
            throw std::out_of_range(Messages::error_latency_tlb_locality_invalid(val_ll, max_locality_kb));
          }
          requested_latency_tlb_locality_kb_ll = val_ll;
          config.user_specified_latency_tlb_locality = true;
        } else {
          throw std::invalid_argument(Messages::error_missing_value(OPT_LATENCY_TLB_LOCALITY_LONG));
        }
        latency_tlb_locality_seen = true;
      } else if (is_option(arg, OPT_CACHE_SIZE_SHORT, OPT_CACHE_SIZE_LONG)) {
        // Already parsed in first pass, skip it and its value in second pass
        if (++i >= argc) {
          // Error: This shouldn't happen (first pass should have validated it), but handle defensively
          throw std::invalid_argument(Messages::error_missing_value(OPT_CACHE_SIZE_LONG));
        }
        // Silently skip - already parsed and validated in first pass
        continue;
      } else if (is_option(arg, OPT_SWEEP_SHORT, OPT_SWEEP_LONG)) {
        if (++i < argc) {
          config.sweep_specs.push_back(parse_sweep_spec(argv[i]));
          config.run_sweep = true;
        } else {
          throw std::invalid_argument(Messages::error_missing_value(OPT_SWEEP_LONG));
        }
      } else if (is_option(arg, OPT_SWEEP_MAX_RUNS_SHORT, OPT_SWEEP_MAX_RUNS_LONG)) {
        if (sweep_max_runs_seen)
          throw std::invalid_argument(Messages::error_duplicate_option(OPT_SWEEP_MAX_RUNS_LONG));
        if (++i < argc) {
          long long val_ll = parse_signed_decimal_or_throw(argv[i]);
          if (val_ll <= 0) {
            throw std::out_of_range("sweep-max-runs must be a positive integer");
          }
          config.sweep_max_runs = static_cast<size_t>(val_ll);
        } else {
          throw std::invalid_argument(Messages::error_missing_value(OPT_SWEEP_MAX_RUNS_LONG));
        }
        sweep_max_runs_seen = true;
      } else if (is_option(arg, OPT_OUTPUT_SHORT, OPT_OUTPUT_LONG)) {
        if (output_seen)
          throw std::invalid_argument(Messages::error_duplicate_option(OPT_OUTPUT_LONG));
        if (++i < argc) {
          config.output_file = argv[i];
        } else
          // Error: Missing required value
          throw std::invalid_argument(Messages::error_missing_value(OPT_OUTPUT_LONG));
        output_seen = true;
      } else if (arg == OPT_SEED_LONG) {
        if (seed_seen)
          throw std::invalid_argument(Messages::error_duplicate_option(OPT_SEED_LONG));
        if (++i < argc) {
          parsed_general_seed = parse_unsigned_decimal_or_throw(argv[i]);
        } else {
          throw std::invalid_argument(Messages::error_missing_value(OPT_SEED_LONG));
        }
        seed_seen = true;
      } else if (is_option(arg, OPT_BENCHMARK_SHORT, OPT_BENCHMARK_LONG)) {
        config.run_benchmark = true;
        if (config.run_patterns) {
          throw std::invalid_argument(Messages::error_mutually_exclusive_modes(OPT_BENCHMARK_LONG, OPT_PATTERNS_LONG));
        }
      } else if (is_option(arg, OPT_PATTERNS_SHORT, OPT_PATTERNS_LONG)) {
        config.run_patterns = true;
        if (config.run_benchmark) {
          throw std::invalid_argument(Messages::error_mutually_exclusive_modes(OPT_BENCHMARK_LONG, OPT_PATTERNS_LONG));
        }
      } else if (is_option(arg, OPT_NON_CACHEABLE_SHORT, OPT_NON_CACHEABLE_LONG)) {
        config.use_non_cacheable = true;
      } else if (is_option(arg, OPT_ONLY_BANDWIDTH_SHORT, OPT_ONLY_BANDWIDTH_LONG)) {
        config.only_bandwidth = true;
      } else if (is_option(arg, OPT_ONLY_LATENCY_SHORT, OPT_ONLY_LATENCY_LONG)) {
        config.only_latency = true;
      } else if (is_option(arg, OPT_THREADS_SHORT, OPT_THREADS_LONG)) {
        if (threads_seen)
          throw std::invalid_argument(Messages::error_duplicate_option(OPT_THREADS_LONG));
        if (++i < argc) {
          long long val_ll = parse_signed_decimal_or_throw(argv[i]);
          // Error: Value validation - out of valid range
          if (val_ll <= 0 || val_ll > std::numeric_limits<int>::max())
            throw std::out_of_range(Messages::error_threads_invalid(val_ll, 1, std::numeric_limits<int>::max()));
          requested_threads_ll = val_ll;
        } else
          // Error: Missing required value
          throw std::invalid_argument(Messages::error_missing_value(OPT_THREADS_LONG));
        threads_seen = true;
      } else if (is_option(arg, OPT_HELP_SHORT, OPT_HELP_LONG)) {
        config.help_printed = true;
        print_help(argv[0]);
        return EXIT_SUCCESS;  // Special return value for help (not an error)
      } else {
        // Error: Unknown option
        throw std::invalid_argument(Messages::error_unknown_option(arg));
      }
    } catch (const std::invalid_argument &e) {
      // Exception caught: Convert to return code
      // Error: Invalid argument (missing value, unknown option, etc.)
      std::cerr << Messages::error_prefix();
      if (is_option(arg, OPT_SWEEP_SHORT, OPT_SWEEP_LONG) && i < argc) {
        std::cerr << Messages::error_invalid_value(arg, argv[i], e.what());
      } else {
        std::cerr << e.what();
      }
      std::cerr << std::endl;
      print_usage(argv[0]);
      return EXIT_FAILURE;  // Convert exception to return code
    } catch (const std::out_of_range &e) {
      // Exception caught: Convert to return code
      // Error: Value out of valid range
      std::cerr << Messages::error_prefix() << Messages::error_invalid_value(arg, argv[i], e.what()) << std::endl;
      print_usage(argv[0]);
      return EXIT_FAILURE;  // Convert exception to return code
    }
  }

  // Set buffer size from user request or default
  if (requested_buffer_size_mb_ll != -1) {
    config.buffer_size_mb = static_cast<unsigned long>(requested_buffer_size_mb_ll);
  }
  // Otherwise use default (already set in struct)

  // Set threads from user request or default
  if (requested_threads_ll != -1) {
    int requested_threads = static_cast<int>(requested_threads_ll);
    if (requested_threads > max_cores) {
      std::cerr << Messages::warning_prefix() << Messages::warning_threads_capped(requested_threads, max_cores) << std::endl;
      config.num_threads = max_cores;
    } else {
      config.num_threads = requested_threads;
    }
    config.user_specified_threads = true;
  }

  if (requested_latency_tlb_locality_kb_ll != -1) {
    config.latency_tlb_locality_bytes =
        static_cast<size_t>(requested_latency_tlb_locality_kb_ll) * Constants::BYTES_PER_KB;
  }

  if (seed_seen) {
    if (config.run_benchmark) {
      config.benchmark_seed = parsed_general_seed;
      config.user_specified_benchmark_seed = true;
    } else {
      config.pattern_seed = parsed_general_seed;
      config.user_specified_pattern_seed = true;
    }
  } else if (config.run_patterns) {
    config.pattern_seed = generate_seed();
  } else if (config.run_benchmark) {
    config.benchmark_seed = generate_seed();
  }

  return EXIT_SUCCESS;  // All arguments parsed successfully
}

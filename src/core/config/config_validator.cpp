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
 * @file config_validator.cpp
 * @brief Configuration validation implementation
 *
 * This file implements configuration validation logic that ensures the parsed
 * configuration is internally consistent and meets system resource constraints.
 * Validation includes:
 * - Validating flag combinations with pattern and TLB analysis modes
 * - Enforcing memory limits based on available system memory
 * - Verifying buffer size meets minimum requirements
 * - Capping buffer sizes to prevent system memory exhaustion
 *
 * The validator queries system information to determine available memory and
 * applies appropriate limits while respecting user-specified configurations
 * where possible. Uses return codes for error handling to integrate cleanly
 * with the main program flow.
 *
 * @note Memory limits are calculated as a fraction of available system memory
 * @note Minimum buffer size constraints ensure valid benchmark execution
 */

#include "core/config/config.h"
#include "core/config/constants.h"
#include "core/config/sweep_utils.h"
#include "core/system/system_info.h"
#include "output/console/messages/messages_api.h"
#include <cstdlib>
#include <cstdint>   // uintptr_t
#include <iostream>
#include <limits>
#include <set>

namespace {

int validate_analyze_tlb_stride(size_t stride_bytes, size_t page_size_bytes) {
  if (stride_bytes == 0) {
    std::cerr << Messages::error_prefix()
              << Messages::error_latency_stride_invalid(0, 1, std::numeric_limits<long long>::max())
              << std::endl;
    return EXIT_FAILURE;
  }

  if ((stride_bytes % sizeof(uintptr_t)) != 0) {
    std::cerr << Messages::error_prefix()
              << Messages::error_latency_stride_alignment(stride_bytes, sizeof(uintptr_t))
              << std::endl;
    return EXIT_FAILURE;
  }

  if (stride_bytes > page_size_bytes) {
    std::cerr << Messages::error_prefix()
              << Messages::error_analyze_tlb_stride_exceeds_page(stride_bytes, page_size_bytes)
              << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

bool sweep_parameter_allowed(const BenchmarkConfig& config, SweepParameter parameter) {
  if (config.analyze_tlb) {
    return parameter == SweepParameter::LatencyStrideBytes ||
           parameter == SweepParameter::LatencyChainMode ||
           parameter == SweepParameter::TlbDensity;
  }

  return false;
}

std::string mode_name_for_sweep(const BenchmarkConfig& config) {
  if (config.analyze_tlb) {
    return "--analyze-tlb";
  }
  return "";
}

const char* conflicting_direct_option_for_sweep(const BenchmarkConfig& config,
                                                SweepParameter parameter) {
  switch (parameter) {
    case SweepParameter::BufferSizeMb:
      return config.user_specified_buffer_size ? "--buffer-size" : nullptr;
    case SweepParameter::CacheSizeKb:
      return config.use_custom_cache_size ? "--cache-size" : nullptr;
    case SweepParameter::Threads:
      return config.user_specified_threads ? "--threads" : nullptr;
    case SweepParameter::LatencyTlbLocalityKb:
      return config.user_specified_latency_tlb_locality ? "--latency-tlb-locality-kb" : nullptr;
    case SweepParameter::LatencyStrideBytes:
      return config.user_specified_latency_stride ? "--latency-stride-bytes" : nullptr;
    case SweepParameter::LatencyChainMode:
      return config.user_specified_latency_chain_mode ? "--latency-chain-mode" : nullptr;
    case SweepParameter::TlbDensity:
      return config.user_specified_tlb_density ? "--tlb-density" : nullptr;
  }
  return nullptr;
}

const char* canonical_sweep_parameter_name(SweepParameter parameter) {
  switch (parameter) {
    case SweepParameter::BufferSizeMb:
      return "buffer-size";
    case SweepParameter::CacheSizeKb:
      return "cache-size";
    case SweepParameter::Threads:
      return "threads";
    case SweepParameter::LatencyTlbLocalityKb:
      return "latency-tlb-locality-kb";
    case SweepParameter::LatencyStrideBytes:
      return "latency-stride-bytes";
    case SweepParameter::LatencyChainMode:
      return "latency-chain-mode";
    case SweepParameter::TlbDensity:
      return "tlb-density";
  }
  return "";
}

}  // namespace

/**
 * @brief Error Handling Strategy for this module:
 *
 * This function uses RETURN CODES (EXIT_SUCCESS/EXIT_FAILURE) for error handling.
 *
 * Rationale:
 * - Simple validation logic with straightforward error paths
 * - Called early in program lifecycle (before exceptions are expected)
 * - Consistent with other validation functions
 * - No deep call stacks - return codes are sufficient
 * - Integrates with main() program flow which uses exit codes
 *
 * Error handling:
 * - Validation errors: Returns EXIT_FAILURE immediately with error message
 * - All errors are logged to std::cerr before returning
 * - Returns EXIT_SUCCESS on success
 *
 * Callers should check return value and handle EXIT_FAILURE appropriately.
 */
int validate_config(BenchmarkConfig& config) {
  const size_t page_size = get_config_page_size_bytes();

  if (!config.analyze_tlb) {
    if (config.user_specified_seed) {
      std::cerr << Messages::error_prefix()
                << Messages::error_seed_requires_supported_mode()
                << std::endl;
      return EXIT_FAILURE;
    }
    std::cerr << Messages::error_prefix()
              << Messages::error_missing_primary_mode()
              << std::endl;
    return EXIT_FAILURE;
  }

  if (config.run_sweep) {
    if (config.sweep_specs.empty()) {
      std::cerr << Messages::error_prefix() << Messages::error_sweep_requires_parameter() << std::endl;
      return EXIT_FAILURE;
    }
    if (config.output_file.empty()) {
      std::cerr << Messages::error_prefix() << Messages::error_sweep_requires_output() << std::endl;
      return EXIT_FAILURE;
    }

    const std::string mode_name = mode_name_for_sweep(config);
    std::set<SweepParameter> seen_parameters;
    for (const SweepSpec& spec : config.sweep_specs) {
      const std::string parameter_name =
          spec.parameter_name.empty() ? canonical_sweep_parameter_name(spec.parameter) : spec.parameter_name;
      if (!seen_parameters.insert(spec.parameter).second) {
        std::cerr << Messages::error_prefix()
                  << Messages::error_duplicate_sweep_parameter(parameter_name)
                  << std::endl;
        return EXIT_FAILURE;
      }
      if (!sweep_parameter_allowed(config, spec.parameter)) {
        std::cerr << Messages::error_prefix()
                  << Messages::error_sweep_parameter_not_allowed(parameter_name, mode_name)
                  << std::endl;
        return EXIT_FAILURE;
      }
      if (const char* conflicting_option =
              conflicting_direct_option_for_sweep(config, spec.parameter)) {
        std::cerr << Messages::error_prefix()
                  << Messages::error_sweep_option_conflict(conflicting_option, parameter_name)
                  << std::endl;
        return EXIT_FAILURE;
      }
      if (config.analyze_tlb && spec.parameter == SweepParameter::LatencyChainMode) {
        for (const SweepValue& value : spec.values) {
          if (value.latency_chain_mode == LatencyChainMode::GlobalRandom) {
            std::cerr << Messages::error_prefix()
                      << Messages::error_analyze_tlb_global_random_unsupported()
                      << std::endl;
            return EXIT_FAILURE;
          }
        }
      }
      if (config.analyze_tlb && spec.parameter == SweepParameter::LatencyStrideBytes) {
        for (const SweepValue& value : spec.values) {
          if (value.integer_value <= 0) {
            std::cerr << Messages::error_prefix()
                      << Messages::error_latency_stride_invalid(
                             value.integer_value, 1, std::numeric_limits<long long>::max())
                      << std::endl;
            return EXIT_FAILURE;
          }
          if (validate_analyze_tlb_stride(static_cast<size_t>(value.integer_value), page_size) != EXIT_SUCCESS) {
            return EXIT_FAILURE;
          }
        }
      }
    }

    const size_t sweep_run_count = calculate_sweep_run_count_from_specs(config.sweep_specs);
    if (sweep_run_count == 0 || sweep_run_count > config.sweep_max_runs) {
      std::cerr << Messages::error_prefix()
                << Messages::error_sweep_too_many_runs(sweep_run_count, config.sweep_max_runs)
                << std::endl;
      return EXIT_FAILURE;
    }
  }

  if (config.analyze_tlb) {
    return validate_analyze_tlb_stride(config.latency_stride_bytes, page_size);
  }

  return EXIT_FAILURE;
}

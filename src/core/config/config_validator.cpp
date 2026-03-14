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
 * - Checking for mutually exclusive flags (--only-bandwidth vs --only-latency)
 * - Validating flag combinations with pattern mode
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
#include "core/system/system_info.h"
#include "output/console/messages.h"
#include <iostream>
#include <unistd.h>  // getpagesize
#include <cstdint>   // uintptr_t
#include <limits>
#include <cstdlib>

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
  if (config.analyze_tlb) {
    return EXIT_SUCCESS;
  }

  const size_t page_size = static_cast<size_t>(getpagesize());

  // Error: Validate latency stride settings.
  if (config.latency_stride_bytes == 0) {
    std::cerr << Messages::error_prefix()
              << Messages::error_latency_stride_invalid(0, 1, std::numeric_limits<long long>::max())
              << std::endl;
    return EXIT_FAILURE;
  }

  if ((config.latency_stride_bytes % sizeof(uintptr_t)) != 0) {
    std::cerr << Messages::error_prefix()
              << Messages::error_latency_stride_alignment(config.latency_stride_bytes, sizeof(uintptr_t))
              << std::endl;
    return EXIT_FAILURE;
  }

  // Error: Validate mutually exclusive flags
  if (config.only_bandwidth && config.only_latency) {
    std::cerr << Messages::error_prefix() << Messages::error_incompatible_flags() << std::endl;
    return EXIT_FAILURE;  // Return code: validation error
  }
  
  // Error: Validate flags with -patterns
  if (config.run_patterns && (config.only_bandwidth || config.only_latency)) {
    std::cerr << Messages::error_prefix() << Messages::error_only_flags_with_patterns() << std::endl;
    return EXIT_FAILURE;  // Return code: validation error
  }
  
  // Error: Validate -only-bandwidth incompatibilities
  if (config.only_bandwidth) {
    if (config.use_custom_cache_size) {
      std::cerr << Messages::error_prefix() << Messages::error_only_bandwidth_with_cache_size() << std::endl;
      return EXIT_FAILURE;  // Return code: validation error
    }
    if (config.user_specified_latency_samples) {
      std::cerr << Messages::error_prefix() << Messages::error_only_bandwidth_with_latency_samples() << std::endl;
      return EXIT_FAILURE;  // Return code: validation error
    }
  }
  
  // Error: Validate -only-latency incompatibilities
  if (config.only_latency) {
    if (config.user_specified_iterations) {
      std::cerr << Messages::error_prefix() << Messages::error_only_latency_with_iterations() << std::endl;
      return EXIT_FAILURE;  // Return code: validation error
    }
  }

  if (!config.only_bandwidth) {
    if (config.use_custom_cache_size && config.custom_cache_size_bytes > 0) {
      const size_t custom_pointer_count = config.custom_cache_size_bytes / config.latency_stride_bytes;
      if (custom_pointer_count < 2) {
        std::cerr << Messages::error_prefix()
                  << Messages::error_buffer_stride_invalid_latency_chain(custom_pointer_count,
                                                                         config.custom_cache_size_bytes,
                                                                         config.latency_stride_bytes)
                  << std::endl;
        return EXIT_FAILURE;
      }
    }

    if (!config.use_custom_cache_size) {
      if (config.l1_cache_size > 0) {
        const size_t l1_pointer_count = config.l1_cache_size / config.latency_stride_bytes;
        if (l1_pointer_count < 2) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_buffer_stride_invalid_latency_chain(l1_pointer_count,
                                                                           config.l1_cache_size,
                                                                           config.latency_stride_bytes)
                    << std::endl;
          return EXIT_FAILURE;
        }
      }

      if (config.l2_cache_size > 0) {
        const size_t l2_pointer_count = config.l2_cache_size / config.latency_stride_bytes;
        if (l2_pointer_count < 2) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_buffer_stride_invalid_latency_chain(l2_pointer_count,
                                                                           config.l2_cache_size,
                                                                           config.latency_stride_bytes)
                    << std::endl;
          return EXIT_FAILURE;
        }
      }
    }
  }

  // Zero-size disabling behavior is only supported with -only-latency.
  const bool cache_latency_disabled = (config.custom_cache_size_kb_ll == 0);
  const bool main_latency_disabled = (config.buffer_size_mb == 0);

  if (cache_latency_disabled && !config.only_latency) {
    std::cerr << Messages::error_prefix() << Messages::error_cache_size_zero_requires_only_latency() << std::endl;
    return EXIT_FAILURE;  // Return code: validation error
  }

  if (main_latency_disabled && !config.only_latency) {
    std::cerr << Messages::error_prefix() << Messages::error_buffersize_zero_requires_only_latency() << std::endl;
    return EXIT_FAILURE;  // Return code: validation error
  }

  if (config.only_latency && cache_latency_disabled && main_latency_disabled) {
    std::cerr << Messages::error_prefix() << Messages::error_only_latency_requires_latency_target() << std::endl;
    return EXIT_FAILURE;  // Return code: validation error
  }

  if (config.latency_tlb_locality_bytes > 0 && (config.latency_tlb_locality_bytes % page_size) != 0) {
    const size_t locality_kb = config.latency_tlb_locality_bytes / Constants::BYTES_PER_KB;
    const size_t page_kb = page_size / Constants::BYTES_PER_KB;
    std::cerr << Messages::error_prefix()
              << Messages::error_latency_tlb_locality_page_multiple(locality_kb, page_kb)
              << std::endl;
    return EXIT_FAILURE;  // Return code: validation error
  }

  if (config.latency_tlb_locality_bytes > 0 &&
      (config.latency_tlb_locality_bytes / config.latency_stride_bytes) < 2) {
    std::cerr << Messages::error_prefix()
              << Messages::error_latency_tlb_locality_too_small_for_stride(
                     config.latency_tlb_locality_bytes, config.latency_stride_bytes)
              << std::endl;
    return EXIT_FAILURE;  // Return code: validation error
  }
  
  /**
   * Memory-cap model notes:
   * - This per-main-buffer cap is an early bound for the user-facing main buffer.
   * - It reflects phased execution peak for main-memory buffers (1x or 2x).
   * - Full peak concurrent validation, including cache phases, is performed by
   *   calculate_total_allocation_bytes() before benchmark execution starts.
   */
  // Calculate memory limit
  unsigned long available_mem_mb = get_available_memory_mb();
  unsigned long max_allowed_mb_per_buffer = 0;
  unsigned long required_main_buffers = 2;  // Default mode peak: src + dst

  if (config.only_latency) {
    required_main_buffers = 1;  // Latency-only mode: lat
  } else if (config.only_bandwidth || config.run_patterns) {
    required_main_buffers = 2;  // Bandwidth/pattern mode: src + dst
  }

  if (available_mem_mb > 0) {
    config.max_total_allowed_mb = static_cast<unsigned long>(available_mem_mb * Constants::MEMORY_LIMIT_FACTOR);
    // Divide by mode-specific main-buffer peak count.
    // Full peak validation (including cache phases) is performed by
    // calculate_total_allocation_bytes() before benchmark execution.
    max_allowed_mb_per_buffer = config.max_total_allowed_mb / required_main_buffers;
  } else {
    std::cerr << Messages::warning_prefix() << Messages::warning_cannot_get_memory() << std::endl;
    config.max_total_allowed_mb = Constants::FALLBACK_TOTAL_LIMIT_MB;
    // Divide by mode-specific main-buffer peak count.
    // Full peak validation (including cache phases) is performed by
    // calculate_total_allocation_bytes() before benchmark execution.
    max_allowed_mb_per_buffer = config.max_total_allowed_mb / required_main_buffers;
    std::cout << Messages::info_setting_max_fallback(max_allowed_mb_per_buffer) << std::endl;
  }
  
  if (max_allowed_mb_per_buffer < Constants::MINIMUM_LIMIT_MB_PER_BUFFER) {
    std::cout << Messages::info_calculated_max_less_than_min(max_allowed_mb_per_buffer, Constants::MINIMUM_LIMIT_MB_PER_BUFFER) << std::endl;
    max_allowed_mb_per_buffer = Constants::MINIMUM_LIMIT_MB_PER_BUFFER;
  }

  // Validate and cap buffer size.
  if (config.buffer_size_mb > max_allowed_mb_per_buffer) {
    std::cerr << Messages::warning_prefix() << Messages::warning_buffer_size_exceeds_limit(config.buffer_size_mb, max_allowed_mb_per_buffer) << std::endl;
    config.buffer_size_mb = max_allowed_mb_per_buffer;
  }

  // Calculate final buffer size in bytes
  config.buffer_size = static_cast<size_t>(config.buffer_size_mb) * Constants::BYTES_PER_MB;

  // Sanity checks
  // Error: Sanity check - buffer size calculation should be consistent
  if (config.buffer_size_mb > 0 && (config.buffer_size == 0 || config.buffer_size / Constants::BYTES_PER_MB != config.buffer_size_mb)) {
    std::cerr << Messages::error_prefix() << Messages::error_buffer_size_calculation(config.buffer_size_mb) << std::endl;
    return EXIT_FAILURE;  // Return code: calculation error
  }
  
  // Error: Buffer size must meet minimum requirements (page size and minimum latency buffer size)
  // Skip this check when main memory latency is explicitly disabled (-only-latency with -buffersize 0).
  if (config.buffer_size_mb > 0 &&
      (config.buffer_size < page_size || (config.buffer_size / config.latency_stride_bytes) < 2)) {
    std::cerr << Messages::error_prefix() << Messages::error_buffer_size_too_small(config.buffer_size) << std::endl;
    return EXIT_FAILURE;  // Return code: validation error
  }

  return EXIT_SUCCESS;  // All validations passed
}

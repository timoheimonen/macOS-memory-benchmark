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
#include "core/config/config.h"
#include "core/config/constants.h"
#include "core/system/system_info.h"
#include "output/console/messages.h"
#include <iostream>
#include <unistd.h>  // getpagesize
#include <cstdlib>

int validate_config(BenchmarkConfig& config) {
  // Validate mutually exclusive flags
  if (config.only_bandwidth && config.only_latency) {
    std::cerr << Messages::error_prefix() << Messages::error_incompatible_flags() << std::endl;
    return EXIT_FAILURE;
  }
  
  // Validate flags with -patterns
  if (config.run_patterns && (config.only_bandwidth || config.only_latency)) {
    std::cerr << Messages::error_prefix() << Messages::error_only_flags_with_patterns() << std::endl;
    return EXIT_FAILURE;
  }
  
  // Validate -only-bandwidth incompatibilities
  if (config.only_bandwidth) {
    if (config.use_custom_cache_size) {
      std::cerr << Messages::error_prefix() << Messages::error_only_bandwidth_with_cache_size() << std::endl;
      return EXIT_FAILURE;
    }
    if (config.user_specified_latency_samples) {
      std::cerr << Messages::error_prefix() << Messages::error_only_bandwidth_with_latency_samples() << std::endl;
      return EXIT_FAILURE;
    }
  }
  
  // Validate -only-latency incompatibilities
  if (config.only_latency) {
    if (config.user_specified_iterations) {
      std::cerr << Messages::error_prefix() << Messages::error_only_latency_with_iterations() << std::endl;
      return EXIT_FAILURE;
    }
  }
  
  // Calculate memory limit
  unsigned long available_mem_mb = get_available_memory_mb();
  unsigned long max_allowed_mb_per_buffer = 0;

  if (available_mem_mb > 0) {
    config.max_total_allowed_mb = static_cast<unsigned long>(available_mem_mb * Constants::MEMORY_LIMIT_FACTOR);
    // Divide by 3 to account for the 3 main buffers: src, dst, and lat.
    // Note: This per-buffer calculation does NOT include cache buffers (L1, L2, custom)
    // and their bandwidth counterparts. The total memory check in buffer_manager.cpp
    // is necessary to ensure all buffers (main + cache) fit within max_total_allowed_mb.
    max_allowed_mb_per_buffer = config.max_total_allowed_mb / 3;
  } else {
    std::cerr << Messages::warning_prefix() << Messages::warning_cannot_get_memory() << std::endl;
    config.max_total_allowed_mb = Constants::FALLBACK_TOTAL_LIMIT_MB;
    // Divide by 3 to account for the 3 main buffers: src, dst, and lat.
    // Note: This per-buffer calculation does NOT include cache buffers (L1, L2, custom)
    // and their bandwidth counterparts. The total memory check in buffer_manager.cpp
    // is necessary to ensure all buffers (main + cache) fit within max_total_allowed_mb.
    max_allowed_mb_per_buffer = config.max_total_allowed_mb / 3;
    std::cout << Messages::info_setting_max_fallback(max_allowed_mb_per_buffer) << std::endl;
  }
  
  if (max_allowed_mb_per_buffer < Constants::MINIMUM_LIMIT_MB_PER_BUFFER) {
    std::cout << Messages::info_calculated_max_less_than_min(max_allowed_mb_per_buffer, Constants::MINIMUM_LIMIT_MB_PER_BUFFER) << std::endl;
    max_allowed_mb_per_buffer = Constants::MINIMUM_LIMIT_MB_PER_BUFFER;
  }

  // Validate and cap buffer size (only if not latency-only, or if latency-only but buffer_size is needed)
  // For latency-only mode, we still need buffer_size for main memory latency test, but we use default
  if (!config.only_latency) {
    // For bandwidth tests, validate and cap buffer size
    if (config.buffer_size_mb > max_allowed_mb_per_buffer) {
      std::cerr << Messages::warning_prefix() << Messages::warning_buffer_size_exceeds_limit(config.buffer_size_mb, max_allowed_mb_per_buffer) << std::endl;
      config.buffer_size_mb = max_allowed_mb_per_buffer;
    }
  } else {
    // For latency-only, we still need a buffer for main memory latency test
    // Use default if not set, but don't cap it (latency test uses less memory)
    if (config.buffer_size_mb == 0) {
      config.buffer_size_mb = Constants::DEFAULT_BUFFER_SIZE_MB;
    }
    if (config.buffer_size_mb > max_allowed_mb_per_buffer) {
      std::cerr << Messages::warning_prefix() << Messages::warning_buffer_size_exceeds_limit(config.buffer_size_mb, max_allowed_mb_per_buffer) << std::endl;
      config.buffer_size_mb = max_allowed_mb_per_buffer;
    }
  }

  // Calculate final buffer size in bytes
  config.buffer_size = static_cast<size_t>(config.buffer_size_mb) * Constants::BYTES_PER_MB;

  // Sanity checks
  size_t page_size = getpagesize();
  
  if (config.buffer_size_mb > 0 && (config.buffer_size == 0 || config.buffer_size / Constants::BYTES_PER_MB != config.buffer_size_mb)) {
    std::cerr << Messages::error_prefix() << Messages::error_buffer_size_calculation(config.buffer_size_mb) << std::endl;
    return EXIT_FAILURE;
  }
  
  // For latency-only, we still need buffer_size for main memory latency test
  if (config.buffer_size < page_size || config.buffer_size < Constants::MIN_LATENCY_BUFFER_SIZE) {
    std::cerr << Messages::error_prefix() << Messages::error_buffer_size_too_small(config.buffer_size) << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}


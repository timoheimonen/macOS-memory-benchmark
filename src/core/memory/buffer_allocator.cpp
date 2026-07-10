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
 * @file buffer_allocator.cpp
 * @brief Pattern-buffer allocation and phased peak-memory accounting.
 */

#include "core/memory/buffer_allocator.h"

#include "core/config/config.h"
#include "core/config/constants.h"
#include "core/memory/buffer_manager.h"
#include "core/memory/memory_manager.h"
#include "output/console/messages/messages_api.h"
#include "utils/numeric_utils.h"

#include <cstdlib>
#include <iostream>
#include <utility>

namespace {

int validate_main_buffer_size_for_bandwidth(const BenchmarkConfig& config) {
  if (config.buffer_size == 0 && !config.only_latency) {
    std::cerr << Messages::error_prefix()
              << Messages::error_main_buffer_size_zero() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int validate_memory_limit(const BenchmarkConfig& config,
                          size_t bytes_required) {
  if (config.max_total_allowed_mb == 0) {
    return EXIT_SUCCESS;
  }

  const unsigned long required_mb =
      static_cast<unsigned long>(bytes_required / Constants::BYTES_PER_MB);
  if (required_mb > config.max_total_allowed_mb) {
    std::cerr << Messages::error_prefix()
              << Messages::error_total_memory_exceeds_limit(
                     required_mb, config.max_total_allowed_mb)
              << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int calculate_doubled_bytes(size_t buffer_size, size_t& doubled_bytes) {
  if (!NumericUtils::checked_multiply(buffer_size, 2, doubled_bytes)) {
    std::cerr << Messages::error_prefix()
              << Messages::error_buffer_size_overflow_calculation()
              << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int calculate_pattern_allocation_bytes(const BenchmarkConfig& config,
                                       size_t& allocation_bytes) {
  allocation_bytes = 0;
  if (config.buffer_size == 0) {
    std::cerr << Messages::error_prefix()
              << Messages::error_main_buffer_size_zero() << std::endl;
    return EXIT_FAILURE;
  }
  if (calculate_doubled_bytes(config.buffer_size, allocation_bytes) !=
      EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  return validate_memory_limit(config, allocation_bytes);
}

int add_to_phase_bytes(size_t value, size_t& phase_bytes) {
  if (!NumericUtils::checked_add(phase_bytes, value, phase_bytes)) {
    std::cerr << Messages::error_prefix()
              << Messages::error_total_memory_overflow() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int add_doubled_to_phase_bytes(size_t value, size_t& phase_bytes) {
  size_t doubled_bytes = 0;
  if (calculate_doubled_bytes(value, doubled_bytes) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  return add_to_phase_bytes(doubled_bytes, phase_bytes);
}

void update_peak(size_t candidate, size_t& peak_memory_bytes) {
  if (candidate > peak_memory_bytes) {
    peak_memory_bytes = candidate;
  }
}

/**
 * @brief Calculate the largest concurrent phase allocation.
 *
 * Standard execution allocates per phase; pattern execution holds one main
 * source/destination pair for the full command. Every sum and product is
 * checked through NumericUtils before it participates in the peak.
 */
int calculate_peak_allocation_bytes(const BenchmarkConfig& config,
                                    size_t& peak_memory_bytes) {
  peak_memory_bytes = 0;
  if (validate_main_buffer_size_for_bandwidth(config) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }

  if (!config.only_latency && config.buffer_size > 0) {
    size_t main_bandwidth_bytes = 0;
    if (calculate_doubled_bytes(config.buffer_size, main_bandwidth_bytes) !=
        EXIT_SUCCESS) {
      return EXIT_FAILURE;
    }
    update_peak(main_bandwidth_bytes, peak_memory_bytes);
  }

  if (!config.only_bandwidth && !config.run_patterns &&
      config.buffer_size > 0) {
    update_peak(config.buffer_size, peak_memory_bytes);
  }

  if (!config.run_patterns && config.use_custom_cache_size) {
    if (!config.only_latency && config.custom_buffer_size > 0) {
      size_t custom_bandwidth_bytes = 0;
      if (calculate_doubled_bytes(config.custom_buffer_size,
                                  custom_bandwidth_bytes) != EXIT_SUCCESS) {
        return EXIT_FAILURE;
      }
      update_peak(custom_bandwidth_bytes, peak_memory_bytes);
    }
    if (!config.only_bandwidth) {
      update_peak(config.custom_buffer_size, peak_memory_bytes);
    }
  } else if (!config.run_patterns) {
    if (!config.only_latency) {
      size_t cache_bandwidth_bytes = 0;
      if (add_doubled_to_phase_bytes(config.l1_buffer_size,
                                     cache_bandwidth_bytes) != EXIT_SUCCESS ||
          add_doubled_to_phase_bytes(config.l2_buffer_size,
                                     cache_bandwidth_bytes) != EXIT_SUCCESS) {
        return EXIT_FAILURE;
      }
      update_peak(cache_bandwidth_bytes, peak_memory_bytes);
    }

    if (!config.only_bandwidth) {
      size_t cache_latency_bytes = 0;
      if (add_to_phase_bytes(config.l1_buffer_size, cache_latency_bytes) !=
              EXIT_SUCCESS ||
          add_to_phase_bytes(config.l2_buffer_size, cache_latency_bytes) !=
              EXIT_SUCCESS) {
        return EXIT_FAILURE;
      }
      update_peak(cache_latency_bytes, peak_memory_bytes);
    }
  }

  return validate_memory_limit(config, peak_memory_bytes);
}

MmapPtr allocate_pattern_mapping(const BenchmarkConfig& config,
                                 const char* name) {
  if (config.use_non_cacheable) {
    return allocate_buffer_non_cacheable(config.buffer_size, name);
  }
  return allocate_buffer(config.buffer_size, name);
}

}  // namespace

int allocate_pattern_buffers(const BenchmarkConfig& config,
                             PatternBuffers& buffers) {
  size_t allocation_bytes = 0;
  if (calculate_pattern_allocation_bytes(config, allocation_bytes) !=
      EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  (void)allocation_bytes;

  PatternBuffers candidate;
  candidate.src_buffer_ptr = allocate_pattern_mapping(config, "src_buffer");
  if (!candidate.src_buffer_ptr) {
    return EXIT_FAILURE;
  }

  candidate.dst_buffer_ptr = allocate_pattern_mapping(config, "dst_buffer");
  if (!candidate.dst_buffer_ptr) {
    return EXIT_FAILURE;
  }

  buffers = std::move(candidate);
  return EXIT_SUCCESS;
}

int calculate_total_allocation_bytes(const BenchmarkConfig& config,
                                     size_t& total_memory_bytes) {
  return calculate_peak_allocation_bytes(config, total_memory_bytes);
}

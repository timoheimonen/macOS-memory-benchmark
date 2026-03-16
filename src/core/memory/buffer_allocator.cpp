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
 * @file buffer_allocator.cpp
 * @brief Buffer allocation logic
 *
 * Orchestrates allocation of all benchmark buffers with comprehensive validation.
 * Handles conditional allocation based on benchmark mode (bandwidth/latency/patterns),
 * cache configuration (L1/L2/custom), and memory limits.
 *
 * Key features:
 * - Overflow-safe arithmetic for buffer size calculations
 * - Total memory validation against system limits
 * - Conditional allocation based on benchmark configuration flags
 * - Support for regular and non-cacheable buffer variants
 * - Comprehensive error propagation from low-level allocators
 *
 * Buffer categories:
 * - Main buffers: src, dst (bandwidth), lat (latency)
 * - Cache latency buffers: l1, l2, or custom
 * - Cache bandwidth buffers: l1_bw_src/dst, l2_bw_src/dst, or custom_bw_src/dst
 */

#include "core/memory/buffer_allocator.h"
#include "core/memory/buffer_manager.h"  // BenchmarkBuffers
#include "core/config/config.h"  // BenchmarkConfig
#include "core/memory/memory_manager.h"  // allocate_buffer, allocate_buffer_non_cacheable
#include "core/config/constants.h"
#include "output/console/messages/messages_api.h"
#include <cstdlib>  // EXIT_SUCCESS, EXIT_FAILURE
#include <limits>   // std::numeric_limits
#include <iostream> // std::cerr

namespace {

/**
 * @brief Adds two size_t values with overflow detection.
 * @param current Current accumulator value
 * @param delta Value to add
 * @param[out] out Sum when no overflow occurs
 * @return true when addition succeeds, false on overflow
 */
bool checked_add(size_t current, size_t delta, size_t& out) {
  if (current > std::numeric_limits<size_t>::max() - delta) {
    return false;
  }
  out = current + delta;
  return true;
}

/**
 * @brief Multiplies a size_t value by two with overflow detection.
 * @param value Input value
 * @param[out] out Doubled value when no overflow occurs
 * @return true when multiplication succeeds, false on overflow
 */
bool checked_mul_by_two(size_t value, size_t& out) {
  if (value > std::numeric_limits<size_t>::max() / 2) {
    return false;
  }
  out = value * 2;
  return true;
}

/**
 * @brief Validates that main-memory bandwidth paths have a usable buffer.
 *
 * Standard and bandwidth-only paths require source/destination main buffers.
 * Latency-only mode is the only mode where main buffer size may be zero.
 */
int validate_main_buffer_size_for_bandwidth(const BenchmarkConfig& config) {
  if (config.buffer_size == 0 && !config.only_latency) {
    std::cerr << Messages::error_prefix() << Messages::error_main_buffer_size_zero() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

/**
 * @brief Validates required bytes against the configured global memory cap.
 * @param config Benchmark configuration containing max_total_allowed_mb
 * @param bytes_required Required bytes to validate
 * @return EXIT_SUCCESS when within limit, EXIT_FAILURE otherwise
 */
int validate_memory_limit(const BenchmarkConfig& config, size_t bytes_required) {
  if (config.max_total_allowed_mb == 0) {
    return EXIT_SUCCESS;
  }

  unsigned long required_mb = static_cast<unsigned long>(bytes_required / Constants::BYTES_PER_MB);
  if (required_mb > config.max_total_allowed_mb) {
    std::cerr << Messages::error_prefix()
              << Messages::error_total_memory_exceeds_limit(required_mb, config.max_total_allowed_mb)
              << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

/**
 * @brief Calculates full up-front allocation bytes for allocate_all_buffers().
 *
 * This preserves compatibility for call sites that still allocate all required
 * buffers in one step (pattern mode and legacy allocator path).
 */
int calculate_full_allocation_bytes_internal(const BenchmarkConfig& config, size_t& total_memory_bytes) {
  total_memory_bytes = 0;

  // Validate buffer sizes before calculation
  // Error: Zero buffer size is invalid for bandwidth tests (latency-only mode is exception)
  if (validate_main_buffer_size_for_bandwidth(config) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }

  // Main buffers - conditionally account based on flags
  if (!config.only_latency) {
    // Need src and dst buffers for bandwidth tests
    // Error: Check multiplication overflow first: ensure buffer_size * 2 won't overflow
    size_t main_buffer_double = 0;
    if (!checked_mul_by_two(config.buffer_size, main_buffer_double)) {
      std::cerr << Messages::error_prefix() << Messages::error_buffer_size_overflow_calculation() << std::endl;
      return EXIT_FAILURE;  // Return code: arithmetic overflow error
    }
    if (!checked_add(total_memory_bytes, main_buffer_double, total_memory_bytes)) {
      std::cerr << Messages::error_prefix() << Messages::error_total_memory_overflow() << std::endl;
      return EXIT_FAILURE;
    }
  }

  if (!config.only_bandwidth && !config.run_patterns) {
    // Need lat buffer for latency tests (but not for pattern benchmarks, which are bandwidth-only)
    if (config.buffer_size > 0) {
      // Error: Check addition overflow when accumulating total memory
      if (!checked_add(total_memory_bytes, config.buffer_size, total_memory_bytes)) {
        std::cerr << Messages::error_prefix() << Messages::error_total_memory_overflow() << std::endl;
        return EXIT_FAILURE;  // Return code: arithmetic overflow error
      }
    }
  }

  // Cache buffers - conditionally account based on flags (skip for pattern-only runs)
  if (!config.run_patterns) {
    if (config.use_custom_cache_size) {
      if (config.custom_buffer_size > 0) {
        if (!config.only_bandwidth) {
          // Need custom latency buffer
          // Error: Check addition overflow when accumulating total memory
          if (!checked_add(total_memory_bytes, config.custom_buffer_size, total_memory_bytes)) {
            std::cerr << Messages::error_prefix() << Messages::error_total_memory_overflow() << std::endl;
            return EXIT_FAILURE;  // Return code: arithmetic overflow error
          }
        }
        if (!config.only_latency) {
          // Need custom bandwidth buffers
          size_t custom_bw_double = 0;
          if (!checked_mul_by_two(config.custom_buffer_size, custom_bw_double)) {
            std::cerr << Messages::error_prefix() << Messages::error_buffer_size_overflow_calculation() << std::endl;
            return EXIT_FAILURE;
          }
          if (!checked_add(total_memory_bytes, custom_bw_double, total_memory_bytes)) {
            std::cerr << Messages::error_prefix() << Messages::error_total_memory_overflow() << std::endl;
            return EXIT_FAILURE;
          }
        }
      }
    } else {
      if (config.l1_buffer_size > 0) {
        if (!config.only_bandwidth) {
          // Need L1 latency buffer
          if (!checked_add(total_memory_bytes, config.l1_buffer_size, total_memory_bytes)) {
            std::cerr << Messages::error_prefix() << Messages::error_total_memory_overflow() << std::endl;
            return EXIT_FAILURE;
          }
        }
        if (!config.only_latency) {
          // Need L1 bandwidth buffers
          size_t l1_bw_double = 0;
          if (!checked_mul_by_two(config.l1_buffer_size, l1_bw_double)) {
            std::cerr << Messages::error_prefix() << Messages::error_buffer_size_overflow_calculation() << std::endl;
            return EXIT_FAILURE;
          }
          if (!checked_add(total_memory_bytes, l1_bw_double, total_memory_bytes)) {
            std::cerr << Messages::error_prefix() << Messages::error_total_memory_overflow() << std::endl;
            return EXIT_FAILURE;
          }
        }
      }
      if (config.l2_buffer_size > 0) {
        if (!config.only_bandwidth) {
          // Need L2 latency buffer
          if (!checked_add(total_memory_bytes, config.l2_buffer_size, total_memory_bytes)) {
            std::cerr << Messages::error_prefix() << Messages::error_total_memory_overflow() << std::endl;
            return EXIT_FAILURE;
          }
        }
        if (!config.only_latency) {
          // Need L2 bandwidth buffers
          size_t l2_bw_double = 0;
          if (!checked_mul_by_two(config.l2_buffer_size, l2_bw_double)) {
            std::cerr << Messages::error_prefix() << Messages::error_buffer_size_overflow_calculation() << std::endl;
            return EXIT_FAILURE;
          }
          if (!checked_add(total_memory_bytes, l2_bw_double, total_memory_bytes)) {
            std::cerr << Messages::error_prefix() << Messages::error_total_memory_overflow() << std::endl;
            return EXIT_FAILURE;
          }
        }
      }
    }
  }

  return validate_memory_limit(config, total_memory_bytes);
}

/**
 * @brief Calculates the peak concurrent memory footprint for phased benchmark execution.
 *
 * Unlike the legacy "allocate everything up-front" model, the benchmark now allocates
 * buffers per execution phase (main bandwidth, cache bandwidth, cache latency, main latency).
 * This helper computes the highest simultaneous byte requirement across those phases.
 *
 * Peak candidates considered:
 * - Main bandwidth phase: src + dst (2 * main buffer)
 * - Main latency phase: lat buffer (1 * main buffer)
 * - Cache bandwidth phase:
 *   - Custom cache mode: custom src + custom dst (2 * custom buffer)
 *   - Auto cache mode: L1 pair + L2 pair (2 * L1 + 2 * L2)
 * - Cache latency phase:
 *   - Custom cache mode: custom latency buffer (1 * custom buffer)
 *   - Auto cache mode: L1 latency + L2 latency (L1 + L2)
 *
 * The function keeps full overflow protection and validates the resulting peak value
 * against config.max_total_allowed_mb.
 */
int calculate_peak_allocation_bytes_internal(const BenchmarkConfig& config, size_t& peak_memory_bytes) {
  peak_memory_bytes = 0;

  if (validate_main_buffer_size_for_bandwidth(config) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }

  auto update_peak = [&peak_memory_bytes](size_t candidate) {
    if (candidate > peak_memory_bytes) {
      peak_memory_bytes = candidate;
    }
  };

  if (!config.only_latency && config.buffer_size > 0) {
    size_t main_bandwidth_bytes = 0;
    if (!checked_mul_by_two(config.buffer_size, main_bandwidth_bytes)) {
      std::cerr << Messages::error_prefix() << Messages::error_buffer_size_overflow_calculation() << std::endl;
      return EXIT_FAILURE;
    }
    update_peak(main_bandwidth_bytes);
  }

  if (!config.only_bandwidth && !config.run_patterns && config.buffer_size > 0) {
    update_peak(config.buffer_size);
  }

  if (!config.run_patterns) {
    if (config.use_custom_cache_size) {
      if (config.custom_buffer_size > 0) {
        if (!config.only_latency) {
          size_t custom_bandwidth_bytes = 0;
          if (!checked_mul_by_two(config.custom_buffer_size, custom_bandwidth_bytes)) {
            std::cerr << Messages::error_prefix() << Messages::error_buffer_size_overflow_calculation() << std::endl;
            return EXIT_FAILURE;
          }
          update_peak(custom_bandwidth_bytes);
        }

        if (!config.only_bandwidth) {
          update_peak(config.custom_buffer_size);
        }
      }
    } else {
      if (!config.only_latency) {
        size_t cache_bandwidth_peak = 0;
        if (config.l1_buffer_size > 0) {
          size_t l1_bandwidth_bytes = 0;
          if (!checked_mul_by_two(config.l1_buffer_size, l1_bandwidth_bytes)) {
            std::cerr << Messages::error_prefix() << Messages::error_buffer_size_overflow_calculation() << std::endl;
            return EXIT_FAILURE;
          }
          if (!checked_add(cache_bandwidth_peak, l1_bandwidth_bytes, cache_bandwidth_peak)) {
            std::cerr << Messages::error_prefix() << Messages::error_total_memory_overflow() << std::endl;
            return EXIT_FAILURE;
          }
        }

        if (config.l2_buffer_size > 0) {
          size_t l2_bandwidth_bytes = 0;
          if (!checked_mul_by_two(config.l2_buffer_size, l2_bandwidth_bytes)) {
            std::cerr << Messages::error_prefix() << Messages::error_buffer_size_overflow_calculation() << std::endl;
            return EXIT_FAILURE;
          }
          if (!checked_add(cache_bandwidth_peak, l2_bandwidth_bytes, cache_bandwidth_peak)) {
            std::cerr << Messages::error_prefix() << Messages::error_total_memory_overflow() << std::endl;
            return EXIT_FAILURE;
          }
        }

        update_peak(cache_bandwidth_peak);
      }

      if (!config.only_bandwidth) {
        size_t cache_latency_peak = 0;
        if (!checked_add(cache_latency_peak, config.l1_buffer_size, cache_latency_peak)) {
          std::cerr << Messages::error_prefix() << Messages::error_total_memory_overflow() << std::endl;
          return EXIT_FAILURE;
        }
        if (!checked_add(cache_latency_peak, config.l2_buffer_size, cache_latency_peak)) {
          std::cerr << Messages::error_prefix() << Messages::error_total_memory_overflow() << std::endl;
          return EXIT_FAILURE;
        }

        update_peak(cache_latency_peak);
      }
    }
  }

  return validate_memory_limit(config, peak_memory_bytes);
}

}  // namespace

/**
 * @brief Returns peak concurrent allocation bytes for phased benchmark execution.
 *
 * Public callers use this for user-facing reporting and global memory-limit
 * validation before execution begins.
 */
int calculate_total_allocation_bytes(const BenchmarkConfig& config, size_t& total_memory_bytes) {
  return calculate_peak_allocation_bytes_internal(config, total_memory_bytes);
}

/**
 * @brief Allocates all benchmark buffers based on configuration.
 *
 * Orchestrates the allocation of all required buffers for the benchmark run:
 * 1. Validates buffer sizes and configuration
 * 2. Calculates total memory requirements with overflow checking
 * 3. Validates against system memory limits
 * 4. Conditionally allocates buffers based on benchmark mode:
 *    - Main buffers (src, dst, lat) for bandwidth and latency tests
 *    - Cache buffers (L1, L2, or custom) for cache-specific tests
 *    - Pattern benchmark mode skips cache buffers entirely
 * 5. Chooses regular or non-cacheable allocation based on config flags
 *
 * Buffer allocation is conditional on configuration flags:
 * - only_latency: Skip bandwidth buffers (src, dst)
 * - only_bandwidth: Skip latency buffers (lat, cache latency)
 * - run_patterns: Skip cache buffers (pattern tests use main buffers only)
 * - use_custom_cache_size: Use custom buffers instead of L1/L2
 * - use_non_cacheable: Use best-effort non-cacheable allocation
 *
 * Error Handling Strategy:
 * - Uses RETURN CODES (EXIT_SUCCESS/EXIT_FAILURE) for error handling
 * - Validation errors: Returns EXIT_FAILURE immediately with error message
 * - Memory allocation errors: Checks null pointers from allocate_buffer() calls
 *   (which uses null pointer returns - see memory_manager.cpp)
 * - All errors are logged to std::cerr before returning
 *
 * Rationale:
 * - Called from main() program flow, which uses exit codes
 * - Orchestrates multiple memory allocations - simple error propagation needed
 * - No deep call stacks - return codes are sufficient
 * - Consistent with program-level error handling pattern
 *
 * @param[in]  config   Benchmark configuration with buffer sizes, flags, and limits
 * @param[out] buffers  Buffer management structure to populate with allocated buffers
 *
 * @return EXIT_SUCCESS (0) if all required buffers are allocated successfully
 * @return EXIT_FAILURE (1) if validation fails, overflow detected, or allocation fails
 *
 * @note All buffers are initialized to nullptr before allocation begins.
 * @note The total memory calculation accounts for all conditionally allocated buffers.
 * @note Overflow checking is performed for all size multiplications and additions.
 * @note Callers should check return value and handle EXIT_FAILURE appropriately.
 *
 * @see allocate_buffer() for the low-level allocation function
 * @see allocate_buffer_non_cacheable() for non-cached allocation variant
 * @see BenchmarkConfig for configuration options
 * @see BenchmarkBuffers for buffer management structure
 */
int allocate_all_buffers(const BenchmarkConfig& config, BenchmarkBuffers& buffers) {
  size_t total_memory = 0;
  if (calculate_full_allocation_bytes_internal(config, total_memory) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  (void)total_memory;
  
  // Initialize all buffers with nullptr and size 0
  buffers.src_buffer_ptr = MmapPtr(nullptr, MmapDeleter{0});
  buffers.dst_buffer_ptr = MmapPtr(nullptr, MmapDeleter{0});
  buffers.lat_buffer_ptr = MmapPtr(nullptr, MmapDeleter{0});
  buffers.l1_buffer_ptr = MmapPtr(nullptr, MmapDeleter{0});
  buffers.l2_buffer_ptr = MmapPtr(nullptr, MmapDeleter{0});
  buffers.l1_bw_src_ptr = MmapPtr(nullptr, MmapDeleter{0});
  buffers.l1_bw_dst_ptr = MmapPtr(nullptr, MmapDeleter{0});
  buffers.l2_bw_src_ptr = MmapPtr(nullptr, MmapDeleter{0});
  buffers.l2_bw_dst_ptr = MmapPtr(nullptr, MmapDeleter{0});
  buffers.custom_buffer_ptr = MmapPtr(nullptr, MmapDeleter{0});
  buffers.custom_bw_src_ptr = MmapPtr(nullptr, MmapDeleter{0});
  buffers.custom_bw_dst_ptr = MmapPtr(nullptr, MmapDeleter{0});

  // Allocate source buffer (only if not latency-only)
  // Note: allocate_buffer() returns null pointer on failure (see memory_manager.cpp)
  // Error: Memory allocation failed - propagate error using return code
  if (!config.only_latency && config.buffer_size > 0) {
    if (config.use_non_cacheable) {
      buffers.src_buffer_ptr = allocate_buffer_non_cacheable(config.buffer_size, "src_buffer");
    } else {
      buffers.src_buffer_ptr = allocate_buffer(config.buffer_size, "src_buffer");
    }
    if (!buffers.src_buffer_ptr) {
      return EXIT_FAILURE;  // Return code: memory allocation failure (null pointer from allocate_buffer)
    }
  }

  // Allocate destination buffer (only if not latency-only)
  // Error: Memory allocation failed - propagate error using return code
  if (!config.only_latency && config.buffer_size > 0) {
    if (config.use_non_cacheable) {
      buffers.dst_buffer_ptr = allocate_buffer_non_cacheable(config.buffer_size, "dst_buffer");
    } else {
      buffers.dst_buffer_ptr = allocate_buffer(config.buffer_size, "dst_buffer");
    }
    if (!buffers.dst_buffer_ptr) {
      return EXIT_FAILURE;  // Return code: memory allocation failure
    }
  }

  // Allocate latency buffer (only if not bandwidth-only and not pattern benchmarks)
  // Error: Memory allocation failed - propagate error using return code
  if (!config.only_bandwidth && !config.run_patterns && config.buffer_size > 0) {
    if (config.use_non_cacheable) {
      buffers.lat_buffer_ptr = allocate_buffer_non_cacheable(config.buffer_size, "lat_buffer");
    } else {
      buffers.lat_buffer_ptr = allocate_buffer(config.buffer_size, "lat_buffer");
    }
    if (!buffers.lat_buffer_ptr) {
      return EXIT_FAILURE;  // Return code: memory allocation failure
    }
  }

  // Allocate cache latency test buffers (only if not bandwidth-only and not pattern-only)
  if (!config.only_bandwidth && !config.run_patterns) {
    if (config.use_custom_cache_size) {
      // Allocate custom cache latency test buffer
      if (config.custom_buffer_size > 0) {
        if (config.use_non_cacheable) {
          buffers.custom_buffer_ptr = allocate_buffer_non_cacheable(config.custom_buffer_size, "custom_buffer");
        } else {
          buffers.custom_buffer_ptr = allocate_buffer(config.custom_buffer_size, "custom_buffer");
        }
        // Error: Memory allocation failed
        if (!buffers.custom_buffer_ptr) {
          return EXIT_FAILURE;  // Return code: memory allocation failure
        }
      }
    } else {
      // Allocate L1/L2 cache latency test buffers
      if (config.l1_buffer_size > 0) {
        if (config.use_non_cacheable) {
          buffers.l1_buffer_ptr = allocate_buffer_non_cacheable(config.l1_buffer_size, "l1_buffer");
        } else {
          buffers.l1_buffer_ptr = allocate_buffer(config.l1_buffer_size, "l1_buffer");
        }
        // Error: Memory allocation failed
        if (!buffers.l1_buffer_ptr) {
          return EXIT_FAILURE;  // Return code: memory allocation failure
        }
      }

      if (config.l2_buffer_size > 0) {
        if (config.use_non_cacheable) {
          buffers.l2_buffer_ptr = allocate_buffer_non_cacheable(config.l2_buffer_size, "l2_buffer");
        } else {
          buffers.l2_buffer_ptr = allocate_buffer(config.l2_buffer_size, "l2_buffer");
        }
        // Error: Memory allocation failed
        if (!buffers.l2_buffer_ptr) {
          return EXIT_FAILURE;  // Return code: memory allocation failure
        }
      }
    }
  }

  // Allocate cache bandwidth test buffers (only if not latency-only and not pattern-only)
  if (!config.only_latency && !config.run_patterns) {
    if (config.use_custom_cache_size) {
      // Allocate custom cache bandwidth test buffers
      if (config.custom_buffer_size > 0) {
        // Allocate source buffer for custom cache bandwidth tests
        if (config.use_non_cacheable) {
          buffers.custom_bw_src_ptr = allocate_buffer_non_cacheable(config.custom_buffer_size, "custom_bw_src_buffer");
        } else {
          buffers.custom_bw_src_ptr = allocate_buffer(config.custom_buffer_size, "custom_bw_src_buffer");
        }
        // Error: Memory allocation failed
        if (!buffers.custom_bw_src_ptr) {
          return EXIT_FAILURE;  // Return code: memory allocation failure
        }
        // Allocate destination buffer for custom cache bandwidth tests
        if (config.use_non_cacheable) {
          buffers.custom_bw_dst_ptr = allocate_buffer_non_cacheable(config.custom_buffer_size, "custom_bw_dst_buffer");
        } else {
          buffers.custom_bw_dst_ptr = allocate_buffer(config.custom_buffer_size, "custom_bw_dst_buffer");
        }
        // Error: Memory allocation failed
        if (!buffers.custom_bw_dst_ptr) {
          return EXIT_FAILURE;  // Return code: memory allocation failure
        }
      }
    } else {
      // Allocate L1/L2 cache bandwidth test buffers
      if (config.l1_buffer_size > 0) {
        // Allocate source buffer for L1 bandwidth tests
        if (config.use_non_cacheable) {
          buffers.l1_bw_src_ptr = allocate_buffer_non_cacheable(config.l1_buffer_size, "l1_bw_src_buffer");
        } else {
          buffers.l1_bw_src_ptr = allocate_buffer(config.l1_buffer_size, "l1_bw_src_buffer");
        }
        // Error: Memory allocation failed
        if (!buffers.l1_bw_src_ptr) {
          return EXIT_FAILURE;  // Return code: memory allocation failure
        }
        // Allocate destination buffer for L1 bandwidth tests
        if (config.use_non_cacheable) {
          buffers.l1_bw_dst_ptr = allocate_buffer_non_cacheable(config.l1_buffer_size, "l1_bw_dst_buffer");
        } else {
          buffers.l1_bw_dst_ptr = allocate_buffer(config.l1_buffer_size, "l1_bw_dst_buffer");
        }
        // Error: Memory allocation failed
        if (!buffers.l1_bw_dst_ptr) {
          return EXIT_FAILURE;  // Return code: memory allocation failure
        }
      }

      if (config.l2_buffer_size > 0) {
        // Allocate source buffer for L2 bandwidth tests
        if (config.use_non_cacheable) {
          buffers.l2_bw_src_ptr = allocate_buffer_non_cacheable(config.l2_buffer_size, "l2_bw_src_buffer");
        } else {
          buffers.l2_bw_src_ptr = allocate_buffer(config.l2_buffer_size, "l2_bw_src_buffer");
        }
        // Error: Memory allocation failed
        if (!buffers.l2_bw_src_ptr) {
          return EXIT_FAILURE;  // Return code: memory allocation failure
        }
        // Allocate destination buffer for L2 bandwidth tests
        if (config.use_non_cacheable) {
          buffers.l2_bw_dst_ptr = allocate_buffer_non_cacheable(config.l2_buffer_size, "l2_bw_dst_buffer");
        } else {
          buffers.l2_bw_dst_ptr = allocate_buffer(config.l2_buffer_size, "l2_bw_dst_buffer");
        }
        // Error: Memory allocation failed
        if (!buffers.l2_bw_dst_ptr) {
          return EXIT_FAILURE;  // Return code: memory allocation failure
        }
      }
    }
  }

  return EXIT_SUCCESS;  // All buffers allocated successfully
}

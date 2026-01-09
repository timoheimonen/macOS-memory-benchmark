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
#include "output/console/messages.h"
#include <cstdlib>  // EXIT_SUCCESS, EXIT_FAILURE
#include <limits>   // std::numeric_limits
#include <iostream> // std::cerr

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
  // Validate buffer sizes before allocation
  // Error: Zero buffer size is invalid for bandwidth tests (latency-only mode is exception)
  if (config.buffer_size == 0 && !config.only_latency) {
    std::cerr << Messages::error_prefix() << Messages::error_main_buffer_size_zero() << std::endl;
    return EXIT_FAILURE;  // Return code: validation error
  }
  
  // Calculate total memory requirement and check for overflow
  size_t total_memory = 0;
  
  // Main buffers - conditionally allocate based on flags
  if (!config.only_latency) {
    // Need src and dst buffers for bandwidth tests
    // Error: Check multiplication overflow first: ensure buffer_size * 2 won't overflow
    // This prevents undefined behavior from integer overflow
    if (config.buffer_size > std::numeric_limits<size_t>::max() / 2) {
      std::cerr << Messages::error_prefix() << Messages::error_buffer_size_overflow_calculation() << std::endl;
      return EXIT_FAILURE;  // Return code: arithmetic overflow error
    }
    size_t main_buffer_double = config.buffer_size * 2;  // Safe: multiplication checked above
    total_memory += main_buffer_double;  // src, dst
  }
  if (!config.only_bandwidth && !config.run_patterns) {
    // Need lat buffer for latency tests (but not for pattern benchmarks, which are bandwidth-only)
    if (config.buffer_size > 0) {
      total_memory += config.buffer_size;  // lat
    }
  }
  
  // Cache buffers - conditionally allocate based on flags (skip for pattern-only runs)
  if (!config.run_patterns) {
    if (config.use_custom_cache_size) {
      if (config.custom_buffer_size > 0) {
        if (!config.only_bandwidth) {
          // Need custom latency buffer
          // Error: Check addition overflow when accumulating total memory
          if (total_memory > std::numeric_limits<size_t>::max() - config.custom_buffer_size) {
            std::cerr << Messages::error_prefix() << Messages::error_total_memory_overflow() << std::endl;
            return EXIT_FAILURE;  // Return code: arithmetic overflow error
          }
          total_memory += config.custom_buffer_size;  // custom
        }
        if (!config.only_latency) {
          // Need custom bandwidth buffers
          if (config.custom_buffer_size > std::numeric_limits<size_t>::max() / 2) {
            std::cerr << Messages::error_prefix() << Messages::error_buffer_size_overflow_calculation() << std::endl;
            return EXIT_FAILURE;
          }
          size_t custom_bw_double = config.custom_buffer_size * 2;  // custom_bw_src, custom_bw_dst
          if (total_memory > std::numeric_limits<size_t>::max() - custom_bw_double) {
            std::cerr << Messages::error_prefix() << Messages::error_total_memory_overflow() << std::endl;
            return EXIT_FAILURE;
          }
          total_memory += custom_bw_double;  // custom_bw_src, custom_bw_dst
        }
      }
    } else {
      if (config.l1_buffer_size > 0) {
        if (!config.only_bandwidth) {
          // Need L1 latency buffer
          if (total_memory > std::numeric_limits<size_t>::max() - config.l1_buffer_size) {
            std::cerr << Messages::error_prefix() << Messages::error_total_memory_overflow() << std::endl;
            return EXIT_FAILURE;
          }
          total_memory += config.l1_buffer_size;  // l1
        }
        if (!config.only_latency) {
          // Need L1 bandwidth buffers
          if (config.l1_buffer_size > std::numeric_limits<size_t>::max() / 2) {
            std::cerr << Messages::error_prefix() << Messages::error_buffer_size_overflow_calculation() << std::endl;
            return EXIT_FAILURE;
          }
          size_t l1_bw_double = config.l1_buffer_size * 2;  // l1_bw_src, l1_bw_dst
          if (total_memory > std::numeric_limits<size_t>::max() - l1_bw_double) {
            std::cerr << Messages::error_prefix() << Messages::error_total_memory_overflow() << std::endl;
            return EXIT_FAILURE;
          }
          total_memory += l1_bw_double;  // l1_bw_src, l1_bw_dst
        }
      }
      if (config.l2_buffer_size > 0) {
        if (!config.only_bandwidth) {
          // Need L2 latency buffer
          if (total_memory > std::numeric_limits<size_t>::max() - config.l2_buffer_size) {
            std::cerr << Messages::error_prefix() << Messages::error_total_memory_overflow() << std::endl;
            return EXIT_FAILURE;
          }
          total_memory += config.l2_buffer_size;  // l2
        }
        if (!config.only_latency) {
          // Need L2 bandwidth buffers
          if (config.l2_buffer_size > std::numeric_limits<size_t>::max() / 2) {
            std::cerr << Messages::error_prefix() << Messages::error_buffer_size_overflow_calculation() << std::endl;
            return EXIT_FAILURE;
          }
          size_t l2_bw_double = config.l2_buffer_size * 2;  // l2_bw_src, l2_bw_dst
          if (total_memory > std::numeric_limits<size_t>::max() - l2_bw_double) {
            std::cerr << Messages::error_prefix() << Messages::error_total_memory_overflow() << std::endl;
            return EXIT_FAILURE;
          }
          total_memory += l2_bw_double;  // l2_bw_src, l2_bw_dst
        }
      }
    }
  }
  
  // Validate total memory requirement against availability limit.
  // This check is necessary because the per-buffer limit calculation in config.cpp
  // (max_total_allowed_mb / 3) only accounts for the 3 main buffers (src, dst, lat)
  // and does not include cache buffers (L1, L2, custom) and their bandwidth counterparts.
  // This ensures the combined memory usage of all buffers stays within the total limit.
  if (config.max_total_allowed_mb > 0) {
    // Error: Total memory requirement exceeds system limit
    unsigned long total_memory_mb = static_cast<unsigned long>(total_memory / Constants::BYTES_PER_MB);
    if (total_memory_mb > config.max_total_allowed_mb) {
      std::cerr << Messages::error_prefix() << Messages::error_total_memory_exceeds_limit(total_memory_mb, config.max_total_allowed_mb) << std::endl;
      return EXIT_FAILURE;  // Return code: resource limit exceeded
    }
  }
  
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


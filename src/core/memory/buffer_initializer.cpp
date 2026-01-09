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
 * @file buffer_initializer.cpp
 * @brief Buffer initialization logic
 *
 * Orchestrates initialization of all allocated benchmark buffers with appropriate
 * test data or pointer chains. Handles conditional initialization based on
 * benchmark mode and validates that required buffers are allocated.
 *
 * Key features:
 * - Conditional initialization based on benchmark flags (bandwidth/latency/patterns)
 * - Validation that buffers are non-null before initialization
 * - Bandwidth buffers: Filled with deterministic patterns for testing
 * - Latency buffers: Set up with randomized pointer-chasing chains
 * - Support for both main and cache-specific buffer sets
 *
 * Buffer initialization types:
 * - Bandwidth buffers: src (patterned), dst (zeroed) via initialize_buffers()
 * - Latency buffers: Circular pointer chains via setup_latency_chain()
 */

#include "core/memory/buffer_initializer.h"
#include "core/memory/buffer_manager.h"  // BenchmarkBuffers
#include "core/config/config.h"  // BenchmarkConfig
#include "core/memory/memory_utils.h"  // initialize_buffers, setup_latency_chain
#include "core/config/constants.h"
#include "output/console/messages.h"
#include <cstdlib>  // EXIT_SUCCESS, EXIT_FAILURE
#include <iostream> // std::cerr

/**
 * @brief Initializes all allocated benchmark buffers with test data or pointer chains.
 *
 * Orchestrates initialization of all buffers that were allocated by allocate_all_buffers():
 * 1. Validates that required buffers are non-null
 * 2. Initializes main memory buffers conditionally:
 *    - Bandwidth buffers (src, dst): Fill with test patterns
 *    - Latency buffer (lat): Set up pointer-chasing chain
 * 3. Initializes cache buffers conditionally:
 *    - Cache latency buffers (L1/L2/custom): Set up pointer-chasing chains
 *    - Cache bandwidth buffers: Fill with test patterns
 *
 * Buffer initialization is conditional on configuration flags:
 * - only_latency: Skip bandwidth buffer initialization (src, dst)
 * - only_bandwidth: Skip latency buffer initialization (lat, cache latency)
 * - run_patterns: Skip cache buffer initialization (patterns use main buffers only)
 * - use_custom_cache_size: Initialize custom buffers instead of L1/L2
 *
 * Bandwidth buffer initialization:
 * - Source: Filled with repeating 0-255 byte pattern for deterministic testing
 * - Destination: Zeroed to detect copy operations
 *
 * Latency buffer initialization:
 * - Circular pointer chain with randomized ordering
 * - Stride of LATENCY_STRIDE_BYTES between consecutive pointers
 * - Defeats hardware prefetchers for accurate latency measurement
 *
 * @param[in,out] buffers  Buffer management structure with allocated buffers to initialize
 * @param[in]     config   Benchmark configuration with buffer sizes and flags
 *
 * @return EXIT_SUCCESS (0) if all required buffers are initialized successfully
 * @return EXIT_FAILURE (1) if any required buffer is null or initialization fails
 *
 * @note This function must be called after allocate_all_buffers() succeeds.
 * @note Null buffer pointers for optional buffers (based on config flags) are expected.
 * @note Null buffer pointers for required buffers indicate allocation errors.
 *
 * @see allocate_all_buffers() which must be called first
 * @see initialize_buffers() for bandwidth buffer initialization
 * @see setup_latency_chain() for latency buffer initialization
 * @see BenchmarkBuffers for buffer management structure
 * @see BenchmarkConfig for configuration options
 */
int initialize_all_buffers(BenchmarkBuffers& buffers, const BenchmarkConfig& config) {
  // Initialize main memory buffers conditionally
  if (!config.only_latency) {
    // Validate bandwidth buffers are allocated
    if (buffers.src_buffer() == nullptr || buffers.dst_buffer() == nullptr) {
      std::cerr << Messages::error_prefix() << Messages::error_main_buffers_not_allocated() << std::endl;
      return EXIT_FAILURE;
    }
    // Initialize main memory bandwidth buffers
    if (initialize_buffers(buffers.src_buffer(), buffers.dst_buffer(), config.buffer_size) != EXIT_SUCCESS) {
      return EXIT_FAILURE;
    }
  }
  
  if (!config.only_bandwidth && !config.run_patterns) {
    // Validate latency buffer is allocated
    if (buffers.lat_buffer() == nullptr) {
      std::cerr << Messages::error_prefix() << Messages::error_main_buffers_not_allocated() << std::endl;
      return EXIT_FAILURE;
    }
    // Setup main memory latency chain
    if (setup_latency_chain(buffers.lat_buffer(), config.buffer_size, Constants::LATENCY_STRIDE_BYTES) != EXIT_SUCCESS) {
      return EXIT_FAILURE;
    }
  }
  
  // Setup cache latency chains (only if not bandwidth-only and not pattern-only)
  if (!config.only_bandwidth && !config.run_patterns) {
    if (config.use_custom_cache_size) {
      if (config.custom_buffer_size > 0) {
        if (buffers.custom_buffer() == nullptr) {
          std::cerr << Messages::error_prefix() << Messages::error_custom_buffer_not_allocated() << std::endl;
          return EXIT_FAILURE;
        }
        if (setup_latency_chain(buffers.custom_buffer(), config.custom_buffer_size, Constants::LATENCY_STRIDE_BYTES) != EXIT_SUCCESS) {
          return EXIT_FAILURE;
        }
      }
    } else {
      if (config.l1_buffer_size > 0) {
        if (buffers.l1_buffer() == nullptr) {
          std::cerr << Messages::error_prefix() << Messages::error_l1_buffer_not_allocated() << std::endl;
          return EXIT_FAILURE;
        }
        if (setup_latency_chain(buffers.l1_buffer(), config.l1_buffer_size, Constants::LATENCY_STRIDE_BYTES) != EXIT_SUCCESS) {
          return EXIT_FAILURE;
        }
      }
      if (config.l2_buffer_size > 0) {
        if (buffers.l2_buffer() == nullptr) {
          std::cerr << Messages::error_prefix() << Messages::error_l2_buffer_not_allocated() << std::endl;
          return EXIT_FAILURE;
        }
        if (setup_latency_chain(buffers.l2_buffer(), config.l2_buffer_size, Constants::LATENCY_STRIDE_BYTES) != EXIT_SUCCESS) {
          return EXIT_FAILURE;
        }
      }
    }
  }
  
  // Initialize cache bandwidth test buffers (only if not latency-only and not pattern-only)
  if (!config.only_latency && !config.run_patterns) {
    if (config.use_custom_cache_size) {
      if (config.custom_buffer_size > 0) {
        if (buffers.custom_bw_src() == nullptr || buffers.custom_bw_dst() == nullptr) {
          std::cerr << Messages::error_prefix() << Messages::error_custom_bandwidth_buffers_not_allocated() << std::endl;
          return EXIT_FAILURE;
        }
        if (initialize_buffers(buffers.custom_bw_src(), buffers.custom_bw_dst(), config.custom_buffer_size) != EXIT_SUCCESS) {
          return EXIT_FAILURE;
        }
      }
    } else {
      if (config.l1_buffer_size > 0) {
        if (buffers.l1_bw_src() == nullptr || buffers.l1_bw_dst() == nullptr) {
          std::cerr << Messages::error_prefix() << Messages::error_l1_bandwidth_buffers_not_allocated() << std::endl;
          return EXIT_FAILURE;
        }
        if (initialize_buffers(buffers.l1_bw_src(), buffers.l1_bw_dst(), config.l1_buffer_size) != EXIT_SUCCESS) {
          return EXIT_FAILURE;
        }
      }
      if (config.l2_buffer_size > 0) {
        if (buffers.l2_bw_src() == nullptr || buffers.l2_bw_dst() == nullptr) {
          std::cerr << Messages::error_prefix() << Messages::error_l2_bandwidth_buffers_not_allocated() << std::endl;
          return EXIT_FAILURE;
        }
        if (initialize_buffers(buffers.l2_bw_src(), buffers.l2_bw_dst(), config.l2_buffer_size) != EXIT_SUCCESS) {
          return EXIT_FAILURE;
        }
      }
    }
  }

  return EXIT_SUCCESS;
}


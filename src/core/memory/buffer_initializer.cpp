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
#include "core/memory/buffer_initializer.h"
#include "core/memory/buffer_manager.h"  // BenchmarkBuffers
#include "core/config/config.h"  // BenchmarkConfig
#include "core/memory/memory_utils.h"  // initialize_buffers, setup_latency_chain
#include "core/config/constants.h"
#include "output/console/messages.h"
#include <cstdlib>  // EXIT_SUCCESS, EXIT_FAILURE
#include <iostream> // std::cerr

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


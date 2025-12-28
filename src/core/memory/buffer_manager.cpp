// Copyright 2025 Timo Heimonen <timo.heimonen@proton.me>
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
#include "core/memory/buffer_manager.h"
#include "core/config/config.h"  // BenchmarkConfig
#include "core/memory/memory_utils.h"  // initialize_buffers, setup_latency_chain
#include "core/config/constants.h"
#include "output/console/messages.h"
#include <cstdlib>  // EXIT_SUCCESS, EXIT_FAILURE
#include <limits>   // std::numeric_limits
#include <iostream> // std::cerr

int allocate_all_buffers(const BenchmarkConfig& config, BenchmarkBuffers& buffers) {
  // Validate buffer sizes before allocation
  if (config.buffer_size == 0) {
    std::cerr << Messages::error_prefix() << Messages::error_main_buffer_size_zero() << std::endl;
    return EXIT_FAILURE;
  }
  
  // Calculate total memory requirement and check for overflow
  size_t total_memory = 0;
  
  // Main buffers (3x buffer_size)
  if (config.buffer_size > std::numeric_limits<size_t>::max() / 3) {
    std::cerr << Messages::error_prefix() << Messages::error_buffer_size_overflow_calculation() << std::endl;
    return EXIT_FAILURE;
  }
  total_memory += config.buffer_size * 3;  // src, dst, lat
  
  // Cache buffers
  if (config.use_custom_cache_size) {
    if (config.custom_buffer_size > 0) {
      if (total_memory > std::numeric_limits<size_t>::max() - config.custom_buffer_size * 3) {
        std::cerr << Messages::error_prefix() << Messages::error_total_memory_overflow() << std::endl;
        return EXIT_FAILURE;
      }
      total_memory += config.custom_buffer_size * 3;  // custom, custom_bw_src, custom_bw_dst
    }
  } else {
    if (config.l1_buffer_size > 0) {
      if (total_memory > std::numeric_limits<size_t>::max() - config.l1_buffer_size * 3) {
        std::cerr << Messages::error_prefix() << Messages::error_total_memory_overflow() << std::endl;
        return EXIT_FAILURE;
      }
      total_memory += config.l1_buffer_size * 3;  // l1, l1_bw_src, l1_bw_dst
    }
    if (config.l2_buffer_size > 0) {
      if (total_memory > std::numeric_limits<size_t>::max() - config.l2_buffer_size * 3) {
        std::cerr << Messages::error_prefix() << Messages::error_total_memory_overflow() << std::endl;
        return EXIT_FAILURE;
      }
      total_memory += config.l2_buffer_size * 3;  // l2, l2_bw_src, l2_bw_dst
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

  // Allocate source buffer (with cache-discouraging hints if requested)
  if (config.use_non_cacheable) {
    buffers.src_buffer_ptr = allocate_buffer_non_cacheable(config.buffer_size, "src_buffer");
  } else {
    buffers.src_buffer_ptr = allocate_buffer(config.buffer_size, "src_buffer");
  }
  if (!buffers.src_buffer_ptr) {
    return EXIT_FAILURE;
  }

  // Allocate destination buffer (with cache-discouraging hints if requested)
  if (config.use_non_cacheable) {
    buffers.dst_buffer_ptr = allocate_buffer_non_cacheable(config.buffer_size, "dst_buffer");
  } else {
    buffers.dst_buffer_ptr = allocate_buffer(config.buffer_size, "dst_buffer");
  }
  if (!buffers.dst_buffer_ptr) {
    // src_buffer_ptr will be cleaned up automatically upon return
    return EXIT_FAILURE;
  }

  // Allocate latency buffer (with cache-discouraging hints if requested)
  if (config.use_non_cacheable) {
    buffers.lat_buffer_ptr = allocate_buffer_non_cacheable(config.buffer_size, "lat_buffer");
  } else {
    buffers.lat_buffer_ptr = allocate_buffer(config.buffer_size, "lat_buffer");
  }
  if (!buffers.lat_buffer_ptr) {
    // src_buffer_ptr and dst_buffer_ptr will be cleaned up automatically upon return
    return EXIT_FAILURE;
  }

  // Allocate cache latency test buffers (with cache-discouraging hints if requested)
  if (config.use_custom_cache_size) {
    // Allocate custom cache latency test buffer
    if (config.custom_buffer_size > 0) {
      if (config.use_non_cacheable) {
        buffers.custom_buffer_ptr = allocate_buffer_non_cacheable(config.custom_buffer_size, "custom_buffer");
      } else {
        buffers.custom_buffer_ptr = allocate_buffer(config.custom_buffer_size, "custom_buffer");
      }
      if (!buffers.custom_buffer_ptr) {
        return EXIT_FAILURE;
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
      if (!buffers.l1_buffer_ptr) {
        return EXIT_FAILURE;
      }
    }

    if (config.l2_buffer_size > 0) {
      if (config.use_non_cacheable) {
        buffers.l2_buffer_ptr = allocate_buffer_non_cacheable(config.l2_buffer_size, "l2_buffer");
      } else {
        buffers.l2_buffer_ptr = allocate_buffer(config.l2_buffer_size, "l2_buffer");
      }
      if (!buffers.l2_buffer_ptr) {
        return EXIT_FAILURE;
      }
    }
  }

  // Allocate cache bandwidth test buffers (with cache-discouraging hints if requested)
  if (config.use_custom_cache_size) {
    // Allocate custom cache bandwidth test buffers
    if (config.custom_buffer_size > 0) {
      // Allocate source buffer for custom cache bandwidth tests
      if (config.use_non_cacheable) {
        buffers.custom_bw_src_ptr = allocate_buffer_non_cacheable(config.custom_buffer_size, "custom_bw_src_buffer");
      } else {
        buffers.custom_bw_src_ptr = allocate_buffer(config.custom_buffer_size, "custom_bw_src_buffer");
      }
      if (!buffers.custom_bw_src_ptr) {
        return EXIT_FAILURE;
      }
      // Allocate destination buffer for custom cache bandwidth tests
      if (config.use_non_cacheable) {
        buffers.custom_bw_dst_ptr = allocate_buffer_non_cacheable(config.custom_buffer_size, "custom_bw_dst_buffer");
      } else {
        buffers.custom_bw_dst_ptr = allocate_buffer(config.custom_buffer_size, "custom_bw_dst_buffer");
      }
      if (!buffers.custom_bw_dst_ptr) {
        return EXIT_FAILURE;
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
      if (!buffers.l1_bw_src_ptr) {
        return EXIT_FAILURE;
      }
      // Allocate destination buffer for L1 bandwidth tests
      if (config.use_non_cacheable) {
        buffers.l1_bw_dst_ptr = allocate_buffer_non_cacheable(config.l1_buffer_size, "l1_bw_dst_buffer");
      } else {
        buffers.l1_bw_dst_ptr = allocate_buffer(config.l1_buffer_size, "l1_bw_dst_buffer");
      }
      if (!buffers.l1_bw_dst_ptr) {
        return EXIT_FAILURE;
      }
    }

    if (config.l2_buffer_size > 0) {
      // Allocate source buffer for L2 bandwidth tests
      if (config.use_non_cacheable) {
        buffers.l2_bw_src_ptr = allocate_buffer_non_cacheable(config.l2_buffer_size, "l2_bw_src_buffer");
      } else {
        buffers.l2_bw_src_ptr = allocate_buffer(config.l2_buffer_size, "l2_bw_src_buffer");
      }
      if (!buffers.l2_bw_src_ptr) {
        return EXIT_FAILURE;
      }
      // Allocate destination buffer for L2 bandwidth tests
      if (config.use_non_cacheable) {
        buffers.l2_bw_dst_ptr = allocate_buffer_non_cacheable(config.l2_buffer_size, "l2_bw_dst_buffer");
      } else {
        buffers.l2_bw_dst_ptr = allocate_buffer(config.l2_buffer_size, "l2_bw_dst_buffer");
      }
      if (!buffers.l2_bw_dst_ptr) {
        return EXIT_FAILURE;
      }
    }
  }

  return EXIT_SUCCESS;
}

int initialize_all_buffers(BenchmarkBuffers& buffers, const BenchmarkConfig& config) {
  // Validate main buffers are allocated
  if (buffers.src_buffer() == nullptr || buffers.dst_buffer() == nullptr || buffers.lat_buffer() == nullptr) {
    std::cerr << Messages::error_prefix() << Messages::error_main_buffers_not_allocated() << std::endl;
    return EXIT_FAILURE;
  }
  
  // Initialize main memory buffers
  if (initialize_buffers(buffers.src_buffer(), buffers.dst_buffer(), config.buffer_size) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  
  if (setup_latency_chain(buffers.lat_buffer(), config.buffer_size, Constants::LATENCY_STRIDE_BYTES) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  
  // Setup cache latency chains
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
  
  // Initialize cache bandwidth test buffers (not pointer chains, regular data)
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

  return EXIT_SUCCESS;
}


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
#include "buffer_manager.h"
#include "config.h"  // BenchmarkConfig
#include "benchmark.h"  // initialize_buffers, setup_latency_chain
#include "constants.h"
#include <cstdlib>  // EXIT_SUCCESS, EXIT_FAILURE

int allocate_all_buffers(const BenchmarkConfig& config, BenchmarkBuffers& buffers) {
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

  // Allocate source buffer
  buffers.src_buffer_ptr = allocate_buffer(config.buffer_size, "src_buffer");
  if (!buffers.src_buffer_ptr) {
    return EXIT_FAILURE;
  }

  // Allocate destination buffer
  buffers.dst_buffer_ptr = allocate_buffer(config.buffer_size, "dst_buffer");
  if (!buffers.dst_buffer_ptr) {
    // src_buffer_ptr will be cleaned up automatically upon return
    return EXIT_FAILURE;
  }

  // Allocate latency buffer
  buffers.lat_buffer_ptr = allocate_buffer(config.buffer_size, "lat_buffer");
  if (!buffers.lat_buffer_ptr) {
    // src_buffer_ptr and dst_buffer_ptr will be cleaned up automatically upon return
    return EXIT_FAILURE;
  }

  // Allocate cache latency test buffers
  if (config.use_custom_cache_size) {
    // Allocate custom cache latency test buffer
    if (config.custom_buffer_size > 0) {
      buffers.custom_buffer_ptr = allocate_buffer(config.custom_buffer_size, "custom_buffer");
      if (!buffers.custom_buffer_ptr) {
        return EXIT_FAILURE;
      }
    }
  } else {
    // Allocate L1/L2 cache latency test buffers
    if (config.l1_buffer_size > 0) {
      buffers.l1_buffer_ptr = allocate_buffer(config.l1_buffer_size, "l1_buffer");
      if (!buffers.l1_buffer_ptr) {
        return EXIT_FAILURE;
      }
    }

    if (config.l2_buffer_size > 0) {
      buffers.l2_buffer_ptr = allocate_buffer(config.l2_buffer_size, "l2_buffer");
      if (!buffers.l2_buffer_ptr) {
        return EXIT_FAILURE;
      }
    }
  }

  // Allocate cache bandwidth test buffers
  if (config.use_custom_cache_size) {
    // Allocate custom cache bandwidth test buffers
    if (config.custom_buffer_size > 0) {
      // Allocate source buffer for custom cache bandwidth tests
      buffers.custom_bw_src_ptr = allocate_buffer(config.custom_buffer_size, "custom_bw_src_buffer");
      if (!buffers.custom_bw_src_ptr) {
        return EXIT_FAILURE;
      }
      // Allocate destination buffer for custom cache bandwidth tests
      buffers.custom_bw_dst_ptr = allocate_buffer(config.custom_buffer_size, "custom_bw_dst_buffer");
      if (!buffers.custom_bw_dst_ptr) {
        return EXIT_FAILURE;
      }
    }
  } else {
    // Allocate L1/L2 cache bandwidth test buffers
    if (config.l1_buffer_size > 0) {
      // Allocate source buffer for L1 bandwidth tests
      buffers.l1_bw_src_ptr = allocate_buffer(config.l1_buffer_size, "l1_bw_src_buffer");
      if (!buffers.l1_bw_src_ptr) {
        return EXIT_FAILURE;
      }
      // Allocate destination buffer for L1 bandwidth tests
      buffers.l1_bw_dst_ptr = allocate_buffer(config.l1_buffer_size, "l1_bw_dst_buffer");
      if (!buffers.l1_bw_dst_ptr) {
        return EXIT_FAILURE;
      }
    }

    if (config.l2_buffer_size > 0) {
      // Allocate source buffer for L2 bandwidth tests
      buffers.l2_bw_src_ptr = allocate_buffer(config.l2_buffer_size, "l2_bw_src_buffer");
      if (!buffers.l2_bw_src_ptr) {
        return EXIT_FAILURE;
      }
      // Allocate destination buffer for L2 bandwidth tests
      buffers.l2_bw_dst_ptr = allocate_buffer(config.l2_buffer_size, "l2_bw_dst_buffer");
      if (!buffers.l2_bw_dst_ptr) {
        return EXIT_FAILURE;
      }
    }
  }

  return EXIT_SUCCESS;
}

int initialize_all_buffers(BenchmarkBuffers& buffers, const BenchmarkConfig& config) {
  // Initialize main memory buffers
  initialize_buffers(buffers.src_buffer(), buffers.dst_buffer(), config.buffer_size);
  setup_latency_chain(buffers.lat_buffer(), config.buffer_size, Constants::LATENCY_STRIDE_BYTES);
  
  // Setup cache latency chains
  if (config.use_custom_cache_size) {
    if (config.custom_buffer_size > 0 && buffers.custom_buffer() != nullptr) {
      setup_latency_chain(buffers.custom_buffer(), config.custom_buffer_size, Constants::LATENCY_STRIDE_BYTES);
    }
  } else {
    if (config.l1_buffer_size > 0 && buffers.l1_buffer() != nullptr) {
      setup_latency_chain(buffers.l1_buffer(), config.l1_buffer_size, Constants::LATENCY_STRIDE_BYTES);
    }
    if (config.l2_buffer_size > 0 && buffers.l2_buffer() != nullptr) {
      setup_latency_chain(buffers.l2_buffer(), config.l2_buffer_size, Constants::LATENCY_STRIDE_BYTES);
    }
  }
  
  // Initialize cache bandwidth test buffers (not pointer chains, regular data)
  if (config.use_custom_cache_size) {
    if (config.custom_buffer_size > 0 && buffers.custom_bw_src() != nullptr && buffers.custom_bw_dst() != nullptr) {
      initialize_buffers(buffers.custom_bw_src(), buffers.custom_bw_dst(), config.custom_buffer_size);
    }
  } else {
    if (config.l1_buffer_size > 0 && buffers.l1_bw_src() != nullptr && buffers.l1_bw_dst() != nullptr) {
      initialize_buffers(buffers.l1_bw_src(), buffers.l1_bw_dst(), config.l1_buffer_size);
    }
    if (config.l2_buffer_size > 0 && buffers.l2_bw_src() != nullptr && buffers.l2_bw_dst() != nullptr) {
      initialize_buffers(buffers.l2_bw_src(), buffers.l2_bw_dst(), config.l2_buffer_size);
    }
  }

  return EXIT_SUCCESS;
}


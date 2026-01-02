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
 * @file buffer_manager.h
 * @brief Buffer management for benchmark memory allocations
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 *
 * This header provides structures and functions to manage all benchmark buffers,
 * including allocation, initialization, and accessor methods.
 */
#ifndef BUFFER_MANAGER_H
#define BUFFER_MANAGER_H

#include "core/memory/memory_manager.h"  // MmapPtr
#include "core/memory/buffer_allocator.h"  // allocate_all_buffers
#include "core/memory/buffer_initializer.h"  // initialize_all_buffers
#include <cstddef>  // size_t
#include <cstdlib>  // EXIT_SUCCESS, EXIT_FAILURE

// Forward declaration to avoid including config.h in header
struct BenchmarkConfig;

/**
 * @struct BenchmarkBuffers
 * @brief Structure containing all benchmark buffers
 *
 * All buffers are managed via unique_ptr (MmapPtr) and will be automatically
 * freed when the structure is destroyed. Provides helper methods to access
 * raw pointers for compatibility with C-style function interfaces.
 */
struct BenchmarkBuffers {
  // Main memory buffers
  MmapPtr src_buffer_ptr;   ///< Source buffer for read/copy tests
  MmapPtr dst_buffer_ptr;   ///< Destination buffer for write/copy tests
  MmapPtr lat_buffer_ptr;   ///< Latency test buffer
  
  // Cache latency test buffers
  MmapPtr l1_buffer_ptr;    ///< L1 cache latency test buffer
  MmapPtr l2_buffer_ptr;    ///< L2 cache latency test buffer
  MmapPtr custom_buffer_ptr;  ///< Custom cache latency test buffer
  
  // Cache bandwidth test buffers (source and destination)
  MmapPtr l1_bw_src_ptr;    ///< L1 cache bandwidth test source buffer
  MmapPtr l1_bw_dst_ptr;    ///< L1 cache bandwidth test destination buffer
  MmapPtr l2_bw_src_ptr;    ///< L2 cache bandwidth test source buffer
  MmapPtr l2_bw_dst_ptr;    ///< L2 cache bandwidth test destination buffer
  MmapPtr custom_bw_src_ptr;  ///< Custom cache bandwidth test source buffer
  MmapPtr custom_bw_dst_ptr;  ///< Custom cache bandwidth test destination buffer
  
  // Helper methods to get raw pointers (for functions that need void*)
  /**
   * @brief Get raw pointer to source buffer
   * @return Raw void* pointer to source buffer
   */
  void* src_buffer() const { return src_buffer_ptr.get(); }
  
  /**
   * @brief Get raw pointer to destination buffer
   * @return Raw void* pointer to destination buffer
   */
  void* dst_buffer() const { return dst_buffer_ptr.get(); }
  
  /**
   * @brief Get raw pointer to latency test buffer
   * @return Raw void* pointer to latency test buffer
   */
  void* lat_buffer() const { return lat_buffer_ptr.get(); }
  
  /**
   * @brief Get raw pointer to L1 cache latency buffer
   * @return Raw void* pointer to L1 cache latency buffer
   */
  void* l1_buffer() const { return l1_buffer_ptr.get(); }
  
  /**
   * @brief Get raw pointer to L2 cache latency buffer
   * @return Raw void* pointer to L2 cache latency buffer
   */
  void* l2_buffer() const { return l2_buffer_ptr.get(); }
  
  /**
   * @brief Get raw pointer to custom cache latency buffer
   * @return Raw void* pointer to custom cache latency buffer
   */
  void* custom_buffer() const { return custom_buffer_ptr.get(); }
  
  /**
   * @brief Get raw pointer to L1 cache bandwidth source buffer
   * @return Raw void* pointer to L1 cache bandwidth source buffer
   */
  void* l1_bw_src() const { return l1_bw_src_ptr.get(); }
  
  /**
   * @brief Get raw pointer to L1 cache bandwidth destination buffer
   * @return Raw void* pointer to L1 cache bandwidth destination buffer
   */
  void* l1_bw_dst() const { return l1_bw_dst_ptr.get(); }
  
  /**
   * @brief Get raw pointer to L2 cache bandwidth source buffer
   * @return Raw void* pointer to L2 cache bandwidth source buffer
   */
  void* l2_bw_src() const { return l2_bw_src_ptr.get(); }
  
  /**
   * @brief Get raw pointer to L2 cache bandwidth destination buffer
   * @return Raw void* pointer to L2 cache bandwidth destination buffer
   */
  void* l2_bw_dst() const { return l2_bw_dst_ptr.get(); }
  
  /**
   * @brief Get raw pointer to custom cache bandwidth source buffer
   * @return Raw void* pointer to custom cache bandwidth source buffer
   */
  void* custom_bw_src() const { return custom_bw_src_ptr.get(); }
  
  /**
   * @brief Get raw pointer to custom cache bandwidth destination buffer
   * @return Raw void* pointer to custom cache bandwidth destination buffer
   */
  void* custom_bw_dst() const { return custom_bw_dst_ptr.get(); }
};

#endif // BUFFER_MANAGER_H


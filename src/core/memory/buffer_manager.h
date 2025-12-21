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
#ifndef BUFFER_MANAGER_H
#define BUFFER_MANAGER_H

#include "core/memory/memory_manager.h"  // MmapPtr
#include <cstddef>  // size_t
#include <cstdlib>  // EXIT_SUCCESS, EXIT_FAILURE

// Forward declaration to avoid including config.h in header
struct BenchmarkConfig;

// Structure containing all benchmark buffers
// All buffers are managed via unique_ptr and will be automatically freed
struct BenchmarkBuffers {
  // Main memory buffers
  MmapPtr src_buffer_ptr;   // Source buffer for read/copy tests
  MmapPtr dst_buffer_ptr;   // Destination buffer for write/copy tests
  MmapPtr lat_buffer_ptr;   // Latency test buffer
  
  // Cache latency test buffers
  MmapPtr l1_buffer_ptr;    // L1 cache latency test buffer
  MmapPtr l2_buffer_ptr;    // L2 cache latency test buffer
  MmapPtr custom_buffer_ptr;  // Custom cache latency test buffer
  
  // Cache bandwidth test buffers (source and destination)
  MmapPtr l1_bw_src_ptr;    // L1 cache bandwidth test source buffer
  MmapPtr l1_bw_dst_ptr;    // L1 cache bandwidth test destination buffer
  MmapPtr l2_bw_src_ptr;    // L2 cache bandwidth test source buffer
  MmapPtr l2_bw_dst_ptr;    // L2 cache bandwidth test destination buffer
  MmapPtr custom_bw_src_ptr;  // Custom cache bandwidth test source buffer
  MmapPtr custom_bw_dst_ptr;  // Custom cache bandwidth test destination buffer
  
  // Helper methods to get raw pointers (for functions that need void*)
  void* src_buffer() const { return src_buffer_ptr.get(); }
  void* dst_buffer() const { return dst_buffer_ptr.get(); }
  void* lat_buffer() const { return lat_buffer_ptr.get(); }
  void* l1_buffer() const { return l1_buffer_ptr.get(); }
  void* l2_buffer() const { return l2_buffer_ptr.get(); }
  void* custom_buffer() const { return custom_buffer_ptr.get(); }
  void* l1_bw_src() const { return l1_bw_src_ptr.get(); }
  void* l1_bw_dst() const { return l1_bw_dst_ptr.get(); }
  void* l2_bw_src() const { return l2_bw_src_ptr.get(); }
  void* l2_bw_dst() const { return l2_bw_dst_ptr.get(); }
  void* custom_bw_src() const { return custom_bw_src_ptr.get(); }
  void* custom_bw_dst() const { return custom_bw_dst_ptr.get(); }
};

// Allocate all buffers based on configuration
// Returns EXIT_SUCCESS on success, EXIT_FAILURE on error
int allocate_all_buffers(const BenchmarkConfig& config, BenchmarkBuffers& buffers);

// Initialize all buffers (fill data and setup latency chains)
// Returns EXIT_SUCCESS on success, EXIT_FAILURE on error
int initialize_all_buffers(BenchmarkBuffers& buffers, const BenchmarkConfig& config);

#endif // BUFFER_MANAGER_H


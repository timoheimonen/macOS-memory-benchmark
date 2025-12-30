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
#ifndef WARMUP_INTERNAL_H
#define WARMUP_INTERNAL_H

#include <atomic>
#include <iostream>
#include <thread>
#include <vector>
#include <cstddef>

// macOS specific QoS
#include <mach/mach.h>
#include <pthread/qos.h>

#include "output/console/messages.h"
#include "core/memory/memory_utils.h"

// Forward declaration for join_threads (defined in utils.cpp, declared in benchmark.h)
void join_threads(std::vector<std::thread>& threads);

// Helper function to calculate warmup size: min(buffer_size, max(64MB, buffer_size * 0.1))
// This ensures meaningful warmup for small buffers while preventing excessive overhead on large buffers.
inline size_t calculate_warmup_size(size_t buffer_size) {
  const size_t min_warmup = 64 * 1024 * 1024;  // 64MB
  const size_t percent_warmup = static_cast<size_t>(buffer_size * 0.1);
  const size_t effective_warmup = (min_warmup > percent_warmup) ? min_warmup : percent_warmup;
  return (buffer_size < effective_warmup) ? buffer_size : effective_warmup;
}

// Generic parallel warmup function for multi-threaded operations.
// 'buffer': Primary memory region (or destination for copy operations).
// 'size': Total size of the buffer in bytes.
// 'num_threads': Number of concurrent threads to use.
// 'chunk_operation': Function to execute on each chunk (takes chunk_start and chunk_size).
// 'set_qos': Whether to set QoS class for worker threads (default: false).
// 'src_buffer': Optional source buffer for copy operations (default: nullptr).
// 'dummy_checksum': Optional atomic variable to accumulate checksum results (default: nullptr).
// 'warmup_size': Optional warmup size limit (default: calculated from buffer_size).
template<typename ChunkOp>
void warmup_parallel(void* buffer, size_t size, int num_threads, ChunkOp chunk_operation, 
                     bool set_qos = false, void* src_buffer = nullptr, 
                     std::atomic<uint64_t>* dummy_checksum = nullptr, size_t warmup_size = 0) {
  // Calculate warmup size if not provided
  if (warmup_size == 0) {
    warmup_size = calculate_warmup_size(size);
  }
  // Use the smaller of buffer size and warmup size
  size_t effective_size = (size < warmup_size) ? size : warmup_size;
  
  // Vector to store thread objects.
  std::vector<std::thread> threads;
  // Pre-allocate space for efficiency.
  threads.reserve(num_threads);
  // Tracks the current position in the buffer.
  size_t offset = 0;
  // Calculate the base size of the memory chunk per thread.
  size_t chunk_base_size = effective_size / num_threads;
  // Calculate the remainder to distribute among threads.
  size_t chunk_remainder = effective_size % num_threads;

  // Early validation: return early if no work to do or invalid thread count.
  // This avoids unnecessary thread creation overhead when no meaningful work can be performed.
  if (effective_size == 0 || num_threads <= 0) {
    return;  // No work to do or invalid thread count
  }

  // Create and launch threads.
  for (int t = 0; t < num_threads; ++t) {
    // Calculate the exact size for this thread's chunk, adding remainder if needed.
    size_t original_chunk_size = chunk_base_size + (t < chunk_remainder ? 1 : 0);
    // If chunk size is zero (e.g., size < num_threads), skip creating a thread.
    if (original_chunk_size == 0) continue;
    
    // Calculate the original end position for this chunk
    size_t original_chunk_end = offset + original_chunk_size;
    
    // Calculate the start address for this thread's chunk.
    char* unaligned_start = static_cast<char*>(buffer) + offset;
    // Align chunk_start to cache line boundary to prevent false sharing
    char* chunk_start = static_cast<char*>(align_ptr_to_cache_line(unaligned_start));
    
    // Calculate the source start address for copy operations.
    // Maintain the same offset relationship between src and dst to preserve data correspondence
    char* src_chunk = nullptr;
    if (src_buffer) {
      char* src_unaligned_start = static_cast<char*>(src_buffer) + offset;
      // Apply the same alignment offset to maintain src/dst correspondence
      size_t alignment_offset = chunk_start - unaligned_start;
      src_chunk = src_unaligned_start + alignment_offset;
    }
    
    // Calculate chunk size to cover the original range [offset, offset + original_chunk_size)
    // This ensures no gaps: chunk covers from aligned start to original end
    char* original_end_ptr = static_cast<char*>(buffer) + original_chunk_end;
    size_t current_chunk_size = original_end_ptr - chunk_start;
    
    // Ensure we don't exceed the effective buffer size
    char* buffer_end = static_cast<char*>(buffer) + effective_size;
    if (chunk_start + current_chunk_size > buffer_end) {
      current_chunk_size = buffer_end - chunk_start;
    }
    
    // Skip if chunk size is zero or negative after alignment adjustments
    if (current_chunk_size == 0 || chunk_start >= buffer_end) {
      // Still update offset to prevent infinite loops, but skip thread creation
      offset = original_chunk_end;
      continue;
    }

    // Create a new thread to execute the chunk operation.
    threads.emplace_back([chunk_start, src_chunk, current_chunk_size, chunk_operation, set_qos, dummy_checksum]() {
      // Set QoS for this worker thread if requested.
      if (set_qos) {
        kern_return_t qos_ret = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
        if (qos_ret != KERN_SUCCESS) {
          std::cerr << Messages::warning_prefix() << Messages::warning_qos_failed_worker_thread(qos_ret) << std::endl;
        }
      }
      // Execute the chunk operation.
      chunk_operation(chunk_start, src_chunk, current_chunk_size, dummy_checksum);
    });
    
    // Move the offset for the next thread's chunk using the actual end position
    offset = original_chunk_end;
  }

  // Check if any threads were created. If all chunks were zero after alignment,
  // no threads were created and we should return early to avoid unnecessary overhead.
  if (threads.empty()) {
    return;  // No threads created (all chunks were zero after alignment)
  }

  // Wait for all worker threads to complete.
  join_threads(threads);
}

// Generic single-threaded warmup function.
// 'operation': Function to execute (takes no parameters or specific parameters based on operation type).
template<typename Op>
void warmup_single(Op operation) {
  // Set QoS for single-threaded warmup operations to ensure highest priority
  kern_return_t qos_ret = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  if (qos_ret != KERN_SUCCESS) {
    std::cerr << Messages::warning_prefix() << Messages::warning_qos_failed(qos_ret) << std::endl;
  }
  operation();
}

#endif // WARMUP_INTERNAL_H


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
#include "core/config/constants.h"

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
  
  // Early validation: return early if no work to do or invalid thread count.
  // This avoids unnecessary thread creation overhead when no meaningful work can be performed.
  if (effective_size == 0 || num_threads <= 0) {
    return;  // No work to do or invalid thread count
  }

  // Vector to store thread objects.
  std::vector<std::thread> threads;
  // Pre-allocate space for efficiency.
  threads.reserve(num_threads);

  // Process unaligned prefix bytes before the main loop to ensure full buffer coverage
  // This ensures that bytes between buffer and the first cache-line boundary are warmed up
  char* buffer_start = static_cast<char*>(buffer);
  char* aligned_buffer_start = static_cast<char*>(align_ptr_to_cache_line(buffer_start));
  size_t prefix_size = aligned_buffer_start - buffer_start;
  
  // Clamp prefix_size to effective_size to handle tiny buffers (smaller than alignment gap)
  // For very small buffers, process the entire buffer as a single unaligned chunk
  if (prefix_size >= effective_size) {
    // Buffer is smaller than alignment gap or starts unaligned and entire buffer fits in prefix
    char* src_prefix = nullptr;
    if (src_buffer) {
      src_prefix = static_cast<char*>(src_buffer);
    }
    // Process entire buffer as single unaligned chunk
    chunk_operation(buffer_start, src_prefix, effective_size, dummy_checksum);
    return;  // All work done
  }
  
  // Process prefix if it exists (prefix_size < effective_size at this point)
  if (prefix_size > 0) {
    char* src_prefix = nullptr;
    if (src_buffer) {
      src_prefix = static_cast<char*>(src_buffer);
    }
    // Process prefix on main thread to ensure full coverage
    chunk_operation(buffer_start, src_prefix, prefix_size, dummy_checksum);
  }
  
  // Calculate remaining size after prefix processing
  size_t remaining_size = effective_size - prefix_size;
  
  // Tracks the current position in the buffer, starting from the first cache-line aligned position
  size_t offset = prefix_size;
  // Calculate the base size of the memory chunk per thread for remaining work
  size_t chunk_base_size = remaining_size / num_threads;
  // Calculate the remainder to distribute among threads
  size_t chunk_remainder = remaining_size % num_threads;

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
    // Calculate alignment offset early to validate chunk size
    size_t alignment_offset = alignment_offset_to_cache_line(unaligned_start);
    
    // Process alignment gap (bytes between unaligned_start and aligned chunk_start) before thread launch
    // This ensures every byte in [offset, original_chunk_end) is warmed, preventing cold gaps
    if (alignment_offset > 0 && alignment_offset < original_chunk_size) {
      // Process the alignment gap on main thread to ensure full coverage
      // (Only when there's a gap AND an aligned portion remains)
      char* src_gap = nullptr;
      if (src_buffer) {
        src_gap = static_cast<char*>(src_buffer) + offset;
      }
      chunk_operation(unaligned_start, src_gap, alignment_offset, dummy_checksum);
    }
    
    // If the entire chunk is within the alignment gap, process it all unaligned
    if (original_chunk_size <= alignment_offset) {
      // Process the entire chunk unaligned on main thread to prevent unwarmed holes
      char* src_chunk = nullptr;
      if (src_buffer) {
        src_chunk = static_cast<char*>(src_buffer) + offset;
      }
      chunk_operation(unaligned_start, src_chunk, original_chunk_size, dummy_checksum);
      // Then skip thread creation (no aligned portion to process)
      offset = original_chunk_end;
      continue;
    }
    
    // Align chunk_start to cache line boundary to prevent false sharing
    char* chunk_start = static_cast<char*>(align_ptr_to_cache_line(unaligned_start));
    
    // Calculate the source start address for copy operations.
    // Maintain the same offset relationship between src and dst to preserve data correspondence
    char* src_chunk = nullptr;
    if (src_buffer) {
      char* src_unaligned_start = static_cast<char*>(src_buffer) + offset;
      // Apply the same alignment offset to maintain src/dst correspondence
      src_chunk = src_unaligned_start + alignment_offset;
    }
    
    // Calculate the original end pointer for this chunk
    char* original_end_ptr = static_cast<char*>(buffer) + original_chunk_end;
    char* buffer_end = static_cast<char*>(buffer) + effective_size;
    
    // Validate that chunk_start is before original_end_ptr to prevent underflow
    if (chunk_start >= original_end_ptr) {
      // Alignment pushed start beyond end - skip this chunk
      offset = original_chunk_end;
      continue;
    }
    
    // Calculate chunk_end to cover the original range [offset, offset + original_chunk_size)
    // Use original_end_ptr to ensure full coverage of the original chunk
    char* chunk_end = original_end_ptr;
    
    // Calculate chunk size from aligned start to original end
    size_t current_chunk_size = chunk_end - chunk_start;
    
    // Ensure we don't exceed the effective buffer size
    if (chunk_start + current_chunk_size > buffer_end) {
      current_chunk_size = (chunk_start < buffer_end) ? (buffer_end - chunk_start) : 0;
    }
    
    // Skip if chunk size is zero or chunk_start is beyond buffer_end
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
    
    // Move offset forward using original chunk end to ensure forward progress
    // This prevents negative pointer differences when aligned_end_ptr would be before buffer
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


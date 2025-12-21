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
#include <atomic>    // For std::atomic
#include <iostream>  // For std::cout
#include <thread>    // For std::thread
#include <vector>    // For std::vector
#include <unistd.h>  // For getpagesize()

// macOS specific QoS
#include <mach/mach.h>    // For kern_return_t
#include <pthread/qos.h>  // For pthread_set_qos_class_self_np

#include "benchmark.h"  // Includes definitions for assembly loops etc.
#include "messages.h"   // For error and warning messages

// Helper function to calculate warmup size: min(buffer_size, max(64MB, buffer_size * 0.1))
// This ensures meaningful warmup for small buffers while preventing excessive overhead on large buffers.
static size_t calculate_warmup_size(size_t buffer_size) {
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

  // Create and launch threads.
  for (int t = 0; t < num_threads; ++t) {
    // Calculate the exact size for this thread's chunk, adding remainder if needed.
    size_t current_chunk_size = chunk_base_size + (t < chunk_remainder ? 1 : 0);
    // If chunk size is zero (e.g., size < num_threads), skip creating a thread.
    if (current_chunk_size == 0) continue;
    // Calculate the start address for this thread's chunk.
    char* chunk_start = static_cast<char*>(buffer) + offset;
    // Calculate the source start address for copy operations.
    char* src_chunk = src_buffer ? static_cast<char*>(src_buffer) + offset : nullptr;

    // Create a new thread to execute the chunk operation.
    threads.emplace_back([chunk_start, src_chunk, current_chunk_size, chunk_operation, set_qos, dummy_checksum]() {
      // Set QoS for this worker thread if requested.
      if (set_qos) {
        kern_return_t qos_ret = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
        if (qos_ret != KERN_SUCCESS) {
          std::cerr << Messages::warning_qos_failed_worker_thread(qos_ret) << std::endl;
        }
      }
      // Execute the chunk operation.
      chunk_operation(chunk_start, src_chunk, current_chunk_size, dummy_checksum);
    });
    // Move the offset for the next thread's chunk.
    offset += current_chunk_size;
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
    std::cerr << Messages::warning_qos_failed(qos_ret) << std::endl;
  }
  operation();
}

// Warms up memory by reading from the buffer using multiple threads.
// 'buffer': Memory region to read from.
// 'size': Total size of the buffer in bytes.
// 'num_threads': Number of concurrent threads to use.
// 'dummy_checksum': Atomic variable to accumulate a dummy result (prevents optimization).
void warmup_read(void* buffer, size_t size, int num_threads, std::atomic<uint64_t>& dummy_checksum) {
  size_t warmup_size = calculate_warmup_size(size);
  auto read_chunk_op = [](char* chunk_start, char* /* src_chunk */, size_t chunk_size, 
                          std::atomic<uint64_t>* checksum) {
    // Call the assembly read loop function (defined elsewhere).
    uint64_t result = memory_read_loop_asm(chunk_start, chunk_size);
    // Atomically combine the local checksum with the global dummy value.
    // Using relaxed memory order as strict consistency isn't needed for warmup.
    if (checksum) {
      checksum->fetch_xor(result, std::memory_order_relaxed);
    }
  };
  warmup_parallel(buffer, size, num_threads, read_chunk_op, true, nullptr, &dummy_checksum, warmup_size);
}

// Warms up memory by writing to the buffer using multiple threads.
// 'buffer': Memory region to write to.
// 'size': Total size of the buffer in bytes.
// 'num_threads': Number of concurrent threads to use.
void warmup_write(void* buffer, size_t size, int num_threads) {
  size_t warmup_size = calculate_warmup_size(size);
  auto write_chunk_op = [](char* chunk_start, char* /* src_chunk */, size_t chunk_size, 
                           std::atomic<uint64_t>* /* checksum */) {
    memory_write_loop_asm(chunk_start, chunk_size);
  };
  warmup_parallel(buffer, size, num_threads, write_chunk_op, true, nullptr, nullptr, warmup_size);
}

// Warms up memory by copying data between buffers using multiple threads.
// 'dst': Destination memory region.
// 'src': Source memory region.
// 'size': Total size of data to copy in bytes.
// 'num_threads': Number of concurrent threads to use.
void warmup_copy(void* dst, void* src, size_t size, int num_threads) {
  size_t warmup_size = calculate_warmup_size(size);
  auto copy_chunk_op = [](char* dst_chunk, char* src_chunk, size_t chunk_size, 
                          std::atomic<uint64_t>* /* checksum */) {
    memory_copy_loop_asm(dst_chunk, src_chunk, chunk_size);
  };
  warmup_parallel(dst, size, num_threads, copy_chunk_op, true, src, nullptr, warmup_size);
}

// Warms up memory for latency test by page prefaulting (single thread).
// This ensures pages are mapped and page faults are removed, but does not
// build/run the pointer chain, allowing for more accurate "cold-ish" latency measurements.
// 'buffer': Memory region to warm up.
// 'buffer_size': Total size of the buffer in bytes.
void warmup_latency(void* buffer, size_t buffer_size) {
  auto latency_op = [buffer, buffer_size]() {
    // Only perform warmup if buffer is valid.
    if (buffer == nullptr || buffer_size == 0) {
      return;
    }
    
    // Get actual system page size (4KB on most systems, 16KB on some Apple Silicon)
    const size_t page_size = static_cast<size_t>(getpagesize());
    char* buf = static_cast<char*>(buffer);
    
    // Touch each page with a 1-byte read/write to ensure it's mapped
    // This removes page faults without building the pointer chain
    for (size_t offset = 0; offset < buffer_size; offset += page_size) {
      // Read one byte to trigger page mapping
      volatile char dummy = buf[offset];
      (void)dummy;  // Prevent optimization
      // Write one byte to ensure write access is established
      buf[offset] = buf[offset];
    }
  };
  warmup_single(latency_op);
}

// Warms up memory for cache latency test by page prefaulting (single thread).
// This ensures pages are mapped and page faults are removed, but does not
// build/run the pointer chain, allowing for more accurate "cold-ish" cache latency measurements.
// 'buffer': Memory region to warm up.
// 'buffer_size': Total size of the buffer in bytes.
void warmup_cache_latency(void* buffer, size_t buffer_size) {
  // Use the same implementation as warmup_latency since they're identical.
  warmup_latency(buffer, buffer_size);
}

// Warms up cache bandwidth test by reading from the buffer (single thread).
// 'src_buffer': Source memory region to read from (cache-sized).
// 'size': Total size of the buffer in bytes (cache size).
// 'num_threads': Unused parameter (kept for API compatibility).
// 'dummy_checksum': Atomic variable to accumulate a dummy result (prevents optimization).
void warmup_cache_read(void* src_buffer, size_t size, int num_threads, std::atomic<uint64_t>& dummy_checksum) {
  (void)num_threads;  // Unused parameter
  auto read_op = [src_buffer, size, &dummy_checksum]() {
    // Call the assembly read loop function directly (single-threaded).
    uint64_t checksum = memory_read_loop_asm(src_buffer, size);
    // Store result to prevent optimization.
    dummy_checksum.fetch_xor(checksum, std::memory_order_relaxed);
  };
  warmup_single(read_op);
}

// Warms up cache bandwidth test by writing to the buffer (single thread).
// 'dst_buffer': Destination memory region to write to (cache-sized).
// 'size': Total size of the buffer in bytes (cache size).
// 'num_threads': Unused parameter (kept for API compatibility).
void warmup_cache_write(void* dst_buffer, size_t size, int num_threads) {
  (void)num_threads;  // Unused parameter
  auto write_op = [dst_buffer, size]() {
    // Call the assembly write loop function directly (single-threaded).
    memory_write_loop_asm(dst_buffer, size);
  };
  warmup_single(write_op);
}

// Warms up cache bandwidth test by copying data between buffers (single thread).
// 'dst': Destination memory region (cache-sized).
// 'src': Source memory region (cache-sized).
// 'size': Total size of data to copy in bytes (cache size).
// 'num_threads': Unused parameter (kept for API compatibility).
void warmup_cache_copy(void* dst, void* src, size_t size, int num_threads) {
  (void)num_threads;  // Unused parameter
  auto copy_op = [dst, src, size]() {
    // Call the assembly copy loop function directly (single-threaded).
    memory_copy_loop_asm(dst, src, size);
  };
  warmup_single(copy_op);
}

// Warms up memory by reading from the buffer using strided access pattern.
// 'buffer': Memory region to read from.
// 'size': Total size of the buffer in bytes.
// 'stride': Stride size in bytes (must be >= 32, aligned, and <= size).
// 'num_threads': Number of concurrent threads to use.
// 'dummy_checksum': Atomic variable to accumulate a dummy result (prevents optimization).
void warmup_read_strided(void* buffer, size_t size, size_t stride, int num_threads, std::atomic<uint64_t>& dummy_checksum) {
  // Validate stride
  if (stride < 32) {
    std::cerr << Messages::error_prefix() << Messages::error_stride_too_small() << std::endl;
    return;
  }
  if (stride > size) {
    std::cerr << Messages::error_prefix() << Messages::error_stride_too_large(stride, size) << std::endl;
    return;
  }
  if (stride % 32 != 0) {
    std::cerr << Messages::warning_stride_not_aligned(stride) << std::endl;
  }
  
  size_t warmup_size = calculate_warmup_size(size);
  auto read_chunk_op = [stride](char* chunk_start, char* /* src_chunk */, size_t chunk_size, 
                                 std::atomic<uint64_t>* checksum) {
    // Use strided read for warmup
    uint64_t result = memory_read_strided_loop_asm(chunk_start, chunk_size, stride);
    if (checksum) {
      checksum->fetch_xor(result, std::memory_order_relaxed);
    }
  };
  warmup_parallel(buffer, size, num_threads, read_chunk_op, true, nullptr, &dummy_checksum, warmup_size);
}

// Warms up memory by writing to the buffer using strided access pattern.
// 'buffer': Memory region to write to.
// 'size': Total size of the buffer in bytes.
// 'stride': Stride size in bytes (must be >= 32, aligned, and <= size).
// 'num_threads': Number of concurrent threads to use.
void warmup_write_strided(void* buffer, size_t size, size_t stride, int num_threads) {
  // Validate stride
  if (stride < 32) {
    std::cerr << Messages::error_prefix() << Messages::error_stride_too_small() << std::endl;
    return;
  }
  if (stride > size) {
    std::cerr << Messages::error_prefix() << Messages::error_stride_too_large(stride, size) << std::endl;
    return;
  }
  if (stride % 32 != 0) {
    std::cerr << Messages::warning_stride_not_aligned(stride) << std::endl;
  }
  
  size_t warmup_size = calculate_warmup_size(size);
  auto write_chunk_op = [stride](char* chunk_start, char* /* src_chunk */, size_t chunk_size, 
                                  std::atomic<uint64_t>* /* checksum */) {
    // Use strided write for warmup
    memory_write_strided_loop_asm(chunk_start, chunk_size, stride);
  };
  warmup_parallel(buffer, size, num_threads, write_chunk_op, true, nullptr, nullptr, warmup_size);
}

// Warms up memory by copying data between buffers using strided access pattern.
// 'dst': Destination memory region.
// 'src': Source memory region.
// 'size': Total size of data to copy in bytes.
// 'stride': Stride size in bytes (must be >= 32, aligned, and <= size).
// 'num_threads': Number of concurrent threads to use.
void warmup_copy_strided(void* dst, void* src, size_t size, size_t stride, int num_threads) {
  // Validate stride
  if (stride < 32) {
    std::cerr << Messages::error_prefix() << Messages::error_stride_too_small() << std::endl;
    return;
  }
  if (stride > size) {
    std::cerr << Messages::error_prefix() << Messages::error_stride_too_large(stride, size) << std::endl;
    return;
  }
  if (stride % 32 != 0) {
    std::cerr << Messages::warning_stride_not_aligned(stride) << std::endl;
  }
  
  size_t warmup_size = calculate_warmup_size(size);
  auto copy_chunk_op = [stride](char* dst_chunk, char* src_chunk, size_t chunk_size, 
                                 std::atomic<uint64_t>* /* checksum */) {
    // Use strided copy for warmup
    memory_copy_strided_loop_asm(dst_chunk, src_chunk, chunk_size, stride);
  };
  warmup_parallel(dst, size, num_threads, copy_chunk_op, true, src, nullptr, warmup_size);
}

// Warms up memory by reading from the buffer using random access pattern.
// 'buffer': Memory region to read from.
// 'indices': Vector of byte offsets (must be 32-byte aligned and within buffer bounds).
// 'num_threads': Number of concurrent threads to use.
// 'dummy_checksum': Atomic variable to accumulate a dummy result (prevents optimization).
void warmup_read_random(void* buffer, const std::vector<size_t>& indices, int num_threads, std::atomic<uint64_t>& dummy_checksum) {
  if (indices.empty()) {
    std::cerr << Messages::error_prefix() << Messages::error_indices_empty() << std::endl;
    return;
  }
  
  // Validate indices (check first few as sample)
  size_t sample_size = (indices.size() < 100) ? indices.size() : 100;
  for (size_t i = 0; i < sample_size; ++i) {
    if (indices[i] % 32 != 0) {
      std::cerr << Messages::error_prefix() << Messages::error_index_not_aligned(i, indices[i]) << std::endl;
      return;
    }
  }
  
  // Use the indices directly - they should already be validated by the caller
  // For multi-threading, we'll split the indices across threads
  // If num_threads is 0 or indices is empty, return early (already checked empty above)
  if (num_threads <= 0) {
    return;
  }
  
  // For small index sets or single thread, just run directly
  if (num_threads == 1 || indices.size() <= static_cast<size_t>(num_threads)) {
    uint64_t result = memory_read_random_loop_asm(buffer, indices.data(), indices.size());
    dummy_checksum.fetch_xor(result, std::memory_order_relaxed);
    return;
  }
  
  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  
  size_t indices_per_thread = indices.size() / num_threads;
  size_t remainder = indices.size() % num_threads;
  
  size_t offset = 0;
  for (int t = 0; t < num_threads; ++t) {
    size_t thread_count = indices_per_thread + (t < static_cast<int>(remainder) ? 1 : 0);
    if (thread_count == 0 || offset >= indices.size()) continue;
    
    size_t start_idx = offset;
    size_t end_idx = offset + thread_count;
    if (end_idx > indices.size()) end_idx = indices.size();
    offset = end_idx;
    
    // Create a copy of the indices for this thread to avoid race conditions
    std::vector<size_t> thread_indices(indices.begin() + start_idx, indices.begin() + end_idx);
    
    threads.emplace_back([buffer, thread_indices, &dummy_checksum]() {
      kern_return_t qos_ret = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
      if (qos_ret != KERN_SUCCESS) {
        std::cerr << Messages::warning_qos_failed_worker_thread(qos_ret) << std::endl;
      }
      
      if (!thread_indices.empty()) {
        uint64_t result = memory_read_random_loop_asm(buffer, thread_indices.data(), thread_indices.size());
        dummy_checksum.fetch_xor(result, std::memory_order_relaxed);
      }
    });
  }
  
  join_threads(threads);
}

// Warms up memory by writing to the buffer using random access pattern.
// 'buffer': Memory region to write to.
// 'indices': Vector of byte offsets (must be 32-byte aligned and within buffer bounds).
// 'num_threads': Number of concurrent threads to use.
void warmup_write_random(void* buffer, const std::vector<size_t>& indices, int num_threads) {
  if (indices.empty()) {
    std::cerr << Messages::error_prefix() << Messages::error_indices_empty() << std::endl;
    return;
  }
  
  // Validate indices (check first few as sample)
  size_t sample_size = (indices.size() < 100) ? indices.size() : 100;
  for (size_t i = 0; i < sample_size; ++i) {
    if (indices[i] % 32 != 0) {
      std::cerr << Messages::error_prefix() << Messages::error_index_not_aligned(i, indices[i]) << std::endl;
      return;
    }
  }
  
  // Use the indices directly - they should already be validated by the caller
  // For multi-threading, we'll split the indices across threads
  if (num_threads <= 0) {
    return;
  }
  
  // For small index sets or single thread, just run directly
  if (num_threads == 1 || indices.size() <= static_cast<size_t>(num_threads)) {
    memory_write_random_loop_asm(buffer, indices.data(), indices.size());
    return;
  }
  
  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  
  size_t indices_per_thread = indices.size() / num_threads;
  size_t remainder = indices.size() % num_threads;
  
  size_t offset = 0;
  for (int t = 0; t < num_threads; ++t) {
    size_t thread_count = indices_per_thread + (t < static_cast<int>(remainder) ? 1 : 0);
    if (thread_count == 0 || offset >= indices.size()) continue;
    
    size_t start_idx = offset;
    size_t end_idx = offset + thread_count;
    if (end_idx > indices.size()) end_idx = indices.size();
    offset = end_idx;
    
    // Create a copy of the indices for this thread to avoid race conditions
    std::vector<size_t> thread_indices(indices.begin() + start_idx, indices.begin() + end_idx);
    
    threads.emplace_back([buffer, thread_indices]() {
      kern_return_t qos_ret = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
      if (qos_ret != KERN_SUCCESS) {
        std::cerr << Messages::warning_qos_failed_worker_thread(qos_ret) << std::endl;
      }
      
      if (!thread_indices.empty()) {
        memory_write_random_loop_asm(buffer, thread_indices.data(), thread_indices.size());
      }
    });
  }
  
  join_threads(threads);
}

// Warms up memory by copying data between buffers using random access pattern.
// 'dst': Destination memory region.
// 'src': Source memory region.
// 'indices': Vector of byte offsets (must be 32-byte aligned and within buffer bounds).
// 'num_threads': Number of concurrent threads to use.
void warmup_copy_random(void* dst, void* src, const std::vector<size_t>& indices, int num_threads) {
  if (indices.empty()) {
    std::cerr << Messages::error_prefix() << Messages::error_indices_empty() << std::endl;
    return;
  }
  
  // Validate indices (check first few as sample)
  size_t sample_size = (indices.size() < 100) ? indices.size() : 100;
  for (size_t i = 0; i < sample_size; ++i) {
    if (indices[i] % 32 != 0) {
      std::cerr << Messages::error_prefix() << Messages::error_index_not_aligned(i, indices[i]) << std::endl;
      return;
    }
  }
  
  // Use the indices directly - they should already be validated by the caller
  // For multi-threading, we'll split the indices across threads
  if (num_threads <= 0) {
    return;
  }
  
  // For small index sets or single thread, just run directly
  if (num_threads == 1 || indices.size() <= static_cast<size_t>(num_threads)) {
    memory_copy_random_loop_asm(dst, src, indices.data(), indices.size());
    return;
  }
  
  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  
  size_t indices_per_thread = indices.size() / num_threads;
  size_t remainder = indices.size() % num_threads;
  
  size_t offset = 0;
  for (int t = 0; t < num_threads; ++t) {
    size_t thread_count = indices_per_thread + (t < static_cast<int>(remainder) ? 1 : 0);
    if (thread_count == 0 || offset >= indices.size()) continue;
    
    size_t start_idx = offset;
    size_t end_idx = offset + thread_count;
    if (end_idx > indices.size()) end_idx = indices.size();
    offset = end_idx;
    
    // Create a copy of the indices for this thread to avoid race conditions
    std::vector<size_t> thread_indices(indices.begin() + start_idx, indices.begin() + end_idx);
    
    threads.emplace_back([dst, src, thread_indices]() {
      kern_return_t qos_ret = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
      if (qos_ret != KERN_SUCCESS) {
        std::cerr << Messages::warning_qos_failed_worker_thread(qos_ret) << std::endl;
      }
      
      if (!thread_indices.empty()) {
        memory_copy_random_loop_asm(dst, src, thread_indices.data(), thread_indices.size());
      }
    });
  }
  
  join_threads(threads);
}
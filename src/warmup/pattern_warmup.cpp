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
#include <iostream>  // For std::cerr
#include <thread>    // For std::thread
#include <vector>    // For std::vector

// macOS specific QoS
#include <mach/mach.h>    // For kern_return_t
#include <pthread/qos.h>  // For pthread_set_qos_class_self_np

#include "utils/benchmark.h"  // Includes definitions for assembly loops etc.
#include "output/console/messages.h"   // For error and warning messages
#include "warmup/warmup.h"
#include "warmup/warmup_internal.h"

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
    size_t num_iterations = (chunk_size + stride - 1) / stride;
    uint64_t result = memory_read_strided_loop_asm(chunk_start, chunk_size, stride, num_iterations);
    if (checksum) {
      checksum->fetch_xor(result, std::memory_order_release);
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
    size_t num_iterations = (chunk_size + stride - 1) / stride;
    memory_write_strided_loop_asm(chunk_start, chunk_size, stride, num_iterations);
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
    size_t num_iterations = (chunk_size + stride - 1) / stride;
    memory_copy_strided_loop_asm(dst_chunk, src_chunk, chunk_size, stride, num_iterations);
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
    dummy_checksum.fetch_xor(result, std::memory_order_release);
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
        dummy_checksum.fetch_xor(result, std::memory_order_release);
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


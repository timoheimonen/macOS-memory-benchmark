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
#include <cstdio>    // For fprintf
#include <iostream>  // For std::cout
#include <thread>    // For std::thread
#include <vector>    // For std::vector
#include <unistd.h>  // For getpagesize()

// macOS specific QoS
#include <mach/mach.h>    // For kern_return_t
#include <pthread/qos.h>  // For pthread_set_qos_class_self_np

#include "benchmark.h"  // Includes definitions for assembly loops etc.

// Generic parallel warmup function for multi-threaded operations.
// 'buffer': Primary memory region (or destination for copy operations).
// 'size': Total size of the buffer in bytes.
// 'num_threads': Number of concurrent threads to use.
// 'chunk_operation': Function to execute on each chunk (takes chunk_start and chunk_size).
// 'set_qos': Whether to set QoS class for worker threads (default: false).
// 'src_buffer': Optional source buffer for copy operations (default: nullptr).
// 'dummy_checksum': Optional atomic variable to accumulate checksum results (default: nullptr).
template<typename ChunkOp>
void warmup_parallel(void* buffer, size_t size, int num_threads, ChunkOp chunk_operation, 
                     bool set_qos = false, void* src_buffer = nullptr, 
                     std::atomic<uint64_t>* dummy_checksum = nullptr) {
  // Vector to store thread objects.
  std::vector<std::thread> threads;
  // Pre-allocate space for efficiency.
  threads.reserve(num_threads);
  // Tracks the current position in the buffer.
  size_t offset = 0;
  // Calculate the base size of the memory chunk per thread.
  size_t chunk_base_size = size / num_threads;
  // Calculate the remainder to distribute among threads.
  size_t chunk_remainder = size % num_threads;

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
          fprintf(stderr, "Warning: Failed to set QoS class for warmup worker thread (code: %d)\n", qos_ret);
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
    fprintf(stderr, "Warning: Failed to set QoS class for single-threaded warmup (code: %d)\n", qos_ret);
  }
  operation();
}

// Warms up memory by reading from the buffer using multiple threads.
// 'buffer': Memory region to read from.
// 'size': Total size of the buffer in bytes.
// 'num_threads': Number of concurrent threads to use.
// 'dummy_checksum': Atomic variable to accumulate a dummy result (prevents optimization).
void warmup_read(void* buffer, size_t size, int num_threads, std::atomic<uint64_t>& dummy_checksum) {
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
  warmup_parallel(buffer, size, num_threads, read_chunk_op, true, nullptr, &dummy_checksum);
}

// Warms up memory by writing to the buffer using multiple threads.
// 'buffer': Memory region to write to.
// 'size': Total size of the buffer in bytes.
// 'num_threads': Number of concurrent threads to use.
void warmup_write(void* buffer, size_t size, int num_threads) {
  auto write_chunk_op = [](char* chunk_start, char* /* src_chunk */, size_t chunk_size, 
                           std::atomic<uint64_t>* /* checksum */) {
    memory_write_loop_asm(chunk_start, chunk_size);
  };
  warmup_parallel(buffer, size, num_threads, write_chunk_op, true);
}

// Warms up memory by copying data between buffers using multiple threads.
// 'dst': Destination memory region.
// 'src': Source memory region.
// 'size': Total size of data to copy in bytes.
// 'num_threads': Number of concurrent threads to use.
void warmup_copy(void* dst, void* src, size_t size, int num_threads) {
  auto copy_chunk_op = [](char* dst_chunk, char* src_chunk, size_t chunk_size, 
                          std::atomic<uint64_t>* /* checksum */) {
    memory_copy_loop_asm(dst_chunk, src_chunk, chunk_size);
  };
  warmup_parallel(dst, size, num_threads, copy_chunk_op, true, src);
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
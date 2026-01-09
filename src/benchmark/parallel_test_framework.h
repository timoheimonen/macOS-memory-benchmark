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
/**
 * @file parallel_test_framework.h
 * @brief Generic parallel test framework for multi-threaded benchmark execution
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2025
 *
 * This header provides template-based functions for executing parallel memory
 * benchmarks across multiple threads with proper synchronization and cache-line
 * alignment. The framework handles thread creation, work distribution, Quality of
 * Service (QoS) settings, and timing measurements.
 *
 * Key features:
 * - Generic template interface supporting custom work functions
 * - Automatic cache-line alignment to prevent false sharing
 * - Synchronized start gate for accurate timing
 * - Thread-safe execution with minimal overhead
 * - macOS QoS integration for performance cores
 * - Support for both single-buffer and dual-buffer (copy) operations
 *
 * The framework provides two main template functions:
 * - run_parallel_test(): For read/write operations on a single buffer
 * - run_parallel_test_copy(): For copy operations requiring source and destination buffers
 */
#ifndef PARALLEL_TEST_FRAMEWORK_H
#define PARALLEL_TEST_FRAMEWORK_H

#include <atomic>                // For std::atomic
#include <condition_variable>    // For start gate sync
#include <cstdio>                // For fprintf, stderr
#include <functional>            // For std::function
#include <iostream>              // For std::cout
#include <mutex>                 // For std::mutex
#include <thread>                // For std::thread
#include <vector>                // For std::vector

// macOS specific includes
#include <mach/mach.h>          // For kern_return_t
#include <pthread/qos.h>        // For pthread_set_qos_class_self_np

#include "utils/benchmark.h"  // Include benchmark definitions (assembly funcs, HighResTimer)
#include "core/memory/memory_utils.h"  // For align_ptr_to_cache_line

// --- Generic Parallel Test Framework ---

/**
 * @brief Run a parallel test with automatic work distribution across threads
 * @tparam WorkFunction Function type for per-thread work (void(void* chunk_start, size_t chunk_size, int iterations))
 * @param buffer Pointer to the memory buffer to operate on
 * @param size Total size of the buffer in bytes
 * @param iterations Number of times to perform the operation
 * @param num_threads Number of threads to use for parallel execution
 * @param timer Reference to high-resolution timer for measuring execution time
 * @param work_function Function to call for each thread's chunk. Signature: void(void* chunk_start, size_t chunk_size, int iterations)
 * @param thread_name Name of the thread type for QoS error messages (e.g., "read", "write", "copy")
 * @return Total duration in seconds, or 0.0 if no work was performed
 *
 * This function distributes the buffer across multiple threads with cache-line alignment
 * to prevent false sharing. All threads are synchronized to start simultaneously after
 * the timer begins, ensuring accurate timing measurements. Each thread is configured
 * with QOS_CLASS_USER_INTERACTIVE for optimal performance on macOS.
 *
 * @note The buffer is divided evenly among threads, with each chunk aligned to cache-line
 *       boundaries. Remainder bytes are distributed across the first N threads.
 * @warning Returns 0.0 if size is 0, num_threads <= 0, or no threads could be created
 */
template<typename WorkFunction>
double run_parallel_test(void *buffer, size_t size, int iterations, int num_threads, HighResTimer &timer,
                         WorkFunction work_function, const char *thread_name) {
  std::vector<std::thread> threads;
  threads.reserve(num_threads);  // Pre-allocate vector space for threads.

  std::mutex start_mutex;
  std::condition_variable start_cv;
  bool start_flag = false;  // Gate so timing starts after threads are ready

  // Divide work among threads (chunks calculated once).
  size_t offset = 0;
  size_t chunk_base_size = size / num_threads;
  size_t chunk_remainder = size % num_threads;

  // Early validation: return 0.0 if no work to do or invalid thread count.
  // This avoids timer overhead when no meaningful work can be performed.
  if (size == 0 || num_threads <= 0) {
    return 0.0;  // No work to do or invalid thread count
  }

  // Launch threads once; each handles its chunk for all iterations.
  for (int t = 0; t < num_threads; ++t) {
    // Calculate the exact size for this thread's chunk, adding remainder if needed.
    size_t original_chunk_size = chunk_base_size + (t < chunk_remainder ? 1 : 0);
    if (original_chunk_size == 0) continue;  // Avoid launching threads for zero work.

    // Calculate the original end position for this chunk
    size_t original_chunk_end = offset + original_chunk_size;
    
    // Calculate the start address for this thread's chunk.
    char *unaligned_start = static_cast<char *>(buffer) + offset;
    // Align chunk_start to cache line boundary to prevent false sharing
    char *chunk_start = static_cast<char *>(align_ptr_to_cache_line(unaligned_start));
    
    // Calculate chunk size to cover the original range [offset, offset + original_chunk_size)
    // This ensures no gaps: chunk covers from aligned start to original end
    char *original_end_ptr = static_cast<char *>(buffer) + original_chunk_end;
    
    // Check if alignment moved us past the original range
    // If so, fallback to unaligned start to ensure coverage
    if (chunk_start >= original_end_ptr) {
      chunk_start = unaligned_start;
    }
    
    size_t current_chunk_size = original_end_ptr - chunk_start;
    
    // Ensure we don't exceed the buffer size
    char *buffer_end = static_cast<char *>(buffer) + size;
    if (chunk_start + current_chunk_size > buffer_end) {
      current_chunk_size = buffer_end - chunk_start;
    }
    
    // Skip if chunk size is zero or negative after alignment adjustments
    if (current_chunk_size == 0 || chunk_start >= buffer_end) {
      // Still update offset to prevent infinite loops, but skip thread creation
      offset = original_chunk_end;
      continue;
    }
    
    // Capture aligned chunk_start and adjusted chunk_size for the thread
    char *thread_chunk_start = chunk_start;
    size_t thread_chunk_size = current_chunk_size;
    
    // iterations loop inside thread lambda for re-use (avoids per-iteration creation).
    threads.emplace_back([thread_chunk_start, thread_chunk_size, iterations, &start_mutex, &start_cv,
                          &start_flag, work_function, thread_name]() {
      // Set QoS for this worker thread
      kern_return_t qos_ret = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
      if (qos_ret != KERN_SUCCESS) {
        fprintf(stderr, "Warning: Failed to set QoS class for %s worker thread (code: %d)\n", thread_name, qos_ret);
      }

      // Wait for the main thread to start the timer before beginning work.
      {
        std::unique_lock<std::mutex> lk(start_mutex);
        start_cv.wait(lk, [&start_flag] { return start_flag; });
      }

      // Execute the work function for this chunk with aligned pointer
      work_function(thread_chunk_start, thread_chunk_size, iterations);
    });
    
    // Move the offset for the next thread's chunk using the actual end position
    offset = original_chunk_end;
  }

  // Check if any threads were created. If all chunks were zero after alignment,
  // no threads were created and we should return 0.0 to avoid misleading timer measurements.
  if (threads.empty()) {
    return 0.0;  // No threads created (all chunks were zero after alignment)
  }

  {
    std::unique_lock<std::mutex> lk(start_mutex);
    timer.start();   // Start timing after all threads are created and waiting.
    start_flag = true;
  }
  start_cv.notify_all();

  join_threads(threads);           // Wait for all threads to finish (joined once after all iterations).
  double duration = timer.stop();  // Stop timing after all work.
  return duration;  // Return total time elapsed.
}

/**
 * @brief Run a parallel copy test with automatic work distribution across threads
 * @tparam WorkFunction Function type for per-thread copy work (void(void* dst_chunk, void* src_chunk, size_t chunk_size, int iterations))
 * @param dst Pointer to the destination buffer
 * @param src Pointer to the source buffer
 * @param size Total size of the buffers in bytes
 * @param iterations Number of times to perform the copy operation
 * @param num_threads Number of threads to use for parallel execution
 * @param timer Reference to high-resolution timer for measuring execution time
 * @param work_function Function to call for each thread's chunk. Signature: void(void* dst_chunk, void* src_chunk, size_t chunk_size, int iterations)
 * @param thread_name Name of the thread type for QoS error messages (e.g., "copy")
 * @return Total duration in seconds, or 0.0 if no work was performed
 *
 * This function distributes copy operations across multiple threads with cache-line alignment
 * on the destination buffer to prevent false sharing. The source buffer alignment is adjusted
 * to maintain data correspondence with the destination buffer (ensuring src[i] maps to dst[i]).
 * All threads are synchronized to start simultaneously after the timer begins.
 *
 * @note Both buffers are divided evenly among threads. Destination chunks are aligned to
 *       cache-line boundaries, and source chunks maintain the same alignment offset to
 *       preserve data integrity.
 * @warning Returns 0.0 if size is 0, num_threads <= 0, or no threads could be created
 */
template<typename WorkFunction>
double run_parallel_test_copy(void *dst, void *src, size_t size, int iterations, int num_threads, HighResTimer &timer,
                               WorkFunction work_function, const char *thread_name) {
  std::vector<std::thread> threads;
  threads.reserve(num_threads);  // Pre-allocate vector space for threads.

  std::mutex start_mutex;
  std::condition_variable start_cv;
  bool start_flag = false;  // Gate so timing starts after threads are ready

  // Divide work among threads (chunks calculated once).
  size_t offset = 0;
  size_t chunk_base_size = size / num_threads;
  size_t chunk_remainder = size % num_threads;

  // Early validation: return 0.0 if no work to do or invalid thread count.
  // This avoids timer overhead when no meaningful work can be performed.
  if (size == 0 || num_threads <= 0) {
    return 0.0;  // No work to do or invalid thread count
  }

  // Launch threads once; each handles its chunk for all iterations.
  for (int t = 0; t < num_threads; ++t) {
    // Calculate the exact size for this thread's chunk, adding remainder if needed.
    size_t original_chunk_size = chunk_base_size + (t < chunk_remainder ? 1 : 0);
    if (original_chunk_size == 0) continue;  // Avoid launching threads for zero work.

    // Calculate the original end position for this chunk
    size_t original_chunk_end = offset + original_chunk_size;
    
    // Calculate unaligned start positions
    char *dst_unaligned_start = static_cast<char *>(dst) + offset;
    char *src_unaligned_start = static_cast<char *>(src) + offset;
    
    // Align dst to cache line boundary to prevent false sharing
    char *dst_chunk = static_cast<char *>(align_ptr_to_cache_line(dst_unaligned_start));
    
    // Calculate chunk size to cover the original range [offset, offset + original_chunk_size)
    // This ensures no gaps: chunk covers from aligned start to original end
    char *dst_original_end = static_cast<char *>(dst) + original_chunk_end;
    
    // Check if alignment moved us past the original range
    // If so, fallback to unaligned start to ensure coverage
    if (dst_chunk >= dst_original_end) {
      dst_chunk = dst_unaligned_start;
    }
    
    // Maintain src/dst correspondence by applying the same alignment offset
    // This ensures data integrity: src[i] always maps to dst[i]
    size_t alignment_offset = dst_chunk - dst_unaligned_start;
    char *src_chunk = src_unaligned_start + alignment_offset;
    
    size_t current_chunk_size = dst_original_end - dst_chunk;
    
    // Ensure we don't exceed the buffer size
    char *dst_end = static_cast<char *>(dst) + size;
    if (dst_chunk + current_chunk_size > dst_end) {
      current_chunk_size = dst_end - dst_chunk;
    }
    
    // Skip if chunk size is zero or negative after alignment adjustments
    if (current_chunk_size == 0 || dst_chunk >= dst_end) {
      // Still update offset to prevent infinite loops, but skip thread creation
      offset = original_chunk_end;
      continue;
    }
    
    // Capture aligned chunk pointers and adjusted chunk_size for the thread
    char *thread_dst_chunk = dst_chunk;
    char *thread_src_chunk = src_chunk;
    size_t thread_chunk_size = current_chunk_size;
    
    // iterations loop inside thread lambda for re-use (avoids per-iteration creation).
    threads.emplace_back([thread_dst_chunk, thread_src_chunk, thread_chunk_size, iterations, &start_mutex, &start_cv,
                          &start_flag, work_function, thread_name]() {
      // Set QoS for this worker thread
      kern_return_t qos_ret = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
      if (qos_ret != KERN_SUCCESS) {
        fprintf(stderr, "Warning: Failed to set QoS class for %s worker thread (code: %d)\n", thread_name, qos_ret);
      }

      // Wait for the main thread to start the timer before beginning work.
      {
        std::unique_lock<std::mutex> lk(start_mutex);
        start_cv.wait(lk, [&start_flag] { return start_flag; });
      }

      // Execute the work function for this chunk with aligned pointers
      work_function(thread_dst_chunk, thread_src_chunk, thread_chunk_size, iterations);
    });
    
    // Move the offset for the next thread's chunk using the actual end position
    offset = original_chunk_end;
  }

  // Check if any threads were created. If all chunks were zero after alignment,
  // no threads were created and we should return 0.0 to avoid misleading timer measurements.
  if (threads.empty()) {
    return 0.0;  // No threads created (all chunks were zero after alignment)
  }

  {
    std::unique_lock<std::mutex> lk(start_mutex);
    timer.start();   // Start timing after all threads are created and waiting.
    start_flag = true;
  }
  start_cv.notify_all();

  join_threads(threads);           // Wait for all threads to finish (joined once after all iterations).
  double duration = timer.stop();  // Stop timing after all work.
  return duration;  // Return total time elapsed.
}

#endif // PARALLEL_TEST_FRAMEWORK_H


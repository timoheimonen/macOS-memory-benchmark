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
 * @brief Build contiguous per-thread ranges with aligned internal boundaries.
 *
 * Produces `num_threads` contiguous ranges that cover [0, size) exactly once.
 * Internal boundaries are aligned to cache-line boundaries (rounded up) using
 * `alignment_base` as the pointer reference. This keeps dst chunk starts aligned
 * for most threads while preserving full byte coverage (no gaps/overlaps).
 *
 * @param alignment_base Base pointer used for boundary alignment calculations
 * @param size Total size of the covered range in bytes
 * @param num_threads Number of ranges to produce (must be > 0)
 * @return Vector of boundary offsets of size num_threads + 1
 */
inline std::vector<size_t> build_aligned_chunk_boundaries(void* alignment_base, size_t size, int num_threads) {
  const size_t thread_count = static_cast<size_t>(num_threads);
  std::vector<size_t> boundaries(thread_count + 1, 0);

  // Start from an even byte partition.
  size_t offset = 0;
  const size_t chunk_base_size = size / thread_count;
  const size_t chunk_remainder = size % thread_count;
  for (size_t t = 0; t < thread_count; ++t) {
    const size_t chunk_size = chunk_base_size + (t < chunk_remainder ? 1 : 0);
    offset += chunk_size;
    boundaries[t + 1] = offset;
  }

  // Align internal boundaries while preserving monotonic ordering.
  char* base_ptr = static_cast<char*>(alignment_base);
  for (size_t i = 1; i < thread_count; ++i) {
    char* boundary_ptr = base_ptr + boundaries[i];
    char* aligned_boundary_ptr = static_cast<char*>(align_ptr_to_cache_line(boundary_ptr));
    size_t aligned_offset = static_cast<size_t>(aligned_boundary_ptr - base_ptr);

    if (aligned_offset > size) {
      aligned_offset = size;
    }
    if (aligned_offset < boundaries[i - 1]) {
      aligned_offset = boundaries[i - 1];
    }
    boundaries[i] = aligned_offset;
  }

  boundaries[0] = 0;
  boundaries[thread_count] = size;

  // Final monotonicity guard after clamping.
  for (size_t i = 1; i < boundaries.size(); ++i) {
    if (boundaries[i] < boundaries[i - 1]) {
      boundaries[i] = boundaries[i - 1];
    }
  }

  return boundaries;
}

/**
 * @brief Shared parallel test runner for single-buffer and copy variants.
 * @tparam MakeThreadFunction Function type creating a worker callable for std::thread
 * @param alignment_base Base pointer used for chunk-boundary alignment
 * @param size Total size of the covered range in bytes
 * @param iterations Number of operation iterations executed by each worker
 * @param num_threads Number of worker threads to launch
 * @param timer Reference to high-resolution timer for measuring execution time
 * @param make_thread Function that builds a worker callable for one chunk
 * @return Total duration in seconds, or 0.0 if no work was performed
 */
template<typename MakeThreadFunction>
double run_parallel_test_common(void* alignment_base, size_t size, int iterations, int num_threads,
                                HighResTimer& timer, MakeThreadFunction make_thread) {
  // Early validation: return 0.0 if no work to do or invalid thread count.
  // This avoids unnecessary setup and prevents division-by-zero in partitioning.
  if (size == 0 || num_threads <= 0) {
    return 0.0;
  }

  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(num_threads));  // Pre-allocate vector space for threads.

  std::mutex start_mutex;
  std::condition_variable start_cv;
  bool start_flag = false;  // Gate so timing starts after threads are ready

  // Build contiguous chunk boundaries with aligned internal split points.
  std::vector<size_t> boundaries = build_aligned_chunk_boundaries(alignment_base, size, num_threads);

  // Launch threads once; each handles its chunk for all iterations.
  for (size_t t = 0; t < static_cast<size_t>(num_threads); ++t) {
    size_t chunk_start_offset = boundaries[t];
    size_t chunk_end_offset = boundaries[t + 1];
    if (chunk_end_offset <= chunk_start_offset) {
      continue;
    }

    size_t thread_chunk_size = chunk_end_offset - chunk_start_offset;
    threads.emplace_back(make_thread(chunk_start_offset, thread_chunk_size, iterations, start_mutex, start_cv,
                                     start_flag));
  }

  // Check if any threads were created. If all chunks were zero after alignment,
  // no threads were created and we should return 0.0 to avoid misleading timer measurements.
  if (threads.empty()) {
    return 0.0;  // No threads created (all chunks were zero after alignment)
  }

  {
    std::unique_lock<std::mutex> lk(start_mutex);
    timer.start();  // Start timing after all threads are created and waiting.
    start_flag = true;
  }
  start_cv.notify_all();

  join_threads(threads);           // Wait for all threads to finish (joined once after all iterations).
  double duration = timer.stop();  // Stop timing after all work.
  return duration;                 // Return total time elapsed.
}

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
  char* buffer_start = static_cast<char*>(buffer);
  auto make_thread = [buffer_start, work_function, thread_name](size_t chunk_start_offset, size_t thread_chunk_size,
                                                                 int iterations_local, std::mutex& start_mutex,
                                                                 std::condition_variable& start_cv,
                                                                 bool& start_flag) {
    char* thread_chunk_start = buffer_start + chunk_start_offset;
    return [thread_chunk_start, thread_chunk_size, iterations_local, &start_mutex, &start_cv, &start_flag,
            work_function, thread_name]() {
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
      work_function(thread_chunk_start, thread_chunk_size, iterations_local);
    };
  };

  return run_parallel_test_common(buffer, size, iterations, num_threads, timer, make_thread);
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
  char* dst_start = static_cast<char*>(dst);
  char* src_start = static_cast<char*>(src);
  auto make_thread = [dst_start, src_start, work_function,
                      thread_name](size_t chunk_start_offset, size_t thread_chunk_size, int iterations_local,
                                   std::mutex& start_mutex, std::condition_variable& start_cv, bool& start_flag) {
    char* thread_dst_chunk = dst_start + chunk_start_offset;
    char* thread_src_chunk = src_start + chunk_start_offset;
    return [thread_dst_chunk, thread_src_chunk, thread_chunk_size, iterations_local, &start_mutex, &start_cv,
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
      work_function(thread_dst_chunk, thread_src_chunk, thread_chunk_size, iterations_local);
    };
  };

  return run_parallel_test_common(dst, size, iterations, num_threads, timer, make_thread);
}

#endif // PARALLEL_TEST_FRAMEWORK_H

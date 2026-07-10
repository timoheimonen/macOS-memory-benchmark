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
#include <limits>                // For std::numeric_limits
#include <mutex>                 // For std::mutex
#include <thread>                // For std::thread
#include <string>                // For std::string
#include <system_error>          // For std::system_error
#include <utility>               // For std::move
#include <vector>                // For std::vector

// macOS specific includes
#include <mach/mach.h>          // For kern_return_t
#include <pthread/qos.h>        // For pthread_set_qos_class_self_np

#include "utils/benchmark.h"  // Include benchmark definitions (assembly funcs, HighResTimer)
#include "core/memory/memory_utils.h"  // For align_ptr_to_cache_line
#include "output/console/messages/messages_api.h"

// --- Generic Parallel Test Framework ---

struct ParallelExecutionMetadata {
  int requested_workers = 0;
  int created_workers = 0;
  size_t qos_successful_workers = 0;
  size_t qos_failed_workers = 0;
  bool worker_startup_failed = false;
};

/** @brief Deterministic test-only fault seam for worker creation handling. */
struct ParallelExecutionTestControl {
  int fail_before_worker_index = -1;
};

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
 * @tparam MakeWorkFunction Function type creating the measured work callable for one worker
 * @param alignment_base Base pointer used for chunk-boundary alignment
 * @param size Total size of the covered range in bytes
 * @param iterations Number of operation iterations executed by each worker
 * @param num_threads Number of worker threads to launch
 * @param timer Reference to high-resolution timer for measuring execution time
 * @param thread_name Name used for worker QoS diagnostics
 * @param make_work Function that builds measured work for one chunk
 * @return Total duration in seconds, or 0.0 if no work was performed
 */
template <typename MakeWorkFunction>
double run_parallel_test_common(void* alignment_base, size_t size, int iterations, int num_threads, HighResTimer& timer,
                                const char* thread_name, MakeWorkFunction make_work,
                                const std::vector<size_t>* planned_boundaries = nullptr,
                                ParallelExecutionMetadata* execution_metadata = nullptr,
                                const ParallelExecutionTestControl* test_control = nullptr) {
  if (execution_metadata != nullptr) {
    *execution_metadata = {};
    execution_metadata->requested_workers = num_threads;
  }
  // Early validation: return 0.0 if no work to do or invalid thread count.
  // This avoids unnecessary setup and prevents division-by-zero in partitioning.
  if (size == 0 || num_threads <= 0) {
    return 0.0;
  }

  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(num_threads));  // Pre-allocate vector space for threads.

  std::mutex state_mutex;
  std::condition_variable state_cv;
  size_t ready_workers = 0;
  bool start_flag = false;
  bool measurement_complete = false;
  double measured_duration = 0.0;
  std::atomic<size_t> remaining_workers{0};
  std::atomic<size_t> qos_successful_workers{0};
  std::atomic<size_t> qos_failed_workers{0};

  // Use a finalized external plan when supplied; otherwise build the normal
  // contiguous chunk boundaries before worker creation and timing.
  std::vector<size_t> boundaries = planned_boundaries != nullptr
                                       ? *planned_boundaries
                                       : build_aligned_chunk_boundaries(alignment_base, size, num_threads);
  if (boundaries.size() != static_cast<size_t>(num_threads) + 1 || boundaries.front() != 0 ||
      boundaries.back() != size) {
    return 0.0;
  }
  for (size_t index = 1; index < boundaries.size(); ++index) {
    if (boundaries[index] < boundaries[index - 1] ||
        (planned_boundaries != nullptr && boundaries[index] == boundaries[index - 1])) {
      return 0.0;
    }
  }

  bool worker_startup_failed = false;
  // Launch threads once; each handles its chunk for all iterations.
  try {
    for (size_t t = 0; t < static_cast<size_t>(num_threads); ++t) {
      size_t chunk_start_offset = boundaries[t];
      size_t chunk_end_offset = boundaries[t + 1];
      if (chunk_end_offset <= chunk_start_offset) {
        continue;
      }

      size_t thread_chunk_size = chunk_end_offset - chunk_start_offset;
      const size_t worker_index = threads.size();
      auto measured_work =
          make_work(chunk_start_offset, thread_chunk_size, iterations, worker_index);
      if (test_control != nullptr && test_control->fail_before_worker_index >= 0 &&
          worker_index ==
              static_cast<size_t>(test_control->fail_before_worker_index)) {
        throw std::system_error(
            std::make_error_code(std::errc::resource_unavailable_try_again));
      }
      threads.emplace_back([measured_work = std::move(measured_work), &state_mutex, &state_cv,
                            &ready_workers, &start_flag, &measurement_complete,
                            &measured_duration, &remaining_workers,
                            &qos_successful_workers, &qos_failed_workers,
                            &timer, thread_name]() mutable {
        // QoS setup is preparation and must complete before the timed start gate.
        kern_return_t qos_ret =
            pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
        if (qos_ret != KERN_SUCCESS) {
          qos_failed_workers.fetch_add(1, std::memory_order_relaxed);
          std::cerr << Messages::warning_prefix()
                    << Messages::warning_qos_failed_benchmark_worker(thread_name,
                                                                     qos_ret)
                    << std::endl;
        } else {
          qos_successful_workers.fetch_add(1, std::memory_order_relaxed);
        }

        {
          std::unique_lock<std::mutex> lock(state_mutex);
          ++ready_workers;
          state_cv.notify_all();
          state_cv.wait(lock, [&start_flag] { return start_flag; });
        }

        measured_work();

        // Complete this worker's memory effects before publishing completion.
        asm volatile("dsb ish" ::: "memory");
        if (remaining_workers.fetch_sub(1, std::memory_order_acq_rel) == 1) {
          const double duration = timer.stop();
          {
            std::lock_guard<std::mutex> lock(state_mutex);
            measured_duration = duration;
            measurement_complete = true;
          }
          state_cv.notify_one();
        }
      });
    }
  } catch (const std::system_error&) {
    worker_startup_failed = true;
  }

  // Check if any threads were created. If all chunks were zero after alignment,
  // no threads were created and we should return 0.0 to avoid misleading timer measurements.
  if (threads.empty()) {
    if (execution_metadata != nullptr) {
      execution_metadata->worker_startup_failed = worker_startup_failed;
    }
    return 0.0;  // No threads created (all chunks were zero after alignment)
  }

  remaining_workers.store(threads.size(), std::memory_order_relaxed);
  {
    std::unique_lock<std::mutex> lock(state_mutex);
    state_cv.wait(lock, [&ready_workers, &threads] { return ready_workers == threads.size(); });
    timer.start();
    start_flag = true;
  }
  state_cv.notify_all();

  {
    std::unique_lock<std::mutex> lock(state_mutex);
    state_cv.wait(lock, [&measurement_complete] { return measurement_complete; });
  }

  const int created_workers = static_cast<int>(threads.size());
  // Thread exit and join are deliberately outside the measured interval.
  join_threads(threads);
  if (execution_metadata != nullptr) {
    execution_metadata->created_workers = created_workers;
    execution_metadata->qos_successful_workers =
        qos_successful_workers.load(std::memory_order_relaxed);
    execution_metadata->qos_failed_workers =
        qos_failed_workers.load(std::memory_order_relaxed);
    execution_metadata->worker_startup_failed = worker_startup_failed;
  }
  return worker_startup_failed ? 0.0 : measured_duration;
}

/**
 * @brief Run indexed single-buffer work with finalized worker boundaries.
 *
 * This is used by pattern planners whose exact work accounting depends on the
 * executor consuming the same worker ranges without repartitioning.
 */
template <typename WorkFunction>
double run_parallel_test_indexed_with_boundaries(void* buffer, size_t size, int iterations, HighResTimer& timer,
                                                 const std::vector<size_t>& boundaries, WorkFunction work_function,
                                                 const char* thread_name,
                                                 ParallelExecutionMetadata* execution_metadata = nullptr) {
  if (boundaries.size() < 2 || boundaries.size() - 1 > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return 0.0;
  }
  char* buffer_start = static_cast<char*>(buffer);
  auto make_work = [buffer_start, work_function](size_t chunk_start_offset, size_t thread_chunk_size,
                                                 int iterations_local, size_t worker_index) {
    char* thread_chunk_start = buffer_start + chunk_start_offset;
    return [thread_chunk_start, thread_chunk_size, iterations_local, worker_index, work_function]() {
      work_function(thread_chunk_start, thread_chunk_size, iterations_local, worker_index);
    };
  };
  return run_parallel_test_common(buffer, size, iterations, static_cast<int>(boundaries.size() - 1), timer, thread_name,
                                  make_work, &boundaries, execution_metadata);
}

/** @brief Run indexed copy work with finalized worker boundaries. */
template <typename WorkFunction>
double run_parallel_test_copy_indexed_with_boundaries(void* dst, void* src, size_t size, int iterations,
                                                      HighResTimer& timer, const std::vector<size_t>& boundaries,
                                                      WorkFunction work_function, const char* thread_name,
                                                      ParallelExecutionMetadata* execution_metadata = nullptr) {
  if (boundaries.size() < 2 || boundaries.size() - 1 > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return 0.0;
  }
  char* dst_start = static_cast<char*>(dst);
  char* src_start = static_cast<char*>(src);
  auto make_work = [dst_start, src_start, work_function](size_t chunk_start_offset, size_t thread_chunk_size,
                                                         int iterations_local, size_t worker_index) {
    char* thread_dst_chunk = dst_start + chunk_start_offset;
    char* thread_src_chunk = src_start + chunk_start_offset;
    return [thread_dst_chunk, thread_src_chunk, thread_chunk_size, iterations_local, worker_index, work_function]() {
      work_function(thread_dst_chunk, thread_src_chunk, thread_chunk_size, iterations_local, worker_index);
    };
  };
  return run_parallel_test_common(dst, size, iterations, static_cast<int>(boundaries.size() - 1), timer, thread_name,
                                  make_work, &boundaries, execution_metadata);
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
  auto make_work = [buffer_start, work_function](size_t chunk_start_offset, size_t thread_chunk_size,
                                                  int iterations_local,
                                                  size_t /* worker_index */) {
    char* thread_chunk_start = buffer_start + chunk_start_offset;
    return [thread_chunk_start, thread_chunk_size, iterations_local, work_function]() {
      work_function(thread_chunk_start, thread_chunk_size, iterations_local);
    };
  };

  return run_parallel_test_common(buffer, size, iterations, num_threads, timer, thread_name, make_work);
}

/**
 * @brief Run single-buffer work that needs a stable per-worker output slot.
 *
 * The indexed callback can publish a worker-local checksum without a contended
 * atomic. Callers combine those slots after this function returns, outside the
 * measured interval.
 */
template<typename WorkFunction>
double run_parallel_test_indexed(void* buffer, size_t size, int iterations,
                                 int num_threads, HighResTimer& timer,
                                 WorkFunction work_function,
                                 const char* thread_name) {
  char* buffer_start = static_cast<char*>(buffer);
  auto make_work = [buffer_start, work_function](size_t chunk_start_offset,
                                                  size_t thread_chunk_size,
                                                  int iterations_local,
                                                  size_t worker_index) {
    char* thread_chunk_start = buffer_start + chunk_start_offset;
    return [thread_chunk_start, thread_chunk_size, iterations_local, worker_index,
            work_function]() {
      work_function(thread_chunk_start, thread_chunk_size, iterations_local, worker_index);
    };
  };

  return run_parallel_test_common(buffer, size, iterations, num_threads, timer,
                                  thread_name, make_work);
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
  auto make_work = [dst_start, src_start, work_function](size_t chunk_start_offset,
                                                         size_t thread_chunk_size,
                                                         int iterations_local,
                                                         size_t /* worker_index */) {
    char* thread_dst_chunk = dst_start + chunk_start_offset;
    char* thread_src_chunk = src_start + chunk_start_offset;
    return [thread_dst_chunk, thread_src_chunk, thread_chunk_size, iterations_local,
            work_function]() {
      work_function(thread_dst_chunk, thread_src_chunk, thread_chunk_size, iterations_local);
    };
  };

  return run_parallel_test_common(dst, size, iterations, num_threads, timer, thread_name, make_work);
}

#endif // PARALLEL_TEST_FRAMEWORK_H

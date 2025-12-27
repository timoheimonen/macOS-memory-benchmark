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

// --- Generic Parallel Test Framework ---

// Generic function to run parallel tests with a common threading pattern.
// 'size': Total size of the buffer(s) in bytes.
// 'iterations': How many times to perform the operation.
// 'num_threads': Number of threads to use.
// 'timer': High-resolution timer for measuring execution time.
// 'work_function': Function to call for each thread chunk. Takes (chunk_start, chunk_size, iterations).
// 'thread_name': Name of the thread type for QoS error messages (e.g., "read", "write", "copy").
// Returns: Total duration in seconds.
template<typename WorkFunction>
double run_parallel_test(size_t size, int iterations, int num_threads, HighResTimer &timer,
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

  // Launch threads once; each handles its chunk for all iterations.
  for (int t = 0; t < num_threads; ++t) {
    size_t current_chunk_size = chunk_base_size + (t < chunk_remainder ? 1 : 0);
    if (current_chunk_size == 0) continue;  // Avoid launching threads for zero work.

    // Store original offset for this thread (before alignment)
    size_t original_offset = offset;
    
    // iterations loop inside thread lambda for re-use (avoids per-iteration creation).
    threads.emplace_back([current_chunk_size, original_offset, iterations, &start_mutex, &start_cv,
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

      // Execute the work function for this chunk
      work_function(original_offset, current_chunk_size, iterations);
    });
    
    // Update offset for the next chunk using original chunk size calculation
    offset += (chunk_base_size + (t < chunk_remainder ? 1 : 0));
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


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

#include "benchmark.h"  // Include benchmark definitions (assembly funcs, HighResTimer)

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

    // iterations loop inside thread lambda for re-use (avoids per-iteration creation).
    threads.emplace_back([current_chunk_size, offset, iterations, &start_mutex, &start_cv,
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
      work_function(offset, current_chunk_size, iterations);
    });
    offset += current_chunk_size;  // Update offset for the next chunk.
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

// --- Benchmark Execution Functions ---

// Executes the multi-threaded read bandwidth benchmark.
// 'buffer': Pointer to the memory buffer to read.
// 'size': Size of the buffer in bytes.
// 'iterations': How many times to read the entire buffer.
// 'num_threads': Number of threads to use.
// 'checksum': Atomic variable to accumulate checksums from threads (ensures work isn't optimized away).
// 'timer': High-resolution timer for measuring execution time.
// Returns: Total duration in seconds.
double run_read_test(void *buffer, size_t size, int iterations, int num_threads, std::atomic<uint64_t> &checksum,
                     HighResTimer &timer) {
  checksum = 0;  // Ensure checksum starts at 0 for the measurement pass.
  
  // Define the work function for read operations
  auto read_work = [buffer, &checksum](size_t offset, size_t chunk_size, int iters) {
    char *chunk_start = static_cast<char *>(buffer) + offset;
    
    // Accumulate checksum locally to avoid atomic operations in the inner loop.
    uint64_t local_checksum = 0;
    for (int i = 0; i < iters; ++i) {
      // Call external assembly function for reading.
      uint64_t thread_checksum = memory_read_loop_asm(chunk_start, chunk_size);
      // Combine result locally (non-atomic).
      local_checksum ^= thread_checksum;
    }
    // Atomically combine final result (one atomic per thread, relaxed order is sufficient).
    checksum.fetch_xor(local_checksum, std::memory_order_relaxed);
  };
  
  return run_parallel_test(size, iterations, num_threads, timer, read_work, "read");
}

// Executes the multi-threaded write bandwidth benchmark.
// 'buffer': Pointer to the memory buffer to write to.
// 'size': Size of the buffer in bytes.
// 'iterations': How many times to write the entire buffer.
// 'num_threads': Number of threads to use.
// 'timer': High-resolution timer for measuring execution time.
// Returns: Total duration in seconds.
double run_write_test(void *buffer, size_t size, int iterations, int num_threads, HighResTimer &timer) {
  // Define the work function for write operations
  auto write_work = [buffer](size_t offset, size_t chunk_size, int iters) {
    char *chunk_start = static_cast<char *>(buffer) + offset;
    
    for (int i = 0; i < iters; ++i) {
      memory_write_loop_asm(chunk_start, chunk_size);
    }
  };
  
  return run_parallel_test(size, iterations, num_threads, timer, write_work, "write");
}

// Executes the multi-threaded copy bandwidth benchmark.
// 'dst': Pointer to the destination buffer.
// 'src': Pointer to the source buffer.
// 'size': Size of the data to copy in bytes.
// 'iterations': How many times to copy the data.
// 'num_threads': Number of threads to use.
// 'timer': High-resolution timer for measuring execution time.
// Returns: Total duration in seconds.
double run_copy_test(void *dst, void *src, size_t size, int iterations, int num_threads, HighResTimer &timer) {
  // Define the work function for copy operations
  auto copy_work = [dst, src](size_t offset, size_t chunk_size, int iters) {
    char *src_chunk = static_cast<char *>(src) + offset;
    char *dst_chunk = static_cast<char *>(dst) + offset;
    
    for (int i = 0; i < iters; ++i) {
      memory_copy_loop_asm(dst_chunk, src_chunk, chunk_size);
    }
  };
  
  return run_parallel_test(size, iterations, num_threads, timer, copy_work, "copy");
}

// Executes the single-threaded memory latency benchmark.
// 'buffer': Pointer to the buffer containing the pointer chain.
// 'num_accesses': Total number of pointer dereferences to perform.
// 'timer': High-resolution timer for measuring execution time.
// 'latency_samples': Optional output vector to store per-sample latencies (if provided and sample_count > 0).
// 'sample_count': Number of samples to collect (if 0, uses single measurement).
// Returns: Total duration in nanoseconds.
double run_latency_test(void *buffer, size_t num_accesses, HighResTimer &timer,
                        std::vector<double> *latency_samples, int sample_count) {
  // Get the starting address of the pointer chain.
  uintptr_t *lat_start_ptr = static_cast<uintptr_t *>(buffer);
  
  // If sample collection is requested
  if (latency_samples != nullptr && sample_count > 0) {
    latency_samples->clear();
    latency_samples->reserve(sample_count);
    
    size_t accesses_per_sample = num_accesses / sample_count;
    if (accesses_per_sample == 0) accesses_per_sample = 1;  // Ensure at least 1 access per sample
    
    double total_duration_ns = 0.0;
    
    for (int i = 0; i < sample_count; ++i) {
      timer.start();
      memory_latency_chase_asm(lat_start_ptr, accesses_per_sample);
      double sample_duration_ns = timer.stop_ns();
      double sample_latency_ns = sample_duration_ns / static_cast<double>(accesses_per_sample);
      latency_samples->push_back(sample_latency_ns);
      total_duration_ns += sample_duration_ns;
    }
    
    return total_duration_ns;
  } else {
    // Original single measurement behavior
    timer.start();  // Start timing.
    // Call external assembly function to chase the pointer chain.
    memory_latency_chase_asm(lat_start_ptr, num_accesses);
    // Stop timing, getting result in nanoseconds for latency.
    double duration_ns = timer.stop_ns();
    return duration_ns;  // Return total time elapsed in nanoseconds.
  }
}

// Executes the single-threaded cache latency benchmark for a specific cache level.
// Uses the same pointer chasing methodology as the main memory latency test.
// 'buffer': Pointer to the buffer containing the pointer chain (should fit in target cache).
// 'buffer_size': Size of the buffer in bytes (for validation/future use).
// 'num_accesses': Total number of pointer dereferences to perform.
// 'timer': High-resolution timer for measuring execution time.
// 'latency_samples': Optional output vector to store per-sample latencies (if provided and sample_count > 0).
// 'sample_count': Number of samples to collect (if 0, uses single measurement).
// Returns: Total duration in nanoseconds.
double run_cache_latency_test(void *buffer, size_t buffer_size, size_t num_accesses, HighResTimer &timer,
                              std::vector<double> *latency_samples, int sample_count) {
  // Get the starting address of the pointer chain.
  uintptr_t *lat_start_ptr = static_cast<uintptr_t *>(buffer);
  
  // If sample collection is requested
  if (latency_samples != nullptr && sample_count > 0) {
    latency_samples->clear();
    latency_samples->reserve(sample_count);
    
    size_t accesses_per_sample = num_accesses / sample_count;
    if (accesses_per_sample == 0) accesses_per_sample = 1;  // Ensure at least 1 access per sample
    
    double total_duration_ns = 0.0;
    
    for (int i = 0; i < sample_count; ++i) {
      timer.start();
      memory_latency_chase_asm(lat_start_ptr, accesses_per_sample);
      double sample_duration_ns = timer.stop_ns();
      double sample_latency_ns = sample_duration_ns / static_cast<double>(accesses_per_sample);
      latency_samples->push_back(sample_latency_ns);
      total_duration_ns += sample_duration_ns;
    }
    
    return total_duration_ns;
  } else {
    // Original single measurement behavior
    timer.start();  // Start timing.
    // Call external assembly function to chase the pointer chain (same as main latency test).
    memory_latency_chase_asm(lat_start_ptr, num_accesses);
    // Stop timing, getting result in nanoseconds for latency.
    double duration_ns = timer.stop_ns();
    return duration_ns;  // Return total time elapsed in nanoseconds.
  }
}
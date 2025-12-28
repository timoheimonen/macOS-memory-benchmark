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
#include <cstdint>               // For uint64_t

#include "benchmark/benchmark_tests.h"  // Function declarations
#include "core/timing/timer.h"  // HighResTimer
#include "asm/asm_functions.h"  // Assembly function declarations
#include "benchmark/parallel_test_framework.h"

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
  // Initialize checksum to 0 before measurement pass (thread-safe atomic assignment)
  checksum.store(0, std::memory_order_relaxed);
  
  // Define the work function for read operations
  // Alignment is now handled in the framework, so we receive aligned chunk_start directly
  auto read_work = [&checksum](char *chunk_start, size_t chunk_size, int iters) {
    // Accumulate checksum locally to avoid atomic operations in the inner loop.
    // This reduces contention and improves performance in multi-threaded scenarios.
    uint64_t local_checksum = 0;
    for (int i = 0; i < iters; ++i) {
      // Call external assembly function for reading.
      uint64_t thread_checksum = memory_read_loop_asm(chunk_start, chunk_size);
      // Combine result locally (non-atomic). XOR is commutative and associative,
      // so the order of combination doesn't matter for checksum correctness.
      local_checksum ^= thread_checksum;
    }
    // Atomically combine final result (one atomic per thread).
    // Using release memory order ensures:
    // 1. All previous operations in this thread are visible before the checksum update
    // 2. The checksum update is properly synchronized with thread completion
    // 3. XOR is commutative and associative, so order doesn't matter for correctness
    // 4. Each thread accumulates locally first, reducing contention
    checksum.fetch_xor(local_checksum, std::memory_order_release);
  };
  
  return run_parallel_test(buffer, size, iterations, num_threads, timer, read_work, "read");
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
  // Alignment is now handled in the framework, so we receive aligned chunk_start directly
  auto write_work = [](char *chunk_start, size_t chunk_size, int iters) {
    for (int i = 0; i < iters; ++i) {
      memory_write_loop_asm(chunk_start, chunk_size);
    }
  };
  
  return run_parallel_test(buffer, size, iterations, num_threads, timer, write_work, "write");
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
  // Alignment and src/dst correspondence are now handled in the framework,
  // so we receive aligned chunk pointers directly
  auto copy_work = [](char *dst_chunk, char *src_chunk, size_t chunk_size, int iters) {
    for (int i = 0; i < iters; ++i) {
      memory_copy_loop_asm(dst_chunk, src_chunk, chunk_size);
    }
  };
  
  return run_parallel_test_copy(dst, src, size, iterations, num_threads, timer, copy_work, "copy");
}


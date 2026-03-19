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
 * @file bandwidth_tests.cpp
 * @brief Bandwidth test implementations
 *
 * Implements multi-threaded bandwidth benchmark functions for read, write, and copy operations.
 * Uses parallel test framework for thread coordination and assembly functions for low-level
 * memory operations. Supports checksum validation to prevent compiler optimizations.
 *
 * Key features:
 * - Multi-threaded execution with automatic work division
 * - Atomic checksum accumulation for read validation
 * - Alignment-aware chunking handled by parallel framework
 * - High-resolution timing via HighResTimer
 */

#include <atomic>                // For std::atomic
#include <cstdint>               // For uint64_t

#include "benchmark/benchmark_tests.h"  // Function declarations
#include "core/timing/timer.h"  // HighResTimer
#include "asm/asm_functions.h"  // Assembly function declarations
#include "benchmark/parallel_test_framework.h"

namespace {

double run_read_test_impl(void* buffer,
                          size_t size,
                          int iterations,
                          int num_threads,
                          std::atomic<uint64_t>& checksum,
                          HighResTimer& timer,
                          uint64_t (*read_func)(const void*, size_t)) {
  checksum.store(0, std::memory_order_relaxed);

  auto read_work = [&checksum, read_func](char* chunk_start, size_t chunk_size, int iters) {
    uint64_t local_checksum = 0;
    for (int i = 0; i < iters; ++i) {
      uint64_t thread_checksum = read_func(chunk_start, chunk_size);
      local_checksum ^= thread_checksum;
    }
    checksum.fetch_xor(local_checksum, std::memory_order_release);
  };

  return run_parallel_test(buffer, size, iterations, num_threads, timer, read_work, "read");
}

double run_write_test_impl(void* buffer,
                           size_t size,
                           int iterations,
                           int num_threads,
                           HighResTimer& timer,
                           void (*write_func)(void*, size_t)) {
  auto write_work = [write_func](char* chunk_start, size_t chunk_size, int iters) {
    for (int i = 0; i < iters; ++i) {
      write_func(chunk_start, chunk_size);
    }
  };

  return run_parallel_test(buffer, size, iterations, num_threads, timer, write_work, "write");
}

double run_copy_test_impl(void* dst,
                          void* src,
                          size_t size,
                          int iterations,
                          int num_threads,
                          HighResTimer& timer,
                          void (*copy_func)(void*, const void*, size_t)) {
  auto copy_work = [copy_func](char* dst_chunk, char* src_chunk, size_t chunk_size, int iters) {
    for (int i = 0; i < iters; ++i) {
      copy_func(dst_chunk, src_chunk, chunk_size);
    }
  };

  return run_parallel_test_copy(dst, src, size, iterations, num_threads, timer, copy_work, "copy");
}

}  // namespace

/**
 * @brief Executes the multi-threaded read bandwidth benchmark.
 *
 * Measures memory read bandwidth using multiple threads. Each thread reads its assigned
 * portion of the buffer multiple times, accumulating a checksum to prevent compiler
 * optimizations from eliminating the read operations.
 *
 * Thread coordination:
 * - Work is divided among threads by the parallel test framework
 * - Each thread accumulates checksums locally to reduce atomic contention
 * - Final checksums are combined atomically using XOR (commutative and associative)
 * - Uses memory_order_release for proper synchronization
 *
 * @param[in]     buffer       Pointer to the memory buffer to read. Must be non-null.
 * @param[in]     size         Size of the buffer in bytes.
 * @param[in]     iterations   How many times to read the entire buffer.
 * @param[in]     num_threads  Number of threads to use for parallel execution.
 * @param[in,out] checksum     Atomic variable to accumulate checksums from threads.
 *                             Initialized to 0 before measurement, prevents optimization.
 * @param[in,out] timer        High-resolution timer for measuring execution time.
 *
 * @return Total duration in seconds
 *
 * @note The checksum accumulation ensures the compiler cannot optimize away read operations.
 * @note Uses assembly function memory_read_loop_asm() for actual memory reads.
 * @note Thread synchronization and alignment are handled by run_parallel_test().
 *
 * @see run_write_test() for write bandwidth measurement
 * @see run_copy_test() for copy bandwidth measurement
 * @see memory_read_loop_asm() for the low-level read implementation
 */
double run_read_test(void *buffer, size_t size, int iterations, int num_threads, std::atomic<uint64_t> &checksum,
                     HighResTimer &timer) {
  return run_read_test_impl(buffer, size, iterations, num_threads, checksum, timer, memory_read_loop_asm);
}

double run_read_test_with_kernel(void* buffer,
                                 size_t size,
                                 int iterations,
                                 int num_threads,
                                 std::atomic<uint64_t>& checksum,
                                 HighResTimer& timer,
                                 uint64_t (*read_func)(const void*, size_t)) {
  return run_read_test_impl(buffer, size, iterations, num_threads, checksum, timer, read_func);
}

/**
 * @brief Executes the multi-threaded write bandwidth benchmark.
 *
 * Measures memory write bandwidth using multiple threads. Each thread writes to its assigned
 * portion of the buffer multiple times. No checksum is needed as the write operations have
 * observable side effects that prevent compiler optimization.
 *
 * Thread coordination:
 * - Work is divided among threads by the parallel test framework
 * - Each thread writes to non-overlapping memory regions
 * - Synchronization ensures accurate timing of parallel execution
 *
 * @param[out]    buffer       Pointer to the memory buffer to write to. Must be non-null.
 * @param[in]     size         Size of the buffer in bytes.
 * @param[in]     iterations   How many times to write the entire buffer.
 * @param[in]     num_threads  Number of threads to use for parallel execution.
 * @param[in,out] timer        High-resolution timer for measuring execution time.
 *
 * @return Total duration in seconds
 *
 * @note Uses assembly function memory_write_loop_asm() for actual memory writes.
 * @note Thread synchronization and alignment are handled by run_parallel_test().
 * @note No checksum validation needed - writes have observable side effects.
 *
 * @see run_read_test() for read bandwidth measurement
 * @see run_copy_test() for copy bandwidth measurement
 * @see memory_write_loop_asm() for the low-level write implementation
 */
double run_write_test(void *buffer, size_t size, int iterations, int num_threads, HighResTimer &timer) {
  return run_write_test_impl(buffer, size, iterations, num_threads, timer, memory_write_loop_asm);
}

double run_write_test_with_kernel(void* buffer,
                                  size_t size,
                                  int iterations,
                                  int num_threads,
                                  HighResTimer& timer,
                                  void (*write_func)(void*, size_t)) {
  return run_write_test_impl(buffer, size, iterations, num_threads, timer, write_func);
}

/**
 * @brief Executes the multi-threaded copy bandwidth benchmark.
 *
 * Measures memory copy bandwidth using multiple threads. Each thread copies its assigned
 * portion from source to destination buffer multiple times. This test measures the combined
 * read+write bandwidth of the memory subsystem.
 *
 * Thread coordination:
 * - Work is divided among threads by the parallel test framework
 * - Each thread copies between corresponding non-overlapping regions
 * - Source and destination alignment is handled automatically
 *
 * Bandwidth calculation:
 * - Copy operations involve both read and write
 * - Effective bandwidth = (bytes_read + bytes_written) / time
 * - Actual calculation uses COPY_OPERATION_MULTIPLIER constant
 *
 * @param[out]    dst          Pointer to the destination buffer. Must be non-null.
 * @param[in]     src          Pointer to the source buffer. Must be non-null.
 * @param[in]     size         Size of the data to copy in bytes.
 * @param[in]     iterations   How many times to copy the data.
 * @param[in]     num_threads  Number of threads to use for parallel execution.
 * @param[in,out] timer        High-resolution timer for measuring execution time.
 *
 * @return Total duration in seconds
 *
 * @note Uses assembly function memory_copy_loop_asm() for actual memory copy.
 * @note Thread synchronization and alignment are handled by run_parallel_test_copy().
 * @note Copy bandwidth includes both read and write operations.
 *
 * @see run_read_test() for read bandwidth measurement
 * @see run_write_test() for write bandwidth measurement
 * @see memory_copy_loop_asm() for the low-level copy implementation
 */
double run_copy_test(void *dst, void *src, size_t size, int iterations, int num_threads, HighResTimer &timer) {
  return run_copy_test_impl(dst, src, size, iterations, num_threads, timer, memory_copy_loop_asm);
}

double run_copy_test_with_kernel(void* dst,
                                 void* src,
                                 size_t size,
                                 int iterations,
                                 int num_threads,
                                 HighResTimer& timer,
                                 void (*copy_func)(void*, const void*, size_t)) {
  return run_copy_test_impl(dst, src, size, iterations, num_threads, timer, copy_func);
}

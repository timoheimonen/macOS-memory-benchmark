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
#include "core/memory/memory_utils.h"
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
  auto read_work = [buffer, size, &checksum](size_t offset, size_t original_chunk_size, int iters) {
    // Calculate original end position
    size_t original_chunk_end = offset + original_chunk_size;
    
    char *unaligned_start = static_cast<char *>(buffer) + offset;
    // Align chunk_start to cache line boundary to prevent false sharing
    char *chunk_start = static_cast<char *>(align_ptr_to_cache_line(unaligned_start));
    
    // Calculate chunk size to cover the original range [offset, offset + original_chunk_size)
    // This ensures no gaps: chunk covers from aligned start to original end
    char *original_end_ptr = static_cast<char *>(buffer) + original_chunk_end;
    size_t chunk_size = original_end_ptr - chunk_start;
    
    // Ensure we don't exceed the buffer size
    char *buffer_end = static_cast<char *>(buffer) + size;
    if (chunk_start + chunk_size > buffer_end) {
      chunk_size = buffer_end - chunk_start;
    }
    
    // Skip if chunk size is zero or negative after alignment adjustments
    if (chunk_size == 0 || chunk_start >= buffer_end) return;
    
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
    // Using relaxed memory order is safe because:
    // 1. XOR is commutative and associative, so order doesn't matter
    // 2. We only care about the final checksum value, not intermediate states
    // 3. Each thread accumulates locally first, reducing contention
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
  auto write_work = [buffer, size](size_t offset, size_t original_chunk_size, int iters) {
    // Calculate original end position
    size_t original_chunk_end = offset + original_chunk_size;
    
    char *unaligned_start = static_cast<char *>(buffer) + offset;
    // Align chunk_start to cache line boundary to prevent false sharing
    char *chunk_start = static_cast<char *>(align_ptr_to_cache_line(unaligned_start));
    
    // Calculate chunk size to cover the original range [offset, offset + original_chunk_size)
    // This ensures no gaps: chunk covers from aligned start to original end
    char *original_end_ptr = static_cast<char *>(buffer) + original_chunk_end;
    size_t chunk_size = original_end_ptr - chunk_start;
    
    // Ensure we don't exceed the buffer size
    char *buffer_end = static_cast<char *>(buffer) + size;
    if (chunk_start + chunk_size > buffer_end) {
      chunk_size = buffer_end - chunk_start;
    }
    
    // Skip if chunk size is zero or negative after alignment adjustments
    if (chunk_size == 0 || chunk_start >= buffer_end) return;
    
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
  auto copy_work = [dst, src, size](size_t offset, size_t original_chunk_size, int iters) {
    // Calculate original end position
    size_t original_chunk_end = offset + original_chunk_size;
    
    // Calculate unaligned start positions
    char *dst_unaligned_start = static_cast<char *>(dst) + offset;
    char *src_unaligned_start = static_cast<char *>(src) + offset;
    
    // Align dst to cache line boundary to prevent false sharing
    char *dst_chunk = static_cast<char *>(align_ptr_to_cache_line(dst_unaligned_start));
    
    // Maintain src/dst correspondence by applying the same alignment offset
    // This ensures data integrity: src[i] always maps to dst[i]
    size_t alignment_offset = dst_chunk - dst_unaligned_start;
    char *src_chunk = src_unaligned_start + alignment_offset;
    
    // Calculate chunk size to cover the original range [offset, offset + original_chunk_size)
    // This ensures no gaps: chunk covers from aligned start to original end
    char *dst_original_end = static_cast<char *>(dst) + original_chunk_end;
    size_t chunk_size = dst_original_end - dst_chunk;
    
    // Ensure we don't exceed the buffer size
    char *dst_end = static_cast<char *>(dst) + size;
    if (dst_chunk + chunk_size > dst_end) {
      chunk_size = dst_end - dst_chunk;
    }
    
    // Skip if chunk size is zero or negative after alignment adjustments
    if (chunk_size == 0 || dst_chunk >= dst_end) return;
    
    for (int i = 0; i < iters; ++i) {
      memory_copy_loop_asm(dst_chunk, src_chunk, chunk_size);
    }
  };
  
  return run_parallel_test(size, iterations, num_threads, timer, copy_work, "copy");
}


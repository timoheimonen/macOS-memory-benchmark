// Copyright 2026 Timo Heimonen <timo.heimonen@proton.me>
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
 * @file helpers.cpp
 * @brief Helper functions for pattern benchmark test execution
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 *
 * This file provides helper functions that wrap the parallel test framework
 * for pattern-based memory access benchmarks. These functions handle the
 * execution of read, write, and copy operations for various access patterns
 * including sequential, strided, and random access.
 *
 * The helpers integrate assembly-level memory access functions with the
 * multi-threaded parallel test framework, managing checksums, timing, and
 * thread coordination.
 */
#include "pattern_benchmark/pattern_benchmark.h"
#include "utils/benchmark.h"
#include "benchmark/parallel_test_framework.h"
#include "asm/asm_functions.h"
#include "core/config/constants.h"
#include <atomic>

// ============================================================================
// Helper Functions for Pattern Tests
// ============================================================================

// Helper function to run a pattern read test (multi-threaded)
double run_pattern_read_test(void* buffer, size_t size, int iterations,
                             uint64_t (*read_func)(const void*, size_t),
                             std::atomic<uint64_t>& checksum, HighResTimer& timer,
                             int num_threads) {
  checksum.store(0, std::memory_order_relaxed);

  // Create work function that captures the read_func pointer
  auto read_work = [&checksum, read_func](char *chunk_start, size_t chunk_size, int iters) {
    uint64_t local_checksum = 0;
    for (int i = 0; i < iters; ++i) {
      uint64_t result = read_func(chunk_start, chunk_size);
      local_checksum ^= result;
    }
    checksum.fetch_xor(local_checksum, std::memory_order_release);
  };

  return run_parallel_test(buffer, size, iterations, num_threads, timer, read_work, "pattern_read");
}

// Helper function to run a pattern write test (multi-threaded)
double run_pattern_write_test(void* buffer, size_t size, int iterations,
                              void (*write_func)(void*, size_t),
                              HighResTimer& timer, int num_threads) {
  // Create work function that captures the write_func pointer
  auto write_work = [write_func](char *chunk_start, size_t chunk_size, int iters) {
    for (int i = 0; i < iters; ++i) {
      write_func(chunk_start, chunk_size);
    }
  };

  return run_parallel_test(buffer, size, iterations, num_threads, timer, write_work, "pattern_write");
}

// Helper function to run a pattern copy test (multi-threaded)
double run_pattern_copy_test(void* dst, void* src, size_t size, int iterations,
                             void (*copy_func)(void*, const void*, size_t),
                             HighResTimer& timer, int num_threads) {
  // Create work function that captures the copy_func pointer
  auto copy_work = [copy_func](char *dst_chunk, char *src_chunk, size_t chunk_size, int iters) {
    for (int i = 0; i < iters; ++i) {
      copy_func(dst_chunk, src_chunk, chunk_size);
    }
  };

  return run_parallel_test_copy(dst, src, size, iterations, num_threads, timer, copy_work, "pattern_copy");
}

// Helper function to run a strided pattern read test (multi-threaded)
double run_pattern_read_strided_test(void* buffer, size_t size, size_t stride, int iterations,
                                     std::atomic<uint64_t>& checksum, HighResTimer& timer,
                                     int num_threads) {
  using namespace Constants;
  checksum.store(0, std::memory_order_relaxed);

  // Create work function that captures stride
  auto strided_read_work = [&checksum, stride](char *chunk_start, size_t chunk_size, int iters) {
    uint64_t local_checksum = 0;

    // Calculate effective size (accounting for access size)
    if (chunk_size <= PATTERN_ACCESS_SIZE_BYTES) {
      return;  // Skip if chunk too small
    }
    size_t effective_size = chunk_size - PATTERN_ACCESS_SIZE_BYTES;

    // Calculate number of strided accesses for this chunk
    size_t num_accesses = (effective_size + stride - 1) / stride;  // Ceiling division

    for (int i = 0; i < iters; ++i) {
      uint64_t result = memory_read_strided_loop_asm(chunk_start, effective_size, stride, num_accesses);
      local_checksum ^= result;
    }
    checksum.fetch_xor(local_checksum, std::memory_order_release);
  };

  return run_parallel_test(buffer, size, iterations, num_threads, timer, strided_read_work, "strided_read");
}

// Helper function to run a strided pattern write test (multi-threaded)
double run_pattern_write_strided_test(void* buffer, size_t size, size_t stride, int iterations,
                                       HighResTimer& timer, int num_threads) {
  using namespace Constants;

  // Create work function that captures stride
  auto strided_write_work = [stride](char *chunk_start, size_t chunk_size, int iters) {
    // Calculate effective size (accounting for access size)
    if (chunk_size <= PATTERN_ACCESS_SIZE_BYTES) {
      return;  // Skip if chunk too small
    }
    size_t effective_size = chunk_size - PATTERN_ACCESS_SIZE_BYTES;

    // Calculate number of strided accesses for this chunk
    size_t num_accesses = (effective_size + stride - 1) / stride;  // Ceiling division

    for (int i = 0; i < iters; ++i) {
      memory_write_strided_loop_asm(chunk_start, effective_size, stride, num_accesses);
    }
  };

  return run_parallel_test(buffer, size, iterations, num_threads, timer, strided_write_work, "strided_write");
}

// Helper function to run a strided pattern copy test (multi-threaded)
double run_pattern_copy_strided_test(void* dst, void* src, size_t size, size_t stride, int iterations,
                                     HighResTimer& timer, int num_threads) {
  using namespace Constants;

  // Create work function that captures stride
  auto strided_copy_work = [stride](char *dst_chunk, char *src_chunk, size_t chunk_size, int iters) {
    // Calculate effective size (accounting for access size)
    if (chunk_size <= PATTERN_ACCESS_SIZE_BYTES) {
      return;  // Skip if chunk too small
    }
    size_t effective_size = chunk_size - PATTERN_ACCESS_SIZE_BYTES;

    // Calculate number of strided accesses for this chunk
    size_t num_accesses = (effective_size + stride - 1) / stride;  // Ceiling division

    for (int i = 0; i < iters; ++i) {
      memory_copy_strided_loop_asm(dst_chunk, src_chunk, effective_size, stride, num_accesses);
    }
  };

  return run_parallel_test_copy(dst, src, size, iterations, num_threads, timer, strided_copy_work, "strided_copy");
}

// Helper function to run a random pattern read test (multi-threaded)
double run_pattern_read_random_test(void* buffer, const std::vector<size_t>& indices, int iterations,
                                    std::atomic<uint64_t>& checksum, HighResTimer& timer,
                                    int num_threads, size_t buffer_size) {
  using namespace Constants;
  checksum.store(0, std::memory_order_relaxed);

  char* buffer_start = static_cast<char*>(buffer);

  // Create work function that filters indices for each chunk
  auto random_read_work = [&checksum, &indices, buffer_start](
                          char *chunk_start, size_t chunk_size, int iters) {
    uint64_t local_checksum = 0;

    // Calculate this chunk's offset from buffer start
    size_t chunk_offset = chunk_start - buffer_start;
    size_t chunk_end = chunk_offset + chunk_size;

    // Filter indices that fall within this chunk's range
    std::vector<size_t> chunk_indices;
    chunk_indices.reserve(indices.size() / 4);  // Estimate

    for (size_t idx : indices) {
      // Check if this index falls within our chunk
      if (idx >= chunk_offset && idx < chunk_end - PATTERN_ACCESS_SIZE_BYTES) {
        // Convert to chunk-relative offset
        chunk_indices.push_back(idx - chunk_offset);
      }
    }

    // Skip if no valid indices for this chunk
    if (chunk_indices.empty()) {
      return;
    }

    // Run iterations with chunk-specific indices
    for (int i = 0; i < iters; ++i) {
      uint64_t result = memory_read_random_loop_asm(chunk_start, chunk_indices.data(),
                                                     chunk_indices.size());
      local_checksum ^= result;
    }
    checksum.fetch_xor(local_checksum, std::memory_order_release);
  };

  return run_parallel_test(buffer, buffer_size, iterations, num_threads, timer,
                          random_read_work, "random_read");
}

// Helper function to run a random pattern write test (multi-threaded)
double run_pattern_write_random_test(void* buffer, const std::vector<size_t>& indices, int iterations,
                                     HighResTimer& timer, int num_threads, size_t buffer_size) {
  using namespace Constants;
  char* buffer_start = static_cast<char*>(buffer);

  // Create work function that filters indices for each chunk
  auto random_write_work = [&indices, buffer_start](char *chunk_start, size_t chunk_size, int iters) {
    // Calculate this chunk's offset from buffer start
    size_t chunk_offset = chunk_start - buffer_start;
    size_t chunk_end = chunk_offset + chunk_size;

    // Filter indices that fall within this chunk's range
    std::vector<size_t> chunk_indices;
    chunk_indices.reserve(indices.size() / 4);  // Estimate

    for (size_t idx : indices) {
      // Check if this index falls within our chunk
      if (idx >= chunk_offset && idx < chunk_end - PATTERN_ACCESS_SIZE_BYTES) {
        // Convert to chunk-relative offset
        chunk_indices.push_back(idx - chunk_offset);
      }
    }

    // Skip if no valid indices for this chunk
    if (chunk_indices.empty()) {
      return;
    }

    // Run iterations with chunk-specific indices
    for (int i = 0; i < iters; ++i) {
      memory_write_random_loop_asm(chunk_start, chunk_indices.data(), chunk_indices.size());
    }
  };

  return run_parallel_test(buffer, buffer_size, iterations, num_threads, timer,
                          random_write_work, "random_write");
}

// Helper function to run a random pattern copy test (multi-threaded)
double run_pattern_copy_random_test(void* dst, void* src, const std::vector<size_t>& indices, int iterations,
                                    HighResTimer& timer, int num_threads, size_t buffer_size) {
  using namespace Constants;
  char* dst_buffer_start = static_cast<char*>(dst);

  // Create work function that filters indices for each chunk
  auto random_copy_work = [&indices, dst_buffer_start](
                          char *dst_chunk, char *src_chunk, size_t chunk_size, int iters) {
    // Calculate this chunk's offset from buffer start
    size_t chunk_offset = dst_chunk - dst_buffer_start;
    size_t chunk_end = chunk_offset + chunk_size;

    // Filter indices that fall within this chunk's range
    std::vector<size_t> chunk_indices;
    chunk_indices.reserve(indices.size() / 4);  // Estimate

    for (size_t idx : indices) {
      // Check if this index falls within our chunk
      if (idx >= chunk_offset && idx < chunk_end - PATTERN_ACCESS_SIZE_BYTES) {
        // Convert to chunk-relative offset
        chunk_indices.push_back(idx - chunk_offset);
      }
    }

    // Skip if no valid indices for this chunk
    if (chunk_indices.empty()) {
      return;
    }

    // Run iterations with chunk-specific indices
    for (int i = 0; i < iters; ++i) {
      memory_copy_random_loop_asm(dst_chunk, src_chunk, chunk_indices.data(), chunk_indices.size());
    }
  };

  return run_parallel_test_copy(dst, src, buffer_size, iterations, num_threads, timer,
                               random_copy_work, "random_copy");
}

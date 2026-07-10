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
#include "pattern_benchmark/pattern_work_plan.h"
#include "utils/benchmark.h"
#include "benchmark/parallel_test_framework.h"
#include "asm/asm_functions.h"
#include "core/config/constants.h"
#include <algorithm>
#include <atomic>

// ============================================================================
// Helper Functions for Pattern Tests
// ============================================================================

namespace {

inline bool is_valid_strided_chunk(size_t chunk_size, size_t stride) {
  using namespace Constants;
  return stride >= PATTERN_ACCESS_SIZE_BYTES &&
         stride % PATTERN_ACCESS_SIZE_BYTES == 0 &&
         chunk_size >= stride &&
         chunk_size - stride >= PATTERN_ACCESS_SIZE_BYTES;
}

const PatternRandomWorkerIndices* find_random_worker_indices(
    const std::vector<PatternRandomWorkerIndices>& worker_indices,
    size_t chunk_offset,
    size_t chunk_size) {
  const auto worker = std::lower_bound(
      worker_indices.begin(), worker_indices.end(), chunk_offset,
      [](const PatternRandomWorkerIndices& candidate, size_t offset) {
        return candidate.offset_bytes < offset;
      });
  if (worker == worker_indices.end() || worker->offset_bytes != chunk_offset ||
      worker->span_bytes != chunk_size) {
    return nullptr;
  }
  return &*worker;
}

}  // namespace

// Helper function to run a pattern read test (multi-threaded)
double run_pattern_read_test(void* buffer, size_t size, int iterations,
                             uint64_t (*read_func)(const void*, size_t),
                             std::atomic<uint64_t>& checksum, HighResTimer& timer,
                             int num_threads) {
  checksum.store(0, std::memory_order_relaxed);
  if (num_threads <= 0) {
    return 0.0;
  }
  std::vector<uint64_t> worker_checksums(static_cast<size_t>(num_threads), 0);

  auto read_work = [&worker_checksums, read_func](char *chunk_start, size_t chunk_size,
                                                  int iters, size_t worker_index) {
    uint64_t local_checksum = 0;
    for (int i = 0; i < iters; ++i) {
      uint64_t result = read_func(chunk_start, chunk_size);
      local_checksum ^= result;
    }
    worker_checksums[worker_index] = local_checksum;
  };

  const double duration = run_parallel_test_indexed(
      buffer, size, iterations, num_threads, timer, read_work, "pattern_read");
  uint64_t combined_checksum = 0;
  for (uint64_t worker_checksum : worker_checksums) {
    combined_checksum ^= worker_checksum;
  }
  checksum.store(combined_checksum, std::memory_order_release);
  return duration;
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
  if (num_threads <= 0) {
    return 0.0;
  }
  std::vector<uint64_t> worker_checksums(static_cast<size_t>(num_threads), 0);

  auto strided_read_work = [&worker_checksums, stride](char *chunk_start,
                                                       size_t chunk_size, int iters,
                                                       size_t worker_index) {
    if (!is_valid_strided_chunk(chunk_size, stride)) {
      return;
    }
    const uint64_t result = memory_read_strided_phased_loop_asm(
        chunk_start, chunk_size, stride, static_cast<size_t>(iters), 0);
    worker_checksums[worker_index] = result;
  };

  const double duration = run_parallel_test_indexed(
      buffer, size, iterations, num_threads, timer, strided_read_work, "strided_read");
  uint64_t combined_checksum = 0;
  for (uint64_t worker_checksum : worker_checksums) {
    combined_checksum ^= worker_checksum;
  }
  checksum.store(combined_checksum, std::memory_order_release);
  return duration;
}

// Helper function to run a strided pattern write test (multi-threaded)
double run_pattern_write_strided_test(void* buffer, size_t size, size_t stride, int iterations,
                                       HighResTimer& timer, int num_threads) {
  using namespace Constants;

  // Create work function that captures stride
  auto strided_write_work = [stride](char *chunk_start, size_t chunk_size, int iters) {
    if (!is_valid_strided_chunk(chunk_size, stride)) {
      return;
    }
    memory_write_strided_phased_loop_asm(chunk_start, chunk_size, stride,
                                         static_cast<size_t>(iters), 0);
  };

  return run_parallel_test(buffer, size, iterations, num_threads, timer, strided_write_work, "strided_write");
}

// Helper function to run a strided pattern copy test (multi-threaded)
double run_pattern_copy_strided_test(void* dst, void* src, size_t size, size_t stride, int iterations,
                                     HighResTimer& timer, int num_threads) {
  using namespace Constants;

  // Create work function that captures stride
  auto strided_copy_work = [stride](char *dst_chunk, char *src_chunk, size_t chunk_size, int iters) {
    if (!is_valid_strided_chunk(chunk_size, stride)) {
      return;
    }
    memory_copy_strided_phased_loop_asm(dst_chunk, src_chunk, chunk_size, stride,
                                        static_cast<size_t>(iters), 0);
  };

  return run_parallel_test_copy(dst, src, size, iterations, num_threads, timer, strided_copy_work, "strided_copy");
}

// Helper function to run a random pattern read test (multi-threaded)
double run_pattern_read_random_test(
    void* buffer,
    const std::vector<PatternRandomWorkerIndices>& worker_indices,
    int iterations,
    std::atomic<uint64_t>& checksum,
    HighResTimer& timer,
    int num_threads,
    size_t buffer_size) {
  checksum.store(0, std::memory_order_relaxed);
  if (num_threads <= 0) {
    return 0.0;
  }
  std::vector<uint64_t> worker_checksums(static_cast<size_t>(num_threads), 0);
  char* buffer_start = static_cast<char*>(buffer);

  auto random_read_work = [&worker_checksums, &worker_indices, buffer_start](
                          char *chunk_start, size_t chunk_size, int iters,
                          size_t worker_index) {
    uint64_t local_checksum = 0;
    const size_t chunk_offset = static_cast<size_t>(chunk_start - buffer_start);
    const PatternRandomWorkerIndices* worker =
        find_random_worker_indices(worker_indices, chunk_offset, chunk_size);
    if (worker == nullptr || worker->indices.empty()) {
      return;
    }

    for (int i = 0; i < iters; ++i) {
      uint64_t result =
          memory_read_random_loop_asm(chunk_start, worker->indices.data(), worker->indices.size());
      local_checksum ^= result;
    }
    worker_checksums[worker_index] = local_checksum;
  };

  const double duration = run_parallel_test_indexed(
      buffer, buffer_size, iterations, num_threads, timer, random_read_work,
      "random_read");
  uint64_t combined_checksum = 0;
  for (uint64_t worker_checksum : worker_checksums) {
    combined_checksum ^= worker_checksum;
  }
  checksum.store(combined_checksum, std::memory_order_release);
  return duration;
}

// Helper function to run a random pattern write test (multi-threaded)
double run_pattern_write_random_test(
    void* buffer,
    const std::vector<PatternRandomWorkerIndices>& worker_indices,
    int iterations,
    HighResTimer& timer,
    int num_threads,
    size_t buffer_size) {
  char* buffer_start = static_cast<char*>(buffer);

  auto random_write_work = [&worker_indices, buffer_start](
                           char *chunk_start, size_t chunk_size, int iters) {
    const size_t chunk_offset = static_cast<size_t>(chunk_start - buffer_start);
    const PatternRandomWorkerIndices* worker =
        find_random_worker_indices(worker_indices, chunk_offset, chunk_size);
    if (worker == nullptr || worker->indices.empty()) {
      return;
    }

    for (int i = 0; i < iters; ++i) {
      memory_write_random_loop_asm(chunk_start, worker->indices.data(), worker->indices.size());
    }
  };

  return run_parallel_test(buffer, buffer_size, iterations, num_threads, timer,
                          random_write_work, "random_write");
}

// Helper function to run a random pattern copy test (multi-threaded)
double run_pattern_copy_random_test(
    void* dst,
    void* src,
    const std::vector<PatternRandomWorkerIndices>& worker_indices,
    int iterations,
    HighResTimer& timer,
    int num_threads,
    size_t buffer_size) {
  char* dst_buffer_start = static_cast<char*>(dst);

  auto random_copy_work = [&worker_indices, dst_buffer_start](
                          char *dst_chunk, char *src_chunk, size_t chunk_size, int iters) {
    const size_t chunk_offset = static_cast<size_t>(dst_chunk - dst_buffer_start);
    const PatternRandomWorkerIndices* worker =
        find_random_worker_indices(worker_indices, chunk_offset, chunk_size);
    if (worker == nullptr || worker->indices.empty()) {
      return;
    }

    for (int i = 0; i < iters; ++i) {
      memory_copy_random_loop_asm(dst_chunk, src_chunk, worker->indices.data(),
                                  worker->indices.size());
    }
  };

  return run_parallel_test_copy(dst, src, buffer_size, iterations, num_threads, timer,
                               random_copy_work, "random_copy");
}

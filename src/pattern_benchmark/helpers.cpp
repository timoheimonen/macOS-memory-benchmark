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
#include <atomic>
#include <limits>

// ============================================================================
// Helper Functions for Pattern Tests
// ============================================================================

namespace {

template <typename WorkerRange>
std::vector<size_t> build_finalized_boundaries(const std::vector<WorkerRange>& workers) {
  if (workers.empty()) return {};
  std::vector<size_t> boundaries;
  boundaries.reserve(workers.size() + 1);
  boundaries.push_back(0);
  size_t expected_offset = 0;
  for (const WorkerRange& worker : workers) {
    if (worker.offset_bytes != expected_offset || worker.span_bytes == 0 ||
        worker.span_bytes > std::numeric_limits<size_t>::max() - expected_offset) {
      return {};
    }
    expected_offset += worker.span_bytes;
    boundaries.push_back(expected_offset);
  }
  return boundaries;
}

bool validate_random_worker_plan(
    const std::vector<PatternRandomWorkerIndices>& workers,
    const std::vector<size_t>& boundaries) {
  if (workers.size() > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      boundaries.size() != workers.size() + 1) {
    return false;
  }
  for (size_t worker_index = 0; worker_index < workers.size(); ++worker_index) {
    const PatternRandomWorkerIndices& worker = workers[worker_index];
    if (worker.offset_bytes != boundaries[worker_index] ||
        worker.span_bytes != boundaries[worker_index + 1] - boundaries[worker_index] ||
        worker.indices.empty() || worker.span_bytes < Constants::PATTERN_ACCESS_SIZE_BYTES) {
      return false;
    }
    for (size_t offset : worker.indices) {
      if (offset > worker.span_bytes - Constants::PATTERN_ACCESS_SIZE_BYTES) return false;
    }
  }
  return true;
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
double run_pattern_read_strided_test(void* buffer, const PatternWorkPlan& plan, std::atomic<uint64_t>& checksum,
                                     HighResTimer& timer) {
  checksum.store(0, std::memory_order_relaxed);
  const std::vector<size_t> boundaries = build_finalized_boundaries(plan.workers);
  if (boundaries.empty() || plan.passes == 0 || plan.passes > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return 0.0;
  }
  const size_t size = boundaries.back();
  const int iterations = static_cast<int>(plan.passes);
  const size_t stride = plan.stride_bytes;
  std::vector<uint64_t> worker_checksums(plan.workers.size(), 0);

  auto strided_read_work = [&worker_checksums, stride](char* chunk_start, size_t chunk_size, int iters,
                                                      size_t worker_index) {
    const uint64_t result =
        memory_read_strided_phased_loop_asm(chunk_start, chunk_size, stride, static_cast<size_t>(iters), 0);
    worker_checksums[worker_index] = result;
  };

  const double duration = run_parallel_test_indexed_with_boundaries(buffer, size, iterations, timer, boundaries,
                                                                    strided_read_work, "strided_read");
  uint64_t combined_checksum = 0;
  for (uint64_t worker_checksum : worker_checksums) {
    combined_checksum ^= worker_checksum;
  }
  checksum.store(combined_checksum, std::memory_order_release);
  return duration;
}

// Helper function to run a strided pattern write test (multi-threaded)
double run_pattern_write_strided_test(void* buffer, const PatternWorkPlan& plan, HighResTimer& timer) {
  const std::vector<size_t> boundaries = build_finalized_boundaries(plan.workers);
  if (boundaries.empty() || plan.passes == 0 || plan.passes > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return 0.0;
  }
  const size_t size = boundaries.back();
  const int iterations = static_cast<int>(plan.passes);
  const size_t stride = plan.stride_bytes;

  auto strided_write_work = [stride](char* chunk_start, size_t chunk_size, int iters,
                                     size_t /* worker_index */) {
    memory_write_strided_phased_loop_asm(chunk_start, chunk_size, stride, static_cast<size_t>(iters), 0);
  };

  return run_parallel_test_indexed_with_boundaries(buffer, size, iterations, timer, boundaries, strided_write_work,
                                                   "strided_write");
}

// Helper function to run a strided pattern copy test (multi-threaded)
double run_pattern_copy_strided_test(void* dst, void* src, const PatternWorkPlan& plan, HighResTimer& timer) {
  const std::vector<size_t> boundaries = build_finalized_boundaries(plan.workers);
  if (boundaries.empty() || plan.passes == 0 || plan.passes > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return 0.0;
  }
  const size_t size = boundaries.back();
  const int iterations = static_cast<int>(plan.passes);
  const size_t stride = plan.stride_bytes;

  auto strided_copy_work = [stride](char* dst_chunk, char* src_chunk, size_t chunk_size, int iters,
                                   size_t /* worker_index */) {
    memory_copy_strided_phased_loop_asm(dst_chunk, src_chunk, chunk_size, stride, static_cast<size_t>(iters), 0);
  };

  return run_parallel_test_copy_indexed_with_boundaries(dst, src, size, iterations, timer, boundaries,
                                                        strided_copy_work, "strided_copy");
}

// Helper function to run a random pattern read test (multi-threaded)
double run_pattern_read_random_test(void* buffer, const std::vector<PatternRandomWorkerIndices>& worker_indices,
                                    int iterations, std::atomic<uint64_t>& checksum, HighResTimer& timer) {
  checksum.store(0, std::memory_order_relaxed);
  const std::vector<size_t> boundaries = build_finalized_boundaries(worker_indices);
  if (boundaries.empty() || !validate_random_worker_plan(worker_indices, boundaries)) {
    return 0.0;
  }
  const size_t buffer_size = boundaries.back();
  std::vector<uint64_t> worker_checksums(worker_indices.size(), 0);
  char* buffer_start = static_cast<char*>(buffer);

  auto make_work = [buffer_start, &worker_checksums, &worker_indices](
                       size_t chunk_start_offset, size_t /* chunk_size */, int iters,
                       size_t worker_index) {
    char* chunk_start = buffer_start + chunk_start_offset;
    const PatternRandomWorkerIndices& worker = worker_indices[worker_index];
    const size_t* indices = worker.indices.data();
    const size_t index_count = worker.indices.size();
    uint64_t* worker_checksum = &worker_checksums[worker_index];
    return [chunk_start, indices, index_count, iters, worker_checksum]() {
      uint64_t local_checksum = 0;
      for (int i = 0; i < iters; ++i) {
        local_checksum ^= memory_read_random_loop_asm(chunk_start, indices, index_count);
      }
      *worker_checksum = local_checksum;
    };
  };

  const double duration = run_parallel_test_common(
      buffer, buffer_size, iterations, static_cast<int>(worker_indices.size()), timer,
      "random_read", make_work, &boundaries);
  uint64_t combined_checksum = 0;
  for (uint64_t worker_checksum : worker_checksums) {
    combined_checksum ^= worker_checksum;
  }
  checksum.store(combined_checksum, std::memory_order_release);
  return duration;
}

// Helper function to run a random pattern write test (multi-threaded)
double run_pattern_write_random_test(void* buffer, const std::vector<PatternRandomWorkerIndices>& worker_indices,
                                     int iterations, HighResTimer& timer) {
  const std::vector<size_t> boundaries = build_finalized_boundaries(worker_indices);
  if (boundaries.empty() || !validate_random_worker_plan(worker_indices, boundaries)) return 0.0;
  const size_t buffer_size = boundaries.back();
  char* buffer_start = static_cast<char*>(buffer);

  auto make_work = [buffer_start, &worker_indices](size_t chunk_start_offset,
                                                   size_t /* chunk_size */, int iters,
                                                   size_t worker_index) {
    char* chunk_start = buffer_start + chunk_start_offset;
    const PatternRandomWorkerIndices& worker = worker_indices[worker_index];
    const size_t* indices = worker.indices.data();
    const size_t index_count = worker.indices.size();
    return [chunk_start, indices, index_count, iters]() {
      for (int i = 0; i < iters; ++i) {
        memory_write_random_loop_asm(chunk_start, indices, index_count);
      }
    };
  };

  return run_parallel_test_common(buffer, buffer_size, iterations,
                                  static_cast<int>(worker_indices.size()), timer,
                                  "random_write", make_work, &boundaries);
}

// Helper function to run a random pattern copy test (multi-threaded)
double run_pattern_copy_random_test(void* dst, void* src, const std::vector<PatternRandomWorkerIndices>& worker_indices,
                                    int iterations, HighResTimer& timer) {
  const std::vector<size_t> boundaries = build_finalized_boundaries(worker_indices);
  if (boundaries.empty() || !validate_random_worker_plan(worker_indices, boundaries)) return 0.0;
  const size_t buffer_size = boundaries.back();
  char* dst_start = static_cast<char*>(dst);
  char* src_start = static_cast<char*>(src);

  auto make_work = [dst_start, src_start, &worker_indices](
                       size_t chunk_start_offset, size_t /* chunk_size */, int iters,
                       size_t worker_index) {
    char* dst_chunk = dst_start + chunk_start_offset;
    char* src_chunk = src_start + chunk_start_offset;
    const PatternRandomWorkerIndices& worker = worker_indices[worker_index];
    const size_t* indices = worker.indices.data();
    const size_t index_count = worker.indices.size();
    return [dst_chunk, src_chunk, indices, index_count, iters]() {
      for (int i = 0; i < iters; ++i) {
        memory_copy_random_loop_asm(dst_chunk, src_chunk, indices, index_count);
      }
    };
  };

  return run_parallel_test_common(dst, buffer_size, iterations,
                                  static_cast<int>(worker_indices.size()), timer,
                                  "random_copy", make_work, &boundaries);
}

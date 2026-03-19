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
 * @file pattern_warmup.cpp
 * @brief Pattern benchmark warmup functions
 */

#include <atomic>    // For std::atomic
#include <iostream>  // For std::cerr
#include <thread>    // For std::thread
#include <vector>    // For std::vector

// macOS specific QoS
#include <mach/mach.h>    // For kern_return_t
#include <pthread/qos.h>  // For pthread_set_qos_class_self_np

#include "utils/benchmark.h"  // Includes definitions for assembly loops etc.
#include "output/console/messages/messages_api.h"   // For error and warning messages
#include "warmup/warmup.h"
#include "warmup/warmup_internal.h"
#include "core/config/constants.h"

namespace {

bool validate_strided_warmup_stride(size_t stride, size_t size) {
  if (stride < 32) {
    std::cerr << Messages::error_prefix() << Messages::error_stride_too_small() << std::endl;
    return false;
  }
  if (stride > size) {
    std::cerr << Messages::error_prefix() << Messages::error_stride_too_large(stride, size) << std::endl;
    return false;
  }
  if (stride % 32 != 0) {
    std::cerr << Messages::warning_prefix() << Messages::warning_stride_not_aligned(stride) << std::endl;
  }
  return true;
}

bool validate_random_warmup_indices(const std::vector<size_t>& indices) {
  if (indices.empty()) {
    std::cerr << Messages::error_prefix() << Messages::error_indices_empty() << std::endl;
    return false;
  }

  size_t sample_size = (indices.size() < 100) ? indices.size() : 100;
  for (size_t i = 0; i < sample_size; ++i) {
    if (indices[i] % 32 != 0) {
      std::cerr << Messages::error_prefix() << Messages::error_index_not_aligned(i, indices[i]) << std::endl;
      return false;
    }
  }

  return true;
}

uint64_t run_strided_read_warmup_kernel(const void* src,
                                        size_t byte_count,
                                        size_t stride,
                                        size_t num_iterations) {
  using namespace Constants;

  switch (stride) {
    case PATTERN_STRIDE_CACHE_LINE:
      return memory_read_strided_64_loop_asm(src, byte_count, num_iterations);
    case PATTERN_STRIDE_PAGE:
      return memory_read_strided_4096_loop_asm(src, byte_count, num_iterations);
    case PATTERN_STRIDE_PAGE_16K:
      return memory_read_strided_16384_loop_asm(src, byte_count, num_iterations);
    case PATTERN_STRIDE_SUPERPAGE_2MB:
      return memory_read_strided_2mb_loop_asm(src, byte_count, num_iterations);
    default:
      return memory_read_strided_loop_asm(src, byte_count, stride, num_iterations);
  }
}

void run_strided_write_warmup_kernel(void* dst,
                                     size_t byte_count,
                                     size_t stride,
                                     size_t num_iterations) {
  using namespace Constants;

  switch (stride) {
    case PATTERN_STRIDE_CACHE_LINE:
      memory_write_strided_64_loop_asm(dst, byte_count, num_iterations);
      return;
    case PATTERN_STRIDE_PAGE:
      memory_write_strided_4096_loop_asm(dst, byte_count, num_iterations);
      return;
    case PATTERN_STRIDE_PAGE_16K:
      memory_write_strided_16384_loop_asm(dst, byte_count, num_iterations);
      return;
    case PATTERN_STRIDE_SUPERPAGE_2MB:
      memory_write_strided_2mb_loop_asm(dst, byte_count, num_iterations);
      return;
    default:
      memory_write_strided_loop_asm(dst, byte_count, stride, num_iterations);
      return;
  }
}

void run_strided_copy_warmup_kernel(void* dst,
                                    const void* src,
                                    size_t byte_count,
                                    size_t stride,
                                    size_t num_iterations) {
  using namespace Constants;

  switch (stride) {
    case PATTERN_STRIDE_CACHE_LINE:
      memory_copy_strided_64_loop_asm(dst, src, byte_count, num_iterations);
      return;
    case PATTERN_STRIDE_PAGE:
      memory_copy_strided_4096_loop_asm(dst, src, byte_count, num_iterations);
      return;
    case PATTERN_STRIDE_PAGE_16K:
      memory_copy_strided_16384_loop_asm(dst, src, byte_count, num_iterations);
      return;
    case PATTERN_STRIDE_SUPERPAGE_2MB:
      memory_copy_strided_2mb_loop_asm(dst, src, byte_count, num_iterations);
      return;
    default:
      memory_copy_strided_loop_asm(dst, src, byte_count, stride, num_iterations);
      return;
  }
}

template<typename RandomOp>
void run_random_warmup_operation(const std::vector<size_t>& indices, int num_threads, RandomOp operation) {
  if (!validate_random_warmup_indices(indices)) {
    return;
  }

  if (num_threads <= 0) {
    return;
  }

  if (num_threads == 1 || indices.size() <= static_cast<size_t>(num_threads)) {
    operation(indices.data(), indices.size());
    return;
  }

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  size_t indices_per_thread = indices.size() / num_threads;
  size_t remainder = indices.size() % num_threads;

  size_t offset = 0;
  for (int t = 0; t < num_threads; ++t) {
    size_t thread_count = indices_per_thread + (t < static_cast<int>(remainder) ? 1 : 0);
    if (thread_count == 0 || offset >= indices.size()) continue;

    size_t start_idx = offset;
    size_t end_idx = offset + thread_count;
    if (end_idx > indices.size()) end_idx = indices.size();
    offset = end_idx;

    std::vector<size_t> thread_indices(indices.begin() + start_idx, indices.begin() + end_idx);

    threads.emplace_back([thread_indices = std::move(thread_indices), operation]() {
      kern_return_t qos_ret = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
      if (qos_ret != KERN_SUCCESS) {
        std::cerr << Messages::warning_prefix() << Messages::warning_qos_failed_worker_thread(qos_ret) << std::endl;
      }

      if (!thread_indices.empty()) {
        operation(thread_indices.data(), thread_indices.size());
      }
    });
  }

  join_threads(threads);
}

}  // namespace

/**
 * @brief Warms up memory by reading from the buffer using strided access pattern.
 *
 * @param buffer Memory region to read from
 * @param size Total size of the buffer in bytes
 * @param stride Stride size in bytes (must be >= 32, aligned, and <= size)
 * @param num_threads Number of concurrent threads to use
 * @param dummy_checksum Atomic variable to accumulate a dummy result (prevents optimization)
 */
void warmup_read_strided(void* buffer, size_t size, size_t stride, int num_threads, std::atomic<uint64_t>& dummy_checksum) {
  if (!validate_strided_warmup_stride(stride, size)) {
    return;
  }
  
  size_t warmup_size = calculate_warmup_size(size);
  auto read_chunk_op = [stride](char* chunk_start, char* /* src_chunk */, size_t chunk_size,
                                 std::atomic<uint64_t>* checksum) {
    // Use strided read for warmup
    size_t num_iterations = (chunk_size + stride - 1) / stride;
    uint64_t result = run_strided_read_warmup_kernel(chunk_start, chunk_size, stride, num_iterations);
    if (checksum) {
      checksum->fetch_xor(result, std::memory_order_release);
    }
  };
  warmup_parallel(buffer, size, num_threads, read_chunk_op, true, nullptr, &dummy_checksum, warmup_size);
}

/**
 * @brief Warms up memory by writing to the buffer using strided access pattern.
 *
 * @param buffer Memory region to write to
 * @param size Total size of the buffer in bytes
 * @param stride Stride size in bytes (must be >= 32, aligned, and <= size)
 * @param num_threads Number of concurrent threads to use
 */
void warmup_write_strided(void* buffer, size_t size, size_t stride, int num_threads) {
  if (!validate_strided_warmup_stride(stride, size)) {
    return;
  }
  
  size_t warmup_size = calculate_warmup_size(size);
  auto write_chunk_op = [stride](char* chunk_start, char* /* src_chunk */, size_t chunk_size,
                                  std::atomic<uint64_t>* /* checksum */) {
    // Use strided write for warmup
    size_t num_iterations = (chunk_size + stride - 1) / stride;
    run_strided_write_warmup_kernel(chunk_start, chunk_size, stride, num_iterations);
  };
  warmup_parallel(buffer, size, num_threads, write_chunk_op, true, nullptr, nullptr, warmup_size);
}

/**
 * @brief Warms up memory by copying data between buffers using strided access pattern.
 *
 * @param dst Destination memory region
 * @param src Source memory region
 * @param size Total size of data to copy in bytes
 * @param stride Stride size in bytes (must be >= 32, aligned, and <= size)
 * @param num_threads Number of concurrent threads to use
 */
void warmup_copy_strided(void* dst, void* src, size_t size, size_t stride, int num_threads) {
  if (!validate_strided_warmup_stride(stride, size)) {
    return;
  }
  
  size_t warmup_size = calculate_warmup_size(size);
  auto copy_chunk_op = [stride](char* dst_chunk, char* src_chunk, size_t chunk_size,
                                 std::atomic<uint64_t>* /* checksum */) {
    // Use strided copy for warmup
    size_t num_iterations = (chunk_size + stride - 1) / stride;
    run_strided_copy_warmup_kernel(dst_chunk, src_chunk, chunk_size, stride, num_iterations);
  };
  warmup_parallel(dst, size, num_threads, copy_chunk_op, true, src, nullptr, warmup_size);
}

/**
 * @brief Warms up memory by reading from the buffer using random access pattern.
 *
 * @param buffer Memory region to read from
 * @param indices Vector of byte offsets (must be 32-byte aligned and within buffer bounds)
 * @param num_threads Number of concurrent threads to use
 * @param dummy_checksum Atomic variable to accumulate a dummy result (prevents optimization)
 */
void warmup_read_random(void* buffer, const std::vector<size_t>& indices, int num_threads, std::atomic<uint64_t>& dummy_checksum) {
  run_random_warmup_operation(indices, num_threads,
                              [buffer, &dummy_checksum](const size_t* random_indices, size_t random_count) {
                                uint64_t result = memory_read_random_loop_asm(buffer, random_indices, random_count);
                                dummy_checksum.fetch_xor(result, std::memory_order_release);
                              });
}

/**
 * @brief Warms up memory by writing to the buffer using random access pattern.
 *
 * @param buffer Memory region to write to
 * @param indices Vector of byte offsets (must be 32-byte aligned and within buffer bounds)
 * @param num_threads Number of concurrent threads to use
 */
void warmup_write_random(void* buffer, const std::vector<size_t>& indices, int num_threads) {
  run_random_warmup_operation(indices, num_threads,
                              [buffer](const size_t* random_indices, size_t random_count) {
                                memory_write_random_loop_asm(buffer, random_indices, random_count);
                              });
}

/**
 * @brief Warms up memory by copying data between buffers using random access pattern.
 *
 * @param dst Destination memory region
 * @param src Source memory region
 * @param indices Vector of byte offsets (must be 32-byte aligned and within buffer bounds)
 * @param num_threads Number of concurrent threads to use
 */
void warmup_copy_random(void* dst, void* src, const std::vector<size_t>& indices, int num_threads) {
  run_random_warmup_operation(indices, num_threads,
                              [dst, src](const size_t* random_indices, size_t random_count) {
                                memory_copy_random_loop_asm(dst, src, random_indices, random_count);
                              });
}

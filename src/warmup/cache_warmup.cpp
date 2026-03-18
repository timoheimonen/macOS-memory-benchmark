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
 * @file cache_warmup.cpp
 * @brief Cache-specific warmup functions
 */

#include "warmup/warmup.h"
#include "warmup/warmup_internal.h"

/**
 * @brief Warms up cache bandwidth test by reading from the buffer using multiple threads.
 *
 * Cache buffers are small (typically L1 ~128KB, L2 ~12-16MB), so we warm the entire buffer.
 * Unlike main memory warmup which uses calculate_warmup_size() to limit overhead,
 * cache warmup explicitly uses the full buffer size to ensure complete cache coverage.
 *
 * @param src_buffer Source memory region to read from (cache-sized)
 * @param size Total size of the buffer in bytes (cache size)
 * @param num_threads Number of concurrent threads to use
 * @param dummy_checksum Atomic variable to accumulate a dummy result (prevents optimization)
 */
void warmup_cache_read(void* src_buffer, size_t size, int num_threads, std::atomic<uint64_t>& dummy_checksum) {
  if (src_buffer == nullptr || size == 0) {
    return;
  }
  size_t warmup_size = size;
  warmup_parallel(src_buffer, size, num_threads, warmup_read_chunk_op, true, nullptr, &dummy_checksum, warmup_size);
}

/**
 * @brief Warms up cache bandwidth test by writing to the buffer using multiple threads.
 *
 * Cache buffers are small (typically L1 ~128KB, L2 ~12-16MB), so we warm the entire buffer.
 * Unlike main memory warmup which uses calculate_warmup_size() to limit overhead,
 * cache warmup explicitly uses the full buffer size to ensure complete cache coverage.
 *
 * @param dst_buffer Destination memory region to write to (cache-sized)
 * @param size Total size of the buffer in bytes (cache size)
 * @param num_threads Number of concurrent threads to use
 */
void warmup_cache_write(void* dst_buffer, size_t size, int num_threads) {
  if (dst_buffer == nullptr || size == 0) {
    return;
  }
  size_t warmup_size = size;
  warmup_parallel(dst_buffer, size, num_threads, warmup_write_chunk_op, true, nullptr, nullptr, warmup_size);
}

/**
 * @brief Warms up cache bandwidth test by copying data between buffers using multiple threads.
 *
 * Cache buffers are small (typically L1 ~128KB, L2 ~12-16MB), so we warm the entire buffer.
 * Unlike main memory warmup which uses calculate_warmup_size() to limit overhead,
 * cache warmup explicitly uses the full buffer size to ensure complete cache coverage.
 *
 * @param dst Destination memory region (cache-sized)
 * @param src Source memory region (cache-sized)
 * @param size Total size of data to copy in bytes (cache size)
 * @param num_threads Number of concurrent threads to use
 */
void warmup_cache_copy(void* dst, void* src, size_t size, int num_threads) {
  if (dst == nullptr || src == nullptr || size == 0) {
    return;
  }
  size_t warmup_size = size;
  warmup_parallel(dst, size, num_threads, warmup_copy_chunk_op, true, src, nullptr, warmup_size);
}

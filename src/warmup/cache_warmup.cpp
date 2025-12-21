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
#include <atomic>    // For std::atomic

#include "benchmark.h"  // Includes definitions for assembly loops etc.
#include "warmup/warmup.h"
#include "warmup/warmup_internal.h"

// Warms up cache bandwidth test by reading from the buffer (single thread).
// 'src_buffer': Source memory region to read from (cache-sized).
// 'size': Total size of the buffer in bytes (cache size).
// 'num_threads': Unused parameter (kept for API compatibility).
// 'dummy_checksum': Atomic variable to accumulate a dummy result (prevents optimization).
void warmup_cache_read(void* src_buffer, size_t size, int num_threads, std::atomic<uint64_t>& dummy_checksum) {
  (void)num_threads;  // Unused parameter
  auto read_op = [src_buffer, size, &dummy_checksum]() {
    // Call the assembly read loop function directly (single-threaded).
    uint64_t checksum = memory_read_loop_asm(src_buffer, size);
    // Store result to prevent optimization.
    dummy_checksum.fetch_xor(checksum, std::memory_order_relaxed);
  };
  warmup_single(read_op);
}

// Warms up cache bandwidth test by writing to the buffer (single thread).
// 'dst_buffer': Destination memory region to write to (cache-sized).
// 'size': Total size of the buffer in bytes (cache size).
// 'num_threads': Unused parameter (kept for API compatibility).
void warmup_cache_write(void* dst_buffer, size_t size, int num_threads) {
  (void)num_threads;  // Unused parameter
  auto write_op = [dst_buffer, size]() {
    // Call the assembly write loop function directly (single-threaded).
    memory_write_loop_asm(dst_buffer, size);
  };
  warmup_single(write_op);
}

// Warms up cache bandwidth test by copying data between buffers (single thread).
// 'dst': Destination memory region (cache-sized).
// 'src': Source memory region (cache-sized).
// 'size': Total size of data to copy in bytes (cache size).
// 'num_threads': Unused parameter (kept for API compatibility).
void warmup_cache_copy(void* dst, void* src, size_t size, int num_threads) {
  (void)num_threads;  // Unused parameter
  auto copy_op = [dst, src, size]() {
    // Call the assembly copy loop function directly (single-threaded).
    memory_copy_loop_asm(dst, src, size);
  };
  warmup_single(copy_op);
}


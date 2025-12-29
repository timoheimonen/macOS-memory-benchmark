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
#include "utils/benchmark.h"  // Includes definitions for assembly loops etc.
#include "warmup/warmup.h"
#include "warmup/warmup_internal.h"

// Warms up memory by reading from the buffer using multiple threads.
// 'buffer': Memory region to read from.
// 'size': Total size of the buffer in bytes.
// 'num_threads': Number of concurrent threads to use.
// 'dummy_checksum': Atomic variable to accumulate a dummy result (prevents optimization).
void warmup_read(void* buffer, size_t size, int num_threads, std::atomic<uint64_t>& dummy_checksum) {
  size_t warmup_size = calculate_warmup_size(size);
  auto read_chunk_op = [](char* chunk_start, char* /* src_chunk */, size_t chunk_size, 
                          std::atomic<uint64_t>* checksum) {
    // Call the assembly read loop function (defined elsewhere).
    uint64_t result = memory_read_loop_asm(chunk_start, chunk_size);
    // Atomically combine the local checksum with the global dummy value.
    // Using release memory order ensures proper visibility when threads complete.
    if (checksum) {
      checksum->fetch_xor(result, std::memory_order_release);
    }
  };
  warmup_parallel(buffer, size, num_threads, read_chunk_op, true, nullptr, &dummy_checksum, warmup_size);
}

// Warms up memory by writing to the buffer using multiple threads.
// 'buffer': Memory region to write to.
// 'size': Total size of the buffer in bytes.
// 'num_threads': Number of concurrent threads to use.
void warmup_write(void* buffer, size_t size, int num_threads) {
  size_t warmup_size = calculate_warmup_size(size);
  auto write_chunk_op = [](char* chunk_start, char* /* src_chunk */, size_t chunk_size, 
                           std::atomic<uint64_t>* /* checksum */) {
    memory_write_loop_asm(chunk_start, chunk_size);
  };
  warmup_parallel(buffer, size, num_threads, write_chunk_op, true, nullptr, nullptr, warmup_size);
}

// Warms up memory by copying data between buffers using multiple threads.
// 'dst': Destination memory region.
// 'src': Source memory region.
// 'size': Total size of data to copy in bytes.
// 'num_threads': Number of concurrent threads to use.
void warmup_copy(void* dst, void* src, size_t size, int num_threads) {
  size_t warmup_size = calculate_warmup_size(size);
  auto copy_chunk_op = [](char* dst_chunk, char* src_chunk, size_t chunk_size, 
                          std::atomic<uint64_t>* /* checksum */) {
    memory_copy_loop_asm(dst_chunk, src_chunk, chunk_size);
  };
  warmup_parallel(dst, size, num_threads, copy_chunk_op, true, src, nullptr, warmup_size);
}


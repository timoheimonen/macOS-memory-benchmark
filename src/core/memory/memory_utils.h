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
 * @file memory_utils.h
 * @brief Memory utility functions for buffer management and cache line alignment
 *
 * This file provides utility functions for memory operations commonly used in benchmarks:
 * - Cache line alignment functions to prevent false sharing in multi-threaded scenarios
 * - Pointer alignment utilities for optimal memory access patterns
 * - Latency chain setup for pointer-chasing memory latency tests
 * - Buffer initialization routines for benchmark preparation
 *
 * All inline alignment functions ensure 64-byte (cache line) boundaries to optimize
 * memory access and prevent performance degradation from false sharing.
 *
 * @note Cache line size is defined as 64 bytes (Constants::CACHE_LINE_SIZE_BYTES)
 */

#ifndef MEMORY_UTILS_H
#define MEMORY_UTILS_H

#include <cstddef>  // size_t
#include <cstdint>  // uintptr_t
#include "core/config/constants.h"

/**
 * @brief Align an offset to the next cache line boundary (rounds up)
 * @param offset The offset to align
 * @return The aligned offset (rounded up to next 64-byte boundary)
 * 
 * This function ensures that memory offsets are aligned to cache line boundaries
 * to prevent false sharing between threads accessing adjacent memory regions.
 */
inline size_t align_to_cache_line(size_t offset) {
  const size_t mask = Constants::CACHE_LINE_SIZE_BYTES - 1;
  return (offset + mask) & ~mask;
}

/**
 * @brief Align a pointer to the next cache line boundary (rounds up)
 * @param ptr The pointer to align
 * @return The aligned pointer (rounded up to next 64-byte boundary)
 * 
 * This function ensures that memory pointers are aligned to cache line boundaries
 * to prevent false sharing between threads accessing adjacent memory regions.
 */
inline void* align_ptr_to_cache_line(void* ptr) {
  uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  uintptr_t aligned_addr = align_to_cache_line(addr);
  return reinterpret_cast<void*>(aligned_addr);
}

/**
 * @brief Calculate the offset needed to align a pointer to cache line boundary
 * @param ptr The pointer to align
 * @return The number of bytes to add to ptr to reach the next cache line boundary
 */
inline size_t alignment_offset_to_cache_line(void* ptr) {
  uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  uintptr_t aligned_addr = align_to_cache_line(addr);
  return aligned_addr - addr;
}

// --- Memory Utility Functions ---
/**
 * @brief Prepare buffer for latency test by setting up pointer-chasing chain
 * @param buffer Pointer to the buffer to initialize
 * @param buffer_size Size of the buffer in bytes
 * @param stride Stride size in bytes between linked pointers
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on error
 *
 * Creates a linked list structure in memory where each pointer points to the
 * next location, enabling pointer-chasing latency measurements.
 */
int setup_latency_chain(void* buffer, size_t buffer_size, size_t stride);

/**
 * @brief Initialize data buffers with test data
 * @param src_buffer Pointer to source buffer
 * @param dst_buffer Pointer to destination buffer
 * @param buffer_size Size of buffers in bytes
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on error
 *
 * Fills source buffer with test data and zeros destination buffer.
 */
int initialize_buffers(void* src_buffer, void* dst_buffer, size_t buffer_size);

#endif // MEMORY_UTILS_H


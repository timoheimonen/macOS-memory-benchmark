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
 * @file warmup.h
 * @brief Memory warmup functions for benchmark preparation
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2025
 *
 * This header provides functions to warm up memory and caches before running
 * benchmarks. Warming up ensures consistent performance measurements by
 * pre-loading data into memory and caches.
 */
#ifndef WARMUP_H
#define WARMUP_H

#include <atomic>
#include <vector>
#include <cstddef>

// --- Basic Warmup Functions ---
/**
 * @brief Warms up memory by reading from the buffer using multiple threads.
 * @param buffer Pointer to the buffer to warm up
 * @param size Size of the buffer in bytes
 * @param num_threads Number of threads to use for warmup
 * @param dummy_checksum Atomic checksum accumulator (used to prevent optimization)
 */
void warmup_read(void* buffer, size_t size, int num_threads, std::atomic<uint64_t>& dummy_checksum);

/**
 * @brief Warms up memory by writing to the buffer using multiple threads.
 * @param buffer Pointer to the buffer to warm up
 * @param size Size of the buffer in bytes
 * @param num_threads Number of threads to use for warmup
 */
void warmup_write(void* buffer, size_t size, int num_threads);

/**
 * @brief Warms up memory by copying data between buffers using multiple threads.
 * @param dst Pointer to the destination buffer
 * @param src Pointer to the source buffer
 * @param size Size of the buffers in bytes
 * @param num_threads Number of threads to use for warmup
 */
void warmup_copy(void* dst, void* src, size_t size, int num_threads);

// --- Latency Warmup Functions ---
/**
 * @brief Warms up memory for latency test by page prefaulting (single thread).
 * @param buffer Pointer to the buffer to warm up
 * @param buffer_size Size of the buffer in bytes
 * @note Uses single-threaded page prefaulting to ensure pages are resident in memory
 */
void warmup_latency(void* buffer, size_t buffer_size);

/**
 * @brief Warms up memory for cache latency test by page prefaulting (single thread).
 * @param buffer Pointer to the buffer to warm up
 * @param buffer_size Size of the buffer in bytes
 * @note Uses single-threaded page prefaulting to ensure pages are resident in memory
 */
void warmup_cache_latency(void* buffer, size_t buffer_size);

// --- Cache Warmup Functions ---
/**
 * @brief Warms up cache bandwidth test by reading from the buffer.
 * @param src_buffer Pointer to the source buffer
 * @param size Size of the buffer in bytes
 * @param num_threads Number of threads to use for warmup
 * @param dummy_checksum Atomic checksum accumulator (used to prevent optimization)
 */
void warmup_cache_read(void* src_buffer, size_t size, int num_threads, std::atomic<uint64_t>& dummy_checksum);

/**
 * @brief Warms up cache bandwidth test by writing to the buffer.
 * @param dst_buffer Pointer to the destination buffer
 * @param size Size of the buffer in bytes
 * @param num_threads Number of threads to use for warmup
 */
void warmup_cache_write(void* dst_buffer, size_t size, int num_threads);

/**
 * @brief Warms up cache bandwidth test by copying data between buffers.
 * @param dst Pointer to the destination buffer
 * @param src Pointer to the source buffer
 * @param size Size of the buffers in bytes
 * @param num_threads Number of threads to use for warmup
 */
void warmup_cache_copy(void* dst, void* src, size_t size, int num_threads);

// --- Pattern-Specific Warmup Functions ---
/**
 * @brief Warms up memory by reading from the buffer using strided access pattern.
 * @param buffer Pointer to the buffer to warm up
 * @param size Size of the buffer in bytes
 * @param stride Stride size in bytes between accesses
 * @param num_threads Number of threads to use for warmup
 * @param dummy_checksum Atomic checksum accumulator (used to prevent optimization)
 */
void warmup_read_strided(void* buffer, size_t size, size_t stride, int num_threads, std::atomic<uint64_t>& dummy_checksum);

/**
 * @brief Warms up memory by writing to the buffer using strided access pattern.
 * @param buffer Pointer to the buffer to warm up
 * @param size Size of the buffer in bytes
 * @param stride Stride size in bytes between accesses
 * @param num_threads Number of threads to use for warmup
 */
void warmup_write_strided(void* buffer, size_t size, size_t stride, int num_threads);

/**
 * @brief Warms up memory by copying data between buffers using strided access pattern.
 * @param dst Pointer to the destination buffer
 * @param src Pointer to the source buffer
 * @param size Size of the buffers in bytes
 * @param stride Stride size in bytes between accesses
 * @param num_threads Number of threads to use for warmup
 */
void warmup_copy_strided(void* dst, void* src, size_t size, size_t stride, int num_threads);

/**
 * @brief Warms up memory by reading from the buffer using random access pattern.
 * @param buffer Pointer to the buffer to warm up
 * @param indices Vector of byte offsets for random access
 * @param num_threads Number of threads to use for warmup
 * @param dummy_checksum Atomic checksum accumulator (used to prevent optimization)
 */
void warmup_read_random(void* buffer, const std::vector<size_t>& indices, int num_threads, std::atomic<uint64_t>& dummy_checksum);

/**
 * @brief Warms up memory by writing to the buffer using random access pattern.
 * @param buffer Pointer to the buffer to warm up
 * @param indices Vector of byte offsets for random access
 * @param num_threads Number of threads to use for warmup
 */
void warmup_write_random(void* buffer, const std::vector<size_t>& indices, int num_threads);

/**
 * @brief Warms up memory by copying data between buffers using random access pattern.
 * @param dst Pointer to the destination buffer
 * @param src Pointer to the source buffer
 * @param indices Vector of byte offsets for random access
 * @param num_threads Number of threads to use for warmup
 */
void warmup_copy_random(void* dst, void* src, const std::vector<size_t>& indices, int num_threads);

#endif // WARMUP_H


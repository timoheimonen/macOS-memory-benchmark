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
#include <cstddef>
#include <cstdint>
#include <vector>

struct PatternRandomWorkerIndices;
struct PatternWorkPlan;

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
 * @brief Warms up finalized strided worker ranges with the measured read kernel.
 * @param buffer Pointer to the buffer to warm up
 * @param plan Validated work plan whose worker partitions and phase period are used
 * @param dummy_checksum Atomic checksum accumulator (used to prevent optimization)
 * @pre `plan` has measured status and contains finalized worker ranges
 */
void warmup_read_strided(void* buffer, const PatternWorkPlan& plan,
                         std::atomic<uint64_t>& dummy_checksum);

/**
 * @brief Warms up finalized strided worker ranges with the measured write kernel.
 * @param buffer Pointer to the buffer to warm up
 * @param plan Validated work plan whose worker partitions and phase period are used
 * @pre `plan` has measured status and contains finalized worker ranges
 */
void warmup_write_strided(void* buffer, const PatternWorkPlan& plan);

/**
 * @brief Warms up finalized strided worker ranges with the measured copy kernel.
 * @param dst Pointer to the destination buffer
 * @param src Pointer to the source buffer
 * @param plan Validated work plan whose worker partitions and phase period are used
 * @pre `plan` has measured status and contains finalized worker ranges
 */
void warmup_copy_strided(void* dst, void* src, const PatternWorkPlan& plan);

/**
 * @brief Warms up finalized random worker chunks with the measured read kernel.
 * @param buffer Pointer to the buffer to warm up
 * @param worker_indices Validated worker partitions with chunk-relative offsets
 * @param dummy_checksum Atomic checksum accumulator (used to prevent optimization)
 * @pre Every worker offset is valid within its finalized chunk
 */
void warmup_read_random(
    void* buffer,
    const std::vector<PatternRandomWorkerIndices>& worker_indices,
    std::atomic<uint64_t>& dummy_checksum);

/**
 * @brief Warms up finalized random worker chunks with the measured write kernel.
 * @param buffer Pointer to the buffer to warm up
 * @param worker_indices Validated worker partitions with chunk-relative offsets
 * @pre Every worker offset is valid within its finalized chunk
 */
void warmup_write_random(
    void* buffer,
    const std::vector<PatternRandomWorkerIndices>& worker_indices);

/**
 * @brief Warms up finalized random worker chunks with the measured copy kernel.
 * @param dst Pointer to the destination buffer
 * @param src Pointer to the source buffer
 * @param worker_indices Validated worker partitions with chunk-relative offsets
 * @pre Every worker offset is valid within its finalized chunk
 */
void warmup_copy_random(
    void* dst, void* src,
    const std::vector<PatternRandomWorkerIndices>& worker_indices);

#endif // WARMUP_H

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
#ifndef WARMUP_H
#define WARMUP_H

#include <atomic>
#include <vector>
#include <cstddef>

// --- Basic Warmup Functions ---
// Warms up memory by reading from the buffer using multiple threads.
void warmup_read(void* buffer, size_t size, int num_threads, std::atomic<uint64_t>& dummy_checksum);

// Warms up memory by writing to the buffer using multiple threads.
void warmup_write(void* buffer, size_t size, int num_threads);

// Warms up memory by copying data between buffers using multiple threads.
void warmup_copy(void* dst, void* src, size_t size, int num_threads);

// --- Latency Warmup Functions ---
// Warms up memory for latency test by page prefaulting (single thread).
void warmup_latency(void* buffer, size_t buffer_size);

// Warms up memory for cache latency test by page prefaulting (single thread).
void warmup_cache_latency(void* buffer, size_t buffer_size);

// --- Cache Warmup Functions ---
// Warms up cache bandwidth test by reading from the buffer (single thread).
void warmup_cache_read(void* src_buffer, size_t size, int num_threads, std::atomic<uint64_t>& dummy_checksum);

// Warms up cache bandwidth test by writing to the buffer (single thread).
void warmup_cache_write(void* dst_buffer, size_t size, int num_threads);

// Warms up cache bandwidth test by copying data between buffers (single thread).
void warmup_cache_copy(void* dst, void* src, size_t size, int num_threads);

// --- Pattern-Specific Warmup Functions ---
// Warms up memory by reading from the buffer using strided access pattern.
void warmup_read_strided(void* buffer, size_t size, size_t stride, int num_threads, std::atomic<uint64_t>& dummy_checksum);

// Warms up memory by writing to the buffer using strided access pattern.
void warmup_write_strided(void* buffer, size_t size, size_t stride, int num_threads);

// Warms up memory by copying data between buffers using strided access pattern.
void warmup_copy_strided(void* dst, void* src, size_t size, size_t stride, int num_threads);

// Warms up memory by reading from the buffer using random access pattern.
void warmup_read_random(void* buffer, const std::vector<size_t>& indices, int num_threads, std::atomic<uint64_t>& dummy_checksum);

// Warms up memory by writing to the buffer using random access pattern.
void warmup_write_random(void* buffer, const std::vector<size_t>& indices, int num_threads);

// Warms up memory by copying data between buffers using random access pattern.
void warmup_copy_random(void* dst, void* src, const std::vector<size_t>& indices, int num_threads);

#endif // WARMUP_H


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
 * @file benchmark_tests.h
 * @brief Benchmark test execution functions
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2025
 *
 * This header provides functions to execute various memory benchmark tests
 * including read, write, copy, and latency measurements.
 */
#ifndef BENCHMARK_TESTS_H
#define BENCHMARK_TESTS_H

#include <cstddef>  // size_t
#include <cstdint>  // uint64_t
#include <atomic>   // std::atomic
#include <vector>   // std::vector

// Forward declaration
struct HighResTimer;

// --- Benchmark Test Functions ---
/**
 * @brief Run read benchmark test
 * @param buffer Pointer to buffer to read from
 * @param size Size of buffer in bytes
 * @param iterations Number of iterations to run
 * @param num_threads Number of threads to use
 * @param checksum Reference to atomic checksum accumulator
 * @param timer Reference to high-resolution timer
 * @return Total elapsed time in seconds
 */
double run_read_test(void* buffer, size_t size, int iterations, int num_threads, std::atomic<uint64_t>& checksum, HighResTimer& timer);

/**
 * @brief Run write benchmark test
 * @param buffer Pointer to buffer to write to
 * @param size Size of buffer in bytes
 * @param iterations Number of iterations to run
 * @param num_threads Number of threads to use
 * @param timer Reference to high-resolution timer
 * @return Total elapsed time in seconds
 */
double run_write_test(void* buffer, size_t size, int iterations, int num_threads, HighResTimer& timer);

/**
 * @brief Run copy benchmark test
 * @param dst Pointer to destination buffer
 * @param src Pointer to source buffer
 * @param size Size of buffers in bytes
 * @param iterations Number of iterations to run
 * @param num_threads Number of threads to use
 * @param timer Reference to high-resolution timer
 * @return Total elapsed time in seconds
 */
double run_copy_test(void* dst, void* src, size_t size, int iterations, int num_threads, HighResTimer& timer);

/**
 * @brief Run latency benchmark test
 * @param buffer Pointer to latency test buffer (must be initialized with setup_latency_chain)
 * @param num_accesses Number of pointer-chasing accesses to perform
 * @param timer Reference to high-resolution timer
 * @param latency_samples Optional pointer to vector to store individual latency samples
 * @param sample_count Number of samples to collect (0 = collect all)
 * @return Average latency per access in nanoseconds
 */
double run_latency_test(void* buffer, size_t num_accesses, HighResTimer& timer, std::vector<double>* latency_samples = nullptr, int sample_count = 0);

/**
 * @brief Run cache latency benchmark test
 * @param buffer Pointer to cache latency test buffer (must be initialized with setup_latency_chain)
 * @param buffer_size Size of buffer in bytes
 * @param num_accesses Number of pointer-chasing accesses to perform
 * @param timer Reference to high-resolution timer
 * @param latency_samples Optional pointer to vector to store individual latency samples
 * @param sample_count Number of samples to collect (0 = collect all)
 * @return Average latency per access in nanoseconds
 */
double run_cache_latency_test(void* buffer, size_t buffer_size, size_t num_accesses, HighResTimer& timer, std::vector<double>* latency_samples = nullptr, int sample_count = 0);

#endif // BENCHMARK_TESTS_H


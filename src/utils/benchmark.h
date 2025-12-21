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
 * @file benchmark.h
 * @brief Main benchmark utilities and function declarations
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2025
 *
 * This header provides the main benchmark utilities including high-resolution
 * timing, system information, memory utilities, warmup functions, benchmark
 * test functions, and assembly function declarations.
 */
#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <vector>
#include <chrono>
#include <numeric>
#include <cstdlib>
#include <cstddef>
#include <random>
#include <algorithm>
#include <cstdint>
#include <thread>
#include <functional>
#include <atomic>
#include <string>
#include <stdexcept>
#include <limits>
#include <iostream> // For error/output streams
#include <iomanip> // For output formatting

// macOS specific: High-resolution timer
#include <mach/mach_time.h>

// --- Version Information ---
/**
 * @def SOFTVERSION
 * @brief Software version number
 */
#define SOFTVERSION 0.50f

// --- High-resolution timer helper ---
/**
 * @struct HighResTimer
 * @brief High-resolution timer using macOS mach_absolute_time()
 *
 * Provides nanosecond-precision timing using the macOS mach timing API.
 * Automatically handles timebase conversion for accurate measurements.
 */
struct HighResTimer {
    uint64_t start_ticks = 0;           ///< Ticks at timer start
    mach_timebase_info_data_t timebase_info; ///< Timebase info for conversion

    /**
     * @brief Constructor - initializes timebase info
     */
    HighResTimer();
    
    /**
     * @brief Start the timer
     */
    void start();
    
    /**
     * @brief Stop timer and return elapsed time in seconds
     * @return Elapsed time in seconds
     */
    double stop();
    
    /**
     * @brief Stop timer and return elapsed time in nanoseconds
     * @return Elapsed time in nanoseconds
     */
    double stop_ns();
};

// Forward declarations
struct BenchmarkConfig;
struct BenchmarkStatistics;

// --- System Info Functions (system_info.cpp) ---
/**
 * @brief Get number of performance cores
 * @return Number of performance cores
 */
int get_performance_cores();

/**
 * @brief Get number of efficiency cores
 * @return Number of efficiency cores
 */
int get_efficiency_cores();

/**
 * @brief Get total logical core count
 * @return Total number of logical cores
 */
int get_total_logical_cores();

/**
 * @brief Get processor model name
 * @return Processor model name as string
 */
std::string get_processor_name();

/**
 * @brief Get available system memory in MB
 * @return Available system memory in megabytes
 */
unsigned long get_available_memory_mb();

/**
 * @brief Get L1 data cache size for performance cores
 * @return L1 data cache size in bytes
 */
size_t get_l1_cache_size();

/**
 * @brief Get L2 cache size for performance cores
 * @return L2 cache size in bytes
 */
size_t get_l2_cache_size();

// --- Memory Utility Functions (memory_utils.cpp) ---
/**
 * @brief Prepare buffer for latency test by setting up pointer-chasing chain
 * @param buffer Pointer to the buffer to initialize
 * @param buffer_size Size of the buffer in bytes
 * @param stride Stride size in bytes between linked pointers
 *
 * Creates a linked list structure in memory where each pointer points to the
 * next location, enabling pointer-chasing latency measurements.
 */
void setup_latency_chain(void* buffer, size_t buffer_size, size_t stride);

/**
 * @brief Initialize data buffers with test data
 * @param src_buffer Pointer to source buffer
 * @param dst_buffer Pointer to destination buffer
 * @param buffer_size Size of buffers in bytes
 *
 * Fills source buffer with test data and zeros destination buffer.
 */
void initialize_buffers(void* src_buffer, void* dst_buffer, size_t buffer_size);

// --- Warmup Functions (warmup/) ---
#include "warmup/warmup.h"

// --- Benchmark Test Functions (benchmark_tests.cpp) ---
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

// --- Utility Functions (utils.cpp) ---
/**
 * @brief Join all threads in vector and clear it
 * @param threads Reference to vector of threads to join
 */
void join_threads(std::vector<std::thread>& threads);

/**
 * @brief Show progress indicator (spinner)
 *
 * Displays a simple spinner animation to indicate progress.
 */
void show_progress();

// --- Output/Printing Functions (output_printer.cpp) ---
void print_usage(const char* prog_name); // Print command-line usage instructions
void print_configuration(size_t buffer_size, size_t buffer_size_mb, int iterations, int loop_count, bool use_non_cacheable, const std::string& cpu_name, int perf_cores, int eff_cores, int num_threads); // Print benchmark setup details
void print_results(int loop, size_t buffer_size, size_t buffer_size_mb, int iterations, int num_threads,
    double read_bw_gb_s, double total_read_time,
    double write_bw_gb_s, double total_write_time,
    double copy_bw_gb_s, double total_copy_time,
    double l1_latency_ns, double l2_latency_ns,
    size_t l1_buffer_size, size_t l2_buffer_size,
    double l1_read_bw_gb_s, double l1_write_bw_gb_s, double l1_copy_bw_gb_s,
    double l2_read_bw_gb_s, double l2_write_bw_gb_s, double l2_copy_bw_gb_s,
    double average_latency_ns, double total_lat_time_ns,
    bool use_custom_cache_size, double custom_latency_ns, size_t custom_buffer_size,
    double custom_read_bw_gb_s, double custom_write_bw_gb_s, double custom_copy_bw_gb_s); // Print results for one loop
void print_cache_info(size_t l1_cache_size, size_t l2_cache_size, bool use_custom_cache_size, size_t custom_cache_size_bytes); // Print cache size information

// --- Statistics Functions (statistics.cpp) ---
void print_statistics(int loop_count,
                      const std::vector<double>& all_read_bw,
                      const std::vector<double>& all_write_bw,
                      const std::vector<double>& all_copy_bw,
                      const std::vector<double>& all_l1_latency,
                      const std::vector<double>& all_l2_latency,
                      const std::vector<double>& all_l1_read_bw,
                      const std::vector<double>& all_l1_write_bw,
                      const std::vector<double>& all_l1_copy_bw,
                      const std::vector<double>& all_l2_read_bw,
                      const std::vector<double>& all_l2_write_bw,
                      const std::vector<double>& all_l2_copy_bw,
                      const std::vector<double>& all_main_mem_latency,
                      bool use_custom_cache_size,
                      const std::vector<double>& all_custom_latency,
                      const std::vector<double>& all_custom_read_bw,
                      const std::vector<double>& all_custom_write_bw,
                      const std::vector<double>& all_custom_copy_bw,
                      const std::vector<double>& all_main_mem_latency_samples,
                      const std::vector<double>& all_l1_latency_samples,
                      const std::vector<double>& all_l2_latency_samples,
                      const std::vector<double>& all_custom_latency_samples); // Print summary statistics after all loops


// --- Assembly Function Declarations ---
/**
 * @defgroup asm_functions Assembly Functions
 * @brief Optimized assembly functions for memory operations
 * @{
 */

extern "C" {
    /**
     * @brief Optimized memory copy loop (assembly)
     * @param dst Destination buffer pointer
     * @param src Source buffer pointer
     * @param byteCount Number of bytes to copy
     */
    void memory_copy_loop_asm(void* dst, const void* src, size_t byteCount);
    
    /**
     * @brief Optimized memory read loop (assembly)
     * @param src Source buffer pointer
     * @param byteCount Number of bytes to read
     * @return Checksum of read data
     */
    uint64_t memory_read_loop_asm(const void* src, size_t byteCount);
    
    /**
     * @brief Optimized memory write loop (assembly)
     * @param dst Destination buffer pointer
     * @param byteCount Number of bytes to write
     */
    void memory_write_loop_asm(void* dst, size_t byteCount);
    
    /**
     * @brief Pointer chasing loop for latency measurement (assembly)
     * @param start_pointer Starting pointer for pointer-chasing chain
     * @param count Number of pointer-chasing accesses to perform
     */
    void memory_latency_chase_asm(uintptr_t* start_pointer, size_t count);
    
    // Pattern-specific assembly functions
    // Reverse sequential
    /**
     * @brief Optimized reverse sequential memory read loop (assembly)
     * @param src Source buffer pointer
     * @param byteCount Number of bytes to read
     * @return Checksum of read data
     */
    uint64_t memory_read_reverse_loop_asm(const void* src, size_t byteCount);
    
    /**
     * @brief Optimized reverse sequential memory write loop (assembly)
     * @param dst Destination buffer pointer
     * @param byteCount Number of bytes to write
     */
    void memory_write_reverse_loop_asm(void* dst, size_t byteCount);
    
    /**
     * @brief Optimized reverse sequential memory copy loop (assembly)
     * @param dst Destination buffer pointer
     * @param src Source buffer pointer
     * @param byteCount Number of bytes to copy
     */
    void memory_copy_reverse_loop_asm(void* dst, const void* src, size_t byteCount);
    
    // Strided access
    /**
     * @brief Optimized strided memory read loop (assembly)
     * @param src Source buffer pointer
     * @param byteCount Total buffer size in bytes
     * @param stride Stride size in bytes between accesses
     * @return Checksum of read data
     */
    uint64_t memory_read_strided_loop_asm(const void* src, size_t byteCount, size_t stride);
    
    /**
     * @brief Optimized strided memory write loop (assembly)
     * @param dst Destination buffer pointer
     * @param byteCount Total buffer size in bytes
     * @param stride Stride size in bytes between accesses
     */
    void memory_write_strided_loop_asm(void* dst, size_t byteCount, size_t stride);
    
    /**
     * @brief Optimized strided memory copy loop (assembly)
     * @param dst Destination buffer pointer
     * @param src Source buffer pointer
     * @param byteCount Total buffer size in bytes
     * @param stride Stride size in bytes between accesses
     */
    void memory_copy_strided_loop_asm(void* dst, const void* src, size_t byteCount, size_t stride);
    
    // Random access
    /**
     * @brief Optimized random access memory read loop (assembly)
     * @param src Source buffer pointer
     * @param indices Array of byte offsets for random access
     * @param num_accesses Number of random accesses to perform
     * @return Checksum of read data
     */
    uint64_t memory_read_random_loop_asm(const void* src, const size_t* indices, size_t num_accesses);
    
    /**
     * @brief Optimized random access memory write loop (assembly)
     * @param dst Destination buffer pointer
     * @param indices Array of byte offsets for random access
     * @param num_accesses Number of random accesses to perform
     */
    void memory_write_random_loop_asm(void* dst, const size_t* indices, size_t num_accesses);
    
    /**
     * @brief Optimized random access memory copy loop (assembly)
     * @param dst Destination buffer pointer
     * @param src Source buffer pointer
     * @param indices Array of byte offsets for random access
     * @param num_accesses Number of random accesses to perform
     */
    void memory_copy_random_loop_asm(void* dst, const void* src, const size_t* indices, size_t num_accesses);
}
/** @} */

#endif // BENCHMARK_H
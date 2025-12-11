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
#define SOFTVERSION 0.41f // Software version

// --- High-resolution timer helper ---
struct HighResTimer {
    uint64_t start_ticks = 0;           // Ticks at timer start
    mach_timebase_info_data_t timebase_info; // Timebase info for conversion

    HighResTimer(); // Constructor
    void start();   // Start the timer
    double stop();  // Stop timer, return seconds
    double stop_ns(); // Stop timer, return nanoseconds
};

// --- System Info Functions (system_info.cpp) ---
int get_performance_cores();          // Get number of performance cores
int get_efficiency_cores();           // Get number of efficiency cores
int get_total_logical_cores();        // Get total logical core count
std::string get_processor_name();     // Get processor model name
unsigned long get_available_memory_mb(); // Get available system memory in MB
size_t get_l1_cache_size();           // Get L1 data cache size for performance cores (bytes)
size_t get_l2_cache_size();           // Get L2 cache size for performance cores (bytes)

// --- Memory Utility Functions (memory_utils.cpp) ---
void setup_latency_chain(void* buffer, size_t buffer_size, size_t stride); // Prepare buffer for latency test
void initialize_buffers(void* src_buffer, void* dst_buffer, size_t buffer_size); // Initialize data buffers

// --- Warmup Functions (warmup.cpp) ---
void warmup_read(void* buffer, size_t size, int num_threads, std::atomic<uint64_t>& dummy_checksum); // Warmup for read tests
void warmup_write(void* buffer, size_t size, int num_threads); // Warmup for write tests
void warmup_copy(void* dst, void* src, size_t size, int num_threads); // Warmup for copy tests
void warmup_latency(void* buffer, size_t num_accesses); // Warmup for latency tests
void warmup_cache_latency(void* buffer, size_t num_accesses); // Warmup for cache latency tests

// --- Benchmark Test Functions (benchmark_tests.cpp) ---
double run_read_test(void* buffer, size_t size, int iterations, int num_threads, std::atomic<uint64_t>& checksum, HighResTimer& timer); // Run read benchmark
double run_write_test(void* buffer, size_t size, int iterations, int num_threads, HighResTimer& timer); // Run write benchmark
double run_copy_test(void* dst, void* src, size_t size, int iterations, int num_threads, HighResTimer& timer); // Run copy benchmark
double run_latency_test(void* buffer, size_t num_accesses, HighResTimer& timer); // Run latency benchmark
double run_cache_latency_test(void* buffer, size_t buffer_size, size_t num_accesses, HighResTimer& timer); // Run cache latency benchmark

// --- Utility Functions (utils.cpp) ---
void print_usage(const char* prog_name); // Print command-line usage instructions
void print_configuration(size_t buffer_size, size_t buffer_size_mb, int iterations, int loop_count, const std::string& cpu_name, int perf_cores, int eff_cores, int num_threads); // Print benchmark setup details
void print_results(int loop, size_t buffer_size, size_t buffer_size_mb, int iterations, int num_threads,
    double read_bw_gb_s, double total_read_time,
    double write_bw_gb_s, double total_write_time,
    double copy_bw_gb_s, double total_copy_time,
    double l1_latency_ns, double l2_latency_ns,
    size_t l1_buffer_size, size_t l2_buffer_size,
    double average_latency_ns, double total_lat_time_ns); // Print results for one loop
void print_statistics(int loop_count,
                      const std::vector<double>& all_read_bw,
                      const std::vector<double>& all_write_bw,
                      const std::vector<double>& all_copy_bw,
                      const std::vector<double>& all_l1_latency,
                      const std::vector<double>& all_l2_latency,
                      const std::vector<double>& all_main_mem_latency); // Print summary statistics after all loops
void print_cache_info(size_t l1_cache_size, size_t l2_cache_size); // Print cache size information


// --- Assembly Function Declarations ---
extern "C" {
    // Optimized memory copy loop (assembly)
    void memory_copy_loop_asm(void* dst, const void* src, size_t byteCount);
    // Optimized memory read loop (assembly), returns checksum
    uint64_t memory_read_loop_asm(const void* src, size_t byteCount);
    // Optimized memory write loop (assembly)
    void memory_write_loop_asm(void* dst, size_t byteCount);
    // Pointer chasing loop for latency measurement (assembly)
    void memory_latency_chase_asm(uintptr_t* start_pointer, size_t count);
}

#endif // BENCHMARK_H
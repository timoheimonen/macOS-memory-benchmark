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
 * @file output_printer.h
 * @brief Console output and printing functions
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2025
 *
 * This header provides functions for printing benchmark configuration, results,
 * and usage information to the console.
 */
#ifndef OUTPUT_PRINTER_H
#define OUTPUT_PRINTER_H

#include <cstddef>  // size_t
#include <string>   // std::string

// --- Output/Printing Functions ---
/**
 * @brief Print command-line usage instructions
 * @param prog_name Program name (typically argv[0])
 */
void print_usage(const char* prog_name);

/**
 * @brief Print benchmark setup details
 * @param buffer_size Buffer size in bytes
 * @param buffer_size_mb Buffer size in megabytes
 * @param iterations Number of iterations
 * @param loop_count Number of loops
 * @param use_non_cacheable Whether non-cacheable memory is used
 * @param cpu_name CPU name
 * @param perf_cores Number of performance cores
 * @param eff_cores Number of efficiency cores
 * @param num_threads Number of threads
 * @param only_bandwidth Whether only bandwidth tests are run
 * @param only_latency Whether only latency tests are run
 * @param run_patterns Whether pattern benchmarks are run (bandwidth-only, uses 2x buffers)
 */
void print_configuration(size_t buffer_size, size_t buffer_size_mb, int iterations, int loop_count, bool use_non_cacheable, const std::string& cpu_name, int perf_cores, int eff_cores, int num_threads, bool only_bandwidth, bool only_latency, bool run_patterns);

/**
 * @brief Print results for one loop
 * @param loop Loop number
 * @param buffer_size Buffer size in bytes
 * @param buffer_size_mb Buffer size in megabytes
 * @param iterations Number of iterations
 * @param num_threads Number of threads
 * @param read_bw_gb_s Read bandwidth in GB/s
 * @param total_read_time Total read time in seconds
 * @param write_bw_gb_s Write bandwidth in GB/s
 * @param total_write_time Total write time in seconds
 * @param copy_bw_gb_s Copy bandwidth in GB/s
 * @param total_copy_time Total copy time in seconds
 * @param l1_latency_ns L1 cache latency in nanoseconds
 * @param l2_latency_ns L2 cache latency in nanoseconds
 * @param l1_buffer_size L1 buffer size in bytes
 * @param l2_buffer_size L2 buffer size in bytes
 * @param l1_read_bw_gb_s L1 read bandwidth in GB/s
 * @param l1_write_bw_gb_s L1 write bandwidth in GB/s
 * @param l1_copy_bw_gb_s L1 copy bandwidth in GB/s
 * @param l2_read_bw_gb_s L2 read bandwidth in GB/s
 * @param l2_write_bw_gb_s L2 write bandwidth in GB/s
 * @param l2_copy_bw_gb_s L2 copy bandwidth in GB/s
 * @param average_latency_ns Average latency in nanoseconds
 * @param total_lat_time_ns Total latency time in nanoseconds
 * @param use_custom_cache_size Whether custom cache size is used
 * @param custom_latency_ns Custom cache latency in nanoseconds
 * @param custom_buffer_size Custom buffer size in bytes
 * @param custom_read_bw_gb_s Custom read bandwidth in GB/s
 * @param custom_write_bw_gb_s Custom write bandwidth in GB/s
 * @param custom_copy_bw_gb_s Custom copy bandwidth in GB/s
 * @param user_specified_threads Whether user specified thread count
 * @param only_bandwidth Whether only bandwidth tests are run
 * @param only_latency Whether only latency tests are run
 */
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
    double custom_read_bw_gb_s, double custom_write_bw_gb_s, double custom_copy_bw_gb_s,
    bool user_specified_threads, bool only_bandwidth, bool only_latency);

/**
 * @brief Print cache size information
 * @param l1_cache_size L1 cache size in bytes
 * @param l2_cache_size L2 cache size in bytes
 * @param use_custom_cache_size Whether custom cache size is used
 * @param custom_cache_size_bytes Custom cache size in bytes
 */
void print_cache_info(size_t l1_cache_size, size_t l2_cache_size, bool use_custom_cache_size, size_t custom_cache_size_bytes);

#endif // OUTPUT_PRINTER_H


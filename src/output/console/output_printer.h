// Copyright 2026 Timo Heimonen <timo.heimonen@proton.me>
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
 * @date 2026
 *
 * This header provides functions for printing benchmark configuration, results,
 * and usage information to the console.
 */
#ifndef OUTPUT_PRINTER_H
#define OUTPUT_PRINTER_H

#include <cstddef>  // size_t
#include <string>   // std::string

struct BenchmarkConfig;
struct BenchmarkResults;

// --- Output/Printing Functions ---
/**
 * @brief Print command-line usage instructions
 * @param prog_name Program name (typically argv[0])
 */
void print_usage(const char* prog_name);

/**
 * @brief Print help text to stdout (for -h/--help)
 * @param prog_name Program name (typically argv[0])
 */
void print_help(const char* prog_name);

/**
 * @brief Print benchmark setup details
 * @param buffer_size Buffer size in bytes
 * @param buffer_size_mb Buffer size in megabytes
 * @param total_allocation_bytes Peak concurrently allocated bytes across enabled benchmark phases
 * @param iterations Number of iterations
 * @param loop_count Number of loops
 * @param use_non_cacheable Whether best-effort cache-discouraging allocation hints were requested
 * @param latency_stride_bytes Stride used by latency pointer chains
 * @param latency_chain_mode_name Chain construction mode used by latency pointer chains
 * @param latency_tlb_locality_bytes TLB-locality window for latency chains (0 = global random)
 * @param cpu_name CPU name
 * @param perf_cores Number of performance cores
 * @param eff_cores Number of efficiency cores
 * @param num_threads Number of threads
 * @param only_bandwidth Whether only bandwidth tests are run
 * @param only_latency Whether only latency tests are run
 * @param run_patterns Whether pattern benchmarks are run (bandwidth-only, uses 2x buffers)
 * @param user_specified_iterations Whether --iterations explicitly disables pattern auto-calibration
 */
void print_configuration(size_t buffer_size, size_t buffer_size_mb, size_t total_allocation_bytes, int iterations,
                          int loop_count, bool use_non_cacheable, size_t latency_stride_bytes,
                          const std::string& latency_chain_mode_name, size_t latency_tlb_locality_bytes,
                          const std::string& cpu_name, int perf_cores, int eff_cores, int num_threads,
                          bool only_bandwidth, bool only_latency, bool run_patterns,
                          bool user_specified_iterations);

/**
 * @brief Print results for one loop
 * @param loop Zero-based loop index
 * @param config Configuration used for the loop and output selection
 * @param results Status-aware measurements produced by the loop
 */
void print_results(int loop, const BenchmarkConfig& config, const BenchmarkResults& results);

/**
 * @brief Print cache size information
 * @param l1_cache_size L1 cache size in bytes
 * @param l2_cache_size L2 cache size in bytes
 * @param use_custom_cache_size Whether custom cache size is used
 * @param custom_cache_size_bytes Custom cache size in bytes
 */
void print_cache_info(size_t l1_cache_size, size_t l2_cache_size, bool use_custom_cache_size, size_t custom_cache_size_bytes);

#endif // OUTPUT_PRINTER_H

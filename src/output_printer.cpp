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
#include <iomanip>    // Required for std::setprecision, std::fixed (output formatting)
#include <iostream>   // Required for std::cout, std::cerr
#include <sstream>    // Required for std::ostringstream

#include "benchmark.h"  // Include common definitions/constants (e.g., SOFTVERSION)
#include "constants.h"  // Include constants for default values
#include "messages.h"   // Include centralized messages

// --- Helper function to print usage instructions ---
// Displays how to use the program via command-line arguments.
// 'prog_name': The name of the executable (typically argv[0]).
void print_usage(const char *prog_name) {
  // Output usage syntax, version, options, and an example to standard error.
  std::ostringstream version_str;
  version_str << SOFTVERSION;
  std::cerr << Messages::usage_header(version_str.str())
            << Messages::usage_options(prog_name)
            << Messages::usage_example(prog_name);
}

// --- Print Configuration ---
// Outputs the configuration parameters the benchmark will run with.
// 'buffer_size': The final size (in bytes) of each test buffer after memory checks.
// 'buffer_size_mb': The requested/capped buffer size in Megabytes.
// 'iterations': Number of iterations per bandwidth test per loop.
// 'loop_count': How many times the entire set of tests will be repeated.
// 'cpu_name': Detected processor name string.
// 'perf_cores': Number of detected performance cores.
// 'eff_cores': Number of detected efficiency cores.
// 'num_threads': Total number of threads used for bandwidth tests (usually equals total cores).
void print_configuration(size_t buffer_size, size_t buffer_size_mb, int iterations, int loop_count,
                         const std::string &cpu_name, int perf_cores, int eff_cores, int num_threads) {
  // Print benchmark header and copyright/license info.
  std::ostringstream version_str;
  version_str << SOFTVERSION;
  std::cout << Messages::config_header(version_str.str()) << std::endl;
  std::cout << Messages::config_copyright() << std::endl;
  std::cout << Messages::config_license() << std::endl;
  // Display buffer sizes (actual MiB and requested/capped MB).
  std::cout << Messages::config_buffer_size(buffer_size / (1024.0 * 1024.0), buffer_size_mb) << std::endl;
  // Display total approximate memory allocated for the three buffers.
  std::cout << Messages::config_total_allocation(3.0 * buffer_size / (1024.0 * 1024.0)) << std::endl;
  // Display test repetition counts.
  std::cout << Messages::config_iterations(iterations) << std::endl;
  std::cout << Messages::config_loop_count(loop_count) << std::endl;
  // Display CPU information if successfully retrieved.
  if (!cpu_name.empty()) {
    std::cout << Messages::config_processor_name(cpu_name) << std::endl;
  } else {
    std::cout << Messages::config_processor_name_error() << std::endl;
  }
  // Display P-core and E-core counts if available.
  if (perf_cores > 0 || eff_cores > 0) {
    std::cout << Messages::config_performance_cores(perf_cores) << std::endl;
    std::cout << Messages::config_efficiency_cores(eff_cores) << std::endl;
  }
  // Display total cores detected and threads used for bandwidth tests.
  std::cout << Messages::config_total_cores(num_threads) << std::endl;
}

// --- Print Results for a single loop ---
// Outputs the performance results measured during one complete benchmark loop.
// 'loop': Index of the loop being reported (0-based).
// 'buffer_size', 'buffer_size_mb', 'iterations', 'num_threads': Config parameters for context.
// 'read_bw_gb_s', 'total_read_time': Results from the read test.
// 'write_bw_gb_s', 'total_write_time': Results from the write test.
// 'copy_bw_gb_s', 'total_copy_time': Results from the copy test.
// 'l1_latency_ns', 'l2_latency_ns': Results from cache latency tests.
// 'l1_buffer_size', 'l2_buffer_size': Buffer sizes used for cache tests.
// 'l1_read_bw_gb_s', 'l1_write_bw_gb_s', 'l1_copy_bw_gb_s': L1 cache bandwidth results.
// 'l2_read_bw_gb_s', 'l2_write_bw_gb_s', 'l2_copy_bw_gb_s': L2 cache bandwidth results.
// 'average_latency_ns', 'total_lat_time_ns': Results from the main memory latency test.
// 'use_custom_cache_size': Flag indicating if custom cache size is being used.
// 'custom_latency_ns', 'custom_buffer_size': Custom cache latency results and buffer size.
// 'custom_read_bw_gb_s', 'custom_write_bw_gb_s', 'custom_copy_bw_gb_s': Custom cache bandwidth results.
void print_results(int loop, size_t buffer_size, size_t buffer_size_mb, int iterations, int num_threads,
                   double read_bw_gb_s, double total_read_time, double write_bw_gb_s, double total_write_time,
                   double copy_bw_gb_s, double total_copy_time, 
                   double l1_latency_ns, double l2_latency_ns,
                   size_t l1_buffer_size, size_t l2_buffer_size,
                   double l1_read_bw_gb_s, double l1_write_bw_gb_s, double l1_copy_bw_gb_s,
                   double l2_read_bw_gb_s, double l2_write_bw_gb_s, double l2_copy_bw_gb_s,
                   double average_latency_ns, double total_lat_time_ns,
                   bool use_custom_cache_size, double custom_latency_ns, size_t custom_buffer_size,
                   double custom_read_bw_gb_s, double custom_write_bw_gb_s, double custom_copy_bw_gb_s) {
  // Print a header indicating the current loop number.
  std::cout << Messages::results_loop_header(loop) << std::endl;
  // Set output to fixed-point notation for consistent formatting.
  std::cout << std::fixed;

  // Display Main Memory Bandwidth test results.
  std::cout << Messages::results_main_memory_bandwidth(num_threads) << std::endl;
  std::cout << std::setprecision(Constants::BANDWIDTH_PRECISION);
  std::cout << Messages::results_read_bandwidth(read_bw_gb_s, total_read_time) << std::endl;
  std::cout << Messages::results_write_bandwidth(write_bw_gb_s, total_write_time) << std::endl;
  std::cout << Messages::results_copy_bandwidth(copy_bw_gb_s, total_copy_time) << std::endl;
  
  // Display main memory latency test results.
  std::cout << Messages::results_main_memory_latency() << std::endl;
  std::cout << Messages::results_latency_total_time(total_lat_time_ns / 1e9) << std::endl;
  std::cout << std::setprecision(2);
  std::cout << Messages::results_latency_average(average_latency_ns) << std::endl;

  // Display cache bandwidth test results.
  std::cout << Messages::results_cache_bandwidth() << std::endl;
  std::cout << std::setprecision(Constants::BANDWIDTH_PRECISION);
  if (use_custom_cache_size) {
    // Display custom cache bandwidth results
    if (custom_buffer_size > 0) {
      std::cout << Messages::results_custom_cache() << std::endl;
      std::cout << Messages::results_cache_read_bandwidth(custom_read_bw_gb_s);
      if (custom_buffer_size < 1024) {
        std::cout << Messages::results_buffer_size_bytes(custom_buffer_size) << std::endl;
      } else if (custom_buffer_size < 1024 * 1024) {
        std::cout << Messages::results_buffer_size_kb(custom_buffer_size / 1024.0) << std::endl;
      } else {
        std::cout << Messages::results_buffer_size_mb(custom_buffer_size / (1024.0 * 1024.0)) << std::endl;
      }
      std::cout << Messages::results_cache_write_bandwidth(custom_write_bw_gb_s) << std::endl;
      std::cout << Messages::results_cache_copy_bandwidth(custom_copy_bw_gb_s) << std::endl;
    }
  } else {
    // Display L1/L2 cache bandwidth results
    if (l1_buffer_size > 0) {
      std::cout << Messages::results_l1_cache() << std::endl;
      std::cout << Messages::results_cache_read_bandwidth(l1_read_bw_gb_s);
      if (l1_buffer_size < 1024) {
        std::cout << Messages::results_buffer_size_bytes(l1_buffer_size) << std::endl;
      } else if (l1_buffer_size < 1024 * 1024) {
        std::cout << Messages::results_buffer_size_kb(l1_buffer_size / 1024.0) << std::endl;
      } else {
        std::cout << Messages::results_buffer_size_mb(l1_buffer_size / (1024.0 * 1024.0)) << std::endl;
      }
      std::cout << Messages::results_cache_write_bandwidth(l1_write_bw_gb_s) << std::endl;
      std::cout << Messages::results_cache_copy_bandwidth(l1_copy_bw_gb_s) << std::endl;
    }
    if (l2_buffer_size > 0) {
      std::cout << Messages::results_l2_cache() << std::endl;
      std::cout << Messages::results_cache_read_bandwidth(l2_read_bw_gb_s);
      if (l2_buffer_size < 1024) {
        std::cout << Messages::results_buffer_size_bytes(l2_buffer_size) << std::endl;
      } else if (l2_buffer_size < 1024 * 1024) {
        std::cout << Messages::results_buffer_size_kb(l2_buffer_size / 1024.0) << std::endl;
      } else {
        std::cout << Messages::results_buffer_size_mb(l2_buffer_size / (1024.0 * 1024.0)) << std::endl;
      }
      std::cout << Messages::results_cache_write_bandwidth(l2_write_bw_gb_s) << std::endl;
      std::cout << Messages::results_cache_copy_bandwidth(l2_copy_bw_gb_s) << std::endl;
    }
  }

  // Display cache latency test results.
  std::cout << Messages::results_cache_latency() << std::endl;
  std::cout << std::setprecision(2);
  if (use_custom_cache_size) {
    // Display custom cache latency results
    if (custom_buffer_size > 0) {
      if (custom_buffer_size < 1024) {
        std::cout << Messages::results_cache_latency_custom_ns(custom_latency_ns, custom_buffer_size) << std::endl;
      } else if (custom_buffer_size < 1024 * 1024) {
        std::cout << Messages::results_cache_latency_custom_ns_kb(custom_latency_ns, custom_buffer_size / 1024.0) << std::endl;
      } else {
        std::cout << Messages::results_cache_latency_custom_ns_mb(custom_latency_ns, custom_buffer_size / (1024.0 * 1024.0)) << std::endl;
      }
    }
  } else {
    // Display L1/L2 cache latency results
    if (l1_buffer_size > 0) {
      if (l1_buffer_size < 1024) {
        std::cout << Messages::results_cache_latency_l1_ns(l1_latency_ns, l1_buffer_size) << std::endl;
      } else if (l1_buffer_size < 1024 * 1024) {
        std::cout << Messages::results_cache_latency_l1_ns_kb(l1_latency_ns, l1_buffer_size / 1024.0) << std::endl;
      } else {
        std::cout << Messages::results_cache_latency_l1_ns_mb(l1_latency_ns, l1_buffer_size / (1024.0 * 1024.0)) << std::endl;
      }
    }
    if (l2_buffer_size > 0) {
      if (l2_buffer_size < 1024) {
        std::cout << Messages::results_cache_latency_l2_ns(l2_latency_ns, l2_buffer_size) << std::endl;
      } else if (l2_buffer_size < 1024 * 1024) {
        std::cout << Messages::results_cache_latency_l2_ns_kb(l2_latency_ns, l2_buffer_size / 1024.0) << std::endl;
      } else {
        std::cout << Messages::results_cache_latency_l2_ns_mb(l2_latency_ns, l2_buffer_size / (1024.0 * 1024.0)) << std::endl;
      }
    }
  }
  // Print a separator after the loop results.
  std::cout << Messages::results_separator() << std::endl;
}

// --- Print Cache Information ---
// Outputs the detected cache sizes for L1 and L2 cache levels, or custom cache size.
// 'l1_cache_size': L1 data cache size in bytes (per P-core).
// 'l2_cache_size': L2 cache size in bytes (per P-core cluster).
// 'use_custom_cache_size': Flag indicating if custom cache size is being used.
// 'custom_cache_size_bytes': Custom cache size in bytes (only used if use_custom_cache_size is true).
void print_cache_info(size_t l1_cache_size, size_t l2_cache_size, bool use_custom_cache_size, size_t custom_cache_size_bytes) {
  std::cout << Messages::cache_info_header() << std::endl;
  std::cout << std::fixed << std::setprecision(2);
  
  if (use_custom_cache_size) {
    // Display custom cache size.
    std::cout << Messages::cache_size_custom(custom_cache_size_bytes) << std::endl;
  } else {
    // Display L1 cache size.
    std::cout << Messages::cache_size_l1(l1_cache_size) << std::endl;
    
    // Display L2 cache size.
    std::cout << Messages::cache_size_l2(l2_cache_size) << std::endl;
  }
  
}

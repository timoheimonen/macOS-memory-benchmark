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
 * @file output_printer.cpp
 * @brief Console output formatting
 */

#include <iomanip>    // Required for std::setprecision, std::fixed (output formatting)
#include <iostream>   // Required for std::cout, std::cerr
#include <sstream>    // Required for std::ostringstream

#include "core/config/version.h"  // SOFTVERSION
#include "core/config/constants.h"  // Include constants for default values
#include "output/console/messages.h"   // Include centralized messages
#include "output/console/output_printer.h"  // Function declarations

/**
 * @brief Displays program usage instructions via command-line arguments.
 *
 * @param prog_name The name of the executable (typically argv[0])
 */
void print_usage(const char *prog_name) {
  // Output usage syntax, version, options, and an example to standard error.
  std::cerr << Messages::usage_header(SOFTVERSION)
            << Messages::usage_options(prog_name)
            << Messages::usage_example(prog_name);
}

/**
 * @brief Outputs the configuration parameters the benchmark will run with.
 *
 * @param buffer_size The final size (in bytes) of each test buffer after memory checks
 * @param buffer_size_mb The requested/capped buffer size in Megabytes
 * @param iterations Number of iterations per bandwidth test per loop
 * @param loop_count How many times the entire set of tests will be repeated
 * @param use_non_cacheable Flag indicating if non-cacheable memory hints are enabled
 * @param cpu_name Detected processor name string
 * @param perf_cores Number of detected performance cores
 * @param eff_cores Number of detected efficiency cores
 * @param num_threads Total number of threads used for bandwidth tests
 * @param only_bandwidth Whether only bandwidth tests are run
 * @param only_latency Whether only latency tests are run
 * @param run_patterns Whether pattern benchmarks are run (bandwidth-only, uses 2x buffers)
 */
void print_configuration(size_t buffer_size, size_t buffer_size_mb, int iterations, int loop_count,
                         bool use_non_cacheable, const std::string &cpu_name, int perf_cores, int eff_cores, int num_threads, bool only_bandwidth, bool only_latency, bool run_patterns) {
  // Print benchmark header and copyright/license info.
  std::cout << Messages::config_header(SOFTVERSION) << std::endl;
  std::cout << Messages::config_copyright() << std::endl;
  std::cout << Messages::config_license() << std::endl;
  
  // Display buffer sizes conditionally
  if (buffer_size > 0) {
    // Display buffer sizes (actual MiB and requested/capped MB).
    std::cout << Messages::config_buffer_size(buffer_size / (1024.0 * 1024.0), buffer_size_mb) << std::endl;
    
    // Calculate and display total approximate memory allocated
    double total_mib = 0.0;
    if (only_bandwidth || run_patterns) {
      // Bandwidth-only or pattern benchmarks: src + dst = 2x
      total_mib = 2.0 * buffer_size / (1024.0 * 1024.0);
    } else if (only_latency) {
      // Latency-only: lat = 1x
      total_mib = buffer_size / (1024.0 * 1024.0);
    } else {
      // Normal run: src + dst + lat = 3x
      total_mib = 3.0 * buffer_size / (1024.0 * 1024.0);
    }
    std::cout << Messages::config_total_allocation(total_mib) << std::endl;
  }
  
  // Display test repetition counts conditionally
  if (!only_latency) {
    std::cout << Messages::config_iterations(iterations) << std::endl;
  }
  std::cout << Messages::config_loop_count(loop_count) << std::endl;
  // Display non-cacheable memory hints status.
  std::cout << Messages::config_non_cacheable(use_non_cacheable) << std::endl;
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

/**
 * @brief Outputs the performance results measured during one complete benchmark loop.
 *
 * @param loop Index of the loop being reported (0-based)
 * @param buffer_size Buffer size in bytes
 * @param buffer_size_mb Buffer size in megabytes
 * @param iterations Number of iterations per test
 * @param num_threads Number of threads used
 * @param read_bw_gb_s Read bandwidth in GB/s
 * @param total_read_time Total read test time in seconds
 * @param write_bw_gb_s Write bandwidth in GB/s
 * @param total_write_time Total write test time in seconds
 * @param copy_bw_gb_s Copy bandwidth in GB/s
 * @param total_copy_time Total copy test time in seconds
 * @param l1_latency_ns L1 cache latency in nanoseconds
 * @param l2_latency_ns L2 cache latency in nanoseconds
 * @param l1_buffer_size L1 cache buffer size in bytes
 * @param l2_buffer_size L2 cache buffer size in bytes
 * @param l1_read_bw_gb_s L1 cache read bandwidth in GB/s
 * @param l1_write_bw_gb_s L1 cache write bandwidth in GB/s
 * @param l1_copy_bw_gb_s L1 cache copy bandwidth in GB/s
 * @param l2_read_bw_gb_s L2 cache read bandwidth in GB/s
 * @param l2_write_bw_gb_s L2 cache write bandwidth in GB/s
 * @param l2_copy_bw_gb_s L2 cache copy bandwidth in GB/s
 * @param average_latency_ns Main memory average latency in nanoseconds
 * @param total_lat_time_ns Main memory total latency test time in nanoseconds
 * @param use_custom_cache_size Flag indicating if custom cache size is being used
 * @param custom_latency_ns Custom cache latency in nanoseconds
 * @param custom_buffer_size Custom cache buffer size in bytes
 * @param custom_read_bw_gb_s Custom cache read bandwidth in GB/s
 * @param custom_write_bw_gb_s Custom cache write bandwidth in GB/s
 * @param custom_copy_bw_gb_s Custom cache copy bandwidth in GB/s
 * @param user_specified_threads Whether user specified thread count
 * @param only_bandwidth Whether only bandwidth tests are run
 * @param only_latency Whether only latency tests are run
 */
void print_results(int loop, size_t buffer_size, size_t buffer_size_mb, int iterations, int num_threads,
                   double read_bw_gb_s, double total_read_time, double write_bw_gb_s, double total_write_time,
                   double copy_bw_gb_s, double total_copy_time,
                   double l1_latency_ns, double l2_latency_ns,
                   size_t l1_buffer_size, size_t l2_buffer_size,
                   double l1_read_bw_gb_s, double l1_write_bw_gb_s, double l1_copy_bw_gb_s,
                   double l2_read_bw_gb_s, double l2_write_bw_gb_s, double l2_copy_bw_gb_s,
                   double average_latency_ns, double total_lat_time_ns,
                   bool use_custom_cache_size, double custom_latency_ns, size_t custom_buffer_size,
                   double custom_read_bw_gb_s, double custom_write_bw_gb_s, double custom_copy_bw_gb_s,
                   bool user_specified_threads, bool only_bandwidth, bool only_latency) {
  // Print a header indicating the current loop number.
  std::cout << Messages::results_loop_header(loop) << std::endl;
  // Set output to fixed-point notation for consistent formatting.
  std::cout << std::fixed;

  // Display Main Memory Bandwidth test results (skip if only latency tests).
  if (!only_latency) {
    std::cout << Messages::results_main_memory_bandwidth(num_threads) << std::endl;
    std::cout << std::setprecision(Constants::BANDWIDTH_PRECISION);
    std::cout << Messages::results_read_bandwidth(read_bw_gb_s, total_read_time) << std::endl;
    std::cout << Messages::results_write_bandwidth(write_bw_gb_s, total_write_time) << std::endl;
    std::cout << Messages::results_copy_bandwidth(copy_bw_gb_s, total_copy_time) << std::endl;
  }
  
  // Display main memory latency test results (skip if only bandwidth tests).
  if (!only_bandwidth) {
    std::cout << Messages::results_main_memory_latency() << std::endl;
    std::cout << Messages::results_latency_total_time(total_lat_time_ns / 1e9) << std::endl;
    std::cout << std::setprecision(Constants::LATENCY_PRECISION);
    std::cout << Messages::results_latency_average(average_latency_ns) << std::endl;
  }

  // Display cache bandwidth test results (skip if only latency tests).
  if (!only_latency) {
    // Cache tests use user-specified threads if set, otherwise single-threaded
    int cache_threads = user_specified_threads ? num_threads : Constants::SINGLE_THREAD;
    std::cout << Messages::results_cache_bandwidth(cache_threads) << std::endl;
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
  }

  // Display cache latency test results (skip if only bandwidth tests).
  if (!only_bandwidth) {
    std::cout << Messages::results_cache_latency() << std::endl;
    std::cout << std::setprecision(Constants::LATENCY_PRECISION);
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
  }
  // Print a separator after the loop results.
  std::cout << Messages::results_separator() << std::endl;
}

/**
 * @brief Outputs the detected cache sizes for L1 and L2 cache levels, or custom cache size.
 *
 * @param l1_cache_size L1 data cache size in bytes (per P-core)
 * @param l2_cache_size L2 cache size in bytes (per P-core cluster)
 * @param use_custom_cache_size Flag indicating if custom cache size is being used
 * @param custom_cache_size_bytes Custom cache size in bytes
 */
void print_cache_info(size_t l1_cache_size, size_t l2_cache_size, bool use_custom_cache_size, size_t custom_cache_size_bytes) {
  std::cout << Messages::cache_info_header() << std::endl;
  std::cout << std::fixed << std::setprecision(Constants::LATENCY_PRECISION);
  
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

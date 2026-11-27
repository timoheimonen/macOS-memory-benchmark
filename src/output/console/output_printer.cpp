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
 * @file output_printer.cpp
 * @brief Console output formatting
 */

#include <iomanip>    // Required for std::setprecision, std::fixed (output formatting)
#include <iostream>   // Required for std::cout, std::cerr
#include <sstream>    // Required for std::ostringstream

#include "core/config/version.h"  // SOFTVERSION
#include "core/config/constants.h"  // Include constants for default values
#include "core/config/config.h"
#include "output/console/messages/messages_api.h"   // Include centralized messages
#include "output/console/output_printer.h"  // Function declarations

/**
 * @brief Displays help text to standard output (for -h/--help).
 *
 * @param prog_name The name of the executable (typically argv[0])
 */
void print_help(const char *prog_name) {
  const char* safe_prog_name =
      (prog_name != nullptr && prog_name[0] != '\0') ? prog_name : "memory_benchmark";
  std::cout << Messages::usage_header(SOFTVERSION)
            << Messages::usage_options(safe_prog_name)
            << Messages::usage_example(safe_prog_name);
}

void print_runtime_banner() {
  std::cout << Messages::config_header(SOFTVERSION) << std::endl;
  std::cout << Messages::config_copyright() << std::endl;
  std::cout << Messages::config_license() << std::endl;
}

/**
 * @brief Outputs the configuration parameters the benchmark will run with.
 *
 * @param buffer_size The final size (in bytes) of each test buffer after memory checks
 * @param buffer_size_mb The requested/capped buffer size in Megabytes
 * @param total_allocation_bytes Peak concurrently allocated bytes across enabled benchmark phases
 * @param iterations Number of iterations per bandwidth test per loop
 * @param loop_count How many times the entire set of tests will be repeated
 * @param use_non_cacheable Flag indicating if non-cacheable memory hints are enabled
 * @param latency_stride_bytes Stride used by latency pointer chains
 * @param latency_chain_mode_name Chain construction mode used by latency pointer chains
 * @param latency_tlb_locality_bytes TLB-locality window for latency chains (0 = global random)
 * @param cpu_name Detected processor name string
 * @param perf_cores Number of detected performance cores
 * @param eff_cores Number of detected efficiency cores
 * @param num_threads Total number of threads used for bandwidth tests
 */
void print_configuration(size_t buffer_size, size_t buffer_size_mb, size_t total_allocation_bytes,
                         int iterations, int loop_count, bool use_non_cacheable,
                         size_t latency_stride_bytes,
                         const std::string& latency_chain_mode_name,
                         size_t latency_tlb_locality_bytes,
                         const std::string &cpu_name, int perf_cores, int eff_cores, int num_threads) {
  // Display buffer sizes conditionally
  if (buffer_size > 0) {
    // Display buffer sizes (actual MiB and requested/capped MB).
    std::cout << Messages::config_buffer_size(buffer_size / (1024.0 * 1024.0), buffer_size_mb) << std::endl;
  }

  if (total_allocation_bytes > 0) {
    const double total_mib = total_allocation_bytes / (1024.0 * 1024.0);
    std::cout << Messages::config_total_allocation(total_mib) << std::endl;
  }

  std::cout << Messages::config_iterations(iterations) << std::endl;
  std::cout << Messages::config_loop_count(loop_count) << std::endl;
  // Display non-cacheable memory hints status.
  std::cout << Messages::config_non_cacheable(use_non_cacheable) << std::endl;
  std::cout << Messages::config_latency_stride(latency_stride_bytes) << std::endl;
  std::cout << Messages::config_latency_chain_mode(latency_chain_mode_name) << std::endl;
  std::cout << Messages::config_latency_tlb_locality(latency_tlb_locality_bytes) << std::endl;
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
  std::cout << Messages::config_total_cores(perf_cores + eff_cores) << std::endl;
  std::cout << Messages::config_benchmark_threads(num_threads) << std::endl;
}

/**
 * @brief Outputs the detected cache sizes for L1 and L2 cache levels, or custom cache size.
 *
 * @param l1_cache_size L1 data cache size in bytes (per P-core)
 * @param l2_cache_size L2 cache size in bytes (per P-core cluster)
 * @param use_custom_cache_size Flag indicating if custom cache size is being used
 * @param custom_cache_size_bytes Custom cache size in bytes
 */
void print_cache_info(size_t l1_cache_size, size_t l2_cache_size,
                      bool use_custom_cache_size,
                      size_t custom_cache_size_bytes) {
  std::cout << Messages::cache_info_header() << std::endl;
  std::cout << std::fixed << std::setprecision(Constants::LATENCY_PRECISION);
  
  if (use_custom_cache_size) {
    // Display custom cache size or indicate when cache latency is disabled.
    if (custom_cache_size_bytes == 0) {
      std::cout << Messages::cache_size_custom_disabled() << std::endl;
    } else {
      std::cout << Messages::cache_size_custom(custom_cache_size_bytes) << std::endl;
    }
  } else {
    // Display L1 cache size.
    std::cout << Messages::cache_size_l1(l1_cache_size) << std::endl;
    
    // Display L2 cache size.
    std::cout << Messages::cache_size_l2(l2_cache_size) << std::endl;
  }
  
}

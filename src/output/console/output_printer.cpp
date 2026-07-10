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
#include "benchmark/benchmark_runner.h"
#include "output/console/messages/messages_api.h"   // Include centralized messages
#include "output/console/output_printer.h"  // Function declarations

namespace {

std::string unavailable_measurement(const std::string& label,
                                    const BenchmarkMeasurement& measurement) {
  return Messages::results_measurement_unavailable(
      label, benchmark_measurement_status_to_string(measurement.status),
      measurement.status_reason);
}

void print_cache_buffer_suffix(size_t buffer_size) {
  if (buffer_size < 1024) {
    std::cout << Messages::results_buffer_size_bytes(buffer_size) << std::endl;
  } else if (buffer_size < 1024 * 1024) {
    std::cout << Messages::results_buffer_size_kb(buffer_size / 1024.0) << std::endl;
  } else {
    std::cout << Messages::results_buffer_size_mb(
                     buffer_size / (1024.0 * 1024.0))
              << std::endl;
  }
}

void print_cache_bandwidth_measurements(const BenchmarkMeasurement& read,
                                        const BenchmarkMeasurement& write,
                                        const BenchmarkMeasurement& copy,
                                        size_t buffer_size) {
  if (read.is_measured()) {
    std::cout << Messages::results_cache_read_bandwidth(*read.value);
  } else {
    std::cout << unavailable_measurement("    Read ", read);
  }
  print_cache_buffer_suffix(buffer_size);

  std::cout << (write.is_measured()
                    ? Messages::results_cache_write_bandwidth(*write.value)
                    : unavailable_measurement("    Write", write))
            << std::endl;
  std::cout << (copy.is_measured()
                    ? Messages::results_cache_copy_bandwidth(*copy.value)
                    : unavailable_measurement("    Copy ", copy))
            << std::endl;
}

void print_cache_latency_measurement(const std::string& label,
                                     const BenchmarkMeasurement& measurement,
                                     size_t buffer_size) {
  if (!measurement.is_measured()) {
    std::cout << unavailable_measurement("  " + label, measurement);
    print_cache_buffer_suffix(buffer_size);
    return;
  }

  const double latency_ns = *measurement.value;
  if (label == "Custom Cache") {
    if (buffer_size < 1024) {
      std::cout << Messages::results_cache_latency_custom_ns(latency_ns, buffer_size)
                << std::endl;
    } else if (buffer_size < 1024 * 1024) {
      std::cout << Messages::results_cache_latency_custom_ns_kb(
                       latency_ns, buffer_size / 1024.0)
                << std::endl;
    } else {
      std::cout << Messages::results_cache_latency_custom_ns_mb(
                       latency_ns, buffer_size / (1024.0 * 1024.0))
                << std::endl;
    }
    return;
  }

  const bool is_l1 = label == "L1 Cache";
  if (buffer_size < 1024) {
    std::cout << (is_l1 ? Messages::results_cache_latency_l1_ns(latency_ns, buffer_size)
                        : Messages::results_cache_latency_l2_ns(latency_ns, buffer_size))
              << std::endl;
  } else if (buffer_size < 1024 * 1024) {
    std::cout << (is_l1 ? Messages::results_cache_latency_l1_ns_kb(
                              latency_ns, buffer_size / 1024.0)
                        : Messages::results_cache_latency_l2_ns_kb(
                              latency_ns, buffer_size / 1024.0))
              << std::endl;
  } else {
    std::cout << (is_l1 ? Messages::results_cache_latency_l1_ns_mb(
                              latency_ns, buffer_size / (1024.0 * 1024.0))
                        : Messages::results_cache_latency_l2_ns_mb(
                              latency_ns, buffer_size / (1024.0 * 1024.0)))
              << std::endl;
  }
}

}  // namespace

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
 * @brief Displays help text to standard output (for -h/--help).
 *
 * @param prog_name The name of the executable (typically argv[0])
 */
void print_help(const char *prog_name) {
  std::cout << Messages::usage_header(SOFTVERSION)
            << Messages::usage_options(prog_name)
            << Messages::usage_example(prog_name);
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
 * @param latency_tlb_locality_bytes TLB-locality window for latency chains (0 = disabled)
 * @param cpu_name Detected processor name string
 * @param perf_cores Number of detected performance cores
 * @param eff_cores Number of detected efficiency cores
 * @param num_threads Total number of threads used for bandwidth tests
 * @param only_bandwidth Whether only bandwidth tests are run
 * @param only_latency Whether only latency tests are run
 * @param run_patterns Whether pattern benchmarks are run (bandwidth-only, uses 2x buffers)
 */
void print_configuration(size_t buffer_size, size_t buffer_size_mb, size_t total_allocation_bytes,
                         int iterations, int loop_count, bool use_non_cacheable,
                         size_t latency_stride_bytes,
                         const std::string& latency_chain_mode_name,
                         size_t latency_tlb_locality_bytes,
                         const std::string &cpu_name, int perf_cores, int eff_cores, int num_threads,
                         bool only_bandwidth, bool only_latency, bool run_patterns,
                         bool user_specified_iterations) {
  // Print benchmark header and copyright/license info.
  std::cout << Messages::config_header(SOFTVERSION) << std::endl;
  std::cout << Messages::config_copyright() << std::endl;
  std::cout << Messages::config_license() << std::endl;
  
  // Display buffer sizes conditionally
  if (buffer_size > 0) {
    // Display buffer sizes (actual MiB and requested/capped MB).
    std::cout << Messages::config_buffer_size(buffer_size / (1024.0 * 1024.0), buffer_size_mb) << std::endl;
  }

  if (total_allocation_bytes > 0) {
    const double total_mib = total_allocation_bytes / (1024.0 * 1024.0);
    std::cout << Messages::config_total_allocation(total_mib) << std::endl;
  }
  
  // Display test repetition counts conditionally
  if (!only_latency) {
    if (run_patterns && !user_specified_iterations) {
      std::cout << Messages::config_pattern_iterations_auto(
                       Constants::PATTERN_CALIBRATION_TARGET_SECONDS,
                       Constants::PATTERN_CALIBRATION_MIN_SECONDS,
                       Constants::PATTERN_CALIBRATION_MAX_SECONDS)
                << std::endl;
    } else if (!user_specified_iterations) {
      std::cout << Messages::config_benchmark_iterations_auto(
                       Constants::BENCHMARK_CALIBRATION_TARGET_SECONDS,
                       Constants::BENCHMARK_CALIBRATION_MIN_SECONDS,
                       Constants::BENCHMARK_CALIBRATION_MAX_SECONDS)
                << std::endl;
    } else {
      std::cout << Messages::config_iterations(iterations) << std::endl;
    }
  }
  if (!only_bandwidth && !run_patterns) {
    std::cout << Messages::config_latency_calibration(
                     Constants::BENCHMARK_LATENCY_TARGET_SECONDS,
                     Constants::BENCHMARK_LATENCY_CALIBRATION_MIN_SECONDS,
                     Constants::BENCHMARK_LATENCY_CALIBRATION_MAX_SECONDS,
                     Constants::BENCHMARK_LATENCY_MIN_COMPLETE_CYCLES)
              << std::endl;
  }
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
 * @param latency_tlb_locality_bytes Active locality window used for headline latency (0 = global random)
 * @param total_lat_time_ns Main memory total latency test time in nanoseconds
 * @param use_custom_cache_size Flag indicating if custom cache size is being used
 * @param custom_latency_ns Custom cache latency in nanoseconds
 * @param custom_buffer_size Custom cache buffer size in bytes
 * @param custom_read_bw_gb_s Custom cache read bandwidth in GB/s
 * @param custom_write_bw_gb_s Custom cache write bandwidth in GB/s
 * @param custom_copy_bw_gb_s Custom cache copy bandwidth in GB/s
 * @param has_auto_tlb_breakdown Whether automatic TLB hit/miss breakdown is available
 * @param tlb_hit_latency_ns TLB hit-biased latency in nanoseconds
 * @param tlb_miss_latency_ns TLB miss-biased latency in nanoseconds
 * @param page_walk_penalty_ns Estimated page-walk penalty in nanoseconds
 * @param user_specified_threads Whether user specified thread count
 * @param only_bandwidth Whether only bandwidth tests are run
 * @param only_latency Whether only latency tests are run
 */
void print_results(int loop, const BenchmarkConfig& config,
                   const BenchmarkResults& results) {
  // Print a header indicating the current loop number.
  std::cout << Messages::results_loop_header(loop) << std::endl;
  // Set output to fixed-point notation for consistent formatting.
  std::cout << std::fixed;

  // Display Main Memory Bandwidth test results (skip if only latency tests).
  if (!config.only_latency) {
    std::cout << Messages::results_main_memory_bandwidth(config.num_threads)
              << std::endl;
    std::cout << std::setprecision(Constants::BANDWIDTH_PRECISION);
    std::cout << (results.main_read_bandwidth.is_measured()
                      ? Messages::results_read_bandwidth(
                            *results.main_read_bandwidth.value,
                            results.main_read_bandwidth.elapsed_seconds)
                      : unavailable_measurement("  Read ",
                                                results.main_read_bandwidth))
              << std::endl;
    std::cout << (results.main_write_bandwidth.is_measured()
                      ? Messages::results_write_bandwidth(
                            *results.main_write_bandwidth.value,
                            results.main_write_bandwidth.elapsed_seconds)
                      : unavailable_measurement("  Write",
                                                results.main_write_bandwidth))
              << std::endl;
    std::cout << (results.main_copy_bandwidth.is_measured()
                      ? Messages::results_copy_bandwidth(
                            *results.main_copy_bandwidth.value,
                            results.main_copy_bandwidth.elapsed_seconds)
                      : unavailable_measurement("  Copy ",
                                                results.main_copy_bandwidth))
              << std::endl;
  }

  // Display main memory latency test results (skip if only bandwidth tests
  // or when main memory latency path is disabled).
  if (!config.only_bandwidth && config.buffer_size > 0) {
    std::cout << Messages::results_main_memory_latency() << std::endl;
    std::cout << std::setprecision(Constants::LATENCY_PRECISION);
    if (results.main_latency.is_measured()) {
      std::cout << Messages::results_latency_total_time(
                       results.main_latency.elapsed_seconds)
                << std::endl;
      std::cout << Messages::results_latency_average(
                       *results.main_latency.value,
                       config.latency_tlb_locality_bytes)
                << std::endl;
    } else {
      std::cout << unavailable_measurement("  Average latency",
                                          results.main_latency)
                << std::endl;
    }
    if (results.locality_16k_latency.is_measured() &&
        results.global_random_latency.is_measured() &&
        results.locality_latency_delta.is_measured()) {
      std::cout << Messages::results_latency_tlb_hit(
                       *results.locality_16k_latency.value)
                << std::endl;
      std::cout << Messages::results_latency_tlb_miss(
                       *results.global_random_latency.value)
                << std::endl;
      std::cout << Messages::results_latency_page_walk_penalty(
                       *results.locality_latency_delta.value)
                << std::endl;
    }
  }

  // Display cache bandwidth test results (skip if only latency tests).
  if (!config.only_latency) {
    // Cache tests use user-specified threads if set, otherwise single-threaded
    int cache_threads = config.user_specified_threads ? config.num_threads
                                                      : Constants::SINGLE_THREAD;
    std::cout << Messages::results_cache_bandwidth(cache_threads) << std::endl;
    std::cout << std::setprecision(Constants::BANDWIDTH_PRECISION);
    if (config.use_custom_cache_size) {
      if (config.custom_buffer_size > 0) {
        std::cout << Messages::results_custom_cache() << std::endl;
        print_cache_bandwidth_measurements(
            results.custom_read_bandwidth, results.custom_write_bandwidth,
            results.custom_copy_bandwidth, config.custom_buffer_size);
      }
    } else {
      if (config.l1_buffer_size > 0) {
        std::cout << Messages::results_l1_cache() << std::endl;
        print_cache_bandwidth_measurements(
            results.l1_read_bandwidth, results.l1_write_bandwidth,
            results.l1_copy_bandwidth, config.l1_buffer_size);
      }
      if (config.l2_buffer_size > 0) {
        std::cout << Messages::results_l2_cache() << std::endl;
        print_cache_bandwidth_measurements(
            results.l2_read_bandwidth, results.l2_write_bandwidth,
            results.l2_copy_bandwidth, config.l2_buffer_size);
      }
    }
  }

  // Display cache latency test results (skip if only bandwidth tests).
  const bool has_cache_latency_results =
      config.use_custom_cache_size
          ? config.custom_buffer_size > 0
          : (config.l1_buffer_size > 0 || config.l2_buffer_size > 0);
  if (!config.only_bandwidth && has_cache_latency_results) {
    std::cout << Messages::results_cache_latency() << std::endl;
    std::cout << std::setprecision(Constants::LATENCY_PRECISION);
    if (config.use_custom_cache_size) {
      if (config.custom_buffer_size > 0) {
        print_cache_latency_measurement("Custom Cache", results.custom_latency,
                                        config.custom_buffer_size);
      }
    } else {
      if (config.l1_buffer_size > 0) {
        print_cache_latency_measurement("L1 Cache", results.l1_latency,
                                        config.l1_buffer_size);
      }
      if (config.l2_buffer_size > 0) {
        print_cache_latency_measurement("L2 Cache", results.l2_latency,
                                        config.l2_buffer_size);
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

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
#include "benchmark/benchmark_results.h"
#include "core/config/config.h"           // BenchmarkConfig
#include "benchmark/benchmark_runner.h" // BenchmarkResults
#include "benchmark/benchmark_executor.h" // TimingResults
#include "core/config/constants.h"
#include <limits>  // std::numeric_limits
#include <cmath>   // std::isnan, std::isinf

// Helper function to calculate bandwidth for a single cache level
void calculate_single_bandwidth(size_t buffer_size, int iterations,
                               double read_time, double write_time, double copy_time,
                               double& read_bw_gb_s, double& write_bw_gb_s, double& copy_bw_gb_s) {
  // Initialize outputs to 0
  read_bw_gb_s = 0.0;
  write_bw_gb_s = 0.0;
  copy_bw_gb_s = 0.0;
  
  // Check for overflow before multiplication: iterations * buffer_size
  if (iterations > 0 && buffer_size > 0) {
    // Check if iterations * buffer_size would overflow size_t
    if (static_cast<size_t>(iterations) > std::numeric_limits<size_t>::max() / buffer_size) {
      // Overflow would occur, use double precision calculation instead
      double total_bytes_read = static_cast<double>(iterations) * static_cast<double>(buffer_size);
      double total_bytes_written = total_bytes_read;
      double total_bytes_copied_op = total_bytes_read * Constants::COPY_OPERATION_MULTIPLIER;
      
      if (read_time > 0 && !std::isnan(read_time) && !std::isinf(read_time)) {
        read_bw_gb_s = total_bytes_read / read_time / Constants::NANOSECONDS_PER_SECOND;
        // Validate result is finite and non-negative
        if (std::isnan(read_bw_gb_s) || std::isinf(read_bw_gb_s) || read_bw_gb_s < 0.0) {
          read_bw_gb_s = 0.0;
        }
      }
      if (write_time > 0 && !std::isnan(write_time) && !std::isinf(write_time)) {
        write_bw_gb_s = total_bytes_written / write_time / Constants::NANOSECONDS_PER_SECOND;
        // Validate result is finite and non-negative
        if (std::isnan(write_bw_gb_s) || std::isinf(write_bw_gb_s) || write_bw_gb_s < 0.0) {
          write_bw_gb_s = 0.0;
        }
      }
      if (copy_time > 0 && !std::isnan(copy_time) && !std::isinf(copy_time)) {
        copy_bw_gb_s = total_bytes_copied_op / copy_time / Constants::NANOSECONDS_PER_SECOND;
        // Validate result is finite and non-negative
        if (std::isnan(copy_bw_gb_s) || std::isinf(copy_bw_gb_s) || copy_bw_gb_s < 0.0) {
          copy_bw_gb_s = 0.0;
        }
      }
    } else {
      // No overflow, use integer calculation
      size_t total_bytes_read = static_cast<size_t>(iterations) * buffer_size;
      size_t total_bytes_written = total_bytes_read;
      size_t total_bytes_copied_op = total_bytes_read * Constants::COPY_OPERATION_MULTIPLIER;
      
      if (read_time > 0 && !std::isnan(read_time) && !std::isinf(read_time)) {
        read_bw_gb_s = static_cast<double>(total_bytes_read) / read_time / Constants::NANOSECONDS_PER_SECOND;
        // Validate result is finite and non-negative
        if (std::isnan(read_bw_gb_s) || std::isinf(read_bw_gb_s) || read_bw_gb_s < 0.0) {
          read_bw_gb_s = 0.0;
        }
      }
      if (write_time > 0 && !std::isnan(write_time) && !std::isinf(write_time)) {
        write_bw_gb_s = static_cast<double>(total_bytes_written) / write_time / Constants::NANOSECONDS_PER_SECOND;
        // Validate result is finite and non-negative
        if (std::isnan(write_bw_gb_s) || std::isinf(write_bw_gb_s) || write_bw_gb_s < 0.0) {
          write_bw_gb_s = 0.0;
        }
      }
      if (copy_time > 0 && !std::isnan(copy_time) && !std::isinf(copy_time)) {
        copy_bw_gb_s = static_cast<double>(total_bytes_copied_op) / 
                       copy_time / Constants::NANOSECONDS_PER_SECOND;
        // Validate result is finite and non-negative
        if (std::isnan(copy_bw_gb_s) || std::isinf(copy_bw_gb_s) || copy_bw_gb_s < 0.0) {
          copy_bw_gb_s = 0.0;
        }
      }
    }
  }
}

// Calculate bandwidth results from timing data
void calculate_bandwidth_results(const BenchmarkConfig& config, const TimingResults& timings, 
                                 BenchmarkResults& results) {
  // Main memory bandwidth calculations
  calculate_single_bandwidth(config.buffer_size, config.iterations,
                             timings.total_read_time, timings.total_write_time, timings.total_copy_time,
                             results.read_bw_gb_s, results.write_bw_gb_s, results.copy_bw_gb_s);
  
  // Cache bandwidth calculations
  int cache_iterations = config.iterations * Constants::CACHE_ITERATIONS_MULTIPLIER;
  
  if (config.use_custom_cache_size) {
    if (config.custom_buffer_size > 0) {
      calculate_single_bandwidth(config.custom_buffer_size, cache_iterations,
                                 timings.custom_read_time, timings.custom_write_time, timings.custom_copy_time,
                                 results.custom_read_bw_gb_s, results.custom_write_bw_gb_s, results.custom_copy_bw_gb_s);
    }
  } else {
    if (config.l1_buffer_size > 0) {
      calculate_single_bandwidth(config.l1_buffer_size, cache_iterations,
                                 timings.l1_read_time, timings.l1_write_time, timings.l1_copy_time,
                                 results.l1_read_bw_gb_s, results.l1_write_bw_gb_s, results.l1_copy_bw_gb_s);
    }
    if (config.l2_buffer_size > 0) {
      calculate_single_bandwidth(config.l2_buffer_size, cache_iterations,
                                 timings.l2_read_time, timings.l2_write_time, timings.l2_copy_time,
                                 results.l2_read_bw_gb_s, results.l2_write_bw_gb_s, results.l2_copy_bw_gb_s);
    }
  }
}

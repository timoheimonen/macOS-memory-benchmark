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
 * @file benchmark_results.cpp
 * @brief Result calculation implementations
 *
 * Implements bandwidth calculation functions that convert raw timing data into
 * bandwidth measurements (GB/s). Handles overflow-safe arithmetic, validation,
 * and special handling for copy operations which involve both read and write.
 *
 * Key features:
 * - Overflow-safe calculations for large buffer sizes and iteration counts
 * - Validation of timing inputs (NaN, infinity, negative values)
 * - Separate calculations for main memory and cache buffers
 * - Conditional calculation based on benchmark configuration
 * - Copy bandwidth multiplier to account for read+write operations
 *
 * Bandwidth formula:
 * - bandwidth_gb_s = total_bytes / time_seconds / NANOSECONDS_PER_SECOND
 * - For copy: total_bytes *= COPY_OPERATION_MULTIPLIER (accounts for read+write)
 */

#include "benchmark/benchmark_results.h"
#include "core/config/config.h"           // BenchmarkConfig
#include "benchmark/benchmark_runner.h" // BenchmarkResults
#include "benchmark/benchmark_executor.h" // TimingResults
#include "core/config/constants.h"
#include <limits>  // std::numeric_limits
#include <cmath>   // std::isnan, std::isinf

namespace {

int calculate_cache_iterations_saturated(int iterations) {
  if (iterations <= 0) {
    return 0;
  }

  if (iterations > std::numeric_limits<int>::max() / Constants::CACHE_ITERATIONS_MULTIPLIER) {
    return std::numeric_limits<int>::max();
  }

  return iterations * Constants::CACHE_ITERATIONS_MULTIPLIER;
}

void set_bandwidth_measurements(size_t buffer_size,
                                int iterations,
                                double read_time,
                                double write_time,
                                double copy_time,
                                BenchmarkMeasurementStatus read_status,
                                BenchmarkMeasurementStatus write_status,
                                BenchmarkMeasurementStatus copy_status,
                                BenchmarkMeasurement& read_measurement,
                                BenchmarkMeasurement& write_measurement,
                                BenchmarkMeasurement& copy_measurement) {
  double read_bandwidth = 0.0;
  double write_bandwidth = 0.0;
  double copy_bandwidth = 0.0;
  calculate_single_bandwidth(buffer_size, iterations, read_time, write_time, copy_time,
                             read_bandwidth, write_bandwidth, copy_bandwidth);

  auto set_one = [buffer_size, iterations](BenchmarkMeasurementStatus status,
                                           double bandwidth,
                                           double elapsed_seconds,
                                           bool is_copy,
                                           BenchmarkMeasurement& measurement) {
    measurement.status = status;
    measurement.passes = iterations > 0 ? static_cast<size_t>(iterations) : 0;
    if (iterations > 0 && buffer_size <= std::numeric_limits<size_t>::max() /
                                               static_cast<size_t>(iterations)) {
      const size_t base_payload = buffer_size * static_cast<size_t>(iterations);
      if (!is_copy || base_payload <= std::numeric_limits<size_t>::max() /
                                         Constants::COPY_OPERATION_MULTIPLIER) {
        measurement.exact_payload_bytes =
            is_copy ? base_payload * Constants::COPY_OPERATION_MULTIPLIER : base_payload;
      }
    }

    if (status != BenchmarkMeasurementStatus::Measured) {
      measurement.value.reset();
      return;
    }
    if (bandwidth <= 0.0 || !std::isfinite(bandwidth) || elapsed_seconds <= 0.0 ||
        !std::isfinite(elapsed_seconds)) {
      set_measurement_unavailable(measurement, BenchmarkMeasurementStatus::Invalid,
                                  "invalid measured bandwidth or duration");
      return;
    }
    set_measurement_value(measurement, bandwidth, elapsed_seconds);
  };

  set_one(read_status, read_bandwidth, read_time, false, read_measurement);
  set_one(write_status, write_bandwidth, write_time, false, write_measurement);
  set_one(copy_status, copy_bandwidth, copy_time, true, copy_measurement);
}

}  // namespace

/**
 * @brief Helper function to calculate bandwidth for a single memory level (main or cache).
 *
 * Converts raw timing measurements into bandwidth (GB/s) for read, write, and copy operations.
 * Performs overflow-safe arithmetic and validates all inputs to ensure correct results.
 *
 * Overflow handling:
 * - Checks if iterations * buffer_size would overflow size_t
 * - Falls back to double-precision arithmetic if overflow detected
 * - Prevents undefined behavior from integer overflow
 *
 * Validation:
 * - Checks for zero, negative, NaN, or infinite timing values
 * - Sets bandwidth to 0.0 for invalid inputs
 * - Validates calculated bandwidth is finite and non-negative
 *
 * Copy operation special handling:
 * - Multiplies bytes by COPY_OPERATION_MULTIPLIER (typically 2)
 * - Accounts for both read and write operations in copy
 *
 * @param[in]  buffer_size   Size of the buffer in bytes
 * @param[in]  iterations    Number of times the operation was repeated
 * @param[in]  read_time     Total read time in nanoseconds
 * @param[in]  write_time    Total write time in nanoseconds
 * @param[in]  copy_time     Total copy time in nanoseconds
 * @param[out] read_bw_gb_s  Calculated read bandwidth in GB/s (0.0 on error)
 * @param[out] write_bw_gb_s Calculated write bandwidth in GB/s (0.0 on error)
 * @param[out] copy_bw_gb_s  Calculated copy bandwidth in GB/s (0.0 on error)
 *
 * @note All output parameters are initialized to 0.0 before calculation.
 * @note Uses double-precision fallback to handle large values safely.
 * @note Copy bandwidth includes both read and write (2x data movement).
 *
 * @see calculate_bandwidth_results() which calls this for each memory level
 */
void calculate_single_bandwidth(size_t buffer_size, int iterations,
                               double read_time, double write_time, double copy_time,
                               double& read_bw_gb_s, double& write_bw_gb_s, double& copy_bw_gb_s) {
  // Initialize outputs to 0
  read_bw_gb_s = 0.0;
  write_bw_gb_s = 0.0;
  copy_bw_gb_s = 0.0;
  
  if (iterations <= 0 || buffer_size == 0) {
    return;
  }

  auto calculate_valid_bandwidth = [](double total_bytes, double elapsed_time) {
    if (elapsed_time <= 0 || std::isnan(elapsed_time) || std::isinf(elapsed_time)) {
      return 0.0;
    }

    double bandwidth = total_bytes / elapsed_time / Constants::NANOSECONDS_PER_SECOND;
    if (std::isnan(bandwidth) || std::isinf(bandwidth) || bandwidth < 0.0) {
      return 0.0;
    }
    return bandwidth;
  };

  const bool base_bytes_overflow =
      static_cast<size_t>(iterations) > std::numeric_limits<size_t>::max() / buffer_size;

  if (base_bytes_overflow) {
    const double total_bytes_read = static_cast<double>(iterations) * static_cast<double>(buffer_size);
    const double total_bytes_copied_op = total_bytes_read * Constants::COPY_OPERATION_MULTIPLIER;

    read_bw_gb_s = calculate_valid_bandwidth(total_bytes_read, read_time);
    write_bw_gb_s = calculate_valid_bandwidth(total_bytes_read, write_time);
    copy_bw_gb_s = calculate_valid_bandwidth(total_bytes_copied_op, copy_time);
    return;
  }

  const size_t total_bytes_read = static_cast<size_t>(iterations) * buffer_size;
  const bool copy_bytes_overflow =
      total_bytes_read > std::numeric_limits<size_t>::max() / Constants::COPY_OPERATION_MULTIPLIER;

  if (copy_bytes_overflow) {
    const double total_bytes_read_double = static_cast<double>(total_bytes_read);
    const double total_bytes_copied_op = total_bytes_read_double * Constants::COPY_OPERATION_MULTIPLIER;

    read_bw_gb_s = calculate_valid_bandwidth(total_bytes_read_double, read_time);
    write_bw_gb_s = calculate_valid_bandwidth(total_bytes_read_double, write_time);
    copy_bw_gb_s = calculate_valid_bandwidth(total_bytes_copied_op, copy_time);
    return;
  }

  const size_t total_bytes_copied_op = total_bytes_read * Constants::COPY_OPERATION_MULTIPLIER;

  read_bw_gb_s = calculate_valid_bandwidth(static_cast<double>(total_bytes_read), read_time);
  write_bw_gb_s = calculate_valid_bandwidth(static_cast<double>(total_bytes_read), write_time);
  copy_bw_gb_s = calculate_valid_bandwidth(static_cast<double>(total_bytes_copied_op), copy_time);
}

/**
 * @brief Calculate bandwidth results from timing data for all memory levels.
 *
 * Orchestrates bandwidth calculation for:
 * - Main memory (read, write, copy)
 * - Cache levels (L1/L2 or custom)
 *
 * Conditional calculation based on configuration:
 * - use_custom_cache_size: Calculate for custom cache buffer
 * - Otherwise: Calculate for L1 and/or L2 if buffer sizes > 0
 *
 * Cache iterations:
 * - Multiplied by CACHE_ITERATIONS_MULTIPLIER for better measurement accuracy
 * - Cache tests need more iterations due to faster access times
 *
 * @param[in]  config   Benchmark configuration with buffer sizes and flags
 * @param[in]  timings  Raw timing measurements from benchmark execution
 * @param[out] results  Benchmark results structure to populate with calculated bandwidths
 *
 * @note Main memory uses config.iterations directly.
 * @note Cache tests use config.iterations * CACHE_ITERATIONS_MULTIPLIER.
 * @note All bandwidth values in results are in GB/s.
 *
 * @see calculate_single_bandwidth() for the per-level calculation logic
 * @see BenchmarkConfig for configuration options
 * @see TimingResults for raw timing data structure
 * @see BenchmarkResults for output structure
 */
void calculate_bandwidth_results(const BenchmarkConfig& config, const TimingResults& timings,
                                 BenchmarkResults& results) {
  // Main memory bandwidth calculations
  set_bandwidth_measurements(config.buffer_size, config.iterations,
                             timings.total_read_time, timings.total_write_time,
                             timings.total_copy_time, timings.total_read_status,
                             timings.total_write_status, timings.total_copy_status,
                             results.main_read_bandwidth, results.main_write_bandwidth,
                             results.main_copy_bandwidth);
  
  // Cache bandwidth calculations
  int cache_iterations = calculate_cache_iterations_saturated(config.iterations);
  
  if (config.use_custom_cache_size) {
    if (config.custom_buffer_size > 0) {
      set_bandwidth_measurements(config.custom_buffer_size, cache_iterations,
                                 timings.custom_read_time, timings.custom_write_time,
                                 timings.custom_copy_time, timings.custom_read_status,
                                 timings.custom_write_status, timings.custom_copy_status,
                                 results.custom_read_bandwidth,
                                 results.custom_write_bandwidth,
                                 results.custom_copy_bandwidth);
    }
  } else {
    if (config.l1_buffer_size > 0) {
      set_bandwidth_measurements(config.l1_buffer_size, cache_iterations,
                                 timings.l1_read_time, timings.l1_write_time,
                                 timings.l1_copy_time, timings.l1_read_status,
                                 timings.l1_write_status, timings.l1_copy_status,
                                 results.l1_read_bandwidth,
                                 results.l1_write_bandwidth,
                                 results.l1_copy_bandwidth);
    }
    if (config.l2_buffer_size > 0) {
      set_bandwidth_measurements(config.l2_buffer_size, cache_iterations,
                                 timings.l2_read_time, timings.l2_write_time,
                                 timings.l2_copy_time, timings.l2_read_status,
                                 timings.l2_write_status, timings.l2_copy_status,
                                 results.l2_read_bandwidth,
                                 results.l2_write_bandwidth,
                                 results.l2_copy_bandwidth);
    }
  }
}

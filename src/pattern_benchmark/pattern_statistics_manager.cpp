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
 * @file pattern_statistics_manager.cpp
 * @brief Statistics collection and management for pattern benchmarks
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 *
 * This file implements the top-level coordinator for running multiple
 * pattern benchmark loops and collecting aggregated statistics. It manages
 * the execution of all pattern types across multiple iterations and stores
 * results for statistical analysis.
 *
 * Coordinates execution of:
 * - Multiple benchmark loops (user-configurable loop count)
 * - All pattern types (forward, reverse, strided, random)
 * - Result aggregation into PatternStatistics structure
 */
#include "pattern_benchmark/pattern_benchmark.h"
#include "core/config/config.h"
#include "core/signal/signal_handler.h"
#include "output/console/messages/messages_api.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>

namespace {

using PatternValuesMember = std::vector<double> PatternStatistics::*;

constexpr std::array<PatternValuesMember,
                     static_cast<size_t>(PatternKind::Count) *
                         static_cast<size_t>(PatternOperation::Count)>
    kPatternValueMembers = {
        &PatternStatistics::all_forward_read_bw,
        &PatternStatistics::all_forward_write_bw,
        &PatternStatistics::all_forward_copy_bw,
        &PatternStatistics::all_reverse_read_bw,
        &PatternStatistics::all_reverse_write_bw,
        &PatternStatistics::all_reverse_copy_bw,
        &PatternStatistics::all_strided_64_read_bw,
        &PatternStatistics::all_strided_64_write_bw,
        &PatternStatistics::all_strided_64_copy_bw,
        &PatternStatistics::all_strided_4096_read_bw,
        &PatternStatistics::all_strided_4096_write_bw,
        &PatternStatistics::all_strided_4096_copy_bw,
        &PatternStatistics::all_strided_16384_read_bw,
        &PatternStatistics::all_strided_16384_write_bw,
        &PatternStatistics::all_strided_16384_copy_bw,
        &PatternStatistics::all_strided_2mb_read_bw,
        &PatternStatistics::all_strided_2mb_write_bw,
        &PatternStatistics::all_strided_2mb_copy_bw,
        &PatternStatistics::all_random_read_bw,
        &PatternStatistics::all_random_write_bw,
        &PatternStatistics::all_random_copy_bw,
};

double& legacy_bandwidth(PatternResults& results, PatternKind kind,
                         PatternOperation operation) {
  switch (kind) {
    case PatternKind::SequentialForward:
      if (operation == PatternOperation::Read) return results.forward_read_bw;
      if (operation == PatternOperation::Write) return results.forward_write_bw;
      return results.forward_copy_bw;
    case PatternKind::SequentialReverse:
      if (operation == PatternOperation::Read) return results.reverse_read_bw;
      if (operation == PatternOperation::Write) return results.reverse_write_bw;
      return results.reverse_copy_bw;
    case PatternKind::Strided64:
      if (operation == PatternOperation::Read) return results.strided_64_read_bw;
      if (operation == PatternOperation::Write) return results.strided_64_write_bw;
      return results.strided_64_copy_bw;
    case PatternKind::Strided4096:
      if (operation == PatternOperation::Read) return results.strided_4096_read_bw;
      if (operation == PatternOperation::Write) return results.strided_4096_write_bw;
      return results.strided_4096_copy_bw;
    case PatternKind::Strided16384:
      if (operation == PatternOperation::Read) return results.strided_16384_read_bw;
      if (operation == PatternOperation::Write) return results.strided_16384_write_bw;
      return results.strided_16384_copy_bw;
    case PatternKind::Strided2MiB:
      if (operation == PatternOperation::Read) return results.strided_2mb_read_bw;
      if (operation == PatternOperation::Write) return results.strided_2mb_write_bw;
      return results.strided_2mb_copy_bw;
    case PatternKind::Random:
      if (operation == PatternOperation::Read) return results.random_read_bw;
      if (operation == PatternOperation::Write) return results.random_write_bw;
      return results.random_copy_bw;
    case PatternKind::Count:
      break;
  }
  return results.forward_read_bw;
}

}  // namespace

PatternMeasurement& get_pattern_measurement(PatternResults& results, PatternKind kind,
                                            PatternOperation operation) {
  return results.measurements[pattern_measurement_index(kind, operation)];
}

const PatternMeasurement& get_pattern_measurement(const PatternResults& results,
                                                  PatternKind kind,
                                                  PatternOperation operation) {
  return results.measurements[pattern_measurement_index(kind, operation)];
}

void set_pattern_measurement(PatternResults& results, PatternKind kind,
                             PatternOperation operation,
                             PatternMeasurement measurement) {
  legacy_bandwidth(results, kind, operation) =
      measurement.bandwidth_gb_s.value_or(0.0);
  get_pattern_measurement(results, kind, operation) = std::move(measurement);
}

PatternStatisticsData calculate_pattern_statistics(const std::vector<double>& values) {
  return calculate_descriptive_statistics(values);
}

void initialize_pattern_statistics(PatternStatistics& stats,
                                   size_t expected_loop_count) {
  stats.loop_results.clear();
  if (expected_loop_count > 0) {
    stats.loop_results.reserve(expected_loop_count);
  }
  for (PatternValuesMember member : kPatternValueMembers) {
    std::vector<double>& values = stats.*member;
    values.clear();
    if (expected_loop_count > 0) {
      values.reserve(expected_loop_count);
    }
  }
}

void collect_pattern_loop_result(PatternStatistics& stats,
                                 PatternResults loop_result) {
  for (size_t index = 0; index < loop_result.measurements.size(); ++index) {
    const PatternMeasurement& measurement = loop_result.measurements[index];
    if (measurement.status == PatternMeasurementStatus::Measured &&
        measurement.bandwidth_gb_s.has_value()) {
      (stats.*kPatternValueMembers[index])
          .push_back(*measurement.bandwidth_gb_s);
    }
  }
  stats.loop_results.push_back(std::move(loop_result));
}

// ============================================================================
// Public API Functions
// ============================================================================

int run_all_pattern_benchmarks(const BenchmarkBuffers& buffers,
                               const BenchmarkConfig& config,
                               PatternStatistics& stats) {
  initialize_pattern_statistics(
      stats, config.loop_count > 0 ? static_cast<size_t>(config.loop_count) : 0);

  std::cout << Messages::msg_running_pattern_benchmarks() << std::flush;

  // Main pattern benchmark loop
  for (int loop = 0; loop < config.loop_count; ++loop) {
    // Check for Ctrl+C between pattern loops
    if (signal_received()) {
      std::cout << std::endl << Messages::msg_interrupted_by_user() << std::endl;
      return EXIT_SUCCESS;
    }

    try {
      PatternResults loop_results;
      
      // Run pattern benchmarks for this loop
      int status = run_pattern_benchmarks(buffers, config, loop_results,
                                          static_cast<size_t>(loop));
      if (status != EXIT_SUCCESS) {
        return status;
      }
      
      collect_pattern_loop_result(stats, std::move(loop_results));
      
      // Print simple progress message for each loop
      if (config.loop_count > 1) {
        std::cout << '\r' << std::flush;  // Clear progress indicator
        std::cout << Messages::msg_pattern_benchmark_loop_completed(loop + 1, config.loop_count) << std::endl;
      }
    } catch (const std::exception &e) {
      std::cerr << Messages::error_benchmark_loop(loop, e.what()) << std::endl;
      return EXIT_FAILURE;
    }
  }
  
  return EXIT_SUCCESS;
}

PatternResults extract_pattern_results_at(const PatternStatistics& stats, size_t index) {
  PatternResults result;

  if (!stats.loop_results.empty()) {
    return stats.loop_results[std::min(index, stats.loop_results.size() - 1)];
  }

  if (stats.all_forward_read_bw.empty()) {
    return result;
  }

  if (index >= stats.all_forward_read_bw.size()) {
    index = stats.all_forward_read_bw.size() - 1;
  }

  result.forward_read_bw = stats.all_forward_read_bw[index];
  result.forward_write_bw = stats.all_forward_write_bw[index];
  result.forward_copy_bw = stats.all_forward_copy_bw[index];
  result.reverse_read_bw = stats.all_reverse_read_bw[index];
  result.reverse_write_bw = stats.all_reverse_write_bw[index];
  result.reverse_copy_bw = stats.all_reverse_copy_bw[index];
  result.strided_64_read_bw = stats.all_strided_64_read_bw[index];
  result.strided_64_write_bw = stats.all_strided_64_write_bw[index];
  result.strided_64_copy_bw = stats.all_strided_64_copy_bw[index];
  result.strided_4096_read_bw = stats.all_strided_4096_read_bw[index];
  result.strided_4096_write_bw = stats.all_strided_4096_write_bw[index];
  result.strided_4096_copy_bw = stats.all_strided_4096_copy_bw[index];
  result.strided_16384_read_bw = stats.all_strided_16384_read_bw[index];
  result.strided_16384_write_bw = stats.all_strided_16384_write_bw[index];
  result.strided_16384_copy_bw = stats.all_strided_16384_copy_bw[index];
  result.strided_2mb_read_bw = stats.all_strided_2mb_read_bw[index];
  result.strided_2mb_write_bw = stats.all_strided_2mb_write_bw[index];
  result.strided_2mb_copy_bw = stats.all_strided_2mb_copy_bw[index];
  result.random_read_bw = stats.all_random_read_bw[index];
  result.random_write_bw = stats.all_random_write_bw[index];
  result.random_copy_bw = stats.all_random_copy_bw[index];

  return result;
}

PatternResults extract_pattern_median_results(const PatternStatistics& stats) {
  PatternResults result;
  if (stats.loop_results.empty()) {
    return extract_pattern_results_at(stats, 0);
  }

  result = stats.loop_results.front();
  for (size_t kind_index = 0; kind_index < static_cast<size_t>(PatternKind::Count);
       ++kind_index) {
    const PatternKind kind = static_cast<PatternKind>(kind_index);
    for (size_t operation_index = 0;
         operation_index < static_cast<size_t>(PatternOperation::Count);
         ++operation_index) {
      const PatternOperation operation =
          static_cast<PatternOperation>(operation_index);
      std::vector<double> values;
      for (const PatternResults& loop_result : stats.loop_results) {
        const PatternMeasurement& measurement =
            get_pattern_measurement(loop_result, kind, operation);
        if (measurement.status == PatternMeasurementStatus::Measured &&
            measurement.bandwidth_gb_s.has_value()) {
          values.push_back(*measurement.bandwidth_gb_s);
        }
      }

      PatternMeasurement headline = get_pattern_measurement(result, kind, operation);
      if (!values.empty()) {
        headline.status = PatternMeasurementStatus::Measured;
        headline.status_reason.clear();
        headline.bandwidth_gb_s = calculate_pattern_statistics(values).median;
      } else {
        headline.bandwidth_gb_s.reset();
      }
      set_pattern_measurement(result, kind, operation, std::move(headline));
    }
  }
  return result;
}

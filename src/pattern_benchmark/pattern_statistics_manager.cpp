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
#include "core/memory/buffer_allocator.h"
#include "core/memory/buffer_initializer.h"
#include "core/memory/buffer_manager.h"
#include "core/signal/signal_handler.h"
#include "output/console/messages/messages_api.h"
#include "utils/numeric_utils.h"
#include "utils/utils.h"
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

void apply_pattern_loop_summary(PatternResults& results,
                                const PatternLoopSummary& summary) {
  results.status = summary.status;
  results.status_reason = summary.status_reason;
  results.planned_measurements = summary.planned_measurements;
  results.completed_measurements = summary.completed_measurements;
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
  get_pattern_measurement(results, kind, operation) = std::move(measurement);
}

PatternStatisticsData calculate_pattern_statistics(const std::vector<double>& values) {
  return calculate_descriptive_statistics(values);
}

const char* pattern_run_status_to_string(PatternRunStatus status) {
  switch (status) {
    case PatternRunStatus::NotStarted:
      return "not-started";
    case PatternRunStatus::Complete:
      return "complete";
    case PatternRunStatus::Partial:
      return "partial";
    case PatternRunStatus::Interrupted:
      return "interrupted";
    case PatternRunStatus::Failed:
      return "failed";
  }
  return "failed";
}

PatternLoopSummary summarize_pattern_loop(
    const PatternResults& results, bool execution_failed,
    bool interruption_requested,
    const std::string& execution_failure_reason) {
  PatternLoopSummary summary;
  const PatternMeasurement* first_invalid = nullptr;
  const PatternMeasurement* first_interrupted = nullptr;
  const PatternMeasurement* first_incomplete = nullptr;

  for (const PatternMeasurement& measurement : results.measurements) {
    const bool complete =
        (measurement.status == PatternMeasurementStatus::Measured &&
         measurement.bandwidth_gb_s.has_value()) ||
        measurement.status == PatternMeasurementStatus::Skipped;
    if (complete) {
      ++summary.completed_measurements;
      continue;
    }
    if (measurement.status == PatternMeasurementStatus::Invalid &&
        first_invalid == nullptr) {
      first_invalid = &measurement;
    } else if (measurement.status == PatternMeasurementStatus::Interrupted &&
               first_interrupted == nullptr) {
      first_interrupted = &measurement;
    }
    if (first_incomplete == nullptr) {
      first_incomplete = &measurement;
    }
  }

  if (execution_failed) {
    summary.status = PatternRunStatus::Failed;
    summary.status_reason =
        !execution_failure_reason.empty()
            ? execution_failure_reason
            : first_invalid != nullptr &&
                      !first_invalid->status_reason.empty()
                  ? first_invalid->status_reason
                  : Messages::pattern_reason_loop_execution_failed();
  } else if (first_invalid != nullptr) {
    summary.status = PatternRunStatus::Failed;
    summary.status_reason =
        first_invalid->status_reason.empty()
            ? Messages::pattern_reason_invalid_measurement()
            : first_invalid->status_reason;
  } else if (summary.completed_measurements ==
             summary.planned_measurements) {
    summary.status = PatternRunStatus::Complete;
  } else if (interruption_requested || first_interrupted != nullptr) {
    summary.status = PatternRunStatus::Interrupted;
    summary.status_reason =
        first_interrupted != nullptr &&
                !first_interrupted->status_reason.empty()
            ? first_interrupted->status_reason
            : Messages::pattern_reason_loop_interrupted();
  } else {
    summary.status = PatternRunStatus::Partial;
    summary.status_reason =
        first_incomplete != nullptr && !first_incomplete->status_reason.empty()
            ? first_incomplete->status_reason
            : Messages::pattern_reason_loop_incomplete();
  }
  return summary;
}

void initialize_pattern_statistics(PatternStatistics& stats,
                                   size_t expected_loop_count) {
  stats.status = PatternRunStatus::NotStarted;
  stats.status_reason.clear();
  stats.planned_loops = expected_loop_count;
  stats.completed_loops = 0;
  stats.completed_measurements = 0;
  stats.planned_measurements = NumericUtils::saturating_multiply(
      expected_loop_count, kPatternMeasurementsPerLoop);
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
  if (loop_result.status == PatternRunStatus::NotStarted) {
    apply_pattern_loop_summary(loop_result,
                               summarize_pattern_loop(loop_result));
  }
  if (loop_result.status == PatternRunStatus::Complete) {
    for (size_t index = 0; index < loop_result.measurements.size(); ++index) {
      const PatternMeasurement& measurement = loop_result.measurements[index];
      if (measurement.status == PatternMeasurementStatus::Measured &&
          measurement.bandwidth_gb_s.has_value()) {
        (stats.*kPatternValueMembers[index])
            .push_back(*measurement.bandwidth_gb_s);
      }
    }
  }
  stats.completed_measurements = NumericUtils::saturating_add(
      stats.completed_measurements, loop_result.completed_measurements);
  if (loop_result.status == PatternRunStatus::Complete) {
    ++stats.completed_loops;
  }

  if (loop_result.status == PatternRunStatus::Failed) {
    stats.status = PatternRunStatus::Failed;
    stats.status_reason = loop_result.status_reason;
  } else if (stats.status != PatternRunStatus::Failed &&
             loop_result.status == PatternRunStatus::Interrupted) {
    stats.status = PatternRunStatus::Interrupted;
    stats.status_reason = loop_result.status_reason;
  } else if (stats.status != PatternRunStatus::Failed &&
             stats.status != PatternRunStatus::Interrupted &&
             stats.completed_loops == stats.planned_loops &&
             stats.completed_measurements == stats.planned_measurements) {
    stats.status = PatternRunStatus::Complete;
    stats.status_reason.clear();
  } else if (stats.status != PatternRunStatus::Failed &&
             stats.status != PatternRunStatus::Interrupted) {
    stats.status = PatternRunStatus::Partial;
    if (loop_result.status == PatternRunStatus::Partial &&
        !loop_result.status_reason.empty()) {
      stats.status_reason = loop_result.status_reason;
    } else if (stats.status_reason.empty()) {
      stats.status_reason = Messages::pattern_reason_loops_remain();
    }
  }
  stats.loop_results.push_back(std::move(loop_result));
}

// ============================================================================
// Public API Functions
// ============================================================================

namespace {

int run_all_pattern_benchmarks_impl(
    const BenchmarkConfig& config, PatternStatistics& stats,
    const PatternRunnerTestHooks* test_hooks) {
  initialize_pattern_statistics(
      stats, config.loop_count > 0 ? static_cast<size_t>(config.loop_count) : 0);

  PatternBuffers buffers;
  const int allocation_status =
      test_hooks != nullptr && test_hooks->allocate_buffers
          ? test_hooks->allocate_buffers(config, buffers)
          : allocate_pattern_buffers(config, buffers);
  if (allocation_status != EXIT_SUCCESS) {
    stats.status = PatternRunStatus::Failed;
    stats.status_reason = Messages::pattern_reason_buffers_allocation_failed();
    clear_progress();
    return EXIT_FAILURE;
  }
  const int initialization_status =
      test_hooks != nullptr && test_hooks->initialize_buffers
          ? test_hooks->initialize_buffers(buffers, config.buffer_size)
          : initialize_pattern_buffers(buffers, config.buffer_size);
  if (initialization_status != EXIT_SUCCESS) {
    stats.status = PatternRunStatus::Failed;
    stats.status_reason =
        Messages::pattern_reason_buffers_initialization_failed();
    clear_progress();
    return EXIT_FAILURE;
  }

  const auto stop_requested = [test_hooks]() {
    return test_hooks != nullptr && test_hooks->stop_requested
               ? test_hooks->stop_requested()
               : signal_received();
  };

  std::cout << Messages::msg_running_pattern_benchmarks() << std::flush;

  // Main pattern benchmark loop
  for (int loop = 0; loop < config.loop_count; ++loop) {
    // Check for Ctrl+C between pattern loops
    if (stop_requested()) {
      stats.status = PatternRunStatus::Interrupted;
      stats.status_reason = Messages::pattern_reason_loop_interrupted();
      clear_progress();
      std::cout << std::endl << Messages::msg_interrupted_by_user() << std::endl;
      return EXIT_SUCCESS;
    }

    PatternResults loop_results;
    loop_results.loop_index = static_cast<size_t>(loop);
    int status = EXIT_FAILURE;
    try {
      status =
          test_hooks != nullptr && test_hooks->execute_loop
              ? test_hooks->execute_loop(buffers, config, loop_results,
                                         static_cast<size_t>(loop))
              : run_pattern_benchmarks(buffers, config, loop_results,
                                       static_cast<size_t>(loop));
    } catch (const std::exception &e) {
      loop_results.loop_index = static_cast<size_t>(loop);
      apply_pattern_loop_summary(
          loop_results,
          summarize_pattern_loop(
              loop_results, true, false,
              Messages::pattern_reason_loop_exception(e.what())));
      collect_pattern_loop_result(stats, std::move(loop_results));
      clear_progress();
      std::cerr << Messages::error_benchmark_loop(loop, e.what()) << std::endl;
      return EXIT_FAILURE;
    } catch (...) {
      loop_results.loop_index = static_cast<size_t>(loop);
      apply_pattern_loop_summary(
          loop_results,
          summarize_pattern_loop(
              loop_results, true, false,
              Messages::pattern_reason_unknown_loop_exception()));
      collect_pattern_loop_result(stats, std::move(loop_results));
      clear_progress();
      std::cerr << Messages::error_prefix()
                << Messages::pattern_reason_unknown_loop_exception()
                << std::endl;
      return EXIT_FAILURE;
    }

    loop_results.loop_index = static_cast<size_t>(loop);
    const bool interrupted_after_execution = stop_requested();
    apply_pattern_loop_summary(
        loop_results,
        summarize_pattern_loop(
            loop_results, status != EXIT_SUCCESS,
            interrupted_after_execution));
    collect_pattern_loop_result(stats, std::move(loop_results));

    if (status != EXIT_SUCCESS) {
      clear_progress();
      return EXIT_FAILURE;
    }
    if (stats.loop_results.back().status == PatternRunStatus::Failed) {
      clear_progress();
      return EXIT_FAILURE;
    }
    if (interrupted_after_execution &&
        stats.loop_results.back().status == PatternRunStatus::Complete &&
        loop + 1 < config.loop_count) {
      stats.status = PatternRunStatus::Interrupted;
      stats.status_reason = Messages::pattern_reason_loop_interrupted();
      clear_progress();
      std::cout << std::endl << Messages::msg_interrupted_by_user()
                << std::endl;
      return EXIT_SUCCESS;
    }
    if (stats.loop_results.back().status == PatternRunStatus::Interrupted) {
      clear_progress();
      std::cout << std::endl << Messages::msg_interrupted_by_user()
                << std::endl;
      return EXIT_SUCCESS;
    }

    // Print simple progress message for each loop
    if (config.loop_count > 1) {
      clear_progress();
      std::cout
          << Messages::msg_pattern_benchmark_loop_completed(loop + 1,
                                                             config.loop_count)
          << std::endl;
    }
  }

  if (stats.completed_loops == stats.planned_loops &&
      stats.completed_measurements == stats.planned_measurements) {
    stats.status = PatternRunStatus::Complete;
    stats.status_reason.clear();
  } else if (stats.status != PatternRunStatus::Failed &&
             stats.status != PatternRunStatus::Interrupted) {
    stats.status = PatternRunStatus::Partial;
    if (stats.status_reason.empty()) {
      stats.status_reason = Messages::pattern_reason_loops_remain();
    }
  }
  clear_progress();
  return EXIT_SUCCESS;
}

}  // namespace

int run_all_pattern_benchmarks(const BenchmarkConfig& config,
                               PatternStatistics& stats,
                               const PatternRunnerTestHooks* test_hooks) {
  try {
    return run_all_pattern_benchmarks_impl(config, stats, test_hooks);
  } catch (const std::exception& e) {
    stats.status = PatternRunStatus::Failed;
    stats.status_reason = Messages::pattern_reason_coordinator_exception(e.what());
    clear_progress();
    std::cerr << Messages::error_prefix() << stats.status_reason << std::endl;
    return EXIT_FAILURE;
  } catch (...) {
    stats.status = PatternRunStatus::Failed;
    stats.status_reason =
        Messages::pattern_reason_unknown_coordinator_exception();
    clear_progress();
    std::cerr << Messages::error_prefix() << stats.status_reason << std::endl;
    return EXIT_FAILURE;
  }
}

PatternResults extract_pattern_results_at(const PatternStatistics& stats, size_t index) {
  PatternResults result;

  return stats.loop_results.empty()
             ? result
             : stats.loop_results[std::min(index,
                                           stats.loop_results.size() - 1)];
}

PatternResults extract_pattern_median_results(const PatternStatistics& stats) {
  PatternResults result;
  if (stats.loop_results.empty()) {
    return extract_pattern_results_at(stats, 0);
  }

  const auto complete_loop =
      std::find_if(stats.loop_results.begin(), stats.loop_results.end(),
                   [](const PatternResults& loop_result) {
                     return loop_result.status == PatternRunStatus::Complete;
                   });
  if (complete_loop == stats.loop_results.end()) {
    return stats.loop_results.front();
  }
  result = *complete_loop;
  for (size_t kind_index = 0; kind_index < static_cast<size_t>(PatternKind::Count);
       ++kind_index) {
    const PatternKind kind = static_cast<PatternKind>(kind_index);
    for (size_t operation_index = 0;
         operation_index < static_cast<size_t>(PatternOperation::Count);
         ++operation_index) {
      const PatternOperation operation =
          static_cast<PatternOperation>(operation_index);
      std::vector<double> values;
      const PatternMeasurement* representative = nullptr;
      for (const PatternResults& loop_result : stats.loop_results) {
        if (loop_result.status != PatternRunStatus::Complete) {
          continue;
        }
        const PatternMeasurement& measurement =
            get_pattern_measurement(loop_result, kind, operation);
        if (representative == nullptr ||
            (representative->status != PatternMeasurementStatus::Measured &&
             measurement.status == PatternMeasurementStatus::Measured)) {
          representative = &measurement;
        }
        if (measurement.status == PatternMeasurementStatus::Measured &&
            measurement.bandwidth_gb_s.has_value()) {
          values.push_back(*measurement.bandwidth_gb_s);
        }
      }

      PatternMeasurement headline = representative != nullptr
                                        ? *representative
                                        : get_pattern_measurement(result, kind,
                                                                  operation);
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

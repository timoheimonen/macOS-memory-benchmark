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
 * @file patterns.cpp
 * @brief JSON output generation for pattern benchmark results
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2025
 *
 * This file builds the JSON structure for pattern benchmark results including
 * sequential (forward/reverse), strided (64B/4096B), and random access patterns.
 * Each pattern includes bandwidth measurements (read/write/copy) with values
 * and optional statistical aggregation.
 */
// This file uses the nlohmann/json library for JSON parsing and generation.
// Library: https://github.com/nlohmann/json
// License: MIT License
//
#include "output/json/json_output/json_output_api.h"
#include "core/config/constants.h"
#include "pattern_benchmark/pattern_benchmark.h" // For PatternStatistics
#include "third_party/nlohmann/json.hpp"   // JSON library

#include <string>
#include <vector>

namespace {

nlohmann::json serialize_pattern_measurement(const PatternMeasurement& measurement) {
  nlohmann::json output;
  output["status"] = pattern_measurement_status_to_string(measurement.status);
  output["reason"] = measurement.status_reason;
  output["value_gb_s"] = measurement.bandwidth_gb_s.has_value()
                              ? nlohmann::json(*measurement.bandwidth_gb_s)
                              : nlohmann::json(nullptr);
  output["elapsed_seconds"] = measurement.elapsed_seconds;
  output["pilot_elapsed_seconds"] = measurement.pilot_elapsed_seconds;
  output["automatic_calibration"] = measurement.automatic_calibration;
  output["access_size_bytes"] = measurement.access_size_bytes;
  output["stride_bytes"] = measurement.stride_bytes == 0
                                ? nlohmann::json(nullptr)
                                : nlohmann::json(measurement.stride_bytes);
  output["requested_threads"] = measurement.requested_threads;
  output["effective_threads"] = measurement.effective_threads;
  output["accesses_per_pass"] = measurement.accesses_per_pass;
  output["accesses_per_pass_semantics"] =
      measurement.stride_bytes > 0 ? "phase-zero-count" : "constant-count";
  output["min_accesses_per_pass"] = measurement.min_accesses_per_pass;
  output["max_accesses_per_pass"] = measurement.max_accesses_per_pass;
  output["passes"] = measurement.passes;
  output["total_accesses"] = measurement.total_accesses;
  output["total_payload_bytes"] = measurement.total_payload_bytes;
  output["distinct_address_count"] = measurement.distinct_address_count;
  output["logical_working_set_bytes"] = measurement.logical_working_set_bytes;
  output["completed_phase_cycles"] = measurement.completed_phase_cycles;
  output["phase_period_passes"] = measurement.phase_period_passes;
  output["native_page_size_bytes"] = measurement.native_page_size_bytes;
  output["stride_equals_native_page_size"] =
      measurement.stride_equals_native_page_size;
  output["large_page_backing_verified"] = measurement.large_page_backing_verified;
  output["benchmark_loop_index"] = measurement.benchmark_loop_index;
  output["pattern_order_index"] = measurement.pattern_order_index;
  output["seed"] = measurement.has_seed
                       ? nlohmann::json(std::to_string(measurement.seed))
                       : nlohmann::json(nullptr);
  return output;
}

nlohmann::json serialize_pattern_statistics(const std::vector<double>& values) {
  if (values.empty()) return nullptr;
  const PatternStatisticsData statistics = calculate_pattern_statistics(values);
  return {{"mean", statistics.average},
          {"median_p50", statistics.median},
          {"p90", statistics.p90},
          {"p95", statistics.p95},
          {"p99", statistics.p99},
          {"stddev", statistics.stddev},
          {"coefficient_of_variation_pct",
           statistics.coefficient_of_variation_pct},
          {"median_absolute_deviation",
           statistics.median_absolute_deviation},
          {"min", statistics.min},
          {"max", statistics.max}};
}

nlohmann::json build_operation_json(const PatternStatistics& stats, PatternKind kind,
                                    PatternOperation operation,
                                    const std::vector<double>& fallback_values) {
  std::vector<double> values;
  nlohmann::json samples = nlohmann::json::array();
  const PatternMeasurement* representative = nullptr;
  for (const PatternResults& loop_result : stats.loop_results) {
    const PatternMeasurement& measurement =
        get_pattern_measurement(loop_result, kind, operation);
    samples.push_back(serialize_pattern_measurement(measurement));
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
  if (stats.loop_results.empty()) {
    values = fallback_values;
  }

  nlohmann::json output;
  output["status"] = !values.empty()
                         ? "measured"
                         : representative != nullptr
                               ? pattern_measurement_status_to_string(representative->status)
                               : "not-run";
  output["reason"] = !values.empty() || representative == nullptr
                         ? ""
                         : representative->status_reason;
  output["headline"] = values.empty()
                           ? "none"
                           : values.size() > 1 ? "median_p50"
                                               : "single_measurement";
  output["value_gb_s"] = values.empty()
                             ? nlohmann::json(nullptr)
                             : nlohmann::json(
                                   calculate_pattern_statistics(values).median);
  output["values_gb_s"] = values;
  output["statistics"] = serialize_pattern_statistics(values);
  output["measurements"] = std::move(samples);
  return output;
}

nlohmann::json build_pattern_json(
    const PatternStatistics& stats, PatternKind kind,
    const std::vector<double>& read_values,
    const std::vector<double>& write_values,
    const std::vector<double>& copy_values) {
  nlohmann::json output;
  const PatternMeasurement* representative = nullptr;
  for (const PatternResults& loop_result : stats.loop_results) {
    for (PatternOperation operation : {PatternOperation::Read,
                                       PatternOperation::Write,
                                       PatternOperation::Copy}) {
      const PatternMeasurement& candidate =
          get_pattern_measurement(loop_result, kind, operation);
      if (representative == nullptr ||
          (representative->status != PatternMeasurementStatus::Measured &&
           candidate.status == PatternMeasurementStatus::Measured)) {
        representative = &candidate;
      }
    }
  }

  output["methodology_version"] = Constants::PATTERN_METHODOLOGY_VERSION;
  output["warmup_semantics"] = "steady-state-same-shape";
  output["access_size_bytes"] = representative != nullptr
                                    ? representative->access_size_bytes
                                    : 0;
  output["stride_bytes"] = representative == nullptr ||
                                    representative->stride_bytes == 0
                                ? nlohmann::json(nullptr)
                                : nlohmann::json(representative->stride_bytes);
  output["requested_threads"] = representative != nullptr
                                    ? representative->requested_threads
                                    : 0;
  output["effective_threads"] = representative != nullptr
                                    ? representative->effective_threads
                                    : 0;
  output["native_page_size_bytes"] = representative != nullptr
                                         ? representative->native_page_size_bytes
                                         : 0;
  output["stride_equals_native_page_size"] =
      representative != nullptr && representative->stride_equals_native_page_size;
  output["large_page_backing_verified"] =
      representative != nullptr && representative->large_page_backing_verified;
  output["large_page_backing_status"] =
      kind == PatternKind::Strided2MiB ? "not-verified" : "not-applicable";
  output["seed"] = representative != nullptr && representative->has_seed
                       ? nlohmann::json(std::to_string(representative->seed))
                       : nlohmann::json(nullptr);

  output[JsonKeys::BANDWIDTH] = {
      {JsonKeys::READ_GB_S,
       build_operation_json(stats, kind, PatternOperation::Read, read_values)},
      {JsonKeys::WRITE_GB_S,
       build_operation_json(stats, kind, PatternOperation::Write, write_values)},
      {JsonKeys::COPY_GB_S,
       build_operation_json(stats, kind, PatternOperation::Copy, copy_values)}};
  return output;
}

}  // namespace

// Build patterns JSON object from PatternStatistics
nlohmann::json build_patterns_json(const PatternStatistics& stats) {
  nlohmann::json patterns;

  const bool has_results = !stats.loop_results.empty() ||
                           !stats.all_forward_read_bw.empty();
  if (!has_results) return patterns;

  patterns[JsonKeys::SEQUENTIAL_FORWARD] = build_pattern_json(
      stats, PatternKind::SequentialForward, stats.all_forward_read_bw,
      stats.all_forward_write_bw, stats.all_forward_copy_bw);
  patterns[JsonKeys::SEQUENTIAL_REVERSE] = build_pattern_json(
      stats, PatternKind::SequentialReverse, stats.all_reverse_read_bw,
      stats.all_reverse_write_bw, stats.all_reverse_copy_bw);
  patterns[JsonKeys::STRIDED_64] = build_pattern_json(
      stats, PatternKind::Strided64, stats.all_strided_64_read_bw,
      stats.all_strided_64_write_bw, stats.all_strided_64_copy_bw);
  patterns[JsonKeys::STRIDED_4096] = build_pattern_json(
      stats, PatternKind::Strided4096, stats.all_strided_4096_read_bw,
      stats.all_strided_4096_write_bw, stats.all_strided_4096_copy_bw);
  patterns[JsonKeys::STRIDED_16384] = build_pattern_json(
      stats, PatternKind::Strided16384, stats.all_strided_16384_read_bw,
      stats.all_strided_16384_write_bw, stats.all_strided_16384_copy_bw);
  patterns[JsonKeys::STRIDED_2MB] = build_pattern_json(
      stats, PatternKind::Strided2MiB, stats.all_strided_2mb_read_bw,
      stats.all_strided_2mb_write_bw, stats.all_strided_2mb_copy_bw);
  patterns[JsonKeys::RANDOM] = build_pattern_json(
      stats, PatternKind::Random, stats.all_random_read_bw,
      stats.all_random_write_bw, stats.all_random_copy_bw);
  
  return patterns;
}

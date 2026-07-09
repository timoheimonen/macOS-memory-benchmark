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
 * @file tlb_analysis_json.cpp
 * @brief JSON serialization helpers for standalone TLB analysis mode
 */

#include "benchmark/tlb_analysis_json.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

#include "benchmark/tlb_analysis.h"
#include "core/config/config.h"
#include "core/config/constants.h"
#include "core/config/version.h"
#include "core/memory/memory_utils.h"
#include "output/json/json_output/json_output_api.h"

namespace {

nlohmann::ordered_json build_tlb_boundary_json(const TlbBoundaryDetection& boundary,
                                               size_t inferred_entries) {
  nlohmann::ordered_json boundary_json;
  boundary_json["detected"] = boundary.detected;
  if (!boundary.detected) {
    return boundary_json;
  }

  boundary_json["segment_start_index"] = boundary.segment_start_index;
  boundary_json["boundary_index"] = boundary.boundary_index;
  boundary_json["boundary_locality_bytes"] = boundary.boundary_locality_bytes;
  boundary_json["boundary_locality_kb"] = boundary.boundary_locality_bytes / Constants::BYTES_PER_KB;
  boundary_json["baseline_ns"] = boundary.baseline_ns;
  boundary_json["boundary_latency_ns"] = boundary.boundary_latency_ns;
  boundary_json["step_ns"] = boundary.step_ns;
  boundary_json["step_percent"] = boundary.step_percent;
  boundary_json["persistent_jump"] = boundary.persistent_jump;
  boundary_json["overlaps_private_cache_knee"] = boundary.overlaps_private_cache_knee;
  boundary_json["confidence"] = boundary.confidence;
  boundary_json["inferred_entries"] = inferred_entries;
  boundary_json["inferred_entries_method"] = "range_midpoint";
  return boundary_json;
}

nlohmann::ordered_json build_private_cache_knee_json(const PrivateCacheKneeDetection& knee,
                                                     bool interference_elevated,
                                                     size_t distance_to_l1_bytes,
                                                     size_t distance_to_l1_pages) {
  nlohmann::ordered_json knee_json;
  knee_json["detected"] = knee.detected;
  if (!knee.detected) {
    return knee_json;
  }

  knee_json["boundary_index"] = knee.boundary_index;
  knee_json["boundary_locality_bytes"] = knee.boundary_locality_bytes;
  knee_json["boundary_locality_kb"] = knee.boundary_locality_bytes / Constants::BYTES_PER_KB;
  knee_json["step_ns"] = knee.step_ns;
  knee_json["step_percent"] = knee.step_percent;
  knee_json["confidence"] = knee.confidence;
  knee_json["candidate_type"] = knee.strong_private_cache_candidate ? "strong_private_cache" : "early_cache";
  knee_json["may_interfere_with_tlb"] = knee.may_interfere_with_tlb;
  knee_json["interference_elevated"] = interference_elevated;
  knee_json["distance_to_l1_bytes"] = distance_to_l1_bytes;
  knee_json["distance_to_l1_pages"] = distance_to_l1_pages;
  return knee_json;
}

std::string build_utc_timestamp() {
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);
  std::tm utc_time;
  gmtime_r(&time_t, &utc_time);
  std::ostringstream timestamp_str;
  timestamp_str << std::put_time(&utc_time, "%Y-%m-%dT%H:%M:%SZ");
  return timestamp_str.str();
}

}  // namespace

int save_tlb_analysis_to_json(const TlbAnalysisJsonContext& context) {
  if (context.config.output_file.empty()) {
    return EXIT_SUCCESS;
  }

  const LatencyChainMode effective_chain_mode =
      resolve_latency_chain_mode(context.config.latency_chain_mode,
                                 context.page_walk_baseline_locality_bytes);

  nlohmann::ordered_json json_output;
  json_output[JsonKeys::CONFIGURATION] = {
      {JsonKeys::MODE, Constants::TLB_ANALYSIS_JSON_MODE_NAME},
      {"schema_version", 2},
      {"methodology_version", "locality-sweep-v2-balanced-rounds"},
      {JsonKeys::CPU_NAME, context.cpu_name},
      {JsonKeys::MACOS_VERSION, context.config.macos_version},
      {JsonKeys::PERFORMANCE_CORES, context.perf_cores},
      {JsonKeys::EFFICIENCY_CORES, context.eff_cores},
      {JsonKeys::PAGE_SIZE_BYTES, context.page_size_bytes},
      {JsonKeys::L1_CACHE_SIZE_BYTES, context.l1_cache_size_bytes},
      {JsonKeys::LATENCY_STRIDE_BYTES, context.stride_bytes},
      {"latency_chain_mode_requested", latency_chain_mode_to_string(context.config.latency_chain_mode)},
      {JsonKeys::LATENCY_CHAIN_MODE, latency_chain_mode_to_string(effective_chain_mode)},
      {JsonKeys::TLB_DENSITY, tlb_sweep_density_to_string(context.config.tlb_sweep_density)},
      {"seed", context.config.tlb_seed},
      {"seed_source", context.config.user_specified_tlb_seed ? "user" : "generated"},
      {"schedule_policy", "seeded-cyclic-latin"},
      {JsonKeys::LATENCY_SAMPLE_COUNT, static_cast<int>(context.loops_per_point)},
      {"accesses_per_loop", context.accesses_per_loop},
      {"tlb_guard_bytes", context.tlb_guard_bytes},
      {"buffer_size_mb", context.selected_buffer_mb},
      {"buffer_locked", context.buffer_locked},
      {"fine_sweep_added_points", context.fine_sweep_added_points}};
  json_output[JsonKeys::EXECUTION_TIME_SEC] = context.total_execution_time_sec;

  nlohmann::ordered_json sweep_json = nlohmann::ordered_json::array();
  for (size_t i = 0; i < context.localities_bytes.size(); ++i) {
    nlohmann::ordered_json point;
    point["locality_bytes"] = context.localities_bytes[i];
    point["locality_kb"] = context.localities_bytes[i] / Constants::BYTES_PER_KB;
    const auto planned_point = std::find_if(
        context.sweep_points.begin(),
        context.sweep_points.end(),
        [&](const TlbSweepPoint& candidate) {
          return candidate.locality_bytes == context.localities_bytes[i];
        });
    if (planned_point != context.sweep_points.end()) {
      point["point_index"] = planned_point->point_index;
      point["requested_pages"] = planned_point->requested_pages;
      point["effective_pages"] = planned_point->effective_pages;
      point["pointer_count"] = planned_point->pointer_count;
      point["stride_bytes"] = planned_point->stride_bytes;
      point["refinement_source"] = planned_point->refinement_source;
      point["bracket_lower_bytes"] = planned_point->bracket_lower_bytes;
      point["bracket_upper_bytes"] = planned_point->bracket_upper_bytes;
    }
    point["loop_latencies_ns"] = context.sweep_loop_latencies_ns[i];
    point["p50_latency_ns"] = context.p50_latency_ns[i];
    point["measurements"] = nlohmann::ordered_json::array();
    for (const TlbMeasurementRecord& record : context.measurement_records) {
      if (record.pass == TlbMeasurementPass::LargeLocality ||
          record.locality_bytes != context.localities_bytes[i]) {
        continue;
      }
      point["measurements"].push_back({
          {"pass", tlb_measurement_pass_to_string(record.pass)},
          {"round_index", record.round_index},
          {"order_index", record.order_index},
          {"seed", record.seed},
          {"latency_ns", record.latency_ns},
      });
    }
    sweep_json.push_back(point);
  }

  nlohmann::ordered_json tlb_json;
  tlb_json["status"] = context.analysis_status;
  tlb_json["planned_points"] = context.planned_points;
  tlb_json["measured_points"] = context.measured_points;
  tlb_json["conclusions_valid"] = context.conclusions_valid;
  const size_t planned_measurements =
      context.loops_per_point != 0 &&
              context.planned_points >
                  std::numeric_limits<size_t>::max() / context.loops_per_point
          ? std::numeric_limits<size_t>::max()
          : context.planned_points * context.loops_per_point;
  const size_t completed_measurements = static_cast<size_t>(std::count_if(
      context.measurement_records.begin(),
      context.measurement_records.end(),
      [](const TlbMeasurementRecord& record) {
        return record.pass != TlbMeasurementPass::LargeLocality;
      }));
  tlb_json["rounds_per_pass"] = context.loops_per_point;
  tlb_json["planned_measurements"] = planned_measurements;
  tlb_json["completed_measurements"] = completed_measurements;
  tlb_json["measurement_records"] = nlohmann::ordered_json::array();
  for (const TlbMeasurementRecord& record : context.measurement_records) {
    tlb_json["measurement_records"].push_back({
        {"pass", tlb_measurement_pass_to_string(record.pass)},
        {"point_index", record.point_index},
        {"locality_bytes", record.locality_bytes},
        {"round_index", record.round_index},
        {"order_index", record.order_index},
        {"seed", record.seed},
        {"latency_ns", record.latency_ns},
    });
  }
  tlb_json["sweep"] = sweep_json;
  if (context.conclusions_valid) {
    tlb_json["l1_tlb_detection"] = build_tlb_boundary_json(context.l1_boundary, context.l1_entries);
    tlb_json["l2_tlb_detection"] = build_tlb_boundary_json(context.l2_boundary, context.l2_entries);
    tlb_json["private_cache_knee"] = build_private_cache_knee_json(context.private_cache_knee,
                                                                    context.private_cache_interference_elevated,
                                                                    context.private_cache_to_l1_distance_bytes,
                                                                    context.private_cache_to_l1_distance_pages);
  } else {
    const nlohmann::ordered_json suppressed = {
        {"detected", false},
        {"reason", "analysis incomplete; conclusions suppressed"}};
    tlb_json["l1_tlb_detection"] = suppressed;
    tlb_json["l2_tlb_detection"] = suppressed;
    tlb_json["private_cache_knee"] = suppressed;
  }

  if (context.conclusions_valid && context.l1_boundary.detected) {
    tlb_json["l1_tlb_detection"]["inferred_entries_min"] = context.l1_entries_min;
    tlb_json["l1_tlb_detection"]["inferred_entries_max"] = context.l1_entries_max;
  }
  if (context.conclusions_valid && context.l2_boundary.detected) {
    tlb_json["l2_tlb_detection"]["inferred_entries_min"] = context.l2_entries_min;
    tlb_json["l2_tlb_detection"]["inferred_entries_max"] = context.l2_entries_max;
  }

  const bool large_locality_delta_available =
      context.conclusions_valid && context.page_walk_comparison_completed;
  nlohmann::ordered_json large_locality_delta = {
      {"available", large_locality_delta_available},
      {"baseline_locality_kb", context.page_walk_baseline_locality_bytes / Constants::BYTES_PER_KB},
      {"comparison_locality_mb", context.page_walk_comparison_locality_bytes / Constants::BYTES_PER_MB},
      {"baseline_p50_ns", context.page_walk_baseline_ns},
      {"interpretation", "large-locality latency delta; not an isolated page-table-walk cost"}};

  if (large_locality_delta_available) {
    large_locality_delta["comparison_loop_latencies_ns"] =
        context.page_walk_comparison_loop_latencies_ns;
    large_locality_delta["comparison_p50_ns"] = context.page_walk_comparison_p50_ns;
    large_locality_delta["delta_ns"] = context.page_walk_penalty_ns;
    large_locality_delta["measurements"] = nlohmann::ordered_json::array();
    for (const TlbMeasurementRecord& record : context.measurement_records) {
      if (record.pass != TlbMeasurementPass::LargeLocality) {
        continue;
      }
      large_locality_delta["measurements"].push_back({
          {"round_index", record.round_index},
          {"order_index", record.order_index},
          {"seed", record.seed},
          {"latency_ns", record.latency_ns},
      });
    }
  } else if (!context.conclusions_valid) {
    large_locality_delta["reason"] = "analysis incomplete; delta suppressed";
  } else if (!context.can_measure_page_walk_penalty) {
    large_locality_delta["reason"] = "requires at least 512 MB analysis buffer";
  } else {
    large_locality_delta["reason"] = "comparison measurement did not complete";
  }
  tlb_json["large_locality_latency_delta"] = large_locality_delta;

  nlohmann::ordered_json deprecated_page_walk_penalty = large_locality_delta;
  deprecated_page_walk_penalty["deprecated"] = true;
  deprecated_page_walk_penalty["replacement"] = "large_locality_latency_delta";
  if (large_locality_delta_available) {
    deprecated_page_walk_penalty.erase("delta_ns");
    deprecated_page_walk_penalty["penalty_ns"] = context.page_walk_penalty_ns;
  }
  tlb_json["page_walk_penalty"] = deprecated_page_walk_penalty;

  json_output["tlb_analysis"] = tlb_json;
  json_output[JsonKeys::TIMESTAMP] = build_utc_timestamp();
  json_output[JsonKeys::VERSION] = SOFTVERSION;

  std::filesystem::path file_path(context.config.output_file);
  if (file_path.is_relative()) {
    file_path = std::filesystem::current_path() / file_path;
  }

  return write_json_to_file(file_path, json_output);
}

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

nlohmann::ordered_json build_tlb_evidence_json(
    const TlbBoundaryEvidence& evidence) {
  return {
      {"available", evidence.available},
      {"passed", evidence.passed},
      {"effect_ns", evidence.effect_ns},
      {"minimum_effect_ns", evidence.minimum_effect_ns},
      {"noise_floor_ns", evidence.noise_floor_ns},
      {"effect_ci_95_ns",
       {{"lower", evidence.effect_ci.lower_ns},
        {"upper", evidence.effect_ci.upper_ns},
        {"paired_sample_count", evidence.effect_ci.paired_sample_count},
        {"bootstrap_resamples", evidence.effect_ci.bootstrap_resamples}}},
      {"persistence_points_passed", evidence.persistence_points_passed},
      {"persistence_points_required", evidence.persistence_points_required},
      {"rejection_reason", evidence.rejection_reason},
  };
}

nlohmann::ordered_json build_tlb_candidate_json(
    const TlbBoundaryCandidate& candidate) {
  return {
      {"accepted", candidate.accepted},
      {"boundary_index", candidate.boundary_index},
      {"boundary_locality_bytes", candidate.boundary_locality_bytes},
      {"bracket_lower_bytes", candidate.bracket_lower_bytes},
      {"bracket_upper_bytes", candidate.bracket_upper_bytes},
      {"discovery", build_tlb_evidence_json(candidate.discovery)},
      {"validation", build_tlb_evidence_json(candidate.validation)},
  };
}

nlohmann::ordered_json build_tlb_boundary_json(const TlbBoundaryDetection& boundary,
                                               size_t inferred_entries) {
  nlohmann::ordered_json boundary_json;
  boundary_json["detected"] = boundary.detected;
  boundary_json["signal"] = "translation_delta_ns";
  boundary_json["candidates"] = nlohmann::ordered_json::array();
  for (const TlbBoundaryCandidate& candidate : boundary.candidates) {
    boundary_json["candidates"].push_back(
        build_tlb_candidate_json(candidate));
  }
  if (!boundary.detected) {
    boundary_json["reason"] =
        "no candidate passed discovery and independent validation";
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
  boundary_json["bracket_lower_bytes"] = boundary.bracket_lower_bytes;
  boundary_json["bracket_upper_bytes"] = boundary.bracket_upper_bytes;
  boundary_json["overlaps_private_cache_knee"] = boundary.overlaps_private_cache_knee;
  boundary_json["confidence"] = boundary.confidence;
  boundary_json["discovery"] = build_tlb_evidence_json(boundary.discovery);
  boundary_json["validation"] = build_tlb_evidence_json(boundary.validation);
  boundary_json["inferred_entries"] = inferred_entries;
  boundary_json["inferred_entries_method"] =
      "validated-bracket-range-midpoint-estimate";
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

double median_values(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const size_t midpoint = values.size() / 2;
  if ((values.size() % 2) == 0) {
    return 0.5 * (values[midpoint - 1] + values[midpoint]);
  }
  return values[midpoint];
}

nlohmann::ordered_json build_tlb_chain_diagnostics_json(
    const TlbChainDiagnostics& diagnostics) {
  return {
      {"layout", tlb_chain_layout_to_string(diagnostics.layout)},
      {"traversal_policy",
       tlb_chain_traversal_policy_to_string(diagnostics.traversal_policy)},
      {"requested_pages", diagnostics.requested_pages},
      {"actual_pages", diagnostics.actual_pages},
      {"node_count", diagnostics.node_count},
      {"unique_cache_lines", diagnostics.unique_cache_lines},
      {"max_nodes_per_page", diagnostics.max_nodes_per_page},
      {"byte_span", diagnostics.byte_span},
      {"page_size_bytes", diagnostics.page_size_bytes},
      {"requested_stride_bytes", diagnostics.requested_stride_bytes},
      {"effective_node_spacing_bytes",
       diagnostics.effective_node_spacing_bytes},
      {"integrity_verified", diagnostics.integrity_verified},
  };
}

nlohmann::ordered_json build_tlb_measurement_record_json(
    const TlbMeasurementRecord& record) {
  nlohmann::ordered_json measurement = {
      {"pass", tlb_measurement_pass_to_string(record.pass)},
      {"point_index", record.point_index},
      {"locality_bytes", record.locality_bytes},
      {"round_index", record.round_index},
      {"order_index", record.order_index},
      {"seed", record.seed},
      {"latency_ns", record.latency_ns},
  };
  if (!record.paired.available) {
    measurement["paired_control"] = {{"available", false}};
    return measurement;
  }

  measurement["paired_control"] = {
      {"available", true},
      {"pair_order",
       record.paired.spread_measured_first ? "spread-first"
                                           : "packed-first"},
      {"spread",
       {{"seed", record.paired.spread.seed},
        {"latency_ns", record.paired.spread.latency_ns},
        {"pilot_access_count", record.paired.spread.pilot_access_count},
        {"pilot_duration_ns", record.paired.spread.pilot_duration_ns},
        {"access_count", record.paired.spread.access_count},
        {"chain",
         build_tlb_chain_diagnostics_json(
             record.paired.spread.diagnostics)}}},
      {"packed",
       {{"seed", record.paired.packed.seed},
        {"latency_ns", record.paired.packed.latency_ns},
        {"pilot_access_count", record.paired.packed.pilot_access_count},
        {"pilot_duration_ns", record.paired.packed.pilot_duration_ns},
        {"access_count", record.paired.packed.access_count},
        {"chain",
         build_tlb_chain_diagnostics_json(
             record.paired.packed.diagnostics)}}},
      {"translation_delta_ns", record.paired.translation_delta_ns},
  };
  return measurement;
}

}  // namespace

int save_tlb_analysis_to_json(const TlbAnalysisJsonContext& context) {
  if (context.config.output_file.empty()) {
    return EXIT_SUCCESS;
  }

  const LatencyChainMode effective_chain_mode =
      resolve_latency_chain_mode(context.config.latency_chain_mode,
                                 context.page_walk_baseline_locality_bytes);
  const size_t minimum_rounds = context.runtime_profile.min_rounds > 0
                                    ? context.runtime_profile.min_rounds
                                    : context.maximum_rounds_per_pass;
  const size_t maximum_rounds = context.runtime_profile.max_rounds > 0
                                    ? context.runtime_profile.max_rounds
                                    : context.maximum_rounds_per_pass;
  const std::string runtime_profile_name =
      context.runtime_profile.name.empty() ? "fixed" : context.runtime_profile.name;

  nlohmann::ordered_json json_output;
  json_output[JsonKeys::CONFIGURATION] = {
      {JsonKeys::MODE, Constants::TLB_ANALYSIS_JSON_MODE_NAME},
      {"schema_version", 4},
      {"methodology_version", "page-native-paired-adaptive-validated-v4"},
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
      {"chain_model", "one-node-per-spread-page-with-packed-control"},
      {"translation_delta_definition",
       "same-round spread_latency_ns - packed_latency_ns"},
      {"boundary_signal", "translation_delta_ns"},
      {"changepoint_method", "paired-point-median-bootstrap"},
      {"confidence_interval", "deterministic-percentile-bootstrap-95"},
      {"bootstrap_resamples", 2000},
      {"minimum_effect_ns", 0.5},
      {"persistence_points_required", 2},
      {"independent_validation_required", true},
      {JsonKeys::LATENCY_SAMPLE_COUNT, static_cast<int>(maximum_rounds)},
      {"runtime_profile", runtime_profile_name},
      {"adaptive_rounds",
       {{"minimum", minimum_rounds},
        {"maximum", maximum_rounds},
        {"ci_width_target_ns", context.runtime_profile.ci_width_target_ns},
        {"bootstrap_resamples",
         context.runtime_profile.convergence_bootstrap_resamples}}},
      {"access_calibration",
       {{"target_duration_ns", context.runtime_profile.target_measurement_ns},
        {"minimum_chain_cycles",
         context.runtime_profile.minimum_chain_cycles},
        {"profile_access_cap",
         context.runtime_profile.maximum_accesses > 0
             ? context.runtime_profile.maximum_accesses
             : context.profile_access_cap},
        {"policy",
         "pilot-timed whole-chain cycles; minimum cycles may exceed profile access cap"}}},
      {"tlb_guard_bytes", context.tlb_guard_bytes},
      {"buffer_size_mb", context.selected_buffer_mb},
      {"buffer_locked", context.buffer_locked},
      {"memory_budget",
       {{"available_memory_mb", context.available_memory_mb},
        {"budget_mb", context.memory_budget_mb},
        {"estimated_peak_memory_bytes",
         context.estimated_peak_memory_bytes},
        {"policy",
         "largest 1024/512/256 MB candidate whose predicted peak fits the available-memory budget"}}},
      {"buffer_lock",
       {{"locked", context.buffer_locked},
        {"errno", context.buffer_lock_errno},
        {"error", context.buffer_lock_error},
        {"policy", "best-effort; continue unlocked on failure"}}},
      {"base_work_estimate",
       {{"point_count", context.base_work_estimate.point_count},
        {"minimum_rounds", context.base_work_estimate.min_rounds},
        {"maximum_rounds", context.base_work_estimate.max_rounds},
        {"maximum_pilot_accesses_per_measurement",
         context.base_work_estimate.maximum_pilot_accesses_per_measurement},
        {"maximum_accesses_per_measurement",
         context.base_work_estimate.maximum_accesses_per_measurement},
        {"maximum_pointer_accesses",
         context.base_work_estimate.maximum_pointer_accesses},
        {"estimated_peak_memory_bytes",
         context.base_work_estimate.estimated_peak_memory_bytes},
        {"rough_minimum_duration_sec",
         context.base_work_estimate.estimated_min_duration_sec},
        {"rough_maximum_duration_sec",
         context.base_work_estimate.estimated_max_duration_sec}}},
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
    std::vector<double> spread_latencies_ns;
    std::vector<double> packed_latencies_ns;
    std::vector<double> translation_deltas_ns;
    std::vector<double> validation_spread_latencies_ns;
    std::vector<double> validation_packed_latencies_ns;
    std::vector<double> validation_translation_deltas_ns;
    bool physical_metadata_added = false;
    for (const TlbMeasurementRecord& record : context.measurement_records) {
      if (record.pass == TlbMeasurementPass::LargeLocality ||
          record.locality_bytes != context.localities_bytes[i]) {
        continue;
      }
      point["measurements"].push_back(
          build_tlb_measurement_record_json(record));
      if (!record.paired.available) {
        continue;
      }
      if (record.pass == TlbMeasurementPass::Validation) {
        validation_spread_latencies_ns.push_back(
            record.paired.spread.latency_ns);
        validation_packed_latencies_ns.push_back(
            record.paired.packed.latency_ns);
        validation_translation_deltas_ns.push_back(
            record.paired.translation_delta_ns);
      } else {
        spread_latencies_ns.push_back(record.paired.spread.latency_ns);
        packed_latencies_ns.push_back(record.paired.packed.latency_ns);
        translation_deltas_ns.push_back(
            record.paired.translation_delta_ns);
      }
      if (!physical_metadata_added) {
        point["actual_pages"] =
            record.paired.spread.diagnostics.actual_pages;
        point["packed_actual_pages"] =
            record.paired.packed.diagnostics.actual_pages;
        point["actual_node_count"] =
            record.paired.spread.diagnostics.node_count;
        point["actual_unique_cache_lines"] =
            record.paired.spread.diagnostics.unique_cache_lines;
        point["spread_chain"] = build_tlb_chain_diagnostics_json(
            record.paired.spread.diagnostics);
        point["packed_chain"] = build_tlb_chain_diagnostics_json(
            record.paired.packed.diagnostics);
        physical_metadata_added = true;
      }
    }
    if (!spread_latencies_ns.empty()) {
      point["spread_loop_latencies_ns"] = spread_latencies_ns;
      point["packed_loop_latencies_ns"] = packed_latencies_ns;
      point["translation_deltas_ns"] = translation_deltas_ns;
      point["spread_p50_latency_ns"] = median_values(spread_latencies_ns);
      point["packed_p50_latency_ns"] = median_values(packed_latencies_ns);
      point["translation_delta_p50_ns"] =
          median_values(translation_deltas_ns);
      point["translation_delta_definition"] =
          "same-round spread_latency_ns - packed_latency_ns";
    }
    if (!validation_translation_deltas_ns.empty()) {
      point["validation_spread_loop_latencies_ns"] =
          validation_spread_latencies_ns;
      point["validation_packed_loop_latencies_ns"] =
          validation_packed_latencies_ns;
      point["validation_translation_deltas_ns"] =
          validation_translation_deltas_ns;
      point["validation_translation_delta_p50_ns"] =
          median_values(validation_translation_deltas_ns);
    }
    sweep_json.push_back(point);
  }

  nlohmann::ordered_json tlb_json;
  tlb_json["status"] = context.analysis_status;
  tlb_json["planned_points"] = context.planned_points;
  tlb_json["measured_points"] = context.measured_points;
  tlb_json["validation_planned_points"] =
      context.validation_planned_points;
  tlb_json["validation_measured_points"] =
      context.validation_measured_points;
  tlb_json["validation_complete"] = context.validation_complete;
  tlb_json["conclusions_valid"] = context.conclusions_valid;
  const size_t total_planned_points =
      context.planned_points > std::numeric_limits<size_t>::max() -
                                   context.validation_planned_points
          ? std::numeric_limits<size_t>::max()
          : context.planned_points + context.validation_planned_points;
  const size_t maximum_planned_measurements =
      maximum_rounds != 0 &&
              total_planned_points >
                  std::numeric_limits<size_t>::max() / maximum_rounds
          ? std::numeric_limits<size_t>::max()
          : total_planned_points * maximum_rounds;
  const size_t minimum_planned_measurements =
      minimum_rounds != 0 &&
              total_planned_points >
                  std::numeric_limits<size_t>::max() / minimum_rounds
          ? std::numeric_limits<size_t>::max()
          : total_planned_points * minimum_rounds;
  const size_t completed_measurements = static_cast<size_t>(std::count_if(
      context.measurement_records.begin(),
      context.measurement_records.end(),
      [](const TlbMeasurementRecord& record) {
        return record.pass != TlbMeasurementPass::LargeLocality;
      }));
  tlb_json["rounds_per_pass"] = {
      {"minimum", minimum_rounds},
      {"maximum", maximum_rounds},
      {"adaptive", context.runtime_profile.max_rounds > 0}};
  tlb_json["minimum_planned_measurements"] =
      minimum_planned_measurements;
  tlb_json["maximum_planned_measurements"] =
      maximum_planned_measurements;
  tlb_json["planned_measurements"] = maximum_planned_measurements;
  tlb_json["completed_measurements"] = completed_measurements;
  tlb_json["minimum_planned_measurement_pairs"] =
      minimum_planned_measurements;
  tlb_json["maximum_planned_measurement_pairs"] =
      maximum_planned_measurements;
  tlb_json["planned_measurement_pairs"] = maximum_planned_measurements;
  tlb_json["completed_measurement_pairs"] = completed_measurements;
  tlb_json["planned_raw_measurements"] =
      maximum_planned_measurements > std::numeric_limits<size_t>::max() / 2
          ? std::numeric_limits<size_t>::max()
          : maximum_planned_measurements * 2;
  const size_t completed_pairs = static_cast<size_t>(std::count_if(
      context.measurement_records.begin(),
      context.measurement_records.end(),
      [](const TlbMeasurementRecord& record) {
        return record.pass != TlbMeasurementPass::LargeLocality &&
               record.paired.available;
      }));
  tlb_json["completed_raw_measurements"] =
      completed_pairs > std::numeric_limits<size_t>::max() / 2
          ? std::numeric_limits<size_t>::max()
          : completed_pairs * 2;
  tlb_json["pass_summaries"] = nlohmann::ordered_json::array();
  for (const TlbPassExecutionSummary& summary : context.pass_summaries) {
    tlb_json["pass_summaries"].push_back({
        {"pass", tlb_measurement_pass_to_string(summary.pass)},
        {"point_count", summary.point_count},
        {"rounds_completed", summary.rounds_completed},
        {"converged", summary.converged},
        {"status", summary.complete ? "complete" : "interrupted"},
        {"completion_reason",
         summary.converged
             ? "ci-target-reached"
             : (summary.complete ? "maximum-rounds-reached" : "interrupted")},
    });
  }
  tlb_json["measurement_records"] = nlohmann::ordered_json::array();
  for (const TlbMeasurementRecord& record : context.measurement_records) {
    tlb_json["measurement_records"].push_back(
        build_tlb_measurement_record_json(record));
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
    tlb_json["l1_tlb_detection"]["primary_result"] =
        "inferred_entries_min..inferred_entries_max";
  }
  if (context.conclusions_valid && context.l2_boundary.detected) {
    tlb_json["l2_tlb_detection"]["inferred_entries_min"] = context.l2_entries_min;
    tlb_json["l2_tlb_detection"]["inferred_entries_max"] = context.l2_entries_max;
    tlb_json["l2_tlb_detection"]["primary_result"] =
        "inferred_entries_min..inferred_entries_max";
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
      large_locality_delta["measurements"].push_back(
          build_tlb_measurement_record_json(record));
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

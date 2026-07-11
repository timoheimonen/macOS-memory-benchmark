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
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <string>

#include "benchmark/tlb_analysis.h"
#include "core/config/config.h"
#include "core/config/constants.h"
#include "core/config/version.h"
#include "core/memory/memory_utils.h"
#include "output/json/json_output/json_output_api.h"
#include "utils/json_utils.h"

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
      {"pointer_nodes", diagnostics.node_count},
      {"unique_cache_lines", diagnostics.unique_cache_lines},
      {"pointers_per_page_max", diagnostics.max_nodes_per_page},
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
      {"seed", std::to_string(record.seed)},
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
       {{"seed", std::to_string(record.paired.spread.seed)},
        {"latency_ns", record.paired.spread.latency_ns},
        {"pilot_access_count", record.paired.spread.pilot_access_count},
        {"pilot_duration_ns", record.paired.spread.pilot_duration_ns},
        {"access_count", record.paired.spread.access_count},
        {"chain",
         build_tlb_chain_diagnostics_json(
             record.paired.spread.diagnostics)}}},
      {"packed",
       {{"seed", std::to_string(record.paired.packed.seed)},
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
      {"chain_mode_comparability",
       "compare only results with identical effective chain mode; increasing-page modes are order/prefetch-sensitive"},
      {JsonKeys::TLB_DENSITY, tlb_sweep_density_to_string(context.config.tlb_sweep_density)},
      {"seed", std::to_string(context.config.tlb_seed)},
      {"seed_source", context.config.user_specified_tlb_seed ? "user" : "generated"},
      {"seed_encoding", "uint64-decimal-string"},
      {"seed_derivation",
       {{"measurement_task",
         "splitmix64(splitmix64(splitmix64(base_seed xor pass) xor round_index) xor point_index)"},
        {"chain_layout",
         "splitmix64(task_seed xor layout-domain-constant)"}}},
      {"schedule_policy", "seeded-cyclic-latin"},
      {"chain_model", "one-node-per-spread-page-with-packed-control"},
      {"latency_interpretation",
       "cache-hot pointer-chain timings; virtual locality is not the active data "
       "footprint; values are not direct DRAM latency"},
      {"translation_delta_definition",
       "same-round spread_latency_ns - packed_latency_ns"},
      {"boundary_signal", "translation_delta_ns"},
      {"changepoint_method", "paired-point-median-bootstrap"},
      {"confidence_interval", "deterministic-percentile-bootstrap-95"},
      {"bootstrap_resamples", 2000},
      {"minimum_effect_ns", 0.5},
      {"persistence_points_required", 2},
      {"independent_validation_required", true},
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
      {"main_thread_qos",
       {{"requested", context.config.main_thread_qos_requested},
        {"requested_class", "user-interactive"},
        {"applied", context.config.main_thread_qos_applied},
        {"code", context.config.main_thread_qos_code},
        {"policy", "best-effort; continue on failure"}}},
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
      point["stride_bytes"] = planned_point->stride_bytes;
      point["refinement_source"] = planned_point->refinement_source;
      point["bracket_lower_bytes"] = planned_point->bracket_lower_bytes;
      point["bracket_upper_bytes"] = planned_point->bracket_upper_bytes;
    }
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
        point["pointer_nodes"] =
            record.paired.spread.diagnostics.node_count;
        point["spread_pointers_per_page_max"] =
            record.paired.spread.diagnostics.max_nodes_per_page;
        point["packed_pointers_per_page_max"] =
            record.paired.packed.diagnostics.max_nodes_per_page;
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
      const TlbPairedPointSummary summary = summarize_tlb_paired_point(
          context.measurement_records,
          context.localities_bytes[i],
          {TlbMeasurementPass::Base, TlbMeasurementPass::Refinement});
      point["spread_loop_latencies_ns"] = spread_latencies_ns;
      point["packed_loop_latencies_ns"] = packed_latencies_ns;
      point["translation_deltas_ns"] = translation_deltas_ns;
      if (summary.available) {
        point["spread_p50_latency_ns"] = summary.spread_p50_ns;
        point["packed_p50_latency_ns"] = summary.packed_p50_ns;
        point["translation_delta_p50_ns"] =
            summary.translation_delta_p50_ns;
        point["active_cache_line_footprint_bytes"] =
            summary.active_cache_line_footprint_bytes;
        point["short_cycle_diagnostic"] =
            summary.short_cycle_diagnostic;
      }
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
  const size_t validation_record_count = static_cast<size_t>(std::count_if(
      context.measurement_records.begin(),
      context.measurement_records.end(),
      [](const TlbMeasurementRecord& record) {
        return record.pass == TlbMeasurementPass::Validation;
      }));
  const bool validation_required = context.validation_planned_points > 0;
  const bool analysis_complete = context.analysis_status == "complete";
  bool reported_validation_complete = false;
  std::string validation_status = "not-run";
  if (!validation_required && analysis_complete) {
    reported_validation_complete = true;
    validation_status = "not-required";
  } else if (validation_required && context.validation_complete &&
             context.validation_measured_points ==
                 context.validation_planned_points) {
    reported_validation_complete = true;
    validation_status = "complete";
  } else if (validation_record_count > 0) {
    validation_status = context.analysis_status == "interrupted"
                            ? "interrupted"
                            : "partial";
  }
  tlb_json["validation_required"] = validation_required;
  tlb_json["validation_status"] = validation_status;
  tlb_json["validation_complete"] = reported_validation_complete;
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
  const size_t completed_base_validation_records =
      static_cast<size_t>(std::count_if(
          context.measurement_records.begin(),
          context.measurement_records.end(),
          [](const TlbMeasurementRecord& record) {
            return record.pass != TlbMeasurementPass::LargeLocality;
          }));
  const size_t completed_large_locality_records =
      context.measurement_records.size() - completed_base_validation_records;
  const size_t completed_base_validation_pairs =
      static_cast<size_t>(std::count_if(
          context.measurement_records.begin(),
          context.measurement_records.end(),
          [](const TlbMeasurementRecord& record) {
            return record.pass != TlbMeasurementPass::LargeLocality &&
                   record.paired.available;
          }));
  const size_t completed_large_locality_pairs =
      static_cast<size_t>(std::count_if(
          context.measurement_records.begin(),
          context.measurement_records.end(),
          [](const TlbMeasurementRecord& record) {
            return record.pass == TlbMeasurementPass::LargeLocality &&
                   record.paired.available;
          }));
  const size_t total_completed_pairs =
      completed_base_validation_pairs >
              std::numeric_limits<size_t>::max() -
                  completed_large_locality_pairs
          ? std::numeric_limits<size_t>::max()
          : completed_base_validation_pairs +
                completed_large_locality_pairs;
  const auto raw_measurement_count = [](size_t pair_count) {
    return pair_count > std::numeric_limits<size_t>::max() / 2
               ? std::numeric_limits<size_t>::max()
               : pair_count * 2;
  };
  tlb_json["rounds_per_pass"] = {
      {"minimum", minimum_rounds},
      {"maximum", maximum_rounds},
      {"adaptive", context.runtime_profile.max_rounds > 0}};
  tlb_json["minimum_planned_base_validation_pairs"] =
      minimum_planned_measurements;
  tlb_json["maximum_planned_base_validation_pairs"] =
      maximum_planned_measurements;
  tlb_json["planned_base_validation_pairs"] =
      maximum_planned_measurements;
  tlb_json["planned_base_validation_raw_measurements"] =
      raw_measurement_count(maximum_planned_measurements);
  tlb_json["completed_base_validation_measurement_records"] =
      completed_base_validation_records;
  tlb_json["completed_base_validation_pairs"] =
      completed_base_validation_pairs;
  tlb_json["completed_base_validation_raw_measurements"] =
      raw_measurement_count(completed_base_validation_pairs);
  tlb_json["completed_large_locality_measurement_records"] =
      completed_large_locality_records;
  tlb_json["completed_large_locality_pairs"] =
      completed_large_locality_pairs;
  tlb_json["completed_large_locality_raw_measurements"] =
      raw_measurement_count(completed_large_locality_pairs);
  tlb_json["total_completed_measurement_records"] =
      context.measurement_records.size();
  tlb_json["total_completed_measurement_pairs"] =
      total_completed_pairs;
  tlb_json["total_completed_raw_measurements"] =
      raw_measurement_count(total_completed_pairs);
  tlb_json["measurement_counter_scope"] = {
      {"base_validation", "base and validation passes"},
      {"large_locality", "large-locality pass"},
      {"total", "all serialized measurement passes"},
  };
  tlb_json["pass_summaries"] = nlohmann::ordered_json::array();
  for (const TlbPassExecutionSummary& summary : context.pass_summaries) {
    const char* status = "complete";
    const char* completion_reason = summary.converged
                                        ? "ci-target-reached"
                                        : "maximum-rounds-reached";
    if (summary.status == TlbScheduleExecutionStatus::Interrupted) {
      status = "interrupted";
      completion_reason = "interrupted";
    } else if (summary.status == TlbScheduleExecutionStatus::Error) {
      status = "error";
      completion_reason = "measurement-error";
    }
    tlb_json["pass_summaries"].push_back({
        {"pass", tlb_measurement_pass_to_string(summary.pass)},
        {"point_count", summary.point_count},
        {"rounds_completed", summary.rounds_completed},
        {"converged", summary.converged},
        {"status", status},
        {"completion_reason", completion_reason},
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

  const bool large_locality_comparison_available =
      context.conclusions_valid && context.page_walk_comparison_completed;
  const TlbPairedPointSummary large_locality_summary =
      summarize_tlb_paired_point(
          context.measurement_records,
          context.page_walk_comparison_locality_bytes,
          {TlbMeasurementPass::LargeLocality});
  const bool large_locality_paired_available =
      large_locality_comparison_available && large_locality_summary.available;
  nlohmann::ordered_json large_locality_paired = {
      {"available", large_locality_paired_available},
      {"comparison_locality_mb",
       context.page_walk_comparison_locality_bytes /
           Constants::BYTES_PER_MB},
      {"translation_delta_definition",
       "median of same-round (spread_latency_ns - packed_latency_ns)"},
      {"interpretation",
       "cache-hot paired translation stress; not DRAM latency and not an isolated page-table-walk cost"}};
  if (large_locality_paired_available) {
    large_locality_paired["spread_p50_ns"] =
        large_locality_summary.spread_p50_ns;
    large_locality_paired["packed_p50_ns"] =
        large_locality_summary.packed_p50_ns;
    large_locality_paired["translation_delta_p50_ns"] =
        large_locality_summary.translation_delta_p50_ns;
    large_locality_paired["spread_actual_pages"] =
        large_locality_summary.spread_actual_pages;
    large_locality_paired["packed_actual_pages"] =
        large_locality_summary.packed_actual_pages;
    large_locality_paired["unique_cache_lines"] =
        large_locality_summary.unique_cache_lines;
    large_locality_paired["active_cache_line_footprint_bytes"] =
        large_locality_summary.active_cache_line_footprint_bytes;
    large_locality_paired["pointer_nodes"] =
        large_locality_summary.node_count;
    large_locality_paired["measurements"] =
        nlohmann::ordered_json::array();
    for (const TlbMeasurementRecord& record : context.measurement_records) {
      if (record.pass == TlbMeasurementPass::LargeLocality) {
        large_locality_paired["measurements"].push_back(
            build_tlb_measurement_record_json(record));
      }
    }
  } else if (!context.conclusions_valid) {
    large_locality_paired["reason"] =
        "analysis incomplete; paired comparison suppressed";
  } else if (!context.can_measure_page_walk_penalty) {
    large_locality_paired["reason"] =
        "requires at least 512 MiB analysis buffer";
  } else {
    large_locality_paired["reason"] =
        "paired comparison measurement did not complete";
  }
  tlb_json["large_locality_paired_comparison"] =
      large_locality_paired;

  json_output["tlb_analysis"] = tlb_json;
  json_output[JsonKeys::TIMESTAMP] = build_utc_timestamp();
  json_output[JsonKeys::VERSION] = SOFTVERSION;

  std::filesystem::path file_path(context.config.output_file);
  if (file_path.is_relative()) {
    file_path = std::filesystem::current_path() / file_path;
  }

  return write_json_to_file(file_path, json_output);
}

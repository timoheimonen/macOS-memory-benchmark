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

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>

#include "benchmark/tlb_analysis.h"
#include "core/config/config.h"
#include "core/config/constants.h"
#include "core/config/version.h"
#include "output/json/json_output.h"

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
  boundary_json["confidence"] = boundary.confidence;
  boundary_json["inferred_entries"] = inferred_entries;
  return boundary_json;
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

  nlohmann::ordered_json json_output;
  json_output[JsonKeys::CONFIGURATION] = {
      {JsonKeys::MODE, Constants::TLB_ANALYSIS_JSON_MODE_NAME},
      {JsonKeys::CPU_NAME, context.cpu_name},
      {JsonKeys::PERFORMANCE_CORES, context.perf_cores},
      {JsonKeys::EFFICIENCY_CORES, context.eff_cores},
      {JsonKeys::PAGE_SIZE_BYTES, context.page_size_bytes},
      {JsonKeys::L1_CACHE_SIZE_BYTES, context.l1_cache_size_bytes},
      {JsonKeys::LATENCY_STRIDE_BYTES, context.stride_bytes},
      {JsonKeys::LATENCY_SAMPLE_COUNT, static_cast<int>(context.loops_per_point)},
      {"accesses_per_loop", context.accesses_per_loop},
      {"tlb_guard_bytes", context.tlb_guard_bytes},
      {"buffer_size_mb", context.selected_buffer_mb},
      {"buffer_locked", context.buffer_locked}};
  json_output[JsonKeys::EXECUTION_TIME_SEC] = context.total_execution_time_sec;

  nlohmann::ordered_json sweep_json = nlohmann::ordered_json::array();
  for (size_t i = 0; i < context.localities_bytes.size(); ++i) {
    nlohmann::ordered_json point;
    point["locality_bytes"] = context.localities_bytes[i];
    point["locality_kb"] = context.localities_bytes[i] / Constants::BYTES_PER_KB;
    point["loop_latencies_ns"] = context.sweep_loop_latencies_ns[i];
    point["p50_latency_ns"] = context.p50_latency_ns[i];
    sweep_json.push_back(point);
  }

  nlohmann::ordered_json tlb_json;
  tlb_json["sweep"] = sweep_json;
  tlb_json["l1_tlb_detection"] = build_tlb_boundary_json(context.l1_boundary, context.l1_entries);
  tlb_json["l2_tlb_detection"] = build_tlb_boundary_json(context.l2_boundary, context.l2_entries);
  tlb_json["page_walk_penalty"] = {
      {"available", context.can_measure_page_walk_penalty},
      {"baseline_locality_kb", context.page_walk_baseline_locality_bytes / Constants::BYTES_PER_KB},
      {"comparison_locality_mb", context.page_walk_comparison_locality_bytes / Constants::BYTES_PER_MB},
      {"baseline_p50_ns", context.page_walk_baseline_ns}};

  if (context.can_measure_page_walk_penalty) {
    tlb_json["page_walk_penalty"]["comparison_loop_latencies_ns"] =
        context.page_walk_comparison_loop_latencies_ns;
    tlb_json["page_walk_penalty"]["comparison_p50_ns"] = context.page_walk_comparison_p50_ns;
    tlb_json["page_walk_penalty"]["penalty_ns"] = context.page_walk_penalty_ns;
  } else {
    tlb_json["page_walk_penalty"]["reason"] = "requires at least 512 MB analysis buffer";
  }

  json_output["tlb_analysis"] = tlb_json;
  json_output[JsonKeys::TIMESTAMP] = build_utc_timestamp();
  json_output[JsonKeys::VERSION] = SOFTVERSION;

  std::filesystem::path file_path(context.config.output_file);
  if (file_path.is_relative()) {
    file_path = std::filesystem::current_path() / file_path;
  }

  return write_json_to_file(file_path, json_output);
}

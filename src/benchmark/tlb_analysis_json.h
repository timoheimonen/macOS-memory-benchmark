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
 * @file tlb_analysis_json.h
 * @brief JSON serialization helpers for standalone TLB analysis mode
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 */

#ifndef TLB_ANALYSIS_JSON_H
#define TLB_ANALYSIS_JSON_H

#include <cstddef>
#include <string>
#include <vector>

#include "benchmark/tlb_analysis.h"
#include "benchmark/tlb_runtime_policy.h"

struct BenchmarkConfig;

/**
 * @struct TlbAnalysisJsonContext
 * @brief Input bundle for writing standalone TLB analysis JSON output.
 */
struct TlbAnalysisJsonContext {
  const BenchmarkConfig& config;
  const std::string& cpu_name;
  int perf_cores;
  int eff_cores;
  size_t page_size_bytes;
  size_t l1_cache_size_bytes;
  size_t tlb_guard_bytes;
  size_t stride_bytes;
  size_t maximum_rounds_per_pass;
  size_t profile_access_cap;
  size_t page_walk_baseline_locality_bytes;
  size_t page_walk_comparison_locality_bytes;
  size_t selected_buffer_mb;
  bool buffer_locked;
  std::string analysis_status;
  size_t planned_points;
  size_t measured_points;
  size_t validation_planned_points;
  size_t validation_measured_points;
  bool validation_complete;
  bool conclusions_valid;
  const std::vector<TlbSweepPoint>& sweep_points;
  const std::vector<TlbMeasurementRecord>& measurement_records;
  const std::vector<size_t>& localities_bytes;
  const TlbBoundaryDetection& l1_boundary;
  const TlbBoundaryDetection& l2_boundary;
  const PrivateCacheKneeDetection& private_cache_knee;
  size_t l1_entries;
  size_t l2_entries;
  size_t l1_entries_min;
  size_t l1_entries_max;
  size_t l2_entries_min;
  size_t l2_entries_max;
  size_t fine_sweep_added_points;
  bool private_cache_interference_elevated;
  size_t private_cache_to_l1_distance_bytes;
  size_t private_cache_to_l1_distance_pages;
  bool can_measure_page_walk_penalty;
  bool page_walk_comparison_completed;
  double total_execution_time_sec;
  TlbRuntimeProfile runtime_profile;
  size_t available_memory_mb = 0;
  size_t memory_budget_mb = 0;
  size_t estimated_peak_memory_bytes = 0;
  int buffer_lock_errno = 0;
  std::string buffer_lock_error;
  TlbWorkEstimate base_work_estimate;
  std::vector<TlbPassExecutionSummary> pass_summaries;
};

/**
 * @brief Save standalone TLB analysis results to JSON output file.
 * @param context TLB analysis result bundle for serialization.
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on error.
 */
int save_tlb_analysis_to_json(const TlbAnalysisJsonContext& context);

#endif  // TLB_ANALYSIS_JSON_H

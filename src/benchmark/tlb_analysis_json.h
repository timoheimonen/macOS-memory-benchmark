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
  size_t loops_per_point;
  size_t accesses_per_loop;
  size_t page_walk_baseline_locality_bytes;
  size_t page_walk_comparison_locality_bytes;
  size_t selected_buffer_mb;
  bool buffer_locked;
  const std::vector<size_t>& localities_bytes;
  const std::vector<std::vector<double>>& sweep_loop_latencies_ns;
  const std::vector<double>& p50_latency_ns;
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
  bool can_measure_page_walk_penalty;
  const std::vector<double>& page_walk_comparison_loop_latencies_ns;
  double page_walk_comparison_p50_ns;
  double page_walk_baseline_ns;
  double page_walk_penalty_ns;
  double total_execution_time_sec;
};

/**
 * @brief Save standalone TLB analysis results to JSON output file.
 * @param context TLB analysis result bundle for serialization.
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on error.
 */
int save_tlb_analysis_to_json(const TlbAnalysisJsonContext& context);

#endif  // TLB_ANALYSIS_JSON_H

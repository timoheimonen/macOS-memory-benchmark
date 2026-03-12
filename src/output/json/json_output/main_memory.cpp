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
 * @file main_memory.cpp
 * @brief JSON output generation for main memory benchmark results
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2025
 *
 * This file builds the JSON structure for main memory benchmark results
 * including bandwidth (read/write/copy) and latency measurements with
 * values and optional statistical aggregation.
 */
// This file uses the nlohmann/json library for JSON parsing and generation.
// Library: https://github.com/nlohmann/json
// License: MIT License
//
#include "output/json/json_output.h"
#include "core/config/config.h"     // For BenchmarkConfig
#include "benchmark/benchmark_runner.h"  // For BenchmarkStatistics
#include "utils/json_utils.h"  // calculate_json_statistics
#include "third_party/nlohmann/json.hpp"   // JSON library

namespace {

void add_metric_with_statistics(nlohmann::json& parent,
                                const char* metric_key,
                                const std::vector<double>& values) {
  if (values.empty()) {
    return;
  }

  parent[metric_key] = nlohmann::json::object();
  parent[metric_key][JsonKeys::VALUES] = values;
  if (values.size() > 1) {
    parent[metric_key][JsonKeys::STATISTICS] = calculate_json_statistics(values);
  }
}

}  // namespace

// Build main memory results JSON object
nlohmann::json build_main_memory_json(const BenchmarkConfig& config, const BenchmarkStatistics& stats) {
  nlohmann::json main_memory;
  
  // Add bandwidth results (skip if only latency tests)
  if (!config.only_latency) {
    add_bandwidth_results(main_memory, 
                          stats.all_read_bw_gb_s,
                          stats.all_write_bw_gb_s,
                          stats.all_copy_bw_gb_s);
  }
  
  // Add latency results (skip if only bandwidth tests)
  if (!config.only_bandwidth) {
    add_latency_results(main_memory,
                        stats.all_average_latency_ns,
                        stats.all_main_mem_latency_samples);

    if (main_memory.contains(JsonKeys::LATENCY) &&
        (!stats.all_tlb_hit_latency_ns.empty() ||
         !stats.all_tlb_miss_latency_ns.empty() ||
         !stats.all_page_walk_penalty_ns.empty())) {
      nlohmann::json auto_tlb_breakdown = nlohmann::json::object();
      add_metric_with_statistics(auto_tlb_breakdown,
                                 JsonKeys::TLB_HIT_NS,
                                 stats.all_tlb_hit_latency_ns);
      add_metric_with_statistics(auto_tlb_breakdown,
                                 JsonKeys::TLB_MISS_NS,
                                 stats.all_tlb_miss_latency_ns);
      add_metric_with_statistics(auto_tlb_breakdown,
                                 JsonKeys::PAGE_WALK_PENALTY_NS,
                                 stats.all_page_walk_penalty_ns);
      if (!auto_tlb_breakdown.empty()) {
        main_memory[JsonKeys::LATENCY][JsonKeys::AUTO_TLB_BREAKDOWN] = auto_tlb_breakdown;
      }
    }

    const auto& diagnostics = config.main_latency_chain_diagnostics;
    if (main_memory.contains(JsonKeys::LATENCY) && diagnostics.pointer_count > 0) {
      main_memory[JsonKeys::LATENCY][JsonKeys::CHAIN_DIAGNOSTICS] = {
          {JsonKeys::POINTER_COUNT, diagnostics.pointer_count},
          {JsonKeys::UNIQUE_PAGES_TOUCHED, diagnostics.unique_pages_touched},
          {JsonKeys::PAGE_SIZE_BYTES, diagnostics.page_size_bytes},
          {JsonKeys::STRIDE_BYTES, diagnostics.stride_bytes}};
    }
  }
  
  return main_memory;
}

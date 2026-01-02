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
#include "benchmark/benchmark_statistics_collector.h"
#include "benchmark/benchmark_runner.h"  // BenchmarkStatistics, BenchmarkResults
#include "core/config/config.h"           // BenchmarkConfig

// Initialize statistics structure by clearing and pre-allocating vectors
void initialize_statistics(BenchmarkStatistics& stats, const BenchmarkConfig& config) {
  // Initialize result vectors
  stats.all_read_bw_gb_s.clear();
  stats.all_write_bw_gb_s.clear();
  stats.all_copy_bw_gb_s.clear();
  stats.all_l1_latency_ns.clear();
  stats.all_l2_latency_ns.clear();
  stats.all_average_latency_ns.clear();
  stats.all_l1_read_bw_gb_s.clear();
  stats.all_l1_write_bw_gb_s.clear();
  stats.all_l1_copy_bw_gb_s.clear();
  stats.all_l2_read_bw_gb_s.clear();
  stats.all_l2_write_bw_gb_s.clear();
  stats.all_l2_copy_bw_gb_s.clear();
  stats.all_custom_latency_ns.clear();
  stats.all_custom_read_bw_gb_s.clear();
  stats.all_custom_write_bw_gb_s.clear();
  stats.all_custom_copy_bw_gb_s.clear();
  
  // Initialize sample vectors
  stats.all_main_mem_latency_samples.clear();
  stats.all_l1_latency_samples.clear();
  stats.all_l2_latency_samples.clear();
  stats.all_custom_latency_samples.clear();

  // Pre-allocate vector space if needed
  if (config.loop_count > 0) {
    stats.all_read_bw_gb_s.reserve(config.loop_count);
    stats.all_write_bw_gb_s.reserve(config.loop_count);
    stats.all_copy_bw_gb_s.reserve(config.loop_count);
    stats.all_average_latency_ns.reserve(config.loop_count);
    if (config.use_custom_cache_size) {
      if (config.custom_buffer_size > 0) {
        stats.all_custom_latency_ns.reserve(config.loop_count);
        stats.all_custom_read_bw_gb_s.reserve(config.loop_count);
        stats.all_custom_write_bw_gb_s.reserve(config.loop_count);
        stats.all_custom_copy_bw_gb_s.reserve(config.loop_count);
        // Pre-allocate sample vectors (latency_sample_count samples per loop)
        stats.all_custom_latency_samples.reserve(config.loop_count * config.latency_sample_count);
      }
    } else {
      stats.all_l1_latency_ns.reserve(config.loop_count);
      stats.all_l2_latency_ns.reserve(config.loop_count);
      if (config.l1_buffer_size > 0) {
        stats.all_l1_read_bw_gb_s.reserve(config.loop_count);
        stats.all_l1_write_bw_gb_s.reserve(config.loop_count);
        stats.all_l1_copy_bw_gb_s.reserve(config.loop_count);
        stats.all_l1_latency_samples.reserve(config.loop_count * config.latency_sample_count);
      }
      if (config.l2_buffer_size > 0) {
        stats.all_l2_read_bw_gb_s.reserve(config.loop_count);
        stats.all_l2_write_bw_gb_s.reserve(config.loop_count);
        stats.all_l2_copy_bw_gb_s.reserve(config.loop_count);
        stats.all_l2_latency_samples.reserve(config.loop_count * config.latency_sample_count);
      }
    }
    // Pre-allocate main memory sample vector
    stats.all_main_mem_latency_samples.reserve(config.loop_count * config.latency_sample_count);
  }
}

// Collect results from a single benchmark loop into statistics
void collect_loop_results(BenchmarkStatistics& stats, const BenchmarkResults& loop_results, const BenchmarkConfig& config) {
  // Store results for this loop
  stats.all_read_bw_gb_s.push_back(loop_results.read_bw_gb_s);
  stats.all_write_bw_gb_s.push_back(loop_results.write_bw_gb_s);
  stats.all_copy_bw_gb_s.push_back(loop_results.copy_bw_gb_s);
  if (config.use_custom_cache_size) {
    if (config.custom_buffer_size > 0) {
      stats.all_custom_latency_ns.push_back(loop_results.custom_latency_ns);
      stats.all_custom_read_bw_gb_s.push_back(loop_results.custom_read_bw_gb_s);
      stats.all_custom_write_bw_gb_s.push_back(loop_results.custom_write_bw_gb_s);
      stats.all_custom_copy_bw_gb_s.push_back(loop_results.custom_copy_bw_gb_s);
    }
  } else {
    if (config.l1_buffer_size > 0) {
      stats.all_l1_latency_ns.push_back(loop_results.l1_latency_ns);
      stats.all_l1_read_bw_gb_s.push_back(loop_results.l1_read_bw_gb_s);
      stats.all_l1_write_bw_gb_s.push_back(loop_results.l1_write_bw_gb_s);
      stats.all_l1_copy_bw_gb_s.push_back(loop_results.l1_copy_bw_gb_s);
    }
    if (config.l2_buffer_size > 0) {
      stats.all_l2_latency_ns.push_back(loop_results.l2_latency_ns);
      stats.all_l2_read_bw_gb_s.push_back(loop_results.l2_read_bw_gb_s);
      stats.all_l2_write_bw_gb_s.push_back(loop_results.l2_write_bw_gb_s);
      stats.all_l2_copy_bw_gb_s.push_back(loop_results.l2_copy_bw_gb_s);
    }
  }
  stats.all_average_latency_ns.push_back(loop_results.average_latency_ns);
  
  // Collect latency samples from this loop
  if (!loop_results.latency_samples.empty()) {
    stats.all_main_mem_latency_samples.insert(stats.all_main_mem_latency_samples.end(),
                                               loop_results.latency_samples.begin(),
                                               loop_results.latency_samples.end());
  }
  
  if (config.use_custom_cache_size) {
    if (config.custom_buffer_size > 0 && !loop_results.custom_latency_samples.empty()) {
      stats.all_custom_latency_samples.insert(stats.all_custom_latency_samples.end(),
                                             loop_results.custom_latency_samples.begin(),
                                             loop_results.custom_latency_samples.end());
    }
  } else {
    if (config.l1_buffer_size > 0 && !loop_results.l1_latency_samples.empty()) {
      stats.all_l1_latency_samples.insert(stats.all_l1_latency_samples.end(),
                                           loop_results.l1_latency_samples.begin(),
                                           loop_results.l1_latency_samples.end());
    }
    if (config.l2_buffer_size > 0 && !loop_results.l2_latency_samples.empty()) {
      stats.all_l2_latency_samples.insert(stats.all_l2_latency_samples.end(),
                                           loop_results.l2_latency_samples.begin(),
                                           loop_results.l2_latency_samples.end());
    }
  }
}


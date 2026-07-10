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
 * @file benchmark_statistics_collector.cpp
 * @brief Statistics collection
 *
 * Implements statistics collection functions for benchmark results. Manages initialization
 * and accumulation of results from multiple benchmark loops for statistical analysis
 * (mean, median, percentiles, variance).
 *
 * Key features:
 * - Vector pre-allocation to minimize reallocations
 * - Separate storage for each metric (bandwidth, latency)
 * - Sample collection for latency distribution analysis
 * - Conditional storage based on configuration (L1/L2 vs custom)
 *
 * Data collected:
 * - Main memory: read/write/copy bandwidth, latency, latency samples
 * - Cache levels: read/write/copy bandwidth, latency, latency samples
 * - All metrics stored across multiple loops for aggregate statistics
 */

#include "benchmark/benchmark_statistics_collector.h"
#include "benchmark/benchmark_runner.h"  // BenchmarkStatistics, BenchmarkResults
#include "core/config/config.h"           // BenchmarkConfig

namespace {

void append_measured_value(std::vector<double>& values,
                           const BenchmarkMeasurement& measurement) {
  if (measurement.is_measured()) {
    values.push_back(*measurement.value);
  }
}

void append_measurement_samples(std::vector<double>& samples,
                                const BenchmarkMeasurement& measurement) {
  if (measurement.is_measured() && !measurement.samples.empty()) {
    samples.insert(samples.end(), measurement.samples.begin(),
                   measurement.samples.end());
  }
}

}  // namespace

/**
 * @brief Initialize statistics structure by clearing and pre-allocating vectors.
 *
 * Prepares the statistics structure for data collection by:
 * 1. Clearing all result vectors
 * 2. Pre-allocating space based on loop count and configuration
 * 3. Conditional pre-allocation for L1/L2 or custom cache metrics
 *
 * Pre-allocation benefits:
 * - Minimizes vector reallocations during collection
 * - Improves performance for high loop counts
 * - Reduces memory fragmentation
 *
 * Vector categories:
 * - Main memory metrics: Always pre-allocated (read/write/copy bandwidth, latency)
 * - Cache metrics: Conditionally pre-allocated based on use_custom_cache_size
 * - Sample vectors: Pre-allocated for loop_count * latency_sample_count
 *
 * @param[out] stats  Statistics structure to initialize
 * @param[in]  config Configuration (loop_count, cache sizes, sample_count, flags)
 *
 * @note All vectors are cleared before pre-allocation.
 * @note Pre-allocation only occurs if loop_count > 0.
 * @note Sample vectors account for multiple samples per loop.
 *
 * @see collect_loop_results() which populates the initialized structure
 */
void initialize_statistics(BenchmarkStatistics& stats, const BenchmarkConfig& config) {
  stats.status = BenchmarkRunStatus::NotStarted;
  stats.status_reason.clear();
  stats.planned_loops = config.loop_count > 0 ? static_cast<size_t>(config.loop_count) : 0;
  stats.completed_loops = 0;
  stats.planned_measurements = 0;
  stats.completed_measurements = 0;
  stats.loop_results.clear();
  // Initialize result vectors
  stats.all_read_bw_gb_s.clear();
  stats.all_write_bw_gb_s.clear();
  stats.all_copy_bw_gb_s.clear();
  stats.all_l1_latency_ns.clear();
  stats.all_l2_latency_ns.clear();
  stats.all_average_latency_ns.clear();
  stats.all_tlb_hit_latency_ns.clear();
  stats.all_tlb_miss_latency_ns.clear();
  stats.all_page_walk_penalty_ns.clear();
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
    stats.loop_results.reserve(static_cast<size_t>(config.loop_count));
    stats.all_read_bw_gb_s.reserve(config.loop_count);
    stats.all_write_bw_gb_s.reserve(config.loop_count);
    stats.all_copy_bw_gb_s.reserve(config.loop_count);
    stats.all_average_latency_ns.reserve(config.loop_count);
    stats.all_tlb_hit_latency_ns.reserve(config.loop_count);
    stats.all_tlb_miss_latency_ns.reserve(config.loop_count);
    stats.all_page_walk_penalty_ns.reserve(config.loop_count);
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

/**
 * @brief Collect results from a single benchmark loop into statistics.
 *
 * Appends results from one benchmark loop to the statistics vectors for later
 * aggregate analysis. Conditionally collects cache metrics based on configuration.
 *
 * Data collected:
 * - Main memory bandwidth (read, write, copy)
 * - Main memory latency and samples
 * - Cache bandwidth and latency (L1/L2 or custom)
 * - Cache latency samples
 *
 * Conditional collection:
 * - use_custom_cache_size: Collect custom cache metrics
 * - Otherwise: Collect L1 and/or L2 metrics if buffer_size > 0
 *
 * Sample collection:
 * - Latency samples inserted using vector::insert for efficiency
 * - Samples from all loops combined for distribution analysis
 *
 * @param[in,out] stats        Statistics structure to append results to
 * @param[in]     loop_results Results from a single benchmark loop
 * @param[in]     config       Configuration (cache sizes, flags for conditional collection)
 *
 * @note Assumes stats was initialized by initialize_statistics().
 * @note Vector push_back operations are efficient due to pre-allocation.
 * @note Sample vectors grow by sample_count elements per loop.
 *
 * @see initialize_statistics() which must be called first
 * @see BenchmarkStatistics for the structure definition
 * @see BenchmarkResults for loop results structure
 */
void collect_loop_results(BenchmarkStatistics& stats, const BenchmarkResults& loop_results, const BenchmarkConfig& config) {
  const bool main_latency_enabled =
      (!config.only_bandwidth && config.buffer_size > 0 && config.lat_num_accesses > 0);

  stats.loop_results.push_back(loop_results);
  stats.planned_measurements += loop_results.planned_measurements;
  stats.completed_measurements += loop_results.completed_measurements;
  if (loop_results.status == BenchmarkRunStatus::Complete) {
    ++stats.completed_loops;
  }

  // Store results for this loop
  append_measured_value(stats.all_read_bw_gb_s, loop_results.main_read_bandwidth);
  append_measured_value(stats.all_write_bw_gb_s, loop_results.main_write_bandwidth);
  append_measured_value(stats.all_copy_bw_gb_s, loop_results.main_copy_bandwidth);
  if (config.use_custom_cache_size) {
    if (config.custom_buffer_size > 0) {
      append_measured_value(stats.all_custom_latency_ns, loop_results.custom_latency);
      append_measured_value(stats.all_custom_read_bw_gb_s,
                            loop_results.custom_read_bandwidth);
      append_measured_value(stats.all_custom_write_bw_gb_s,
                            loop_results.custom_write_bandwidth);
      append_measured_value(stats.all_custom_copy_bw_gb_s,
                            loop_results.custom_copy_bandwidth);
    }
  } else {
    if (config.l1_buffer_size > 0) {
      append_measured_value(stats.all_l1_latency_ns, loop_results.l1_latency);
      append_measured_value(stats.all_l1_read_bw_gb_s,
                            loop_results.l1_read_bandwidth);
      append_measured_value(stats.all_l1_write_bw_gb_s,
                            loop_results.l1_write_bandwidth);
      append_measured_value(stats.all_l1_copy_bw_gb_s,
                            loop_results.l1_copy_bandwidth);
    }
    if (config.l2_buffer_size > 0) {
      append_measured_value(stats.all_l2_latency_ns, loop_results.l2_latency);
      append_measured_value(stats.all_l2_read_bw_gb_s,
                            loop_results.l2_read_bandwidth);
      append_measured_value(stats.all_l2_write_bw_gb_s,
                            loop_results.l2_write_bandwidth);
      append_measured_value(stats.all_l2_copy_bw_gb_s,
                            loop_results.l2_copy_bandwidth);
    }
  }
  if (main_latency_enabled) {
    append_measured_value(stats.all_average_latency_ns, loop_results.main_latency);
    append_measured_value(stats.all_tlb_hit_latency_ns,
                          loop_results.locality_16k_latency);
    append_measured_value(stats.all_tlb_miss_latency_ns,
                          loop_results.global_random_latency);
    append_measured_value(stats.all_page_walk_penalty_ns,
                          loop_results.locality_latency_delta);
  }
  
  // Collect latency samples from this loop
  if (main_latency_enabled) {
    append_measurement_samples(stats.all_main_mem_latency_samples,
                               loop_results.main_latency);
  }
  
  if (config.use_custom_cache_size) {
    if (config.custom_buffer_size > 0) {
      append_measurement_samples(stats.all_custom_latency_samples,
                                 loop_results.custom_latency);
    }
  } else {
    if (config.l1_buffer_size > 0) {
      append_measurement_samples(stats.all_l1_latency_samples,
                                 loop_results.l1_latency);
    }
    if (config.l2_buffer_size > 0) {
      append_measurement_samples(stats.all_l2_latency_samples,
                                 loop_results.l2_latency);
    }
  }
}

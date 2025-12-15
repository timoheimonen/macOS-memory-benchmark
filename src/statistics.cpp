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
#include <algorithm>  // Required for std::min_element, std::max_element, std::sort (finding min/max, sorting)
#include <cmath>       // Required for std::sqrt (standard deviation calculation)
#include <iomanip>     // Required for std::setprecision, std::fixed (output formatting)
#include <iostream>    // Required for std::cout
#include <numeric>     // Required for std::accumulate (calculating sums)
#include <vector>      // Required for std::vector

// --- Helper structures and functions for statistics ---

// Structure to hold calculated statistics (average, min, max, percentiles, stddev)
struct Statistics {
  double average;
  double min;
  double max;
  double median;    // P50
  double p90;
  double p95;
  double p99;
  double stddev;
};

// Calculate statistics (average, min, max, percentiles, stddev) from a vector of values
static Statistics calculate_statistics(const std::vector<double> &values) {
  if (values.empty()) {
    return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  }

  double sum = std::accumulate(values.begin(), values.end(), 0.0);
  double avg = sum / values.size();
  double min_val = *std::min_element(values.begin(), values.end());
  double max_val = *std::max_element(values.begin(), values.end());

  // Sort copy for percentiles
  std::vector<double> sorted = values;
  std::sort(sorted.begin(), sorted.end());
  size_t n = sorted.size();

  // Helper function for percentile calculation (linear interpolation)
  auto percentile = [&sorted, n](double p) -> double {
    if (n == 0) return 0.0;
    if (n == 1) return sorted[0];
    double index = p * (n - 1);
    size_t lower = static_cast<size_t>(index);
    size_t upper = lower + 1;
    if (upper >= n) return sorted[n - 1];
    double weight = index - lower;
    return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
  };

  // Calculate percentiles
  double median = percentile(0.50);  // P50
  double p90 = percentile(0.90);
  double p95 = percentile(0.95);
  double p99 = percentile(0.99);

  // Stddev
  double variance = 0.0;
  for (double v : values) variance += (v - avg) * (v - avg);
  double stddev_val = std::sqrt(variance / n);

  return {avg, min_val, max_val, median, p90, p95, p99, stddev_val};
}

// Print statistics for a single metric (used for main memory bandwidth)
static void print_metric_statistics(const std::string &metric_name, const Statistics &stats, int precision = 3) {
  std::cout << metric_name << ":" << std::endl;
  std::cout << "  Average: " << std::setprecision(precision) << stats.average << std::endl;
  std::cout << "  Median (P50): " << std::setprecision(precision) << stats.median << std::endl;
  std::cout << "  P90: " << std::setprecision(precision) << stats.p90 << std::endl;
  std::cout << "  P95: " << std::setprecision(precision) << stats.p95 << std::endl;
  std::cout << "  P99: " << std::setprecision(precision) << stats.p99 << std::endl;
  std::cout << "  Stddev: " << std::setprecision(precision) << stats.stddev << std::endl;
  std::cout << "  Min:     " << std::setprecision(precision) << stats.min << std::endl;
  std::cout << "  Max:     " << std::setprecision(precision) << stats.max << std::endl;
}

// Print bandwidth statistics for a cache level (L1, L2, or Custom)
static void print_cache_bandwidth_statistics(const std::string &cache_name,
                                              const std::vector<double> &read_bw,
                                              const std::vector<double> &write_bw,
                                              const std::vector<double> &copy_bw) {
  if (read_bw.empty() && write_bw.empty() && copy_bw.empty()) {
    return;
  }
  
  std::cout << "\n" << cache_name << " Cache Bandwidth (GB/s):" << std::endl;
  
  if (!read_bw.empty()) {
    Statistics read_stats = calculate_statistics(read_bw);
    std::cout << "  Read:" << std::endl;
    std::cout << "    Average: " << std::setprecision(3) << read_stats.average << std::endl;
    std::cout << "    Median (P50): " << std::setprecision(3) << read_stats.median << std::endl;
    std::cout << "    P90: " << std::setprecision(3) << read_stats.p90 << std::endl;
    std::cout << "    P95: " << std::setprecision(3) << read_stats.p95 << std::endl;
    std::cout << "    P99: " << std::setprecision(3) << read_stats.p99 << std::endl;
    std::cout << "    Stddev: " << std::setprecision(3) << read_stats.stddev << std::endl;
    std::cout << "    Min:     " << std::setprecision(3) << read_stats.min << std::endl;
    std::cout << "    Max:     " << std::setprecision(3) << read_stats.max << std::endl;
  }
  
  if (!write_bw.empty()) {
    Statistics write_stats = calculate_statistics(write_bw);
    std::cout << "  Write:" << std::endl;
    std::cout << "    Average: " << std::setprecision(3) << write_stats.average << std::endl;
    std::cout << "    Median (P50): " << std::setprecision(3) << write_stats.median << std::endl;
    std::cout << "    P90: " << std::setprecision(3) << write_stats.p90 << std::endl;
    std::cout << "    P95: " << std::setprecision(3) << write_stats.p95 << std::endl;
    std::cout << "    P99: " << std::setprecision(3) << write_stats.p99 << std::endl;
    std::cout << "    Stddev: " << std::setprecision(3) << write_stats.stddev << std::endl;
    std::cout << "    Min:     " << std::setprecision(3) << write_stats.min << std::endl;
    std::cout << "    Max:     " << std::setprecision(3) << write_stats.max << std::endl;
  }
  
  if (!copy_bw.empty()) {
    Statistics copy_stats = calculate_statistics(copy_bw);
    std::cout << "  Copy:" << std::endl;
    std::cout << "    Average: " << std::setprecision(3) << copy_stats.average << std::endl;
    std::cout << "    Median (P50): " << std::setprecision(3) << copy_stats.median << std::endl;
    std::cout << "    P90: " << std::setprecision(3) << copy_stats.p90 << std::endl;
    std::cout << "    P95: " << std::setprecision(3) << copy_stats.p95 << std::endl;
    std::cout << "    P99: " << std::setprecision(3) << copy_stats.p99 << std::endl;
    std::cout << "    Stddev: " << std::setprecision(3) << copy_stats.stddev << std::endl;
    std::cout << "    Min:     " << std::setprecision(3) << copy_stats.min << std::endl;
    std::cout << "    Max:     " << std::setprecision(3) << copy_stats.max << std::endl;
  }
}

// Print latency statistics for a cache level (L1, L2, or Custom)
// Uses full sample distribution for percentiles if available, otherwise uses loop averages
static void print_cache_latency_statistics(const std::string &cache_name,
                                            const std::vector<double> &latency,
                                            const std::vector<double> &latency_samples) {
  if (latency.empty()) {
    return;
  }
  
  // Calculate statistics from loop averages (for average, min, max)
  Statistics latency_stats = calculate_statistics(latency);
  
  // Use full sample distribution for percentiles if available
  Statistics sample_stats;
  bool use_samples = !latency_samples.empty();
  if (use_samples) {
    sample_stats = calculate_statistics(latency_samples);
  }
  
  std::cout << "  " << cache_name << " Cache:" << std::endl;
  std::cout << "    Average: " << std::setprecision(2) << latency_stats.average << std::endl;
  if (use_samples) {
    std::cout << "    Median (P50): " << std::setprecision(2) << sample_stats.median << " (from " << latency_samples.size() << " samples)" << std::endl;
    std::cout << "    P90: " << std::setprecision(2) << sample_stats.p90 << std::endl;
    std::cout << "    P95: " << std::setprecision(2) << sample_stats.p95 << std::endl;
    std::cout << "    P99: " << std::setprecision(2) << sample_stats.p99 << std::endl;
    std::cout << "    Stddev: " << std::setprecision(2) << sample_stats.stddev << std::endl;
    std::cout << "    Min:     " << std::setprecision(2) << sample_stats.min << std::endl;
    std::cout << "    Max:     " << std::setprecision(2) << sample_stats.max << std::endl;
  } else {
    std::cout << "    Median (P50): " << std::setprecision(2) << latency_stats.median << std::endl;
    std::cout << "    P90: " << std::setprecision(2) << latency_stats.p90 << std::endl;
    std::cout << "    P95: " << std::setprecision(2) << latency_stats.p95 << std::endl;
    std::cout << "    P99: " << std::setprecision(2) << latency_stats.p99 << std::endl;
    std::cout << "    Stddev: " << std::setprecision(2) << latency_stats.stddev << std::endl;
    std::cout << "    Min:     " << std::setprecision(2) << latency_stats.min << std::endl;
    std::cout << "    Max:     " << std::setprecision(2) << latency_stats.max << std::endl;
  }
}

// --- Print Statistics across all loops ---
// Calculates and displays summary statistics (Avg/Min/Max) if more than one loop was run.
// 'loop_count': The total number of loops that were executed.
// 'all_read_bw', 'all_write_bw', 'all_copy_bw': Vectors holding bandwidth results from each individual loop.
// 'all_l1_latency', 'all_l2_latency': Vectors holding cache latency results from each loop.
// 'all_l1_read_bw', 'all_l1_write_bw', 'all_l1_copy_bw': Vectors holding L1 cache bandwidth results from each loop.
// 'all_l2_read_bw', 'all_l2_write_bw', 'all_l2_copy_bw': Vectors holding L2 cache bandwidth results from each loop.
// 'all_main_mem_latency': Vector holding main memory latency results from each loop.
// 'use_custom_cache_size': Flag indicating if custom cache size is being used.
// 'all_custom_latency', 'all_custom_read_bw', 'all_custom_write_bw', 'all_custom_copy_bw': Custom cache results vectors.
// 'all_main_mem_latency_samples', 'all_l1_latency_samples', 'all_l2_latency_samples', 'all_custom_latency_samples': Full sample distributions.
void print_statistics(int loop_count, const std::vector<double> &all_read_bw, const std::vector<double> &all_write_bw,
                      const std::vector<double> &all_copy_bw, 
                      const std::vector<double> &all_l1_latency, const std::vector<double> &all_l2_latency,
                      const std::vector<double> &all_l1_read_bw, const std::vector<double> &all_l1_write_bw,
                      const std::vector<double> &all_l1_copy_bw,
                      const std::vector<double> &all_l2_read_bw, const std::vector<double> &all_l2_write_bw,
                      const std::vector<double> &all_l2_copy_bw,
                      const std::vector<double> &all_main_mem_latency,
                      bool use_custom_cache_size,
                      const std::vector<double> &all_custom_latency,
                      const std::vector<double> &all_custom_read_bw,
                      const std::vector<double> &all_custom_write_bw,
                      const std::vector<double> &all_custom_copy_bw,
                      const std::vector<double> &all_main_mem_latency_samples,
                      const std::vector<double> &all_l1_latency_samples,
                      const std::vector<double> &all_l2_latency_samples,
                      const std::vector<double> &all_custom_latency_samples) {
  // Don't print statistics if only one loop ran or if there's no data.
  if (loop_count <= 1 || all_read_bw.empty()) return;

  // Print statistics header.
  std::cout << "\n--- Statistics Across " << loop_count << " Loops ---" << std::endl;
  std::cout << std::fixed;

  // Display Main Memory Bandwidth statistics.
  Statistics read_stats = calculate_statistics(all_read_bw);
  Statistics write_stats = calculate_statistics(all_write_bw);
  Statistics copy_stats = calculate_statistics(all_copy_bw);
  
  print_metric_statistics("Read Bandwidth (GB/s)", read_stats);
  std::cout << "\n";
  print_metric_statistics("Write Bandwidth (GB/s)", write_stats);
  std::cout << "\n";
  print_metric_statistics("Copy Bandwidth (GB/s)", copy_stats);

  // Display Cache Bandwidth statistics.
  if (use_custom_cache_size) {
    print_cache_bandwidth_statistics("Custom", all_custom_read_bw, all_custom_write_bw, all_custom_copy_bw);
  } else {
    print_cache_bandwidth_statistics("L1", all_l1_read_bw, all_l1_write_bw, all_l1_copy_bw);
    print_cache_bandwidth_statistics("L2", all_l2_read_bw, all_l2_write_bw, all_l2_copy_bw);
  }

  // Display Cache Latency statistics.
  std::cout << "\nCache Latency (ns):" << std::endl;
  if (use_custom_cache_size) {
    print_cache_latency_statistics("Custom", all_custom_latency, all_custom_latency_samples);
  } else {
    print_cache_latency_statistics("L1", all_l1_latency, all_l1_latency_samples);
    print_cache_latency_statistics("L2", all_l2_latency, all_l2_latency_samples);
  }

  // Display Main Memory Latency statistics.
  Statistics main_mem_latency_stats = calculate_statistics(all_main_mem_latency);
  bool use_main_mem_samples = !all_main_mem_latency_samples.empty();
  Statistics main_mem_sample_stats;
  if (use_main_mem_samples) {
    main_mem_sample_stats = calculate_statistics(all_main_mem_latency_samples);
  }
  
  std::cout << "\nMain Memory Latency (ns):" << std::endl;
  std::cout << "  Average: " << std::setprecision(2) << main_mem_latency_stats.average << std::endl;
  if (use_main_mem_samples) {
    std::cout << "  Median (P50): " << std::setprecision(2) << main_mem_sample_stats.median << " (from " << all_main_mem_latency_samples.size() << " samples)" << std::endl;
    std::cout << "  P90: " << std::setprecision(2) << main_mem_sample_stats.p90 << std::endl;
    std::cout << "  P95: " << std::setprecision(2) << main_mem_sample_stats.p95 << std::endl;
    std::cout << "  P99: " << std::setprecision(2) << main_mem_sample_stats.p99 << std::endl;
    std::cout << "  Stddev: " << std::setprecision(2) << main_mem_sample_stats.stddev << std::endl;
    std::cout << "  Min:     " << std::setprecision(2) << main_mem_sample_stats.min << std::endl;
    std::cout << "  Max:     " << std::setprecision(2) << main_mem_sample_stats.max << std::endl;
  } else {
    std::cout << "  Median (P50): " << std::setprecision(2) << main_mem_latency_stats.median << std::endl;
    std::cout << "  P90: " << std::setprecision(2) << main_mem_latency_stats.p90 << std::endl;
    std::cout << "  P95: " << std::setprecision(2) << main_mem_latency_stats.p95 << std::endl;
    std::cout << "  P99: " << std::setprecision(2) << main_mem_latency_stats.p99 << std::endl;
    std::cout << "  Stddev: " << std::setprecision(2) << main_mem_latency_stats.stddev << std::endl;
    std::cout << "  Min:     " << std::setprecision(2) << main_mem_latency_stats.min << std::endl;
    std::cout << "  Max:     " << std::setprecision(2) << main_mem_latency_stats.max << std::endl;
  }
  
  // Print a final separator after statistics.
  std::cout << "----------------------------------" << std::endl;
}

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
#include "pattern_benchmark/pattern_benchmark.h"
#include "core/config/constants.h"
#include "output/console/messages.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

// ============================================================================
// Output Formatting Functions
// ============================================================================

// Format percentage difference from baseline
static std::string format_percentage(double baseline, double value) {
  using namespace Constants;
  if (baseline == 0.0) return Messages::pattern_na();
  double pct = ((value - baseline) / baseline) * 100.0;
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(PATTERN_PERCENTAGE_PRECISION);
  if (pct >= 0) {
    oss << " (+" << pct << "%)";
  } else {
    oss << " (" << pct << "%)";
  }
  return oss.str();
}

// Print sequential pattern results
static void print_sequential_results(const PatternResults& results) {
  using namespace Constants;
  
  // Sequential Forward (baseline)
  std::cout << Messages::pattern_sequential_forward() << "\n";
  std::cout << Messages::pattern_read_label() << std::fixed << std::setprecision(PATTERN_BANDWIDTH_PRECISION) 
            << results.forward_read_bw << Messages::pattern_bandwidth_unit_newline();
  std::cout << Messages::pattern_write_label() << results.forward_write_bw << Messages::pattern_bandwidth_unit_newline();
  std::cout << Messages::pattern_copy_label() << results.forward_copy_bw << Messages::pattern_bandwidth_unit_newline() << "\n";
  
  // Sequential Reverse
  std::cout << Messages::pattern_sequential_reverse() << "\n";
  std::cout << Messages::pattern_read_label() << results.reverse_read_bw << Messages::pattern_bandwidth_unit()
            << format_percentage(results.forward_read_bw, results.reverse_read_bw) << "\n";
  std::cout << Messages::pattern_write_label() << results.reverse_write_bw << Messages::pattern_bandwidth_unit()
            << format_percentage(results.forward_write_bw, results.reverse_write_bw) << "\n";
  std::cout << Messages::pattern_copy_label() << results.reverse_copy_bw << Messages::pattern_bandwidth_unit()
            << format_percentage(results.forward_copy_bw, results.reverse_copy_bw) << "\n\n";
}

// Print strided pattern results
static void print_strided_results(const PatternResults& results, const std::string& stride_name, 
                                  double read_bw, double write_bw, double copy_bw) {
  std::cout << Messages::pattern_strided(stride_name) << "\n";
  std::cout << Messages::pattern_read_label() << read_bw << Messages::pattern_bandwidth_unit()
            << format_percentage(results.forward_read_bw, read_bw) << "\n";
  std::cout << Messages::pattern_write_label() << write_bw << Messages::pattern_bandwidth_unit()
            << format_percentage(results.forward_write_bw, write_bw) << "\n";
  std::cout << Messages::pattern_copy_label() << copy_bw << Messages::pattern_bandwidth_unit()
            << format_percentage(results.forward_copy_bw, copy_bw) << "\n\n";
}

// Print random pattern results
static void print_random_results(const PatternResults& results) {
  std::cout << Messages::pattern_random_uniform() << "\n";
  std::cout << Messages::pattern_read_label() << results.random_read_bw << Messages::pattern_bandwidth_unit()
            << format_percentage(results.forward_read_bw, results.random_read_bw) << "\n";
  std::cout << Messages::pattern_write_label() << results.random_write_bw << Messages::pattern_bandwidth_unit()
            << format_percentage(results.forward_write_bw, results.random_write_bw) << "\n";
  std::cout << Messages::pattern_copy_label() << results.random_copy_bw << Messages::pattern_bandwidth_unit()
            << format_percentage(results.forward_copy_bw, results.random_copy_bw) << "\n\n";
}

// Calculate pattern efficiency metrics
static void calculate_efficiency_metrics(const PatternResults& results,
                                         double& seq_coherence,
                                         double& prefetch_effectiveness,
                                         double& cache_thrashing,
                                         double& tlb_pressure) {
  using namespace Constants;
  
  double forward_total = results.forward_read_bw + results.forward_write_bw + results.forward_copy_bw;
  double reverse_total = results.reverse_read_bw + results.reverse_write_bw + results.reverse_copy_bw;
  double strided_64_total = results.strided_64_read_bw + results.strided_64_write_bw + results.strided_64_copy_bw;
  double strided_4096_total = results.strided_4096_read_bw + results.strided_4096_write_bw + results.strided_4096_copy_bw;
  double random_total = results.random_read_bw + results.random_write_bw + results.random_copy_bw;
  
  // Sequential coherence: ratio of reverse to forward
  seq_coherence = (reverse_total / forward_total) * 100.0;
  
  // Prefetcher effectiveness: ratio of strided 64B to forward (cache line stride should be well-prefetched)
  prefetch_effectiveness = (strided_64_total / forward_total) * 100.0;
  
  // Cache thrashing potential: based on strided 4096B performance (page stride causes more misses)
  cache_thrashing = (strided_4096_total / forward_total) * 100.0;
  
  // TLB pressure: based on random vs strided 4096B (random has more TLB misses)
  tlb_pressure = (random_total / strided_4096_total) * 100.0;
}

// Get cache thrashing level string
static const std::string& get_cache_thrashing_level(double cache_thrashing) {
  using namespace Constants;
  if (cache_thrashing > PATTERN_CACHE_THRASHING_HIGH_THRESHOLD) {
    return Messages::pattern_cache_thrashing_low();
  } else if (cache_thrashing > PATTERN_CACHE_THRASHING_MEDIUM_THRESHOLD) {
    return Messages::pattern_cache_thrashing_medium();
  } else {
    return Messages::pattern_cache_thrashing_high();
  }
}

// Get TLB pressure level string
static const std::string& get_tlb_pressure_level(double tlb_pressure) {
  using namespace Constants;
  if (tlb_pressure > PATTERN_TLB_PRESSURE_MINIMAL_THRESHOLD) {
    return Messages::pattern_tlb_pressure_minimal();
  } else if (tlb_pressure > PATTERN_TLB_PRESSURE_MODERATE_THRESHOLD) {
    return Messages::pattern_tlb_pressure_moderate();
  } else {
    return Messages::pattern_tlb_pressure_high();
  }
}

// Print pattern efficiency analysis
static void print_efficiency_analysis(const PatternResults& results) {
  using namespace Constants;
  
  double seq_coherence, prefetch_effectiveness, cache_thrashing, tlb_pressure;
  calculate_efficiency_metrics(results, seq_coherence, prefetch_effectiveness, 
                                cache_thrashing, tlb_pressure);
  
  std::cout << Messages::pattern_efficiency_analysis() << "\n";
  std::cout << "- " << Messages::pattern_sequential_coherence() << " " << std::fixed 
            << std::setprecision(PATTERN_PERCENTAGE_PRECISION) << seq_coherence << "%\n";
  std::cout << "- " << Messages::pattern_prefetcher_effectiveness() << " " << prefetch_effectiveness << "%\n";
  std::cout << "- " << Messages::pattern_cache_thrashing_potential() << " " 
            << get_cache_thrashing_level(cache_thrashing) << "\n";
  std::cout << "- " << Messages::pattern_tlb_pressure() << " " 
            << get_tlb_pressure_level(tlb_pressure) << "\n";
  std::cout << "\n";
}

void print_pattern_results(const PatternResults& results) {
  using namespace Constants;
  
  std::cout << Messages::pattern_separator();
  
  // Print all pattern results
  print_sequential_results(results);
  print_strided_results(results, Messages::pattern_cache_line_64b(), 
                        results.strided_64_read_bw, 
                        results.strided_64_write_bw, 
                        results.strided_64_copy_bw);
  print_strided_results(results, Messages::pattern_page_4096b(), 
                        results.strided_4096_read_bw, 
                        results.strided_4096_write_bw, 
                        results.strided_4096_copy_bw);
  print_random_results(results);
  
  // Print efficiency analysis
  print_efficiency_analysis(results);
}

// ============================================================================
// Pattern Statistics Functions
// ============================================================================

// Structure to hold calculated statistics (average, min, max, percentiles, stddev)
struct PatternStatisticsData {
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
static PatternStatisticsData calculate_pattern_statistics(const std::vector<double> &values) {
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

// Print statistics for a single pattern metric
static void print_pattern_metric_statistics(const std::string &metric_name, const PatternStatisticsData &stats) {
  using namespace Constants;
  std::cout << Messages::statistics_metric_name(metric_name) << std::endl;
  std::cout << Messages::statistics_average(stats.average, Constants::PATTERN_BANDWIDTH_PRECISION) << std::endl;
  std::cout << Messages::statistics_median_p50(stats.median, Constants::PATTERN_BANDWIDTH_PRECISION) << std::endl;
  std::cout << Messages::statistics_p90(stats.p90, Constants::PATTERN_BANDWIDTH_PRECISION) << std::endl;
  std::cout << Messages::statistics_p95(stats.p95, Constants::PATTERN_BANDWIDTH_PRECISION) << std::endl;
  std::cout << Messages::statistics_p99(stats.p99, Constants::PATTERN_BANDWIDTH_PRECISION) << std::endl;
  std::cout << Messages::statistics_stddev(stats.stddev, Constants::PATTERN_BANDWIDTH_PRECISION) << std::endl;
  std::cout << Messages::statistics_min(stats.min, Constants::PATTERN_BANDWIDTH_PRECISION) << std::endl;
  std::cout << Messages::statistics_max(stats.max, Constants::PATTERN_BANDWIDTH_PRECISION) << std::endl;
}

// Print statistics for a pattern type (read, write, copy)
static void print_pattern_type_statistics(const std::string &pattern_name,
                                          const std::vector<double> &read_bw,
                                          const std::vector<double> &write_bw,
                                          const std::vector<double> &copy_bw) {
  if (read_bw.empty() && write_bw.empty() && copy_bw.empty()) {
    return;
  }
  
  std::cout << Messages::statistics_cache_bandwidth_header(pattern_name) << std::endl;
  
  if (!read_bw.empty()) {
    PatternStatisticsData read_stats = calculate_pattern_statistics(read_bw);
    std::cout << Messages::statistics_cache_read() << std::endl;
    std::cout << "    " << Messages::statistics_average(read_stats.average, Constants::PATTERN_BANDWIDTH_PRECISION) << std::endl;
    std::cout << "    " << Messages::statistics_median_p50(read_stats.median, Constants::PATTERN_BANDWIDTH_PRECISION) << std::endl;
    std::cout << "    " << Messages::statistics_p90(read_stats.p90, Constants::PATTERN_BANDWIDTH_PRECISION) << std::endl;
    std::cout << "    " << Messages::statistics_p95(read_stats.p95, Constants::PATTERN_BANDWIDTH_PRECISION) << std::endl;
    std::cout << "    " << Messages::statistics_p99(read_stats.p99, Constants::PATTERN_BANDWIDTH_PRECISION) << std::endl;
    std::cout << "    " << Messages::statistics_stddev(read_stats.stddev, Constants::PATTERN_BANDWIDTH_PRECISION) << std::endl;
    std::cout << "    " << Messages::statistics_min(read_stats.min, Constants::PATTERN_BANDWIDTH_PRECISION) << std::endl;
    std::cout << "    " << Messages::statistics_max(read_stats.max, Constants::PATTERN_BANDWIDTH_PRECISION) << std::endl;
  }
  
  if (!write_bw.empty()) {
    PatternStatisticsData write_stats = calculate_pattern_statistics(write_bw);
    std::cout << Messages::statistics_cache_write() << std::endl;
    std::cout << "    " << Messages::statistics_average(write_stats.average, Constants::PATTERN_BANDWIDTH_PRECISION) << std::endl;
    std::cout << "    " << Messages::statistics_median_p50(write_stats.median, Constants::PATTERN_BANDWIDTH_PRECISION) << std::endl;
    std::cout << "    " << Messages::statistics_p90(write_stats.p90, Constants::PATTERN_BANDWIDTH_PRECISION) << std::endl;
    std::cout << "    " << Messages::statistics_p95(write_stats.p95, Constants::PATTERN_BANDWIDTH_PRECISION) << std::endl;
    std::cout << "    " << Messages::statistics_p99(write_stats.p99, Constants::PATTERN_BANDWIDTH_PRECISION) << std::endl;
    std::cout << "    " << Messages::statistics_stddev(write_stats.stddev, Constants::PATTERN_BANDWIDTH_PRECISION) << std::endl;
    std::cout << "    " << Messages::statistics_min(write_stats.min, Constants::PATTERN_BANDWIDTH_PRECISION) << std::endl;
    std::cout << "    " << Messages::statistics_max(write_stats.max, Constants::PATTERN_BANDWIDTH_PRECISION) << std::endl;
  }
  
  if (!copy_bw.empty()) {
    PatternStatisticsData copy_stats = calculate_pattern_statistics(copy_bw);
    std::cout << Messages::statistics_cache_copy() << std::endl;
    std::cout << "    " << Messages::statistics_average(copy_stats.average, Constants::PATTERN_BANDWIDTH_PRECISION) << std::endl;
    std::cout << "    " << Messages::statistics_median_p50(copy_stats.median, Constants::PATTERN_BANDWIDTH_PRECISION) << std::endl;
    std::cout << "    " << Messages::statistics_p90(copy_stats.p90, Constants::PATTERN_BANDWIDTH_PRECISION) << std::endl;
    std::cout << "    " << Messages::statistics_p95(copy_stats.p95, Constants::PATTERN_BANDWIDTH_PRECISION) << std::endl;
    std::cout << "    " << Messages::statistics_p99(copy_stats.p99, Constants::PATTERN_BANDWIDTH_PRECISION) << std::endl;
    std::cout << "    " << Messages::statistics_stddev(copy_stats.stddev, Constants::PATTERN_BANDWIDTH_PRECISION) << std::endl;
    std::cout << "    " << Messages::statistics_min(copy_stats.min, Constants::PATTERN_BANDWIDTH_PRECISION) << std::endl;
    std::cout << "    " << Messages::statistics_max(copy_stats.max, Constants::PATTERN_BANDWIDTH_PRECISION) << std::endl;
  }
}

void print_pattern_statistics(int loop_count, const PatternStatistics& stats) {
  using namespace Constants;
  
  // Don't print statistics if only one loop ran or if there's no data
  if (loop_count <= 1 || stats.all_forward_read_bw.empty()) {
    return;
  }

  // Print statistics header
  std::cout << Messages::statistics_header(loop_count) << std::endl;
  std::cout << std::fixed;

  // Display Sequential Forward (baseline) statistics
  std::string forward_name = Messages::pattern_sequential_forward();
  if (!forward_name.empty() && forward_name.back() == ':') {
    forward_name.pop_back();
  }
  print_pattern_type_statistics(forward_name, 
                                 stats.all_forward_read_bw,
                                 stats.all_forward_write_bw,
                                 stats.all_forward_copy_bw);
  std::cout << "\n";

  // Display Sequential Reverse statistics
  std::string reverse_name = Messages::pattern_sequential_reverse();
  if (!reverse_name.empty() && reverse_name.back() == ':') {
    reverse_name.pop_back();
  }
  print_pattern_type_statistics(reverse_name,
                                 stats.all_reverse_read_bw,
                                 stats.all_reverse_write_bw,
                                 stats.all_reverse_copy_bw);
  std::cout << "\n";

  // Display Strided 64B statistics
  std::string strided_64_name = Messages::pattern_strided(Messages::pattern_cache_line_64b());
  if (!strided_64_name.empty() && strided_64_name.back() == ':') {
    strided_64_name.pop_back();
  }
  print_pattern_type_statistics(strided_64_name,
                                 stats.all_strided_64_read_bw,
                                 stats.all_strided_64_write_bw,
                                 stats.all_strided_64_copy_bw);
  std::cout << "\n";

  // Display Strided 4096B statistics
  std::string strided_4096_name = Messages::pattern_strided(Messages::pattern_page_4096b());
  if (!strided_4096_name.empty() && strided_4096_name.back() == ':') {
    strided_4096_name.pop_back();
  }
  print_pattern_type_statistics(strided_4096_name,
                                 stats.all_strided_4096_read_bw,
                                 stats.all_strided_4096_write_bw,
                                 stats.all_strided_4096_copy_bw);
  std::cout << "\n";

  // Display Random Uniform statistics
  std::string random_name = Messages::pattern_random_uniform();
  if (!random_name.empty() && random_name.back() == ':') {
    random_name.pop_back();
  }
  print_pattern_type_statistics(random_name,
                                 stats.all_random_read_bw,
                                 stats.all_random_write_bw,
                                 stats.all_random_copy_bw);
  
  // Print a final separator after statistics
  std::cout << Messages::statistics_footer() << std::endl;
}


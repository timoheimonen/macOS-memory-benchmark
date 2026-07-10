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
/**
 * @file output.cpp
 * @brief Console output formatting for pattern benchmark results
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2025
 *
 * This file provides functions for formatting and displaying pattern benchmark
 * results to the console. It handles result presentation, statistical analysis,
 * and efficiency metrics calculation.
 *
 * Output features:
 * - Individual pattern results with percentage comparisons to baseline
 * - Statistical analysis across multiple loops (average, percentiles, stddev)
 * - Explicit unavailable statuses and robust repeated-loop summaries
 * - Formatted tables with appropriate precision and units
 */
#include "pattern_benchmark/pattern_benchmark.h"
#include "core/config/constants.h"
#include "output/console/messages/messages_api.h"
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

// Format percentage difference from a measured baseline.
static std::string format_percentage(const PatternMeasurement& baseline,
                                     const PatternMeasurement& value) {
  using namespace Constants;
  if (!baseline.bandwidth_gb_s.has_value() || !value.bandwidth_gb_s.has_value() ||
      *baseline.bandwidth_gb_s == 0.0) {
    return {};
  }
  const double pct = ((*value.bandwidth_gb_s - *baseline.bandwidth_gb_s) /
                      *baseline.bandwidth_gb_s) *
                     100.0;
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(PATTERN_PERCENTAGE_PRECISION);
  if (pct >= 0) {
    oss << " (+" << pct << "%)";
  } else {
    oss << " (" << pct << "%)";
  }
  return oss.str();
}

static void print_measurement_line(const std::string& label,
                                   const PatternMeasurement& measurement,
                                   const PatternMeasurement* baseline = nullptr) {
  std::cout << label;
  if (measurement.status != PatternMeasurementStatus::Measured ||
      !measurement.bandwidth_gb_s.has_value()) {
    std::cout << Messages::pattern_measurement_unavailable(
                     pattern_measurement_status_to_string(measurement.status),
                     measurement.status_reason)
              << "\n";
    return;
  }

  std::cout << std::fixed << std::setprecision(Constants::PATTERN_BANDWIDTH_PRECISION)
            << *measurement.bandwidth_gb_s << Messages::pattern_bandwidth_unit();
  if (baseline != nullptr) {
    std::cout << format_percentage(*baseline, measurement);
  }
  std::cout << "\n";
}

// Print sequential pattern results
static void print_sequential_results(const PatternResults& results) {
  std::cout << Messages::pattern_sequential_forward() << "\n";
  const PatternMeasurement& forward_read = get_pattern_measurement(
      results, PatternKind::SequentialForward, PatternOperation::Read);
  const PatternMeasurement& forward_write = get_pattern_measurement(
      results, PatternKind::SequentialForward, PatternOperation::Write);
  const PatternMeasurement& forward_copy = get_pattern_measurement(
      results, PatternKind::SequentialForward, PatternOperation::Copy);
  print_measurement_line(Messages::pattern_read_label(), forward_read);
  print_measurement_line(Messages::pattern_write_label(), forward_write);
  print_measurement_line(Messages::pattern_copy_label(), forward_copy);
  std::cout << "\n";

  std::cout << Messages::pattern_sequential_reverse() << "\n";
  print_measurement_line(
      Messages::pattern_read_label(),
      get_pattern_measurement(results, PatternKind::SequentialReverse,
                              PatternOperation::Read),
      &forward_read);
  print_measurement_line(
      Messages::pattern_write_label(),
      get_pattern_measurement(results, PatternKind::SequentialReverse,
                              PatternOperation::Write),
      &forward_write);
  print_measurement_line(
      Messages::pattern_copy_label(),
      get_pattern_measurement(results, PatternKind::SequentialReverse,
                              PatternOperation::Copy),
      &forward_copy);
  std::cout << "\n";
}

// Print strided pattern results
static void print_strided_results(const PatternResults& results,
                                  const std::string& stride_name,
                                  PatternKind kind) {
  std::cout << Messages::pattern_strided(stride_name) << "\n";
  print_measurement_line(
      Messages::pattern_read_label(),
      get_pattern_measurement(results, kind, PatternOperation::Read),
      &get_pattern_measurement(results, PatternKind::SequentialForward,
                               PatternOperation::Read));
  print_measurement_line(
      Messages::pattern_write_label(),
      get_pattern_measurement(results, kind, PatternOperation::Write),
      &get_pattern_measurement(results, PatternKind::SequentialForward,
                               PatternOperation::Write));
  print_measurement_line(
      Messages::pattern_copy_label(),
      get_pattern_measurement(results, kind, PatternOperation::Copy),
      &get_pattern_measurement(results, PatternKind::SequentialForward,
                               PatternOperation::Copy));
  std::cout << "\n";
}

// Print random pattern results
static void print_random_results(const PatternResults& results) {
  std::cout << Messages::pattern_random_uniform() << "\n";
  print_measurement_line(
      Messages::pattern_read_label(),
      get_pattern_measurement(results, PatternKind::Random, PatternOperation::Read),
      &get_pattern_measurement(results, PatternKind::SequentialForward,
                               PatternOperation::Read));
  print_measurement_line(
      Messages::pattern_write_label(),
      get_pattern_measurement(results, PatternKind::Random, PatternOperation::Write),
      &get_pattern_measurement(results, PatternKind::SequentialForward,
                               PatternOperation::Write));
  print_measurement_line(
      Messages::pattern_copy_label(),
      get_pattern_measurement(results, PatternKind::Random, PatternOperation::Copy),
      &get_pattern_measurement(results, PatternKind::SequentialForward,
                               PatternOperation::Copy));
  std::cout << "\n";
}

void print_pattern_results(const PatternResults& results) {
  using namespace Constants;
  
  std::cout << Messages::pattern_separator();
  
  // Print all pattern results
  print_sequential_results(results);
  print_strided_results(results, Messages::pattern_cache_line_64b(),
                        PatternKind::Strided64);
  print_strided_results(results, Messages::pattern_page_4096b(),
                        PatternKind::Strided4096);
  print_strided_results(results, Messages::pattern_page_16384b(),
                        PatternKind::Strided16384);
  print_strided_results(results, Messages::pattern_superpage_2mb(),
                        PatternKind::Strided2MiB);
  print_random_results(results);
}

// ============================================================================
// Pattern Statistics Functions
// ============================================================================

// Print statistics for a pattern type (read, write, copy)
static void print_pattern_type_statistics(const std::string &pattern_name,
                                          const std::vector<double> &read_bw,
                                          const std::vector<double> &write_bw,
                                          const std::vector<double> &copy_bw,
                                          double cv_threshold_pct,
                                          std::vector<std::string>& noise_warnings) {
  if (read_bw.empty() && write_bw.empty() && copy_bw.empty()) {
    return;
  }
  
  std::cout << Messages::statistics_pattern_bandwidth_header(pattern_name) << std::endl;
  
  if (!read_bw.empty()) {
    PatternStatisticsData read_stats = calculate_pattern_statistics(read_bw);
    std::cout << Messages::statistics_cache_read() << std::endl;
    std::cout << "    " << Messages::statistics_average(read_stats.average, Constants::PATTERN_BANDWIDTH_PRECISION) << std::endl;
    std::cout << "    " << Messages::statistics_median_p50(read_stats.median, Constants::PATTERN_BANDWIDTH_PRECISION) << std::endl;
    std::cout << "    " << Messages::statistics_p90(read_stats.p90, Constants::PATTERN_BANDWIDTH_PRECISION) << std::endl;
    std::cout << "    " << Messages::statistics_p95(read_stats.p95, Constants::PATTERN_BANDWIDTH_PRECISION) << std::endl;
    std::cout << "    " << Messages::statistics_p99(read_stats.p99, Constants::PATTERN_BANDWIDTH_PRECISION) << std::endl;
    std::cout << "    " << Messages::statistics_stddev(read_stats.stddev, Constants::PATTERN_BANDWIDTH_PRECISION) << std::endl;
    std::cout << "    " << Messages::statistics_coefficient_of_variation(
                                  read_stats.coefficient_of_variation_pct)
              << std::endl;
    std::cout << "    " << Messages::statistics_median_absolute_deviation(
                                  read_stats.median_absolute_deviation,
                                  Constants::PATTERN_BANDWIDTH_PRECISION)
              << std::endl;
    if (read_stats.coefficient_of_variation_pct > cv_threshold_pct) {
      noise_warnings.push_back(Messages::warning_pattern_measurement_noisy(
          pattern_name + " read", read_stats.coefficient_of_variation_pct,
          cv_threshold_pct));
    }
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
    std::cout << "    " << Messages::statistics_coefficient_of_variation(
                                  write_stats.coefficient_of_variation_pct)
              << std::endl;
    std::cout << "    " << Messages::statistics_median_absolute_deviation(
                                  write_stats.median_absolute_deviation,
                                  Constants::PATTERN_BANDWIDTH_PRECISION)
              << std::endl;
    if (write_stats.coefficient_of_variation_pct > cv_threshold_pct) {
      noise_warnings.push_back(Messages::warning_pattern_measurement_noisy(
          pattern_name + " write", write_stats.coefficient_of_variation_pct,
          cv_threshold_pct));
    }
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
    std::cout << "    " << Messages::statistics_coefficient_of_variation(
                                  copy_stats.coefficient_of_variation_pct)
              << std::endl;
    std::cout << "    " << Messages::statistics_median_absolute_deviation(
                                  copy_stats.median_absolute_deviation,
                                  Constants::PATTERN_BANDWIDTH_PRECISION)
              << std::endl;
    if (copy_stats.coefficient_of_variation_pct > cv_threshold_pct) {
      noise_warnings.push_back(Messages::warning_pattern_measurement_noisy(
          pattern_name + " copy", copy_stats.coefficient_of_variation_pct,
          cv_threshold_pct));
    }
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
  std::vector<std::string> noise_warnings;

  // Display Sequential Forward (baseline) statistics
  std::string forward_name = Messages::pattern_sequential_forward();
  if (!forward_name.empty() && forward_name.back() == ':') {
    forward_name.pop_back();
  }
  print_pattern_type_statistics(forward_name, 
                                 stats.all_forward_read_bw,
                                 stats.all_forward_write_bw,
                                 stats.all_forward_copy_bw,
                                 PATTERN_STREAMING_CV_WARNING_PCT,
                                 noise_warnings);
  std::cout << "\n";

  // Display Sequential Reverse statistics
  std::string reverse_name = Messages::pattern_sequential_reverse();
  if (!reverse_name.empty() && reverse_name.back() == ':') {
    reverse_name.pop_back();
  }
  print_pattern_type_statistics(reverse_name,
                                 stats.all_reverse_read_bw,
                                 stats.all_reverse_write_bw,
                                 stats.all_reverse_copy_bw,
                                 PATTERN_STREAMING_CV_WARNING_PCT,
                                 noise_warnings);
  std::cout << "\n";

  // Display Strided 64B statistics
  std::string strided_64_name = Messages::pattern_strided(Messages::pattern_cache_line_64b());
  if (!strided_64_name.empty() && strided_64_name.back() == ':') {
    strided_64_name.pop_back();
  }
  print_pattern_type_statistics(strided_64_name,
                                 stats.all_strided_64_read_bw,
                                 stats.all_strided_64_write_bw,
                                 stats.all_strided_64_copy_bw,
                                 PATTERN_STREAMING_CV_WARNING_PCT,
                                 noise_warnings);
  std::cout << "\n";

  // Display Strided 4096B statistics
  std::string strided_4096_name = Messages::pattern_strided(Messages::pattern_page_4096b());
  if (!strided_4096_name.empty() && strided_4096_name.back() == ':') {
    strided_4096_name.pop_back();
  }
  print_pattern_type_statistics(strided_4096_name,
                                 stats.all_strided_4096_read_bw,
                                 stats.all_strided_4096_write_bw,
                                 stats.all_strided_4096_copy_bw,
                                 PATTERN_SPARSE_CV_WARNING_PCT,
                                 noise_warnings);
  std::cout << "\n";

  // Display Strided 16384B statistics
  std::string strided_16384_name = Messages::pattern_strided(Messages::pattern_page_16384b());
  if (!strided_16384_name.empty() && strided_16384_name.back() == ':') {
    strided_16384_name.pop_back();
  }
  print_pattern_type_statistics(strided_16384_name,
                                 stats.all_strided_16384_read_bw,
                                 stats.all_strided_16384_write_bw,
                                 stats.all_strided_16384_copy_bw,
                                 PATTERN_SPARSE_CV_WARNING_PCT,
                                 noise_warnings);
  std::cout << "\n";

  // Display Strided 2MB statistics
  std::string strided_2mb_name = Messages::pattern_strided(Messages::pattern_superpage_2mb());
  if (!strided_2mb_name.empty() && strided_2mb_name.back() == ':') {
    strided_2mb_name.pop_back();
  }
  print_pattern_type_statistics(strided_2mb_name,
                                 stats.all_strided_2mb_read_bw,
                                 stats.all_strided_2mb_write_bw,
                                 stats.all_strided_2mb_copy_bw,
                                 PATTERN_SPARSE_CV_WARNING_PCT,
                                 noise_warnings);
  std::cout << "\n";

  // Display Random Uniform statistics
  std::string random_name = Messages::pattern_random_uniform();
  if (!random_name.empty() && random_name.back() == ':') {
    random_name.pop_back();
  }
  print_pattern_type_statistics(random_name,
                                 stats.all_random_read_bw,
                                 stats.all_random_write_bw,
                                 stats.all_random_copy_bw,
                                 PATTERN_SPARSE_CV_WARNING_PCT,
                                 noise_warnings);
  
  // Print a final separator after statistics
  std::cout << Messages::statistics_footer() << std::endl;
  std::cout.flush();
  for (const std::string& warning : noise_warnings) {
    std::cerr << Messages::warning_prefix() << warning << std::endl;
  }
}

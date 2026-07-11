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
 * @file statistics.cpp
 * @brief Statistics calculation and display
 */

#include "output/console/statistics.h"

#include "core/config/constants.h"
#include "output/console/messages/messages_api.h"
#include "output/console/statistics_renderer.h"
#include "utils/descriptive_statistics.h"

#include <algorithm>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <string>
#include <vector>

/**
 * @brief Render a summary with standard benchmark variability diagnostics.
 */
static void print_statistics_summary(
    const DescriptiveStatistics& statistics, int precision,
    const std::string& line_prefix, const std::string& variability_prefix,
    const std::string& metric_name, bool warn_high_cv = true,
    size_t sample_count = 0) {
  StatisticsSummaryRenderOptions options;
  options.precision = precision;
  options.line_prefix = line_prefix;
  options.variability_prefix = variability_prefix;
  options.median_from_samples = sample_count != 0;
  options.sample_count = sample_count;
  if (warn_high_cv && statistics.coefficient_of_variation_pct >
                          Constants::BENCHMARK_CV_WARNING_PCT) {
    options.inline_diagnostic =
        Messages::warning_prefix() +
        Messages::warning_benchmark_high_cv(
            metric_name, statistics.coefficient_of_variation_pct,
            Constants::BENCHMARK_CV_WARNING_PCT);
  }
  render_statistics_summary(std::cout, statistics, options);
}

/**
 * @brief Print statistics for a single metric (used for main memory bandwidth).
 *
 * @param metric_name Name of the metric being displayed
 * @param statistics Calculated descriptive statistics
 * @param precision Output precision for formatting (default: BANDWIDTH_PRECISION)
 */
static void print_metric_statistics(
    const std::string& metric_name, const DescriptiveStatistics& statistics,
    int precision = Constants::BANDWIDTH_PRECISION) {
  std::cout << Messages::statistics_metric_name(metric_name) << std::endl;
  print_statistics_summary(statistics, precision, "", "", metric_name);
}

/**
 * @brief Print bandwidth statistics for a cache level (L1, L2, or Custom).
 *
 * @param cache_name Name of the cache level (e.g., "L1", "L2", "Custom")
 * @param read_bw Vector of read bandwidth values
 * @param write_bw Vector of write bandwidth values
 * @param copy_bw Vector of copy bandwidth values
 */
static void print_cache_bandwidth_statistics(const std::string &cache_name,
                                              const std::vector<double> &read_bw,
                                              const std::vector<double> &write_bw,
                                              const std::vector<double> &copy_bw) {
  if (read_bw.empty() && write_bw.empty() && copy_bw.empty()) {
    return;
  }
  
  std::cout << Messages::statistics_cache_bandwidth_header(cache_name) << std::endl;

  const auto print_population = [&cache_name](
                                    const std::vector<double>& values,
                                    const std::string& heading,
                                    const std::string& operation_name) {
    if (values.empty()) {
      return;
    }
    std::cout << heading << std::endl;
    print_statistics_summary(calculate_descriptive_statistics(values),
                             Constants::BANDWIDTH_PRECISION, "    ", "  ",
                             cache_name + " " + operation_name +
                                 " bandwidth");
  };

  print_population(read_bw, Messages::statistics_cache_read(), "read");
  print_population(write_bw, Messages::statistics_cache_write(), "write");
  print_population(copy_bw, Messages::statistics_cache_copy(), "copy");
}

/**
 * @brief Print latency statistics for a cache level (L1, L2, or Custom).
 *
 * Loop-headline repeatability and the optional separate sample-window
 * distribution are printed as distinct statistical populations.
 *
 * @param cache_name Name of the cache level (e.g., "L1", "L2", "Custom")
 * @param latency Vector of latency values (loop averages)
 * @param latency_samples Vector of full sample distribution (if available)
 */
static void print_cache_latency_statistics(const std::string &cache_name,
                                            const std::vector<double> &latency,
                                            const std::vector<double> &latency_samples) {
  if (latency.empty()) {
    return;
  }
  
  // Calculate repeatability only from continuous per-loop headlines.
  DescriptiveStatistics latency_stats =
      calculate_descriptive_statistics(latency);
  
  // Use full sample distribution for percentiles if available
  DescriptiveStatistics sample_stats{};
  bool use_samples = !latency_samples.empty();
  if (use_samples) {
    sample_stats = calculate_descriptive_statistics(latency_samples);
  }
  
  std::cout << Messages::statistics_cache_latency_name(cache_name) << std::endl;
  print_statistics_summary(latency_stats, Constants::LATENCY_PRECISION,
                           "    ", "  ", cache_name + " latency");

  if (use_samples) {
    std::cout << Messages::statistics_pooled_sample_distribution(
                     latency_samples.size())
              << std::endl;
    print_statistics_summary(sample_stats, Constants::LATENCY_PRECISION,
                             "  ", "  ", cache_name + " latency samples",
                             false, latency_samples.size());
  }
}

/**
 * @brief Calculates and displays summary statistics if more than one loop was run.
 *
 * @param loop_count The total number of loops that were executed
 * @param all_read_bw Vector holding read bandwidth results from each loop
 * @param all_write_bw Vector holding write bandwidth results from each loop
 * @param all_copy_bw Vector holding copy bandwidth results from each loop
 * @param all_l1_latency Vector holding L1 cache latency results from each loop
 * @param all_l2_latency Vector holding L2 cache latency results from each loop
 * @param all_l1_read_bw Vector holding L1 cache read bandwidth results from each loop
 * @param all_l1_write_bw Vector holding L1 cache write bandwidth results from each loop
 * @param all_l1_copy_bw Vector holding L1 cache copy bandwidth results from each loop
 * @param all_l2_read_bw Vector holding L2 cache read bandwidth results from each loop
 * @param all_l2_write_bw Vector holding L2 cache write bandwidth results from each loop
 * @param all_l2_copy_bw Vector holding L2 cache copy bandwidth results from each loop
 * @param all_main_mem_latency Vector holding main memory latency results from each loop
 * @param all_tlb_hit_latency Legacy internal name for 16 KiB locality latency results
 * @param all_tlb_miss_latency Legacy internal name for global-random latency results
 * @param all_page_walk_penalty Legacy internal name for paired global-random minus 16 KiB locality deltas
 * @param use_custom_cache_size Flag indicating if custom cache size is being used
 * @param all_custom_latency Vector holding custom cache latency results from each loop
 * @param all_custom_read_bw Vector holding custom cache read bandwidth results from each loop
 * @param all_custom_write_bw Vector holding custom cache write bandwidth results from each loop
 * @param all_custom_copy_bw Vector holding custom cache copy bandwidth results from each loop
 * @param all_main_mem_latency_samples Full sample distribution for main memory latency
 * @param all_l1_latency_samples Full sample distribution for L1 cache latency
 * @param all_l2_latency_samples Full sample distribution for L2 cache latency
 * @param all_custom_latency_samples Full sample distribution for custom cache latency
 * @param only_bandwidth Whether only bandwidth tests are run
 * @param only_latency Whether only latency tests are run
 */
void print_statistics(int loop_count, const std::vector<double> &all_read_bw, const std::vector<double> &all_write_bw,
                      const std::vector<double> &all_copy_bw,
                      const std::vector<double> &all_l1_latency, const std::vector<double> &all_l2_latency,
                      const std::vector<double> &all_l1_read_bw, const std::vector<double> &all_l1_write_bw,
                      const std::vector<double> &all_l1_copy_bw,
                      const std::vector<double> &all_l2_read_bw, const std::vector<double> &all_l2_write_bw,
                      const std::vector<double> &all_l2_copy_bw,
                      const std::vector<double> &all_main_mem_latency,
                      const std::vector<double> &all_tlb_hit_latency,
                      const std::vector<double> &all_tlb_miss_latency,
                      const std::vector<double> &all_page_walk_penalty,
                      bool use_custom_cache_size,
                      const std::vector<double> &all_custom_latency,
                      const std::vector<double> &all_custom_read_bw,
                      const std::vector<double> &all_custom_write_bw,
                      const std::vector<double> &all_custom_copy_bw,
                      const std::vector<double> &all_main_mem_latency_samples,
                      const std::vector<double> &all_l1_latency_samples,
                      const std::vector<double> &all_l2_latency_samples,
                      const std::vector<double> &all_custom_latency_samples,
                      bool only_bandwidth,
                      bool only_latency) {
  // Don't print statistics if only one loop ran or if no enabled metric has data.
  if (loop_count <= 1) return;

  const auto has_any_population = [](std::initializer_list<const std::vector<double>*> populations) {
    return std::any_of(populations.begin(), populations.end(),
                       [](const std::vector<double>* values) {
                         return values != nullptr && !values->empty();
                       });
  };
  const bool has_main_bandwidth =
      has_any_population({&all_read_bw, &all_write_bw, &all_copy_bw});
  const bool has_cache_bandwidth = use_custom_cache_size
                                       ? has_any_population({&all_custom_read_bw,
                                                             &all_custom_write_bw,
                                                             &all_custom_copy_bw})
                                       : has_any_population({&all_l1_read_bw,
                                                             &all_l1_write_bw,
                                                             &all_l1_copy_bw,
                                                             &all_l2_read_bw,
                                                             &all_l2_write_bw,
                                                             &all_l2_copy_bw});
  const bool has_bandwidth = has_main_bandwidth || has_cache_bandwidth;
  const bool has_cache_latency = use_custom_cache_size
                                     ? !all_custom_latency.empty()
                                     : (!all_l1_latency.empty() || !all_l2_latency.empty());
  const bool has_latency = !all_main_mem_latency.empty() || has_cache_latency;

  if ((only_latency && !has_latency) ||
      (only_bandwidth && !has_bandwidth) ||
      (!only_latency && !only_bandwidth && !has_bandwidth && !has_latency)) {
    return;
  }

  size_t measured_loop_count = 0;
  auto include_measurement_count = [&measured_loop_count](const std::vector<double>& values) {
    measured_loop_count = std::max(measured_loop_count, values.size());
  };
  for (const std::vector<double>* values : {
           &all_read_bw, &all_write_bw, &all_copy_bw, &all_l1_latency,
           &all_l2_latency, &all_l1_read_bw, &all_l1_write_bw,
           &all_l1_copy_bw, &all_l2_read_bw, &all_l2_write_bw,
           &all_l2_copy_bw, &all_main_mem_latency, &all_tlb_hit_latency,
           &all_tlb_miss_latency, &all_page_walk_penalty,
           &all_custom_latency, &all_custom_read_bw, &all_custom_write_bw,
           &all_custom_copy_bw}) {
    include_measurement_count(*values);
  }

  // Print requested and realized loop counts separately after interruptions.
  std::cout << Messages::statistics_header(loop_count, measured_loop_count)
            << std::endl;
  std::cout << std::fixed;

  // Display Main Memory Bandwidth statistics (skip if only latency tests).
  if (!only_latency) {
    bool printed_main_bandwidth = false;
    const auto print_main_bandwidth = [&printed_main_bandwidth](
                                          const std::string& name,
                                          const std::vector<double>& values) {
      if (values.empty()) {
        return;
      }
      if (printed_main_bandwidth) {
        std::cout << "\n";
      }
      print_metric_statistics(name, calculate_descriptive_statistics(values));
      printed_main_bandwidth = true;
    };

    print_main_bandwidth("Read Bandwidth (GB/s)", all_read_bw);
    print_main_bandwidth("Write Bandwidth (GB/s)", all_write_bw);
    print_main_bandwidth("Copy Bandwidth (GB/s)", all_copy_bw);

    // Display Cache Bandwidth statistics.
    if (use_custom_cache_size) {
      print_cache_bandwidth_statistics("Custom", all_custom_read_bw, all_custom_write_bw, all_custom_copy_bw);
    } else {
      print_cache_bandwidth_statistics("L1", all_l1_read_bw, all_l1_write_bw, all_l1_copy_bw);
      print_cache_bandwidth_statistics("L2", all_l2_read_bw, all_l2_write_bw, all_l2_copy_bw);
    }
  }

  // Display Cache Latency statistics (skip if only bandwidth tests).
  if (!only_bandwidth) {
    const bool has_cache_latency_stats = use_custom_cache_size
                                             ? !all_custom_latency.empty()
                                             : (!all_l1_latency.empty() || !all_l2_latency.empty());
    if (has_cache_latency_stats) {
      std::cout << Messages::statistics_cache_latency_header() << std::endl;
      if (use_custom_cache_size) {
        print_cache_latency_statistics("Custom", all_custom_latency, all_custom_latency_samples);
      } else {
        print_cache_latency_statistics("L1", all_l1_latency, all_l1_latency_samples);
        print_cache_latency_statistics("L2", all_l2_latency, all_l2_latency_samples);
      }
    }

    // Display Main Memory Latency statistics.
    if (!all_main_mem_latency.empty()) {
      DescriptiveStatistics main_mem_latency_stats =
          calculate_descriptive_statistics(all_main_mem_latency);
      bool use_main_mem_samples = !all_main_mem_latency_samples.empty();
      DescriptiveStatistics main_mem_sample_stats{};
      if (use_main_mem_samples) {
        main_mem_sample_stats =
            calculate_descriptive_statistics(all_main_mem_latency_samples);
      }

      std::cout << Messages::statistics_main_memory_latency_header() << std::endl;
      print_statistics_summary(main_mem_latency_stats,
                               Constants::LATENCY_PRECISION, "", "",
                               "main-memory latency");
      if (use_main_mem_samples) {
        std::cout << Messages::statistics_pooled_sample_distribution(
                         all_main_mem_latency_samples.size())
                  << std::endl;
        print_statistics_summary(
            main_mem_sample_stats, Constants::LATENCY_PRECISION, "  ", "  ",
            "main-memory latency samples", false,
            all_main_mem_latency_samples.size());
      }

      if (!all_tlb_hit_latency.empty()) {
        std::cout << "\n";
        print_metric_statistics(Messages::statistics_tlb_hit_latency_metric_name(),
                                calculate_descriptive_statistics(
                                    all_tlb_hit_latency),
                                Constants::LATENCY_PRECISION);
      }
      if (!all_tlb_miss_latency.empty()) {
        std::cout << "\n";
        print_metric_statistics(Messages::statistics_tlb_miss_latency_metric_name(),
                                calculate_descriptive_statistics(
                                    all_tlb_miss_latency),
                                Constants::LATENCY_PRECISION);
      }
      if (!all_page_walk_penalty.empty()) {
        std::cout << "\n";
        print_metric_statistics(Messages::statistics_page_walk_penalty_metric_name(),
                                calculate_descriptive_statistics(
                                    all_page_walk_penalty),
                                Constants::LATENCY_PRECISION);
      }
    }
  }
  
  // Print a final separator after statistics.
  std::cout << Messages::statistics_footer() << std::endl;
}

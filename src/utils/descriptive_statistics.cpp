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

/**
 * @file descriptive_statistics.cpp
 * @brief Shared descriptive-statistics implementation.
 */

#include "utils/descriptive_statistics.h"

#include <algorithm>
#include <cmath>

namespace {

double linear_percentile(const std::vector<double>& sorted_values,
                         double percentile) {
  if (sorted_values.size() == 1) {
    return sorted_values.front();
  }

  const double index = percentile * (sorted_values.size() - 1);
  const size_t lower = static_cast<size_t>(index);
  const size_t upper = lower + 1;
  if (upper >= sorted_values.size()) {
    return sorted_values.back();
  }

  const double upper_weight = index - lower;
  return sorted_values[lower] * (1.0 - upper_weight) +
         sorted_values[upper] * upper_weight;
}

}  // namespace

DescriptiveStatistics calculate_descriptive_statistics(
    const std::vector<double>& values) {
  DescriptiveStatistics statistics;
  if (values.empty()) {
    return statistics;
  }

  statistics.sample_count = values.size();

  double sum = 0.0;
  for (double value : values) {
    sum += value;
  }
  statistics.average = sum / values.size();

  std::vector<double> sorted_values = values;
  std::sort(sorted_values.begin(), sorted_values.end());
  statistics.min = sorted_values.front();
  statistics.max = sorted_values.back();
  statistics.median = linear_percentile(sorted_values, 0.50);
  statistics.p90 = linear_percentile(sorted_values, 0.90);
  statistics.p95 = linear_percentile(sorted_values, 0.95);
  statistics.p99 = linear_percentile(sorted_values, 0.99);

  double squared_deviation_sum = 0.0;
  for (double value : values) {
    const double deviation = value - statistics.average;
    squared_deviation_sum += deviation * deviation;
  }
  if (values.size() > 1) {
    statistics.stddev =
        std::sqrt(squared_deviation_sum / (values.size() - 1));
  }

  if (std::isfinite(statistics.average) &&
      std::isfinite(statistics.stddev) && statistics.average != 0.0) {
    statistics.coefficient_of_variation_pct =
        std::abs(statistics.stddev / statistics.average) * 100.0;
    statistics.coefficient_of_variation_defined = true;
  }

  std::vector<double> absolute_deviations;
  absolute_deviations.reserve(values.size());
  for (double value : values) {
    absolute_deviations.push_back(std::abs(value - statistics.median));
  }
  std::sort(absolute_deviations.begin(), absolute_deviations.end());
  statistics.median_absolute_deviation =
      linear_percentile(absolute_deviations, 0.50);

  return statistics;
}

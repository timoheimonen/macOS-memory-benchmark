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
 * @file descriptive_statistics.h
 * @brief Shared descriptive-statistics calculations for measurement data.
 */

#ifndef DESCRIPTIVE_STATISTICS_H
#define DESCRIPTIVE_STATISTICS_H

#include <cstddef>
#include <vector>

struct DescriptiveStatistics {
  size_t sample_count = 0;
  double average = 0.0;
  double min = 0.0;
  double max = 0.0;
  double median = 0.0;
  double p90 = 0.0;
  double p95 = 0.0;
  double p99 = 0.0;
  double stddev = 0.0;
  double coefficient_of_variation_pct = 0.0;
  bool coefficient_of_variation_defined = false;
  double median_absolute_deviation = 0.0;
};

/**
 * @brief Calculate descriptive statistics for a measurement population.
 *
 * Percentiles use linear interpolation over the sorted values. Standard
 * deviation is the sample standard deviation (n - 1), and a single sample
 * therefore has a standard deviation of zero. The coefficient of variation
 * is defined only when both its inputs are finite and the average is nonzero.
 *
 * @param values Finite measurement values. Their input order is retained for
 *        the sum and variance accumulation order.
 * @return Calculated statistics, or a default-initialized structure when the
 *         population is empty.
 * @note Supplying non-finite measurements violates the function precondition;
 *       this cold-path helper does not filter or sanitize its input.
 */
DescriptiveStatistics calculate_descriptive_statistics(
    const std::vector<double>& values);

#endif  // DESCRIPTIVE_STATISTICS_H

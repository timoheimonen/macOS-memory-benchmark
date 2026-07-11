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
 * @file statistics_renderer.cpp
 * @brief Reusable console rendering for descriptive-statistics summaries
 */

#include "output/console/statistics_renderer.h"

#include "output/console/messages/messages_api.h"

#include <ostream>

void render_statistics_summary(
    std::ostream& output, const DescriptiveStatistics& statistics,
    const StatisticsSummaryRenderOptions& options) {
  output << options.line_prefix
         << Messages::statistics_average(statistics.average,
                                         options.precision)
         << '\n';
  if (options.median_from_samples) {
    // The sample-aware message includes the complete four-space indentation.
    output << Messages::statistics_median_p50_from_samples(
                  statistics.median, options.sample_count, options.precision)
           << '\n';
  } else {
    output << options.line_prefix
           << Messages::statistics_median_p50(statistics.median,
                                              options.precision)
           << '\n';
  }
  output << options.line_prefix
         << Messages::statistics_p90(statistics.p90, options.precision)
         << '\n';
  output << options.line_prefix
         << Messages::statistics_p95(statistics.p95, options.precision)
         << '\n';
  output << options.line_prefix
         << Messages::statistics_p99(statistics.p99, options.precision)
         << '\n';
  output << options.line_prefix
         << Messages::statistics_stddev(statistics.stddev, options.precision)
         << '\n';
  output << options.variability_prefix
         << Messages::statistics_coefficient_of_variation(
                statistics.coefficient_of_variation_pct,
                options.cv_precision)
         << '\n';
  output << options.variability_prefix
         << Messages::statistics_median_absolute_deviation(
                statistics.median_absolute_deviation, options.precision)
         << '\n';
  if (!options.inline_diagnostic.empty()) {
    output << options.variability_prefix << options.inline_diagnostic << '\n';
  }
  output << options.line_prefix
         << Messages::statistics_min(statistics.min, options.precision)
         << '\n';
  output << options.line_prefix
         << Messages::statistics_max(statistics.max, options.precision)
         << '\n';
}

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
 * @file test_statistics_renderer.cpp
 * @brief Unit tests for reusable statistics-summary console rendering
 */

#include <gtest/gtest.h>

#include "output/console/statistics_renderer.h"

#include <sstream>
#include <string>

namespace {

DescriptiveStatistics representative_statistics() {
  DescriptiveStatistics statistics;
  statistics.sample_count = 7;
  statistics.average = 1.25;
  statistics.min = -1.0;
  statistics.max = 9.0;
  statistics.median = 2.5;
  statistics.p90 = 7.0;
  statistics.p95 = 8.0;
  statistics.p99 = 8.8;
  statistics.stddev = 3.25;
  statistics.coefficient_of_variation_pct = 6.5;
  statistics.coefficient_of_variation_defined = true;
  statistics.median_absolute_deviation = 0.75;
  return statistics;
}

}  // namespace

TEST(StatisticsRendererTest, RendersCanonicalOrderPrecisionAndPrefixes) {
  StatisticsSummaryRenderOptions options;
  options.precision = 2;
  options.cv_precision = 1;
  options.line_prefix = ">";
  options.variability_prefix = "@";

  std::ostringstream output;
  render_statistics_summary(output, representative_statistics(), options);

  EXPECT_EQ(output.str(),
            ">  Average: 1.25\n"
            ">  Median (P50): 2.50\n"
            ">  P90: 7.00\n"
            ">  P95: 8.00\n"
            ">  P99: 8.80\n"
            ">  Stddev: 3.25\n"
            "@  CV:      6.5%\n"
            "@  Median absolute deviation: 0.75\n"
            ">  Min:     -1.00\n"
            ">  Max:     9.00\n");
}

TEST(StatisticsRendererTest, SampleMedianOwnsIndentationAndReportsCount) {
  StatisticsSummaryRenderOptions options;
  options.precision = 2;
  options.line_prefix = "  ";
  options.variability_prefix = "  ";
  options.median_from_samples = true;
  options.sample_count = 7;

  std::ostringstream output;
  render_statistics_summary(output, representative_statistics(), options);

  const std::string rendered = output.str();
  EXPECT_NE(rendered.find("    Average: 1.25\n"), std::string::npos);
  EXPECT_NE(rendered.find("    Median (P50): 2.50 (from 7 samples)\n"),
            std::string::npos);
  EXPECT_EQ(rendered.find("      Median (P50):"), std::string::npos);
}

TEST(StatisticsRendererTest, PlacesInlineDiagnosticBetweenVariabilityAndRange) {
  StatisticsSummaryRenderOptions options;
  options.precision = 2;
  options.variability_prefix = "  ";
  options.inline_diagnostic = "Warning: unstable measurement";

  std::ostringstream output;
  render_statistics_summary(output, representative_statistics(), options);

  const std::string rendered = output.str();
  const size_t mad_position = rendered.find("Median absolute deviation:");
  const size_t diagnostic_position =
      rendered.find("  Warning: unstable measurement\n");
  const size_t min_position = rendered.find("Min:");
  ASSERT_NE(mad_position, std::string::npos);
  ASSERT_NE(diagnostic_position, std::string::npos);
  ASSERT_NE(min_position, std::string::npos);
  EXPECT_LT(mad_position, diagnostic_position);
  EXPECT_LT(diagnostic_position, min_position);
}

TEST(StatisticsRendererTest, RendersUndefinedCoefficientAsStoredNumericZero) {
  DescriptiveStatistics statistics = representative_statistics();
  statistics.coefficient_of_variation_pct = 0.0;
  statistics.coefficient_of_variation_defined = false;

  std::ostringstream output;
  render_statistics_summary(output, statistics, {});

  EXPECT_NE(output.str().find("  CV:      0.0%\n"), std::string::npos);
}

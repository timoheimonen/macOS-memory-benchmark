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

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "utils/descriptive_statistics.h"

TEST(DescriptiveStatisticsTest, EmptyPopulationReturnsUndefinedDefaults) {
  const DescriptiveStatistics statistics =
      calculate_descriptive_statistics({});

  EXPECT_EQ(statistics.sample_count, 0U);
  EXPECT_DOUBLE_EQ(statistics.average, 0.0);
  EXPECT_DOUBLE_EQ(statistics.min, 0.0);
  EXPECT_DOUBLE_EQ(statistics.max, 0.0);
  EXPECT_DOUBLE_EQ(statistics.median, 0.0);
  EXPECT_DOUBLE_EQ(statistics.p90, 0.0);
  EXPECT_DOUBLE_EQ(statistics.p95, 0.0);
  EXPECT_DOUBLE_EQ(statistics.p99, 0.0);
  EXPECT_DOUBLE_EQ(statistics.stddev, 0.0);
  EXPECT_DOUBLE_EQ(statistics.coefficient_of_variation_pct, 0.0);
  EXPECT_FALSE(statistics.coefficient_of_variation_defined);
  EXPECT_DOUBLE_EQ(statistics.median_absolute_deviation, 0.0);
}

TEST(DescriptiveStatisticsTest, SingleNonzeroSampleHasDefinedZeroVariation) {
  const DescriptiveStatistics statistics =
      calculate_descriptive_statistics({42.0});

  EXPECT_EQ(statistics.sample_count, 1U);
  EXPECT_DOUBLE_EQ(statistics.average, 42.0);
  EXPECT_DOUBLE_EQ(statistics.min, 42.0);
  EXPECT_DOUBLE_EQ(statistics.max, 42.0);
  EXPECT_DOUBLE_EQ(statistics.median, 42.0);
  EXPECT_DOUBLE_EQ(statistics.p90, 42.0);
  EXPECT_DOUBLE_EQ(statistics.p95, 42.0);
  EXPECT_DOUBLE_EQ(statistics.p99, 42.0);
  EXPECT_DOUBLE_EQ(statistics.stddev, 0.0);
  EXPECT_TRUE(statistics.coefficient_of_variation_defined);
  EXPECT_DOUBLE_EQ(statistics.coefficient_of_variation_pct, 0.0);
  EXPECT_DOUBLE_EQ(statistics.median_absolute_deviation, 0.0);
}

TEST(DescriptiveStatisticsTest, SingleZeroHasUndefinedVariation) {
  const DescriptiveStatistics statistics =
      calculate_descriptive_statistics({0.0});

  EXPECT_EQ(statistics.sample_count, 1U);
  EXPECT_DOUBLE_EQ(statistics.stddev, 0.0);
  EXPECT_FALSE(statistics.coefficient_of_variation_defined);
  EXPECT_DOUBLE_EQ(statistics.coefficient_of_variation_pct, 0.0);
}

TEST(DescriptiveStatisticsTest,
     UsesLinearPercentilesSampleDeviationAndMedianAbsoluteDeviation) {
  const DescriptiveStatistics statistics =
      calculate_descriptive_statistics({10.0, 20.0, 30.0, 40.0});

  EXPECT_EQ(statistics.sample_count, 4U);
  EXPECT_DOUBLE_EQ(statistics.average, 25.0);
  EXPECT_DOUBLE_EQ(statistics.min, 10.0);
  EXPECT_DOUBLE_EQ(statistics.max, 40.0);
  EXPECT_DOUBLE_EQ(statistics.median, 25.0);
  EXPECT_DOUBLE_EQ(statistics.p90, 37.0);
  EXPECT_DOUBLE_EQ(statistics.p95, 38.5);
  EXPECT_NEAR(statistics.p99, 39.7, 1e-12);
  EXPECT_NEAR(statistics.stddev, 12.909944487358056, 1e-12);
  EXPECT_TRUE(statistics.coefficient_of_variation_defined);
  EXPECT_NEAR(statistics.coefficient_of_variation_pct, 51.63977794943222,
              1e-12);
  EXPECT_DOUBLE_EQ(statistics.median_absolute_deviation, 10.0);
}

TEST(DescriptiveStatisticsTest, CoefficientOfVariationUsesAbsoluteMean) {
  const DescriptiveStatistics statistics =
      calculate_descriptive_statistics({-10.0, -20.0, -30.0});

  EXPECT_DOUBLE_EQ(statistics.average, -20.0);
  EXPECT_DOUBLE_EQ(statistics.stddev, 10.0);
  EXPECT_TRUE(statistics.coefficient_of_variation_defined);
  EXPECT_DOUBLE_EQ(statistics.coefficient_of_variation_pct, 50.0);
}

TEST(DescriptiveStatisticsTest,
     UnsortedOddPopulationPreservesInputAndUsesOddMedianDeviation) {
  const std::vector<double> values = {5.0, 1.0, 4.0, 2.0, 3.0};
  const std::vector<double> original = values;

  const DescriptiveStatistics statistics =
      calculate_descriptive_statistics(values);

  EXPECT_EQ(values, original);
  EXPECT_DOUBLE_EQ(statistics.average, 3.0);
  EXPECT_DOUBLE_EQ(statistics.median, 3.0);
  EXPECT_DOUBLE_EQ(statistics.median_absolute_deviation, 1.0);
}

TEST(DescriptiveStatisticsTest, ConstantPopulationHasDefinedZeroVariation) {
  const DescriptiveStatistics statistics =
      calculate_descriptive_statistics({7.0, 7.0, 7.0});

  EXPECT_DOUBLE_EQ(statistics.stddev, 0.0);
  EXPECT_TRUE(statistics.coefficient_of_variation_defined);
  EXPECT_DOUBLE_EQ(statistics.coefficient_of_variation_pct, 0.0);
}

TEST(DescriptiveStatisticsTest, ZeroMeanHasUndefinedCoefficientOfVariation) {
  const DescriptiveStatistics statistics =
      calculate_descriptive_statistics({-1.0, 1.0});

  EXPECT_DOUBLE_EQ(statistics.average, 0.0);
  EXPECT_FALSE(statistics.coefficient_of_variation_defined);
  EXPECT_DOUBLE_EQ(statistics.coefficient_of_variation_pct, 0.0);
}

TEST(DescriptiveStatisticsTest,
     NonFiniteDerivedDeviationLeavesCoefficientOfVariationUndefined) {
  const double maximum = std::numeric_limits<double>::max();
  const DescriptiveStatistics statistics =
      calculate_descriptive_statistics({maximum, -maximum, maximum});

  EXPECT_TRUE(std::isfinite(statistics.average));
  EXPECT_TRUE(std::isinf(statistics.stddev));
  EXPECT_FALSE(statistics.coefficient_of_variation_defined);
  EXPECT_DOUBLE_EQ(statistics.coefficient_of_variation_pct, 0.0);
}

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
  const DescriptiveStatistics statistics = calculate_descriptive_statistics({});

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
  const DescriptiveStatistics statistics = calculate_descriptive_statistics({42.0});

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

TEST(DescriptiveStatisticsTest, UsesLinearPercentilesSampleDeviationAndMedianAbsoluteDeviation) {
  const DescriptiveStatistics statistics = calculate_descriptive_statistics({10.0, 20.0, 30.0, 40.0});

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
  EXPECT_NEAR(statistics.coefficient_of_variation_pct, 51.63977794943222, 1e-12);
  EXPECT_DOUBLE_EQ(statistics.median_absolute_deviation, 10.0);
}

TEST(DescriptiveStatisticsTest, CoefficientOfVariationUsesAbsoluteMean) {
  const DescriptiveStatistics statistics = calculate_descriptive_statistics({-10.0, -20.0, -30.0});

  EXPECT_DOUBLE_EQ(statistics.average, -20.0);
  EXPECT_DOUBLE_EQ(statistics.stddev, 10.0);
  EXPECT_TRUE(statistics.coefficient_of_variation_defined);
  EXPECT_DOUBLE_EQ(statistics.coefficient_of_variation_pct, 50.0);
}

TEST(DescriptiveStatisticsTest, UnsortedOddPopulationUsesOddMedianDeviation) {
  const std::vector<double> values = {5.0, 1.0, 4.0, 2.0, 3.0};

  const DescriptiveStatistics statistics = calculate_descriptive_statistics(values);

  EXPECT_DOUBLE_EQ(statistics.average, 3.0);
  EXPECT_DOUBLE_EQ(statistics.median, 3.0);
  EXPECT_DOUBLE_EQ(statistics.median_absolute_deviation, 1.0);
}

TEST(DescriptiveStatisticsTest, CoefficientOfVariationZeroAndConstantEdgeCases) {
  struct VariationCase {
    const char* name;
    std::vector<double> values;
    double expected_average;
    bool expects_zero_deviation;
    bool expected_defined;
  };
  const std::vector<VariationCase> cases = {
      {"single zero", {0.0}, 0.0, true, false},
      {"constant nonzero population", {7.0, 7.0, 7.0}, 7.0, true, true},
      {"zero mean population", {-1.0, 1.0}, 0.0, false, false},
  };

  for (const VariationCase& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    const DescriptiveStatistics statistics = calculate_descriptive_statistics(test_case.values);

    EXPECT_EQ(statistics.sample_count, test_case.values.size());
    EXPECT_DOUBLE_EQ(statistics.average, test_case.expected_average);
    if (test_case.expects_zero_deviation) {
      EXPECT_DOUBLE_EQ(statistics.stddev, 0.0);
    }
    EXPECT_EQ(statistics.coefficient_of_variation_defined, test_case.expected_defined);
    EXPECT_DOUBLE_EQ(statistics.coefficient_of_variation_pct, 0.0);
  }
}

TEST(DescriptiveStatisticsTest, NonFiniteDerivedDeviationLeavesCoefficientOfVariationUndefined) {
  const double maximum = std::numeric_limits<double>::max();
  const DescriptiveStatistics statistics = calculate_descriptive_statistics({maximum, -maximum, maximum});

  EXPECT_TRUE(std::isfinite(statistics.average));
  EXPECT_TRUE(std::isinf(statistics.stddev));
  EXPECT_FALSE(statistics.coefficient_of_variation_defined);
  EXPECT_DOUBLE_EQ(statistics.coefficient_of_variation_pct, 0.0);
}

TEST(DescriptiveStatisticsTest, ReusableWorkspaceMatchesLegacyPathWithoutCapacityGrowth) {
  const std::vector<double> values = {5.0, 1.0, 4.0, 2.0, 3.0};
  const DescriptiveStatistics expected = calculate_descriptive_statistics(values);
  std::vector<double> sorted_workspace;
  std::vector<double> deviation_workspace;
  sorted_workspace.reserve(values.size());
  deviation_workspace.reserve(values.size());
  const size_t sorted_capacity = sorted_workspace.capacity();
  const size_t deviation_capacity = deviation_workspace.capacity();
  const double* sorted_data = sorted_workspace.data();
  const double* deviation_data = deviation_workspace.data();

  const DescriptiveStatistics actual = calculate_descriptive_statistics(values, sorted_workspace, deviation_workspace);

  EXPECT_EQ(actual.sample_count, expected.sample_count);
  EXPECT_DOUBLE_EQ(actual.average, expected.average);
  EXPECT_DOUBLE_EQ(actual.min, expected.min);
  EXPECT_DOUBLE_EQ(actual.max, expected.max);
  EXPECT_DOUBLE_EQ(actual.median, expected.median);
  EXPECT_DOUBLE_EQ(actual.p90, expected.p90);
  EXPECT_DOUBLE_EQ(actual.p95, expected.p95);
  EXPECT_DOUBLE_EQ(actual.p99, expected.p99);
  EXPECT_DOUBLE_EQ(actual.stddev, expected.stddev);
  EXPECT_DOUBLE_EQ(actual.coefficient_of_variation_pct, expected.coefficient_of_variation_pct);
  EXPECT_EQ(actual.coefficient_of_variation_defined, expected.coefficient_of_variation_defined);
  EXPECT_DOUBLE_EQ(actual.median_absolute_deviation, expected.median_absolute_deviation);
  EXPECT_EQ(sorted_workspace.capacity(), sorted_capacity);
  EXPECT_EQ(deviation_workspace.capacity(), deviation_capacity);
  EXPECT_EQ(sorted_workspace.data(), sorted_data);
  EXPECT_EQ(deviation_workspace.data(), deviation_data);

  const DescriptiveStatistics empty = calculate_descriptive_statistics({}, sorted_workspace, deviation_workspace);
  EXPECT_EQ(empty.sample_count, 0u);
  EXPECT_TRUE(sorted_workspace.empty());
  EXPECT_TRUE(deviation_workspace.empty());
  EXPECT_EQ(sorted_workspace.capacity(), sorted_capacity);
  EXPECT_EQ(deviation_workspace.capacity(), deviation_capacity);
  EXPECT_EQ(sorted_workspace.data(), sorted_data);
  EXPECT_EQ(deviation_workspace.data(), deviation_data);
}

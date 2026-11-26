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

TEST(DescriptiveStatisticsTest, ComputesPercentilesDeviationAndMadAcrossPopulations) {
  struct PopulationCase {
    const char* name;
    std::vector<double> values;
    double average;
    double min;
    double max;
    double median;
    double p90;
    double p95;
    double p99;
    double stddev;
    double cv_pct;
    double mad;
    bool check_full_percentiles;
  };
  const std::vector<PopulationCase> cases = {
      {"single nonzero sample", {42.0}, 42.0, 42.0, 42.0, 42.0, 42.0, 42.0, 42.0, 0.0, 0.0, 0.0, true},
      {"even linear population", {10.0, 20.0, 30.0, 40.0}, 25.0, 10.0, 40.0, 25.0, 37.0, 38.5, 39.7,
       12.909944487358056, 51.63977794943222, 10.0, true},
      {"unsorted odd population", {5.0, 1.0, 4.0, 2.0, 3.0}, 3.0, 1.0, 5.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, false},
  };

  for (const PopulationCase& entry : cases) {
    SCOPED_TRACE(entry.name);
    const DescriptiveStatistics statistics = calculate_descriptive_statistics(entry.values);

    EXPECT_EQ(statistics.sample_count, entry.values.size());
    EXPECT_DOUBLE_EQ(statistics.average, entry.average);
    EXPECT_DOUBLE_EQ(statistics.min, entry.min);
    EXPECT_DOUBLE_EQ(statistics.max, entry.max);
    EXPECT_DOUBLE_EQ(statistics.median, entry.median);
    EXPECT_DOUBLE_EQ(statistics.median_absolute_deviation, entry.mad);
    if (entry.check_full_percentiles) {
      EXPECT_DOUBLE_EQ(statistics.p90, entry.p90);
      EXPECT_DOUBLE_EQ(statistics.p95, entry.p95);
      EXPECT_NEAR(statistics.p99, entry.p99, 1e-12);
      EXPECT_NEAR(statistics.stddev, entry.stddev, 1e-12);
      EXPECT_TRUE(statistics.coefficient_of_variation_defined);
      EXPECT_NEAR(statistics.coefficient_of_variation_pct, entry.cv_pct, 1e-12);
    }
  }
}

TEST(DescriptiveStatisticsTest, CoefficientOfVariationHandlesSignedZeroConstantAndNonFiniteCases) {
  const double maximum = std::numeric_limits<double>::max();
  struct VariationCase {
    const char* name;
    std::vector<double> values;
    bool check_exact_average;
    double expected_average;
    bool check_exact_stddev;
    double expected_stddev;
    bool expect_infinite_stddev;
    bool expected_defined;
    double expected_cv_pct;
  };
  const std::vector<VariationCase> cases = {
      {"negative mean uses absolute mean", {-10.0, -20.0, -30.0}, true, -20.0, true, 10.0, false, true, 50.0},
      {"single zero", {0.0}, true, 0.0, true, 0.0, false, false, 0.0},
      {"constant nonzero population", {7.0, 7.0, 7.0}, true, 7.0, true, 0.0, false, true, 0.0},
      {"zero mean population", {-1.0, 1.0}, true, 0.0, false, 0.0, false, false, 0.0},
      {"non-finite derived deviation", {maximum, -maximum, maximum}, false, 0.0, false, 0.0, true, false, 0.0},
  };

  for (const VariationCase& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    const DescriptiveStatistics statistics = calculate_descriptive_statistics(test_case.values);

    EXPECT_EQ(statistics.sample_count, test_case.values.size());
    EXPECT_TRUE(std::isfinite(statistics.average));
    if (test_case.check_exact_average) {
      EXPECT_DOUBLE_EQ(statistics.average, test_case.expected_average);
    }
    if (test_case.check_exact_stddev) {
      EXPECT_DOUBLE_EQ(statistics.stddev, test_case.expected_stddev);
    }
    if (test_case.expect_infinite_stddev) {
      EXPECT_TRUE(std::isinf(statistics.stddev));
    }
    EXPECT_EQ(statistics.coefficient_of_variation_defined, test_case.expected_defined);
    EXPECT_DOUBLE_EQ(statistics.coefficient_of_variation_pct, test_case.expected_cv_pct);
  }
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

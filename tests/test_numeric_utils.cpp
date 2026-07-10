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

#include <gtest/gtest.h>

#include <limits>

#include "core/config/constants.h"
#include "utils/numeric_utils.h"

TEST(NumericUtilsTest, CheckedAddHandlesBoundariesWithoutMutatingOnFailure) {
  const size_t maximum = std::numeric_limits<size_t>::max();
  size_t result = 99;

  ASSERT_TRUE(NumericUtils::checked_add(0, 0, result));
  EXPECT_EQ(result, 0u);
  ASSERT_TRUE(NumericUtils::checked_add(maximum - 1, 1, result));
  EXPECT_EQ(result, maximum);

  result = 99;
  EXPECT_FALSE(NumericUtils::checked_add(maximum, 1, result));
  EXPECT_EQ(result, 99u);

  size_t aliased = maximum - 1;
  ASSERT_TRUE(NumericUtils::checked_add(aliased, 1, aliased));
  EXPECT_EQ(aliased, maximum);
  EXPECT_FALSE(NumericUtils::checked_add(aliased, 1, aliased));
  EXPECT_EQ(aliased, maximum);
}

TEST(NumericUtilsTest,
     CheckedMultiplyHandlesBoundariesWithoutMutatingOnFailure) {
  const size_t maximum = std::numeric_limits<size_t>::max();
  size_t result = 99;

  ASSERT_TRUE(NumericUtils::checked_multiply(maximum, 0, result));
  EXPECT_EQ(result, 0u);
  ASSERT_TRUE(NumericUtils::checked_multiply(maximum, 1, result));
  EXPECT_EQ(result, maximum);
  ASSERT_TRUE(NumericUtils::checked_multiply(maximum / 2, 2, result));
  EXPECT_EQ(result, maximum - 1);

  result = 99;
  EXPECT_FALSE(
      NumericUtils::checked_multiply(maximum / 2 + 1, 2, result));
  EXPECT_EQ(result, 99u);

  size_t aliased = maximum / 2;
  ASSERT_TRUE(NumericUtils::checked_multiply(aliased, 2, aliased));
  EXPECT_EQ(aliased, maximum - 1);
}

TEST(NumericUtilsTest, CheckedRoundUpHandlesExactAndOverflowBoundaries) {
  const size_t maximum = std::numeric_limits<size_t>::max();
  const size_t largest_multiple = maximum - maximum % 64;
  size_t result = 99;

  ASSERT_TRUE(NumericUtils::checked_round_up(0, 64, result));
  EXPECT_EQ(result, 0u);
  ASSERT_TRUE(NumericUtils::checked_round_up(64, 64, result));
  EXPECT_EQ(result, 64u);
  ASSERT_TRUE(NumericUtils::checked_round_up(65, 64, result));
  EXPECT_EQ(result, 128u);
  ASSERT_TRUE(
      NumericUtils::checked_round_up(largest_multiple, 64, result));
  EXPECT_EQ(result, largest_multiple);

  result = 99;
  EXPECT_FALSE(
      NumericUtils::checked_round_up(largest_multiple + 1, 64, result));
  EXPECT_EQ(result, 99u);
  EXPECT_FALSE(NumericUtils::checked_round_up(1, 0, result));
  EXPECT_EQ(result, 99u);
  ASSERT_TRUE(NumericUtils::checked_round_up(maximum, 1, result));
  EXPECT_EQ(result, maximum);
}

TEST(NumericUtilsTest, SaturatingArithmeticUsesDocumentedSentinels) {
  const size_t maximum = std::numeric_limits<size_t>::max();
  const size_t largest_multiple = maximum - maximum % 64;

  EXPECT_EQ(NumericUtils::saturating_add(maximum, 1), maximum);
  EXPECT_EQ(NumericUtils::saturating_add(maximum - 1, 1), maximum);
  EXPECT_EQ(NumericUtils::saturating_multiply(maximum, 2), maximum);
  EXPECT_EQ(NumericUtils::saturating_multiply(maximum, 0), 0u);
  EXPECT_EQ(NumericUtils::saturating_round_up(largest_multiple + 1, 64),
            maximum);
  EXPECT_EQ(NumericUtils::saturating_round_up(64, 64), 64u);
  EXPECT_EQ(NumericUtils::saturating_round_up(64, 0), 0u);
}

TEST(NumericUtilsTest, MinimumPilotCountPreservesGoldenAndBoundaryCases) {
  const size_t maximum = std::numeric_limits<size_t>::max();

  EXPECT_EQ(NumericUtils::calculate_minimum_pilot_count(
                192, 8 * Constants::BYTES_PER_MB, 100000),
            43691u);
  EXPECT_EQ(NumericUtils::calculate_minimum_pilot_count(256, 1024, 10), 4u);
  EXPECT_EQ(NumericUtils::calculate_minimum_pilot_count(300, 1000, 10),
            4u);
  EXPECT_EQ(NumericUtils::calculate_minimum_pilot_count(1, 1000, 99), 99u);
  EXPECT_EQ(NumericUtils::calculate_minimum_pilot_count(10, 0, 99), 1u);
  EXPECT_EQ(NumericUtils::calculate_minimum_pilot_count(0, 1000, 99), 0u);
  EXPECT_EQ(NumericUtils::calculate_minimum_pilot_count(10, 1000, 0), 0u);
  EXPECT_EQ(
      NumericUtils::calculate_minimum_pilot_count(1, maximum, maximum),
      maximum);
  EXPECT_EQ(
      NumericUtils::calculate_minimum_pilot_count(maximum, maximum, maximum),
      1u);
}

TEST(NumericUtilsTest, DurationScalingPreservesAllModeGoldenCases) {
  EXPECT_EQ(NumericUtils::calculate_duration_scaled_count(
                0.010, 100, 0.150, 1, 10000),
            1500u);
  EXPECT_EQ(NumericUtils::calculate_duration_scaled_count(
                0.010, 100, 0.150, 16, 10000, 16),
            1504u);
  EXPECT_EQ(NumericUtils::calculate_duration_scaled_count(
                0.010, 100000, 0.250, 1000, 10000000),
            2500000u);
  EXPECT_EQ(NumericUtils::calculate_duration_scaled_count(
                1.0, 1000, 0.001, 2000, 10000000),
            2000u);
  EXPECT_EQ(NumericUtils::calculate_duration_scaled_count(
                0.001, 100000, 1.0, 1000, 500000),
            500000u);
}

TEST(NumericUtilsTest, DurationScalingRejectsEveryInvalidInputClass) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();

  EXPECT_EQ(NumericUtils::calculate_duration_scaled_count(
                0.0, 100, 0.150, 1, 10000),
            0u);
  EXPECT_EQ(NumericUtils::calculate_duration_scaled_count(
                -1.0, 100, 0.150, 1, 10000),
            0u);
  EXPECT_EQ(NumericUtils::calculate_duration_scaled_count(
                nan, 100, 0.150, 1, 10000),
            0u);
  EXPECT_EQ(NumericUtils::calculate_duration_scaled_count(
                infinity, 100, 0.150, 1, 10000),
            0u);
  EXPECT_EQ(NumericUtils::calculate_duration_scaled_count(
                0.010, 0, 0.150, 1, 10000),
            0u);
  EXPECT_EQ(NumericUtils::calculate_duration_scaled_count(
                0.010, 100, 0.0, 1, 10000),
            0u);
  EXPECT_EQ(NumericUtils::calculate_duration_scaled_count(
                0.010, 100, -1.0, 1, 10000),
            0u);
  EXPECT_EQ(NumericUtils::calculate_duration_scaled_count(
                0.010, 100, nan, 1, 10000),
            0u);
  EXPECT_EQ(NumericUtils::calculate_duration_scaled_count(
                0.010, 100, infinity, 1, 10000),
            0u);
  EXPECT_EQ(NumericUtils::calculate_duration_scaled_count(
                0.010, 100, 0.150, 0, 10000),
            0u);
  EXPECT_EQ(NumericUtils::calculate_duration_scaled_count(
                0.010, 100, 0.150, 2, 1),
            0u);
  EXPECT_EQ(NumericUtils::calculate_duration_scaled_count(
                0.010, 100, 0.150, 1, 10000, 0),
            0u);
}

TEST(NumericUtilsTest, DurationScalingPreservesQuantizationBoundaries) {
  const size_t maximum = std::numeric_limits<size_t>::max();

  EXPECT_EQ(NumericUtils::calculate_duration_scaled_count(
                1.0, 17, 1.0, 1, 20, 8),
            16u);
  EXPECT_EQ(NumericUtils::calculate_duration_scaled_count(
                1.0, 10, 1.0, 10, 10, 6),
            0u);
  EXPECT_EQ(NumericUtils::calculate_duration_scaled_count(
                1.0, maximum, 1.0, 1, maximum, 2),
            maximum - 1);
  EXPECT_EQ(NumericUtils::calculate_duration_scaled_count(
                1.0, 10, 1.0, 10, 10, 11),
            0u);
  EXPECT_EQ(NumericUtils::calculate_duration_scaled_count(
                1.0, 16, 1.0, 1, 32, 16),
            16u);
}

TEST(NumericUtilsTest, DurationScalingClampsHugeFiniteRatioBeforeCast) {
  EXPECT_EQ(NumericUtils::calculate_duration_scaled_count(
                std::numeric_limits<double>::denorm_min(),
                std::numeric_limits<size_t>::max(),
                std::numeric_limits<double>::max(), 1, 123),
            123u);
}

TEST(NumericUtilsTest, CalibrationPolicyConstantsRemainExactAliases) {
  EXPECT_DOUBLE_EQ(Constants::BANDWIDTH_CALIBRATION_TARGET_SECONDS, 0.150);
  EXPECT_DOUBLE_EQ(Constants::BANDWIDTH_CALIBRATION_MIN_SECONDS, 0.100);
  EXPECT_DOUBLE_EQ(Constants::BANDWIDTH_CALIBRATION_MAX_SECONDS, 0.250);
  EXPECT_EQ(Constants::BANDWIDTH_CALIBRATION_MAX_CORRECTIONS, 2u);
  EXPECT_EQ(Constants::BANDWIDTH_CALIBRATION_MIN_PILOT_BYTES,
            8 * Constants::BYTES_PER_MB);
  EXPECT_EQ(Constants::BANDWIDTH_CALIBRATION_MAX_PASSES, 1000000000u);

  EXPECT_EQ(Constants::BENCHMARK_CALIBRATION_TARGET_SECONDS,
            Constants::BANDWIDTH_CALIBRATION_TARGET_SECONDS);
  EXPECT_EQ(Constants::PATTERN_CALIBRATION_TARGET_SECONDS,
            Constants::BANDWIDTH_CALIBRATION_TARGET_SECONDS);
  EXPECT_EQ(Constants::BENCHMARK_CALIBRATION_MIN_SECONDS,
            Constants::BANDWIDTH_CALIBRATION_MIN_SECONDS);
  EXPECT_EQ(Constants::PATTERN_CALIBRATION_MIN_SECONDS,
            Constants::BANDWIDTH_CALIBRATION_MIN_SECONDS);
  EXPECT_EQ(Constants::BENCHMARK_CALIBRATION_MAX_SECONDS,
            Constants::BANDWIDTH_CALIBRATION_MAX_SECONDS);
  EXPECT_EQ(Constants::PATTERN_CALIBRATION_MAX_SECONDS,
            Constants::BANDWIDTH_CALIBRATION_MAX_SECONDS);
  EXPECT_EQ(Constants::BENCHMARK_CALIBRATION_MAX_CORRECTIONS,
            Constants::BANDWIDTH_CALIBRATION_MAX_CORRECTIONS);
  EXPECT_EQ(Constants::PATTERN_CALIBRATION_MAX_CORRECTIONS,
            Constants::BANDWIDTH_CALIBRATION_MAX_CORRECTIONS);
  EXPECT_EQ(Constants::BENCHMARK_CALIBRATION_MIN_PILOT_BYTES,
            Constants::BANDWIDTH_CALIBRATION_MIN_PILOT_BYTES);
  EXPECT_EQ(Constants::PATTERN_CALIBRATION_MIN_PILOT_BYTES,
            Constants::BANDWIDTH_CALIBRATION_MIN_PILOT_BYTES);
  EXPECT_EQ(Constants::BENCHMARK_CALIBRATION_MAX_PASSES,
            Constants::BANDWIDTH_CALIBRATION_MAX_PASSES);
  EXPECT_EQ(Constants::PATTERN_CALIBRATION_MAX_PASSES,
            Constants::BANDWIDTH_CALIBRATION_MAX_PASSES);
}

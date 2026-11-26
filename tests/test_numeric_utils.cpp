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

TEST(NumericUtilsTest, CheckedAndSaturatingArithmeticHandleBoundariesAndAliasing) {
  const size_t maximum = std::numeric_limits<size_t>::max();
  const size_t largest_multiple = maximum - maximum % 64;
  constexpr size_t kSentinel = 99;
  enum class Op { Add, Mul, RoundUp };
  const struct { const char* name; Op op; size_t a; size_t b; bool ok; size_t value; } cases[] = {
      {"add zero", Op::Add, 0, 0, true, 0u},
      {"add max", Op::Add, maximum - 1, 1, true, maximum},
      {"add overflow", Op::Add, maximum, 1, false, kSentinel},
      {"mul zero", Op::Mul, maximum, 0, true, 0u},
      {"mul one", Op::Mul, maximum, 1, true, maximum},
      {"mul half*2", Op::Mul, maximum / 2, 2, true, maximum - 1},
      {"mul overflow", Op::Mul, maximum / 2 + 1, 2, false, kSentinel},
      {"round zero", Op::RoundUp, 0, 64, true, 0u},
      {"round exact", Op::RoundUp, 64, 64, true, 64u},
      {"round next", Op::RoundUp, 65, 64, true, 128u},
      {"round largest", Op::RoundUp, largest_multiple, 64, true, largest_multiple},
      {"round overflow", Op::RoundUp, largest_multiple + 1, 64, false, kSentinel},
      {"round zero align", Op::RoundUp, 1, 0, false, kSentinel},
      {"round align one", Op::RoundUp, maximum, 1, true, maximum},
  };
  for (const auto& entry : cases) {
    SCOPED_TRACE(entry.name);
    size_t result = kSentinel;
    const bool ok = entry.op == Op::Add
                        ? NumericUtils::checked_add(entry.a, entry.b, result)
                        : (entry.op == Op::Mul ? NumericUtils::checked_multiply(entry.a, entry.b, result)
                                               : NumericUtils::checked_round_up(entry.a, entry.b, result));
    EXPECT_EQ(ok, entry.ok);
    EXPECT_EQ(result, entry.value);
  }

  size_t aliased = maximum - 1;
  ASSERT_TRUE(NumericUtils::checked_add(aliased, 1, aliased));
  EXPECT_EQ(aliased, maximum);
  EXPECT_FALSE(NumericUtils::checked_add(aliased, 1, aliased));
  EXPECT_EQ(aliased, maximum);
  aliased = maximum / 2;
  ASSERT_TRUE(NumericUtils::checked_multiply(aliased, 2, aliased));
  EXPECT_EQ(aliased, maximum - 1);

  EXPECT_EQ(NumericUtils::saturating_add(maximum, 1), maximum);
  EXPECT_EQ(NumericUtils::saturating_add(maximum - 1, 1), maximum);
  EXPECT_EQ(NumericUtils::saturating_multiply(maximum, 2), maximum);
  EXPECT_EQ(NumericUtils::saturating_multiply(maximum, 0), 0u);
  EXPECT_EQ(NumericUtils::saturating_round_up(largest_multiple + 1, 64), maximum);
  EXPECT_EQ(NumericUtils::saturating_round_up(64, 64), 64u);
  EXPECT_EQ(NumericUtils::saturating_round_up(64, 0), 0u);
}

TEST(NumericUtilsTest, MinimumPilotCountPreservesGoldenAndBoundaryCases) {
  const size_t maximum = std::numeric_limits<size_t>::max();
  const struct { const char* name; size_t unit; size_t min_bytes; size_t max_count; size_t expected; } cases[] = {
      {"192B/8MiB", 192, 8 * Constants::BYTES_PER_MB, 100000, 43691u},
      {"exact", 256, 1024, 10, 4u},
      {"ceil", 300, 1000, 10, 4u},
      {"clamped", 1, 1000, 99, 99u},
      {"zero min", 10, 0, 99, 1u},
      {"zero unit", 0, 1000, 99, 0u},
      {"zero max", 10, 1000, 0, 0u},
      {"max min", 1, maximum, maximum, maximum},
      {"max unit", maximum, maximum, maximum, 1u},
  };
  for (const auto& entry : cases) {
    SCOPED_TRACE(entry.name);
    EXPECT_EQ(NumericUtils::calculate_minimum_pilot_count(entry.unit, entry.min_bytes, entry.max_count),
              entry.expected);
  }
}

TEST(NumericUtilsTest, CalibrationPolicyDurationScalingQuantizationAndRatioClamp) {
  const size_t maximum = std::numeric_limits<size_t>::max();
  const struct {
    const char* name; double elapsed; size_t count; double target; size_t min_c; size_t max_c; size_t q; size_t out;
  } cases[] = {
      {"linear", 0.010, 100, 0.150, 1, 10000, 1, 1500u},
      {"quantized", 0.010, 100, 0.150, 16, 10000, 16, 1504u},
      {"large latency", 0.010, 100000, 0.250, 1000, 10000000, 1, 2500000u},
      {"clamp min", 1.0, 1000, 0.001, 2000, 10000000, 1, 2000u},
      {"clamp max", 0.001, 100000, 1.0, 1000, 500000, 1, 500000u},
      {"quantize down", 1.0, 17, 1.0, 1, 20, 8, 16u},
      {"no quantum in range", 1.0, 10, 1.0, 10, 10, 6, 0u},
      {"quantize near max", 1.0, maximum, 1.0, 1, maximum, 2, maximum - 1},
      {"quantum > max", 1.0, 10, 1.0, 10, 10, 11, 0u},
      {"exact quantum", 1.0, 16, 1.0, 1, 32, 16, 16u},
      {"huge finite ratio", std::numeric_limits<double>::denorm_min(), maximum,
       std::numeric_limits<double>::max(), 1, 123, 1, 123u},
  };
  for (const auto& entry : cases) {
    SCOPED_TRACE(entry.name);
    EXPECT_EQ(NumericUtils::calculate_duration_scaled_count(entry.elapsed, entry.count, entry.target, entry.min_c,
                                                            entry.max_c, entry.q),
              entry.out);
  }

  EXPECT_DOUBLE_EQ(Constants::BANDWIDTH_CALIBRATION_TARGET_SECONDS, 0.150);
  EXPECT_DOUBLE_EQ(Constants::BANDWIDTH_CALIBRATION_MIN_SECONDS, 0.100);
  EXPECT_DOUBLE_EQ(Constants::BANDWIDTH_CALIBRATION_MAX_SECONDS, 0.250);
  EXPECT_EQ(Constants::BANDWIDTH_CALIBRATION_MAX_CORRECTIONS, 2u);
  EXPECT_EQ(Constants::BANDWIDTH_CALIBRATION_MIN_PILOT_BYTES, 8 * Constants::BYTES_PER_MB);
  EXPECT_EQ(Constants::BANDWIDTH_CALIBRATION_MAX_PASSES, 1000000000u);
}

TEST(NumericUtilsTest, DurationScalingRejectsEveryInvalidInputClass) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  const struct {
    const char* name; double elapsed; size_t count; double target; size_t min_c; size_t max_c; size_t q;
  } cases[] = {
      {"zero elapsed", 0.0, 100, 0.150, 1, 10000, 1},      {"neg elapsed", -1.0, 100, 0.150, 1, 10000, 1},
      {"nan elapsed", nan, 100, 0.150, 1, 10000, 1},       {"inf elapsed", inf, 100, 0.150, 1, 10000, 1},
      {"zero count", 0.010, 0, 0.150, 1, 10000, 1},        {"zero target", 0.010, 100, 0.0, 1, 10000, 1},
      {"neg target", 0.010, 100, -1.0, 1, 10000, 1},       {"nan target", 0.010, 100, nan, 1, 10000, 1},
      {"inf target", 0.010, 100, inf, 1, 10000, 1},        {"zero min", 0.010, 100, 0.150, 0, 10000, 1},
      {"min > max", 0.010, 100, 0.150, 2, 1, 1},           {"zero quantum", 0.010, 100, 0.150, 1, 10000, 0},
  };
  for (const auto& entry : cases) {
    SCOPED_TRACE(entry.name);
    EXPECT_EQ(NumericUtils::calculate_duration_scaled_count(entry.elapsed, entry.count, entry.target, entry.min_c,
                                                            entry.max_c, entry.q),
              0u);
  }
}

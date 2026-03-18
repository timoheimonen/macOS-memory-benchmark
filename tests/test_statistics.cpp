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
 * @file test_statistics.cpp
 * @brief Unit tests for statistics calculation logic exercised via print_statistics()
 *
 * calculate_statistics() is file-scoped (static) in statistics.cpp so it
 * cannot be called directly.  These tests drive it via print_statistics(),
 * capture stdout, and parse the numeric values embedded in the message strings
 * to verify mathematical correctness.
 *
 * Actual output label format (from statistics_messages.cpp):
 *   "Average:"  "Median (P50):"  "P90:"  "P95:"  "P99:"  "Stddev:"  "Min:"  "Max:"
 */
#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include "test_statistics_helpers.h"
#include "output/console/statistics.h"

namespace {

const std::vector<double>& kE = test_statistics_helpers::empty_values();
using test_statistics_helpers::capture_bw;
using test_statistics_helpers::capture_lat;

// ---------------------------------------------------------------------------
// Parse the first floating-point number that follows the given label string.
// Returns quiet_NaN when the label is absent or no parseable number follows.
//
// Label strings must match the actual output exactly, e.g.:
//   "Average:"  "Median (P50):"  "P90:"  "P95:"  "P99:"
//   "Stddev:"   "Min:"           "Max:"
// ---------------------------------------------------------------------------
double after(const std::string& out, const std::string& label) {
  size_t p = out.find(label);
  if (p == std::string::npos) return std::numeric_limits<double>::quiet_NaN();
  p += label.size();
  while (p < out.size() && !(std::isdigit(static_cast<unsigned char>(out[p])) || out[p] == '-')) ++p;
  if (p >= out.size()) return std::numeric_limits<double>::quiet_NaN();
  return std::stod(out.substr(p));
}

}  // namespace

// ---------------------------------------------------------------------------
// Guard-condition tests
// ---------------------------------------------------------------------------

// loop_count == 1 must produce no output at all.
TEST(StatisticsTest, SingleLoopProducesNoOutput) {
  testing::internal::CaptureStdout();
  print_statistics(
      1, {100.0},
      kE, kE, kE, kE, kE, kE, kE, kE, kE, kE,
      kE, kE, kE, kE,
      false,
      kE, kE, kE, kE,
      kE, kE, kE, kE,
      false, false);
  EXPECT_TRUE(testing::internal::GetCapturedStdout().empty());
}

// loop_count == 0 must also produce no output.
TEST(StatisticsTest, ZeroLoopsProducesNoOutput) {
  testing::internal::CaptureStdout();
  print_statistics(
      0, {42.0},
      kE, kE, kE, kE, kE, kE, kE, kE, kE, kE,
      kE, kE, kE, kE,
      false,
      kE, kE, kE, kE,
      kE, kE, kE, kE,
      false, false);
  EXPECT_TRUE(testing::internal::GetCapturedStdout().empty());
}

// only_latency=true with empty latency vectors suppresses output even when
// loop_count > 1.
TEST(StatisticsTest, OnlyLatencyModeEmptyLatencyProducesNoOutput) {
  testing::internal::CaptureStdout();
  print_statistics(
      3, {10.0, 20.0, 30.0},
      kE, kE, kE, kE, kE, kE, kE, kE, kE, kE,
      kE, kE, kE, kE,
      false,
      kE, kE, kE, kE,
      kE, kE, kE, kE,
      false, true);
  EXPECT_TRUE(testing::internal::GetCapturedStdout().empty());
}

// ---------------------------------------------------------------------------
// Average (mean) correctness
// ---------------------------------------------------------------------------

// Single value: average equals that value.
TEST(StatisticsTest, SingleValueAverageEqualsValue) {
  EXPECT_NEAR(after(capture_bw({7.5}), "Average:"), 7.5, 1e-4);
}

// Two values: average is arithmetic mean.
TEST(StatisticsTest, TwoValuesAverageIsArithmeticMean) {
  // (3 + 7) / 2 = 5.0
  EXPECT_NEAR(after(capture_bw({3.0, 7.0}), "Average:"), 5.0, 1e-4);
}

// Five values: sum=15, mean=3.
TEST(StatisticsTest, FiveValuesAverageCorrect) {
  EXPECT_NEAR(after(capture_bw({1.0, 2.0, 3.0, 4.0, 5.0}), "Average:"), 3.0, 1e-4);
}

// ---------------------------------------------------------------------------
// Min / Max
// ---------------------------------------------------------------------------

TEST(StatisticsTest, MinCorrect) {
  EXPECT_NEAR(after(capture_bw({5.0, 1.0, 9.0, 3.0, 7.0}), "Min:"), 1.0, 1e-4);
}

TEST(StatisticsTest, MaxCorrect) {
  EXPECT_NEAR(after(capture_bw({5.0, 1.0, 9.0, 3.0, 7.0}), "Max:"), 9.0, 1e-4);
}

// ---------------------------------------------------------------------------
// Median (P50) with linear-interpolation formula
// ---------------------------------------------------------------------------

// Odd count: sorted {1,2,3,4,5}, index = 0.5*(5-1) = 2.0 → exact element = 3.
TEST(StatisticsTest, MedianOddCountIsMiddleElement) {
  EXPECT_NEAR(after(capture_bw({5.0, 1.0, 3.0, 4.0, 2.0}), "Median (P50):"), 3.0, 1e-4);
}

// Even count: sorted {1,2,3,4}, index = 0.5*3 = 1.5
// → lower=1 (val=2), upper=2 (val=3), weight=0.5 → 2.5
TEST(StatisticsTest, MedianEvenCountLinearInterpolation) {
  EXPECT_NEAR(after(capture_bw({4.0, 1.0, 3.0, 2.0}), "Median (P50):"), 2.5, 1e-4);
}

// ---------------------------------------------------------------------------
// Percentile interpolation
// ---------------------------------------------------------------------------

// 10 sorted values 1..10
// P90: index = 0.9*9 = 8.1 → lower=8 (val=9), upper=9 (val=10)
//      → 9*(1-0.1) + 10*0.1 = 8.1 + 1.0 = 9.1
TEST(StatisticsTest, P90LinearInterpolation) {
  std::string out = capture_bw({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0});
  EXPECT_NEAR(after(out, "P90:"), 9.1, 1e-3);
}

// P95: index = 0.95*9 = 8.55 → lower=8 (val=9), upper=9 (val=10)
//      → 9*0.45 + 10*0.55 = 4.05 + 5.5 = 9.55
TEST(StatisticsTest, P95LinearInterpolation) {
  std::string out = capture_bw({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0});
  EXPECT_NEAR(after(out, "P95:"), 9.55, 1e-3);
}

// P99: index = 0.99*9 = 8.91 → lower=8 (val=9), upper=9 (val=10)
//      → 9*(1-0.91) + 10*0.91 = 0.81 + 9.1 = 9.91
TEST(StatisticsTest, P99LinearInterpolation) {
  std::string out = capture_bw({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0});
  EXPECT_NEAR(after(out, "P99:"), 9.91, 1e-3);
}

// ---------------------------------------------------------------------------
// Standard deviation (Bessel-corrected sample stddev)
// ---------------------------------------------------------------------------

// Single value: stddev must be 0 (n==1 branch in calculate_statistics).
TEST(StatisticsTest, SingleValueStddevIsZero) {
  EXPECT_NEAR(after(capture_bw({7.5}), "Stddev:"), 0.0, 1e-9);
}

// Two identical values: stddev must be 0.
TEST(StatisticsTest, IdenticalValuesStddevIsZero) {
  EXPECT_NEAR(after(capture_bw({5.5, 5.5}), "Stddev:"), 0.0, 1e-9);
}

// Three values {2, 4, 6}: mean=4, deviations {-2, 0, +2}
// variance = (4+0+4)/(3-1) = 4.0, stddev = 2.0
TEST(StatisticsTest, StddevThreeKnownValues) {
  EXPECT_NEAR(after(capture_bw({2.0, 4.0, 6.0}), "Stddev:"), 2.0, 1e-4);
}

// ---------------------------------------------------------------------------
// Mode-flag filtering
// ---------------------------------------------------------------------------

// only_latency=true: bandwidth metric labels must be absent.
TEST(StatisticsTest, OnlyLatencyModeOmitsBandwidthSection) {
  std::string out = capture_lat({80.0, 100.0, 120.0});
  EXPECT_EQ(out.find("Read Bandwidth"), std::string::npos);
}

// only_latency=true: Average must appear (latency section present).
TEST(StatisticsTest, OnlyLatencyModeIncludesLatencySection) {
  std::string out = capture_lat({80.0, 100.0, 120.0});
  EXPECT_NE(out.find("Average:"), std::string::npos);
}

// only_bandwidth=true: "Main Memory Latency" section must be absent even when
// latency data is supplied.
TEST(StatisticsTest, OnlyBandwidthModeOmitsLatencySection) {
  testing::internal::CaptureStdout();
  std::vector<double> bw = {10.0, 20.0};
  std::vector<double> lat = {100.0, 200.0};
  print_statistics(
      2, bw, bw, bw,
      kE, kE, kE, kE, kE, kE, kE, kE,
      lat,
      kE, kE, kE,
      false,
      kE, kE, kE, kE,
      kE, kE, kE, kE,
      true, false);
  std::string out = testing::internal::GetCapturedStdout();
  EXPECT_EQ(out.find("Main Memory Latency"), std::string::npos);
  EXPECT_NE(out.find("Read Bandwidth"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Latency path correctness
// ---------------------------------------------------------------------------

// Mean of {80, 100, 120} = 100.0
TEST(StatisticsTest, LatencyAverageCorrect) {
  EXPECT_NEAR(after(capture_lat({80.0, 100.0, 120.0}), "Average:"), 100.0, 1e-2);
}

// Stddev of {80, 100, 120}: deviations {-20, 0, +20}
// variance = (400+0+400)/2 = 400, stddev = 20.0
TEST(StatisticsTest, LatencyStddevCorrect) {
  EXPECT_NEAR(after(capture_lat({80.0, 100.0, 120.0}), "Stddev:"), 20.0, 1e-2);
}

// Median of {80, 100, 120} (odd count, middle element) = 100.0
TEST(StatisticsTest, LatencyMedianCorrect) {
  EXPECT_NEAR(after(capture_lat({80.0, 100.0, 120.0}), "Median (P50):"), 100.0, 1e-2);
}

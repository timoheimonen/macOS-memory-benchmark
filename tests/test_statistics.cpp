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
 * @brief Unit tests for statistics output routing exercised via print_statistics()
 *
 * The shared calculator and renderer have focused numeric and formatting tests
 * in their respective test files. These tests retain only integration-level
 * routing and population-selection coverage.
 */
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "output/console/statistics.h"
#include "test_statistics_helpers.h"

namespace {

const std::vector<double>& kE = test_statistics_helpers::empty_values();
using test_statistics_helpers::capture_auto_tlb_breakdown;
using test_statistics_helpers::capture_main_bandwidth;

}  // namespace

// ---------------------------------------------------------------------------
// Guard-condition tests
// ---------------------------------------------------------------------------

// loop_count == 1 must produce no output at all.
TEST(StatisticsTest, SingleLoopProducesNoOutput) {
  testing::internal::CaptureStdout();
  print_statistics(1, {100.0}, kE, kE, kE, kE, kE, kE, kE, kE, kE, kE, kE, kE, kE, kE, false, kE, kE, kE, kE, kE, kE,
                   kE, kE, false, false);
  EXPECT_TRUE(testing::internal::GetCapturedStdout().empty());
}

// only_latency=true with empty latency vectors suppresses output even when
// loop_count > 1.
TEST(StatisticsTest, OnlyLatencyModeEmptyLatencyProducesNoOutput) {
  testing::internal::CaptureStdout();
  print_statistics(3, {10.0, 20.0, 30.0}, kE, kE, kE, kE, kE, kE, kE, kE, kE, kE, kE, kE, kE, kE, false, kE, kE, kE, kE,
                   kE, kE, kE, kE, false, true);
  EXPECT_TRUE(testing::internal::GetCapturedStdout().empty());
}

// ---------------------------------------------------------------------------
// Mode-flag filtering
// ---------------------------------------------------------------------------

// only_bandwidth=true: "Main Memory Latency" section must be absent even when
// latency data is supplied.
TEST(StatisticsTest, OnlyBandwidthModeOmitsLatencySection) {
  testing::internal::CaptureStdout();
  std::vector<double> bw = {10.0, 20.0};
  std::vector<double> lat = {100.0, 200.0};
  print_statistics(2, bw, bw, bw, kE, kE, kE, kE, kE, kE, kE, kE, lat, kE, kE, kE, false, kE, kE, kE, kE, kE, kE, kE,
                   kE, true, false);
  std::string out = testing::internal::GetCapturedStdout();
  EXPECT_EQ(out.find("Main Memory Latency"), std::string::npos);
  EXPECT_NE(out.find("Read Bandwidth"), std::string::npos);
}

TEST(StatisticsTest, OnlyLatencyModeOmitsPopulatedBandwidthAndIncludesLatency) {
  const std::vector<double> bandwidth = {10.0, 20.0};
  const std::vector<double> latency = {100.0, 200.0};
  testing::internal::CaptureStdout();
  print_statistics(2, bandwidth, bandwidth, bandwidth, kE, kE, kE, kE, kE, kE, kE, kE, latency, kE, kE, kE, false, kE,
                   kE, kE, kE, kE, kE, kE, kE, false, true);
  const std::string output = testing::internal::GetCapturedStdout();

  EXPECT_NE(output.find("Main Memory Latency (ns):"), std::string::npos);
  EXPECT_EQ(output.find("Read Bandwidth (GB/s):"), std::string::npos);
  EXPECT_EQ(output.find("Write Bandwidth (GB/s):"), std::string::npos);
  EXPECT_EQ(output.find("Copy Bandwidth (GB/s):"), std::string::npos);
}

TEST(StatisticsTest, PrintsOnlyAvailableMainBandwidthPopulations) {
  struct BandwidthPopulationCase {
    const char* name;
    std::vector<double> read;
    std::vector<double> write;
    std::vector<double> copy;
    bool expects_read;
    bool expects_write;
    bool expects_copy;
  };
  const std::vector<BandwidthPopulationCase> cases = {
      {"read only", {10.0, 12.0}, {}, {}, true, false, false},
      {"write only", {}, {20.0, 22.0}, {}, false, true, false},
      {"copy only", {}, {}, {30.0, 32.0}, false, false, true},
      {"read and copy", {10.0, 12.0}, {}, {30.0, 32.0}, true, false, true},
  };

  for (const BandwidthPopulationCase& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    const std::string output = capture_main_bandwidth(test_case.read, test_case.write, test_case.copy);

    EXPECT_EQ(output.find("Read Bandwidth (GB/s):") != std::string::npos, test_case.expects_read);
    EXPECT_EQ(output.find("Write Bandwidth (GB/s):") != std::string::npos, test_case.expects_write);
    EXPECT_EQ(output.find("Copy Bandwidth (GB/s):") != std::string::npos, test_case.expects_copy);
  }
}

TEST(StatisticsTest, LatencySamplesRemainSeparateFromLoopHeadlineStatistics) {
  testing::internal::CaptureStdout();
  print_statistics(2, kE, kE, kE, kE, kE, kE, kE, kE, kE, kE, kE, {10.0, 20.0}, kE, kE, kE, false, kE, kE, kE, kE,
                   {100.0, 200.0}, kE, kE, kE, false, true);
  const std::string output = testing::internal::GetCapturedStdout();

  EXPECT_NE(output.find("Median (P50): 15.00"), std::string::npos);
  EXPECT_NE(output.find("Pooled Separate Sample-Window Distribution (2 samples)"), std::string::npos);
  EXPECT_NE(output.find("Median (P50): 150.00 (from 2 samples)"), std::string::npos);
}

TEST(StatisticsTest, HeaderReportsRequestedAndMeasuredLoopCountsSeparately) {
  const std::string output = capture_auto_tlb_breakdown({10.0, 11.0}, {9.0, 10.0}, {20.0, 21.0}, {11.0, 11.0}, 5);
  EXPECT_NE(output.find("Statistics Across 2 Measured Loops (5 Requested)"), std::string::npos);
  EXPECT_NE(output.find("16 KiB Locality Latency (ns):"), std::string::npos);
  EXPECT_NE(output.find("Global-Random Latency (ns):"), std::string::npos);
  EXPECT_NE(output.find("Locality Latency Delta, Global - 16 KiB (ns):"), std::string::npos);
}

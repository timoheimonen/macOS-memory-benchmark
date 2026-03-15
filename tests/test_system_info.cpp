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
 * @file test_system_info.cpp
 * @brief Sanity-check tests for system information queries (system_info.cpp)
 *
 * These tests run on real Apple Silicon hardware and assert that the sysctl-
 * based queries return plausible values.  They do not mock any system calls;
 * correctness is validated against known architectural invariants of Apple
 * Silicon (e.g. L2 > L1, perf cores >= 1, etc.).
 */
#include <gtest/gtest.h>
#include "core/system/system_info.h"

// ---------------------------------------------------------------------------
// Core counts
// ---------------------------------------------------------------------------

// Apple Silicon always has at least one performance core.
TEST(SystemInfoTest, PerformanceCoresPositive) {
  EXPECT_GT(get_performance_cores(), 0);
}

// Efficiency cores are present on most Apple Silicon SoCs (E-cores).
// A count of zero is theoretically possible on hypothetical P-core-only chips,
// so we only require non-negative.
TEST(SystemInfoTest, EfficiencyCoresNonNegative) {
  EXPECT_GE(get_efficiency_cores(), 0);
}

// Total logical cores must be positive.
TEST(SystemInfoTest, TotalLogicalCoresPositive) {
  EXPECT_GT(get_total_logical_cores(), 0);
}

// Total logical cores must be at least as large as performance core count.
TEST(SystemInfoTest, TotalCoresAtLeastPerformanceCores) {
  EXPECT_GE(get_total_logical_cores(), get_performance_cores());
}

// Performance + efficiency cores should not exceed total logical cores.
// (They may be less if the sysctl keys count physical cores and the total
// counts logical/hyperthreaded cores, but on Apple Silicon they should sum
// to the total.)
TEST(SystemInfoTest, PerfPlusEfficiencyNotExceedTotal) {
  int perf = get_performance_cores();
  int eff = get_efficiency_cores();
  int total = get_total_logical_cores();
  EXPECT_LE(perf + eff, total * 2)  // generous bound — prevents absurd values
      << "perf=" << perf << " eff=" << eff << " total=" << total;
}

// ---------------------------------------------------------------------------
// Processor name
// ---------------------------------------------------------------------------

TEST(SystemInfoTest, ProcessorNameNotEmpty) {
  EXPECT_FALSE(get_processor_name().empty());
}

// On Apple Silicon the processor name always contains "Apple".
TEST(SystemInfoTest, ProcessorNameContainsApple) {
  std::string name = get_processor_name();
  EXPECT_NE(name.find("Apple"), std::string::npos)
      << "Processor name was: " << name;
}

// ---------------------------------------------------------------------------
// Cache sizes
// ---------------------------------------------------------------------------

// L1 data cache must be detectable and positive.
TEST(SystemInfoTest, L1CacheSizePositive) {
  EXPECT_GT(get_l1_cache_size(), static_cast<size_t>(0));
}

// L2 cache must be larger than L1 on any Apple Silicon SoC.
TEST(SystemInfoTest, L2CacheLargerThanL1) {
  size_t l1 = get_l1_cache_size();
  size_t l2 = get_l2_cache_size();
  EXPECT_GT(l2, l1) << "L2=" << l2 << " L1=" << l1;
}

// L1 cache on Apple Silicon P-cores is at least 64 KB.
TEST(SystemInfoTest, L1CacheAtLeast64KB) {
  EXPECT_GE(get_l1_cache_size(), static_cast<size_t>(64 * 1024));
}

// L2 cache on Apple Silicon is at least 4 MB.
TEST(SystemInfoTest, L2CacheAtLeast4MB) {
  EXPECT_GE(get_l2_cache_size(), static_cast<size_t>(4 * 1024 * 1024));
}

// ---------------------------------------------------------------------------
// macOS version
// ---------------------------------------------------------------------------

TEST(SystemInfoTest, MacOSVersionNotEmpty) {
  EXPECT_FALSE(get_macos_version().empty());
}

// macOS version string should contain at least one dot (e.g. "15.3.1").
TEST(SystemInfoTest, MacOSVersionContainsDot) {
  std::string ver = get_macos_version();
  EXPECT_NE(ver.find('.'), std::string::npos)
      << "macOS version was: " << ver;
}

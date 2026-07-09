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

#include <algorithm>

#include "benchmark/tlb_sweep_planner.h"
#include "core/config/constants.h"

TEST(TlbSweepPlannerTest, HighDensityBasePlanIsPageConsistent) {
  const size_t page_size = 16 * Constants::BYTES_PER_KB;
  const size_t stride = 256;

  const std::vector<TlbSweepPoint> points =
      build_tlb_base_sweep_plan(stride, page_size, TlbSweepDensity::High);

  ASSERT_EQ(points.size(), 29u);
  for (size_t i = 0; i < points.size(); ++i) {
    EXPECT_EQ(points[i].point_index, i);
    EXPECT_EQ(points[i].locality_bytes % page_size, 0u);
    EXPECT_EQ(points[i].requested_pages, points[i].locality_bytes / page_size);
    EXPECT_EQ(points[i].effective_pages, points[i].requested_pages);
    EXPECT_EQ(points[i].pointer_count, points[i].effective_pages);
    EXPECT_EQ(points[i].refinement_source, "base");
  }
}

TEST(TlbSweepPlannerTest, InvalidSizesProduceEmptyBasePlan) {
  EXPECT_TRUE(build_tlb_base_sweep_plan(0, 16384, TlbSweepDensity::Low).empty());
  EXPECT_TRUE(build_tlb_base_sweep_plan(64, 0, TlbSweepDensity::Low).empty());
  EXPECT_TRUE(build_tlb_base_sweep_plan(136, 16384, TlbSweepDensity::Low).empty());
  EXPECT_TRUE(build_tlb_base_sweep_plan(32768, 16384, TlbSweepDensity::Low).empty());
}

TEST(TlbSweepPlannerTest, RefinementPlanIsAlignedDeduplicatedAndTracksSources) {
  const size_t page_size = 16 * Constants::BYTES_PER_KB;
  const std::vector<size_t> localities = {
      page_size,
      2 * page_size,
      4 * page_size,
  };
  const std::vector<TlbRefinementTarget> targets = {
      {1, "private-cache"},
      {1, "l1"},
  };

  const std::vector<TlbSweepPoint> points = build_tlb_refinement_plan(
      localities, targets, 256, page_size, page_size, 4 * page_size);

  ASSERT_FALSE(points.empty());
  for (size_t i = 0; i < points.size(); ++i) {
    EXPECT_EQ(points[i].point_index, i);
    EXPECT_EQ(points[i].locality_bytes % page_size, 0u);
    EXPECT_GT(points[i].locality_bytes, page_size);
    EXPECT_LT(points[i].locality_bytes, 4 * page_size);
    EXPECT_NE(points[i].refinement_source.find("private-cache"), std::string::npos);
    EXPECT_NE(points[i].refinement_source.find("l1"), std::string::npos);
  }
  const std::vector<size_t> point_localities = tlb_point_localities(points);
  EXPECT_TRUE(std::is_sorted(point_localities.begin(), point_localities.end()));
  EXPECT_EQ(std::adjacent_find(point_localities.begin(), point_localities.end()),
            point_localities.end());
}

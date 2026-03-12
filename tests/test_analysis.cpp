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

#include <algorithm>
#include <vector>

#include "benchmark/tlb_analysis.h"
#include "core/config/constants.h"

TEST(AnalysisTest, DetectBoundaryFindsL1Transition) {
  const std::vector<size_t> localities = {
      64 * Constants::BYTES_PER_KB,
      128 * Constants::BYTES_PER_KB,
      256 * Constants::BYTES_PER_KB,
      512 * Constants::BYTES_PER_KB,
      1 * Constants::BYTES_PER_MB,
      2 * Constants::BYTES_PER_MB,
  };
  const std::vector<double> latencies_ns = {10.0, 10.2, 10.1, 10.3, 13.6, 13.8};

  const TlbBoundaryDetection boundary = detect_tlb_boundary(localities, latencies_ns, 0);
  EXPECT_TRUE(boundary.detected);
  EXPECT_EQ(boundary.boundary_index, 4u);
  EXPECT_EQ(boundary.boundary_locality_bytes, 1u * Constants::BYTES_PER_MB);
  EXPECT_EQ(boundary.confidence, "High");
  EXPECT_GE(boundary.step_ns, 2.0);
}

TEST(AnalysisTest, DetectBoundaryReturnsNotDetectedForFlatData) {
  const std::vector<size_t> localities = {
      64 * Constants::BYTES_PER_KB,
      128 * Constants::BYTES_PER_KB,
      256 * Constants::BYTES_PER_KB,
      512 * Constants::BYTES_PER_KB,
      1 * Constants::BYTES_PER_MB,
      2 * Constants::BYTES_PER_MB,
  };
  const std::vector<double> latencies_ns = {10.0, 10.3, 10.4, 10.6, 10.8, 10.9};

  const TlbBoundaryDetection boundary = detect_tlb_boundary(localities, latencies_ns, 0);
  EXPECT_FALSE(boundary.detected);
}

TEST(AnalysisTest, DetectBoundaryFindsSecondTransitionFromNewSegment) {
  const std::vector<size_t> localities = {
      64 * Constants::BYTES_PER_KB,
      128 * Constants::BYTES_PER_KB,
      256 * Constants::BYTES_PER_KB,
      512 * Constants::BYTES_PER_KB,
      1 * Constants::BYTES_PER_MB,
      2 * Constants::BYTES_PER_MB,
      4 * Constants::BYTES_PER_MB,
      8 * Constants::BYTES_PER_MB,
      16 * Constants::BYTES_PER_MB,
      32 * Constants::BYTES_PER_MB,
      64 * Constants::BYTES_PER_MB,
  };
  const std::vector<double> latencies_ns = {
      10.0, 10.1, 10.2, 10.1, 13.4, 13.5, 13.6, 13.7, 13.8, 14.0, 25.0,
  };

  const TlbBoundaryDetection l1_boundary = detect_tlb_boundary(localities, latencies_ns, 0);
  ASSERT_TRUE(l1_boundary.detected);
  EXPECT_EQ(l1_boundary.boundary_index, 4u);

  const TlbBoundaryDetection l2_boundary =
      detect_tlb_boundary(localities, latencies_ns, l1_boundary.boundary_index);
  EXPECT_TRUE(l2_boundary.detected);
  EXPECT_EQ(l2_boundary.boundary_index, 10u);
  EXPECT_EQ(l2_boundary.boundary_locality_bytes, 64u * Constants::BYTES_PER_MB);
}

TEST(AnalysisTest, DetectBoundaryWithGuardSkipsLikelyCacheTransition) {
  const std::vector<size_t> localities = {
      16 * Constants::BYTES_PER_KB,
      64 * Constants::BYTES_PER_KB,
      128 * Constants::BYTES_PER_KB,
      256 * Constants::BYTES_PER_KB,
      512 * Constants::BYTES_PER_KB,
      1 * Constants::BYTES_PER_MB,
      2 * Constants::BYTES_PER_MB,
      4 * Constants::BYTES_PER_MB,
  };
  const std::vector<double> latencies_ns = {10.0, 10.1, 12.4, 12.2, 12.3, 12.5, 12.7, 17.5};

  const size_t l1d_bytes = 128 * Constants::BYTES_PER_KB;
  const size_t page_bytes = 16 * Constants::BYTES_PER_KB;
  const size_t tlb_guard_bytes = std::max<size_t>(2 * l1d_bytes, 64 * page_bytes);

  const TlbBoundaryDetection boundary =
      detect_tlb_boundary(localities, latencies_ns, 0, tlb_guard_bytes);
  EXPECT_TRUE(boundary.detected);
  EXPECT_EQ(boundary.boundary_locality_bytes, 4u * Constants::BYTES_PER_MB);
}

TEST(AnalysisTest, InferTlbEntriesFromBoundaryAndPageSize) {
  const size_t entries = infer_tlb_entries(4 * Constants::BYTES_PER_MB, 16 * Constants::BYTES_PER_KB);
  EXPECT_EQ(entries, 256u);
}

TEST(AnalysisTest, ConfidenceClassification) {
  EXPECT_EQ(classify_tlb_confidence(5.0, 0.20, true), "High");
  EXPECT_EQ(classify_tlb_confidence(5.0, 0.20, false), "Medium");
  EXPECT_EQ(classify_tlb_confidence(2.1, 0.11, false), "Low");
}

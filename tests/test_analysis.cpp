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
#include <utility>
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
  // Only 1 future point available (index 5), so multi-point persistence
  // cannot reach majority (2 of 3) → confidence is Medium, not High.
  EXPECT_EQ(boundary.confidence, "Medium");
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

TEST(AnalysisTest, InferTlbEntriesRangeFromBoundaryWindow) {
  const std::vector<size_t> localities = {
      2 * Constants::BYTES_PER_MB,
      3 * Constants::BYTES_PER_MB,
      4 * Constants::BYTES_PER_MB,
  };
  const std::pair<size_t, size_t> range = infer_tlb_entries_range(
      localities,
      1,
      16 * Constants::BYTES_PER_KB);

  EXPECT_EQ(range.first, 128u);
  EXPECT_EQ(range.second, 192u);
}

TEST(AnalysisTest, InferTlbEntriesEstimateUsesBoundaryWindowMidpoint) {
  const std::vector<size_t> localities = {
      2 * Constants::BYTES_PER_MB,
      3 * Constants::BYTES_PER_MB,
      4 * Constants::BYTES_PER_MB,
  };
  const size_t estimate = infer_tlb_entries_estimate(
      localities,
      1,
      16 * Constants::BYTES_PER_KB);

  EXPECT_EQ(estimate, 160u);
}

TEST(AnalysisTest, DetectPrivateCacheKneeNearOneMegabyte) {
  const std::vector<size_t> localities = {
      256 * Constants::BYTES_PER_KB,
      512 * Constants::BYTES_PER_KB,
      768 * Constants::BYTES_PER_KB,
      1 * Constants::BYTES_PER_MB,
      2 * Constants::BYTES_PER_MB,
      4 * Constants::BYTES_PER_MB,
  };
  const std::vector<double> latencies_ns = {10.0, 10.1, 10.2, 14.5, 14.7, 15.0};

  const PrivateCacheKneeDetection knee = detect_private_cache_knee(localities, latencies_ns);
  EXPECT_TRUE(knee.detected);
  EXPECT_EQ(knee.boundary_locality_bytes, 1u * Constants::BYTES_PER_MB);
  EXPECT_TRUE(knee.strong_private_cache_candidate);
  EXPECT_FALSE(knee.early_cache_candidate);
  EXPECT_TRUE(knee.may_interfere_with_tlb);
}

TEST(AnalysisTest, DetectPrivateCacheKneeClassifiesEarlyCandidate) {
  const std::vector<size_t> localities = {
      256 * Constants::BYTES_PER_KB,
      512 * Constants::BYTES_PER_KB,
      768 * Constants::BYTES_PER_KB,
      1 * Constants::BYTES_PER_MB,
      2 * Constants::BYTES_PER_MB,
  };
  const std::vector<double> latencies_ns = {10.0, 12.8, 12.9, 13.0, 13.1};

  const PrivateCacheKneeDetection knee = detect_private_cache_knee(localities, latencies_ns);
  EXPECT_TRUE(knee.detected);
  EXPECT_EQ(knee.boundary_locality_bytes, 512u * Constants::BYTES_PER_KB);
  EXPECT_FALSE(knee.strong_private_cache_candidate);
  EXPECT_TRUE(knee.early_cache_candidate);
  EXPECT_FALSE(knee.may_interfere_with_tlb);
}

TEST(AnalysisTest, ConfidenceClassification) {
  EXPECT_EQ(classify_tlb_confidence(5.0, 0.20, true), "High");
  EXPECT_EQ(classify_tlb_confidence(5.0, 0.20, false), "Medium");
  EXPECT_EQ(classify_tlb_confidence(2.1, 0.11, false), "Low");
}

TEST(AnalysisTest, DetectBoundaryMultiPointPersistenceSurvivesNoiseDip) {
  // Step at 4MB, noise dip at 8MB, recovery at 16MB — should still be detected
  // with persistent_jump = true via majority rule (2 of 3 future points pass).
  const std::vector<size_t> localities = {
      1 * Constants::BYTES_PER_MB,   2 * Constants::BYTES_PER_MB,
      4 * Constants::BYTES_PER_MB,   8 * Constants::BYTES_PER_MB,
      16 * Constants::BYTES_PER_MB,  32 * Constants::BYTES_PER_MB,
      64 * Constants::BYTES_PER_MB,
  };
  const std::vector<double> latencies_ns = {
      10.0, 10.1, 20.0, 12.5, 23.0, 24.0, 25.0,
  };

  const TlbBoundaryDetection boundary = detect_tlb_boundary(localities, latencies_ns, 0);
  ASSERT_TRUE(boundary.detected);
  EXPECT_EQ(boundary.boundary_index, 2u);
  EXPECT_EQ(boundary.boundary_locality_bytes, 4u * Constants::BYTES_PER_MB);
  // 2 of 3 future points (16MB, 32MB) exceed threshold → persistent_jump = true.
  EXPECT_TRUE(boundary.persistent_jump);
  EXPECT_EQ(boundary.confidence, "High");
}

TEST(AnalysisTest, DetectBoundaryRejectsNoisyStepByIQR) {
  // P50 shows a step at 4MB, but the raw loop IQRs overlap — should be rejected.
  // Baseline rows use bimodal {5.0, 15.0} distribution: P50=10, Q3=15.
  // Candidate has wide {13..25} distribution: P50=20, Q1=13.
  // baseline avg Q3 (15.0) >= candidate Q1 (13.0) → overlap → reject.
  const std::vector<size_t> localities = {
      16 * Constants::BYTES_PER_KB,   64 * Constants::BYTES_PER_KB,
      128 * Constants::BYTES_PER_KB,  256 * Constants::BYTES_PER_KB,
      512 * Constants::BYTES_PER_KB,  1 * Constants::BYTES_PER_MB,
      2 * Constants::BYTES_PER_MB,    4 * Constants::BYTES_PER_MB,
  };
  const std::vector<double> latencies_ns = {
      10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 20.0,
  };

  std::vector<std::vector<double>> loop_latencies(8, std::vector<double>(30));
  // Baseline rows: bimodal {5.0, 15.0} → P50=10, Q1=5, Q3=15.
  for (size_t row = 0; row < 7; ++row) {
    for (size_t k = 0; k < 30; ++k) {
      loop_latencies[row][k] = (k < 15) ? 5.0 : 15.0;
    }
  }
  // Candidate row: wide variance centered at 20 → Q1=13, Q3=23.
  for (size_t k = 0; k < 30; ++k) {
    loop_latencies[7][k] = 13.0 + static_cast<double>(k % 7) * 2.0;
  }

  const TlbBoundaryDetection boundary =
      detect_tlb_boundary(localities, latencies_ns, 0, 0, &loop_latencies);
  EXPECT_FALSE(boundary.detected);
}

TEST(AnalysisTest, DetectBoundaryAcceptsClearStepWithIQR) {
  // Same bimodal baseline as the reject test, but candidate has tight variance
  // so IQRs don't overlap → boundary accepted.
  const std::vector<size_t> localities = {
      16 * Constants::BYTES_PER_KB,   64 * Constants::BYTES_PER_KB,
      128 * Constants::BYTES_PER_KB,  256 * Constants::BYTES_PER_KB,
      512 * Constants::BYTES_PER_KB,  1 * Constants::BYTES_PER_MB,
      2 * Constants::BYTES_PER_MB,    4 * Constants::BYTES_PER_MB,
  };
  const std::vector<double> latencies_ns = {
      10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 22.0,
  };

  std::vector<std::vector<double>> loop_latencies(8, std::vector<double>(30));
  // Baseline rows: bimodal {5.0, 15.0} → P50=10, Q1=5, Q3=15.
  for (size_t row = 0; row < 7; ++row) {
    for (size_t k = 0; k < 30; ++k) {
      loop_latencies[row][k] = (k < 15) ? 5.0 : 15.0;
    }
  }
  // Candidate row: tight variance centered at 22 → Q1=21, Q3=23.
  // baseline avg Q3 (15.0) < candidate Q1 (21.0) → no overlap → accepted.
  for (size_t k = 0; k < 30; ++k) {
    loop_latencies[7][k] = 22.0 + (static_cast<double>(k % 3) - 1.0) * 1.0;
  }

  const TlbBoundaryDetection boundary =
      detect_tlb_boundary(localities, latencies_ns, 0, 0, &loop_latencies);
  ASSERT_TRUE(boundary.detected);
  EXPECT_EQ(boundary.boundary_index, 7u);
  EXPECT_TRUE(boundary.persistent_jump);
}

TEST(AnalysisTest, DetectBoundaryLastPointStrongStepGetsMediumConfidence) {
  // Last sweep point with a strong step (>8 ns) should get Medium confidence
  // (not Low) even though there are no future points for persistence.
  const std::vector<size_t> localities = {
      64 * Constants::BYTES_PER_KB,
      128 * Constants::BYTES_PER_KB,
      256 * Constants::BYTES_PER_KB,
      512 * Constants::BYTES_PER_KB,
      1 * Constants::BYTES_PER_MB,
  };
  const std::vector<double> latencies_ns = {10.0, 10.1, 10.2, 10.3, 20.0};

  const TlbBoundaryDetection boundary = detect_tlb_boundary(localities, latencies_ns, 0);
  ASSERT_TRUE(boundary.detected);
  EXPECT_EQ(boundary.boundary_index, 4u);
  EXPECT_TRUE(boundary.persistent_jump);
  EXPECT_NE(boundary.confidence, "Low");
}

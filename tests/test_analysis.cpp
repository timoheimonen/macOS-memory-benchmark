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
#include <cmath>
#include <utility>
#include <vector>

#include "benchmark/tlb_analysis.h"
#include "benchmark/tlb_sweep_planner.h"
#include "core/config/constants.h"

namespace {

TlbRoundPointMatrix make_round_point_matrix(
    const std::vector<double>& point_levels,
    size_t round_count = 12) {
  TlbRoundPointMatrix matrix(
      round_count, std::vector<double>(point_levels.size(), 0.0));
  for (size_t round = 0; round < round_count; ++round) {
    const double common_drift = static_cast<double>(round) * 0.02;
    for (size_t point = 0; point < point_levels.size(); ++point) {
      const double bounded_noise =
          (static_cast<double>((round + point) % 3) - 1.0) * 0.01;
      matrix[round][point] = point_levels[point] + common_drift + bounded_noise;
    }
  }
  return matrix;
}

}  // namespace

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

TEST(AnalysisTest, RobustBoundaryAcceptsPersistentStepWithIndependentValidation) {
  const size_t page_size = 16 * Constants::BYTES_PER_KB;
  const std::vector<size_t> localities = {
      page_size, 2 * page_size, 4 * page_size,
      8 * page_size, 16 * page_size, 32 * page_size,
  };
  const TlbRoundPointMatrix discovery =
      make_round_point_matrix({0.10, 0.12, 2.10, 2.20, 2.25, 2.30});
  const TlbRoundPointMatrix validation =
      make_round_point_matrix({0.08, 0.10, 2.00, 2.15, 2.20, 2.25});

  const TlbBoundaryDetection boundary = detect_tlb_boundary_robust(
      localities, discovery, &validation, 0, 0, 123456);

  ASSERT_TRUE(boundary.detected);
  EXPECT_EQ(boundary.boundary_index, 2u);
  EXPECT_EQ(boundary.bracket_lower_bytes, 2u * page_size);
  EXPECT_EQ(boundary.bracket_upper_bytes, 4u * page_size);
  EXPECT_TRUE(boundary.discovery.passed);
  EXPECT_TRUE(boundary.validation.passed);
  EXPECT_EQ(boundary.discovery.persistence_points_passed, 2u);
  EXPECT_GT(boundary.discovery.effect_ci.lower_ns,
            boundary.discovery.noise_floor_ns);
}

TEST(AnalysisTest, RobustBoundaryRejectsSingleSpikeThatReturnsToBaseline) {
  const size_t page_size = 16 * Constants::BYTES_PER_KB;
  const std::vector<size_t> localities = {
      page_size, 2 * page_size, 4 * page_size,
      8 * page_size, 16 * page_size, 32 * page_size,
  };
  const TlbRoundPointMatrix discovery =
      make_round_point_matrix({0.10, 0.12, 5.00, 0.15, 0.14, 0.16});
  const TlbRoundPointMatrix validation = discovery;

  const TlbBoundaryDetection boundary = detect_tlb_boundary_robust(
      localities, discovery, &validation, 0, 0, 77);

  EXPECT_FALSE(boundary.detected);
  ASSERT_FALSE(boundary.candidates.empty());
  EXPECT_EQ(boundary.candidates.front().boundary_index, 2u);
  EXPECT_FALSE(boundary.candidates.front().discovery.passed);
  EXPECT_EQ(boundary.candidates.front().discovery.rejection_reason,
            "persistence-not-confirmed");
}

TEST(AnalysisTest, RobustBoundaryRequiresIndependentValidationEvidence) {
  const size_t page_size = 16 * Constants::BYTES_PER_KB;
  const std::vector<size_t> localities = {
      page_size, 2 * page_size, 4 * page_size,
      8 * page_size, 16 * page_size, 32 * page_size,
  };
  const TlbRoundPointMatrix discovery =
      make_round_point_matrix({0.10, 0.12, 2.10, 2.20, 2.25, 2.30});
  const TlbRoundPointMatrix flat_validation =
      make_round_point_matrix({0.10, 0.12, 0.13, 0.14, 0.15, 0.16});

  const TlbBoundaryDetection boundary = detect_tlb_boundary_robust(
      localities, discovery, &flat_validation, 0, 0, 99);

  EXPECT_FALSE(boundary.detected);
  ASSERT_FALSE(boundary.candidates.empty());
  EXPECT_TRUE(boundary.candidates.front().discovery.passed);
  EXPECT_FALSE(boundary.candidates.front().validation.passed);
  EXPECT_EQ(boundary.candidates.front().validation.rejection_reason,
            "effect-below-minimum");
}

TEST(AnalysisTest, RobustBoundaryBootstrapIsDeterministicForSameSeed) {
  const size_t page_size = 16 * Constants::BYTES_PER_KB;
  const std::vector<size_t> localities = {
      page_size, 2 * page_size, 4 * page_size,
      8 * page_size, 16 * page_size, 32 * page_size,
  };
  const TlbRoundPointMatrix discovery =
      make_round_point_matrix({0.10, 0.12, 2.10, 2.20, 2.25, 2.30});
  const TlbRoundPointMatrix validation =
      make_round_point_matrix({0.08, 0.10, 2.00, 2.15, 2.20, 2.25});

  const TlbBoundaryDetection first = detect_tlb_boundary_robust(
      localities, discovery, &validation, 0, 0, 4242);
  const TlbBoundaryDetection second = detect_tlb_boundary_robust(
      localities, discovery, &validation, 0, 0, 4242);

  ASSERT_TRUE(first.detected);
  ASSERT_TRUE(second.detected);
  EXPECT_EQ(first.confidence, second.confidence);
  EXPECT_DOUBLE_EQ(first.discovery.effect_ci.lower_ns,
                   second.discovery.effect_ci.lower_ns);
  EXPECT_DOUBLE_EQ(first.discovery.effect_ci.upper_ns,
                   second.discovery.effect_ci.upper_ns);
  EXPECT_DOUBLE_EQ(first.validation.effect_ci.lower_ns,
                   second.validation.effect_ci.lower_ns);
  EXPECT_DOUBLE_EQ(first.validation.effect_ci.upper_ns,
                   second.validation.effect_ci.upper_ns);
}

TEST(AnalysisTest, TranslationDeltaMatrixPreservesRoundAndPointCoordinates) {
  const std::vector<size_t> localities = {16384, 32768};
  TlbMeasurementRecord base;
  base.pass = TlbMeasurementPass::Base;
  base.locality_bytes = localities[1];
  base.round_index = 1;
  base.paired.available = true;
  base.paired.translation_delta_ns = 3.5;
  TlbMeasurementRecord validation = base;
  validation.pass = TlbMeasurementPass::Validation;
  validation.locality_bytes = localities[0];
  validation.round_index = 0;
  validation.paired.translation_delta_ns = 1.25;

  const TlbRoundPointMatrix matrix = build_tlb_translation_delta_matrix(
      localities, {base, validation}, {TlbMeasurementPass::Base});

  ASSERT_EQ(matrix.size(), 2u);
  ASSERT_EQ(matrix[0].size(), 2u);
  EXPECT_TRUE(std::isnan(matrix[0][0]));
  EXPECT_TRUE(std::isnan(matrix[0][1]));
  EXPECT_TRUE(std::isnan(matrix[1][0]));
  EXPECT_DOUBLE_EQ(matrix[1][1], 3.5);
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

TEST(AnalysisTest, RefinementPointsArePageAlignedAndInsideBracket) {
  const size_t page_size = 16 * Constants::BYTES_PER_KB;
  const std::vector<size_t> localities = {
      page_size,
      2 * page_size,
      4 * page_size,
  };

  const std::vector<size_t> points =
      build_tlb_refinement_points(localities, 1, page_size, 4 * page_size, page_size);

  ASSERT_FALSE(points.empty());
  EXPECT_TRUE(std::is_sorted(points.begin(), points.end()));
  EXPECT_EQ(std::adjacent_find(points.begin(), points.end()), points.end());
  for (size_t point : points) {
    EXPECT_EQ(point % page_size, 0u);
    EXPECT_GT(point, page_size);
    EXPECT_LT(point, 4 * page_size);
  }
}

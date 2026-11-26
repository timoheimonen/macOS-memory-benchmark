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
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "benchmark/tlb_analysis.h"
#include "core/config/config.h"
#include "core/config/constants.h"
#include "core/config/version.h"
#include "output/console/messages/messages_api.h"

namespace {

TlbRoundPointMatrix make_round_point_matrix(const std::vector<double>& point_levels, size_t round_count = 12) {
  TlbRoundPointMatrix matrix(round_count, std::vector<double>(point_levels.size(), 0.0));
  for (size_t round = 0; round < round_count; ++round) {
    const double common_drift = static_cast<double>(round) * 0.02;
    for (size_t point = 0; point < point_levels.size(); ++point) {
      const double bounded_noise = (static_cast<double>((round + point) % 3) - 1.0) * 0.01;
      matrix[round][point] = point_levels[point] + common_drift + bounded_noise;
    }
  }
  return matrix;
}

TlbMeasurementRecord make_paired_summary_record(TlbMeasurementPass pass, size_t locality_bytes,
                                                double spread_latency_ns, double packed_latency_ns,
                                                double translation_delta_ns, size_t node_count = 128) {
  TlbMeasurementRecord record;
  record.pass = pass;
  record.locality_bytes = locality_bytes;
  record.paired.available = true;
  record.paired.spread.latency_ns = spread_latency_ns;
  record.paired.packed.latency_ns = packed_latency_ns;
  record.paired.translation_delta_ns = translation_delta_ns;
  record.paired.spread.diagnostics.actual_pages = node_count;
  record.paired.spread.diagnostics.node_count = node_count;
  record.paired.spread.diagnostics.unique_cache_lines = node_count;
  record.paired.packed.diagnostics.actual_pages = 1;
  record.paired.packed.diagnostics.node_count = node_count;
  record.paired.packed.diagnostics.unique_cache_lines = node_count;
  return record;
}

TlbAnalysisExecutionSeam make_tlb_execution_seam() {
  TlbAnalysisExecutionSeam seam;
  seam.page_size_bytes = 16 * Constants::BYTES_PER_KB;
  seam.l1_cache_size_bytes = 128 * Constants::BYTES_PER_KB;
  seam.selected_buffer_mb = 256;
  seam.available_memory_mb = 4096;
  seam.cpu_name = "Injected Apple CPU";
  seam.elapsed_seconds = []() { return 1.0; };
  return seam;
}

TlbScheduleExecutionResult make_pass_result(TlbMeasurementPass pass, const std::vector<TlbSweepPoint>& points,
                                            TlbScheduleExecutionStatus status, size_t completed_point_count,
                                            size_t rounds_completed) {
  TlbScheduleExecutionResult result;
  result.status = status;
  result.rounds_completed = rounds_completed;
  result.converged = status == TlbScheduleExecutionStatus::Complete;
  completed_point_count = std::min(completed_point_count, points.size());
  for (size_t i = 0; i < completed_point_count; ++i) {
    TlbMeasurementRecord record;
    record.pass = pass;
    record.point_index = points[i].point_index;
    record.locality_bytes = points[i].locality_bytes;
    record.round_index = 0;
    record.order_index = i;
    record.latency_ns = 10.0 + static_cast<double>(i);
    result.records.push_back(record);
  }
  return result;
}

int run_tlb_analysis_collect_silently(
    const BenchmarkConfig& config,
    const TlbStopRequested& stop_requested,
    const TlbAnalysisExecutionSeam& seam,
    nlohmann::ordered_json& result_json) {
  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  const int result = run_tlb_analysis_collect(
      config, stop_requested, seam, result_json);
  (void)testing::internal::GetCapturedStderr();
  const std::string standard_output = testing::internal::GetCapturedStdout();
  EXPECT_EQ(standard_output.find(Messages::config_header(SOFTVERSION)),
            std::string::npos)
      << standard_output;
  EXPECT_EQ(standard_output.find(Messages::usage_header(SOFTVERSION)),
            std::string::npos)
      << standard_output;
  return result;
}

}  // namespace

TEST(AnalysisTest, CoordinatorCollectReportsStopCompletePartialInterruptedAndErrorEvidence) {
  struct CoordinatorCase {
    const char* name;
    bool stop_before_first;
    TlbScheduleExecutionStatus pass_status;
    size_t completed_points;
    size_t rounds_completed;
    int expected_exit_code;
    TlbAnalysisCoordinatorStatus expected_status;
    std::string expected_status_text;
    size_t expected_completed_passes;
    bool expected_conclusions_valid;
    size_t expected_executor_calls;
  };
  const std::vector<CoordinatorCase> cases = {
      {"stop before first task", true, TlbScheduleExecutionStatus::Interrupted, 0, 0, EXIT_SUCCESS,
       TlbAnalysisCoordinatorStatus::Interrupted, "interrupted", 0, false, 0},
      {"partial", false, TlbScheduleExecutionStatus::Complete, 4, 1, EXIT_SUCCESS,
       TlbAnalysisCoordinatorStatus::Partial, "partial", 1, false, 1},
      {"interrupted", false, TlbScheduleExecutionStatus::Interrupted, 4, 0, EXIT_SUCCESS,
       TlbAnalysisCoordinatorStatus::Interrupted, "interrupted", 0, false, 1},
      {"error", false, TlbScheduleExecutionStatus::Error, 5, 1, EXIT_FAILURE, TlbAnalysisCoordinatorStatus::Error,
       "error", 0, false, 1},
      {"complete", false, TlbScheduleExecutionStatus::Complete, 15, 7, EXIT_SUCCESS,
       TlbAnalysisCoordinatorStatus::Complete, "complete", 1, true, 1},
  };

  for (const CoordinatorCase& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    BenchmarkConfig config;
    config.tlb_sweep_density = TlbSweepDensity::Low;
    TlbAnalysisExecutionSeam seam = make_tlb_execution_seam();
    bool observed = false;
    TlbAnalysisCoordinatorSummary summary;
    std::vector<TlbMeasurementPass> executed_passes;
    seam.execute_pass = [&](TlbMeasurementPass pass, const std::vector<TlbSweepPoint>& points) {
      executed_passes.push_back(pass);
      return make_pass_result(pass, points, test_case.pass_status, test_case.completed_points,
                              test_case.rounds_completed);
    };
    seam.observe_summary = [&](const TlbAnalysisCoordinatorSummary& value) {
      observed = true;
      summary = value;
    };

    nlohmann::ordered_json result_json;
    EXPECT_EQ(run_tlb_analysis_collect_silently(
                  config, [stop = test_case.stop_before_first]() { return stop; }, seam, result_json),
              test_case.expected_exit_code);
    EXPECT_EQ(executed_passes.size(), test_case.expected_executor_calls);
    if (test_case.expected_executor_calls == 1u) {
      EXPECT_EQ(executed_passes, (std::vector<TlbMeasurementPass>{TlbMeasurementPass::Base}));
    }
    ASSERT_TRUE(observed);
    EXPECT_EQ(summary.status, test_case.expected_status);
    EXPECT_EQ(summary.status_text, test_case.expected_status_text);
    EXPECT_EQ(summary.planned_points, 15u);
    EXPECT_EQ(summary.completed_points, test_case.completed_points);
    EXPECT_EQ(summary.planned_passes, 1u);
    EXPECT_EQ(summary.completed_passes, test_case.expected_completed_passes);
    EXPECT_EQ(summary.measurement_record_count, test_case.completed_points);
    EXPECT_EQ(summary.conclusions_valid, test_case.expected_conclusions_valid);
    EXPECT_FALSE(summary.large_locality_planned);
    ASSERT_EQ(summary.pass_summaries.size(), 1u);
    EXPECT_EQ(summary.pass_summaries[0].pass, TlbMeasurementPass::Base);
    EXPECT_EQ(summary.pass_summaries[0].point_count, 15u);
    EXPECT_EQ(summary.pass_summaries[0].rounds_completed, test_case.rounds_completed);
    EXPECT_EQ(summary.pass_summaries[0].status, test_case.pass_status);
    if (test_case.stop_before_first) {
      EXPECT_FALSE(summary.pass_summaries[0].complete);
    } else if (test_case.expected_status == TlbAnalysisCoordinatorStatus::Complete) {
      EXPECT_EQ(summary.validation_planned_points, 0u);
      EXPECT_EQ(summary.validation_completed_points, 0u);
      EXPECT_DOUBLE_EQ(summary.elapsed_seconds, 1.0);
      EXPECT_FALSE(summary.large_locality_completed);
      EXPECT_TRUE(summary.pass_summaries[0].converged);
      EXPECT_TRUE(summary.pass_summaries[0].complete);
    }
    ASSERT_FALSE(result_json.empty());
    EXPECT_EQ(result_json["tlb_analysis"]["status"], test_case.expected_status_text);
    EXPECT_EQ(result_json["tlb_analysis"]["conclusions_valid"], test_case.expected_conclusions_valid);
    EXPECT_FALSE(result_json["tlb_analysis"].contains("status_reason"));
  }
}

TEST(AnalysisTest, PairedSummaryComputesMedianDeltaAndRejectsInvalidOrUnavailableRecords) {
  {
    SCOPED_TRACE("median of same-round deltas with pass filter");
    const size_t locality = 2 * Constants::BYTES_PER_MB;
    const std::vector<TlbMeasurementRecord> records = {
        make_paired_summary_record(TlbMeasurementPass::Base, locality, 10.0, 9.0, 1.0),
        make_paired_summary_record(TlbMeasurementPass::Base, locality, 100.0, 90.0, 10.0),
        make_paired_summary_record(TlbMeasurementPass::Base, locality, 101.0, 100.0, 1.0),
        make_paired_summary_record(TlbMeasurementPass::Validation, locality, 1000.0, 1.0, 999.0),
    };

    const TlbPairedPointSummary summary = summarize_tlb_paired_point(records, locality, {TlbMeasurementPass::Base});
    ASSERT_TRUE(summary.available);
    EXPECT_DOUBLE_EQ(summary.spread_p50_ns, 100.0);
    EXPECT_DOUBLE_EQ(summary.packed_p50_ns, 90.0);
    EXPECT_DOUBLE_EQ(summary.translation_delta_p50_ns, 1.0);
    EXPECT_NE(summary.translation_delta_p50_ns, summary.spread_p50_ns - summary.packed_p50_ns);
    EXPECT_EQ(summary.spread_actual_pages, 128u);
    EXPECT_EQ(summary.packed_actual_pages, 1u);
    EXPECT_EQ(summary.unique_cache_lines, 128u);
    EXPECT_EQ(summary.active_cache_line_footprint_bytes, 128u * Constants::CACHE_LINE_SIZE_BYTES);
    EXPECT_EQ(summary.node_count, 128u);
    EXPECT_FALSE(summary.short_cycle_diagnostic);
  }

  const size_t locality = Constants::BYTES_PER_MB;
  TlbMeasurementRecord short_cycle = make_paired_summary_record(TlbMeasurementPass::Base, locality, 10.0, 8.0, 2.0, 32);
  {
    SCOPED_TRACE("short cycle diagnostic");
    const TlbPairedPointSummary short_summary =
        summarize_tlb_paired_point({short_cycle}, locality, {TlbMeasurementPass::Base});
    ASSERT_TRUE(short_summary.available);
    EXPECT_TRUE(short_summary.short_cycle_diagnostic);
  }

  TlbMeasurementRecord mismatched_cache_lines = short_cycle;
  mismatched_cache_lines.paired.packed.diagnostics.unique_cache_lines = 31;
  TlbMeasurementRecord unavailable = make_paired_summary_record(TlbMeasurementPass::Base, locality, 10.0, 8.0, 2.0);
  unavailable.paired.available = false;

  struct RejectionCase {
    const char* name;
    std::vector<TlbMeasurementRecord> records;
  };
  const std::vector<RejectionCase> rejection_cases = {
      {"inconsistent diagnostics", {short_cycle, mismatched_cache_lines}},
      {"only unavailable record", {unavailable}},
      {"mixed available and unavailable",
       {make_paired_summary_record(TlbMeasurementPass::Base, locality, 10.0, 8.0, 2.0), unavailable}},
      {"no matching pass", {make_paired_summary_record(TlbMeasurementPass::Validation, locality, 10.0, 8.0, 2.0)}},
  };
  for (const RejectionCase& entry : rejection_cases) {
    SCOPED_TRACE(entry.name);
    EXPECT_FALSE(summarize_tlb_paired_point(entry.records, locality, {TlbMeasurementPass::Base}).available);
  }
}

TEST(AnalysisTest, DetectBoundaryHandlesTransitionsFlatDataChainedSegmentsAndGuard) {
  const std::vector<size_t> six_localities = {
      64 * Constants::BYTES_PER_KB,  128 * Constants::BYTES_PER_KB, 256 * Constants::BYTES_PER_KB,
      512 * Constants::BYTES_PER_KB, 1 * Constants::BYTES_PER_MB,   2 * Constants::BYTES_PER_MB,
  };
  const std::vector<size_t> eleven_localities = {
      64 * Constants::BYTES_PER_KB,  128 * Constants::BYTES_PER_KB, 256 * Constants::BYTES_PER_KB,
      512 * Constants::BYTES_PER_KB, 1 * Constants::BYTES_PER_MB,   2 * Constants::BYTES_PER_MB,
      4 * Constants::BYTES_PER_MB,   8 * Constants::BYTES_PER_MB,   16 * Constants::BYTES_PER_MB,
      32 * Constants::BYTES_PER_MB,  64 * Constants::BYTES_PER_MB,
  };
  const std::vector<size_t> guarded_localities = {
      16 * Constants::BYTES_PER_KB,  64 * Constants::BYTES_PER_KB,  128 * Constants::BYTES_PER_KB,
      256 * Constants::BYTES_PER_KB, 512 * Constants::BYTES_PER_KB, 1 * Constants::BYTES_PER_MB,
      2 * Constants::BYTES_PER_MB,   4 * Constants::BYTES_PER_MB,
  };
  const size_t l1d_bytes = 128 * Constants::BYTES_PER_KB;
  const size_t page_bytes = 16 * Constants::BYTES_PER_KB;
  const size_t tlb_guard_bytes = std::max<size_t>(2 * l1d_bytes, 64 * page_bytes);

  struct BoundaryCase {
    const char* name;
    std::vector<size_t> localities;
    std::vector<double> latencies_ns;
    size_t start_index;
    size_t guard_bytes;
    bool expected_detected;
    size_t expected_index;
    size_t expected_locality_bytes;
    std::string expected_confidence;
    double min_step_ns;
    bool check_chained_second;
    size_t expected_second_index;
    size_t expected_second_locality_bytes;
  };
  const std::vector<BoundaryCase> cases = {
      {"l1 transition", six_localities, {10.0, 10.2, 10.1, 10.3, 13.6, 13.8}, 0, 0, true, 4u,
       1u * Constants::BYTES_PER_MB, "Medium", 2.0, false, 0u, 0u},
      {"flat data", six_localities, {10.0, 10.3, 10.4, 10.6, 10.8, 10.9}, 0, 0, false, 0u, 0u, "", 0.0, false, 0u, 0u},
      {"second transition from new segment", eleven_localities,
       {10.0, 10.1, 10.2, 10.1, 13.4, 13.5, 13.6, 13.7, 13.8, 14.0, 25.0}, 0, 0, true, 4u, 1u * Constants::BYTES_PER_MB,
       "", 0.0, true, 10u, 64u * Constants::BYTES_PER_MB},
      {"guard skips likely cache transition", guarded_localities, {10.0, 10.1, 12.4, 12.2, 12.3, 12.5, 12.7, 17.5}, 0,
       tlb_guard_bytes, true, 7u, 4u * Constants::BYTES_PER_MB, "", 0.0, false, 0u, 0u},
  };

  for (const BoundaryCase& entry : cases) {
    SCOPED_TRACE(entry.name);
    const TlbBoundaryDetection boundary =
        detect_tlb_boundary(entry.localities, entry.latencies_ns, entry.start_index, entry.guard_bytes);
    EXPECT_EQ(boundary.detected, entry.expected_detected);
    if (!entry.expected_detected) continue;
    EXPECT_EQ(boundary.boundary_index, entry.expected_index);
    EXPECT_EQ(boundary.boundary_locality_bytes, entry.expected_locality_bytes);
    // Only 1 future point available for l1 transition (index 5), so multi-point
    // persistence cannot reach majority (2 of 3) -> confidence is Medium.
    if (!entry.expected_confidence.empty()) EXPECT_EQ(boundary.confidence, entry.expected_confidence);
    if (entry.min_step_ns > 0.0) EXPECT_GE(boundary.step_ns, entry.min_step_ns);
    if (entry.check_chained_second) {
      const TlbBoundaryDetection second =
          detect_tlb_boundary(entry.localities, entry.latencies_ns, boundary.boundary_index);
      EXPECT_TRUE(second.detected);
      EXPECT_EQ(second.boundary_index, entry.expected_second_index);
      EXPECT_EQ(second.boundary_locality_bytes, entry.expected_second_locality_bytes);
    }
  }
}

TEST(AnalysisTest, RobustBoundaryValidatesPersistentStepsAndRejectsSpikesOrFlatValidation) {
  const size_t page_size = 16 * Constants::BYTES_PER_KB;
  const std::vector<size_t> localities = {
      page_size, 2 * page_size, 4 * page_size, 8 * page_size, 16 * page_size, 32 * page_size,
  };
  struct RobustCase {
    const char* name;
    std::vector<double> discovery_levels;
    std::vector<double> validation_levels;
    uint64_t seed;
    bool expected_detected;
    bool expected_discovery_passed;
    bool expected_validation_passed;
    std::string expected_rejection_reason;
    bool check_validation_reason;
  };
  const std::vector<RobustCase> cases = {
      {"persistent step with independent validation",
       {0.10, 0.12, 2.10, 2.20, 2.25, 2.30},
       {0.08, 0.10, 2.00, 2.15, 2.20, 2.25},
       123456, true, true, true, "", false},
      {"single spike returns to baseline",
       {0.10, 0.12, 5.00, 0.15, 0.14, 0.16},
       {0.10, 0.12, 5.00, 0.15, 0.14, 0.16},
       77, false, false, false, "persistence-not-confirmed", false},
      {"flat validation rejects discovery step",
       {0.10, 0.12, 2.10, 2.20, 2.25, 2.30},
       {0.10, 0.12, 0.13, 0.14, 0.15, 0.16},
       99, false, true, false, "effect-below-minimum", true},
  };

  for (const RobustCase& entry : cases) {
    SCOPED_TRACE(entry.name);
    const TlbRoundPointMatrix discovery = make_round_point_matrix(entry.discovery_levels);
    const TlbRoundPointMatrix validation = make_round_point_matrix(entry.validation_levels);
    const TlbBoundaryDetection boundary =
        detect_tlb_boundary_robust(localities, discovery, &validation, 0, 0, entry.seed);

    EXPECT_EQ(boundary.detected, entry.expected_detected);
    if (entry.expected_detected) {
      EXPECT_EQ(boundary.boundary_index, 2u);
      EXPECT_EQ(boundary.bracket_lower_bytes, 2u * page_size);
      EXPECT_EQ(boundary.bracket_upper_bytes, 4u * page_size);
      EXPECT_TRUE(boundary.discovery.passed);
      EXPECT_TRUE(boundary.validation.passed);
      EXPECT_EQ(boundary.discovery.persistence_points_passed, 2u);
      EXPECT_GT(boundary.discovery.effect_ci.lower_ns, boundary.discovery.noise_floor_ns);

      const TlbBoundaryDetection second =
          detect_tlb_boundary_robust(localities, discovery, &validation, 0, 0, entry.seed);
      ASSERT_TRUE(second.detected);
      EXPECT_EQ(boundary.confidence, second.confidence);
      EXPECT_DOUBLE_EQ(boundary.discovery.effect_ci.lower_ns, second.discovery.effect_ci.lower_ns);
      EXPECT_DOUBLE_EQ(boundary.discovery.effect_ci.upper_ns, second.discovery.effect_ci.upper_ns);
      EXPECT_DOUBLE_EQ(boundary.validation.effect_ci.lower_ns, second.validation.effect_ci.lower_ns);
      EXPECT_DOUBLE_EQ(boundary.validation.effect_ci.upper_ns, second.validation.effect_ci.upper_ns);
    } else {
      ASSERT_FALSE(boundary.candidates.empty());
      EXPECT_EQ(boundary.candidates.front().boundary_index, 2u);
      EXPECT_EQ(boundary.candidates.front().discovery.passed, entry.expected_discovery_passed);
      if (entry.check_validation_reason) {
        EXPECT_EQ(boundary.candidates.front().validation.passed, entry.expected_validation_passed);
        EXPECT_EQ(boundary.candidates.front().validation.rejection_reason, entry.expected_rejection_reason);
      } else {
        EXPECT_EQ(boundary.candidates.front().discovery.rejection_reason, entry.expected_rejection_reason);
      }
    }
  }
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

  const TlbRoundPointMatrix matrix =
      build_tlb_translation_delta_matrix(localities, {base, validation}, {TlbMeasurementPass::Base});

  ASSERT_EQ(matrix.size(), 2u);
  ASSERT_EQ(matrix[0].size(), 2u);
  EXPECT_TRUE(std::isnan(matrix[0][0]));
  EXPECT_TRUE(std::isnan(matrix[0][1]));
  EXPECT_TRUE(std::isnan(matrix[1][0]));
  EXPECT_DOUBLE_EQ(matrix[1][1], 3.5);
}

TEST(AnalysisTest, DetectPrivateCacheKneeClassifiesStrongAndEarlyCandidates) {
  const std::vector<size_t> localities = {
      256 * Constants::BYTES_PER_KB, 512 * Constants::BYTES_PER_KB, 768 * Constants::BYTES_PER_KB,
      1 * Constants::BYTES_PER_MB,   2 * Constants::BYTES_PER_MB,   4 * Constants::BYTES_PER_MB,
  };
  const std::vector<double> latencies_ns = {10.0, 10.1, 10.2, 14.5, 14.7, 15.0};

  const PrivateCacheKneeDetection knee = detect_private_cache_knee(localities, latencies_ns);
  EXPECT_TRUE(knee.detected);
  EXPECT_EQ(knee.boundary_locality_bytes, 1u * Constants::BYTES_PER_MB);
  EXPECT_TRUE(knee.strong_private_cache_candidate);
  EXPECT_FALSE(knee.early_cache_candidate);
  EXPECT_TRUE(knee.may_interfere_with_tlb);

  const std::vector<size_t> early_localities = {
      256 * Constants::BYTES_PER_KB, 512 * Constants::BYTES_PER_KB, 768 * Constants::BYTES_PER_KB,
      1 * Constants::BYTES_PER_MB,   2 * Constants::BYTES_PER_MB,
  };
  const std::vector<double> early_latencies_ns = {10.0, 12.8, 12.9, 13.0, 13.1};

  const PrivateCacheKneeDetection early_knee = detect_private_cache_knee(early_localities, early_latencies_ns);
  EXPECT_TRUE(early_knee.detected);
  EXPECT_EQ(early_knee.boundary_locality_bytes, 512u * Constants::BYTES_PER_KB);
  EXPECT_FALSE(early_knee.strong_private_cache_candidate);
  EXPECT_TRUE(early_knee.early_cache_candidate);
  EXPECT_FALSE(early_knee.may_interfere_with_tlb);
}

TEST(AnalysisTest, PersistenceSurvivesNoiseDipAndConfidenceClassificationMatchesPolicy) {
  const struct {
    double step_ns;
    double relative_step;
    bool persistent;
    const char* expected;
  } confidence_cases[] = {
      {5.0, 0.20, true, "High"},
      {5.0, 0.20, false, "Medium"},
      {2.1, 0.11, false, "Low"},
  };
  for (const auto& entry : confidence_cases) {
    SCOPED_TRACE(entry.expected);
    EXPECT_EQ(classify_tlb_confidence(entry.step_ns, entry.relative_step, entry.persistent), entry.expected);
  }

  SCOPED_TRACE("multi-point persistence survives noise dip");
  // Step at 4MB, noise dip at 8MB, recovery at 16MB — should still be detected
  // with persistent_jump = true via majority rule (2 of 3 future points pass).
  const std::vector<size_t> localities = {
      1 * Constants::BYTES_PER_MB,  2 * Constants::BYTES_PER_MB,  4 * Constants::BYTES_PER_MB,
      8 * Constants::BYTES_PER_MB,  16 * Constants::BYTES_PER_MB, 32 * Constants::BYTES_PER_MB,
      64 * Constants::BYTES_PER_MB,
  };
  const std::vector<double> latencies_ns = {
      10.0, 10.1, 20.0, 12.0, 23.0, 24.0, 25.0,
  };

  const TlbBoundaryDetection boundary = detect_tlb_boundary(localities, latencies_ns, 0);
  ASSERT_TRUE(boundary.detected);
  EXPECT_EQ(boundary.boundary_index, 2u);
  EXPECT_EQ(boundary.boundary_locality_bytes, 4u * Constants::BYTES_PER_MB);
  // 2 of 3 future points (16MB, 32MB) exceed threshold -> persistent_jump = true.
  EXPECT_TRUE(boundary.persistent_jump);
  EXPECT_EQ(boundary.confidence, "High");
}

TEST(AnalysisTest, DetectBoundaryIqrDistinguishesNoisyAndClearSteps) {
  // P50 shows a step at 4MB, but the raw loop IQRs overlap — should be rejected.
  // Baseline rows use bimodal {5.0, 15.0} distribution: P50=10, Q3=15.
  // Candidate has wide {13..25} distribution: P50=20, Q1=13.
  // baseline avg Q3 (15.0) >= candidate Q1 (13.0) → overlap → reject.
  const std::vector<size_t> localities = {
      16 * Constants::BYTES_PER_KB,  64 * Constants::BYTES_PER_KB,  128 * Constants::BYTES_PER_KB,
      256 * Constants::BYTES_PER_KB, 512 * Constants::BYTES_PER_KB, 1 * Constants::BYTES_PER_MB,
      2 * Constants::BYTES_PER_MB,   4 * Constants::BYTES_PER_MB,
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

  const TlbBoundaryDetection boundary = detect_tlb_boundary(localities, latencies_ns, 0, 0, &loop_latencies);
  EXPECT_FALSE(boundary.detected);

  // Same bimodal baseline as the reject test, but candidate has tight variance
  // so IQRs don't overlap → boundary accepted.
  const std::vector<double> clear_latencies_ns = {
      10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 22.0,
  };

  std::vector<std::vector<double>> clear_loop_latencies = loop_latencies;
  // Candidate row: tight variance centered at 22 → Q1=21, Q3=23.
  // baseline avg Q3 (15.0) < candidate Q1 (21.0) → no overlap → accepted.
  for (size_t k = 0; k < 30; ++k) {
    clear_loop_latencies[7][k] = 22.0 + (static_cast<double>(k % 3) - 1.0) * 1.0;
  }

  const TlbBoundaryDetection clear_boundary =
      detect_tlb_boundary(localities, clear_latencies_ns, 0, 0, &clear_loop_latencies);
  ASSERT_TRUE(clear_boundary.detected);
  EXPECT_EQ(clear_boundary.boundary_index, 7u);
  EXPECT_TRUE(clear_boundary.persistent_jump);
}

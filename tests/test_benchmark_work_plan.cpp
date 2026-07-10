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

#include <cmath>
#include <limits>
#include <stdexcept>

#include "benchmark/benchmark_work_plan.h"
#include "core/config/constants.h"

TEST(BenchmarkWorkPlanTest, RejectsInvalidBandwidthParameters) {
  EXPECT_EQ(build_benchmark_bandwidth_work_plan(
                0, 1, 1, BenchmarkTarget::MainMemory,
                BenchmarkOperation::Read)
                .status,
            BenchmarkMeasurementStatus::Invalid);
  EXPECT_EQ(build_benchmark_bandwidth_work_plan(
                4096, 0, 1, BenchmarkTarget::MainMemory,
                BenchmarkOperation::Read)
                .status,
            BenchmarkMeasurementStatus::Invalid);
  EXPECT_EQ(build_benchmark_bandwidth_work_plan(
                4096, 1, 0, BenchmarkTarget::MainMemory,
                BenchmarkOperation::Read)
                .status,
            BenchmarkMeasurementStatus::Invalid);
}

TEST(BenchmarkWorkPlanTest, CoversUnevenBufferExactlyWithAlignedBoundaries) {
  const size_t buffer_size = (7 * Constants::BYTES_PER_MB) + 123;
  const BenchmarkWorkPlan plan = build_benchmark_bandwidth_work_plan(
      buffer_size, 10, 3, BenchmarkTarget::MainMemory,
      BenchmarkOperation::Read);

  ASSERT_EQ(plan.status, BenchmarkMeasurementStatus::Measured);
  ASSERT_EQ(plan.boundaries.size(), 11u);
  EXPECT_EQ(plan.boundaries.front(), 0u);
  EXPECT_EQ(plan.boundaries.back(), buffer_size);
  size_t covered = 0;
  for (size_t index = 0; index < plan.workers.size(); ++index) {
    EXPECT_EQ(plan.workers[index].offset_bytes, covered);
    covered += plan.workers[index].span_bytes;
    if (index + 1 < plan.workers.size()) {
      EXPECT_EQ(covered % Constants::CACHE_LINE_SIZE_BYTES, 0u);
    }
  }
  EXPECT_EQ(covered, buffer_size);
  EXPECT_EQ(plan.total_payload_bytes, buffer_size * 3);
}

TEST(BenchmarkWorkPlanTest, CopyAccountingIncludesReadAndWriteSides) {
  const BenchmarkWorkPlan plan = build_benchmark_bandwidth_work_plan(
      4096, 4, 5, BenchmarkTarget::L2, BenchmarkOperation::Copy);
  ASSERT_EQ(plan.status, BenchmarkMeasurementStatus::Measured);
  EXPECT_EQ(plan.payload_bytes_per_pass, 8192u);
  EXPECT_EQ(plan.total_payload_bytes, 40960u);
}

TEST(BenchmarkWorkPlanTest, ReducesWorkersForTinyBuffers) {
  const BenchmarkWorkPlan plan = build_benchmark_bandwidth_work_plan(
      64, 10, 1, BenchmarkTarget::L1, BenchmarkOperation::Write);
  ASSERT_EQ(plan.status, BenchmarkMeasurementStatus::Measured);
  EXPECT_EQ(plan.effective_threads, 1);
  ASSERT_EQ(plan.workers.size(), 1u);
  EXPECT_EQ(plan.workers[0].span_bytes, 64u);
}

TEST(BenchmarkWorkPlanTest, RejectsPayloadOverflow) {
  BenchmarkWorkPlan plan = build_benchmark_bandwidth_work_plan(
      std::numeric_limits<size_t>::max(), 1, 1,
      BenchmarkTarget::MainMemory, BenchmarkOperation::Copy);
  EXPECT_EQ(plan.status, BenchmarkMeasurementStatus::Invalid);
}

TEST(BenchmarkWorkPlanTest, LatencyPlanRoundsToCompleteCycles) {
  const BenchmarkLatencyWorkPlan plan = build_benchmark_latency_work_plan(
      4096, 256, 257, 16, 10000, BenchmarkTarget::MainMemory, 42);
  ASSERT_EQ(plan.status, BenchmarkMeasurementStatus::Measured);
  EXPECT_EQ(plan.chain_node_count, 16u);
  EXPECT_EQ(plan.access_count, 272u);
  EXPECT_EQ(plan.complete_chain_cycles, 17u);
  EXPECT_EQ(plan.seed, 42u);
}

TEST(BenchmarkWorkPlanTest, LatencyPlanEnforcesMinimumCycles) {
  const BenchmarkLatencyWorkPlan plan = build_benchmark_latency_work_plan(
      4096, 256, 1, 16, 10000, BenchmarkTarget::L1, 7);
  ASSERT_EQ(plan.status, BenchmarkMeasurementStatus::Measured);
  EXPECT_EQ(plan.access_count, 256u);
  EXPECT_EQ(plan.complete_chain_cycles, 16u);
}

TEST(BenchmarkWorkPlanTest, CalibrationScalesClampsAndQuantizes) {
  EXPECT_EQ(calculate_benchmark_pilot_passes(
                192, 8 * Constants::BYTES_PER_MB, 100000),
            43691u);
  EXPECT_EQ(calculate_benchmark_calibrated_count(0.010, 100, 0.150, 1,
                                                 10000),
            1500u);
  EXPECT_EQ(calculate_benchmark_calibrated_count(0.010, 100, 0.150, 16,
                                                 10000, 16),
            1504u);
  EXPECT_EQ(calculate_benchmark_calibrated_count(0.0, 100, 0.150, 1,
                                                 10000),
            0u);
}

TEST(BenchmarkWorkPlanTest, CyclicOrderIsDeterministicAndBalanced) {
  EXPECT_EQ(build_benchmark_cyclic_order(4, 0),
            (std::vector<size_t>{0, 1, 2, 3}));
  EXPECT_EQ(build_benchmark_cyclic_order(4, 1),
            (std::vector<size_t>{1, 2, 3, 0}));
  EXPECT_EQ(build_benchmark_cyclic_order(4, 4),
            (std::vector<size_t>{0, 1, 2, 3}));
  EXPECT_TRUE(build_benchmark_cyclic_order(0, 3).empty());
}

TEST(BenchmarkWorkPlanTest, SeedDerivationIsStableAndDomainSeparated) {
  EXPECT_EQ(derive_benchmark_seed(42, 1), derive_benchmark_seed(42, 1));
  EXPECT_NE(derive_benchmark_seed(42, 1), derive_benchmark_seed(42, 2));
  EXPECT_NE(derive_benchmark_seed(42, 1), derive_benchmark_seed(43, 1));
  EXPECT_NE(derive_benchmark_seed(0, 0), 0u);
}

TEST(BenchmarkWorkPlanTest, PhaseScheduleChecksEveryBoundaryWithoutExecutingKernels) {
  const std::vector<size_t> order = {2, 0, 3, 1};
  for (size_t stop_check = 0; stop_check < order.size() * 2; ++stop_check) {
    size_t check_index = 0;
    size_t callback_count = 0;
    const BenchmarkPhaseExecutionResult result =
        execute_benchmark_phase_schedule(
            order,
            [&] { return check_index++ == stop_check; },
            [&](size_t, size_t) { ++callback_count; });

    EXPECT_TRUE(result.interrupted) << "stop check " << stop_check;
    EXPECT_EQ(result.completed_phases, stop_check / 2)
        << "stop check " << stop_check;
    EXPECT_EQ(callback_count,
              stop_check / 2 + (stop_check % 2 == 0 ? 0 : 1))
        << "stop check " << stop_check;
    EXPECT_EQ(result.realized_phase_indexes.size(), callback_count);
  }

  const BenchmarkPhaseExecutionResult complete =
      execute_benchmark_phase_schedule(order, [] { return false; },
                                       [](size_t, size_t) {});
  EXPECT_FALSE(complete.interrupted);
  EXPECT_EQ(complete.completed_phases, order.size());
  EXPECT_EQ(complete.realized_phase_indexes, order);
}

TEST(BenchmarkWorkPlanTest, PhaseSchedulePropagatesInjectedPreparationFailures) {
  for (size_t failing_phase = 0; failing_phase < 4; ++failing_phase) {
    EXPECT_THROW(
        execute_benchmark_phase_schedule(
            {0, 1, 2, 3}, [] { return false; },
            [&](size_t, size_t phase_index) {
              if (phase_index == failing_phase) {
                throw std::runtime_error("injected phase preparation failure");
              }
            }),
        std::runtime_error);
  }
}

TEST(BenchmarkWorkPlanTest, ElapsedValidationRejectsInjectedInvalidTiming) {
  EXPECT_FALSE(benchmark_elapsed_is_valid(0.0));
  EXPECT_FALSE(benchmark_elapsed_is_valid(-1.0));
  EXPECT_FALSE(benchmark_elapsed_is_valid(
      std::numeric_limits<double>::quiet_NaN()));
  EXPECT_FALSE(benchmark_elapsed_is_valid(
      std::numeric_limits<double>::infinity()));
  EXPECT_TRUE(benchmark_elapsed_is_valid(0.001));
}

TEST(BenchmarkWorkPlanTest, DurationQualityDistinguishesMinimumWorkLimit) {
  EXPECT_TRUE(benchmark_duration_in_window(0.250, 0.100, 0.300));
  EXPECT_EQ(classify_benchmark_duration_quality(0.250, 16, 0.100, 0.300),
            "within-target-window");
  EXPECT_EQ(classify_benchmark_duration_quality(0.410, 16, 0.100, 0.300,
                                                true),
            "minimum-complete-cycles-exceed-window");
  EXPECT_EQ(classify_benchmark_duration_quality(0.410, 16, 0.100, 0.300),
            "above-target-window");
}

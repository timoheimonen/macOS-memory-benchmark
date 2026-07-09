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

#include <set>
#include <vector>

#include "benchmark/tlb_measurement_scheduler.h"

namespace {

std::vector<TlbSweepPoint> make_points(size_t count) {
  std::vector<TlbSweepPoint> points(count);
  for (size_t i = 0; i < count; ++i) {
    points[i].point_index = i;
    points[i].locality_bytes = (i + 1) * 16384;
  }
  return points;
}

void expect_same_schedule(const std::vector<TlbMeasurementTask>& lhs,
                          const std::vector<TlbMeasurementTask>& rhs) {
  ASSERT_EQ(lhs.size(), rhs.size());
  for (size_t i = 0; i < lhs.size(); ++i) {
    EXPECT_EQ(lhs[i].pass, rhs[i].pass);
    EXPECT_EQ(lhs[i].point_index, rhs[i].point_index);
    EXPECT_EQ(lhs[i].locality_bytes, rhs[i].locality_bytes);
    EXPECT_EQ(lhs[i].round_index, rhs[i].round_index);
    EXPECT_EQ(lhs[i].order_index, rhs[i].order_index);
    EXPECT_EQ(lhs[i].seed, rhs[i].seed);
  }
}

}  // namespace

TEST(TlbMeasurementSchedulerTest, SameSeedProducesSameScheduleAndTaskSeeds) {
  const std::vector<TlbSweepPoint> points = make_points(5);
  const std::vector<TlbMeasurementTask> first =
      build_tlb_measurement_schedule(points, 7, 123456, TlbMeasurementPass::Base);
  const std::vector<TlbMeasurementTask> second =
      build_tlb_measurement_schedule(points, 7, 123456, TlbMeasurementPass::Base);

  expect_same_schedule(first, second);
}

TEST(TlbMeasurementSchedulerTest, DifferentBaseSeedsDeriveDifferentTaskSeeds) {
  EXPECT_NE(derive_tlb_measurement_seed(1, TlbMeasurementPass::Base, 2, 3),
            derive_tlb_measurement_seed(2, TlbMeasurementPass::Base, 2, 3));
  EXPECT_NE(derive_tlb_measurement_seed(1, TlbMeasurementPass::Base, 2, 3),
            derive_tlb_measurement_seed(1, TlbMeasurementPass::Refinement, 2, 3));
}

TEST(TlbMeasurementSchedulerTest, ValidationPassHasDistinctNameAndSeedDomain) {
  EXPECT_STREQ(tlb_measurement_pass_to_string(TlbMeasurementPass::Validation),
               "validation");
  EXPECT_NE(derive_tlb_measurement_seed(1, TlbMeasurementPass::Base, 2, 3),
            derive_tlb_measurement_seed(1, TlbMeasurementPass::Validation, 2, 3));
}

TEST(TlbMeasurementSchedulerTest, CyclicRoundsBalancePointOrder) {
  const size_t point_count = 5;
  const std::vector<TlbMeasurementTask> schedule = build_tlb_measurement_schedule(
      make_points(point_count), point_count, 42, TlbMeasurementPass::Base);

  ASSERT_EQ(schedule.size(), point_count * point_count);
  std::vector<std::set<size_t>> positions_by_point(point_count);
  for (size_t round = 0; round < point_count; ++round) {
    std::set<size_t> points_in_round;
    for (size_t order = 0; order < point_count; ++order) {
      const TlbMeasurementTask& task = schedule[round * point_count + order];
      EXPECT_EQ(task.round_index, round);
      EXPECT_EQ(task.order_index, order);
      points_in_round.insert(task.point_index);
      positions_by_point[task.point_index].insert(order);
    }
    EXPECT_EQ(points_in_round.size(), point_count);
  }
  for (const std::set<size_t>& positions : positions_by_point) {
    EXPECT_EQ(positions.size(), point_count);
  }
}

TEST(TlbMeasurementSchedulerTest, SchedulePreservesPlannerPointIdentifiers) {
  std::vector<TlbSweepPoint> points = make_points(2);
  points[0].point_index = 10;
  points[1].point_index = 20;

  const std::vector<TlbMeasurementTask> schedule =
      build_tlb_measurement_schedule(points, 1, 42, TlbMeasurementPass::Refinement);

  ASSERT_EQ(schedule.size(), 2u);
  const std::set<size_t> identifiers = {schedule[0].point_index, schedule[1].point_index};
  EXPECT_EQ(identifiers, (std::set<size_t>{10, 20}));
}

TEST(TlbMeasurementSchedulerTest, InjectedStopReturnsDeterministicPartialRecords) {
  const std::vector<TlbMeasurementTask> schedule = build_tlb_measurement_schedule(
      make_points(4), 3, 99, TlbMeasurementPass::Refinement);
  size_t measured_count = 0;

  const TlbScheduleExecutionResult result = execute_tlb_measurement_schedule(
      schedule,
      [&measured_count]() { return measured_count >= 3; },
      [&measured_count](const TlbMeasurementTask&,
                        TlbMeasurementSample& sample) {
        ++measured_count;
        sample.latency_ns = static_cast<double>(measured_count);
        return TlbTaskMeasureStatus::Success;
      });

  EXPECT_EQ(result.status, TlbScheduleExecutionStatus::Interrupted);
  ASSERT_EQ(result.records.size(), 3u);
  EXPECT_DOUBLE_EQ(result.records.back().latency_ns, 3.0);
}

TEST(TlbMeasurementSchedulerTest, MeasurementErrorStopsWithoutAppendingInvalidRecord) {
  const std::vector<TlbMeasurementTask> schedule = build_tlb_measurement_schedule(
      make_points(2), 2, 7, TlbMeasurementPass::Base);

  const TlbScheduleExecutionResult result = execute_tlb_measurement_schedule(
      schedule,
      []() { return false; },
      [](const TlbMeasurementTask&, TlbMeasurementSample&) {
        return TlbTaskMeasureStatus::Error;
      });

  EXPECT_EQ(result.status, TlbScheduleExecutionStatus::Error);
  EXPECT_TRUE(result.records.empty());
}

TEST(TlbMeasurementSchedulerTest, PreservesPairedMeasurementMetadata) {
  const std::vector<TlbMeasurementTask> schedule = build_tlb_measurement_schedule(
      make_points(1), 1, 7, TlbMeasurementPass::Base);

  const TlbScheduleExecutionResult result = execute_tlb_measurement_schedule(
      schedule,
      []() { return false; },
      [](const TlbMeasurementTask&, TlbMeasurementSample& sample) {
        sample.latency_ns = 12.0;
        sample.paired.available = true;
        sample.paired.spread.seed = 101;
        sample.paired.spread.latency_ns = 12.0;
        sample.paired.packed.seed = 202;
        sample.paired.packed.latency_ns = 7.5;
        sample.paired.translation_delta_ns = 4.5;
        return TlbTaskMeasureStatus::Success;
      });

  ASSERT_EQ(result.status, TlbScheduleExecutionStatus::Complete);
  ASSERT_EQ(result.records.size(), 1U);
  EXPECT_TRUE(result.records[0].paired.available);
  EXPECT_EQ(result.records[0].paired.spread.seed, 101U);
  EXPECT_EQ(result.records[0].paired.packed.seed, 202U);
  EXPECT_DOUBLE_EQ(result.records[0].paired.translation_delta_ns, 4.5);
}

TEST(TlbMeasurementSchedulerTest, PairOrderAlternatesAcrossRoundAndOrderParity) {
  TlbMeasurementTask task;
  task.round_index = 0;
  task.order_index = 0;
  EXPECT_TRUE(tlb_measure_spread_first(task));
  task.order_index = 1;
  EXPECT_FALSE(tlb_measure_spread_first(task));
  task.round_index = 1;
  EXPECT_TRUE(tlb_measure_spread_first(task));
}

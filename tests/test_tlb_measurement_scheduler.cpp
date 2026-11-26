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
#include "utils/seed_utils.h"

namespace {

std::vector<TlbSweepPoint> make_points(size_t count) {
  std::vector<TlbSweepPoint> points(count);
  for (size_t i = 0; i < count; ++i) {
    points[i].point_index = i;
    points[i].locality_bytes = (i + 1) * 16384;
  }
  return points;
}

void expect_same_schedule(const std::vector<TlbMeasurementTask>& lhs, const std::vector<TlbMeasurementTask>& rhs) {
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

TEST(TlbMeasurementSchedulerTest, SeedDerivationGoldensAndScheduleDeterminism) {
  {
    SCOPED_TRACE("same seed schedule determinism and pair ordering");
    const std::vector<TlbSweepPoint> points = make_points(5);
    const std::vector<TlbMeasurementTask> first =
        build_tlb_measurement_schedule(points, 7, 123456, TlbMeasurementPass::Base);
    const std::vector<TlbMeasurementTask> second =
        build_tlb_measurement_schedule(points, 7, 123456, TlbMeasurementPass::Base);

    expect_same_schedule(first, second);
    ASSERT_GE(first.size(), 7u);
    EXPECT_TRUE(tlb_measure_spread_first(first[0]));
    EXPECT_FALSE(tlb_measure_spread_first(first[1]));
    EXPECT_TRUE(tlb_measure_spread_first(first[6]));
  }

  SCOPED_TRACE("splitmix64 and task seed goldens");
  EXPECT_EQ(SeedUtils::splitmix64(0), 0xe220a8397b1dcdafULL);
  EXPECT_EQ(SeedUtils::splitmix64(1), 0x910a2dec89025cc1ULL);
  EXPECT_EQ(SeedUtils::splitmix64(0xffffffffffffffffULL), 0xe4d971771b652c20ULL);
  EXPECT_EQ(derive_tlb_measurement_seed(42, TlbMeasurementPass::Base, 0, 0), 0x6310bf04d8207f46ULL);
  EXPECT_EQ(derive_tlb_measurement_seed(42, TlbMeasurementPass::Refinement, 2, 3), 0x64a0c8a842e4f6b4ULL);
  EXPECT_NE(derive_tlb_measurement_seed(1, TlbMeasurementPass::Base, 2, 3),
            derive_tlb_measurement_seed(2, TlbMeasurementPass::Base, 2, 3));
  EXPECT_NE(derive_tlb_measurement_seed(1, TlbMeasurementPass::Base, 2, 3),
            derive_tlb_measurement_seed(1, TlbMeasurementPass::Refinement, 2, 3));
  EXPECT_STREQ(tlb_measurement_pass_to_string(TlbMeasurementPass::Validation), "validation");
  EXPECT_NE(derive_tlb_measurement_seed(1, TlbMeasurementPass::Base, 2, 3),
            derive_tlb_measurement_seed(1, TlbMeasurementPass::Validation, 2, 3));
}

TEST(TlbMeasurementSchedulerTest, CyclicRoundsBalancePointOrderAndPreserveIdentifiers) {
  {
    SCOPED_TRACE("cyclic round balance");
    const size_t point_count = 5;
    const std::vector<TlbMeasurementTask> schedule =
        build_tlb_measurement_schedule(make_points(point_count), point_count, 42, TlbMeasurementPass::Base);

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

  SCOPED_TRACE("preserve planner point identifiers");
  std::vector<TlbSweepPoint> points = make_points(2);
  points[0].point_index = 10;
  points[1].point_index = 20;

  const std::vector<TlbMeasurementTask> schedule =
      build_tlb_measurement_schedule(points, 1, 42, TlbMeasurementPass::Refinement);

  ASSERT_EQ(schedule.size(), 2u);
  const std::set<size_t> identifiers = {schedule[0].point_index, schedule[1].point_index};
  EXPECT_EQ(identifiers, (std::set<size_t>{10, 20}));
}

TEST(TlbMeasurementSchedulerTest, InterruptionAndMeasurementErrorPreserveValidPrefixAndRoundCounts) {
  struct StopOrErrorCase {
    const char* name;
    size_t point_count;
    size_t round_count;
    uint64_t seed;
    TlbMeasurementPass pass;
    bool stop_enabled;
    size_t stop_after_records;
    size_t error_on_call;
    TlbScheduleExecutionStatus expected_status;
    size_t expected_calls;
    size_t expected_records;
    size_t expected_rounds_completed;
  };
  const std::vector<StopOrErrorCase> cases = {
      {"stop before first task", 3, 2, 99, TlbMeasurementPass::Base, true, 0, 0,
       TlbScheduleExecutionStatus::Interrupted, 0, 0, 0},
      {"stop mid round", 4, 3, 99, TlbMeasurementPass::Refinement, true, 2, 0,
       TlbScheduleExecutionStatus::Interrupted, 2, 2, 0},
      {"stop after complete round", 3, 3, 99, TlbMeasurementPass::Validation, true, 3, 0,
       TlbScheduleExecutionStatus::Interrupted, 3, 3, 1},
      {"measurement error after records", 3, 2, 7, TlbMeasurementPass::Base, false, 0, 3,
       TlbScheduleExecutionStatus::Error, 3, 2, 0},
  };

  for (const StopOrErrorCase& entry : cases) {
    SCOPED_TRACE(entry.name);
    const std::vector<TlbMeasurementTask> schedule =
        build_tlb_measurement_schedule(make_points(entry.point_count), entry.round_count, entry.seed, entry.pass);
    size_t measure_calls = 0;

    const TlbScheduleExecutionResult result = execute_tlb_measurement_schedule(
        schedule,
        [&]() { return entry.stop_enabled && measure_calls >= entry.stop_after_records; },
        [&](const TlbMeasurementTask&, TlbMeasurementSample& sample) {
          ++measure_calls;
          if (entry.error_on_call != 0 && measure_calls == entry.error_on_call) {
            return TlbTaskMeasureStatus::Error;
          }
          sample.latency_ns = static_cast<double>(measure_calls);
          return TlbTaskMeasureStatus::Success;
        });

    EXPECT_EQ(result.status, entry.expected_status);
    EXPECT_EQ(measure_calls, entry.expected_calls);
    ASSERT_EQ(result.records.size(), entry.expected_records);
    for (size_t i = 0; i < result.records.size(); ++i) {
      EXPECT_DOUBLE_EQ(result.records[i].latency_ns, static_cast<double>(i + 1));
    }
    EXPECT_EQ(result.rounds_completed, entry.expected_rounds_completed);
    EXPECT_FALSE(result.converged);
  }
}

TEST(TlbMeasurementSchedulerTest, StopsAtConvergenceOrMaximumAfterCompleteRounds) {
  struct ScheduleCase {
    size_t points;
    size_t maximum;
    size_t converged_at;
    size_t completed;
  };
  for (const ScheduleCase& entry : {ScheduleCase{3, 5, 2, 2}, ScheduleCase{2, 3, 0, 3}}) {
    SCOPED_TRACE(entry.converged_at);
    const auto schedule =
        build_tlb_measurement_schedule(make_points(entry.points), entry.maximum, 7, TlbMeasurementPass::Base);
    const TlbScheduleExecutionResult result = execute_tlb_measurement_schedule(
        schedule, []() { return false; },
        [](const TlbMeasurementTask&, TlbMeasurementSample& sample) {
          sample.latency_ns = 1.0;
          return TlbTaskMeasureStatus::Success;
        },
        [&](size_t completed_rounds, const std::vector<TlbMeasurementRecord>& records) {
          EXPECT_EQ(records.size(), completed_rounds * entry.points);
          return completed_rounds == entry.converged_at;
        });
    EXPECT_EQ(result.status, TlbScheduleExecutionStatus::Complete);
    EXPECT_EQ(result.converged, entry.converged_at != 0);
    EXPECT_EQ(result.rounds_completed, entry.completed);
    EXPECT_EQ(result.records.size(), entry.completed * entry.points);
  }
}

TEST(TlbMeasurementSchedulerTest, PreservesPairedMeasurementMetadata) {
  const std::vector<TlbMeasurementTask> schedule =
      build_tlb_measurement_schedule(make_points(1), 1, 7, TlbMeasurementPass::Base);

  const TlbScheduleExecutionResult result = execute_tlb_measurement_schedule(
      schedule, []() { return false; },
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

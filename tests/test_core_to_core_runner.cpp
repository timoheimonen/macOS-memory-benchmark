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
 * @file test_core_to_core_runner.cpp
 * @brief Unit tests for standalone core-to-core measurement loop
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 */

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "benchmark/core_to_core_latency_internal.h"
#include "core/config/constants.h"
#include "core/timing/timer.h"
#include "test_timer_system_calls.h"

namespace {

uint64_t deterministic_timer_ticks() { return 100; }

using ScopedDeterministicTimerSystemCalls = test_timer_system_calls::ScopedTimerSystemCalls<deterministic_timer_ticks>;

CoreToCoreWorkPlan make_small_work_plan() {
  CoreToCoreWorkPlan plan;
  plan.calibrated = true;
  plan.warmup_round_trips = Constants::CORE_TO_CORE_WARMUP_ROUND_TRIPS;
  plan.headline_round_trips = Constants::CORE_TO_CORE_HEADLINE_ROUND_TRIPS;
  plan.sample_window_round_trips = Constants::CORE_TO_CORE_SAMPLE_WINDOW_ROUND_TRIPS;
  return plan;
}

}  // namespace

TEST(CoreToCoreRunnerTest, WorkPlanUsesDurationTargetsFromExcludedPilot) {
  CoreToCoreWorkPlan plan;

  ASSERT_TRUE(build_core_to_core_work_plan(0.010, plan));
  EXPECT_TRUE(plan.calibrated);
  EXPECT_EQ(plan.calibration_round_trips, Constants::CORE_TO_CORE_CALIBRATION_ROUND_TRIPS);
  EXPECT_DOUBLE_EQ(plan.calibration_round_trip_ns, 100.0);
  EXPECT_EQ(plan.warmup_round_trips, 250000u);
  EXPECT_EQ(plan.headline_round_trips, 2500000u);
  EXPECT_EQ(plan.sample_window_round_trips, 10000u);
}

TEST(CoreToCoreRunnerTest, ResponderWorkAccountingRejectsOverflowBeforeThreads) {
  const ScopedDeterministicTimerSystemCalls timer_system_calls;
  const ScenarioDescriptor scenario = {
      Constants::CORE_TO_CORE_SCENARIO_NO_AFFINITY,
      Constants::CORE_TO_CORE_AFFINITY_HINT_DISABLED,
      Constants::CORE_TO_CORE_AFFINITY_TAG_NONE,
      Constants::CORE_TO_CORE_AFFINITY_TAG_NONE,
  };

  CoreToCoreWorkPlan plan = make_small_work_plan();
  plan.warmup_round_trips = std::numeric_limits<size_t>::max();
  plan.headline_round_trips = 1;
  ScenarioMeasurement add_overflow;
  EXPECT_FALSE(execute_single_scenario(scenario, plan, 0, add_overflow));
  EXPECT_EQ(add_overflow.status, CoreToCoreMeasurementStatus::Failed);
  EXPECT_EQ(add_overflow.status_reason, "invalid-or-overflowing-work-plan");

  plan = make_small_work_plan();
  plan.warmup_round_trips = 1;
  plan.headline_round_trips = 1;
  plan.sample_window_round_trips = std::numeric_limits<size_t>::max();
  ScenarioMeasurement multiply_overflow;
  EXPECT_FALSE(execute_single_scenario(scenario, plan, 2, multiply_overflow));
  EXPECT_EQ(multiply_overflow.status, CoreToCoreMeasurementStatus::Failed);
  EXPECT_EQ(multiply_overflow.status_reason, "invalid-or-overflowing-work-plan");
}

TEST(CoreToCoreRunnerTest, DurationQualityIncludesExactWindowBoundaries) {
  const double minimum = Constants::CORE_TO_CORE_HEADLINE_MIN_SECONDS;
  const double maximum = Constants::CORE_TO_CORE_HEADLINE_MAX_SECONDS;

  EXPECT_EQ(classify_core_to_core_duration_quality(std::nextafter(minimum, 0.0)), "below-target-window");
  EXPECT_EQ(classify_core_to_core_duration_quality(minimum), "within-target-window");
  EXPECT_EQ(classify_core_to_core_duration_quality((minimum + maximum) * 0.5), "within-target-window");
  EXPECT_EQ(classify_core_to_core_duration_quality(maximum), "within-target-window");
  EXPECT_EQ(classify_core_to_core_duration_quality(std::nextafter(maximum, std::numeric_limits<double>::infinity())),
            "above-target-window");
}

TEST(CoreToCoreRunnerTest, ScenarioOrderRotatesAcrossLoops) {
  EXPECT_EQ(build_core_to_core_scenario_order(3, 0), (std::vector<size_t>{0, 1, 2}));
  EXPECT_EQ(build_core_to_core_scenario_order(3, 1), (std::vector<size_t>{1, 2, 0}));
  EXPECT_EQ(build_core_to_core_scenario_order(3, 2), (std::vector<size_t>{2, 0, 1}));
  EXPECT_EQ(build_core_to_core_scenario_order(3, 3), (std::vector<size_t>{0, 1, 2}));
  EXPECT_TRUE(build_core_to_core_scenario_order(0, 4).empty());
}

TEST(CoreToCoreRunnerTest, LoopRecordSampleRangesCountOnlySamplesAppendedToPool) {
  CoreToCoreLatencyScenarioResult scenario_result;
  scenario_result.sample_round_trip_ns = {5.0};

  ScenarioMeasurement invalid_measurement;
  invalid_measurement.status = CoreToCoreMeasurementStatus::Invalid;
  invalid_measurement.status_reason = "invalid-sample-elapsed";
  invalid_measurement.samples_ns = {10.0, 11.0};
  append_core_to_core_loop_record(scenario_result, 0, 2, invalid_measurement);

  ASSERT_EQ(scenario_result.loop_records.size(), 1u);
  EXPECT_EQ(scenario_result.sample_round_trip_ns, (std::vector<double>{5.0}));
  EXPECT_EQ(scenario_result.loop_records[0].sample_start_index, 1u);
  EXPECT_EQ(scenario_result.loop_records[0].completed_sample_windows, 0u);
  EXPECT_EQ(scenario_result.completed_loops, 0u);

  ScenarioMeasurement measured;
  measured.status = CoreToCoreMeasurementStatus::Measured;
  measured.round_trip_ns = 20.0;
  measured.headline_elapsed_seconds = 0.2;
  measured.duration_quality = "within-target-window";
  measured.samples_ns = {21.0, 22.0};
  append_core_to_core_loop_record(scenario_result, 1, 0, measured);

  ASSERT_EQ(scenario_result.loop_records.size(), 2u);
  EXPECT_EQ(scenario_result.sample_round_trip_ns, (std::vector<double>{5.0, 21.0, 22.0}));
  EXPECT_EQ(scenario_result.loop_records[1].sample_start_index, 1u);
  EXPECT_EQ(scenario_result.loop_records[1].completed_sample_windows, 2u);
  EXPECT_EQ(scenario_result.completed_loops, 1u);
}

TEST(CoreToCoreRunnerTest, DeterministicFailureSeamsReturnExplicitStatus) {
  const ScopedDeterministicTimerSystemCalls timer_system_calls;
  const ScenarioDescriptor scenario = {
      Constants::CORE_TO_CORE_SCENARIO_NO_AFFINITY,
      Constants::CORE_TO_CORE_AFFINITY_HINT_DISABLED,
      Constants::CORE_TO_CORE_AFFINITY_TAG_NONE,
      Constants::CORE_TO_CORE_AFFINITY_TAG_NONE,
  };
  const CoreToCoreWorkPlan plan = make_small_work_plan();

  ScenarioMeasurement timer_failure;
  CoreToCoreFailureInjection fail_timer;
  fail_timer.fail_timer_creation = true;
  EXPECT_FALSE(execute_single_scenario(scenario, plan, 0, timer_failure, &fail_timer));
  EXPECT_EQ(timer_failure.status, CoreToCoreMeasurementStatus::Failed);
  EXPECT_EQ(timer_failure.status_reason, "timer-creation-failed");

  ScenarioMeasurement responder_failure;
  CoreToCoreFailureInjection fail_responder;
  fail_responder.fail_responder_startup = true;
  EXPECT_FALSE(execute_single_scenario(scenario, plan, 0, responder_failure, &fail_responder));
  EXPECT_EQ(responder_failure.status_reason, "responder-thread-startup-failed");
}

TEST(CoreToCoreRunnerTest, InitiatorStartupFailureCleansUpResponderIntegration) {
  const ScenarioDescriptor scenario = {
      Constants::CORE_TO_CORE_SCENARIO_NO_AFFINITY,
      Constants::CORE_TO_CORE_AFFINITY_HINT_DISABLED,
      Constants::CORE_TO_CORE_AFFINITY_TAG_NONE,
      Constants::CORE_TO_CORE_AFFINITY_TAG_NONE,
  };
  const CoreToCoreWorkPlan plan = make_small_work_plan();
  ScenarioMeasurement initiator_failure;
  CoreToCoreFailureInjection fail_initiator;
  fail_initiator.fail_initiator_startup = true;
  EXPECT_FALSE(execute_single_scenario(scenario, plan, 0, initiator_failure, &fail_initiator));
  EXPECT_EQ(initiator_failure.status_reason, "initiator-thread-startup-failed");
}

TEST(CoreToCoreRunnerTest, ExecuteSingleScenarioProducesHeadlineAndSamplesIntegration) {
  const ScenarioDescriptor scenario = {
      Constants::CORE_TO_CORE_SCENARIO_NO_AFFINITY,
      Constants::CORE_TO_CORE_AFFINITY_HINT_DISABLED,
      Constants::CORE_TO_CORE_AFFINITY_TAG_NONE,
      Constants::CORE_TO_CORE_AFFINITY_TAG_NONE,
  };
  ScenarioMeasurement measurement;
  const CoreToCoreWorkPlan plan = make_small_work_plan();

  const bool ok = execute_single_scenario(scenario, plan, 1, measurement);

  ASSERT_TRUE(ok);
  EXPECT_EQ(measurement.status, CoreToCoreMeasurementStatus::Measured);
  EXPECT_GT(measurement.round_trip_ns, 0.0);
  ASSERT_EQ(measurement.samples_ns.size(), 1u);
  EXPECT_GT(measurement.samples_ns[0], 0.0);
  EXPECT_FALSE(std::isnan(measurement.round_trip_ns));
  EXPECT_FALSE(std::isinf(measurement.round_trip_ns));
}

TEST(CoreToCoreRunnerTest, ExecuteSingleScenarioSupportsZeroSamplesIntegration) {
  const ScenarioDescriptor scenario = {
      Constants::CORE_TO_CORE_SCENARIO_DIFFERENT_AFFINITY,
      Constants::CORE_TO_CORE_AFFINITY_HINT_ENABLED,
      Constants::CORE_TO_CORE_AFFINITY_TAG_PRIMARY,
      Constants::CORE_TO_CORE_AFFINITY_TAG_SECONDARY,
  };
  ScenarioMeasurement measurement;
  const CoreToCoreWorkPlan plan = make_small_work_plan();

  const bool ok = execute_single_scenario(scenario, plan, 0, measurement);

  ASSERT_TRUE(ok);
  EXPECT_EQ(measurement.status, CoreToCoreMeasurementStatus::Measured);
  EXPECT_GT(measurement.round_trip_ns, 0.0);
  EXPECT_TRUE(measurement.samples_ns.empty());
  EXPECT_EQ(measurement.initiator_hint.affinity_requested, scenario.apply_affinity);
  EXPECT_EQ(measurement.responder_hint.affinity_requested, scenario.apply_affinity);
  EXPECT_EQ(measurement.initiator_hint.affinity_tag, scenario.initiator_affinity_tag);
  EXPECT_EQ(measurement.responder_hint.affinity_tag, scenario.responder_affinity_tag);
}

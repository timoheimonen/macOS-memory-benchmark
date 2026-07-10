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
#include <string>
#include <utility>
#include <vector>

#include "benchmark/core_to_core_latency_internal.h"
#include "benchmark/core_to_core_sweep_runner.h"
#include "core/config/constants.h"
#include "core/timing/timer.h"
#include "test_timer_system_calls.h"

namespace {

uint64_t deterministic_timer_ticks() { return 100; }

using ScopedDeterministicTimerSystemCalls =
    test_timer_system_calls::ScopedTimerSystemCalls<deterministic_timer_ticks>;

CoreToCoreWorkPlan make_small_work_plan() {
  CoreToCoreWorkPlan plan;
  plan.calibrated = true;
  plan.warmup_round_trips = Constants::CORE_TO_CORE_WARMUP_ROUND_TRIPS;
  plan.headline_round_trips = Constants::CORE_TO_CORE_HEADLINE_ROUND_TRIPS;
  plan.sample_window_round_trips = Constants::CORE_TO_CORE_SAMPLE_WINDOW_ROUND_TRIPS;
  return plan;
}

using Json = nlohmann::ordered_json;

std::vector<Json> make_core_to_core_sweep_parameters(size_t count) {
  std::vector<Json> parameters;
  parameters.reserve(count);
  for (size_t index = 0; index < count; ++index) {
    parameters.push_back({{"latency-samples", index + 1}});
  }
  return parameters;
}

Json make_core_to_core_sweep_result(const std::string& status, bool measurements_complete) {
  return {{"core_to_core_latency", {{"status", status}, {"measurements_complete", measurements_complete}}}};
}

SweepExecutionHooks make_core_to_core_sweep_hooks(const std::vector<SweepRunOutcome>& outcomes,
                                                  std::vector<Json>& checkpoints, std::vector<bool>& announce_flags,
                                                  size_t& executed_runs) {
  SweepExecutionHooks hooks;
  hooks.execute_run = [&](size_t run_index) {
    ++executed_runs;
    return outcomes.at(run_index);
  };
  hooks.stop_requested = []() { return false; };
  hooks.elapsed_seconds = []() { return 1.5; };
  hooks.utc_timestamp = []() { return "2026-01-01T00:00:00Z"; };
  hooks.write_checkpoint = [&](const Json& output, bool announce_success) {
    checkpoints.push_back(output);
    announce_flags.push_back(announce_success);
    return EXIT_SUCCESS;
  };
  return hooks;
}

}  // namespace

TEST(CoreToCoreRunnerTest, CalibrationRoundsScaleAndClampDeterministically) {
  EXPECT_EQ(calculate_core_to_core_calibrated_round_trips(0.010, 100000, 0.250, 1000, 10000000), 2500000u);
  EXPECT_EQ(calculate_core_to_core_calibrated_round_trips(1.0, 1000, 0.001, 2000, 10000000), 2000u);
  EXPECT_EQ(calculate_core_to_core_calibrated_round_trips(0.001, 100000, 1.0, 1000, 500000), 500000u);
  EXPECT_EQ(calculate_core_to_core_calibrated_round_trips(0.0, 100000, 0.250, 1000, 10000000), 0u);
}

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

TEST(CoreToCoreRunnerTest, SummaryStatisticsUseExactLinearPercentilesAndSampleVariance) {
  const CoreToCoreSummaryStats stats = calculate_core_to_core_summary_stats({5.0, 1.0, 4.0, 2.0, 3.0});
  const double expected_sample_stddev = std::sqrt(2.5);

  EXPECT_DOUBLE_EQ(stats.median, 3.0);
  EXPECT_DOUBLE_EQ(stats.p90, 4.6);
  EXPECT_DOUBLE_EQ(stats.p95, 4.8);
  EXPECT_DOUBLE_EQ(stats.p99, 4.96);
  EXPECT_DOUBLE_EQ(stats.sample_stddev, expected_sample_stddev);
  EXPECT_DOUBLE_EQ(stats.coefficient_of_variation_pct, expected_sample_stddev / 3.0 * 100.0);
  EXPECT_DOUBLE_EQ(stats.median_absolute_deviation, 1.0);
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

TEST(CoreToCoreRunnerTest, SweepCompletesAndCheckpointsEachAttempt) {
  const std::vector<SweepRunOutcome> outcomes = {
      {EXIT_SUCCESS, make_core_to_core_sweep_result("complete", true), ""},
      {EXIT_SUCCESS, make_core_to_core_sweep_result("complete", true), ""},
  };
  std::vector<Json> checkpoints;
  std::vector<bool> announce_flags;
  size_t executed_runs = 0;

  const SweepExecutionResult execution = execute_core_to_core_sweep_plan(
      make_core_to_core_sweep_parameters(2), Json::object(),
      make_core_to_core_sweep_hooks(outcomes, checkpoints, announce_flags, executed_runs));

  ASSERT_EQ(execution.exit_code, EXIT_SUCCESS);
  EXPECT_EQ(executed_runs, 2u);
  ASSERT_EQ(checkpoints.size(), 2u);
  EXPECT_EQ(checkpoints[0]["status"], "partial");
  EXPECT_EQ(checkpoints[0]["attempted_runs"], 1u);
  EXPECT_EQ(checkpoints[0]["completed_runs"], 1u);
  EXPECT_FALSE(checkpoints[0]["conclusions_valid"]);
  EXPECT_EQ(execution.output_json["status"], "complete");
  EXPECT_EQ(execution.output_json["planned_runs"], 2u);
  EXPECT_EQ(execution.output_json["attempted_runs"], 2u);
  EXPECT_EQ(execution.output_json["completed_runs"], 2u);
  EXPECT_TRUE(execution.output_json["conclusions_valid"]);
  EXPECT_EQ(announce_flags, (std::vector<bool>{false, true}));
}

TEST(CoreToCoreRunnerTest, SweepInterruptedAttemptIsRetainedButNotCompleted) {
  const std::vector<SweepRunOutcome> outcomes = {
      {EXIT_SUCCESS, make_core_to_core_sweep_result("complete", true), ""},
      {EXIT_SUCCESS, make_core_to_core_sweep_result("interrupted", false), ""},
  };
  std::vector<Json> checkpoints;
  std::vector<bool> announce_flags;
  size_t executed_runs = 0;

  const SweepExecutionResult execution = execute_core_to_core_sweep_plan(
      make_core_to_core_sweep_parameters(2), Json::object(),
      make_core_to_core_sweep_hooks(outcomes, checkpoints, announce_flags, executed_runs));

  ASSERT_EQ(execution.exit_code, EXIT_SUCCESS);
  EXPECT_EQ(executed_runs, 2u);
  EXPECT_EQ(execution.output_json["status"], "interrupted");
  EXPECT_EQ(execution.output_json["attempted_runs"], 2u);
  EXPECT_EQ(execution.output_json["completed_runs"], 1u);
  EXPECT_FALSE(execution.output_json["conclusions_valid"]);
  ASSERT_EQ(execution.output_json["runs"].size(), 2u);
  EXPECT_EQ(execution.output_json["runs"][0]["status"], "complete");
  EXPECT_EQ(execution.output_json["runs"][1]["status"], "interrupted");
  EXPECT_EQ(execution.output_json["runs"][1]["status_reason"], "nested-core-to-core-run-interrupted");
}

TEST(CoreToCoreRunnerTest, SweepCompleteStatusWithoutCompleteMeasurementsIsPartial) {
  const std::vector<SweepRunOutcome> outcomes = {
      {EXIT_SUCCESS, make_core_to_core_sweep_result("complete", false), ""},
  };
  std::vector<Json> checkpoints;
  std::vector<bool> announce_flags;
  size_t executed_runs = 0;

  const SweepExecutionResult execution = execute_core_to_core_sweep_plan(
      make_core_to_core_sweep_parameters(1), Json::object(),
      make_core_to_core_sweep_hooks(outcomes, checkpoints, announce_flags, executed_runs));

  ASSERT_EQ(execution.exit_code, EXIT_SUCCESS);
  EXPECT_EQ(execution.output_json["status"], "partial");
  EXPECT_EQ(execution.output_json["status_reason"], "nested-core-to-core-result-incomplete");
  EXPECT_EQ(execution.output_json["attempted_runs"], 1u);
  EXPECT_EQ(execution.output_json["completed_runs"], 0u);
  EXPECT_FALSE(execution.output_json["conclusions_valid"]);
  EXPECT_EQ(execution.output_json["runs"][0]["status"], "partial");
}

TEST(CoreToCoreRunnerTest, SweepExecutionFailurePreservesPriorCheckpointedRun) {
  const std::vector<SweepRunOutcome> outcomes = {
      {EXIT_SUCCESS, make_core_to_core_sweep_result("complete", true), ""},
      {EXIT_FAILURE, make_core_to_core_sweep_result("failed", false), "simulated-core-to-core-failure"},
  };
  std::vector<Json> checkpoints;
  std::vector<bool> announce_flags;
  size_t executed_runs = 0;

  const SweepExecutionResult execution = execute_core_to_core_sweep_plan(
      make_core_to_core_sweep_parameters(2), Json::object(),
      make_core_to_core_sweep_hooks(outcomes, checkpoints, announce_flags, executed_runs));

  ASSERT_EQ(execution.exit_code, EXIT_FAILURE);
  EXPECT_EQ(executed_runs, 2u);
  ASSERT_EQ(checkpoints.size(), 2u);
  EXPECT_EQ(checkpoints[0]["runs"][0]["status"], "complete");
  EXPECT_EQ(execution.output_json["status"], "failed");
  EXPECT_EQ(execution.output_json["attempted_runs"], 2u);
  EXPECT_EQ(execution.output_json["completed_runs"], 1u);
  EXPECT_FALSE(execution.output_json["conclusions_valid"]);
  EXPECT_EQ(execution.output_json["runs"][1]["status"], "failed");
  EXPECT_EQ(execution.output_json["runs"][1]["status_reason"], "simulated-core-to-core-failure");
}

TEST(CoreToCoreRunnerTest, SweepCheckpointFailureStopsFurtherAttempts) {
  const std::vector<SweepRunOutcome> outcomes = {
      {EXIT_SUCCESS, make_core_to_core_sweep_result("complete", true), ""},
      {EXIT_SUCCESS, make_core_to_core_sweep_result("complete", true), ""},
      {EXIT_SUCCESS, make_core_to_core_sweep_result("complete", true), ""},
  };
  std::vector<Json> attempted_checkpoints;
  size_t executed_runs = 0;
  SweepExecutionHooks hooks;
  hooks.execute_run = [&](size_t run_index) {
    ++executed_runs;
    return outcomes.at(run_index);
  };
  hooks.stop_requested = []() { return false; };
  hooks.elapsed_seconds = []() { return 2.0; };
  hooks.utc_timestamp = []() { return "2026-01-01T00:00:00Z"; };
  hooks.write_checkpoint = [&](const Json& output, bool) {
    attempted_checkpoints.push_back(output);
    return attempted_checkpoints.size() == 2 ? EXIT_FAILURE : EXIT_SUCCESS;
  };

  const SweepExecutionResult execution =
      execute_core_to_core_sweep_plan(make_core_to_core_sweep_parameters(3), Json::object(), hooks);

  ASSERT_EQ(execution.exit_code, EXIT_FAILURE);
  EXPECT_EQ(executed_runs, 2u);
  ASSERT_EQ(attempted_checkpoints.size(), 2u);
  EXPECT_EQ(attempted_checkpoints[0]["completed_runs"], 1u);
  EXPECT_EQ(execution.output_json["status"], "failed");
  EXPECT_EQ(execution.output_json["status_reason"], "checkpoint-write-failed");
  EXPECT_EQ(execution.output_json["attempted_runs"], 2u);
  EXPECT_EQ(execution.output_json["completed_runs"], 2u);
  EXPECT_FALSE(execution.output_json["conclusions_valid"]);
}

TEST(CoreToCoreRunnerTest, SweepInterruptionAfterCompleteRunKeepsCompletionCount) {
  const std::vector<SweepRunOutcome> outcomes = {
      {EXIT_SUCCESS, make_core_to_core_sweep_result("complete", true), ""},
      {EXIT_SUCCESS, make_core_to_core_sweep_result("complete", true), ""},
  };
  std::vector<Json> checkpoints;
  std::vector<bool> announce_flags;
  size_t executed_runs = 0;
  size_t stop_checks = 0;
  SweepExecutionHooks hooks = make_core_to_core_sweep_hooks(outcomes, checkpoints, announce_flags, executed_runs);
  hooks.stop_requested = [&]() {
    ++stop_checks;
    return stop_checks >= 2;
  };

  const SweepExecutionResult execution =
      execute_core_to_core_sweep_plan(make_core_to_core_sweep_parameters(2), Json::object(), hooks);

  ASSERT_EQ(execution.exit_code, EXIT_SUCCESS);
  EXPECT_EQ(executed_runs, 1u);
  EXPECT_EQ(execution.output_json["status"], "interrupted");
  EXPECT_EQ(execution.output_json["status_reason"], "interruption-requested-after-complete-run");
  EXPECT_EQ(execution.output_json["attempted_runs"], 1u);
  EXPECT_EQ(execution.output_json["completed_runs"], 1u);
  EXPECT_FALSE(execution.output_json["conclusions_valid"]);
  EXPECT_EQ(execution.output_json["runs"][0]["status"], "complete");
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

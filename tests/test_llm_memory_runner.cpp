// Copyright 2026 Timo Heimonen <timo.heimonen@proton.me>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/config/constants.h"
#include "llm_memory/llm_runner.h"
#include "utils/numeric_utils.h"

namespace {

struct TaskRecord {
  LlmRunnerTaskContext context;
  size_t steps = 0;
  size_t payload_bytes = 0;
  std::string plan_identity;
};

struct CheckpointRecord {
  LlmCheckpointKind kind = LlmCheckpointKind::MeasurementTerminal;
  LlmRunStatus status = LlmRunStatus::NotStarted;
  size_t planned_loops = 0;
  size_t attempted_loops = 0;
  size_t completed_loops = 0;
  size_t planned_measurements = 0;
  size_t attempted_measurements = 0;
  size_t terminal_measurements = 0;
  size_t measured_measurements = 0;
  size_t logical_checkpoint_attempts = 0;
  size_t successful_logical_checkpoints = 0;
  bool terminal_checkpoint_attempted = false;
  bool terminal_checkpoint_completed = false;
  bool interruption_requested = false;
};

LlmMemoryConfig explicit_config(size_t loop_count = 3, size_t iterations = 4) {
  LlmMemoryConfig config;
  config.weight_size_mb = 1;
  config.layer_count = 1;
  config.query_head_count = 1;
  config.kv_head_count = 1;
  config.head_dimension = 16;
  config.kv_element_bytes = 1;
  config.visible_context_tokens = 2;
  config.batch_size = 1;
  config.requested_workers = 2;
  config.available_workers = 2;
  config.iterations = iterations;
  config.loop_count = loop_count;
  config.seed = 42;
  config.user_specified_iterations = true;
  config.user_specified_seed = true;
  config.user_specified_workers = true;
  return config;
}

LlmMemoryConfig automatic_config(size_t loop_count = 1) {
  LlmMemoryConfig config = explicit_config(loop_count);
  config.iterations = 0;
  config.user_specified_iterations = false;
  return config;
}

LlmMemoryWorkPlanRequest plan_request(const LlmMemoryConfig& config) {
  LlmMemoryWorkPlanRequest request;
  request.geometry.active_weight_bytes = config.weight_size_mb * Constants::BYTES_PER_MB;
  request.geometry.layer_count = config.layer_count;
  request.geometry.query_head_count = config.query_head_count;
  request.geometry.kv_head_count = config.kv_head_count;
  request.geometry.head_dimension = config.head_dimension;
  request.geometry.kv_element_bytes = config.kv_element_bytes;
  request.geometry.visible_context_tokens = config.visible_context_tokens;
  request.geometry.batch_size = config.batch_size;
  request.requested_workers = config.requested_workers;
  request.available_workers = config.available_workers;
  request.available_memory_bytes = 8ULL * 1024ULL * Constants::BYTES_PER_MB;
  request.mapping_granularity_bytes = 1;
  request.base_seed = config.seed;
  return request;
}

LlmMemoryWorkPlan build_runner_admitted_plan(const LlmMemoryConfig& config) {
  LlmMemoryWorkPlanRequest request = plan_request(config);
  const LlmMemoryWorkPlan preliminary = build_llm_memory_work_plan(request);
  if (!preliminary.valid) {
    return build_llm_memory_work_plan(request);
  }
  const LlmExecutorAuxiliaryEstimate executor = calculate_llm_executor_auxiliary_estimate(preliminary);
  const LlmRunnerAuxiliaryEstimate runner = calculate_llm_runner_auxiliary_estimate(config, preliminary);
  if (!executor.valid || !runner.valid ||
      !NumericUtils::checked_add(executor.checksum_auxiliary_bytes, runner.checksum_auxiliary_bytes,
                                 request.checksum_auxiliary_bytes) ||
      !NumericUtils::checked_add(executor.orchestration_auxiliary_bytes, runner.orchestration_auxiliary_bytes,
                                 request.orchestration_auxiliary_bytes)) {
    return build_llm_memory_work_plan(request);
  }
  return build_llm_memory_work_plan(request);
}

LlmExecutorResult successful_execution(const LlmMemoryWorkPlan& model_plan, double elapsed_seconds) {
  LlmExecutorResult result;
  result.valid = true;
  result.reason_code = LlmExecutorReason::VALID;
  result.elapsed_seconds = elapsed_seconds;
  result.requested_workers = model_plan.effective_workers;
  result.created_workers = model_plan.effective_workers;
  result.completed_workers = model_plan.effective_workers;
  result.qos_successful_workers = model_plan.effective_workers;
  result.kernel_succeeded = true;
  result.timer_started = true;
  result.timer_stopped = true;
  result.checksum_evaluated = true;
  result.checksum_valid = true;
  result.expected_checksums.resize(model_plan.effective_workers);
  result.actual_checksums = result.expected_checksums;
  result.expected_run_checksum = {11, 22};
  result.actual_run_checksum = result.expected_run_checksum;
  return result;
}

class FakeRunnerExecutor {
 public:
  std::vector<TaskRecord> calls;
  std::function<void(const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext&, size_t,
                     LlmExecutorResult&)>
      mutate;

  LlmTaskExecutor callback() {
    return [this](const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan& task_plan,
                  const LlmRunnerTaskContext& context) {
      calls.push_back({context, task_plan.steps, task_plan.effective_payload_bytes, task_plan.plan_identity});
      LlmExecutorResult result = successful_execution(model_plan, 0.150);
      if (mutate) {
        mutate(model_plan, task_plan, context, calls.size(), result);
      }
      return result;
    };
  }
};

CheckpointRecord capture_checkpoint(const LlmMemoryResult& result, LlmCheckpointKind kind) {
  CheckpointRecord record;
  record.kind = kind;
  record.status = result.status;
  record.planned_loops = result.counters.planned_loops;
  record.attempted_loops = result.counters.attempted_loops;
  record.completed_loops = result.counters.completed_loops;
  record.planned_measurements = result.counters.planned_measurements;
  record.attempted_measurements = result.counters.attempted_measurements;
  record.terminal_measurements = result.counters.terminal_measurements;
  record.measured_measurements = result.counters.measured_measurements;
  record.logical_checkpoint_attempts = result.logical_checkpoint_attempts;
  record.successful_logical_checkpoints = result.successful_logical_checkpoints;
  record.terminal_checkpoint_attempted = result.terminal_checkpoint_attempted;
  record.terminal_checkpoint_completed = result.terminal_checkpoint_completed;
  record.interruption_requested = result.interruption_requested;
  return record;
}

LlmRunnerHooks recording_checkpoints(std::vector<CheckpointRecord>& checkpoints) {
  LlmRunnerHooks hooks;
  hooks.stop_requested = []() { return false; };
  hooks.checkpoint = [&](const LlmMemoryResult& result, LlmCheckpointKind kind) {
    checkpoints.push_back(capture_checkpoint(result, kind));
    return EXIT_SUCCESS;
  };
  return hooks;
}

std::vector<TaskRecord> records_for_scenario(const std::vector<TaskRecord>& records, LlmScenario scenario,
                                             bool measurements) {
  std::vector<TaskRecord> selected;
  for (const TaskRecord& record : records) {
    if (record.context.scenario == scenario &&
        (record.context.kind == LlmRunnerTaskKind::Measurement) == measurements) {
      selected.push_back(record);
    }
  }
  return selected;
}

void expect_interrupted_tail(const LlmMemoryResult& result, size_t measured_prefix) {
  ASSERT_LE(measured_prefix, result.measurements.size());
  for (size_t index = 0; index < result.measurements.size(); ++index) {
    const LlmMeasurementState& measurement = result.measurements[index];
    if (index < measured_prefix) {
      EXPECT_EQ(measurement.status, LlmMeasurementStatus::Measured) << index;
      EXPECT_TRUE(measurement.elapsed_seconds.has_value()) << index;
    } else {
      EXPECT_EQ(measurement.status, LlmMeasurementStatus::Interrupted) << index;
      EXPECT_EQ(measurement.reason_code, LlmRunnerReason::INTERRUPTION_BEFORE_TASK) << index;
      EXPECT_FALSE(measurement.elapsed_seconds.has_value()) << index;
      EXPECT_EQ(measurement.completed_steps, 0u) << index;
    }
  }
}

TEST(LlmMemoryRunnerTest, ExplicitWarmupsUseExactMeasurementShapeAndCompleteBalancedRun) {
  const LlmMemoryConfig config = explicit_config();
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  FakeRunnerExecutor executor;
  std::vector<CheckpointRecord> checkpoints;
  LlmMemoryResult result;

  ASSERT_EQ(run_llm_memory_suite(config, plan, executor.callback(), result, recording_checkpoints(checkpoints)),
            EXIT_SUCCESS);
  ASSERT_TRUE(result.initialized);
  EXPECT_EQ(result.status, LlmRunStatus::Complete);
  EXPECT_TRUE(result.results_complete);
  EXPECT_TRUE(result.conclusions_valid);
  EXPECT_TRUE(result.scenario_order_balance_complete);
  EXPECT_FALSE(result.interruption_requested);
  ASSERT_EQ(executor.calls.size(), 12u);
  for (size_t index = 0; index < kLlmScenarioCount; ++index) {
    const TaskRecord& warmup = executor.calls[index];
    const LlmScenarioWorkPlan& frozen = result.frozen_scenario_plans.scenarios[index];
    EXPECT_EQ(warmup.context.kind, LlmRunnerTaskKind::Warmup);
    EXPECT_EQ(warmup.context.purpose, "explicit-warmup");
    EXPECT_EQ(warmup.context.scenario, static_cast<LlmScenario>(index));
    EXPECT_EQ(warmup.steps, config.iterations);
    EXPECT_EQ(warmup.steps, frozen.steps);
    EXPECT_EQ(warmup.payload_bytes, frozen.effective_payload_bytes);
    EXPECT_EQ(warmup.plan_identity, frozen.plan_identity);
    const std::vector<TaskRecord> measured = records_for_scenario(executor.calls, warmup.context.scenario, true);
    ASSERT_EQ(measured.size(), config.loop_count);
    for (const TaskRecord& record : measured) {
      EXPECT_EQ(record.steps, frozen.steps);
      EXPECT_EQ(record.payload_bytes, frozen.effective_payload_bytes);
      EXPECT_EQ(record.plan_identity, frozen.plan_identity);
    }
  }

  ASSERT_EQ(result.loops.size(), 3u);
  EXPECT_EQ(result.loops[0].planned_order,
            (std::array<LlmScenario, 3>{LlmScenario::WeightsOnly, LlmScenario::KvOnly, LlmScenario::Mixed}));
  EXPECT_EQ(result.loops[1].planned_order,
            (std::array<LlmScenario, 3>{LlmScenario::KvOnly, LlmScenario::Mixed, LlmScenario::WeightsOnly}));
  EXPECT_EQ(result.loops[2].planned_order,
            (std::array<LlmScenario, 3>{LlmScenario::Mixed, LlmScenario::WeightsOnly, LlmScenario::KvOnly}));
  for (const LlmLoopRecord& loop : result.loops) {
    EXPECT_EQ(loop.realized_order_count, kLlmScenarioCount);
    EXPECT_EQ(loop.realized_order, loop.planned_order);
  }

  EXPECT_EQ(result.counters.planned_loops, 3u);
  EXPECT_EQ(result.counters.attempted_loops, 3u);
  EXPECT_EQ(result.counters.completed_loops, 3u);
  EXPECT_EQ(result.counters.planned_measurements, 9u);
  EXPECT_EQ(result.counters.attempted_measurements, 9u);
  EXPECT_EQ(result.counters.terminal_measurements, 9u);
  EXPECT_EQ(result.counters.measured_measurements, 9u);
  EXPECT_EQ(result.counters.planned_synthetic_steps, 36u);
  EXPECT_EQ(result.counters.completed_synthetic_steps, 36u);
  size_t expected_payload_per_loop = 0;
  for (const LlmScenarioWorkPlan& scenario : result.frozen_scenario_plans.scenarios) {
    expected_payload_per_loop += scenario.effective_payload_bytes;
  }
  EXPECT_EQ(result.counters.planned_exact_payload_bytes, expected_payload_per_loop * 3);
  EXPECT_EQ(result.counters.completed_exact_payload_bytes, expected_payload_per_loop * 3);
  for (const LlmMeasurementState& measurement : result.measurements) {
    ASSERT_LT(measurement.frozen_plan_index, kLlmScenarioCount);
    const LlmScenarioWorkPlan& frozen = result.frozen_scenario_plans.scenarios[measurement.frozen_plan_index];
    EXPECT_EQ(measurement.scenario, frozen.scenario);
    EXPECT_EQ(measurement.planned_steps, frozen.steps);
    EXPECT_EQ(measurement.planned_exact_payload_bytes, frozen.effective_payload_bytes);
    EXPECT_EQ(measurement.execution.expected_checksums.size(), plan.effective_workers);
    EXPECT_EQ(measurement.execution.actual_checksums.size(), plan.effective_workers);
  }

  ASSERT_EQ(checkpoints.size(), 10u);
  for (size_t index = 0; index < 9; ++index) {
    EXPECT_EQ(checkpoints[index].kind, LlmCheckpointKind::MeasurementTerminal);
    EXPECT_EQ(checkpoints[index].planned_loops, 3u);
    EXPECT_EQ(checkpoints[index].attempted_loops, index / kLlmScenarioCount + 1);
    EXPECT_EQ(checkpoints[index].completed_loops, (index + 1) / kLlmScenarioCount);
    EXPECT_EQ(checkpoints[index].planned_measurements, 9u);
    EXPECT_EQ(checkpoints[index].attempted_measurements, index + 1);
    EXPECT_EQ(checkpoints[index].terminal_measurements, index + 1);
    EXPECT_EQ(checkpoints[index].measured_measurements, index + 1);
    EXPECT_EQ(checkpoints[index].logical_checkpoint_attempts, index + 1);
    EXPECT_EQ(checkpoints[index].successful_logical_checkpoints, index + 1);
    EXPECT_FALSE(checkpoints[index].terminal_checkpoint_attempted);
    EXPECT_FALSE(checkpoints[index].terminal_checkpoint_completed);
    EXPECT_FALSE(checkpoints[index].interruption_requested);
    EXPECT_EQ(checkpoints[index].status, index == 8 ? LlmRunStatus::Complete : LlmRunStatus::Partial);
  }
  EXPECT_EQ(checkpoints.back().kind, LlmCheckpointKind::CommandTerminal);
  EXPECT_EQ(checkpoints.back().status, LlmRunStatus::Complete);
  EXPECT_EQ(checkpoints.back().attempted_measurements, 9u);
  EXPECT_EQ(checkpoints.back().terminal_measurements, 9u);
  EXPECT_EQ(checkpoints.back().measured_measurements, 9u);
  EXPECT_EQ(checkpoints.back().logical_checkpoint_attempts, 10u);
  EXPECT_EQ(checkpoints.back().successful_logical_checkpoints, 10u);
  EXPECT_TRUE(checkpoints.back().terminal_checkpoint_attempted);
  EXPECT_TRUE(checkpoints.back().terminal_checkpoint_completed);
  EXPECT_EQ(result.logical_checkpoint_attempts, 10u);
  EXPECT_EQ(result.successful_logical_checkpoints, 10u);
  EXPECT_TRUE(result.terminal_checkpoint_completed);

  for (const LlmScenarioAggregate& aggregate : result.aggregates) {
    EXPECT_EQ(aggregate.status, "complete");
    EXPECT_EQ(aggregate.stability_quality, "stable");
    EXPECT_EQ(aggregate.effective_payload_gb_s.values.size(), 3u);
    EXPECT_TRUE(aggregate.effective_payload_gb_s.headline.has_value());
  }
}

TEST(LlmMemoryRunnerTest, CompleteSingleLoopIsInspectableButComparativeConclusionsAreInvalid) {
  const LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  FakeRunnerExecutor executor;
  LlmMemoryResult result;

  ASSERT_EQ(run_llm_memory_suite(config, plan, executor.callback(), result), EXIT_SUCCESS);
  EXPECT_EQ(result.status, LlmRunStatus::Complete);
  EXPECT_TRUE(result.results_complete);
  EXPECT_FALSE(result.scenario_order_balance_complete);
  EXPECT_FALSE(result.conclusions_valid);
  EXPECT_EQ(result.quality_warnings, (std::vector<std::string_view>{"scenario-order-not-balanced"}));
  EXPECT_EQ(result.logical_checkpoint_attempts, 4u);
  for (const LlmScenarioAggregate& aggregate : result.aggregates) {
    EXPECT_EQ(aggregate.stability_quality, "insufficient-samples");
    ASSERT_EQ(aggregate.effective_payload_gb_s.values.size(), 1u);
    EXPECT_EQ(aggregate.effective_payload_gb_s.headline, aggregate.effective_payload_gb_s.values.front());
  }
}

TEST(LlmMemoryRunnerTest, OversizedExecutorBackingIsCanonicalizedIntoFrozenRunnerCapacities) {
  const LlmMemoryConfig config = explicit_config();
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  constexpr size_t oversized_capacity = 4096;
  FakeRunnerExecutor executor;
  executor.mutate = [](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context,
                       size_t, LlmExecutorResult& execution) {
    if (context.kind != LlmRunnerTaskKind::Measurement) {
      return;
    }
    execution.reason_code.reserve(oversized_capacity);
    execution.expected_checksums.reserve(oversized_capacity);
    execution.actual_checksums.reserve(oversized_capacity);
  };
  LlmMemoryResult result;

  ASSERT_EQ(run_llm_memory_suite(config, plan, executor.callback(), result), EXIT_SUCCESS);
  EXPECT_TRUE(result.results_complete);
  size_t retained_checksum_bytes = 0;
  for (const LlmMeasurementState& measurement : result.measurements) {
    EXPECT_EQ(measurement.execution.reason_code, LlmExecutorReason::VALID);
    EXPECT_LT(measurement.execution.reason_code.capacity(), oversized_capacity);
    EXPECT_LT(measurement.execution.expected_checksums.capacity(), oversized_capacity);
    EXPECT_LT(measurement.execution.actual_checksums.capacity(), oversized_capacity);
    retained_checksum_bytes +=
        (measurement.execution.expected_checksums.capacity() + measurement.execution.actual_checksums.capacity()) *
        sizeof(LlmWorkerChecksum);
  }
  EXPECT_LE(retained_checksum_bytes, result.runner_auxiliary.retained_checksum_bytes);
  EXPECT_EQ(result.measurements.capacity(), result.counters.planned_measurements);
  EXPECT_EQ(result.loops.capacity(), config.loop_count);
  EXPECT_EQ(result.statistics_workspace.sorted_values.capacity(), config.loop_count);
  EXPECT_EQ(result.statistics_workspace.absolute_deviations.capacity(), config.loop_count);
  for (const LlmScenarioAggregate& aggregate : result.aggregates) {
    EXPECT_EQ(aggregate.step_latency_seconds.values.capacity(), config.loop_count);
    EXPECT_EQ(aggregate.synthetic_memory_steps_per_second.values.capacity(), config.loop_count);
    EXPECT_EQ(aggregate.effective_payload_gb_s.values.capacity(), config.loop_count);
  }
}

TEST(LlmMemoryRunnerTest, AutomaticCalibrationIsScenarioSpecificAndFrozenBeforeLoopZero) {
  const LlmMemoryConfig config = automatic_config(2);
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  FakeRunnerExecutor executor;
  executor.mutate = [](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context,
                       size_t, LlmExecutorResult& result) {
    const size_t index = static_cast<size_t>(context.scenario);
    if (context.purpose == "pilot") {
      result.elapsed_seconds = 0.010 * (index + 1);
    } else if (context.purpose == "duration-trial") {
      result.elapsed_seconds = 0.150;
    } else if (context.kind == LlmRunnerTaskKind::Measurement) {
      result.elapsed_seconds = 0.400;
    }
  };
  LlmMemoryResult result;

  ASSERT_EQ(run_llm_memory_suite(config, plan, executor.callback(), result), EXIT_SUCCESS);
  ASSERT_TRUE(result.frozen_scenario_plans.valid);
  for (LlmScenario scenario : {LlmScenario::WeightsOnly, LlmScenario::KvOnly, LlmScenario::Mixed}) {
    const size_t index = static_cast<size_t>(scenario);
    const std::vector<TaskRecord> excluded = records_for_scenario(executor.calls, scenario, false);
    ASSERT_EQ(excluded.size(), 4u);
    EXPECT_EQ(excluded[0].context.purpose, "pilot-warmup");
    EXPECT_EQ(excluded[0].steps, 1u);
    EXPECT_EQ(excluded[1].context.purpose, "pilot");
    EXPECT_EQ(excluded[2].context.purpose, "duration-trial-warmup");
    EXPECT_EQ(excluded[3].context.purpose, "duration-trial");
    EXPECT_EQ(excluded[2].steps, excluded[3].steps);
    EXPECT_EQ(excluded[2].payload_bytes, excluded[3].payload_bytes);
    EXPECT_EQ(excluded[2].plan_identity, excluded[3].plan_identity);
    const LlmScenarioLimits limits = calculate_llm_scenario_limits(plan.geometry, scenario);
    const size_t expected_steps = calculate_llm_calibrated_steps(0.010 * (index + 1), excluded[1].steps, limits);
    EXPECT_EQ(excluded[2].steps, expected_steps);
    EXPECT_EQ(result.frozen_scenario_plans.scenarios[index].steps, expected_steps);
    EXPECT_EQ(excluded[3].payload_bytes, result.frozen_scenario_plans.scenarios[index].effective_payload_bytes);
    EXPECT_EQ(excluded[3].plan_identity, result.frozen_scenario_plans.scenarios[index].plan_identity);

    const std::vector<TaskRecord> measured = records_for_scenario(executor.calls, scenario, true);
    ASSERT_EQ(measured.size(), 2u);
    EXPECT_EQ(measured[0].steps, expected_steps);
    EXPECT_EQ(measured[1].steps, expected_steps);
    EXPECT_EQ(measured[0].plan_identity, measured[1].plan_identity);
  }
  EXPECT_EQ(result.calibration_attempts[0].size(), 4u);
  EXPECT_EQ(result.calibration_attempts[1].size(), 4u);
  EXPECT_EQ(result.calibration_attempts[2].size(), 4u);
}

TEST(LlmMemoryRunnerTest, AutomaticCalibrationUsesAtMostTwoCorrectionsWithoutExtraWarmups) {
  const LlmMemoryConfig config = automatic_config();
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  FakeRunnerExecutor executor;
  executor.mutate = [](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context,
                       size_t, LlmExecutorResult& result) {
    if (context.purpose == "pilot") {
      result.elapsed_seconds = 0.010;
    } else if (context.purpose == "duration-trial") {
      result.elapsed_seconds = 0.050;
    } else if (context.purpose == "correction-trial-1") {
      result.elapsed_seconds = 0.300;
    } else if (context.purpose == "correction-trial-2") {
      result.elapsed_seconds = 0.050;
    }
  };
  LlmMemoryResult result;

  ASSERT_EQ(run_llm_memory_suite(config, plan, executor.callback(), result), EXIT_SUCCESS);
  for (LlmScenario scenario : {LlmScenario::WeightsOnly, LlmScenario::KvOnly, LlmScenario::Mixed}) {
    const size_t index = static_cast<size_t>(scenario);
    const std::vector<TaskRecord> excluded = records_for_scenario(executor.calls, scenario, false);
    ASSERT_EQ(excluded.size(), 6u);
    EXPECT_EQ(excluded[4].context.purpose, "correction-trial-1");
    EXPECT_EQ(excluded[5].context.purpose, "correction-trial-2");
    EXPECT_EQ(result.frozen_scenario_plans.scenarios[index].steps, excluded[5].steps);
    EXPECT_EQ(std::count_if(excluded.begin(), excluded.end(),
                            [](const TaskRecord& record) { return record.context.kind == LlmRunnerTaskKind::Warmup; }),
              2);
  }
}

TEST(LlmMemoryRunnerTest, StopBeforeFirstTaskInterruptsAllSlotsWithoutExecutorWork) {
  const LlmMemoryConfig config = explicit_config(2);
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  FakeRunnerExecutor executor;
  std::vector<CheckpointRecord> checkpoints;
  LlmRunnerHooks hooks = recording_checkpoints(checkpoints);
  hooks.stop_requested = []() { return true; };
  LlmMemoryResult result;

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor.callback(), result, hooks), EXIT_SUCCESS);
  EXPECT_TRUE(executor.calls.empty());
  EXPECT_EQ(result.status, LlmRunStatus::Interrupted);
  EXPECT_TRUE(result.interruption_requested);
  EXPECT_EQ(result.counters.attempted_loops, 0u);
  EXPECT_EQ(result.counters.attempted_measurements, 0u);
  EXPECT_EQ(result.counters.terminal_measurements, 6u);
  EXPECT_TRUE(result.frozen_scenario_plans.valid);
  EXPECT_GT(result.counters.planned_synthetic_steps, 0u);
  EXPECT_GT(result.counters.planned_exact_payload_bytes, 0u);
  for (const LlmMeasurementState& measurement : result.measurements) {
    EXPECT_LT(measurement.frozen_plan_index, kLlmScenarioCount);
    EXPECT_EQ(measurement.planned_steps, config.iterations);
    EXPECT_GT(measurement.planned_exact_payload_bytes, 0u);
  }
  expect_interrupted_tail(result, 0);
  ASSERT_EQ(checkpoints.size(), 1u);
  EXPECT_EQ(checkpoints[0].kind, LlmCheckpointKind::CommandTerminal);
}

TEST(LlmMemoryRunnerTest, StopDuringStartedMeasurementKeepsCurrentTaskAndInterruptsTail) {
  const LlmMemoryConfig config = explicit_config(2);
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  bool stop = false;
  FakeRunnerExecutor executor;
  executor.mutate = [&](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context,
                        size_t, LlmExecutorResult&) {
    if (context.kind == LlmRunnerTaskKind::Measurement) {
      stop = true;
    }
  };
  std::vector<CheckpointRecord> checkpoints;
  LlmRunnerHooks hooks = recording_checkpoints(checkpoints);
  hooks.stop_requested = [&]() { return stop; };
  LlmMemoryResult result;

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor.callback(), result, hooks), EXIT_SUCCESS);
  EXPECT_EQ(executor.calls.size(), 4u);
  EXPECT_EQ(result.status, LlmRunStatus::Interrupted);
  EXPECT_TRUE(result.interruption_requested);
  expect_interrupted_tail(result, 1);
  EXPECT_EQ(result.counters.attempted_measurements, 1u);
  EXPECT_EQ(result.counters.planned_loops, 2u);
  EXPECT_EQ(result.counters.attempted_loops, 1u);
  EXPECT_EQ(result.counters.completed_loops, 0u);
  EXPECT_EQ(result.counters.planned_measurements, 6u);
  EXPECT_EQ(result.counters.terminal_measurements, 6u);
  EXPECT_EQ(result.counters.measured_measurements, 1u);
  EXPECT_EQ(result.counters.completed_synthetic_steps, config.iterations);
  EXPECT_EQ(result.counters.completed_exact_payload_bytes, result.measurements[0].planned_exact_payload_bytes);
  ASSERT_EQ(checkpoints.size(), 2u);
  EXPECT_EQ(checkpoints[0].kind, LlmCheckpointKind::MeasurementTerminal);
  EXPECT_EQ(checkpoints[1].kind, LlmCheckpointKind::CommandTerminal);
}

TEST(LlmMemoryRunnerTest, StopRaisedByMeasurementCheckpointAddsOnlyCommandTerminalSnapshot) {
  const LlmMemoryConfig config = explicit_config(2);
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  bool stop = false;
  FakeRunnerExecutor executor;
  std::vector<CheckpointRecord> checkpoints;
  LlmRunnerHooks hooks;
  hooks.stop_requested = [&]() { return stop; };
  hooks.checkpoint = [&](const LlmMemoryResult& snapshot, LlmCheckpointKind kind) {
    checkpoints.push_back(capture_checkpoint(snapshot, kind));
    if (kind == LlmCheckpointKind::MeasurementTerminal) {
      stop = true;
    }
    return EXIT_SUCCESS;
  };
  LlmMemoryResult result;

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor.callback(), result, hooks), EXIT_SUCCESS);
  ASSERT_EQ(checkpoints.size(), 2u);
  EXPECT_EQ(checkpoints[0].kind, LlmCheckpointKind::MeasurementTerminal);
  EXPECT_EQ(checkpoints[0].status, LlmRunStatus::Partial);
  EXPECT_EQ(checkpoints[1].kind, LlmCheckpointKind::CommandTerminal);
  EXPECT_EQ(checkpoints[1].status, LlmRunStatus::Interrupted);
  EXPECT_EQ(result.counters.attempted_measurements, 1u);
  expect_interrupted_tail(result, 1);
}

TEST(LlmMemoryRunnerTest, StopAfterLastSuccessfulTaskRetainsCompleteAcceptedEvidence) {
  const LlmMemoryConfig config = explicit_config();
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  bool stop = false;
  FakeRunnerExecutor executor;
  std::vector<CheckpointRecord> checkpoints;
  LlmRunnerHooks hooks;
  hooks.stop_requested = [&]() { return stop; };
  hooks.checkpoint = [&](const LlmMemoryResult& snapshot, LlmCheckpointKind kind) {
    checkpoints.push_back(capture_checkpoint(snapshot, kind));
    if (kind == LlmCheckpointKind::MeasurementTerminal && snapshot.counters.measured_measurements == 9) {
      stop = true;
    }
    return EXIT_SUCCESS;
  };
  LlmMemoryResult result;

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor.callback(), result, hooks), EXIT_SUCCESS);
  EXPECT_TRUE(result.interruption_requested);
  EXPECT_EQ(result.status, LlmRunStatus::Complete);
  EXPECT_TRUE(result.results_complete);
  EXPECT_TRUE(result.scenario_order_balance_complete);
  EXPECT_TRUE(result.conclusions_valid);
  EXPECT_EQ(result.counters.measured_measurements, 9u);
  EXPECT_EQ(result.logical_checkpoint_attempts, 10u);
  ASSERT_EQ(checkpoints.size(), 10u);
  EXPECT_FALSE(checkpoints[8].interruption_requested);
  EXPECT_TRUE(checkpoints.back().interruption_requested);
}

TEST(LlmMemoryRunnerTest, StopHookExceptionAfterLastMeasurementInvalidatesOtherwiseCompleteConclusions) {
  const LlmMemoryConfig config = explicit_config();
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  bool throw_from_stop = false;
  FakeRunnerExecutor executor;
  std::vector<CheckpointRecord> checkpoints;
  LlmRunnerHooks hooks;
  hooks.stop_requested = [&]() {
    if (throw_from_stop) {
      throw std::runtime_error("injected stop failure");
    }
    return false;
  };
  hooks.checkpoint = [&](const LlmMemoryResult& snapshot, LlmCheckpointKind kind) {
    checkpoints.push_back(capture_checkpoint(snapshot, kind));
    if (kind == LlmCheckpointKind::MeasurementTerminal && snapshot.counters.measured_measurements == 9) {
      throw_from_stop = true;
    }
    return EXIT_SUCCESS;
  };
  LlmMemoryResult result;

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor.callback(), result, hooks), EXIT_FAILURE);
  EXPECT_EQ(result.status, LlmRunStatus::Failed);
  EXPECT_EQ(result.reason_code, LlmRunnerReason::RUNNER_EXCEPTION);
  EXPECT_EQ(result.diagnostic, "injected stop failure");
  EXPECT_TRUE(result.results_complete);
  EXPECT_TRUE(result.scenario_order_balance_complete);
  EXPECT_FALSE(result.conclusions_valid);
  EXPECT_EQ(result.counters.measured_measurements, 9u);
  ASSERT_EQ(checkpoints.size(), 10u);
  EXPECT_EQ(checkpoints[8].kind, LlmCheckpointKind::MeasurementTerminal);
  EXPECT_EQ(checkpoints.back().kind, LlmCheckpointKind::CommandTerminal);
  EXPECT_EQ(checkpoints.back().status, LlmRunStatus::Failed);
}

TEST(LlmMemoryRunnerTest, ExecutorFailureWinsSimultaneousStopAndPreservesInterruptedTail) {
  const LlmMemoryConfig config = explicit_config(2);
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  bool stop = false;
  FakeRunnerExecutor executor;
  executor.mutate = [&](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context,
                        size_t, LlmExecutorResult& execution) {
    if (context.kind == LlmRunnerTaskKind::Measurement) {
      stop = true;
      execution.valid = false;
      execution.reason_code = LlmExecutorReason::KERNEL_FAILED;
      execution.kernel_succeeded = false;
      execution.checksum_valid = false;
    }
  };
  LlmRunnerHooks hooks;
  hooks.stop_requested = [&]() { return stop; };
  LlmMemoryResult result;

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor.callback(), result, hooks), EXIT_FAILURE);
  EXPECT_EQ(result.status, LlmRunStatus::Failed);
  EXPECT_EQ(result.reason_code, LlmExecutorReason::KERNEL_FAILED);
  EXPECT_TRUE(result.interruption_requested);
  ASSERT_EQ(result.measurements[0].status, LlmMeasurementStatus::Failed);
  EXPECT_EQ(result.measurements[0].reason_code, LlmExecutorReason::KERNEL_FAILED);
  for (size_t index = 1; index < result.measurements.size(); ++index) {
    EXPECT_EQ(result.measurements[index].status, LlmMeasurementStatus::Interrupted);
  }
}

TEST(LlmMemoryRunnerTest, ChecksumMismatchWinsSimultaneousStopAndIsExcludedFromAggregates) {
  const LlmMemoryConfig config = explicit_config(2);
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  bool stop = false;
  FakeRunnerExecutor executor;
  executor.mutate = [&](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context,
                        size_t, LlmExecutorResult& execution) {
    if (context.kind == LlmRunnerTaskKind::Measurement) {
      stop = true;
      execution.valid = false;
      execution.reason_code = LlmExecutorReason::CHECKSUM_MISMATCH;
      execution.checksum_valid = false;
    }
  };
  LlmRunnerHooks hooks;
  hooks.stop_requested = [&]() { return stop; };
  LlmMemoryResult result;

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor.callback(), result, hooks), EXIT_FAILURE);
  EXPECT_EQ(result.status, LlmRunStatus::Failed);
  EXPECT_EQ(result.reason_code, LlmExecutorReason::CHECKSUM_MISMATCH);
  EXPECT_TRUE(result.interruption_requested);
  EXPECT_EQ(result.measurements[0].status, LlmMeasurementStatus::Invalid);
  EXPECT_FALSE(result.measurements[0].elapsed_seconds.has_value());
  EXPECT_TRUE(result.aggregates[0].effective_payload_gb_s.values.empty());
}

TEST(LlmMemoryRunnerTest, MeasurementCheckpointFailureIsTerminalAndNeverRetried) {
  const LlmMemoryConfig config = explicit_config(2);
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  FakeRunnerExecutor executor;
  std::vector<LlmCheckpointKind> kinds;
  LlmRunnerHooks hooks;
  hooks.stop_requested = []() { return false; };
  hooks.checkpoint = [&](const LlmMemoryResult&, LlmCheckpointKind kind) {
    kinds.push_back(kind);
    return EXIT_FAILURE;
  };
  LlmMemoryResult result;

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor.callback(), result, hooks), EXIT_FAILURE);
  EXPECT_EQ(kinds, (std::vector<LlmCheckpointKind>{LlmCheckpointKind::MeasurementTerminal}));
  EXPECT_EQ(result.status, LlmRunStatus::Failed);
  EXPECT_EQ(result.reason_code, LlmRunnerReason::CHECKPOINT_WRITE_FAILED);
  EXPECT_TRUE(result.checkpoint_failed);
  EXPECT_FALSE(result.terminal_checkpoint_attempted);
  EXPECT_EQ(result.counters.attempted_measurements, 1u);
  EXPECT_EQ(result.measurements[0].status, LlmMeasurementStatus::Measured);
  EXPECT_TRUE(result.measurements[0].elapsed_seconds.has_value());
  EXPECT_TRUE(result.measurements[0].synthetic_step_latency_seconds.has_value());
  EXPECT_TRUE(result.measurements[0].synthetic_memory_steps_per_second.has_value());
  EXPECT_TRUE(result.measurements[0].effective_payload_gb_s.has_value());
  EXPECT_TRUE(result.measurements[0].checksum_valid);
  EXPECT_TRUE(result.measurements[0].execution.valid);
  EXPECT_TRUE(result.measurements[0].execution.checksum_valid);
  EXPECT_EQ(result.measurements[0].execution.expected_checksums.size(), plan.effective_workers);
  EXPECT_EQ(result.measurements[0].execution.actual_checksums.size(), plan.effective_workers);
  EXPECT_EQ(result.aggregates[0].effective_payload_gb_s.values.size(), 1u);
  EXPECT_EQ(result.counters.planned_loops, 2u);
  EXPECT_EQ(result.counters.attempted_loops, 1u);
  EXPECT_EQ(result.counters.completed_loops, 0u);
  EXPECT_EQ(result.counters.planned_measurements, 6u);
  EXPECT_EQ(result.counters.terminal_measurements, 6u);
  EXPECT_EQ(result.counters.measured_measurements, 1u);
  EXPECT_EQ(result.counters.completed_synthetic_steps, config.iterations);
  EXPECT_EQ(result.counters.completed_exact_payload_bytes, result.measurements[0].planned_exact_payload_bytes);
  for (size_t index = 1; index < result.measurements.size(); ++index) {
    EXPECT_EQ(result.measurements[index].status, LlmMeasurementStatus::Failed);
  }

}

TEST(LlmMemoryRunnerTest, CanonicalResultReasonsDetachFromEveryOwningDomain) {
  std::string runner_reason = LlmRunnerReason::CHECKPOINT_WRITE_FAILED;
  std::string work_plan_reason = LlmWorkPlanReason::EXACT_PAYLOAD_CAP_EXCEEDED;
  std::string executor_reason = LlmExecutorReason::KERNEL_FAILED;

  const std::string_view canonical_runner = canonicalize_llm_result_reason_code(runner_reason);
  const std::string_view canonical_work_plan = canonicalize_llm_result_reason_code(work_plan_reason);
  const std::string_view canonical_executor = canonicalize_llm_result_reason_code(executor_reason);
  std::fill(runner_reason.begin(), runner_reason.end(), 'x');
  std::fill(work_plan_reason.begin(), work_plan_reason.end(), 'x');
  std::fill(executor_reason.begin(), executor_reason.end(), 'x');

  EXPECT_EQ(canonical_runner, LlmRunnerReason::CHECKPOINT_WRITE_FAILED);
  EXPECT_EQ(canonical_runner.data(),
            canonicalize_llm_result_reason_code(LlmRunnerReason::CHECKPOINT_WRITE_FAILED).data());
  EXPECT_EQ(canonical_work_plan, LlmWorkPlanReason::EXACT_PAYLOAD_CAP_EXCEEDED);
  EXPECT_EQ(canonical_work_plan.data(),
            canonicalize_llm_result_reason_code(LlmWorkPlanReason::EXACT_PAYLOAD_CAP_EXCEEDED).data());
  EXPECT_EQ(canonical_executor, LlmExecutorReason::KERNEL_FAILED);
  EXPECT_EQ(canonical_executor.data(),
            canonicalize_llm_result_reason_code(LlmExecutorReason::KERNEL_FAILED).data());
  EXPECT_EQ(canonicalize_llm_result_reason_code("not-a-reason"), LlmRunnerReason::RUNNER_UNKNOWN_EXCEPTION);
}

TEST(LlmMemoryRunnerTest, ResultCopiesAndMovesRetainStaticRunnerFailureAfterSourcesReset) {
  const LlmMemoryConfig config = explicit_config(2);
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  FakeRunnerExecutor executor;
  executor.mutate = [](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context,
                       size_t, LlmExecutorResult&) {
    if (context.kind == LlmRunnerTaskKind::Measurement) {
      throw std::runtime_error("injected runner failure");
    }
  };
  LlmMemoryResult result;

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor.callback(), result), EXIT_FAILURE);
  ASSERT_EQ(result.measurements.size(), config.loop_count * kLlmScenarioCount);
  EXPECT_EQ(result.measurements[0].reason_code, LlmRunnerReason::RUNNER_EXCEPTION);

  LlmMemoryResult copied = result;
  LlmMemoryResult moved = std::move(copied);
  const std::string_view canonical_runner_exception =
      canonicalize_llm_result_reason_code(LlmRunnerReason::RUNNER_EXCEPTION);
  const std::string_view canonical_not_run =
      canonicalize_llm_result_reason_code(LlmRunnerReason::NOT_RUN_AFTER_RUNTIME_FAILURE);
  EXPECT_EQ(moved.measurements[0].reason_code.data(), canonical_runner_exception.data());
  result = LlmMemoryResult{};
  copied = LlmMemoryResult{};

  EXPECT_EQ(moved.measurements[0].reason_code, LlmRunnerReason::RUNNER_EXCEPTION);
  EXPECT_EQ(moved.measurements[0].reason_code.data(), canonical_runner_exception.data());
  for (size_t index = 1; index < moved.measurements.size(); ++index) {
    EXPECT_EQ(moved.measurements[index].reason_code, LlmRunnerReason::NOT_RUN_AFTER_RUNTIME_FAILURE);
    EXPECT_EQ(moved.measurements[index].reason_code.data(), canonical_not_run.data());
  }
}

TEST(LlmMemoryRunnerTest, CheckpointFailureWinsPendingStopAndInterruptsOnlyUnstartedTail) {
  const LlmMemoryConfig config = explicit_config(2);
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  bool stop = false;
  FakeRunnerExecutor executor;
  std::vector<LlmCheckpointKind> kinds;
  LlmRunnerHooks hooks;
  hooks.stop_requested = [&]() { return stop; };
  hooks.checkpoint = [&](const LlmMemoryResult&, LlmCheckpointKind kind) {
    kinds.push_back(kind);
    stop = true;
    return EXIT_FAILURE;
  };
  LlmMemoryResult result;

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor.callback(), result, hooks), EXIT_FAILURE);
  EXPECT_EQ(kinds, (std::vector<LlmCheckpointKind>{LlmCheckpointKind::MeasurementTerminal}));
  EXPECT_EQ(result.status, LlmRunStatus::Failed);
  EXPECT_EQ(result.reason_code, LlmRunnerReason::CHECKPOINT_WRITE_FAILED);
  EXPECT_TRUE(result.checkpoint_failed);
  EXPECT_TRUE(result.interruption_requested);
  EXPECT_EQ(result.measurements[0].status, LlmMeasurementStatus::Measured);
  EXPECT_EQ(result.counters.planned_loops, 2u);
  EXPECT_EQ(result.counters.attempted_loops, 1u);
  EXPECT_EQ(result.counters.completed_loops, 0u);
  EXPECT_EQ(result.counters.planned_measurements, 6u);
  EXPECT_EQ(result.counters.attempted_measurements, 1u);
  EXPECT_EQ(result.counters.terminal_measurements, 6u);
  EXPECT_EQ(result.counters.measured_measurements, 1u);
  for (size_t index = 1; index < result.measurements.size(); ++index) {
    EXPECT_EQ(result.measurements[index].status, LlmMeasurementStatus::Interrupted) << index;
    EXPECT_EQ(result.measurements[index].reason_code, LlmRunnerReason::INTERRUPTION_BEFORE_TASK) << index;
  }
}

TEST(LlmMemoryRunnerTest, TypedAndUnknownCheckpointExceptionsAreTerminalAndNeverRetried) {
  for (bool typed : {true, false}) {
    SCOPED_TRACE(typed ? "typed" : "unknown");
    const LlmMemoryConfig config = explicit_config(2);
    const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    FakeRunnerExecutor executor;
    std::vector<LlmCheckpointKind> kinds;
    LlmRunnerHooks hooks;
    hooks.stop_requested = []() { return false; };
    hooks.checkpoint = [&](const LlmMemoryResult&, LlmCheckpointKind kind) -> int {
      kinds.push_back(kind);
      if (typed) {
        throw std::runtime_error("injected checkpoint failure");
      }
      throw 19;
    };
    LlmMemoryResult result;

    EXPECT_EQ(run_llm_memory_suite(config, plan, executor.callback(), result, hooks), EXIT_FAILURE);
    EXPECT_EQ(kinds, (std::vector<LlmCheckpointKind>{LlmCheckpointKind::MeasurementTerminal}));
    EXPECT_EQ(result.status, LlmRunStatus::Failed);
    EXPECT_EQ(result.reason_code,
              typed ? LlmRunnerReason::RUNNER_EXCEPTION : LlmRunnerReason::RUNNER_UNKNOWN_EXCEPTION);
    EXPECT_TRUE(result.checkpoint_failed);
    EXPECT_EQ(result.logical_checkpoint_attempts, 1u);
    EXPECT_EQ(result.successful_logical_checkpoints, 0u);
    EXPECT_FALSE(result.terminal_checkpoint_attempted);
    EXPECT_EQ(result.measurements[0].status, LlmMeasurementStatus::Measured);
    for (size_t index = 1; index < result.measurements.size(); ++index) {
      EXPECT_EQ(result.measurements[index].status, LlmMeasurementStatus::Failed) << index;
    }
    if (typed) {
      EXPECT_EQ(result.diagnostic, "injected checkpoint failure");
    }
  }
}

TEST(LlmMemoryRunnerTest, CheckpointReturnFailureOverridesEarlierExecutorExceptionWithoutRetry) {
  const LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  FakeRunnerExecutor executor;
  executor.mutate = [](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context,
                       size_t, LlmExecutorResult&) {
    if (context.kind == LlmRunnerTaskKind::Measurement) {
      throw std::runtime_error("earlier executor diagnostic");
    }
  };
  std::vector<LlmCheckpointKind> kinds;
  LlmRunnerHooks hooks;
  hooks.stop_requested = []() { return false; };
  hooks.checkpoint = [&](const LlmMemoryResult&, LlmCheckpointKind kind) {
    kinds.push_back(kind);
    return EXIT_FAILURE;
  };
  LlmMemoryResult result;

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor.callback(), result, hooks), EXIT_FAILURE);
  EXPECT_EQ(kinds, (std::vector<LlmCheckpointKind>{LlmCheckpointKind::MeasurementTerminal}));
  EXPECT_EQ(result.status, LlmRunStatus::Failed);
  EXPECT_EQ(result.reason_code, LlmRunnerReason::CHECKPOINT_WRITE_FAILED);
  EXPECT_TRUE(result.diagnostic.empty());
  EXPECT_TRUE(result.checkpoint_failed);
  EXPECT_EQ(result.logical_checkpoint_attempts, 1u);
  EXPECT_EQ(result.successful_logical_checkpoints, 0u);
  EXPECT_FALSE(result.terminal_checkpoint_attempted);
}

TEST(LlmMemoryRunnerTest, CommandTerminalCheckpointFailureIsNotRetried) {
  const LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  FakeRunnerExecutor executor;
  std::vector<LlmCheckpointKind> kinds;
  LlmRunnerHooks hooks;
  hooks.stop_requested = []() { return false; };
  hooks.checkpoint = [&](const LlmMemoryResult&, LlmCheckpointKind kind) {
    kinds.push_back(kind);
    return kind == LlmCheckpointKind::CommandTerminal ? EXIT_FAILURE : EXIT_SUCCESS;
  };
  LlmMemoryResult result;

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor.callback(), result, hooks), EXIT_FAILURE);
  ASSERT_EQ(kinds.size(), 4u);
  EXPECT_EQ(kinds.back(), LlmCheckpointKind::CommandTerminal);
  EXPECT_EQ(result.logical_checkpoint_attempts, 4u);
  EXPECT_EQ(result.successful_logical_checkpoints, 3u);
  EXPECT_TRUE(result.terminal_checkpoint_attempted);
  EXPECT_FALSE(result.terminal_checkpoint_completed);
  EXPECT_EQ(result.status, LlmRunStatus::Failed);
  EXPECT_EQ(result.reason_code, LlmRunnerReason::CHECKPOINT_WRITE_FAILED);
  EXPECT_TRUE(result.results_complete);
  EXPECT_FALSE(result.conclusions_valid);
  EXPECT_EQ(result.counters.measured_measurements, 3u);
}

TEST(LlmMemoryRunnerTest, MeasuredOnlyStatisticsKeepRawValuesAndClassifyCvThreshold) {
  const LlmMemoryConfig config = explicit_config();
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  constexpr std::array<double, 3> stable_values = {95.0, 100.0, 105.0};
  constexpr std::array<double, 3> noisy_values = {90.0, 100.0, 110.0};
  FakeRunnerExecutor executor;
  executor.mutate = [&](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan& task_plan,
                        const LlmRunnerTaskContext& context, size_t, LlmExecutorResult& execution) {
    if (context.kind != LlmRunnerTaskKind::Measurement) {
      return;
    }
    const double gb_s = context.scenario == LlmScenario::WeightsOnly ? stable_values[context.loop_index]
                                                                     : noisy_values[context.loop_index];
    execution.elapsed_seconds = static_cast<double>(task_plan.effective_payload_bytes) / gb_s / 1.0e9;
  };
  LlmMemoryResult result;

  ASSERT_EQ(run_llm_memory_suite(config, plan, executor.callback(), result), EXIT_SUCCESS);
  const LlmScenarioAggregate& weights = result.aggregates[0];
  EXPECT_EQ(weights.effective_payload_gb_s.values.size(), 3u);
  EXPECT_NEAR(weights.effective_payload_gb_s.statistics.coefficient_of_variation_pct, 5.0, 1e-10);
  EXPECT_EQ(weights.stability_quality, "stable");
  EXPECT_DOUBLE_EQ(*weights.effective_payload_gb_s.headline, 100.0);

  for (size_t index : {1u, 2u}) {
    const LlmScenarioAggregate& aggregate = result.aggregates[index];
    EXPECT_EQ(aggregate.effective_payload_gb_s.values.size(), 3u);
    EXPECT_GT(aggregate.effective_payload_gb_s.statistics.coefficient_of_variation_pct,
              Constants::LLM_STREAMING_CV_WARNING_PCT);
    EXPECT_EQ(aggregate.stability_quality, "noisy");
  }
  EXPECT_EQ(result.quality_warnings, (std::vector<std::string_view>{"kv_only-high-cv", "mixed-high-cv"}));
  EXPECT_TRUE(result.results_complete);
  EXPECT_TRUE(result.conclusions_valid);
}

TEST(LlmMemoryRunnerTest, PartialAggregatesRetainEarlierMeasuredValueAndExcludeLaterInvalidValue) {
  const LlmMemoryConfig config = explicit_config();
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  size_t weights_measurements = 0;
  FakeRunnerExecutor executor;
  executor.mutate = [&](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan& task_plan,
                        const LlmRunnerTaskContext& context, size_t, LlmExecutorResult& execution) {
    if (context.kind != LlmRunnerTaskKind::Measurement || context.scenario != LlmScenario::WeightsOnly) {
      return;
    }
    ++weights_measurements;
    execution.elapsed_seconds = static_cast<double>(task_plan.effective_payload_bytes) / 100.0 / 1.0e9;
    if (weights_measurements == 2) {
      execution.valid = false;
      execution.reason_code = LlmExecutorReason::CHECKSUM_MISMATCH;
      execution.checksum_valid = false;
    }
  };
  LlmMemoryResult result;

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor.callback(), result), EXIT_FAILURE);
  EXPECT_EQ(result.status, LlmRunStatus::Failed);
  EXPECT_EQ(result.reason_code, LlmExecutorReason::CHECKSUM_MISMATCH);
  ASSERT_EQ(result.aggregates[0].effective_payload_gb_s.values.size(), 1u);
  EXPECT_NEAR(result.aggregates[0].effective_payload_gb_s.values.front(), 100.0, 1e-10);
  EXPECT_EQ(result.aggregates[0].status, "partial");
  EXPECT_EQ(result.aggregates[0].stability_quality, "insufficient-samples");
  EXPECT_EQ(result.counters.measured_measurements, 5u);
  EXPECT_EQ(result.counters.attempted_measurements, 6u);
  EXPECT_EQ(result.measurements[5].status, LlmMeasurementStatus::Invalid);
  EXPECT_FALSE(result.measurements[5].effective_payload_gb_s.has_value());
}

TEST(LlmMemoryRunnerTest, MalformedMeasurementChecksumCardinalityIsRejectedWithoutRetainedVectorsOrAggregate) {
  for (int worker_delta : {-1, 1}) {
    SCOPED_TRACE(worker_delta < 0 ? "too-few" : "too-many");
    const LlmMemoryConfig config = explicit_config(1);
    const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    ASSERT_GE(plan.effective_workers, 2u);
    FakeRunnerExecutor executor;
    executor.mutate = [worker_delta](const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan&,
                                     const LlmRunnerTaskContext& context, size_t, LlmExecutorResult& execution) {
      if (context.kind != LlmRunnerTaskKind::Measurement) {
        return;
      }
      const size_t malformed_count = static_cast<size_t>(static_cast<int>(model_plan.effective_workers) + worker_delta);
      execution.expected_checksums.resize(malformed_count);
      execution.actual_checksums.resize(malformed_count);
    };
    LlmMemoryResult result;

    EXPECT_EQ(run_llm_memory_suite(config, plan, executor.callback(), result), EXIT_FAILURE);
    EXPECT_EQ(result.status, LlmRunStatus::Failed);
    EXPECT_EQ(result.reason_code, LlmExecutorReason::INVALID_RESOURCES);
    ASSERT_FALSE(result.measurements.empty());
    EXPECT_EQ(result.measurements[0].status, LlmMeasurementStatus::Failed);
    EXPECT_EQ(result.measurements[0].reason_code, LlmExecutorReason::INVALID_RESOURCES);
    EXPECT_TRUE(result.measurements[0].execution.expected_checksums.empty());
    EXPECT_TRUE(result.measurements[0].execution.actual_checksums.empty());
    EXPECT_TRUE(result.aggregates[0].effective_payload_gb_s.values.empty());
    EXPECT_EQ(result.counters.measured_measurements, 0u);
  }
}

TEST(LlmMemoryRunnerTest, MalformedExcludedChecksumCardinalityUsesInvalidResourcesReason) {
  const LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  ASSERT_GE(plan.effective_workers, 2u);
  FakeRunnerExecutor executor;
  executor.mutate = [](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context,
                       size_t, LlmExecutorResult& execution) {
    if (context.kind == LlmRunnerTaskKind::Warmup) {
      execution.expected_checksums.resize(1);
      execution.actual_checksums.resize(1);
    }
  };
  LlmMemoryResult result;

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor.callback(), result), EXIT_FAILURE);
  EXPECT_EQ(result.status, LlmRunStatus::Failed);
  EXPECT_EQ(result.reason_code, LlmExecutorReason::INVALID_RESOURCES);
  ASSERT_EQ(result.calibration_attempts[0].size(), 1u);
  EXPECT_FALSE(result.calibration_attempts[0][0].valid);
  EXPECT_EQ(result.calibration_attempts[0][0].reason_code, LlmExecutorReason::INVALID_RESOURCES);
}

TEST(LlmMemoryRunnerTest, CalibrationFailureRetainsAttemptAndFinalizesMeasurementSlots) {
  const LlmMemoryConfig config = automatic_config(2);
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  FakeRunnerExecutor executor;
  executor.mutate = [](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context,
                       size_t, LlmExecutorResult& execution) {
    if (context.purpose == "pilot") {
      execution.valid = false;
      execution.reason_code = LlmExecutorReason::KERNEL_FAILED;
      execution.kernel_succeeded = false;
      execution.checksum_valid = false;
    }
  };
  std::vector<CheckpointRecord> checkpoints;
  LlmMemoryResult result;

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor.callback(), result, recording_checkpoints(checkpoints)),
            EXIT_FAILURE);
  ASSERT_EQ(result.calibration_attempts[0].size(), 2u);
  EXPECT_EQ(result.calibration_attempts[0][1].purpose, "pilot");
  EXPECT_FALSE(result.calibration_attempts[0][1].valid);
  EXPECT_EQ(result.calibration_attempts[0][1].reason_code, LlmExecutorReason::KERNEL_FAILED);
  EXPECT_EQ(result.counters.attempted_measurements, 0u);
  EXPECT_EQ(result.counters.terminal_measurements, 6u);
  for (const LlmMeasurementState& measurement : result.measurements) {
    EXPECT_EQ(measurement.status, LlmMeasurementStatus::Failed);
  }
  ASSERT_EQ(checkpoints.size(), 1u);
  EXPECT_EQ(checkpoints[0].kind, LlmCheckpointKind::CommandTerminal);
}

TEST(LlmMemoryRunnerTest, SuccessfulExcludedTaskWinsStopAndInterruptsBeforeMeasurements) {
  const LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  bool stop = false;
  FakeRunnerExecutor executor;
  executor.mutate = [&](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context,
                        size_t, LlmExecutorResult&) {
    if (context.kind == LlmRunnerTaskKind::Warmup) {
      stop = true;
    }
  };
  std::vector<CheckpointRecord> checkpoints;
  LlmRunnerHooks hooks = recording_checkpoints(checkpoints);
  hooks.stop_requested = [&]() { return stop; };
  LlmMemoryResult result;

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor.callback(), result, hooks), EXIT_SUCCESS);
  EXPECT_EQ(result.status, LlmRunStatus::Interrupted);
  EXPECT_TRUE(result.interruption_requested);
  ASSERT_EQ(result.calibration_attempts[0].size(), 1u);
  EXPECT_TRUE(result.calibration_attempts[0][0].terminal);
  EXPECT_TRUE(result.calibration_attempts[0][0].valid);
  EXPECT_EQ(result.calibration_attempts[0][0].purpose, "explicit-warmup");
  EXPECT_EQ(result.counters.attempted_measurements, 0u);
  EXPECT_EQ(result.counters.terminal_measurements, 3u);
  expect_interrupted_tail(result, 0);
  ASSERT_EQ(checkpoints.size(), 1u);
  EXPECT_EQ(checkpoints[0].kind, LlmCheckpointKind::CommandTerminal);
}

TEST(LlmMemoryRunnerTest, ExcludedTaskFailureWinsSimultaneousStopAndRetainsAttemptEvidence) {
  const LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  bool stop = false;
  FakeRunnerExecutor executor;
  executor.mutate = [&](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context,
                        size_t, LlmExecutorResult& execution) {
    if (context.kind == LlmRunnerTaskKind::Warmup) {
      stop = true;
      execution.valid = false;
      execution.reason_code = LlmExecutorReason::KERNEL_FAILED;
      execution.kernel_succeeded = false;
      execution.checksum_valid = false;
    }
  };
  LlmRunnerHooks hooks;
  hooks.stop_requested = [&]() { return stop; };
  LlmMemoryResult result;

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor.callback(), result, hooks), EXIT_FAILURE);
  EXPECT_EQ(result.status, LlmRunStatus::Failed);
  EXPECT_EQ(result.reason_code, LlmExecutorReason::KERNEL_FAILED);
  EXPECT_TRUE(result.interruption_requested);
  ASSERT_EQ(result.calibration_attempts[0].size(), 1u);
  EXPECT_TRUE(result.calibration_attempts[0][0].terminal);
  EXPECT_FALSE(result.calibration_attempts[0][0].valid);
  EXPECT_EQ(result.calibration_attempts[0][0].reason_code, LlmExecutorReason::KERNEL_FAILED);
  EXPECT_EQ(result.calibration_attempts[0][0].steps, config.iterations);
  EXPECT_TRUE(result.frozen_scenario_plans.valid);
  EXPECT_GT(result.counters.planned_synthetic_steps, 0u);
  EXPECT_GT(result.counters.planned_exact_payload_bytes, 0u);
  for (const LlmMeasurementState& measurement : result.measurements) {
    EXPECT_LT(measurement.frozen_plan_index, kLlmScenarioCount);
    EXPECT_EQ(measurement.planned_steps, config.iterations);
    EXPECT_GT(measurement.planned_exact_payload_bytes, 0u);
  }
  expect_interrupted_tail(result, 0);
}

TEST(LlmMemoryRunnerTest, ExcludedExecutorExceptionRetainsTerminalAttemptBeforeCommandFailure) {
  const LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  FakeRunnerExecutor executor;
  executor.mutate = [](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context,
                       size_t, LlmExecutorResult&) {
    if (context.kind == LlmRunnerTaskKind::Warmup) {
      throw std::runtime_error("excluded executor failure");
    }
  };
  std::vector<CheckpointRecord> checkpoints;
  LlmMemoryResult result;

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor.callback(), result, recording_checkpoints(checkpoints)),
            EXIT_FAILURE);
  EXPECT_EQ(result.status, LlmRunStatus::Failed);
  EXPECT_EQ(result.reason_code, LlmRunnerReason::RUNNER_EXCEPTION);
  EXPECT_EQ(result.diagnostic, "excluded executor failure");
  ASSERT_EQ(result.calibration_attempts[0].size(), 1u);
  EXPECT_TRUE(result.calibration_attempts[0][0].terminal);
  EXPECT_FALSE(result.calibration_attempts[0][0].valid);
  EXPECT_EQ(result.calibration_attempts[0][0].reason_code, LlmRunnerReason::RUNNER_EXCEPTION);
  EXPECT_EQ(result.calibration_attempts[0][0].work_plan_identity,
            build_llm_scenario_work_plan(plan, LlmScenario::WeightsOnly, config.iterations, true).plan_identity);
  ASSERT_EQ(checkpoints.size(), 1u);
  EXPECT_EQ(checkpoints[0].kind, LlmCheckpointKind::CommandTerminal);
}

TEST(LlmMemoryRunnerTest, TypedAndUnknownExecutorExceptionsRemainInsideRunnerBoundary) {
  for (bool typed : {true, false}) {
    SCOPED_TRACE(typed ? "typed" : "unknown");
    const LlmMemoryConfig config = explicit_config(1);
    const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    FakeRunnerExecutor executor;
    executor.mutate = [typed](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context,
                              size_t, LlmExecutorResult&) {
      if (context.kind == LlmRunnerTaskKind::Measurement) {
        if (typed) {
          throw std::runtime_error("injected executor failure");
        }
        throw 17;
      }
    };
    std::vector<CheckpointRecord> checkpoints;
    LlmMemoryResult result;
    EXPECT_EQ(run_llm_memory_suite(config, plan, executor.callback(), result, recording_checkpoints(checkpoints)),
              EXIT_FAILURE);
    EXPECT_EQ(result.status, LlmRunStatus::Failed);
    EXPECT_EQ(result.reason_code,
              typed ? LlmRunnerReason::RUNNER_EXCEPTION : LlmRunnerReason::RUNNER_UNKNOWN_EXCEPTION);
    if (typed) {
      EXPECT_EQ(result.diagnostic, "injected executor failure");
    }
    EXPECT_EQ(result.counters.attempted_measurements, 1u);
    EXPECT_EQ(result.counters.terminal_measurements, 3u);
    for (const LlmMeasurementState& measurement : result.measurements) {
      EXPECT_EQ(measurement.status, LlmMeasurementStatus::Failed);
    }
    ASSERT_EQ(checkpoints.size(), 2u);
    EXPECT_EQ(checkpoints[0].kind, LlmCheckpointKind::MeasurementTerminal);
    EXPECT_EQ(checkpoints[0].attempted_measurements, 1u);
    EXPECT_EQ(checkpoints[0].terminal_measurements, 3u);
    EXPECT_EQ(checkpoints[0].measured_measurements, 0u);
    EXPECT_EQ(checkpoints[1].kind, LlmCheckpointKind::CommandTerminal);
    EXPECT_EQ(result.logical_checkpoint_attempts, 2u);
    EXPECT_EQ(result.successful_logical_checkpoints, 2u);
    EXPECT_TRUE(result.terminal_checkpoint_completed);
  }
}

TEST(LlmMemoryRunnerTest, ExecutorExceptionStillObservesStopRaisedByItsMeasurementCheckpoint) {
  const LlmMemoryConfig config = explicit_config(2);
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  bool stop = false;
  FakeRunnerExecutor executor;
  executor.mutate = [](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context,
                       size_t, LlmExecutorResult&) {
    if (context.kind == LlmRunnerTaskKind::Measurement) {
      throw std::runtime_error("injected executor failure");
    }
  };
  std::vector<LlmCheckpointKind> kinds;
  LlmRunnerHooks hooks;
  hooks.stop_requested = [&]() { return stop; };
  hooks.checkpoint = [&](const LlmMemoryResult&, LlmCheckpointKind kind) {
    kinds.push_back(kind);
    if (kind == LlmCheckpointKind::MeasurementTerminal) {
      stop = true;
    }
    return EXIT_SUCCESS;
  };
  LlmMemoryResult result;

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor.callback(), result, hooks), EXIT_FAILURE);
  EXPECT_EQ(kinds, (std::vector<LlmCheckpointKind>{LlmCheckpointKind::MeasurementTerminal,
                                                   LlmCheckpointKind::CommandTerminal}));
  EXPECT_EQ(result.status, LlmRunStatus::Failed);
  EXPECT_EQ(result.reason_code, LlmRunnerReason::RUNNER_EXCEPTION);
  EXPECT_TRUE(result.interruption_requested);
  EXPECT_EQ(result.measurements[0].status, LlmMeasurementStatus::Failed);
  for (size_t index = 1; index < result.measurements.size(); ++index) {
    EXPECT_EQ(result.measurements[index].status, LlmMeasurementStatus::Interrupted) << index;
  }
}

TEST(LlmMemoryRunnerTest, ExplicitLaterScenarioCapFailsPreflightBeforeAnyExecutorOrCheckpoint) {
  LlmMemoryConfig config = explicit_config(1, 1);
  config.layer_count = 32;
  config.query_head_count = 8;
  config.kv_head_count = 8;
  config.head_dimension = 64;
  config.kv_element_bytes = 2;
  config.visible_context_tokens = 4096;
  const LlmMemoryWorkPlan initial_plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(initial_plan.valid) << initial_plan.reason_code;
  const LlmScenarioLimits weights_limits =
      calculate_llm_scenario_limits(initial_plan.geometry, LlmScenario::WeightsOnly);
  const LlmScenarioLimits kv_limits = calculate_llm_scenario_limits(initial_plan.geometry, LlmScenario::KvOnly);
  ASSERT_TRUE(weights_limits.valid);
  ASSERT_TRUE(kv_limits.valid);
  ASSERT_LT(kv_limits.effective_maximum_steps, weights_limits.effective_maximum_steps);
  config.iterations = kv_limits.effective_maximum_steps + 1;
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  EXPECT_TRUE(build_llm_scenario_work_plan(plan, LlmScenario::WeightsOnly, config.iterations, true).valid);
  const LlmScenarioWorkPlan invalid_kv =
      build_llm_scenario_work_plan(plan, LlmScenario::KvOnly, config.iterations, true);
  ASSERT_FALSE(invalid_kv.valid);
  EXPECT_EQ(invalid_kv.reason_code, LlmWorkPlanReason::EXACT_PAYLOAD_CAP_EXCEEDED);
  FakeRunnerExecutor executor;
  size_t checkpoint_calls = 0;
  LlmRunnerHooks hooks;
  hooks.stop_requested = []() { return false; };
  hooks.checkpoint = [&](const LlmMemoryResult&, LlmCheckpointKind) {
    ++checkpoint_calls;
    return EXIT_SUCCESS;
  };
  LlmMemoryResult result;

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor.callback(), result, hooks), EXIT_FAILURE);
  EXPECT_FALSE(result.initialized);
  EXPECT_EQ(result.reason_code, LlmWorkPlanReason::EXACT_PAYLOAD_CAP_EXCEEDED);
  EXPECT_TRUE(executor.calls.empty());
  EXPECT_EQ(checkpoint_calls, 0u);
}

TEST(LlmMemoryRunnerTest, CounterAndAuxiliaryOverflowFailBeforeInitializationOrExecutorCall) {
  LlmMemoryConfig config = explicit_config();
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  config.loop_count = std::numeric_limits<size_t>::max();
  const LlmRunnerAuxiliaryEstimate estimate = calculate_llm_runner_auxiliary_estimate(config, plan);
  EXPECT_FALSE(estimate.valid);
  EXPECT_EQ(estimate.reason_code, LlmRunnerReason::AUXILIARY_BYTES_OVERFLOW);
  FakeRunnerExecutor executor;
  LlmMemoryResult result;

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor.callback(), result), EXIT_FAILURE);
  EXPECT_FALSE(result.initialized);
  EXPECT_EQ(result.reason_code, LlmRunnerReason::AUXILIARY_BYTES_OVERFLOW);
  EXPECT_TRUE(executor.calls.empty());
  EXPECT_EQ(result.logical_checkpoint_attempts, 0u);
}

TEST(LlmMemoryRunnerTest, RunnerAuxiliaryEstimateHasExactCategoryBreakdownForExplicitAndAutomaticModes) {
  std::array<LlmRunnerAuxiliaryEstimate, 2> estimates;
  for (size_t mode = 0; mode < estimates.size(); ++mode) {
    const LlmMemoryConfig config = mode == 0 ? explicit_config(2) : automatic_config(2);
    const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    estimates[mode] = calculate_llm_runner_auxiliary_estimate(config, plan);
    const LlmRunnerAuxiliaryEstimate& estimate = estimates[mode];
    ASSERT_TRUE(estimate.valid) << estimate.reason_code;
    const size_t planned_measurements = config.loop_count * kLlmScenarioCount;
    const size_t attempts_per_scenario =
        config.user_specified_iterations ? 1 : 4 + Constants::LLM_CALIBRATION_MAX_CORRECTIONS;
    EXPECT_EQ(estimate.measurement_record_bytes, planned_measurements * sizeof(LlmMeasurementState));
    EXPECT_EQ(estimate.loop_record_bytes, config.loop_count * sizeof(LlmLoopRecord));
    EXPECT_EQ(estimate.calibration_record_bytes,
              attempts_per_scenario * kLlmScenarioCount * sizeof(LlmCalibrationAttempt));
    EXPECT_GT(estimate.calibration_identity_bytes, 0u);
    EXPECT_EQ(estimate.aggregate_value_bytes, planned_measurements * 3 * sizeof(double));
    EXPECT_EQ(estimate.statistics_workspace_bytes, config.loop_count * 2 * sizeof(double));
    EXPECT_EQ(estimate.warning_record_bytes, (kLlmScenarioCount + 1) * sizeof(std::string_view));
    EXPECT_GT(estimate.fixed_metadata_bytes, 0u);
    EXPECT_EQ(estimate.retained_checksum_bytes,
              planned_measurements * plan.effective_workers * 2 * sizeof(LlmWorkerChecksum));
    EXPECT_EQ(estimate.checksum_auxiliary_bytes, estimate.retained_checksum_bytes);
    EXPECT_EQ(estimate.orchestration_auxiliary_bytes,
              estimate.measurement_record_bytes + estimate.loop_record_bytes + estimate.calibration_record_bytes +
                  estimate.calibration_identity_bytes + estimate.aggregate_value_bytes +
                  estimate.statistics_workspace_bytes + estimate.warning_record_bytes + estimate.fixed_metadata_bytes);
    EXPECT_EQ(estimate.total_auxiliary_bytes,
              estimate.checksum_auxiliary_bytes + estimate.orchestration_auxiliary_bytes);
  }
  EXPECT_EQ(estimates[1].calibration_record_bytes,
            estimates[0].calibration_record_bytes * (4 + Constants::LLM_CALIBRATION_MAX_CORRECTIONS));
  EXPECT_EQ(estimates[1].calibration_identity_bytes,
            estimates[0].calibration_identity_bytes * (4 + Constants::LLM_CALIBRATION_MAX_CORRECTIONS));
}

TEST(LlmMemoryRunnerTest, RunnerAuxiliaryBudgetAcceptsExactBoundaryAndRejectsOneByteLess) {
  const LlmMemoryConfig config = explicit_config();
  LlmMemoryWorkPlanRequest request = plan_request(config);
  const LlmMemoryWorkPlan preliminary = build_llm_memory_work_plan(request);
  ASSERT_TRUE(preliminary.valid) << preliminary.reason_code;
  const LlmExecutorAuxiliaryEstimate executor_auxiliary = calculate_llm_executor_auxiliary_estimate(preliminary);
  const LlmRunnerAuxiliaryEstimate runner_auxiliary = calculate_llm_runner_auxiliary_estimate(config, preliminary);
  ASSERT_TRUE(executor_auxiliary.valid);
  ASSERT_TRUE(runner_auxiliary.valid);
  ASSERT_TRUE(NumericUtils::checked_add(executor_auxiliary.checksum_auxiliary_bytes,
                                        runner_auxiliary.checksum_auxiliary_bytes, request.checksum_auxiliary_bytes));
  ASSERT_TRUE(NumericUtils::checked_add(executor_auxiliary.orchestration_auxiliary_bytes,
                                        runner_auxiliary.orchestration_auxiliary_bytes,
                                        request.orchestration_auxiliary_bytes));
  const LlmMemoryWorkPlan exact = build_llm_memory_work_plan(request);
  ASSERT_TRUE(exact.valid) << exact.reason_code;
  FakeRunnerExecutor exact_executor;
  LlmMemoryResult exact_result;
  EXPECT_EQ(run_llm_memory_suite(config, exact, exact_executor.callback(), exact_result), EXIT_SUCCESS);
  EXPECT_TRUE(exact_result.results_complete);

  ASSERT_GT(request.orchestration_auxiliary_bytes, 0u);
  ASSERT_GT(request.checksum_auxiliary_bytes, 0u);
  for (bool checksum_short : {false, true}) {
    SCOPED_TRACE(checksum_short ? "checksum" : "orchestration");
    LlmMemoryWorkPlanRequest short_request = request;
    if (checksum_short) {
      --short_request.checksum_auxiliary_bytes;
    } else {
      --short_request.orchestration_auxiliary_bytes;
    }
    const LlmMemoryWorkPlan one_byte_short = build_llm_memory_work_plan(short_request);
    ASSERT_TRUE(one_byte_short.valid) << one_byte_short.reason_code;
    FakeRunnerExecutor rejected_executor;
    LlmMemoryResult rejected_result;
    EXPECT_EQ(run_llm_memory_suite(config, one_byte_short, rejected_executor.callback(), rejected_result),
              EXIT_FAILURE);
    EXPECT_EQ(rejected_result.reason_code, LlmRunnerReason::AUXILIARY_BUDGET_INSUFFICIENT);
    EXPECT_FALSE(rejected_result.initialized);
    EXPECT_TRUE(rejected_executor.calls.empty());
  }
}

TEST(LlmMemoryRunnerTest, StableTaskAndCheckpointTokensCoverUnknownValues) {
  EXPECT_STREQ(llm_runner_task_kind_to_string(LlmRunnerTaskKind::Warmup), "warmup");
  EXPECT_STREQ(llm_runner_task_kind_to_string(LlmRunnerTaskKind::Calibration), "calibration");
  EXPECT_STREQ(llm_runner_task_kind_to_string(LlmRunnerTaskKind::Measurement), "measurement");
  EXPECT_STREQ(llm_runner_task_kind_to_string(static_cast<LlmRunnerTaskKind>(99)), "unknown");
  EXPECT_STREQ(llm_checkpoint_kind_to_string(LlmCheckpointKind::MeasurementTerminal), "measurement-terminal");
  EXPECT_STREQ(llm_checkpoint_kind_to_string(LlmCheckpointKind::CommandTerminal), "command-terminal");
  EXPECT_STREQ(llm_checkpoint_kind_to_string(static_cast<LlmCheckpointKind>(99)), "unknown");
}

}  // namespace

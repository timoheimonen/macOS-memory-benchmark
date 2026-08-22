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
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "core/config/constants.h"
#include "llm_memory/llm_cpu_backend.h"
#include "llm_memory/llm_runner.h"
#include "utils/numeric_utils.h"

namespace {

struct TaskRecord {
  LlmRunnerTaskContext context;
  size_t work_units = 0;
  size_t payload_bytes = 0;
  std::string plan_identity;
  LlmWorkUnitKind work_unit_kind = LlmWorkUnitKind::DecodeStep;
  LlmKvWriteKind kv_write_kind = LlmKvWriteKind::None;
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
  config.user_specified_context_tokens = true;
  return config;
}

LlmMemoryConfig automatic_config(size_t loop_count = 1) {
  LlmMemoryConfig config = explicit_config(loop_count);
  config.iterations = 0;
  config.user_specified_iterations = false;
  return config;
}

LlmMemoryConfig explicit_prefill_config(size_t loop_count = 2,
                                        size_t iterations = 3) {
  LlmMemoryConfig config = explicit_config(loop_count, iterations);
  config.phase = LlmPhase::Prefill;
  config.visible_context_tokens = 0;
  config.prompt_tokens = 5;
  config.attention_query_tile_tokens = 2;
  config.user_specified_context_tokens = false;
  config.user_specified_prompt_tokens = true;
  config.user_specified_attention_query_tile_tokens = true;
  return config;
}

LlmMemoryConfig automatic_prefill_config(size_t loop_count = 1) {
  LlmMemoryConfig config = explicit_prefill_config(loop_count);
  config.iterations = 0;
  config.user_specified_iterations = false;
  return config;
}

LlmMemoryWorkPlanRequest plan_request(const LlmMemoryConfig& config) {
  LlmMemoryWorkPlanRequest request;
  request.backend = config.backend;
  request.geometry.phase = config.phase;
  request.geometry.kv_layout = config.kv_layout;
  request.geometry.kv_block_tokens = config.kv_block_tokens;
  request.geometry.active_weight_bytes = config.weight_size_mb * Constants::BYTES_PER_MB;
  request.geometry.layer_count = config.layer_count;
  request.geometry.query_head_count = config.query_head_count;
  request.geometry.kv_head_count = config.kv_head_count;
  request.geometry.head_dimension = config.head_dimension;
  request.geometry.kv_element_bytes = config.kv_element_bytes;
  request.geometry.visible_context_tokens = config.visible_context_tokens;
  request.geometry.prompt_tokens = config.prompt_tokens;
  request.geometry.attention_query_tile_tokens =
      config.attention_query_tile_tokens;
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
  LlmMemoryWorkPlanDraft draft = prepare_llm_memory_work_plan(request);
  if (!draft.valid) {
    return finalize_llm_memory_work_plan(std::move(draft), 0, 0);
  }
  const LlmExecutorAuxiliaryEstimate executor =
      calculate_llm_executor_auxiliary_estimate(
          draft.auxiliary_preflight);
  const LlmRunnerAuxiliaryEstimate runner =
      calculate_llm_runner_auxiliary_estimate(
          config, draft.auxiliary_preflight);
  if (!executor.valid || !runner.valid ||
      !NumericUtils::checked_add(executor.checksum_auxiliary_bytes, runner.checksum_auxiliary_bytes,
                                 request.checksum_auxiliary_bytes) ||
      !NumericUtils::checked_add(executor.orchestration_auxiliary_bytes, runner.orchestration_auxiliary_bytes,
                                 request.orchestration_auxiliary_bytes)) {
    return finalize_llm_memory_work_plan(std::move(draft), 0, 0);
  }
  return finalize_llm_memory_work_plan(
      std::move(draft), request.checksum_auxiliary_bytes,
      request.orchestration_auxiliary_bytes);
}

LlmTaskExecutionResult successful_execution(const LlmMemoryWorkPlan& model_plan,
                                             const LlmScenarioWorkPlan& task_plan,
                                             const LlmRunnerTaskContext& context,
                                             double elapsed_seconds) {
  LlmTaskExecutionResult result;
  result.status = LlmTaskExecutionStatus::Complete;
  result.reason_code = LlmBackendReason::VALID;
  result.identity.backend = model_plan.backend;
  result.identity.phase = model_plan.phase;
  result.identity.kv_layout = model_plan.kv_layout;
  result.identity.work_unit_kind = task_plan.work_unit_kind;
  result.identity.kv_write_kind = task_plan.kv_write_kind;
  result.identity.task_kind = context.kind;
  result.identity.scenario = context.scenario;
  result.identity.attempt_index = context.attempt_index;
  result.identity.loop_index = context.loop_index;
  result.identity.order_position = context.order_position;
  result.identity.purpose = context.purpose;
  result.identity.model_plan_identity = model_plan.plan_identity;
  result.identity.scenario_plan_identity = task_plan.plan_identity;
  result.timing.evaluated = true;
  result.timing.valid = true;
  result.timing.elapsed_seconds = elapsed_seconds;
  result.completion.planned_work_units = task_plan.work_units;
  result.completion.completed_work_units = task_plan.work_units;
  result.completion.completed_effective_model_payload_bytes =
      task_plan.effective_model_payload_bytes;
  result.completion.completed_layout_metadata_lookup_count =
      task_plan.layout_metadata_lookup_count;
  result.completion.completed_layout_metadata_read_bytes =
      task_plan.layout_metadata_read_bytes;
  result.completion.completed_task_accounted_bytes =
      task_plan.task_accounted_bytes;
  result.validation.evaluated = true;
  result.validation.valid = true;
  return result;
}

void fail_execution(LlmTaskExecutionResult& execution) {
  execution.status = LlmTaskExecutionStatus::Failed;
  execution.reason_code = LlmBackendReason::RESOURCES_NOT_PREPARED;
}

void invalidate_execution(LlmTaskExecutionResult& execution) {
  execution.validation.valid = false;
}

class FakeLlmBackend final : public LlmBackend {
 public:
  std::vector<TaskRecord> calls;
  std::function<void(const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext&, size_t,
                     LlmTaskExecutionResult&)>
      mutate;
  LlmBackendLifecycleResult initialization = {LlmBackendStatus::Ready,
                                               LlmBackendReason::VALID};
  LlmBackendLifecycleResult plan_resolution = {LlmBackendStatus::Ready,
                                               LlmBackendReason::VALID};
  LlmBackendLifecycleResult preparation = {LlmBackendStatus::Ready,
                                           LlmBackendReason::VALID};
  LlmBackendLifecycleResult release = {LlmBackendStatus::Ready,
                                       LlmBackendReason::VALID};
  size_t initialize_calls = 0;
  size_t plan_resolution_calls = 0;
  size_t preparation_calls = 0;
  size_t release_calls = 0;
  bool clear_initialization_reason_on_release = false;
  std::string initialization_reason_storage;

  FakeLlmBackend() { evidence_.backend = LlmMemoryBackend::Cpu; }

  LlmMemoryBackend kind() const noexcept override {
    return LlmMemoryBackend::Cpu;
  }

  LlmBackendAuxiliaryEstimate calculate_auxiliary_estimate(
      const LlmMemoryWorkPlan& model_plan) const noexcept override {
    if (model_plan.phase == LlmPhase::Prefill) {
      LlmBackendAuxiliaryEstimate estimate;
      estimate.valid = true;
      estimate.reason_code = LlmBackendReason::VALID;
      return estimate;
    }
    const LlmExecutorAuxiliaryEstimate cpu =
        calculate_llm_executor_auxiliary_estimate(model_plan);
    LlmBackendAuxiliaryEstimate estimate;
    estimate.valid = cpu.valid;
    estimate.reason_code = cpu.reason_code;
    estimate.checksum_auxiliary_bytes = cpu.checksum_auxiliary_bytes;
    estimate.orchestration_auxiliary_bytes = cpu.orchestration_auxiliary_bytes;
    estimate.total_auxiliary_bytes = cpu.total_auxiliary_bytes;
    return estimate;
  }

  LlmBackendLifecycleResult initialize(
      const LlmMemoryConfig&) noexcept override {
    ++initialize_calls;
    evidence_.initialization = initialization;
    return initialization;
  }

  LlmBackendLifecycleResult resolve_execution_plan(
      const LlmMemoryWorkPlan&) noexcept override {
    ++plan_resolution_calls;
    evidence_.plan_resolution = plan_resolution;
    return plan_resolution;
  }

  LlmBackendLifecycleResult prepare_resources(
      const LlmMemoryWorkPlan&) noexcept override {
    ++preparation_calls;
    evidence_.preparation = preparation;
    return preparation;
  }

  LlmTaskExecutionResult execute_task(
      const LlmMemoryWorkPlan& model_plan,
      const LlmScenarioWorkPlan& task_plan,
      const LlmRunnerTaskContext& context) override {
    calls.push_back({context, task_plan.work_units,
                     task_plan.effective_model_payload_bytes,
                     task_plan.plan_identity, task_plan.work_unit_kind,
                     task_plan.kv_write_kind});
    LlmTaskExecutionResult result =
        successful_execution(model_plan, task_plan, context, 0.150);
    if (mutate) {
      mutate(model_plan, task_plan, context, calls.size(), result);
    }
    return result;
  }

  const LlmBackendEvidence& evidence() const noexcept override {
    return evidence_;
  }

  LlmBackendLifecycleResult release_resources() noexcept override {
    ++release_calls;
    if (clear_initialization_reason_on_release) {
      initialization_reason_storage.clear();
    }
    evidence_.release = release;
    return release;
  }

 private:
  LlmBackendEvidence evidence_;
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
      EXPECT_EQ(measurement.completed_work_units, 0u) << index;
    }
  }
}

TEST(LlmMemoryRunnerTest,
     ExplicitFreezeWarmsCanonicalScenariosThenCompletesCyclicRun) {
  const LlmMemoryConfig config = explicit_config();
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  FakeLlmBackend executor;
  std::vector<CheckpointRecord> checkpoints;
  std::vector<bool> release_completed_at_checkpoint;
  LlmRunnerHooks hooks;
  hooks.stop_requested = []() { return false; };
  hooks.checkpoint = [&](const LlmMemoryResult& snapshot,
                         LlmCheckpointKind kind) {
    checkpoints.push_back(capture_checkpoint(snapshot, kind));
    release_completed_at_checkpoint.push_back(
        executor.evidence().release.status == LlmBackendStatus::Ready);
    return EXIT_SUCCESS;
  };
  LlmMemoryResult result;

  ASSERT_EQ(run_llm_memory_suite(config, plan, executor, result, hooks),
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
    EXPECT_EQ(warmup.context.purpose, "frozen_measurement_warmup");
    EXPECT_EQ(warmup.context.scenario, static_cast<LlmScenario>(index));
    EXPECT_EQ(warmup.work_units, config.iterations);
    EXPECT_EQ(warmup.work_units, frozen.work_units);
    EXPECT_EQ(warmup.payload_bytes, frozen.effective_model_payload_bytes);
    EXPECT_EQ(warmup.plan_identity, frozen.plan_identity);
    const std::vector<TaskRecord> measured = records_for_scenario(executor.calls, warmup.context.scenario, true);
    ASSERT_EQ(measured.size(), config.loop_count);
    for (const TaskRecord& record : measured) {
      EXPECT_EQ(record.work_units, frozen.work_units);
      EXPECT_EQ(record.payload_bytes, frozen.effective_model_payload_bytes);
      EXPECT_EQ(record.plan_identity, frozen.plan_identity);
    }
  }

  size_t call_index = kLlmScenarioCount;
  for (size_t loop_index = 0; loop_index < config.loop_count; ++loop_index) {
    const std::array<LlmScenario, kLlmScenarioCount> expected_order =
        build_llm_scenario_order(loop_index);
    for (size_t position = 0; position < kLlmScenarioCount; ++position) {
      ASSERT_LT(call_index, executor.calls.size());
      const TaskRecord& measurement = executor.calls[call_index++];
      EXPECT_EQ(measurement.context.kind,
                LlmRunnerTaskKind::Measurement);
      EXPECT_EQ(measurement.context.purpose, "measurement");
      EXPECT_EQ(measurement.context.scenario, expected_order[position]);
      EXPECT_EQ(measurement.context.loop_index, loop_index);
      EXPECT_EQ(measurement.context.order_position, position);
      const LlmScenarioWorkPlan& frozen =
          result.frozen_scenario_plans
              .scenarios[static_cast<size_t>(expected_order[position])];
      EXPECT_EQ(measurement.work_units, frozen.work_units);
      EXPECT_EQ(measurement.payload_bytes,
                frozen.effective_model_payload_bytes);
      EXPECT_EQ(measurement.plan_identity, frozen.plan_identity);
    }
  }
  EXPECT_EQ(call_index, executor.calls.size());

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
  EXPECT_EQ(result.counters.planned_work_units, 36u);
  EXPECT_EQ(result.counters.completed_work_units, 36u);
  EXPECT_EQ(result.counters.planned_layout_metadata_lookup_count, 0u);
  EXPECT_EQ(result.counters.completed_layout_metadata_lookup_count, 0u);
  EXPECT_EQ(result.counters.planned_layout_metadata_read_bytes, 0u);
  EXPECT_EQ(result.counters.completed_layout_metadata_read_bytes, 0u);
  size_t expected_payload_per_loop = 0;
  for (const LlmScenarioWorkPlan& scenario : result.frozen_scenario_plans.scenarios) {
    EXPECT_EQ(scenario.work_unit_kind, LlmWorkUnitKind::DecodeStep);
    EXPECT_EQ(scenario.kv_write_kind,
              scenario.scenario == LlmScenario::WeightsOnly
                  ? LlmKvWriteKind::None
                  : LlmKvWriteKind::CurrentTokenAppend);
    EXPECT_EQ(scenario.layout_metadata_lookup_count_per_work_unit, 0u);
    EXPECT_EQ(scenario.layout_metadata_read_bytes_per_work_unit, 0u);
    EXPECT_EQ(scenario.accounted_bytes_per_work_unit,
              scenario.effective_model_payload_bytes_per_work_unit);
    EXPECT_EQ(scenario.layout_metadata_lookup_count, 0u);
    EXPECT_EQ(scenario.layout_metadata_read_bytes, 0u);
    EXPECT_EQ(scenario.task_accounted_bytes,
              scenario.effective_model_payload_bytes);
    expected_payload_per_loop += scenario.effective_model_payload_bytes;
  }
  EXPECT_EQ(result.counters.planned_effective_model_payload_bytes, expected_payload_per_loop * 3);
  EXPECT_EQ(result.counters.completed_effective_model_payload_bytes, expected_payload_per_loop * 3);
  EXPECT_EQ(result.counters.planned_task_accounted_bytes,
            expected_payload_per_loop * 3);
  EXPECT_EQ(result.counters.completed_task_accounted_bytes,
            expected_payload_per_loop * 3);
  for (const LlmMeasurementState& measurement : result.measurements) {
    ASSERT_LT(measurement.frozen_plan_index, kLlmScenarioCount);
    const LlmScenarioWorkPlan& frozen = result.frozen_scenario_plans.scenarios[measurement.frozen_plan_index];
    EXPECT_EQ(measurement.scenario, frozen.scenario);
    EXPECT_EQ(measurement.work_unit_kind, frozen.work_unit_kind);
    EXPECT_EQ(measurement.kv_write_kind, frozen.kv_write_kind);
    EXPECT_EQ(measurement.planned_work_units, frozen.work_units);
    EXPECT_EQ(measurement.planned_effective_model_payload_bytes, frozen.effective_model_payload_bytes);
    EXPECT_EQ(measurement.completed_effective_model_payload_bytes,
              frozen.effective_model_payload_bytes);
    EXPECT_EQ(measurement.layout_metadata_lookup_count_per_work_unit, 0u);
    EXPECT_EQ(measurement.layout_metadata_read_bytes_per_work_unit, 0u);
    EXPECT_EQ(measurement.accounted_bytes_per_work_unit,
              frozen.effective_model_payload_bytes_per_work_unit);
    EXPECT_EQ(measurement.planned_task_accounted_bytes,
              frozen.effective_model_payload_bytes);
    EXPECT_EQ(measurement.completed_task_accounted_bytes,
              frozen.effective_model_payload_bytes);
    EXPECT_EQ(measurement.execution.status, LlmTaskExecutionStatus::Complete);
    EXPECT_EQ(measurement.execution.reason_code, LlmBackendReason::VALID);
    EXPECT_TRUE(measurement.execution.timing.valid);
    EXPECT_TRUE(measurement.execution.validation.valid);
    EXPECT_TRUE(std::holds_alternative<std::monostate>(
        measurement.execution.backend_evidence));
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
  EXPECT_EQ(executor.release_calls, 1u);
  ASSERT_EQ(release_completed_at_checkpoint.size(), checkpoints.size());
  for (size_t index = 0; index + 1 < release_completed_at_checkpoint.size();
       ++index) {
    EXPECT_FALSE(release_completed_at_checkpoint[index]);
  }
  EXPECT_TRUE(release_completed_at_checkpoint.back());

  for (const LlmScenarioAggregate& aggregate : result.aggregates) {
    EXPECT_EQ(aggregate.status, "complete");
    EXPECT_EQ(aggregate.stability_quality, "stable");
    EXPECT_EQ(aggregate.effective_model_payload_gb_s.values.size(), 3u);
    EXPECT_TRUE(aggregate.effective_model_payload_gb_s.headline.has_value());
  }
}

TEST(LlmMemoryRunnerTest, CompleteSingleLoopIsInspectableButComparativeConclusionsAreInvalid) {
  const LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  FakeLlmBackend executor;
  LlmMemoryResult result;

  ASSERT_EQ(run_llm_memory_suite(config, plan, executor, result), EXIT_SUCCESS);
  EXPECT_EQ(result.status, LlmRunStatus::Complete);
  EXPECT_TRUE(result.results_complete);
  EXPECT_FALSE(result.scenario_order_balance_complete);
  EXPECT_FALSE(result.conclusions_valid);
  EXPECT_EQ(result.quality_warnings, (std::vector<std::string_view>{"scenario-order-not-balanced"}));
  EXPECT_EQ(result.logical_checkpoint_attempts, 4u);
  for (const LlmScenarioAggregate& aggregate : result.aggregates) {
    EXPECT_EQ(aggregate.stability_quality, "insufficient-samples");
    ASSERT_EQ(aggregate.effective_model_payload_gb_s.values.size(), 1u);
    EXPECT_EQ(aggregate.effective_model_payload_gb_s.headline, aggregate.effective_model_payload_gb_s.values.front());
  }
}

TEST(LlmMemoryRunnerTest, InitializationUnsupportedAndFailedDoNotExecuteOrFallback) {
  struct Case {
    LlmBackendStatus backend_status;
    LlmRunStatus run_status;
    const char* reason_code;
  };
  constexpr std::array<Case, 2> cases = {{
      {LlmBackendStatus::Unsupported, LlmRunStatus::Unsupported,
       LlmBackendReason::TASK_UNSUPPORTED},
      {LlmBackendStatus::Failed, LlmRunStatus::NotStarted,
       LlmBackendReason::BACKEND_INITIALIZATION_FAILED},
  }};

  const LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  for (const Case& test_case : cases) {
    SCOPED_TRACE(test_case.reason_code);
    FakeLlmBackend backend;
    backend.initialization = {test_case.backend_status,
                              test_case.reason_code};
    LlmMemoryResult result;

    EXPECT_EQ(run_llm_memory_suite(config, plan, backend, result),
              EXIT_FAILURE);
    EXPECT_EQ(result.initialized,
              test_case.backend_status == LlmBackendStatus::Unsupported);
    EXPECT_EQ(result.status, test_case.run_status);
    EXPECT_EQ(result.reason_code, test_case.reason_code);
    EXPECT_EQ(backend.kind(), LlmMemoryBackend::Cpu);
    EXPECT_EQ(backend.initialize_calls, 1u);
    EXPECT_EQ(backend.plan_resolution_calls, 0u);
    EXPECT_EQ(backend.preparation_calls, 0u);
    EXPECT_TRUE(backend.calls.empty());
    EXPECT_EQ(backend.release_calls, 1u);
  }
}

TEST(LlmMemoryRunnerTest,
     UnsupportedLifecycleIsTerminalAtEveryPhaseAndCanonicalBeforeRelease) {
  enum class UnsupportedPhase { Initialization, PlanResolution, Preparation };
  constexpr std::array<UnsupportedPhase, 3> phases = {
      UnsupportedPhase::Initialization, UnsupportedPhase::PlanResolution,
      UnsupportedPhase::Preparation};
  const LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;

  for (UnsupportedPhase phase : phases) {
    FakeLlmBackend backend;
    backend.initialization_reason_storage = "backend-private-unsupported";
    const LlmBackendLifecycleResult unsupported = {
        LlmBackendStatus::Unsupported,
        backend.initialization_reason_storage};
    if (phase == UnsupportedPhase::Initialization) {
      backend.initialization = unsupported;
      backend.clear_initialization_reason_on_release = true;
    } else if (phase == UnsupportedPhase::PlanResolution) {
      backend.plan_resolution = unsupported;
    } else {
      backend.preparation = unsupported;
    }
    LlmMemoryResult result;

    EXPECT_EQ(run_llm_memory_suite(config, plan, backend, result),
              EXIT_FAILURE);
    EXPECT_EQ(result.status, LlmRunStatus::Unsupported);
    EXPECT_EQ(result.reason_code, LlmBackendReason::TASK_UNSUPPORTED);
    EXPECT_TRUE(backend.calls.empty());
    EXPECT_EQ(backend.initialize_calls, 1u);
    EXPECT_EQ(backend.plan_resolution_calls,
              phase == UnsupportedPhase::Initialization ? 0u : 1u);
    EXPECT_EQ(backend.preparation_calls,
              phase == UnsupportedPhase::Preparation ? 1u : 0u);
    EXPECT_EQ(backend.release_calls, 1u);
  }
}

TEST(LlmMemoryRunnerTest,
     LifecycleFailuresBeforeTasksRemainUninitializedAndDoNotCheckpoint) {
  enum class FailedPhase { Initialization, PlanResolution, Preparation };
  constexpr std::array<FailedPhase, 3> phases = {
      FailedPhase::Initialization, FailedPhase::PlanResolution,
      FailedPhase::Preparation};
  const LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;

  for (FailedPhase phase : phases) {
    FakeLlmBackend backend;
    const LlmBackendLifecycleResult failure = {
        LlmBackendStatus::Failed,
        LlmBackendReason::BACKEND_INITIALIZATION_FAILED};
    if (phase == FailedPhase::Initialization) {
      backend.initialization = failure;
    } else if (phase == FailedPhase::PlanResolution) {
      backend.plan_resolution = failure;
    } else {
      backend.preparation = failure;
    }
    size_t checkpoint_calls = 0;
    LlmRunnerHooks hooks;
    hooks.checkpoint = [&](const LlmMemoryResult&, LlmCheckpointKind) {
      ++checkpoint_calls;
      return EXIT_SUCCESS;
    };
    LlmMemoryResult result;

    EXPECT_EQ(run_llm_memory_suite(config, plan, backend, result, hooks),
              EXIT_FAILURE);
    EXPECT_FALSE(result.initialized);
    EXPECT_EQ(result.status, LlmRunStatus::NotStarted);
    EXPECT_EQ(result.reason_code,
              LlmBackendReason::BACKEND_INITIALIZATION_FAILED);
    EXPECT_TRUE(backend.calls.empty());
    EXPECT_EQ(checkpoint_calls, 0u);
    EXPECT_EQ(backend.initialize_calls, 1u);
    EXPECT_EQ(backend.plan_resolution_calls,
              phase == FailedPhase::Initialization ? 0u : 1u);
    EXPECT_EQ(backend.preparation_calls,
              phase == FailedPhase::Preparation ? 1u : 0u);
    EXPECT_EQ(backend.release_calls, 1u);
  }
}

TEST(LlmMemoryRunnerTest, OversizedGenericReasonIsCanonicalizedIntoFrozenRunnerCapacities) {
  const LlmMemoryConfig config = explicit_config();
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  constexpr size_t oversized_capacity = 4096;
  FakeLlmBackend executor;
  executor.mutate = [](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context,
                       size_t, LlmTaskExecutionResult& execution) {
    if (context.kind != LlmRunnerTaskKind::Measurement) {
      return;
    }
    execution.reason_code.reserve(oversized_capacity);
  };
  LlmMemoryResult result;

  ASSERT_EQ(run_llm_memory_suite(config, plan, executor, result), EXIT_SUCCESS);
  EXPECT_TRUE(result.results_complete);
  for (const LlmMeasurementState& measurement : result.measurements) {
    EXPECT_EQ(measurement.execution.reason_code, LlmBackendReason::VALID);
    EXPECT_LT(measurement.execution.reason_code.capacity(), oversized_capacity);
    EXPECT_TRUE(std::holds_alternative<std::monostate>(
        measurement.execution.backend_evidence));
  }
  EXPECT_EQ(result.measurements.capacity(), result.counters.planned_measurements);
  EXPECT_EQ(result.loops.capacity(), config.loop_count);
  EXPECT_EQ(result.statistics_workspace.sorted_values.capacity(), config.loop_count);
  EXPECT_EQ(result.statistics_workspace.absolute_deviations.capacity(), config.loop_count);
  for (const LlmScenarioAggregate& aggregate : result.aggregates) {
    EXPECT_EQ(aggregate.work_unit_latency_seconds.values.capacity(), config.loop_count);
    EXPECT_EQ(aggregate.synthetic_memory_work_units_per_second.values.capacity(), config.loop_count);
    EXPECT_EQ(aggregate.effective_model_payload_gb_s.values.capacity(), config.loop_count);
  }
}

TEST(LlmMemoryRunnerTest,
     AutomaticCalibrationFreezesAtomicallyThenWarmsBeforeCyclicMeasurements) {
  const LlmMemoryConfig config = automatic_config(2);
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  FakeLlmBackend executor;
  executor.mutate = [](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context,
                       size_t, LlmTaskExecutionResult& result) {
    const size_t index = static_cast<size_t>(context.scenario);
    if (context.purpose == "pilot") {
      result.timing.elapsed_seconds = 0.010 * (index + 1);
    } else if (context.kind == LlmRunnerTaskKind::Measurement) {
      result.timing.elapsed_seconds = 0.400;
    }
  };
  LlmMemoryResult result;

  ASSERT_EQ(run_llm_memory_suite(config, plan, executor, result), EXIT_SUCCESS);
  ASSERT_TRUE(result.frozen_scenario_plans.valid);
  ASSERT_EQ(executor.calls.size(), 18u);
  constexpr std::array<LlmScenario, kLlmScenarioCount> kScenarios = {
      LlmScenario::WeightsOnly, LlmScenario::KvOnly, LlmScenario::Mixed};
  size_t call_index = 0;
  for (LlmScenario scenario : kScenarios) {
    const size_t index = static_cast<size_t>(scenario);
    const LlmScenarioLimits limits =
        calculate_llm_scenario_limits(plan.geometry, scenario);
    ASSERT_TRUE(limits.valid);
    const TaskRecord& shape_warmup = executor.calls[call_index++];
    const TaskRecord& pilot = executor.calls[call_index++];
    const TaskRecord& correction = executor.calls[call_index++];
    EXPECT_EQ(shape_warmup.context.kind, LlmRunnerTaskKind::Warmup);
    EXPECT_EQ(shape_warmup.context.purpose,
              "calibration_shape_warmup");
    EXPECT_EQ(shape_warmup.context.scenario, scenario);
    EXPECT_EQ(pilot.context.kind, LlmRunnerTaskKind::Calibration);
    EXPECT_EQ(pilot.context.purpose, "pilot");
    EXPECT_EQ(pilot.context.scenario, scenario);
    EXPECT_EQ(shape_warmup.work_units, pilot.work_units);
    EXPECT_EQ(shape_warmup.payload_bytes, pilot.payload_bytes);
    EXPECT_EQ(shape_warmup.plan_identity, pilot.plan_identity);
    EXPECT_EQ(pilot.work_units, calculate_llm_pilot_work_units(limits));
    EXPECT_EQ(correction.context.kind, LlmRunnerTaskKind::Calibration);
    EXPECT_EQ(correction.context.purpose, "correction");
    EXPECT_EQ(correction.context.scenario, scenario);
    const size_t expected_work_units =
        calculate_llm_calibrated_work_units(
            0.010 * (index + 1), pilot.work_units, limits);
    EXPECT_EQ(correction.work_units, expected_work_units);
    EXPECT_EQ(result.frozen_scenario_plans.scenarios[index].work_units,
              expected_work_units);
    EXPECT_EQ(correction.payload_bytes,
              result.frozen_scenario_plans.scenarios[index]
                  .effective_model_payload_bytes);
    EXPECT_EQ(correction.plan_identity,
              result.frozen_scenario_plans.scenarios[index].plan_identity);
  }

  EXPECT_EQ(call_index, 3u * kLlmScenarioCount);
  for (LlmScenario scenario : kScenarios) {
    const size_t index = static_cast<size_t>(scenario);
    const TaskRecord& frozen_warmup = executor.calls[call_index++];
    const LlmScenarioWorkPlan& frozen =
        result.frozen_scenario_plans.scenarios[index];
    EXPECT_EQ(frozen_warmup.context.kind, LlmRunnerTaskKind::Warmup);
    EXPECT_EQ(frozen_warmup.context.purpose,
              "frozen_measurement_warmup");
    EXPECT_EQ(frozen_warmup.context.scenario, scenario);
    EXPECT_EQ(frozen_warmup.work_units, frozen.work_units);
    EXPECT_EQ(frozen_warmup.payload_bytes,
              frozen.effective_model_payload_bytes);
    EXPECT_EQ(frozen_warmup.plan_identity, frozen.plan_identity);
  }

  EXPECT_EQ(call_index, 4u * kLlmScenarioCount);
  for (size_t loop_index = 0; loop_index < config.loop_count; ++loop_index) {
    const std::array<LlmScenario, kLlmScenarioCount> expected_order =
        build_llm_scenario_order(loop_index);
    for (size_t position = 0; position < kLlmScenarioCount; ++position) {
      const LlmScenario scenario = expected_order[position];
      const LlmScenarioWorkPlan& frozen =
          result.frozen_scenario_plans
              .scenarios[static_cast<size_t>(scenario)];
      const TaskRecord& measurement = executor.calls[call_index++];
      EXPECT_EQ(measurement.context.kind,
                LlmRunnerTaskKind::Measurement);
      EXPECT_EQ(measurement.context.purpose, "measurement");
      EXPECT_EQ(measurement.context.scenario, scenario);
      EXPECT_EQ(measurement.context.loop_index, loop_index);
      EXPECT_EQ(measurement.context.order_position, position);
      EXPECT_EQ(measurement.work_units, frozen.work_units);
      EXPECT_EQ(measurement.payload_bytes,
                frozen.effective_model_payload_bytes);
      EXPECT_EQ(measurement.plan_identity, frozen.plan_identity);
    }
  }
  EXPECT_EQ(call_index, executor.calls.size());
  EXPECT_EQ(result.calibration_attempts[0].size(), 4u);
  EXPECT_EQ(result.calibration_attempts[1].size(), 4u);
  EXPECT_EQ(result.calibration_attempts[2].size(), 4u);
}

TEST(LlmMemoryRunnerTest,
     AutomaticCorrectionTrialsHaveNoGeneralWarmup) {
  const LlmMemoryConfig config = automatic_config();
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  FakeLlmBackend executor;
  executor.mutate = [](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context,
                       size_t, LlmTaskExecutionResult& result) {
    if (context.purpose == "pilot") {
      result.timing.elapsed_seconds = 0.010;
    } else if (context.purpose == "correction") {
      result.timing.elapsed_seconds =
          context.attempt_index == 2 ? 0.050 : 0.150;
    }
  };
  LlmMemoryResult result;

  ASSERT_EQ(run_llm_memory_suite(config, plan, executor, result), EXIT_SUCCESS);
  for (LlmScenario scenario : {LlmScenario::WeightsOnly, LlmScenario::KvOnly, LlmScenario::Mixed}) {
    const size_t index = static_cast<size_t>(scenario);
    const std::vector<TaskRecord> excluded = records_for_scenario(executor.calls, scenario, false);
    ASSERT_EQ(excluded.size(), 5u);
    EXPECT_EQ(excluded[0].context.purpose,
              "calibration_shape_warmup");
    EXPECT_EQ(excluded[0].context.kind, LlmRunnerTaskKind::Warmup);
    EXPECT_EQ(excluded[1].context.purpose, "pilot");
    EXPECT_EQ(excluded[1].context.kind, LlmRunnerTaskKind::Calibration);
    EXPECT_EQ(excluded[2].context.purpose, "correction");
    EXPECT_EQ(excluded[2].context.kind, LlmRunnerTaskKind::Calibration);
    EXPECT_EQ(excluded[3].context.purpose, "correction");
    EXPECT_EQ(excluded[3].context.kind, LlmRunnerTaskKind::Calibration);
    EXPECT_EQ(excluded[4].context.purpose,
              "frozen_measurement_warmup");
    EXPECT_EQ(excluded[4].context.kind, LlmRunnerTaskKind::Warmup);
    EXPECT_EQ(result.frozen_scenario_plans.scenarios[index].work_units,
              excluded[3].work_units);
    EXPECT_EQ(excluded[4].plan_identity,
              result.frozen_scenario_plans.scenarios[index].plan_identity);
    EXPECT_EQ(std::count_if(excluded.begin(), excluded.end(),
                            [](const TaskRecord& record) { return record.context.kind == LlmRunnerTaskKind::Warmup; }),
              2);
    EXPECT_EQ(std::count_if(
                  excluded.begin(), excluded.end(),
                  [](const TaskRecord& record) {
                    return record.context.purpose ==
                           "single_unit_confirmation_warmup";
                  }),
              0);
  }
}

TEST(LlmMemoryRunnerTest,
     SingleUnitConfirmationWarmsOnlyNewShapeAndFreezesAboveTargetEvidence) {
  LlmMemoryConfig config = automatic_prefill_config();
  config.weight_size_mb = 8;
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  ASSERT_EQ(plan.phase, LlmPhase::Prefill);
  ASSERT_EQ(plan.work_unit_kind, LlmWorkUnitKind::PrefillOperation);
  const LlmScenarioLimits weights_limits = calculate_llm_scenario_limits(
      plan.geometry, LlmScenario::WeightsOnly);
  const LlmScenarioLimits kv_limits =
      calculate_llm_scenario_limits(plan.geometry, LlmScenario::KvOnly);
  const LlmScenarioLimits mixed_limits =
      calculate_llm_scenario_limits(plan.geometry, LlmScenario::Mixed);
  ASSERT_TRUE(weights_limits.valid);
  ASSERT_TRUE(kv_limits.valid);
  ASSERT_TRUE(mixed_limits.valid);
  EXPECT_EQ(calculate_llm_pilot_work_units(weights_limits), 1u);
  EXPECT_GT(calculate_llm_pilot_work_units(kv_limits), 1u);
  EXPECT_EQ(calculate_llm_pilot_work_units(mixed_limits), 1u);

  FakeLlmBackend executor;
  executor.mutate = [](const LlmMemoryWorkPlan&,
                       const LlmScenarioWorkPlan& task_plan,
                       const LlmRunnerTaskContext& context, size_t,
                       LlmTaskExecutionResult& result) {
    if (context.purpose == "pilot") {
      result.timing.elapsed_seconds =
          static_cast<double>(task_plan.work_units) * 0.300;
    } else if (context.purpose == "single_unit_confirmation") {
      result.timing.elapsed_seconds = 0.300;
    }
  };
  LlmMemoryResult result;

  ASSERT_EQ(run_llm_memory_suite(config, plan, executor, result),
            EXIT_SUCCESS);
  ASSERT_TRUE(result.frozen_scenario_plans.valid);
  for (const LlmScenarioWorkPlan& frozen :
       result.frozen_scenario_plans.scenarios) {
    EXPECT_EQ(frozen.work_units, 1u);
    EXPECT_EQ(frozen.work_unit_kind, LlmWorkUnitKind::PrefillOperation);
    const LlmKvWriteKind expected_write =
        frozen.scenario == LlmScenario::WeightsOnly
            ? LlmKvWriteKind::None
            : LlmKvWriteKind::FullPromptPopulation;
    EXPECT_EQ(frozen.kv_write_kind, expected_write);
  }

  const std::vector<TaskRecord> weights = records_for_scenario(
      executor.calls, LlmScenario::WeightsOnly, false);
  const std::vector<TaskRecord> kv =
      records_for_scenario(executor.calls, LlmScenario::KvOnly, false);
  const std::vector<TaskRecord> mixed =
      records_for_scenario(executor.calls, LlmScenario::Mixed, false);
  ASSERT_EQ(weights.size(), 3u);
  ASSERT_EQ(kv.size(), 5u);
  ASSERT_EQ(mixed.size(), 3u);
  EXPECT_EQ(weights[0].context.purpose, "calibration_shape_warmup");
  EXPECT_EQ(weights[1].context.purpose, "pilot");
  EXPECT_EQ(weights[2].context.purpose, "frozen_measurement_warmup");
  EXPECT_EQ(mixed[0].context.purpose, "calibration_shape_warmup");
  EXPECT_EQ(mixed[1].context.purpose, "pilot");
  EXPECT_EQ(mixed[2].context.purpose, "frozen_measurement_warmup");
  EXPECT_EQ(kv[0].context.purpose, "calibration_shape_warmup");
  EXPECT_EQ(kv[1].context.purpose, "pilot");
  EXPECT_EQ(kv[2].context.purpose,
            "single_unit_confirmation_warmup");
  EXPECT_EQ(kv[2].context.kind, LlmRunnerTaskKind::Warmup);
  EXPECT_EQ(kv[2].work_units, 1u);
  EXPECT_EQ(kv[3].context.purpose, "single_unit_confirmation");
  EXPECT_EQ(kv[3].context.kind, LlmRunnerTaskKind::Calibration);
  EXPECT_EQ(kv[3].work_units, 1u);
  EXPECT_EQ(kv[4].context.purpose, "frozen_measurement_warmup");

  ASSERT_EQ(result.calibration_attempts[0].size(), 3u);
  ASSERT_EQ(result.calibration_attempts[1].size(), 5u);
  ASSERT_EQ(result.calibration_attempts[2].size(), 3u);
  EXPECT_EQ(result.calibration_attempts[0][1].duration_quality,
            "above-target-single-work-unit");
  EXPECT_EQ(result.calibration_attempts[1][3].duration_quality,
            "above-target-single-work-unit");
  EXPECT_EQ(result.calibration_attempts[2][1].duration_quality,
            "above-target-single-work-unit");
  for (const TaskRecord& call : executor.calls) {
    EXPECT_EQ(call.work_unit_kind, LlmWorkUnitKind::PrefillOperation);
    const LlmKvWriteKind expected_write =
        call.context.scenario == LlmScenario::WeightsOnly
            ? LlmKvWriteKind::None
            : LlmKvWriteKind::FullPromptPopulation;
    EXPECT_EQ(call.kv_write_kind, expected_write);
  }
}

TEST(LlmMemoryRunnerTest,
     FakeBackendRunsGenericPrefillWorkUnitsAndAccountsFullPromptWrites) {
  const LlmMemoryConfig config = explicit_prefill_config();
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  ASSERT_EQ(plan.phase, LlmPhase::Prefill);
  ASSERT_EQ(plan.work_unit_kind, LlmWorkUnitKind::PrefillOperation);
  ASSERT_TRUE(plan.geometry.prefill.has_value());
  EXPECT_FALSE(plan.geometry.decode.has_value());
  EXPECT_EQ(plan.geometry.prefill->prompt_tokens, config.prompt_tokens);
  EXPECT_EQ(plan.geometry.prefill->attention_query_tile_tokens,
            config.attention_query_tile_tokens);

  FakeLlmBackend backend;
  LlmMemoryResult result;
  ASSERT_EQ(run_llm_memory_suite(config, plan, backend, result),
            EXIT_SUCCESS)
      << result.reason_code << ": " << result.diagnostic;
  ASSERT_TRUE(result.frozen_scenario_plans.valid);
  ASSERT_EQ(result.measurements.size(),
            config.loop_count * kLlmScenarioCount);
  ASSERT_EQ(backend.calls.size(),
            kLlmScenarioCount + result.measurements.size());

  size_t payload_bytes_per_loop = 0;
  for (size_t index = 0; index < kLlmScenarioCount; ++index) {
    const LlmScenarioWorkPlan& frozen =
        result.frozen_scenario_plans.scenarios[index];
    const LlmKvWriteKind expected_write =
        frozen.scenario == LlmScenario::WeightsOnly
            ? LlmKvWriteKind::None
            : LlmKvWriteKind::FullPromptPopulation;
    EXPECT_EQ(frozen.work_unit_kind, LlmWorkUnitKind::PrefillOperation);
    EXPECT_EQ(frozen.kv_write_kind, expected_write);
    EXPECT_EQ(frozen.work_units, config.iterations);
    if (frozen.scenario == LlmScenario::WeightsOnly) {
      EXPECT_EQ(frozen.kv_write_bytes, 0u);
    } else {
      EXPECT_GT(frozen.kv_write_bytes, 0u);
    }
    payload_bytes_per_loop += frozen.effective_model_payload_bytes;
  }

  const size_t expected_work_units =
      config.loop_count * kLlmScenarioCount * config.iterations;
  EXPECT_EQ(result.counters.planned_work_units, expected_work_units);
  EXPECT_EQ(result.counters.completed_work_units, expected_work_units);
  EXPECT_EQ(result.counters.planned_effective_model_payload_bytes,
            payload_bytes_per_loop * config.loop_count);
  EXPECT_EQ(result.counters.completed_effective_model_payload_bytes,
            payload_bytes_per_loop * config.loop_count);
  EXPECT_EQ(result.counters.planned_layout_metadata_lookup_count, 0u);
  EXPECT_EQ(result.counters.completed_layout_metadata_lookup_count, 0u);

  for (const TaskRecord& call : backend.calls) {
    const LlmKvWriteKind expected_write =
        call.context.scenario == LlmScenario::WeightsOnly
            ? LlmKvWriteKind::None
            : LlmKvWriteKind::FullPromptPopulation;
    EXPECT_EQ(call.work_unit_kind, LlmWorkUnitKind::PrefillOperation);
    EXPECT_EQ(call.kv_write_kind, expected_write);
  }
  for (const LlmMeasurementState& measurement : result.measurements) {
    const LlmKvWriteKind expected_write =
        measurement.scenario == LlmScenario::WeightsOnly
            ? LlmKvWriteKind::None
            : LlmKvWriteKind::FullPromptPopulation;
    EXPECT_EQ(measurement.work_unit_kind,
              LlmWorkUnitKind::PrefillOperation);
    EXPECT_EQ(measurement.kv_write_kind, expected_write);
    EXPECT_EQ(measurement.planned_work_units, config.iterations);
    EXPECT_EQ(measurement.completed_work_units, config.iterations);
  }
}

TEST(LlmMemoryRunnerTest,
     ProductionCpuBackendKeepsPurePrefillAtUnsupportedBoundary) {
  const LlmMemoryConfig config = explicit_prefill_config(1, 1);
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;

  std::unique_ptr<LlmBackend> backend = create_llm_cpu_backend();
  ASSERT_NE(backend, nullptr);
  const LlmBackendAuxiliaryEstimate auxiliary =
      backend->calculate_auxiliary_estimate(plan);
  ASSERT_TRUE(auxiliary.valid) << auxiliary.reason_code;
  EXPECT_EQ(auxiliary.reason_code, LlmBackendReason::VALID);
  EXPECT_EQ(auxiliary.checksum_auxiliary_bytes, 0u);
  EXPECT_EQ(auxiliary.orchestration_auxiliary_bytes, 0u);
  EXPECT_EQ(auxiliary.total_auxiliary_bytes, 0u);

  const LlmBackendLifecycleResult initialization =
      backend->initialize(config);
  ASSERT_EQ(initialization.status, LlmBackendStatus::Ready);
  EXPECT_EQ(initialization.reason_code, LlmBackendReason::VALID);
  const LlmBackendLifecycleResult resolution =
      backend->resolve_execution_plan(plan);
  EXPECT_EQ(resolution.status, LlmBackendStatus::Unsupported);
  EXPECT_EQ(resolution.reason_code, LlmBackendReason::TASK_UNSUPPORTED);
  const LlmBackendLifecycleResult release = backend->release_resources();
  EXPECT_EQ(release.status, LlmBackendStatus::Ready);
  EXPECT_EQ(release.reason_code, LlmBackendReason::VALID);
}

TEST(LlmMemoryRunnerTest, StopBeforeFirstTaskInterruptsAllSlotsWithoutExecutorWork) {
  const LlmMemoryConfig config = explicit_config(2);
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  FakeLlmBackend executor;
  std::vector<CheckpointRecord> checkpoints;
  LlmRunnerHooks hooks = recording_checkpoints(checkpoints);
  hooks.stop_requested = []() { return true; };
  LlmMemoryResult result;

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor, result, hooks), EXIT_SUCCESS);
  EXPECT_TRUE(executor.calls.empty());
  EXPECT_EQ(result.status, LlmRunStatus::Interrupted);
  EXPECT_TRUE(result.interruption_requested);
  EXPECT_EQ(result.counters.attempted_loops, 0u);
  EXPECT_EQ(result.counters.attempted_measurements, 0u);
  EXPECT_EQ(result.counters.terminal_measurements, 6u);
  EXPECT_TRUE(result.frozen_scenario_plans.valid);
  EXPECT_GT(result.counters.planned_work_units, 0u);
  EXPECT_GT(result.counters.planned_effective_model_payload_bytes, 0u);
  for (const LlmMeasurementState& measurement : result.measurements) {
    EXPECT_LT(measurement.frozen_plan_index, kLlmScenarioCount);
    EXPECT_EQ(measurement.planned_work_units, config.iterations);
    EXPECT_GT(measurement.planned_effective_model_payload_bytes, 0u);
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
  FakeLlmBackend executor;
  executor.mutate = [&](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context,
                        size_t, LlmTaskExecutionResult&) {
    if (context.kind == LlmRunnerTaskKind::Measurement) {
      stop = true;
    }
  };
  std::vector<CheckpointRecord> checkpoints;
  LlmRunnerHooks hooks = recording_checkpoints(checkpoints);
  hooks.stop_requested = [&]() { return stop; };
  LlmMemoryResult result;

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor, result, hooks), EXIT_SUCCESS);
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
  EXPECT_EQ(result.counters.completed_work_units, config.iterations);
  EXPECT_EQ(result.counters.completed_effective_model_payload_bytes,
            result.measurements[0].planned_effective_model_payload_bytes);
  ASSERT_EQ(checkpoints.size(), 2u);
  EXPECT_EQ(checkpoints[0].kind, LlmCheckpointKind::MeasurementTerminal);
  EXPECT_EQ(checkpoints[1].kind, LlmCheckpointKind::CommandTerminal);
}

TEST(LlmMemoryRunnerTest, StopRaisedByMeasurementCheckpointAddsOnlyCommandTerminalSnapshot) {
  const LlmMemoryConfig config = explicit_config(2);
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  bool stop = false;
  FakeLlmBackend executor;
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

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor, result, hooks), EXIT_SUCCESS);
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
  FakeLlmBackend executor;
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

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor, result, hooks), EXIT_SUCCESS);
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
  FakeLlmBackend executor;
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

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor, result, hooks), EXIT_FAILURE);
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

TEST(LlmMemoryRunnerTest, BackendFailureWinsSimultaneousStopAndPreservesInterruptedTail) {
  const LlmMemoryConfig config = explicit_config(2);
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  bool stop = false;
  FakeLlmBackend executor;
  executor.mutate = [&](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context,
                        size_t, LlmTaskExecutionResult& execution) {
    if (context.kind == LlmRunnerTaskKind::Measurement) {
      stop = true;
      fail_execution(execution);
    }
  };
  LlmRunnerHooks hooks;
  hooks.stop_requested = [&]() { return stop; };
  LlmMemoryResult result;

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor, result, hooks), EXIT_FAILURE);
  EXPECT_EQ(result.status, LlmRunStatus::Failed);
  EXPECT_EQ(result.reason_code, LlmBackendReason::RESOURCES_NOT_PREPARED);
  EXPECT_TRUE(result.interruption_requested);
  ASSERT_EQ(result.measurements[0].status, LlmMeasurementStatus::Failed);
  EXPECT_EQ(result.measurements[0].reason_code,
            LlmBackendReason::RESOURCES_NOT_PREPARED);
  for (size_t index = 1; index < result.measurements.size(); ++index) {
    EXPECT_EQ(result.measurements[index].status, LlmMeasurementStatus::Interrupted);
  }
}

TEST(LlmMemoryRunnerTest, ValidationFailureWinsSimultaneousStopAndIsExcludedFromAggregates) {
  const LlmMemoryConfig config = explicit_config(2);
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  bool stop = false;
  FakeLlmBackend executor;
  executor.mutate = [&](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context,
                        size_t, LlmTaskExecutionResult& execution) {
    if (context.kind == LlmRunnerTaskKind::Measurement) {
      stop = true;
      invalidate_execution(execution);
    }
  };
  LlmRunnerHooks hooks;
  hooks.stop_requested = [&]() { return stop; };
  LlmMemoryResult result;

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor, result, hooks), EXIT_FAILURE);
  EXPECT_EQ(result.status, LlmRunStatus::Failed);
  EXPECT_EQ(result.reason_code, LlmBackendReason::VALIDATION_FAILED);
  EXPECT_TRUE(result.interruption_requested);
  EXPECT_EQ(result.measurements[0].status, LlmMeasurementStatus::Invalid);
  EXPECT_FALSE(result.measurements[0].elapsed_seconds.has_value());
  EXPECT_TRUE(result.aggregates[0].effective_model_payload_gb_s.values.empty());
}

TEST(LlmMemoryRunnerTest, MeasurementCheckpointFailureIsTerminalAndNeverRetried) {
  const LlmMemoryConfig config = explicit_config(2);
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  FakeLlmBackend executor;
  std::vector<LlmCheckpointKind> kinds;
  LlmRunnerHooks hooks;
  hooks.stop_requested = []() { return false; };
  hooks.checkpoint = [&](const LlmMemoryResult&, LlmCheckpointKind kind) {
    kinds.push_back(kind);
    return EXIT_FAILURE;
  };
  LlmMemoryResult result;

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor, result, hooks), EXIT_FAILURE);
  EXPECT_EQ(kinds, (std::vector<LlmCheckpointKind>{LlmCheckpointKind::MeasurementTerminal}));
  EXPECT_EQ(result.status, LlmRunStatus::Failed);
  EXPECT_EQ(result.reason_code, LlmRunnerReason::CHECKPOINT_WRITE_FAILED);
  EXPECT_TRUE(result.checkpoint_failed);
  EXPECT_FALSE(result.terminal_checkpoint_attempted);
  EXPECT_EQ(result.counters.attempted_measurements, 1u);
  EXPECT_EQ(result.measurements[0].status, LlmMeasurementStatus::Measured);
  EXPECT_TRUE(result.measurements[0].elapsed_seconds.has_value());
  EXPECT_TRUE(result.measurements[0].synthetic_work_unit_latency_seconds.has_value());
  EXPECT_TRUE(result.measurements[0].synthetic_memory_work_units_per_second.has_value());
  EXPECT_TRUE(result.measurements[0].effective_model_payload_gb_s.has_value());
  EXPECT_TRUE(result.measurements[0].checksum_valid);
  EXPECT_EQ(result.measurements[0].execution.status,
            LlmTaskExecutionStatus::Complete);
  EXPECT_TRUE(result.measurements[0].execution.timing.valid);
  EXPECT_TRUE(result.measurements[0].execution.validation.valid);
  EXPECT_TRUE(std::holds_alternative<std::monostate>(
      result.measurements[0].execution.backend_evidence));
  EXPECT_EQ(result.aggregates[0].effective_model_payload_gb_s.values.size(), 1u);
  EXPECT_EQ(result.counters.planned_loops, 2u);
  EXPECT_EQ(result.counters.attempted_loops, 1u);
  EXPECT_EQ(result.counters.completed_loops, 0u);
  EXPECT_EQ(result.counters.planned_measurements, 6u);
  EXPECT_EQ(result.counters.terminal_measurements, 6u);
  EXPECT_EQ(result.counters.measured_measurements, 1u);
  EXPECT_EQ(result.counters.completed_work_units, config.iterations);
  EXPECT_EQ(result.counters.completed_effective_model_payload_bytes,
            result.measurements[0].planned_effective_model_payload_bytes);
  for (size_t index = 1; index < result.measurements.size(); ++index) {
    EXPECT_EQ(result.measurements[index].status, LlmMeasurementStatus::Failed);
  }

}

TEST(LlmMemoryRunnerTest, CanonicalResultReasonsDetachFromEveryOwningDomain) {
  std::string runner_reason = LlmRunnerReason::CHECKPOINT_WRITE_FAILED;
  std::string work_plan_reason = LlmWorkPlanReason::TASK_ACCOUNTED_BYTES_CAP_EXCEEDED;
  std::string executor_reason = LlmExecutorReason::KERNEL_FAILED;
  std::string paged_option_reason =
      LlmWorkPlanReason::KV_BLOCK_TOKENS_NOT_APPLICABLE;

  const std::string_view canonical_runner = canonicalize_llm_result_reason_code(runner_reason);
  const std::string_view canonical_work_plan = canonicalize_llm_result_reason_code(work_plan_reason);
  const std::string_view canonical_executor = canonicalize_llm_result_reason_code(executor_reason);
  const std::string_view canonical_paged_option =
      canonicalize_llm_result_reason_code(paged_option_reason);
  std::fill(runner_reason.begin(), runner_reason.end(), 'x');
  std::fill(work_plan_reason.begin(), work_plan_reason.end(), 'x');
  std::fill(executor_reason.begin(), executor_reason.end(), 'x');
  std::fill(paged_option_reason.begin(), paged_option_reason.end(), 'x');

  EXPECT_EQ(canonical_runner, LlmRunnerReason::CHECKPOINT_WRITE_FAILED);
  EXPECT_EQ(canonical_runner.data(),
            canonicalize_llm_result_reason_code(LlmRunnerReason::CHECKPOINT_WRITE_FAILED).data());
  EXPECT_EQ(canonical_work_plan, LlmWorkPlanReason::TASK_ACCOUNTED_BYTES_CAP_EXCEEDED);
  EXPECT_EQ(canonical_work_plan.data(),
            canonicalize_llm_result_reason_code(LlmWorkPlanReason::TASK_ACCOUNTED_BYTES_CAP_EXCEEDED).data());
  EXPECT_EQ(canonical_executor, LlmExecutorReason::KERNEL_FAILED);
  EXPECT_EQ(canonical_executor.data(),
            canonicalize_llm_result_reason_code(LlmExecutorReason::KERNEL_FAILED).data());
  EXPECT_EQ(canonical_paged_option,
            LlmWorkPlanReason::KV_BLOCK_TOKENS_NOT_APPLICABLE);
  EXPECT_EQ(canonical_paged_option.data(),
            canonicalize_llm_result_reason_code(
                LlmWorkPlanReason::KV_BLOCK_TOKENS_NOT_APPLICABLE)
                .data());
  EXPECT_EQ(canonicalize_llm_result_reason_code("not-a-reason"), LlmRunnerReason::RUNNER_UNKNOWN_EXCEPTION);
}

TEST(LlmMemoryRunnerTest,
     PagedPostValidationFailureCanonicalizesExactlyThroughRunnerFailure) {
  std::string owned_reason =
      LlmExecutorReason::PAGED_POST_VALIDATION_FAILED;
  const std::string_view canonical =
      canonicalize_llm_result_reason_code(owned_reason);
  std::fill(owned_reason.begin(), owned_reason.end(), 'x');
  EXPECT_EQ(canonical, LlmExecutorReason::PAGED_POST_VALIDATION_FAILED);
  EXPECT_NE(canonical, LlmRunnerReason::RUNNER_UNKNOWN_EXCEPTION);
  EXPECT_EQ(
      canonical.data(),
      canonicalize_llm_result_reason_code(
          LlmExecutorReason::PAGED_POST_VALIDATION_FAILED)
          .data());

  LlmMemoryConfig config = explicit_config(1);
  config.kv_layout = LlmKvLayout::Paged;
  config.kv_block_tokens = 2;
  config.user_specified_kv_layout = true;
  config.user_specified_kv_block_tokens = true;
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  FakeLlmBackend executor;
  executor.mutate = [](const LlmMemoryWorkPlan&,
                       const LlmScenarioWorkPlan&,
                       const LlmRunnerTaskContext& context, size_t,
                       LlmTaskExecutionResult& execution) {
    if (context.kind != LlmRunnerTaskKind::Measurement) {
      return;
    }
    execution.status = LlmTaskExecutionStatus::Invalid;
    execution.reason_code =
        std::string(LlmExecutorReason::PAGED_POST_VALIDATION_FAILED);
    execution.validation.evaluated = true;
    execution.validation.valid = false;
    LlmCpuTaskEvidence evidence;
    evidence.executor.valid = false;
    evidence.executor.reason_code =
        std::string(LlmExecutorReason::PAGED_POST_VALIDATION_FAILED);
    evidence.executor.post_validation_evaluated = true;
    evidence.executor.post_validation_valid = false;
    execution.backend_evidence = std::move(evidence);
  };
  LlmMemoryResult result;

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor, result),
            EXIT_FAILURE);
  EXPECT_EQ(result.status, LlmRunStatus::Failed);
  EXPECT_EQ(result.reason_code,
            LlmExecutorReason::PAGED_POST_VALIDATION_FAILED);
  EXPECT_NE(result.reason_code, LlmRunnerReason::RUNNER_UNKNOWN_EXCEPTION);
  ASSERT_FALSE(result.measurements.empty());
  const LlmMeasurementState& measurement = result.measurements.front();
  EXPECT_EQ(measurement.status, LlmMeasurementStatus::Invalid);
  EXPECT_EQ(measurement.reason_code,
            LlmExecutorReason::PAGED_POST_VALIDATION_FAILED);
  EXPECT_EQ(measurement.execution.reason_code,
            LlmExecutorReason::PAGED_POST_VALIDATION_FAILED);
  const auto* retained =
      std::get_if<LlmCpuTaskEvidence>(&measurement.execution.backend_evidence);
  ASSERT_NE(retained, nullptr);
  EXPECT_EQ(retained->executor.reason_code,
            LlmExecutorReason::PAGED_POST_VALIDATION_FAILED);
  EXPECT_TRUE(retained->executor.post_validation_evaluated);
  EXPECT_FALSE(retained->executor.post_validation_valid);
}

TEST(LlmMemoryRunnerTest,
     AcceptedCpuMeasurementRetainsPagedPostValidationEvidence) {
  LlmMemoryConfig config = explicit_config(1);
  config.kv_layout = LlmKvLayout::Paged;
  config.kv_block_tokens = 2;
  config.user_specified_kv_layout = true;
  config.user_specified_kv_block_tokens = true;
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  ASSERT_EQ(plan.kv_layout, LlmKvLayout::Paged);
  const LlmCpuExecutionPlan* cpu_plan = get_llm_cpu_execution_plan(plan);
  ASSERT_NE(cpu_plan, nullptr);
  const size_t effective_workers = cpu_plan->effective_workers;

  FakeLlmBackend executor;
  executor.mutate = [effective_workers](
                        const LlmMemoryWorkPlan&,
                        const LlmScenarioWorkPlan&,
                        const LlmRunnerTaskContext& context, size_t,
                        LlmTaskExecutionResult& execution) {
    if (context.kind != LlmRunnerTaskKind::Measurement) {
      return;
    }
    LlmCpuTaskEvidence evidence;
    evidence.executor.valid = true;
    evidence.executor.reason_code = LlmExecutorReason::VALID;
    evidence.executor.elapsed_seconds = execution.timing.elapsed_seconds;
    evidence.executor.requested_workers = effective_workers;
    evidence.executor.created_workers = effective_workers;
    evidence.executor.completed_workers = effective_workers;
    evidence.executor.qos_successful_workers = effective_workers;
    evidence.executor.kernel_succeeded = true;
    evidence.executor.timer_started = true;
    evidence.executor.timer_stopped = true;
    evidence.executor.checksum_evaluated = true;
    evidence.executor.checksum_valid = true;
    evidence.executor.post_validation_evaluated = true;
    evidence.executor.post_validation_valid = true;
    execution.backend_evidence = std::move(evidence);
  };
  LlmMemoryResult result;

  ASSERT_EQ(run_llm_memory_suite(config, plan, executor, result), EXIT_SUCCESS);
  ASSERT_TRUE(result.results_complete);
  ASSERT_EQ(result.measurements.size(), kLlmScenarioCount);
  for (const LlmMeasurementState& measurement : result.measurements) {
    SCOPED_TRACE(::testing::Message()
                 << "scenario=" << static_cast<size_t>(measurement.scenario));
    ASSERT_EQ(measurement.status, LlmMeasurementStatus::Measured);
    ASSERT_TRUE(measurement.execution_evidence_available);
    const auto* retained = std::get_if<LlmCpuTaskEvidence>(
        &measurement.execution.backend_evidence);
    ASSERT_NE(retained, nullptr);
    EXPECT_EQ(retained->executor.reason_code, LlmExecutorReason::VALID);
    EXPECT_TRUE(retained->executor.post_validation_evaluated);
    EXPECT_TRUE(retained->executor.post_validation_valid);
  }
}

TEST(LlmMemoryRunnerTest, ResultCopiesAndMovesRetainStaticRunnerFailureAfterSourcesReset) {
  const LlmMemoryConfig config = explicit_config(2);
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  FakeLlmBackend executor;
  executor.mutate = [](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&,
                       const LlmRunnerTaskContext& context, size_t,
                       LlmTaskExecutionResult&) {
    if (context.kind == LlmRunnerTaskKind::Measurement) {
      throw std::runtime_error("injected runner failure");
    }
  };
  LlmMemoryResult result;

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor, result), EXIT_FAILURE);
  ASSERT_EQ(result.measurements.size(), config.loop_count * kLlmScenarioCount);
  EXPECT_EQ(result.measurements[0].reason_code,
            LlmRunnerReason::RUNNER_EXCEPTION);

  LlmMemoryResult copied = result;
  LlmMemoryResult moved = std::move(copied);
  const std::string_view canonical_runner_exception =
      canonicalize_llm_result_reason_code(
          LlmRunnerReason::RUNNER_EXCEPTION);
  const std::string_view canonical_not_run =
      canonicalize_llm_result_reason_code(LlmRunnerReason::NOT_RUN_AFTER_RUNTIME_FAILURE);
  EXPECT_EQ(moved.measurements[0].reason_code.data(),
            canonical_runner_exception.data());
  result = LlmMemoryResult{};
  copied = LlmMemoryResult{};

  EXPECT_EQ(moved.measurements[0].reason_code,
            LlmRunnerReason::RUNNER_EXCEPTION);
  EXPECT_EQ(moved.measurements[0].reason_code.data(),
            canonical_runner_exception.data());
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
  FakeLlmBackend executor;
  std::vector<LlmCheckpointKind> kinds;
  LlmRunnerHooks hooks;
  hooks.stop_requested = [&]() { return stop; };
  hooks.checkpoint = [&](const LlmMemoryResult&, LlmCheckpointKind kind) {
    kinds.push_back(kind);
    stop = true;
    return EXIT_FAILURE;
  };
  LlmMemoryResult result;

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor, result, hooks), EXIT_FAILURE);
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
    FakeLlmBackend executor;
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

    EXPECT_EQ(run_llm_memory_suite(config, plan, executor, result, hooks), EXIT_FAILURE);
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
  FakeLlmBackend executor;
  executor.mutate = [](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&,
                       const LlmRunnerTaskContext& context, size_t,
                       LlmTaskExecutionResult&) {
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

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor, result, hooks), EXIT_FAILURE);
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
  FakeLlmBackend executor;
  std::vector<LlmCheckpointKind> kinds;
  LlmRunnerHooks hooks;
  hooks.stop_requested = []() { return false; };
  hooks.checkpoint = [&](const LlmMemoryResult&, LlmCheckpointKind kind) {
    kinds.push_back(kind);
    return kind == LlmCheckpointKind::CommandTerminal ? EXIT_FAILURE : EXIT_SUCCESS;
  };
  LlmMemoryResult result;

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor, result, hooks), EXIT_FAILURE);
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
  FakeLlmBackend executor;
  executor.mutate = [&](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan& task_plan,
                        const LlmRunnerTaskContext& context, size_t, LlmTaskExecutionResult& execution) {
    if (context.kind != LlmRunnerTaskKind::Measurement) {
      return;
    }
    const double gb_s = context.scenario == LlmScenario::WeightsOnly ? stable_values[context.loop_index]
                                                                     : noisy_values[context.loop_index];
    execution.timing.elapsed_seconds = static_cast<double>(task_plan.effective_model_payload_bytes) / gb_s / 1.0e9;
  };
  LlmMemoryResult result;

  ASSERT_EQ(run_llm_memory_suite(config, plan, executor, result), EXIT_SUCCESS);
  const LlmScenarioAggregate& weights = result.aggregates[0];
  EXPECT_EQ(weights.effective_model_payload_gb_s.values.size(), 3u);
  EXPECT_NEAR(weights.effective_model_payload_gb_s.statistics.coefficient_of_variation_pct, 5.0, 1e-10);
  EXPECT_EQ(weights.stability_quality, "stable");
  EXPECT_DOUBLE_EQ(*weights.effective_model_payload_gb_s.headline, 100.0);

  for (size_t index : {1u, 2u}) {
    const LlmScenarioAggregate& aggregate = result.aggregates[index];
    EXPECT_EQ(aggregate.effective_model_payload_gb_s.values.size(), 3u);
    EXPECT_GT(aggregate.effective_model_payload_gb_s.statistics.coefficient_of_variation_pct,
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
  FakeLlmBackend executor;
  executor.mutate = [&](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan& task_plan,
                        const LlmRunnerTaskContext& context, size_t, LlmTaskExecutionResult& execution) {
    if (context.kind != LlmRunnerTaskKind::Measurement || context.scenario != LlmScenario::WeightsOnly) {
      return;
    }
    ++weights_measurements;
    execution.timing.elapsed_seconds = static_cast<double>(task_plan.effective_model_payload_bytes) / 100.0 / 1.0e9;
    if (weights_measurements == 2) {
      invalidate_execution(execution);
    }
  };
  LlmMemoryResult result;

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor, result), EXIT_FAILURE);
  EXPECT_EQ(result.status, LlmRunStatus::Failed);
  EXPECT_EQ(result.reason_code, LlmBackendReason::VALIDATION_FAILED);
  ASSERT_EQ(result.aggregates[0].effective_model_payload_gb_s.values.size(), 1u);
  EXPECT_NEAR(result.aggregates[0].effective_model_payload_gb_s.values.front(), 100.0, 1e-10);
  EXPECT_EQ(result.aggregates[0].status, "partial");
  EXPECT_EQ(result.aggregates[0].stability_quality, "insufficient-samples");
  EXPECT_EQ(result.counters.measured_measurements, 5u);
  EXPECT_EQ(result.counters.attempted_measurements, 6u);
  EXPECT_EQ(result.measurements[5].status, LlmMeasurementStatus::Invalid);
  EXPECT_FALSE(result.measurements[5].effective_model_payload_gb_s.has_value());
}

TEST(LlmMemoryRunnerTest, InvalidAuthoritativeElapsedIsRejectedByCommonAcceptance) {
  const LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  FakeLlmBackend executor;
  executor.mutate = [](const LlmMemoryWorkPlan&,
                       const LlmScenarioWorkPlan&,
                       const LlmRunnerTaskContext& context, size_t,
                       LlmTaskExecutionResult& execution) {
    if (context.kind == LlmRunnerTaskKind::Measurement) {
      execution.timing.valid = false;
      execution.timing.elapsed_seconds = 0.0;
    }
  };
  LlmMemoryResult result;

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor, result), EXIT_FAILURE);
  EXPECT_EQ(result.status, LlmRunStatus::Failed);
  EXPECT_EQ(result.reason_code,
            LlmBackendReason::INVALID_AUTHORITATIVE_ELAPSED);
  ASSERT_FALSE(result.measurements.empty());
  EXPECT_EQ(result.measurements[0].status, LlmMeasurementStatus::Invalid);
  EXPECT_EQ(result.measurements[0].reason_code,
            LlmBackendReason::INVALID_AUTHORITATIVE_ELAPSED);
  EXPECT_FALSE(result.measurements[0].elapsed_seconds.has_value());
  EXPECT_TRUE(result.aggregates[0].effective_model_payload_gb_s.values.empty());
}

TEST(LlmMemoryRunnerTest, MeasurementCompletionMismatchIsRejectedWithoutAggregate) {
  const LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  FakeLlmBackend executor;
  executor.mutate = [](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context,
                       size_t, LlmTaskExecutionResult& execution) {
    if (context.kind == LlmRunnerTaskKind::Measurement) {
      --execution.completion.completed_work_units;
    }
  };
  LlmMemoryResult result;

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor, result), EXIT_FAILURE);
  EXPECT_EQ(result.status, LlmRunStatus::Failed);
  EXPECT_EQ(result.reason_code, LlmBackendReason::TASK_COMPLETION_MISMATCH);
  ASSERT_FALSE(result.measurements.empty());
  EXPECT_EQ(result.measurements[0].status, LlmMeasurementStatus::Failed);
  EXPECT_EQ(result.measurements[0].reason_code,
            LlmBackendReason::TASK_COMPLETION_MISMATCH);
  EXPECT_TRUE(std::holds_alternative<std::monostate>(
      result.measurements[0].execution.backend_evidence));
  EXPECT_TRUE(result.aggregates[0].effective_model_payload_gb_s.values.empty());
  EXPECT_EQ(result.counters.measured_measurements, 0u);
}

TEST(LlmMemoryRunnerTest, ExcludedCompletionMismatchRetainsGenericReason) {
  const LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  FakeLlmBackend executor;
  executor.mutate = [](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&,
                       const LlmRunnerTaskContext& context, size_t,
                       LlmTaskExecutionResult& execution) {
    if (context.kind == LlmRunnerTaskKind::Warmup) {
      --execution.completion.completed_task_accounted_bytes;
    }
  };
  LlmMemoryResult result;

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor, result), EXIT_FAILURE);
  EXPECT_EQ(result.status, LlmRunStatus::Failed);
  EXPECT_EQ(result.reason_code, LlmBackendReason::TASK_COMPLETION_MISMATCH);
  ASSERT_EQ(result.calibration_attempts[0].size(), 1u);
  EXPECT_FALSE(result.calibration_attempts[0][0].valid);
  EXPECT_EQ(result.calibration_attempts[0][0].reason_code,
            LlmBackendReason::TASK_COMPLETION_MISMATCH);
}

TEST(LlmMemoryRunnerTest, CalibrationFailureRetainsAttemptAndFinalizesMeasurementSlots) {
  const LlmMemoryConfig config = automatic_config(2);
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  FakeLlmBackend executor;
  executor.mutate = [](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context,
                       size_t, LlmTaskExecutionResult& execution) {
    if (context.purpose == "pilot") {
      fail_execution(execution);
    }
  };
  std::vector<CheckpointRecord> checkpoints;
  LlmMemoryResult result;

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor, result, recording_checkpoints(checkpoints)),
            EXIT_FAILURE);
  ASSERT_EQ(result.calibration_attempts[0].size(), 2u);
  EXPECT_EQ(result.calibration_attempts[0][1].purpose, "pilot");
  EXPECT_FALSE(result.calibration_attempts[0][1].valid);
  EXPECT_EQ(result.calibration_attempts[0][1].reason_code,
            LlmBackendReason::RESOURCES_NOT_PREPARED);
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
  FakeLlmBackend executor;
  executor.mutate = [&](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context,
                        size_t, LlmTaskExecutionResult&) {
    if (context.kind == LlmRunnerTaskKind::Warmup) {
      stop = true;
    }
  };
  std::vector<CheckpointRecord> checkpoints;
  LlmRunnerHooks hooks = recording_checkpoints(checkpoints);
  hooks.stop_requested = [&]() { return stop; };
  LlmMemoryResult result;

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor, result, hooks), EXIT_SUCCESS);
  EXPECT_EQ(result.status, LlmRunStatus::Interrupted);
  EXPECT_TRUE(result.interruption_requested);
  ASSERT_EQ(result.calibration_attempts[0].size(), 1u);
  EXPECT_TRUE(result.calibration_attempts[0][0].terminal);
  EXPECT_TRUE(result.calibration_attempts[0][0].valid);
  EXPECT_EQ(result.calibration_attempts[0][0].purpose,
            "frozen_measurement_warmup");
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
  FakeLlmBackend executor;
  executor.mutate = [&](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context,
                        size_t, LlmTaskExecutionResult& execution) {
    if (context.kind == LlmRunnerTaskKind::Warmup) {
      stop = true;
      fail_execution(execution);
    }
  };
  LlmRunnerHooks hooks;
  hooks.stop_requested = [&]() { return stop; };
  LlmMemoryResult result;

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor, result, hooks), EXIT_FAILURE);
  EXPECT_EQ(result.status, LlmRunStatus::Failed);
  EXPECT_EQ(result.reason_code, LlmBackendReason::RESOURCES_NOT_PREPARED);
  EXPECT_TRUE(result.interruption_requested);
  ASSERT_EQ(result.calibration_attempts[0].size(), 1u);
  EXPECT_TRUE(result.calibration_attempts[0][0].terminal);
  EXPECT_FALSE(result.calibration_attempts[0][0].valid);
  EXPECT_EQ(result.calibration_attempts[0][0].reason_code,
            LlmBackendReason::RESOURCES_NOT_PREPARED);
  EXPECT_EQ(result.calibration_attempts[0][0].work_units, config.iterations);
  EXPECT_TRUE(result.frozen_scenario_plans.valid);
  EXPECT_GT(result.counters.planned_work_units, 0u);
  EXPECT_GT(result.counters.planned_effective_model_payload_bytes, 0u);
  for (const LlmMeasurementState& measurement : result.measurements) {
    EXPECT_LT(measurement.frozen_plan_index, kLlmScenarioCount);
    EXPECT_EQ(measurement.planned_work_units, config.iterations);
    EXPECT_GT(measurement.planned_effective_model_payload_bytes, 0u);
  }
  expect_interrupted_tail(result, 0);
}

TEST(LlmMemoryRunnerTest, ExcludedExecutorExceptionRetainsTerminalAttemptBeforeCommandFailure) {
  const LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  FakeLlmBackend executor;
  executor.mutate = [](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&,
                       const LlmRunnerTaskContext& context, size_t,
                       LlmTaskExecutionResult&) {
    if (context.kind == LlmRunnerTaskKind::Warmup) {
      throw std::runtime_error("excluded executor failure");
    }
  };
  std::vector<CheckpointRecord> checkpoints;
  LlmMemoryResult result;

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor, result, recording_checkpoints(checkpoints)),
            EXIT_FAILURE);
  EXPECT_EQ(result.status, LlmRunStatus::Failed);
  EXPECT_EQ(result.reason_code, LlmRunnerReason::RUNNER_EXCEPTION);
  EXPECT_EQ(result.diagnostic, "excluded executor failure");
  ASSERT_EQ(result.calibration_attempts[0].size(), 1u);
  EXPECT_TRUE(result.calibration_attempts[0][0].terminal);
  EXPECT_FALSE(result.calibration_attempts[0][0].valid);
  EXPECT_EQ(result.calibration_attempts[0][0].reason_code,
            LlmRunnerReason::RUNNER_EXCEPTION);
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
    FakeLlmBackend executor;
    executor.mutate = [typed](const LlmMemoryWorkPlan&,
                              const LlmScenarioWorkPlan&,
                              const LlmRunnerTaskContext& context, size_t,
                              LlmTaskExecutionResult&) {
      if (context.kind == LlmRunnerTaskKind::Measurement) {
        if (typed) {
          throw std::runtime_error("injected executor failure");
        }
        throw 17;
      }
    };
    std::vector<CheckpointRecord> checkpoints;
    LlmMemoryResult result;
    EXPECT_EQ(run_llm_memory_suite(config, plan, executor, result, recording_checkpoints(checkpoints)),
              EXIT_FAILURE);
    EXPECT_EQ(result.status, LlmRunStatus::Failed);
    EXPECT_EQ(result.reason_code,
              typed ? LlmRunnerReason::RUNNER_EXCEPTION
                    : LlmRunnerReason::RUNNER_UNKNOWN_EXCEPTION);
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
  FakeLlmBackend executor;
  executor.mutate = [](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&,
                       const LlmRunnerTaskContext& context, size_t,
                       LlmTaskExecutionResult&) {
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

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor, result, hooks), EXIT_FAILURE);
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
  ASSERT_LT(kv_limits.effective_maximum_work_units, weights_limits.effective_maximum_work_units);
  config.iterations = kv_limits.effective_maximum_work_units + 1;
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  EXPECT_TRUE(build_llm_scenario_work_plan(plan, LlmScenario::WeightsOnly, config.iterations, true).valid);
  const LlmScenarioWorkPlan invalid_kv =
      build_llm_scenario_work_plan(plan, LlmScenario::KvOnly, config.iterations, true);
  ASSERT_FALSE(invalid_kv.valid);
  EXPECT_EQ(invalid_kv.reason_code, LlmWorkPlanReason::TASK_ACCOUNTED_BYTES_CAP_EXCEEDED);
  FakeLlmBackend executor;
  size_t checkpoint_calls = 0;
  LlmRunnerHooks hooks;
  hooks.stop_requested = []() { return false; };
  hooks.checkpoint = [&](const LlmMemoryResult&, LlmCheckpointKind) {
    ++checkpoint_calls;
    return EXIT_SUCCESS;
  };
  LlmMemoryResult result;

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor, result, hooks), EXIT_FAILURE);
  EXPECT_FALSE(result.initialized);
  EXPECT_EQ(result.reason_code, LlmWorkPlanReason::TASK_ACCOUNTED_BYTES_CAP_EXCEEDED);
  EXPECT_TRUE(executor.calls.empty());
  EXPECT_EQ(checkpoint_calls, 0u);
}

TEST(LlmMemoryRunnerTest,
     SelectorMismatchAndAuxiliaryOverflowFailBeforeInitializationOrExecutorCall) {
  const LlmMemoryConfig baseline_config = explicit_config();
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(baseline_config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  struct MismatchCase {
    void (*mutate)(LlmMemoryConfig&);
    const char* reason_code;
  };
  const std::array<MismatchCase, 3> mismatch_cases = {{
      {+[](LlmMemoryConfig& candidate) {
         candidate.backend = LlmMemoryBackend::Metal;
       },
       LlmRunnerReason::BACKEND_UNAVAILABLE},
      {+[](LlmMemoryConfig& candidate) {
         candidate.phase = LlmPhase::Prefill;
         candidate.visible_context_tokens = 0;
         candidate.prompt_tokens = 2;
         candidate.attention_query_tile_tokens = 1;
         candidate.user_specified_context_tokens = false;
         candidate.user_specified_prompt_tokens = true;
         candidate.user_specified_attention_query_tile_tokens = true;
       },
       LlmRunnerReason::CONFIG_WORK_PLAN_MISMATCH},
      {+[](LlmMemoryConfig& candidate) {
         candidate.kv_layout = LlmKvLayout::Paged;
         candidate.kv_block_tokens = 2;
         candidate.user_specified_kv_layout = true;
         candidate.user_specified_kv_block_tokens = true;
       },
       LlmRunnerReason::CONFIG_WORK_PLAN_MISMATCH},
  }};
  for (const MismatchCase& mismatch_case : mismatch_cases) {
    LlmMemoryConfig mismatched = baseline_config;
    mismatch_case.mutate(mismatched);
    FakeLlmBackend mismatch_executor;
    LlmMemoryResult mismatch_result;
    EXPECT_EQ(run_llm_memory_suite(mismatched, plan,
                                   mismatch_executor,
                                   mismatch_result),
              EXIT_FAILURE);
    EXPECT_FALSE(mismatch_result.initialized);
    EXPECT_EQ(mismatch_result.reason_code, mismatch_case.reason_code);
    EXPECT_TRUE(mismatch_executor.calls.empty());
    EXPECT_EQ(mismatch_executor.initialize_calls, 0u);
    EXPECT_EQ(mismatch_result.logical_checkpoint_attempts, 0u);
  }

  LlmMemoryConfig config = explicit_config();
  config.loop_count = std::numeric_limits<size_t>::max();
  const LlmRunnerAuxiliaryEstimate estimate = calculate_llm_runner_auxiliary_estimate(config, plan);
  EXPECT_FALSE(estimate.valid);
  EXPECT_EQ(estimate.reason_code, LlmRunnerReason::AUXILIARY_BYTES_OVERFLOW);
  FakeLlmBackend executor;
  LlmMemoryResult result;

  EXPECT_EQ(run_llm_memory_suite(config, plan, executor, result), EXIT_FAILURE);
  EXPECT_FALSE(result.initialized);
  EXPECT_EQ(result.reason_code, LlmRunnerReason::INVALID_CONFIG);
  EXPECT_TRUE(executor.calls.empty());
  EXPECT_EQ(executor.initialize_calls, 0u);
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
    const LlmCpuExecutionPlan* cpu_plan =
        get_llm_cpu_execution_plan(plan);
    ASSERT_NE(cpu_plan, nullptr);
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
              planned_measurements * cpu_plan->effective_workers * 2 *
                  sizeof(LlmWorkerChecksum));
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
  FakeLlmBackend exact_executor;
  LlmMemoryResult exact_result;
  EXPECT_EQ(run_llm_memory_suite(config, exact, exact_executor, exact_result), EXIT_SUCCESS);
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
    FakeLlmBackend rejected_executor;
    LlmMemoryResult rejected_result;
    EXPECT_EQ(run_llm_memory_suite(config, one_byte_short, rejected_executor, rejected_result),
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

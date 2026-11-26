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
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "core/config/constants.h"
#include "llm_memory/llm_cpu_backend.h"
#include "llm_memory/llm_metal_backend.h"
#include "llm_memory/llm_runner.h"
#include "utils/numeric_utils.h"

namespace {

const LlmScenarioWorkPlan& frozen_plan(const LlmMemoryResult& result, size_t index) {
  return result.scenario_plans.at(result.frozen_plan_handles.at(index)).plan;
}

const LlmScenarioWorkPlan& measurement_plan(const LlmMemoryResult& result, const LlmMeasurementState& measurement) {
  return result.scenario_plans.at(measurement.plan_handle).plan;
}

bool frozen_plans_valid(const LlmMemoryResult& result) {
  return std::all_of(result.frozen_plan_handles.begin(), result.frozen_plan_handles.end(),
                     [&](size_t handle) { return handle < result.scenario_plans.size(); });
}

LlmColdChecks fake_passed_checks(const LlmMemoryWorkPlan& model, LlmScenario scenario) {
  const bool kv = scenario != LlmScenario::WeightsOnly;
  const bool paged = model.kv_layout == LlmKvLayout::Paged;
  const bool prefill = model.phase == LlmPhase::Prefill;
  const size_t tokens = prefill ? model.geometry.prefill->prompt_tokens : model.geometry.decode->visible_context_tokens;
  const bool padding = paged && tokens % model.geometry.kv_block_tokens != 0;
  const bool cpu = model.backend == LlmMemoryBackend::Cpu;
  LlmColdChecks checks = {{{LlmColdCheckKind::PostValidationStructure, cpu ? paged || prefill || kv : kv},
                           {prefill ? LlmColdCheckKind::KvPrefillFinalSamples
                                    : (kv || !cpu ? LlmColdCheckKind::KvAppendFinal
                                                  : LlmColdCheckKind::KvAppendUnchanged),
                            kv || (cpu && paged && !prefill)},
                           {LlmColdCheckKind::KvPaddingCanary, padding && (cpu || kv)}}};
  for (auto& check : checks) {
    if (check.applicable) {
      check.evaluated = true;
      check.valid = true;
      check.reason_code = "valid";
    }
  }
  return checks;
}

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

LlmMemoryConfig make_metal_variant(LlmMemoryConfig config) {
  config.backend = LlmMemoryBackend::Metal;
  config.requested_workers = 0;
  config.available_workers = 0;
  config.user_specified_workers = false;
  config.user_specified_backend = true;
  return config;
}

LlmMemoryConfig make_paged_variant(LlmMemoryConfig config) {
  config.kv_layout = LlmKvLayout::Paged;
  config.kv_block_tokens = 2;
  config.user_specified_kv_layout = true;
  config.user_specified_kv_block_tokens = true;
  return config;
}

LlmMemoryConfig explicit_metal_config(size_t loop_count = 1, size_t iterations = 2) {
  return make_metal_variant(explicit_config(loop_count, iterations));
}

LlmMemoryConfig explicit_metal_paged_config(size_t loop_count = 1, size_t iterations = 2) {
  return make_paged_variant(explicit_metal_config(loop_count, iterations));
}

LlmMemoryConfig explicit_prefill_config(size_t loop_count = 2, size_t iterations = 3) {
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

LlmMemoryConfig explicit_metal_prefill_config(size_t loop_count = 1, size_t iterations = 1) {
  return make_metal_variant(explicit_prefill_config(loop_count, iterations));
}

LlmMemoryConfig explicit_metal_paged_prefill_config(size_t loop_count = 1, size_t iterations = 1) {
  return make_paged_variant(explicit_metal_prefill_config(loop_count, iterations));
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
  request.geometry.attention_query_tile_tokens = config.attention_query_tile_tokens;
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
  if (!draft.valid) return finalize_llm_memory_work_plan(std::move(draft), 0, 0);
  const auto executor = calculate_llm_executor_auxiliary_estimate(draft.auxiliary_preflight);
  const auto runner = calculate_llm_runner_auxiliary_estimate(config, draft.auxiliary_preflight);
  if (!executor.valid || !runner.valid ||
      !NumericUtils::checked_add(executor.checksum_auxiliary_bytes, runner.checksum_auxiliary_bytes,
                                 request.checksum_auxiliary_bytes) ||
      !NumericUtils::checked_add(executor.orchestration_auxiliary_bytes, runner.orchestration_auxiliary_bytes,
                                 request.orchestration_auxiliary_bytes)) {
    return finalize_llm_memory_work_plan(std::move(draft), 0, 0);
  }
  return finalize_llm_memory_work_plan(std::move(draft), request.checksum_auxiliary_bytes,
                                       request.orchestration_auxiliary_bytes);
}

LlmMetalResourcePlanRequest metal_resource_request(const LlmMemoryWorkPlan& logical_plan,
                                                   size_t command_auxiliary_bytes) {
  LlmMetalResourcePlanRequest request;
  request.geometry = logical_plan.geometry;
  if (logical_plan.kv_layout == LlmKvLayout::Paged) {
    const size_t tokens = logical_plan.phase == LlmPhase::Prefill
                              ? (logical_plan.geometry.prefill ? logical_plan.geometry.prefill->prompt_tokens : 0)
                              : (logical_plan.geometry.decode ? logical_plan.geometry.decode->visible_context_tokens
                                                              : 0);
    request.paged_layout = build_llm_kv_layout_plan(
        {tokens, logical_plan.geometry.kv_block_tokens, logical_plan.geometry.layer_count,
         logical_plan.geometry.batch_size, logical_plan.geometry.k_or_v_record_bytes_per_layer});
  }
  request.argument_buffer_encoded_length = 256;
  request.argument_buffer_alignment = 16;
  request.max_buffer_length = Constants::LLM_METAL_SEGMENT_CAPACITY_BYTES;
  request.available_memory_bytes = 8ULL * 1024ULL * Constants::BYTES_PER_MB;
  request.host_mapping_granularity_bytes = 1;
  EXPECT_TRUE(NumericUtils::checked_add(logical_plan.memory_budget.request.planner_storage_bytes,
                                        command_auxiliary_bytes, request.additional_owned_bytes));
  return request;
}

LlmMemoryWorkPlan build_metal_runner_plan(const LlmMemoryConfig& config, bool resolve_runtime_plan) {
  LlmMemoryWorkPlanDraft draft = prepare_llm_memory_work_plan(plan_request(config));
  if (!draft.valid) return finalize_llm_memory_work_plan(std::move(draft), 0, 0);
  if (resolve_runtime_plan) {
    LlmMetalExecutionPlan provisional = build_llm_metal_execution_plan(metal_resource_request(draft.candidate, 0));
    if (!attach_llm_metal_execution_plan(draft, std::move(provisional))) {
      return finalize_llm_memory_work_plan(std::move(draft), 0, 0);
    }
  }
  const auto runner = calculate_llm_runner_auxiliary_estimate(config, draft.auxiliary_preflight);
  if (!runner.valid) return finalize_llm_memory_work_plan(std::move(draft), 0, 0);
  if (!resolve_runtime_plan) {
    return finalize_llm_memory_work_plan(std::move(draft), runner.checksum_auxiliary_bytes,
                                         runner.orchestration_auxiliary_bytes);
  }
  size_t command_aux = 0;
  if (!NumericUtils::checked_add(runner.checksum_auxiliary_bytes, runner.orchestration_auxiliary_bytes, command_aux)) {
    return finalize_llm_memory_work_plan(std::move(draft), 0, 0);
  }
  LlmMetalExecutionPlan exact = build_llm_metal_execution_plan(metal_resource_request(draft.candidate, command_aux));
  return finalize_llm_memory_work_plan(std::move(draft), std::move(exact), runner.checksum_auxiliary_bytes,
                                       runner.orchestration_auxiliary_bytes);
}

LlmTaskExecutionResult successful_execution(const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan& task_plan,
                                            const LlmRunnerTaskContext& context, double elapsed_seconds) {
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
  result.completion.completed_effective_model_payload_bytes = task_plan.effective_model_payload_bytes;
  result.completion.completed_layout_metadata_lookup_count = task_plan.layout_metadata_lookup_count;
  result.completion.completed_layout_metadata_read_bytes = task_plan.layout_metadata_read_bytes;
  result.completion.completed_task_accounted_bytes = task_plan.task_accounted_bytes;
  result.validation.evaluated = true;
  result.validation.valid = true;
  if (model_plan.backend == LlmMemoryBackend::Cpu) {
    LlmExecutorResult cpu;
    const size_t workers = get_llm_cpu_execution_plan(model_plan)->effective_workers;
    cpu.valid = true;
    cpu.reason_code = LlmExecutorReason::VALID;
    cpu.requested_workers = workers;
    cpu.created_workers = workers;
    cpu.completed_workers = workers;
    cpu.qos_successful_workers = workers;
    cpu.kernel_succeeded = true;
    cpu.timer_started = true;
    cpu.timer_stopped = true;
    cpu.checksum_evaluated = true;
    cpu.checksum_valid = true;
    cpu.post_validation_evaluated = true;
    cpu.post_validation_valid = true;
    cpu.actual_checksums.resize(workers);
    cpu.expected_checksums.resize(workers);
    cpu.cold_checks = fake_passed_checks(model_plan, task_plan.scenario);
    result.backend_evidence = LlmCpuTaskEvidence{std::move(cpu)};
  }
  return result;
}

void fail_execution(LlmTaskExecutionResult& execution) {
  execution.status = LlmTaskExecutionStatus::Failed;
  execution.reason_code = LlmBackendReason::RESOURCES_NOT_PREPARED;
}

void invalidate_execution(LlmTaskExecutionResult& execution) { execution.validation.valid = false; }

class FakeLlmBackend final : public LlmBackend {
 public:
  std::vector<TaskRecord> calls;
  std::function<void(const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext&, size_t,
                     LlmTaskExecutionResult&)>
      mutate;
  LlmBackendLifecycleResult initialization = {LlmBackendStatus::Ready, LlmBackendReason::VALID};
  LlmBackendLifecycleResult plan_resolution = {LlmBackendStatus::Ready, LlmBackendReason::VALID};
  LlmBackendLifecycleResult preparation = {LlmBackendStatus::Ready, LlmBackendReason::VALID};
  LlmBackendLifecycleResult release = {LlmBackendStatus::Ready, LlmBackendReason::VALID};
  size_t initialize_calls = 0;
  size_t plan_resolution_calls = 0;
  size_t preparation_calls = 0;
  size_t release_calls = 0;
  mutable size_t expected_calls = 0;
  bool clear_initialization_reason_on_release = false;
  std::string initialization_reason_storage;
  size_t auxiliary_growth_after_initialize = 0;

  explicit FakeLlmBackend(LlmMemoryBackend backend = LlmMemoryBackend::Cpu) : backend_(backend) {
    evidence_.backend = backend_;
    if (backend_ == LlmMemoryBackend::Metal) evidence_.backend_evidence = LlmMetalBackendEvidence{};
  }

  LlmMemoryBackend kind() const noexcept override { return backend_; }

  LlmBackendAuxiliaryEstimate calculate_auxiliary_estimate(const LlmMemoryWorkPlan& model) const noexcept override {
    LlmBackendAuxiliaryEstimate estimate;
    if (backend_ == LlmMemoryBackend::Metal) {
      const bool valid_metal =
          model.backend == LlmMemoryBackend::Metal && get_llm_metal_execution_plan(model) != nullptr;
      estimate.valid = valid_metal;
      estimate.reason_code = valid_metal ? LlmBackendReason::VALID : LlmBackendReason::BACKEND_MISMATCH;
    } else if (model.phase == LlmPhase::Prefill) {
      estimate.valid = true;
      estimate.reason_code = LlmBackendReason::VALID;
      return estimate;
    } else {
      const auto cpu = calculate_llm_executor_auxiliary_estimate(model);
      estimate.valid = cpu.valid;
      estimate.reason_code = cpu.reason_code;
      estimate.checksum_auxiliary_bytes = cpu.checksum_auxiliary_bytes;
      estimate.orchestration_auxiliary_bytes = cpu.orchestration_auxiliary_bytes;
    }
    if (estimate.valid && initialize_calls != 0 &&
        !NumericUtils::checked_add(estimate.orchestration_auxiliary_bytes, auxiliary_growth_after_initialize,
                                   estimate.orchestration_auxiliary_bytes)) {
      estimate.valid = false;
      estimate.reason_code = LlmRunnerReason::AUXILIARY_BYTES_OVERFLOW;
      return estimate;
    }
    if (estimate.valid && !NumericUtils::checked_add(estimate.checksum_auxiliary_bytes,
                                                     estimate.orchestration_auxiliary_bytes,
                                                     estimate.total_auxiliary_bytes)) {
      estimate.valid = false;
      estimate.reason_code = LlmRunnerReason::AUXILIARY_BYTES_OVERFLOW;
    }
    return estimate;
  }

  LlmBackendLifecycleResult initialize(const LlmMemoryConfig&) noexcept override {
    ++initialize_calls;
    evidence_.initialization = initialization;
    return initialization;
  }

  LlmBackendLifecycleResult resolve_execution_plan(const LlmMemoryWorkPlan&) noexcept override {
    ++plan_resolution_calls;
    evidence_.plan_resolution = plan_resolution;
    return plan_resolution;
  }

  LlmBackendLifecycleResult prepare_resources(const LlmMemoryWorkPlan&) noexcept override {
    ++preparation_calls;
    evidence_.preparation = preparation;
    return preparation;
  }

  LlmExpectedChecksumResult expected_cpu_checksum(const LlmMemoryWorkPlan& model,
                                                  const LlmScenarioWorkPlan&) const noexcept override {
    ++expected_calls;
    LlmExpectedChecksumResult expected;
    expected.valid = true;
    expected.reason_code = LlmExecutorReason::VALID;
    expected.workers.resize(get_llm_cpu_execution_plan(model)->effective_workers);
    return expected;
  }

  LlmTaskExecutionResult execute_task(const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan& task_plan,
                                      const LlmRunnerTaskContext& context) override {
    calls.push_back({context, task_plan.work_units, task_plan.effective_model_payload_bytes, task_plan.plan_identity,
                     task_plan.work_unit_kind, task_plan.kv_write_kind});
    LlmTaskExecutionResult result = successful_execution(model_plan, task_plan, context, 0.150);
    if (backend_ == LlmMemoryBackend::Metal) {
      LlmMetalTaskEvidence metal;
      metal.cold_checks = fake_passed_checks(model_plan, task_plan.scenario);
      metal.timed_pipeline_available = true;
      metal.pipeline_label = std::string("membenchmark.llm-metal.pipeline.") +
                             (model_plan.phase == LlmPhase::Prefill ? "prefill-" : "decode-") +
                             (model_plan.kv_layout == LlmKvLayout::Paged ? "paged.fake" : "contiguous.fake");
      metal.pipeline_thread_execution_width = 32;
      metal.pipeline_max_total_threads_per_threadgroup = 256;
      metal.grid_plan_available = true;
      metal.grid_plan.valid = true;
      metal.grid_plan.reason_code = LlmMetalPlanReason::VALID;
      metal.grid_plan.threads_per_threadgroup = 256;
      metal.grid_plan.actual_threadgroups = 1;
      metal.grid_plan.work_units = task_plan.work_units;
      metal.grid_plan.paged_semantic_lookups = task_plan.layout_metadata_lookup_count;
      if (model_plan.phase == LlmPhase::Prefill &&
          (model_plan.kv_layout == LlmKvLayout::Contiguous || task_plan.scenario == LlmScenario::WeightsOnly)) {
        EXPECT_TRUE(calculate_llm_metal_prefill_serial_range_visits_per_lane(
            model_plan, task_plan, metal.grid_plan.serial_range_visits_per_lane));
      }
      metal.grid_plan.identity = "fake-metal-grid-v1";
      metal.timing_evaluated = true;
      metal.timing_valid = true;
      metal.gpu_start_seconds = 1.0;
      metal.gpu_end_seconds = 1.150;
      metal.gpu_elapsed_seconds = 0.150;
      metal.host_timing_evaluated = true;
      metal.host_submit_to_completion_seconds = 0.151;
      metal.host_wait_seconds = 0.150;
      metal.reset_command_buffer_count = 1;
      metal.timed_command_buffer_count = 1;
      metal.post_validation_command_buffer_count = 1;
      metal.timed_compute_encoder_count = 1;
      metal.timed_workload_dispatch_count = 1;
      metal.reset_command_status = "complete";
      metal.timed_command_status = "complete";
      metal.post_validation_command_status = "complete";
      metal.checksum_evaluated = true;
      metal.checksum_valid = true;
      metal.checksum_algorithm_version = model_plan.component_identities.checksum_pattern_version;
      metal.kv_write_validation_evaluated = task_plan.kv_write_kind != LlmKvWriteKind::None;
      metal.kv_write_validation_valid = true;
      metal.post_validation_evaluated = true;
      metal.post_validation_valid = true;
      result.backend_evidence = std::move(metal);
    }
    if (mutate) mutate(model_plan, task_plan, context, calls.size(), result);
    return result;
  }

  const LlmBackendEvidence& evidence() const noexcept override { return evidence_; }

  LlmBackendLifecycleResult release_resources() noexcept override {
    ++release_calls;
    if (clear_initialization_reason_on_release) initialization_reason_storage.clear();
    evidence_.release = release;
    return release;
  }

 private:
  LlmMemoryBackend backend_ = LlmMemoryBackend::Cpu;
  LlmBackendEvidence evidence_;
};

CheckpointRecord capture_checkpoint(const LlmMemoryResult& result, LlmCheckpointKind kind) {
  return {kind,
          result.status,
          result.counters.planned_loops,
          result.counters.attempted_loops,
          result.counters.completed_loops,
          result.counters.planned_measurements,
          result.counters.attempted_measurements,
          result.counters.terminal_measurements,
          result.counters.measured_measurements,
          result.logical_checkpoint_attempts,
          result.successful_logical_checkpoints,
          result.terminal_checkpoint_attempted,
          result.terminal_checkpoint_completed,
          result.interruption_requested};
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
    const bool is_measurement = record.context.kind == LlmRunnerTaskKind::Measurement;
    if (record.context.scenario == scenario && is_measurement == measurements) selected.push_back(record);
  }
  return selected;
}

void expect_lifecycle_calls(const FakeLlmBackend& backend, size_t init, size_t resolve, size_t prepare,
                            size_t release) {
  EXPECT_EQ(backend.initialize_calls, init);
  EXPECT_EQ(backend.plan_resolution_calls, resolve);
  EXPECT_EQ(backend.preparation_calls, prepare);
  EXPECT_EQ(backend.release_calls, release);
}

void expect_single_terminal_checkpoint(const std::vector<CheckpointRecord>& checkpoints, LlmRunStatus status) {
  ASSERT_EQ(checkpoints.size(), 1u);
  EXPECT_EQ(checkpoints.front().kind, LlmCheckpointKind::CommandTerminal);
  EXPECT_EQ(checkpoints.front().status, status);
  EXPECT_FALSE(checkpoints.front().terminal_checkpoint_completed);
}

void expect_interrupted_tail(const LlmMemoryResult& result, size_t measured_prefix) {
  ASSERT_LE(measured_prefix, result.measurements.size());
  for (size_t i = 0; i < result.measurements.size(); ++i) {
    const LlmMeasurementState& m = result.measurements[i];
    if (i < measured_prefix) {
      EXPECT_EQ(m.status, LlmMeasurementStatus::Measured) << i;
      EXPECT_TRUE(derive_llm_measurement_metrics(m).latency_seconds.has_value()) << i;
    } else {
      EXPECT_EQ(m.status, LlmMeasurementStatus::Interrupted) << i;
      EXPECT_EQ(m.reason_code, LlmRunnerReason::INTERRUPTION_BEFORE_TASK) << i;
      EXPECT_FALSE(derive_llm_measurement_metrics(m).latency_seconds.has_value()) << i;
      EXPECT_EQ(m.execution.completion.completed_work_units, 0u) << i;
    }
  }
}

void expect_frozen_measurement_plans_populated(const LlmMemoryResult& result, size_t expected_work_units) {
  EXPECT_TRUE(frozen_plans_valid(result));
  EXPECT_GT(result.counters.planned_work_units, 0u);
  EXPECT_GT(result.counters.planned_effective_model_payload_bytes, 0u);
  for (const LlmMeasurementState& m : result.measurements) {
    EXPECT_LT(static_cast<size_t>(m.scenario), kLlmScenarioCount);
    EXPECT_EQ(measurement_plan(result, m).work_units, expected_work_units);
    EXPECT_GT(measurement_plan(result, m).effective_model_payload_bytes, 0u);
  }
}

void expect_task_matches(const TaskRecord& task, LlmRunnerTaskKind kind, const char* purpose, LlmScenario scenario,
                         const LlmScenarioWorkPlan& expected) {
  EXPECT_EQ(task.context.kind, kind);
  EXPECT_EQ(task.context.purpose, purpose);
  EXPECT_EQ(task.context.scenario, scenario);
  EXPECT_EQ(task.work_units, expected.work_units);
  EXPECT_EQ(task.payload_bytes, expected.effective_model_payload_bytes);
  EXPECT_EQ(task.plan_identity, expected.plan_identity);
}

void expect_loop_and_measurement_counters(const LlmRunCounters& c, size_t p_loops, size_t a_loops, size_t c_loops,
                                          size_t p_meas, size_t a_meas, size_t t_meas, size_t m_meas) {
  EXPECT_EQ(c.planned_loops, p_loops);
  EXPECT_EQ(c.attempted_loops, a_loops);
  EXPECT_EQ(c.completed_loops, c_loops);
  EXPECT_EQ(c.planned_measurements, p_meas);
  EXPECT_EQ(c.attempted_measurements, a_meas);
  EXPECT_EQ(c.terminal_measurements, t_meas);
  EXPECT_EQ(c.measured_measurements, m_meas);
}

void expect_preflight_rejection(const LlmMemoryConfig& config, const LlmMemoryWorkPlan& plan,
                                const char* expected_reason) {
  FakeLlmBackend backend;
  LlmMemoryResult result;
  EXPECT_EQ(run_llm_memory_suite(config, plan, backend, result), EXIT_FAILURE);
  EXPECT_FALSE(result.initialized);
  if (expected_reason != nullptr) EXPECT_EQ(result.reason_code, expected_reason);
  EXPECT_TRUE(backend.calls.empty());
  EXPECT_EQ(backend.initialize_calls, 0u);
  EXPECT_EQ(result.logical_checkpoint_attempts, 0u);
}

}  // namespace

TEST(LlmMemoryRunnerTest, ExplicitFreezeWarmsCanonicalScenariosThenCompletesCyclicRun) {
  const LlmMemoryConfig config = explicit_config();
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  FakeLlmBackend executor;
  std::vector<CheckpointRecord> checkpoints;
  std::vector<bool> release_completed_at_checkpoint;
  LlmRunnerHooks hooks;
  hooks.stop_requested = []() { return false; };
  hooks.checkpoint = [&](const LlmMemoryResult& snapshot, LlmCheckpointKind kind) {
    checkpoints.push_back(capture_checkpoint(snapshot, kind));
    release_completed_at_checkpoint.push_back(executor.evidence().release.status == LlmBackendStatus::Ready);
    return EXIT_SUCCESS;
  };
  LlmMemoryResult result;

  ASSERT_EQ(run_llm_memory_suite(config, plan, executor, result, hooks), EXIT_SUCCESS);
  ASSERT_TRUE(result.initialized);
  EXPECT_EQ(result.status, LlmRunStatus::Complete);
  EXPECT_TRUE(result.results_complete);
  EXPECT_TRUE(result.run_accepted);
  EXPECT_TRUE(result.scenario_order_balance_complete);
  EXPECT_FALSE(result.interruption_requested);
  ASSERT_EQ(executor.calls.size(), 12u);
  for (size_t i = 0; i < kLlmScenarioCount; ++i) {
    const auto scenario = static_cast<LlmScenario>(i);
    const LlmScenarioWorkPlan& frozen = frozen_plan(result, i);
    EXPECT_EQ(frozen.work_units, config.iterations);
    expect_task_matches(executor.calls[i], LlmRunnerTaskKind::Warmup, "frozen_measurement_warmup", scenario, frozen);
    const auto measured = records_for_scenario(executor.calls, scenario, true);
    ASSERT_EQ(measured.size(), config.loop_count);
    for (const TaskRecord& record : measured) {
      expect_task_matches(record, LlmRunnerTaskKind::Measurement, "measurement", scenario, frozen);
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

  expect_loop_and_measurement_counters(result.counters, 3, 3, 3, 9, 9, 9, 9);
  EXPECT_EQ(result.counters.planned_work_units, 36u);
  EXPECT_EQ(result.counters.completed_work_units, 36u);
  EXPECT_EQ(result.counters.planned_layout_metadata_lookup_count, 0u);
  EXPECT_EQ(result.counters.completed_layout_metadata_lookup_count, 0u);
  EXPECT_EQ(result.counters.planned_layout_metadata_read_bytes, 0u);
  EXPECT_EQ(result.counters.completed_layout_metadata_read_bytes, 0u);
  size_t payload_per_loop = 0;
  for (size_t handle : result.frozen_plan_handles) {
    payload_per_loop += result.scenario_plans.at(handle).plan.effective_model_payload_bytes;
  }
  EXPECT_EQ(result.counters.planned_effective_model_payload_bytes, payload_per_loop * 3);
  EXPECT_EQ(result.counters.completed_effective_model_payload_bytes, payload_per_loop * 3);
  EXPECT_EQ(result.counters.planned_task_accounted_bytes, payload_per_loop * 3);
  EXPECT_EQ(result.counters.completed_task_accounted_bytes, payload_per_loop * 3);

  for (const LlmMeasurementState& m : result.measurements) {
    ASSERT_LT(static_cast<size_t>(m.scenario), kLlmScenarioCount);
    const LlmScenarioWorkPlan& frozen = frozen_plan(result, static_cast<size_t>(m.scenario));
    const LlmScenarioWorkPlan& mplan = measurement_plan(result, m);
    EXPECT_EQ(m.scenario, frozen.scenario);
    EXPECT_EQ(mplan.work_unit_kind, frozen.work_unit_kind);
    EXPECT_EQ(mplan.kv_write_kind, frozen.kv_write_kind);
    EXPECT_EQ(mplan.work_units, frozen.work_units);
    EXPECT_EQ(mplan.effective_model_payload_bytes, frozen.effective_model_payload_bytes);
    EXPECT_EQ(m.execution.completion.completed_effective_model_payload_bytes, frozen.effective_model_payload_bytes);
    EXPECT_EQ(m.execution.completion.completed_task_accounted_bytes, frozen.effective_model_payload_bytes);
    EXPECT_EQ(m.execution.status, LlmTaskExecutionStatus::Complete);
    EXPECT_EQ(m.execution.reason_code, LlmBackendReason::VALID);
    EXPECT_TRUE(m.execution.timing.valid);
    EXPECT_TRUE(m.execution.validation.valid);
    EXPECT_TRUE(std::holds_alternative<LlmCpuRetainedEvidence>(m.execution.backend_evidence));
  }

  ASSERT_EQ(checkpoints.size(), 4u);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(checkpoints[i].kind, LlmCheckpointKind::MeasurementTerminal);
    EXPECT_EQ(checkpoints[i].planned_loops, 3u);
    EXPECT_EQ(checkpoints[i].attempted_loops, i + 1);
    EXPECT_EQ(checkpoints[i].completed_loops, i + 1);
    EXPECT_EQ(checkpoints[i].planned_measurements, 9u);
    EXPECT_EQ(checkpoints[i].attempted_measurements, (i + 1) * 3);
    EXPECT_EQ(checkpoints[i].terminal_measurements, (i + 1) * 3);
    EXPECT_EQ(checkpoints[i].measured_measurements, (i + 1) * 3);
    EXPECT_EQ(checkpoints[i].logical_checkpoint_attempts, i + 1);
    EXPECT_EQ(checkpoints[i].successful_logical_checkpoints, i);
    EXPECT_FALSE(checkpoints[i].terminal_checkpoint_attempted);
    EXPECT_FALSE(checkpoints[i].terminal_checkpoint_completed);
    EXPECT_FALSE(checkpoints[i].interruption_requested);
    EXPECT_EQ(checkpoints[i].status, i == 2 ? LlmRunStatus::Complete : LlmRunStatus::Partial);
  }
  EXPECT_EQ(checkpoints.back().kind, LlmCheckpointKind::CommandTerminal);
  EXPECT_EQ(checkpoints.back().status, LlmRunStatus::Complete);
  EXPECT_EQ(checkpoints.back().attempted_measurements, 9u);
  EXPECT_EQ(checkpoints.back().terminal_measurements, 9u);
  EXPECT_EQ(checkpoints.back().measured_measurements, 9u);
  EXPECT_EQ(checkpoints.back().logical_checkpoint_attempts, 4u);
  EXPECT_EQ(checkpoints.back().successful_logical_checkpoints, 3u);
  EXPECT_TRUE(checkpoints.back().terminal_checkpoint_attempted);
  EXPECT_FALSE(checkpoints.back().terminal_checkpoint_completed);
  EXPECT_EQ(result.logical_checkpoint_attempts, 4u);
  EXPECT_EQ(result.successful_logical_checkpoints, 4u);
  EXPECT_TRUE(result.terminal_checkpoint_completed);
  EXPECT_EQ(executor.release_calls, 1u);
  ASSERT_EQ(release_completed_at_checkpoint.size(), checkpoints.size());
  for (size_t i = 0; i + 1 < release_completed_at_checkpoint.size(); ++i) {
    EXPECT_FALSE(release_completed_at_checkpoint[i]);
  }
  EXPECT_TRUE(release_completed_at_checkpoint.back());

  for (const LlmScenarioAggregate& aggregate : result.aggregates) {
    EXPECT_EQ(aggregate.status, "complete");
    EXPECT_EQ(aggregate.observed_cv_classification, "below-threshold");
    EXPECT_EQ(aggregate.accepted_measurement_ids.size(), 3u);
    EXPECT_TRUE(aggregate.effective_model_payload_gb_s.headline.has_value());
  }
}

TEST(LlmMemoryRunnerTest, PositionBalanceRetainsDirectedCarryoverAndPartialBlocks) {
  constexpr std::array<size_t, 21> expected = {0, 1, 2, 1, 2, 0, 2, 0, 1, 0, 1, 2, 1, 2, 0, 2, 0, 1, 0, 1, 2};
  for (size_t count : {1u, 2u, 3u, 4u, 6u, 7u}) {
    SCOPED_TRACE(count);
    const LlmMemoryConfig config = explicit_config(count);
    const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    FakeLlmBackend backend;
    LlmMemoryResult result;
    ASSERT_EQ(run_llm_memory_suite(config, plan, backend, result), EXIT_SUCCESS);
    EXPECT_TRUE(result.run_accepted);
    EXPECT_EQ(result.status, LlmRunStatus::Complete);
    EXPECT_TRUE(result.results_complete);
    if (count == 1) {
      EXPECT_EQ(result.quality_warnings, (std::vector<std::string_view>{"scenario-order-not-balanced"}));
      EXPECT_EQ(result.logical_checkpoint_attempts, 2u);
      for (const auto& aggregate : result.aggregates) {
        EXPECT_EQ(aggregate.observed_cv_classification, "insufficient-samples");
        ASSERT_EQ(aggregate.accepted_measurement_ids.size(), 1u);
        EXPECT_EQ(aggregate.effective_model_payload_gb_s.headline,
                  derive_llm_measurement_metrics(result.measurements.at(aggregate.accepted_measurement_ids.front()))
                      .payload_gb_s);
      }
    }
    EXPECT_EQ(result.scenario_order_balance_complete, count % 3 == 0);
    ASSERT_EQ(backend.calls.size(), 3 + 3 * count);
    std::array<std::array<size_t, 3>, 3> transitions{};
    for (size_t i = 0; i < 3; ++i) {
      EXPECT_EQ(backend.calls[i].context.kind, LlmRunnerTaskKind::Warmup);
      EXPECT_EQ(backend.calls[i].context.scenario, static_cast<LlmScenario>(i));
      EXPECT_EQ(backend.calls[i].work_units, config.iterations);
    }
    for (size_t i = 0; i < count * 3; ++i) {
      const TaskRecord& task = backend.calls[i + 3];
      EXPECT_EQ(task.context.kind, LlmRunnerTaskKind::Measurement);
      EXPECT_EQ(task.context.scenario, static_cast<LlmScenario>(expected[i]));
      EXPECT_EQ(task.context.loop_index, i / 3);
      EXPECT_EQ(task.context.order_position, i % 3);
      EXPECT_EQ(task.work_units, config.iterations);
      if (i != 0) {
        const size_t prev = static_cast<size_t>(backend.calls[i + 2].context.scenario);
        const size_t curr = static_cast<size_t>(task.context.scenario);
        ASSERT_LT(prev, 3u);
        ASSERT_LT(curr, 3u);
        ++transitions[prev][curr];
      }
    }
    if (count == 3) {
      EXPECT_EQ(transitions, (std::array<std::array<size_t, 3>, 3>{{{0, 2, 1}, {0, 0, 2}, {2, 1, 0}}}));
    }
    if (count == 6) {
      EXPECT_EQ(transitions, (std::array<std::array<size_t, 3>, 3>{{{0, 4, 2}, {1, 0, 4}, {4, 2, 0}}}));
    }
  }
}

TEST(LlmMemoryRunnerTest, ReadyProfilesReuseInitializationAndRetainTaggedTaskEvidence) {
  const LlmMemoryConfig cpu_paged = make_paged_variant(explicit_config(1));
  for (const auto& config : {explicit_metal_config(), explicit_metal_paged_config(), explicit_metal_prefill_config(),
                             explicit_metal_paged_prefill_config(), cpu_paged}) {
    SCOPED_TRACE(::testing::Message() << "backend=" << static_cast<size_t>(config.backend)
                                      << " phase=" << static_cast<size_t>(config.phase)
                                      << " layout=" << static_cast<size_t>(config.kv_layout));
    const bool metal_backend = config.backend == LlmMemoryBackend::Metal;
    const bool prefill = config.phase == LlmPhase::Prefill;
    const bool paged = config.kv_layout == LlmKvLayout::Paged;
    const auto plan = metal_backend ? build_metal_runner_plan(config, true) : build_runner_admitted_plan(config);
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    FakeLlmBackend backend(config.backend);
    ASSERT_EQ(backend.initialize(config).status, LlmBackendStatus::Ready);
    LlmMemoryResult result;
    ASSERT_EQ(run_llm_memory_suite(config, plan, backend, result), EXIT_SUCCESS);
    expect_lifecycle_calls(backend, 1, 1, 1, 1);
    EXPECT_EQ(result.status, LlmRunStatus::Complete);
    EXPECT_TRUE(result.results_complete);
    ASSERT_TRUE(frozen_plans_valid(result));
    ASSERT_EQ(result.measurements.size(), kLlmScenarioCount);
    size_t total_lookups = 0;
    for (const auto& m : result.measurements) {
      const bool weights_only = m.scenario == LlmScenario::WeightsOnly;
      const auto& task = measurement_plan(result, m);
      const size_t lookups = !paged || weights_only ? 0 : (prefill ? 15 : 3);
      EXPECT_EQ(m.status, LlmMeasurementStatus::Measured);
      EXPECT_TRUE(m.execution_evidence_available);
      EXPECT_EQ(task.work_units, config.iterations);
      EXPECT_EQ(m.execution.completion.completed_work_units, config.iterations);
      EXPECT_EQ(task.work_unit_kind, prefill ? LlmWorkUnitKind::PrefillOperation : LlmWorkUnitKind::DecodeStep);
      EXPECT_EQ(task.kv_write_kind, weights_only ? LlmKvWriteKind::None
                                                 : (prefill ? LlmKvWriteKind::FullPromptPopulation
                                                            : LlmKvWriteKind::CurrentTokenAppend));
      if (metal_backend) {
        EXPECT_EQ(task.layout_metadata_lookup_count_per_work_unit, lookups);
        EXPECT_EQ(task.layout_metadata_read_bytes_per_work_unit, lookups * sizeof(uint32_t));
        EXPECT_EQ(task.layout_metadata_lookup_count, lookups * config.iterations);
        EXPECT_EQ(m.execution.completion.completed_layout_metadata_lookup_count, lookups * config.iterations);
        total_lookups += lookups * config.iterations;
        const auto* metal = get_llm_metal_task_evidence(m.execution);
        ASSERT_NE(metal, nullptr);
        EXPECT_EQ(get_llm_cpu_task_evidence(m.execution), nullptr);
        EXPECT_TRUE(metal->timed_pipeline_available && metal->grid_plan_available && metal->grid_plan.valid);
        EXPECT_TRUE(metal->grid_plan.threadgroup_accounted_bytes.empty());
        EXPECT_TRUE(metal->checksum_evaluated && metal->checksum_valid);
        EXPECT_TRUE(metal->kv_write_validation_valid && metal->post_validation_valid);
        EXPECT_EQ(metal->pipeline_label, std::string("membenchmark.llm-metal.pipeline.") +
                                             (prefill ? "prefill-" : "decode-") +
                                             (paged ? "paged.fake" : "contiguous.fake"));
        if (prefill) EXPECT_EQ(metal->kv_write_validation_evaluated, !weights_only);
        if (prefill && paged) {
          EXPECT_EQ(metal->grid_plan.paged_semantic_lookups, lookups);
          EXPECT_EQ(metal->grid_plan.serial_range_visits_per_lane, weights_only ? 1u : 0u);
        }
      } else {
        const auto* cpu = std::get_if<LlmCpuRetainedEvidence>(&m.execution.backend_evidence);
        ASSERT_NE(cpu, nullptr);
        EXPECT_EQ(cpu->executor.reason_code, LlmExecutorReason::VALID);
        EXPECT_TRUE(cpu->executor.post_validation_evaluated && cpu->executor.post_validation_valid);
      }
    }
    if (metal_backend) {
      EXPECT_EQ(result.counters.planned_layout_metadata_lookup_count, total_lookups);
      EXPECT_EQ(result.counters.completed_layout_metadata_lookup_count, total_lookups);
      for (size_t i = 0; i < kLlmScenarioCount; ++i) {
        ASSERT_EQ(result.calibration_attempt_counts[i], 1u);
        const auto& compact = result.calibration_attempts[i][0].execution;
        EXPECT_FALSE(compact.cpu_evidence_available);
        EXPECT_TRUE(compact.metal_evidence_available);
        ASSERT_TRUE(compact.metal.has_value());
        EXPECT_TRUE(compact.metal->checksum_valid);
      }
    }
  }
}

TEST(LlmMemoryRunnerTest, MetalPreinitTerminalOutcomesAndAuxiliaryReadmission) {
  {
    SCOPED_TRACE("preinitialized_metal_terminal_outcomes");
    for (bool failed : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "failed=" << failed);
      const LlmMemoryConfig config = explicit_metal_config();
      LlmMemoryWorkPlan unresolved = build_metal_runner_plan(config, false);
      ASSERT_TRUE(unresolved.valid) << unresolved.reason_code;
      const auto* unresolved_exec = get_llm_metal_execution_plan(unresolved);
      ASSERT_NE(unresolved_exec, nullptr);
      ASSERT_FALSE(unresolved_exec->valid);
      unresolved.memory_budget.request.checksum_auxiliary_bytes = 0;
      unresolved.memory_budget.request.orchestration_auxiliary_bytes = 0;

      const auto expected_backend_status = failed ? LlmBackendStatus::Failed : LlmBackendStatus::Unsupported;
      const auto expected_run_status = failed ? LlmRunStatus::Failed : LlmRunStatus::Unsupported;
      const char* expected_reason =
          failed ? LlmBackendReason::METAL_KERNEL_COMPILATION_FAILED : LlmBackendReason::METAL_DEVICE_UNAVAILABLE;
      FakeLlmBackend backend(LlmMemoryBackend::Metal);
      backend.initialization = {expected_backend_status, expected_reason};
      ASSERT_EQ(backend.initialize(config).status, expected_backend_status);
      std::vector<CheckpointRecord> checkpoints;
      LlmMemoryResult result;
      EXPECT_EQ(run_llm_memory_suite(config, unresolved, backend, result, recording_checkpoints(checkpoints)),
                EXIT_FAILURE);
      ASSERT_TRUE(result.initialized);
      EXPECT_EQ(result.status, expected_run_status);
      EXPECT_EQ(result.reason_code, expected_reason);
      expect_lifecycle_calls(backend, 1, 0, 0, 1);
      EXPECT_TRUE(backend.calls.empty());
      expect_single_terminal_checkpoint(checkpoints, expected_run_status);

      // Unresolved Metal paged plans still determine padding-canary applicability from geometry.
      LlmMemoryConfig padded_metal = explicit_metal_paged_prefill_config();
      padded_metal.prompt_tokens = 5;
      padded_metal.kv_block_tokens = 4;
      const LlmMemoryWorkPlan unresolved_padded = build_metal_runner_plan(padded_metal, false);
      ASSERT_TRUE(unresolved_padded.valid) << unresolved_padded.reason_code;
      const auto* unresolved_padded_metal = get_llm_metal_execution_plan(unresolved_padded);
      ASSERT_NE(unresolved_padded_metal, nullptr);
      EXPECT_FALSE(unresolved_padded_metal->resources.paged_layout.has_value());
      EXPECT_FALSE(required_llm_cold_checks(unresolved_padded, LlmScenario::WeightsOnly)[2].applicable);
      EXPECT_TRUE(required_llm_cold_checks(unresolved_padded, LlmScenario::KvOnly)[2].applicable);
      EXPECT_TRUE(required_llm_cold_checks(unresolved_padded, LlmScenario::Mixed)[2].applicable);

      const LlmMemoryWorkPlan not_preinitialized = build_metal_runner_plan(config, false);
      ASSERT_TRUE(not_preinitialized.valid) << not_preinitialized.reason_code;
      FakeLlmBackend fresh_backend(LlmMemoryBackend::Metal);
      LlmMemoryResult rejected;
      EXPECT_EQ(run_llm_memory_suite(config, not_preinitialized, fresh_backend, rejected), EXIT_FAILURE);
      EXPECT_FALSE(rejected.initialized);
      EXPECT_EQ(rejected.reason_code, LlmRunnerReason::INVALID_MODEL_WORK_PLAN);
      expect_lifecycle_calls(fresh_backend, 0, 0, 0, 0);
    }
  }
  {
    SCOPED_TRACE("ready_metal_runtime_plan_failure");
    const LlmMemoryConfig config = explicit_metal_config();
    LlmMemoryWorkPlanDraft draft = prepare_llm_memory_work_plan(plan_request(config));
    ASSERT_TRUE(draft.valid) << draft.reason_code;
    LlmMetalResourcePlanRequest invalid_req = metal_resource_request(draft.candidate, 0);
    invalid_req.argument_buffer_encoded_length = 0;
    ASSERT_TRUE(attach_llm_metal_execution_plan(draft, build_llm_metal_execution_plan(invalid_req)));
    const auto aux = calculate_llm_runner_auxiliary_estimate(config, draft.auxiliary_preflight);
    ASSERT_TRUE(aux.valid) << aux.reason_code;
    size_t cmd_aux = 0;
    ASSERT_TRUE(NumericUtils::checked_add(aux.checksum_auxiliary_bytes, aux.orchestration_auxiliary_bytes, cmd_aux));
    ASSERT_TRUE(NumericUtils::checked_add(draft.candidate.memory_budget.request.planner_storage_bytes, cmd_aux,
                                          invalid_req.additional_owned_bytes));
    const LlmMemoryWorkPlan terminal_plan =
        finalize_llm_memory_work_plan(std::move(draft), build_llm_metal_execution_plan(invalid_req),
                                      aux.checksum_auxiliary_bytes, aux.orchestration_auxiliary_bytes);
    ASSERT_TRUE(terminal_plan.valid) << terminal_plan.reason_code;
    const auto* runtime_plan = get_llm_metal_execution_plan(terminal_plan);
    ASSERT_NE(runtime_plan, nullptr);
    ASSERT_FALSE(runtime_plan->valid);
    ASSERT_EQ(runtime_plan->reason_code, LlmMetalPlanReason::ARGUMENT_ENCODER_LENGTH_ZERO);

    FakeLlmBackend backend(LlmMemoryBackend::Metal);
    ASSERT_EQ(backend.initialize(config).status, LlmBackendStatus::Ready);
    std::vector<CheckpointRecord> checkpoints;
    LlmMemoryResult result;
    EXPECT_EQ(run_llm_memory_suite(config, terminal_plan, backend, result, recording_checkpoints(checkpoints)),
              EXIT_FAILURE);
    ASSERT_TRUE(result.initialized);
    EXPECT_EQ(result.status, LlmRunStatus::Failed);
    EXPECT_EQ(result.reason_code, LlmMetalPlanReason::ARGUMENT_ENCODER_LENGTH_ZERO);
    expect_lifecycle_calls(backend, 1, 0, 0, 1);
    EXPECT_TRUE(backend.calls.empty());
    expect_single_terminal_checkpoint(checkpoints, LlmRunStatus::Failed);
  }
  {
    SCOPED_TRACE("backend_auxiliary_growth_readmission");
    const LlmMemoryConfig config = explicit_config(1, 1);
    const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    FakeLlmBackend backend;
    backend.auxiliary_growth_after_initialize = 1;
    std::vector<CheckpointRecord> checkpoints;
    LlmMemoryResult result;

    EXPECT_EQ(run_llm_memory_suite(config, plan, backend, result, recording_checkpoints(checkpoints)), EXIT_FAILURE);
    ASSERT_TRUE(result.initialized);
    EXPECT_EQ(result.status, LlmRunStatus::Failed);
    EXPECT_EQ(result.reason_code, LlmRunnerReason::AUXILIARY_BUDGET_INSUFFICIENT);
    expect_lifecycle_calls(backend, 1, 0, 0, 1);
    EXPECT_TRUE(backend.calls.empty());
    expect_single_terminal_checkpoint(checkpoints, LlmRunStatus::Failed);
  }
}

TEST(LlmMemoryRunnerTest, TerminalLifecycleOutcomesAreCanonicalAndCheckpointedAtEveryPhase) {
  const auto config = explicit_config(1);
  const auto plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  for (bool failed : {false, true}) {
    for (size_t phase = 0; phase < 3; ++phase) {
      SCOPED_TRACE(::testing::Message() << "failed=" << failed << " phase=" << phase);
      FakeLlmBackend backend;
      backend.initialization_reason_storage = "backend-private-unsupported";
      const LlmBackendLifecycleResult terminal = {
          failed ? LlmBackendStatus::Failed : LlmBackendStatus::Unsupported,
          failed ? std::string_view(LlmBackendReason::BACKEND_INITIALIZATION_FAILED)
                 : std::string_view(backend.initialization_reason_storage)};
      if (phase == 0) {
        backend.initialization = terminal;
        backend.clear_initialization_reason_on_release = true;
      } else if (phase == 1) {
        backend.plan_resolution = terminal;
      } else {
        backend.preparation = terminal;
      }
      std::vector<CheckpointRecord> checkpoints;
      LlmMemoryResult result;
      EXPECT_EQ(run_llm_memory_suite(config, plan, backend, result, recording_checkpoints(checkpoints)), EXIT_FAILURE);
      EXPECT_TRUE(result.initialized);
      EXPECT_EQ(result.status, failed ? LlmRunStatus::Failed : LlmRunStatus::Unsupported);
      EXPECT_EQ(result.reason_code,
                failed ? LlmBackendReason::BACKEND_INITIALIZATION_FAILED : LlmBackendReason::TASK_UNSUPPORTED);
      EXPECT_TRUE(backend.calls.empty());
      expect_lifecycle_calls(backend, 1, phase == 0 ? 0 : 1, phase == 2 ? 1 : 0, 1);
      expect_single_terminal_checkpoint(checkpoints, result.status);
      EXPECT_TRUE(checkpoints.front().terminal_checkpoint_attempted);
      EXPECT_EQ(result.logical_checkpoint_attempts, 1u);
      EXPECT_EQ(result.successful_logical_checkpoints, 1u);
      EXPECT_TRUE(result.terminal_checkpoint_completed);
    }
  }
}

TEST(LlmMemoryRunnerTest, AutomaticCalibrationFreezesWithCorrectionTrialsAndTargetedWarmup) {
  constexpr std::array<LlmScenario, kLlmScenarioCount> kScenarios = {LlmScenario::WeightsOnly, LlmScenario::KvOnly,
                                                                     LlmScenario::Mixed};
  {
    SCOPED_TRACE("atomic_freeze_and_cyclic_measurements");
    const LlmMemoryConfig config = automatic_config(2);
    const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    FakeLlmBackend executor;
    executor.mutate = [](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context,
                         size_t, LlmTaskExecutionResult& result) {
      if (context.purpose == "pilot") {
        result.timing.elapsed_seconds = 0.010 * (static_cast<size_t>(context.scenario) + 1);
      } else if (context.kind == LlmRunnerTaskKind::Measurement) {
        result.timing.elapsed_seconds = 0.400;
      }
    };
    LlmMemoryResult result;

    ASSERT_EQ(run_llm_memory_suite(config, plan, executor, result), EXIT_SUCCESS);
    ASSERT_TRUE(frozen_plans_valid(result));
    ASSERT_EQ(executor.calls.size(), 18u);
    size_t call_index = 0;
    for (LlmScenario scenario : kScenarios) {
      const size_t idx = static_cast<size_t>(scenario);
      const TaskRecord& shape_warmup = executor.calls[call_index++];
      const TaskRecord& pilot = executor.calls[call_index++];
      const TaskRecord& correction = executor.calls[call_index++];
      EXPECT_EQ(shape_warmup.context.kind, LlmRunnerTaskKind::Warmup);
      EXPECT_EQ(shape_warmup.context.purpose, "calibration_shape_warmup");
      EXPECT_EQ(shape_warmup.context.scenario, scenario);
      EXPECT_EQ(pilot.context.kind, LlmRunnerTaskKind::Calibration);
      EXPECT_EQ(pilot.context.purpose, "pilot");
      EXPECT_EQ(pilot.context.scenario, scenario);
      EXPECT_EQ(shape_warmup.work_units, pilot.work_units);
      EXPECT_EQ(shape_warmup.payload_bytes, pilot.payload_bytes);
      EXPECT_EQ(shape_warmup.plan_identity, pilot.plan_identity);
      expect_task_matches(correction, LlmRunnerTaskKind::Calibration, "correction", scenario, frozen_plan(result, idx));
    }

    EXPECT_EQ(call_index, 3u * kLlmScenarioCount);
    for (LlmScenario scenario : kScenarios) {
      expect_task_matches(executor.calls[call_index++], LlmRunnerTaskKind::Warmup, "frozen_measurement_warmup",
                          scenario, frozen_plan(result, static_cast<size_t>(scenario)));
    }
    EXPECT_EQ(call_index, 4u * kLlmScenarioCount);
    for (size_t loop_idx = 0; loop_idx < config.loop_count; ++loop_idx) {
      const auto expected_order = build_llm_scenario_order(loop_idx);
      for (size_t pos = 0; pos < kLlmScenarioCount; ++pos) {
        const LlmScenario scenario = expected_order[pos];
        const TaskRecord& m = executor.calls[call_index++];
        expect_task_matches(m, LlmRunnerTaskKind::Measurement, "measurement", scenario,
                            frozen_plan(result, static_cast<size_t>(scenario)));
        EXPECT_EQ(m.context.loop_index, loop_idx);
        EXPECT_EQ(m.context.order_position, pos);
      }
    }
    EXPECT_EQ(call_index, executor.calls.size());
    for (size_t i = 0; i < kLlmScenarioCount; ++i) EXPECT_EQ(result.calibration_attempts[i].size(), 4u);
  }
  {
    SCOPED_TRACE("correction_trials_have_no_general_warmup");
    const LlmMemoryConfig config = automatic_config();
    const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    FakeLlmBackend executor;
    executor.mutate = [](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context,
                         size_t, LlmTaskExecutionResult& result) {
      if (context.purpose == "pilot") {
        result.timing.elapsed_seconds = 0.010;
      } else if (context.purpose == "correction") {
        result.timing.elapsed_seconds = context.attempt_index == 2 ? 0.050 : 0.150;
      }
    };
    LlmMemoryResult result;

    ASSERT_EQ(run_llm_memory_suite(config, plan, executor, result), EXIT_SUCCESS);
    const std::array<std::pair<LlmRunnerTaskKind, const char*>, 5> expected_seq = {{
        {LlmRunnerTaskKind::Warmup, "calibration_shape_warmup"},
        {LlmRunnerTaskKind::Calibration, "pilot"},
        {LlmRunnerTaskKind::Calibration, "correction"},
        {LlmRunnerTaskKind::Calibration, "correction"},
        {LlmRunnerTaskKind::Warmup, "frozen_measurement_warmup"},
    }};
    for (LlmScenario scenario : kScenarios) {
      const size_t idx = static_cast<size_t>(scenario);
      const auto excluded = records_for_scenario(executor.calls, scenario, false);
      ASSERT_EQ(excluded.size(), expected_seq.size());
      for (size_t i = 0; i < expected_seq.size(); ++i) {
        EXPECT_EQ(excluded[i].context.kind, expected_seq[i].first) << i;
        EXPECT_EQ(excluded[i].context.purpose, expected_seq[i].second) << i;
      }
      EXPECT_EQ(frozen_plan(result, idx).work_units, excluded[3].work_units);
      EXPECT_EQ(excluded[4].plan_identity, frozen_plan(result, idx).plan_identity);
      EXPECT_EQ(std::count_if(excluded.begin(), excluded.end(),
                              [](const TaskRecord& r) { return r.context.kind == LlmRunnerTaskKind::Warmup; }),
                2);
      EXPECT_EQ(std::count_if(excluded.begin(), excluded.end(),
                              [](const TaskRecord& r) {
                                return r.context.purpose == "single_unit_confirmation_warmup";
                              }),
                0);
    }
  }
}

TEST(LlmMemoryRunnerTest, SingleUnitConfirmationWarmsOnlyNewShapeAndFreezesAboveTargetEvidence) {
  for (bool metal : {false, true}) {
    SCOPED_TRACE(::testing::Message() << "metal=" << metal);
    LlmMemoryConfig config = metal ? explicit_metal_prefill_config() : automatic_prefill_config();
    config.iterations = 0;
    config.user_specified_iterations = false;
    config.weight_size_mb = 8;
    const LlmMemoryWorkPlan plan = metal ? build_metal_runner_plan(config, true) : build_runner_admitted_plan(config);
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    ASSERT_EQ(plan.phase, LlmPhase::Prefill);
    ASSERT_EQ(plan.work_unit_kind, LlmWorkUnitKind::PrefillOperation);
    const auto weights_limits = calculate_llm_scenario_limits(plan.geometry, LlmScenario::WeightsOnly);
    const auto kv_limits = calculate_llm_scenario_limits(plan.geometry, LlmScenario::KvOnly);
    const auto mixed_limits = calculate_llm_scenario_limits(plan.geometry, LlmScenario::Mixed);
    ASSERT_TRUE(weights_limits.valid && kv_limits.valid && mixed_limits.valid);
    EXPECT_EQ(calculate_llm_pilot_work_units(weights_limits), 1u);
    EXPECT_GT(calculate_llm_pilot_work_units(kv_limits), 1u);
    EXPECT_EQ(calculate_llm_pilot_work_units(mixed_limits), 1u);

    FakeLlmBackend executor(metal ? LlmMemoryBackend::Metal : LlmMemoryBackend::Cpu);
    executor.mutate = [](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan& task_plan,
                         const LlmRunnerTaskContext& context, size_t, LlmTaskExecutionResult& result) {
      if (context.purpose == "pilot") result.timing.elapsed_seconds = static_cast<double>(task_plan.work_units) * 0.300;
      else if (context.purpose == "single_unit_confirmation") result.timing.elapsed_seconds = 0.300;
      if (auto* evidence = std::get_if<LlmMetalTaskEvidence>(&result.backend_evidence)) {
        evidence->gpu_end_seconds = evidence->gpu_start_seconds + result.timing.elapsed_seconds;
        evidence->gpu_elapsed_seconds = result.timing.elapsed_seconds;
      }
    };
    LlmMemoryResult result;

    ASSERT_EQ(run_llm_memory_suite(config, plan, executor, result), EXIT_SUCCESS);
    ASSERT_TRUE(frozen_plans_valid(result));
    for (size_t handle : result.frozen_plan_handles) {
      const auto& frozen = result.scenario_plans.at(handle).plan;
      EXPECT_EQ(frozen.work_units, 1u);
      EXPECT_EQ(frozen.work_unit_kind, LlmWorkUnitKind::PrefillOperation);
      EXPECT_EQ(frozen.kv_write_kind, frozen.scenario == LlmScenario::WeightsOnly
                                          ? LlmKvWriteKind::None
                                          : LlmKvWriteKind::FullPromptPopulation);
    }

    const auto weights = records_for_scenario(executor.calls, LlmScenario::WeightsOnly, false);
    const auto kv = records_for_scenario(executor.calls, LlmScenario::KvOnly, false);
    const auto mixed = records_for_scenario(executor.calls, LlmScenario::Mixed, false);
    ASSERT_EQ(weights.size(), 3u);
    ASSERT_EQ(kv.size(), 5u);
    ASSERT_EQ(mixed.size(), 3u);
    for (const auto& simple : {weights, mixed}) {
      EXPECT_EQ(simple[0].context.purpose, "calibration_shape_warmup");
      EXPECT_EQ(simple[1].context.purpose, "pilot");
      EXPECT_EQ(simple[2].context.purpose, "frozen_measurement_warmup");
    }
    EXPECT_EQ(kv[0].context.purpose, "calibration_shape_warmup");
    EXPECT_EQ(kv[1].context.purpose, "pilot");
    EXPECT_EQ(kv[2].context.purpose, "single_unit_confirmation_warmup");
    EXPECT_EQ(kv[2].context.kind, LlmRunnerTaskKind::Warmup);
    EXPECT_EQ(kv[2].work_units, 1u);
    EXPECT_EQ(kv[3].context.purpose, "single_unit_confirmation");
    EXPECT_EQ(kv[3].context.kind, LlmRunnerTaskKind::Calibration);
    EXPECT_EQ(kv[3].work_units, 1u);
    EXPECT_EQ(kv[4].context.purpose, "frozen_measurement_warmup");

    ASSERT_EQ(result.calibration_attempts[0].size(), 3u);
    ASSERT_EQ(result.calibration_attempts[1].size(), 5u);
    ASSERT_EQ(result.calibration_attempts[2].size(), 3u);
    EXPECT_EQ(result.calibration_attempts[0][1].duration_quality, "above-target-single-work-unit");
    EXPECT_EQ(result.calibration_attempts[1][3].duration_quality, "above-target-single-work-unit");
    EXPECT_EQ(result.calibration_attempts[2][1].duration_quality, "above-target-single-work-unit");
    for (const TaskRecord& call : executor.calls) {
      EXPECT_EQ(call.work_unit_kind, LlmWorkUnitKind::PrefillOperation);
      EXPECT_EQ(call.kv_write_kind, call.context.scenario == LlmScenario::WeightsOnly
                                        ? LlmKvWriteKind::None
                                        : LlmKvWriteKind::FullPromptPopulation);
    }
  }
}

TEST(LlmMemoryRunnerTest, StopBeforeDuringOrAtCheckpointInterruptsRemainingTail) {
  const LlmMemoryConfig config = explicit_config(2);
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  {
    SCOPED_TRACE("stop_before_measurements");
    for (bool after_warmup : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "after_warmup=" << after_warmup);
      bool stop = !after_warmup;
      FakeLlmBackend executor;
      executor.mutate = [&](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context,
                            size_t, LlmTaskExecutionResult&) {
        if (context.kind == LlmRunnerTaskKind::Warmup) stop = true;
      };
      std::vector<CheckpointRecord> checkpoints;
      LlmRunnerHooks hooks = recording_checkpoints(checkpoints);
      hooks.stop_requested = [&]() { return stop; };
      LlmMemoryResult result;

      EXPECT_EQ(run_llm_memory_suite(config, plan, executor, result, hooks), EXIT_SUCCESS);
      EXPECT_EQ(executor.calls.size(), after_warmup ? 1u : 0u);
      if (after_warmup) {
        ASSERT_EQ(result.calibration_attempts[0].size(), 1u);
        EXPECT_TRUE(result.calibration_attempts[0][0].terminal);
        EXPECT_TRUE(result.calibration_attempts[0][0].valid);
        EXPECT_EQ(result.calibration_attempts[0][0].purpose, "frozen_measurement_warmup");
      }
      EXPECT_EQ(result.status, LlmRunStatus::Interrupted);
      EXPECT_TRUE(result.interruption_requested);
      expect_loop_and_measurement_counters(result.counters, 2, 0, 0, 6, 0, 6, 0);
      expect_frozen_measurement_plans_populated(result, config.iterations);
      expect_interrupted_tail(result, 0);
      expect_single_terminal_checkpoint(checkpoints, LlmRunStatus::Interrupted);
    }
  }
  {
    SCOPED_TRACE("stop_during_started_measurement");
    bool stop = false;
    FakeLlmBackend executor;
    executor.mutate = [&](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context,
                          size_t, LlmTaskExecutionResult&) {
      if (context.kind == LlmRunnerTaskKind::Measurement) stop = true;
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
    expect_loop_and_measurement_counters(result.counters, 2, 1, 0, 6, 1, 6, 1);
    EXPECT_EQ(result.counters.completed_work_units, config.iterations);
    EXPECT_EQ(result.counters.completed_effective_model_payload_bytes,
              measurement_plan(result, result.measurements[0]).effective_model_payload_bytes);
    expect_single_terminal_checkpoint(checkpoints, LlmRunStatus::Interrupted);
  }
  {
    SCOPED_TRACE("stop_raised_by_measurement_checkpoint");
    bool stop = false;
    FakeLlmBackend executor;
    std::vector<CheckpointRecord> checkpoints;
    LlmRunnerHooks hooks;
    hooks.stop_requested = [&]() { return stop; };
    hooks.checkpoint = [&](const LlmMemoryResult& snapshot, LlmCheckpointKind kind) {
      checkpoints.push_back(capture_checkpoint(snapshot, kind));
      if (kind == LlmCheckpointKind::MeasurementTerminal) stop = true;
      return EXIT_SUCCESS;
    };
    LlmMemoryResult result;

    EXPECT_EQ(run_llm_memory_suite(config, plan, executor, result, hooks), EXIT_SUCCESS);
    ASSERT_EQ(checkpoints.size(), 2u);
    EXPECT_EQ(checkpoints[0].kind, LlmCheckpointKind::MeasurementTerminal);
    EXPECT_EQ(checkpoints[0].status, LlmRunStatus::Partial);
    EXPECT_EQ(checkpoints[1].kind, LlmCheckpointKind::CommandTerminal);
    EXPECT_EQ(checkpoints[1].status, LlmRunStatus::Interrupted);
    EXPECT_EQ(result.counters.attempted_measurements, 3u);
    expect_interrupted_tail(result, 3);
  }
}

TEST(LlmMemoryRunnerTest, StopAndStopExceptionAfterLastMeasurement) {
  const LlmMemoryConfig config = explicit_config();
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  for (bool throw_exception : {false, true}) {
    SCOPED_TRACE(::testing::Message() << "throw_exception=" << throw_exception);
    bool trigger = false;
    FakeLlmBackend executor;
    std::vector<CheckpointRecord> checkpoints;
    LlmRunnerHooks hooks;
    hooks.stop_requested = [&]() {
      if (trigger && throw_exception) throw std::runtime_error("injected stop failure");
      return trigger;
    };
    hooks.checkpoint = [&](const LlmMemoryResult& snapshot, LlmCheckpointKind kind) {
      checkpoints.push_back(capture_checkpoint(snapshot, kind));
      if (kind == LlmCheckpointKind::MeasurementTerminal && snapshot.counters.measured_measurements == 9) {
        trigger = true;
      }
      return EXIT_SUCCESS;
    };
    LlmMemoryResult result;

    EXPECT_EQ(run_llm_memory_suite(config, plan, executor, result, hooks),
              throw_exception ? EXIT_FAILURE : EXIT_SUCCESS);
    EXPECT_TRUE(result.results_complete);
    EXPECT_TRUE(result.scenario_order_balance_complete);
    EXPECT_EQ(result.counters.measured_measurements, 9u);
    ASSERT_EQ(checkpoints.size(), 4u);
    EXPECT_EQ(checkpoints[2].kind, LlmCheckpointKind::MeasurementTerminal);
    EXPECT_EQ(checkpoints[2].measured_measurements, 9u);
    EXPECT_EQ(checkpoints.back().kind, LlmCheckpointKind::CommandTerminal);
    if (throw_exception) {
      EXPECT_EQ(result.status, LlmRunStatus::Failed);
      EXPECT_EQ(result.reason_code, LlmRunnerReason::RUNNER_EXCEPTION);
      EXPECT_EQ(result.diagnostic, "injected stop failure");
      EXPECT_FALSE(result.run_accepted);
      EXPECT_EQ(checkpoints[2].status, LlmRunStatus::Complete);
      EXPECT_EQ(checkpoints.back().status, LlmRunStatus::Failed);
    } else {
      EXPECT_TRUE(result.interruption_requested);
      EXPECT_EQ(result.status, LlmRunStatus::Complete);
      EXPECT_TRUE(result.run_accepted);
      EXPECT_EQ(result.logical_checkpoint_attempts, 4u);
      EXPECT_FALSE(checkpoints[2].interruption_requested);
      EXPECT_TRUE(checkpoints.back().interruption_requested);
    }
  }
}

TEST(LlmMemoryRunnerTest, BackendAndValidationFailuresWinSimultaneousStopAndPreserveInterruptedTail) {
  for (bool validation_failure : {false, true}) {
    SCOPED_TRACE(::testing::Message() << "validation_failure=" << validation_failure);
    const LlmMemoryConfig config = explicit_config(2);
    const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    bool stop = false;
    FakeLlmBackend executor;
    executor.mutate = [&](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context,
                          size_t, LlmTaskExecutionResult& execution) {
      if (context.kind == LlmRunnerTaskKind::Measurement) {
        stop = true;
        if (validation_failure) invalidate_execution(execution);
        else fail_execution(execution);
      }
    };
    LlmRunnerHooks hooks;
    hooks.stop_requested = [&]() { return stop; };
    LlmMemoryResult result;

    EXPECT_EQ(run_llm_memory_suite(config, plan, executor, result, hooks), EXIT_FAILURE);
    const char* reason =
        validation_failure ? LlmBackendReason::VALIDATION_FAILED : LlmBackendReason::RESOURCES_NOT_PREPARED;
    EXPECT_EQ(result.status, LlmRunStatus::Failed);
    EXPECT_EQ(result.reason_code, reason);
    EXPECT_TRUE(result.interruption_requested);
    ASSERT_EQ(result.measurements[0].status,
              validation_failure ? LlmMeasurementStatus::Invalid : LlmMeasurementStatus::Failed);
    EXPECT_EQ(result.measurements[0].reason_code, reason);
    EXPECT_FALSE(derive_llm_measurement_metrics(result.measurements[0]).latency_seconds.has_value());
    EXPECT_TRUE(result.aggregates[0].accepted_measurement_ids.empty());
    for (size_t i = 1; i < result.measurements.size(); ++i) {
      EXPECT_EQ(result.measurements[i].status, LlmMeasurementStatus::Interrupted);
    }
  }
}

TEST(LlmMemoryRunnerTest, CheckpointReturnFailuresAreTerminalWithoutRetry) {
  {
    SCOPED_TRACE("measurement_checkpoint_failure");
    const LlmMemoryConfig config = explicit_config(2);
    const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    for (bool request_stop : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "request_stop=" << request_stop);
      bool stop = false;
      FakeLlmBackend executor;
      std::vector<LlmCheckpointKind> kinds;
      LlmRunnerHooks hooks;
      hooks.stop_requested = [&]() { return stop; };
      hooks.checkpoint = [&](const LlmMemoryResult&, LlmCheckpointKind kind) {
        kinds.push_back(kind);
        stop = request_stop;
        return EXIT_FAILURE;
      };
      LlmMemoryResult result;

      EXPECT_EQ(run_llm_memory_suite(config, plan, executor, result, hooks), EXIT_FAILURE);
      EXPECT_EQ(kinds, (std::vector<LlmCheckpointKind>{LlmCheckpointKind::MeasurementTerminal}));
      EXPECT_EQ(result.status, LlmRunStatus::Failed);
      EXPECT_EQ(result.reason_code, LlmRunnerReason::CHECKPOINT_WRITE_FAILED);
      EXPECT_TRUE(result.checkpoint_failed);
      EXPECT_EQ(result.interruption_requested, request_stop);
      EXPECT_FALSE(result.terminal_checkpoint_attempted);
      const auto& m0 = result.measurements[0];
      const auto metrics0 = derive_llm_measurement_metrics(m0);
      EXPECT_EQ(m0.status, LlmMeasurementStatus::Measured);
      EXPECT_TRUE(metrics0.latency_seconds && metrics0.work_units_per_second && metrics0.payload_gb_s);
      EXPECT_TRUE(get_llm_cpu_task_evidence(m0.execution)->checksum_valid);
      EXPECT_EQ(m0.execution.status, LlmTaskExecutionStatus::Complete);
      EXPECT_TRUE(m0.execution.timing.valid && m0.execution.validation.valid);
      EXPECT_TRUE(std::holds_alternative<LlmCpuRetainedEvidence>(m0.execution.backend_evidence));
      EXPECT_EQ(result.aggregates[0].accepted_measurement_ids.size(), 1u);
      expect_loop_and_measurement_counters(result.counters, 2, 1, 1, 6, 3, 6, 3);
      EXPECT_EQ(result.counters.completed_work_units, 3 * config.iterations);
      EXPECT_EQ(result.counters.completed_effective_model_payload_bytes,
                measurement_plan(result, result.measurements[0]).effective_model_payload_bytes +
                    measurement_plan(result, result.measurements[1]).effective_model_payload_bytes +
                    measurement_plan(result, result.measurements[2]).effective_model_payload_bytes);
      for (size_t i = 3; i < result.measurements.size(); ++i) {
        EXPECT_EQ(result.measurements[i].status,
                  request_stop ? LlmMeasurementStatus::Interrupted : LlmMeasurementStatus::Failed);
        if (request_stop) EXPECT_EQ(result.measurements[i].reason_code, LlmRunnerReason::INTERRUPTION_BEFORE_TASK);
      }
    }
  }
  {
    SCOPED_TRACE("command_terminal_checkpoint_failure");
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
    ASSERT_EQ(kinds.size(), 2u);
    EXPECT_EQ(kinds.back(), LlmCheckpointKind::CommandTerminal);
    EXPECT_EQ(result.logical_checkpoint_attempts, 2u);
    EXPECT_EQ(result.successful_logical_checkpoints, 1u);
    EXPECT_TRUE(result.terminal_checkpoint_attempted);
    EXPECT_FALSE(result.terminal_checkpoint_completed);
    EXPECT_EQ(result.status, LlmRunStatus::Failed);
    EXPECT_EQ(result.reason_code, LlmRunnerReason::CHECKPOINT_WRITE_FAILED);
    EXPECT_TRUE(result.results_complete);
    EXPECT_FALSE(result.run_accepted);
    EXPECT_EQ(result.counters.measured_measurements, 3u);
  }
}

TEST(LlmMemoryRunnerTest, CheckpointExceptionsAndReturnOverridePrecedence) {
  {
    SCOPED_TRACE("checkpoint_exceptions");
    const LlmMemoryConfig config = explicit_config(2);
    const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    for (bool typed : {true, false}) {
      SCOPED_TRACE(typed ? "typed" : "unknown");
      FakeLlmBackend executor;
      std::vector<LlmCheckpointKind> kinds;
      LlmRunnerHooks hooks;
      hooks.stop_requested = []() { return false; };
      hooks.checkpoint = [&](const LlmMemoryResult&, LlmCheckpointKind kind) -> int {
        kinds.push_back(kind);
        if (typed) throw std::runtime_error("injected checkpoint failure");
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
      for (size_t i = 3; i < result.measurements.size(); ++i) {
        EXPECT_EQ(result.measurements[i].status, LlmMeasurementStatus::Failed) << i;
      }
      if (typed) EXPECT_EQ(result.diagnostic, "injected checkpoint failure");
    }
  }
  {
    SCOPED_TRACE("checkpoint_return_overrides_executor_exception");
    const LlmMemoryConfig config = explicit_config(1);
    const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    FakeLlmBackend executor;
    executor.mutate = [](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context,
                         size_t, LlmTaskExecutionResult&) {
      if (context.kind == LlmRunnerTaskKind::Measurement) throw std::runtime_error("earlier executor diagnostic");
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
    EXPECT_EQ(kinds, (std::vector<LlmCheckpointKind>{LlmCheckpointKind::CommandTerminal}));
    EXPECT_EQ(result.status, LlmRunStatus::Failed);
    EXPECT_EQ(result.reason_code, LlmRunnerReason::CHECKPOINT_WRITE_FAILED);
    EXPECT_TRUE(result.diagnostic.empty());
    EXPECT_TRUE(result.checkpoint_failed);
    EXPECT_EQ(result.logical_checkpoint_attempts, 1u);
    EXPECT_EQ(result.successful_logical_checkpoints, 0u);
    EXPECT_TRUE(result.terminal_checkpoint_attempted);
  }
}

TEST(LlmMemoryRunnerTest, CanonicalResultReasonsDetachAndSurviveCopiesMoves) {
  {
    SCOPED_TRACE("detach_from_owning_domains");
    for (const auto* expected :
         {LlmRunnerReason::CHECKPOINT_WRITE_FAILED, LlmWorkPlanReason::TASK_ACCOUNTED_BYTES_CAP_EXCEEDED,
          LlmExecutorReason::KERNEL_FAILED, LlmWorkPlanReason::KV_BLOCK_TOKENS_NOT_APPLICABLE,
          LlmExecutorReason::PREFILL_POST_VALIDATION_FAILED, LlmExecutorReason::PAGED_POST_VALIDATION_FAILED,
          LlmExecutorReason::DECODE_POST_VALIDATION_FAILED, LlmBackendReason::KV_WRITE_VALIDATION_MISMATCH}) {
      SCOPED_TRACE(expected);
      std::string owned(expected);
      const auto canonical = canonicalize_llm_result_reason_code(owned);
      std::fill(owned.begin(), owned.end(), 'x');
      EXPECT_EQ(canonical, expected);
      EXPECT_EQ(canonical.data(), canonicalize_llm_result_reason_code(expected).data());
    }
    EXPECT_EQ(canonicalize_llm_result_reason_code("not-a-reason"), LlmRunnerReason::RUNNER_UNKNOWN_EXCEPTION);
  }
  {
    SCOPED_TRACE("copies_and_moves_retain_static_reasons");
    const LlmMemoryConfig config = explicit_config(2);
    const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    FakeLlmBackend executor;
    executor.mutate = [](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context,
                         size_t, LlmTaskExecutionResult&) {
      if (context.kind == LlmRunnerTaskKind::Measurement) throw std::runtime_error("injected runner failure");
    };
    LlmMemoryResult result;

    EXPECT_EQ(run_llm_memory_suite(config, plan, executor, result), EXIT_FAILURE);
    ASSERT_EQ(result.measurements.size(), config.loop_count * kLlmScenarioCount);
    EXPECT_EQ(result.measurements[0].reason_code, LlmRunnerReason::RUNNER_EXCEPTION);

    LlmMemoryResult copied = result;
    LlmMemoryResult moved = std::move(copied);
    const auto canonical_exc = canonicalize_llm_result_reason_code(LlmRunnerReason::RUNNER_EXCEPTION);
    const auto canonical_not_run = canonicalize_llm_result_reason_code(LlmRunnerReason::NOT_RUN_AFTER_RUNTIME_FAILURE);
    EXPECT_EQ(moved.measurements[0].reason_code.data(), canonical_exc.data());
    result = LlmMemoryResult{};
    copied = LlmMemoryResult{};

    EXPECT_EQ(moved.measurements[0].reason_code, LlmRunnerReason::RUNNER_EXCEPTION);
    EXPECT_EQ(moved.measurements[0].reason_code.data(), canonical_exc.data());
    for (size_t i = 1; i < moved.measurements.size(); ++i) {
      EXPECT_EQ(moved.measurements[i].reason_code, LlmRunnerReason::NOT_RUN_AFTER_RUNTIME_FAILURE);
      EXPECT_EQ(moved.measurements[i].reason_code.data(), canonical_not_run.data());
    }
  }
}

TEST(LlmMemoryRunnerTest, MeasuredOnlyAggregatesClassifyCvAndExcludeInvalidValues) {
  const LlmMemoryConfig config = explicit_config();
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  {
    SCOPED_TRACE("cv_threshold_classification");
    constexpr std::array<double, 3> stable_values = {95.0, 100.0, 105.0};
    constexpr std::array<double, 3> noisy_values = {90.0, 100.0, 110.0};
    FakeLlmBackend executor;
    executor.mutate = [&](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan& task_plan,
                          const LlmRunnerTaskContext& context, size_t, LlmTaskExecutionResult& execution) {
      if (context.kind != LlmRunnerTaskKind::Measurement) return;
      const double gb_s = context.scenario == LlmScenario::WeightsOnly ? stable_values[context.loop_index]
                                                                       : noisy_values[context.loop_index];
      execution.timing.elapsed_seconds = static_cast<double>(task_plan.effective_model_payload_bytes) / gb_s / 1.0e9;
    };
    LlmMemoryResult result;

    ASSERT_EQ(run_llm_memory_suite(config, plan, executor, result), EXIT_SUCCESS);
    const LlmScenarioAggregate& weights = result.aggregates[0];
    EXPECT_EQ(weights.accepted_measurement_ids.size(), 3u);
    EXPECT_NEAR(weights.effective_model_payload_gb_s.statistics.coefficient_of_variation_pct, 5.0, 1e-10);
    EXPECT_EQ(weights.observed_cv_classification, "below-threshold");
    EXPECT_DOUBLE_EQ(*weights.effective_model_payload_gb_s.headline, 100.0);

    for (size_t i : {1u, 2u}) {
      const LlmScenarioAggregate& agg = result.aggregates[i];
      EXPECT_EQ(agg.accepted_measurement_ids.size(), 3u);
      EXPECT_GT(agg.effective_model_payload_gb_s.statistics.coefficient_of_variation_pct,
                Constants::LLM_STREAMING_CV_WARNING_PCT);
      EXPECT_EQ(agg.observed_cv_classification, "above-threshold");
    }
    EXPECT_EQ(result.quality_warnings, (std::vector<std::string_view>{"kv_only-high-cv", "mixed-high-cv"}));
    EXPECT_TRUE(result.results_complete && result.run_accepted);
  }
  {
    SCOPED_TRACE("partial_aggregates_exclude_invalid");
    size_t weights_measurements = 0;
    FakeLlmBackend executor;
    executor.mutate = [&](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan& task_plan,
                          const LlmRunnerTaskContext& context, size_t, LlmTaskExecutionResult& execution) {
      if (context.kind != LlmRunnerTaskKind::Measurement || context.scenario != LlmScenario::WeightsOnly) return;
      ++weights_measurements;
      execution.timing.elapsed_seconds = static_cast<double>(task_plan.effective_model_payload_bytes) / 100.0 / 1.0e9;
      if (weights_measurements == 2) invalidate_execution(execution);
    };
    LlmMemoryResult result;

    EXPECT_EQ(run_llm_memory_suite(config, plan, executor, result), EXIT_FAILURE);
    EXPECT_EQ(result.status, LlmRunStatus::Failed);
    EXPECT_EQ(result.reason_code, LlmBackendReason::VALIDATION_FAILED);
    ASSERT_EQ(result.aggregates[0].accepted_measurement_ids.size(), 1u);
    EXPECT_NEAR(
        *derive_llm_measurement_metrics(result.measurements.at(result.aggregates[0].accepted_measurement_ids.front()))
             .payload_gb_s,
        100.0, 1e-10);
    EXPECT_EQ(result.aggregates[0].status, "partial");
    EXPECT_EQ(result.aggregates[0].observed_cv_classification, "insufficient-samples");
    EXPECT_EQ(result.counters.measured_measurements, 5u);
    EXPECT_EQ(result.counters.attempted_measurements, 6u);
    EXPECT_EQ(result.measurements[5].status, LlmMeasurementStatus::Invalid);
    EXPECT_FALSE(derive_llm_measurement_metrics(result.measurements[5]).payload_gb_s.has_value());
  }
}

TEST(LlmMemoryRunnerTest, InvalidTimingAndCompletionMismatchRejectedInMeasuredAndExcludedTasks) {
  const LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  {
    SCOPED_TRACE("measured_invalid_timing_and_completion");
    for (bool completion_mismatch : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "completion_mismatch=" << completion_mismatch);
      FakeLlmBackend executor;
      executor.mutate = [completion_mismatch](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&,
                                              const LlmRunnerTaskContext& context, size_t,
                                              LlmTaskExecutionResult& execution) {
        if (context.kind != LlmRunnerTaskKind::Measurement) return;
        if (completion_mismatch) {
          --execution.completion.completed_work_units;
        } else {
          execution.timing.valid = false;
          execution.timing.elapsed_seconds = 0.0;
        }
      };
      LlmMemoryResult result;

      EXPECT_EQ(run_llm_memory_suite(config, plan, executor, result), EXIT_FAILURE);
      const char* reason = completion_mismatch ? LlmBackendReason::TASK_COMPLETION_MISMATCH
                                               : LlmBackendReason::INVALID_AUTHORITATIVE_ELAPSED;
      EXPECT_EQ(result.status, LlmRunStatus::Failed);
      EXPECT_EQ(result.reason_code, reason);
      ASSERT_FALSE(result.measurements.empty());
      EXPECT_EQ(result.measurements[0].status,
                completion_mismatch ? LlmMeasurementStatus::Failed : LlmMeasurementStatus::Invalid);
      EXPECT_EQ(result.measurements[0].reason_code, reason);
      EXPECT_FALSE(derive_llm_measurement_metrics(result.measurements[0]).latency_seconds.has_value());
      EXPECT_TRUE(result.aggregates[0].accepted_measurement_ids.empty());
      EXPECT_TRUE(std::holds_alternative<LlmCpuRetainedEvidence>(result.measurements[0].execution.backend_evidence));
      EXPECT_EQ(result.counters.measured_measurements, 0u);
    }
  }
  {
    SCOPED_TRACE("excluded_completion_mismatch");
    FakeLlmBackend executor;
    executor.mutate = [](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context,
                         size_t, LlmTaskExecutionResult& execution) {
      if (context.kind == LlmRunnerTaskKind::Warmup) --execution.completion.completed_task_accounted_bytes;
    };
    LlmMemoryResult result;

    EXPECT_EQ(run_llm_memory_suite(config, plan, executor, result), EXIT_FAILURE);
    EXPECT_EQ(result.status, LlmRunStatus::Failed);
    EXPECT_EQ(result.reason_code, LlmBackendReason::TASK_COMPLETION_MISMATCH);
    ASSERT_EQ(result.calibration_attempts[0].size(), 1u);
    EXPECT_FALSE(result.calibration_attempts[0][0].valid);
    EXPECT_EQ(result.calibration_attempts[0][0].reason_code, LlmBackendReason::TASK_COMPLETION_MISMATCH);
  }
}

TEST(LlmMemoryRunnerTest, CalibrationAndExcludedTaskFailuresFinalizeSlotsAndWinStop) {
  {
    SCOPED_TRACE("calibration_failure_finalizes_slots");
    const LlmMemoryConfig config = automatic_config(2);
    const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    FakeLlmBackend executor;
    executor.mutate = [](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context,
                         size_t, LlmTaskExecutionResult& execution) {
      if (context.purpose == "pilot") {
        fail_execution(execution);
        LlmCpuTaskEvidence cold;
        cold.executor.cold_checks = make_llm_cold_checks(true, LlmColdCheckKind::KvAppendFinal, true, true);
        resolve_llm_cold_check(cold.executor.cold_checks, 2, false, LlmExecutorReason::PAGED_POST_VALIDATION_FAILED);
        execution.backend_evidence = std::move(cold);
      }
    };
    std::vector<CheckpointRecord> checkpoints;
    LlmMemoryResult result;

    EXPECT_EQ(run_llm_memory_suite(config, plan, executor, result, recording_checkpoints(checkpoints)), EXIT_FAILURE);
    ASSERT_EQ(result.calibration_attempts[0].size(), 2u);
    EXPECT_EQ(result.calibration_attempts[0][1].purpose, "pilot");
    EXPECT_FALSE(result.calibration_attempts[0][1].valid);
    EXPECT_EQ(result.calibration_attempts[0][1].reason_code, LlmBackendReason::RESOURCES_NOT_PREPARED);
    const auto& checks = result.calibration_attempts[0][1].execution.cpu_cold_checks;
    EXPECT_TRUE(checks[0].applicable);
    EXPECT_FALSE(checks[0].evaluated);
    EXPECT_FALSE(checks[1].evaluated);
    EXPECT_TRUE(checks[2].evaluated);
    EXPECT_FALSE(checks[2].valid);
    EXPECT_EQ(checks[2].reason_code, LlmExecutorReason::PAGED_POST_VALIDATION_FAILED);
    EXPECT_EQ(result.counters.attempted_measurements, 0u);
    EXPECT_EQ(result.counters.terminal_measurements, 6u);
    for (const LlmMeasurementState& m : result.measurements) EXPECT_EQ(m.status, LlmMeasurementStatus::Failed);
    expect_single_terminal_checkpoint(checkpoints, LlmRunStatus::Failed);
  }
  {
    SCOPED_TRACE("excluded_task_failure_wins_stop");
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
    EXPECT_EQ(result.calibration_attempts[0][0].reason_code, LlmBackendReason::RESOURCES_NOT_PREPARED);
    EXPECT_EQ(result.scenario_plans.at(result.calibration_attempts[0][0].plan_handle).plan.work_units,
              config.iterations);
    expect_frozen_measurement_plans_populated(result, config.iterations);
    expect_interrupted_tail(result, 0);
  }
}

TEST(LlmMemoryRunnerTest, ExecutorExceptionsInExcludedAndMeasuredTasksStayContained) {
  const LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  {
    SCOPED_TRACE("excluded_executor_exception");
    FakeLlmBackend executor;
    executor.mutate = [](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext& context,
                         size_t, LlmTaskExecutionResult&) {
      if (context.kind == LlmRunnerTaskKind::Warmup) throw std::runtime_error("excluded executor failure");
    };
    std::vector<CheckpointRecord> checkpoints;
    LlmMemoryResult result;

    EXPECT_EQ(run_llm_memory_suite(config, plan, executor, result, recording_checkpoints(checkpoints)), EXIT_FAILURE);
    EXPECT_EQ(result.status, LlmRunStatus::Failed);
    EXPECT_EQ(result.reason_code, LlmRunnerReason::RUNNER_EXCEPTION);
    EXPECT_EQ(result.diagnostic, "excluded executor failure");
    ASSERT_EQ(result.calibration_attempts[0].size(), 1u);
    EXPECT_TRUE(result.calibration_attempts[0][0].terminal);
    EXPECT_FALSE(result.calibration_attempts[0][0].valid);
    EXPECT_EQ(result.calibration_attempts[0][0].reason_code, LlmRunnerReason::RUNNER_EXCEPTION);
    EXPECT_EQ(result.scenario_plans.at(result.calibration_attempts[0][0].plan_handle).plan.plan_identity,
              build_llm_scenario_work_plan(plan, LlmScenario::WeightsOnly, config.iterations, true).plan_identity);
    expect_single_terminal_checkpoint(checkpoints, LlmRunStatus::Failed);
  }
  {
    SCOPED_TRACE("measured_typed_and_unknown_exceptions");
    for (bool request_stop : {false, true}) {
      for (bool typed : {true, false}) {
        SCOPED_TRACE(::testing::Message() << "request_stop=" << request_stop << " typed=" << typed);
        bool stop = false;
        FakeLlmBackend executor;
        executor.mutate = [typed, request_stop, &stop](const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&,
                                                       const LlmRunnerTaskContext& context, size_t,
                                                       LlmTaskExecutionResult&) {
          if (context.kind == LlmRunnerTaskKind::Measurement) {
            stop = request_stop;
            if (typed) throw std::runtime_error("injected executor failure");
            throw 17;
          }
        };
        std::vector<CheckpointRecord> checkpoints;
        LlmRunnerHooks hooks = recording_checkpoints(checkpoints);
        hooks.stop_requested = [&]() { return stop; };
        LlmMemoryResult result;

        EXPECT_EQ(run_llm_memory_suite(config, plan, executor, result, hooks), EXIT_FAILURE);
        EXPECT_EQ(result.status, LlmRunStatus::Failed);
        EXPECT_EQ(result.reason_code,
                  typed ? LlmRunnerReason::RUNNER_EXCEPTION : LlmRunnerReason::RUNNER_UNKNOWN_EXCEPTION);
        if (typed) EXPECT_EQ(result.diagnostic, "injected executor failure");
        EXPECT_EQ(result.counters.attempted_measurements, 1u);
        EXPECT_EQ(result.counters.terminal_measurements, 3u);
        EXPECT_EQ(result.interruption_requested, request_stop);
        EXPECT_EQ(result.measurements[0].status, LlmMeasurementStatus::Failed);
        for (size_t i = 1; i < result.measurements.size(); ++i) {
          EXPECT_EQ(result.measurements[i].status,
                    request_stop ? LlmMeasurementStatus::Interrupted : LlmMeasurementStatus::Failed);
        }
        expect_single_terminal_checkpoint(checkpoints, LlmRunStatus::Failed);
        EXPECT_EQ(checkpoints[0].attempted_measurements, 1u);
        EXPECT_EQ(checkpoints[0].terminal_measurements, 3u);
        EXPECT_EQ(checkpoints[0].measured_measurements, 0u);
        EXPECT_EQ(result.logical_checkpoint_attempts, 1u);
        EXPECT_EQ(result.successful_logical_checkpoints, 1u);
        EXPECT_TRUE(result.terminal_checkpoint_completed);
      }
    }
  }
}

TEST(LlmMemoryRunnerTest, PreflightRejectsScenarioCapsSelectorMismatchesAndOverflow) {
  {
    SCOPED_TRACE("explicit_later_scenario_cap");
    LlmMemoryConfig config = explicit_config(1, 1);
    config.layer_count = 32;
    config.query_head_count = 8;
    config.kv_head_count = 8;
    config.head_dimension = 64;
    config.kv_element_bytes = 2;
    config.visible_context_tokens = 4096;
    const LlmMemoryWorkPlan initial_plan = build_runner_admitted_plan(config);
    ASSERT_TRUE(initial_plan.valid) << initial_plan.reason_code;
    const auto weights_limits = calculate_llm_scenario_limits(initial_plan.geometry, LlmScenario::WeightsOnly);
    const auto kv_limits = calculate_llm_scenario_limits(initial_plan.geometry, LlmScenario::KvOnly);
    ASSERT_TRUE(weights_limits.valid && kv_limits.valid);
    ASSERT_LT(kv_limits.effective_maximum_work_units, weights_limits.effective_maximum_work_units);
    config.iterations = kv_limits.effective_maximum_work_units + 1;
    const LlmMemoryWorkPlan plan = build_runner_admitted_plan(config);
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    EXPECT_TRUE(build_llm_scenario_work_plan(plan, LlmScenario::WeightsOnly, config.iterations, true).valid);
    const auto invalid_kv = build_llm_scenario_work_plan(plan, LlmScenario::KvOnly, config.iterations, true);
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
  {
    SCOPED_TRACE("selector_mismatch_and_auxiliary_overflow");
    const LlmMemoryConfig baseline = explicit_config();
    const LlmMemoryWorkPlan plan = build_runner_admitted_plan(baseline);
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    struct MismatchCase {
      void (*mutate)(LlmMemoryConfig&);
      const char* reason_code;
    };
    const std::array<MismatchCase, 3> mismatch_cases = {{
        {+[](LlmMemoryConfig& c) { c.backend = LlmMemoryBackend::Metal; }, LlmRunnerReason::BACKEND_UNAVAILABLE},
        {+[](LlmMemoryConfig& c) {
           c.phase = LlmPhase::Prefill;
           c.visible_context_tokens = 0;
           c.prompt_tokens = 2;
           c.attention_query_tile_tokens = 1;
           c.user_specified_context_tokens = false;
           c.user_specified_prompt_tokens = true;
           c.user_specified_attention_query_tile_tokens = true;
         },
         LlmRunnerReason::CONFIG_WORK_PLAN_MISMATCH},
        {+[](LlmMemoryConfig& c) { c = make_paged_variant(c); }, LlmRunnerReason::CONFIG_WORK_PLAN_MISMATCH},
    }};
    for (const MismatchCase& tc : mismatch_cases) {
      SCOPED_TRACE(tc.reason_code);
      LlmMemoryConfig mismatched = baseline;
      tc.mutate(mismatched);
      expect_preflight_rejection(mismatched, plan, tc.reason_code);
    }

    LlmMemoryConfig config = explicit_config();
    config.loop_count = 0;
    EXPECT_FALSE(calculate_llm_runner_auxiliary_estimate(config, plan).valid);
    expect_preflight_rejection(config, plan, nullptr);

    config.loop_count = std::numeric_limits<size_t>::max();
    const auto estimate = calculate_llm_runner_auxiliary_estimate(config, plan);
    EXPECT_FALSE(estimate.valid);
    EXPECT_EQ(estimate.reason_code, LlmRunnerReason::AUXILIARY_BYTES_OVERFLOW);
    expect_preflight_rejection(config, plan, LlmRunnerReason::INVALID_CONFIG);
  }
}

TEST(LlmMemoryRunnerTest, RunnerAuxiliaryBudgetAcceptsExactBoundaryAndRejectsOneByteLess) {
  const LlmMemoryConfig config = explicit_config();
  LlmMemoryWorkPlanRequest request = plan_request(config);
  const LlmMemoryWorkPlan preliminary = build_llm_memory_work_plan(request);
  ASSERT_TRUE(preliminary.valid) << preliminary.reason_code;
  const auto exec_aux = calculate_llm_executor_auxiliary_estimate(preliminary);
  const auto run_aux = calculate_llm_runner_auxiliary_estimate(config, preliminary);
  ASSERT_TRUE(exec_aux.valid && run_aux.valid);
  ASSERT_TRUE(NumericUtils::checked_add(exec_aux.checksum_auxiliary_bytes, run_aux.checksum_auxiliary_bytes,
                                        request.checksum_auxiliary_bytes));
  ASSERT_TRUE(NumericUtils::checked_add(exec_aux.orchestration_auxiliary_bytes, run_aux.orchestration_auxiliary_bytes,
                                        request.orchestration_auxiliary_bytes));
  const LlmMemoryWorkPlan exact = build_llm_memory_work_plan(request);
  ASSERT_TRUE(exact.valid) << exact.reason_code;
  FakeLlmBackend exact_executor;
  LlmMemoryResult exact_result;
  EXPECT_EQ(run_llm_memory_suite(config, exact, exact_executor, exact_result), EXIT_SUCCESS);
  EXPECT_TRUE(exact_result.results_complete);

  for (bool checksum_extra : {false, true}) {
    auto above_req = request;
    if (checksum_extra) ++above_req.checksum_auxiliary_bytes;
    else ++above_req.orchestration_auxiliary_bytes;
    const auto above = build_llm_memory_work_plan(above_req);
    ASSERT_TRUE(above.valid);
    FakeLlmBackend above_executor;
    LlmMemoryResult above_result;
    EXPECT_EQ(run_llm_memory_suite(config, above, above_executor, above_result), EXIT_SUCCESS);
    EXPECT_TRUE(above_result.run_accepted);
  }

  ASSERT_GT(request.orchestration_auxiliary_bytes, 0u);
  ASSERT_GT(request.checksum_auxiliary_bytes, 0u);
  for (bool checksum_short : {false, true}) {
    SCOPED_TRACE(checksum_short ? "checksum" : "orchestration");
    auto short_req = request;
    if (checksum_short) --short_req.checksum_auxiliary_bytes;
    else --short_req.orchestration_auxiliary_bytes;
    const auto one_byte_short = build_llm_memory_work_plan(short_req);
    ASSERT_TRUE(one_byte_short.valid) << one_byte_short.reason_code;
    expect_preflight_rejection(config, one_byte_short, LlmRunnerReason::AUXILIARY_BUDGET_INSUFFICIENT);
  }
}

TEST(LlmMemoryRunnerTest, CanonicalPlanRegistrationExpectationAndSnapshotValidation) {
  {
    SCOPED_TRACE("canonical_insertion_and_snapshot_remap");
    const auto config = explicit_config(1, 8);
    const auto model = build_runner_admitted_plan(config);
    ASSERT_TRUE(model.valid);
    FakeLlmBackend backend;
    LlmMemoryResult result;
    const auto eight = build_llm_scenario_work_plan(model, LlmScenario::Mixed, 8, true);
    const auto two = build_llm_scenario_work_plan(model, LlmScenario::Mixed, 2, true);
    const auto first = register_llm_scenario_plan(result, model, eight, backend);
    EXPECT_EQ(register_llm_scenario_plan(result, model, eight, backend), first);
    auto changed_payload = eight;
    ++changed_payload.effective_model_payload_bytes;
    EXPECT_THROW(register_llm_scenario_plan(result, model, changed_payload, backend), std::invalid_argument);
    EXPECT_EQ(backend.expected_calls, 1u);
    const auto warm = build_llm_scenario_work_plan(model, LlmScenario::WeightsOnly, 1, false);
    const auto warm_handle = register_llm_scenario_plan(result, model, warm, backend);
    prepare_llm_result_snapshot(result);
    EXPECT_EQ(result.snapshot_plan_refs.at(warm_handle), 0u);
    EXPECT_EQ(result.snapshot_plan_refs.at(first), 1u);
    EXPECT_EQ(backend.expected_calls, 2u);
    EXPECT_EQ(result.scenario_plans.size(), 2u);
    const auto second = register_llm_scenario_plan(result, model, two, backend);
    EXPECT_EQ(backend.expected_calls, 3u);
    EXPECT_EQ(result.scenario_plans.size(), 3u);
    result.frozen_plan_handles[2] = first;
    for (size_t t = 3; t < 20; ++t) {
      register_llm_scenario_plan(result, model, build_llm_scenario_work_plan(model, LlmScenario::WeightsOnly, t, true),
                                 backend);
    }
    EXPECT_EQ(result.scenario_plans.at(first).plan.work_units, 8u);
    EXPECT_EQ(result.scenario_plans.at(second).plan.work_units, 2u);
    prepare_llm_result_snapshot(result);
    EXPECT_LT(result.snapshot_plan_refs.at(second), result.snapshot_plan_refs.at(first));
    EXPECT_EQ(result.snapshot_plan_order.at(result.snapshot_plan_refs.at(first)), first);
    EXPECT_EQ(result.frozen_plan_handles[2], first);
    const size_t calls = backend.expected_calls;
    prepare_llm_result_snapshot(result);
    EXPECT_EQ(backend.expected_calls, calls);
    auto contradiction = eight;
    ++contradiction.weight_read_bytes;
    EXPECT_THROW(register_llm_scenario_plan(result, model, contradiction, backend), std::invalid_argument);
    EXPECT_EQ(backend.expected_calls, calls);
    EXPECT_EQ(find_llm_scenario_plan(result, kLlmNoTaskIndex), nullptr);
  }
  {
    SCOPED_TRACE("cpu_expectation_requires_prepared_resources");
    const auto config = explicit_config(1, 1);
    const auto model = build_runner_admitted_plan(config);
    const auto plan = build_llm_scenario_work_plan(model, LlmScenario::Mixed, 1, true);
    auto backend = create_llm_cpu_backend();
    const auto unavailable = backend->expected_cpu_checksum(model, plan);
    EXPECT_FALSE(unavailable.valid);
    EXPECT_TRUE(unavailable.workers.empty());
    LlmMemoryResult result;
    const auto handle = register_llm_scenario_plan(result, model, plan, *backend);
    EXPECT_FALSE(result.scenario_plans.at(handle).expected.available);
    EXPECT_TRUE(result.scenario_plans.at(handle).expected.cpu_workers.empty());
  }
  {
    SCOPED_TRACE("snapshot_rejects_contradictory_references");
    const auto config = explicit_config(1, 1);
    const auto model = build_runner_admitted_plan(config);
    FakeLlmBackend backend;
    LlmMemoryResult result;
    ASSERT_EQ(run_llm_memory_suite(config, model, backend, result), EXIT_SUCCESS);
    ASSERT_GT(result.calibration_attempt_counts[0], 0u);
    for (size_t mutation = 0; mutation < 5; ++mutation) {
      SCOPED_TRACE(mutation);
      auto invalid = result;
      switch (mutation) {
        case 0:
          std::swap(invalid.frozen_plan_handles[0], invalid.frozen_plan_handles[1]);
          break;
        case 1:
          invalid.calibration_attempts[0][0].plan_handle = invalid.frozen_plan_handles[1];
          break;
        case 2:
          invalid.calibration_attempts[0][0].scenario = LlmScenario::KvOnly;
          break;
        case 3:
          invalid.measurements[0].plan_handle = kLlmNoTaskIndex;
          break;
        case 4:
          ++invalid.measurements[0].execution.completion.completed_work_units;
          break;
      }
      EXPECT_THROW(prepare_llm_result_snapshot(invalid), std::invalid_argument);
    }
  }
}

TEST(LlmMemoryRunnerTest, RequiredColdChecksAndWholeBackendEvidenceCannotBeOmitted) {
  const auto config = explicit_config(1, 1);
  const auto model = build_runner_admitted_plan(config);
  ASSERT_TRUE(model.valid);
  for (bool missing_whole : {false, true}) {
    SCOPED_TRACE(missing_whole);
    FakeLlmBackend backend;
    backend.mutate = [missing_whole](const auto&, const auto&, const LlmRunnerTaskContext& context, size_t,
                                     LlmTaskExecutionResult& execution) {
      if (context.kind != LlmRunnerTaskKind::Measurement || context.scenario != LlmScenario::KvOnly) return;
      if (missing_whole) execution.backend_evidence = std::monostate{};
      else std::get<LlmCpuTaskEvidence>(execution.backend_evidence).executor.cold_checks[1].evaluated = false;
    };
    LlmMemoryResult result;
    EXPECT_EQ(run_llm_memory_suite(config, model, backend, result), EXIT_FAILURE);
    EXPECT_FALSE(result.run_accepted);
    ASSERT_EQ(result.measurements.size(), 3u);
    EXPECT_EQ(result.measurements[1].status, LlmMeasurementStatus::Invalid);
    EXPECT_EQ(result.measurements[1].reason_code, LlmBackendReason::VALIDATION_FAILED);
    EXPECT_FALSE(derive_llm_measurement_metrics(result.measurements[1]).payload_gb_s);
    EXPECT_TRUE(result.aggregates[1].accepted_measurement_ids.empty());
    EXPECT_TRUE(result.measurements[1].execution.timing.valid);
  }
}

TEST(LlmMemoryRunnerTest, BoundedSnapshotsObservePriorWritersAndDoNotSortBetweenSnapshots) {
  for (size_t count : {size_t{1}, size_t{3}, size_t{12}, size_t{48}}) {
    const auto config = explicit_config(count, 1);
    const auto model = build_runner_admitted_plan(config);
    ASSERT_TRUE(model.valid);
    for (bool progress : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "count=" << count << " progress=" << progress);
      FakeLlmBackend backend;
      LlmMemoryResult result;
      size_t callbacks = 0;
      size_t previous_statistics = 0;
      LlmRunnerHooks hooks;
      hooks.progress_snapshots = progress;
      hooks.observe_file_writes = [&]() { return std::make_pair(callbacks, callbacks); };
      hooks.checkpoint = [&](const LlmMemoryResult& snapshot, LlmCheckpointKind kind) {
        EXPECT_EQ(snapshot.prior_file_writer_attempts, callbacks);
        EXPECT_EQ(snapshot.prior_successful_file_writes, callbacks);
        EXPECT_EQ(snapshot.snapshot_preparation_attempts, callbacks + 1);
        EXPECT_EQ(snapshot.exact_statistics_passes, previous_statistics + 9);
        previous_statistics = snapshot.exact_statistics_passes;
        if (kind == LlmCheckpointKind::MeasurementTerminal) {
          EXPECT_EQ(snapshot.counters.completed_loops % snapshot.snapshot_interval_loops, 0u);
          EXPECT_EQ(snapshot.counters.terminal_measurements, snapshot.counters.completed_loops * 3);
        } else {
          EXPECT_FALSE(snapshot.terminal_checkpoint_completed);
        }
        ++callbacks;
        return EXIT_SUCCESS;
      };
      EXPECT_EQ(run_llm_memory_suite(config, model, backend, result, hooks), EXIT_SUCCESS);
      const size_t interval = std::max(size_t{1}, count / 8 + (count % 8 != 0));
      EXPECT_EQ(callbacks, (progress ? count / interval : 0) + 1);
      EXPECT_LE(callbacks, 9u);
      EXPECT_TRUE(result.run_accepted);
      EXPECT_EQ(backend.expected_calls, 3u);
      for (const auto& aggregate : result.aggregates) {
        EXPECT_EQ(aggregate.accepted_measurement_ids.size(), count);
        EXPECT_EQ(aggregate.effective_model_payload_gb_s.statistics.sample_count, count);
      }
    }
  }
}

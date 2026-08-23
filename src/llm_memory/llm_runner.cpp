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

/**
 * @file llm_runner.cpp
 * @brief LLM memory-profile calibration, ordering, status, and checkpoint flow
 */

#include "llm_memory/llm_runner.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <utility>

#include "core/config/constants.h"
#include "core/signal/signal_handler.h"
#include "utils/numeric_utils.h"

namespace {

constexpr std::array<LlmScenario, kLlmScenarioCount> kLlmScenarios = {LlmScenario::WeightsOnly, LlmScenario::KvOnly,
                                                                      LlmScenario::Mixed};
constexpr size_t kLlmRunnerDiagnosticCapacity = 512;
constexpr size_t kLlmRunnerReasonCapacity = 64;
constexpr size_t kLlmRunnerMaximumWarnings = kLlmScenarioCount + 1;
constexpr size_t kLlmMetalTaskPipelineLabelCapacity = 128;
constexpr size_t kLlmMetalTaskChecksumAlgorithmCapacity = 64;
constexpr size_t kLlmMetalTaskCommandStatusCapacity = 32;
constexpr size_t kLlmMetalTaskGridIdentityCapacity =
    4096 + Constants::LLM_METAL_MAX_THREADGROUPS_PER_GRID * 64;

bool is_activated_metal_profile(LlmPhase phase,
                                LlmKvLayout kv_layout) noexcept {
  return (phase == LlmPhase::Decode || phase == LlmPhase::Prefill) &&
         (kv_layout == LlmKvLayout::Contiguous ||
          kv_layout == LlmKvLayout::Paged);
}

constexpr std::string_view kLlmExecutorReasons[] = {
    LlmExecutorReason::VALID,
    LlmExecutorReason::INVALID_WORK_PLAN,
    LlmExecutorReason::INVALID_DESCRIPTOR_LAYOUT,
    LlmExecutorReason::AUXILIARY_BYTES_OVERFLOW,
    LlmExecutorReason::MEMORY_BUDGET_EXCEEDED,
    LlmExecutorReason::OUTPUT_NOT_EMPTY,
    LlmExecutorReason::WEIGHT_MAPPING_FAILED,
    LlmExecutorReason::K_MAPPING_FAILED,
    LlmExecutorReason::V_MAPPING_FAILED,
    LlmExecutorReason::PAGED_POST_VALIDATION_FAILED,
    LlmExecutorReason::PREFILL_POST_VALIDATION_FAILED,
    LlmExecutorReason::DESCRIPTOR_ALLOCATION_FAILED,
    LlmExecutorReason::INITIALIZATION_FAILED,
    LlmExecutorReason::INVALID_RESOURCES,
    LlmExecutorReason::INVALID_SCENARIO_PLAN,
    LlmExecutorReason::SCENARIO_PLAN_MISMATCH,
    LlmExecutorReason::EXPECTED_CHECKSUM_OVERFLOW,
    LlmExecutorReason::EXPECTED_CHECKSUM_ALLOCATION_FAILED,
    LlmExecutorReason::WORKER_STARTUP_FAILED,
    LlmExecutorReason::KERNEL_FAILED,
    LlmExecutorReason::INVALID_ELAPSED_TIME,
    LlmExecutorReason::CHECKSUM_MISMATCH,
};
constexpr std::string_view kLlmRunnerReasons[] = {
    LlmRunnerReason::NOT_STARTED,
    LlmRunnerReason::COMPLETE,
    LlmRunnerReason::PARTIAL_RESULTS,
    LlmRunnerReason::INTERRUPTION_REQUESTED,
    LlmRunnerReason::INTERRUPTION_BEFORE_TASK,
    LlmRunnerReason::NOT_RUN_AFTER_RUNTIME_FAILURE,
    LlmRunnerReason::INVALID_CONFIG,
    LlmRunnerReason::INVALID_MODEL_WORK_PLAN,
    LlmRunnerReason::CONFIG_WORK_PLAN_MISMATCH,
    LlmRunnerReason::BACKEND_UNAVAILABLE,
    LlmRunnerReason::AUXILIARY_BYTES_OVERFLOW,
    LlmRunnerReason::AUXILIARY_BUDGET_INSUFFICIENT,
    LlmRunnerReason::PLANNED_COUNTER_OVERFLOW,
    LlmRunnerReason::CALIBRATION_SCALING_FAILED,
    LlmRunnerReason::FROZEN_PLAN_MISMATCH,
    LlmRunnerReason::INVALID_DERIVED_METRIC,
    LlmRunnerReason::CHECKPOINT_WRITE_FAILED,
    LlmRunnerReason::RUNNER_EXCEPTION,
    LlmRunnerReason::RUNNER_UNKNOWN_EXCEPTION,
};
constexpr std::string_view kLlmBackendReasons[] = {
    LlmBackendReason::VALID,
    LlmBackendReason::NOT_INITIALIZED,
    LlmBackendReason::BACKEND_MISMATCH,
    LlmBackendReason::INVALID_BACKEND,
    LlmBackendReason::BACKEND_INITIALIZATION_FAILED,
    LlmBackendReason::TIMER_UNAVAILABLE,
    LlmBackendReason::EXECUTION_PLAN_MISMATCH,
    LlmBackendReason::RESOURCES_NOT_PREPARED,
    LlmBackendReason::RESOURCE_RELEASE_FAILED,
    LlmBackendReason::TASK_IDENTITY_MISMATCH,
    LlmBackendReason::TASK_COMPLETION_MISMATCH,
    LlmBackendReason::VALIDATION_NOT_EVALUATED,
    LlmBackendReason::VALIDATION_FAILED,
    LlmBackendReason::INVALID_AUTHORITATIVE_ELAPSED,
    LlmBackendReason::TASK_UNSUPPORTED,
    LlmBackendReason::METAL_DEVICE_UNAVAILABLE,
    LlmBackendReason::UNIFIED_MEMORY_REQUIRED,
    LlmBackendReason::APPLE7_FAMILY_REQUIRED,
    LlmBackendReason::ARGUMENT_BUFFER_TIER2_REQUIRED,
    LlmBackendReason::METAL_MAX_BUFFER_LENGTH_BELOW_SEGMENT_CAPACITY,
    LlmBackendReason::METAL_COMMAND_QUEUE_CREATION_FAILED,
    LlmBackendReason::METAL_KERNEL_COMPILATION_FAILED,
    LlmBackendReason::METAL_PIPELINE_CREATION_FAILED,
    LlmBackendReason::METAL_ARGUMENT_ENCODER_CREATION_FAILED,
    LlmBackendReason::METAL_ARGUMENT_BUFFER_LAYOUT_INVALID,
    LlmBackendReason::SEGMENT_COUNT_CAP_EXCEEDED,
    LlmBackendReason::PAGED_BLOCK_EXCEEDS_SEGMENT_CAPACITY,
    LlmBackendReason::MEMORY_BUDGET_EXCEEDED,
    LlmBackendReason::METAL_RESOURCE_ALLOCATION_FAILED,
    LlmBackendReason::METAL_RESOURCE_INITIALIZATION_FAILED,
    LlmBackendReason::PREPARATION_INTERRUPTED,
    LlmBackendReason::PLAN_RESOURCE_IDENTITY_MISMATCH,
    LlmBackendReason::STATUS_RESET_COMMAND_FAILED,
    LlmBackendReason::TIMED_COMMAND_BUFFER_ERROR,
    LlmBackendReason::INVALID_GPU_TIMESTAMPS,
    LlmBackendReason::TIMED_CHECKSUM_MISMATCH,
    LlmBackendReason::POST_VALIDATION_COMMAND_FAILED,
    LlmBackendReason::KV_WRITE_VALIDATION_MISMATCH,
    LlmBackendReason::PADDING_CANARY_MISMATCH,
};
constexpr std::string_view kLlmWorkPlanReasons[] = {
    LlmWorkPlanReason::VALID,
    LlmWorkPlanReason::ACTIVE_WEIGHT_BYTES_ZERO,
    LlmWorkPlanReason::LAYER_COUNT_ZERO,
    LlmWorkPlanReason::QUERY_HEAD_COUNT_ZERO,
    LlmWorkPlanReason::KV_HEAD_COUNT_ZERO,
    LlmWorkPlanReason::HEAD_DIMENSION_ZERO,
    LlmWorkPlanReason::INVALID_KV_ELEMENT_BYTES,
    LlmWorkPlanReason::CONTEXT_TOKENS_ZERO,
    LlmWorkPlanReason::CONTEXT_TOKENS_NOT_APPLICABLE,
    LlmWorkPlanReason::PROMPT_TOKENS_NOT_APPLICABLE,
    LlmWorkPlanReason::QUERY_TILE_TOKENS_NOT_APPLICABLE,
    LlmWorkPlanReason::KV_BLOCK_TOKENS_NOT_APPLICABLE,
    LlmWorkPlanReason::BATCH_SIZE_ZERO,
    LlmWorkPlanReason::QUERY_HEADS_BELOW_KV_HEADS,
    LlmWorkPlanReason::QUERY_HEADS_NOT_DIVISIBLE_BY_KV_HEADS,
    LlmWorkPlanReason::KV_VECTOR_BYTES_OVERFLOW,
    LlmWorkPlanReason::KV_RECORD_BYTES_OVERFLOW,
    LlmWorkPlanReason::KV_LAYER_RECORD_BYTES_OVERFLOW,
    LlmWorkPlanReason::KV_BYTES_PER_TOKEN_OVERFLOW,
    LlmWorkPlanReason::KV_SEQUENCE_BYTES_OVERFLOW,
    LlmWorkPlanReason::KV_MAPPING_BYTES_OVERFLOW,
    LlmWorkPlanReason::KV_CAPACITY_BYTES_OVERFLOW,
    LlmWorkPlanReason::KV_WRITE_BYTES_OVERFLOW,
    LlmWorkPlanReason::KV_READ_BYTES_OVERFLOW,
    LlmWorkPlanReason::KV_ONLY_PAYLOAD_OVERFLOW,
    LlmWorkPlanReason::MIXED_PAYLOAD_OVERFLOW,
    LlmWorkPlanReason::TOTAL_DATA_BYTES_OVERFLOW,
    LlmWorkPlanReason::REQUESTED_WORKERS_ZERO,
    LlmWorkPlanReason::AVAILABLE_WORKERS_ZERO,
    LlmWorkPlanReason::NO_EXECUTABLE_WORKER,
    LlmWorkPlanReason::LAYER_SEQUENCE_COUNT_OVERFLOW,
    LlmWorkPlanReason::DESCRIPTOR_COUNT_OVERFLOW,
    LlmWorkPlanReason::DESCRIPTOR_BYTES_OVERFLOW,
    LlmWorkPlanReason::PLANNER_STORAGE_BYTES_OVERFLOW,
    LlmWorkPlanReason::MAPPING_GRANULARITY_ZERO,
    LlmWorkPlanReason::MAPPING_ROUND_UP_OVERFLOW,
    LlmWorkPlanReason::AUXILIARY_BYTES_OVERFLOW,
    LlmWorkPlanReason::AUXILIARY_PREFLIGHT_MISMATCH,
    LlmWorkPlanReason::MEMORY_REQUIREMENT_OVERFLOW,
    LlmWorkPlanReason::MEMORY_BUDGET_OVERFLOW,
    LlmWorkPlanReason::MEMORY_BUDGET_EXCEEDED,
    LlmWorkPlanReason::WITHIN_MEMORY_BUDGET,
    LlmWorkPlanReason::PLANNER_ALLOCATION_FAILED,
    LlmWorkPlanReason::BLOCK_TABLE_MAPPING_FAILED,
    LlmWorkPlanReason::BLOCK_TABLE_MATERIALIZATION_FAILED,
    LlmWorkPlanReason::BLOCK_TABLE_PROTECTION_FAILED,
    LlmWorkPlanReason::INVALID_SCENARIO,
    LlmWorkPlanReason::INVALID_MODEL_WORK_PLAN,
    LlmWorkPlanReason::INVALID_BACKEND,
    LlmWorkPlanReason::METAL_WORKERS_NOT_APPLICABLE,
    LlmWorkPlanReason::INVALID_PHASE,
    LlmWorkPlanReason::INVALID_KV_LAYOUT,
    LlmWorkPlanReason::JSON_INTEGER_OUT_OF_RANGE,
    LlmWorkPlanReason::GUARDRAIL_BELOW_ONE_WORK_UNIT,
    LlmWorkPlanReason::WORK_UNIT_COUNT_ZERO,
    LlmWorkPlanReason::WORK_UNIT_CAP_EXCEEDED,
    LlmWorkPlanReason::TASK_ACCOUNTED_BYTES_OVERFLOW,
    LlmWorkPlanReason::TASK_ACCOUNTED_BYTES_CAP_EXCEEDED,
};
constexpr std::string_view kLlmMetalPlanReasons[] = {
    LlmMetalPlanReason::VALID,
    LlmMetalPlanReason::INVALID_GEOMETRY,
    LlmMetalPlanReason::PAGED_LAYOUT_REQUIRED,
    LlmMetalPlanReason::PAGED_LAYOUT_MISMATCH,
    LlmMetalPlanReason::ARGUMENT_ENCODER_LENGTH_ZERO,
    LlmMetalPlanReason::ARGUMENT_ENCODER_ALIGNMENT_INVALID,
    LlmMetalPlanReason::ARGUMENT_BUFFER_LAYOUT_INVALID,
    LlmMetalPlanReason::RESOURCE_LENGTH_OVERFLOW,
    LlmMetalPlanReason::RESOURCE_LENGTH_EXCEEDS_MAX_BUFFER,
    LlmMetalPlanReason::MEMORY_BUDGET_OVERFLOW,
    LlmMetalPlanReason::MEMORY_BUDGET_EXCEEDED,
    LlmMetalPlanReason::PIPELINE_WIDTH_ZERO,
    LlmMetalPlanReason::PIPELINE_THREAD_LIMIT_INVALID,
    LlmMetalPlanReason::OWNER_COST_COUNT_MISMATCH,
    LlmMetalPlanReason::OWNER_COUNT_OVERFLOW,
    LlmMetalPlanReason::OWNER_STRIDE_CAP_EXCEEDED,
    LlmMetalPlanReason::VECTOR_ITERATION_CAP_EXCEEDED,
    LlmMetalPlanReason::SERIAL_RANGE_VISIT_CAP_EXCEEDED,
    LlmMetalPlanReason::SERIAL_RANGE_VISIT_COUNT_OVERFLOW,
    LlmMetalPlanReason::SEMANTIC_VISIT_CAP_EXCEEDED,
    LlmMetalPlanReason::WORK_UNITS_PER_DISPATCH_CAP_EXCEEDED,
    LlmMetalPlanReason::PLANNER_ALLOCATION_FAILED,
};

enum class CalibrationOutcome {
  Success = 0,
  Interrupted,
  Failed,
};

size_t scenario_index(LlmScenario scenario) noexcept {
  switch (scenario) {
    case LlmScenario::WeightsOnly:
      return 0;
    case LlmScenario::KvOnly:
      return 1;
    case LlmScenario::Mixed:
      return 2;
  }
  return kLlmScenarioCount;
}

bool checked_add_to(size_t value, size_t& total) noexcept {
  size_t next = 0;
  if (!NumericUtils::checked_add(total, value, next)) {
    return false;
  }
  total = next;
  return true;
}

bool add_metal_task_string_capacity(size_t record_count,
                                    size_t& total) noexcept {
  size_t bytes_per_record = 0;
  size_t all_records = 0;
  const auto add_string_capacity = [&](size_t capacity) {
    size_t with_null = 0;
    size_t conservative = 0;
    return NumericUtils::checked_add(capacity, static_cast<size_t>(1),
                                     with_null) &&
           NumericUtils::checked_multiply(with_null, static_cast<size_t>(2),
                                          conservative) &&
           checked_add_to(conservative, bytes_per_record);
  };
  return add_string_capacity(kLlmMetalTaskPipelineLabelCapacity) &&
         add_string_capacity(kLlmRunnerReasonCapacity) &&
         add_string_capacity(kLlmMetalTaskGridIdentityCapacity) &&
         add_string_capacity(kLlmMetalTaskCommandStatusCapacity) &&
         add_string_capacity(kLlmMetalTaskCommandStatusCapacity) &&
         add_string_capacity(kLlmMetalTaskCommandStatusCapacity) &&
         add_string_capacity(kLlmMetalTaskChecksumAlgorithmCapacity) &&
         add_string_capacity(Constants::LLM_METAL_DIAGNOSTIC_MAX_BYTES) &&
         add_string_capacity(Constants::LLM_METAL_DIAGNOSTIC_MAX_BYTES) &&
         checked_add_to(
             Constants::LLM_METAL_MAX_THREADGROUPS_PER_GRID *
                 sizeof(size_t),
             bytes_per_record) &&
         NumericUtils::checked_multiply(bytes_per_record, record_count,
                                        all_records) &&
         checked_add_to(all_records, total);
}

std::string_view canonical_calibration_purpose(std::string_view purpose) noexcept {
  constexpr std::array<std::string_view, 6> kPurposes = {
      "calibration_shape_warmup", "pilot", "correction",
      "single_unit_confirmation_warmup", "single_unit_confirmation",
      "frozen_measurement_warmup",
  };
  for (std::string_view candidate : kPurposes) {
    if (purpose == candidate) {
      return candidate;
    }
  }
  return "unknown";
}

void assign_diagnostic(LlmMemoryResult& result, const char* diagnostic) {
  result.diagnostic.assign(diagnostic == nullptr ? "" : diagnostic, 0,
                           std::min(kLlmRunnerDiagnosticCapacity,
                                    diagnostic == nullptr ? size_t{0} : std::char_traits<char>::length(diagnostic)));
}

void reset_uninitialized_failure(LlmMemoryResult& result, const char* reason_code) noexcept {
  try {
    result = LlmMemoryResult{};
    result.reason_code = reason_code;
  } catch (...) {
  }
}

bool inputs_match(const LlmMemoryConfig& config, const LlmMemoryWorkPlan& plan) {
  const LlmMemoryConfigValidation validation = validate_llm_memory_config(config);
  const LlmCpuExecutionPlan* cpu_plan = get_llm_cpu_execution_plan(plan);
  const LlmMetalExecutionPlan* metal_plan =
      get_llm_metal_execution_plan(plan);
  if (!validation.valid || !plan.valid || !plan.geometry.valid || plan.plan_identity.empty()) {
    return false;
  }
  const LlmGeometry& geometry = plan.geometry;
  const bool phase_geometry_matches =
      config.phase == LlmPhase::Decode
          ? geometry.decode.has_value() && !geometry.prefill.has_value() &&
                config.visible_context_tokens ==
                    geometry.decode->visible_context_tokens
          : geometry.prefill.has_value() && !geometry.decode.has_value() &&
                config.prompt_tokens == geometry.prefill->prompt_tokens &&
                config.attention_query_tile_tokens ==
                    geometry.prefill->attention_query_tile_tokens;
  const bool backend_inputs_match =
      config.backend == LlmMemoryBackend::Cpu
          ? cpu_plan != nullptr &&
                config.requested_workers == cpu_plan->requested_workers &&
                config.available_workers == cpu_plan->available_workers
          : config.backend == LlmMemoryBackend::Metal &&
                metal_plan != nullptr && config.requested_workers == 0 &&
                config.available_workers == 0 &&
                is_activated_metal_profile(config.phase,
                                           config.kv_layout);
  return config.backend == plan.backend && config.phase == plan.phase &&
         config.kv_layout == plan.kv_layout && geometry.phase == plan.phase &&
         geometry.kv_layout == plan.kv_layout && phase_geometry_matches &&
         validation.active_weight_bytes == geometry.active_weight_bytes_per_work_unit &&
         config.layer_count == geometry.layer_count && config.query_head_count == geometry.query_head_count &&
         config.kv_head_count == geometry.kv_head_count && config.head_dimension == geometry.head_dimension &&
         config.kv_element_bytes == geometry.kv_element_bytes &&
         config.kv_block_tokens == geometry.kv_block_tokens &&
         config.batch_size == geometry.batch_size && backend_inputs_match &&
         config.seed == plan.base_seed;
}

std::string_view runner_budget_admission_reason(
    const LlmMemoryWorkPlan& plan, const LlmRunnerAuxiliaryEstimate& runner_auxiliary,
    const LlmBackend& backend) noexcept {
  const LlmBackendAuxiliaryEstimate backend_auxiliary = backend.calculate_auxiliary_estimate(plan);
  if (!backend_auxiliary.valid) {
    return backend_auxiliary.reason_code;
  }
  if (!runner_auxiliary.valid) {
    return runner_auxiliary.reason_code;
  }
  size_t required_checksum = 0;
  size_t required_orchestration = 0;
  if (!NumericUtils::checked_add(backend_auxiliary.checksum_auxiliary_bytes, runner_auxiliary.checksum_auxiliary_bytes,
                                 required_checksum) ||
      !NumericUtils::checked_add(backend_auxiliary.orchestration_auxiliary_bytes,
                                 runner_auxiliary.orchestration_auxiliary_bytes, required_orchestration)) {
    return LlmRunnerReason::AUXILIARY_BYTES_OVERFLOW;
  }
  if (plan.memory_budget.request.checksum_auxiliary_bytes < required_checksum ||
      plan.memory_budget.request.orchestration_auxiliary_bytes < required_orchestration) {
    return LlmRunnerReason::AUXILIARY_BUDGET_INSUFFICIENT;
  }
  return LlmExecutorReason::VALID;
}

bool stop_requested(const LlmRunnerHooks& hooks) {
  return hooks.stop_requested ? hooks.stop_requested() : signal_received();
}

bool pending_stop_after_failure(const LlmMemoryResult& result, const LlmRunnerHooks& hooks) noexcept {
  if (result.interruption_requested) {
    return true;
  }
  try {
    return stop_requested(hooks);
  } catch (...) {
    return false;
  }
}

const LlmExecutorResult* cpu_executor_evidence(
    const LlmTaskExecutionResult& execution) noexcept {
  const auto* cpu = std::get_if<LlmCpuTaskEvidence>(&execution.backend_evidence);
  return cpu == nullptr ? nullptr : &cpu->executor;
}

LlmTaskExecutionEvidence compact_execution(
    const LlmTaskExecutionResult& execution) {
  LlmTaskExecutionEvidence evidence;
  evidence.status = execution.status;
  evidence.valid = execution.status == LlmTaskExecutionStatus::Complete;
  evidence.reason_code = canonicalize_llm_result_reason_code(execution.reason_code);
  evidence.elapsed_seconds = execution.timing.elapsed_seconds;
  evidence.timing_evaluated = execution.timing.evaluated;
  evidence.timing_valid = execution.timing.valid;
  evidence.completion = execution.completion;
  evidence.validation_evaluated = execution.validation.evaluated;
  evidence.validation_valid = execution.validation.valid;
  const LlmExecutorResult* cpu = cpu_executor_evidence(execution);
  if (cpu != nullptr) {
    const bool checksum_evidence_complete =
        cpu->checksum_evaluated && cpu->requested_workers != 0 &&
        cpu->expected_checksums.size() == cpu->requested_workers &&
        cpu->actual_checksums.size() == cpu->requested_workers;
    evidence.cpu_evidence_available = true;
    evidence.requested_workers = cpu->requested_workers;
    evidence.created_workers = cpu->created_workers;
    evidence.completed_workers = cpu->completed_workers;
    evidence.qos_successful_workers = cpu->qos_successful_workers;
    evidence.qos_failed_workers = cpu->qos_failed_workers;
    evidence.worker_startup_failed = cpu->worker_startup_failed;
    evidence.kernel_succeeded = cpu->kernel_succeeded;
    evidence.timer_started = cpu->timer_started;
    evidence.timer_stopped = cpu->timer_stopped;
    evidence.checksum_evaluated = checksum_evidence_complete;
    evidence.checksum_valid = checksum_evidence_complete && cpu->checksum_valid;
    if (checksum_evidence_complete) {
      evidence.expected_run_checksum = cpu->expected_run_checksum;
      evidence.actual_run_checksum = cpu->actual_run_checksum;
    }
  }
  const LlmMetalTaskEvidence* const metal =
      get_llm_metal_task_evidence(execution);
  if (metal != nullptr) {
    evidence.metal_evidence_available = true;
    evidence.metal = *metal;
  }
  evidence.available = true;
  return evidence;
}

bool task_identity_matches(const LlmTaskIdentity& identity,
                           const LlmMemoryWorkPlan& model_plan,
                           const LlmScenarioWorkPlan& task_plan,
                           const LlmRunnerTaskContext& context) noexcept {
  return identity.backend == model_plan.backend &&
         identity.phase == model_plan.phase &&
         identity.kv_layout == model_plan.kv_layout &&
         identity.work_unit_kind == task_plan.work_unit_kind &&
         identity.kv_write_kind == task_plan.kv_write_kind &&
         identity.task_kind == context.kind &&
         identity.scenario == task_plan.scenario &&
         identity.scenario == context.scenario &&
         identity.attempt_index == context.attempt_index &&
         identity.loop_index == context.loop_index &&
         identity.order_position == context.order_position &&
         identity.purpose == context.purpose &&
         identity.model_plan_identity == model_plan.plan_identity &&
         identity.scenario_plan_identity == task_plan.plan_identity;
}

bool task_completion_matches(const LlmTaskCompletion& completion,
                             const LlmScenarioWorkPlan& task_plan) noexcept {
  return completion.planned_work_units == task_plan.work_units &&
         completion.completed_work_units == task_plan.work_units &&
         completion.completed_effective_model_payload_bytes ==
             task_plan.effective_model_payload_bytes &&
         completion.completed_layout_metadata_lookup_count ==
             task_plan.layout_metadata_lookup_count &&
         completion.completed_layout_metadata_read_bytes ==
             task_plan.layout_metadata_read_bytes &&
         completion.completed_task_accounted_bytes ==
             task_plan.task_accounted_bytes;
}

bool execution_is_accepted(const LlmTaskExecutionResult& execution,
                           const LlmMemoryWorkPlan& model_plan,
                           const LlmScenarioWorkPlan& task_plan,
                           const LlmRunnerTaskContext& context) noexcept {
  return execution.status == LlmTaskExecutionStatus::Complete &&
         execution.reason_code == LlmBackendReason::VALID &&
         task_identity_matches(execution.identity, model_plan, task_plan,
                               context) &&
         execution.timing.evaluated && execution.timing.valid &&
         std::isfinite(execution.timing.elapsed_seconds) &&
         execution.timing.elapsed_seconds > 0.0 &&
         task_completion_matches(execution.completion, task_plan) &&
         execution.validation.evaluated && execution.validation.valid;
}

std::string_view execution_failure_reason(
    const LlmTaskExecutionResult& execution,
    const LlmMemoryWorkPlan& model_plan,
    const LlmScenarioWorkPlan& task_plan,
    const LlmRunnerTaskContext& context) noexcept {
  const std::string_view reason_code =
      canonicalize_llm_result_reason_code(execution.reason_code);
  if (execution.status == LlmTaskExecutionStatus::Unsupported) {
    return reason_code == LlmRunnerReason::RUNNER_UNKNOWN_EXCEPTION
               ? std::string_view(LlmBackendReason::TASK_UNSUPPORTED)
               : reason_code;
  }
  if (execution.status == LlmTaskExecutionStatus::Failed ||
      execution.status == LlmTaskExecutionStatus::Invalid) {
    return reason_code == LlmBackendReason::VALID
               ? std::string_view(LlmExecutorReason::INVALID_RESOURCES)
               : reason_code;
  }
  if (!task_identity_matches(execution.identity, model_plan, task_plan,
                             context)) {
    return LlmBackendReason::TASK_IDENTITY_MISMATCH;
  }
  if (!task_completion_matches(execution.completion, task_plan)) {
    return LlmBackendReason::TASK_COMPLETION_MISMATCH;
  }
  if (!execution.timing.evaluated || !execution.timing.valid ||
      !std::isfinite(execution.timing.elapsed_seconds) ||
      execution.timing.elapsed_seconds <= 0.0) {
    return LlmBackendReason::INVALID_AUTHORITATIVE_ELAPSED;
  }
  if (!execution.validation.evaluated) {
    return LlmBackendReason::VALIDATION_NOT_EVALUATED;
  }
  if (!execution.validation.valid) {
    return LlmBackendReason::VALIDATION_FAILED;
  }
  return reason_code == LlmBackendReason::VALID
             ? std::string_view(LlmExecutorReason::INVALID_RESOURCES)
             : reason_code;
}

LlmMeasurementStatus execution_failure_status(
    const LlmTaskExecutionResult& execution,
    std::string_view reason_code) noexcept {
  return execution.status == LlmTaskExecutionStatus::Invalid ||
                 reason_code == LlmExecutorReason::CHECKSUM_MISMATCH ||
                 reason_code == LlmExecutorReason::INVALID_ELAPSED_TIME ||
                 reason_code == LlmBackendReason::INVALID_AUTHORITATIVE_ELAPSED ||
                 reason_code == LlmBackendReason::VALIDATION_FAILED
             ? LlmMeasurementStatus::Invalid
             : LlmMeasurementStatus::Failed;
}

void clear_measurement_values(LlmMeasurementState& measurement) {
  measurement.completed_work_units = 0;
  measurement.completed_effective_model_payload_bytes = 0;
  measurement.completed_layout_metadata_lookup_count = 0;
  measurement.completed_layout_metadata_read_bytes = 0;
  measurement.completed_task_accounted_bytes = 0;
  measurement.elapsed_seconds.reset();
  measurement.synthetic_work_unit_latency_seconds.reset();
  measurement.synthetic_memory_work_units_per_second.reset();
  measurement.effective_model_payload_gb_s.reset();
  measurement.checksum_valid = false;
}

void update_metric_aggregate(LlmMetricAggregate& aggregate, LlmStatisticsWorkspace& workspace) {
  aggregate.statistics = DescriptiveStatistics{};
  aggregate.headline.reset();
  if (aggregate.values.empty()) {
    return;
  }
  aggregate.statistics =
      calculate_descriptive_statistics(aggregate.values, workspace.sorted_values, workspace.absolute_deviations);
  aggregate.headline = aggregate.values.size() == 1 ? aggregate.values.front() : aggregate.statistics.median;
}

void update_scenario_aggregate(LlmMemoryResult& result, const LlmMeasurementState& measurement) {
  const size_t index = scenario_index(measurement.scenario);
  if (index >= result.aggregates.size() || measurement.status != LlmMeasurementStatus::Measured ||
      !measurement.checksum_valid || !measurement.synthetic_work_unit_latency_seconds.has_value() ||
      !measurement.synthetic_memory_work_units_per_second.has_value() ||
      !measurement.effective_model_payload_gb_s.has_value()) {
    return;
  }

  LlmScenarioAggregate& aggregate = result.aggregates[index];
  aggregate.work_unit_latency_seconds.values.push_back(*measurement.synthetic_work_unit_latency_seconds);
  aggregate.synthetic_memory_work_units_per_second.values.push_back(
      *measurement.synthetic_memory_work_units_per_second);
  aggregate.effective_model_payload_gb_s.values.push_back(*measurement.effective_model_payload_gb_s);
  update_metric_aggregate(aggregate.work_unit_latency_seconds, result.statistics_workspace);
  update_metric_aggregate(aggregate.synthetic_memory_work_units_per_second, result.statistics_workspace);
  update_metric_aggregate(aggregate.effective_model_payload_gb_s, result.statistics_workspace);

  const size_t sample_count = aggregate.effective_model_payload_gb_s.values.size();
  aggregate.status = sample_count == result.counters.planned_loops ? "complete" : "partial";
  if (sample_count < 3) {
    aggregate.stability_quality = "insufficient-samples";
  } else if (aggregate.effective_model_payload_gb_s.statistics.coefficient_of_variation_defined &&
             aggregate.effective_model_payload_gb_s.statistics.coefficient_of_variation_pct >
                 Constants::LLM_STREAMING_CV_WARNING_PCT) {
    aggregate.stability_quality = "noisy";
  } else {
    aggregate.stability_quality = "stable";
  }
}

void rebuild_quality_warnings(LlmMemoryResult& result) {
  result.quality_warnings.clear();
  for (const LlmScenarioAggregate& aggregate : result.aggregates) {
    if (aggregate.stability_quality != "noisy") {
      continue;
    }
    switch (aggregate.scenario) {
      case LlmScenario::WeightsOnly:
        result.quality_warnings.push_back("weights_only-high-cv");
        break;
      case LlmScenario::KvOnly:
        result.quality_warnings.push_back("kv_only-high-cv");
        break;
      case LlmScenario::Mixed:
        result.quality_warnings.push_back("mixed-high-cv");
        break;
    }
  }
  if (result.results_complete && !result.scenario_order_balance_complete) {
    result.quality_warnings.push_back("scenario-order-not-balanced");
  }
}

bool order_is_balanced(const LlmMemoryResult& result) noexcept {
  if (!result.results_complete || result.loops.empty()) {
    return false;
  }
  std::array<std::array<size_t, kLlmScenarioCount>, kLlmScenarioCount> counts{};
  for (const LlmLoopRecord& loop : result.loops) {
    if (loop.realized_order_count != kLlmScenarioCount) {
      return false;
    }
    for (size_t position = 0; position < kLlmScenarioCount; ++position) {
      const size_t index = scenario_index(loop.realized_order[position]);
      if (index >= kLlmScenarioCount) {
        return false;
      }
      ++counts[index][position];
    }
  }
  for (const auto& scenario_counts : counts) {
    if (scenario_counts[0] == 0 || !std::all_of(scenario_counts.begin() + 1, scenario_counts.end(),
                                                [&](size_t count) { return count == scenario_counts[0]; })) {
      return false;
    }
  }
  return true;
}

void update_completion_state(LlmMemoryResult& result) {
  result.results_complete = result.counters.planned_measurements != 0 &&
                            result.counters.measured_measurements == result.counters.planned_measurements;
  result.scenario_order_balance_complete = order_is_balanced(result);

  if (result.status != LlmRunStatus::Failed &&
      result.status != LlmRunStatus::Unsupported) {
    if (result.results_complete) {
      result.status = LlmRunStatus::Complete;
      result.reason_code = LlmRunnerReason::COMPLETE;
    } else if (result.interruption_requested) {
      result.status = LlmRunStatus::Interrupted;
      result.reason_code = LlmRunnerReason::INTERRUPTION_REQUESTED;
    } else if (result.counters.terminal_measurements != 0 || result.counters.attempted_measurements != 0) {
      result.status = LlmRunStatus::Partial;
      result.reason_code = LlmRunnerReason::PARTIAL_RESULTS;
    } else {
      result.status = LlmRunStatus::NotStarted;
      result.reason_code = LlmRunnerReason::NOT_STARTED;
    }
  }

  result.conclusions_valid = result.status == LlmRunStatus::Complete && result.results_complete &&
                             result.scenario_order_balance_complete && !result.checkpoint_failed;
  rebuild_quality_warnings(result);
}

void finalize_remaining_interrupted(LlmMemoryResult& result) {
  for (LlmMeasurementState& measurement : result.measurements) {
    const bool synthetic_failure = !measurement.attempted && measurement.status == LlmMeasurementStatus::Failed &&
                                   measurement.reason_code == LlmRunnerReason::NOT_RUN_AFTER_RUNTIME_FAILURE;
    if (measurement.status == LlmMeasurementStatus::NotRun || synthetic_failure) {
      if (measurement.status == LlmMeasurementStatus::NotRun) {
        ++result.counters.terminal_measurements;
      }
      measurement.status = LlmMeasurementStatus::Interrupted;
      measurement.reason_code = LlmRunnerReason::INTERRUPTION_BEFORE_TASK;
      clear_measurement_values(measurement);
    }
  }
}

void finalize_remaining_failed(LlmMemoryResult& result) {
  for (LlmMeasurementState& measurement : result.measurements) {
    if (!measurement.attempted && measurement.status == LlmMeasurementStatus::NotRun) {
      ++result.counters.terminal_measurements;
      measurement.status = LlmMeasurementStatus::Failed;
      measurement.reason_code = LlmRunnerReason::NOT_RUN_AFTER_RUNTIME_FAILURE;
      clear_measurement_values(measurement);
    }
  }
}

void finalize_failure(LlmMemoryResult& result, std::string_view reason_code, bool interruption_pending) {
  const std::string_view stable_reason_code = canonicalize_llm_result_reason_code(reason_code);
  for (LlmMeasurementState& measurement : result.measurements) {
    if (measurement.attempted && measurement.status == LlmMeasurementStatus::NotRun) {
      ++result.counters.terminal_measurements;
      measurement.status = LlmMeasurementStatus::Failed;
      measurement.reason_code = stable_reason_code;
      clear_measurement_values(measurement);
    }
  }
  if (interruption_pending) {
    result.interruption_requested = true;
    finalize_remaining_interrupted(result);
  } else {
    finalize_remaining_failed(result);
  }
  result.status = LlmRunStatus::Failed;
  result.reason_code = stable_reason_code;
  update_completion_state(result);
}

bool invoke_checkpoint(LlmMemoryResult& result, LlmCheckpointKind kind, const LlmRunnerHooks& hooks) {
  ++result.logical_checkpoint_attempts;
  ++result.successful_logical_checkpoints;
  if (kind == LlmCheckpointKind::CommandTerminal) {
    result.terminal_checkpoint_attempted = true;
    result.terminal_checkpoint_completed = true;
  }
  if (!hooks.checkpoint) {
    return true;
  }

  int status = EXIT_FAILURE;
  bool checkpoint_threw = false;
  std::string_view checkpoint_failure_reason = LlmRunnerReason::CHECKPOINT_WRITE_FAILED;
  try {
    status = hooks.checkpoint(result, kind);
  } catch (const std::exception& error) {
    checkpoint_threw = true;
    checkpoint_failure_reason = LlmRunnerReason::RUNNER_EXCEPTION;
    assign_diagnostic(result, error.what());
  } catch (...) {
    checkpoint_threw = true;
    checkpoint_failure_reason = LlmRunnerReason::RUNNER_UNKNOWN_EXCEPTION;
    result.diagnostic.clear();
  }
  if (status != EXIT_SUCCESS) {
    --result.successful_logical_checkpoints;
    if (kind == LlmCheckpointKind::CommandTerminal) {
      result.terminal_checkpoint_completed = false;
    }
    result.checkpoint_failed = true;
    result.status = LlmRunStatus::Failed;
    result.reason_code = checkpoint_failure_reason;
    if (!checkpoint_threw) {
      result.diagnostic.clear();
    }
    result.conclusions_valid = false;
    return false;
  }
  return true;
}

void refresh_calibration_references(LlmMemoryResult& result) noexcept {
  for (LlmMeasurementState& measurement : result.measurements) {
    const size_t index = scenario_index(measurement.scenario);
    if (index < kLlmScenarioCount) {
      measurement.calibration_attempt_count = result.calibration_attempt_counts[index];
    }
  }
}

bool release_backend_once(LlmBackend& backend, bool& release_attempted,
                          LlmMemoryResult& result) {
  if (release_attempted) {
    return backend.evidence().release.status == LlmBackendStatus::Ready;
  }
  release_attempted = true;
  const LlmBackendLifecycleResult release = backend.release_resources();
  if (release.status == LlmBackendStatus::Ready) {
    return true;
  }
  const std::string_view reason = canonicalize_llm_result_reason_code(
      release.reason_code.empty() ? LlmBackendReason::RESOURCE_RELEASE_FAILED
                                  : release.reason_code);
  finalize_failure(result, reason, result.interruption_requested);
  return false;
}

int fail_uninitialized_backend_transition(
    LlmMemoryResult& result, const char* reason_code, LlmBackend& backend,
    bool& release_attempted) {
  reset_uninitialized_failure(result, reason_code);
  release_backend_once(backend, release_attempted, result);
  return EXIT_FAILURE;
}

int finish_terminal(LlmMemoryResult& result, const LlmRunnerHooks& hooks,
                    LlmBackend& backend, bool& release_attempted) {
  refresh_calibration_references(result);
  update_completion_state(result);
  const bool release_succeeded =
      release_backend_once(backend, release_attempted, result);
  if (result.checkpoint_failed || !invoke_checkpoint(result, LlmCheckpointKind::CommandTerminal, hooks)) {
    update_completion_state(result);
    return EXIT_FAILURE;
  }
  if (!release_succeeded) {
    return EXIT_FAILURE;
  }
  return result.status == LlmRunStatus::Failed ||
                 result.status == LlmRunStatus::Unsupported
             ? EXIT_FAILURE
             : EXIT_SUCCESS;
}

size_t calibration_capacity(const LlmMemoryConfig& config) noexcept {
  return config.user_specified_iterations ? 1 : 4 + Constants::LLM_CALIBRATION_MAX_CORRECTIONS;
}

bool calculate_calibration_identity_capacities(const LlmMemoryConfig& config, const LlmMemoryWorkPlan& model_plan,
                                               std::array<size_t, kLlmScenarioCount>& capacities) {
  for (size_t index = 0; index < kLlmScenarioCount; ++index) {
    const LlmScenarioLimits limits = calculate_llm_scenario_limits(
        model_plan.geometry, kLlmScenarios[index], model_plan.backend);
    if (!limits.valid) {
      return false;
    }
    const LlmScenarioWorkPlan maximum_plan = build_llm_scenario_work_plan(
        model_plan, kLlmScenarios[index], limits.effective_maximum_work_units, config.user_specified_iterations);
    if (!maximum_plan.valid) {
      return false;
    }
    capacities[index] = maximum_plan.plan_identity.size();
  }
  return true;
}

bool add_capacity_bytes(size_t capacity, size_t element_size, size_t& total) noexcept {
  size_t bytes = 0;
  return NumericUtils::checked_multiply(capacity, element_size, bytes) && checked_add_to(bytes, total);
}

bool add_string_backing(const std::string& value, size_t& total) noexcept {
  const size_t inline_capacity = std::string{}.capacity();
  if (value.capacity() <= inline_capacity) {
    return true;
  }
  size_t bytes = 0;
  return NumericUtils::checked_add(value.capacity(), static_cast<size_t>(1), bytes) && checked_add_to(bytes, total);
}

bool add_metal_task_string_backing(const LlmMetalTaskEvidence& metal,
                                   size_t& total) noexcept {
  return add_string_backing(metal.pipeline_label, total) &&
         add_string_backing(metal.grid_plan.reason_code, total) &&
         add_string_backing(metal.grid_plan.identity, total) &&
         add_capacity_bytes(
             metal.grid_plan.threadgroup_accounted_bytes.capacity(),
             sizeof(size_t), total) &&
         add_string_backing(metal.reset_command_status, total) &&
         add_string_backing(metal.timed_command_status, total) &&
         add_string_backing(metal.post_validation_command_status, total) &&
         add_string_backing(metal.checksum_algorithm_version, total) &&
         add_string_backing(metal.error.domain, total) &&
         add_string_backing(metal.error.description, total);
}

LlmRunnerAuxiliaryEstimate calculate_actual_runner_backing(const LlmMemoryResult& result) noexcept {
  LlmRunnerAuxiliaryEstimate actual;
  if (!add_capacity_bytes(result.measurements.capacity(), sizeof(LlmMeasurementState),
                          actual.measurement_record_bytes) ||
      !add_capacity_bytes(result.loops.capacity(), sizeof(LlmLoopRecord), actual.loop_record_bytes)) {
    return actual;
  }

  for (const std::vector<LlmCalibrationAttempt>& attempts : result.calibration_attempts) {
    if (!add_capacity_bytes(attempts.capacity(), sizeof(LlmCalibrationAttempt), actual.calibration_record_bytes)) {
      return actual;
    }
    for (const LlmCalibrationAttempt& attempt : attempts) {
      if (!add_string_backing(attempt.work_plan_identity, actual.calibration_identity_bytes)) {
        return actual;
      }
      if (attempt.execution.metal_evidence_available &&
          attempt.execution.metal.has_value() &&
          !add_metal_task_string_backing(*attempt.execution.metal,
                                         actual.fixed_metadata_bytes)) {
        return actual;
      }
    }
  }

  for (const LlmScenarioAggregate& aggregate : result.aggregates) {
    if (!add_capacity_bytes(aggregate.work_unit_latency_seconds.values.capacity(), sizeof(double),
                            actual.aggregate_value_bytes) ||
        !add_capacity_bytes(aggregate.synthetic_memory_work_units_per_second.values.capacity(), sizeof(double),
                            actual.aggregate_value_bytes) ||
        !add_capacity_bytes(aggregate.effective_model_payload_gb_s.values.capacity(), sizeof(double),
                            actual.aggregate_value_bytes)) {
      return actual;
    }
  }
  if (!add_capacity_bytes(result.statistics_workspace.sorted_values.capacity(), sizeof(double),
                          actual.statistics_workspace_bytes) ||
      !add_capacity_bytes(result.statistics_workspace.absolute_deviations.capacity(), sizeof(double),
                          actual.statistics_workspace_bytes) ||
      !add_capacity_bytes(result.quality_warnings.capacity(), sizeof(std::string_view), actual.warning_record_bytes)) {
    return actual;
  }

  if (!add_string_backing(result.reason_code, actual.fixed_metadata_bytes) ||
      !add_string_backing(result.diagnostic, actual.fixed_metadata_bytes) ||
      !add_string_backing(result.frozen_scenario_plans.reason_code, actual.fixed_metadata_bytes) ||
      !add_string_backing(result.frozen_scenario_plans.model_plan_identity, actual.fixed_metadata_bytes) ||
      !add_string_backing(result.frozen_scenario_plans.plan_identity, actual.fixed_metadata_bytes)) {
    return actual;
  }
  for (const LlmScenarioWorkPlan& scenario : result.frozen_scenario_plans.scenarios) {
    if (!add_string_backing(scenario.reason_code, actual.fixed_metadata_bytes) ||
        !add_string_backing(scenario.model_plan_identity, actual.fixed_metadata_bytes) ||
        !add_string_backing(scenario.plan_identity, actual.fixed_metadata_bytes)) {
      return actual;
    }
  }
  for (const LlmMeasurementState& measurement : result.measurements) {
    if (!add_string_backing(measurement.execution.reason_code,
                            actual.fixed_metadata_bytes)) {
      return actual;
    }
    const LlmExecutorResult* cpu =
        cpu_executor_evidence(measurement.execution);
    if (cpu != nullptr &&
        (!add_string_backing(cpu->reason_code, actual.fixed_metadata_bytes) ||
         !add_capacity_bytes(cpu->expected_checksums.capacity(),
                             sizeof(LlmWorkerChecksum),
                             actual.retained_checksum_bytes) ||
         !add_capacity_bytes(cpu->actual_checksums.capacity(),
                             sizeof(LlmWorkerChecksum),
                             actual.retained_checksum_bytes))) {
      return actual;
    }
    const LlmMetalTaskEvidence* const metal =
        get_llm_metal_task_evidence(measurement.execution);
    if (metal != nullptr &&
        !add_metal_task_string_backing(*metal,
                                       actual.fixed_metadata_bytes)) {
      return actual;
    }
  }

  actual.checksum_auxiliary_bytes = actual.retained_checksum_bytes;
  if (!checked_add_to(actual.measurement_record_bytes, actual.orchestration_auxiliary_bytes) ||
      !checked_add_to(actual.loop_record_bytes, actual.orchestration_auxiliary_bytes) ||
      !checked_add_to(actual.calibration_record_bytes, actual.orchestration_auxiliary_bytes) ||
      !checked_add_to(actual.calibration_identity_bytes, actual.orchestration_auxiliary_bytes) ||
      !checked_add_to(actual.aggregate_value_bytes, actual.orchestration_auxiliary_bytes) ||
      !checked_add_to(actual.statistics_workspace_bytes, actual.orchestration_auxiliary_bytes) ||
      !checked_add_to(actual.warning_record_bytes, actual.orchestration_auxiliary_bytes) ||
      !checked_add_to(actual.fixed_metadata_bytes, actual.orchestration_auxiliary_bytes) ||
      !NumericUtils::checked_add(actual.checksum_auxiliary_bytes, actual.orchestration_auxiliary_bytes,
                                 actual.total_auxiliary_bytes)) {
    return actual;
  }
  actual.valid = true;
  actual.reason_code = LlmExecutorReason::VALID;
  return actual;
}

bool actual_runner_backing_is_covered(const LlmMemoryResult& result) noexcept {
  const LlmRunnerAuxiliaryEstimate actual = calculate_actual_runner_backing(result);
  return actual.valid &&
         actual.orchestration_auxiliary_bytes <= result.runner_auxiliary.orchestration_auxiliary_bytes &&
         actual.checksum_auxiliary_bytes <= result.runner_auxiliary.checksum_auxiliary_bytes;
}

bool initialize_result(const LlmMemoryConfig& config, const LlmMemoryWorkPlan& model_plan,
                       const LlmRunnerAuxiliaryEstimate& auxiliary, LlmMemoryResult& result) {
  result = LlmMemoryResult{};
  result.runner_auxiliary = auxiliary;
  result.counters.planned_loops = config.loop_count;
  if (!NumericUtils::checked_multiply(config.loop_count, kLlmScenarioCount, result.counters.planned_measurements)) {
    result = LlmMemoryResult{};
    return false;
  }

  std::array<size_t, kLlmScenarioCount> identity_capacities{};
  if (!calculate_calibration_identity_capacities(config, model_plan, identity_capacities)) {
    result = LlmMemoryResult{};
    return false;
  }

  result.diagnostic.reserve(kLlmRunnerDiagnosticCapacity);
  result.reason_code.reserve(kLlmRunnerReasonCapacity);
  result.quality_warnings.reserve(kLlmRunnerMaximumWarnings);
  result.statistics_workspace.sorted_values.reserve(config.loop_count);
  result.statistics_workspace.absolute_deviations.reserve(config.loop_count);
  const size_t attempts_per_scenario = calibration_capacity(config);
  for (size_t index = 0; index < kLlmScenarioCount; ++index) {
    result.aggregates[index].scenario = kLlmScenarios[index];
    result.aggregates[index].work_unit_latency_seconds.values.reserve(config.loop_count);
    result.aggregates[index].synthetic_memory_work_units_per_second.values.reserve(config.loop_count);
    result.aggregates[index].effective_model_payload_gb_s.values.reserve(config.loop_count);
    result.calibration_attempts[index].resize(attempts_per_scenario);
    for (LlmCalibrationAttempt& attempt : result.calibration_attempts[index]) {
      attempt.scenario = kLlmScenarios[index];
      attempt.work_unit_kind = model_plan.work_unit_kind;
      attempt.kv_write_kind =
          llm_kv_write_kind_for(model_plan.phase, attempt.scenario);
      attempt.work_plan_identity.reserve(identity_capacities[index]);
    }
  }
  result.loops.reserve(config.loop_count);
  result.measurements.reserve(result.counters.planned_measurements);
  const LlmCpuExecutionPlan* cpu_plan =
      get_llm_cpu_execution_plan(model_plan);
  for (size_t loop_index = 0; loop_index < config.loop_count; ++loop_index) {
    LlmLoopRecord loop;
    loop.loop_index = loop_index;
    loop.planned_order = build_llm_scenario_order(loop_index);
    for (size_t position = 0; position < kLlmScenarioCount; ++position) {
      LlmMeasurementState measurement;
      measurement.scenario = loop.planned_order[position];
      measurement.work_unit_kind = model_plan.work_unit_kind;
      measurement.kv_write_kind =
          llm_kv_write_kind_for(model_plan.phase, measurement.scenario);
      measurement.loop_index = loop_index;
      measurement.order_position = position;
      measurement.requested_workers =
          cpu_plan == nullptr ? 0 : cpu_plan->requested_workers;
      measurement.effective_workers =
          cpu_plan == nullptr ? 0 : cpu_plan->effective_workers;
      measurement.working_set_bytes = model_plan.geometry.total_data_mapping_bytes;
      measurement.execution.reason_code.reserve(kLlmRunnerReasonCapacity);
      if (cpu_plan != nullptr) {
        measurement.execution.backend_evidence = LlmCpuTaskEvidence{};
        auto& cpu = std::get<LlmCpuTaskEvidence>(
            measurement.execution.backend_evidence).executor;
        cpu.reason_code.reserve(kLlmRunnerReasonCapacity);
        cpu.expected_checksums.reserve(cpu_plan->effective_workers);
        cpu.actual_checksums.reserve(cpu_plan->effective_workers);
      }
      loop.measurement_indexes[position] = result.measurements.size();
      result.measurements.push_back(std::move(measurement));
    }
    result.loops.push_back(std::move(loop));
  }
  if (!actual_runner_backing_is_covered(result)) {
    result = LlmMemoryResult{};
    result.reason_code = LlmRunnerReason::AUXILIARY_BUDGET_INSUFFICIENT;
    return false;
  }
  result.initialized = true;
  return true;
}

int finish_unsupported_run(
    const LlmMemoryConfig& config, const LlmMemoryWorkPlan& model_plan,
    const LlmRunnerAuxiliaryEstimate& auxiliary, LlmMemoryResult& result,
    const LlmRunnerHooks& hooks, std::string_view reason_code,
    LlmBackend& backend, bool& release_attempted) {
  if (!initialize_result(config, model_plan, auxiliary, result)) {
    if (result.reason_code != LlmRunnerReason::AUXILIARY_BUDGET_INSUFFICIENT) {
      result.reason_code = LlmRunnerReason::PLANNED_COUNTER_OVERFLOW;
    }
    release_backend_once(backend, release_attempted, result);
    return EXIT_FAILURE;
  }
  const std::string_view canonical_reason =
      canonicalize_llm_result_reason_code(reason_code);
  result.status = LlmRunStatus::Unsupported;
  result.reason_code =
      canonical_reason == LlmRunnerReason::RUNNER_UNKNOWN_EXCEPTION
          ? std::string_view(LlmBackendReason::TASK_UNSUPPORTED)
          : canonical_reason;
  return finish_terminal(result, hooks, backend, release_attempted);
}

int finish_failed_backend_transition(
    const LlmMemoryConfig& config, const LlmMemoryWorkPlan& model_plan,
    const LlmRunnerAuxiliaryEstimate& auxiliary, LlmMemoryResult& result,
    const LlmRunnerHooks& hooks, std::string_view reason_code,
    std::string_view fallback_reason, LlmBackend& backend,
    bool& release_attempted) {
  if (!initialize_result(config, model_plan, auxiliary, result)) {
    if (result.reason_code != LlmRunnerReason::AUXILIARY_BUDGET_INSUFFICIENT) {
      result.reason_code = LlmRunnerReason::PLANNED_COUNTER_OVERFLOW;
    }
    release_backend_once(backend, release_attempted, result);
    return EXIT_FAILURE;
  }
  const std::string_view canonical_reason =
      canonicalize_llm_result_reason_code(
          reason_code.empty() ? fallback_reason : reason_code);
  finalize_failure(
      result,
      canonical_reason == LlmRunnerReason::RUNNER_UNKNOWN_EXCEPTION
          ? fallback_reason
          : canonical_reason,
      false);
  return finish_terminal(result, hooks, backend, release_attempted);
}

struct ExcludedTaskOutcome {
  bool accepted = false;
  std::string_view reason_code = LlmRunnerReason::NOT_STARTED;
  double elapsed_seconds = 0.0;
};

ExcludedTaskOutcome execute_excluded_task(const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan& task_plan,
                                          LlmRunnerTaskKind kind, std::string_view purpose,
                                          LlmBackend& backend, LlmMemoryResult& result) {
  const size_t index = scenario_index(task_plan.scenario);
  if (index >= kLlmScenarioCount) {
    throw std::runtime_error("runner calibration scenario invalid");
  }
  const size_t attempt_index = result.calibration_attempt_counts[index];
  if (attempt_index >= result.calibration_attempts[index].size()) {
    throw std::runtime_error("runner calibration storage exhausted");
  }
  LlmRunnerTaskContext context;
  context.kind = kind;
  context.purpose = canonical_calibration_purpose(purpose);
  context.scenario = task_plan.scenario;
  context.attempt_index = attempt_index;

  LlmCalibrationAttempt& attempt = result.calibration_attempts[index][attempt_index];
  ++result.calibration_attempt_counts[index];
  attempt.scenario = task_plan.scenario;
  attempt.work_unit_kind = task_plan.work_unit_kind;
  attempt.kv_write_kind = task_plan.kv_write_kind;
  attempt.purpose = canonical_calibration_purpose(purpose);
  attempt.explicit_iterations = task_plan.explicit_iterations;
  attempt.work_units = task_plan.work_units;
  attempt.weight_read_bytes = task_plan.weight_read_bytes;
  attempt.kv_read_bytes = task_plan.kv_read_bytes;
  attempt.kv_write_bytes = task_plan.kv_write_bytes;
  attempt.effective_model_payload_bytes = task_plan.effective_model_payload_bytes;
  attempt.layout_metadata_lookup_count =
      task_plan.layout_metadata_lookup_count;
  attempt.layout_metadata_read_bytes = task_plan.layout_metadata_read_bytes;
  attempt.task_accounted_bytes = task_plan.task_accounted_bytes;
  attempt.work_plan_identity.assign(task_plan.plan_identity);
  try {
    LlmTaskExecutionResult execution =
        backend.execute_task(model_plan, task_plan, context);
    attempt.execution = compact_execution(execution);
    attempt.duration_quality =
        classify_llm_duration_quality(
            execution.timing.elapsed_seconds, task_plan.work_units,
            calculate_llm_scenario_limits(
                model_plan.geometry, task_plan.scenario,
                model_plan.backend));
    attempt.terminal = true;
    attempt.valid = execution_is_accepted(execution, model_plan, task_plan,
                                          context);
    attempt.reason_code =
        attempt.valid
            ? std::string_view(LlmBackendReason::VALID)
            : execution_failure_reason(execution, model_plan, task_plan,
                                       context);
    attempt.execution.valid = attempt.valid;
    attempt.execution.reason_code = attempt.reason_code;

    ExcludedTaskOutcome outcome;
    outcome.accepted = attempt.valid;
    outcome.reason_code = attempt.reason_code;
    outcome.elapsed_seconds = execution.timing.elapsed_seconds;
    return outcome;
  } catch (const std::exception&) {
    attempt.terminal = true;
    attempt.valid = false;
    attempt.reason_code = LlmRunnerReason::RUNNER_EXCEPTION;
    attempt.execution.reason_code = LlmRunnerReason::RUNNER_EXCEPTION;
    throw;
  } catch (...) {
    attempt.terminal = true;
    attempt.valid = false;
    attempt.reason_code = LlmRunnerReason::RUNNER_UNKNOWN_EXCEPTION;
    attempt.execution.reason_code = LlmRunnerReason::RUNNER_UNKNOWN_EXCEPTION;
    throw;
  }
}

CalibrationOutcome stop_or_failure_after_excluded(const ExcludedTaskOutcome& outcome, LlmMemoryResult& result,
                                                  const LlmRunnerHooks& hooks) {
  const bool stop_after_task = stop_requested(hooks);
  if (!outcome.accepted) {
    result.interruption_requested = stop_after_task;
    result.status = LlmRunStatus::Failed;
    result.reason_code = outcome.reason_code;
    return CalibrationOutcome::Failed;
  }
  if (stop_after_task) {
    result.interruption_requested = true;
    return CalibrationOutcome::Interrupted;
  }
  return CalibrationOutcome::Success;
}

CalibrationOutcome run_excluded(const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan& task_plan,
                                LlmRunnerTaskKind kind, std::string_view purpose, LlmBackend& backend,
                                LlmMemoryResult& result, const LlmRunnerHooks& hooks,
                                double* elapsed_seconds = nullptr) {
  if (stop_requested(hooks)) {
    result.interruption_requested = true;
    return CalibrationOutcome::Interrupted;
  }
  const ExcludedTaskOutcome outcome =
      execute_excluded_task(model_plan, task_plan, kind, purpose, backend,
                            result);
  if (elapsed_seconds != nullptr) {
    *elapsed_seconds = outcome.elapsed_seconds;
  }
  return stop_or_failure_after_excluded(outcome, result, hooks);
}

CalibrationOutcome calibrate_scenario(const LlmMemoryWorkPlan& model_plan,
                                      LlmScenario scenario,
                                      LlmBackend& backend,
                                      LlmMemoryResult& result,
                                      const LlmRunnerHooks& hooks,
                                      size_t& frozen_work_units) {
  const LlmScenarioLimits limits = calculate_llm_scenario_limits(
      model_plan.geometry, scenario, model_plan.backend);
  if (!limits.valid) {
    result.status = LlmRunStatus::Failed;
    result.reason_code = limits.reason_code;
    return CalibrationOutcome::Failed;
  }

  size_t work_units = calculate_llm_pilot_work_units(limits);
  if (work_units == 0) {
    result.status = LlmRunStatus::Failed;
    result.reason_code = LlmRunnerReason::CALIBRATION_SCALING_FAILED;
    return CalibrationOutcome::Failed;
  }
  LlmScenarioWorkPlan latest_plan = build_llm_scenario_work_plan(model_plan, scenario, work_units, false);
  if (!latest_plan.valid) {
    result.status = LlmRunStatus::Failed;
    result.reason_code = latest_plan.reason_code;
    return CalibrationOutcome::Failed;
  }
  CalibrationOutcome outcome = run_excluded(
      model_plan, latest_plan, LlmRunnerTaskKind::Warmup,
      "calibration_shape_warmup", backend, result, hooks);
  if (outcome != CalibrationOutcome::Success) {
    return outcome;
  }
  bool single_unit_warmed = latest_plan.work_units == 1;
  double elapsed_seconds = 0.0;
  outcome = run_excluded(model_plan, latest_plan,
                         LlmRunnerTaskKind::Calibration, "pilot", backend,
                         result, hooks,
                         &elapsed_seconds);
  if (outcome != CalibrationOutcome::Success) {
    return outcome;
  }

  for (size_t correction = 0;
       !llm_duration_in_target_window(elapsed_seconds) &&
       correction < Constants::LLM_CALIBRATION_MAX_CORRECTIONS;
       ++correction) {
    if (latest_plan.work_units == 1 &&
        elapsed_seconds > Constants::LLM_CALIBRATION_MAX_SECONDS) {
      break;
    }
    const size_t corrected_work_units = calculate_llm_calibrated_work_units(
        elapsed_seconds, latest_plan.work_units, limits);
    if (corrected_work_units == 0) {
      result.status = LlmRunStatus::Failed;
      result.reason_code = LlmRunnerReason::CALIBRATION_SCALING_FAILED;
      return CalibrationOutcome::Failed;
    }
    if (corrected_work_units == latest_plan.work_units) {
      break;
    }
    latest_plan = build_llm_scenario_work_plan(model_plan, scenario, corrected_work_units, false);
    if (!latest_plan.valid) {
      result.status = LlmRunStatus::Failed;
      result.reason_code = latest_plan.reason_code;
      return CalibrationOutcome::Failed;
    }
    const bool confirms_single_unit = latest_plan.work_units == 1;
    if (confirms_single_unit && !single_unit_warmed) {
      outcome = run_excluded(
          model_plan, latest_plan, LlmRunnerTaskKind::Warmup,
          "single_unit_confirmation_warmup", backend, result, hooks);
      if (outcome != CalibrationOutcome::Success) {
        return outcome;
      }
      single_unit_warmed = true;
    }
    const std::string_view purpose =
        confirms_single_unit ? "single_unit_confirmation" : "correction";
    outcome = run_excluded(model_plan, latest_plan,
                           LlmRunnerTaskKind::Calibration, purpose, backend,
                           result, hooks,
                           &elapsed_seconds);
    if (outcome != CalibrationOutcome::Success) {
      return outcome;
    }
  }

  frozen_work_units = latest_plan.work_units;
  return CalibrationOutcome::Success;
}

CalibrationOutcome warm_frozen_scenarios(
    const LlmMemoryWorkPlan& model_plan, LlmBackend& backend,
    LlmMemoryResult& result, const LlmRunnerHooks& hooks) {
  if (!result.frozen_scenario_plans.valid) {
    result.status = LlmRunStatus::Failed;
    result.reason_code = LlmRunnerReason::FROZEN_PLAN_MISMATCH;
    return CalibrationOutcome::Failed;
  }
  for (size_t index = 0; index < kLlmScenarioCount; ++index) {
    const LlmScenarioWorkPlan& frozen =
        result.frozen_scenario_plans.scenarios[index];
    if (!frozen.valid || frozen.scenario != kLlmScenarios[index]) {
      result.status = LlmRunStatus::Failed;
      result.reason_code = LlmRunnerReason::FROZEN_PLAN_MISMATCH;
      return CalibrationOutcome::Failed;
    }
    const CalibrationOutcome outcome = run_excluded(
        model_plan, frozen, LlmRunnerTaskKind::Warmup,
        "frozen_measurement_warmup", backend, result, hooks);
    if (outcome != CalibrationOutcome::Success) {
      return outcome;
    }
  }
  return CalibrationOutcome::Success;
}

bool assign_frozen_plans(LlmMemoryResult& result, const LlmMemoryWorkPlan& model_plan) {
  size_t total_work_units = 0;
  size_t total_model_payload = 0;
  size_t total_layout_metadata_lookups = 0;
  size_t total_layout_metadata_bytes = 0;
  size_t total_task_accounted_bytes = 0;
  for (LlmMeasurementState& measurement : result.measurements) {
    const size_t index = scenario_index(measurement.scenario);
    if (index >= kLlmScenarioCount) {
      return false;
    }
    const LlmScenarioWorkPlan& plan = result.frozen_scenario_plans.scenarios[index];
    measurement.frozen_plan_index = index;
    measurement.work_unit_kind = plan.work_unit_kind;
    measurement.kv_write_kind = plan.kv_write_kind;
    measurement.explicit_iterations = plan.explicit_iterations;
    measurement.planned_work_units = plan.work_units;
    measurement.weight_read_bytes_per_work_unit = plan.weight_read_bytes_per_work_unit;
    measurement.kv_read_bytes_per_work_unit = plan.kv_read_bytes_per_work_unit;
    measurement.kv_write_bytes_per_work_unit = plan.kv_write_bytes_per_work_unit;
    measurement.effective_model_payload_bytes_per_work_unit = plan.effective_model_payload_bytes_per_work_unit;
    measurement.layout_metadata_lookup_count_per_work_unit =
        plan.layout_metadata_lookup_count_per_work_unit;
    measurement.layout_metadata_read_bytes_per_work_unit =
        plan.layout_metadata_read_bytes_per_work_unit;
    measurement.accounted_bytes_per_work_unit =
        plan.accounted_bytes_per_work_unit;
    measurement.planned_weight_read_bytes = plan.weight_read_bytes;
    measurement.planned_kv_read_bytes = plan.kv_read_bytes;
    measurement.planned_kv_write_bytes = plan.kv_write_bytes;
    measurement.planned_effective_model_payload_bytes = plan.effective_model_payload_bytes;
    measurement.planned_layout_metadata_lookup_count =
        plan.layout_metadata_lookup_count;
    measurement.planned_layout_metadata_read_bytes =
        plan.layout_metadata_read_bytes;
    measurement.planned_task_accounted_bytes = plan.task_accounted_bytes;
    measurement.calibration_attempt_count = result.calibration_attempt_counts[index];

    if (measurement.scenario == LlmScenario::Mixed && plan.effective_model_payload_bytes_per_work_unit != 0) {
      const long double total = static_cast<long double>(plan.effective_model_payload_bytes_per_work_unit);
      measurement.weight_payload_fraction =
          static_cast<double>(static_cast<long double>(plan.weight_read_bytes_per_work_unit) / total);
      measurement.kv_read_payload_fraction =
          static_cast<double>(static_cast<long double>(plan.kv_read_bytes_per_work_unit) / total);
      measurement.kv_write_payload_fraction =
          static_cast<double>(static_cast<long double>(plan.kv_write_bytes_per_work_unit) / total);
    }

    if (!checked_add_to(plan.work_units, total_work_units) ||
        !checked_add_to(plan.effective_model_payload_bytes,
                        total_model_payload) ||
        !checked_add_to(plan.layout_metadata_lookup_count,
                        total_layout_metadata_lookups) ||
        !checked_add_to(plan.layout_metadata_read_bytes,
                        total_layout_metadata_bytes) ||
        !checked_add_to(plan.task_accounted_bytes,
                        total_task_accounted_bytes)) {
      return false;
    }
  }
  result.counters.planned_work_units = total_work_units;
  result.counters.planned_effective_model_payload_bytes = total_model_payload;
  result.counters.planned_layout_metadata_lookup_count =
      total_layout_metadata_lookups;
  result.counters.planned_layout_metadata_read_bytes =
      total_layout_metadata_bytes;
  result.counters.planned_task_accounted_bytes = total_task_accounted_bytes;
  return model_plan.valid;
}

void retain_task_evidence(LlmTaskExecutionResult& retained,
                          LlmTaskExecutionResult& execution,
                          const LlmMemoryWorkPlan& model_plan) {
  if (auto* metal =
          std::get_if<LlmMetalTaskEvidence>(&execution.backend_evidence)) {
    retained.backend_evidence = std::move(*metal);
    return;
  }
  auto* source = std::get_if<LlmCpuTaskEvidence>(&execution.backend_evidence);
  if (source == nullptr) {
    retained.backend_evidence = std::monostate{};
    return;
  }
  auto* target = std::get_if<LlmCpuTaskEvidence>(&retained.backend_evidence);
  if (target == nullptr) {
    retained.backend_evidence = LlmCpuTaskEvidence{};
    target = std::get_if<LlmCpuTaskEvidence>(&retained.backend_evidence);
  }
  LlmExecutorResult& destination = target->executor;
  LlmExecutorResult& input = source->executor;
  destination.valid = input.valid;
  destination.reason_code.assign(
      canonicalize_llm_result_reason_code(input.reason_code));
  destination.elapsed_seconds = input.elapsed_seconds;
  destination.requested_workers = input.requested_workers;
  destination.created_workers = input.created_workers;
  destination.completed_workers = input.completed_workers;
  destination.qos_successful_workers = input.qos_successful_workers;
  destination.qos_failed_workers = input.qos_failed_workers;
  destination.worker_startup_failed = input.worker_startup_failed;
  destination.kernel_succeeded = input.kernel_succeeded;
  destination.timer_started = input.timer_started;
  destination.timer_stopped = input.timer_stopped;
  destination.checksum_evaluated = input.checksum_evaluated;
  destination.checksum_valid = input.checksum_valid;
  destination.post_validation_evaluated =
      input.post_validation_evaluated;
  destination.post_validation_valid = input.post_validation_valid;
  destination.expected_run_checksum = input.expected_run_checksum;
  destination.actual_run_checksum = input.actual_run_checksum;
  const LlmCpuExecutionPlan* cpu_plan =
      get_llm_cpu_execution_plan(model_plan);
  const size_t effective_workers =
      cpu_plan == nullptr ? 0 : cpu_plan->effective_workers;
  if (input.expected_checksums.size() == effective_workers &&
      input.actual_checksums.size() == effective_workers) {
    destination.expected_checksums.assign(input.expected_checksums.begin(),
                                          input.expected_checksums.end());
    destination.actual_checksums.assign(input.actual_checksums.begin(),
                                        input.actual_checksums.end());
  } else {
    destination.expected_checksums.clear();
    destination.actual_checksums.clear();
  }
}

void populate_measurement(LlmMeasurementState& measurement,
                          LlmTaskExecutionResult execution,
                          const LlmMemoryWorkPlan& model_plan,
                          const LlmScenarioWorkPlan& task_plan,
                          const LlmRunnerTaskContext& context) {
  const bool accepted =
      execution_is_accepted(execution, model_plan, task_plan, context);
  const std::string_view failure_reason =
      accepted
          ? std::string_view(LlmBackendReason::VALID)
          : execution_failure_reason(execution, model_plan, task_plan,
                                     context);
  LlmTaskExecutionResult& retained = measurement.execution;
  retained.status = execution.status;
  retained.reason_code.assign(accepted ? LlmBackendReason::VALID
                                       : failure_reason);
  retained.identity = {};
  retained.timing = execution.timing;
  retained.completion = execution.completion;
  retained.validation = execution.validation;
  retain_task_evidence(retained, execution, model_plan);

  const LlmExecutorResult* cpu = cpu_executor_evidence(retained);
  if (cpu != nullptr) {
    measurement.qos_successful_workers = cpu->qos_successful_workers;
    measurement.qos_failed_workers = cpu->qos_failed_workers;
  }
  measurement.duration_quality =
      classify_llm_duration_quality(
          measurement.execution.timing.elapsed_seconds,
          task_plan.work_units,
          calculate_llm_scenario_limits(
              model_plan.geometry, task_plan.scenario,
              model_plan.backend));
  if (!accepted) {
    measurement.reason_code = failure_reason;
    measurement.execution.status = execution.status;
    measurement.status =
        execution_failure_status(execution, measurement.reason_code);
    clear_measurement_values(measurement);
    measurement.execution_evidence_available = true;
    return;
  }

  const long double elapsed =
      static_cast<long double>(measurement.execution.timing.elapsed_seconds);
  const long double work_units = static_cast<long double>(task_plan.work_units);
  const long double latency = elapsed / work_units;
  const long double work_units_per_second = work_units / elapsed;
  const long double bandwidth = static_cast<long double>(task_plan.effective_model_payload_bytes) / elapsed / 1.0e9L;
  const double latency_value = static_cast<double>(latency);
  const double work_units_per_second_value = static_cast<double>(work_units_per_second);
  const double bandwidth_value = static_cast<double>(bandwidth);
  if (!std::isfinite(latency_value) || latency_value <= 0.0 || !std::isfinite(work_units_per_second_value) ||
      work_units_per_second_value <= 0.0 || !std::isfinite(bandwidth_value) || bandwidth_value <= 0.0) {
    measurement.status = LlmMeasurementStatus::Invalid;
    measurement.reason_code = LlmRunnerReason::INVALID_DERIVED_METRIC;
    clear_measurement_values(measurement);
    measurement.execution_evidence_available = true;
    return;
  }

  measurement.status = LlmMeasurementStatus::Measured;
  measurement.reason_code = "measured";
  measurement.completed_work_units = task_plan.work_units;
  measurement.completed_effective_model_payload_bytes = task_plan.effective_model_payload_bytes;
  measurement.completed_layout_metadata_lookup_count =
      task_plan.layout_metadata_lookup_count;
  measurement.completed_layout_metadata_read_bytes =
      task_plan.layout_metadata_read_bytes;
  measurement.completed_task_accounted_bytes = task_plan.task_accounted_bytes;
  measurement.elapsed_seconds = measurement.execution.timing.elapsed_seconds;
  measurement.synthetic_work_unit_latency_seconds = latency_value;
  measurement.synthetic_memory_work_units_per_second = work_units_per_second_value;
  measurement.effective_model_payload_gb_s = bandwidth_value;
  measurement.checksum_valid = true;
  measurement.execution_evidence_available = true;
}

void record_terminal_measurement(LlmMemoryResult& result, const LlmLoopRecord& loop,
                                 const LlmMeasurementState& measurement) {
  ++result.counters.terminal_measurements;
  if (measurement.status != LlmMeasurementStatus::Measured) {
    return;
  }
  ++result.counters.measured_measurements;
  result.counters.completed_work_units += measurement.completed_work_units;
  result.counters.completed_effective_model_payload_bytes += measurement.completed_effective_model_payload_bytes;
  result.counters.completed_layout_metadata_lookup_count +=
      measurement.completed_layout_metadata_lookup_count;
  result.counters.completed_layout_metadata_read_bytes +=
      measurement.completed_layout_metadata_read_bytes;
  result.counters.completed_task_accounted_bytes +=
      measurement.completed_task_accounted_bytes;
  if (loop.realized_order_count == kLlmScenarioCount) {
    ++result.counters.completed_loops;
  }
  update_scenario_aggregate(result, measurement);
}

int fail_initialized_run(LlmMemoryResult& result, const LlmRunnerHooks& hooks,
                         std::string_view reason_code, LlmBackend& backend,
                         bool& release_attempted);

int run_measurements(const LlmMemoryWorkPlan& model_plan, LlmBackend& backend,
                     LlmMemoryResult& result,
                     const LlmRunnerHooks& hooks,
                     bool& release_attempted) {
  for (LlmLoopRecord& loop : result.loops) {
    for (size_t position = 0; position < kLlmScenarioCount; ++position) {
      if (stop_requested(hooks)) {
        result.interruption_requested = true;
        finalize_remaining_interrupted(result);
        return finish_terminal(result, hooks, backend, release_attempted);
      }
      LlmMeasurementState& measurement = result.measurements[loop.measurement_indexes[position]];
      const size_t index = scenario_index(measurement.scenario);
      const LlmScenarioWorkPlan& task_plan = result.frozen_scenario_plans.scenarios[index];
      if (loop.realized_order_count == 0) {
        ++result.counters.attempted_loops;
      }
      measurement.attempted = true;
      ++result.counters.attempted_measurements;
      loop.realized_order[loop.realized_order_count++] = measurement.scenario;
      bool measurement_checkpoint_attempted = false;
      try {
        LlmRunnerTaskContext context;
        context.kind = LlmRunnerTaskKind::Measurement;
        context.purpose = "measurement";
        context.scenario = measurement.scenario;
        context.loop_index = loop.loop_index;
        context.order_position = position;
        LlmTaskExecutionResult execution =
            backend.execute_task(model_plan, task_plan, context);
        populate_measurement(measurement, std::move(execution), model_plan,
                             task_plan, context);
        record_terminal_measurement(result, loop, measurement);

        const bool stop_after_task = stop_requested(hooks);
        if (stop_after_task) {
          result.interruption_requested = true;
        }
        const bool task_failed =
            measurement.status == LlmMeasurementStatus::Invalid || measurement.status == LlmMeasurementStatus::Failed;
        if (task_failed) {
          finalize_failure(result, measurement.reason_code, stop_after_task);
        } else if (stop_after_task) {
          finalize_remaining_interrupted(result);
          update_completion_state(result);
        } else {
          update_completion_state(result);
        }

        measurement_checkpoint_attempted = true;
        if (!invoke_checkpoint(result, LlmCheckpointKind::MeasurementTerminal, hooks)) {
          const bool pending_stop = stop_after_task || pending_stop_after_failure(result, hooks);
          finalize_failure(result, result.reason_code, pending_stop);
          release_backend_once(backend, release_attempted, result);
          return EXIT_FAILURE;
        }

        bool post_checkpoint_stop = stop_after_task;
        if (!post_checkpoint_stop) {
          post_checkpoint_stop = stop_requested(hooks);
        }
        if (post_checkpoint_stop) {
          result.interruption_requested = true;
          finalize_remaining_interrupted(result);
          update_completion_state(result);
        }
        if (task_failed || post_checkpoint_stop) {
          return finish_terminal(result, hooks, backend,
                                 release_attempted);
        }
      } catch (const std::exception& error) {
        assign_diagnostic(result, error.what());
        bool pending_stop = pending_stop_after_failure(result, hooks);
        finalize_failure(result, LlmRunnerReason::RUNNER_EXCEPTION, pending_stop);
        if (!measurement_checkpoint_attempted) {
          const bool checkpoint_succeeded = invoke_checkpoint(result, LlmCheckpointKind::MeasurementTerminal, hooks);
          pending_stop = pending_stop || pending_stop_after_failure(result, hooks);
          finalize_failure(result, result.reason_code, pending_stop);
          if (!checkpoint_succeeded) {
            release_backend_once(backend, release_attempted, result);
            return EXIT_FAILURE;
          }
        }
        return finish_terminal(result, hooks, backend, release_attempted);
      } catch (...) {
        result.diagnostic.clear();
        bool pending_stop = pending_stop_after_failure(result, hooks);
        finalize_failure(result, LlmRunnerReason::RUNNER_UNKNOWN_EXCEPTION, pending_stop);
        if (!measurement_checkpoint_attempted) {
          const bool checkpoint_succeeded = invoke_checkpoint(result, LlmCheckpointKind::MeasurementTerminal, hooks);
          pending_stop = pending_stop || pending_stop_after_failure(result, hooks);
          finalize_failure(result, result.reason_code, pending_stop);
          if (!checkpoint_succeeded) {
            release_backend_once(backend, release_attempted, result);
            return EXIT_FAILURE;
          }
        }
        return finish_terminal(result, hooks, backend, release_attempted);
      }
    }
  }
  return finish_terminal(result, hooks, backend, release_attempted);
}

void trim_calibration_attempts(LlmMemoryResult& result) {
  for (size_t index = 0; index < kLlmScenarioCount; ++index) {
    if (result.calibration_attempts[index].size() > result.calibration_attempt_counts[index]) {
      result.calibration_attempts[index].resize(result.calibration_attempt_counts[index]);
    }
  }
}

int fail_initialized_run(LlmMemoryResult& result, const LlmRunnerHooks& hooks,
                         std::string_view reason_code, LlmBackend& backend,
                         bool& release_attempted) {
  trim_calibration_attempts(result);
  finalize_failure(result, reason_code, result.interruption_requested);
  return finish_terminal(result, hooks, backend, release_attempted);
}

}  // namespace

std::string_view canonicalize_llm_result_reason_code(std::string_view reason_code) noexcept {
  for (std::string_view candidate : kLlmRunnerReasons) {
    if (reason_code == candidate) {
      return candidate;
    }
  }
  for (std::string_view candidate : kLlmWorkPlanReasons) {
    if (reason_code == candidate) {
      return candidate;
    }
  }
  for (std::string_view candidate : kLlmExecutorReasons) {
    if (reason_code == candidate) {
      return candidate;
    }
  }
  for (std::string_view candidate : kLlmBackendReasons) {
    if (reason_code == candidate) {
      return candidate;
    }
  }
  for (std::string_view candidate : kLlmMetalPlanReasons) {
    if (reason_code == candidate) {
      return candidate;
    }
  }
  return LlmRunnerReason::RUNNER_UNKNOWN_EXCEPTION;
}

LlmRunnerAuxiliaryEstimate calculate_llm_runner_auxiliary_estimate(
    const LlmMemoryConfig& config,
    const LlmAuxiliaryPreflightView& preflight) noexcept {
  LlmRunnerAuxiliaryEstimate estimate;
  try {
    const bool backend_shape_valid =
        preflight.backend == config.backend &&
        preflight.kv_layout == config.kv_layout &&
        ((preflight.backend == LlmMemoryBackend::Cpu &&
          preflight.effective_workers != 0) ||
         (preflight.backend == LlmMemoryBackend::Metal &&
          preflight.effective_workers == 0 &&
          is_activated_metal_profile(config.phase,
                                     config.kv_layout)));
    if (!preflight.valid || config.loop_count == 0 ||
        !backend_shape_valid) {
      estimate.reason_code = LlmRunnerReason::INVALID_MODEL_WORK_PLAN;
      return estimate;
    }

    size_t planned_measurements = 0;
    const size_t attempts_per_scenario = calibration_capacity(config);
    size_t total_calibration_attempts = 0;
    size_t aggregate_value_count = 0;
    size_t retained_worker_checksums = 0;
    if (!NumericUtils::checked_multiply(
            config.loop_count, kLlmScenarioCount, planned_measurements) ||
        !NumericUtils::checked_multiply(
            attempts_per_scenario, kLlmScenarioCount,
            total_calibration_attempts) ||
        !NumericUtils::checked_multiply(
            planned_measurements, static_cast<size_t>(3),
            aggregate_value_count) ||
        !NumericUtils::checked_multiply(
            planned_measurements, preflight.effective_workers,
            retained_worker_checksums) ||
        !NumericUtils::checked_multiply(
            retained_worker_checksums, static_cast<size_t>(2),
            retained_worker_checksums) ||
        !NumericUtils::checked_multiply(
            planned_measurements, sizeof(LlmMeasurementState),
            estimate.measurement_record_bytes) ||
        !NumericUtils::checked_multiply(
            config.loop_count, sizeof(LlmLoopRecord),
            estimate.loop_record_bytes) ||
        !NumericUtils::checked_multiply(
            total_calibration_attempts, sizeof(LlmCalibrationAttempt),
            estimate.calibration_record_bytes) ||
        !NumericUtils::checked_multiply(
            aggregate_value_count, sizeof(double),
            estimate.aggregate_value_bytes) ||
        !NumericUtils::checked_multiply(
            config.loop_count, static_cast<size_t>(2 * sizeof(double)),
            estimate.statistics_workspace_bytes) ||
        !NumericUtils::checked_multiply(
            kLlmRunnerMaximumWarnings, sizeof(std::string_view),
            estimate.warning_record_bytes) ||
        !NumericUtils::checked_multiply(
            retained_worker_checksums, sizeof(LlmWorkerChecksum),
            estimate.retained_checksum_bytes)) {
      return estimate;
    }

    size_t maximum_active_identity = 0;
    for (size_t index = 0; index < kLlmScenarioCount; ++index) {
      const size_t identity_capacity =
          preflight.maximum_scenario_plan_identity_bytes[index];
      maximum_active_identity =
          std::max(maximum_active_identity, identity_capacity);
      size_t identity_with_null = 0;
      size_t conservative_identity = 0;
      size_t scenario_identity_total = 0;
      if (!NumericUtils::checked_add(
              identity_capacity, static_cast<size_t>(1),
              identity_with_null) ||
          !NumericUtils::checked_multiply(
              identity_with_null, static_cast<size_t>(2),
              conservative_identity) ||
          !NumericUtils::checked_multiply(
              conservative_identity, attempts_per_scenario,
              scenario_identity_total) ||
          !checked_add_to(scenario_identity_total,
                          estimate.calibration_identity_bytes)) {
        return estimate;
      }
    }

    const auto add_conservative_length =
        [&](size_t length, size_t& total) {
          size_t with_null = 0;
          size_t doubled = 0;
          return NumericUtils::checked_add(
                     length, static_cast<size_t>(1), with_null) &&
                 NumericUtils::checked_multiply(
                     with_null, static_cast<size_t>(2), doubled) &&
                 checked_add_to(doubled, total);
        };
    if (!add_conservative_length(
            preflight.frozen_reason_code_bytes,
            estimate.fixed_metadata_bytes) ||
        !add_conservative_length(
            preflight.frozen_model_plan_identity_bytes,
            estimate.fixed_metadata_bytes) ||
        !add_conservative_length(
            preflight.frozen_plan_identity_bytes,
            estimate.fixed_metadata_bytes)) {
      return estimate;
    }
    for (size_t index = 0; index < kLlmScenarioCount; ++index) {
      if (!add_conservative_length(
              preflight.frozen_scenario_reason_code_bytes[index],
              estimate.fixed_metadata_bytes) ||
          !add_conservative_length(
              preflight.frozen_scenario_model_plan_identity_bytes[index],
              estimate.fixed_metadata_bytes) ||
          !add_conservative_length(
              preflight.frozen_scenario_plan_identity_bytes[index],
              estimate.fixed_metadata_bytes)) {
        return estimate;
      }
    }

    size_t active_plan_string_bytes = 0;
    size_t active_plan_identity_bytes = 0;
    size_t active_model_identity_bytes = 0;
    size_t measurement_reason_bytes = 0;
    if (!NumericUtils::checked_add(
            maximum_active_identity, static_cast<size_t>(1),
            active_plan_identity_bytes) ||
        !NumericUtils::checked_add(
            preflight.model_plan_identity_bytes, static_cast<size_t>(1),
            active_model_identity_bytes) ||
        !NumericUtils::checked_add(
            active_plan_identity_bytes, active_model_identity_bytes,
            active_plan_string_bytes) ||
        !NumericUtils::checked_multiply(
            active_plan_string_bytes, static_cast<size_t>(4),
            active_plan_string_bytes) ||
        !checked_add_to(active_plan_string_bytes,
                        estimate.fixed_metadata_bytes) ||
        !NumericUtils::checked_multiply(
            planned_measurements,
            2 * (kLlmRunnerReasonCapacity + 1),
            measurement_reason_bytes) ||
        !checked_add_to(measurement_reason_bytes,
                        estimate.fixed_metadata_bytes) ||
        !checked_add_to(2 * (kLlmRunnerReasonCapacity + 1),
                        estimate.fixed_metadata_bytes) ||
        !checked_add_to(2 * (kLlmRunnerDiagnosticCapacity + 1),
                        estimate.fixed_metadata_bytes)) {
      return estimate;
    }
    if (preflight.backend == LlmMemoryBackend::Metal) {
      size_t retained_metal_task_records = 0;
      if (!NumericUtils::checked_add(planned_measurements,
                                     total_calibration_attempts,
                                     retained_metal_task_records) ||
          !add_metal_task_string_capacity(
              retained_metal_task_records,
              estimate.fixed_metadata_bytes)) {
        return estimate;
      }
    }

    estimate.checksum_auxiliary_bytes =
        estimate.retained_checksum_bytes;
    size_t orchestration = 0;
    if (!checked_add_to(estimate.measurement_record_bytes, orchestration) ||
        !checked_add_to(estimate.loop_record_bytes, orchestration) ||
        !checked_add_to(estimate.calibration_record_bytes, orchestration) ||
        !checked_add_to(estimate.calibration_identity_bytes,
                        orchestration) ||
        !checked_add_to(estimate.aggregate_value_bytes, orchestration) ||
        !checked_add_to(estimate.statistics_workspace_bytes,
                        orchestration) ||
        !checked_add_to(estimate.warning_record_bytes, orchestration) ||
        !checked_add_to(estimate.fixed_metadata_bytes, orchestration)) {
      return estimate;
    }
    estimate.orchestration_auxiliary_bytes = orchestration;
    if (!NumericUtils::checked_add(
            estimate.checksum_auxiliary_bytes,
            estimate.orchestration_auxiliary_bytes,
            estimate.total_auxiliary_bytes)) {
      return estimate;
    }
    estimate.valid = true;
    estimate.reason_code = LlmExecutorReason::VALID;
    return estimate;
  } catch (...) {
    estimate = LlmRunnerAuxiliaryEstimate{};
    estimate.reason_code = LlmRunnerReason::RUNNER_EXCEPTION;
    return estimate;
  }
}

LlmRunnerAuxiliaryEstimate calculate_llm_runner_auxiliary_estimate(const LlmMemoryConfig& config,
                                                                   const LlmMemoryWorkPlan& model_plan) noexcept {
  LlmRunnerAuxiliaryEstimate estimate;
  try {
    const LlmCpuExecutionPlan* cpu_plan =
        get_llm_cpu_execution_plan(model_plan);
    const LlmMetalExecutionPlan* const metal_plan =
        get_llm_metal_execution_plan(model_plan);
    const bool backend_plan_valid =
        (model_plan.backend == LlmMemoryBackend::Cpu && cpu_plan != nullptr &&
         cpu_plan->effective_workers != 0) ||
        (model_plan.backend == LlmMemoryBackend::Metal &&
         metal_plan != nullptr &&
         is_activated_metal_profile(model_plan.phase,
                                    model_plan.kv_layout));
    if (!model_plan.valid || config.loop_count == 0 ||
        config.backend != model_plan.backend || !backend_plan_valid) {
      estimate.reason_code = LlmRunnerReason::INVALID_MODEL_WORK_PLAN;
      return estimate;
    }

    size_t planned_measurements = 0;
    const size_t attempts_per_scenario = calibration_capacity(config);
    size_t total_calibration_attempts = 0;
    size_t aggregate_value_count = 0;
    size_t retained_worker_checksums = 0;
    if (!NumericUtils::checked_multiply(config.loop_count, kLlmScenarioCount, planned_measurements) ||
        !NumericUtils::checked_multiply(attempts_per_scenario, kLlmScenarioCount, total_calibration_attempts) ||
        !NumericUtils::checked_multiply(planned_measurements, static_cast<size_t>(3), aggregate_value_count)) {
      return estimate;
    }
    const size_t retained_records_per_measurement =
        cpu_plan == nullptr ? 0 : cpu_plan->effective_workers;
    if (!NumericUtils::checked_multiply(planned_measurements,
                                        retained_records_per_measurement,
                                        retained_worker_checksums) ||
        !NumericUtils::checked_multiply(retained_worker_checksums, static_cast<size_t>(2), retained_worker_checksums) ||
        !NumericUtils::checked_multiply(planned_measurements, sizeof(LlmMeasurementState),
                                        estimate.measurement_record_bytes) ||
        !NumericUtils::checked_multiply(config.loop_count, sizeof(LlmLoopRecord), estimate.loop_record_bytes) ||
        !NumericUtils::checked_multiply(total_calibration_attempts, sizeof(LlmCalibrationAttempt),
                                        estimate.calibration_record_bytes) ||
        !NumericUtils::checked_multiply(aggregate_value_count, sizeof(double), estimate.aggregate_value_bytes) ||
        !NumericUtils::checked_multiply(config.loop_count, static_cast<size_t>(2 * sizeof(double)),
                                        estimate.statistics_workspace_bytes) ||
        !NumericUtils::checked_multiply(kLlmRunnerMaximumWarnings, sizeof(std::string_view),
                                        estimate.warning_record_bytes) ||
        !NumericUtils::checked_multiply(retained_worker_checksums, sizeof(LlmWorkerChecksum),
                                        estimate.retained_checksum_bytes)) {
      return estimate;
    }

    std::array<size_t, kLlmScenarioCount> identity_capacities{};
    if (!calculate_calibration_identity_capacities(config, model_plan, identity_capacities)) {
      estimate.reason_code = LlmRunnerReason::INVALID_MODEL_WORK_PLAN;
      return estimate;
    }
    std::array<size_t, kLlmScenarioCount> maximum_work_units{};
    size_t maximum_active_identity = 0;
    for (size_t index = 0; index < kLlmScenarioCount; ++index) {
      const LlmScenarioLimits limits = calculate_llm_scenario_limits(
          model_plan.geometry, kLlmScenarios[index], model_plan.backend);
      maximum_work_units[index] = limits.effective_maximum_work_units;
      maximum_active_identity = std::max(maximum_active_identity, identity_capacities[index]);
      size_t identity_with_null = 0;
      size_t conservative_identity = 0;
      size_t scenario_identity_total = 0;
      if (!NumericUtils::checked_add(identity_capacities[index], static_cast<size_t>(1), identity_with_null) ||
          !NumericUtils::checked_multiply(identity_with_null, static_cast<size_t>(2), conservative_identity) ||
          !NumericUtils::checked_multiply(conservative_identity, attempts_per_scenario, scenario_identity_total) ||
          !checked_add_to(scenario_identity_total, estimate.calibration_identity_bytes)) {
        return estimate;
      }
    }

    const LlmFrozenScenarioPlans maximum_frozen =
        freeze_llm_scenario_work_plans(model_plan, maximum_work_units, config.user_specified_iterations);
    if (!maximum_frozen.valid) {
      estimate.reason_code = LlmRunnerReason::INVALID_MODEL_WORK_PLAN;
      return estimate;
    }
    const auto add_conservative_string = [&](const std::string& value, size_t& total) {
      size_t with_null = 0;
      size_t doubled = 0;
      return NumericUtils::checked_add(value.size(), static_cast<size_t>(1), with_null) &&
             NumericUtils::checked_multiply(with_null, static_cast<size_t>(2), doubled) &&
             checked_add_to(doubled, total);
    };
    if (!add_conservative_string(maximum_frozen.reason_code, estimate.fixed_metadata_bytes) ||
        !add_conservative_string(maximum_frozen.model_plan_identity, estimate.fixed_metadata_bytes) ||
        !add_conservative_string(maximum_frozen.plan_identity, estimate.fixed_metadata_bytes)) {
      return estimate;
    }
    for (const LlmScenarioWorkPlan& scenario : maximum_frozen.scenarios) {
      if (!add_conservative_string(scenario.reason_code, estimate.fixed_metadata_bytes) ||
          !add_conservative_string(scenario.model_plan_identity, estimate.fixed_metadata_bytes) ||
          !add_conservative_string(scenario.plan_identity, estimate.fixed_metadata_bytes)) {
        return estimate;
      }
    }
    size_t active_plan_string_bytes = 0;
    size_t active_plan_identity_bytes = 0;
    size_t active_model_identity_bytes = 0;
    size_t measurement_reason_bytes = 0;
    if (!NumericUtils::checked_add(maximum_active_identity, static_cast<size_t>(1), active_plan_identity_bytes) ||
        !NumericUtils::checked_add(model_plan.plan_identity.size(), static_cast<size_t>(1),
                                   active_model_identity_bytes) ||
        !NumericUtils::checked_add(active_plan_identity_bytes, active_model_identity_bytes, active_plan_string_bytes) ||
        !NumericUtils::checked_multiply(active_plan_string_bytes, static_cast<size_t>(4), active_plan_string_bytes) ||
        !checked_add_to(active_plan_string_bytes, estimate.fixed_metadata_bytes) ||
        !NumericUtils::checked_multiply(planned_measurements, 2 * (kLlmRunnerReasonCapacity + 1),
                                        measurement_reason_bytes) ||
        !checked_add_to(measurement_reason_bytes, estimate.fixed_metadata_bytes) ||
        !checked_add_to(2 * (kLlmRunnerReasonCapacity + 1), estimate.fixed_metadata_bytes) ||
        !checked_add_to(2 * (kLlmRunnerDiagnosticCapacity + 1), estimate.fixed_metadata_bytes)) {
      return estimate;
    }
    if (model_plan.backend == LlmMemoryBackend::Metal) {
      size_t retained_metal_task_records = 0;
      if (!NumericUtils::checked_add(planned_measurements,
                                     total_calibration_attempts,
                                     retained_metal_task_records) ||
          !add_metal_task_string_capacity(
              retained_metal_task_records,
              estimate.fixed_metadata_bytes)) {
        return estimate;
      }
    }

    estimate.checksum_auxiliary_bytes = estimate.retained_checksum_bytes;
    size_t orchestration = 0;
    if (!checked_add_to(estimate.measurement_record_bytes, orchestration) ||
        !checked_add_to(estimate.loop_record_bytes, orchestration) ||
        !checked_add_to(estimate.calibration_record_bytes, orchestration) ||
        !checked_add_to(estimate.calibration_identity_bytes, orchestration) ||
        !checked_add_to(estimate.aggregate_value_bytes, orchestration) ||
        !checked_add_to(estimate.statistics_workspace_bytes, orchestration) ||
        !checked_add_to(estimate.warning_record_bytes, orchestration) ||
        !checked_add_to(estimate.fixed_metadata_bytes, orchestration)) {
      return estimate;
    }
    estimate.orchestration_auxiliary_bytes = orchestration;
    if (!NumericUtils::checked_add(estimate.checksum_auxiliary_bytes, estimate.orchestration_auxiliary_bytes,
                                   estimate.total_auxiliary_bytes)) {
      return estimate;
    }
    estimate.valid = true;
    estimate.reason_code = LlmExecutorReason::VALID;
    return estimate;
  } catch (...) {
    estimate = LlmRunnerAuxiliaryEstimate{};
    estimate.reason_code = LlmRunnerReason::RUNNER_EXCEPTION;
    return estimate;
  }
}

int run_llm_memory_suite(const LlmMemoryConfig& config,
                         const LlmMemoryWorkPlan& model_plan,
                         LlmBackend& backend, LlmMemoryResult& result,
                         const LlmRunnerHooks& hooks) {
  bool release_attempted = false;
  try {
    result = LlmMemoryResult{};
    if (backend.kind() != config.backend ||
        backend.kind() != model_plan.backend) {
      result.reason_code = LlmRunnerReason::BACKEND_UNAVAILABLE;
      return EXIT_FAILURE;
    }
    const LlmMemoryConfigValidation config_validation = validate_llm_memory_config(config);
    if (!config_validation.valid) {
      result.reason_code = LlmRunnerReason::INVALID_CONFIG;
      return EXIT_FAILURE;
    }
    if (!model_plan.valid) {
      result.reason_code = LlmRunnerReason::INVALID_MODEL_WORK_PLAN;
      return EXIT_FAILURE;
    }
    if (!inputs_match(config, model_plan)) {
      result.reason_code = LlmRunnerReason::CONFIG_WORK_PLAN_MISMATCH;
      return EXIT_FAILURE;
    }
    if (config.user_specified_iterations) {
      for (LlmScenario scenario : kLlmScenarios) {
        const LlmScenarioWorkPlan explicit_plan =
            build_llm_scenario_work_plan(model_plan, scenario, config.iterations, true);
        if (!explicit_plan.valid) {
          result.reason_code = explicit_plan.reason_code;
          return EXIT_FAILURE;
        }
      }
    }
    const LlmRunnerAuxiliaryEstimate auxiliary = calculate_llm_runner_auxiliary_estimate(config, model_plan);
    if (!auxiliary.valid) {
      result.reason_code = auxiliary.reason_code;
      return EXIT_FAILURE;
    }

    LlmBackendLifecycleResult initialization =
        backend.evidence().initialization;
    if (initialization.status == LlmBackendStatus::Unsupported) {
      return finish_unsupported_run(config, model_plan, auxiliary, result,
                                    hooks, initialization.reason_code, backend,
                                    release_attempted);
    }
    if (initialization.status == LlmBackendStatus::Failed) {
      if (initialization.reason_code ==
          LlmBackendReason::TIMER_UNAVAILABLE) {
        return fail_uninitialized_backend_transition(
            result, LlmBackendReason::TIMER_UNAVAILABLE, backend,
            release_attempted);
      }
      return finish_failed_backend_transition(
          config, model_plan, auxiliary, result, hooks,
          initialization.reason_code.empty()
              ? std::string_view(LlmBackendReason::BACKEND_INITIALIZATION_FAILED)
              : std::string_view(initialization.reason_code),
          LlmBackendReason::BACKEND_INITIALIZATION_FAILED, backend,
          release_attempted);
    }

    if (model_plan.backend == LlmMemoryBackend::Metal) {
      const LlmMetalExecutionPlan* const metal_plan =
          get_llm_metal_execution_plan(model_plan);
      if (metal_plan == nullptr || !metal_plan->valid ||
          !metal_plan->resources.valid || metal_plan->identity.empty()) {
        if (initialization.status == LlmBackendStatus::Ready) {
          const std::string_view runtime_plan_reason =
              metal_plan != nullptr &&
                      !metal_plan->reason_code.empty() &&
                      metal_plan->reason_code != LlmMetalPlanReason::VALID
                  ? std::string_view(metal_plan->reason_code)
                  : metal_plan != nullptr &&
                            !metal_plan->resources.reason_code.empty() &&
                            metal_plan->resources.reason_code !=
                                LlmMetalPlanReason::VALID
                        ? std::string_view(
                              metal_plan->resources.reason_code)
                        : std::string_view(
                              LlmRunnerReason::INVALID_MODEL_WORK_PLAN);
          return finish_failed_backend_transition(
              config, model_plan, auxiliary, result, hooks,
              runtime_plan_reason,
              LlmRunnerReason::INVALID_MODEL_WORK_PLAN, backend,
              release_attempted);
        }
        result.reason_code = LlmRunnerReason::INVALID_MODEL_WORK_PLAN;
        return EXIT_FAILURE;
      }
    }
    const std::string_view budget_reason =
        runner_budget_admission_reason(model_plan, auxiliary, backend);
    if (budget_reason != LlmExecutorReason::VALID) {
      if (initialization.status == LlmBackendStatus::Ready) {
        return finish_failed_backend_transition(
            config, model_plan, auxiliary, result, hooks, budget_reason,
            LlmRunnerReason::AUXILIARY_BUDGET_INSUFFICIENT,
            backend, release_attempted);
      }
      result.reason_code = budget_reason;
      return EXIT_FAILURE;
    }

    if (initialization.status == LlmBackendStatus::NotStarted) {
      initialization = backend.initialize(config);
    }
    if (initialization.status == LlmBackendStatus::Unsupported) {
      return finish_unsupported_run(config, model_plan, auxiliary, result,
                                    hooks, initialization.reason_code, backend,
                                    release_attempted);
    }
    if (initialization.status != LlmBackendStatus::Ready) {
      if (initialization.reason_code ==
          LlmBackendReason::TIMER_UNAVAILABLE) {
        return fail_uninitialized_backend_transition(
            result, LlmBackendReason::TIMER_UNAVAILABLE, backend,
            release_attempted);
      }
      return finish_failed_backend_transition(
          config, model_plan, auxiliary, result, hooks,
          initialization.reason_code.empty()
              ? std::string_view(LlmBackendReason::BACKEND_INITIALIZATION_FAILED)
              : std::string_view(initialization.reason_code),
          LlmBackendReason::BACKEND_INITIALIZATION_FAILED, backend,
          release_attempted);
    }

    // Initialization may populate backend-owned capability vectors and
    // diagnostics. Re-evaluate the full estimate before plan resolution or
    // resource allocation so a NotStarted backend cannot outgrow admission.
    const std::string_view initialized_budget_reason =
        runner_budget_admission_reason(model_plan, auxiliary, backend);
    if (initialized_budget_reason != LlmExecutorReason::VALID) {
      return finish_failed_backend_transition(
          config, model_plan, auxiliary, result, hooks,
          initialized_budget_reason,
          LlmRunnerReason::AUXILIARY_BUDGET_INSUFFICIENT, backend,
          release_attempted);
    }

    const LlmBackendLifecycleResult plan_resolution =
        backend.resolve_execution_plan(model_plan);
    if (plan_resolution.status == LlmBackendStatus::Unsupported) {
      return finish_unsupported_run(config, model_plan, auxiliary, result,
                                    hooks, plan_resolution.reason_code, backend,
                                    release_attempted);
    }
    if (plan_resolution.status != LlmBackendStatus::Ready) {
      return finish_failed_backend_transition(
          config, model_plan, auxiliary, result, hooks,
          plan_resolution.reason_code,
          LlmBackendReason::EXECUTION_PLAN_MISMATCH, backend,
          release_attempted);
    }

    const LlmBackendLifecycleResult preparation =
        backend.prepare_resources(model_plan);
    if (preparation.status == LlmBackendStatus::Unsupported) {
      return finish_unsupported_run(config, model_plan, auxiliary, result,
                                    hooks, preparation.reason_code, backend,
                                    release_attempted);
    }
    if (preparation.status != LlmBackendStatus::Ready) {
      return finish_failed_backend_transition(
          config, model_plan, auxiliary, result, hooks,
          preparation.reason_code,
          LlmBackendReason::RESOURCES_NOT_PREPARED, backend,
          release_attempted);
    }

    if (!initialize_result(config, model_plan, auxiliary, result)) {
      if (result.reason_code != LlmRunnerReason::AUXILIARY_BUDGET_INSUFFICIENT) {
        result.reason_code = LlmRunnerReason::PLANNED_COUNTER_OVERFLOW;
      }
      release_backend_once(backend, release_attempted, result);
      return EXIT_FAILURE;
    }

    std::array<size_t, kLlmScenarioCount> frozen_work_units{};
    if (config.user_specified_iterations) {
      frozen_work_units.fill(config.iterations);
    } else {
      for (size_t index = 0; index < kLlmScenarioCount; ++index) {
        const CalibrationOutcome outcome = calibrate_scenario(
            model_plan, kLlmScenarios[index], backend, result, hooks,
            frozen_work_units[index]);
        if (outcome == CalibrationOutcome::Interrupted) {
          trim_calibration_attempts(result);
          finalize_remaining_interrupted(result);
          return finish_terminal(result, hooks, backend, release_attempted);
        }
        if (outcome == CalibrationOutcome::Failed) {
          return fail_initialized_run(result, hooks, result.reason_code,
                                      backend, release_attempted);
        }
      }
    }

    result.frozen_scenario_plans = freeze_llm_scenario_work_plans(
        model_plan, frozen_work_units, config.user_specified_iterations);
    if (!result.frozen_scenario_plans.valid) {
      return fail_initialized_run(result, hooks,
                                  result.frozen_scenario_plans.reason_code,
                                  backend, release_attempted);
    }
    for (size_t index = 0;
         !config.user_specified_iterations && index < kLlmScenarioCount;
         ++index) {
      if (result.calibration_attempt_counts[index] == 0) {
        return fail_initialized_run(
            result, hooks, LlmRunnerReason::FROZEN_PLAN_MISMATCH, backend,
            release_attempted);
      }
      const LlmCalibrationAttempt& latest = result.calibration_attempts[index]
          [result.calibration_attempt_counts[index] - 1];
      if (result.frozen_scenario_plans.scenarios[index].work_units != frozen_work_units[index] ||
          result.frozen_scenario_plans.scenarios[index].plan_identity != latest.work_plan_identity) {
        return fail_initialized_run(
            result, hooks, LlmRunnerReason::FROZEN_PLAN_MISMATCH, backend,
            release_attempted);
      }
    }
    if (!assign_frozen_plans(result, model_plan)) {
      return fail_initialized_run(
          result, hooks, LlmRunnerReason::PLANNED_COUNTER_OVERFLOW,
          backend, release_attempted);
    }
    if (!actual_runner_backing_is_covered(result)) {
      return fail_initialized_run(
          result, hooks, LlmRunnerReason::AUXILIARY_BUDGET_INSUFFICIENT,
          backend, release_attempted);
    }
    const CalibrationOutcome frozen_warmup =
        warm_frozen_scenarios(model_plan, backend, result, hooks);
    if (frozen_warmup == CalibrationOutcome::Interrupted) {
      trim_calibration_attempts(result);
      finalize_remaining_interrupted(result);
      return finish_terminal(result, hooks, backend, release_attempted);
    }
    if (frozen_warmup == CalibrationOutcome::Failed) {
      return fail_initialized_run(result, hooks, result.reason_code, backend,
                                  release_attempted);
    }
    trim_calibration_attempts(result);
    refresh_calibration_references(result);
    if (!actual_runner_backing_is_covered(result)) {
      return fail_initialized_run(
          result, hooks, LlmRunnerReason::AUXILIARY_BUDGET_INSUFFICIENT,
          backend, release_attempted);
    }
    update_completion_state(result);
    return run_measurements(model_plan, backend, result, hooks,
                            release_attempted);
  } catch (const std::exception& error) {
    if (!result.initialized) {
      reset_uninitialized_failure(result, LlmRunnerReason::RUNNER_EXCEPTION);
      release_backend_once(backend, release_attempted, result);
      return EXIT_FAILURE;
    }
    try {
      assign_diagnostic(result, error.what());
    } catch (...) {
      result.diagnostic.clear();
    }
    if (!result.interruption_requested) {
      try {
        result.interruption_requested = stop_requested(hooks);
      } catch (...) {
      }
    }
    try {
      return fail_initialized_run(result, hooks,
                                  LlmRunnerReason::RUNNER_EXCEPTION,
                                  backend, release_attempted);
    } catch (...) {
      result.status = LlmRunStatus::Failed;
      result.reason_code = LlmRunnerReason::RUNNER_EXCEPTION;
      result.results_complete = false;
      result.conclusions_valid = false;
      release_backend_once(backend, release_attempted, result);
      return EXIT_FAILURE;
    }
  } catch (...) {
    if (!result.initialized) {
      reset_uninitialized_failure(result, LlmRunnerReason::RUNNER_UNKNOWN_EXCEPTION);
      release_backend_once(backend, release_attempted, result);
      return EXIT_FAILURE;
    }
    result.diagnostic.clear();
    if (!result.interruption_requested) {
      try {
        result.interruption_requested = stop_requested(hooks);
      } catch (...) {
      }
    }
    try {
      return fail_initialized_run(result, hooks,
                                  LlmRunnerReason::RUNNER_UNKNOWN_EXCEPTION,
                                  backend, release_attempted);
    } catch (...) {
      result.status = LlmRunStatus::Failed;
      result.reason_code = LlmRunnerReason::RUNNER_UNKNOWN_EXCEPTION;
      result.results_complete = false;
      result.conclusions_valid = false;
      release_backend_once(backend, release_attempted, result);
      return EXIT_FAILURE;
    }
  }
}

const char* llm_runner_task_kind_to_string(LlmRunnerTaskKind kind) noexcept {
  switch (kind) {
    case LlmRunnerTaskKind::Warmup:
      return "warmup";
    case LlmRunnerTaskKind::Calibration:
      return "calibration";
    case LlmRunnerTaskKind::Measurement:
      return "measurement";
  }
  return "unknown";
}

const char* llm_checkpoint_kind_to_string(LlmCheckpointKind kind) noexcept {
  switch (kind) {
    case LlmCheckpointKind::MeasurementTerminal:
      return "measurement-terminal";
    case LlmCheckpointKind::CommandTerminal:
      return "command-terminal";
  }
  return "unknown";
}

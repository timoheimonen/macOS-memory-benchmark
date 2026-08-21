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
    LlmRunnerReason::EXECUTOR_UNAVAILABLE,
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
constexpr std::string_view kLlmWorkPlanReasons[] = {
    LlmWorkPlanReason::VALID,
    LlmWorkPlanReason::ACTIVE_WEIGHT_BYTES_ZERO,
    LlmWorkPlanReason::LAYER_COUNT_ZERO,
    LlmWorkPlanReason::QUERY_HEAD_COUNT_ZERO,
    LlmWorkPlanReason::KV_HEAD_COUNT_ZERO,
    LlmWorkPlanReason::HEAD_DIMENSION_ZERO,
    LlmWorkPlanReason::INVALID_KV_ELEMENT_BYTES,
    LlmWorkPlanReason::CONTEXT_TOKENS_ZERO,
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
    LlmWorkPlanReason::KV_APPEND_BYTES_OVERFLOW,
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
    LlmWorkPlanReason::MEMORY_REQUIREMENT_OVERFLOW,
    LlmWorkPlanReason::MEMORY_BUDGET_OVERFLOW,
    LlmWorkPlanReason::MEMORY_BUDGET_EXCEEDED,
    LlmWorkPlanReason::WITHIN_MEMORY_BUDGET,
    LlmWorkPlanReason::PLANNER_ALLOCATION_FAILED,
    LlmWorkPlanReason::INVALID_SCENARIO,
    LlmWorkPlanReason::INVALID_MODEL_WORK_PLAN,
    LlmWorkPlanReason::PAYLOAD_CAP_BELOW_ONE_STEP,
    LlmWorkPlanReason::STEP_COUNT_ZERO,
    LlmWorkPlanReason::STEP_CAP_EXCEEDED,
    LlmWorkPlanReason::EXACT_PAYLOAD_OVERFLOW,
    LlmWorkPlanReason::EXACT_PAYLOAD_CAP_EXCEEDED,
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

std::string_view canonical_executor_reason(std::string_view reason_code) noexcept {
  for (std::string_view candidate : kLlmExecutorReasons) {
    if (reason_code == candidate) {
      return candidate;
    }
  }
  return LlmExecutorReason::INVALID_RESOURCES;
}

std::string_view canonical_calibration_purpose(std::string_view purpose) noexcept {
  constexpr std::array<std::string_view, 7> kPurposes = {
      "explicit-warmup",    "pilot-warmup",       "pilot", "duration-trial-warmup", "duration-trial",
      "correction-trial-1", "correction-trial-2",
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
  if (!validation.valid || !plan.valid || !plan.geometry.valid || plan.plan_identity.empty()) {
    return false;
  }
  const LlmGeometry& geometry = plan.geometry;
  return validation.active_weight_bytes == geometry.active_weight_bytes_per_step &&
         config.layer_count == geometry.layer_count && config.query_head_count == geometry.query_head_count &&
         config.kv_head_count == geometry.kv_head_count && config.head_dimension == geometry.head_dimension &&
         config.kv_element_bytes == geometry.kv_element_bytes &&
         config.visible_context_tokens == geometry.visible_context_tokens && config.batch_size == geometry.batch_size &&
         config.requested_workers == plan.requested_workers && config.available_workers == plan.available_workers &&
         config.seed == plan.base_seed;
}

std::string_view runner_budget_admission_reason(
    const LlmMemoryWorkPlan& plan, const LlmRunnerAuxiliaryEstimate& runner_auxiliary) noexcept {
  const LlmExecutorAuxiliaryEstimate executor_auxiliary = calculate_llm_executor_auxiliary_estimate(plan);
  if (!executor_auxiliary.valid) {
    return executor_auxiliary.reason_code;
  }
  if (!runner_auxiliary.valid) {
    return runner_auxiliary.reason_code;
  }
  size_t required_checksum = 0;
  size_t required_orchestration = 0;
  if (!NumericUtils::checked_add(executor_auxiliary.checksum_auxiliary_bytes, runner_auxiliary.checksum_auxiliary_bytes,
                                 required_checksum) ||
      !NumericUtils::checked_add(executor_auxiliary.orchestration_auxiliary_bytes,
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

LlmTaskExecutionEvidence compact_execution(const LlmExecutorResult& execution) {
  LlmTaskExecutionEvidence evidence;
  evidence.valid = execution.valid;
  evidence.reason_code = canonical_executor_reason(execution.reason_code);
  evidence.elapsed_seconds = execution.elapsed_seconds;
  evidence.requested_workers = execution.requested_workers;
  evidence.created_workers = execution.created_workers;
  evidence.completed_workers = execution.completed_workers;
  evidence.qos_successful_workers = execution.qos_successful_workers;
  evidence.qos_failed_workers = execution.qos_failed_workers;
  evidence.worker_startup_failed = execution.worker_startup_failed;
  evidence.kernel_succeeded = execution.kernel_succeeded;
  evidence.timer_started = execution.timer_started;
  evidence.timer_stopped = execution.timer_stopped;
  evidence.checksum_valid = execution.checksum_valid;
  evidence.expected_run_checksum = execution.expected_run_checksum;
  evidence.actual_run_checksum = execution.actual_run_checksum;
  return evidence;
}

bool execution_lifecycle_is_complete(const LlmExecutorResult& execution, size_t effective_workers) noexcept {
  return execution.requested_workers == effective_workers && execution.created_workers == effective_workers &&
         execution.completed_workers == effective_workers &&
         execution.qos_successful_workers <= effective_workers &&
         execution.qos_failed_workers == effective_workers - execution.qos_successful_workers &&
         !execution.worker_startup_failed && execution.kernel_succeeded && execution.timer_started &&
         execution.timer_stopped && execution.expected_checksums.size() == effective_workers &&
         execution.actual_checksums.size() == effective_workers;
}

bool execution_is_accepted(const LlmExecutorResult& execution, size_t effective_workers) noexcept {
  return execution.valid && canonical_executor_reason(execution.reason_code) == LlmExecutorReason::VALID &&
         execution_lifecycle_is_complete(execution, effective_workers) && execution.checksum_valid &&
         std::isfinite(execution.elapsed_seconds) && execution.elapsed_seconds > 0.0;
}

std::string_view execution_failure_reason(const LlmExecutorResult& execution, size_t effective_workers) noexcept {
  const std::string_view reason_code = canonical_executor_reason(execution.reason_code);
  if (!execution.valid || reason_code != LlmExecutorReason::VALID ||
      !execution_lifecycle_is_complete(execution, effective_workers)) {
    return reason_code == LlmExecutorReason::VALID ? std::string_view(LlmExecutorReason::INVALID_RESOURCES)
                                                   : reason_code;
  }
  if (!execution.checksum_valid) {
    return LlmExecutorReason::CHECKSUM_MISMATCH;
  }
  return LlmExecutorReason::INVALID_ELAPSED_TIME;
}

LlmMeasurementStatus execution_failure_status(std::string_view reason_code) noexcept {
  return reason_code == LlmExecutorReason::CHECKSUM_MISMATCH || reason_code == LlmExecutorReason::INVALID_ELAPSED_TIME
             ? LlmMeasurementStatus::Invalid
             : LlmMeasurementStatus::Failed;
}

void clear_measurement_values(LlmMeasurementState& measurement) {
  measurement.completed_steps = 0;
  measurement.completed_exact_payload_bytes = 0;
  measurement.elapsed_seconds.reset();
  measurement.synthetic_step_latency_seconds.reset();
  measurement.synthetic_memory_steps_per_second.reset();
  measurement.effective_payload_gb_s.reset();
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
      !measurement.checksum_valid || !measurement.synthetic_step_latency_seconds.has_value() ||
      !measurement.synthetic_memory_steps_per_second.has_value() || !measurement.effective_payload_gb_s.has_value()) {
    return;
  }

  LlmScenarioAggregate& aggregate = result.aggregates[index];
  aggregate.step_latency_seconds.values.push_back(*measurement.synthetic_step_latency_seconds);
  aggregate.synthetic_memory_steps_per_second.values.push_back(*measurement.synthetic_memory_steps_per_second);
  aggregate.effective_payload_gb_s.values.push_back(*measurement.effective_payload_gb_s);
  update_metric_aggregate(aggregate.step_latency_seconds, result.statistics_workspace);
  update_metric_aggregate(aggregate.synthetic_memory_steps_per_second, result.statistics_workspace);
  update_metric_aggregate(aggregate.effective_payload_gb_s, result.statistics_workspace);

  const size_t sample_count = aggregate.effective_payload_gb_s.values.size();
  aggregate.status = sample_count == result.counters.planned_loops ? "complete" : "partial";
  if (sample_count < 3) {
    aggregate.stability_quality = "insufficient-samples";
  } else if (aggregate.effective_payload_gb_s.statistics.coefficient_of_variation_defined &&
             aggregate.effective_payload_gb_s.statistics.coefficient_of_variation_pct >
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

  if (result.status != LlmRunStatus::Failed) {
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

int finish_terminal(LlmMemoryResult& result, const LlmRunnerHooks& hooks) {
  refresh_calibration_references(result);
  update_completion_state(result);
  if (result.checkpoint_failed || !invoke_checkpoint(result, LlmCheckpointKind::CommandTerminal, hooks)) {
    update_completion_state(result);
    return EXIT_FAILURE;
  }
  return result.status == LlmRunStatus::Failed ? EXIT_FAILURE : EXIT_SUCCESS;
}

size_t calibration_capacity(const LlmMemoryConfig& config) noexcept {
  return config.user_specified_iterations ? 1 : 4 + Constants::LLM_CALIBRATION_MAX_CORRECTIONS;
}

bool calculate_calibration_identity_capacities(const LlmMemoryConfig& config, const LlmMemoryWorkPlan& model_plan,
                                               std::array<size_t, kLlmScenarioCount>& capacities) {
  for (size_t index = 0; index < kLlmScenarioCount; ++index) {
    const LlmScenarioLimits limits = calculate_llm_scenario_limits(model_plan.geometry, kLlmScenarios[index]);
    if (!limits.valid) {
      return false;
    }
    const LlmScenarioWorkPlan maximum_plan = build_llm_scenario_work_plan(
        model_plan, kLlmScenarios[index], limits.effective_maximum_steps, config.user_specified_iterations);
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
    }
  }

  for (const LlmScenarioAggregate& aggregate : result.aggregates) {
    if (!add_capacity_bytes(aggregate.step_latency_seconds.values.capacity(), sizeof(double),
                            actual.aggregate_value_bytes) ||
        !add_capacity_bytes(aggregate.synthetic_memory_steps_per_second.values.capacity(), sizeof(double),
                            actual.aggregate_value_bytes) ||
        !add_capacity_bytes(aggregate.effective_payload_gb_s.values.capacity(), sizeof(double),
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
    if (!add_string_backing(measurement.execution.reason_code, actual.fixed_metadata_bytes) ||
        !add_capacity_bytes(measurement.execution.expected_checksums.capacity(), sizeof(LlmWorkerChecksum),
                            actual.retained_checksum_bytes) ||
        !add_capacity_bytes(measurement.execution.actual_checksums.capacity(), sizeof(LlmWorkerChecksum),
                            actual.retained_checksum_bytes)) {
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
    result.aggregates[index].step_latency_seconds.values.reserve(config.loop_count);
    result.aggregates[index].synthetic_memory_steps_per_second.values.reserve(config.loop_count);
    result.aggregates[index].effective_payload_gb_s.values.reserve(config.loop_count);
    result.calibration_attempts[index].resize(attempts_per_scenario);
    for (LlmCalibrationAttempt& attempt : result.calibration_attempts[index]) {
      attempt.scenario = kLlmScenarios[index];
      attempt.work_plan_identity.reserve(identity_capacities[index]);
    }
  }
  result.loops.reserve(config.loop_count);
  result.measurements.reserve(result.counters.planned_measurements);
  for (size_t loop_index = 0; loop_index < config.loop_count; ++loop_index) {
    LlmLoopRecord loop;
    loop.loop_index = loop_index;
    loop.planned_order = build_llm_scenario_order(loop_index);
    for (size_t position = 0; position < kLlmScenarioCount; ++position) {
      LlmMeasurementState measurement;
      measurement.scenario = loop.planned_order[position];
      measurement.loop_index = loop_index;
      measurement.order_position = position;
      measurement.requested_workers = model_plan.requested_workers;
      measurement.effective_workers = model_plan.effective_workers;
      measurement.working_set_bytes = model_plan.geometry.total_data_mapping_bytes;
      measurement.execution.reason_code.reserve(kLlmRunnerReasonCapacity);
      measurement.execution.expected_checksums.reserve(model_plan.effective_workers);
      measurement.execution.actual_checksums.reserve(model_plan.effective_workers);
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

struct ExcludedTaskOutcome {
  bool accepted = false;
  std::string_view reason_code = LlmRunnerReason::NOT_STARTED;
  double elapsed_seconds = 0.0;
};

ExcludedTaskOutcome execute_excluded_task(const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan& task_plan,
                                          LlmRunnerTaskKind kind, std::string_view purpose,
                                          const LlmTaskExecutor& executor, LlmMemoryResult& result) {
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
  attempt.purpose = canonical_calibration_purpose(purpose);
  attempt.explicit_iterations = task_plan.explicit_iterations;
  attempt.steps = task_plan.steps;
  attempt.weight_read_bytes = task_plan.weight_read_bytes;
  attempt.kv_read_bytes = task_plan.kv_read_bytes;
  attempt.kv_append_write_bytes = task_plan.kv_append_write_bytes;
  attempt.effective_payload_bytes = task_plan.effective_payload_bytes;
  attempt.work_plan_identity.assign(task_plan.plan_identity);
  try {
    LlmExecutorResult execution = executor(model_plan, task_plan, context);
    attempt.execution = compact_execution(execution);
    attempt.duration_quality =
        classify_llm_duration_quality(execution.elapsed_seconds, task_plan.steps,
                                      calculate_llm_scenario_limits(model_plan.geometry, task_plan.scenario));
    attempt.terminal = true;
    attempt.valid = execution_is_accepted(execution, model_plan.effective_workers);
    attempt.reason_code = attempt.valid ? std::string_view(LlmExecutorReason::VALID)
                                        : execution_failure_reason(execution, model_plan.effective_workers);
    attempt.execution.valid = attempt.valid;
    attempt.execution.reason_code = attempt.reason_code;

    ExcludedTaskOutcome outcome;
    outcome.accepted = attempt.valid;
    outcome.reason_code = attempt.reason_code;
    outcome.elapsed_seconds = execution.elapsed_seconds;
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
                                LlmRunnerTaskKind kind, std::string_view purpose, const LlmTaskExecutor& executor,
                                LlmMemoryResult& result, const LlmRunnerHooks& hooks,
                                double* elapsed_seconds = nullptr) {
  if (stop_requested(hooks)) {
    result.interruption_requested = true;
    return CalibrationOutcome::Interrupted;
  }
  const ExcludedTaskOutcome outcome = execute_excluded_task(model_plan, task_plan, kind, purpose, executor, result);
  if (elapsed_seconds != nullptr) {
    *elapsed_seconds = outcome.elapsed_seconds;
  }
  return stop_or_failure_after_excluded(outcome, result, hooks);
}

CalibrationOutcome calibrate_scenario(const LlmMemoryConfig& config, const LlmMemoryWorkPlan& model_plan,
                                      LlmScenario scenario, const LlmTaskExecutor& executor, LlmMemoryResult& result,
                                      const LlmRunnerHooks& hooks, size_t& frozen_steps) {
  const LlmScenarioLimits limits = calculate_llm_scenario_limits(model_plan.geometry, scenario);
  if (!limits.valid) {
    result.status = LlmRunStatus::Failed;
    result.reason_code = limits.reason_code;
    return CalibrationOutcome::Failed;
  }

  if (config.user_specified_iterations) {
    const size_t index = scenario_index(scenario);
    if (!result.frozen_scenario_plans.valid || index >= kLlmScenarioCount) {
      result.status = LlmRunStatus::Failed;
      result.reason_code = LlmRunnerReason::FROZEN_PLAN_MISMATCH;
      return CalibrationOutcome::Failed;
    }
    const LlmScenarioWorkPlan& explicit_plan = result.frozen_scenario_plans.scenarios[index];
    if (!explicit_plan.valid || explicit_plan.scenario != scenario || explicit_plan.steps != config.iterations ||
        !explicit_plan.explicit_iterations) {
      result.status = LlmRunStatus::Failed;
      result.reason_code = LlmRunnerReason::FROZEN_PLAN_MISMATCH;
      return CalibrationOutcome::Failed;
    }
    const CalibrationOutcome warmup =
        run_excluded(model_plan, explicit_plan, LlmRunnerTaskKind::Warmup, "explicit-warmup", executor, result, hooks);
    if (warmup != CalibrationOutcome::Success) {
      return warmup;
    }
    frozen_steps = config.iterations;
    return CalibrationOutcome::Success;
  }

  CalibrationOutcome outcome = CalibrationOutcome::Failed;
  {
    const LlmScenarioWorkPlan pilot_warmup = build_llm_scenario_work_plan(model_plan, scenario, 1, false);
    if (!pilot_warmup.valid) {
      result.status = LlmRunStatus::Failed;
      result.reason_code = pilot_warmup.reason_code;
      return CalibrationOutcome::Failed;
    }
    outcome =
        run_excluded(model_plan, pilot_warmup, LlmRunnerTaskKind::Warmup, "pilot-warmup", executor, result, hooks);
  }
  if (outcome != CalibrationOutcome::Success) {
    return outcome;
  }

  size_t steps = calculate_llm_pilot_steps(limits);
  if (steps == 0) {
    result.status = LlmRunStatus::Failed;
    result.reason_code = LlmRunnerReason::CALIBRATION_SCALING_FAILED;
    return CalibrationOutcome::Failed;
  }
  LlmScenarioWorkPlan latest_plan = build_llm_scenario_work_plan(model_plan, scenario, steps, false);
  if (!latest_plan.valid) {
    result.status = LlmRunStatus::Failed;
    result.reason_code = latest_plan.reason_code;
    return CalibrationOutcome::Failed;
  }
  double elapsed_seconds = 0.0;
  outcome = run_excluded(model_plan, latest_plan, LlmRunnerTaskKind::Calibration, "pilot", executor, result, hooks,
                         &elapsed_seconds);
  if (outcome != CalibrationOutcome::Success) {
    return outcome;
  }

  steps = calculate_llm_calibrated_steps(elapsed_seconds, latest_plan.steps, limits);
  if (steps == 0) {
    result.status = LlmRunStatus::Failed;
    result.reason_code = LlmRunnerReason::CALIBRATION_SCALING_FAILED;
    return CalibrationOutcome::Failed;
  }
  latest_plan = build_llm_scenario_work_plan(model_plan, scenario, steps, false);
  if (!latest_plan.valid) {
    result.status = LlmRunStatus::Failed;
    result.reason_code = latest_plan.reason_code;
    return CalibrationOutcome::Failed;
  }
  outcome = run_excluded(model_plan, latest_plan, LlmRunnerTaskKind::Warmup, "duration-trial-warmup", executor, result,
                         hooks);
  if (outcome != CalibrationOutcome::Success) {
    return outcome;
  }
  outcome = run_excluded(model_plan, latest_plan, LlmRunnerTaskKind::Calibration, "duration-trial", executor, result,
                         hooks, &elapsed_seconds);
  if (outcome != CalibrationOutcome::Success) {
    return outcome;
  }

  for (size_t correction = 1;
       !llm_duration_in_target_window(elapsed_seconds) && correction <= Constants::LLM_CALIBRATION_MAX_CORRECTIONS;
       ++correction) {
    const size_t corrected_steps = calculate_llm_calibrated_steps(elapsed_seconds, latest_plan.steps, limits);
    if (corrected_steps == 0) {
      result.status = LlmRunStatus::Failed;
      result.reason_code = LlmRunnerReason::CALIBRATION_SCALING_FAILED;
      return CalibrationOutcome::Failed;
    }
    if (corrected_steps == latest_plan.steps) {
      break;
    }
    latest_plan = build_llm_scenario_work_plan(model_plan, scenario, corrected_steps, false);
    if (!latest_plan.valid) {
      result.status = LlmRunStatus::Failed;
      result.reason_code = latest_plan.reason_code;
      return CalibrationOutcome::Failed;
    }
    const std::string_view purpose = correction == 1 ? "correction-trial-1" : "correction-trial-2";
    outcome = run_excluded(model_plan, latest_plan, LlmRunnerTaskKind::Calibration, purpose, executor, result, hooks,
                           &elapsed_seconds);
    if (outcome != CalibrationOutcome::Success) {
      return outcome;
    }
  }

  frozen_steps = latest_plan.steps;
  return CalibrationOutcome::Success;
}

bool assign_frozen_plans(LlmMemoryResult& result, const LlmMemoryWorkPlan& model_plan) {
  size_t total_steps = 0;
  size_t total_payload = 0;
  for (LlmMeasurementState& measurement : result.measurements) {
    const size_t index = scenario_index(measurement.scenario);
    if (index >= kLlmScenarioCount) {
      return false;
    }
    const LlmScenarioWorkPlan& plan = result.frozen_scenario_plans.scenarios[index];
    measurement.frozen_plan_index = index;
    measurement.explicit_iterations = plan.explicit_iterations;
    measurement.planned_steps = plan.steps;
    measurement.weight_read_bytes_per_step = plan.weight_read_bytes_per_step;
    measurement.kv_read_bytes_per_step = plan.kv_read_bytes_per_step;
    measurement.kv_append_write_bytes_per_step = plan.kv_append_write_bytes_per_step;
    measurement.effective_payload_bytes_per_step = plan.effective_payload_bytes_per_step;
    measurement.planned_weight_read_bytes = plan.weight_read_bytes;
    measurement.planned_kv_read_bytes = plan.kv_read_bytes;
    measurement.planned_kv_append_write_bytes = plan.kv_append_write_bytes;
    measurement.planned_exact_payload_bytes = plan.effective_payload_bytes;
    measurement.calibration_attempt_count = result.calibration_attempt_counts[index];

    if (measurement.scenario == LlmScenario::Mixed && plan.effective_payload_bytes_per_step != 0) {
      const long double total = static_cast<long double>(plan.effective_payload_bytes_per_step);
      measurement.weight_payload_fraction =
          static_cast<double>(static_cast<long double>(plan.weight_read_bytes_per_step) / total);
      measurement.kv_read_payload_fraction =
          static_cast<double>(static_cast<long double>(plan.kv_read_bytes_per_step) / total);
      measurement.kv_write_payload_fraction =
          static_cast<double>(static_cast<long double>(plan.kv_append_write_bytes_per_step) / total);
    }

    if (!checked_add_to(plan.steps, total_steps) || !checked_add_to(plan.effective_payload_bytes, total_payload)) {
      return false;
    }
  }
  result.counters.planned_synthetic_steps = total_steps;
  result.counters.planned_exact_payload_bytes = total_payload;
  return model_plan.valid;
}

void populate_measurement(LlmMeasurementState& measurement, LlmExecutorResult execution,
                          const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan& task_plan) {
  LlmExecutorResult& retained = measurement.execution;
  retained.valid = execution.valid;
  retained.reason_code.assign(canonical_executor_reason(execution.reason_code));
  retained.elapsed_seconds = execution.elapsed_seconds;
  retained.requested_workers = execution.requested_workers;
  retained.created_workers = execution.created_workers;
  retained.completed_workers = execution.completed_workers;
  retained.qos_successful_workers = execution.qos_successful_workers;
  retained.qos_failed_workers = execution.qos_failed_workers;
  retained.worker_startup_failed = execution.worker_startup_failed;
  retained.kernel_succeeded = execution.kernel_succeeded;
  retained.timer_started = execution.timer_started;
  retained.timer_stopped = execution.timer_stopped;
  retained.checksum_valid = execution.checksum_valid;
  retained.expected_run_checksum = execution.expected_run_checksum;
  retained.actual_run_checksum = execution.actual_run_checksum;
  if (execution.expected_checksums.size() == model_plan.effective_workers &&
      execution.actual_checksums.size() == model_plan.effective_workers) {
    retained.expected_checksums.assign(execution.expected_checksums.begin(), execution.expected_checksums.end());
    retained.actual_checksums.assign(execution.actual_checksums.begin(), execution.actual_checksums.end());
  } else {
    retained.valid = false;
    retained.reason_code.assign(LlmExecutorReason::INVALID_RESOURCES);
    retained.checksum_valid = false;
    retained.expected_checksums.clear();
    retained.actual_checksums.clear();
  }
  measurement.qos_successful_workers = measurement.execution.qos_successful_workers;
  measurement.qos_failed_workers = measurement.execution.qos_failed_workers;
  measurement.duration_quality =
      classify_llm_duration_quality(measurement.execution.elapsed_seconds, task_plan.steps,
                                    calculate_llm_scenario_limits(model_plan.geometry, task_plan.scenario));
  if (!execution_is_accepted(measurement.execution, model_plan.effective_workers)) {
    measurement.reason_code = execution_failure_reason(measurement.execution, model_plan.effective_workers);
    measurement.execution.valid = false;
    measurement.execution.reason_code.assign(measurement.reason_code);
    measurement.status = execution_failure_status(measurement.reason_code);
    clear_measurement_values(measurement);
    return;
  }

  const long double elapsed = static_cast<long double>(measurement.execution.elapsed_seconds);
  const long double steps = static_cast<long double>(task_plan.steps);
  const long double latency = elapsed / steps;
  const long double steps_per_second = steps / elapsed;
  const long double bandwidth = static_cast<long double>(task_plan.effective_payload_bytes) / elapsed / 1.0e9L;
  const double latency_value = static_cast<double>(latency);
  const double steps_per_second_value = static_cast<double>(steps_per_second);
  const double bandwidth_value = static_cast<double>(bandwidth);
  if (!std::isfinite(latency_value) || latency_value <= 0.0 || !std::isfinite(steps_per_second_value) ||
      steps_per_second_value <= 0.0 || !std::isfinite(bandwidth_value) || bandwidth_value <= 0.0) {
    measurement.status = LlmMeasurementStatus::Invalid;
    measurement.reason_code = LlmRunnerReason::INVALID_DERIVED_METRIC;
    clear_measurement_values(measurement);
    return;
  }

  measurement.status = LlmMeasurementStatus::Measured;
  measurement.reason_code = "measured";
  measurement.completed_steps = task_plan.steps;
  measurement.completed_exact_payload_bytes = task_plan.effective_payload_bytes;
  measurement.elapsed_seconds = measurement.execution.elapsed_seconds;
  measurement.synthetic_step_latency_seconds = latency_value;
  measurement.synthetic_memory_steps_per_second = steps_per_second_value;
  measurement.effective_payload_gb_s = bandwidth_value;
  measurement.checksum_valid = true;
}

void record_terminal_measurement(LlmMemoryResult& result, const LlmLoopRecord& loop,
                                 const LlmMeasurementState& measurement) {
  ++result.counters.terminal_measurements;
  if (measurement.status != LlmMeasurementStatus::Measured) {
    return;
  }
  ++result.counters.measured_measurements;
  result.counters.completed_synthetic_steps += measurement.completed_steps;
  result.counters.completed_exact_payload_bytes += measurement.completed_exact_payload_bytes;
  if (loop.realized_order_count == kLlmScenarioCount) {
    ++result.counters.completed_loops;
  }
  update_scenario_aggregate(result, measurement);
}

int fail_initialized_run(LlmMemoryResult& result, const LlmRunnerHooks& hooks, std::string_view reason_code);

int run_measurements(const LlmMemoryWorkPlan& model_plan, const LlmTaskExecutor& executor, LlmMemoryResult& result,
                     const LlmRunnerHooks& hooks) {
  for (LlmLoopRecord& loop : result.loops) {
    for (size_t position = 0; position < kLlmScenarioCount; ++position) {
      if (stop_requested(hooks)) {
        result.interruption_requested = true;
        finalize_remaining_interrupted(result);
        return finish_terminal(result, hooks);
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
        LlmExecutorResult execution = executor(model_plan, task_plan, context);
        populate_measurement(measurement, std::move(execution), model_plan, task_plan);
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
          return finish_terminal(result, hooks);
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
            return EXIT_FAILURE;
          }
        }
        return finish_terminal(result, hooks);
      } catch (...) {
        result.diagnostic.clear();
        bool pending_stop = pending_stop_after_failure(result, hooks);
        finalize_failure(result, LlmRunnerReason::RUNNER_UNKNOWN_EXCEPTION, pending_stop);
        if (!measurement_checkpoint_attempted) {
          const bool checkpoint_succeeded = invoke_checkpoint(result, LlmCheckpointKind::MeasurementTerminal, hooks);
          pending_stop = pending_stop || pending_stop_after_failure(result, hooks);
          finalize_failure(result, result.reason_code, pending_stop);
          if (!checkpoint_succeeded) {
            return EXIT_FAILURE;
          }
        }
        return finish_terminal(result, hooks);
      }
    }
  }
  return finish_terminal(result, hooks);
}

void trim_calibration_attempts(LlmMemoryResult& result) {
  for (size_t index = 0; index < kLlmScenarioCount; ++index) {
    if (result.calibration_attempts[index].size() > result.calibration_attempt_counts[index]) {
      result.calibration_attempts[index].resize(result.calibration_attempt_counts[index]);
    }
  }
}

int fail_initialized_run(LlmMemoryResult& result, const LlmRunnerHooks& hooks, std::string_view reason_code) {
  trim_calibration_attempts(result);
  finalize_failure(result, reason_code, result.interruption_requested);
  return finish_terminal(result, hooks);
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
  return LlmRunnerReason::RUNNER_UNKNOWN_EXCEPTION;
}

LlmRunnerAuxiliaryEstimate calculate_llm_runner_auxiliary_estimate(const LlmMemoryConfig& config,
                                                                   const LlmMemoryWorkPlan& model_plan) noexcept {
  LlmRunnerAuxiliaryEstimate estimate;
  try {
    if (!model_plan.valid || model_plan.effective_workers == 0 || config.loop_count == 0) {
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
    if (!NumericUtils::checked_multiply(planned_measurements, model_plan.effective_workers,
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
    std::array<size_t, kLlmScenarioCount> maximum_steps{};
    size_t maximum_active_identity = 0;
    for (size_t index = 0; index < kLlmScenarioCount; ++index) {
      const LlmScenarioLimits limits = calculate_llm_scenario_limits(model_plan.geometry, kLlmScenarios[index]);
      maximum_steps[index] = limits.effective_maximum_steps;
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
        freeze_llm_scenario_work_plans(model_plan, maximum_steps, config.user_specified_iterations);
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

int run_llm_memory_suite(const LlmMemoryConfig& config, const LlmMemoryWorkPlan& model_plan,
                         const LlmTaskExecutor& executor, LlmMemoryResult& result, const LlmRunnerHooks& hooks) {
  try {
    result = LlmMemoryResult{};
    if (!executor) {
      result.reason_code = LlmRunnerReason::EXECUTOR_UNAVAILABLE;
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
    const std::string_view budget_reason = runner_budget_admission_reason(model_plan, auxiliary);
    if (budget_reason != LlmExecutorReason::VALID) {
      result.reason_code = budget_reason;
      return EXIT_FAILURE;
    }

    if (!initialize_result(config, model_plan, auxiliary, result)) {
      if (result.reason_code != LlmRunnerReason::AUXILIARY_BUDGET_INSUFFICIENT) {
        result.reason_code = LlmRunnerReason::PLANNED_COUNTER_OVERFLOW;
      }
      return EXIT_FAILURE;
    }

    std::array<size_t, kLlmScenarioCount> frozen_steps{};
    if (config.user_specified_iterations) {
      frozen_steps.fill(config.iterations);
      result.frozen_scenario_plans = freeze_llm_scenario_work_plans(model_plan, frozen_steps, true);
      if (!result.frozen_scenario_plans.valid) {
        return fail_initialized_run(result, hooks, result.frozen_scenario_plans.reason_code);
      }
      if (!assign_frozen_plans(result, model_plan)) {
        return fail_initialized_run(result, hooks, LlmRunnerReason::PLANNED_COUNTER_OVERFLOW);
      }
      if (!actual_runner_backing_is_covered(result)) {
        return fail_initialized_run(result, hooks, LlmRunnerReason::AUXILIARY_BUDGET_INSUFFICIENT);
      }
    }
    for (size_t index = 0; index < kLlmScenarioCount; ++index) {
      const CalibrationOutcome outcome =
          calibrate_scenario(config, model_plan, kLlmScenarios[index], executor, result, hooks, frozen_steps[index]);
      if (outcome == CalibrationOutcome::Interrupted) {
        trim_calibration_attempts(result);
        finalize_remaining_interrupted(result);
        return finish_terminal(result, hooks);
      }
      if (outcome == CalibrationOutcome::Failed) {
        return fail_initialized_run(result, hooks, result.reason_code);
      }
    }
    trim_calibration_attempts(result);

    if (!config.user_specified_iterations) {
      result.frozen_scenario_plans = freeze_llm_scenario_work_plans(model_plan, frozen_steps, false);
      if (!result.frozen_scenario_plans.valid) {
        return fail_initialized_run(result, hooks, result.frozen_scenario_plans.reason_code);
      }
    }
    for (size_t index = 0; index < kLlmScenarioCount; ++index) {
      const LlmCalibrationAttempt& latest =
          result.calibration_attempts[index][result.calibration_attempt_counts[index] - 1];
      if (result.frozen_scenario_plans.scenarios[index].steps != frozen_steps[index] ||
          result.frozen_scenario_plans.scenarios[index].plan_identity != latest.work_plan_identity) {
        return fail_initialized_run(result, hooks, LlmRunnerReason::FROZEN_PLAN_MISMATCH);
      }
    }
    if (!assign_frozen_plans(result, model_plan)) {
      return fail_initialized_run(result, hooks, LlmRunnerReason::PLANNED_COUNTER_OVERFLOW);
    }
    if (!actual_runner_backing_is_covered(result)) {
      return fail_initialized_run(result, hooks, LlmRunnerReason::AUXILIARY_BUDGET_INSUFFICIENT);
    }
    update_completion_state(result);
    return run_measurements(model_plan, executor, result, hooks);
  } catch (const std::exception& error) {
    if (!result.initialized) {
      reset_uninitialized_failure(result, LlmRunnerReason::RUNNER_EXCEPTION);
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
      return fail_initialized_run(result, hooks, LlmRunnerReason::RUNNER_EXCEPTION);
    } catch (...) {
      result.status = LlmRunStatus::Failed;
      result.reason_code = LlmRunnerReason::RUNNER_EXCEPTION;
      result.results_complete = false;
      result.conclusions_valid = false;
      return EXIT_FAILURE;
    }
  } catch (...) {
    if (!result.initialized) {
      reset_uninitialized_failure(result, LlmRunnerReason::RUNNER_UNKNOWN_EXCEPTION);
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
      return fail_initialized_run(result, hooks, LlmRunnerReason::RUNNER_UNKNOWN_EXCEPTION);
    } catch (...) {
      result.status = LlmRunStatus::Failed;
      result.reason_code = LlmRunnerReason::RUNNER_UNKNOWN_EXCEPTION;
      result.results_complete = false;
      result.conclusions_valid = false;
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

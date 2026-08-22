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
 * @file llm_cpu_backend.cpp
 * @brief Generic-backend adapter around the existing CPU executor
 */

#include "llm_memory/llm_cpu_backend.h"

#include <cmath>
#include <optional>
#include <utility>

#include "core/timing/timer.h"

namespace {

LlmTaskIdentity build_task_identity(const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan& scenario_plan,
                                    const LlmRunnerTaskContext& context) noexcept {
  LlmTaskIdentity identity;
  identity.backend = model_plan.backend;
  identity.phase = model_plan.phase;
  identity.kv_layout = model_plan.kv_layout;
  identity.work_unit_kind = scenario_plan.work_unit_kind;
  identity.kv_write_kind = scenario_plan.kv_write_kind;
  identity.task_kind = context.kind;
  identity.scenario = context.scenario;
  identity.attempt_index = context.attempt_index;
  identity.loop_index = context.loop_index;
  identity.order_position = context.order_position;
  identity.purpose = context.purpose;
  identity.model_plan_identity = model_plan.plan_identity;
  identity.scenario_plan_identity = scenario_plan.plan_identity;
  return identity;
}

bool cpu_lifecycle_complete(const LlmExecutorResult& execution, size_t effective_workers) noexcept {
  return effective_workers != 0 && execution.requested_workers == effective_workers &&
         execution.created_workers == effective_workers && execution.completed_workers == effective_workers &&
         execution.qos_successful_workers <= effective_workers &&
         execution.qos_failed_workers == effective_workers - execution.qos_successful_workers &&
         !execution.worker_startup_failed && execution.kernel_succeeded && execution.timer_started &&
         execution.timer_stopped && execution.expected_checksums.size() == effective_workers &&
         execution.actual_checksums.size() == effective_workers;
}

class LlmCpuBackend final : public LlmBackend {
 public:
  LlmCpuBackend() { evidence_.backend = LlmMemoryBackend::Cpu; }

  LlmMemoryBackend kind() const noexcept override { return LlmMemoryBackend::Cpu; }

  LlmBackendAuxiliaryEstimate calculate_auxiliary_estimate(
      const LlmMemoryWorkPlan& model_plan) const noexcept override {
    LlmBackendAuxiliaryEstimate estimate;
    const LlmExecutorAuxiliaryEstimate cpu = calculate_llm_executor_auxiliary_estimate(model_plan);
    estimate.valid = cpu.valid;
    estimate.reason_code = cpu.reason_code;
    estimate.checksum_auxiliary_bytes = cpu.checksum_auxiliary_bytes;
    estimate.orchestration_auxiliary_bytes = cpu.orchestration_auxiliary_bytes;
    estimate.total_auxiliary_bytes = cpu.total_auxiliary_bytes;
    estimate.backend_evidence = cpu;
    return estimate;
  }

  LlmBackendLifecycleResult initialize(const LlmMemoryConfig& config) override {
    reset_owned_state();
    evidence_ = LlmBackendEvidence{};
    evidence_.backend = LlmMemoryBackend::Cpu;
    if (config.backend != LlmMemoryBackend::Cpu) {
      evidence_.initialization = {LlmBackendStatus::Failed, LlmBackendReason::BACKEND_MISMATCH};
      return evidence_.initialization;
    }
    timer_ = HighResTimer::create();
    if (!timer_.has_value()) {
      evidence_.initialization = {LlmBackendStatus::Failed, LlmBackendReason::TIMER_UNAVAILABLE};
      return evidence_.initialization;
    }
    initialized_ = true;
    evidence_.initialization = {LlmBackendStatus::Ready, LlmBackendReason::VALID};
    return evidence_.initialization;
  }

  LlmBackendLifecycleResult resolve_execution_plan(const LlmMemoryWorkPlan& model_plan) override {
    if (!initialized_) {
      evidence_.plan_resolution = {LlmBackendStatus::Failed, LlmBackendReason::NOT_INITIALIZED};
      return evidence_.plan_resolution;
    }
    const LlmCpuExecutionPlan* cpu_plan = get_llm_cpu_execution_plan(model_plan);
    if (!model_plan.valid || model_plan.backend != LlmMemoryBackend::Cpu || cpu_plan == nullptr ||
        cpu_plan->effective_workers == 0 || model_plan.plan_identity.empty()) {
      evidence_.plan_resolution = {LlmBackendStatus::Failed, LlmBackendReason::EXECUTION_PLAN_MISMATCH};
      return evidence_.plan_resolution;
    }
    resolved_plan_identity_ = model_plan.plan_identity;
    plan_resolved_ = true;
    evidence_.plan_resolution = {LlmBackendStatus::Ready, LlmBackendReason::VALID};
    return evidence_.plan_resolution;
  }

  LlmBackendLifecycleResult prepare_resources(const LlmMemoryWorkPlan& model_plan) override {
    if (!initialized_ || !plan_resolved_ || resolved_plan_identity_ != model_plan.plan_identity) {
      evidence_.preparation = {LlmBackendStatus::Failed, LlmBackendReason::EXECUTION_PLAN_MISMATCH};
      return evidence_.preparation;
    }
    LlmResourcePreparationResult preparation = prepare_llm_execution_resources(model_plan, resources_);
    evidence_.backend_evidence = LlmCpuBackendEvidence{std::move(preparation)};
    const auto& retained_preparation = std::get<LlmCpuBackendEvidence>(evidence_.backend_evidence).preparation;
    if (!retained_preparation.valid) {
      evidence_.preparation = {LlmBackendStatus::Failed, retained_preparation.reason_code};
      return evidence_.preparation;
    }
    resources_prepared_ = true;
    evidence_.preparation = {LlmBackendStatus::Ready, LlmBackendReason::VALID};
    return evidence_.preparation;
  }

  LlmTaskExecutionResult execute_task(const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan& scenario_plan,
                                      const LlmRunnerTaskContext& context) override {
    if (!resources_prepared_ || !timer_.has_value() || resolved_plan_identity_ != model_plan.plan_identity) {
      LlmTaskExecutionResult result;
      result.status = LlmTaskExecutionStatus::Failed;
      result.reason_code = LlmBackendReason::RESOURCES_NOT_PREPARED;
      result.identity = build_task_identity(model_plan, scenario_plan, context);
      result.completion.planned_work_units = scenario_plan.work_units;
      return result;
    }
    return adapt_llm_cpu_executor_result(model_plan, scenario_plan, context,
                                         execute_llm_scenario(model_plan, scenario_plan, resources_, *timer_));
  }

  const LlmBackendEvidence& evidence() const noexcept override { return evidence_; }

  LlmBackendLifecycleResult release_resources() noexcept override {
    reset_owned_state();
    evidence_.release = {LlmBackendStatus::Ready, LlmBackendReason::VALID};
    return evidence_.release;
  }

 private:
  /** Reset command-owned CPU state without recording a lifecycle transition. */
  void reset_owned_state() noexcept {
    resources_ = LlmExecutionResources{};
    timer_.reset();
    resources_prepared_ = false;
    plan_resolved_ = false;
    initialized_ = false;
    resolved_plan_identity_.clear();
  }

  bool initialized_ = false;
  bool plan_resolved_ = false;
  bool resources_prepared_ = false;
  std::string resolved_plan_identity_;
  std::optional<HighResTimer> timer_;
  LlmExecutionResources resources_;
  LlmBackendEvidence evidence_;
};

}  // namespace

LlmTaskExecutionResult adapt_llm_cpu_executor_result(const LlmMemoryWorkPlan& model_plan,
                                                     const LlmScenarioWorkPlan& scenario_plan,
                                                     const LlmRunnerTaskContext& context,
                                                     LlmExecutorResult executor_result) {
  LlmTaskExecutionResult result;
  result.identity = build_task_identity(model_plan, scenario_plan, context);
  result.completion.planned_work_units = scenario_plan.work_units;

  const LlmCpuExecutionPlan* cpu_plan = get_llm_cpu_execution_plan(model_plan);
  const size_t effective_workers = cpu_plan == nullptr ? 0 : cpu_plan->effective_workers;
  const bool lifecycle_complete = cpu_lifecycle_complete(executor_result, effective_workers);
  const bool checksum_evidence_complete = lifecycle_complete && executor_result.checksum_evaluated;
  result.timing.evaluated = executor_result.timer_started && executor_result.timer_stopped;
  result.timing.elapsed_seconds = executor_result.elapsed_seconds;
  result.timing.valid = result.timing.evaluated && std::isfinite(executor_result.elapsed_seconds) &&
                        executor_result.elapsed_seconds > 0.0;
  result.validation.evaluated = checksum_evidence_complete;
  result.validation.valid = checksum_evidence_complete && executor_result.checksum_valid;

  const std::string_view original_reason = executor_result.reason_code;
  const bool accepted = executor_result.valid && original_reason == LlmExecutorReason::VALID && lifecycle_complete &&
                        result.timing.valid && result.validation.evaluated && result.validation.valid;
  if (accepted) {
    result.status = LlmTaskExecutionStatus::Complete;
    result.reason_code = LlmExecutorReason::VALID;
    result.completion.completed_work_units = scenario_plan.work_units;
    result.completion.completed_effective_model_payload_bytes = scenario_plan.effective_model_payload_bytes;
    result.completion.completed_layout_metadata_lookup_count = scenario_plan.layout_metadata_lookup_count;
    result.completion.completed_layout_metadata_read_bytes = scenario_plan.layout_metadata_read_bytes;
    result.completion.completed_task_accounted_bytes = scenario_plan.task_accounted_bytes;
  } else if (original_reason == LlmExecutorReason::CHECKSUM_MISMATCH ||
             (lifecycle_complete && result.validation.evaluated && !result.validation.valid)) {
    result.status = LlmTaskExecutionStatus::Invalid;
    result.reason_code = LlmExecutorReason::CHECKSUM_MISMATCH;
  } else if (original_reason == LlmExecutorReason::INVALID_ELAPSED_TIME ||
             (lifecycle_complete && !result.timing.valid)) {
    result.status = LlmTaskExecutionStatus::Invalid;
    result.reason_code = LlmExecutorReason::INVALID_ELAPSED_TIME;
  } else {
    result.status = LlmTaskExecutionStatus::Failed;
    result.reason_code =
        original_reason == LlmExecutorReason::VALID ? LlmExecutorReason::INVALID_RESOURCES : original_reason;
  }
  result.backend_evidence = LlmCpuTaskEvidence{std::move(executor_result)};
  return result;
}

std::unique_ptr<LlmBackend> create_llm_cpu_backend() { return std::make_unique<LlmCpuBackend>(); }

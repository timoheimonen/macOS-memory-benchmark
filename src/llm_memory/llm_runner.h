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
 * @file llm_runner.h
 * @brief Backend-independent LLM memory-profile orchestration and results
 */

#ifndef LLM_RUNNER_H
#define LLM_RUNNER_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "llm_memory/llm_backend.h"
#include "utils/descriptive_statistics.h"

/** Stable machine-readable reasons owned by the LLM runner. */
namespace LlmRunnerReason {
inline constexpr const char* NOT_STARTED = "not-started";
inline constexpr const char* COMPLETE = "complete";
inline constexpr const char* PARTIAL_RESULTS = "partial-results";
inline constexpr const char* INTERRUPTION_REQUESTED = "interruption-requested";
inline constexpr const char* INTERRUPTION_BEFORE_TASK = "interruption-before-task";
inline constexpr const char* NOT_RUN_AFTER_RUNTIME_FAILURE = "not-run-after-runtime-failure";
inline constexpr const char* INVALID_CONFIG = "invalid-config";
inline constexpr const char* INVALID_MODEL_WORK_PLAN = "invalid-model-work-plan";
inline constexpr const char* CONFIG_WORK_PLAN_MISMATCH = "config-work-plan-mismatch";
inline constexpr const char* BACKEND_UNAVAILABLE = "backend-unavailable";
inline constexpr const char* AUXILIARY_BYTES_OVERFLOW = "runner-auxiliary-bytes-overflow";
inline constexpr const char* AUXILIARY_BUDGET_INSUFFICIENT = "runner-auxiliary-budget-insufficient";
inline constexpr const char* PLANNED_COUNTER_OVERFLOW = "planned-counter-overflow";
inline constexpr const char* CALIBRATION_SCALING_FAILED = "calibration-scaling-failed";
inline constexpr const char* FROZEN_PLAN_MISMATCH = "frozen-plan-mismatch";
inline constexpr const char* INVALID_DERIVED_METRIC = "invalid-derived-metric";
inline constexpr const char* CHECKPOINT_WRITE_FAILED = "checkpoint-write-failed";
inline constexpr const char* RUNNER_EXCEPTION = "runner-exception";
inline constexpr const char* RUNNER_UNKNOWN_EXCEPTION = "runner-unknown-exception";
}  // namespace LlmRunnerReason

/** Logical persistence point bound to the command-scoped output transport. */
enum class LlmCheckpointKind : uint8_t {
  MeasurementTerminal = 0,
  CommandTerminal,
};

/** Stable insertion handle; public snapshot indices are a separate sorted map. */
using LlmPlanHandle = size_t;

/** Once-per-plan expected witness; no runtime-valid assertion is implied. */
struct LlmCanonicalExpectedChecksum {
  bool available = false;
  std::string reason_code = "not-evaluated";
  std::vector<LlmWorkerChecksum> cpu_workers;
  LlmRunChecksum cpu_run{0, 0};
  LlmMetalDualMod32Checksum metal_run;
};

/** Unique scenario/T/explicit content with its sole retained expected witness. */
struct LlmCanonicalScenarioPlan {
  LlmScenarioWorkPlan plan;
  LlmCanonicalExpectedChecksum expected;
};

struct LlmCpuRetainedEvidence {
  LlmCpuRuntimeEvidence executor;
};

/** Runtime-only retained result: no copied task identity or expected checksum. */
struct LlmRetainedExecution {
  LlmTaskExecutionStatus status = LlmTaskExecutionStatus::NotStarted;
  std::string reason_code = LlmBackendReason::NOT_INITIALIZED;
  LlmAuthoritativeTiming timing;
  LlmCompletedWork completion;
  LlmTaskValidation validation;
  std::variant<std::monostate, LlmCpuRetainedEvidence, LlmMetalRuntimeEvidence> backend_evidence;
};

/** Read-only runtime variant access; returned pointers borrow the result. */
inline const LlmCpuRuntimeEvidence* get_llm_cpu_task_evidence(const LlmRetainedExecution& execution) noexcept {
  const auto* cpu = std::get_if<LlmCpuRetainedEvidence>(&execution.backend_evidence);
  return cpu ? &cpu->executor : nullptr;
}
inline const LlmMetalRuntimeEvidence* get_llm_metal_task_evidence(const LlmRetainedExecution& execution) noexcept {
  return std::get_if<LlmMetalRuntimeEvidence>(&execution.backend_evidence);
}

/** Compact excluded-task evidence without retained per-worker vectors. */
struct LlmTaskExecutionEvidence {
  std::optional<MachTimingSnapshot> cpu_raw;
  LlmColdChecks cpu_cold_checks;  ///< Compact CPU copy; Metal owns its own array.
  bool available = false;  ///< Complete generic task evidence was retained.
  bool valid = false;
  std::string_view reason_code = LlmRunnerReason::NOT_STARTED;
  LlmTaskExecutionStatus status = LlmTaskExecutionStatus::NotStarted;
  double elapsed_seconds = 0.0;
  bool timing_evaluated = false;
  bool timing_valid = false;
  LlmCompletedWork completion;
  bool validation_evaluated = false;
  bool validation_valid = false;
  bool cpu_evidence_available = false;
  size_t requested_workers = 0;
  size_t created_workers = 0;
  size_t completed_workers = 0;
  size_t qos_successful_workers = 0;
  size_t qos_failed_workers = 0;
  bool worker_startup_failed = false;
  bool kernel_succeeded = false;
  bool timer_started = false;
  bool timer_stopped = false;
  bool checksum_evaluated = false;  ///< CPU checksum detail, when applicable.
  bool checksum_valid = false;
  LlmRunChecksum actual_run_checksum{0, 0};
  bool metal_evidence_available = false;
  std::optional<LlmMetalRuntimeEvidence> metal;
};

/** One excluded warmup, pilot, duration trial, or correction record. */
struct LlmCalibrationAttempt {
  LlmPlanHandle plan_handle = kLlmNoTaskIndex;
  LlmScenario scenario = LlmScenario::WeightsOnly;
  std::string_view purpose = "not-run";
  LlmTaskExecutionEvidence execution;
  std::string_view duration_quality = "not-run";
  bool terminal = false;
  bool valid = false;
  std::string_view reason_code = LlmRunnerReason::NOT_STARTED;
};

/** Raw values, shared descriptive statistics, and one headline metric. */
struct LlmMetricAggregate {
  DescriptiveStatistics statistics;
  std::optional<double> headline;
};

/** Status-bearing record for one planned scenario measurement. */
struct LlmMeasurementState {
  LlmPlanHandle plan_handle = kLlmNoTaskIndex;
  LlmScenario scenario = LlmScenario::WeightsOnly;
  LlmMeasurementStatus status = LlmMeasurementStatus::NotRun;
  std::string_view reason_code = "not-run";
  size_t loop_index = 0;
  size_t order_position = 0;
  bool attempted = false;
  bool execution_evidence_available = false;
  std::string_view duration_quality = "not-run";
  LlmRetainedExecution execution;
};

/** On-demand rates all derived from the same accepted duration/work/payload. */
struct LlmDerivedMetrics {
  std::optional<double> latency_seconds;
  std::optional<double> work_units_per_second;
  std::optional<double> payload_gb_s;
};
LlmDerivedMetrics derive_llm_measurement_metrics(const LlmMeasurementState& measurement) noexcept;

/** Planned and realized scenario order for one count-loop. */
struct LlmLoopRecord {
  size_t loop_index = 0;
  std::array<LlmScenario, kLlmScenarioCount> planned_order{};
  std::array<LlmScenario, kLlmScenarioCount> realized_order{};
  size_t realized_order_count = 0;
  std::array<size_t, kLlmScenarioCount> measurement_indexes{};
};

/** Measured-only distributions and repeatability classification per scenario. */
struct LlmScenarioAggregate {
  LlmScenario scenario = LlmScenario::WeightsOnly;
  LlmMetricAggregate work_unit_latency_seconds;
  LlmMetricAggregate synthetic_memory_work_units_per_second;
  LlmMetricAggregate effective_model_payload_gb_s;
  std::string_view status = "unavailable";
  std::string_view observed_cv_classification = "insufficient-samples";
  std::vector<size_t> accepted_measurement_ids;
};

/** Reused sorted and deviation storage for allocation-free statistics updates. */
struct LlmStatisticsWorkspace {
  std::vector<double> extracted_values;
  std::vector<double> sorted_values;
  std::vector<double> absolute_deviations;
};

/** Exact lifecycle counters retained by partial and terminal run results. */
struct LlmRunCounters {
  size_t planned_loops = 0;
  size_t attempted_loops = 0;
  size_t completed_loops = 0;
  size_t planned_measurements = 0;
  size_t attempted_measurements = 0;
  size_t terminal_measurements = 0;
  size_t measured_measurements = 0;
  size_t planned_work_units = 0;
  size_t completed_work_units = 0;
  size_t planned_effective_model_payload_bytes = 0;
  size_t completed_effective_model_payload_bytes = 0;
  size_t planned_layout_metadata_lookup_count = 0;
  size_t completed_layout_metadata_lookup_count = 0;
  size_t planned_layout_metadata_read_bytes = 0;
  size_t completed_layout_metadata_read_bytes = 0;
  size_t planned_task_accounted_bytes = 0;
  size_t completed_task_accounted_bytes = 0;
};

/**
 * Conservative checked runner peak that must coexist with backend resources.
 *
 * `reason_code` always references static storage and remains valid when the
 * estimate is copied or moved.
 */
struct LlmRunnerAuxiliaryEstimate {
  bool valid = false;
  std::string_view reason_code = LlmRunnerReason::AUXILIARY_BYTES_OVERFLOW;
  size_t measurement_record_bytes = 0;
  size_t loop_record_bytes = 0;
  size_t calibration_record_bytes = 0;
  size_t calibration_identity_bytes = 0;
  size_t aggregate_value_bytes = 0;
  size_t statistics_workspace_bytes = 0;
  size_t warning_record_bytes = 0;
  size_t fixed_metadata_bytes = 0;
  size_t retained_checksum_bytes = 0;
  size_t checksum_auxiliary_bytes = 0;
  size_t orchestration_auxiliary_bytes = 0;
  size_t total_auxiliary_bytes = 0;
};

/**
 * Full runner result in complete, partial, interrupted, unsupported, or failed
 * state after initialization.
 *
 * Every `string_view` member in this result graph references static token
 * storage, so copying or moving the result does not invalidate those views.
 */
struct LlmMemoryResult {
  bool initialized = false;
  LlmRunStatus status = LlmRunStatus::NotStarted;
  std::string reason_code = LlmRunnerReason::NOT_STARTED;
  std::string diagnostic;
  bool interruption_requested = false;
  bool results_complete = false;
  bool run_accepted = false;
  bool scenario_order_balance_complete = false;
  bool checkpoint_failed = false;
  bool terminal_checkpoint_attempted = false;
  bool terminal_checkpoint_completed = false;
  size_t prior_file_writer_attempts = 0;
  size_t prior_successful_file_writes = 0;
  std::string_view snapshot_request = "none";
  size_t logical_checkpoint_attempts = 0;
  size_t successful_logical_checkpoints = 0;
  LlmRunCounters counters;
  LlmRunnerAuxiliaryEstimate runner_auxiliary;
  // Append-only canonical storage: entries become immutable after registration.
  std::vector<LlmCanonicalScenarioPlan> scenario_plans;
  std::array<LlmPlanHandle, kLlmScenarioCount> frozen_plan_handles{
      kLlmNoTaskIndex, kLlmNoTaskIndex, kLlmNoTaskIndex};
  std::vector<LlmPlanHandle> snapshot_plan_order;
  std::vector<size_t> snapshot_plan_refs;
  size_t snapshot_preparation_attempts = 0;
  size_t exact_statistics_passes = 0;
  size_t snapshot_interval_loops = 1;
  std::array<std::vector<LlmCalibrationAttempt>, kLlmScenarioCount> calibration_attempts;
  std::array<size_t, kLlmScenarioCount> calibration_attempt_counts{};
  std::vector<LlmLoopRecord> loops;
  std::vector<LlmMeasurementState> measurements;
  std::array<LlmScenarioAggregate, kLlmScenarioCount> aggregates;
  LlmStatisticsWorkspace statistics_workspace;
  std::vector<std::string_view> quality_warnings;
};

/** Insert/reuse exact scenario content. Contradictory keys throw before publication.
 * Handles survive vector growth; no pointer may be held across insertion.
 * Registered entries must not be mutated or erased. The result has one writer;
 * readers may inspect it only while that writer is idle. */
LlmPlanHandle register_llm_scenario_plan(LlmMemoryResult& result,
    const LlmMemoryWorkPlan& model, const LlmScenarioWorkPlan& plan, LlmBackend& backend);
const LlmCanonicalScenarioPlan* find_llm_scenario_plan(const LlmMemoryResult& result,
    LlmPlanHandle handle) noexcept;
/** Applicability from admitted phase/layout/scenario, before execution exists. */
LlmColdChecks required_llm_cold_checks(const LlmMemoryWorkPlan& model, LlmScenario scenario) noexcept;
/** Build one consistent sorted reference map and exact accepted-ID statistics.
 * Requires immutable registered content and exclusive access to the result.
 * Rejects unknown or scenario-contradictory references with std::invalid_argument;
 * allocation failures propagate to the owning checkpoint boundary. */
void prepare_llm_result_snapshot(LlmMemoryResult& result);

/** Deterministic stop and logical-checkpoint seams. */
struct LlmRunnerHooks {
  bool progress_snapshots = true;  ///< False for stdout/disabled transports; terminal statistics still finalize.
  std::function<bool()> stop_requested;
  std::function<std::pair<size_t, size_t>()> observe_file_writes;
  std::function<int(const LlmMemoryResult&, LlmCheckpointKind)> checkpoint;
};

/**
 * Calculate a checked conservative runner peak for pre-allocation admission.
 *
 * The calculation may allocate temporary scenario-plan identities, but it
 * contains construction failures and returns a stable invalid estimate.
 *
 * @return A valid estimate, or a stable reason-bearing invalid estimate when
 *         checked arithmetic or plan construction fails.
 */
LlmRunnerAuxiliaryEstimate calculate_llm_runner_auxiliary_estimate(const LlmMemoryConfig& config,
                                                                   const LlmMemoryWorkPlan& model_plan) noexcept;

/** Calculate exact runner backing before paged table materialization. */
LlmRunnerAuxiliaryEstimate calculate_llm_runner_auxiliary_estimate(
    const LlmMemoryConfig& config,
    const LlmAuxiliaryPreflightView& preflight) noexcept;

/**
 * Resolve excluded work, freeze all scenarios, and execute balanced loops.
 *
 * The runner checks stop only between whole backend tasks. A successfully
 * completed current task remains measured, while a backend-task, validation,
 * timing, or checkpoint failure remains authoritative over a simultaneous
 * stop. Every
 * attempted measurement's terminal transition and the distinct
 * command-terminal state is offered to the logical checkpoint hook. A failed
 * checkpoint is never retried.
 *
 * @param config Validated command configuration paired with @p model_plan.
 * @param model_plan Immutable pointer-free geometry and descriptor plan.
 * @param backend Required command-owned backend. The runner initializes,
 *        validates its tagged execution plan, prepares resources, executes
 *        whole tasks, and releases resources before its command-terminal
 *        checkpoint. Unit tests inject an Objective-C-free fake backend.
 * @param result Reset on entry. Preflight failures leave an uninitialized
 *        reason-bearing result; admitted runs retain initialized terminal or
 *        partial evidence.
 * @param hooks Optional stop and logical-checkpoint callbacks.
 * @return `EXIT_SUCCESS` for complete or graceful interrupted execution;
 *         `EXIT_FAILURE` for invalid input, execution, checksum, timer,
 *         arithmetic, exception, or checkpoint failure.
 * @note Synchronous and not thread-safe. Callbacks and inputs must outlive the
 *       call and must not mutate the model plan.
 */
int run_llm_memory_suite(const LlmMemoryConfig& config,
                         const LlmMemoryWorkPlan& model_plan,
                         LlmBackend& backend, LlmMemoryResult& result,
                         const LlmRunnerHooks& hooks = {});

/**
 * Canonicalize a known result reason into copy- and move-safe static storage.
 *
 * Runner-, work-plan-, and executor-owned reason domains are accepted. An
 * unknown value maps to `runner-unknown-exception`.
 *
 * @param reason_code Reason value; its backing storage need only outlive this
 *        call.
 * @return A stable process-lifetime token suitable for result `string_view`
 *         fields.
 */
std::string_view canonicalize_llm_result_reason_code(std::string_view reason_code) noexcept;

/** Stable token helpers for runner task and checkpoint identities. */
const char* llm_runner_task_kind_to_string(LlmRunnerTaskKind kind) noexcept;
const char* llm_checkpoint_kind_to_string(LlmCheckpointKind kind) noexcept;

#endif  // LLM_RUNNER_H

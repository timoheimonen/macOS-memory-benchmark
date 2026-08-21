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
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "llm_memory/llm_executor.h"
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
inline constexpr const char* EXECUTOR_UNAVAILABLE = "executor-unavailable";
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

/** Excluded or measured task category exposed to deterministic executors. */
enum class LlmRunnerTaskKind : uint8_t {
  Warmup = 0,
  Calibration,
  Measurement,
};

/** Logical persistence point; transport binding is supplied by Phase 5. */
enum class LlmCheckpointKind : uint8_t {
  MeasurementTerminal = 0,
  CommandTerminal,
};

inline constexpr size_t kLlmNoTaskIndex = std::numeric_limits<size_t>::max();

/** Cold-path identity for one executor callback invocation. */
struct LlmRunnerTaskContext {
  LlmRunnerTaskKind kind = LlmRunnerTaskKind::Warmup;
  std::string_view purpose;
  LlmScenario scenario = LlmScenario::WeightsOnly;
  size_t attempt_index = kLlmNoTaskIndex;
  size_t loop_index = kLlmNoTaskIndex;
  size_t order_position = kLlmNoTaskIndex;
};

/** Compact excluded-task evidence without retained per-worker vectors. */
struct LlmTaskExecutionEvidence {
  bool valid = false;
  std::string_view reason_code = LlmRunnerReason::NOT_STARTED;
  double elapsed_seconds = 0.0;
  size_t requested_workers = 0;
  size_t created_workers = 0;
  size_t completed_workers = 0;
  size_t qos_successful_workers = 0;
  size_t qos_failed_workers = 0;
  bool worker_startup_failed = false;
  bool kernel_succeeded = false;
  bool timer_started = false;
  bool timer_stopped = false;
  bool checksum_valid = false;
  LlmRunChecksum expected_run_checksum{0, 0};
  LlmRunChecksum actual_run_checksum{0, 0};
};

/** One excluded warmup, pilot, duration trial, or correction record. */
struct LlmCalibrationAttempt {
  LlmScenario scenario = LlmScenario::WeightsOnly;
  std::string_view purpose = "not-run";
  bool explicit_iterations = false;
  size_t steps = 0;
  size_t weight_read_bytes = 0;
  size_t kv_read_bytes = 0;
  size_t kv_append_write_bytes = 0;
  size_t effective_payload_bytes = 0;
  std::string work_plan_identity;
  LlmTaskExecutionEvidence execution;
  std::string_view duration_quality = "not-run";
  bool terminal = false;
  bool valid = false;
  std::string_view reason_code = LlmRunnerReason::NOT_STARTED;
};

/** Raw values, shared descriptive statistics, and one headline metric. */
struct LlmMetricAggregate {
  std::vector<double> values;
  DescriptiveStatistics statistics;
  std::optional<double> headline;
};

/** Status-bearing record for one planned scenario measurement. */
struct LlmMeasurementState {
  LlmScenario scenario = LlmScenario::WeightsOnly;
  LlmMeasurementStatus status = LlmMeasurementStatus::NotRun;
  std::string_view reason_code = "not-run";
  size_t loop_index = 0;
  size_t order_position = 0;
  bool attempted = false;
  size_t requested_workers = 0;
  size_t effective_workers = 0;
  size_t qos_successful_workers = 0;
  size_t qos_failed_workers = 0;
  size_t frozen_plan_index = kLlmNoTaskIndex;
  bool explicit_iterations = false;
  std::string_view duration_quality = "not-run";
  size_t calibration_attempt_count = 0;
  size_t planned_steps = 0;
  size_t completed_steps = 0;
  size_t weight_read_bytes_per_step = 0;
  size_t kv_read_bytes_per_step = 0;
  size_t kv_append_write_bytes_per_step = 0;
  size_t effective_payload_bytes_per_step = 0;
  size_t planned_weight_read_bytes = 0;
  size_t planned_kv_read_bytes = 0;
  size_t planned_kv_append_write_bytes = 0;
  size_t planned_exact_payload_bytes = 0;
  size_t completed_exact_payload_bytes = 0;
  std::optional<double> elapsed_seconds;
  std::optional<double> synthetic_step_latency_seconds;
  std::optional<double> synthetic_memory_steps_per_second;
  std::optional<double> effective_payload_gb_s;
  std::optional<double> weight_payload_fraction;
  std::optional<double> kv_read_payload_fraction;
  std::optional<double> kv_write_payload_fraction;
  size_t working_set_bytes = 0;
  bool checksum_valid = false;
  LlmExecutorResult execution;
};

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
  LlmMetricAggregate step_latency_seconds;
  LlmMetricAggregate synthetic_memory_steps_per_second;
  LlmMetricAggregate effective_payload_gb_s;
  std::string_view status = "unavailable";
  std::string_view stability_quality = "insufficient-samples";
};

/** Reused sorted and deviation storage for allocation-free statistics updates. */
struct LlmStatisticsWorkspace {
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
  size_t planned_synthetic_steps = 0;
  size_t completed_synthetic_steps = 0;
  size_t planned_exact_payload_bytes = 0;
  size_t completed_exact_payload_bytes = 0;
};

/**
 * Conservative checked runner peak that must coexist with executor resources.
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
 * Full initialized, partial, interrupted, failed, or complete runner result.
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
  bool conclusions_valid = false;
  bool scenario_order_balance_complete = false;
  bool checkpoint_failed = false;
  bool terminal_checkpoint_attempted = false;
  bool terminal_checkpoint_completed = false;
  size_t logical_checkpoint_attempts = 0;
  size_t successful_logical_checkpoints = 0;
  LlmRunCounters counters;
  LlmRunnerAuxiliaryEstimate runner_auxiliary;
  LlmFrozenScenarioPlans frozen_scenario_plans;
  std::array<std::vector<LlmCalibrationAttempt>, kLlmScenarioCount> calibration_attempts;
  std::array<size_t, kLlmScenarioCount> calibration_attempt_counts{};
  std::vector<LlmLoopRecord> loops;
  std::vector<LlmMeasurementState> measurements;
  std::array<LlmScenarioAggregate, kLlmScenarioCount> aggregates;
  LlmStatisticsWorkspace statistics_workspace;
  std::vector<std::string_view> quality_warnings;
};

/** Backend-independent task callback; exceptions are contained by the runner. */
using LlmTaskExecutor =
    std::function<LlmExecutorResult(const LlmMemoryWorkPlan&, const LlmScenarioWorkPlan&, const LlmRunnerTaskContext&)>;

/** Deterministic stop and logical-checkpoint seams. */
struct LlmRunnerHooks {
  std::function<bool()> stop_requested;
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

/**
 * Resolve excluded work, freeze all scenarios, and execute balanced loops.
 *
 * The runner checks stop only between whole executor tasks. A successfully
 * completed current task remains measured, while an executor, checksum, timer,
 * or checkpoint failure remains authoritative over a simultaneous stop. Every
 * attempted measurement's terminal transition and the distinct
 * command-terminal state is offered to the logical checkpoint hook. A failed
 * checkpoint is never retried.
 *
 * @param config Validated command configuration paired with @p model_plan.
 * @param model_plan Immutable pointer-free geometry and descriptor plan.
 * @param executor Required backend callback. Phase 5 binds production
 *        resources and `execute_llm_scenario`; unit tests inject a fake.
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
int run_llm_memory_suite(const LlmMemoryConfig& config, const LlmMemoryWorkPlan& model_plan,
                         const LlmTaskExecutor& executor, LlmMemoryResult& result, const LlmRunnerHooks& hooks = {});

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

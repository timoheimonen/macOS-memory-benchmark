// Copyright 2026 Timo Heimonen <timo.heimonen@proton.me>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

/**
 * @file gpu_runner.h
 * @brief Backend-independent GPU bandwidth orchestration and result model
 */

#ifndef GPU_RUNNER_H
#define GPU_RUNNER_H

#include <array>
#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "core/system/benchmark_qos.h"
#include "gpu_bandwidth/gpu_backend.h"
#include "gpu_bandwidth/gpu_bandwidth.h"
#include "utils/descriptive_statistics.h"

class JsonOutputSession;

enum class GpuMeasurementStatus {
  NotRun = 0,
  Measured,
  Interrupted,
  Invalid,
  Failed,
};

enum class GpuRunStatus {
  NotStarted = 0,
  Complete,
  Partial,
  Interrupted,
  Failed,
  Unsupported,
};

struct GpuMemoryBudget {
  bool valid = false;
  std::string reason_code = "memory-budget-not-calculated";
  size_t requested_buffer_bytes = 0;
  size_t auxiliary_bytes = 0;
  size_t required_total_bytes = 0;
  size_t available_memory_bytes = 0;
  size_t memory_budget_bytes = 0;
  bool used_fallback = false;
};

struct GpuCalibrationAttempt {
  GpuOperation operation = GpuOperation::Read;
  std::string purpose;
  size_t passes = 0;
  size_t exact_payload_bytes = 0;
  GpuWorkPlan work_plan;
  GpuBackendPhaseResult warmup;
  GpuBackendPhaseResult precondition;
  GpuTimedResult timed;
  GpuValidationResult validation;
  std::string duration_quality = "not-run";
  bool terminal = false;
  bool valid = false;
  std::string reason_code = "not-run";
};

struct GpuMeasurement {
  GpuMeasurementStatus status = GpuMeasurementStatus::NotRun;
  std::string reason_code = "not-run";
  std::optional<double> value_gb_s;
  GpuOperation operation = GpuOperation::Read;
  size_t loop_index = 0;
  size_t operation_order_position = 0;
  GpuWorkPlan work_plan;
  GpuBackendPhaseResult warmup;
  GpuBackendPhaseResult precondition;
  GpuTimedResult timed;
  GpuValidationResult validation;
  GpuEnvironmentSnapshot environment_before;
  GpuEnvironmentSnapshot environment_after;
  std::string duration_quality = "not-run";
  bool attempted = false;
  bool timed_command_completed = false;
  bool validation_terminal = false;
};

struct GpuLoopRecord {
  size_t loop_index = 0;
  std::vector<GpuOperation> planned_order;
  std::vector<GpuOperation> realized_order;
  std::array<size_t, 3> measurement_indexes{};
};

struct GpuOperationAggregate {
  GpuOperation operation = GpuOperation::Read;
  std::vector<double> values_gb_s;
  DescriptiveStatistics statistics;
  std::optional<double> headline_gb_s;
  std::string status = "unavailable";
  std::string stability_quality = "insufficient-samples";
};

struct GpuRunCounters {
  size_t planned_loops = 0;
  size_t attempted_loops = 0;
  size_t completed_loops = 0;
  size_t planned_measurements = 0;
  size_t attempted_measurements = 0;
  size_t terminal_measurements = 0;
  size_t completed_measurements = 0;
  size_t validated_measurements = 0;
};

struct GpuRunResult {
  std::string timestamp;
  GpuRunStatus status = GpuRunStatus::NotStarted;
  std::string reason_code = "not-started";
  bool interruption_requested = false;
  bool results_complete = false;
  bool conclusions_valid = false;
  bool operation_order_balance_complete = false;
  double elapsed_host_seconds = 0.0;
  MainThreadQosResult main_thread_qos;
  GpuRunCounters counters;
  GpuBackendInitialization backend_initialization;
  GpuMemoryBudget memory_budget;
  GpuAllocationResult allocation;
  GpuEnvironmentSnapshot environment_start;
  GpuEnvironmentSnapshot environment_end;
  std::array<GpuWorkPlan, 3> work_plans;
  std::array<std::vector<GpuCalibrationAttempt>, 3> calibration_attempts;
  std::vector<GpuLoopRecord> loops;
  std::vector<GpuMeasurement> measurements;
  std::array<GpuOperationAggregate, 3> aggregates;
  std::vector<std::string> quality_warnings;
};

/** Narrow deterministic seams for stop and atomic-checkpoint race tests. */
struct GpuRunnerTestHooks {
  std::function<bool()> stop_requested;
  std::function<int(const GpuRunResult&)> checkpoint;
};

GpuMemoryBudget calculate_gpu_memory_budget(size_t buffer_size_bytes,
                                            size_t available_memory_bytes,
                                            size_t auxiliary_bytes);

/**
 * @brief Run calibration and all balanced operation tasks using legacy file
 *        checkpointing.
 *
 * This overload is a file-only adapter: empty `config.output_file` disables
 * persistence, while a non-empty target must be a real file rather than the
 * exact stdout sentinel. Command code must classify output targets and call
 * the `JsonOutputSession` overload. The runner owns completion-wins
 * orchestration but never polls stop state inside a started
 * warmup/precondition/timed/validation task. Exceptions from hooks are
 * converted to an explicit failed run; production backends are required to
 * honor their noexcept boundary.
 *
 * @param config Validated GPU configuration. A non-empty `output_file` names
 *        a real file checkpoint target.
 * @param backend Backend instance owned by the caller for the call duration.
 * @param result Receives initialized, partial, terminal, or failed state.
 * @param hooks Optional deterministic seams for stop and checkpoint behavior.
 * @return `EXIT_SUCCESS` for complete or graceful interrupted execution;
 *         `EXIT_FAILURE` for failed or unsupported execution or persistence
 *         failure.
 * @pre A non-empty `config.output_file` is not the exact stdout sentinel `-`.
 * @note Called synchronously and not thread-safe. Inputs and hook state must
 *       not be accessed concurrently for the call duration.
 */
int run_gpu_bandwidth_suite(const GpuBandwidthConfig& config,
                            GpuBackend& backend,
                            GpuRunResult& result,
                            const GpuRunnerTestHooks& hooks = {});

/**
 * @brief Run the suite with command-owned JSON checkpoint transport.
 *
 * Hook checkpoints take precedence over the output session. Otherwise each
 * logical checkpoint is offered to @p output_session: file sessions persist
 * the existing atomic checkpoint sequence, while stdout sessions perform lazy
 * no-op persistence and retain the terminal result in @p result for the
 * command boundary to serialize exactly once. The session must outlive this
 * call and must not be shared concurrently. This overload does not emit the
 * final stdout document.
 *
 * @param config Validated GPU command configuration retained in schema v1.
 * @param backend Backend instance owned by the caller for the call duration.
 * @param result Receives initialized, partial, terminal, or failed state.
 * @param output_session Command-owned checkpoint transport.
 * @param hooks Optional deterministic test seams; hook callbacks override the
 *        session checkpoint sink and exceptions are contained as failed runs.
 * @return `EXIT_SUCCESS` for complete or graceful interrupted execution;
 *         `EXIT_FAILURE` for failed or unsupported execution or persistence
 *         failure.
 * @pre `output_session` represents `config.output_file`, classified with
 *      `JsonFilePathPolicy::PreserveRaw` by the command boundary.
 * @note Called synchronously and not thread-safe. The backend, result, session,
 *       and hook state must not be accessed concurrently for the call duration.
 */
int run_gpu_bandwidth_suite(const GpuBandwidthConfig& config,
                            GpuBackend& backend,
                            GpuRunResult& result,
                            JsonOutputSession& output_session,
                            const GpuRunnerTestHooks& hooks = {});

const char* gpu_measurement_status_to_string(GpuMeasurementStatus status);
const char* gpu_run_status_to_string(GpuRunStatus status);

#endif  // GPU_RUNNER_H

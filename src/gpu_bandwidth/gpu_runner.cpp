// Copyright 2026 Timo Heimonen <timo.heimonen@proton.me>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

/**
 * @file gpu_runner.cpp
 * @brief Backend-independent GPU bandwidth orchestration
 */

#include "gpu_bandwidth/gpu_runner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <limits>

#include "core/config/constants.h"
#include "core/signal/signal_handler.h"
#include "gpu_bandwidth/gpu_json.h"
#include "utils/json_utils.h"
#include "utils/numeric_utils.h"

namespace {

constexpr std::array<GpuOperation, kGpuOperationCount> kGpuOperations = {
    GpuOperation::Read, GpuOperation::Write, GpuOperation::Copy};
constexpr const char* kUnstartedRuntimeFailureReason =
    "not-run-after-runtime-failure";

size_t operation_index(GpuOperation operation) {
  switch (operation) {
    case GpuOperation::Read:
      return 0;
    case GpuOperation::Write:
      return 1;
    case GpuOperation::Copy:
      return 2;
  }
  return kGpuOperationCount;
}

const GpuPipelineMetadata& pipeline_for_operation(
    const GpuDeviceMetadata& device, GpuOperation operation) {
  switch (operation) {
    case GpuOperation::Read:
      return device.read_pipeline;
    case GpuOperation::Write:
      return device.write_pipeline;
    case GpuOperation::Copy:
      return device.copy_pipeline;
  }
  return device.read_pipeline;
}

bool backend_phase_completed(const GpuBackendPhaseResult& phase) {
  return phase.status == GpuBackendStatus::Success &&
         phase.command_status == GpuCommandStatus::Completed;
}

bool gpu_timer_is_valid(const GpuTimedResult& timed) {
  if (!std::isfinite(timed.gpu_start_seconds) ||
      !std::isfinite(timed.gpu_end_seconds) ||
      !std::isfinite(timed.gpu_elapsed_seconds) ||
      !std::isfinite(timed.host_wall_seconds) ||
      timed.gpu_start_seconds < 0.0 ||
      timed.gpu_end_seconds <= timed.gpu_start_seconds ||
      timed.gpu_elapsed_seconds <= 0.0 || timed.host_wall_seconds <= 0.0) {
    return false;
  }
  const double timestamp_delta =
      timed.gpu_end_seconds - timed.gpu_start_seconds;
  const double tolerance =
      std::max(1e-12, std::abs(timestamp_delta) * 1e-9);
  return std::abs(timestamp_delta - timed.gpu_elapsed_seconds) <= tolerance;
}

struct AttemptExecution {
  GpuBackendPhaseResult warmup;
  GpuBackendPhaseResult precondition;
  GpuTimedResult timed;
  GpuValidationResult validation;
  GpuMeasurementStatus status = GpuMeasurementStatus::Failed;
  std::string reason_code = "attempt-not-run";
  bool timed_command_completed = false;
  bool validation_terminal = false;
};

AttemptExecution execute_backend_attempt(GpuBackend& backend,
                                         const GpuWorkPlan& plan) {
  AttemptExecution execution;
  const GpuBackendAttemptRequest request =
      build_gpu_backend_attempt_request(plan);

  execution.warmup = backend.run_warmup(request);
  if (!backend_phase_completed(execution.warmup)) {
    execution.reason_code = execution.warmup.reason_code;
    return execution;
  }

  execution.precondition = backend.run_precondition(request);
  if (!backend_phase_completed(execution.precondition)) {
    execution.reason_code = execution.precondition.reason_code;
    return execution;
  }

  execution.timed = backend.run_timed(request);
  if (!backend_phase_completed(execution.timed)) {
    execution.reason_code = execution.timed.reason_code;
    return execution;
  }
  execution.timed_command_completed = true;

  if (!gpu_timer_is_valid(execution.timed)) {
    execution.validation.status = GpuBackendStatus::Success;
    execution.validation.reason_code = "not-run-timer-invalid";
    execution.validation.command_status = GpuCommandStatus::NotRun;
    execution.validation.validation_status =
        GpuValidationStatus::NotRunTimerInvalid;
    execution.validation_terminal = true;
    execution.status = GpuMeasurementStatus::Invalid;
    execution.reason_code = "invalid-gpu-timestamp";
    return execution;
  }

  execution.validation = backend.run_validation(request, execution.timed);
  execution.validation_terminal = true;
  if (execution.validation.status != GpuBackendStatus::Success ||
      execution.validation.validation_status == GpuValidationStatus::Error) {
    execution.status = GpuMeasurementStatus::Failed;
    execution.reason_code = execution.validation.reason_code;
    return execution;
  }
  if (execution.timed.actual_accumulator !=
      execution.timed.expected_accumulator) {
    execution.status = GpuMeasurementStatus::Invalid;
    execution.reason_code = "timed-accumulator-mismatch";
    return execution;
  }
  if (execution.validation.validation_status !=
      GpuValidationStatus::Passed) {
    execution.status = GpuMeasurementStatus::Invalid;
    execution.reason_code = execution.validation.reason_code.empty()
                                ? "validation-mismatch"
                                : execution.validation.reason_code;
    return execution;
  }

  execution.status = GpuMeasurementStatus::Measured;
  execution.reason_code = "measured";
  return execution;
}

bool initialize_result(const GpuBandwidthConfig& config,
                       GpuRunResult& result) {
  const MainThreadQosResult qos = result.main_thread_qos;
  result = GpuRunResult{};
  result.main_thread_qos = qos;
  result.timestamp = build_utc_timestamp();
  result.counters.planned_loops = config.loop_count;
  for (GpuOperation operation : kGpuOperations) {
    const size_t index = operation_index(operation);
    result.work_plans[index].operation = operation;
    result.aggregates[index].operation = operation;
  }

  if (!NumericUtils::checked_multiply(
          config.loop_count, kGpuOperationCount,
          result.counters.planned_measurements)) {
    return false;
  }

  result.loops.reserve(config.loop_count);
  result.measurements.reserve(result.counters.planned_measurements);
  for (size_t loop_index = 0; loop_index < config.loop_count; ++loop_index) {
    GpuLoopRecord loop;
    loop.loop_index = loop_index;
    const std::array<GpuOperation, kGpuOperationCount> order =
        build_gpu_operation_order(loop_index);
    loop.planned_order.assign(order.begin(), order.end());
    for (size_t position = 0; position < order.size(); ++position) {
      GpuMeasurement measurement;
      measurement.operation = order[position];
      measurement.work_plan.operation = measurement.operation;
      measurement.loop_index = loop_index;
      measurement.operation_order_position = position;
      loop.measurement_indexes[position] = result.measurements.size();
      result.measurements.push_back(std::move(measurement));
    }
    result.loops.push_back(std::move(loop));
  }
  return true;
}

void recompute_counters(GpuRunResult& result) {
  result.counters.attempted_loops = 0;
  result.counters.completed_loops = 0;
  result.counters.attempted_measurements = 0;
  result.counters.terminal_measurements = 0;
  result.counters.completed_measurements = 0;
  result.counters.validated_measurements = 0;

  for (const GpuMeasurement& measurement : result.measurements) {
    if (measurement.attempted) {
      ++result.counters.attempted_measurements;
    }
    if (measurement.status != GpuMeasurementStatus::NotRun) {
      ++result.counters.terminal_measurements;
    }
    if (measurement.timed_command_completed &&
        measurement.validation_terminal) {
      ++result.counters.completed_measurements;
    }
    if (measurement.status == GpuMeasurementStatus::Measured) {
      ++result.counters.validated_measurements;
    }
  }

  for (const GpuLoopRecord& loop : result.loops) {
    bool attempted = false;
    bool complete = true;
    for (size_t measurement_index : loop.measurement_indexes) {
      const GpuMeasurement& measurement =
          result.measurements[measurement_index];
      attempted = attempted || measurement.attempted;
      complete = complete &&
                 measurement.status == GpuMeasurementStatus::Measured;
    }
    result.counters.attempted_loops += attempted ? 1 : 0;
    result.counters.completed_loops += complete ? 1 : 0;
  }
}

void recompute_aggregates(GpuRunResult& result) {
  for (GpuOperationAggregate& aggregate : result.aggregates) {
    aggregate.values_gb_s.clear();
    aggregate.statistics = DescriptiveStatistics{};
    aggregate.headline_gb_s.reset();
    aggregate.status = "unavailable";
    aggregate.stability_quality = "insufficient-samples";
  }

  for (const GpuMeasurement& measurement : result.measurements) {
    if (measurement.status != GpuMeasurementStatus::Measured ||
        !measurement.value_gb_s.has_value()) {
      continue;
    }
    result.aggregates[operation_index(measurement.operation)]
        .values_gb_s.push_back(*measurement.value_gb_s);
  }

  result.quality_warnings.clear();
  for (GpuOperationAggregate& aggregate : result.aggregates) {
    if (aggregate.values_gb_s.empty()) {
      continue;
    }
    aggregate.statistics =
        calculate_descriptive_statistics(aggregate.values_gb_s);
    aggregate.headline_gb_s = aggregate.values_gb_s.size() == 1
                                  ? aggregate.values_gb_s.front()
                                  : aggregate.statistics.median;
    aggregate.status =
        aggregate.values_gb_s.size() == result.counters.planned_loops
            ? "complete"
            : "partial";
    if (aggregate.values_gb_s.size() < 3) {
      aggregate.stability_quality = "insufficient-samples";
    } else if (aggregate.statistics.coefficient_of_variation_defined &&
               aggregate.statistics.coefficient_of_variation_pct >
                   Constants::GPU_STREAMING_CV_WARNING_PCT) {
      aggregate.stability_quality = "noisy";
      result.quality_warnings.push_back(
          std::string(gpu_operation_to_string(aggregate.operation)) +
          "-high-cv");
    } else {
      aggregate.stability_quality = "stable";
    }
  }
}

void update_completion_state(GpuRunResult& result) {
  recompute_counters(result);
  recompute_aggregates(result);
  const bool all_validated =
      result.counters.validated_measurements ==
          result.counters.planned_measurements &&
      result.counters.planned_measurements != 0;
  if (result.status != GpuRunStatus::Failed &&
      result.status != GpuRunStatus::Unsupported &&
      result.status != GpuRunStatus::Interrupted) {
    result.status = all_validated ? GpuRunStatus::Complete
                                  : GpuRunStatus::Partial;
    result.reason_code = all_validated ? "complete" : "partial-results";
  }
  result.results_complete =
      result.status == GpuRunStatus::Complete && all_validated;
  result.conclusions_valid = result.results_complete;
  result.operation_order_balance_complete =
      all_validated &&
      result.counters.completed_loops % kGpuOperationCount == 0;
  if (result.results_complete &&
      !result.operation_order_balance_complete) {
    result.quality_warnings.push_back("operation-order-not-balanced");
  }

  const auto environment_non_nominal = [](const GpuEnvironmentSnapshot& env) {
    return (env.thermal_state != "nominal" &&
            env.thermal_state != "unavailable") ||
           (env.low_power_mode_available && env.low_power_mode_enabled);
  };
  bool environment_warning =
      environment_non_nominal(result.environment_start) ||
      environment_non_nominal(result.environment_end);
  for (const GpuMeasurement& measurement : result.measurements) {
    environment_warning =
        environment_warning ||
        environment_non_nominal(measurement.environment_before) ||
        environment_non_nominal(measurement.environment_after);
  }
  if (environment_warning) {
    result.quality_warnings.push_back("environment-not-nominal");
  }
  if (result.allocation.exceeds_recommended_working_set) {
    result.quality_warnings.push_back(
        "recommended-working-set-exceeded");
  }
}

void finalize_remaining_interrupted(GpuRunResult& result) {
  for (GpuMeasurement& measurement : result.measurements) {
    const bool synthetic_runtime_failure =
        !measurement.attempted &&
        measurement.status == GpuMeasurementStatus::Failed &&
        measurement.reason_code == kUnstartedRuntimeFailureReason;
    if (measurement.status == GpuMeasurementStatus::NotRun ||
        synthetic_runtime_failure) {
      measurement.status = GpuMeasurementStatus::Interrupted;
      measurement.reason_code = "interruption-before-task";
      measurement.value_gb_s.reset();
    }
  }
  recompute_counters(result);
  recompute_aggregates(result);
}

/** Mark unstarted measurement slots terminal after an in-run failure. */
void finalize_unstarted_failed(GpuRunResult& result) {
  for (GpuMeasurement& measurement : result.measurements) {
    if (!measurement.attempted &&
        measurement.status == GpuMeasurementStatus::NotRun) {
      measurement.status = GpuMeasurementStatus::Failed;
      measurement.reason_code = kUnstartedRuntimeFailureReason;
      measurement.value_gb_s.reset();
    }
  }
  recompute_counters(result);
  recompute_aggregates(result);
}

bool stop_requested(const GpuRunnerTestHooks& hooks) {
  return hooks.stop_requested ? hooks.stop_requested() : signal_received();
}

int write_checkpoint(const GpuBandwidthConfig& config,
                     const GpuRunResult& result,
                     const GpuRunnerTestHooks& hooks) {
  if (hooks.checkpoint) {
    return hooks.checkpoint(result);
  }
  if (config.output_file.empty()) {
    return EXIT_SUCCESS;
  }
  return save_gpu_bandwidth_json(config, result, false);
}

bool checkpoint_enabled(const GpuBandwidthConfig& config,
                        const GpuRunnerTestHooks& hooks) {
  return hooks.checkpoint || !config.output_file.empty();
}

/** Apply the terminal-task checkpoint and its one post-checkpoint stop read. */
int checkpoint_terminal_state(const GpuBandwidthConfig& config,
                              GpuRunResult& result,
                              const GpuRunnerTestHooks& hooks,
                              bool stop_seen_before_checkpoint,
                              bool& should_stop) {
  should_stop = stop_seen_before_checkpoint ||
                result.status == GpuRunStatus::Failed;
  if (checkpoint_enabled(config, hooks) &&
      write_checkpoint(config, result, hooks) != EXIT_SUCCESS) {
    if (stop_requested(hooks)) {
      result.interruption_requested = true;
      finalize_remaining_interrupted(result);
    } else {
      finalize_unstarted_failed(result);
    }
    result.status = GpuRunStatus::Failed;
    result.reason_code = "checkpoint-write-failed";
    result.results_complete = false;
    result.conclusions_valid = false;
    should_stop = true;
    return EXIT_FAILURE;
  }

  const bool post_checkpoint_stop = stop_requested(hooks);
  if (post_checkpoint_stop && !stop_seen_before_checkpoint) {
    result.interruption_requested = true;
    finalize_remaining_interrupted(result);
    if (result.status != GpuRunStatus::Failed) {
      result.status = GpuRunStatus::Interrupted;
      result.reason_code = "interruption-requested";
    }
    result.results_complete = false;
    result.conclusions_valid = false;
    should_stop = true;
    if (checkpoint_enabled(config, hooks) &&
        write_checkpoint(config, result, hooks) != EXIT_SUCCESS) {
      result.status = GpuRunStatus::Failed;
      result.reason_code = "checkpoint-write-failed";
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}

GpuWorkPlan make_plan(const GpuBandwidthConfig& config,
                      const GpuRunResult& result,
                      GpuOperation operation, size_t passes,
                      bool explicit_iterations) {
  const GpuPipelineMetadata& pipeline =
      pipeline_for_operation(result.backend_initialization.device,
                             operation);
  const GpuResourceMetadata& data_resource = result.allocation.buffer_a;
  const GpuResourceMetadata& status_resource =
      result.allocation.status_buffer;
  const GpuCompilationMetadata& compilation =
      result.backend_initialization.compilation;
  GpuWorkPlanRequest request;
  request.operation = operation;
  request.requested_buffer_bytes = config.buffer_size_bytes;
  request.passes = passes;
  request.base_seed = config.seed;
  request.thread_execution_width = pipeline.thread_execution_width;
  request.max_total_threads_per_threadgroup =
      pipeline.max_total_threads_per_threadgroup;
  request.explicit_iterations = explicit_iterations;
  request.data_resource_options = data_resource.resource_options;
  request.status_resource_options = status_resource.resource_options;
  request.data_storage_mode = data_resource.storage_mode;
  request.data_hazard_tracking_mode =
      data_resource.hazard_tracking_mode;
  request.status_storage_mode = status_resource.storage_mode;
  request.status_hazard_tracking_mode =
      status_resource.hazard_tracking_mode;
  request.kernel_revision = compilation.kernel_revision;
  request.kernel_source_sha256 = compilation.kernel_source_sha256;
  request.msl_language_version = compilation.msl_language_version;
  return build_gpu_work_plan(request);
}

void assign_frozen_plans_to_slots(GpuRunResult& result) {
  for (GpuMeasurement& measurement : result.measurements) {
    measurement.work_plan =
        result.work_plans[operation_index(measurement.operation)];
  }
}

int interrupt_before_measurements(const GpuBandwidthConfig& config,
                                  GpuRunResult& result,
                                  const GpuRunnerTestHooks& hooks) {
  result.interruption_requested = true;
  finalize_remaining_interrupted(result);
  result.status = GpuRunStatus::Interrupted;
  result.reason_code = "interruption-requested";
  result.results_complete = false;
  result.conclusions_valid = false;
  if (checkpoint_enabled(config, hooks) &&
      write_checkpoint(config, result, hooks) != EXIT_SUCCESS) {
    result.status = GpuRunStatus::Failed;
    result.reason_code = "checkpoint-write-failed";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int calibrate_operation(const GpuBandwidthConfig& config,
                        GpuBackend& backend, GpuRunResult& result,
                        const GpuRunnerTestHooks& hooks,
                        GpuOperation operation,
                        const std::function<void()>& update_elapsed) {
  const size_t index = operation_index(operation);
  const GpuPassLimits limits =
      calculate_gpu_pass_limits(config.buffer_size_bytes, operation);
  if (!limits.valid) {
    result.status = GpuRunStatus::Failed;
    result.reason_code = limits.reason_code;
    return EXIT_FAILURE;
  }

  if (config.user_specified_iterations) {
    result.work_plans[index] = make_plan(
        config, result, operation, config.iterations, true);
    if (!result.work_plans[index].valid) {
      result.status = GpuRunStatus::Failed;
      result.reason_code = result.work_plans[index].reason_code;
      return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
  }

  size_t passes = calculate_gpu_pilot_passes(limits);
  std::string purpose = "pilot";
  size_t correction_count = 0;
  GpuWorkPlan latest_plan;

  while (true) {
    if (stop_requested(hooks)) {
      update_elapsed();
      return interrupt_before_measurements(config, result, hooks);
    }

    latest_plan = make_plan(config, result, operation, passes, false);
    if (!latest_plan.valid) {
      result.status = GpuRunStatus::Failed;
      result.reason_code = latest_plan.reason_code;
      return EXIT_FAILURE;
    }

    const AttemptExecution executed =
        execute_backend_attempt(backend, latest_plan);
    GpuCalibrationAttempt attempt;
    attempt.operation = operation;
    attempt.purpose = purpose;
    attempt.passes = latest_plan.passes;
    attempt.exact_payload_bytes = latest_plan.exact_payload_bytes;
    attempt.work_plan = latest_plan;
    attempt.warmup = executed.warmup;
    attempt.precondition = executed.precondition;
    attempt.timed = executed.timed;
    attempt.validation = executed.validation;
    attempt.duration_quality = classify_gpu_duration_quality(
        executed.timed.gpu_elapsed_seconds, latest_plan.passes, limits);
    attempt.terminal = true;
    attempt.valid = executed.status == GpuMeasurementStatus::Measured;
    attempt.reason_code = executed.reason_code;
    result.calibration_attempts[index].push_back(std::move(attempt));

    if (executed.status != GpuMeasurementStatus::Measured) {
      result.status = GpuRunStatus::Failed;
      result.reason_code = executed.reason_code;
      if (stop_requested(hooks)) {
        result.interruption_requested = true;
      }
      return EXIT_FAILURE;
    }
    if (stop_requested(hooks)) {
      update_elapsed();
      return interrupt_before_measurements(config, result, hooks);
    }

    const double elapsed = executed.timed.gpu_elapsed_seconds;
    if (purpose == "pilot") {
      passes = calculate_gpu_calibrated_passes(
          elapsed, latest_plan.passes, limits);
      if (passes == 0) {
        result.status = GpuRunStatus::Failed;
        result.reason_code = "calibration-scaling-failed";
        return EXIT_FAILURE;
      }
      purpose = "duration-trial";
      continue;
    }

    if (gpu_duration_in_target_window(elapsed) ||
        correction_count >= Constants::GPU_CALIBRATION_MAX_CORRECTIONS) {
      break;
    }
    const size_t corrected_passes = calculate_gpu_calibrated_passes(
        elapsed, latest_plan.passes, limits);
    if (corrected_passes == 0 || corrected_passes == latest_plan.passes) {
      break;
    }
    passes = corrected_passes;
    ++correction_count;
    purpose = "correction-trial-" + std::to_string(correction_count);
  }

  result.work_plans[index] = latest_plan;
  return EXIT_SUCCESS;
}

void populate_measurement_from_attempt(GpuMeasurement& measurement,
                                       const AttemptExecution& attempt) {
  measurement.warmup = attempt.warmup;
  measurement.precondition = attempt.precondition;
  measurement.timed = attempt.timed;
  measurement.validation = attempt.validation;
  measurement.timed_command_completed = attempt.timed_command_completed;
  measurement.validation_terminal = attempt.validation_terminal;
  measurement.status = attempt.status;
  measurement.reason_code = attempt.reason_code;
  measurement.duration_quality = classify_gpu_duration_quality(
      attempt.timed.gpu_elapsed_seconds, measurement.work_plan.passes,
      calculate_gpu_pass_limits(measurement.work_plan.effective_buffer_bytes,
                                measurement.operation));

  if (attempt.status != GpuMeasurementStatus::Measured) {
    measurement.value_gb_s.reset();
    return;
  }
  const long double bandwidth =
      static_cast<long double>(measurement.work_plan.exact_payload_bytes) /
      static_cast<long double>(attempt.timed.gpu_elapsed_seconds) / 1.0e9L;
  const double value = static_cast<double>(bandwidth);
  if (!std::isfinite(value) || value <= 0.0) {
    measurement.status = GpuMeasurementStatus::Invalid;
    measurement.reason_code = "invalid-bandwidth-value";
    measurement.value_gb_s.reset();
    return;
  }
  measurement.value_gb_s = value;
}

void release_backend(GpuBackend& backend, GpuRunResult& result) {
  const GpuAllocationResult release = backend.release_resources();
  result.allocation.current_allocated_size_after_release =
      release.current_allocated_size_after_release;
  result.environment_end = backend.snapshot_environment();
  if (release.status == GpuBackendStatus::Failed &&
      result.status != GpuRunStatus::Failed) {
    finalize_unstarted_failed(result);
    result.status = GpuRunStatus::Failed;
    result.reason_code = release.reason_code;
    result.results_complete = false;
    result.conclusions_valid = false;
  }
  update_completion_state(result);
}

}  // namespace

GpuMemoryBudget calculate_gpu_memory_budget(size_t buffer_size_bytes,
                                            size_t available_memory_bytes,
                                            size_t auxiliary_bytes) {
  GpuMemoryBudget budget;
  budget.requested_buffer_bytes = buffer_size_bytes;
  budget.auxiliary_bytes = auxiliary_bytes;
  budget.available_memory_bytes = available_memory_bytes;

  size_t two_buffers = 0;
  if (!NumericUtils::checked_multiply(buffer_size_bytes, 2,
                                      two_buffers) ||
      !NumericUtils::checked_add(two_buffers, auxiliary_bytes,
                                 budget.required_total_bytes)) {
    budget.reason_code = "memory-requirement-overflow";
    return budget;
  }

  if (available_memory_bytes == 0) {
    budget.used_fallback = true;
    if (!NumericUtils::checked_multiply(
            static_cast<size_t>(Constants::FALLBACK_TOTAL_LIMIT_MB),
            Constants::BYTES_PER_MB, budget.memory_budget_bytes)) {
      budget.reason_code = "memory-budget-overflow";
      return budget;
    }
  } else {
    const long double scaled =
        static_cast<long double>(available_memory_bytes) *
        Constants::MEMORY_LIMIT_FACTOR;
    if (scaled > static_cast<long double>(
                     std::numeric_limits<size_t>::max())) {
      budget.reason_code = "memory-budget-overflow";
      return budget;
    }
    budget.memory_budget_bytes = static_cast<size_t>(scaled);
  }

  if (budget.required_total_bytes > budget.memory_budget_bytes) {
    budget.reason_code = "memory-budget-exceeded";
    return budget;
  }
  budget.valid = true;
  budget.reason_code = "within-memory-budget";
  return budget;
}

GpuBackendAttemptRequest build_gpu_backend_attempt_request(
    const GpuWorkPlan& plan) {
  GpuBackendAttemptRequest request;
  request.operation = plan.operation;
  request.buffer_size_bytes = plan.effective_buffer_bytes;
  request.passes = plan.passes;
  request.operation_seed = plan.operation_seed;
  request.vector_count = plan.vector_count;
  request.tail_bytes = plan.tail_bytes;
  request.threads_per_threadgroup = plan.threads_per_threadgroup;
  request.threadgroups_per_grid = plan.threadgroups_per_grid;
  request.grid_threads = plan.grid_threads;
  return request;
}

int run_gpu_bandwidth_suite(const GpuBandwidthConfig& config,
                            GpuBackend& backend,
                            GpuRunResult& result,
                            const GpuRunnerTestHooks& hooks) {
  const auto start_time = std::chrono::steady_clock::now();
  bool resources_allocated = false;
  const auto update_elapsed = [&]() {
    result.elapsed_host_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_time).count();
  };
  const auto finish_pre_run_failure = [&]() {
    update_elapsed();
    if (checkpoint_enabled(config, hooks) &&
        write_checkpoint(config, result, hooks) != EXIT_SUCCESS) {
      result.status = GpuRunStatus::Failed;
      result.reason_code = "checkpoint-write-failed";
    }
    return EXIT_FAILURE;
  };

  try {
    if (!initialize_result(config, result)) {
      result.status = GpuRunStatus::Failed;
      result.reason_code = "planned-measurement-count-overflow";
      return finish_pre_run_failure();
    }

    result.backend_initialization = backend.initialize();
    if (result.backend_initialization.status != GpuBackendStatus::Success) {
      result.status = result.backend_initialization.status ==
                              GpuBackendStatus::Unsupported
                          ? GpuRunStatus::Unsupported
                          : GpuRunStatus::Failed;
      result.reason_code = result.backend_initialization.reason_code;
      return finish_pre_run_failure();
    }

    if (config.buffer_size_bytes >
        result.backend_initialization.device.max_buffer_length) {
      result.status = GpuRunStatus::Failed;
      result.reason_code = "max-buffer-length-exceeded";
      return finish_pre_run_failure();
    }

    result.memory_budget = calculate_gpu_memory_budget(
        config.buffer_size_bytes,
        static_cast<size_t>(
            result.backend_initialization.device.available_memory_bytes),
        Constants::GPU_AUXILIARY_BUFFER_BYTES);
    if (!result.memory_budget.valid) {
      result.status = GpuRunStatus::Failed;
      result.reason_code = result.memory_budget.reason_code;
      return finish_pre_run_failure();
    }

    GpuAllocationRequest allocation_request;
    allocation_request.buffer_size_bytes = config.buffer_size_bytes;
    allocation_request.auxiliary_bytes =
        Constants::GPU_AUXILIARY_BUFFER_BYTES;
    allocation_request.memory_budget_bytes =
        result.memory_budget.memory_budget_bytes;
    result.allocation = backend.allocate_resources(allocation_request);
    if (result.allocation.status != GpuBackendStatus::Success) {
      result.status = GpuRunStatus::Failed;
      result.reason_code = result.allocation.reason_code;
      return finish_pre_run_failure();
    }
    resources_allocated = true;
    result.environment_start = backend.snapshot_environment();

    for (GpuOperation operation : kGpuOperations) {
      const int calibration_status = calibrate_operation(
          config, backend, result, hooks, operation, update_elapsed);
      if (calibration_status != EXIT_SUCCESS ||
          result.status == GpuRunStatus::Interrupted) {
        assign_frozen_plans_to_slots(result);
        if (result.status != GpuRunStatus::Interrupted) {
          if (result.interruption_requested) {
            finalize_remaining_interrupted(result);
          } else {
            finalize_unstarted_failed(result);
          }
          update_completion_state(result);
        }
        update_elapsed();
        if (result.status != GpuRunStatus::Interrupted &&
            result.reason_code != "checkpoint-write-failed" &&
            hooks.checkpoint &&
            write_checkpoint(config, result, hooks) != EXIT_SUCCESS) {
          if (stop_requested(hooks)) {
            result.interruption_requested = true;
            finalize_remaining_interrupted(result);
          } else {
            finalize_unstarted_failed(result);
          }
          result.status = GpuRunStatus::Failed;
          result.reason_code = "checkpoint-write-failed";
        }
        release_backend(backend, result);
        update_elapsed();
        if (!hooks.checkpoint && !config.output_file.empty() &&
            save_gpu_bandwidth_json(config, result, false) != EXIT_SUCCESS) {
          result.status = GpuRunStatus::Failed;
          result.reason_code = "checkpoint-write-failed";
          result.results_complete = false;
          result.conclusions_valid = false;
          return EXIT_FAILURE;
        }
        return result.status == GpuRunStatus::Interrupted
                   ? EXIT_SUCCESS
                   : EXIT_FAILURE;
      }
    }
    assign_frozen_plans_to_slots(result);

    bool stop_all = false;
    for (GpuLoopRecord& loop : result.loops) {
      for (size_t position = 0; position < loop.planned_order.size();
           ++position) {
        if (stop_requested(hooks)) {
          update_elapsed();
          const int interrupted =
              interrupt_before_measurements(config, result, hooks);
          stop_all = true;
          if (interrupted != EXIT_SUCCESS) {
            break;
          }
          break;
        }

        const size_t measurement_index =
            loop.measurement_indexes[position];
        GpuMeasurement& measurement =
            result.measurements[measurement_index];
        measurement.attempted = true;
        loop.realized_order.push_back(measurement.operation);
        measurement.environment_before = backend.snapshot_environment();
        const AttemptExecution attempt = execute_backend_attempt(
            backend, measurement.work_plan);
        measurement.environment_after = backend.snapshot_environment();
        populate_measurement_from_attempt(measurement, attempt);

        if (measurement.status == GpuMeasurementStatus::Failed ||
            measurement.status == GpuMeasurementStatus::Invalid) {
          result.status = GpuRunStatus::Failed;
          result.reason_code = measurement.reason_code;
        }
        update_completion_state(result);

        const bool stop_after_task = stop_requested(hooks);
        if (stop_after_task) {
          result.interruption_requested = true;
          finalize_remaining_interrupted(result);
          if (result.status != GpuRunStatus::Failed) {
            result.status = GpuRunStatus::Interrupted;
            result.reason_code = "interruption-requested";
          }
          result.results_complete = false;
          result.conclusions_valid = false;
          update_completion_state(result);
        } else if (result.status == GpuRunStatus::Failed) {
          finalize_unstarted_failed(result);
          update_completion_state(result);
        }

        update_elapsed();
        bool terminal_stop = false;
        if (checkpoint_terminal_state(config, result, hooks,
                                      stop_after_task,
                                      terminal_stop) != EXIT_SUCCESS) {
          stop_all = true;
          break;
        }
        if (terminal_stop || result.status == GpuRunStatus::Failed) {
          stop_all = true;
          break;
        }
      }
      if (stop_all) {
        break;
      }
    }

    update_completion_state(result);
    release_backend(backend, result);
    resources_allocated = false;
    update_elapsed();

    // The terminal measurement checkpoint intentionally retains resources.
    // A final production-only replacement records the post-release allocation
    // snapshot; injected checkpoint tests keep the exact task-boundary count.
    if (!hooks.checkpoint && !config.output_file.empty() &&
        save_gpu_bandwidth_json(config, result, false) != EXIT_SUCCESS) {
      finalize_unstarted_failed(result);
      result.status = GpuRunStatus::Failed;
      result.reason_code = "checkpoint-write-failed";
      result.results_complete = false;
      result.conclusions_valid = false;
      return EXIT_FAILURE;
    }

    return result.status == GpuRunStatus::Failed ||
                   result.status == GpuRunStatus::Unsupported
               ? EXIT_FAILURE
               : EXIT_SUCCESS;
  } catch (const std::exception&) {
    result.status = GpuRunStatus::Failed;
    result.reason_code = "runner-exception";
  } catch (...) {
    result.status = GpuRunStatus::Failed;
    result.reason_code = "runner-unknown-exception";
  }

  if (resources_allocated) {
    finalize_unstarted_failed(result);
    release_backend(backend, result);
  }
  result.results_complete = false;
  result.conclusions_valid = false;
  update_elapsed();
  if (!hooks.checkpoint && !config.output_file.empty() &&
      save_gpu_bandwidth_json(config, result, false) != EXIT_SUCCESS) {
    result.status = GpuRunStatus::Failed;
    result.reason_code = "checkpoint-write-failed";
  }
  return EXIT_FAILURE;
}

const char* gpu_measurement_status_to_string(GpuMeasurementStatus status) {
  switch (status) {
    case GpuMeasurementStatus::NotRun:
      return "not-run";
    case GpuMeasurementStatus::Measured:
      return "measured";
    case GpuMeasurementStatus::Interrupted:
      return "interrupted";
    case GpuMeasurementStatus::Invalid:
      return "invalid";
    case GpuMeasurementStatus::Failed:
      return "failed";
  }
  return "failed";
}

const char* gpu_run_status_to_string(GpuRunStatus status) {
  switch (status) {
    case GpuRunStatus::NotStarted:
      return "not-started";
    case GpuRunStatus::Complete:
      return "complete";
    case GpuRunStatus::Partial:
      return "partial";
    case GpuRunStatus::Interrupted:
      return "interrupted";
    case GpuRunStatus::Failed:
      return "failed";
    case GpuRunStatus::Unsupported:
      return "unsupported";
  }
  return "failed";
}

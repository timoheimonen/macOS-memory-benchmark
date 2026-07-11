// Copyright 2026 Timo Heimonen <timo.heimonen@proton.me>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

/**
 * @file gpu_json.cpp
 * @brief GPU bandwidth JSON schema v1 implementation
 */

#include "gpu_bandwidth/gpu_json.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>

#include "core/config/constants.h"
#include "core/config/version.h"
#include "output/json/json_output/json_output_api.h"
#include "utils/json_utils.h"

namespace {

using OrderedJson = nlohmann::ordered_json;

size_t operation_index(GpuOperation operation) {
  switch (operation) {
    case GpuOperation::Read:
      return 0;
    case GpuOperation::Write:
      return 1;
    case GpuOperation::Copy:
      return 2;
  }
  return 0;
}

OrderedJson finite_or_null(double value) {
  return std::isfinite(value) ? OrderedJson(value) : OrderedJson(nullptr);
}

OrderedJson optional_finite_or_null(const std::optional<double>& value) {
  return value.has_value() && std::isfinite(*value)
             ? OrderedJson(*value)
             : OrderedJson(nullptr);
}

OrderedJson error_json(const GpuErrorDiagnostic& error) {
  return OrderedJson{{"domain", error.domain.empty() ? OrderedJson(nullptr)
                                                     : OrderedJson(error.domain)},
                     {"code", error.code},
                     {"description",
                      error.description.empty() ? OrderedJson(nullptr)
                                                : OrderedJson(error.description)}};
}

OrderedJson dual_checksum_json(const GpuDualChecksum& checksum) {
  return OrderedJson{{"first_uint32_decimal", std::to_string(checksum.first)},
                     {"second_uint32_decimal", std::to_string(checksum.second)}};
}

OrderedJson phase_json(const GpuBackendPhaseResult& phase) {
  return OrderedJson{{"status", gpu_backend_status_to_string(phase.status)},
                     {"reason_code", phase.reason_code},
                     {"command_buffer_status",
                      gpu_command_status_to_string(phase.command_status)},
                     {"command_buffer_count", phase.command_buffer_count},
                     {"compute_encoder_count", phase.compute_encoder_count},
                     {"dispatch_count", phase.dispatch_count},
                     {"data_initialization_dispatch_count",
                      phase.data_initialization_dispatch_count},
                     {"benchmark_operation_dispatch_count",
                      phase.benchmark_operation_dispatch_count},
                     {"validation_dispatch_count",
                      phase.validation_dispatch_count},
                     {"status_reset_count", phase.status_reset_count},
                     {"error", error_json(phase.error)}};
}

OrderedJson timed_json(const GpuTimedResult& timed) {
  OrderedJson output = phase_json(timed);
  const bool completed = timed.command_status == GpuCommandStatus::Completed;
  output["gpu_start_seconds"] = completed
                                      ? finite_or_null(timed.gpu_start_seconds)
                                      : OrderedJson(nullptr);
  output["gpu_end_seconds"] = completed
                                    ? finite_or_null(timed.gpu_end_seconds)
                                    : OrderedJson(nullptr);
  output["gpu_elapsed_seconds"] = completed
                                        ? finite_or_null(timed.gpu_elapsed_seconds)
                                        : OrderedJson(nullptr);
  output["host_wall_seconds"] = completed
                                      ? finite_or_null(timed.host_wall_seconds)
                                      : OrderedJson(nullptr);
  output["host_submit_seconds"] = completed
                                        ? finite_or_null(timed.host_submit_seconds)
                                        : OrderedJson(nullptr);
  output["host_wait_end_seconds"] = completed
                                          ? finite_or_null(timed.host_wait_end_seconds)
                                          : OrderedJson(nullptr);
  output["queue_delay_seconds"] =
      timed.queue_delay_available
          ? finite_or_null(timed.queue_delay_seconds)
          : OrderedJson(nullptr);
  output["expected_timed_accumulator"] =
      completed ? dual_checksum_json(timed.expected_accumulator)
                : OrderedJson(nullptr);
  output["actual_timed_accumulator"] =
      completed ? dual_checksum_json(timed.actual_accumulator)
                : OrderedJson(nullptr);
  return output;
}

OrderedJson validation_json(const GpuValidationResult& validation) {
  OrderedJson output = phase_json(validation);
  output["validation_status"] =
      gpu_validation_status_to_string(validation.validation_status);
  output["timed_accumulator_algorithm"] =
      validation.timed_accumulator_algorithm;
  output["final_checksum_algorithm"] =
      validation.final_checksum_algorithm;
  const bool terminal =
      validation.command_status == GpuCommandStatus::Completed;
  output["expected_final_checksum"] =
      terminal ? dual_checksum_json(validation.expected_final_checksum)
               : OrderedJson(nullptr);
  output["actual_final_checksum"] =
      terminal ? dual_checksum_json(validation.actual_final_checksum)
               : OrderedJson(nullptr);
  return output;
}

OrderedJson environment_snapshot_json(
    const GpuEnvironmentSnapshot& environment) {
  return OrderedJson{
      {"thermal_state", environment.thermal_state},
      {"low_power_mode_available", environment.low_power_mode_available},
      {"low_power_mode_enabled",
       environment.low_power_mode_available
           ? OrderedJson(environment.low_power_mode_enabled)
           : OrderedJson(nullptr)},
      {"current_allocated_size_bytes",
       std::to_string(environment.current_allocated_size)}};
}

OrderedJson pipeline_json(const GpuPipelineMetadata& pipeline) {
  return OrderedJson{
      {"label", pipeline.label},
      {"thread_execution_width", pipeline.thread_execution_width},
      {"max_total_threads_per_threadgroup",
       pipeline.max_total_threads_per_threadgroup}};
}

OrderedJson resource_json(const GpuResourceMetadata& resource) {
  return OrderedJson{{"label", resource.label},
                     {"storage_mode", resource.storage_mode},
                     {"cpu_cache_mode", resource.cpu_cache_mode},
                     {"hazard_tracking_mode", resource.hazard_tracking_mode},
                     {"resource_options_uint64_decimal",
                      std::to_string(resource.resource_options)},
                     {"length_bytes", std::to_string(resource.length_bytes)}};
}

OrderedJson work_plan_json(const GpuWorkPlan& plan) {
  return OrderedJson{
      {"valid", plan.valid},
      {"reason_code", plan.reason_code},
      {"operation", gpu_operation_to_string(plan.operation)},
      {"requested_buffer_bytes", std::to_string(plan.requested_buffer_bytes)},
      {"effective_buffer_bytes", std::to_string(plan.effective_buffer_bytes)},
      {"base_seed_uint64_decimal", std::to_string(plan.base_seed)},
      {"operation_seed_uint64_decimal", std::to_string(plan.operation_seed)},
      {"passes", plan.passes},
      {"payload_multiplier", plan.payload_multiplier},
      {"bytes_per_pass", std::to_string(plan.bytes_per_pass)},
      {"exact_payload_bytes", std::to_string(plan.exact_payload_bytes)},
      {"maximum_passes_by_dispatch", plan.maximum_passes_by_dispatch},
      {"maximum_passes_by_payload", plan.maximum_passes_by_payload},
      {"effective_maximum_passes", plan.effective_maximum_passes},
      {"dispatch_cap_is_limiting", plan.dispatch_cap_is_limiting},
      {"payload_cap_is_limiting", plan.payload_cap_is_limiting},
      {"vector_width_bytes", plan.vector_width_bytes},
      {"vector_count", std::to_string(plan.vector_count)},
      {"tail_bytes", plan.tail_bytes},
      {"thread_execution_width", plan.thread_execution_width},
      {"max_total_threads_per_threadgroup",
       plan.max_total_threads_per_threadgroup},
      {"threads_per_threadgroup", plan.threads_per_threadgroup},
      {"required_threadgroups_per_grid",
       plan.required_threadgroups_per_grid},
      {"maximum_threadgroups_per_grid",
       plan.maximum_threadgroups_per_grid},
      {"threadgroups_per_grid", plan.threadgroups_per_grid},
      {"grid_threads", plan.grid_threads},
      {"dispatch_count", plan.dispatch_count},
      {"measured_command_buffer_count",
       plan.measured_command_buffer_count},
      {"measured_compute_encoder_count",
       plan.measured_compute_encoder_count},
      {"resource_options",
       OrderedJson{{"data_uint64_decimal",
                    std::to_string(plan.data_resource_options)},
                   {"status_uint64_decimal",
                    std::to_string(plan.status_resource_options)},
                   {"data_storage_mode", plan.data_storage_mode},
                   {"data_hazard_tracking_mode",
                    plan.data_hazard_tracking_mode},
                   {"status_storage_mode", plan.status_storage_mode},
                   {"status_hazard_tracking_mode",
                    plan.status_hazard_tracking_mode}}},
      {"kernel_revision", plan.kernel_revision},
      {"kernel_source_sha256", plan.kernel_source_sha256},
      {"msl_language_version", plan.msl_language_version},
      {"methodology_version", plan.methodology_version},
      {"timing_policy", plan.timing_policy},
      {"pass_mapping", plan.pass_mapping},
      {"warmup_policy", plan.warmup_policy},
      {"calibration_policy", plan.calibration_policy},
      {"duration_quality_policy", plan.duration_quality_policy},
      {"timed_accumulator_algorithm",
       plan.timed_accumulator_algorithm},
      {"final_checksum_algorithm", plan.final_checksum_algorithm},
      {"plan_identity", plan.plan_identity}};
}

OrderedJson measurement_json(const GpuMeasurement& measurement,
                             const GpuRunResult& result) {
  const GpuAllocationResult& allocation = result.allocation;
  OrderedJson output{
      {"status", gpu_measurement_status_to_string(measurement.status)},
      {"reason_code", measurement.reason_code},
      {"value_gb_s", optional_finite_or_null(measurement.value_gb_s)},
      {"units", "GB/s"},
      {"operation", gpu_operation_to_string(measurement.operation)},
      {"loop_index", measurement.loop_index},
      {"operation_order_position", measurement.operation_order_position},
      {"work_policy",
       measurement.work_plan.valid &&
               measurement.work_plan.passes > 0
           ? (result.calibration_attempts[operation_index(
                  measurement.operation)].empty()
                  ? "explicit-fixed-work"
                  : "automatic-frozen-work")
           : "not-resolved"},
      {"duration_quality", measurement.duration_quality},
      {"work_plan", work_plan_json(measurement.work_plan)},
      {"warmup", phase_json(measurement.warmup)},
      {"precondition", phase_json(measurement.precondition)},
      {"timed", timed_json(measurement.timed)},
      {"validation", validation_json(measurement.validation)},
      {"environment_before",
       environment_snapshot_json(measurement.environment_before)},
      {"environment_after",
       environment_snapshot_json(measurement.environment_after)}};

  output["resources"] = OrderedJson{
      {"buffer_a", resource_json(allocation.buffer_a)},
      {"buffer_b", resource_json(allocation.buffer_b)},
      {"status_buffer", resource_json(allocation.status_buffer)}};
  output["kernel_revision"] =
      result.backend_initialization.compilation.kernel_revision;
  output["kernel_source_sha256"] =
      result.backend_initialization.compilation.kernel_source_sha256;
  output["msl_language_version"] =
      result.backend_initialization.compilation.msl_language_version;
  return output;
}

OrderedJson calibration_attempt_json(const GpuCalibrationAttempt& attempt) {
  return OrderedJson{
      {"operation", gpu_operation_to_string(attempt.operation)},
      {"purpose", attempt.purpose},
      {"passes", attempt.passes},
      {"exact_payload_bytes", std::to_string(attempt.exact_payload_bytes)},
      {"work_plan", work_plan_json(attempt.work_plan)},
      {"terminal", attempt.terminal},
      {"valid", attempt.valid},
      {"reason_code", attempt.reason_code},
      {"duration_quality", attempt.duration_quality},
      {"warmup", phase_json(attempt.warmup)},
      {"precondition", phase_json(attempt.precondition)},
      {"timed", timed_json(attempt.timed)},
      {"validation", validation_json(attempt.validation)}};
}

OrderedJson aggregate_json(const GpuOperationAggregate& aggregate) {
  OrderedJson values = OrderedJson::array();
  for (double value : aggregate.values_gb_s) {
    values.push_back(finite_or_null(value));
  }
  OrderedJson output{
      {"operation", gpu_operation_to_string(aggregate.operation)},
      {"status", aggregate.status},
      {"sample_count", aggregate.values_gb_s.size()},
      {"headline_semantics",
       aggregate.values_gb_s.size() > 1 ? "median-p50"
                                       : "single-measurement"},
      {"headline_gb_s", optional_finite_or_null(aggregate.headline_gb_s)},
      {"values_gb_s", std::move(values)},
      {"statistics", calculate_json_statistics(aggregate.values_gb_s)},
      {"stability_quality", aggregate.stability_quality},
      {"cv_warning_threshold_pct",
       Constants::GPU_STREAMING_CV_WARNING_PCT}};
  return output;
}

OrderedJson compilation_json(const GpuCompilationMetadata& compilation) {
  return OrderedJson{
      {"compilation_mode", compilation.compilation_mode},
      {"msl_language_version", compilation.msl_language_version},
      {"floating_point_math", compilation.floating_point_math},
      {"preprocessor_macros", compilation.preprocessor_macros},
      {"kernel_revision", compilation.kernel_revision},
      {"kernel_source_sha256", compilation.kernel_source_sha256},
      {"compiler_diagnostics",
       compilation.compiler_diagnostics.empty()
           ? OrderedJson(nullptr)
           : OrderedJson(compilation.compiler_diagnostics)},
      {"compiler_identifier", compilation.compiler_identifier},
      {"build_sdk", compilation.build_sdk},
      {"deployment_target", compilation.deployment_target}};
}

OrderedJson device_json(const GpuDeviceMetadata& device) {
  OrderedJson families = OrderedJson::array();
  for (const std::string& family : device.supported_families) {
    families.push_back(family);
  }
  return OrderedJson{
      {"device_name", device.device_name},
      {"registry_id_uint64_decimal", std::to_string(device.registry_id)},
      {"has_unified_memory", device.has_unified_memory},
      {"required_apple7_family_supported",
       device.required_apple7_family_supported},
      {"supported_families", std::move(families)},
      {"max_buffer_length_bytes", std::to_string(device.max_buffer_length)},
      {"recommended_max_working_set_size_bytes",
       std::to_string(device.recommended_max_working_set_size)},
      {"current_allocated_size_at_initialization_bytes",
       std::to_string(device.current_allocated_size)},
      {"available_memory_bytes",
       std::to_string(device.available_memory_bytes)},
      {"available_memory_source", device.available_memory_source},
      {"pipelines",
       OrderedJson{{"read", pipeline_json(device.read_pipeline)},
                   {"write", pipeline_json(device.write_pipeline)},
                   {"copy", pipeline_json(device.copy_pipeline)},
                   {"initialization",
                    pipeline_json(device.initialization_pipeline)},
                   {"validation",
                    pipeline_json(device.validation_pipeline)}}}};
}

OrderedJson allocation_json(const GpuAllocationResult& allocation) {
  return OrderedJson{
      {"status", gpu_backend_status_to_string(allocation.status)},
      {"reason_code", allocation.reason_code},
      {"error", error_json(allocation.error)},
      {"requested_buffer_size_bytes",
       std::to_string(allocation.requested_buffer_size_bytes)},
      {"auxiliary_bytes", std::to_string(allocation.auxiliary_bytes)},
      {"required_total_bytes",
       std::to_string(allocation.required_total_bytes)},
      {"memory_budget_bytes",
       std::to_string(allocation.memory_budget_bytes)},
      {"current_allocated_size_before_bytes",
       std::to_string(allocation.current_allocated_size_before)},
      {"current_allocated_size_peak_bytes",
       std::to_string(allocation.current_allocated_size_peak)},
      {"current_allocated_size_after_release_bytes",
       std::to_string(allocation.current_allocated_size_after_release)},
      {"recommended_working_set_available",
       allocation.recommended_working_set_available},
      {"recommended_working_set_headroom_bytes",
       allocation.recommended_working_set_available
           ? OrderedJson(std::to_string(
                 allocation.recommended_working_set_headroom_bytes))
           : OrderedJson(nullptr)},
      {"recommended_working_set_headroom_fraction",
       allocation.recommended_working_set_available
           ? finite_or_null(
                 allocation.recommended_working_set_headroom_fraction)
           : OrderedJson(nullptr)},
      {"exceeds_recommended_working_set",
       allocation.exceeds_recommended_working_set},
      {"buffer_a", resource_json(allocation.buffer_a)},
      {"buffer_b", resource_json(allocation.buffer_b)},
      {"status_buffer", resource_json(allocation.status_buffer)}};
}

}  // namespace

nlohmann::ordered_json build_gpu_bandwidth_json(
    const GpuBandwidthConfig& config, const GpuRunResult& result) {
  OrderedJson output;
  output["software_version"] = SOFTVERSION;
  output["version"] = SOFTVERSION;
  output["timestamp"] =
      result.timestamp.empty() ? build_utc_timestamp() : result.timestamp;
  output["schema_version"] = Constants::GPU_JSON_SCHEMA_VERSION;
  output["mode"] = Constants::GPU_JSON_MODE_NAME;
  output["methodology_version"] = Constants::GPU_METHODOLOGY_VERSION;
  output["status"] = gpu_run_status_to_string(result.status);
  output["reason_code"] = result.reason_code;
  output["interruption_requested"] = result.interruption_requested;
  output["results_complete"] = result.results_complete;
  output["conclusions_valid"] = result.conclusions_valid;
  output["operation_order_balance_complete"] =
      result.operation_order_balance_complete;
  output["execution_time_sec"] = finite_or_null(result.elapsed_host_seconds);
  output["dram_residency"] = "unverified";
  output["payload_semantics"] =
      "effective-kernel-payload-divided-by-metal-gpu-time";
  output["copy_payload_semantics"] = "aggregate-read-plus-write";
  output["methodology"] = OrderedJson{
      {"primary_timing",
       "command_buffer_gpu_end_time_minus_gpu_start_time"},
      {"measured_command_buffers_per_attempt", 1},
      {"measured_compute_encoders_per_attempt", 1},
      {"compute_encoder_dispatch_type", "serial"},
      {"pass_mapping", "one-full-buffer-pass-per-dispatch"},
      {"grid_stride_mapping_version", "gpu-grid-stride-v1"},
      {"warmup_policy", "steady-state-warm-memory"},
      {"data_resource_policy", "private-tracked-suite-resident-pair"},
      {"calibration_target_seconds",
       Constants::GPU_CALIBRATION_TARGET_SECONDS},
      {"calibration_min_seconds",
       Constants::GPU_CALIBRATION_MIN_SECONDS},
      {"calibration_max_seconds",
       Constants::GPU_CALIBRATION_MAX_SECONDS},
      {"calibration_max_corrections",
       Constants::GPU_CALIBRATION_MAX_CORRECTIONS},
      {"calibration_min_pilot_payload_bytes",
       std::to_string(Constants::GPU_CALIBRATION_MIN_PILOT_BYTES)},
      {"maximum_dispatches_per_measurement",
       Constants::GPU_MAX_DISPATCHES_PER_MEASUREMENT},
      {"maximum_exact_payload_bytes",
       std::to_string(Constants::GPU_MAX_EXACT_PAYLOAD_BYTES)},
      {"maximum_threadgroups_per_grid",
       Constants::GPU_MAX_THREADGROUPS_PER_GRID},
      {"timed_observable_reduction", "dual-mod32-threadgroup-reduction-v2"},
      {"reduction_traffic_in_payload", false},
      {"reduction_traffic_in_gpu_time", true},
      {"final_validation", "dual-mod32-v1"}};

  OrderedJson argv = OrderedJson::array();
  for (const std::string& argument : config.argv) {
    argv.push_back(argument);
  }
  output["configuration"] = OrderedJson{
      {"buffer_size_mb", config.buffer_size_mb},
      {"buffer_size_bytes", std::to_string(config.buffer_size_bytes)},
      {"iterations",
       config.user_specified_iterations ? OrderedJson(config.iterations)
                                        : OrderedJson(nullptr)},
      {"work_policy",
       config.user_specified_iterations ? "explicit-fixed-work"
                                        : "automatic-calibration"},
      {"loop_count", config.loop_count},
      {"base_seed_uint64_decimal", std::to_string(config.seed)},
      {"seed_source", config.user_specified_seed ? "user" : "generated"},
      {"output_file",
       config.output_file.empty() ? OrderedJson(nullptr)
                                  : OrderedJson(config.output_file)},
      {"argv", std::move(argv)}};

  output["counters"] = OrderedJson{
      {"planned_loops", result.counters.planned_loops},
      {"attempted_loops", result.counters.attempted_loops},
      {"completed_loops", result.counters.completed_loops},
      {"planned_measurements", result.counters.planned_measurements},
      {"attempted_measurements", result.counters.attempted_measurements},
      {"terminal_measurements", result.counters.terminal_measurements},
      {"completed_measurements", result.counters.completed_measurements},
      {"validated_measurements", result.counters.validated_measurements}};

  const GpuDeviceMetadata& device =
      result.backend_initialization.device;
  output["environment"] = OrderedJson{
      {"macos_product_version", device.macos_product_version},
      {"macos_build", device.macos_build},
      {"hardware_model", device.hardware_model},
      {"physical_memory_bytes", std::to_string(device.physical_memory_bytes)},
      {"main_thread_qos",
       OrderedJson{{"requested", result.main_thread_qos.requested},
                   {"applied", result.main_thread_qos.applied},
                   {"code", result.main_thread_qos.code}}},
      {"start", environment_snapshot_json(result.environment_start)},
      {"end", environment_snapshot_json(result.environment_end)}};

  output["backend"] = OrderedJson{
      {"initialization_status",
       gpu_backend_status_to_string(result.backend_initialization.status)},
      {"reason_code", result.backend_initialization.reason_code},
      {"error", error_json(result.backend_initialization.error)},
      {"device", device_json(device)},
      {"compilation",
       compilation_json(result.backend_initialization.compilation)},
      {"allocation", allocation_json(result.allocation)}};

  output["memory_budget"] = OrderedJson{
      {"valid", result.memory_budget.valid},
      {"reason_code", result.memory_budget.reason_code},
      {"requested_buffer_bytes",
       std::to_string(result.memory_budget.requested_buffer_bytes)},
      {"auxiliary_bytes",
       std::to_string(result.memory_budget.auxiliary_bytes)},
      {"required_total_bytes",
       std::to_string(result.memory_budget.required_total_bytes)},
      {"available_memory_bytes",
       std::to_string(result.memory_budget.available_memory_bytes)},
      {"memory_budget_bytes",
       std::to_string(result.memory_budget.memory_budget_bytes)},
      {"used_fallback", result.memory_budget.used_fallback}};

  OrderedJson work_plans = OrderedJson::array();
  OrderedJson calibration = OrderedJson::object();
  OrderedJson aggregates = OrderedJson::object();
  for (size_t index = 0; index < kGpuOperationCount; ++index) {
    const GpuOperation operation = static_cast<GpuOperation>(index);
    work_plans.push_back(work_plan_json(result.work_plans[index]));
    OrderedJson attempts = OrderedJson::array();
    for (const GpuCalibrationAttempt& attempt :
         result.calibration_attempts[index]) {
      attempts.push_back(calibration_attempt_json(attempt));
    }
    calibration[gpu_operation_to_string(operation)] = std::move(attempts);
    aggregates[gpu_operation_to_string(operation)] =
        aggregate_json(result.aggregates[index]);
  }
  output["work_plans"] = std::move(work_plans);
  output["excluded_calibration_attempts"] = std::move(calibration);

  OrderedJson measurements = OrderedJson::array();
  for (const GpuMeasurement& measurement : result.measurements) {
    measurements.push_back(measurement_json(measurement, result));
  }
  output["measurements"] = measurements;

  OrderedJson loops = OrderedJson::array();
  for (const GpuLoopRecord& loop : result.loops) {
    OrderedJson planned = OrderedJson::array();
    OrderedJson realized = OrderedJson::array();
    OrderedJson loop_measurements = OrderedJson::array();
    for (GpuOperation operation : loop.planned_order) {
      planned.push_back(gpu_operation_to_string(operation));
    }
    for (GpuOperation operation : loop.realized_order) {
      realized.push_back(gpu_operation_to_string(operation));
    }
    for (size_t measurement_index : loop.measurement_indexes) {
      loop_measurements.push_back(
          measurement_json(result.measurements[measurement_index], result));
    }
    loops.push_back(OrderedJson{{"loop_index", loop.loop_index},
                                {"planned_order", std::move(planned)},
                                {"realized_order", std::move(realized)},
                                {"measurements",
                                 std::move(loop_measurements)}});
  }
  output["loop_records"] = std::move(loops);
  output["aggregates"] = std::move(aggregates);
  output["quality_warnings"] = result.quality_warnings;
  return output;
}

int save_gpu_bandwidth_json(const GpuBandwidthConfig& config,
                            const GpuRunResult& result,
                            bool announce_success) {
  if (config.output_file.empty()) {
    return EXIT_SUCCESS;
  }
  return write_json_to_file(std::filesystem::path(config.output_file),
                            build_gpu_bandwidth_json(config, result),
                            announce_success);
}

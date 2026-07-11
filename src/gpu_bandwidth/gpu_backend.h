// Copyright 2026 Timo Heimonen <timo.heimonen@proton.me>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

/**
 * @file gpu_backend.h
 * @brief Pure C++ contract for the private Metal GPU backend
 */

#ifndef GPU_BACKEND_H
#define GPU_BACKEND_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gpu_bandwidth/gpu_work_plan.h"

enum class GpuBackendStatus {
  NotRun = 0,
  Success,
  Failed,
  Unsupported,
};

enum class GpuCommandStatus {
  NotRun = 0,
  Completed,
  Error,
};

enum class GpuValidationStatus {
  NotRun = 0,
  Passed,
  Mismatch,
  Error,
  NotRunTimerInvalid,
};

struct GpuErrorDiagnostic {
  std::string domain;
  long long code = 0;
  std::string description;
};

struct GpuDualChecksum {
  uint32_t first = 0;
  uint32_t second = 0;

  bool operator==(const GpuDualChecksum& other) const {
    return first == other.first && second == other.second;
  }
  bool operator!=(const GpuDualChecksum& other) const {
    return !(*this == other);
  }
};

struct GpuPipelineMetadata {
  std::string label;
  size_t thread_execution_width = 0;
  size_t max_total_threads_per_threadgroup = 0;
};

struct GpuCompilationMetadata {
  std::string compilation_mode = "runtime-source";
  std::string msl_language_version = "2.3";
  std::string floating_point_math = "not_applicable_integer_only";
  std::string preprocessor_macros = "none";
  std::string kernel_revision;
  std::string kernel_source_sha256;
  std::string compiler_diagnostics;
  std::string compiler_identifier;
  std::string build_sdk;
  std::string deployment_target;
};

struct GpuDeviceMetadata {
  std::string macos_product_version;
  std::string macos_build;
  std::string hardware_model;
  uint64_t physical_memory_bytes = 0;
  std::string device_name;
  uint64_t registry_id = 0;
  bool has_unified_memory = false;
  bool required_apple7_family_supported = false;
  std::vector<std::string> supported_families;
  size_t max_buffer_length = 0;
  uint64_t recommended_max_working_set_size = 0;
  uint64_t current_allocated_size = 0;
  uint64_t available_memory_bytes = 0;
  std::string available_memory_source;
  GpuPipelineMetadata read_pipeline;
  GpuPipelineMetadata write_pipeline;
  GpuPipelineMetadata copy_pipeline;
  GpuPipelineMetadata initialization_pipeline;
  GpuPipelineMetadata validation_pipeline;
};

struct GpuResourceMetadata {
  std::string label;
  std::string storage_mode;
  std::string cpu_cache_mode;
  std::string hazard_tracking_mode;
  uint64_t resource_options = 0;
  size_t length_bytes = 0;
};

struct GpuEnvironmentSnapshot {
  std::string thermal_state = "unavailable";
  bool low_power_mode_available = false;
  bool low_power_mode_enabled = false;
  uint64_t current_allocated_size = 0;
};

struct GpuBackendInitialization {
  GpuBackendStatus status = GpuBackendStatus::Failed;
  std::string reason_code = "backend-not-initialized";
  GpuErrorDiagnostic error;
  GpuDeviceMetadata device;
  GpuCompilationMetadata compilation;
};

struct GpuAllocationRequest {
  size_t buffer_size_bytes = 0;
  size_t auxiliary_bytes = 0;
  size_t memory_budget_bytes = 0;
};

struct GpuAllocationResult {
  GpuBackendStatus status = GpuBackendStatus::Failed;
  std::string reason_code = "allocation-not-attempted";
  GpuErrorDiagnostic error;
  size_t requested_buffer_size_bytes = 0;
  size_t auxiliary_bytes = 0;
  size_t required_total_bytes = 0;
  size_t memory_budget_bytes = 0;
  uint64_t current_allocated_size_before = 0;
  uint64_t current_allocated_size_peak = 0;
  uint64_t current_allocated_size_after_release = 0;
  bool recommended_working_set_available = false;
  int64_t recommended_working_set_headroom_bytes = 0;
  double recommended_working_set_headroom_fraction = 0.0;
  bool exceeds_recommended_working_set = false;
  GpuResourceMetadata buffer_a;
  GpuResourceMetadata buffer_b;
  GpuResourceMetadata status_buffer;
};

/** Fully resolved request for one excluded or measured GPU attempt. */
struct GpuBackendAttemptRequest {
  GpuOperation operation = GpuOperation::Read;
  size_t buffer_size_bytes = 0;
  size_t passes = 0;
  uint64_t operation_seed = 0;
  size_t vector_count = 0;
  size_t tail_bytes = 0;
  size_t threads_per_threadgroup = 0;
  size_t threadgroups_per_grid = 0;
  size_t grid_threads = 0;
};

struct GpuBackendPhaseResult {
  GpuBackendStatus status = GpuBackendStatus::NotRun;
  std::string reason_code = "phase-not-run";
  GpuCommandStatus command_status = GpuCommandStatus::NotRun;
  GpuErrorDiagnostic error;
  size_t command_buffer_count = 0;
  size_t compute_encoder_count = 0;
  size_t dispatch_count = 0;
  size_t data_initialization_dispatch_count = 0;
  size_t benchmark_operation_dispatch_count = 0;
  size_t validation_dispatch_count = 0;
  size_t status_reset_count = 0;
};

struct GpuTimedResult : GpuBackendPhaseResult {
  double gpu_start_seconds = 0.0;
  double gpu_end_seconds = 0.0;
  double gpu_elapsed_seconds = 0.0;
  double host_wall_seconds = 0.0;
  double host_submit_seconds = 0.0;
  double host_wait_end_seconds = 0.0;
  bool queue_delay_available = false;
  double queue_delay_seconds = 0.0;
  GpuDualChecksum expected_accumulator;
  GpuDualChecksum actual_accumulator;
};

struct GpuValidationResult : GpuBackendPhaseResult {
  GpuValidationStatus validation_status = GpuValidationStatus::NotRun;
  std::string timed_accumulator_algorithm = "gpu-dual-mod32-v2";
  std::string final_checksum_algorithm = "not-applicable";
  GpuDualChecksum expected_final_checksum;
  GpuDualChecksum actual_final_checksum;
};

/**
 * @brief Objective-C-free backend boundary used by production and fake tests.
 *
 * All calls are synchronous and return only after their command buffers reach a
 * terminal state. Implementations retain suite resources until
 * `release_resources()` or destruction and must not throw across this boundary.
 * One runner owns an instance at a time; implementations are not thread-safe
 * and callers must not invoke phase or lifecycle methods concurrently.
 */
class GpuBackend {
 public:
  virtual ~GpuBackend() = default;

  virtual GpuBackendInitialization initialize() noexcept = 0;
  virtual GpuAllocationResult allocate_resources(
      const GpuAllocationRequest& request) noexcept = 0;
  virtual GpuEnvironmentSnapshot snapshot_environment() noexcept = 0;
  virtual GpuBackendPhaseResult run_warmup(
      const GpuBackendAttemptRequest& request) noexcept = 0;
  virtual GpuBackendPhaseResult run_precondition(
      const GpuBackendAttemptRequest& request) noexcept = 0;
  virtual GpuTimedResult run_timed(
      const GpuBackendAttemptRequest& request) noexcept = 0;
  virtual GpuValidationResult run_validation(
      const GpuBackendAttemptRequest& request,
      const GpuTimedResult& timed_result) noexcept = 0;

  /** Test/audit path: copy the last operation's final private buffer to CPU. */
  virtual GpuBackendPhaseResult readback_last_output(
      std::vector<uint8_t>& output) noexcept = 0;

  virtual GpuAllocationResult release_resources() noexcept = 0;
};

/** Create the private Objective-C++ Metal backend without initializing it. */
std::unique_ptr<GpuBackend> create_metal_gpu_backend();

/** Hash the canonical embedded MSL bytes without creating a Metal device. */
std::string canonical_gpu_kernel_source_sha256();

/**
 * Compute the independent CPU oracle for the versioned timed accumulator.
 *
 * This helper performs O(passes) work and does not allocate buffers or create
 * a Metal device, so large power-of-two work plans can be checked in unit
 * tests without executing a hardware-dependent benchmark.
 */
GpuDualChecksum calculate_expected_gpu_timed_accumulator(
    const GpuBackendAttemptRequest& request);

const char* gpu_backend_status_to_string(GpuBackendStatus status);
const char* gpu_command_status_to_string(GpuCommandStatus status);
const char* gpu_validation_status_to_string(GpuValidationStatus status);

#endif  // GPU_BACKEND_H

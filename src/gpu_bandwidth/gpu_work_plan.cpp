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
 * @file gpu_work_plan.cpp
 * @brief Pure deterministic GPU work-plan implementation
 */

#include "gpu_bandwidth/gpu_work_plan.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "utils/cyclic_order.h"
#include "utils/numeric_utils.h"
#include "utils/seed_utils.h"

namespace {

constexpr uint64_t kGpuReadSeedDomain = 0x4750555f52454144ULL;   // GPU_READ
constexpr uint64_t kGpuWriteSeedDomain = 0x4750555752495445ULL;  // GPUWRITE
constexpr uint64_t kGpuCopySeedDomain = 0x4750555f434f5059ULL;   // GPU_COPY

bool is_valid_operation(GpuOperation operation) {
  switch (operation) {
    case GpuOperation::Read:
    case GpuOperation::Write:
    case GpuOperation::Copy:
      return true;
  }
  return false;
}

size_t operation_payload_multiplier(GpuOperation operation) {
  if (!is_valid_operation(operation)) {
    return 0;
  }
  return operation == GpuOperation::Copy
             ? static_cast<size_t>(Constants::COPY_OPERATION_MULTIPLIER)
             : 1;
}

const char* exceeded_cap_reason(const GpuWorkPlanRequest& request,
                                const GpuPassLimits& limits) {
  const bool payload_is_stricter =
      limits.maximum_passes_by_payload <
      limits.maximum_passes_by_dispatch;
  if (payload_is_stricter &&
      request.passes > limits.maximum_passes_by_payload) {
    return request.explicit_iterations
               ? GpuWorkPlanReason::EXPLICIT_PAYLOAD_CAP_EXCEEDED
               : GpuWorkPlanReason::PAYLOAD_CAP_EXCEEDED;
  }
  if (request.passes > limits.maximum_passes_by_dispatch) {
    return request.explicit_iterations
               ? GpuWorkPlanReason::EXPLICIT_DISPATCH_CAP_EXCEEDED
               : GpuWorkPlanReason::DISPATCH_CAP_EXCEEDED;
  }
  return request.explicit_iterations
             ? GpuWorkPlanReason::EXPLICIT_PAYLOAD_CAP_EXCEEDED
             : GpuWorkPlanReason::PAYLOAD_CAP_EXCEEDED;
}

void append_identity_field(std::string& identity, const char* name,
                           const std::string& value) {
  identity += '|';
  identity += name;
  identity += '=';
  identity += value;
}

void append_identity_field(std::string& identity, const char* name,
                           const char* value) {
  append_identity_field(identity, name, std::string(value));
}

template <typename Integer>
void append_identity_field(std::string& identity, const char* name,
                           Integer value) {
  append_identity_field(identity, name, std::to_string(value));
}

}  // namespace

const char* gpu_operation_to_string(GpuOperation operation) {
  switch (operation) {
    case GpuOperation::Read:
      return "read";
    case GpuOperation::Write:
      return "write";
    case GpuOperation::Copy:
      return "copy";
  }
  return "unknown";
}

uint64_t gpu_operation_seed_domain(GpuOperation operation) {
  switch (operation) {
    case GpuOperation::Read:
      return kGpuReadSeedDomain;
    case GpuOperation::Write:
      return kGpuWriteSeedDomain;
    case GpuOperation::Copy:
      return kGpuCopySeedDomain;
  }
  return 0;
}

uint64_t derive_gpu_operation_seed(uint64_t base_seed,
                                   GpuOperation operation) {
  const uint64_t domain = gpu_operation_seed_domain(operation);
  return domain == 0 ? 0 : SeedUtils::splitmix64(base_seed ^ domain);
}

std::array<GpuOperation, kGpuOperationCount> build_gpu_operation_order(
    size_t loop_index) {
  constexpr std::array<GpuOperation, kGpuOperationCount> kBaseOrder = {
      GpuOperation::Read, GpuOperation::Write, GpuOperation::Copy};
  std::array<GpuOperation, kGpuOperationCount> order{};
  const std::vector<size_t> indexes =
      build_cyclic_order(kGpuOperationCount, loop_index);
  for (size_t position = 0; position < indexes.size(); ++position) {
    order[position] = kBaseOrder[indexes[position]];
  }
  return order;
}

GpuPassLimits calculate_gpu_pass_limits(size_t buffer_size_bytes,
                                        GpuOperation operation) {
  GpuPassLimits limits;
  if (!is_valid_operation(operation)) {
    limits.reason_code = GpuWorkPlanReason::INVALID_OPERATION;
    return limits;
  }
  if (buffer_size_bytes == 0) {
    limits.reason_code = GpuWorkPlanReason::BUFFER_SIZE_ZERO;
    return limits;
  }

  limits.payload_multiplier = operation_payload_multiplier(operation);
  if (!NumericUtils::checked_multiply(buffer_size_bytes,
                                      limits.payload_multiplier,
                                      limits.bytes_per_pass)) {
    limits.reason_code = GpuWorkPlanReason::PAYLOAD_PER_PASS_OVERFLOW;
    return limits;
  }
  limits.maximum_passes_by_payload =
      Constants::GPU_MAX_EXACT_PAYLOAD_BYTES / limits.bytes_per_pass;
  limits.effective_maximum_passes =
      std::min(limits.maximum_passes_by_dispatch,
               limits.maximum_passes_by_payload);
  limits.dispatch_cap_is_limiting =
      limits.maximum_passes_by_dispatch <=
      limits.maximum_passes_by_payload;
  limits.payload_cap_is_limiting =
      limits.maximum_passes_by_payload <=
      limits.maximum_passes_by_dispatch;
  if (limits.effective_maximum_passes == 0) {
    limits.reason_code = GpuWorkPlanReason::PAYLOAD_CAP_BELOW_ONE_PASS;
    return limits;
  }

  limits.valid = true;
  return limits;
}

size_t calculate_gpu_pilot_passes(const GpuPassLimits& limits) {
  if (!limits.valid) {
    return 0;
  }
  return NumericUtils::calculate_minimum_pilot_count(
      limits.bytes_per_pass, Constants::GPU_CALIBRATION_MIN_PILOT_BYTES,
      limits.effective_maximum_passes);
}

size_t calculate_gpu_calibrated_passes(double attempt_duration_seconds,
                                       size_t attempt_passes,
                                       const GpuPassLimits& limits) {
  if (!limits.valid) {
    return 0;
  }
  return NumericUtils::calculate_duration_scaled_count(
      attempt_duration_seconds, attempt_passes,
      Constants::GPU_CALIBRATION_TARGET_SECONDS, 1,
      limits.effective_maximum_passes);
}

bool gpu_duration_in_target_window(double elapsed_seconds) {
  return std::isfinite(elapsed_seconds) &&
         elapsed_seconds >= Constants::GPU_CALIBRATION_MIN_SECONDS &&
         elapsed_seconds <= Constants::GPU_CALIBRATION_MAX_SECONDS;
}

std::string classify_gpu_duration_quality(double elapsed_seconds,
                                          size_t passes,
                                          const GpuPassLimits& limits) {
  if (!limits.valid || passes == 0 ||
      passes > limits.effective_maximum_passes) {
    return "invalid-work-plan";
  }
  if (!std::isfinite(elapsed_seconds) || elapsed_seconds <= 0.0) {
    return "invalid-duration";
  }
  if (gpu_duration_in_target_window(elapsed_seconds)) {
    return "within-target-window";
  }
  if (elapsed_seconds < Constants::GPU_CALIBRATION_MIN_SECONDS) {
    if (passes == limits.effective_maximum_passes) {
      return limits.dispatch_cap_is_limiting
                 ? "dispatch-cap-below-target"
                 : "payload-cap-below-target";
    }
    return "below-target-window";
  }
  if (passes == 1) {
    return "single-pass-exceeds-window";
  }
  return "above-target-window";
}

GpuWorkPlan build_gpu_work_plan(const GpuWorkPlanRequest& request) {
  GpuWorkPlan plan;
  plan.operation = request.operation;
  plan.requested_buffer_bytes = request.requested_buffer_bytes;
  plan.effective_buffer_bytes = request.requested_buffer_bytes;
  plan.base_seed = request.base_seed;
  plan.thread_execution_width = request.thread_execution_width;
  plan.max_total_threads_per_threadgroup =
      request.max_total_threads_per_threadgroup;
  plan.maximum_threadgroups_per_grid =
      request.maximum_threadgroups_per_grid;
  plan.data_resource_options = request.data_resource_options;
  plan.status_resource_options = request.status_resource_options;
  plan.data_storage_mode = request.data_storage_mode;
  plan.data_hazard_tracking_mode = request.data_hazard_tracking_mode;
  plan.status_storage_mode = request.status_storage_mode;
  plan.status_hazard_tracking_mode = request.status_hazard_tracking_mode;
  plan.kernel_revision = request.kernel_revision;
  plan.kernel_source_sha256 = request.kernel_source_sha256;
  plan.msl_language_version = request.msl_language_version;
  plan.methodology_version = Constants::GPU_METHODOLOGY_VERSION;
  plan.timing_policy = "one-command-buffer-one-serial-encoder";
  plan.pass_mapping = "one-full-buffer-pass-per-dispatch";
  plan.warmup_policy = "excluded-full-buffer-warmup-before-each-attempt";
  plan.calibration_policy = request.explicit_iterations
                                ? "explicit-fixed-work"
                                : "automatic-150ms-100-250ms-max-two-corrections";
  plan.duration_quality_policy = "report-without-performance-retry-after-freeze";
  plan.timed_accumulator_algorithm = "gpu-dual-mod32-v2";
  plan.final_checksum_algorithm =
      request.operation == GpuOperation::Read ? "not-applicable"
                                              : "gpu-dual-mod32-v1";

  const GpuPassLimits limits = calculate_gpu_pass_limits(
      request.requested_buffer_bytes, request.operation);
  plan.payload_multiplier = limits.payload_multiplier;
  plan.bytes_per_pass = limits.bytes_per_pass;
  plan.maximum_passes_by_dispatch = limits.maximum_passes_by_dispatch;
  plan.maximum_passes_by_payload = limits.maximum_passes_by_payload;
  plan.effective_maximum_passes = limits.effective_maximum_passes;
  plan.dispatch_cap_is_limiting = limits.dispatch_cap_is_limiting;
  plan.payload_cap_is_limiting = limits.payload_cap_is_limiting;
  if (!limits.valid) {
    plan.reason_code = limits.reason_code;
    return plan;
  }
  plan.operation_seed =
      derive_gpu_operation_seed(request.base_seed, request.operation);

  if (request.passes == 0) {
    plan.reason_code = GpuWorkPlanReason::PASS_COUNT_ZERO;
    return plan;
  }
  if (request.passes > limits.effective_maximum_passes) {
    plan.reason_code = exceeded_cap_reason(request, limits);
    return plan;
  }
  if (request.thread_execution_width == 0) {
    plan.reason_code = GpuWorkPlanReason::THREAD_EXECUTION_WIDTH_ZERO;
    return plan;
  }
  if (request.max_total_threads_per_threadgroup == 0) {
    plan.reason_code =
        GpuWorkPlanReason::MAX_THREADS_PER_THREADGROUP_ZERO;
    return plan;
  }
  if (request.maximum_threadgroups_per_grid == 0) {
    plan.reason_code = GpuWorkPlanReason::THREADGROUP_LIMIT_ZERO;
    return plan;
  }

  const size_t thread_cap =
      std::min(Constants::GPU_THREADS_PER_THREADGROUP_CAP,
               request.max_total_threads_per_threadgroup);
  plan.threads_per_threadgroup =
      (thread_cap / request.thread_execution_width) *
      request.thread_execution_width;
  if (plan.threads_per_threadgroup == 0) {
    plan.reason_code =
        GpuWorkPlanReason::THREAD_EXECUTION_WIDTH_EXCEEDS_CAP;
    return plan;
  }

  plan.vector_count =
      request.requested_buffer_bytes / Constants::GPU_VECTOR_WIDTH_BYTES;
  plan.tail_bytes =
      request.requested_buffer_bytes % Constants::GPU_VECTOR_WIDTH_BYTES;
  plan.required_threadgroups_per_grid =
      plan.vector_count / plan.threads_per_threadgroup +
      (plan.vector_count % plan.threads_per_threadgroup != 0 ? 1 : 0);
  plan.threadgroups_per_grid = std::max<size_t>(
      1, std::min(plan.required_threadgroups_per_grid,
                  request.maximum_threadgroups_per_grid));
  if (!NumericUtils::checked_multiply(plan.threadgroups_per_grid,
                                      plan.threads_per_threadgroup,
                                      plan.grid_threads)) {
    plan.reason_code = GpuWorkPlanReason::GRID_THREAD_COUNT_OVERFLOW;
    return plan;
  }
  if (!NumericUtils::checked_multiply(limits.bytes_per_pass,
                                      request.passes,
                                      plan.exact_payload_bytes)) {
    plan.reason_code = GpuWorkPlanReason::EXACT_PAYLOAD_OVERFLOW;
    return plan;
  }

  plan.passes = request.passes;
  plan.dispatch_count = request.passes;
  plan.measured_command_buffer_count = 1;
  plan.measured_compute_encoder_count = 1;
  plan.valid = true;
  plan.plan_identity = build_gpu_plan_identity(plan);
  return plan;
}

std::string build_gpu_plan_identity(const GpuWorkPlan& plan) {
  if (!plan.valid) {
    return "";
  }

  std::string identity = Constants::GPU_WORK_PLAN_IDENTITY_VERSION;
  append_identity_field(identity, "operation",
                        gpu_operation_to_string(plan.operation));
  append_identity_field(identity, "requested_buffer_bytes",
                        plan.requested_buffer_bytes);
  append_identity_field(identity, "effective_buffer_bytes",
                        plan.effective_buffer_bytes);
  append_identity_field(identity, "passes", plan.passes);
  append_identity_field(identity, "bytes_per_pass", plan.bytes_per_pass);
  append_identity_field(identity, "exact_payload_bytes",
                        plan.exact_payload_bytes);
  append_identity_field(identity, "base_seed", plan.base_seed);
  append_identity_field(identity, "operation_seed", plan.operation_seed);
  append_identity_field(identity, "vector_width_bytes",
                        plan.vector_width_bytes);
  append_identity_field(identity, "vector_count", plan.vector_count);
  append_identity_field(identity, "tail_bytes", plan.tail_bytes);
  append_identity_field(identity, "thread_execution_width",
                        plan.thread_execution_width);
  append_identity_field(identity, "max_total_threads_per_threadgroup",
                        plan.max_total_threads_per_threadgroup);
  append_identity_field(identity, "threads_per_threadgroup",
                        plan.threads_per_threadgroup);
  append_identity_field(identity, "maximum_threadgroups_per_grid",
                        plan.maximum_threadgroups_per_grid);
  append_identity_field(identity, "threadgroups_per_grid",
                        plan.threadgroups_per_grid);
  append_identity_field(identity, "grid_threads", plan.grid_threads);
  append_identity_field(identity, "dispatch_count", plan.dispatch_count);
  append_identity_field(identity, "data_resource_options",
                        plan.data_resource_options);
  append_identity_field(identity, "status_resource_options",
                        plan.status_resource_options);
  append_identity_field(identity, "data_storage_mode",
                        plan.data_storage_mode);
  append_identity_field(identity, "data_hazard_tracking_mode",
                        plan.data_hazard_tracking_mode);
  append_identity_field(identity, "status_storage_mode",
                        plan.status_storage_mode);
  append_identity_field(identity, "status_hazard_tracking_mode",
                        plan.status_hazard_tracking_mode);
  append_identity_field(identity, "kernel_revision", plan.kernel_revision);
  append_identity_field(identity, "kernel_source_sha256",
                        plan.kernel_source_sha256);
  append_identity_field(identity, "msl_language_version",
                        plan.msl_language_version);
  append_identity_field(identity, "methodology_version",
                        plan.methodology_version);
  append_identity_field(identity, "timing_policy", plan.timing_policy);
  append_identity_field(identity, "pass_mapping", plan.pass_mapping);
  append_identity_field(identity, "warmup_policy", plan.warmup_policy);
  append_identity_field(identity, "calibration_policy",
                        plan.calibration_policy);
  append_identity_field(identity, "duration_quality_policy",
                        plan.duration_quality_policy);
  append_identity_field(identity, "timed_accumulator_algorithm",
                        plan.timed_accumulator_algorithm);
  append_identity_field(identity, "final_checksum_algorithm",
                        plan.final_checksum_algorithm);
  return identity;
}

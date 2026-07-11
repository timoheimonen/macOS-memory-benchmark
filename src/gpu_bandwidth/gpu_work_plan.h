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
 * @file gpu_work_plan.h
 * @brief Overflow-safe deterministic work planning for GPU bandwidth
 */

#ifndef GPU_WORK_PLAN_H
#define GPU_WORK_PLAN_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include "core/config/constants.h"

enum class GpuOperation {
  Read = 0,
  Write,
  Copy,
};

constexpr size_t kGpuOperationCount = 3;

/** Stable semantic reason codes emitted by the pure planner. */
namespace GpuWorkPlanReason {
inline constexpr const char* INVALID_OPERATION = "invalid-operation";
inline constexpr const char* BUFFER_SIZE_ZERO = "buffer-size-zero";
inline constexpr const char* PAYLOAD_PER_PASS_OVERFLOW =
    "payload-per-pass-overflow";
inline constexpr const char* PAYLOAD_CAP_BELOW_ONE_PASS =
    "payload-cap-below-one-pass";
inline constexpr const char* PASS_COUNT_ZERO = "pass-count-zero";
inline constexpr const char* DISPATCH_CAP_EXCEEDED =
    "dispatch-cap-exceeded";
inline constexpr const char* PAYLOAD_CAP_EXCEEDED =
    "payload-cap-exceeded";
inline constexpr const char* EXPLICIT_DISPATCH_CAP_EXCEEDED =
    "explicit-dispatch-cap-exceeded";
inline constexpr const char* EXPLICIT_PAYLOAD_CAP_EXCEEDED =
    "explicit-payload-cap-exceeded";
inline constexpr const char* THREAD_EXECUTION_WIDTH_ZERO =
    "thread-execution-width-zero";
inline constexpr const char* MAX_THREADS_PER_THREADGROUP_ZERO =
    "max-threads-per-threadgroup-zero";
inline constexpr const char* THREAD_EXECUTION_WIDTH_EXCEEDS_CAP =
    "thread-execution-width-exceeds-cap";
inline constexpr const char* THREADGROUP_LIMIT_ZERO =
    "threadgroup-limit-zero";
inline constexpr const char* GRID_THREAD_COUNT_OVERFLOW =
    "grid-thread-count-overflow";
inline constexpr const char* EXACT_PAYLOAD_OVERFLOW =
    "exact-payload-overflow";
}  // namespace GpuWorkPlanReason

/** Payload- and dispatch-derived pass guardrails for one operation. */
struct GpuPassLimits {
  bool valid = false;
  std::string reason_code;
  size_t payload_multiplier = 0;
  size_t bytes_per_pass = 0;
  size_t maximum_passes_by_dispatch =
      Constants::GPU_MAX_DISPATCHES_PER_MEASUREMENT;
  size_t maximum_passes_by_payload = 0;
  size_t effective_maximum_passes = 0;
  bool dispatch_cap_is_limiting = false;
  bool payload_cap_is_limiting = false;
};

/** Inputs needed to resolve one immutable excluded or measured work plan. */
struct GpuWorkPlanRequest {
  GpuOperation operation = GpuOperation::Read;
  size_t requested_buffer_bytes = 0;
  size_t passes = 0;
  uint64_t base_seed = 0;
  size_t thread_execution_width = 0;
  size_t max_total_threads_per_threadgroup = 0;
  bool explicit_iterations = false;
  // Production retains the methodology default; validation may compare the
  // frozen 2048/4096/8192 candidate set through this cold-path input.
  size_t maximum_threadgroups_per_grid =
      Constants::GPU_MAX_THREADGROUPS_PER_GRID;
  uint64_t data_resource_options = 0;
  uint64_t status_resource_options = 0;
  std::string data_storage_mode = "private";
  std::string data_hazard_tracking_mode = "tracked";
  std::string status_storage_mode = "shared";
  std::string status_hazard_tracking_mode = "tracked";
  std::string kernel_revision;
  std::string kernel_source_sha256;
  std::string msl_language_version = "2.3";
};

/**
 * @brief Fully resolved frozen GPU work and exact accounting.
 *
 * A valid plan always describes one measured command buffer and one serial
 * compute encoder. Each pass is one full-buffer dispatch. Metal objects remain
 * backend-owned; the planner receives and freezes the actual resource-option
 * values and normalized storage/hazard modes for provenance and identity.
 */
struct GpuWorkPlan {
  bool valid = false;
  std::string reason_code;
  GpuOperation operation = GpuOperation::Read;
  size_t requested_buffer_bytes = 0;
  size_t effective_buffer_bytes = 0;
  uint64_t base_seed = 0;
  uint64_t operation_seed = 0;

  size_t passes = 0;
  size_t payload_multiplier = 0;
  size_t bytes_per_pass = 0;
  size_t exact_payload_bytes = 0;
  size_t maximum_passes_by_dispatch = 0;
  size_t maximum_passes_by_payload = 0;
  size_t effective_maximum_passes = 0;
  bool dispatch_cap_is_limiting = false;
  bool payload_cap_is_limiting = false;

  size_t vector_width_bytes = Constants::GPU_VECTOR_WIDTH_BYTES;
  size_t vector_count = 0;
  size_t tail_bytes = 0;
  size_t thread_execution_width = 0;
  size_t max_total_threads_per_threadgroup = 0;
  size_t threads_per_threadgroup = 0;
  size_t required_threadgroups_per_grid = 0;
  size_t maximum_threadgroups_per_grid = 0;
  size_t threadgroups_per_grid = 0;
  size_t grid_threads = 0;

  size_t dispatch_count = 0;
  size_t measured_command_buffer_count = 0;
  size_t measured_compute_encoder_count = 0;

  uint64_t data_resource_options = 0;
  uint64_t status_resource_options = 0;
  std::string data_storage_mode;
  std::string data_hazard_tracking_mode;
  std::string status_storage_mode;
  std::string status_hazard_tracking_mode;
  std::string kernel_revision;
  std::string kernel_source_sha256;
  std::string msl_language_version;
  std::string methodology_version;
  std::string timing_policy;
  std::string pass_mapping;
  std::string warmup_policy;
  std::string calibration_policy;
  std::string duration_quality_policy;
  std::string timed_accumulator_algorithm;
  std::string final_checksum_algorithm;
  std::string plan_identity;
};

/** Return the stable schema token for an operation, or `unknown`. */
const char* gpu_operation_to_string(GpuOperation operation);

/** Return the versioned 64-bit seed domain for an operation, or zero. */
uint64_t gpu_operation_seed_domain(GpuOperation operation);

/** Derive the operation seed as SplitMix64(base seed XOR operation domain). */
uint64_t derive_gpu_operation_seed(uint64_t base_seed,
                                   GpuOperation operation);

/** Build read/write/copy order rotated once per loop. */
std::array<GpuOperation, kGpuOperationCount> build_gpu_operation_order(
    size_t loop_index);

/**
 * @brief Resolve exact bytes-per-pass and dispatch/payload pass ceilings.
 * @return Invalid limits with a stable reason when one pass is impossible.
 */
GpuPassLimits calculate_gpu_pass_limits(size_t buffer_size_bytes,
                                        GpuOperation operation);

/** Select the smallest 8 MiB-floor pilot count within effective guardrails. */
size_t calculate_gpu_pilot_passes(const GpuPassLimits& limits);

/** Scale a valid excluded attempt toward 150 ms within effective guardrails. */
size_t calculate_gpu_calibrated_passes(double attempt_duration_seconds,
                                       size_t attempt_passes,
                                       const GpuPassLimits& limits);

/** Return true for a finite duration in the inclusive 100-250 ms window. */
bool gpu_duration_in_target_window(double elapsed_seconds);

/** Classify an automatic attempt, including guardrail-limited short work. */
std::string classify_gpu_duration_quality(double elapsed_seconds,
                                          size_t passes,
                                          const GpuPassLimits& limits);

/**
 * @brief Resolve exact payload, grid geometry, seed, and frozen identity.
 * @return A valid plan, or an invalid plan carrying a stable reason code.
 * @note The requested buffer is never silently reduced.
 */
GpuWorkPlan build_gpu_work_plan(const GpuWorkPlanRequest& request);

/** Return the canonical stable identity of a valid frozen work plan. */
std::string build_gpu_plan_identity(const GpuWorkPlan& plan);

#endif  // GPU_WORK_PLAN_H

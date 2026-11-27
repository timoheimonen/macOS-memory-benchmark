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
 * @file cpu_bandwidth_executor.h
 * @brief Persistent buffer preparation, synchronized ARM64 worker gate, and cold checks
 */

#ifndef CPU_BANDWIDTH_EXECUTOR_H
#define CPU_BANDWIDTH_EXECUTOR_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "core/memory/memory_manager.h"
#include "core/timing/timer.h"
#include "cpu_bandwidth/cpu_bandwidth_oracle.h"
#include "cpu_bandwidth/cpu_bandwidth_work_plan.h"

namespace CpuBandwidthExecutorReason {
inline constexpr const char* VALID = "valid";
inline constexpr const char* INVALID_RESOURCES = "invalid-resources";
inline constexpr const char* BUFFER_ALLOCATION_FAILED = "buffer-allocation-failed";
inline constexpr const char* SRC_READONLY_FAILED = "src-readonly-protection-failed";
inline constexpr const char* PREPARATION_INTERRUPTED = "preparation-interrupted";
inline constexpr const char* INVALID_OPERATION_PLAN = "invalid-operation-plan";
inline constexpr const char* WORKER_STARTUP_FAILED = "worker-startup-failed";
inline constexpr const char* KERNEL_FAILED = "kernel-failed";
inline constexpr const char* INVALID_ELAPSED_TIME = "invalid-elapsed-time";
inline constexpr const char* CHECKSUM_MISMATCH = "checksum-mismatch";
inline constexpr const char* WRITE_POST_VALIDATION_FAILED =
    "write-post-validation-failed";
inline constexpr const char* COPY_POST_VALIDATION_FAILED =
    "copy-post-validation-failed";
inline constexpr const char* STRUCTURE_VALIDATION_FAILED =
    "post-validation-structure-failed";
}  // namespace CpuBandwidthExecutorReason

/** Ordered cold validation check entry (`execution.validation.checks[]`). */
struct CpuBandwidthValidationCheck {
  std::string kind;
  bool applicable = false;
  std::optional<bool> evaluated;
  std::optional<bool> valid;
  std::string reason_code = "not-applicable";
};

/** Persistent command-lifetime src/dst buffers for one target. */
struct CpuBandwidthTargetBuffers {
  CpuBandwidthTarget target = CpuBandwidthTarget::MainMemory;
  bool active = false;
  bool valid = false;
  AllocatedBufferEvidence src;
  AllocatedBufferEvidence dst;
};

/** Persistent buffer set across all active targets. */
struct CpuBandwidthResourceSet {
  bool valid = false;
  std::string reason_code = CpuBandwidthExecutorReason::INVALID_RESOURCES;
  std::array<CpuBandwidthTargetBuffers, kCpuBandwidthMaxTargets> targets;
};

/** Result of executing one operation attempt (calibration or measured). */
struct CpuBandwidthOperationExecution {
  bool available = false;
  bool valid = false;
  std::string reason_code = CpuBandwidthExecutorReason::INVALID_RESOURCES;
  size_t requested_workers = 0;
  size_t created_workers = 0;
  size_t completed_workers = 0;
  size_t qos_successful_workers = 0;
  size_t qos_failed_workers = 0;
  bool worker_startup_failed = false;
  bool kernel_succeeded = false;
  bool timer_started = false;
  bool timer_stopped = false;
  bool timing_evaluated = false;
  bool timing_valid = false;
  double elapsed_seconds = 0.0;
  std::optional<MachTimingSnapshot> cpu_raw;
  std::optional<uint64_t> write_pattern;
  std::optional<uint64_t> attempt_seed;
  bool checksum_applicable = false;
  bool checksum_evaluated = false;
  bool checksum_valid = false;
  std::vector<CpuBandwidthExpectedWorkerChecksum> actual_worker_checksums;
  CpuBandwidthRunChecksum actual_run_checksum{};
  std::array<CpuBandwidthValidationCheck, 5> validation_checks;
};

/** Deterministic test seams for executor fault/corruption injection. */
struct CpuBandwidthExecutorTestControl {
  int fail_before_worker_index = -1;
  int (*set_worker_qos)(void* context, size_t worker_index) noexcept = nullptr;
  void* qos_context = nullptr;
  bool force_kernel_fail = false;
  bool force_zero_elapsed = false;
  std::function<void(void* dst, size_t bytes)> corrupt_dst_after_kernel;
};

/** Build the 5 default not-evaluated validation check entries for an operation. */
std::array<CpuBandwidthValidationCheck, 5> make_default_cpu_bandwidth_checks(
    CpuBandwidthOperation op);

/**
 * Allocate persistent command-lifetime buffers, initialize deterministic `src`
 * GF(2) patterns, and enforce `mprotect(PROT_READ)` on every `src` buffer.
 */
bool prepare_cpu_bandwidth_resources(
    const CpuBandwidthWorkPlan& plan,
    bool non_cacheable,
    CpuBandwidthResourceSet& resources,
    const std::function<bool()>& stop_requested = {});

/** Release all allocated buffers cleanly. */
void release_cpu_bandwidth_resources(CpuBandwidthResourceSet& resources) noexcept;

/**
 * Execute one operation (`read`, `write`, `copy`) with an all-or-cancel
 * synchronized worker start gate, Mach raw timing, and cold post-validation.
 */
CpuBandwidthOperationExecution execute_cpu_bandwidth_operation(
    const CpuBandwidthConfig& config,
    const CpuBandwidthTargetPlan& target_plan,
    const CpuBandwidthOperationPlan& op_plan,
    const CpuBandwidthTargetBuffers& buffers,
    HighResTimer& timer,
    std::optional<uint64_t> write_attempt_seed,
    std::optional<uint64_t> write_pattern,
    uint64_t pre_copy_sentinel_pattern,
    const CpuBandwidthExecutorTestControl* test_control = nullptr) noexcept;

#endif  // CPU_BANDWIDTH_EXECUTOR_H

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
 * @file cpu_latency_executor.h
 * @brief Persistent chain buffers, ARM64 chase execution, sample windows, and checks
 */

#ifndef CPU_LATENCY_EXECUTOR_H
#define CPU_LATENCY_EXECUTOR_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "core/memory/memory_manager.h"
#include "core/timing/timer.h"
#include "cpu_latency/cpu_latency_chain.h"
#include "cpu_latency/cpu_latency_work_plan.h"

namespace CpuLatencyExecutorReason {
inline constexpr const char* VALID = "valid";
inline constexpr const char* INVALID_RESOURCES = "invalid-resources";
inline constexpr const char* BUFFER_ALLOCATION_FAILED =
    "buffer-allocation-failed";
inline constexpr const char* CHAIN_GENERATION_FAILED =
    "chain-generation-failed";
inline constexpr const char* CHAIN_HAMILTONIAN_FAILED =
    "chain-hamiltonian-prevalidation-failed";
inline constexpr const char* CHAIN_READONLY_FAILED =
    "chain-readonly-protection-failed";
inline constexpr const char* PREPARATION_INTERRUPTED =
    "preparation-interrupted";
inline constexpr const char* INVALID_EXECUTION_PLAN = "invalid-execution-plan";
inline constexpr const char* KERNEL_FAILED = "kernel-failed";
inline constexpr const char* INVALID_ELAPSED_TIME = "invalid-elapsed-time";
inline constexpr const char* STRUCTURE_VALIDATION_FAILED =
    "post-validation-structure-failed";
inline constexpr const char* MAIN_FINAL_POINTER_FAILED =
    "main-final-pointer-failed";
inline constexpr const char* SAMPLE_WINDOW_END_POINTER_FAILED =
    "sample-window-end-pointers-failed";
}  // namespace CpuLatencyExecutorReason

/** Ordered validation check entry (`execution.validation.checks[]`). */
struct CpuLatencyValidationCheck {
  std::string kind;
  bool applicable = false;
  std::optional<bool> evaluated;
  std::optional<bool> valid;
  std::string reason_code = "not-applicable";
};

/** Serialized chain metadata under `resolved_plan.chain_identities.<key>`. */
struct CpuLatencyChainIdentity {
  bool active = false;
  uint64_t chain_seed = 0;
  size_t node_count = 0;
  size_t stride_bytes = 0;
  LatencyChainMode chain_mode = LatencyChainMode::GlobalRandom;
  size_t tlb_locality_kb = 0;
  std::string permutation_sha256;
};

/** Persistent command-lifetime chain buffer for one target. */
struct CpuLatencyTargetBuffer {
  CpuLatencyTarget target = CpuLatencyTarget::MainMemory;
  bool active = false;
  bool valid = false;
  bool hamiltonian_prevalidated = false;
  AllocatedBufferEvidence buffer;
  CpuLatencyChainIdentity chain_identity;
  std::vector<uint64_t> indices;
};

/** Persistent buffer set across all active targets + optional locality compare. */
struct CpuLatencyResourceSet {
  bool valid = false;
  std::string reason_code = CpuLatencyExecutorReason::INVALID_RESOURCES;
  std::array<CpuLatencyTargetBuffer, kCpuLatencyMaxTargets> targets;
  bool compare_locality_active = false;
  AllocatedBufferEvidence compare_buffer;
  CpuLatencyChainIdentity locality_16k_identity;
  CpuLatencyChainIdentity locality_global_identity;
  std::vector<uint64_t> locality_16k_indices;
  std::vector<uint64_t> locality_global_indices;
};

/** End-pointer check result for the main continuous chase. */
struct CpuLatencyFinalPointerVerification {
  size_t expected_node = 0;
  std::optional<size_t> actual_node;
  std::optional<bool> valid;
};

/** Separate-pass sample windows with per-window Mach ticks and end-node checks. */
struct CpuLatencySampleWindows {
  uint32_t timebase_numer = 0;
  uint32_t timebase_denom = 0;
  size_t sample_count = 0;
  std::vector<uint64_t> sample_delta_ticks;
  std::vector<double> samples_ns;
  bool validation_valid = false;
  bool all_end_pointers_valid = false;
  std::vector<std::optional<size_t>> end_node_indices;
};

/** Result of executing one target attempt (calibration or measured). */
struct CpuLatencyAttemptExecution {
  bool available = false;
  bool valid = false;
  std::string reason_code = CpuLatencyExecutorReason::INVALID_RESOURCES;
  bool kernel_succeeded = false;
  bool timer_started = false;
  bool timer_stopped = false;
  bool timing_evaluated = false;
  bool timing_valid = false;
  double elapsed_seconds = 0.0;
  std::optional<MachTimingSnapshot> cpu_raw;
  CpuLatencyFinalPointerVerification final_pointer;
  std::optional<CpuLatencySampleWindows> sample_windows;
  std::array<CpuLatencyValidationCheck, 5> validation_checks;
};

/** Result of executing one `--compare-locality` subrun (`locality_16k` or `locality_global`). */
struct CpuLatencyLocalitySubrunResult {
  std::string status = "not_run";
  bool attempted = false;
  size_t completed_accesses = 0;
  std::optional<double> latency_ns;
  CpuLatencyFinalPointerVerification final_pointer;
  bool timing_evaluated = false;
  bool timing_valid = false;
  double elapsed_seconds = 0.0;
  std::optional<MachTimingSnapshot> cpu_raw;
  std::string reason_code = "not-run";
};

/** Deterministic test seams for executor fault/corruption injection. */
struct CpuLatencyExecutorTestControl {
  bool force_kernel_fail = false;
  bool force_zero_elapsed = false;
  bool force_hamiltonian_fail = false;
  std::function<uintptr_t*(uintptr_t*, size_t)> chase_override;
  std::function<bool(void*, size_t, int)> mprotect_override;
};

/** Build the 5 default not-evaluated validation check entries. */
std::array<CpuLatencyValidationCheck, 5> make_default_cpu_latency_checks(
    bool sample_windows_applicable);

/**
 * Allocate persistent command-lifetime buffers, build canonicalized SplitMix64
 * chains, prevalidate single Hamiltonian cycles, and enforce `mprotect(PROT_READ)`.
 */
bool prepare_cpu_latency_resources(
    const CpuLatencyWorkPlan& plan,
    bool non_cacheable,
    CpuLatencyResourceSet& resources,
    const std::function<bool()>& stop_requested = {});

/** Release all allocated buffers cleanly. */
void release_cpu_latency_resources(CpuLatencyResourceSet& resources) noexcept;

/**
 * Execute one latency measurement attempt (continuous main pass + optional
 * separate sample-window pass) with raw Mach timing and end-pointer checks.
 */
CpuLatencyAttemptExecution execute_cpu_latency_attempt(
    const CpuLatencyTargetPlan& target_plan,
    const CpuLatencyTargetExecutionPlan& exec_plan,
    const CpuLatencyTargetBuffer& buffer,
    HighResTimer& timer,
    size_t requested_samples,
    const CpuLatencyExecutorTestControl* test_control = nullptr) noexcept;

/**
 * Populate `resources.compare_buffer` for one `--compare-locality` subrun,
 * prevalidate Hamiltonian cycle, protect `PROT_READ`, and run timed chase.
 */
CpuLatencyLocalitySubrunResult execute_cpu_latency_locality_subrun(
    const CpuLatencyTargetPlan& sub_plan,
    const CpuLatencyChainIdentity& identity,
    const std::vector<uint64_t>& indices,
    size_t complete_chain_cycles,
    CpuLatencyResourceSet& resources,
    HighResTimer& timer,
    const CpuLatencyExecutorTestControl* test_control = nullptr) noexcept;

#endif  // CPU_LATENCY_EXECUTOR_H

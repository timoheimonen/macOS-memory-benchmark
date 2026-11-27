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
 * @file cpu_bandwidth_work_plan.h
 * @brief Deterministic target geometry, 64B worker spans, seeds, and budget preflight
 */

#ifndef CPU_BANDWIDTH_WORK_PLAN_H
#define CPU_BANDWIDTH_WORK_PLAN_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "cpu_bandwidth/cpu_bandwidth.h"

/** Canonical bandwidth targets in schema v1 order. */
enum class CpuBandwidthTarget : uint8_t {
  MainMemory = 0,
  L1,
  L2,
  Custom,
};

inline constexpr size_t kCpuBandwidthMaxTargets = 4;

/** Canonical bandwidth operations in schema v1 order. */
enum class CpuBandwidthOperation : uint8_t {
  Read = 0,
  Write,
  Copy,
};

inline constexpr size_t kCpuBandwidthOperationCount = 3;

inline constexpr int kCpuBandwidthJsonSchemaVersion = 1;
inline constexpr const char* kCpuBandwidthJsonModeName = "cpu_bandwidth";
inline constexpr const char* kCpuBandwidthMethodologyVersion =
    "cpu-bandwidth-v1-arm64-calibrated-seeded-balanced";
inline constexpr const char* kCpuBandwidthComponentVersion =
    "cpu-bandwidth-components-v1";
inline constexpr const char* kCpuBandwidthWorkPlanVersion =
    "cpu-bandwidth-work-plan-v1";
inline constexpr const char* kCpuBandwidthBufferPatternVersion =
    "cpu-bw-buffer-pattern-v1";
inline constexpr const char* kCpuBandwidthReadChecksumVersion =
    "cpu-bw-read-checksum-v1";
inline constexpr const char* kCpuBandwidthRunPolicyVersion =
    "bounded-loop-snapshots-v1";

inline constexpr uint64_t kCpuBandwidthDomainMainMemory = 0x4342575F4D41494EULL;
inline constexpr uint64_t kCpuBandwidthDomainL1 = 0x4342575F4C315F5FULL;
inline constexpr uint64_t kCpuBandwidthDomainL2 = 0x4342575F4C325F5FULL;
inline constexpr uint64_t kCpuBandwidthDomainCustom = 0x4342575F43555354ULL;
inline constexpr uint64_t kCpuBandwidthDomainPreCopySentinel = 0x4342575F434F5059ULL;

inline constexpr size_t kCpuBandwidthMaxPasses =
    Constants::BANDWIDTH_CALIBRATION_MAX_PASSES;
inline constexpr size_t kCpuBandwidthMaxExactPayloadBytes =
    64ULL * 1024ULL * Constants::BYTES_PER_MB;
inline constexpr size_t kCpuBandwidthMaxMeasurements = 100000;
inline constexpr double kCpuBandwidthCvWarningPct = 5.0;

namespace CpuBandwidthWorkPlanReason {
inline constexpr const char* VALID = "valid";
inline constexpr const char* NO_ACTIVE_TARGETS = "target-omission";
inline constexpr const char* INVALID_CACHE_SIZE = "invalid-cache-size";
inline constexpr const char* INVALID_WORKER_COUNT = "invalid-worker-count";
inline constexpr const char* INVALID_LOOP_COUNT = "invalid-loop-count";
inline constexpr const char* INVALID_ITERATIONS = "invalid-iterations";
inline constexpr const char* INVALID_PAGE_SIZE = "invalid-page-size";
inline constexpr const char* TARGET_BYTES_OVERFLOW = "target-bytes-overflow";
inline constexpr const char* WORK_CAP_EXCEEDED = "work-cap-exceeded";
inline constexpr const char* MEASUREMENT_LIMIT_EXCEEDED = "measurement-limit-exceeded";
inline constexpr const char* JSON_INTEGER_OUT_OF_RANGE = "json-integer-out-of-range";
inline constexpr const char* MEMORY_BUDGET_OVERFLOW = "memory-budget-overflow";
inline constexpr const char* MEMORY_BUDGET_EXCEEDED = "memory-budget-exceeded";
inline constexpr const char* WITHIN_MEMORY_BUDGET = "within-memory-budget";
}  // namespace CpuBandwidthWorkPlanReason

/** Contiguous byte span assigned to one worker thread. */
struct CpuBandwidthWorkerSpan {
  size_t offset_bytes = 0;
  size_t span_bytes = 0;
};

/** Resolved geometry, seed, and worker plan for one target. */
struct CpuBandwidthTargetPlan {
  CpuBandwidthTarget target = CpuBandwidthTarget::MainMemory;
  bool enabled = false;
  size_t bytes = 0;
  size_t page_rounded_bytes = 0;
  size_t effective_workers = 0;
  std::vector<CpuBandwidthWorkerSpan> worker_spans;
  std::vector<size_t> boundaries;  ///< Size effective_workers + 1, covering [0, bytes].
  uint64_t src_seed = 0;
};

/** Expected checksum entry for one worker in a read operation plan. */
struct CpuBandwidthExpectedWorkerChecksum {
  size_t worker_index = 0;
  uint64_t state_a = 0;
  uint64_t state_b = 0;
  uint64_t exact_bytes_read = 0;
  uint64_t pass_count = 0;
};

/** Folded run checksum pair `(state_a, state_b)`. */
struct CpuBandwidthRunChecksum {
  uint64_t state_a = 0;
  uint64_t state_b = 0;
};

/** Expected checksum structure stored once per `operation_plans[]` entry. */
struct CpuBandwidthExpectedChecksum {
  bool applicable = false;
  bool available = false;
  std::string reason_code = "not-applicable";
  std::vector<CpuBandwidthExpectedWorkerChecksum> workers;
  CpuBandwidthRunChecksum run_checksum{};
};

/** Canonical operation work plan referenced by calibration attempts & measurements. */
struct CpuBandwidthOperationPlan {
  bool valid = false;
  std::string reason_code = CpuBandwidthWorkPlanReason::VALID;
  size_t plan_ref = 0;
  CpuBandwidthTarget target = CpuBandwidthTarget::MainMemory;
  CpuBandwidthOperation operation = CpuBandwidthOperation::Read;
  size_t passes = 0;
  bool explicit_iterations = false;
  size_t exact_payload_bytes = 0;
  uint64_t op_seed = 0;
  CpuBandwidthExpectedChecksum expected_checksum;
};

/** Conservative JSON DOM peak reserve estimate. */
struct CpuBandwidthJsonPeakEstimate {
  bool valid = false;
  std::string reason_code = "json-output-peak-bytes-overflow";
  size_t fixed_schema_bytes = 0;
  size_t input_string_bytes = 0;
  size_t measurement_record_bytes = 0;
  size_t worker_checksum_bytes = 0;
  size_t total_bytes = 0;
};

/** Strict preflight memory budget record. */
struct CpuBandwidthMemoryBudget {
  size_t requested_data_bytes = 0;
  size_t committed_data_bytes = 0;
  size_t resource_rounding_bytes = 0;
  size_t setup_peak_bytes = 0;
  size_t transient_peak_bytes = 0;
  size_t runtime_peak_bytes = 0;
  size_t known_owned_peak_bytes = 0;
  size_t available_memory_bytes = 0;
  size_t allowed_memory_bytes = 0;
  size_t admitted_budget_bytes = 0;
  bool used_fallback = false;
  bool valid = false;
  std::string reason_code = CpuBandwidthWorkPlanReason::MEMORY_BUDGET_OVERFLOW;
};

/** System hardware inputs consumed by pure work planning. */
struct CpuBandwidthSystemInfo {
  size_t l1_cache_bytes = 0;
  size_t l2_cache_bytes = 0;
  size_t page_size_bytes = 0;
  size_t available_memory_bytes = 0;
};

/** Component identity strings emitted under `resolved_plan.component_identities`. */
struct CpuBandwidthComponentIdentities {
  std::string version = kCpuBandwidthComponentVersion;
  std::string work_plan_version = kCpuBandwidthWorkPlanVersion;
  std::string buffer_pattern_version = kCpuBandwidthBufferPatternVersion;
  std::string read_checksum_version = kCpuBandwidthReadChecksumVersion;
  std::string run_policy_version = kCpuBandwidthRunPolicyVersion;
};

/** Complete pure work plan built prior to buffer allocation. */
struct CpuBandwidthWorkPlan {
  bool valid = false;
  std::string reason_code = CpuBandwidthWorkPlanReason::NO_ACTIVE_TARGETS;
  std::string plan_identity;
  CpuBandwidthComponentIdentities component_identities;
  uint64_t base_seed = 0;
  uint64_t pre_copy_sentinel_seed = 0;
  std::array<CpuBandwidthTargetPlan, kCpuBandwidthMaxTargets> targets;
  std::vector<CpuBandwidthTarget> active_targets;
  CpuBandwidthJsonPeakEstimate json_peak_estimate;
  CpuBandwidthMemoryBudget memory_budget;

  const CpuBandwidthTargetPlan* find_target(CpuBandwidthTarget target) const noexcept;
};

/** Return canonical lowercase token for a target. */
const char* cpu_bandwidth_target_to_string(CpuBandwidthTarget target) noexcept;

/** Return canonical lowercase token for an operation. */
const char* cpu_bandwidth_operation_to_string(CpuBandwidthOperation op) noexcept;

/** Return canonical array index `0..3` for a target. */
size_t cpu_bandwidth_target_index(CpuBandwidthTarget target) noexcept;

/** Return canonical array index `0..2` for an operation. */
size_t cpu_bandwidth_operation_index(CpuBandwidthOperation op) noexcept;

/** Return fixed domain separation constant for a target. */
uint64_t cpu_bandwidth_target_domain_id(CpuBandwidthTarget target) noexcept;

/** Derive `target_src_seed = splitmix64_mix(base_seed ^ target_domain)`. */
uint64_t derive_cpu_bandwidth_target_seed(uint64_t base_seed,
                                          CpuBandwidthTarget target) noexcept;

/** Derive `pre_copy_sentinel_seed = splitmix64_mix(base_seed ^ kDomainPreCopySentinel)`. */
uint64_t derive_cpu_bandwidth_pre_copy_sentinel_seed(uint64_t base_seed) noexcept;

/** Derive deterministic `op_seed` for `(target_src_seed, op, passes)`. */
uint64_t derive_cpu_bandwidth_op_seed(uint64_t target_src_seed,
                                      CpuBandwidthOperation op,
                                      size_t passes) noexcept;

/**
 * Partition `total_bytes` across `workers` matching Python `partition_64b`,
 * guaranteeing `offset_bytes % 64 == 0` for `idx > 0` whenever possible.
 */
std::vector<CpuBandwidthWorkerSpan> partition_cpu_bandwidth_64b(
    size_t total_bytes, size_t workers);

/** Build a deterministic work plan and perform strict memory budget preflight. */
CpuBandwidthWorkPlan build_cpu_bandwidth_work_plan(
    const CpuBandwidthConfig& config,
    const CpuBandwidthSystemInfo& sys_info);

/** Build and validate one operation plan (checking 64 GiB and pass caps). */
CpuBandwidthOperationPlan build_cpu_bandwidth_operation_plan(
    const CpuBandwidthTargetPlan& target_plan,
    CpuBandwidthOperation op,
    size_t passes,
    bool explicit_iterations);

/** Compute maximum allowed passes for a `(target_bytes, op)` respecting caps. */
size_t calculate_cpu_bandwidth_max_passes(
    size_t target_bytes, CpuBandwidthOperation op) noexcept;

#endif  // CPU_BANDWIDTH_WORK_PLAN_H

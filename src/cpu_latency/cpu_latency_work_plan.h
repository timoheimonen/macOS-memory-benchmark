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
 * @file cpu_latency_work_plan.h
 * @brief Deterministic target geometry, chain seeds, and memory budget preflight
 */

#ifndef CPU_LATENCY_WORK_PLAN_H
#define CPU_LATENCY_WORK_PLAN_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "cpu_latency/cpu_latency.h"
#include "cpu_latency/cpu_latency_chain.h"

/** Canonical latency targets in schema v1 `CANONICAL_TARGETS` order. */
enum class CpuLatencyTarget : uint8_t {
  L1 = 0,
  L2,
  Custom,
  MainMemory,
};

inline constexpr size_t kCpuLatencyMaxTargets = 4;

inline constexpr int kCpuLatencyJsonSchemaVersion = 1;
inline constexpr const char* kCpuLatencyJsonModeName = "cpu_latency";
inline constexpr const char* kCpuLatencyMethodologyVersion =
    "cpu-latency-v1-arm64-calibrated-seeded-balanced";
inline constexpr const char* kCpuLatencyComponentVersion =
    "cpu-latency-components-v1";
inline constexpr const char* kCpuLatencyWorkPlanVersion =
    "cpu-latency-work-plan-v1";
inline constexpr const char* kCpuLatencyRunPolicyVersion =
    "bounded-loop-snapshots-v1";

inline constexpr uint64_t kCpuLatencyDomainL1 = 0x434C41545F4C315FULL;
inline constexpr uint64_t kCpuLatencyDomainL2 = 0x434C41545F4C325FULL;
inline constexpr uint64_t kCpuLatencyDomainCustom = 0x434C41545F435553ULL;
inline constexpr uint64_t kCpuLatencyDomainMainMemory = 0x434C41545F4D4149ULL;
inline constexpr uint64_t kCpuLatencyDomainLocality16k = 0x434C41545F4C3136ULL;
inline constexpr uint64_t kCpuLatencyDomainLocalityGlobal =
    0x434C41545F474C4FULL;

inline constexpr size_t kCpuLatencyMaxCliIterations = 1000000000ULL;
inline constexpr size_t kCpuLatencyMaxAccesses =
    Constants::BENCHMARK_LATENCY_MAX_ACCESSES;
inline constexpr size_t kCpuLatencyMinCompleteCycles =
    Constants::BENCHMARK_LATENCY_MIN_COMPLETE_CYCLES;
inline constexpr size_t kCpuLatencyMaxMeasurements = 100000;
inline constexpr size_t kCpuLatencyMaxTargetPlans = 4096;
inline constexpr size_t kCpuLatencyLocality16kBytes = 16 * 1024;
inline constexpr size_t kCpuLatencyLocality16kKb = 16;
inline constexpr size_t kCpuLatencyCompareLocalityRoundsPerLoop = 3;
inline constexpr double kCpuLatencyCvWarningPct =
    Constants::BENCHMARK_CV_WARNING_PCT;

namespace CpuLatencyWorkPlanReason {
inline constexpr const char* VALID = "valid";
inline constexpr const char* NO_ACTIVE_TARGETS = "target-omission";
inline constexpr const char* INVALID_CACHE_SIZE = "invalid-cache-size";
inline constexpr const char* INVALID_LOOP_COUNT = "invalid-loop-count";
inline constexpr const char* INVALID_ITERATIONS = "invalid-iterations";
inline constexpr const char* INVALID_STRIDE = "invalid-stride";
inline constexpr const char* INVALID_SAMPLES = "invalid-samples";
inline constexpr const char* CHAIN_MODE_REQUIRES_LOCALITY =
    "chain-mode-requires-locality";
inline constexpr const char* LOCALITY_NOT_PAGE_MULTIPLE =
    "locality-not-page-multiple";
inline constexpr const char* CHAIN_GEOMETRY_INVALID = "chain-geometry-invalid";
inline constexpr const char* COMPARE_LOCALITY_INVALID =
    "compare-locality-invalid";
inline constexpr const char* INVALID_PAGE_SIZE = "invalid-page-size";
inline constexpr const char* TARGET_BYTES_OVERFLOW = "target-bytes-overflow";
inline constexpr const char* WORK_CAP_EXCEEDED = "work-cap-exceeded";
inline constexpr const char* MEASUREMENT_LIMIT_EXCEEDED =
    "measurement-limit-exceeded";
inline constexpr const char* JSON_INTEGER_OUT_OF_RANGE =
    "json-integer-out-of-range";
inline constexpr const char* MEMORY_BUDGET_OVERFLOW = "memory-budget-overflow";
inline constexpr const char* MEMORY_BUDGET_EXCEEDED = "memory-budget-exceeded";
inline constexpr const char* WITHIN_MEMORY_BUDGET = "within-memory-budget";
}  // namespace CpuLatencyWorkPlanReason

/** Resolved geometry, locality, and seed plan for one latency target. */
struct CpuLatencyTargetPlan {
  CpuLatencyTarget target = CpuLatencyTarget::MainMemory;
  bool enabled = false;
  size_t bytes = 0;
  size_t page_rounded_bytes = 0;
  size_t node_count = 0;
  size_t stride_bytes = 0;
  LatencyChainMode effective_mode = LatencyChainMode::GlobalRandom;
  size_t tlb_locality_kb = 0;
  size_t locality_nodes = 0;
  uint64_t chain_seed = 0;
};

/** Canonical target execution plan referenced by calibration & measurements. */
struct CpuLatencyTargetExecutionPlan {
  bool valid = false;
  std::string reason_code = CpuLatencyWorkPlanReason::VALID;
  size_t plan_ref = 0;
  CpuLatencyTarget target = CpuLatencyTarget::MainMemory;
  size_t complete_chain_cycles = 0;
  size_t access_count = 0;
  bool explicit_iterations = false;
};

/** Conservative JSON DOM peak reserve estimate. */
struct CpuLatencyJsonPeakEstimate {
  bool valid = false;
  std::string reason_code = "json-output-peak-bytes-overflow";
  size_t fixed_schema_bytes = 0;
  size_t input_string_bytes = 0;
  size_t measurement_record_bytes = 0;
  size_t sample_window_bytes = 0;
  size_t total_bytes = 0;
};

/** Strict preflight memory budget record. */
struct CpuLatencyMemoryBudget {
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
  std::string reason_code = CpuLatencyWorkPlanReason::MEMORY_BUDGET_OVERFLOW;
};

/** System hardware inputs consumed by pure work planning. */
struct CpuLatencySystemInfo {
  size_t l1_cache_bytes = 0;
  size_t l2_cache_bytes = 0;
  size_t page_size_bytes = 0;
  size_t available_memory_bytes = 0;
};

/** Component identity strings emitted under `resolved_plan.component_identities`. */
struct CpuLatencyComponentIdentities {
  std::string version = kCpuLatencyComponentVersion;
  std::string work_plan_version = kCpuLatencyWorkPlanVersion;
  std::string chain_generator_version = kCpuLatencyChainGeneratorVersion;
  std::string chain_version = kCpuLatencyChainVersion;
  std::string run_policy_version = kCpuLatencyRunPolicyVersion;
};

/** Complete pure work plan built prior to buffer allocation. */
struct CpuLatencyWorkPlan {
  bool valid = false;
  std::string reason_code = CpuLatencyWorkPlanReason::NO_ACTIVE_TARGETS;
  std::string plan_identity;
  CpuLatencyComponentIdentities component_identities;
  uint64_t base_seed = 0;
  LatencyChainMode effective_chain_mode = LatencyChainMode::GlobalRandom;
  size_t effective_tlb_locality_kb = 0;
  size_t locality_nodes = 0;
  std::array<CpuLatencyTargetPlan, kCpuLatencyMaxTargets> targets;
  std::vector<CpuLatencyTarget> active_targets;
  bool compare_locality = false;
  std::optional<CpuLatencyTargetPlan> locality_16k_plan;
  std::optional<CpuLatencyTargetPlan> locality_global_plan;
  CpuLatencyJsonPeakEstimate json_peak_estimate;
  CpuLatencyMemoryBudget memory_budget;

  const CpuLatencyTargetPlan* find_target(
      CpuLatencyTarget target) const noexcept;
};

/** Return canonical lowercase token for a target. */
const char* cpu_latency_target_to_string(CpuLatencyTarget target) noexcept;

/** Return canonical array index `0..3` for a target. */
size_t cpu_latency_target_index(CpuLatencyTarget target) noexcept;

/** Return fixed domain separation constant for a target. */
uint64_t cpu_latency_target_domain_id(CpuLatencyTarget target) noexcept;

/** Derive `chain_seed = splitmix64_mix(base_seed ^ domain_id)`. */
uint64_t derive_cpu_latency_chain_seed(uint64_t base_seed,
                                       uint64_t domain_id) noexcept;

/** Compute maximum complete chain cycles allowed for @p node_count. */
size_t calculate_cpu_latency_max_cycles(size_t node_count) noexcept;

/** Build a deterministic work plan and perform strict memory budget preflight. */
CpuLatencyWorkPlan build_cpu_latency_work_plan(
    const CpuLatencyConfig& config,
    const CpuLatencySystemInfo& sys_info);

/** Build and validate one target execution plan (`complete_chain_cycles`). */
CpuLatencyTargetExecutionPlan build_cpu_latency_target_execution_plan(
    const CpuLatencyTargetPlan& target_plan,
    size_t complete_chain_cycles,
    bool explicit_iterations);

#endif  // CPU_LATENCY_WORK_PLAN_H

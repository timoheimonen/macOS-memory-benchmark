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
 * @file cpu_latency_work_plan.cpp
 * @brief Deterministic target geometry, chain seeds, and memory budget preflight
 */

#include "cpu_latency/cpu_latency_work_plan.h"

#include <algorithm>
#include <limits>
#include <sstream>

#include "core/audit/deterministic_rng.h"
#include "utils/numeric_utils.h"

namespace {

constexpr size_t kJsonFixedSchemaPeakBytes = 2ULL * Constants::BYTES_PER_MB;
constexpr size_t kJsonInputStringExpansionFactor = 16;
constexpr size_t kJsonMeasurementPeakBytes = 16ULL * 1024ULL;
constexpr size_t kJsonSampleWindowPeakBytes = 128ULL;
constexpr size_t kSetupBaseScratchBytes = 4096ULL;

CpuLatencyJsonPeakEstimate estimate_json_peak(
    const CpuLatencyConfig& config,
    const std::vector<CpuLatencyTarget>& active_targets,
    const std::array<CpuLatencyTargetPlan, kCpuLatencyMaxTargets>& targets) noexcept {
  CpuLatencyJsonPeakEstimate estimate;
  if (config.output_file.empty() && !config.collect_json_for_sweep) {
    estimate.valid = true;
    estimate.reason_code = "valid";
    return estimate;
  }

  size_t raw_input_bytes = config.output_file.size();
  for (const std::string& arg : config.argv) {
    if (!NumericUtils::checked_add(raw_input_bytes, arg.size(), raw_input_bytes)) {
      return estimate;
    }
  }

  const size_t target_count = active_targets.size();
  const size_t calibration_attempts_per_target =
      config.iterations.has_value()
          ? size_t{1}
          : size_t{4} + Constants::BANDWIDTH_CALIBRATION_MAX_CORRECTIONS;

  size_t planned_measurements = 0;
  size_t total_target_plans = 0;
  size_t total_calibration_attempts = 0;
  size_t locality_rounds = 0;
  size_t total_records = 0;
  if (!NumericUtils::checked_multiply(config.loop_count, target_count,
                                      planned_measurements) ||
      !NumericUtils::checked_multiply(target_count,
                                      calibration_attempts_per_target + 1,
                                      total_target_plans) ||
      !NumericUtils::checked_multiply(target_count,
                                      calibration_attempts_per_target,
                                      total_calibration_attempts) ||
      !NumericUtils::checked_add(planned_measurements, total_calibration_attempts,
                                 total_records) ||
      !NumericUtils::checked_add(total_records, total_target_plans,
                                 total_records)) {
    return estimate;
  }

  if (config.compare_locality) {
    if (!NumericUtils::checked_multiply(
            config.loop_count, kCpuLatencyCompareLocalityRoundsPerLoop,
            locality_rounds) ||
        !NumericUtils::checked_add(total_records, locality_rounds,
                                   total_records)) {
      return estimate;
    }
  }

  size_t sample_window_bytes = 0;
  if (config.latency_samples > 0) {
    size_t max_nodes = 0;
    for (CpuLatencyTarget t : active_targets) {
      max_nodes =
          std::max(max_nodes, targets[cpu_latency_target_index(t)].node_count);
    }
    size_t upper_samples_per_row = config.latency_samples;
    if (config.iterations.has_value()) {
      size_t max_acc = 0;
      if (NumericUtils::checked_multiply(max_nodes, *config.iterations,
                                         max_acc)) {
        upper_samples_per_row = std::min(upper_samples_per_row, max_acc);
      }
    }
    size_t total_samples = 0;
    if (!NumericUtils::checked_multiply(planned_measurements,
                                        upper_samples_per_row, total_samples) ||
        !NumericUtils::checked_multiply(total_samples,
                                        kJsonSampleWindowPeakBytes,
                                        sample_window_bytes)) {
      return estimate;
    }
  }

  if (!NumericUtils::checked_multiply(raw_input_bytes,
                                      kJsonInputStringExpansionFactor,
                                      estimate.input_string_bytes) ||
      !NumericUtils::checked_multiply(total_records, kJsonMeasurementPeakBytes,
                                      estimate.measurement_record_bytes)) {
    return estimate;
  }

  estimate.sample_window_bytes = sample_window_bytes;
  estimate.fixed_schema_bytes = kJsonFixedSchemaPeakBytes;
  size_t total = estimate.fixed_schema_bytes;
  if (!NumericUtils::checked_add(total, estimate.input_string_bytes, total) ||
      !NumericUtils::checked_add(total, estimate.measurement_record_bytes,
                                 total) ||
      !NumericUtils::checked_add(total, estimate.sample_window_bytes, total)) {
    return estimate;
  }

  estimate.total_bytes = total;
  estimate.valid = true;
  estimate.reason_code = "valid";
  return estimate;
}

std::string build_plan_identity(const CpuLatencyConfig& config,
                                LatencyChainMode effective_mode,
                                size_t effective_locality_kb) {
  std::ostringstream oss;
  oss << kCpuLatencyWorkPlanVersion
      << "|buffer_size_mb=" << config.buffer_size_mb;
  if (!config.custom_cache_size_kb.has_value()) {
    oss << "|cache_mode=auto|custom_cache_size_kb=none";
  } else if (*config.custom_cache_size_kb == 0) {
    oss << "|cache_mode=disabled|custom_cache_size_kb=0";
  } else {
    oss << "|cache_mode=custom|custom_cache_size_kb="
        << *config.custom_cache_size_kb;
  }
  oss << "|stride=" << config.latency_stride_bytes
      << "|mode=" << latency_chain_mode_to_string(effective_mode)
      << "|tlb_locality_kb=" << effective_locality_kb
      << "|samples=" << config.latency_samples
      << "|compare_locality=" << (config.compare_locality ? 1 : 0);
  if (config.iterations.has_value()) {
    oss << "|iterations=" << *config.iterations;
  } else {
    oss << "|iterations=auto";
  }
  oss << "|loop_count=" << config.loop_count
      << "|non_cacheable=" << (config.non_cacheable ? 1 : 0);
  return oss.str();
}

}  // namespace

const CpuLatencyTargetPlan* CpuLatencyWorkPlan::find_target(
    CpuLatencyTarget target) const noexcept {
  const size_t idx = cpu_latency_target_index(target);
  if (idx >= targets.size() || !targets[idx].enabled) {
    return nullptr;
  }
  return &targets[idx];
}

const char* cpu_latency_target_to_string(CpuLatencyTarget target) noexcept {
  switch (target) {
    case CpuLatencyTarget::L1:
      return "l1";
    case CpuLatencyTarget::L2:
      return "l2";
    case CpuLatencyTarget::Custom:
      return "custom";
    case CpuLatencyTarget::MainMemory:
      return "main_memory";
  }
  return "unknown";
}

size_t cpu_latency_target_index(CpuLatencyTarget target) noexcept {
  return static_cast<size_t>(target);
}

uint64_t cpu_latency_target_domain_id(CpuLatencyTarget target) noexcept {
  switch (target) {
    case CpuLatencyTarget::L1:
      return kCpuLatencyDomainL1;
    case CpuLatencyTarget::L2:
      return kCpuLatencyDomainL2;
    case CpuLatencyTarget::Custom:
      return kCpuLatencyDomainCustom;
    case CpuLatencyTarget::MainMemory:
      return kCpuLatencyDomainMainMemory;
  }
  return kCpuLatencyDomainMainMemory;
}

uint64_t derive_cpu_latency_chain_seed(uint64_t base_seed,
                                       uint64_t domain_id) noexcept {
  return splitmix64_mix(base_seed ^ domain_id);
}

size_t calculate_cpu_latency_max_cycles(size_t node_count) noexcept {
  if (node_count == 0) {
    return 0;
  }
  return kCpuLatencyMaxAccesses / node_count;
}

CpuLatencyWorkPlan build_cpu_latency_work_plan(
    const CpuLatencyConfig& config,
    const CpuLatencySystemInfo& sys_info) {
  CpuLatencyWorkPlan plan;
  plan.base_seed = config.seed;
  plan.compare_locality = config.compare_locality;

  for (size_t i = 0; i < kCpuLatencyMaxTargets; ++i) {
    plan.targets[i].target = static_cast<CpuLatencyTarget>(i);
  }

  if (config.loop_count == 0 ||
      config.loop_count > Constants::LLM_JSON_MAX_SAFE_INTEGER) {
    plan.reason_code = CpuLatencyWorkPlanReason::INVALID_LOOP_COUNT;
    return plan;
  }
  if (config.buffer_size_mb > Constants::LLM_JSON_MAX_SAFE_INTEGER) {
    plan.reason_code = CpuLatencyWorkPlanReason::JSON_INTEGER_OUT_OF_RANGE;
    return plan;
  }
  if (config.custom_cache_size_kb.has_value()) {
    const size_t kb = *config.custom_cache_size_kb;
    if (kb != 0 &&
        (kb < static_cast<size_t>(Constants::MIN_CACHE_SIZE_KB) ||
         kb > static_cast<size_t>(Constants::MAX_CACHE_SIZE_KB))) {
      plan.reason_code = CpuLatencyWorkPlanReason::INVALID_CACHE_SIZE;
      return plan;
    }
  }
  if (config.iterations.has_value()) {
    if (*config.iterations == 0 ||
        *config.iterations > kCpuLatencyMaxAccesses) {
      plan.reason_code = CpuLatencyWorkPlanReason::INVALID_ITERATIONS;
      return plan;
    }
  }
  if (config.latency_samples >
      static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    plan.reason_code = CpuLatencyWorkPlanReason::INVALID_SAMPLES;
    return plan;
  }
  if (config.latency_stride_bytes < sizeof(uintptr_t) ||
      config.latency_stride_bytes >
          static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
      (config.latency_stride_bytes % sizeof(uintptr_t)) != 0) {
    plan.reason_code = CpuLatencyWorkPlanReason::INVALID_STRIDE;
    return plan;
  }
  if (sys_info.page_size_bytes == 0) {
    plan.reason_code = CpuLatencyWorkPlanReason::INVALID_PAGE_SIZE;
    return plan;
  }

  size_t locality_bytes = 0;
  if (!NumericUtils::checked_multiply(config.latency_tlb_locality_kb,
                                      Constants::BYTES_PER_KB,
                                      locality_bytes)) {
    plan.reason_code = CpuLatencyWorkPlanReason::MEMORY_BUDGET_OVERFLOW;
    return plan;
  }

  const LatencyChainMode effective_mode =
      resolve_latency_chain_mode(config.latency_chain_mode, locality_bytes);
  plan.effective_chain_mode = effective_mode;
  if (effective_mode != LatencyChainMode::GlobalRandom) {
    if (locality_bytes == 0) {
      plan.reason_code =
          CpuLatencyWorkPlanReason::CHAIN_MODE_REQUIRES_LOCALITY;
      return plan;
    }
    if ((locality_bytes % sys_info.page_size_bytes) != 0) {
      plan.reason_code = CpuLatencyWorkPlanReason::LOCALITY_NOT_PAGE_MULTIPLE;
      return plan;
    }
    plan.locality_nodes = locality_bytes / config.latency_stride_bytes;
    if (plan.locality_nodes < 2) {
      plan.reason_code = CpuLatencyWorkPlanReason::CHAIN_GEOMETRY_INVALID;
      return plan;
    }
    plan.effective_tlb_locality_kb = config.latency_tlb_locality_kb;
  } else {
    plan.locality_nodes = 0;
    plan.effective_tlb_locality_kb = config.latency_tlb_locality_kb;
  }

  if (config.compare_locality) {
    if (config.buffer_size_mb == 0 ||
        config.latency_stride_bytes > (kCpuLatencyLocality16kBytes / 2) ||
        (kCpuLatencyLocality16kBytes % sys_info.page_size_bytes) != 0) {
      plan.reason_code = CpuLatencyWorkPlanReason::COMPARE_LOCALITY_INVALID;
      return plan;
    }
  }

  const auto configure_target = [&](CpuLatencyTarget target,
                                    size_t target_bytes) -> bool {
    if (target_bytes == 0) {
      plan.reason_code = CpuLatencyWorkPlanReason::NO_ACTIVE_TARGETS;
      return false;
    }
    size_t rounded = 0;
    if (!NumericUtils::checked_round_up(target_bytes, sys_info.page_size_bytes,
                                        rounded)) {
      plan.reason_code = CpuLatencyWorkPlanReason::MEMORY_BUDGET_OVERFLOW;
      return false;
    }
    const size_t node_count = target_bytes / config.latency_stride_bytes;
    if (node_count < 2) {
      plan.reason_code = CpuLatencyWorkPlanReason::CHAIN_GEOMETRY_INVALID;
      return false;
    }
    if (node_count > kCpuLatencyMaxAccesses) {
      plan.reason_code = CpuLatencyWorkPlanReason::WORK_CAP_EXCEEDED;
      return false;
    }

    CpuLatencyTargetPlan& tplan =
        plan.targets[cpu_latency_target_index(target)];
    tplan.target = target;
    tplan.enabled = true;
    tplan.bytes = target_bytes;
    tplan.page_rounded_bytes = rounded;
    tplan.node_count = node_count;
    tplan.stride_bytes = config.latency_stride_bytes;
    tplan.effective_mode = effective_mode;
    tplan.tlb_locality_kb = config.latency_tlb_locality_kb;
    tplan.locality_nodes = plan.locality_nodes;
    tplan.chain_seed = derive_cpu_latency_chain_seed(
        config.seed, cpu_latency_target_domain_id(target));
    plan.active_targets.push_back(target);
    return true;
  };

  // Canonical order MUST match CANONICAL_TARGETS = ("l1", "l2", "custom", "main_memory").
  if (!config.custom_cache_size_kb.has_value()) {
    if (sys_info.l1_cache_bytes > 0 &&
        !configure_target(CpuLatencyTarget::L1, sys_info.l1_cache_bytes)) {
      return plan;
    }
    if (sys_info.l2_cache_bytes > 0 &&
        !configure_target(CpuLatencyTarget::L2, sys_info.l2_cache_bytes)) {
      return plan;
    }
  } else if (*config.custom_cache_size_kb > 0) {
    size_t custom_bytes = 0;
    if (!NumericUtils::checked_multiply(*config.custom_cache_size_kb,
                                        Constants::BYTES_PER_KB,
                                        custom_bytes)) {
      plan.reason_code = CpuLatencyWorkPlanReason::TARGET_BYTES_OVERFLOW;
      return plan;
    }
    if (!configure_target(CpuLatencyTarget::Custom, custom_bytes)) {
      return plan;
    }
  }

  if (config.buffer_size_mb > 0) {
    size_t main_bytes = 0;
    if (!NumericUtils::checked_multiply(config.buffer_size_mb,
                                        Constants::BYTES_PER_MB, main_bytes)) {
      plan.reason_code = CpuLatencyWorkPlanReason::TARGET_BYTES_OVERFLOW;
      return plan;
    }
    if (!configure_target(CpuLatencyTarget::MainMemory, main_bytes)) {
      return plan;
    }
  }

  if (plan.active_targets.empty()) {
    plan.reason_code = CpuLatencyWorkPlanReason::NO_ACTIVE_TARGETS;
    return plan;
  }

  if (config.compare_locality) {
    const CpuLatencyTargetPlan* main_plan =
        plan.find_target(CpuLatencyTarget::MainMemory);
    if (main_plan == nullptr) {
      plan.reason_code = CpuLatencyWorkPlanReason::COMPARE_LOCALITY_INVALID;
      return plan;
    }
    CpuLatencyTargetPlan loc_16k = *main_plan;
    loc_16k.effective_mode = LatencyChainMode::RandomInBoxRandomBox;
    loc_16k.tlb_locality_kb = kCpuLatencyLocality16kKb;
    loc_16k.locality_nodes =
        kCpuLatencyLocality16kBytes / config.latency_stride_bytes;
    loc_16k.chain_seed =
        derive_cpu_latency_chain_seed(config.seed, kCpuLatencyDomainLocality16k);
    plan.locality_16k_plan = std::move(loc_16k);

    CpuLatencyTargetPlan loc_global = *main_plan;
    loc_global.effective_mode = LatencyChainMode::GlobalRandom;
    loc_global.tlb_locality_kb = 0;
    loc_global.locality_nodes = 0;
    loc_global.chain_seed = derive_cpu_latency_chain_seed(
        config.seed, kCpuLatencyDomainLocalityGlobal);
    plan.locality_global_plan = std::move(loc_global);
  }

  size_t planned_measurements = 0;
  if (!NumericUtils::checked_multiply(config.loop_count,
                                      plan.active_targets.size(),
                                      planned_measurements) ||
      planned_measurements > kCpuLatencyMaxMeasurements) {
    plan.reason_code = CpuLatencyWorkPlanReason::MEASUREMENT_LIMIT_EXCEEDED;
    return plan;
  }

  plan.plan_identity =
      build_plan_identity(config, effective_mode, config.latency_tlb_locality_kb);

  // Strict memory budget calculation.
  CpuLatencyMemoryBudget& budget = plan.memory_budget;
  budget.available_memory_bytes = sys_info.available_memory_bytes;
  if (sys_info.available_memory_bytes == 0) {
    budget.used_fallback = true;
    if (!NumericUtils::checked_multiply(
            static_cast<size_t>(Constants::FALLBACK_TOTAL_LIMIT_MB),
            Constants::BYTES_PER_MB, budget.allowed_memory_bytes)) {
      budget.reason_code = CpuLatencyWorkPlanReason::MEMORY_BUDGET_OVERFLOW;
      plan.reason_code = budget.reason_code;
      return plan;
    }
  } else {
    const long double scaled =
        static_cast<long double>(sys_info.available_memory_bytes) *
        Constants::MEMORY_LIMIT_FACTOR;
    if (scaled > static_cast<long double>(std::numeric_limits<size_t>::max())) {
      budget.reason_code = CpuLatencyWorkPlanReason::MEMORY_BUDGET_OVERFLOW;
      plan.reason_code = budget.reason_code;
      return plan;
    }
    budget.allowed_memory_bytes = static_cast<size_t>(scaled);
  }
  budget.admitted_budget_bytes = budget.allowed_memory_bytes;

  size_t requested_data = 0;
  size_t committed_data = 0;
  size_t max_nodes = plan.locality_nodes;
  for (CpuLatencyTarget target : plan.active_targets) {
    const CpuLatencyTargetPlan& tplan =
        plan.targets[cpu_latency_target_index(target)];
    if (!NumericUtils::checked_add(requested_data, tplan.bytes,
                                   requested_data) ||
        !NumericUtils::checked_add(committed_data, tplan.page_rounded_bytes,
                                   committed_data)) {
      budget.reason_code = CpuLatencyWorkPlanReason::MEMORY_BUDGET_OVERFLOW;
      plan.reason_code = budget.reason_code;
      return plan;
    }
    max_nodes = std::max(max_nodes, tplan.node_count);
  }
  if (config.compare_locality) {
    const CpuLatencyTargetPlan* main_plan =
        plan.find_target(CpuLatencyTarget::MainMemory);
    if (!NumericUtils::checked_add(requested_data, main_plan->bytes,
                                   requested_data) ||
        !NumericUtils::checked_add(committed_data, main_plan->page_rounded_bytes,
                                   committed_data)) {
      budget.reason_code = CpuLatencyWorkPlanReason::MEMORY_BUDGET_OVERFLOW;
      plan.reason_code = budget.reason_code;
      return plan;
    }
  }
  budget.requested_data_bytes = requested_data;
  budget.committed_data_bytes = committed_data;
  budget.resource_rounding_bytes = committed_data - requested_data;

  plan.json_peak_estimate =
      estimate_json_peak(config, plan.active_targets, plan.targets);
  if (!plan.json_peak_estimate.valid) {
    budget.reason_code = CpuLatencyWorkPlanReason::MEMORY_BUDGET_OVERFLOW;
    plan.reason_code = budget.reason_code;
    return plan;
  }

  size_t chain_scratch = 0;
  if (!NumericUtils::checked_multiply(max_nodes, sizeof(uint64_t),
                                      chain_scratch) ||
      !NumericUtils::checked_add(kSetupBaseScratchBytes, chain_scratch,
                                 budget.setup_peak_bytes)) {
    budget.reason_code = CpuLatencyWorkPlanReason::MEMORY_BUDGET_OVERFLOW;
    plan.reason_code = budget.reason_code;
    return plan;
  }

  size_t runner_aux = 0;
  if (!NumericUtils::checked_multiply(planned_measurements, size_t{512},
                                      runner_aux) ||
      !NumericUtils::checked_add(runner_aux, budget.setup_peak_bytes,
                                 runner_aux) ||
      !NumericUtils::checked_add(runner_aux, plan.json_peak_estimate.total_bytes,
                                 budget.transient_peak_bytes) ||
      !NumericUtils::checked_add(committed_data, budget.transient_peak_bytes,
                                 budget.runtime_peak_bytes)) {
    budget.reason_code = CpuLatencyWorkPlanReason::MEMORY_BUDGET_OVERFLOW;
    plan.reason_code = budget.reason_code;
    return plan;
  }
  budget.known_owned_peak_bytes = budget.runtime_peak_bytes;

  if (budget.known_owned_peak_bytes > budget.admitted_budget_bytes) {
    budget.valid = false;
    budget.reason_code = CpuLatencyWorkPlanReason::MEMORY_BUDGET_EXCEEDED;
    plan.valid = false;
    plan.reason_code = CpuLatencyWorkPlanReason::MEMORY_BUDGET_EXCEEDED;
    return plan;
  }

  budget.valid = true;
  budget.reason_code = CpuLatencyWorkPlanReason::WITHIN_MEMORY_BUDGET;

  if (config.iterations.has_value()) {
    for (CpuLatencyTarget target : plan.active_targets) {
      const CpuLatencyTargetPlan& tplan =
          plan.targets[cpu_latency_target_index(target)];
      const CpuLatencyTargetExecutionPlan exec_plan =
          build_cpu_latency_target_execution_plan(tplan, *config.iterations,
                                                  true);
      if (!exec_plan.valid) {
        plan.valid = false;
        plan.reason_code = exec_plan.reason_code;
        return plan;
      }
    }
  }

  plan.valid = true;
  plan.reason_code = CpuLatencyWorkPlanReason::VALID;
  return plan;
}

CpuLatencyTargetExecutionPlan build_cpu_latency_target_execution_plan(
    const CpuLatencyTargetPlan& target_plan,
    size_t complete_chain_cycles,
    bool explicit_iterations) {
  CpuLatencyTargetExecutionPlan plan;
  plan.target = target_plan.target;
  plan.complete_chain_cycles = complete_chain_cycles;
  plan.explicit_iterations = explicit_iterations;

  if (!target_plan.enabled || target_plan.bytes == 0 ||
      target_plan.node_count < 2) {
    plan.reason_code = CpuLatencyWorkPlanReason::NO_ACTIVE_TARGETS;
    return plan;
  }
  if (complete_chain_cycles == 0 ||
      complete_chain_cycles > kCpuLatencyMaxAccesses) {
    plan.reason_code = CpuLatencyWorkPlanReason::WORK_CAP_EXCEEDED;
    return plan;
  }
  size_t accesses = 0;
  if (!NumericUtils::checked_multiply(complete_chain_cycles,
                                      target_plan.node_count, accesses) ||
      accesses > kCpuLatencyMaxAccesses) {
    plan.reason_code = CpuLatencyWorkPlanReason::WORK_CAP_EXCEEDED;
    return plan;
  }

  plan.access_count = accesses;
  plan.valid = true;
  plan.reason_code = CpuLatencyWorkPlanReason::VALID;
  return plan;
}

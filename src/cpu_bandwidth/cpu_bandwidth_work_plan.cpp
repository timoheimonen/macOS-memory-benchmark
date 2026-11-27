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
 * @file cpu_bandwidth_work_plan.cpp
 * @brief Deterministic target geometry, 64B worker spans, seeds, and budget preflight
 */

#include "cpu_bandwidth/cpu_bandwidth_work_plan.h"

#include <algorithm>
#include <limits>
#include <sstream>

#include "core/audit/deterministic_rng.h"
#include "utils/numeric_utils.h"

namespace {

constexpr size_t kJsonFixedSchemaPeakBytes = 2ULL * Constants::BYTES_PER_MB;
constexpr size_t kJsonInputStringExpansionFactor = 16;
constexpr size_t kJsonMeasurementPeakBytes = 64ULL * 1024ULL;
constexpr size_t kJsonWorkerChecksumPeakBytes = 64ULL * 1024ULL;
constexpr size_t kSetupBaseScratchBytes = 4096ULL;

CpuBandwidthJsonPeakEstimate estimate_json_peak(
    const CpuBandwidthConfig& config,
    const std::vector<CpuBandwidthTarget>& active_targets,
    const std::array<CpuBandwidthTargetPlan, kCpuBandwidthMaxTargets>& targets) noexcept {
  CpuBandwidthJsonPeakEstimate estimate;
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
  const size_t calibration_attempts_per_op =
      config.iterations.has_value()
          ? size_t{1}
          : size_t{4} + Constants::BANDWIDTH_CALIBRATION_MAX_CORRECTIONS;

  size_t per_loop_measurements = 0;
  size_t planned_measurements = 0;
  size_t total_op_plans = 0;
  size_t total_calibration_attempts = 0;
  size_t total_records = 0;
  if (!NumericUtils::checked_multiply(target_count, kCpuBandwidthOperationCount,
                                      per_loop_measurements) ||
      !NumericUtils::checked_multiply(config.loop_count, per_loop_measurements,
                                      planned_measurements) ||
      !NumericUtils::checked_multiply(per_loop_measurements,
                                      calibration_attempts_per_op + 1,
                                      total_op_plans) ||
      !NumericUtils::checked_multiply(per_loop_measurements,
                                      calibration_attempts_per_op,
                                      total_calibration_attempts) ||
      !NumericUtils::checked_add(planned_measurements, total_calibration_attempts,
                                 total_records) ||
      !NumericUtils::checked_add(total_records, total_op_plans, total_records)) {
    return estimate;
  }

  size_t sum_worker_checksum_records = 0;
  for (CpuBandwidthTarget t : active_targets) {
    const size_t workers = targets[cpu_bandwidth_target_index(t)].effective_workers;
    const size_t read_rows_for_target =
        config.loop_count + calibration_attempts_per_op + 1;
    size_t target_worker_records = 0;
    if (!NumericUtils::checked_multiply(read_rows_for_target, workers,
                                        target_worker_records) ||
        !NumericUtils::checked_add(sum_worker_checksum_records,
                                   target_worker_records,
                                   sum_worker_checksum_records)) {
      return estimate;
    }
  }

  if (!NumericUtils::checked_multiply(raw_input_bytes,
                                      kJsonInputStringExpansionFactor,
                                      estimate.input_string_bytes) ||
      !NumericUtils::checked_multiply(total_records, kJsonMeasurementPeakBytes,
                                      estimate.measurement_record_bytes) ||
      !NumericUtils::checked_multiply(sum_worker_checksum_records,
                                      kJsonWorkerChecksumPeakBytes,
                                      estimate.worker_checksum_bytes)) {
    return estimate;
  }

  estimate.fixed_schema_bytes = kJsonFixedSchemaPeakBytes;
  size_t total = estimate.fixed_schema_bytes;
  if (!NumericUtils::checked_add(total, estimate.input_string_bytes, total) ||
      !NumericUtils::checked_add(total, estimate.measurement_record_bytes, total) ||
      !NumericUtils::checked_add(total, estimate.worker_checksum_bytes, total)) {
    return estimate;
  }

  estimate.total_bytes = total;
  estimate.valid = true;
  estimate.reason_code = "valid";
  return estimate;
}

std::string build_plan_identity(const CpuBandwidthConfig& config,
                                size_t effective_main_workers) {
  std::ostringstream oss;
  oss << kCpuBandwidthWorkPlanVersion
      << "|buffer_size_mb=" << config.buffer_size_mb;
  if (!config.custom_cache_size_kb.has_value()) {
    oss << "|cache_mode=auto|custom_cache_size_kb=none";
  } else if (*config.custom_cache_size_kb == 0) {
    oss << "|cache_mode=disabled|custom_cache_size_kb=0";
  } else {
    oss << "|cache_mode=custom|custom_cache_size_kb="
        << *config.custom_cache_size_kb;
  }
  oss << "|workers=" << effective_main_workers
      << "|worker_source=" << (config.user_specified_workers ? "user" : "auto");
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

const CpuBandwidthTargetPlan* CpuBandwidthWorkPlan::find_target(
    CpuBandwidthTarget target) const noexcept {
  const size_t idx = cpu_bandwidth_target_index(target);
  if (idx >= targets.size() || !targets[idx].enabled) {
    return nullptr;
  }
  return &targets[idx];
}

const char* cpu_bandwidth_target_to_string(CpuBandwidthTarget target) noexcept {
  switch (target) {
    case CpuBandwidthTarget::MainMemory:
      return "main_memory";
    case CpuBandwidthTarget::L1:
      return "l1";
    case CpuBandwidthTarget::L2:
      return "l2";
    case CpuBandwidthTarget::Custom:
      return "custom";
  }
  return "unknown";
}

const char* cpu_bandwidth_operation_to_string(CpuBandwidthOperation op) noexcept {
  switch (op) {
    case CpuBandwidthOperation::Read:
      return "read";
    case CpuBandwidthOperation::Write:
      return "write";
    case CpuBandwidthOperation::Copy:
      return "copy";
  }
  return "unknown";
}

size_t cpu_bandwidth_target_index(CpuBandwidthTarget target) noexcept {
  return static_cast<size_t>(target);
}

size_t cpu_bandwidth_operation_index(CpuBandwidthOperation op) noexcept {
  return static_cast<size_t>(op);
}

uint64_t cpu_bandwidth_target_domain_id(CpuBandwidthTarget target) noexcept {
  switch (target) {
    case CpuBandwidthTarget::MainMemory:
      return kCpuBandwidthDomainMainMemory;
    case CpuBandwidthTarget::L1:
      return kCpuBandwidthDomainL1;
    case CpuBandwidthTarget::L2:
      return kCpuBandwidthDomainL2;
    case CpuBandwidthTarget::Custom:
      return kCpuBandwidthDomainCustom;
  }
  return kCpuBandwidthDomainMainMemory;
}

uint64_t derive_cpu_bandwidth_target_seed(uint64_t base_seed,
                                          CpuBandwidthTarget target) noexcept {
  return splitmix64_mix(base_seed ^ cpu_bandwidth_target_domain_id(target));
}

uint64_t derive_cpu_bandwidth_pre_copy_sentinel_seed(
    uint64_t base_seed) noexcept {
  return splitmix64_mix(base_seed ^ kCpuBandwidthDomainPreCopySentinel);
}

uint64_t derive_cpu_bandwidth_op_seed(uint64_t target_src_seed,
                                      CpuBandwidthOperation op,
                                      size_t passes) noexcept {
  const uint64_t op_tag =
      (static_cast<uint64_t>(cpu_bandwidth_operation_index(op)) + 1ULL) << 56U;
  return splitmix64_mix(target_src_seed ^ op_tag ^
                        static_cast<uint64_t>(passes));
}

std::vector<CpuBandwidthWorkerSpan> partition_cpu_bandwidth_64b(
    size_t total_bytes, size_t workers) {
  if (workers == 0) {
    return {};
  }
  const size_t active = std::min(total_bytes, workers);
  if (active == 0) {
    return std::vector<CpuBandwidthWorkerSpan>(workers, {0, 0});
  }

  std::vector<size_t> boundaries;
  boundaries.reserve(active + 1);
  boundaries.push_back(0);
  for (size_t rank = 1; rank < active; ++rank) {
    size_t target = (total_bytes / active) * rank +
                    std::min(rank, total_bytes % active);
    const size_t lo = boundaries.back() + 1;
    const size_t hi = total_bytes - (active - rank);
    target = std::max(lo, std::min(hi, target));
    const size_t aligned_lo = ((lo + 63) / 64) * 64;
    const size_t aligned_hi = (hi / 64) * 64;
    if (aligned_lo <= aligned_hi) {
      const size_t down = std::max(aligned_lo, std::min(aligned_hi, (target / 64) * 64));
      const size_t up = down < target ? std::min(aligned_hi, down + 64) : down;
      const size_t dist_down = target >= down ? target - down : down - target;
      const size_t dist_up = up >= target ? up - target : target - up;
      target = (dist_down <= dist_up) ? down : up;
    }
    boundaries.push_back(target);
  }
  boundaries.push_back(total_bytes);

  std::vector<CpuBandwidthWorkerSpan> spans;
  spans.reserve(workers);
  for (size_t i = 0; i < active; ++i) {
    spans.push_back({boundaries[i], boundaries[i + 1] - boundaries[i]});
  }
  for (size_t i = active; i < workers; ++i) {
    spans.push_back({0, 0});
  }
  return spans;
}

CpuBandwidthWorkPlan build_cpu_bandwidth_work_plan(
    const CpuBandwidthConfig& config,
    const CpuBandwidthSystemInfo& sys_info) {
  CpuBandwidthWorkPlan plan;
  plan.base_seed = config.seed;
  plan.pre_copy_sentinel_seed =
      derive_cpu_bandwidth_pre_copy_sentinel_seed(config.seed);

  for (size_t i = 0; i < kCpuBandwidthMaxTargets; ++i) {
    plan.targets[i].target = static_cast<CpuBandwidthTarget>(i);
  }

  if (config.requested_workers == 0 || config.available_workers == 0 ||
      config.requested_workers > Constants::LLM_JSON_MAX_SAFE_INTEGER ||
      config.available_workers > Constants::LLM_JSON_MAX_SAFE_INTEGER) {
    plan.reason_code = CpuBandwidthWorkPlanReason::INVALID_WORKER_COUNT;
    return plan;
  }
  if (config.loop_count == 0 ||
      config.loop_count > Constants::LLM_JSON_MAX_SAFE_INTEGER) {
    plan.reason_code = CpuBandwidthWorkPlanReason::INVALID_LOOP_COUNT;
    return plan;
  }
  if (config.buffer_size_mb > Constants::LLM_JSON_MAX_SAFE_INTEGER) {
    plan.reason_code = CpuBandwidthWorkPlanReason::JSON_INTEGER_OUT_OF_RANGE;
    return plan;
  }
  if (config.custom_cache_size_kb.has_value()) {
    const size_t kb = *config.custom_cache_size_kb;
    if (kb != 0 &&
        (kb < static_cast<size_t>(Constants::MIN_CACHE_SIZE_KB) ||
         kb > static_cast<size_t>(Constants::MAX_CACHE_SIZE_KB))) {
      plan.reason_code = CpuBandwidthWorkPlanReason::INVALID_CACHE_SIZE;
      return plan;
    }
  }
  if (config.iterations.has_value()) {
    if (*config.iterations == 0 ||
        *config.iterations > kCpuBandwidthMaxPasses) {
      plan.reason_code = CpuBandwidthWorkPlanReason::INVALID_ITERATIONS;
      return plan;
    }
  }
  if (sys_info.page_size_bytes == 0) {
    plan.reason_code = CpuBandwidthWorkPlanReason::INVALID_PAGE_SIZE;
    return plan;
  }

  const auto configure_target = [&](CpuBandwidthTarget target,
                                    size_t target_bytes) -> bool {
    if (target_bytes == 0) {
      plan.reason_code = CpuBandwidthWorkPlanReason::NO_ACTIVE_TARGETS;
      return false;
    }
    size_t rounded = 0;
    if (!NumericUtils::checked_round_up(target_bytes, sys_info.page_size_bytes,
                                        rounded)) {
      plan.reason_code = CpuBandwidthWorkPlanReason::MEMORY_BUDGET_OVERFLOW;
      return false;
    }
    const size_t desired = config.user_specified_workers
                               ? config.requested_workers
                               : (target == CpuBandwidthTarget::MainMemory
                                      ? config.requested_workers
                                      : size_t{1});
    const size_t max_aligned =
        std::max<size_t>(1, target_bytes / Constants::CACHE_LINE_SIZE_BYTES);
    const size_t effective =
        std::min({desired, config.available_workers, max_aligned});
    if (effective == 0) {
      plan.reason_code = CpuBandwidthWorkPlanReason::INVALID_WORKER_COUNT;
      return false;
    }

    CpuBandwidthTargetPlan& tplan =
        plan.targets[cpu_bandwidth_target_index(target)];
    tplan.target = target;
    tplan.enabled = true;
    tplan.bytes = target_bytes;
    tplan.page_rounded_bytes = rounded;
    tplan.effective_workers = effective;
    tplan.worker_spans = partition_cpu_bandwidth_64b(target_bytes, effective);
    tplan.boundaries.reserve(effective + 1);
    tplan.boundaries.push_back(0);
    for (const auto& sp : tplan.worker_spans) {
      tplan.boundaries.push_back(sp.offset_bytes + sp.span_bytes);
    }
    tplan.src_seed = derive_cpu_bandwidth_target_seed(config.seed, target);
    plan.active_targets.push_back(target);
    return true;
  };

  if (config.buffer_size_mb > 0) {
    size_t main_bytes = 0;
    if (!NumericUtils::checked_multiply(config.buffer_size_mb,
                                        Constants::BYTES_PER_MB, main_bytes)) {
      plan.reason_code = CpuBandwidthWorkPlanReason::TARGET_BYTES_OVERFLOW;
      return plan;
    }
    if (!configure_target(CpuBandwidthTarget::MainMemory, main_bytes)) {
      return plan;
    }
  }

  if (!config.custom_cache_size_kb.has_value()) {
    if (sys_info.l1_cache_bytes > 0 &&
        !configure_target(CpuBandwidthTarget::L1, sys_info.l1_cache_bytes)) {
      return plan;
    }
    if (sys_info.l2_cache_bytes > 0 &&
        !configure_target(CpuBandwidthTarget::L2, sys_info.l2_cache_bytes)) {
      return plan;
    }
  } else if (*config.custom_cache_size_kb > 0) {
    size_t custom_bytes = 0;
    if (!NumericUtils::checked_multiply(*config.custom_cache_size_kb,
                                        Constants::BYTES_PER_KB,
                                        custom_bytes)) {
      plan.reason_code = CpuBandwidthWorkPlanReason::TARGET_BYTES_OVERFLOW;
      return plan;
    }
    if (!configure_target(CpuBandwidthTarget::Custom, custom_bytes)) {
      return plan;
    }
  }

  if (plan.active_targets.empty()) {
    plan.reason_code = CpuBandwidthWorkPlanReason::NO_ACTIVE_TARGETS;
    return plan;
  }

  size_t planned_measurements = 0;
  if (!NumericUtils::checked_multiply(
          config.loop_count,
          plan.active_targets.size() * kCpuBandwidthOperationCount,
          planned_measurements) ||
      planned_measurements > kCpuBandwidthMaxMeasurements) {
    plan.reason_code = CpuBandwidthWorkPlanReason::MEASUREMENT_LIMIT_EXCEEDED;
    return plan;
  }

  // Build plan identity using main_memory workers or first active target workers.
  const CpuBandwidthTargetPlan* main_plan =
      plan.find_target(CpuBandwidthTarget::MainMemory);
  const size_t identity_workers =
      main_plan != nullptr
          ? main_plan->effective_workers
          : plan.targets[cpu_bandwidth_target_index(plan.active_targets.front())]
                .effective_workers;
  plan.plan_identity = build_plan_identity(config, identity_workers);

  // Strict memory budget calculation.
  CpuBandwidthMemoryBudget& budget = plan.memory_budget;
  budget.available_memory_bytes = sys_info.available_memory_bytes;
  if (sys_info.available_memory_bytes == 0) {
    budget.used_fallback = true;
    if (!NumericUtils::checked_multiply(
            static_cast<size_t>(Constants::FALLBACK_TOTAL_LIMIT_MB),
            Constants::BYTES_PER_MB, budget.allowed_memory_bytes)) {
      budget.reason_code = CpuBandwidthWorkPlanReason::MEMORY_BUDGET_OVERFLOW;
      plan.reason_code = budget.reason_code;
      return plan;
    }
  } else {
    const long double scaled =
        static_cast<long double>(sys_info.available_memory_bytes) *
        Constants::MEMORY_LIMIT_FACTOR;
    if (scaled > static_cast<long double>(std::numeric_limits<size_t>::max())) {
      budget.reason_code = CpuBandwidthWorkPlanReason::MEMORY_BUDGET_OVERFLOW;
      plan.reason_code = budget.reason_code;
      return plan;
    }
    budget.allowed_memory_bytes = static_cast<size_t>(scaled);
  }
  budget.admitted_budget_bytes = budget.allowed_memory_bytes;

  size_t requested_data = 0;
  size_t committed_data = 0;
  size_t max_workers = 1;
  for (CpuBandwidthTarget target : plan.active_targets) {
    const CpuBandwidthTargetPlan& tplan =
        plan.targets[cpu_bandwidth_target_index(target)];
    size_t req_pair = 0;
    size_t com_pair = 0;
    if (!NumericUtils::checked_multiply(tplan.bytes, size_t{2}, req_pair) ||
        !NumericUtils::checked_multiply(tplan.page_rounded_bytes, size_t{2},
                                        com_pair) ||
        !NumericUtils::checked_add(requested_data, req_pair, requested_data) ||
        !NumericUtils::checked_add(committed_data, com_pair, committed_data)) {
      budget.reason_code = CpuBandwidthWorkPlanReason::MEMORY_BUDGET_OVERFLOW;
      plan.reason_code = budget.reason_code;
      return plan;
    }
    max_workers = std::max(max_workers, tplan.effective_workers);
  }
  budget.requested_data_bytes = requested_data;
  budget.committed_data_bytes = committed_data;
  budget.resource_rounding_bytes = committed_data - requested_data;

  plan.json_peak_estimate =
      estimate_json_peak(config, plan.active_targets, plan.targets);
  if (!plan.json_peak_estimate.valid) {
    budget.reason_code = CpuBandwidthWorkPlanReason::MEMORY_BUDGET_OVERFLOW;
    plan.reason_code = budget.reason_code;
    return plan;
  }

  size_t worker_scratch = 0;
  if (!NumericUtils::checked_multiply(max_workers, size_t{64}, worker_scratch) ||
      !NumericUtils::checked_add(kSetupBaseScratchBytes, worker_scratch,
                                 budget.setup_peak_bytes)) {
    budget.reason_code = CpuBandwidthWorkPlanReason::MEMORY_BUDGET_OVERFLOW;
    plan.reason_code = budget.reason_code;
    return plan;
  }

  size_t runner_aux = 0;
  if (!NumericUtils::checked_multiply(planned_measurements, size_t{512},
                                      runner_aux) ||
      !NumericUtils::checked_add(runner_aux, kSetupBaseScratchBytes,
                                 runner_aux) ||
      !NumericUtils::checked_add(runner_aux, plan.json_peak_estimate.total_bytes,
                                 budget.transient_peak_bytes) ||
      !NumericUtils::checked_add(committed_data, budget.transient_peak_bytes,
                                 budget.runtime_peak_bytes)) {
    budget.reason_code = CpuBandwidthWorkPlanReason::MEMORY_BUDGET_OVERFLOW;
    plan.reason_code = budget.reason_code;
    return plan;
  }
  budget.known_owned_peak_bytes = budget.runtime_peak_bytes;

  if (budget.known_owned_peak_bytes > budget.admitted_budget_bytes) {
    budget.valid = false;
    budget.reason_code = CpuBandwidthWorkPlanReason::MEMORY_BUDGET_EXCEEDED;
    plan.valid = false;
    plan.reason_code = CpuBandwidthWorkPlanReason::MEMORY_BUDGET_EXCEEDED;
    return plan;
  }

  budget.valid = true;
  budget.reason_code = CpuBandwidthWorkPlanReason::WITHIN_MEMORY_BUDGET;

  // Validate explicit iteration work caps for every active target & operation.
  if (config.iterations.has_value()) {
    for (CpuBandwidthTarget target : plan.active_targets) {
      const CpuBandwidthTargetPlan& tplan =
          plan.targets[cpu_bandwidth_target_index(target)];
      for (size_t op_i = 0; op_i < kCpuBandwidthOperationCount; ++op_i) {
        const auto op = static_cast<CpuBandwidthOperation>(op_i);
        const CpuBandwidthOperationPlan op_plan =
            build_cpu_bandwidth_operation_plan(tplan, op, *config.iterations,
                                               true);
        if (!op_plan.valid) {
          plan.valid = false;
          plan.reason_code = op_plan.reason_code;
          return plan;
        }
      }
    }
  }

  plan.valid = true;
  plan.reason_code = CpuBandwidthWorkPlanReason::VALID;
  return plan;
}

size_t calculate_cpu_bandwidth_max_passes(
    size_t target_bytes, CpuBandwidthOperation op) noexcept {
  if (target_bytes == 0) {
    return 0;
  }
  const size_t multiplier =
      (op == CpuBandwidthOperation::Copy) ? Constants::COPY_OPERATION_MULTIPLIER : 1;
  size_t bytes_per_pass = 0;
  if (!NumericUtils::checked_multiply(target_bytes, multiplier, bytes_per_pass) ||
      bytes_per_pass == 0) {
    return 0;
  }
  return std::min(kCpuBandwidthMaxPasses,
                  kCpuBandwidthMaxExactPayloadBytes / bytes_per_pass);
}

CpuBandwidthOperationPlan build_cpu_bandwidth_operation_plan(
    const CpuBandwidthTargetPlan& target_plan,
    CpuBandwidthOperation op,
    size_t passes,
    bool explicit_iterations) {
  CpuBandwidthOperationPlan plan;
  plan.target = target_plan.target;
  plan.operation = op;
  plan.passes = passes;
  plan.explicit_iterations = explicit_iterations;

  if (!target_plan.enabled || target_plan.bytes == 0) {
    plan.reason_code = CpuBandwidthWorkPlanReason::NO_ACTIVE_TARGETS;
    return plan;
  }
  if (passes == 0 || passes > kCpuBandwidthMaxPasses) {
    plan.reason_code = CpuBandwidthWorkPlanReason::WORK_CAP_EXCEEDED;
    return plan;
  }
  const size_t multiplier =
      (op == CpuBandwidthOperation::Copy) ? Constants::COPY_OPERATION_MULTIPLIER : 1;
  size_t bytes_per_pass = 0;
  if (!NumericUtils::checked_multiply(target_plan.bytes, multiplier,
                                      bytes_per_pass) ||
      !NumericUtils::checked_multiply(passes, bytes_per_pass,
                                      plan.exact_payload_bytes) ||
      plan.exact_payload_bytes > kCpuBandwidthMaxExactPayloadBytes) {
    plan.reason_code = CpuBandwidthWorkPlanReason::WORK_CAP_EXCEEDED;
    return plan;
  }

  plan.op_seed =
      derive_cpu_bandwidth_op_seed(target_plan.src_seed, op, passes);
  plan.valid = true;
  plan.reason_code = CpuBandwidthWorkPlanReason::VALID;
  return plan;
}

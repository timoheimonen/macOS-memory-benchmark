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
 * @file patterns_work_plan.cpp
 * @brief Deterministic 7-pattern geometry, 64B worker spans, seeds, and budget preflight
 */

#include "pattern_benchmark/patterns_work_plan.h"

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

PatternsJsonPeakEstimate estimate_json_peak(
    const PatternsWorkPlanConfig& config,
    const std::vector<PatternKind>& active_patterns,
    const std::array<PatternPlanEntry, kPatternsMaxKinds>& patterns) noexcept {
  PatternsJsonPeakEstimate estimate;
  if (config.output_file.empty() && !config.collect_json_for_sweep) {
    estimate.valid = true;
    estimate.reason_code = PatternsWorkPlanReason::VALID;
    return estimate;
  }

  size_t raw_input_bytes = config.output_file.size();
  for (const std::string& arg : config.argv) {
    if (!NumericUtils::checked_add(raw_input_bytes, arg.size(), raw_input_bytes)) {
      return estimate;
    }
  }

  const size_t pattern_count = active_patterns.size();
  const size_t calibration_attempts_per_op =
      config.iterations.has_value()
          ? size_t{1}
          : size_t{4} + Constants::BANDWIDTH_CALIBRATION_MAX_CORRECTIONS;

  size_t per_loop_measurements = 0;
  size_t planned_measurements = 0;
  size_t total_op_plans = 0;
  size_t total_calibration_attempts = 0;
  size_t total_records = 0;
  if (!NumericUtils::checked_multiply(pattern_count, kPatternsOperationCount,
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
  for (PatternKind kind : active_patterns) {
    const size_t workers = patterns[patterns_kind_index(kind)].effective_workers;
    const size_t read_rows_for_pattern =
        config.loop_count + calibration_attempts_per_op + 1;
    size_t pattern_worker_records = 0;
    if (!NumericUtils::checked_multiply(read_rows_for_pattern, workers,
                                        pattern_worker_records) ||
        !NumericUtils::checked_add(sum_worker_checksum_records,
                                   pattern_worker_records,
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
  estimate.reason_code = PatternsWorkPlanReason::VALID;
  return estimate;
}

std::string build_plan_identity(const PatternsWorkPlanConfig& config,
                                size_t identity_workers) {
  std::ostringstream oss;
  oss << kPatternsWorkPlanVersion
      << "|buffer_size_mb=" << config.buffer_size_mb
      << "|workers=" << identity_workers
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

void populate_boundaries(PatternPlanEntry& entry) {
  entry.boundaries.clear();
  if (entry.worker_spans.empty()) {
    return;
  }
  entry.boundaries.reserve(entry.worker_spans.size() + 1);
  entry.boundaries.push_back(entry.worker_spans.front().offset_bytes);
  for (const auto& sp : entry.worker_spans) {
    entry.boundaries.push_back(sp.offset_bytes + sp.span_bytes);
  }
}

void distribute_random_accesses(std::vector<PatternsWorkerSpan>& spans,
                                size_t total_selected) {
  const size_t workers = spans.size();
  if (workers == 0 || total_selected < workers) {
    return;
  }

  size_t total_slots = 0;
  for (const auto& sp : spans) {
    total_slots += sp.span_bytes / kPatternsAccessSizeBytes;
  }
  if (total_slots < total_selected) {
    return;
  }

  // Start with 1 access per worker, then distribute remaining proportionally to
  // remaining capacity `(span_slots - 1)`.
  size_t distributed = 0;
  const size_t remaining = total_selected - workers;
  const size_t sum_capacity = total_slots - workers;
  for (auto& sp : spans) {
    const size_t span_slots = sp.span_bytes / kPatternsAccessSizeBytes;
    const size_t cap = (span_slots > 0) ? (span_slots - 1) : 0;
    size_t add = 0;
    if (remaining > 0 && sum_capacity > 0) {
      add = static_cast<size_t>(
          (static_cast<unsigned __int128>(remaining) * cap) / sum_capacity);
      if (add > cap) {
        add = cap;
      }
    }
    sp.random_access_count = 1 + add;
    sp.accesses_per_pass = sp.random_access_count;
    distributed += sp.random_access_count;
  }

  size_t leftover = total_selected - distributed;
  for (size_t pass = 0; leftover > 0; ++pass) {
    bool progressed = false;
    for (auto& sp : spans) {
      if (leftover == 0) {
        break;
      }
      const size_t span_slots = sp.span_bytes / kPatternsAccessSizeBytes;
      if (sp.random_access_count < span_slots) {
        ++sp.random_access_count;
        sp.accesses_per_pass = sp.random_access_count;
        --leftover;
        progressed = true;
      }
    }
    if (!progressed) {
      break;
    }
    (void)pass;
  }
}

}  // namespace

const PatternPlanEntry* PatternsWorkPlan::find_pattern(
    PatternKind kind) const noexcept {
  const size_t idx = patterns_kind_index(kind);
  if (idx >= patterns.size() || !patterns[idx].enabled) {
    return nullptr;
  }
  return &patterns[idx];
}

const PatternPlanEntry* PatternsWorkPlan::entry(
    PatternKind kind) const noexcept {
  const size_t idx = patterns_kind_index(kind);
  if (idx >= patterns.size()) {
    return nullptr;
  }
  return &patterns[idx];
}

const char* patterns_kind_to_string(PatternKind kind) noexcept {
  switch (kind) {
    case PatternKind::SequentialForward:
      return "sequential_forward";
    case PatternKind::SequentialReverse:
      return "sequential_reverse";
    case PatternKind::Strided64:
      return "strided_64";
    case PatternKind::Strided4096:
      return "strided_4096";
    case PatternKind::Strided16384:
      return "strided_16384";
    case PatternKind::Strided2MiB:
      return "strided_2mb";
    case PatternKind::Random:
      return "random";
  }
  return "unknown";
}

const char* patterns_operation_to_string(PatternOperation op) noexcept {
  switch (op) {
    case PatternOperation::Read:
      return "read";
    case PatternOperation::Write:
      return "write";
    case PatternOperation::Copy:
      return "copy";
  }
  return "unknown";
}

size_t patterns_kind_index(PatternKind kind) noexcept {
  return static_cast<size_t>(kind);
}

size_t patterns_operation_index(PatternOperation op) noexcept {
  return static_cast<size_t>(op);
}

bool patterns_kind_is_strided(PatternKind kind) noexcept {
  return kind == PatternKind::Strided64 || kind == PatternKind::Strided4096 ||
         kind == PatternKind::Strided16384 || kind == PatternKind::Strided2MiB;
}

size_t patterns_kind_stride_bytes(PatternKind kind) noexcept {
  switch (kind) {
    case PatternKind::Strided64:
      return Constants::PATTERN_STRIDE_CACHE_LINE;
    case PatternKind::Strided4096:
      return Constants::PATTERN_STRIDE_PAGE;
    case PatternKind::Strided16384:
      return Constants::PATTERN_STRIDE_PAGE_16K;
    case PatternKind::Strided2MiB:
      return Constants::PATTERN_STRIDE_SUPERPAGE_2MB;
    case PatternKind::SequentialForward:
    case PatternKind::SequentialReverse:
    case PatternKind::Random:
      return 0;
  }
  return 0;
}

bool patterns_kind_is_streaming(PatternKind kind) noexcept {
  return kind == PatternKind::SequentialForward ||
         kind == PatternKind::SequentialReverse ||
         kind == PatternKind::Strided64;
}

double patterns_cv_warning_threshold_pct(PatternKind kind) noexcept {
  return patterns_kind_is_streaming(kind) ? kPatternsStreamingCvWarningPct
                                          : kPatternsSparseCvWarningPct;
}

uint64_t derive_patterns_src_seed(uint64_t base_seed) noexcept {
  return splitmix64_mix(base_seed ^ kPatternDomainSrc);
}

uint64_t derive_patterns_random_domain_seed(uint64_t base_seed) noexcept {
  return splitmix64_mix(base_seed ^ kPatternDomainRandom);
}

uint64_t derive_patterns_pre_op_sentinel_seed(uint64_t base_seed) noexcept {
  return splitmix64_mix(base_seed ^ kPatternDomainPreOpSentinel);
}

uint64_t derive_patterns_op_seed(uint64_t src_seed,
                                 PatternKind kind,
                                 PatternOperation op,
                                 size_t passes) noexcept {
  const uint64_t kind_tag =
      (static_cast<uint64_t>(patterns_kind_index(kind)) + 1ULL) << 48U;
  const uint64_t op_tag =
      (static_cast<uint64_t>(patterns_operation_index(op)) + 1ULL) << 56U;
  return splitmix64_mix(src_seed ^ kind_tag ^ op_tag ^
                        static_cast<uint64_t>(passes));
}

std::vector<PatternsWorkerSpan> partition_patterns_64b(
    size_t total_bytes, size_t workers) {
  if (workers == 0) {
    return {};
  }
  const size_t active = std::min(total_bytes, workers);
  if (active == 0) {
    return std::vector<PatternsWorkerSpan>(workers, {0, 0, 0, 0});
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
      const size_t down =
          std::max(aligned_lo, std::min(aligned_hi, (target / 64) * 64));
      const size_t up = down < target ? std::min(aligned_hi, down + 64) : down;
      const size_t dist_down = target >= down ? target - down : down - target;
      const size_t dist_up = up >= target ? up - target : target - up;
      target = (dist_down <= dist_up) ? down : up;
    }
    boundaries.push_back(target);
  }
  boundaries.push_back(total_bytes);

  std::vector<PatternsWorkerSpan> spans;
  spans.reserve(workers);
  for (size_t i = 0; i < active; ++i) {
    spans.push_back({boundaries[i], boundaries[i + 1] - boundaries[i], 0, 0});
  }
  for (size_t i = active; i < workers; ++i) {
    spans.push_back({0, 0, 0, 0});
  }
  return spans;
}

PatternsWorkPlan build_patterns_work_plan(
    const PatternsWorkPlanConfig& config,
    const PatternsSystemInfo& sys_info) {
  PatternsWorkPlan plan;
  plan.base_seed = config.seed;
  plan.src_seed = derive_patterns_src_seed(config.seed);
  plan.random_domain_seed = derive_patterns_random_domain_seed(config.seed);
  plan.pre_op_sentinel_seed = derive_patterns_pre_op_sentinel_seed(config.seed);

  for (size_t i = 0; i < kPatternsMaxKinds; ++i) {
    const auto kind = static_cast<PatternKind>(i);
    PatternPlanEntry& entry = plan.patterns[i];
    entry.kind = kind;
    entry.access_size_bytes = kPatternsAccessSizeBytes;
    entry.stride_bytes = patterns_kind_stride_bytes(kind);
    entry.phase_period_passes =
        (entry.stride_bytes > 0) ? (entry.stride_bytes / kPatternsAccessSizeBytes) : 0;
    entry.native_page_size_bytes = sys_info.page_size_bytes;
    entry.stride_equals_native_page_size =
        (entry.stride_bytes > 0 && sys_info.page_size_bytes > 0 &&
         entry.stride_bytes == sys_info.page_size_bytes);
    entry.large_page_backing_verified = false;
    entry.large_page_backing_status =
        (kind == PatternKind::Strided2MiB) ? "not-verified" : "not-applicable";
    entry.random_workload.random_seed = plan.random_domain_seed;
  }

  if (config.requested_workers == 0 || config.available_workers == 0 ||
      config.requested_workers > Constants::LLM_JSON_MAX_SAFE_INTEGER ||
      config.available_workers > Constants::LLM_JSON_MAX_SAFE_INTEGER) {
    plan.reason_code = PatternsWorkPlanReason::INVALID_WORKER_COUNT;
    return plan;
  }
  if (config.loop_count == 0 ||
      config.loop_count > Constants::LLM_JSON_MAX_SAFE_INTEGER) {
    plan.reason_code = PatternsWorkPlanReason::INVALID_LOOP_COUNT;
    return plan;
  }
  if (config.buffer_size_mb == 0) {
    plan.reason_code = PatternsWorkPlanReason::INVALID_BUFFER_SIZE;
    return plan;
  }
  if (config.buffer_size_mb > Constants::LLM_JSON_MAX_SAFE_INTEGER) {
    plan.reason_code = PatternsWorkPlanReason::JSON_INTEGER_OUT_OF_RANGE;
    return plan;
  }
  if (config.iterations.has_value()) {
    if (*config.iterations == 0 || *config.iterations > kPatternMaxPasses) {
      plan.reason_code = PatternsWorkPlanReason::INVALID_ITERATIONS;
      return plan;
    }
  }
  if (sys_info.page_size_bytes == 0) {
    plan.reason_code = PatternsWorkPlanReason::INVALID_PAGE_SIZE;
    return plan;
  }

  size_t buffer_bytes = 0;
  if (!NumericUtils::checked_multiply(config.buffer_size_mb,
                                      Constants::BYTES_PER_MB, buffer_bytes)) {
    plan.reason_code = PatternsWorkPlanReason::BUFFER_BYTES_OVERFLOW;
    return plan;
  }
  plan.buffer_bytes = buffer_bytes;

  size_t page_rounded = 0;
  if (!NumericUtils::checked_round_up(buffer_bytes, sys_info.page_size_bytes,
                                      page_rounded)) {
    plan.reason_code = PatternsWorkPlanReason::MEMORY_BUDGET_OVERFLOW;
    return plan;
  }
  plan.page_rounded_bytes = page_rounded;

  for (size_t i = 0; i < kPatternsMaxKinds; ++i) {
    PatternPlanEntry& entry = plan.patterns[i];
    entry.buffer_bytes = buffer_bytes;
    const PatternKind kind = entry.kind;

    if (kind == PatternKind::SequentialForward ||
        kind == PatternKind::SequentialReverse) {
      const size_t max_aligned =
          std::max<size_t>(1, buffer_bytes / Constants::CACHE_LINE_SIZE_BYTES);
      const size_t effective = std::min(
          {config.requested_workers, config.available_workers, max_aligned});
      if (effective == 0) {
        plan.reason_code = PatternsWorkPlanReason::INVALID_WORKER_COUNT;
        return plan;
      }
      entry.enabled = true;
      entry.status = PatternsWorkPlanReason::ACTIVE;
      entry.reason_code = PatternsWorkPlanReason::ACTIVE;
      entry.effective_workers = effective;
      entry.worker_spans = partition_patterns_64b(buffer_bytes, effective);
      size_t total_accesses = 0;
      for (auto& sp : entry.worker_spans) {
        sp.accesses_per_pass = sp.span_bytes / kPatternsAccessSizeBytes;
        sp.random_access_count = 0;
        total_accesses += sp.accesses_per_pass;
      }
      entry.accesses_per_pass = total_accesses;
      populate_boundaries(entry);
      if (kind == PatternKind::SequentialForward) {
        plan.buffer_worker_spans = entry.worker_spans;
      }
      plan.active_patterns.push_back(kind);
    } else if (patterns_kind_is_strided(kind)) {
      const size_t min_worker_span =
          entry.stride_bytes + kPatternsAccessSizeBytes;
      if (buffer_bytes < min_worker_span) {
        entry.enabled = false;
        entry.status = "skipped";
        entry.reason_code = PatternsWorkPlanReason::SKIPPED_BUFFER_TOO_SMALL;
        entry.effective_workers = 0;
        entry.accesses_per_pass = 0;
        continue;
      }

      const size_t max_workers_by_span = buffer_bytes / min_worker_span;
      const size_t first_candidate = std::min(
          {config.requested_workers, config.available_workers, max_workers_by_span});

      bool populated = false;
      size_t chosen_workers = 0;
      std::vector<PatternsWorkerSpan> chosen_spans;
      size_t chosen_accesses_per_pass = 0;

      for (size_t candidate = first_candidate; candidate >= 1; --candidate) {
        std::vector<PatternsWorkerSpan> spans =
            partition_patterns_64b(buffer_bytes, candidate);
        bool all_valid = (spans.size() == candidate);
        size_t total_phase0 = 0;
        for (auto& sp : spans) {
          if (sp.span_bytes < min_worker_span ||
              (sp.offset_bytes % kPatternsAccessSizeBytes) != 0 ||
              (sp.span_bytes % kPatternsAccessSizeBytes) != 0) {
            all_valid = false;
            break;
          }
          sp.accesses_per_pass =
              1 + (sp.span_bytes - kPatternsAccessSizeBytes) / entry.stride_bytes;
          if (sp.accesses_per_pass < 2) {
            all_valid = false;
            break;
          }
          sp.random_access_count = 0;
          total_phase0 += sp.accesses_per_pass;
        }
        if (all_valid) {
          populated = true;
          chosen_workers = candidate;
          chosen_spans = std::move(spans);
          chosen_accesses_per_pass = total_phase0;
          break;
        }
      }

      if (!populated) {
        entry.enabled = false;
        entry.status = "skipped";
        entry.reason_code = PatternsWorkPlanReason::SKIPPED_BUFFER_TOO_SMALL;
        entry.effective_workers = 0;
        entry.accesses_per_pass = 0;
        continue;
      }

      entry.enabled = true;
      entry.status = PatternsWorkPlanReason::ACTIVE;
      entry.reason_code = PatternsWorkPlanReason::ACTIVE;
      entry.effective_workers = chosen_workers;
      entry.worker_spans = std::move(chosen_spans);
      entry.accesses_per_pass = chosen_accesses_per_pass;
      populate_boundaries(entry);
      plan.active_patterns.push_back(kind);
    } else {
      // PatternKind::Random
      const size_t total_slots = buffer_bytes / kPatternsAccessSizeBytes;
      const size_t clamped = std::clamp(total_slots, kPatternsMinRandomAccesses,
                                        kPatternsMaxRandomAccesses);
      const size_t selected = std::min(total_slots, clamped);
      const size_t max_aligned =
          std::max<size_t>(1, buffer_bytes / Constants::CACHE_LINE_SIZE_BYTES);
      const size_t effective = std::min(
          {config.requested_workers, config.available_workers, max_aligned, selected});
      if (effective == 0) {
        plan.reason_code = PatternsWorkPlanReason::INVALID_WORKER_COUNT;
        return plan;
      }
      entry.enabled = true;
      entry.status = PatternsWorkPlanReason::ACTIVE;
      entry.reason_code = PatternsWorkPlanReason::ACTIVE;
      entry.effective_workers = effective;
      entry.worker_spans = partition_patterns_64b(buffer_bytes, effective);
      distribute_random_accesses(entry.worker_spans, selected);
      entry.accesses_per_pass = selected;
      entry.random_workload.random_seed = plan.random_domain_seed;
      entry.random_workload.selected_accesses_per_pass = selected;
      entry.random_workload.distinct_address_count = selected;
      entry.random_workload.logical_working_set_bytes = 0;
      populate_boundaries(entry);
      plan.active_patterns.push_back(kind);
    }
  }

  size_t per_loop_measurements = 0;
  size_t planned_measurements = 0;
  if (!NumericUtils::checked_multiply(plan.active_patterns.size(),
                                      kPatternsOperationCount,
                                      per_loop_measurements) ||
      !NumericUtils::checked_multiply(config.loop_count, per_loop_measurements,
                                      planned_measurements) ||
      planned_measurements > kPatternMaxMeasurements) {
    plan.reason_code = PatternsWorkPlanReason::MEASUREMENT_LIMIT_EXCEEDED;
    return plan;
  }

  const PatternPlanEntry* fwd_plan =
      plan.find_pattern(PatternKind::SequentialForward);
  const size_t identity_workers =
      (fwd_plan != nullptr) ? fwd_plan->effective_workers : 0;
  plan.plan_identity = build_plan_identity(config, identity_workers);

  // Strict memory budget calculation.
  PatternsMemoryBudget& budget = plan.memory_budget;
  budget.available_memory_bytes = sys_info.available_memory_bytes;
  if (sys_info.available_memory_bytes == 0) {
    budget.used_fallback = true;
    if (!NumericUtils::checked_multiply(
            static_cast<size_t>(Constants::FALLBACK_TOTAL_LIMIT_MB),
            Constants::BYTES_PER_MB, budget.allowed_memory_bytes)) {
      budget.reason_code = PatternsWorkPlanReason::MEMORY_BUDGET_OVERFLOW;
      plan.reason_code = budget.reason_code;
      return plan;
    }
  } else {
    const long double scaled =
        static_cast<long double>(sys_info.available_memory_bytes) *
        Constants::MEMORY_LIMIT_FACTOR;
    if (scaled > static_cast<long double>(std::numeric_limits<size_t>::max())) {
      budget.reason_code = PatternsWorkPlanReason::MEMORY_BUDGET_OVERFLOW;
      plan.reason_code = budget.reason_code;
      return plan;
    }
    budget.allowed_memory_bytes = static_cast<size_t>(scaled);
  }
  budget.admitted_budget_bytes = budget.allowed_memory_bytes;

  size_t requested_data = 0;
  size_t committed_data = 0;
  if (!NumericUtils::checked_multiply(buffer_bytes, size_t{2}, requested_data) ||
      !NumericUtils::checked_multiply(page_rounded, size_t{2}, committed_data)) {
    budget.reason_code = PatternsWorkPlanReason::MEMORY_BUDGET_OVERFLOW;
    plan.reason_code = budget.reason_code;
    return plan;
  }
  budget.requested_data_bytes = requested_data;
  budget.committed_data_bytes = committed_data;
  budget.resource_rounding_bytes = committed_data - requested_data;

  const PatternPlanEntry* random_entry = plan.find_pattern(PatternKind::Random);
  const size_t selected_random =
      (random_entry != nullptr)
          ? random_entry->random_workload.selected_accesses_per_pass
          : 0;
  size_t index_bytes_single = 0;
  if (!NumericUtils::checked_multiply(selected_random, sizeof(size_t),
                                      index_bytes_single) ||
      !NumericUtils::checked_multiply(index_bytes_single, size_t{2},
                                      budget.index_structure_bytes)) {
    budget.reason_code = PatternsWorkPlanReason::MEMORY_BUDGET_OVERFLOW;
    plan.reason_code = budget.reason_code;
    return plan;
  }

  size_t max_workers = 1;
  for (PatternKind kind : plan.active_patterns) {
    max_workers = std::max(
        max_workers, plan.patterns[patterns_kind_index(kind)].effective_workers);
  }

  plan.json_peak_estimate =
      estimate_json_peak(config, plan.active_patterns, plan.patterns);
  if (!plan.json_peak_estimate.valid) {
    budget.reason_code = PatternsWorkPlanReason::MEMORY_BUDGET_OVERFLOW;
    plan.reason_code = budget.reason_code;
    return plan;
  }

  size_t worker_scratch = 0;
  if (!NumericUtils::checked_multiply(max_workers, size_t{64}, worker_scratch) ||
      !NumericUtils::checked_add(kSetupBaseScratchBytes, worker_scratch,
                                 budget.setup_peak_bytes) ||
      !NumericUtils::checked_add(budget.setup_peak_bytes,
                                 budget.index_structure_bytes,
                                 budget.setup_peak_bytes)) {
    budget.reason_code = PatternsWorkPlanReason::MEMORY_BUDGET_OVERFLOW;
    plan.reason_code = budget.reason_code;
    return plan;
  }

  size_t runner_aux = 0;
  if (!NumericUtils::checked_multiply(planned_measurements, size_t{512},
                                      runner_aux) ||
      !NumericUtils::checked_add(runner_aux, kSetupBaseScratchBytes,
                                 runner_aux) ||
      !NumericUtils::checked_add(runner_aux, budget.index_structure_bytes,
                                 runner_aux) ||
      !NumericUtils::checked_add(runner_aux, plan.json_peak_estimate.total_bytes,
                                 budget.transient_peak_bytes) ||
      !NumericUtils::checked_add(committed_data, budget.transient_peak_bytes,
                                 budget.runtime_peak_bytes)) {
    budget.reason_code = PatternsWorkPlanReason::MEMORY_BUDGET_OVERFLOW;
    plan.reason_code = budget.reason_code;
    return plan;
  }
  budget.known_owned_peak_bytes = budget.runtime_peak_bytes;

  if (budget.known_owned_peak_bytes > budget.admitted_budget_bytes) {
    budget.valid = false;
    budget.reason_code = PatternsWorkPlanReason::MEMORY_BUDGET_EXCEEDED;
    plan.valid = false;
    plan.reason_code = PatternsWorkPlanReason::MEMORY_BUDGET_EXCEEDED;
    return plan;
  }

  budget.valid = true;
  budget.reason_code = PatternsWorkPlanReason::WITHIN_MEMORY_BUDGET;

  // Validate explicit iteration work caps for every active pattern & operation.
  if (config.iterations.has_value()) {
    for (PatternKind kind : plan.active_patterns) {
      const PatternPlanEntry& pplan = plan.patterns[patterns_kind_index(kind)];
      for (size_t op_i = 0; op_i < kPatternsOperationCount; ++op_i) {
        const auto op = static_cast<PatternOperation>(op_i);
        const PatternsOperationPlan op_plan =
            build_patterns_operation_plan(pplan, plan.src_seed, op,
                                          *config.iterations, true);
        if (!op_plan.valid) {
          plan.valid = false;
          plan.reason_code = op_plan.reason_code;
          return plan;
        }
      }
    }
  }

  plan.valid = true;
  plan.reason_code = PatternsWorkPlanReason::VALID;
  return plan;
}

PatternsOperationPlan build_patterns_operation_plan(
    const PatternPlanEntry& pattern_plan,
    uint64_t src_seed,
    PatternOperation op,
    size_t passes,
    bool explicit_iterations) {
  PatternsOperationPlan plan;
  plan.pattern = pattern_plan.kind;
  plan.operation = op;
  plan.passes = passes;
  plan.explicit_iterations = explicit_iterations;

  if (!pattern_plan.enabled || pattern_plan.buffer_bytes == 0 ||
      pattern_plan.worker_spans.empty()) {
    plan.reason_code = pattern_plan.reason_code.empty()
                           ? PatternsWorkPlanReason::SKIPPED_BUFFER_TOO_SMALL
                           : pattern_plan.reason_code;
    return plan;
  }
  if (passes == 0 || passes > kPatternMaxPasses) {
    plan.reason_code = PatternsWorkPlanReason::WORK_CAP_EXCEEDED;
    return plan;
  }

  if (pattern_plan.kind == PatternKind::SequentialForward ||
      pattern_plan.kind == PatternKind::SequentialReverse) {
    plan.accesses_per_pass = pattern_plan.accesses_per_pass;
    plan.min_accesses_per_pass = plan.accesses_per_pass;
    plan.max_accesses_per_pass = plan.accesses_per_pass;
    if (plan.accesses_per_pass == 0 ||
        !NumericUtils::checked_multiply(passes, plan.accesses_per_pass,
                                        plan.total_accesses)) {
      plan.reason_code = PatternsWorkPlanReason::WORK_CAP_EXCEEDED;
      return plan;
    }
    plan.distinct_address_count = plan.accesses_per_pass;
    plan.logical_working_set_bytes = pattern_plan.buffer_bytes;
    plan.completed_phase_cycles = 0;
  } else if (pattern_plan.kind == PatternKind::Random) {
    plan.accesses_per_pass =
        pattern_plan.random_workload.selected_accesses_per_pass;
    plan.min_accesses_per_pass = plan.accesses_per_pass;
    plan.max_accesses_per_pass = plan.accesses_per_pass;
    if (plan.accesses_per_pass == 0 ||
        !NumericUtils::checked_multiply(passes, plan.accesses_per_pass,
                                        plan.total_accesses)) {
      plan.reason_code = PatternsWorkPlanReason::WORK_CAP_EXCEEDED;
      return plan;
    }
    plan.distinct_address_count =
        pattern_plan.random_workload.distinct_address_count;
    plan.logical_working_set_bytes =
        pattern_plan.random_workload.logical_working_set_bytes;
    plan.completed_phase_cycles = 0;
  } else {
    // Strided patterns with 2D phase rotation.
    const size_t phase_period = pattern_plan.phase_period_passes;
    if (phase_period == 0 || pattern_plan.stride_bytes == 0 ||
        pattern_plan.stride_bytes % kPatternsAccessSizeBytes != 0) {
      plan.reason_code = PatternsWorkPlanReason::WORK_CAP_EXCEEDED;
      return plan;
    }

    size_t accesses_per_cycle = 0;
    for (const auto& sp : pattern_plan.worker_spans) {
      const size_t aligned_slots = sp.span_bytes / kPatternsAccessSizeBytes;
      if (aligned_slots < 2 ||
          !NumericUtils::checked_add(accesses_per_cycle, aligned_slots,
                                     accesses_per_cycle)) {
        plan.reason_code = PatternsWorkPlanReason::WORK_CAP_EXCEEDED;
        return plan;
      }
    }

    plan.completed_phase_cycles = passes / phase_period;
    const size_t remaining_phases = passes % phase_period;
    if (!NumericUtils::checked_multiply(plan.completed_phase_cycles,
                                        accesses_per_cycle,
                                        plan.total_accesses)) {
      plan.reason_code = PatternsWorkPlanReason::WORK_CAP_EXCEEDED;
      return plan;
    }

    const size_t executed_phase_count = std::min(passes, phase_period);
    const size_t last_executed_phase = executed_phase_count - 1;
    plan.accesses_per_pass = pattern_plan.accesses_per_pass;
    plan.min_accesses_per_pass = 0;
    plan.max_accesses_per_pass = 0;
    plan.distinct_address_count = 0;
    plan.logical_working_set_bytes = 0;

    for (const auto& sp : pattern_plan.worker_spans) {
      const size_t aligned_slots = sp.span_bytes / kPatternsAccessSizeBytes;
      const size_t full_blocks = aligned_slots / phase_period;
      const size_t tail_slots = aligned_slots % phase_period;

      const size_t phase_zero_accesses =
          1 + (aligned_slots - 1) / phase_period;
      const size_t last_phase_accesses =
          last_executed_phase < aligned_slots
              ? 1 + (aligned_slots - 1 - last_executed_phase) / phase_period
              : 0;
      if (!NumericUtils::checked_add(plan.max_accesses_per_pass,
                                     phase_zero_accesses,
                                     plan.max_accesses_per_pass) ||
          !NumericUtils::checked_add(plan.min_accesses_per_pass,
                                     last_phase_accesses,
                                     plan.min_accesses_per_pass)) {
        plan.reason_code = PatternsWorkPlanReason::WORK_CAP_EXCEEDED;
        return plan;
      }

      size_t partial_full_blocks = 0;
      if (!NumericUtils::checked_multiply(full_blocks, remaining_phases,
                                          partial_full_blocks)) {
        plan.reason_code = PatternsWorkPlanReason::WORK_CAP_EXCEEDED;
        return plan;
      }
      const size_t partial_accesses =
          partial_full_blocks + std::min(tail_slots, remaining_phases);
      if (!NumericUtils::checked_add(plan.total_accesses, partial_accesses,
                                     plan.total_accesses)) {
        plan.reason_code = PatternsWorkPlanReason::WORK_CAP_EXCEEDED;
        return plan;
      }

      size_t distinct_full_blocks = 0;
      if (!NumericUtils::checked_multiply(full_blocks, executed_phase_count,
                                          distinct_full_blocks)) {
        plan.reason_code = PatternsWorkPlanReason::WORK_CAP_EXCEEDED;
        return plan;
      }
      const size_t selected_tail_slots =
          std::min(tail_slots, executed_phase_count);
      const size_t worker_distinct =
          distinct_full_blocks + selected_tail_slots;
      if (!NumericUtils::checked_add(plan.distinct_address_count, worker_distinct,
                                     plan.distinct_address_count)) {
        plan.reason_code = PatternsWorkPlanReason::WORK_CAP_EXCEEDED;
        return plan;
      }

      size_t last_selected_slot = 0;
      bool has_selected_slot = false;
      if (full_blocks > 0 && executed_phase_count > 0) {
        last_selected_slot =
            (full_blocks - 1) * phase_period + executed_phase_count - 1;
        has_selected_slot = true;
      }
      if (selected_tail_slots > 0) {
        last_selected_slot =
            full_blocks * phase_period + selected_tail_slots - 1;
        has_selected_slot = true;
      }
      if (has_selected_slot) {
        size_t worker_logical_bytes = 0;
        if (!NumericUtils::checked_multiply(last_selected_slot + 1,
                                            kPatternsAccessSizeBytes,
                                            worker_logical_bytes) ||
            !NumericUtils::checked_add(plan.logical_working_set_bytes,
                                       worker_logical_bytes,
                                       plan.logical_working_set_bytes)) {
          plan.reason_code = PatternsWorkPlanReason::WORK_CAP_EXCEEDED;
          return plan;
        }
      }
    }
  }

  size_t touched_bytes = 0;
  if (!NumericUtils::checked_multiply(plan.distinct_address_count,
                                      kPatternsAccessSizeBytes, touched_bytes)) {
    plan.reason_code = PatternsWorkPlanReason::WORK_CAP_EXCEEDED;
    return plan;
  }
  plan.has_untouched_dst_bytes = (touched_bytes < pattern_plan.buffer_bytes);

  const size_t multiplier =
      (op == PatternOperation::Copy) ? Constants::COPY_OPERATION_MULTIPLIER : 1;
  size_t base_payload_bytes = 0;
  if (!NumericUtils::checked_multiply(plan.total_accesses,
                                      kPatternsAccessSizeBytes,
                                      base_payload_bytes) ||
      !NumericUtils::checked_multiply(base_payload_bytes, multiplier,
                                      plan.exact_payload_bytes) ||
      plan.exact_payload_bytes > kPatternMaxExactPayloadBytes) {
    plan.reason_code = PatternsWorkPlanReason::WORK_CAP_EXCEEDED;
    return plan;
  }

  plan.op_seed =
      derive_patterns_op_seed(src_seed, pattern_plan.kind, op, passes);
  plan.valid = true;
  plan.reason_code = PatternsWorkPlanReason::VALID;
  return plan;
}

size_t calculate_patterns_max_passes(
    const PatternPlanEntry& pattern_plan, PatternOperation op) noexcept {
  if (!pattern_plan.enabled || pattern_plan.buffer_bytes == 0 ||
      pattern_plan.worker_spans.empty()) {
    return 0;
  }
  const size_t multiplier =
      (op == PatternOperation::Copy) ? Constants::COPY_OPERATION_MULTIPLIER : 1;

  if (!patterns_kind_is_strided(pattern_plan.kind)) {
    size_t bytes_per_pass = 0;
    if (pattern_plan.accesses_per_pass == 0 ||
        !NumericUtils::checked_multiply(pattern_plan.accesses_per_pass,
                                        kPatternsAccessSizeBytes, bytes_per_pass) ||
        !NumericUtils::checked_multiply(bytes_per_pass, multiplier,
                                        bytes_per_pass) ||
        bytes_per_pass == 0) {
      return 0;
    }
    return std::min(kPatternMaxPasses,
                    kPatternMaxExactPayloadBytes / bytes_per_pass);
  }

  // Strided: total_accesses is monotonically non-decreasing in passes.
  const auto fits = [&](size_t candidate_passes) -> bool {
    const PatternsOperationPlan op_plan = build_patterns_operation_plan(
        pattern_plan, 0ULL, op, candidate_passes, true);
    return op_plan.valid;
  };
  if (!fits(1)) {
    return 0;
  }
  if (fits(kPatternMaxPasses)) {
    return kPatternMaxPasses;
  }
  size_t lo = 1;
  size_t hi = kPatternMaxPasses;
  while (lo < hi) {
    const size_t mid = lo + (hi - lo + 1) / 2;
    if (fits(mid)) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return lo;
}

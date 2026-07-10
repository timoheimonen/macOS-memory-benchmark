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
//
/**
 * @file pattern_work_plan.cpp
 * @brief Deterministic work planning for pattern benchmarks
 */
#include "pattern_benchmark/pattern_work_plan.h"

#include "output/console/messages/messages_api.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "core/config/constants.h"

namespace {

bool checked_add(size_t lhs, size_t rhs, size_t& result) {
  if (lhs > std::numeric_limits<size_t>::max() - rhs) {
    return false;
  }
  result = lhs + rhs;
  return true;
}

bool checked_multiply(size_t lhs, size_t rhs, size_t& result) {
  if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
    return false;
  }
  result = lhs * rhs;
  return true;
}

std::vector<size_t> build_aligned_boundaries(size_t size, int num_threads) {
  const size_t thread_count = static_cast<size_t>(num_threads);
  std::vector<size_t> boundaries(thread_count + 1, 0);
  const size_t chunk_base_size = size / thread_count;
  const size_t chunk_remainder = size % thread_count;

  size_t offset = 0;
  for (size_t thread = 0; thread < thread_count; ++thread) {
    offset += chunk_base_size + (thread < chunk_remainder ? 1 : 0);
    boundaries[thread + 1] = offset;
  }

  for (size_t index = 1; index < thread_count; ++index) {
    const size_t remainder = boundaries[index] % Constants::CACHE_LINE_SIZE_BYTES;
    if (remainder != 0) {
      const size_t adjustment = Constants::CACHE_LINE_SIZE_BYTES - remainder;
      boundaries[index] = adjustment > size - boundaries[index] ? size : boundaries[index] + adjustment;
    }
    boundaries[index] = std::min(boundaries[index], size);
    boundaries[index] = std::max(boundaries[index], boundaries[index - 1]);
  }

  boundaries.front() = 0;
  boundaries.back() = size;
  return boundaries;
}

bool populate_worker_ranges(PatternWorkPlan& plan, size_t buffer_size, int worker_count) {
  const std::vector<size_t> boundaries = build_aligned_boundaries(buffer_size, worker_count);
  std::vector<PatternWorkerRange> workers;
  workers.reserve(static_cast<size_t>(worker_count));

  size_t total_accesses = 0;
  size_t total_payload = 0;
  for (int worker = 0; worker < worker_count; ++worker) {
    const size_t start = boundaries[static_cast<size_t>(worker)];
    const size_t end = boundaries[static_cast<size_t>(worker) + 1];
    const size_t span = end - start;
    if (span < plan.access_size_bytes) {
      return false;
    }

    const size_t accesses = 1 + (span - plan.access_size_bytes) / plan.stride_bytes;
    if (accesses < 2) {
      return false;
    }

    size_t worker_payload = 0;
    if (!checked_multiply(accesses, plan.access_size_bytes, worker_payload) ||
        !checked_add(total_accesses, accesses, total_accesses) ||
        !checked_add(total_payload, worker_payload, total_payload)) {
      plan.status = PatternMeasurementStatus::Invalid;
      plan.status_reason = Messages::pattern_reason_work_plan_byte_overflow();
      return false;
    }

    workers.push_back({start, span, accesses, worker_payload});
  }

  plan.workers = std::move(workers);
  plan.effective_threads = worker_count;
  plan.accesses_per_pass = total_accesses;
  plan.payload_bytes_per_pass = total_payload;
  return true;
}

bool calculate_phase_metrics(const PatternWorkPlan& plan, size_t passes,
                             size_t& total_accesses, size_t& distinct_addresses,
                             size_t& logical_working_set_bytes,
                             size_t& completed_phase_cycles,
                             size_t& min_accesses_per_pass,
                             size_t& max_accesses_per_pass) {
  if (passes == 0 || plan.access_size_bytes == 0 || plan.stride_bytes == 0 ||
      plan.stride_bytes % plan.access_size_bytes != 0 || plan.workers.empty() ||
      passes > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return false;
  }

  const size_t phase_period = plan.stride_bytes / plan.access_size_bytes;
  if (phase_period == 0) {
    return false;
  }

  size_t accesses_per_cycle = 0;
  for (const PatternWorkerRange& worker : plan.workers) {
    const size_t aligned_slots = worker.span_bytes / plan.access_size_bytes;
    if (aligned_slots < 2 || !checked_add(accesses_per_cycle, aligned_slots,
                                          accesses_per_cycle)) {
      return false;
    }
  }

  completed_phase_cycles = passes / phase_period;
  const size_t remaining_phases = passes % phase_period;
  if (!checked_multiply(completed_phase_cycles, accesses_per_cycle, total_accesses)) {
    return false;
  }

  const size_t executed_phase_count = std::min(passes, phase_period);
  const size_t last_executed_phase = executed_phase_count - 1;
  min_accesses_per_pass = 0;
  max_accesses_per_pass = 0;
  for (const PatternWorkerRange& worker : plan.workers) {
    const size_t aligned_slots = worker.span_bytes / plan.access_size_bytes;
    const size_t phase_zero_accesses =
        1 + (aligned_slots - 1) / phase_period;
    const size_t last_phase_accesses =
        last_executed_phase < aligned_slots
            ? 1 + (aligned_slots - 1 - last_executed_phase) / phase_period
            : 0;
    if (!checked_add(max_accesses_per_pass, phase_zero_accesses,
                     max_accesses_per_pass) ||
        !checked_add(min_accesses_per_pass, last_phase_accesses,
                     min_accesses_per_pass)) {
      return false;
    }
  }

  for (const PatternWorkerRange& worker : plan.workers) {
    const size_t aligned_slots = worker.span_bytes / plan.access_size_bytes;
    const size_t full_blocks = aligned_slots / phase_period;
    const size_t tail_slots = aligned_slots % phase_period;
    size_t full_block_accesses = 0;
    if (!checked_multiply(full_blocks, remaining_phases, full_block_accesses)) {
      return false;
    }
    const size_t partial_accesses = full_block_accesses +
                                    std::min(tail_slots, remaining_phases);
    if (!checked_add(total_accesses, partial_accesses, total_accesses)) {
      return false;
    }
  }

  const size_t distinct_phases = std::min(passes, phase_period);
  distinct_addresses = 0;
  logical_working_set_bytes = 0;
  for (const PatternWorkerRange& worker : plan.workers) {
    const size_t aligned_slots = worker.span_bytes / plan.access_size_bytes;
    const size_t full_blocks = aligned_slots / phase_period;
    const size_t tail_slots = aligned_slots % phase_period;
    size_t full_block_accesses = 0;
    if (!checked_multiply(full_blocks, distinct_phases, full_block_accesses)) {
      return false;
    }
    const size_t selected_tail_slots = std::min(tail_slots, distinct_phases);
    const size_t worker_distinct = full_block_accesses + selected_tail_slots;
    if (!checked_add(distinct_addresses, worker_distinct, distinct_addresses)) {
      return false;
    }

    size_t last_selected_slot = 0;
    bool has_selected_slot = false;
    if (full_blocks > 0 && distinct_phases > 0) {
      last_selected_slot = (full_blocks - 1) * phase_period + distinct_phases - 1;
      has_selected_slot = true;
    }
    if (selected_tail_slots > 0) {
      last_selected_slot = full_blocks * phase_period + selected_tail_slots - 1;
      has_selected_slot = true;
    }
    if (has_selected_slot) {
      size_t worker_logical_bytes = 0;
      if (!checked_multiply(last_selected_slot + 1, plan.access_size_bytes,
                            worker_logical_bytes) ||
          !checked_add(logical_working_set_bytes, worker_logical_bytes,
                       logical_working_set_bytes)) {
        return false;
      }
    }
  }

  return true;
}

}  // namespace

PatternWorkPlan build_strided_pattern_work_plan(size_t buffer_size, size_t stride, size_t access_size,
                                                int requested_threads, int base_passes,
                                                size_t minimum_total_payload_bytes) {
  PatternWorkPlan plan;
  plan.stride_bytes = stride;
  plan.access_size_bytes = access_size;
  plan.requested_threads = requested_threads;

  if (stride == 0 || access_size == 0 || requested_threads <= 0 || base_passes <= 0) {
    plan.status_reason = Messages::pattern_reason_invalid_work_plan_parameters();
    return plan;
  }

  size_t minimum_worker_span = 0;
  if (!checked_add(stride, access_size, minimum_worker_span)) {
    plan.status_reason = Messages::pattern_reason_stride_access_sum_overflow();
    return plan;
  }
  if (buffer_size < minimum_worker_span) {
    plan.status = PatternMeasurementStatus::Skipped;
    plan.status_reason =
        Messages::pattern_reason_buffer_lacks_two_strided_accesses();
    return plan;
  }

  const size_t maximum_workers_by_span = buffer_size / minimum_worker_span;
  const size_t maximum_int_workers = static_cast<size_t>(std::numeric_limits<int>::max());
  const int first_candidate = static_cast<int>(
      std::min({static_cast<size_t>(requested_threads), maximum_workers_by_span, maximum_int_workers}));

  bool populated = false;
  for (int candidate = first_candidate; candidate >= 1; --candidate) {
    plan.workers.clear();
    plan.effective_threads = 0;
    plan.accesses_per_pass = 0;
    plan.payload_bytes_per_pass = 0;
    if (populate_worker_ranges(plan, buffer_size, candidate)) {
      populated = true;
      break;
    }
    if (plan.status_reason == Messages::pattern_reason_work_plan_byte_overflow()) {
      return plan;
    }
  }

  if (!populated || plan.payload_bytes_per_pass == 0) {
    plan.status = PatternMeasurementStatus::Skipped;
    plan.status_reason =
        Messages::pattern_reason_no_valid_strided_worker_partition();
    return plan;
  }

  const size_t minimum_passes = static_cast<size_t>(base_passes);
  const size_t maximum_passes = static_cast<size_t>(std::numeric_limits<int>::max());
  size_t selected_passes = minimum_passes;
  if (minimum_total_payload_bytes > 0) {
    PatternWorkPlan maximum_plan = plan;
    if (!set_strided_pattern_passes(maximum_plan, maximum_passes) ||
        maximum_plan.total_payload_bytes < minimum_total_payload_bytes) {
      plan.status = PatternMeasurementStatus::Invalid;
      plan.status_reason = Messages::pattern_reason_work_plan_pass_limit();
      plan.workers.clear();
      return plan;
    }

    size_t lower = minimum_passes;
    size_t upper = maximum_passes;
    while (lower < upper) {
      const size_t middle = lower + (upper - lower) / 2;
      PatternWorkPlan candidate = plan;
      if (!set_strided_pattern_passes(candidate, middle)) {
        plan.status = PatternMeasurementStatus::Invalid;
        plan.status_reason = Messages::pattern_reason_work_plan_total_overflow();
        plan.workers.clear();
        return plan;
      }
      if (candidate.total_payload_bytes >= minimum_total_payload_bytes) {
        upper = middle;
      } else {
        lower = middle + 1;
      }
    }
    selected_passes = lower;
  }

  if (!set_strided_pattern_passes(plan, selected_passes)) {
    plan.status = PatternMeasurementStatus::Invalid;
    plan.status_reason = Messages::pattern_reason_work_plan_total_overflow();
    plan.workers.clear();
    return plan;
  }

  plan.status = PatternMeasurementStatus::Measured;
  plan.status_reason.clear();
  return plan;
}

bool set_strided_pattern_passes(PatternWorkPlan& plan, size_t passes) {
  size_t total_accesses = 0;
  size_t distinct_addresses = 0;
  size_t logical_working_set_bytes = 0;
  size_t completed_phase_cycles = 0;
  size_t min_accesses_per_pass = 0;
  size_t max_accesses_per_pass = 0;
  if (!calculate_phase_metrics(plan, passes, total_accesses, distinct_addresses,
                               logical_working_set_bytes, completed_phase_cycles,
                               min_accesses_per_pass, max_accesses_per_pass) ||
      !checked_multiply(total_accesses, plan.access_size_bytes,
                        plan.total_payload_bytes)) {
    return false;
  }

  plan.passes = passes;
  plan.total_accesses = total_accesses;
  plan.phase_period_passes = plan.stride_bytes / plan.access_size_bytes;
  plan.min_accesses_per_pass = min_accesses_per_pass;
  plan.max_accesses_per_pass = max_accesses_per_pass;
  plan.distinct_address_count = distinct_addresses;
  plan.logical_working_set_bytes = logical_working_set_bytes;
  plan.completed_phase_cycles = completed_phase_cycles;
  return true;
}

size_t calculate_pattern_pilot_passes(size_t payload_bytes_per_pass,
                                      size_t minimum_pilot_payload_bytes,
                                      size_t maximum_passes) {
  if (payload_bytes_per_pass == 0 || maximum_passes == 0) {
    return 0;
  }
  const size_t quotient = minimum_pilot_payload_bytes / payload_bytes_per_pass;
  const size_t remainder = minimum_pilot_payload_bytes % payload_bytes_per_pass;
  const size_t required = quotient + (remainder != 0 ? 1 : 0);
  return std::min(maximum_passes, std::max<size_t>(1, required));
}

size_t calculate_pattern_calibrated_passes(double pilot_duration_seconds,
                                           size_t pilot_passes,
                                           double target_duration_seconds,
                                           size_t minimum_passes,
                                           size_t maximum_passes) {
  if (!std::isfinite(pilot_duration_seconds) || pilot_duration_seconds <= 0.0 ||
      pilot_passes == 0 || !std::isfinite(target_duration_seconds) ||
      target_duration_seconds <= 0.0 || minimum_passes == 0 ||
      maximum_passes < minimum_passes) {
    return 0;
  }

  const long double scaled =
      static_cast<long double>(pilot_passes) * target_duration_seconds /
      pilot_duration_seconds;
  if (scaled >= static_cast<long double>(maximum_passes)) {
    return maximum_passes;
  }
  const size_t rounded_up = static_cast<size_t>(std::ceil(scaled));
  return std::max(minimum_passes, rounded_up);
}

const char* pattern_measurement_status_to_string(PatternMeasurementStatus status) {
  switch (status) {
    case PatternMeasurementStatus::Measured:
      return "measured";
    case PatternMeasurementStatus::Skipped:
      return "skipped";
    case PatternMeasurementStatus::Interrupted:
      return "interrupted";
    case PatternMeasurementStatus::Invalid:
      return "invalid";
  }
  return "invalid";
}

std::vector<PatternRandomWorkerIndices> build_random_worker_indices(
    size_t buffer_size, size_t access_size, int requested_threads,
    const std::vector<size_t>& global_indices) {
  if (buffer_size < access_size || access_size == 0 || requested_threads <= 0) {
    return {};
  }

  const size_t maximum_workers =
      std::min(static_cast<size_t>(requested_threads), buffer_size / access_size);
  if (maximum_workers == 0 ||
      maximum_workers > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return {};
  }

  const std::vector<size_t> boundaries =
      build_aligned_boundaries(buffer_size, static_cast<int>(maximum_workers));
  std::vector<PatternRandomWorkerIndices> workers(maximum_workers);
  for (size_t worker = 0; worker < maximum_workers; ++worker) {
    workers[worker].offset_bytes = boundaries[worker];
    workers[worker].span_bytes = boundaries[worker + 1] - boundaries[worker];
    workers[worker].indices.reserve(global_indices.size() / maximum_workers + 1);
  }

  const size_t last_valid_offset = buffer_size - access_size;
  for (size_t index : global_indices) {
    if (index > last_valid_offset) {
      continue;
    }

    const auto boundary = std::upper_bound(boundaries.begin() + 1, boundaries.end(), index);
    if (boundary == boundaries.end()) {
      continue;
    }
    const size_t worker = static_cast<size_t>(boundary - boundaries.begin() - 1);
    const size_t worker_end = boundaries[worker + 1];
    if (index > worker_end || access_size > worker_end - index) {
      continue;
    }
    workers[worker].indices.push_back(index - boundaries[worker]);
  }

  return workers;
}

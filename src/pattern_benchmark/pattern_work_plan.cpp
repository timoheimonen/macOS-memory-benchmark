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

#include <algorithm>
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
      plan.status_reason = "strided work-plan byte accounting overflow";
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

}  // namespace

PatternWorkPlan build_strided_pattern_work_plan(size_t buffer_size, size_t stride, size_t access_size,
                                                int requested_threads, int base_passes,
                                                size_t minimum_total_payload_bytes) {
  PatternWorkPlan plan;
  plan.stride_bytes = stride;
  plan.access_size_bytes = access_size;
  plan.requested_threads = requested_threads;

  if (stride == 0 || access_size == 0 || requested_threads <= 0 || base_passes <= 0) {
    plan.status_reason = "invalid strided work-plan parameters";
    return plan;
  }

  size_t minimum_worker_span = 0;
  if (!checked_add(stride, access_size, minimum_worker_span)) {
    plan.status_reason = "stride and access-size sum overflows";
    return plan;
  }
  if (buffer_size < minimum_worker_span) {
    plan.status = PatternMeasurementStatus::Skipped;
    plan.status_reason = "buffer cannot provide two strided accesses";
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
    if (plan.status_reason == "strided work-plan byte accounting overflow") {
      return plan;
    }
  }

  if (!populated || plan.payload_bytes_per_pass == 0) {
    plan.status = PatternMeasurementStatus::Skipped;
    plan.status_reason = "no valid worker partition contains a stride transition";
    return plan;
  }

  const size_t required_passes = minimum_total_payload_bytes / plan.payload_bytes_per_pass +
                                 (minimum_total_payload_bytes % plan.payload_bytes_per_pass != 0 ? 1 : 0);
  plan.passes = std::max(static_cast<size_t>(base_passes), required_passes);
  if (plan.passes > static_cast<size_t>(std::numeric_limits<int>::max())) {
    plan.status = PatternMeasurementStatus::Invalid;
    plan.status_reason = "strided work plan exceeds executor pass limit";
    plan.workers.clear();
    return plan;
  }

  if (!checked_multiply(plan.accesses_per_pass, plan.passes, plan.total_accesses) ||
      !checked_multiply(plan.payload_bytes_per_pass, plan.passes, plan.total_payload_bytes)) {
    plan.status = PatternMeasurementStatus::Invalid;
    plan.status_reason = "strided work-plan total accounting overflow";
    plan.workers.clear();
    return plan;
  }

  plan.status = PatternMeasurementStatus::Measured;
  plan.status_reason.clear();
  return plan;
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

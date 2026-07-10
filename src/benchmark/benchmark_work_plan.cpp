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
 * @file benchmark_work_plan.cpp
 * @brief Deterministic work planning and calibration for --benchmark
 */
#include "benchmark/benchmark_work_plan.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "core/config/constants.h"
#include "output/console/messages/messages_api.h"

namespace {

bool checked_multiply(size_t lhs, size_t rhs, size_t& result) {
  if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
    return false;
  }
  result = lhs * rhs;
  return true;
}

std::vector<size_t> build_boundaries(size_t size, int worker_count) {
  const size_t workers = static_cast<size_t>(worker_count);
  std::vector<size_t> boundaries(workers + 1, 0);
  const size_t base = size / workers;
  const size_t remainder = size % workers;
  size_t offset = 0;
  for (size_t worker = 0; worker < workers; ++worker) {
    offset += base + (worker < remainder ? 1 : 0);
    boundaries[worker + 1] = offset;
  }

  for (size_t index = 1; index < workers; ++index) {
    const size_t misalignment = boundaries[index] % Constants::CACHE_LINE_SIZE_BYTES;
    if (misalignment != 0) {
      const size_t adjustment = Constants::CACHE_LINE_SIZE_BYTES - misalignment;
      boundaries[index] = adjustment > size - boundaries[index]
                              ? size
                              : boundaries[index] + adjustment;
    }
    boundaries[index] = std::max(boundaries[index], boundaries[index - 1]);
  }
  boundaries.front() = 0;
  boundaries.back() = size;
  return boundaries;
}

bool populate_workers(BenchmarkWorkPlan& plan, int worker_count) {
  const std::vector<size_t> boundaries =
      build_boundaries(plan.buffer_size_bytes, worker_count);
  std::vector<BenchmarkWorkerRange> workers;
  workers.reserve(static_cast<size_t>(worker_count));
  for (int worker = 0; worker < worker_count; ++worker) {
    const size_t start = boundaries[static_cast<size_t>(worker)];
    const size_t end = boundaries[static_cast<size_t>(worker) + 1];
    if (end <= start) {
      return false;
    }
    workers.push_back({start, end - start});
  }
  plan.boundaries = boundaries;
  plan.workers = workers;
  plan.effective_threads = worker_count;
  return true;
}

size_t round_up_to_quantum(size_t value, size_t quantum) {
  if (quantum <= 1 || value == 0) {
    return value;
  }
  const size_t remainder = value % quantum;
  if (remainder == 0) {
    return value;
  }
  const size_t adjustment = quantum - remainder;
  if (value > std::numeric_limits<size_t>::max() - adjustment) {
    return 0;
  }
  return value + adjustment;
}

}  // namespace

size_t benchmark_bandwidth_state_index(BenchmarkTarget target,
                                       BenchmarkOperation operation) {
  const size_t target_index = static_cast<size_t>(target);
  const size_t operation_index = static_cast<size_t>(operation);
  return target_index * 3 + std::min<size_t>(operation_index, 2);
}

size_t benchmark_latency_state_index(BenchmarkTarget target) {
  return static_cast<size_t>(target);
}

BenchmarkWorkPlan build_benchmark_bandwidth_work_plan(
    size_t buffer_size_bytes, int requested_threads, size_t passes,
    BenchmarkTarget target, BenchmarkOperation operation) {
  BenchmarkWorkPlan plan;
  plan.buffer_size_bytes = buffer_size_bytes;
  plan.requested_threads = requested_threads;
  plan.target = target;
  plan.operation = operation;

  if (buffer_size_bytes == 0 || requested_threads <= 0 || passes == 0 ||
      operation == BenchmarkOperation::Latency) {
    plan.status_reason = Messages::benchmark_reason_invalid_bandwidth_plan();
    return plan;
  }

  const size_t maximum_workers_by_span = std::max<size_t>(
      1, buffer_size_bytes / Constants::CACHE_LINE_SIZE_BYTES);
  const int first_candidate = static_cast<int>(std::min(
      static_cast<size_t>(requested_threads), maximum_workers_by_span));
  bool populated = false;
  for (int candidate = first_candidate; candidate >= 1; --candidate) {
    if (populate_workers(plan, candidate)) {
      populated = true;
      break;
    }
  }
  if (!populated) {
    plan.status_reason = Messages::benchmark_reason_no_worker_partition();
    return plan;
  }

  plan.payload_bytes_per_pass = buffer_size_bytes;
  if (operation == BenchmarkOperation::Copy) {
    if (!checked_multiply(plan.payload_bytes_per_pass,
                          Constants::COPY_OPERATION_MULTIPLIER,
                          plan.payload_bytes_per_pass)) {
      plan.status_reason = Messages::benchmark_reason_copy_payload_overflow();
      return plan;
    }
  }
  if (!set_benchmark_work_plan_passes(plan, passes)) {
    plan.status_reason = Messages::benchmark_reason_total_payload_overflow();
    return plan;
  }
  plan.status = BenchmarkMeasurementStatus::Measured;
  plan.status_reason.clear();
  return plan;
}

bool set_benchmark_work_plan_passes(BenchmarkWorkPlan& plan, size_t passes) {
  if (passes == 0 || passes > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      plan.payload_bytes_per_pass == 0) {
    return false;
  }
  size_t total_payload = 0;
  if (!checked_multiply(plan.payload_bytes_per_pass, passes, total_payload)) {
    return false;
  }
  plan.passes = passes;
  plan.total_payload_bytes = total_payload;
  return true;
}

BenchmarkLatencyWorkPlan build_benchmark_latency_work_plan(
    size_t buffer_size_bytes, size_t stride_bytes, size_t desired_access_count,
    size_t minimum_complete_cycles, size_t maximum_access_count,
    BenchmarkTarget target, uint64_t seed) {
  BenchmarkLatencyWorkPlan plan;
  plan.buffer_size_bytes = buffer_size_bytes;
  plan.stride_bytes = stride_bytes;
  plan.target = target;
  plan.seed = seed;

  if (buffer_size_bytes == 0 || stride_bytes == 0 || minimum_complete_cycles == 0 ||
      maximum_access_count == 0) {
    plan.status_reason = Messages::benchmark_reason_invalid_latency_plan();
    return plan;
  }
  plan.chain_node_count = buffer_size_bytes / stride_bytes;
  if (plan.chain_node_count < 2) {
    plan.status = BenchmarkMeasurementStatus::Skipped;
    plan.status_reason = Messages::benchmark_reason_latency_chain_too_short();
    return plan;
  }

  size_t minimum_accesses = 0;
  if (!checked_multiply(plan.chain_node_count, minimum_complete_cycles,
                        minimum_accesses) ||
      minimum_accesses > maximum_access_count) {
    plan.status_reason =
        Messages::benchmark_reason_minimum_cycles_exceed_limit();
    return plan;
  }
  const size_t requested = std::max(desired_access_count, minimum_accesses);
  const size_t rounded = round_up_to_quantum(requested, plan.chain_node_count);
  if (rounded == 0 || rounded > maximum_access_count) {
    plan.status_reason =
        Messages::benchmark_reason_rounded_accesses_exceed_limit();
    return plan;
  }
  plan.access_count = rounded;
  plan.complete_chain_cycles = rounded / plan.chain_node_count;
  plan.status = BenchmarkMeasurementStatus::Measured;
  return plan;
}

size_t calculate_benchmark_pilot_passes(size_t payload_bytes_per_pass,
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

size_t calculate_benchmark_calibrated_count(double pilot_duration_seconds,
                                            size_t pilot_count,
                                            double target_duration_seconds,
                                            size_t minimum_count,
                                            size_t maximum_count,
                                            size_t quantum) {
  if (!std::isfinite(pilot_duration_seconds) || pilot_duration_seconds <= 0.0 ||
      pilot_count == 0 || !std::isfinite(target_duration_seconds) ||
      target_duration_seconds <= 0.0 || minimum_count == 0 ||
      maximum_count < minimum_count || quantum == 0) {
    return 0;
  }

  const long double scaled =
      static_cast<long double>(pilot_count) * target_duration_seconds /
      pilot_duration_seconds;
  size_t count = scaled >= static_cast<long double>(maximum_count)
                     ? maximum_count
                     : static_cast<size_t>(std::ceil(scaled));
  count = std::max(count, minimum_count);
  count = round_up_to_quantum(count, quantum);
  if (count == 0 || count > maximum_count) {
    const size_t maximum_quantized = maximum_count - maximum_count % quantum;
    return maximum_quantized >= minimum_count ? maximum_quantized : 0;
  }
  return count;
}

std::vector<size_t> build_benchmark_cyclic_order(size_t item_count,
                                                 size_t loop_index) {
  std::vector<size_t> order;
  order.reserve(item_count);
  if (item_count == 0) {
    return order;
  }
  const size_t start = loop_index % item_count;
  for (size_t position = 0; position < item_count; ++position) {
    order.push_back((start + position) % item_count);
  }
  return order;
}

BenchmarkPhaseExecutionResult execute_benchmark_phase_schedule(
    const std::vector<size_t>& phase_order,
    const std::function<bool()>& stop_requested,
    const std::function<void(size_t, size_t)>& execute_phase) {
  BenchmarkPhaseExecutionResult result;
  result.realized_phase_indexes.reserve(phase_order.size());
  for (size_t position = 0; position < phase_order.size(); ++position) {
    if (stop_requested && stop_requested()) {
      result.interrupted = true;
      break;
    }
    const size_t phase_index = phase_order[position];
    result.realized_phase_indexes.push_back(phase_index);
    execute_phase(position, phase_index);
    if (stop_requested && stop_requested()) {
      result.interrupted = true;
      break;
    }
    ++result.completed_phases;
  }
  return result;
}

bool benchmark_elapsed_is_valid(double elapsed) {
  return elapsed > 0.0 && std::isfinite(elapsed);
}

bool benchmark_duration_in_window(double elapsed_seconds,
                                  double minimum_seconds,
                                  double maximum_seconds) {
  return elapsed_seconds >= minimum_seconds &&
         elapsed_seconds <= maximum_seconds;
}

std::string classify_benchmark_duration_quality(
    double elapsed_seconds, size_t count, double minimum_seconds,
    double maximum_seconds, bool minimum_work_limited) {
  if (benchmark_duration_in_window(elapsed_seconds, minimum_seconds,
                                   maximum_seconds)) {
    return "within-target-window";
  }
  if (minimum_work_limited && elapsed_seconds > maximum_seconds) {
    return "minimum-complete-cycles-exceed-window";
  }
  if (count == 1 && elapsed_seconds > maximum_seconds) {
    return "single-pass-exceeds-window";
  }
  return elapsed_seconds < minimum_seconds ? "below-target-window"
                                           : "above-target-window";
}

uint64_t derive_benchmark_seed(uint64_t base_seed, uint64_t domain) {
  uint64_t value = base_seed ^ (domain + 0x9e3779b97f4a7c15ULL);
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  value ^= value >> 31U;
  return value != 0 ? value : 0x9e3779b97f4a7c15ULL;
}

const char* benchmark_target_to_string(BenchmarkTarget target) {
  switch (target) {
    case BenchmarkTarget::MainMemory:
      return "main-memory";
    case BenchmarkTarget::L1:
      return "l1";
    case BenchmarkTarget::L2:
      return "l2";
    case BenchmarkTarget::Custom:
      return "custom";
  }
  return "main-memory";
}

const char* benchmark_operation_to_string(BenchmarkOperation operation) {
  switch (operation) {
    case BenchmarkOperation::Read:
      return "read";
    case BenchmarkOperation::Write:
      return "write";
    case BenchmarkOperation::Copy:
      return "copy";
    case BenchmarkOperation::Latency:
      return "latency";
  }
  return "read";
}

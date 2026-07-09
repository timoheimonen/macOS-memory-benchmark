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
 * @file tlb_runtime_policy.cpp
 * @brief Deterministic workload, convergence, and memory policy for TLB analysis
 */

#include "benchmark/tlb_runtime_policy.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "core/config/constants.h"

namespace {

constexpr size_t kPilotAccessFloor = 4096;
constexpr size_t kPilotMinimumCycles = 2;
constexpr double kBootstrapConfidenceLevel = 0.95;
constexpr double kTlbMemoryBudgetFraction = 0.30;
constexpr size_t kTlbMemoryReserveMb = 1024;
constexpr size_t kFallbackMemoryBudgetMb = 384;
constexpr size_t kScratchFixedOverheadBytes = Constants::BYTES_PER_MB;
constexpr size_t kScratchBytesPerNode = 256;
constexpr double kDurationOverheadFactor = 1.25;

size_t saturating_multiply(size_t left, size_t right) {
  if (left == 0 || right == 0) {
    return 0;
  }
  if (left > std::numeric_limits<size_t>::max() / right) {
    return std::numeric_limits<size_t>::max();
  }
  return left * right;
}

size_t saturating_add(size_t left, size_t right) {
  return left > std::numeric_limits<size_t>::max() - right
             ? std::numeric_limits<size_t>::max()
             : left + right;
}

size_t round_up_to_multiple(size_t value, size_t multiple) {
  if (multiple == 0) {
    return 0;
  }
  const size_t remainder = value % multiple;
  if (remainder == 0) {
    return value;
  }
  const size_t increment = multiple - remainder;
  return value > std::numeric_limits<size_t>::max() - increment
             ? std::numeric_limits<size_t>::max()
             : value + increment;
}

double median_in_place(std::vector<double>& values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const size_t midpoint = values.size() / 2;
  if ((values.size() % 2) == 0) {
    return 0.5 * (values[midpoint - 1] + values[midpoint]);
  }
  return values[midpoint];
}

double percentile(const std::vector<double>& sorted_values,
                  double probability) {
  if (sorted_values.empty()) {
    return 0.0;
  }
  const double bounded_probability = std::clamp(probability, 0.0, 1.0);
  const size_t index = static_cast<size_t>(
      bounded_probability * static_cast<double>(sorted_values.size() - 1));
  return sorted_values[index];
}

double bootstrap_median_ci_width(const std::vector<double>& samples,
                                 size_t resamples,
                                 uint64_t seed,
                                 TlbConvergenceScratch& scratch) {
  if (samples.empty() || resamples == 0) {
    return std::numeric_limits<double>::infinity();
  }

  std::mt19937_64 random(seed);
  std::uniform_int_distribution<size_t> sample_index(0, samples.size() - 1);
  scratch.resample.resize(samples.size());
  scratch.medians.clear();
  if (scratch.medians.capacity() < resamples) {
    scratch.medians.reserve(resamples);
  }
  for (size_t iteration = 0; iteration < resamples; ++iteration) {
    for (double& value : scratch.resample) {
      value = samples[sample_index(random)];
    }
    scratch.medians.push_back(median_in_place(scratch.resample));
  }
  std::sort(scratch.medians.begin(), scratch.medians.end());
  const double tail = (1.0 - kBootstrapConfidenceLevel) / 2.0;
  return percentile(scratch.medians, 1.0 - tail) -
         percentile(scratch.medians, tail);
}

}  // namespace

TlbRuntimeProfile tlb_runtime_profile_for_density(TlbSweepDensity density) {
  switch (density) {
    case TlbSweepDensity::Low:
      return {"quick", 7, 12, 5 * 1000 * 1000ULL, 8, 1 * 1000 * 1000,
              0.50, 400};
    case TlbSweepDensity::Medium:
      return {"standard", 10, 20, 10 * 1000 * 1000ULL, 16,
              2 * 1000 * 1000, 0.30, 600};
    case TlbSweepDensity::High:
      return {"exhaustive", 15, 30, 20 * 1000 * 1000ULL, 32,
              5 * 1000 * 1000, 0.15, 1000};
  }
  return {"standard", 10, 20, 10 * 1000 * 1000ULL, 16,
          2 * 1000 * 1000, 0.30, 600};
}

size_t calculate_tlb_pilot_accesses(size_t node_count) {
  if (node_count == 0 ||
      node_count > std::numeric_limits<size_t>::max() /
                       kPilotMinimumCycles) {
    return 0;
  }
  const size_t minimum_cycle_accesses =
      saturating_multiply(node_count, kPilotMinimumCycles);
  return round_up_to_multiple(
      std::max(kPilotAccessFloor, minimum_cycle_accesses), node_count);
}

size_t calculate_tlb_calibrated_accesses(
    size_t node_count,
    size_t pilot_accesses,
    double pilot_duration_ns,
    const TlbRuntimeProfile& profile) {
  if (node_count == 0 || pilot_accesses == 0 ||
      !std::isfinite(pilot_duration_ns) || pilot_duration_ns <= 0.0 ||
      profile.target_measurement_ns == 0 ||
      profile.minimum_chain_cycles == 0) {
    return 0;
  }
  if (node_count > std::numeric_limits<size_t>::max() /
                       profile.minimum_chain_cycles) {
    return 0;
  }

  const size_t minimum_accesses = saturating_multiply(
      node_count, profile.minimum_chain_cycles);
  const double ns_per_access =
      pilot_duration_ns / static_cast<double>(pilot_accesses);
  const double desired_double =
      static_cast<double>(profile.target_measurement_ns) / ns_per_access;
  size_t desired_accesses = desired_double >=
                                    static_cast<double>(
                                        std::numeric_limits<size_t>::max())
                                ? std::numeric_limits<size_t>::max()
                                : static_cast<size_t>(std::ceil(desired_double));
  const size_t effective_maximum =
      std::max(profile.maximum_accesses, minimum_accesses);
  desired_accesses =
      std::clamp(desired_accesses, minimum_accesses, effective_maximum);
  const size_t rounded = round_up_to_multiple(desired_accesses, node_count);
  return std::min(rounded, effective_maximum - (effective_maximum % node_count));
}

TlbConvergenceSummary evaluate_tlb_convergence(
    const std::vector<std::vector<double>>& samples_by_point,
    const TlbRuntimeProfile& profile,
    uint64_t bootstrap_seed,
    TlbConvergenceScratch* scratch) {
  TlbConvergenceSummary summary;
  if (samples_by_point.empty() || profile.min_rounds == 0 ||
      profile.ci_width_target_ns <= 0.0 ||
      profile.convergence_bootstrap_resamples == 0) {
    return summary;
  }

  TlbConvergenceScratch local_scratch;
  TlbConvergenceScratch& reusable_scratch =
      scratch == nullptr ? local_scratch : *scratch;
  summary.converged = true;
  for (size_t point_index = 0; point_index < samples_by_point.size();
       ++point_index) {
    const std::vector<double>& samples = samples_by_point[point_index];
    if (samples.size() < profile.min_rounds) {
      summary.converged = false;
      continue;
    }
    const double width = bootstrap_median_ci_width(
        samples,
        profile.convergence_bootstrap_resamples,
        bootstrap_seed ^ static_cast<uint64_t>(point_index),
        reusable_scratch);
    summary.maximum_ci_width_ns =
        std::max(summary.maximum_ci_width_ns, width);
    ++summary.evaluated_points;
    if (!std::isfinite(width) || width > profile.ci_width_target_ns) {
      summary.converged = false;
    }
  }
  if (summary.evaluated_points != samples_by_point.size()) {
    summary.converged = false;
  }
  return summary;
}

size_t calculate_tlb_memory_budget_mb(size_t available_memory_mb) {
  if (available_memory_mb == 0) {
    return kFallbackMemoryBudgetMb;
  }
  const size_t fractional_budget = static_cast<size_t>(
      static_cast<double>(available_memory_mb) * kTlbMemoryBudgetFraction);
  const size_t reserve_budget = available_memory_mb > kTlbMemoryReserveMb
                                    ? available_memory_mb - kTlbMemoryReserveMb
                                    : available_memory_mb / 2;
  return std::min(fractional_budget, reserve_budget);
}

size_t estimate_tlb_scratch_bytes(size_t maximum_node_count) {
  const size_t variable_bytes =
      saturating_multiply(maximum_node_count, kScratchBytesPerNode);
  if (variable_bytes > std::numeric_limits<size_t>::max() -
                           kScratchFixedOverheadBytes) {
    return std::numeric_limits<size_t>::max();
  }
  return variable_bytes + kScratchFixedOverheadBytes;
}

size_t estimate_tlb_peak_memory_bytes(size_t buffer_size_bytes,
                                      size_t maximum_node_count) {
  const size_t scratch_bytes = estimate_tlb_scratch_bytes(maximum_node_count);
  if (buffer_size_bytes >
      std::numeric_limits<size_t>::max() - scratch_bytes) {
    return std::numeric_limits<size_t>::max();
  }
  return buffer_size_bytes + scratch_bytes;
}

bool tlb_buffer_fits_memory_budget(size_t candidate_buffer_mb,
                                   size_t page_size_bytes,
                                   size_t memory_budget_mb,
                                   size_t& estimated_peak_memory_bytes) {
  estimated_peak_memory_bytes = 0;
  if (candidate_buffer_mb == 0 || page_size_bytes == 0 ||
      candidate_buffer_mb >
          std::numeric_limits<size_t>::max() / Constants::BYTES_PER_MB) {
    return false;
  }
  const size_t candidate_bytes =
      candidate_buffer_mb * Constants::BYTES_PER_MB;
  const size_t maximum_node_count = candidate_bytes / page_size_bytes;
  estimated_peak_memory_bytes =
      estimate_tlb_peak_memory_bytes(candidate_bytes, maximum_node_count);
  const size_t budget_bytes =
      memory_budget_mb >
              std::numeric_limits<size_t>::max() / Constants::BYTES_PER_MB
          ? std::numeric_limits<size_t>::max()
          : memory_budget_mb * Constants::BYTES_PER_MB;
  return estimated_peak_memory_bytes <= budget_bytes;
}

TlbWorkEstimate estimate_tlb_work(size_t point_count,
                                  size_t estimated_peak_memory_bytes,
                                  size_t maximum_node_count,
                                  const TlbRuntimeProfile& profile) {
  TlbWorkEstimate estimate;
  estimate.point_count = point_count;
  estimate.min_rounds = profile.min_rounds;
  estimate.max_rounds = profile.max_rounds;
  estimate.estimated_peak_memory_bytes = estimated_peak_memory_bytes;
  estimate.maximum_pilot_accesses_per_measurement =
      calculate_tlb_pilot_accesses(maximum_node_count);
  estimate.maximum_accesses_per_measurement = std::max(
      profile.maximum_accesses,
      saturating_multiply(maximum_node_count, profile.minimum_chain_cycles));
  const size_t maximum_combined_accesses = saturating_add(
      estimate.maximum_pilot_accesses_per_measurement,
      estimate.maximum_accesses_per_measurement);
  estimate.maximum_pointer_accesses = saturating_multiply(
      saturating_multiply(
          saturating_multiply(point_count, profile.max_rounds), 2),
      maximum_combined_accesses);
  const double measurement_sec =
      static_cast<double>(profile.target_measurement_ns) / 1.0e9;
  estimate.estimated_min_duration_sec =
      static_cast<double>(point_count) *
      static_cast<double>(profile.min_rounds) * 2.0 *
      measurement_sec * kDurationOverheadFactor;
  estimate.estimated_max_duration_sec =
      static_cast<double>(point_count) *
      static_cast<double>(profile.max_rounds) * 2.0 *
      measurement_sec * kDurationOverheadFactor;
  return estimate;
}

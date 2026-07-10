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
 * @file tlb_runtime_policy.h
 * @brief Deterministic workload, convergence, and memory policy for TLB analysis
 */

#ifndef TLB_RUNTIME_POLICY_H
#define TLB_RUNTIME_POLICY_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "core/config/config.h"

struct TlbRuntimeProfile {
  std::string name;
  size_t min_rounds = 0;
  size_t max_rounds = 0;
  uint64_t target_measurement_ns = 0;
  size_t minimum_chain_cycles = 0;
  size_t maximum_accesses = 0;
  double ci_width_target_ns = 0.0;
  size_t convergence_bootstrap_resamples = 0;
};

struct TlbConvergenceSummary {
  bool converged = false;
  size_t evaluated_points = 0;
  double maximum_ci_width_ns = 0.0;
};

struct TlbConvergenceScratch {
  std::vector<double> resample;
  std::vector<double> medians;
};

struct TlbWorkEstimate {
  size_t point_count = 0;
  size_t min_rounds = 0;
  size_t max_rounds = 0;
  size_t maximum_pilot_accesses_per_measurement = 0;
  size_t maximum_accesses_per_measurement = 0;
  size_t maximum_pointer_accesses = 0;
  size_t estimated_peak_memory_bytes = 0;
  double estimated_min_duration_sec = 0.0;
  double estimated_max_duration_sec = 0.0;
};

/** Resolve the quick, standard, or exhaustive runtime policy. */
TlbRuntimeProfile tlb_runtime_profile_for_density(TlbSweepDensity density);

/** Select a short pilot that still traverses whole chain cycles. */
size_t calculate_tlb_pilot_accesses(size_t node_count);

/** Calibrate the main timed access count from one pilot measurement. */
size_t calculate_tlb_calibrated_accesses(
    size_t node_count,
    size_t pilot_accesses,
    double pilot_duration_ns,
    const TlbRuntimeProfile& profile);

/** Check whether every point's deterministic bootstrap median CI is narrow enough. */
TlbConvergenceSummary evaluate_tlb_convergence(
    const std::vector<std::vector<double>>& samples_by_point,
    const TlbRuntimeProfile& profile,
    uint64_t bootstrap_seed,
    TlbConvergenceScratch* scratch = nullptr);

/** Derive the conservative TLB allocation budget from currently available memory. */
size_t calculate_tlb_memory_budget_mb(size_t available_memory_mb);

/** Estimate retained chain-builder and validator scratch storage. */
size_t estimate_tlb_scratch_bytes(size_t maximum_node_count);

/** Estimate the buffer plus retained scratch peak. */
size_t estimate_tlb_peak_memory_bytes(size_t buffer_size_bytes,
                                      size_t maximum_node_count);

/** Check a buffer candidate against the memory budget before mmap(). */
bool tlb_buffer_fits_memory_budget(size_t candidate_buffer_mb,
                                   size_t page_size_bytes,
                                   size_t memory_budget_mb,
                                   size_t& estimated_peak_memory_bytes);

/** Build a conservative paired-measurement work and timed-window estimate. */
TlbWorkEstimate estimate_tlb_work(size_t point_count,
                                  size_t estimated_peak_memory_bytes,
                                  size_t maximum_node_count,
                                  const TlbRuntimeProfile& profile);

#endif  // TLB_RUNTIME_POLICY_H

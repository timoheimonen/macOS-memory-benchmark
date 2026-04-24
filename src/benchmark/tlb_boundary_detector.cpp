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
 * @file tlb_boundary_detector.cpp
 * @brief Boundary detection helpers for standalone TLB analysis mode
 */

#include "benchmark/tlb_analysis.h"

#include "core/config/constants.h"

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr double kRelativeThreshold = 0.10;
constexpr double kAbsoluteThresholdNs = 2.0;

// Multi-point persistence: check up to 3 future points, require majority.
constexpr size_t kPersistenceWindowSize = 3;
constexpr size_t kPersistenceMajorityRequired = 2;

// Last-point strong-step compensation thresholds (when no future points exist).
constexpr double kLastPointStrongStepNs = 8.0;
constexpr double kLastPointStrongStepPercent = 0.25;

constexpr size_t kPrivateCacheKneeMinBytes = 512 * Constants::BYTES_PER_KB;
constexpr size_t kStrongPrivateCacheKneeMinBytes = 768 * Constants::BYTES_PER_KB;
constexpr size_t kPrivateCacheKneeMaxBytes = 2 * Constants::BYTES_PER_MB;

/**
 * @brief Recency-weighted average over a range.
 *
 * Recent points receive higher weight: point j gets weight (j - start + 1).
 * This reduces the drag from early measurements that may have had different
 * thermal or frequency conditions.
 */
double recency_weighted_average(const std::vector<double>& values, size_t start, size_t end) {
  if (start >= end || end > values.size()) {
    return 0.0;
  }

  double weighted_sum = 0.0;
  double weight_total = 0.0;
  for (size_t j = start; j < end; ++j) {
    const double w = static_cast<double>(j - start + 1);
    weighted_sum += values[j] * w;
    weight_total += w;
  }
  return (weight_total > 0.0) ? (weighted_sum / weight_total) : 0.0;
}

/**
 * @brief Compute Q1 and Q3 from a vector of raw loop values.
 *
 * @param values Raw loop latencies (will be sorted internally via copy)
 * @param[out] q1 First quartile
 * @param[out] q3 Third quartile
 */
void compute_quartiles(const std::vector<double>& values, double& q1, double& q3) {
  if (values.size() < 4) {
    q1 = 0.0;
    q3 = 0.0;
    return;
  }

  std::vector<double> sorted = values;
  std::sort(sorted.begin(), sorted.end());

  const size_t n = sorted.size();
  const size_t q1_idx = n / 4;
  const size_t q3_idx = (3 * n) / 4;

  q1 = sorted[q1_idx];
  q3 = sorted[q3_idx];
}

/**
 * @brief Compute the average Q3 across all points in [start, end) from loop_latencies.
 *
 * This represents the upper-noise bound of the baseline segment.
 */
double average_baseline_q3(const std::vector<std::vector<double>>& loop_latencies,
                           size_t start,
                           size_t end) {
  if (start >= end || end > loop_latencies.size()) {
    return 0.0;
  }

  double sum_q3 = 0.0;
  for (size_t j = start; j < end; ++j) {
    double q1_unused = 0.0;
    double q3 = 0.0;
    compute_quartiles(loop_latencies[j], q1_unused, q3);
    sum_q3 += q3;
  }
  return sum_q3 / static_cast<double>(end - start);
}

}  // namespace

std::string classify_tlb_confidence(double step_ns, double step_percent, bool persistent_jump) {
  const bool strong_step = (step_ns >= 4.0) || (step_percent >= 0.15);

  if (strong_step && persistent_jump) {
    return "High";
  }
  if (strong_step || persistent_jump) {
    return "Medium";
  }
  return "Low";
}

size_t infer_tlb_entries(size_t locality_bytes, size_t page_size_bytes) {
  if (page_size_bytes == 0) {
    return 0;
  }
  return locality_bytes / page_size_bytes;
}

std::pair<size_t, size_t> infer_tlb_entries_range(const std::vector<size_t>& locality_bytes,
                                                  size_t boundary_index,
                                                  size_t page_size_bytes) {
  if (page_size_bytes == 0 || locality_bytes.empty() || boundary_index >= locality_bytes.size()) {
    return {0, 0};
  }

  const size_t lower_locality_bytes = (boundary_index > 0)
                                           ? locality_bytes[boundary_index - 1]
                                           : locality_bytes[boundary_index];
  const size_t upper_locality_bytes = locality_bytes[boundary_index];
  return {lower_locality_bytes / page_size_bytes, upper_locality_bytes / page_size_bytes};
}

TlbBoundaryDetection detect_tlb_boundary(const std::vector<size_t>& locality_bytes,
                                         const std::vector<double>& p50_latency_ns,
                                         size_t segment_start_index,
                                         size_t min_locality_bytes,
                                         const std::vector<std::vector<double>>* loop_latencies) {
  TlbBoundaryDetection result;
  result.segment_start_index = segment_start_index;

  if (locality_bytes.size() != p50_latency_ns.size() ||
      p50_latency_ns.size() < 2 ||
      segment_start_index >= p50_latency_ns.size() - 1) {
    return result;
  }

  const bool has_loop_data = (loop_latencies != nullptr) &&
                             (loop_latencies->size() == p50_latency_ns.size());

  for (size_t i = segment_start_index + 1; i < p50_latency_ns.size(); ++i) {
    const double baseline_ns = recency_weighted_average(p50_latency_ns, segment_start_index, i);
    const double boundary_ns = p50_latency_ns[i];
    const double step_ns = boundary_ns - baseline_ns;
    const double step_percent = (baseline_ns > 0.0) ? (step_ns / baseline_ns) : 0.0;
    const double threshold_ns = std::max(kAbsoluteThresholdNs, baseline_ns * kRelativeThreshold);

    if (step_ns < threshold_ns) {
      continue;
    }

    if (locality_bytes[i] < min_locality_bytes) {
      continue;
    }

    // IQR-overlap rejection: if the baseline's upper noise (avg Q3) overlaps with the
    // candidate's lower quartile (Q1), the step is within measurement noise.
    if (has_loop_data) {
      double candidate_q1 = 0.0;
      double candidate_q3_unused = 0.0;
      compute_quartiles((*loop_latencies)[i], candidate_q1, candidate_q3_unused);

      const double baseline_q3 =
          average_baseline_q3(*loop_latencies, segment_start_index, i);

      if (baseline_q3 > 0.0 && candidate_q1 > 0.0 && baseline_q3 >= candidate_q1) {
        continue;
      }
    }

    // Multi-point persistence: check up to kPersistenceWindowSize future points.
    size_t persistent_count = 0;
    const size_t window_end =
        std::min(i + 1 + kPersistenceWindowSize, p50_latency_ns.size());
    for (size_t j = i + 1; j < window_end; ++j) {
      const double future_step_ns = p50_latency_ns[j] - baseline_ns;
      if (future_step_ns >= threshold_ns) {
        ++persistent_count;
      }
    }
    const bool persistent_jump = persistent_count >= kPersistenceMajorityRequired;

    // At the last sweep point(s), compensate for lack of future data:
    // if the step itself is very strong, treat it as persistent.
    const bool is_last_point = (i + 1 >= p50_latency_ns.size());
    const bool strong_last_point =
        is_last_point &&
        ((step_ns >= kLastPointStrongStepNs) || (step_percent >= kLastPointStrongStepPercent));

    const bool effective_persistent = persistent_jump || strong_last_point;

    result.detected = true;
    result.boundary_index = i;
    result.boundary_locality_bytes = locality_bytes[i];
    result.baseline_ns = baseline_ns;
    result.boundary_latency_ns = boundary_ns;
    result.step_ns = step_ns;
    result.step_percent = step_percent;
    result.persistent_jump = effective_persistent;
    result.confidence = classify_tlb_confidence(step_ns, step_percent, effective_persistent);
    return result;
  }

  return result;
}

PrivateCacheKneeDetection detect_private_cache_knee(
    const std::vector<size_t>& locality_bytes,
    const std::vector<double>& p50_latency_ns,
    const std::vector<std::vector<double>>* loop_latencies) {
  PrivateCacheKneeDetection result;
  const TlbBoundaryDetection boundary =
      detect_tlb_boundary(locality_bytes, p50_latency_ns, 0, kPrivateCacheKneeMinBytes, loop_latencies);
  if (!boundary.detected) {
    return result;
  }

  if (boundary.boundary_locality_bytes > kPrivateCacheKneeMaxBytes) {
    return result;
  }

  result.detected = true;
  result.boundary_index = boundary.boundary_index;
  result.boundary_locality_bytes = boundary.boundary_locality_bytes;
  result.step_ns = boundary.step_ns;
  result.step_percent = boundary.step_percent;
  result.confidence = boundary.confidence;
  result.strong_private_cache_candidate =
      boundary.boundary_locality_bytes >= kStrongPrivateCacheKneeMinBytes;
  result.early_cache_candidate = !result.strong_private_cache_candidate;
  result.may_interfere_with_tlb = result.strong_private_cache_candidate;
  return result;
}

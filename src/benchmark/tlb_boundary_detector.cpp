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
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr double kRelativeThreshold = 0.10;
constexpr double kAbsoluteThresholdNs = 2.0;

// Multiplier applied to the median baseline IQR for adaptive noise gating.
// On quiet systems the baseline IQR is small (<0.5 ns), so this rarely
// raises the threshold above the hardcoded floor. On noisy/congested
// systems it automatically raises the detection bar to suppress spurious
// boundary claims.
constexpr double kNoiseMultiplier = 1.0;

// Multi-point persistence: check up to 3 future points, require majority.
constexpr size_t kPersistenceWindowSize = 3;
constexpr size_t kPersistenceMajorityRequired = 2;

// Last-point strong-step compensation thresholds (when no future points exist).
constexpr double kLastPointStrongStepNs = 8.0;
constexpr double kLastPointStrongStepPercent = 0.25;

constexpr double kRobustMinimumEffectNs = 0.5;
constexpr double kRobustMinimumNoiseFloorNs = 0.1;
constexpr double kRobustNoiseMultiplier = 1.5;
constexpr size_t kRobustPersistencePoints = 2;
constexpr size_t kRobustMinimumPairedSamples = 7;
constexpr size_t kBootstrapResamples = 2000;
constexpr double kBootstrapConfidenceLevel = 0.95;

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

/**
 * @brief Estimate typical baseline noise as the median IQR across baseline points.
 *
 * For each baseline point j in [start, end), computes the IQR of its raw loop
 * latencies (when available). Returns the median of those IQRs as a robust
 * noise-floor estimate. Requires at least 3 baseline points; otherwise
 * returns 0.0 to fall back to hardcoded thresholds.
 */
double estimate_baseline_iqr_noise(const std::vector<std::vector<double>>& loop_latencies,
                                   size_t start,
                                   size_t end) {
  if (end < start + 3 || end > loop_latencies.size()) {
    return 0.0;
  }

  std::vector<double> iqrs;
  iqrs.reserve(end - start);
  for (size_t j = start; j < end; ++j) {
    double q1 = 0.0;
    double q3 = 0.0;
    compute_quartiles(loop_latencies[j], q1, q3);
    if (q3 > q1) {
      iqrs.push_back(q3 - q1);
    }
  }

  if (iqrs.empty()) {
    return 0.0;
  }

  std::sort(iqrs.begin(), iqrs.end());
  const size_t mid = iqrs.size() / 2;
  if ((iqrs.size() % 2) == 0) {
    return 0.5 * (iqrs[mid - 1] + iqrs[mid]);
  }
  return iqrs[mid];
}

double median_value(std::vector<double> values) {
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

std::vector<double> paired_point_effects(const TlbRoundPointMatrix& matrix,
                                         size_t baseline_index,
                                         size_t target_index) {
  std::vector<double> effects;
  effects.reserve(matrix.size());
  for (const std::vector<double>& round : matrix) {
    if (baseline_index >= round.size() || target_index >= round.size()) {
      continue;
    }
    const double baseline = round[baseline_index];
    const double target = round[target_index];
    if (std::isfinite(baseline) && std::isfinite(target)) {
      effects.push_back(target - baseline);
    }
  }
  return effects;
}

double matrix_column_median(const TlbRoundPointMatrix& matrix, size_t column) {
  std::vector<double> values;
  values.reserve(matrix.size());
  for (const std::vector<double>& round : matrix) {
    if (column < round.size() && std::isfinite(round[column])) {
      values.push_back(round[column]);
    }
  }
  return median_value(std::move(values));
}

double percentile(const std::vector<double>& sorted_values, double probability) {
  if (sorted_values.empty()) {
    return 0.0;
  }
  const double bounded_probability = std::clamp(probability, 0.0, 1.0);
  const size_t index = static_cast<size_t>(
      bounded_probability * static_cast<double>(sorted_values.size() - 1));
  return sorted_values[index];
}

TlbBootstrapInterval bootstrap_median_interval(const std::vector<double>& effects,
                                               uint64_t seed) {
  TlbBootstrapInterval interval;
  interval.paired_sample_count = effects.size();
  interval.bootstrap_resamples = kBootstrapResamples;
  interval.confidence_level = kBootstrapConfidenceLevel;
  if (effects.empty()) {
    return interval;
  }

  std::mt19937_64 random(seed);
  std::uniform_int_distribution<size_t> sample_index(0, effects.size() - 1);
  std::vector<double> sample(effects.size());
  std::vector<double> bootstrap_medians;
  bootstrap_medians.reserve(kBootstrapResamples);
  for (size_t iteration = 0; iteration < kBootstrapResamples; ++iteration) {
    for (double& value : sample) {
      value = effects[sample_index(random)];
    }
    bootstrap_medians.push_back(median_value(sample));
  }
  std::sort(bootstrap_medians.begin(), bootstrap_medians.end());
  const double tail = (1.0 - kBootstrapConfidenceLevel) / 2.0;
  interval.lower_ns = percentile(bootstrap_medians, tail);
  interval.upper_ns = percentile(bootstrap_medians, 1.0 - tail);
  return interval;
}

double estimate_robust_noise_floor(const TlbRoundPointMatrix& matrix,
                                   size_t segment_start_index,
                                   size_t boundary_index) {
  std::vector<double> absolute_effects;
  for (size_t index = segment_start_index + 1; index < boundary_index; ++index) {
    const std::vector<double> effects = paired_point_effects(matrix, index - 1, index);
    for (double effect : effects) {
      absolute_effects.push_back(std::abs(effect));
    }
  }
  return std::max(kRobustMinimumNoiseFloorNs,
                  kRobustNoiseMultiplier * median_value(std::move(absolute_effects)));
}

TlbBoundaryEvidence evaluate_robust_evidence(const TlbRoundPointMatrix& matrix,
                                             size_t point_count,
                                             size_t segment_start_index,
                                             size_t boundary_index,
                                             uint64_t bootstrap_seed) {
  TlbBoundaryEvidence evidence;
  evidence.minimum_effect_ns = kRobustMinimumEffectNs;
  evidence.persistence_points_required = kRobustPersistencePoints;
  if (boundary_index == 0 || boundary_index >= point_count) {
    evidence.rejection_reason = "invalid-boundary-index";
    return evidence;
  }

  const std::vector<double> candidate_effects =
      paired_point_effects(matrix, boundary_index - 1, boundary_index);
  evidence.effect_ns = median_value(candidate_effects);
  evidence.effect_ci = bootstrap_median_interval(candidate_effects, bootstrap_seed);
  evidence.noise_floor_ns =
      estimate_robust_noise_floor(matrix, segment_start_index, boundary_index);
  evidence.available = candidate_effects.size() >= kRobustMinimumPairedSamples;
  if (!evidence.available) {
    evidence.rejection_reason = "insufficient-paired-samples";
    return evidence;
  }
  if (evidence.effect_ns < evidence.minimum_effect_ns) {
    evidence.rejection_reason = "effect-below-minimum";
    return evidence;
  }
  if (evidence.effect_ci.lower_ns <= evidence.noise_floor_ns) {
    evidence.rejection_reason = "confidence-interval-overlaps-noise-floor";
    return evidence;
  }
  if (boundary_index + kRobustPersistencePoints >= point_count) {
    evidence.rejection_reason = "insufficient-persistence-points";
    return evidence;
  }

  for (size_t offset = 1; offset <= kRobustPersistencePoints; ++offset) {
    const std::vector<double> persistence_effects =
        paired_point_effects(matrix, boundary_index - 1, boundary_index + offset);
    if (persistence_effects.size() < kRobustMinimumPairedSamples) {
      continue;
    }
    const TlbBootstrapInterval persistence_ci = bootstrap_median_interval(
        persistence_effects,
        bootstrap_seed ^ (0x9e3779b97f4a7c15ULL * offset));
    if (median_value(persistence_effects) >= evidence.minimum_effect_ns &&
        persistence_ci.lower_ns > evidence.noise_floor_ns) {
      ++evidence.persistence_points_passed;
    }
  }
  if (evidence.persistence_points_passed != kRobustPersistencePoints) {
    evidence.rejection_reason = "persistence-not-confirmed";
    return evidence;
  }

  evidence.passed = true;
  return evidence;
}

bool includes_pass(const std::vector<TlbMeasurementPass>& passes,
                   TlbMeasurementPass pass) {
  return std::find(passes.begin(), passes.end(), pass) != passes.end();
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

size_t infer_tlb_entries_estimate(const std::vector<size_t>& locality_bytes,
                                  size_t boundary_index,
                                  size_t page_size_bytes) {
  const std::pair<size_t, size_t> range =
      infer_tlb_entries_range(locality_bytes, boundary_index, page_size_bytes);
  if (range.first == 0 && range.second == 0) {
    return 0;
  }
  return range.first + ((range.second - range.first) / 2);
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
    double noise_boost = 0.0;
    if (has_loop_data && i > segment_start_index + 2) {
      noise_boost = kNoiseMultiplier *
                    estimate_baseline_iqr_noise(*loop_latencies, segment_start_index, i);
    }
    const double threshold_ns =
        std::max({kAbsoluteThresholdNs, baseline_ns * kRelativeThreshold, noise_boost});

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

TlbRoundPointMatrix build_tlb_translation_delta_matrix(
    const std::vector<size_t>& locality_bytes,
    const std::vector<TlbMeasurementRecord>& records,
    const std::vector<TlbMeasurementPass>& included_passes) {
  size_t round_count = 0;
  for (const TlbMeasurementRecord& record : records) {
    if (record.paired.available && includes_pass(included_passes, record.pass)) {
      round_count = std::max(round_count, record.round_index + 1);
    }
  }

  TlbRoundPointMatrix matrix(
      round_count,
      std::vector<double>(locality_bytes.size(),
                          std::numeric_limits<double>::quiet_NaN()));
  for (const TlbMeasurementRecord& record : records) {
    if (!record.paired.available || !includes_pass(included_passes, record.pass) ||
        record.round_index >= matrix.size()) {
      continue;
    }
    const auto locality = std::lower_bound(locality_bytes.begin(),
                                           locality_bytes.end(),
                                           record.locality_bytes);
    if (locality == locality_bytes.end() || *locality != record.locality_bytes) {
      continue;
    }
    const size_t point_index = static_cast<size_t>(
        std::distance(locality_bytes.begin(), locality));
    matrix[record.round_index][point_index] =
        record.paired.translation_delta_ns;
  }
  return matrix;
}

TlbBoundaryDetection detect_tlb_boundary_robust(
    const std::vector<size_t>& locality_bytes,
    const TlbRoundPointMatrix& discovery_matrix,
    const TlbRoundPointMatrix* validation_matrix,
    size_t segment_start_index,
    size_t min_locality_bytes,
    uint64_t bootstrap_seed) {
  TlbBoundaryDetection result;
  result.segment_start_index = segment_start_index;
  if (locality_bytes.size() < 2 ||
      segment_start_index >= locality_bytes.size() - 1) {
    return result;
  }

  bool selected_discovery_candidate = false;
  for (size_t index = segment_start_index + 1;
       index < locality_bytes.size(); ++index) {
    if (locality_bytes[index] < min_locality_bytes) {
      continue;
    }

    TlbBoundaryCandidate candidate;
    candidate.boundary_index = index;
    candidate.boundary_locality_bytes = locality_bytes[index];
    candidate.bracket_lower_bytes = locality_bytes[index - 1];
    candidate.bracket_upper_bytes = locality_bytes[index];
    candidate.discovery = evaluate_robust_evidence(
        discovery_matrix,
        locality_bytes.size(),
        segment_start_index,
        index,
        bootstrap_seed ^ static_cast<uint64_t>(index));

    // Flat points are not candidates. Material effects rejected by CI,
    // persistence, or validation remain available for JSON diagnostics.
    if (candidate.discovery.effect_ns < kRobustMinimumEffectNs) {
      continue;
    }

    if (validation_matrix != nullptr && candidate.discovery.passed) {
      candidate.validation = evaluate_robust_evidence(
          *validation_matrix,
          locality_bytes.size(),
          segment_start_index,
          index,
          bootstrap_seed ^ 0xd1b54a32d192ed03ULL ^
              static_cast<uint64_t>(index));
    } else if (candidate.discovery.passed) {
      candidate.validation.rejection_reason = "validation-pass-not-provided";
    } else {
      candidate.validation.rejection_reason = "discovery-not-accepted";
    }
    candidate.accepted =
        candidate.discovery.passed && candidate.validation.passed;
    result.candidates.push_back(candidate);

    const bool select_candidate =
        (!result.detected && candidate.accepted) ||
        (!selected_discovery_candidate && validation_matrix == nullptr &&
         candidate.discovery.passed);
    if (!select_candidate) {
      continue;
    }

    selected_discovery_candidate = true;
    result.detected = candidate.accepted;
    result.boundary_index = index;
    result.boundary_locality_bytes = locality_bytes[index];
    result.bracket_lower_bytes = locality_bytes[index - 1];
    result.bracket_upper_bytes = locality_bytes[index];
    result.baseline_ns = matrix_column_median(discovery_matrix, index - 1);
    result.boundary_latency_ns = matrix_column_median(discovery_matrix, index);
    result.step_ns = candidate.discovery.effect_ns;
    result.step_percent = result.baseline_ns != 0.0
                              ? result.step_ns / std::abs(result.baseline_ns)
                              : 0.0;
    result.persistent_jump = candidate.discovery.persistence_points_passed ==
                             kRobustPersistencePoints;
    result.discovery = candidate.discovery;
    result.validation = candidate.validation;
    if (candidate.accepted) {
      const double strong_threshold = std::max(
          1.0, 2.0 * candidate.discovery.noise_floor_ns);
      const bool high_confidence =
          candidate.discovery.effect_ci.lower_ns >= strong_threshold &&
          candidate.validation.effect_ci.lower_ns >= strong_threshold;
      result.confidence = high_confidence ? "High" : "Medium";
    }
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

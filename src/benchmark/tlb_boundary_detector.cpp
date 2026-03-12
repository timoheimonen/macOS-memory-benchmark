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

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr double kRelativeThreshold = 0.10;
constexpr double kAbsoluteThresholdNs = 2.0;

double average_range(const std::vector<double>& values, size_t start, size_t end) {
  if (start >= end || end > values.size()) {
    return 0.0;
  }

  const double sum = std::accumulate(values.begin() + static_cast<std::ptrdiff_t>(start),
                                     values.begin() + static_cast<std::ptrdiff_t>(end), 0.0);
  const double count = static_cast<double>(end - start);
  return (count > 0.0) ? (sum / count) : 0.0;
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

TlbBoundaryDetection detect_tlb_boundary(const std::vector<size_t>& locality_bytes,
                                         const std::vector<double>& p50_latency_ns,
                                         size_t segment_start_index,
                                         size_t min_locality_bytes) {
  TlbBoundaryDetection result;
  result.segment_start_index = segment_start_index;

  if (locality_bytes.size() != p50_latency_ns.size() ||
      p50_latency_ns.size() < 2 ||
      segment_start_index >= p50_latency_ns.size() - 1) {
    return result;
  }

  for (size_t i = segment_start_index + 1; i < p50_latency_ns.size(); ++i) {
    const double baseline_ns = average_range(p50_latency_ns, segment_start_index, i);
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

    bool persistent_jump = false;
    if (i + 1 < p50_latency_ns.size()) {
      const double next_step_ns = p50_latency_ns[i + 1] - baseline_ns;
      persistent_jump = next_step_ns >= threshold_ns;
    }

    result.detected = true;
    result.boundary_index = i;
    result.boundary_locality_bytes = locality_bytes[i];
    result.baseline_ns = baseline_ns;
    result.boundary_latency_ns = boundary_ns;
    result.step_ns = step_ns;
    result.step_percent = step_percent;
    result.persistent_jump = persistent_jump;
    result.confidence = classify_tlb_confidence(step_ns, step_percent, persistent_jump);
    return result;
  }

  return result;
}

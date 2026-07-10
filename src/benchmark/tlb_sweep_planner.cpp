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
 * @file tlb_sweep_planner.cpp
 * @brief Pure sweep-point planning helpers for standalone TLB analysis
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 */

#include "benchmark/tlb_sweep_planner.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <map>

#include "core/config/constants.h"

namespace {

constexpr size_t kSweepMinLocalityBytes = 16 * Constants::BYTES_PER_KB;
constexpr size_t kSweepMaxLocalityBytes = 256 * Constants::BYTES_PER_MB;
constexpr size_t kRefinementSubdivisions = 8;

const std::array<size_t, 15> kLocalitySweepLowBytes = {
    16 * Constants::BYTES_PER_KB,        64 * Constants::BYTES_PER_KB,
    128 * Constants::BYTES_PER_KB,       256 * Constants::BYTES_PER_KB,
    512 * Constants::BYTES_PER_KB,       1 * Constants::BYTES_PER_MB,
    2 * Constants::BYTES_PER_MB,         4 * Constants::BYTES_PER_MB,
    8 * Constants::BYTES_PER_MB,         12 * Constants::BYTES_PER_MB,
    16 * Constants::BYTES_PER_MB,        32 * Constants::BYTES_PER_MB,
    64 * Constants::BYTES_PER_MB,        128 * Constants::BYTES_PER_MB,
    256 * Constants::BYTES_PER_MB,
};

const std::array<size_t, 29> kLocalitySweepHighBytes = {
    16 * Constants::BYTES_PER_KB,        32 * Constants::BYTES_PER_KB,
    64 * Constants::BYTES_PER_KB,        96 * Constants::BYTES_PER_KB,
    128 * Constants::BYTES_PER_KB,       192 * Constants::BYTES_PER_KB,
    256 * Constants::BYTES_PER_KB,       384 * Constants::BYTES_PER_KB,
    512 * Constants::BYTES_PER_KB,       768 * Constants::BYTES_PER_KB,
    1 * Constants::BYTES_PER_MB,         1536 * Constants::BYTES_PER_KB,
    2 * Constants::BYTES_PER_MB,         3 * Constants::BYTES_PER_MB,
    4 * Constants::BYTES_PER_MB,         6 * Constants::BYTES_PER_MB,
    8 * Constants::BYTES_PER_MB,         10 * Constants::BYTES_PER_MB,
    12 * Constants::BYTES_PER_MB,        14 * Constants::BYTES_PER_MB,
    16 * Constants::BYTES_PER_MB,        24 * Constants::BYTES_PER_MB,
    32 * Constants::BYTES_PER_MB,        48 * Constants::BYTES_PER_MB,
    64 * Constants::BYTES_PER_MB,        96 * Constants::BYTES_PER_MB,
    128 * Constants::BYTES_PER_MB,       192 * Constants::BYTES_PER_MB,
    256 * Constants::BYTES_PER_MB,
};

size_t calculate_min_sweep_locality(size_t stride_bytes) {
  if (stride_bytes > (std::numeric_limits<size_t>::max() / 2)) {
    return std::numeric_limits<size_t>::max();
  }
  return std::max(kSweepMinLocalityBytes, 2 * stride_bytes);
}

size_t align_up(size_t value, size_t alignment) {
  if (alignment == 0) {
    return value;
  }
  const size_t remainder = value % alignment;
  if (remainder == 0) {
    return value;
  }
  const size_t increment = alignment - remainder;
  return value > std::numeric_limits<size_t>::max() - increment
             ? std::numeric_limits<size_t>::max()
             : value + increment;
}

size_t bounded_double(size_t value, size_t maximum) {
  return value > std::numeric_limits<size_t>::max() / 2
             ? maximum
             : std::min(maximum, value * 2);
}

TlbSweepPoint make_point(size_t locality_bytes,
                         size_t stride_bytes,
                         size_t page_size_bytes,
                         const std::string& source,
                         size_t bracket_lower_bytes,
                         size_t bracket_upper_bytes) {
  TlbSweepPoint point;
  point.requested_pages = page_size_bytes == 0 ? 0 : locality_bytes / page_size_bytes;
  point.effective_pages = point.requested_pages;
  point.locality_bytes = locality_bytes;
  point.stride_bytes = stride_bytes;
  point.pointer_count = point.effective_pages;
  point.refinement_source = source;
  point.bracket_lower_bytes = bracket_lower_bytes;
  point.bracket_upper_bytes = bracket_upper_bytes;
  return point;
}

void append_source(std::string& existing, const std::string& source) {
  if (source.empty() || existing == source) {
    return;
  }
  if (!existing.empty()) {
    existing += "+";
  }
  existing += source;
}

}  // namespace

const char* tlb_sweep_density_to_string(TlbSweepDensity density) {
  switch (density) {
    case TlbSweepDensity::Low:
      return "low";
    case TlbSweepDensity::Medium:
      return "medium";
    case TlbSweepDensity::High:
      return "high";
  }
  return "high";
}

bool tlb_density_enables_refinement(TlbSweepDensity density) {
  return density != TlbSweepDensity::Low;
}

std::vector<TlbSweepPoint> build_tlb_base_sweep_plan(size_t stride_bytes,
                                                     size_t page_size_bytes,
                                                     TlbSweepDensity density) {
  std::vector<TlbSweepPoint> points;
  if (stride_bytes == 0 || page_size_bytes == 0 ||
      (stride_bytes % sizeof(uintptr_t)) != 0 || stride_bytes > page_size_bytes ||
      page_size_bytes < sizeof(uintptr_t)) {
    return points;
  }

  const size_t min_locality_bytes =
      align_up(calculate_min_sweep_locality(stride_bytes), page_size_bytes);
  if (min_locality_bytes > kSweepMaxLocalityBytes) {
    return points;
  }

  const size_t* canonical_localities = density == TlbSweepDensity::High
                                           ? kLocalitySweepHighBytes.data()
                                           : kLocalitySweepLowBytes.data();
  const size_t canonical_count = density == TlbSweepDensity::High
                                     ? kLocalitySweepHighBytes.size()
                                     : kLocalitySweepLowBytes.size();

  std::vector<size_t> localities;
  localities.reserve(canonical_count + 1);
  localities.push_back(min_locality_bytes);
  for (size_t i = 0; i < canonical_count; ++i) {
    if (canonical_localities[i] >= min_locality_bytes &&
        canonical_localities[i] <= kSweepMaxLocalityBytes) {
      const size_t aligned_locality = align_up(canonical_localities[i], page_size_bytes);
      if (aligned_locality <= kSweepMaxLocalityBytes) {
        localities.push_back(aligned_locality);
      }
    }
  }
  std::sort(localities.begin(), localities.end());
  localities.erase(std::unique(localities.begin(), localities.end()), localities.end());

  points.reserve(localities.size());
  for (size_t locality_bytes : localities) {
    points.push_back(make_point(locality_bytes,
                                stride_bytes,
                                page_size_bytes,
                                "base",
                                locality_bytes,
                                locality_bytes));
  }
  for (size_t i = 0; i < points.size(); ++i) {
    points[i].point_index = i;
  }
  return points;
}

std::vector<size_t> build_tlb_refinement_points(const std::vector<size_t>& localities_bytes,
                                                size_t boundary_index,
                                                size_t min_locality_bytes,
                                                size_t max_locality_bytes,
                                                size_t alignment_bytes) {
  std::vector<size_t> points;
  if (localities_bytes.empty() || boundary_index >= localities_bytes.size() ||
      alignment_bytes == 0) {
    return points;
  }

  const size_t boundary = localities_bytes[boundary_index];
  const size_t lower = boundary_index > 0 ? localities_bytes[boundary_index - 1]
                                         : min_locality_bytes;
  const size_t upper = boundary_index + 1 < localities_bytes.size()
                           ? localities_bytes[boundary_index + 1]
                           : bounded_double(boundary, max_locality_bytes);
  if (upper <= lower) {
    return points;
  }

  for (size_t step = 1; step < kRefinementSubdivisions; ++step) {
    size_t candidate = lower + ((upper - lower) * step) / kRefinementSubdivisions;
    candidate -= candidate % alignment_bytes;
    if (candidate > lower && candidate < upper && candidate >= min_locality_bytes &&
        candidate <= max_locality_bytes &&
        std::find(localities_bytes.begin(), localities_bytes.end(), candidate) ==
            localities_bytes.end()) {
      points.push_back(candidate);
    }
  }
  std::sort(points.begin(), points.end());
  points.erase(std::unique(points.begin(), points.end()), points.end());
  return points;
}

std::vector<TlbSweepPoint> build_tlb_refinement_plan(
    const std::vector<size_t>& localities_bytes,
    const std::vector<TlbRefinementTarget>& targets,
    size_t stride_bytes,
    size_t page_size_bytes,
    size_t min_locality_bytes,
    size_t max_locality_bytes) {
  std::map<size_t, TlbSweepPoint> by_locality;
  for (const TlbRefinementTarget& target : targets) {
    if (target.boundary_index >= localities_bytes.size()) {
      continue;
    }
    const size_t lower = target.boundary_index > 0
                             ? localities_bytes[target.boundary_index - 1]
                             : min_locality_bytes;
    const size_t upper = target.boundary_index + 1 < localities_bytes.size()
                             ? localities_bytes[target.boundary_index + 1]
                             : bounded_double(localities_bytes[target.boundary_index],
                                              max_locality_bytes);
    const std::vector<size_t> candidates = build_tlb_refinement_points(localities_bytes,
                                                                        target.boundary_index,
                                                                        min_locality_bytes,
                                                                        max_locality_bytes,
                                                                        page_size_bytes);
    for (size_t locality_bytes : candidates) {
      auto [it, inserted] = by_locality.emplace(
          locality_bytes,
          make_point(locality_bytes,
                     stride_bytes,
                     page_size_bytes,
                     target.source,
                     lower,
                     upper));
      if (!inserted) {
        append_source(it->second.refinement_source, target.source);
        it->second.bracket_lower_bytes = std::min(it->second.bracket_lower_bytes, lower);
        it->second.bracket_upper_bytes = std::max(it->second.bracket_upper_bytes, upper);
      }
    }
  }

  std::vector<TlbSweepPoint> points;
  points.reserve(by_locality.size());
  for (auto& entry : by_locality) {
    entry.second.point_index = points.size();
    points.push_back(std::move(entry.second));
  }
  return points;
}

std::vector<size_t> tlb_point_localities(const std::vector<TlbSweepPoint>& points) {
  std::vector<size_t> localities;
  localities.reserve(points.size());
  for (const TlbSweepPoint& point : points) {
    localities.push_back(point.locality_bytes);
  }
  return localities;
}

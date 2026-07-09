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
 * @file tlb_sweep_planner.h
 * @brief Pure sweep-point planning helpers for standalone TLB analysis
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 */

#ifndef TLB_SWEEP_PLANNER_H
#define TLB_SWEEP_PLANNER_H

#include <cstddef>
#include <string>
#include <vector>

#include "core/config/config.h"

/** Planned locality point and the page-accounting metadata used by analysis output. */
struct TlbSweepPoint {
  size_t point_index = 0;
  size_t requested_pages = 0;
  size_t effective_pages = 0;
  size_t locality_bytes = 0;
  size_t stride_bytes = 0;
  size_t pointer_count = 0;
  std::string refinement_source = "base";
  size_t bracket_lower_bytes = 0;
  size_t bracket_upper_bytes = 0;
};

/** Boundary candidate that requests refinement and labels its source. */
struct TlbRefinementTarget {
  size_t boundary_index = 0;
  std::string source;
};

const char* tlb_sweep_density_to_string(TlbSweepDensity density);

bool tlb_density_enables_refinement(TlbSweepDensity density);

/** Build the canonical page-aligned base grid without executing benchmark code. */
std::vector<TlbSweepPoint> build_tlb_base_sweep_plan(size_t stride_bytes,
                                                     size_t page_size_bytes,
                                                     TlbSweepDensity density);

/** Build deduplicated page-aligned refinement points for boundary targets. */
std::vector<TlbSweepPoint> build_tlb_refinement_plan(
    const std::vector<size_t>& localities_bytes,
    const std::vector<TlbRefinementTarget>& targets,
    size_t stride_bytes,
    size_t page_size_bytes,
    size_t min_locality_bytes,
    size_t max_locality_bytes);

/** Build raw refinement localities strictly inside one measured bracket. */
std::vector<size_t> build_tlb_refinement_points(const std::vector<size_t>& localities_bytes,
                                                size_t boundary_index,
                                                size_t min_locality_bytes,
                                                size_t max_locality_bytes,
                                                size_t alignment_bytes);

/** Extract locality byte values from a planned point vector. */
std::vector<size_t> tlb_point_localities(const std::vector<TlbSweepPoint>& points);

#endif  // TLB_SWEEP_PLANNER_H

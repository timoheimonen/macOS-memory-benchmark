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
 * @file analysis.h
 * @brief Standalone TLB analysis mode interfaces
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 */

#ifndef ANALYSIS_H
#define ANALYSIS_H

#include <cstddef>  // size_t
#include <string>
#include <vector>

/**
 * @struct TlbBoundaryDetection
 * @brief Boundary-detection result for TLB working-set transition analysis
 */
struct TlbBoundaryDetection {
  bool detected = false;
  size_t segment_start_index = 0;
  size_t boundary_index = 0;
  size_t boundary_locality_bytes = 0;
  double baseline_ns = 0.0;
  double boundary_latency_ns = 0.0;
  double step_ns = 0.0;
  double step_percent = 0.0;
  bool persistent_jump = false;
  std::string confidence;
};

/**
 * @brief Detect the first boundary where latency rises by >=10% or >=2ns.
 * @param locality_bytes Locality windows used in measurement order
 * @param p50_latency_ns P50 latency values corresponding to locality windows
 * @param segment_start_index Index where running-baseline segment starts
 * @param min_locality_bytes Minimum locality window to accept as a TLB boundary
 * @return Boundary-detection result
 */
TlbBoundaryDetection detect_tlb_boundary(const std::vector<size_t>& locality_bytes,
                                         const std::vector<double>& p50_latency_ns,
                                         size_t segment_start_index,
                                         size_t min_locality_bytes = 0);

/**
 * @brief Infer TLB entries from locality boundary and page size.
 * @param locality_bytes Boundary locality window in bytes
 * @param page_size_bytes System page size in bytes
 * @return Inferred TLB entries (0 if page size is invalid)
 */
size_t infer_tlb_entries(size_t locality_bytes, size_t page_size_bytes);

/**
 * @brief Classify confidence for a detected boundary.
 * @param step_ns Absolute latency step in nanoseconds
 * @param step_percent Relative step ratio (e.g. 0.12 = 12%)
 * @param persistent_jump Whether the jump persists at next locality point
 * @return Confidence level string (High/Medium/Low)
 */
std::string classify_tlb_confidence(double step_ns, double step_percent, bool persistent_jump);

/**
 * @brief Run standalone TLB analysis benchmark mode.
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on error
 */
int run_tlb_analysis();

#endif  // ANALYSIS_H

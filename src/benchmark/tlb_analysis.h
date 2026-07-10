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
 * @file tlb_analysis.h
 * @brief Standalone TLB analysis mode interfaces
 * @author Timo Heimonen <timo.heimonen@proton.me>
 * @date 2026
 */

#ifndef TLB_ANALYSIS_H
#define TLB_ANALYSIS_H

#include <cstddef>  // size_t
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "benchmark/tlb_measurement_scheduler.h"

// Forward declaration
struct BenchmarkConfig;

/** Deterministic percentile-bootstrap interval for a paired median effect. */
struct TlbBootstrapInterval {
  double lower_ns = 0.0;
  double upper_ns = 0.0;
  double confidence_level = 0.95;
  size_t paired_sample_count = 0;
  size_t bootstrap_resamples = 0;
};

/** Discovery or validation evidence for one boundary candidate. */
struct TlbBoundaryEvidence {
  bool available = false;
  bool passed = false;
  double effect_ns = 0.0;
  double minimum_effect_ns = 0.0;
  double noise_floor_ns = 0.0;
  TlbBootstrapInterval effect_ci;
  size_t persistence_points_passed = 0;
  size_t persistence_points_required = 2;
  std::string rejection_reason;
};

/** Accepted or rejected changepoint candidate retained for diagnostics. */
struct TlbBoundaryCandidate {
  size_t boundary_index = 0;
  size_t boundary_locality_bytes = 0;
  size_t bracket_lower_bytes = 0;
  size_t bracket_upper_bytes = 0;
  TlbBoundaryEvidence discovery;
  TlbBoundaryEvidence validation;
  bool accepted = false;
};

/** Round-major matrix. Rows are rounds and columns are locality points. */
using TlbRoundPointMatrix = std::vector<std::vector<double>>;

/** Deterministic aggregate of same-round spread/packed measurements for one locality. */
struct TlbPairedPointSummary {
  bool available = false;
  double spread_p50_ns = 0.0;
  double packed_p50_ns = 0.0;
  double translation_delta_p50_ns = 0.0;
  size_t spread_actual_pages = 0;
  size_t packed_actual_pages = 0;
  size_t unique_cache_lines = 0;
  size_t active_cache_line_footprint_bytes = 0;
  size_t node_count = 0;
  bool short_cycle_diagnostic = false;
};

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
  bool overlaps_private_cache_knee = false;
  size_t bracket_lower_bytes = 0;
  size_t bracket_upper_bytes = 0;
  TlbBoundaryEvidence discovery;
  TlbBoundaryEvidence validation;
  std::vector<TlbBoundaryCandidate> candidates;
  std::string confidence;
};

/**
 * @struct PrivateCacheKneeDetection
 * @brief Detected private-cache knee candidate in locality sweep.
 */
struct PrivateCacheKneeDetection {
  bool detected = false;
  size_t boundary_index = 0;
  size_t boundary_locality_bytes = 0;
  double step_ns = 0.0;
  double step_percent = 0.0;
  std::string confidence;
  bool strong_private_cache_candidate = false;
  bool early_cache_candidate = false;
  bool may_interfere_with_tlb = false;
};

/**
 * @brief Detect the first boundary where latency rises by >=10% or >=2ns.
 *
 * Uses multi-point persistence (3-point window, majority rule), recency-weighted
 * baseline, and optional IQR-overlap rejection when loop-level data is provided.
 *
 * @param locality_bytes Locality windows used in measurement order
 * @param p50_latency_ns P50 latency values corresponding to locality windows
 * @param segment_start_index Index where running-baseline segment starts
 * @param min_locality_bytes Minimum locality window to accept as a TLB boundary
 * @param loop_latencies Optional per-point raw loop latencies for IQR gating (nullptr to skip)
 * @return Boundary-detection result
 */
TlbBoundaryDetection detect_tlb_boundary(const std::vector<size_t>& locality_bytes,
                                         const std::vector<double>& p50_latency_ns,
                                         size_t segment_start_index,
                                         size_t min_locality_bytes = 0,
                                         const std::vector<std::vector<double>>* loop_latencies = nullptr);

/**
 * @brief Build a round x point translation-delta matrix from measurement records.
 *
 * Only records from the requested pass set are included. Missing cells are NaN.
 */
TlbRoundPointMatrix build_tlb_translation_delta_matrix(
    const std::vector<size_t>& locality_bytes,
    const std::vector<TlbMeasurementRecord>& records,
    const std::vector<TlbMeasurementPass>& included_passes);

/**
 * @brief Aggregate one locality without mixing measurement passes.
 *
 * The translation-delta P50 is the median of stored same-round deltas, not the
 * difference between independently aggregated spread and packed medians. Chain
 * diagnostics must agree across every included record or the summary is unavailable.
 */
TlbPairedPointSummary summarize_tlb_paired_point(
    const std::vector<TlbMeasurementRecord>& records,
    size_t locality_bytes,
    const std::vector<TlbMeasurementPass>& included_passes);

/**
 * @brief Detect a persistent translation-delta changepoint with independent validation.
 *
 * Discovery and validation use paired point-to-point effects, a deterministic
 * percentile-bootstrap 95% interval, a predefined minimum effect, and two
 * following persistence points. A boundary is accepted only when both passes
 * satisfy the same criteria. When validation_matrix is null, discovery evidence
 * and candidates are returned but `detected` remains false.
 */
TlbBoundaryDetection detect_tlb_boundary_robust(
    const std::vector<size_t>& locality_bytes,
    const TlbRoundPointMatrix& discovery_matrix,
    const TlbRoundPointMatrix* validation_matrix,
    size_t segment_start_index,
    size_t min_locality_bytes,
    uint64_t bootstrap_seed);

/**
 * @brief Infer TLB entries from locality boundary and page size.
 * @param locality_bytes Boundary locality window in bytes
 * @param page_size_bytes System page size in bytes
 * @return Inferred TLB entries (0 if page size is invalid)
 */
size_t infer_tlb_entries(size_t locality_bytes, size_t page_size_bytes);

/**
 * @brief Infer entry range from boundary and previous locality point.
 * @param locality_bytes Sweep localities corresponding to detector input
 * @param boundary_index Index of detected boundary
 * @param page_size_bytes System page size in bytes
 * @return Pair(min_entries, max_entries)
 */
std::pair<size_t, size_t> infer_tlb_entries_range(const std::vector<size_t>& locality_bytes,
                                                  size_t boundary_index,
                                                  size_t page_size_bytes);

/**
 * @brief Infer a midpoint entry estimate from the detected boundary window.
 * @param locality_bytes Sweep localities corresponding to detector input
 * @param boundary_index Index of detected boundary
 * @param page_size_bytes System page size in bytes
 * @return Midpoint estimate between previous locality and detected boundary
 */
size_t infer_tlb_entries_estimate(const std::vector<size_t>& locality_bytes,
                                  size_t boundary_index,
                                  size_t page_size_bytes);

/**
 * @brief Classify confidence for a detected boundary.
 * @param step_ns Absolute latency step in nanoseconds
 * @param step_percent Relative step ratio (e.g. 0.12 = 12%)
 * @param persistent_jump Whether the jump persists at next locality point
 * @return Confidence level string (High/Medium/Low)
 */
std::string classify_tlb_confidence(double step_ns, double step_percent, bool persistent_jump);

/**
 * @brief Detect likely private-cache knee near 1 MB region.
 * @param locality_bytes Locality windows used in measurement order
 * @param p50_latency_ns P50 latency values corresponding to locality windows
 * @param loop_latencies Optional per-point raw loop latencies for IQR gating (nullptr to skip)
 * @return Cache-knee detection metadata
 */
PrivateCacheKneeDetection detect_private_cache_knee(
    const std::vector<size_t>& locality_bytes,
    const std::vector<double>& p50_latency_ns,
    const std::vector<std::vector<double>>* loop_latencies = nullptr);

/**
 * @brief Run standalone TLB analysis benchmark mode.
 * @param config Benchmark configuration (supports optional output_file)
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on error
 */
int run_tlb_analysis(const BenchmarkConfig& config);

/**
 * @brief Run standalone TLB analysis with an injectable stop predicate.
 * @param config Benchmark configuration.
 * @param stop_requested Non-blocking stop predicate checked between scheduled measurements.
 */
int run_tlb_analysis(const BenchmarkConfig& config,
                     const TlbStopRequested& stop_requested);

#endif  // TLB_ANALYSIS_H

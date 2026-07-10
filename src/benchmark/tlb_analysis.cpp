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
 * @file tlb_analysis.cpp
 * @brief Standalone TLB analysis mode implementation
 */

#include "benchmark/tlb_analysis.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

#include "benchmark/benchmark_tests.h"
#include "benchmark/tlb_analysis_json.h"
#include "benchmark/tlb_chain.h"
#include "benchmark/tlb_measurement_scheduler.h"
#include "benchmark/tlb_runtime_policy.h"
#include "benchmark/tlb_sweep_planner.h"
#include "core/config/config.h"
#include "core/config/constants.h"
#include "core/config/version.h"
#include "core/memory/memory_manager.h"
#include "core/memory/memory_utils.h"
#include "core/signal/signal_handler.h"
#include "core/system/system_info.h"
#include "core/timing/timer.h"
#include "output/console/messages/messages_api.h"
#include "warmup/warmup.h"

namespace {

constexpr size_t kPageWalkComparisonLocalityBytes = 512 * Constants::BYTES_PER_MB;
constexpr size_t kPageWalkMinimumBufferMb = 512;

const std::array<size_t, 3> kBufferCandidateMb = {1024, 512, 256};

struct LocalityMeasurement {
  size_t locality_bytes = 0;
  double p50_latency_ns = 0.0;
  std::vector<double> loop_latencies_ns;
};

size_t maximum_tlb_node_count(const std::vector<TlbSweepPoint>& points) {
  size_t maximum_node_count = 0;
  for (const TlbSweepPoint& point : points) {
    maximum_node_count = std::max(maximum_node_count, point.pointer_count);
  }
  return maximum_node_count;
}

std::string tlb_pass_completion_reason(
    const TlbScheduleExecutionResult& result) {
  if (result.status == TlbScheduleExecutionStatus::Interrupted) {
    return "interrupted";
  }
  if (result.status == TlbScheduleExecutionStatus::Error) {
    return "measurement error";
  }
  return result.converged ? "CI target reached" : "maximum rounds reached";
}

void print_tlb_work_estimate(const std::string& pass_name,
                             const TlbWorkEstimate& estimate) {
  std::cout << Messages::report_tlb_work_estimate(
                   pass_name,
                   estimate.point_count,
                   estimate.min_rounds,
                   estimate.max_rounds,
                   estimate.estimated_min_duration_sec,
                   estimate.estimated_max_duration_sec)
            << std::endl;
}

TlbPassExecutionSummary summarize_tlb_pass(
    TlbMeasurementPass pass,
    size_t point_count,
    const TlbScheduleExecutionResult& result) {
  return {pass,
          point_count,
          result.rounds_completed,
          result.converged,
          result.status == TlbScheduleExecutionStatus::Complete,
          result.status};
}

void print_tlb_pass_completion(
    TlbMeasurementPass pass,
    const TlbScheduleExecutionResult& result) {
  std::cout << Messages::report_tlb_pass_completion(
                   tlb_measurement_pass_to_string(pass),
                   result.rounds_completed,
                   tlb_pass_completion_reason(result))
            << std::endl;
}

class TlbMeasureSpinner {
 public:
  TlbMeasureSpinner() : enabled_(isatty(fileno(stderr)) == 1) {}

  ~TlbMeasureSpinner() {
    clear();
  }

  void tick(size_t locality_kb, size_t loop_index, size_t loop_total) {
    if (!enabled_) {
      return;
    }

    static constexpr char kFrames[] = {'|', '/', '-', '\\'};
    const char frame = kFrames[frame_index_ % 4];
    ++frame_index_;

    std::ostringstream oss;
    oss << frame << " Measuring locality " << locality_kb << " KB"
        << " (loop " << loop_index << "/" << loop_total << ")";
    render(oss.str());
  }

 private:
  void render(const std::string& text) {
    std::string padded = text;
    if (last_text_len_ > text.size()) {
      padded.append(last_text_len_ - text.size(), ' ');
    }
    std::fputs(("\r" + padded).c_str(), stderr);
    std::fflush(stderr);
    last_text_len_ = text.size();
  }

  void clear() {
    if (!enabled_ || last_text_len_ == 0) {
      return;
    }
    std::fputs(("\r" + std::string(last_text_len_, ' ') + "\r").c_str(), stderr);
    std::fflush(stderr);
    last_text_len_ = 0;
  }

  bool enabled_ = false;
  size_t frame_index_ = 0;
  size_t last_text_len_ = 0;
};

void flatten_measurements(const std::vector<LocalityMeasurement>& measurements,
                          std::vector<size_t>& out_localities,
                          std::vector<double>& out_p50,
                          std::vector<std::vector<double>>& out_loops) {
  out_localities.clear();
  out_p50.clear();
  out_loops.clear();
  out_localities.reserve(measurements.size());
  out_p50.reserve(measurements.size());
  out_loops.reserve(measurements.size());

  for (const LocalityMeasurement& measurement : measurements) {
    out_localities.push_back(measurement.locality_bytes);
    out_p50.push_back(measurement.p50_latency_ns);
    out_loops.push_back(measurement.loop_latencies_ns);
  }
}

double median(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }

  std::sort(values.begin(), values.end());
  const size_t mid = values.size() / 2;
  if ((values.size() % 2) == 0) {
    return 0.5 * (values[mid - 1] + values[mid]);
  }
  return values[mid];
}

void append_validation_indices(const TlbBoundaryDetection& detection,
                               size_t point_count,
                               std::vector<size_t>& indices) {
  for (const TlbBoundaryCandidate& candidate : detection.candidates) {
    if (!candidate.discovery.passed || candidate.boundary_index == 0) {
      continue;
    }
    const size_t first = candidate.boundary_index - 1;
    const size_t last = std::min(candidate.boundary_index + 2,
                                 point_count - 1);
    for (size_t index = first; index <= last; ++index) {
      indices.push_back(index);
    }
  }
}

std::vector<TlbSweepPoint> build_validation_plan(
    const std::vector<size_t>& locality_bytes,
    const TlbBoundaryDetection& l1_discovery,
    const TlbBoundaryDetection& l2_discovery,
    size_t stride_bytes,
    size_t page_size_bytes,
    size_t first_point_index) {
  std::vector<size_t> indices;
  if (!locality_bytes.empty()) {
    append_validation_indices(l1_discovery, locality_bytes.size(), indices);
    append_validation_indices(l2_discovery, locality_bytes.size(), indices);
  }
  std::sort(indices.begin(), indices.end());
  indices.erase(std::unique(indices.begin(), indices.end()), indices.end());

  std::vector<TlbSweepPoint> points;
  points.reserve(indices.size());
  for (size_t index : indices) {
    TlbSweepPoint point;
    point.point_index = first_point_index + points.size();
    point.locality_bytes = locality_bytes[index];
    point.requested_pages = point.locality_bytes / page_size_bytes;
    point.effective_pages = point.requested_pages;
    point.stride_bytes = stride_bytes;
    point.pointer_count = point.effective_pages;
    point.refinement_source = "validation";
    point.bracket_lower_bytes = index > 0 ? locality_bytes[index - 1]
                                          : locality_bytes[index];
    point.bracket_upper_bytes = locality_bytes[index];
    points.push_back(point);
  }
  return points;
}

std::pair<size_t, size_t> validated_entry_range(
    const TlbBoundaryDetection& boundary,
    size_t page_size_bytes) {
  if (!boundary.detected || page_size_bytes == 0) {
    return {0, 0};
  }
  return {boundary.bracket_lower_bytes / page_size_bytes,
          boundary.bracket_upper_bytes / page_size_bytes};
}

bool boundary_matches_private_cache_knee(const TlbBoundaryDetection& boundary,
                                         const PrivateCacheKneeDetection& private_cache_knee) {
  return boundary.detected &&
         private_cache_knee.detected &&
         boundary.boundary_index == private_cache_knee.boundary_index &&
         boundary.boundary_locality_bytes == private_cache_knee.boundary_locality_bytes;
}

TlbBoundaryDetection detect_l1_tlb_boundary(
    const std::vector<size_t>& localities_bytes,
    const std::vector<double>& p50_latency_ns,
    size_t tlb_guard_bytes,
    const PrivateCacheKneeDetection& private_cache_knee,
    const std::vector<std::vector<double>>& sweep_loop_latencies_ns) {
  TlbBoundaryDetection direct_boundary =
      detect_tlb_boundary(localities_bytes, p50_latency_ns, 0, tlb_guard_bytes, &sweep_loop_latencies_ns);
  if (!private_cache_knee.detected) {
    return direct_boundary;
  }

  if (boundary_matches_private_cache_knee(direct_boundary, private_cache_knee)) {
    direct_boundary.overlaps_private_cache_knee = true;
    return direct_boundary;
  }

  if (!localities_bytes.empty() && private_cache_knee.boundary_index < localities_bytes.size() - 1) {
    const size_t offset_segment_start = private_cache_knee.boundary_index + 1;
    TlbBoundaryDetection offset_boundary =
        detect_tlb_boundary(localities_bytes,
                            p50_latency_ns,
                            offset_segment_start,
                            tlb_guard_bytes,
                            &sweep_loop_latencies_ns);
    if (offset_boundary.detected) {
      return offset_boundary;
    }
  }

  return direct_boundary;
}

MmapPtr try_allocate_analysis_buffer(size_t size_bytes) {
  if (size_bytes == 0) {
    return MmapPtr(nullptr, MmapDeleter{0});
  }

  void* ptr = mmap(nullptr, size_bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED) {
    return MmapPtr(nullptr, MmapDeleter{0});
  }

  (void)madvise(ptr, size_bytes, MADV_WILLNEED);
  return MmapPtr(ptr, MmapDeleter{size_bytes});
}

TlbChainTraversalPolicy tlb_chain_policy_for_mode(LatencyChainMode mode) {
  switch (mode) {
    case LatencyChainMode::SameRandomInBoxIncreasingBox:
      return TlbChainTraversalPolicy::IncreasingPagesSharedOffset;
    case LatencyChainMode::DiffRandomInBoxIncreasingBox:
      return TlbChainTraversalPolicy::IncreasingPagesRandomOffsets;
    case LatencyChainMode::Auto:
    case LatencyChainMode::GlobalRandom:
    case LatencyChainMode::RandomInBoxRandomBox:
      return TlbChainTraversalPolicy::RandomPagesRandomOffsets;
  }
  return TlbChainTraversalPolicy::RandomPagesRandomOffsets;
}

TlbTaskMeasureStatus measure_tlb_chain(
    void* latency_buffer,
    size_t buffer_size_bytes,
    size_t stride_bytes,
    size_t page_size_bytes,
    const TlbMeasurementTask& task,
    TlbChainLayout layout,
    TlbChainTraversalPolicy traversal_policy,
    HighResTimer& timer,
    const TlbRuntimeProfile& runtime_profile,
    TlbChainScratch& scratch,
    TlbChainMeasurement& measurement) {
  if (task.locality_bytes == 0 ||
      (task.locality_bytes % page_size_bytes) != 0) {
    return TlbTaskMeasureStatus::Error;
  }

  measurement.seed = derive_tlb_chain_layout_seed(task.seed, layout);
  const size_t requested_pages = task.locality_bytes / page_size_bytes;
  const TlbChainBuildResult chain = build_tlb_chain(latency_buffer,
                                                    buffer_size_bytes,
                                                    requested_pages,
                                                    page_size_bytes,
                                                    stride_bytes,
                                                    layout,
                                                    traversal_policy,
                                                    measurement.seed,
                                                    scratch);
  if (chain.status != TlbChainBuildStatus::Success) {
    std::cerr << Messages::error_prefix()
              << Messages::error_tlb_chain_setup_failed(
                     task.locality_bytes / Constants::BYTES_PER_KB,
                     tlb_chain_layout_to_string(layout),
                     tlb_chain_build_status_to_string(chain.status),
                     tlb_chain_validation_status_to_string(
                         chain.validation_status))
              << std::endl;
    return TlbTaskMeasureStatus::Error;
  }

  measurement.diagnostics = chain.diagnostics;
  const size_t warmup_bytes =
      layout == TlbChainLayout::Spread
          ? task.locality_bytes
          : chain.diagnostics.actual_pages * page_size_bytes;
  warmup_latency(latency_buffer, std::min(warmup_bytes, buffer_size_bytes));
  measurement.pilot_access_count =
      calculate_tlb_pilot_accesses(chain.diagnostics.node_count);
  measurement.pilot_duration_ns = run_latency_test(
      chain.chain_head, measurement.pilot_access_count, timer, nullptr, 0);
  if (measurement.pilot_access_count == 0 ||
      measurement.pilot_duration_ns <= 0.0 ||
      !std::isfinite(measurement.pilot_duration_ns)) {
    std::cerr << Messages::error_prefix()
              << Messages::error_tlb_analysis_invalid_measurement(
                     task.locality_bytes / Constants::BYTES_PER_KB,
                     static_cast<int>(task.round_index + 1))
              << std::endl;
    return TlbTaskMeasureStatus::Error;
  }
  measurement.access_count = calculate_tlb_calibrated_accesses(
      chain.diagnostics.node_count,
      measurement.pilot_access_count,
      measurement.pilot_duration_ns,
      runtime_profile);
  if (measurement.access_count == 0) {
    std::cerr << Messages::error_prefix()
              << Messages::error_tlb_analysis_invalid_measurement(
                     task.locality_bytes / Constants::BYTES_PER_KB,
                     static_cast<int>(task.round_index + 1))
              << std::endl;
    return TlbTaskMeasureStatus::Error;
  }
  const double total_latency_ns = run_latency_test(
      chain.chain_head, measurement.access_count, timer, nullptr, 0);
  if (total_latency_ns <= 0.0 || std::isnan(total_latency_ns) ||
      std::isinf(total_latency_ns)) {
    std::cerr << Messages::error_prefix()
              << Messages::error_tlb_analysis_invalid_measurement(
                     task.locality_bytes / Constants::BYTES_PER_KB,
                     static_cast<int>(task.round_index + 1))
              << std::endl;
    return TlbTaskMeasureStatus::Error;
  }

  measurement.latency_ns =
      total_latency_ns / static_cast<double>(measurement.access_count);
  return TlbTaskMeasureStatus::Success;
}

void append_measurement_records(
    const std::vector<TlbMeasurementRecord>& records,
    std::vector<LocalityMeasurement>& measurements) {
  for (const TlbMeasurementRecord& record : records) {
    auto existing = std::find_if(
        measurements.begin(),
        measurements.end(),
        [&record](const LocalityMeasurement& measurement) {
          return measurement.locality_bytes == record.locality_bytes;
        });
    if (existing == measurements.end()) {
      measurements.push_back(LocalityMeasurement{
          record.locality_bytes, record.latency_ns, {record.latency_ns}});
    } else {
      existing->loop_latencies_ns.push_back(record.latency_ns);
    }
  }
  for (LocalityMeasurement& measurement : measurements) {
    if (!measurement.loop_latencies_ns.empty()) {
      measurement.p50_latency_ns = median(measurement.loop_latencies_ns);
    }
  }
}

/**
 * @brief Execute a balanced seeded round schedule and aggregate locality medians.
 *
 * Each round measures every supplied point once. The scheduler rotates the seeded
 * point order between rounds so locality and elapsed run time are not correlated.
 */
TlbScheduleExecutionResult measure_scheduled_points(
    void* latency_buffer,
    size_t buffer_size_bytes,
    size_t stride_bytes,
    size_t page_size_bytes,
    const std::vector<TlbSweepPoint>& points,
    LatencyChainMode chain_mode,
    HighResTimer& timer,
    uint64_t base_seed,
    TlbMeasurementPass pass,
    const TlbRuntimeProfile& runtime_profile,
    const TlbStopRequested& stop_requested,
    std::vector<LocalityMeasurement>& measurements) {
  TlbMeasureSpinner spinner;
  TlbChainScratch chain_scratch;
  TlbConvergenceScratch convergence_scratch;
  std::vector<std::vector<double>> convergence_samples(points.size());
  const std::vector<TlbMeasurementTask> schedule =
      build_tlb_measurement_schedule(
          points, runtime_profile.max_rounds, base_seed, pass);
  TlbScheduleExecutionResult result = execute_tlb_measurement_schedule(
      schedule,
      stop_requested,
      [&](const TlbMeasurementTask& task, TlbMeasurementSample& sample) {
        const size_t locality_kb = task.locality_bytes / Constants::BYTES_PER_KB;
        spinner.tick(locality_kb,
                     task.round_index + 1,
                     runtime_profile.max_rounds);

        sample.paired.available = true;
        sample.paired.spread_measured_first =
            tlb_measure_spread_first(task);
        const TlbChainTraversalPolicy traversal_policy =
            tlb_chain_policy_for_mode(chain_mode);
        auto measure_layout = [&](TlbChainLayout layout,
                                  TlbChainMeasurement& measurement) {
          return measure_tlb_chain(latency_buffer,
                                   buffer_size_bytes,
                                   stride_bytes,
                                   page_size_bytes,
                                   task,
                                   layout,
                                   traversal_policy,
                                   timer,
                                   runtime_profile,
                                   chain_scratch,
                                   measurement);
        };

        if (sample.paired.spread_measured_first) {
          if (measure_layout(TlbChainLayout::Spread,
                             sample.paired.spread) !=
                  TlbTaskMeasureStatus::Success ||
              measure_layout(TlbChainLayout::Packed,
                             sample.paired.packed) !=
                  TlbTaskMeasureStatus::Success) {
            return TlbTaskMeasureStatus::Error;
          }
        } else if (measure_layout(TlbChainLayout::Packed,
                                  sample.paired.packed) !=
                       TlbTaskMeasureStatus::Success ||
                   measure_layout(TlbChainLayout::Spread,
                                  sample.paired.spread) !=
                       TlbTaskMeasureStatus::Success) {
          return TlbTaskMeasureStatus::Error;
        }

        sample.latency_ns = sample.paired.spread.latency_ns;
        sample.paired.translation_delta_ns =
            sample.paired.spread.latency_ns -
            sample.paired.packed.latency_ns;
        const auto point = std::find_if(
            points.begin(),
            points.end(),
            [&task](const TlbSweepPoint& candidate) {
              return candidate.locality_bytes == task.locality_bytes;
            });
        if (point != points.end()) {
          convergence_samples[static_cast<size_t>(
              std::distance(points.begin(), point))]
              .push_back(sample.paired.translation_delta_ns);
        }
        return TlbTaskMeasureStatus::Success;
      },
      [&](size_t completed_rounds,
          const std::vector<TlbMeasurementRecord>&) {
        if (completed_rounds < runtime_profile.min_rounds) {
          return false;
        }
        const TlbConvergenceSummary convergence = evaluate_tlb_convergence(
            convergence_samples,
            runtime_profile,
            base_seed ^ static_cast<uint64_t>(pass),
            &convergence_scratch);
        return convergence.converged;
      });

  append_measurement_records(result.records, measurements);
  return result;
}

}  // namespace

TlbPairedPointSummary summarize_tlb_paired_point(
    const std::vector<TlbMeasurementRecord>& records,
    size_t locality_bytes,
    const std::vector<TlbMeasurementPass>& included_passes) {
  TlbPairedPointSummary summary;
  std::vector<double> spread_latencies_ns;
  std::vector<double> packed_latencies_ns;
  std::vector<double> translation_deltas_ns;
  bool diagnostics_initialized = false;

  for (const TlbMeasurementRecord& record : records) {
    if (record.locality_bytes != locality_bytes ||
        std::find(included_passes.begin(), included_passes.end(), record.pass) ==
            included_passes.end()) {
      continue;
    }
    if (!record.paired.available ||
        !std::isfinite(record.paired.spread.latency_ns) ||
        !std::isfinite(record.paired.packed.latency_ns) ||
        !std::isfinite(record.paired.translation_delta_ns)) {
      return TlbPairedPointSummary{};
    }

    const TlbChainDiagnostics& spread = record.paired.spread.diagnostics;
    const TlbChainDiagnostics& packed = record.paired.packed.diagnostics;
    if (spread.node_count == 0 || spread.actual_pages == 0 ||
        packed.actual_pages == 0 || spread.unique_cache_lines == 0 ||
        spread.node_count != packed.node_count ||
        spread.unique_cache_lines != packed.unique_cache_lines ||
        spread.unique_cache_lines >
            std::numeric_limits<size_t>::max() /
                Constants::CACHE_LINE_SIZE_BYTES) {
      return TlbPairedPointSummary{};
    }

    if (!diagnostics_initialized) {
      summary.spread_actual_pages = spread.actual_pages;
      summary.packed_actual_pages = packed.actual_pages;
      summary.unique_cache_lines = spread.unique_cache_lines;
      summary.node_count = spread.node_count;
      diagnostics_initialized = true;
    } else if (summary.spread_actual_pages != spread.actual_pages ||
               summary.packed_actual_pages != packed.actual_pages ||
               summary.unique_cache_lines != spread.unique_cache_lines ||
               summary.node_count != spread.node_count) {
      return TlbPairedPointSummary{};
    }

    spread_latencies_ns.push_back(record.paired.spread.latency_ns);
    packed_latencies_ns.push_back(record.paired.packed.latency_ns);
    translation_deltas_ns.push_back(record.paired.translation_delta_ns);
  }

  if (!diagnostics_initialized || spread_latencies_ns.empty()) {
    return TlbPairedPointSummary{};
  }

  summary.spread_p50_ns = median(std::move(spread_latencies_ns));
  summary.packed_p50_ns = median(std::move(packed_latencies_ns));
  summary.translation_delta_p50_ns = median(std::move(translation_deltas_ns));
  summary.active_cache_line_footprint_bytes =
      summary.unique_cache_lines * Constants::CACHE_LINE_SIZE_BYTES;
  summary.short_cycle_diagnostic = summary.node_count < 64;
  summary.available = true;
  return summary;
}

/**
 * @brief Run standalone TLB analysis benchmark mode.
 *
 * Workflow:
 * - Print program header and run banner.
 * - Try 1024/512/256MB buffers in descending order and select the largest candidate whose
 *   predicted peak fits the memory budget and whose allocation succeeds.
 * - Use configured latency stride (default from `--latency-stride-bytes`).
 * - Calibrate whole-chain access counts and measure adaptive balanced seeded rounds.
 * - Optionally run a separate 512MB large-locality comparison pass.
 * - Infer L1/L2 boundaries and emit the TLB analysis report.
 *
 * Large-locality paired comparison:
 * - Reports spread, packed, and same-round translation-delta P50 values plus the
 *   active cache-line footprint.
 * - Only available when the selected buffer is at least 512MB and the paired
 *   comparison pass completes; the raw spread baseline delta is JSON compatibility data.
 *
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on allocation or measurement error.
 */
namespace {

int run_tlb_analysis_impl(
    const BenchmarkConfig& config,
    const TlbStopRequested& stop_requested,
    const TlbAnalysisExecutionSeam* execution_seam);

}  // namespace

int run_tlb_analysis(const BenchmarkConfig& config) {
  return run_tlb_analysis_impl(
      config, []() { return signal_received(); }, nullptr);
}

int run_tlb_analysis(const BenchmarkConfig& config,
                     const TlbStopRequested& stop_requested) {
  return run_tlb_analysis_impl(config, stop_requested, nullptr);
}

int run_tlb_analysis(const BenchmarkConfig& config,
                     const TlbStopRequested& stop_requested,
                     const TlbAnalysisExecutionSeam& execution_seam) {
  return run_tlb_analysis_impl(config, stop_requested, &execution_seam);
}

namespace {

int run_tlb_analysis_impl(
    const BenchmarkConfig& config,
    const TlbStopRequested& stop_requested,
    const TlbAnalysisExecutionSeam* execution_seam) {
  std::optional<std::chrono::steady_clock::time_point> analysis_start;
  if (execution_seam == nullptr || !execution_seam->elapsed_seconds) {
    analysis_start = std::chrono::steady_clock::now();
  }
  std::cout << Messages::usage_header(SOFTVERSION);
  std::cout << Messages::msg_running_tlb_analysis() << std::endl;

  bool interrupted = false;
  bool interrupt_reported = false;
  auto report_interrupt_once = [&]() {
    if (!interrupt_reported) {
      std::cout << std::endl << Messages::msg_interrupted_by_user() << std::endl;
      interrupt_reported = true;
    }
  };

  const size_t analysis_stride_bytes = config.latency_stride_bytes;
  const TlbSweepDensity sweep_density = config.tlb_sweep_density;
  const TlbRuntimeProfile runtime_profile =
      tlb_runtime_profile_for_density(sweep_density);
  const bool refinement_enabled = tlb_density_enables_refinement(sweep_density);

  if (execution_seam != nullptr &&
      (execution_seam->page_size_bytes < sizeof(uintptr_t) ||
       execution_seam->l1_cache_size_bytes == 0 ||
       execution_seam->selected_buffer_mb == 0 ||
       execution_seam->available_memory_mb == 0 ||
       !execution_seam->execute_pass)) {
    return EXIT_FAILURE;
  }

  const size_t page_size_bytes = execution_seam != nullptr
                                     ? execution_seam->page_size_bytes
                                     : static_cast<size_t>(getpagesize());
  if (analysis_stride_bytes == 0) {
    std::cerr << Messages::error_prefix()
              << Messages::error_latency_stride_invalid(0, 1, std::numeric_limits<long long>::max())
              << std::endl;
    return EXIT_FAILURE;
  }
  if ((analysis_stride_bytes % sizeof(uintptr_t)) != 0) {
    std::cerr << Messages::error_prefix()
              << Messages::error_latency_stride_alignment(analysis_stride_bytes, sizeof(uintptr_t))
              << std::endl;
    return EXIT_FAILURE;
  }
  if (analysis_stride_bytes > page_size_bytes) {
    std::cerr << Messages::error_prefix()
              << Messages::error_analyze_tlb_stride_exceeds_page(analysis_stride_bytes, page_size_bytes)
              << std::endl;
    return EXIT_FAILURE;
  }
  const size_t l1_cache_size_bytes = execution_seam != nullptr
                                         ? execution_seam->l1_cache_size_bytes
                                         : get_l1_cache_size();
  const std::string cpu_name = execution_seam != nullptr
                                   ? execution_seam->cpu_name
                                   : get_processor_name();

  MmapPtr latency_buffer(nullptr, MmapDeleter{0});
  size_t selected_buffer_mb = 0;
  size_t selected_buffer_bytes = 0;
  size_t estimated_peak_memory_bytes = 0;
  const size_t available_memory_mb = execution_seam != nullptr
                                         ? execution_seam->available_memory_mb
                                         : get_available_memory_mb();
  const size_t memory_budget_mb =
      calculate_tlb_memory_budget_mb(available_memory_mb);

  if (execution_seam != nullptr) {
    if (execution_seam->selected_buffer_mb >
        std::numeric_limits<size_t>::max() / Constants::BYTES_PER_MB) {
      return EXIT_FAILURE;
    }
    selected_buffer_mb = execution_seam->selected_buffer_mb;
    selected_buffer_bytes = selected_buffer_mb * Constants::BYTES_PER_MB;
    estimated_peak_memory_bytes = estimate_tlb_peak_memory_bytes(
        selected_buffer_bytes, selected_buffer_bytes / page_size_bytes);
  } else {
    for (size_t candidate_mb : kBufferCandidateMb) {
      size_t candidate_peak_memory_bytes = 0;
      if (!tlb_buffer_fits_memory_budget(candidate_mb,
                                         page_size_bytes,
                                         memory_budget_mb,
                                         candidate_peak_memory_bytes)) {
        continue;
      }
      const size_t candidate_bytes = candidate_mb * Constants::BYTES_PER_MB;
      MmapPtr candidate_buffer = try_allocate_analysis_buffer(candidate_bytes);
      if (candidate_buffer != nullptr) {
        latency_buffer = std::move(candidate_buffer);
        selected_buffer_mb = candidate_mb;
        selected_buffer_bytes = candidate_bytes;
        estimated_peak_memory_bytes = candidate_peak_memory_bytes;
        break;
      }
    }
  }

  if ((execution_seam == nullptr && latency_buffer == nullptr) ||
      selected_buffer_mb == 0) {
    std::cerr << Messages::error_prefix()
              << Messages::error_tlb_analysis_insufficient_memory()
              << std::endl;
    return EXIT_FAILURE;
  }

  const size_t pointer_count = selected_buffer_bytes / analysis_stride_bytes;
  if (pointer_count < 2) {
    std::cerr << Messages::error_prefix()
              << Messages::error_buffer_stride_invalid_latency_chain(
                     pointer_count, selected_buffer_bytes, analysis_stride_bytes)
              << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<TlbSweepPoint> sweep_points =
      build_tlb_base_sweep_plan(analysis_stride_bytes, page_size_bytes, sweep_density);
  if (sweep_points.empty()) {
    std::cerr << Messages::error_prefix()
              << Messages::error_buffer_stride_invalid_latency_chain(
                     pointer_count, selected_buffer_bytes, analysis_stride_bytes)
              << std::endl;
    return EXIT_FAILURE;
  }
  const std::vector<size_t> localities_bytes = tlb_point_localities(sweep_points);

  const size_t page_walk_baseline_locality_bytes = localities_bytes.front();
  const LatencyChainMode effective_chain_mode =
      resolve_latency_chain_mode(config.latency_chain_mode, page_walk_baseline_locality_bytes);
  const bool can_measure_page_walk_penalty = selected_buffer_mb >= kPageWalkMinimumBufferMb;

  bool buffer_locked = false;
  int buffer_lock_errno = 0;
  std::string buffer_lock_error;
  if (execution_seam == nullptr) {
    if (mlock(latency_buffer.get(), selected_buffer_bytes) == 0) {
      buffer_locked = true;
    } else {
      buffer_lock_errno = errno;
      buffer_lock_error = std::strerror(buffer_lock_errno);
      std::cerr << Messages::warning_prefix()
                << Messages::warning_tlb_mlock_failed(buffer_lock_errno,
                                                       buffer_lock_error)
                << std::endl;
    }
  }

  std::optional<HighResTimer> timer_opt;
  if (execution_seam == nullptr) {
    timer_opt = HighResTimer::create();
  }
  if (execution_seam == nullptr && !timer_opt) {
    if (buffer_locked) {
      (void)munlock(latency_buffer.get(), selected_buffer_bytes);
    }
    std::cerr << Messages::error_prefix()
              << Messages::error_tlb_analysis_timer_creation_failed()
              << std::endl;
    return EXIT_FAILURE;
  }
  const TlbWorkEstimate base_work_estimate = estimate_tlb_work(
      sweep_points.size(),
      estimated_peak_memory_bytes,
      maximum_tlb_node_count(sweep_points),
      runtime_profile);

  std::cout << std::endl;
  std::cout << Messages::report_tlb_settings_header() << std::endl;
  std::cout << Messages::report_tlb_run_summary(
                   cpu_name,
                   page_size_bytes,
                   analysis_stride_bytes,
                   runtime_profile.name,
                   latency_chain_mode_to_string(config.latency_chain_mode),
                   latency_chain_mode_to_string(effective_chain_mode),
                   config.tlb_seed,
                   config.user_specified_tlb_seed)
            << std::endl;
  std::cout << Messages::report_tlb_resource_summary(
                   selected_buffer_mb,
                   buffer_locked,
                   config.main_thread_qos_requested,
                   config.main_thread_qos_applied,
                   config.main_thread_qos_code,
                   memory_budget_mb,
                   estimated_peak_memory_bytes)
            << std::endl;
  std::cout << Messages::report_tlb_sweep_plan(
                   localities_bytes.front(),
                   localities_bytes.back(),
                   localities_bytes.size(),
                   can_measure_page_walk_penalty,
                   kPageWalkComparisonLocalityBytes,
                   kPageWalkMinimumBufferMb,
                   selected_buffer_mb)
            << std::endl;
  print_tlb_work_estimate("base", base_work_estimate);
  std::cout << std::endl;

  std::vector<LocalityMeasurement> measurements;
  measurements.reserve(localities_bytes.size() + 16);
  std::vector<TlbMeasurementRecord> measurement_records;
  measurement_records.reserve(
      localities_bytes.size() * runtime_profile.max_rounds);
  std::vector<TlbPassExecutionSummary> pass_summaries;
  pass_summaries.reserve(4);
  bool measurement_error = false;

  auto execute_measurement_pass =
      [&](const std::vector<TlbSweepPoint>& points,
          TlbMeasurementPass pass,
          std::vector<LocalityMeasurement>& pass_measurements) {
        if (execution_seam != nullptr && stop_requested && stop_requested()) {
          TlbScheduleExecutionResult result;
          result.status = TlbScheduleExecutionStatus::Interrupted;
          return result;
        }
        if (execution_seam != nullptr) {
          TlbScheduleExecutionResult result =
              execution_seam->execute_pass(pass, points);
          append_measurement_records(result.records, pass_measurements);
          return result;
        }
        return measure_scheduled_points(
            latency_buffer.get(),
            selected_buffer_bytes,
            analysis_stride_bytes,
            page_size_bytes,
            points,
            effective_chain_mode,
            *timer_opt,
            config.tlb_seed,
            pass,
            runtime_profile,
            stop_requested,
            pass_measurements);
      };

  std::cout << std::fixed;
  std::cout.precision(Constants::LATENCY_PRECISION);

  const TlbScheduleExecutionResult base_result = execute_measurement_pass(
      sweep_points, TlbMeasurementPass::Base, measurements);
  measurement_records.insert(measurement_records.end(),
                             base_result.records.begin(),
                             base_result.records.end());
  if (base_result.status == TlbScheduleExecutionStatus::Error) {
    measurement_error = true;
  }
  pass_summaries.push_back(summarize_tlb_pass(
      TlbMeasurementPass::Base, sweep_points.size(), base_result));
  print_tlb_pass_completion(TlbMeasurementPass::Base, base_result);
  if (base_result.status == TlbScheduleExecutionStatus::Interrupted) {
    interrupted = true;
    report_interrupt_once();
  }

  if (measurements.empty() && !interrupted && !measurement_error) {
    if (buffer_locked) {
      (void)munlock(latency_buffer.get(), selected_buffer_bytes);
    }
    return EXIT_FAILURE;
  }

  const bool base_sweep_complete =
      base_result.status == TlbScheduleExecutionStatus::Complete;

  std::cout << Messages::report_tlb_sweep_legend() << std::endl;
  std::sort(measurements.begin(), measurements.end(), [](const LocalityMeasurement& a, const LocalityMeasurement& b) {
    return a.locality_bytes < b.locality_bytes;
  });
  for (size_t locality_index = 0; locality_index < measurements.size(); ++locality_index) {
    const TlbPairedPointSummary summary = summarize_tlb_paired_point(
        measurement_records,
        measurements[locality_index].locality_bytes,
        {TlbMeasurementPass::Base});
    if (!summary.available) {
      continue;
    }
    std::cout << Messages::report_tlb_paired_locality_progress(
                     locality_index + 1,
                     localities_bytes.size(),
                     measurements[locality_index].locality_bytes,
                     summary.spread_p50_ns,
                     summary.packed_p50_ns,
                     summary.translation_delta_p50_ns,
                     summary.active_cache_line_footprint_bytes,
                     summary.short_cycle_diagnostic)
              << std::endl;
  }

  std::vector<size_t> final_localities_bytes;
  std::vector<double> p50_latency_ns;
  std::vector<std::vector<double>> sweep_loop_latencies_ns;
  flatten_measurements(measurements, final_localities_bytes, p50_latency_ns, sweep_loop_latencies_ns);

  const size_t tlb_guard_bytes = std::max<size_t>(2 * l1_cache_size_bytes, 64 * page_size_bytes);

  PrivateCacheKneeDetection private_cache_knee =
      detect_private_cache_knee(final_localities_bytes, p50_latency_ns, &sweep_loop_latencies_ns);

  TlbBoundaryDetection l1_boundary = detect_l1_tlb_boundary(final_localities_bytes,
                                                            p50_latency_ns,
                                                            tlb_guard_bytes,
                                                            private_cache_knee,
                                                            sweep_loop_latencies_ns);

  TlbBoundaryDetection l2_boundary;
  if (l1_boundary.detected && l1_boundary.boundary_index < final_localities_bytes.size() - 1) {
    const size_t l2_segment_start = std::min(l1_boundary.boundary_index + 2,
                                             final_localities_bytes.size() - 2);
    const size_t l2_guard_bytes = std::max(tlb_guard_bytes, l1_boundary.boundary_locality_bytes);
    l2_boundary = detect_tlb_boundary(final_localities_bytes,
                                      p50_latency_ns,
                                      l2_segment_start,
                                      l2_guard_bytes,
                                      &sweep_loop_latencies_ns);
  }

  std::vector<TlbRefinementTarget> refinement_targets;
  if (refinement_enabled && base_sweep_complete) {
    if (private_cache_knee.detected) {
      refinement_targets.push_back(
          TlbRefinementTarget{private_cache_knee.boundary_index, "private-cache"});
    }
    if (l1_boundary.detected) {
      refinement_targets.push_back(TlbRefinementTarget{l1_boundary.boundary_index, "l1"});
    }
    if (l2_boundary.detected) {
      refinement_targets.push_back(TlbRefinementTarget{l2_boundary.boundary_index, "l2"});
    }
  }

  std::vector<TlbSweepPoint> refinement_points = build_tlb_refinement_plan(
      final_localities_bytes,
      refinement_targets,
      analysis_stride_bytes,
      page_size_bytes,
      localities_bytes.front(),
      localities_bytes.back());
  for (size_t i = 0; i < refinement_points.size(); ++i) {
    refinement_points[i].point_index = sweep_points.size() + i;
  }
  const size_t planned_points = sweep_points.size() + refinement_points.size();

  if (!refinement_points.empty() && !interrupted && !measurement_error) {
    std::cout << Messages::msg_tlb_analysis_refinement_start(refinement_points.size()) << std::endl;
    print_tlb_work_estimate(
        "refinement",
        estimate_tlb_work(refinement_points.size(),
                          estimated_peak_memory_bytes,
                          maximum_tlb_node_count(refinement_points),
                          runtime_profile));
  }

  size_t fine_sweep_added_points = 0;
  bool refinement_complete = refinement_points.empty();
  if (!refinement_points.empty() && !interrupted && !measurement_error) {
    const size_t measurements_before_refinement = measurements.size();
    const TlbScheduleExecutionResult refinement_result =
        execute_measurement_pass(refinement_points,
                                 TlbMeasurementPass::Refinement,
                                 measurements);
    measurement_records.insert(measurement_records.end(),
                               refinement_result.records.begin(),
                               refinement_result.records.end());
    if (refinement_result.status == TlbScheduleExecutionStatus::Error) {
      measurement_error = true;
    }
    pass_summaries.push_back(summarize_tlb_pass(
        TlbMeasurementPass::Refinement,
        refinement_points.size(),
        refinement_result));
    print_tlb_pass_completion(TlbMeasurementPass::Refinement,
                              refinement_result);
    if (refinement_result.status == TlbScheduleExecutionStatus::Interrupted) {
      interrupted = true;
      report_interrupt_once();
    }
    refinement_complete =
        refinement_result.status == TlbScheduleExecutionStatus::Complete;
    fine_sweep_added_points = measurements.size() - measurements_before_refinement;
  }
  sweep_points.insert(sweep_points.end(), refinement_points.begin(), refinement_points.end());

  std::sort(measurements.begin(), measurements.end(), [](const LocalityMeasurement& a, const LocalityMeasurement& b) {
    return a.locality_bytes < b.locality_bytes;
  });
  flatten_measurements(measurements, final_localities_bytes, p50_latency_ns, sweep_loop_latencies_ns);

  if (!interrupted && stop_requested && stop_requested()) {
    interrupted = true;
    report_interrupt_once();
  }
  const bool main_sweep_complete =
      base_sweep_complete && refinement_complete && !interrupted &&
      !measurement_error &&
      final_localities_bytes.size() == planned_points;

  private_cache_knee = detect_private_cache_knee(final_localities_bytes, p50_latency_ns, &sweep_loop_latencies_ns);

  const TlbRoundPointMatrix discovery_matrix =
      build_tlb_translation_delta_matrix(
          final_localities_bytes,
          measurement_records,
          {TlbMeasurementPass::Base, TlbMeasurementPass::Refinement});
  TlbBoundaryDetection l1_discovery;
  TlbBoundaryDetection l2_discovery;
  if (main_sweep_complete) {
    l1_discovery = detect_tlb_boundary_robust(final_localities_bytes,
                                              discovery_matrix,
                                              nullptr,
                                              0,
                                              tlb_guard_bytes,
                                              config.tlb_seed);
    if (l1_discovery.discovery.passed &&
        l1_discovery.boundary_index + 2 < final_localities_bytes.size()) {
      const size_t l2_segment_start = l1_discovery.boundary_index + 1;
      const size_t l2_guard_bytes = std::max(
          tlb_guard_bytes, l1_discovery.boundary_locality_bytes);
      l2_discovery = detect_tlb_boundary_robust(
          final_localities_bytes,
          discovery_matrix,
          nullptr,
          l2_segment_start,
          l2_guard_bytes,
          config.tlb_seed ^ 0x94d049bb133111ebULL);
    }
  }

  const std::vector<TlbSweepPoint> validation_points = build_validation_plan(
      final_localities_bytes,
      l1_discovery,
      l2_discovery,
      analysis_stride_bytes,
      page_size_bytes,
      sweep_points.size());
  bool validation_complete = validation_points.empty();
  size_t validation_measured_points = 0;
  if (!validation_points.empty() && !interrupted && !measurement_error) {
    std::cout << Messages::msg_tlb_analysis_validation_start(
                     validation_points.size())
              << std::endl;
    print_tlb_work_estimate(
        "validation",
        estimate_tlb_work(validation_points.size(),
                          estimated_peak_memory_bytes,
                          maximum_tlb_node_count(validation_points),
                          runtime_profile));
    std::vector<LocalityMeasurement> validation_measurements;
    const TlbScheduleExecutionResult validation_result =
        execute_measurement_pass(validation_points,
                                 TlbMeasurementPass::Validation,
                                 validation_measurements);
    measurement_records.insert(measurement_records.end(),
                               validation_result.records.begin(),
                               validation_result.records.end());
    validation_measured_points = validation_measurements.size();
    if (validation_result.status == TlbScheduleExecutionStatus::Error) {
      measurement_error = true;
    }
    pass_summaries.push_back(summarize_tlb_pass(
        TlbMeasurementPass::Validation,
        validation_points.size(),
        validation_result));
    print_tlb_pass_completion(TlbMeasurementPass::Validation,
                              validation_result);
    if (validation_result.status == TlbScheduleExecutionStatus::Interrupted) {
      interrupted = true;
      report_interrupt_once();
    }
    validation_complete =
        validation_result.status == TlbScheduleExecutionStatus::Complete &&
        validation_measured_points == validation_points.size();
  }

  const TlbRoundPointMatrix validation_matrix =
      build_tlb_translation_delta_matrix(
          final_localities_bytes,
          measurement_records,
          {TlbMeasurementPass::Validation});
  l1_boundary = TlbBoundaryDetection{};
  l2_boundary = TlbBoundaryDetection{};
  if (main_sweep_complete && validation_complete && !interrupted &&
      !measurement_error) {
    l1_boundary = detect_tlb_boundary_robust(final_localities_bytes,
                                             discovery_matrix,
                                             &validation_matrix,
                                             0,
                                             tlb_guard_bytes,
                                             config.tlb_seed);
    if (l1_boundary.detected &&
        l1_boundary.boundary_index + 2 < final_localities_bytes.size()) {
      const size_t l2_segment_start = l1_boundary.boundary_index + 1;
      const size_t l2_guard_bytes = std::max(
          tlb_guard_bytes, l1_boundary.boundary_locality_bytes);
      l2_boundary = detect_tlb_boundary_robust(
          final_localities_bytes,
          discovery_matrix,
          &validation_matrix,
          l2_segment_start,
          l2_guard_bytes,
          config.tlb_seed ^ 0x94d049bb133111ebULL);
    }
  }

  size_t private_cache_to_l1_distance_bytes = 0;
  size_t private_cache_to_l1_distance_pages = 0;
  bool private_cache_interference_elevated = false;
  if (private_cache_knee.detected) {
    if (l1_boundary.detected && l1_boundary.boundary_locality_bytes > private_cache_knee.boundary_locality_bytes) {
      private_cache_to_l1_distance_bytes =
          l1_boundary.boundary_locality_bytes - private_cache_knee.boundary_locality_bytes;
      private_cache_to_l1_distance_pages =
          (page_size_bytes > 0) ? (private_cache_to_l1_distance_bytes / page_size_bytes) : 0;

      bool l1_within_two_x_locality = false;
      if (private_cache_knee.boundary_locality_bytes <= (std::numeric_limits<size_t>::max() / 2)) {
        l1_within_two_x_locality =
            l1_boundary.boundary_locality_bytes <= (2 * private_cache_knee.boundary_locality_bytes);
      }

      if (private_cache_knee.strong_private_cache_candidate) {
        private_cache_interference_elevated =
            l1_within_two_x_locality || (private_cache_to_l1_distance_pages < 128);
      } else {
        private_cache_interference_elevated = private_cache_to_l1_distance_pages < 64;
      }
    } else {
      private_cache_interference_elevated = private_cache_knee.strong_private_cache_candidate;
    }
  }

  std::vector<double> page_walk_512mb_loop_latencies_ns;
  double page_walk_512mb_p50_ns = 0.0;
  TlbPairedPointSummary large_locality_summary;
  bool page_walk_comparison_completed = false;
  bool large_locality_pass_completed = false;
  const bool large_locality_planned =
      can_measure_page_walk_penalty && main_sweep_complete &&
      validation_complete && !interrupted && !measurement_error;
  if (large_locality_planned) {
    TlbSweepPoint comparison_point;
    comparison_point.point_index =
        planned_points + validation_points.size();
    comparison_point.requested_pages = kPageWalkComparisonLocalityBytes / page_size_bytes;
    comparison_point.effective_pages = comparison_point.requested_pages;
    comparison_point.locality_bytes = kPageWalkComparisonLocalityBytes;
    comparison_point.stride_bytes = analysis_stride_bytes;
    comparison_point.pointer_count = comparison_point.requested_pages;
    comparison_point.refinement_source = "large-locality";
    print_tlb_work_estimate(
        "large-locality",
        estimate_tlb_work(1,
                          estimated_peak_memory_bytes,
                          comparison_point.requested_pages,
                          runtime_profile));
    std::vector<LocalityMeasurement> comparison_measurements;
    const TlbScheduleExecutionResult comparison_result =
        execute_measurement_pass({comparison_point},
                                 TlbMeasurementPass::LargeLocality,
                                 comparison_measurements);
    measurement_records.insert(measurement_records.end(),
                               comparison_result.records.begin(),
                               comparison_result.records.end());
    pass_summaries.push_back(summarize_tlb_pass(
        TlbMeasurementPass::LargeLocality, 1, comparison_result));
    print_tlb_pass_completion(TlbMeasurementPass::LargeLocality,
                              comparison_result);
    if (comparison_result.status == TlbScheduleExecutionStatus::Interrupted) {
      interrupted = true;
      report_interrupt_once();
    } else if (comparison_result.status == TlbScheduleExecutionStatus::Error) {
      measurement_error = true;
    }
    large_locality_pass_completed =
        comparison_result.status == TlbScheduleExecutionStatus::Complete;
    if (comparison_result.status == TlbScheduleExecutionStatus::Complete &&
        !comparison_measurements.empty()) {
      page_walk_512mb_p50_ns = comparison_measurements.front().p50_latency_ns;
      page_walk_512mb_loop_latencies_ns =
          comparison_measurements.front().loop_latencies_ns;
      large_locality_summary = summarize_tlb_paired_point(
          measurement_records,
          kPageWalkComparisonLocalityBytes,
          {TlbMeasurementPass::LargeLocality});
      page_walk_comparison_completed = large_locality_summary.available;
    }
  }

  if (!interrupted && stop_requested && stop_requested()) {
    interrupted = true;
    report_interrupt_once();
  }

  const bool conclusions_valid =
      main_sweep_complete && validation_complete && !interrupted &&
      !measurement_error;
  const std::string analysis_status = measurement_error
                                          ? "error"
                                          : (conclusions_valid
                                                 ? "complete"
                                                 : (interrupted
                                                        ? "interrupted"
                                                        : "partial"));
  if (!conclusions_valid) {
    private_cache_knee = PrivateCacheKneeDetection{};
    l1_boundary = TlbBoundaryDetection{};
    l2_boundary = TlbBoundaryDetection{};
    private_cache_to_l1_distance_bytes = 0;
    private_cache_to_l1_distance_pages = 0;
    private_cache_interference_elevated = false;
    page_walk_comparison_completed = false;
  }

  if (buffer_locked) {
    (void)munlock(latency_buffer.get(), selected_buffer_bytes);
  }

  const std::pair<size_t, size_t> l1_entry_range =
      validated_entry_range(l1_boundary, page_size_bytes);
  const std::pair<size_t, size_t> l2_entry_range =
      validated_entry_range(l2_boundary, page_size_bytes);
  const size_t l1_entries = l1_boundary.detected
                                ? l1_entry_range.first +
                                      ((l1_entry_range.second -
                                        l1_entry_range.first) /
                                       2)
                                : 0;
  const size_t l2_entries = l2_boundary.detected
                                 ? l2_entry_range.first +
                                       ((l2_entry_range.second -
                                         l2_entry_range.first) /
                                        2)
                                 : 0;

  const double page_walk_baseline_ns = p50_latency_ns.empty() ? 0.0 : p50_latency_ns.front();
  const double page_walk_penalty_ns =
      page_walk_comparison_completed ? (page_walk_512mb_p50_ns - page_walk_baseline_ns) : 0.0;

  std::cout << std::endl;
  std::cout << Messages::report_tlb_header() << std::endl;
  std::cout << Messages::report_tlb_run_summary(
                   cpu_name,
                   page_size_bytes,
                   analysis_stride_bytes,
                   runtime_profile.name,
                   latency_chain_mode_to_string(config.latency_chain_mode),
                   latency_chain_mode_to_string(effective_chain_mode),
                   config.tlb_seed,
                   config.user_specified_tlb_seed)
            << std::endl;
  if (fine_sweep_added_points > 0) {
    std::cout << Messages::report_tlb_fine_sweep(
                     fine_sweep_added_points,
                     final_localities_bytes.size())
              << std::endl;
  }
  std::cout << Messages::report_tlb_analysis_status(analysis_status,
                                                    planned_points,
                                                    final_localities_bytes.size(),
                                                    conclusions_valid)
            << std::endl;
  if (runtime_profile.name == "quick") {
    std::cout << Messages::report_tlb_quick_profile_note() << std::endl;
  }

  std::cout << std::endl;
  std::cout << Messages::report_tlb_private_cache_section() << std::endl;
  if (!conclusions_valid) {
    std::cout << Messages::report_tlb_conclusions_unavailable(analysis_status) << std::endl;
  } else if (private_cache_knee.detected) {
    const size_t private_cache_locality_kb =
        private_cache_knee.boundary_locality_bytes / Constants::BYTES_PER_KB;
    std::cout << Messages::report_tlb_boundary_kb(
                     private_cache_locality_kb)
              << std::endl;
    std::cout << Messages::report_tlb_confidence(private_cache_knee.confidence,
                                                 private_cache_knee.step_ns,
                                                 private_cache_knee.step_percent)
              << std::endl;
    std::cout << Messages::report_tlb_private_cache_candidate(
                     private_cache_knee.strong_private_cache_candidate)
              << std::endl;
    std::cout << Messages::report_tlb_private_cache_interference(
                     private_cache_interference_elevated,
                     private_cache_locality_kb)
              << std::endl;
    if (private_cache_to_l1_distance_bytes > 0) {
      std::cout << Messages::report_tlb_private_cache_l1_distance(
                       private_cache_to_l1_distance_bytes / Constants::BYTES_PER_KB,
                       private_cache_to_l1_distance_pages)
                << std::endl;
    }
  } else {
    std::cout << Messages::report_tlb_not_detected() << std::endl;
  }

  std::cout << std::endl;
  std::cout << Messages::report_tlb_l1_section() << std::endl;
  if (!conclusions_valid) {
    std::cout << Messages::report_tlb_conclusions_unavailable(analysis_status) << std::endl;
  } else if (l1_boundary.detected) {
    std::cout << Messages::report_tlb_boundary_kb(
                     l1_boundary.boundary_locality_bytes / Constants::BYTES_PER_KB)
              << std::endl;
    std::cout << Messages::report_tlb_inferred_entries_range(
                     l1_entry_range.first,
                     l1_entry_range.second)
              << std::endl;
    std::cout << Messages::report_tlb_inferred_size_entries(l1_entries) << std::endl;
    if (l1_boundary.overlaps_private_cache_knee) {
      std::cout << Messages::report_tlb_private_cache_overlap() << std::endl;
    }
    std::cout << Messages::report_tlb_statistical_confidence(
                     l1_boundary.confidence,
                     l1_boundary.discovery.effect_ns,
                     l1_boundary.discovery.effect_ci.lower_ns,
                     l1_boundary.discovery.effect_ci.upper_ns,
                     l1_boundary.validation.effect_ci.lower_ns,
                     l1_boundary.validation.effect_ci.upper_ns)
              << std::endl;
  } else {
    std::cout << Messages::report_tlb_not_detected() << std::endl;
  }

  std::cout << std::endl;
  std::cout << Messages::report_tlb_l2_section() << std::endl;
  if (!conclusions_valid) {
    std::cout << Messages::report_tlb_conclusions_unavailable(analysis_status) << std::endl;
  } else if (l2_boundary.detected) {
    std::cout << Messages::report_tlb_boundary_kb(
                     l2_boundary.boundary_locality_bytes / Constants::BYTES_PER_KB)
              << std::endl;
    std::cout << Messages::report_tlb_inferred_entries_range(
                     l2_entry_range.first,
                     l2_entry_range.second)
              << std::endl;
    std::cout << Messages::report_tlb_inferred_reach_entries(l2_entries) << std::endl;
    std::cout << Messages::report_tlb_statistical_confidence(
                     l2_boundary.confidence,
                     l2_boundary.discovery.effect_ns,
                     l2_boundary.discovery.effect_ci.lower_ns,
                     l2_boundary.discovery.effect_ci.upper_ns,
                     l2_boundary.validation.effect_ci.lower_ns,
                     l2_boundary.validation.effect_ci.upper_ns)
              << std::endl;
  } else {
    std::cout << Messages::report_tlb_not_detected() << std::endl;
  }
  std::cout << std::endl;
  if (!conclusions_valid ||
      (can_measure_page_walk_penalty && !page_walk_comparison_completed)) {
    std::cout << Messages::report_tlb_large_locality_paired_interrupted()
              << std::endl;
  } else if (page_walk_comparison_completed) {
    std::cout << Messages::report_tlb_large_locality_paired_comparison(
                     kPageWalkComparisonLocalityBytes,
                     large_locality_summary.spread_p50_ns,
                     large_locality_summary.packed_p50_ns,
                     large_locality_summary.translation_delta_p50_ns,
                     large_locality_summary.spread_actual_pages,
                     large_locality_summary.packed_actual_pages,
                     large_locality_summary.unique_cache_lines,
                     large_locality_summary.active_cache_line_footprint_bytes)
              << std::endl;
  } else {
    std::cout << Messages::report_tlb_large_locality_paired_unavailable(
                     kPageWalkMinimumBufferMb,
                     selected_buffer_mb)
              << std::endl;
  }

  const double total_execution_time_sec =
      execution_seam != nullptr && execution_seam->elapsed_seconds
          ? execution_seam->elapsed_seconds()
          : std::chrono::duration<double>(
                std::chrono::steady_clock::now() - *analysis_start)
                .count();
  if (execution_seam != nullptr && execution_seam->observe_summary) {
    TlbAnalysisCoordinatorSummary summary;
    if (measurement_error) {
      summary.status = TlbAnalysisCoordinatorStatus::Error;
    } else if (interrupted) {
      summary.status = TlbAnalysisCoordinatorStatus::Interrupted;
    } else if (conclusions_valid) {
      summary.status = TlbAnalysisCoordinatorStatus::Complete;
    } else {
      summary.status = TlbAnalysisCoordinatorStatus::Partial;
    }
    summary.status_text = analysis_status;
    summary.planned_points = planned_points;
    summary.completed_points = final_localities_bytes.size();
    summary.validation_planned_points = validation_points.size();
    summary.validation_completed_points = validation_measured_points;
    summary.planned_passes = pass_summaries.size();
    summary.completed_passes = static_cast<size_t>(std::count_if(
        pass_summaries.begin(),
        pass_summaries.end(),
        [](const TlbPassExecutionSummary& pass_summary) {
          return pass_summary.complete;
        }));
    summary.measurement_record_count = measurement_records.size();
    summary.elapsed_seconds = total_execution_time_sec;
    summary.conclusions_valid = conclusions_valid;
    summary.large_locality_planned = large_locality_planned;
    summary.large_locality_completed = large_locality_pass_completed;
    summary.pass_summaries = pass_summaries;
    execution_seam->observe_summary(summary);
  }
  const TlbAnalysisJsonContext json_context = {
      config,
      cpu_name,
      config.perf_cores,
      config.eff_cores,
      page_size_bytes,
      l1_cache_size_bytes,
      tlb_guard_bytes,
      analysis_stride_bytes,
      runtime_profile.max_rounds,
      runtime_profile.maximum_accesses,
      page_walk_baseline_locality_bytes,
      kPageWalkComparisonLocalityBytes,
      selected_buffer_mb,
      buffer_locked,
      analysis_status,
      planned_points,
      final_localities_bytes.size(),
      validation_points.size(),
      validation_measured_points,
      validation_complete,
      conclusions_valid,
      sweep_points,
      measurement_records,
      final_localities_bytes,
      sweep_loop_latencies_ns,
      p50_latency_ns,
      l1_boundary,
      l2_boundary,
      private_cache_knee,
      l1_entries,
      l2_entries,
      l1_entry_range.first,
      l1_entry_range.second,
      l2_entry_range.first,
      l2_entry_range.second,
      fine_sweep_added_points,
      private_cache_interference_elevated,
      private_cache_to_l1_distance_bytes,
      private_cache_to_l1_distance_pages,
      can_measure_page_walk_penalty,
      page_walk_comparison_completed,
      page_walk_512mb_loop_latencies_ns,
      page_walk_512mb_p50_ns,
      page_walk_baseline_ns,
      page_walk_penalty_ns,
      total_execution_time_sec,
      runtime_profile,
      available_memory_mb,
      memory_budget_mb,
      estimated_peak_memory_bytes,
      buffer_lock_errno,
      buffer_lock_error,
      base_work_estimate,
      pass_summaries,
  };

  if (save_tlb_analysis_to_json(json_context) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }

  return measurement_error ? EXIT_FAILURE : EXIT_SUCCESS;
}

}  // namespace

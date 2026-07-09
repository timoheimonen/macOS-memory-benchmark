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
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

#include "benchmark/benchmark_tests.h"
#include "benchmark/tlb_analysis_json.h"
#include "benchmark/tlb_measurement_scheduler.h"
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

// Fixed analysis parameters for standalone TLB mode.
constexpr size_t kLoopsPerPoint = 30;
constexpr size_t kAccessesPerLoop = 25 * 1000 * 1000;
constexpr size_t kPageWalkComparisonLocalityBytes = 512 * Constants::BYTES_PER_MB;
constexpr size_t kPageWalkMinimumBufferMb = 512;

const std::array<size_t, 3> kBufferCandidateMb = {1024, 512, 256};

struct LocalityMeasurement {
  size_t locality_bytes = 0;
  double p50_latency_ns = 0.0;
  std::vector<double> loop_latencies_ns;
};

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
    const std::vector<TlbSweepPoint>& points,
    LatencyChainMode chain_mode,
    HighResTimer& timer,
    uint64_t base_seed,
    TlbMeasurementPass pass,
    const TlbStopRequested& stop_requested,
    std::vector<LocalityMeasurement>& measurements) {
  TlbMeasureSpinner spinner;
  const std::vector<TlbMeasurementTask> schedule =
      build_tlb_measurement_schedule(points, kLoopsPerPoint, base_seed, pass);
  TlbScheduleExecutionResult result = execute_tlb_measurement_schedule(
      schedule,
      stop_requested,
      [&](const TlbMeasurementTask& task, double& latency_ns) {
        const size_t locality_kb = task.locality_bytes / Constants::BYTES_PER_KB;
        spinner.tick(locality_kb, task.round_index + 1, kLoopsPerPoint);

        if (setup_latency_chain(latency_buffer,
                                buffer_size_bytes,
                                stride_bytes,
                                task.locality_bytes,
                                nullptr,
                                chain_mode,
                                task.seed) != EXIT_SUCCESS) {
          return TlbTaskMeasureStatus::Error;
        }

        warmup_latency(latency_buffer, buffer_size_bytes);
        const double total_latency_ns =
            run_latency_test(latency_buffer, kAccessesPerLoop, timer, nullptr, 0);
        if (total_latency_ns <= 0.0 || std::isnan(total_latency_ns) ||
            std::isinf(total_latency_ns)) {
          std::cerr << Messages::error_prefix()
                    << Messages::error_tlb_analysis_invalid_measurement(
                           locality_kb, static_cast<int>(task.round_index + 1))
                    << std::endl;
          return TlbTaskMeasureStatus::Error;
        }

        latency_ns = total_latency_ns / static_cast<double>(kAccessesPerLoop);
        return TlbTaskMeasureStatus::Success;
      });

  for (const TlbMeasurementRecord& record : result.records) {
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
  return result;
}

}  // namespace

/**
 * @brief Run standalone TLB analysis benchmark mode.
 *
 * Workflow:
 * - Print program header and run banner.
 * - Allocate analysis buffer with fallback order: 1024MB -> 512MB -> 256MB.
 * - Use configured latency stride (default from `--latency-stride-bytes`).
 * - Measure the page-aligned locality grid through balanced seeded rounds.
 * - Optionally run a separate 512MB large-locality comparison pass.
 * - Infer L1/L2 boundaries and emit the TLB analysis report.
 *
 * Large-locality delta:
 * - Reported as P50(512MB) - P50(effective baseline locality).
 * - Only computed when selected analysis buffer is at least 512MB and the comparison pass completes.
 *
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on allocation or measurement error.
 */
int run_tlb_analysis(const BenchmarkConfig& config) {
  return run_tlb_analysis(config, []() { return signal_received(); });
}

int run_tlb_analysis(const BenchmarkConfig& config,
                     const TlbStopRequested& stop_requested) {
  const auto analysis_start = std::chrono::steady_clock::now();
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
  const bool refinement_enabled = tlb_density_enables_refinement(sweep_density);

  const size_t page_size_bytes = static_cast<size_t>(getpagesize());
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
  if ((page_size_bytes % analysis_stride_bytes) != 0) {
    std::cerr << Messages::error_prefix()
              << Messages::error_analyze_tlb_stride_must_divide_page(analysis_stride_bytes, page_size_bytes)
              << std::endl;
    return EXIT_FAILURE;
  }
  const size_t l1_cache_size_bytes = get_l1_cache_size();
  const std::string cpu_name = get_processor_name();

  MmapPtr latency_buffer(nullptr, MmapDeleter{0});
  size_t selected_buffer_mb = 0;
  size_t selected_buffer_bytes = 0;

  for (size_t candidate_mb : kBufferCandidateMb) {
    const size_t candidate_bytes = candidate_mb * Constants::BYTES_PER_MB;
    MmapPtr candidate_buffer = try_allocate_analysis_buffer(candidate_bytes);
    if (candidate_buffer != nullptr) {
      latency_buffer = std::move(candidate_buffer);
      selected_buffer_mb = candidate_mb;
      selected_buffer_bytes = candidate_bytes;
      break;
    }
  }

  if (latency_buffer == nullptr || selected_buffer_mb == 0) {
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
  if (mlock(latency_buffer.get(), selected_buffer_bytes) == 0) {
    buffer_locked = true;
  }

  auto timer_opt = HighResTimer::create();
  if (!timer_opt) {
    if (buffer_locked) {
      (void)munlock(latency_buffer.get(), selected_buffer_bytes);
    }
    std::cerr << Messages::error_prefix()
              << Messages::error_tlb_analysis_timer_creation_failed()
              << std::endl;
    return EXIT_FAILURE;
  }
  auto& timer = *timer_opt;

  std::cout << std::endl;
  std::cout << Messages::report_tlb_settings_header() << std::endl;
  std::cout << Messages::report_tlb_cpu(cpu_name) << std::endl;
  std::cout << Messages::report_tlb_page_size(page_size_bytes) << std::endl;
  std::cout << Messages::report_tlb_buffer(selected_buffer_mb, buffer_locked) << std::endl;
  std::cout << Messages::report_tlb_stride(analysis_stride_bytes) << std::endl;
  std::cout << Messages::report_tlb_density(tlb_sweep_density_to_string(sweep_density)) << std::endl;
  std::cout << Messages::report_tlb_seed(config.tlb_seed,
                                         config.user_specified_tlb_seed)
            << std::endl;
  std::cout << Messages::report_tlb_schedule_policy() << std::endl;
  std::cout << Messages::report_tlb_chain_mode_requested(
                   latency_chain_mode_to_string(config.latency_chain_mode))
            << std::endl;
  std::cout << Messages::report_tlb_chain_mode_effective(
                   latency_chain_mode_to_string(effective_chain_mode))
            << std::endl;
  std::cout << Messages::report_tlb_loop_config(kLoopsPerPoint, kAccessesPerLoop) << std::endl;
  std::cout << Messages::report_tlb_sweep_range(localities_bytes.front(),
                                                localities_bytes.back(),
                                                localities_bytes.size())
            << std::endl;
  std::cout << Messages::report_tlb_page_walk_config(can_measure_page_walk_penalty,
                                                     kPageWalkComparisonLocalityBytes / Constants::BYTES_PER_MB,
                                                     kPageWalkMinimumBufferMb,
                                                     selected_buffer_mb)
            << std::endl;
  std::cout << std::endl;

  std::vector<LocalityMeasurement> measurements;
  measurements.reserve(localities_bytes.size() + 16);
  std::vector<TlbMeasurementRecord> measurement_records;
  measurement_records.reserve(localities_bytes.size() * kLoopsPerPoint);

  std::cout << std::fixed;
  std::cout.precision(Constants::LATENCY_PRECISION);

  const TlbScheduleExecutionResult base_result = measure_scheduled_points(
      latency_buffer.get(),
      selected_buffer_bytes,
      analysis_stride_bytes,
      sweep_points,
      config.latency_chain_mode,
      timer,
      config.tlb_seed,
      TlbMeasurementPass::Base,
      stop_requested,
      measurements);
  measurement_records.insert(measurement_records.end(),
                             base_result.records.begin(),
                             base_result.records.end());
  if (base_result.status == TlbScheduleExecutionStatus::Error) {
    if (buffer_locked) {
      (void)munlock(latency_buffer.get(), selected_buffer_bytes);
    }
    return EXIT_FAILURE;
  }
  if (base_result.status == TlbScheduleExecutionStatus::Interrupted) {
    interrupted = true;
    report_interrupt_once();
  }

  if (measurements.empty() && !interrupted) {
    if (buffer_locked) {
      (void)munlock(latency_buffer.get(), selected_buffer_bytes);
    }
    return EXIT_FAILURE;
  }

  const bool base_sweep_complete =
      base_result.status == TlbScheduleExecutionStatus::Complete;

  std::sort(measurements.begin(), measurements.end(), [](const LocalityMeasurement& a, const LocalityMeasurement& b) {
    return a.locality_bytes < b.locality_bytes;
  });
  for (size_t locality_index = 0; locality_index < measurements.size(); ++locality_index) {
    std::cout << Messages::msg_tlb_analysis_locality_progress(
                     locality_index + 1,
                     localities_bytes.size(),
                     measurements[locality_index].locality_bytes / Constants::BYTES_PER_KB)
              << " — " << measurements[locality_index].p50_latency_ns << " ns" << std::endl;
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

  if (!refinement_points.empty() && !interrupted) {
    std::cout << Messages::msg_tlb_analysis_refinement_start(refinement_points.size()) << std::endl;
  }

  size_t fine_sweep_added_points = 0;
  bool refinement_complete = refinement_points.empty();
  if (!refinement_points.empty() && !interrupted) {
    const size_t measurements_before_refinement = measurements.size();
    const TlbScheduleExecutionResult refinement_result = measure_scheduled_points(
        latency_buffer.get(),
        selected_buffer_bytes,
        analysis_stride_bytes,
        refinement_points,
        config.latency_chain_mode,
        timer,
        config.tlb_seed,
        TlbMeasurementPass::Refinement,
        stop_requested,
        measurements);
    measurement_records.insert(measurement_records.end(),
                               refinement_result.records.begin(),
                               refinement_result.records.end());
    if (refinement_result.status == TlbScheduleExecutionStatus::Error) {
      if (buffer_locked) {
        (void)munlock(latency_buffer.get(), selected_buffer_bytes);
      }
      return EXIT_FAILURE;
    }
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
      final_localities_bytes.size() == planned_points;

  private_cache_knee = detect_private_cache_knee(final_localities_bytes, p50_latency_ns, &sweep_loop_latencies_ns);

  l1_boundary = detect_l1_tlb_boundary(final_localities_bytes,
                                       p50_latency_ns,
                                       tlb_guard_bytes,
                                       private_cache_knee,
                                       sweep_loop_latencies_ns);

  l2_boundary = TlbBoundaryDetection{};
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
  bool page_walk_comparison_completed = false;
  if (can_measure_page_walk_penalty && main_sweep_complete) {
    TlbSweepPoint comparison_point;
    comparison_point.point_index = planned_points;
    comparison_point.requested_pages = kPageWalkComparisonLocalityBytes / page_size_bytes;
    comparison_point.effective_pages = comparison_point.requested_pages;
    comparison_point.locality_bytes = kPageWalkComparisonLocalityBytes;
    comparison_point.stride_bytes = analysis_stride_bytes;
    comparison_point.pointer_count = kPageWalkComparisonLocalityBytes / analysis_stride_bytes;
    comparison_point.refinement_source = "large-locality";
    std::vector<LocalityMeasurement> comparison_measurements;
    const TlbScheduleExecutionResult comparison_result = measure_scheduled_points(
        latency_buffer.get(),
        selected_buffer_bytes,
        analysis_stride_bytes,
        {comparison_point},
        config.latency_chain_mode,
        timer,
        config.tlb_seed,
        TlbMeasurementPass::LargeLocality,
        stop_requested,
        comparison_measurements);
    measurement_records.insert(measurement_records.end(),
                               comparison_result.records.begin(),
                               comparison_result.records.end());
    if (comparison_result.status == TlbScheduleExecutionStatus::Interrupted) {
      interrupted = true;
      report_interrupt_once();
    } else if (comparison_result.status == TlbScheduleExecutionStatus::Error) {
      if (buffer_locked) {
        (void)munlock(latency_buffer.get(), selected_buffer_bytes);
      }
      return EXIT_FAILURE;
    } else if (!comparison_measurements.empty()) {
      page_walk_512mb_p50_ns = comparison_measurements.front().p50_latency_ns;
      page_walk_512mb_loop_latencies_ns =
          comparison_measurements.front().loop_latencies_ns;
      std::cout << Messages::msg_tlb_analysis_page_walk_progress(
                       kPageWalkComparisonLocalityBytes / Constants::BYTES_PER_MB)
                << " — " << page_walk_512mb_p50_ns << " ns" << std::endl;
      page_walk_comparison_completed = true;
    }
  }

  if (!interrupted && stop_requested && stop_requested()) {
    interrupted = true;
    report_interrupt_once();
  }

  const bool conclusions_valid = main_sweep_complete && !interrupted;
  const std::string analysis_status = conclusions_valid
                                          ? "complete"
                                          : (interrupted ? "interrupted" : "partial");
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

  const std::pair<size_t, size_t> l1_entry_range = l1_boundary.detected
                                                       ? infer_tlb_entries_range(final_localities_bytes,
                                                                                 l1_boundary.boundary_index,
                                                                                 page_size_bytes)
                                                       : std::make_pair(static_cast<size_t>(0), static_cast<size_t>(0));
  const std::pair<size_t, size_t> l2_entry_range = l2_boundary.detected
                                                       ? infer_tlb_entries_range(final_localities_bytes,
                                                                                 l2_boundary.boundary_index,
                                                                                 page_size_bytes)
                                                       : std::make_pair(static_cast<size_t>(0), static_cast<size_t>(0));
  const size_t l1_entries = l1_boundary.detected
                                ? infer_tlb_entries_estimate(final_localities_bytes,
                                                             l1_boundary.boundary_index,
                                                             page_size_bytes)
                                : 0;
  const size_t l2_entries = l2_boundary.detected
                                 ? infer_tlb_entries_estimate(final_localities_bytes,
                                                              l2_boundary.boundary_index,
                                                              page_size_bytes)
                                 : 0;

  const double page_walk_baseline_ns = p50_latency_ns.empty() ? 0.0 : p50_latency_ns.front();
  const double page_walk_penalty_ns =
      page_walk_comparison_completed ? (page_walk_512mb_p50_ns - page_walk_baseline_ns) : 0.0;

  std::cout << std::endl;
  std::cout << Messages::report_tlb_header() << std::endl;
  std::cout << Messages::report_tlb_cpu(cpu_name) << std::endl;
  std::cout << Messages::report_tlb_page_size(page_size_bytes) << std::endl;
  std::cout << Messages::report_tlb_buffer(selected_buffer_mb, buffer_locked) << std::endl;
  std::cout << Messages::report_tlb_stride(analysis_stride_bytes) << std::endl;
  std::cout << Messages::report_tlb_density(tlb_sweep_density_to_string(sweep_density)) << std::endl;
  std::cout << Messages::report_tlb_seed(config.tlb_seed,
                                         config.user_specified_tlb_seed)
            << std::endl;
  std::cout << Messages::report_tlb_schedule_policy() << std::endl;
  std::cout << Messages::report_tlb_chain_mode(
                   latency_chain_mode_to_string(effective_chain_mode))
            << std::endl;
  std::cout << Messages::report_tlb_loop_config(kLoopsPerPoint, kAccessesPerLoop) << std::endl;
  if (!final_localities_bytes.empty()) {
    std::cout << Messages::report_tlb_sweep_range(final_localities_bytes.front(),
                                                  final_localities_bytes.back(),
                                                  final_localities_bytes.size())
              << std::endl;
  }
  std::cout << Messages::report_tlb_fine_sweep(fine_sweep_added_points,
                                               final_localities_bytes.size())
            << std::endl;
  std::cout << Messages::report_tlb_analysis_status(analysis_status,
                                                    planned_points,
                                                    final_localities_bytes.size(),
                                                    conclusions_valid)
            << std::endl;

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
    std::cout << Messages::report_tlb_inferred_size_entries(l1_entries) << std::endl;
    std::cout << Messages::report_tlb_inferred_entries_range(
                     l1_entry_range.first,
                     l1_entry_range.second)
              << std::endl;
    if (l1_boundary.overlaps_private_cache_knee) {
      std::cout << Messages::report_tlb_private_cache_overlap() << std::endl;
    }
    std::cout << Messages::report_tlb_confidence(l1_boundary.confidence,
                                                 l1_boundary.step_ns,
                                                 l1_boundary.step_percent)
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
    std::cout << Messages::report_tlb_inferred_reach_entries(l2_entries) << std::endl;
    std::cout << Messages::report_tlb_inferred_entries_range(
                     l2_entry_range.first,
                     l2_entry_range.second)
              << std::endl;
    std::cout << Messages::report_tlb_confidence(l2_boundary.confidence,
                                                 l2_boundary.step_ns,
                                                 l2_boundary.step_percent)
              << std::endl;
  } else {
    std::cout << Messages::report_tlb_not_detected() << std::endl;
  }
  if (!conclusions_valid || (can_measure_page_walk_penalty && !page_walk_comparison_completed)) {
    std::cout << Messages::report_tlb_page_walk_penalty_interrupted(
                     page_walk_baseline_locality_bytes / Constants::BYTES_PER_KB,
                     kPageWalkComparisonLocalityBytes / Constants::BYTES_PER_MB)
              << std::endl;
  } else if (page_walk_comparison_completed) {
    std::cout << Messages::report_tlb_page_walk_penalty(
                     page_walk_penalty_ns,
                     page_walk_baseline_locality_bytes / Constants::BYTES_PER_KB,
                     kPageWalkComparisonLocalityBytes / Constants::BYTES_PER_MB)
              << std::endl;
  } else {
    std::cout << Messages::report_tlb_page_walk_penalty_unavailable(
                     page_walk_baseline_locality_bytes / Constants::BYTES_PER_KB,
                     kPageWalkComparisonLocalityBytes / Constants::BYTES_PER_MB,
                     kPageWalkMinimumBufferMb,
                     selected_buffer_mb)
              << std::endl;
  }

  const auto analysis_end = std::chrono::steady_clock::now();
  const double total_execution_time_sec =
      std::chrono::duration<double>(analysis_end - analysis_start).count();
  const TlbAnalysisJsonContext json_context = {
      config,
      cpu_name,
      config.perf_cores,
      config.eff_cores,
      page_size_bytes,
      l1_cache_size_bytes,
      tlb_guard_bytes,
      analysis_stride_bytes,
      kLoopsPerPoint,
      kAccessesPerLoop,
      page_walk_baseline_locality_bytes,
      kPageWalkComparisonLocalityBytes,
      selected_buffer_mb,
      buffer_locked,
      analysis_status,
      planned_points,
      final_localities_bytes.size(),
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
  };

  if (save_tlb_analysis_to_json(json_context) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

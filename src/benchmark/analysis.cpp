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
 * @file analysis.cpp
 * @brief Standalone TLB analysis mode implementation
 */

#include "benchmark/analysis.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

#include "benchmark/benchmark_tests.h"
#include "core/config/config.h"
#include "core/config/constants.h"
#include "core/config/version.h"
#include "core/memory/memory_manager.h"
#include "core/memory/memory_utils.h"
#include "core/system/system_info.h"
#include "core/timing/timer.h"
#include "output/console/messages.h"
#include "output/json/json_output.h"
#include "warmup/warmup.h"

namespace {

// Fixed analysis parameters for standalone TLB mode.
constexpr size_t kLatencyStrideBytes = 64;
constexpr size_t kLoopsPerPoint = 30;
constexpr size_t kAccessesPerLoop = 25 * 1000 * 1000;
constexpr size_t kPageWalkBaselineLocalityBytes = 16 * Constants::BYTES_PER_KB;
constexpr size_t kPageWalkComparisonLocalityBytes = 512 * Constants::BYTES_PER_MB;
constexpr size_t kPageWalkMinimumBufferMb = 512;
constexpr double kRelativeThreshold = 0.10;
constexpr double kAbsoluteThresholdNs = 2.0;

const std::array<size_t, 15> kLocalitySweepBytes = {
    16 * Constants::BYTES_PER_KB,        64 * Constants::BYTES_PER_KB,
    128 * Constants::BYTES_PER_KB,       256 * Constants::BYTES_PER_KB,
    512 * Constants::BYTES_PER_KB,       1 * Constants::BYTES_PER_MB,
    2 * Constants::BYTES_PER_MB,         4 * Constants::BYTES_PER_MB,
    8 * Constants::BYTES_PER_MB,         12 * Constants::BYTES_PER_MB,
    16 * Constants::BYTES_PER_MB,        32 * Constants::BYTES_PER_MB,
    64 * Constants::BYTES_PER_MB,        128 * Constants::BYTES_PER_MB,
    256 * Constants::BYTES_PER_MB,
};

const std::array<size_t, 3> kBufferCandidateMb = {1024, 512, 256};

/**
 * @brief Compute arithmetic mean for a half-open range.
 *
 * Calculates the average of `values[start, end)`. Returns `0.0` when the
 * range is invalid or empty.
 *
 * @param[in] values Source value vector
 * @param[in] start  Inclusive start index
 * @param[in] end    Exclusive end index
 *
 * @return Mean value for the requested range, or `0.0` on invalid input.
 */
double average_range(const std::vector<double>& values, size_t start, size_t end) {
  if (start >= end || end > values.size()) {
    return 0.0;
  }

  const double sum = std::accumulate(values.begin() + static_cast<std::ptrdiff_t>(start),
                                     values.begin() + static_cast<std::ptrdiff_t>(end), 0.0);
  const double count = static_cast<double>(end - start);
  return (count > 0.0) ? (sum / count) : 0.0;
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
 * @brief Measure one locality point and return P50 latency.
 *
 * Rebuilds the pointer chain and runs the latency loop `kLoopsPerPoint`
 * times, then computes median latency per access for the locality window.
 *
 * @param[in]     latency_buffer      Latency buffer base pointer
 * @param[in]     buffer_size_bytes   Buffer size in bytes
 * @param[in]     locality_bytes      Locality window size in bytes
 * @param[in,out] timer               High-resolution timer instance
 * @param[out]    out_p50_latency_ns  Measured P50 latency in ns/access
 * @param[out]    out_loop_latencies_ns Optional raw loop latencies (ns/access)
 *
 * @return true on success, false on setup/measurement failure.
 */
bool measure_locality_p50(void* latency_buffer,
                          size_t buffer_size_bytes,
                          size_t locality_bytes,
                          HighResTimer& timer,
                          double& out_p50_latency_ns,
                          std::vector<double>* out_loop_latencies_ns = nullptr) {
  const size_t locality_kb = locality_bytes / Constants::BYTES_PER_KB;

  std::vector<double> loop_latencies_ns;
  loop_latencies_ns.reserve(kLoopsPerPoint);

  for (size_t loop = 0; loop < kLoopsPerPoint; ++loop) {
    if (setup_latency_chain(latency_buffer, buffer_size_bytes,
                            kLatencyStrideBytes, locality_bytes) != EXIT_SUCCESS) {
      return false;
    }

    warmup_latency(latency_buffer, buffer_size_bytes);

    const double total_latency_ns =
        run_latency_test(latency_buffer, kAccessesPerLoop, timer, nullptr, 0);
    if (total_latency_ns <= 0.0 || std::isnan(total_latency_ns) || std::isinf(total_latency_ns)) {
      std::cerr << Messages::error_prefix()
                << Messages::error_tlb_analysis_invalid_measurement(locality_kb,
                                                                     static_cast<int>(loop + 1))
                << std::endl;
      return false;
    }

    const double latency_ns_per_access = total_latency_ns / static_cast<double>(kAccessesPerLoop);
    loop_latencies_ns.push_back(latency_ns_per_access);
  }

  out_p50_latency_ns = median(loop_latencies_ns);
  if (out_loop_latencies_ns != nullptr) {
    *out_loop_latencies_ns = loop_latencies_ns;
  }
  return true;
}

nlohmann::ordered_json build_tlb_boundary_json(const TlbBoundaryDetection& boundary,
                                               size_t inferred_entries) {
  nlohmann::ordered_json boundary_json;
  boundary_json["detected"] = boundary.detected;
  if (!boundary.detected) {
    return boundary_json;
  }

  boundary_json["segment_start_index"] = boundary.segment_start_index;
  boundary_json["boundary_index"] = boundary.boundary_index;
  boundary_json["boundary_locality_bytes"] = boundary.boundary_locality_bytes;
  boundary_json["boundary_locality_kb"] = boundary.boundary_locality_bytes / Constants::BYTES_PER_KB;
  boundary_json["baseline_ns"] = boundary.baseline_ns;
  boundary_json["boundary_latency_ns"] = boundary.boundary_latency_ns;
  boundary_json["step_ns"] = boundary.step_ns;
  boundary_json["step_percent"] = boundary.step_percent;
  boundary_json["persistent_jump"] = boundary.persistent_jump;
  boundary_json["confidence"] = boundary.confidence;
  boundary_json["inferred_entries"] = inferred_entries;
  return boundary_json;
}

std::string build_utc_timestamp() {
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);
  std::tm utc_time;
  gmtime_r(&time_t, &utc_time);
  std::ostringstream timestamp_str;
  timestamp_str << std::put_time(&utc_time, "%Y-%m-%dT%H:%M:%SZ");
  return timestamp_str.str();
}

int save_tlb_analysis_to_json(
    const BenchmarkConfig& config,
    const std::string& cpu_name,
    size_t page_size_bytes,
    size_t l1_cache_size_bytes,
    size_t tlb_guard_bytes,
    size_t selected_buffer_mb,
    bool buffer_locked,
    const std::vector<size_t>& localities_bytes,
    const std::vector<std::vector<double>>& sweep_loop_latencies_ns,
    const std::vector<double>& p50_latency_ns,
    const TlbBoundaryDetection& l1_boundary,
    const TlbBoundaryDetection& l2_boundary,
    size_t l1_entries,
    size_t l2_entries,
    bool can_measure_page_walk_penalty,
    const std::vector<double>& page_walk_512mb_loop_latencies_ns,
    double page_walk_512mb_p50_ns,
    double page_walk_baseline_ns,
    double page_walk_penalty_ns,
    double total_execution_time_sec) {
  if (config.output_file.empty()) {
    return EXIT_SUCCESS;
  }

  nlohmann::ordered_json json_output;
  json_output[JsonKeys::CONFIGURATION] = {
      {"mode", "analyze_tlb"},
      {JsonKeys::CPU_NAME, cpu_name},
      {JsonKeys::PAGE_SIZE_BYTES, page_size_bytes},
      {JsonKeys::L1_CACHE_SIZE_BYTES, l1_cache_size_bytes},
      {JsonKeys::LATENCY_STRIDE_BYTES, kLatencyStrideBytes},
      {JsonKeys::LATENCY_SAMPLE_COUNT, static_cast<int>(kLoopsPerPoint)},
      {"accesses_per_loop", kAccessesPerLoop},
      {"tlb_guard_bytes", tlb_guard_bytes},
      {"buffer_size_mb", selected_buffer_mb},
      {"buffer_locked", buffer_locked}};
  json_output[JsonKeys::EXECUTION_TIME_SEC] = total_execution_time_sec;

  nlohmann::ordered_json sweep_json = nlohmann::ordered_json::array();
  for (size_t i = 0; i < localities_bytes.size(); ++i) {
    nlohmann::ordered_json point;
    point["locality_bytes"] = localities_bytes[i];
    point["locality_kb"] = localities_bytes[i] / Constants::BYTES_PER_KB;
    point["loop_latencies_ns"] = sweep_loop_latencies_ns[i];
    point["p50_latency_ns"] = p50_latency_ns[i];
    sweep_json.push_back(point);
  }

  nlohmann::ordered_json tlb_json;
  tlb_json["sweep"] = sweep_json;
  tlb_json["l1_tlb_detection"] = build_tlb_boundary_json(l1_boundary, l1_entries);
  tlb_json["l2_tlb_detection"] = build_tlb_boundary_json(l2_boundary, l2_entries);
  tlb_json["page_walk_penalty"] = {
      {"available", can_measure_page_walk_penalty},
      {"baseline_locality_kb", kPageWalkBaselineLocalityBytes / Constants::BYTES_PER_KB},
      {"comparison_locality_mb", kPageWalkComparisonLocalityBytes / Constants::BYTES_PER_MB},
      {"baseline_p50_ns", page_walk_baseline_ns}};
  if (can_measure_page_walk_penalty) {
    tlb_json["page_walk_penalty"]["comparison_loop_latencies_ns"] = page_walk_512mb_loop_latencies_ns;
    tlb_json["page_walk_penalty"]["comparison_p50_ns"] = page_walk_512mb_p50_ns;
    tlb_json["page_walk_penalty"]["penalty_ns"] = page_walk_penalty_ns;
  } else {
    tlb_json["page_walk_penalty"]["reason"] =
        "requires at least 512 MB analysis buffer";
  }
  json_output["tlb_analysis"] = tlb_json;
  json_output[JsonKeys::TIMESTAMP] = build_utc_timestamp();
  json_output[JsonKeys::VERSION] = SOFTVERSION;

  std::filesystem::path file_path(config.output_file);
  if (file_path.is_relative()) {
    file_path = std::filesystem::current_path() / file_path;
  }
  return write_json_to_file(file_path, json_output);
}

}  // namespace

/**
 * @brief Classify boundary confidence level.
 *
 * Confidence is based on step magnitude and whether the jump persists at the
 * next locality point.
 *
 * @param[in] step_ns         Absolute boundary step in nanoseconds
 * @param[in] step_percent    Relative boundary step (0.10 = 10%)
 * @param[in] persistent_jump True when next point remains above threshold
 *
 * @return "High", "Medium", or "Low" confidence label.
 */
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

/**
 * @brief Infer TLB entry count from locality boundary and page size.
 *
 * @param[in] locality_bytes  Boundary locality window in bytes
 * @param[in] page_size_bytes System page size in bytes
 *
 * @return Inferred entry count, or 0 if page size is zero.
 */
size_t infer_tlb_entries(size_t locality_bytes, size_t page_size_bytes) {
  if (page_size_bytes == 0) {
    return 0;
  }
  return locality_bytes / page_size_bytes;
}

/**
 * @brief Detect first latency boundary in a locality segment.
 *
 * Scans from `segment_start_index + 1` onward and marks the first point where
 * latency rises by at least `max(2ns, 10% of baseline)` against a running
 * average baseline built from the current segment.
 *
 * @param[in] locality_bytes       Locality windows in measurement order
 * @param[in] p50_latency_ns       P50 latency values for each locality point
 * @param[in] segment_start_index  Start index of the active baseline segment
 * @param[in] min_locality_bytes   Minimum locality accepted as TLB boundary
 *
 * @return Boundary detection result including index, step, and confidence.
 */
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

/**
 * @brief Run standalone TLB analysis benchmark mode.
 *
 * Workflow:
 * - Print program header and run banner.
 * - Allocate analysis buffer with fallback order: 1024MB -> 512MB -> 256MB.
 * - Sweep locality windows from 16KB to 256MB and collect P50 latencies.
 * - Optionally run a separate 512MB locality pass for page-walk penalty.
 * - Infer L1/L2 boundaries and emit the TLB analysis report.
 *
 * Page-walk penalty:
 * - Reported as P50(512MB) - P50(16KB).
 * - Only computed when selected analysis buffer is at least 512MB.
 *
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on allocation or measurement error.
 */
int run_tlb_analysis(const BenchmarkConfig& config) {
  const auto analysis_start = std::chrono::steady_clock::now();
  std::cout << Messages::usage_header(SOFTVERSION);
  std::cout << Messages::msg_running_tlb_analysis() << std::endl;

  const size_t page_size_bytes = static_cast<size_t>(getpagesize());
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

  std::vector<size_t> localities_bytes(kLocalitySweepBytes.begin(), kLocalitySweepBytes.end());
  std::vector<double> p50_latency_ns;
  std::vector<std::vector<double>> sweep_loop_latencies_ns;
  p50_latency_ns.reserve(localities_bytes.size());
  sweep_loop_latencies_ns.reserve(localities_bytes.size());

  for (size_t locality_index = 0; locality_index < localities_bytes.size(); ++locality_index) {
    const size_t locality_bytes = localities_bytes[locality_index];
    const size_t locality_kb = locality_bytes / Constants::BYTES_PER_KB;

    std::cout << Messages::msg_tlb_analysis_locality_progress(locality_index + 1,
                                                              localities_bytes.size(),
                                                              locality_kb)
              << std::endl;

    double locality_p50_ns = 0.0;
    std::vector<double> loop_latencies_ns;
    if (!measure_locality_p50(latency_buffer.get(),
                              selected_buffer_bytes,
                              locality_bytes,
                              timer,
                              locality_p50_ns,
                              &loop_latencies_ns)) {
      if (buffer_locked) {
        (void)munlock(latency_buffer.get(), selected_buffer_bytes);
      }
      return EXIT_FAILURE;
    }

    p50_latency_ns.push_back(locality_p50_ns);
    sweep_loop_latencies_ns.push_back(std::move(loop_latencies_ns));
  }

  const bool can_measure_page_walk_penalty = selected_buffer_mb >= kPageWalkMinimumBufferMb;
  std::vector<double> page_walk_512mb_loop_latencies_ns;
  double page_walk_512mb_p50_ns = 0.0;
  if (can_measure_page_walk_penalty) {
    std::cout << Messages::msg_tlb_analysis_page_walk_progress(
                     kPageWalkComparisonLocalityBytes / Constants::BYTES_PER_MB)
              << std::endl;
    if (!measure_locality_p50(latency_buffer.get(),
                              selected_buffer_bytes,
                              kPageWalkComparisonLocalityBytes,
                              timer,
                              page_walk_512mb_p50_ns,
                              &page_walk_512mb_loop_latencies_ns)) {
      if (buffer_locked) {
        (void)munlock(latency_buffer.get(), selected_buffer_bytes);
      }
      return EXIT_FAILURE;
    }
  }

  if (buffer_locked) {
    (void)munlock(latency_buffer.get(), selected_buffer_bytes);
  }

  const size_t tlb_guard_bytes = std::max<size_t>(2 * l1_cache_size_bytes, 64 * page_size_bytes);

  const TlbBoundaryDetection l1_boundary =
      detect_tlb_boundary(localities_bytes, p50_latency_ns, 0, tlb_guard_bytes);

  TlbBoundaryDetection l2_boundary;
  if (l1_boundary.detected && l1_boundary.boundary_index < localities_bytes.size() - 1) {
    l2_boundary = detect_tlb_boundary(localities_bytes, p50_latency_ns,
                                      l1_boundary.boundary_index, tlb_guard_bytes);
  }

  const size_t l1_entries = l1_boundary.detected
                                ? infer_tlb_entries(l1_boundary.boundary_locality_bytes, page_size_bytes)
                                : 0;
  const size_t l2_entries = l2_boundary.detected
                                ? infer_tlb_entries(l2_boundary.boundary_locality_bytes, page_size_bytes)
                                : 0;

  const size_t page_walk_baseline_index = 0;
  const double page_walk_baseline_ns = (p50_latency_ns.size() > page_walk_baseline_index)
                                            ? p50_latency_ns[page_walk_baseline_index]
                                            : 0.0;
  const double page_walk_penalty_ns = page_walk_512mb_p50_ns - page_walk_baseline_ns;

  std::cout << std::endl;
  std::cout << Messages::report_tlb_header() << std::endl;
  std::cout << Messages::report_tlb_cpu(cpu_name) << std::endl;
  std::cout << Messages::report_tlb_page_size(page_size_bytes) << std::endl;
  std::cout << Messages::report_tlb_buffer(selected_buffer_mb, buffer_locked) << std::endl;
  std::cout << Messages::report_tlb_stride(kLatencyStrideBytes) << std::endl;
  std::cout << Messages::report_tlb_loop_config(kLoopsPerPoint, kAccessesPerLoop) << std::endl;

  std::cout << std::endl;
  std::cout << Messages::report_tlb_l1_section() << std::endl;
  if (l1_boundary.detected) {
    std::cout << Messages::report_tlb_boundary_kb(
                     l1_boundary.boundary_locality_bytes / Constants::BYTES_PER_KB)
              << std::endl;
    std::cout << Messages::report_tlb_inferred_size_entries(l1_entries) << std::endl;
    std::cout << Messages::report_tlb_confidence(l1_boundary.confidence,
                                                 l1_boundary.step_ns,
                                                 l1_boundary.step_percent)
              << std::endl;
  } else {
    std::cout << Messages::report_tlb_not_detected() << std::endl;
  }

  std::cout << std::endl;
  std::cout << Messages::report_tlb_l2_section() << std::endl;
  if (l2_boundary.detected) {
    std::cout << Messages::report_tlb_boundary_kb(
                     l2_boundary.boundary_locality_bytes / Constants::BYTES_PER_KB)
              << std::endl;
    std::cout << Messages::report_tlb_inferred_reach_entries(l2_entries) << std::endl;
    std::cout << Messages::report_tlb_confidence(l2_boundary.confidence,
                                                 l2_boundary.step_ns,
                                                 l2_boundary.step_percent)
              << std::endl;
    if (can_measure_page_walk_penalty) {
      std::cout << Messages::report_tlb_page_walk_penalty(
                       page_walk_penalty_ns,
                       kPageWalkBaselineLocalityBytes / Constants::BYTES_PER_KB,
                       kPageWalkComparisonLocalityBytes / Constants::BYTES_PER_MB)
                << std::endl;
    } else {
      std::cout << Messages::report_tlb_page_walk_penalty_unavailable(
                       kPageWalkMinimumBufferMb,
                       selected_buffer_mb)
                << std::endl;
    }
  } else {
    std::cout << Messages::report_tlb_not_detected() << std::endl;
    if (can_measure_page_walk_penalty) {
      std::cout << Messages::report_tlb_page_walk_penalty(
                       page_walk_penalty_ns,
                       kPageWalkBaselineLocalityBytes / Constants::BYTES_PER_KB,
                       kPageWalkComparisonLocalityBytes / Constants::BYTES_PER_MB)
                << std::endl;
    } else {
      std::cout << Messages::report_tlb_page_walk_penalty_unavailable(
                       kPageWalkMinimumBufferMb,
                       selected_buffer_mb)
                << std::endl;
    }
  }

  const auto analysis_end = std::chrono::steady_clock::now();
  const double total_execution_time_sec =
      std::chrono::duration<double>(analysis_end - analysis_start).count();
  if (save_tlb_analysis_to_json(config,
                                cpu_name,
                                page_size_bytes,
                                l1_cache_size_bytes,
                                tlb_guard_bytes,
                                selected_buffer_mb,
                                buffer_locked,
                                localities_bytes,
                                sweep_loop_latencies_ns,
                                p50_latency_ns,
                                l1_boundary,
                                l2_boundary,
                                l1_entries,
                                l2_entries,
                                can_measure_page_walk_penalty,
                                page_walk_512mb_loop_latencies_ns,
                                page_walk_512mb_p50_ns,
                                page_walk_baseline_ns,
                                page_walk_penalty_ns,
                                total_execution_time_sec) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

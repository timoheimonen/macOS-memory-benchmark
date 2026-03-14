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
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

#include "benchmark/benchmark_tests.h"
#include "benchmark/tlb_analysis_json.h"
#include "core/config/config.h"
#include "core/config/constants.h"
#include "core/config/version.h"
#include "core/memory/memory_manager.h"
#include "core/memory/memory_utils.h"
#include "core/system/system_info.h"
#include "core/timing/timer.h"
#include "output/console/messages.h"
#include "warmup/warmup.h"

namespace {

// Fixed analysis parameters for standalone TLB mode.
constexpr size_t kLoopsPerPoint = 30;
constexpr size_t kAccessesPerLoop = 25 * 1000 * 1000;
constexpr size_t kSweepMinLocalityBytes = 16 * Constants::BYTES_PER_KB;
constexpr size_t kSweepMaxLocalityBytes = 256 * Constants::BYTES_PER_MB;
constexpr size_t kPageWalkComparisonLocalityBytes = 512 * Constants::BYTES_PER_MB;
constexpr size_t kPageWalkMinimumBufferMb = 512;

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

size_t calculate_min_sweep_locality(size_t stride_bytes) {
  if (stride_bytes > (std::numeric_limits<size_t>::max() / 2)) {
    return std::numeric_limits<size_t>::max();
  }

  return std::max(kSweepMinLocalityBytes, 2 * stride_bytes);
}

std::vector<size_t> build_effective_locality_sweep(size_t stride_bytes) {
  const size_t min_locality_bytes = calculate_min_sweep_locality(stride_bytes);
  std::vector<size_t> localities;

  if (min_locality_bytes > kSweepMaxLocalityBytes) {
    return localities;
  }

  localities.reserve(kLocalitySweepBytes.size() + 1);
  if (std::none_of(kLocalitySweepBytes.begin(),
                   kLocalitySweepBytes.end(),
                   [min_locality_bytes](size_t value) {
                     return value == min_locality_bytes;
                   })) {
    localities.push_back(min_locality_bytes);
  }

  for (size_t locality_bytes : kLocalitySweepBytes) {
    if (locality_bytes < min_locality_bytes || locality_bytes > kSweepMaxLocalityBytes) {
      continue;
    }
    localities.push_back(locality_bytes);
  }

  std::sort(localities.begin(), localities.end());
  localities.erase(std::unique(localities.begin(), localities.end()), localities.end());
  return localities;
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
 * @param[in]     stride_bytes        Pointer-chase stride in bytes
 * @param[in]     locality_bytes      Locality window size in bytes
 * @param[in]     chain_mode          Pointer-chain construction mode
 * @param[in,out] timer               High-resolution timer instance
 * @param[out]    out_p50_latency_ns  Measured P50 latency in ns/access
 * @param[out]    out_loop_latencies_ns Optional raw loop latencies (ns/access)
 *
 * @return true on success, false on setup/measurement failure.
 */
bool measure_locality_p50(void* latency_buffer,
                          size_t buffer_size_bytes,
                          size_t stride_bytes,
                          size_t locality_bytes,
                          LatencyChainMode chain_mode,
                          HighResTimer& timer,
                          double& out_p50_latency_ns,
                          std::vector<double>* out_loop_latencies_ns = nullptr) {
  const size_t locality_kb = locality_bytes / Constants::BYTES_PER_KB;

  std::vector<double> loop_latencies_ns;
  loop_latencies_ns.reserve(kLoopsPerPoint);

  for (size_t loop = 0; loop < kLoopsPerPoint; ++loop) {
    if (setup_latency_chain(latency_buffer, buffer_size_bytes,
                            stride_bytes, locality_bytes, nullptr, chain_mode) != EXIT_SUCCESS) {
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

}  // namespace

/**
 * @brief Run standalone TLB analysis benchmark mode.
 *
 * Workflow:
 * - Print program header and run banner.
 * - Allocate analysis buffer with fallback order: 1024MB -> 512MB -> 256MB.
 * - Use configured latency stride (default from `-latency-stride-bytes`).
 * - Sweep locality windows from max(16KB, 2*stride) to 256MB and collect P50 latencies.
 * - Optionally run a separate 512MB locality pass for page-walk penalty.
 * - Infer L1/L2 boundaries and emit the TLB analysis report.
 *
 * Page-walk penalty:
 * - Reported as P50(512MB) - P50(effective baseline locality).
 * - Only computed when selected analysis buffer is at least 512MB.
 *
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on allocation or measurement error.
 */
int run_tlb_analysis(const BenchmarkConfig& config) {
  const auto analysis_start = std::chrono::steady_clock::now();
  std::cout << Messages::usage_header(SOFTVERSION);
  std::cout << Messages::msg_running_tlb_analysis() << std::endl;

  if (config.latency_stride_bytes == 0) {
    std::cerr << Messages::error_prefix()
              << Messages::error_latency_stride_invalid(0, 1, std::numeric_limits<long long>::max())
              << std::endl;
    return EXIT_FAILURE;
  }
  if ((config.latency_stride_bytes % sizeof(void*)) != 0) {
    std::cerr << Messages::error_prefix()
              << Messages::error_latency_stride_alignment(config.latency_stride_bytes, sizeof(void*))
              << std::endl;
    return EXIT_FAILURE;
  }
  const size_t analysis_stride_bytes = config.latency_stride_bytes;

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

  const size_t pointer_count = selected_buffer_bytes / analysis_stride_bytes;
  if (pointer_count < 2) {
    std::cerr << Messages::error_prefix()
              << Messages::error_buffer_stride_invalid_latency_chain(
                     pointer_count, selected_buffer_bytes, analysis_stride_bytes)
              << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<size_t> localities_bytes = build_effective_locality_sweep(analysis_stride_bytes);
  if (localities_bytes.empty()) {
    std::cerr << Messages::error_prefix()
              << Messages::error_buffer_stride_invalid_latency_chain(
                     pointer_count, selected_buffer_bytes, analysis_stride_bytes)
              << std::endl;
    return EXIT_FAILURE;
  }

  const size_t page_walk_baseline_locality_bytes = localities_bytes.front();
  const LatencyChainMode effective_chain_mode =
      resolve_latency_chain_mode(config.latency_chain_mode, page_walk_baseline_locality_bytes);

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
                              analysis_stride_bytes,
                              locality_bytes,
                              config.latency_chain_mode,
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
                              analysis_stride_bytes,
                              kPageWalkComparisonLocalityBytes,
                              config.latency_chain_mode,
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

  const double page_walk_baseline_ns = p50_latency_ns.empty() ? 0.0 : p50_latency_ns.front();
  const double page_walk_penalty_ns = page_walk_512mb_p50_ns - page_walk_baseline_ns;

  std::cout << std::endl;
  std::cout << Messages::report_tlb_header() << std::endl;
  std::cout << Messages::report_tlb_cpu(cpu_name) << std::endl;
  std::cout << Messages::report_tlb_page_size(page_size_bytes) << std::endl;
  std::cout << Messages::report_tlb_buffer(selected_buffer_mb, buffer_locked) << std::endl;
  std::cout << Messages::report_tlb_stride(analysis_stride_bytes) << std::endl;
  std::cout << Messages::report_tlb_chain_mode(
                   latency_chain_mode_to_string(effective_chain_mode))
            << std::endl;
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
  } else {
    std::cout << Messages::report_tlb_not_detected() << std::endl;
    if (can_measure_page_walk_penalty) {
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
      total_execution_time_sec,
  };

  if (save_tlb_analysis_to_json(json_context) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

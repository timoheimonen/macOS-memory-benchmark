// Copyright 2025 Timo Heimonen <timo.heimonen@proton.me>
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
 * @file results_messages.cpp
 * @brief Results output messages
 *
 * Provides implementations for benchmark results message generation functions.
 * Includes formatted messages for displaying bandwidth and latency test results,
 * cache performance metrics, and buffer size information.
 */

#include "messages_api.h"
#include "core/config/constants.h"  // For default values
#include <sstream>
#include <iomanip>
#include <cmath>        // For std::isinf, std::isnan

namespace Messages {

// --- Results Output Messages ---
std::string results_loop_header(int loop) {
  std::ostringstream oss;
  oss << "\n--- Results (Loop " << (loop + 1) << ") ---";
  return oss.str();
}

std::string results_main_memory_bandwidth(int num_threads) {
  std::ostringstream oss;
  oss << "Main Memory Bandwidth Tests (multi-threaded, " << num_threads << " threads):";
  return oss.str();
}

std::string results_read_bandwidth(double bw_gb_s, double total_time) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(Constants::BANDWIDTH_PRECISION);
  oss << "  Read : " << bw_gb_s << " GB/s (Total time: ";
  oss << std::setprecision(Constants::TIME_PRECISION) << total_time << " s)";
  return oss.str();
}

std::string results_write_bandwidth(double bw_gb_s, double total_time) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(Constants::BANDWIDTH_PRECISION);
  oss << "  Write: " << bw_gb_s << " GB/s (Total time: ";
  oss << std::setprecision(Constants::TIME_PRECISION) << total_time << " s)";
  return oss.str();
}

std::string results_copy_bandwidth(double bw_gb_s, double total_time) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(Constants::BANDWIDTH_PRECISION);
  oss << "  Copy : " << bw_gb_s << " GB/s (Total time: ";
  oss << std::setprecision(Constants::TIME_PRECISION) << total_time << " s)";
  return oss.str();
}

std::string results_main_memory_latency() {
  return "\nMain Memory Latency Test (single-threaded, pointer chase):";
}

std::string results_latency_total_time(double total_time_sec) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(Constants::TIME_PRECISION);
  oss << "  Total time: " << total_time_sec << " s";
  return oss.str();
}

namespace {

std::string format_locality_label(size_t locality_bytes) {
  std::ostringstream locality;

  if (locality_bytes == 0) {
    return "global random locality";
  }

  if (locality_bytes < 1024) {
    locality << locality_bytes << " B locality";
    return locality.str();
  }

  locality << std::fixed << std::setprecision(2);
  if (locality_bytes < 1024 * 1024) {
    locality << (locality_bytes / 1024.0) << " KB locality";
    return locality.str();
  }

  locality << (locality_bytes / (1024.0 * 1024.0)) << " MB locality";
  return locality.str();
}

}  // namespace

std::string results_latency_average(double latency_ns, size_t locality_bytes) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(Constants::LATENCY_PRECISION);
  oss << "  Average latency (" << format_locality_label(locality_bytes) << "): " << latency_ns << " ns";
  return oss.str();
}

std::string results_latency_tlb_hit(double latency_ns) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(Constants::LATENCY_PRECISION);
  oss << "  16 KiB locality latency: " << latency_ns << " ns";
  return oss.str();
}

std::string results_latency_tlb_miss(double latency_ns) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(Constants::LATENCY_PRECISION);
  oss << "  Global-random latency: " << latency_ns << " ns";
  return oss.str();
}

std::string results_latency_page_walk_penalty(double penalty_ns) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(Constants::LATENCY_PRECISION);
  oss << "  Locality latency delta (global - 16 KiB): " << penalty_ns << " ns";
  return oss.str();
}

std::string results_cache_bandwidth(int num_threads) {
  std::ostringstream oss;
  if (num_threads == 1) {
    oss << "\nCache Bandwidth Tests (single-threaded):";
  } else {
    oss << "\nCache Bandwidth Tests (multi-threaded, " << num_threads << " threads):";
  }
  return oss.str();
}

std::string results_cache_latency() {
  return "\nCache Latency Tests (single-threaded, pointer chase):";
}

std::string results_custom_cache() {
  return "  Custom Cache:";
}

std::string results_l1_cache() {
  return "  L1 Cache:";
}

std::string results_l2_cache() {
  return "  L2 Cache:";
}

std::string results_cache_read_bandwidth(double bw_gb_s) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(Constants::BANDWIDTH_PRECISION);
  oss << "    Read : " << bw_gb_s << " GB/s";
  return oss.str();
}

std::string results_cache_write_bandwidth(double bw_gb_s) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(Constants::BANDWIDTH_PRECISION);
  oss << "    Write: " << bw_gb_s << " GB/s";
  return oss.str();
}

std::string results_cache_copy_bandwidth(double bw_gb_s) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(Constants::BANDWIDTH_PRECISION);
  oss << "    Copy : " << bw_gb_s << " GB/s";
  return oss.str();
}

std::string results_buffer_size_bytes(size_t buffer_size) {
  std::ostringstream oss;
  oss << " (Buffer size: " << buffer_size << " B)";
  return oss.str();
}

std::string results_buffer_size_kb(double buffer_size_kb) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(Constants::LATENCY_PRECISION);
  oss << " (Buffer size: " << buffer_size_kb << " KB)";
  return oss.str();
}

std::string results_buffer_size_mb(double buffer_size_mb) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(Constants::LATENCY_PRECISION);
  oss << " (Buffer size: " << buffer_size_mb << " MB)";
  return oss.str();
}

std::string results_separator() {
  return "--------------";
}

std::string results_cache_latency_custom_ns(double latency_ns, size_t buffer_size) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(Constants::LATENCY_PRECISION);
  if (std::isinf(latency_ns) || std::isnan(latency_ns)) {
    oss << "  Custom Cache: N/A ns (Buffer size: " << buffer_size << " B)";
  } else {
    oss << "  Custom Cache: " << latency_ns << " ns (Buffer size: " << buffer_size << " B)";
  }
  return oss.str();
}

std::string results_cache_latency_custom_ns_kb(double latency_ns, double buffer_size_kb) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(Constants::LATENCY_PRECISION);
  if (std::isinf(latency_ns) || std::isnan(latency_ns)) {
    oss << "  Custom Cache: N/A ns (Buffer size: " << buffer_size_kb << " KB)";
  } else {
    oss << "  Custom Cache: " << latency_ns << " ns (Buffer size: " << buffer_size_kb << " KB)";
  }
  return oss.str();
}

std::string results_cache_latency_custom_ns_mb(double latency_ns, double buffer_size_mb) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(Constants::LATENCY_PRECISION);
  if (std::isinf(latency_ns) || std::isnan(latency_ns)) {
    oss << "  Custom Cache: N/A ns (Buffer size: " << buffer_size_mb << " MB)";
  } else {
    oss << "  Custom Cache: " << latency_ns << " ns (Buffer size: " << buffer_size_mb << " MB)";
  }
  return oss.str();
}

std::string results_cache_latency_l1_ns(double latency_ns, size_t buffer_size) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(Constants::LATENCY_PRECISION);
  if (std::isinf(latency_ns) || std::isnan(latency_ns)) {
    oss << "  L1 Cache: N/A ns (Buffer size: " << buffer_size << " B)";
  } else {
    oss << "  L1 Cache: " << latency_ns << " ns (Buffer size: " << buffer_size << " B)";
  }
  return oss.str();
}

std::string results_cache_latency_l1_ns_kb(double latency_ns, double buffer_size_kb) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(Constants::LATENCY_PRECISION);
  if (std::isinf(latency_ns) || std::isnan(latency_ns)) {
    oss << "  L1 Cache: N/A ns (Buffer size: " << buffer_size_kb << " KB)";
  } else {
    oss << "  L1 Cache: " << latency_ns << " ns (Buffer size: " << buffer_size_kb << " KB)";
  }
  return oss.str();
}

std::string results_cache_latency_l1_ns_mb(double latency_ns, double buffer_size_mb) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(Constants::LATENCY_PRECISION);
  if (std::isinf(latency_ns) || std::isnan(latency_ns)) {
    oss << "  L1 Cache: N/A ns (Buffer size: " << buffer_size_mb << " MB)";
  } else {
    oss << "  L1 Cache: " << latency_ns << " ns (Buffer size: " << buffer_size_mb << " MB)";
  }
  return oss.str();
}

std::string results_cache_latency_l2_ns(double latency_ns, size_t buffer_size) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(Constants::LATENCY_PRECISION);
  if (std::isinf(latency_ns) || std::isnan(latency_ns)) {
    oss << "  L2 Cache: N/A ns (Buffer size: " << buffer_size << " B)";
  } else {
    oss << "  L2 Cache: " << latency_ns << " ns (Buffer size: " << buffer_size << " B)";
  }
  return oss.str();
}

std::string results_cache_latency_l2_ns_kb(double latency_ns, double buffer_size_kb) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(Constants::LATENCY_PRECISION);
  if (std::isinf(latency_ns) || std::isnan(latency_ns)) {
    oss << "  L2 Cache: N/A ns (Buffer size: " << buffer_size_kb << " KB)";
  } else {
    oss << "  L2 Cache: " << latency_ns << " ns (Buffer size: " << buffer_size_kb << " KB)";
  }
  return oss.str();
}

std::string results_cache_latency_l2_ns_mb(double latency_ns, double buffer_size_mb) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(Constants::LATENCY_PRECISION);
  if (std::isinf(latency_ns) || std::isnan(latency_ns)) {
    oss << "  L2 Cache: N/A ns (Buffer size: " << buffer_size_mb << " MB)";
  } else {
    oss << "  L2 Cache: " << latency_ns << " ns (Buffer size: " << buffer_size_mb << " MB)";
  }
  return oss.str();
}

std::string results_measurement_unavailable(const std::string& label,
                                            const std::string& status,
                                            const std::string& reason) {
  std::ostringstream oss;
  oss << label << ": N/A [" << status;
  if (!reason.empty()) {
    oss << ": " << reason;
  }
  oss << "]";
  return oss.str();
}

#define BENCHMARK_REASON_FUNCTION(name, text) \
  const std::string& name() {                 \
    static const std::string message = text;  \
    return message;                           \
  }

BENCHMARK_REASON_FUNCTION(benchmark_reason_interrupted_before_measurement,
                          "interrupted before measurement")
BENCHMARK_REASON_FUNCTION(benchmark_reason_interrupted_by_user,
                          "interrupted by user")
BENCHMARK_REASON_FUNCTION(benchmark_reason_planned_measurements_unavailable,
                          "one or more planned measurements unavailable")
BENCHMARK_REASON_FUNCTION(benchmark_reason_invalid_locality_work,
                          "invalid locality-comparison work")
BENCHMARK_REASON_FUNCTION(benchmark_reason_locality_comparison_unavailable,
                          "paired locality comparison unavailable")
BENCHMARK_REASON_FUNCTION(benchmark_reason_interrupted_calibration_pilot,
                          "interrupted during calibration pilot")
BENCHMARK_REASON_FUNCTION(benchmark_reason_invalid_calibration_pilot,
                          "invalid calibration pilot duration")
BENCHMARK_REASON_FUNCTION(benchmark_reason_interrupted_measured_operation,
                          "interrupted during measured operation")
BENCHMARK_REASON_FUNCTION(benchmark_reason_invalid_bandwidth_duration,
                          "invalid measured bandwidth duration")
BENCHMARK_REASON_FUNCTION(benchmark_reason_invalid_bandwidth_value,
                          "invalid measured bandwidth value")
BENCHMARK_REASON_FUNCTION(benchmark_reason_interrupted_latency_pilot,
                          "interrupted during latency calibration pilot")
BENCHMARK_REASON_FUNCTION(benchmark_reason_interrupted_latency_measurement,
                          "interrupted during latency measurement")
BENCHMARK_REASON_FUNCTION(benchmark_reason_invalid_latency_measurement,
                          "invalid latency duration or access count")
BENCHMARK_REASON_FUNCTION(benchmark_reason_invalid_cache_latency_measurement,
                          "invalid cache latency duration or access count")
BENCHMARK_REASON_FUNCTION(benchmark_reason_invalid_main_latency_measurement,
                          "invalid main-memory latency duration")
BENCHMARK_REASON_FUNCTION(benchmark_reason_invalid_bandwidth_measurement,
                          "invalid measured bandwidth or duration")
BENCHMARK_REASON_FUNCTION(benchmark_reason_loops_remain,
                          "benchmark loops remain")
BENCHMARK_REASON_FUNCTION(benchmark_reason_checkpoint_failed,
                          "failed to checkpoint standard benchmark JSON")
BENCHMARK_REASON_FUNCTION(benchmark_reason_unknown_loop_exception,
                          "standard benchmark loop threw an unknown exception")
BENCHMARK_REASON_FUNCTION(
    benchmark_reason_unknown_coordinator_exception,
    "standard benchmark coordinator threw an unknown exception")
BENCHMARK_REASON_FUNCTION(benchmark_reason_invalid_bandwidth_plan,
                          "invalid bandwidth work-plan parameters")
BENCHMARK_REASON_FUNCTION(benchmark_reason_no_worker_partition,
                          "no valid aligned worker partition")
BENCHMARK_REASON_FUNCTION(benchmark_reason_copy_payload_overflow,
                          "copy payload overflow")
BENCHMARK_REASON_FUNCTION(benchmark_reason_total_payload_overflow,
                          "total payload overflow or pass limit")
BENCHMARK_REASON_FUNCTION(benchmark_reason_invalid_latency_plan,
                          "invalid latency work-plan parameters")
BENCHMARK_REASON_FUNCTION(benchmark_reason_latency_chain_too_short,
                          "latency chain requires at least two nodes")
BENCHMARK_REASON_FUNCTION(benchmark_reason_minimum_cycles_exceed_limit,
                          "minimum complete-cycle access count exceeds limit")
BENCHMARK_REASON_FUNCTION(benchmark_reason_rounded_accesses_exceed_limit,
                          "rounded complete-cycle access count exceeds limit")

#undef BENCHMARK_REASON_FUNCTION

std::string benchmark_reason_prepare_failed(const std::string& phase_name) {
  return "failed to prepare " + phase_name + " buffers";
}

std::string benchmark_reason_coordinator_exception(const std::string& error) {
  return "standard benchmark coordinator exception: " + error;
}

std::string benchmark_reason_latency_chain_setup_failed(
    const std::string& phase_name) {
  return "failed to construct " + phase_name + " latency chain";
}

} // namespace Messages

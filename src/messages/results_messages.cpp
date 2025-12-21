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
#include "messages/messages.h"
#include "constants.h"  // For default values
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

std::string results_latency_average(double latency_ns) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);
  oss << "  Average latency: " << latency_ns << " ns";
  return oss.str();
}

std::string results_cache_bandwidth() {
  return "\nCache Bandwidth Tests (single-threaded):";
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
  oss << std::fixed << std::setprecision(2);
  oss << " (Buffer size: " << buffer_size_kb << " KB)";
  return oss.str();
}

std::string results_buffer_size_mb(double buffer_size_mb) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);
  oss << " (Buffer size: " << buffer_size_mb << " MB)";
  return oss.str();
}

std::string results_cache_latency_ns(double latency_ns) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);
  oss << " " << latency_ns << " ns";
  return oss.str();
}

std::string results_separator() {
  return "--------------";
}

std::string results_cache_latency_custom_ns(double latency_ns, size_t buffer_size) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);
  if (std::isinf(latency_ns) || std::isnan(latency_ns)) {
    oss << "  Custom Cache: N/A ns (Buffer size: " << buffer_size << " B)";
  } else {
    oss << "  Custom Cache: " << latency_ns << " ns (Buffer size: " << buffer_size << " B)";
  }
  return oss.str();
}

std::string results_cache_latency_custom_ns_kb(double latency_ns, double buffer_size_kb) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);
  if (std::isinf(latency_ns) || std::isnan(latency_ns)) {
    oss << "  Custom Cache: N/A ns (Buffer size: " << buffer_size_kb << " KB)";
  } else {
    oss << "  Custom Cache: " << latency_ns << " ns (Buffer size: " << buffer_size_kb << " KB)";
  }
  return oss.str();
}

std::string results_cache_latency_custom_ns_mb(double latency_ns, double buffer_size_mb) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);
  if (std::isinf(latency_ns) || std::isnan(latency_ns)) {
    oss << "  Custom Cache: N/A ns (Buffer size: " << buffer_size_mb << " MB)";
  } else {
    oss << "  Custom Cache: " << latency_ns << " ns (Buffer size: " << buffer_size_mb << " MB)";
  }
  return oss.str();
}

std::string results_cache_latency_l1_ns(double latency_ns, size_t buffer_size) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);
  if (std::isinf(latency_ns) || std::isnan(latency_ns)) {
    oss << "  L1 Cache: N/A ns (Buffer size: " << buffer_size << " B)";
  } else {
    oss << "  L1 Cache: " << latency_ns << " ns (Buffer size: " << buffer_size << " B)";
  }
  return oss.str();
}

std::string results_cache_latency_l1_ns_kb(double latency_ns, double buffer_size_kb) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);
  if (std::isinf(latency_ns) || std::isnan(latency_ns)) {
    oss << "  L1 Cache: N/A ns (Buffer size: " << buffer_size_kb << " KB)";
  } else {
    oss << "  L1 Cache: " << latency_ns << " ns (Buffer size: " << buffer_size_kb << " KB)";
  }
  return oss.str();
}

std::string results_cache_latency_l1_ns_mb(double latency_ns, double buffer_size_mb) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);
  if (std::isinf(latency_ns) || std::isnan(latency_ns)) {
    oss << "  L1 Cache: N/A ns (Buffer size: " << buffer_size_mb << " MB)";
  } else {
    oss << "  L1 Cache: " << latency_ns << " ns (Buffer size: " << buffer_size_mb << " MB)";
  }
  return oss.str();
}

std::string results_cache_latency_l2_ns(double latency_ns, size_t buffer_size) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);
  if (std::isinf(latency_ns) || std::isnan(latency_ns)) {
    oss << "  L2 Cache: N/A ns (Buffer size: " << buffer_size << " B)";
  } else {
    oss << "  L2 Cache: " << latency_ns << " ns (Buffer size: " << buffer_size << " B)";
  }
  return oss.str();
}

std::string results_cache_latency_l2_ns_kb(double latency_ns, double buffer_size_kb) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);
  if (std::isinf(latency_ns) || std::isnan(latency_ns)) {
    oss << "  L2 Cache: N/A ns (Buffer size: " << buffer_size_kb << " KB)";
  } else {
    oss << "  L2 Cache: " << latency_ns << " ns (Buffer size: " << buffer_size_kb << " KB)";
  }
  return oss.str();
}

std::string results_cache_latency_l2_ns_mb(double latency_ns, double buffer_size_mb) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);
  if (std::isinf(latency_ns) || std::isnan(latency_ns)) {
    oss << "  L2 Cache: N/A ns (Buffer size: " << buffer_size_mb << " MB)";
  } else {
    oss << "  L2 Cache: " << latency_ns << " ns (Buffer size: " << buffer_size_mb << " MB)";
  }
  return oss.str();
}

} // namespace Messages


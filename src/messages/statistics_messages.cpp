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
#include <sstream>
#include <iomanip>

namespace Messages {

// --- Statistics Messages ---
std::string statistics_header(int loop_count) {
  std::ostringstream oss;
  oss << "\n--- Statistics Across " << loop_count << " Loops ---";
  return oss.str();
}

std::string statistics_metric_name(const std::string& metric_name) {
  return metric_name + ":";
}

std::string statistics_average(double value, int precision) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision);
  oss << "  Average: " << value;
  return oss.str();
}

std::string statistics_median_p50(double value, int precision) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision);
  oss << "  Median (P50): " << value;
  return oss.str();
}

std::string statistics_p90(double value, int precision) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision);
  oss << "  P90: " << value;
  return oss.str();
}

std::string statistics_p95(double value, int precision) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision);
  oss << "  P95: " << value;
  return oss.str();
}

std::string statistics_p99(double value, int precision) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision);
  oss << "  P99: " << value;
  return oss.str();
}

std::string statistics_stddev(double value, int precision) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision);
  oss << "  Stddev: " << value;
  return oss.str();
}

std::string statistics_min(double value, int precision) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision);
  oss << "  Min:     " << value;
  return oss.str();
}

std::string statistics_max(double value, int precision) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision);
  oss << "  Max:     " << value;
  return oss.str();
}

std::string statistics_cache_bandwidth_header(const std::string& cache_name) {
  std::ostringstream oss;
  oss << "\n" << cache_name << " Cache Bandwidth (GB/s):";
  return oss.str();
}

std::string statistics_cache_read() {
  return "  Read:";
}

std::string statistics_cache_write() {
  return "  Write:";
}

std::string statistics_cache_copy() {
  return "  Copy:";
}

std::string statistics_cache_latency_header() {
  return "\nCache Latency (ns):";
}

std::string statistics_cache_latency_name(const std::string& cache_name) {
  return "  " + cache_name + " Cache:";
}

std::string statistics_median_p50_from_samples(double value, size_t sample_count, int precision) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision);
  oss << "    Median (P50): " << value << " (from " << sample_count << " samples)";
  return oss.str();
}

std::string statistics_main_memory_latency_header() {
  return "\nMain Memory Latency (ns):";
}

std::string statistics_footer() {
  return "----------------------------------";
}

} // namespace Messages


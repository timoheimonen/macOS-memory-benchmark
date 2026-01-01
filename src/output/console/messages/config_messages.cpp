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
#include "messages.h"
#include <sstream>
#include <iomanip>

namespace Messages {

// --- Configuration Output Messages ---
std::string config_header(const std::string& version) {
  std::ostringstream oss;
  oss << "----- macOS-memory-benchmark v" << version << " -----";
  return oss.str();
}

std::string config_copyright() {
  return "Copyright 2025-2026 Timo Heimonen <timo.heimonen@proton.me>";
}

std::string config_license() {
  return "This program is free software: you can redistribute it and/or modify\n"
         "it under the terms of the GNU General Public License as published by\n"
         "the Free Software Foundation, either version 3 of the License, or\n"
         "(at your option) any later version.\n"
         "This program is distributed in the hope that it will be useful,\n"
         "but WITHOUT ANY WARRANTY; without even the implied warranty of\n"
         "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n"
         "See <https://www.gnu.org/licenses/> for more details.\n";
}

std::string config_buffer_size(double buffer_size_mib, unsigned long buffer_size_mb) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);
  oss << "Buffer Size (per buffer): " << buffer_size_mib << " MiB (" << buffer_size_mb << " MB requested/capped)";
  return oss.str();
}

std::string config_total_allocation(double total_mib) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);
  oss << "Total Allocation Size: ~" << total_mib << " MiB";
  return oss.str();
}

std::string config_iterations(int iterations) {
  std::ostringstream oss;
  oss << "Iterations (per R/W/Copy test per loop): " << iterations;
  return oss.str();
}

std::string config_loop_count(int loop_count) {
  std::ostringstream oss;
  oss << "Loop Count (total benchmark repetitions): " << loop_count;
  return oss.str();
}

std::string config_non_cacheable(bool use_non_cacheable) {
  std::ostringstream oss;
  oss << "Non-Cacheable Memory Hints: " << (use_non_cacheable ? "Enabled" : "Disabled");
  return oss.str();
}

std::string config_processor_name(const std::string& cpu_name) {
  return "\nProcessor Name: " + cpu_name;
}

std::string config_processor_name_error() {
  return "Could not retrieve processor name.";
}

std::string config_performance_cores(int perf_cores) {
  std::ostringstream oss;
  oss << "  Performance Cores: " << perf_cores;
  return oss.str();
}

std::string config_efficiency_cores(int eff_cores) {
  std::ostringstream oss;
  oss << "  Efficiency Cores: " << eff_cores;
  return oss.str();
}

std::string config_total_cores(int num_threads) {
  std::ostringstream oss;
  oss << "  Total CPU Cores Detected: " << num_threads;
  return oss.str();
}

} // namespace Messages


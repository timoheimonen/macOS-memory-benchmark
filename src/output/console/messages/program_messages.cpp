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
 * @file program_messages.cpp
 * @brief Main program messages
 *
 * Provides implementations for main program message generation functions including
 * benchmark execution status messages, program usage/help text, and version information.
 */

#include "messages.h"
#include "core/config/constants.h"  // For default values
#include <sstream>
#include <iomanip>

namespace Messages {

// --- Main Program Messages ---
const std::string& msg_running_benchmarks() {
  static const std::string msg = "\nRunning benchmarks...";
  return msg;
}

std::string msg_done_total_time(double total_time_sec) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(Constants::TIME_PRECISION);
  oss << "\nDone. Total execution time: " << total_time_sec << " s";
  return oss.str();
}

const std::string& msg_running_pattern_benchmarks() {
  static const std::string msg = "\nRunning Pattern Benchmarks...\n";
  return msg;
}

const std::string& msg_running_tlb_analysis() {
  static const std::string msg = "\nRunning standalone TLB analysis...";
  return msg;
}

std::string msg_tlb_analysis_locality_progress(size_t current_index, size_t total_count, size_t locality_kb) {
  std::ostringstream oss;
  oss << "  [" << current_index << "/" << total_count << "] Locality " << locality_kb << " KB";
  return oss.str();
}

std::string msg_tlb_analysis_page_walk_progress(size_t locality_mb) {
  std::ostringstream oss;
  oss << "  [Page Walk] Locality " << locality_mb << " MB";
  return oss.str();
}

// --- Usage/Help Messages ---
std::string usage_header(const std::string& version) {
  std::ostringstream oss;
  oss << "Copyright 2025-2026 Timo Heimonen <timo.heimonen@proton.me>\n"
      << "Version: " << version << "\n"
      << "License: GNU GPL v3. See <https://www.gnu.org/licenses/>\n"
      << "This program is free software: you can redistribute it and/or modify\n"
      << "it under the terms of the GNU General Public License as published by\n"
      << "the Free Software Foundation, either version 3 of the License, or\n"
      << "(at your option) any later version.\n"
      << "This program is distributed in the hope that it will be useful,\n"
      << "but WITHOUT ANY WARRANTY; without even the implied warranty of\n"
      << "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n"
      << "Link: https://github.com/timoheimonen/macOS-memory-benchmark\n\n";
  return oss.str();
}

std::string usage_options(const std::string& prog_name) {
  std::ostringstream oss;
  oss << "Usage: " << prog_name << " [options]\n"
      << "Options:\n"
      << "  -iterations <count>   Number of iterations for R/W/Copy tests (default: " << Constants::DEFAULT_ITERATIONS << ")\n"
      << "  -buffersize <size_mb> Size for EACH of the 3 buffers in Megabytes (MB) as integer (default: " << Constants::DEFAULT_BUFFER_SIZE_MB << ").\n"
      << "                        The maximum allowed <size_mb> is automatically determined such that\n"
      << "                        3 * <size_mb> does not exceed ~80% of available system memory.\n"
      << "                        In -only-latency mode, -buffersize 0 disables main memory latency.\n"
      << "  -count <count>        Number of full loops (read/write/copy/latency) (default: " << Constants::DEFAULT_LOOP_COUNT << ").\n"
      << "                        When count > 1, statistics include percentiles (P50/P90/P95/P99) and stddev.\n"
      << "  -analyze-tlb          Run standalone TLB analysis benchmark mode (allows optional -output <file> only).\n"
      << "  -latency-samples <count> Number of latency samples to collect per test (default: " << Constants::DEFAULT_LATENCY_SAMPLE_COUNT << ")\n"
      << "  -latency-stride-bytes <bytes> Stride used by latency pointer chains (default: "
      << Constants::LATENCY_STRIDE_BYTES << " bytes).\n"
      << "                        Must be > 0 and a multiple of pointer size (typically 8 bytes).\n"
      << "  -latency-tlb-locality-kb <size_kb> TLB-locality window for latency pointer chains (default: 16 KB; set 0 to disable).\n"
      << "                        Must be a multiple of system page size (typically 4 KB or 16 KB).\n"
      << "  -threads <count>      Number of threads to use for benchmarks (default: detected\n"
      << "                        CPU core count). Applies to all benchmarks including cache tests.\n"
      << "                        If specified value exceeds available cores, it will be capped to\n"
      << "                        the maximum number of cores with a warning.\n"
      << "  -cache-size <size_kb> Custom cache size in Kilobytes (KB) as integer ("
      << Constants::MIN_CACHE_SIZE_KB << " KB to " << Constants::MAX_CACHE_SIZE_KB << " KB).\n"
      << "                        Minimum is 16 KB (system page size). When set, skips automatic\n"
      << "                        L1/L2 cache size detection and only performs bandwidth and latency\n"
      << "                        tests for the custom cache size.\n"
      << "                        In -only-latency mode, -cache-size 0 disables cache latency.\n"
      << "  -patterns             Run pattern benchmarks (sequential forward/reverse, strided,\n"
      << "                        and random access patterns). When set, only pattern benchmarks\n"
      << "                        are executed, skipping standard bandwidth and latency tests.\n"
      << "                        use with -buffersize <size_mb> to set the buffer size for the pattern benchmarks.\n"
      << "  -only-bandwidth       Run only bandwidth tests (read/write/copy for main memory and cache).\n"
      << "                        Skips all latency tests. Cannot be used with -patterns, -cache-size,\n"
      << "                        or -latency-samples.\n"
      << "  -only-latency         Run only latency tests (main memory and cache latency).\n"
      << "                        Skips all bandwidth tests. Cannot be used with -patterns or -iterations.\n"
      << "                        Use -buffersize 0 to disable main memory latency, or -cache-size 0\n"
      << "                        to disable cache latency.\n"
      << "  -non-cacheable        Apply cache-discouraging hints to src/dst buffers.\n"
      << "                        Uses madvise() hints to discourage caching, but does NOT provide\n"
      << "                        true non-cacheable memory (user-space cannot modify page tables).\n"
      << "                        Best-effort approach that may reduce but not eliminate caching.\n"
      << "  -output <file>        Save benchmark results to JSON file. If path is relative,\n"
      << "                        file is saved in current working directory.\n"
      << "  -h, --help            Show this help message and exit\n\n";
  return oss.str();
}

std::string usage_example(const std::string& prog_name) {
  std::ostringstream oss;
  oss << "Example: " << prog_name << " -iterations 2000 -buffersize 1024 -output results.json\n";
  return oss.str();
}

const std::string& report_tlb_header() {
  static const std::string msg = "--- TLB Analysis Report ---";
  return msg;
}

std::string report_tlb_cpu(const std::string& cpu_name) {
  if (cpu_name.empty()) {
    return "CPU: Unknown";
  }
  return "CPU: " + cpu_name;
}

std::string report_tlb_page_size(size_t page_size_bytes) {
  std::ostringstream oss;
  oss << "Page Size: " << page_size_bytes << " bytes";
  return oss.str();
}

std::string report_tlb_buffer(size_t buffer_mb, bool locked) {
  std::ostringstream oss;
  oss << "Buffer: " << buffer_mb << " MB (" << (locked ? "Locked" : "Allocated") << ")";
  return oss.str();
}

std::string report_tlb_stride(size_t stride_bytes) {
  std::ostringstream oss;
  oss << "Stride: " << stride_bytes << " bytes";
  return oss.str();
}

std::string report_tlb_loop_config(size_t loops_per_point, size_t accesses_per_loop) {
  std::ostringstream oss;
  oss << "Loops per Point: " << loops_per_point << ", Accesses per Loop: " << accesses_per_loop;
  return oss.str();
}

const std::string& report_tlb_l1_section() {
  static const std::string msg = "[L1 TLB Detection]";
  return msg;
}

const std::string& report_tlb_l2_section() {
  static const std::string msg = "[L2 TLB / Page Walk]";
  return msg;
}

std::string report_tlb_boundary_kb(size_t boundary_kb) {
  std::ostringstream oss;
  oss << "  Boundary: " << boundary_kb << " KB";
  return oss.str();
}

std::string report_tlb_inferred_size_entries(size_t entries) {
  std::ostringstream oss;
  oss << "  Inferred Size: " << entries << " entries";
  return oss.str();
}

std::string report_tlb_inferred_reach_entries(size_t entries) {
  std::ostringstream oss;
  oss << "  Inferred Reach: " << entries << " entries";
  return oss.str();
}

std::string report_tlb_confidence(const std::string& confidence, double step_ns, double step_percent) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(1);
  oss << "  Confidence: " << confidence << " (Step: +" << step_ns << "ns, +"
      << (step_percent * 100.0) << "%)";
  return oss.str();
}

std::string report_tlb_page_walk_penalty(double penalty_ns, size_t from_kb, size_t to_mb) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(1);
  oss << "  Page Table Walk Penalty (" << from_kb << "KB -> " << to_mb << "MB): ~"
      << penalty_ns << "ns";
  return oss.str();
}

std::string report_tlb_page_walk_penalty_unavailable(size_t required_buffer_mb, size_t selected_buffer_mb) {
  std::ostringstream oss;
  oss << "  Page Table Walk Penalty (16KB -> 512MB): N/A "
      << "(requires " << required_buffer_mb << " MB or larger analysis buffer, selected "
      << selected_buffer_mb << " MB)";
  return oss.str();
}

const std::string& report_tlb_not_detected() {
  static const std::string msg = "  Boundary: Not detected";
  return msg;
}

} // namespace Messages

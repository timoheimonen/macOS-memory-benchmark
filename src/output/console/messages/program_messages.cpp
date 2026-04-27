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

#include "messages_api.h"
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

std::string msg_pattern_benchmark_loop_completed(int current_loop, int total_loops) {
  std::ostringstream oss;
  oss << "Pattern benchmarks - Loop " << current_loop << "/" << total_loops << " completed";
  return oss.str();
}

std::string msg_results_saved_to(const std::string& file_path) {
  return "Results saved to: " + file_path;
}

const std::string& msg_running_tlb_analysis() {
  static const std::string msg = "\nRunning standalone TLB analysis...";
  return msg;
}

const std::string& msg_interrupted_by_user() {
  static const std::string msg = "\nInterrupted by user. Partial results shown.";
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

std::string msg_tlb_analysis_refinement_start(size_t point_count) {
  std::ostringstream oss;
  oss << "Starting refinement sweep (" << point_count << " points)...";
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
      << "  -benchmark            Run standard memory benchmark (bandwidth and latency).\n"
      << "  -iterations <count>   Number of iterations for R/W/Copy tests (default: " << Constants::DEFAULT_ITERATIONS << ")\n"
      << "  -buffersize <size_mb> Main-memory buffer size in Megabytes (MB) as integer (default: " << Constants::DEFAULT_BUFFER_SIZE_MB << ").\n"
      << "                        Peak concurrent main-memory allocation is up to 2 * <size_mb>\n"
      << "                        (read/copy source + destination). In -only-latency mode peak is 1 * <size_mb>.\n"
      << "                        In -only-latency mode, -buffersize 0 disables main memory latency.\n"
      << "  -count <count>        Number of full loops (read/write/copy/latency) (default: " << Constants::DEFAULT_LOOP_COUNT << ").\n"
      << "                        When count > 1, statistics include percentiles (P50/P90/P95/P99) and stddev.\n"
      << "  -analyze-tlb          Run standalone TLB analysis benchmark mode (allows optional -output <file>, -latency-stride-bytes <bytes>,\n"
      << "                        -latency-chain-mode <mode>, and -tlb-density <low|medium|high> only).\n"
      << "  -tlb-density <level>  Sweep density for -analyze-tlb: low, medium, high (default: high).\n"
      << "                        low = 15-point base sweep, no refinement. medium = 15-point base + refinement.\n"
      << "                        high = 29-point base + refinement.\n"
      << "  -analyze-core2core    Run standalone core-to-core cache-line handoff benchmark mode\n"
      << "                        (allows optional -output <file>, -count <count>, and -latency-samples <count> only).\n"
      << "  -latency-samples <count> Number of latency samples to collect per test (default: " << Constants::DEFAULT_LATENCY_SAMPLE_COUNT << ")\n"
      << "  -latency-stride-bytes <bytes> Stride used by latency pointer chains (default: "
      << Constants::LATENCY_STRIDE_BYTES << " bytes).\n"
      << "                        Must be > 0 and a multiple of pointer size (typically 8 bytes).\n"
      << "  -latency-chain-mode <mode> Pointer-chain construction policy. Modes: auto (default), global-random,\n"
      << "                        random-box, same-random-in-box, diff-random-in-box.\n"
      << "                        -analyze-tlb rejects global-random because locality sweep labels would be misleading.\n"
      << "  -latency-tlb-locality-kb <size_kb> TLB-locality window for latency pointer chains (default: "
      << Constants::DEFAULT_LATENCY_TLB_LOCALITY_KB
      << " KB; set 0 for global random).\n"
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
      << "                        Cannot be used with -benchmark.\n"
      << "                        use with -buffersize <size_mb> to set the buffer size for the pattern benchmarks.\n"
      << "  -only-bandwidth       Run only bandwidth tests (read/write/copy for main memory and cache).\n"
      << "                        Skips all latency tests. Requires -benchmark. Cannot be used with -patterns,\n"
      << "                        -cache-size, or -latency-samples.\n"
      << "  -only-latency         Run only latency tests (main memory and cache latency).\n"
      << "                        Skips all bandwidth tests. Requires -benchmark. Cannot be used with -patterns\n"
      << "                        or -iterations.\n"
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
  oss << "Example: " << prog_name << " -benchmark -iterations 2000 -buffersize 1024 -output results.json\n";
  return oss.str();
}

const std::string& report_tlb_header() {
  static const std::string msg = "--- TLB Analysis Report ---";
  return msg;
}

const std::string& report_tlb_settings_header() {
  static const std::string msg = "--- TLB Analysis Settings ---";
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

std::string report_tlb_density(const std::string& density_name) {
  return "Sweep Density: " + density_name;
}

std::string report_tlb_chain_mode(const std::string& chain_mode_name) {
  return "Chain Mode: " + chain_mode_name;
}

std::string report_tlb_chain_mode_requested(const std::string& chain_mode_name) {
  return "Requested Chain Mode: " + chain_mode_name;
}

std::string report_tlb_chain_mode_effective(const std::string& chain_mode_name) {
  return "Effective Chain Mode: " + chain_mode_name;
}

std::string report_tlb_loop_config(size_t loops_per_point, size_t accesses_per_loop) {
  std::ostringstream oss;
  oss << "Loops per Point: " << loops_per_point << ", Accesses per Loop: " << accesses_per_loop;
  return oss.str();
}

std::string report_tlb_sweep_range(size_t start_locality_bytes,
                                   size_t end_locality_bytes,
                                   size_t point_count) {
  std::ostringstream oss;
  oss << "Sweep Locality Range: ";

  if (start_locality_bytes < 1024 * 1024) {
    oss << (start_locality_bytes / 1024) << " KB";
  } else {
    oss << (start_locality_bytes / (1024 * 1024)) << " MB";
  }

  oss << " -> ";

  if (end_locality_bytes < 1024 * 1024) {
    oss << (end_locality_bytes / 1024) << " KB";
  } else {
    oss << (end_locality_bytes / (1024 * 1024)) << " MB";
  }

  oss << " (" << point_count << " points)";
  return oss.str();
}

std::string report_tlb_page_walk_config(bool enabled,
                                        size_t comparison_locality_mb,
                                        size_t required_buffer_mb,
                                        size_t selected_buffer_mb) {
  std::ostringstream oss;
  if (enabled) {
    oss << "Page-Walk Comparison: Enabled (" << comparison_locality_mb << " MB locality)";
    return oss.str();
  }

  oss << "Page-Walk Comparison: Disabled (requires " << required_buffer_mb
      << " MB analysis buffer, selected " << selected_buffer_mb << " MB)";
  return oss.str();
}

std::string report_tlb_fine_sweep(size_t added_points, size_t total_points) {
  std::ostringstream oss;
  oss << "Fine Sweep: Added " << added_points << " refinement point";
  if (added_points != 1) {
    oss << "s";
  }
  oss << ", total " << total_points << " points";
  return oss.str();
}

const std::string& report_tlb_private_cache_section() {
  static const std::string msg = "[Private Cache Detection]";
  return msg;
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
  oss << "  Inferred Size Estimate: ~" << entries << " entries";
  return oss.str();
}

std::string report_tlb_inferred_reach_entries(size_t entries) {
  std::ostringstream oss;
  oss << "  Inferred Reach Estimate: ~" << entries << " entries";
  return oss.str();
}

std::string report_tlb_inferred_entries_range(size_t min_entries, size_t max_entries) {
  std::ostringstream oss;
  oss << "  Inferred Entry Range: " << min_entries;
  if (max_entries != min_entries) {
    oss << "-" << max_entries;
  }
  oss << " entries";
  return oss.str();
}

const std::string& report_tlb_private_cache_overlap() {
  static const std::string msg =
      "  Private Cache Overlap: yes (kept as ambiguous L1 TLB candidate)";
  return msg;
}

std::string report_tlb_confidence(const std::string& confidence, double step_ns, double step_percent) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(1);
  oss << "  Confidence: " << confidence << " (Step: +" << step_ns << "ns, +"
      << (step_percent * 100.0) << "%)";
  return oss.str();
}

std::string report_tlb_private_cache_candidate(bool strong_private_cache_candidate) {
  if (strong_private_cache_candidate) {
    return "  Candidate Type: Strong private-cache candidate";
  }
  return "  Candidate Type: Early-cache candidate";
}

std::string report_tlb_private_cache_interference(bool elevated_risk, size_t locality_kb) {
  std::ostringstream oss;
  if (elevated_risk) {
    oss << "  TLB Interference Risk: Elevated near " << locality_kb << " KB locality";
    return oss.str();
  }
  oss << "  TLB Interference Risk: Low near " << locality_kb << " KB locality";
  return oss.str();
}

std::string report_tlb_private_cache_l1_distance(size_t distance_kb, size_t distance_pages) {
  std::ostringstream oss;
  oss << "  Distance to L1 TLB Boundary: " << distance_kb << " KB (" << distance_pages << " pages)";
  return oss.str();
}

std::string report_tlb_page_walk_penalty(double penalty_ns, size_t from_kb, size_t to_mb) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(1);
  oss << "  Page Table Walk Penalty (" << from_kb << "KB -> " << to_mb << "MB): ~"
      << penalty_ns << "ns";
  return oss.str();
}

std::string report_tlb_page_walk_penalty_unavailable(size_t from_kb,
                                                     size_t to_mb,
                                                     size_t required_buffer_mb,
                                                     size_t selected_buffer_mb) {
  std::ostringstream oss;
  oss << "  Page Table Walk Penalty (" << from_kb << "KB -> " << to_mb << "MB): N/A "
      << "(requires " << required_buffer_mb << " MB or larger analysis buffer, selected "
      << selected_buffer_mb << " MB)";
  return oss.str();
}

std::string report_tlb_page_walk_penalty_interrupted(size_t from_kb, size_t to_mb) {
  std::ostringstream oss;
  oss << "  Page Table Walk Penalty (" << from_kb << "KB -> " << to_mb
      << "MB): N/A (comparison measurement did not complete)";
  return oss.str();
}

const std::string& report_tlb_not_detected() {
  static const std::string msg = "  Boundary: Not detected";
  return msg;
}

} // namespace Messages

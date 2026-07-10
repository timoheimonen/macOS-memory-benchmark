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

#include <cmath>
#include <iomanip>
#include <sstream>

namespace Messages {

namespace {

std::string format_binary_size(size_t bytes) {
  std::ostringstream oss;
  if (bytes >= Constants::BYTES_PER_MB &&
      (bytes % Constants::BYTES_PER_MB) == 0) {
    oss << bytes / Constants::BYTES_PER_MB << " MiB";
  } else if (bytes >= Constants::BYTES_PER_KB &&
             (bytes % Constants::BYTES_PER_KB) == 0) {
    oss << bytes / Constants::BYTES_PER_KB << " KiB";
  } else {
    oss << bytes << " bytes";
  }
  return oss.str();
}

double normalize_tlb_display_latency(double latency_ns) {
  constexpr double kHalfLastDisplayedUnitNs = 0.005;
  return std::abs(latency_ns) < kHalfLastDisplayedUnitNs ? 0.0 : latency_ns;
}

}  // namespace

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

std::string msg_running_sweep(size_t run_count) {
  std::ostringstream oss;
  oss << "\nRunning sweep with " << run_count << " run";
  if (run_count != 1) {
    oss << "s";
  }
  oss << "...";
  return oss.str();
}

std::string msg_sweep_run_progress(size_t current_run, size_t total_runs) {
  std::ostringstream oss;
  oss << "\nSweep run " << current_run << "/" << total_runs;
  return oss.str();
}

std::string msg_tlb_analysis_refinement_start(size_t point_count) {
  std::ostringstream oss;
  oss << "Starting refinement sweep (" << point_count << " points)...";
  return oss.str();
}

std::string msg_tlb_analysis_validation_start(size_t point_count) {
  std::ostringstream oss;
  oss << "Running independent TLB boundary validation pass (" << point_count
      << " locality points)...";
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
      << "Long options require --; single dash is only for one-character aliases.\n"
      << "Options:\n"
      << "  -B, --benchmark       Run calibrated, seeded, balanced standard bandwidth/latency benchmark.\n"
      << "                        JSON uses standard schema 2 methodology "
      << Constants::BENCHMARK_METHODOLOGY_VERSION << ".\n"
      << "                        Continuous latency targets 250 ms in a 100-300 ms window.\n"
      << "  -i, --iterations <count>\n"
      << "                        Exact measured R/W/Copy pass count when explicitly supplied.\n"
      << "                        When omitted, --benchmark and --patterns calibrate toward 150 ms.\n"
      << "  -b, --buffer-size <size_mb>\n"
      << "                        Main-memory buffer size in Megabytes (MB) as integer (default: " << Constants::DEFAULT_BUFFER_SIZE_MB << ").\n"
      << "                        Peak concurrent main-memory allocation is up to 2 * <size_mb>\n"
      << "                        (read/copy source + destination). In --only-latency mode peak is 1 * <size_mb>.\n"
      << "                        In --only-latency mode, --buffer-size 0 disables main memory latency.\n"
      << "  -r, --count <count>   Number of full loops (read/write/copy/latency) (default: " << Constants::DEFAULT_LOOP_COUNT << ").\n"
      << "                        When count > 1, median P50 is the headline; statistics also include\n"
      << "                        P90/P95/P99, stddev, CV, MAD, min, and max.\n"
      << "  -T, --analyze-tlb     Run standalone TLB analysis with paired bootstrap CI and independent\n"
      << "                        boundary validation (allows optional -o/--output <file>,\n"
      << "                        -s/--latency-stride-bytes <bytes>, -m/--latency-chain-mode <mode>,\n"
      << "                        -D/--tlb-density <low|medium|high>, --seed <uint64>,\n"
      << "                        -S/--sweep <key=...>,\n"
      << "                        and -X/--sweep-max-runs <count> only).\n"
      << "                        JSON output uses schema 4 with exact string seeds and scoped counters.\n"
      << "  -D, --tlb-density <level>\n"
      << "                        Runtime profile for --analyze-tlb: low, medium, high (default: medium).\n"
      << "                        low/quick = 15 points, no refinement, 7-12 adaptive rounds.\n"
      << "                        medium/standard = 15 points + refinement, 10-20 rounds.\n"
      << "                        high/exhaustive = 29 points + refinement, 15-30 rounds.\n"
      << "      --seed <uint64>\n"
      << "                        Reproducible workload/schedule seed for --benchmark and --patterns,\n"
      << "                        or planner, round-order, and pointer-chain seed for --analyze-tlb.\n"
      << "                        Generated once per command when omitted.\n"
      << "  -C, --analyze-core2core\n"
      << "                        Run calibrated, balanced core-to-core cache-line handoff analysis.\n"
      << "                        Continuous headlines target 250 ms; repeated headlines use median P50.\n"
      << "                        Defaults to 3 loops for median/CV repeatability diagnostics.\n"
      << "                        JSON uses core-to-core schema 2 with per-loop audit metadata\n"
      << "                        (allows optional -o/--output <file>, -r/--count <count>, -n/--latency-samples <count>,\n"
      << "                        and sweep over count or latency-samples only).\n"
      << "  -n, --latency-samples <count>\n"
      << "                        Number of latency samples to collect per test (default: " << Constants::DEFAULT_LATENCY_SAMPLE_COUNT << ")\n"
      << "                        Samples use a separate pass and do not define the continuous headline.\n"
      << "                        Core-to-core sample windows auto-calibrate toward 1 ms.\n"
      << "  -s, --latency-stride-bytes <bytes>\n"
      << "                        Stride used by latency pointer chains (default: "
      << Constants::LATENCY_STRIDE_BYTES << " bytes).\n"
      << "                        Must be > 0 and a multiple of pointer size (typically 8 bytes).\n"
      << "                        With --analyze-tlb, stride must not exceed the system page size;\n"
      << "                        page-size divisibility is not required. Sweep values are preflight checked.\n"
      << "  -m, --latency-chain-mode <mode>\n"
      << "                        Pointer-chain construction policy. Modes: auto (default), global-random,\n"
      << "                        random-box, same-random-in-box, diff-random-in-box.\n"
      << "                        --analyze-tlb rejects global-random because locality sweep labels would be misleading.\n"
      << "  -l, --latency-tlb-locality-kb <size_kb>\n"
      << "                        TLB-locality window for latency pointer chains (default: "
      << Constants::DEFAULT_LATENCY_TLB_LOCALITY_KB
      << " KB).\n"
      << "                        With auto, 0 selects global random. Locality-using modes require\n"
      << "                        a non-zero system-page multiple; explicit global-random ignores this value.\n"
      << "                        When omitted in --benchmark, reports a paired 16 KiB-locality vs\n"
      << "                        global-random comparison; the delta is not an isolated page-walk cost.\n"
      << "  -t, --threads <count> Number of bandwidth workers. When omitted, main memory and\n"
      << "                        patterns use the detected CPU core count; cache bandwidth uses one.\n"
      << "                        An explicit value applies to all bandwidth targets; latency remains\n"
      << "                        single-threaded. Requests above available cores are capped.\n"
      << "  -k, --cache-size <size_kb>\n"
      << "                        Custom cache size in Kilobytes (KB) as integer ("
      << Constants::MIN_CACHE_SIZE_KB << " KB to " << Constants::MAX_CACHE_SIZE_KB << " KB).\n"
      << "                        Minimum is 16 KB (system page size). When set, skips automatic\n"
      << "                        L1/L2 cache size detection and only performs bandwidth and latency\n"
      << "                        tests for the custom cache size.\n"
      << "                        In --only-latency mode, --cache-size 0 disables cache latency.\n"
      << "  -P, --patterns        Run pattern benchmarks (sequential forward/reverse, strided,\n"
      << "                        and random access patterns). When set, only pattern benchmarks\n"
      << "                        are executed, skipping standard bandwidth and latency tests.\n"
      << "                        Samples auto-calibrate toward 150 ms unless --iterations is explicit.\n"
      << "                        Cannot be used with --benchmark.\n"
      << "                        use with --buffer-size <size_mb> to set the buffer size for the pattern benchmarks.\n"
      << "  -W, --only-bandwidth  Run only bandwidth tests (read/write/copy for main memory and cache).\n"
      << "                        Skips all latency tests. Requires --benchmark. Cannot be used with --patterns,\n"
      << "                        --cache-size, or --latency-samples.\n"
      << "  -L, --only-latency    Run only latency tests (main memory and cache latency).\n"
      << "                        Skips all bandwidth tests. Requires --benchmark. Cannot be used with --patterns\n"
      << "                        or --iterations.\n"
      << "                        Use --buffer-size 0 to disable main memory latency, or --cache-size 0\n"
      << "                        to disable cache latency.\n"
      << "  -u, --non-cacheable   Apply cache-discouraging hints to src/dst buffers.\n"
      << "                        Uses madvise() hints to discourage caching, but does NOT provide\n"
      << "                        true non-cacheable memory (user-space cannot modify page tables).\n"
      << "                        Best-effort approach that may reduce but not eliminate caching.\n"
      << "  -o, --output <file>   Save benchmark results to JSON file. If path is relative,\n"
      << "                        file is saved in current working directory. Standard runs checkpoint\n"
      << "                        atomically after completed loops and expose incomplete status.\n"
      << "  -S, --sweep <key=a,b> Run a Cartesian sweep over one parameter. Repeat for multiple\n"
      << "                        parameters. Supported keys: buffer-size, cache-size, threads,\n"
      << "                        latency-tlb-locality-kb, latency-stride-bytes,\n"
      << "                        latency-chain-mode, tlb-density. With --analyze-tlb,\n"
      << "                        supported keys are latency-stride-bytes, latency-chain-mode,\n"
      << "                        and tlb-density. With --analyze-core2core,\n"
      << "                        supported keys are count and latency-samples. Requires --output <file>.\n"
      << "  -X, --sweep-max-runs <n>\n"
      << "                        Maximum generated sweep runs (default: " << Constants::DEFAULT_SWEEP_MAX_RUNS
      << "; --analyze-tlb: " << Constants::DEFAULT_ANALYZE_TLB_SWEEP_MAX_RUNS << ").\n"
      << "  -h, --help            Show this help message and exit\n\n";
  return oss.str();
}

std::string usage_example(const std::string& prog_name) {
  std::ostringstream oss;
  oss << "Example: " << prog_name << " --benchmark --iterations 2000 --buffer-size 1024 --output results.json\n";
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

std::string report_tlb_run_summary(const std::string& cpu_name,
                                   size_t page_size_bytes,
                                   size_t stride_bytes,
                                   const std::string& profile_name,
                                   const std::string& requested_chain_mode,
                                   const std::string& effective_chain_mode,
                                   uint64_t seed,
                                   bool user_specified_seed) {
  std::ostringstream oss;
  oss << "Run: " << (cpu_name.empty() ? "Unknown CPU" : cpu_name)
      << " | page " << format_binary_size(page_size_bytes)
      << " | stride " << stride_bytes << " B | " << profile_name
      << " | mode " << requested_chain_mode;
  if (requested_chain_mode != effective_chain_mode) {
    oss << "->" << effective_chain_mode;
  }
  oss << " | seed " << seed << " ("
      << (user_specified_seed ? "user" : "generated") << ")";
  return oss.str();
}

std::string report_tlb_resource_summary(size_t buffer_mb,
                                        bool buffer_locked,
                                        bool qos_requested,
                                        bool qos_applied,
                                        int qos_code,
                                        size_t memory_budget_mb,
                                        size_t estimated_peak_memory_bytes) {
  const double peak_memory_mib =
      static_cast<double>(estimated_peak_memory_bytes) /
      static_cast<double>(Constants::BYTES_PER_MB);
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(1)
      << "Resources: " << buffer_mb << " MiB buffer ("
      << (buffer_locked ? "locked" : "unlocked") << ") | QoS ";
  if (!qos_requested) {
    oss << "not requested";
  } else if (qos_applied) {
    oss << "applied";
  } else {
    oss << "failed (code " << qos_code << "; best-effort)";
  }
  oss << " | estimated peak/budget " << peak_memory_mib << "/"
      << memory_budget_mb << " MiB";
  return oss.str();
}

std::string report_tlb_sweep_plan(size_t start_locality_bytes,
                                  size_t end_locality_bytes,
                                  size_t point_count,
                                  bool large_comparison_enabled,
                                  size_t comparison_locality_bytes,
                                  size_t required_buffer_mb,
                                  size_t selected_buffer_mb) {
  std::ostringstream oss;
  oss << "Sweep: " << format_binary_size(start_locality_bytes) << " -> "
      << format_binary_size(end_locality_bytes) << " (" << point_count
      << " points) | large comparison ";
  if (large_comparison_enabled) {
    oss << format_binary_size(comparison_locality_bytes) << " enabled";
  } else {
    oss << "unavailable (requires " << required_buffer_mb
        << " MiB buffer, selected " << selected_buffer_mb << " MiB)";
  }
  return oss.str();
}

const std::string& report_tlb_sweep_legend() {
  static const std::string msg =
      "  Cache-hot P50 ns/access: delta=spread-packed; active=cache-line "
      "footprint; * <64-node short-cycle diagnostic";
  return msg;
}

const std::string& report_tlb_quick_profile_note() {
  static const std::string msg =
      "Profile Note: quick results are screening estimates; confirm boundaries "
      "with medium or high.";
  return msg;
}

std::string report_tlb_paired_locality_progress(
    size_t current_index,
    size_t total_count,
    size_t locality_bytes,
    double spread_p50_ns,
    double packed_p50_ns,
    double translation_delta_p50_ns,
    size_t active_cache_line_footprint_bytes,
    bool short_cycle_diagnostic) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(Constants::LATENCY_PRECISION)
      << "  [" << current_index << "/" << total_count << "] "
      << format_binary_size(locality_bytes) << " — delta "
      << normalize_tlb_display_latency(translation_delta_p50_ns)
      << " ns (spread " << normalize_tlb_display_latency(spread_p50_ns)
      << ", packed " << normalize_tlb_display_latency(packed_p50_ns)
      << "; active " << format_binary_size(active_cache_line_footprint_bytes)
      << ")";
  if (short_cycle_diagnostic) {
    oss << " *";
  }
  return oss.str();
}

std::string report_tlb_work_estimate(const std::string& pass_name,
                                     size_t point_count,
                                     size_t min_rounds,
                                     size_t max_rounds,
                                     double estimated_min_duration_sec,
                                     double estimated_max_duration_sec) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2)
      << "Work Estimate [" << pass_name << "]: " << point_count
      << " points, " << min_rounds << "-" << max_rounds
      << " rounds, rough duration " << estimated_min_duration_sec << "-"
      << estimated_max_duration_sec << " s";
  return oss.str();
}

std::string report_tlb_pass_completion(const std::string& pass_name,
                                       size_t rounds_completed,
                                       const std::string& completion_reason) {
  std::ostringstream oss;
  oss << "Pass Completion [" << pass_name << "]: " << rounds_completed
      << " rounds (" << completion_reason << ")";
  return oss.str();
}

std::string report_tlb_fine_sweep(size_t added_points, size_t total_points) {
  std::ostringstream oss;
  oss << "Refinement: +" << added_points << " point";
  if (added_points != 1) oss << "s";
  oss << " (" << total_points << " total)";
  return oss.str();
}

std::string report_tlb_analysis_status(const std::string& status,
                                       size_t planned_points,
                                       size_t measured_points,
                                       bool conclusions_valid) {
  std::ostringstream oss;
  oss << "Analysis Status: " << status << " (measured " << measured_points << "/"
      << planned_points << " planned points, conclusions "
      << (conclusions_valid ? "valid" : "suppressed") << ")";
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
  static const std::string msg = "[L2 TLB Detection]";
  return msg;
}

std::string report_tlb_boundary_kb(size_t boundary_kb) {
  std::ostringstream oss;
  oss << "  Boundary: " << boundary_kb << " KiB";
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

std::string report_tlb_statistical_confidence(const std::string& confidence,
                                              double effect_ns,
                                              double discovery_ci_lower_ns,
                                              double discovery_ci_upper_ns,
                                              double validation_ci_lower_ns,
                                              double validation_ci_upper_ns) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);
  oss << "  Confidence: " << confidence << " (paired effect " << effect_ns
      << "ns; discovery 95% CI " << discovery_ci_lower_ns << ".."
      << discovery_ci_upper_ns << "ns; validation 95% CI "
      << validation_ci_lower_ns << ".." << validation_ci_upper_ns << "ns)";
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
    oss << "  TLB Interference Risk: Elevated near " << locality_kb << " KiB locality";
    return oss.str();
  }
  oss << "  TLB Interference Risk: Low near " << locality_kb << " KiB locality";
  return oss.str();
}

std::string report_tlb_private_cache_l1_distance(size_t distance_kb, size_t distance_pages) {
  std::ostringstream oss;
  oss << "  Distance to L1 TLB Boundary: " << distance_kb << " KiB ("
      << distance_pages << " pages)";
  return oss.str();
}

std::string report_tlb_large_locality_paired_comparison(
    size_t locality_bytes,
    double spread_p50_ns,
    double packed_p50_ns,
    double translation_delta_p50_ns,
    size_t spread_actual_pages,
    size_t packed_actual_pages,
    size_t unique_cache_lines,
    size_t active_cache_line_footprint_bytes) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(Constants::LATENCY_PRECISION)
      << "[Large-Locality Paired Comparison]\n"
      << "  " << format_binary_size(locality_bytes) << " virtual | "
      << format_binary_size(active_cache_line_footprint_bytes) << " active ("
      << unique_cache_lines << " lines) | " << spread_actual_pages << "/"
      << packed_actual_pages << " spread/packed pages\n"
      << "  P50: delta "
      << normalize_tlb_display_latency(translation_delta_p50_ns)
      << " ns/access (spread "
      << normalize_tlb_display_latency(spread_p50_ns) << ", packed "
      << normalize_tlb_display_latency(packed_p50_ns) << ")\n"
      << "  Cache-hot paired translation stress; not DRAM latency or an "
         "isolated page-table-walk cost";
  return oss.str();
}

std::string report_tlb_large_locality_paired_unavailable(
    size_t required_buffer_mb,
    size_t selected_buffer_mb) {
  std::ostringstream oss;
  oss << "[Large-Locality Paired Comparison]\n"
      << "  Result: N/A (requires " << required_buffer_mb
      << " MiB or larger analysis buffer, selected " << selected_buffer_mb
      << " MiB)";
  return oss.str();
}

const std::string& report_tlb_large_locality_paired_interrupted() {
  static const std::string msg =
      "[Large-Locality Paired Comparison]\n"
      "  Result: N/A (analysis incomplete or comparison measurement did not complete)";
  return msg;
}

std::string report_tlb_conclusions_unavailable(const std::string& status) {
  return "  Conclusion: Suppressed because analysis status is " + status;
}

const std::string& report_tlb_not_detected() {
  static const std::string msg = "  Boundary: Not detected";
  return msg;
}

} // namespace Messages

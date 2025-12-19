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
#include "messages.h"
#include "benchmark.h"  // For SOFTVERSION
#include "constants.h"  // For default values
#include <sstream>
#include <iomanip>

namespace Messages {

// --- Error Messages ---
const std::string& error_prefix() {
  static const std::string msg = "Error: ";
  return msg;
}

std::string error_missing_value(const std::string& option) {
  return "Missing value for " + option;
}

std::string error_invalid_value(const std::string& option, const std::string& value, const std::string& reason) {
  return "Invalid value for " + option + ": " + value + " (" + reason + ")";
}

std::string error_unknown_option(const std::string& option) {
  return "Unknown option: " + option;
}

std::string error_buffer_size_calculation(unsigned long size_mb) {
  std::ostringstream oss;
  oss << "Buffer size calculation error (" << size_mb << " MB).";
  return oss.str();
}

std::string error_buffer_size_too_small(size_t size_bytes) {
  std::ostringstream oss;
  oss << "Final buffer size (" << size_bytes << " bytes) is too small.";
  return oss.str();
}

std::string error_cache_size_invalid(long long min_kb, long long max_kb, long long max_mb) {
  std::ostringstream oss;
  oss << "cache-size invalid (must be between " << min_kb << " KB and " << max_kb 
      << " KB (" << max_mb << " MB))";
  return oss.str();
}

const std::string& error_iterations_invalid() {
  static const std::string msg = "iterations invalid";
  return msg;
}

const std::string& error_buffersize_invalid() {
  static const std::string msg = "buffersize invalid";
  return msg;
}

const std::string& error_count_invalid() {
  static const std::string msg = "count invalid";
  return msg;
}

const std::string& error_latency_samples_invalid() {
  static const std::string msg = "latency-samples invalid";
  return msg;
}

std::string error_mmap_failed(const std::string& buffer_name) {
  return "mmap failed for " + buffer_name;
}

std::string error_madvise_failed(const std::string& buffer_name) {
  return "madvise failed for " + buffer_name;
}

std::string error_benchmark_tests(const std::string& error) {
  return "Error during benchmark tests: " + error;
}

std::string error_benchmark_loop(int loop, const std::string& error) {
  std::ostringstream oss;
  oss << "Error during benchmark loop " << loop << ": " << error;
  return oss.str();
}

// --- Warning Messages ---
const std::string& warning_cannot_get_memory() {
  static const std::string msg = "Warning: Cannot get available memory. Using fallback limit.";
  return msg;
}

std::string warning_buffer_size_exceeds_limit(unsigned long requested_mb, unsigned long limit_mb) {
  std::ostringstream oss;
  oss << "Warning: Requested buffer size (" << requested_mb << " MB) > limit ("
      << limit_mb << " MB). Using limit.";
  return oss.str();
}

std::string warning_qos_failed(int code) {
  std::ostringstream oss;
  oss << "Warning: Failed to set QoS class for main thread (code: " << code << ")";
  return oss.str();
}

// --- Info Messages ---
std::string info_setting_max_fallback(unsigned long max_mb) {
  std::ostringstream oss;
  oss << "Info: Setting max per buffer to fallback: " << max_mb << " MB.";
  return oss.str();
}

std::string info_calculated_max_less_than_min(unsigned long max_mb, unsigned long min_mb) {
  std::ostringstream oss;
  oss << "Info: Calculated max (" << max_mb << " MB) < min (" << min_mb
      << " MB). Using min.";
  return oss.str();
}

std::string info_custom_cache_rounded_up(unsigned long original_kb, unsigned long rounded_kb) {
  std::ostringstream oss;
  oss << "Info: Custom cache size (" << original_kb << " KB) rounded up to "
      << rounded_kb << " KB (system page size)";
  return oss.str();
}

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

// --- Usage/Help Messages ---
std::string usage_header(const std::string& version) {
  std::ostringstream oss;
  oss << "Version: " << version << " by Timo Heimonen <timo.heimonen@proton.me>\n"
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
      << "  -count <count>        Number of full loops (read/write/copy/latency) (default: " << Constants::DEFAULT_LOOP_COUNT << ").\n"
      << "                        When count > 1, statistics include percentiles (P50/P90/P95/P99) and stddev.\n"
      << "  -latency-samples <count> Number of latency samples to collect per test (default: " << Constants::DEFAULT_LATENCY_SAMPLE_COUNT << ")\n"
      << "  -cache-size <size_kb> Custom cache size in Kilobytes (KB) as integer (16 KB to 524288 KB).\n"
      << "                        Minimum is 16 KB (system page size). When set, skips automatic\n"
      << "                        L1/L2 cache size detection and only performs bandwidth and latency\n"
      << "                        tests for the custom cache size.\n"
      << "  -h, --help            Show this help message and exit\n\n";
  return oss.str();
}

std::string usage_example(const std::string& prog_name) {
  std::ostringstream oss;
  oss << "Example: " << prog_name << " -iterations 500 -buffersize 1024\n"
      << "Example: " << prog_name << " -cache-size 256\n";
  return oss.str();
}

// --- Configuration Output Messages ---
std::string config_header(const std::string& version) {
  std::ostringstream oss;
  oss << "----- macOS-memory-benchmark v" << version << " -----";
  return oss.str();
}

std::string config_copyright() {
  return "Copyright 2025 Timo Heimonen <timo.heimonen@proton.me>";
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
  oss << "Total Allocation Size: ~" << total_mib << " MiB (for 3 buffers)";
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

// --- Cache Info Messages ---
std::string cache_info_header() {
  return "\nDetected Cache Sizes:";
}

std::string cache_size_custom(size_t size_bytes) {
  std::ostringstream oss;
  oss << "  Custom Cache Size: ";
  if (size_bytes < 1024) {
    oss << size_bytes << " B";
  } else if (size_bytes < 1024 * 1024) {
    oss << std::fixed << std::setprecision(2) << size_bytes / 1024.0 << " KB";
  } else {
    oss << std::fixed << std::setprecision(2) << size_bytes / (1024.0 * 1024.0) << " MB";
  }
  return oss.str();
}

std::string cache_size_l1(size_t size_bytes) {
  std::ostringstream oss;
  oss << "  L1 Cache Size: ";
  if (size_bytes < 1024) {
    oss << size_bytes << " B (per P-core)";
  } else if (size_bytes < 1024 * 1024) {
    oss << std::fixed << std::setprecision(2) << size_bytes / 1024.0 << " KB (per P-core)";
  } else {
    oss << std::fixed << std::setprecision(2) << size_bytes / (1024.0 * 1024.0) << " MB (per P-core)";
  }
  return oss.str();
}

std::string cache_size_l2(size_t size_bytes) {
  std::ostringstream oss;
  oss << "  L2 Cache Size: ";
  if (size_bytes < 1024) {
    oss << size_bytes << " B (per P-core cluster)";
  } else if (size_bytes < 1024 * 1024) {
    oss << std::fixed << std::setprecision(2) << size_bytes / 1024.0 << " KB (per P-core cluster)";
  } else {
    oss << std::fixed << std::setprecision(2) << size_bytes / (1024.0 * 1024.0) << " MB (per P-core cluster)";
  }
  return oss.str();
}

std::string cache_size_per_pcore() {
  return " (per P-core)";
}

std::string cache_size_per_pcore_cluster() {
  return " (per P-core cluster)";
}

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
  oss << "  Custom Cache: " << latency_ns << " ns (Buffer size: " << buffer_size << " B)";
  return oss.str();
}

std::string results_cache_latency_custom_ns_kb(double latency_ns, double buffer_size_kb) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);
  oss << "  Custom Cache: " << latency_ns << " ns (Buffer size: " << buffer_size_kb << " KB)";
  return oss.str();
}

std::string results_cache_latency_custom_ns_mb(double latency_ns, double buffer_size_mb) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);
  oss << "  Custom Cache: " << latency_ns << " ns (Buffer size: " << buffer_size_mb << " MB)";
  return oss.str();
}

std::string results_cache_latency_l1_ns(double latency_ns, size_t buffer_size) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);
  oss << "  L1 Cache: " << latency_ns << " ns (Buffer size: " << buffer_size << " B)";
  return oss.str();
}

std::string results_cache_latency_l1_ns_kb(double latency_ns, double buffer_size_kb) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);
  oss << "  L1 Cache: " << latency_ns << " ns (Buffer size: " << buffer_size_kb << " KB)";
  return oss.str();
}

std::string results_cache_latency_l1_ns_mb(double latency_ns, double buffer_size_mb) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);
  oss << "  L1 Cache: " << latency_ns << " ns (Buffer size: " << buffer_size_mb << " MB)";
  return oss.str();
}

std::string results_cache_latency_l2_ns(double latency_ns, size_t buffer_size) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);
  oss << "  L2 Cache: " << latency_ns << " ns (Buffer size: " << buffer_size << " B)";
  return oss.str();
}

std::string results_cache_latency_l2_ns_kb(double latency_ns, double buffer_size_kb) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);
  oss << "  L2 Cache: " << latency_ns << " ns (Buffer size: " << buffer_size_kb << " KB)";
  return oss.str();
}

std::string results_cache_latency_l2_ns_mb(double latency_ns, double buffer_size_mb) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);
  oss << "  L2 Cache: " << latency_ns << " ns (Buffer size: " << buffer_size_mb << " MB)";
  return oss.str();
}

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


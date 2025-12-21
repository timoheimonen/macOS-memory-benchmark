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
#ifndef MESSAGES_H
#define MESSAGES_H

#include <string>

// Centralized namespace for all text messages, error messages, and output strings
namespace Messages {

// --- Error Messages ---
const std::string& error_prefix();
std::string error_missing_value(const std::string& option);
std::string error_invalid_value(const std::string& option, const std::string& value, const std::string& reason);
std::string error_unknown_option(const std::string& option);
std::string error_buffer_size_calculation(unsigned long size_mb);
std::string error_buffer_size_too_small(size_t size_bytes);
std::string error_cache_size_invalid(long long min_kb, long long max_kb, long long max_mb);
const std::string& error_iterations_invalid();
const std::string& error_buffersize_invalid();
const std::string& error_count_invalid();
const std::string& error_latency_samples_invalid();
std::string error_mmap_failed(const std::string& buffer_name);
std::string error_madvise_failed(const std::string& buffer_name);
std::string error_benchmark_tests(const std::string& error);
std::string error_benchmark_loop(int loop, const std::string& error);
std::string error_json_parse_failed(const std::string& error_details);
std::string error_json_file_read_failed(const std::string& file_path, const std::string& error_details);
std::string error_file_write_failed(const std::string& file_path, const std::string& error_details);
std::string error_file_permission_denied(const std::string& file_path);
std::string error_file_directory_creation_failed(const std::string& dir_path, const std::string& error_details);
std::string error_stride_too_small();
std::string error_stride_too_large(size_t stride, size_t buffer_size);
std::string error_indices_empty();
std::string error_index_out_of_bounds(size_t index, size_t index_value, size_t buffer_size);
std::string error_index_not_aligned(size_t index, size_t index_value);

// --- Warning Messages ---
const std::string& warning_cannot_get_memory();
std::string warning_buffer_size_exceeds_limit(unsigned long requested_mb, unsigned long limit_mb);
std::string warning_qos_failed(int code);
std::string warning_stride_not_aligned(size_t stride);
std::string warning_qos_failed_worker_thread(int code);
std::string warning_madvise_random_failed(const std::string& buffer_name, const std::string& error_msg);

// --- Info Messages ---
std::string info_setting_max_fallback(unsigned long max_mb);
std::string info_calculated_max_less_than_min(unsigned long max_mb, unsigned long min_mb);
std::string info_custom_cache_rounded_up(unsigned long original_kb, unsigned long rounded_kb);

// --- Main Program Messages ---
const std::string& msg_running_benchmarks();
std::string msg_done_total_time(double total_time_sec);

// --- Usage/Help Messages ---
std::string usage_header(const std::string& version);
std::string usage_options(const std::string& prog_name);
std::string usage_example(const std::string& prog_name);

// --- Configuration Output Messages ---
std::string config_header(const std::string& version);
std::string config_copyright();
std::string config_license();
std::string config_buffer_size(double buffer_size_mib, unsigned long buffer_size_mb);
std::string config_total_allocation(double total_mib);
std::string config_iterations(int iterations);
std::string config_loop_count(int loop_count);
std::string config_non_cacheable(bool use_non_cacheable);
std::string config_processor_name(const std::string& cpu_name);
std::string config_processor_name_error();
std::string config_performance_cores(int perf_cores);
std::string config_efficiency_cores(int eff_cores);
std::string config_total_cores(int num_threads);

// --- Cache Info Messages ---
std::string cache_info_header();
std::string cache_size_custom(size_t size_bytes);
std::string cache_size_l1(size_t size_bytes);
std::string cache_size_l2(size_t size_bytes);
std::string cache_size_per_pcore();
std::string cache_size_per_pcore_cluster();

// --- Results Output Messages ---
std::string results_loop_header(int loop);
std::string results_main_memory_bandwidth(int num_threads);
std::string results_read_bandwidth(double bw_gb_s, double total_time);
std::string results_write_bandwidth(double bw_gb_s, double total_time);
std::string results_copy_bandwidth(double bw_gb_s, double total_time);
std::string results_main_memory_latency();
std::string results_latency_total_time(double total_time_sec);
std::string results_latency_average(double latency_ns);
std::string results_cache_bandwidth();
std::string results_cache_latency();
std::string results_custom_cache();
std::string results_l1_cache();
std::string results_l2_cache();
std::string results_cache_read_bandwidth(double bw_gb_s);
std::string results_cache_write_bandwidth(double bw_gb_s);
std::string results_cache_copy_bandwidth(double bw_gb_s);
std::string results_buffer_size_bytes(size_t buffer_size);
std::string results_buffer_size_kb(double buffer_size_kb);
std::string results_buffer_size_mb(double buffer_size_mb);
std::string results_cache_latency_ns(double latency_ns);
std::string results_separator();
std::string results_cache_latency_custom_ns(double latency_ns, size_t buffer_size);
std::string results_cache_latency_custom_ns_kb(double latency_ns, double buffer_size_kb);
std::string results_cache_latency_custom_ns_mb(double latency_ns, double buffer_size_mb);
std::string results_cache_latency_l1_ns(double latency_ns, size_t buffer_size);
std::string results_cache_latency_l1_ns_kb(double latency_ns, double buffer_size_kb);
std::string results_cache_latency_l1_ns_mb(double latency_ns, double buffer_size_mb);
std::string results_cache_latency_l2_ns(double latency_ns, size_t buffer_size);
std::string results_cache_latency_l2_ns_kb(double latency_ns, double buffer_size_kb);
std::string results_cache_latency_l2_ns_mb(double latency_ns, double buffer_size_mb);

// --- Statistics Messages ---
std::string statistics_header(int loop_count);
std::string statistics_metric_name(const std::string& metric_name);
std::string statistics_average(double value, int precision = 3);
std::string statistics_median_p50(double value, int precision = 3);
std::string statistics_p90(double value, int precision = 3);
std::string statistics_p95(double value, int precision = 3);
std::string statistics_p99(double value, int precision = 3);
std::string statistics_stddev(double value, int precision = 3);
std::string statistics_min(double value, int precision = 3);
std::string statistics_max(double value, int precision = 3);
std::string statistics_cache_bandwidth_header(const std::string& cache_name);
std::string statistics_cache_read();
std::string statistics_cache_write();
std::string statistics_cache_copy();
std::string statistics_cache_latency_header();
std::string statistics_cache_latency_name(const std::string& cache_name);
std::string statistics_median_p50_from_samples(double value, size_t sample_count, int precision = 2);
std::string statistics_main_memory_latency_header();
std::string statistics_footer();

} // namespace Messages

#endif // MESSAGES_H


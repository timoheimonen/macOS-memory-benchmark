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
 * @file messages_api.h
 * @brief Centralized message namespace for all application output strings
 *
 * This file declares the Messages namespace containing all user-facing text messages
 * used throughout the application. It provides a centralized location for managing:
 * - Error messages with consistent formatting and prefixes
 * - Warning messages for non-critical issues
 * - Informational messages for user guidance
 * - Configuration and result output formatting
 * - Statistical analysis output strings
 * - Pattern benchmark descriptive labels
 *
 * Centralizing messages improves maintainability, enables consistent error reporting,
 * and facilitates future internationalization efforts.
 *
 * @note All messages support dynamic content through function parameters
 * @note Constant messages are returned by reference for efficiency
 */

#ifndef MESSAGES_MESSAGES_API_H
#define MESSAGES_MESSAGES_API_H

#include <cstdint>
#include <string>

/**
 * @brief Centralized namespace for all text messages, error messages, and output strings
 *
 * The Messages namespace provides functions that return formatted strings for every
 * user-facing message in the application. This design ensures consistency in output
 * formatting and makes it easy to modify messages across the entire codebase.
 */
namespace Messages {

// --- Error Messages ---
const std::string& error_prefix();
std::string error_missing_value(const std::string& option);
std::string error_invalid_value(const std::string& option, const std::string& value, const std::string& reason);
std::string error_unknown_option(const std::string& option);
std::string error_duplicate_option(const std::string& option);
std::string error_buffer_size_calculation(unsigned long size_mb);
std::string error_buffer_size_too_small(size_t size_bytes);
std::string error_cache_size_invalid(long long min_kb, long long max_kb, long long max_mb);
std::string error_iterations_invalid(long long value, long long min_val, long long max_val);
std::string error_buffersize_invalid(long long value, unsigned long max_val);
std::string error_count_invalid(long long value, long long min_val, long long max_val);
std::string error_latency_samples_invalid(long long value, long long min_val, long long max_val);
std::string error_latency_stride_invalid(long long value, long long min_val, long long max_val);
std::string error_latency_stride_alignment(size_t value_bytes, size_t alignment_bytes);
std::string error_analyze_tlb_stride_exceeds_page(size_t stride_bytes, size_t page_size_bytes);
std::string error_latency_tlb_locality_invalid(long long value, long long max_val);
std::string error_latency_chain_mode_invalid();
std::string error_latency_chain_mode_requires_locality(const std::string& mode_name);
const std::string& error_analyze_tlb_global_random_unsupported();
std::string error_latency_tlb_locality_page_multiple(size_t value_kb, size_t page_size_kb);
std::string error_latency_tlb_locality_too_small_for_stride(size_t locality_bytes, size_t stride_bytes);
std::string error_threads_invalid(long long value, long long min_val, long long max_val);
const std::string& error_analyze_tlb_must_be_used_alone();
const std::string& error_seed_requires_patterns();
const std::string& error_seed_requires_benchmark_or_patterns();
std::string error_duplicate_sweep_parameter(const std::string& parameter_name);
const std::string& error_analyze_core_to_core_must_be_used_alone();
const std::string& error_core_to_core_timer_creation_failed();
const std::string& error_tlb_analysis_insufficient_memory();
const std::string& error_tlb_analysis_timer_creation_failed();
const std::string& error_timer_creation_failed();
std::string error_tlb_analysis_invalid_measurement(size_t locality_kb, int loop_number);
std::string error_tlb_chain_setup_failed(size_t locality_kb,
                                         const std::string& layout,
                                         const std::string& build_status,
                                         const std::string& validation_status);
std::string error_mmap_failed(const std::string& buffer_name);
std::string error_madvise_failed(const std::string& buffer_name);
std::string error_munmap_failed();
std::string error_sysctlbyname_failed(const std::string& operation, const std::string& key);
std::string error_mach_timebase_info_failed(const std::string& error_details);
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
std::string error_buffer_too_small_strided(size_t min_bytes);
std::string error_no_iterations_strided();
std::string error_buffer_size_zero(const std::string& buffer_name);
const std::string& error_main_buffer_size_zero();
const std::string& error_buffer_size_overflow_calculation();
const std::string& error_total_memory_overflow();
std::string error_total_memory_exceeds_limit(unsigned long total_mb, unsigned long max_mb);
const std::string& error_main_buffers_not_allocated();
const std::string& error_custom_buffer_not_allocated();
const std::string& error_l1_buffer_not_allocated();
const std::string& error_l2_buffer_not_allocated();
const std::string& error_custom_bandwidth_buffers_not_allocated();
const std::string& error_l1_bandwidth_buffers_not_allocated();
const std::string& error_l2_bandwidth_buffers_not_allocated();
const std::string& error_buffer_pointer_null_latency_chain();
const std::string& error_stride_zero_latency_chain();
std::string error_buffer_stride_invalid_latency_chain(size_t num_pointers, size_t buffer_size, size_t stride);
const std::string& error_buffer_too_small_for_pointers();
std::string error_offset_exceeds_bounds(size_t offset, size_t max_offset);
std::string error_next_pointer_offset_exceeds_bounds(size_t offset, size_t max_offset);
const std::string& error_source_buffer_null();
const std::string& error_destination_buffer_null();
const std::string& error_buffer_size_zero_generic();
const std::string& error_calculated_custom_buffer_size_zero();
const std::string& error_l1_cache_size_overflow();
const std::string& error_l2_cache_size_overflow();
const std::string& error_calculated_l1_buffer_size_zero();
const std::string& error_calculated_l2_buffer_size_zero();
const std::string& error_latency_access_count_overflow();
const std::string& error_latency_access_count_negative();
const std::string& error_incompatible_flags();
const std::string& error_only_flags_with_patterns();
const std::string& error_only_bandwidth_with_cache_size();
const std::string& error_only_bandwidth_with_latency_samples();
const std::string& error_buffersize_zero_requires_only_latency();
const std::string& error_cache_size_zero_requires_only_latency();
const std::string& error_only_latency_requires_latency_target();
const std::string& error_only_latency_with_buffersize();
const std::string& error_only_latency_with_iterations();
std::string error_mutually_exclusive_modes(const std::string& mode1, const std::string& mode2);
const std::string& error_only_flags_require_benchmark();
const std::string& error_sweep_requires_parameter();
std::string error_sweep_too_many_runs(size_t run_count, size_t max_runs);
std::string error_sweep_parameter_not_allowed(const std::string& parameter_name, const std::string& mode_name);
const std::string& error_sweep_requires_output();
std::string error_sweep_temp_json_parse_failed(const std::string& file_path, const std::string& error_details);

// --- Warning Messages ---
const std::string& warning_prefix();
const std::string& warning_cannot_get_memory();
std::string warning_buffer_size_exceeds_limit(unsigned long requested_mb, unsigned long limit_mb);
std::string warning_qos_failed(int code);
std::string warning_stride_not_aligned(size_t stride);
std::string warning_qos_failed_worker_thread(int code);
std::string warning_madvise_random_failed(const std::string& buffer_name, const std::string& error_msg);
std::string warning_tlb_mlock_failed(int error_code,
                                     const std::string& error_message);
const std::string& warning_core_count_detection_failed();
const std::string& warning_mach_host_self_failed();
std::string warning_host_page_size_failed(const std::string& error_details);
std::string warning_host_statistics64_failed(const std::string& error_details);
const std::string& warning_l1_cache_size_detection_failed();
const std::string& warning_l2_cache_size_detection_failed_m1();
const std::string& warning_l2_cache_size_detection_failed_m2_m3_m4_m5();
const std::string& warning_l2_cache_size_detection_failed_generic();
std::string warning_threads_capped(int requested, int max_cores);

// --- Info Messages ---
std::string info_setting_max_fallback(unsigned long max_mb);
std::string info_calculated_max_less_than_min(unsigned long max_mb, unsigned long min_mb);
std::string info_custom_cache_rounded_up(unsigned long original_kb, unsigned long rounded_kb);

// --- Main Program Messages ---
const std::string& msg_running_benchmarks();
std::string msg_done_total_time(double total_time_sec);
const std::string& msg_running_pattern_benchmarks();
std::string msg_pattern_benchmark_loop_completed(int current_loop, int total_loops);
std::string msg_results_saved_to(const std::string& file_path);
const std::string& msg_running_tlb_analysis();
const std::string& msg_running_core_to_core_analysis();
const std::string& msg_interrupted_by_user();
std::string msg_running_sweep(size_t run_count);
std::string msg_sweep_run_progress(size_t current_run, size_t total_runs);
std::string msg_core_to_core_scenario_progress(size_t current_loop,
                                               size_t total_loops,
                                               const std::string& scenario_name);
std::string msg_tlb_analysis_refinement_start(size_t point_count);
std::string msg_tlb_analysis_validation_start(size_t point_count);

// --- Core-to-Core Report Messages ---
const std::string& report_core_to_core_header();
const std::string& report_core_to_core_scheduler_note();
std::string report_core_to_core_cpu(const std::string& cpu_name);
std::string report_core_to_core_cores(int perf_cores, int eff_cores);
std::string report_core_to_core_loop_config(int loop_count,
                                            int sample_count,
                                            size_t headline_round_trips,
                                            size_t sample_window_round_trips);
std::string report_core_to_core_scenario_title(const std::string& scenario_name);
std::string report_core_to_core_round_trip(double round_trip_ns);
std::string report_core_to_core_one_way_estimate(double one_way_ns);
std::string report_core_to_core_samples(size_t sample_count);
std::string report_core_to_core_hint_status(const std::string& thread_role,
                                            bool qos_applied,
                                            int qos_code,
                                            bool affinity_requested,
                                            bool affinity_applied,
                                            int affinity_code,
                                            int affinity_tag);

// --- TLB Analysis Report Messages ---
const std::string& report_tlb_header();
const std::string& report_tlb_settings_header();
std::string report_tlb_run_summary(const std::string& cpu_name,
                                   size_t page_size_bytes,
                                   size_t stride_bytes,
                                   const std::string& profile_name,
                                   const std::string& requested_chain_mode,
                                   const std::string& effective_chain_mode,
                                   uint64_t seed,
                                   bool user_specified_seed);
std::string report_tlb_resource_summary(size_t buffer_mb,
                                        bool buffer_locked,
                                        bool qos_requested,
                                        bool qos_applied,
                                        int qos_code,
                                        size_t memory_budget_mb,
                                        size_t estimated_peak_memory_bytes);
std::string report_tlb_sweep_plan(size_t start_locality_bytes,
                                  size_t end_locality_bytes,
                                  size_t point_count,
                                  bool large_comparison_enabled,
                                  size_t comparison_locality_bytes,
                                  size_t required_buffer_mb,
                                  size_t selected_buffer_mb);
const std::string& report_tlb_sweep_legend();
const std::string& report_tlb_quick_profile_note();
std::string report_tlb_paired_locality_progress(size_t current_index,
                                                size_t total_count,
                                                size_t locality_bytes,
                                                double spread_p50_ns,
                                                double packed_p50_ns,
                                                double translation_delta_p50_ns,
                                                size_t active_cache_line_footprint_bytes,
                                                bool short_cycle_diagnostic);
std::string report_tlb_work_estimate(const std::string& pass_name,
                                     size_t point_count,
                                     size_t min_rounds,
                                     size_t max_rounds,
                                     double estimated_min_duration_sec,
                                     double estimated_max_duration_sec);
std::string report_tlb_pass_completion(const std::string& pass_name,
                                       size_t rounds_completed,
                                       const std::string& completion_reason);
std::string report_tlb_fine_sweep(size_t added_points, size_t total_points);
std::string report_tlb_analysis_status(const std::string& status,
                                       size_t planned_points,
                                       size_t measured_points,
                                       bool conclusions_valid);
const std::string& report_tlb_l1_section();
const std::string& report_tlb_l2_section();
const std::string& report_tlb_private_cache_section();
std::string report_tlb_boundary_kb(size_t boundary_kb);
std::string report_tlb_inferred_size_entries(size_t entries);
std::string report_tlb_inferred_reach_entries(size_t entries);
std::string report_tlb_inferred_entries_range(size_t min_entries, size_t max_entries);
const std::string& report_tlb_private_cache_overlap();
std::string report_tlb_confidence(const std::string& confidence, double step_ns, double step_percent);
std::string report_tlb_statistical_confidence(const std::string& confidence,
                                              double effect_ns,
                                              double discovery_ci_lower_ns,
                                              double discovery_ci_upper_ns,
                                              double validation_ci_lower_ns,
                                              double validation_ci_upper_ns);
std::string report_tlb_private_cache_candidate(bool strong_private_cache_candidate);
std::string report_tlb_private_cache_interference(bool elevated_risk, size_t locality_kb);
std::string report_tlb_private_cache_l1_distance(size_t distance_kb, size_t distance_pages);
std::string report_tlb_large_locality_paired_comparison(
    size_t locality_bytes,
    double spread_p50_ns,
    double packed_p50_ns,
    double translation_delta_p50_ns,
    size_t spread_actual_pages,
    size_t packed_actual_pages,
    size_t unique_cache_lines,
    size_t active_cache_line_footprint_bytes);
std::string report_tlb_large_locality_paired_unavailable(size_t required_buffer_mb,
                                                         size_t selected_buffer_mb);
const std::string& report_tlb_large_locality_paired_interrupted();
std::string report_tlb_conclusions_unavailable(const std::string& status);
const std::string& report_tlb_not_detected();

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
std::string config_pattern_iterations_auto(double target_seconds,
                                           double min_seconds,
                                           double max_seconds);
std::string config_loop_count(int loop_count);
std::string config_non_cacheable(bool use_non_cacheable);
std::string config_latency_stride(size_t stride_bytes);
std::string config_latency_chain_mode(const std::string& mode_name);
std::string config_latency_tlb_locality(size_t locality_bytes);
std::string config_processor_name(const std::string& cpu_name);
std::string config_processor_name_error();
std::string config_performance_cores(int perf_cores);
std::string config_efficiency_cores(int eff_cores);
std::string config_total_cores(int num_threads);
std::string config_benchmark_threads(int num_threads);

// --- Cache Info Messages ---
std::string cache_info_header();
std::string cache_size_custom(size_t size_bytes);
std::string cache_size_custom_disabled();
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
std::string results_latency_average(double latency_ns, size_t locality_bytes);
std::string results_latency_tlb_hit(double latency_ns);
std::string results_latency_tlb_miss(double latency_ns);
std::string results_latency_page_walk_penalty(double penalty_ns);
std::string results_cache_bandwidth(int num_threads);
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
std::string results_measurement_unavailable(const std::string& label,
                                            const std::string& status,
                                            const std::string& reason);

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
std::string statistics_pattern_bandwidth_header(const std::string& pattern_name);
std::string statistics_coefficient_of_variation(double value, int precision = 1);
std::string statistics_cache_read();
std::string statistics_cache_write();
std::string statistics_cache_copy();
std::string statistics_cache_latency_header();
std::string statistics_cache_latency_name(const std::string& cache_name);
std::string statistics_median_p50_from_samples(double value, size_t sample_count, int precision = 2);
std::string statistics_main_memory_latency_header();
std::string statistics_tlb_hit_latency_metric_name();
std::string statistics_tlb_miss_latency_metric_name();
std::string statistics_page_walk_penalty_metric_name();
std::string statistics_footer();

// --- Pattern Benchmark Messages ---
const std::string& pattern_sequential_forward();
const std::string& pattern_sequential_reverse();
std::string pattern_strided(const std::string& stride_name);
const std::string& pattern_random_uniform();
const std::string& pattern_cache_line_64b();
const std::string& pattern_page_4096b();
const std::string& pattern_page_16384b();
const std::string& pattern_superpage_2mb();
const std::string& pattern_separator();
const std::string& pattern_read_label();
const std::string& pattern_write_label();
const std::string& pattern_copy_label();
const std::string& pattern_bandwidth_unit();
const std::string& pattern_bandwidth_unit_newline();
std::string pattern_measurement_unavailable(const std::string& status,
                                            const std::string& reason);
std::string warning_pattern_measurement_noisy(const std::string& metric,
                                              double cv_pct,
                                              double threshold_pct);
const std::string& pattern_reason_measurement_not_completed();
const std::string& pattern_reason_timer_creation_failed();
const std::string& pattern_reason_calibration_or_accounting_failed();
const std::string& pattern_reason_no_valid_random_workload();
const std::string& pattern_reason_stride_transition_unavailable();
const std::string& pattern_reason_copy_accounting_overflow();
const std::string& pattern_reason_invalid_strided_timing();
const std::string& pattern_reason_work_plan_byte_overflow();
const std::string& pattern_reason_invalid_work_plan_parameters();
const std::string& pattern_reason_stride_access_sum_overflow();
const std::string& pattern_reason_buffer_lacks_two_strided_accesses();
const std::string& pattern_reason_no_valid_strided_worker_partition();
const std::string& pattern_reason_work_plan_pass_limit();
const std::string& pattern_reason_work_plan_total_overflow();

} // namespace Messages

#endif // MESSAGES_MESSAGES_API_H

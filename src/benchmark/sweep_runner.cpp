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
 * @file sweep_runner.cpp
 * @brief Multi-configuration benchmark sweep runner implementation.
 */

#include "benchmark/sweep_runner.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include <mach/mach.h>
#include <pthread/qos.h>

#include "benchmark/benchmark_runner.h"
#include "benchmark/tlb_analysis.h"
#include "core/config/config.h"
#include "core/config/constants.h"
#include "core/config/version.h"
#include "core/memory/buffer_allocator.h"
#include "core/memory/buffer_initializer.h"
#include "core/memory/buffer_manager.h"
#include "core/signal/signal_handler.h"
#include "core/system/system_info.h"
#include "core/timing/timer.h"
#include "output/console/messages/messages_api.h"
#include "output/console/output_printer.h"
#include "output/console/statistics.h"
#include "output/json/json_output/json_output_api.h"
#include "pattern_benchmark/pattern_benchmark.h"
#include "third_party/nlohmann/json.hpp"
#include "utils/json_utils.h"

namespace {

struct SweepAssignment {
  const SweepSpec* spec = nullptr;
  const SweepValue* value = nullptr;
};

std::string build_utc_timestamp() {
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);
  std::tm utc_time;
  gmtime_r(&time_t, &utc_time);
  std::ostringstream timestamp_str;
  timestamp_str << std::put_time(&utc_time, "%Y-%m-%dT%H:%M:%SZ");
  return timestamp_str.str();
}

std::string base_mode_name(const BenchmarkConfig& config) {
  if (config.analyze_tlb) {
    return Constants::TLB_ANALYSIS_JSON_MODE_NAME;
  }
  if (config.run_patterns) {
    return Constants::PATTERNS_JSON_MODE_NAME;
  }
  return Constants::BENCHMARK_JSON_MODE_NAME;
}

nlohmann::ordered_json build_sweep_parameters_json(const BenchmarkConfig& config) {
  nlohmann::ordered_json params;
  for (const SweepSpec& spec : config.sweep_specs) {
    nlohmann::ordered_json values = nlohmann::ordered_json::array();
    for (const SweepValue& value : spec.values) {
      if (spec.parameter == SweepParameter::LatencyChainMode ||
          spec.parameter == SweepParameter::TlbDensity) {
        values.push_back(value.raw_value);
      } else {
        values.push_back(value.integer_value);
      }
    }
    params[spec.parameter_name] = values;
  }
  return params;
}

nlohmann::ordered_json build_assignment_json(const std::vector<SweepAssignment>& assignments) {
  nlohmann::ordered_json params;
  for (const SweepAssignment& assignment : assignments) {
    const SweepSpec& spec = *assignment.spec;
    const SweepValue& value = *assignment.value;
    if (spec.parameter == SweepParameter::LatencyChainMode ||
        spec.parameter == SweepParameter::TlbDensity) {
      params[spec.parameter_name] = value.raw_value;
    } else {
      params[spec.parameter_name] = value.integer_value;
    }
  }
  return params;
}

void apply_assignment(BenchmarkConfig& config, const SweepAssignment& assignment) {
  const SweepSpec& spec = *assignment.spec;
  const SweepValue& value = *assignment.value;

  switch (spec.parameter) {
    case SweepParameter::BufferSizeMb:
      config.buffer_size_mb = static_cast<unsigned long>(value.integer_value);
      config.user_specified_buffersize = true;
      break;
    case SweepParameter::CacheSizeKb:
      config.custom_cache_size_kb_ll = value.integer_value;
      config.use_custom_cache_size = true;
      config.custom_cache_size_bytes =
          static_cast<size_t>(value.integer_value) * Constants::BYTES_PER_KB;
      break;
    case SweepParameter::Threads: {
      const int requested_threads = static_cast<int>(value.integer_value);
      const int max_cores = get_total_logical_cores();
      config.num_threads = (max_cores > 0) ? std::min(requested_threads, max_cores) : requested_threads;
      config.user_specified_threads = true;
      break;
    }
    case SweepParameter::LatencyTlbLocalityKb:
      config.latency_tlb_locality_bytes =
          static_cast<size_t>(value.integer_value) * Constants::BYTES_PER_KB;
      config.user_specified_latency_tlb_locality = true;
      break;
    case SweepParameter::LatencyStrideBytes:
      config.latency_stride_bytes = static_cast<size_t>(value.integer_value);
      config.user_specified_latency_stride = true;
      break;
    case SweepParameter::LatencyChainMode:
      config.latency_chain_mode = value.latency_chain_mode;
      config.user_specified_latency_chain_mode = true;
      break;
    case SweepParameter::TlbDensity:
      config.tlb_sweep_density = value.tlb_sweep_density;
      break;
  }
}

BenchmarkConfig build_run_config(const BenchmarkConfig& base_config,
                                 const std::vector<SweepAssignment>& assignments) {
  BenchmarkConfig run_config = base_config;
  run_config.run_sweep = false;
  run_config.sweep_specs.clear();
  run_config.output_file.clear();

  for (const SweepAssignment& assignment : assignments) {
    apply_assignment(run_config, assignment);
  }
  return run_config;
}

void append_assignments_recursive(const BenchmarkConfig& config,
                                  size_t spec_index,
                                  std::vector<SweepAssignment>& current,
                                  std::vector<std::vector<SweepAssignment>>& out_assignments) {
  if (spec_index == config.sweep_specs.size()) {
    out_assignments.push_back(current);
    return;
  }

  const SweepSpec& spec = config.sweep_specs[spec_index];
  for (const SweepValue& value : spec.values) {
    current.push_back(SweepAssignment{&spec, &value});
    append_assignments_recursive(config, spec_index + 1, current, out_assignments);
    current.pop_back();
  }
}

std::vector<std::vector<SweepAssignment>> build_sweep_assignments(const BenchmarkConfig& config) {
  std::vector<std::vector<SweepAssignment>> assignments;
  assignments.reserve(calculate_sweep_run_count(config));
  std::vector<SweepAssignment> current;
  append_assignments_recursive(config, 0, current, assignments);
  return assignments;
}

int finalize_run_config(BenchmarkConfig& config) {
  if (validate_config(config) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  calculate_buffer_sizes(config);
  calculate_access_counts(config);
  size_t peak_allocation_bytes = 0;
  return calculate_total_allocation_bytes(config, peak_allocation_bytes);
}

int run_standard_sweep_point(BenchmarkConfig& run_config, nlohmann::ordered_json& result_json) {
  auto timer_opt = HighResTimer::create();
  if (!timer_opt) {
    std::cerr << Messages::error_prefix() << Messages::error_timer_creation_failed() << std::endl;
    return EXIT_FAILURE;
  }
  auto& timer = *timer_opt;
  timer.start();

  if (finalize_run_config(run_config) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }

  BenchmarkBuffers buffers;
  BenchmarkStatistics stats;
  if (run_all_benchmarks(buffers, run_config, stats) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }

  const double elapsed_sec = timer.stop();
  print_statistics(run_config.loop_count, stats.all_read_bw_gb_s, stats.all_write_bw_gb_s, stats.all_copy_bw_gb_s,
                   stats.all_l1_latency_ns, stats.all_l2_latency_ns,
                   stats.all_l1_read_bw_gb_s, stats.all_l1_write_bw_gb_s, stats.all_l1_copy_bw_gb_s,
                   stats.all_l2_read_bw_gb_s, stats.all_l2_write_bw_gb_s, stats.all_l2_copy_bw_gb_s,
                   stats.all_average_latency_ns,
                   stats.all_tlb_hit_latency_ns,
                   stats.all_tlb_miss_latency_ns,
                   stats.all_page_walk_penalty_ns,
                   run_config.use_custom_cache_size,
                   stats.all_custom_latency_ns, stats.all_custom_read_bw_gb_s,
                   stats.all_custom_write_bw_gb_s, stats.all_custom_copy_bw_gb_s,
                   stats.all_main_mem_latency_samples,
                   stats.all_l1_latency_samples,
                   stats.all_l2_latency_samples,
                   stats.all_custom_latency_samples,
                   run_config.only_bandwidth,
                   run_config.only_latency);
  result_json = build_results_json(run_config, stats, elapsed_sec);
  return EXIT_SUCCESS;
}

int run_pattern_sweep_point(BenchmarkConfig& run_config, nlohmann::ordered_json& result_json) {
  auto timer_opt = HighResTimer::create();
  if (!timer_opt) {
    std::cerr << Messages::error_prefix() << Messages::error_timer_creation_failed() << std::endl;
    return EXIT_FAILURE;
  }
  auto& timer = *timer_opt;
  timer.start();

  if (finalize_run_config(run_config) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }

  BenchmarkBuffers buffers;
  if (allocate_all_buffers(run_config, buffers) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  if (initialize_all_buffers(buffers, run_config) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }

  PatternStatistics stats;
  if (run_all_pattern_benchmarks(buffers, run_config, stats) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }

  if (run_config.loop_count == 1) {
    print_pattern_results(extract_pattern_results_at(stats, 0));
  } else if (!stats.all_forward_read_bw.empty()) {
    print_pattern_results(extract_pattern_results_at(stats, stats.all_forward_read_bw.size() - 1));
    print_pattern_statistics(run_config.loop_count, stats);
  }

  const double elapsed_sec = timer.stop();
  result_json = build_pattern_results_json(run_config, stats, elapsed_sec);
  return EXIT_SUCCESS;
}

int run_tlb_sweep_point(BenchmarkConfig& run_config,
                        size_t run_index,
                        nlohmann::ordered_json& result_json) {
  const std::filesystem::path temp_path =
      std::filesystem::temp_directory_path() /
      ("memory_benchmark_sweep_" + std::to_string(static_cast<long long>(getpid())) +
       "_" + std::to_string(run_index) + ".json");
  run_config.output_file = temp_path.string();

  if (run_tlb_analysis(run_config) != EXIT_SUCCESS) {
    std::error_code ignored;
    std::filesystem::remove(temp_path, ignored);
    return EXIT_FAILURE;
  }

  nlohmann::json parsed_json;
  std::string error_message;
  if (!parse_json_from_file(temp_path.string(), parsed_json, error_message)) {
    std::cerr << Messages::error_prefix()
              << Messages::error_sweep_temp_json_parse_failed(temp_path.string(), error_message)
              << std::endl;
    std::error_code ignored;
    std::filesystem::remove(temp_path, ignored);
    return EXIT_FAILURE;
  }

  std::error_code ignored;
  std::filesystem::remove(temp_path, ignored);
  result_json = parsed_json;
  return EXIT_SUCCESS;
}

int run_sweep_point(BenchmarkConfig& run_config,
                    size_t run_index,
                    nlohmann::ordered_json& result_json) {
  if (run_config.analyze_tlb) {
    return run_tlb_sweep_point(run_config, run_index, result_json);
  }
  if (run_config.run_patterns) {
    return run_pattern_sweep_point(run_config, result_json);
  }
  return run_standard_sweep_point(run_config, result_json);
}

}  // namespace

size_t calculate_sweep_run_count(const BenchmarkConfig& config) {
  size_t run_count = 1;
  for (const SweepSpec& spec : config.sweep_specs) {
    if (spec.values.empty()) {
      return 0;
    }
    if (run_count > std::numeric_limits<size_t>::max() / spec.values.size()) {
      return std::numeric_limits<size_t>::max();
    }
    run_count *= spec.values.size();
  }
  return run_count;
}

int run_sweep_mode(const BenchmarkConfig& base_config) {
  const size_t run_count = calculate_sweep_run_count(base_config);
  std::cout << Messages::usage_header(SOFTVERSION);
  std::cout << Messages::msg_running_sweep(run_count) << std::endl;

  kern_return_t qos_ret = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  if (qos_ret != KERN_SUCCESS) {
    std::cerr << Messages::warning_prefix() << Messages::warning_qos_failed(qos_ret) << std::endl;
  }

  block_benchmark_signals();

  auto total_timer_opt = HighResTimer::create();
  if (!total_timer_opt) {
    std::cerr << Messages::error_prefix() << Messages::error_timer_creation_failed() << std::endl;
    restore_signal_mask();
    return EXIT_FAILURE;
  }
  auto& total_timer = *total_timer_opt;
  total_timer.start();

  nlohmann::ordered_json output_json;
  output_json[JsonKeys::CONFIGURATION] = {
      {JsonKeys::MODE, Constants::SWEEP_JSON_MODE_NAME},
      {"base_mode", base_mode_name(base_config)},
      {"run_count", run_count},
      {"sweep_max_runs", base_config.sweep_max_runs},
      {"sweep_parameters", build_sweep_parameters_json(base_config)}};

  nlohmann::ordered_json runs_json = nlohmann::ordered_json::array();
  const std::vector<std::vector<SweepAssignment>> assignments = build_sweep_assignments(base_config);
  for (size_t i = 0; i < assignments.size(); ++i) {
    if (signal_received()) {
      std::cout << Messages::msg_interrupted_by_user() << std::endl;
      break;
    }

    std::cout << Messages::msg_sweep_run_progress(i + 1, assignments.size()) << std::endl;
    BenchmarkConfig run_config = build_run_config(base_config, assignments[i]);
    nlohmann::ordered_json result_json;
    if (run_sweep_point(run_config, i, result_json) != EXIT_SUCCESS) {
      restore_signal_mask();
      return EXIT_FAILURE;
    }

    nlohmann::ordered_json run_json;
    run_json["index"] = i;
    run_json["parameters"] = build_assignment_json(assignments[i]);
    run_json["result"] = result_json;
    runs_json.push_back(run_json);
  }

  const double total_elapsed_sec = total_timer.stop();
  output_json["runs"] = runs_json;
  output_json[JsonKeys::EXECUTION_TIME_SEC] = total_elapsed_sec;
  output_json[JsonKeys::TIMESTAMP] = build_utc_timestamp();
  output_json[JsonKeys::VERSION] = SOFTVERSION;

  std::filesystem::path file_path(base_config.output_file);
  if (file_path.is_relative()) {
    file_path = std::filesystem::current_path() / file_path;
  }

  restore_signal_mask();
  const int write_result = write_json_to_file(file_path, output_json);
  if (write_result == EXIT_SUCCESS) {
    std::cout << Messages::msg_done_total_time(total_elapsed_sec) << std::endl;
  }
  return write_result;
}

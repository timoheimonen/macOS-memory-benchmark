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
#include <filesystem>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

#include <mach/mach.h>
#include <pthread/qos.h>

#include "benchmark/benchmark_runner.h"
#include "benchmark/tlb_analysis.h"
#include "core/config/config.h"
#include "core/config/constants.h"
#include "core/config/sweep_utils.h"
#include "core/config/version.h"
#include "core/memory/buffer_allocator.h"
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

std::string base_mode_name(const BenchmarkConfig& config) {
  if (config.analyze_tlb) {
    return Constants::TLB_ANALYSIS_JSON_MODE_NAME;
  }
  if (config.run_patterns) {
    return Constants::PATTERNS_JSON_MODE_NAME;
  }
  return Constants::BENCHMARK_JSON_MODE_NAME;
}

SweepNestedMode nested_mode_for_config(const BenchmarkConfig& config) {
  if (config.analyze_tlb) {
    return SweepNestedMode::TlbAnalysis;
  }
  if (config.run_patterns) {
    return SweepNestedMode::Patterns;
  }
  return SweepNestedMode::Standard;
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

  PatternStatistics stats;
  if (run_all_pattern_benchmarks(run_config, stats) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }

  if (run_config.loop_count == 1) {
    print_pattern_results(extract_pattern_results_at(stats, 0));
  } else if (!stats.loop_results.empty()) {
    print_pattern_results(extract_pattern_median_results(stats));
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

std::string optional_string(const nlohmann::ordered_json& object, const char* key) {
  if (!object.is_object() || !object.contains(key) || !object[key].is_string()) {
    return "";
  }
  return object[key].get<std::string>();
}

bool optional_bool(const nlohmann::ordered_json& object, const char* key, bool fallback = false) {
  if (!object.is_object() || !object.contains(key) || !object[key].is_boolean()) {
    return fallback;
  }
  return object[key].get<bool>();
}

size_t completed_run_count(const nlohmann::ordered_json& runs_json) {
  size_t completed_runs = 0;
  if (!runs_json.is_array()) {
    return completed_runs;
  }
  for (const nlohmann::ordered_json& run_json : runs_json) {
    if (optional_string(run_json, "status") == "complete") {
      ++completed_runs;
    }
  }
  return completed_runs;
}

void update_sweep_output(nlohmann::ordered_json& output_json, const nlohmann::ordered_json& runs_json,
                         const std::string& status, const std::string& status_reason, size_t planned_runs,
                         double elapsed_sec, const std::string& timestamp) {
  const size_t completed_runs = completed_run_count(runs_json);
  output_json["status"] = status;
  output_json["status_reason"] =
      status_reason.empty() ? nlohmann::ordered_json(nullptr) : nlohmann::ordered_json(status_reason);
  output_json["planned_runs"] = planned_runs;
  output_json["attempted_runs"] = runs_json.is_array() ? runs_json.size() : 0;
  output_json["completed_runs"] = completed_runs;
  output_json["conclusions_valid"] = status == "complete" && completed_runs == planned_runs;
  output_json["runs"] = runs_json;
  output_json[JsonKeys::EXECUTION_TIME_SEC] = elapsed_sec;
  output_json[JsonKeys::TIMESTAMP] = timestamp;
  output_json[JsonKeys::VERSION] = SOFTVERSION;
}

SweepNestedCompletion classify_standard_completion(const nlohmann::ordered_json& result_json) {
  const std::string status = optional_string(result_json, "status");
  const std::string reason = optional_string(result_json, "status_reason");
  if (status == "complete" && optional_bool(result_json, "results_complete")) {
    return {SweepAttemptStatus::Complete, ""};
  }
  if (status == "interrupted") {
    return {SweepAttemptStatus::Interrupted, reason.empty() ? "nested-run-interrupted" : reason};
  }
  if (status == "failed") {
    return {SweepAttemptStatus::Failed, reason.empty() ? "nested-run-failed" : reason};
  }
  return {SweepAttemptStatus::Partial, reason.empty() ? "nested-standard-result-incomplete" : reason};
}

SweepNestedCompletion classify_tlb_completion(const nlohmann::ordered_json& result_json) {
  if (!result_json.is_object() || !result_json.contains("tlb_analysis") || !result_json["tlb_analysis"].is_object()) {
    return {SweepAttemptStatus::Partial, "missing-tlb-analysis-result"};
  }
  const nlohmann::ordered_json& analysis = result_json["tlb_analysis"];
  const std::string status = optional_string(analysis, "status");
  const std::string reason = optional_string(analysis, "status_reason");
  if (status == "complete" && optional_bool(analysis, "conclusions_valid")) {
    return {SweepAttemptStatus::Complete, ""};
  }
  if (status == "interrupted") {
    return {SweepAttemptStatus::Interrupted, reason.empty() ? "nested-tlb-run-interrupted" : reason};
  }
  if (status == "failed") {
    return {SweepAttemptStatus::Failed, reason.empty() ? "nested-tlb-run-failed" : reason};
  }
  return {SweepAttemptStatus::Partial, reason.empty() ? "nested-tlb-result-incomplete" : reason};
}

SweepNestedCompletion classify_core_to_core_completion(const nlohmann::ordered_json& result_json) {
  if (!result_json.is_object() || !result_json.contains("core_to_core_latency") ||
      !result_json["core_to_core_latency"].is_object()) {
    return {SweepAttemptStatus::Partial, "missing-core-to-core-result"};
  }
  const nlohmann::ordered_json& result = result_json["core_to_core_latency"];
  const std::string status = optional_string(result, "status");
  if (status == "complete" && optional_bool(result, "measurements_complete")) {
    return {SweepAttemptStatus::Complete, ""};
  }
  if (status == "interrupted") {
    return {SweepAttemptStatus::Interrupted, "nested-core-to-core-run-interrupted"};
  }
  if (status == "failed") {
    return {SweepAttemptStatus::Failed, "nested-core-to-core-run-failed"};
  }
  return {SweepAttemptStatus::Partial, "nested-core-to-core-result-incomplete"};
}

SweepNestedCompletion classify_pattern_completion(const nlohmann::ordered_json& result_json) {
  if (!result_json.is_object() || !result_json.contains(JsonKeys::CONFIGURATION) ||
      !result_json[JsonKeys::CONFIGURATION].is_object() ||
      !result_json[JsonKeys::CONFIGURATION].contains(JsonKeys::LOOP_COUNT) ||
      !result_json[JsonKeys::CONFIGURATION][JsonKeys::LOOP_COUNT].is_number_integer() ||
      !result_json.contains(JsonKeys::PATTERNS) || !result_json[JsonKeys::PATTERNS].is_object()) {
    return {SweepAttemptStatus::Partial, "missing-pattern-completion-metadata"};
  }

  const long long configured_loops = result_json[JsonKeys::CONFIGURATION][JsonKeys::LOOP_COUNT].get<long long>();
  if (configured_loops <= 0) {
    return {SweepAttemptStatus::Partial, "invalid-pattern-loop-count"};
  }
  const size_t expected_loops = static_cast<size_t>(configured_loops);
  const size_t expected_operation_count =
      static_cast<size_t>(PatternKind::Count) * static_cast<size_t>(PatternOperation::Count);

  size_t operation_count = 0;
  bool partial = false;
  bool interrupted = false;
  bool failed = false;
  std::string first_reason;
  for (const auto& pattern_entry : result_json[JsonKeys::PATTERNS].items()) {
    const nlohmann::ordered_json& pattern = pattern_entry.value();
    if (!pattern.is_object() || !pattern.contains(JsonKeys::BANDWIDTH) || !pattern[JsonKeys::BANDWIDTH].is_object()) {
      partial = true;
      continue;
    }
    for (const auto& operation_entry : pattern[JsonKeys::BANDWIDTH].items()) {
      ++operation_count;
      const nlohmann::ordered_json& operation = operation_entry.value();
      if (!operation.is_object() || !operation.contains("measurements") || !operation["measurements"].is_array()) {
        partial = true;
        continue;
      }
      const nlohmann::ordered_json& measurements = operation["measurements"];
      if (measurements.size() != expected_loops) {
        partial = true;
      }
      for (const nlohmann::ordered_json& measurement : measurements) {
        const std::string status = optional_string(measurement, "status");
        const std::string reason = optional_string(measurement, "reason");
        if (first_reason.empty() && !reason.empty()) {
          first_reason = reason;
        }
        if (status == "interrupted") {
          interrupted = true;
        } else if (status == "failed" || status == "invalid") {
          failed = true;
        } else if (status != "measured" && status != "skipped") {
          partial = true;
        }
      }
    }
  }

  if (operation_count != expected_operation_count) {
    partial = true;
  }
  if (failed) {
    return {SweepAttemptStatus::Failed, first_reason.empty() ? "nested-pattern-run-failed" : first_reason};
  }
  if (interrupted) {
    return {SweepAttemptStatus::Interrupted, first_reason.empty() ? "nested-pattern-run-interrupted" : first_reason};
  }
  if (partial) {
    return {SweepAttemptStatus::Partial, first_reason.empty() ? "nested-pattern-result-incomplete" : first_reason};
  }
  return {SweepAttemptStatus::Complete, ""};
}

}  // namespace

const char* sweep_attempt_status_to_string(SweepAttemptStatus status) {
  switch (status) {
    case SweepAttemptStatus::Complete:
      return "complete";
    case SweepAttemptStatus::Partial:
      return "partial";
    case SweepAttemptStatus::Interrupted:
      return "interrupted";
    case SweepAttemptStatus::Failed:
      return "failed";
  }
  return "failed";
}

SweepNestedCompletion classify_sweep_nested_completion(SweepNestedMode mode,
                                                       const nlohmann::ordered_json& result_json) {
  switch (mode) {
    case SweepNestedMode::Standard:
      return classify_standard_completion(result_json);
    case SweepNestedMode::Patterns:
      return classify_pattern_completion(result_json);
    case SweepNestedMode::TlbAnalysis:
      return classify_tlb_completion(result_json);
    case SweepNestedMode::CoreToCore:
      return classify_core_to_core_completion(result_json);
  }
  return {SweepAttemptStatus::Failed, "unknown-sweep-nested-mode"};
}

SweepExecutionResult execute_sweep_plan(SweepNestedMode mode, const std::vector<nlohmann::ordered_json>& run_parameters,
                                        nlohmann::ordered_json initial_output, const SweepExecutionHooks& hooks) {
  SweepExecutionResult execution;
  execution.output_json = std::move(initial_output);
  nlohmann::ordered_json runs_json = nlohmann::ordered_json::array();

  const auto elapsed_seconds = [&hooks]() { return hooks.elapsed_seconds ? hooks.elapsed_seconds() : 0.0; };
  const auto utc_timestamp = [&hooks]() {
    return hooks.utc_timestamp ? hooks.utc_timestamp() : build_utc_timestamp();
  };
  const auto stop_requested = [&hooks]() { return hooks.stop_requested && hooks.stop_requested(); };
  const auto write_checkpoint = [&hooks](const nlohmann::ordered_json& output, bool announce_success) {
    if (!hooks.write_checkpoint) {
      return EXIT_FAILURE;
    }
    return hooks.write_checkpoint(output, announce_success);
  };
  const auto checkpoint = [&](const std::string& status, const std::string& reason, bool announce_success) {
    update_sweep_output(execution.output_json, runs_json, status, reason, run_parameters.size(), elapsed_seconds(),
                        utc_timestamp());
    return write_checkpoint(execution.output_json, announce_success);
  };

  if (!hooks.execute_run) {
    update_sweep_output(execution.output_json, runs_json, "failed", "missing-sweep-run-executor", run_parameters.size(),
                        elapsed_seconds(), utc_timestamp());
    execution.exit_code = EXIT_FAILURE;
    return execution;
  }

  if (run_parameters.empty()) {
    if (checkpoint("complete", "", true) != EXIT_SUCCESS) {
      update_sweep_output(execution.output_json, runs_json, "failed", "checkpoint-write-failed", 0, elapsed_seconds(),
                          utc_timestamp());
      execution.exit_code = EXIT_FAILURE;
      return execution;
    }
    execution.exit_code = EXIT_SUCCESS;
    return execution;
  }

  for (size_t run_index = 0; run_index < run_parameters.size(); ++run_index) {
    if (stop_requested()) {
      if (checkpoint("interrupted", "interruption-requested-before-run", true) != EXIT_SUCCESS) {
        update_sweep_output(execution.output_json, runs_json, "failed", "checkpoint-write-failed",
                            run_parameters.size(), elapsed_seconds(), utc_timestamp());
        execution.exit_code = EXIT_FAILURE;
        return execution;
      }
      execution.exit_code = EXIT_SUCCESS;
      return execution;
    }

    const SweepRunOutcome outcome = hooks.execute_run(run_index);
    SweepNestedCompletion completion;
    if (outcome.exit_code == EXIT_SUCCESS) {
      completion = classify_sweep_nested_completion(mode, outcome.result_json);
    } else {
      completion.status = SweepAttemptStatus::Failed;
      completion.reason = outcome.failure_reason.empty() ? "nested-run-execution-failed" : outcome.failure_reason;
    }

    nlohmann::ordered_json run_json;
    run_json["index"] = run_index;
    run_json["parameters"] = run_parameters[run_index];
    run_json["status"] = sweep_attempt_status_to_string(completion.status);
    run_json["status_reason"] =
        completion.reason.empty() ? nlohmann::ordered_json(nullptr) : nlohmann::ordered_json(completion.reason);
    run_json["result"] = outcome.result_json.empty() ? nlohmann::ordered_json(nullptr) : outcome.result_json;
    runs_json.push_back(std::move(run_json));

    const bool interrupted_after_run = stop_requested();
    std::string sweep_status;
    std::string sweep_reason;
    bool terminal = false;
    int terminal_exit_code = EXIT_SUCCESS;
    if (outcome.exit_code != EXIT_SUCCESS || completion.status == SweepAttemptStatus::Failed) {
      sweep_status = "failed";
      sweep_reason = completion.reason;
      terminal = true;
      terminal_exit_code = EXIT_FAILURE;
    } else if (completion.status == SweepAttemptStatus::Interrupted) {
      sweep_status = "interrupted";
      sweep_reason = completion.reason;
      terminal = true;
    } else if (interrupted_after_run) {
      sweep_status = "interrupted";
      sweep_reason = completion.status == SweepAttemptStatus::Complete ? "interruption-requested-after-complete-run"
                                                                       : "interruption-requested-after-incomplete-run";
      terminal = true;
    } else if (completion.status == SweepAttemptStatus::Partial) {
      sweep_status = "partial";
      sweep_reason = completion.reason;
      terminal = true;
    } else if (run_index + 1 == run_parameters.size()) {
      sweep_status = "complete";
      terminal = true;
    } else {
      sweep_status = "partial";
      sweep_reason = "sweep-runs-remain";
    }

    const bool announce_success = terminal && terminal_exit_code == EXIT_SUCCESS;
    if (checkpoint(sweep_status, sweep_reason, announce_success) != EXIT_SUCCESS) {
      update_sweep_output(execution.output_json, runs_json, "failed", "checkpoint-write-failed", run_parameters.size(),
                          elapsed_seconds(), utc_timestamp());
      execution.exit_code = EXIT_FAILURE;
      return execution;
    }
    if (terminal) {
      execution.exit_code = terminal_exit_code;
      return execution;
    }
  }

  update_sweep_output(execution.output_json, runs_json, "failed", "sweep-coordinator-ended-without-terminal-status",
                      run_parameters.size(), elapsed_seconds(), utc_timestamp());
  execution.exit_code = EXIT_FAILURE;
  return execution;
}

size_t calculate_sweep_run_count(const BenchmarkConfig& config) {
  return calculate_sweep_run_count_from_specs(config.sweep_specs);
}

int run_sweep_mode(const BenchmarkConfig& base_config) {
  const size_t run_count = calculate_sweep_run_count(base_config);
  const std::vector<std::vector<SweepAssignment>> assignments = build_sweep_assignments(base_config);

  // Validate every generated configuration before starting the first potentially long run.
  for (const std::vector<SweepAssignment>& assignment : assignments) {
    BenchmarkConfig preflight_config = build_run_config(base_config, assignment);
    if (validate_config(preflight_config) != EXIT_SUCCESS) {
      return EXIT_FAILURE;
    }
  }

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
  output_json[JsonKeys::CONFIGURATION] = {{JsonKeys::MODE, Constants::SWEEP_JSON_MODE_NAME},
                                          {"base_mode", base_mode_name(base_config)},
                                          {"run_count", run_count},
                                          {"sweep_max_runs", base_config.sweep_max_runs},
                                          {"sweep_parameters", build_sweep_parameters_json(base_config)},
                                          {"main_thread_qos",
                                           {{"requested", true},
                                            {"requested_class", "user-interactive"},
                                            {"applied", qos_ret == KERN_SUCCESS},
                                            {"code", qos_ret},
                                            {"policy", "best-effort; continue on failure"}}}};

  std::filesystem::path file_path(base_config.output_file);
  if (file_path.is_relative()) {
    file_path = std::filesystem::current_path() / file_path;
  }

  std::vector<nlohmann::ordered_json> run_parameters;
  run_parameters.reserve(assignments.size());
  for (const std::vector<SweepAssignment>& assignment : assignments) {
    run_parameters.push_back(build_assignment_json(assignment));
  }

  SweepExecutionHooks hooks;
  hooks.execute_run = [&](size_t run_index) {
    std::cout << Messages::msg_sweep_run_progress(run_index + 1, assignments.size()) << std::endl;
    BenchmarkConfig run_config = build_run_config(base_config, assignments[run_index]);
    run_config.main_thread_qos_requested = true;
    run_config.main_thread_qos_applied = qos_ret == KERN_SUCCESS;
    run_config.main_thread_qos_code = qos_ret;
    SweepRunOutcome outcome;
    outcome.exit_code = run_sweep_point(run_config, run_index, outcome.result_json);
    if (outcome.exit_code != EXIT_SUCCESS) {
      outcome.failure_reason = "nested-run-execution-failed";
    }
    return outcome;
  };
  hooks.stop_requested = []() { return signal_received(); };
  hooks.elapsed_seconds = [&]() { return total_timer.stop(); };
  hooks.write_checkpoint = [&](const nlohmann::ordered_json& checkpoint, bool announce_success) {
    return write_json_to_file(file_path, checkpoint, announce_success);
  };

  const SweepExecutionResult execution =
      execute_sweep_plan(nested_mode_for_config(base_config), run_parameters, std::move(output_json), hooks);

  restore_signal_mask();
  if (execution.output_json.value("status", "failed") == "interrupted") {
    const nlohmann::ordered_json& runs = execution.output_json["runs"];
    if (!runs.is_array() || runs.empty() || runs.back().value("status", "failed") == "complete") {
      std::cout << Messages::msg_interrupted_by_user() << std::endl;
    }
  }
  if (execution.exit_code == EXIT_SUCCESS) {
    const double total_elapsed_sec = execution.output_json.value(JsonKeys::EXECUTION_TIME_SEC, 0.0);
    std::cout << Messages::msg_done_total_time(total_elapsed_sec) << std::endl;
  }
  return execution.exit_code;
}

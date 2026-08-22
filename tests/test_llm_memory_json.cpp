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

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/config/constants.h"
#include "core/config/version.h"
#include "llm_memory/llm_json.h"
#include "utils/numeric_utils.h"

namespace {

using OrderedJson = nlohmann::ordered_json;

constexpr uint64_t kAboveJsonExactInteger = 9007199254740993ULL;

LlmMemoryConfig explicit_config(size_t loop_count = 3) {
  LlmMemoryConfig config;
  config.weight_size_mb = 1;
  config.layer_count = 2;
  config.query_head_count = 4;
  config.kv_head_count = 2;
  config.head_dimension = 8;
  config.kv_element_bytes = 2;
  config.visible_context_tokens = 3;
  config.batch_size = 1;
  config.requested_workers = 2;
  config.available_workers = 2;
  config.iterations = 4;
  config.loop_count = loop_count;
  config.seed = std::numeric_limits<uint64_t>::max();
  config.user_specified_iterations = true;
  config.user_specified_seed = true;
  config.user_specified_workers = true;
  config.output_file = "--literal-output-name.json";
  config.argv = {"memory_benchmark", "--llm-memory", "--output", "--literal-output-name.json"};
  return config;
}

LlmMemoryWorkPlanRequest plan_request(const LlmMemoryConfig& config) {
  LlmMemoryWorkPlanRequest request;
  request.geometry.active_weight_bytes = config.weight_size_mb * Constants::BYTES_PER_MB;
  request.geometry.layer_count = config.layer_count;
  request.geometry.query_head_count = config.query_head_count;
  request.geometry.kv_head_count = config.kv_head_count;
  request.geometry.head_dimension = config.head_dimension;
  request.geometry.kv_element_bytes = config.kv_element_bytes;
  request.geometry.visible_context_tokens = config.visible_context_tokens;
  request.geometry.batch_size = config.batch_size;
  request.requested_workers = config.requested_workers;
  request.available_workers = config.available_workers;
  request.available_memory_bytes = 8ULL * 1024ULL * Constants::BYTES_PER_MB;
  request.mapping_granularity_bytes = 4096;
  request.base_seed = config.seed;
  return request;
}

LlmMemoryWorkPlan admitted_plan(const LlmMemoryConfig& config) {
  LlmMemoryWorkPlanRequest request = plan_request(config);
  const LlmMemoryWorkPlan preliminary = build_llm_memory_work_plan(request);
  if (!preliminary.valid) {
    return build_llm_memory_work_plan(request);
  }

  const LlmExecutorAuxiliaryEstimate executor = calculate_llm_executor_auxiliary_estimate(preliminary);
  const LlmRunnerAuxiliaryEstimate runner = calculate_llm_runner_auxiliary_estimate(config, preliminary);
  const LlmJsonPeakEstimate json_peak = calculate_llm_json_peak_estimate(config, preliminary);
  size_t runner_and_executor_orchestration_bytes = 0;
  if (!executor.valid || !runner.valid || !json_peak.valid ||
      !NumericUtils::checked_add(executor.checksum_auxiliary_bytes, runner.checksum_auxiliary_bytes,
                                 request.checksum_auxiliary_bytes) ||
      !NumericUtils::checked_add(executor.orchestration_auxiliary_bytes, runner.orchestration_auxiliary_bytes,
                                 runner_and_executor_orchestration_bytes) ||
      !NumericUtils::checked_add(runner_and_executor_orchestration_bytes, json_peak.total_bytes,
                                 request.orchestration_auxiliary_bytes)) {
    return build_llm_memory_work_plan(request);
  }
  return build_llm_memory_work_plan(request);
}

LlmReadChecksumComponent checksum_component(uint64_t offset) {
  return {std::numeric_limits<uint64_t>::max() - offset, kAboveJsonExactInteger + offset,
          kAboveJsonExactInteger + 100 + offset, 17 + offset};
}

LlmExecutorResult successful_execution(const LlmMemoryWorkPlan& plan, double elapsed_seconds = 0.150) {
  LlmExecutorResult execution;
  execution.valid = true;
  execution.reason_code = LlmExecutorReason::VALID;
  execution.elapsed_seconds = elapsed_seconds;
  execution.requested_workers = plan.effective_workers;
  execution.created_workers = plan.effective_workers;
  execution.completed_workers = plan.effective_workers;
  execution.qos_successful_workers = plan.effective_workers;
  execution.kernel_succeeded = true;
  execution.timer_started = true;
  execution.timer_stopped = true;
  execution.checksum_evaluated = true;
  execution.checksum_valid = true;
  execution.expected_checksums.resize(plan.effective_workers);
  for (size_t worker_index = 0; worker_index < plan.effective_workers; ++worker_index) {
    execution.expected_checksums[worker_index] = {checksum_component(worker_index * 3),
                                                  checksum_component(worker_index * 3 + 1),
                                                  checksum_component(worker_index * 3 + 2)};
  }
  execution.actual_checksums = execution.expected_checksums;
  execution.expected_run_checksum = {std::numeric_limits<uint64_t>::max(), kAboveJsonExactInteger};
  execution.actual_run_checksum = execution.expected_run_checksum;
  return execution;
}

LlmMemoryResult complete_result(const LlmMemoryConfig& config, const LlmMemoryWorkPlan& plan) {
  LlmMemoryResult result;
  const LlmTaskExecutor executor = [](const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan&,
                                      const LlmRunnerTaskContext&) { return successful_execution(model_plan); };
  EXPECT_EQ(run_llm_memory_suite(config, plan, executor, result), EXIT_SUCCESS);
  return result;
}

LlmResourcePreparationResult preparation_for(const LlmMemoryWorkPlan& plan) {
  LlmResourcePreparationResult preparation;
  preparation.valid = true;
  preparation.reason_code = LlmExecutorReason::VALID;
  preparation.auxiliary = calculate_llm_executor_auxiliary_estimate(plan);
  preparation.memory_budget = plan.memory_budget;
  preparation.initialization.complete = true;
  preparation.initialization.weight_bytes = plan.geometry.active_weight_bytes_per_work_unit;
  preparation.initialization.k_bytes = plan.geometry.k_mapping_bytes;
  preparation.initialization.v_bytes = plan.geometry.v_mapping_bytes;
  preparation.initialization.total_bytes = plan.geometry.total_data_mapping_bytes;
  preparation.initialization.non_empty_weight_spans = plan.total_layer_descriptors;
  preparation.initialization.non_empty_k_spans = plan.total_sequence_descriptors;
  preparation.initialization.non_empty_v_spans = plan.total_sequence_descriptors;
  return preparation;
}

LlmResultMetadata fixed_metadata(const LlmMemoryConfig& config, const LlmMemoryWorkPlan& plan) {
  LlmResultMetadata metadata;
  metadata.timestamp = "2000-01-01T00:00:00Z";
  metadata.processor_name = "Test Apple Silicon";
  metadata.macos_version = "test-macos";
  metadata.performance_core_count = 4;
  metadata.efficiency_core_count = 2;
  metadata.logical_core_count = 6;
  metadata.page_size_bytes = 16384;
  metadata.l1_data_cache_bytes = 128 * 1024;
  metadata.l2_data_cache_bytes = 0;
  metadata.available_memory_bytes = 4ULL * 1024ULL * Constants::BYTES_PER_MB;
  metadata.available_memory_source = "test-provider";
  metadata.json_peak_estimate = calculate_llm_json_peak_estimate(config, plan);
  metadata.main_thread_qos = {true, true, 0};
  metadata.environment_start.thermal_state = "nominal";
  metadata.environment_start.low_power_mode_available = true;
  metadata.environment_start.low_power_mode_enabled = false;
  metadata.environment_start.physical_memory_bytes = std::numeric_limits<uint64_t>::max();
  metadata.environment_end = metadata.environment_start;
  return metadata;
}

std::vector<std::string> json_string_array(const OrderedJson& array) {
  std::vector<std::string> output;
  for (const OrderedJson& value : array) {
    output.push_back(value.get<std::string>());
  }
  return output;
}

void expect_exact_keys(const OrderedJson& object, std::initializer_list<const char*> keys) {
  ASSERT_TRUE(object.is_object());
  EXPECT_EQ(object.size(), keys.size());
  for (const char* key : keys) {
    EXPECT_TRUE(object.contains(key)) << key;
  }
}

}  // namespace

TEST(LlmMemoryJsonTest, CompleteDocumentHasExactTopLevelIdentityAndAuditableNestedEvidence) {
  const LlmMemoryConfig config = explicit_config();
  const LlmMemoryWorkPlan plan = admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  const LlmMemoryResult result = complete_result(config, plan);
  ASSERT_TRUE(result.results_complete);
  const LlmResourcePreparationResult preparation = preparation_for(plan);
  const LlmResultMetadata metadata = fixed_metadata(config, plan);

  const OrderedJson document = build_llm_memory_json(config, plan, preparation, metadata, result);

  const std::array<const char*, 28> top_level_keys = {
      "schema_version",       "mode",                "backend",        "phase",
      "kv_layout",            "methodology_version", "software",       "configuration",
      "resolved_plan",        "backend_evidence",    "memory_budget",  "calibration",
      "measurements",         "aggregates",          "status",         "reason_code",
      "results_complete",     "conclusions_valid",   "interpretation", "diagnostic",
      "interruption_requested", "scenario_order_balance_complete",       "seeds",
      "counters",             "checkpoint_lifecycle", "loop_records",   "environment",
      "quality_warnings"};
  ASSERT_EQ(document.size(), top_level_keys.size());
  for (const char* key : top_level_keys) {
    EXPECT_TRUE(document.contains(key)) << key;
  }

  expect_exact_keys(
      document["configuration"],
      {"backend", "phase", "kv_layout", "weight_size_mb", "layer_count", "query_head_count", "kv_head_count",
       "head_dimension", "kv_element_bytes", "visible_context_tokens", "batch_size", "requested_workers",
       "available_workers", "worker_source", "iterations", "work_policy", "loop_count",
       "base_seed_uint64_decimal", "seed_source", "output_file", "argv", "resolved_sources"});
  expect_exact_keys(document["configuration"]["resolved_sources"],
                    {"backend", "phase", "kv_layout", "workers", "iterations", "seed"});
  expect_exact_keys(document["resolved_plan"]["methodology"], {"methodology_version",
                                              "backend",
                                              "phase",
                                              "kv_layout",
                                              "work_unit_kind",
                                              "weight_passes_per_work_unit",
                                              "kv_replay_factor",
                                              "schedule_version",
                                              "warmup_policy",
                                              "context_policy",
                                              "scenario_order_policy",
                                              "timing_policy",
                                              "cache_policy",
                                              "calibration_policy",
                                              "calibration_target_seconds",
                                              "calibration_min_seconds",
                                              "calibration_max_seconds",
                                              "calibration_max_corrections",
                                              "calibration_min_pilot_accounted_bytes",
                                              "maximum_work_units_per_measurement",
                                              "maximum_accounted_bytes_per_task",
                                              "repeatability_cv_warning_threshold_pct",
                                              "calibration_excluded_from_results",
                                              "timed_region_exclusions",
                                              "resource_abi_version",
                                              "buffer_pattern_version",
                                              "write_pattern_version",
                                              "checksum_pattern_version"});
  expect_exact_keys(document["resolved_plan"]["geometry"], {"valid",
                                           "reason_code",
                                           "phase",
                                           "work_unit_kind",
                                           "decode",
                                           "prefill",
                                           "attention_kind",
                                           "active_weight_bytes_per_work_unit",
                                           "layer_count",
                                           "query_head_count",
                                           "kv_head_count",
                                           "query_heads_per_kv_head",
                                           "head_dimension",
                                           "kv_element_bytes",
                                           "batch_size",
                                           "kv_vector_bytes",
                                           "k_or_v_record_bytes_per_layer",
                                           "kv_record_bytes_per_layer",
                                           "kv_bytes_per_visible_token",
                                           "k_or_v_sequence_visible_bytes",
                                           "k_mapping_bytes",
                                           "v_mapping_bytes",
                                           "kv_capacity_bytes",
                                           "weight_read_bytes_per_work_unit",
                                           "kv_read_bytes_per_work_unit",
                                           "kv_write_bytes_per_work_unit",
                                           "kv_only_effective_model_payload_bytes_per_work_unit",
                                           "mixed_effective_model_payload_bytes_per_work_unit",
                                           "total_data_mapping_bytes",
                                           "traffic_crossover_numerator",
                                           "traffic_crossover_denominator",
                                           "traffic_crossover_context_tokens"});
  expect_exact_keys(document["aggregates"]["traffic_diagnostics"],
                    {"classification_version", "traffic_crossover_numerator", "traffic_crossover_denominator",
                     "traffic_crossover_context_tokens", "current_visible_context_tokens",
                     "current_weight_read_payload_bytes_per_work_unit", "current_kv_read_payload_bytes_per_work_unit",
                     "current_weight_to_kv_read_payload_ratio", "current_context_classification",
                     "classification_is_payload_only", "scenario_headlines"});
  expect_exact_keys(document["aggregates"]["traffic_diagnostics"]["scenario_headlines"]["mixed"],
                    {"synthetic_work_unit_latency_seconds",
                     "synthetic_memory_work_units_per_second",
                     "effective_model_payload_gb_s"});
  expect_exact_keys(document["memory_budget"],
                    {"resource_rounding_bytes", "transient_peak_bytes", "known_owned_peak_bytes",
                     "admitted_budget_bytes", "valid", "reason_code", "request", "available_memory_bytes",
                     "allowed_memory_bytes", "used_fallback"});
  expect_exact_keys(document["memory_budget"]["request"],
                    {"valid", "reason_code", "mapping_granularity_bytes", "requested_weight_mapping_bytes",
                     "requested_k_mapping_bytes", "requested_v_mapping_bytes", "committed_weight_mapping_bytes",
                     "committed_k_mapping_bytes", "committed_v_mapping_bytes", "requested_data_bytes",
                     "committed_data_bytes", "descriptor_bytes", "planner_storage_bytes", "checksum_auxiliary_bytes",
                     "orchestration_auxiliary_bytes", "auxiliary_bytes", "required_total_bytes"});
  expect_exact_keys(document["backend_evidence"], {"cpu", "metal"});
  ASSERT_TRUE(document["backend_evidence"]["cpu"].is_object());
  EXPECT_TRUE(document["backend_evidence"]["metal"].is_null());
  const OrderedJson& cpu_resources = document["backend_evidence"]["cpu"]["resources"];
  expect_exact_keys(cpu_resources,
                    {"valid", "reason_code", "model_plan_identity", "mappings", "descriptors", "executor_auxiliary",
                     "json_output_peak_estimate", "allocation_memory_budget", "initialization"});
  expect_exact_keys(cpu_resources["mappings"], {"policy", "full_size_physical_mappings", "weight", "k", "v",
                                                 "requested_data_bytes", "committed_data_bytes"});
  expect_exact_keys(cpu_resources["mappings"]["weight"], {"requested_bytes", "committed_bytes"});
  expect_exact_keys(cpu_resources["descriptors"],
                    {"abi_version", "layer_descriptors_per_worker", "sequence_descriptors_per_worker",
                     "total_layer_descriptors", "total_sequence_descriptors", "descriptor_bytes"});
  expect_exact_keys(cpu_resources["executor_auxiliary"],
                    {"valid", "reason_code", "static_reference_bytes", "expected_checksum_bytes",
                     "actual_checksum_bytes", "run_checksum_bytes", "worker_status_bytes", "thread_handle_bytes",
                     "checksum_auxiliary_bytes", "orchestration_auxiliary_bytes", "total_auxiliary_bytes"});
  expect_exact_keys(cpu_resources["json_output_peak_estimate"],
                    {"enabled", "valid", "reason_code", "policy", "fixed_schema_bytes", "input_string_bytes",
                     "measurement_record_bytes", "worker_checksum_bytes", "total_bytes"});
  expect_exact_keys(cpu_resources["initialization"],
                    {"complete", "pattern_version", "pre_touch_policy", "separate_reference_read_pass",
                     "static_references_accumulated_during_initialization", "weight_bytes", "k_bytes", "v_bytes",
                     "total_bytes", "non_empty_weight_spans", "non_empty_k_spans", "non_empty_v_spans"});
  expect_exact_keys(document["seeds"],
                    {"base_seed_uint64_decimal", "source", "buffer_domain_seeds", "scenario_domain_seeds"});
  expect_exact_keys(document["seeds"]["buffer_domain_seeds"],
                    {"weight_uint64_decimal", "k_uint64_decimal", "v_uint64_decimal"});
  expect_exact_keys(document["seeds"]["scenario_domain_seeds"], {"weights_only", "kv_only", "mixed"});
  expect_exact_keys(document["resolved_plan"],
                    {"valid", "reason_code", "plan_identity", "methodology_version", "backend", "phase",
                     "kv_layout", "work_unit_kind", "geometry", "layout", "resources", "component_identities",
                     "methodology", "model_work_plan", "frozen_scenario_work_plans"});
  expect_exact_keys(document["resolved_plan"]["geometry"]["decode"], {"visible_context_tokens"});
  EXPECT_TRUE(document["resolved_plan"]["geometry"]["prefill"].is_null());
  expect_exact_keys(document["resolved_plan"]["layout"],
                    {"kv_layout", "kv_block_tokens", "blocks_per_sequence", "physical_blocks_per_layer",
                     "last_block_tokens", "last_block_valid_bytes", "block_table_entries", "block_table_bytes",
                     "permutation_domain_uint64_hex", "permutation_seed_uint64_decimal",
                     "permutation_algorithm_version", "permutation_sha256"});
  expect_exact_keys(document["resolved_plan"]["resources"],
                    {"weight_logical_bytes", "k_logical_bytes", "v_logical_bytes", "k_physical_length_bytes",
                     "v_physical_length_bytes", "k_layout_padding_bytes", "v_layout_padding_bytes",
                     "block_table_bytes"});
  expect_exact_keys(document["resolved_plan"]["component_identities"],
                    {"logical_profile_version", "kv_layout_version", "permutation_version",
                     "backend_executor_version", "resource_abi_version", "schedule_version",
                     "timer_policy_version", "buffer_pattern_version", "write_pattern_version",
                     "checksum_pattern_version", "msl_revision", "msl_source_sha256", "identity"});
  expect_exact_keys(document["resolved_plan"]["model_work_plan"], {"valid",
                                                  "reason_code",
                                                  "plan_identity",
                                                  "methodology_version",
                                                  "backend",
                                                  "phase",
                                                  "kv_layout",
                                                  "work_unit_kind",
                                                  "component_identity",
                                                  "weight_passes_per_work_unit",
                                                  "kv_replay_factor",
                                                  "requested_workers",
                                                  "available_workers",
                                                  "effective_workers",
                                                  "worker_plan_count",
                                                  "weight_layer_count",
                                                  "layer_descriptors_per_worker",
                                                  "sequence_descriptors_per_worker",
                                                  "total_layer_descriptors",
                                                  "total_sequence_descriptors",
                                                  "descriptor_bytes",
                                                  "planner_storage_bytes"});
  expect_exact_keys(document["resolved_plan"]["frozen_scenario_work_plans"],
                    {"valid", "reason_code", "explicit_iterations", "model_plan_identity", "plan_identity",
                     "scenarios"});
  expect_exact_keys(document["resolved_plan"]["frozen_scenario_work_plans"]["scenarios"][0],
                    {"valid", "reason_code", "scenario", "work_unit_kind", "kv_write_kind", "explicit_iterations",
                     "model_plan_identity",
                     "scenario_seed_uint64_decimal", "work_units",
                     "weight_read_bytes_per_work_unit",
                     "kv_read_bytes_per_work_unit",
                     "kv_write_bytes_per_work_unit", "effective_model_payload_bytes_per_work_unit",
                     "layout_metadata_lookup_count_per_work_unit", "layout_metadata_read_bytes_per_work_unit",
                     "accounted_bytes_per_work_unit", "weight_read_bytes", "kv_read_bytes", "kv_write_bytes",
                     "effective_model_payload_bytes", "layout_metadata_lookup_count", "layout_metadata_read_bytes",
                     "task_accounted_bytes", "maximum_work_units_by_work_unit_cap",
                     "maximum_work_units_by_guardrail", "effective_maximum_work_units", "plan_identity"});
  expect_exact_keys(document["calibration"], {"excluded_from_results", "attempts"});
  expect_exact_keys(document["calibration"]["attempts"], {"weights_only", "kv_only", "mixed"});
  expect_exact_keys(document["calibration"]["attempts"]["weights_only"][0],
                    {"attempt_index", "scenario", "work_unit_kind", "kv_write_kind", "purpose",
                     "explicit_iterations", "work_units", "weight_read_bytes", "kv_read_bytes", "kv_write_bytes",
                     "effective_model_payload_bytes", "layout_metadata_lookup_count", "layout_metadata_read_bytes",
                     "task_accounted_bytes", "work_plan_identity",
                     "duration_quality", "terminal", "valid", "reason_code", "execution"});
  expect_exact_keys(document["calibration"]["attempts"]["weights_only"][0]["execution"],
                    {"status", "reason_code", "valid", "elapsed_seconds", "requested_workers", "created_workers",
                     "completed_workers", "qos_successful_workers", "qos_failed_workers", "worker_startup_failed",
                     "kernel_succeeded", "timer_started", "timer_stopped", "checksum"});
  expect_exact_keys(
      document["calibration"]["attempts"]["weights_only"][0]["execution"]["checksum"],
      {"status", "reason_code", "algorithm_version", "checksum_valid", "expected_run_checksum", "actual_run_checksum"});
  expect_exact_keys(
      document["counters"],
      {"planned_loops", "attempted_loops", "completed_loops", "planned_measurements", "attempted_measurements",
       "terminal_measurements", "measured_measurements", "planned_work_units", "completed_work_units",
       "planned_effective_model_payload_bytes", "completed_effective_model_payload_bytes",
       "planned_layout_metadata_lookup_count", "completed_layout_metadata_lookup_count",
       "planned_layout_metadata_read_bytes", "completed_layout_metadata_read_bytes", "planned_task_accounted_bytes",
       "completed_task_accounted_bytes", "runner_auxiliary"});
  expect_exact_keys(
      document["counters"]["runner_auxiliary"],
      {"valid", "reason_code", "measurement_record_bytes", "loop_record_bytes", "calibration_record_bytes",
       "calibration_identity_bytes", "aggregate_value_bytes", "statistics_workspace_bytes", "warning_record_bytes",
       "fixed_metadata_bytes", "retained_checksum_bytes", "checksum_auxiliary_bytes", "orchestration_auxiliary_bytes",
       "total_auxiliary_bytes"});
  expect_exact_keys(
      document["checkpoint_lifecycle"],
      {"checkpoint_failed", "logical_checkpoint_attempts", "successful_logical_checkpoints",
       "terminal_checkpoint_attempted", "terminal_checkpoint_completed", "checkpoint_policy",
       "file_checkpoint_failure_is_terminal_and_not_retried", "stdout_intermediate_checkpoints_are_lazy"});
  expect_exact_keys(document["loop_records"][0],
                    {"loop_index", "planned_order", "realized_order", "realized_order_count", "measurement_indexes"});
  expect_exact_keys(document["measurements"][0], {"scenario",
                                                  "work_unit_kind",
                                                  "kv_write_kind",
                                                  "loop_index",
                                                  "order_position",
                                                  "status",
                                                  "reason_code",
                                                  "attempted",
                                                  "requested_workers",
                                                  "effective_workers",
                                                  "qos_successful_workers",
                                                  "qos_failed_workers",
                                                  "frozen_plan_index",
                                                  "frozen_work_plan_identity",
                                                  "scenario_seed_uint64_decimal",
                                                  "explicit_iterations",
                                                  "work_policy",
                                                  "duration_quality",
                                                  "calibration_attempt_count",
                                                  "calibration_attempt_indexes",
                                                  "planned_work_units",
                                                  "completed_work_units",
                                                  "weight_read_bytes_per_work_unit",
                                                  "kv_read_bytes_per_work_unit",
                                                  "kv_write_bytes_per_work_unit",
                                                  "effective_model_payload_bytes_per_work_unit",
                                                  "layout_metadata_lookup_count_per_work_unit",
                                                  "layout_metadata_read_bytes_per_work_unit",
                                                  "accounted_bytes_per_work_unit",
                                                  "planned_weight_read_bytes",
                                                  "planned_kv_read_bytes",
                                                  "planned_kv_write_bytes",
                                                  "planned_effective_model_payload_bytes",
                                                  "completed_effective_model_payload_bytes",
                                                  "planned_layout_metadata_lookup_count",
                                                  "completed_layout_metadata_lookup_count",
                                                  "planned_layout_metadata_read_bytes",
                                                  "completed_layout_metadata_read_bytes",
                                                  "planned_task_accounted_bytes",
                                                  "completed_task_accounted_bytes",
                                                  "elapsed_seconds",
                                                  "synthetic_work_unit_latency_seconds",
                                                  "synthetic_memory_work_units_per_second",
                                                  "effective_model_payload_gb_s",
                                                  "weight_payload_fraction",
                                                  "kv_read_payload_fraction",
                                                  "kv_write_payload_fraction",
                                                  "working_set",
                                                  "execution",
                                                  "checksum"});
  expect_exact_keys(document["measurements"][0]["working_set"],
                    {"bytes", "full_size_physical_mappings", "cacheable", "kv_layout", "fixed_visible_context_tokens",
                     "current_token_slot_included"});
  expect_exact_keys(document["measurements"][0]["execution"],
                    {"status", "reason_code", "valid", "elapsed_seconds", "requested_workers", "created_workers",
                     "completed_workers", "qos_successful_workers", "qos_failed_workers", "worker_startup_failed",
                     "kernel_succeeded", "timer_started", "timer_stopped"});
  expect_exact_keys(document["measurements"][0]["checksum"],
                    {"status", "reason_code", "initialization_pattern_version", "append_pattern_version",
                     "read_checksum_version", "checksum_valid", "expected_worker_checksums", "actual_worker_checksums",
                     "expected_run_checksum", "actual_run_checksum"});
  expect_exact_keys(document["measurements"][0]["checksum"]["expected_worker_checksums"][0],
                    {"worker_index", "weight", "k", "v"});
  expect_exact_keys(
      document["measurements"][0]["checksum"]["expected_worker_checksums"][0]["weight"],
      {"state_a_uint64_decimal", "state_b_uint64_decimal", "exact_bytes_read", "span_count_uint64_decimal"});
  expect_exact_keys(document["measurements"][0]["checksum"]["expected_run_checksum"],
                    {"state_a_uint64_decimal", "state_b_uint64_decimal"});
  expect_exact_keys(document["aggregates"], {"scenarios", "traffic_diagnostics"});
  expect_exact_keys(document["aggregates"]["scenarios"], {"weights_only", "kv_only", "mixed"});
  expect_exact_keys(document["aggregates"]["scenarios"]["mixed"],
                    {"scenario", "status", "stability_quality", "cv_warning_threshold_pct",
                     "synthetic_work_unit_latency_seconds",
                     "synthetic_memory_work_units_per_second",
                     "effective_model_payload_gb_s"});
  expect_exact_keys(document["aggregates"]["scenarios"]["mixed"]["effective_model_payload_gb_s"],
                    {"units", "sample_count", "headline_semantics", "headline", "values", "statistics"});
  expect_exact_keys(document["aggregates"]["scenarios"]["mixed"]["effective_model_payload_gb_s"]["statistics"],
                    {"sample_count", "average", "min", "max", "median", "p90", "p95", "p99", "stddev",
                     "coefficient_of_variation_pct", "median_absolute_deviation"});
  expect_exact_keys(document["environment"],
                    {"processor_name", "macos_version", "performance_core_count", "efficiency_core_count",
                     "logical_core_count", "page_size_bytes", "l1_data_cache_bytes", "l2_data_cache_bytes",
                     "available_memory_bytes", "available_memory_source", "main_thread_qos", "start", "end"});
  expect_exact_keys(document["environment"]["main_thread_qos"], {"requested", "applied", "code"});
  expect_exact_keys(document["environment"]["start"],
                    {"thermal_state", "low_power_mode_available", "low_power_mode_enabled", "physical_memory_bytes"});
  expect_exact_keys(
      document["interpretation"],
      {"result_scope", "reported_rate", "backend", "phase", "work_unit_kind", "transformer_math_included",
       "framework_scheduler_and_dispatch_included", "compute_memory_overlap_included", "physical_dram_traffic_measured",
       "dram_residency", "cache_residency", "fixed_context_includes_current_token_slot", "kv_layout",
       "payload_semantics", "layout_metadata_timed_but_excluded_from_effective_model_payload_gb_s",
       "prefill_transformer_compute_or_ttft_prediction_included", "private_metal_storage_implies_separate_vram",
       "cross_backend_performance_distributions_combined", "traffic_classification_semantics",
       "full_size_working_set_reduces_but_does_not_prove_dram_residency", "comparability_requires"});

  EXPECT_EQ(document["software"]["version"], SOFTVERSION);
  EXPECT_EQ(document["software"]["timestamp"], metadata.timestamp);
  EXPECT_EQ(document["schema_version"], 1);
  EXPECT_EQ(document["mode"], "llm_memory");
  EXPECT_EQ(document["backend"], "cpu");
  EXPECT_EQ(document["phase"], "decode");
  EXPECT_EQ(document["kv_layout"], "contiguous");
  EXPECT_EQ(document["methodology_version"], Constants::LLM_CPU_DECODE_CONTIGUOUS_METHODOLOGY_VERSION);
  EXPECT_EQ(document["status"], "complete");
  EXPECT_TRUE(document["results_complete"]);
  EXPECT_TRUE(document["conclusions_valid"]);
  EXPECT_TRUE(document["scenario_order_balance_complete"]);
  EXPECT_TRUE(document["diagnostic"].is_null());

  EXPECT_EQ(document["configuration"]["output_file"], "--literal-output-name.json");
  EXPECT_EQ(document["configuration"]["argv"], config.argv);
  EXPECT_EQ(document["configuration"]["work_policy"], "explicit_fixed_work");
  EXPECT_EQ(document["configuration"]["kv_element_bytes"], "2");
  EXPECT_EQ(document["configuration"]["base_seed_uint64_decimal"], "18446744073709551615");
  EXPECT_EQ(document["resolved_plan"]["geometry"]["active_weight_bytes_per_work_unit"],
            std::to_string(plan.geometry.active_weight_bytes_per_work_unit));
  EXPECT_TRUE(document["resolved_plan"]["geometry"]["active_weight_bytes_per_work_unit"].is_string());
  EXPECT_EQ(document["resolved_plan"]["geometry"]["kv_capacity_bytes"],
            std::to_string(plan.geometry.kv_capacity_bytes));
  EXPECT_EQ(document["resolved_plan"]["geometry"]["kv_element_bytes"], "2");
  EXPECT_EQ(cpu_resources["model_plan_identity"], plan.plan_identity);
  EXPECT_TRUE(cpu_resources["json_output_peak_estimate"]["enabled"]);
  EXPECT_TRUE(cpu_resources["json_output_peak_estimate"]["valid"]);
  EXPECT_EQ(cpu_resources["json_output_peak_estimate"]["reason_code"], LlmJsonReason::VALID);
  EXPECT_EQ(cpu_resources["json_output_peak_estimate"]["total_bytes"],
            std::to_string(metadata.json_peak_estimate.total_bytes));
  EXPECT_EQ(cpu_resources["initialization"]["pattern_version"], Constants::LLM_BUFFER_PATTERN_VERSION);
  EXPECT_FALSE(cpu_resources["initialization"]["separate_reference_read_pass"]);
  EXPECT_EQ(document["resolved_plan"]["model_work_plan"]["plan_identity"], plan.plan_identity);
  EXPECT_EQ(document["resolved_plan"]["frozen_scenario_work_plans"]["scenarios"].size(), kLlmScenarioCount);
  EXPECT_EQ(document["calibration"]["attempts"]["weights_only"].size(), 1u);
  EXPECT_EQ(document["loop_records"].size(), 3u);
  EXPECT_EQ(document["measurements"].size(), 9u);
  EXPECT_EQ(document["aggregates"]["scenarios"]["mixed"]["effective_model_payload_gb_s"]["sample_count"], 3u);
  EXPECT_EQ(document["environment"]["start"]["physical_memory_bytes"], "18446744073709551615");
  EXPECT_EQ(document["interpretation"]["reported_rate"], "synthetic_memory_work_units_per_second");
  EXPECT_FALSE(document["interpretation"]["transformer_math_included"]);
  EXPECT_FALSE(document["interpretation"]["physical_dram_traffic_measured"]);
}

TEST(LlmMemoryJsonTest, TrafficClassificationUsesExactPayloadEqualityForNearCrossover) {
  LlmGeometryRequest request{8, 1, 1, 1, 1, 1, 4, 1};
  LlmGeometry geometry = resolve_llm_geometry(request);
  ASSERT_TRUE(geometry.valid) << geometry.reason_code;
  ASSERT_EQ(geometry.weight_read_bytes_per_work_unit, geometry.kv_read_bytes_per_work_unit);
  EXPECT_STREQ(classify_llm_traffic_payload(geometry), "near_crossover");

  request.active_weight_bytes = 9;
  geometry = resolve_llm_geometry(request);
  ASSERT_TRUE(geometry.valid) << geometry.reason_code;
  EXPECT_STREQ(classify_llm_traffic_payload(geometry), "weight_payload_dominant");

  request.active_weight_bytes = 7;
  geometry = resolve_llm_geometry(request);
  ASSERT_TRUE(geometry.valid) << geometry.reason_code;
  EXPECT_STREQ(classify_llm_traffic_payload(geometry), "kv_read_payload_dominant");

  EXPECT_STREQ(classify_llm_traffic_payload(LlmGeometry{}), "unavailable");
}

TEST(LlmMemoryJsonTest, JsonPeakEstimateIsZeroWhenOutputIsDisabled) {
  LlmMemoryConfig config = explicit_config(1);
  config.output_file.clear();
  const LlmJsonPeakEstimate estimate = calculate_llm_json_peak_estimate(config, LlmMemoryWorkPlan{});

  EXPECT_TRUE(estimate.valid);
  EXPECT_EQ(estimate.reason_code, LlmJsonReason::VALID);
  EXPECT_EQ(estimate.fixed_schema_bytes, 0u);
  EXPECT_EQ(estimate.input_string_bytes, 0u);
  EXPECT_EQ(estimate.measurement_record_bytes, 0u);
  EXPECT_EQ(estimate.worker_checksum_bytes, 0u);
  EXPECT_EQ(estimate.total_bytes, 0u);
}

TEST(LlmMemoryJsonTest, JsonPeakEstimateScalesWithMeasurementsAndWorkersAndIsAdmitted) {
  const LlmMemoryConfig config = explicit_config(3);
  LlmMemoryWorkPlanRequest request = plan_request(config);
  const LlmMemoryWorkPlan preliminary = build_llm_memory_work_plan(request);
  ASSERT_TRUE(preliminary.valid) << preliminary.reason_code;
  const LlmExecutorAuxiliaryEstimate executor = calculate_llm_executor_auxiliary_estimate(preliminary);
  const LlmRunnerAuxiliaryEstimate runner = calculate_llm_runner_auxiliary_estimate(config, preliminary);
  const LlmJsonPeakEstimate estimate = calculate_llm_json_peak_estimate(config, preliminary);
  ASSERT_TRUE(executor.valid);
  ASSERT_TRUE(runner.valid);
  ASSERT_TRUE(estimate.valid);

  EXPECT_GT(estimate.fixed_schema_bytes, 0u);
  EXPECT_GT(estimate.input_string_bytes, 0u);
  EXPECT_GT(estimate.measurement_record_bytes, 0u);
  EXPECT_GT(estimate.worker_checksum_bytes, estimate.measurement_record_bytes);
  EXPECT_EQ(estimate.total_bytes, estimate.fixed_schema_bytes + estimate.input_string_bytes +
                                      estimate.measurement_record_bytes + estimate.worker_checksum_bytes);

  const LlmMemoryWorkPlan admitted = admitted_plan(config);
  ASSERT_TRUE(admitted.valid) << admitted.reason_code;
  EXPECT_EQ(admitted.memory_budget.request.orchestration_auxiliary_bytes,
            executor.orchestration_auxiliary_bytes + runner.orchestration_auxiliary_bytes + estimate.total_bytes);
}

TEST(LlmMemoryJsonTest, JsonPeakEstimateRejectsCountArithmeticOverflow) {
  LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  config.loop_count = std::numeric_limits<size_t>::max();

  const LlmJsonPeakEstimate estimate = calculate_llm_json_peak_estimate(config, plan);
  EXPECT_FALSE(estimate.valid);
  EXPECT_EQ(estimate.reason_code, LlmJsonReason::PEAK_BYTES_OVERFLOW);
  EXPECT_EQ(estimate.total_bytes, 0u);
}

TEST(LlmMemoryJsonTest, ExactByteSeedAndChecksumIntegersAreCanonicalDecimalStrings) {
  const LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  const LlmMemoryResult result = complete_result(config, plan);
  LlmResultMetadata metadata = fixed_metadata(config, plan);
  metadata.available_memory_bytes = 0;
  metadata.available_memory_source = "unavailable";
  const OrderedJson document = build_llm_memory_json(config, plan, preparation_for(plan), metadata, result);

  const OrderedJson& measurement = document["measurements"][0];
  EXPECT_EQ(measurement["work_unit_kind"], "decode_step");
  EXPECT_EQ(measurement["kv_write_kind"], "none");
  EXPECT_TRUE(measurement["planned_work_units"].is_number_unsigned());
  EXPECT_TRUE(measurement["completed_work_units"].is_number_unsigned());
  const std::array<const char*, 15> decimal_measurement_fields = {
      "weight_read_bytes_per_work_unit",
      "kv_read_bytes_per_work_unit",
      "kv_write_bytes_per_work_unit",
      "effective_model_payload_bytes_per_work_unit",
      "layout_metadata_lookup_count_per_work_unit",
      "layout_metadata_read_bytes_per_work_unit",
      "accounted_bytes_per_work_unit",
      "planned_effective_model_payload_bytes",
      "completed_effective_model_payload_bytes",
      "planned_layout_metadata_lookup_count",
      "completed_layout_metadata_lookup_count",
      "planned_layout_metadata_read_bytes",
      "completed_layout_metadata_read_bytes",
      "planned_task_accounted_bytes",
      "completed_task_accounted_bytes"};
  for (const char* field : decimal_measurement_fields) {
    EXPECT_TRUE(measurement[field].is_string()) << field;
  }
  EXPECT_EQ(measurement["layout_metadata_lookup_count_per_work_unit"], "0");
  EXPECT_EQ(measurement["layout_metadata_read_bytes_per_work_unit"], "0");
  EXPECT_TRUE(measurement["synthetic_work_unit_latency_seconds"].is_number());
  EXPECT_TRUE(measurement["synthetic_memory_work_units_per_second"].is_number());
  EXPECT_TRUE(measurement["effective_model_payload_gb_s"].is_number());

  EXPECT_TRUE(document["resolved_plan"]["geometry"]["decode"].is_object());
  EXPECT_TRUE(document["resolved_plan"]["geometry"]["prefill"].is_null());
  EXPECT_TRUE(document["resolved_plan"]["layout"]["kv_block_tokens"].is_null());
  EXPECT_TRUE(document["resolved_plan"]["layout"]["block_table_bytes"].is_null());
  EXPECT_TRUE(document["resolved_plan"]["resources"]["weight_logical_bytes"].is_string());
  EXPECT_TRUE(document["resolved_plan"]["resources"]["block_table_bytes"].is_null());
  EXPECT_TRUE(document["resolved_plan"]["component_identities"]["permutation_version"].is_null());
  EXPECT_TRUE(document["resolved_plan"]["component_identities"]["msl_revision"].is_null());
  EXPECT_TRUE(document["resolved_plan"]["component_identities"]["msl_source_sha256"].is_null());
  EXPECT_TRUE(document["backend_evidence"]["cpu"].is_object());
  EXPECT_TRUE(document["backend_evidence"]["metal"].is_null());
  for (const char* field : {"resource_rounding_bytes", "transient_peak_bytes", "known_owned_peak_bytes",
                            "admitted_budget_bytes"}) {
    EXPECT_TRUE(document["memory_budget"][field].is_string()) << field;
  }

  EXPECT_EQ(document["seeds"]["base_seed_uint64_decimal"], "18446744073709551615");
  const OrderedJson& checksum = document["measurements"][0]["checksum"];
  ASSERT_EQ(checksum["status"], "valid");
  ASSERT_TRUE(checksum["checksum_valid"]);
  EXPECT_EQ(checksum["expected_worker_checksums"][0]["weight"]["state_a_uint64_decimal"], "18446744073709551615");
  EXPECT_EQ(checksum["expected_worker_checksums"][0]["weight"]["state_b_uint64_decimal"], "9007199254740993");
  EXPECT_EQ(checksum["expected_worker_checksums"][0]["weight"]["exact_bytes_read"], "9007199254741093");
  EXPECT_TRUE(checksum["expected_worker_checksums"][0]["weight"]["span_count_uint64_decimal"].is_string());
  EXPECT_EQ(checksum["actual_run_checksum"]["state_a_uint64_decimal"], "18446744073709551615");
  EXPECT_TRUE(document["counters"]["planned_work_units"].is_string());
  EXPECT_TRUE(document["counters"]["completed_effective_model_payload_bytes"].is_string());
  EXPECT_TRUE(document["environment"]["available_memory_bytes"].is_null());
  EXPECT_EQ(document["environment"]["available_memory_source"], "unavailable");
}

TEST(LlmMemoryJsonTest, InterruptedRunnerSerializesUnavailableMetricsExecutionQosAndChecksumAsNull) {
  const LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  LlmRunnerHooks hooks;
  hooks.stop_requested = []() { return true; };
  LlmMemoryResult result;
  const LlmTaskExecutor executor = [](const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan&,
                                      const LlmRunnerTaskContext&) { return successful_execution(model_plan); };
  ASSERT_EQ(run_llm_memory_suite(config, plan, executor, result, hooks), EXIT_SUCCESS);
  ASSERT_EQ(result.status, LlmRunStatus::Interrupted);
  ASSERT_FALSE(result.measurements.empty());

  const OrderedJson document =
      build_llm_memory_json(config, plan, preparation_for(plan), fixed_metadata(config, plan), result);
  EXPECT_EQ(document["status"], "interrupted");
  EXPECT_FALSE(document["results_complete"]);
  EXPECT_FALSE(document["conclusions_valid"]);
  const OrderedJson& serialized = document["measurements"][0];
  EXPECT_EQ(serialized["status"], "interrupted");
  EXPECT_EQ(serialized["reason_code"], LlmRunnerReason::INTERRUPTION_BEFORE_TASK);
  EXPECT_FALSE(serialized["attempted"]);
  EXPECT_TRUE(serialized["qos_successful_workers"].is_null());
  EXPECT_TRUE(serialized["qos_failed_workers"].is_null());
  EXPECT_EQ(serialized["completed_work_units"], 0u);
  EXPECT_EQ(serialized["completed_effective_model_payload_bytes"], "0");
  EXPECT_TRUE(serialized["elapsed_seconds"].is_null());
  EXPECT_TRUE(serialized["synthetic_work_unit_latency_seconds"].is_null());
  EXPECT_TRUE(serialized["synthetic_memory_work_units_per_second"].is_null());
  EXPECT_TRUE(serialized["effective_model_payload_gb_s"].is_null());
  EXPECT_EQ(serialized["execution"]["status"], "not_run");
  EXPECT_TRUE(serialized["execution"]["valid"].is_null());
  EXPECT_EQ(serialized["checksum"]["status"], "not_evaluated");
  EXPECT_EQ(serialized["checksum"]["reason_code"], LlmRunnerReason::INTERRUPTION_BEFORE_TASK);
  EXPECT_TRUE(serialized["checksum"]["checksum_valid"].is_null());
  EXPECT_TRUE(serialized["checksum"]["expected_worker_checksums"].is_null());
  EXPECT_TRUE(serialized["checksum"]["actual_run_checksum"].is_null());
  // The immutable frozen plan remains available despite the interruption.
  EXPECT_TRUE(serialized["planned_effective_model_payload_bytes"].is_string());
}

TEST(LlmMemoryJsonTest, ExecutorExceptionUsesRunnerReasonAndNullUnavailableExecutionEvidence) {
  const LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  const LlmTaskExecutor executor = [](const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan&,
                                      const LlmRunnerTaskContext& context) {
    if (context.kind == LlmRunnerTaskKind::Measurement) {
      throw std::runtime_error("injected JSON exception path");
    }
    return successful_execution(model_plan);
  };
  LlmMemoryResult result;
  ASSERT_EQ(run_llm_memory_suite(config, plan, executor, result), EXIT_FAILURE);
  ASSERT_EQ(result.status, LlmRunStatus::Failed);

  const OrderedJson document =
      build_llm_memory_json(config, plan, preparation_for(plan), fixed_metadata(config, plan), result);
  EXPECT_EQ(document["status"], "failed");
  EXPECT_EQ(document["reason_code"], LlmRunnerReason::RUNNER_EXCEPTION);
  EXPECT_EQ(document["diagnostic"], "injected JSON exception path");
  const OrderedJson& measurement = document["measurements"][0];
  EXPECT_TRUE(measurement["attempted"]);
  EXPECT_EQ(measurement["status"], "failed");
  EXPECT_EQ(measurement["reason_code"], LlmRunnerReason::RUNNER_EXCEPTION);
  EXPECT_TRUE(measurement["qos_successful_workers"].is_null());
  EXPECT_TRUE(measurement["qos_failed_workers"].is_null());
  EXPECT_EQ(measurement["execution"]["status"], "unavailable");
  EXPECT_EQ(measurement["execution"]["reason_code"], LlmRunnerReason::RUNNER_EXCEPTION);
  EXPECT_TRUE(measurement["execution"]["valid"].is_null());
  EXPECT_TRUE(measurement["execution"]["requested_workers"].is_null());
  EXPECT_TRUE(measurement["execution"]["kernel_succeeded"].is_null());
  EXPECT_EQ(measurement["checksum"]["status"], "not_evaluated");
  EXPECT_EQ(measurement["checksum"]["reason_code"], LlmRunnerReason::RUNNER_EXCEPTION);
  EXPECT_TRUE(measurement["checksum"]["checksum_valid"].is_null());
}

TEST(LlmMemoryJsonTest, ExcludedExecutorExceptionUsesRunnerReasonAndNullUnavailableExecutionEvidence) {
  const LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  const LlmTaskExecutor executor = [](const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan&,
                                      const LlmRunnerTaskContext& context) {
    if (context.kind == LlmRunnerTaskKind::Warmup) {
      throw std::runtime_error("injected excluded JSON exception path");
    }
    return successful_execution(model_plan);
  };
  LlmMemoryResult result;
  ASSERT_EQ(run_llm_memory_suite(config, plan, executor, result), EXIT_FAILURE);

  const OrderedJson document =
      build_llm_memory_json(config, plan, preparation_for(plan), fixed_metadata(config, plan), result);
  const OrderedJson& execution = document["calibration"]["attempts"]["weights_only"][0]["execution"];
  EXPECT_EQ(execution["status"], "unavailable");
  EXPECT_EQ(execution["reason_code"], LlmRunnerReason::RUNNER_EXCEPTION);
  EXPECT_TRUE(execution["valid"].is_null());
  EXPECT_TRUE(execution["requested_workers"].is_null());
  EXPECT_TRUE(execution["kernel_succeeded"].is_null());
  EXPECT_EQ(execution["checksum"]["status"], "not_evaluated");
  EXPECT_EQ(execution["checksum"]["reason_code"], LlmRunnerReason::RUNNER_EXCEPTION);
  EXPECT_TRUE(execution["checksum"]["checksum_valid"].is_null());
}

TEST(LlmMemoryJsonTest, InvalidElapsedMeasurementLeavesChecksumEvidenceNotEvaluatedAndNull) {
  const LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  const LlmTaskExecutor executor = [](const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan&,
                                      const LlmRunnerTaskContext& context) {
    LlmExecutorResult execution = successful_execution(model_plan);
    if (context.kind == LlmRunnerTaskKind::Measurement) {
      execution.valid = false;
      execution.reason_code = LlmExecutorReason::INVALID_ELAPSED_TIME;
      execution.elapsed_seconds = 0.0;
      execution.checksum_evaluated = false;
      execution.checksum_valid = false;
    }
    return execution;
  };
  LlmMemoryResult result;
  ASSERT_EQ(run_llm_memory_suite(config, plan, executor, result), EXIT_FAILURE);

  const OrderedJson document =
      build_llm_memory_json(config, plan, preparation_for(plan), fixed_metadata(config, plan), result);
  const OrderedJson& measurement = document["measurements"][0];
  EXPECT_EQ(measurement["status"], "invalid");
  EXPECT_EQ(measurement["reason_code"], LlmExecutorReason::INVALID_ELAPSED_TIME);
  EXPECT_EQ(measurement["execution"]["status"], "invalid");
  EXPECT_EQ(measurement["execution"]["reason_code"], LlmExecutorReason::INVALID_ELAPSED_TIME);
  EXPECT_FALSE(measurement["execution"]["valid"]);
  EXPECT_EQ(measurement["checksum"]["status"], "not_evaluated");
  EXPECT_EQ(measurement["checksum"]["reason_code"], LlmExecutorReason::INVALID_ELAPSED_TIME);
  EXPECT_TRUE(measurement["checksum"]["checksum_valid"].is_null());
  EXPECT_TRUE(measurement["checksum"]["expected_worker_checksums"].is_null());
  EXPECT_TRUE(measurement["checksum"]["actual_worker_checksums"].is_null());
  EXPECT_TRUE(measurement["checksum"]["expected_run_checksum"].is_null());
  EXPECT_TRUE(measurement["checksum"]["actual_run_checksum"].is_null());
}

TEST(LlmMemoryJsonTest, InvalidElapsedExcludedTaskLeavesCompactChecksumEvidenceNotEvaluatedAndNull) {
  const LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  const LlmTaskExecutor executor = [](const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan&,
                                      const LlmRunnerTaskContext& context) {
    LlmExecutorResult execution = successful_execution(model_plan);
    if (context.kind == LlmRunnerTaskKind::Warmup) {
      execution.valid = false;
      execution.reason_code = LlmExecutorReason::INVALID_ELAPSED_TIME;
      execution.elapsed_seconds = 0.0;
      execution.checksum_evaluated = false;
      execution.checksum_valid = false;
    }
    return execution;
  };
  LlmMemoryResult result;
  ASSERT_EQ(run_llm_memory_suite(config, plan, executor, result), EXIT_FAILURE);

  const OrderedJson document =
      build_llm_memory_json(config, plan, preparation_for(plan), fixed_metadata(config, plan), result);
  const OrderedJson& execution = document["calibration"]["attempts"]["weights_only"][0]["execution"];
  EXPECT_EQ(execution["status"], "invalid");
  EXPECT_EQ(execution["reason_code"], LlmExecutorReason::INVALID_ELAPSED_TIME);
  EXPECT_FALSE(execution["valid"]);
  EXPECT_EQ(execution["checksum"]["status"], "not_evaluated");
  EXPECT_EQ(execution["checksum"]["reason_code"], LlmExecutorReason::INVALID_ELAPSED_TIME);
  EXPECT_TRUE(execution["checksum"]["checksum_valid"].is_null());
  EXPECT_TRUE(execution["checksum"]["expected_run_checksum"].is_null());
  EXPECT_TRUE(execution["checksum"]["actual_run_checksum"].is_null());
}

TEST(LlmMemoryJsonTest, MalformedExcludedChecksumCardinalityDoesNotPublishCompactChecksumEvidence) {
  const LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  ASSERT_GE(plan.effective_workers, 2u);
  const LlmTaskExecutor executor = [](const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan&,
                                      const LlmRunnerTaskContext& context) {
    LlmExecutorResult execution = successful_execution(model_plan);
    if (context.kind == LlmRunnerTaskKind::Warmup) {
      execution.expected_checksums.resize(1);
      execution.actual_checksums.resize(1);
    }
    return execution;
  };
  LlmMemoryResult result;
  ASSERT_EQ(run_llm_memory_suite(config, plan, executor, result), EXIT_FAILURE);

  const OrderedJson document =
      build_llm_memory_json(config, plan, preparation_for(plan), fixed_metadata(config, plan), result);
  const OrderedJson& attempt = document["calibration"]["attempts"]["weights_only"][0];
  EXPECT_FALSE(attempt["valid"]);
  EXPECT_EQ(attempt["reason_code"], LlmExecutorReason::INVALID_RESOURCES);
  const OrderedJson& execution = attempt["execution"];
  EXPECT_EQ(execution["status"], "invalid");
  EXPECT_EQ(execution["reason_code"], LlmExecutorReason::INVALID_RESOURCES);
  EXPECT_FALSE(execution["valid"]);
  EXPECT_EQ(execution["checksum"]["status"], "not_evaluated");
  EXPECT_EQ(execution["checksum"]["reason_code"], LlmExecutorReason::INVALID_RESOURCES);
  EXPECT_TRUE(execution["checksum"]["checksum_valid"].is_null());
  EXPECT_TRUE(execution["checksum"]["expected_run_checksum"].is_null());
  EXPECT_TRUE(execution["checksum"]["actual_run_checksum"].is_null());
}

TEST(LlmMemoryJsonTest, ChecksumMismatchSerializesEvaluatedFalseInsteadOfMissingNull) {
  const LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  const LlmTaskExecutor executor = [](const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan&,
                                      const LlmRunnerTaskContext& context) {
    LlmExecutorResult execution = successful_execution(model_plan);
    if (context.kind == LlmRunnerTaskKind::Measurement) {
      execution.checksum_valid = false;
    }
    return execution;
  };
  LlmMemoryResult result;
  ASSERT_EQ(run_llm_memory_suite(config, plan, executor, result), EXIT_FAILURE);

  const OrderedJson document =
      build_llm_memory_json(config, plan, preparation_for(plan), fixed_metadata(config, plan), result);
  const OrderedJson& measurement = document["measurements"][0];
  EXPECT_EQ(measurement["status"], "invalid");
  EXPECT_EQ(measurement["reason_code"], LlmExecutorReason::CHECKSUM_MISMATCH);
  EXPECT_EQ(measurement["checksum"]["status"], "invalid");
  EXPECT_EQ(measurement["checksum"]["reason_code"], LlmExecutorReason::CHECKSUM_MISMATCH);
  EXPECT_FALSE(measurement["checksum"]["checksum_valid"]);
  EXPECT_TRUE(measurement["checksum"]["expected_worker_checksums"].is_array());
  EXPECT_TRUE(measurement["checksum"]["actual_worker_checksums"].is_array());
}

TEST(LlmMemoryJsonTest, MeasurementCheckpointSerializesPartialStatusAndUnavailableTailContract) {
  const LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  const LlmTaskExecutor executor = [](const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan&,
                                      const LlmRunnerTaskContext&) { return successful_execution(model_plan); };
  LlmMemoryResult partial_snapshot;
  bool captured_partial_snapshot = false;
  LlmRunnerHooks hooks;
  hooks.checkpoint = [&](const LlmMemoryResult& checkpoint_result, LlmCheckpointKind kind) {
    if (!captured_partial_snapshot && kind == LlmCheckpointKind::MeasurementTerminal) {
      partial_snapshot = checkpoint_result;
      captured_partial_snapshot = true;
    }
    return EXIT_SUCCESS;
  };
  LlmMemoryResult result;
  ASSERT_EQ(run_llm_memory_suite(config, plan, executor, result, hooks), EXIT_SUCCESS);
  ASSERT_TRUE(captured_partial_snapshot);

  const OrderedJson document =
      build_llm_memory_json(config, plan, preparation_for(plan), fixed_metadata(config, plan), partial_snapshot);
  EXPECT_EQ(document["status"], "partial");
  EXPECT_EQ(document["reason_code"], LlmRunnerReason::PARTIAL_RESULTS);
  EXPECT_FALSE(document["interruption_requested"]);
  EXPECT_FALSE(document["results_complete"]);
  EXPECT_FALSE(document["conclusions_valid"]);
  EXPECT_FALSE(document["scenario_order_balance_complete"]);
  EXPECT_EQ(document["counters"]["planned_measurements"], 3u);
  EXPECT_EQ(document["counters"]["attempted_measurements"], 1u);
  EXPECT_EQ(document["counters"]["terminal_measurements"], 1u);
  EXPECT_EQ(document["counters"]["measured_measurements"], 1u);
  EXPECT_EQ(document["checkpoint_lifecycle"]["logical_checkpoint_attempts"], 1u);
  EXPECT_EQ(document["checkpoint_lifecycle"]["successful_logical_checkpoints"], 1u);
  EXPECT_FALSE(document["checkpoint_lifecycle"]["terminal_checkpoint_attempted"]);
  EXPECT_FALSE(document["checkpoint_lifecycle"]["terminal_checkpoint_completed"]);

  size_t measured_count = 0;
  size_t not_run_count = 0;
  for (const OrderedJson& measurement : document["measurements"]) {
    if (measurement["status"] == "measured") {
      ++measured_count;
      EXPECT_EQ(measurement["reason_code"], "measured");
      EXPECT_TRUE(measurement["attempted"]);
      EXPECT_TRUE(measurement["completed_work_units"].is_number_unsigned());
      EXPECT_TRUE(measurement["elapsed_seconds"].is_number());
      EXPECT_EQ(measurement["execution"]["status"], "valid");
      EXPECT_EQ(measurement["checksum"]["status"], "valid");
      EXPECT_TRUE(measurement["checksum"]["checksum_valid"]);
      continue;
    }

    ASSERT_EQ(measurement["status"], "not_run");
    ++not_run_count;
    EXPECT_EQ(measurement["reason_code"], "not-run");
    EXPECT_FALSE(measurement["attempted"]);
    EXPECT_TRUE(measurement["qos_successful_workers"].is_null());
    EXPECT_EQ(measurement["completed_work_units"], 0u);
    EXPECT_EQ(measurement["completed_effective_model_payload_bytes"], "0");
    EXPECT_TRUE(measurement["elapsed_seconds"].is_null());
    EXPECT_TRUE(measurement["effective_model_payload_gb_s"].is_null());
    EXPECT_EQ(measurement["execution"]["status"], "not_run");
    EXPECT_EQ(measurement["execution"]["reason_code"], "not-run");
    EXPECT_TRUE(measurement["execution"]["valid"].is_null());
    EXPECT_TRUE(measurement["execution"]["requested_workers"].is_null());
    EXPECT_EQ(measurement["checksum"]["status"], "not_evaluated");
    EXPECT_EQ(measurement["checksum"]["reason_code"], "not-run");
    EXPECT_TRUE(measurement["checksum"]["checksum_valid"].is_null());
  }
  EXPECT_EQ(measured_count, 1u);
  EXPECT_EQ(not_run_count, 2u);
}

TEST(LlmMemoryJsonTest, CheckpointFailureRetainsMeasuredPrefixAndNullFailedTailWithoutRetry) {
  const LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  const LlmTaskExecutor executor = [](const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan&,
                                      const LlmRunnerTaskContext&) { return successful_execution(model_plan); };
  LlmRunnerHooks hooks;
  hooks.checkpoint = [](const LlmMemoryResult&, LlmCheckpointKind) { return EXIT_FAILURE; };
  LlmMemoryResult result;
  ASSERT_EQ(run_llm_memory_suite(config, plan, executor, result, hooks), EXIT_FAILURE);

  const OrderedJson document =
      build_llm_memory_json(config, plan, preparation_for(plan), fixed_metadata(config, plan), result);
  EXPECT_EQ(document["status"], "failed");
  EXPECT_EQ(document["reason_code"], LlmRunnerReason::CHECKPOINT_WRITE_FAILED);
  EXPECT_TRUE(document["checkpoint_lifecycle"]["checkpoint_failed"]);
  EXPECT_EQ(document["checkpoint_lifecycle"]["logical_checkpoint_attempts"], 1u);
  EXPECT_EQ(document["checkpoint_lifecycle"]["successful_logical_checkpoints"], 0u);
  EXPECT_FALSE(document["checkpoint_lifecycle"]["terminal_checkpoint_attempted"]);
  EXPECT_FALSE(document["checkpoint_lifecycle"]["terminal_checkpoint_completed"]);
  EXPECT_EQ(document["measurements"][0]["status"], "measured");
  EXPECT_EQ(document["measurements"][1]["status"], "failed");
  EXPECT_TRUE(document["measurements"][1]["qos_successful_workers"].is_null());
  EXPECT_TRUE(document["measurements"][1]["effective_model_payload_gb_s"].is_null());
  EXPECT_TRUE(document["measurements"][1]["checksum"]["checksum_valid"].is_null());
}

TEST(LlmMemoryJsonTest, QualityWarningsMergeAndDeduplicateInStableConsoleAgreementOrder) {
  const LlmMemoryConfig config = explicit_config();
  const LlmMemoryWorkPlan plan = admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  LlmMemoryResult result = complete_result(config, plan);
  result.quality_warnings = {"mixed-high-cv", "environment-not-nominal", "mixed-high-cv"};
  for (LlmMeasurementState& measurement : result.measurements) {
    measurement.duration_quality = "above-target-single-work-unit";
  }
  result.measurements[0].qos_failed_workers = 1;

  LlmResultMetadata metadata = fixed_metadata(config, plan);
  metadata.environment_end.thermal_state = "serious";
  metadata.main_thread_qos = {true, false, 1};
  metadata.l2_data_cache_bytes = 2 * Constants::BYTES_PER_MB;

  const OrderedJson document = build_llm_memory_json(config, plan, preparation_for(plan), metadata, result);
  EXPECT_EQ(
      json_string_array(document["quality_warnings"]),
      (std::vector<std::string>{"mixed-high-cv", "environment-not-nominal", "main-thread-qos-not-applied",
                                "worker-qos-not-applied", "weight-working-set-cache-dominant",
                                "kv-working-set-cache-dominant", "weights_only-duration-above-target-single-work-unit",
                                "kv_only-duration-above-target-single-work-unit",
                                "mixed-duration-above-target-single-work-unit"}));
}

TEST(LlmMemoryJsonTest, LoopRecordsExposeOnlyRealizedPrefixAndAllMeasurementIndexes) {
  const LlmMemoryConfig config = explicit_config(1);
  const LlmMemoryWorkPlan plan = admitted_plan(config);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  LlmMemoryResult result = complete_result(config, plan);
  result.loops[0].realized_order_count = 1;

  const OrderedJson document =
      build_llm_memory_json(config, plan, preparation_for(plan), fixed_metadata(config, plan), result);
  const OrderedJson& loop = document["loop_records"][0];
  EXPECT_EQ(loop["planned_order"].size(), kLlmScenarioCount);
  EXPECT_EQ(loop["realized_order"].size(), 1u);
  EXPECT_EQ(loop["realized_order"][0], "weights_only");
  EXPECT_EQ(loop["measurement_indexes"].size(), kLlmScenarioCount);
  EXPECT_EQ(loop["measurement_indexes"], (OrderedJson::array({0, 1, 2})));
}

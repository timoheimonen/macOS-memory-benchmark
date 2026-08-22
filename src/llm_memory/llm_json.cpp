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

/**
 * @file llm_json.cpp
 * @brief Auditable backend-neutral JSON schema v1 for LLM memory profiles
 */

#include "llm_memory/llm_json.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/config/constants.h"
#include "core/config/version.h"
#include "utils/json_utils.h"
#include "utils/numeric_utils.h"

namespace {

using OrderedJson = nlohmann::ordered_json;

constexpr std::array<LlmScenario, kLlmScenarioCount> kScenarios = {LlmScenario::WeightsOnly, LlmScenario::KvOnly,
                                                                   LlmScenario::Mixed};

// Reserve one live ordered-JSON DOM and its serialized transport form. The
// deliberately generous per-record allowances keep admission independent of
// allocator and standard-library implementation details while preserving the
// count/worker scaling of the schema.
constexpr size_t kJsonFixedSchemaPeakBytes = 2 * Constants::BYTES_PER_MB;
constexpr size_t kJsonInputStringExpansionFactor = 16;
constexpr size_t kJsonMeasurementPeakBytes = 64 * 1024;
constexpr size_t kJsonWorkerChecksumPeakBytes = 64 * 1024;

size_t scenario_index(LlmScenario scenario) noexcept {
  switch (scenario) {
    case LlmScenario::WeightsOnly:
      return 0;
    case LlmScenario::KvOnly:
      return 1;
    case LlmScenario::Mixed:
      return 2;
  }
  return kLlmScenarioCount;
}

template <typename Integer>
OrderedJson decimal_string(Integer value) {
  return OrderedJson(std::to_string(value));
}

template <typename Integer>
OrderedJson decimal_or_null(Integer value, bool available) {
  return available ? decimal_string(value) : OrderedJson(nullptr);
}

template <typename Integer>
OrderedJson number_or_null(Integer value, bool available) {
  return available ? OrderedJson(value) : OrderedJson(nullptr);
}

OrderedJson finite_or_null(double value) { return std::isfinite(value) ? OrderedJson(value) : OrderedJson(nullptr); }

OrderedJson positive_finite_or_null(double value, bool available = true) {
  return available && std::isfinite(value) && value > 0.0 ? OrderedJson(value) : OrderedJson(nullptr);
}

OrderedJson optional_finite_or_null(const std::optional<double>& value) {
  return value.has_value() && std::isfinite(*value) ? OrderedJson(*value) : OrderedJson(nullptr);
}

OrderedJson non_empty_or_null(const std::string& value) {
  return value.empty() ? OrderedJson(nullptr) : OrderedJson(value);
}

OrderedJson optional_string_or_null(const std::optional<std::string>& value) {
  return value.has_value() ? OrderedJson(*value) : OrderedJson(nullptr);
}

OrderedJson argv_json(const std::vector<std::string>& argv) {
  OrderedJson output = OrderedJson::array();
  for (const std::string& argument : argv) {
    output.push_back(argument);
  }
  return output;
}

OrderedJson configuration_json(const LlmMemoryConfig& config) {
  OrderedJson resolved_sources;
  resolved_sources["backend"] = "default";
  resolved_sources["phase"] = "default";
  resolved_sources["kv_layout"] = "default";
  resolved_sources["workers"] = config.user_specified_workers ? "explicit" : "detected";
  resolved_sources["iterations"] = config.user_specified_iterations ? "explicit" : "automatic";
  resolved_sources["seed"] = config.user_specified_seed ? "explicit" : "generated";

  OrderedJson output;
  output["backend"] = llm_memory_backend_to_string(config.backend);
  output["phase"] = llm_phase_to_string(config.phase);
  output["kv_layout"] = llm_kv_layout_to_string(config.kv_layout);
  output["weight_size_mb"] = config.weight_size_mb;
  output["layer_count"] = config.layer_count;
  output["query_head_count"] = config.query_head_count;
  output["kv_head_count"] = config.kv_head_count;
  output["head_dimension"] = config.head_dimension;
  output["kv_element_bytes"] = decimal_string(config.kv_element_bytes);
  output["visible_context_tokens"] = config.visible_context_tokens;
  output["batch_size"] = config.batch_size;
  output["requested_workers"] = config.requested_workers;
  output["available_workers"] = config.available_workers;
  output["worker_source"] = config.user_specified_workers ? "user" : "detected";
  output["iterations"] = number_or_null(config.iterations, config.user_specified_iterations);
  output["work_policy"] = config.user_specified_iterations ? "explicit_fixed_work" : "automatic_calibration";
  output["loop_count"] = config.loop_count;
  output["base_seed_uint64_decimal"] = decimal_string(config.seed);
  output["seed_source"] = config.user_specified_seed ? "user" : "generated";
  // Empty is an exact console-only target, not missing configuration.
  output["output_file"] = config.output_file;
  output["argv"] = argv_json(config.argv);
  output["resolved_sources"] = std::move(resolved_sources);
  return output;
}

OrderedJson methodology_json(const LlmMemoryWorkPlan& plan) {
  OrderedJson exclusions = OrderedJson::array();
  exclusions.push_back("mapping-allocation");
  exclusions.push_back("buffer-initialization-and-pre-touch");
  exclusions.push_back("descriptor-materialization-and-validation");
  exclusions.push_back("worker-thread-creation-and-qos");
  exclusions.push_back("same-shape-warmup");
  exclusions.push_back("calibration-attempts");
  exclusions.push_back("checksum-reference-generation-and-validation");
  exclusions.push_back("worker-join-and-json-serialization");

  OrderedJson output;
  output["methodology_version"] = plan.methodology_version;
  output["backend"] = llm_memory_backend_to_string(plan.backend);
  output["phase"] = llm_phase_to_string(plan.phase);
  output["kv_layout"] = llm_kv_layout_to_string(plan.kv_layout);
  output["work_unit_kind"] = llm_work_unit_kind_to_string(plan.work_unit_kind);
  output["weight_passes_per_work_unit"] = plan.weight_passes_per_work_unit;
  output["kv_replay_factor"] = plan.kv_replay_factor;
  output["schedule_version"] = plan.component_identities.schedule_version;
  output["warmup_policy"] = "same-shape-excluded-steady-state-warm-memory";
  output["context_policy"] = "fixed-visible-context-including-current-token-slot";
  output["scenario_order_policy"] = "cyclic-rotation-across-count-loops";
  output["timing_policy"] = "synchronized-start-to-last-worker-completion-per-scenario-task";
  output["cache_policy"] = "regular-cacheable-no-explicit-flush-between-scenarios";
  output["calibration_policy"] = "per-scenario-automatic-or-explicit-frozen-before-loop-zero";
  output["calibration_target_seconds"] = Constants::LLM_CALIBRATION_TARGET_SECONDS;
  output["calibration_min_seconds"] = Constants::LLM_CALIBRATION_MIN_SECONDS;
  output["calibration_max_seconds"] = Constants::LLM_CALIBRATION_MAX_SECONDS;
  output["calibration_max_corrections"] = Constants::LLM_CALIBRATION_MAX_CORRECTIONS;
  output["calibration_min_pilot_accounted_bytes"] = decimal_string(Constants::LLM_CALIBRATION_MIN_PILOT_BYTES);
  output["maximum_work_units_per_measurement"] = Constants::LLM_MAX_WORK_UNITS_PER_MEASUREMENT;
  output["maximum_accounted_bytes_per_task"] = decimal_string(Constants::LLM_MAX_ACCOUNTED_BYTES_PER_TASK);
  output["repeatability_cv_warning_threshold_pct"] = Constants::LLM_STREAMING_CV_WARNING_PCT;
  output["calibration_excluded_from_results"] = true;
  output["timed_region_exclusions"] = std::move(exclusions);
  output["resource_abi_version"] = plan.component_identities.resource_abi_version;
  output["buffer_pattern_version"] = plan.component_identities.buffer_pattern_version;
  output["write_pattern_version"] = plan.component_identities.write_pattern_version;
  output["checksum_pattern_version"] = plan.component_identities.checksum_pattern_version;
  return output;
}

OrderedJson geometry_json(const LlmGeometry& geometry) {
  const bool available = geometry.valid;
  OrderedJson decode = nullptr;
  if (geometry.decode.has_value()) {
    decode = OrderedJson{{"visible_context_tokens", geometry.decode->visible_context_tokens}};
  }

  OrderedJson prefill = nullptr;
  if (geometry.prefill.has_value()) {
    prefill = OrderedJson{{"prompt_tokens", geometry.prefill->prompt_tokens},
                          {"attention_query_tile_tokens", geometry.prefill->attention_query_tile_tokens},
                          {"tile_count", decimal_string(geometry.prefill->tile_count)},
                          {"attention_prefix_token_visits_per_sequence",
                           decimal_string(geometry.prefill->attention_prefix_token_visits_per_sequence)},
                          {"causal_token_pairs_per_sequence",
                           decimal_string(geometry.prefill->causal_token_pairs_per_sequence)},
                          {"logical_attention_pairs", decimal_string(geometry.prefill->logical_attention_pairs)},
                          {"logical_attention_fma_terms",
                           decimal_string(geometry.prefill->logical_attention_fma_terms)}};
  }

  OrderedJson output;
  output["valid"] = geometry.valid;
  output["reason_code"] = geometry.reason_code;
  output["phase"] = llm_phase_to_string(geometry.phase);
  output["work_unit_kind"] = llm_work_unit_kind_to_string(geometry.work_unit_kind);
  output["decode"] = std::move(decode);
  output["prefill"] = std::move(prefill);
  output["attention_kind"] =
      available ? OrderedJson(llm_attention_kind_to_string(geometry.attention_kind)) : OrderedJson(nullptr);
  output["active_weight_bytes_per_work_unit"] = decimal_or_null(geometry.active_weight_bytes_per_work_unit, available);
  output["layer_count"] = number_or_null(geometry.layer_count, available);
  output["query_head_count"] = number_or_null(geometry.query_head_count, available);
  output["kv_head_count"] = number_or_null(geometry.kv_head_count, available);
  output["query_heads_per_kv_head"] = number_or_null(geometry.query_heads_per_kv_head, available);
  output["head_dimension"] = number_or_null(geometry.head_dimension, available);
  output["kv_element_bytes"] = decimal_or_null(geometry.kv_element_bytes, available);
  output["batch_size"] = number_or_null(geometry.batch_size, available);
  output["kv_vector_bytes"] = decimal_or_null(geometry.kv_vector_bytes, available);
  output["k_or_v_record_bytes_per_layer"] = decimal_or_null(geometry.k_or_v_record_bytes_per_layer, available);
  output["kv_record_bytes_per_layer"] = decimal_or_null(geometry.kv_record_bytes_per_layer, available);
  output["kv_bytes_per_visible_token"] = decimal_or_null(geometry.kv_bytes_per_visible_token, available);
  output["k_or_v_sequence_visible_bytes"] = decimal_or_null(geometry.k_or_v_sequence_visible_bytes, available);
  output["k_mapping_bytes"] = decimal_or_null(geometry.k_mapping_bytes, available);
  output["v_mapping_bytes"] = decimal_or_null(geometry.v_mapping_bytes, available);
  output["kv_capacity_bytes"] = decimal_or_null(geometry.kv_capacity_bytes, available);
  output["weight_read_bytes_per_work_unit"] = decimal_or_null(geometry.weight_read_bytes_per_work_unit, available);
  output["kv_read_bytes_per_work_unit"] = decimal_or_null(geometry.kv_read_bytes_per_work_unit, available);
  output["kv_write_bytes_per_work_unit"] = decimal_or_null(geometry.kv_write_bytes_per_work_unit, available);
  output["kv_only_effective_model_payload_bytes_per_work_unit"] =
      decimal_or_null(geometry.kv_only_effective_model_payload_bytes_per_work_unit, available);
  output["mixed_effective_model_payload_bytes_per_work_unit"] =
      decimal_or_null(geometry.mixed_effective_model_payload_bytes_per_work_unit, available);
  output["total_data_mapping_bytes"] = decimal_or_null(geometry.total_data_mapping_bytes, available);
  output["traffic_crossover_numerator"] = decimal_or_null(geometry.traffic_crossover_numerator, available);
  output["traffic_crossover_denominator"] = decimal_or_null(geometry.traffic_crossover_denominator, available);
  output["traffic_crossover_context_tokens"] =
      available ? finite_or_null(geometry.traffic_crossover_context_tokens) : OrderedJson(nullptr);
  return output;
}

OrderedJson layout_json(const LlmMemoryWorkPlan& plan) {
  OrderedJson output;
  output["kv_layout"] = llm_kv_layout_to_string(plan.kv_layout);
  output["kv_block_tokens"] = nullptr;
  output["blocks_per_sequence"] = nullptr;
  output["physical_blocks_per_layer"] = nullptr;
  output["last_block_tokens"] = nullptr;
  output["last_block_valid_bytes"] = nullptr;
  output["block_table_entries"] = nullptr;
  output["block_table_bytes"] = nullptr;
  output["permutation_domain_uint64_hex"] = nullptr;
  output["permutation_seed_uint64_decimal"] = nullptr;
  output["permutation_algorithm_version"] = nullptr;
  output["permutation_sha256"] = nullptr;
  return output;
}

OrderedJson resolved_resources_json(const LlmMemoryWorkPlan& plan) {
  const LlmGeometry& geometry = plan.geometry;
  OrderedJson output;
  output["weight_logical_bytes"] = decimal_string(geometry.active_weight_bytes_per_work_unit);
  output["k_logical_bytes"] = decimal_string(geometry.k_mapping_bytes);
  output["v_logical_bytes"] = decimal_string(geometry.v_mapping_bytes);
  output["k_physical_length_bytes"] = decimal_string(geometry.k_mapping_bytes);
  output["v_physical_length_bytes"] = decimal_string(geometry.v_mapping_bytes);
  output["k_layout_padding_bytes"] = decimal_string(0);
  output["v_layout_padding_bytes"] = decimal_string(0);
  output["block_table_bytes"] = nullptr;
  return output;
}

OrderedJson component_identities_json(const LlmComponentIdentities& components) {
  OrderedJson output;
  output["logical_profile_version"] = components.logical_profile_version;
  output["kv_layout_version"] = components.kv_layout_version;
  output["permutation_version"] = optional_string_or_null(components.permutation_version);
  output["backend_executor_version"] = components.backend_executor_version;
  output["resource_abi_version"] = components.resource_abi_version;
  output["schedule_version"] = components.schedule_version;
  output["timer_policy_version"] = components.timer_policy_version;
  output["buffer_pattern_version"] = components.buffer_pattern_version;
  output["write_pattern_version"] = components.write_pattern_version;
  output["checksum_pattern_version"] = components.checksum_pattern_version;
  output["msl_revision"] = optional_string_or_null(components.msl_revision);
  output["msl_source_sha256"] = optional_string_or_null(components.msl_source_sha256);
  output["identity"] = components.identity;
  return output;
}

OrderedJson statistics_json(const DescriptiveStatistics& statistics, bool available) {
  if (!available) {
    return nullptr;
  }
  OrderedJson output;
  output["sample_count"] = statistics.sample_count;
  output["average"] = finite_or_null(statistics.average);
  output["min"] = finite_or_null(statistics.min);
  output["max"] = finite_or_null(statistics.max);
  output["median"] = finite_or_null(statistics.median);
  output["p90"] = finite_or_null(statistics.p90);
  output["p95"] = finite_or_null(statistics.p95);
  output["p99"] = finite_or_null(statistics.p99);
  output["stddev"] = finite_or_null(statistics.stddev);
  output["coefficient_of_variation_pct"] = statistics.coefficient_of_variation_defined
                                               ? finite_or_null(statistics.coefficient_of_variation_pct)
                                               : OrderedJson(nullptr);
  output["median_absolute_deviation"] = finite_or_null(statistics.median_absolute_deviation);
  return output;
}

OrderedJson metric_aggregate_json(const LlmMetricAggregate& aggregate, const char* units) {
  OrderedJson values = OrderedJson::array();
  for (double value : aggregate.values) {
    values.push_back(finite_or_null(value));
  }
  OrderedJson output;
  output["units"] = units;
  output["sample_count"] = aggregate.values.size();
  output["headline_semantics"] =
      aggregate.values.empty() ? "unavailable" : (aggregate.values.size() == 1 ? "single_measurement" : "median_p50");
  output["headline"] = optional_finite_or_null(aggregate.headline);
  output["values"] = std::move(values);
  output["statistics"] = statistics_json(aggregate.statistics, !aggregate.values.empty());
  return output;
}

OrderedJson scenario_aggregate_json(const LlmScenarioAggregate& aggregate) {
  OrderedJson output;
  output["scenario"] = llm_scenario_to_string(aggregate.scenario);
  output["status"] = std::string(aggregate.status);
  output["stability_quality"] = std::string(aggregate.stability_quality);
  output["cv_warning_threshold_pct"] = Constants::LLM_STREAMING_CV_WARNING_PCT;
  output["synthetic_work_unit_latency_seconds"] =
      metric_aggregate_json(aggregate.work_unit_latency_seconds, "seconds_per_synthetic_memory_work_unit");
  output["synthetic_memory_work_units_per_second"] =
      metric_aggregate_json(aggregate.synthetic_memory_work_units_per_second, "synthetic_memory_work_units_per_second");
  output["effective_model_payload_gb_s"] = metric_aggregate_json(aggregate.effective_model_payload_gb_s, "GB/s");
  return output;
}

OrderedJson traffic_diagnostics_json(const LlmGeometry& geometry,
                                     const std::array<LlmScenarioAggregate, kLlmScenarioCount>& aggregates) {
  const bool available = geometry.valid;
  const bool decode_available = available && geometry.decode.has_value();
  OrderedJson headlines = OrderedJson::object();
  for (LlmScenario scenario : kScenarios) {
    const size_t index = scenario_index(scenario);
    const LlmScenarioAggregate& aggregate = aggregates[index];
    headlines[llm_scenario_to_string(scenario)] = OrderedJson{
        {"synthetic_work_unit_latency_seconds", optional_finite_or_null(aggregate.work_unit_latency_seconds.headline)},
        {"synthetic_memory_work_units_per_second",
         optional_finite_or_null(aggregate.synthetic_memory_work_units_per_second.headline)},
        {"effective_model_payload_gb_s", optional_finite_or_null(aggregate.effective_model_payload_gb_s.headline)}};
  }

  OrderedJson ratio = nullptr;
  if (available && geometry.kv_read_bytes_per_work_unit != 0) {
    const long double exact_ratio = static_cast<long double>(geometry.weight_read_bytes_per_work_unit) /
                                    static_cast<long double>(geometry.kv_read_bytes_per_work_unit);
    const double ratio_value = static_cast<double>(exact_ratio);
    ratio = finite_or_null(ratio_value);
  }

  OrderedJson output;
  output["classification_version"] = Constants::LLM_TRAFFIC_CLASSIFICATION_VERSION;
  output["traffic_crossover_numerator"] = decimal_or_null(geometry.traffic_crossover_numerator, available);
  output["traffic_crossover_denominator"] = decimal_or_null(geometry.traffic_crossover_denominator, available);
  output["traffic_crossover_context_tokens"] =
      available ? finite_or_null(geometry.traffic_crossover_context_tokens) : OrderedJson(nullptr);
  output["current_visible_context_tokens"] =
      decode_available ? OrderedJson(geometry.decode->visible_context_tokens) : OrderedJson(nullptr);
  output["current_weight_read_payload_bytes_per_work_unit"] =
      decimal_or_null(geometry.weight_read_bytes_per_work_unit, available);
  output["current_kv_read_payload_bytes_per_work_unit"] =
      decimal_or_null(geometry.kv_read_bytes_per_work_unit, available);
  output["current_weight_to_kv_read_payload_ratio"] = std::move(ratio);
  output["current_context_classification"] =
      available ? OrderedJson(classify_llm_traffic_payload(geometry)) : OrderedJson(nullptr);
  output["classification_is_payload_only"] = true;
  output["scenario_headlines"] = std::move(headlines);
  return output;
}

OrderedJson budget_request_json(const LlmMemoryBudgetRequest& request) {
  OrderedJson output;
  output["valid"] = request.valid;
  output["reason_code"] = request.reason_code;
  output["mapping_granularity_bytes"] = decimal_string(request.mapping_granularity_bytes);
  output["requested_weight_mapping_bytes"] = decimal_string(request.requested_weight_mapping_bytes);
  output["requested_k_mapping_bytes"] = decimal_string(request.requested_k_mapping_bytes);
  output["requested_v_mapping_bytes"] = decimal_string(request.requested_v_mapping_bytes);
  output["committed_weight_mapping_bytes"] = decimal_string(request.committed_weight_mapping_bytes);
  output["committed_k_mapping_bytes"] = decimal_string(request.committed_k_mapping_bytes);
  output["committed_v_mapping_bytes"] = decimal_string(request.committed_v_mapping_bytes);
  output["requested_data_bytes"] = decimal_string(request.requested_data_bytes);
  output["committed_data_bytes"] = decimal_string(request.committed_data_bytes);
  output["descriptor_bytes"] = decimal_string(request.descriptor_bytes);
  output["planner_storage_bytes"] = decimal_string(request.planner_storage_bytes);
  output["checksum_auxiliary_bytes"] = decimal_string(request.checksum_auxiliary_bytes);
  output["orchestration_auxiliary_bytes"] = decimal_string(request.orchestration_auxiliary_bytes);
  output["auxiliary_bytes"] = decimal_string(request.auxiliary_bytes);
  output["required_total_bytes"] = decimal_string(request.required_total_bytes);
  return output;
}

OrderedJson memory_budget_json(const LlmMemoryBudget& budget) {
  const LlmMemoryBudgetRequest& request = budget.request;
  const size_t resource_rounding_bytes =
      request.committed_data_bytes >= request.requested_data_bytes
          ? request.committed_data_bytes - request.requested_data_bytes
          : 0;
  OrderedJson output;
  output["resource_rounding_bytes"] = decimal_string(resource_rounding_bytes);
  output["transient_peak_bytes"] = decimal_string(request.auxiliary_bytes);
  output["known_owned_peak_bytes"] = decimal_string(request.required_total_bytes);
  output["admitted_budget_bytes"] = decimal_string(budget.allowed_memory_bytes);
  output["valid"] = budget.valid;
  output["reason_code"] = budget.reason_code;
  output["request"] = budget_request_json(budget.request);
  output["available_memory_bytes"] = decimal_string(budget.available_memory_bytes);
  output["allowed_memory_bytes"] = decimal_string(budget.allowed_memory_bytes);
  output["used_fallback"] = budget.used_fallback;
  return output;
}

OrderedJson executor_auxiliary_json(const LlmExecutorAuxiliaryEstimate& auxiliary) {
  OrderedJson output;
  output["valid"] = auxiliary.valid;
  output["reason_code"] = std::string(auxiliary.reason_code);
  output["static_reference_bytes"] = decimal_string(auxiliary.static_reference_bytes);
  output["expected_checksum_bytes"] = decimal_string(auxiliary.expected_checksum_bytes);
  output["actual_checksum_bytes"] = decimal_string(auxiliary.actual_checksum_bytes);
  output["run_checksum_bytes"] = decimal_string(auxiliary.run_checksum_bytes);
  output["worker_status_bytes"] = decimal_string(auxiliary.worker_status_bytes);
  output["thread_handle_bytes"] = decimal_string(auxiliary.thread_handle_bytes);
  output["checksum_auxiliary_bytes"] = decimal_string(auxiliary.checksum_auxiliary_bytes);
  output["orchestration_auxiliary_bytes"] = decimal_string(auxiliary.orchestration_auxiliary_bytes);
  output["total_auxiliary_bytes"] = decimal_string(auxiliary.total_auxiliary_bytes);
  return output;
}

OrderedJson json_peak_estimate_json(const LlmJsonPeakEstimate& estimate, bool enabled) {
  OrderedJson output;
  output["enabled"] = enabled;
  output["valid"] = estimate.valid;
  output["reason_code"] = std::string(estimate.reason_code);
  output["policy"] = "conservative-live-dom-plus-serialized-transport";
  output["fixed_schema_bytes"] = decimal_string(estimate.fixed_schema_bytes);
  output["input_string_bytes"] = decimal_string(estimate.input_string_bytes);
  output["measurement_record_bytes"] = decimal_string(estimate.measurement_record_bytes);
  output["worker_checksum_bytes"] = decimal_string(estimate.worker_checksum_bytes);
  output["total_bytes"] = decimal_string(estimate.total_bytes);
  return output;
}

OrderedJson resources_json(const LlmMemoryWorkPlan& plan, const LlmResourcePreparationResult& preparation,
                           const LlmJsonPeakEstimate& json_peak_estimate, bool json_output_enabled) {
  const LlmCpuExecutionPlan* cpu_execution_plan = get_llm_cpu_execution_plan(plan);
  const LlmCpuExecutionPlan unavailable_cpu_execution_plan;
  const LlmCpuExecutionPlan& cpu =
      cpu_execution_plan == nullptr ? unavailable_cpu_execution_plan : *cpu_execution_plan;
  const LlmMemoryBudgetRequest& request = preparation.memory_budget.request;
  OrderedJson mappings;
  mappings["policy"] = "private_anonymous_regular_cacheable";
  mappings["full_size_physical_mappings"] = true;
  mappings["weight"] = OrderedJson{{"requested_bytes", decimal_string(request.requested_weight_mapping_bytes)},
                                   {"committed_bytes", decimal_string(request.committed_weight_mapping_bytes)}};
  mappings["k"] = OrderedJson{{"requested_bytes", decimal_string(request.requested_k_mapping_bytes)},
                              {"committed_bytes", decimal_string(request.committed_k_mapping_bytes)}};
  mappings["v"] = OrderedJson{{"requested_bytes", decimal_string(request.requested_v_mapping_bytes)},
                              {"committed_bytes", decimal_string(request.committed_v_mapping_bytes)}};
  mappings["requested_data_bytes"] = decimal_string(request.requested_data_bytes);
  mappings["committed_data_bytes"] = decimal_string(request.committed_data_bytes);

  const LlmInitializationEvidence& initialization = preparation.initialization;
  OrderedJson initialization_json;
  initialization_json["complete"] = initialization.complete;
  initialization_json["pattern_version"] = plan.component_identities.buffer_pattern_version;
  initialization_json["pre_touch_policy"] = "write-every-requested-mapping-byte-once";
  initialization_json["separate_reference_read_pass"] = false;
  initialization_json["static_references_accumulated_during_initialization"] = true;
  initialization_json["weight_bytes"] = decimal_string(initialization.weight_bytes);
  initialization_json["k_bytes"] = decimal_string(initialization.k_bytes);
  initialization_json["v_bytes"] = decimal_string(initialization.v_bytes);
  initialization_json["total_bytes"] = decimal_string(initialization.total_bytes);
  initialization_json["non_empty_weight_spans"] = initialization.non_empty_weight_spans;
  initialization_json["non_empty_k_spans"] = initialization.non_empty_k_spans;
  initialization_json["non_empty_v_spans"] = initialization.non_empty_v_spans;

  OrderedJson descriptors;
  descriptors["abi_version"] = plan.component_identities.resource_abi_version;
  descriptors["layer_descriptors_per_worker"] = cpu.layer_descriptors_per_worker;
  descriptors["sequence_descriptors_per_worker"] = cpu.sequence_descriptors_per_worker;
  descriptors["total_layer_descriptors"] = cpu.total_layer_descriptors;
  descriptors["total_sequence_descriptors"] = cpu.total_sequence_descriptors;
  descriptors["descriptor_bytes"] = decimal_string(cpu.descriptor_bytes);

  OrderedJson output;
  output["valid"] = preparation.valid;
  output["reason_code"] = preparation.reason_code;
  output["model_plan_identity"] = non_empty_or_null(plan.plan_identity);
  output["mappings"] = std::move(mappings);
  output["descriptors"] = std::move(descriptors);
  output["executor_auxiliary"] = executor_auxiliary_json(preparation.auxiliary);
  output["json_output_peak_estimate"] = json_peak_estimate_json(json_peak_estimate, json_output_enabled);
  output["allocation_memory_budget"] = memory_budget_json(preparation.memory_budget);
  output["initialization"] = std::move(initialization_json);
  return output;
}

OrderedJson seeds_json(const LlmMemoryConfig& config, const LlmMemoryWorkPlan& plan) {
  OrderedJson buffers;
  buffers["weight_uint64_decimal"] = decimal_string(plan.weight_buffer_seed);
  buffers["k_uint64_decimal"] = decimal_string(plan.k_buffer_seed);
  buffers["v_uint64_decimal"] = decimal_string(plan.v_buffer_seed);

  OrderedJson scenarios = OrderedJson::object();
  for (LlmScenario scenario : kScenarios) {
    scenarios[llm_scenario_to_string(scenario)] = decimal_string(plan.scenario_seeds[scenario_index(scenario)]);
  }

  OrderedJson output;
  output["base_seed_uint64_decimal"] = decimal_string(plan.base_seed);
  output["source"] = config.user_specified_seed ? "user" : "generated";
  output["buffer_domain_seeds"] = std::move(buffers);
  output["scenario_domain_seeds"] = std::move(scenarios);
  return output;
}

OrderedJson model_work_plan_json(const LlmMemoryWorkPlan& plan) {
  const LlmCpuExecutionPlan* cpu_execution_plan = get_llm_cpu_execution_plan(plan);
  const LlmCpuExecutionPlan unavailable_cpu_execution_plan;
  const LlmCpuExecutionPlan& cpu =
      cpu_execution_plan == nullptr ? unavailable_cpu_execution_plan : *cpu_execution_plan;
  OrderedJson output;
  output["valid"] = plan.valid;
  output["reason_code"] = plan.reason_code;
  output["plan_identity"] = non_empty_or_null(plan.plan_identity);
  output["methodology_version"] = plan.methodology_version;
  output["backend"] = llm_memory_backend_to_string(plan.backend);
  output["phase"] = llm_phase_to_string(plan.phase);
  output["kv_layout"] = llm_kv_layout_to_string(plan.kv_layout);
  output["work_unit_kind"] = llm_work_unit_kind_to_string(plan.work_unit_kind);
  output["component_identity"] = non_empty_or_null(plan.component_identities.identity);
  output["weight_passes_per_work_unit"] = plan.weight_passes_per_work_unit;
  output["kv_replay_factor"] = plan.kv_replay_factor;
  output["requested_workers"] = cpu.requested_workers;
  output["available_workers"] = cpu.available_workers;
  output["effective_workers"] = cpu.effective_workers;
  output["worker_plan_count"] = cpu.workers.size();
  output["weight_layer_count"] = plan.weight_layers.size();
  output["layer_descriptors_per_worker"] = cpu.layer_descriptors_per_worker;
  output["sequence_descriptors_per_worker"] = cpu.sequence_descriptors_per_worker;
  output["total_layer_descriptors"] = cpu.total_layer_descriptors;
  output["total_sequence_descriptors"] = cpu.total_sequence_descriptors;
  output["descriptor_bytes"] = decimal_string(cpu.descriptor_bytes);
  output["planner_storage_bytes"] = decimal_string(cpu.planner_storage_bytes);
  return output;
}

OrderedJson scenario_work_plan_json(const LlmScenarioWorkPlan& plan, LlmScenario expected_scenario) {
  const bool available = plan.valid;
  OrderedJson output;
  output["valid"] = plan.valid;
  output["reason_code"] = plan.reason_code;
  output["scenario"] = llm_scenario_to_string(expected_scenario);
  output["work_unit_kind"] = llm_work_unit_kind_to_string(plan.work_unit_kind);
  output["kv_write_kind"] = llm_kv_write_kind_to_string(plan.kv_write_kind);
  output["explicit_iterations"] = available ? OrderedJson(plan.explicit_iterations) : OrderedJson(nullptr);
  output["model_plan_identity"] = available ? non_empty_or_null(plan.model_plan_identity) : OrderedJson(nullptr);
  output["scenario_seed_uint64_decimal"] = decimal_or_null(plan.scenario_seed, available);
  output["work_units"] = number_or_null(plan.work_units, available);
  output["weight_read_bytes_per_work_unit"] = decimal_or_null(plan.weight_read_bytes_per_work_unit, available);
  output["kv_read_bytes_per_work_unit"] = decimal_or_null(plan.kv_read_bytes_per_work_unit, available);
  output["kv_write_bytes_per_work_unit"] = decimal_or_null(plan.kv_write_bytes_per_work_unit, available);
  output["effective_model_payload_bytes_per_work_unit"] =
      decimal_or_null(plan.effective_model_payload_bytes_per_work_unit,
                      available);
  output["layout_metadata_lookup_count_per_work_unit"] =
      decimal_or_null(plan.layout_metadata_lookup_count_per_work_unit, available);
  output["layout_metadata_read_bytes_per_work_unit"] =
      decimal_or_null(plan.layout_metadata_read_bytes_per_work_unit, available);
  output["accounted_bytes_per_work_unit"] = decimal_or_null(plan.accounted_bytes_per_work_unit, available);
  output["weight_read_bytes"] = decimal_or_null(plan.weight_read_bytes, available);
  output["kv_read_bytes"] = decimal_or_null(plan.kv_read_bytes, available);
  output["kv_write_bytes"] = decimal_or_null(plan.kv_write_bytes, available);
  output["effective_model_payload_bytes"] = decimal_or_null(plan.effective_model_payload_bytes, available);
  output["layout_metadata_lookup_count"] = decimal_or_null(plan.layout_metadata_lookup_count, available);
  output["layout_metadata_read_bytes"] = decimal_or_null(plan.layout_metadata_read_bytes, available);
  output["task_accounted_bytes"] = decimal_or_null(plan.task_accounted_bytes, available);
  output["maximum_work_units_by_work_unit_cap"] = number_or_null(plan.maximum_work_units_by_work_unit_cap, available);
  output["maximum_work_units_by_guardrail"] = number_or_null(plan.maximum_work_units_by_guardrail, available);
  output["effective_maximum_work_units"] = number_or_null(plan.effective_maximum_work_units, available);
  output["plan_identity"] = available ? non_empty_or_null(plan.plan_identity) : OrderedJson(nullptr);
  return output;
}

OrderedJson frozen_scenario_plans_json(const LlmFrozenScenarioPlans& frozen) {
  OrderedJson scenarios = OrderedJson::array();
  for (LlmScenario scenario : kScenarios) {
    scenarios.push_back(scenario_work_plan_json(frozen.scenarios[scenario_index(scenario)], scenario));
  }

  OrderedJson output;
  output["valid"] = frozen.valid;
  output["reason_code"] = frozen.reason_code;
  output["explicit_iterations"] = frozen.valid ? OrderedJson(frozen.explicit_iterations) : OrderedJson(nullptr);
  output["model_plan_identity"] = frozen.valid ? non_empty_or_null(frozen.model_plan_identity) : OrderedJson(nullptr);
  output["plan_identity"] = frozen.valid ? non_empty_or_null(frozen.plan_identity) : OrderedJson(nullptr);
  output["scenarios"] = std::move(scenarios);
  return output;
}

OrderedJson run_checksum_json(const LlmRunChecksum& checksum) {
  return OrderedJson{{"state_a_uint64_decimal", decimal_string(checksum.state_a)},
                     {"state_b_uint64_decimal", decimal_string(checksum.state_b)}};
}

OrderedJson checksum_component_json(const LlmReadChecksumComponent& component) {
  OrderedJson output;
  output["state_a_uint64_decimal"] = decimal_string(component.state_a);
  output["state_b_uint64_decimal"] = decimal_string(component.state_b);
  output["exact_bytes_read"] = decimal_string(component.exact_bytes_read);
  output["span_count_uint64_decimal"] = decimal_string(component.span_count);
  return output;
}

OrderedJson worker_checksums_json(const std::vector<LlmWorkerChecksum>& checksums) {
  OrderedJson output = OrderedJson::array();
  for (size_t worker_index = 0; worker_index < checksums.size(); ++worker_index) {
    const LlmWorkerChecksum& checksum = checksums[worker_index];
    output.push_back(OrderedJson{{"worker_index", worker_index},
                                 {"weight", checksum_component_json(checksum.weight)},
                                 {"k", checksum_component_json(checksum.k)},
                                 {"v", checksum_component_json(checksum.v)}});
  }
  return output;
}

bool compact_checksum_evaluated(const LlmTaskExecutionEvidence& execution) noexcept {
  return execution.available && execution.checksum_evaluated;
}

bool measurement_checksum_evaluated(const LlmMeasurementState& measurement) noexcept {
  const LlmExecutorResult* cpu =
      get_llm_cpu_task_evidence(measurement.execution);
  return measurement.execution_evidence_available && cpu != nullptr &&
         cpu->checksum_evaluated &&
         cpu->expected_checksums.size() == cpu->requested_workers &&
         cpu->actual_checksums.size() == cpu->requested_workers;
}

const char* checksum_status(bool evaluated, bool valid) noexcept {
  if (!evaluated) {
    return "not_evaluated";
  }
  return valid ? "valid" : "invalid";
}

OrderedJson compact_execution_json(const LlmTaskExecutionEvidence& execution, std::string_view fallback_reason_code) {
  const bool available = execution.available;
  const bool checksum_evaluated = compact_checksum_evaluated(execution);
  const std::string reason_code(available ? execution.reason_code : fallback_reason_code);
  OrderedJson checksum;
  checksum["status"] = checksum_status(checksum_evaluated, execution.checksum_valid);
  checksum["reason_code"] = reason_code;
  checksum["algorithm_version"] = Constants::LLM_READ_CHECKSUM_VERSION;
  checksum["checksum_valid"] = checksum_evaluated ? OrderedJson(execution.checksum_valid) : OrderedJson(nullptr);
  checksum["expected_run_checksum"] =
      checksum_evaluated ? run_checksum_json(execution.expected_run_checksum) : OrderedJson(nullptr);
  checksum["actual_run_checksum"] =
      checksum_evaluated ? run_checksum_json(execution.actual_run_checksum) : OrderedJson(nullptr);

  OrderedJson output;
  output["status"] = !available ? "unavailable" : (execution.valid ? "valid" : "invalid");
  output["reason_code"] = reason_code;
  output["valid"] = available ? OrderedJson(execution.valid) : OrderedJson(nullptr);
  output["elapsed_seconds"] =
      positive_finite_or_null(execution.elapsed_seconds,
                              available && execution.timing_evaluated &&
                                  execution.timing_valid);
  const bool cpu_available = available && execution.cpu_evidence_available;
  output["requested_workers"] =
      number_or_null(execution.requested_workers, cpu_available);
  output["created_workers"] =
      number_or_null(execution.created_workers, cpu_available);
  output["completed_workers"] =
      number_or_null(execution.completed_workers, cpu_available);
  output["qos_successful_workers"] =
      number_or_null(execution.qos_successful_workers, cpu_available);
  output["qos_failed_workers"] =
      number_or_null(execution.qos_failed_workers, cpu_available);
  output["worker_startup_failed"] =
      cpu_available ? OrderedJson(execution.worker_startup_failed)
                    : OrderedJson(nullptr);
  output["kernel_succeeded"] =
      cpu_available ? OrderedJson(execution.kernel_succeeded)
                    : OrderedJson(nullptr);
  output["timer_started"] =
      cpu_available ? OrderedJson(execution.timer_started)
                    : OrderedJson(nullptr);
  output["timer_stopped"] =
      cpu_available ? OrderedJson(execution.timer_stopped)
                    : OrderedJson(nullptr);
  output["checksum"] = std::move(checksum);
  return output;
}

OrderedJson calibration_attempt_json(const LlmCalibrationAttempt& attempt, size_t attempt_index) {
  const bool plan_available = !attempt.work_plan_identity.empty();
  OrderedJson output;
  output["attempt_index"] = attempt_index;
  output["scenario"] = llm_scenario_to_string(attempt.scenario);
  output["work_unit_kind"] = llm_work_unit_kind_to_string(attempt.work_unit_kind);
  output["kv_write_kind"] = llm_kv_write_kind_to_string(attempt.kv_write_kind);
  output["purpose"] = std::string(attempt.purpose);
  output["explicit_iterations"] = attempt.explicit_iterations;
  output["work_units"] = number_or_null(attempt.work_units, plan_available);
  output["weight_read_bytes"] = decimal_or_null(attempt.weight_read_bytes, plan_available);
  output["kv_read_bytes"] = decimal_or_null(attempt.kv_read_bytes, plan_available);
  output["kv_write_bytes"] = decimal_or_null(attempt.kv_write_bytes, plan_available);
  output["effective_model_payload_bytes"] = decimal_or_null(attempt.effective_model_payload_bytes, plan_available);
  output["layout_metadata_lookup_count"] = decimal_or_null(attempt.layout_metadata_lookup_count, plan_available);
  output["layout_metadata_read_bytes"] = decimal_or_null(attempt.layout_metadata_read_bytes, plan_available);
  output["task_accounted_bytes"] = decimal_or_null(attempt.task_accounted_bytes, plan_available);
  output["work_plan_identity"] = non_empty_or_null(attempt.work_plan_identity);
  output["duration_quality"] = std::string(attempt.duration_quality);
  output["terminal"] = attempt.terminal;
  output["valid"] = attempt.valid;
  output["reason_code"] = std::string(attempt.reason_code);
  output["execution"] = compact_execution_json(attempt.execution, attempt.reason_code);
  return output;
}

OrderedJson excluded_calibration_json(const LlmMemoryResult& result) {
  OrderedJson output = OrderedJson::object();
  for (LlmScenario scenario : kScenarios) {
    const size_t index = scenario_index(scenario);
    OrderedJson attempts = OrderedJson::array();
    for (size_t attempt_index = 0; attempt_index < result.calibration_attempts[index].size(); ++attempt_index) {
      attempts.push_back(calibration_attempt_json(result.calibration_attempts[index][attempt_index], attempt_index));
    }
    output[llm_scenario_to_string(scenario)] = std::move(attempts);
  }
  return output;
}

OrderedJson runner_auxiliary_json(const LlmRunnerAuxiliaryEstimate& auxiliary) {
  OrderedJson output;
  output["valid"] = auxiliary.valid;
  output["reason_code"] = std::string(auxiliary.reason_code);
  output["measurement_record_bytes"] = decimal_string(auxiliary.measurement_record_bytes);
  output["loop_record_bytes"] = decimal_string(auxiliary.loop_record_bytes);
  output["calibration_record_bytes"] = decimal_string(auxiliary.calibration_record_bytes);
  output["calibration_identity_bytes"] = decimal_string(auxiliary.calibration_identity_bytes);
  output["aggregate_value_bytes"] = decimal_string(auxiliary.aggregate_value_bytes);
  output["statistics_workspace_bytes"] = decimal_string(auxiliary.statistics_workspace_bytes);
  output["warning_record_bytes"] = decimal_string(auxiliary.warning_record_bytes);
  output["fixed_metadata_bytes"] = decimal_string(auxiliary.fixed_metadata_bytes);
  output["retained_checksum_bytes"] = decimal_string(auxiliary.retained_checksum_bytes);
  output["checksum_auxiliary_bytes"] = decimal_string(auxiliary.checksum_auxiliary_bytes);
  output["orchestration_auxiliary_bytes"] = decimal_string(auxiliary.orchestration_auxiliary_bytes);
  output["total_auxiliary_bytes"] = decimal_string(auxiliary.total_auxiliary_bytes);
  return output;
}

OrderedJson counters_json(const LlmMemoryResult& result) {
  const LlmRunCounters& counters = result.counters;
  OrderedJson output;
  output["planned_loops"] = counters.planned_loops;
  output["attempted_loops"] = counters.attempted_loops;
  output["completed_loops"] = counters.completed_loops;
  output["planned_measurements"] = counters.planned_measurements;
  output["attempted_measurements"] = counters.attempted_measurements;
  output["terminal_measurements"] = counters.terminal_measurements;
  output["measured_measurements"] = counters.measured_measurements;
  output["planned_work_units"] = decimal_string(counters.planned_work_units);
  output["completed_work_units"] = decimal_string(counters.completed_work_units);
  output["planned_effective_model_payload_bytes"] = decimal_string(counters.planned_effective_model_payload_bytes);
  output["completed_effective_model_payload_bytes"] = decimal_string(counters.completed_effective_model_payload_bytes);
  output["planned_layout_metadata_lookup_count"] = decimal_string(counters.planned_layout_metadata_lookup_count);
  output["completed_layout_metadata_lookup_count"] = decimal_string(counters.completed_layout_metadata_lookup_count);
  output["planned_layout_metadata_read_bytes"] = decimal_string(counters.planned_layout_metadata_read_bytes);
  output["completed_layout_metadata_read_bytes"] = decimal_string(counters.completed_layout_metadata_read_bytes);
  output["planned_task_accounted_bytes"] = decimal_string(counters.planned_task_accounted_bytes);
  output["completed_task_accounted_bytes"] = decimal_string(counters.completed_task_accounted_bytes);
  output["runner_auxiliary"] = runner_auxiliary_json(result.runner_auxiliary);
  return output;
}

OrderedJson checkpoint_lifecycle_json(const LlmMemoryResult& result) {
  OrderedJson output;
  output["checkpoint_failed"] = result.checkpoint_failed;
  output["logical_checkpoint_attempts"] = result.logical_checkpoint_attempts;
  output["successful_logical_checkpoints"] = result.successful_logical_checkpoints;
  output["terminal_checkpoint_attempted"] = result.terminal_checkpoint_attempted;
  output["terminal_checkpoint_completed"] = result.terminal_checkpoint_completed;
  output["checkpoint_policy"] = "after-each-terminal-measurement-and-at-command-terminal";
  output["file_checkpoint_failure_is_terminal_and_not_retried"] = true;
  output["stdout_intermediate_checkpoints_are_lazy"] = true;
  return output;
}

OrderedJson loop_records_json(const LlmMemoryResult& result) {
  OrderedJson output = OrderedJson::array();
  for (const LlmLoopRecord& loop : result.loops) {
    OrderedJson planned = OrderedJson::array();
    OrderedJson realized = OrderedJson::array();
    OrderedJson measurement_indexes = OrderedJson::array();
    for (LlmScenario scenario : loop.planned_order) {
      planned.push_back(llm_scenario_to_string(scenario));
    }
    const size_t realized_count = std::min(loop.realized_order_count, kLlmScenarioCount);
    for (size_t index = 0; index < realized_count; ++index) {
      realized.push_back(llm_scenario_to_string(loop.realized_order[index]));
    }
    for (size_t measurement_index : loop.measurement_indexes) {
      measurement_indexes.push_back(measurement_index == kLlmNoTaskIndex ? OrderedJson(nullptr)
                                                                         : OrderedJson(measurement_index));
    }
    output.push_back(OrderedJson{{"loop_index", loop.loop_index},
                                 {"planned_order", std::move(planned)},
                                 {"realized_order", std::move(realized)},
                                 {"realized_order_count", loop.realized_order_count},
                                 {"measurement_indexes", std::move(measurement_indexes)}});
  }
  return output;
}

OrderedJson measurement_execution_json(const LlmMeasurementState& measurement) {
  const bool attempted = measurement.attempted;
  const bool available = measurement.execution_evidence_available;
  const LlmTaskExecutionResult& execution = measurement.execution;
  const LlmExecutorResult* cpu = get_llm_cpu_task_evidence(execution);
  const bool valid =
      available && execution.status == LlmTaskExecutionStatus::Complete &&
      execution.reason_code == LlmBackendReason::VALID &&
      execution.validation.evaluated && execution.validation.valid &&
      execution.timing.evaluated && execution.timing.valid;
  OrderedJson output;
  output["status"] =
      !attempted ? "not_run"
                 : (!available ? "unavailable" : (valid ? "valid" : "invalid"));
  output["reason_code"] = available ? execution.reason_code : std::string(measurement.reason_code);
  output["valid"] = available ? OrderedJson(valid) : OrderedJson(nullptr);
  output["elapsed_seconds"] = positive_finite_or_null(
      execution.timing.elapsed_seconds,
      available && execution.timing.evaluated && execution.timing.valid);
  const bool cpu_available = available && cpu != nullptr;
  output["requested_workers"] =
      number_or_null(cpu_available ? cpu->requested_workers : 0,
                     cpu_available);
  output["created_workers"] =
      number_or_null(cpu_available ? cpu->created_workers : 0,
                     cpu_available);
  output["completed_workers"] =
      number_or_null(cpu_available ? cpu->completed_workers : 0,
                     cpu_available);
  output["qos_successful_workers"] =
      number_or_null(cpu_available ? cpu->qos_successful_workers : 0,
                     cpu_available);
  output["qos_failed_workers"] =
      number_or_null(cpu_available ? cpu->qos_failed_workers : 0,
                     cpu_available);
  output["worker_startup_failed"] =
      cpu_available ? OrderedJson(cpu->worker_startup_failed)
                    : OrderedJson(nullptr);
  output["kernel_succeeded"] =
      cpu_available ? OrderedJson(cpu->kernel_succeeded)
                    : OrderedJson(nullptr);
  output["timer_started"] =
      cpu_available ? OrderedJson(cpu->timer_started)
                    : OrderedJson(nullptr);
  output["timer_stopped"] =
      cpu_available ? OrderedJson(cpu->timer_stopped)
                    : OrderedJson(nullptr);
  return output;
}

OrderedJson measurement_checksum_json(const LlmMeasurementState& measurement) {
  const bool evaluated = measurement_checksum_evaluated(measurement);
  const LlmTaskExecutionResult& execution = measurement.execution;
  const LlmExecutorResult* cpu = get_llm_cpu_task_evidence(execution);
  const bool checksum_valid = evaluated && cpu != nullptr &&
                              cpu->checksum_valid;
  OrderedJson output;
  output["status"] = checksum_status(evaluated, checksum_valid);
  output["reason_code"] =
      measurement.execution_evidence_available ? execution.reason_code : std::string(measurement.reason_code);
  output["initialization_pattern_version"] = Constants::LLM_BUFFER_PATTERN_VERSION;
  output["append_pattern_version"] = Constants::LLM_APPEND_PATTERN_VERSION;
  output["read_checksum_version"] = Constants::LLM_READ_CHECKSUM_VERSION;
  output["checksum_valid"] =
      evaluated ? OrderedJson(checksum_valid) : OrderedJson(nullptr);
  output["expected_worker_checksums"] =
      evaluated ? worker_checksums_json(cpu->expected_checksums)
                : OrderedJson(nullptr);
  output["actual_worker_checksums"] =
      evaluated ? worker_checksums_json(cpu->actual_checksums)
                : OrderedJson(nullptr);
  output["expected_run_checksum"] =
      evaluated ? run_checksum_json(cpu->expected_run_checksum)
                : OrderedJson(nullptr);
  output["actual_run_checksum"] =
      evaluated ? run_checksum_json(cpu->actual_run_checksum)
                : OrderedJson(nullptr);
  return output;
}

OrderedJson calibration_indexes_json(size_t count) {
  OrderedJson output = OrderedJson::array();
  for (size_t index = 0; index < count; ++index) {
    output.push_back(index);
  }
  return output;
}

OrderedJson measurement_json(const LlmMeasurementState& measurement, const LlmMemoryResult& result,
                             const LlmMemoryWorkPlan& model_plan) {
  const bool plan_available = measurement.frozen_plan_index < kLlmScenarioCount &&
                              result.frozen_scenario_plans.scenarios[measurement.frozen_plan_index].valid;
  const LlmScenarioWorkPlan* frozen_plan =
      plan_available ? &result.frozen_scenario_plans.scenarios[measurement.frozen_plan_index] : nullptr;

  OrderedJson working_set;
  working_set["bytes"] = decimal_string(measurement.working_set_bytes);
  working_set["full_size_physical_mappings"] = true;
  working_set["cacheable"] = true;
  working_set["kv_layout"] = llm_kv_layout_to_string(model_plan.kv_layout);
  working_set["fixed_visible_context_tokens"] =
      model_plan.geometry.decode.has_value() ? OrderedJson(model_plan.geometry.decode->visible_context_tokens)
                                             : OrderedJson(nullptr);
  working_set["current_token_slot_included"] = model_plan.phase == LlmPhase::Decode ? OrderedJson(true)
                                                                                   : OrderedJson(nullptr);

  OrderedJson output;
  output["scenario"] = llm_scenario_to_string(measurement.scenario);
  output["work_unit_kind"] = llm_work_unit_kind_to_string(measurement.work_unit_kind);
  output["kv_write_kind"] = llm_kv_write_kind_to_string(measurement.kv_write_kind);
  output["loop_index"] = measurement.loop_index;
  output["order_position"] = measurement.order_position;
  output["status"] = llm_measurement_status_to_string(measurement.status);
  output["reason_code"] = std::string(measurement.reason_code);
  output["attempted"] = measurement.attempted;
  output["requested_workers"] = measurement.requested_workers;
  output["effective_workers"] = measurement.effective_workers;
  output["qos_successful_workers"] =
      number_or_null(measurement.qos_successful_workers, measurement.execution_evidence_available);
  output["qos_failed_workers"] =
      number_or_null(measurement.qos_failed_workers, measurement.execution_evidence_available);
  output["frozen_plan_index"] = number_or_null(measurement.frozen_plan_index, plan_available);
  output["frozen_work_plan_identity"] =
      frozen_plan == nullptr ? OrderedJson(nullptr) : non_empty_or_null(frozen_plan->plan_identity);
  output["scenario_seed_uint64_decimal"] =
      frozen_plan == nullptr ? OrderedJson(nullptr) : decimal_string(frozen_plan->scenario_seed);
  output["explicit_iterations"] = plan_available ? OrderedJson(measurement.explicit_iterations) : OrderedJson(nullptr);
  output["work_policy"] =
      !plan_available ? OrderedJson(nullptr)
                      : OrderedJson(measurement.explicit_iterations ? "explicit_fixed_work" : "automatic_calibration");
  output["duration_quality"] = std::string(measurement.duration_quality);
  output["calibration_attempt_count"] = measurement.calibration_attempt_count;
  output["calibration_attempt_indexes"] = calibration_indexes_json(measurement.calibration_attempt_count);
  // Work-unit counts remain JSON integers, while all exact potentially-large
  // counts and byte quantities remain decimal strings, including zero-valued
  // not-run records. This keeps field types independent of run status.
  output["planned_work_units"] = measurement.planned_work_units;
  output["completed_work_units"] = measurement.completed_work_units;
  output["weight_read_bytes_per_work_unit"] = decimal_string(measurement.weight_read_bytes_per_work_unit);
  output["kv_read_bytes_per_work_unit"] = decimal_string(measurement.kv_read_bytes_per_work_unit);
  output["kv_write_bytes_per_work_unit"] = decimal_string(measurement.kv_write_bytes_per_work_unit);
  output["effective_model_payload_bytes_per_work_unit"] =
      decimal_string(measurement.effective_model_payload_bytes_per_work_unit);
  output["layout_metadata_lookup_count_per_work_unit"] =
      decimal_string(measurement.layout_metadata_lookup_count_per_work_unit);
  output["layout_metadata_read_bytes_per_work_unit"] =
      decimal_string(measurement.layout_metadata_read_bytes_per_work_unit);
  output["accounted_bytes_per_work_unit"] = decimal_string(measurement.accounted_bytes_per_work_unit);
  output["planned_weight_read_bytes"] = decimal_string(measurement.planned_weight_read_bytes);
  output["planned_kv_read_bytes"] = decimal_string(measurement.planned_kv_read_bytes);
  output["planned_kv_write_bytes"] = decimal_string(measurement.planned_kv_write_bytes);
  output["planned_effective_model_payload_bytes"] =
      decimal_string(measurement.planned_effective_model_payload_bytes);
  output["completed_effective_model_payload_bytes"] =
      decimal_string(measurement.completed_effective_model_payload_bytes);
  output["planned_layout_metadata_lookup_count"] =
      decimal_string(measurement.planned_layout_metadata_lookup_count);
  output["completed_layout_metadata_lookup_count"] =
      decimal_string(measurement.completed_layout_metadata_lookup_count);
  output["planned_layout_metadata_read_bytes"] = decimal_string(measurement.planned_layout_metadata_read_bytes);
  output["completed_layout_metadata_read_bytes"] =
      decimal_string(measurement.completed_layout_metadata_read_bytes);
  output["planned_task_accounted_bytes"] = decimal_string(measurement.planned_task_accounted_bytes);
  output["completed_task_accounted_bytes"] = decimal_string(measurement.completed_task_accounted_bytes);
  output["elapsed_seconds"] = optional_finite_or_null(measurement.elapsed_seconds);
  output["synthetic_work_unit_latency_seconds"] =
      optional_finite_or_null(measurement.synthetic_work_unit_latency_seconds);
  output["synthetic_memory_work_units_per_second"] =
      optional_finite_or_null(
          measurement.synthetic_memory_work_units_per_second);
  output["effective_model_payload_gb_s"] = optional_finite_or_null(measurement.effective_model_payload_gb_s);
  output["weight_payload_fraction"] = optional_finite_or_null(measurement.weight_payload_fraction);
  output["kv_read_payload_fraction"] = optional_finite_or_null(measurement.kv_read_payload_fraction);
  output["kv_write_payload_fraction"] = optional_finite_or_null(measurement.kv_write_payload_fraction);
  output["working_set"] = std::move(working_set);
  output["execution"] = measurement_execution_json(measurement);
  output["checksum"] = measurement_checksum_json(measurement);
  return output;
}

OrderedJson measurements_json(const LlmMemoryResult& result, const LlmMemoryWorkPlan& model_plan) {
  OrderedJson output = OrderedJson::array();
  for (const LlmMeasurementState& measurement : result.measurements) {
    output.push_back(measurement_json(measurement, result, model_plan));
  }
  return output;
}

OrderedJson scenario_aggregates_json(const LlmMemoryResult& result) {
  OrderedJson output = OrderedJson::object();
  for (LlmScenario scenario : kScenarios) {
    const size_t index = scenario_index(scenario);
    output[llm_scenario_to_string(scenario)] = scenario_aggregate_json(result.aggregates[index]);
  }
  return output;
}

OrderedJson software_json(const LlmResultMetadata& metadata) {
  OrderedJson output;
  output["version"] = SOFTVERSION;
  output["timestamp"] = metadata.timestamp.empty() ? build_utc_timestamp() : metadata.timestamp;
  return output;
}

OrderedJson resolved_plan_json(const LlmMemoryWorkPlan& plan, const LlmFrozenScenarioPlans& frozen) {
  OrderedJson output;
  output["valid"] = plan.valid;
  output["reason_code"] = plan.reason_code;
  output["plan_identity"] = non_empty_or_null(plan.plan_identity);
  output["methodology_version"] = plan.methodology_version;
  output["backend"] = llm_memory_backend_to_string(plan.backend);
  output["phase"] = llm_phase_to_string(plan.phase);
  output["kv_layout"] = llm_kv_layout_to_string(plan.kv_layout);
  output["work_unit_kind"] = llm_work_unit_kind_to_string(plan.work_unit_kind);
  output["geometry"] = geometry_json(plan.geometry);
  output["layout"] = layout_json(plan);
  output["resources"] = resolved_resources_json(plan);
  output["component_identities"] = component_identities_json(plan.component_identities);
  output["methodology"] = methodology_json(plan);
  output["model_work_plan"] = model_work_plan_json(plan);
  output["frozen_scenario_work_plans"] = frozen_scenario_plans_json(frozen);
  return output;
}

OrderedJson backend_evidence_json(const LlmMemoryWorkPlan& plan, const LlmBackendEvidence& backend_evidence,
                                  const LlmJsonPeakEstimate& json_peak_estimate, bool json_output_enabled) {
  OrderedJson cpu = nullptr;
  if (plan.backend == LlmMemoryBackend::Cpu) {
    const LlmCpuExecutionPlan* cpu_execution_plan =
        get_llm_cpu_execution_plan(plan);
    const LlmCpuExecutionPlan unavailable_cpu_execution_plan;
    const LlmCpuExecutionPlan& cpu_plan =
        cpu_execution_plan == nullptr ? unavailable_cpu_execution_plan
                                      : *cpu_execution_plan;
    const LlmResourcePreparationResult* preparation =
        get_llm_cpu_preparation(backend_evidence);
    const LlmResourcePreparationResult unavailable;
    const LlmResourcePreparationResult& cpu_preparation =
        preparation == nullptr ? unavailable : *preparation;
    cpu = OrderedJson{{"requested_workers", cpu_plan.requested_workers},
                      {"available_workers", cpu_plan.available_workers},
                      {"effective_workers", cpu_plan.effective_workers},
                      {"resource_abi_version", plan.component_identities.resource_abi_version},
                      {"schedule_version", plan.component_identities.schedule_version},
                      {"timer_policy_version", plan.component_identities.timer_policy_version},
                      {"resources", resources_json(plan, cpu_preparation, json_peak_estimate, json_output_enabled)}};
  }

  OrderedJson output;
  output["cpu"] = std::move(cpu);
  output["metal"] = nullptr;
  return output;
}

OrderedJson calibration_json(const LlmMemoryResult& result) {
  OrderedJson output;
  output["excluded_from_results"] = true;
  output["attempts"] = excluded_calibration_json(result);
  return output;
}

OrderedJson aggregates_json(const LlmMemoryWorkPlan& plan, const LlmMemoryResult& result) {
  OrderedJson output;
  output["scenarios"] = scenario_aggregates_json(result);
  output["traffic_diagnostics"] = traffic_diagnostics_json(plan.geometry, result.aggregates);
  return output;
}

OrderedJson environment_snapshot_json(const LlmHostEnvironmentSnapshot& snapshot) {
  OrderedJson output;
  output["thermal_state"] = snapshot.thermal_state;
  output["low_power_mode_available"] = snapshot.low_power_mode_available;
  output["low_power_mode_enabled"] =
      snapshot.low_power_mode_available ? OrderedJson(snapshot.low_power_mode_enabled) : OrderedJson(nullptr);
  output["physical_memory_bytes"] =
      snapshot.physical_memory_bytes == 0 ? OrderedJson(nullptr) : decimal_string(snapshot.physical_memory_bytes);
  return output;
}

OrderedJson environment_json(const LlmResultMetadata& metadata) {
  OrderedJson output;
  output["processor_name"] = metadata.processor_name;
  output["macos_version"] = metadata.macos_version;
  output["performance_core_count"] = metadata.performance_core_count;
  output["efficiency_core_count"] = metadata.efficiency_core_count;
  output["logical_core_count"] = metadata.logical_core_count;
  output["page_size_bytes"] = decimal_string(metadata.page_size_bytes);
  output["l1_data_cache_bytes"] = decimal_string(metadata.l1_data_cache_bytes);
  output["l2_data_cache_bytes"] = decimal_string(metadata.l2_data_cache_bytes);
  output["available_memory_bytes"] =
      metadata.available_memory_bytes == 0 ? OrderedJson(nullptr) : decimal_string(metadata.available_memory_bytes);
  output["available_memory_source"] = metadata.available_memory_source;
  output["main_thread_qos"] = OrderedJson{{"requested", metadata.main_thread_qos.requested},
                                          {"applied", metadata.main_thread_qos.applied},
                                          {"code", metadata.main_thread_qos.code}};
  output["start"] = environment_snapshot_json(metadata.environment_start);
  output["end"] = environment_snapshot_json(metadata.environment_end);
  return output;
}

bool environment_is_non_nominal(const LlmHostEnvironmentSnapshot& snapshot) noexcept {
  const bool thermal_non_nominal = snapshot.thermal_state != "nominal" && snapshot.thermal_state != "unavailable";
  const bool low_power = snapshot.low_power_mode_available && snapshot.low_power_mode_enabled;
  return thermal_non_nominal || low_power;
}

void append_warning(std::vector<std::string>& warnings, std::string warning) {
  if (std::find(warnings.begin(), warnings.end(), warning) == warnings.end()) {
    warnings.push_back(std::move(warning));
  }
}

std::vector<std::string> collect_quality_warning_tokens(const LlmMemoryWorkPlan& plan,
                                                        const LlmResultMetadata& metadata,
                                                        const LlmMemoryResult& result) {
  std::vector<std::string> warnings;
  warnings.reserve(result.quality_warnings.size() + 9);
  for (std::string_view warning : result.quality_warnings) {
    append_warning(warnings, std::string(warning));
  }

  if (environment_is_non_nominal(metadata.environment_start) || environment_is_non_nominal(metadata.environment_end)) {
    append_warning(warnings, "environment-not-nominal");
  }
  if (metadata.main_thread_qos.requested && !metadata.main_thread_qos.applied) {
    append_warning(warnings, "main-thread-qos-not-applied");
  }
  if (std::any_of(result.measurements.begin(), result.measurements.end(),
                  [](const LlmMeasurementState& measurement) { return measurement.qos_failed_workers > 0; })) {
    append_warning(warnings, "worker-qos-not-applied");
  }
  if (metadata.l2_data_cache_bytes > 0 && plan.geometry.valid &&
      plan.geometry.active_weight_bytes_per_work_unit <= metadata.l2_data_cache_bytes) {
    append_warning(warnings, "weight-working-set-cache-dominant");
  }
  if (metadata.l2_data_cache_bytes > 0 && plan.geometry.valid &&
      plan.geometry.kv_capacity_bytes <= metadata.l2_data_cache_bytes) {
    append_warning(warnings, "kv-working-set-cache-dominant");
  }
  for (const LlmMeasurementState& measurement : result.measurements) {
    if (measurement.status != LlmMeasurementStatus::Measured ||
        measurement.duration_quality == "within-target-window") {
      continue;
    }
    append_warning(warnings, std::string(llm_scenario_to_string(measurement.scenario)) + "-duration-" +
                                 std::string(measurement.duration_quality));
  }

  return warnings;
}

OrderedJson quality_warnings_json(const LlmMemoryWorkPlan& plan, const LlmResultMetadata& metadata,
                                  const LlmMemoryResult& result) {
  OrderedJson output = OrderedJson::array();
  for (const std::string& warning : collect_llm_quality_warning_tokens(plan, metadata, result)) {
    output.push_back(warning);
  }
  return output;
}

OrderedJson interpretation_json(const LlmMemoryWorkPlan& plan) {
  OrderedJson comparability = OrderedJson::array();
  comparability.push_back("same-backend");
  comparability.push_back("same-methodology-version");
  comparability.push_back("same-frozen-work-plan-and-model-geometry");
  comparability.push_back("same-software-version");
  comparability.push_back("sufficiently-similar-hardware");
  comparability.push_back("sufficiently-similar-thermal-and-power-state");

  OrderedJson output;
  output["result_scope"] = "synthetic-memory-only-work-unit-not-real-inference-token-or-prefill-latency";
  output["reported_rate"] = "synthetic_memory_work_units_per_second";
  output["backend"] = llm_memory_backend_to_string(plan.backend);
  output["phase"] = llm_phase_to_string(plan.phase);
  output["work_unit_kind"] = llm_work_unit_kind_to_string(plan.work_unit_kind);
  output["transformer_math_included"] = false;
  output["framework_scheduler_and_dispatch_included"] = false;
  output["compute_memory_overlap_included"] = false;
  output["physical_dram_traffic_measured"] = false;
  output["dram_residency"] = "unverified";
  output["cache_residency"] = "unverified";
  output["fixed_context_includes_current_token_slot"] =
      plan.phase == LlmPhase::Decode ? OrderedJson(true) : OrderedJson(nullptr);
  output["kv_layout"] = llm_kv_layout_to_string(plan.kv_layout);
  output["payload_semantics"] = "exact-logical-weight-plus-kv-read-plus-kv-write-bytes-divided-by-elapsed-time";
  output["layout_metadata_timed_but_excluded_from_effective_model_payload_gb_s"] = true;
  output["prefill_transformer_compute_or_ttft_prediction_included"] = false;
  output["private_metal_storage_implies_separate_vram"] = false;
  output["cross_backend_performance_distributions_combined"] = false;
  output["traffic_classification_semantics"] = "exact-weight-vs-kv-read-logical-payload-only-not-hardware-bottleneck";
  output["full_size_working_set_reduces_but_does_not_prove_dram_residency"] = true;
  output["comparability_requires"] = std::move(comparability);
  return output;
}

}  // namespace

LlmJsonPeakEstimate calculate_llm_json_peak_estimate(const LlmMemoryConfig& config,
                                                     const LlmMemoryWorkPlan& model_plan) noexcept {
  LlmJsonPeakEstimate estimate;
  if (config.output_file.empty()) {
    estimate.valid = true;
    estimate.reason_code = LlmJsonReason::VALID;
    return estimate;
  }
  if (!model_plan.valid) {
    estimate.reason_code = LlmJsonReason::INVALID_MODEL_WORK_PLAN;
    return estimate;
  }

  size_t raw_input_bytes = config.output_file.size();
  const auto add_string_bytes = [&](const std::string& value) {
    return NumericUtils::checked_add(raw_input_bytes, value.size(), raw_input_bytes);
  };
  const auto add_optional_string_bytes = [&](const std::optional<std::string>& value) {
    return !value.has_value() || add_string_bytes(*value);
  };
  for (const std::string& argument : config.argv) {
    if (!add_string_bytes(argument)) {
      return estimate;
    }
  }
  const LlmComponentIdentities& components = model_plan.component_identities;
  if (!add_string_bytes(model_plan.plan_identity) || !add_string_bytes(model_plan.methodology_version) ||
      !add_string_bytes(components.logical_profile_version) || !add_string_bytes(components.kv_layout_version) ||
      !add_optional_string_bytes(components.permutation_version) ||
      !add_string_bytes(components.backend_executor_version) || !add_string_bytes(components.resource_abi_version) ||
      !add_string_bytes(components.schedule_version) || !add_string_bytes(components.timer_policy_version) ||
      !add_string_bytes(components.buffer_pattern_version) || !add_string_bytes(components.write_pattern_version) ||
      !add_string_bytes(components.checksum_pattern_version) || !add_optional_string_bytes(components.msl_revision) ||
      !add_optional_string_bytes(components.msl_source_sha256) || !add_string_bytes(components.identity)) {
    return estimate;
  }

  size_t planned_measurements = 0;
  size_t worker_checksum_records = 0;
  const LlmCpuExecutionPlan* cpu_execution_plan =
      get_llm_cpu_execution_plan(model_plan);
  if (cpu_execution_plan == nullptr) {
    estimate.reason_code = LlmJsonReason::INVALID_MODEL_WORK_PLAN;
    return estimate;
  }
  if (!NumericUtils::checked_multiply(config.loop_count, kLlmScenarioCount, planned_measurements) ||
      !NumericUtils::checked_multiply(planned_measurements, cpu_execution_plan->effective_workers,
                                      worker_checksum_records) ||
      !NumericUtils::checked_multiply(raw_input_bytes, kJsonInputStringExpansionFactor, estimate.input_string_bytes) ||
      !NumericUtils::checked_multiply(planned_measurements, kJsonMeasurementPeakBytes,
                                      estimate.measurement_record_bytes) ||
      !NumericUtils::checked_multiply(worker_checksum_records, kJsonWorkerChecksumPeakBytes,
                                      estimate.worker_checksum_bytes)) {
    return estimate;
  }

  estimate.fixed_schema_bytes = kJsonFixedSchemaPeakBytes;
  size_t total = estimate.fixed_schema_bytes;
  if (!NumericUtils::checked_add(total, estimate.input_string_bytes, total) ||
      !NumericUtils::checked_add(total, estimate.measurement_record_bytes, total) ||
      !NumericUtils::checked_add(total, estimate.worker_checksum_bytes, total)) {
    return estimate;
  }
  estimate.total_bytes = total;
  estimate.valid = true;
  estimate.reason_code = LlmJsonReason::VALID;
  return estimate;
}

const char* classify_llm_traffic_payload(const LlmGeometry& geometry) noexcept {
  if (!geometry.valid) {
    return "unavailable";
  }
  if (geometry.weight_read_bytes_per_work_unit == geometry.kv_read_bytes_per_work_unit) {
    return "near_crossover";
  }
  return geometry.weight_read_bytes_per_work_unit > geometry.kv_read_bytes_per_work_unit ? "weight_payload_dominant"
                                                                               : "kv_read_payload_dominant";
}

std::vector<std::string> collect_llm_quality_warning_tokens(const LlmMemoryWorkPlan& model_plan,
                                                            const LlmResultMetadata& metadata,
                                                            const LlmMemoryResult& result) {
  return collect_quality_warning_tokens(model_plan, metadata, result);
}

nlohmann::ordered_json build_llm_memory_json(const LlmMemoryConfig& config, const LlmMemoryWorkPlan& model_plan,
                                             const LlmBackendEvidence& backend_evidence,
                                             const LlmResultMetadata& metadata, const LlmMemoryResult& result) {
  const LlmResourcePreparationResult* preparation =
      get_llm_cpu_preparation(backend_evidence);
  const LlmMemoryBudget& admitted_memory_budget =
      preparation == nullptr ? model_plan.memory_budget
                             : preparation->memory_budget;
  OrderedJson output;
  output["schema_version"] = Constants::LLM_JSON_SCHEMA_VERSION;
  output["mode"] = Constants::LLM_JSON_MODE_NAME;
  output["backend"] = llm_memory_backend_to_string(model_plan.backend);
  output["phase"] = llm_phase_to_string(model_plan.phase);
  output["kv_layout"] = llm_kv_layout_to_string(model_plan.kv_layout);
  output["methodology_version"] = model_plan.methodology_version;
  output["software"] = software_json(metadata);
  output["configuration"] = configuration_json(config);
  output["resolved_plan"] = resolved_plan_json(model_plan, result.frozen_scenario_plans);
  output["backend_evidence"] = backend_evidence_json(model_plan, backend_evidence, metadata.json_peak_estimate,
                                                     !config.output_file.empty());
  output["memory_budget"] = memory_budget_json(admitted_memory_budget);
  output["calibration"] = calibration_json(result);
  output["measurements"] = measurements_json(result, model_plan);
  output["aggregates"] = aggregates_json(model_plan, result);
  output["status"] = llm_run_status_to_string(result.status);
  output["reason_code"] = result.reason_code;
  output["results_complete"] = result.results_complete;
  output["conclusions_valid"] = result.conclusions_valid;
  output["interpretation"] = interpretation_json(model_plan);

  // Additional lifecycle and audit evidence remains part of schema v1 but is
  // deliberately outside the minimum generic vocabulary frozen by Phase 0.
  output["diagnostic"] = non_empty_or_null(result.diagnostic);
  output["interruption_requested"] = result.interruption_requested;
  output["scenario_order_balance_complete"] = result.scenario_order_balance_complete;
  output["seeds"] = seeds_json(config, model_plan);
  output["counters"] = counters_json(result);
  output["checkpoint_lifecycle"] = checkpoint_lifecycle_json(result);
  output["loop_records"] = loop_records_json(result);
  output["environment"] = environment_json(metadata);
  output["quality_warnings"] = quality_warnings_json(model_plan, metadata, result);
  return output;
}

nlohmann::ordered_json build_llm_memory_json(
    const LlmMemoryConfig& config, const LlmMemoryWorkPlan& model_plan,
    const LlmResourcePreparationResult& preparation,
    const LlmResultMetadata& metadata, const LlmMemoryResult& result) {
  LlmBackendEvidence evidence;
  evidence.backend = LlmMemoryBackend::Cpu;
  evidence.backend_evidence = LlmCpuBackendEvidence{preparation};
  return build_llm_memory_json(config, model_plan, evidence, metadata, result);
}

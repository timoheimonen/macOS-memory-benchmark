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

/**
 * @file llm_output.cpp
 * @brief Human-readable output for the resolved LLM memory profile
 */

#include "llm_memory/llm_output.h"

#include <array>
#include <iostream>
#include <string>
#include <string_view>

#include "core/config/constants.h"
#include "llm_memory/llm_json.h"
#include "output/console/messages/messages_api.h"

namespace {

constexpr size_t scenario_index(LlmScenario scenario) noexcept { return static_cast<size_t>(scenario); }

const LlmScenarioAggregate& aggregate_for(const LlmMemoryResult& result, LlmScenario scenario) {
  return result.aggregates[scenario_index(scenario)];
}

const LlmPagedCpuExecutionPlan* paged_cpu_execution_plan(const LlmMemoryWorkPlan& plan) noexcept {
  const LlmCpuExecutionPlan* const cpu = get_llm_cpu_execution_plan(plan);
  if (plan.kv_layout != LlmKvLayout::Paged || cpu == nullptr || !cpu->paged.has_value()) {
    return nullptr;
  }
  return &*cpu->paged;
}

/** Return backend-neutral paged geometry without materializing a table. */
const LlmKvLayoutPlan* paged_layout_plan(const LlmMemoryWorkPlan& plan) noexcept {
  const LlmPagedCpuExecutionPlan* const cpu = paged_cpu_execution_plan(plan);
  if (cpu != nullptr) {
    return &cpu->layout;
  }
  const LlmMetalExecutionPlan* const metal = get_llm_metal_execution_plan(plan);
  if (plan.kv_layout != LlmKvLayout::Paged || metal == nullptr || !metal->resources.paged_layout.has_value()) {
    return nullptr;
  }
  return &*metal->resources.paged_layout;
}

/** Return the materialized permutation identity published by the selected backend. */
const LlmKvPermutationIdentity* paged_permutation_identity(
    const LlmMemoryWorkPlan& plan, const LlmBackendEvidence& backend_evidence) noexcept {
  const LlmPagedCpuExecutionPlan* const cpu = paged_cpu_execution_plan(plan);
  if (cpu != nullptr) {
    return &cpu->permutation;
  }
  const LlmMetalBackendEvidence* const metal = get_llm_metal_backend_evidence(backend_evidence);
  if (plan.backend != LlmMemoryBackend::Metal || plan.kv_layout != LlmKvLayout::Paged || metal == nullptr ||
      !metal->resources.table_permutation.has_value()) {
    return nullptr;
  }
  return &*metal->resources.table_permutation;
}

void print_paged_layout_evidence(const LlmMemoryWorkPlan& plan, const LlmBackendEvidence& backend_evidence,
                                 std::string_view work_unit_name) {
  const LlmKvLayoutPlan* const layout = paged_layout_plan(plan);
  const LlmKvPermutationIdentity* const permutation = paged_permutation_identity(plan, backend_evidence);
  if (layout == nullptr || permutation == nullptr) {
    return;
  }
  Messages::LlmPagedLayoutReportValues values;
  values.block_tokens = layout->kv_block_tokens;
  values.blocks_per_sequence = layout->blocks_per_sequence;
  values.physical_blocks_per_layer = layout->physical_blocks_per_layer;
  values.total_physical_blocks = layout->total_physical_blocks;
  values.block_bytes = layout->block_bytes;
  values.terminal_block_tokens = layout->last_block_tokens;
  values.terminal_valid_bytes = layout->last_block_valid_bytes;
  values.k_logical_bytes = plan.geometry.k_logical_bytes;
  values.k_physical_bytes = plan.geometry.k_mapping_bytes;
  values.k_padding_bytes = plan.geometry.k_layout_padding_bytes;
  values.v_logical_bytes = plan.geometry.v_logical_bytes;
  values.v_physical_bytes = plan.geometry.v_mapping_bytes;
  values.v_padding_bytes = plan.geometry.v_layout_padding_bytes;
  values.block_table_entries = layout->block_table_entries;
  const LlmPagedCpuExecutionPlan* const cpu = paged_cpu_execution_plan(plan);
  const LlmMetalExecutionPlan* const metal = get_llm_metal_execution_plan(plan);
  values.block_table_bytes = cpu != nullptr ? cpu->block_table_logical_bytes : layout->memory.block_table_bytes;
  values.block_table_page_rounded_bytes =
      cpu != nullptr ? cpu->block_table_mapping_bytes
                     : metal != nullptr ? metal->resources.host_permutation_mapping_bytes : 0;
  values.permutation_version = permutation->algorithm_version;
  values.permutation_seed = permutation->resolved_seed;
  values.permutation_sha256 = permutation->sha256;
  values.permutation_identity = permutation->identity;
  const LlmScenarioLimits kv_only_limits = calculate_llm_scenario_limits(plan.geometry, LlmScenario::KvOnly);
  if (kv_only_limits.valid) {
    values.metadata_lookups_per_work_unit = kv_only_limits.layout_metadata_lookup_count_per_work_unit;
    values.metadata_bytes_per_work_unit = kv_only_limits.layout_metadata_read_bytes_per_work_unit;
    values.accounted_bytes_per_work_unit = kv_only_limits.accounted_bytes_per_work_unit;
  } else if (cpu != nullptr && plan.phase == LlmPhase::Decode) {
    values.metadata_lookups_per_work_unit = cpu->ownership.total_layout_metadata_lookup_count_per_work_unit;
    values.metadata_bytes_per_work_unit = cpu->ownership.total_layout_metadata_read_bytes_per_work_unit;
    values.accounted_bytes_per_work_unit = cpu->ownership.total_accounted_bytes_per_work_unit;
  }
  values.work_unit_name = work_unit_name;
  std::cout << Messages::report_llm_memory_paged_layout(values) << '\n';
}

void print_metal_backend_evidence(const LlmMemoryWorkPlan& plan, const LlmBackendEvidence& backend_evidence) {
  if (plan.backend != LlmMemoryBackend::Metal) {
    return;
  }
  const LlmMetalBackendEvidence* const metal = get_llm_metal_backend_evidence(backend_evidence);
  if (metal == nullptr) {
    return;
  }
  const LlmMetalCapabilityEvidence& capability = metal->capability;
  if (!capability.device_name.empty()) {
    std::cout << Messages::report_llm_memory_metal_backend(
                     capability.device_name, capability.registry_id, capability.required_apple7_family_supported,
                     capability.has_unified_memory, capability.argument_buffers_tier2_supported,
                     capability.max_buffer_length, capability.recommended_max_working_set_size)
              << '\n';
  }

  const LlmMetalExecutionPlan* const execution = get_llm_metal_execution_plan(plan);
  if (execution == nullptr || !execution->valid || !execution->resources.valid) {
    return;
  }
  const LlmMetalResourcePlan& planned = execution->resources;
  const LlmMetalResourceEvidence& actual = metal->resources;
  std::cout << Messages::report_llm_memory_metal_resources(
                   planned.weight_segments.segment_count, planned.k_segments.segment_count,
                   planned.v_segments.segment_count, planned.limits.segment_capacity_bytes,
                   capability.argument_buffer_encoded_length, actual.committed_resource_bytes,
                   actual.known_owned_peak_bytes, actual.admitted_budget_bytes)
            << '\n';
}

void print_metal_task_evidence(const LlmMemoryWorkPlan& plan, const LlmMemoryResult& result) {
  if (plan.backend != LlmMemoryBackend::Metal) {
    return;
  }
  std::array<bool, kLlmScenarioCount> printed{};
  for (const LlmMeasurementState& measurement : result.measurements) {
    const size_t index = scenario_index(measurement.scenario);
    if (index >= printed.size() || printed[index]) {
      continue;
    }
    const LlmMetalTaskEvidence* const metal = get_llm_metal_task_evidence(measurement.execution);
    if (metal == nullptr) {
      continue;
    }
    const size_t threadgroups = metal->grid_plan_available ? metal->grid_plan.actual_threadgroups : 0;
    const size_t threads_per_threadgroup = metal->grid_plan_available ? metal->grid_plan.threads_per_threadgroup : 0;
    std::cout << Messages::report_llm_memory_metal_task(
                     llm_scenario_to_string(measurement.scenario), metal->pipeline_label, threadgroups,
                     threads_per_threadgroup, metal->timing_evaluated,
                     metal->timing_valid, metal->gpu_elapsed_seconds,
                     metal->checksum_evaluated,
                     metal->checksum_valid, measurement.scenario != LlmScenario::WeightsOnly,
                     metal->kv_write_validation_evaluated,
                     metal->kv_write_validation_valid,
                     metal->padding_canary_applicable, metal->padding_canary_evaluated, metal->padding_canary_valid)
              << '\n';
    printed[index] = true;
  }
}

void emit_quality_warning(std::string_view token, const LlmMemoryWorkPlan& model_plan,
                          const LlmResultMetadata& metadata, const LlmMemoryResult& result) {
  if (token == "weights_only-high-cv") {
    const LlmScenarioAggregate& aggregate = aggregate_for(result, LlmScenario::WeightsOnly);
    std::cerr << Messages::warning_prefix()
              << Messages::warning_llm_memory_high_cv(
                     llm_scenario_to_string(LlmScenario::WeightsOnly),
                     aggregate.effective_model_payload_gb_s.statistics.coefficient_of_variation_pct,
                     Constants::LLM_STREAMING_CV_WARNING_PCT)
              << std::endl;
    return;
  }
  if (token == "kv_only-high-cv") {
    const LlmScenarioAggregate& aggregate = aggregate_for(result, LlmScenario::KvOnly);
    std::cerr << Messages::warning_prefix()
              << Messages::warning_llm_memory_high_cv(
                     llm_scenario_to_string(LlmScenario::KvOnly),
                     aggregate.effective_model_payload_gb_s.statistics.coefficient_of_variation_pct,
                     Constants::LLM_STREAMING_CV_WARNING_PCT)
              << std::endl;
    return;
  }
  if (token == "mixed-high-cv") {
    const LlmScenarioAggregate& aggregate = aggregate_for(result, LlmScenario::Mixed);
    std::cerr << Messages::warning_prefix()
              << Messages::warning_llm_memory_high_cv(
                     llm_scenario_to_string(LlmScenario::Mixed),
                     aggregate.effective_model_payload_gb_s.statistics.coefficient_of_variation_pct,
                     Constants::LLM_STREAMING_CV_WARNING_PCT)
              << std::endl;
    return;
  }
  if (token == "scenario-order-not-balanced") {
    std::cerr << Messages::warning_prefix() << Messages::warning_llm_memory_order_not_balanced() << std::endl;
    return;
  }
  if (token == "environment-not-nominal") {
    std::cerr << Messages::warning_prefix() << Messages::warning_llm_memory_environment_not_nominal() << std::endl;
    return;
  }
  if (token == "main-thread-qos-not-applied") {
    std::cerr << Messages::warning_prefix()
              << Messages::warning_llm_memory_main_thread_qos_not_applied(metadata.main_thread_qos.code) << std::endl;
    return;
  }
  if (token == "worker-qos-not-applied") {
    std::cerr << Messages::warning_prefix() << Messages::warning_llm_memory_worker_qos_not_applied() << std::endl;
    return;
  }
  if (token == "weight-working-set-cache-dominant") {
    std::cerr << Messages::warning_prefix()
              << Messages::warning_llm_memory_weight_cache_dominant(
                     model_plan.geometry.active_weight_bytes_per_work_unit, metadata.l2_data_cache_bytes)
              << std::endl;
    return;
  }
  if (token == "kv-working-set-cache-dominant") {
    std::cerr << Messages::warning_prefix()
              << Messages::warning_llm_memory_kv_cache_dominant(model_plan.geometry.kv_capacity_bytes,
                                                                metadata.l2_data_cache_bytes)
              << std::endl;
    return;
  }

  for (LlmScenario scenario : {LlmScenario::WeightsOnly, LlmScenario::KvOnly, LlmScenario::Mixed}) {
    const std::string scenario_token = llm_scenario_to_string(scenario);
    const std::string prefix = scenario_token + "-duration-";
    if (token.substr(0, prefix.size()) != prefix) {
      continue;
    }
    std::cerr << Messages::warning_prefix()
              << Messages::warning_llm_memory_duration_quality(scenario_token, std::string(token.substr(prefix.size())))
              << std::endl;
    return;
  }
}

void print_headline(const LlmScenarioAggregate& aggregate, std::string_view work_unit_name,
                    std::string_view plural_work_unit_name) {
  if (!aggregate.work_unit_latency_seconds.headline.has_value() ||
      !aggregate.effective_model_payload_gb_s.headline.has_value()) {
    return;
  }

  const bool include_work_units_per_second = aggregate.scenario == LlmScenario::Mixed;
  if (include_work_units_per_second && !aggregate.synthetic_memory_work_units_per_second.headline.has_value()) {
    return;
  }
  const double work_units_per_second = aggregate.synthetic_memory_work_units_per_second.headline.value_or(0.0);
  const std::string scenario_name =
      Messages::report_llm_memory_scenario_name(llm_scenario_to_string(aggregate.scenario));
  std::cout << Messages::report_llm_memory_scenario_headline(
                   scenario_name, std::string(work_unit_name), std::string(plural_work_unit_name),
                   *aggregate.work_unit_latency_seconds.headline * 1000.0, work_units_per_second,
                   *aggregate.effective_model_payload_gb_s.headline, include_work_units_per_second)
            << std::endl;
}

}  // namespace

void print_llm_memory_console_report(const LlmMemoryWorkPlan& model_plan, const LlmBackendEvidence& backend_evidence,
                                     const LlmResultMetadata& metadata, const LlmMemoryResult& result) {
  const std::string work_unit_kind = llm_work_unit_kind_to_string(model_plan.work_unit_kind);
  const std::string work_unit_name = Messages::report_llm_memory_work_unit_name(work_unit_kind, false);
  const std::string plural_work_unit_name = Messages::report_llm_memory_work_unit_name(work_unit_kind, true);

  std::cout << Messages::report_llm_memory_header(llm_memory_backend_to_string(model_plan.backend),
                                                  llm_phase_to_string(model_plan.phase), work_unit_kind,
                                                  llm_kv_layout_to_string(model_plan.kv_layout))
            << std::endl;
  print_metal_backend_evidence(model_plan, backend_evidence);
  if (model_plan.geometry.decode.has_value()) {
    std::cout << Messages::report_llm_memory_decode_geometry(model_plan.geometry.decode->visible_context_tokens,
                                                             model_plan.geometry.traffic_crossover_context_tokens)
              << std::endl;
  }
  if (model_plan.geometry.prefill.has_value()) {
    const LlmPrefillGeometry& prefill = *model_plan.geometry.prefill;
    std::cout << Messages::report_llm_memory_prefill_geometry(
                     prefill.prompt_tokens, prefill.attention_query_tile_tokens, prefill.tile_count,
                     prefill.attention_prefix_token_visits_per_sequence, prefill.causal_token_pairs_per_sequence,
                     prefill.logical_attention_pairs, prefill.logical_attention_fma_terms)
              << std::endl;
  }
  print_paged_layout_evidence(model_plan, backend_evidence, work_unit_name);
  std::cout << Messages::report_llm_memory_payload(
                   work_unit_name, model_plan.geometry.active_weight_bytes_per_work_unit,
                   model_plan.geometry.kv_read_bytes_per_work_unit, model_plan.geometry.kv_write_bytes_per_work_unit)
            << "\n\n";
  for (const LlmScenarioAggregate& aggregate : result.aggregates) {
    print_headline(aggregate, work_unit_name, plural_work_unit_name);
  }
  print_metal_task_evidence(model_plan, result);
  std::cout << Messages::report_llm_memory_interpretation_note(
                   llm_phase_to_string(model_plan.phase), llm_kv_layout_to_string(model_plan.kv_layout), work_unit_name)
            << std::endl;
  std::cout.flush();

  for (const std::string& token : collect_llm_quality_warning_tokens(model_plan, metadata, result)) {
    emit_quality_warning(token, model_plan, metadata, result);
  }
}

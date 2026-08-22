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
 * @file llm_work_plan.cpp
 * @brief Pure checked logical and backend-specific LLM work planning
 */

#include "llm_memory/llm_work_plan.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

#include "utils/numeric_utils.h"
#include "utils/seed_utils.h"

namespace {

constexpr uint64_t kWeightBufferSeedDomain = 0x4C4C4D5745494748ULL;
constexpr uint64_t kKBufferSeedDomain = 0x4C4C4D4B42554631ULL;
constexpr uint64_t kVBufferSeedDomain = 0x4C4C4D5642554631ULL;
constexpr uint64_t kWeightsOnlyScenarioSeedDomain = 0x4C4C4D5357454947ULL;
constexpr uint64_t kKvOnlyScenarioSeedDomain = 0x4C4C4D534B564F4EULL;
constexpr uint64_t kMixedScenarioSeedDomain = 0x4C4C4D534D495845ULL;

bool valid_kv_element_bytes(size_t bytes) {
  return bytes == 1 || bytes == 2 || bytes == 4;
}

bool json_integer_is_safe(size_t value) {
  return value <= Constants::LLM_JSON_MAX_SAFE_INTEGER;
}

LlmAttentionKind classify_attention(size_t query_heads, size_t kv_heads) {
  if (query_heads == kv_heads) {
    return LlmAttentionKind::Mha;
  }
  return kv_heads == 1 ? LlmAttentionKind::Mqa : LlmAttentionKind::Gqa;
}

size_t scenario_index(LlmScenario scenario) {
  return static_cast<size_t>(scenario);
}

bool valid_scenario(LlmScenario scenario) {
  return scenario_index(scenario) < kLlmScenarioCount;
}

/**
 * @brief Split one exact range while prioritizing aligned internal boundaries.
 *
 * Active returned spans are non-empty. If a particular layer or sequence is
 * smaller than the effective team, its exact bytes go to the lowest-indexed
 * workers and the remaining workers receive canonical empty ranges. When any
 * 32-byte-aligned absolute boundary can leave at least one byte for every
 * remaining active worker, the closest such boundary to a quotient/remainder
 * split is selected. The union is exactly the input range.
 */
std::vector<LlmByteRange> partition_range(size_t offset, size_t span,
                                          size_t worker_count) {
  std::vector<LlmByteRange> ranges;
  if (worker_count == 0) {
    return ranges;
  }

  ranges.reserve(worker_count);
  const size_t active_workers = std::min(span, worker_count);
  if (active_workers == 0) {
    ranges.resize(worker_count);
    return ranges;
  }
  const size_t base = span / active_workers;
  const size_t remainder = span % active_workers;
  size_t previous_local = 0;
  for (size_t boundary_index = 1; boundary_index < active_workers;
       ++boundary_index) {
    const size_t balanced_local =
        base * boundary_index + std::min(boundary_index, remainder);
    const size_t minimum_local = previous_local + 1;
    const size_t remaining_workers = active_workers - boundary_index;
    const size_t maximum_local = span - remaining_workers;
    size_t selected_local =
        std::clamp(balanced_local, minimum_local, maximum_local);

    const size_t minimum_absolute = offset + minimum_local;
    const size_t maximum_absolute = offset + maximum_local;
    const size_t remainder_to_alignment =
        minimum_absolute % Constants::LLM_RANGE_ALIGNMENT_BYTES;
    const size_t alignment_delta =
        remainder_to_alignment == 0
            ? 0
            : Constants::LLM_RANGE_ALIGNMENT_BYTES - remainder_to_alignment;
    if (minimum_absolute <=
        std::numeric_limits<size_t>::max() - alignment_delta) {
      const size_t first_aligned = minimum_absolute + alignment_delta;
      const size_t last_aligned =
          maximum_absolute -
          maximum_absolute % Constants::LLM_RANGE_ALIGNMENT_BYTES;
      if (first_aligned <= last_aligned) {
        const size_t balanced_absolute = offset + selected_local;
        size_t aligned_down =
            balanced_absolute -
            balanced_absolute % Constants::LLM_RANGE_ALIGNMENT_BYTES;
        aligned_down = std::clamp(aligned_down, first_aligned, last_aligned);
        size_t aligned_up = aligned_down;
        if (aligned_down < balanced_absolute &&
            aligned_down <=
                std::numeric_limits<size_t>::max() -
                    Constants::LLM_RANGE_ALIGNMENT_BYTES) {
          aligned_up = std::min(
              last_aligned,
              aligned_down + Constants::LLM_RANGE_ALIGNMENT_BYTES);
        }
        const size_t distance_down = balanced_absolute >= aligned_down
                                         ? balanced_absolute - aligned_down
                                         : aligned_down - balanced_absolute;
        const size_t distance_up = balanced_absolute >= aligned_up
                                       ? balanced_absolute - aligned_up
                                       : aligned_up - balanced_absolute;
        const size_t selected_absolute =
            distance_down <= distance_up ? aligned_down : aligned_up;
        selected_local = selected_absolute - offset;
      }
    }

    ranges.push_back(
        {offset + previous_local, selected_local - previous_local});
    previous_local = selected_local;
  }
  ranges.push_back({offset + previous_local, span - previous_local});
  ranges.resize(worker_count);
  return ranges;
}

LlmByteRange intersect_ranges(const LlmByteRange& lhs,
                              const LlmByteRange& rhs) {
  const size_t lhs_end = lhs.offset_bytes + lhs.span_bytes;
  const size_t rhs_end = rhs.offset_bytes + rhs.span_bytes;
  const size_t start = std::max(lhs.offset_bytes, rhs.offset_bytes);
  const size_t end = std::min(lhs_end, rhs_end);
  return end > start ? LlmByteRange{start, end - start} : LlmByteRange{};
}

void append_identity_field(std::string& identity, const char* name,
                           const std::string& value) {
  identity += '|';
  identity += name;
  identity += '=';
  identity += value;
}

void append_identity_field(std::string& identity, const char* name,
                           const char* value) {
  append_identity_field(identity, name, std::string(value));
}

template <typename Integer>
void append_identity_field(std::string& identity, const char* name,
                           Integer value) {
  append_identity_field(identity, name, std::to_string(value));
}

void append_component_identity(std::string& identity, const char* name,
                               const std::string& value) {
  identity += '|';
  identity += name;
  identity += '=';
  identity += std::to_string(value.size());
  identity += ':';
  identity += value;
}

void append_component_identity(
    std::string& identity, const char* name,
    const std::optional<std::string>& value) {
  if (value.has_value()) {
    append_component_identity(identity, name, *value);
    return;
  }
  identity += '|';
  identity += name;
  identity += "=null";
}

std::string build_model_plan_identity(const LlmMemoryWorkPlan& plan) {
  const LlmCpuExecutionPlan* const cpu_plan =
      get_llm_cpu_execution_plan(plan);
  if (cpu_plan == nullptr) {
    return {};
  }
  const LlmGeometry& geometry = plan.geometry;
  std::string identity = Constants::LLM_WORK_PLAN_IDENTITY_VERSION;
  append_identity_field(identity, "backend",
                        llm_memory_backend_to_string(plan.backend));
  append_identity_field(identity, "phase", llm_phase_to_string(plan.phase));
  append_identity_field(identity, "kv_layout",
                        llm_kv_layout_to_string(plan.kv_layout));
  append_identity_field(identity, "work_unit_kind",
                        llm_work_unit_kind_to_string(plan.work_unit_kind));
  append_identity_field(identity, "methodology", plan.methodology_version);
  append_identity_field(identity, "component_identity_size",
                        plan.component_identities.identity.size());
  append_identity_field(identity, "component_identity",
                        plan.component_identities.identity);
  append_identity_field(identity, "range_alignment",
                        Constants::LLM_RANGE_ALIGNMENT_BYTES);
  append_identity_field(identity, "weight_passes_per_work_unit",
                        plan.weight_passes_per_work_unit);
  append_identity_field(identity, "kv_replay_factor",
                        plan.kv_replay_factor);
  append_identity_field(identity, "weight",
                        geometry.active_weight_bytes_per_work_unit);
  append_identity_field(identity, "layers", geometry.layer_count);
  append_identity_field(identity, "query_heads", geometry.query_head_count);
  append_identity_field(identity, "kv_heads", geometry.kv_head_count);
  append_identity_field(identity, "query_heads_per_kv_head",
                        geometry.query_heads_per_kv_head);
  append_identity_field(identity, "attention_kind",
                        llm_attention_kind_to_string(geometry.attention_kind));
  append_identity_field(identity, "head_dim", geometry.head_dimension);
  append_identity_field(identity, "kv_element_bytes",
                        geometry.kv_element_bytes);
  append_identity_field(identity, "decode_context",
                        geometry.decode->visible_context_tokens);
  append_identity_field(identity, "batch", geometry.batch_size);
  append_identity_field(identity, "kv_vector_bytes",
                        geometry.kv_vector_bytes);
  append_identity_field(identity, "k_or_v_record_bytes_per_layer",
                        geometry.k_or_v_record_bytes_per_layer);
  append_identity_field(identity, "kv_record_bytes_per_layer",
                        geometry.kv_record_bytes_per_layer);
  append_identity_field(identity, "kv_bytes_per_visible_token",
                        geometry.kv_bytes_per_visible_token);
  append_identity_field(identity, "k_or_v_sequence_visible_bytes",
                        geometry.k_or_v_sequence_visible_bytes);
  append_identity_field(identity, "kv_block_tokens",
                        geometry.kv_block_tokens);
  append_identity_field(identity, "kv_blocks_per_sequence",
                        geometry.kv_blocks_per_sequence);
  append_identity_field(identity, "physical_blocks_per_layer",
                        geometry.physical_blocks_per_layer);
  append_identity_field(identity, "total_physical_blocks",
                        geometry.total_physical_blocks);
  append_identity_field(identity, "kv_block_bytes", geometry.kv_block_bytes);
  append_identity_field(identity, "last_block_tokens",
                        geometry.last_block_tokens);
  append_identity_field(identity, "last_block_valid_bytes",
                        geometry.last_block_valid_bytes);
  append_identity_field(identity, "decode_append_offset_in_last_block",
                        geometry.decode_append_offset_in_last_block);
  append_identity_field(identity, "k_logical_bytes", geometry.k_logical_bytes);
  append_identity_field(identity, "v_logical_bytes", geometry.v_logical_bytes);
  append_identity_field(identity, "k_layout_padding_bytes",
                        geometry.k_layout_padding_bytes);
  append_identity_field(identity, "v_layout_padding_bytes",
                        geometry.v_layout_padding_bytes);
  append_identity_field(identity, "block_table_entries",
                        geometry.block_table_entries);
  append_identity_field(identity, "block_table_bytes",
                        geometry.block_table_bytes);
  append_identity_field(identity, "k_mapping_bytes",
                        geometry.k_mapping_bytes);
  append_identity_field(identity, "v_mapping_bytes",
                        geometry.v_mapping_bytes);
  append_identity_field(identity, "kv_capacity_bytes",
                        geometry.kv_capacity_bytes);
  append_identity_field(identity, "weight_read_bytes_per_work_unit",
                        geometry.weight_read_bytes_per_work_unit);
  append_identity_field(identity, "kv_read_bytes_per_work_unit",
                        geometry.kv_read_bytes_per_work_unit);
  append_identity_field(identity, "kv_write_bytes_per_work_unit",
                        geometry.kv_write_bytes_per_work_unit);
  append_identity_field(identity, "kv_only_payload_bytes_per_work_unit",
                        geometry.kv_only_effective_model_payload_bytes_per_work_unit);
  append_identity_field(identity, "mixed_payload_bytes_per_work_unit",
                        geometry.mixed_effective_model_payload_bytes_per_work_unit);
  append_identity_field(identity, "total_data_mapping_bytes",
                        geometry.total_data_mapping_bytes);
  append_identity_field(identity, "traffic_crossover_numerator",
                        geometry.traffic_crossover_numerator);
  append_identity_field(identity, "traffic_crossover_denominator",
                        geometry.traffic_crossover_denominator);
  append_identity_field(identity, "requested_workers",
                        cpu_plan->requested_workers);
  append_identity_field(identity, "effective_workers",
                        cpu_plan->effective_workers);
  append_identity_field(identity, "layer_descriptors_per_worker",
                        cpu_plan->layer_descriptors_per_worker);
  append_identity_field(identity, "sequence_descriptors_per_worker",
                        cpu_plan->sequence_descriptors_per_worker);
  append_identity_field(identity, "total_layer_descriptors",
                        cpu_plan->total_layer_descriptors);
  append_identity_field(identity, "total_sequence_descriptors",
                        cpu_plan->total_sequence_descriptors);
  append_identity_field(identity, "descriptor_bytes",
                        cpu_plan->descriptor_bytes);
  if (cpu_plan->paged.has_value()) {
    append_identity_field(identity, "paged_layout_identity_size",
                          cpu_plan->paged->layout_identity.size());
    append_identity_field(identity, "paged_layout_identity",
                          cpu_plan->paged->layout_identity);
    append_identity_field(identity, "paged_execution_identity_size",
                          cpu_plan->paged->execution_identity.size());
    append_identity_field(identity, "paged_execution_identity",
                          cpu_plan->paged->execution_identity);
  }
  append_identity_field(identity, "base_seed", plan.base_seed);
  append_identity_field(identity, "weight_buffer_seed",
                        plan.weight_buffer_seed);
  append_identity_field(identity, "k_buffer_seed", plan.k_buffer_seed);
  append_identity_field(identity, "v_buffer_seed", plan.v_buffer_seed);
  append_identity_field(identity, "weights_only_scenario_seed",
                        plan.scenario_seeds[0]);
  append_identity_field(identity, "kv_only_scenario_seed",
                        plan.scenario_seeds[1]);
  append_identity_field(identity, "mixed_scenario_seed",
                        plan.scenario_seeds[2]);
  return identity;
}

std::string build_scenario_plan_identity(const LlmScenarioWorkPlan& plan) {
  std::string identity = Constants::LLM_WORK_PLAN_IDENTITY_VERSION;
  append_identity_field(identity, "model_plan_identity_size",
                        plan.model_plan_identity.size());
  append_identity_field(identity, "model_plan_identity",
                        plan.model_plan_identity);
  append_identity_field(identity, "scenario",
                        llm_scenario_to_string(plan.scenario));
  append_identity_field(identity, "work_unit_kind",
                        llm_work_unit_kind_to_string(plan.work_unit_kind));
  append_identity_field(identity, "kv_write_kind",
                        llm_kv_write_kind_to_string(plan.kv_write_kind));
  append_identity_field(identity, "scenario_seed", plan.scenario_seed);
  append_identity_field(identity, "explicit",
                        plan.explicit_iterations ? 1 : 0);
  append_identity_field(identity, "work_units", plan.work_units);
  append_identity_field(identity, "weight_read_bytes_per_work_unit",
                        plan.weight_read_bytes_per_work_unit);
  append_identity_field(identity, "kv_read_bytes_per_work_unit",
                        plan.kv_read_bytes_per_work_unit);
  append_identity_field(identity, "kv_write_bytes_per_work_unit",
                        plan.kv_write_bytes_per_work_unit);
  append_identity_field(identity, "effective_model_payload_bytes_per_work_unit",
                        plan.effective_model_payload_bytes_per_work_unit);
  append_identity_field(identity,
                        "layout_metadata_lookup_count_per_work_unit",
                        plan.layout_metadata_lookup_count_per_work_unit);
  append_identity_field(identity,
                        "layout_metadata_read_bytes_per_work_unit",
                        plan.layout_metadata_read_bytes_per_work_unit);
  append_identity_field(identity, "accounted_bytes_per_work_unit",
                        plan.accounted_bytes_per_work_unit);
  append_identity_field(identity, "weight_read_bytes",
                        plan.weight_read_bytes);
  append_identity_field(identity, "kv_read_bytes", plan.kv_read_bytes);
  append_identity_field(identity, "kv_write_bytes",
                        plan.kv_write_bytes);
  append_identity_field(identity, "effective_model_payload_bytes",
                        plan.effective_model_payload_bytes);
  append_identity_field(identity, "layout_metadata_lookup_count",
                        plan.layout_metadata_lookup_count);
  append_identity_field(identity, "layout_metadata_read_bytes",
                        plan.layout_metadata_read_bytes);
  append_identity_field(identity, "task_accounted_bytes",
                        plan.task_accounted_bytes);
  append_identity_field(identity, "maximum_work_units_by_work_unit_cap",
                        plan.maximum_work_units_by_work_unit_cap);
  append_identity_field(identity, "maximum_work_units_by_guardrail",
                        plan.maximum_work_units_by_guardrail);
  append_identity_field(identity, "effective_maximum_work_units",
                        plan.effective_maximum_work_units);
  return identity;
}

bool calculate_planner_storage_bytes(size_t layer_count,
                                     size_t sequence_count_per_worker,
                                     size_t worker_count,
                                     bool paged_layout,
                                     size_t ownership_assignment_count,
                                     size_t& planner_storage_bytes) {
  size_t weight_layer_bytes = 0;
  size_t worker_object_bytes = 0;
  size_t layer_template_count = 0;
  size_t layer_template_bytes = 0;
  size_t sequence_template_count = 0;
  size_t sequence_template_bytes = 0;
  size_t ownership_assignment_bytes = 0;
  size_t ownership_worker_bytes = 0;
  if (!NumericUtils::checked_multiply(layer_count, sizeof(LlmByteRange),
                                      weight_layer_bytes) ||
      !NumericUtils::checked_multiply(worker_count,
                                      sizeof(LlmWorkerWorkPlan),
                                      worker_object_bytes) ||
      !NumericUtils::checked_multiply(worker_count, layer_count,
                                      layer_template_count) ||
      !NumericUtils::checked_multiply(layer_template_count,
                                      sizeof(LlmLayerRangeTemplate),
                                      layer_template_bytes) ||
      !NumericUtils::checked_multiply(worker_count,
                                      sequence_count_per_worker,
                                      sequence_template_count) ||
      !NumericUtils::checked_multiply(
          sequence_template_count,
          paged_layout ? sizeof(LlmPagedKvAssignmentTemplate)
                       : sizeof(LlmKvSequenceRangeTemplate),
          sequence_template_bytes) ||
      !NumericUtils::checked_multiply(
          ownership_assignment_count, sizeof(LlmKvCpuBlockAssignment),
          ownership_assignment_bytes) ||
      !NumericUtils::checked_multiply(
          paged_layout ? worker_count : 0, sizeof(size_t),
          ownership_worker_bytes)) {
    return false;
  }

  size_t outer_storage_bytes = 0;
  size_t template_storage_bytes = 0;
  size_t ownership_storage_bytes = 0;
  return NumericUtils::checked_add(weight_layer_bytes, worker_object_bytes,
                                   outer_storage_bytes) &&
         NumericUtils::checked_add(layer_template_bytes,
                                   sequence_template_bytes,
                                   template_storage_bytes) &&
         NumericUtils::checked_add(ownership_assignment_bytes,
                                   ownership_worker_bytes,
                                   ownership_storage_bytes) &&
         NumericUtils::checked_add(outer_storage_bytes,
                                   template_storage_bytes,
                                   planner_storage_bytes) &&
         NumericUtils::checked_add(planner_storage_bytes,
                                   ownership_storage_bytes,
                                   planner_storage_bytes);
}

bool add_allocation_capacity(size_t capacity, size_t element_bytes,
                             size_t& total_bytes) {
  size_t allocation_bytes = 0;
  return NumericUtils::checked_multiply(capacity, element_bytes,
                                        allocation_bytes) &&
         NumericUtils::checked_add(total_bytes, allocation_bytes,
                                   total_bytes);
}

bool calculate_actual_planner_storage_bytes(
    const LlmMemoryWorkPlan& plan, size_t& planner_storage_bytes) {
  const LlmCpuExecutionPlan* const cpu_plan =
      get_llm_cpu_execution_plan(plan);
  if (cpu_plan == nullptr) {
    return false;
  }
  planner_storage_bytes = 0;
  if (!add_allocation_capacity(plan.weight_layers.capacity(),
                               sizeof(LlmByteRange),
                               planner_storage_bytes) ||
      !add_allocation_capacity(cpu_plan->workers.capacity(),
                               sizeof(LlmWorkerWorkPlan),
                               planner_storage_bytes)) {
    return false;
  }
  for (const LlmWorkerWorkPlan& worker : cpu_plan->workers) {
    if (!add_allocation_capacity(worker.layers.capacity(),
                                 sizeof(LlmLayerRangeTemplate),
                                 planner_storage_bytes) ||
        !add_allocation_capacity(worker.sequences.capacity(),
                                 sizeof(LlmKvSequenceRangeTemplate),
                                 planner_storage_bytes) ||
        !add_allocation_capacity(
            worker.paged_assignments.capacity(),
            sizeof(LlmPagedKvAssignmentTemplate),
                                 planner_storage_bytes)) {
      return false;
    }
  }
  if (cpu_plan->paged.has_value() &&
      (!add_allocation_capacity(
           cpu_plan->paged->ownership.assignments.capacity(),
           sizeof(LlmKvCpuBlockAssignment), planner_storage_bytes) ||
       !add_allocation_capacity(
           cpu_plan->paged->ownership.worker_accounted_bytes_per_work_unit.capacity(),
           sizeof(size_t), planner_storage_bytes))) {
    return false;
  }
  return true;
}

bool finalize_plan_identities(LlmMemoryWorkPlan& plan) {
  const bool paged_layout = plan.kv_layout == LlmKvLayout::Paged;
  plan.methodology_version =
      build_llm_methodology_version(plan.backend, plan.phase,
                                   plan.kv_layout);
  plan.component_identities.logical_profile_version =
      Constants::LLM_LOGICAL_PROFILE_VERSION;
  plan.component_identities.kv_layout_version =
      paged_layout ? Constants::LLM_PAGED_KV_LAYOUT_VERSION
                   : Constants::LLM_CONTIGUOUS_KV_LAYOUT_VERSION;
  if (paged_layout) {
    plan.component_identities.permutation_version =
        Constants::LLM_KV_BLOCK_PERMUTATION_VERSION;
  } else {
    plan.component_identities.permutation_version.reset();
  }
  plan.component_identities.backend_executor_version =
      paged_layout ? Constants::LLM_PAGED_CPU_EXECUTOR_VERSION
                   : Constants::LLM_CPU_EXECUTOR_VERSION;
  plan.component_identities.resource_abi_version =
      paged_layout ? Constants::LLM_PAGED_DESCRIPTOR_ABI_VERSION
                   : Constants::LLM_DESCRIPTOR_ABI_VERSION;
  plan.component_identities.schedule_version =
      paged_layout ? Constants::LLM_PAGED_CPU_SCHEDULE_VERSION
                   : Constants::LLM_CPU_SCHEDULE_VERSION;
  plan.component_identities.timer_policy_version =
      Constants::LLM_CPU_TIMER_POLICY_VERSION;
  plan.component_identities.buffer_pattern_version =
      paged_layout ? Constants::LLM_PAGED_BUFFER_PATTERN_VERSION
                   : Constants::LLM_BUFFER_PATTERN_VERSION;
  plan.component_identities.write_pattern_version =
      Constants::LLM_APPEND_PATTERN_VERSION;
  plan.component_identities.checksum_pattern_version =
      paged_layout ? Constants::LLM_PAGED_READ_CHECKSUM_VERSION
                   : Constants::LLM_READ_CHECKSUM_VERSION;
  plan.component_identities.msl_revision.reset();
  plan.component_identities.msl_source_sha256.reset();
  plan.component_identities.identity =
      serialize_llm_component_identities(plan.component_identities);
  plan.plan_identity = build_model_plan_identity(plan);
  return !plan.methodology_version.empty() &&
         !plan.component_identities.identity.empty() &&
         !plan.plan_identity.empty();
}

bool prepare_paged_identity_shape(LlmMemoryWorkPlan& plan) {
  LlmCpuExecutionPlan* const cpu_plan = get_llm_cpu_execution_plan(plan);
  if (cpu_plan == nullptr || !cpu_plan->paged.has_value()) {
    return false;
  }
  LlmPagedCpuExecutionPlan& paged = *cpu_plan->paged;
  paged.block_table_logical_bytes = paged.layout.memory.block_table_bytes;
  paged.block_table_mapping_bytes =
      plan.memory_budget.request.committed_block_table_mapping_bytes;
  paged.block_table_read_only = false;
  paged.table_validation.valid = true;
  paged.table_validation.interrupted = false;
  paged.table_validation.reason_code = LlmKvLayoutReason::VALID;
  paged.table_validation.expected_entries =
      paged.layout.block_table_entries;
  paged.table_validation.examined_entries =
      paged.layout.block_table_entries;
  paged.table_validation.validation_bitset_bytes =
      paged.layout.memory.validation_bitset_bytes;
  paged.permutation = build_llm_kv_permutation_identity(
      paged.layout, derive_llm_kv_permutation_seed(plan.base_seed),
      std::string(64, '0'));
  paged.layout_identity =
      serialize_llm_kv_layout_identity(paged.layout, paged.permutation);
  paged.execution_identity = paged.ownership.identity;
  return !paged.permutation.identity.empty() &&
         !paged.layout_identity.empty() &&
         !paged.execution_identity.empty();
}

bool add_identity_bytes(size_t bytes, size_t& total) {
  return NumericUtils::checked_add(total, bytes, total);
}

bool build_auxiliary_preflight_view(
    LlmMemoryWorkPlan& plan, LlmAuxiliaryPreflightView& view) {
  view = {};
  LlmCpuExecutionPlan* const cpu_plan = get_llm_cpu_execution_plan(plan);
  if (cpu_plan == nullptr || !plan.memory_budget.valid ||
      plan.plan_identity.empty() || cpu_plan->effective_workers == 0) {
    return false;
  }

  const bool original_valid = plan.valid;
  plan.valid = true;
  view.backend = plan.backend;
  view.kv_layout = plan.kv_layout;
  view.effective_workers = cpu_plan->effective_workers;
  view.total_layer_descriptors = cpu_plan->total_layer_descriptors;
  view.total_sequence_descriptors = cpu_plan->total_sequence_descriptors;
  view.k_or_v_static_reference_count =
      cpu_plan->paged.has_value()
          ? cpu_plan->paged->layout.total_physical_blocks
          : cpu_plan->total_sequence_descriptors;
  view.model_plan_identity_bytes = plan.plan_identity.size();

  constexpr std::array<LlmScenario, kLlmScenarioCount> kScenarios = {
      LlmScenario::WeightsOnly, LlmScenario::KvOnly,
      LlmScenario::Mixed};
  std::array<size_t, kLlmScenarioCount> maximum_work_units{};
  for (size_t index = 0; index < kLlmScenarioCount; ++index) {
    const LlmScenarioLimits limits =
        calculate_llm_scenario_limits(plan.geometry, kScenarios[index]);
    if (!limits.valid) {
      plan.valid = original_valid;
      return false;
    }
    maximum_work_units[index] = limits.effective_maximum_work_units;
    const LlmScenarioWorkPlan scenario = build_llm_scenario_work_plan(
        plan, kScenarios[index], maximum_work_units[index], false);
    if (!scenario.valid) {
      plan.valid = original_valid;
      return false;
    }
    view.maximum_scenario_plan_identity_bytes[index] =
        scenario.plan_identity.size();
  }
  const LlmFrozenScenarioPlans frozen =
      freeze_llm_scenario_work_plans(plan, maximum_work_units, false);
  if (!frozen.valid) {
    plan.valid = original_valid;
    return false;
  }
  view.frozen_reason_code_bytes = frozen.reason_code.size();
  view.frozen_model_plan_identity_bytes =
      frozen.model_plan_identity.size();
  view.frozen_plan_identity_bytes = frozen.plan_identity.size();
  for (size_t index = 0; index < kLlmScenarioCount; ++index) {
    view.frozen_scenario_reason_code_bytes[index] =
        frozen.scenarios[index].reason_code.size();
    view.frozen_scenario_model_plan_identity_bytes[index] =
        frozen.scenarios[index].model_plan_identity.size();
    view.frozen_scenario_plan_identity_bytes[index] =
        frozen.scenarios[index].plan_identity.size();
  }

  size_t json_identity_bytes = 0;
  const LlmComponentIdentities& components = plan.component_identities;
  const auto add_string = [&](const std::string& value) {
    return add_identity_bytes(value.size(), json_identity_bytes);
  };
  const auto add_optional = [&](const std::optional<std::string>& value) {
    return !value.has_value() || add_string(*value);
  };
  bool json_valid =
      add_string(plan.plan_identity) &&
      add_string(plan.methodology_version) &&
      add_string(components.logical_profile_version) &&
      add_string(components.kv_layout_version) &&
      add_optional(components.permutation_version) &&
      add_string(components.backend_executor_version) &&
      add_string(components.resource_abi_version) &&
      add_string(components.schedule_version) &&
      add_string(components.timer_policy_version) &&
      add_string(components.buffer_pattern_version) &&
      add_string(components.write_pattern_version) &&
      add_string(components.checksum_pattern_version) &&
      add_optional(components.msl_revision) &&
      add_optional(components.msl_source_sha256) &&
      add_string(components.identity);
  if (json_valid && cpu_plan->paged.has_value()) {
    const LlmPagedCpuExecutionPlan& paged = *cpu_plan->paged;
    json_valid =
        add_string(paged.layout.geometry_identity) &&
        add_string(paged.layout_identity) &&
        add_string(paged.execution_identity) &&
        add_string(paged.table_validation.reason_code) &&
        add_string(paged.permutation.algorithm_version) &&
        add_string(paged.permutation.domain_uint64_hex) &&
        add_string(paged.permutation.sha256) &&
        add_string(paged.permutation.identity) &&
        add_string(paged.ownership.reason_code) &&
        add_string(paged.ownership.layout_geometry_identity) &&
        add_string(paged.ownership.identity);
  }
  plan.valid = original_valid;
  if (!json_valid) {
    return false;
  }
  view.json_identity_string_bytes = json_identity_bytes;
  view.valid = true;
  return true;
}

bool auxiliary_preflight_views_match(
    const LlmAuxiliaryPreflightView& lhs,
    const LlmAuxiliaryPreflightView& rhs) {
  return lhs.valid && rhs.valid && lhs.backend == rhs.backend &&
         lhs.kv_layout == rhs.kv_layout &&
         lhs.effective_workers == rhs.effective_workers &&
         lhs.total_layer_descriptors == rhs.total_layer_descriptors &&
         lhs.total_sequence_descriptors == rhs.total_sequence_descriptors &&
         lhs.k_or_v_static_reference_count ==
             rhs.k_or_v_static_reference_count &&
         lhs.model_plan_identity_bytes == rhs.model_plan_identity_bytes &&
         lhs.maximum_scenario_plan_identity_bytes ==
             rhs.maximum_scenario_plan_identity_bytes &&
         lhs.frozen_reason_code_bytes == rhs.frozen_reason_code_bytes &&
         lhs.frozen_model_plan_identity_bytes ==
             rhs.frozen_model_plan_identity_bytes &&
         lhs.frozen_plan_identity_bytes == rhs.frozen_plan_identity_bytes &&
         lhs.frozen_scenario_reason_code_bytes ==
             rhs.frozen_scenario_reason_code_bytes &&
         lhs.frozen_scenario_model_plan_identity_bytes ==
             rhs.frozen_scenario_model_plan_identity_bytes &&
         lhs.frozen_scenario_plan_identity_bytes ==
             rhs.frozen_scenario_plan_identity_bytes &&
         lhs.json_identity_string_bytes == rhs.json_identity_string_bytes;
}

void discard_executable_templates(LlmMemoryWorkPlan& plan) {
  std::vector<LlmByteRange>().swap(plan.weight_layers);
  plan.methodology_version.clear();
  plan.component_identities = {};
  plan.plan_identity.clear();
  LlmCpuExecutionPlan* const cpu_plan = get_llm_cpu_execution_plan(plan);
  if (cpu_plan != nullptr) {
    std::vector<LlmWorkerWorkPlan>().swap(cpu_plan->workers);
    cpu_plan->paged.reset();
    cpu_plan->effective_workers = 0;
  }
}

LlmMemoryWorkPlan invalid_config_plan(const std::string& reason_code) {
  LlmMemoryWorkPlan plan;
  plan.reason_code = reason_code;
  return plan;
}

}  // namespace

const LlmCpuExecutionPlan* get_llm_cpu_execution_plan(
    const LlmMemoryWorkPlan& plan) noexcept {
  if (plan.backend != LlmMemoryBackend::Cpu) {
    return nullptr;
  }
  return std::get_if<LlmCpuExecutionPlan>(&plan.backend_execution_plan);
}

LlmCpuExecutionPlan* get_llm_cpu_execution_plan(
    LlmMemoryWorkPlan& plan) noexcept {
  if (plan.backend != LlmMemoryBackend::Cpu) {
    return nullptr;
  }
  return std::get_if<LlmCpuExecutionPlan>(&plan.backend_execution_plan);
}

std::string build_llm_methodology_version(LlmMemoryBackend backend,
                                          LlmPhase phase,
                                          LlmKvLayout layout) {
  std::string methodology = "llm-memory-v1-";
  methodology += llm_memory_backend_to_string(backend);
  methodology += '-';
  methodology += llm_phase_to_string(phase);
  methodology += '-';
  methodology += llm_kv_layout_to_string(layout);
  return methodology;
}

std::string serialize_llm_component_identities(
    const LlmComponentIdentities& components) {
  std::string identity = Constants::LLM_COMPONENT_IDENTITY_VERSION;
  append_component_identity(identity, "logical_profile_version",
                            components.logical_profile_version);
  append_component_identity(identity, "kv_layout_version",
                            components.kv_layout_version);
  append_component_identity(identity, "permutation_version",
                            components.permutation_version);
  append_component_identity(identity, "backend_executor_version",
                            components.backend_executor_version);
  append_component_identity(identity, "resource_abi_version",
                            components.resource_abi_version);
  append_component_identity(identity, "schedule_version",
                            components.schedule_version);
  append_component_identity(identity, "timer_policy_version",
                            components.timer_policy_version);
  append_component_identity(identity, "buffer_pattern_version",
                            components.buffer_pattern_version);
  append_component_identity(identity, "write_pattern_version",
                            components.write_pattern_version);
  append_component_identity(identity, "checksum_pattern_version",
                            components.checksum_pattern_version);
  append_component_identity(identity, "msl_revision",
                            components.msl_revision);
  append_component_identity(identity, "msl_source_sha256",
                            components.msl_source_sha256);
  return identity;
}

LlmMemoryWorkPlan::LlmMemoryWorkPlan(LlmMemoryWorkPlan&& other) noexcept {
  *this = std::move(other);
}

LlmMemoryWorkPlan& LlmMemoryWorkPlan::operator=(
    LlmMemoryWorkPlan&& other) noexcept {
  if (this == &other) {
    return *this;
  }

  valid = other.valid;
  reason_code = std::move(other.reason_code);
  geometry = std::move(other.geometry);
  backend = other.backend;
  phase = other.phase;
  kv_layout = other.kv_layout;
  work_unit_kind = other.work_unit_kind;
  weight_passes_per_work_unit = other.weight_passes_per_work_unit;
  kv_replay_factor = other.kv_replay_factor;
  base_seed = other.base_seed;
  weight_buffer_seed = other.weight_buffer_seed;
  k_buffer_seed = other.k_buffer_seed;
  v_buffer_seed = other.v_buffer_seed;
  scenario_seeds = other.scenario_seeds;
  memory_budget = std::move(other.memory_budget);
  weight_layers = std::move(other.weight_layers);
  methodology_version = std::move(other.methodology_version);
  component_identities = std::move(other.component_identities);
  plan_identity = std::move(other.plan_identity);
  backend_execution_plan = std::move(other.backend_execution_plan);

  other.valid = false;
  other.reason_code.clear();
  other.geometry.valid = false;
  other.base_seed = 0;
  other.weight_buffer_seed = 0;
  other.k_buffer_seed = 0;
  other.v_buffer_seed = 0;
  other.scenario_seeds = {};
  other.memory_budget.valid = false;
  other.memory_budget.request.valid = false;
  other.weight_layers.clear();
  other.backend = LlmMemoryBackend::Cpu;
  other.phase = LlmPhase::Decode;
  other.kv_layout = LlmKvLayout::Contiguous;
  other.work_unit_kind = LlmWorkUnitKind::DecodeStep;
  other.methodology_version.clear();
  other.component_identities = {};
  other.plan_identity.clear();
  other.backend_execution_plan = LlmCpuExecutionPlan{};
  return *this;
}

uint64_t llm_seed_domain_value(LlmSeedDomain domain) {
  switch (domain) {
    case LlmSeedDomain::WeightBuffer:
      return kWeightBufferSeedDomain;
    case LlmSeedDomain::KBuffer:
      return kKBufferSeedDomain;
    case LlmSeedDomain::VBuffer:
      return kVBufferSeedDomain;
    case LlmSeedDomain::WeightsOnlyScenario:
      return kWeightsOnlyScenarioSeedDomain;
    case LlmSeedDomain::KvOnlyScenario:
      return kKvOnlyScenarioSeedDomain;
    case LlmSeedDomain::MixedScenario:
      return kMixedScenarioSeedDomain;
  }
  return 0;
}

uint64_t derive_llm_domain_seed(uint64_t base_seed, LlmSeedDomain domain) {
  const uint64_t domain_value = llm_seed_domain_value(domain);
  return domain_value == 0 ? 0
                           : SeedUtils::splitmix64(base_seed ^ domain_value);
}

LlmGeometry resolve_llm_geometry(const LlmGeometryRequest& request) {
  LlmGeometry geometry;
  geometry.phase = request.phase;
  geometry.kv_layout = request.kv_layout;
  geometry.work_unit_kind = llm_work_unit_kind_for_phase(request.phase);
  geometry.active_weight_bytes_per_work_unit = request.active_weight_bytes;
  geometry.layer_count = request.layer_count;
  geometry.query_head_count = request.query_head_count;
  geometry.kv_head_count = request.kv_head_count;
  geometry.head_dimension = request.head_dimension;
  geometry.kv_element_bytes = request.kv_element_bytes;
  geometry.batch_size = request.batch_size;
  geometry.kv_block_tokens = request.kv_block_tokens;

  if (request.phase != LlmPhase::Decode) {
    geometry.reason_code = LlmWorkPlanReason::PHASE_NOT_ACTIVATED;
    return geometry;
  }
  if (request.active_weight_bytes == 0) {
    geometry.reason_code = LlmWorkPlanReason::ACTIVE_WEIGHT_BYTES_ZERO;
    return geometry;
  }
  if (request.layer_count == 0) {
    geometry.reason_code = LlmWorkPlanReason::LAYER_COUNT_ZERO;
    return geometry;
  }
  if (request.query_head_count == 0) {
    geometry.reason_code = LlmWorkPlanReason::QUERY_HEAD_COUNT_ZERO;
    return geometry;
  }
  if (request.kv_head_count == 0) {
    geometry.reason_code = LlmWorkPlanReason::KV_HEAD_COUNT_ZERO;
    return geometry;
  }
  if (request.head_dimension == 0) {
    geometry.reason_code = LlmWorkPlanReason::HEAD_DIMENSION_ZERO;
    return geometry;
  }
  if (!valid_kv_element_bytes(request.kv_element_bytes)) {
    geometry.reason_code = LlmWorkPlanReason::INVALID_KV_ELEMENT_BYTES;
    return geometry;
  }
  if (request.visible_context_tokens == 0) {
    geometry.reason_code = LlmWorkPlanReason::CONTEXT_TOKENS_ZERO;
    return geometry;
  }
  if (request.kv_layout == LlmKvLayout::Contiguous &&
      request.kv_block_tokens != 0) {
    geometry.reason_code =
        LlmWorkPlanReason::KV_BLOCK_TOKENS_NOT_APPLICABLE;
    return geometry;
  }
  if (request.batch_size == 0) {
    geometry.reason_code = LlmWorkPlanReason::BATCH_SIZE_ZERO;
    return geometry;
  }
  if (request.query_head_count < request.kv_head_count) {
    geometry.reason_code = LlmWorkPlanReason::QUERY_HEADS_BELOW_KV_HEADS;
    return geometry;
  }
  if (request.query_head_count % request.kv_head_count != 0) {
    geometry.reason_code =
        LlmWorkPlanReason::QUERY_HEADS_NOT_DIVISIBLE_BY_KV_HEADS;
    return geometry;
  }

  geometry.query_heads_per_kv_head =
      request.query_head_count / request.kv_head_count;
  geometry.attention_kind =
      classify_attention(request.query_head_count, request.kv_head_count);
  if (!NumericUtils::checked_multiply(
          request.head_dimension, request.kv_element_bytes,
          geometry.kv_vector_bytes)) {
    geometry.reason_code = LlmWorkPlanReason::KV_VECTOR_BYTES_OVERFLOW;
    return geometry;
  }
  if (!NumericUtils::checked_multiply(
          request.kv_head_count, geometry.kv_vector_bytes,
          geometry.k_or_v_record_bytes_per_layer)) {
    geometry.reason_code = LlmWorkPlanReason::KV_RECORD_BYTES_OVERFLOW;
    return geometry;
  }
  if (!NumericUtils::checked_multiply(
          geometry.k_or_v_record_bytes_per_layer, 2,
          geometry.kv_record_bytes_per_layer)) {
    geometry.reason_code =
        LlmWorkPlanReason::KV_LAYER_RECORD_BYTES_OVERFLOW;
    return geometry;
  }
  if (!NumericUtils::checked_multiply(
          request.layer_count, geometry.kv_record_bytes_per_layer,
          geometry.kv_bytes_per_visible_token)) {
    geometry.reason_code = LlmWorkPlanReason::KV_BYTES_PER_TOKEN_OVERFLOW;
    return geometry;
  }
  if (!NumericUtils::checked_multiply(
          request.visible_context_tokens,
          geometry.k_or_v_record_bytes_per_layer,
          geometry.k_or_v_sequence_visible_bytes)) {
    geometry.reason_code = LlmWorkPlanReason::KV_SEQUENCE_BYTES_OVERFLOW;
    return geometry;
  }

  size_t layer_batch_count = 0;
  if (!NumericUtils::checked_multiply(request.layer_count, request.batch_size,
                                      layer_batch_count) ||
      !NumericUtils::checked_multiply(
          layer_batch_count, geometry.k_or_v_sequence_visible_bytes,
          geometry.k_logical_bytes)) {
    geometry.reason_code = LlmWorkPlanReason::KV_MAPPING_BYTES_OVERFLOW;
    return geometry;
  }
  geometry.v_logical_bytes = geometry.k_logical_bytes;
  if (request.kv_layout == LlmKvLayout::Paged) {
    const LlmKvLayoutPlan paged_layout = build_llm_kv_layout_plan(
        {request.visible_context_tokens, request.kv_block_tokens,
         request.layer_count, request.batch_size,
         geometry.k_or_v_record_bytes_per_layer});
    if (!paged_layout.valid) {
      geometry.reason_code = paged_layout.reason_code;
      return geometry;
    }
    geometry.kv_blocks_per_sequence = paged_layout.blocks_per_sequence;
    geometry.physical_blocks_per_layer =
        paged_layout.physical_blocks_per_layer;
    geometry.total_physical_blocks = paged_layout.total_physical_blocks;
    geometry.kv_block_bytes = paged_layout.block_bytes;
    geometry.last_block_tokens = paged_layout.last_block_tokens;
    geometry.last_block_valid_bytes = paged_layout.last_block_valid_bytes;
    geometry.decode_append_offset_in_last_block =
        paged_layout.decode_append_offset_in_last_block;
    geometry.k_layout_padding_bytes =
        paged_layout.memory.k_layout_padding_bytes;
    geometry.v_layout_padding_bytes =
        paged_layout.memory.v_layout_padding_bytes;
    geometry.block_table_entries = paged_layout.block_table_entries;
    geometry.block_table_bytes = paged_layout.memory.block_table_bytes;
    geometry.k_mapping_bytes = paged_layout.memory.k_physical_bytes;
    geometry.v_mapping_bytes = paged_layout.memory.v_physical_bytes;
  } else if (request.kv_layout == LlmKvLayout::Contiguous) {
    geometry.k_mapping_bytes = geometry.k_logical_bytes;
    geometry.v_mapping_bytes = geometry.v_logical_bytes;
  } else {
    geometry.reason_code = LlmWorkPlanReason::KV_LAYOUT_NOT_ACTIVATED;
    return geometry;
  }
  if (!NumericUtils::checked_add(geometry.k_mapping_bytes,
                                 geometry.v_mapping_bytes,
                                 geometry.kv_capacity_bytes)) {
    geometry.reason_code = LlmWorkPlanReason::KV_CAPACITY_BYTES_OVERFLOW;
    return geometry;
  }

  geometry.weight_read_bytes_per_work_unit = request.active_weight_bytes;
  if (!NumericUtils::checked_multiply(
          request.batch_size, geometry.kv_bytes_per_visible_token,
          geometry.kv_write_bytes_per_work_unit)) {
    geometry.reason_code = LlmWorkPlanReason::KV_WRITE_BYTES_OVERFLOW;
    return geometry;
  }
  if (!NumericUtils::checked_multiply(
          request.visible_context_tokens,
          geometry.kv_write_bytes_per_work_unit,
          geometry.kv_read_bytes_per_work_unit)) {
    geometry.reason_code = LlmWorkPlanReason::KV_READ_BYTES_OVERFLOW;
    return geometry;
  }
  if (!NumericUtils::checked_add(request.active_weight_bytes,
                                 geometry.kv_capacity_bytes,
                                 geometry.total_data_mapping_bytes)) {
    geometry.reason_code = LlmWorkPlanReason::TOTAL_DATA_BYTES_OVERFLOW;
    return geometry;
  }
  if (!NumericUtils::checked_add(
          geometry.kv_read_bytes_per_work_unit,
          geometry.kv_write_bytes_per_work_unit,
          geometry.kv_only_effective_model_payload_bytes_per_work_unit)) {
    geometry.reason_code = LlmWorkPlanReason::KV_ONLY_PAYLOAD_OVERFLOW;
    return geometry;
  }
  if (!NumericUtils::checked_add(
          geometry.weight_read_bytes_per_work_unit,
          geometry.kv_only_effective_model_payload_bytes_per_work_unit,
          geometry.mixed_effective_model_payload_bytes_per_work_unit)) {
    geometry.reason_code = LlmWorkPlanReason::MIXED_PAYLOAD_OVERFLOW;
    return geometry;
  }
  geometry.traffic_crossover_numerator = request.active_weight_bytes;
  geometry.traffic_crossover_denominator =
      geometry.kv_write_bytes_per_work_unit;
  geometry.traffic_crossover_context_tokens = static_cast<double>(
      static_cast<long double>(geometry.traffic_crossover_numerator) /
      static_cast<long double>(geometry.traffic_crossover_denominator));
  geometry.decode = LlmDecodeGeometry{request.visible_context_tokens};
  geometry.prefill.reset();
  geometry.valid = true;
  geometry.reason_code = LlmWorkPlanReason::VALID;
  return geometry;
}

LlmMemoryBudgetRequest build_llm_memory_budget_request(
    const LlmGeometry& geometry, size_t descriptor_bytes,
    size_t planner_storage_bytes,
    size_t checksum_auxiliary_bytes, size_t orchestration_auxiliary_bytes,
    size_t mapping_granularity_bytes, size_t block_table_mapping_bytes,
    size_t layout_transient_bytes) {
  LlmMemoryBudgetRequest request;
  request.mapping_granularity_bytes = mapping_granularity_bytes;
  request.descriptor_bytes = descriptor_bytes;
  request.planner_storage_bytes = planner_storage_bytes;
  request.checksum_auxiliary_bytes = checksum_auxiliary_bytes;
  request.orchestration_auxiliary_bytes = orchestration_auxiliary_bytes;
  request.requested_block_table_mapping_bytes = block_table_mapping_bytes;
  request.layout_transient_bytes = layout_transient_bytes;
  if (!geometry.valid) {
    request.reason_code = geometry.reason_code;
    return request;
  }
  if (mapping_granularity_bytes == 0) {
    request.reason_code = LlmWorkPlanReason::MAPPING_GRANULARITY_ZERO;
    return request;
  }

  request.requested_weight_mapping_bytes =
      geometry.active_weight_bytes_per_work_unit;
  request.requested_k_mapping_bytes = geometry.k_mapping_bytes;
  request.requested_v_mapping_bytes = geometry.v_mapping_bytes;
  request.requested_data_bytes = geometry.total_data_mapping_bytes;
  if (!NumericUtils::checked_round_up(
          request.requested_weight_mapping_bytes, mapping_granularity_bytes,
          request.committed_weight_mapping_bytes) ||
      !NumericUtils::checked_round_up(
          request.requested_k_mapping_bytes, mapping_granularity_bytes,
          request.committed_k_mapping_bytes) ||
      !NumericUtils::checked_round_up(
          request.requested_v_mapping_bytes, mapping_granularity_bytes,
          request.committed_v_mapping_bytes) ||
      (block_table_mapping_bytes != 0 &&
       !NumericUtils::checked_round_up(
           block_table_mapping_bytes, mapping_granularity_bytes,
           request.committed_block_table_mapping_bytes))) {
    request.reason_code = LlmWorkPlanReason::MAPPING_ROUND_UP_OVERFLOW;
    return request;
  }

  size_t committed_kv_bytes = 0;
  if (!NumericUtils::checked_add(request.committed_k_mapping_bytes,
                                 request.committed_v_mapping_bytes,
                                 committed_kv_bytes) ||
      !NumericUtils::checked_add(request.committed_weight_mapping_bytes,
                                 committed_kv_bytes,
                                 request.committed_data_bytes)) {
    request.reason_code = LlmWorkPlanReason::MEMORY_REQUIREMENT_OVERFLOW;
    return request;
  }
  size_t descriptor_and_planner = 0;
  size_t checksum_and_orchestration = 0;
  if (!NumericUtils::checked_add(descriptor_bytes, planner_storage_bytes,
                                 descriptor_and_planner) ||
      !NumericUtils::checked_add(checksum_auxiliary_bytes,
                                 orchestration_auxiliary_bytes,
                                 checksum_and_orchestration) ||
      !NumericUtils::checked_add(descriptor_and_planner,
                                 checksum_and_orchestration,
                                 request.auxiliary_bytes)) {
    request.reason_code = LlmWorkPlanReason::AUXILIARY_BYTES_OVERFLOW;
    return request;
  }
  size_t resident_runtime_bytes = 0;
  if (!NumericUtils::checked_add(
          request.committed_data_bytes,
          request.committed_block_table_mapping_bytes,
          resident_runtime_bytes) ||
      !NumericUtils::checked_add(resident_runtime_bytes,
                                 request.auxiliary_bytes,
                                 request.runtime_peak_bytes) ||
      !NumericUtils::checked_add(
          request.committed_block_table_mapping_bytes,
          request.layout_transient_bytes, request.setup_peak_bytes) ||
      !NumericUtils::checked_add(request.setup_peak_bytes,
                                 request.planner_storage_bytes,
                                 request.setup_peak_bytes)) {
    request.reason_code = LlmWorkPlanReason::MEMORY_REQUIREMENT_OVERFLOW;
    return request;
  }
  request.required_total_bytes =
      std::max(request.setup_peak_bytes, request.runtime_peak_bytes);

  request.valid = true;
  request.reason_code = LlmWorkPlanReason::VALID;
  return request;
}

LlmMemoryBudget evaluate_llm_memory_budget(
    const LlmMemoryBudgetRequest& request, size_t available_memory_bytes) {
  LlmMemoryBudget budget;
  budget.request = request;
  budget.available_memory_bytes = available_memory_bytes;
  if (!request.valid) {
    budget.reason_code = request.reason_code;
    return budget;
  }

  if (available_memory_bytes == 0) {
    budget.used_fallback = true;
    if (!NumericUtils::checked_multiply(
            static_cast<size_t>(Constants::FALLBACK_TOTAL_LIMIT_MB),
            Constants::BYTES_PER_MB, budget.allowed_memory_bytes)) {
      budget.reason_code = LlmWorkPlanReason::MEMORY_BUDGET_OVERFLOW;
      return budget;
    }
  } else {
    const long double scaled =
        static_cast<long double>(available_memory_bytes) *
        Constants::MEMORY_LIMIT_FACTOR;
    if (scaled >
        static_cast<long double>(std::numeric_limits<size_t>::max())) {
      budget.reason_code = LlmWorkPlanReason::MEMORY_BUDGET_OVERFLOW;
      return budget;
    }
    budget.allowed_memory_bytes = static_cast<size_t>(scaled);
  }

  if (request.required_total_bytes > budget.allowed_memory_bytes) {
    budget.reason_code = LlmWorkPlanReason::MEMORY_BUDGET_EXCEEDED;
    return budget;
  }
  budget.valid = true;
  budget.reason_code = LlmWorkPlanReason::WITHIN_MEMORY_BUDGET;
  return budget;
}

LlmMemoryWorkPlan build_llm_memory_work_plan_candidate(
    const LlmMemoryWorkPlanRequest& request,
    const LlmKvStopRequested& stop_requested) {
  LlmMemoryWorkPlan plan;
  plan.backend = request.backend;
  if (request.backend == LlmMemoryBackend::Metal) {
    plan.backend_execution_plan = LlmMetalExecutionPlan{};
  }
  plan.phase = request.geometry.phase;
  plan.kv_layout = request.geometry.kv_layout;
  plan.work_unit_kind = llm_work_unit_kind_for_phase(plan.phase);
  plan.base_seed = request.base_seed;
  plan.weight_buffer_seed =
      derive_llm_domain_seed(request.base_seed, LlmSeedDomain::WeightBuffer);
  plan.k_buffer_seed =
      derive_llm_domain_seed(request.base_seed, LlmSeedDomain::KBuffer);
  plan.v_buffer_seed =
      derive_llm_domain_seed(request.base_seed, LlmSeedDomain::VBuffer);
  plan.scenario_seeds = {
      derive_llm_domain_seed(request.base_seed,
                             LlmSeedDomain::WeightsOnlyScenario),
      derive_llm_domain_seed(request.base_seed,
                             LlmSeedDomain::KvOnlyScenario),
      derive_llm_domain_seed(request.base_seed,
                             LlmSeedDomain::MixedScenario)};
  if (request.backend != LlmMemoryBackend::Cpu) {
    plan.reason_code = LlmWorkPlanReason::BACKEND_NOT_ACTIVATED;
    return plan;
  }
  LlmCpuExecutionPlan* const cpu_plan = get_llm_cpu_execution_plan(plan);
  if (cpu_plan == nullptr) {
    plan.reason_code = LlmWorkPlanReason::BACKEND_NOT_ACTIVATED;
    return plan;
  }
  cpu_plan->requested_workers = request.requested_workers;
  cpu_plan->available_workers = request.available_workers;
  plan.geometry = resolve_llm_geometry(request.geometry);
  if (!plan.geometry.valid) {
    plan.reason_code = plan.geometry.reason_code;
    return plan;
  }
  if (!json_integer_is_safe(request.requested_workers) ||
      !json_integer_is_safe(request.available_workers)) {
    plan.reason_code = LlmWorkPlanReason::JSON_INTEGER_OUT_OF_RANGE;
    return plan;
  }
  if (request.requested_workers == 0) {
    plan.reason_code = LlmWorkPlanReason::REQUESTED_WORKERS_ZERO;
    return plan;
  }
  if (request.available_workers == 0) {
    plan.reason_code = LlmWorkPlanReason::AVAILABLE_WORKERS_ZERO;
    return plan;
  }

  const bool paged_layout = plan.kv_layout == LlmKvLayout::Paged;
  LlmKvLayoutPlan paged_geometry;
  size_t paged_layer_sequence_count = 0;
  size_t paged_worker_coverage_limit = 0;
  if (paged_layout) {
    paged_geometry = build_llm_kv_layout_plan(
        {request.geometry.visible_context_tokens,
         request.geometry.kv_block_tokens, request.geometry.layer_count,
         request.geometry.batch_size,
         plan.geometry.k_or_v_record_bytes_per_layer});
    if (!paged_geometry.valid) {
      plan.reason_code = paged_geometry.reason_code;
      return plan;
    }
    if (!NumericUtils::checked_multiply(
            paged_geometry.layer_count, paged_geometry.batch_size,
            paged_layer_sequence_count) ||
        !NumericUtils::checked_add(
            paged_layer_sequence_count,
            paged_geometry.blocks_per_sequence - 1,
            paged_worker_coverage_limit)) {
      plan.reason_code = LlmWorkPlanReason::LAYER_SEQUENCE_COUNT_OVERFLOW;
      return plan;
    }
  }

  const size_t weight_layer_base =
      request.geometry.active_weight_bytes / request.geometry.layer_count;
  const size_t weight_layer_remainder =
      request.geometry.active_weight_bytes % request.geometry.layer_count;
  const size_t maximum_weight_layer_bytes =
      weight_layer_base + (weight_layer_remainder != 0 ? 1 : 0);
  const size_t maximum_kv_workers =
      paged_layout
          ? std::min(paged_geometry.total_physical_blocks,
                     paged_worker_coverage_limit)
                   : plan.geometry.k_or_v_sequence_visible_bytes;
  const size_t maximum_shared_worker_count =
      std::min(maximum_weight_layer_bytes, maximum_kv_workers);
  cpu_plan->effective_workers =
      std::min({request.requested_workers, request.available_workers,
                maximum_shared_worker_count});
  if (cpu_plan->effective_workers == 0) {
    plan.reason_code = LlmWorkPlanReason::NO_EXECUTABLE_WORKER;
    return plan;
  }

  size_t ownership_assignment_count = 0;
  if (paged_layout) {
    const size_t active_workers = std::min(
        cpu_plan->effective_workers, paged_geometry.blocks_per_sequence);
    if (!NumericUtils::checked_multiply(
            paged_layer_sequence_count, active_workers,
            ownership_assignment_count)) {
      plan.reason_code = LlmWorkPlanReason::PLANNER_STORAGE_BYTES_OVERFLOW;
      return plan;
    }
  }

  cpu_plan->layer_descriptors_per_worker = plan.geometry.layer_count;
  if (!NumericUtils::checked_multiply(
          plan.geometry.layer_count, plan.geometry.batch_size,
          cpu_plan->sequence_descriptors_per_worker)) {
    plan.reason_code = LlmWorkPlanReason::LAYER_SEQUENCE_COUNT_OVERFLOW;
    return plan;
  }
  if (!NumericUtils::checked_multiply(
          cpu_plan->layer_descriptors_per_worker,
          cpu_plan->effective_workers, cpu_plan->total_layer_descriptors) ||
      !NumericUtils::checked_multiply(
          cpu_plan->sequence_descriptors_per_worker,
          cpu_plan->effective_workers,
          cpu_plan->total_sequence_descriptors)) {
    plan.reason_code = LlmWorkPlanReason::DESCRIPTOR_COUNT_OVERFLOW;
    return plan;
  }
  size_t layer_descriptor_bytes = 0;
  size_t sequence_descriptor_bytes = 0;
  if (!NumericUtils::checked_multiply(
          cpu_plan->total_layer_descriptors,
          paged_layout ? sizeof(LlmPagedLayerDescriptor)
                       : sizeof(LlmLayerDescriptor),
          layer_descriptor_bytes) ||
      !NumericUtils::checked_multiply(
          cpu_plan->total_sequence_descriptors,
          paged_layout ? sizeof(LlmPagedKvAssignmentDescriptor)
                       : sizeof(LlmKvSequenceDescriptor),
          sequence_descriptor_bytes) ||
      !NumericUtils::checked_add(layer_descriptor_bytes,
                                 sequence_descriptor_bytes,
                                 cpu_plan->descriptor_bytes)) {
    plan.reason_code = LlmWorkPlanReason::DESCRIPTOR_BYTES_OVERFLOW;
    return plan;
  }

  if (!calculate_planner_storage_bytes(
          plan.geometry.layer_count,
          cpu_plan->sequence_descriptors_per_worker,
          cpu_plan->effective_workers, paged_layout,
          ownership_assignment_count, cpu_plan->planner_storage_bytes)) {
    plan.reason_code = LlmWorkPlanReason::PLANNER_STORAGE_BYTES_OVERFLOW;
    return plan;
  }
  if (!json_integer_is_safe(request.geometry.layer_count) ||
      !json_integer_is_safe(request.geometry.query_head_count) ||
      !json_integer_is_safe(request.geometry.kv_head_count) ||
      !json_integer_is_safe(request.geometry.head_dimension) ||
      !json_integer_is_safe(request.geometry.visible_context_tokens) ||
      !json_integer_is_safe(request.geometry.kv_block_tokens) ||
      !json_integer_is_safe(request.geometry.batch_size) ||
      !json_integer_is_safe(cpu_plan->layer_descriptors_per_worker) ||
      !json_integer_is_safe(cpu_plan->sequence_descriptors_per_worker) ||
      !json_integer_is_safe(cpu_plan->total_layer_descriptors) ||
      !json_integer_is_safe(cpu_plan->total_sequence_descriptors)) {
    plan.reason_code = LlmWorkPlanReason::JSON_INTEGER_OUT_OF_RANGE;
    return plan;
  }

  plan.memory_budget.request = build_llm_memory_budget_request(
      plan.geometry, cpu_plan->descriptor_bytes,
      cpu_plan->planner_storage_bytes, request.checksum_auxiliary_bytes,
      request.orchestration_auxiliary_bytes,
      request.mapping_granularity_bytes,
      paged_layout ? paged_geometry.memory.block_table_bytes : 0,
      paged_layout
          ? paged_geometry.memory.validation_bitset_bytes
          : 0);
  plan.memory_budget = evaluate_llm_memory_budget(
      plan.memory_budget.request, request.available_memory_bytes);
  if (!plan.memory_budget.valid) {
    plan.reason_code = plan.memory_budget.reason_code;
    return plan;
  }

  if (paged_layout) {
    LlmKvCpuOwnershipPlan ownership =
        build_llm_paged_decode_kv_cpu_ownership_plan(
            paged_geometry, cpu_plan->effective_workers, stop_requested);
    if (!ownership.valid) {
      plan.reason_code = ownership.reason_code;
      return plan;
    }
    cpu_plan->paged.emplace();
    cpu_plan->paged->layout = std::move(paged_geometry);
    cpu_plan->paged->ownership = std::move(ownership);
  }

  try {
    plan.weight_layers.reserve(plan.geometry.layer_count);
    size_t weight_offset = 0;
    for (size_t layer = 0; layer < plan.geometry.layer_count; ++layer) {
      const size_t layer_bytes =
          weight_layer_base + (layer < weight_layer_remainder ? 1 : 0);
      plan.weight_layers.push_back({weight_offset, layer_bytes});
      weight_offset += layer_bytes;
    }

    cpu_plan->workers.resize(cpu_plan->effective_workers);
    for (size_t worker = 0; worker < cpu_plan->effective_workers; ++worker) {
      cpu_plan->workers[worker].worker_index = worker;
      cpu_plan->workers[worker].layers.reserve(plan.geometry.layer_count);
      if (paged_layout) {
        cpu_plan->workers[worker].paged_assignments.resize(
            cpu_plan->sequence_descriptors_per_worker);
        for (size_t layer = 0; layer < plan.geometry.layer_count; ++layer) {
          for (size_t batch = 0; batch < plan.geometry.batch_size; ++batch) {
            LlmPagedKvAssignmentTemplate& assignment =
                cpu_plan->workers[worker]
                    .paged_assignments[layer * plan.geometry.batch_size +
                                       batch];
            assignment.layer_index = layer;
            assignment.batch_sequence_index = batch;
          }
        }
      } else {
        cpu_plan->workers[worker].sequences.reserve(
            cpu_plan->sequence_descriptors_per_worker);
      }
    }

    for (size_t layer = 0; layer < plan.geometry.layer_count; ++layer) {
      const std::vector<LlmByteRange> weight_ranges = partition_range(
          plan.weight_layers[layer].offset_bytes,
          plan.weight_layers[layer].span_bytes, cpu_plan->effective_workers);
      const size_t first_sequence_index = layer * plan.geometry.batch_size;
      for (size_t worker = 0; worker < cpu_plan->effective_workers; ++worker) {
        const LlmByteRange weight =
            worker < weight_ranges.size() ? weight_ranges[worker]
                                          : LlmByteRange{};
        cpu_plan->workers[worker].layers.push_back(
            {weight, first_sequence_index, plan.geometry.batch_size, layer});
      }

      if (paged_layout) {
        continue;
      }
      for (size_t batch = 0; batch < plan.geometry.batch_size; ++batch) {
        const size_t sequence_index = first_sequence_index + batch;
        const size_t visible_offset =
            sequence_index * plan.geometry.k_or_v_sequence_visible_bytes;
        const LlmByteRange visible_record{
            visible_offset, plan.geometry.k_or_v_sequence_visible_bytes};
        const size_t append_offset =
            visible_offset +
            (plan.geometry.decode->visible_context_tokens - 1) *
                plan.geometry.k_or_v_record_bytes_per_layer;
        const LlmByteRange append_record{
            append_offset, plan.geometry.k_or_v_record_bytes_per_layer};
        const std::vector<LlmByteRange> visible_ranges = partition_range(
            visible_record.offset_bytes, visible_record.span_bytes,
            cpu_plan->effective_workers);
        for (size_t worker = 0; worker < cpu_plan->effective_workers;
             ++worker) {
          const LlmByteRange visible =
              worker < visible_ranges.size() ? visible_ranges[worker]
                                             : LlmByteRange{};
          const LlmByteRange append = intersect_ranges(visible, append_record);
          const size_t append_record_byte_offset =
              append.span_bytes == 0 ? 0 : append.offset_bytes - append_offset;
          cpu_plan->workers[worker].sequences.push_back(
              {visible, visible, append, append, layer, batch,
               append_record_byte_offset});
        }
      }
    }
    if (paged_layout) {
      for (const LlmKvCpuBlockAssignment& source :
           cpu_plan->paged->ownership.assignments) {
        if (source.worker_index >= cpu_plan->effective_workers ||
            source.layer_index >= plan.geometry.layer_count ||
            source.batch_sequence_index >= plan.geometry.batch_size) {
          throw std::length_error("invalid paged ownership");
        }
        const size_t assignment_index =
            source.layer_index * plan.geometry.batch_size +
            source.batch_sequence_index;
        LlmPagedKvAssignmentTemplate& destination =
            cpu_plan->workers[source.worker_index]
                .paged_assignments[assignment_index];
        if (destination.block_count != 0 || source.block_count == 0) {
          throw std::length_error("duplicate paged ownership");
        }
        destination.first_logical_block = source.first_logical_block;
        destination.block_count = source.block_count;
      }
    }
  } catch (const std::bad_alloc&) {
    discard_executable_templates(plan);
    plan.reason_code = LlmWorkPlanReason::PLANNER_ALLOCATION_FAILED;
    return plan;
  } catch (const std::length_error&) {
    discard_executable_templates(plan);
    plan.reason_code = LlmWorkPlanReason::PLANNER_ALLOCATION_FAILED;
    return plan;
  }

  size_t actual_planner_storage_bytes = 0;
  if (!calculate_actual_planner_storage_bytes(
          plan, actual_planner_storage_bytes)) {
    discard_executable_templates(plan);
    plan.reason_code = LlmWorkPlanReason::PLANNER_STORAGE_BYTES_OVERFLOW;
    return plan;
  }
  cpu_plan->planner_storage_bytes = actual_planner_storage_bytes;
  plan.memory_budget.request = build_llm_memory_budget_request(
      plan.geometry, cpu_plan->descriptor_bytes,
      cpu_plan->planner_storage_bytes, request.checksum_auxiliary_bytes,
      request.orchestration_auxiliary_bytes,
      request.mapping_granularity_bytes,
      paged_layout ? cpu_plan->paged->layout.memory.block_table_bytes : 0,
      paged_layout
          ? cpu_plan->paged->layout.memory.validation_bitset_bytes
          : 0);
  plan.memory_budget = evaluate_llm_memory_budget(
      plan.memory_budget.request, request.available_memory_bytes);
  if (!plan.memory_budget.valid) {
    discard_executable_templates(plan);
    plan.reason_code = plan.memory_budget.reason_code;
    return plan;
  }

  try {
    if ((paged_layout && !prepare_paged_identity_shape(plan)) ||
        !finalize_plan_identities(plan)) {
      discard_executable_templates(plan);
      plan.reason_code =
          LlmWorkPlanReason::AUXILIARY_PREFLIGHT_MISMATCH;
      return plan;
    }
  } catch (const std::bad_alloc&) {
    discard_executable_templates(plan);
    plan.reason_code = LlmWorkPlanReason::PLANNER_ALLOCATION_FAILED;
    return plan;
  } catch (const std::length_error&) {
    discard_executable_templates(plan);
    plan.reason_code = LlmWorkPlanReason::PLANNER_ALLOCATION_FAILED;
    return plan;
  }

  plan.valid = false;
  plan.reason_code = LlmWorkPlanReason::VALID;
  return plan;
}

LlmMemoryWorkPlanDraft prepare_llm_memory_work_plan(
    const LlmMemoryWorkPlanRequest& request,
    const LlmKvStopRequested& stop_requested) {
  LlmMemoryWorkPlanDraft draft;
  draft.candidate =
      build_llm_memory_work_plan_candidate(request, stop_requested);
  draft.reason_code = draft.candidate.reason_code;
  if (draft.candidate.reason_code != LlmWorkPlanReason::VALID) {
    return draft;
  }
  try {
    if (!build_auxiliary_preflight_view(
            draft.candidate, draft.auxiliary_preflight)) {
      discard_executable_templates(draft.candidate);
      draft.candidate.reason_code =
          LlmWorkPlanReason::AUXILIARY_PREFLIGHT_MISMATCH;
      draft.reason_code = draft.candidate.reason_code;
      return draft;
    }
  } catch (const std::bad_alloc&) {
    discard_executable_templates(draft.candidate);
    draft.candidate.reason_code =
        LlmWorkPlanReason::PLANNER_ALLOCATION_FAILED;
    draft.reason_code = draft.candidate.reason_code;
    return draft;
  } catch (const std::length_error&) {
    discard_executable_templates(draft.candidate);
    draft.candidate.reason_code =
        LlmWorkPlanReason::PLANNER_ALLOCATION_FAILED;
    draft.reason_code = draft.candidate.reason_code;
    return draft;
  }
  draft.valid = true;
  draft.reason_code = LlmWorkPlanReason::VALID;
  return draft;
}

LlmMemoryWorkPlan finalize_llm_memory_work_plan(
    LlmMemoryWorkPlanDraft&& draft, size_t checksum_auxiliary_bytes,
    size_t orchestration_auxiliary_bytes,
    const LlmKvStopRequested& stop_requested) {
  LlmMemoryWorkPlan plan = std::move(draft.candidate);
  if (!draft.valid || plan.reason_code != LlmWorkPlanReason::VALID ||
      plan.valid) {
    if (plan.reason_code == LlmWorkPlanReason::VALID) {
      plan.reason_code = draft.reason_code;
    }
    return plan;
  }
  LlmCpuExecutionPlan* const cpu_plan = get_llm_cpu_execution_plan(plan);
  if (cpu_plan == nullptr || cpu_plan->effective_workers == 0) {
    plan.reason_code = LlmWorkPlanReason::INVALID_MODEL_WORK_PLAN;
    return plan;
  }
  const bool paged_layout = plan.kv_layout == LlmKvLayout::Paged;
  const size_t block_table_bytes =
      paged_layout && cpu_plan->paged.has_value()
          ? cpu_plan->paged->layout.memory.block_table_bytes
          : 0;
  const size_t validation_bitset_bytes =
      paged_layout && cpu_plan->paged.has_value()
          ? cpu_plan->paged->layout.memory.validation_bitset_bytes
          : 0;
  plan.memory_budget.request = build_llm_memory_budget_request(
      plan.geometry, cpu_plan->descriptor_bytes,
      cpu_plan->planner_storage_bytes, checksum_auxiliary_bytes,
      orchestration_auxiliary_bytes,
      plan.memory_budget.request.mapping_granularity_bytes,
      block_table_bytes, validation_bitset_bytes);
  plan.memory_budget = evaluate_llm_memory_budget(
      plan.memory_budget.request, plan.memory_budget.available_memory_bytes);
  if (!plan.memory_budget.valid) {
    discard_executable_templates(plan);
    plan.reason_code = plan.memory_budget.reason_code;
    return plan;
  }

  try {
    if (paged_layout) {
      if (!cpu_plan->paged.has_value()) {
        plan.reason_code = LlmWorkPlanReason::INVALID_MODEL_WORK_PLAN;
        discard_executable_templates(plan);
        return plan;
      }
      LlmPagedCpuExecutionPlan& paged = *cpu_plan->paged;
      paged.block_table_logical_bytes =
          paged.layout.memory.block_table_bytes;
      paged.block_table_mapping_bytes =
          plan.memory_budget.request.committed_block_table_mapping_bytes;
      paged.block_table_mapping = allocate_buffer(
          paged.block_table_mapping_bytes, "LLM paged KV block table");
      if (paged.block_table_mapping == nullptr) {
        plan.reason_code = LlmWorkPlanReason::BLOCK_TABLE_MAPPING_FAILED;
        discard_executable_templates(plan);
        return plan;
      }
      const LlmKvInPlaceBlockTableMaterialization materialization =
          materialize_llm_kv_block_table_in_place(
              paged.layout, derive_llm_kv_permutation_seed(plan.base_seed),
              static_cast<uint32_t*>(paged.block_table_mapping.get()),
              paged.layout.block_table_entries,
              Constants::LLM_KV_BLOCK_TABLE_HASH_CHUNK_ENTRIES,
              stop_requested);
      if (!materialization.valid) {
        plan.reason_code = materialization.reason_code;
        discard_executable_templates(plan);
        return plan;
      }
      paged.table_validation = materialization.validation;
      paged.permutation = materialization.permutation;
      paged.layout_identity =
          serialize_llm_kv_layout_identity(paged.layout,
                                           paged.permutation);
      paged.execution_identity = paged.ownership.identity;
      if (paged.layout_identity.empty()) {
        plan.reason_code =
            LlmWorkPlanReason::BLOCK_TABLE_MATERIALIZATION_FAILED;
        discard_executable_templates(plan);
        return plan;
      }
      if (!protect_buffer_read_only(paged.block_table_mapping.get(),
                                    paged.block_table_mapping_bytes)) {
        plan.reason_code =
            LlmWorkPlanReason::BLOCK_TABLE_PROTECTION_FAILED;
        discard_executable_templates(plan);
        return plan;
      }
      paged.block_table_read_only = true;
    }

    if (!finalize_plan_identities(plan)) {
      plan.reason_code =
          LlmWorkPlanReason::AUXILIARY_PREFLIGHT_MISMATCH;
      discard_executable_templates(plan);
      return plan;
    }
    plan.valid = true;
    if (draft.auxiliary_preflight.valid) {
      LlmAuxiliaryPreflightView finalized_view;
      if (!build_auxiliary_preflight_view(plan, finalized_view) ||
          !auxiliary_preflight_views_match(
              draft.auxiliary_preflight, finalized_view)) {
        plan.valid = false;
        plan.reason_code =
            LlmWorkPlanReason::AUXILIARY_PREFLIGHT_MISMATCH;
        discard_executable_templates(plan);
        return plan;
      }
    }
  } catch (const std::bad_alloc&) {
    plan.valid = false;
    plan.reason_code = LlmWorkPlanReason::PLANNER_ALLOCATION_FAILED;
    discard_executable_templates(plan);
    return plan;
  } catch (const std::length_error&) {
    plan.valid = false;
    plan.reason_code = LlmWorkPlanReason::PLANNER_ALLOCATION_FAILED;
    discard_executable_templates(plan);
    return plan;
  }

  plan.reason_code = LlmWorkPlanReason::VALID;
  return plan;
}

LlmMemoryWorkPlan build_llm_memory_work_plan(
    const LlmMemoryWorkPlanRequest& request,
    const LlmKvStopRequested& stop_requested) {
  LlmMemoryWorkPlanDraft draft;
  draft.candidate =
      build_llm_memory_work_plan_candidate(request, stop_requested);
  draft.valid =
      draft.candidate.reason_code == LlmWorkPlanReason::VALID;
  draft.reason_code = draft.candidate.reason_code;
  return finalize_llm_memory_work_plan(
      std::move(draft), request.checksum_auxiliary_bytes,
      request.orchestration_auxiliary_bytes, stop_requested);
}

LlmMemoryWorkPlanDraft prepare_llm_memory_work_plan(
    const LlmMemoryConfig& config, size_t available_workers,
    size_t available_memory_bytes, size_t mapping_granularity_bytes,
    const LlmKvStopRequested& stop_requested) {
  const LlmMemoryConfigValidation validation =
      validate_llm_memory_config(config);
  if (!validation.valid) {
    LlmMemoryWorkPlanDraft draft;
    draft.reason_code = validation.reason_code;
    draft.candidate = invalid_config_plan(validation.reason_code);
    return draft;
  }

  LlmMemoryWorkPlanRequest request;
  request.geometry = {validation.active_weight_bytes,
                      config.layer_count,
                      config.query_head_count,
                      config.kv_head_count,
                      config.head_dimension,
                      config.kv_element_bytes,
                      config.visible_context_tokens,
                      config.batch_size,
                      config.kv_block_tokens,
                      config.phase,
                      config.kv_layout};
  request.backend = config.backend;
  request.requested_workers = config.requested_workers;
  request.available_workers = available_workers;
  request.available_memory_bytes = available_memory_bytes;
  request.mapping_granularity_bytes = mapping_granularity_bytes;
  request.base_seed = config.seed;
  return prepare_llm_memory_work_plan(request, stop_requested);
}

LlmMemoryWorkPlan build_llm_memory_work_plan(
    const LlmMemoryConfig& config, size_t available_workers,
    size_t available_memory_bytes, size_t mapping_granularity_bytes,
    size_t checksum_auxiliary_bytes,
    size_t orchestration_auxiliary_bytes,
    const LlmKvStopRequested& stop_requested) {
  const LlmMemoryConfigValidation validation =
      validate_llm_memory_config(config);
  if (!validation.valid) {
    return invalid_config_plan(validation.reason_code);
  }

  LlmMemoryWorkPlanRequest request;
  request.geometry = {validation.active_weight_bytes,
                      config.layer_count,
                      config.query_head_count,
                      config.kv_head_count,
                      config.head_dimension,
                      config.kv_element_bytes,
                      config.visible_context_tokens,
                      config.batch_size,
                      config.kv_block_tokens,
                      config.phase,
                      config.kv_layout};
  request.backend = config.backend;
  request.requested_workers = config.requested_workers;
  request.available_workers = available_workers;
  request.available_memory_bytes = available_memory_bytes;
  request.mapping_granularity_bytes = mapping_granularity_bytes;
  request.checksum_auxiliary_bytes = checksum_auxiliary_bytes;
  request.orchestration_auxiliary_bytes = orchestration_auxiliary_bytes;
  request.base_seed = config.seed;
  return build_llm_memory_work_plan(request, stop_requested);
}

bool readmit_llm_memory_work_plan(
    LlmMemoryWorkPlan& plan, size_t checksum_auxiliary_bytes,
    size_t orchestration_auxiliary_bytes) noexcept {
  try {
    LlmCpuExecutionPlan* const cpu_plan = get_llm_cpu_execution_plan(plan);
    if (!plan.valid || cpu_plan == nullptr) {
      return false;
    }
    const size_t block_table_bytes =
        cpu_plan->paged.has_value()
            ? cpu_plan->paged->layout.memory.block_table_bytes
            : 0;
    const size_t transient_bytes =
        cpu_plan->paged.has_value()
            ? cpu_plan->paged->layout.memory.validation_bitset_bytes
            : 0;
    const LlmMemoryBudgetRequest request = build_llm_memory_budget_request(
        plan.geometry, cpu_plan->descriptor_bytes,
        cpu_plan->planner_storage_bytes, checksum_auxiliary_bytes,
        orchestration_auxiliary_bytes,
        plan.memory_budget.request.mapping_granularity_bytes,
        block_table_bytes, transient_bytes);
    plan.memory_budget = evaluate_llm_memory_budget(
        request, plan.memory_budget.available_memory_bytes);
    if (!plan.memory_budget.valid) {
      plan.valid = false;
      plan.reason_code = plan.memory_budget.reason_code;
      return false;
    }
    plan.reason_code = LlmWorkPlanReason::VALID;
    return true;
  } catch (...) {
    plan.valid = false;
    plan.reason_code = LlmWorkPlanReason::MEMORY_REQUIREMENT_OVERFLOW;
    return false;
  }
}

LlmScenarioLimits calculate_llm_scenario_limits(
    const LlmGeometry& geometry, LlmScenario scenario) {
  LlmScenarioLimits limits;
  limits.scenario = scenario;
  if (!geometry.valid) {
    limits.reason_code = geometry.reason_code;
    return limits;
  }
  if (!valid_scenario(scenario)) {
    limits.reason_code = LlmWorkPlanReason::INVALID_SCENARIO;
    return limits;
  }
  limits.work_unit_kind = geometry.work_unit_kind;
  limits.kv_write_kind = llm_kv_write_kind_for(geometry.phase, scenario);

  switch (scenario) {
    case LlmScenario::WeightsOnly:
      limits.weight_read_bytes_per_work_unit =
          geometry.weight_read_bytes_per_work_unit;
      limits.effective_model_payload_bytes_per_work_unit =
          geometry.weight_read_bytes_per_work_unit;
      break;
    case LlmScenario::KvOnly:
      limits.kv_read_bytes_per_work_unit = geometry.kv_read_bytes_per_work_unit;
      limits.kv_write_bytes_per_work_unit =
          geometry.kv_write_bytes_per_work_unit;
      limits.effective_model_payload_bytes_per_work_unit =
          geometry.kv_only_effective_model_payload_bytes_per_work_unit;
      break;
    case LlmScenario::Mixed:
      limits.weight_read_bytes_per_work_unit =
          geometry.weight_read_bytes_per_work_unit;
      limits.kv_read_bytes_per_work_unit = geometry.kv_read_bytes_per_work_unit;
      limits.kv_write_bytes_per_work_unit =
          geometry.kv_write_bytes_per_work_unit;
      limits.effective_model_payload_bytes_per_work_unit =
          geometry.mixed_effective_model_payload_bytes_per_work_unit;
      break;
  }

  limits.layout_metadata_lookup_count_per_work_unit = 0;
  limits.layout_metadata_read_bytes_per_work_unit = 0;
  if (geometry.kv_layout == LlmKvLayout::Paged &&
      scenario != LlmScenario::WeightsOnly) {
    size_t lookups_per_layer_sequence = 0;
    size_t layer_sequence_count = 0;
    if (!NumericUtils::checked_multiply(
            geometry.kv_blocks_per_sequence, 2,
            lookups_per_layer_sequence) ||
        !NumericUtils::checked_add(lookups_per_layer_sequence, 1,
                                   lookups_per_layer_sequence) ||
        !NumericUtils::checked_multiply(
            geometry.layer_count, geometry.batch_size,
            layer_sequence_count) ||
        !NumericUtils::checked_multiply(
            layer_sequence_count, lookups_per_layer_sequence,
            limits.layout_metadata_lookup_count_per_work_unit) ||
        !NumericUtils::checked_multiply(
            limits.layout_metadata_lookup_count_per_work_unit,
            Constants::LLM_KV_BLOCK_TABLE_ENTRY_BYTES,
            limits.layout_metadata_read_bytes_per_work_unit)) {
      limits.reason_code = LlmWorkPlanReason::TASK_ACCOUNTED_BYTES_OVERFLOW;
      return limits;
    }
  }
  if (!NumericUtils::checked_add(
          limits.effective_model_payload_bytes_per_work_unit,
          limits.layout_metadata_read_bytes_per_work_unit,
          limits.accounted_bytes_per_work_unit)) {
    limits.reason_code = LlmWorkPlanReason::TASK_ACCOUNTED_BYTES_OVERFLOW;
    return limits;
  }
  limits.maximum_work_units_by_guardrail =
      Constants::LLM_MAX_ACCOUNTED_BYTES_PER_TASK /
      limits.accounted_bytes_per_work_unit;
  limits.effective_maximum_work_units =
      std::min(limits.maximum_work_units_by_work_unit_cap,
               limits.maximum_work_units_by_guardrail);
  if (limits.effective_maximum_work_units == 0) {
    limits.reason_code = LlmWorkPlanReason::GUARDRAIL_BELOW_ONE_WORK_UNIT;
    return limits;
  }
  limits.valid = true;
  limits.reason_code = LlmWorkPlanReason::VALID;
  return limits;
}

LlmScenarioWorkPlan build_llm_scenario_work_plan(
    const LlmMemoryWorkPlan& model_plan, LlmScenario scenario, size_t work_units,
    bool explicit_iterations) {
  LlmScenarioWorkPlan plan;
  plan.scenario = scenario;
  plan.explicit_iterations = explicit_iterations;
  if (!model_plan.valid || model_plan.plan_identity.empty()) {
    plan.reason_code = LlmWorkPlanReason::INVALID_MODEL_WORK_PLAN;
    return plan;
  }
  plan.model_plan_identity = model_plan.plan_identity;
  if (valid_scenario(scenario)) {
    plan.scenario_seed = model_plan.scenario_seeds[scenario_index(scenario)];
  }
  const LlmScenarioLimits limits =
      calculate_llm_scenario_limits(model_plan.geometry, scenario);
  if (!limits.valid) {
    plan.reason_code = limits.reason_code;
    return plan;
  }
  plan.maximum_work_units_by_work_unit_cap = limits.maximum_work_units_by_work_unit_cap;
  plan.maximum_work_units_by_guardrail = limits.maximum_work_units_by_guardrail;
  plan.effective_maximum_work_units = limits.effective_maximum_work_units;
  plan.work_unit_kind = limits.work_unit_kind;
  plan.kv_write_kind = limits.kv_write_kind;
  if (work_units == 0) {
    plan.reason_code = LlmWorkPlanReason::WORK_UNIT_COUNT_ZERO;
    return plan;
  }
  if (work_units > limits.maximum_work_units_by_work_unit_cap) {
    plan.reason_code = LlmWorkPlanReason::WORK_UNIT_CAP_EXCEEDED;
    return plan;
  }

  plan.work_units = work_units;
  plan.weight_read_bytes_per_work_unit = limits.weight_read_bytes_per_work_unit;
  plan.kv_read_bytes_per_work_unit = limits.kv_read_bytes_per_work_unit;
  plan.kv_write_bytes_per_work_unit =
      limits.kv_write_bytes_per_work_unit;
  plan.effective_model_payload_bytes_per_work_unit =
      limits.effective_model_payload_bytes_per_work_unit;
  plan.layout_metadata_lookup_count_per_work_unit =
      limits.layout_metadata_lookup_count_per_work_unit;
  plan.layout_metadata_read_bytes_per_work_unit =
      limits.layout_metadata_read_bytes_per_work_unit;
  plan.accounted_bytes_per_work_unit = limits.accounted_bytes_per_work_unit;
  if (!NumericUtils::checked_multiply(plan.weight_read_bytes_per_work_unit, work_units,
                                      plan.weight_read_bytes) ||
      !NumericUtils::checked_multiply(plan.kv_read_bytes_per_work_unit, work_units,
                                      plan.kv_read_bytes) ||
      !NumericUtils::checked_multiply(
          plan.kv_write_bytes_per_work_unit, work_units,
          plan.kv_write_bytes) ||
      !NumericUtils::checked_multiply(
          plan.effective_model_payload_bytes_per_work_unit, work_units,
          plan.effective_model_payload_bytes) ||
      !NumericUtils::checked_multiply(
          plan.layout_metadata_lookup_count_per_work_unit, work_units,
          plan.layout_metadata_lookup_count) ||
      !NumericUtils::checked_multiply(
          plan.layout_metadata_read_bytes_per_work_unit, work_units,
          plan.layout_metadata_read_bytes) ||
      !NumericUtils::checked_multiply(plan.accounted_bytes_per_work_unit,
                                      work_units,
                                      plan.task_accounted_bytes)) {
    plan.reason_code = LlmWorkPlanReason::TASK_ACCOUNTED_BYTES_OVERFLOW;
    return plan;
  }
  if (plan.task_accounted_bytes >
      Constants::LLM_MAX_ACCOUNTED_BYTES_PER_TASK) {
    plan.reason_code = LlmWorkPlanReason::TASK_ACCOUNTED_BYTES_CAP_EXCEEDED;
    return plan;
  }

  plan.plan_identity = build_scenario_plan_identity(plan);
  plan.valid = true;
  plan.reason_code = LlmWorkPlanReason::VALID;
  return plan;
}

LlmFrozenScenarioPlans freeze_llm_scenario_work_plans(
    const LlmMemoryWorkPlan& model_plan,
    const std::array<size_t, kLlmScenarioCount>& work_units,
    bool explicit_iterations) {
  LlmFrozenScenarioPlans frozen;
  frozen.explicit_iterations = explicit_iterations;
  if (!model_plan.valid || model_plan.plan_identity.empty()) {
    frozen.reason_code = LlmWorkPlanReason::INVALID_MODEL_WORK_PLAN;
    return frozen;
  }
  frozen.model_plan_identity = model_plan.plan_identity;
  constexpr std::array<LlmScenario, kLlmScenarioCount> kScenarios = {
      LlmScenario::WeightsOnly, LlmScenario::KvOnly, LlmScenario::Mixed};
  for (size_t index = 0; index < kLlmScenarioCount; ++index) {
    frozen.scenarios[index] = build_llm_scenario_work_plan(
        model_plan, kScenarios[index], work_units[index], explicit_iterations);
    if (!frozen.scenarios[index].valid) {
      frozen.reason_code = frozen.scenarios[index].reason_code;
      frozen.scenarios = {};
      frozen.model_plan_identity.clear();
      return frozen;
    }
  }

  std::string identity = Constants::LLM_WORK_PLAN_IDENTITY_VERSION;
  append_identity_field(identity, "kind", "frozen_scenarios");
  append_identity_field(identity, "explicit",
                        frozen.explicit_iterations ? 1 : 0);
  append_identity_field(identity, "model_plan_identity_size",
                        frozen.model_plan_identity.size());
  append_identity_field(identity, "model_plan_identity",
                        frozen.model_plan_identity);
  constexpr std::array<const char*, kLlmScenarioCount> kIdentitySizeNames = {
      "weights_only_identity_size", "kv_only_identity_size",
      "mixed_identity_size"};
  constexpr std::array<const char*, kLlmScenarioCount> kIdentityNames = {
      "weights_only_identity", "kv_only_identity", "mixed_identity"};
  for (size_t index = 0; index < frozen.scenarios.size(); ++index) {
    append_identity_field(identity, kIdentitySizeNames[index],
                          frozen.scenarios[index].plan_identity.size());
    append_identity_field(identity, kIdentityNames[index],
                          frozen.scenarios[index].plan_identity);
  }
  frozen.plan_identity = std::move(identity);
  frozen.valid = true;
  frozen.reason_code = LlmWorkPlanReason::VALID;
  return frozen;
}

size_t calculate_llm_pilot_work_units(const LlmScenarioLimits& limits) {
  if (!limits.valid) {
    return 0;
  }
  return NumericUtils::calculate_minimum_pilot_count(
      limits.accounted_bytes_per_work_unit,
      Constants::LLM_CALIBRATION_MIN_PILOT_BYTES,
      limits.effective_maximum_work_units);
}

size_t calculate_llm_calibrated_work_units(double attempt_duration_seconds,
                                           size_t attempt_work_units,
                                           const LlmScenarioLimits& limits) {
  if (!limits.valid) {
    return 0;
  }
  return NumericUtils::calculate_duration_scaled_count(
      attempt_duration_seconds, attempt_work_units,
      Constants::LLM_CALIBRATION_TARGET_SECONDS, 1,
      limits.effective_maximum_work_units);
}

bool llm_duration_in_target_window(double elapsed_seconds) {
  return std::isfinite(elapsed_seconds) &&
         elapsed_seconds >= Constants::LLM_CALIBRATION_MIN_SECONDS &&
         elapsed_seconds <= Constants::LLM_CALIBRATION_MAX_SECONDS;
}

std::string_view classify_llm_duration_quality(
    double elapsed_seconds, size_t work_units,
    const LlmScenarioLimits& limits) noexcept {
  if (!limits.valid || !std::isfinite(elapsed_seconds) ||
      elapsed_seconds <= 0.0 || work_units == 0 ||
      work_units > limits.effective_maximum_work_units) {
    return "invalid-duration";
  }
  if (llm_duration_in_target_window(elapsed_seconds)) {
    return "within-target-window";
  }
  if (work_units == 1 &&
      elapsed_seconds > Constants::LLM_CALIBRATION_MAX_SECONDS) {
    return "above-target-single-work-unit";
  }
  if (work_units == limits.effective_maximum_work_units &&
      elapsed_seconds < Constants::LLM_CALIBRATION_MIN_SECONDS) {
    return "guardrail-limited-below-target";
  }
  return elapsed_seconds < Constants::LLM_CALIBRATION_MIN_SECONDS
             ? "below-target-window"
             : "above-target-window";
}

std::array<LlmScenario, kLlmScenarioCount> build_llm_scenario_order(
    size_t loop_index) {
  constexpr std::array<LlmScenario, kLlmScenarioCount> kBaseOrder = {
      LlmScenario::WeightsOnly, LlmScenario::KvOnly, LlmScenario::Mixed};
  std::array<LlmScenario, kLlmScenarioCount> order{};
  const size_t rotation = loop_index % kLlmScenarioCount;
  for (size_t position = 0; position < kLlmScenarioCount; ++position) {
    order[position] =
        kBaseOrder[(rotation + position) % kLlmScenarioCount];
  }
  return order;
}

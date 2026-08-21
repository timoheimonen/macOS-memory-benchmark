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
 * @brief Pure checked geometry, range, budget, and calibration planning
 */

#include "llm_memory/llm_work_plan.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

#include "utils/cyclic_order.h"
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

std::string build_model_plan_identity(const LlmMemoryWorkPlan& plan) {
  const LlmGeometry& geometry = plan.geometry;
  std::string identity = Constants::LLM_WORK_PLAN_IDENTITY_VERSION;
  append_identity_field(identity, "backend", plan.backend);
  append_identity_field(identity, "phase", plan.phase);
  append_identity_field(identity, "methodology", plan.methodology_version);
  append_identity_field(identity, "descriptor_abi",
                        plan.descriptor_abi_version);
  append_identity_field(identity, "buffer_pattern",
                        plan.buffer_pattern_version);
  append_identity_field(identity, "worker_schedule", plan.worker_schedule);
  append_identity_field(identity, "kv_layout", plan.kv_layout);
  append_identity_field(identity, "range_alignment",
                        Constants::LLM_RANGE_ALIGNMENT_BYTES);
  append_identity_field(identity, "weight_passes_per_step",
                        plan.weight_passes_per_step);
  append_identity_field(identity, "kv_replay_factor",
                        plan.kv_replay_factor);
  append_identity_field(identity, "weight",
                        geometry.active_weight_bytes_per_step);
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
  append_identity_field(identity, "context",
                        geometry.visible_context_tokens);
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
  append_identity_field(identity, "k_mapping_bytes",
                        geometry.k_mapping_bytes);
  append_identity_field(identity, "v_mapping_bytes",
                        geometry.v_mapping_bytes);
  append_identity_field(identity, "kv_capacity_bytes",
                        geometry.kv_capacity_bytes);
  append_identity_field(identity, "weight_read_bytes_per_step",
                        geometry.weight_read_bytes_per_step);
  append_identity_field(identity, "kv_read_bytes_per_step",
                        geometry.kv_read_bytes_per_step);
  append_identity_field(identity, "kv_append_write_bytes_per_step",
                        geometry.kv_append_write_bytes_per_step);
  append_identity_field(identity, "kv_only_payload_bytes_per_step",
                        geometry.kv_only_effective_payload_bytes_per_step);
  append_identity_field(identity, "mixed_payload_bytes_per_step",
                        geometry.mixed_effective_payload_bytes_per_step);
  append_identity_field(identity, "total_data_mapping_bytes",
                        geometry.total_data_mapping_bytes);
  append_identity_field(identity, "traffic_crossover_numerator",
                        geometry.traffic_crossover_numerator);
  append_identity_field(identity, "traffic_crossover_denominator",
                        geometry.traffic_crossover_denominator);
  append_identity_field(identity, "requested_workers",
                        plan.requested_workers);
  append_identity_field(identity, "effective_workers",
                        plan.effective_workers);
  append_identity_field(identity, "layer_descriptors_per_worker",
                        plan.layer_descriptors_per_worker);
  append_identity_field(identity, "sequence_descriptors_per_worker",
                        plan.sequence_descriptors_per_worker);
  append_identity_field(identity, "total_layer_descriptors",
                        plan.total_layer_descriptors);
  append_identity_field(identity, "total_sequence_descriptors",
                        plan.total_sequence_descriptors);
  append_identity_field(identity, "descriptor_bytes", plan.descriptor_bytes);
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
  append_identity_field(identity, "scenario_seed", plan.scenario_seed);
  append_identity_field(identity, "explicit",
                        plan.explicit_iterations ? 1 : 0);
  append_identity_field(identity, "steps", plan.steps);
  append_identity_field(identity, "weight_read_bytes_per_step",
                        plan.weight_read_bytes_per_step);
  append_identity_field(identity, "kv_read_bytes_per_step",
                        plan.kv_read_bytes_per_step);
  append_identity_field(identity, "kv_append_write_bytes_per_step",
                        plan.kv_append_write_bytes_per_step);
  append_identity_field(identity, "effective_payload_bytes_per_step",
                        plan.effective_payload_bytes_per_step);
  append_identity_field(identity, "weight_read_bytes",
                        plan.weight_read_bytes);
  append_identity_field(identity, "kv_read_bytes", plan.kv_read_bytes);
  append_identity_field(identity, "kv_append_write_bytes",
                        plan.kv_append_write_bytes);
  append_identity_field(identity, "effective_payload_bytes",
                        plan.effective_payload_bytes);
  append_identity_field(identity, "maximum_steps_by_step_cap",
                        plan.maximum_steps_by_step_cap);
  append_identity_field(identity, "maximum_steps_by_payload_cap",
                        plan.maximum_steps_by_payload_cap);
  append_identity_field(identity, "effective_maximum_steps",
                        plan.effective_maximum_steps);
  return identity;
}

bool calculate_planner_storage_bytes(size_t layer_count,
                                     size_t sequence_count_per_worker,
                                     size_t worker_count,
                                     size_t& planner_storage_bytes) {
  size_t weight_layer_bytes = 0;
  size_t worker_object_bytes = 0;
  size_t layer_template_count = 0;
  size_t layer_template_bytes = 0;
  size_t sequence_template_count = 0;
  size_t sequence_template_bytes = 0;
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
          sequence_template_count, sizeof(LlmKvSequenceRangeTemplate),
          sequence_template_bytes)) {
    return false;
  }

  size_t outer_storage_bytes = 0;
  size_t template_storage_bytes = 0;
  return NumericUtils::checked_add(weight_layer_bytes, worker_object_bytes,
                                   outer_storage_bytes) &&
         NumericUtils::checked_add(layer_template_bytes,
                                   sequence_template_bytes,
                                   template_storage_bytes) &&
         NumericUtils::checked_add(outer_storage_bytes,
                                   template_storage_bytes,
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
  planner_storage_bytes = 0;
  if (!add_allocation_capacity(plan.weight_layers.capacity(),
                               sizeof(LlmByteRange),
                               planner_storage_bytes) ||
      !add_allocation_capacity(plan.workers.capacity(),
                               sizeof(LlmWorkerWorkPlan),
                               planner_storage_bytes)) {
    return false;
  }
  for (const LlmWorkerWorkPlan& worker : plan.workers) {
    if (!add_allocation_capacity(worker.layers.capacity(),
                                 sizeof(LlmLayerRangeTemplate),
                                 planner_storage_bytes) ||
        !add_allocation_capacity(worker.sequences.capacity(),
                                 sizeof(LlmKvSequenceRangeTemplate),
                                 planner_storage_bytes)) {
      return false;
    }
  }
  return true;
}

void discard_executable_templates(LlmMemoryWorkPlan& plan) {
  std::vector<LlmByteRange>().swap(plan.weight_layers);
  std::vector<LlmWorkerWorkPlan>().swap(plan.workers);
  plan.effective_workers = 0;
}

LlmMemoryWorkPlan invalid_config_plan(const std::string& reason_code) {
  LlmMemoryWorkPlan plan;
  plan.reason_code = reason_code;
  return plan;
}

}  // namespace

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
  requested_workers = other.requested_workers;
  available_workers = other.available_workers;
  effective_workers = other.effective_workers;
  layer_descriptors_per_worker = other.layer_descriptors_per_worker;
  sequence_descriptors_per_worker = other.sequence_descriptors_per_worker;
  total_layer_descriptors = other.total_layer_descriptors;
  total_sequence_descriptors = other.total_sequence_descriptors;
  descriptor_bytes = other.descriptor_bytes;
  planner_storage_bytes = other.planner_storage_bytes;
  base_seed = other.base_seed;
  weight_buffer_seed = other.weight_buffer_seed;
  k_buffer_seed = other.k_buffer_seed;
  v_buffer_seed = other.v_buffer_seed;
  scenario_seeds = other.scenario_seeds;
  memory_budget = std::move(other.memory_budget);
  weight_layers = std::move(other.weight_layers);
  workers = std::move(other.workers);
  descriptor_abi_version = std::move(other.descriptor_abi_version);
  backend = std::move(other.backend);
  phase = std::move(other.phase);
  weight_passes_per_step = other.weight_passes_per_step;
  kv_replay_factor = other.kv_replay_factor;
  buffer_pattern_version = std::move(other.buffer_pattern_version);
  methodology_version = std::move(other.methodology_version);
  worker_schedule = std::move(other.worker_schedule);
  kv_layout = std::move(other.kv_layout);
  plan_identity = std::move(other.plan_identity);

  other.valid = false;
  other.reason_code.clear();
  other.geometry.valid = false;
  other.requested_workers = 0;
  other.available_workers = 0;
  other.effective_workers = 0;
  other.layer_descriptors_per_worker = 0;
  other.sequence_descriptors_per_worker = 0;
  other.total_layer_descriptors = 0;
  other.total_sequence_descriptors = 0;
  other.descriptor_bytes = 0;
  other.planner_storage_bytes = 0;
  other.base_seed = 0;
  other.weight_buffer_seed = 0;
  other.k_buffer_seed = 0;
  other.v_buffer_seed = 0;
  other.scenario_seeds = {};
  other.memory_budget.valid = false;
  other.memory_budget.request.valid = false;
  other.weight_layers.clear();
  other.workers.clear();
  other.descriptor_abi_version.clear();
  other.backend.clear();
  other.phase.clear();
  other.buffer_pattern_version.clear();
  other.methodology_version.clear();
  other.worker_schedule.clear();
  other.kv_layout.clear();
  other.plan_identity.clear();
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
  geometry.active_weight_bytes_per_step = request.active_weight_bytes;
  geometry.layer_count = request.layer_count;
  geometry.query_head_count = request.query_head_count;
  geometry.kv_head_count = request.kv_head_count;
  geometry.head_dimension = request.head_dimension;
  geometry.kv_element_bytes = request.kv_element_bytes;
  geometry.visible_context_tokens = request.visible_context_tokens;
  geometry.batch_size = request.batch_size;

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
          geometry.k_mapping_bytes)) {
    geometry.reason_code = LlmWorkPlanReason::KV_MAPPING_BYTES_OVERFLOW;
    return geometry;
  }
  geometry.v_mapping_bytes = geometry.k_mapping_bytes;
  if (!NumericUtils::checked_add(geometry.k_mapping_bytes,
                                 geometry.v_mapping_bytes,
                                 geometry.kv_capacity_bytes)) {
    geometry.reason_code = LlmWorkPlanReason::KV_CAPACITY_BYTES_OVERFLOW;
    return geometry;
  }

  geometry.weight_read_bytes_per_step = request.active_weight_bytes;
  if (!NumericUtils::checked_multiply(
          request.batch_size, geometry.kv_bytes_per_visible_token,
          geometry.kv_append_write_bytes_per_step)) {
    geometry.reason_code = LlmWorkPlanReason::KV_APPEND_BYTES_OVERFLOW;
    return geometry;
  }
  if (!NumericUtils::checked_multiply(
          request.visible_context_tokens,
          geometry.kv_append_write_bytes_per_step,
          geometry.kv_read_bytes_per_step)) {
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
          geometry.kv_read_bytes_per_step,
          geometry.kv_append_write_bytes_per_step,
          geometry.kv_only_effective_payload_bytes_per_step)) {
    geometry.reason_code = LlmWorkPlanReason::KV_ONLY_PAYLOAD_OVERFLOW;
    return geometry;
  }
  if (!NumericUtils::checked_add(
          geometry.weight_read_bytes_per_step,
          geometry.kv_only_effective_payload_bytes_per_step,
          geometry.mixed_effective_payload_bytes_per_step)) {
    geometry.reason_code = LlmWorkPlanReason::MIXED_PAYLOAD_OVERFLOW;
    return geometry;
  }

  geometry.traffic_crossover_numerator = request.active_weight_bytes;
  geometry.traffic_crossover_denominator =
      geometry.kv_append_write_bytes_per_step;
  geometry.traffic_crossover_context_tokens = static_cast<double>(
      static_cast<long double>(geometry.traffic_crossover_numerator) /
      static_cast<long double>(geometry.traffic_crossover_denominator));
  geometry.valid = true;
  geometry.reason_code = LlmWorkPlanReason::VALID;
  return geometry;
}

LlmMemoryBudgetRequest build_llm_memory_budget_request(
    const LlmGeometry& geometry, size_t descriptor_bytes,
    size_t planner_storage_bytes,
    size_t checksum_auxiliary_bytes, size_t orchestration_auxiliary_bytes,
    size_t mapping_granularity_bytes) {
  LlmMemoryBudgetRequest request;
  request.mapping_granularity_bytes = mapping_granularity_bytes;
  request.descriptor_bytes = descriptor_bytes;
  request.planner_storage_bytes = planner_storage_bytes;
  request.checksum_auxiliary_bytes = checksum_auxiliary_bytes;
  request.orchestration_auxiliary_bytes = orchestration_auxiliary_bytes;
  if (!geometry.valid) {
    request.reason_code = geometry.reason_code;
    return request;
  }
  if (mapping_granularity_bytes == 0) {
    request.reason_code = LlmWorkPlanReason::MAPPING_GRANULARITY_ZERO;
    return request;
  }

  request.requested_weight_mapping_bytes =
      geometry.active_weight_bytes_per_step;
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
          request.committed_v_mapping_bytes)) {
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
  if (!NumericUtils::checked_add(request.committed_data_bytes,
                                 request.auxiliary_bytes,
                                 request.required_total_bytes)) {
    request.reason_code = LlmWorkPlanReason::MEMORY_REQUIREMENT_OVERFLOW;
    return request;
  }

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

LlmMemoryWorkPlan build_llm_memory_work_plan(
    const LlmMemoryWorkPlanRequest& request) {
  LlmMemoryWorkPlan plan;
  plan.requested_workers = request.requested_workers;
  plan.available_workers = request.available_workers;
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
  plan.geometry = resolve_llm_geometry(request.geometry);
  if (!plan.geometry.valid) {
    plan.reason_code = plan.geometry.reason_code;
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

  const size_t weight_layer_base =
      request.geometry.active_weight_bytes / request.geometry.layer_count;
  const size_t weight_layer_remainder =
      request.geometry.active_weight_bytes % request.geometry.layer_count;
  const size_t maximum_weight_layer_bytes =
      weight_layer_base + (weight_layer_remainder != 0 ? 1 : 0);
  const size_t maximum_shared_worker_count =
      std::min(maximum_weight_layer_bytes,
               plan.geometry.k_or_v_sequence_visible_bytes);
  plan.effective_workers =
      std::min({request.requested_workers, request.available_workers,
                maximum_shared_worker_count});
  if (plan.effective_workers == 0) {
    plan.reason_code = LlmWorkPlanReason::NO_EXECUTABLE_WORKER;
    return plan;
  }

  plan.layer_descriptors_per_worker = plan.geometry.layer_count;
  if (!NumericUtils::checked_multiply(
          plan.geometry.layer_count, plan.geometry.batch_size,
          plan.sequence_descriptors_per_worker)) {
    plan.reason_code = LlmWorkPlanReason::LAYER_SEQUENCE_COUNT_OVERFLOW;
    return plan;
  }
  if (!NumericUtils::checked_multiply(
          plan.layer_descriptors_per_worker, plan.effective_workers,
          plan.total_layer_descriptors) ||
      !NumericUtils::checked_multiply(
          plan.sequence_descriptors_per_worker, plan.effective_workers,
          plan.total_sequence_descriptors)) {
    plan.reason_code = LlmWorkPlanReason::DESCRIPTOR_COUNT_OVERFLOW;
    return plan;
  }
  size_t layer_descriptor_bytes = 0;
  size_t sequence_descriptor_bytes = 0;
  if (!NumericUtils::checked_multiply(
          plan.total_layer_descriptors, sizeof(LlmLayerDescriptor),
          layer_descriptor_bytes) ||
      !NumericUtils::checked_multiply(
          plan.total_sequence_descriptors, sizeof(LlmKvSequenceDescriptor),
          sequence_descriptor_bytes) ||
      !NumericUtils::checked_add(layer_descriptor_bytes,
                                 sequence_descriptor_bytes,
                                 plan.descriptor_bytes)) {
    plan.reason_code = LlmWorkPlanReason::DESCRIPTOR_BYTES_OVERFLOW;
    return plan;
  }

  if (!calculate_planner_storage_bytes(
          plan.geometry.layer_count, plan.sequence_descriptors_per_worker,
          plan.effective_workers, plan.planner_storage_bytes)) {
    plan.reason_code = LlmWorkPlanReason::PLANNER_STORAGE_BYTES_OVERFLOW;
    return plan;
  }

  plan.memory_budget.request = build_llm_memory_budget_request(
      plan.geometry, plan.descriptor_bytes, plan.planner_storage_bytes,
      request.checksum_auxiliary_bytes,
      request.orchestration_auxiliary_bytes,
      request.mapping_granularity_bytes);
  plan.memory_budget = evaluate_llm_memory_budget(
      plan.memory_budget.request, request.available_memory_bytes);
  if (!plan.memory_budget.valid) {
    plan.reason_code = plan.memory_budget.reason_code;
    return plan;
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

    plan.workers.resize(plan.effective_workers);
    for (size_t worker = 0; worker < plan.effective_workers; ++worker) {
      plan.workers[worker].worker_index = worker;
      plan.workers[worker].layers.reserve(plan.geometry.layer_count);
      plan.workers[worker].sequences.reserve(
          plan.sequence_descriptors_per_worker);
    }

    for (size_t layer = 0; layer < plan.geometry.layer_count; ++layer) {
      const std::vector<LlmByteRange> weight_ranges = partition_range(
          plan.weight_layers[layer].offset_bytes,
          plan.weight_layers[layer].span_bytes, plan.effective_workers);
      const size_t first_sequence_index = layer * plan.geometry.batch_size;
      for (size_t worker = 0; worker < plan.effective_workers; ++worker) {
        const LlmByteRange weight =
            worker < weight_ranges.size() ? weight_ranges[worker]
                                          : LlmByteRange{};
        plan.workers[worker].layers.push_back(
            {weight, first_sequence_index, plan.geometry.batch_size, layer});
      }

      for (size_t batch = 0; batch < plan.geometry.batch_size; ++batch) {
        const size_t sequence_index = first_sequence_index + batch;
        const size_t visible_offset =
            sequence_index * plan.geometry.k_or_v_sequence_visible_bytes;
        const LlmByteRange visible_record{
            visible_offset, plan.geometry.k_or_v_sequence_visible_bytes};
        const size_t append_offset =
            visible_offset +
            (plan.geometry.visible_context_tokens - 1) *
                plan.geometry.k_or_v_record_bytes_per_layer;
        const LlmByteRange append_record{
            append_offset, plan.geometry.k_or_v_record_bytes_per_layer};
        const std::vector<LlmByteRange> visible_ranges = partition_range(
            visible_record.offset_bytes, visible_record.span_bytes,
            plan.effective_workers);
        for (size_t worker = 0; worker < plan.effective_workers; ++worker) {
          const LlmByteRange visible =
              worker < visible_ranges.size() ? visible_ranges[worker]
                                             : LlmByteRange{};
          const LlmByteRange append = intersect_ranges(visible, append_record);
          const size_t append_record_byte_offset =
              append.span_bytes == 0 ? 0 : append.offset_bytes - append_offset;
          plan.workers[worker].sequences.push_back(
              {visible, visible, append, append, layer, batch,
               append_record_byte_offset});
        }
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
  plan.planner_storage_bytes = actual_planner_storage_bytes;
  plan.memory_budget.request = build_llm_memory_budget_request(
      plan.geometry, plan.descriptor_bytes, plan.planner_storage_bytes,
      request.checksum_auxiliary_bytes,
      request.orchestration_auxiliary_bytes,
      request.mapping_granularity_bytes);
  plan.memory_budget = evaluate_llm_memory_budget(
      plan.memory_budget.request, request.available_memory_bytes);
  if (!plan.memory_budget.valid) {
    discard_executable_templates(plan);
    plan.reason_code = plan.memory_budget.reason_code;
    return plan;
  }

  plan.descriptor_abi_version = Constants::LLM_DESCRIPTOR_ABI_VERSION;
  plan.backend = Constants::LLM_BACKEND_NAME;
  plan.phase = Constants::LLM_PHASE_NAME;
  plan.buffer_pattern_version = Constants::LLM_BUFFER_PATTERN_VERSION;
  plan.methodology_version = Constants::LLM_METHODOLOGY_VERSION;
  plan.worker_schedule = Constants::LLM_WORKER_SCHEDULE;
  plan.kv_layout = Constants::LLM_KV_LAYOUT;
  plan.plan_identity = build_model_plan_identity(plan);
  plan.valid = true;
  plan.reason_code = LlmWorkPlanReason::VALID;
  return plan;
}

LlmMemoryWorkPlan build_llm_memory_work_plan(
    const LlmMemoryConfig& config, size_t available_workers,
    size_t available_memory_bytes, size_t mapping_granularity_bytes,
    size_t checksum_auxiliary_bytes,
    size_t orchestration_auxiliary_bytes) {
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
                      config.batch_size};
  request.requested_workers = config.requested_workers;
  request.available_workers = available_workers;
  request.available_memory_bytes = available_memory_bytes;
  request.mapping_granularity_bytes = mapping_granularity_bytes;
  request.checksum_auxiliary_bytes = checksum_auxiliary_bytes;
  request.orchestration_auxiliary_bytes = orchestration_auxiliary_bytes;
  request.base_seed = config.seed;
  return build_llm_memory_work_plan(request);
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

  switch (scenario) {
    case LlmScenario::WeightsOnly:
      limits.weight_read_bytes_per_step =
          geometry.weight_read_bytes_per_step;
      limits.effective_payload_bytes_per_step =
          geometry.weight_read_bytes_per_step;
      break;
    case LlmScenario::KvOnly:
      limits.kv_read_bytes_per_step = geometry.kv_read_bytes_per_step;
      limits.kv_append_write_bytes_per_step =
          geometry.kv_append_write_bytes_per_step;
      limits.effective_payload_bytes_per_step =
          geometry.kv_only_effective_payload_bytes_per_step;
      break;
    case LlmScenario::Mixed:
      limits.weight_read_bytes_per_step =
          geometry.weight_read_bytes_per_step;
      limits.kv_read_bytes_per_step = geometry.kv_read_bytes_per_step;
      limits.kv_append_write_bytes_per_step =
          geometry.kv_append_write_bytes_per_step;
      limits.effective_payload_bytes_per_step =
          geometry.mixed_effective_payload_bytes_per_step;
      break;
  }

  limits.maximum_steps_by_payload_cap =
      Constants::LLM_MAX_EXACT_PAYLOAD_BYTES /
      limits.effective_payload_bytes_per_step;
  limits.effective_maximum_steps =
      std::min(limits.maximum_steps_by_step_cap,
               limits.maximum_steps_by_payload_cap);
  if (limits.effective_maximum_steps == 0) {
    limits.reason_code = LlmWorkPlanReason::PAYLOAD_CAP_BELOW_ONE_STEP;
    return limits;
  }
  limits.valid = true;
  limits.reason_code = LlmWorkPlanReason::VALID;
  return limits;
}

LlmScenarioWorkPlan build_llm_scenario_work_plan(
    const LlmMemoryWorkPlan& model_plan, LlmScenario scenario, size_t steps,
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
  plan.maximum_steps_by_step_cap = limits.maximum_steps_by_step_cap;
  plan.maximum_steps_by_payload_cap = limits.maximum_steps_by_payload_cap;
  plan.effective_maximum_steps = limits.effective_maximum_steps;
  if (steps == 0) {
    plan.reason_code = LlmWorkPlanReason::STEP_COUNT_ZERO;
    return plan;
  }
  if (steps > limits.maximum_steps_by_step_cap) {
    plan.reason_code = LlmWorkPlanReason::STEP_CAP_EXCEEDED;
    return plan;
  }

  plan.steps = steps;
  plan.weight_read_bytes_per_step = limits.weight_read_bytes_per_step;
  plan.kv_read_bytes_per_step = limits.kv_read_bytes_per_step;
  plan.kv_append_write_bytes_per_step =
      limits.kv_append_write_bytes_per_step;
  plan.effective_payload_bytes_per_step =
      limits.effective_payload_bytes_per_step;
  if (!NumericUtils::checked_multiply(plan.weight_read_bytes_per_step, steps,
                                      plan.weight_read_bytes) ||
      !NumericUtils::checked_multiply(plan.kv_read_bytes_per_step, steps,
                                      plan.kv_read_bytes) ||
      !NumericUtils::checked_multiply(
          plan.kv_append_write_bytes_per_step, steps,
          plan.kv_append_write_bytes) ||
      !NumericUtils::checked_multiply(
          plan.effective_payload_bytes_per_step, steps,
          plan.effective_payload_bytes)) {
    plan.reason_code = LlmWorkPlanReason::EXACT_PAYLOAD_OVERFLOW;
    return plan;
  }
  if (plan.effective_payload_bytes >
      Constants::LLM_MAX_EXACT_PAYLOAD_BYTES) {
    plan.reason_code = LlmWorkPlanReason::EXACT_PAYLOAD_CAP_EXCEEDED;
    return plan;
  }

  plan.plan_identity = build_scenario_plan_identity(plan);
  plan.valid = true;
  plan.reason_code = LlmWorkPlanReason::VALID;
  return plan;
}

LlmFrozenScenarioPlans freeze_llm_scenario_work_plans(
    const LlmMemoryWorkPlan& model_plan,
    const std::array<size_t, kLlmScenarioCount>& steps,
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
        model_plan, kScenarios[index], steps[index], explicit_iterations);
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

size_t calculate_llm_pilot_steps(const LlmScenarioLimits& limits) {
  if (!limits.valid) {
    return 0;
  }
  return NumericUtils::calculate_minimum_pilot_count(
      limits.effective_payload_bytes_per_step,
      Constants::LLM_CALIBRATION_MIN_PILOT_BYTES,
      limits.effective_maximum_steps);
}

size_t calculate_llm_calibrated_steps(double attempt_duration_seconds,
                                      size_t attempt_steps,
                                      const LlmScenarioLimits& limits) {
  if (!limits.valid) {
    return 0;
  }
  return NumericUtils::calculate_duration_scaled_count(
      attempt_duration_seconds, attempt_steps,
      Constants::LLM_CALIBRATION_TARGET_SECONDS, 1,
      limits.effective_maximum_steps);
}

bool llm_duration_in_target_window(double elapsed_seconds) {
  return std::isfinite(elapsed_seconds) &&
         elapsed_seconds >= Constants::LLM_CALIBRATION_MIN_SECONDS &&
         elapsed_seconds <= Constants::LLM_CALIBRATION_MAX_SECONDS;
}

std::string classify_llm_duration_quality(double elapsed_seconds,
                                          size_t steps,
                                          const LlmScenarioLimits& limits) {
  if (!limits.valid || !std::isfinite(elapsed_seconds) ||
      elapsed_seconds <= 0.0 || steps == 0 ||
      steps > limits.effective_maximum_steps) {
    return "invalid-duration";
  }
  if (llm_duration_in_target_window(elapsed_seconds)) {
    return "within-target-window";
  }
  if (steps == 1 &&
      elapsed_seconds > Constants::LLM_CALIBRATION_MAX_SECONDS) {
    return "single-step-over-target";
  }
  if (steps == limits.effective_maximum_steps &&
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
  const std::vector<size_t> indexes =
      build_cyclic_order(kLlmScenarioCount, loop_index);
  for (size_t position = 0; position < indexes.size(); ++position) {
    order[position] = kBaseOrder[indexes[position]];
  }
  return order;
}

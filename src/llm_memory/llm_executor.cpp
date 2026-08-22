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
 * @file llm_executor.cpp
 * @brief CPU LLM resource preparation and synchronized scenario execution
 */

#include "llm_memory/llm_executor.h"

#include <mach/mach.h>
#include <pthread/qos.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <system_error>
#include <thread>
#include <utility>

#include "core/timing/timer.h"
#include "utils/numeric_utils.h"

namespace {

constexpr uint64_t kBufferPatternMultiplier = 0x9E3779B97F4A7C15ULL;

constexpr uint64_t kAppendStepMultiplier = 0x9E3779B97F4A7C15ULL;
constexpr uint64_t kAppendLayerMultiplier = 0xBF58476D1CE4E5B9ULL;
constexpr uint64_t kAppendBatchMultiplier = 0x94D049BB133111EBULL;
constexpr uint64_t kAppendWordMultiplier = 0xD6E8FEB86659FD93ULL;
constexpr uint64_t kAppendKDomain = 0x4B4B4B4B4B4B4B4BULL;
constexpr uint64_t kAppendVDomain = 0x5656565656565656ULL;

constexpr uint64_t kChecksumInitialA = 0x243F6A8885A308D3ULL;
constexpr uint64_t kChecksumInitialB = 0x13198A2E03707344ULL;
constexpr uint64_t kChecksumWeightDomain = 0x5745494748545F31ULL;
constexpr uint64_t kChecksumKDomain = 0x4B5F524541445F31ULL;
constexpr uint64_t kChecksumVDomain = 0x565F524541445F31ULL;

constexpr uint64_t kPagedPatternLayerMultiplier = 0xA24BAED4963EE407ULL;
constexpr uint64_t kPagedPatternPhysicalMultiplier = 0x9FB21C651E98DF25ULL;
constexpr uint64_t kPagedPatternWordMultiplier = 0xC13FA9A902A6328FULL;
constexpr uint64_t kPagedLookupLogicalMultiplier = 0x9E3779B97F4A7C15ULL;
constexpr uint64_t kPagedLookupWorkUnitMultiplier = 0xBF58476D1CE4E5B9ULL;
constexpr uint64_t kPagedLookupVisitMultiplier = 0x94D049BB133111EBULL;
constexpr uint64_t kPagedLookupStateBStep = 0xD6E8FEB86659FD93ULL;

constexpr uint64_t kRunInitialA = 0x6A09E667F3BCC909ULL;
constexpr uint64_t kRunInitialB = 0xBB67AE8584CAA73BULL;

uint64_t rotate_left(uint64_t value, unsigned int shift) noexcept {
  return (value << shift) | (value >> (64U - shift));
}

uint64_t checksum_domain(LlmChecksumComponent component) noexcept {
  switch (component) {
    case LlmChecksumComponent::Weight:
      return kChecksumWeightDomain;
    case LlmChecksumComponent::K:
      return kChecksumKDomain;
    case LlmChecksumComponent::V:
      return kChecksumVDomain;
  }
  return 0;
}

uint64_t append_domain(LlmChecksumComponent component) noexcept {
  switch (component) {
    case LlmChecksumComponent::K:
      return kAppendKDomain;
    case LlmChecksumComponent::V:
      return kAppendVDomain;
    case LlmChecksumComponent::Weight:
      return 0;
  }
  return 0;
}

bool checked_range_end(const LlmByteRange& range, size_t limit, size_t& end) noexcept {
  if (range.span_bytes == 0) {
    return range.offset_bytes == 0;
  }
  return NumericUtils::checked_add(range.offset_bytes, range.span_bytes, end) && end <= limit;
}

bool equal_ranges(const LlmByteRange& lhs, const LlmByteRange& rhs) noexcept {
  return lhs.offset_bytes == rhs.offset_bytes && lhs.span_bytes == rhs.span_bytes;
}

bool checked_add_to(size_t value, size_t& total) noexcept;

bool paged_layout_matches_model(
    const LlmMemoryWorkPlan& plan,
    const LlmPagedCpuExecutionPlan& paged) noexcept {
  const LlmGeometry& geometry = plan.geometry;
  const LlmKvLayoutPlan& layout = paged.layout;
  const size_t visible_tokens =
      plan.phase == LlmPhase::Decode && geometry.decode.has_value()
          ? geometry.decode->visible_context_tokens
      : plan.phase == LlmPhase::Prefill && geometry.prefill.has_value()
          ? geometry.prefill->prompt_tokens
          : 0;
  if (visible_tokens == 0 || geometry.layer_count == 0 ||
      geometry.batch_size == 0 || geometry.kv_block_tokens == 0 ||
      geometry.k_or_v_record_bytes_per_layer == 0) {
    return false;
  }

  const size_t expected_blocks_per_sequence =
      visible_tokens / geometry.kv_block_tokens +
      (visible_tokens % geometry.kv_block_tokens != 0 ? 1 : 0);
  size_t expected_physical_blocks_per_layer = 0;
  size_t expected_total_physical_blocks = 0;
  size_t expected_block_bytes = 0;
  size_t expected_last_block_start = 0;
  size_t expected_last_block_valid_bytes = 0;
  size_t expected_append_offset = 0;
  size_t expected_layer_sequences = 0;
  size_t expected_logical_records = 0;
  size_t expected_k_logical_bytes = 0;
  size_t expected_k_physical_bytes = 0;
  size_t expected_block_table_bytes = 0;
  size_t expected_transient_peak_bytes = 0;
  size_t expected_resident_layout_bytes = 0;
  size_t expected_known_owned_peak_bytes = 0;
  size_t expected_kv_capacity_bytes = 0;
  size_t expected_total_data_bytes = 0;
  if (visible_tokens == 0 || expected_blocks_per_sequence == 0 ||
      !NumericUtils::checked_multiply(
          geometry.batch_size, expected_blocks_per_sequence,
          expected_physical_blocks_per_layer) ||
      !NumericUtils::checked_multiply(
          geometry.layer_count, expected_physical_blocks_per_layer,
          expected_total_physical_blocks) ||
      !NumericUtils::checked_multiply(
          geometry.kv_block_tokens,
          geometry.k_or_v_record_bytes_per_layer,
          expected_block_bytes) ||
      !NumericUtils::checked_multiply(
          expected_blocks_per_sequence - 1, geometry.kv_block_tokens,
          expected_last_block_start) ||
      expected_last_block_start >= visible_tokens ||
      !NumericUtils::checked_multiply(
          visible_tokens - expected_last_block_start,
          geometry.k_or_v_record_bytes_per_layer,
          expected_last_block_valid_bytes) ||
      !NumericUtils::checked_multiply(
          visible_tokens - expected_last_block_start - 1,
          geometry.k_or_v_record_bytes_per_layer,
          expected_append_offset) ||
      !NumericUtils::checked_multiply(
          geometry.layer_count, geometry.batch_size,
          expected_layer_sequences) ||
      !NumericUtils::checked_multiply(
          expected_layer_sequences, visible_tokens,
          expected_logical_records) ||
      !NumericUtils::checked_multiply(
          expected_logical_records,
          geometry.k_or_v_record_bytes_per_layer,
          expected_k_logical_bytes) ||
      !NumericUtils::checked_multiply(
          expected_total_physical_blocks, expected_block_bytes,
          expected_k_physical_bytes) ||
      !NumericUtils::checked_multiply(
          expected_physical_blocks_per_layer,
          Constants::LLM_KV_BLOCK_TABLE_ENTRY_BYTES,
          expected_block_table_bytes)) {
    return false;
  }

  const size_t expected_validation_bitset_bytes =
      expected_physical_blocks_per_layer / 8 +
      (expected_physical_blocks_per_layer % 8 != 0 ? 1 : 0);
  if (expected_k_physical_bytes < expected_k_logical_bytes ||
      !NumericUtils::checked_add(
          expected_block_table_bytes, expected_validation_bitset_bytes,
          expected_transient_peak_bytes) ||
      !NumericUtils::checked_add(
          expected_k_physical_bytes, expected_k_physical_bytes,
          expected_resident_layout_bytes) ||
      !NumericUtils::checked_add(
          expected_resident_layout_bytes, expected_block_table_bytes,
          expected_resident_layout_bytes) ||
      !NumericUtils::checked_add(
          expected_resident_layout_bytes, expected_validation_bitset_bytes,
          expected_known_owned_peak_bytes) ||
      !NumericUtils::checked_add(
          expected_k_physical_bytes, expected_k_physical_bytes,
          expected_kv_capacity_bytes) ||
      !NumericUtils::checked_add(
          geometry.active_weight_bytes_per_work_unit,
          expected_kv_capacity_bytes, expected_total_data_bytes)) {
    return false;
  }
  const size_t expected_padding_bytes =
      expected_k_physical_bytes - expected_k_logical_bytes;

  return layout.reason_code == LlmKvLayoutReason::VALID &&
         layout.sequence_tokens == visible_tokens &&
         layout.kv_block_tokens == geometry.kv_block_tokens &&
         layout.layer_count == geometry.layer_count &&
         layout.batch_size == geometry.batch_size &&
         layout.k_or_v_record_bytes_per_layer ==
             geometry.k_or_v_record_bytes_per_layer &&
         layout.blocks_per_sequence == expected_blocks_per_sequence &&
         layout.physical_blocks_per_layer ==
             expected_physical_blocks_per_layer &&
         layout.total_physical_blocks == expected_total_physical_blocks &&
         layout.block_bytes == expected_block_bytes &&
         layout.last_block_tokens ==
             visible_tokens - expected_last_block_start &&
         layout.last_block_valid_bytes == expected_last_block_valid_bytes &&
         layout.decode_append_offset_in_last_block == expected_append_offset &&
         layout.block_table_entries == expected_physical_blocks_per_layer &&
         layout.permutation_iterations ==
             expected_physical_blocks_per_layer - 1 &&
         layout.validation_entries == expected_physical_blocks_per_layer &&
         layout.hash_entries == expected_physical_blocks_per_layer &&
         layout.upload_bytes == expected_block_table_bytes &&
         layout.memory.k_logical_bytes == expected_k_logical_bytes &&
         layout.memory.v_logical_bytes == expected_k_logical_bytes &&
         layout.memory.k_physical_bytes == expected_k_physical_bytes &&
         layout.memory.v_physical_bytes == expected_k_physical_bytes &&
         layout.memory.k_layout_padding_bytes == expected_padding_bytes &&
         layout.memory.v_layout_padding_bytes == expected_padding_bytes &&
         layout.memory.block_table_bytes == expected_block_table_bytes &&
         layout.memory.validation_bitset_bytes ==
             expected_validation_bitset_bytes &&
         layout.memory.transient_peak_bytes == expected_transient_peak_bytes &&
         layout.memory.resident_layout_bytes == expected_resident_layout_bytes &&
         layout.memory.known_owned_peak_bytes == expected_known_owned_peak_bytes &&
         !layout.geometry_identity.empty() &&
         geometry.kv_blocks_per_sequence == expected_blocks_per_sequence &&
         geometry.physical_blocks_per_layer ==
             expected_physical_blocks_per_layer &&
         geometry.total_physical_blocks == expected_total_physical_blocks &&
         geometry.kv_block_bytes == expected_block_bytes &&
         geometry.last_block_tokens ==
             visible_tokens - expected_last_block_start &&
         geometry.last_block_valid_bytes == expected_last_block_valid_bytes &&
         geometry.decode_append_offset_in_last_block ==
             (plan.phase == LlmPhase::Decode ? expected_append_offset : 0) &&
         geometry.k_logical_bytes == expected_k_logical_bytes &&
         geometry.v_logical_bytes == expected_k_logical_bytes &&
         geometry.k_mapping_bytes == expected_k_physical_bytes &&
         geometry.v_mapping_bytes == expected_k_physical_bytes &&
         geometry.k_layout_padding_bytes == expected_padding_bytes &&
         geometry.v_layout_padding_bytes == expected_padding_bytes &&
         geometry.block_table_entries == expected_physical_blocks_per_layer &&
         geometry.block_table_bytes == expected_block_table_bytes &&
         geometry.kv_capacity_bytes == expected_kv_capacity_bytes &&
         geometry.total_data_mapping_bytes == expected_total_data_bytes &&
         plan.memory_budget.request.requested_k_mapping_bytes ==
             expected_k_physical_bytes &&
         plan.memory_budget.request.requested_v_mapping_bytes ==
             expected_k_physical_bytes &&
         plan.memory_budget.request.requested_data_bytes ==
             expected_total_data_bytes &&
         plan.memory_budget.request.requested_block_table_mapping_bytes ==
             expected_block_table_bytes &&
         plan.memory_budget.request.layout_transient_bytes ==
             expected_validation_bitset_bytes &&
         paged.table_validation.valid &&
         !paged.table_validation.interrupted &&
         paged.table_validation.expected_entries ==
             expected_physical_blocks_per_layer &&
         paged.table_validation.examined_entries ==
             expected_physical_blocks_per_layer &&
         paged.table_validation.validation_bitset_bytes ==
             expected_validation_bitset_bytes &&
         paged.permutation.algorithm_version ==
             Constants::LLM_KV_BLOCK_PERMUTATION_VERSION &&
         paged.permutation.entry_count == expected_physical_blocks_per_layer &&
         !paged.permutation.sha256.empty() &&
         validate_llm_kv_layout_identity(
             layout, paged.permutation, paged.layout_identity) &&
         (plan.phase == LlmPhase::Prefill
              ? !paged.execution_identity.empty()
              : paged.ownership.layout_geometry_identity ==
                        layout.geometry_identity &&
                    paged.execution_identity == paged.ownership.identity);
}

bool validate_paged_work_plan_layout(const LlmMemoryWorkPlan& plan,
                                     const LlmCpuExecutionPlan& cpu_plan) noexcept {
  if (!cpu_plan.paged.has_value()) {
    return false;
  }
  const LlmPagedCpuExecutionPlan& paged = *cpu_plan.paged;
  const LlmKvLayoutPlan& layout = paged.layout;
  size_t expected_sequences = 0;
  size_t expected_total_layers = 0;
  size_t expected_total_assignments = 0;
  size_t layer_bytes = 0;
  size_t assignment_bytes = 0;
  size_t expected_descriptor_bytes = 0;
  if (!plan.valid || !plan.geometry.valid || !plan.memory_budget.valid ||
      plan.kv_layout != LlmKvLayout::Paged || !layout.valid ||
      !paged_layout_matches_model(plan, paged) ||
      !paged.table_validation.valid || paged.permutation.entry_count !=
                                            layout.block_table_entries ||
      paged.block_table() == nullptr || !paged.block_table_read_only ||
      paged.block_table_logical_bytes != layout.memory.block_table_bytes ||
      paged.block_table_mapping_bytes !=
          plan.memory_budget.request.committed_block_table_mapping_bytes ||
      paged.layout_identity.empty() || paged.execution_identity.empty() ||
      cpu_plan.effective_workers == 0 ||
      cpu_plan.workers.size() != cpu_plan.effective_workers ||
      plan.weight_layers.size() != plan.geometry.layer_count ||
      cpu_plan.layer_descriptors_per_worker != plan.geometry.layer_count ||
      !NumericUtils::checked_multiply(plan.geometry.layer_count,
                                      plan.geometry.batch_size,
                                      expected_sequences) ||
      expected_sequences != cpu_plan.sequence_descriptors_per_worker ||
      !NumericUtils::checked_multiply(cpu_plan.effective_workers,
                                      cpu_plan.layer_descriptors_per_worker,
                                      expected_total_layers) ||
      !NumericUtils::checked_multiply(cpu_plan.effective_workers,
                                      expected_sequences,
                                      expected_total_assignments) ||
      expected_total_layers != cpu_plan.total_layer_descriptors ||
      expected_total_assignments != cpu_plan.total_sequence_descriptors ||
      !NumericUtils::checked_multiply(expected_total_layers,
                                      sizeof(LlmPagedLayerDescriptor),
                                      layer_bytes) ||
      !NumericUtils::checked_multiply(
          expected_total_assignments,
          sizeof(LlmPagedKvAssignmentDescriptor), assignment_bytes) ||
      !NumericUtils::checked_add(layer_bytes, assignment_bytes,
                                 expected_descriptor_bytes) ||
      expected_descriptor_bytes != cpu_plan.descriptor_bytes ||
      plan.geometry.k_mapping_bytes != layout.memory.k_physical_bytes ||
      plan.geometry.v_mapping_bytes != layout.memory.v_physical_bytes ||
      plan.geometry.kv_blocks_per_sequence != layout.blocks_per_sequence ||
      plan.geometry.kv_block_bytes != layout.block_bytes) {
    return false;
  }

  size_t weight_cursor = 0;
  for (size_t layer = 0; layer < plan.geometry.layer_count; ++layer) {
    const LlmByteRange& layer_range = plan.weight_layers[layer];
    size_t layer_end = 0;
    if (layer_range.span_bytes == 0 ||
        layer_range.offset_bytes != weight_cursor ||
        !checked_range_end(layer_range,
                           plan.geometry.active_weight_bytes_per_work_unit,
                           layer_end)) {
      return false;
    }
    size_t worker_cursor = layer_range.offset_bytes;
    for (size_t worker_index = 0;
         worker_index < cpu_plan.effective_workers; ++worker_index) {
      const LlmWorkerWorkPlan& worker = cpu_plan.workers[worker_index];
      if (worker.worker_index != worker_index ||
          worker.layers.size() != cpu_plan.layer_descriptors_per_worker ||
          !worker.sequences.empty() ||
          worker.paged_assignments.size() != expected_sequences) {
        return false;
      }
      const LlmLayerRangeTemplate& worker_layer = worker.layers[layer];
      size_t worker_end = 0;
      if (worker_layer.layer_index != layer ||
          worker_layer.first_sequence_index !=
              layer * plan.geometry.batch_size ||
          worker_layer.sequence_count != plan.geometry.batch_size ||
          !checked_range_end(worker_layer.weight,
                             plan.geometry.active_weight_bytes_per_work_unit,
                             worker_end)) {
        return false;
      }
      if (worker_layer.weight.span_bytes != 0) {
        if (worker_layer.weight.offset_bytes != worker_cursor ||
            worker_end > layer_end) {
          return false;
        }
        worker_cursor = worker_end;
      }
    }
    if (worker_cursor != layer_end) {
      return false;
    }
    weight_cursor = layer_end;

    for (size_t batch = 0; batch < plan.geometry.batch_size; ++batch) {
      const size_t assignment_index =
          layer * plan.geometry.batch_size + batch;
      size_t owned_blocks = 0;
      for (size_t worker_index = 0;
           worker_index < cpu_plan.effective_workers; ++worker_index) {
        const LlmPagedKvAssignmentTemplate& assignment =
            cpu_plan.workers[worker_index]
                .paged_assignments[assignment_index];
        size_t range_end = 0;
        if (assignment.layer_index != layer ||
            assignment.batch_sequence_index != batch ||
            (assignment.block_count == 0 &&
             assignment.first_logical_block != 0) ||
            (assignment.block_count != 0 &&
             (!NumericUtils::checked_add(assignment.first_logical_block,
                                         assignment.block_count, range_end) ||
              range_end > layout.blocks_per_sequence)) ||
            !NumericUtils::checked_add(owned_blocks,
                                       assignment.block_count,
                                       owned_blocks)) {
          return false;
        }
        if (assignment.block_count == 0) {
          continue;
        }
        for (size_t other = 0; other < worker_index; ++other) {
          const LlmPagedKvAssignmentTemplate& previous =
              cpu_plan.workers[other]
                  .paged_assignments[assignment_index];
          if (previous.block_count == 0) {
            continue;
          }
          const size_t previous_end =
              previous.first_logical_block + previous.block_count;
          if (assignment.first_logical_block < previous_end &&
              previous.first_logical_block < range_end) {
            return false;
          }
        }
      }
      if (owned_blocks != layout.blocks_per_sequence) {
        return false;
      }
    }
  }
  return weight_cursor == plan.geometry.active_weight_bytes_per_work_unit;
}

bool validate_prefill_work_plan_layout(
    const LlmMemoryWorkPlan& plan,
    const LlmCpuExecutionPlan& cpu_plan) noexcept {
  if (!plan.valid || !plan.geometry.valid || !plan.memory_budget.valid ||
      plan.phase != LlmPhase::Prefill ||
      plan.kv_layout != LlmKvLayout::Contiguous ||
      !plan.geometry.prefill.has_value() || !plan.prefill_plan.has_value() ||
      !plan.prefill_plan->valid || !cpu_plan.prefill.has_value() ||
      !validate_llm_prefill_cpu_execution_evidence(plan) ||
      cpu_plan.effective_workers == 0 ||
      cpu_plan.workers.size() != cpu_plan.effective_workers ||
      plan.weight_layers.size() != plan.geometry.layer_count ||
      cpu_plan.layer_descriptors_per_worker != plan.geometry.layer_count) {
    return false;
  }
  const LlmPrefillCpuExecutionPlan& prefill = *cpu_plan.prefill;
  size_t sequences_per_scenario = 0;
  size_t sequences_per_worker = 0;
  size_t expected_total_layers = 0;
  size_t expected_total_sequences = 0;
  size_t layer_bytes = 0;
  size_t sequence_bytes = 0;
  size_t descriptor_bytes = 0;
  if (!NumericUtils::checked_multiply(
          plan.geometry.layer_count, plan.geometry.batch_size,
          sequences_per_scenario) ||
      sequences_per_scenario !=
          prefill.sequence_descriptors_per_scenario_per_worker ||
      !NumericUtils::checked_multiply(sequences_per_scenario,
                                      kLlmScenarioCount,
                                      sequences_per_worker) ||
      sequences_per_worker != cpu_plan.sequence_descriptors_per_worker ||
      !NumericUtils::checked_multiply(cpu_plan.effective_workers,
                                      plan.geometry.layer_count,
                                      expected_total_layers) ||
      !NumericUtils::checked_multiply(cpu_plan.effective_workers,
                                      sequences_per_worker,
                                      expected_total_sequences) ||
      expected_total_layers != cpu_plan.total_layer_descriptors ||
      expected_total_sequences != cpu_plan.total_sequence_descriptors ||
      !NumericUtils::checked_multiply(expected_total_layers,
                                      sizeof(LlmPrefillLayerDescriptor),
                                      layer_bytes) ||
      !NumericUtils::checked_multiply(
          expected_total_sequences,
          sizeof(LlmPrefillKvSequenceDescriptor), sequence_bytes) ||
      !NumericUtils::checked_add(layer_bytes, sequence_bytes,
                                 descriptor_bytes) ||
      descriptor_bytes != cpu_plan.descriptor_bytes ||
      prefill.identity.empty()) {
    return false;
  }

  size_t weight_cursor = 0;
  for (size_t layer = 0; layer < plan.geometry.layer_count; ++layer) {
    const LlmByteRange& layer_range = plan.weight_layers[layer];
    size_t layer_end = 0;
    if (layer_range.span_bytes == 0 ||
        layer_range.offset_bytes != weight_cursor ||
        !checked_range_end(layer_range,
                           plan.geometry.active_weight_bytes_per_work_unit,
                           layer_end)) {
      return false;
    }
    size_t worker_cursor = layer_range.offset_bytes;
    for (size_t worker_index = 0;
         worker_index < cpu_plan.effective_workers; ++worker_index) {
      const LlmWorkerWorkPlan& worker = cpu_plan.workers[worker_index];
      if (worker.worker_index != worker_index ||
          worker.layers.size() != plan.geometry.layer_count ||
          !worker.sequences.empty() || !worker.paged_assignments.empty() ||
          worker.prefill_sequences.size() != sequences_per_worker) {
        return false;
      }
      const LlmLayerRangeTemplate& worker_layer = worker.layers[layer];
      size_t worker_end = 0;
      if (worker_layer.layer_index != layer ||
          worker_layer.first_sequence_index !=
              layer * plan.geometry.batch_size ||
          worker_layer.sequence_count != plan.geometry.batch_size ||
          !checked_range_end(worker_layer.weight,
                             plan.geometry.active_weight_bytes_per_work_unit,
                             worker_end)) {
        return false;
      }
      if (worker_layer.weight.span_bytes != 0) {
        if (worker_layer.weight.offset_bytes != worker_cursor ||
            worker_end > layer_end) {
          return false;
        }
        worker_cursor = worker_end;
      }
    }
    if (worker_cursor != layer_end) {
      return false;
    }
    weight_cursor = layer_end;
  }
  if (weight_cursor != plan.geometry.active_weight_bytes_per_work_unit) {
    return false;
  }

  constexpr std::array<LlmScenario, kLlmScenarioCount> kScenarios = {
      LlmScenario::WeightsOnly, LlmScenario::KvOnly,
      LlmScenario::Mixed};
  const std::array<size_t, kLlmScenarioCount> expected_totals = {
      plan.geometry.weight_read_bytes_per_work_unit,
      plan.geometry.kv_only_effective_model_payload_bytes_per_work_unit,
      plan.geometry.mixed_effective_model_payload_bytes_per_work_unit};
  for (size_t scenario_index = 0; scenario_index < kScenarios.size();
       ++scenario_index) {
    const LlmPrefillCpuScenarioExecutionPlan& scenario =
        prefill.scenarios[scenario_index];
    if (scenario.scenario != kScenarios[scenario_index] ||
        scenario.identity.empty() ||
        scenario.worker_accounted_bytes_per_work_unit.size() !=
            cpu_plan.effective_workers) {
      return false;
    }
    size_t total_accounted = 0;
    size_t minimum = std::numeric_limits<size_t>::max();
    size_t maximum = 0;
    for (size_t bytes : scenario.worker_accounted_bytes_per_work_unit) {
      if (!checked_add_to(bytes, total_accounted)) {
        return false;
      }
      minimum = std::min(minimum, bytes);
      maximum = std::max(maximum, bytes);
    }
    if (total_accounted != expected_totals[scenario_index] ||
        scenario.minimum_worker_accounted_bytes_per_work_unit != minimum ||
        scenario.maximum_worker_accounted_bytes_per_work_unit != maximum ||
        scenario.worker_accounted_imbalance_bytes_per_work_unit !=
            maximum - minimum) {
      return false;
    }

    for (size_t layer = 0; layer < plan.geometry.layer_count; ++layer) {
      for (size_t batch = 0; batch < plan.geometry.batch_size; ++batch) {
        const size_t row = layer * plan.geometry.batch_size + batch;
        const size_t scenario_base = scenario_index * sequences_per_scenario;
        size_t owned_tokens = 0;
        for (size_t worker_index = 0;
             worker_index < cpu_plan.effective_workers; ++worker_index) {
          const LlmPrefillKvSequenceRangeTemplate& sequence =
              cpu_plan.workers[worker_index]
                  .prefill_sequences[scenario_base + row];
          if (sequence.layer_index != layer ||
              sequence.batch_sequence_index != batch ||
              !equal_ranges(sequence.k_owned, sequence.v_owned)) {
            return false;
          }
          if (scenario_index ==
              static_cast<size_t>(LlmScenario::WeightsOnly)) {
            if (sequence.first_token != 0 ||
                sequence.owned_token_count != 0 ||
                sequence.k_owned.offset_bytes != 0 ||
                sequence.k_owned.span_bytes != 0) {
              return false;
            }
            continue;
          }
          size_t token_end = 0;
          size_t expected_row_offset = 0;
          size_t expected_owned_offset = 0;
          size_t expected_owned_bytes = 0;
          size_t range_end = 0;
          if (sequence.owned_token_count == 0) {
            if (sequence.first_token != 0 ||
                sequence.k_owned.offset_bytes != 0 ||
                sequence.k_owned.span_bytes != 0) {
              return false;
            }
            continue;
          }
          if (!NumericUtils::checked_add(sequence.first_token,
                                         sequence.owned_token_count,
                                         token_end) ||
              token_end > plan.geometry.prefill->prompt_tokens ||
              !NumericUtils::checked_multiply(
                  row, plan.geometry.k_or_v_sequence_visible_bytes,
                  expected_row_offset) ||
              !NumericUtils::checked_multiply(
                  sequence.first_token,
                  plan.geometry.k_or_v_record_bytes_per_layer,
                  expected_owned_offset) ||
              !NumericUtils::checked_add(expected_row_offset,
                                         expected_owned_offset,
                                         expected_owned_offset) ||
              !NumericUtils::checked_multiply(
                  sequence.owned_token_count,
                  plan.geometry.k_or_v_record_bytes_per_layer,
                  expected_owned_bytes) ||
              sequence.k_owned.offset_bytes != expected_owned_offset ||
              sequence.k_owned.span_bytes != expected_owned_bytes ||
              !checked_range_end(sequence.k_owned,
                                 plan.geometry.k_mapping_bytes, range_end) ||
              !checked_add_to(sequence.owned_token_count, owned_tokens)) {
            return false;
          }
          for (size_t previous_worker = 0;
               previous_worker < worker_index; ++previous_worker) {
            const LlmPrefillKvSequenceRangeTemplate& previous =
                cpu_plan.workers[previous_worker]
                    .prefill_sequences[scenario_base + row];
            if (previous.owned_token_count == 0) {
              continue;
            }
            const size_t previous_end =
                previous.first_token + previous.owned_token_count;
            if (sequence.first_token < previous_end &&
                previous.first_token < token_end) {
              return false;
            }
          }
        }
        if (scenario_index !=
                static_cast<size_t>(LlmScenario::WeightsOnly) &&
            owned_tokens != plan.geometry.prefill->prompt_tokens) {
          return false;
        }
      }
    }
  }
  return true;
}

bool validate_paged_prefill_work_plan_layout(
    const LlmMemoryWorkPlan& plan,
    const LlmCpuExecutionPlan& cpu_plan) noexcept {
  if (!plan.valid || !plan.geometry.valid || !plan.memory_budget.valid ||
      plan.phase != LlmPhase::Prefill ||
      plan.kv_layout != LlmKvLayout::Paged ||
      !plan.geometry.prefill.has_value() || !plan.prefill_plan.has_value() ||
      !plan.prefill_plan->valid || !cpu_plan.prefill.has_value() ||
      !cpu_plan.paged.has_value() ||
      !validate_llm_prefill_cpu_execution_evidence(plan) ||
      !paged_layout_matches_model(plan, *cpu_plan.paged) ||
      !cpu_plan.paged->layout.valid ||
      !cpu_plan.paged->table_validation.valid ||
      cpu_plan.paged->permutation.entry_count !=
          cpu_plan.paged->layout.block_table_entries ||
      cpu_plan.paged->block_table() == nullptr ||
      !cpu_plan.paged->block_table_read_only ||
      cpu_plan.paged->block_table_logical_bytes !=
          cpu_plan.paged->layout.memory.block_table_bytes ||
      cpu_plan.paged->block_table_mapping_bytes !=
          plan.memory_budget.request.committed_block_table_mapping_bytes ||
      cpu_plan.paged->layout_identity.empty() ||
      cpu_plan.paged->execution_identity != cpu_plan.prefill->identity ||
      cpu_plan.effective_workers == 0 ||
      cpu_plan.workers.size() != cpu_plan.effective_workers ||
      plan.weight_layers.size() != plan.geometry.layer_count ||
      cpu_plan.layer_descriptors_per_worker != plan.geometry.layer_count) {
    return false;
  }

  const LlmKvLayoutPlan& layout = cpu_plan.paged->layout;
  size_t assignments_per_scenario = 0;
  size_t assignments_per_worker = 0;
  size_t expected_total_layers = 0;
  size_t expected_total_assignments = 0;
  size_t layer_bytes = 0;
  size_t assignment_bytes = 0;
  size_t descriptor_bytes = 0;
  if (!NumericUtils::checked_multiply(
          plan.geometry.layer_count, plan.geometry.batch_size,
          assignments_per_scenario) ||
      assignments_per_scenario !=
          cpu_plan.prefill->sequence_descriptors_per_scenario_per_worker ||
      !NumericUtils::checked_multiply(
          assignments_per_scenario, kLlmScenarioCount,
          assignments_per_worker) ||
      assignments_per_worker != cpu_plan.sequence_descriptors_per_worker ||
      !NumericUtils::checked_multiply(
          cpu_plan.effective_workers, plan.geometry.layer_count,
          expected_total_layers) ||
      !NumericUtils::checked_multiply(
          cpu_plan.effective_workers, assignments_per_worker,
          expected_total_assignments) ||
      expected_total_layers != cpu_plan.total_layer_descriptors ||
      expected_total_assignments != cpu_plan.total_sequence_descriptors ||
      !NumericUtils::checked_multiply(
          expected_total_layers, sizeof(LlmPagedPrefillLayerDescriptor),
          layer_bytes) ||
      !NumericUtils::checked_multiply(
          expected_total_assignments,
          sizeof(LlmPagedPrefillKvAssignmentDescriptor), assignment_bytes) ||
      !NumericUtils::checked_add(
          layer_bytes, assignment_bytes, descriptor_bytes) ||
      descriptor_bytes != cpu_plan.descriptor_bytes ||
      plan.geometry.k_mapping_bytes != layout.memory.k_physical_bytes ||
      plan.geometry.v_mapping_bytes != layout.memory.v_physical_bytes ||
      plan.geometry.kv_blocks_per_sequence != layout.blocks_per_sequence ||
      plan.geometry.kv_block_bytes != layout.block_bytes) {
    return false;
  }

  size_t weight_cursor = 0;
  for (size_t layer = 0; layer < plan.geometry.layer_count; ++layer) {
    const LlmByteRange& layer_range = plan.weight_layers[layer];
    size_t layer_end = 0;
    if (layer_range.span_bytes == 0 ||
        layer_range.offset_bytes != weight_cursor ||
        !checked_range_end(
            layer_range, plan.geometry.active_weight_bytes_per_work_unit,
            layer_end)) {
      return false;
    }
    size_t worker_cursor = layer_range.offset_bytes;
    for (size_t worker_index = 0;
         worker_index < cpu_plan.effective_workers; ++worker_index) {
      const LlmWorkerWorkPlan& worker = cpu_plan.workers[worker_index];
      if (worker.worker_index != worker_index ||
          worker.layers.size() != plan.geometry.layer_count ||
          !worker.sequences.empty() || !worker.paged_assignments.empty() ||
          !worker.prefill_sequences.empty() ||
          worker.paged_prefill_assignments.size() !=
              assignments_per_worker) {
        return false;
      }
      const LlmLayerRangeTemplate& worker_layer = worker.layers[layer];
      size_t worker_end = 0;
      if (worker_layer.layer_index != layer ||
          worker_layer.first_sequence_index !=
              layer * plan.geometry.batch_size ||
          worker_layer.sequence_count != plan.geometry.batch_size ||
          !checked_range_end(
              worker_layer.weight,
              plan.geometry.active_weight_bytes_per_work_unit, worker_end)) {
        return false;
      }
      if (worker_layer.weight.span_bytes != 0) {
        if (worker_layer.weight.offset_bytes != worker_cursor ||
            worker_end > layer_end) {
          return false;
        }
        worker_cursor = worker_end;
      }
    }
    if (worker_cursor != layer_end) {
      return false;
    }
    weight_cursor = layer_end;
  }
  if (weight_cursor != plan.geometry.active_weight_bytes_per_work_unit) {
    return false;
  }

  for (size_t scenario_index = 0; scenario_index < kLlmScenarioCount;
       ++scenario_index) {
    const size_t scenario_base =
        scenario_index * assignments_per_scenario;
    for (size_t layer = 0; layer < plan.geometry.layer_count; ++layer) {
      for (size_t batch = 0; batch < plan.geometry.batch_size; ++batch) {
        const size_t row = layer * plan.geometry.batch_size + batch;
        size_t owned_blocks = 0;
        for (size_t worker_index = 0;
             worker_index < cpu_plan.effective_workers; ++worker_index) {
          const LlmPagedPrefillKvAssignmentTemplate& assignment =
              cpu_plan.workers[worker_index]
                  .paged_prefill_assignments[scenario_base + row];
          if (assignment.layer_index != layer ||
              assignment.batch_sequence_index != batch ||
              (assignment.block_count == 0 &&
               assignment.first_logical_block != 0)) {
            return false;
          }
          if (assignment.block_count == 0) {
            continue;
          }
          size_t assignment_end = 0;
          if (scenario_index ==
                  static_cast<size_t>(LlmScenario::WeightsOnly) ||
              !NumericUtils::checked_add(
                  assignment.first_logical_block, assignment.block_count,
                  assignment_end) ||
              assignment_end > layout.blocks_per_sequence ||
              !checked_add_to(assignment.block_count, owned_blocks)) {
            return false;
          }
          for (size_t previous_worker = 0;
               previous_worker < worker_index; ++previous_worker) {
            const LlmPagedPrefillKvAssignmentTemplate& previous =
                cpu_plan.workers[previous_worker]
                    .paged_prefill_assignments[scenario_base + row];
            if (previous.block_count == 0) {
              continue;
            }
            const size_t previous_end =
                previous.first_logical_block + previous.block_count;
            if (assignment.first_logical_block < previous_end &&
                previous.first_logical_block < assignment_end) {
              return false;
            }
          }
        }
        if (scenario_index ==
                static_cast<size_t>(LlmScenario::WeightsOnly)
                ? owned_blocks != 0
                : owned_blocks != layout.blocks_per_sequence) {
          return false;
        }
      }
    }
  }
  return true;
}

LlmByteRange intersect_ranges(const LlmByteRange& lhs, const LlmByteRange& rhs) noexcept {
  size_t lhs_end = 0;
  size_t rhs_end = 0;
  if (lhs.span_bytes == 0 || rhs.span_bytes == 0 ||
      !NumericUtils::checked_add(lhs.offset_bytes, lhs.span_bytes, lhs_end) ||
      !NumericUtils::checked_add(rhs.offset_bytes, rhs.span_bytes, rhs_end)) {
    return {};
  }
  const size_t start = std::max(lhs.offset_bytes, rhs.offset_bytes);
  const size_t end = std::min(lhs_end, rhs_end);
  return end > start ? LlmByteRange{start, end - start} : LlmByteRange{};
}

/**
 * Validate the complete pointer-free layout before any mapping is created.
 *
 * This deliberately rechecks exact worker unions rather than trusting `valid`:
 * later descriptor materialization must never turn a corrupted offset into an
 * out-of-bounds pointer. The validation is allocation-free and does not mutate
 * the finalized work plan.
 */
bool validate_work_plan_layout(const LlmMemoryWorkPlan& plan) noexcept {
  const LlmCpuExecutionPlan* const cpu_plan =
      get_llm_cpu_execution_plan(plan);
  if (cpu_plan != nullptr && plan.phase == LlmPhase::Prefill) {
    return plan.kv_layout == LlmKvLayout::Paged
               ? validate_paged_prefill_work_plan_layout(plan, *cpu_plan)
               : validate_prefill_work_plan_layout(plan, *cpu_plan);
  }
  if (cpu_plan != nullptr && plan.kv_layout == LlmKvLayout::Paged) {
    return validate_paged_work_plan_layout(plan, *cpu_plan);
  }
  size_t weight_and_k_bytes = 0;
  size_t expected_total_data_bytes = 0;
  if (cpu_plan == nullptr || !plan.valid || !plan.geometry.valid ||
      !plan.memory_budget.valid || cpu_plan->effective_workers == 0 ||
      plan.geometry.layer_count == 0 || plan.geometry.batch_size == 0 || plan.geometry.k_mapping_bytes == 0 ||
      plan.geometry.v_mapping_bytes == 0 || plan.geometry.k_mapping_bytes != plan.geometry.v_mapping_bytes ||
      !NumericUtils::checked_add(plan.geometry.active_weight_bytes_per_work_unit, plan.geometry.k_mapping_bytes,
                                 weight_and_k_bytes) ||
      !NumericUtils::checked_add(weight_and_k_bytes, plan.geometry.v_mapping_bytes, expected_total_data_bytes) ||
      expected_total_data_bytes != plan.geometry.total_data_mapping_bytes ||
      cpu_plan->workers.size() != cpu_plan->effective_workers ||
      plan.weight_layers.size() != plan.geometry.layer_count ||
      cpu_plan->layer_descriptors_per_worker != plan.geometry.layer_count) {
    return false;
  }

  size_t expected_sequences_per_worker = 0;
  size_t expected_total_layers = 0;
  size_t expected_total_sequences = 0;
  size_t layer_descriptor_bytes = 0;
  size_t sequence_descriptor_bytes = 0;
  size_t expected_descriptor_bytes = 0;
  if (!NumericUtils::checked_multiply(plan.geometry.layer_count, plan.geometry.batch_size,
                                      expected_sequences_per_worker) ||
      expected_sequences_per_worker != cpu_plan->sequence_descriptors_per_worker ||
      !NumericUtils::checked_multiply(cpu_plan->effective_workers,
                                      cpu_plan->layer_descriptors_per_worker,
                                      expected_total_layers) ||
      !NumericUtils::checked_multiply(cpu_plan->effective_workers,
                                      cpu_plan->sequence_descriptors_per_worker,
                                      expected_total_sequences) ||
      expected_total_layers != cpu_plan->total_layer_descriptors ||
      expected_total_sequences != cpu_plan->total_sequence_descriptors ||
      !NumericUtils::checked_multiply(expected_total_layers, sizeof(LlmLayerDescriptor), layer_descriptor_bytes) ||
      !NumericUtils::checked_multiply(expected_total_sequences, sizeof(LlmKvSequenceDescriptor),
                                      sequence_descriptor_bytes) ||
      !NumericUtils::checked_add(layer_descriptor_bytes, sequence_descriptor_bytes, expected_descriptor_bytes) ||
      expected_descriptor_bytes != cpu_plan->descriptor_bytes) {
    return false;
  }

  size_t weight_mapping_cursor = 0;
  for (size_t layer = 0; layer < plan.geometry.layer_count; ++layer) {
    const LlmByteRange& layer_range = plan.weight_layers[layer];
    size_t layer_end = 0;
    if (layer_range.span_bytes == 0 || layer_range.offset_bytes != weight_mapping_cursor ||
        !checked_range_end(layer_range, plan.geometry.active_weight_bytes_per_work_unit, layer_end)) {
      return false;
    }

    size_t worker_cursor = layer_range.offset_bytes;
    const size_t expected_first_sequence = layer * plan.geometry.batch_size;
    for (size_t worker_index = 0;
         worker_index < cpu_plan->effective_workers; ++worker_index) {
      const LlmWorkerWorkPlan& worker = cpu_plan->workers[worker_index];
      if (worker.worker_index != worker_index ||
          worker.layers.size() != cpu_plan->layer_descriptors_per_worker ||
          worker.sequences.size() !=
              cpu_plan->sequence_descriptors_per_worker) {
        return false;
      }
      const LlmLayerRangeTemplate& worker_layer = worker.layers[layer];
      if (worker_layer.first_sequence_index != expected_first_sequence ||
          worker_layer.sequence_count != plan.geometry.batch_size || worker_layer.layer_index != layer) {
        return false;
      }
      size_t worker_end = 0;
      if (!checked_range_end(worker_layer.weight, plan.geometry.active_weight_bytes_per_work_unit, worker_end)) {
        return false;
      }
      if (worker_layer.weight.span_bytes != 0) {
        if (worker_layer.weight.offset_bytes != worker_cursor || worker_end > layer_end) {
          return false;
        }
        worker_cursor = worker_end;
      }
    }
    if (worker_cursor != layer_end) {
      return false;
    }
    weight_mapping_cursor = layer_end;
  }
  if (weight_mapping_cursor != plan.geometry.active_weight_bytes_per_work_unit) {
    return false;
  }

  for (size_t layer = 0; layer < plan.geometry.layer_count; ++layer) {
    for (size_t batch = 0; batch < plan.geometry.batch_size; ++batch) {
      const size_t sequence_index = layer * plan.geometry.batch_size + batch;
      size_t visible_offset = 0;
      size_t visible_end = 0;
      size_t append_token_offset = 0;
      size_t append_offset = 0;
      if (!plan.geometry.decode.has_value() ||
          !NumericUtils::checked_multiply(sequence_index, plan.geometry.k_or_v_sequence_visible_bytes,
                                          visible_offset) ||
          !NumericUtils::checked_add(visible_offset, plan.geometry.k_or_v_sequence_visible_bytes, visible_end) ||
          visible_end > plan.geometry.k_mapping_bytes ||
          !NumericUtils::checked_multiply(
              plan.geometry.decode->visible_context_tokens - 1,
                                          plan.geometry.k_or_v_record_bytes_per_layer, append_token_offset) ||
          !NumericUtils::checked_add(visible_offset, append_token_offset, append_offset)) {
        return false;
      }
      const LlmByteRange append_record{append_offset, plan.geometry.k_or_v_record_bytes_per_layer};

      size_t worker_cursor = visible_offset;
      for (size_t worker_index = 0;
           worker_index < cpu_plan->effective_workers; ++worker_index) {
        const LlmKvSequenceRangeTemplate& sequence =
            cpu_plan->workers[worker_index].sequences[sequence_index];
        if (sequence.layer_index != layer || sequence.batch_sequence_index != batch ||
            !equal_ranges(sequence.k_visible, sequence.v_visible) ||
            !equal_ranges(sequence.k_append, sequence.v_append)) {
          return false;
        }

        size_t visible_worker_end = 0;
        size_t v_visible_worker_end = 0;
        if (!checked_range_end(sequence.k_visible, plan.geometry.k_mapping_bytes, visible_worker_end) ||
            !checked_range_end(sequence.v_visible, plan.geometry.v_mapping_bytes, v_visible_worker_end) ||
            visible_worker_end != v_visible_worker_end) {
          return false;
        }
        if (sequence.k_visible.span_bytes != 0) {
          if (sequence.k_visible.offset_bytes != worker_cursor || visible_worker_end > visible_end) {
            return false;
          }
          worker_cursor = visible_worker_end;
        }

        const LlmByteRange expected_append = intersect_ranges(sequence.k_visible, append_record);
        if (!equal_ranges(sequence.k_append, expected_append)) {
          return false;
        }
        if (expected_append.span_bytes == 0) {
          if (sequence.append_record_byte_offset != 0) {
            return false;
          }
        } else if (sequence.append_record_byte_offset != expected_append.offset_bytes - append_offset) {
          return false;
        }
      }
      if (worker_cursor != visible_end) {
        return false;
      }
    }
  }
  return true;
}

bool checked_add_to(size_t value, size_t& total) noexcept {
  size_t updated = 0;
  if (!NumericUtils::checked_add(total, value, updated)) {
    return false;
  }
  total = updated;
  return true;
}

bool buffer_output_is_empty(const LlmBufferSet& output) noexcept {
  return output.weight == nullptr && output.k == nullptr && output.v == nullptr;
}

bool resource_output_is_empty(const LlmExecutionResources& output) noexcept {
  return !output.valid && output.model_plan_identity.empty() && buffer_output_is_empty(output.buffers) &&
         output.layer_descriptors == nullptr && output.sequence_descriptors == nullptr &&
         output.paged_layer_descriptors == nullptr &&
         output.paged_assignment_descriptors == nullptr &&
         output.prefill_layer_descriptors == nullptr &&
         output.prefill_sequence_descriptors == nullptr &&
         output.paged_prefill_layer_descriptors == nullptr &&
         output.paged_prefill_assignment_descriptors == nullptr &&
         output.weight_references == nullptr && output.k_references == nullptr && output.v_references == nullptr &&
         output.paged_k_block_references == nullptr &&
         output.paged_v_block_references == nullptr &&
         output.block_table == nullptr && output.block_table_entries == 0 &&
         output.paged_block_reference_count == 0 &&
         output.worker_count == 0 && output.layer_descriptors_per_worker == 0 &&
         output.sequence_descriptors_per_worker == 0 && output.total_layer_descriptors == 0 &&
         output.total_sequence_descriptors == 0;
}

bool calculate_budget_with_executor_auxiliary(const LlmMemoryWorkPlan& plan,
                                              const LlmExecutorAuxiliaryEstimate& auxiliary,
                                              LlmMemoryBudget& budget) noexcept {
  const LlmCpuExecutionPlan* const cpu_plan =
      get_llm_cpu_execution_plan(plan);
  if (cpu_plan == nullptr || !auxiliary.valid) {
    return false;
  }
  // Admission is evidence, not a permission to silently expand the plan after
  // it was frozen. A caller must rebuild the work plan with at least these
  // executor-owned bytes before any mapping can be attempted.
  if (plan.memory_budget.request.checksum_auxiliary_bytes < auxiliary.checksum_auxiliary_bytes ||
      plan.memory_budget.request.orchestration_auxiliary_bytes < auxiliary.orchestration_auxiliary_bytes) {
    budget = plan.memory_budget;
    budget.valid = false;
    budget.reason_code = LlmWorkPlanReason::MEMORY_BUDGET_EXCEEDED;
    return false;
  }
  const size_t checksum_bytes = plan.memory_budget.request.checksum_auxiliary_bytes;
  const size_t orchestration_bytes = plan.memory_budget.request.orchestration_auxiliary_bytes;
  const LlmMemoryBudgetRequest request =
      build_llm_memory_budget_request(
          plan.geometry, cpu_plan->descriptor_bytes,
          cpu_plan->planner_storage_bytes, checksum_bytes,
          orchestration_bytes,
          plan.memory_budget.request.mapping_granularity_bytes,
          cpu_plan->paged.has_value()
              ? cpu_plan->paged->layout.memory.block_table_bytes
              : 0,
          cpu_plan->paged.has_value()
              ? cpu_plan->paged->layout.memory.validation_bitset_bytes
              : 0);
  budget = evaluate_llm_memory_budget(request, plan.memory_budget.available_memory_bytes);
  return budget.valid;
}

uint64_t low_byte_mask(size_t byte_count) noexcept {
  return byte_count >= sizeof(uint64_t) ? std::numeric_limits<uint64_t>::max() : (uint64_t{1} << (byte_count * 8U)) - 1;
}

/** Generate at most eight pattern bytes beginning at an arbitrary byte. */
uint64_t pattern_byte_word(uint64_t seed, size_t absolute_byte_offset, size_t byte_count) noexcept {
  const uint64_t word_index = static_cast<uint64_t>(absolute_byte_offset / sizeof(uint64_t));
  const size_t byte_in_word = absolute_byte_offset % sizeof(uint64_t);
  uint64_t value = llm_buffer_pattern_word(seed, word_index) >> (byte_in_word * 8U);
  const size_t first_word_bytes = sizeof(uint64_t) - byte_in_word;
  if (byte_count > first_word_bytes) {
    value |= llm_buffer_pattern_word(seed, word_index + 1) << (first_word_bytes * 8U);
  }
  return value & low_byte_mask(byte_count);
}

/**
 * Initialize one exact finalized span and accumulate its read reference.
 *
 * Local checksum-word parity starts at zero even when the mapping offset is
 * unaligned. `memcpy` keeps every 1-7 byte tail exact and avoids undefined
 * unaligned integer stores.
 */
bool initialize_span(uint8_t* mapping, size_t mapping_bytes, uint64_t seed, size_t offset, size_t span_bytes,
                     LlmStaticSpanReference& reference) noexcept {
  reference = {};
  if (span_bytes == 0) {
    return offset == 0;
  }
  size_t end = 0;
  if (mapping == nullptr || !NumericUtils::checked_add(offset, span_bytes, end) || end > mapping_bytes) {
    return false;
  }

  reference.span_bytes = static_cast<uint64_t>(span_bytes);
  size_t local_offset = 0;
  size_t word_index = 0;
  while (local_offset < span_bytes) {
    const size_t word_bytes = std::min(sizeof(uint64_t), span_bytes - local_offset);
    const uint64_t word = pattern_byte_word(seed, offset + local_offset, word_bytes);
    std::memcpy(mapping + offset + local_offset, &word, word_bytes);
    if ((word_index & 1U) == 0) {
      reference.span_even += word;
    } else {
      reference.span_odd += word;
    }
    local_offset += word_bytes;
    ++word_index;
  }
  return true;
}

uint64_t paged_pattern_byte_word(uint64_t seed, size_t layer_index,
                                 uint32_t physical_block_id,
                                 size_t block_byte_offset,
                                 size_t byte_count) noexcept {
  const uint64_t word_index = block_byte_offset / sizeof(uint64_t);
  const size_t byte_in_word = block_byte_offset % sizeof(uint64_t);
  uint64_t value = llm_paged_buffer_pattern_word(
                       seed, layer_index, physical_block_id, word_index) >>
                   (byte_in_word * 8U);
  const size_t first_word_bytes = sizeof(uint64_t) - byte_in_word;
  if (byte_count > first_word_bytes) {
    value |= llm_paged_buffer_pattern_word(
                 seed, layer_index, physical_block_id, word_index + 1) <<
             (first_word_bytes * 8U);
  }
  return value & low_byte_mask(byte_count);
}

bool initialize_paged_block(
    uint8_t* mapping, size_t mapping_bytes, uint64_t seed,
    size_t layer_index, size_t physical_blocks_per_layer,
    uint32_t physical_block_id, size_t block_bytes, size_t valid_bytes,
    LlmStaticSpanReference& reference) noexcept {
  reference = {};
  size_t global_block_index = 0;
  size_t block_offset = 0;
  size_t block_end = 0;
  if (mapping == nullptr || block_bytes == 0 || valid_bytes == 0 ||
      valid_bytes > block_bytes || physical_blocks_per_layer == 0 ||
      physical_block_id >= physical_blocks_per_layer ||
      !NumericUtils::checked_multiply(
          layer_index, physical_blocks_per_layer, global_block_index) ||
      !NumericUtils::checked_add(global_block_index,
                                 static_cast<size_t>(physical_block_id),
                                 global_block_index) ||
      !NumericUtils::checked_multiply(global_block_index, block_bytes,
                                      block_offset) ||
      !NumericUtils::checked_add(block_offset, block_bytes, block_end) ||
      block_end > mapping_bytes) {
    return false;
  }
  reference.span_bytes = valid_bytes;
  size_t local_offset = 0;
  size_t reference_word_index = 0;
  while (local_offset < block_bytes) {
    const size_t write_bytes =
        std::min(sizeof(uint64_t), block_bytes - local_offset);
    const uint64_t word = paged_pattern_byte_word(
        seed, layer_index, physical_block_id, local_offset, write_bytes);
    std::memcpy(mapping + block_offset + local_offset, &word, write_bytes);
    if (local_offset < valid_bytes) {
      const size_t reference_bytes =
          std::min(write_bytes, valid_bytes - local_offset);
      const uint64_t reference_word = word & low_byte_mask(reference_bytes);
      if ((reference_word_index & 1U) == 0) {
        reference.span_even += reference_word;
      } else {
        reference.span_odd += reference_word;
      }
      ++reference_word_index;
    }
    local_offset += write_bytes;
  }
  return true;
}

bool add_initialized_span(const LlmStaticSpanReference& reference, size_t& initialized_bytes,
                          size_t& non_empty_spans) noexcept {
  if (reference.span_bytes == 0) {
    return true;
  }
  if (reference.span_bytes > std::numeric_limits<size_t>::max() ||
      !checked_add_to(static_cast<size_t>(reference.span_bytes), initialized_bytes) ||
      non_empty_spans == std::numeric_limits<size_t>::max()) {
    return false;
  }
  ++non_empty_spans;
  return true;
}

bool materialize_descriptors(const LlmMemoryWorkPlan& plan, LlmExecutionResources& resources) noexcept {
  const LlmCpuExecutionPlan* const cpu_plan =
      get_llm_cpu_execution_plan(plan);
  uint8_t* const weight = static_cast<uint8_t*>(resources.buffers.weight.get());
  uint8_t* const k = static_cast<uint8_t*>(resources.buffers.k.get());
  uint8_t* const v = static_cast<uint8_t*>(resources.buffers.v.get());
  if (cpu_plan == nullptr || weight == nullptr || k == nullptr ||
      v == nullptr) {
    return false;
  }

  if (plan.phase == LlmPhase::Prefill &&
      plan.kv_layout == LlmKvLayout::Paged) {
    if (!cpu_plan->prefill.has_value() || !cpu_plan->paged.has_value() ||
        !plan.geometry.prefill.has_value() ||
        resources.block_table == nullptr ||
        resources.paged_prefill_layer_descriptors == nullptr ||
        resources.paged_prefill_assignment_descriptors == nullptr) {
      return false;
    }
    const LlmKvLayoutPlan& layout = cpu_plan->paged->layout;
    const size_t layer_pool_bytes =
        layout.physical_blocks_per_layer * layout.block_bytes;
    for (size_t worker_index = 0;
         worker_index < cpu_plan->effective_workers; ++worker_index) {
      const LlmWorkerWorkPlan& worker = cpu_plan->workers[worker_index];
      const size_t layer_base =
          worker_index * cpu_plan->layer_descriptors_per_worker;
      const size_t assignment_base =
          worker_index * cpu_plan->sequence_descriptors_per_worker;
      for (size_t layer = 0;
           layer < cpu_plan->layer_descriptors_per_worker; ++layer) {
        const LlmLayerRangeTemplate& source = worker.layers[layer];
        LlmPagedPrefillLayerDescriptor& destination =
            resources.paged_prefill_layer_descriptors[layer_base + layer];
        destination.weight_ptr =
            source.weight.span_bytes == 0
                ? nullptr
                : weight + source.weight.offset_bytes;
        destination.weight_bytes = source.weight.span_bytes;
        destination.first_assignment_index = source.first_sequence_index;
        destination.assignment_count = source.sequence_count;
        destination.layer_index = source.layer_index;
        destination.reserved_zero = 0;
      }
      for (size_t assignment_index = 0;
           assignment_index < cpu_plan->sequence_descriptors_per_worker;
           ++assignment_index) {
        const LlmPagedPrefillKvAssignmentTemplate& source =
            worker.paged_prefill_assignments[assignment_index];
        LlmPagedPrefillKvAssignmentDescriptor& destination =
            resources.paged_prefill_assignment_descriptors[
                assignment_base + assignment_index];
        destination.block_table_row =
            resources.block_table +
            source.batch_sequence_index * layout.blocks_per_sequence;
        destination.k_layer_pool =
            k + source.layer_index * layer_pool_bytes;
        destination.v_layer_pool =
            v + source.layer_index * layer_pool_bytes;
        destination.first_logical_block = source.first_logical_block;
        destination.owned_block_count = source.block_count;
        destination.blocks_per_sequence = layout.blocks_per_sequence;
        destination.block_tokens = layout.kv_block_tokens;
        destination.block_bytes = layout.block_bytes;
        destination.last_block_valid_bytes = layout.last_block_valid_bytes;
        destination.prompt_tokens = plan.geometry.prefill->prompt_tokens;
        destination.attention_query_tile_tokens =
            plan.geometry.prefill->attention_query_tile_tokens;
        destination.record_bytes =
            plan.geometry.k_or_v_record_bytes_per_layer;
        destination.layer_index = source.layer_index;
        destination.batch_sequence_index =
            source.batch_sequence_index;
      }
    }
    return true;
  }

  if (plan.phase == LlmPhase::Prefill &&
      plan.kv_layout == LlmKvLayout::Contiguous) {
    if (!cpu_plan->prefill.has_value() ||
        !plan.geometry.prefill.has_value() ||
        resources.prefill_layer_descriptors == nullptr ||
        resources.prefill_sequence_descriptors == nullptr) {
      return false;
    }
    for (size_t worker_index = 0;
         worker_index < cpu_plan->effective_workers; ++worker_index) {
      const LlmWorkerWorkPlan& worker = cpu_plan->workers[worker_index];
      const size_t layer_base =
          worker_index * cpu_plan->layer_descriptors_per_worker;
      const size_t sequence_base =
          worker_index * cpu_plan->sequence_descriptors_per_worker;
      for (size_t layer = 0;
           layer < cpu_plan->layer_descriptors_per_worker; ++layer) {
        const LlmLayerRangeTemplate& source = worker.layers[layer];
        LlmPrefillLayerDescriptor& destination =
            resources.prefill_layer_descriptors[layer_base + layer];
        destination.weight_ptr =
            source.weight.span_bytes == 0
                ? nullptr
                : weight + source.weight.offset_bytes;
        destination.weight_bytes = source.weight.span_bytes;
        destination.first_sequence_index = source.first_sequence_index;
        destination.sequence_count = source.sequence_count;
        destination.layer_index = source.layer_index;
        destination.reserved_zero = 0;
      }
      for (size_t sequence_index = 0;
           sequence_index < cpu_plan->sequence_descriptors_per_worker;
           ++sequence_index) {
        const LlmPrefillKvSequenceRangeTemplate& source =
            worker.prefill_sequences[sequence_index];
        LlmPrefillKvSequenceDescriptor& destination =
            resources.prefill_sequence_descriptors[sequence_base +
                                                   sequence_index];
        destination.k_owned_ptr =
            source.k_owned.span_bytes == 0
                ? nullptr
                : k + source.k_owned.offset_bytes;
        destination.v_owned_ptr =
            source.v_owned.span_bytes == 0
                ? nullptr
                : v + source.v_owned.offset_bytes;
        destination.first_token = source.first_token;
        destination.owned_token_count = source.owned_token_count;
        destination.prompt_tokens =
            plan.geometry.prefill->prompt_tokens;
        destination.attention_query_tile_tokens =
            plan.geometry.prefill->attention_query_tile_tokens;
        destination.record_bytes =
            plan.geometry.k_or_v_record_bytes_per_layer;
        destination.layer_index = source.layer_index;
        destination.batch_sequence_index = source.batch_sequence_index;
        destination.reserved_zero = 0;
      }
    }
    return true;
  }

  if (plan.kv_layout == LlmKvLayout::Paged) {
    if (!cpu_plan->paged.has_value() || resources.block_table == nullptr ||
        resources.paged_layer_descriptors == nullptr ||
        resources.paged_assignment_descriptors == nullptr) {
      return false;
    }
    const LlmKvLayoutPlan& layout = cpu_plan->paged->layout;
    const size_t layer_pool_bytes =
        layout.physical_blocks_per_layer * layout.block_bytes;
    for (size_t worker_index = 0;
         worker_index < cpu_plan->effective_workers; ++worker_index) {
      const LlmWorkerWorkPlan& worker = cpu_plan->workers[worker_index];
      const size_t layer_base =
          worker_index * cpu_plan->layer_descriptors_per_worker;
      const size_t assignment_base =
          worker_index * cpu_plan->sequence_descriptors_per_worker;
      for (size_t layer = 0;
           layer < cpu_plan->layer_descriptors_per_worker; ++layer) {
        const LlmLayerRangeTemplate& source = worker.layers[layer];
        LlmPagedLayerDescriptor& destination =
            resources.paged_layer_descriptors[layer_base + layer];
        destination.weight_ptr =
            source.weight.span_bytes == 0
                ? nullptr
                : weight + source.weight.offset_bytes;
        destination.weight_bytes = source.weight.span_bytes;
        destination.first_assignment_index = source.first_sequence_index;
        destination.assignment_count = source.sequence_count;
        destination.layer_index = source.layer_index;
        destination.reserved_zero = 0;
      }
      for (size_t assignment_index = 0;
           assignment_index < cpu_plan->sequence_descriptors_per_worker;
           ++assignment_index) {
        const LlmPagedKvAssignmentTemplate& source =
            worker.paged_assignments[assignment_index];
        LlmPagedKvAssignmentDescriptor& destination =
            resources.paged_assignment_descriptors[assignment_base +
                                                   assignment_index];
        destination.block_table_row =
            resources.block_table +
            source.batch_sequence_index * layout.blocks_per_sequence;
        destination.k_layer_pool =
            k + source.layer_index * layer_pool_bytes;
        destination.v_layer_pool =
            v + source.layer_index * layer_pool_bytes;
        destination.first_logical_block = source.first_logical_block;
        destination.owned_block_count = source.block_count;
        destination.blocks_per_sequence = layout.blocks_per_sequence;
        destination.block_bytes = layout.block_bytes;
        destination.last_block_valid_bytes = layout.last_block_valid_bytes;
        destination.decode_append_offset =
            layout.decode_append_offset_in_last_block;
        destination.append_record_bytes =
            layout.k_or_v_record_bytes_per_layer;
        destination.layer_index = source.layer_index;
        destination.batch_sequence_index =
            source.batch_sequence_index;
      }
    }
    return true;
  }

  for (size_t worker_index = 0;
       worker_index < cpu_plan->effective_workers; ++worker_index) {
    const LlmWorkerWorkPlan& worker = cpu_plan->workers[worker_index];
    const size_t layer_base =
        worker_index * cpu_plan->layer_descriptors_per_worker;
    const size_t sequence_base =
        worker_index * cpu_plan->sequence_descriptors_per_worker;
    for (size_t layer = 0;
         layer < cpu_plan->layer_descriptors_per_worker; ++layer) {
      const LlmLayerRangeTemplate& source = worker.layers[layer];
      LlmLayerDescriptor& destination = resources.layer_descriptors[layer_base + layer];
      destination.weight_ptr = source.weight.span_bytes == 0 ? nullptr : weight + source.weight.offset_bytes;
      destination.weight_bytes = source.weight.span_bytes;
      destination.first_sequence_index = source.first_sequence_index;
      destination.sequence_count = source.sequence_count;
      destination.layer_index = source.layer_index;
      destination.reserved_zero = 0;
    }
    for (size_t sequence_index = 0;
         sequence_index < cpu_plan->sequence_descriptors_per_worker;
         ++sequence_index) {
      const LlmKvSequenceRangeTemplate& source = worker.sequences[sequence_index];
      LlmKvSequenceDescriptor& destination = resources.sequence_descriptors[sequence_base + sequence_index];
      destination.k_visible_ptr = source.k_visible.span_bytes == 0 ? nullptr : k + source.k_visible.offset_bytes;
      destination.k_visible_bytes = source.k_visible.span_bytes;
      destination.v_visible_ptr = source.v_visible.span_bytes == 0 ? nullptr : v + source.v_visible.offset_bytes;
      destination.v_visible_bytes = source.v_visible.span_bytes;
      destination.k_append_ptr = source.k_append.span_bytes == 0 ? nullptr : k + source.k_append.offset_bytes;
      destination.k_append_bytes = source.k_append.span_bytes;
      destination.v_append_ptr = source.v_append.span_bytes == 0 ? nullptr : v + source.v_append.offset_bytes;
      destination.v_append_bytes = source.v_append.span_bytes;
      destination.batch_sequence_index = source.batch_sequence_index;
      destination.append_record_byte_offset = source.k_append.span_bytes == 0 ? 0 : source.append_record_byte_offset;
    }
  }
  return true;
}

bool initialize_resources(const LlmMemoryWorkPlan& plan, LlmExecutionResources& resources,
                          LlmInitializationEvidence& evidence) noexcept {
  const LlmCpuExecutionPlan* const cpu_plan =
      get_llm_cpu_execution_plan(plan);
  uint8_t* const weight = static_cast<uint8_t*>(resources.buffers.weight.get());
  uint8_t* const k = static_cast<uint8_t*>(resources.buffers.k.get());
  uint8_t* const v = static_cast<uint8_t*>(resources.buffers.v.get());
  if (cpu_plan == nullptr) {
    return false;
  }

  for (size_t worker_index = 0;
       worker_index < cpu_plan->effective_workers; ++worker_index) {
    const LlmWorkerWorkPlan& worker = cpu_plan->workers[worker_index];
    const size_t layer_base =
        worker_index * cpu_plan->layer_descriptors_per_worker;
    const size_t sequence_base =
        worker_index * cpu_plan->sequence_descriptors_per_worker;
    for (size_t layer = 0;
         layer < cpu_plan->layer_descriptors_per_worker; ++layer) {
      LlmStaticSpanReference& reference = resources.weight_references[layer_base + layer];
      const LlmByteRange& range = worker.layers[layer].weight;
      if (!initialize_span(weight, plan.geometry.active_weight_bytes_per_work_unit, plan.weight_buffer_seed,
                           range.offset_bytes, range.span_bytes, reference) ||
          !add_initialized_span(reference, evidence.weight_bytes, evidence.non_empty_weight_spans)) {
        return false;
      }
    }
    if (plan.kv_layout == LlmKvLayout::Paged ||
        plan.phase == LlmPhase::Prefill) {
      continue;
    }
    for (size_t sequence_index = 0;
         sequence_index < cpu_plan->sequence_descriptors_per_worker;
         ++sequence_index) {
      const LlmKvSequenceRangeTemplate& sequence = worker.sequences[sequence_index];
      LlmStaticSpanReference& k_reference = resources.k_references[sequence_base + sequence_index];
      LlmStaticSpanReference& v_reference = resources.v_references[sequence_base + sequence_index];
      if (!initialize_span(k, plan.geometry.k_mapping_bytes, plan.k_buffer_seed, sequence.k_visible.offset_bytes,
                           sequence.k_visible.span_bytes, k_reference) ||
          !add_initialized_span(k_reference, evidence.k_bytes, evidence.non_empty_k_spans) ||
          !initialize_span(v, plan.geometry.v_mapping_bytes, plan.v_buffer_seed, sequence.v_visible.offset_bytes,
                           sequence.v_visible.span_bytes, v_reference) ||
          !add_initialized_span(v_reference, evidence.v_bytes, evidence.non_empty_v_spans)) {
        return false;
      }
    }
  }

  if (plan.phase == LlmPhase::Prefill &&
      plan.kv_layout == LlmKvLayout::Contiguous) {
    LlmStaticSpanReference k_reference;
    LlmStaticSpanReference v_reference;
    if (!initialize_span(k, plan.geometry.k_mapping_bytes,
                         plan.k_buffer_seed, 0,
                         plan.geometry.k_mapping_bytes, k_reference) ||
        !add_initialized_span(k_reference, evidence.k_bytes,
                              evidence.non_empty_k_spans) ||
        !initialize_span(v, plan.geometry.v_mapping_bytes,
                         plan.v_buffer_seed, 0,
                         plan.geometry.v_mapping_bytes, v_reference) ||
        !add_initialized_span(v_reference, evidence.v_bytes,
                              evidence.non_empty_v_spans)) {
      return false;
    }
  }

  if (plan.kv_layout == LlmKvLayout::Paged) {
    if (!cpu_plan->paged.has_value() || resources.block_table == nullptr ||
        resources.paged_k_block_references == nullptr ||
        resources.paged_v_block_references == nullptr) {
      return false;
    }
    const LlmKvLayoutPlan& layout = cpu_plan->paged->layout;
    for (size_t layer = 0; layer < layout.layer_count; ++layer) {
      for (size_t batch = 0; batch < layout.batch_size; ++batch) {
        const size_t table_row = batch * layout.blocks_per_sequence;
        for (size_t logical_block = 0;
             logical_block < layout.blocks_per_sequence; ++logical_block) {
          const uint32_t physical_id =
              resources.block_table[table_row + logical_block];
          if (physical_id >= layout.physical_blocks_per_layer) {
            return false;
          }
          size_t reference_index = 0;
          if (!NumericUtils::checked_multiply(
                  layer, layout.physical_blocks_per_layer,
                  reference_index) ||
              !NumericUtils::checked_add(
                  reference_index, static_cast<size_t>(physical_id),
                  reference_index) ||
              reference_index >= resources.paged_block_reference_count) {
            return false;
          }
          const size_t valid_bytes =
              logical_block + 1 == layout.blocks_per_sequence
                  ? layout.last_block_valid_bytes
                  : layout.block_bytes;
          LlmStaticSpanReference& k_reference =
              resources.paged_k_block_references[reference_index];
          LlmStaticSpanReference& v_reference =
              resources.paged_v_block_references[reference_index];
          if (!initialize_paged_block(
                  k, plan.geometry.k_mapping_bytes, plan.k_buffer_seed,
                  layer, layout.physical_blocks_per_layer, physical_id,
                  layout.block_bytes, valid_bytes, k_reference) ||
              !initialize_paged_block(
                  v, plan.geometry.v_mapping_bytes, plan.v_buffer_seed,
                  layer, layout.physical_blocks_per_layer, physical_id,
                  layout.block_bytes, valid_bytes, v_reference) ||
              !checked_add_to(layout.block_bytes, evidence.k_bytes) ||
              !checked_add_to(layout.block_bytes, evidence.v_bytes)) {
            return false;
          }
          ++evidence.non_empty_k_spans;
          ++evidence.non_empty_v_spans;
        }
      }
    }
    evidence.block_table_logical_bytes =
        cpu_plan->paged->block_table_logical_bytes;
    evidence.block_table_mapping_bytes =
        cpu_plan->paged->block_table_mapping_bytes;
    evidence.block_table_read_only = cpu_plan->paged->block_table_read_only;
    evidence.k_layout_padding_bytes =
        layout.memory.k_layout_padding_bytes;
    evidence.v_layout_padding_bytes =
        layout.memory.v_layout_padding_bytes;
  }

  if (evidence.weight_bytes != plan.geometry.active_weight_bytes_per_work_unit ||
      evidence.k_bytes != plan.geometry.k_mapping_bytes || evidence.v_bytes != plan.geometry.v_mapping_bytes ||
      !NumericUtils::checked_add(evidence.weight_bytes, evidence.k_bytes, evidence.total_bytes) ||
      !checked_add_to(evidence.v_bytes, evidence.total_bytes) ||
      evidence.total_bytes != plan.geometry.total_data_mapping_bytes) {
    return false;
  }
  evidence.complete = true;
  return true;
}

bool materialized_resources_match_plan(const LlmMemoryWorkPlan& plan, const LlmExecutionResources& resources) noexcept {
  const LlmCpuExecutionPlan* const cpu_plan =
      get_llm_cpu_execution_plan(plan);
  if (cpu_plan != nullptr && plan.phase == LlmPhase::Prefill &&
      plan.kv_layout == LlmKvLayout::Paged) {
    if (!cpu_plan->prefill.has_value() || !cpu_plan->paged.has_value() ||
        !plan.geometry.prefill.has_value() ||
        !validate_work_plan_layout(plan) || !resources.valid ||
        resources.model_plan_identity != plan.plan_identity ||
        !resources.buffers.complete() ||
        resources.layer_descriptors != nullptr ||
        resources.sequence_descriptors != nullptr ||
        resources.paged_layer_descriptors != nullptr ||
        resources.paged_assignment_descriptors != nullptr ||
        resources.prefill_layer_descriptors != nullptr ||
        resources.prefill_sequence_descriptors != nullptr ||
        resources.paged_prefill_layer_descriptors == nullptr ||
        resources.paged_prefill_assignment_descriptors == nullptr ||
        resources.weight_references == nullptr ||
        resources.k_references != nullptr || resources.v_references != nullptr ||
        resources.paged_k_block_references == nullptr ||
        resources.paged_v_block_references == nullptr ||
        resources.block_table != cpu_plan->paged->block_table() ||
        resources.block_table_entries !=
            cpu_plan->paged->layout.block_table_entries ||
        resources.paged_block_reference_count !=
            cpu_plan->paged->layout.total_physical_blocks ||
        resources.worker_count != cpu_plan->effective_workers ||
        resources.layer_descriptors_per_worker !=
            cpu_plan->layer_descriptors_per_worker ||
        resources.sequence_descriptors_per_worker !=
            cpu_plan->sequence_descriptors_per_worker ||
        resources.total_layer_descriptors !=
            cpu_plan->total_layer_descriptors ||
        resources.total_sequence_descriptors !=
            cpu_plan->total_sequence_descriptors ||
        !resources.initialization.complete ||
        resources.initialization.weight_bytes !=
            plan.geometry.active_weight_bytes_per_work_unit ||
        resources.initialization.k_bytes != plan.geometry.k_mapping_bytes ||
        resources.initialization.v_bytes != plan.geometry.v_mapping_bytes ||
        !resources.initialization.block_table_read_only) {
      return false;
    }
    const uint8_t* const weight =
        static_cast<const uint8_t*>(resources.buffers.weight.get());
    const uint8_t* const k =
        static_cast<const uint8_t*>(resources.buffers.k.get());
    const uint8_t* const v =
        static_cast<const uint8_t*>(resources.buffers.v.get());
    const LlmKvLayoutPlan& layout = cpu_plan->paged->layout;
    const size_t layer_pool_bytes =
        layout.physical_blocks_per_layer * layout.block_bytes;
    for (size_t worker_index = 0;
         worker_index < cpu_plan->effective_workers; ++worker_index) {
      const LlmWorkerWorkPlan& worker = cpu_plan->workers[worker_index];
      const size_t layer_base =
          worker_index * cpu_plan->layer_descriptors_per_worker;
      const size_t assignment_base =
          worker_index * cpu_plan->sequence_descriptors_per_worker;
      for (size_t layer = 0;
           layer < cpu_plan->layer_descriptors_per_worker; ++layer) {
        const LlmLayerRangeTemplate& source = worker.layers[layer];
        const LlmPagedPrefillLayerDescriptor& descriptor =
            resources.paged_prefill_layer_descriptors[layer_base + layer];
        const uint8_t* const expected_weight =
            source.weight.span_bytes == 0
                ? nullptr
                : weight + source.weight.offset_bytes;
        if (descriptor.weight_ptr != expected_weight ||
            descriptor.weight_bytes != source.weight.span_bytes ||
            descriptor.first_assignment_index !=
                source.first_sequence_index ||
            descriptor.assignment_count != source.sequence_count ||
            descriptor.layer_index != source.layer_index ||
            descriptor.reserved_zero != 0 ||
            resources.weight_references[layer_base + layer].span_bytes !=
                source.weight.span_bytes) {
          return false;
        }
      }
      for (size_t assignment_index = 0;
           assignment_index < cpu_plan->sequence_descriptors_per_worker;
           ++assignment_index) {
        const LlmPagedPrefillKvAssignmentTemplate& source =
            worker.paged_prefill_assignments[assignment_index];
        const LlmPagedPrefillKvAssignmentDescriptor& descriptor =
            resources.paged_prefill_assignment_descriptors[
                assignment_base + assignment_index];
        if (descriptor.block_table_row !=
                resources.block_table +
                    source.batch_sequence_index *
                        layout.blocks_per_sequence ||
            descriptor.k_layer_pool !=
                const_cast<uint8_t*>(k) +
                    source.layer_index * layer_pool_bytes ||
            descriptor.v_layer_pool !=
                const_cast<uint8_t*>(v) +
                    source.layer_index * layer_pool_bytes ||
            descriptor.first_logical_block !=
                source.first_logical_block ||
            descriptor.owned_block_count != source.block_count ||
            descriptor.blocks_per_sequence != layout.blocks_per_sequence ||
            descriptor.block_tokens != layout.kv_block_tokens ||
            descriptor.block_bytes != layout.block_bytes ||
            descriptor.last_block_valid_bytes !=
                layout.last_block_valid_bytes ||
            descriptor.prompt_tokens !=
                plan.geometry.prefill->prompt_tokens ||
            descriptor.attention_query_tile_tokens !=
                plan.geometry.prefill->attention_query_tile_tokens ||
            descriptor.record_bytes !=
                layout.k_or_v_record_bytes_per_layer ||
            descriptor.layer_index != source.layer_index ||
            descriptor.batch_sequence_index !=
                source.batch_sequence_index) {
          return false;
        }
      }
    }
    return true;
  }
  if (cpu_plan != nullptr && plan.phase == LlmPhase::Prefill) {
    if (!cpu_plan->prefill.has_value() ||
        !plan.geometry.prefill.has_value() ||
        !validate_work_plan_layout(plan) || !resources.valid ||
        resources.model_plan_identity != plan.plan_identity ||
        !resources.buffers.complete() ||
        resources.layer_descriptors != nullptr ||
        resources.sequence_descriptors != nullptr ||
        resources.paged_layer_descriptors != nullptr ||
        resources.paged_assignment_descriptors != nullptr ||
        resources.prefill_layer_descriptors == nullptr ||
        resources.prefill_sequence_descriptors == nullptr ||
        resources.paged_prefill_layer_descriptors != nullptr ||
        resources.paged_prefill_assignment_descriptors != nullptr ||
        resources.weight_references == nullptr ||
        resources.k_references != nullptr ||
        resources.v_references != nullptr ||
        resources.paged_k_block_references != nullptr ||
        resources.paged_v_block_references != nullptr ||
        resources.block_table != nullptr ||
        resources.block_table_entries != 0 ||
        resources.paged_block_reference_count != 0 ||
        resources.worker_count != cpu_plan->effective_workers ||
        resources.layer_descriptors_per_worker !=
            cpu_plan->layer_descriptors_per_worker ||
        resources.sequence_descriptors_per_worker !=
            cpu_plan->sequence_descriptors_per_worker ||
        resources.total_layer_descriptors !=
            cpu_plan->total_layer_descriptors ||
        resources.total_sequence_descriptors !=
            cpu_plan->total_sequence_descriptors ||
        !resources.initialization.complete ||
        resources.initialization.weight_bytes !=
            plan.geometry.active_weight_bytes_per_work_unit ||
        resources.initialization.k_bytes != plan.geometry.k_mapping_bytes ||
        resources.initialization.v_bytes != plan.geometry.v_mapping_bytes) {
      return false;
    }
    const uint8_t* const weight =
        static_cast<const uint8_t*>(resources.buffers.weight.get());
    const uint8_t* const k =
        static_cast<const uint8_t*>(resources.buffers.k.get());
    const uint8_t* const v =
        static_cast<const uint8_t*>(resources.buffers.v.get());
    for (size_t worker_index = 0;
         worker_index < cpu_plan->effective_workers; ++worker_index) {
      const LlmWorkerWorkPlan& worker = cpu_plan->workers[worker_index];
      const size_t layer_base =
          worker_index * cpu_plan->layer_descriptors_per_worker;
      const size_t sequence_base =
          worker_index * cpu_plan->sequence_descriptors_per_worker;
      for (size_t layer = 0; layer < plan.geometry.layer_count; ++layer) {
        const LlmLayerRangeTemplate& source = worker.layers[layer];
        const LlmPrefillLayerDescriptor& descriptor =
            resources.prefill_layer_descriptors[layer_base + layer];
        const uint8_t* const expected_weight =
            source.weight.span_bytes == 0
                ? nullptr
                : weight + source.weight.offset_bytes;
        if (descriptor.weight_ptr != expected_weight ||
            descriptor.weight_bytes != source.weight.span_bytes ||
            descriptor.first_sequence_index != source.first_sequence_index ||
            descriptor.sequence_count != source.sequence_count ||
            descriptor.layer_index != source.layer_index ||
            descriptor.reserved_zero != 0 ||
            resources.weight_references[layer_base + layer].span_bytes !=
                source.weight.span_bytes) {
          return false;
        }
      }
      for (size_t sequence_index = 0;
           sequence_index < cpu_plan->sequence_descriptors_per_worker;
           ++sequence_index) {
        const LlmPrefillKvSequenceRangeTemplate& source =
            worker.prefill_sequences[sequence_index];
        const LlmPrefillKvSequenceDescriptor& descriptor =
            resources.prefill_sequence_descriptors[sequence_base +
                                                   sequence_index];
        uint8_t* const expected_k =
            source.k_owned.span_bytes == 0
                ? nullptr
                : const_cast<uint8_t*>(k) + source.k_owned.offset_bytes;
        uint8_t* const expected_v =
            source.v_owned.span_bytes == 0
                ? nullptr
                : const_cast<uint8_t*>(v) + source.v_owned.offset_bytes;
        if (descriptor.k_owned_ptr != expected_k ||
            descriptor.v_owned_ptr != expected_v ||
            descriptor.first_token != source.first_token ||
            descriptor.owned_token_count != source.owned_token_count ||
            descriptor.prompt_tokens !=
                plan.geometry.prefill->prompt_tokens ||
            descriptor.attention_query_tile_tokens !=
                plan.geometry.prefill->attention_query_tile_tokens ||
            descriptor.record_bytes !=
                plan.geometry.k_or_v_record_bytes_per_layer ||
            descriptor.layer_index != source.layer_index ||
            descriptor.batch_sequence_index !=
                source.batch_sequence_index ||
            descriptor.reserved_zero != 0) {
          return false;
        }
      }
    }
    return true;
  }
  if (cpu_plan != nullptr && plan.kv_layout == LlmKvLayout::Paged) {
    if (!cpu_plan->paged.has_value() ||
        !validate_work_plan_layout(plan) || !resources.valid ||
        resources.model_plan_identity != plan.plan_identity ||
        !resources.buffers.complete() ||
        resources.layer_descriptors != nullptr ||
        resources.sequence_descriptors != nullptr ||
        resources.paged_layer_descriptors == nullptr ||
        resources.paged_assignment_descriptors == nullptr ||
        resources.prefill_layer_descriptors != nullptr ||
        resources.prefill_sequence_descriptors != nullptr ||
        resources.paged_prefill_layer_descriptors != nullptr ||
        resources.paged_prefill_assignment_descriptors != nullptr ||
        resources.weight_references == nullptr ||
        resources.k_references != nullptr || resources.v_references != nullptr ||
        resources.paged_k_block_references == nullptr ||
        resources.paged_v_block_references == nullptr ||
        resources.block_table != cpu_plan->paged->block_table() ||
        resources.block_table_entries !=
            cpu_plan->paged->layout.block_table_entries ||
        resources.paged_block_reference_count !=
            cpu_plan->paged->layout.total_physical_blocks ||
        resources.worker_count != cpu_plan->effective_workers ||
        resources.layer_descriptors_per_worker !=
            cpu_plan->layer_descriptors_per_worker ||
        resources.sequence_descriptors_per_worker !=
            cpu_plan->sequence_descriptors_per_worker ||
        resources.total_layer_descriptors !=
            cpu_plan->total_layer_descriptors ||
        resources.total_sequence_descriptors !=
            cpu_plan->total_sequence_descriptors ||
        !resources.initialization.complete ||
        resources.initialization.weight_bytes !=
            plan.geometry.active_weight_bytes_per_work_unit ||
        resources.initialization.k_bytes != plan.geometry.k_mapping_bytes ||
        resources.initialization.v_bytes != plan.geometry.v_mapping_bytes ||
        !resources.initialization.block_table_read_only) {
      return false;
    }
    const uint8_t* const weight =
        static_cast<const uint8_t*>(resources.buffers.weight.get());
    const uint8_t* const k =
        static_cast<const uint8_t*>(resources.buffers.k.get());
    const uint8_t* const v =
        static_cast<const uint8_t*>(resources.buffers.v.get());
    const LlmKvLayoutPlan& layout = cpu_plan->paged->layout;
    const size_t layer_pool_bytes =
        layout.physical_blocks_per_layer * layout.block_bytes;
    for (size_t worker_index = 0;
         worker_index < cpu_plan->effective_workers; ++worker_index) {
      const LlmWorkerWorkPlan& worker = cpu_plan->workers[worker_index];
      const size_t layer_base =
          worker_index * cpu_plan->layer_descriptors_per_worker;
      const size_t assignment_base =
          worker_index * cpu_plan->sequence_descriptors_per_worker;
      for (size_t layer = 0;
           layer < cpu_plan->layer_descriptors_per_worker; ++layer) {
        const LlmLayerRangeTemplate& source = worker.layers[layer];
        const LlmPagedLayerDescriptor& descriptor =
            resources.paged_layer_descriptors[layer_base + layer];
        const uint8_t* expected_weight =
            source.weight.span_bytes == 0
                ? nullptr
                : weight + source.weight.offset_bytes;
        if (descriptor.weight_ptr != expected_weight ||
            descriptor.weight_bytes != source.weight.span_bytes ||
            descriptor.first_assignment_index !=
                source.first_sequence_index ||
            descriptor.assignment_count != source.sequence_count ||
            descriptor.layer_index != source.layer_index ||
            descriptor.reserved_zero != 0 ||
            resources.weight_references[layer_base + layer].span_bytes !=
                source.weight.span_bytes) {
          return false;
        }
      }
      for (size_t assignment_index = 0;
           assignment_index < cpu_plan->sequence_descriptors_per_worker;
           ++assignment_index) {
        const LlmPagedKvAssignmentTemplate& source =
            worker.paged_assignments[assignment_index];
        const LlmPagedKvAssignmentDescriptor& descriptor =
            resources.paged_assignment_descriptors[assignment_base +
                                                   assignment_index];
        if (descriptor.block_table_row !=
                resources.block_table +
                    source.batch_sequence_index *
                        layout.blocks_per_sequence ||
            descriptor.k_layer_pool !=
                const_cast<uint8_t*>(k) +
                    source.layer_index * layer_pool_bytes ||
            descriptor.v_layer_pool !=
                const_cast<uint8_t*>(v) +
                    source.layer_index * layer_pool_bytes ||
            descriptor.first_logical_block !=
                source.first_logical_block ||
            descriptor.owned_block_count != source.block_count ||
            descriptor.blocks_per_sequence != layout.blocks_per_sequence ||
            descriptor.block_bytes != layout.block_bytes ||
            descriptor.last_block_valid_bytes !=
                layout.last_block_valid_bytes ||
            descriptor.decode_append_offset !=
                layout.decode_append_offset_in_last_block ||
            descriptor.append_record_bytes !=
                layout.k_or_v_record_bytes_per_layer ||
            descriptor.layer_index != source.layer_index ||
            descriptor.batch_sequence_index !=
                source.batch_sequence_index) {
          return false;
        }
      }
    }
    return true;
  }
  if (cpu_plan == nullptr || !validate_work_plan_layout(plan) ||
      !resources.valid || resources.model_plan_identity != plan.plan_identity ||
      !resources.buffers.complete() || resources.layer_descriptors == nullptr ||
      resources.sequence_descriptors == nullptr ||
      resources.paged_layer_descriptors != nullptr ||
      resources.paged_assignment_descriptors != nullptr ||
      resources.prefill_layer_descriptors != nullptr ||
      resources.prefill_sequence_descriptors != nullptr ||
      resources.paged_prefill_layer_descriptors != nullptr ||
      resources.paged_prefill_assignment_descriptors != nullptr ||
      resources.weight_references == nullptr ||
      resources.k_references == nullptr || resources.v_references == nullptr ||
      resources.paged_k_block_references != nullptr ||
      resources.paged_v_block_references != nullptr ||
      resources.block_table != nullptr || resources.block_table_entries != 0 ||
      resources.paged_block_reference_count != 0 ||
      resources.worker_count != cpu_plan->effective_workers ||
      resources.layer_descriptors_per_worker !=
          cpu_plan->layer_descriptors_per_worker ||
      resources.sequence_descriptors_per_worker !=
          cpu_plan->sequence_descriptors_per_worker ||
      resources.total_layer_descriptors !=
          cpu_plan->total_layer_descriptors ||
      resources.total_sequence_descriptors !=
          cpu_plan->total_sequence_descriptors ||
      !resources.initialization.complete ||
      resources.initialization.weight_bytes != plan.geometry.active_weight_bytes_per_work_unit ||
      resources.initialization.k_bytes != plan.geometry.k_mapping_bytes ||
      resources.initialization.v_bytes != plan.geometry.v_mapping_bytes) {
    return false;
  }

  const uint8_t* const weight = static_cast<const uint8_t*>(resources.buffers.weight.get());
  const uint8_t* const k = static_cast<const uint8_t*>(resources.buffers.k.get());
  const uint8_t* const v = static_cast<const uint8_t*>(resources.buffers.v.get());
  for (size_t worker_index = 0;
       worker_index < cpu_plan->effective_workers; ++worker_index) {
    const LlmWorkerWorkPlan& worker = cpu_plan->workers[worker_index];
    const size_t layer_base =
        worker_index * cpu_plan->layer_descriptors_per_worker;
    const size_t sequence_base =
        worker_index * cpu_plan->sequence_descriptors_per_worker;
    for (size_t layer = 0;
         layer < cpu_plan->layer_descriptors_per_worker; ++layer) {
      const LlmLayerRangeTemplate& source = worker.layers[layer];
      const LlmLayerDescriptor& descriptor = resources.layer_descriptors[layer_base + layer];
      const LlmStaticSpanReference& reference = resources.weight_references[layer_base + layer];
      const uint8_t* expected_pointer = source.weight.span_bytes == 0 ? nullptr : weight + source.weight.offset_bytes;
      if (descriptor.weight_ptr != expected_pointer || descriptor.weight_bytes != source.weight.span_bytes ||
          descriptor.first_sequence_index != source.first_sequence_index ||
          descriptor.sequence_count != source.sequence_count || descriptor.layer_index != source.layer_index ||
          descriptor.reserved_zero != 0 || reference.span_bytes != source.weight.span_bytes) {
        return false;
      }
    }
    for (size_t sequence_index = 0;
         sequence_index < cpu_plan->sequence_descriptors_per_worker;
         ++sequence_index) {
      const LlmKvSequenceRangeTemplate& source = worker.sequences[sequence_index];
      const LlmKvSequenceDescriptor& descriptor = resources.sequence_descriptors[sequence_base + sequence_index];
      const uint8_t* expected_k_visible =
          source.k_visible.span_bytes == 0 ? nullptr : k + source.k_visible.offset_bytes;
      const uint8_t* expected_v_visible =
          source.v_visible.span_bytes == 0 ? nullptr : v + source.v_visible.offset_bytes;
      uint8_t* expected_k_append =
          source.k_append.span_bytes == 0 ? nullptr : const_cast<uint8_t*>(k) + source.k_append.offset_bytes;
      uint8_t* expected_v_append =
          source.v_append.span_bytes == 0 ? nullptr : const_cast<uint8_t*>(v) + source.v_append.offset_bytes;
      const uint64_t expected_record_offset = source.k_append.span_bytes == 0 ? 0 : source.append_record_byte_offset;
      if (descriptor.k_visible_ptr != expected_k_visible || descriptor.k_visible_bytes != source.k_visible.span_bytes ||
          descriptor.v_visible_ptr != expected_v_visible || descriptor.v_visible_bytes != source.v_visible.span_bytes ||
          descriptor.k_append_ptr != expected_k_append || descriptor.k_append_bytes != source.k_append.span_bytes ||
          descriptor.v_append_ptr != expected_v_append || descriptor.v_append_bytes != source.v_append.span_bytes ||
          descriptor.batch_sequence_index != source.batch_sequence_index ||
          descriptor.append_record_byte_offset != expected_record_offset ||
          resources.k_references[sequence_base + sequence_index].span_bytes != source.k_visible.span_bytes ||
          resources.v_references[sequence_base + sequence_index].span_bytes != source.v_visible.span_bytes) {
        return false;
      }
    }
  }
  return true;
}

bool scenario_plan_matches_model(const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan& scenario_plan) {
  if (!scenario_plan.valid || scenario_plan.work_units == 0 || llm_scenario_flags(scenario_plan.scenario) == 0 ||
      scenario_plan.model_plan_identity != model_plan.plan_identity) {
    return false;
  }
  const size_t scenario_index = static_cast<size_t>(scenario_plan.scenario);
  if (scenario_index >= kLlmScenarioCount || scenario_plan.scenario_seed != model_plan.scenario_seeds[scenario_index]) {
    return false;
  }
  const LlmScenarioWorkPlan rebuilt = build_llm_scenario_work_plan(
      model_plan, scenario_plan.scenario, scenario_plan.work_units, scenario_plan.explicit_iterations);
  return rebuilt.valid && rebuilt.reason_code == scenario_plan.reason_code &&
         rebuilt.work_unit_kind == scenario_plan.work_unit_kind &&
         rebuilt.kv_write_kind == scenario_plan.kv_write_kind &&
         rebuilt.explicit_iterations == scenario_plan.explicit_iterations &&
         rebuilt.model_plan_identity == scenario_plan.model_plan_identity &&
         rebuilt.scenario_seed == scenario_plan.scenario_seed &&
         rebuilt.work_units == scenario_plan.work_units &&
         rebuilt.weight_read_bytes_per_work_unit == scenario_plan.weight_read_bytes_per_work_unit &&
         rebuilt.kv_read_bytes_per_work_unit == scenario_plan.kv_read_bytes_per_work_unit &&
         rebuilt.kv_write_bytes_per_work_unit == scenario_plan.kv_write_bytes_per_work_unit &&
         rebuilt.effective_model_payload_bytes_per_work_unit ==
             scenario_plan.effective_model_payload_bytes_per_work_unit &&
         rebuilt.layout_metadata_lookup_count_per_work_unit ==
             scenario_plan.layout_metadata_lookup_count_per_work_unit &&
         rebuilt.layout_metadata_read_bytes_per_work_unit ==
             scenario_plan.layout_metadata_read_bytes_per_work_unit &&
         rebuilt.accounted_bytes_per_work_unit == scenario_plan.accounted_bytes_per_work_unit &&
         rebuilt.weight_read_bytes == scenario_plan.weight_read_bytes &&
         rebuilt.kv_read_bytes == scenario_plan.kv_read_bytes &&
         rebuilt.kv_write_bytes == scenario_plan.kv_write_bytes &&
         rebuilt.effective_model_payload_bytes == scenario_plan.effective_model_payload_bytes &&
         rebuilt.layout_metadata_lookup_count == scenario_plan.layout_metadata_lookup_count &&
         rebuilt.layout_metadata_read_bytes == scenario_plan.layout_metadata_read_bytes &&
         rebuilt.task_accounted_bytes == scenario_plan.task_accounted_bytes &&
         rebuilt.maximum_work_units_by_work_unit_cap ==
             scenario_plan.maximum_work_units_by_work_unit_cap &&
         rebuilt.maximum_work_units_by_guardrail == scenario_plan.maximum_work_units_by_guardrail &&
         rebuilt.effective_maximum_work_units == scenario_plan.effective_maximum_work_units &&
         rebuilt.plan_identity == scenario_plan.plan_identity;
}

uint8_t append_byte(uint64_t scenario_seed, uint64_t task_local_work_unit, uint64_t layer_index,
                    uint64_t batch_sequence_index, size_t record_byte_offset, LlmChecksumComponent component) noexcept {
  const uint64_t word = llm_append_word(scenario_seed, task_local_work_unit, layer_index, batch_sequence_index,
                                        static_cast<uint64_t>(record_byte_offset / sizeof(uint64_t)), component);
  return static_cast<uint8_t>(word >> ((record_byte_offset % sizeof(uint64_t)) * 8U));
}

/**
 * Replace an initialized span's append suffix in sum space.
 *
 * Only affected local checksum words are regenerated. Prefix bytes in a
 * mid-word intersection come from the versioned buffer formula, and suffix
 * bytes come from canonical append-record offsets. This handles worker splits
 * whose read-word and append-word alignments differ without rereading memory.
 */
bool append_adjusted_reference(const LlmStaticSpanReference& static_reference,
                               const LlmKvSequenceRangeTemplate& sequence, uint64_t buffer_seed, uint64_t scenario_seed,
                               uint64_t task_local_work_unit, LlmChecksumComponent component,
                               LlmStaticSpanReference& adjusted) noexcept {
  adjusted = static_reference;
  const LlmByteRange& visible = component == LlmChecksumComponent::K ? sequence.k_visible : sequence.v_visible;
  const LlmByteRange& append = component == LlmChecksumComponent::K ? sequence.k_append : sequence.v_append;
  if (static_reference.span_bytes != visible.span_bytes) {
    return false;
  }
  if (append.span_bytes == 0) {
    return sequence.append_record_byte_offset == 0;
  }

  size_t visible_end = 0;
  size_t append_end = 0;
  if (visible.span_bytes == 0 || append.offset_bytes < visible.offset_bytes ||
      !NumericUtils::checked_add(visible.offset_bytes, visible.span_bytes, visible_end) ||
      !NumericUtils::checked_add(append.offset_bytes, append.span_bytes, append_end) || append_end != visible_end) {
    return false;
  }
  const size_t append_local_start = append.offset_bytes - visible.offset_bytes;
  const size_t first_affected_word = append_local_start / sizeof(uint64_t);
  const size_t total_words = (visible.span_bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t);
  for (size_t word_index = first_affected_word; word_index < total_words; ++word_index) {
    const size_t local_word_start = word_index * sizeof(uint64_t);
    const size_t word_bytes = std::min(sizeof(uint64_t), visible.span_bytes - local_word_start);
    const uint64_t initialized_word =
        pattern_byte_word(buffer_seed, visible.offset_bytes + local_word_start, word_bytes);
    uint64_t appended_word = 0;
    for (size_t byte_index = 0; byte_index < word_bytes; ++byte_index) {
      const size_t local_byte = local_word_start + byte_index;
      uint8_t value = 0;
      if (local_byte < append_local_start) {
        value = static_cast<uint8_t>(initialized_word >> (byte_index * 8U));
      } else {
        const size_t record_byte = sequence.append_record_byte_offset + local_byte - append_local_start;
        value = append_byte(scenario_seed, task_local_work_unit, sequence.layer_index, sequence.batch_sequence_index,
                            record_byte, component);
      }
      appended_word |= static_cast<uint64_t>(value) << (byte_index * 8U);
    }
    if ((word_index & 1U) == 0) {
      adjusted.span_even = adjusted.span_even - initialized_word + appended_word;
    } else {
      adjusted.span_odd = adjusted.span_odd - initialized_word + appended_word;
    }
  }
  return true;
}

bool paged_append_adjusted_reference(
    const LlmStaticSpanReference& static_reference,
    const LlmKvLayoutPlan& layout, uint32_t physical_block_id,
    uint64_t buffer_seed, uint64_t scenario_seed,
    uint64_t task_local_work_unit, size_t layer_index,
    size_t batch_sequence_index, LlmChecksumComponent component,
    LlmStaticSpanReference& adjusted) noexcept {
  adjusted = static_reference;
  if (static_reference.span_bytes != layout.last_block_valid_bytes ||
      layout.decode_append_offset_in_last_block >
          layout.last_block_valid_bytes ||
      layout.k_or_v_record_bytes_per_layer >
          layout.last_block_valid_bytes -
              layout.decode_append_offset_in_last_block) {
    return false;
  }
  const size_t append_start =
      layout.decode_append_offset_in_last_block;
  const size_t total_words =
      (layout.last_block_valid_bytes + sizeof(uint64_t) - 1) /
      sizeof(uint64_t);
  const size_t first_affected_word = append_start / sizeof(uint64_t);
  for (size_t word_index = first_affected_word; word_index < total_words;
       ++word_index) {
    const size_t local_word_start = word_index * sizeof(uint64_t);
    const size_t word_bytes =
        std::min(sizeof(uint64_t),
                 layout.last_block_valid_bytes - local_word_start);
    const uint64_t initialized_word = paged_pattern_byte_word(
        buffer_seed, layer_index, physical_block_id, local_word_start,
        word_bytes);
    uint64_t appended_word = 0;
    for (size_t byte_index = 0; byte_index < word_bytes; ++byte_index) {
      const size_t local_byte = local_word_start + byte_index;
      const uint8_t value =
          local_byte < append_start
              ? static_cast<uint8_t>(initialized_word >>
                                     (byte_index * 8U))
              : append_byte(scenario_seed, task_local_work_unit,
                            layer_index, batch_sequence_index,
                            local_byte - append_start, component);
      appended_word |=
          static_cast<uint64_t>(value) << (byte_index * 8U);
    }
    if ((word_index & 1U) == 0) {
      adjusted.span_even =
          adjusted.span_even - initialized_word + appended_word;
    } else {
      adjusted.span_odd =
          adjusted.span_odd - initialized_word + appended_word;
    }
  }
  return true;
}

bool absorb_reference(LlmReadChecksumComponent& checksum, const LlmStaticSpanReference& reference) noexcept {
  if (reference.span_bytes == 0) {
    return true;
  }
  if (checksum.exact_bytes_read > std::numeric_limits<uint64_t>::max() - reference.span_bytes ||
      checksum.span_count == std::numeric_limits<uint64_t>::max()) {
    return false;
  }
  const uint64_t ordinal = checksum.span_count;
  checksum.state_a = rotate_left(checksum.state_a + reference.span_even + kAppendStepMultiplier * (ordinal + 1), 17);
  checksum.state_b = rotate_left(
      checksum.state_b + reference.span_odd + reference.span_bytes + kAppendWordMultiplier * (ordinal + 1), 29);
  checksum.exact_bytes_read += reference.span_bytes;
  ++checksum.span_count;
  return true;
}

bool add_checksum_bytes(uint64_t bytes, uint64_t& total) noexcept {
  if (total > std::numeric_limits<uint64_t>::max() - bytes) {
    return false;
  }
  total += bytes;
  return true;
}

using LlmWide = unsigned __int128;

LlmWide prefill_wide_floor_sum(LlmWide count, LlmWide denominator,
                               LlmWide slope, LlmWide intercept) noexcept {
  LlmWide answer = 0;
  while (true) {
    if (slope >= denominator) {
      answer += (count * (count - 1) / 2) * (slope / denominator);
      slope %= denominator;
    }
    if (intercept >= denominator) {
      answer += count * (intercept / denominator);
      intercept %= denominator;
    }
    const LlmWide maximum = slope * count + intercept;
    if (maximum < denominator) {
      return answer;
    }
    count = maximum / denominator;
    intercept = maximum % denominator;
    std::swap(denominator, slope);
  }
}

uint64_t prefill_affine_extracted_task_sum(
    uint64_t first_value, size_t operation_count, unsigned int shift,
    unsigned int bit_count) noexcept {
  constexpr uint64_t kOperationDelta = 0x9E3779B97F4A7C15ULL;
  const LlmWide count = operation_count;
  const LlmWide modulus = LlmWide{1} << 64;
  const LlmWide wraps = prefill_wide_floor_sum(
      count, modulus, kOperationDelta, first_value);
  const auto quotient_sum = [&](unsigned int quotient_shift) {
    if (quotient_shift == 64) {
      return LlmWide{0};
    }
    const LlmWide denominator = LlmWide{1} << quotient_shift;
    return prefill_wide_floor_sum(count, denominator, kOperationDelta,
                                  first_value) -
           (modulus / denominator) * wraps;
  };
  const LlmWide low = quotient_sum(shift);
  const LlmWide high = quotient_sum(shift + bit_count);
  return static_cast<uint64_t>(low - (LlmWide{1} << bit_count) * high);
}

bool calculate_prefill_byte_range_task_checksum(
    uint64_t scenario_seed, size_t operation_count, size_t layer_index,
    size_t batch_index, LlmPrefillKvDomain domain, size_t first_byte,
    size_t byte_count, uint64_t& even_sum, uint64_t& odd_sum) noexcept {
  if (operation_count == 0) {
    return false;
  }
  size_t end_byte = 0;
  if (!NumericUtils::checked_add(first_byte, byte_count, end_byte)) {
    return false;
  }
  if (byte_count == 0) {
    return true;
  }
  size_t first_word = first_byte / sizeof(uint64_t);
  const size_t last_word = (end_byte - 1) / sizeof(uint64_t);
  const auto add_partial = [&](size_t word_index, unsigned int byte_shift,
                               unsigned int bytes) {
    const uint64_t first_value = llm_prefill_affine64_word(
        scenario_seed, 0, layer_index, batch_index, domain, word_index);
    const uint64_t extracted = prefill_affine_extracted_task_sum(
        first_value, operation_count, byte_shift * 8U, bytes * 8U);
    const uint64_t contribution = extracted << (byte_shift * 8U);
    if ((word_index & 1U) == 0) {
      even_sum += contribution;
    } else {
      odd_sum += contribution;
    }
  };

  const unsigned int first_shift =
      static_cast<unsigned int>(first_byte % sizeof(uint64_t));
  if (first_word == last_word) {
    add_partial(first_word, first_shift,
                static_cast<unsigned int>(byte_count));
    return true;
  }
  if (first_shift != 0) {
    add_partial(first_word, first_shift,
                static_cast<unsigned int>(sizeof(uint64_t) - first_shift));
    ++first_word;
  }
  const size_t complete_word_end = end_byte / sizeof(uint64_t);
  if (complete_word_end > first_word) {
    const LlmPrefillAffine64Checksum complete =
        calculate_llm_prefill_affine64_task_checksum(
            scenario_seed, operation_count, layer_index, batch_index, domain,
            first_word, complete_word_end - first_word);
    if (!complete.valid) {
      return false;
    }
    even_sum += complete.even_logical_word_sum;
    odd_sum += complete.odd_logical_word_sum;
  }
  const unsigned int final_bytes =
      static_cast<unsigned int>(end_byte % sizeof(uint64_t));
  if (final_bytes != 0) {
    add_partial(last_word, 0, final_bytes);
  }
  return true;
}

bool calculate_prefill_byte_range_operation_checksum(
    uint64_t scenario_seed, size_t operation_ordinal, size_t layer_index,
    size_t batch_index, LlmPrefillKvDomain domain, size_t first_byte,
    size_t byte_count, uint64_t& even_sum, uint64_t& odd_sum) noexcept {
  size_t end_byte = 0;
  if (!NumericUtils::checked_add(first_byte, byte_count, end_byte)) {
    return false;
  }
  if (byte_count == 0) {
    return true;
  }
  size_t first_word = first_byte / sizeof(uint64_t);
  const size_t last_word = (end_byte - 1) / sizeof(uint64_t);
  const auto add_partial = [&](size_t word_index, unsigned int byte_shift,
                               unsigned int bytes) {
    const uint64_t word = llm_prefill_affine64_word(
        scenario_seed, operation_ordinal, layer_index, batch_index, domain,
        word_index);
    const uint64_t extracted =
        (word >> (byte_shift * 8U)) & low_byte_mask(bytes);
    const uint64_t contribution = extracted << (byte_shift * 8U);
    if ((word_index & 1U) == 0) {
      even_sum += contribution;
    } else {
      odd_sum += contribution;
    }
  };

  const unsigned int first_shift =
      static_cast<unsigned int>(first_byte % sizeof(uint64_t));
  if (first_word == last_word) {
    add_partial(first_word, first_shift,
                static_cast<unsigned int>(byte_count));
    return true;
  }
  if (first_shift != 0) {
    add_partial(first_word, first_shift,
                static_cast<unsigned int>(sizeof(uint64_t) - first_shift));
    ++first_word;
  }
  const size_t complete_word_end = end_byte / sizeof(uint64_t);
  if (complete_word_end > first_word) {
    const LlmPrefillAffine64Checksum complete =
        calculate_llm_prefill_affine64_checksum(
            scenario_seed, operation_ordinal, layer_index, batch_index,
            domain, first_word, complete_word_end - first_word);
    if (!complete.valid) {
      return false;
    }
    even_sum += complete.even_logical_word_sum;
    odd_sum += complete.odd_logical_word_sum;
  }
  const unsigned int final_bytes =
      static_cast<unsigned int>(end_byte % sizeof(uint64_t));
  if (final_bytes != 0) {
    add_partial(last_word, 0, final_bytes);
  }
  return true;
}

LlmExpectedChecksumResult calculate_prefill_expected_checksums(
    const LlmMemoryWorkPlan& model_plan,
    const LlmScenarioWorkPlan& scenario_plan,
    const LlmExecutionResources& resources) {
  LlmExpectedChecksumResult result;
  const LlmCpuExecutionPlan* const cpu_plan =
      get_llm_cpu_execution_plan(model_plan);
  if (cpu_plan == nullptr || !cpu_plan->prefill.has_value() ||
      !model_plan.geometry.prefill.has_value() ||
      !materialized_resources_match_plan(model_plan, resources) ||
      !scenario_plan_matches_model(model_plan, scenario_plan)) {
    result.reason_code = LlmExecutorReason::INVALID_RESOURCES;
    return result;
  }
  const bool reads_weight =
      (llm_scenario_flags(scenario_plan.scenario) &
       kLlmScenarioFlagWeight) != 0;
  const bool reads_kv =
      (llm_scenario_flags(scenario_plan.scenario) &
       kLlmScenarioFlagKv) != 0;
  const size_t per_scenario =
      cpu_plan->prefill->sequence_descriptors_per_scenario_per_worker;
  const size_t scenario_base =
      static_cast<size_t>(scenario_plan.scenario) * per_scenario;
  result.workers.resize(cpu_plan->effective_workers);
  uint64_t total_weight_bytes = 0;
  uint64_t total_k_bytes = 0;
  uint64_t total_v_bytes = 0;
  for (size_t worker_index = 0;
       worker_index < cpu_plan->effective_workers; ++worker_index) {
    LlmWorkerChecksum& checksum = result.workers[worker_index];
    checksum.weight = initial_llm_read_checksum(LlmChecksumComponent::Weight);
    checksum.k = {};
    checksum.v = {};
    const LlmStaticSpanReference* const weight_references =
        resources.worker_weight_references(worker_index);
    if (weight_references == nullptr) {
      result.reason_code = LlmExecutorReason::INVALID_RESOURCES;
      result.workers.clear();
      return result;
    }
    if (reads_weight) {
      for (size_t operation = 0; operation < scenario_plan.work_units;
           ++operation) {
        for (size_t layer = 0; layer < model_plan.geometry.layer_count;
             ++layer) {
          if (!absorb_reference(checksum.weight,
                                weight_references[layer])) {
            result.reason_code =
                LlmExecutorReason::EXPECTED_CHECKSUM_OVERFLOW;
            result.workers.clear();
            return result;
          }
        }
      }
    }
    if (reads_kv) {
      const LlmWorkerWorkPlan& worker = cpu_plan->workers[worker_index];
      for (size_t row = 0; row < per_scenario; ++row) {
        const LlmPrefillKvSequenceRangeTemplate& sequence =
            worker.prefill_sequences[scenario_base + row];
        if (sequence.owned_token_count == 0) {
          continue;
        }
        size_t first_byte = 0;
        if (!NumericUtils::checked_multiply(
                sequence.first_token,
                model_plan.geometry.k_or_v_record_bytes_per_layer,
                first_byte)) {
          result.reason_code =
              LlmExecutorReason::EXPECTED_CHECKSUM_OVERFLOW;
          result.workers.clear();
          return result;
        }
        size_t owned_end = 0;
        if (!NumericUtils::checked_add(sequence.first_token,
                                       sequence.owned_token_count,
                                       owned_end)) {
          result.reason_code =
              LlmExecutorReason::EXPECTED_CHECKSUM_OVERFLOW;
          result.workers.clear();
          return result;
        }
        const size_t query_tile_tokens =
            model_plan.geometry.prefill->attention_query_tile_tokens;
        size_t tile_index = sequence.first_token / query_tile_tokens;
        size_t tile_end = tile_index * query_tile_tokens;
        while (tile_end < owned_end) {
          tile_end += std::min(query_tile_tokens,
                               model_plan.geometry.prefill->prompt_tokens -
                                   tile_end);
          const size_t visible_owned_tokens =
              std::min(tile_end, owned_end) - sequence.first_token;
          size_t span_bytes = 0;
          if (!NumericUtils::checked_multiply(
                  visible_owned_tokens,
                  model_plan.geometry.k_or_v_record_bytes_per_layer,
                  span_bytes)) {
            result.reason_code =
                LlmExecutorReason::EXPECTED_CHECKSUM_OVERFLOW;
            result.workers.clear();
            return result;
          }
          const size_t visit_multiplicity =
              tile_end >= owned_end
                  ? model_plan.geometry.prefill->tile_count - tile_index
                  : size_t{1};
          uint64_t k_even = 0;
          uint64_t k_odd = 0;
          uint64_t v_even = 0;
          uint64_t v_odd = 0;
          size_t span_bytes_per_task = 0;
          size_t repeated_span_bytes = 0;
          size_t repeated_span_count = 0;
          if (!calculate_prefill_byte_range_task_checksum(
                  scenario_plan.scenario_seed, scenario_plan.work_units,
                  sequence.layer_index, sequence.batch_sequence_index,
                  LlmPrefillKvDomain::K, first_byte, span_bytes, k_even,
                  k_odd) ||
              !calculate_prefill_byte_range_task_checksum(
                  scenario_plan.scenario_seed, scenario_plan.work_units,
                  sequence.layer_index, sequence.batch_sequence_index,
                  LlmPrefillKvDomain::V, first_byte, span_bytes, v_even,
                  v_odd) ||
              !NumericUtils::checked_multiply(
                  span_bytes, scenario_plan.work_units,
                  span_bytes_per_task) ||
              !NumericUtils::checked_multiply(
                  span_bytes_per_task, visit_multiplicity,
                  repeated_span_bytes) ||
              !NumericUtils::checked_multiply(
                  scenario_plan.work_units, visit_multiplicity,
                  repeated_span_count) ||
              repeated_span_bytes > std::numeric_limits<uint64_t>::max() ||
              repeated_span_count > std::numeric_limits<uint64_t>::max() ||
              checksum.k.exact_bytes_read >
                  std::numeric_limits<uint64_t>::max() -
                      repeated_span_bytes ||
              checksum.v.exact_bytes_read >
                  std::numeric_limits<uint64_t>::max() -
                      repeated_span_bytes ||
              checksum.k.span_count >
                  std::numeric_limits<uint64_t>::max() -
                      repeated_span_count ||
              checksum.v.span_count >
                  std::numeric_limits<uint64_t>::max() -
                      repeated_span_count) {
            result.reason_code =
                LlmExecutorReason::EXPECTED_CHECKSUM_OVERFLOW;
            result.workers.clear();
            return result;
          }
          const uint64_t multiplicity =
              static_cast<uint64_t>(visit_multiplicity);
          checksum.k.state_a += k_even * multiplicity;
          checksum.k.state_b += k_odd * multiplicity;
          checksum.v.state_a += v_even * multiplicity;
          checksum.v.state_b += v_odd * multiplicity;
          checksum.k.exact_bytes_read += repeated_span_bytes;
          checksum.v.exact_bytes_read += repeated_span_bytes;
          checksum.k.span_count += repeated_span_count;
          checksum.v.span_count += repeated_span_count;
          if (tile_end >= owned_end) {
            break;
          }
          ++tile_index;
        }
      }
    }
    if (!add_checksum_bytes(checksum.weight.exact_bytes_read,
                            total_weight_bytes) ||
        !add_checksum_bytes(checksum.k.exact_bytes_read, total_k_bytes) ||
        !add_checksum_bytes(checksum.v.exact_bytes_read, total_v_bytes)) {
      result.reason_code = LlmExecutorReason::EXPECTED_CHECKSUM_OVERFLOW;
      result.workers.clear();
      return result;
    }
  }
  uint64_t total_kv_bytes = 0;
  if (!add_checksum_bytes(total_k_bytes, total_kv_bytes) ||
      !add_checksum_bytes(total_v_bytes, total_kv_bytes) ||
      total_weight_bytes != scenario_plan.weight_read_bytes ||
      total_kv_bytes != scenario_plan.kv_read_bytes) {
    result.reason_code = LlmExecutorReason::INVALID_DESCRIPTOR_LAYOUT;
    result.workers.clear();
    return result;
  }
  result.run_checksum =
      fold_llm_worker_checksums(result.workers.data(), result.workers.size());
  result.valid = true;
  result.reason_code = LlmExecutorReason::VALID;
  return result;
}

LlmExpectedChecksumResult calculate_paged_prefill_expected_checksums(
    const LlmMemoryWorkPlan& model_plan,
    const LlmScenarioWorkPlan& scenario_plan,
    const LlmExecutionResources& resources) {
  LlmExpectedChecksumResult result;
  const LlmCpuExecutionPlan* const cpu_plan =
      get_llm_cpu_execution_plan(model_plan);
  if (cpu_plan == nullptr || !cpu_plan->prefill.has_value() ||
      !cpu_plan->paged.has_value() ||
      !model_plan.geometry.prefill.has_value() ||
      !materialized_resources_match_plan(model_plan, resources) ||
      !scenario_plan_matches_model(model_plan, scenario_plan)) {
    result.reason_code = LlmExecutorReason::INVALID_RESOURCES;
    return result;
  }
  const LlmKvLayoutPlan& layout = cpu_plan->paged->layout;
  const size_t scenario_index =
      static_cast<size_t>(scenario_plan.scenario);
  if (scenario_index >= kLlmScenarioCount) {
    return result;
  }
  const size_t assignments_per_scenario =
      cpu_plan->prefill->sequence_descriptors_per_scenario_per_worker;
  const size_t scenario_base = scenario_index * assignments_per_scenario;
  const uint64_t flags = llm_scenario_flags(scenario_plan.scenario);
  const bool reads_weight = (flags & kLlmScenarioFlagWeight) != 0;
  const bool reads_kv = (flags & kLlmScenarioFlagKv) != 0;
  const size_t prompt_tokens =
      model_plan.geometry.prefill->prompt_tokens;
  const size_t query_tile_tokens =
      model_plan.geometry.prefill->attention_query_tile_tokens;
  const size_t record_bytes =
      model_plan.geometry.k_or_v_record_bytes_per_layer;

  result.workers.resize(cpu_plan->effective_workers);
  uint64_t total_weight_bytes = 0;
  uint64_t total_k_bytes = 0;
  uint64_t total_v_bytes = 0;
  uint64_t total_lookup_count = 0;
  for (size_t worker_index = 0;
       worker_index < cpu_plan->effective_workers; ++worker_index) {
    LlmWorkerChecksum& checksum = result.workers[worker_index];
    checksum.weight = initial_llm_read_checksum(LlmChecksumComponent::Weight);
    checksum.k = {};
    checksum.v = {};
    const LlmWorkerWorkPlan& worker = cpu_plan->workers[worker_index];
    const LlmStaticSpanReference* const weight_references =
        resources.worker_weight_references(worker_index);
    if (weight_references == nullptr) {
      result.reason_code = LlmExecutorReason::INVALID_RESOURCES;
      result.workers.clear();
      return result;
    }

    for (size_t operation = 0; operation < scenario_plan.work_units;
         ++operation) {
      for (size_t layer = 0; layer < layout.layer_count; ++layer) {
        if (reads_weight &&
            !absorb_reference(checksum.weight, weight_references[layer])) {
          result.reason_code =
              LlmExecutorReason::EXPECTED_CHECKSUM_OVERFLOW;
          result.workers.clear();
          return result;
        }
        if (!reads_kv) {
          continue;
        }
        for (size_t batch = 0; batch < layout.batch_size; ++batch) {
          const size_t row = layer * layout.batch_size + batch;
          const LlmPagedPrefillKvAssignmentTemplate& assignment =
              worker.paged_prefill_assignments[scenario_base + row];
          if (assignment.block_count == 0) {
            continue;
          }
          size_t assignment_end = 0;
          size_t first_owned_token = 0;
          if (!NumericUtils::checked_add(
                  assignment.first_logical_block, assignment.block_count,
                  assignment_end) ||
              assignment_end > layout.blocks_per_sequence ||
              !NumericUtils::checked_multiply(
                  assignment.first_logical_block, layout.kv_block_tokens,
                  first_owned_token)) {
            result.reason_code =
                LlmExecutorReason::INVALID_DESCRIPTOR_LAYOUT;
            result.workers.clear();
            return result;
          }
          const size_t table_row = batch * layout.blocks_per_sequence;

          for (size_t logical_block = assignment.first_logical_block;
               logical_block < assignment_end; ++logical_block) {
            const size_t table_index = table_row + logical_block;
            if (table_index >= resources.block_table_entries) {
              result.reason_code =
                  LlmExecutorReason::INVALID_DESCRIPTOR_LAYOUT;
              result.workers.clear();
              return result;
            }
            const uint32_t physical_id = resources.block_table[table_index];
            if (physical_id >= layout.physical_blocks_per_layer ||
                total_lookup_count ==
                    std::numeric_limits<uint64_t>::max()) {
              result.reason_code =
                  LlmExecutorReason::INVALID_DESCRIPTOR_LAYOUT;
              result.workers.clear();
              return result;
            }
            mix_llm_paged_lookup(
                checksum.k, table_index, physical_id, 0, operation);
            ++total_lookup_count;
          }

          size_t tile_end =
              (first_owned_token / query_tile_tokens) * query_tile_tokens;
          while (tile_end < prompt_tokens) {
            tile_end +=
                std::min(query_tile_tokens, prompt_tokens - tile_end);
            size_t visited_block_end =
                tile_end / layout.kv_block_tokens +
                (tile_end % layout.kv_block_tokens != 0 ? 1 : 0);
            visited_block_end = std::min(visited_block_end, assignment_end);

            for (size_t logical_block = assignment.first_logical_block;
                 logical_block < visited_block_end; ++logical_block) {
              const size_t table_index = table_row + logical_block;
              const uint32_t physical_id = resources.block_table[table_index];
              size_t first_byte = 0;
              size_t prefix_bytes = 0;
              if (physical_id >= layout.physical_blocks_per_layer ||
                  !NumericUtils::checked_multiply(
                      logical_block, layout.block_bytes, first_byte) ||
                  !NumericUtils::checked_multiply(
                      tile_end, record_bytes, prefix_bytes) ||
                  prefix_bytes <= first_byte ||
                  total_lookup_count ==
                      std::numeric_limits<uint64_t>::max()) {
                result.reason_code =
                    LlmExecutorReason::INVALID_DESCRIPTOR_LAYOUT;
                result.workers.clear();
                return result;
              }
              const size_t visit_bytes =
                  std::min(layout.block_bytes, prefix_bytes - first_byte);
              mix_llm_paged_lookup(
                  checksum.k, table_index, physical_id, 1, operation);
              ++total_lookup_count;
              if (!calculate_prefill_byte_range_operation_checksum(
                      scenario_plan.scenario_seed, operation, layer, batch,
                      LlmPrefillKvDomain::K, first_byte, visit_bytes,
                      checksum.k.state_a, checksum.k.state_b) ||
                  checksum.k.exact_bytes_read >
                      std::numeric_limits<uint64_t>::max() - visit_bytes ||
                  checksum.k.span_count ==
                      std::numeric_limits<uint64_t>::max()) {
                result.reason_code =
                    LlmExecutorReason::EXPECTED_CHECKSUM_OVERFLOW;
                result.workers.clear();
                return result;
              }
              checksum.k.exact_bytes_read += visit_bytes;
              ++checksum.k.span_count;
            }

            for (size_t logical_block = assignment.first_logical_block;
                 logical_block < visited_block_end; ++logical_block) {
              const size_t table_index = table_row + logical_block;
              const uint32_t physical_id = resources.block_table[table_index];
              size_t first_byte = 0;
              size_t prefix_bytes = 0;
              if (physical_id >= layout.physical_blocks_per_layer ||
                  !NumericUtils::checked_multiply(
                      logical_block, layout.block_bytes, first_byte) ||
                  !NumericUtils::checked_multiply(
                      tile_end, record_bytes, prefix_bytes) ||
                  prefix_bytes <= first_byte ||
                  total_lookup_count ==
                      std::numeric_limits<uint64_t>::max()) {
                result.reason_code =
                    LlmExecutorReason::INVALID_DESCRIPTOR_LAYOUT;
                result.workers.clear();
                return result;
              }
              const size_t visit_bytes =
                  std::min(layout.block_bytes, prefix_bytes - first_byte);
              mix_llm_paged_lookup(
                  checksum.v, table_index, physical_id, 2, operation);
              ++total_lookup_count;
              if (!calculate_prefill_byte_range_operation_checksum(
                      scenario_plan.scenario_seed, operation, layer, batch,
                      LlmPrefillKvDomain::V, first_byte, visit_bytes,
                      checksum.v.state_a, checksum.v.state_b) ||
                  checksum.v.exact_bytes_read >
                      std::numeric_limits<uint64_t>::max() - visit_bytes ||
                  checksum.v.span_count ==
                      std::numeric_limits<uint64_t>::max()) {
                result.reason_code =
                    LlmExecutorReason::EXPECTED_CHECKSUM_OVERFLOW;
                result.workers.clear();
                return result;
              }
              checksum.v.exact_bytes_read += visit_bytes;
              ++checksum.v.span_count;
            }
          }
        }
      }
    }
    if (!add_checksum_bytes(
            checksum.weight.exact_bytes_read, total_weight_bytes) ||
        !add_checksum_bytes(checksum.k.exact_bytes_read, total_k_bytes) ||
        !add_checksum_bytes(checksum.v.exact_bytes_read, total_v_bytes)) {
      result.reason_code = LlmExecutorReason::EXPECTED_CHECKSUM_OVERFLOW;
      result.workers.clear();
      return result;
    }
  }

  uint64_t total_kv_bytes = 0;
  uint64_t total_lookup_bytes = 0;
  if (!add_checksum_bytes(total_k_bytes, total_kv_bytes) ||
      !add_checksum_bytes(total_v_bytes, total_kv_bytes) ||
      total_lookup_count >
          std::numeric_limits<uint64_t>::max() /
              Constants::LLM_KV_BLOCK_TABLE_ENTRY_BYTES) {
    result.reason_code = LlmExecutorReason::EXPECTED_CHECKSUM_OVERFLOW;
    result.workers.clear();
    return result;
  }
  total_lookup_bytes =
      total_lookup_count * Constants::LLM_KV_BLOCK_TABLE_ENTRY_BYTES;
  if (total_weight_bytes != scenario_plan.weight_read_bytes ||
      total_kv_bytes != scenario_plan.kv_read_bytes ||
      total_lookup_count != scenario_plan.layout_metadata_lookup_count ||
      total_lookup_bytes != scenario_plan.layout_metadata_read_bytes) {
    result.reason_code = LlmExecutorReason::INVALID_DESCRIPTOR_LAYOUT;
    result.workers.clear();
    return result;
  }
  result.run_checksum =
      fold_llm_worker_checksums(result.workers.data(), result.workers.size());
  result.valid = true;
  result.reason_code = LlmExecutorReason::VALID;
  return result;
}

LlmExpectedChecksumResult calculate_paged_expected_checksums(
    const LlmMemoryWorkPlan& model_plan,
    const LlmScenarioWorkPlan& scenario_plan,
    const LlmExecutionResources& resources) {
  LlmExpectedChecksumResult result;
  const LlmCpuExecutionPlan* const cpu_plan =
      get_llm_cpu_execution_plan(model_plan);
  if (cpu_plan == nullptr || !cpu_plan->paged.has_value() ||
      !materialized_resources_match_plan(model_plan, resources) ||
      !scenario_plan_matches_model(model_plan, scenario_plan)) {
    result.reason_code = LlmExecutorReason::INVALID_RESOURCES;
    return result;
  }
  const LlmKvLayoutPlan& layout = cpu_plan->paged->layout;
  result.workers.resize(cpu_plan->effective_workers);
  const uint64_t flags = llm_scenario_flags(scenario_plan.scenario);
  const bool reads_weight = (flags & kLlmScenarioFlagWeight) != 0;
  const bool reads_kv = (flags & kLlmScenarioFlagKv) != 0;
  uint64_t total_weight_bytes = 0;
  uint64_t total_k_bytes = 0;
  uint64_t total_v_bytes = 0;

  for (size_t worker_index = 0;
       worker_index < cpu_plan->effective_workers; ++worker_index) {
    LlmWorkerChecksum& checksum = result.workers[worker_index];
    checksum.weight = initial_llm_read_checksum(LlmChecksumComponent::Weight);
    checksum.k = initial_llm_read_checksum(LlmChecksumComponent::K);
    checksum.v = initial_llm_read_checksum(LlmChecksumComponent::V);
    const LlmWorkerWorkPlan& worker = cpu_plan->workers[worker_index];
    const LlmStaticSpanReference* const weight_references =
        resources.worker_weight_references(worker_index);
    if (weight_references == nullptr) {
      result.reason_code = LlmExecutorReason::INVALID_RESOURCES;
      result.workers.clear();
      return result;
    }

    for (size_t step = 0; step < scenario_plan.work_units; ++step) {
      for (size_t layer = 0; layer < layout.layer_count; ++layer) {
        if (reads_weight &&
            !absorb_reference(checksum.weight,
                              weight_references[layer])) {
          result.reason_code =
              LlmExecutorReason::EXPECTED_CHECKSUM_OVERFLOW;
          result.workers.clear();
          return result;
        }
        if (!reads_kv) {
          continue;
        }
        for (size_t batch = 0; batch < layout.batch_size; ++batch) {
          const LlmPagedKvAssignmentTemplate& assignment =
              worker.paged_assignments[layer * layout.batch_size + batch];
          if (assignment.block_count == 0) {
            continue;
          }
          const size_t assignment_end =
              assignment.first_logical_block + assignment.block_count;
          const bool owns_last =
              assignment_end == layout.blocks_per_sequence;
          const size_t table_row = batch * layout.blocks_per_sequence;
          if (owns_last) {
            const size_t logical_block = layout.blocks_per_sequence - 1;
            const uint32_t physical_id =
                resources.block_table[table_row + logical_block];
            mix_llm_paged_lookup(
                checksum.k, table_row + logical_block, physical_id, 0,
                step);
          }

          for (size_t block_offset = 0;
               block_offset < assignment.block_count; ++block_offset) {
            const size_t logical_block =
                assignment.first_logical_block + block_offset;
            const uint32_t physical_id =
                resources.block_table[table_row + logical_block];
            const size_t reference_index =
                layer * layout.physical_blocks_per_layer + physical_id;
            mix_llm_paged_lookup(
                checksum.k, table_row + logical_block, physical_id, 1,
                step);
            LlmStaticSpanReference adjusted;
            const LlmStaticSpanReference* reference =
                &resources.paged_k_block_references[reference_index];
            if (logical_block + 1 == layout.blocks_per_sequence) {
              if (!paged_append_adjusted_reference(
                      *reference, layout, physical_id,
                      model_plan.k_buffer_seed, scenario_plan.scenario_seed,
                      step, layer, batch, LlmChecksumComponent::K,
                      adjusted)) {
                result.reason_code =
                    LlmExecutorReason::EXPECTED_CHECKSUM_OVERFLOW;
                result.workers.clear();
                return result;
              }
              reference = &adjusted;
            }
            if (!absorb_reference(checksum.k, *reference)) {
              result.reason_code =
                  LlmExecutorReason::EXPECTED_CHECKSUM_OVERFLOW;
              result.workers.clear();
              return result;
            }
          }

          for (size_t block_offset = 0;
               block_offset < assignment.block_count; ++block_offset) {
            const size_t logical_block =
                assignment.first_logical_block + block_offset;
            const uint32_t physical_id =
                resources.block_table[table_row + logical_block];
            const size_t reference_index =
                layer * layout.physical_blocks_per_layer + physical_id;
            mix_llm_paged_lookup(
                checksum.v, table_row + logical_block, physical_id, 2,
                step);
            LlmStaticSpanReference adjusted;
            const LlmStaticSpanReference* reference =
                &resources.paged_v_block_references[reference_index];
            if (logical_block + 1 == layout.blocks_per_sequence) {
              if (!paged_append_adjusted_reference(
                      *reference, layout, physical_id,
                      model_plan.v_buffer_seed, scenario_plan.scenario_seed,
                      step, layer, batch, LlmChecksumComponent::V,
                      adjusted)) {
                result.reason_code =
                    LlmExecutorReason::EXPECTED_CHECKSUM_OVERFLOW;
                result.workers.clear();
                return result;
              }
              reference = &adjusted;
            }
            if (!absorb_reference(checksum.v, *reference)) {
              result.reason_code =
                  LlmExecutorReason::EXPECTED_CHECKSUM_OVERFLOW;
              result.workers.clear();
              return result;
            }
          }
        }
      }
    }
    if (!add_checksum_bytes(checksum.weight.exact_bytes_read,
                            total_weight_bytes) ||
        !add_checksum_bytes(checksum.k.exact_bytes_read, total_k_bytes) ||
        !add_checksum_bytes(checksum.v.exact_bytes_read, total_v_bytes)) {
      result.reason_code = LlmExecutorReason::EXPECTED_CHECKSUM_OVERFLOW;
      result.workers.clear();
      return result;
    }
  }
  uint64_t total_kv_bytes = 0;
  if (!add_checksum_bytes(total_k_bytes, total_kv_bytes) ||
      !add_checksum_bytes(total_v_bytes, total_kv_bytes) ||
      total_weight_bytes != scenario_plan.weight_read_bytes ||
      total_kv_bytes != scenario_plan.kv_read_bytes) {
    result.reason_code = LlmExecutorReason::INVALID_DESCRIPTOR_LAYOUT;
    result.workers.clear();
    return result;
  }
  result.run_checksum =
      fold_llm_worker_checksums(result.workers.data(), result.workers.size());
  result.valid = true;
  result.reason_code = LlmExecutorReason::VALID;
  return result;
}

bool validate_paged_post_execution(
    const LlmMemoryWorkPlan& model_plan,
    const LlmScenarioWorkPlan& scenario_plan,
    const LlmExecutionResources& resources) noexcept {
  const LlmCpuExecutionPlan* const cpu_plan =
      get_llm_cpu_execution_plan(model_plan);
  if (cpu_plan == nullptr || !cpu_plan->paged.has_value() ||
      resources.block_table == nullptr || !resources.buffers.complete()) {
    return false;
  }
  const LlmKvLayoutPlan& layout = cpu_plan->paged->layout;
  const uint8_t* const k =
      static_cast<const uint8_t*>(resources.buffers.k.get());
  const uint8_t* const v =
      static_cast<const uint8_t*>(resources.buffers.v.get());
  const bool wrote_kv =
      (llm_scenario_flags(scenario_plan.scenario) & kLlmScenarioFlagKv) != 0;
  const uint64_t final_work_unit = scenario_plan.work_units - 1;
  for (size_t layer = 0; layer < layout.layer_count; ++layer) {
    for (size_t batch = 0; batch < layout.batch_size; ++batch) {
      const size_t table_index =
          batch * layout.blocks_per_sequence +
          (layout.blocks_per_sequence - 1);
      const uint32_t physical_id = resources.block_table[table_index];
      if (physical_id >= layout.physical_blocks_per_layer) {
        return false;
      }
      const size_t global_block =
          layer * layout.physical_blocks_per_layer + physical_id;
      const size_t block_offset = global_block * layout.block_bytes;
      for (size_t record_byte = 0;
           record_byte < layout.k_or_v_record_bytes_per_layer;
           ++record_byte) {
        const size_t block_byte_offset =
            layout.decode_append_offset_in_last_block + record_byte;
        const size_t byte_offset = block_offset + block_byte_offset;
        const uint8_t expected_k =
            wrote_kv
                ? append_byte(scenario_plan.scenario_seed,
                              final_work_unit, layer, batch, record_byte,
                              LlmChecksumComponent::K)
                : static_cast<uint8_t>(paged_pattern_byte_word(
                      model_plan.k_buffer_seed, layer, physical_id,
                      block_byte_offset, 1));
        const uint8_t expected_v =
            wrote_kv
                ? append_byte(scenario_plan.scenario_seed,
                              final_work_unit, layer, batch, record_byte,
                              LlmChecksumComponent::V)
                : static_cast<uint8_t>(paged_pattern_byte_word(
                      model_plan.v_buffer_seed, layer, physical_id,
                      block_byte_offset, 1));
        if (k[byte_offset] != expected_k || v[byte_offset] != expected_v) {
          return false;
        }
      }
      for (size_t padding_byte = layout.last_block_valid_bytes;
           padding_byte < layout.block_bytes; ++padding_byte) {
        const uint8_t expected_k = static_cast<uint8_t>(
            paged_pattern_byte_word(model_plan.k_buffer_seed, layer,
                                    physical_id, padding_byte, 1));
        const uint8_t expected_v = static_cast<uint8_t>(
            paged_pattern_byte_word(model_plan.v_buffer_seed, layer,
                                    physical_id, padding_byte, 1));
        if (k[block_offset + padding_byte] != expected_k ||
            v[block_offset + padding_byte] != expected_v) {
          return false;
        }
      }
    }
  }
  return true;
}

bool validate_paged_prefill_post_execution(
    const LlmMemoryWorkPlan& model_plan,
    const LlmScenarioWorkPlan& scenario_plan,
    const LlmExecutionResources& resources) noexcept {
  const LlmCpuExecutionPlan* const cpu_plan =
      get_llm_cpu_execution_plan(model_plan);
  if (model_plan.phase != LlmPhase::Prefill ||
      model_plan.kv_layout != LlmKvLayout::Paged ||
      !model_plan.geometry.prefill.has_value() || cpu_plan == nullptr ||
      !cpu_plan->prefill.has_value() || !cpu_plan->paged.has_value() ||
      resources.block_table == nullptr || !resources.buffers.complete() ||
      scenario_plan.work_units == 0) {
    return false;
  }
  const LlmKvLayoutPlan& layout = cpu_plan->paged->layout;
  const uint8_t* const k =
      static_cast<const uint8_t*>(resources.buffers.k.get());
  const uint8_t* const v =
      static_cast<const uint8_t*>(resources.buffers.v.get());
  const bool wrote_kv =
      (llm_scenario_flags(scenario_plan.scenario) &
       kLlmScenarioFlagKv) != 0;
  const size_t final_operation = scenario_plan.work_units - 1;

  for (size_t layer = 0; layer < layout.layer_count; ++layer) {
    for (size_t batch = 0; batch < layout.batch_size; ++batch) {
      const size_t table_row = batch * layout.blocks_per_sequence;
      for (size_t logical_block = 0;
           logical_block < layout.blocks_per_sequence; ++logical_block) {
        const size_t table_index = table_row + logical_block;
        if (table_index >= resources.block_table_entries) {
          return false;
        }
        const uint32_t physical_id = resources.block_table[table_index];
        if (physical_id >= layout.physical_blocks_per_layer) {
          return false;
        }
        size_t global_block = 0;
        size_t physical_offset = 0;
        size_t logical_offset = 0;
        if (!NumericUtils::checked_multiply(
                layer, layout.physical_blocks_per_layer, global_block) ||
            !NumericUtils::checked_add(
                global_block, static_cast<size_t>(physical_id),
                global_block) ||
            !NumericUtils::checked_multiply(
                global_block, layout.block_bytes, physical_offset) ||
            !NumericUtils::checked_multiply(
                logical_block, layout.block_bytes, logical_offset)) {
          return false;
        }
        const size_t valid_bytes =
            logical_block + 1 == layout.blocks_per_sequence
                ? layout.last_block_valid_bytes
                : layout.block_bytes;
        if (valid_bytes == 0) {
          return false;
        }
        if (wrote_kv) {
          size_t logical_end = 0;
          if (!NumericUtils::checked_add(
                  logical_offset, valid_bytes, logical_end)) {
            return false;
          }
          const size_t midpoint = logical_offset + valid_bytes / 2;
          const std::array<size_t, 3> representative_words = {
              logical_offset / sizeof(uint64_t),
              midpoint / sizeof(uint64_t),
              (logical_end - 1) / sizeof(uint64_t)};
          for (size_t logical_word : representative_words) {
            const size_t word_start = logical_word * sizeof(uint64_t);
            const size_t sample_start = std::max(logical_offset, word_start);
            size_t sample_end = 0;
            if (!NumericUtils::checked_add(
                    sample_start,
                    sizeof(uint64_t) - sample_start % sizeof(uint64_t),
                    sample_end)) {
              return false;
            }
            sample_end = std::min(logical_end, sample_end);
            for (size_t logical_byte = sample_start;
                 logical_byte < sample_end; ++logical_byte) {
              const size_t block_byte = logical_byte - logical_offset;
              const size_t mapping_offset = physical_offset + block_byte;
              const uint8_t expected_k = llm_prefill_affine64_byte(
                  scenario_plan.scenario_seed, final_operation, layer, batch,
                  LlmPrefillKvDomain::K, logical_byte);
              const uint8_t expected_v = llm_prefill_affine64_byte(
                  scenario_plan.scenario_seed, final_operation, layer, batch,
                  LlmPrefillKvDomain::V, logical_byte);
              if (k[mapping_offset] != expected_k ||
                  v[mapping_offset] != expected_v) {
                return false;
              }
            }
          }
        }
        for (size_t padding_byte = valid_bytes;
             padding_byte < layout.block_bytes; ++padding_byte) {
          const uint8_t expected_k = static_cast<uint8_t>(
              paged_pattern_byte_word(
                  model_plan.k_buffer_seed, layer, physical_id,
                  padding_byte, 1));
          const uint8_t expected_v = static_cast<uint8_t>(
              paged_pattern_byte_word(
                  model_plan.v_buffer_seed, layer, physical_id,
                  padding_byte, 1));
          if (k[physical_offset + padding_byte] != expected_k ||
              v[physical_offset + padding_byte] != expected_v) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

bool validate_prefill_post_execution(
    const LlmMemoryWorkPlan& model_plan,
    const LlmScenarioWorkPlan& scenario_plan,
    const LlmExecutionResources& resources) noexcept {
  const LlmCpuExecutionPlan* const cpu_plan =
      get_llm_cpu_execution_plan(model_plan);
  if (model_plan.phase != LlmPhase::Prefill ||
      model_plan.kv_layout != LlmKvLayout::Contiguous ||
      !model_plan.geometry.prefill.has_value() ||
      cpu_plan == nullptr || !cpu_plan->prefill.has_value() ||
      !resources.buffers.complete() || scenario_plan.work_units == 0) {
    return false;
  }
  if ((llm_scenario_flags(scenario_plan.scenario) &
       kLlmScenarioFlagKv) == 0) {
    return true;
  }
  const uint8_t* const k =
      static_cast<const uint8_t*>(resources.buffers.k.get());
  const uint8_t* const v =
      static_cast<const uint8_t*>(resources.buffers.v.get());
  const size_t final_operation = scenario_plan.work_units - 1;
  const size_t scenario_index = static_cast<size_t>(scenario_plan.scenario);
  if (scenario_index >= kLlmScenarioCount) {
    return false;
  }
  const size_t per_scenario =
      cpu_plan->prefill->sequence_descriptors_per_scenario_per_worker;
  const size_t scenario_base = scenario_index * per_scenario;
  for (size_t layer = 0; layer < model_plan.geometry.layer_count; ++layer) {
    for (size_t batch = 0; batch < model_plan.geometry.batch_size; ++batch) {
      size_t row = 0;
      size_t row_offset = 0;
      if (!NumericUtils::checked_multiply(layer,
                                          model_plan.geometry.batch_size,
                                          row) ||
          !NumericUtils::checked_add(row, batch, row) ||
          !NumericUtils::checked_multiply(
              row, model_plan.geometry.k_or_v_sequence_visible_bytes,
              row_offset)) {
        return false;
      }
      for (size_t worker = 0; worker < cpu_plan->effective_workers;
           ++worker) {
        const LlmPrefillKvSequenceRangeTemplate& sequence =
            cpu_plan->workers[worker]
                .prefill_sequences[scenario_base + row];
        if (sequence.owned_token_count == 0) {
          continue;
        }
        size_t first_byte = 0;
        size_t owned_bytes = 0;
        size_t end_byte = 0;
        if (sequence.layer_index != layer ||
            sequence.batch_sequence_index != batch ||
            !NumericUtils::checked_multiply(
                sequence.first_token,
                model_plan.geometry.k_or_v_record_bytes_per_layer,
                first_byte) ||
            !NumericUtils::checked_multiply(
                sequence.owned_token_count,
                model_plan.geometry.k_or_v_record_bytes_per_layer,
                owned_bytes) ||
            !NumericUtils::checked_add(first_byte, owned_bytes, end_byte) ||
            owned_bytes == 0 ||
            end_byte >
                model_plan.geometry.k_or_v_sequence_visible_bytes) {
          return false;
        }
        const size_t midpoint = first_byte + owned_bytes / 2;
        const std::array<size_t, 3> representative_words = {
            first_byte / sizeof(uint64_t),
            midpoint / sizeof(uint64_t),
            (end_byte - 1) / sizeof(uint64_t)};
        for (size_t logical_word : representative_words) {
          const size_t word_start = logical_word * sizeof(uint64_t);
          const size_t sample_start = std::max(first_byte, word_start);
          const size_t sample_end =
              std::min(end_byte, word_start + sizeof(uint64_t));
          for (size_t logical_byte = sample_start;
               logical_byte < sample_end; ++logical_byte) {
            const size_t offset = row_offset + logical_byte;
            if (k[offset] != llm_prefill_affine64_byte(
                                 scenario_plan.scenario_seed,
                                 final_operation, layer, batch,
                                 LlmPrefillKvDomain::K, logical_byte) ||
                v[offset] != llm_prefill_affine64_byte(
                                 scenario_plan.scenario_seed,
                                 final_operation, layer, batch,
                                 LlmPrefillKvDomain::V, logical_byte)) {
              return false;
            }
          }
        }
      }
    }
  }
  return true;
}

/** Restore only the mutable decode append slots before an untimed task. */
bool reset_paged_append_slots(
    const LlmMemoryWorkPlan& model_plan,
    const LlmExecutionResources& resources) noexcept {
  if (model_plan.kv_layout != LlmKvLayout::Paged ||
      model_plan.phase != LlmPhase::Decode) {
    return true;
  }
  const LlmCpuExecutionPlan* const cpu_plan =
      get_llm_cpu_execution_plan(model_plan);
  if (cpu_plan == nullptr || !cpu_plan->paged.has_value() ||
      resources.block_table == nullptr || !resources.buffers.complete()) {
    return false;
  }
  const LlmKvLayoutPlan& layout = cpu_plan->paged->layout;
  size_t append_end_in_block = 0;
  if (layout.blocks_per_sequence == 0 ||
      layout.physical_blocks_per_layer == 0 || layout.block_bytes == 0 ||
      layout.k_or_v_record_bytes_per_layer == 0 ||
      !NumericUtils::checked_add(
          layout.decode_append_offset_in_last_block,
          layout.k_or_v_record_bytes_per_layer, append_end_in_block) ||
      append_end_in_block > layout.last_block_valid_bytes ||
      append_end_in_block > layout.block_bytes) {
    return false;
  }

  uint8_t* const k =
      static_cast<uint8_t*>(resources.buffers.k.get());
  uint8_t* const v =
      static_cast<uint8_t*>(resources.buffers.v.get());
  for (size_t layer = 0; layer < layout.layer_count; ++layer) {
    for (size_t batch = 0; batch < layout.batch_size; ++batch) {
      size_t table_index = 0;
      if (!NumericUtils::checked_multiply(
              batch, layout.blocks_per_sequence, table_index) ||
          !NumericUtils::checked_add(
              table_index, layout.blocks_per_sequence - 1,
              table_index) ||
          table_index >= resources.block_table_entries) {
        return false;
      }
      const uint32_t physical_id = resources.block_table[table_index];
      size_t global_block = 0;
      size_t block_offset = 0;
      size_t append_offset = 0;
      size_t append_end = 0;
      if (physical_id >= layout.physical_blocks_per_layer ||
          !NumericUtils::checked_multiply(
              layer, layout.physical_blocks_per_layer, global_block) ||
          !NumericUtils::checked_add(
              global_block, static_cast<size_t>(physical_id),
              global_block) ||
          !NumericUtils::checked_multiply(
              global_block, layout.block_bytes, block_offset) ||
          !NumericUtils::checked_add(
              block_offset, layout.decode_append_offset_in_last_block,
              append_offset) ||
          !NumericUtils::checked_add(
              append_offset, layout.k_or_v_record_bytes_per_layer,
              append_end) ||
          append_end > model_plan.geometry.k_mapping_bytes ||
          append_end > model_plan.geometry.v_mapping_bytes) {
        return false;
      }

      size_t record_offset = 0;
      while (record_offset < layout.k_or_v_record_bytes_per_layer) {
        const size_t write_bytes =
            std::min(sizeof(uint64_t),
                     layout.k_or_v_record_bytes_per_layer - record_offset);
        const size_t block_byte_offset =
            layout.decode_append_offset_in_last_block + record_offset;
        const uint64_t k_word = paged_pattern_byte_word(
            model_plan.k_buffer_seed, layer, physical_id,
            block_byte_offset, write_bytes);
        const uint64_t v_word = paged_pattern_byte_word(
            model_plan.v_buffer_seed, layer, physical_id,
            block_byte_offset, write_bytes);
        std::memcpy(k + append_offset + record_offset, &k_word,
                    write_bytes);
        std::memcpy(v + append_offset + record_offset, &v_word,
                    write_bytes);
        record_offset += write_bytes;
      }
    }
  }
  return true;
}

bool equal_component(const LlmReadChecksumComponent& lhs, const LlmReadChecksumComponent& rhs) noexcept {
  return lhs.state_a == rhs.state_a && lhs.state_b == rhs.state_b && lhs.exact_bytes_read == rhs.exact_bytes_read &&
         lhs.span_count == rhs.span_count;
}

bool equal_worker_checksum(const LlmWorkerChecksum& lhs, const LlmWorkerChecksum& rhs) noexcept {
  return equal_component(lhs.weight, rhs.weight) && equal_component(lhs.k, rhs.k) && equal_component(lhs.v, rhs.v);
}

bool production_kernel_invoke(void*, const LlmKernelInvocation& invocation) noexcept {
  if (invocation.output == nullptr) {
    return false;
  }
  if (invocation.phase == LlmPhase::Prefill &&
      invocation.kv_layout == LlmKvLayout::Paged) {
    llm_prefill_memory_paged_asm(
        invocation.paged_prefill_layers,
        invocation.paged_prefill_assignments, invocation.layer_count,
        invocation.work_unit_count, invocation.scenario_flags,
        invocation.scenario_seed, invocation.output);
    return true;
  }
  if (invocation.phase == LlmPhase::Prefill) {
    llm_prefill_memory_asm(
        invocation.prefill_layers, invocation.prefill_sequences,
        invocation.layer_count, invocation.work_unit_count,
        invocation.scenario_flags, invocation.scenario_seed,
        invocation.output);
    return true;
  }
  if (invocation.kv_layout == LlmKvLayout::Paged) {
    llm_decode_memory_paged_asm(
        invocation.paged_layers, invocation.paged_assignments,
        invocation.layer_count, invocation.work_unit_count,
        invocation.scenario_flags, invocation.scenario_seed,
        invocation.output);
    return true;
  }
  llm_decode_memory_asm(invocation.layers, invocation.sequences, invocation.layer_count, invocation.work_unit_count,
                        invocation.scenario_flags, invocation.scenario_seed, invocation.output);
  return true;
}

void observe_event(const LlmExecutorTestControl* control, LlmExecutorEvent event, size_t worker_index) noexcept {
  if (control != nullptr && control->observe_event != nullptr) {
    control->observe_event(control->event_context, event, worker_index);
  }
}

int set_worker_qos(const LlmExecutorTestControl* control, size_t worker_index) noexcept {
  if (control != nullptr && control->set_worker_qos != nullptr) {
    return control->set_worker_qos(control->qos_context, worker_index);
  }
  return pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
}

void join_threads(std::vector<std::thread>& threads) noexcept {
  for (std::thread& thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }
}

}  // namespace

bool LlmBufferSet::complete() const noexcept { return weight != nullptr && k != nullptr && v != nullptr; }

const LlmLayerDescriptor* LlmExecutionResources::worker_layers(size_t worker_index) const noexcept {
  if (worker_index >= worker_count || layer_descriptors == nullptr) {
    return nullptr;
  }
  return layer_descriptors.get() + worker_index * layer_descriptors_per_worker;
}

const LlmKvSequenceDescriptor* LlmExecutionResources::worker_sequences(size_t worker_index) const noexcept {
  if (worker_index >= worker_count || sequence_descriptors == nullptr) {
    return nullptr;
  }
  return sequence_descriptors.get() + worker_index * sequence_descriptors_per_worker;
}

const LlmStaticSpanReference* LlmExecutionResources::worker_weight_references(size_t worker_index) const noexcept {
  if (worker_index >= worker_count || weight_references == nullptr) {
    return nullptr;
  }
  return weight_references.get() + worker_index * layer_descriptors_per_worker;
}

const LlmStaticSpanReference* LlmExecutionResources::worker_k_references(size_t worker_index) const noexcept {
  if (worker_index >= worker_count || k_references == nullptr) {
    return nullptr;
  }
  return k_references.get() + worker_index * sequence_descriptors_per_worker;
}

const LlmStaticSpanReference* LlmExecutionResources::worker_v_references(size_t worker_index) const noexcept {
  if (worker_index >= worker_count || v_references == nullptr) {
    return nullptr;
  }
  return v_references.get() + worker_index * sequence_descriptors_per_worker;
}

const LlmPagedLayerDescriptor* LlmExecutionResources::worker_paged_layers(
    size_t worker_index) const noexcept {
  if (worker_index >= worker_count || paged_layer_descriptors == nullptr) {
    return nullptr;
  }
  return paged_layer_descriptors.get() +
         worker_index * layer_descriptors_per_worker;
}

const LlmPagedKvAssignmentDescriptor*
LlmExecutionResources::worker_paged_assignments(
    size_t worker_index) const noexcept {
  if (worker_index >= worker_count ||
      paged_assignment_descriptors == nullptr) {
    return nullptr;
  }
  return paged_assignment_descriptors.get() +
         worker_index * sequence_descriptors_per_worker;
}

const LlmPrefillLayerDescriptor*
LlmExecutionResources::worker_prefill_layers(size_t worker_index) const noexcept {
  if (worker_index >= worker_count || prefill_layer_descriptors == nullptr) {
    return nullptr;
  }
  return prefill_layer_descriptors.get() +
         worker_index * layer_descriptors_per_worker;
}

const LlmPrefillKvSequenceDescriptor*
LlmExecutionResources::worker_prefill_sequences(
    size_t worker_index, LlmScenario scenario) const noexcept {
  const size_t scenario_index = static_cast<size_t>(scenario);
  if (worker_index >= worker_count ||
      scenario_index >= kLlmScenarioCount ||
      prefill_sequence_descriptors == nullptr ||
      sequence_descriptors_per_worker % kLlmScenarioCount != 0) {
    return nullptr;
  }
  const size_t per_scenario =
      sequence_descriptors_per_worker / kLlmScenarioCount;
  return prefill_sequence_descriptors.get() +
         worker_index * sequence_descriptors_per_worker +
         scenario_index * per_scenario;
}

const LlmPagedPrefillLayerDescriptor*
LlmExecutionResources::worker_paged_prefill_layers(
    size_t worker_index) const noexcept {
  if (worker_index >= worker_count ||
      paged_prefill_layer_descriptors == nullptr) {
    return nullptr;
  }
  return paged_prefill_layer_descriptors.get() +
         worker_index * layer_descriptors_per_worker;
}

const LlmPagedPrefillKvAssignmentDescriptor*
LlmExecutionResources::worker_paged_prefill_assignments(
    size_t worker_index, LlmScenario scenario) const noexcept {
  const size_t scenario_index = static_cast<size_t>(scenario);
  if (worker_index >= worker_count ||
      scenario_index >= kLlmScenarioCount ||
      paged_prefill_assignment_descriptors == nullptr ||
      sequence_descriptors_per_worker % kLlmScenarioCount != 0) {
    return nullptr;
  }
  const size_t per_scenario =
      sequence_descriptors_per_worker / kLlmScenarioCount;
  return paged_prefill_assignment_descriptors.get() +
         worker_index * sequence_descriptors_per_worker +
         scenario_index * per_scenario;
}

uint64_t llm_scenario_flags(LlmScenario scenario) noexcept {
  switch (scenario) {
    case LlmScenario::WeightsOnly:
      return kLlmScenarioFlagWeight;
    case LlmScenario::KvOnly:
      return kLlmScenarioFlagKv;
    case LlmScenario::Mixed:
      return kLlmScenarioFlagMixed;
  }
  return 0;
}

uint64_t llm_buffer_pattern_word(uint64_t buffer_domain_seed, uint64_t absolute_mapping_word_index) noexcept {
  return buffer_domain_seed + kBufferPatternMultiplier * (absolute_mapping_word_index + 1);
}

uint64_t llm_paged_buffer_pattern_word(
    uint64_t buffer_domain_seed, uint64_t layer_index,
    uint64_t physical_block_id, uint64_t block_word_index) noexcept {
  return buffer_domain_seed +
         kPagedPatternLayerMultiplier * (layer_index + 1) +
         kPagedPatternPhysicalMultiplier * (physical_block_id + 1) +
         kPagedPatternWordMultiplier * (block_word_index + 1);
}

uint64_t llm_append_word(uint64_t scenario_seed, uint64_t task_local_work_unit, uint64_t layer_index,
                         uint64_t batch_sequence_index, uint64_t record_word_index,
                         LlmChecksumComponent component) noexcept {
  return scenario_seed +
         kAppendStepMultiplier * (task_local_work_unit + 1) +
         kAppendLayerMultiplier * (layer_index + 1) +
         kAppendBatchMultiplier * (batch_sequence_index + 1) +
         kAppendWordMultiplier * (record_word_index + 1) +
         append_domain(component);
}

LlmReadChecksumComponent initial_llm_read_checksum(LlmChecksumComponent component) noexcept {
  const uint64_t domain = checksum_domain(component);
  return {kChecksumInitialA ^ domain, kChecksumInitialB + domain, 0, 0};
}

void mix_llm_paged_lookup(LlmReadChecksumComponent& checksum,
                          uint64_t global_logical_index,
                          uint32_t physical_block_id,
                          uint64_t semantic_visit_kind,
                          uint64_t work_unit_ordinal) noexcept {
  const uint64_t term =
      (global_logical_index + 1) *
          (static_cast<uint64_t>(physical_block_id) + 1) +
      (semantic_visit_kind + 1) * kPagedLookupVisitMultiplier +
      (work_unit_ordinal + 1) * kPagedLookupWorkUnitMultiplier;
  checksum.state_a = rotate_left(
      checksum.state_a + term + kPagedLookupLogicalMultiplier, 13);
  checksum.state_b = rotate_left(
      checksum.state_b ^ (term + kPagedLookupStateBStep), 31);
}

LlmRunChecksum fold_llm_worker_checksums(const LlmWorkerChecksum* workers, size_t worker_count) noexcept {
  LlmRunChecksum result{kRunInitialA, kRunInitialB};
  if (workers == nullptr) {
    return result;
  }
  uint64_t tuple_ordinal = 0;
  for (size_t worker_index = 0; worker_index < worker_count; ++worker_index) {
    const LlmReadChecksumComponent* const components[] = {&workers[worker_index].weight, &workers[worker_index].k,
                                                          &workers[worker_index].v};
    for (const LlmReadChecksumComponent* component : components) {
      result.state_a =
          rotate_left(result.state_a + component->state_a + kAppendStepMultiplier * (tuple_ordinal + 1), 23);
      result.state_b =
          rotate_left(result.state_b + component->state_b + component->exact_bytes_read +
                          kAppendWordMultiplier * component->span_count + kAppendBatchMultiplier * (tuple_ordinal + 1),
                      41);
      ++tuple_ordinal;
    }
  }
  return result;
}

LlmExecutorAuxiliaryEstimate calculate_llm_executor_auxiliary_estimate(
    const LlmAuxiliaryPreflightView& preflight) noexcept {
  LlmExecutorAuxiliaryEstimate estimate;
  if (!preflight.valid || preflight.backend != LlmMemoryBackend::Cpu ||
      preflight.effective_workers == 0) {
    estimate.reason_code = LlmExecutorReason::INVALID_WORK_PLAN;
    return estimate;
  }

  size_t sequence_reference_count = 0;
  size_t reference_count = 0;
  if (!NumericUtils::checked_multiply(
          preflight.k_or_v_static_reference_count, 2,
                                      sequence_reference_count) ||
      !NumericUtils::checked_add(preflight.total_layer_descriptors,
                                 sequence_reference_count,
                                 reference_count) ||
      !NumericUtils::checked_multiply(reference_count, sizeof(LlmStaticSpanReference),
                                      estimate.static_reference_bytes) ||
      !NumericUtils::checked_multiply(preflight.effective_workers,
                                      sizeof(LlmWorkerChecksum),
                                      estimate.expected_checksum_bytes)) {
    return estimate;
  }
  estimate.actual_checksum_bytes = estimate.expected_checksum_bytes;
  if (!NumericUtils::checked_multiply(2, sizeof(LlmRunChecksum), estimate.run_checksum_bytes) ||
      !NumericUtils::checked_multiply(preflight.effective_workers,
                                      sizeof(uint8_t),
                                      estimate.worker_status_bytes) ||
      !NumericUtils::checked_multiply(preflight.effective_workers,
                                      sizeof(std::thread),
                                      estimate.thread_handle_bytes)) {
    return estimate;
  }

  if (!checked_add_to(estimate.static_reference_bytes, estimate.checksum_auxiliary_bytes) ||
      !checked_add_to(estimate.expected_checksum_bytes, estimate.checksum_auxiliary_bytes) ||
      !checked_add_to(estimate.actual_checksum_bytes, estimate.checksum_auxiliary_bytes) ||
      !checked_add_to(estimate.run_checksum_bytes, estimate.checksum_auxiliary_bytes) ||
      !checked_add_to(estimate.worker_status_bytes, estimate.orchestration_auxiliary_bytes) ||
      !checked_add_to(estimate.thread_handle_bytes, estimate.orchestration_auxiliary_bytes) ||
      !NumericUtils::checked_add(estimate.checksum_auxiliary_bytes, estimate.orchestration_auxiliary_bytes,
                                 estimate.total_auxiliary_bytes)) {
    estimate = {};
    estimate.reason_code = LlmExecutorReason::AUXILIARY_BYTES_OVERFLOW;
    return estimate;
  }
  estimate.valid = true;
  estimate.reason_code = LlmExecutorReason::VALID;
  return estimate;
}

LlmExecutorAuxiliaryEstimate calculate_llm_executor_auxiliary_estimate(
    const LlmMemoryWorkPlan& plan) noexcept {
  const LlmCpuExecutionPlan* const cpu_plan =
      get_llm_cpu_execution_plan(plan);
  if (cpu_plan == nullptr || !validate_work_plan_layout(plan)) {
    LlmExecutorAuxiliaryEstimate estimate;
    estimate.reason_code = LlmExecutorReason::INVALID_WORK_PLAN;
    return estimate;
  }
  LlmAuxiliaryPreflightView preflight;
  preflight.valid = true;
  preflight.backend = plan.backend;
  preflight.kv_layout = plan.kv_layout;
  preflight.effective_workers = cpu_plan->effective_workers;
  preflight.total_layer_descriptors =
      cpu_plan->total_layer_descriptors;
  preflight.total_sequence_descriptors =
      cpu_plan->total_sequence_descriptors;
  preflight.k_or_v_static_reference_count =
      plan.phase == LlmPhase::Prefill &&
              plan.kv_layout == LlmKvLayout::Contiguous
          ? 0
      : cpu_plan->paged.has_value()
          ? cpu_plan->paged->layout.total_physical_blocks
          : cpu_plan->total_sequence_descriptors;
  return calculate_llm_executor_auxiliary_estimate(preflight);
}

LlmBufferAllocationResult allocate_llm_buffers(const LlmMemoryWorkPlan& plan, LlmBufferSet& output) noexcept {
  LlmBufferAllocationResult result;
  try {
    if (!buffer_output_is_empty(output)) {
      result.reason_code = LlmExecutorReason::OUTPUT_NOT_EMPTY;
      return result;
    }
    if (!validate_work_plan_layout(plan)) {
      result.reason_code = LlmExecutorReason::INVALID_WORK_PLAN;
      return result;
    }
    result.auxiliary = calculate_llm_executor_auxiliary_estimate(plan);
    if (!result.auxiliary.valid) {
      result.reason_code = result.auxiliary.reason_code;
      return result;
    }
    if (!calculate_budget_with_executor_auxiliary(plan, result.auxiliary, result.memory_budget)) {
      result.reason_code = result.memory_budget.request.valid ? LlmExecutorReason::MEMORY_BUDGET_EXCEEDED
                                                              : LlmExecutorReason::AUXILIARY_BYTES_OVERFLOW;
      return result;
    }

    LlmBufferSet candidate;
    candidate.weight = allocate_buffer(plan.geometry.active_weight_bytes_per_work_unit, "LLM weight buffer");
    if (candidate.weight == nullptr) {
      result.reason_code = LlmExecutorReason::WEIGHT_MAPPING_FAILED;
      return result;
    }
    candidate.k = allocate_buffer(plan.geometry.k_mapping_bytes, "LLM K buffer");
    if (candidate.k == nullptr) {
      result.reason_code = LlmExecutorReason::K_MAPPING_FAILED;
      return result;
    }
    candidate.v = allocate_buffer(plan.geometry.v_mapping_bytes, "LLM V buffer");
    if (candidate.v == nullptr) {
      result.reason_code = LlmExecutorReason::V_MAPPING_FAILED;
      return result;
    }

    output = std::move(candidate);
    result.valid = true;
    result.reason_code = LlmExecutorReason::VALID;
    return result;
  } catch (...) {
    result.valid = false;
    if (result.reason_code == LlmExecutorReason::VALID) {
      result.reason_code = LlmExecutorReason::WEIGHT_MAPPING_FAILED;
    }
    return result;
  }
}

LlmResourcePreparationResult prepare_llm_execution_resources(const LlmMemoryWorkPlan& plan,
                                                             LlmExecutionResources& output) noexcept {
  LlmResourcePreparationResult result;
  try {
    const LlmCpuExecutionPlan* const cpu_plan =
        get_llm_cpu_execution_plan(plan);
    if (!resource_output_is_empty(output)) {
      result.reason_code = LlmExecutorReason::OUTPUT_NOT_EMPTY;
      return result;
    }
    if (cpu_plan == nullptr || !validate_work_plan_layout(plan)) {
      result.reason_code = LlmExecutorReason::INVALID_WORK_PLAN;
      return result;
    }

    LlmExecutionResources candidate;
    const LlmBufferAllocationResult allocation = allocate_llm_buffers(plan, candidate.buffers);
    result.auxiliary = allocation.auxiliary;
    result.memory_budget = allocation.memory_budget;
    if (!allocation.valid) {
      result.reason_code = allocation.reason_code;
      return result;
    }

    candidate.weight_references.reset(
        new (std::nothrow)
            LlmStaticSpanReference[cpu_plan->total_layer_descriptors]);
    if (plan.phase == LlmPhase::Prefill &&
        plan.kv_layout == LlmKvLayout::Paged &&
        cpu_plan->prefill.has_value() && cpu_plan->paged.has_value()) {
      candidate.paged_prefill_layer_descriptors.reset(
          new (std::nothrow) LlmPagedPrefillLayerDescriptor[
              cpu_plan->total_layer_descriptors]);
      candidate.paged_prefill_assignment_descriptors.reset(
          new (std::nothrow) LlmPagedPrefillKvAssignmentDescriptor[
              cpu_plan->total_sequence_descriptors]);
      candidate.paged_k_block_references.reset(
          new (std::nothrow) LlmStaticSpanReference[
              cpu_plan->paged->layout.total_physical_blocks]);
      candidate.paged_v_block_references.reset(
          new (std::nothrow) LlmStaticSpanReference[
              cpu_plan->paged->layout.total_physical_blocks]);
      candidate.block_table = cpu_plan->paged->block_table();
      candidate.block_table_entries =
          cpu_plan->paged->layout.block_table_entries;
      candidate.paged_block_reference_count =
          cpu_plan->paged->layout.total_physical_blocks;
    } else if (plan.phase == LlmPhase::Prefill &&
               cpu_plan->prefill.has_value()) {
      candidate.prefill_layer_descriptors.reset(
          new (std::nothrow)
              LlmPrefillLayerDescriptor[cpu_plan->total_layer_descriptors]);
      candidate.prefill_sequence_descriptors.reset(
          new (std::nothrow) LlmPrefillKvSequenceDescriptor[
              cpu_plan->total_sequence_descriptors]);
    } else if (plan.kv_layout == LlmKvLayout::Paged &&
        cpu_plan->paged.has_value()) {
      candidate.paged_layer_descriptors.reset(
          new (std::nothrow)
              LlmPagedLayerDescriptor[cpu_plan->total_layer_descriptors]);
      candidate.paged_assignment_descriptors.reset(
          new (std::nothrow) LlmPagedKvAssignmentDescriptor[
              cpu_plan->total_sequence_descriptors]);
      candidate.paged_k_block_references.reset(
          new (std::nothrow) LlmStaticSpanReference[
              cpu_plan->paged->layout.total_physical_blocks]);
      candidate.paged_v_block_references.reset(
          new (std::nothrow) LlmStaticSpanReference[
              cpu_plan->paged->layout.total_physical_blocks]);
      candidate.block_table = cpu_plan->paged->block_table();
      candidate.block_table_entries =
          cpu_plan->paged->layout.block_table_entries;
      candidate.paged_block_reference_count =
          cpu_plan->paged->layout.total_physical_blocks;
    } else {
      candidate.layer_descriptors.reset(
          new (std::nothrow)
              LlmLayerDescriptor[cpu_plan->total_layer_descriptors]);
      candidate.sequence_descriptors.reset(
          new (std::nothrow) LlmKvSequenceDescriptor[
              cpu_plan->total_sequence_descriptors]);
      candidate.k_references.reset(
          new (std::nothrow) LlmStaticSpanReference[
              cpu_plan->total_sequence_descriptors]);
      candidate.v_references.reset(
          new (std::nothrow) LlmStaticSpanReference[
              cpu_plan->total_sequence_descriptors]);
    }
    const bool descriptor_allocation_failed =
        candidate.weight_references == nullptr ||
        (plan.phase == LlmPhase::Prefill &&
                 plan.kv_layout == LlmKvLayout::Paged
             ? candidate.paged_prefill_layer_descriptors == nullptr ||
                   candidate.paged_prefill_assignment_descriptors == nullptr ||
                   candidate.paged_k_block_references == nullptr ||
                   candidate.paged_v_block_references == nullptr ||
                   candidate.block_table == nullptr
         : plan.phase == LlmPhase::Prefill
             ? candidate.prefill_layer_descriptors == nullptr ||
                   candidate.prefill_sequence_descriptors == nullptr
         : plan.kv_layout == LlmKvLayout::Paged
             ? candidate.paged_layer_descriptors == nullptr ||
                   candidate.paged_assignment_descriptors == nullptr ||
                   candidate.paged_k_block_references == nullptr ||
                   candidate.paged_v_block_references == nullptr ||
                   candidate.block_table == nullptr
             : candidate.layer_descriptors == nullptr ||
                   candidate.sequence_descriptors == nullptr ||
                   candidate.k_references == nullptr ||
                   candidate.v_references == nullptr);
    if (descriptor_allocation_failed) {
      result.reason_code = LlmExecutorReason::DESCRIPTOR_ALLOCATION_FAILED;
      return result;
    }

    candidate.model_plan_identity = plan.plan_identity;
    candidate.worker_count = cpu_plan->effective_workers;
    candidate.layer_descriptors_per_worker =
        cpu_plan->layer_descriptors_per_worker;
    candidate.sequence_descriptors_per_worker =
        cpu_plan->sequence_descriptors_per_worker;
    candidate.total_layer_descriptors = cpu_plan->total_layer_descriptors;
    candidate.total_sequence_descriptors =
        cpu_plan->total_sequence_descriptors;
    candidate.auxiliary = allocation.auxiliary;
    candidate.memory_budget = allocation.memory_budget;
    if (!materialize_descriptors(plan, candidate)) {
      result.reason_code = LlmExecutorReason::INVALID_DESCRIPTOR_LAYOUT;
      return result;
    }
    if (!initialize_resources(plan, candidate, candidate.initialization)) {
      result.reason_code = LlmExecutorReason::INITIALIZATION_FAILED;
      return result;
    }
    candidate.valid = true;
    if (!materialized_resources_match_plan(plan, candidate)) {
      result.reason_code = LlmExecutorReason::INVALID_DESCRIPTOR_LAYOUT;
      return result;
    }

    result.initialization = candidate.initialization;
    output = std::move(candidate);
    result.valid = true;
    result.reason_code = LlmExecutorReason::VALID;
    return result;
  } catch (const std::bad_alloc&) {
    result.valid = false;
    result.reason_code = LlmExecutorReason::DESCRIPTOR_ALLOCATION_FAILED;
    return result;
  } catch (...) {
    result.valid = false;
    result.reason_code = LlmExecutorReason::INITIALIZATION_FAILED;
    return result;
  }
}

LlmExpectedChecksumResult calculate_llm_expected_checksums(const LlmMemoryWorkPlan& model_plan,
                                                           const LlmScenarioWorkPlan& scenario_plan,
                                                           const LlmExecutionResources& resources) noexcept {
  LlmExpectedChecksumResult result;
  try {
    const LlmCpuExecutionPlan* const cpu_plan =
        get_llm_cpu_execution_plan(model_plan);
    if (cpu_plan == nullptr ||
        !materialized_resources_match_plan(model_plan, resources)) {
      result.reason_code = LlmExecutorReason::INVALID_RESOURCES;
      return result;
    }
    if (!scenario_plan_matches_model(model_plan, scenario_plan)) {
      result.reason_code = LlmExecutorReason::SCENARIO_PLAN_MISMATCH;
      return result;
    }
    if (model_plan.phase == LlmPhase::Prefill &&
        model_plan.kv_layout == LlmKvLayout::Paged) {
      return calculate_paged_prefill_expected_checksums(
          model_plan, scenario_plan, resources);
    }
    if (model_plan.phase == LlmPhase::Prefill) {
      return calculate_prefill_expected_checksums(
          model_plan, scenario_plan, resources);
    }
    if (model_plan.kv_layout == LlmKvLayout::Paged) {
      return calculate_paged_expected_checksums(model_plan, scenario_plan,
                                                resources);
    }

    result.workers.resize(cpu_plan->effective_workers);
    const bool reads_weight = (llm_scenario_flags(scenario_plan.scenario) & kLlmScenarioFlagWeight) != 0;
    const bool reads_kv = (llm_scenario_flags(scenario_plan.scenario) & kLlmScenarioFlagKv) != 0;
    uint64_t total_weight_bytes = 0;
    uint64_t total_k_bytes = 0;
    uint64_t total_v_bytes = 0;

    for (size_t worker_index = 0;
         worker_index < cpu_plan->effective_workers; ++worker_index) {
      LlmWorkerChecksum& checksum = result.workers[worker_index];
      checksum.weight = initial_llm_read_checksum(LlmChecksumComponent::Weight);
      checksum.k = initial_llm_read_checksum(LlmChecksumComponent::K);
      checksum.v = initial_llm_read_checksum(LlmChecksumComponent::V);
      const LlmWorkerWorkPlan& worker = cpu_plan->workers[worker_index];
      const LlmStaticSpanReference* const weight_references = resources.worker_weight_references(worker_index);
      const LlmStaticSpanReference* const k_references = resources.worker_k_references(worker_index);
      const LlmStaticSpanReference* const v_references = resources.worker_v_references(worker_index);
      if (weight_references == nullptr || k_references == nullptr || v_references == nullptr) {
        result.reason_code = LlmExecutorReason::INVALID_RESOURCES;
        result.workers.clear();
        return result;
      }

      for (size_t step = 0; step < scenario_plan.work_units; ++step) {
        for (size_t layer = 0; layer < model_plan.geometry.layer_count; ++layer) {
          if (reads_weight && !absorb_reference(checksum.weight, weight_references[layer])) {
            result.reason_code = LlmExecutorReason::EXPECTED_CHECKSUM_OVERFLOW;
            result.workers.clear();
            return result;
          }
          if (!reads_kv) {
            continue;
          }
          const LlmLayerRangeTemplate& layer_descriptor = worker.layers[layer];
          for (size_t batch_offset = 0; batch_offset < layer_descriptor.sequence_count; ++batch_offset) {
            const size_t sequence_index = layer_descriptor.first_sequence_index + batch_offset;
            if (sequence_index >= worker.sequences.size()) {
              result.reason_code = LlmExecutorReason::INVALID_DESCRIPTOR_LAYOUT;
              result.workers.clear();
              return result;
            }
            const LlmKvSequenceRangeTemplate& sequence = worker.sequences[sequence_index];
            LlmStaticSpanReference k_adjusted;
            LlmStaticSpanReference v_adjusted;
            if (!append_adjusted_reference(k_references[sequence_index], sequence, model_plan.k_buffer_seed,
                                           scenario_plan.scenario_seed, static_cast<uint64_t>(step),
                                           LlmChecksumComponent::K, k_adjusted) ||
                !append_adjusted_reference(v_references[sequence_index], sequence, model_plan.v_buffer_seed,
                                           scenario_plan.scenario_seed, static_cast<uint64_t>(step),
                                           LlmChecksumComponent::V, v_adjusted) ||
                !absorb_reference(checksum.k, k_adjusted) || !absorb_reference(checksum.v, v_adjusted)) {
              result.reason_code = LlmExecutorReason::EXPECTED_CHECKSUM_OVERFLOW;
              result.workers.clear();
              return result;
            }
          }
        }
      }

      if (!add_checksum_bytes(checksum.weight.exact_bytes_read, total_weight_bytes) ||
          !add_checksum_bytes(checksum.k.exact_bytes_read, total_k_bytes) ||
          !add_checksum_bytes(checksum.v.exact_bytes_read, total_v_bytes)) {
        result.reason_code = LlmExecutorReason::EXPECTED_CHECKSUM_OVERFLOW;
        result.workers.clear();
        return result;
      }
    }

    uint64_t total_kv_bytes = 0;
    if (!add_checksum_bytes(total_k_bytes, total_kv_bytes) || !add_checksum_bytes(total_v_bytes, total_kv_bytes) ||
        total_weight_bytes != scenario_plan.weight_read_bytes || total_kv_bytes != scenario_plan.kv_read_bytes) {
      result.reason_code = LlmExecutorReason::INVALID_DESCRIPTOR_LAYOUT;
      result.workers.clear();
      return result;
    }
    result.run_checksum = fold_llm_worker_checksums(result.workers.data(), result.workers.size());
    result.valid = true;
    result.reason_code = LlmExecutorReason::VALID;
    return result;
  } catch (const std::bad_alloc&) {
    result.valid = false;
    result.reason_code = LlmExecutorReason::EXPECTED_CHECKSUM_ALLOCATION_FAILED;
    result.workers.clear();
    return result;
  } catch (...) {
    result.valid = false;
    result.reason_code = LlmExecutorReason::INVALID_SCENARIO_PLAN;
    result.workers.clear();
    return result;
  }
}

LlmKernelAdapter production_llm_kernel_adapter() noexcept { return {production_kernel_invoke, nullptr}; }

LlmExecutorResult execute_llm_scenario(const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan& scenario_plan,
                                       const LlmExecutionResources& resources, HighResTimer& timer,
                                       LlmKernelAdapter kernel, const LlmExecutorTestControl* test_control) noexcept {
  LlmExecutorResult result;
  const LlmCpuExecutionPlan* const cpu_plan =
      get_llm_cpu_execution_plan(model_plan);
  result.requested_workers =
      cpu_plan == nullptr ? 0 : cpu_plan->effective_workers;
  try {
    if (cpu_plan == nullptr ||
        !materialized_resources_match_plan(model_plan, resources)) {
      result.reason_code = LlmExecutorReason::INVALID_RESOURCES;
      return result;
    }
    if (!scenario_plan_matches_model(model_plan, scenario_plan)) {
      result.reason_code = LlmExecutorReason::SCENARIO_PLAN_MISMATCH;
      return result;
    }
    if (kernel.invoke == nullptr) {
      result.reason_code = LlmExecutorReason::KERNEL_FAILED;
      return result;
    }

    LlmExpectedChecksumResult expected = calculate_llm_expected_checksums(model_plan, scenario_plan, resources);
    if (!expected.valid) {
      result.reason_code = expected.reason_code;
      return result;
    }
    result.expected_checksums = std::move(expected.workers);
    result.expected_run_checksum = expected.run_checksum;
    if (!reset_paged_append_slots(model_plan, resources)) {
      result.reason_code = LlmExecutorReason::INVALID_RESOURCES;
      return result;
    }
    result.actual_checksums.resize(cpu_plan->effective_workers);

    std::vector<uint8_t> worker_succeeded;
    std::vector<std::thread> threads;
    std::mutex state_mutex;
    std::condition_variable state_cv;
    size_t ready_workers = 0;
    bool start_workers = false;
    bool cancel_workers = false;
    bool measurement_complete = false;
    bool timer_stop_succeeded = false;
    double measured_duration = 0.0;
    std::atomic<size_t> remaining_workers{cpu_plan->effective_workers};
    std::atomic<size_t> completed_workers{0};
    std::atomic<size_t> qos_successful_workers{0};
    std::atomic<size_t> qos_failed_workers{0};

    try {
      worker_succeeded.resize(cpu_plan->effective_workers, 0);
      threads.reserve(cpu_plan->effective_workers);
      for (size_t worker_index = 0;
           worker_index < cpu_plan->effective_workers; ++worker_index) {
        if (test_control != nullptr && test_control->fail_before_worker_index >= 0 &&
            worker_index == static_cast<size_t>(test_control->fail_before_worker_index)) {
          throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again));
        }
        threads.emplace_back([&, worker_index] {
          const int qos_result = set_worker_qos(test_control, worker_index);
          if (qos_result == KERN_SUCCESS) {
            qos_successful_workers.fetch_add(1, std::memory_order_relaxed);
          } else {
            qos_failed_workers.fetch_add(1, std::memory_order_relaxed);
          }

          {
            std::unique_lock<std::mutex> lock(state_mutex);
            ++ready_workers;
            observe_event(test_control, LlmExecutorEvent::WorkerReady, worker_index);
            state_cv.notify_all();
            state_cv.wait(lock, [&] { return start_workers || cancel_workers; });
            if (cancel_workers) {
              lock.unlock();
              observe_event(test_control, LlmExecutorEvent::WorkerCancelled, worker_index);
              return;
            }
          }

          LlmKernelInvocation invocation;
          invocation.layers = resources.worker_layers(worker_index);
          invocation.sequences = resources.worker_sequences(worker_index);
          invocation.layer_count = static_cast<uint64_t>(
              cpu_plan->layer_descriptors_per_worker);
          invocation.work_unit_count =
              static_cast<uint64_t>(scenario_plan.work_units);
          invocation.scenario_flags =
              llm_scenario_flags(scenario_plan.scenario);
          invocation.scenario_seed = scenario_plan.scenario_seed;
          invocation.output = &result.actual_checksums[worker_index];
          invocation.worker_index = worker_index;
          invocation.kv_layout = model_plan.kv_layout;
          invocation.phase = model_plan.phase;
          invocation.paged_layers =
              resources.worker_paged_layers(worker_index);
          invocation.paged_assignments =
              resources.worker_paged_assignments(worker_index);
          invocation.prefill_layers =
              resources.worker_prefill_layers(worker_index);
          invocation.prefill_sequences =
              (invocation.scenario_flags & kLlmScenarioFlagKv) == 0
                  ? nullptr
                  : resources.worker_prefill_sequences(
                        worker_index, scenario_plan.scenario);
          invocation.paged_prefill_layers =
              resources.worker_paged_prefill_layers(worker_index);
          invocation.paged_prefill_assignments =
              (invocation.scenario_flags & kLlmScenarioFlagKv) == 0
                  ? nullptr
                  : resources.worker_paged_prefill_assignments(
                        worker_index, scenario_plan.scenario);
          observe_event(test_control, LlmExecutorEvent::KernelStarted, worker_index);
          bool succeeded = false;
          try {
            succeeded = kernel.invoke(kernel.context, invocation);
          } catch (...) {
            succeeded = false;
          }
          worker_succeeded[worker_index] = succeeded ? 1 : 0;
          observe_event(test_control, LlmExecutorEvent::KernelCompleted, worker_index);

          asm volatile("dsb ish" ::: "memory");
          completed_workers.fetch_add(1, std::memory_order_relaxed);
          if (remaining_workers.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            double duration = 0.0;
            bool stopped = false;
            try {
              duration = timer.stop();
              stopped = true;
            } catch (...) {
              stopped = false;
            }
            if (stopped) {
              observe_event(test_control, LlmExecutorEvent::TimerStopped, worker_index);
            }
            {
              std::lock_guard<std::mutex> lock(state_mutex);
              measured_duration = duration;
              timer_stop_succeeded = stopped;
              measurement_complete = true;
            }
            state_cv.notify_one();
          }
        });
      }
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(state_mutex);
        cancel_workers = true;
      }
      state_cv.notify_all();
      join_threads(threads);
      result.created_workers = threads.size();
      result.qos_successful_workers = qos_successful_workers.load(std::memory_order_relaxed);
      result.qos_failed_workers = qos_failed_workers.load(std::memory_order_relaxed);
      result.worker_startup_failed = true;
      result.reason_code = LlmExecutorReason::WORKER_STARTUP_FAILED;
      return result;
    }

    try {
      {
        std::unique_lock<std::mutex> lock(state_mutex);
        state_cv.wait(
            lock,
            [&] { return ready_workers == cpu_plan->effective_workers; });
        timer.start();
        result.timer_started = true;
        observe_event(test_control, LlmExecutorEvent::TimerStarted,
                      cpu_plan->effective_workers);
        start_workers = true;
      }
      state_cv.notify_all();

      {
        std::unique_lock<std::mutex> lock(state_mutex);
        state_cv.wait(lock, [&] { return measurement_complete; });
      }
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(state_mutex);
        if (!start_workers) {
          cancel_workers = true;
        }
      }
      state_cv.notify_all();
      join_threads(threads);
      result.created_workers = threads.size();
      result.completed_workers = completed_workers.load(std::memory_order_relaxed);
      result.qos_successful_workers = qos_successful_workers.load(std::memory_order_relaxed);
      result.qos_failed_workers = qos_failed_workers.load(std::memory_order_relaxed);
      result.timer_stopped = timer_stop_succeeded;
      result.reason_code = LlmExecutorReason::INVALID_ELAPSED_TIME;
      return result;
    }
    join_threads(threads);
    result.created_workers = threads.size();
    result.completed_workers = completed_workers.load(std::memory_order_relaxed);
    result.qos_successful_workers = qos_successful_workers.load(std::memory_order_relaxed);
    result.qos_failed_workers = qos_failed_workers.load(std::memory_order_relaxed);
    result.elapsed_seconds = measured_duration;
    result.timer_stopped = timer_stop_succeeded;

    result.kernel_succeeded =
        std::all_of(worker_succeeded.begin(), worker_succeeded.end(), [](uint8_t succeeded) { return succeeded != 0; });
    if (!result.kernel_succeeded) {
      result.reason_code = LlmExecutorReason::KERNEL_FAILED;
      return result;
    }
    if (!result.timer_stopped || !std::isfinite(result.elapsed_seconds) || result.elapsed_seconds <= 0.0) {
      result.reason_code = LlmExecutorReason::INVALID_ELAPSED_TIME;
      return result;
    }

    observe_event(test_control, LlmExecutorEvent::ChecksumValidationStarted,
                  cpu_plan->effective_workers);
    result.actual_run_checksum =
        fold_llm_worker_checksums(result.actual_checksums.data(), result.actual_checksums.size());
    result.checksum_valid = result.expected_checksums.size() == result.actual_checksums.size();
    for (size_t worker_index = 0; result.checksum_valid && worker_index < result.expected_checksums.size();
         ++worker_index) {
      result.checksum_valid =
          equal_worker_checksum(result.expected_checksums[worker_index], result.actual_checksums[worker_index]);
    }
    result.checksum_evaluated = true;

    result.post_validation_evaluated = true;
    result.post_validation_valid =
        model_plan.phase == LlmPhase::Prefill &&
                model_plan.kv_layout == LlmKvLayout::Paged
            ? validate_paged_prefill_post_execution(
                  model_plan, scenario_plan, resources)
        : model_plan.phase == LlmPhase::Prefill
            ? validate_prefill_post_execution(model_plan, scenario_plan,
                                               resources)
        : model_plan.kv_layout != LlmKvLayout::Paged ||
              validate_paged_post_execution(model_plan, scenario_plan,
                                             resources);
    if (!result.checksum_valid) {
      result.reason_code = LlmExecutorReason::CHECKSUM_MISMATCH;
      return result;
    }
    if (!result.post_validation_valid) {
      result.reason_code =
          model_plan.phase == LlmPhase::Prefill
              ? LlmExecutorReason::PREFILL_POST_VALIDATION_FAILED
              : LlmExecutorReason::PAGED_POST_VALIDATION_FAILED;
      return result;
    }

    result.valid = true;
    result.reason_code = LlmExecutorReason::VALID;
    return result;
  } catch (const std::bad_alloc&) {
    result.valid = false;
    result.reason_code = LlmExecutorReason::EXPECTED_CHECKSUM_ALLOCATION_FAILED;
    return result;
  } catch (...) {
    result.valid = false;
    result.reason_code = LlmExecutorReason::KERNEL_FAILED;
    return result;
  }
}

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

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "core/config/constants.h"
#include "llm_memory/llm_kv_layout.h"
#include "llm_memory/llm_metal_backend.h"
#include "llm_memory/llm_work_plan.h"
#include "test_memory_system_calls.h"
#include "utils/numeric_utils.h"

namespace {

constexpr size_t kGiB = 1024ULL * 1024ULL * 1024ULL;

const LlmCpuExecutionPlan& cpu_execution_plan(
    const LlmMemoryWorkPlan& plan) {
  const LlmCpuExecutionPlan* const cpu_plan =
      get_llm_cpu_execution_plan(plan);
  if (cpu_plan == nullptr) {
    throw std::logic_error("expected CPU execution plan");
  }
  return *cpu_plan;
}

LlmCpuExecutionPlan& cpu_execution_plan(LlmMemoryWorkPlan& plan) {
  LlmCpuExecutionPlan* const cpu_plan = get_llm_cpu_execution_plan(plan);
  if (cpu_plan == nullptr) {
    throw std::logic_error("expected CPU execution plan");
  }
  return *cpu_plan;
}

const LlmPagedCpuExecutionPlan& paged_cpu_execution_plan(
    const LlmMemoryWorkPlan& plan) {
  const LlmCpuExecutionPlan& cpu_plan = cpu_execution_plan(plan);
  if (!cpu_plan.paged.has_value()) {
    throw std::logic_error("expected paged CPU execution plan");
  }
  return *cpu_plan.paged;
}

LlmGeometryRequest large_geometry_request() {
  return {4 * kGiB, 32, 32, 8, 128, 2, 8192, 1};
}

LlmGeometryRequest small_geometry_request(size_t batch_size = 1) {
  // K+V across both layers is exactly 128 bytes per visible token.
  return {1024, 2, 4, 2, 8, 2, 3, batch_size};
}

LlmGeometryRequest paged_geometry_request(
    size_t context_tokens = 35, size_t block_tokens = 16,
    size_t batch_size = 2) {
  // One K or V token record is 32 bytes in each of the two layers.
  return {1024, 2, 4, 2, 8, 2, context_tokens, batch_size,
          block_tokens, LlmPhase::Decode, LlmKvLayout::Paged};
}

LlmMemoryWorkPlanRequest work_plan_request(
    const LlmGeometryRequest& geometry, size_t requested_workers = 2,
    size_t available_workers = 2) {
  LlmMemoryWorkPlanRequest request;
  request.geometry = geometry;
  request.requested_workers = requested_workers;
  request.available_workers = available_workers;
  request.available_memory_bytes = 16 * kGiB;
  request.mapping_granularity_bytes = 1;
  request.base_seed = 42;
  return request;
}

LlmMemoryWorkPlanRequest metal_work_plan_request() {
  LlmMemoryWorkPlanRequest request =
      work_plan_request(small_geometry_request(), 0, 0);
  request.backend = LlmMemoryBackend::Metal;
  return request;
}

LlmMetalResourcePlanRequest metal_resource_request(
    const LlmMemoryWorkPlan& logical_plan,
    size_t command_auxiliary_bytes = 0) {
  LlmMetalResourcePlanRequest request;
  request.geometry = logical_plan.geometry;
  request.argument_buffer_encoded_length = 256;
  request.argument_buffer_alignment = 16;
  request.max_buffer_length =
      Constants::LLM_METAL_SEGMENT_CAPACITY_BYTES;
  request.available_memory_bytes = 16 * kGiB;
  request.host_mapping_granularity_bytes = 1;
  EXPECT_TRUE(NumericUtils::checked_add(
      logical_plan.memory_budget.request.planner_storage_bytes,
      command_auxiliary_bytes, request.additional_owned_bytes));
  return request;
}

size_t external_string_backing_bytes(const std::string& value) {
  const std::uintptr_t data =
      reinterpret_cast<std::uintptr_t>(value.data());
  const std::uintptr_t object =
      reinterpret_cast<std::uintptr_t>(&value);
  return data >= object && data < object + sizeof(value)
             ? 0
             : value.capacity() + 1;
}

size_t segment_plan_backing_bytes(
    const LlmKvSegmentPlan& segment_plan) {
  return segment_plan.segment_lengths.capacity() * sizeof(size_t) +
         external_string_backing_bytes(segment_plan.reason_code) +
         external_string_backing_bytes(segment_plan.identity);
}

size_t metal_execution_plan_backing_bytes(
    const LlmMetalExecutionPlan& execution) {
  const LlmMetalResourcePlan& resources = execution.resources;
  size_t bytes =
      external_string_backing_bytes(execution.reason_code) +
      external_string_backing_bytes(execution.msl_revision) +
      external_string_backing_bytes(execution.msl_source_sha256) +
      external_string_backing_bytes(execution.identity) +
      external_string_backing_bytes(resources.reason_code) +
      external_string_backing_bytes(resources.identity) +
      segment_plan_backing_bytes(resources.weight_segments) +
      segment_plan_backing_bytes(resources.k_segments) +
      segment_plan_backing_bytes(resources.v_segments) +
      external_string_backing_bytes(
          resources.argument_buffer.reason_code) +
      external_string_backing_bytes(
          resources.argument_buffer.identity) +
      resources.planned_resources.capacity() *
          sizeof(LlmMetalPlannedResource);
  if (resources.table_segments.has_value()) {
    bytes += segment_plan_backing_bytes(*resources.table_segments);
  }
  if (resources.paged_layout.has_value()) {
    bytes += external_string_backing_bytes(
                 resources.paged_layout->reason_code) +
             external_string_backing_bytes(
                 resources.paged_layout->geometry_identity);
  }
  return bytes;
}

void expect_invalid_plan(const LlmMemoryWorkPlan& plan,
                         const std::string& reason) {
  EXPECT_FALSE(plan.valid);
  EXPECT_EQ(plan.reason_code, reason);
  EXPECT_TRUE(plan.weight_layers.empty());
  if (const LlmCpuExecutionPlan* const cpu_plan =
          get_llm_cpu_execution_plan(plan)) {
    EXPECT_TRUE(cpu_plan->workers.empty());
  } else {
    EXPECT_TRUE(std::holds_alternative<LlmMetalExecutionPlan>(
        plan.backend_execution_plan));
  }
  EXPECT_TRUE(plan.plan_identity.empty());
}

LlmByteRange union_for_sequence(
    const LlmMemoryWorkPlan& plan, size_t sequence_index,
    LlmByteRange LlmKvSequenceRangeTemplate::*member) {
  size_t first = std::numeric_limits<size_t>::max();
  size_t cursor = 0;
  size_t total = 0;
  bool found_range = false;
  for (const LlmWorkerWorkPlan& worker :
       cpu_execution_plan(plan).workers) {
    const LlmByteRange& range = worker.sequences[sequence_index].*member;
    if (range.span_bytes == 0) {
      continue;
    }
    if (!found_range) {
      first = range.offset_bytes;
      cursor = first;
      found_range = true;
    }
    EXPECT_EQ(range.offset_bytes, cursor);
    cursor = range.offset_bytes + range.span_bytes;
    total += range.span_bytes;
  }
  EXPECT_EQ(total, found_range ? cursor - first : 0u);
  return total == 0 ? LlmByteRange{} : LlmByteRange{first, total};
}

LlmByteRange union_for_weight_layer(const LlmMemoryWorkPlan& plan,
                                    size_t layer_index) {
  const LlmByteRange& layer = plan.weight_layers[layer_index];
  size_t cursor = layer.offset_bytes;
  size_t total = 0;
  for (const LlmWorkerWorkPlan& worker :
       cpu_execution_plan(plan).workers) {
    const LlmByteRange& range = worker.layers[layer_index].weight;
    if (range.span_bytes == 0) {
      continue;
    }
    EXPECT_EQ(range.offset_bytes, cursor);
    EXPECT_GE(range.offset_bytes, layer.offset_bytes);
    EXPECT_LE(range.offset_bytes + range.span_bytes,
              layer.offset_bytes + layer.span_bytes);
    cursor = range.offset_bytes + range.span_bytes;
    total += range.span_bytes;
  }
  EXPECT_EQ(total, layer.span_bytes);
  EXPECT_EQ(cursor, layer.offset_bytes + layer.span_bytes);
  return {layer.offset_bytes, total};
}

void expect_byte_ranges_equal(const LlmByteRange& actual,
                              const LlmByteRange& expected) {
  EXPECT_EQ(actual.offset_bytes, expected.offset_bytes);
  EXPECT_EQ(actual.span_bytes, expected.span_bytes);
}

void expect_geometries_equal(const LlmGeometry& actual,
                             const LlmGeometry& expected) {
  EXPECT_EQ(actual.valid, expected.valid);
  EXPECT_EQ(actual.reason_code, expected.reason_code);
  EXPECT_EQ(actual.phase, expected.phase);
  EXPECT_EQ(actual.kv_layout, expected.kv_layout);
  EXPECT_EQ(actual.work_unit_kind, expected.work_unit_kind);
  EXPECT_EQ(actual.decode.has_value(), expected.decode.has_value());
  if (actual.decode.has_value() && expected.decode.has_value()) {
    EXPECT_EQ(actual.decode->visible_context_tokens,
              expected.decode->visible_context_tokens);
  }
  EXPECT_EQ(actual.prefill.has_value(), expected.prefill.has_value());
  if (actual.prefill.has_value() && expected.prefill.has_value()) {
    EXPECT_EQ(actual.prefill->prompt_tokens, expected.prefill->prompt_tokens);
    EXPECT_EQ(actual.prefill->attention_query_tile_tokens,
              expected.prefill->attention_query_tile_tokens);
    EXPECT_EQ(actual.prefill->tile_count, expected.prefill->tile_count);
    EXPECT_EQ(actual.prefill->attention_prefix_token_visits_per_sequence,
              expected.prefill->attention_prefix_token_visits_per_sequence);
    EXPECT_EQ(actual.prefill->causal_token_pairs_per_sequence,
              expected.prefill->causal_token_pairs_per_sequence);
    EXPECT_EQ(actual.prefill->logical_attention_pairs,
              expected.prefill->logical_attention_pairs);
    EXPECT_EQ(actual.prefill->logical_attention_fma_terms,
              expected.prefill->logical_attention_fma_terms);
    EXPECT_EQ(actual.prefill->paged_prefix_block_visits_per_sequence,
              expected.prefill->paged_prefix_block_visits_per_sequence);
  }
  EXPECT_EQ(actual.attention_kind, expected.attention_kind);
  EXPECT_EQ(actual.active_weight_bytes_per_work_unit,
            expected.active_weight_bytes_per_work_unit);
  EXPECT_EQ(actual.layer_count, expected.layer_count);
  EXPECT_EQ(actual.query_head_count, expected.query_head_count);
  EXPECT_EQ(actual.kv_head_count, expected.kv_head_count);
  EXPECT_EQ(actual.query_heads_per_kv_head,
            expected.query_heads_per_kv_head);
  EXPECT_EQ(actual.head_dimension, expected.head_dimension);
  EXPECT_EQ(actual.kv_element_bytes, expected.kv_element_bytes);
  EXPECT_EQ(actual.batch_size, expected.batch_size);
  EXPECT_EQ(actual.kv_vector_bytes, expected.kv_vector_bytes);
  EXPECT_EQ(actual.k_or_v_record_bytes_per_layer,
            expected.k_or_v_record_bytes_per_layer);
  EXPECT_EQ(actual.kv_record_bytes_per_layer,
            expected.kv_record_bytes_per_layer);
  EXPECT_EQ(actual.kv_bytes_per_visible_token,
            expected.kv_bytes_per_visible_token);
  EXPECT_EQ(actual.k_or_v_sequence_visible_bytes,
            expected.k_or_v_sequence_visible_bytes);
  EXPECT_EQ(actual.kv_block_tokens, expected.kv_block_tokens);
  EXPECT_EQ(actual.kv_blocks_per_sequence,
            expected.kv_blocks_per_sequence);
  EXPECT_EQ(actual.physical_blocks_per_layer,
            expected.physical_blocks_per_layer);
  EXPECT_EQ(actual.total_physical_blocks,
            expected.total_physical_blocks);
  EXPECT_EQ(actual.kv_block_bytes, expected.kv_block_bytes);
  EXPECT_EQ(actual.last_block_tokens, expected.last_block_tokens);
  EXPECT_EQ(actual.last_block_valid_bytes,
            expected.last_block_valid_bytes);
  EXPECT_EQ(actual.decode_append_offset_in_last_block,
            expected.decode_append_offset_in_last_block);
  EXPECT_EQ(actual.k_logical_bytes, expected.k_logical_bytes);
  EXPECT_EQ(actual.v_logical_bytes, expected.v_logical_bytes);
  EXPECT_EQ(actual.k_layout_padding_bytes,
            expected.k_layout_padding_bytes);
  EXPECT_EQ(actual.v_layout_padding_bytes,
            expected.v_layout_padding_bytes);
  EXPECT_EQ(actual.block_table_entries, expected.block_table_entries);
  EXPECT_EQ(actual.block_table_bytes, expected.block_table_bytes);
  EXPECT_EQ(
      actual.layout_metadata_lookups_per_layer_sequence_per_work_unit,
      expected.layout_metadata_lookups_per_layer_sequence_per_work_unit);
  EXPECT_EQ(actual.k_mapping_bytes, expected.k_mapping_bytes);
  EXPECT_EQ(actual.v_mapping_bytes, expected.v_mapping_bytes);
  EXPECT_EQ(actual.kv_capacity_bytes, expected.kv_capacity_bytes);
  EXPECT_EQ(actual.weight_read_bytes_per_work_unit,
            expected.weight_read_bytes_per_work_unit);
  EXPECT_EQ(actual.kv_read_bytes_per_work_unit,
            expected.kv_read_bytes_per_work_unit);
  EXPECT_EQ(actual.kv_write_bytes_per_work_unit,
            expected.kv_write_bytes_per_work_unit);
  EXPECT_EQ(actual.kv_only_effective_model_payload_bytes_per_work_unit,
            expected.kv_only_effective_model_payload_bytes_per_work_unit);
  EXPECT_EQ(actual.mixed_effective_model_payload_bytes_per_work_unit,
            expected.mixed_effective_model_payload_bytes_per_work_unit);
  EXPECT_EQ(actual.total_data_mapping_bytes,
            expected.total_data_mapping_bytes);
  EXPECT_EQ(actual.traffic_crossover_numerator,
            expected.traffic_crossover_numerator);
  EXPECT_EQ(actual.traffic_crossover_denominator,
            expected.traffic_crossover_denominator);
  EXPECT_DOUBLE_EQ(actual.traffic_crossover_context_tokens,
                   expected.traffic_crossover_context_tokens);
}

void expect_budget_requests_equal(const LlmMemoryBudgetRequest& actual,
                                  const LlmMemoryBudgetRequest& expected) {
  EXPECT_EQ(actual.valid, expected.valid);
  EXPECT_EQ(actual.reason_code, expected.reason_code);
  EXPECT_EQ(actual.mapping_granularity_bytes,
            expected.mapping_granularity_bytes);
  EXPECT_EQ(actual.requested_weight_mapping_bytes,
            expected.requested_weight_mapping_bytes);
  EXPECT_EQ(actual.requested_k_mapping_bytes,
            expected.requested_k_mapping_bytes);
  EXPECT_EQ(actual.requested_v_mapping_bytes,
            expected.requested_v_mapping_bytes);
  EXPECT_EQ(actual.committed_weight_mapping_bytes,
            expected.committed_weight_mapping_bytes);
  EXPECT_EQ(actual.committed_k_mapping_bytes,
            expected.committed_k_mapping_bytes);
  EXPECT_EQ(actual.committed_v_mapping_bytes,
            expected.committed_v_mapping_bytes);
  EXPECT_EQ(actual.requested_block_table_mapping_bytes,
            expected.requested_block_table_mapping_bytes);
  EXPECT_EQ(actual.committed_block_table_mapping_bytes,
            expected.committed_block_table_mapping_bytes);
  EXPECT_EQ(actual.requested_data_bytes, expected.requested_data_bytes);
  EXPECT_EQ(actual.committed_data_bytes, expected.committed_data_bytes);
  EXPECT_EQ(actual.layout_transient_bytes,
            expected.layout_transient_bytes);
  EXPECT_EQ(actual.setup_peak_bytes, expected.setup_peak_bytes);
  EXPECT_EQ(actual.runtime_peak_bytes, expected.runtime_peak_bytes);
  EXPECT_EQ(actual.descriptor_bytes, expected.descriptor_bytes);
  EXPECT_EQ(actual.planner_storage_bytes, expected.planner_storage_bytes);
  EXPECT_EQ(actual.checksum_auxiliary_bytes,
            expected.checksum_auxiliary_bytes);
  EXPECT_EQ(actual.orchestration_auxiliary_bytes,
            expected.orchestration_auxiliary_bytes);
  EXPECT_EQ(actual.auxiliary_bytes, expected.auxiliary_bytes);
  EXPECT_EQ(actual.required_total_bytes, expected.required_total_bytes);
}

void expect_memory_budgets_equal(const LlmMemoryBudget& actual,
                                 const LlmMemoryBudget& expected) {
  EXPECT_EQ(actual.valid, expected.valid);
  EXPECT_EQ(actual.reason_code, expected.reason_code);
  expect_budget_requests_equal(actual.request, expected.request);
  EXPECT_EQ(actual.available_memory_bytes, expected.available_memory_bytes);
  EXPECT_EQ(actual.allowed_memory_bytes, expected.allowed_memory_bytes);
  EXPECT_EQ(actual.used_fallback, expected.used_fallback);
}

std::vector<uint32_t> paged_table_entries(
    const LlmMemoryWorkPlan& plan) {
  const LlmPagedCpuExecutionPlan& paged = paged_cpu_execution_plan(plan);
  if (paged.block_table() == nullptr) {
    throw std::logic_error("expected materialized paged block table");
  }
  return {paged.block_table(),
          paged.block_table() + paged.layout.block_table_entries};
}

void expect_equivalent_executable_plans(const LlmMemoryWorkPlan& actual,
                                        const LlmMemoryWorkPlan& expected) {
  const LlmCpuExecutionPlan& actual_cpu = cpu_execution_plan(actual);
  const LlmCpuExecutionPlan& expected_cpu = cpu_execution_plan(expected);
  EXPECT_EQ(actual.valid, expected.valid);
  EXPECT_EQ(actual.reason_code, expected.reason_code);
  expect_geometries_equal(actual.geometry, expected.geometry);
  EXPECT_EQ(actual_cpu.requested_workers, expected_cpu.requested_workers);
  EXPECT_EQ(actual_cpu.available_workers, expected_cpu.available_workers);
  EXPECT_EQ(actual_cpu.effective_workers, expected_cpu.effective_workers);
  EXPECT_EQ(actual_cpu.layer_descriptors_per_worker,
            expected_cpu.layer_descriptors_per_worker);
  EXPECT_EQ(actual_cpu.sequence_descriptors_per_worker,
            expected_cpu.sequence_descriptors_per_worker);
  EXPECT_EQ(actual_cpu.total_layer_descriptors,
            expected_cpu.total_layer_descriptors);
  EXPECT_EQ(actual_cpu.total_sequence_descriptors,
            expected_cpu.total_sequence_descriptors);
  EXPECT_EQ(actual_cpu.descriptor_bytes, expected_cpu.descriptor_bytes);
  EXPECT_EQ(actual_cpu.planner_storage_bytes,
            expected_cpu.planner_storage_bytes);
  EXPECT_EQ(actual.base_seed, expected.base_seed);
  EXPECT_EQ(actual.weight_buffer_seed, expected.weight_buffer_seed);
  EXPECT_EQ(actual.k_buffer_seed, expected.k_buffer_seed);
  EXPECT_EQ(actual.v_buffer_seed, expected.v_buffer_seed);
  EXPECT_EQ(actual.scenario_seeds, expected.scenario_seeds);
  expect_memory_budgets_equal(actual.memory_budget, expected.memory_budget);
  EXPECT_EQ(actual.backend, expected.backend);
  EXPECT_EQ(actual.phase, expected.phase);
  EXPECT_EQ(actual.kv_layout, expected.kv_layout);
  EXPECT_EQ(actual.work_unit_kind, expected.work_unit_kind);
  EXPECT_EQ(actual.weight_passes_per_work_unit,
            expected.weight_passes_per_work_unit);
  EXPECT_EQ(actual.kv_replay_factor, expected.kv_replay_factor);
  EXPECT_EQ(actual.methodology_version, expected.methodology_version);
  EXPECT_EQ(actual.component_identities.logical_profile_version,
            expected.component_identities.logical_profile_version);
  EXPECT_EQ(actual.component_identities.kv_layout_version,
            expected.component_identities.kv_layout_version);
  EXPECT_EQ(actual.component_identities.permutation_version,
            expected.component_identities.permutation_version);
  EXPECT_EQ(actual.component_identities.backend_executor_version,
            expected.component_identities.backend_executor_version);
  EXPECT_EQ(actual.component_identities.resource_abi_version,
            expected.component_identities.resource_abi_version);
  EXPECT_EQ(actual.component_identities.schedule_version,
            expected.component_identities.schedule_version);
  EXPECT_EQ(actual.component_identities.timer_policy_version,
            expected.component_identities.timer_policy_version);
  EXPECT_EQ(actual.component_identities.buffer_pattern_version,
            expected.component_identities.buffer_pattern_version);
  EXPECT_EQ(actual.component_identities.write_pattern_version,
            expected.component_identities.write_pattern_version);
  EXPECT_EQ(actual.component_identities.checksum_pattern_version,
            expected.component_identities.checksum_pattern_version);
  EXPECT_EQ(actual.component_identities.msl_revision,
            expected.component_identities.msl_revision);
  EXPECT_EQ(actual.component_identities.msl_source_sha256,
            expected.component_identities.msl_source_sha256);
  EXPECT_EQ(actual.component_identities.identity,
            expected.component_identities.identity);
  EXPECT_EQ(actual.plan_identity, expected.plan_identity);

  ASSERT_EQ(actual.weight_layers.size(), expected.weight_layers.size());
  EXPECT_EQ(actual.weight_layers.capacity(), expected.weight_layers.capacity());
  for (size_t layer = 0; layer < actual.weight_layers.size(); ++layer) {
    SCOPED_TRACE(::testing::Message() << "weight layer " << layer);
    expect_byte_ranges_equal(actual.weight_layers[layer],
                             expected.weight_layers[layer]);
  }

  ASSERT_EQ(actual_cpu.workers.size(), expected_cpu.workers.size());
  EXPECT_EQ(actual_cpu.workers.capacity(), expected_cpu.workers.capacity());
  for (size_t worker_index = 0;
       worker_index < actual_cpu.workers.size();
       ++worker_index) {
    SCOPED_TRACE(::testing::Message() << "worker " << worker_index);
    const LlmWorkerWorkPlan& actual_worker =
        actual_cpu.workers[worker_index];
    const LlmWorkerWorkPlan& expected_worker =
        expected_cpu.workers[worker_index];
    EXPECT_EQ(actual_worker.worker_index, expected_worker.worker_index);
    ASSERT_EQ(actual_worker.layers.size(), expected_worker.layers.size());
    EXPECT_EQ(actual_worker.layers.capacity(), expected_worker.layers.capacity());
    for (size_t layer = 0; layer < actual_worker.layers.size(); ++layer) {
      SCOPED_TRACE(::testing::Message() << "layer template " << layer);
      const LlmLayerRangeTemplate& actual_layer = actual_worker.layers[layer];
      const LlmLayerRangeTemplate& expected_layer =
          expected_worker.layers[layer];
      expect_byte_ranges_equal(actual_layer.weight, expected_layer.weight);
      EXPECT_EQ(actual_layer.first_sequence_index,
                expected_layer.first_sequence_index);
      EXPECT_EQ(actual_layer.sequence_count, expected_layer.sequence_count);
      EXPECT_EQ(actual_layer.layer_index, expected_layer.layer_index);
    }
    ASSERT_EQ(actual_worker.sequences.size(),
              expected_worker.sequences.size());
    EXPECT_EQ(actual_worker.sequences.capacity(),
              expected_worker.sequences.capacity());
    for (size_t sequence = 0; sequence < actual_worker.sequences.size();
         ++sequence) {
      SCOPED_TRACE(::testing::Message() << "sequence template " << sequence);
      const LlmKvSequenceRangeTemplate& actual_sequence =
          actual_worker.sequences[sequence];
      const LlmKvSequenceRangeTemplate& expected_sequence =
          expected_worker.sequences[sequence];
      expect_byte_ranges_equal(actual_sequence.k_visible,
                               expected_sequence.k_visible);
      expect_byte_ranges_equal(actual_sequence.v_visible,
                               expected_sequence.v_visible);
      expect_byte_ranges_equal(actual_sequence.k_append,
                               expected_sequence.k_append);
      expect_byte_ranges_equal(actual_sequence.v_append,
                               expected_sequence.v_append);
      EXPECT_EQ(actual_sequence.layer_index,
                expected_sequence.layer_index);
      EXPECT_EQ(actual_sequence.batch_sequence_index,
                expected_sequence.batch_sequence_index);
      EXPECT_EQ(actual_sequence.append_record_byte_offset,
                expected_sequence.append_record_byte_offset);
    }
  }
}

LlmKvLayoutRequest paged_golden_request() {
  return {35, 16, 2, 2, 32};
}

size_t segment_length_sum(const LlmKvSegmentPlan& plan) {
  size_t total = 0;
  for (size_t length : plan.segment_lengths) {
    total += length;
  }
  return total;
}

void expect_length_prefixed_identity_field(std::string_view identity,
                                           std::string_view name,
                                           std::string_view value) {
  const std::string expected = "|" + std::string(name) + "=" +
                               std::to_string(value.size()) + ":" +
                               std::string(value);
  EXPECT_NE(identity.find(expected), std::string_view::npos)
      << "missing canonical field " << name;
}

class LlmMemoryWorkPlanSystemCallsTest
    : public FakeMemorySystemCallsTest {};

}  // namespace

TEST(LlmMemoryWorkPlanTest, ConstantsAndIdentitiesMatchFrozenContract) {
  EXPECT_EQ(Constants::LLM_DEFAULT_KV_ELEMENT_BYTES, 2u);
  EXPECT_EQ(Constants::LLM_DEFAULT_BATCH_SIZE, 1u);
  EXPECT_EQ(Constants::LLM_DEFAULT_LOOP_COUNT, 3u);
  EXPECT_EQ(Constants::LLM_RANGE_ALIGNMENT_BYTES, 32u);
  EXPECT_DOUBLE_EQ(Constants::LLM_CALIBRATION_TARGET_SECONDS, 0.150);
  EXPECT_DOUBLE_EQ(Constants::LLM_CALIBRATION_MIN_SECONDS, 0.100);
  EXPECT_DOUBLE_EQ(Constants::LLM_CALIBRATION_MAX_SECONDS, 0.250);
  EXPECT_EQ(Constants::LLM_CALIBRATION_MAX_CORRECTIONS, 2u);
  EXPECT_EQ(Constants::LLM_CALIBRATION_MIN_PILOT_BYTES,
            8 * Constants::BYTES_PER_MB);
  EXPECT_EQ(Constants::LLM_MAX_WORK_UNITS_PER_MEASUREMENT, 1000000000u);
  EXPECT_EQ(Constants::LLM_MAX_ACCOUNTED_BYTES_PER_TASK,
            64ULL * 1024ULL * Constants::BYTES_PER_MB);
  EXPECT_DOUBLE_EQ(Constants::LLM_STREAMING_CV_WARNING_PCT, 5.0);
  EXPECT_EQ(Constants::LLM_JSON_SCHEMA_VERSION, 1);
  EXPECT_STREQ(Constants::LLM_JSON_MODE_NAME, "llm_memory");
  EXPECT_EQ(Constants::LLM_WEIGHT_PASSES_PER_WORK_UNIT, 1u);
  EXPECT_EQ(Constants::LLM_KV_REPLAY_FACTOR, 1u);
  EXPECT_STREQ(Constants::LLM_CPU_DECODE_CONTIGUOUS_METHODOLOGY_VERSION,
               "llm-memory-v1-cpu-decode-contiguous");
  EXPECT_STREQ(Constants::LLM_CPU_PREFILL_PAGED_METHODOLOGY_VERSION,
               "llm-memory-v1-cpu-prefill-paged");
  EXPECT_STREQ(Constants::LLM_COMPONENT_IDENTITY_VERSION,
               "llm-memory-components-v1");
  EXPECT_STREQ(Constants::LLM_LOGICAL_PROFILE_VERSION,
               "decode_steady_fixed_context");
  EXPECT_STREQ(Constants::LLM_CONTIGUOUS_KV_LAYOUT_VERSION,
               "contiguous_layer_batch_token_head_dimension");
  EXPECT_STREQ(Constants::LLM_CPU_EXECUTOR_VERSION,
               "llm-cpu-executor-v1-arm64-decode-contiguous");
  EXPECT_STREQ(Constants::LLM_DESCRIPTOR_ABI_VERSION,
               "llm-memory-descriptor-abi-v1");
  EXPECT_STREQ(Constants::LLM_CPU_SCHEDULE_VERSION,
               "worker-local-layer-order-no-per-layer-global-barrier");
  EXPECT_STREQ(Constants::LLM_CPU_TIMER_POLICY_VERSION,
               "synchronized-start-to-last-worker-completion-per-scenario-task");
  EXPECT_STREQ(Constants::LLM_BUFFER_PATTERN_VERSION,
               "llm-buffer-pattern-v1");
  EXPECT_STREQ(Constants::LLM_APPEND_PATTERN_VERSION,
               "llm-kv-append-affine64-v1");
  EXPECT_STREQ(Constants::LLM_READ_CHECKSUM_VERSION,
               "llm-read-checksum-v1");
  EXPECT_EQ(Constants::LLM_KV_BLOCK_TABLE_ENTRY_BYTES, sizeof(uint32_t));
  EXPECT_STREQ(Constants::LLM_PAGED_KV_LAYOUT_VERSION,
               "paged-uint32-block-table-full-blocks-v1");
  EXPECT_STREQ(Constants::LLM_KV_BLOCK_PERMUTATION_VERSION,
               "splitmix64-fisher-yates-rejection-v1");
  EXPECT_STREQ(Constants::LLM_PAGED_CPU_EXECUTION_IDENTITY_VERSION,
               "llm-paged-cpu-execution-v1");
  EXPECT_STREQ(Constants::LLM_PAGED_CPU_SCHEDULE_VERSION,
               "decode-kv-accounted-prefix-balanced-rotating-v1");
  EXPECT_STREQ(Constants::LLM_PAGED_CPU_EXECUTOR_VERSION,
               "llm-cpu-executor-v1-arm64-decode-paged");
  EXPECT_STREQ(Constants::LLM_PAGED_DESCRIPTOR_ABI_VERSION,
               "llm-memory-paged-descriptor-abi-v1");
  EXPECT_STREQ(Constants::LLM_PAGED_BUFFER_PATTERN_VERSION,
               "llm-paged-physical-buffer-pattern-v1");
  EXPECT_STREQ(Constants::LLM_PAGED_READ_CHECKSUM_VERSION,
               "llm-paged-read-checksum-v1");
  EXPECT_STREQ(Constants::LLM_PREFILL_PAGED_CPU_EXECUTOR_VERSION,
               "llm-cpu-executor-v1-arm64-prefill-paged");
  EXPECT_STREQ(Constants::LLM_PREFILL_PAGED_DESCRIPTOR_ABI_VERSION,
               "llm-memory-prefill-paged-descriptor-abi-v1");
  EXPECT_STREQ(LlmPrefillVersion::PAGED_CHECKSUM_ORACLE,
               "llm-prefill-paged-affine64-lookup-v1");
  EXPECT_EQ(build_llm_methodology_version(
                LlmMemoryBackend::Cpu, LlmPhase::Decode,
                LlmKvLayout::Contiguous),
            Constants::LLM_CPU_DECODE_CONTIGUOUS_METHODOLOGY_VERSION);
  EXPECT_EQ(build_llm_methodology_version(
                LlmMemoryBackend::Cpu, LlmPhase::Decode,
                LlmKvLayout::Paged),
            "llm-memory-v1-cpu-decode-paged");
  EXPECT_EQ(build_llm_methodology_version(
                LlmMemoryBackend::Metal, LlmPhase::Prefill,
                LlmKvLayout::Paged),
            "llm-memory-v1-metal-prefill-paged");

  EXPECT_EQ(llm_seed_domain_value(LlmSeedDomain::WeightBuffer),
            0x4C4C4D5745494748ULL);
  EXPECT_EQ(llm_seed_domain_value(LlmSeedDomain::KBuffer),
            0x4C4C4D4B42554631ULL);
  EXPECT_EQ(llm_seed_domain_value(LlmSeedDomain::VBuffer),
            0x4C4C4D5642554631ULL);
  EXPECT_EQ(llm_seed_domain_value(LlmSeedDomain::WeightsOnlyScenario),
            0x4C4C4D5357454947ULL);
  EXPECT_EQ(llm_seed_domain_value(LlmSeedDomain::KvOnlyScenario),
            0x4C4C4D534B564F4EULL);
  EXPECT_EQ(llm_seed_domain_value(LlmSeedDomain::MixedScenario),
            0x4C4C4D534D495845ULL);
  EXPECT_EQ(derive_llm_domain_seed(42, LlmSeedDomain::WeightBuffer),
            0x8BD800BF72EAB1DDULL);
  EXPECT_EQ(derive_llm_domain_seed(42, LlmSeedDomain::KBuffer),
            0x8A0264E87A34112EULL);
  EXPECT_EQ(derive_llm_domain_seed(42, LlmSeedDomain::VBuffer),
            0x44277B99E6E1B8F2ULL);
  EXPECT_EQ(derive_llm_domain_seed(42,
                                   LlmSeedDomain::WeightsOnlyScenario),
            0xB7EE10336825D686ULL);
  EXPECT_EQ(derive_llm_domain_seed(42, LlmSeedDomain::KvOnlyScenario),
            0xBCB71B21E0B42F3FULL);
  EXPECT_EQ(derive_llm_domain_seed(42, LlmSeedDomain::MixedScenario),
            0xBFC7A78C248B8244ULL);
  const auto invalid_domain = static_cast<LlmSeedDomain>(255);
  EXPECT_EQ(llm_seed_domain_value(invalid_domain), 0u);
  EXPECT_EQ(derive_llm_domain_seed(42, invalid_domain), 0u);
}

TEST(LlmMemoryWorkPlanTest,
     LargeGeometryMatchesPhaseZeroGoldenAndSeparateMappings) {
  const LlmGeometry geometry =
      resolve_llm_geometry(large_geometry_request());
  ASSERT_TRUE(geometry.valid) << geometry.reason_code;
  EXPECT_EQ(geometry.phase, LlmPhase::Decode);
  EXPECT_EQ(geometry.kv_layout, LlmKvLayout::Contiguous);
  EXPECT_EQ(geometry.work_unit_kind, LlmWorkUnitKind::DecodeStep);
  ASSERT_TRUE(geometry.decode.has_value());
  EXPECT_EQ(geometry.decode->visible_context_tokens, 8192u);
  EXPECT_FALSE(geometry.prefill.has_value());
  EXPECT_EQ(geometry.attention_kind, LlmAttentionKind::Gqa);
  EXPECT_EQ(geometry.query_heads_per_kv_head, 4u);
  EXPECT_EQ(geometry.kv_vector_bytes, 256u);
  EXPECT_EQ(geometry.k_or_v_record_bytes_per_layer, 2048u);
  EXPECT_EQ(geometry.kv_record_bytes_per_layer, 4096u);
  EXPECT_EQ(geometry.kv_bytes_per_visible_token, 131072u);
  EXPECT_EQ(geometry.k_or_v_sequence_visible_bytes, 16 * Constants::BYTES_PER_MB);
  EXPECT_EQ(geometry.k_mapping_bytes, 512 * Constants::BYTES_PER_MB);
  EXPECT_EQ(geometry.v_mapping_bytes, 512 * Constants::BYTES_PER_MB);
  EXPECT_EQ(geometry.kv_capacity_bytes, kGiB);
  EXPECT_EQ(geometry.weight_read_bytes_per_work_unit, 4 * kGiB);
  EXPECT_EQ(geometry.kv_read_bytes_per_work_unit, kGiB);
  EXPECT_EQ(geometry.kv_write_bytes_per_work_unit, 131072u);
  EXPECT_EQ(geometry.kv_only_effective_model_payload_bytes_per_work_unit,
            1073872896u);
  EXPECT_EQ(geometry.mixed_effective_model_payload_bytes_per_work_unit,
            5368840192u);
  EXPECT_EQ(geometry.total_data_mapping_bytes, 5 * kGiB);
  EXPECT_EQ(geometry.traffic_crossover_numerator, 4294967296u);
  EXPECT_EQ(geometry.traffic_crossover_denominator, 131072u);
  EXPECT_DOUBLE_EQ(geometry.traffic_crossover_context_tokens, 32768.0);
}

TEST(LlmMemoryWorkPlanTest,
     SmallAndBatchedPayloadTotalsKeepWeightsUnscaled) {
  const LlmGeometry single =
      resolve_llm_geometry(small_geometry_request(1));
  ASSERT_TRUE(single.valid);
  EXPECT_EQ(single.kv_bytes_per_visible_token, 128u);
  EXPECT_EQ(single.weight_read_bytes_per_work_unit, 1024u);
  EXPECT_EQ(single.kv_read_bytes_per_work_unit, 384u);
  EXPECT_EQ(single.kv_write_bytes_per_work_unit, 128u);
  EXPECT_EQ(single.kv_only_effective_model_payload_bytes_per_work_unit, 512u);
  EXPECT_EQ(single.mixed_effective_model_payload_bytes_per_work_unit, 1536u);

  const LlmMemoryWorkPlan single_model = build_llm_memory_work_plan(
      work_plan_request(small_geometry_request(1), 2, 2));
  ASSERT_TRUE(single_model.valid) << single_model.reason_code;
  const LlmScenarioWorkPlan weights = build_llm_scenario_work_plan(
      single_model, LlmScenario::WeightsOnly, 4, true);
  const LlmScenarioWorkPlan kv =
      build_llm_scenario_work_plan(single_model, LlmScenario::KvOnly, 4,
                                   true);
  const LlmScenarioWorkPlan mixed =
      build_llm_scenario_work_plan(single_model, LlmScenario::Mixed, 4,
                                   true);
  ASSERT_TRUE(weights.valid);
  ASSERT_TRUE(kv.valid);
  ASSERT_TRUE(mixed.valid);
  EXPECT_EQ(weights.work_unit_kind, LlmWorkUnitKind::DecodeStep);
  EXPECT_EQ(weights.kv_write_kind, LlmKvWriteKind::None);
  EXPECT_EQ(kv.work_unit_kind, LlmWorkUnitKind::DecodeStep);
  EXPECT_EQ(kv.kv_write_kind, LlmKvWriteKind::CurrentTokenAppend);
  EXPECT_EQ(mixed.work_unit_kind, LlmWorkUnitKind::DecodeStep);
  EXPECT_EQ(mixed.kv_write_kind, LlmKvWriteKind::CurrentTokenAppend);
  EXPECT_EQ(weights.weight_read_bytes, 4096u);
  EXPECT_EQ(weights.kv_read_bytes, 0u);
  EXPECT_EQ(weights.kv_write_bytes, 0u);
  EXPECT_EQ(weights.effective_model_payload_bytes, 4096u);
  EXPECT_EQ(kv.weight_read_bytes, 0u);
  EXPECT_EQ(kv.kv_read_bytes, 1536u);
  EXPECT_EQ(kv.kv_write_bytes, 512u);
  EXPECT_EQ(kv.effective_model_payload_bytes, 2048u);
  EXPECT_EQ(mixed.weight_read_bytes, 4096u);
  EXPECT_EQ(mixed.kv_read_bytes, 1536u);
  EXPECT_EQ(mixed.kv_write_bytes, 512u);
  EXPECT_EQ(mixed.effective_model_payload_bytes, 6144u);
  for (const LlmScenarioWorkPlan* scenario : {&weights, &kv, &mixed}) {
    EXPECT_EQ(scenario->layout_metadata_lookup_count_per_work_unit, 0u);
    EXPECT_EQ(scenario->layout_metadata_read_bytes_per_work_unit, 0u);
    EXPECT_EQ(scenario->accounted_bytes_per_work_unit,
              scenario->effective_model_payload_bytes_per_work_unit);
    EXPECT_EQ(scenario->layout_metadata_lookup_count, 0u);
    EXPECT_EQ(scenario->layout_metadata_read_bytes, 0u);
    EXPECT_EQ(scenario->task_accounted_bytes,
              scenario->effective_model_payload_bytes);
  }

  const LlmGeometry batched =
      resolve_llm_geometry(small_geometry_request(2));
  ASSERT_TRUE(batched.valid);
  EXPECT_EQ(batched.weight_read_bytes_per_work_unit, 1024u);
  EXPECT_EQ(batched.kv_read_bytes_per_work_unit, 768u);
  EXPECT_EQ(batched.kv_write_bytes_per_work_unit, 256u);
  const LlmMemoryWorkPlan batched_model = build_llm_memory_work_plan(
      work_plan_request(small_geometry_request(2), 2, 2));
  ASSERT_TRUE(batched_model.valid) << batched_model.reason_code;
  const LlmScenarioWorkPlan batched_mixed = build_llm_scenario_work_plan(
      batched_model, LlmScenario::Mixed, 4, true);
  ASSERT_TRUE(batched_mixed.valid);
  EXPECT_EQ(batched_mixed.weight_read_bytes, 4096u);
  EXPECT_EQ(batched_mixed.kv_read_bytes, 3072u);
  EXPECT_EQ(batched_mixed.kv_write_bytes, 1024u);
  EXPECT_EQ(batched_mixed.effective_model_payload_bytes, 8192u);
}

TEST(LlmMemoryWorkPlanTest, ClassifiesMhaGqaAndMqaFromHeadSharing) {
  LlmGeometryRequest request = small_geometry_request();
  request.query_head_count = 2;
  request.kv_head_count = 2;
  const LlmGeometry mha = resolve_llm_geometry(request);
  ASSERT_TRUE(mha.valid);
  EXPECT_EQ(mha.attention_kind, LlmAttentionKind::Mha);
  EXPECT_EQ(mha.query_heads_per_kv_head, 1u);

  request.query_head_count = 4;
  const LlmGeometry gqa = resolve_llm_geometry(request);
  ASSERT_TRUE(gqa.valid);
  EXPECT_EQ(gqa.attention_kind, LlmAttentionKind::Gqa);
  EXPECT_EQ(gqa.query_heads_per_kv_head, 2u);
  request.query_head_count = 8;
  const LlmGeometry wider_queries = resolve_llm_geometry(request);
  ASSERT_TRUE(wider_queries.valid);
  EXPECT_EQ(wider_queries.kv_bytes_per_visible_token,
            gqa.kv_bytes_per_visible_token);

  request.query_head_count = 4;
  request.kv_head_count = 1;
  const LlmGeometry mqa = resolve_llm_geometry(request);
  ASSERT_TRUE(mqa.valid);
  EXPECT_EQ(mqa.attention_kind, LlmAttentionKind::Mqa);
  EXPECT_EQ(mqa.query_heads_per_kv_head, 4u);

  request.query_head_count = 1;
  const LlmGeometry one_by_one = resolve_llm_geometry(request);
  ASSERT_TRUE(one_by_one.valid);
  EXPECT_EQ(one_by_one.attention_kind, LlmAttentionKind::Mha);
}

TEST(LlmMemoryWorkPlanTest,
     RawGeometryRejectsSemanticErrorsAndPlannerPropagatesReason) {
  struct InvalidCase {
    LlmGeometryRequest request;
    const char* reason_code;
  };
  const std::array<InvalidCase, 10> invalid_cases = {{
      {{0, 2, 4, 2, 8, 2, 3, 1},
       LlmWorkPlanReason::ACTIVE_WEIGHT_BYTES_ZERO},
      {{1024, 0, 4, 2, 8, 2, 3, 1},
       LlmWorkPlanReason::LAYER_COUNT_ZERO},
      {{1024, 2, 0, 2, 8, 2, 3, 1},
       LlmWorkPlanReason::QUERY_HEAD_COUNT_ZERO},
      {{1024, 2, 4, 0, 8, 2, 3, 1},
       LlmWorkPlanReason::KV_HEAD_COUNT_ZERO},
      {{1024, 2, 4, 2, 0, 2, 3, 1},
       LlmWorkPlanReason::HEAD_DIMENSION_ZERO},
      {{1024, 2, 4, 2, 8, 3, 3, 1},
       LlmWorkPlanReason::INVALID_KV_ELEMENT_BYTES},
      {{1024, 2, 4, 2, 8, 2, 0, 1},
       LlmWorkPlanReason::CONTEXT_TOKENS_ZERO},
      {{1024, 2, 4, 2, 8, 2, 3, 0},
       LlmWorkPlanReason::BATCH_SIZE_ZERO},
      {{1024, 2, 1, 2, 8, 2, 3, 1},
       LlmWorkPlanReason::QUERY_HEADS_BELOW_KV_HEADS},
      {{1024, 2, 3, 2, 8, 2, 3, 1},
       LlmWorkPlanReason::QUERY_HEADS_NOT_DIVISIBLE_BY_KV_HEADS},
  }};

  for (const InvalidCase& invalid_case : invalid_cases) {
    SCOPED_TRACE(invalid_case.reason_code);
    const LlmGeometry geometry = resolve_llm_geometry(invalid_case.request);
    EXPECT_FALSE(geometry.valid);
    EXPECT_EQ(geometry.reason_code, invalid_case.reason_code);
  }

  expect_invalid_plan(
      build_llm_memory_work_plan(
          work_plan_request(invalid_cases[5].request, 2, 2)),
      LlmWorkPlanReason::INVALID_KV_ELEMENT_BYTES);

  LlmGeometryRequest incompatible_geometry = small_geometry_request();
  incompatible_geometry.phase = LlmPhase::Prefill;
  EXPECT_EQ(resolve_llm_geometry(incompatible_geometry).reason_code,
            LlmWorkPlanReason::CONTEXT_TOKENS_NOT_APPLICABLE);
  incompatible_geometry.phase = LlmPhase::Decode;
  incompatible_geometry.kv_layout = LlmKvLayout::Paged;
  EXPECT_EQ(resolve_llm_geometry(incompatible_geometry).reason_code,
            LlmKvLayoutReason::BLOCK_TOKENS_ZERO);

  LlmMemoryWorkPlanRequest inactive_backend =
      work_plan_request(small_geometry_request(), 2, 2);
  inactive_backend.backend = LlmMemoryBackend::Metal;
  expect_invalid_plan(build_llm_memory_work_plan(inactive_backend),
                      LlmWorkPlanReason::METAL_WORKERS_NOT_APPLICABLE);
}

TEST(LlmMemoryWorkPlanTest,
     RejectsUnsafeJsonIntegersBeforePlanningOrAllocation) {
  constexpr size_t kUnsafeInteger =
      Constants::LLM_JSON_MAX_SAFE_INTEGER + 1;

  LlmGeometryRequest geometry = small_geometry_request();
  geometry.query_head_count = kUnsafeInteger;
  geometry.kv_head_count = 1;
  expect_invalid_plan(build_llm_memory_work_plan(
                          work_plan_request(geometry, 1, 1)),
                      LlmWorkPlanReason::JSON_INTEGER_OUT_OF_RANGE);

  LlmMemoryWorkPlanRequest request =
      work_plan_request(small_geometry_request());
  request.requested_workers = kUnsafeInteger;
  expect_invalid_plan(build_llm_memory_work_plan(request),
                      LlmWorkPlanReason::JSON_INTEGER_OUT_OF_RANGE);

  request = work_plan_request(small_geometry_request());
  request.available_workers = kUnsafeInteger;
  expect_invalid_plan(build_llm_memory_work_plan(request),
                      LlmWorkPlanReason::JSON_INTEGER_OUT_OF_RANGE);

  geometry = {100000000, 100000000, 1, 1, 1, 1, 1, 100000000};
  request = work_plan_request(geometry, 1, 1);
  request.available_memory_bytes = std::numeric_limits<size_t>::max();
  expect_invalid_plan(build_llm_memory_work_plan(request),
                      LlmWorkPlanReason::JSON_INTEGER_OUT_OF_RANGE);
}

TEST(LlmMemoryWorkPlanTest, CrossoverKeepsExactRationalAndDoubleEstimate) {
  LlmGeometryRequest request = small_geometry_request(3);
  request.active_weight_bytes = 1000;
  const LlmGeometry geometry = resolve_llm_geometry(request);
  ASSERT_TRUE(geometry.valid);
  EXPECT_EQ(geometry.traffic_crossover_numerator, 1000u);
  EXPECT_EQ(geometry.traffic_crossover_denominator, 384u);
  EXPECT_NEAR(geometry.traffic_crossover_context_tokens,
              1000.0 / 384.0, 1e-15);
}

TEST(LlmMemoryWorkPlanTest,
     WeightLayersUseQuotientRemainderWithoutDroppedBytes) {
  LlmGeometryRequest geometry = small_geometry_request();
  geometry.active_weight_bytes = 1025;
  geometry.layer_count = 3;
  const LlmMemoryWorkPlan plan =
      build_llm_memory_work_plan(work_plan_request(geometry, 2, 2));
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  ASSERT_EQ(plan.weight_layers.size(), 3u);
  EXPECT_EQ(plan.weight_layers[0].offset_bytes, 0u);
  EXPECT_EQ(plan.weight_layers[0].span_bytes, 342u);
  EXPECT_EQ(plan.weight_layers[1].offset_bytes, 342u);
  EXPECT_EQ(plan.weight_layers[1].span_bytes, 342u);
  EXPECT_EQ(plan.weight_layers[2].offset_bytes, 684u);
  EXPECT_EQ(plan.weight_layers[2].span_bytes, 341u);
  EXPECT_EQ(plan.weight_layers.back().offset_bytes +
                plan.weight_layers.back().span_bytes,
            1025u);
  for (size_t layer = 0; layer < plan.weight_layers.size(); ++layer) {
    const LlmByteRange worker_union = union_for_weight_layer(plan, layer);
    EXPECT_EQ(worker_union.offset_bytes,
              plan.weight_layers[layer].offset_bytes);
    EXPECT_EQ(worker_union.span_bytes, plan.weight_layers[layer].span_bytes);
  }
}

TEST(LlmMemoryWorkPlanTest, LayerBatchTokenLayoutOffsetsAreExact) {
  const LlmGeometryRequest geometry = {1024, 2, 4, 2, 8, 2, 3, 2};
  const LlmMemoryWorkPlan plan =
      build_llm_memory_work_plan(work_plan_request(geometry, 2, 2));
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  ASSERT_EQ(cpu_execution_plan(plan).sequence_descriptors_per_worker, 4u);
  const std::array<size_t, 4> expected_visible = {0, 96, 192, 288};
  const std::array<size_t, 4> expected_append = {64, 160, 256, 352};
  for (size_t sequence = 0; sequence < expected_visible.size(); ++sequence) {
    const LlmByteRange k_visible = union_for_sequence(
        plan, sequence, &LlmKvSequenceRangeTemplate::k_visible);
    const LlmByteRange v_visible = union_for_sequence(
        plan, sequence, &LlmKvSequenceRangeTemplate::v_visible);
    const LlmByteRange k_append = union_for_sequence(
        plan, sequence, &LlmKvSequenceRangeTemplate::k_append);
    const LlmByteRange v_append = union_for_sequence(
        plan, sequence, &LlmKvSequenceRangeTemplate::v_append);
    EXPECT_EQ(k_visible.offset_bytes, expected_visible[sequence]);
    EXPECT_EQ(k_visible.span_bytes, 96u);
    EXPECT_EQ(v_visible.offset_bytes, expected_visible[sequence]);
    EXPECT_EQ(v_visible.span_bytes, 96u);
    EXPECT_EQ(k_append.offset_bytes, expected_append[sequence]);
    EXPECT_EQ(k_append.span_bytes, 32u);
    EXPECT_EQ(v_append.offset_bytes, expected_append[sequence]);
    EXPECT_EQ(v_append.span_bytes, 32u);
  }
  EXPECT_EQ(plan.geometry.k_mapping_bytes, 384u);
  EXPECT_EQ(plan.geometry.v_mapping_bytes, 384u);
}

TEST(LlmMemoryWorkPlanTest,
     WorkerRangesAreBoundedDisjointExactAndAlignedWhenPossible) {
  for (size_t span : {1u, 31u, 32u, 33u, 63u, 64u, 65u, 95u, 96u,
                      97u}) {
    const size_t workers = std::min<size_t>(3, span);
    const LlmGeometryRequest geometry = {span, 1, 1, 1, span, 1, 1, 1};
    const LlmMemoryWorkPlan plan = build_llm_memory_work_plan(
        work_plan_request(geometry, workers, workers));
    ASSERT_TRUE(plan.valid) << "span=" << span << " " << plan.reason_code;
    const LlmCpuExecutionPlan& cpu_plan = cpu_execution_plan(plan);
    ASSERT_EQ(cpu_plan.effective_workers, workers);

    size_t next_offset = 0;
    for (size_t worker = 0; worker < workers; ++worker) {
      const LlmByteRange& range =
          cpu_plan.workers[worker].sequences[0].k_visible;
      EXPECT_EQ(range.offset_bytes, next_offset) << "span=" << span;
      EXPECT_GT(range.span_bytes, 0u) << "span=" << span;
      next_offset += range.span_bytes;
      if (worker + 1 == workers) {
        continue;
      }
      const size_t remaining_workers = workers - worker - 1;
      const size_t minimum_boundary =
          cpu_plan.workers[worker].sequences[0].k_visible.offset_bytes + 1;
      const size_t maximum_boundary = span - remaining_workers;
      const size_t first_aligned =
          ((minimum_boundary + 31) / 32) * 32;
      if (first_aligned <= maximum_boundary) {
        EXPECT_EQ(next_offset % 32, 0u) << "span=" << span;
      }
    }
    EXPECT_EQ(next_offset, span);
  }
}

TEST(LlmMemoryWorkPlanTest,
     WeightPartitionsUseAbsoluteAlignmentForNonzeroLayerOffsets) {
  const LlmGeometryRequest geometry = {199, 2, 1, 1, 128, 1, 1, 1};
  const LlmMemoryWorkPlan plan =
      build_llm_memory_work_plan(work_plan_request(geometry, 3, 3));
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  const LlmCpuExecutionPlan& cpu_plan = cpu_execution_plan(plan);
  ASSERT_EQ(cpu_plan.effective_workers, 3u);
  ASSERT_EQ(plan.weight_layers.size(), 2u);
  EXPECT_EQ(plan.weight_layers[1].offset_bytes, 100u);
  EXPECT_EQ(plan.weight_layers[1].span_bytes, 99u);

  size_t cursor = plan.weight_layers[1].offset_bytes;
  for (size_t worker = 0; worker < cpu_plan.effective_workers; ++worker) {
    const LlmByteRange& range = cpu_plan.workers[worker].layers[1].weight;
    EXPECT_EQ(range.offset_bytes, cursor);
    EXPECT_GT(range.span_bytes, 0u);
    cursor += range.span_bytes;
    if (worker + 1 < cpu_plan.effective_workers) {
      EXPECT_EQ(cursor % Constants::LLM_RANGE_ALIGNMENT_BYTES, 0u);
    }
  }
  EXPECT_EQ(cursor, 199u);
}

TEST(LlmMemoryWorkPlanTest,
     AppendIntersectionsPreserveMidWordRecordByteOffsets) {
  const LlmGeometryRequest geometry = {64, 1, 1, 1, 19, 1, 2, 1};
  const LlmMemoryWorkPlan plan =
      build_llm_memory_work_plan(work_plan_request(geometry, 2, 2));
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  const LlmCpuExecutionPlan& cpu_plan = cpu_execution_plan(plan);
  ASSERT_EQ(cpu_plan.workers.size(), 2u);
  const LlmKvSequenceRangeTemplate& first =
      cpu_plan.workers[0].sequences[0];
  const LlmKvSequenceRangeTemplate& second =
      cpu_plan.workers[1].sequences[0];
  EXPECT_EQ(first.k_visible.offset_bytes, 0u);
  EXPECT_EQ(first.k_visible.span_bytes, 32u);
  EXPECT_EQ(second.k_visible.offset_bytes, 32u);
  EXPECT_EQ(second.k_visible.span_bytes, 6u);
  EXPECT_EQ(first.k_append.offset_bytes, 19u);
  EXPECT_EQ(first.k_append.span_bytes, 13u);
  EXPECT_EQ(second.k_append.offset_bytes, 32u);
  EXPECT_EQ(second.k_append.span_bytes, 6u);
  EXPECT_EQ(first.append_record_byte_offset, 0u);
  EXPECT_EQ(second.append_record_byte_offset, 13u);
  EXPECT_EQ(first.v_visible.offset_bytes, first.k_visible.offset_bytes);
  EXPECT_EQ(first.v_append.span_bytes, first.k_append.span_bytes);
  EXPECT_EQ(second.v_append.offset_bytes, second.k_append.offset_bytes);
}

TEST(LlmMemoryWorkPlanTest,
     BatchDescriptorsStaySeparateAndLayerLinksRemainLocal) {
  const LlmGeometryRequest geometry = {1024, 2, 4, 2, 8, 2, 3, 2};
  const LlmMemoryWorkPlan plan =
      build_llm_memory_work_plan(work_plan_request(geometry, 2, 2));
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  const LlmCpuExecutionPlan& cpu_plan = cpu_execution_plan(plan);
  EXPECT_EQ(cpu_plan.layer_descriptors_per_worker, 2u);
  EXPECT_EQ(cpu_plan.sequence_descriptors_per_worker, 4u);
  EXPECT_EQ(cpu_plan.total_layer_descriptors, 4u);
  EXPECT_EQ(cpu_plan.total_sequence_descriptors, 8u);
  for (const LlmWorkerWorkPlan& worker : cpu_plan.workers) {
    ASSERT_EQ(worker.layers.size(), 2u);
    ASSERT_EQ(worker.sequences.size(), 4u);
    for (size_t layer = 0; layer < 2; ++layer) {
      EXPECT_EQ(worker.layers[layer].first_sequence_index, layer * 2);
      EXPECT_EQ(worker.layers[layer].sequence_count, 2u);
      EXPECT_EQ(worker.layers[layer].layer_index, layer);
      for (size_t batch = 0; batch < 2; ++batch) {
        const LlmKvSequenceRangeTemplate& sequence =
            worker.sequences[layer * 2 + batch];
        EXPECT_EQ(sequence.layer_index, layer);
        EXPECT_EQ(sequence.batch_sequence_index, batch);
      }
    }
    const size_t batch_zero_end =
        worker.sequences[0].k_visible.offset_bytes +
        worker.sequences[0].k_visible.span_bytes;
    EXPECT_LE(batch_zero_end, worker.sequences[1].k_visible.offset_bytes);
  }
}

TEST(LlmMemoryWorkPlanTest,
     PlannerStorageUsesActualRetainedCapacitiesAndEntersPeakBudget) {
  LlmMemoryWorkPlanRequest request =
      work_plan_request(small_geometry_request(2), 3, 3);
  request.checksum_auxiliary_bytes = 7;
  request.orchestration_auxiliary_bytes = 9;
  const LlmMemoryWorkPlan plan = build_llm_memory_work_plan(request);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  const LlmCpuExecutionPlan& cpu_plan = cpu_execution_plan(plan);

  size_t actual_capacity_bytes =
      plan.weight_layers.capacity() * sizeof(LlmByteRange) +
      cpu_plan.workers.capacity() * sizeof(LlmWorkerWorkPlan);
  for (const LlmWorkerWorkPlan& worker : cpu_plan.workers) {
    actual_capacity_bytes +=
        worker.layers.capacity() * sizeof(LlmLayerRangeTemplate);
    actual_capacity_bytes +=
        worker.sequences.capacity() * sizeof(LlmKvSequenceRangeTemplate);
  }
  const size_t requested_element_bytes =
      plan.geometry.layer_count * sizeof(LlmByteRange) +
      cpu_plan.effective_workers * sizeof(LlmWorkerWorkPlan) +
      cpu_plan.effective_workers * plan.geometry.layer_count *
          sizeof(LlmLayerRangeTemplate) +
      cpu_plan.effective_workers * plan.geometry.layer_count *
          plan.geometry.batch_size * sizeof(LlmKvSequenceRangeTemplate);
  EXPECT_GE(actual_capacity_bytes, requested_element_bytes);
  EXPECT_EQ(cpu_plan.planner_storage_bytes, actual_capacity_bytes);
  EXPECT_EQ(plan.memory_budget.request.planner_storage_bytes,
            actual_capacity_bytes);
  EXPECT_EQ(plan.memory_budget.request.auxiliary_bytes,
            cpu_plan.descriptor_bytes + actual_capacity_bytes + 7 + 9);
  EXPECT_EQ(plan.memory_budget.request.required_total_bytes,
            plan.memory_budget.request.committed_data_bytes +
                plan.memory_budget.request.auxiliary_bytes);
}

TEST(LlmMemoryWorkPlanTest,
     ConfigAdapterForwardsResolvedFieldsBudgetInputsAndValidation) {
  LlmMemoryConfig config;
  config.weight_size_mb = 1;
  config.layer_count = 2;
  config.query_head_count = 4;
  config.kv_head_count = 2;
  config.head_dimension = 8;
  config.kv_element_bytes = 4;
  config.visible_context_tokens = 3;
  config.batch_size = 2;
  config.requested_workers = 3;
  config.seed = 99;

  const LlmMemoryWorkPlan plan = build_llm_memory_work_plan(
      config, 5, 16 * kGiB, 4096, 7, 9);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  EXPECT_EQ(plan.geometry.active_weight_bytes_per_work_unit,
            Constants::BYTES_PER_MB);
  EXPECT_EQ(plan.geometry.layer_count, 2u);
  EXPECT_EQ(plan.geometry.query_head_count, 4u);
  EXPECT_EQ(plan.geometry.kv_head_count, 2u);
  EXPECT_EQ(plan.geometry.head_dimension, 8u);
  EXPECT_EQ(plan.geometry.kv_element_bytes, 4u);
  ASSERT_TRUE(plan.geometry.decode.has_value());
  EXPECT_EQ(plan.geometry.decode->visible_context_tokens, 3u);
  EXPECT_FALSE(plan.geometry.prefill.has_value());
  EXPECT_EQ(plan.geometry.batch_size, 2u);
  EXPECT_EQ(plan.backend, LlmMemoryBackend::Cpu);
  EXPECT_EQ(plan.phase, LlmPhase::Decode);
  EXPECT_EQ(plan.kv_layout, LlmKvLayout::Contiguous);
  EXPECT_EQ(plan.work_unit_kind, LlmWorkUnitKind::DecodeStep);
  EXPECT_EQ(cpu_execution_plan(plan).requested_workers, 3u);
  EXPECT_EQ(cpu_execution_plan(plan).available_workers, 5u);
  EXPECT_EQ(cpu_execution_plan(plan).effective_workers, 3u);
  EXPECT_EQ(plan.base_seed, 99u);
  EXPECT_EQ(plan.memory_budget.available_memory_bytes, 16 * kGiB);
  EXPECT_FALSE(plan.memory_budget.used_fallback);
  EXPECT_EQ(plan.memory_budget.request.mapping_granularity_bytes, 4096u);
  EXPECT_EQ(plan.memory_budget.request.checksum_auxiliary_bytes, 7u);
  EXPECT_EQ(plan.memory_budget.request.orchestration_auxiliary_bytes, 9u);

  config.weight_size_mb = 0;
  expect_invalid_plan(
      build_llm_memory_work_plan(config, 5, 16 * kGiB, 4096, 7, 9),
      LlmMemoryConfigReason::WEIGHT_SIZE_REQUIRED);
}

TEST(LlmMemoryWorkPlanTest,
     RequestedWorkersReduceUntilEveryWorkerHasExecutableWork) {
  const LlmGeometryRequest geometry = {2, 2, 1, 1, 1, 1, 1, 1};
  const LlmMemoryWorkPlan plan =
      build_llm_memory_work_plan(work_plan_request(geometry, 100, 100));
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  const LlmCpuExecutionPlan& cpu_plan = cpu_execution_plan(plan);
  EXPECT_EQ(cpu_plan.requested_workers, 100u);
  EXPECT_EQ(cpu_plan.available_workers, 100u);
  EXPECT_EQ(cpu_plan.effective_workers, 1u);
  EXPECT_EQ(plan.geometry.active_weight_bytes_per_work_unit, 2u);
  EXPECT_EQ(plan.geometry.kv_capacity_bytes, 4u);
  ASSERT_EQ(cpu_plan.workers.size(), 1u);
  EXPECT_FALSE(cpu_plan.workers[0].layers.empty());
  EXPECT_FALSE(cpu_plan.workers[0].sequences.empty());
}

TEST(LlmMemoryWorkPlanTest,
     AvailableWorkersCapAnOtherwiseExecutableRequestedTeam) {
  const LlmGeometryRequest geometry = {100, 1, 1, 1, 8, 1, 1, 1};
  const LlmMemoryWorkPlan plan =
      build_llm_memory_work_plan(work_plan_request(geometry, 5, 2));
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  const LlmCpuExecutionPlan& cpu_plan = cpu_execution_plan(plan);
  EXPECT_EQ(cpu_plan.requested_workers, 5u);
  EXPECT_EQ(cpu_plan.available_workers, 2u);
  EXPECT_EQ(cpu_plan.effective_workers, 2u);
  ASSERT_EQ(cpu_plan.workers.size(), 2u);
  for (const LlmWorkerWorkPlan& worker : cpu_plan.workers) {
    EXPECT_GT(worker.layers[0].weight.span_bytes, 0u);
    EXPECT_GT(worker.sequences[0].k_visible.span_bytes, 0u);
  }
}

TEST(LlmMemoryWorkPlanTest,
     SmallerLayerSpansRemainExactWithoutScenarioEmptyWorkers) {
  // The three-worker team has KV work for every worker. The two-byte second
  // weight layer still retains its exact union with one canonical empty range;
  // that worker has weight work in the first layer.
  const LlmGeometryRequest geometry = {5, 2, 1, 1, 4, 1, 1, 1};
  const LlmMemoryWorkPlan plan =
      build_llm_memory_work_plan(work_plan_request(geometry, 4, 4));
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  const LlmCpuExecutionPlan& cpu_plan = cpu_execution_plan(plan);
  ASSERT_EQ(cpu_plan.effective_workers, 3u);

  size_t second_layer_bytes = 0;
  size_t empty_second_layer_ranges = 0;
  size_t first_sequence_bytes = 0;
  size_t empty_first_sequence_ranges = 0;
  for (const LlmWorkerWorkPlan& worker : cpu_plan.workers) {
    second_layer_bytes += worker.layers[1].weight.span_bytes;
    empty_second_layer_ranges +=
        worker.layers[1].weight.span_bytes == 0 ? 1 : 0;
    first_sequence_bytes += worker.sequences[0].k_visible.span_bytes;
    empty_first_sequence_ranges +=
        worker.sequences[0].k_visible.span_bytes == 0 ? 1 : 0;
    if (worker.layers[1].weight.span_bytes == 0) {
      EXPECT_EQ(worker.layers[1].weight.offset_bytes, 0u);
    }
    if (worker.sequences[0].k_visible.span_bytes == 0) {
      EXPECT_EQ(worker.sequences[0].k_visible.offset_bytes, 0u);
    }
  }
  EXPECT_EQ(second_layer_bytes, 2u);
  EXPECT_EQ(empty_second_layer_ranges, 1u);
  EXPECT_EQ(first_sequence_bytes, 4u);
  EXPECT_EQ(empty_first_sequence_ranges, 0u);

  for (const LlmWorkerWorkPlan& worker : cpu_plan.workers) {
    size_t worker_weight_bytes = 0;
    size_t worker_kv_bytes = 0;
    for (const LlmLayerRangeTemplate& layer : worker.layers) {
      worker_weight_bytes += layer.weight.span_bytes;
    }
    for (const LlmKvSequenceRangeTemplate& sequence : worker.sequences) {
      worker_kv_bytes += sequence.k_visible.span_bytes;
    }
    EXPECT_GT(worker_weight_bytes, 0u);
    EXPECT_GT(worker_kv_bytes, 0u);
  }
}

TEST(LlmMemoryWorkPlanTest, DescriptorAbiMatchesPhaseZeroGoldenOffsets) {
  static_assert(!std::is_copy_constructible_v<LlmMemoryWorkPlan>);
  static_assert(!std::is_copy_assignable_v<LlmMemoryWorkPlan>);
  static_assert(std::is_nothrow_move_constructible_v<LlmMemoryWorkPlan>);
  static_assert(std::is_nothrow_move_assignable_v<LlmMemoryWorkPlan>);
  static_assert(std::is_standard_layout_v<LlmLayerDescriptor>);
  static_assert(std::is_standard_layout_v<LlmKvSequenceDescriptor>);
  static_assert(std::is_standard_layout_v<LlmPagedLayerDescriptor>);
  static_assert(
      std::is_standard_layout_v<LlmPagedKvAssignmentDescriptor>);
  EXPECT_EQ(alignof(LlmLayerDescriptor), 16u);
  EXPECT_EQ(sizeof(LlmLayerDescriptor), 48u);
  EXPECT_EQ(offsetof(LlmLayerDescriptor, weight_ptr), 0u);
  EXPECT_EQ(offsetof(LlmLayerDescriptor, weight_bytes), 8u);
  EXPECT_EQ(offsetof(LlmLayerDescriptor, first_sequence_index), 16u);
  EXPECT_EQ(offsetof(LlmLayerDescriptor, sequence_count), 24u);
  EXPECT_EQ(offsetof(LlmLayerDescriptor, layer_index), 32u);
  EXPECT_EQ(offsetof(LlmLayerDescriptor, reserved_zero), 40u);
  EXPECT_EQ(alignof(LlmKvSequenceDescriptor), 16u);
  EXPECT_EQ(sizeof(LlmKvSequenceDescriptor), 80u);
  EXPECT_EQ(offsetof(LlmKvSequenceDescriptor, k_visible_ptr), 0u);
  EXPECT_EQ(offsetof(LlmKvSequenceDescriptor, k_visible_bytes), 8u);
  EXPECT_EQ(offsetof(LlmKvSequenceDescriptor, v_visible_ptr), 16u);
  EXPECT_EQ(offsetof(LlmKvSequenceDescriptor, v_visible_bytes), 24u);
  EXPECT_EQ(offsetof(LlmKvSequenceDescriptor, k_append_ptr), 32u);
  EXPECT_EQ(offsetof(LlmKvSequenceDescriptor, k_append_bytes), 40u);
  EXPECT_EQ(offsetof(LlmKvSequenceDescriptor, v_append_ptr), 48u);
  EXPECT_EQ(offsetof(LlmKvSequenceDescriptor, v_append_bytes), 56u);
  EXPECT_EQ(offsetof(LlmKvSequenceDescriptor, batch_sequence_index), 64u);
  EXPECT_EQ(offsetof(LlmKvSequenceDescriptor, append_record_byte_offset),
            72u);
  EXPECT_EQ(alignof(LlmPagedLayerDescriptor), 16u);
  EXPECT_EQ(sizeof(LlmPagedLayerDescriptor), 48u);
  EXPECT_EQ(offsetof(LlmPagedLayerDescriptor, weight_ptr), 0u);
  EXPECT_EQ(offsetof(LlmPagedLayerDescriptor, weight_bytes), 8u);
  EXPECT_EQ(offsetof(LlmPagedLayerDescriptor, first_assignment_index),
            16u);
  EXPECT_EQ(offsetof(LlmPagedLayerDescriptor, assignment_count), 24u);
  EXPECT_EQ(offsetof(LlmPagedLayerDescriptor, layer_index), 32u);
  EXPECT_EQ(offsetof(LlmPagedLayerDescriptor, reserved_zero), 40u);
  EXPECT_EQ(alignof(LlmPagedKvAssignmentDescriptor), 16u);
  EXPECT_EQ(sizeof(LlmPagedKvAssignmentDescriptor), 96u);
  EXPECT_EQ(offsetof(LlmPagedKvAssignmentDescriptor, block_table_row),
            0u);
  EXPECT_EQ(offsetof(LlmPagedKvAssignmentDescriptor, k_layer_pool), 8u);
  EXPECT_EQ(offsetof(LlmPagedKvAssignmentDescriptor, v_layer_pool), 16u);
  EXPECT_EQ(offsetof(LlmPagedKvAssignmentDescriptor,
                     first_logical_block),
            24u);
  EXPECT_EQ(offsetof(LlmPagedKvAssignmentDescriptor, owned_block_count),
            32u);
  EXPECT_EQ(offsetof(LlmPagedKvAssignmentDescriptor, blocks_per_sequence),
            40u);
  EXPECT_EQ(offsetof(LlmPagedKvAssignmentDescriptor, block_bytes), 48u);
  EXPECT_EQ(offsetof(LlmPagedKvAssignmentDescriptor,
                     last_block_valid_bytes),
            56u);
  EXPECT_EQ(offsetof(LlmPagedKvAssignmentDescriptor,
                     decode_append_offset),
            64u);
  EXPECT_EQ(offsetof(LlmPagedKvAssignmentDescriptor, append_record_bytes),
            72u);
  EXPECT_EQ(offsetof(LlmPagedKvAssignmentDescriptor, layer_index), 80u);
  EXPECT_EQ(offsetof(LlmPagedKvAssignmentDescriptor,
                     batch_sequence_index),
            88u);
}

TEST(LlmMemoryWorkPlanTest,
     CpuExecutionPlanAccessorsRequireMatchingBackendAndVariant) {
  LlmMemoryWorkPlan plan;
  EXPECT_EQ(get_llm_cpu_execution_plan(plan),
            &cpu_execution_plan(plan));
  const LlmMemoryWorkPlan& const_plan = plan;
  EXPECT_EQ(get_llm_cpu_execution_plan(const_plan),
            &cpu_execution_plan(const_plan));

  plan.backend_execution_plan = LlmMetalExecutionPlan{};
  EXPECT_EQ(get_llm_cpu_execution_plan(plan), nullptr);
  EXPECT_EQ(get_llm_cpu_execution_plan(const_plan), nullptr);

  plan.backend_execution_plan = LlmCpuExecutionPlan{};
  plan.backend = LlmMemoryBackend::Metal;
  EXPECT_EQ(get_llm_cpu_execution_plan(plan), nullptr);
  EXPECT_EQ(get_llm_cpu_execution_plan(const_plan), nullptr);
}

TEST(LlmMemoryWorkPlanTest,
     MetalDecodeContiguousDraftAndFinalPlanStayWorkerFree) {
  LlmMemoryWorkPlanDraft draft =
      prepare_llm_memory_work_plan(metal_work_plan_request());
  ASSERT_TRUE(draft.valid) << draft.reason_code;
  EXPECT_FALSE(draft.candidate.valid);
  EXPECT_EQ(draft.candidate.reason_code, LlmWorkPlanReason::VALID);
  ASSERT_TRUE(draft.candidate.geometry.valid);
  ASSERT_EQ(draft.candidate.weight_layers.size(),
            draft.candidate.geometry.layer_count);
  ASSERT_TRUE(draft.auxiliary_preflight.valid);
  EXPECT_EQ(draft.auxiliary_preflight.backend,
            LlmMemoryBackend::Metal);
  EXPECT_EQ(draft.auxiliary_preflight.effective_workers, 0u);
  EXPECT_EQ(draft.candidate.methodology_version,
            "llm-memory-v1-metal-decode-contiguous");
  EXPECT_EQ(draft.candidate.component_identities.backend_executor_version,
            LlmMetalDecodeContiguousVersion::EXECUTOR);
  EXPECT_EQ(draft.candidate.component_identities.schedule_version,
            LlmMetalDecodeContiguousVersion::SCHEDULE);
  EXPECT_EQ(draft.candidate.component_identities.buffer_pattern_version,
            LlmMetalDecodeContiguousVersion::BUFFER_PATTERN);
  EXPECT_EQ(draft.candidate.component_identities.write_pattern_version,
            LlmMetalDecodeContiguousVersion::WRITE_PATTERN);
  EXPECT_EQ(draft.candidate.component_identities.checksum_pattern_version,
            LlmMetalDecodeContiguousVersion::CHECKSUM);
  EXPECT_FALSE(draft.candidate.component_identities.msl_revision.has_value());
  const LlmMetalExecutionPlan* unresolved =
      get_llm_metal_execution_plan(draft.candidate);
  ASSERT_NE(unresolved, nullptr);
  EXPECT_FALSE(unresolved->valid);

  LlmMemoryWorkPlan plan = finalize_llm_memory_work_plan(
      std::move(draft), 19, 23);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  unresolved = get_llm_metal_execution_plan(plan);
  ASSERT_NE(unresolved, nullptr);
  EXPECT_FALSE(unresolved->valid);
  EXPECT_EQ(plan.memory_budget.request.checksum_auxiliary_bytes, 19u);
  EXPECT_EQ(plan.memory_budget.request.orchestration_auxiliary_bytes, 23u);
  const LlmScenarioWorkPlan scenario = build_llm_scenario_work_plan(
      plan, LlmScenario::Mixed, 2, true);
  ASSERT_TRUE(scenario.valid) << scenario.reason_code;
  EXPECT_EQ(scenario.work_unit_kind, LlmWorkUnitKind::DecodeStep);
}

TEST(LlmMemoryWorkPlanTest,
     MetalScenarioPlansEnforceDispatchWorkUnitCapBeforeExecution) {
  LlmMemoryWorkPlanDraft draft =
      prepare_llm_memory_work_plan(metal_work_plan_request());
  ASSERT_TRUE(draft.valid) << draft.reason_code;
  LlmMemoryWorkPlan plan =
      finalize_llm_memory_work_plan(std::move(draft), 0, 0);
  ASSERT_TRUE(plan.valid) << plan.reason_code;

  const LlmScenarioWorkPlan exact = build_llm_scenario_work_plan(
      plan, LlmScenario::WeightsOnly,
      Constants::LLM_METAL_MAX_WORK_UNITS_PER_DISPATCH, true);
  ASSERT_TRUE(exact.valid) << exact.reason_code;
  EXPECT_EQ(exact.maximum_work_units_by_work_unit_cap,
            Constants::LLM_METAL_MAX_WORK_UNITS_PER_DISPATCH);
  EXPECT_EQ(exact.effective_maximum_work_units,
            Constants::LLM_METAL_MAX_WORK_UNITS_PER_DISPATCH);

  const LlmScenarioWorkPlan above = build_llm_scenario_work_plan(
      plan, LlmScenario::WeightsOnly,
      Constants::LLM_METAL_MAX_WORK_UNITS_PER_DISPATCH + 1, true);
  EXPECT_FALSE(above.valid);
  EXPECT_EQ(above.reason_code, LlmWorkPlanReason::WORK_UNIT_CAP_EXCEEDED);
}

TEST(LlmMemoryWorkPlanTest,
     MetalRuntimePlanFinalizationBindsExactResourcesAndAuxiliaryBytes) {
  LlmMemoryWorkPlanDraft draft =
      prepare_llm_memory_work_plan(metal_work_plan_request());
  ASSERT_TRUE(draft.valid) << draft.reason_code;
  const std::string unresolved_identity = draft.candidate.plan_identity;

  LlmMetalExecutionPlan provisional = build_llm_metal_execution_plan(
      metal_resource_request(draft.candidate));
  ASSERT_TRUE(provisional.valid) << provisional.reason_code;
  const size_t weight_layer_backing_bytes =
      draft.candidate.weight_layers.capacity() * sizeof(LlmByteRange);
  const size_t provisional_execution_backing_bytes =
      metal_execution_plan_backing_bytes(provisional);
  const size_t admitted_execution_backing_bytes =
      provisional_execution_backing_bytes +
      LlmMetalPlannerAccounting::
          RUNTIME_IDENTITY_GROWTH_RESERVE_BYTES;
  const size_t expected_two_pass_planner_peak_bytes =
      weight_layer_backing_bytes +
      2 * admitted_execution_backing_bytes;
  const size_t expected_planned_resource_count =
      provisional.resources.planned_resources.size();
  const size_t expected_persistent_resource_count =
      static_cast<size_t>(std::count_if(
          provisional.resources.planned_resources.begin(),
          provisional.resources.planned_resources.end(),
          [](const LlmMetalPlannedResource& resource) {
            return resource.persistent;
          }));
  ASSERT_TRUE(attach_llm_metal_execution_plan(
      draft, std::move(provisional)));
  EXPECT_NE(draft.candidate.plan_identity, unresolved_identity);
  ASSERT_TRUE(draft.auxiliary_preflight.valid);
  EXPECT_EQ(draft.candidate.memory_budget.request.planner_storage_bytes,
            expected_two_pass_planner_peak_bytes);
  EXPECT_EQ(draft.auxiliary_preflight.metal_planned_resource_count,
            expected_planned_resource_count);
  EXPECT_EQ(draft.auxiliary_preflight.metal_persistent_resource_count,
            expected_persistent_resource_count);
  EXPECT_EQ(
      draft.auxiliary_preflight
          .metal_resolved_execution_plan_backing_bytes,
      admitted_execution_backing_bytes);
  const size_t admitted_resolved_plan_identity_backing_bytes =
      2 * (draft.auxiliary_preflight.model_plan_identity_bytes + 1);
  EXPECT_EQ(
      draft.auxiliary_preflight
          .metal_resolved_plan_identity_backing_bytes,
      admitted_resolved_plan_identity_backing_bytes);

  constexpr size_t kChecksumAuxiliaryBytes = 111;
  constexpr size_t kOrchestrationAuxiliaryBytes = 222;
  constexpr size_t kCommandAuxiliaryBytes =
      kChecksumAuxiliaryBytes + kOrchestrationAuxiliaryBytes;
  LlmMetalExecutionPlan exact = build_llm_metal_execution_plan(
      metal_resource_request(draft.candidate, kCommandAuxiliaryBytes));
  ASSERT_TRUE(exact.valid) << exact.reason_code;
  EXPECT_LE(metal_execution_plan_backing_bytes(exact),
            admitted_execution_backing_bytes);
  const std::string execution_identity = exact.identity;
  LlmMemoryWorkPlan plan = finalize_llm_memory_work_plan(
      std::move(draft), std::move(exact), kChecksumAuxiliaryBytes,
      kOrchestrationAuxiliaryBytes);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  const LlmMetalExecutionPlan* const metal =
      get_llm_metal_execution_plan(plan);
  ASSERT_NE(metal, nullptr);
  ASSERT_TRUE(metal->valid) << metal->reason_code;
  EXPECT_EQ(metal->identity, execution_identity);
  EXPECT_EQ(metal->resources.additional_owned_bytes,
            plan.memory_budget.request.planner_storage_bytes +
                kCommandAuxiliaryBytes);
  EXPECT_EQ(plan.memory_budget.request.checksum_auxiliary_bytes,
            kChecksumAuxiliaryBytes);
  EXPECT_EQ(plan.memory_budget.request.orchestration_auxiliary_bytes,
            kOrchestrationAuxiliaryBytes);
  EXPECT_EQ(plan.memory_budget.request.auxiliary_bytes,
            plan.memory_budget.request.planner_storage_bytes +
                kCommandAuxiliaryBytes);
  EXPECT_EQ(plan.memory_budget.request.required_total_bytes,
            metal->resources.known_owned_peak_bytes);
  size_t resolved_execution_plan_backing_bytes = 0;
  size_t resolved_plan_identity_backing_bytes = 0;
  ASSERT_TRUE(calculate_llm_metal_resolved_plan_backing_bytes(
      plan, plan.plan_identity.size(),
      resolved_execution_plan_backing_bytes,
      resolved_plan_identity_backing_bytes));
  EXPECT_EQ(resolved_execution_plan_backing_bytes,
            metal_execution_plan_backing_bytes(*metal));
  EXPECT_LE(resolved_execution_plan_backing_bytes,
            admitted_execution_backing_bytes);
  EXPECT_EQ(resolved_plan_identity_backing_bytes,
            2 * (plan.plan_identity.size() + 1));
  EXPECT_LE(resolved_plan_identity_backing_bytes,
            admitted_resolved_plan_identity_backing_bytes);
  ASSERT_TRUE(plan.component_identities.msl_revision.has_value());
  ASSERT_TRUE(plan.component_identities.msl_source_sha256.has_value());
  EXPECT_EQ(*plan.component_identities.msl_revision,
            metal->msl_revision);
  EXPECT_EQ(*plan.component_identities.msl_source_sha256,
            metal->msl_source_sha256);

  const std::string plan_identity = plan.plan_identity;
  EXPECT_TRUE(readmit_llm_memory_work_plan(
      plan, kChecksumAuxiliaryBytes, kOrchestrationAuxiliaryBytes));
  EXPECT_EQ(plan.plan_identity, plan_identity);
  EXPECT_FALSE(readmit_llm_memory_work_plan(
      plan, kChecksumAuxiliaryBytes + 1,
      kOrchestrationAuxiliaryBytes));
  EXPECT_TRUE(plan.valid);
}

TEST(LlmMemoryWorkPlanTest,
     MetalPlanningRejectsInactiveProfilesAndMismatchedExactAuxiliary) {
  LlmMemoryWorkPlanRequest workers = metal_work_plan_request();
  workers.requested_workers = 1;
  expect_invalid_plan(build_llm_memory_work_plan(workers),
                      LlmWorkPlanReason::METAL_WORKERS_NOT_APPLICABLE);

  LlmMemoryWorkPlanRequest prefill = metal_work_plan_request();
  prefill.geometry.phase = LlmPhase::Prefill;
  prefill.geometry.visible_context_tokens = 0;
  prefill.geometry.prompt_tokens = 3;
  prefill.geometry.attention_query_tile_tokens = 2;
  expect_invalid_plan(build_llm_memory_work_plan(prefill),
                      LlmWorkPlanReason::PHASE_NOT_ACTIVATED);

  LlmMemoryWorkPlanRequest paged = metal_work_plan_request();
  paged.geometry = paged_geometry_request();
  expect_invalid_plan(build_llm_memory_work_plan(paged),
                      LlmWorkPlanReason::KV_LAYOUT_NOT_ACTIVATED);

  LlmMemoryWorkPlanDraft draft =
      prepare_llm_memory_work_plan(metal_work_plan_request());
  ASSERT_TRUE(draft.valid) << draft.reason_code;
  ASSERT_TRUE(attach_llm_metal_execution_plan(
      draft, build_llm_metal_execution_plan(
                 metal_resource_request(draft.candidate))));
  LlmMetalExecutionPlan mismatched = build_llm_metal_execution_plan(
      metal_resource_request(draft.candidate, 44));
  ASSERT_TRUE(mismatched.valid) << mismatched.reason_code;
  LlmMemoryWorkPlan invalid = finalize_llm_memory_work_plan(
      std::move(draft), std::move(mismatched), 21, 22);
  ASSERT_TRUE(invalid.valid) << invalid.reason_code;
  const LlmMetalExecutionPlan* const rejected_runtime =
      get_llm_metal_execution_plan(invalid);
  ASSERT_NE(rejected_runtime, nullptr);
  EXPECT_FALSE(rejected_runtime->valid);
  EXPECT_EQ(rejected_runtime->reason_code,
            LlmWorkPlanReason::INVALID_MODEL_WORK_PLAN);
  EXPECT_EQ(invalid.reason_code, LlmWorkPlanReason::VALID);
}

TEST(LlmMemoryWorkPlanTest,
     MetalRuntimePlanningFailureRetainsLogicalTerminalPlan) {
  LlmMemoryWorkPlanDraft draft =
      prepare_llm_memory_work_plan(metal_work_plan_request());
  ASSERT_TRUE(draft.valid) << draft.reason_code;
  LlmMetalResourcePlanRequest invalid_request =
      metal_resource_request(draft.candidate);
  invalid_request.argument_buffer_encoded_length = 0;
  LlmMetalExecutionPlan provisional =
      build_llm_metal_execution_plan(invalid_request);
  ASSERT_FALSE(provisional.valid);
  ASSERT_EQ(provisional.reason_code,
            LlmMetalPlanReason::ARGUMENT_ENCODER_LENGTH_ZERO);
  ASSERT_TRUE(attach_llm_metal_execution_plan(
      draft, std::move(provisional)));
  ASSERT_TRUE(draft.valid);
  const LlmMetalExecutionPlan* retained =
      get_llm_metal_execution_plan(draft.candidate);
  ASSERT_NE(retained, nullptr);
  EXPECT_FALSE(retained->valid);
  EXPECT_EQ(retained->reason_code,
            LlmMetalPlanReason::ARGUMENT_ENCODER_LENGTH_ZERO);

  constexpr size_t kChecksumAuxiliaryBytes = 17;
  constexpr size_t kOrchestrationAuxiliaryBytes = 29;
  invalid_request.additional_owned_bytes =
      draft.candidate.memory_budget.request.planner_storage_bytes +
      kChecksumAuxiliaryBytes + kOrchestrationAuxiliaryBytes;
  LlmMemoryWorkPlan plan = finalize_llm_memory_work_plan(
      std::move(draft),
      build_llm_metal_execution_plan(invalid_request),
      kChecksumAuxiliaryBytes, kOrchestrationAuxiliaryBytes);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  retained = get_llm_metal_execution_plan(plan);
  ASSERT_NE(retained, nullptr);
  EXPECT_FALSE(retained->valid);
  EXPECT_EQ(retained->reason_code,
            LlmMetalPlanReason::ARGUMENT_ENCODER_LENGTH_ZERO);
  EXPECT_EQ(plan.reason_code, LlmWorkPlanReason::VALID);
  EXPECT_EQ(plan.memory_budget.request.checksum_auxiliary_bytes,
            kChecksumAuxiliaryBytes);
  EXPECT_EQ(plan.memory_budget.request.orchestration_auxiliary_bytes,
            kOrchestrationAuxiliaryBytes);
}

TEST(LlmMemoryWorkPlanTest,
     MoveOperationsPreserveDestinationAndInvalidateSource) {
  const LlmMemoryWorkPlan constructor_expected = build_llm_memory_work_plan(
      work_plan_request(small_geometry_request(), 2, 2));
  ASSERT_TRUE(constructor_expected.valid) << constructor_expected.reason_code;
  LlmMemoryWorkPlan constructor_source = build_llm_memory_work_plan(
      work_plan_request(small_geometry_request(), 2, 2));
  ASSERT_TRUE(constructor_source.valid) << constructor_source.reason_code;

  LlmMemoryWorkPlan constructed(std::move(constructor_source));
  ASSERT_TRUE(constructed.valid) << constructed.reason_code;
  expect_equivalent_executable_plans(constructed, constructor_expected);
  const LlmScenarioWorkPlan constructed_scenario =
      build_llm_scenario_work_plan(constructed, LlmScenario::Mixed, 2,
                                   true);
  ASSERT_TRUE(constructed_scenario.valid) << constructed_scenario.reason_code;
  EXPECT_EQ(constructed_scenario.model_plan_identity,
            constructor_expected.plan_identity);
  EXPECT_EQ(constructed_scenario.scenario_seed,
            constructor_expected.scenario_seeds[2]);
  EXPECT_FALSE(constructor_source.valid);
  EXPECT_FALSE(constructor_source.geometry.valid);
  EXPECT_FALSE(constructor_source.memory_budget.valid);
  EXPECT_EQ(cpu_execution_plan(constructor_source).effective_workers, 0u);
  EXPECT_TRUE(constructor_source.weight_layers.empty());
  EXPECT_TRUE(cpu_execution_plan(constructor_source).workers.empty());
  EXPECT_TRUE(constructor_source.plan_identity.empty());
  EXPECT_EQ(build_llm_scenario_work_plan(
                constructor_source, LlmScenario::Mixed, 1, true)
                .reason_code,
            LlmWorkPlanReason::INVALID_MODEL_WORK_PLAN);
  EXPECT_EQ(freeze_llm_scenario_work_plans(
                constructor_source, {1, 1, 1}, true)
                .reason_code,
            LlmWorkPlanReason::INVALID_MODEL_WORK_PLAN);

  const LlmMemoryWorkPlan assignment_expected = build_llm_memory_work_plan(
      work_plan_request(small_geometry_request(2), 2, 2));
  ASSERT_TRUE(assignment_expected.valid) << assignment_expected.reason_code;
  LlmMemoryWorkPlan assignment_source = build_llm_memory_work_plan(
      work_plan_request(small_geometry_request(2), 2, 2));
  ASSERT_TRUE(assignment_source.valid) << assignment_source.reason_code;
  LlmMemoryWorkPlan assigned;
  assigned = std::move(assignment_source);
  ASSERT_TRUE(assigned.valid) << assigned.reason_code;
  expect_equivalent_executable_plans(assigned, assignment_expected);
  const LlmScenarioWorkPlan assigned_scenario =
      build_llm_scenario_work_plan(assigned, LlmScenario::KvOnly, 2,
                                   false);
  ASSERT_TRUE(assigned_scenario.valid) << assigned_scenario.reason_code;
  EXPECT_EQ(assigned_scenario.model_plan_identity,
            assignment_expected.plan_identity);
  EXPECT_EQ(assigned_scenario.scenario_seed,
            assignment_expected.scenario_seeds[1]);
  EXPECT_FALSE(assignment_source.valid);
  EXPECT_FALSE(assignment_source.geometry.valid);
  EXPECT_FALSE(assignment_source.memory_budget.valid);
  EXPECT_EQ(cpu_execution_plan(assignment_source).effective_workers, 0u);
  EXPECT_TRUE(assignment_source.weight_layers.empty());
  EXPECT_TRUE(cpu_execution_plan(assignment_source).workers.empty());
  EXPECT_TRUE(assignment_source.plan_identity.empty());

}

TEST(LlmMemoryWorkPlanTest, ScenarioOrderUsesOverflowSafeCyclicRotations) {
  EXPECT_EQ(build_llm_scenario_order(0),
            (std::array<LlmScenario, 3>{LlmScenario::WeightsOnly,
                                        LlmScenario::KvOnly,
                                        LlmScenario::Mixed}));
  EXPECT_EQ(build_llm_scenario_order(1),
            (std::array<LlmScenario, 3>{LlmScenario::KvOnly,
                                        LlmScenario::Mixed,
                                        LlmScenario::WeightsOnly}));
  EXPECT_EQ(build_llm_scenario_order(2),
            (std::array<LlmScenario, 3>{LlmScenario::Mixed,
                                        LlmScenario::WeightsOnly,
                                        LlmScenario::KvOnly}));
  EXPECT_EQ(build_llm_scenario_order(3), build_llm_scenario_order(0));
  EXPECT_EQ(build_llm_scenario_order(std::numeric_limits<size_t>::max()),
            build_llm_scenario_order(
                std::numeric_limits<size_t>::max() % 3));
}

TEST(LlmMemoryWorkPlanTest,
     CalibrationIsScenarioSpecificAndUsesSharedNumericPolicy) {
  const LlmGeometry geometry =
      resolve_llm_geometry(small_geometry_request());
  ASSERT_TRUE(geometry.valid);
  const LlmScenarioLimits weights =
      calculate_llm_scenario_limits(geometry, LlmScenario::WeightsOnly);
  const LlmScenarioLimits kv =
      calculate_llm_scenario_limits(geometry, LlmScenario::KvOnly);
  const LlmScenarioLimits mixed =
      calculate_llm_scenario_limits(geometry, LlmScenario::Mixed);
  ASSERT_TRUE(weights.valid);
  ASSERT_TRUE(kv.valid);
  ASSERT_TRUE(mixed.valid);
  EXPECT_EQ(calculate_llm_pilot_work_units(weights), 8192u);
  EXPECT_EQ(calculate_llm_pilot_work_units(kv), 16384u);
  EXPECT_EQ(calculate_llm_pilot_work_units(mixed), 5462u);
  EXPECT_EQ(calculate_llm_calibrated_work_units(0.010, 100, weights), 1500u);
  EXPECT_EQ(calculate_llm_calibrated_work_units(0.010, 100, kv), 1500u);
  EXPECT_EQ(calculate_llm_calibrated_work_units(0.010, 100, mixed), 1500u);
  EXPECT_EQ(calculate_llm_calibrated_work_units(0.0, 100, weights), 0u);
  EXPECT_EQ(calculate_llm_calibrated_work_units(
                std::numeric_limits<double>::quiet_NaN(), 100, weights),
            0u);
  EXPECT_EQ(calculate_llm_calibrated_work_units(
                std::numeric_limits<double>::infinity(), 100, weights),
            0u);
  const LlmScenarioLimits invalid = calculate_llm_scenario_limits(
      geometry, static_cast<LlmScenario>(99));
  EXPECT_FALSE(invalid.valid);
  EXPECT_EQ(invalid.reason_code, LlmWorkPlanReason::INVALID_SCENARIO);
  EXPECT_EQ(calculate_llm_pilot_work_units(invalid), 0u);
  const LlmMemoryWorkPlan model_plan = build_llm_memory_work_plan(
      work_plan_request(small_geometry_request(), 2, 2));
  ASSERT_TRUE(model_plan.valid) << model_plan.reason_code;
  const LlmScenarioWorkPlan invalid_scenario_plan =
      build_llm_scenario_work_plan(model_plan,
                                   static_cast<LlmScenario>(99), 1, false);
  EXPECT_FALSE(invalid_scenario_plan.valid);
  EXPECT_EQ(invalid_scenario_plan.reason_code,
            LlmWorkPlanReason::INVALID_SCENARIO);
  EXPECT_EQ(invalid_scenario_plan.scenario_seed, 0u);
  EXPECT_TRUE(invalid_scenario_plan.plan_identity.empty());

  const LlmGeometry payload_limited_geometry =
      resolve_llm_geometry({kGiB, 1, 1, 1, 1, 1, 1, 1});
  ASSERT_TRUE(payload_limited_geometry.valid);
  const LlmScenarioLimits limited_weights = calculate_llm_scenario_limits(
      payload_limited_geometry, LlmScenario::WeightsOnly);
  const LlmScenarioLimits limited_kv = calculate_llm_scenario_limits(
      payload_limited_geometry, LlmScenario::KvOnly);
  const LlmScenarioLimits limited_mixed = calculate_llm_scenario_limits(
      payload_limited_geometry, LlmScenario::Mixed);
  ASSERT_TRUE(limited_weights.valid);
  ASSERT_TRUE(limited_kv.valid);
  ASSERT_TRUE(limited_mixed.valid);
  EXPECT_EQ(limited_weights.effective_maximum_work_units, 64u);
  EXPECT_EQ(limited_kv.effective_maximum_work_units,
            Constants::LLM_MAX_WORK_UNITS_PER_MEASUREMENT);
  EXPECT_EQ(limited_mixed.effective_maximum_work_units, 63u);
  EXPECT_EQ(calculate_llm_calibrated_work_units(0.001, 100, limited_weights),
            64u);
  EXPECT_EQ(calculate_llm_calibrated_work_units(0.001, 100, limited_kv),
            15000u);
  EXPECT_EQ(calculate_llm_calibrated_work_units(0.001, 100, limited_mixed),
            63u);
  EXPECT_EQ(classify_llm_duration_quality(
                0.050, limited_weights.effective_maximum_work_units,
                limited_weights),
            "guardrail-limited-below-target");
}

TEST(LlmMemoryWorkPlanTest,
     DurationQualityUsesInclusiveWindowAndSingleWorkUnitException) {
  const LlmGeometry geometry =
      resolve_llm_geometry(small_geometry_request());
  const LlmScenarioLimits limits =
      calculate_llm_scenario_limits(geometry, LlmScenario::Mixed);
  ASSERT_TRUE(limits.valid);
  EXPECT_TRUE(llm_duration_in_target_window(0.100));
  EXPECT_TRUE(llm_duration_in_target_window(0.250));
  EXPECT_FALSE(llm_duration_in_target_window(0.099999));
  EXPECT_FALSE(llm_duration_in_target_window(0.250001));
  EXPECT_FALSE(llm_duration_in_target_window(
      std::numeric_limits<double>::quiet_NaN()));
  EXPECT_EQ(classify_llm_duration_quality(0.150, 1, limits),
            "within-target-window");
  EXPECT_EQ(classify_llm_duration_quality(0.300, 1, limits),
            "above-target-single-work-unit");
  EXPECT_EQ(classify_llm_duration_quality(0.050, 2, limits),
            "below-target-window");
  EXPECT_EQ(classify_llm_duration_quality(0.300, 2, limits),
            "above-target-window");
  EXPECT_EQ(classify_llm_duration_quality(0.0, 1, limits),
            "invalid-duration");
}

TEST(LlmMemoryWorkPlanTest, ExplicitWorkUnitsStayExactAndRespectBothCaps) {
  LlmGeometryRequest request = {1, 1, 1, 1, 1, 1, 1, 1};
  LlmGeometry geometry = resolve_llm_geometry(request);
  ASSERT_TRUE(geometry.valid);
  const LlmScenarioLimits tiny =
      calculate_llm_scenario_limits(geometry, LlmScenario::WeightsOnly);
  ASSERT_TRUE(tiny.valid);
  LlmMemoryWorkPlan model_plan = build_llm_memory_work_plan(
      work_plan_request(request, 1, 1));
  ASSERT_TRUE(model_plan.valid) << model_plan.reason_code;
  EXPECT_EQ(tiny.effective_maximum_work_units,
            Constants::LLM_MAX_WORK_UNITS_PER_MEASUREMENT);
  EXPECT_TRUE(build_llm_scenario_work_plan(
                  model_plan, LlmScenario::WeightsOnly,
                  tiny.effective_maximum_work_units, true)
                  .valid);
  EXPECT_EQ(build_llm_scenario_work_plan(
                model_plan, LlmScenario::WeightsOnly,
                tiny.effective_maximum_work_units + 1, true)
                .reason_code,
            LlmWorkPlanReason::WORK_UNIT_CAP_EXCEEDED);

  request.active_weight_bytes = kGiB;
  geometry = resolve_llm_geometry(request);
  ASSERT_TRUE(geometry.valid);
  model_plan = build_llm_memory_work_plan(work_plan_request(request, 1, 1));
  ASSERT_TRUE(model_plan.valid) << model_plan.reason_code;
  const LlmScenarioLimits payload_limited =
      calculate_llm_scenario_limits(geometry, LlmScenario::WeightsOnly);
  ASSERT_TRUE(payload_limited.valid);
  EXPECT_EQ(payload_limited.effective_maximum_work_units, 64u);
  const LlmScenarioWorkPlan boundary = build_llm_scenario_work_plan(
      model_plan, LlmScenario::WeightsOnly, 64, true);
  ASSERT_TRUE(boundary.valid);
  EXPECT_EQ(boundary.work_units, 64u);
  EXPECT_EQ(boundary.effective_model_payload_bytes,
            Constants::LLM_MAX_ACCOUNTED_BYTES_PER_TASK);
  EXPECT_EQ(build_llm_scenario_work_plan(
                model_plan, LlmScenario::WeightsOnly, 65, true)
                .reason_code,
            LlmWorkPlanReason::TASK_ACCOUNTED_BYTES_CAP_EXCEEDED);

  const LlmMemoryWorkPlan frozen_model_plan = build_llm_memory_work_plan(
      work_plan_request(small_geometry_request(), 2, 2));
  ASSERT_TRUE(frozen_model_plan.valid) << frozen_model_plan.reason_code;
  const LlmFrozenScenarioPlans frozen = freeze_llm_scenario_work_plans(
      frozen_model_plan, {4, 4, 4}, true);
  ASSERT_TRUE(frozen.valid) << frozen.reason_code;
  EXPECT_TRUE(frozen.explicit_iterations);
  for (const LlmScenarioWorkPlan& scenario : frozen.scenarios) {
    EXPECT_EQ(scenario.work_units, 4u);
    EXPECT_TRUE(scenario.explicit_iterations);
  }
}

TEST(LlmMemoryWorkPlanTest,
     GeometryAndPlannerOverflowsReturnReasonsWithoutExecutableTemplates) {
  const size_t maximum = std::numeric_limits<size_t>::max();
  LlmGeometryRequest geometry = {1, 1, 1, 1, maximum, 2, 1, 1};
  EXPECT_EQ(resolve_llm_geometry(geometry).reason_code,
            LlmWorkPlanReason::KV_VECTOR_BYTES_OVERFLOW);

  geometry = {1, 1, 2, 2, maximum / 2 + 1, 1, 1, 1};
  EXPECT_EQ(resolve_llm_geometry(geometry).reason_code,
            LlmWorkPlanReason::KV_RECORD_BYTES_OVERFLOW);

  geometry = {1, 1, 1, 1, maximum / 2 + 1, 1, 1, 1};
  EXPECT_EQ(resolve_llm_geometry(geometry).reason_code,
            LlmWorkPlanReason::KV_LAYER_RECORD_BYTES_OVERFLOW);

  geometry = {1, maximum, 1, 1, 1, 1, 1, 1};
  EXPECT_EQ(resolve_llm_geometry(geometry).reason_code,
            LlmWorkPlanReason::KV_BYTES_PER_TOKEN_OVERFLOW);

  geometry = {1, 1, 1, 1, 2, 1, maximum, 1};
  EXPECT_EQ(resolve_llm_geometry(geometry).reason_code,
            LlmWorkPlanReason::KV_SEQUENCE_BYTES_OVERFLOW);

  geometry = {1, maximum / 2, 1, 1, 1, 1, 1, 3};
  EXPECT_EQ(resolve_llm_geometry(geometry).reason_code,
            LlmWorkPlanReason::KV_MAPPING_BYTES_OVERFLOW);

  geometry = {1, 1, 1, 1, 1, 1, 1, maximum / 2 + 1};
  EXPECT_EQ(resolve_llm_geometry(geometry).reason_code,
            LlmWorkPlanReason::KV_CAPACITY_BYTES_OVERFLOW);

  geometry = {maximum, 1, 1, 1, 1, 1, 1, 1};
  EXPECT_EQ(resolve_llm_geometry(geometry).reason_code,
            LlmWorkPlanReason::TOTAL_DATA_BYTES_OVERFLOW);

  geometry = {1, 1, 1, 1, 1, 1, 1, maximum / 2};
  EXPECT_EQ(resolve_llm_geometry(geometry).reason_code,
            LlmWorkPlanReason::KV_ONLY_PAYLOAD_OVERFLOW);

  geometry = {maximum / 2, 1, 1, 1, 1, 1, 1, maximum / 4};
  EXPECT_EQ(resolve_llm_geometry(geometry).reason_code,
            LlmWorkPlanReason::MIXED_PAYLOAD_OVERFLOW);

  const size_t descriptor_overflow_layers = maximum / 128 + 1;
  geometry = {1, descriptor_overflow_layers, 1, 1, 1, 1, 1, 1};
  expect_invalid_plan(
      build_llm_memory_work_plan(work_plan_request(geometry, 1, 1)),
      LlmWorkPlanReason::DESCRIPTOR_BYTES_OVERFLOW);

  const size_t planner_overflow_layers = maximum / 144 + 1;
  geometry = {1, planner_overflow_layers, 1, 1, 1, 1, 1, 1};
  expect_invalid_plan(
      build_llm_memory_work_plan(work_plan_request(geometry, 1, 1)),
      LlmWorkPlanReason::PLANNER_STORAGE_BYTES_OVERFLOW);

  geometry = {maximum - 4094, 1, 1, 1, 1, 1, 1, 1};
  LlmMemoryWorkPlanRequest rounded = work_plan_request(geometry, 1, 1);
  rounded.mapping_granularity_bytes = 4096;
  rounded.available_memory_bytes = maximum;
  expect_invalid_plan(build_llm_memory_work_plan(rounded),
                      LlmWorkPlanReason::MAPPING_ROUND_UP_OVERFLOW);

  geometry = {maximum - 4, 1, 1, 1, 1, 1, 1, 1};
  const LlmGeometry valid_near_limit = resolve_llm_geometry(geometry);
  ASSERT_TRUE(valid_near_limit.valid) << valid_near_limit.reason_code;
  EXPECT_EQ(build_llm_memory_budget_request(valid_near_limit, 0, 0, 0, 0, 2)
                .reason_code,
            LlmWorkPlanReason::MEMORY_REQUIREMENT_OVERFLOW);

  LlmMemoryWorkPlanRequest invalid_workers =
      work_plan_request(small_geometry_request(), 0, 1);
  expect_invalid_plan(build_llm_memory_work_plan(invalid_workers),
                      LlmWorkPlanReason::REQUESTED_WORKERS_ZERO);
  invalid_workers = work_plan_request(small_geometry_request(), 1, 0);
  expect_invalid_plan(build_llm_memory_work_plan(invalid_workers),
                      LlmWorkPlanReason::AVAILABLE_WORKERS_ZERO);
}

TEST(LlmMemoryWorkPlanTest,
     ScenarioTotalsDetectOverflowBeforePayloadCapComparison) {
  LlmGeometryRequest request = {Constants::LLM_MAX_ACCOUNTED_BYTES_PER_TASK,
                                1, 1, 1, 1, 1, 1, 1};
  const LlmGeometry geometry = resolve_llm_geometry(request);
  ASSERT_TRUE(geometry.valid);
  LlmMemoryWorkPlanRequest model_request =
      work_plan_request(request, 1, 1);
  model_request.available_memory_bytes = std::numeric_limits<size_t>::max();
  const LlmMemoryWorkPlan overflow_model =
      build_llm_memory_work_plan(model_request);
  ASSERT_TRUE(overflow_model.valid) << overflow_model.reason_code;
  const LlmScenarioWorkPlan overflow = build_llm_scenario_work_plan(
      overflow_model, LlmScenario::WeightsOnly,
      Constants::LLM_MAX_WORK_UNITS_PER_MEASUREMENT, true);
  EXPECT_FALSE(overflow.valid);
  EXPECT_EQ(overflow.reason_code,
            LlmWorkPlanReason::TASK_ACCOUNTED_BYTES_OVERFLOW);
  EXPECT_TRUE(overflow.plan_identity.empty());

  const LlmGeometry above_payload_cap = resolve_llm_geometry(
      {Constants::LLM_MAX_ACCOUNTED_BYTES_PER_TASK + 1,
       1, 1, 1, 1, 1, 1, 1});
  ASSERT_TRUE(above_payload_cap.valid) << above_payload_cap.reason_code;
  const LlmScenarioLimits impossible_single_work_unit =
      calculate_llm_scenario_limits(above_payload_cap,
                                    LlmScenario::WeightsOnly);
  EXPECT_FALSE(impossible_single_work_unit.valid);
  EXPECT_EQ(impossible_single_work_unit.reason_code,
            LlmWorkPlanReason::GUARDRAIL_BELOW_ONE_WORK_UNIT);

  const LlmMemoryWorkPlan model_plan = build_llm_memory_work_plan(
      work_plan_request(small_geometry_request(), 2, 2));
  ASSERT_TRUE(model_plan.valid) << model_plan.reason_code;
  const LlmFrozenScenarioPlans invalid = freeze_llm_scenario_work_plans(
      model_plan, {1, 0, 1}, false);
  EXPECT_FALSE(invalid.valid);
  EXPECT_EQ(invalid.reason_code, LlmWorkPlanReason::WORK_UNIT_COUNT_ZERO);
  EXPECT_TRUE(invalid.plan_identity.empty());
  for (const LlmScenarioWorkPlan& scenario : invalid.scenarios) {
    EXPECT_FALSE(scenario.valid);
    EXPECT_TRUE(scenario.plan_identity.empty());
  }
}

TEST(LlmMemoryWorkPlanTest,
     MemoryBudgetRoundsThreeMappingsAndAccountsForAuxiliaryCategories) {
  const LlmGeometry geometry =
      resolve_llm_geometry({4097, 1, 1, 1, 1, 1, 1, 1});
  ASSERT_TRUE(geometry.valid);
  const LlmMemoryBudgetRequest request = build_llm_memory_budget_request(
      geometry, 128, 64, 7, 9, 4096);
  ASSERT_TRUE(request.valid) << request.reason_code;
  EXPECT_EQ(request.requested_weight_mapping_bytes, 4097u);
  EXPECT_EQ(request.requested_k_mapping_bytes, 1u);
  EXPECT_EQ(request.requested_v_mapping_bytes, 1u);
  EXPECT_EQ(request.requested_data_bytes, 4099u);
  EXPECT_EQ(request.committed_weight_mapping_bytes, 8192u);
  EXPECT_EQ(request.committed_k_mapping_bytes, 4096u);
  EXPECT_EQ(request.committed_v_mapping_bytes, 4096u);
  EXPECT_EQ(request.committed_data_bytes, 16384u);
  EXPECT_EQ(request.descriptor_bytes, 128u);
  EXPECT_EQ(request.planner_storage_bytes, 64u);
  EXPECT_EQ(request.checksum_auxiliary_bytes, 7u);
  EXPECT_EQ(request.orchestration_auxiliary_bytes, 9u);
  EXPECT_EQ(request.auxiliary_bytes, 208u);
  EXPECT_EQ(request.required_total_bytes, 16592u);

  const LlmMemoryBudgetRequest invalid_request =
      build_llm_memory_budget_request(geometry, 0, 0, 0, 0, 0);
  EXPECT_EQ(invalid_request.reason_code,
            LlmWorkPlanReason::MAPPING_GRANULARITY_ZERO);
  const LlmMemoryBudget invalid_budget =
      evaluate_llm_memory_budget(invalid_request, 1000);
  EXPECT_FALSE(invalid_budget.valid);
  EXPECT_EQ(invalid_budget.reason_code, invalid_request.reason_code);
  EXPECT_EQ(invalid_budget.available_memory_bytes, 1000u);
  EXPECT_EQ(invalid_budget.allowed_memory_bytes, 0u);
  EXPECT_FALSE(invalid_budget.used_fallback);
  EXPECT_EQ(build_llm_memory_budget_request(
                geometry, std::numeric_limits<size_t>::max(), 1, 0, 0, 1)
                .reason_code,
            LlmWorkPlanReason::AUXILIARY_BYTES_OVERFLOW);

  const size_t maximum = std::numeric_limits<size_t>::max();
  EXPECT_EQ(build_llm_memory_budget_request(
                geometry, 0, 0, maximum, 1, 1)
                .reason_code,
            LlmWorkPlanReason::AUXILIARY_BYTES_OVERFLOW);
  EXPECT_EQ(build_llm_memory_budget_request(
                geometry, maximum - 1, 0, 2, 0, 1)
                .reason_code,
            LlmWorkPlanReason::AUXILIARY_BYTES_OVERFLOW);

  const LlmGeometry minimal_geometry =
      resolve_llm_geometry({1, 1, 1, 1, 1, 1, 1, 1});
  ASSERT_TRUE(minimal_geometry.valid);
  EXPECT_EQ(build_llm_memory_budget_request(
                minimal_geometry, maximum - 2, 0, 0, 0, 1)
                .reason_code,
            LlmWorkPlanReason::MEMORY_REQUIREMENT_OVERFLOW);
}

TEST(LlmMemoryWorkPlanTest,
     MemoryBudgetAcceptsBoundaryRejectsExcessAndRecordsFallback) {
  LlmMemoryBudgetRequest request;
  request.valid = true;
  request.reason_code = LlmWorkPlanReason::VALID;
  request.required_total_bytes = 800;
  LlmMemoryBudget budget = evaluate_llm_memory_budget(request, 1000);
  ASSERT_TRUE(budget.valid) << budget.reason_code;
  EXPECT_EQ(budget.available_memory_bytes, 1000u);
  EXPECT_EQ(budget.allowed_memory_bytes, 800u);
  EXPECT_FALSE(budget.used_fallback);

  request.required_total_bytes = budget.allowed_memory_bytes + 1;
  budget = evaluate_llm_memory_budget(request, 1000);
  EXPECT_FALSE(budget.valid);
  EXPECT_EQ(budget.reason_code, LlmWorkPlanReason::MEMORY_BUDGET_EXCEEDED);

  request.required_total_bytes =
      static_cast<size_t>(Constants::FALLBACK_TOTAL_LIMIT_MB) *
      Constants::BYTES_PER_MB;
  budget = evaluate_llm_memory_budget(request, 0);
  ASSERT_TRUE(budget.valid) << budget.reason_code;
  EXPECT_TRUE(budget.used_fallback);
  EXPECT_EQ(budget.available_memory_bytes, 0u);
  EXPECT_EQ(budget.allowed_memory_bytes, 2 * kGiB);
  ++request.required_total_bytes;
  budget = evaluate_llm_memory_budget(request, 0);
  EXPECT_FALSE(budget.valid);
  EXPECT_EQ(budget.reason_code, LlmWorkPlanReason::MEMORY_BUDGET_EXCEEDED);
}

TEST(LlmMemoryWorkPlanTest,
     WorkPlanIdentityIsDeterministicAndSensitiveToFrozenInputs) {
  LlmMemoryWorkPlanRequest request =
      work_plan_request(small_geometry_request(), 2, 2);
  const LlmMemoryWorkPlan first = build_llm_memory_work_plan(request);
  const LlmMemoryWorkPlan second = build_llm_memory_work_plan(request);
  ASSERT_TRUE(first.valid);
  ASSERT_TRUE(second.valid);
  EXPECT_EQ(first.plan_identity, second.plan_identity);
  EXPECT_FALSE(first.plan_identity.empty());
  EXPECT_EQ(first.backend, LlmMemoryBackend::Cpu);
  EXPECT_EQ(first.phase, LlmPhase::Decode);
  EXPECT_EQ(first.kv_layout, LlmKvLayout::Contiguous);
  EXPECT_EQ(first.work_unit_kind, LlmWorkUnitKind::DecodeStep);
  EXPECT_EQ(first.weight_passes_per_work_unit, 1u);
  EXPECT_EQ(first.kv_replay_factor, 1u);
  EXPECT_EQ(first.methodology_version,
            Constants::LLM_CPU_DECODE_CONTIGUOUS_METHODOLOGY_VERSION);
  EXPECT_EQ(first.component_identities.logical_profile_version,
            Constants::LLM_LOGICAL_PROFILE_VERSION);
  EXPECT_EQ(first.component_identities.kv_layout_version,
            Constants::LLM_CONTIGUOUS_KV_LAYOUT_VERSION);
  EXPECT_FALSE(first.component_identities.permutation_version.has_value());
  EXPECT_EQ(first.component_identities.backend_executor_version,
            Constants::LLM_CPU_EXECUTOR_VERSION);
  EXPECT_EQ(first.component_identities.resource_abi_version,
            Constants::LLM_DESCRIPTOR_ABI_VERSION);
  EXPECT_EQ(first.component_identities.schedule_version,
            Constants::LLM_CPU_SCHEDULE_VERSION);
  EXPECT_EQ(first.component_identities.timer_policy_version,
            Constants::LLM_CPU_TIMER_POLICY_VERSION);
  EXPECT_EQ(first.component_identities.buffer_pattern_version,
            Constants::LLM_BUFFER_PATTERN_VERSION);
  EXPECT_EQ(first.component_identities.write_pattern_version,
            Constants::LLM_APPEND_PATTERN_VERSION);
  EXPECT_EQ(first.component_identities.checksum_pattern_version,
            Constants::LLM_READ_CHECKSUM_VERSION);
  EXPECT_FALSE(first.component_identities.msl_revision.has_value());
  EXPECT_FALSE(first.component_identities.msl_source_sha256.has_value());
  const std::string expected_component_identity =
      "llm-memory-components-v1"
      "|logical_profile_version=27:decode_steady_fixed_context"
      "|kv_layout_version=43:contiguous_layer_batch_token_head_dimension"
      "|permutation_version=null"
      "|backend_executor_version=43:llm-cpu-executor-v1-arm64-decode-contiguous"
      "|resource_abi_version=28:llm-memory-descriptor-abi-v1"
      "|schedule_version=52:worker-local-layer-order-no-per-layer-global-barrier"
      "|timer_policy_version=62:synchronized-start-to-last-worker-completion-per-scenario-task"
      "|buffer_pattern_version=21:llm-buffer-pattern-v1"
      "|write_pattern_version=25:llm-kv-append-affine64-v1"
      "|checksum_pattern_version=20:llm-read-checksum-v1"
      "|msl_revision=null"
      "|msl_source_sha256=null";
  EXPECT_EQ(serialize_llm_component_identities(first.component_identities),
            expected_component_identity);
  EXPECT_EQ(first.component_identities.identity,
            expected_component_identity);
  EXPECT_EQ(first.component_identities.identity.rfind(
                Constants::LLM_COMPONENT_IDENTITY_VERSION, 0),
            0u);
  EXPECT_EQ(first.weight_buffer_seed,
            derive_llm_domain_seed(42, LlmSeedDomain::WeightBuffer));
  EXPECT_EQ(first.k_buffer_seed,
            derive_llm_domain_seed(42, LlmSeedDomain::KBuffer));
  EXPECT_EQ(first.v_buffer_seed,
            derive_llm_domain_seed(42, LlmSeedDomain::VBuffer));
  EXPECT_EQ(first.scenario_seeds[0], derive_llm_domain_seed(
                                         42, LlmSeedDomain::WeightsOnlyScenario));
  EXPECT_EQ(first.scenario_seeds[1], derive_llm_domain_seed(
                                         42, LlmSeedDomain::KvOnlyScenario));
  EXPECT_EQ(first.scenario_seeds[2], derive_llm_domain_seed(
                                         42, LlmSeedDomain::MixedScenario));

  LlmMemoryWorkPlanRequest environment_variant = request;
  environment_variant.available_workers = 5;
  environment_variant.available_memory_bytes = 0;
  environment_variant.mapping_granularity_bytes = 4096;
  environment_variant.checksum_auxiliary_bytes = 123;
  environment_variant.orchestration_auxiliary_bytes = 456;
  const LlmMemoryWorkPlan same_execution_different_budget =
      build_llm_memory_work_plan(environment_variant);
  ASSERT_TRUE(same_execution_different_budget.valid)
      << same_execution_different_budget.reason_code;
  EXPECT_TRUE(same_execution_different_budget.memory_budget.used_fallback);
  EXPECT_NE(first.memory_budget.request.required_total_bytes,
            same_execution_different_budget.memory_budget.request
                .required_total_bytes);
  EXPECT_EQ(first.plan_identity,
            same_execution_different_budget.plan_identity);

  LlmMemoryWorkPlanRequest smaller_team_request =
      work_plan_request(small_geometry_request(), 2, 1);
  const LlmMemoryWorkPlan smaller_team =
      build_llm_memory_work_plan(smaller_team_request);
  ASSERT_TRUE(smaller_team.valid) << smaller_team.reason_code;
  EXPECT_EQ(cpu_execution_plan(first).requested_workers,
            cpu_execution_plan(smaller_team).requested_workers);
  EXPECT_EQ(cpu_execution_plan(first).effective_workers, 2u);
  EXPECT_EQ(cpu_execution_plan(smaller_team).effective_workers, 1u);
  EXPECT_NE(first.plan_identity, smaller_team.plan_identity);

  ++request.base_seed;
  const LlmMemoryWorkPlan different_seed =
      build_llm_memory_work_plan(request);
  ASSERT_TRUE(different_seed.valid);
  EXPECT_NE(first.plan_identity, different_seed.plan_identity);

  const LlmFrozenScenarioPlans frozen_a =
      freeze_llm_scenario_work_plans(first, {4, 5, 6}, false);
  const LlmFrozenScenarioPlans frozen_b =
      freeze_llm_scenario_work_plans(second, {4, 5, 6}, false);
  const LlmFrozenScenarioPlans frozen_c =
      freeze_llm_scenario_work_plans(first, {4, 5, 7}, false);
  const LlmFrozenScenarioPlans frozen_d =
      freeze_llm_scenario_work_plans(different_seed, {4, 5, 6}, false);
  const LlmFrozenScenarioPlans frozen_explicit =
      freeze_llm_scenario_work_plans(first, {4, 5, 6}, true);
  const LlmFrozenScenarioPlans frozen_smaller_team =
      freeze_llm_scenario_work_plans(smaller_team, {4, 5, 6}, false);
  ASSERT_TRUE(frozen_a.valid);
  ASSERT_TRUE(frozen_b.valid);
  ASSERT_TRUE(frozen_c.valid);
  ASSERT_TRUE(frozen_d.valid);
  ASSERT_TRUE(frozen_explicit.valid);
  ASSERT_TRUE(frozen_smaller_team.valid);
  EXPECT_EQ(frozen_a.model_plan_identity, first.plan_identity);
  EXPECT_EQ(frozen_a.scenarios[0].scenario_seed,
            first.scenario_seeds[0]);
  EXPECT_EQ(frozen_a.scenarios[1].scenario_seed,
            first.scenario_seeds[1]);
  EXPECT_EQ(frozen_a.scenarios[2].scenario_seed,
            first.scenario_seeds[2]);
  EXPECT_EQ(frozen_a.plan_identity, frozen_b.plan_identity);
  EXPECT_NE(frozen_a.plan_identity, frozen_c.plan_identity);
  EXPECT_NE(frozen_a.plan_identity, frozen_d.plan_identity);
  EXPECT_NE(frozen_a.plan_identity, frozen_smaller_team.plan_identity);
  EXPECT_FALSE(frozen_a.explicit_iterations);
  EXPECT_TRUE(frozen_explicit.explicit_iterations);
  EXPECT_NE(frozen_a.scenarios[0].plan_identity,
            frozen_explicit.scenarios[0].plan_identity);
  EXPECT_NE(frozen_a.plan_identity, frozen_explicit.plan_identity);

  const LlmScenarioWorkPlan automatic_scenario =
      build_llm_scenario_work_plan(first, LlmScenario::Mixed, 6, false);
  const LlmScenarioWorkPlan explicit_scenario =
      build_llm_scenario_work_plan(first, LlmScenario::Mixed, 6, true);
  ASSERT_TRUE(automatic_scenario.valid);
  ASSERT_TRUE(explicit_scenario.valid);
  EXPECT_NE(automatic_scenario.plan_identity,
            explicit_scenario.plan_identity);

  LlmMemoryWorkPlan invalid_model;
  const LlmFrozenScenarioPlans invalid_frozen =
      freeze_llm_scenario_work_plans(invalid_model, {4, 5, 6}, false);
  EXPECT_FALSE(invalid_frozen.valid);
  EXPECT_EQ(invalid_frozen.reason_code,
            LlmWorkPlanReason::INVALID_MODEL_WORK_PLAN);
}

TEST(LlmMemoryWorkPlanTest,
     ScenarioAndFrozenIdentitiesSeparateEqualAggregatePayloads) {
  const LlmGeometryRequest geometry_a = {1024, 2, 4, 2, 8, 2, 3, 1};
  const LlmGeometryRequest geometry_b = {1024, 2, 4, 2, 16, 2, 1, 1};
  const LlmMemoryWorkPlan model_a = build_llm_memory_work_plan(
      work_plan_request(geometry_a, 2, 2));
  const LlmMemoryWorkPlan model_b = build_llm_memory_work_plan(
      work_plan_request(geometry_b, 2, 2));
  ASSERT_TRUE(model_a.valid) << model_a.reason_code;
  ASSERT_TRUE(model_b.valid) << model_b.reason_code;

  const LlmScenarioWorkPlan scenario_a = build_llm_scenario_work_plan(
      model_a, LlmScenario::KvOnly, 5, false);
  const LlmScenarioWorkPlan scenario_b = build_llm_scenario_work_plan(
      model_b, LlmScenario::KvOnly, 5, false);
  ASSERT_TRUE(scenario_a.valid);
  ASSERT_TRUE(scenario_b.valid);
  EXPECT_EQ(scenario_a.effective_model_payload_bytes,
            scenario_b.effective_model_payload_bytes);
  EXPECT_NE(scenario_a.kv_read_bytes, scenario_b.kv_read_bytes);
  EXPECT_NE(scenario_a.kv_write_bytes,
            scenario_b.kv_write_bytes);
  EXPECT_NE(scenario_a.plan_identity, scenario_b.plan_identity);
  EXPECT_EQ(scenario_a.model_plan_identity, model_a.plan_identity);

  const LlmFrozenScenarioPlans frozen_a =
      freeze_llm_scenario_work_plans(model_a, {5, 5, 5}, false);
  const LlmFrozenScenarioPlans frozen_b =
      freeze_llm_scenario_work_plans(model_b, {5, 5, 5}, false);
  ASSERT_TRUE(frozen_a.valid);
  ASSERT_TRUE(frozen_b.valid);
  EXPECT_NE(frozen_a.plan_identity, frozen_b.plan_identity);
}

TEST(LlmMemoryWorkPlanTest,
     PagedActiveModelPlanMaterializesOneReadOnlyTableAndExactBudget) {
  const LlmMemoryWorkPlan plan = build_llm_memory_work_plan(
      work_plan_request(paged_geometry_request(), 5, 5));
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  EXPECT_EQ(plan.reason_code, LlmWorkPlanReason::VALID);
  EXPECT_EQ(plan.backend, LlmMemoryBackend::Cpu);
  EXPECT_EQ(plan.phase, LlmPhase::Decode);
  EXPECT_EQ(plan.kv_layout, LlmKvLayout::Paged);
  EXPECT_EQ(plan.methodology_version, "llm-memory-v1-cpu-decode-paged");

  const LlmGeometry& geometry = plan.geometry;
  ASSERT_TRUE(geometry.valid) << geometry.reason_code;
  ASSERT_TRUE(geometry.decode.has_value());
  EXPECT_EQ(geometry.decode->visible_context_tokens, 35u);
  EXPECT_EQ(geometry.kv_block_tokens, 16u);
  EXPECT_EQ(geometry.kv_blocks_per_sequence, 3u);
  EXPECT_EQ(geometry.physical_blocks_per_layer, 6u);
  EXPECT_EQ(geometry.total_physical_blocks, 12u);
  EXPECT_EQ(geometry.kv_block_bytes, 512u);
  EXPECT_EQ(geometry.last_block_tokens, 3u);
  EXPECT_EQ(geometry.last_block_valid_bytes, 96u);
  EXPECT_EQ(geometry.decode_append_offset_in_last_block, 64u);
  EXPECT_EQ(geometry.k_logical_bytes, 4480u);
  EXPECT_EQ(geometry.v_logical_bytes, 4480u);
  EXPECT_EQ(geometry.k_mapping_bytes, 6144u);
  EXPECT_EQ(geometry.v_mapping_bytes, 6144u);
  EXPECT_EQ(geometry.k_layout_padding_bytes, 1664u);
  EXPECT_EQ(geometry.v_layout_padding_bytes, 1664u);
  EXPECT_EQ(geometry.block_table_entries, 6u);
  EXPECT_EQ(geometry.block_table_bytes, 24u);
  EXPECT_EQ(geometry.kv_capacity_bytes, 12288u);
  EXPECT_EQ(geometry.total_data_mapping_bytes, 13312u);

  const LlmCpuExecutionPlan& cpu_plan = cpu_execution_plan(plan);
  EXPECT_EQ(cpu_plan.requested_workers, 5u);
  EXPECT_EQ(cpu_plan.available_workers, 5u);
  EXPECT_EQ(cpu_plan.effective_workers, 5u);
  EXPECT_EQ(cpu_plan.layer_descriptors_per_worker, 2u);
  EXPECT_EQ(cpu_plan.sequence_descriptors_per_worker, 4u);
  EXPECT_EQ(cpu_plan.total_layer_descriptors, 10u);
  EXPECT_EQ(cpu_plan.total_sequence_descriptors, 20u);
  EXPECT_EQ(cpu_plan.descriptor_bytes, 2400u);
  ASSERT_EQ(cpu_plan.workers.size(), 5u);
  for (const LlmWorkerWorkPlan& worker : cpu_plan.workers) {
    EXPECT_EQ(worker.layers.size(), 2u);
    EXPECT_TRUE(worker.sequences.empty());
    EXPECT_EQ(worker.paged_assignments.size(), 4u);
  }

  ASSERT_TRUE(cpu_plan.paged.has_value());
  const LlmPagedCpuExecutionPlan& paged = *cpu_plan.paged;
  EXPECT_EQ(paged.layout.blocks_per_sequence, 3u);
  EXPECT_EQ(paged.layout.physical_blocks_per_layer, 6u);
  EXPECT_EQ(paged.layout.block_table_entries, 6u);
  EXPECT_NE(paged.block_table_mapping.get(), nullptr);
  EXPECT_EQ(paged.block_table(), paged.block_table_mapping.get());
  EXPECT_EQ(paged.block_table_logical_bytes, 24u);
  EXPECT_EQ(paged.block_table_mapping_bytes, 24u);
  EXPECT_TRUE(paged.block_table_read_only);
  EXPECT_TRUE(paged.table_validation.valid);
  EXPECT_FALSE(paged.table_validation.interrupted);
  EXPECT_EQ(paged.table_validation.reason_code, LlmKvLayoutReason::VALID);
  EXPECT_EQ(paged.table_validation.expected_entries, 6u);
  EXPECT_EQ(paged.table_validation.examined_entries, 6u);
  EXPECT_EQ(paged.table_validation.validation_bitset_bytes, 1u);
  EXPECT_EQ(paged.permutation.algorithm_version,
            Constants::LLM_KV_BLOCK_PERMUTATION_VERSION);
  EXPECT_EQ(paged.permutation.resolved_seed,
            derive_llm_kv_permutation_seed(42));
  EXPECT_EQ(paged.permutation.entry_count, 6u);
  EXPECT_EQ(paged.permutation.sha256.size(), 64u);
  EXPECT_EQ(paged_table_entries(plan),
            (std::vector<uint32_t>{2, 3, 5, 1, 4, 0}));
  EXPECT_EQ(paged.layout_identity,
            serialize_llm_kv_layout_identity(paged.layout,
                                             paged.permutation));
  EXPECT_EQ(paged.layout_identity.rfind(
                Constants::LLM_KV_LAYOUT_PLAN_IDENTITY_VERSION, 0),
            0u);
  EXPECT_EQ(paged.execution_identity, paged.ownership.identity);
  EXPECT_EQ(paged.execution_identity.rfind(
                Constants::LLM_PAGED_CPU_EXECUTION_IDENTITY_VERSION, 0),
            0u);
  EXPECT_NE(plan.plan_identity.find(
                "|paged_layout_identity_size=" +
                std::to_string(paged.layout_identity.size())),
            std::string::npos);
  EXPECT_NE(plan.plan_identity.find(
                "|paged_layout_identity=" + paged.layout_identity),
            std::string::npos);
  EXPECT_NE(plan.plan_identity.find(
                "|paged_execution_identity_size=" +
                std::to_string(paged.execution_identity.size())),
            std::string::npos);
  EXPECT_NE(plan.plan_identity.find(
                "|paged_execution_identity=" + paged.execution_identity),
            std::string::npos);

  EXPECT_EQ(plan.component_identities.logical_profile_version,
            Constants::LLM_LOGICAL_PROFILE_VERSION);
  EXPECT_EQ(plan.component_identities.kv_layout_version,
            Constants::LLM_PAGED_KV_LAYOUT_VERSION);
  ASSERT_TRUE(plan.component_identities.permutation_version.has_value());
  EXPECT_EQ(*plan.component_identities.permutation_version,
            Constants::LLM_KV_BLOCK_PERMUTATION_VERSION);
  EXPECT_EQ(plan.component_identities.backend_executor_version,
            Constants::LLM_PAGED_CPU_EXECUTOR_VERSION);
  EXPECT_EQ(plan.component_identities.resource_abi_version,
            Constants::LLM_PAGED_DESCRIPTOR_ABI_VERSION);
  EXPECT_EQ(plan.component_identities.schedule_version,
            Constants::LLM_PAGED_CPU_SCHEDULE_VERSION);
  EXPECT_EQ(plan.component_identities.buffer_pattern_version,
            Constants::LLM_PAGED_BUFFER_PATTERN_VERSION);
  EXPECT_EQ(plan.component_identities.write_pattern_version,
            Constants::LLM_APPEND_PATTERN_VERSION);
  EXPECT_EQ(plan.component_identities.checksum_pattern_version,
            Constants::LLM_PAGED_READ_CHECKSUM_VERSION);

  const LlmMemoryBudgetRequest& budget = plan.memory_budget.request;
  ASSERT_TRUE(budget.valid) << budget.reason_code;
  EXPECT_EQ(budget.mapping_granularity_bytes, 1u);
  EXPECT_EQ(budget.requested_weight_mapping_bytes, 1024u);
  EXPECT_EQ(budget.requested_k_mapping_bytes, 6144u);
  EXPECT_EQ(budget.requested_v_mapping_bytes, 6144u);
  EXPECT_EQ(budget.committed_weight_mapping_bytes, 1024u);
  EXPECT_EQ(budget.committed_k_mapping_bytes, 6144u);
  EXPECT_EQ(budget.committed_v_mapping_bytes, 6144u);
  EXPECT_EQ(budget.requested_block_table_mapping_bytes, 24u);
  EXPECT_EQ(budget.committed_block_table_mapping_bytes, 24u);
  EXPECT_EQ(budget.requested_data_bytes, 13312u);
  EXPECT_EQ(budget.committed_data_bytes, 13312u);
  EXPECT_EQ(budget.layout_transient_bytes, 1u);
  EXPECT_EQ(budget.descriptor_bytes, 2400u);
  EXPECT_EQ(budget.planner_storage_bytes, cpu_plan.planner_storage_bytes);
  EXPECT_EQ(budget.auxiliary_bytes,
            budget.descriptor_bytes + budget.planner_storage_bytes);
  EXPECT_EQ(budget.setup_peak_bytes,
            24u + 1u + budget.planner_storage_bytes);
  EXPECT_EQ(budget.runtime_peak_bytes,
            13312u + 24u + budget.auxiliary_bytes);
  EXPECT_EQ(budget.required_total_bytes, budget.runtime_peak_bytes);
}

TEST(LlmMemoryWorkPlanTest,
     PagedActivePermutationAndIdentityAreSeedDeterministic) {
  LlmMemoryWorkPlanRequest request =
      work_plan_request(paged_geometry_request(), 5, 5);
  const LlmMemoryWorkPlan first = build_llm_memory_work_plan(request);
  const LlmMemoryWorkPlan second = build_llm_memory_work_plan(request);
  ++request.base_seed;
  const LlmMemoryWorkPlan different_seed =
      build_llm_memory_work_plan(request);
  ASSERT_TRUE(first.valid) << first.reason_code;
  ASSERT_TRUE(second.valid) << second.reason_code;
  ASSERT_TRUE(different_seed.valid) << different_seed.reason_code;

  const LlmPagedCpuExecutionPlan& first_paged =
      paged_cpu_execution_plan(first);
  const LlmPagedCpuExecutionPlan& second_paged =
      paged_cpu_execution_plan(second);
  const LlmPagedCpuExecutionPlan& different_paged =
      paged_cpu_execution_plan(different_seed);
  EXPECT_EQ(paged_table_entries(first), paged_table_entries(second));
  EXPECT_EQ(first_paged.permutation.resolved_seed,
            second_paged.permutation.resolved_seed);
  EXPECT_EQ(first_paged.permutation.sha256,
            second_paged.permutation.sha256);
  EXPECT_EQ(first_paged.permutation.identity,
            second_paged.permutation.identity);
  EXPECT_EQ(first_paged.layout_identity, second_paged.layout_identity);
  EXPECT_EQ(first_paged.execution_identity,
            second_paged.execution_identity);
  EXPECT_EQ(first.plan_identity, second.plan_identity);

  EXPECT_EQ(paged_table_entries(different_seed),
            (std::vector<uint32_t>{0, 1, 2, 3, 4, 5}));
  EXPECT_NE(paged_table_entries(first), paged_table_entries(different_seed));
  EXPECT_NE(first_paged.permutation.resolved_seed,
            different_paged.permutation.resolved_seed);
  EXPECT_NE(first_paged.permutation.sha256,
            different_paged.permutation.sha256);
  EXPECT_NE(first_paged.permutation.identity,
            different_paged.permutation.identity);
  EXPECT_NE(first_paged.layout_identity, different_paged.layout_identity);
  EXPECT_EQ(first_paged.execution_identity,
            different_paged.execution_identity);
  EXPECT_NE(first.plan_identity, different_seed.plan_identity);
  EXPECT_TRUE(first_paged.block_table_read_only);
  EXPECT_TRUE(second_paged.block_table_read_only);
  EXPECT_TRUE(different_paged.block_table_read_only);
  EXPECT_NE(first_paged.block_table(), second_paged.block_table());
}

TEST(LlmMemoryWorkPlanTest,
     PagedActivePlanPropagatesPreparationStopPredicate) {
  size_t stop_checks = 0;
  const LlmMemoryWorkPlan plan = build_llm_memory_work_plan(
      work_plan_request(paged_geometry_request(), 5, 5),
      [&stop_checks]() {
        ++stop_checks;
        return stop_checks == 2;
      });

  expect_invalid_plan(plan, LlmKvLayoutReason::PREPARATION_INTERRUPTED);
  EXPECT_EQ(stop_checks, 2u);
  EXPECT_FALSE(cpu_execution_plan(plan).paged.has_value());
}

TEST(LlmMemoryWorkPlanTest,
     PagedActivePlanAdmitsEstimatedPlannerStorageBeforeOwnershipAllocation) {
  LlmMemoryWorkPlanRequest request =
      work_plan_request(paged_geometry_request(), 5, 5);
  request.available_memory_bytes = 1;

  const LlmMemoryWorkPlan plan = build_llm_memory_work_plan(request);
  expect_invalid_plan(plan, LlmWorkPlanReason::MEMORY_BUDGET_EXCEEDED);
  EXPECT_FALSE(cpu_execution_plan(plan).paged.has_value());
  EXPECT_GT(plan.memory_budget.request.planner_storage_bytes, 0u);
}

TEST(LlmMemoryWorkPlanTest,
     PagedActiveWorkerTemplatesMirrorRotatedBlockOwnership) {
  const LlmMemoryWorkPlan plan = build_llm_memory_work_plan(
      work_plan_request(paged_geometry_request(), 5, 5));
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  const LlmCpuExecutionPlan& cpu_plan = cpu_execution_plan(plan);
  const LlmPagedCpuExecutionPlan& paged = paged_cpu_execution_plan(plan);
  const LlmKvCpuOwnershipPlan& ownership = paged.ownership;
  ASSERT_EQ(cpu_plan.effective_workers, 5u);
  ASSERT_EQ(paged.layout.blocks_per_sequence, 3u);
  EXPECT_TRUE(ownership.valid);
  EXPECT_EQ(ownership.worker_count, 5u);
  EXPECT_EQ(ownership.layer_sequence_count, 4u);
  EXPECT_EQ(ownership.total_owned_blocks, 12u);
  EXPECT_EQ(ownership.assignments.size(), 12u);
  EXPECT_EQ(ownership.total_model_payload_bytes_per_work_unit, 9216u);
  EXPECT_EQ(ownership.total_layout_metadata_lookup_count_per_work_unit,
            28u);
  EXPECT_EQ(ownership.total_layout_metadata_read_bytes_per_work_unit,
            112u);
  EXPECT_EQ(ownership.total_accounted_bytes_per_work_unit, 9328u);
  EXPECT_EQ(ownership.worker_accounted_bytes_per_work_unit,
            (std::vector<size_t>{1300, 2064, 2332, 2332, 1300}));
  EXPECT_EQ(ownership.minimum_worker_accounted_bytes_per_work_unit, 1300u);
  EXPECT_EQ(ownership.maximum_worker_accounted_bytes_per_work_unit, 2332u);
  EXPECT_EQ(ownership.worker_accounted_imbalance_bytes_per_work_unit,
            1032u);

  size_t source_index = 0;
  for (size_t ordinal = 0; ordinal < 4; ++ordinal) {
    const size_t layer = ordinal / 2;
    const size_t batch = ordinal % 2;
    for (size_t rank = 0; rank < 3; ++rank) {
      ASSERT_LT(source_index, ownership.assignments.size());
      const LlmKvCpuBlockAssignment& source =
          ownership.assignments[source_index++];
      const size_t expected_worker = (ordinal + rank) % 5;
      EXPECT_EQ(source.layer_index, layer);
      EXPECT_EQ(source.batch_sequence_index, batch);
      EXPECT_EQ(source.worker_index, expected_worker);
      EXPECT_EQ(source.first_logical_block, rank);
      EXPECT_EQ(source.block_count, 1u);

      const LlmPagedKvAssignmentTemplate& destination =
          cpu_plan.workers[expected_worker].paged_assignments[ordinal];
      EXPECT_EQ(destination.layer_index, layer);
      EXPECT_EQ(destination.batch_sequence_index, batch);
      EXPECT_EQ(destination.first_logical_block, rank);
      EXPECT_EQ(destination.block_count, 1u);
    }
    size_t row_blocks = 0;
    size_t row_nonempty_workers = 0;
    for (const LlmWorkerWorkPlan& worker : cpu_plan.workers) {
      const LlmPagedKvAssignmentTemplate& destination =
          worker.paged_assignments[ordinal];
      EXPECT_EQ(destination.layer_index, layer);
      EXPECT_EQ(destination.batch_sequence_index, batch);
      row_blocks += destination.block_count;
      row_nonempty_workers += destination.block_count != 0 ? 1 : 0;
    }
    EXPECT_EQ(row_blocks, 3u);
    EXPECT_EQ(row_nonempty_workers, 3u);
  }
  EXPECT_EQ(source_index, ownership.assignments.size());
}

TEST(LlmMemoryWorkPlanTest,
     PagedActivePlanGivesEveryEffectiveWorkerKvOwnership) {
  const LlmMemoryWorkPlan plan = build_llm_memory_work_plan(
      work_plan_request(paged_geometry_request(), 8, 8));
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  const LlmCpuExecutionPlan& cpu_plan = cpu_execution_plan(plan);
  ASSERT_EQ(cpu_plan.effective_workers, 6u);

  for (const LlmWorkerWorkPlan& worker : cpu_plan.workers) {
    size_t owned_blocks = 0;
    for (const LlmPagedKvAssignmentTemplate& assignment :
         worker.paged_assignments) {
      owned_blocks += assignment.block_count;
    }
    EXPECT_GT(owned_blocks, 0u) << "worker " << worker.worker_index;
  }
}

TEST(LlmMemoryWorkPlanTest,
     PagedActiveScenariosUseExactLookupAccountingAndGuardrails) {
  const LlmMemoryWorkPlan model_plan = build_llm_memory_work_plan(
      work_plan_request(paged_geometry_request(), 5, 5));
  ASSERT_TRUE(model_plan.valid) << model_plan.reason_code;
  constexpr size_t kExpectedLookups = 2u * 2u * (2u * 3u + 1u);
  static_assert(kExpectedLookups == 28u);

  struct ScenarioGolden {
    LlmScenario scenario;
    size_t weight_bytes;
    size_t kv_read_bytes;
    size_t kv_write_bytes;
    size_t payload_bytes;
    size_t lookup_count;
    size_t metadata_bytes;
    size_t accounted_bytes;
  };
  const std::array<ScenarioGolden, 3> goldens = {{
      {LlmScenario::WeightsOnly, 1024, 0, 0, 1024, 0, 0, 1024},
      {LlmScenario::KvOnly, 0, 8960, 256, 9216, kExpectedLookups,
       112, 9328},
      {LlmScenario::Mixed, 1024, 8960, 256, 10240,
       kExpectedLookups, 112, 10352},
  }};

  for (const ScenarioGolden& golden : goldens) {
    SCOPED_TRACE(static_cast<int>(golden.scenario));
    const LlmScenarioLimits limits =
        calculate_llm_scenario_limits(model_plan.geometry, golden.scenario);
    ASSERT_TRUE(limits.valid) << limits.reason_code;
    EXPECT_EQ(limits.weight_read_bytes_per_work_unit,
              golden.weight_bytes);
    EXPECT_EQ(limits.kv_read_bytes_per_work_unit, golden.kv_read_bytes);
    EXPECT_EQ(limits.kv_write_bytes_per_work_unit,
              golden.kv_write_bytes);
    EXPECT_EQ(limits.effective_model_payload_bytes_per_work_unit,
              golden.payload_bytes);
    EXPECT_EQ(limits.layout_metadata_lookup_count_per_work_unit,
              golden.lookup_count);
    EXPECT_EQ(limits.layout_metadata_read_bytes_per_work_unit,
              golden.metadata_bytes);
    EXPECT_EQ(limits.accounted_bytes_per_work_unit,
              golden.accounted_bytes);
    const size_t expected_guardrail =
        Constants::LLM_MAX_ACCOUNTED_BYTES_PER_TASK /
        golden.accounted_bytes;
    EXPECT_EQ(limits.maximum_work_units_by_guardrail,
              expected_guardrail);
    EXPECT_EQ(limits.effective_maximum_work_units, expected_guardrail);

    const LlmScenarioWorkPlan task = build_llm_scenario_work_plan(
        model_plan, golden.scenario, 3, true);
    ASSERT_TRUE(task.valid) << task.reason_code;
    EXPECT_EQ(task.work_units, 3u);
    EXPECT_EQ(task.weight_read_bytes, 3u * golden.weight_bytes);
    EXPECT_EQ(task.kv_read_bytes, 3u * golden.kv_read_bytes);
    EXPECT_EQ(task.kv_write_bytes, 3u * golden.kv_write_bytes);
    EXPECT_EQ(task.effective_model_payload_bytes,
              3u * golden.payload_bytes);
    EXPECT_EQ(task.layout_metadata_lookup_count,
              3u * golden.lookup_count);
    EXPECT_EQ(task.layout_metadata_read_bytes,
              3u * golden.metadata_bytes);
    EXPECT_EQ(task.task_accounted_bytes, 3u * golden.accounted_bytes);

    const LlmScenarioWorkPlan boundary = build_llm_scenario_work_plan(
        model_plan, golden.scenario, expected_guardrail, true);
    ASSERT_TRUE(boundary.valid) << boundary.reason_code;
    EXPECT_LE(boundary.task_accounted_bytes,
              Constants::LLM_MAX_ACCOUNTED_BYTES_PER_TASK);
    const LlmScenarioWorkPlan excess = build_llm_scenario_work_plan(
        model_plan, golden.scenario, expected_guardrail + 1, true);
    EXPECT_FALSE(excess.valid);
    EXPECT_EQ(excess.reason_code,
              LlmWorkPlanReason::TASK_ACCOUNTED_BYTES_CAP_EXCEEDED);
  }

  const LlmFrozenScenarioPlans frozen =
      freeze_llm_scenario_work_plans(model_plan, {3, 3, 3}, true);
  ASSERT_TRUE(frozen.valid) << frozen.reason_code;
  EXPECT_EQ(frozen.model_plan_identity, model_plan.plan_identity);
  for (size_t index = 0; index < goldens.size(); ++index) {
    EXPECT_EQ(frozen.scenarios[index].scenario, goldens[index].scenario);
    EXPECT_EQ(frozen.scenarios[index].layout_metadata_lookup_count,
              3u * goldens[index].lookup_count);
    EXPECT_EQ(frozen.scenarios[index].layout_metadata_read_bytes,
              3u * goldens[index].metadata_bytes);
    EXPECT_EQ(frozen.scenarios[index].task_accounted_bytes,
              3u * goldens[index].accounted_bytes);
  }
}

TEST(LlmMemoryWorkPlanTest,
     PagedActiveGeometryDistinguishesFullAndPartialLastBlocks) {
  struct GeometryGolden {
    size_t context_tokens;
    size_t blocks_per_sequence;
    size_t physical_blocks_per_layer;
    size_t total_physical_blocks;
    size_t last_block_tokens;
    size_t last_block_valid_bytes;
    size_t append_offset;
    size_t logical_bytes;
    size_t physical_bytes;
    size_t padding_bytes;
    size_t table_entries;
    size_t table_bytes;
  };
  const std::array<GeometryGolden, 3> goldens = {{
      {32, 2, 4, 8, 16, 512, 480, 4096, 4096, 0, 4, 16},
      {33, 3, 6, 12, 1, 32, 0, 4224, 6144, 1920, 6, 24},
      {3, 1, 2, 4, 3, 96, 64, 384, 2048, 1664, 2, 8},
  }};
  for (const GeometryGolden& golden : goldens) {
    SCOPED_TRACE(golden.context_tokens);
    const LlmMemoryWorkPlan plan = build_llm_memory_work_plan(
        work_plan_request(
            paged_geometry_request(golden.context_tokens, 16, 2), 5, 5));
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    const LlmGeometry& geometry = plan.geometry;
    EXPECT_EQ(geometry.kv_blocks_per_sequence,
              golden.blocks_per_sequence);
    EXPECT_EQ(geometry.physical_blocks_per_layer,
              golden.physical_blocks_per_layer);
    EXPECT_EQ(geometry.total_physical_blocks,
              golden.total_physical_blocks);
    EXPECT_EQ(geometry.kv_block_bytes, 512u);
    EXPECT_EQ(geometry.last_block_tokens, golden.last_block_tokens);
    EXPECT_EQ(geometry.last_block_valid_bytes,
              golden.last_block_valid_bytes);
    EXPECT_EQ(geometry.decode_append_offset_in_last_block,
              golden.append_offset);
    EXPECT_EQ(geometry.k_logical_bytes, golden.logical_bytes);
    EXPECT_EQ(geometry.v_logical_bytes, golden.logical_bytes);
    EXPECT_EQ(geometry.k_mapping_bytes, golden.physical_bytes);
    EXPECT_EQ(geometry.v_mapping_bytes, golden.physical_bytes);
    EXPECT_EQ(geometry.k_layout_padding_bytes, golden.padding_bytes);
    EXPECT_EQ(geometry.v_layout_padding_bytes, golden.padding_bytes);
    EXPECT_EQ(geometry.block_table_entries, golden.table_entries);
    EXPECT_EQ(geometry.block_table_bytes, golden.table_bytes);
    const LlmPagedCpuExecutionPlan& paged =
        paged_cpu_execution_plan(plan);
    EXPECT_EQ(paged.block_table_logical_bytes, golden.table_bytes);
    EXPECT_TRUE(paged.block_table_read_only);
  }
}

TEST(LlmMemoryWorkPlanTest,
     ContiguousGeometryRejectsNonzeroPagedBlockSize) {
  LlmGeometryRequest request = small_geometry_request();
  request.kv_block_tokens = 16;

  const LlmGeometry geometry = resolve_llm_geometry(request);
  EXPECT_FALSE(geometry.valid);
  EXPECT_EQ(geometry.reason_code,
            LlmWorkPlanReason::KV_BLOCK_TOKENS_NOT_APPLICABLE);
  expect_invalid_plan(
      build_llm_memory_work_plan(work_plan_request(request)),
      LlmWorkPlanReason::KV_BLOCK_TOKENS_NOT_APPLICABLE);
}

TEST(LlmMemoryWorkPlanTest,
     PagedActivePlanRejectsRawInvalidBlockSizesWithStableReasons) {
  struct InvalidCase {
    size_t block_tokens;
    const char* reason_code;
  };
  const size_t uint32_maximum =
      static_cast<size_t>(std::numeric_limits<uint32_t>::max());
  const std::array<InvalidCase, 4> invalid_cases = {{
      {0, LlmKvLayoutReason::BLOCK_TOKENS_ZERO},
      {3, LlmKvLayoutReason::BLOCK_TOKENS_NOT_POWER_OF_TWO},
      {uint32_maximum,
       LlmKvLayoutReason::BLOCK_TOKENS_NOT_POWER_OF_TWO},
      {uint32_maximum + 1,
       LlmKvLayoutReason::BLOCK_TOKENS_EXCEEDS_UINT32},
  }};
  for (const InvalidCase& test_case : invalid_cases) {
    SCOPED_TRACE(test_case.reason_code);
    const LlmGeometryRequest geometry_request =
        paged_geometry_request(35, test_case.block_tokens, 2);
    const LlmGeometry geometry = resolve_llm_geometry(geometry_request);
    EXPECT_FALSE(geometry.valid);
    EXPECT_EQ(geometry.reason_code, test_case.reason_code);

    const LlmMemoryWorkPlan plan = build_llm_memory_work_plan(
        work_plan_request(geometry_request, 5, 5));
    expect_invalid_plan(plan, test_case.reason_code);
    EXPECT_FALSE(plan.geometry.valid);
    EXPECT_EQ(plan.geometry.reason_code, test_case.reason_code);
    EXPECT_FALSE(cpu_execution_plan(plan).paged.has_value());
  }
}

TEST_F(LlmMemoryWorkPlanSystemCallsTest,
       PagedPreflightPublishesNoTableBeforeFullAuxiliaryAdmission) {
  LlmMemoryWorkPlanDraft draft = prepare_llm_memory_work_plan(
      work_plan_request(paged_geometry_request(), 5, 5));

  ASSERT_TRUE(draft.valid) << draft.reason_code;
  ASSERT_TRUE(draft.auxiliary_preflight.valid);
  EXPECT_FALSE(draft.candidate.valid);
  const LlmPagedCpuExecutionPlan& candidate_paged =
      paged_cpu_execution_plan(draft.candidate);
  EXPECT_EQ(candidate_paged.block_table(), nullptr);
  EXPECT_FALSE(candidate_paged.block_table_read_only);
  EXPECT_EQ(state.map_calls, 0u);
  EXPECT_EQ(state.protect_calls, 0u);

  const LlmMemoryWorkPlan plan = finalize_llm_memory_work_plan(
      std::move(draft), 123, 456);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  EXPECT_EQ(plan.memory_budget.request.checksum_auxiliary_bytes, 123u);
  EXPECT_EQ(plan.memory_budget.request.orchestration_auxiliary_bytes, 456u);
  EXPECT_EQ(state.map_calls, 1u);
  EXPECT_EQ(state.protect_calls, 1u);
  EXPECT_TRUE(paged_cpu_execution_plan(plan).block_table_read_only);
}

TEST_F(LlmMemoryWorkPlanSystemCallsTest,
       PagedFinalAdmissionRejectsBeforeTableMapping) {
  LlmMemoryWorkPlanDraft draft = prepare_llm_memory_work_plan(
      work_plan_request(paged_geometry_request(), 5, 5));
  ASSERT_TRUE(draft.valid) << draft.reason_code;
  ASSERT_EQ(state.map_calls, 0u);

  const LlmMemoryWorkPlan plan = finalize_llm_memory_work_plan(
      std::move(draft), 0, 15 * kGiB);

  expect_invalid_plan(plan, LlmWorkPlanReason::MEMORY_BUDGET_EXCEEDED);
  EXPECT_EQ(state.map_calls, 0u);
  EXPECT_EQ(state.protect_calls, 0u);
}

TEST_F(LlmMemoryWorkPlanSystemCallsTest,
       PagedBlockTableMappingFailureDiscardsExecutableCandidate) {
  state.fail_map_on_call = 1;
  testing::internal::CaptureStderr();
  const LlmMemoryWorkPlan plan = build_llm_memory_work_plan(
      work_plan_request(paged_geometry_request(), 5, 5));
  testing::internal::GetCapturedStderr();

  expect_invalid_plan(plan, LlmWorkPlanReason::BLOCK_TABLE_MAPPING_FAILED);
  const LlmCpuExecutionPlan& cpu_plan = cpu_execution_plan(plan);
  EXPECT_EQ(cpu_plan.effective_workers, 0u);
  EXPECT_FALSE(cpu_plan.paged.has_value());
  EXPECT_EQ(state.map_calls, 1u);
  EXPECT_EQ(state.advise_calls, 0u);
  EXPECT_EQ(state.protect_calls, 0u);
  EXPECT_EQ(state.unmap_calls, 0u);
  EXPECT_EQ(state.last_map_size, 24u);
}

TEST_F(LlmMemoryWorkPlanSystemCallsTest,
       PagedBlockTableProtectionFailureUnmapsAndDiscardsCandidate) {
  state.protect_result = -1;
  const LlmMemoryWorkPlan plan = build_llm_memory_work_plan(
      work_plan_request(paged_geometry_request(), 5, 5));

  expect_invalid_plan(plan,
                      LlmWorkPlanReason::BLOCK_TABLE_PROTECTION_FAILED);
  const LlmCpuExecutionPlan& cpu_plan = cpu_execution_plan(plan);
  EXPECT_EQ(cpu_plan.effective_workers, 0u);
  EXPECT_FALSE(cpu_plan.paged.has_value());
  EXPECT_EQ(state.map_calls, 1u);
  EXPECT_EQ(state.advise_calls, 1u);
  EXPECT_EQ(state.protect_calls, 1u);
  EXPECT_EQ(state.last_protect_size, 24u);
  EXPECT_EQ(state.last_protect_flags, PROT_READ);
  EXPECT_EQ(state.unmap_calls, 1u);
  EXPECT_EQ(state.last_unmapped_pointer, state.storage.data());
  EXPECT_EQ(state.last_unmapped_size, 24u);
}

TEST(LlmMemoryWorkPlanTest,
     PagedReadmissionPreservesTablePointerHashAndWorkloadIdentity) {
  LlmMemoryWorkPlanRequest request =
      work_plan_request(paged_geometry_request(), 5, 5);
  request.checksum_auxiliary_bytes = 7;
  request.orchestration_auxiliary_bytes = 9;
  LlmMemoryWorkPlan plan = build_llm_memory_work_plan(request);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  const LlmPagedCpuExecutionPlan& before =
      paged_cpu_execution_plan(plan);
  const uint32_t* const table_pointer = before.block_table();
  const std::vector<uint32_t> table = paged_table_entries(plan);
  const std::string table_hash = before.permutation.sha256;
  const std::string permutation_identity = before.permutation.identity;
  const std::string layout_identity = before.layout_identity;
  const std::string execution_identity = before.execution_identity;
  const std::string component_identity =
      plan.component_identities.identity;
  const std::string model_plan_identity = plan.plan_identity;
  const size_t required_before =
      plan.memory_budget.request.required_total_bytes;

  ASSERT_TRUE(readmit_llm_memory_work_plan(plan, 111, 222));
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  EXPECT_EQ(plan.reason_code, LlmWorkPlanReason::VALID);
  EXPECT_EQ(plan.memory_budget.request.checksum_auxiliary_bytes, 111u);
  EXPECT_EQ(plan.memory_budget.request.orchestration_auxiliary_bytes, 222u);
  EXPECT_NE(plan.memory_budget.request.required_total_bytes,
            required_before);
  const LlmPagedCpuExecutionPlan& after = paged_cpu_execution_plan(plan);
  EXPECT_EQ(after.block_table(), table_pointer);
  EXPECT_EQ(paged_table_entries(plan), table);
  EXPECT_TRUE(after.block_table_read_only);
  EXPECT_EQ(after.permutation.sha256, table_hash);
  EXPECT_EQ(after.permutation.identity, permutation_identity);
  EXPECT_EQ(after.layout_identity, layout_identity);
  EXPECT_EQ(after.execution_identity, execution_identity);
  EXPECT_EQ(plan.component_identities.identity, component_identity);
  EXPECT_EQ(plan.plan_identity, model_plan_identity);
}

TEST(LlmMemoryWorkPlanTest,
     PagedLayoutGeometryBudgetAndSetupMatchFrozenGolden) {
  const LlmKvLayoutPlan plan =
      build_llm_kv_layout_plan(paged_golden_request());
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  EXPECT_EQ(plan.reason_code, LlmKvLayoutReason::VALID);
  EXPECT_EQ(plan.blocks_per_sequence, 3u);
  EXPECT_EQ(plan.physical_blocks_per_layer, 6u);
  EXPECT_EQ(plan.total_physical_blocks, 12u);
  EXPECT_EQ(plan.block_bytes, 512u);
  EXPECT_EQ(plan.last_block_tokens, 3u);
  EXPECT_EQ(plan.last_block_valid_bytes, 96u);
  EXPECT_EQ(plan.decode_append_offset_in_last_block, 64u);
  EXPECT_EQ(plan.block_table_entries, 6u);

  EXPECT_EQ(plan.memory.k_logical_bytes, 4480u);
  EXPECT_EQ(plan.memory.v_logical_bytes, 4480u);
  EXPECT_EQ(plan.memory.k_physical_bytes, 6144u);
  EXPECT_EQ(plan.memory.v_physical_bytes, 6144u);
  EXPECT_EQ(plan.memory.k_layout_padding_bytes, 1664u);
  EXPECT_EQ(plan.memory.v_layout_padding_bytes, 1664u);
  EXPECT_EQ(plan.memory.block_table_bytes, 24u);
  EXPECT_EQ(plan.memory.validation_bitset_bytes, 1u);
  EXPECT_EQ(plan.memory.transient_peak_bytes, 25u);
  EXPECT_EQ(plan.memory.resident_layout_bytes, 12312u);
  EXPECT_EQ(plan.memory.known_owned_peak_bytes, 12313u);

  EXPECT_EQ(plan.permutation_iterations, 5u);
  EXPECT_EQ(plan.validation_entries, 6u);
  EXPECT_EQ(plan.hash_entries, 6u);
  EXPECT_EQ(plan.upload_bytes, 24u);
  EXPECT_EQ(plan.geometry_identity.rfind(
                Constants::LLM_KV_LAYOUT_GEOMETRY_IDENTITY_VERSION, 0),
            0u);
}

TEST(LlmMemoryWorkPlanTest,
     PagedLayoutRejectsZeroAndOverflowButAcceptsUint32BoundaryPurely) {
  struct InvalidCase {
    LlmKvLayoutRequest request;
    const char* reason_code;
  };
  const std::array<InvalidCase, 5> zero_cases = {{
      {{0, 16, 2, 2, 32}, LlmKvLayoutReason::SEQUENCE_TOKENS_ZERO},
      {{35, 0, 2, 2, 32}, LlmKvLayoutReason::BLOCK_TOKENS_ZERO},
      {{35, 16, 0, 2, 32}, LlmKvLayoutReason::LAYER_COUNT_ZERO},
      {{35, 16, 2, 0, 32}, LlmKvLayoutReason::BATCH_SIZE_ZERO},
      {{35, 16, 2, 2, 0}, LlmKvLayoutReason::RECORD_BYTES_ZERO},
  }};
  for (const InvalidCase& test_case : zero_cases) {
    SCOPED_TRACE(test_case.reason_code);
    const LlmKvLayoutPlan plan =
        build_llm_kv_layout_plan(test_case.request);
    EXPECT_FALSE(plan.valid);
    EXPECT_EQ(plan.reason_code, test_case.reason_code);
    EXPECT_TRUE(plan.geometry_identity.empty());
  }

  const size_t uint32_boundary =
      static_cast<size_t>(std::numeric_limits<uint32_t>::max());
  const std::array<InvalidCase, 3> block_token_cases = {{
      {{35, 3, 2, 2, 32},
       LlmKvLayoutReason::BLOCK_TOKENS_NOT_POWER_OF_TWO},
      {{35, uint32_boundary, 2, 2, 32},
       LlmKvLayoutReason::BLOCK_TOKENS_NOT_POWER_OF_TWO},
      {{35, uint32_boundary + 1, 2, 2, 32},
       LlmKvLayoutReason::BLOCK_TOKENS_EXCEEDS_UINT32},
  }};
  for (const InvalidCase& test_case : block_token_cases) {
    SCOPED_TRACE(test_case.reason_code);
    const LlmKvLayoutPlan plan =
        build_llm_kv_layout_plan(test_case.request);
    EXPECT_FALSE(plan.valid);
    EXPECT_EQ(plan.reason_code, test_case.reason_code);
    EXPECT_TRUE(plan.geometry_identity.empty());
  }

  const size_t maximum = std::numeric_limits<size_t>::max();
  const size_t maximum_block_tokens = size_t{1} << 31;
  const std::array<InvalidCase, 4> overflow_cases = {{
      {{2, 1, 1, maximum, 1},
       LlmKvLayoutReason::PHYSICAL_BLOCK_COUNT_OVERFLOW},
      {{1, 2, 1, 1, maximum}, LlmKvLayoutReason::BLOCK_BYTES_OVERFLOW},
      {{2, 1, maximum, 1, 1},
       LlmKvLayoutReason::LOGICAL_BYTES_OVERFLOW},
      {{1, maximum_block_tokens, 2, 1,
        maximum / maximum_block_tokens},
       LlmKvLayoutReason::PHYSICAL_BYTES_OVERFLOW},
  }};
  for (const InvalidCase& test_case : overflow_cases) {
    SCOPED_TRACE(test_case.reason_code);
    const LlmKvLayoutPlan plan =
        build_llm_kv_layout_plan(test_case.request);
    EXPECT_FALSE(plan.valid);
    EXPECT_EQ(plan.reason_code, test_case.reason_code);
    EXPECT_TRUE(plan.geometry_identity.empty());
  }

  const LlmKvLayoutPlan exact = build_llm_kv_layout_plan(
      {uint32_boundary, 1, 1, 1, 1});
  ASSERT_TRUE(exact.valid) << exact.reason_code;
  EXPECT_EQ(exact.blocks_per_sequence, uint32_boundary);
  EXPECT_EQ(exact.physical_blocks_per_layer, uint32_boundary);
  EXPECT_EQ(exact.block_table_entries, uint32_boundary);
  EXPECT_EQ(exact.permutation_iterations, uint32_boundary - 1);
  EXPECT_EQ(exact.memory.block_table_bytes,
            uint32_boundary * sizeof(uint32_t));
  EXPECT_EQ(exact.memory.validation_bitset_bytes,
            uint32_boundary / 8 + 1);

  const LlmKvLayoutPlan excess = build_llm_kv_layout_plan(
      {uint32_boundary + 1, 1, 1, 1, 1});
  EXPECT_FALSE(excess.valid);
  EXPECT_EQ(excess.reason_code,
            LlmKvLayoutReason::BLOCK_ID_RANGE_EXCEEDED);
  EXPECT_TRUE(excess.geometry_identity.empty());
}

TEST(LlmMemoryWorkPlanTest,
     PagedPermutationMatchesGoldensAcrossChunksAndPublishesNoPartialTable) {
  const LlmKvLayoutPlan layout =
      build_llm_kv_layout_plan({8, 1, 1, 1, 1});
  ASSERT_TRUE(layout.valid) << layout.reason_code;

  const std::vector<uint32_t> direct_entries = {2, 5, 0, 3, 4, 6, 1, 7};
  constexpr std::string_view kDirectHash =
      "9d1cfab79005723a285fec9a5716b53baa7a6c0501e3d17434bfb31ea88935d1";
  const std::array<size_t, 4> chunk_sizes = {
      1, 3, 8, std::numeric_limits<size_t>::max()};
  for (size_t chunk_size : chunk_sizes) {
    SCOPED_TRACE(chunk_size);
    const LlmKvBlockTable table =
        materialize_llm_kv_block_table(layout, 0, chunk_size);
    ASSERT_TRUE(table.valid) << table.reason_code;
    EXPECT_FALSE(table.interrupted);
    EXPECT_EQ(table.entries, direct_entries);
    EXPECT_TRUE(table.validation.valid);
    EXPECT_EQ(table.validation.examined_entries, 8u);
    EXPECT_EQ(table.validation.validation_bitset_bytes, 1u);
    EXPECT_EQ(table.permutation.resolved_seed, 0u);
    EXPECT_EQ(table.permutation.domain, 0x4C4C4D4B56504731ULL);
    EXPECT_EQ(table.permutation.domain_uint64_hex, "0x4c4c4d4b56504731");
    EXPECT_EQ(table.permutation.entry_count, 8u);
    EXPECT_EQ(table.permutation.sha256, kDirectHash);
    EXPECT_FALSE(table.permutation.identity.empty());
  }

  EXPECT_EQ(derive_llm_kv_permutation_seed(42),
            8109369757063363730ULL);
  const LlmKvBlockTable derived = materialize_llm_kv_block_table(
      layout, derive_llm_kv_permutation_seed(42), 3);
  ASSERT_TRUE(derived.valid) << derived.reason_code;
  EXPECT_EQ(derived.entries,
            (std::vector<uint32_t>{0, 6, 2, 3, 7, 1, 5, 4}));
  EXPECT_EQ(derived.permutation.sha256,
            "4032b29a855010d82199c15c3f3e2b94582b86e67b3add8cb86bebc425f9c2b4");

  struct ValidationCase {
    std::vector<uint32_t> entries;
    const char* reason_code;
  };
  const std::array<ValidationCase, 5> invalid_tables = {{
      {{0, 1, std::numeric_limits<uint32_t>::max(), 3},
       LlmKvLayoutReason::TABLE_INVALID_SENTINEL},
      {{0, 1, 4, 3}, LlmKvLayoutReason::TABLE_ID_OUT_OF_RANGE},
      {{0, 1, 1, 3}, LlmKvLayoutReason::TABLE_DUPLICATE_ID},
      {{0, 1, 2}, LlmKvLayoutReason::TABLE_ENTRY_COUNT_MISMATCH},
      {{0, 1, 2, 3, 4}, LlmKvLayoutReason::TABLE_ENTRY_COUNT_MISMATCH},
  }};
  for (const ValidationCase& test_case : invalid_tables) {
    SCOPED_TRACE(test_case.reason_code);
    const LlmKvBlockTableValidation validation =
        validate_llm_kv_block_table(4, test_case.entries);
    EXPECT_FALSE(validation.valid);
    EXPECT_FALSE(validation.interrupted);
    EXPECT_EQ(validation.reason_code, test_case.reason_code);
  }
  EXPECT_EQ(validate_llm_kv_block_table(0, {}).reason_code,
            LlmKvLayoutReason::TABLE_ENTRY_COUNT_MISMATCH);

  const size_t validation_interrupt_entries =
      Constants::LLM_KV_PREPARATION_POLL_INTERVAL_ENTRIES + 8;
  std::vector<uint32_t> validation_interrupt_table;
  validation_interrupt_table.reserve(validation_interrupt_entries);
  for (size_t entry = 0; entry < validation_interrupt_entries; ++entry) {
    validation_interrupt_table.push_back(static_cast<uint32_t>(entry));
  }
  size_t validation_stop_checks = 0;
  const LlmKvBlockTableValidation validation_interrupted =
      validate_llm_kv_block_table(
          validation_interrupt_entries, validation_interrupt_table,
          [&validation_stop_checks]() {
            ++validation_stop_checks;
            return validation_stop_checks == 2;
          });
  EXPECT_FALSE(validation_interrupted.valid);
  EXPECT_TRUE(validation_interrupted.interrupted);
  EXPECT_EQ(validation_interrupted.reason_code,
            LlmKvLayoutReason::PREPARATION_INTERRUPTED);
  EXPECT_EQ(validation_interrupted.examined_entries, 0u);
  EXPECT_EQ(validation_stop_checks, 2u);

  const LlmKvBlockTable zero_chunk =
      materialize_llm_kv_block_table(layout, 0, 0);
  EXPECT_FALSE(zero_chunk.valid);
  EXPECT_EQ(zero_chunk.reason_code,
            LlmKvLayoutReason::HASH_CHUNK_ENTRIES_ZERO);
  EXPECT_TRUE(zero_chunk.entries.empty());

  const size_t interrupt_entries =
      Constants::LLM_KV_PREPARATION_POLL_INTERVAL_ENTRIES + 1;
  const LlmKvLayoutPlan interrupt_layout =
      build_llm_kv_layout_plan({interrupt_entries, 1, 1, 1, 1});
  ASSERT_TRUE(interrupt_layout.valid) << interrupt_layout.reason_code;
  size_t stop_checks = 0;
  const LlmKvBlockTable interrupted = materialize_llm_kv_block_table(
      interrupt_layout, 0, 1024, [&stop_checks]() {
        ++stop_checks;
        return stop_checks == 2;
      });
  EXPECT_FALSE(interrupted.valid);
  EXPECT_TRUE(interrupted.interrupted);
  EXPECT_EQ(interrupted.reason_code,
            LlmKvLayoutReason::PREPARATION_INTERRUPTED);
  EXPECT_TRUE(interrupted.entries.empty());
  EXPECT_TRUE(interrupted.permutation.sha256.empty());
  EXPECT_EQ(stop_checks, 2u);

  size_t throwing_stop_checks = 0;
  EXPECT_THROW(
      static_cast<void>(materialize_llm_kv_block_table(
          layout, 0, 3, [&throwing_stop_checks]() {
            ++throwing_stop_checks;
            if (throwing_stop_checks == 8) {
              throw std::runtime_error("hash-stage stop failure");
            }
            return false;
          })),
      std::runtime_error);
  EXPECT_EQ(throwing_stop_checks, 8u);
}

TEST(LlmMemoryWorkPlanTest,
     PagedInPlacePermutationMatchesOwnedOutputAndRejectsInvalidStorage) {
  const LlmKvLayoutPlan layout =
      build_llm_kv_layout_plan({8, 1, 1, 1, 1});
  ASSERT_TRUE(layout.valid) << layout.reason_code;
  const LlmKvBlockTable owned =
      materialize_llm_kv_block_table(layout, 0, 3);
  ASSERT_TRUE(owned.valid) << owned.reason_code;

  const std::array<size_t, 4> chunk_sizes = {
      1, 3, 8, std::numeric_limits<size_t>::max()};
  for (size_t chunk_size : chunk_sizes) {
    SCOPED_TRACE(chunk_size);
    std::array<uint32_t, 8> entries{};
    const LlmKvInPlaceBlockTableMaterialization materialized =
        materialize_llm_kv_block_table_in_place(
            layout, 0, entries.data(), entries.size(), chunk_size);
    ASSERT_TRUE(materialized.valid) << materialized.reason_code;
    EXPECT_FALSE(materialized.interrupted);
    EXPECT_EQ(std::vector<uint32_t>(entries.begin(), entries.end()),
              owned.entries);
    EXPECT_TRUE(materialized.validation.valid);
    EXPECT_EQ(materialized.validation.reason_code,
              LlmKvLayoutReason::VALID);
    EXPECT_EQ(materialized.validation.expected_entries,
              owned.validation.expected_entries);
    EXPECT_EQ(materialized.validation.examined_entries,
              owned.validation.examined_entries);
    EXPECT_EQ(materialized.validation.validation_bitset_bytes,
              owned.validation.validation_bitset_bytes);
    EXPECT_EQ(materialized.permutation.algorithm_version,
              owned.permutation.algorithm_version);
    EXPECT_EQ(materialized.permutation.domain,
              owned.permutation.domain);
    EXPECT_EQ(materialized.permutation.domain_uint64_hex,
              owned.permutation.domain_uint64_hex);
    EXPECT_EQ(materialized.permutation.resolved_seed,
              owned.permutation.resolved_seed);
    EXPECT_EQ(materialized.permutation.entry_count,
              owned.permutation.entry_count);
    EXPECT_EQ(materialized.permutation.sha256,
              owned.permutation.sha256);
    EXPECT_EQ(materialized.permutation.identity,
              owned.permutation.identity);
  }

  const LlmKvInPlaceBlockTableMaterialization null_output =
      materialize_llm_kv_block_table_in_place(
          layout, 0, nullptr, layout.block_table_entries, 3);
  EXPECT_FALSE(null_output.valid);
  EXPECT_FALSE(null_output.interrupted);
  EXPECT_EQ(null_output.reason_code,
            LlmKvLayoutReason::TABLE_OUTPUT_NULL);
  EXPECT_TRUE(null_output.permutation.sha256.empty());
  EXPECT_TRUE(null_output.permutation.identity.empty());

  std::array<uint32_t, 8> mismatched_entries;
  mismatched_entries.fill(std::numeric_limits<uint32_t>::max());
  const std::array<uint32_t, 8> untouched_entries = mismatched_entries;
  const LlmKvInPlaceBlockTableMaterialization count_mismatch =
      materialize_llm_kv_block_table_in_place(
          layout, 0, mismatched_entries.data(),
          mismatched_entries.size() - 1, 3);
  EXPECT_FALSE(count_mismatch.valid);
  EXPECT_FALSE(count_mismatch.interrupted);
  EXPECT_EQ(count_mismatch.reason_code,
            LlmKvLayoutReason::TABLE_ENTRY_COUNT_MISMATCH);
  EXPECT_EQ(mismatched_entries, untouched_entries);
}

TEST(LlmMemoryWorkPlanTest,
     PagedInPlacePermutationPollsStopDuringCallerStorageInitialization) {
  const size_t entry_count =
      Constants::LLM_KV_PREPARATION_POLL_INTERVAL_ENTRIES + 1;
  const LlmKvLayoutPlan layout =
      build_llm_kv_layout_plan({entry_count, 1, 1, 1, 1});
  ASSERT_TRUE(layout.valid) << layout.reason_code;
  std::vector<uint32_t> entries(
      entry_count, std::numeric_limits<uint32_t>::max());
  size_t stop_checks = 0;
  const LlmKvInPlaceBlockTableMaterialization interrupted =
      materialize_llm_kv_block_table_in_place(
          layout, 0, entries.data(), entries.size(), 1024,
          [&stop_checks]() {
            ++stop_checks;
            return stop_checks == 2;
          });
  EXPECT_FALSE(interrupted.valid);
  EXPECT_TRUE(interrupted.interrupted);
  EXPECT_EQ(interrupted.reason_code,
            LlmKvLayoutReason::PREPARATION_INTERRUPTED);
  EXPECT_FALSE(interrupted.validation.valid);
  EXPECT_TRUE(interrupted.permutation.sha256.empty());
  EXPECT_TRUE(interrupted.permutation.identity.empty());
  EXPECT_EQ(stop_checks, 2u);
  EXPECT_EQ(entries.front(), 0u);
  EXPECT_EQ(entries.back(), std::numeric_limits<uint32_t>::max());
}

TEST(LlmMemoryWorkPlanTest,
     PagedDecodeScenariosAccountModelMetadataAndCheckedTaskTotals) {
  const LlmKvLayoutPlan layout =
      build_llm_kv_layout_plan(paged_golden_request());
  ASSERT_TRUE(layout.valid) << layout.reason_code;
  const LlmKvBlockTable table = materialize_llm_kv_block_table(
      layout, derive_llm_kv_permutation_seed(42));
  ASSERT_TRUE(table.valid) << table.reason_code;
  const std::string layout_identity =
      serialize_llm_kv_layout_identity(layout, table.permutation);
  ASSERT_FALSE(layout_identity.empty());

  struct ScenarioCase {
    LlmScenario scenario;
    size_t payload_per_work_unit;
    size_t lookups_per_work_unit;
    size_t metadata_bytes_per_work_unit;
    size_t accounted_bytes_per_work_unit;
  };
  const std::array<ScenarioCase, 3> cases = {{
      {LlmScenario::WeightsOnly, 1024, 0, 0, 1024},
      {LlmScenario::KvOnly, 9216, 28, 112, 9328},
      {LlmScenario::Mixed, 10240, 28, 112, 10352},
  }};
  constexpr size_t kWorkUnits = 3;
  for (const ScenarioCase& test_case : cases) {
    SCOPED_TRACE(llm_scenario_to_string(test_case.scenario));
    const LlmPagedDecodeWorkloadPlan plan =
        build_llm_paged_decode_workload_plan(
            layout, test_case.scenario, kWorkUnits,
            test_case.payload_per_work_unit, table.permutation);
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    EXPECT_EQ(plan.layout_identity, layout_identity);
    EXPECT_EQ(plan.layout_metadata_lookup_count_per_layer_sequence,
              test_case.lookups_per_work_unit == 0 ? 0u : 7u);
    EXPECT_EQ(plan.layout_metadata_lookup_count_per_work_unit,
              test_case.lookups_per_work_unit);
    EXPECT_EQ(plan.layout_metadata_read_bytes_per_work_unit,
              test_case.metadata_bytes_per_work_unit);
    EXPECT_EQ(plan.accounted_bytes_per_work_unit,
              test_case.accounted_bytes_per_work_unit);
    EXPECT_EQ(plan.maximum_work_units_by_guardrail,
              Constants::LLM_MAX_ACCOUNTED_BYTES_PER_TASK /
                  test_case.accounted_bytes_per_work_unit);
    EXPECT_EQ(plan.maximum_work_units_by_work_unit_cap,
              Constants::LLM_MAX_WORK_UNITS_PER_MEASUREMENT);
    EXPECT_EQ(plan.effective_maximum_work_units,
              std::min(Constants::LLM_MAX_WORK_UNITS_PER_MEASUREMENT,
                       plan.maximum_work_units_by_guardrail));
    EXPECT_EQ(plan.effective_model_payload_bytes,
              kWorkUnits * test_case.payload_per_work_unit);
    EXPECT_EQ(plan.layout_metadata_lookup_count,
              kWorkUnits * test_case.lookups_per_work_unit);
    EXPECT_EQ(plan.layout_metadata_read_bytes,
              kWorkUnits * test_case.metadata_bytes_per_work_unit);
    EXPECT_EQ(plan.task_accounted_bytes,
              kWorkUnits * test_case.accounted_bytes_per_work_unit);
    EXPECT_FALSE(plan.identity.empty());
  }

  const size_t maximum = std::numeric_limits<size_t>::max();
  const LlmPagedDecodeWorkloadPlan add_overflow =
      build_llm_paged_decode_workload_plan(
          layout, LlmScenario::KvOnly, 1, maximum, table.permutation);
  EXPECT_FALSE(add_overflow.valid);
  EXPECT_EQ(add_overflow.reason_code,
            LlmKvLayoutReason::TASK_TOTAL_OVERFLOW);
  EXPECT_TRUE(add_overflow.identity.empty());

  const LlmPagedDecodeWorkloadPlan work_unit_cap_excess =
      build_llm_paged_decode_workload_plan(
          layout, LlmScenario::WeightsOnly,
          Constants::LLM_MAX_WORK_UNITS_PER_MEASUREMENT + 1, 1,
          table.permutation);
  EXPECT_FALSE(work_unit_cap_excess.valid);
  EXPECT_EQ(work_unit_cap_excess.reason_code,
            LlmKvLayoutReason::WORK_UNIT_CAP_EXCEEDED);
  EXPECT_TRUE(work_unit_cap_excess.identity.empty());

  const LlmPagedDecodeWorkloadPlan exact_work_unit_cap =
      build_llm_paged_decode_workload_plan(
          layout, LlmScenario::WeightsOnly,
          Constants::LLM_MAX_WORK_UNITS_PER_MEASUREMENT, 1,
          table.permutation);
  ASSERT_TRUE(exact_work_unit_cap.valid)
      << exact_work_unit_cap.reason_code;
  EXPECT_EQ(exact_work_unit_cap.effective_maximum_work_units,
            Constants::LLM_MAX_WORK_UNITS_PER_MEASUREMENT);
  EXPECT_EQ(exact_work_unit_cap.task_accounted_bytes,
            Constants::LLM_MAX_WORK_UNITS_PER_MEASUREMENT);

  const size_t exact_work_units =
      Constants::LLM_MAX_ACCOUNTED_BYTES_PER_TASK / 1024;
  const LlmPagedDecodeWorkloadPlan exact_guardrail =
      build_llm_paged_decode_workload_plan(
          layout, LlmScenario::WeightsOnly, exact_work_units, 1024,
          table.permutation);
  ASSERT_TRUE(exact_guardrail.valid) << exact_guardrail.reason_code;
  EXPECT_EQ(exact_guardrail.maximum_work_units_by_guardrail,
            exact_work_units);
  EXPECT_EQ(exact_guardrail.task_accounted_bytes,
            Constants::LLM_MAX_ACCOUNTED_BYTES_PER_TASK);
  const LlmPagedDecodeWorkloadPlan excess_guardrail =
      build_llm_paged_decode_workload_plan(
          layout, LlmScenario::WeightsOnly, exact_work_units + 1, 1024,
          table.permutation);
  EXPECT_FALSE(excess_guardrail.valid);
  EXPECT_EQ(excess_guardrail.reason_code,
            LlmKvLayoutReason::TASK_ACCOUNTED_BYTES_CAP_EXCEEDED);

  const LlmPagedDecodeWorkloadPlan irreducible =
      build_llm_paged_decode_workload_plan(
          layout, LlmScenario::WeightsOnly, 1,
          Constants::LLM_MAX_ACCOUNTED_BYTES_PER_TASK + 1,
          table.permutation);
  EXPECT_FALSE(irreducible.valid);
  EXPECT_EQ(irreducible.reason_code,
            LlmKvLayoutReason::GUARDRAIL_BELOW_ONE_WORK_UNIT);

  const LlmPagedDecodeWorkloadPlan invalid_identity =
      build_llm_paged_decode_workload_plan(
          layout, LlmScenario::KvOnly, 1, 9216,
          LlmKvPermutationIdentity{});
  EXPECT_FALSE(invalid_identity.valid);
  EXPECT_EQ(invalid_identity.reason_code,
            LlmKvLayoutReason::INVALID_LAYOUT_IDENTITY);
}

TEST(LlmMemoryWorkPlanTest,
     PagedCpuOwnershipIsBlockExclusiveRotatingAndLookupInvariant) {
  const LlmKvLayoutPlan layout =
      build_llm_kv_layout_plan(paged_golden_request());
  ASSERT_TRUE(layout.valid) << layout.reason_code;

  constexpr size_t kWorkerCount = 5;
  const LlmKvCpuOwnershipPlan rotating =
      build_llm_paged_decode_kv_cpu_ownership_plan(layout, kWorkerCount);
  ASSERT_TRUE(rotating.valid) << rotating.reason_code;
  EXPECT_EQ(rotating.layer_sequence_count, 4u);
  EXPECT_EQ(rotating.total_owned_blocks, 12u);
  ASSERT_EQ(rotating.assignments.size(), 12u);
  const std::array<size_t, 3> expected_valid_tokens = {16, 16, 3};
  const std::array<size_t, 3> expected_model_payload = {1024, 1024, 256};
  const std::array<size_t, 3> expected_lookups = {2, 2, 3};

  for (size_t ordinal = 0; ordinal < rotating.layer_sequence_count;
       ++ordinal) {
    const size_t expected_layer = ordinal / layout.batch_size;
    const size_t expected_batch = ordinal % layout.batch_size;
    size_t next_block = 0;
    std::array<bool, 3> seen = {false, false, false};
    for (size_t rank = 0; rank < layout.blocks_per_sequence; ++rank) {
      const LlmKvCpuBlockAssignment& assignment =
          rotating.assignments[ordinal * layout.blocks_per_sequence + rank];
      EXPECT_EQ(assignment.layer_index, expected_layer);
      EXPECT_EQ(assignment.batch_sequence_index, expected_batch);
      EXPECT_EQ(assignment.worker_index,
                (ordinal + rank) % kWorkerCount);
      EXPECT_EQ(assignment.first_logical_block, next_block);
      EXPECT_EQ(assignment.block_count, 1u);
      ASSERT_LT(assignment.first_logical_block, seen.size());
      EXPECT_FALSE(seen[assignment.first_logical_block]);
      seen[assignment.first_logical_block] = true;
      EXPECT_EQ(assignment.valid_token_count, expected_valid_tokens[rank]);
      EXPECT_EQ(assignment.model_payload_bytes_per_work_unit,
                expected_model_payload[rank]);
      EXPECT_EQ(assignment.layout_metadata_lookup_count_per_work_unit,
                expected_lookups[rank]);
      EXPECT_EQ(assignment.layout_metadata_read_bytes_per_work_unit,
                expected_lookups[rank] * sizeof(uint32_t));
      next_block += assignment.block_count;
    }
    EXPECT_EQ(next_block, layout.blocks_per_sequence);
    EXPECT_EQ(seen, (std::array<bool, 3>{true, true, true}));
  }
  EXPECT_EQ(rotating.total_model_payload_bytes_per_work_unit, 9216u);
  EXPECT_EQ(rotating.total_layout_metadata_lookup_count_per_work_unit, 28u);
  EXPECT_EQ(rotating.total_layout_metadata_read_bytes_per_work_unit, 112u);
  EXPECT_EQ(rotating.total_accounted_bytes_per_work_unit, 9328u);
  EXPECT_EQ(rotating.worker_accounted_bytes_per_work_unit,
            (std::vector<size_t>{1300, 2064, 2332, 2332, 1300}));
  EXPECT_EQ(rotating.minimum_worker_accounted_bytes_per_work_unit, 1300u);
  EXPECT_EQ(rotating.maximum_worker_accounted_bytes_per_work_unit, 2332u);
  EXPECT_EQ(rotating.worker_accounted_imbalance_bytes_per_work_unit, 1032u);

  const std::array<size_t, 5> worker_counts = {1, 2, 3, 5, 8};
  for (size_t worker_count : worker_counts) {
    SCOPED_TRACE(worker_count);
    const LlmKvCpuOwnershipPlan plan =
        build_llm_paged_decode_kv_cpu_ownership_plan(layout, worker_count);
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    EXPECT_EQ(plan.total_owned_blocks, 12u);
    EXPECT_EQ(plan.total_model_payload_bytes_per_work_unit, 9216u);
    EXPECT_EQ(plan.total_layout_metadata_lookup_count_per_work_unit, 28u);
    EXPECT_EQ(plan.total_layout_metadata_read_bytes_per_work_unit, 112u);
    EXPECT_EQ(plan.total_accounted_bytes_per_work_unit, 9328u);
    EXPECT_EQ(plan.worker_accounted_bytes_per_work_unit.size(),
              worker_count);
    for (size_t worker = 0;
         worker < plan.worker_accounted_bytes_per_work_unit.size();
         ++worker) {
      if (worker_count == 8 && worker >= 6) {
        EXPECT_EQ(plan.worker_accounted_bytes_per_work_unit[worker], 0u);
      } else {
        EXPECT_GT(plan.worker_accounted_bytes_per_work_unit[worker], 0u);
      }
    }
  }
  EXPECT_EQ(build_llm_paged_decode_kv_cpu_ownership_plan(layout, 0)
                .reason_code,
            LlmKvLayoutReason::WORKER_COUNT_ZERO);

  const LlmKvCpuOwnershipPlan two_workers =
      build_llm_paged_decode_kv_cpu_ownership_plan(layout, 2);
  ASSERT_TRUE(two_workers.valid) << two_workers.reason_code;
  ASSERT_EQ(two_workers.assignments.size(), 8u);
  for (size_t ordinal = 0; ordinal < two_workers.layer_sequence_count;
       ++ordinal) {
    const size_t offset = ordinal * 2;
    EXPECT_EQ(two_workers.assignments[offset].first_logical_block, 0u);
    EXPECT_EQ(two_workers.assignments[offset].block_count, 1u);
    EXPECT_EQ(two_workers.assignments[offset + 1].first_logical_block, 1u);
    EXPECT_EQ(two_workers.assignments[offset + 1].block_count, 2u);
  }
  EXPECT_EQ(two_workers.worker_accounted_bytes_per_work_unit,
            (std::vector<size_t>{4664, 4664}));
  EXPECT_EQ(two_workers.worker_accounted_imbalance_bytes_per_work_unit, 0u);

  const LlmKvLayoutPlan tie_layout =
      build_llm_kv_layout_plan({10, 4, 1, 1, 2});
  ASSERT_TRUE(tie_layout.valid) << tie_layout.reason_code;
  const LlmKvCpuOwnershipPlan tie =
      build_llm_paged_decode_kv_cpu_ownership_plan(tie_layout, 2);
  ASSERT_TRUE(tie.valid) << tie.reason_code;
  ASSERT_EQ(tie.assignments.size(), 2u);
  EXPECT_EQ(tie.assignments[0].first_logical_block, 0u);
  EXPECT_EQ(tie.assignments[0].block_count, 1u);
  EXPECT_EQ(tie.assignments[1].first_logical_block, 1u);
  EXPECT_EQ(tie.assignments[1].block_count, 2u);
  EXPECT_EQ(tie.worker_accounted_bytes_per_work_unit,
            (std::vector<size_t>{24, 48}));
}

TEST(LlmMemoryWorkPlanTest,
     PagedCpuOwnershipPollsStopAndPublishesNoPartialAssignments) {
  const size_t layer_count =
      Constants::LLM_KV_PREPARATION_POLL_INTERVAL_ENTRIES + 1;
  const LlmKvLayoutPlan layout =
      build_llm_kv_layout_plan({1, 1, layer_count, 1, 1});
  ASSERT_TRUE(layout.valid) << layout.reason_code;
  size_t stop_checks = 0;

  const LlmKvCpuOwnershipPlan ownership =
      build_llm_paged_decode_kv_cpu_ownership_plan(
          layout, 1, [&stop_checks]() {
            ++stop_checks;
            return stop_checks == 4;
          });

  EXPECT_FALSE(ownership.valid);
  EXPECT_EQ(ownership.reason_code,
            LlmKvLayoutReason::PREPARATION_INTERRUPTED);
  EXPECT_TRUE(ownership.assignments.empty());
  EXPECT_TRUE(ownership.worker_accounted_bytes_per_work_unit.empty());
  EXPECT_TRUE(ownership.identity.empty());
  EXPECT_EQ(stop_checks, 4u);
}

TEST(LlmMemoryWorkPlanTest,
     PagedMetalSegmentsHonorInjectedWholeElementBoundariesAndExactSums) {
  struct BoundaryCase {
    size_t element_count;
    std::vector<size_t> expected_lengths;
  };
  const std::array<BoundaryCase, 3> boundary_cases = {{
      {15, {15}},
      {16, {16}},
      {17, {16, 1}},
  }};
  const LlmKvMetalSegmentLimits byte_limits = {16, 4};
  for (const BoundaryCase& test_case : boundary_cases) {
    SCOPED_TRACE(test_case.element_count);
    const LlmKvSegmentPlan plan = build_llm_kv_segment_plan(
        test_case.element_count, 1, byte_limits);
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    EXPECT_EQ(plan.segment_lengths, test_case.expected_lengths);
    EXPECT_EQ(segment_length_sum(plan), test_case.element_count);
    EXPECT_EQ(plan.total_length_bytes, test_case.element_count);
  }

  const LlmKvSegmentPlan non_dividing =
      build_llm_kv_segment_plan(5, 6, {16, 3});
  ASSERT_TRUE(non_dividing.valid) << non_dividing.reason_code;
  EXPECT_EQ(non_dividing.elements_per_segment, 2u);
  EXPECT_EQ(non_dividing.segment_count, 3u);
  EXPECT_EQ(non_dividing.segment_lengths,
            (std::vector<size_t>{12, 12, 6}));
  EXPECT_EQ(non_dividing.maximum_addressable_elements, 6u);
  EXPECT_EQ(non_dividing.maximum_addressable_bytes, 36u);
  EXPECT_EQ(non_dividing.unused_nominal_segment_capacity_bytes, 12u);
  EXPECT_EQ(non_dividing.total_length_bytes, 30u);
  EXPECT_EQ(segment_length_sum(non_dividing), 30u);

  const LlmKvSegmentPlan exact_256 =
      build_llm_kv_segment_plan(512, 4, {8, 256});
  ASSERT_TRUE(exact_256.valid) << exact_256.reason_code;
  EXPECT_EQ(exact_256.elements_per_segment, 2u);
  ASSERT_EQ(exact_256.segment_lengths.size(), 256u);
  for (size_t length : exact_256.segment_lengths) {
    EXPECT_EQ(length, 8u);
  }
  EXPECT_EQ(segment_length_sum(exact_256), 2048u);
  const LlmKvSegmentPlan excess_257 =
      build_llm_kv_segment_plan(513, 4, {8, 256});
  EXPECT_FALSE(excess_257.valid);
  EXPECT_EQ(excess_257.reason_code,
            LlmKvLayoutReason::SEGMENT_COUNT_EXCEEDS_CAP);
  EXPECT_TRUE(excess_257.segment_lengths.empty());

  const LlmKvSegmentPlan oversized_element =
      build_llm_kv_segment_plan(1, 17, {16, 4});
  EXPECT_FALSE(oversized_element.valid);
  EXPECT_EQ(oversized_element.reason_code,
            LlmKvLayoutReason::SEGMENT_ELEMENT_EXCEEDS_CAPACITY);

  const LlmKvLayoutPlan layout =
      build_llm_kv_layout_plan(paged_golden_request());
  ASSERT_TRUE(layout.valid) << layout.reason_code;
  const LlmKvMetalSegmentPlan composite =
      build_llm_kv_metal_segment_plan(layout, {1024, 16});
  ASSERT_TRUE(composite.valid) << composite.reason_code;
  EXPECT_EQ(composite.k_or_v_pool.elements_per_segment, 2u);
  EXPECT_EQ(composite.k_or_v_pool.segment_count, 6u);
  EXPECT_EQ(segment_length_sum(composite.k_or_v_pool),
            layout.memory.k_physical_bytes);
  EXPECT_EQ(composite.block_table.segment_count, 1u);
  EXPECT_EQ(composite.block_table.segment_lengths,
            (std::vector<size_t>{24}));
  EXPECT_EQ(segment_length_sum(composite.block_table),
            layout.memory.block_table_bytes);
}

TEST(LlmMemoryWorkPlanTest,
     PagedCanonicalIdentitiesAreDeterministicAndSensitiveByLayer) {
  const LlmKvLayoutPlan layout_a =
      build_llm_kv_layout_plan(paged_golden_request());
  const LlmKvLayoutPlan layout_b =
      build_llm_kv_layout_plan(paged_golden_request());
  const LlmKvLayoutPlan layout_changed =
      build_llm_kv_layout_plan({36, 16, 2, 2, 32});
  ASSERT_TRUE(layout_a.valid) << layout_a.reason_code;
  ASSERT_TRUE(layout_b.valid) << layout_b.reason_code;
  ASSERT_TRUE(layout_changed.valid) << layout_changed.reason_code;
  EXPECT_EQ(layout_a.geometry_identity, layout_b.geometry_identity);
  EXPECT_NE(layout_a.geometry_identity, layout_changed.geometry_identity);

  const LlmKvBlockTable permutation_a =
      materialize_llm_kv_block_table(layout_a, 0, 1);
  const LlmKvBlockTable permutation_b =
      materialize_llm_kv_block_table(layout_b, 0, 3);
  const LlmKvBlockTable permutation_changed = materialize_llm_kv_block_table(
      layout_a, derive_llm_kv_permutation_seed(42), 3);
  ASSERT_TRUE(permutation_a.valid) << permutation_a.reason_code;
  ASSERT_TRUE(permutation_b.valid) << permutation_b.reason_code;
  ASSERT_TRUE(permutation_changed.valid) << permutation_changed.reason_code;

  const std::string serialized_layout_a =
      serialize_llm_kv_layout_identity(layout_a, permutation_a.permutation);
  const std::string serialized_layout_b =
      serialize_llm_kv_layout_identity(layout_b, permutation_b.permutation);
  const std::string serialized_layout_changed = serialize_llm_kv_layout_identity(
      layout_a, permutation_changed.permutation);
  EXPECT_EQ(serialized_layout_a, serialized_layout_b);
  EXPECT_NE(serialized_layout_a, serialized_layout_changed);
  EXPECT_TRUE(validate_llm_kv_layout_identity(
      layout_a, permutation_a.permutation, serialized_layout_a));
  EXPECT_FALSE(validate_llm_kv_layout_identity(
      layout_a, permutation_changed.permutation, serialized_layout_a));
  std::string tampered_layout_identity = serialized_layout_a;
  ASSERT_FALSE(tampered_layout_identity.empty());
  tampered_layout_identity.back() ^= 1;
  EXPECT_FALSE(validate_llm_kv_layout_identity(
      layout_a, permutation_a.permutation, tampered_layout_identity));
  EXPECT_EQ(serialized_layout_a.rfind(
                Constants::LLM_KV_LAYOUT_PLAN_IDENTITY_VERSION, 0),
            0u);
  expect_length_prefixed_identity_field(
      serialized_layout_a, "geometry_identity", layout_a.geometry_identity);
  expect_length_prefixed_identity_field(
      serialized_layout_a, "permutation_algorithm_version",
      permutation_a.permutation.algorithm_version);
  expect_length_prefixed_identity_field(
      serialized_layout_a, "permutation_domain_uint64_hex",
      permutation_a.permutation.domain_uint64_hex);
  expect_length_prefixed_identity_field(
      serialized_layout_a, "permutation_sha256",
      permutation_a.permutation.sha256);

  const LlmPagedDecodeWorkloadPlan workload_a =
      build_llm_paged_decode_workload_plan(
          layout_a, LlmScenario::KvOnly, 3, 9216,
          permutation_a.permutation);
  const LlmPagedDecodeWorkloadPlan workload_b =
      build_llm_paged_decode_workload_plan(
          layout_b, LlmScenario::KvOnly, 3, 9216,
          permutation_b.permutation);
  const LlmPagedDecodeWorkloadPlan workload_changed =
      build_llm_paged_decode_workload_plan(
          layout_a, LlmScenario::KvOnly, 4, 9216,
          permutation_a.permutation);
  const LlmPagedDecodeWorkloadPlan workload_seed_changed =
      build_llm_paged_decode_workload_plan(
          layout_a, LlmScenario::KvOnly, 3, 9216,
          permutation_changed.permutation);
  ASSERT_TRUE(workload_a.valid) << workload_a.reason_code;
  ASSERT_TRUE(workload_b.valid) << workload_b.reason_code;
  ASSERT_TRUE(workload_changed.valid) << workload_changed.reason_code;
  ASSERT_TRUE(workload_seed_changed.valid)
      << workload_seed_changed.reason_code;
  EXPECT_EQ(workload_a.identity, workload_b.identity);
  EXPECT_NE(workload_a.identity, workload_changed.identity);
  EXPECT_NE(workload_a.identity, workload_seed_changed.identity);
  EXPECT_EQ(workload_a.identity.rfind(
                Constants::LLM_PAGED_DECODE_WORKLOAD_IDENTITY_VERSION, 0),
            0u);
  expect_length_prefixed_identity_field(
      workload_a.identity, "layout_identity", serialized_layout_a);

  const LlmKvCpuOwnershipPlan ownership_a =
      build_llm_paged_decode_kv_cpu_ownership_plan(layout_a, 5);
  const LlmKvCpuOwnershipPlan ownership_b =
      build_llm_paged_decode_kv_cpu_ownership_plan(layout_b, 5);
  const LlmKvCpuOwnershipPlan ownership_changed =
      build_llm_paged_decode_kv_cpu_ownership_plan(layout_a, 4);
  ASSERT_TRUE(ownership_a.valid) << ownership_a.reason_code;
  ASSERT_TRUE(ownership_b.valid) << ownership_b.reason_code;
  ASSERT_TRUE(ownership_changed.valid) << ownership_changed.reason_code;
  const std::string cpu_a = serialize_llm_kv_cpu_execution_identity(
      workload_a, ownership_a);
  const std::string cpu_b = serialize_llm_kv_cpu_execution_identity(
      workload_b, ownership_b);
  const std::string cpu_changed = serialize_llm_kv_cpu_execution_identity(
      workload_a, ownership_changed);
  EXPECT_EQ(cpu_a, cpu_b);
  EXPECT_NE(cpu_a, cpu_changed);
  EXPECT_EQ(cpu_a.rfind(
                Constants::LLM_PAGED_CPU_EXECUTION_IDENTITY_VERSION, 0),
            0u);
  expect_length_prefixed_identity_field(
      cpu_a, "workload_identity", workload_a.identity);
  expect_length_prefixed_identity_field(
      cpu_a, "ownership_identity", ownership_a.identity);
  const LlmKvCpuOwnershipPlan cross_layout_ownership =
      build_llm_paged_decode_kv_cpu_ownership_plan(layout_changed, 5);
  ASSERT_TRUE(cross_layout_ownership.valid)
      << cross_layout_ownership.reason_code;
  EXPECT_TRUE(serialize_llm_kv_cpu_execution_identity(
                  workload_a, cross_layout_ownership)
                  .empty());
  LlmKvCpuOwnershipPlan missing_ownership_identity = ownership_a;
  missing_ownership_identity.identity.clear();
  EXPECT_TRUE(serialize_llm_kv_cpu_execution_identity(
                  workload_a, missing_ownership_identity)
                  .empty());

  const LlmKvMetalSegmentPlan segments_a =
      build_llm_kv_metal_segment_plan(layout_a, {1024, 16});
  const LlmKvMetalSegmentPlan segments_b =
      build_llm_kv_metal_segment_plan(layout_b, {1024, 16});
  const LlmKvMetalSegmentPlan segments_changed =
      build_llm_kv_metal_segment_plan(layout_a, {1536, 16});
  ASSERT_TRUE(segments_a.valid) << segments_a.reason_code;
  ASSERT_TRUE(segments_b.valid) << segments_b.reason_code;
  ASSERT_TRUE(segments_changed.valid) << segments_changed.reason_code;
  const std::string metal_a = serialize_llm_kv_metal_execution_identity(
      workload_a, segments_a);
  const std::string metal_b = serialize_llm_kv_metal_execution_identity(
      workload_b, segments_b);
  const std::string metal_changed = serialize_llm_kv_metal_execution_identity(
      workload_a, segments_changed);
  EXPECT_EQ(metal_a, metal_b);
  EXPECT_NE(metal_a, metal_changed);
  EXPECT_EQ(metal_a.rfind(
                Constants::LLM_PAGED_METAL_EXECUTION_IDENTITY_VERSION, 0),
            0u);
  expect_length_prefixed_identity_field(
      metal_a, "workload_identity", workload_a.identity);
  expect_length_prefixed_identity_field(
      metal_a, "segment_identity", segments_a.identity);
  const LlmKvMetalSegmentPlan cross_layout_segments =
      build_llm_kv_metal_segment_plan(layout_changed, {1024, 16});
  ASSERT_TRUE(cross_layout_segments.valid)
      << cross_layout_segments.reason_code;
  EXPECT_TRUE(serialize_llm_kv_metal_execution_identity(
                  workload_a, cross_layout_segments)
                  .empty());
  LlmPagedDecodeWorkloadPlan missing_workload_identity = workload_a;
  missing_workload_identity.identity.clear();
  EXPECT_TRUE(serialize_llm_kv_metal_execution_identity(
                  missing_workload_identity, segments_a)
                  .empty());
}

namespace {

LlmPrefillPlanRequest exact_prefill_request(
    size_t prompt_tokens = 5, size_t query_tile_tokens = 2,
    size_t block_tokens = 0) {
  // Paired K/V bytes across both layers are 128 bytes per prompt token.
  return {1024, prompt_tokens, query_tile_tokens, 2, 2, 4, 8, 32,
          block_tokens};
}

LlmGeometryRequest integrated_prefill_geometry_request(
    LlmKvLayout layout, size_t prompt_tokens = 5,
    size_t query_tile_tokens = 2, size_t block_tokens = 0) {
  LlmGeometryRequest request;
  request.active_weight_bytes = 1024;
  request.layer_count = 2;
  request.query_head_count = 4;
  request.kv_head_count = 2;
  request.head_dimension = 8;
  request.kv_element_bytes = 2;
  request.batch_size = 2;
  request.kv_block_tokens = block_tokens;
  request.phase = LlmPhase::Prefill;
  request.kv_layout = layout;
  request.prompt_tokens = prompt_tokens;
  request.attention_query_tile_tokens = query_tile_tokens;
  return request;
}

struct IndependentPrefillCounts {
  size_t tile_count = 0;
  size_t prefix_token_visits = 0;
  size_t causal_pairs = 0;
  size_t block_count = 0;
  size_t prefix_block_visits = 0;
  size_t semantic_lookups = 0;
  std::vector<size_t> token_data_visits;
  std::vector<size_t> block_data_visits;
  std::vector<size_t> block_lookups;
};

IndependentPrefillCounts enumerate_prefill_counts(
    size_t prompt_tokens, size_t query_tile_tokens,
    size_t block_tokens) {
  IndependentPrefillCounts result;
  result.token_data_visits.assign(prompt_tokens, 1);
  if (block_tokens != 0) {
    result.block_count =
        prompt_tokens / block_tokens +
        (prompt_tokens % block_tokens == 0 ? 0 : 1);
    result.block_data_visits.assign(result.block_count, 0);
    result.block_lookups.assign(result.block_count, 1);
  }

  size_t tile_end = 0;
  while (tile_end < prompt_tokens) {
    tile_end += std::min(query_tile_tokens, prompt_tokens - tile_end);
    ++result.tile_count;
    result.prefix_token_visits += tile_end;
    if (block_tokens != 0) {
      result.prefix_block_visits +=
          tile_end / block_tokens + (tile_end % block_tokens == 0 ? 0 : 1);
    }
    for (size_t token = 0; token < tile_end; ++token) {
      ++result.token_data_visits[token];
    }
    if (block_tokens != 0) {
      for (size_t block = 0; block < result.block_count; ++block) {
        if (block * block_tokens >= tile_end) {
          break;
        }
        result.block_lookups[block] += 2;
      }
    }
  }
  for (size_t token = 0; token < prompt_tokens; ++token) {
    result.causal_pairs += token + 1;
    if (block_tokens != 0) {
      result.block_data_visits[token / block_tokens] +=
          result.token_data_visits[token];
    }
  }
  if (block_tokens != 0) {
    result.semantic_lookups = std::accumulate(
        result.block_lookups.begin(), result.block_lookups.end(),
        size_t{0});
  }
  return result;
}

void expect_ownership_plans_equal(
    const LlmPrefillCpuOwnershipPlan& actual,
    const LlmPrefillCpuOwnershipPlan& expected) {
  EXPECT_EQ(actual.valid, expected.valid);
  EXPECT_EQ(actual.reason_code, expected.reason_code);
  EXPECT_EQ(actual.identity, expected.identity);
  EXPECT_EQ(actual.unit_kind, expected.unit_kind);
  EXPECT_EQ(actual.scenario, expected.scenario);
  EXPECT_EQ(actual.worker_count, expected.worker_count);
  EXPECT_EQ(actual.worker_rotation, expected.worker_rotation);
  EXPECT_EQ(actual.logical_unit_count, expected.logical_unit_count);
  EXPECT_EQ(actual.active_worker_count, expected.active_worker_count);
  EXPECT_EQ(actual.weight_shards_included,
            expected.weight_shards_included);
  EXPECT_EQ(actual.total_weight_shard_bytes,
            expected.total_weight_shard_bytes);
  EXPECT_EQ(actual.total_kv_model_payload_bytes,
            expected.total_kv_model_payload_bytes);
  EXPECT_EQ(actual.total_layout_metadata_lookup_count,
            expected.total_layout_metadata_lookup_count);
  EXPECT_EQ(actual.total_layout_metadata_read_bytes,
            expected.total_layout_metadata_read_bytes);
  EXPECT_EQ(actual.total_kv_accounted_bytes,
            expected.total_kv_accounted_bytes);
  EXPECT_EQ(actual.total_scenario_accounted_bytes,
            expected.total_scenario_accounted_bytes);
  EXPECT_EQ(actual.worker_weight_shard_bytes,
            expected.worker_weight_shard_bytes);
  EXPECT_EQ(actual.worker_kv_model_payload_bytes,
            expected.worker_kv_model_payload_bytes);
  EXPECT_EQ(actual.worker_layout_metadata_lookup_count,
            expected.worker_layout_metadata_lookup_count);
  EXPECT_EQ(actual.worker_layout_metadata_read_bytes,
            expected.worker_layout_metadata_read_bytes);
  EXPECT_EQ(actual.worker_scenario_accounted_bytes,
            expected.worker_scenario_accounted_bytes);
  ASSERT_EQ(actual.assignments.size(), expected.assignments.size());
  for (size_t index = 0; index < actual.assignments.size(); ++index) {
    const LlmPrefillCpuAssignment& lhs = actual.assignments[index];
    const LlmPrefillCpuAssignment& rhs = expected.assignments[index];
    EXPECT_EQ(lhs.range_rank, rhs.range_rank);
    EXPECT_EQ(lhs.worker_index, rhs.worker_index);
    EXPECT_EQ(lhs.first_unit, rhs.first_unit);
    EXPECT_EQ(lhs.unit_count, rhs.unit_count);
    EXPECT_EQ(lhs.kv_cost.model_payload_bytes,
              rhs.kv_cost.model_payload_bytes);
    EXPECT_EQ(lhs.kv_cost.layout_metadata_lookup_count,
              rhs.kv_cost.layout_metadata_lookup_count);
    EXPECT_EQ(lhs.kv_cost.accounted_bytes,
              rhs.kv_cost.accounted_bytes);
  }
}

void expect_kv_ownership_exact_union(
    const LlmPrefillPlan& prefill,
    LlmPrefillPartitionUnitKind unit_kind, size_t worker_count,
    size_t expected_model_payload_bytes,
    size_t expected_layout_metadata_lookups) {
  const LlmPrefillCpuOwnershipPlan ownership =
      build_llm_prefill_cpu_ownership_plan(
          prefill, {unit_kind, LlmScenario::KvOnly, worker_count, 0, {}});
  ASSERT_TRUE(ownership.valid) << ownership.reason_code;
  ASSERT_EQ(ownership.assignments.size(), worker_count);

  size_t cursor = 0;
  for (const LlmPrefillCpuAssignment& assignment : ownership.assignments) {
    EXPECT_EQ(assignment.first_unit, cursor);
    EXPECT_GT(assignment.unit_count, 0u);
    cursor += assignment.unit_count;
  }
  const size_t expected_unit_count =
      unit_kind == LlmPrefillPartitionUnitKind::ContiguousToken
          ? prefill.prompt_tokens
          : prefill.blocks_per_sequence;
  EXPECT_EQ(cursor, expected_unit_count);
  EXPECT_EQ(ownership.total_kv_model_payload_bytes,
            expected_model_payload_bytes);
  EXPECT_EQ(ownership.total_layout_metadata_lookup_count,
            expected_layout_metadata_lookups);
  EXPECT_EQ(ownership.total_kv_model_payload_bytes,
            std::accumulate(
                ownership.worker_kv_model_payload_bytes.begin(),
                ownership.worker_kv_model_payload_bytes.end(), size_t{0}));
  EXPECT_EQ(ownership.total_layout_metadata_lookup_count,
            std::accumulate(
                ownership.worker_layout_metadata_lookup_count.begin(),
                ownership.worker_layout_metadata_lookup_count.end(),
                size_t{0}));
  EXPECT_EQ(ownership.total_scenario_accounted_bytes,
            std::accumulate(
                ownership.worker_scenario_accounted_bytes.begin(),
                ownership.worker_scenario_accounted_bytes.end(), size_t{0}));
}

constexpr uint64_t kIndependentPrefillPhaseDomain =
    0x50524546494C4C31ULL;
constexpr uint64_t kIndependentPrefillOperationMultiplier =
    0x9E3779B97F4A7C15ULL;
constexpr uint64_t kIndependentPrefillLayerMultiplier =
    0xBF58476D1CE4E5B9ULL;
constexpr uint64_t kIndependentPrefillBatchMultiplier =
    0x94D049BB133111EBULL;
constexpr uint64_t kIndependentPrefillWordMultiplier =
    0xD6E8FEB86659FD93ULL;
constexpr uint64_t kIndependentPrefillKDomain =
    0x4B4B4B4B4B4B4B4BULL;
constexpr uint64_t kIndependentPrefillVDomain =
    0x5656565656565656ULL;

uint64_t independent_prefill_affine_word(
    uint64_t scenario_seed, uint64_t operation_ordinal,
    uint64_t layer_index, uint64_t batch_sequence_index,
    LlmPrefillKvDomain domain, uint64_t logical_word_index) {
  const uint64_t domain_term =
      domain == LlmPrefillKvDomain::K ? kIndependentPrefillKDomain
                                     : kIndependentPrefillVDomain;
  return scenario_seed + kIndependentPrefillPhaseDomain +
         kIndependentPrefillOperationMultiplier *
             (operation_ordinal + 1) +
         kIndependentPrefillLayerMultiplier * (layer_index + 1) +
         kIndependentPrefillBatchMultiplier *
             (batch_sequence_index + 1) +
         domain_term +
         kIndependentPrefillWordMultiplier * (logical_word_index + 1);
}

LlmPrefillAffine64Checksum enumerate_prefill_task_checksum(
    uint64_t scenario_seed, size_t operation_count,
    uint64_t layer_index, uint64_t batch_sequence_index,
    LlmPrefillKvDomain domain, size_t first_logical_word,
    size_t logical_word_count) {
  LlmPrefillAffine64Checksum checksum;
  for (size_t operation = 0; operation < operation_count; ++operation) {
    for (size_t offset = 0; offset < logical_word_count; ++offset) {
      const size_t logical_word = first_logical_word + offset;
      const uint64_t value = independent_prefill_affine_word(
          scenario_seed, operation, layer_index, batch_sequence_index,
          domain, logical_word);
      ++checksum.exact_word_count;
      if ((logical_word & 1U) == 0) {
        ++checksum.even_logical_word_count;
        checksum.even_logical_word_sum += value;
      } else {
        ++checksum.odd_logical_word_count;
        checksum.odd_logical_word_sum += value;
      }
    }
  }
  checksum.valid = true;
  checksum.reason_code = LlmPrefillReason::VALID;
  return checksum;
}

}  // namespace

TEST(LlmMemoryWorkPlanTest,
     PrefillProductionGoldenPayloadAndAttentionMathAreExact) {
  const LlmPrefillPlan plan =
      resolve_llm_prefill_plan(exact_prefill_request());
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  EXPECT_EQ(plan.full_query_tile_count, 2u);
  EXPECT_EQ(plan.final_query_tile_tokens, 1u);
  EXPECT_EQ(plan.tile_count, 3u);
  EXPECT_EQ(plan.attention_prefix_token_visits_per_sequence, 11u);
  EXPECT_EQ(plan.causal_token_pairs_per_sequence, 15u);
  EXPECT_EQ(plan.logical_attention_pairs, 240u);
  EXPECT_EQ(plan.logical_attention_fma_terms, 1920u);
  EXPECT_EQ(plan.kv_bytes_per_token, 128u);
  EXPECT_EQ(plan.weight_read_bytes_per_work_unit, 1024u);
  EXPECT_EQ(plan.kv_read_bytes_per_work_unit, 2816u);
  EXPECT_EQ(plan.kv_write_bytes_per_work_unit, 1280u);
  EXPECT_EQ(plan.kv_only_payload_bytes_per_work_unit, 4096u);
  EXPECT_EQ(plan.mixed_payload_bytes_per_work_unit, 5120u);
}

TEST(LlmMemoryWorkPlanTest,
     PrefillBoundaryTilesAndRemaindersUseExactClosedForms) {
  const LlmPrefillPlan token_tiles =
      resolve_llm_prefill_plan(exact_prefill_request(5, 1));
  const LlmPrefillPlan whole_prompt_tile =
      resolve_llm_prefill_plan(exact_prefill_request(5, 5));
  const LlmPrefillPlan remainder_tiles =
      resolve_llm_prefill_plan(exact_prefill_request(7, 3));
  ASSERT_TRUE(token_tiles.valid) << token_tiles.reason_code;
  ASSERT_TRUE(whole_prompt_tile.valid) << whole_prompt_tile.reason_code;
  ASSERT_TRUE(remainder_tiles.valid) << remainder_tiles.reason_code;

  EXPECT_EQ(token_tiles.tile_count, 5u);
  EXPECT_EQ(token_tiles.attention_prefix_token_visits_per_sequence, 15u);
  EXPECT_EQ(token_tiles.weight_read_bytes_per_work_unit, 1024u);
  EXPECT_EQ(whole_prompt_tile.tile_count, 1u);
  EXPECT_EQ(whole_prompt_tile.attention_prefix_token_visits_per_sequence, 5u);
  EXPECT_EQ(whole_prompt_tile.weight_read_bytes_per_work_unit, 1024u);
  EXPECT_EQ(remainder_tiles.full_query_tile_count, 2u);
  EXPECT_EQ(remainder_tiles.final_query_tile_tokens, 1u);
  EXPECT_EQ(remainder_tiles.tile_count, 3u);
  EXPECT_EQ(remainder_tiles.attention_prefix_token_visits_per_sequence, 16u);
  EXPECT_EQ(remainder_tiles.causal_token_pairs_per_sequence, 28u);

  EXPECT_EQ(resolve_llm_prefill_plan(exact_prefill_request(0, 1)).reason_code,
            LlmPrefillReason::PROMPT_TOKENS_ZERO);
  EXPECT_EQ(resolve_llm_prefill_plan(exact_prefill_request(5, 0)).reason_code,
            LlmPrefillReason::QUERY_TILE_TOKENS_ZERO);
  EXPECT_EQ(resolve_llm_prefill_plan(exact_prefill_request(5, 6)).reason_code,
            LlmPrefillReason::QUERY_TILE_TOKENS_EXCEEDS_PROMPT);
}

TEST(LlmMemoryWorkPlanTest,
     PrefillPagedClosedFormsAndTerminalBlockBytesAreExact) {
  const LlmPrefillPlan first =
      resolve_llm_prefill_plan(exact_prefill_request(5, 2, 2));
  const LlmPrefillPlan second =
      resolve_llm_prefill_plan(exact_prefill_request(7, 3, 2));
  const LlmPrefillPlan terminal =
      resolve_llm_prefill_plan(exact_prefill_request(6, 2, 4));
  ASSERT_TRUE(first.valid) << first.reason_code;
  ASSERT_TRUE(second.valid) << second.reason_code;
  ASSERT_TRUE(terminal.valid) << terminal.reason_code;

  EXPECT_EQ(first.blocks_per_sequence, 3u);
  EXPECT_EQ(first.prefix_block_visits_per_sequence, 6u);
  EXPECT_EQ(first.layout_metadata_lookups_per_layer_sequence, 15u);
  EXPECT_EQ(second.blocks_per_sequence, 4u);
  EXPECT_EQ(second.prefix_block_visits_per_sequence, 9u);
  EXPECT_EQ(second.layout_metadata_lookups_per_layer_sequence, 22u);

  const LlmPrefillUnitRangeCost terminal_block =
      calculate_llm_prefill_paged_block_cost(terminal, 1);
  ASSERT_TRUE(terminal_block.valid) << terminal_block.reason_code;
  EXPECT_EQ(terminal_block.valid_token_count, 2u);
  EXPECT_EQ(terminal_block.data_visit_count, 4u);
  EXPECT_EQ(terminal_block.model_payload_bytes, 256u);
  EXPECT_EQ(terminal_block.layout_metadata_lookup_count, 3u);
  EXPECT_EQ(terminal_block.layout_metadata_read_bytes, 12u);
  EXPECT_EQ(terminal_block.accounted_bytes, 268u);
}

TEST(LlmMemoryWorkPlanTest,
     PrefillSmallDomainMatchesIndependentEnumeration) {
  for (size_t prompt_tokens = 1; prompt_tokens <= 12; ++prompt_tokens) {
    for (size_t query_tile_tokens = 1;
         query_tile_tokens <= prompt_tokens; ++query_tile_tokens) {
      for (size_t block_tokens = 1; block_tokens <= 8; ++block_tokens) {
        SCOPED_TRACE(::testing::Message()
                     << "P=" << prompt_tokens << " Q="
                     << query_tile_tokens << " G=" << block_tokens);
        const IndependentPrefillCounts expected =
            enumerate_prefill_counts(prompt_tokens, query_tile_tokens,
                                     block_tokens);
        const LlmPrefillPlan plan = resolve_llm_prefill_plan(
            exact_prefill_request(prompt_tokens, query_tile_tokens,
                                  block_tokens));
        ASSERT_TRUE(plan.valid) << plan.reason_code;
        EXPECT_EQ(plan.tile_count, expected.tile_count);
        EXPECT_EQ(plan.attention_prefix_token_visits_per_sequence,
                  expected.prefix_token_visits);
        EXPECT_EQ(plan.causal_token_pairs_per_sequence,
                  expected.causal_pairs);
        EXPECT_EQ(plan.blocks_per_sequence, expected.block_count);
        EXPECT_EQ(plan.prefix_block_visits_per_sequence,
                  expected.prefix_block_visits);
        EXPECT_EQ(plan.layout_metadata_lookups_per_layer_sequence,
                  expected.semantic_lookups);

        for (size_t token = 0; token < prompt_tokens; ++token) {
          const LlmPrefillUnitRangeCost cost =
              calculate_llm_prefill_contiguous_token_cost(plan, token);
          ASSERT_TRUE(cost.valid) << cost.reason_code;
          EXPECT_EQ(cost.data_visit_count,
                    expected.token_data_visits[token]);
          EXPECT_EQ(cost.model_payload_bytes,
                    expected.token_data_visits[token] *
                        plan.kv_record_bytes_per_layer);
        }
        for (size_t block = 0; block < expected.block_count; ++block) {
          const LlmPrefillUnitRangeCost cost =
              calculate_llm_prefill_paged_block_cost(plan, block);
          ASSERT_TRUE(cost.valid) << cost.reason_code;
          EXPECT_EQ(cost.data_visit_count,
                    expected.block_data_visits[block]);
          EXPECT_EQ(cost.layout_metadata_lookup_count,
                    expected.block_lookups[block]);
        }

        const size_t expected_model_payload_bytes =
            std::accumulate(expected.token_data_visits.begin(),
                            expected.token_data_visits.end(), size_t{0}) *
            plan.kv_record_bytes_per_layer;
        if (block_tokens == 1) {
          for (size_t workers = 1;
               workers <= std::min(prompt_tokens, size_t{4}); ++workers) {
            expect_kv_ownership_exact_union(
                plan, LlmPrefillPartitionUnitKind::ContiguousToken,
                workers, expected_model_payload_bytes, 0);
          }
        }
        for (size_t workers = 1;
             workers <= std::min(expected.block_count, size_t{4});
             ++workers) {
          expect_kv_ownership_exact_union(
              plan, LlmPrefillPartitionUnitKind::PagedBlock, workers,
              expected_model_payload_bytes, expected.semantic_lookups);
        }
      }
    }
  }
}

TEST(LlmMemoryWorkPlanTest,
     PrefillCheckedHelpersPreserveOutputOnFailureAndRejectOverflow) {
  size_t output = 99;
  EXPECT_FALSE(checked_llm_prefill_ceil_divide(1, 0, output));
  EXPECT_EQ(output, 99u);
  EXPECT_TRUE(checked_llm_prefill_ceil_divide(
      std::numeric_limits<size_t>::max(), 2, output));
  EXPECT_EQ(output, std::numeric_limits<size_t>::max() / 2 + 1);

  output = 99;
  EXPECT_TRUE(checked_llm_prefill_triangular(3, output));
  EXPECT_EQ(output, 6u);
  output = 99;
  EXPECT_FALSE(checked_llm_prefill_triangular(
      std::numeric_limits<size_t>::max(), output));
  EXPECT_EQ(output, 99u);

  output = 99;
  EXPECT_TRUE(checked_llm_prefill_floor_sum(4, 5, 3, 2, output));
  EXPECT_EQ(output, 4u);
  output = 99;
  EXPECT_FALSE(checked_llm_prefill_floor_sum(4, 0, 3, 2, output));
  EXPECT_EQ(output, 99u);
  output = 99;
  EXPECT_FALSE(checked_llm_prefill_floor_sum(
      std::numeric_limits<size_t>::max(), 1,
      std::numeric_limits<size_t>::max(),
      std::numeric_limits<size_t>::max(), output));
  EXPECT_EQ(output, 99u);
}

TEST(LlmMemoryWorkPlanTest,
     PrefillContiguousTokenAndPagedBlockRangeCostsAreExact) {
  const LlmPrefillPlan plan =
      resolve_llm_prefill_plan(exact_prefill_request(5, 2, 2));
  ASSERT_TRUE(plan.valid) << plan.reason_code;

  const std::array<size_t, 5> expected_token_visits = {4, 4, 3, 3, 2};
  for (size_t token = 0; token < expected_token_visits.size(); ++token) {
    const LlmPrefillUnitRangeCost cost =
        calculate_llm_prefill_contiguous_token_cost(plan, token);
    ASSERT_TRUE(cost.valid) << cost.reason_code;
    EXPECT_EQ(cost.valid_token_count, 1u);
    EXPECT_EQ(cost.data_visit_count, expected_token_visits[token]);
    EXPECT_EQ(cost.model_payload_bytes,
              expected_token_visits[token] * 64);
    EXPECT_EQ(cost.layout_metadata_read_bytes, 0u);
    EXPECT_EQ(cost.accounted_bytes, cost.model_payload_bytes);
  }
  const LlmPrefillUnitRangeCost token_range =
      calculate_llm_prefill_contiguous_token_range_cost(plan, 1, 3);
  ASSERT_TRUE(token_range.valid) << token_range.reason_code;
  EXPECT_EQ(token_range.data_visit_count, 10u);
  EXPECT_EQ(token_range.model_payload_bytes, 640u);

  constexpr std::array<size_t, 3> kBlockDataVisits = {8, 6, 2};
  constexpr std::array<size_t, 3> kBlockLookups = {7, 5, 3};
  constexpr std::array<size_t, 3> kBlockAccountedBytes = {540, 404, 140};
  for (size_t block = 0; block < kBlockDataVisits.size(); ++block) {
    const LlmPrefillUnitRangeCost cost =
        calculate_llm_prefill_paged_block_cost(plan, block);
    ASSERT_TRUE(cost.valid) << cost.reason_code;
    EXPECT_EQ(cost.data_visit_count, kBlockDataVisits[block]);
    EXPECT_EQ(cost.layout_metadata_lookup_count, kBlockLookups[block]);
    EXPECT_EQ(cost.accounted_bytes, kBlockAccountedBytes[block]);
  }
  const LlmPrefillUnitRangeCost all_blocks =
      calculate_llm_prefill_paged_block_range_cost(plan, 0, 3);
  ASSERT_TRUE(all_blocks.valid) << all_blocks.reason_code;
  EXPECT_EQ(all_blocks.valid_token_count, 5u);
  EXPECT_EQ(all_blocks.data_visit_count, 16u);
  EXPECT_EQ(all_blocks.model_payload_bytes, 1024u);
  EXPECT_EQ(all_blocks.layout_metadata_lookup_count, 15u);
  EXPECT_EQ(all_blocks.layout_metadata_read_bytes, 60u);
  EXPECT_EQ(all_blocks.accounted_bytes, 1084u);

  EXPECT_EQ(calculate_llm_prefill_contiguous_token_cost(plan, 5).reason_code,
            LlmPrefillReason::INVALID_LOGICAL_RANGE);
  EXPECT_EQ(calculate_llm_prefill_paged_block_cost(plan, 3).reason_code,
            LlmPrefillReason::INVALID_LOGICAL_RANGE);
}

TEST(LlmMemoryWorkPlanTest,
     PrefillCpuOwnershipIsDeterministicUsesLowerTieAndCoversExactUnion) {
  const LlmPrefillPlan equal_tokens =
      resolve_llm_prefill_plan(exact_prefill_request(3, 3));
  ASSERT_TRUE(equal_tokens.valid) << equal_tokens.reason_code;
  const LlmPrefillCpuOwnershipPlan tie =
      build_llm_prefill_cpu_ownership_plan(
          equal_tokens,
          {LlmPrefillPartitionUnitKind::ContiguousToken,
           LlmScenario::KvOnly, 2, 1, {}});
  ASSERT_TRUE(tie.valid) << tie.reason_code;
  ASSERT_EQ(tie.assignments.size(), 2u);
  EXPECT_EQ(tie.assignments[0].worker_index, 1u);
  EXPECT_EQ(tie.assignments[0].first_unit, 0u);
  EXPECT_EQ(tie.assignments[0].unit_count, 1u);
  EXPECT_EQ(tie.assignments[1].worker_index, 0u);
  EXPECT_EQ(tie.assignments[1].first_unit, 1u);
  EXPECT_EQ(tie.assignments[1].unit_count, 2u);

  const LlmPrefillPlan paged =
      resolve_llm_prefill_plan(exact_prefill_request(7, 3, 2));
  ASSERT_TRUE(paged.valid) << paged.reason_code;
  const LlmPrefillCpuOwnershipRequest request{
      LlmPrefillPartitionUnitKind::PagedBlock, LlmScenario::Mixed,
      3, 1, {341, 341, 342}};
  const LlmPrefillCpuOwnershipPlan first =
      build_llm_prefill_cpu_ownership_plan(paged, request);
  const LlmPrefillCpuOwnershipPlan second =
      build_llm_prefill_cpu_ownership_plan(paged, request);
  ASSERT_TRUE(first.valid) << first.reason_code;
  ASSERT_TRUE(second.valid) << second.reason_code;
  expect_ownership_plans_equal(first, second);
  EXPECT_EQ(first.identity.rfind(LlmPrefillVersion::CPU_PARTITION, 0), 0u);
  EXPECT_NE(first.identity.find("|unit_kind=paged_block"),
            std::string::npos);
  EXPECT_NE(first.identity.find("|scenario=mixed"), std::string::npos);
  EXPECT_NE(first.identity.find("|worker_rotation=1"), std::string::npos);
  EXPECT_NE(first.identity.find("|assignment_count=3"), std::string::npos);

  size_t cursor = 0;
  for (size_t rank = 0; rank < first.assignments.size(); ++rank) {
    const LlmPrefillCpuAssignment& assignment = first.assignments[rank];
    EXPECT_EQ(assignment.range_rank, rank);
    EXPECT_EQ(assignment.first_unit, cursor);
    EXPECT_GT(assignment.unit_count, 0u);
    cursor += assignment.unit_count;
  }
  EXPECT_EQ(cursor, paged.blocks_per_sequence);
  EXPECT_EQ(first.active_worker_count, 3u);
  EXPECT_EQ(first.total_weight_shard_bytes, 1024u);
  EXPECT_EQ(first.total_kv_model_payload_bytes,
            std::accumulate(first.worker_kv_model_payload_bytes.begin(),
                            first.worker_kv_model_payload_bytes.end(),
                            size_t{0}));
  EXPECT_EQ(first.total_layout_metadata_lookup_count,
            std::accumulate(
                first.worker_layout_metadata_lookup_count.begin(),
                first.worker_layout_metadata_lookup_count.end(), size_t{0}));
  EXPECT_EQ(first.total_scenario_accounted_bytes,
            std::accumulate(first.worker_scenario_accounted_bytes.begin(),
                            first.worker_scenario_accounted_bytes.end(),
                            size_t{0}));
  const auto worker_range = std::minmax_element(
      first.worker_scenario_accounted_bytes.begin(),
      first.worker_scenario_accounted_bytes.end());
  EXPECT_EQ(first.minimum_worker_accounted_bytes, *worker_range.first);
  EXPECT_EQ(first.maximum_worker_accounted_bytes, *worker_range.second);
  EXPECT_EQ(first.worker_accounted_imbalance_bytes,
            *worker_range.second - *worker_range.first);

  LlmPrefillCpuOwnershipRequest changed_rotation = request;
  changed_rotation.worker_rotation = 2;
  const LlmPrefillCpuOwnershipPlan rotated =
      build_llm_prefill_cpu_ownership_plan(paged, changed_rotation);
  ASSERT_TRUE(rotated.valid) << rotated.reason_code;
  EXPECT_NE(first.identity, rotated.identity);

  LlmPrefillCpuOwnershipRequest changed_weights = request;
  changed_weights.worker_weight_shard_bytes = {340, 342, 342};
  const LlmPrefillCpuOwnershipPlan reweighted =
      build_llm_prefill_cpu_ownership_plan(paged, changed_weights);
  ASSERT_TRUE(reweighted.valid) << reweighted.reason_code;
  EXPECT_NE(first.identity, reweighted.identity);

  const LlmPrefillCpuOwnershipPlan kv_only =
      build_llm_prefill_cpu_ownership_plan(
          paged, {LlmPrefillPartitionUnitKind::PagedBlock,
                  LlmScenario::KvOnly, 3, 1, {}});
  ASSERT_TRUE(kv_only.valid) << kv_only.reason_code;
  EXPECT_NE(first.identity, kv_only.identity);

  const LlmPrefillPlan changed_geometry =
      resolve_llm_prefill_plan(exact_prefill_request(7, 2, 2));
  const LlmPrefillCpuOwnershipPlan changed_geometry_ownership =
      build_llm_prefill_cpu_ownership_plan(changed_geometry, request);
  ASSERT_TRUE(changed_geometry_ownership.valid)
      << changed_geometry_ownership.reason_code;
  EXPECT_NE(first.identity, changed_geometry_ownership.identity);
}

TEST(LlmMemoryWorkPlanTest,
     PrefillCpuOwnershipFailurePublishesNoPartialEvidence) {
  const LlmPrefillPlan prefill =
      resolve_llm_prefill_plan(exact_prefill_request(1, 1));
  ASSERT_TRUE(prefill.valid) << prefill.reason_code;
  const LlmPrefillCpuOwnershipPlan overflow =
      build_llm_prefill_cpu_ownership_plan(
          prefill,
          {LlmPrefillPartitionUnitKind::ContiguousToken,
           LlmScenario::Mixed, 2, 0,
           {0, std::numeric_limits<size_t>::max()}});

  EXPECT_FALSE(overflow.valid);
  EXPECT_EQ(overflow.reason_code,
            LlmPrefillReason::OWNERSHIP_COUNT_OVERFLOW);
  EXPECT_TRUE(overflow.identity.empty());
  EXPECT_TRUE(overflow.assignments.empty());
  EXPECT_TRUE(overflow.worker_weight_shard_bytes.empty());
  EXPECT_TRUE(overflow.worker_kv_model_payload_bytes.empty());
  EXPECT_TRUE(overflow.worker_layout_metadata_lookup_count.empty());
  EXPECT_TRUE(overflow.worker_layout_metadata_read_bytes.empty());
  EXPECT_TRUE(overflow.worker_scenario_accounted_bytes.empty());
  EXPECT_EQ(overflow.logical_unit_count, 0u);
  EXPECT_EQ(overflow.active_worker_count, 0u);
  EXPECT_EQ(overflow.total_weight_shard_bytes, 0u);
  EXPECT_EQ(overflow.total_kv_accounted_bytes, 0u);
  EXPECT_EQ(overflow.total_scenario_accounted_bytes, 0u);
  EXPECT_EQ(overflow.minimum_worker_accounted_bytes, 0u);
  EXPECT_EQ(overflow.maximum_worker_accounted_bytes, 0u);
  EXPECT_EQ(overflow.worker_accounted_imbalance_bytes, 0u);
}

TEST(LlmMemoryWorkPlanTest,
     PrefillModelPlansFreezeGenericWorkUnitsAndBindAllInputs) {
  LlmMemoryWorkPlan contiguous = build_llm_memory_work_plan(
      work_plan_request(integrated_prefill_geometry_request(
          LlmKvLayout::Contiguous)));
  LlmMemoryWorkPlan paged = build_llm_memory_work_plan(
      work_plan_request(integrated_prefill_geometry_request(
          LlmKvLayout::Paged, 5, 2, 2)));
  LlmMemoryWorkPlan changed_query_tile = build_llm_memory_work_plan(
      work_plan_request(integrated_prefill_geometry_request(
          LlmKvLayout::Contiguous, 5, 5)));
  LlmMemoryWorkPlan changed_prompt = build_llm_memory_work_plan(
      work_plan_request(integrated_prefill_geometry_request(
          LlmKvLayout::Contiguous, 6, 2)));
  ASSERT_TRUE(contiguous.valid) << contiguous.reason_code;
  ASSERT_TRUE(paged.valid) << paged.reason_code;
  ASSERT_TRUE(changed_query_tile.valid) << changed_query_tile.reason_code;
  ASSERT_TRUE(changed_prompt.valid) << changed_prompt.reason_code;

  for (const LlmMemoryWorkPlan* plan :
       {&contiguous, &paged, &changed_query_tile, &changed_prompt}) {
    EXPECT_EQ(plan->phase, LlmPhase::Prefill);
    EXPECT_EQ(plan->work_unit_kind, LlmWorkUnitKind::PrefillOperation);
    EXPECT_FALSE(plan->geometry.decode.has_value());
    ASSERT_TRUE(plan->geometry.prefill.has_value());
    ASSERT_TRUE(plan->prefill_plan.has_value());
    EXPECT_TRUE(plan->prefill_plan->valid);
    EXPECT_EQ(plan->component_identities.logical_profile_version,
              Constants::LLM_PREFILL_LOGICAL_PROFILE_VERSION);
    EXPECT_EQ(plan->component_identities.schedule_version,
              LlmPrefillVersion::OWNER_LOCAL_SCHEDULE);
    EXPECT_EQ(plan->component_identities.write_pattern_version,
              LlmPrefillVersion::WRITE_PATTERN);
    EXPECT_EQ(plan->component_identities.checksum_pattern_version,
              plan->kv_layout == LlmKvLayout::Paged
                  ? LlmPrefillVersion::PAGED_CHECKSUM_ORACLE
                  : LlmPrefillVersion::CHECKSUM_ORACLE);
  }
  EXPECT_EQ(contiguous.methodology_version,
            "llm-memory-v1-cpu-prefill-contiguous");
  EXPECT_EQ(paged.methodology_version,
            "llm-memory-v1-cpu-prefill-paged");
  EXPECT_FALSE(contiguous.component_identities.permutation_version.has_value());
  ASSERT_TRUE(paged.component_identities.permutation_version.has_value());
  EXPECT_EQ(*paged.component_identities.permutation_version,
            Constants::LLM_KV_BLOCK_PERMUTATION_VERSION);
  EXPECT_FALSE(cpu_execution_plan(contiguous).paged.has_value());
  ASSERT_TRUE(cpu_execution_plan(contiguous).prefill.has_value());
  ASSERT_TRUE(cpu_execution_plan(paged).paged.has_value());
  ASSERT_TRUE(cpu_execution_plan(paged).prefill.has_value());
  EXPECT_EQ(contiguous.component_identities.backend_executor_version,
            Constants::LLM_PREFILL_CPU_EXECUTOR_VERSION);
  EXPECT_EQ(contiguous.component_identities.resource_abi_version,
            Constants::LLM_PREFILL_DESCRIPTOR_ABI_VERSION);
  EXPECT_EQ(paged.component_identities.backend_executor_version,
            Constants::LLM_PREFILL_PAGED_CPU_EXECUTOR_VERSION);
  EXPECT_EQ(paged.component_identities.resource_abi_version,
            Constants::LLM_PREFILL_PAGED_DESCRIPTOR_ABI_VERSION);
  EXPECT_EQ(paged_cpu_execution_plan(paged).execution_identity,
            cpu_execution_plan(paged).prefill->identity);
  EXPECT_TRUE(paged_cpu_execution_plan(paged).block_table_read_only);
  EXPECT_TRUE(validate_llm_prefill_cpu_execution_evidence(paged));
  EXPECT_GT(paged.memory_budget.request.requested_block_table_mapping_bytes,
            0u);
  EXPECT_EQ(contiguous.geometry.prefill->prompt_tokens, 5u);
  EXPECT_EQ(contiguous.geometry.prefill->attention_query_tile_tokens, 2u);
  EXPECT_EQ(contiguous.geometry.prefill->tile_count, 3u);
  EXPECT_EQ(
      contiguous.geometry.prefill->attention_prefix_token_visits_per_sequence,
      11u);
  EXPECT_EQ(
      paged.geometry.prefill->paged_prefix_block_visits_per_sequence, 6u);
  EXPECT_NE(contiguous.plan_identity, paged.plan_identity);
  EXPECT_NE(contiguous.plan_identity, changed_query_tile.plan_identity);
  EXPECT_NE(contiguous.plan_identity, changed_prompt.plan_identity);

  const LlmScenarioLimits weights = calculate_llm_scenario_limits(
      paged.geometry, LlmScenario::WeightsOnly);
  const LlmScenarioLimits kv = calculate_llm_scenario_limits(
      paged.geometry, LlmScenario::KvOnly);
  const LlmScenarioLimits mixed = calculate_llm_scenario_limits(
      paged.geometry, LlmScenario::Mixed);
  ASSERT_TRUE(weights.valid) << weights.reason_code;
  ASSERT_TRUE(kv.valid) << kv.reason_code;
  ASSERT_TRUE(mixed.valid) << mixed.reason_code;
  EXPECT_EQ(weights.work_unit_kind, LlmWorkUnitKind::PrefillOperation);
  EXPECT_EQ(kv.work_unit_kind, LlmWorkUnitKind::PrefillOperation);
  EXPECT_EQ(mixed.work_unit_kind, LlmWorkUnitKind::PrefillOperation);
  EXPECT_EQ(weights.kv_write_kind, LlmKvWriteKind::None);
  EXPECT_EQ(kv.kv_write_kind, LlmKvWriteKind::FullPromptPopulation);
  EXPECT_EQ(mixed.kv_write_kind, LlmKvWriteKind::FullPromptPopulation);
  EXPECT_EQ(weights.accounted_bytes_per_work_unit, 1024u);
  EXPECT_EQ(kv.effective_model_payload_bytes_per_work_unit, 4096u);
  EXPECT_EQ(kv.layout_metadata_lookup_count_per_work_unit, 60u);
  EXPECT_EQ(kv.layout_metadata_read_bytes_per_work_unit, 240u);
  EXPECT_EQ(kv.accounted_bytes_per_work_unit, 4336u);
  EXPECT_EQ(mixed.effective_model_payload_bytes_per_work_unit, 5120u);
  EXPECT_EQ(mixed.accounted_bytes_per_work_unit, 5360u);

  const std::array<size_t, kLlmScenarioCount> work_units = {2, 3, 4};
  const LlmFrozenScenarioPlans frozen =
      freeze_llm_scenario_work_plans(paged, work_units, true);
  ASSERT_TRUE(frozen.valid) << frozen.reason_code;
  EXPECT_TRUE(frozen.explicit_iterations);
  EXPECT_EQ(frozen.model_plan_identity, paged.plan_identity);
  EXPECT_FALSE(frozen.plan_identity.empty());
  for (size_t index = 0; index < frozen.scenarios.size(); ++index) {
    const LlmScenarioWorkPlan& scenario = frozen.scenarios[index];
    EXPECT_TRUE(scenario.valid) << scenario.reason_code;
    EXPECT_EQ(scenario.work_unit_kind, LlmWorkUnitKind::PrefillOperation);
    EXPECT_EQ(scenario.work_units, work_units[index]);
    EXPECT_EQ(scenario.model_plan_identity, paged.plan_identity);
  }
  EXPECT_EQ(frozen.scenarios[0].kv_write_kind, LlmKvWriteKind::None);
  EXPECT_EQ(frozen.scenarios[1].kv_write_kind,
            LlmKvWriteKind::FullPromptPopulation);
  EXPECT_EQ(frozen.scenarios[2].kv_write_kind,
            LlmKvWriteKind::FullPromptPopulation);

  const LlmFrozenScenarioPlans changed_frozen =
      freeze_llm_scenario_work_plans(paged, {2, 3, 5}, true);
  ASSERT_TRUE(changed_frozen.valid) << changed_frozen.reason_code;
  EXPECT_NE(frozen.plan_identity, changed_frozen.plan_identity);

  EXPECT_EQ(kv.maximum_work_units_by_guardrail,
            Constants::LLM_MAX_ACCOUNTED_BYTES_PER_TASK /
                kv.accounted_bytes_per_work_unit);
  EXPECT_EQ(kv.effective_maximum_work_units,
            kv.maximum_work_units_by_guardrail);
  const LlmScenarioWorkPlan guardrail_boundary =
      build_llm_scenario_work_plan(
          paged, LlmScenario::KvOnly,
          kv.maximum_work_units_by_guardrail, true);
  ASSERT_TRUE(guardrail_boundary.valid) << guardrail_boundary.reason_code;
  EXPECT_LE(guardrail_boundary.task_accounted_bytes,
            Constants::LLM_MAX_ACCOUNTED_BYTES_PER_TASK);
  const LlmScenarioWorkPlan guardrail_exceeded =
      build_llm_scenario_work_plan(
          paged, LlmScenario::KvOnly,
          kv.maximum_work_units_by_guardrail + 1, true);
  EXPECT_FALSE(guardrail_exceeded.valid);
  EXPECT_EQ(guardrail_exceeded.reason_code,
            LlmWorkPlanReason::TASK_ACCOUNTED_BYTES_CAP_EXCEEDED);
}

TEST(LlmMemoryWorkPlanTest,
     PrefillDescriptorAbiAndScenarioOwnershipSetsAreExact) {
  EXPECT_EQ(sizeof(LlmPrefillLayerDescriptor), 48u);
  EXPECT_EQ(alignof(LlmPrefillLayerDescriptor), 16u);
  EXPECT_EQ(offsetof(LlmPrefillLayerDescriptor, weight_ptr), 0u);
  EXPECT_EQ(offsetof(LlmPrefillLayerDescriptor, weight_bytes), 8u);
  EXPECT_EQ(offsetof(LlmPrefillLayerDescriptor, first_sequence_index), 16u);
  EXPECT_EQ(offsetof(LlmPrefillLayerDescriptor, sequence_count), 24u);
  EXPECT_EQ(offsetof(LlmPrefillLayerDescriptor, layer_index), 32u);
  EXPECT_EQ(offsetof(LlmPrefillLayerDescriptor, reserved_zero), 40u);
  EXPECT_EQ(sizeof(LlmPrefillKvSequenceDescriptor), 80u);
  EXPECT_EQ(alignof(LlmPrefillKvSequenceDescriptor), 16u);
  EXPECT_EQ(offsetof(LlmPrefillKvSequenceDescriptor, k_owned_ptr), 0u);
  EXPECT_EQ(offsetof(LlmPrefillKvSequenceDescriptor, v_owned_ptr), 8u);
  EXPECT_EQ(offsetof(LlmPrefillKvSequenceDescriptor, first_token), 16u);
  EXPECT_EQ(offsetof(LlmPrefillKvSequenceDescriptor, owned_token_count), 24u);
  EXPECT_EQ(offsetof(LlmPrefillKvSequenceDescriptor, prompt_tokens), 32u);
  EXPECT_EQ(
      offsetof(LlmPrefillKvSequenceDescriptor,
               attention_query_tile_tokens),
      40u);
  EXPECT_EQ(offsetof(LlmPrefillKvSequenceDescriptor, record_bytes), 48u);
  EXPECT_EQ(offsetof(LlmPrefillKvSequenceDescriptor, layer_index), 56u);
  EXPECT_EQ(
      offsetof(LlmPrefillKvSequenceDescriptor, batch_sequence_index), 64u);
  EXPECT_EQ(offsetof(LlmPrefillKvSequenceDescriptor, reserved_zero), 72u);
  EXPECT_EQ(sizeof(LlmPagedPrefillLayerDescriptor), 48u);
  EXPECT_EQ(alignof(LlmPagedPrefillLayerDescriptor), 16u);
  EXPECT_EQ(offsetof(LlmPagedPrefillLayerDescriptor, weight_ptr), 0u);
  EXPECT_EQ(offsetof(LlmPagedPrefillLayerDescriptor, weight_bytes), 8u);
  EXPECT_EQ(
      offsetof(LlmPagedPrefillLayerDescriptor, first_assignment_index), 16u);
  EXPECT_EQ(offsetof(LlmPagedPrefillLayerDescriptor, assignment_count), 24u);
  EXPECT_EQ(offsetof(LlmPagedPrefillLayerDescriptor, layer_index), 32u);
  EXPECT_EQ(offsetof(LlmPagedPrefillLayerDescriptor, reserved_zero), 40u);
  EXPECT_EQ(sizeof(LlmPagedPrefillKvAssignmentDescriptor), 112u);
  EXPECT_EQ(alignof(LlmPagedPrefillKvAssignmentDescriptor), 16u);
  EXPECT_EQ(offsetof(LlmPagedPrefillKvAssignmentDescriptor,
                     block_table_row), 0u);
  EXPECT_EQ(offsetof(LlmPagedPrefillKvAssignmentDescriptor, k_layer_pool),
            8u);
  EXPECT_EQ(offsetof(LlmPagedPrefillKvAssignmentDescriptor, v_layer_pool),
            16u);
  EXPECT_EQ(offsetof(LlmPagedPrefillKvAssignmentDescriptor,
                     first_logical_block), 24u);
  EXPECT_EQ(offsetof(LlmPagedPrefillKvAssignmentDescriptor,
                     owned_block_count), 32u);
  EXPECT_EQ(offsetof(LlmPagedPrefillKvAssignmentDescriptor,
                     blocks_per_sequence), 40u);
  EXPECT_EQ(offsetof(LlmPagedPrefillKvAssignmentDescriptor, block_tokens),
            48u);
  EXPECT_EQ(offsetof(LlmPagedPrefillKvAssignmentDescriptor, block_bytes),
            56u);
  EXPECT_EQ(offsetof(LlmPagedPrefillKvAssignmentDescriptor,
                     last_block_valid_bytes), 64u);
  EXPECT_EQ(offsetof(LlmPagedPrefillKvAssignmentDescriptor, prompt_tokens),
            72u);
  EXPECT_EQ(offsetof(LlmPagedPrefillKvAssignmentDescriptor,
                     attention_query_tile_tokens), 80u);
  EXPECT_EQ(offsetof(LlmPagedPrefillKvAssignmentDescriptor, record_bytes),
            88u);
  EXPECT_EQ(offsetof(LlmPagedPrefillKvAssignmentDescriptor, layer_index),
            96u);
  EXPECT_EQ(offsetof(LlmPagedPrefillKvAssignmentDescriptor,
                     batch_sequence_index), 104u);

  const LlmMemoryWorkPlan plan = build_llm_memory_work_plan(
      work_plan_request(integrated_prefill_geometry_request(
                            LlmKvLayout::Contiguous, 7, 3),
                        3, 3));
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  const LlmCpuExecutionPlan& cpu = cpu_execution_plan(plan);
  ASSERT_TRUE(cpu.prefill.has_value());
  const LlmPrefillCpuExecutionPlan& prefill = *cpu.prefill;
  const size_t rows = plan.geometry.layer_count * plan.geometry.batch_size;
  EXPECT_EQ(prefill.sequence_descriptors_per_scenario_per_worker, rows);
  EXPECT_EQ(cpu.sequence_descriptors_per_worker,
            rows * kLlmScenarioCount);
  EXPECT_FALSE(prefill.identity.empty());

  const std::array<size_t, kLlmScenarioCount> expected_scope_counts = {
      plan.geometry.layer_count, rows, rows};
  const std::array<size_t, kLlmScenarioCount> expected_totals = {
      plan.geometry.weight_read_bytes_per_work_unit,
      plan.geometry.kv_only_effective_model_payload_bytes_per_work_unit,
      plan.geometry.mixed_effective_model_payload_bytes_per_work_unit};
  std::array<std::string, kLlmScenarioCount> scenario_identities;
  for (size_t scenario_index = 0; scenario_index < kLlmScenarioCount;
       ++scenario_index) {
    const LlmPrefillCpuScenarioExecutionPlan& scenario =
        prefill.scenarios[scenario_index];
    EXPECT_EQ(static_cast<size_t>(scenario.scenario), scenario_index);
    EXPECT_EQ(scenario.ownership_scopes.size(),
              expected_scope_counts[scenario_index]);
    ASSERT_EQ(scenario.worker_accounted_bytes_per_work_unit.size(),
              cpu.effective_workers);
    EXPECT_EQ(std::accumulate(
                  scenario.worker_accounted_bytes_per_work_unit.begin(),
                  scenario.worker_accounted_bytes_per_work_unit.end(),
                  size_t{0}),
              expected_totals[scenario_index]);
    EXPECT_EQ(scenario.worker_accounted_imbalance_bytes_per_work_unit,
              scenario.maximum_worker_accounted_bytes_per_work_unit -
                  scenario.minimum_worker_accounted_bytes_per_work_unit);
    scenario_identities[scenario_index] = scenario.identity;
    EXPECT_FALSE(scenario.identity.empty());

    size_t scope_weight_bytes = 0;
    for (const LlmPrefillCpuOwnershipPlan& scope :
         scenario.ownership_scopes) {
      EXPECT_EQ(scope.scenario, scenario.scenario);
      scope_weight_bytes += scope.total_weight_shard_bytes;
    }
    const size_t expected_weight_bytes =
        scenario.scenario == LlmScenario::KvOnly
            ? 0
            : plan.geometry.active_weight_bytes_per_work_unit;
    EXPECT_EQ(scope_weight_bytes, expected_weight_bytes);
  }
  EXPECT_NE(scenario_identities[0], scenario_identities[1]);
  EXPECT_NE(scenario_identities[0], scenario_identities[2]);
  EXPECT_NE(scenario_identities[1], scenario_identities[2]);

  for (size_t worker = 0; worker < cpu.effective_workers; ++worker) {
    ASSERT_EQ(cpu.workers[worker].prefill_sequences.size(),
              rows * kLlmScenarioCount);
    for (size_t row = 0; row < rows; ++row) {
      const LlmPrefillKvSequenceRangeTemplate& weights =
          cpu.workers[worker].prefill_sequences[row];
      EXPECT_EQ(weights.owned_token_count, 0u);
      EXPECT_EQ(weights.k_owned.span_bytes, 0u);
      EXPECT_EQ(weights.v_owned.span_bytes, 0u);
    }
  }
  for (size_t scenario_index : {static_cast<size_t>(LlmScenario::KvOnly),
                                static_cast<size_t>(LlmScenario::Mixed)}) {
    for (size_t row = 0; row < rows; ++row) {
      size_t owned_tokens = 0;
      for (const LlmWorkerWorkPlan& worker : cpu.workers) {
        owned_tokens +=
            worker.prefill_sequences[scenario_index * rows + row]
                .owned_token_count;
      }
      EXPECT_EQ(owned_tokens, plan.geometry.prefill->prompt_tokens);
    }
  }
}

TEST(LlmMemoryWorkPlanTest,
     PagedPrefillScenarioMajorBlocksAndMetadataAreWorkerInvariant) {
  LlmMemoryWorkPlan one_worker = build_llm_memory_work_plan(
      work_plan_request(integrated_prefill_geometry_request(
                            LlmKvLayout::Paged, 7, 3, 2),
                        1, 1));
  LlmMemoryWorkPlan three_workers = build_llm_memory_work_plan(
      work_plan_request(integrated_prefill_geometry_request(
                            LlmKvLayout::Paged, 7, 3, 2),
                        3, 3));
  ASSERT_TRUE(one_worker.valid) << one_worker.reason_code;
  ASSERT_TRUE(three_workers.valid) << three_workers.reason_code;
  ASSERT_TRUE(validate_llm_prefill_cpu_execution_evidence(one_worker));
  ASSERT_TRUE(validate_llm_prefill_cpu_execution_evidence(three_workers));

  const LlmCpuExecutionPlan& cpu = cpu_execution_plan(three_workers);
  ASSERT_TRUE(cpu.paged.has_value());
  ASSERT_TRUE(cpu.prefill.has_value());
  EXPECT_EQ(cpu.effective_workers, 3u);
  const size_t rows =
      three_workers.geometry.layer_count * three_workers.geometry.batch_size;
  EXPECT_EQ(rows, 4u);
  EXPECT_EQ(cpu.sequence_descriptors_per_worker,
            rows * kLlmScenarioCount);
  EXPECT_EQ(cpu.descriptor_bytes,
            cpu.total_layer_descriptors *
                    sizeof(LlmPagedPrefillLayerDescriptor) +
                cpu.total_sequence_descriptors *
                    sizeof(LlmPagedPrefillKvAssignmentDescriptor));

  constexpr size_t kExpectedLookups = 88;
  constexpr size_t kExpectedLookupBytes =
      kExpectedLookups * sizeof(uint32_t);
  const std::array<size_t, kLlmScenarioCount> expected_accounted = {
      1024, 5888 + kExpectedLookupBytes,
      6912 + kExpectedLookupBytes};
  for (size_t scenario_index_value = 0;
       scenario_index_value < kLlmScenarioCount;
       ++scenario_index_value) {
    const LlmPrefillCpuScenarioExecutionPlan& scenario =
        cpu.prefill->scenarios[scenario_index_value];
    size_t lookup_count = 0;
    size_t lookup_bytes = 0;
    size_t accounted_bytes = 0;
    for (const LlmPrefillCpuOwnershipPlan& scope :
         scenario.ownership_scopes) {
      EXPECT_EQ(scope.unit_kind,
                LlmPrefillPartitionUnitKind::PagedBlock);
      lookup_count += scope.total_layout_metadata_lookup_count;
      lookup_bytes += scope.total_layout_metadata_read_bytes;
      accounted_bytes += scope.total_scenario_accounted_bytes;
    }
    const bool kv_active =
        scenario.scenario != LlmScenario::WeightsOnly;
    EXPECT_EQ(lookup_count, kv_active ? kExpectedLookups : 0u);
    EXPECT_EQ(lookup_bytes, kv_active ? kExpectedLookupBytes : 0u);
    EXPECT_EQ(accounted_bytes, expected_accounted[scenario_index_value]);
    EXPECT_EQ(std::accumulate(
                  scenario.worker_accounted_bytes_per_work_unit.begin(),
                  scenario.worker_accounted_bytes_per_work_unit.end(),
                  size_t{0}),
              expected_accounted[scenario_index_value]);
  }

  for (const LlmWorkerWorkPlan& worker : cpu.workers) {
    EXPECT_TRUE(worker.prefill_sequences.empty());
    ASSERT_EQ(worker.paged_prefill_assignments.size(),
              rows * kLlmScenarioCount);
    for (size_t row = 0; row < rows; ++row) {
      const LlmPagedPrefillKvAssignmentTemplate& weights =
          worker.paged_prefill_assignments[row];
      EXPECT_EQ(weights.first_logical_block, 0u);
      EXPECT_EQ(weights.block_count, 0u);
    }
  }
  for (size_t scenario_index_value :
       {static_cast<size_t>(LlmScenario::KvOnly),
        static_cast<size_t>(LlmScenario::Mixed)}) {
    for (size_t row = 0; row < rows; ++row) {
      std::vector<std::pair<size_t, size_t>> ranges;
      for (const LlmWorkerWorkPlan& worker : cpu.workers) {
        const LlmPagedPrefillKvAssignmentTemplate& assignment =
            worker.paged_prefill_assignments[
                scenario_index_value * rows + row];
        if (assignment.block_count != 0) {
          ranges.emplace_back(assignment.first_logical_block,
                              assignment.block_count);
        }
      }
      std::sort(ranges.begin(), ranges.end());
      size_t cursor = 0;
      for (const auto& range : ranges) {
        EXPECT_EQ(range.first, cursor);
        cursor += range.second;
      }
      EXPECT_EQ(cursor,
                three_workers.prefill_plan->blocks_per_sequence);
    }
  }

  const auto total_lookups = [](const LlmMemoryWorkPlan& plan) {
    const LlmPrefillCpuScenarioExecutionPlan& kv =
        cpu_execution_plan(plan)
            .prefill->scenarios[static_cast<size_t>(LlmScenario::KvOnly)];
    size_t total = 0;
    for (const LlmPrefillCpuOwnershipPlan& scope : kv.ownership_scopes) {
      total += scope.total_layout_metadata_lookup_count;
    }
    return total;
  };
  EXPECT_EQ(total_lookups(one_worker), kExpectedLookups);
  EXPECT_EQ(total_lookups(three_workers), kExpectedLookups);

  LlmCpuExecutionPlan& mutable_cpu = cpu_execution_plan(three_workers);
  const std::string execution_identity =
      mutable_cpu.paged->execution_identity;
  mutable_cpu.paged->execution_identity += "-tampered";
  EXPECT_FALSE(validate_llm_prefill_cpu_execution_evidence(three_workers));
  mutable_cpu.paged->execution_identity = execution_identity;
  mutable_cpu.paged->layout_identity += "-tampered";
  EXPECT_FALSE(validate_llm_prefill_cpu_execution_evidence(three_workers));
}

TEST(LlmMemoryWorkPlanTest,
     PagedPrefillFinalizationPollsAndPublishesNoPartialTable) {
  size_t stop_checks = 0;
  const LlmMemoryWorkPlan plan = build_llm_memory_work_plan(
      work_plan_request(integrated_prefill_geometry_request(
                            LlmKvLayout::Paged, 7, 3, 2),
                        3, 3),
      [&stop_checks]() {
        ++stop_checks;
        return stop_checks == 2;
      });

  expect_invalid_plan(plan, LlmKvLayoutReason::PREPARATION_INTERRUPTED);
  EXPECT_EQ(stop_checks, 2u);
  EXPECT_FALSE(cpu_execution_plan(plan).paged.has_value());
  EXPECT_FALSE(cpu_execution_plan(plan).prefill.has_value());
}

TEST(LlmMemoryWorkPlanTest,
     PrefillPlannerStorageAccountsExactRetainedIdentityCapacities) {
  const auto expect_exact_storage = [](LlmKvLayout layout) {
    const LlmMemoryWorkPlan plan = build_llm_memory_work_plan(
        work_plan_request(integrated_prefill_geometry_request(
                              layout, 7, 3,
                              layout == LlmKvLayout::Paged ? 2 : 0),
                          3, 3));
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    const LlmCpuExecutionPlan& cpu = cpu_execution_plan(plan);
    ASSERT_TRUE(cpu.prefill.has_value());

    size_t expected =
        plan.weight_layers.capacity() * sizeof(LlmByteRange) +
        cpu.workers.capacity() * sizeof(LlmWorkerWorkPlan);
    for (const LlmWorkerWorkPlan& worker : cpu.workers) {
      expected +=
          worker.layers.capacity() * sizeof(LlmLayerRangeTemplate);
      expected +=
          worker.sequences.capacity() * sizeof(LlmKvSequenceRangeTemplate);
      expected += worker.paged_assignments.capacity() *
                  sizeof(LlmPagedKvAssignmentTemplate);
      expected += worker.prefill_sequences.capacity() *
                  sizeof(LlmPrefillKvSequenceRangeTemplate);
      expected += worker.paged_prefill_assignments.capacity() *
                  sizeof(LlmPagedPrefillKvAssignmentTemplate);
    }
    const auto add_external_string_capacity =
        [&](const std::string& value) {
          const uintptr_t data = reinterpret_cast<uintptr_t>(value.data());
          const uintptr_t object = reinterpret_cast<uintptr_t>(&value);
          if (data < object || data >= object + sizeof(value)) {
            expected += value.capacity() + 1;
          }
        };
    add_external_string_capacity(cpu.prefill->identity);
    for (const LlmPrefillCpuScenarioExecutionPlan& scenario :
         cpu.prefill->scenarios) {
      expected += scenario.ownership_scopes.capacity() *
                  sizeof(LlmPrefillCpuOwnershipPlan);
      expected += scenario.worker_accounted_bytes_per_work_unit.capacity() *
                  sizeof(size_t);
      add_external_string_capacity(scenario.identity);
      for (const LlmPrefillCpuOwnershipPlan& scope :
           scenario.ownership_scopes) {
        expected += scope.assignments.capacity() *
                    sizeof(LlmPrefillCpuAssignment);
        expected += scope.worker_weight_shard_bytes.capacity() *
                    sizeof(size_t);
        expected += scope.worker_kv_model_payload_bytes.capacity() *
                    sizeof(size_t);
        expected += scope.worker_layout_metadata_lookup_count.capacity() *
                    sizeof(size_t);
        expected += scope.worker_layout_metadata_read_bytes.capacity() *
                    sizeof(size_t);
        expected += scope.worker_scenario_accounted_bytes.capacity() *
                    sizeof(size_t);
        add_external_string_capacity(scope.reason_code);
        add_external_string_capacity(scope.identity);
        EXPECT_EQ(scenario.identity.find(scope.identity),
                  std::string::npos);
      }
    }
    EXPECT_EQ(cpu.planner_storage_bytes, expected);
    EXPECT_EQ(plan.memory_budget.request.planner_storage_bytes, expected);
  };

  expect_exact_storage(LlmKvLayout::Contiguous);
  expect_exact_storage(LlmKvLayout::Paged);
}

TEST(LlmMemoryWorkPlanTest,
     PrefillIdentityPreflightOverflowFailsBeforeProportionalAllocation) {
  LlmGeometryRequest geometry;
  geometry.active_weight_bytes = 100000000;
  geometry.layer_count = 100000000;
  geometry.query_head_count = 1;
  geometry.kv_head_count = 1;
  geometry.head_dimension = 1;
  geometry.kv_element_bytes = 1;
  geometry.batch_size = 100000000;
  geometry.phase = LlmPhase::Prefill;
  geometry.kv_layout = LlmKvLayout::Contiguous;
  geometry.prompt_tokens = 1;
  geometry.attention_query_tile_tokens = 1;
  LlmMemoryWorkPlanRequest request = work_plan_request(geometry, 1, 1);
  request.available_memory_bytes = std::numeric_limits<size_t>::max();

  const LlmMemoryWorkPlan plan = build_llm_memory_work_plan(request);
  EXPECT_FALSE(plan.valid);
  EXPECT_EQ(plan.reason_code,
            LlmWorkPlanReason::PLANNER_STORAGE_BYTES_OVERFLOW);
}

TEST(LlmMemoryWorkPlanTest,
     PrefillAffineTaskChecksumMatchesIndependentEnumerationAndFinalOrdinal) {
  constexpr uint64_t kScenarioSeed = 0x0123456789ABCDEFULL;
  constexpr size_t kOperationCount = 2;
  constexpr uint64_t kLayerIndex = 2;
  constexpr uint64_t kBatchIndex = 3;
  constexpr size_t kFirstWord = 1;
  constexpr size_t kWordCount = 5;

  for (const LlmPrefillKvDomain domain :
       {LlmPrefillKvDomain::K, LlmPrefillKvDomain::V}) {
    for (size_t operation = 0; operation < kOperationCount; ++operation) {
      for (size_t offset = 0; offset < kWordCount; ++offset) {
        const size_t logical_word = kFirstWord + offset;
        EXPECT_EQ(llm_prefill_affine64_word(
                      kScenarioSeed, operation, kLayerIndex, kBatchIndex,
                      domain, logical_word),
                  independent_prefill_affine_word(
                      kScenarioSeed, operation, kLayerIndex, kBatchIndex,
                      domain, logical_word));
      }
    }
    const LlmPrefillAffine64Checksum expected =
        enumerate_prefill_task_checksum(
            kScenarioSeed, kOperationCount, kLayerIndex, kBatchIndex,
            domain, kFirstWord, kWordCount);
    const LlmPrefillAffine64Checksum actual =
        calculate_llm_prefill_affine64_task_checksum(
            kScenarioSeed, kOperationCount, kLayerIndex, kBatchIndex,
            domain, kFirstWord, kWordCount);
    ASSERT_TRUE(actual.valid) << actual.reason_code;
    EXPECT_EQ(actual.exact_word_count, expected.exact_word_count);
    EXPECT_EQ(actual.even_logical_word_count,
              expected.even_logical_word_count);
    EXPECT_EQ(actual.odd_logical_word_count,
              expected.odd_logical_word_count);
    EXPECT_EQ(actual.even_logical_word_sum,
              expected.even_logical_word_sum);
    EXPECT_EQ(actual.odd_logical_word_sum,
              expected.odd_logical_word_sum);
  }

  const LlmPrefillAffine64Checksum k_checksum =
      calculate_llm_prefill_affine64_task_checksum(
          kScenarioSeed, kOperationCount, kLayerIndex, kBatchIndex,
          LlmPrefillKvDomain::K, kFirstWord, kWordCount);
  EXPECT_EQ(k_checksum.exact_word_count, 10u);
  EXPECT_EQ(k_checksum.even_logical_word_count, 4u);
  EXPECT_EQ(k_checksum.odd_logical_word_count, 6u);
  EXPECT_EQ(k_checksum.even_logical_word_sum,
            0xDC08129268383AB6ULL);
  EXPECT_EQ(k_checksum.odd_logical_word_sum,
            0xCA0C1BDB9C545811ULL);

  const uint64_t final_word = llm_prefill_affine64_word(
      kScenarioSeed, kOperationCount - 1, kLayerIndex, kBatchIndex,
      LlmPrefillKvDomain::K, kFirstWord);
  EXPECT_EQ(final_word, 0x184BC4108CFF5192ULL);
  EXPECT_NE(final_word,
            llm_prefill_affine64_word(
                kScenarioSeed, 0, kLayerIndex, kBatchIndex,
                LlmPrefillKvDomain::K, kFirstWord));
  for (size_t byte = 0; byte < sizeof(uint64_t); ++byte) {
    EXPECT_EQ(llm_prefill_affine64_byte(
                  kScenarioSeed, kOperationCount - 1, kLayerIndex,
                  kBatchIndex, LlmPrefillKvDomain::K,
                  kFirstWord * sizeof(uint64_t) + byte),
              static_cast<uint8_t>(final_word >> (byte * 8)));
  }

  const LlmPrefillAffine64Checksum zero_operations =
      calculate_llm_prefill_affine64_task_checksum(
          kScenarioSeed, 0, kLayerIndex, kBatchIndex,
          LlmPrefillKvDomain::K, kFirstWord, kWordCount);
  EXPECT_FALSE(zero_operations.valid);
  EXPECT_EQ(zero_operations.reason_code,
            LlmPrefillReason::OPERATION_COUNT_ZERO);
  const LlmPrefillAffine64Checksum overflow =
      calculate_llm_prefill_affine64_task_checksum(
          kScenarioSeed, std::numeric_limits<size_t>::max(),
          kLayerIndex, kBatchIndex, LlmPrefillKvDomain::K, 0, 2);
  EXPECT_FALSE(overflow.valid);
  EXPECT_EQ(overflow.reason_code,
            LlmPrefillReason::AFFINE_WORD_RANGE_OVERFLOW);
  const LlmPrefillAffine64Checksum invalid_domain =
      calculate_llm_prefill_affine64_task_checksum(
          kScenarioSeed, 1, kLayerIndex, kBatchIndex,
          static_cast<LlmPrefillKvDomain>(99), 0, 1);
  EXPECT_FALSE(invalid_domain.valid);
  EXPECT_EQ(invalid_domain.reason_code,
            LlmPrefillReason::AFFINE_DOMAIN_INVALID);
}

TEST(LlmMemoryWorkPlanTest,
     PrefillSemanticTraceWritesBothBlocksBeforePerTileKThenVReads) {
  const LlmPrefillPlan plan =
      resolve_llm_prefill_plan(exact_prefill_request(4, 2, 2));
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  const LlmPrefillSemanticTrace trace =
      build_llm_prefill_semantic_trace(
          plan, {LlmPrefillPartitionUnitKind::PagedBlock, 0, 2, 10});
  ASSERT_TRUE(trace.valid) << trace.reason_code;

  struct ExpectedEvent {
    LlmPrefillSemanticAccess access;
    LlmPrefillKvDomain domain;
    size_t tile_index;
    size_t tile_end;
    size_t block;
    size_t visit_tokens;
  };
  constexpr std::array<ExpectedEvent, 10> kExpected = {{
      {LlmPrefillSemanticAccess::Write, LlmPrefillKvDomain::K, 0, 4, 0, 2},
      {LlmPrefillSemanticAccess::Write, LlmPrefillKvDomain::V, 0, 4, 0, 2},
      {LlmPrefillSemanticAccess::Write, LlmPrefillKvDomain::K, 0, 4, 1, 2},
      {LlmPrefillSemanticAccess::Write, LlmPrefillKvDomain::V, 0, 4, 1, 2},
      {LlmPrefillSemanticAccess::Read, LlmPrefillKvDomain::K, 0, 2, 0, 2},
      {LlmPrefillSemanticAccess::Read, LlmPrefillKvDomain::V, 0, 2, 0, 2},
      {LlmPrefillSemanticAccess::Read, LlmPrefillKvDomain::K, 1, 4, 0, 2},
      {LlmPrefillSemanticAccess::Read, LlmPrefillKvDomain::K, 1, 4, 1, 2},
      {LlmPrefillSemanticAccess::Read, LlmPrefillKvDomain::V, 1, 4, 0, 2},
      {LlmPrefillSemanticAccess::Read, LlmPrefillKvDomain::V, 1, 4, 1, 2},
  }};
  ASSERT_EQ(trace.events.size(), kExpected.size());
  for (size_t index = 0; index < kExpected.size(); ++index) {
    SCOPED_TRACE(index);
    EXPECT_EQ(trace.events[index].access, kExpected[index].access);
    EXPECT_EQ(trace.events[index].domain, kExpected[index].domain);
    EXPECT_EQ(trace.events[index].tile_index,
              kExpected[index].tile_index);
    EXPECT_EQ(trace.events[index].tile_end_token,
              kExpected[index].tile_end);
    EXPECT_EQ(trace.events[index].logical_unit_index,
              kExpected[index].block);
    EXPECT_EQ(trace.events[index].visit_token_count,
              kExpected[index].visit_tokens);
  }
}

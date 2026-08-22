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

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "core/config/constants.h"
#include "llm_memory/llm_kv_layout.h"
#include "llm_memory/llm_work_plan.h"

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

LlmGeometryRequest large_geometry_request() {
  return {4 * kGiB, 32, 32, 8, 128, 2, 8192, 1};
}

LlmGeometryRequest small_geometry_request(size_t batch_size = 1) {
  // K+V across both layers is exactly 128 bytes per visible token.
  return {1024, 2, 4, 2, 8, 2, 3, batch_size};
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
  EXPECT_EQ(actual.requested_data_bytes, expected.requested_data_bytes);
  EXPECT_EQ(actual.committed_data_bytes, expected.committed_data_bytes);
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
  EXPECT_EQ(build_llm_methodology_version(
                LlmMemoryBackend::Cpu, LlmPhase::Decode,
                LlmKvLayout::Contiguous),
            Constants::LLM_CPU_DECODE_CONTIGUOUS_METHODOLOGY_VERSION);
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

  LlmGeometryRequest inactive_geometry = small_geometry_request();
  inactive_geometry.phase = LlmPhase::Prefill;
  EXPECT_EQ(resolve_llm_geometry(inactive_geometry).reason_code,
            LlmWorkPlanReason::PHASE_NOT_ACTIVATED);
  inactive_geometry.phase = LlmPhase::Decode;
  inactive_geometry.kv_layout = LlmKvLayout::Paged;
  EXPECT_EQ(resolve_llm_geometry(inactive_geometry).reason_code,
            LlmWorkPlanReason::KV_LAYOUT_NOT_ACTIVATED);

  LlmMemoryWorkPlanRequest inactive_backend =
      work_plan_request(small_geometry_request(), 2, 2);
  inactive_backend.backend = LlmMemoryBackend::Metal;
  expect_invalid_plan(build_llm_memory_work_plan(inactive_backend),
                      LlmWorkPlanReason::BACKEND_NOT_ACTIVATED);
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

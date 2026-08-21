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
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/config/constants.h"
#include "llm_memory/llm_work_plan.h"

namespace {

constexpr size_t kGiB = 1024ULL * 1024ULL * 1024ULL;

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
  EXPECT_TRUE(plan.workers.empty());
  EXPECT_TRUE(plan.plan_identity.empty());
}

LlmByteRange union_for_sequence(
    const LlmMemoryWorkPlan& plan, size_t sequence_index,
    LlmByteRange LlmKvSequenceRangeTemplate::*member) {
  size_t first = std::numeric_limits<size_t>::max();
  size_t cursor = 0;
  size_t total = 0;
  bool found_range = false;
  for (const LlmWorkerWorkPlan& worker : plan.workers) {
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
  for (const LlmWorkerWorkPlan& worker : plan.workers) {
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
  EXPECT_EQ(actual.attention_kind, expected.attention_kind);
  EXPECT_EQ(actual.active_weight_bytes_per_step,
            expected.active_weight_bytes_per_step);
  EXPECT_EQ(actual.layer_count, expected.layer_count);
  EXPECT_EQ(actual.query_head_count, expected.query_head_count);
  EXPECT_EQ(actual.kv_head_count, expected.kv_head_count);
  EXPECT_EQ(actual.query_heads_per_kv_head,
            expected.query_heads_per_kv_head);
  EXPECT_EQ(actual.head_dimension, expected.head_dimension);
  EXPECT_EQ(actual.kv_element_bytes, expected.kv_element_bytes);
  EXPECT_EQ(actual.visible_context_tokens, expected.visible_context_tokens);
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
  EXPECT_EQ(actual.weight_read_bytes_per_step,
            expected.weight_read_bytes_per_step);
  EXPECT_EQ(actual.kv_read_bytes_per_step,
            expected.kv_read_bytes_per_step);
  EXPECT_EQ(actual.kv_append_write_bytes_per_step,
            expected.kv_append_write_bytes_per_step);
  EXPECT_EQ(actual.kv_only_effective_payload_bytes_per_step,
            expected.kv_only_effective_payload_bytes_per_step);
  EXPECT_EQ(actual.mixed_effective_payload_bytes_per_step,
            expected.mixed_effective_payload_bytes_per_step);
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
  EXPECT_EQ(actual.valid, expected.valid);
  EXPECT_EQ(actual.reason_code, expected.reason_code);
  expect_geometries_equal(actual.geometry, expected.geometry);
  EXPECT_EQ(actual.requested_workers, expected.requested_workers);
  EXPECT_EQ(actual.available_workers, expected.available_workers);
  EXPECT_EQ(actual.effective_workers, expected.effective_workers);
  EXPECT_EQ(actual.layer_descriptors_per_worker,
            expected.layer_descriptors_per_worker);
  EXPECT_EQ(actual.sequence_descriptors_per_worker,
            expected.sequence_descriptors_per_worker);
  EXPECT_EQ(actual.total_layer_descriptors,
            expected.total_layer_descriptors);
  EXPECT_EQ(actual.total_sequence_descriptors,
            expected.total_sequence_descriptors);
  EXPECT_EQ(actual.descriptor_bytes, expected.descriptor_bytes);
  EXPECT_EQ(actual.planner_storage_bytes, expected.planner_storage_bytes);
  EXPECT_EQ(actual.base_seed, expected.base_seed);
  EXPECT_EQ(actual.weight_buffer_seed, expected.weight_buffer_seed);
  EXPECT_EQ(actual.k_buffer_seed, expected.k_buffer_seed);
  EXPECT_EQ(actual.v_buffer_seed, expected.v_buffer_seed);
  EXPECT_EQ(actual.scenario_seeds, expected.scenario_seeds);
  expect_memory_budgets_equal(actual.memory_budget, expected.memory_budget);
  EXPECT_EQ(actual.descriptor_abi_version, expected.descriptor_abi_version);
  EXPECT_EQ(actual.backend, expected.backend);
  EXPECT_EQ(actual.phase, expected.phase);
  EXPECT_EQ(actual.weight_passes_per_step, expected.weight_passes_per_step);
  EXPECT_EQ(actual.kv_replay_factor, expected.kv_replay_factor);
  EXPECT_EQ(actual.buffer_pattern_version, expected.buffer_pattern_version);
  EXPECT_EQ(actual.methodology_version, expected.methodology_version);
  EXPECT_EQ(actual.worker_schedule, expected.worker_schedule);
  EXPECT_EQ(actual.kv_layout, expected.kv_layout);
  EXPECT_EQ(actual.plan_identity, expected.plan_identity);

  ASSERT_EQ(actual.weight_layers.size(), expected.weight_layers.size());
  EXPECT_EQ(actual.weight_layers.capacity(), expected.weight_layers.capacity());
  for (size_t layer = 0; layer < actual.weight_layers.size(); ++layer) {
    SCOPED_TRACE(::testing::Message() << "weight layer " << layer);
    expect_byte_ranges_equal(actual.weight_layers[layer],
                             expected.weight_layers[layer]);
  }

  ASSERT_EQ(actual.workers.size(), expected.workers.size());
  EXPECT_EQ(actual.workers.capacity(), expected.workers.capacity());
  for (size_t worker_index = 0; worker_index < actual.workers.size();
       ++worker_index) {
    SCOPED_TRACE(::testing::Message() << "worker " << worker_index);
    const LlmWorkerWorkPlan& actual_worker = actual.workers[worker_index];
    const LlmWorkerWorkPlan& expected_worker = expected.workers[worker_index];
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
  EXPECT_EQ(Constants::LLM_MAX_STEPS_PER_MEASUREMENT, 1000000000u);
  EXPECT_EQ(Constants::LLM_MAX_EXACT_PAYLOAD_BYTES,
            64ULL * 1024ULL * Constants::BYTES_PER_MB);
  EXPECT_DOUBLE_EQ(Constants::LLM_STREAMING_CV_WARNING_PCT, 5.0);
  EXPECT_EQ(Constants::LLM_JSON_SCHEMA_VERSION, 1);
  EXPECT_STREQ(Constants::LLM_JSON_MODE_NAME, "llm_memory");
  EXPECT_STREQ(Constants::LLM_BACKEND_NAME, "cpu");
  EXPECT_STREQ(Constants::LLM_PHASE_NAME, "decode_steady_fixed_context");
  EXPECT_EQ(Constants::LLM_WEIGHT_PASSES_PER_STEP, 1u);
  EXPECT_EQ(Constants::LLM_KV_REPLAY_FACTOR, 1u);
  EXPECT_STREQ(Constants::LLM_KV_LAYOUT,
               "contiguous_layer_batch_token_head_dimension");
  EXPECT_STREQ(Constants::LLM_WORKER_SCHEDULE,
               "worker-local-layer-order-no-per-layer-global-barrier");
  EXPECT_STREQ(
      Constants::LLM_METHODOLOGY_VERSION,
      "llm-memory-v1-cpu-fixed-context-warm-layer-interleaved");
  EXPECT_STREQ(Constants::LLM_DESCRIPTOR_ABI_VERSION,
               "llm-memory-descriptor-abi-v1");
  EXPECT_STREQ(Constants::LLM_BUFFER_PATTERN_VERSION,
               "llm-buffer-pattern-v1");
  EXPECT_STREQ(Constants::LLM_APPEND_PATTERN_VERSION,
               "llm-kv-append-affine64-v1");
  EXPECT_STREQ(Constants::LLM_READ_CHECKSUM_VERSION,
               "llm-read-checksum-v1");

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
  EXPECT_EQ(geometry.weight_read_bytes_per_step, 4 * kGiB);
  EXPECT_EQ(geometry.kv_read_bytes_per_step, kGiB);
  EXPECT_EQ(geometry.kv_append_write_bytes_per_step, 131072u);
  EXPECT_EQ(geometry.kv_only_effective_payload_bytes_per_step,
            1073872896u);
  EXPECT_EQ(geometry.mixed_effective_payload_bytes_per_step,
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
  EXPECT_EQ(single.weight_read_bytes_per_step, 1024u);
  EXPECT_EQ(single.kv_read_bytes_per_step, 384u);
  EXPECT_EQ(single.kv_append_write_bytes_per_step, 128u);
  EXPECT_EQ(single.kv_only_effective_payload_bytes_per_step, 512u);
  EXPECT_EQ(single.mixed_effective_payload_bytes_per_step, 1536u);

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
  EXPECT_EQ(weights.weight_read_bytes, 4096u);
  EXPECT_EQ(weights.kv_read_bytes, 0u);
  EXPECT_EQ(weights.kv_append_write_bytes, 0u);
  EXPECT_EQ(weights.effective_payload_bytes, 4096u);
  EXPECT_EQ(kv.weight_read_bytes, 0u);
  EXPECT_EQ(kv.kv_read_bytes, 1536u);
  EXPECT_EQ(kv.kv_append_write_bytes, 512u);
  EXPECT_EQ(kv.effective_payload_bytes, 2048u);
  EXPECT_EQ(mixed.weight_read_bytes, 4096u);
  EXPECT_EQ(mixed.kv_read_bytes, 1536u);
  EXPECT_EQ(mixed.kv_append_write_bytes, 512u);
  EXPECT_EQ(mixed.effective_payload_bytes, 6144u);

  const LlmGeometry batched =
      resolve_llm_geometry(small_geometry_request(2));
  ASSERT_TRUE(batched.valid);
  EXPECT_EQ(batched.weight_read_bytes_per_step, 1024u);
  EXPECT_EQ(batched.kv_read_bytes_per_step, 768u);
  EXPECT_EQ(batched.kv_append_write_bytes_per_step, 256u);
  const LlmMemoryWorkPlan batched_model = build_llm_memory_work_plan(
      work_plan_request(small_geometry_request(2), 2, 2));
  ASSERT_TRUE(batched_model.valid) << batched_model.reason_code;
  const LlmScenarioWorkPlan batched_mixed = build_llm_scenario_work_plan(
      batched_model, LlmScenario::Mixed, 4, true);
  ASSERT_TRUE(batched_mixed.valid);
  EXPECT_EQ(batched_mixed.weight_read_bytes, 4096u);
  EXPECT_EQ(batched_mixed.kv_read_bytes, 3072u);
  EXPECT_EQ(batched_mixed.kv_append_write_bytes, 1024u);
  EXPECT_EQ(batched_mixed.effective_payload_bytes, 8192u);
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
  ASSERT_EQ(plan.sequence_descriptors_per_worker, 4u);
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
    ASSERT_EQ(plan.effective_workers, workers);

    size_t next_offset = 0;
    for (size_t worker = 0; worker < workers; ++worker) {
      const LlmByteRange& range = plan.workers[worker].sequences[0].k_visible;
      EXPECT_EQ(range.offset_bytes, next_offset) << "span=" << span;
      EXPECT_GT(range.span_bytes, 0u) << "span=" << span;
      next_offset += range.span_bytes;
      if (worker + 1 == workers) {
        continue;
      }
      const size_t remaining_workers = workers - worker - 1;
      const size_t minimum_boundary =
          plan.workers[worker].sequences[0].k_visible.offset_bytes + 1;
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
  ASSERT_EQ(plan.effective_workers, 3u);
  ASSERT_EQ(plan.weight_layers.size(), 2u);
  EXPECT_EQ(plan.weight_layers[1].offset_bytes, 100u);
  EXPECT_EQ(plan.weight_layers[1].span_bytes, 99u);

  size_t cursor = plan.weight_layers[1].offset_bytes;
  for (size_t worker = 0; worker < plan.effective_workers; ++worker) {
    const LlmByteRange& range = plan.workers[worker].layers[1].weight;
    EXPECT_EQ(range.offset_bytes, cursor);
    EXPECT_GT(range.span_bytes, 0u);
    cursor += range.span_bytes;
    if (worker + 1 < plan.effective_workers) {
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
  ASSERT_EQ(plan.workers.size(), 2u);
  const LlmKvSequenceRangeTemplate& first = plan.workers[0].sequences[0];
  const LlmKvSequenceRangeTemplate& second = plan.workers[1].sequences[0];
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
  EXPECT_EQ(plan.layer_descriptors_per_worker, 2u);
  EXPECT_EQ(plan.sequence_descriptors_per_worker, 4u);
  EXPECT_EQ(plan.total_layer_descriptors, 4u);
  EXPECT_EQ(plan.total_sequence_descriptors, 8u);
  for (const LlmWorkerWorkPlan& worker : plan.workers) {
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

  size_t actual_capacity_bytes =
      plan.weight_layers.capacity() * sizeof(LlmByteRange) +
      plan.workers.capacity() * sizeof(LlmWorkerWorkPlan);
  for (const LlmWorkerWorkPlan& worker : plan.workers) {
    actual_capacity_bytes +=
        worker.layers.capacity() * sizeof(LlmLayerRangeTemplate);
    actual_capacity_bytes +=
        worker.sequences.capacity() * sizeof(LlmKvSequenceRangeTemplate);
  }
  const size_t requested_element_bytes =
      plan.geometry.layer_count * sizeof(LlmByteRange) +
      plan.effective_workers * sizeof(LlmWorkerWorkPlan) +
      plan.effective_workers * plan.geometry.layer_count *
          sizeof(LlmLayerRangeTemplate) +
      plan.effective_workers * plan.geometry.layer_count *
          plan.geometry.batch_size * sizeof(LlmKvSequenceRangeTemplate);
  EXPECT_GE(actual_capacity_bytes, requested_element_bytes);
  EXPECT_EQ(plan.planner_storage_bytes, actual_capacity_bytes);
  EXPECT_EQ(plan.memory_budget.request.planner_storage_bytes,
            actual_capacity_bytes);
  EXPECT_EQ(plan.memory_budget.request.auxiliary_bytes,
            plan.descriptor_bytes + actual_capacity_bytes + 7 + 9);
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
  EXPECT_EQ(plan.geometry.active_weight_bytes_per_step,
            Constants::BYTES_PER_MB);
  EXPECT_EQ(plan.geometry.layer_count, 2u);
  EXPECT_EQ(plan.geometry.query_head_count, 4u);
  EXPECT_EQ(plan.geometry.kv_head_count, 2u);
  EXPECT_EQ(plan.geometry.head_dimension, 8u);
  EXPECT_EQ(plan.geometry.kv_element_bytes, 4u);
  EXPECT_EQ(plan.geometry.visible_context_tokens, 3u);
  EXPECT_EQ(plan.geometry.batch_size, 2u);
  EXPECT_EQ(plan.requested_workers, 3u);
  EXPECT_EQ(plan.available_workers, 5u);
  EXPECT_EQ(plan.effective_workers, 3u);
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
  EXPECT_EQ(plan.requested_workers, 100u);
  EXPECT_EQ(plan.available_workers, 100u);
  EXPECT_EQ(plan.effective_workers, 1u);
  EXPECT_EQ(plan.geometry.active_weight_bytes_per_step, 2u);
  EXPECT_EQ(plan.geometry.kv_capacity_bytes, 4u);
  ASSERT_EQ(plan.workers.size(), 1u);
  EXPECT_FALSE(plan.workers[0].layers.empty());
  EXPECT_FALSE(plan.workers[0].sequences.empty());
}

TEST(LlmMemoryWorkPlanTest,
     AvailableWorkersCapAnOtherwiseExecutableRequestedTeam) {
  const LlmGeometryRequest geometry = {100, 1, 1, 1, 8, 1, 1, 1};
  const LlmMemoryWorkPlan plan =
      build_llm_memory_work_plan(work_plan_request(geometry, 5, 2));
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  EXPECT_EQ(plan.requested_workers, 5u);
  EXPECT_EQ(plan.available_workers, 2u);
  EXPECT_EQ(plan.effective_workers, 2u);
  ASSERT_EQ(plan.workers.size(), 2u);
  for (const LlmWorkerWorkPlan& worker : plan.workers) {
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
  ASSERT_EQ(plan.effective_workers, 3u);

  size_t second_layer_bytes = 0;
  size_t empty_second_layer_ranges = 0;
  size_t first_sequence_bytes = 0;
  size_t empty_first_sequence_ranges = 0;
  for (const LlmWorkerWorkPlan& worker : plan.workers) {
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

  for (const LlmWorkerWorkPlan& worker : plan.workers) {
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
  EXPECT_EQ(constructor_source.effective_workers, 0u);
  EXPECT_TRUE(constructor_source.weight_layers.empty());
  EXPECT_TRUE(constructor_source.workers.empty());
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
  EXPECT_EQ(assignment_source.effective_workers, 0u);
  EXPECT_TRUE(assignment_source.weight_layers.empty());
  EXPECT_TRUE(assignment_source.workers.empty());
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
  EXPECT_EQ(calculate_llm_pilot_steps(weights), 8192u);
  EXPECT_EQ(calculate_llm_pilot_steps(kv), 16384u);
  EXPECT_EQ(calculate_llm_pilot_steps(mixed), 5462u);
  EXPECT_EQ(calculate_llm_calibrated_steps(0.010, 100, weights), 1500u);
  EXPECT_EQ(calculate_llm_calibrated_steps(0.010, 100, kv), 1500u);
  EXPECT_EQ(calculate_llm_calibrated_steps(0.010, 100, mixed), 1500u);
  EXPECT_EQ(calculate_llm_calibrated_steps(0.0, 100, weights), 0u);
  EXPECT_EQ(calculate_llm_calibrated_steps(
                std::numeric_limits<double>::quiet_NaN(), 100, weights),
            0u);
  EXPECT_EQ(calculate_llm_calibrated_steps(
                std::numeric_limits<double>::infinity(), 100, weights),
            0u);
  const LlmScenarioLimits invalid = calculate_llm_scenario_limits(
      geometry, static_cast<LlmScenario>(99));
  EXPECT_FALSE(invalid.valid);
  EXPECT_EQ(invalid.reason_code, LlmWorkPlanReason::INVALID_SCENARIO);
  EXPECT_EQ(calculate_llm_pilot_steps(invalid), 0u);
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
  EXPECT_EQ(limited_weights.effective_maximum_steps, 64u);
  EXPECT_EQ(limited_kv.effective_maximum_steps,
            Constants::LLM_MAX_STEPS_PER_MEASUREMENT);
  EXPECT_EQ(limited_mixed.effective_maximum_steps, 63u);
  EXPECT_EQ(calculate_llm_calibrated_steps(0.001, 100, limited_weights),
            64u);
  EXPECT_EQ(calculate_llm_calibrated_steps(0.001, 100, limited_kv),
            15000u);
  EXPECT_EQ(calculate_llm_calibrated_steps(0.001, 100, limited_mixed),
            63u);
  EXPECT_EQ(classify_llm_duration_quality(
                0.050, limited_weights.effective_maximum_steps,
                limited_weights),
            "guardrail-limited-below-target");
}

TEST(LlmMemoryWorkPlanTest,
     DurationQualityUsesInclusiveWindowAndSingleStepException) {
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
            "single-step-over-target");
  EXPECT_EQ(classify_llm_duration_quality(0.050, 2, limits),
            "below-target-window");
  EXPECT_EQ(classify_llm_duration_quality(0.300, 2, limits),
            "above-target-window");
  EXPECT_EQ(classify_llm_duration_quality(0.0, 1, limits),
            "invalid-duration");
}

TEST(LlmMemoryWorkPlanTest, ExplicitStepsStayExactAndRespectBothCaps) {
  LlmGeometryRequest request = {1, 1, 1, 1, 1, 1, 1, 1};
  LlmGeometry geometry = resolve_llm_geometry(request);
  ASSERT_TRUE(geometry.valid);
  const LlmScenarioLimits tiny =
      calculate_llm_scenario_limits(geometry, LlmScenario::WeightsOnly);
  ASSERT_TRUE(tiny.valid);
  LlmMemoryWorkPlan model_plan = build_llm_memory_work_plan(
      work_plan_request(request, 1, 1));
  ASSERT_TRUE(model_plan.valid) << model_plan.reason_code;
  EXPECT_EQ(tiny.effective_maximum_steps,
            Constants::LLM_MAX_STEPS_PER_MEASUREMENT);
  EXPECT_TRUE(build_llm_scenario_work_plan(
                  model_plan, LlmScenario::WeightsOnly,
                  tiny.effective_maximum_steps, true)
                  .valid);
  EXPECT_EQ(build_llm_scenario_work_plan(
                model_plan, LlmScenario::WeightsOnly,
                tiny.effective_maximum_steps + 1, true)
                .reason_code,
            LlmWorkPlanReason::STEP_CAP_EXCEEDED);

  request.active_weight_bytes = kGiB;
  geometry = resolve_llm_geometry(request);
  ASSERT_TRUE(geometry.valid);
  model_plan = build_llm_memory_work_plan(work_plan_request(request, 1, 1));
  ASSERT_TRUE(model_plan.valid) << model_plan.reason_code;
  const LlmScenarioLimits payload_limited =
      calculate_llm_scenario_limits(geometry, LlmScenario::WeightsOnly);
  ASSERT_TRUE(payload_limited.valid);
  EXPECT_EQ(payload_limited.effective_maximum_steps, 64u);
  const LlmScenarioWorkPlan boundary = build_llm_scenario_work_plan(
      model_plan, LlmScenario::WeightsOnly, 64, true);
  ASSERT_TRUE(boundary.valid);
  EXPECT_EQ(boundary.steps, 64u);
  EXPECT_EQ(boundary.effective_payload_bytes,
            Constants::LLM_MAX_EXACT_PAYLOAD_BYTES);
  EXPECT_EQ(build_llm_scenario_work_plan(
                model_plan, LlmScenario::WeightsOnly, 65, true)
                .reason_code,
            LlmWorkPlanReason::EXACT_PAYLOAD_CAP_EXCEEDED);

  const LlmMemoryWorkPlan frozen_model_plan = build_llm_memory_work_plan(
      work_plan_request(small_geometry_request(), 2, 2));
  ASSERT_TRUE(frozen_model_plan.valid) << frozen_model_plan.reason_code;
  const LlmFrozenScenarioPlans frozen = freeze_llm_scenario_work_plans(
      frozen_model_plan, {4, 4, 4}, true);
  ASSERT_TRUE(frozen.valid) << frozen.reason_code;
  EXPECT_TRUE(frozen.explicit_iterations);
  for (const LlmScenarioWorkPlan& scenario : frozen.scenarios) {
    EXPECT_EQ(scenario.steps, 4u);
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
  LlmGeometryRequest request = {Constants::LLM_MAX_EXACT_PAYLOAD_BYTES,
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
      Constants::LLM_MAX_STEPS_PER_MEASUREMENT, true);
  EXPECT_FALSE(overflow.valid);
  EXPECT_EQ(overflow.reason_code,
            LlmWorkPlanReason::EXACT_PAYLOAD_OVERFLOW);
  EXPECT_TRUE(overflow.plan_identity.empty());

  const LlmGeometry above_payload_cap = resolve_llm_geometry(
      {Constants::LLM_MAX_EXACT_PAYLOAD_BYTES + 1,
       1, 1, 1, 1, 1, 1, 1});
  ASSERT_TRUE(above_payload_cap.valid) << above_payload_cap.reason_code;
  const LlmScenarioLimits impossible_single_step =
      calculate_llm_scenario_limits(above_payload_cap,
                                    LlmScenario::WeightsOnly);
  EXPECT_FALSE(impossible_single_step.valid);
  EXPECT_EQ(impossible_single_step.reason_code,
            LlmWorkPlanReason::PAYLOAD_CAP_BELOW_ONE_STEP);

  const LlmMemoryWorkPlan model_plan = build_llm_memory_work_plan(
      work_plan_request(small_geometry_request(), 2, 2));
  ASSERT_TRUE(model_plan.valid) << model_plan.reason_code;
  const LlmFrozenScenarioPlans invalid = freeze_llm_scenario_work_plans(
      model_plan, {1, 0, 1}, false);
  EXPECT_FALSE(invalid.valid);
  EXPECT_EQ(invalid.reason_code, LlmWorkPlanReason::STEP_COUNT_ZERO);
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
  EXPECT_EQ(first.descriptor_abi_version,
            Constants::LLM_DESCRIPTOR_ABI_VERSION);
  EXPECT_EQ(first.backend, Constants::LLM_BACKEND_NAME);
  EXPECT_EQ(first.phase, Constants::LLM_PHASE_NAME);
  EXPECT_EQ(first.weight_passes_per_step, 1u);
  EXPECT_EQ(first.kv_replay_factor, 1u);
  EXPECT_EQ(first.buffer_pattern_version,
            Constants::LLM_BUFFER_PATTERN_VERSION);
  EXPECT_EQ(first.methodology_version, Constants::LLM_METHODOLOGY_VERSION);
  EXPECT_EQ(first.kv_layout, Constants::LLM_KV_LAYOUT);
  EXPECT_EQ(first.worker_schedule, Constants::LLM_WORKER_SCHEDULE);
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
  EXPECT_EQ(first.requested_workers, smaller_team.requested_workers);
  EXPECT_EQ(first.effective_workers, 2u);
  EXPECT_EQ(smaller_team.effective_workers, 1u);
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
  EXPECT_EQ(scenario_a.effective_payload_bytes,
            scenario_b.effective_payload_bytes);
  EXPECT_NE(scenario_a.kv_read_bytes, scenario_b.kv_read_bytes);
  EXPECT_NE(scenario_a.kv_append_write_bytes,
            scenario_b.kv_append_write_bytes);
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

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
 * @file test_llm_metal_backend.cpp
 * @brief Pure planning and real-device tests for the inactive LLM Metal foundation
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/config/constants.h"
#include "core/system/page_size.h"
#include "llm_memory/llm_kv_layout.h"
#include "llm_memory/llm_metal_backend.h"
#include "llm_memory/llm_work_plan.h"

namespace {

constexpr size_t kGiB = 1024ULL * 1024ULL * 1024ULL;
constexpr size_t kFoundationPipelineCount = 5;
constexpr std::string_view kCanonicalKernelSourceSha256 =
    "50f1deb3273cca4e8b072eaaf842c6fbcabb60971c0953c02f36461d879779c4";

LlmGeometry contiguous_geometry(size_t weight_bytes, size_t context_tokens = 1, size_t layer_count = 1,
                                size_t batch_size = 1, size_t head_dimension = 1) {
  LlmGeometryRequest request;
  request.active_weight_bytes = weight_bytes;
  request.layer_count = layer_count;
  request.query_head_count = 1;
  request.kv_head_count = 1;
  request.head_dimension = head_dimension;
  request.kv_element_bytes = 1;
  request.visible_context_tokens = context_tokens;
  request.batch_size = batch_size;
  return resolve_llm_geometry(request);
}

LlmGeometry paged_geometry(size_t weight_bytes, size_t context_tokens, size_t block_tokens, size_t layer_count = 1,
                           size_t batch_size = 1, size_t head_dimension = 1) {
  LlmGeometryRequest request;
  request.active_weight_bytes = weight_bytes;
  request.layer_count = layer_count;
  request.query_head_count = 1;
  request.kv_head_count = 1;
  request.head_dimension = head_dimension;
  request.kv_element_bytes = 1;
  request.visible_context_tokens = context_tokens;
  request.batch_size = batch_size;
  request.kv_block_tokens = block_tokens;
  request.kv_layout = LlmKvLayout::Paged;
  return resolve_llm_geometry(request);
}

LlmKvLayoutPlan paged_layout_for(const LlmGeometry& geometry) {
  const size_t sequence_tokens =
      geometry.decode.has_value() ? geometry.decode->visible_context_tokens : geometry.prefill->prompt_tokens;
  return build_llm_kv_layout_plan({sequence_tokens, geometry.kv_block_tokens, geometry.layer_count, geometry.batch_size,
                                   geometry.k_or_v_record_bytes_per_layer});
}

LlmMetalResourcePlanRequest resource_request(const LlmGeometry& geometry,
                                             const LlmMetalPlanningLimits& limits = LlmMetalPlanningLimits{}) {
  LlmMetalResourcePlanRequest request;
  request.geometry = geometry;
  if (geometry.kv_layout == LlmKvLayout::Paged && geometry.valid) {
    request.paged_layout = paged_layout_for(geometry);
  }
  request.argument_buffer_encoded_length = 1025;
  request.argument_buffer_alignment = 256;
  request.max_buffer_length = std::max<size_t>(limits.segment_capacity_bytes, 16 * 1024);
  request.available_memory_bytes = 32 * kGiB;
  request.host_mapping_granularity_bytes = 4096;
  request.limits = limits;
  return request;
}

const LlmMetalPlannedResource* find_planned_resource(const LlmMetalResourcePlan& plan, LlmMetalResourcePool pool,
                                                     size_t pool_index = 0) {
  const auto found = std::find_if(plan.planned_resources.begin(), plan.planned_resources.end(),
                                  [pool, pool_index](const LlmMetalPlannedResource& resource) {
                                    return resource.pool == pool && resource.pool_index == pool_index;
                                  });
  return found == plan.planned_resources.end() ? nullptr : &*found;
}

std::vector<LlmMetalAllocatedResource> exact_allocations(const LlmMetalResourcePlan& plan) {
  std::vector<LlmMetalAllocatedResource> allocations;
  allocations.reserve(plan.planned_resources.size());
  for (const LlmMetalPlannedResource& resource : plan.planned_resources) {
    allocations.push_back({resource.pool, resource.pool_index, resource.length_bytes, std::nullopt});
  }
  return allocations;
}

LlmMetalCapabilityProbe ready_capability_probe() {
  LlmMetalCapabilityProbe probe;
  probe.device_available = true;
  probe.has_unified_memory = true;
  probe.apple7_family_supported = true;
  probe.argument_buffers_tier2_supported = true;
  probe.max_buffer_length = Constants::LLM_METAL_SEGMENT_CAPACITY_BYTES;
  probe.command_queue_created = true;
  probe.source_compiled = true;
  probe.foundation_pipeline_count = kFoundationPipelineCount;
  probe.argument_encoder_created = true;
  probe.argument_buffer_encoded_length = 4096;
  probe.argument_buffer_alignment = 256;
  return probe;
}

bool is_stable_capability_unsupported_reason(std::string_view reason) {
  return reason == LlmBackendReason::METAL_DEVICE_UNAVAILABLE || reason == LlmBackendReason::UNIFIED_MEMORY_REQUIRED ||
         reason == LlmBackendReason::APPLE7_FAMILY_REQUIRED ||
         reason == LlmBackendReason::ARGUMENT_BUFFER_TIER2_REQUIRED ||
         reason == LlmBackendReason::METAL_MAX_BUFFER_LENGTH_BELOW_SEGMENT_CAPACITY;
}

LlmMemoryConfig metal_config() {
  LlmMemoryConfig config;
  config.backend = LlmMemoryBackend::Metal;
  return config;
}

LlmMemoryWorkPlan make_metal_model_plan(LlmGeometry geometry, LlmMetalExecutionPlan execution, std::string identity) {
  LlmMemoryWorkPlan plan;
  plan.valid = execution.valid;
  plan.reason_code = execution.reason_code;
  plan.geometry = std::move(geometry);
  plan.backend = LlmMemoryBackend::Metal;
  plan.phase = plan.geometry.phase;
  plan.kv_layout = plan.geometry.kv_layout;
  plan.work_unit_kind = plan.geometry.work_unit_kind;
  plan.base_seed = UINT64_C(0x1020304050607080);
  plan.weight_buffer_seed = UINT64_C(0x1112131415161718);
  plan.k_buffer_seed = UINT64_C(0x2122232425262728);
  plan.v_buffer_seed = UINT64_C(0x3132333435363738);
  plan.plan_identity = std::move(identity);
  plan.backend_execution_plan = std::move(execution);
  return plan;
}

LlmScenarioWorkPlan one_work_unit_scenario(const LlmMemoryWorkPlan& model) {
  LlmScenarioWorkPlan scenario;
  scenario.valid = true;
  scenario.reason_code = LlmWorkPlanReason::VALID;
  scenario.scenario = LlmScenario::WeightsOnly;
  scenario.work_unit_kind = model.work_unit_kind;
  scenario.model_plan_identity = model.plan_identity;
  scenario.work_units = 1;
  scenario.plan_identity = "llm-metal-phase8-no-timed-task";
  return scenario;
}

TEST(LlmMetalBackendTest, CapabilityStateMachineUsesStableFirstFailureOrdering) {
  struct CapabilityCase {
    void (*mutate)(LlmMetalCapabilityProbe&);
    LlmBackendStatus status;
    std::string_view reason;
  };
  const std::array<CapabilityCase, 11> cases = {{
      {+[](LlmMetalCapabilityProbe& probe) { probe.device_available = false; }, LlmBackendStatus::Unsupported,
       LlmBackendReason::METAL_DEVICE_UNAVAILABLE},
      {+[](LlmMetalCapabilityProbe& probe) { probe.has_unified_memory = false; }, LlmBackendStatus::Unsupported,
       LlmBackendReason::UNIFIED_MEMORY_REQUIRED},
      {+[](LlmMetalCapabilityProbe& probe) { probe.apple7_family_supported = false; }, LlmBackendStatus::Unsupported,
       LlmBackendReason::APPLE7_FAMILY_REQUIRED},
      {+[](LlmMetalCapabilityProbe& probe) { probe.argument_buffers_tier2_supported = false; },
       LlmBackendStatus::Unsupported, LlmBackendReason::ARGUMENT_BUFFER_TIER2_REQUIRED},
      {+[](LlmMetalCapabilityProbe& probe) {
         probe.max_buffer_length = Constants::LLM_METAL_SEGMENT_CAPACITY_BYTES - 1;
       },
       LlmBackendStatus::Unsupported, LlmBackendReason::METAL_MAX_BUFFER_LENGTH_BELOW_SEGMENT_CAPACITY},
      {+[](LlmMetalCapabilityProbe& probe) { probe.command_queue_created = false; }, LlmBackendStatus::Failed,
       LlmBackendReason::METAL_COMMAND_QUEUE_CREATION_FAILED},
      {+[](LlmMetalCapabilityProbe& probe) { probe.source_compiled = false; }, LlmBackendStatus::Failed,
       LlmBackendReason::METAL_KERNEL_COMPILATION_FAILED},
      {+[](LlmMetalCapabilityProbe& probe) { probe.foundation_pipeline_count = kFoundationPipelineCount - 1; },
       LlmBackendStatus::Failed, LlmBackendReason::METAL_PIPELINE_CREATION_FAILED},
      {+[](LlmMetalCapabilityProbe& probe) { probe.argument_encoder_created = false; }, LlmBackendStatus::Failed,
       LlmBackendReason::METAL_ARGUMENT_ENCODER_CREATION_FAILED},
      {+[](LlmMetalCapabilityProbe& probe) { probe.argument_buffer_encoded_length = 0; }, LlmBackendStatus::Failed,
       LlmBackendReason::METAL_ARGUMENT_BUFFER_LAYOUT_INVALID},
      {+[](LlmMetalCapabilityProbe& probe) { probe.argument_buffer_alignment = 3; }, LlmBackendStatus::Failed,
       LlmBackendReason::METAL_ARGUMENT_BUFFER_LAYOUT_INVALID},
  }};

  for (const CapabilityCase& test_case : cases) {
    LlmMetalCapabilityProbe probe = ready_capability_probe();
    test_case.mutate(probe);
    const LlmBackendLifecycleResult result = evaluate_llm_metal_capabilities(probe);
    SCOPED_TRACE(test_case.reason);
    EXPECT_EQ(result.status, test_case.status);
    EXPECT_EQ(result.reason_code, test_case.reason);
  }

  const LlmBackendLifecycleResult ready = evaluate_llm_metal_capabilities(ready_capability_probe());
  EXPECT_EQ(ready.status, LlmBackendStatus::Ready);
  EXPECT_EQ(ready.reason_code, LlmBackendReason::VALID);
}

TEST(LlmMetalBackendTest, ArgumentBufferPlanFreezesCanonicalSlotBasesCountsAndCap) {
  const LlmMetalArgumentBufferPlan plan = build_llm_metal_argument_buffer_plan(2, 3, 4, 5);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  EXPECT_EQ(plan.weight_slot_base, 0U);
  EXPECT_EQ(plan.k_slot_base, 256U);
  EXPECT_EQ(plan.v_slot_base, 512U);
  EXPECT_EQ(plan.table_slot_base, 768U);
  EXPECT_EQ(plan.status_slot, 1024U);
  EXPECT_EQ(plan.encoded_resource_slot_count, 1025U);
  EXPECT_EQ(plan.weight_segment_count, 2U);
  EXPECT_EQ(plan.k_segment_count, 3U);
  EXPECT_EQ(plan.v_segment_count, 4U);
  EXPECT_EQ(plan.table_segment_count, 5U);
  EXPECT_EQ(plan.active_resource_count, 15U);
  EXPECT_FALSE(plan.identity.empty());

  const LlmMetalArgumentBufferPlan exact_cap = build_llm_metal_argument_buffer_plan(4, 4, 4, 4, 4);
  ASSERT_TRUE(exact_cap.valid) << exact_cap.reason_code;
  EXPECT_EQ(exact_cap.k_slot_base, 4U);
  EXPECT_EQ(exact_cap.v_slot_base, 8U);
  EXPECT_EQ(exact_cap.table_slot_base, 12U);
  EXPECT_EQ(exact_cap.status_slot, 16U);
  EXPECT_EQ(exact_cap.encoded_resource_slot_count, 17U);
  EXPECT_EQ(exact_cap.active_resource_count, 17U);

  for (size_t overflowing_pool = 0; overflowing_pool < 4; ++overflowing_pool) {
    std::array<size_t, 4> counts = {4, 4, 4, 4};
    counts[overflowing_pool] = 5;
    const LlmMetalArgumentBufferPlan invalid =
        build_llm_metal_argument_buffer_plan(counts[0], counts[1], counts[2], counts[3], 4);
    SCOPED_TRACE(overflowing_pool);
    EXPECT_FALSE(invalid.valid);
    EXPECT_EQ(invalid.reason_code, LlmMetalPlanReason::ARGUMENT_BUFFER_LAYOUT_INVALID);
    EXPECT_TRUE(invalid.identity.empty());
  }
}

TEST(LlmMetalBackendTest, ContiguousPoolsUseExactCanonicalBoundarySegmentsWithoutAllocation) {
  const size_t capacity = Constants::LLM_METAL_SEGMENT_CAPACITY_BYTES;
  const std::array<size_t, 3> lengths = {capacity - 1, capacity, capacity + 1};
  const std::array<std::vector<size_t>, 3> expected = {std::vector<size_t>{capacity - 1}, std::vector<size_t>{capacity},
                                                       std::vector<size_t>{capacity, 1}};

  for (size_t index = 0; index < lengths.size(); ++index) {
    const LlmGeometry geometry = contiguous_geometry(lengths[index], lengths[index]);
    ASSERT_TRUE(geometry.valid) << geometry.reason_code;
    ASSERT_EQ(geometry.k_mapping_bytes, lengths[index]);
    ASSERT_EQ(geometry.v_mapping_bytes, lengths[index]);

    const LlmMetalExecutionPlan plan = build_llm_metal_execution_plan(resource_request(geometry));
    SCOPED_TRACE(lengths[index]);
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    EXPECT_EQ(plan.resources.weight_segments.segment_lengths, expected[index]);
    EXPECT_EQ(plan.resources.k_segments.segment_lengths, expected[index]);
    EXPECT_EQ(plan.resources.v_segments.segment_lengths, expected[index]);
    EXPECT_EQ(plan.resources.weight_segments.total_length_bytes, lengths[index]);
    EXPECT_EQ(plan.resources.k_segments.total_length_bytes, lengths[index]);
    EXPECT_EQ(plan.resources.v_segments.total_length_bytes, lengths[index]);
    EXPECT_FALSE(plan.resources.table_segments.has_value());
  }
}

TEST(LlmMetalBackendTest, ContiguousPoolsRetainEveryExactLengthAcrossMultipleSegments) {
  const size_t capacity = Constants::LLM_METAL_SEGMENT_CAPACITY_BYTES;
  const size_t length = 3 * capacity + 17;
  const LlmGeometry geometry = contiguous_geometry(length, length);
  ASSERT_TRUE(geometry.valid) << geometry.reason_code;
  const LlmMetalExecutionPlan plan = build_llm_metal_execution_plan(resource_request(geometry));
  ASSERT_TRUE(plan.valid) << plan.reason_code;

  const std::vector<size_t> expected = {capacity, capacity, capacity, 17};
  for (const LlmKvSegmentPlan* segments :
       {&plan.resources.weight_segments, &plan.resources.k_segments, &plan.resources.v_segments}) {
    EXPECT_EQ(segments->segment_lengths, expected);
    EXPECT_EQ(segments->segment_count, expected.size());
    EXPECT_EQ(std::accumulate(segments->segment_lengths.begin(), segments->segment_lengths.end(), size_t{0}), length);
  }
}

TEST(LlmMetalBackendTest, MultiGibContiguousPoolRetainsExactBoundedSegmentCountWithoutAllocation) {
  const size_t capacity = Constants::LLM_METAL_SEGMENT_CAPACITY_BYTES;
  const size_t weight_bytes = 5 * kGiB + 17;
  const LlmGeometry geometry = contiguous_geometry(weight_bytes);
  ASSERT_TRUE(geometry.valid) << geometry.reason_code;
  LlmMetalResourcePlanRequest request = resource_request(geometry);
  request.available_memory_bytes = 16 * kGiB;

  const LlmMetalExecutionPlan plan = build_llm_metal_execution_plan(request);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  EXPECT_EQ(plan.resources.weight_segments.segment_count, 21U);
  ASSERT_EQ(plan.resources.weight_segments.segment_lengths.size(), 21U);
  EXPECT_TRUE(std::all_of(plan.resources.weight_segments.segment_lengths.begin(),
                          plan.resources.weight_segments.segment_lengths.end() - 1,
                          [](size_t length) { return length == capacity; }));
  EXPECT_EQ(plan.resources.weight_segments.segment_lengths.back(), 17U);
  EXPECT_EQ(plan.resources.weight_segments.total_length_bytes, weight_bytes);
}

TEST(LlmMetalBackendTest, PagedKvSegmentsKeepWholeBlocksAndTableSegmentsKeepWholeEntries) {
  LlmMetalPlanningLimits limits;
  limits.segment_capacity_bytes = 100;
  limits.segment_slots_per_pool = 32;
  const LlmGeometry geometry = paged_geometry(101, 49, 16, 2, 2, 3);
  ASSERT_TRUE(geometry.valid) << geometry.reason_code;
  ASSERT_EQ(geometry.kv_block_bytes, 48U);
  ASSERT_EQ(geometry.total_physical_blocks, 16U);
  ASSERT_EQ(geometry.block_table_entries, 8U);

  const LlmMetalExecutionPlan plan = build_llm_metal_execution_plan(resource_request(geometry, limits));
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  ASSERT_TRUE(plan.resources.table_segments.has_value());
  EXPECT_EQ(plan.resources.k_segments.elements_per_segment, 2U);
  EXPECT_EQ(plan.resources.k_segments.segment_count, 8U);
  EXPECT_EQ(plan.resources.k_segments.segment_lengths, std::vector<size_t>(8, 96));
  EXPECT_EQ(plan.resources.v_segments.segment_lengths, plan.resources.k_segments.segment_lengths);
  EXPECT_EQ(plan.resources.k_segments.total_length_bytes, geometry.k_mapping_bytes);
  EXPECT_TRUE(std::all_of(plan.resources.k_segments.segment_lengths.begin(),
                          plan.resources.k_segments.segment_lengths.end(),
                          [](size_t length) { return length % 48 == 0; }));
  EXPECT_EQ(plan.resources.table_segments->element_bytes, sizeof(uint32_t));
  EXPECT_EQ(plan.resources.table_segments->segment_lengths, std::vector<size_t>{32});
  EXPECT_EQ(plan.resources.table_segments->total_length_bytes, geometry.block_table_bytes);
}

TEST(LlmMetalBackendTest, PagedPoolsHonorInjectedSegmentBoundaryMinusOneExactAndPlusOne) {
  LlmMetalPlanningLimits limits;
  limits.segment_capacity_bytes = 100;
  limits.segment_slots_per_pool = 8;
  const std::array<size_t, 3> physical_pool_bytes = {99, 100, 101};
  const std::array<std::vector<size_t>, 3> expected = {std::vector<size_t>{99}, std::vector<size_t>{100},
                                                       std::vector<size_t>{100, 1}};

  for (size_t index = 0; index < physical_pool_bytes.size(); ++index) {
    const LlmGeometry geometry = paged_geometry(1, physical_pool_bytes[index], 1);
    ASSERT_TRUE(geometry.valid) << geometry.reason_code;
    ASSERT_EQ(geometry.kv_block_bytes, 1U);
    ASSERT_EQ(geometry.k_mapping_bytes, physical_pool_bytes[index]);
    const LlmMetalExecutionPlan plan = build_llm_metal_execution_plan(resource_request(geometry, limits));
    SCOPED_TRACE(physical_pool_bytes[index]);
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    EXPECT_EQ(plan.resources.k_segments.segment_lengths, expected[index]);
    EXPECT_EQ(plan.resources.v_segments.segment_lengths, expected[index]);
    ASSERT_TRUE(plan.resources.table_segments.has_value());
    EXPECT_TRUE(std::all_of(plan.resources.table_segments->segment_lengths.begin(),
                            plan.resources.table_segments->segment_lengths.end(),
                            [](size_t length) { return length % sizeof(uint32_t) == 0 && length <= 100; }));
  }
}

TEST(LlmMetalBackendTest, PagedTableAndPoolsUseIndependentMultiSegmentWholeElementPlans) {
  LlmMetalPlanningLimits limits;
  limits.segment_capacity_bytes = 20;
  limits.segment_slots_per_pool = 32;
  const LlmGeometry geometry = paged_geometry(41, 29, 4, 2, 4, 1);
  ASSERT_TRUE(geometry.valid) << geometry.reason_code;
  ASSERT_EQ(geometry.kv_block_bytes, 4U);
  ASSERT_EQ(geometry.total_physical_blocks, 64U);
  ASSERT_EQ(geometry.block_table_entries, 32U);

  const LlmMetalExecutionPlan plan = build_llm_metal_execution_plan(resource_request(geometry, limits));
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  ASSERT_TRUE(plan.resources.table_segments.has_value());
  EXPECT_EQ(plan.resources.k_segments.segment_count, 13U);
  EXPECT_EQ(plan.resources.k_segments.segment_lengths.back(), 16U);
  EXPECT_TRUE(std::all_of(plan.resources.k_segments.segment_lengths.begin(),
                          plan.resources.k_segments.segment_lengths.end(),
                          [](size_t length) { return length % 4 == 0 && length <= 20; }));
  EXPECT_EQ(plan.resources.table_segments->segment_count, 7U);
  EXPECT_EQ(plan.resources.table_segments->segment_lengths, (std::vector<size_t>{20, 20, 20, 20, 20, 20, 8}));
  EXPECT_TRUE(std::all_of(plan.resources.table_segments->segment_lengths.begin(),
                          plan.resources.table_segments->segment_lengths.end(),
                          [](size_t length) { return length % sizeof(uint32_t) == 0 && length <= 20; }));
}

TEST(LlmMetalBackendTest, PagedIdentityBindsTailGeometryWhenPhysicalResourcesMatch) {
  const LlmGeometry five_token_geometry = paged_geometry(4097, 5, 4);
  const LlmGeometry seven_token_geometry = paged_geometry(4097, 7, 4);
  ASSERT_TRUE(five_token_geometry.valid) << five_token_geometry.reason_code;
  ASSERT_TRUE(seven_token_geometry.valid) << seven_token_geometry.reason_code;
  ASSERT_EQ(five_token_geometry.k_mapping_bytes, seven_token_geometry.k_mapping_bytes);
  ASSERT_EQ(five_token_geometry.v_mapping_bytes, seven_token_geometry.v_mapping_bytes);
  ASSERT_EQ(five_token_geometry.block_table_bytes, seven_token_geometry.block_table_bytes);

  const LlmMetalExecutionPlan five_token_plan = build_llm_metal_execution_plan(resource_request(five_token_geometry));
  const LlmMetalExecutionPlan seven_token_plan = build_llm_metal_execution_plan(resource_request(seven_token_geometry));
  ASSERT_TRUE(five_token_plan.valid) << five_token_plan.reason_code;
  ASSERT_TRUE(seven_token_plan.valid) << seven_token_plan.reason_code;
  ASSERT_TRUE(five_token_plan.resources.paged_layout.has_value());
  ASSERT_TRUE(seven_token_plan.resources.paged_layout.has_value());
  ASSERT_EQ(five_token_plan.resources.planned_resources.size(), seven_token_plan.resources.planned_resources.size());
  for (size_t index = 0; index < five_token_plan.resources.planned_resources.size(); ++index) {
    const LlmMetalPlannedResource& five_token_resource = five_token_plan.resources.planned_resources[index];
    const LlmMetalPlannedResource& seven_token_resource = seven_token_plan.resources.planned_resources[index];
    EXPECT_EQ(five_token_resource.pool, seven_token_resource.pool);
    EXPECT_EQ(five_token_resource.pool_index, seven_token_resource.pool_index);
    EXPECT_EQ(five_token_resource.length_bytes, seven_token_resource.length_bytes);
    EXPECT_EQ(five_token_resource.persistent, seven_token_resource.persistent);
  }
  EXPECT_NE(five_token_plan.resources.paged_layout->geometry_identity,
            seven_token_plan.resources.paged_layout->geometry_identity);
  EXPECT_NE(five_token_plan.resources.identity, seven_token_plan.resources.identity);
  EXPECT_NE(five_token_plan.identity, seven_token_plan.identity);
}

TEST(LlmMetalBackendTest, PagedPlannerRejectsOversizedBlocksAndMissingOrMismatchedLayouts) {
  LlmMetalPlanningLimits limits;
  limits.segment_capacity_bytes = 63;
  limits.segment_slots_per_pool = 8;
  const LlmGeometry geometry = paged_geometry(64, 17, 16, 1, 1, 4);
  ASSERT_TRUE(geometry.valid) << geometry.reason_code;
  ASSERT_EQ(geometry.kv_block_bytes, 64U);

  LlmMetalResourcePlanRequest request = resource_request(geometry, limits);
  const LlmMetalExecutionPlan oversized = build_llm_metal_execution_plan(request);
  EXPECT_FALSE(oversized.valid);
  EXPECT_EQ(oversized.reason_code, LlmBackendReason::PAGED_BLOCK_EXCEEDS_SEGMENT_CAPACITY);

  request.limits.segment_capacity_bytes = 64;
  request.paged_layout.reset();
  const LlmMetalExecutionPlan missing = build_llm_metal_execution_plan(request);
  EXPECT_FALSE(missing.valid);
  EXPECT_EQ(missing.reason_code, LlmMetalPlanReason::PAGED_LAYOUT_REQUIRED);

  request.paged_layout = paged_layout_for(paged_geometry(64, 18, 16, 1, 1, 4));
  const LlmMetalExecutionPlan mismatched = build_llm_metal_execution_plan(request);
  EXPECT_FALSE(mismatched.valid);
  EXPECT_EQ(mismatched.reason_code, LlmMetalPlanReason::PAGED_LAYOUT_MISMATCH);

  LlmGeometry malformed = geometry;
  malformed.decode.reset();
  request.geometry = malformed;
  const LlmMetalExecutionPlan invalid_geometry = build_llm_metal_execution_plan(request);
  EXPECT_FALSE(invalid_geometry.valid);
  EXPECT_EQ(invalid_geometry.reason_code, LlmMetalPlanReason::INVALID_GEOMETRY);
}

TEST(LlmMetalBackendTest, ContiguousSegmentCapFailureUsesCanonicalMetalReason) {
  LlmMetalPlanningLimits limits;
  limits.segment_capacity_bytes = 100;
  limits.segment_slots_per_pool = 2;
  const LlmGeometry geometry = contiguous_geometry(201);
  const LlmMetalExecutionPlan plan = build_llm_metal_execution_plan(resource_request(geometry, limits));
  EXPECT_FALSE(plan.valid);
  EXPECT_EQ(plan.reason_code, LlmBackendReason::SEGMENT_COUNT_CAP_EXCEEDED);
}

TEST(LlmMetalBackendTest, RuntimeEncoderLengthAlignmentAndFirstAdmissionAreExact) {
  const LlmGeometry geometry = contiguous_geometry(101);
  ASSERT_TRUE(geometry.valid) << geometry.reason_code;
  LlmMetalResourcePlanRequest request = resource_request(geometry);
  request.argument_buffer_encoded_length = 65;
  request.argument_buffer_alignment = 64;
  request.max_buffer_length = Constants::LLM_METAL_SEGMENT_CAPACITY_BYTES;
  request.available_memory_bytes = 10000;
  request.host_mapping_granularity_bytes = 1;
  request.additional_owned_bytes = 17;

  const LlmMetalExecutionPlan plan = build_llm_metal_execution_plan(request);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  EXPECT_EQ(plan.resources.argument_buffer_encoded_length, 65U);
  EXPECT_EQ(plan.resources.argument_buffer_alignment, 64U);
  EXPECT_EQ(plan.msl_revision, "llm-metal-foundation-msl23-v1");
  EXPECT_EQ(plan.msl_source_sha256, kCanonicalKernelSourceSha256);
  const LlmMetalPlannedResource* argument = find_planned_resource(plan.resources, LlmMetalResourcePool::ArgumentBuffer);
  ASSERT_NE(argument, nullptr);
  EXPECT_EQ(argument->length_bytes, 65U);
  EXPECT_EQ(plan.resources.persistent_resource_length_bytes,
            101U + 1U + 1U + 65U + Constants::LLM_METAL_STATUS_BUFFER_BYTES);
  EXPECT_EQ(plan.resources.transient_peak_bytes, 0U);
  EXPECT_EQ(plan.resources.known_owned_peak_bytes, plan.resources.persistent_resource_length_bytes + 17U);
  EXPECT_EQ(plan.resources.admitted_budget_bytes,
            static_cast<size_t>(static_cast<long double>(10000) * Constants::MEMORY_LIMIT_FACTOR));

  request.argument_buffer_encoded_length = 0;
  EXPECT_EQ(build_llm_metal_execution_plan(request).reason_code, LlmMetalPlanReason::ARGUMENT_ENCODER_LENGTH_ZERO);
  request.argument_buffer_encoded_length = 65;
  for (size_t alignment : {size_t{0}, size_t{3}}) {
    request.argument_buffer_alignment = alignment;
    const LlmMetalExecutionPlan invalid = build_llm_metal_execution_plan(request);
    SCOPED_TRACE(alignment);
    EXPECT_FALSE(invalid.valid);
    EXPECT_EQ(invalid.reason_code, LlmMetalPlanReason::ARGUMENT_ENCODER_ALIGNMENT_INVALID);
  }
  request.argument_buffer_alignment = 64;
  request.host_mapping_granularity_bytes = 0;
  const LlmMetalExecutionPlan zero_granularity = build_llm_metal_execution_plan(request);
  EXPECT_FALSE(zero_granularity.valid);
  EXPECT_EQ(zero_granularity.reason_code, LlmWorkPlanReason::MAPPING_GRANULARITY_ZERO);
}

TEST(LlmMetalBackendTest, FirstAdmissionAppliesTheHardBudgetAtTheExactKnownOwnedPeak) {
  const LlmGeometry geometry = contiguous_geometry(101);
  LlmMetalResourcePlanRequest request = resource_request(geometry);
  request.argument_buffer_encoded_length = 65;
  request.argument_buffer_alignment = 64;
  request.max_buffer_length = Constants::LLM_METAL_SEGMENT_CAPACITY_BYTES;
  request.host_mapping_granularity_bytes = 1;
  request.additional_owned_bytes = 17;
  request.available_memory_bytes = 10000;
  const LlmMetalExecutionPlan baseline = build_llm_metal_execution_plan(request);
  ASSERT_TRUE(baseline.valid) << baseline.reason_code;

  const size_t required = baseline.resources.known_owned_peak_bytes;
  size_t first_admitting_available = 0;
  for (size_t available = required; available < 2 * required; ++available) {
    const size_t admitted = static_cast<size_t>(static_cast<long double>(available) * Constants::MEMORY_LIMIT_FACTOR);
    if (admitted >= required) {
      first_admitting_available = available;
      break;
    }
  }
  ASSERT_GT(first_admitting_available, 0U);

  request.available_memory_bytes = first_admitting_available - 1;
  const LlmMetalExecutionPlan rejected = build_llm_metal_execution_plan(request);
  EXPECT_FALSE(rejected.valid);
  EXPECT_EQ(rejected.reason_code, LlmMetalPlanReason::MEMORY_BUDGET_EXCEEDED);

  request.available_memory_bytes = first_admitting_available;
  const LlmMetalExecutionPlan admitted = build_llm_metal_execution_plan(request);
  ASSERT_TRUE(admitted.valid) << admitted.reason_code;
  EXPECT_EQ(admitted.resources.known_owned_peak_bytes, required);
  EXPECT_GE(admitted.resources.admitted_budget_bytes, required);
}

TEST(LlmMetalBackendTest, CommittedAdmissionUsesNullableAllocatedSizeExactlyOncePerResource) {
  LlmMetalPlanningLimits limits;
  limits.segment_capacity_bytes = 100;
  limits.segment_slots_per_pool = 32;
  const LlmGeometry geometry = paged_geometry(101, 49, 16, 2, 2, 3);
  LlmMetalResourcePlanRequest request = resource_request(geometry, limits);
  request.available_memory_bytes = 32 * kGiB;
  request.additional_owned_bytes = 37;
  const LlmMetalExecutionPlan execution = build_llm_metal_execution_plan(request);
  ASSERT_TRUE(execution.valid) << execution.reason_code;

  std::vector<LlmMetalAllocatedResource> allocations = exact_allocations(execution.resources);
  size_t expected_committed = 0;
  size_t expected_rounding = 0;
  for (size_t index = 0; index < allocations.size(); ++index) {
    const size_t rounding = index % 2 == 0 ? index + 1 : 0;
    if (rounding != 0) {
      allocations[index].allocated_size_bytes = allocations[index].length_bytes + rounding;
    }
    expected_committed += allocations[index].length_bytes + rounding;
    expected_rounding += rounding;
  }

  const LlmMetalCommittedAdmission admission = evaluate_llm_metal_committed_admission(execution.resources, allocations);
  ASSERT_TRUE(admission.valid) << admission.reason_code;
  EXPECT_EQ(admission.committed_resource_bytes, expected_committed);
  EXPECT_EQ(admission.resource_rounding_bytes, expected_rounding);
  EXPECT_EQ(admission.transient_peak_bytes, execution.resources.transient_peak_bytes);
  EXPECT_EQ(admission.additional_owned_bytes, execution.resources.additional_owned_bytes);
  EXPECT_EQ(admission.known_owned_peak_bytes, expected_committed + execution.resources.host_permutation_mapping_bytes +
                                                  execution.resources.permutation_validation_bitset_bytes +
                                                  execution.resources.additional_owned_bytes);
  EXPECT_NE(admission.known_owned_peak_bytes, expected_committed + execution.resources.transient_peak_bytes +
                                                  execution.resources.additional_owned_bytes +
                                                  geometry.k_layout_padding_bytes + geometry.v_layout_padding_bytes);
}

TEST(LlmMetalBackendTest, CommittedAdmissionRejectsMissingDuplicateOrMismatchedResources) {
  const LlmGeometry geometry = contiguous_geometry(101);
  const LlmMetalExecutionPlan execution = build_llm_metal_execution_plan(resource_request(geometry));
  ASSERT_TRUE(execution.valid) << execution.reason_code;
  const std::vector<LlmMetalAllocatedResource> baseline = exact_allocations(execution.resources);
  ASSERT_FALSE(baseline.empty());

  std::vector<std::vector<LlmMetalAllocatedResource>> mismatches;
  mismatches.push_back(baseline);
  mismatches.back().pop_back();
  mismatches.push_back(baseline);
  mismatches.back().push_back(baseline.front());
  mismatches.push_back(baseline);
  ++mismatches.back().front().pool_index;
  mismatches.push_back(baseline);
  ++mismatches.back().front().length_bytes;
  mismatches.push_back(baseline);
  mismatches.back().front().allocated_size_bytes = mismatches.back().front().length_bytes - 1;

  for (size_t index = 0; index < mismatches.size(); ++index) {
    const LlmMetalCommittedAdmission invalid =
        evaluate_llm_metal_committed_admission(execution.resources, mismatches[index]);
    SCOPED_TRACE(index);
    EXPECT_FALSE(invalid.valid);
    EXPECT_EQ(invalid.reason_code, index == mismatches.size() - 1 ? LlmMetalPlanReason::RESOURCE_LENGTH_OVERFLOW
                                                                  : LlmBackendReason::PLAN_RESOURCE_IDENTITY_MISMATCH);
  }
}

TEST(LlmMetalBackendTest, GridPlanUsesInjectedWidthAndAllBoundaryCaps) {
  LlmMetalGridRequest request;
  request.owner_count = 1;
  request.visit_bytes = 64 * Constants::LLM_METAL_VECTOR_WIDTH_BYTES;
  request.work_units = 4;
  request.paged_semantic_lookups = 10;
  request.pipeline = {16, 128};
  request.limits.threads_per_threadgroup_cap = 64;
  request.limits.maximum_threadgroups_per_grid = 3;
  request.limits.maximum_owner_ordinals_per_threadgroup = 2;
  request.limits.maximum_vector_iterations_per_lane_per_visit = 2;
  request.limits.maximum_work_units_per_dispatch = 4;
  request.limits.maximum_paged_semantic_lookups_per_task = 10;

  for (size_t owner_count : {size_t{0}, size_t{1}, size_t{2}, size_t{3}, size_t{4}}) {
    request.owner_count = owner_count;
    const LlmMetalGridPlan plan = build_llm_metal_grid_plan(request);
    SCOPED_TRACE(owner_count);
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    EXPECT_EQ(plan.threads_per_threadgroup, 64U);
    EXPECT_EQ(plan.actual_threadgroups, std::min(owner_count, size_t{3}));
    EXPECT_EQ(plan.owner_ordinals_per_threadgroup, owner_count == 0 ? 0U : owner_count == 4 ? 2U : 1U);
    EXPECT_EQ(plan.vector_iterations_per_lane_per_visit, 1U);
  }

  request.owner_count = 6;
  request.visit_bytes = 2 * 64 * Constants::LLM_METAL_VECTOR_WIDTH_BYTES;
  const LlmMetalGridPlan exact = build_llm_metal_grid_plan(request);
  ASSERT_TRUE(exact.valid) << exact.reason_code;
  EXPECT_EQ(exact.actual_threadgroups, 3U);
  EXPECT_EQ(exact.owner_ordinals_per_threadgroup, 2U);
  EXPECT_EQ(exact.vector_iterations_per_lane_per_visit, 2U);
  LlmMetalGridRequest different_cap = request;
  ++different_cap.limits.maximum_work_units_per_dispatch;
  const LlmMetalGridPlan different_identity = build_llm_metal_grid_plan(different_cap);
  ASSERT_TRUE(different_identity.valid) << different_identity.reason_code;
  EXPECT_NE(different_identity.identity, exact.identity);

  request.owner_count = 7;
  EXPECT_EQ(build_llm_metal_grid_plan(request).reason_code, LlmMetalPlanReason::OWNER_STRIDE_CAP_EXCEEDED);
  request.owner_count = 6;
  ++request.visit_bytes;
  EXPECT_EQ(build_llm_metal_grid_plan(request).reason_code, LlmMetalPlanReason::VECTOR_ITERATION_CAP_EXCEEDED);
  --request.visit_bytes;
  ++request.work_units;
  EXPECT_EQ(build_llm_metal_grid_plan(request).reason_code, LlmMetalPlanReason::WORK_UNITS_PER_DISPATCH_CAP_EXCEEDED);
  --request.work_units;
  ++request.paged_semantic_lookups;
  EXPECT_EQ(build_llm_metal_grid_plan(request).reason_code, LlmMetalPlanReason::SEMANTIC_VISIT_CAP_EXCEEDED);
}

TEST(LlmMetalBackendTest, GridPlanReportsExactCyclicThreadgroupOwnerCostsPastGridCap) {
  LlmMetalGridRequest request;
  request.owner_count = 7;
  request.visit_bytes = 1;
  request.work_units = 1;
  request.paged_semantic_lookups = 7;
  request.owner_accounted_bytes = {1, 10, 100, 1000, 10000, 100000, 1000000};
  request.pipeline = {16, 64};
  request.limits.threads_per_threadgroup_cap = 64;
  request.limits.maximum_threadgroups_per_grid = 3;
  request.limits.maximum_owner_ordinals_per_threadgroup = 3;
  request.limits.maximum_vector_iterations_per_lane_per_visit = 1;
  request.limits.maximum_paged_semantic_lookups_per_task = 7;
  request.limits.maximum_work_units_per_dispatch = 1;

  const LlmMetalGridPlan plan = build_llm_metal_grid_plan(request);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  EXPECT_EQ(plan.actual_threadgroups, 3U);
  EXPECT_EQ(plan.owner_ordinals_per_threadgroup, 3U);
  EXPECT_EQ(plan.threadgroup_accounted_bytes, (std::vector<size_t>{1001001, 10010, 100100}));
  EXPECT_EQ(plan.minimum_threadgroup_accounted_bytes, 10010U);
  EXPECT_EQ(plan.maximum_threadgroup_accounted_bytes, 1001001U);
  EXPECT_EQ(plan.threadgroup_accounted_imbalance_bytes, 990991U);
  EXPECT_EQ(
      std::accumulate(plan.threadgroup_accounted_bytes.begin(), plan.threadgroup_accounted_bytes.end(), size_t{0}),
      std::accumulate(request.owner_accounted_bytes.begin(), request.owner_accounted_bytes.end(), size_t{0}));
  EXPECT_FALSE(plan.identity.empty());

  LlmMetalGridRequest collision_left = request;
  collision_left.owner_count = 3;
  collision_left.paged_semantic_lookups = 3;
  collision_left.owner_accounted_bytes = {1, 2, 3};
  const LlmMetalGridPlan left = build_llm_metal_grid_plan(collision_left);
  ASSERT_TRUE(left.valid) << left.reason_code;
  LlmMetalGridRequest collision_right = collision_left;
  collision_right.owner_accounted_bytes = {2, 1, 3};
  const LlmMetalGridPlan right = build_llm_metal_grid_plan(collision_right);
  ASSERT_TRUE(right.valid) << right.reason_code;
  EXPECT_NE(left.threadgroup_accounted_bytes, right.threadgroup_accounted_bytes);
  EXPECT_EQ(left.minimum_threadgroup_accounted_bytes, right.minimum_threadgroup_accounted_bytes);
  EXPECT_EQ(left.maximum_threadgroup_accounted_bytes, right.maximum_threadgroup_accounted_bytes);
  EXPECT_EQ(left.threadgroup_accounted_imbalance_bytes, right.threadgroup_accounted_imbalance_bytes);
  EXPECT_NE(left.identity, right.identity);

  request.owner_accounted_bytes.pop_back();
  const LlmMetalGridPlan mismatch = build_llm_metal_grid_plan(request);
  EXPECT_FALSE(mismatch.valid);
  EXPECT_EQ(mismatch.reason_code, LlmMetalPlanReason::OWNER_COST_COUNT_MISMATCH);
}

TEST(LlmMetalBackendTest, FoundationParameterAbiAndLayoutProbeAreExact) {
  EXPECT_EQ(alignof(LlmMetalFoundationParams), 8U);
  EXPECT_EQ(sizeof(LlmMetalFoundationParams), 64U);
  const std::array<size_t, 10> offsets = {offsetof(LlmMetalFoundationParams, byte_count),
                                          offsetof(LlmMetalFoundationParams, source_offset_bytes),
                                          offsetof(LlmMetalFoundationParams, destination_offset_bytes),
                                          offsetof(LlmMetalFoundationParams, logical_base_bytes),
                                          offsetof(LlmMetalFoundationParams, pattern_seed),
                                          offsetof(LlmMetalFoundationParams, block_bytes),
                                          offsetof(LlmMetalFoundationParams, physical_blocks_per_layer),
                                          offsetof(LlmMetalFoundationParams, pattern_kind),
                                          offsetof(LlmMetalFoundationParams, probe_resource_kind),
                                          offsetof(LlmMetalFoundationParams, probe_resource_slot)};
  EXPECT_EQ(offsets, (std::array<size_t, 10>{0, 8, 16, 24, 32, 40, 48, 52, 56, 60}));

  LlmMetalFoundationParams parameters;
  parameters.byte_count = UINT64_C(0x0102030405060708);
  parameters.source_offset_bytes = UINT64_C(0x1112131415161718);
  parameters.destination_offset_bytes = UINT64_C(0x2122232425262728);
  parameters.logical_base_bytes = UINT64_C(0x3132333435363738);
  parameters.pattern_seed = UINT64_C(0x4142434445464748);
  parameters.block_bytes = UINT64_C(0x5152535455565758);
  parameters.physical_blocks_per_layer = UINT32_C(0x61626364);
  parameters.pattern_kind = UINT32_C(0x71727374);
  parameters.probe_resource_kind = 3;
  parameters.probe_resource_slot = 17;
  constexpr uint64_t kObservedResourceValue = UINT64_C(0xa5);
  LlmMetalLayoutProbeWords words = {
      1,
      64,
      8,
      10,
      0,
      8,
      16,
      24,
      32,
      40,
      48,
      52,
      56,
      60,
      parameters.byte_count,
      parameters.source_offset_bytes,
      parameters.destination_offset_bytes,
      parameters.logical_base_bytes,
      parameters.pattern_seed,
      parameters.block_bytes,
      parameters.physical_blocks_per_layer,
      parameters.pattern_kind,
      parameters.probe_resource_kind,
      parameters.probe_resource_slot,
      kObservedResourceValue,
      parameters.probe_resource_kind,
      parameters.probe_resource_slot,
      1025,
  };
  ASSERT_TRUE(validate_llm_metal_layout_probe(parameters, words, kObservedResourceValue));

  for (size_t index = 0; index < words.size(); ++index) {
    LlmMetalLayoutProbeWords corrupted = words;
    ++corrupted[index];
    SCOPED_TRACE(index);
    EXPECT_FALSE(validate_llm_metal_layout_probe(parameters, corrupted, kObservedResourceValue));
  }
}

TEST(LlmMetalBackendTest, CanonicalEmbeddedMslSourceHashIsFrozenLowercaseSha256) {
  const std::string digest = canonical_llm_metal_kernel_source_sha256();
  EXPECT_EQ(digest, kCanonicalKernelSourceSha256);
  ASSERT_EQ(digest.size(), 64U);
  EXPECT_TRUE(std::all_of(digest.begin(), digest.end(), [](unsigned char value) {
    return std::isdigit(value) != 0 || (value >= 'a' && value <= 'f');
  }));
}

TEST(LlmMetalBackendTest, CheckedMetalExecutionPlanAccessorRejectsBackendOrVariantMismatch) {
  LlmMemoryWorkPlan plan;
  EXPECT_EQ(get_llm_metal_execution_plan(plan), nullptr);

  plan.backend = LlmMemoryBackend::Metal;
  EXPECT_EQ(get_llm_metal_execution_plan(plan), nullptr);

  plan.backend_execution_plan = LlmMetalExecutionPlan{};
  EXPECT_NE(get_llm_metal_execution_plan(plan), nullptr);
  const LlmMemoryWorkPlan& const_plan = plan;
  EXPECT_NE(get_llm_metal_execution_plan(const_plan), nullptr);

  plan.backend = LlmMemoryBackend::Cpu;
  EXPECT_EQ(get_llm_metal_execution_plan(plan), nullptr);
  EXPECT_EQ(get_llm_metal_execution_plan(const_plan), nullptr);
}

TEST(LlmMetalBackendTest, FactoryAndInternalFoundationExposeMetalWithoutTimedEvidence) {
  std::unique_ptr<LlmBackend> factory_backend = create_llm_backend(LlmMemoryBackend::Metal);
  ASSERT_NE(factory_backend, nullptr);
  EXPECT_EQ(factory_backend->kind(), LlmMemoryBackend::Metal);
  const LlmMetalBackendEvidence* factory_metal = get_llm_metal_backend_evidence(factory_backend->evidence());
  ASSERT_NE(factory_metal, nullptr);
  EXPECT_FALSE(factory_metal->timed_results_available);
  LlmAuxiliaryPreflightView preflight;
  preflight.valid = true;
  preflight.backend = LlmMemoryBackend::Metal;
  const LlmBackendAuxiliaryEstimate preflight_estimate = factory_backend->calculate_auxiliary_estimate(preflight);
  EXPECT_FALSE(preflight_estimate.valid);
  EXPECT_EQ(preflight_estimate.reason_code, LlmBackendReason::BACKEND_NOT_ACTIVATED);

  LlmMemoryWorkPlan inactive_plan;
  inactive_plan.backend = LlmMemoryBackend::Metal;
  inactive_plan.backend_execution_plan = LlmMetalExecutionPlan{};
  const LlmBackendAuxiliaryEstimate plan_estimate = factory_backend->calculate_auxiliary_estimate(inactive_plan);
  EXPECT_FALSE(plan_estimate.valid);
  EXPECT_EQ(plan_estimate.reason_code, LlmBackendReason::BACKEND_NOT_ACTIVATED);

  std::unique_ptr<LlmBackend> backend = create_llm_metal_backend();
  ASSERT_NE(backend, nullptr);
  EXPECT_EQ(backend->kind(), LlmMemoryBackend::Metal);
  const LlmBackendEvidence& evidence = backend->evidence();
  EXPECT_EQ(evidence.backend, LlmMemoryBackend::Metal);
  const LlmMetalBackendEvidence* metal = get_llm_metal_backend_evidence(evidence);
  ASSERT_NE(metal, nullptr);
  EXPECT_FALSE(metal->timed_results_available);

  const LlmBackendLifecycleResult released = backend->release_resources();
  EXPECT_EQ(released.status, LlmBackendStatus::Ready);
  EXPECT_EQ(released.reason_code, LlmBackendReason::VALID);
  EXPECT_EQ(backend->release_resources().status, LlmBackendStatus::Ready);
}

class LlmMetalBackendIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    backend_ = create_llm_metal_backend();
    ASSERT_NE(backend_, nullptr);
    const LlmBackendLifecycleResult initialization = backend_->initialize(metal_config());
    if (initialization.status == LlmBackendStatus::Unsupported) {
      ASSERT_TRUE(is_stable_capability_unsupported_reason(initialization.reason_code));
      GTEST_SKIP() << "LLM Metal foundation unsupported: " << initialization.reason_code;
    }
    ASSERT_EQ(initialization.status, LlmBackendStatus::Ready) << initialization.reason_code;
  }

  void TearDown() override {
    if (backend_ != nullptr) {
      backend_->release_resources();
    }
  }

  const LlmMetalBackendEvidence& metal_evidence() const {
    const LlmMetalBackendEvidence* evidence = get_llm_metal_backend_evidence(backend_->evidence());
    if (evidence == nullptr) {
      throw std::logic_error("expected Metal-tagged backend evidence");
    }
    return *evidence;
  }

  LlmMemoryWorkPlan build_device_plan(LlmGeometry geometry, std::string identity,
                                      size_t available_memory_bytes = 2 * kGiB) const {
    const LlmMetalCapabilityEvidence& capability = metal_evidence().capability;
    LlmMetalResourcePlanRequest request = resource_request(geometry);
    request.argument_buffer_encoded_length = capability.argument_buffer_encoded_length;
    request.argument_buffer_alignment = capability.argument_buffer_alignment;
    request.max_buffer_length = capability.max_buffer_length;
    request.available_memory_bytes = available_memory_bytes;
    request.host_mapping_granularity_bytes = get_system_page_size_bytes();
    return make_metal_model_plan(geometry, build_llm_metal_execution_plan(request), std::move(identity));
  }

  std::unique_ptr<LlmBackend> backend_;
};

TEST_F(LlmMetalBackendIntegrationTest, RuntimeMslCapabilityEncoderAndLayoutProbeIntegration) {
  const LlmMetalCapabilityEvidence& capability = metal_evidence().capability;
  EXPECT_FALSE(capability.device_name.empty());
  EXPECT_TRUE(capability.has_unified_memory);
  EXPECT_TRUE(capability.required_apple7_family_supported);
  EXPECT_TRUE(capability.argument_buffers_tier2_supported);
  EXPECT_GE(capability.max_buffer_length, Constants::LLM_METAL_SEGMENT_CAPACITY_BYTES);
  EXPECT_EQ(capability.compilation_mode, "runtime-source");
  EXPECT_EQ(capability.msl_language_version, "2.3");
  EXPECT_EQ(capability.kernel_revision, "llm-metal-foundation-msl23-v1");
  EXPECT_EQ(capability.kernel_source_sha256, kCanonicalKernelSourceSha256);
  EXPECT_GT(capability.argument_buffer_encoded_length, 0U);
  EXPECT_GT(capability.argument_buffer_alignment, 0U);
  EXPECT_EQ(capability.argument_buffer_alignment & (capability.argument_buffer_alignment - 1), 0U);
  EXPECT_FALSE(capability.layout_probe_evaluated);
  EXPECT_FALSE(capability.layout_probe_valid);
  EXPECT_EQ(capability.layout_probe_resource_count, 0U);
  ASSERT_EQ(capability.foundation_pipelines.size(), kFoundationPipelineCount);
  const std::array<std::string_view, kFoundationPipelineCount> labels = {
      "membenchmark.llm-metal.pipeline.initialize", "membenchmark.llm-metal.pipeline.copy",
      "membenchmark.llm-metal.pipeline.layout-probe", "membenchmark.llm-metal.pipeline.validate-bytes",
      "membenchmark.llm-metal.pipeline.validate-table"};
  for (std::string_view label : labels) {
    const auto found = std::find_if(
        capability.foundation_pipelines.begin(), capability.foundation_pipelines.end(),
        [label](const LlmMetalPipelineEvidence& pipeline) { return std::string_view(pipeline.label) == label; });
    ASSERT_NE(found, capability.foundation_pipelines.end()) << label;
    EXPECT_GT(found->thread_execution_width, 0U);
    EXPECT_GT(found->max_total_threads_per_threadgroup, 0U);
  }
  EXPECT_FALSE(metal_evidence().timed_results_available);
}

TEST_F(LlmMetalBackendIntegrationTest, PrivateResourceInitializationAndNoTimedTaskIntegration) {
  LlmMemoryWorkPlan plan =
      build_device_plan(contiguous_geometry(4097, 17, 2, 2, 8), "llm-metal-phase8-small-device-plan");
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  const LlmBackendLifecycleResult resolved = backend_->resolve_execution_plan(plan);
  ASSERT_EQ(resolved.status, LlmBackendStatus::Ready) << resolved.reason_code;
  const LlmBackendLifecycleResult prepared = backend_->prepare_resources(plan);
  ASSERT_EQ(prepared.status, LlmBackendStatus::Ready)
      << prepared.reason_code << ": " << metal_evidence().resources.error.description;

  const LlmMetalResourceEvidence& resources = metal_evidence().resources;
  EXPECT_TRUE(resources.allocation_attempted);
  EXPECT_TRUE(resources.allocation_completed);
  EXPECT_TRUE(resources.initialization_completed);
  EXPECT_TRUE(resources.table_upload_completed);
  EXPECT_TRUE(resources.table_validation_completed);
  EXPECT_FALSE(resources.table_permutation.has_value());
  EXPECT_TRUE(resources.post_validation_completed);
  EXPECT_TRUE(resources.cpu_sample_readback_validation_completed);
  EXPECT_TRUE(resources.candidate_cleanup_completed);
  EXPECT_TRUE(resources.resources_published);
  EXPECT_FALSE(resources.resources.empty());
  EXPECT_TRUE(metal_evidence().capability.layout_probe_evaluated);
  EXPECT_TRUE(metal_evidence().capability.layout_probe_valid);
  EXPECT_EQ(metal_evidence().capability.layout_probe_resource_count, 4U);
  for (const LlmMetalResourceMetadata& resource : resources.resources) {
    SCOPED_TRACE(resource.label);
    EXPECT_GT(resource.length_bytes, 0U);
    EXPECT_LE(resource.length_bytes, metal_evidence().capability.max_buffer_length);
    EXPECT_EQ(resource.hazard_tracking_mode, "tracked");
    if (resource.pool == "weight" || resource.pool == "k" || resource.pool == "v" || resource.pool == "block_table") {
      EXPECT_EQ(resource.storage_mode, "private");
    } else {
      EXPECT_EQ(resource.storage_mode, "shared");
    }
  }
  EXPECT_EQ(resources.known_owned_peak_bytes,
            resources.committed_resource_bytes + resources.transient_peak_bytes + resources.additional_owned_bytes);
  EXPECT_FALSE(metal_evidence().timed_results_available);

  const size_t published_metadata_count = resources.resources.size();
  const uint64_t allocation_peak = resources.current_allocated_size_peak;
  const LlmBackendLifecycleResult prepared_again = backend_->prepare_resources(plan);
  EXPECT_EQ(prepared_again.status, LlmBackendStatus::Ready);
  EXPECT_EQ(metal_evidence().resources.resources.size(), published_metadata_count);
  EXPECT_EQ(metal_evidence().resources.current_allocated_size_peak, allocation_peak);

  const LlmScenarioWorkPlan scenario = one_work_unit_scenario(plan);
  LlmRunnerTaskContext context;
  context.kind = LlmRunnerTaskKind::Measurement;
  context.purpose = "phase-8-no-timed-task";
  context.scenario = scenario.scenario;
  context.attempt_index = 0;
  context.loop_index = 0;
  context.order_position = 0;
  const LlmTaskExecutionResult result = backend_->execute_task(plan, scenario, context);
  EXPECT_EQ(result.status, LlmTaskExecutionStatus::Unsupported);
  EXPECT_EQ(result.reason_code, LlmBackendReason::TASK_UNSUPPORTED);
  EXPECT_FALSE(result.timing.evaluated);
  EXPECT_FALSE(result.timing.valid);
  const LlmMetalTaskEvidence* task_evidence = get_llm_metal_task_evidence(result);
  ASSERT_NE(task_evidence, nullptr);
  EXPECT_FALSE(task_evidence->timed_pipeline_available);
  EXPECT_FALSE(task_evidence->timing_evaluated);
}

TEST_F(LlmMetalBackendIntegrationTest, PagedPrivateTableUploadValidationAndTier2SlotsIntegration) {
  const LlmGeometry geometry = paged_geometry(4097, 17, 4, 2, 2, 8);
  ASSERT_TRUE(geometry.valid) << geometry.reason_code;
  const LlmKvLayoutPlan layout = paged_layout_for(geometry);
  ASSERT_TRUE(layout.valid) << layout.reason_code;
  LlmMemoryWorkPlan plan = build_device_plan(geometry, "llm-metal-phase8-paged-device-plan");
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  const LlmKvBlockTable expected_table = materialize_llm_kv_block_table(
      layout, derive_llm_kv_permutation_seed(plan.base_seed), Constants::LLM_KV_BLOCK_TABLE_HASH_CHUNK_ENTRIES);
  ASSERT_TRUE(expected_table.valid) << expected_table.reason_code;

  ASSERT_EQ(backend_->resolve_execution_plan(plan).status, LlmBackendStatus::Ready);
  const LlmBackendLifecycleResult prepared = backend_->prepare_resources(plan);
  ASSERT_EQ(prepared.status, LlmBackendStatus::Ready)
      << prepared.reason_code << ": " << metal_evidence().resources.error.description;

  const LlmMetalResourceEvidence& resources = metal_evidence().resources;
  EXPECT_TRUE(resources.table_upload_completed);
  EXPECT_TRUE(resources.table_validation_completed);
  EXPECT_TRUE(resources.initialization_completed);
  EXPECT_TRUE(resources.post_validation_completed);
  EXPECT_TRUE(resources.cpu_sample_readback_validation_completed);
  ASSERT_TRUE(resources.table_permutation.has_value());
  EXPECT_EQ(resources.table_permutation->algorithm_version, expected_table.permutation.algorithm_version);
  EXPECT_EQ(resources.table_permutation->domain, expected_table.permutation.domain);
  EXPECT_EQ(resources.table_permutation->domain_uint64_hex, expected_table.permutation.domain_uint64_hex);
  EXPECT_EQ(resources.table_permutation->resolved_seed, expected_table.permutation.resolved_seed);
  EXPECT_EQ(resources.table_permutation->entry_count, expected_table.permutation.entry_count);
  EXPECT_EQ(resources.table_permutation->sha256, expected_table.permutation.sha256);
  EXPECT_EQ(resources.table_permutation->identity, expected_table.permutation.identity);
  EXPECT_EQ(metal_evidence().capability.layout_probe_resource_count, 5U);

  const auto table =
      std::find_if(resources.resources.begin(), resources.resources.end(),
                   [](const LlmMetalResourceMetadata& resource) { return resource.pool == "block_table"; });
  ASSERT_NE(table, resources.resources.end());
  EXPECT_EQ(table->storage_mode, "private");
  EXPECT_EQ(table->length_bytes, layout.memory.block_table_bytes);
  const auto staging =
      std::find_if(resources.resources.begin(), resources.resources.end(),
                   [](const LlmMetalResourceMetadata& resource) { return resource.pool == "staging"; });
  ASSERT_NE(staging, resources.resources.end());
  EXPECT_EQ(staging->storage_mode, "shared");
  EXPECT_EQ(staging->length_bytes, layout.memory.block_table_bytes);
  EXPECT_FALSE(metal_evidence().timed_results_available);
}

TEST_F(LlmMetalBackendIntegrationTest, MutatedResourcePlanIsRejectedBeforeAllocationIntegration) {
  LlmMemoryWorkPlan plan = build_device_plan(contiguous_geometry(4097), "llm-metal-phase8-mutated-resource-plan");
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  LlmMetalExecutionPlan* execution = get_llm_metal_execution_plan(plan);
  ASSERT_NE(execution, nullptr);
  ASSERT_FALSE(execution->resources.planned_resources.empty());
  ++execution->resources.planned_resources.front().length_bytes;

  const LlmBackendLifecycleResult resolved = backend_->resolve_execution_plan(plan);
  EXPECT_EQ(resolved.status, LlmBackendStatus::Failed);
  EXPECT_EQ(resolved.reason_code, LlmBackendReason::PLAN_RESOURCE_IDENTITY_MISMATCH);
  EXPECT_FALSE(metal_evidence().resources.allocation_attempted);
}

TEST_F(LlmMetalBackendIntegrationTest, RuntimePageSizeRejectsASelfConsistentUnderRoundedPlanIntegration) {
  const size_t runtime_page_size = get_system_page_size_bytes();
  ASSERT_GT(runtime_page_size, 0U);
  const LlmGeometry geometry = paged_geometry(4097, 17, 4, 2, 2, 8);
  const LlmMetalCapabilityEvidence& capability = metal_evidence().capability;
  LlmMetalResourcePlanRequest request = resource_request(geometry);
  request.argument_buffer_encoded_length = capability.argument_buffer_encoded_length;
  request.argument_buffer_alignment = capability.argument_buffer_alignment;
  request.max_buffer_length = capability.max_buffer_length;
  request.available_memory_bytes = 2 * kGiB;
  request.host_mapping_granularity_bytes = runtime_page_size == 1 ? 2 : 1;
  LlmMemoryWorkPlan plan = make_metal_model_plan(geometry, build_llm_metal_execution_plan(request),
                                                 "llm-metal-phase8-wrong-host-granularity-plan");
  ASSERT_TRUE(plan.valid) << plan.reason_code;

  const LlmBackendLifecycleResult resolved = backend_->resolve_execution_plan(plan);
  EXPECT_EQ(resolved.status, LlmBackendStatus::Failed);
  EXPECT_EQ(resolved.reason_code, LlmBackendReason::EXECUTION_PLAN_MISMATCH);
  EXPECT_FALSE(metal_evidence().resources.allocation_attempted);
}

TEST_F(LlmMetalBackendIntegrationTest, FailedReresolutionInvalidatesThePreviouslyResolvedCandidateIntegration) {
  LlmMemoryWorkPlan original =
      build_device_plan(contiguous_geometry(4097), "llm-metal-phase8-failed-reresolution-plan");
  LlmMemoryWorkPlan mutated = build_device_plan(contiguous_geometry(4097), "llm-metal-phase8-failed-reresolution-plan");
  ASSERT_TRUE(original.valid) << original.reason_code;
  ASSERT_TRUE(mutated.valid) << mutated.reason_code;
  ASSERT_EQ(backend_->resolve_execution_plan(original).status, LlmBackendStatus::Ready);
  LlmMetalExecutionPlan* mutated_execution = get_llm_metal_execution_plan(mutated);
  ASSERT_NE(mutated_execution, nullptr);
  ASSERT_FALSE(mutated_execution->resources.planned_resources.empty());
  ++mutated_execution->resources.planned_resources.front().length_bytes;

  const LlmBackendLifecycleResult rejected = backend_->resolve_execution_plan(mutated);
  EXPECT_EQ(rejected.status, LlmBackendStatus::Failed);
  EXPECT_EQ(rejected.reason_code, LlmBackendReason::PLAN_RESOURCE_IDENTITY_MISMATCH);
  const LlmBackendLifecycleResult prepared = backend_->prepare_resources(original);
  EXPECT_EQ(prepared.status, LlmBackendStatus::Failed);
  EXPECT_EQ(prepared.reason_code, LlmBackendReason::EXECUTION_PLAN_MISMATCH);
  EXPECT_FALSE(metal_evidence().resources.allocation_attempted);
  EXPECT_FALSE(metal_evidence().resources.resources_published);
}

TEST_F(LlmMetalBackendIntegrationTest, PreparedResourcesRejectPlanReresolutionIntegration) {
  LlmMemoryWorkPlan plan = build_device_plan(contiguous_geometry(4097), "llm-metal-phase8-prepared-reresolution-plan");
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  ASSERT_EQ(backend_->resolve_execution_plan(plan).status, LlmBackendStatus::Ready);
  ASSERT_EQ(backend_->prepare_resources(plan).status, LlmBackendStatus::Ready);
  ASSERT_TRUE(metal_evidence().resources.resources_published);

  const LlmBackendLifecycleResult reresolved = backend_->resolve_execution_plan(plan);
  EXPECT_EQ(reresolved.status, LlmBackendStatus::Failed);
  EXPECT_EQ(reresolved.reason_code, LlmBackendReason::EXECUTION_PLAN_MISMATCH);
  EXPECT_TRUE(metal_evidence().resources.resources_published);

  const LlmBackendLifecycleResult prepared_again = backend_->prepare_resources(plan);
  EXPECT_EQ(prepared_again.status, LlmBackendStatus::Ready);
  EXPECT_TRUE(metal_evidence().resources.resources_published);
}

TEST_F(LlmMetalBackendIntegrationTest, ResourceLargerThanCanonicalSegmentUsesPrivateSegmentsIntegration) {
  const size_t capacity = Constants::LLM_METAL_SEGMENT_CAPACITY_BYTES;
  LlmMemoryWorkPlan plan =
      build_device_plan(contiguous_geometry(capacity + 1), "llm-metal-phase8-over-canonical-segment-plan", 2 * kGiB);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  ASSERT_EQ(backend_->resolve_execution_plan(plan).status, LlmBackendStatus::Ready);
  const LlmBackendLifecycleResult prepared = backend_->prepare_resources(plan);
  ASSERT_EQ(prepared.status, LlmBackendStatus::Ready)
      << prepared.reason_code << ": " << metal_evidence().resources.error.description;

  std::vector<const LlmMetalResourceMetadata*> weight_segments;
  for (const LlmMetalResourceMetadata& resource : metal_evidence().resources.resources) {
    if (resource.pool == "weight") {
      weight_segments.push_back(&resource);
    }
  }
  ASSERT_EQ(weight_segments.size(), 2U);
  std::sort(weight_segments.begin(), weight_segments.end(),
            [](const LlmMetalResourceMetadata* left, const LlmMetalResourceMetadata* right) {
              return left->pool_index < right->pool_index;
            });
  EXPECT_EQ(weight_segments[0]->pool_index, 0U);
  EXPECT_EQ(weight_segments[0]->length_bytes, capacity);
  EXPECT_EQ(weight_segments[1]->pool_index, 1U);
  EXPECT_EQ(weight_segments[1]->length_bytes, 1U);
  for (const LlmMetalResourceMetadata* resource : weight_segments) {
    EXPECT_EQ(resource->storage_mode, "private");
    EXPECT_EQ(resource->hazard_tracking_mode, "tracked");
    EXPECT_LE(resource->length_bytes, metal_evidence().capability.max_buffer_length);
  }
  EXPECT_TRUE(metal_evidence().resources.initialization_completed);
  EXPECT_TRUE(metal_evidence().resources.post_validation_completed);
  EXPECT_TRUE(metal_evidence().resources.cpu_sample_readback_validation_completed);
  EXPECT_EQ(metal_evidence().capability.layout_probe_resource_count, 5U);
}

TEST(LlmMetalBackendFailureInjectionIntegrationTest,
     CandidateAllocationFailureCleansUpAndReleaseIsIdempotentIntegration) {
  LlmMetalBackendTestHooks hooks;
  hooks.fail_allocation_after = 1;
  std::unique_ptr<LlmBackend> backend = create_llm_metal_backend_for_testing(hooks);
  ASSERT_NE(backend, nullptr);
  const LlmBackendLifecycleResult initialization = backend->initialize(metal_config());
  if (initialization.status == LlmBackendStatus::Unsupported) {
    ASSERT_TRUE(is_stable_capability_unsupported_reason(initialization.reason_code));
    GTEST_SKIP() << "LLM Metal foundation unsupported: " << initialization.reason_code;
  }
  ASSERT_EQ(initialization.status, LlmBackendStatus::Ready) << initialization.reason_code;
  const LlmMetalBackendEvidence* initialized_evidence = get_llm_metal_backend_evidence(backend->evidence());
  ASSERT_NE(initialized_evidence, nullptr);

  const LlmGeometry geometry = contiguous_geometry(4097, 17, 2, 2, 8);
  LlmMetalResourcePlanRequest request = resource_request(geometry);
  request.argument_buffer_encoded_length = initialized_evidence->capability.argument_buffer_encoded_length;
  request.argument_buffer_alignment = initialized_evidence->capability.argument_buffer_alignment;
  request.max_buffer_length = initialized_evidence->capability.max_buffer_length;
  request.available_memory_bytes = 2 * kGiB;
  request.host_mapping_granularity_bytes = get_system_page_size_bytes();
  LlmMemoryWorkPlan plan = make_metal_model_plan(geometry, build_llm_metal_execution_plan(request),
                                                 "llm-metal-phase8-allocation-failure-plan");
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  ASSERT_EQ(backend->resolve_execution_plan(plan).status, LlmBackendStatus::Ready);

  const LlmBackendLifecycleResult prepared = backend->prepare_resources(plan);
  EXPECT_EQ(prepared.status, LlmBackendStatus::Failed);
  EXPECT_EQ(prepared.reason_code, LlmBackendReason::METAL_RESOURCE_ALLOCATION_FAILED);
  const LlmMetalBackendEvidence* failed_evidence = get_llm_metal_backend_evidence(backend->evidence());
  ASSERT_NE(failed_evidence, nullptr);
  EXPECT_TRUE(failed_evidence->resources.allocation_attempted);
  EXPECT_FALSE(failed_evidence->resources.allocation_completed);
  EXPECT_FALSE(failed_evidence->resources.initialization_completed);
  EXPECT_TRUE(failed_evidence->resources.candidate_cleanup_completed);
  EXPECT_FALSE(failed_evidence->resources.resources_published);
  ASSERT_EQ(failed_evidence->resources.resources.size(), 1U);
  EXPECT_EQ(failed_evidence->resources.resources.front().pool, "weight");
  EXPECT_EQ(failed_evidence->resources.resources.front().pool_index, 0U);
  EXPECT_LE(failed_evidence->resources.current_allocated_size_after_release,
            failed_evidence->resources.current_allocated_size_peak);

  for (size_t release_index = 0; release_index < 2; ++release_index) {
    const LlmBackendLifecycleResult released = backend->release_resources();
    SCOPED_TRACE(release_index);
    EXPECT_EQ(released.status, LlmBackendStatus::Ready);
    EXPECT_EQ(released.reason_code, LlmBackendReason::VALID);
  }
}

TEST(LlmMetalBackendFailureInjectionIntegrationTest, PreparationStopBeforeAllocationPublishesNothingIntegration) {
  LlmMetalBackendTestHooks hooks;
  hooks.stop_requested = []() { return true; };
  std::unique_ptr<LlmBackend> backend = create_llm_metal_backend_for_testing(hooks);
  ASSERT_NE(backend, nullptr);
  const LlmBackendLifecycleResult initialization = backend->initialize(metal_config());
  if (initialization.status == LlmBackendStatus::Unsupported) {
    ASSERT_TRUE(is_stable_capability_unsupported_reason(initialization.reason_code));
    GTEST_SKIP() << "LLM Metal foundation unsupported: " << initialization.reason_code;
  }
  ASSERT_EQ(initialization.status, LlmBackendStatus::Ready) << initialization.reason_code;
  const LlmMetalBackendEvidence* initialized_evidence = get_llm_metal_backend_evidence(backend->evidence());
  ASSERT_NE(initialized_evidence, nullptr);

  const LlmGeometry geometry = contiguous_geometry(4097, 17, 2, 2, 8);
  LlmMetalResourcePlanRequest request = resource_request(geometry);
  request.argument_buffer_encoded_length = initialized_evidence->capability.argument_buffer_encoded_length;
  request.argument_buffer_alignment = initialized_evidence->capability.argument_buffer_alignment;
  request.max_buffer_length = initialized_evidence->capability.max_buffer_length;
  request.available_memory_bytes = 2 * kGiB;
  request.host_mapping_granularity_bytes = get_system_page_size_bytes();
  LlmMemoryWorkPlan plan = make_metal_model_plan(geometry, build_llm_metal_execution_plan(request),
                                                 "llm-metal-phase8-preparation-stop-plan");
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  ASSERT_EQ(backend->resolve_execution_plan(plan).status, LlmBackendStatus::Ready);

  const LlmBackendLifecycleResult prepared = backend->prepare_resources(plan);
  EXPECT_EQ(prepared.status, LlmBackendStatus::Failed);
  EXPECT_EQ(prepared.reason_code, LlmBackendReason::PREPARATION_INTERRUPTED);
  const LlmMetalBackendEvidence* stopped_evidence = get_llm_metal_backend_evidence(backend->evidence());
  ASSERT_NE(stopped_evidence, nullptr);
  EXPECT_FALSE(stopped_evidence->resources.allocation_attempted);
  EXPECT_FALSE(stopped_evidence->resources.allocation_completed);
  EXPECT_TRUE(stopped_evidence->resources.candidate_cleanup_completed);
  EXPECT_FALSE(stopped_evidence->resources.resources_published);
  EXPECT_TRUE(stopped_evidence->resources.resources.empty());
  EXPECT_EQ(stopped_evidence->resources.error.description, LlmBackendReason::PREPARATION_INTERRUPTED);
}

TEST(LlmMetalBackendFailureInjectionIntegrationTest,
     PostAllocationValidationFailuresReleaseEncodedCandidateIntegration) {
  for (size_t failure_index = 0; failure_index < 2; ++failure_index) {
    SCOPED_TRACE(failure_index);
    LlmMetalBackendTestHooks hooks;
    hooks.force_initialization_mismatch = failure_index == 0;
    hooks.force_layout_probe_mismatch = failure_index == 1;
    std::unique_ptr<LlmBackend> backend = create_llm_metal_backend_for_testing(hooks);
    ASSERT_NE(backend, nullptr);
    const LlmBackendLifecycleResult initialization = backend->initialize(metal_config());
    if (initialization.status == LlmBackendStatus::Unsupported) {
      ASSERT_TRUE(is_stable_capability_unsupported_reason(initialization.reason_code));
      GTEST_SKIP() << "LLM Metal foundation unsupported: " << initialization.reason_code;
    }
    ASSERT_EQ(initialization.status, LlmBackendStatus::Ready) << initialization.reason_code;
    const LlmMetalBackendEvidence* initialized_evidence = get_llm_metal_backend_evidence(backend->evidence());
    ASSERT_NE(initialized_evidence, nullptr);

    const LlmGeometry geometry = contiguous_geometry(4097, 17, 2, 2, 8);
    LlmMetalResourcePlanRequest request = resource_request(geometry);
    request.argument_buffer_encoded_length = initialized_evidence->capability.argument_buffer_encoded_length;
    request.argument_buffer_alignment = initialized_evidence->capability.argument_buffer_alignment;
    request.max_buffer_length = initialized_evidence->capability.max_buffer_length;
    request.available_memory_bytes = 2 * kGiB;
    request.host_mapping_granularity_bytes = get_system_page_size_bytes();
    LlmMemoryWorkPlan plan = make_metal_model_plan(
        geometry, build_llm_metal_execution_plan(request),
        failure_index == 0 ? "llm-metal-phase8-initialization-mismatch-plan" : "llm-metal-phase8-layout-mismatch-plan");
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    ASSERT_EQ(backend->resolve_execution_plan(plan).status, LlmBackendStatus::Ready);

    const LlmBackendLifecycleResult prepared = backend->prepare_resources(plan);
    EXPECT_EQ(prepared.status, LlmBackendStatus::Failed);
    EXPECT_EQ(prepared.reason_code, failure_index == 0 ? LlmBackendReason::METAL_RESOURCE_INITIALIZATION_FAILED
                                                       : LlmBackendReason::METAL_ARGUMENT_BUFFER_LAYOUT_INVALID);
    const LlmMetalBackendEvidence* failed_evidence = get_llm_metal_backend_evidence(backend->evidence());
    ASSERT_NE(failed_evidence, nullptr);
    EXPECT_TRUE(failed_evidence->resources.allocation_completed);
    EXPECT_TRUE(failed_evidence->resources.candidate_cleanup_completed);
    EXPECT_FALSE(failed_evidence->resources.resources_published);
    EXPECT_LE(failed_evidence->resources.current_allocated_size_after_release,
              failed_evidence->resources.current_allocated_size_peak);
    EXPECT_EQ(backend->release_resources().status, LlmBackendStatus::Ready);
  }
}

}  // namespace

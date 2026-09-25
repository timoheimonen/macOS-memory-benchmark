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
 * @brief Pure planning and real-device tests for the LLM Metal backend
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
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
#include "llm_memory/llm_metal_kernels_source.h"
#include "llm_memory/llm_work_plan.h"

namespace {

constexpr size_t kGiB = 1024ULL * 1024ULL * 1024ULL;
constexpr size_t kFoundationPipelineCount = 6;
constexpr size_t kWorkloadPipelineCount = 3;
constexpr std::string_view kCanonicalKernelRevision =
    "llm-metal-decode-prefill-contiguous-paged-msl23-v5";
constexpr std::string_view kCanonicalKernelSourceSha256 =
    "2fb49f1f8045aa01a74329d7b9d6b04134502d845e9f70c53b8f6fd0baf8fdd2";

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

LlmGeometry prefill_contiguous_geometry(
    size_t weight_bytes, size_t prompt_tokens, size_t query_tile_tokens,
    size_t layer_count = 1, size_t batch_size = 1,
    size_t head_dimension = 1) {
  LlmGeometryRequest request;
  request.active_weight_bytes = weight_bytes;
  request.layer_count = layer_count;
  request.query_head_count = 1;
  request.kv_head_count = 1;
  request.head_dimension = head_dimension;
  request.kv_element_bytes = 1;
  request.phase = LlmPhase::Prefill;
  request.prompt_tokens = prompt_tokens;
  request.attention_query_tile_tokens = query_tile_tokens;
  request.batch_size = batch_size;
  return resolve_llm_geometry(request);
}

LlmGeometry prefill_paged_geometry(
    size_t weight_bytes, size_t prompt_tokens, size_t query_tile_tokens,
    size_t block_tokens, size_t layer_count = 1, size_t batch_size = 1,
    size_t head_dimension = 1) {
  LlmGeometryRequest request;
  request.active_weight_bytes = weight_bytes;
  request.layer_count = layer_count;
  request.query_head_count = 1;
  request.kv_head_count = 1;
  request.head_dimension = head_dimension;
  request.kv_element_bytes = 1;
  request.phase = LlmPhase::Prefill;
  request.kv_layout = LlmKvLayout::Paged;
  request.prompt_tokens = prompt_tokens;
  request.attention_query_tile_tokens = query_tile_tokens;
  request.kv_block_tokens = block_tokens;
  request.batch_size = batch_size;
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
  probe.workload_pipeline_count = kWorkloadPipelineCount;
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
  plan.scenario_seeds = {UINT64_C(0x4142434445464748),
                         UINT64_C(0x5152535455565758),
                         UINT64_C(0x6162636465666768)};
  plan.component_identities.checksum_pattern_version =
      plan.phase == LlmPhase::Prefill
          ? plan.kv_layout == LlmKvLayout::Paged
                ? LlmMetalPrefillPagedVersion::CHECKSUM
                : LlmMetalPrefillContiguousVersion::CHECKSUM
          : plan.kv_layout == LlmKvLayout::Paged
                ? LlmMetalDecodePagedVersion::CHECKSUM
                : LlmMetalDecodeContiguousVersion::CHECKSUM;
  plan.plan_identity = std::move(identity);
  plan.backend_execution_plan = std::move(execution);
  if (plan.phase == LlmPhase::Prefill && plan.geometry.prefill.has_value()) {
    plan.prefill_plan = resolve_llm_prefill_plan(
        {plan.geometry.active_weight_bytes_per_work_unit,
         plan.geometry.prefill->prompt_tokens,
         plan.geometry.prefill->attention_query_tile_tokens,
         plan.geometry.layer_count,
         plan.geometry.batch_size,
         plan.geometry.query_head_count,
         plan.geometry.head_dimension,
         plan.geometry.k_or_v_record_bytes_per_layer,
         plan.kv_layout == LlmKvLayout::Paged
             ? plan.geometry.kv_block_tokens
             : 0});
    plan.valid = plan.valid && plan.prefill_plan->valid;
  }
  return plan;
}

LlmRunnerTaskContext measurement_context(LlmScenario scenario,
                                         std::string purpose) {
  LlmRunnerTaskContext context;
  context.kind = LlmRunnerTaskKind::Measurement;
  context.purpose = std::move(purpose);
  context.scenario = scenario;
  context.attempt_index = 0;
  context.loop_index = 0;
  context.order_position = 0;
  return context;
}

constexpr uint32_t kTestPatternMultiplier = UINT32_C(0x9e3779b9);
constexpr uint32_t kTestPagedPatternLayerMultiplier = UINT32_C(0xa24baed5);
constexpr uint32_t kTestPagedPatternPhysicalMultiplier = UINT32_C(0x9fb21c65);
constexpr uint32_t kTestPagedPatternWordMultiplier = UINT32_C(0xc13fa9a9);
constexpr uint32_t kTestAppendWorkUnitMultiplier = UINT32_C(0x85ebca6b);
constexpr uint32_t kTestAppendLayerMultiplier = UINT32_C(0xc2b2ae35);
constexpr uint32_t kTestAppendBatchMultiplier = UINT32_C(0x27d4eb2f);
constexpr uint32_t kTestAppendWordMultiplier = UINT32_C(0x165667b1);
constexpr uint32_t kTestAppendKeyDomain = UINT32_C(0x4b455931);
constexpr uint32_t kTestAppendValueDomain = UINT32_C(0x56414c31);
constexpr uint32_t kTestPrefillWriteKeyDomain = UINT32_C(0x504b5731);
constexpr uint32_t kTestPrefillWriteValueDomain = UINT32_C(0x50565731);
constexpr uint32_t kTestChecksumValueMultiplier = UINT32_C(0x9e3779b1);
constexpr uint32_t kTestChecksumAddressMultiplier = UINT32_C(0x85ebca77);
constexpr uint32_t kTestChecksumWorkUnitMultiplier = UINT32_C(0xc2b2ae3d);
constexpr uint32_t kTestChecksumLayerMultiplier = UINT32_C(0x27d4eb35);
constexpr uint32_t kTestChecksumBatchMultiplier = UINT32_C(0x165667c5);
constexpr uint32_t kTestChecksumValidMaskMultiplier = UINT32_C(0xd3a2646d);
constexpr uint32_t kTestChecksumProfileDomain = UINT32_C(0x4d444331);
constexpr uint32_t kTestChecksumPagedProfileDomain = UINT32_C(0x4d445031);
constexpr uint32_t kTestChecksumPrefillProfileDomain = UINT32_C(0x4d504331);
constexpr uint32_t kTestChecksumPrefillPagedProfileDomain =
    UINT32_C(0x4d505031);
constexpr uint32_t kTestChecksumScenarioHighMultiplier = UINT32_C(0xa24baed5);
constexpr uint32_t kTestChecksumWeightDomain = UINT32_C(0x57474854);
constexpr uint32_t kTestChecksumKeyDomain = UINT32_C(0x4b455943);
constexpr uint32_t kTestChecksumValueDomain = UINT32_C(0x56414c43);
constexpr uint32_t kTestChecksumWeightReadVisit = UINT32_C(0x57524541);
constexpr uint32_t kTestChecksumAppendVisit = UINT32_C(0x41505044);
constexpr uint32_t kTestChecksumKvReadVisit = UINT32_C(0x4b565244);
constexpr uint32_t kTestChecksumPrefillWriteVisit = UINT32_C(0x50575254);
constexpr uint32_t kTestChecksumTileMultiplier = UINT32_C(0xd1b54a35);
constexpr uint32_t kTestChecksumPagedLogicalMultiplier = UINT32_C(0x7f4a7c15);
constexpr uint32_t kTestChecksumPagedPhysicalMultiplier = UINT32_C(0x94d049bb);
constexpr uint32_t kTestChecksumPagedPairMultiplier = UINT32_C(0x369dea0f);
constexpr uint32_t kTestChecksumPagedGlobalBlockLowMultiplier =
    UINT32_C(0xdb4f0b91);
constexpr uint32_t kTestChecksumPagedGlobalBlockHighMultiplier =
    UINT32_C(0xbbe05633);
constexpr uint32_t kTestChecksumPagedSegmentMultiplier =
    UINT32_C(0xa0f2ec75);
constexpr uint32_t kTestChecksumPagedLocalBaseLowMultiplier =
    UINT32_C(0x89e18285);
constexpr uint32_t kTestChecksumPagedLocalBaseHighMultiplier =
    UINT32_C(0xc6d1d6c9);
constexpr uint32_t kTestChecksumPagedAbsoluteBaseLowMultiplier =
    UINT32_C(0xb492b66f);
constexpr uint32_t kTestChecksumPagedAbsoluteBaseHighMultiplier =
    UINT32_C(0x9ae16a3b);
constexpr uint32_t kTestChecksumPagedAddressBindingDomain =
    UINT32_C(0x41444452);
constexpr uint32_t kTestChecksumPagedAddressPairMultiplier =
    UINT32_C(0xd6e8feb9);
constexpr uint32_t kTestChecksumPagedAppendLookupVisit = UINT32_C(0x50414c55);
constexpr uint32_t kTestChecksumPagedKeyLookupVisit = UINT32_C(0x504b4c55);
constexpr uint32_t kTestChecksumPagedValueLookupVisit = UINT32_C(0x50564c55);
constexpr uint32_t kTestChecksumPagedPrefillWriteLookupVisit =
    UINT32_C(0x5050574c);
constexpr uint32_t kTestChecksumPagedPrefillKeyLookupVisit =
    UINT32_C(0x50504b4c);
constexpr uint32_t kTestChecksumPagedPrefillValueLookupVisit =
    UINT32_C(0x5050564c);
constexpr uint32_t kTestChecksumDomainMultiplier = UINT32_C(0x7feb352d);

uint8_t word_byte(uint32_t word, size_t byte_index) {
  return static_cast<uint8_t>(word >> (8U * static_cast<unsigned>(byte_index)));
}

uint32_t independent_contiguous_pattern_word(uint64_t seed,
                                             uint64_t word_index) {
  return static_cast<uint32_t>(seed) +
         kTestPatternMultiplier * static_cast<uint32_t>(word_index + 1U);
}

uint32_t independent_paged_pattern_word(uint64_t seed, size_t layer,
                                        uint32_t physical_id,
                                        uint64_t local_word) {
  return static_cast<uint32_t>(seed) +
         kTestPagedPatternLayerMultiplier *
             static_cast<uint32_t>(layer + 1U) +
         kTestPagedPatternPhysicalMultiplier * (physical_id + 1U) +
         kTestPagedPatternWordMultiplier *
             static_cast<uint32_t>(local_word + 1U);
}

uint32_t independent_decode_append_word(uint64_t scenario_seed,
                                        size_t work_unit, size_t layer,
                                        size_t batch, uint64_t word_index,
                                        LlmMetalResourcePool pool) {
  const uint32_t pool_domain = pool == LlmMetalResourcePool::K
                                   ? kTestAppendKeyDomain
                                   : kTestAppendValueDomain;
  return static_cast<uint32_t>(scenario_seed) +
         kTestAppendWorkUnitMultiplier *
             static_cast<uint32_t>(work_unit + 1U) +
         kTestAppendLayerMultiplier * static_cast<uint32_t>(layer + 1U) +
         kTestAppendBatchMultiplier * static_cast<uint32_t>(batch + 1U) +
         kTestAppendWordMultiplier *
             static_cast<uint32_t>(word_index + 1U) +
         pool_domain;
}

uint32_t independent_checksum_domain(
    uint32_t pool_domain, uint32_t visit_domain, uint64_t scenario_seed,
    size_t work_unit, size_t layer, size_t batch, uint32_t valid_mask) {
  return kTestChecksumProfileDomain + static_cast<uint32_t>(scenario_seed) +
         kTestChecksumScenarioHighMultiplier *
             static_cast<uint32_t>(scenario_seed >> 32U) +
         pool_domain + visit_domain +
         kTestChecksumWorkUnitMultiplier *
             static_cast<uint32_t>(work_unit + 1U) +
         kTestChecksumLayerMultiplier * static_cast<uint32_t>(layer + 1U) +
         kTestChecksumBatchMultiplier * static_cast<uint32_t>(batch + 1U) +
         kTestChecksumValidMaskMultiplier * valid_mask;
}

uint32_t independent_paged_checksum_domain(
    uint32_t pool_domain, uint32_t visit_domain, uint64_t scenario_seed,
    size_t work_unit, size_t layer, size_t batch, size_t logical_table_index,
    uint32_t physical_id, uint32_t valid_mask) {
  const uint32_t logical = static_cast<uint32_t>(logical_table_index + 1U);
  const uint32_t physical = physical_id + 1U;
  return kTestChecksumPagedProfileDomain + static_cast<uint32_t>(scenario_seed) +
         kTestChecksumScenarioHighMultiplier *
             static_cast<uint32_t>(scenario_seed >> 32U) +
         pool_domain + visit_domain +
         kTestChecksumWorkUnitMultiplier *
             static_cast<uint32_t>(work_unit + 1U) +
         kTestChecksumLayerMultiplier * static_cast<uint32_t>(layer + 1U) +
         kTestChecksumBatchMultiplier * static_cast<uint32_t>(batch + 1U) +
         kTestChecksumPagedLogicalMultiplier * logical +
         kTestChecksumPagedPhysicalMultiplier * physical +
         kTestChecksumPagedPairMultiplier * logical * physical +
         kTestChecksumValidMaskMultiplier * valid_mask;
}

uint32_t independent_paged_weight_domain(uint64_t scenario_seed,
                                         size_t work_unit, size_t layer,
                                         uint32_t valid_mask) {
  return kTestChecksumPagedProfileDomain + static_cast<uint32_t>(scenario_seed) +
         kTestChecksumScenarioHighMultiplier *
             static_cast<uint32_t>(scenario_seed >> 32U) +
         kTestChecksumWeightDomain + kTestChecksumWeightReadVisit +
         kTestChecksumWorkUnitMultiplier *
             static_cast<uint32_t>(work_unit + 1U) +
         kTestChecksumLayerMultiplier * static_cast<uint32_t>(layer + 1U) +
         kTestChecksumBatchMultiplier +
         kTestChecksumValidMaskMultiplier * valid_mask;
}

void independent_mix_word(LlmMetalMod32Lane& checksum, uint32_t value,
                          uint64_t word_index, uint32_t domain) {
  checksum.a += value + domain;
  checksum.b += value * kTestChecksumValueMultiplier +
                static_cast<uint32_t>(word_index) *
                    kTestChecksumAddressMultiplier +
                domain * kTestChecksumDomainMultiplier;
}

template <typename ByteAt, typename DomainAt>
void independent_accumulate_masked_words(
    LlmMetalMod32Lane& checksum, size_t range_start, size_t range_length,
    uint64_t word_ordinal_base, ByteAt byte_at, DomainAt domain_at) {
  if (range_length == 0) {
    return;
  }
  const size_t range_end = range_start + range_length;
  const uint64_t first_word = range_start / sizeof(uint32_t);
  const uint64_t last_word = (range_end - 1U) / sizeof(uint32_t);
  for (uint64_t local_word = first_word; local_word <= last_word;
       ++local_word) {
    uint32_t packed = 0;
    uint32_t valid_mask = 0;
    const uint64_t word_start = local_word * sizeof(uint32_t);
    for (size_t byte_index = 0; byte_index < sizeof(uint32_t);
         ++byte_index) {
      const uint64_t local_byte = word_start + byte_index;
      if (local_byte < range_start || local_byte >= range_end) {
        continue;
      }
      valid_mask |= 1U << byte_index;
      packed |= static_cast<uint32_t>(byte_at(local_byte))
                << (8U * static_cast<unsigned>(byte_index));
    }
    independent_mix_word(checksum, packed, word_ordinal_base + local_word,
                         domain_at(valid_mask));
  }
}

template <typename ByteAt>
void independent_accumulate_byte_range(
    LlmMetalMod32Lane& checksum, size_t range_start, size_t range_length,
    uint32_t pool_domain, uint32_t visit_domain, uint64_t scenario_seed,
    size_t work_unit, size_t layer, size_t batch, ByteAt byte_at) {
  const size_t range_end = range_start + range_length;
  const uint64_t first_word = range_start / sizeof(uint32_t);
  const uint64_t last_word = (range_end - 1U) / sizeof(uint32_t);
  for (uint64_t word_index = first_word; word_index <= last_word;
       ++word_index) {
    uint32_t packed = 0;
    uint32_t valid_mask = 0;
    const uint64_t word_start = word_index * sizeof(uint32_t);
    for (size_t byte_index = 0; byte_index < sizeof(uint32_t);
         ++byte_index) {
      const uint64_t absolute_byte = word_start + byte_index;
      if (absolute_byte < range_start || absolute_byte >= range_end) {
        continue;
      }
      valid_mask |= 1U << byte_index;
      packed |= static_cast<uint32_t>(byte_at(absolute_byte))
                << (8U * static_cast<unsigned>(byte_index));
    }
    independent_mix_word(
        checksum, packed, word_index,
        independent_checksum_domain(pool_domain, visit_domain, scenario_seed,
                                    work_unit, layer, batch, valid_mask));
  }
}

LlmMetalDualMod32Checksum independent_decode_checksum_byte_by_byte(
    const LlmMemoryWorkPlan& model_plan,
    const LlmScenarioWorkPlan& scenario_plan) {
  LlmMetalDualMod32Checksum checksum;
  const LlmGeometry& geometry = model_plan.geometry;
  const bool include_weight =
      scenario_plan.scenario != LlmScenario::KvOnly;
  const bool include_kv =
      scenario_plan.scenario != LlmScenario::WeightsOnly;
  const size_t sequence_bytes =
      geometry.decode->visible_context_tokens *
      geometry.k_or_v_record_bytes_per_layer;
  const size_t weight_layer_base =
      geometry.active_weight_bytes_per_work_unit / geometry.layer_count;
  const size_t weight_layer_remainder =
      geometry.active_weight_bytes_per_work_unit % geometry.layer_count;

  for (size_t work_unit = 0; work_unit < scenario_plan.work_units;
       ++work_unit) {
    size_t weight_offset = 0;
    for (size_t layer = 0; layer < geometry.layer_count; ++layer) {
      const size_t weight_bytes =
          weight_layer_base + (layer < weight_layer_remainder ? 1U : 0U);
      if (include_weight && weight_bytes != 0) {
        independent_accumulate_byte_range(
            checksum.weight, weight_offset, weight_bytes,
            kTestChecksumWeightDomain, kTestChecksumWeightReadVisit,
            scenario_plan.scenario_seed, work_unit, layer, 0,
            [&](uint64_t absolute_byte) {
              const uint32_t word = independent_contiguous_pattern_word(
                  model_plan.weight_buffer_seed,
                  absolute_byte / sizeof(uint32_t));
              return word_byte(word, absolute_byte % sizeof(uint32_t));
            });
      }
      weight_offset += weight_bytes;
      if (!include_kv) {
        continue;
      }
      for (size_t batch = 0; batch < geometry.batch_size; ++batch) {
        const size_t sequence_index = layer * geometry.batch_size + batch;
        const size_t sequence_start = sequence_index * sequence_bytes;
        const size_t append_start =
            sequence_start + sequence_bytes -
            geometry.k_or_v_record_bytes_per_layer;
        const size_t append_end =
            append_start + geometry.k_or_v_record_bytes_per_layer;
        const auto accumulate_pool = [&](LlmMetalMod32Lane& lane,
                                         LlmMetalResourcePool pool,
                                         uint64_t initial_seed,
                                         uint32_t pool_domain) {
          const auto appended_byte = [&](uint64_t absolute_byte) {
            const uint32_t word = independent_decode_append_word(
                scenario_plan.scenario_seed, work_unit, layer, batch,
                absolute_byte / sizeof(uint32_t), pool);
            return word_byte(word, absolute_byte % sizeof(uint32_t));
          };
          independent_accumulate_byte_range(
              lane, append_start, geometry.k_or_v_record_bytes_per_layer,
              pool_domain, kTestChecksumAppendVisit,
              scenario_plan.scenario_seed, work_unit, layer, batch,
              appended_byte);
          independent_accumulate_byte_range(
              lane, sequence_start, sequence_bytes, pool_domain,
              kTestChecksumKvReadVisit, scenario_plan.scenario_seed,
              work_unit, layer, batch, [&](uint64_t absolute_byte) {
                if (absolute_byte >= append_start &&
                    absolute_byte < append_end) {
                  return appended_byte(absolute_byte);
                }
                const uint32_t word = independent_contiguous_pattern_word(
                    initial_seed, absolute_byte / sizeof(uint32_t));
                return word_byte(word, absolute_byte % sizeof(uint32_t));
              });
        };
        accumulate_pool(checksum.k, LlmMetalResourcePool::K,
                        model_plan.k_buffer_seed, kTestChecksumKeyDomain);
        accumulate_pool(checksum.v, LlmMetalResourcePool::V,
                        model_plan.v_buffer_seed, kTestChecksumValueDomain);
      }
    }
  }
  return checksum;
}

uint32_t independent_prefill_write_word(
    uint64_t scenario_seed, size_t work_unit, size_t layer, size_t batch,
    uint64_t word_index, LlmMetalResourcePool pool) {
  const uint32_t pool_domain =
      pool == LlmMetalResourcePool::K ? kTestPrefillWriteKeyDomain
                                      : kTestPrefillWriteValueDomain;
  return static_cast<uint32_t>(scenario_seed) +
         kTestAppendWorkUnitMultiplier *
             static_cast<uint32_t>(work_unit + 1U) +
         kTestAppendLayerMultiplier * static_cast<uint32_t>(layer + 1U) +
         kTestAppendBatchMultiplier * static_cast<uint32_t>(batch + 1U) +
         kTestAppendWordMultiplier * static_cast<uint32_t>(word_index + 1U) +
         pool_domain;
}

uint32_t independent_prefill_checksum_domain(
    uint32_t pool_domain, uint32_t visit_domain, uint64_t scenario_seed,
    size_t work_unit, size_t layer, size_t batch, size_t tile_ordinal,
    uint32_t valid_mask) {
  return kTestChecksumPrefillProfileDomain +
         static_cast<uint32_t>(scenario_seed) +
         kTestChecksumScenarioHighMultiplier *
             static_cast<uint32_t>(scenario_seed >> 32U) +
         pool_domain + visit_domain +
         kTestChecksumWorkUnitMultiplier *
             static_cast<uint32_t>(work_unit + 1U) +
         kTestChecksumLayerMultiplier * static_cast<uint32_t>(layer + 1U) +
         kTestChecksumBatchMultiplier * static_cast<uint32_t>(batch + 1U) +
         kTestChecksumTileMultiplier * static_cast<uint32_t>(tile_ordinal) +
         kTestChecksumValidMaskMultiplier * valid_mask;
}

uint32_t independent_prefill_paged_data_domain(
    uint32_t pool_domain, uint32_t visit_domain, uint64_t scenario_seed,
    size_t work_unit, size_t layer, size_t batch, size_t tile_ordinal,
    uint32_t valid_mask) {
  return kTestChecksumPrefillPagedProfileDomain +
         static_cast<uint32_t>(scenario_seed) +
         kTestChecksumScenarioHighMultiplier *
             static_cast<uint32_t>(scenario_seed >> 32U) +
         pool_domain + visit_domain +
         kTestChecksumWorkUnitMultiplier *
             static_cast<uint32_t>(work_unit + 1U) +
         kTestChecksumLayerMultiplier * static_cast<uint32_t>(layer + 1U) +
         kTestChecksumBatchMultiplier * static_cast<uint32_t>(batch + 1U) +
         kTestChecksumTileMultiplier * static_cast<uint32_t>(tile_ordinal) +
         kTestChecksumValidMaskMultiplier * valid_mask;
}

uint32_t independent_prefill_paged_lookup_domain(
    uint32_t pool_domain, uint32_t visit_domain, uint64_t scenario_seed,
    size_t work_unit, size_t layer, size_t batch, size_t tile_ordinal,
    size_t logical_table_index, uint32_t physical_id,
    uint32_t physical_address_token) {
  const uint32_t logical = static_cast<uint32_t>(logical_table_index + 1U);
  const uint32_t physical = physical_id + 1U;
  return kTestChecksumPrefillPagedProfileDomain +
         static_cast<uint32_t>(scenario_seed) +
         kTestChecksumScenarioHighMultiplier *
             static_cast<uint32_t>(scenario_seed >> 32U) +
         pool_domain + visit_domain +
         kTestChecksumWorkUnitMultiplier *
             static_cast<uint32_t>(work_unit + 1U) +
         kTestChecksumLayerMultiplier * static_cast<uint32_t>(layer + 1U) +
         kTestChecksumBatchMultiplier * static_cast<uint32_t>(batch + 1U) +
         kTestChecksumTileMultiplier * static_cast<uint32_t>(tile_ordinal) +
         kTestChecksumPagedLogicalMultiplier * logical +
         kTestChecksumPagedPhysicalMultiplier * physical +
         kTestChecksumPagedPairMultiplier * logical * physical +
         physical_address_token +
         kTestChecksumPagedAddressPairMultiplier * logical *
             physical_address_token;
}

uint32_t independent_prefill_paged_physical_address_token(
    size_t layer, uint32_t physical_id, size_t physical_blocks_per_layer,
    size_t blocks_per_segment, size_t block_bytes,
    size_t global_block_bias = 0) {
  const uint64_t global_block =
      static_cast<uint64_t>(layer) * physical_blocks_per_layer +
      physical_id + global_block_bias;
  const uint64_t segment = global_block / blocks_per_segment;
  const uint64_t segment_local_block_base =
      (global_block % blocks_per_segment) * block_bytes;
  const uint64_t absolute_block_base = global_block * block_bytes;
  return kTestChecksumPagedAddressBindingDomain +
         kTestChecksumPagedGlobalBlockLowMultiplier *
             static_cast<uint32_t>(global_block) +
         kTestChecksumPagedGlobalBlockHighMultiplier *
             static_cast<uint32_t>(global_block >> 32U) +
         kTestChecksumPagedSegmentMultiplier *
             static_cast<uint32_t>(segment) +
         kTestChecksumPagedLocalBaseLowMultiplier *
             static_cast<uint32_t>(segment_local_block_base) +
         kTestChecksumPagedLocalBaseHighMultiplier *
             static_cast<uint32_t>(segment_local_block_base >> 32U) +
         kTestChecksumPagedAbsoluteBaseLowMultiplier *
             static_cast<uint32_t>(absolute_block_base) +
         kTestChecksumPagedAbsoluteBaseHighMultiplier *
             static_cast<uint32_t>(absolute_block_base >> 32U);
}

template <typename ByteAt>
void independent_accumulate_prefill_byte_range(
    LlmMetalMod32Lane& checksum, size_t range_start, size_t range_length,
    uint32_t pool_domain, uint32_t visit_domain, uint64_t scenario_seed,
    size_t work_unit, size_t layer, size_t batch, size_t tile_ordinal,
    ByteAt byte_at) {
  const size_t range_end = range_start + range_length;
  const uint64_t first_word = range_start / sizeof(uint32_t);
  const uint64_t last_word = (range_end - 1U) / sizeof(uint32_t);
  for (uint64_t word_index = first_word; word_index <= last_word;
       ++word_index) {
    uint32_t packed = 0;
    uint32_t valid_mask = 0;
    const uint64_t word_start = word_index * sizeof(uint32_t);
    for (size_t byte_index = 0; byte_index < sizeof(uint32_t);
         ++byte_index) {
      const uint64_t absolute_byte = word_start + byte_index;
      if (absolute_byte < range_start || absolute_byte >= range_end) {
        continue;
      }
      valid_mask |= 1U << byte_index;
      packed |= static_cast<uint32_t>(byte_at(absolute_byte))
                << (8U * static_cast<unsigned>(byte_index));
    }
    independent_mix_word(
        checksum, packed, word_index,
        independent_prefill_checksum_domain(
            pool_domain, visit_domain, scenario_seed, work_unit, layer,
            batch, tile_ordinal, valid_mask));
  }
}

template <typename ByteAt>
void independent_accumulate_prefill_paged_byte_range(
    LlmMetalMod32Lane& checksum, size_t range_start, size_t range_length,
    uint32_t pool_domain, uint32_t visit_domain, uint64_t scenario_seed,
    size_t work_unit, size_t layer, size_t batch, size_t tile_ordinal,
    ByteAt byte_at) {
  if (range_length == 0) {
    return;
  }
  independent_accumulate_masked_words(
      checksum, range_start, range_length, 0, byte_at,
      [&](uint32_t valid_mask) {
        return independent_prefill_paged_data_domain(
            pool_domain, visit_domain, scenario_seed, work_unit, layer,
            batch, tile_ordinal, valid_mask);
      });
}

LlmMetalDualMod32Checksum independent_prefill_checksum_byte_by_byte(
    const LlmMemoryWorkPlan& model_plan,
    const LlmScenarioWorkPlan& scenario_plan) {
  LlmMetalDualMod32Checksum checksum;
  const LlmGeometry& geometry = model_plan.geometry;
  const LlmPrefillGeometry& prefill = *geometry.prefill;
  const bool include_weight =
      scenario_plan.scenario != LlmScenario::KvOnly;
  const bool include_kv =
      scenario_plan.scenario != LlmScenario::WeightsOnly;
  const size_t sequence_bytes =
      prefill.prompt_tokens * geometry.k_or_v_record_bytes_per_layer;
  const size_t weight_layer_base =
      geometry.active_weight_bytes_per_work_unit / geometry.layer_count;
  const size_t weight_layer_remainder =
      geometry.active_weight_bytes_per_work_unit % geometry.layer_count;

  for (size_t work_unit = 0; work_unit < scenario_plan.work_units;
       ++work_unit) {
    size_t weight_offset = 0;
    for (size_t layer = 0; layer < geometry.layer_count; ++layer) {
      const size_t weight_bytes =
          weight_layer_base + (layer < weight_layer_remainder ? 1U : 0U);
      if (include_weight && weight_bytes != 0) {
        independent_accumulate_prefill_byte_range(
            checksum.weight, weight_offset, weight_bytes,
            kTestChecksumWeightDomain, kTestChecksumWeightReadVisit,
            scenario_plan.scenario_seed, work_unit, layer, 0, 0,
            [&](uint64_t absolute_byte) {
              const uint32_t word = independent_contiguous_pattern_word(
                  model_plan.weight_buffer_seed,
                  absolute_byte / sizeof(uint32_t));
              return word_byte(word, absolute_byte % sizeof(uint32_t));
            });
      }
      weight_offset += weight_bytes;
      if (!include_kv) {
        continue;
      }
      for (size_t batch = 0; batch < geometry.batch_size; ++batch) {
        const size_t sequence_start =
            (layer * geometry.batch_size + batch) * sequence_bytes;
        const auto accumulate_pool = [&](LlmMetalMod32Lane& lane,
                                         LlmMetalResourcePool pool,
                                         uint32_t pool_domain) {
          const auto write_byte = [&](uint64_t absolute_byte) {
            const uint32_t word = independent_prefill_write_word(
                scenario_plan.scenario_seed, work_unit, layer, batch,
                absolute_byte / sizeof(uint32_t), pool);
            return word_byte(word, absolute_byte % sizeof(uint32_t));
          };
          for (size_t prompt_token = 0;
               prompt_token < prefill.prompt_tokens; ++prompt_token) {
            independent_accumulate_prefill_byte_range(
                lane,
                sequence_start +
                    prompt_token *
                        geometry.k_or_v_record_bytes_per_layer,
                geometry.k_or_v_record_bytes_per_layer, pool_domain,
                kTestChecksumPrefillWriteVisit,
                scenario_plan.scenario_seed, work_unit, layer, batch, 0,
                write_byte);
          }
          size_t remaining_tokens = prefill.prompt_tokens;
          size_t prefix_tokens = 0;
          size_t tile_ordinal = 0;
          while (remaining_tokens != 0) {
            const size_t tile_tokens = std::min(
                prefill.attention_query_tile_tokens, remaining_tokens);
            prefix_tokens += tile_tokens;
            ++tile_ordinal;
            independent_accumulate_prefill_byte_range(
                lane, sequence_start,
                prefix_tokens * geometry.k_or_v_record_bytes_per_layer,
                pool_domain, kTestChecksumKvReadVisit,
                scenario_plan.scenario_seed, work_unit, layer, batch,
                tile_ordinal, write_byte);
            remaining_tokens -= tile_tokens;
          }
          EXPECT_EQ(tile_ordinal, prefill.tile_count);
        };
        accumulate_pool(checksum.k, LlmMetalResourcePool::K,
                        kTestChecksumKeyDomain);
        accumulate_pool(checksum.v, LlmMetalResourcePool::V,
                        kTestChecksumValueDomain);
      }
    }
  }
  return checksum;
}

LlmMetalDualMod32Checksum independent_paged_prefill_checksum_byte_by_byte(
    const LlmMemoryWorkPlan& model_plan,
    const LlmScenarioWorkPlan& scenario_plan,
    const std::vector<uint32_t>& table,
    size_t physical_address_global_block_bias = 0) {
  LlmMetalDualMod32Checksum checksum;
  const LlmMetalExecutionPlan* execution =
      get_llm_metal_execution_plan(model_plan);
  EXPECT_NE(execution, nullptr);
  if (execution == nullptr || !execution->resources.paged_layout.has_value() ||
      !model_plan.geometry.prefill.has_value()) {
    return checksum;
  }
  const LlmKvLayoutPlan& layout = *execution->resources.paged_layout;
  const size_t blocks_per_segment =
      execution->resources.k_segments.elements_per_segment;
  EXPECT_EQ(table.size(), layout.block_table_entries);
  EXPECT_NE(blocks_per_segment, 0U);
  if (table.size() != layout.block_table_entries ||
      blocks_per_segment == 0) {
    return checksum;
  }
  const LlmGeometry& geometry = model_plan.geometry;
  const LlmPrefillGeometry& prefill = *geometry.prefill;
  const bool include_weight = scenario_plan.scenario != LlmScenario::KvOnly;
  const bool include_kv =
      scenario_plan.scenario != LlmScenario::WeightsOnly;
  const size_t sequence_bytes =
      prefill.prompt_tokens * geometry.k_or_v_record_bytes_per_layer;
  const size_t weight_layer_base =
      geometry.active_weight_bytes_per_work_unit / geometry.layer_count;
  const size_t weight_layer_remainder =
      geometry.active_weight_bytes_per_work_unit % geometry.layer_count;

  for (size_t work_unit = 0; work_unit < scenario_plan.work_units;
       ++work_unit) {
    if (include_weight) {
      size_t weight_start = 0;
      for (size_t layer = 0; layer < geometry.layer_count; ++layer) {
        const size_t weight_bytes =
            weight_layer_base + (layer < weight_layer_remainder ? 1U : 0U);
        independent_accumulate_prefill_paged_byte_range(
            checksum.weight, weight_start, weight_bytes,
            kTestChecksumWeightDomain, kTestChecksumWeightReadVisit,
            scenario_plan.scenario_seed, work_unit, layer, 0, 0,
            [&](uint64_t absolute_byte) {
              return word_byte(
                  independent_contiguous_pattern_word(
                      model_plan.weight_buffer_seed,
                      absolute_byte / sizeof(uint32_t)),
                  absolute_byte % sizeof(uint32_t));
            });
        weight_start += weight_bytes;
      }
    }
    if (!include_kv) {
      continue;
    }

    for (size_t layer = 0; layer < geometry.layer_count; ++layer) {
      for (size_t batch = 0; batch < geometry.batch_size; ++batch) {
        const size_t sequence_start =
            (layer * geometry.batch_size + batch) * sequence_bytes;
        const auto write_byte = [&](uint64_t absolute_byte,
                                    LlmMetalResourcePool pool) {
          return word_byte(
              independent_prefill_write_word(
                  scenario_plan.scenario_seed, work_unit, layer, batch,
                  absolute_byte / sizeof(uint32_t), pool),
              absolute_byte % sizeof(uint32_t));
        };
        const auto mix_lookup = [&](LlmMetalMod32Lane& lane,
                                    size_t logical_table_index,
                                    uint32_t physical_id,
                                    uint32_t pool_domain,
                                    uint32_t visit_domain,
                                    size_t tile_ordinal) {
          independent_mix_word(
              lane, physical_id, logical_table_index,
              independent_prefill_paged_lookup_domain(
                  pool_domain, visit_domain, scenario_plan.scenario_seed,
                  work_unit, layer, batch, tile_ordinal,
                  logical_table_index, physical_id,
                  independent_prefill_paged_physical_address_token(
                      layer, physical_id,
                      layout.physical_blocks_per_layer,
                      blocks_per_segment, layout.block_bytes,
                      physical_address_global_block_bias)));
        };
        const auto accumulate_data = [&](LlmMetalMod32Lane& lane,
                                         LlmMetalResourcePool pool,
                                         uint32_t pool_domain,
                                         uint32_t visit_domain,
                                         size_t tile_ordinal,
                                         size_t range_start,
                                         size_t range_length) {
          independent_accumulate_prefill_paged_byte_range(
              lane, range_start, range_length, pool_domain, visit_domain,
              scenario_plan.scenario_seed, work_unit, layer, batch,
              tile_ordinal,
              [&](uint64_t absolute_byte) {
                return write_byte(absolute_byte, pool);
              });
        };

        for (size_t logical_block = 0;
             logical_block < layout.blocks_per_sequence; ++logical_block) {
          const size_t logical_table_index =
              batch * layout.blocks_per_sequence + logical_block;
          const uint32_t physical_id = table[logical_table_index];
          EXPECT_LT(physical_id, layout.physical_blocks_per_layer);
          mix_lookup(checksum.k, logical_table_index, physical_id,
                     kTestChecksumKeyDomain,
                     kTestChecksumPagedPrefillWriteLookupVisit, 0);
          mix_lookup(checksum.v, logical_table_index, physical_id,
                     kTestChecksumValueDomain,
                     kTestChecksumPagedPrefillWriteLookupVisit, 0);
          const size_t valid_bytes =
              logical_block + 1U == layout.blocks_per_sequence
                  ? layout.last_block_valid_bytes
                  : layout.block_bytes;
          const size_t range_start =
              sequence_start + logical_block * layout.block_bytes;
          accumulate_data(checksum.k, LlmMetalResourcePool::K,
                          kTestChecksumKeyDomain,
                          kTestChecksumPrefillWriteVisit, 0, range_start,
                          valid_bytes);
          accumulate_data(checksum.v, LlmMetalResourcePool::V,
                          kTestChecksumValueDomain,
                          kTestChecksumPrefillWriteVisit, 0, range_start,
                          valid_bytes);
        }

        size_t remaining_tokens = prefill.prompt_tokens;
        size_t prefix_tokens = 0;
        size_t tile_ordinal = 0;
        while (remaining_tokens != 0) {
          const size_t tile_tokens = std::min(
              prefill.attention_query_tile_tokens, remaining_tokens);
          prefix_tokens += tile_tokens;
          ++tile_ordinal;
          for (LlmMetalResourcePool pool : {LlmMetalResourcePool::K,
                                            LlmMetalResourcePool::V}) {
            size_t prefix_bytes =
                prefix_tokens * geometry.k_or_v_record_bytes_per_layer;
            for (size_t logical_block = 0;
                 logical_block < layout.blocks_per_sequence &&
                 prefix_bytes != 0;
                 ++logical_block) {
              const size_t logical_table_index =
                  batch * layout.blocks_per_sequence + logical_block;
              const uint32_t physical_id = table[logical_table_index];
              const bool key = pool == LlmMetalResourcePool::K;
              LlmMetalMod32Lane& lane = key ? checksum.k : checksum.v;
              const uint32_t pool_domain =
                  key ? kTestChecksumKeyDomain : kTestChecksumValueDomain;
              mix_lookup(
                  lane, logical_table_index, physical_id, pool_domain,
                  key ? kTestChecksumPagedPrefillKeyLookupVisit
                      : kTestChecksumPagedPrefillValueLookupVisit,
                  tile_ordinal);
              const size_t visit_bytes =
                  std::min(layout.block_bytes, prefix_bytes);
              accumulate_data(
                  lane, pool, pool_domain, kTestChecksumKvReadVisit,
                  tile_ordinal,
                  sequence_start + logical_block * layout.block_bytes,
                  visit_bytes);
              prefix_bytes -= visit_bytes;
            }
            EXPECT_EQ(prefix_bytes, 0U);
          }
          remaining_tokens -= tile_tokens;
        }
        EXPECT_EQ(tile_ordinal, prefill.tile_count);
      }
    }
  }
  return checksum;
}

LlmMetalDualMod32Checksum independent_paged_decode_checksum_byte_by_byte(
    const LlmMemoryWorkPlan& model_plan,
    const LlmScenarioWorkPlan& scenario_plan,
    const std::vector<uint32_t>& table) {
  LlmMetalDualMod32Checksum checksum;
  const LlmMetalExecutionPlan* execution =
      get_llm_metal_execution_plan(model_plan);
  EXPECT_NE(execution, nullptr);
  if (execution == nullptr || !execution->resources.paged_layout.has_value()) {
    return checksum;
  }
  const LlmKvLayoutPlan& layout = *execution->resources.paged_layout;
  EXPECT_EQ(table.size(), layout.block_table_entries);
  if (table.size() != layout.block_table_entries) {
    return checksum;
  }
  const LlmGeometry& geometry = model_plan.geometry;
  const bool include_weight =
      scenario_plan.scenario != LlmScenario::KvOnly;
  const bool include_kv =
      scenario_plan.scenario != LlmScenario::WeightsOnly;
  const size_t words_per_block =
      layout.block_bytes / sizeof(uint32_t) +
      (layout.block_bytes % sizeof(uint32_t) != 0 ? 1U : 0U);
  const size_t weight_layer_base =
      geometry.active_weight_bytes_per_work_unit / geometry.layer_count;
  const size_t weight_layer_remainder =
      geometry.active_weight_bytes_per_work_unit % geometry.layer_count;

  for (size_t work_unit = 0; work_unit < scenario_plan.work_units;
       ++work_unit) {
    if (include_weight) {
      size_t weight_start = 0;
      for (size_t layer = 0; layer < geometry.layer_count; ++layer) {
        const size_t weight_bytes =
            weight_layer_base + (layer < weight_layer_remainder ? 1U : 0U);
        independent_accumulate_masked_words(
            checksum.weight, weight_start, weight_bytes, 0,
            [&](uint64_t absolute_byte) {
              return word_byte(
                  independent_contiguous_pattern_word(
                      model_plan.weight_buffer_seed,
                      absolute_byte / sizeof(uint32_t)),
                  absolute_byte % sizeof(uint32_t));
            },
            [&](uint32_t valid_mask) {
              return independent_paged_weight_domain(
                  scenario_plan.scenario_seed, work_unit, layer, valid_mask);
            });
        weight_start += weight_bytes;
      }
    }
    if (!include_kv) {
      continue;
    }

    for (size_t layer = 0; layer < layout.layer_count; ++layer) {
      for (size_t batch = 0; batch < layout.batch_size; ++batch) {
        for (size_t logical_block = 0;
             logical_block < layout.blocks_per_sequence; ++logical_block) {
          const size_t logical_table_index =
              batch * layout.blocks_per_sequence + logical_block;
          const uint32_t physical_id = table[logical_table_index];
          const bool terminal =
              logical_block + 1 == layout.blocks_per_sequence;
          const size_t valid_bytes =
              terminal ? layout.last_block_valid_bytes : layout.block_bytes;
          const uint64_t word_ordinal_base =
              static_cast<uint64_t>(logical_table_index) * words_per_block;
          const auto mix_lookup = [&](LlmMetalMod32Lane& lane,
                                      uint32_t pool_domain,
                                      uint32_t visit_domain) {
            independent_mix_word(
                lane, physical_id, logical_table_index,
                independent_paged_checksum_domain(
                    pool_domain, visit_domain, scenario_plan.scenario_seed,
                    work_unit, layer, batch, logical_table_index, physical_id,
                    0));
          };
          const auto append_byte = [&](uint64_t local_byte,
                                       LlmMetalResourcePool pool) {
            return word_byte(
                independent_decode_append_word(
                    scenario_plan.scenario_seed, work_unit, layer, batch,
                    local_byte / sizeof(uint32_t), pool),
                local_byte % sizeof(uint32_t));
          };
          const auto initial_byte = [&](uint64_t local_byte, uint64_t seed) {
            return word_byte(
                independent_paged_pattern_word(
                    seed, layer, physical_id,
                    local_byte / sizeof(uint32_t)),
                local_byte % sizeof(uint32_t));
          };
          const auto accumulate_data = [&](LlmMetalMod32Lane& lane,
                                           LlmMetalResourcePool pool,
                                           uint64_t seed,
                                           uint32_t pool_domain,
                                           uint32_t visit_domain,
                                           size_t range_start,
                                           size_t range_length,
                                           bool appended) {
            independent_accumulate_masked_words(
                lane, range_start, range_length, word_ordinal_base,
                [&](uint64_t local_byte) {
                  return appended
                             ? append_byte(local_byte, pool)
                             : initial_byte(local_byte, seed);
                },
                [&](uint32_t valid_mask) {
                  return independent_paged_checksum_domain(
                      pool_domain, visit_domain,
                      scenario_plan.scenario_seed, work_unit, layer, batch,
                      logical_table_index, physical_id, valid_mask);
                });
          };
          const auto accumulate_scan = [&](LlmMetalMod32Lane& lane,
                                           LlmMetalResourcePool pool,
                                           uint64_t seed,
                                           uint32_t pool_domain) {
            independent_accumulate_masked_words(
                lane, 0, valid_bytes, word_ordinal_base,
                [&](uint64_t local_byte) {
                  return terminal &&
                                 local_byte >=
                                     layout.decode_append_offset_in_last_block
                             ? append_byte(local_byte, pool)
                             : initial_byte(local_byte, seed);
                },
                [&](uint32_t valid_mask) {
                  return independent_paged_checksum_domain(
                      pool_domain, kTestChecksumKvReadVisit,
                      scenario_plan.scenario_seed, work_unit, layer, batch,
                      logical_table_index, physical_id, valid_mask);
                });
          };

          if (terminal) {
            mix_lookup(checksum.k, kTestChecksumKeyDomain,
                       kTestChecksumPagedAppendLookupVisit);
            mix_lookup(checksum.v, kTestChecksumValueDomain,
                       kTestChecksumPagedAppendLookupVisit);
            accumulate_data(
                checksum.k, LlmMetalResourcePool::K,
                model_plan.k_buffer_seed, kTestChecksumKeyDomain,
                kTestChecksumAppendVisit,
                layout.decode_append_offset_in_last_block,
                layout.k_or_v_record_bytes_per_layer, true);
            accumulate_data(
                checksum.v, LlmMetalResourcePool::V,
                model_plan.v_buffer_seed, kTestChecksumValueDomain,
                kTestChecksumAppendVisit,
                layout.decode_append_offset_in_last_block,
                layout.k_or_v_record_bytes_per_layer, true);
          }

          mix_lookup(checksum.k, kTestChecksumKeyDomain,
                     kTestChecksumPagedKeyLookupVisit);
          accumulate_scan(checksum.k, LlmMetalResourcePool::K,
                          model_plan.k_buffer_seed, kTestChecksumKeyDomain);

          mix_lookup(checksum.v, kTestChecksumValueDomain,
                     kTestChecksumPagedValueLookupVisit);
          accumulate_scan(checksum.v, LlmMetalResourcePool::V,
                          model_plan.v_buffer_seed,
                          kTestChecksumValueDomain);
        }
      }
    }
  }
  return checksum;
}

uint32_t independent_triangular_mod32(uint64_t count) {
  const unsigned __int128 wide =
      static_cast<unsigned __int128>(count) *
      (count == 0 ? 0 : count - 1U) / 2U;
  return static_cast<uint32_t>(wide);
}

LlmMetalMod32Lane independent_large_weight_lane(
    size_t weight_bytes, uint64_t weight_seed, uint64_t scenario_seed) {
  LlmMetalMod32Lane checksum;
  const uint64_t full_words = weight_bytes / sizeof(uint32_t);
  const uint32_t count = static_cast<uint32_t>(full_words);
  const uint32_t index_sum = independent_triangular_mod32(full_words);
  const uint32_t value_sum =
      count * static_cast<uint32_t>(weight_seed) +
      kTestPatternMultiplier * (count + index_sum);
  const uint32_t full_domain = independent_checksum_domain(
      kTestChecksumWeightDomain, kTestChecksumWeightReadVisit, scenario_seed,
      0, 0, 0, 0x0fU);
  checksum.a += value_sum + count * full_domain;
  checksum.b += value_sum * kTestChecksumValueMultiplier +
                index_sum * kTestChecksumAddressMultiplier +
                count * full_domain * kTestChecksumDomainMultiplier;
  const size_t tail_bytes = weight_bytes % sizeof(uint32_t);
  if (tail_bytes != 0) {
    const uint32_t valid_mask = (1U << tail_bytes) - 1U;
    const uint32_t byte_mask =
        tail_bytes == 1 ? UINT32_C(0x000000ff)
                        : tail_bytes == 2 ? UINT32_C(0x0000ffff)
                                          : UINT32_C(0x00ffffff);
    independent_mix_word(
        checksum,
        independent_contiguous_pattern_word(weight_seed, full_words) &
            byte_mask,
        full_words,
        independent_checksum_domain(
            kTestChecksumWeightDomain, kTestChecksumWeightReadVisit,
            scenario_seed, 0, 0, 0, valid_mask));
  }
  return checksum;
}

TEST(LlmMetalBackendTest, CapabilityStateMachineUsesStableFirstFailureOrdering) {
  struct CapabilityCase {
    void (*mutate)(LlmMetalCapabilityProbe&);
    LlmBackendStatus status;
    std::string_view reason;
  };
  const std::array<CapabilityCase, 12> cases = {{
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
      {+[](LlmMetalCapabilityProbe& probe) { probe.workload_pipeline_count = kWorkloadPipelineCount - 1; },
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
  const std::array<size_t, 4> lengths = {capacity - 1, capacity, capacity + 1, 3 * capacity + 17};
  const std::array<std::vector<size_t>, 4> expected = {std::vector<size_t>{capacity - 1}, std::vector<size_t>{capacity},
                                                       std::vector<size_t>{capacity, 1},
                                                       std::vector<size_t>{capacity, capacity, capacity, 17}};

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
    for (const LlmKvSegmentPlan* segments :
         {&plan.resources.weight_segments, &plan.resources.k_segments, &plan.resources.v_segments}) {
      EXPECT_EQ(segments->segment_count, expected[index].size());
      EXPECT_EQ(std::accumulate(segments->segment_lengths.begin(), segments->segment_lengths.end(), size_t{0}),
                lengths[index]);
    }
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

TEST(LlmMetalBackendTest, PagedTableAndPoolsUseIndependentMultiSegmentWholeElementPlans) {
  struct Case {
    LlmGeometry geometry;
    size_t capacity;
    size_t blocks_per_segment;
    std::vector<size_t> pool_lengths;
    std::vector<size_t> table_lengths;
  };
  std::vector<size_t> many_pool_lengths(12, 20);
  many_pool_lengths.push_back(16);
  const std::array<Case, 5> cases = {{
      {paged_geometry(101, 49, 16, 2, 2, 3), 100, 2, std::vector<size_t>(8, 96), {32}},
      {paged_geometry(1, 99, 1), 100, 100, {99}, {100, 100, 100, 96}},
      {paged_geometry(1, 100, 1), 100, 100, {100}, {100, 100, 100, 100}},
      {paged_geometry(1, 101, 1), 100, 100, {100, 1}, {100, 100, 100, 100, 4}},
      {paged_geometry(41, 29, 4, 2, 4, 1), 20, 5, many_pool_lengths, {20, 20, 20, 20, 20, 20, 8}},
  }};
  for (const auto& row : cases) {
    SCOPED_TRACE(::testing::Message() << row.capacity << ":" << row.geometry.k_mapping_bytes);
    ASSERT_TRUE(row.geometry.valid) << row.geometry.reason_code;
    LlmMetalPlanningLimits limits;
    limits.segment_capacity_bytes = row.capacity;
    limits.segment_slots_per_pool = 32;
    const auto plan = build_llm_metal_execution_plan(resource_request(row.geometry, limits));
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    ASSERT_TRUE(plan.resources.table_segments.has_value());
    EXPECT_EQ(plan.resources.k_segments.elements_per_segment, row.blocks_per_segment);
    for (const auto* pool : {&plan.resources.k_segments, &plan.resources.v_segments}) {
      EXPECT_EQ(pool->segment_lengths, row.pool_lengths);
      EXPECT_EQ(pool->segment_count, row.pool_lengths.size());
      EXPECT_EQ(pool->total_length_bytes, row.geometry.k_mapping_bytes);
      for (size_t length : pool->segment_lengths) {
        EXPECT_EQ(length % row.geometry.kv_block_bytes, 0U);
        EXPECT_LE(length, row.capacity);
      }
    }
    const auto& table = *plan.resources.table_segments;
    EXPECT_EQ(table.element_bytes, sizeof(uint32_t));
    EXPECT_EQ(table.segment_lengths, row.table_lengths);
    EXPECT_EQ(table.segment_count, row.table_lengths.size());
    EXPECT_EQ(table.total_length_bytes, row.geometry.block_table_bytes);
    for (size_t length : table.segment_lengths) {
      EXPECT_EQ(length % sizeof(uint32_t), 0U);
      EXPECT_LE(length, row.capacity);
    }
  }
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

TEST(LlmMetalBackendTest, PlannerRejectsInvalidLayoutsAndResourceCaps) {
  LlmMetalPlanningLimits limits;
  limits.segment_capacity_bytes = 63;
  limits.segment_slots_per_pool = 8;
  const LlmGeometry geometry = paged_geometry(64, 17, 16, 1, 1, 4);
  ASSERT_TRUE(geometry.valid) << geometry.reason_code;
  struct Case {
    LlmMetalResourcePlanRequest request;
    const char* reason;
  };
  std::vector<Case> cases;
  LlmMetalResourcePlanRequest request = resource_request(geometry, limits);
  cases.push_back({request, LlmBackendReason::PAGED_BLOCK_EXCEEDS_SEGMENT_CAPACITY});
  request.limits.segment_capacity_bytes = 64;
  request.paged_layout.reset();
  cases.push_back({request, LlmMetalPlanReason::PAGED_LAYOUT_REQUIRED});
  request.paged_layout = paged_layout_for(paged_geometry(64, 18, 16, 1, 1, 4));
  cases.push_back({request, LlmMetalPlanReason::PAGED_LAYOUT_MISMATCH});
  request.geometry.decode.reset();
  cases.push_back({request, LlmMetalPlanReason::INVALID_GEOMETRY});
  limits.segment_capacity_bytes = 100;
  limits.segment_slots_per_pool = 2;
  cases.push_back({resource_request(contiguous_geometry(201), limits), LlmBackendReason::SEGMENT_COUNT_CAP_EXCEEDED});
  for (const auto& row : cases) {
    SCOPED_TRACE(row.reason);
    const auto plan = build_llm_metal_execution_plan(row.request);
    EXPECT_FALSE(plan.valid);
    EXPECT_EQ(plan.reason_code, row.reason);
  }
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
  EXPECT_EQ(plan.msl_revision, kCanonicalKernelRevision);
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
  LlmMetalResourcePlanRequest different_serial_cap = request;
  ++different_serial_cap.limits.maximum_serial_range_visits_per_lane_per_task;
  const LlmMetalExecutionPlan different_serial_identity =
      build_llm_metal_execution_plan(different_serial_cap);
  ASSERT_TRUE(different_serial_identity.valid)
      << different_serial_identity.reason_code;
  EXPECT_NE(different_serial_identity.identity, plan.identity);

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
  request.serial_range_visits_per_lane = 12;
  request.pipeline = {16, 128};
  request.limits.threads_per_threadgroup_cap = 64;
  request.limits.maximum_threadgroups_per_grid = 3;
  request.limits.maximum_owner_ordinals_per_threadgroup = 2;
  request.limits.maximum_vector_iterations_per_lane_per_visit = 2;
  request.limits.maximum_serial_range_visits_per_lane_per_task = 12;
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
  LlmMetalGridRequest different_serial_cap = request;
  ++different_serial_cap.limits.maximum_serial_range_visits_per_lane_per_task;
  const LlmMetalGridPlan different_serial_identity =
      build_llm_metal_grid_plan(different_serial_cap);
  ASSERT_TRUE(different_serial_identity.valid)
      << different_serial_identity.reason_code;
  EXPECT_NE(different_serial_identity.identity, exact.identity);

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
  --request.paged_semantic_lookups;
  ++request.serial_range_visits_per_lane;
  EXPECT_EQ(build_llm_metal_grid_plan(request).reason_code,
            LlmMetalPlanReason::SERIAL_RANGE_VISIT_CAP_EXCEEDED);
}

TEST(LlmMetalBackendTest,
     WeightGridReportsExactVectorModuloThreadgroupCostsAndTail) {
  constexpr size_t kWeightBytes = 181;
  constexpr size_t kWorkUnits = 2;
  LlmMetalGridRequest request;
  request.owner_count = 2;
  request.visit_bytes = kWeightBytes;
  request.work_units = kWorkUnits;
  request.pipeline = {4, 4};
  request.limits.threads_per_threadgroup_cap = 4;
  request.limits.maximum_threadgroups_per_grid = 2;
  request.limits.maximum_owner_ordinals_per_threadgroup = 1;
  request.limits.maximum_vector_iterations_per_lane_per_visit = 3;
  request.limits.maximum_work_units_per_dispatch = kWorkUnits;

  const LlmMetalGridPlan plan =
      build_llm_metal_weight_grid_plan(request, kWeightBytes);
  ASSERT_TRUE(plan.valid) << plan.reason_code;
  ASSERT_EQ(plan.actual_threadgroups, 2U);
  ASSERT_EQ(plan.threads_per_threadgroup, 4U);

  std::vector<size_t> independent_costs(plan.actual_threadgroups, 0);
  const size_t grid_threads =
      plan.actual_threadgroups * plan.threads_per_threadgroup;
  for (size_t byte = 0; byte < kWeightBytes; ++byte) {
    const size_t vector_index =
        byte / Constants::LLM_METAL_VECTOR_WIDTH_BYTES;
    const size_t global_lane = vector_index % grid_threads;
    const size_t threadgroup =
        global_lane / plan.threads_per_threadgroup;
    independent_costs[threadgroup] += kWorkUnits;
  }

  EXPECT_EQ(independent_costs, (std::vector<size_t>{234, 128}));
  EXPECT_EQ(plan.threadgroup_accounted_bytes, independent_costs);
  EXPECT_EQ(plan.minimum_threadgroup_accounted_bytes, 128U);
  EXPECT_EQ(plan.maximum_threadgroup_accounted_bytes, 234U);
  EXPECT_EQ(plan.threadgroup_accounted_imbalance_bytes, 106U);
  EXPECT_EQ(std::accumulate(plan.threadgroup_accounted_bytes.begin(),
                            plan.threadgroup_accounted_bytes.end(),
                            size_t{0}),
            kWeightBytes * kWorkUnits);
  EXPECT_NE(plan.identity.find("weight_shard_cost_version="),
            std::string::npos);
  EXPECT_NE(plan.identity.find(
                "llm-metal-weight-vector-grid-stride-cost-v1"),
            std::string::npos);
}

TEST(LlmMetalBackendTest, InitializationPatternHelpersHaveFrozenIndependentGoldens) {
  constexpr uint64_t kSeed = UINT64_C(0x0123456789abcdef);
  EXPECT_EQ(llm_metal_contiguous_pattern_word(kSeed, 0),
            UINT32_C(0x27e347a8));
  EXPECT_EQ(llm_metal_contiguous_pattern_word(kSeed, 1),
            UINT32_C(0xc61ac161));
  EXPECT_EQ(llm_metal_contiguous_pattern_word(kSeed, UINT32_MAX),
            UINT32_C(0x89abcdef));
  EXPECT_EQ(llm_metal_paged_pattern_word(kSeed, 0, 0, 0),
            UINT32_C(0x8ce942d2));
  EXPECT_EQ(llm_metal_paged_pattern_word(kSeed, 1, 3, 7),
            UINT32_C(0x5708ea75));
  EXPECT_EQ(llm_metal_paged_pattern_word(
                kSeed, UINT32_MAX, UINT32_MAX, UINT64_MAX),
            UINT32_C(0x89abcdef));
}

TEST(LlmMetalBackendTest,
     DecodeChecksumMatchesIndependentByteOracleAcrossTailsAndScenarios) {
  for (size_t tail_bytes : {size_t{31}, size_t{32}, size_t{33}}) {
    const LlmGeometry geometry =
        contiguous_geometry(tail_bytes, 3, 2, 2, tail_bytes);
    ASSERT_TRUE(geometry.valid) << geometry.reason_code;
    LlmMemoryWorkPlan model = make_metal_model_plan(
        geometry, build_llm_metal_execution_plan(resource_request(geometry)),
        "llm-metal-independent-byte-oracle-" +
            std::to_string(tail_bytes));
    ASSERT_TRUE(model.valid) << model.reason_code;

    std::optional<LlmMetalMod32Lane> weights_only_lane;
    std::optional<LlmMetalMod32Lane> mixed_lane;
    for (LlmScenario scenario : {LlmScenario::WeightsOnly,
                                 LlmScenario::KvOnly,
                                 LlmScenario::Mixed}) {
      const LlmScenarioWorkPlan scenario_plan =
          build_llm_scenario_work_plan(model, scenario, 2, true);
      ASSERT_TRUE(scenario_plan.valid) << scenario_plan.reason_code;
      const LlmMetalChecksumOracle oracle =
          calculate_llm_metal_decode_contiguous_checksum(model,
                                                         scenario_plan);
      ASSERT_TRUE(oracle.valid) << oracle.reason_code;
      const LlmMetalDualMod32Checksum independent =
          independent_decode_checksum_byte_by_byte(model, scenario_plan);
      SCOPED_TRACE(tail_bytes);
      SCOPED_TRACE(static_cast<int>(scenario));
      EXPECT_TRUE(equal_llm_metal_checksum(oracle.checksum, independent));
      if (scenario == LlmScenario::WeightsOnly) {
        weights_only_lane = oracle.checksum.weight;
        EXPECT_EQ(oracle.checksum.k.a, 0U);
        EXPECT_EQ(oracle.checksum.k.b, 0U);
        EXPECT_EQ(oracle.checksum.v.a, 0U);
        EXPECT_EQ(oracle.checksum.v.b, 0U);
      } else if (scenario == LlmScenario::KvOnly) {
        EXPECT_EQ(oracle.checksum.weight.a, 0U);
        EXPECT_EQ(oracle.checksum.weight.b, 0U);
      } else {
        mixed_lane = oracle.checksum.weight;
      }
    }
    ASSERT_TRUE(weights_only_lane.has_value());
    ASSERT_TRUE(mixed_lane.has_value());
    EXPECT_TRUE(weights_only_lane->a != mixed_lane->a ||
                weights_only_lane->b != mixed_lane->b)
        << "scenario-derived seed must domain-separate the shared weight bytes";
  }
}

TEST(LlmMetalBackendTest,
     PrefillChecksumMatchesIndependentByteOracleAcrossTilesTailsAndScenarios) {
  std::vector<LlmGeometry> geometries;
  for (size_t tail_bytes : {31U, 32U, 33U}) {
    for (const auto [prompt, tile] : {std::pair<size_t, size_t>{5, 1}, {5, 5}, {5, 2}}) {
      geometries.push_back(prefill_contiguous_geometry(tail_bytes, prompt, tile, 2, 2, tail_bytes));
    }
  }
  geometries.push_back(prefill_contiguous_geometry(1, 3, 2, 2, 1, 3));
  for (const auto& geometry : geometries) {
    ASSERT_TRUE(geometry.valid) << geometry.reason_code;
    LlmMemoryWorkPlan model =
        make_metal_model_plan(geometry, build_llm_metal_execution_plan(resource_request(geometry)),
                              "llm-metal-prefill-independent-byte-oracle");
    ASSERT_TRUE(model.valid) << model.reason_code;
    ASSERT_TRUE(model.prefill_plan.has_value());

    for (LlmScenario scenario : {LlmScenario::WeightsOnly, LlmScenario::KvOnly, LlmScenario::Mixed}) {
      const LlmScenarioWorkPlan scenario_plan = build_llm_scenario_work_plan(model, scenario, 2, true);
      ASSERT_TRUE(scenario_plan.valid) << scenario_plan.reason_code;
      const LlmMetalChecksumOracle oracle = calculate_llm_metal_prefill_contiguous_checksum(model, scenario_plan);
      ASSERT_TRUE(oracle.valid) << oracle.reason_code;
      const LlmMetalDualMod32Checksum independent = independent_prefill_checksum_byte_by_byte(model, scenario_plan);
      SCOPED_TRACE(geometry.active_weight_bytes_per_work_unit);
      SCOPED_TRACE(geometry.prefill->attention_query_tile_tokens);
      SCOPED_TRACE(static_cast<int>(scenario));
      EXPECT_TRUE(equal_llm_metal_checksum(oracle.checksum, independent));
      if (scenario == LlmScenario::WeightsOnly) {
        EXPECT_EQ(oracle.checksum.k.a, 0U);
        EXPECT_EQ(oracle.checksum.k.b, 0U);
        EXPECT_EQ(oracle.checksum.v.a, 0U);
        EXPECT_EQ(oracle.checksum.v.b, 0U);
      } else if (scenario == LlmScenario::KvOnly) {
        EXPECT_EQ(oracle.checksum.weight.a, 0U);
        EXPECT_EQ(oracle.checksum.weight.b, 0U);
      }
    }
  }
}

TEST(LlmMetalBackendTest,
     PrefillSerialRangeVisitCountIsExactAndCapsTheOracleBeforeEnumeration) {
  const LlmGeometry geometry =
      prefill_contiguous_geometry(33, 5, 2, 2, 2, 3);
  ASSERT_TRUE(geometry.valid) << geometry.reason_code;
  LlmMemoryWorkPlan model = make_metal_model_plan(
      geometry, build_llm_metal_execution_plan(resource_request(geometry)),
      "llm-metal-prefill-serial-range-count");
  ASSERT_TRUE(model.valid) << model.reason_code;

  for (const auto [scenario, expected] :
       {std::pair<LlmScenario, size_t>{LlmScenario::WeightsOnly, 6},
        {LlmScenario::KvOnly, 192}, {LlmScenario::Mixed, 198}}) {
    const LlmScenarioWorkPlan scenario_plan =
        build_llm_scenario_work_plan(model, scenario, 3, true);
    ASSERT_TRUE(scenario_plan.valid) << scenario_plan.reason_code;
    size_t visits = 999;
    ASSERT_TRUE(calculate_llm_metal_prefill_serial_range_visits_per_lane(
        model, scenario_plan, visits));
    SCOPED_TRACE(static_cast<int>(scenario));
    EXPECT_EQ(visits, expected);
  }

  size_t visits_per_work_unit = 0;
  ASSERT_TRUE(calculate_llm_metal_prefill_serial_range_visits_per_work_unit(
      geometry, LlmScenario::Mixed, visits_per_work_unit));
  ASSERT_EQ(visits_per_work_unit, 66U);
  const LlmScenarioLimits limits = calculate_llm_scenario_limits(
      geometry, LlmScenario::Mixed, LlmMemoryBackend::Metal);
  ASSERT_TRUE(limits.valid) << limits.reason_code;
  EXPECT_EQ(
      limits.maximum_work_units_by_work_unit_cap,
      Constants::LLM_METAL_MAX_SERIAL_RANGE_VISITS_PER_LANE_PER_TASK /
          visits_per_work_unit);
  const LlmScenarioWorkPlan exact = build_llm_scenario_work_plan(
      model, LlmScenario::Mixed, limits.effective_maximum_work_units, true);
  ASSERT_TRUE(exact.valid) << exact.reason_code;
  const LlmScenarioWorkPlan rejected = build_llm_scenario_work_plan(
      model, LlmScenario::Mixed, limits.effective_maximum_work_units + 1,
      true);
  EXPECT_FALSE(rejected.valid);
  EXPECT_EQ(rejected.reason_code, LlmWorkPlanReason::WORK_UNIT_CAP_EXCEEDED);

  LlmScenarioWorkPlan over_cap_scenario = exact;
  ++over_cap_scenario.work_units;
  size_t visits = 0;
  ASSERT_TRUE(calculate_llm_metal_prefill_serial_range_visits_per_lane(
      model, over_cap_scenario, visits));
  ASSERT_GT(
      visits,
      Constants::LLM_METAL_MAX_SERIAL_RANGE_VISITS_PER_LANE_PER_TASK);
  const LlmMetalChecksumOracle oracle =
      calculate_llm_metal_prefill_contiguous_checksum(model,
                                                      over_cap_scenario);
  EXPECT_FALSE(oracle.valid);
  EXPECT_EQ(oracle.reason_code,
            LlmMetalPlanReason::SERIAL_RANGE_VISIT_CAP_EXCEEDED);
}

TEST(LlmMetalBackendTest,
     PrefillVectorSpanIncludesUnalignedSecondSequenceAtLaneThreshold) {
  const LlmGeometry aligned =
      prefill_contiguous_geometry(1, 5, 5, 1, 1, 819);
  const LlmGeometry unaligned =
      prefill_contiguous_geometry(1, 5, 5, 1, 2, 819);
  ASSERT_TRUE(aligned.valid) << aligned.reason_code;
  ASSERT_TRUE(unaligned.valid) << unaligned.reason_code;
  ASSERT_TRUE(aligned.prefill.has_value());
  ASSERT_EQ(aligned.prefill->prompt_tokens *
                aligned.k_or_v_record_bytes_per_layer,
            4095U);

  size_t aligned_span_bytes = 0;
  size_t unaligned_span_bytes = 0;
  ASSERT_TRUE(calculate_llm_metal_prefill_maximum_range_vector_span_bytes(
      aligned, LlmScenario::KvOnly, aligned_span_bytes));
  ASSERT_TRUE(calculate_llm_metal_prefill_maximum_range_vector_span_bytes(
      unaligned, LlmScenario::KvOnly, unaligned_span_bytes));
  EXPECT_EQ(aligned_span_bytes,
            256 * Constants::LLM_METAL_VECTOR_WIDTH_BYTES);
  EXPECT_EQ(unaligned_span_bytes,
            257 * Constants::LLM_METAL_VECTOR_WIDTH_BYTES);

  LlmMetalGridRequest request;
  request.owner_count = 1;
  request.visit_bytes = unaligned_span_bytes;
  request.work_units = 1;
  request.pipeline = {32, 256};
  request.limits.maximum_threadgroups_per_grid = 1;
  request.limits.maximum_vector_iterations_per_lane_per_visit = 2;
  const LlmMetalGridPlan exact = build_llm_metal_grid_plan(request);
  ASSERT_TRUE(exact.valid) << exact.reason_code;
  EXPECT_EQ(exact.threads_per_threadgroup, 256U);
  EXPECT_EQ(exact.vector_iterations_per_lane_per_visit, 2U);

  request.limits.maximum_vector_iterations_per_lane_per_visit = 1;
  const LlmMetalGridPlan rejected = build_llm_metal_grid_plan(request);
  EXPECT_FALSE(rejected.valid);
  EXPECT_EQ(rejected.reason_code,
            LlmMetalPlanReason::VECTOR_ITERATION_CAP_EXCEEDED);
}

TEST(LlmMetalBackendTest,
     PrefillPagedVectorSpanCoversUnalignedBlocksTerminalAndWeightLayers) {
  const LlmGeometry block_ranges =
      prefill_paged_geometry(1, 3, 2, 2, 1, 2, 31);
  const LlmGeometry terminal_ranges =
      prefill_paged_geometry(1, 1, 1, 2, 1, 2, 31);
  const LlmGeometry weight_layers =
      prefill_paged_geometry(65, 1, 1, 1, 2, 1, 1);
  ASSERT_TRUE(block_ranges.valid) << block_ranges.reason_code;
  ASSERT_TRUE(terminal_ranges.valid) << terminal_ranges.reason_code;
  ASSERT_TRUE(weight_layers.valid) << weight_layers.reason_code;
  ASSERT_EQ(block_ranges.kv_block_bytes, 62U);
  ASSERT_EQ(block_ranges.last_block_valid_bytes, 31U);
  ASSERT_EQ(terminal_ranges.last_block_valid_bytes, 31U);

  size_t block_span_bytes = 0;
  size_t terminal_span_bytes = 0;
  size_t weight_span_bytes = 0;
  ASSERT_TRUE(calculate_llm_metal_prefill_maximum_range_vector_span_bytes(
      block_ranges, LlmScenario::KvOnly, block_span_bytes));
  ASSERT_TRUE(calculate_llm_metal_prefill_maximum_range_vector_span_bytes(
      terminal_ranges, LlmScenario::KvOnly, terminal_span_bytes));
  ASSERT_TRUE(calculate_llm_metal_prefill_maximum_range_vector_span_bytes(
      weight_layers, LlmScenario::WeightsOnly, weight_span_bytes));
  EXPECT_EQ(block_span_bytes, 80U);
  EXPECT_EQ(terminal_span_bytes, 48U);
  EXPECT_EQ(weight_span_bytes, 48U);

  LlmMetalGridRequest request;
  request.owner_count = 1;
  request.visit_bytes = block_span_bytes;
  request.work_units = 1;
  request.pipeline = {2, 2};
  request.limits.maximum_threadgroups_per_grid = 1;
  request.limits.maximum_owner_ordinals_per_threadgroup = 1;
  request.limits.maximum_vector_iterations_per_lane_per_visit = 3;
  request.limits.maximum_work_units_per_dispatch = 1;
  const LlmMetalGridPlan exact = build_llm_metal_grid_plan(request);
  ASSERT_TRUE(exact.valid) << exact.reason_code;
  EXPECT_EQ(exact.vector_iterations_per_lane_per_visit, 3U);

  request.limits.maximum_vector_iterations_per_lane_per_visit = 2;
  const LlmMetalGridPlan rejected = build_llm_metal_grid_plan(request);
  EXPECT_FALSE(rejected.valid);
  EXPECT_EQ(rejected.reason_code,
            LlmMetalPlanReason::VECTOR_ITERATION_CAP_EXCEEDED);
}

TEST(LlmMetalBackendTest,
     DecodePagedChecksumMatchesIndependentByteOracleAndDetectsEqualMultiplicitySwap) {
  const LlmGeometry geometry = paged_geometry(33, 6, 4, 2, 2, 3);
  ASSERT_TRUE(geometry.valid) << geometry.reason_code;
  LlmMemoryWorkPlan model = make_metal_model_plan(
      geometry, build_llm_metal_execution_plan(resource_request(geometry)),
      "llm-metal-paged-independent-byte-oracle");
  ASSERT_TRUE(model.valid) << model.reason_code;
  const LlmMetalExecutionPlan* execution = get_llm_metal_execution_plan(model);
  ASSERT_NE(execution, nullptr);
  ASSERT_TRUE(execution->resources.paged_layout.has_value());
  const LlmKvLayoutPlan& layout = *execution->resources.paged_layout;
  ASSERT_EQ(layout.blocks_per_sequence, 2U);
  ASSERT_EQ(layout.last_block_valid_bytes, 6U);
  ASSERT_EQ(layout.decode_append_offset_in_last_block, 3U);
  ASSERT_LT(layout.last_block_valid_bytes, layout.block_bytes);

  const LlmKvBlockTable table = materialize_llm_kv_block_table(
      layout, derive_llm_kv_permutation_seed(model.base_seed),
      Constants::LLM_KV_BLOCK_TABLE_HASH_CHUNK_ENTRIES);
  ASSERT_TRUE(table.valid) << table.reason_code;
  const LlmMetalPagedChecksumSummary summary =
      build_llm_metal_decode_paged_checksum_summary(
          model, table.entries.data(), table.entries.size());
  ASSERT_TRUE(summary.valid);
  EXPECT_EQ(summary.all_owners.count,
            layout.layer_count * layout.block_table_entries);
  EXPECT_EQ(summary.terminal_owners.count,
            layout.layer_count * layout.batch_size);

  for (LlmScenario scenario : {LlmScenario::WeightsOnly,
                               LlmScenario::KvOnly,
                               LlmScenario::Mixed}) {
    const LlmScenarioWorkPlan scenario_plan =
        build_llm_scenario_work_plan(model, scenario, 2, true);
    ASSERT_TRUE(scenario_plan.valid) << scenario_plan.reason_code;
    const LlmMetalChecksumOracle oracle =
        calculate_llm_metal_decode_paged_checksum(model, scenario_plan,
                                                  summary);
    ASSERT_TRUE(oracle.valid) << oracle.reason_code;
    const LlmMetalDualMod32Checksum independent =
        independent_paged_decode_checksum_byte_by_byte(
            model, scenario_plan, table.entries);
    SCOPED_TRACE(static_cast<int>(scenario));
    EXPECT_TRUE(equal_llm_metal_checksum(oracle.checksum, independent))
        << "oracle K=" << oracle.checksum.k.a << "," << oracle.checksum.k.b
        << " V=" << oracle.checksum.v.a << "," << oracle.checksum.v.b
        << " independent K=" << independent.k.a << "," << independent.k.b
        << " V=" << independent.v.a << "," << independent.v.b;
  }

  std::vector<uint32_t> swapped_entries = table.entries;
  const size_t first_nonterminal = 0;
  const size_t second_nonterminal = layout.blocks_per_sequence;
  ASSERT_NE(swapped_entries[first_nonterminal],
            swapped_entries[second_nonterminal]);
  std::swap(swapped_entries[first_nonterminal],
            swapped_entries[second_nonterminal]);
  EXPECT_EQ(std::accumulate(table.entries.begin(), table.entries.end(), 0U),
            std::accumulate(swapped_entries.begin(), swapped_entries.end(),
                            0U));
  const LlmMetalPagedChecksumSummary swapped_summary =
      build_llm_metal_decode_paged_checksum_summary(
          model, swapped_entries.data(), swapped_entries.size());
  ASSERT_TRUE(swapped_summary.valid);
  const LlmScenarioWorkPlan kv_plan = build_llm_scenario_work_plan(
      model, LlmScenario::KvOnly, 2, true);
  ASSERT_TRUE(kv_plan.valid) << kv_plan.reason_code;
  const LlmMetalChecksumOracle canonical =
      calculate_llm_metal_decode_paged_checksum(model, kv_plan, summary);
  const LlmMetalChecksumOracle swapped =
      calculate_llm_metal_decode_paged_checksum(model, kv_plan,
                                                swapped_summary);
  ASSERT_TRUE(canonical.valid);
  ASSERT_TRUE(swapped.valid);
  EXPECT_FALSE(equal_llm_metal_checksum(canonical.checksum,
                                       swapped.checksum));
}

TEST(LlmMetalBackendTest,
     PrefillPagedChecksumMatchesIndependentPermutationAndUnalignedTailOracle) {
  for (size_t record_bytes : {size_t{31}, size_t{33}}) {
    SCOPED_TRACE(record_bytes);
    const LlmGeometry geometry = prefill_paged_geometry(
        65, 5, 2, 2, 2, 2, record_bytes);
    ASSERT_TRUE(geometry.valid) << geometry.reason_code;
    ASSERT_TRUE(geometry.prefill.has_value());
    LlmMemoryWorkPlan model = make_metal_model_plan(
        geometry, build_llm_metal_execution_plan(resource_request(geometry)),
        "llm-metal-prefill-paged-independent-byte-oracle-" +
            std::to_string(record_bytes));
    ASSERT_TRUE(model.valid) << model.reason_code;
    const LlmMetalExecutionPlan* execution =
        get_llm_metal_execution_plan(model);
    ASSERT_NE(execution, nullptr);
    ASSERT_TRUE(execution->resources.paged_layout.has_value());
    const LlmKvLayoutPlan& layout = *execution->resources.paged_layout;
    ASSERT_EQ(layout.blocks_per_sequence, 3U);
    EXPECT_EQ(layout.block_bytes, 2U * record_bytes);
    EXPECT_EQ(layout.last_block_valid_bytes, record_bytes);
    EXPECT_LT(layout.last_block_valid_bytes, layout.block_bytes);
    EXPECT_NE(layout.block_bytes % sizeof(uint32_t), 0U);

    const LlmKvBlockTable table = materialize_llm_kv_block_table(
        layout, derive_llm_kv_permutation_seed(model.base_seed),
        Constants::LLM_KV_BLOCK_TABLE_HASH_CHUNK_ENTRIES);
    ASSERT_TRUE(table.valid) << table.reason_code;
    const LlmMetalPrefillPagedChecksumSummary summary =
        build_llm_metal_prefill_paged_checksum_summary(
            model, table.entries.data(), table.entries.size());
    ASSERT_TRUE(summary.valid);
    EXPECT_EQ(summary.write_lookups.count, 12U);
    EXPECT_EQ(summary.read_lookups.count, 24U);
    EXPECT_EQ(summary.read_tile_ordinal_sum, 56U);

    for (LlmScenario scenario : {LlmScenario::WeightsOnly,
                                 LlmScenario::KvOnly,
                                 LlmScenario::Mixed}) {
      const LlmScenarioWorkPlan scenario_plan =
          build_llm_scenario_work_plan(model, scenario, 2, true);
      ASSERT_TRUE(scenario_plan.valid) << scenario_plan.reason_code;
      if (scenario != LlmScenario::WeightsOnly) {
        EXPECT_EQ(scenario_plan.layout_metadata_lookup_count_per_work_unit,
                  60U);
      }
      const LlmMetalChecksumOracle oracle =
          calculate_llm_metal_prefill_paged_checksum(model, scenario_plan,
                                                      summary);
      ASSERT_TRUE(oracle.valid) << oracle.reason_code;
      const LlmMetalDualMod32Checksum independent =
          independent_paged_prefill_checksum_byte_by_byte(
              model, scenario_plan, table.entries);
      SCOPED_TRACE(static_cast<int>(scenario));
      EXPECT_TRUE(equal_llm_metal_checksum(oracle.checksum, independent))
          << "oracle K=" << oracle.checksum.k.a << ","
          << oracle.checksum.k.b << " V=" << oracle.checksum.v.a << ","
          << oracle.checksum.v.b << " independent K=" << independent.k.a
          << "," << independent.k.b << " V=" << independent.v.a << ","
          << independent.v.b;
    }

    std::vector<uint32_t> swapped_entries = table.entries;
    ASSERT_GE(swapped_entries.size(), 4U);
    ASSERT_NE(swapped_entries[0], swapped_entries[3]);
    std::swap(swapped_entries[0], swapped_entries[3]);
    const LlmMetalPrefillPagedChecksumSummary swapped_summary =
        build_llm_metal_prefill_paged_checksum_summary(
            model, swapped_entries.data(), swapped_entries.size());
    ASSERT_TRUE(swapped_summary.valid);
    EXPECT_EQ(swapped_summary.write_lookups.count,
              summary.write_lookups.count);
    EXPECT_EQ(swapped_summary.read_lookups.count,
              summary.read_lookups.count);
    const LlmScenarioWorkPlan kv_plan = build_llm_scenario_work_plan(
        model, LlmScenario::KvOnly, 2, true);
    ASSERT_TRUE(kv_plan.valid) << kv_plan.reason_code;
    const LlmMetalChecksumOracle canonical =
        calculate_llm_metal_prefill_paged_checksum(model, kv_plan, summary);
    const LlmMetalChecksumOracle swapped =
        calculate_llm_metal_prefill_paged_checksum(model, kv_plan,
                                                    swapped_summary);
    ASSERT_TRUE(canonical.valid);
    ASSERT_TRUE(swapped.valid);
    const LlmMetalDualMod32Checksum coherently_wrong_address =
        independent_paged_prefill_checksum_byte_by_byte(
            model, kv_plan, table.entries, 1);
    EXPECT_FALSE(equal_llm_metal_checksum(
        canonical.checksum, coherently_wrong_address));
    EXPECT_TRUE(equal_llm_metal_checksum(
        swapped.checksum,
        independent_paged_prefill_checksum_byte_by_byte(
            model, kv_plan, swapped_entries)));
    EXPECT_FALSE(equal_llm_metal_checksum(canonical.checksum,
                                         swapped.checksum));
  }
}

TEST(LlmMetalBackendTest,
     DecodeChecksumCrossesTheCanonicalSegmentBoundaryWithoutAllocation) {
  const size_t capacity = Constants::LLM_METAL_SEGMENT_CAPACITY_BYTES;
  const size_t weight_bytes = capacity + 33;
  const LlmGeometry geometry = contiguous_geometry(weight_bytes);
  ASSERT_TRUE(geometry.valid) << geometry.reason_code;
  const LlmMetalExecutionPlan execution =
      build_llm_metal_execution_plan(resource_request(geometry));
  ASSERT_TRUE(execution.valid) << execution.reason_code;
  EXPECT_EQ(execution.resources.weight_segments.segment_lengths,
            (std::vector<size_t>{capacity, 33}));
  LlmMemoryWorkPlan model = make_metal_model_plan(
      geometry, LlmMetalExecutionPlan(execution),
      "llm-metal-canonical-segment-boundary-oracle");
  const LlmScenarioWorkPlan scenario = build_llm_scenario_work_plan(
      model, LlmScenario::WeightsOnly, 1, true);
  ASSERT_TRUE(scenario.valid) << scenario.reason_code;
  const LlmMetalChecksumOracle oracle =
      calculate_llm_metal_decode_contiguous_checksum(model, scenario);
  ASSERT_TRUE(oracle.valid) << oracle.reason_code;
  const LlmMetalMod32Lane expected = independent_large_weight_lane(
      weight_bytes, model.weight_buffer_seed, scenario.scenario_seed);
  EXPECT_EQ(oracle.checksum.weight.a, expected.a);
  EXPECT_EQ(oracle.checksum.weight.b, expected.b);
  EXPECT_EQ(oracle.checksum.k.a, 0U);
  EXPECT_EQ(oracle.checksum.k.b, 0U);
  EXPECT_EQ(oracle.checksum.v.a, 0U);
  EXPECT_EQ(oracle.checksum.v.b, 0U);
}

TEST(LlmMetalBackendTest, FoundationParameterAbiAndLayoutProbeAreExact) {
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

namespace {

template <typename Params, size_t WideCount, size_t NarrowCount, typename Validator>
void expect_parameter_probe_words(size_t parameter_bytes, const std::array<uint64_t, WideCount + NarrowCount>& offsets,
                                  const std::array<uint64_t Params::*, WideCount>& wide_members,
                                  const std::array<uint32_t Params::*, NarrowCount>& narrow_members,
                                  Validator validate) {
  Params parameters;
  constexpr size_t field_count = WideCount + NarrowCount;
  std::array<uint64_t, 4 + 2 * field_count> words{};
  words[0] = 1;
  words[1] = parameter_bytes;
  words[2] = 8;
  words[3] = field_count;
  std::copy(offsets.begin(), offsets.end(), words.begin() + 4);
  for (size_t index = 0; index < WideCount; ++index) {
    const uint64_t value = UINT64_C(0x8102030405060708) + index * UINT64_C(0x0101010101010101);
    parameters.*wide_members[index] = value;
    words[4 + field_count + index] = value;
  }
  for (size_t index = 0; index < NarrowCount; ++index) {
    const uint32_t value = UINT32_C(0x81828384) + static_cast<uint32_t>(index) * UINT32_C(0x01010101);
    parameters.*narrow_members[index] = value;
    words[4 + field_count + WideCount + index] = value;
  }
  ASSERT_TRUE(validate(parameters, words));
  for (size_t index = 0; index < words.size(); ++index) {
    auto corrupted = words;
    ++corrupted[index];
    SCOPED_TRACE(index);
    EXPECT_FALSE(validate(parameters, corrupted));
  }
}

}  // namespace

TEST(LlmMetalBackendTest, ParameterLayoutProbesValidateEveryWord) {
  {
    SCOPED_TRACE("LlmMetalDecodeContiguousParams");
    using P = LlmMetalDecodeContiguousParams;
    expect_parameter_probe_words(
        120, std::array<uint64_t, 17>{0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 108, 112, 116},
        std::array{&P::weight_bytes, &P::k_bytes, &P::v_bytes, &P::segment_capacity_bytes, &P::context_tokens,
                   &P::layer_count, &P::batch_size, &P::record_bytes, &P::work_units, &P::weight_seed, &P::k_seed,
                   &P::v_seed, &P::scenario_seed},
        std::array{&P::weight_segment_count, &P::k_segment_count, &P::v_segment_count, &P::reserved_zero},
        validate_llm_metal_decode_layout_probe);
  }
  {
    SCOPED_TRACE("LlmMetalPrefillContiguousParams");
    using P = LlmMetalPrefillContiguousParams;
    expect_parameter_probe_words(
        136, std::array<uint64_t, 19>{0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 124, 128, 132},
        std::array{&P::weight_bytes, &P::k_bytes, &P::v_bytes, &P::segment_capacity_bytes, &P::prompt_tokens,
                   &P::attention_query_tile_tokens, &P::tile_count, &P::layer_count, &P::batch_size, &P::record_bytes,
                   &P::work_units, &P::weight_seed, &P::k_seed, &P::v_seed, &P::scenario_seed},
        std::array{&P::weight_segment_count, &P::k_segment_count, &P::v_segment_count, &P::reserved_zero},
        validate_llm_metal_prefill_layout_probe);
  }
  {
    SCOPED_TRACE("LlmMetalPrefillPagedParams");
    using P = LlmMetalPrefillPagedParams;
    expect_parameter_probe_words(
        184, std::array<uint64_t, 26>{0,   8,   16,  24,  32,  40,  48,  56,  64,  72,  80,  88,  96,
                                      104, 112, 120, 128, 136, 144, 152, 160, 164, 168, 172, 176, 180},
        std::array{&P::weight_bytes,
                   &P::prompt_tokens,
                   &P::attention_query_tile_tokens,
                   &P::tile_count,
                   &P::layer_count,
                   &P::batch_size,
                   &P::record_bytes,
                   &P::work_units,
                   &P::block_tokens,
                   &P::block_bytes,
                   &P::last_block_valid_bytes,
                   &P::blocks_per_sequence,
                   &P::physical_blocks_per_layer,
                   &P::blocks_per_segment,
                   &P::table_entries_per_segment,
                   &P::segment_capacity_bytes,
                   &P::weight_seed,
                   &P::k_seed,
                   &P::v_seed,
                   &P::scenario_seed},
        std::array{&P::weight_segment_count, &P::k_segment_count, &P::v_segment_count, &P::table_segment_count,
                   &P::reserved_zero, &P::padding_zero},
        validate_llm_metal_prefill_paged_layout_probe);
  }
  {
    SCOPED_TRACE("LlmMetalDecodePagedParams");
    using P = LlmMetalDecodePagedParams;
    expect_parameter_probe_words(
        168, std::array<uint64_t, 24>{0,  8,   16,  24,  32,  40,  48,  56,  64,  72,  80,  88,
                                      96, 104, 112, 120, 128, 136, 144, 148, 152, 156, 160, 164},
        std::array{&P::weight_bytes, &P::context_tokens, &P::layer_count, &P::batch_size, &P::record_bytes,
                   &P::work_units, &P::block_bytes, &P::last_block_valid_bytes, &P::append_offset_in_last_block,
                   &P::blocks_per_sequence, &P::physical_blocks_per_layer, &P::blocks_per_segment,
                   &P::table_entries_per_segment, &P::segment_capacity_bytes, &P::weight_seed, &P::k_seed, &P::v_seed,
                   &P::scenario_seed},
        std::array{&P::weight_segment_count, &P::k_segment_count, &P::v_segment_count, &P::table_segment_count,
                   &P::reserved_zero, &P::padding_zero},
        validate_llm_metal_decode_paged_layout_probe);
  }
}

TEST(LlmMetalBackendTest,
     DecodePagedMslSourceLocksTimedLookupAndOwnerScheduleContract) {
  const std::string_view source = LlmMetalKernelContract::kSource;
  const auto require_source = [&](std::string_view token) {
    EXPECT_NE(source.find(token), std::string_view::npos) << token;
  };
  require_source("device const volatile uint* named_lane_table");
  require_source("threadgroup_barrier(mem_flags::mem_threadgroup);");
  require_source("threadgroup uint* published_physical_id [[threadgroup(1)]]");
  require_source("&resources.status_checksum[kLayoutMetadataLookupCountIndex], 1u");
  require_source("layer * params.physical_blocks_per_layer + ulong(physical_id)");
  require_source("owner += ulong(threadgroup_count)");
  require_source("kChecksumPagedPairMultiplier * logical * physical");
  require_source("llm_metal_validate_decode_paged_appends_padding");
  EXPECT_EQ(source.find("atomic_ulong"), std::string_view::npos);

  const size_t lookup_start = source.find("inline uint paged_timed_table_lookup");
  const size_t lookup_end = source.find("inline void mix_paged_lookup", lookup_start);
  ASSERT_NE(lookup_start, std::string_view::npos);
  ASSERT_NE(lookup_end, std::string_view::npos);
  const std::string_view lookup =
      source.substr(lookup_start, lookup_end - lookup_start);
  const size_t first_barrier =
      lookup.find("threadgroup_barrier(mem_flags::mem_threadgroup);");
  ASSERT_NE(first_barrier, std::string_view::npos);
  EXPECT_NE(lookup.find("threadgroup_barrier(mem_flags::mem_threadgroup);",
                        first_barrier + 1),
            std::string_view::npos);
}

TEST(LlmMetalBackendTest,
     PrefillPagedMslSourceLocksRowWritesThenPerTileKThenVContract) {
  const std::string_view source = LlmMetalKernelContract::kSource;
  const auto require_source = [&](std::string_view token) {
    EXPECT_NE(source.find(token), std::string_view::npos) << token;
  };
  require_source("#if LLM_METAL_PREFILL_PAGED");
  require_source("llm_metal_probe_prefill_paged_parameter_layout");
  require_source("llm_metal_prefill_paged_weights_only");
  require_source("llm_metal_prefill_paged_kv_only");
  require_source("llm_metal_prefill_paged_mixed");
  require_source("llm_metal_validate_prefill_paged_writes_padding");
  require_source(
      "constant uint kChecksumMetalPrefillPagedProfileDomain = 0x4d505031u;");
  require_source(
      "constant uint kChecksumPagedPrefillWriteLookupVisit = 0x5050574cu;");
  require_source(
      "constant uint kChecksumPagedPrefillKeyLookupVisit = 0x50504b4cu;");
  require_source(
      "constant uint kChecksumPagedPrefillValueLookupVisit = 0x5050564cu;");
  require_source(
      "logical_table_index / params.table_entries_per_segment");
  require_source("global_block / params.blocks_per_segment");
  require_source("prefill_paged_physical_address_token(");
  require_source("kChecksumPagedAddressPairMultiplier * logical *");
  require_source("global_block, segment, segment_local_block_base,");
  require_source("resources.key_segments[segment][block_base + block_byte]");
  require_source(
      "resources.value_segments[segment][block_base + block_byte]");
  require_source("const ulong logical_word = vector_index * 4ul + ulong(lane)");
  require_source("range_word_mask(logical_range_start,");
  require_source("const ulong samples_per_block = 6ul");
  require_source("const ulong padding_chunk_count =");
  EXPECT_EQ(source.find("atomic_ulong"), std::string_view::npos);

  const size_t validation_start = source.find(
      "kernel void llm_metal_validate_prefill_paged_writes_padding");
  const size_t validation_end = source.find(
      "kernel void llm_metal_probe_prefill_paged_parameter_layout",
      validation_start);
  ASSERT_NE(validation_start, std::string_view::npos);
  ASSERT_NE(validation_end, std::string_view::npos);
  const std::string_view validation =
      source.substr(validation_start, validation_end - validation_start);
  EXPECT_NE(validation.find(
                "padding_chunk = ulong(global_id)"),
            std::string_view::npos);
  EXPECT_NE(validation.find(
                "padding_chunk += ulong(grid_size)"),
            std::string_view::npos);
  EXPECT_EQ(validation.find(
                "for (ulong sequence = 0ul; sequence < sequence_count;"),
            std::string_view::npos);

  const size_t lookup_start =
      source.find("inline uint prefill_paged_timed_table_lookup");
  const size_t lookup_end =
      source.find("inline void mix_prefill_paged_lookup", lookup_start);
  ASSERT_NE(lookup_start, std::string_view::npos);
  ASSERT_NE(lookup_end, std::string_view::npos);
  const std::string_view lookup =
      source.substr(lookup_start, lookup_end - lookup_start);
  const size_t first_lookup_barrier =
      lookup.find("threadgroup_barrier(mem_flags::mem_threadgroup);");
  ASSERT_NE(first_lookup_barrier, std::string_view::npos);
  EXPECT_NE(lookup.find("threadgroup_barrier(mem_flags::mem_threadgroup);",
                        first_lookup_barrier + 1),
            std::string_view::npos);
  require_source(
      "&resources.status_checksum[kLayoutMetadataLookupCountIndex], 1u");

  const size_t run_start =
      source.find("inline void run_prefill_paged_kv_owner_rows");
  const size_t run_end = source.find(
      "kernel void llm_metal_prefill_paged_weights_only", run_start);
  ASSERT_NE(run_start, std::string_view::npos);
  ASSERT_NE(run_end, std::string_view::npos);
  const std::string_view run = source.substr(run_start, run_end - run_start);
  const size_t work_loop = run.find(
      "for (ulong work_unit = 0ul; work_unit < params.work_units;");
  const size_t owner_loop = run.find("while (first_owner < owner_count)",
                                     work_loop);
  const size_t row_map = run.find(
      "const ulong row = first_owner / params.blocks_per_sequence",
      owner_loop);
  const size_t batch_map =
      run.find("const ulong batch = row % params.batch_size", row_map);
  const size_t layer_map =
      run.find("const ulong layer = row / params.batch_size", batch_map);
  const size_t write_phase = run.find("// Row write phase:", layer_map);
  const size_t write_loop = run.find(
      "for (ulong logical_block = logical_block_first;", write_phase);
  const size_t write_lookup = run.find(
      "prefill_paged_timed_table_lookup(", write_loop);
  const size_t write_mix = run.find(
      "kChecksumPagedPrefillWriteLookupVisit", write_lookup);
  const size_t write_pair =
      run.find("write_prefill_paged_block_pair(", write_mix);
  const size_t device_barrier = run.find(
      "threadgroup_barrier(mem_flags::mem_device);", write_pair);
  const size_t tile_loop =
      run.find("while (remaining_tokens != 0ul)", device_barrier);
  const size_t key_phase = run.find("// Each K visit", tile_loop);
  const size_t key_loop = run.find(
      "for (ulong logical_block = logical_block_first;", key_phase);
  const size_t key_lookup = run.find(
      "prefill_paged_timed_table_lookup(", key_loop);
  const size_t key_mix = run.find(
      "kChecksumPagedPrefillKeyLookupVisit", key_lookup);
  const size_t key_scan =
      run.find("scan_prefill_paged_key_block(", key_mix);
  const size_t value_phase =
      run.find("// The same row prefix is revisited for V", key_scan);
  const size_t value_loop = run.find(
      "for (ulong logical_block = logical_block_first;", value_phase);
  const size_t value_lookup = run.find(
      "prefill_paged_timed_table_lookup(", value_loop);
  const size_t value_mix = run.find(
      "kChecksumPagedPrefillValueLookupVisit", value_lookup);
  const size_t value_scan =
      run.find("scan_prefill_paged_value_block(", value_mix);
  const size_t cyclic_advance = run.find(
      "first_owner += owner_steps * ulong(threadgroup_count)", value_scan);
  ASSERT_NE(work_loop, std::string_view::npos);
  ASSERT_LT(work_loop, owner_loop);
  ASSERT_LT(owner_loop, row_map);
  ASSERT_LT(row_map, batch_map);
  ASSERT_LT(batch_map, layer_map);
  ASSERT_LT(layer_map, write_phase);
  ASSERT_LT(write_phase, write_loop);
  ASSERT_LT(write_loop, write_lookup);
  ASSERT_LT(write_lookup, write_mix);
  ASSERT_LT(write_mix, write_pair);
  ASSERT_LT(write_pair, device_barrier);
  ASSERT_LT(device_barrier, tile_loop);
  ASSERT_LT(tile_loop, key_phase);
  ASSERT_LT(key_phase, key_loop);
  ASSERT_LT(key_loop, key_lookup);
  ASSERT_LT(key_lookup, key_mix);
  ASSERT_LT(key_mix, key_scan);
  ASSERT_LT(key_scan, value_phase);
  ASSERT_LT(value_phase, value_loop);
  ASSERT_LT(value_loop, value_lookup);
  ASSERT_LT(value_lookup, value_mix);
  ASSERT_LT(value_mix, value_scan);
  ASSERT_LT(value_scan, cyclic_advance);
  EXPECT_NE(run.find("logical_block += ulong(threadgroup_count)",
                     write_loop),
            std::string_view::npos);

}

TEST(LlmMetalBackendTest, FactoryAndDirectConstructorExposeTheMetalBackend) {
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
  EXPECT_TRUE(preflight_estimate.valid);
  EXPECT_EQ(preflight_estimate.reason_code, LlmBackendReason::VALID);
  EXPECT_EQ(preflight_estimate.checksum_auxiliary_bytes, 0U);
  EXPECT_GT(preflight_estimate.orchestration_auxiliary_bytes, 0U);
  EXPECT_EQ(preflight_estimate.total_auxiliary_bytes,
            preflight_estimate.orchestration_auxiliary_bytes);
  EXPECT_TRUE(std::holds_alternative<std::monostate>(
      preflight_estimate.backend_evidence));

  LlmMemoryWorkPlan inactive_plan;
  inactive_plan.backend = LlmMemoryBackend::Metal;
  inactive_plan.backend_execution_plan = LlmMetalExecutionPlan{};
  const LlmBackendAuxiliaryEstimate plan_estimate = factory_backend->calculate_auxiliary_estimate(inactive_plan);
  EXPECT_TRUE(plan_estimate.valid);
  EXPECT_EQ(plan_estimate.reason_code, LlmBackendReason::VALID);
  EXPECT_EQ(plan_estimate.checksum_auxiliary_bytes, 0U);
  EXPECT_EQ(plan_estimate.orchestration_auxiliary_bytes,
            preflight_estimate.orchestration_auxiliary_bytes);
  EXPECT_EQ(plan_estimate.total_auxiliary_bytes,
            plan_estimate.orchestration_auxiliary_bytes);
  EXPECT_TRUE(std::holds_alternative<std::monostate>(
      plan_estimate.backend_evidence));

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

TEST(LlmMetalBackendTest, AuxiliaryEstimateScalesWithActiveResourceAndReferenceBacking) {
  std::unique_ptr<LlmBackend> backend = create_llm_metal_backend();
  ASSERT_NE(backend, nullptr);
  LlmAuxiliaryPreflightView smaller;
  smaller.valid = true;
  smaller.backend = LlmMemoryBackend::Metal;
  smaller.metal_planned_resource_count = 5;
  smaller.metal_persistent_resource_count = 4;
  smaller.metal_resolved_execution_plan_backing_bytes = 4096;
  smaller.metal_resolved_plan_identity_backing_bytes = 1024;
  const LlmBackendAuxiliaryEstimate smaller_estimate = backend->calculate_auxiliary_estimate(smaller);
  ASSERT_TRUE(smaller_estimate.valid) << smaller_estimate.reason_code;

  LlmAuxiliaryPreflightView larger_plan_copy = smaller;
  larger_plan_copy.metal_resolved_execution_plan_backing_bytes += 37;
  const LlmBackendAuxiliaryEstimate larger_plan_copy_estimate =
      backend->calculate_auxiliary_estimate(larger_plan_copy);
  ASSERT_TRUE(larger_plan_copy_estimate.valid)
      << larger_plan_copy_estimate.reason_code;
  EXPECT_EQ(larger_plan_copy_estimate.orchestration_auxiliary_bytes -
                smaller_estimate.orchestration_auxiliary_bytes,
            37U);

  LlmAuxiliaryPreflightView larger_plan_identity = smaller;
  larger_plan_identity.metal_resolved_plan_identity_backing_bytes += 41;
  const LlmBackendAuxiliaryEstimate larger_plan_identity_estimate =
      backend->calculate_auxiliary_estimate(larger_plan_identity);
  ASSERT_TRUE(larger_plan_identity_estimate.valid)
      << larger_plan_identity_estimate.reason_code;
  EXPECT_EQ(larger_plan_identity_estimate.orchestration_auxiliary_bytes -
                smaller_estimate.orchestration_auxiliary_bytes,
            41U);

  LlmAuxiliaryPreflightView larger_published_capacity = smaller;
  ++larger_published_capacity.metal_persistent_resource_count;
  const LlmBackendAuxiliaryEstimate larger_published_capacity_estimate =
      backend->calculate_auxiliary_estimate(larger_published_capacity);
  ASSERT_TRUE(larger_published_capacity_estimate.valid)
      << larger_published_capacity_estimate.reason_code;
  EXPECT_EQ(
      larger_published_capacity_estimate.orchestration_auxiliary_bytes -
          smaller_estimate.orchestration_auxiliary_bytes,
      sizeof(void*));

  LlmAuxiliaryPreflightView larger = smaller;
  larger.metal_planned_resource_count = 6;
  larger.metal_persistent_resource_count = 5;
  const LlmBackendAuxiliaryEstimate larger_estimate = backend->calculate_auxiliary_estimate(larger);
  ASSERT_TRUE(larger_estimate.valid) << larger_estimate.reason_code;
  constexpr size_t kMetadataStringCount = 5;
  constexpr size_t kMetadataStringCapacity = 128;
  const size_t expected_resource_growth =
      sizeof(LlmMetalResourceMetadata) +
      kMetadataStringCount * 2 * (kMetadataStringCapacity + 1) +
      sizeof(LlmMetalAllocatedResource) + 2 * sizeof(void*);
  EXPECT_EQ(larger_estimate.orchestration_auxiliary_bytes -
                smaller_estimate.orchestration_auxiliary_bytes,
            expected_resource_growth);
  EXPECT_EQ(larger_estimate.total_auxiliary_bytes,
            larger_estimate.orchestration_auxiliary_bytes);

  larger.metal_planned_resource_count = 4 * Constants::LLM_METAL_SEGMENT_SLOTS_PER_POOL + 4;
  const LlmBackendAuxiliaryEstimate invalid = backend->calculate_auxiliary_estimate(larger);
  EXPECT_FALSE(invalid.valid);
  EXPECT_EQ(invalid.reason_code, LlmExecutorReason::AUXILIARY_BYTES_OVERFLOW);
}

class LlmMetalBackendIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    backend_ = create_llm_backend(LlmMemoryBackend::Metal);
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

  void resolve_and_prepare(LlmMemoryWorkPlan& plan) {
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    const LlmBackendLifecycleResult resolved =
        backend_->resolve_execution_plan(plan);
    ASSERT_EQ(resolved.status, LlmBackendStatus::Ready)
        << resolved.reason_code;
    const LlmBackendLifecycleResult prepared =
        backend_->prepare_resources(plan);
    ASSERT_EQ(prepared.status, LlmBackendStatus::Ready)
        << prepared.reason_code << ": "
        << metal_evidence().resources.error.description;
    EXPECT_TRUE(metal_evidence().timed_results_available);
  }

  void expect_complete_scenario_task(const LlmMemoryWorkPlan& plan,
                                     LlmScenario scenario,
                                     size_t work_units,
                                     size_t minimum_threadgroups = 1) {
    const LlmScenarioWorkPlan scenario_plan = build_llm_scenario_work_plan(
        plan, scenario, work_units, true);
    ASSERT_TRUE(scenario_plan.valid) << scenario_plan.reason_code;
    const LlmRunnerTaskContext context = measurement_context(
        scenario,
        plan.phase == LlmPhase::Prefill &&
                plan.kv_layout == LlmKvLayout::Paged
            ? "phase-12-real-prefill-paged"
        : plan.phase == LlmPhase::Prefill
            ? "phase-11-real-prefill-contiguous"
            : plan.kv_layout == LlmKvLayout::Paged
                  ? "phase-10-real-decode-paged"
                  : "phase-9-real-decode-contiguous");
    const LlmTaskExecutionResult result =
        backend_->execute_task(plan, scenario_plan, context);
    SCOPED_TRACE(static_cast<int>(scenario));
    ASSERT_EQ(result.status, LlmTaskExecutionStatus::Complete)
        << result.reason_code;
    EXPECT_EQ(result.reason_code, LlmBackendReason::VALID);
    EXPECT_TRUE(result.timing.evaluated);
    EXPECT_TRUE(result.timing.valid);
    EXPECT_TRUE(std::isfinite(result.timing.elapsed_seconds));
    EXPECT_GT(result.timing.elapsed_seconds, 0.0);
    EXPECT_EQ(result.completion.planned_work_units, work_units);
    EXPECT_EQ(result.completion.completed_work_units, work_units);
    EXPECT_EQ(result.completion.completed_effective_model_payload_bytes,
              scenario_plan.effective_model_payload_bytes);
    EXPECT_EQ(result.completion.completed_layout_metadata_lookup_count,
              scenario_plan.layout_metadata_lookup_count);
    EXPECT_EQ(result.completion.completed_layout_metadata_read_bytes,
              scenario_plan.layout_metadata_read_bytes);
    EXPECT_EQ(result.completion.completed_task_accounted_bytes,
              scenario_plan.task_accounted_bytes);
    EXPECT_TRUE(result.validation.evaluated);
    EXPECT_TRUE(result.validation.valid);

    const LlmMetalTaskEvidence* task = get_llm_metal_task_evidence(result);
    ASSERT_NE(task, nullptr);
    EXPECT_TRUE(task->timed_pipeline_available);
    const std::array<std::string_view, kWorkloadPipelineCount>
        contiguous_labels = {
            "membenchmark.llm-metal.pipeline.decode-contiguous.weights-only",
            "membenchmark.llm-metal.pipeline.decode-contiguous.kv-only",
            "membenchmark.llm-metal.pipeline.decode-contiguous.mixed"};
    const std::array<std::string_view, kWorkloadPipelineCount> paged_labels = {
        "membenchmark.llm-metal.pipeline.decode-paged.weights-only",
        "membenchmark.llm-metal.pipeline.decode-paged.kv-only",
        "membenchmark.llm-metal.pipeline.decode-paged.mixed"};
    const std::array<std::string_view, kWorkloadPipelineCount> prefill_labels = {
        "membenchmark.llm-metal.pipeline.prefill-contiguous.weights-only",
        "membenchmark.llm-metal.pipeline.prefill-contiguous.kv-only",
        "membenchmark.llm-metal.pipeline.prefill-contiguous.mixed"};
    const std::array<std::string_view, kWorkloadPipelineCount>
        prefill_paged_labels = {
            "membenchmark.llm-metal.pipeline.prefill-paged.weights-only",
            "membenchmark.llm-metal.pipeline.prefill-paged.kv-only",
            "membenchmark.llm-metal.pipeline.prefill-paged.mixed"};
    const auto& labels = plan.phase == LlmPhase::Prefill &&
                                 plan.kv_layout == LlmKvLayout::Paged
                             ? prefill_paged_labels
                         : plan.phase == LlmPhase::Prefill
                             ? prefill_labels
                         : plan.kv_layout == LlmKvLayout::Paged
                             ? paged_labels
                             : contiguous_labels;
    EXPECT_EQ(task->pipeline_label,
              labels[static_cast<size_t>(scenario)]);
    EXPECT_GT(task->pipeline_thread_execution_width, 0U);
    EXPECT_GT(task->pipeline_max_total_threads_per_threadgroup, 0U);
    EXPECT_TRUE(task->grid_plan_available);
    ASSERT_TRUE(task->grid_plan.valid) << task->grid_plan.reason_code;
    EXPECT_EQ(task->checksum_algorithm_version,
              plan.component_identities.checksum_pattern_version);
    EXPECT_GE(task->grid_plan.actual_threadgroups, minimum_threadgroups);
    EXPECT_GT(task->grid_plan.threads_per_threadgroup, 0U);
    EXPECT_EQ(task->grid_plan.work_units, work_units);
    if (plan.phase == LlmPhase::Prefill &&
        (plan.kv_layout == LlmKvLayout::Contiguous ||
         scenario == LlmScenario::WeightsOnly)) {
      size_t expected_serial_range_visits = 0;
      ASSERT_TRUE(calculate_llm_metal_prefill_serial_range_visits_per_lane(
          plan, scenario_plan, expected_serial_range_visits));
      EXPECT_EQ(task->grid_plan.serial_range_visits_per_lane,
                expected_serial_range_visits);
      if (plan.kv_layout == LlmKvLayout::Contiguous) {
        EXPECT_EQ(task->grid_plan.owner_count,
                  task->grid_plan.actual_threadgroups);
        EXPECT_EQ(task->grid_plan.owner_ordinals_per_threadgroup, 1U);
      }
    }
    if ((plan.phase == LlmPhase::Prefill &&
         plan.kv_layout == LlmKvLayout::Paged) ||
        scenario == LlmScenario::WeightsOnly) {
      ASSERT_EQ(task->grid_plan.threadgroup_accounted_bytes.size(),
                task->grid_plan.actual_threadgroups);
      const auto [minimum, maximum] = std::minmax_element(
          task->grid_plan.threadgroup_accounted_bytes.begin(),
          task->grid_plan.threadgroup_accounted_bytes.end());
      EXPECT_EQ(task->grid_plan.minimum_threadgroup_accounted_bytes,
                *minimum);
      EXPECT_EQ(task->grid_plan.maximum_threadgroup_accounted_bytes,
                *maximum);
      EXPECT_EQ(task->grid_plan.threadgroup_accounted_imbalance_bytes,
                *maximum - *minimum);
      EXPECT_EQ(std::accumulate(
                    task->grid_plan.threadgroup_accounted_bytes.begin(),
                    task->grid_plan.threadgroup_accounted_bytes.end(),
                    size_t{0}),
                scenario_plan.task_accounted_bytes);
      const std::string minimum_text = std::to_string(*minimum);
      const std::string maximum_text = std::to_string(*maximum);
      EXPECT_NE(task->grid_plan.identity.find(
                    "minimum_threadgroup_accounted_bytes=" +
                    std::to_string(minimum_text.size()) + ":" +
                    minimum_text + ";"),
                std::string::npos)
          << task->grid_plan.identity;
      EXPECT_NE(task->grid_plan.identity.find(
                    "maximum_threadgroup_accounted_bytes=" +
                    std::to_string(maximum_text.size()) + ":" +
                    maximum_text + ";"),
                std::string::npos)
          << task->grid_plan.identity;
    }
    EXPECT_TRUE(task->timing_evaluated);
    EXPECT_TRUE(task->timing_valid);
    EXPECT_GT(task->gpu_start_seconds, 0.0);
    EXPECT_GT(task->gpu_end_seconds, task->gpu_start_seconds);
    EXPECT_DOUBLE_EQ(task->gpu_elapsed_seconds,
                     task->gpu_end_seconds - task->gpu_start_seconds);
    EXPECT_DOUBLE_EQ(result.timing.elapsed_seconds,
                     task->gpu_elapsed_seconds);
    EXPECT_TRUE(task->host_timing_evaluated);
    EXPECT_TRUE(std::isfinite(task->host_submit_to_completion_seconds));
    EXPECT_TRUE(std::isfinite(task->host_wait_seconds));
    EXPECT_GE(task->host_submit_to_completion_seconds, 0.0);
    EXPECT_GE(task->host_wait_seconds, 0.0);
    EXPECT_EQ(task->reset_command_buffer_count, 1U);
    EXPECT_EQ(task->timed_command_buffer_count, 1U);
    EXPECT_EQ(task->post_validation_command_buffer_count, 1U);
    EXPECT_EQ(task->timed_compute_encoder_count, 1U);
    EXPECT_EQ(task->timed_workload_dispatch_count, 1U);
    EXPECT_EQ(task->reset_command_status, "completed");
    EXPECT_EQ(task->timed_command_status, "completed");
    EXPECT_EQ(task->post_validation_command_status, "completed");
    EXPECT_TRUE(task->checksum_evaluated);
    EXPECT_TRUE(task->checksum_valid);
    LlmMetalChecksumOracle oracle;
    if (plan.phase == LlmPhase::Prefill &&
        plan.kv_layout == LlmKvLayout::Paged) {
      const LlmMetalExecutionPlan* execution =
          get_llm_metal_execution_plan(plan);
      ASSERT_NE(execution, nullptr);
      ASSERT_TRUE(execution->resources.paged_layout.has_value());
      const LlmKvBlockTable table = materialize_llm_kv_block_table(
          *execution->resources.paged_layout,
          derive_llm_kv_permutation_seed(plan.base_seed),
          Constants::LLM_KV_BLOCK_TABLE_HASH_CHUNK_ENTRIES);
      ASSERT_TRUE(table.valid) << table.reason_code;
      const LlmMetalPrefillPagedChecksumSummary summary =
          build_llm_metal_prefill_paged_checksum_summary(
              plan, table.entries.data(), table.entries.size());
      ASSERT_TRUE(summary.valid);
      oracle = calculate_llm_metal_prefill_paged_checksum(
          plan, scenario_plan, summary);
    } else if (plan.phase == LlmPhase::Prefill) {
      oracle = calculate_llm_metal_prefill_contiguous_checksum(
          plan, scenario_plan);
    } else if (plan.kv_layout == LlmKvLayout::Paged) {
      const LlmMetalExecutionPlan* execution =
          get_llm_metal_execution_plan(plan);
      ASSERT_NE(execution, nullptr);
      ASSERT_TRUE(execution->resources.paged_layout.has_value());
      const LlmKvBlockTable table = materialize_llm_kv_block_table(
          *execution->resources.paged_layout,
          derive_llm_kv_permutation_seed(plan.base_seed),
          Constants::LLM_KV_BLOCK_TABLE_HASH_CHUNK_ENTRIES);
      ASSERT_TRUE(table.valid) << table.reason_code;
      const LlmMetalPagedChecksumSummary summary =
          build_llm_metal_decode_paged_checksum_summary(
              plan, table.entries.data(), table.entries.size());
      ASSERT_TRUE(summary.valid);
      oracle = calculate_llm_metal_decode_paged_checksum(plan, scenario_plan,
                                                         summary);
    } else {
      oracle =
          calculate_llm_metal_decode_contiguous_checksum(plan, scenario_plan);
    }
    ASSERT_TRUE(oracle.valid) << oracle.reason_code;
    EXPECT_TRUE(equal_llm_metal_checksum(task->expected_checksum,
                                         oracle.checksum));
    EXPECT_TRUE(equal_llm_metal_checksum(task->expected_checksum,
                                         task->actual_checksum));
    const bool append_applicable = scenario != LlmScenario::WeightsOnly;
    EXPECT_EQ(task->kv_write_validation_evaluated, append_applicable);
    EXPECT_TRUE(task->kv_write_validation_valid);
    const bool padding_applicable =
        append_applicable && plan.kv_layout == LlmKvLayout::Paged &&
        plan.geometry.last_block_valid_bytes < plan.geometry.kv_block_bytes;
    EXPECT_EQ(task->padding_canary_applicable, padding_applicable);
    EXPECT_EQ(task->padding_canary_evaluated, padding_applicable);
    EXPECT_EQ(task->padding_canary_valid, padding_applicable);
    EXPECT_TRUE(task->post_validation_evaluated);
    EXPECT_TRUE(task->post_validation_valid);
  }

  void expect_exact_tail_all_scenarios(size_t tail_bytes) {
    const LlmGeometry geometry =
        contiguous_geometry(tail_bytes, 1, 1, 1, tail_bytes);
    ASSERT_TRUE(geometry.valid) << geometry.reason_code;
    ASSERT_EQ(geometry.active_weight_bytes_per_work_unit, tail_bytes);
    ASSERT_EQ(geometry.k_mapping_bytes, tail_bytes);
    ASSERT_EQ(geometry.v_mapping_bytes, tail_bytes);
    LlmMemoryWorkPlan plan = build_device_plan(
        geometry, "llm-metal-phase9-exact-tail-" +
                      std::to_string(tail_bytes));
    resolve_and_prepare(plan);
    for (LlmScenario scenario : {LlmScenario::WeightsOnly,
                                 LlmScenario::KvOnly,
                                 LlmScenario::Mixed}) {
      expect_complete_scenario_task(plan, scenario, 2);
    }
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
  EXPECT_EQ(capability.kernel_revision, kCanonicalKernelRevision);
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
      "membenchmark.llm-metal.pipeline.layout-probe",
      "membenchmark.llm-metal.pipeline.decode-contiguous-layout-probe",
      "membenchmark.llm-metal.pipeline.validate-bytes",
      "membenchmark.llm-metal.pipeline.validate-table"};
  for (std::string_view label : labels) {
    const auto found = std::find_if(
        capability.foundation_pipelines.begin(), capability.foundation_pipelines.end(),
        [label](const LlmMetalPipelineEvidence& pipeline) { return std::string_view(pipeline.label) == label; });
    ASSERT_NE(found, capability.foundation_pipelines.end()) << label;
    EXPECT_GT(found->thread_execution_width, 0U);
    EXPECT_GT(found->max_total_threads_per_threadgroup, 0U);
  }
  ASSERT_EQ(metal_evidence().workload_pipelines.size(),
            kWorkloadPipelineCount);
  const std::array<std::string_view, kWorkloadPipelineCount>
      workload_labels = {
          "membenchmark.llm-metal.pipeline.decode-contiguous.weights-only",
          "membenchmark.llm-metal.pipeline.decode-contiguous.kv-only",
          "membenchmark.llm-metal.pipeline.decode-contiguous.mixed"};
  for (std::string_view label : workload_labels) {
    const auto found = std::find_if(
        metal_evidence().workload_pipelines.begin(),
        metal_evidence().workload_pipelines.end(),
        [label](const LlmMetalPipelineEvidence& pipeline) {
          return std::string_view(pipeline.label) == label;
        });
    ASSERT_NE(found, metal_evidence().workload_pipelines.end()) << label;
    EXPECT_GT(found->thread_execution_width, 0U);
    EXPECT_GT(found->max_total_threads_per_threadgroup, 0U);
  }
  EXPECT_FALSE(metal_evidence().timed_results_available);
}

TEST_F(LlmMetalBackendIntegrationTest,
       PrivateResourcesAndAllDecodeContiguousScenariosIntegration) {
  LlmMemoryWorkPlan plan =
      build_device_plan(contiguous_geometry(4097, 17, 2, 2, 8),
                        "llm-metal-phase9-small-device-plan");
  resolve_and_prepare(plan);

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
  EXPECT_TRUE(metal_evidence().timed_results_available);

  const size_t published_metadata_count = resources.resources.size();
  const uint64_t allocation_peak = resources.current_allocated_size_peak;
  const LlmBackendLifecycleResult prepared_again = backend_->prepare_resources(plan);
  EXPECT_EQ(prepared_again.status, LlmBackendStatus::Ready);
  EXPECT_EQ(metal_evidence().resources.resources.size(), published_metadata_count);
  EXPECT_EQ(metal_evidence().resources.current_allocated_size_peak, allocation_peak);

  for (LlmScenario scenario : {LlmScenario::WeightsOnly,
                               LlmScenario::KvOnly,
                               LlmScenario::Mixed}) {
    expect_complete_scenario_task(plan, scenario, 2);
  }
}

TEST_F(LlmMetalBackendIntegrationTest, DecodeContiguousExactTailsIntegration) {
  for (size_t tail : {31U, 32U, 33U}) {
    SCOPED_TRACE(tail);
    backend_ = create_llm_metal_backend();
    ASSERT_EQ(backend_->initialize(metal_config()).status, LlmBackendStatus::Ready);
    expect_exact_tail_all_scenarios(tail);
  }
}

TEST_F(LlmMetalBackendIntegrationTest, DecodeUsesMultipleThreadgroupsWorkUnitsLayersAndBatchesIntegration) {
  struct Case {
    size_t weight_bytes;
    size_t context_tokens;
    size_t record_bytes;
    LlmScenario scenario;
    size_t work_units;
  };
  const std::array<Case, 2> cases = {
      {{2 * Constants::BYTES_PER_MB + 33, 17, 8, LlmScenario::Mixed, 3}, {33, 32769, 33, LlmScenario::KvOnly, 2}}};
  for (const Case& test_case : cases) {
    SCOPED_TRACE(static_cast<int>(test_case.scenario));
    backend_ = create_llm_metal_backend();
    ASSERT_EQ(backend_->initialize(metal_config()).status, LlmBackendStatus::Ready);
    LlmMemoryWorkPlan plan = build_device_plan(
        contiguous_geometry(test_case.weight_bytes, test_case.context_tokens, 2, 2, test_case.record_bytes),
        "llm-metal-decode-multigroup-plan");
    resolve_and_prepare(plan);
    expect_complete_scenario_task(plan, test_case.scenario, test_case.work_units, 2);
  }
}

TEST_F(LlmMetalBackendIntegrationTest,
       PrefillZeroLengthWeightLayersAreNoOpsIntegration) {
  LlmMemoryConfig config = metal_config();
  config.phase = LlmPhase::Prefill;
  backend_ = create_llm_metal_backend();
  ASSERT_NE(backend_, nullptr);
  const LlmBackendLifecycleResult initialization = backend_->initialize(config);
  ASSERT_EQ(initialization.status, LlmBackendStatus::Ready)
      << initialization.reason_code;
  LlmMemoryWorkPlan plan = build_device_plan(
      prefill_contiguous_geometry(1, 3, 2, 2, 1, 3),
      "llm-metal-phase11-prefill-zero-length-weight-layer");
  resolve_and_prepare(plan);
  expect_complete_scenario_task(plan, LlmScenario::Mixed, 1);
}

TEST_F(LlmMetalBackendIntegrationTest,
       PrefillMixedUsesMultipleThreadgroupsWorkUnitsLayersBatchesAndRemainderTileIntegration) {
  backend_ = create_llm_metal_backend();
  ASSERT_NE(backend_, nullptr);
  LlmMemoryConfig config = metal_config();
  config.phase = LlmPhase::Prefill;
  ASSERT_EQ(backend_->initialize(config).status, LlmBackendStatus::Ready);
  const LlmGeometry geometry = prefill_contiguous_geometry(
      2 * Constants::BYTES_PER_MB + 33, 32769, 16384, 2, 2, 33);
  ASSERT_TRUE(geometry.valid) << geometry.reason_code;
  ASSERT_TRUE(geometry.prefill.has_value());
  EXPECT_EQ(geometry.prefill->tile_count, 3U);
  LlmMemoryWorkPlan plan = build_device_plan(
      geometry, "llm-metal-phase11-prefill-multigroup");
  resolve_and_prepare(plan);
  expect_complete_scenario_task(plan, LlmScenario::Mixed, 3, 2);
}

TEST_F(LlmMetalBackendIntegrationTest,
       PrefillKvPoolCrossesCanonicalSegmentBoundaryIntegration) {
  backend_ = create_llm_metal_backend();
  ASSERT_NE(backend_, nullptr);
  LlmMemoryConfig config = metal_config();
  config.phase = LlmPhase::Prefill;
  ASSERT_EQ(backend_->initialize(config).status, LlmBackendStatus::Ready);
  const size_t record_bytes =
      Constants::LLM_METAL_SEGMENT_CAPACITY_BYTES / 2 + 1;
  const LlmGeometry geometry =
      prefill_contiguous_geometry(33, 2, 2, 1, 1, record_bytes);
  ASSERT_TRUE(geometry.valid) << geometry.reason_code;
  LlmMemoryWorkPlan plan = build_device_plan(
      geometry, "llm-metal-phase11-prefill-kv-segment-boundary", 4 * kGiB);
  const LlmMetalExecutionPlan* execution =
      get_llm_metal_execution_plan(plan);
  ASSERT_NE(execution, nullptr);
  ASSERT_TRUE(execution->valid) << execution->reason_code;
  EXPECT_EQ(execution->resources.k_segments.segment_lengths,
            (std::vector<size_t>{
                Constants::LLM_METAL_SEGMENT_CAPACITY_BYTES, 2}));
  EXPECT_EQ(execution->resources.v_segments.segment_lengths,
            execution->resources.k_segments.segment_lengths);
  resolve_and_prepare(plan);
  expect_complete_scenario_task(plan, LlmScenario::KvOnly, 1, 2);
}

TEST_F(LlmMetalBackendIntegrationTest,
       PrefillPagedRemaindersMultiThreadgroupPaddingAllScenariosIntegration) {
  backend_ = create_llm_metal_backend();
  ASSERT_NE(backend_, nullptr);
  LlmMemoryConfig config = metal_config();
  config.phase = LlmPhase::Prefill;
  config.kv_layout = LlmKvLayout::Paged;
  const LlmBackendLifecycleResult initialization =
      backend_->initialize(config);
  ASSERT_EQ(initialization.status, LlmBackendStatus::Ready)
      << initialization.reason_code << ": "
      << metal_evidence().capability.error.description;
  EXPECT_EQ(metal_evidence().capability.kernel_revision,
            kCanonicalKernelRevision);
  EXPECT_EQ(metal_evidence().capability.kernel_source_sha256,
            kCanonicalKernelSourceSha256);
  EXPECT_NE(std::find_if(
                metal_evidence().capability.foundation_pipelines.begin(),
                metal_evidence().capability.foundation_pipelines.end(),
                [](const LlmMetalPipelineEvidence& pipeline) {
                  return pipeline.label ==
                         "membenchmark.llm-metal.pipeline.prefill-paged-layout-probe";
                }),
            metal_evidence().capability.foundation_pipelines.end());

  constexpr size_t kPromptTokens = 257;
  constexpr size_t kQueryTileTokens = 64;
  constexpr size_t kBlockTokens = 128;
  constexpr size_t kRecordBytes = 33;
  const LlmGeometry geometry = prefill_paged_geometry(
      4097, kPromptTokens, kQueryTileTokens, kBlockTokens, 2, 2,
      kRecordBytes);
  ASSERT_TRUE(geometry.valid) << geometry.reason_code;
  ASSERT_TRUE(geometry.prefill.has_value());
  EXPECT_EQ(kPromptTokens % kQueryTileTokens, 1U);
  EXPECT_EQ(kPromptTokens % kBlockTokens, 1U);
  EXPECT_EQ(geometry.prefill->tile_count, 5U);
  EXPECT_EQ(geometry.kv_blocks_per_sequence, 3U);
  EXPECT_EQ(geometry.kv_block_bytes, kBlockTokens * kRecordBytes);
  EXPECT_EQ(geometry.last_block_valid_bytes, kRecordBytes);
  EXPECT_LT(geometry.last_block_valid_bytes, geometry.kv_block_bytes);

  LlmMemoryWorkPlan plan = build_device_plan(
      geometry, "llm-metal-phase12-prefill-paged-remainders");
  ASSERT_TRUE(plan.prefill_plan.has_value());
  ASSERT_TRUE(plan.prefill_plan->valid) << plan.prefill_plan->reason_code;
  EXPECT_EQ(plan.prefill_plan->prefix_block_visits_per_sequence, 9U);
  EXPECT_EQ(plan.prefill_plan->layout_metadata_lookups_per_layer_sequence,
            21U);
  resolve_and_prepare(plan);
  EXPECT_EQ(metal_evidence().capability.layout_probe_resource_count, 5U);

  expect_complete_scenario_task(plan, LlmScenario::WeightsOnly, 2);
  expect_complete_scenario_task(plan, LlmScenario::KvOnly, 2, 2);
  expect_complete_scenario_task(plan, LlmScenario::Mixed, 2, 2);
}

TEST_F(LlmMetalBackendIntegrationTest, PagedPrivateTableUploadValidationAndTier2SlotsIntegration) {
  backend_ = create_llm_metal_backend();
  ASSERT_NE(backend_, nullptr);
  LlmMemoryConfig config = metal_config();
  config.kv_layout = LlmKvLayout::Paged;
  const LlmBackendLifecycleResult paged_initialization =
      backend_->initialize(config);
  ASSERT_EQ(paged_initialization.status, LlmBackendStatus::Ready)
      << paged_initialization.reason_code << ": "
      << metal_evidence().capability.error.description;

  const LlmGeometry geometry = paged_geometry(4097, 17, 4, 2, 2, 8);
  ASSERT_TRUE(geometry.valid) << geometry.reason_code;
  const LlmKvLayoutPlan layout = paged_layout_for(geometry);
  ASSERT_TRUE(layout.valid) << layout.reason_code;
  LlmMemoryWorkPlan plan = build_device_plan(geometry, "llm-metal-phase10-paged-device-plan");
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
  EXPECT_TRUE(metal_evidence().timed_results_available);

  for (LlmScenario scenario : {LlmScenario::WeightsOnly,
                               LlmScenario::KvOnly,
                               LlmScenario::Mixed}) {
    const size_t minimum_threadgroups =
        scenario == LlmScenario::WeightsOnly ? 1U : 2U;
    expect_complete_scenario_task(plan, scenario, 2,
                                  minimum_threadgroups);
  }
}

TEST_F(LlmMetalBackendIntegrationTest, PagedKvPoolsCrossCanonicalSegmentBoundaryIntegration) {
  const size_t capacity = Constants::LLM_METAL_SEGMENT_CAPACITY_BYTES;
  const size_t record_bytes = capacity / 512;
  const size_t block_bytes = 2 * record_bytes;
  for (auto phase : {LlmPhase::Decode, LlmPhase::Prefill}) {
    SCOPED_TRACE(static_cast<int>(phase));
    backend_ = create_llm_metal_backend();
    ASSERT_NE(backend_, nullptr);
    LlmMemoryConfig config = metal_config();
    config.phase = phase;
    config.kv_layout = LlmKvLayout::Paged;
    const auto initialization = backend_->initialize(config);
    ASSERT_EQ(initialization.status, LlmBackendStatus::Ready) << initialization.reason_code;
    const LlmGeometry geometry = phase == LlmPhase::Decode
                                     ? paged_geometry(4097, 513, 2, 1, 1, record_bytes)
                                     : prefill_paged_geometry(33, 513, 513, 2, 1, 1, record_bytes);
    ASSERT_TRUE(geometry.valid) << geometry.reason_code;
    EXPECT_EQ(geometry.kv_blocks_per_sequence, 257U);
    EXPECT_EQ(geometry.kv_block_bytes, block_bytes);
    EXPECT_EQ(geometry.last_block_valid_bytes, record_bytes);
    LlmMemoryWorkPlan plan = build_device_plan(geometry, "llm-metal-paged-segment-boundary", 4 * kGiB);
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    const auto* execution = get_llm_metal_execution_plan(plan);
    ASSERT_NE(execution, nullptr);
    ASSERT_TRUE(execution->valid) << execution->reason_code;
    EXPECT_EQ(execution->resources.k_segments.elements_per_segment, 256U);
    EXPECT_EQ(execution->resources.k_segments.segment_lengths, (std::vector<size_t>{capacity, block_bytes}));
    EXPECT_EQ(execution->resources.v_segments.segment_lengths, execution->resources.k_segments.segment_lengths);
    ASSERT_TRUE(execution->resources.table_segments.has_value());
    EXPECT_EQ(execution->resources.table_segments->segment_count, 1U);
    resolve_and_prepare(plan);
    expect_complete_scenario_task(plan, LlmScenario::KvOnly, 1, 2);
    if (phase == LlmPhase::Decode) expect_complete_scenario_task(plan, LlmScenario::Mixed, 1, 2);
  }
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
  expect_complete_scenario_task(plan, LlmScenario::WeightsOnly, 1);
  expect_complete_scenario_task(plan, LlmScenario::Mixed, 1);
}

TEST(LlmMetalBackendFailureInjectionIntegrationTest, PagedPermutationAndPaddingHooksAreDetectedIntegration) {
  struct HookCase {
    bool wrong_permutation;
    bool padding_mismatch;
    std::string_view expected_reason;
  };
  const std::array<HookCase, 2> cases = {{
      {true, false, LlmBackendReason::TIMED_CHECKSUM_MISMATCH},
      {false, true, LlmBackendReason::PADDING_CANARY_MISMATCH},
  }};

  for (LlmPhase phase : {LlmPhase::Decode, LlmPhase::Prefill}) {
    SCOPED_TRACE(static_cast<int>(phase));
    for (size_t index = 0; index < cases.size(); ++index) {
      SCOPED_TRACE(index);
      LlmMetalBackendTestHooks hooks;
      hooks.force_wrong_paged_table_permutation = cases[index].wrong_permutation;
      hooks.force_padding_canary_mismatch = cases[index].padding_mismatch;
      std::unique_ptr<LlmBackend> backend = create_llm_metal_backend_for_testing(hooks);
      ASSERT_NE(backend, nullptr);
      LlmMemoryConfig config = metal_config();
      config.phase = phase;
      config.kv_layout = LlmKvLayout::Paged;
      const LlmBackendLifecycleResult initialization = backend->initialize(config);
      if (initialization.status == LlmBackendStatus::Unsupported) {
        ASSERT_TRUE(is_stable_capability_unsupported_reason(initialization.reason_code));
        GTEST_SKIP() << "LLM Metal paged unsupported: " << initialization.reason_code;
      }
      ASSERT_EQ(initialization.status, LlmBackendStatus::Ready) << initialization.reason_code;
      const LlmMetalBackendEvidence* initialized = get_llm_metal_backend_evidence(backend->evidence());
      ASSERT_NE(initialized, nullptr);

      const LlmGeometry geometry = phase == LlmPhase::Decode ? paged_geometry(4097, 17, 4, 2, 2, 8)
                                                             : prefill_paged_geometry(4097, 5, 2, 2, 2, 2, 33);
      ASSERT_TRUE(geometry.valid) << geometry.reason_code;
      ASSERT_LT(geometry.last_block_valid_bytes, geometry.kv_block_bytes);
      LlmMetalResourcePlanRequest request = resource_request(geometry);
      request.argument_buffer_encoded_length = initialized->capability.argument_buffer_encoded_length;
      request.argument_buffer_alignment = initialized->capability.argument_buffer_alignment;
      request.max_buffer_length = initialized->capability.max_buffer_length;
      request.available_memory_bytes = 2 * kGiB;
      request.host_mapping_granularity_bytes = get_system_page_size_bytes();
      LlmMemoryWorkPlan plan = make_metal_model_plan(geometry, build_llm_metal_execution_plan(request),
                                                     "llm-metal-phase10-paged-hook-" + std::to_string(index));
      ASSERT_TRUE(plan.valid) << plan.reason_code;
      ASSERT_EQ(backend->resolve_execution_plan(plan).status, LlmBackendStatus::Ready);
      const LlmBackendLifecycleResult prepared = backend->prepare_resources(plan);
      ASSERT_EQ(prepared.status, LlmBackendStatus::Ready) << prepared.reason_code;

      const LlmScenarioWorkPlan scenario = build_llm_scenario_work_plan(plan, LlmScenario::Mixed, 2, true);
      ASSERT_TRUE(scenario.valid) << scenario.reason_code;
      const LlmTaskExecutionResult result = backend->execute_task(
          plan, scenario, measurement_context(LlmScenario::Mixed, "phase-10-paged-failure-injection"));
      EXPECT_EQ(result.status, LlmTaskExecutionStatus::Invalid);
      EXPECT_EQ(result.reason_code, cases[index].expected_reason);
      EXPECT_EQ(result.completion.completed_layout_metadata_lookup_count, scenario.layout_metadata_lookup_count);
      EXPECT_EQ(result.completion.completed_layout_metadata_read_bytes, scenario.layout_metadata_read_bytes);
      const LlmMetalTaskEvidence* task = get_llm_metal_task_evidence(result);
      ASSERT_NE(task, nullptr);
      EXPECT_EQ(task->checksum_valid, !cases[index].wrong_permutation);
      EXPECT_TRUE(task->kv_write_validation_evaluated);
      EXPECT_TRUE(task->kv_write_validation_valid);
      EXPECT_TRUE(task->padding_canary_applicable);
      EXPECT_TRUE(task->padding_canary_evaluated);
      EXPECT_EQ(task->padding_canary_valid, !cases[index].padding_mismatch);
      EXPECT_EQ(task->post_validation_valid, !cases[index].padding_mismatch);
      EXPECT_TRUE(task->cold_checks[0].valid);
      EXPECT_TRUE(task->cold_checks[1].valid);
      EXPECT_TRUE(task->cold_checks[2].evaluated);
      EXPECT_EQ(task->cold_checks[2].valid, !cases[index].padding_mismatch);
    }
  }
}

TEST(LlmMetalBackendFailureInjectionIntegrationTest,
     TimedAndPostValidationHooksProduceStableTerminalEvidenceIntegration) {
  struct HookCase {
    size_t index;
    LlmTaskExecutionStatus expected_status;
    std::string_view expected_reason;
  };
  const std::array<HookCase, 5> cases = {{
      {0, LlmTaskExecutionStatus::Invalid,
       LlmBackendReason::INVALID_GPU_TIMESTAMPS},
      {1, LlmTaskExecutionStatus::Invalid,
       LlmBackendReason::TIMED_CHECKSUM_MISMATCH},
      {2, LlmTaskExecutionStatus::Failed,
       LlmBackendReason::POST_VALIDATION_COMMAND_FAILED},
      {3, LlmTaskExecutionStatus::Invalid,
       LlmBackendReason::KV_WRITE_VALIDATION_MISMATCH},
      {4, LlmTaskExecutionStatus::Failed,
       LlmBackendReason::TIMED_COMMAND_BUFFER_ERROR},
  }};

  for (const HookCase& test_case : cases) {
    SCOPED_TRACE(test_case.index);
    LlmMetalBackendTestHooks hooks;
    hooks.force_invalid_gpu_timestamps = test_case.index == 0;
    hooks.force_timed_checksum_mismatch = test_case.index == 1;
    hooks.force_post_validation_command_failure = test_case.index == 2;
    hooks.force_kv_write_validation_mismatch = test_case.index == 3;
    hooks.force_timed_command_failure = test_case.index == 4;
    std::unique_ptr<LlmBackend> backend =
        create_llm_metal_backend_for_testing(hooks);
    ASSERT_NE(backend, nullptr);
    const LlmBackendLifecycleResult initialization =
        backend->initialize(metal_config());
    if (initialization.status == LlmBackendStatus::Unsupported) {
      ASSERT_TRUE(
          is_stable_capability_unsupported_reason(initialization.reason_code));
      GTEST_SKIP() << "LLM Metal decode-contiguous unsupported: "
                   << initialization.reason_code;
    }
    ASSERT_EQ(initialization.status, LlmBackendStatus::Ready)
        << initialization.reason_code;
    const LlmMetalBackendEvidence* initialized =
        get_llm_metal_backend_evidence(backend->evidence());
    ASSERT_NE(initialized, nullptr);

    const LlmGeometry geometry = contiguous_geometry(4097, 17, 2, 2, 8);
    LlmMetalResourcePlanRequest request = resource_request(geometry);
    request.argument_buffer_encoded_length =
        initialized->capability.argument_buffer_encoded_length;
    request.argument_buffer_alignment =
        initialized->capability.argument_buffer_alignment;
    request.max_buffer_length = initialized->capability.max_buffer_length;
    request.available_memory_bytes = 2 * kGiB;
    request.host_mapping_granularity_bytes = get_system_page_size_bytes();
    LlmMemoryWorkPlan plan = make_metal_model_plan(
        geometry, build_llm_metal_execution_plan(request),
        "llm-metal-phase9-task-hook-" + std::to_string(test_case.index));
    ASSERT_TRUE(plan.valid) << plan.reason_code;
    ASSERT_EQ(backend->resolve_execution_plan(plan).status,
              LlmBackendStatus::Ready);
    const LlmBackendLifecycleResult prepared =
        backend->prepare_resources(plan);
    ASSERT_EQ(prepared.status, LlmBackendStatus::Ready)
        << prepared.reason_code;
    const LlmScenarioWorkPlan scenario = build_llm_scenario_work_plan(
        plan, LlmScenario::Mixed, 2, true);
    ASSERT_TRUE(scenario.valid) << scenario.reason_code;
    const LlmTaskExecutionResult result = backend->execute_task(
        plan, scenario,
        measurement_context(LlmScenario::Mixed,
                            "phase-9-failure-injection"));
    EXPECT_EQ(result.status, test_case.expected_status);
    EXPECT_EQ(result.reason_code, test_case.expected_reason);
    EXPECT_EQ(result.completion.planned_work_units, 2U);
    EXPECT_EQ(result.completion.completed_work_units,
              test_case.index == 4 ? 0U : 2U);
    const LlmMetalTaskEvidence* task = get_llm_metal_task_evidence(result);
    ASSERT_NE(task, nullptr);
    EXPECT_TRUE(task->timed_pipeline_available);
    EXPECT_TRUE(task->cold_checks[0].applicable);
    EXPECT_TRUE(task->cold_checks[1].applicable);
    EXPECT_EQ(task->cold_checks[0].evaluated, test_case.index != 2 && test_case.index != 4);
    // Host acceptance injection is not an observed shader-bit mismatch.
    EXPECT_EQ(task->cold_checks[1].valid, test_case.index != 2 && test_case.index != 4);
    EXPECT_EQ(task->timing_evaluated, test_case.index != 4);
    EXPECT_EQ(task->timing_valid,
              test_case.index != 0 && test_case.index != 4);
    EXPECT_EQ(task->checksum_evaluated, test_case.index != 4);
    EXPECT_EQ(task->checksum_valid,
              test_case.index != 1 && test_case.index != 4);
    EXPECT_EQ(task->reset_command_buffer_count, 1U);
    EXPECT_EQ(task->timed_command_buffer_count, 1U);
    EXPECT_EQ(task->post_validation_command_buffer_count,
              test_case.index == 4 ? 0U : 1U);
    EXPECT_EQ(task->reset_command_status, "completed");
    EXPECT_EQ(task->timed_command_status,
              test_case.index == 4 ? "error" : "completed");
    EXPECT_EQ(task->post_validation_command_status,
              test_case.index == 4
                  ? "not-run"
                  : (test_case.index == 2 ? "error" : "completed"));
    EXPECT_EQ(task->post_validation_evaluated,
              test_case.index != 2 && test_case.index != 4);
    EXPECT_EQ(task->kv_write_validation_evaluated,
              test_case.index != 2 && test_case.index != 4);
    EXPECT_EQ(task->kv_write_validation_valid,
              test_case.index != 2 && test_case.index != 3 &&
                  test_case.index != 4);
    EXPECT_EQ(task->post_validation_valid,
              test_case.index != 2 && test_case.index != 3 &&
                  test_case.index != 4);
    EXPECT_EQ(result.validation.evaluated,
              test_case.index != 2 && test_case.index != 4);
    EXPECT_EQ(result.validation.valid,
              test_case.index != 2 && test_case.index != 3 &&
                  test_case.index != 4);
    EXPECT_EQ(backend->release_resources().status, LlmBackendStatus::Ready);
  }
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

TEST(LlmMetalBackendTest, ColdReadbackIndependentVerdicts) {
  for (auto phase : {LlmPhase::Decode, LlmPhase::Prefill}) {
    for (bool writes : {false, true}) {
      for (bool padding : {false, true}) {
        for (bool completed : {false, true}) {
          for (unsigned combination = 0; combination < 8; ++combination) {
            SCOPED_TRACE(::testing::Message() << static_cast<int>(phase) << ":" << writes << ":" << padding << ":"
                                              << completed << ":" << combination);
            const uint32_t flags = ((combination & 1) ? LlmMetalKernelContract::kValidationInvalidParametersBit : 0) |
                                   ((combination & 2) ? LlmMetalKernelContract::kKvWriteValidationMismatchBit : 0) |
                                   ((combination & 4) ? LlmMetalKernelContract::kPaddingCanaryMismatchBit : 0);
            const auto checks = interpret_llm_metal_cold_checks(phase, writes, padding, completed, flags);
            EXPECT_EQ(checks[0].applicable, writes);
            EXPECT_EQ(checks[1].applicable, writes);
            EXPECT_EQ(checks[2].applicable, writes && padding);
            EXPECT_EQ(checks[1].kind, phase == LlmPhase::Decode ? LlmColdCheckKind::KvAppendFinal
                                                                : LlmColdCheckKind::KvPrefillFinalSamples);
            for (size_t slot = 0; slot < checks.size(); ++slot) {
              const bool mismatch = (combination & (1U << slot)) != 0;
              const bool evaluated =
                  checks[slot].applicable && completed && (slot == 0 || mismatch || !(combination & 1));
              EXPECT_EQ(checks[slot].evaluated, evaluated);
              EXPECT_EQ(checks[slot].valid, evaluated && !mismatch && !(combination & 1));
              if (!completed) EXPECT_EQ(checks[slot].reason_code, "not-evaluated");
            }
          }
        }
      }
    }
  }
}

TEST_F(LlmMetalBackendIntegrationTest, PrefillElementWidthsAndWideSingleTokenPreserveTaskContractIntegration) {
  struct Case {
    size_t prompt_tokens;
    size_t query_tile_tokens;
    size_t head_dimension;
    size_t block_tokens;
  };
  // Include a wide P=Q=1 record: replacing token control with vector control
  // must preserve this supported case as well as narrow, split-token tails.
  constexpr std::array<Case, 4> kCases = {{{1, 1, 131072, 1},
                                          {17, 1, 3, 4},
                                          {257, 257, 64, 128},
                                          {129, 16, 5, 16}}};
  for (LlmKvLayout layout : {LlmKvLayout::Contiguous, LlmKvLayout::Paged}) {
    for (size_t element_bytes : {1U, 2U, 4U}) {
      for (const Case& test_case : kCases) {
        SCOPED_TRACE("P=" + std::to_string(test_case.prompt_tokens) +
                     ", E=" + std::to_string(element_bytes) +
                     ", layout=" + std::to_string(static_cast<int>(layout)));
        LlmMemoryConfig config = metal_config();
        config.phase = LlmPhase::Prefill;
        config.kv_layout = layout;
        backend_ = create_llm_metal_backend();
        ASSERT_EQ(backend_->initialize(config).status, LlmBackendStatus::Ready);
        LlmGeometryRequest request;
        request.active_weight_bytes = 4097;
        request.layer_count = 2;
        request.batch_size = 2;
        request.query_head_count = 1;
        request.kv_head_count = 1;
        request.head_dimension = test_case.head_dimension;
        request.kv_element_bytes = element_bytes;
        request.phase = LlmPhase::Prefill;
        request.kv_layout = layout;
        request.prompt_tokens = test_case.prompt_tokens;
        request.attention_query_tile_tokens = test_case.query_tile_tokens;
        request.kv_block_tokens = layout == LlmKvLayout::Paged ? test_case.block_tokens : 0;
        const LlmGeometry geometry = resolve_llm_geometry(request);
        ASSERT_TRUE(geometry.valid) << geometry.reason_code;
        ASSERT_EQ(geometry.k_or_v_record_bytes_per_layer, test_case.head_dimension * element_bytes);
        LlmMemoryWorkPlan plan = build_device_plan(geometry, "prefill-element-width-and-record-boundary");
        resolve_and_prepare(plan);
        for (LlmScenario scenario : {LlmScenario::WeightsOnly, LlmScenario::KvOnly, LlmScenario::Mixed}) {
          expect_complete_scenario_task(plan, scenario, 1);
          expect_complete_scenario_task(plan, scenario, 3);
        }
      }
    }
  }
}

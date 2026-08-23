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
 * @file llm_metal_backend.mm
 * @brief Metal capability, resource, decode, and prefill execution backend
 *
 * Metal and Objective-C ownership remain in this file. The backend is
 * synchronous and command-owned. It exposes experimental decode and contiguous
 * prefill paths with GPU timestamps, timed checksums, and excluded KV-write
 * validation.
 * Every Objective-C entry boundary catches Objective-C and C++ failures, and
 * candidate resources are published only after both admissions, initialization,
 * upload, ABI probing, and excluded validation succeed.
 */

#include "llm_memory/llm_metal_backend.h"

#include "core/config/constants.h"
#include "core/signal/signal_handler.h"
#include "core/system/page_size.h"
#include "llm_memory/llm_metal_kernels_source.h"
#include "utils/hash_utils.h"
#include "utils/numeric_utils.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <new>
#include <string>
#include <string_view>
#include <utility>

namespace {

constexpr size_t kFoundationPipelineCount = 6;
constexpr size_t kWorkloadPipelineCount = 3;
static_assert(kLlmMetalLayoutProbeWordCount == LlmMetalKernelContract::kProbeOutputWordCount);
static_assert(kLlmMetalDecodeLayoutProbeWordCount == LlmMetalKernelContract::kDecodeProbeOutputWordCount);
static_assert(kLlmMetalDecodePagedLayoutProbeWordCount ==
              LlmMetalKernelContract::kDecodePagedProbeOutputWordCount);
static_assert(kLlmMetalPrefillLayoutProbeWordCount ==
              LlmMetalKernelContract::kPrefillProbeOutputWordCount);
static_assert(sizeof(LlmMetalDecodeContiguousParams) == LlmMetalKernelContract::kDecodeParameterAbiSize);
static_assert(alignof(LlmMetalDecodeContiguousParams) == LlmMetalKernelContract::kDecodeParameterAbiAlignment);
static_assert(sizeof(LlmMetalDecodePagedParams) == LlmMetalKernelContract::kDecodePagedParameterAbiSize);
static_assert(alignof(LlmMetalDecodePagedParams) == LlmMetalKernelContract::kDecodePagedParameterAbiAlignment);
static_assert(sizeof(LlmMetalPrefillContiguousParams) ==
              LlmMetalKernelContract::kPrefillParameterAbiSize);
static_assert(alignof(LlmMetalPrefillContiguousParams) ==
              LlmMetalKernelContract::kPrefillParameterAbiAlignment);

bool is_power_of_two(size_t value) noexcept { return value != 0 && (value & (value - 1)) == 0; }

bool checked_ceil_divide(size_t value, size_t divisor, size_t& result) noexcept {
  if (divisor == 0) {
    return false;
  }
  result = value / divisor + (value % divisor != 0 ? 1U : 0U);
  return true;
}

bool checked_subtract(size_t lhs, size_t rhs, size_t& result) noexcept {
  if (rhs > lhs) {
    return false;
  }
  result = lhs - rhs;
  return true;
}

/** Count aligned 16-byte vectors intersected by one byte range. */
bool range_vector_count(size_t range_start, size_t range_length,
                        size_t& vector_count) noexcept {
  const size_t vector_offset =
      range_start % Constants::LLM_METAL_VECTOR_WIDTH_BYTES;
  size_t offset_and_length = 0;
  return NumericUtils::checked_add(vector_offset, range_length,
                                   offset_and_length) &&
         checked_ceil_divide(offset_and_length,
                             Constants::LLM_METAL_VECTOR_WIDTH_BYTES,
                             vector_count);
}

/**
 * Update an exact maximum across contiguous equal-length byte ranges.
 *
 * Starts repeat modulo the 16-byte vector width within at most 16 ranges,
 * so inspecting one complete residue cycle is sufficient for any count.
 */
bool update_maximum_equal_range_vector_count(
    size_t first_range_start, size_t range_length, size_t range_count,
    size_t& maximum_vector_count) noexcept {
  const size_t vector_width = Constants::LLM_METAL_VECTOR_WIDTH_BYTES;
  size_t range_start = first_range_start % vector_width;
  const size_t start_step = range_length % vector_width;
  const size_t offsets_to_inspect = std::min(range_count, vector_width);
  for (size_t index = 0; index < offsets_to_inspect; ++index) {
    size_t vector_count = 0;
    if (!range_vector_count(range_start, range_length, vector_count)) {
      return false;
    }
    maximum_vector_count = std::max(maximum_vector_count, vector_count);
    range_start = (range_start + start_step) % vector_width;
  }
  return true;
}

bool add_auxiliary_allocation(size_t count, size_t element_bytes, size_t& total) noexcept {
  size_t allocation_bytes = 0;
  return NumericUtils::checked_multiply(count, element_bytes, allocation_bytes) &&
         NumericUtils::checked_add(total, allocation_bytes, total);
}

bool add_external_string_backing(const std::string& value, size_t& total) noexcept {
  const std::uintptr_t data = reinterpret_cast<std::uintptr_t>(value.data());
  const std::uintptr_t object = reinterpret_cast<std::uintptr_t>(&value);
  const bool inline_storage = data >= object && data < object + sizeof(value);
  if (inline_storage || value.capacity() == 0) {
    return true;
  }
  size_t allocation_bytes = 0;
  return NumericUtils::checked_add(value.capacity(), size_t{1}, allocation_bytes) &&
         NumericUtils::checked_add(total, allocation_bytes, total);
}

void append_identity_field(std::string& output, std::string_view key, std::string_view value) {
  output.append(key.data(), key.size());
  output.push_back('=');
  output += std::to_string(value.size());
  output.push_back(':');
  output.append(value.data(), value.size());
  output.push_back(';');
}

void append_identity_field(std::string& output, std::string_view key, size_t value) {
  append_identity_field(output, key, std::to_string(value));
}

std::string ns_string(NSString* value) {
  if (value == nil) {
    return {};
  }
  const char* const utf8 = value.UTF8String;
  return utf8 == nullptr ? std::string{} : std::string(utf8);
}

std::string bounded_diagnostic(std::string value) {
  if (value.size() <= Constants::LLM_METAL_DIAGNOSTIC_MAX_BYTES) {
    return value;
  }
  size_t prefix_bytes = Constants::LLM_METAL_DIAGNOSTIC_MAX_BYTES;
  while (prefix_bytes > 0 && prefix_bytes < value.size() &&
         (static_cast<unsigned char>(value[prefix_bytes]) & 0xc0U) == 0x80U) {
    --prefix_bytes;
  }
  value.resize(prefix_bytes);
  return value;
}

LlmMetalErrorDiagnostic error_diagnostic(NSError* error) {
  if (error == nil) {
    return {};
  }
  return {bounded_diagnostic(ns_string(error.domain)), static_cast<long long>(error.code),
          bounded_diagnostic(ns_string(error.localizedDescription))};
}

LlmMetalErrorDiagnostic internal_error(std::string description) {
  return {"macos-memory-benchmark.llm-metal", 0, bounded_diagnostic(std::move(description))};
}

std::string version_macro_string(long value) {
  const long major = value / 10000;
  const long minor = (value / 100) % 100;
  const long patch = value % 100;
  std::string result = std::to_string(major) + "." + std::to_string(minor);
  if (patch != 0) {
    result += "." + std::to_string(patch);
  }
  return result;
}

const char* storage_mode_string(MTLStorageMode mode) noexcept {
  switch (mode) {
    case MTLStorageModeShared:
      return "shared";
    case MTLStorageModeManaged:
      return "managed";
    case MTLStorageModePrivate:
      return "private";
    case MTLStorageModeMemoryless:
      return "memoryless";
  }
  return "unknown";
}

const char* cpu_cache_mode_string(MTLCPUCacheMode mode) noexcept {
  switch (mode) {
    case MTLCPUCacheModeDefaultCache:
      return "default-cache";
    case MTLCPUCacheModeWriteCombined:
      return "write-combined";
  }
  return "unknown";
}

const char* hazard_mode_string(MTLHazardTrackingMode mode) noexcept {
  switch (mode) {
    case MTLHazardTrackingModeDefault:
      return "default";
    case MTLHazardTrackingModeUntracked:
      return "untracked";
    case MTLHazardTrackingModeTracked:
      return "tracked";
  }
  return "unknown";
}

bool is_private_pool(LlmMetalResourcePool pool) noexcept {
  return pool == LlmMetalResourcePool::Weight || pool == LlmMetalResourcePool::K || pool == LlmMetalResourcePool::V ||
         pool == LlmMetalResourcePool::BlockTable;
}

size_t argument_resource_id(LlmMetalResourcePool pool, size_t pool_index) noexcept {
  switch (pool) {
    case LlmMetalResourcePool::Weight:
      return LlmMetalKernelContract::kWeightSegmentBaseId + pool_index;
    case LlmMetalResourcePool::K:
      return LlmMetalKernelContract::kKeySegmentBaseId + pool_index;
    case LlmMetalResourcePool::V:
      return LlmMetalKernelContract::kValueSegmentBaseId + pool_index;
    case LlmMetalResourcePool::BlockTable:
      return LlmMetalKernelContract::kTableSegmentBaseId + pool_index;
    case LlmMetalResourcePool::Status:
      return LlmMetalKernelContract::kStatusChecksumResourceId;
    case LlmMetalResourcePool::ArgumentBuffer:
    case LlmMetalResourcePool::Staging:
      break;
  }
  return std::numeric_limits<size_t>::max();
}

bool same_planned_resource(const LlmMetalPlannedResource& planned, const LlmMetalAllocatedResource& actual) noexcept {
  return planned.pool == actual.pool && planned.pool_index == actual.pool_index &&
         planned.length_bytes == actual.length_bytes;
}

bool same_planning_limits(const LlmMetalPlanningLimits& left, const LlmMetalPlanningLimits& right) noexcept {
  return left.segment_capacity_bytes == right.segment_capacity_bytes &&
         left.segment_slots_per_pool == right.segment_slots_per_pool &&
         left.threads_per_threadgroup_cap == right.threads_per_threadgroup_cap &&
         left.maximum_threadgroups_per_grid == right.maximum_threadgroups_per_grid &&
         left.maximum_owner_ordinals_per_threadgroup == right.maximum_owner_ordinals_per_threadgroup &&
         left.maximum_vector_iterations_per_lane_per_visit == right.maximum_vector_iterations_per_lane_per_visit &&
         left.maximum_serial_range_visits_per_lane_per_task ==
             right.maximum_serial_range_visits_per_lane_per_task &&
         left.maximum_paged_semantic_lookups_per_task == right.maximum_paged_semantic_lookups_per_task &&
         left.maximum_work_units_per_dispatch == right.maximum_work_units_per_dispatch;
}

bool same_segment_plan(const LlmKvSegmentPlan& left, const LlmKvSegmentPlan& right) noexcept {
  return left.valid == right.valid && left.reason_code == right.reason_code &&
         left.element_count == right.element_count && left.element_bytes == right.element_bytes &&
         left.segment_capacity_bytes == right.segment_capacity_bytes &&
         left.segment_slot_cap == right.segment_slot_cap && left.elements_per_segment == right.elements_per_segment &&
         left.segment_count == right.segment_count &&
         left.maximum_addressable_elements == right.maximum_addressable_elements &&
         left.maximum_addressable_bytes == right.maximum_addressable_bytes &&
         left.unused_nominal_segment_capacity_bytes == right.unused_nominal_segment_capacity_bytes &&
         left.total_length_bytes == right.total_length_bytes && left.segment_lengths == right.segment_lengths &&
         left.identity == right.identity;
}

bool same_layout_memory(const LlmKvLayoutMemoryBudget& left, const LlmKvLayoutMemoryBudget& right) noexcept {
  return left.k_logical_bytes == right.k_logical_bytes && left.v_logical_bytes == right.v_logical_bytes &&
         left.k_physical_bytes == right.k_physical_bytes && left.v_physical_bytes == right.v_physical_bytes &&
         left.k_layout_padding_bytes == right.k_layout_padding_bytes &&
         left.v_layout_padding_bytes == right.v_layout_padding_bytes &&
         left.block_table_bytes == right.block_table_bytes &&
         left.validation_bitset_bytes == right.validation_bitset_bytes &&
         left.transient_peak_bytes == right.transient_peak_bytes &&
         left.resident_layout_bytes == right.resident_layout_bytes &&
         left.known_owned_peak_bytes == right.known_owned_peak_bytes;
}

bool same_layout_plan(const LlmKvLayoutPlan& left, const LlmKvLayoutPlan& right) noexcept {
  return left.valid == right.valid && left.reason_code == right.reason_code &&
         left.sequence_tokens == right.sequence_tokens && left.kv_block_tokens == right.kv_block_tokens &&
         left.layer_count == right.layer_count && left.batch_size == right.batch_size &&
         left.k_or_v_record_bytes_per_layer == right.k_or_v_record_bytes_per_layer &&
         left.blocks_per_sequence == right.blocks_per_sequence &&
         left.physical_blocks_per_layer == right.physical_blocks_per_layer &&
         left.total_physical_blocks == right.total_physical_blocks && left.block_bytes == right.block_bytes &&
         left.last_block_tokens == right.last_block_tokens &&
         left.last_block_valid_bytes == right.last_block_valid_bytes &&
         left.decode_append_offset_in_last_block == right.decode_append_offset_in_last_block &&
         left.block_table_entries == right.block_table_entries &&
         left.permutation_iterations == right.permutation_iterations &&
         left.validation_entries == right.validation_entries && left.hash_entries == right.hash_entries &&
         left.upload_bytes == right.upload_bytes && same_layout_memory(left.memory, right.memory) &&
         left.geometry_identity == right.geometry_identity;
}

bool same_optional_segment_plan(const std::optional<LlmKvSegmentPlan>& left,
                                const std::optional<LlmKvSegmentPlan>& right) noexcept {
  return left.has_value() == right.has_value() && (!left.has_value() || same_segment_plan(*left, *right));
}

bool same_optional_layout_plan(const std::optional<LlmKvLayoutPlan>& left,
                               const std::optional<LlmKvLayoutPlan>& right) noexcept {
  return left.has_value() == right.has_value() && (!left.has_value() || same_layout_plan(*left, *right));
}

bool same_argument_buffer_plan(const LlmMetalArgumentBufferPlan& left, const LlmMetalArgumentBufferPlan& right) {
  return left.valid == right.valid && left.reason_code == right.reason_code &&
         left.weight_slot_base == right.weight_slot_base && left.k_slot_base == right.k_slot_base &&
         left.v_slot_base == right.v_slot_base && left.table_slot_base == right.table_slot_base &&
         left.status_slot == right.status_slot &&
         left.encoded_resource_slot_count == right.encoded_resource_slot_count &&
         left.weight_segment_count == right.weight_segment_count && left.k_segment_count == right.k_segment_count &&
         left.v_segment_count == right.v_segment_count && left.table_segment_count == right.table_segment_count &&
         left.active_resource_count == right.active_resource_count && left.identity == right.identity;
}

bool same_planned_resources(const std::vector<LlmMetalPlannedResource>& left,
                            const std::vector<LlmMetalPlannedResource>& right) noexcept {
  if (left.size() != right.size()) {
    return false;
  }
  for (size_t index = 0; index < left.size(); ++index) {
    if (left[index].pool != right[index].pool || left[index].pool_index != right[index].pool_index ||
        left[index].length_bytes != right[index].length_bytes || left[index].persistent != right[index].persistent) {
      return false;
    }
  }
  return true;
}

bool same_resource_plan(const LlmMetalResourcePlan& left, const LlmMetalResourcePlan& right) noexcept {
  return left.valid == right.valid && left.reason_code == right.reason_code &&
         same_planning_limits(left.limits, right.limits) &&
         same_segment_plan(left.weight_segments, right.weight_segments) &&
         same_segment_plan(left.k_segments, right.k_segments) && same_segment_plan(left.v_segments, right.v_segments) &&
         same_optional_segment_plan(left.table_segments, right.table_segments) &&
         same_optional_layout_plan(left.paged_layout, right.paged_layout) &&
         same_argument_buffer_plan(left.argument_buffer, right.argument_buffer) &&
         left.argument_buffer_encoded_length == right.argument_buffer_encoded_length &&
         left.argument_buffer_alignment == right.argument_buffer_alignment &&
         left.max_buffer_length == right.max_buffer_length &&
         left.host_mapping_granularity_bytes == right.host_mapping_granularity_bytes &&
         left.status_buffer_length == right.status_buffer_length &&
         left.staging_buffer_length == right.staging_buffer_length &&
         left.host_permutation_mapping_bytes == right.host_permutation_mapping_bytes &&
         left.permutation_validation_bitset_bytes == right.permutation_validation_bitset_bytes &&
         left.additional_owned_bytes == right.additional_owned_bytes &&
         left.persistent_resource_length_bytes == right.persistent_resource_length_bytes &&
         left.transient_peak_bytes == right.transient_peak_bytes &&
         left.known_owned_peak_bytes == right.known_owned_peak_bytes &&
         left.available_memory_bytes == right.available_memory_bytes &&
         left.admitted_budget_bytes == right.admitted_budget_bytes &&
         same_planned_resources(left.planned_resources, right.planned_resources) && left.identity == right.identity;
}

bool calculate_admitted_budget(size_t available_memory_bytes, size_t& admitted_budget_bytes) noexcept {
  if (available_memory_bytes == 0) {
    return NumericUtils::checked_multiply(static_cast<size_t>(Constants::FALLBACK_TOTAL_LIMIT_MB),
                                          Constants::BYTES_PER_MB, admitted_budget_bytes);
  }
  const long double scaled = static_cast<long double>(available_memory_bytes) * Constants::MEMORY_LIMIT_FACTOR;
  if (scaled > static_cast<long double>(std::numeric_limits<size_t>::max())) {
    return false;
  }
  admitted_budget_bytes = static_cast<size_t>(scaled);
  return true;
}

bool append_planned_segments(LlmMetalResourcePool pool, const LlmKvSegmentPlan& segments,
                             std::vector<LlmMetalPlannedResource>& resources) {
  for (size_t index = 0; index < segments.segment_lengths.size(); ++index) {
    resources.push_back({pool, index, segments.segment_lengths[index], true});
  }
  return true;
}

bool segment_plan_matches_length(const LlmKvSegmentPlan& segments, size_t expected_length,
                                 size_t max_buffer_length) noexcept {
  if (!segments.valid || segments.total_length_bytes != expected_length ||
      segments.segment_count != segments.segment_lengths.size()) {
    return false;
  }
  size_t sum = 0;
  for (size_t length : segments.segment_lengths) {
    if (length == 0 || length > max_buffer_length || !NumericUtils::checked_add(sum, length, sum)) {
      return false;
    }
  }
  return sum == expected_length;
}

uint8_t contiguous_pattern_byte(uint64_t seed, uint64_t absolute_byte) noexcept {
  const uint64_t word_index = absolute_byte / sizeof(uint32_t);
  const unsigned byte_index = static_cast<unsigned>(absolute_byte % sizeof(uint32_t));
  const uint32_t word = llm_metal_contiguous_pattern_word(seed, word_index);
  return static_cast<uint8_t>(word >> (byte_index * 8U));
}

uint8_t paged_pattern_byte(uint64_t seed, uint64_t block_bytes, uint32_t physical_blocks_per_layer,
                           uint64_t absolute_byte) noexcept {
  if (block_bytes == 0 || physical_blocks_per_layer == 0) {
    return 0;
  }
  const uint64_t global_block = absolute_byte / block_bytes;
  const uint64_t block_byte = absolute_byte % block_bytes;
  const uint64_t layer = global_block / static_cast<uint64_t>(physical_blocks_per_layer);
  const uint64_t physical_block = global_block % static_cast<uint64_t>(physical_blocks_per_layer);
  const uint64_t word_index = block_byte / sizeof(uint32_t);
  const unsigned byte_index = static_cast<unsigned>(block_byte % sizeof(uint32_t));
  const uint32_t word = llm_metal_paged_pattern_word(seed, layer, physical_block, word_index);
  return static_cast<uint8_t>(word >> (byte_index * 8U));
}

}  // namespace

uint32_t llm_metal_contiguous_pattern_word(uint64_t seed, uint64_t absolute_word_index) noexcept {
  return static_cast<uint32_t>(seed) +
         LlmMetalKernelContract::kContiguousPatternWordMultiplier *
             static_cast<uint32_t>(absolute_word_index + 1U);
}

uint32_t llm_metal_paged_pattern_word(uint64_t seed, uint64_t layer_index,
                                      uint64_t physical_block,
                                      uint64_t block_word_index) noexcept {
  return static_cast<uint32_t>(seed) +
         LlmMetalKernelContract::kPagedPatternLayerMultiplier *
             static_cast<uint32_t>(layer_index + 1U) +
         LlmMetalKernelContract::kPagedPatternPhysicalMultiplier *
             static_cast<uint32_t>(physical_block + 1U) +
         LlmMetalKernelContract::kPagedPatternWordMultiplier *
             static_cast<uint32_t>(block_word_index + 1U);
}

uint32_t llm_metal_decode_append_word(uint64_t scenario_seed, uint64_t work_unit, uint64_t layer_index,
                                      uint64_t batch_index, uint64_t absolute_word_index,
                                      LlmMetalResourcePool pool) noexcept {
  const uint32_t pool_domain = pool == LlmMetalResourcePool::K
                                   ? LlmMetalKernelContract::kAppendKeyDomain
                                   : LlmMetalKernelContract::kAppendValueDomain;
  return static_cast<uint32_t>(scenario_seed) +
         LlmMetalKernelContract::kAppendWorkUnitMultiplier * static_cast<uint32_t>(work_unit + 1U) +
         LlmMetalKernelContract::kAppendLayerMultiplier * static_cast<uint32_t>(layer_index + 1U) +
         LlmMetalKernelContract::kAppendBatchMultiplier * static_cast<uint32_t>(batch_index + 1U) +
         LlmMetalKernelContract::kAppendWordMultiplier * static_cast<uint32_t>(absolute_word_index + 1U) +
         pool_domain;
}

uint32_t llm_metal_prefill_write_word(
    uint64_t scenario_seed, uint64_t work_unit, uint64_t layer_index,
    uint64_t batch_index, uint64_t absolute_word_index,
    LlmMetalResourcePool pool) noexcept {
  const uint32_t pool_domain =
      pool == LlmMetalResourcePool::K
          ? LlmMetalKernelContract::kPrefillWriteKeyDomain
          : LlmMetalKernelContract::kPrefillWriteValueDomain;
  return static_cast<uint32_t>(scenario_seed) +
         LlmMetalKernelContract::kAppendWorkUnitMultiplier *
             static_cast<uint32_t>(work_unit + 1U) +
         LlmMetalKernelContract::kAppendLayerMultiplier *
             static_cast<uint32_t>(layer_index + 1U) +
         LlmMetalKernelContract::kAppendBatchMultiplier *
             static_cast<uint32_t>(batch_index + 1U) +
         LlmMetalKernelContract::kAppendWordMultiplier *
             static_cast<uint32_t>(absolute_word_index + 1U) +
         pool_domain;
}

bool equal_llm_metal_checksum(const LlmMetalDualMod32Checksum& left,
                              const LlmMetalDualMod32Checksum& right) noexcept {
  return left.weight.a == right.weight.a && left.weight.b == right.weight.b && left.k.a == right.k.a &&
         left.k.b == right.k.b && left.v.a == right.v.a && left.v.b == right.v.b;
}

namespace {

struct Mod32AffinePattern {
  uint32_t base = 0;
  uint32_t step = 0;
};

uint32_t triangular_mod32(uint64_t count) noexcept {
  const unsigned __int128 wide = static_cast<unsigned __int128>(count) * (count == 0 ? 0 : count - 1U) / 2U;
  return static_cast<uint32_t>(wide);
}

uint32_t affine_word(const Mod32AffinePattern& pattern, uint64_t word_index) noexcept {
  return pattern.base + pattern.step * static_cast<uint32_t>(word_index + 1U);
}

uint32_t affine_sum(const Mod32AffinePattern& pattern, uint64_t first_word, uint64_t count) noexcept {
  const uint32_t count32 = static_cast<uint32_t>(count);
  const uint32_t index_sum = count32 * static_cast<uint32_t>(first_word + 1U) + triangular_mod32(count);
  return count32 * pattern.base + pattern.step * index_sum;
}

uint32_t word_index_sum(uint64_t first_word, uint64_t count) noexcept {
  return static_cast<uint32_t>(count) * static_cast<uint32_t>(first_word) + triangular_mod32(count);
}

uint32_t byte_mask_for(uint32_t valid_mask) noexcept {
  uint32_t byte_mask = 0;
  for (uint32_t byte = 0; byte < sizeof(uint32_t); ++byte) {
    if ((valid_mask & (1U << byte)) != 0) {
      byte_mask |= UINT32_C(0xff) << (byte * 8U);
    }
  }
  return byte_mask;
}

uint32_t checksum_domain(uint32_t pool_domain, uint32_t visit_domain, uint64_t scenario_seed,
                         uint64_t work_unit, uint64_t layer, uint64_t batch,
                         uint64_t tile_ordinal, uint32_t valid_mask,
                         uint32_t profile_domain) noexcept {
  return profile_domain +
         static_cast<uint32_t>(scenario_seed) +
         LlmMetalKernelContract::kChecksumScenarioHighMultiplier * static_cast<uint32_t>(scenario_seed >> 32U) +
         pool_domain + visit_domain +
         LlmMetalKernelContract::kChecksumWorkUnitMultiplier * static_cast<uint32_t>(work_unit + 1U) +
         LlmMetalKernelContract::kChecksumLayerMultiplier * static_cast<uint32_t>(layer + 1U) +
         LlmMetalKernelContract::kChecksumBatchMultiplier * static_cast<uint32_t>(batch + 1U) +
         LlmMetalKernelContract::kChecksumTileMultiplier *
             static_cast<uint32_t>(tile_ordinal) +
         LlmMetalKernelContract::kChecksumValidMaskMultiplier * valid_mask;
}

void mix_checksum_word(LlmMetalMod32Lane& checksum, uint32_t value, uint64_t word_index, uint32_t domain) noexcept {
  checksum.a += value + domain;
  checksum.b += value * LlmMetalKernelContract::kChecksumValueMultiplier +
                static_cast<uint32_t>(word_index) * LlmMetalKernelContract::kChecksumAddressMultiplier +
                domain * UINT32_C(0x7feb352d);
}

void mix_full_words(LlmMetalMod32Lane& checksum, const Mod32AffinePattern& pattern, uint64_t first_word,
                    uint64_t count, uint32_t domain) noexcept {
  const uint32_t count32 = static_cast<uint32_t>(count);
  const uint32_t value_sum = affine_sum(pattern, first_word, count);
  checksum.a += value_sum + count32 * domain;
  checksum.b += value_sum * LlmMetalKernelContract::kChecksumValueMultiplier +
                word_index_sum(first_word, count) * LlmMetalKernelContract::kChecksumAddressMultiplier +
                count32 * domain * UINT32_C(0x7feb352d);
}

uint32_t range_boundary_mask(size_t range_start, size_t range_end, uint64_t word_index) noexcept {
  const uint64_t word_start = word_index * sizeof(uint32_t);
  uint32_t valid_mask = 0;
  for (uint32_t byte = 0; byte < sizeof(uint32_t); ++byte) {
    const uint64_t absolute_byte = word_start + byte;
    if (absolute_byte >= range_start && absolute_byte < range_end) {
      valid_mask |= 1U << byte;
    }
  }
  return valid_mask;
}

bool accumulate_pattern_range(LlmMetalMod32Lane& checksum, const Mod32AffinePattern& pattern, size_t range_start,
                              size_t range_length, uint32_t pool_domain, uint32_t visit_domain,
                              uint64_t scenario_seed, uint64_t work_unit, uint64_t layer, uint64_t batch,
                              uint64_t tile_ordinal = 0,
                              uint32_t profile_domain =
                                  LlmMetalKernelContract::kChecksumMetalDecodeContiguousProfileDomain) noexcept {
  size_t range_end = 0;
  if (range_length == 0 || !NumericUtils::checked_add(range_start, range_length, range_end)) {
    return false;
  }
  size_t cursor = range_start;
  if ((cursor % sizeof(uint32_t)) != 0) {
    const uint64_t word_index = cursor / sizeof(uint32_t);
    const uint32_t valid_mask = range_boundary_mask(range_start, range_end, word_index);
    mix_checksum_word(checksum, affine_word(pattern, word_index) & byte_mask_for(valid_mask), word_index,
                      checksum_domain(pool_domain, visit_domain, scenario_seed, work_unit, layer, batch,
                                      tile_ordinal, valid_mask, profile_domain));
    const size_t prefix_bytes = std::min(range_end - cursor, sizeof(uint32_t) - cursor % sizeof(uint32_t));
    cursor += prefix_bytes;
  }
  const uint64_t first_full_word = cursor / sizeof(uint32_t);
  const uint64_t full_word_count = (range_end - cursor) / sizeof(uint32_t);
  if (full_word_count != 0) {
    mix_full_words(checksum, pattern, first_full_word, full_word_count,
                   checksum_domain(pool_domain, visit_domain, scenario_seed, work_unit, layer, batch,
                                   tile_ordinal, 0x0fU, profile_domain));
    cursor += static_cast<size_t>(full_word_count) * sizeof(uint32_t);
  }
  if (cursor < range_end) {
    const uint64_t word_index = cursor / sizeof(uint32_t);
    const uint32_t valid_mask = range_boundary_mask(range_start, range_end, word_index);
    mix_checksum_word(checksum, affine_word(pattern, word_index) & byte_mask_for(valid_mask), word_index,
                      checksum_domain(pool_domain, visit_domain, scenario_seed, work_unit, layer, batch,
                                      tile_ordinal, valid_mask, profile_domain));
  }
  return true;
}

bool range_value_sum(const Mod32AffinePattern& pattern, size_t range_start, size_t range_length,
                     uint32_t& output) noexcept {
  size_t range_end = 0;
  if (range_length == 0 || !NumericUtils::checked_add(range_start, range_length, range_end)) {
    return false;
  }
  output = 0;
  size_t cursor = range_start;
  if ((cursor % sizeof(uint32_t)) != 0) {
    const uint64_t word_index = cursor / sizeof(uint32_t);
    output += affine_word(pattern, word_index) &
              byte_mask_for(range_boundary_mask(range_start, range_end, word_index));
    cursor += std::min(range_end - cursor, sizeof(uint32_t) - cursor % sizeof(uint32_t));
  }
  const uint64_t first_full_word = cursor / sizeof(uint32_t);
  const uint64_t full_word_count = (range_end - cursor) / sizeof(uint32_t);
  output += affine_sum(pattern, first_full_word, full_word_count);
  cursor += static_cast<size_t>(full_word_count) * sizeof(uint32_t);
  if (cursor < range_end) {
    const uint64_t word_index = cursor / sizeof(uint32_t);
    output += affine_word(pattern, word_index) &
              byte_mask_for(range_boundary_mask(range_start, range_end, word_index));
  }
  return true;
}

bool apply_scan_append_correction(LlmMetalMod32Lane& checksum, const Mod32AffinePattern& initial,
                                  const Mod32AffinePattern& appended, size_t range_start,
                                  size_t range_length) noexcept {
  uint32_t initial_sum = 0;
  uint32_t appended_sum = 0;
  if (!range_value_sum(initial, range_start, range_length, initial_sum) ||
      !range_value_sum(appended, range_start, range_length, appended_sum)) {
    return false;
  }
  const uint32_t difference = appended_sum - initial_sum;
  checksum.a += difference;
  checksum.b += difference * LlmMetalKernelContract::kChecksumValueMultiplier;
  return true;
}

Mod32AffinePattern append_pattern(uint64_t scenario_seed, uint64_t work_unit, uint64_t layer, uint64_t batch,
                                  LlmMetalResourcePool pool) noexcept {
  const uint32_t pool_domain = pool == LlmMetalResourcePool::K
                                   ? LlmMetalKernelContract::kAppendKeyDomain
                                   : LlmMetalKernelContract::kAppendValueDomain;
  return {static_cast<uint32_t>(scenario_seed) +
              LlmMetalKernelContract::kAppendWorkUnitMultiplier * static_cast<uint32_t>(work_unit + 1U) +
              LlmMetalKernelContract::kAppendLayerMultiplier * static_cast<uint32_t>(layer + 1U) +
              LlmMetalKernelContract::kAppendBatchMultiplier * static_cast<uint32_t>(batch + 1U) + pool_domain,
          LlmMetalKernelContract::kAppendWordMultiplier};
}

Mod32AffinePattern prefill_write_pattern(
    uint64_t scenario_seed, uint64_t work_unit, uint64_t layer,
    uint64_t batch, LlmMetalResourcePool pool) noexcept {
  const uint32_t pool_domain =
      pool == LlmMetalResourcePool::K
          ? LlmMetalKernelContract::kPrefillWriteKeyDomain
          : LlmMetalKernelContract::kPrefillWriteValueDomain;
  return {static_cast<uint32_t>(scenario_seed) +
              LlmMetalKernelContract::kAppendWorkUnitMultiplier *
                  static_cast<uint32_t>(work_unit + 1U) +
              LlmMetalKernelContract::kAppendLayerMultiplier *
                  static_cast<uint32_t>(layer + 1U) +
              LlmMetalKernelContract::kAppendBatchMultiplier *
                  static_cast<uint32_t>(batch + 1U) +
              pool_domain,
          LlmMetalKernelContract::kAppendWordMultiplier};
}

bool valid_metal_decode_oracle_inputs(const LlmMemoryWorkPlan& model_plan,
                                      const LlmScenarioWorkPlan& scenario_plan) noexcept {
  if (!model_plan.valid || model_plan.backend != LlmMemoryBackend::Metal || model_plan.phase != LlmPhase::Decode ||
      model_plan.kv_layout != LlmKvLayout::Contiguous || !model_plan.geometry.valid ||
      !model_plan.geometry.decode.has_value() || model_plan.geometry.layer_count == 0 ||
      model_plan.geometry.batch_size == 0 || model_plan.geometry.k_or_v_record_bytes_per_layer == 0 ||
      model_plan.geometry.decode->visible_context_tokens == 0 || !scenario_plan.valid ||
      scenario_plan.model_plan_identity != model_plan.plan_identity || scenario_plan.work_units == 0 ||
      scenario_plan.work_units > Constants::LLM_METAL_MAX_WORK_UNITS_PER_DISPATCH) {
    return false;
  }
  switch (scenario_plan.scenario) {
    case LlmScenario::WeightsOnly:
    case LlmScenario::KvOnly:
    case LlmScenario::Mixed:
      return true;
  }
  return false;
}

bool valid_metal_prefill_oracle_inputs(
    const LlmMemoryWorkPlan& model_plan,
    const LlmScenarioWorkPlan& scenario_plan) noexcept {
  if (!model_plan.valid || model_plan.backend != LlmMemoryBackend::Metal ||
      model_plan.phase != LlmPhase::Prefill ||
      model_plan.kv_layout != LlmKvLayout::Contiguous ||
      !model_plan.geometry.valid || !model_plan.geometry.prefill.has_value() ||
      !model_plan.prefill_plan.has_value() ||
      !model_plan.prefill_plan->valid ||
      model_plan.geometry.layer_count == 0 ||
      model_plan.geometry.batch_size == 0 ||
      model_plan.geometry.k_or_v_record_bytes_per_layer == 0 ||
      model_plan.geometry.prefill->prompt_tokens == 0 ||
      model_plan.geometry.prefill->attention_query_tile_tokens == 0 ||
      !scenario_plan.valid ||
      scenario_plan.model_plan_identity != model_plan.plan_identity ||
      scenario_plan.work_units == 0 ||
      scenario_plan.work_units >
          Constants::LLM_METAL_MAX_WORK_UNITS_PER_DISPATCH) {
    return false;
  }
  switch (scenario_plan.scenario) {
    case LlmScenario::WeightsOnly:
    case LlmScenario::KvOnly:
    case LlmScenario::Mixed:
      return true;
  }
  return false;
}

}  // namespace

bool calculate_llm_metal_prefill_maximum_range_vector_span_bytes(
    const LlmGeometry& geometry, LlmScenario scenario,
    size_t& span_bytes) noexcept {
  if (!geometry.valid || geometry.phase != LlmPhase::Prefill ||
      geometry.kv_layout != LlmKvLayout::Contiguous ||
      !geometry.prefill.has_value() || geometry.layer_count == 0 ||
      geometry.batch_size == 0 ||
      geometry.k_or_v_record_bytes_per_layer == 0 ||
      geometry.prefill->prompt_tokens == 0) {
    return false;
  }
  switch (scenario) {
    case LlmScenario::WeightsOnly:
    case LlmScenario::KvOnly:
    case LlmScenario::Mixed:
      break;
    default:
      return false;
  }

  const bool include_weight = scenario != LlmScenario::KvOnly;
  const bool include_kv = scenario != LlmScenario::WeightsOnly;
  size_t maximum_vector_count = 0;
  if (include_weight) {
    const size_t layer_base =
        geometry.active_weight_bytes_per_work_unit / geometry.layer_count;
    const size_t layer_remainder =
        geometry.active_weight_bytes_per_work_unit % geometry.layer_count;
    if (layer_remainder != 0) {
      size_t longer_layer_bytes = 0;
      if (!NumericUtils::checked_add(layer_base, size_t{1},
                                     longer_layer_bytes) ||
          !update_maximum_equal_range_vector_count(
              0, longer_layer_bytes, layer_remainder,
              maximum_vector_count)) {
        return false;
      }
    }
    const size_t shorter_layer_count =
        geometry.layer_count - layer_remainder;
    if (shorter_layer_count != 0) {
      size_t shorter_layer_start = 0;
      if (layer_remainder != 0) {
        size_t longer_layer_bytes = 0;
        if (!NumericUtils::checked_add(layer_base, size_t{1},
                                       longer_layer_bytes) ||
            !NumericUtils::checked_multiply(
                layer_remainder, longer_layer_bytes,
                shorter_layer_start)) {
          return false;
        }
      }
      if (!update_maximum_equal_range_vector_count(
              shorter_layer_start, layer_base, shorter_layer_count,
              maximum_vector_count)) {
        return false;
      }
    }
  }

  if (include_kv) {
    size_t sequence_count = 0;
    size_t sequence_bytes = 0;
    size_t token_range_count = 0;
    if (!NumericUtils::checked_multiply(
            geometry.layer_count, geometry.batch_size, sequence_count) ||
        !NumericUtils::checked_multiply(
            geometry.prefill->prompt_tokens,
            geometry.k_or_v_record_bytes_per_layer, sequence_bytes) ||
        !NumericUtils::checked_multiply(
            sequence_count, geometry.prefill->prompt_tokens,
            token_range_count) ||
        !update_maximum_equal_range_vector_count(
            0, geometry.k_or_v_record_bytes_per_layer, token_range_count,
            maximum_vector_count) ||
        !update_maximum_equal_range_vector_count(
            0, sequence_bytes, sequence_count, maximum_vector_count)) {
      return false;
    }
  }

  size_t checked_span_bytes = 0;
  if (maximum_vector_count == 0 ||
      !NumericUtils::checked_multiply(
          maximum_vector_count, Constants::LLM_METAL_VECTOR_WIDTH_BYTES,
          checked_span_bytes)) {
    return false;
  }
  span_bytes = checked_span_bytes;
  return true;
}

bool calculate_llm_metal_prefill_serial_range_visits_per_lane(
    const LlmMemoryWorkPlan& model_plan,
    const LlmScenarioWorkPlan& scenario_plan, size_t& visits) noexcept {
  if (!valid_metal_prefill_oracle_inputs(model_plan, scenario_plan)) {
    return false;
  }
  size_t visits_per_work_unit = 0;
  if (!calculate_llm_metal_prefill_serial_range_visits_per_work_unit(
          model_plan.geometry, scenario_plan.scenario,
          visits_per_work_unit)) {
    return false;
  }
  size_t checked_visits = 0;
  if (!NumericUtils::checked_multiply(visits_per_work_unit,
                                      scenario_plan.work_units,
                                      checked_visits)) {
    return false;
  }
  visits = checked_visits;
  return true;
}

LlmMetalChecksumOracle calculate_llm_metal_decode_contiguous_checksum(
    const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan& scenario_plan) noexcept {
  LlmMetalChecksumOracle oracle;
  if (!valid_metal_decode_oracle_inputs(model_plan, scenario_plan)) {
    return oracle;
  }
  const LlmGeometry& geometry = model_plan.geometry;
  const bool include_weight = scenario_plan.scenario != LlmScenario::KvOnly;
  const bool include_kv = scenario_plan.scenario != LlmScenario::WeightsOnly;
  const Mod32AffinePattern weight_pattern{static_cast<uint32_t>(model_plan.weight_buffer_seed),
                                          LlmMetalKernelContract::kContiguousPatternWordMultiplier};
  const Mod32AffinePattern key_pattern{static_cast<uint32_t>(model_plan.k_buffer_seed),
                                       LlmMetalKernelContract::kContiguousPatternWordMultiplier};
  const Mod32AffinePattern value_pattern{static_cast<uint32_t>(model_plan.v_buffer_seed),
                                         LlmMetalKernelContract::kContiguousPatternWordMultiplier};
  const size_t weight_layer_base = geometry.active_weight_bytes_per_work_unit / geometry.layer_count;
  const size_t weight_layer_remainder = geometry.active_weight_bytes_per_work_unit % geometry.layer_count;
  size_t sequence_bytes = 0;
  if (!NumericUtils::checked_multiply(geometry.decode->visible_context_tokens,
                                      geometry.k_or_v_record_bytes_per_layer, sequence_bytes)) {
    return oracle;
  }

  for (size_t work_unit = 0; work_unit < scenario_plan.work_units; ++work_unit) {
    size_t weight_offset = 0;
    for (size_t layer = 0; layer < geometry.layer_count; ++layer) {
      const size_t weight_bytes = weight_layer_base + (layer < weight_layer_remainder ? 1U : 0U);
      if (include_weight && weight_bytes != 0 &&
          !accumulate_pattern_range(oracle.checksum.weight, weight_pattern, weight_offset, weight_bytes,
                                    LlmMetalKernelContract::kChecksumWeightDomain,
                                    LlmMetalKernelContract::kChecksumWeightReadVisit, scenario_plan.scenario_seed,
                                    work_unit, layer, 0)) {
        return LlmMetalChecksumOracle{};
      }
      weight_offset += weight_bytes;
      if (!include_kv) {
        continue;
      }
      for (size_t batch = 0; batch < geometry.batch_size; ++batch) {
        size_t sequence_index = 0;
        size_t sequence_start = 0;
        size_t append_start = 0;
        if (!NumericUtils::checked_multiply(layer, geometry.batch_size, sequence_index) ||
            !NumericUtils::checked_add(sequence_index, batch, sequence_index) ||
            !NumericUtils::checked_multiply(sequence_index, sequence_bytes, sequence_start) ||
            !NumericUtils::checked_add(sequence_start, sequence_bytes - geometry.k_or_v_record_bytes_per_layer,
                                       append_start)) {
          return LlmMetalChecksumOracle{};
        }
        const Mod32AffinePattern key_append =
            append_pattern(scenario_plan.scenario_seed, work_unit, layer, batch, LlmMetalResourcePool::K);
        const Mod32AffinePattern value_append =
            append_pattern(scenario_plan.scenario_seed, work_unit, layer, batch, LlmMetalResourcePool::V);
        if (!accumulate_pattern_range(oracle.checksum.k, key_append, append_start,
                                      geometry.k_or_v_record_bytes_per_layer,
                                      LlmMetalKernelContract::kChecksumKeyDomain,
                                      LlmMetalKernelContract::kChecksumAppendVisit, scenario_plan.scenario_seed,
                                      work_unit, layer, batch) ||
            !accumulate_pattern_range(oracle.checksum.v, value_append, append_start,
                                      geometry.k_or_v_record_bytes_per_layer,
                                      LlmMetalKernelContract::kChecksumValueDomain,
                                      LlmMetalKernelContract::kChecksumAppendVisit, scenario_plan.scenario_seed,
                                      work_unit, layer, batch) ||
            !accumulate_pattern_range(oracle.checksum.k, key_pattern, sequence_start, sequence_bytes,
                                      LlmMetalKernelContract::kChecksumKeyDomain,
                                      LlmMetalKernelContract::kChecksumKvReadVisit, scenario_plan.scenario_seed,
                                      work_unit, layer, batch) ||
            !apply_scan_append_correction(oracle.checksum.k, key_pattern, key_append, append_start,
                                          geometry.k_or_v_record_bytes_per_layer) ||
            !accumulate_pattern_range(oracle.checksum.v, value_pattern, sequence_start, sequence_bytes,
                                      LlmMetalKernelContract::kChecksumValueDomain,
                                      LlmMetalKernelContract::kChecksumKvReadVisit, scenario_plan.scenario_seed,
                                      work_unit, layer, batch) ||
            !apply_scan_append_correction(oracle.checksum.v, value_pattern, value_append, append_start,
                                          geometry.k_or_v_record_bytes_per_layer)) {
          return LlmMetalChecksumOracle{};
        }
      }
    }
  }
  oracle.valid = true;
  oracle.reason_code = LlmMetalPlanReason::VALID;
  return oracle;
}

LlmMetalChecksumOracle calculate_llm_metal_prefill_contiguous_checksum(
    const LlmMemoryWorkPlan& model_plan,
    const LlmScenarioWorkPlan& scenario_plan) noexcept {
  LlmMetalChecksumOracle oracle;
  if (!valid_metal_prefill_oracle_inputs(model_plan, scenario_plan)) {
    return oracle;
  }
  size_t serial_range_visits = 0;
  if (!calculate_llm_metal_prefill_serial_range_visits_per_lane(
          model_plan, scenario_plan, serial_range_visits)) {
    oracle.reason_code =
        LlmMetalPlanReason::SERIAL_RANGE_VISIT_COUNT_OVERFLOW;
    return oracle;
  }
  if (serial_range_visits >
      Constants::LLM_METAL_MAX_SERIAL_RANGE_VISITS_PER_LANE_PER_TASK) {
    oracle.reason_code = LlmMetalPlanReason::SERIAL_RANGE_VISIT_CAP_EXCEEDED;
    return oracle;
  }
  const LlmGeometry& geometry = model_plan.geometry;
  const LlmPrefillGeometry& prefill = *geometry.prefill;
  const bool include_weight = scenario_plan.scenario != LlmScenario::KvOnly;
  const bool include_kv = scenario_plan.scenario != LlmScenario::WeightsOnly;
  const Mod32AffinePattern weight_pattern{
      static_cast<uint32_t>(model_plan.weight_buffer_seed),
      LlmMetalKernelContract::kContiguousPatternWordMultiplier};
  const size_t weight_layer_base =
      geometry.active_weight_bytes_per_work_unit / geometry.layer_count;
  const size_t weight_layer_remainder =
      geometry.active_weight_bytes_per_work_unit % geometry.layer_count;
  size_t sequence_bytes = 0;
  if (!NumericUtils::checked_multiply(
          prefill.prompt_tokens,
          geometry.k_or_v_record_bytes_per_layer, sequence_bytes)) {
    return oracle;
  }
  constexpr uint32_t kProfile =
      LlmMetalKernelContract::kChecksumMetalPrefillContiguousProfileDomain;

  for (size_t work_unit = 0; work_unit < scenario_plan.work_units;
       ++work_unit) {
    size_t weight_offset = 0;
    for (size_t layer = 0; layer < geometry.layer_count; ++layer) {
      const size_t weight_bytes =
          weight_layer_base + (layer < weight_layer_remainder ? 1U : 0U);
      if (include_weight && weight_bytes != 0 &&
          !accumulate_pattern_range(
              oracle.checksum.weight, weight_pattern, weight_offset,
              weight_bytes, LlmMetalKernelContract::kChecksumWeightDomain,
              LlmMetalKernelContract::kChecksumWeightReadVisit,
              scenario_plan.scenario_seed, work_unit, layer, 0, 0,
              kProfile)) {
        return LlmMetalChecksumOracle{};
      }
      weight_offset += weight_bytes;
      if (!include_kv) {
        continue;
      }
      for (size_t batch = 0; batch < geometry.batch_size; ++batch) {
        size_t sequence_index = 0;
        size_t sequence_start = 0;
        if (!NumericUtils::checked_multiply(layer, geometry.batch_size,
                                            sequence_index) ||
            !NumericUtils::checked_add(sequence_index, batch,
                                       sequence_index) ||
            !NumericUtils::checked_multiply(sequence_index, sequence_bytes,
                                            sequence_start)) {
          return LlmMetalChecksumOracle{};
        }
        const Mod32AffinePattern key_write = prefill_write_pattern(
            scenario_plan.scenario_seed, work_unit, layer, batch,
            LlmMetalResourcePool::K);
        const Mod32AffinePattern value_write = prefill_write_pattern(
            scenario_plan.scenario_seed, work_unit, layer, batch,
            LlmMetalResourcePool::V);
        for (size_t prompt_token = 0; prompt_token < prefill.prompt_tokens;
             ++prompt_token) {
          size_t token_offset = 0;
          size_t token_start = 0;
          if (!NumericUtils::checked_multiply(
                  prompt_token, geometry.k_or_v_record_bytes_per_layer,
                  token_offset) ||
              !NumericUtils::checked_add(sequence_start, token_offset,
                                         token_start) ||
              !accumulate_pattern_range(
                  oracle.checksum.k, key_write, token_start,
                  geometry.k_or_v_record_bytes_per_layer,
                  LlmMetalKernelContract::kChecksumKeyDomain,
                  LlmMetalKernelContract::kChecksumPrefillWriteVisit,
                  scenario_plan.scenario_seed, work_unit, layer, batch, 0,
                  kProfile) ||
              !accumulate_pattern_range(
                  oracle.checksum.v, value_write, token_start,
                  geometry.k_or_v_record_bytes_per_layer,
                  LlmMetalKernelContract::kChecksumValueDomain,
                  LlmMetalKernelContract::kChecksumPrefillWriteVisit,
                  scenario_plan.scenario_seed, work_unit, layer, batch, 0,
                  kProfile)) {
            return LlmMetalChecksumOracle{};
          }
        }
        size_t remaining_tokens = prefill.prompt_tokens;
        size_t prefix_tokens = 0;
        size_t tile_ordinal = 0;
        while (remaining_tokens != 0) {
          const size_t tile_tokens = std::min(
              prefill.attention_query_tile_tokens, remaining_tokens);
          prefix_tokens += tile_tokens;
          ++tile_ordinal;
          size_t prefix_bytes = 0;
          if (!NumericUtils::checked_multiply(
                  prefix_tokens, geometry.k_or_v_record_bytes_per_layer,
                  prefix_bytes) ||
              !accumulate_pattern_range(
                  oracle.checksum.k, key_write, sequence_start, prefix_bytes,
                  LlmMetalKernelContract::kChecksumKeyDomain,
                  LlmMetalKernelContract::kChecksumKvReadVisit,
                  scenario_plan.scenario_seed, work_unit, layer, batch,
                  tile_ordinal, kProfile) ||
              !accumulate_pattern_range(
                  oracle.checksum.v, value_write, sequence_start,
                  prefix_bytes, LlmMetalKernelContract::kChecksumValueDomain,
                  LlmMetalKernelContract::kChecksumKvReadVisit,
                  scenario_plan.scenario_seed, work_unit, layer, batch,
                  tile_ordinal, kProfile)) {
            return LlmMetalChecksumOracle{};
          }
          remaining_tokens -= tile_tokens;
        }
        if (tile_ordinal != prefill.tile_count) {
          return LlmMetalChecksumOracle{};
        }
      }
    }
  }
  oracle.valid = true;
  oracle.reason_code = LlmMetalPlanReason::VALID;
  return oracle;
}

namespace {

constexpr uint32_t kChecksumDomainMixMultiplier = UINT32_C(0x7feb352d);

uint32_t paged_dynamic_domain(uint64_t scenario_seed, uint64_t work_unit) noexcept {
  return static_cast<uint32_t>(scenario_seed) +
         LlmMetalKernelContract::kChecksumScenarioHighMultiplier *
             static_cast<uint32_t>(scenario_seed >> 32U) +
         LlmMetalKernelContract::kChecksumWorkUnitMultiplier *
             static_cast<uint32_t>(work_unit + 1U);
}

uint32_t paged_static_domain(uint32_t pool_domain, uint32_t visit_domain,
                             uint64_t layer, uint64_t batch,
                             uint64_t logical_table_index,
                             uint64_t physical_id, uint32_t valid_mask) noexcept {
  const uint32_t logical = static_cast<uint32_t>(logical_table_index + 1U);
  const uint32_t physical = static_cast<uint32_t>(physical_id + 1U);
  return LlmMetalKernelContract::kChecksumMetalDecodePagedProfileDomain +
         pool_domain + visit_domain +
         LlmMetalKernelContract::kChecksumLayerMultiplier *
             static_cast<uint32_t>(layer + 1U) +
         LlmMetalKernelContract::kChecksumBatchMultiplier *
             static_cast<uint32_t>(batch + 1U) +
         LlmMetalKernelContract::kChecksumPagedLogicalMultiplier * logical +
         LlmMetalKernelContract::kChecksumPagedPhysicalMultiplier * physical +
         LlmMetalKernelContract::kChecksumPagedPairMultiplier * logical * physical +
         LlmMetalKernelContract::kChecksumValidMaskMultiplier * valid_mask;
}

void mix_checksum_aggregate(LlmMetalMod32Lane& checksum, uint32_t value_sum,
                            uint32_t word_index_sum_value,
                            uint32_t domain_sum) noexcept {
  checksum.a += value_sum + domain_sum;
  checksum.b += value_sum * LlmMetalKernelContract::kChecksumValueMultiplier +
                word_index_sum_value * LlmMetalKernelContract::kChecksumAddressMultiplier +
                domain_sum * kChecksumDomainMixMultiplier;
}

bool add_group_owner(LlmMetalPagedChecksumGroupSummary& group, size_t layer,
                     size_t batch, size_t logical_table_index,
                     uint32_t physical_id) noexcept {
  if (group.count == std::numeric_limits<size_t>::max()) {
    return false;
  }
  ++group.count;
  const uint32_t logical = static_cast<uint32_t>(logical_table_index + 1U);
  const uint32_t physical = physical_id + 1U;
  group.layer_plus_one_sum += static_cast<uint32_t>(layer + 1U);
  group.batch_plus_one_sum += static_cast<uint32_t>(batch + 1U);
  group.logical_plus_one_sum += logical;
  group.physical_plus_one_sum += physical;
  group.logical_physical_pair_sum += logical * physical;
  return true;
}

uint32_t paged_group_static_domain_sum(
    const LlmMetalPagedChecksumGroupSummary& group, uint32_t pool_domain,
    uint32_t visit_domain, uint32_t valid_mask) noexcept {
  const uint32_t count = static_cast<uint32_t>(group.count);
  return count * (LlmMetalKernelContract::kChecksumMetalDecodePagedProfileDomain +
                  pool_domain + visit_domain +
                  LlmMetalKernelContract::kChecksumValidMaskMultiplier * valid_mask) +
         LlmMetalKernelContract::kChecksumLayerMultiplier * group.layer_plus_one_sum +
         LlmMetalKernelContract::kChecksumBatchMultiplier * group.batch_plus_one_sum +
         LlmMetalKernelContract::kChecksumPagedLogicalMultiplier * group.logical_plus_one_sum +
         LlmMetalKernelContract::kChecksumPagedPhysicalMultiplier * group.physical_plus_one_sum +
         LlmMetalKernelContract::kChecksumPagedPairMultiplier *
             group.logical_physical_pair_sum;
}

bool accumulate_paged_static_pattern_range(
    LlmMetalMod32Lane& checksum, size_t& mix_count, uint64_t seed,
    size_t layer, size_t batch, size_t logical_table_index,
    uint32_t physical_id, size_t words_per_block, size_t range_start,
    size_t range_length, uint32_t pool_domain, uint32_t visit_domain) noexcept {
  size_t range_end = 0;
  if (!NumericUtils::checked_add(range_start, range_length, range_end)) {
    return false;
  }
  if (range_length == 0) {
    return true;
  }
  const Mod32AffinePattern pattern{
      static_cast<uint32_t>(seed) +
          LlmMetalKernelContract::kPagedPatternLayerMultiplier *
              static_cast<uint32_t>(layer + 1U) +
          LlmMetalKernelContract::kPagedPatternPhysicalMultiplier *
              static_cast<uint32_t>(physical_id + 1U),
      LlmMetalKernelContract::kPagedPatternWordMultiplier};
  const uint64_t global_word_base =
      static_cast<uint64_t>(logical_table_index) * words_per_block;
  size_t cursor = range_start;
  if ((cursor % sizeof(uint32_t)) != 0) {
    const uint64_t local_word = cursor / sizeof(uint32_t);
    const uint32_t valid_mask = range_boundary_mask(range_start, range_end, local_word);
    mix_checksum_word(
        checksum, affine_word(pattern, local_word) & byte_mask_for(valid_mask),
        global_word_base + local_word,
        paged_static_domain(pool_domain, visit_domain, layer, batch,
                            logical_table_index, physical_id, valid_mask));
    ++mix_count;
    cursor += std::min(range_end - cursor,
                       sizeof(uint32_t) - cursor % sizeof(uint32_t));
  }
  const uint64_t first_full_word = cursor / sizeof(uint32_t);
  const uint64_t full_word_count = (range_end - cursor) / sizeof(uint32_t);
  if (full_word_count != 0) {
    const uint32_t value_sum = affine_sum(pattern, first_full_word, full_word_count);
    const uint32_t address_sum =
        static_cast<uint32_t>(full_word_count) * static_cast<uint32_t>(global_word_base) +
        word_index_sum(first_full_word, full_word_count);
    const uint32_t domain = paged_static_domain(
        pool_domain, visit_domain, layer, batch, logical_table_index,
        physical_id, 0x0fU);
    mix_checksum_aggregate(checksum, value_sum, address_sum,
                           static_cast<uint32_t>(full_word_count) * domain);
    if (!NumericUtils::checked_add(mix_count, static_cast<size_t>(full_word_count), mix_count)) {
      return false;
    }
    cursor += static_cast<size_t>(full_word_count) * sizeof(uint32_t);
  }
  if (cursor < range_end) {
    const uint64_t local_word = cursor / sizeof(uint32_t);
    const uint32_t valid_mask = range_boundary_mask(range_start, range_end, local_word);
    mix_checksum_word(
        checksum, affine_word(pattern, local_word) & byte_mask_for(valid_mask),
        global_word_base + local_word,
        paged_static_domain(pool_domain, visit_domain, layer, batch,
                            logical_table_index, physical_id, valid_mask));
    ++mix_count;
  }
  return true;
}

bool valid_paged_summary_inputs(const LlmMemoryWorkPlan& model_plan,
                                const LlmScenarioWorkPlan& scenario_plan,
                                const LlmMetalPagedChecksumSummary& summary) noexcept {
  const LlmMetalExecutionPlan* execution = get_llm_metal_execution_plan(model_plan);
  if (!summary.valid || !model_plan.valid || model_plan.backend != LlmMemoryBackend::Metal ||
      model_plan.phase != LlmPhase::Decode || model_plan.kv_layout != LlmKvLayout::Paged ||
      !model_plan.geometry.valid || !model_plan.geometry.decode.has_value() || execution == nullptr ||
      !execution->valid || !execution->resources.paged_layout.has_value() || !scenario_plan.valid ||
      scenario_plan.model_plan_identity != model_plan.plan_identity || scenario_plan.work_units == 0 ||
      scenario_plan.work_units > Constants::LLM_METAL_MAX_WORK_UNITS_PER_DISPATCH) {
    return false;
  }
  const LlmKvLayoutPlan& layout = *execution->resources.paged_layout;
  return summary.base_seed == model_plan.base_seed &&
         summary.weight_seed == model_plan.weight_buffer_seed &&
         summary.k_seed == model_plan.k_buffer_seed &&
         summary.v_seed == model_plan.v_buffer_seed &&
         summary.layer_count == layout.layer_count &&
         summary.batch_size == layout.batch_size &&
         summary.record_bytes == layout.k_or_v_record_bytes_per_layer &&
         summary.blocks_per_sequence == layout.blocks_per_sequence &&
         summary.physical_blocks_per_layer == layout.physical_blocks_per_layer &&
         summary.block_bytes == layout.block_bytes &&
         summary.last_block_valid_bytes == layout.last_block_valid_bytes &&
         summary.append_offset_in_last_block == layout.decode_append_offset_in_last_block &&
         summary.words_per_block == (layout.block_bytes / sizeof(uint32_t) +
                                     (layout.block_bytes % sizeof(uint32_t) != 0 ? 1U : 0U));
}

void add_dynamic_domain_to_lane(LlmMetalMod32Lane& checksum, size_t mix_count,
                                uint32_t dynamic_domain) noexcept {
  const uint32_t domain_sum = static_cast<uint32_t>(mix_count) * dynamic_domain;
  checksum.a += domain_sum;
  checksum.b += domain_sum * kChecksumDomainMixMultiplier;
}

void accumulate_paged_lookup(
    LlmMetalMod32Lane& checksum,
    const LlmMetalPagedChecksumGroupSummary& group, uint32_t pool_domain,
    uint32_t visit_domain, uint64_t scenario_seed, size_t work_unit) noexcept {
  const uint32_t count = static_cast<uint32_t>(group.count);
  const uint32_t value_sum = group.physical_plus_one_sum - count;
  const uint32_t address_sum = group.logical_plus_one_sum - count;
  const uint32_t domain_sum =
      paged_group_static_domain_sum(group, pool_domain, visit_domain, 0U) +
      count * paged_dynamic_domain(scenario_seed, work_unit);
  mix_checksum_aggregate(checksum, value_sum, address_sum, domain_sum);
}

uint32_t append_partial_value_sum(
    const LlmMetalPagedChecksumSummary& summary, uint64_t scenario_seed,
    size_t work_unit, uint64_t local_word, uint32_t pool_domain,
    uint32_t valid_mask) noexcept {
  uint32_t sum = 0;
  for (size_t layer = 0; layer < summary.layer_count; ++layer) {
    for (size_t batch = 0; batch < summary.batch_size; ++batch) {
      const uint32_t value =
          static_cast<uint32_t>(scenario_seed) +
          LlmMetalKernelContract::kAppendWorkUnitMultiplier *
              static_cast<uint32_t>(work_unit + 1U) +
          LlmMetalKernelContract::kAppendLayerMultiplier *
              static_cast<uint32_t>(layer + 1U) +
          LlmMetalKernelContract::kAppendBatchMultiplier *
              static_cast<uint32_t>(batch + 1U) +
          LlmMetalKernelContract::kAppendWordMultiplier *
              static_cast<uint32_t>(local_word + 1U) +
          pool_domain;
      sum += value & byte_mask_for(valid_mask);
    }
  }
  return sum;
}

void accumulate_paged_append_range(
    LlmMetalMod32Lane& checksum,
    const LlmMetalPagedChecksumSummary& summary, uint64_t scenario_seed,
    size_t work_unit, uint32_t pool_domain, uint32_t checksum_pool_domain,
    uint32_t visit_domain, size_t range_start, size_t range_end) noexcept {
  if (range_start >= range_end) {
    return;
  }
  const auto& group = summary.terminal_owners;
  const uint32_t count = static_cast<uint32_t>(group.count);
  const uint32_t dynamic = paged_dynamic_domain(scenario_seed, work_unit);
  size_t cursor = range_start;
  const auto mix_partial = [&](uint64_t local_word, uint32_t mask) {
    const uint32_t value_sum = append_partial_value_sum(
        summary, scenario_seed, work_unit, local_word, pool_domain, mask);
    const uint32_t address_sum =
        summary.words_per_block * (group.logical_plus_one_sum - count) +
        count * static_cast<uint32_t>(local_word);
    const uint32_t domain_sum =
        paged_group_static_domain_sum(group, checksum_pool_domain, visit_domain, mask) +
        count * dynamic;
    mix_checksum_aggregate(checksum, value_sum, address_sum, domain_sum);
  };
  if ((cursor % sizeof(uint32_t)) != 0) {
    const uint64_t local_word = cursor / sizeof(uint32_t);
    mix_partial(local_word,
                range_boundary_mask(range_start, range_end, local_word));
    cursor += std::min(range_end - cursor,
                       sizeof(uint32_t) - cursor % sizeof(uint32_t));
  }
  const uint64_t first_full_word = cursor / sizeof(uint32_t);
  const uint64_t full_word_count = (range_end - cursor) / sizeof(uint32_t);
  if (full_word_count != 0) {
    const uint32_t word_count = static_cast<uint32_t>(full_word_count);
    const uint32_t local_word_plus_sum =
        word_count * static_cast<uint32_t>(first_full_word + 1U) +
        triangular_mod32(full_word_count);
    const uint32_t value_sum =
        word_count *
            (count * (static_cast<uint32_t>(scenario_seed) + pool_domain +
                      LlmMetalKernelContract::kAppendWorkUnitMultiplier *
                          static_cast<uint32_t>(work_unit + 1U)) +
             LlmMetalKernelContract::kAppendLayerMultiplier * group.layer_plus_one_sum +
             LlmMetalKernelContract::kAppendBatchMultiplier * group.batch_plus_one_sum) +
        count * LlmMetalKernelContract::kAppendWordMultiplier * local_word_plus_sum;
    const uint32_t logical_zero_sum = group.logical_plus_one_sum - count;
    const uint32_t address_sum =
        word_count * static_cast<uint32_t>(summary.words_per_block) * logical_zero_sum +
        count * word_index_sum(first_full_word, full_word_count);
    const uint32_t per_word_domain =
        paged_group_static_domain_sum(group, checksum_pool_domain, visit_domain, 0x0fU) +
        count * dynamic;
    mix_checksum_aggregate(checksum, value_sum, address_sum,
                           word_count * per_word_domain);
    cursor += static_cast<size_t>(full_word_count) * sizeof(uint32_t);
  }
  if (cursor < range_end) {
    const uint64_t local_word = cursor / sizeof(uint32_t);
    mix_partial(local_word,
                range_boundary_mask(range_start, range_end, local_word));
  }
}

void accumulate_paged_terminal_scan_boundary(
    LlmMetalMod32Lane& checksum,
    const LlmMetalPagedChecksumSummary& summary, uint64_t scenario_seed,
    size_t work_unit, uint32_t pool_domain, uint32_t checksum_pool_domain,
    uint32_t initial_value_sum) noexcept {
  const size_t prefix_bytes =
      summary.append_offset_in_last_block % sizeof(uint32_t);
  if (prefix_bytes == 0) {
    return;
  }
  const auto& group = summary.terminal_owners;
  const uint32_t count = static_cast<uint32_t>(group.count);
  const uint64_t local_word =
      summary.append_offset_in_last_block / sizeof(uint32_t);
  const uint32_t scan_mask = range_boundary_mask(
      0, summary.last_block_valid_bytes, local_word);
  const uint32_t prefix_mask = (1U << prefix_bytes) - 1U;
  const uint32_t append_mask = scan_mask & ~prefix_mask;
  const uint32_t value_sum =
      initial_value_sum +
      append_partial_value_sum(summary, scenario_seed, work_unit, local_word,
                               pool_domain, append_mask);
  const uint32_t address_sum =
      static_cast<uint32_t>(summary.words_per_block) *
          (group.logical_plus_one_sum - count) +
      count * static_cast<uint32_t>(local_word);
  const uint32_t domain_sum =
      paged_group_static_domain_sum(
          group, checksum_pool_domain,
          LlmMetalKernelContract::kChecksumKvReadVisit, scan_mask) +
      count * paged_dynamic_domain(scenario_seed, work_unit);
  mix_checksum_aggregate(checksum, value_sum, address_sum, domain_sum);
}

bool accumulate_paged_weight_checksum(
    LlmMetalMod32Lane& checksum, const LlmMemoryWorkPlan& model_plan,
    const LlmScenarioWorkPlan& scenario_plan) noexcept {
  const LlmGeometry& geometry = model_plan.geometry;
  const Mod32AffinePattern weight_pattern{
      static_cast<uint32_t>(model_plan.weight_buffer_seed),
      LlmMetalKernelContract::kContiguousPatternWordMultiplier};
  const size_t layer_base = geometry.active_weight_bytes_per_work_unit /
                            geometry.layer_count;
  const size_t layer_remainder = geometry.active_weight_bytes_per_work_unit %
                                 geometry.layer_count;
  for (size_t work_unit = 0; work_unit < scenario_plan.work_units; ++work_unit) {
    size_t layer_start = 0;
    for (size_t layer = 0; layer < geometry.layer_count; ++layer) {
      const size_t layer_bytes = layer_base + (layer < layer_remainder ? 1U : 0U);
      if (layer_bytes != 0 && !accumulate_pattern_range(
              checksum, weight_pattern, layer_start, layer_bytes,
              LlmMetalKernelContract::kChecksumWeightDomain,
              LlmMetalKernelContract::kChecksumWeightReadVisit,
              scenario_plan.scenario_seed, work_unit, layer, 0)) {
        return false;
      }
      const size_t first_word = layer_start / sizeof(uint32_t);
      const size_t last_byte = layer_start + layer_bytes;
      const size_t last_word = last_byte / sizeof(uint32_t) +
                               (last_byte % sizeof(uint32_t) != 0 ? 1U : 0U);
      const size_t mix_count = last_word - first_word;
      const uint32_t profile_delta =
          LlmMetalKernelContract::kChecksumMetalDecodePagedProfileDomain -
          LlmMetalKernelContract::kChecksumMetalDecodeContiguousProfileDomain;
      add_dynamic_domain_to_lane(checksum, mix_count, profile_delta);
      layer_start += layer_bytes;
    }
  }
  return true;
}

}  // namespace

LlmMetalPagedChecksumSummary build_llm_metal_decode_paged_checksum_summary(
    const LlmMemoryWorkPlan& model_plan, const uint32_t* table_entries,
    size_t entry_count, const std::function<bool()>& stop_requested) {
  LlmMetalPagedChecksumSummary summary;
  const LlmMetalExecutionPlan* execution = get_llm_metal_execution_plan(model_plan);
  if (!model_plan.valid || model_plan.backend != LlmMemoryBackend::Metal ||
      model_plan.phase != LlmPhase::Decode || model_plan.kv_layout != LlmKvLayout::Paged ||
      execution == nullptr || !execution->valid || !execution->resources.paged_layout.has_value() ||
      table_entries == nullptr) {
    return summary;
  }
  const LlmKvLayoutPlan& layout = *execution->resources.paged_layout;
  if (!layout.valid || entry_count != layout.block_table_entries ||
      layout.blocks_per_sequence == 0 || layout.physical_blocks_per_layer > UINT32_MAX) {
    return summary;
  }
  summary.base_seed = model_plan.base_seed;
  summary.weight_seed = model_plan.weight_buffer_seed;
  summary.k_seed = model_plan.k_buffer_seed;
  summary.v_seed = model_plan.v_buffer_seed;
  summary.layer_count = layout.layer_count;
  summary.batch_size = layout.batch_size;
  summary.record_bytes = layout.k_or_v_record_bytes_per_layer;
  summary.blocks_per_sequence = layout.blocks_per_sequence;
  summary.physical_blocks_per_layer = layout.physical_blocks_per_layer;
  summary.block_bytes = layout.block_bytes;
  summary.last_block_valid_bytes = layout.last_block_valid_bytes;
  summary.append_offset_in_last_block = layout.decode_append_offset_in_last_block;
  summary.words_per_block = layout.block_bytes / sizeof(uint32_t) +
                            (layout.block_bytes % sizeof(uint32_t) != 0 ? 1U : 0U);
  for (size_t layer = 0; layer < layout.layer_count; ++layer) {
    for (size_t logical = 0; logical < entry_count; ++logical) {
      const size_t ordinal = layer * entry_count + logical;
      if ((ordinal % Constants::LLM_KV_PREPARATION_POLL_INTERVAL_ENTRIES) == 0 &&
          stop_requested && stop_requested()) {
        return LlmMetalPagedChecksumSummary{};
      }
      const uint32_t physical = table_entries[logical];
      if (physical >= layout.physical_blocks_per_layer) {
        return LlmMetalPagedChecksumSummary{};
      }
      const size_t batch = logical / layout.blocks_per_sequence;
      const size_t logical_block = logical % layout.blocks_per_sequence;
      if (!add_group_owner(summary.all_owners, layer, batch, logical, physical)) {
        return LlmMetalPagedChecksumSummary{};
      }
      const bool terminal = logical_block + 1 == layout.blocks_per_sequence;
      if (terminal && !add_group_owner(summary.terminal_owners, layer, batch, logical, physical)) {
        return LlmMetalPagedChecksumSummary{};
      }
      const size_t boundary_prefix_bytes =
          layout.decode_append_offset_in_last_block % sizeof(uint32_t);
      const size_t initial_bytes =
          terminal ? layout.decode_append_offset_in_last_block -
                         boundary_prefix_bytes
                   : layout.block_bytes;
      size_t k_mix_count = 0;
      size_t v_mix_count = 0;
      if (!accumulate_paged_static_pattern_range(
              summary.initial_scan_static_checksum.k, k_mix_count,
              model_plan.k_buffer_seed, layer, batch, logical, physical,
              summary.words_per_block, 0, initial_bytes,
              LlmMetalKernelContract::kChecksumKeyDomain,
              LlmMetalKernelContract::kChecksumKvReadVisit) ||
          !accumulate_paged_static_pattern_range(
              summary.initial_scan_static_checksum.v, v_mix_count,
              model_plan.v_buffer_seed, layer, batch, logical, physical,
              summary.words_per_block, 0, initial_bytes,
              LlmMetalKernelContract::kChecksumValueDomain,
              LlmMetalKernelContract::kChecksumKvReadVisit) ||
          k_mix_count != v_mix_count ||
          !NumericUtils::checked_add(summary.initial_scan_mix_count,
                                     k_mix_count,
                                     summary.initial_scan_mix_count)) {
        return LlmMetalPagedChecksumSummary{};
      }
      if (terminal && boundary_prefix_bytes != 0) {
        const uint64_t boundary_word =
            layout.decode_append_offset_in_last_block / sizeof(uint32_t);
        const uint32_t prefix_mask =
            (1U << boundary_prefix_bytes) - 1U;
        const uint32_t byte_mask = byte_mask_for(prefix_mask);
        summary.terminal_boundary_initial_k_value_sum +=
            llm_metal_paged_pattern_word(model_plan.k_buffer_seed, layer,
                                         physical, boundary_word) &
            byte_mask;
        summary.terminal_boundary_initial_v_value_sum +=
            llm_metal_paged_pattern_word(model_plan.v_buffer_seed, layer,
                                         physical, boundary_word) &
            byte_mask;
      }
    }
  }
  size_t expected_all = 0;
  size_t expected_terminal = 0;
  if (!NumericUtils::checked_multiply(layout.layer_count, entry_count, expected_all) ||
      !NumericUtils::checked_multiply(layout.layer_count, layout.batch_size,
                                      expected_terminal) ||
      summary.all_owners.count != expected_all ||
      summary.terminal_owners.count != expected_terminal) {
    return LlmMetalPagedChecksumSummary{};
  }
  summary.valid = true;
  return summary;
}

LlmMetalChecksumOracle calculate_llm_metal_decode_paged_checksum(
    const LlmMemoryWorkPlan& model_plan,
    const LlmScenarioWorkPlan& scenario_plan,
    const LlmMetalPagedChecksumSummary& summary) noexcept {
  LlmMetalChecksumOracle oracle;
  if (!valid_paged_summary_inputs(model_plan, scenario_plan, summary)) {
    return oracle;
  }
  const bool include_weight = scenario_plan.scenario != LlmScenario::KvOnly;
  const bool include_kv = scenario_plan.scenario != LlmScenario::WeightsOnly;
  if (include_weight &&
      !accumulate_paged_weight_checksum(oracle.checksum.weight, model_plan,
                                        scenario_plan)) {
    return LlmMetalChecksumOracle{};
  }
  if (include_kv) {
    size_t scan_append_start = summary.append_offset_in_last_block;
    const size_t boundary_prefix_bytes =
        scan_append_start % sizeof(uint32_t);
    if (boundary_prefix_bytes != 0) {
      scan_append_start += std::min(
          sizeof(uint32_t) - boundary_prefix_bytes,
          summary.last_block_valid_bytes - scan_append_start);
    }
    for (size_t work_unit = 0; work_unit < scenario_plan.work_units;
         ++work_unit) {
      oracle.checksum.k.a += summary.initial_scan_static_checksum.k.a;
      oracle.checksum.k.b += summary.initial_scan_static_checksum.k.b;
      oracle.checksum.v.a += summary.initial_scan_static_checksum.v.a;
      oracle.checksum.v.b += summary.initial_scan_static_checksum.v.b;
      const uint32_t dynamic =
          paged_dynamic_domain(scenario_plan.scenario_seed, work_unit);
      add_dynamic_domain_to_lane(oracle.checksum.k,
                                 summary.initial_scan_mix_count, dynamic);
      add_dynamic_domain_to_lane(oracle.checksum.v,
                                 summary.initial_scan_mix_count, dynamic);

      accumulate_paged_lookup(
          oracle.checksum.k, summary.terminal_owners,
          LlmMetalKernelContract::kChecksumKeyDomain,
          LlmMetalKernelContract::kChecksumPagedAppendLookupVisit,
          scenario_plan.scenario_seed, work_unit);
      accumulate_paged_lookup(
          oracle.checksum.v, summary.terminal_owners,
          LlmMetalKernelContract::kChecksumValueDomain,
          LlmMetalKernelContract::kChecksumPagedAppendLookupVisit,
          scenario_plan.scenario_seed, work_unit);
      accumulate_paged_append_range(
          oracle.checksum.k, summary, scenario_plan.scenario_seed, work_unit,
          LlmMetalKernelContract::kAppendKeyDomain,
          LlmMetalKernelContract::kChecksumKeyDomain,
          LlmMetalKernelContract::kChecksumAppendVisit,
          summary.append_offset_in_last_block,
          summary.last_block_valid_bytes);
      accumulate_paged_append_range(
          oracle.checksum.v, summary, scenario_plan.scenario_seed, work_unit,
          LlmMetalKernelContract::kAppendValueDomain,
          LlmMetalKernelContract::kChecksumValueDomain,
          LlmMetalKernelContract::kChecksumAppendVisit,
          summary.append_offset_in_last_block,
          summary.last_block_valid_bytes);

      accumulate_paged_lookup(
          oracle.checksum.k, summary.all_owners,
          LlmMetalKernelContract::kChecksumKeyDomain,
          LlmMetalKernelContract::kChecksumPagedKeyLookupVisit,
          scenario_plan.scenario_seed, work_unit);
      accumulate_paged_terminal_scan_boundary(
          oracle.checksum.k, summary, scenario_plan.scenario_seed, work_unit,
          LlmMetalKernelContract::kAppendKeyDomain,
          LlmMetalKernelContract::kChecksumKeyDomain,
          summary.terminal_boundary_initial_k_value_sum);
      accumulate_paged_append_range(
          oracle.checksum.k, summary, scenario_plan.scenario_seed, work_unit,
          LlmMetalKernelContract::kAppendKeyDomain,
          LlmMetalKernelContract::kChecksumKeyDomain,
          LlmMetalKernelContract::kChecksumKvReadVisit, scan_append_start,
          summary.last_block_valid_bytes);
      accumulate_paged_lookup(
          oracle.checksum.v, summary.all_owners,
          LlmMetalKernelContract::kChecksumValueDomain,
          LlmMetalKernelContract::kChecksumPagedValueLookupVisit,
          scenario_plan.scenario_seed, work_unit);
      accumulate_paged_terminal_scan_boundary(
          oracle.checksum.v, summary, scenario_plan.scenario_seed, work_unit,
          LlmMetalKernelContract::kAppendValueDomain,
          LlmMetalKernelContract::kChecksumValueDomain,
          summary.terminal_boundary_initial_v_value_sum);
      accumulate_paged_append_range(
          oracle.checksum.v, summary, scenario_plan.scenario_seed, work_unit,
          LlmMetalKernelContract::kAppendValueDomain,
          LlmMetalKernelContract::kChecksumValueDomain,
          LlmMetalKernelContract::kChecksumKvReadVisit, scan_append_start,
          summary.last_block_valid_bytes);
    }
  }
  oracle.valid = true;
  oracle.reason_code = LlmMetalPlanReason::VALID;
  return oracle;
}

LlmBackendLifecycleResult evaluate_llm_metal_capabilities(const LlmMetalCapabilityProbe& probe) noexcept {
  if (!probe.device_available) {
    return {LlmBackendStatus::Unsupported, LlmBackendReason::METAL_DEVICE_UNAVAILABLE};
  }
  if (!probe.has_unified_memory) {
    return {LlmBackendStatus::Unsupported, LlmBackendReason::UNIFIED_MEMORY_REQUIRED};
  }
  if (!probe.apple7_family_supported) {
    return {LlmBackendStatus::Unsupported, LlmBackendReason::APPLE7_FAMILY_REQUIRED};
  }
  if (!probe.argument_buffers_tier2_supported) {
    return {LlmBackendStatus::Unsupported, LlmBackendReason::ARGUMENT_BUFFER_TIER2_REQUIRED};
  }
  if (probe.max_buffer_length < Constants::LLM_METAL_SEGMENT_CAPACITY_BYTES) {
    return {LlmBackendStatus::Unsupported, LlmBackendReason::METAL_MAX_BUFFER_LENGTH_BELOW_SEGMENT_CAPACITY};
  }
  if (!probe.command_queue_created) {
    return {LlmBackendStatus::Failed, LlmBackendReason::METAL_COMMAND_QUEUE_CREATION_FAILED};
  }
  if (!probe.source_compiled) {
    return {LlmBackendStatus::Failed, LlmBackendReason::METAL_KERNEL_COMPILATION_FAILED};
  }
  if (probe.foundation_pipeline_count != kFoundationPipelineCount) {
    return {LlmBackendStatus::Failed, LlmBackendReason::METAL_PIPELINE_CREATION_FAILED};
  }
  if (probe.workload_pipeline_count != kWorkloadPipelineCount) {
    return {LlmBackendStatus::Failed, LlmBackendReason::METAL_PIPELINE_CREATION_FAILED};
  }
  if (!probe.argument_encoder_created) {
    return {LlmBackendStatus::Failed, LlmBackendReason::METAL_ARGUMENT_ENCODER_CREATION_FAILED};
  }
  if (probe.argument_buffer_encoded_length == 0 || !is_power_of_two(probe.argument_buffer_alignment)) {
    return {LlmBackendStatus::Failed, LlmBackendReason::METAL_ARGUMENT_BUFFER_LAYOUT_INVALID};
  }
  return {LlmBackendStatus::Ready, LlmBackendReason::VALID};
}

LlmMetalArgumentBufferPlan build_llm_metal_argument_buffer_plan(size_t weight_segment_count, size_t k_segment_count,
                                                                size_t v_segment_count, size_t table_segment_count,
                                                                size_t segment_slot_cap) {
  LlmMetalArgumentBufferPlan plan;
  if (segment_slot_cap == 0 || weight_segment_count > segment_slot_cap || k_segment_count > segment_slot_cap ||
      v_segment_count > segment_slot_cap || table_segment_count > segment_slot_cap) {
    return plan;
  }
  size_t double_cap = 0;
  size_t triple_cap = 0;
  size_t quadruple_cap = 0;
  if (!NumericUtils::checked_multiply(segment_slot_cap, 2, double_cap) ||
      !NumericUtils::checked_multiply(segment_slot_cap, 3, triple_cap) ||
      !NumericUtils::checked_multiply(segment_slot_cap, 4, quadruple_cap) ||
      !NumericUtils::checked_add(quadruple_cap, 1, plan.encoded_resource_slot_count)) {
    return plan;
  }
  plan.weight_slot_base = 0;
  plan.k_slot_base = segment_slot_cap;
  plan.v_slot_base = double_cap;
  plan.table_slot_base = triple_cap;
  plan.status_slot = quadruple_cap;
  plan.weight_segment_count = weight_segment_count;
  plan.k_segment_count = k_segment_count;
  plan.v_segment_count = v_segment_count;
  plan.table_segment_count = table_segment_count;
  size_t active_without_status = 0;
  if (!NumericUtils::checked_add(weight_segment_count, k_segment_count, active_without_status) ||
      !NumericUtils::checked_add(active_without_status, v_segment_count, active_without_status) ||
      !NumericUtils::checked_add(active_without_status, table_segment_count, active_without_status) ||
      !NumericUtils::checked_add(active_without_status, 1, plan.active_resource_count)) {
    return LlmMetalArgumentBufferPlan{};
  }
  plan.identity = Constants::LLM_METAL_ARGUMENT_BUFFER_ABI_VERSION;
  append_identity_field(plan.identity, "segment_slot_cap", segment_slot_cap);
  append_identity_field(plan.identity, "weight_slot_base", plan.weight_slot_base);
  append_identity_field(plan.identity, "k_slot_base", plan.k_slot_base);
  append_identity_field(plan.identity, "v_slot_base", plan.v_slot_base);
  append_identity_field(plan.identity, "table_slot_base", plan.table_slot_base);
  append_identity_field(plan.identity, "status_slot", plan.status_slot);
  append_identity_field(plan.identity, "encoded_resource_slot_count", plan.encoded_resource_slot_count);
  append_identity_field(plan.identity, "weight_segment_count", weight_segment_count);
  append_identity_field(plan.identity, "k_segment_count", k_segment_count);
  append_identity_field(plan.identity, "v_segment_count", v_segment_count);
  append_identity_field(plan.identity, "table_segment_count", table_segment_count);
  append_identity_field(plan.identity, "active_resource_count", plan.active_resource_count);
  plan.valid = true;
  plan.reason_code = LlmMetalPlanReason::VALID;
  return plan;
}

LlmMetalGridPlan build_llm_metal_grid_plan(const LlmMetalGridRequest& request) {
  LlmMetalGridPlan plan;
  plan.owner_count = request.owner_count;
  plan.work_units = request.work_units;
  plan.paged_semantic_lookups = request.paged_semantic_lookups;
  plan.serial_range_visits_per_lane =
      request.serial_range_visits_per_lane;
  if (request.pipeline.thread_execution_width == 0) {
    return plan;
  }
  const size_t thread_limit =
      std::min(request.pipeline.max_total_threads_per_threadgroup, request.limits.threads_per_threadgroup_cap);
  plan.threads_per_threadgroup =
      (thread_limit / request.pipeline.thread_execution_width) * request.pipeline.thread_execution_width;
  if (plan.threads_per_threadgroup == 0 || request.limits.maximum_threadgroups_per_grid == 0) {
    plan.reason_code = LlmMetalPlanReason::PIPELINE_THREAD_LIMIT_INVALID;
    return plan;
  }
  if (request.work_units > request.limits.maximum_work_units_per_dispatch) {
    plan.reason_code = LlmMetalPlanReason::WORK_UNITS_PER_DISPATCH_CAP_EXCEEDED;
    return plan;
  }
  if (request.serial_range_visits_per_lane >
      request.limits.maximum_serial_range_visits_per_lane_per_task) {
    plan.reason_code = LlmMetalPlanReason::SERIAL_RANGE_VISIT_CAP_EXCEEDED;
    return plan;
  }
  if (request.paged_semantic_lookups > request.limits.maximum_paged_semantic_lookups_per_task) {
    plan.reason_code = LlmMetalPlanReason::SEMANTIC_VISIT_CAP_EXCEEDED;
    return plan;
  }
  plan.actual_threadgroups =
      request.owner_count == 0 ? 0 : std::min(request.owner_count, request.limits.maximum_threadgroups_per_grid);
  if (plan.actual_threadgroups != 0 &&
      !checked_ceil_divide(request.owner_count, plan.actual_threadgroups, plan.owner_ordinals_per_threadgroup)) {
    plan.reason_code = LlmMetalPlanReason::OWNER_COUNT_OVERFLOW;
    return plan;
  }
  if (plan.owner_ordinals_per_threadgroup > request.limits.maximum_owner_ordinals_per_threadgroup) {
    plan.reason_code = LlmMetalPlanReason::OWNER_STRIDE_CAP_EXCEEDED;
    return plan;
  }
  size_t bytes_per_lane_iteration = 0;
  if (!NumericUtils::checked_multiply(plan.threads_per_threadgroup, Constants::LLM_METAL_VECTOR_WIDTH_BYTES,
                                      bytes_per_lane_iteration) ||
      !checked_ceil_divide(request.visit_bytes, bytes_per_lane_iteration, plan.vector_iterations_per_lane_per_visit)) {
    plan.reason_code = LlmMetalPlanReason::OWNER_COUNT_OVERFLOW;
    return plan;
  }
  if (plan.vector_iterations_per_lane_per_visit > request.limits.maximum_vector_iterations_per_lane_per_visit) {
    plan.reason_code = LlmMetalPlanReason::VECTOR_ITERATION_CAP_EXCEEDED;
    return plan;
  }
  if (!request.owner_accounted_bytes.empty()) {
    if (request.owner_accounted_bytes.size() != request.owner_count) {
      plan.reason_code = LlmMetalPlanReason::OWNER_COST_COUNT_MISMATCH;
      return plan;
    }
    try {
      plan.threadgroup_accounted_bytes.assign(plan.actual_threadgroups, 0);
    } catch (const std::bad_alloc&) {
      plan.reason_code = LlmMetalPlanReason::PLANNER_ALLOCATION_FAILED;
      return plan;
    } catch (const std::length_error&) {
      plan.reason_code = LlmMetalPlanReason::PLANNER_ALLOCATION_FAILED;
      return plan;
    }
    for (size_t owner = 0; owner < request.owner_count; ++owner) {
      const size_t threadgroup = owner % plan.actual_threadgroups;
      if (!NumericUtils::checked_add(plan.threadgroup_accounted_bytes[threadgroup],
                                     request.owner_accounted_bytes[owner],
                                     plan.threadgroup_accounted_bytes[threadgroup])) {
        plan.reason_code = LlmMetalPlanReason::OWNER_COUNT_OVERFLOW;
        return plan;
      }
    }
    const auto [minimum, maximum] =
        std::minmax_element(plan.threadgroup_accounted_bytes.begin(), plan.threadgroup_accounted_bytes.end());
    plan.minimum_threadgroup_accounted_bytes = *minimum;
    plan.maximum_threadgroup_accounted_bytes = *maximum;
    if (!checked_subtract(plan.maximum_threadgroup_accounted_bytes, plan.minimum_threadgroup_accounted_bytes,
                          plan.threadgroup_accounted_imbalance_bytes)) {
      plan.reason_code = LlmMetalPlanReason::OWNER_COUNT_OVERFLOW;
      return plan;
    }
  }
  try {
    plan.identity = Constants::LLM_METAL_GRID_PLAN_VERSION;
    append_identity_field(plan.identity, "owner_count", plan.owner_count);
    append_identity_field(plan.identity, "threads_per_threadgroup", plan.threads_per_threadgroup);
    append_identity_field(plan.identity, "actual_threadgroups", plan.actual_threadgroups);
    append_identity_field(plan.identity, "owner_ordinals_per_threadgroup", plan.owner_ordinals_per_threadgroup);
    append_identity_field(plan.identity, "vector_iterations_per_lane_per_visit",
                          plan.vector_iterations_per_lane_per_visit);
    append_identity_field(plan.identity, "work_units", plan.work_units);
    append_identity_field(plan.identity, "paged_semantic_lookups", plan.paged_semantic_lookups);
    append_identity_field(plan.identity, "serial_range_visits_per_lane",
                          plan.serial_range_visits_per_lane);
    append_identity_field(plan.identity, "visit_bytes", request.visit_bytes);
    append_identity_field(plan.identity, "thread_execution_width", request.pipeline.thread_execution_width);
    append_identity_field(plan.identity, "max_total_threads_per_threadgroup",
                          request.pipeline.max_total_threads_per_threadgroup);
    append_identity_field(plan.identity, "threads_per_threadgroup_cap", request.limits.threads_per_threadgroup_cap);
    append_identity_field(plan.identity, "maximum_threadgroups_per_grid", request.limits.maximum_threadgroups_per_grid);
    append_identity_field(plan.identity, "maximum_owner_ordinals_per_threadgroup",
                          request.limits.maximum_owner_ordinals_per_threadgroup);
    append_identity_field(plan.identity, "maximum_vector_iterations_per_lane_per_visit",
                          request.limits.maximum_vector_iterations_per_lane_per_visit);
    append_identity_field(
        plan.identity, "maximum_serial_range_visits_per_lane_per_task",
        request.limits.maximum_serial_range_visits_per_lane_per_task);
    append_identity_field(plan.identity, "maximum_paged_semantic_lookups_per_task",
                          request.limits.maximum_paged_semantic_lookups_per_task);
    append_identity_field(plan.identity, "maximum_work_units_per_dispatch",
                          request.limits.maximum_work_units_per_dispatch);
    append_identity_field(plan.identity, "minimum_threadgroup_accounted_bytes",
                          plan.minimum_threadgroup_accounted_bytes);
    append_identity_field(plan.identity, "maximum_threadgroup_accounted_bytes",
                          plan.maximum_threadgroup_accounted_bytes);
    append_identity_field(plan.identity, "threadgroup_accounted_imbalance_bytes",
                          plan.threadgroup_accounted_imbalance_bytes);
    append_identity_field(plan.identity, "threadgroup_accounted_bytes_count", plan.threadgroup_accounted_bytes.size());
    for (size_t accounted_bytes : plan.threadgroup_accounted_bytes) {
      append_identity_field(plan.identity, "threadgroup_accounted_bytes", accounted_bytes);
    }
  } catch (const std::bad_alloc&) {
    plan.reason_code = LlmMetalPlanReason::PLANNER_ALLOCATION_FAILED;
    return plan;
  } catch (const std::length_error&) {
    plan.reason_code = LlmMetalPlanReason::PLANNER_ALLOCATION_FAILED;
    return plan;
  }
  plan.valid = true;
  plan.reason_code = LlmMetalPlanReason::VALID;
  return plan;
}

LlmMetalExecutionPlan build_llm_metal_execution_plan(const LlmMetalResourcePlanRequest& request) {
  LlmMetalExecutionPlan execution;
  LlmMetalResourcePlan& plan = execution.resources;
  plan.limits = request.limits;
  plan.argument_buffer_encoded_length = request.argument_buffer_encoded_length;
  plan.argument_buffer_alignment = request.argument_buffer_alignment;
  plan.max_buffer_length = request.max_buffer_length;
  plan.host_mapping_granularity_bytes = request.host_mapping_granularity_bytes;
  plan.additional_owned_bytes = request.additional_owned_bytes;
  plan.available_memory_bytes = request.available_memory_bytes;
  if (!request.geometry.valid) {
    execution.reason_code = LlmMetalPlanReason::INVALID_GEOMETRY;
    plan.reason_code = execution.reason_code;
    return execution;
  }
  if (request.host_mapping_granularity_bytes == 0) {
    execution.reason_code = LlmWorkPlanReason::MAPPING_GRANULARITY_ZERO;
    plan.reason_code = execution.reason_code;
    return execution;
  }
  const bool decode_geometry = request.geometry.phase == LlmPhase::Decode &&
                               request.geometry.work_unit_kind == LlmWorkUnitKind::DecodeStep &&
                               request.geometry.decode.has_value() && !request.geometry.prefill.has_value();
  const bool prefill_geometry = request.geometry.phase == LlmPhase::Prefill &&
                                request.geometry.work_unit_kind == LlmWorkUnitKind::PrefillOperation &&
                                request.geometry.prefill.has_value() && !request.geometry.decode.has_value();
  if ((!decode_geometry && !prefill_geometry) ||
      (request.geometry.kv_layout != LlmKvLayout::Contiguous && request.geometry.kv_layout != LlmKvLayout::Paged)) {
    execution.reason_code = LlmMetalPlanReason::INVALID_GEOMETRY;
    plan.reason_code = execution.reason_code;
    return execution;
  }
  if (request.argument_buffer_encoded_length == 0) {
    execution.reason_code = LlmMetalPlanReason::ARGUMENT_ENCODER_LENGTH_ZERO;
    plan.reason_code = execution.reason_code;
    return execution;
  }
  if (!is_power_of_two(request.argument_buffer_alignment)) {
    execution.reason_code = LlmMetalPlanReason::ARGUMENT_ENCODER_ALIGNMENT_INVALID;
    plan.reason_code = execution.reason_code;
    return execution;
  }
  if (request.max_buffer_length < request.limits.segment_capacity_bytes ||
      request.argument_buffer_encoded_length > request.max_buffer_length ||
      plan.status_buffer_length > request.max_buffer_length) {
    execution.reason_code = LlmMetalPlanReason::RESOURCE_LENGTH_EXCEEDS_MAX_BUFFER;
    plan.reason_code = execution.reason_code;
    return execution;
  }
  const LlmKvMetalSegmentLimits segment_limits = {request.limits.segment_capacity_bytes,
                                                  request.limits.segment_slots_per_pool};
  plan.weight_segments =
      build_llm_kv_segment_plan(request.geometry.active_weight_bytes_per_work_unit, 1, segment_limits);
  if (!plan.weight_segments.valid) {
    execution.reason_code = plan.weight_segments.reason_code == LlmKvLayoutReason::SEGMENT_COUNT_EXCEEDS_CAP
                                ? LlmBackendReason::SEGMENT_COUNT_CAP_EXCEEDED
                                : plan.weight_segments.reason_code;
    plan.reason_code = execution.reason_code;
    return execution;
  }

  if (request.geometry.kv_layout == LlmKvLayout::Paged) {
    if (!request.paged_layout.has_value()) {
      execution.reason_code = LlmMetalPlanReason::PAGED_LAYOUT_REQUIRED;
      plan.reason_code = execution.reason_code;
      return execution;
    }
    const LlmKvLayoutPlan& paged = *request.paged_layout;
    const size_t sequence_tokens =
        decode_geometry ? request.geometry.decode->visible_context_tokens : request.geometry.prefill->prompt_tokens;
    if (!paged.valid || paged.sequence_tokens != sequence_tokens ||
        paged.kv_block_tokens != request.geometry.kv_block_tokens ||
        paged.layer_count != request.geometry.layer_count || paged.batch_size != request.geometry.batch_size ||
        paged.k_or_v_record_bytes_per_layer != request.geometry.k_or_v_record_bytes_per_layer ||
        paged.total_physical_blocks != request.geometry.total_physical_blocks ||
        paged.block_bytes != request.geometry.kv_block_bytes ||
        paged.memory.k_physical_bytes != request.geometry.k_mapping_bytes ||
        paged.memory.v_physical_bytes != request.geometry.v_mapping_bytes ||
        paged.memory.block_table_bytes != request.geometry.block_table_bytes) {
      execution.reason_code = LlmMetalPlanReason::PAGED_LAYOUT_MISMATCH;
      plan.reason_code = execution.reason_code;
      return execution;
    }
    const LlmKvMetalSegmentPlan paged_segments = build_llm_kv_metal_segment_plan(paged, segment_limits);
    if (!paged_segments.valid) {
      execution.reason_code = paged_segments.reason_code == LlmKvLayoutReason::SEGMENT_ELEMENT_EXCEEDS_CAPACITY
                                  ? LlmBackendReason::PAGED_BLOCK_EXCEEDS_SEGMENT_CAPACITY
                              : paged_segments.reason_code == LlmKvLayoutReason::SEGMENT_COUNT_EXCEEDS_CAP
                                  ? LlmBackendReason::SEGMENT_COUNT_CAP_EXCEEDED
                                  : paged_segments.reason_code;
      plan.reason_code = execution.reason_code;
      return execution;
    }
    plan.k_segments = paged_segments.k_or_v_pool;
    plan.v_segments = paged_segments.k_or_v_pool;
    plan.table_segments = paged_segments.block_table;
    plan.paged_layout = paged;
    if (!NumericUtils::checked_round_up(paged.memory.block_table_bytes, request.host_mapping_granularity_bytes,
                                        plan.host_permutation_mapping_bytes)) {
      execution.reason_code = LlmMetalPlanReason::RESOURCE_LENGTH_OVERFLOW;
      plan.reason_code = execution.reason_code;
      return execution;
    }
    plan.permutation_validation_bitset_bytes = paged.memory.validation_bitset_bytes;
    plan.staging_buffer_length =
        *std::max_element(plan.table_segments->segment_lengths.begin(), plan.table_segments->segment_lengths.end());
  } else {
    if (request.paged_layout.has_value()) {
      execution.reason_code = LlmMetalPlanReason::PAGED_LAYOUT_MISMATCH;
      plan.reason_code = execution.reason_code;
      return execution;
    }
    plan.k_segments = build_llm_kv_segment_plan(request.geometry.k_mapping_bytes, 1, segment_limits);
    plan.v_segments = build_llm_kv_segment_plan(request.geometry.v_mapping_bytes, 1, segment_limits);
    if (!plan.k_segments.valid || !plan.v_segments.valid) {
      const std::string& segment_reason =
          !plan.k_segments.valid ? plan.k_segments.reason_code : plan.v_segments.reason_code;
      execution.reason_code = segment_reason == LlmKvLayoutReason::SEGMENT_COUNT_EXCEEDS_CAP
                                  ? LlmBackendReason::SEGMENT_COUNT_CAP_EXCEEDED
                                  : segment_reason;
      plan.reason_code = execution.reason_code;
      return execution;
    }
  }

  if (!segment_plan_matches_length(plan.weight_segments, request.geometry.active_weight_bytes_per_work_unit,
                                   request.max_buffer_length) ||
      !segment_plan_matches_length(plan.k_segments, request.geometry.k_mapping_bytes, request.max_buffer_length) ||
      !segment_plan_matches_length(plan.v_segments, request.geometry.v_mapping_bytes, request.max_buffer_length) ||
      (plan.table_segments.has_value() &&
       !segment_plan_matches_length(*plan.table_segments, request.geometry.block_table_bytes,
                                    request.max_buffer_length)) ||
      plan.staging_buffer_length > request.max_buffer_length) {
    execution.reason_code = LlmMetalPlanReason::RESOURCE_LENGTH_EXCEEDS_MAX_BUFFER;
    plan.reason_code = execution.reason_code;
    return execution;
  }

  plan.argument_buffer = build_llm_metal_argument_buffer_plan(
      plan.weight_segments.segment_count, plan.k_segments.segment_count, plan.v_segments.segment_count,
      plan.table_segments.has_value() ? plan.table_segments->segment_count : 0, request.limits.segment_slots_per_pool);
  if (!plan.argument_buffer.valid) {
    execution.reason_code = plan.argument_buffer.reason_code;
    plan.reason_code = execution.reason_code;
    return execution;
  }

  try {
    const size_t table_count = plan.table_segments.has_value() ? plan.table_segments->segment_count : 0;
    const size_t staging_count = plan.staging_buffer_length == 0 ? 0 : 1;
    size_t resource_count = 0;
    if (!NumericUtils::checked_add(plan.weight_segments.segment_count, plan.k_segments.segment_count, resource_count) ||
        !NumericUtils::checked_add(resource_count, plan.v_segments.segment_count, resource_count) ||
        !NumericUtils::checked_add(resource_count, table_count, resource_count) ||
        !NumericUtils::checked_add(resource_count, 2 + staging_count, resource_count)) {
      execution.reason_code = LlmMetalPlanReason::RESOURCE_LENGTH_OVERFLOW;
      plan.reason_code = execution.reason_code;
      return execution;
    }
    plan.planned_resources.reserve(resource_count);
    append_planned_segments(LlmMetalResourcePool::Weight, plan.weight_segments, plan.planned_resources);
    append_planned_segments(LlmMetalResourcePool::K, plan.k_segments, plan.planned_resources);
    append_planned_segments(LlmMetalResourcePool::V, plan.v_segments, plan.planned_resources);
    if (plan.table_segments.has_value()) {
      append_planned_segments(LlmMetalResourcePool::BlockTable, *plan.table_segments, plan.planned_resources);
    }
    plan.planned_resources.push_back(
        {LlmMetalResourcePool::ArgumentBuffer, 0, plan.argument_buffer_encoded_length, true});
    plan.planned_resources.push_back({LlmMetalResourcePool::Status, 0, plan.status_buffer_length, true});
    if (plan.staging_buffer_length != 0) {
      plan.planned_resources.push_back({LlmMetalResourcePool::Staging, 0, plan.staging_buffer_length, false});
    }
  } catch (const std::bad_alloc&) {
    execution.reason_code = LlmMetalPlanReason::PLANNER_ALLOCATION_FAILED;
    plan.reason_code = execution.reason_code;
    return execution;
  } catch (const std::length_error&) {
    execution.reason_code = LlmMetalPlanReason::PLANNER_ALLOCATION_FAILED;
    plan.reason_code = execution.reason_code;
    return execution;
  }

  for (const LlmMetalPlannedResource& resource : plan.planned_resources) {
    if (resource.persistent && !NumericUtils::checked_add(plan.persistent_resource_length_bytes, resource.length_bytes,
                                                          plan.persistent_resource_length_bytes)) {
      execution.reason_code = LlmMetalPlanReason::RESOURCE_LENGTH_OVERFLOW;
      plan.reason_code = execution.reason_code;
      return execution;
    }
  }
  if (!NumericUtils::checked_add(plan.staging_buffer_length, plan.host_permutation_mapping_bytes,
                                 plan.transient_peak_bytes) ||
      !NumericUtils::checked_add(plan.transient_peak_bytes, plan.permutation_validation_bitset_bytes,
                                 plan.transient_peak_bytes) ||
      !NumericUtils::checked_add(plan.persistent_resource_length_bytes, plan.transient_peak_bytes,
                                 plan.known_owned_peak_bytes) ||
      !NumericUtils::checked_add(plan.known_owned_peak_bytes, plan.additional_owned_bytes,
                                 plan.known_owned_peak_bytes) ||
      !calculate_admitted_budget(plan.available_memory_bytes, plan.admitted_budget_bytes)) {
    execution.reason_code = LlmMetalPlanReason::MEMORY_BUDGET_OVERFLOW;
    plan.reason_code = execution.reason_code;
    return execution;
  }
  if (plan.known_owned_peak_bytes > plan.admitted_budget_bytes) {
    execution.reason_code = LlmMetalPlanReason::MEMORY_BUDGET_EXCEEDED;
    plan.reason_code = execution.reason_code;
    return execution;
  }

  try {
    plan.identity = Constants::LLM_METAL_RESOURCE_PLAN_VERSION;
    append_identity_field(plan.identity, "phase", llm_phase_to_string(request.geometry.phase));
    append_identity_field(plan.identity, "kv_layout", llm_kv_layout_to_string(request.geometry.kv_layout));
    append_identity_field(plan.identity, "segment_capacity_bytes", request.limits.segment_capacity_bytes);
    append_identity_field(plan.identity, "segment_slots_per_pool", request.limits.segment_slots_per_pool);
    append_identity_field(plan.identity, "weight_segments", plan.weight_segments.identity);
    append_identity_field(plan.identity, "k_segments", plan.k_segments.identity);
    append_identity_field(plan.identity, "v_segments", plan.v_segments.identity);
    append_identity_field(
        plan.identity, "table_segments",
        plan.table_segments.has_value() ? plan.table_segments->identity : std::string_view{"not-applicable"});
    append_identity_field(
        plan.identity, "paged_layout_geometry",
        plan.paged_layout.has_value() ? plan.paged_layout->geometry_identity : std::string_view{"not-applicable"});
    append_identity_field(plan.identity, "argument_buffer", plan.argument_buffer.identity);
    append_identity_field(plan.identity, "argument_encoded_length", plan.argument_buffer_encoded_length);
    append_identity_field(plan.identity, "argument_alignment", plan.argument_buffer_alignment);
    append_identity_field(plan.identity, "max_buffer_length", plan.max_buffer_length);
    append_identity_field(plan.identity, "host_mapping_granularity_bytes", plan.host_mapping_granularity_bytes);
    append_identity_field(plan.identity, "status_length", plan.status_buffer_length);
    append_identity_field(plan.identity, "staging_length", plan.staging_buffer_length);
    append_identity_field(plan.identity, "persistent_resource_length", plan.persistent_resource_length_bytes);
    append_identity_field(plan.identity, "host_permutation_mapping_bytes", plan.host_permutation_mapping_bytes);
    append_identity_field(plan.identity, "permutation_validation_bitset_bytes",
                          plan.permutation_validation_bitset_bytes);
    append_identity_field(plan.identity, "additional_owned_bytes", plan.additional_owned_bytes);
    append_identity_field(plan.identity, "transient_peak_bytes", plan.transient_peak_bytes);
    append_identity_field(plan.identity, "known_owned_peak_bytes", plan.known_owned_peak_bytes);
    append_identity_field(plan.identity, "available_memory_bytes", plan.available_memory_bytes);
    append_identity_field(plan.identity, "admitted_budget_bytes", plan.admitted_budget_bytes);
    append_identity_field(plan.identity, "planned_resource_count", plan.planned_resources.size());
    for (size_t index = 0; index < plan.planned_resources.size(); ++index) {
      const LlmMetalPlannedResource& resource = plan.planned_resources[index];
      const std::string prefix = "resource_" + std::to_string(index) + "_";
      append_identity_field(plan.identity, prefix + "pool", llm_metal_resource_pool_to_string(resource.pool));
      append_identity_field(plan.identity, prefix + "pool_index", resource.pool_index);
      append_identity_field(plan.identity, prefix + "length_bytes", resource.length_bytes);
      append_identity_field(plan.identity, prefix + "persistent", resource.persistent ? 1U : 0U);
    }
    execution.msl_revision = LlmMetalKernelContract::kRevision;
    execution.msl_source_sha256 = canonical_llm_metal_kernel_source_sha256();
    execution.identity = plan.identity;
    append_identity_field(execution.identity, "msl_revision", execution.msl_revision);
    append_identity_field(execution.identity, "msl_source_sha256", execution.msl_source_sha256);
    append_identity_field(execution.identity, "foundation_parameter_abi",
                          LlmMetalKernelContract::kParameterAbiRevision);
    append_identity_field(
        execution.identity, "workload_parameter_abi",
        request.geometry.phase == LlmPhase::Prefill
            ? LlmMetalKernelContract::kPrefillParameterAbiRevision
            : request.geometry.kv_layout == LlmKvLayout::Paged
                  ? LlmMetalKernelContract::kDecodePagedParameterAbiRevision
                  : LlmMetalKernelContract::kDecodeParameterAbiRevision);
    append_identity_field(execution.identity, "resource_table_abi", LlmMetalKernelContract::kResourceTableAbiRevision);
    append_identity_field(execution.identity, "checksum_algorithm",
                          LlmMetalKernelContract::kChecksumAlgorithmRevision);
    append_identity_field(execution.identity, "threads_per_threadgroup_cap",
                          request.limits.threads_per_threadgroup_cap);
    append_identity_field(execution.identity, "maximum_threadgroups_per_grid",
                          request.limits.maximum_threadgroups_per_grid);
    append_identity_field(execution.identity, "maximum_owner_ordinals_per_threadgroup",
                          request.limits.maximum_owner_ordinals_per_threadgroup);
    append_identity_field(execution.identity, "maximum_vector_iterations_per_lane_per_visit",
                          request.limits.maximum_vector_iterations_per_lane_per_visit);
    append_identity_field(
        execution.identity, "maximum_serial_range_visits_per_lane_per_task",
        request.limits.maximum_serial_range_visits_per_lane_per_task);
    append_identity_field(execution.identity, "maximum_paged_semantic_lookups_per_task",
                          request.limits.maximum_paged_semantic_lookups_per_task);
    append_identity_field(execution.identity, "maximum_work_units_per_dispatch",
                          request.limits.maximum_work_units_per_dispatch);
  } catch (const std::bad_alloc&) {
    execution.reason_code = LlmMetalPlanReason::PLANNER_ALLOCATION_FAILED;
    plan.reason_code = execution.reason_code;
    return execution;
  } catch (const std::length_error&) {
    execution.reason_code = LlmMetalPlanReason::PLANNER_ALLOCATION_FAILED;
    plan.reason_code = execution.reason_code;
    return execution;
  }

  plan.valid = true;
  plan.reason_code = LlmMetalPlanReason::VALID;
  execution.valid = true;
  execution.reason_code = LlmMetalPlanReason::VALID;
  return execution;
}

LlmMetalCommittedAdmission evaluate_llm_metal_committed_admission(
    const LlmMetalResourcePlan& plan, const std::vector<LlmMetalAllocatedResource>& allocations) noexcept {
  LlmMetalCommittedAdmission result;
  result.transient_peak_bytes = plan.transient_peak_bytes;
  result.additional_owned_bytes = plan.additional_owned_bytes;
  result.admitted_budget_bytes = plan.admitted_budget_bytes;
  if (!plan.valid || allocations.size() != plan.planned_resources.size()) {
    result.reason_code = LlmBackendReason::PLAN_RESOURCE_IDENTITY_MISMATCH;
    return result;
  }
  size_t allocated_resource_bytes = 0;
  size_t planned_resource_bytes = 0;
  for (size_t index = 0; index < allocations.size(); ++index) {
    const LlmMetalPlannedResource& planned = plan.planned_resources[index];
    const LlmMetalAllocatedResource& actual = allocations[index];
    if (!same_planned_resource(planned, actual)) {
      result.reason_code = LlmBackendReason::PLAN_RESOURCE_IDENTITY_MISMATCH;
      return result;
    }
    const size_t committed =
        actual.allocated_size_bytes.has_value() ? *actual.allocated_size_bytes : actual.length_bytes;
    if (committed < actual.length_bytes ||
        !NumericUtils::checked_add(allocated_resource_bytes, committed, allocated_resource_bytes) ||
        !NumericUtils::checked_add(planned_resource_bytes, actual.length_bytes, planned_resource_bytes)) {
      result.reason_code = LlmMetalPlanReason::RESOURCE_LENGTH_OVERFLOW;
      return result;
    }
  }
  if (!checked_subtract(allocated_resource_bytes, planned_resource_bytes, result.resource_rounding_bytes) ||
      !NumericUtils::checked_add(allocated_resource_bytes, plan.host_permutation_mapping_bytes,
                                 result.known_owned_peak_bytes) ||
      !NumericUtils::checked_add(result.known_owned_peak_bytes, plan.permutation_validation_bitset_bytes,
                                 result.known_owned_peak_bytes) ||
      !NumericUtils::checked_add(result.known_owned_peak_bytes, plan.additional_owned_bytes,
                                 result.known_owned_peak_bytes)) {
    result.reason_code = LlmMetalPlanReason::MEMORY_BUDGET_OVERFLOW;
    return result;
  }
  result.committed_resource_bytes = allocated_resource_bytes;
  if (result.known_owned_peak_bytes > result.admitted_budget_bytes) {
    result.reason_code = LlmMetalPlanReason::MEMORY_BUDGET_EXCEEDED;
    return result;
  }
  result.valid = true;
  result.reason_code = LlmMetalPlanReason::VALID;
  return result;
}

std::string canonical_llm_metal_kernel_source_sha256() {
  return HashUtils::sha256_hex(LlmMetalKernelContract::kSource);
}

bool validate_llm_metal_layout_probe(const LlmMetalFoundationParams& parameters, const LlmMetalLayoutProbeWords& words,
                                     uint64_t expected_observed_resource_value) noexcept {
  constexpr std::array<uint64_t, 10> kOffsets = {offsetof(LlmMetalFoundationParams, byte_count),
                                                 offsetof(LlmMetalFoundationParams, source_offset_bytes),
                                                 offsetof(LlmMetalFoundationParams, destination_offset_bytes),
                                                 offsetof(LlmMetalFoundationParams, logical_base_bytes),
                                                 offsetof(LlmMetalFoundationParams, pattern_seed),
                                                 offsetof(LlmMetalFoundationParams, block_bytes),
                                                 offsetof(LlmMetalFoundationParams, physical_blocks_per_layer),
                                                 offsetof(LlmMetalFoundationParams, pattern_kind),
                                                 offsetof(LlmMetalFoundationParams, probe_resource_kind),
                                                 offsetof(LlmMetalFoundationParams, probe_resource_slot)};
  const std::array<uint64_t, 10> values = {parameters.byte_count,
                                           parameters.source_offset_bytes,
                                           parameters.destination_offset_bytes,
                                           parameters.logical_base_bytes,
                                           parameters.pattern_seed,
                                           parameters.block_bytes,
                                           parameters.physical_blocks_per_layer,
                                           parameters.pattern_kind,
                                           parameters.probe_resource_kind,
                                           parameters.probe_resource_slot};
  if (words[LlmMetalKernelContract::kProbeAbiVersionIndex] != LlmMetalKernelContract::kFoundationParameterAbiVersion ||
      words[LlmMetalKernelContract::kProbeStructSizeIndex] != sizeof(LlmMetalFoundationParams) ||
      words[LlmMetalKernelContract::kProbeStructAlignmentIndex] != alignof(LlmMetalFoundationParams) ||
      words[LlmMetalKernelContract::kProbeFieldCountIndex] != kOffsets.size()) {
    return false;
  }
  for (size_t index = 0; index < kOffsets.size(); ++index) {
    if (words[LlmMetalKernelContract::kProbeFirstFieldOffsetIndex + index] != kOffsets[index] ||
        words[LlmMetalKernelContract::kProbeFirstFieldValueIndex + index] != values[index]) {
      return false;
    }
  }
  return words[LlmMetalKernelContract::kProbeObservedResourceValueIndex] == expected_observed_resource_value &&
         words[LlmMetalKernelContract::kProbeObservedResourceKindIndex] == parameters.probe_resource_kind &&
         words[LlmMetalKernelContract::kProbeObservedResourceSlotIndex] == parameters.probe_resource_slot &&
         words[LlmMetalKernelContract::kProbeArgumentBufferResourceCountIndex] ==
             LlmMetalKernelContract::kArgumentBufferResourceCount;
}

bool validate_llm_metal_decode_layout_probe(const LlmMetalDecodeContiguousParams& parameters,
                                            const LlmMetalDecodeLayoutProbeWords& words) noexcept {
  constexpr std::array<uint64_t, LlmMetalKernelContract::kDecodeParameterFieldCount> kOffsets = {
      offsetof(LlmMetalDecodeContiguousParams, weight_bytes),
      offsetof(LlmMetalDecodeContiguousParams, k_bytes),
      offsetof(LlmMetalDecodeContiguousParams, v_bytes),
      offsetof(LlmMetalDecodeContiguousParams, segment_capacity_bytes),
      offsetof(LlmMetalDecodeContiguousParams, context_tokens),
      offsetof(LlmMetalDecodeContiguousParams, layer_count),
      offsetof(LlmMetalDecodeContiguousParams, batch_size),
      offsetof(LlmMetalDecodeContiguousParams, record_bytes),
      offsetof(LlmMetalDecodeContiguousParams, work_units),
      offsetof(LlmMetalDecodeContiguousParams, weight_seed),
      offsetof(LlmMetalDecodeContiguousParams, k_seed),
      offsetof(LlmMetalDecodeContiguousParams, v_seed),
      offsetof(LlmMetalDecodeContiguousParams, scenario_seed),
      offsetof(LlmMetalDecodeContiguousParams, weight_segment_count),
      offsetof(LlmMetalDecodeContiguousParams, k_segment_count),
      offsetof(LlmMetalDecodeContiguousParams, v_segment_count),
      offsetof(LlmMetalDecodeContiguousParams, reserved_zero),
  };
  const std::array<uint64_t, LlmMetalKernelContract::kDecodeParameterFieldCount> values = {
      parameters.weight_bytes,
      parameters.k_bytes,
      parameters.v_bytes,
      parameters.segment_capacity_bytes,
      parameters.context_tokens,
      parameters.layer_count,
      parameters.batch_size,
      parameters.record_bytes,
      parameters.work_units,
      parameters.weight_seed,
      parameters.k_seed,
      parameters.v_seed,
      parameters.scenario_seed,
      parameters.weight_segment_count,
      parameters.k_segment_count,
      parameters.v_segment_count,
      parameters.reserved_zero,
  };
  if (words[LlmMetalKernelContract::kDecodeProbeAbiVersionIndex] !=
          LlmMetalKernelContract::kDecodeParameterAbiVersion ||
      words[LlmMetalKernelContract::kDecodeProbeStructSizeIndex] != sizeof(LlmMetalDecodeContiguousParams) ||
      words[LlmMetalKernelContract::kDecodeProbeStructAlignmentIndex] != alignof(LlmMetalDecodeContiguousParams) ||
      words[LlmMetalKernelContract::kDecodeProbeFieldCountIndex] != kOffsets.size()) {
    return false;
  }
  for (size_t index = 0; index < kOffsets.size(); ++index) {
    if (words[LlmMetalKernelContract::kDecodeProbeFirstFieldOffsetIndex + index] != kOffsets[index] ||
        words[LlmMetalKernelContract::kDecodeProbeFirstFieldValueIndex + index] != values[index]) {
      return false;
    }
  }
  return true;
}

bool validate_llm_metal_decode_paged_layout_probe(
    const LlmMetalDecodePagedParams& parameters,
    const LlmMetalDecodePagedLayoutProbeWords& words) noexcept {
  constexpr std::array<uint64_t,
                       LlmMetalKernelContract::kDecodePagedParameterFieldCount>
      kOffsets = {
          offsetof(LlmMetalDecodePagedParams, weight_bytes),
          offsetof(LlmMetalDecodePagedParams, context_tokens),
          offsetof(LlmMetalDecodePagedParams, layer_count),
          offsetof(LlmMetalDecodePagedParams, batch_size),
          offsetof(LlmMetalDecodePagedParams, record_bytes),
          offsetof(LlmMetalDecodePagedParams, work_units),
          offsetof(LlmMetalDecodePagedParams, block_bytes),
          offsetof(LlmMetalDecodePagedParams, last_block_valid_bytes),
          offsetof(LlmMetalDecodePagedParams, append_offset_in_last_block),
          offsetof(LlmMetalDecodePagedParams, blocks_per_sequence),
          offsetof(LlmMetalDecodePagedParams, physical_blocks_per_layer),
          offsetof(LlmMetalDecodePagedParams, blocks_per_segment),
          offsetof(LlmMetalDecodePagedParams, table_entries_per_segment),
          offsetof(LlmMetalDecodePagedParams, segment_capacity_bytes),
          offsetof(LlmMetalDecodePagedParams, weight_seed),
          offsetof(LlmMetalDecodePagedParams, k_seed),
          offsetof(LlmMetalDecodePagedParams, v_seed),
          offsetof(LlmMetalDecodePagedParams, scenario_seed),
          offsetof(LlmMetalDecodePagedParams, weight_segment_count),
          offsetof(LlmMetalDecodePagedParams, k_segment_count),
          offsetof(LlmMetalDecodePagedParams, v_segment_count),
          offsetof(LlmMetalDecodePagedParams, table_segment_count),
          offsetof(LlmMetalDecodePagedParams, reserved_zero),
          offsetof(LlmMetalDecodePagedParams, padding_zero),
      };
  const std::array<uint64_t,
                   LlmMetalKernelContract::kDecodePagedParameterFieldCount>
      values = {
          parameters.weight_bytes,
          parameters.context_tokens,
          parameters.layer_count,
          parameters.batch_size,
          parameters.record_bytes,
          parameters.work_units,
          parameters.block_bytes,
          parameters.last_block_valid_bytes,
          parameters.append_offset_in_last_block,
          parameters.blocks_per_sequence,
          parameters.physical_blocks_per_layer,
          parameters.blocks_per_segment,
          parameters.table_entries_per_segment,
          parameters.segment_capacity_bytes,
          parameters.weight_seed,
          parameters.k_seed,
          parameters.v_seed,
          parameters.scenario_seed,
          parameters.weight_segment_count,
          parameters.k_segment_count,
          parameters.v_segment_count,
          parameters.table_segment_count,
          parameters.reserved_zero,
          parameters.padding_zero,
      };
  if (words[LlmMetalKernelContract::kDecodePagedProbeAbiVersionIndex] !=
          LlmMetalKernelContract::kDecodePagedParameterAbiVersion ||
      words[LlmMetalKernelContract::kDecodePagedProbeStructSizeIndex] !=
          sizeof(LlmMetalDecodePagedParams) ||
      words[LlmMetalKernelContract::kDecodePagedProbeStructAlignmentIndex] !=
          alignof(LlmMetalDecodePagedParams) ||
      words[LlmMetalKernelContract::kDecodePagedProbeFieldCountIndex] !=
          kOffsets.size()) {
    return false;
  }
  for (size_t index = 0; index < kOffsets.size(); ++index) {
    if (words[LlmMetalKernelContract::kDecodePagedProbeFirstFieldOffsetIndex +
              index] != kOffsets[index] ||
        words[LlmMetalKernelContract::kDecodePagedProbeFirstFieldValueIndex +
              index] != values[index]) {
      return false;
    }
  }
  return true;
}

bool validate_llm_metal_prefill_layout_probe(
    const LlmMetalPrefillContiguousParams& parameters,
    const LlmMetalPrefillLayoutProbeWords& words) noexcept {
  constexpr std::array<uint64_t, 19> kOffsets = {
      offsetof(LlmMetalPrefillContiguousParams, weight_bytes),
      offsetof(LlmMetalPrefillContiguousParams, k_bytes),
      offsetof(LlmMetalPrefillContiguousParams, v_bytes),
      offsetof(LlmMetalPrefillContiguousParams, segment_capacity_bytes),
      offsetof(LlmMetalPrefillContiguousParams, prompt_tokens),
      offsetof(LlmMetalPrefillContiguousParams,
               attention_query_tile_tokens),
      offsetof(LlmMetalPrefillContiguousParams, tile_count),
      offsetof(LlmMetalPrefillContiguousParams, layer_count),
      offsetof(LlmMetalPrefillContiguousParams, batch_size),
      offsetof(LlmMetalPrefillContiguousParams, record_bytes),
      offsetof(LlmMetalPrefillContiguousParams, work_units),
      offsetof(LlmMetalPrefillContiguousParams, weight_seed),
      offsetof(LlmMetalPrefillContiguousParams, k_seed),
      offsetof(LlmMetalPrefillContiguousParams, v_seed),
      offsetof(LlmMetalPrefillContiguousParams, scenario_seed),
      offsetof(LlmMetalPrefillContiguousParams, weight_segment_count),
      offsetof(LlmMetalPrefillContiguousParams, k_segment_count),
      offsetof(LlmMetalPrefillContiguousParams, v_segment_count),
      offsetof(LlmMetalPrefillContiguousParams, reserved_zero),
  };
  const std::array<uint64_t, 19> values = {
      parameters.weight_bytes,
      parameters.k_bytes,
      parameters.v_bytes,
      parameters.segment_capacity_bytes,
      parameters.prompt_tokens,
      parameters.attention_query_tile_tokens,
      parameters.tile_count,
      parameters.layer_count,
      parameters.batch_size,
      parameters.record_bytes,
      parameters.work_units,
      parameters.weight_seed,
      parameters.k_seed,
      parameters.v_seed,
      parameters.scenario_seed,
      parameters.weight_segment_count,
      parameters.k_segment_count,
      parameters.v_segment_count,
      parameters.reserved_zero,
  };
  if (words[LlmMetalKernelContract::kPrefillProbeAbiVersionIndex] !=
          LlmMetalKernelContract::kPrefillParameterAbiVersion ||
      words[LlmMetalKernelContract::kPrefillProbeStructSizeIndex] !=
          sizeof(LlmMetalPrefillContiguousParams) ||
      words[LlmMetalKernelContract::kPrefillProbeStructAlignmentIndex] !=
          alignof(LlmMetalPrefillContiguousParams) ||
      words[LlmMetalKernelContract::kPrefillProbeFieldCountIndex] !=
          kOffsets.size()) {
    return false;
  }
  for (size_t index = 0; index < kOffsets.size(); ++index) {
    if (words[LlmMetalKernelContract::kPrefillProbeFirstFieldOffsetIndex +
              index] != kOffsets[index] ||
        words[LlmMetalKernelContract::kPrefillProbeFirstFieldValueIndex +
              index] != values[index]) {
      return false;
    }
  }
  return true;
}

const char* llm_metal_resource_pool_to_string(LlmMetalResourcePool pool) noexcept {
  switch (pool) {
    case LlmMetalResourcePool::Weight:
      return "weight";
    case LlmMetalResourcePool::K:
      return "k";
    case LlmMetalResourcePool::V:
      return "v";
    case LlmMetalResourcePool::BlockTable:
      return "block_table";
    case LlmMetalResourcePool::ArgumentBuffer:
      return "argument_buffer";
    case LlmMetalResourcePool::Status:
      return "status_checksum";
    case LlmMetalResourcePool::Staging:
      return "staging";
  }
  return "unknown";
}

namespace {

struct FoundationPipelines {
  __strong id<MTLComputePipelineState> initialize = nil;
  __strong id<MTLComputePipelineState> copy = nil;
  __strong id<MTLComputePipelineState> probe = nil;
  __strong id<MTLComputePipelineState> probe_decode = nil;
  __strong id<MTLComputePipelineState> validate_bytes = nil;
  __strong id<MTLComputePipelineState> validate_table = nil;
  __strong id<MTLComputePipelineState> decode_weights_only = nil;
  __strong id<MTLComputePipelineState> decode_kv_only = nil;
  __strong id<MTLComputePipelineState> decode_mixed = nil;
  __strong id<MTLComputePipelineState> validate_decode_appends = nil;
};

LlmMetalPipelineEvidence pipeline_evidence(id<MTLComputePipelineState> pipeline, const char* label) {
  LlmMetalPipelineEvidence evidence;
  evidence.label = label;
  if (pipeline != nil) {
    evidence.thread_execution_width = static_cast<size_t>(pipeline.threadExecutionWidth);
    evidence.max_total_threads_per_threadgroup = static_cast<size_t>(pipeline.maxTotalThreadsPerThreadgroup);
  }
  return evidence;
}

constexpr size_t kMetalResourceMetadataStringCount = 5;
constexpr size_t kMetalResourceMetadataStringCapacity = 128;
constexpr size_t kMetalFutureDiagnosticStringCount = 4;
constexpr size_t kMetalBufferReferenceContainerHeaderReserveBytes = 256;
constexpr size_t kMetalBufferReferenceCapacityEntryBytes = sizeof(void*);

/** Model one initWithCapacity container without private capacity APIs. */
bool calculate_metal_buffer_reference_container_bytes(
    size_t requested_capacity, size_t& container_bytes) noexcept {
  size_t capacity_bytes = 0;
  return NumericUtils::checked_multiply(
             requested_capacity,
             kMetalBufferReferenceCapacityEntryBytes, capacity_bytes) &&
         NumericUtils::checked_add(
             kMetalBufferReferenceContainerHeaderReserveBytes,
             capacity_bytes, container_bytes);
}

/**
 * Bound backend-owned host backing that coexists with admitted Metal buffers.
 *
 * Planner storage retains only the simultaneous draft/final two-pass peak.
 * This estimate separately charges the backend's resolved-plan copy and plan
 * identity, capability evidence, active resource metadata, allocation-
 * admission vector, overlapping candidate/published reference arrays, and
 * bounded error strings. Counts come from the attached provisional plan and
 * are rechecked against the finalized plan before the runner begins.
 */
LlmBackendAuxiliaryEstimate calculate_metal_backend_auxiliary_estimate(
    size_t planned_resource_count, size_t persistent_resource_count,
    size_t resolved_execution_plan_backing_bytes,
    size_t resolved_plan_identity_backing_bytes,
    const LlmMetalBackendEvidence& evidence) noexcept {
  LlmBackendAuxiliaryEstimate estimate;
  constexpr size_t kMaximumResourceCount =
      4 * Constants::LLM_METAL_SEGMENT_SLOTS_PER_POOL + 3;
  if (planned_resource_count > kMaximumResourceCount || persistent_resource_count > planned_resource_count) {
    return estimate;
  }

  size_t orchestration_bytes = 0;
  const LlmMetalCapabilityEvidence& capability = evidence.capability;
  const std::array<const std::string*, 9> capability_strings = {
      &capability.device_name,
      &capability.compilation_mode,
      &capability.msl_language_version,
      &capability.kernel_revision,
      &capability.kernel_source_sha256,
      &capability.compiler_diagnostics,
      &capability.compiler_identifier,
      &capability.build_sdk,
      &capability.deployment_target,
  };
  for (const std::string* value : capability_strings) {
    if (!add_external_string_backing(*value, orchestration_bytes)) {
      return estimate;
    }
  }
  if (!add_auxiliary_allocation(capability.supported_families.capacity(), sizeof(std::string),
                                orchestration_bytes)) {
    return estimate;
  }
  for (const std::string& family : capability.supported_families) {
    if (!add_external_string_backing(family, orchestration_bytes)) {
      return estimate;
    }
  }
  const auto add_pipeline_vector = [&](const std::vector<LlmMetalPipelineEvidence>& pipelines) {
    if (!add_auxiliary_allocation(pipelines.capacity(), sizeof(LlmMetalPipelineEvidence), orchestration_bytes)) {
      return false;
    }
    for (const LlmMetalPipelineEvidence& pipeline : pipelines) {
      if (!add_external_string_backing(pipeline.label, orchestration_bytes)) {
        return false;
      }
    }
    return true;
  };
  if (!add_pipeline_vector(capability.foundation_pipelines) || !add_pipeline_vector(evidence.workload_pipelines)) {
    return estimate;
  }

  size_t metadata_string_bytes_per_resource = 0;
  size_t future_diagnostic_bytes = 0;
  size_t candidate_reference_container_bytes = 0;
  size_t published_reference_container_bytes = 0;
  if (!add_auxiliary_allocation(kMetalResourceMetadataStringCount,
                                2 * (kMetalResourceMetadataStringCapacity + 1),
                                metadata_string_bytes_per_resource) ||
      !add_auxiliary_allocation(kMetalFutureDiagnosticStringCount,
                                2 * (Constants::LLM_METAL_DIAGNOSTIC_MAX_BYTES + 1),
                                future_diagnostic_bytes) ||
      !add_auxiliary_allocation(planned_resource_count, sizeof(LlmMetalResourceMetadata), orchestration_bytes) ||
      !add_auxiliary_allocation(planned_resource_count, metadata_string_bytes_per_resource, orchestration_bytes) ||
      !add_auxiliary_allocation(planned_resource_count, sizeof(LlmMetalAllocatedResource), orchestration_bytes) ||
      !calculate_metal_buffer_reference_container_bytes(
          planned_resource_count, candidate_reference_container_bytes) ||
      !calculate_metal_buffer_reference_container_bytes(
          persistent_resource_count,
          published_reference_container_bytes) ||
      !NumericUtils::checked_add(
          orchestration_bytes, candidate_reference_container_bytes,
          orchestration_bytes) ||
      !NumericUtils::checked_add(
          orchestration_bytes, published_reference_container_bytes,
          orchestration_bytes) ||
      !NumericUtils::checked_add(orchestration_bytes, future_diagnostic_bytes, orchestration_bytes) ||
      !NumericUtils::checked_add(
          orchestration_bytes, resolved_execution_plan_backing_bytes,
          orchestration_bytes) ||
      !NumericUtils::checked_add(
          orchestration_bytes, resolved_plan_identity_backing_bytes,
          orchestration_bytes) ||
      !add_auxiliary_allocation(2 * (kMetalResourceMetadataStringCapacity + 1), 1, orchestration_bytes)) {
    return estimate;
  }

  estimate.valid = true;
  estimate.reason_code = LlmBackendReason::VALID;
  estimate.orchestration_auxiliary_bytes = orchestration_bytes;
  estimate.total_auxiliary_bytes = orchestration_bytes;
  return estimate;
}

LlmTaskIdentity metal_task_identity(const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan& scenario_plan,
                                    const LlmRunnerTaskContext& context) noexcept {
  LlmTaskIdentity identity;
  identity.backend = model_plan.backend;
  identity.phase = model_plan.phase;
  identity.kv_layout = model_plan.kv_layout;
  identity.work_unit_kind = scenario_plan.work_unit_kind;
  identity.kv_write_kind = scenario_plan.kv_write_kind;
  identity.task_kind = context.kind;
  identity.scenario = context.scenario;
  identity.attempt_index = context.attempt_index;
  identity.loop_index = context.loop_index;
  identity.order_position = context.order_position;
  identity.purpose = context.purpose;
  identity.model_plan_identity = model_plan.plan_identity;
  identity.scenario_plan_identity = scenario_plan.plan_identity;
  return identity;
}

class LlmMetalBackend final : public LlmBackend {
 public:
  explicit LlmMetalBackend(LlmMetalBackendTestHooks hooks) : hooks_(hooks) {
    evidence_.backend = LlmMemoryBackend::Metal;
    evidence_.backend_evidence = LlmMetalBackendEvidence{};
  }

  ~LlmMetalBackend() override {
    @autoreleasepool {
      release_owned_buffers();
      clear_runtime_objects();
    }
  }

  LlmMemoryBackend kind() const noexcept override { return LlmMemoryBackend::Metal; }

  LlmBackendAuxiliaryEstimate calculate_auxiliary_estimate(
      const LlmAuxiliaryPreflightView& preflight) const noexcept override {
    if (!preflight.valid || preflight.backend != LlmMemoryBackend::Metal) {
      LlmBackendAuxiliaryEstimate estimate;
      estimate.reason_code = LlmBackendReason::BACKEND_MISMATCH;
      return estimate;
    }
    return calculate_metal_backend_auxiliary_estimate(preflight.metal_planned_resource_count,
                                                      preflight.metal_persistent_resource_count,
                                                      preflight.metal_resolved_execution_plan_backing_bytes,
                                                      preflight.metal_resolved_plan_identity_backing_bytes,
                                                      metal_evidence());
  }

  LlmBackendAuxiliaryEstimate calculate_auxiliary_estimate(
      const LlmMemoryWorkPlan& model_plan) const noexcept override {
    const LlmMetalExecutionPlan* const execution = get_llm_metal_execution_plan(model_plan);
    if (model_plan.backend != LlmMemoryBackend::Metal || execution == nullptr) {
      LlmBackendAuxiliaryEstimate estimate;
      estimate.reason_code = LlmBackendReason::BACKEND_MISMATCH;
      return estimate;
    }
    const size_t persistent_resource_count = static_cast<size_t>(
        std::count_if(execution->resources.planned_resources.begin(), execution->resources.planned_resources.end(),
                      [](const LlmMetalPlannedResource& resource) { return resource.persistent; }));
    size_t resolved_execution_plan_backing_bytes = 0;
    size_t resolved_plan_identity_backing_bytes = 0;
    if (execution->valid && execution->resources.valid &&
        !calculate_llm_metal_resolved_plan_backing_bytes(
            model_plan, model_plan.plan_identity.size(),
            resolved_execution_plan_backing_bytes,
            resolved_plan_identity_backing_bytes)) {
      return LlmBackendAuxiliaryEstimate{};
    }
    return calculate_metal_backend_auxiliary_estimate(execution->resources.planned_resources.size(),
                                                      persistent_resource_count,
                                                      resolved_execution_plan_backing_bytes,
                                                      resolved_plan_identity_backing_bytes,
                                                      metal_evidence());
  }

  LlmBackendLifecycleResult initialize(const LlmMemoryConfig& config) noexcept override {
    @autoreleasepool {
      @try {
        try {
          return initialize_impl(config);
        } catch (const std::exception& exception) {
          try {
            return fail_initialization(LlmBackendReason::BACKEND_INITIALIZATION_FAILED,
                                       internal_error(exception.what()));
          } catch (...) {
            return fail_initialization_without_diagnostic(LlmBackendReason::BACKEND_INITIALIZATION_FAILED);
          }
        } catch (...) {
          return fail_initialization_without_diagnostic(LlmBackendReason::BACKEND_INITIALIZATION_FAILED);
        }
      } @catch (NSException* exception) {
        try {
          return fail_initialization(LlmBackendReason::BACKEND_INITIALIZATION_FAILED,
                                     internal_error(ns_string(exception.reason)));
        } catch (...) {
          return fail_initialization_without_diagnostic(LlmBackendReason::BACKEND_INITIALIZATION_FAILED);
        }
      }
    }
  }

  LlmBackendLifecycleResult resolve_execution_plan(const LlmMemoryWorkPlan& model_plan) noexcept override {
    @autoreleasepool {
      @try {
        try {
          return resolve_execution_plan_impl(model_plan);
        } catch (const std::exception& exception) {
          try {
            return fail_plan_resolution(LlmBackendReason::EXECUTION_PLAN_MISMATCH, internal_error(exception.what()));
          } catch (...) {
            return fail_plan_resolution_without_diagnostic(LlmBackendReason::EXECUTION_PLAN_MISMATCH);
          }
        } catch (...) {
          return fail_plan_resolution_without_diagnostic(LlmBackendReason::EXECUTION_PLAN_MISMATCH);
        }
      } @catch (NSException* exception) {
        try {
          return fail_plan_resolution(LlmBackendReason::EXECUTION_PLAN_MISMATCH,
                                      internal_error(ns_string(exception.reason)));
        } catch (...) {
          return fail_plan_resolution_without_diagnostic(LlmBackendReason::EXECUTION_PLAN_MISMATCH);
        }
      }
    }
  }

  LlmBackendLifecycleResult prepare_resources(const LlmMemoryWorkPlan& model_plan) noexcept override {
    @autoreleasepool {
      @try {
        try {
          return prepare_resources_impl(model_plan);
        } catch (const std::exception& exception) {
          try {
            return fail_preparation(LlmBackendReason::METAL_RESOURCE_INITIALIZATION_FAILED,
                                    internal_error(exception.what()));
          } catch (...) {
            return fail_preparation_without_diagnostic(LlmBackendReason::METAL_RESOURCE_INITIALIZATION_FAILED);
          }
        } catch (...) {
          return fail_preparation_without_diagnostic(LlmBackendReason::METAL_RESOURCE_INITIALIZATION_FAILED);
        }
      } @catch (NSException* exception) {
        try {
          return fail_preparation(LlmBackendReason::METAL_RESOURCE_INITIALIZATION_FAILED,
                                  internal_error(ns_string(exception.reason)));
        } catch (...) {
          return fail_preparation_without_diagnostic(LlmBackendReason::METAL_RESOURCE_INITIALIZATION_FAILED);
        }
      }
    }
  }

  LlmTaskExecutionResult execute_task(const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan& scenario_plan,
                                      const LlmRunnerTaskContext& context) noexcept override {
    @autoreleasepool {
      @try {
        try {
          return execute_task_impl(model_plan, scenario_plan, context);
        } catch (const std::exception& exception) {
          LlmTaskExecutionResult result{LlmTaskExecutionStatus::Failed,
                                        LlmBackendReason::TIMED_COMMAND_BUFFER_ERROR};
          result.identity = metal_task_identity(model_plan, scenario_plan, context);
          LlmMetalTaskEvidence metal;
          metal.error = internal_error(exception.what());
          result.backend_evidence = std::move(metal);
          return result;
        } catch (...) {
          LlmTaskExecutionResult result{LlmTaskExecutionStatus::Failed,
                                        LlmBackendReason::TIMED_COMMAND_BUFFER_ERROR};
          result.identity = metal_task_identity(model_plan, scenario_plan, context);
          result.backend_evidence = LlmMetalTaskEvidence{};
          return result;
        }
      } @catch (NSException* exception) {
        LlmTaskExecutionResult result{LlmTaskExecutionStatus::Failed,
                                      LlmBackendReason::TIMED_COMMAND_BUFFER_ERROR};
        result.identity = metal_task_identity(model_plan, scenario_plan, context);
        LlmMetalTaskEvidence metal;
        metal.error = internal_error(ns_string(exception.reason));
        result.backend_evidence = std::move(metal);
        return result;
      }
    }
  }

  const LlmBackendEvidence& evidence() const noexcept override { return evidence_; }

  LlmBackendLifecycleResult release_resources() noexcept override {
    @autoreleasepool {
      @try {
        try {
          release_owned_buffers();
          resources_prepared_ = false;
          preparation_interrupted_ = false;
          plan_resolved_ = false;
          resolved_plan_identity_.clear();
          resolved_execution_plan_ = LlmMetalExecutionPlan{};
          paged_checksum_summary_ = LlmMetalPagedChecksumSummary{};
          LlmMetalBackendEvidence& metal = metal_evidence();
          metal.resources.current_allocated_size_after_release =
              device_ == nil ? 0 : static_cast<uint64_t>(device_.currentAllocatedSize);
          metal.resources.candidate_cleanup_completed = true;
          metal.resources.resources_published = false;
          evidence_.release = {LlmBackendStatus::Ready, LlmBackendReason::VALID};
          return evidence_.release;
        } catch (...) {
          evidence_.release = {LlmBackendStatus::Failed, LlmBackendReason::RESOURCE_RELEASE_FAILED};
          return evidence_.release;
        }
      } @catch (NSException* exception) {
        static_cast<void>(exception);
        evidence_.release = {LlmBackendStatus::Failed, LlmBackendReason::RESOURCE_RELEASE_FAILED};
        return evidence_.release;
      }
    }
  }

 private:
  LlmMetalBackendEvidence& metal_evidence() noexcept {
    return std::get<LlmMetalBackendEvidence>(evidence_.backend_evidence);
  }

  const LlmMetalBackendEvidence& metal_evidence() const noexcept {
    return std::get<LlmMetalBackendEvidence>(evidence_.backend_evidence);
  }

  LlmBackendLifecycleResult fail_initialization(const char* reason, LlmMetalErrorDiagnostic error) {
    LlmMetalBackendEvidence& metal = metal_evidence();
    metal.capability.error = std::move(error);
    initialized_ = true;
    evidence_.initialization = {LlmBackendStatus::Failed, reason};
    return evidence_.initialization;
  }

  LlmBackendLifecycleResult fail_plan_resolution(const char* reason, LlmMetalErrorDiagnostic error) {
    metal_evidence().capability.error = std::move(error);
    evidence_.plan_resolution = {LlmBackendStatus::Failed, reason};
    return evidence_.plan_resolution;
  }

  LlmBackendLifecycleResult fail_preparation(const char* reason, LlmMetalErrorDiagnostic error) {
    release_owned_buffers();
    paged_checksum_summary_ = LlmMetalPagedChecksumSummary{};
    LlmMetalResourceEvidence& resources = metal_evidence().resources;
    resources.error = std::move(error);
    resources.current_allocated_size_after_release =
        device_ == nil ? 0 : static_cast<uint64_t>(device_.currentAllocatedSize);
    resources.candidate_cleanup_completed = true;
    resources.resources_published = false;
    resources_prepared_ = false;
    evidence_.preparation = {LlmBackendStatus::Failed, reason};
    return evidence_.preparation;
  }

  static void clear_diagnostic(LlmMetalErrorDiagnostic& error) noexcept {
    error.domain.clear();
    error.code = 0;
    error.description.clear();
  }

  LlmBackendLifecycleResult fail_initialization_without_diagnostic(const char* reason) noexcept {
    if (auto* metal = std::get_if<LlmMetalBackendEvidence>(&evidence_.backend_evidence)) {
      clear_diagnostic(metal->capability.error);
    }
    initialized_ = true;
    plan_resolved_ = false;
    resources_prepared_ = false;
    evidence_.initialization = {LlmBackendStatus::Failed, reason};
    return evidence_.initialization;
  }

  LlmBackendLifecycleResult fail_plan_resolution_without_diagnostic(const char* reason) noexcept {
    if (auto* metal = std::get_if<LlmMetalBackendEvidence>(&evidence_.backend_evidence)) {
      clear_diagnostic(metal->capability.error);
    }
    evidence_.plan_resolution = {LlmBackendStatus::Failed, reason};
    return evidence_.plan_resolution;
  }

  LlmBackendLifecycleResult fail_preparation_without_diagnostic(const char* reason) noexcept {
    release_owned_buffers();
    paged_checksum_summary_ = LlmMetalPagedChecksumSummary{};
    if (auto* metal = std::get_if<LlmMetalBackendEvidence>(&evidence_.backend_evidence)) {
      clear_diagnostic(metal->resources.error);
      metal->resources.current_allocated_size_after_release = 0;
      metal->resources.candidate_cleanup_completed = true;
      metal->resources.resources_published = false;
    }
    resources_prepared_ = false;
    evidence_.preparation = {LlmBackendStatus::Failed, reason};
    return evidence_.preparation;
  }

  void clear_runtime_objects() noexcept {
    detach_argument_buffer();
    argument_encoder_ = nil;
    pipelines_ = FoundationPipelines{};
    library_ = nil;
    queue_ = nil;
    device_ = nil;
  }

  void detach_argument_buffer() noexcept {
    if (argument_encoder_ == nil) {
      return;
    }
    @try {
      [argument_encoder_ setArgumentBuffer:nil offset:0];
    } @catch (NSException* exception) {
      static_cast<void>(exception);
    }
  }

  void release_owned_buffers() noexcept {
    detach_argument_buffer();
    owned_buffers_ = nil;
  }

  void append_supported_families(std::vector<std::string>& families) {
    struct FamilyEntry {
      NSInteger raw_value;
      const char* name;
    };
    constexpr FamilyEntry kFamilies[] = {{1001, "apple1"},  {1002, "apple2"},  {1003, "apple3"}, {1004, "apple4"},
                                         {1005, "apple5"},  {1006, "apple6"},  {1007, "apple7"}, {1008, "apple8"},
                                         {1009, "apple9"},  {1010, "apple10"}, {2001, "mac1"},   {2002, "mac2"},
                                         {3001, "common1"}, {3002, "common2"}, {3003, "common3"}};
    for (const FamilyEntry& entry : kFamilies) {
      if ([device_ supportsFamily:static_cast<MTLGPUFamily>(entry.raw_value)]) {
        families.emplace_back(entry.name);
      }
    }
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 130000
    if (@available(macOS 13.0, *)) {
      if ([device_ supportsFamily:MTLGPUFamilyMetal3]) {
        families.emplace_back("metal3");
      }
    }
#endif
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
    if (@available(macOS 26.0, *)) {
      if ([device_ supportsFamily:MTLGPUFamilyMetal4]) {
        families.emplace_back("metal4");
      }
    }
#endif
  }

  id<MTLComputePipelineState> create_pipeline(const char* entrypoint, const char* label, NSError** error) {
    if (error != nullptr) {
      *error = nil;
    }
    NSString* const function_name = [NSString stringWithUTF8String:entrypoint];
    id<MTLFunction> function = [library_ newFunctionWithName:function_name];
    if (function == nil) {
      if (error != nullptr) {
        *error = [NSError
            errorWithDomain:@"macos-memory-benchmark.llm-metal"
                       code:1
                   userInfo:@{
                     NSLocalizedDescriptionKey : [NSString stringWithFormat:@"Missing function %@", function_name]
                   }];
      }
      return nil;
    }
    function.label = [NSString stringWithUTF8String:label];
    MTLComputePipelineDescriptor* descriptor = [MTLComputePipelineDescriptor new];
    if (descriptor == nil) {
      if (error != nullptr) {
        *error =
            [NSError errorWithDomain:@"macos-memory-benchmark.llm-metal"
                                code:2
                            userInfo:@{NSLocalizedDescriptionKey : @"MTLComputePipelineDescriptor allocation failed"}];
      }
      return nil;
    }
    descriptor.label = function.label;
    descriptor.computeFunction = function;
    descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
    return [device_ newComputePipelineStateWithDescriptor:descriptor
                                                  options:MTLPipelineOptionNone
                                               reflection:nil
                                                    error:error];
  }

  LlmBackendLifecycleResult initialize_impl(const LlmMemoryConfig& config) {
    release_owned_buffers();
    clear_runtime_objects();
    evidence_ = LlmBackendEvidence{};
    evidence_.backend = LlmMemoryBackend::Metal;
    evidence_.backend_evidence = LlmMetalBackendEvidence{};
    initialized_ = false;
    plan_resolved_ = false;
    resources_prepared_ = false;
    preparation_interrupted_ = false;
    resolved_plan_identity_.clear();
    resolved_execution_plan_ = LlmMetalExecutionPlan{};
    paged_checksum_summary_ = LlmMetalPagedChecksumSummary{};
    table_segment_first_bytes_.fill(0);
    table_segment_first_byte_valid_.fill(false);
    if (config.backend != LlmMemoryBackend::Metal) {
      initialized_ = true;
      evidence_.initialization = {LlmBackendStatus::Failed, LlmBackendReason::BACKEND_MISMATCH};
      return evidence_.initialization;
    }
    const bool decode_profile =
        config.phase == LlmPhase::Decode &&
        (config.kv_layout == LlmKvLayout::Contiguous ||
         config.kv_layout == LlmKvLayout::Paged);
    const bool prefill_profile = config.phase == LlmPhase::Prefill &&
                                 config.kv_layout == LlmKvLayout::Contiguous;
    if (!decode_profile && !prefill_profile) {
      initialized_ = true;
      evidence_.initialization = {LlmBackendStatus::Failed,
                                  LlmBackendReason::TASK_UNSUPPORTED};
      return evidence_.initialization;
    }
    active_phase_ = config.phase;
    active_kv_layout_ = config.kv_layout;

    LlmMetalCapabilityEvidence& capability = metal_evidence().capability;
    capability.kernel_revision = LlmMetalKernelContract::kRevision;
    capability.kernel_source_sha256 = canonical_llm_metal_kernel_source_sha256();
    capability.compiler_identifier = __clang_version__;
#ifdef __MAC_OS_X_VERSION_MAX_ALLOWED
    capability.build_sdk = version_macro_string(__MAC_OS_X_VERSION_MAX_ALLOWED);
#else
    capability.build_sdk = "unknown";
#endif
#ifdef __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__
    capability.deployment_target = version_macro_string(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__);
#else
    capability.deployment_target = "unknown";
#endif

    LlmMetalCapabilityProbe probe;
    device_ = MTLCreateSystemDefaultDevice();
    probe.device_available = device_ != nil;
    if (device_ != nil) {
      capability.device_name = bounded_diagnostic(ns_string(device_.name));
      capability.registry_id = device_.registryID;
      capability.has_unified_memory = device_.hasUnifiedMemory;
      capability.required_apple7_family_supported = [device_ supportsFamily:MTLGPUFamilyApple7];
      capability.argument_buffers_tier2_supported = device_.argumentBuffersSupport == MTLArgumentBuffersTier2;
      capability.max_buffer_length = static_cast<size_t>(device_.maxBufferLength);
      capability.recommended_max_working_set_size = device_.recommendedMaxWorkingSetSize;
      append_supported_families(capability.supported_families);
      probe.has_unified_memory = capability.has_unified_memory;
      probe.apple7_family_supported = capability.required_apple7_family_supported;
      probe.argument_buffers_tier2_supported = capability.argument_buffers_tier2_supported;
      probe.max_buffer_length = capability.max_buffer_length;
    }
    LlmBackendLifecycleResult state = evaluate_llm_metal_capabilities(probe);
    if (state.status == LlmBackendStatus::Unsupported) {
      initialized_ = true;
      evidence_.initialization = state;
      return state;
    }

    queue_ = [device_ newCommandQueue];
    probe.command_queue_created = queue_ != nil;
    if (queue_ != nil) {
      queue_.label = @"membenchmark.llm-metal.command-queue";
    }
    state = evaluate_llm_metal_capabilities(probe);
    if (state.reason_code == LlmBackendReason::METAL_COMMAND_QUEUE_CREATION_FAILED) {
      initialized_ = true;
      evidence_.initialization = state;
      return state;
    }

    MTLCompileOptions* options = [MTLCompileOptions new];
    if (options == nil) {
      capability.error = internal_error("MTLCompileOptions allocation failed");
      initialized_ = true;
      evidence_.initialization = {LlmBackendStatus::Failed, LlmBackendReason::METAL_KERNEL_COMPILATION_FAILED};
      return evidence_.initialization;
    }
    options.languageVersion = MTLLanguageVersion2_3;
    if (active_phase_ == LlmPhase::Prefill) {
      options.preprocessorMacros = @{
        @"LLM_METAL_DECODE_CONTIGUOUS" : @0,
        @"LLM_METAL_DECODE_PAGED" : @0,
        @"LLM_METAL_PREFILL_CONTIGUOUS" : @1
      };
    } else if (active_kv_layout_ == LlmKvLayout::Paged) {
      options.preprocessorMacros = @{
        @"LLM_METAL_DECODE_CONTIGUOUS" : @0,
        @"LLM_METAL_DECODE_PAGED" : @1,
        @"LLM_METAL_PREFILL_CONTIGUOUS" : @0
      };
    } else {
      options.preprocessorMacros = @{
        @"LLM_METAL_DECODE_CONTIGUOUS" : @1,
        @"LLM_METAL_DECODE_PAGED" : @0,
        @"LLM_METAL_PREFILL_CONTIGUOUS" : @0
      };
    }
    NSString* source = [[NSString alloc] initWithBytes:LlmMetalKernelContract::kSource.data()
                                                length:LlmMetalKernelContract::kSource.size()
                                              encoding:NSUTF8StringEncoding];
    NSError* compile_error = nil;
    if (source != nil) {
      library_ = [device_ newLibraryWithSource:source options:options error:&compile_error];
    }
    if (compile_error != nil) {
      capability.compiler_diagnostics = bounded_diagnostic(ns_string(compile_error.localizedDescription));
    }
    probe.source_compiled = library_ != nil;
    if (!probe.source_compiled) {
      capability.error = error_diagnostic(compile_error);
      initialized_ = true;
      evidence_.initialization = {LlmBackendStatus::Failed, LlmBackendReason::METAL_KERNEL_COMPILATION_FAILED};
      return evidence_.initialization;
    }
    library_.label = @"membenchmark.llm-metal.runtime-library";

    NSError* pipeline_error = nil;
    pipelines_.initialize = create_pipeline(LlmMetalKernelContract::kInitializeBytesEntrypoint,
                                            "membenchmark.llm-metal.pipeline.initialize", &pipeline_error);
    if (pipelines_.initialize == nil) {
      capability.error = error_diagnostic(pipeline_error);
      initialized_ = true;
      evidence_.initialization = {LlmBackendStatus::Failed, LlmBackendReason::METAL_PIPELINE_CREATION_FAILED};
      return evidence_.initialization;
    }
    pipelines_.copy = create_pipeline(LlmMetalKernelContract::kCopyBytesEntrypoint,
                                      "membenchmark.llm-metal.pipeline.copy", &pipeline_error);
    if (pipelines_.copy == nil) {
      capability.error = error_diagnostic(pipeline_error);
      initialized_ = true;
      evidence_.initialization = {LlmBackendStatus::Failed, LlmBackendReason::METAL_PIPELINE_CREATION_FAILED};
      return evidence_.initialization;
    }
    pipelines_.probe = create_pipeline(LlmMetalKernelContract::kParameterLayoutProbeEntrypoint,
                                       "membenchmark.llm-metal.pipeline.layout-probe", &pipeline_error);
    if (pipelines_.probe == nil) {
      capability.error = error_diagnostic(pipeline_error);
      initialized_ = true;
      evidence_.initialization = {LlmBackendStatus::Failed, LlmBackendReason::METAL_PIPELINE_CREATION_FAILED};
      return evidence_.initialization;
    }
    const bool paged_profile = active_kv_layout_ == LlmKvLayout::Paged;
    const bool prefill_contiguous_profile =
        active_phase_ == LlmPhase::Prefill;
    const char* const workload_probe_entrypoint =
        prefill_contiguous_profile
            ? LlmMetalKernelContract::kPrefillParameterLayoutProbeEntrypoint
            : paged_profile
                  ? LlmMetalKernelContract::kDecodePagedParameterLayoutProbeEntrypoint
                  : LlmMetalKernelContract::kDecodeParameterLayoutProbeEntrypoint;
    const char* const workload_probe_label =
        prefill_contiguous_profile
            ? "membenchmark.llm-metal.pipeline.prefill-contiguous-layout-probe"
            : paged_profile
                  ? "membenchmark.llm-metal.pipeline.decode-paged-layout-probe"
                  : "membenchmark.llm-metal.pipeline.decode-contiguous-layout-probe";
    pipelines_.probe_decode = create_pipeline(workload_probe_entrypoint,
                                              workload_probe_label,
                                              &pipeline_error);
    if (pipelines_.probe_decode == nil) {
      capability.error = error_diagnostic(pipeline_error);
      initialized_ = true;
      evidence_.initialization = {LlmBackendStatus::Failed, LlmBackendReason::METAL_PIPELINE_CREATION_FAILED};
      return evidence_.initialization;
    }
    pipelines_.validate_bytes = create_pipeline(LlmMetalKernelContract::kValidateBytesEntrypoint,
                                                "membenchmark.llm-metal.pipeline.validate-bytes", &pipeline_error);
    if (pipelines_.validate_bytes == nil) {
      capability.error = error_diagnostic(pipeline_error);
      initialized_ = true;
      evidence_.initialization = {LlmBackendStatus::Failed, LlmBackendReason::METAL_PIPELINE_CREATION_FAILED};
      return evidence_.initialization;
    }
    pipelines_.validate_table = create_pipeline(LlmMetalKernelContract::kValidateTableEntrypoint,
                                                "membenchmark.llm-metal.pipeline.validate-table", &pipeline_error);
    if (pipelines_.validate_table == nil) {
      capability.error = error_diagnostic(pipeline_error);
      initialized_ = true;
      evidence_.initialization = {LlmBackendStatus::Failed, LlmBackendReason::METAL_PIPELINE_CREATION_FAILED};
      return evidence_.initialization;
    }
    const char* const weights_entrypoint =
        prefill_contiguous_profile
            ? LlmMetalKernelContract::kPrefillWeightsOnlyEntrypoint
            : paged_profile
                  ? LlmMetalKernelContract::kDecodePagedWeightsOnlyEntrypoint
                  : LlmMetalKernelContract::kDecodeWeightsOnlyEntrypoint;
    const char* const kv_entrypoint =
        prefill_contiguous_profile
            ? LlmMetalKernelContract::kPrefillKvOnlyEntrypoint
            : paged_profile
                  ? LlmMetalKernelContract::kDecodePagedKvOnlyEntrypoint
                  : LlmMetalKernelContract::kDecodeKvOnlyEntrypoint;
    const char* const mixed_entrypoint =
        prefill_contiguous_profile
            ? LlmMetalKernelContract::kPrefillMixedEntrypoint
            : paged_profile
                  ? LlmMetalKernelContract::kDecodePagedMixedEntrypoint
                  : LlmMetalKernelContract::kDecodeMixedEntrypoint;
    const char* const weights_label =
        prefill_contiguous_profile
            ? "membenchmark.llm-metal.pipeline.prefill-contiguous.weights-only"
            : paged_profile
                  ? "membenchmark.llm-metal.pipeline.decode-paged.weights-only"
                  : "membenchmark.llm-metal.pipeline.decode-contiguous.weights-only";
    const char* const kv_label =
        prefill_contiguous_profile
            ? "membenchmark.llm-metal.pipeline.prefill-contiguous.kv-only"
            : paged_profile
                  ? "membenchmark.llm-metal.pipeline.decode-paged.kv-only"
                  : "membenchmark.llm-metal.pipeline.decode-contiguous.kv-only";
    const char* const mixed_label =
        prefill_contiguous_profile
            ? "membenchmark.llm-metal.pipeline.prefill-contiguous.mixed"
            : paged_profile
                  ? "membenchmark.llm-metal.pipeline.decode-paged.mixed"
                  : "membenchmark.llm-metal.pipeline.decode-contiguous.mixed";
    pipelines_.decode_weights_only = create_pipeline(
        weights_entrypoint, weights_label, &pipeline_error);
    if (pipelines_.decode_weights_only == nil) {
      capability.error = error_diagnostic(pipeline_error);
      initialized_ = true;
      evidence_.initialization = {LlmBackendStatus::Failed, LlmBackendReason::METAL_PIPELINE_CREATION_FAILED};
      return evidence_.initialization;
    }
    pipelines_.decode_kv_only =
        create_pipeline(kv_entrypoint, kv_label, &pipeline_error);
    if (pipelines_.decode_kv_only == nil) {
      capability.error = error_diagnostic(pipeline_error);
      initialized_ = true;
      evidence_.initialization = {LlmBackendStatus::Failed, LlmBackendReason::METAL_PIPELINE_CREATION_FAILED};
      return evidence_.initialization;
    }
    pipelines_.decode_mixed =
        create_pipeline(mixed_entrypoint, mixed_label, &pipeline_error);
    if (pipelines_.decode_mixed == nil) {
      capability.error = error_diagnostic(pipeline_error);
      initialized_ = true;
      evidence_.initialization = {LlmBackendStatus::Failed, LlmBackendReason::METAL_PIPELINE_CREATION_FAILED};
      return evidence_.initialization;
    }
    const char* const validation_entrypoint =
        prefill_contiguous_profile
            ? LlmMetalKernelContract::kValidatePrefillWritesEntrypoint
            : paged_profile
                  ? LlmMetalKernelContract::kValidateDecodePagedEntrypoint
                  : LlmMetalKernelContract::kValidateDecodeAppendsEntrypoint;
    const char* const validation_label =
        prefill_contiguous_profile
            ? "membenchmark.llm-metal.pipeline.validate-prefill-contiguous-writes"
            : paged_profile
                  ? "membenchmark.llm-metal.pipeline.validate-decode-paged-writes-padding"
                  : "membenchmark.llm-metal.pipeline.validate-decode-contiguous-writes";
    pipelines_.validate_decode_appends =
        create_pipeline(validation_entrypoint, validation_label,
                        &pipeline_error);
    if (pipelines_.validate_decode_appends == nil) {
      capability.error = error_diagnostic(pipeline_error);
      initialized_ = true;
      evidence_.initialization = {LlmBackendStatus::Failed, LlmBackendReason::METAL_PIPELINE_CREATION_FAILED};
      return evidence_.initialization;
    }
    const std::array<id<MTLComputePipelineState>, kFoundationPipelineCount> pipeline_array = {
        pipelines_.initialize, pipelines_.copy, pipelines_.probe, pipelines_.probe_decode,
        pipelines_.validate_bytes, pipelines_.validate_table};
    probe.foundation_pipeline_count =
        static_cast<size_t>(std::count_if(pipeline_array.begin(), pipeline_array.end(),
                                          [](id<MTLComputePipelineState> pipeline) { return pipeline != nil; }));
    if (probe.foundation_pipeline_count != kFoundationPipelineCount) {
      capability.error = error_diagnostic(pipeline_error);
      initialized_ = true;
      evidence_.initialization = {LlmBackendStatus::Failed, LlmBackendReason::METAL_PIPELINE_CREATION_FAILED};
      return evidence_.initialization;
    }
    capability.foundation_pipelines = {
        pipeline_evidence(pipelines_.initialize, "membenchmark.llm-metal.pipeline.initialize"),
        pipeline_evidence(pipelines_.copy, "membenchmark.llm-metal.pipeline.copy"),
        pipeline_evidence(pipelines_.probe, "membenchmark.llm-metal.pipeline.layout-probe"),
        pipeline_evidence(pipelines_.probe_decode, workload_probe_label),
        pipeline_evidence(pipelines_.validate_bytes, "membenchmark.llm-metal.pipeline.validate-bytes"),
        pipeline_evidence(pipelines_.validate_table, "membenchmark.llm-metal.pipeline.validate-table")};
    const std::array<id<MTLComputePipelineState>, kWorkloadPipelineCount> workload_pipeline_array = {
        pipelines_.decode_weights_only, pipelines_.decode_kv_only, pipelines_.decode_mixed};
    probe.workload_pipeline_count =
        static_cast<size_t>(std::count_if(workload_pipeline_array.begin(), workload_pipeline_array.end(),
                                          [](id<MTLComputePipelineState> pipeline) { return pipeline != nil; }));
    metal_evidence().workload_pipelines = {
        pipeline_evidence(pipelines_.decode_weights_only, weights_label),
        pipeline_evidence(pipelines_.decode_kv_only, kv_label),
        pipeline_evidence(pipelines_.decode_mixed, mixed_label)};

    id<MTLFunction> probe_function = [library_
        newFunctionWithName:[NSString stringWithUTF8String:LlmMetalKernelContract::kParameterLayoutProbeEntrypoint]];
    argument_encoder_ =
        [probe_function newArgumentEncoderWithBufferIndex:LlmMetalKernelContract::kProbeResourcesBufferIndex];
    probe.argument_encoder_created = argument_encoder_ != nil;
    if (argument_encoder_ != nil) {
      capability.argument_buffer_encoded_length = static_cast<size_t>(argument_encoder_.encodedLength);
      capability.argument_buffer_alignment = static_cast<size_t>(argument_encoder_.alignment);
      probe.argument_buffer_encoded_length = capability.argument_buffer_encoded_length;
      probe.argument_buffer_alignment = capability.argument_buffer_alignment;
    }
    state = evaluate_llm_metal_capabilities(probe);
    initialized_ = true;
    evidence_.initialization = state;
    return state;
  }

  LlmBackendLifecycleResult resolve_execution_plan_impl(const LlmMemoryWorkPlan& model_plan) {
    if (!initialized_ || evidence_.initialization.status != LlmBackendStatus::Ready) {
      evidence_.plan_resolution = {evidence_.initialization.status == LlmBackendStatus::Unsupported
                                       ? LlmBackendStatus::Unsupported
                                       : LlmBackendStatus::Failed,
                                   evidence_.initialization.status == LlmBackendStatus::Unsupported
                                       ? evidence_.initialization.reason_code
                                       : std::string_view{LlmBackendReason::NOT_INITIALIZED}};
      return evidence_.plan_resolution;
    }
    if (resources_prepared_) {
      evidence_.plan_resolution = {LlmBackendStatus::Failed, LlmBackendReason::EXECUTION_PLAN_MISMATCH};
      return evidence_.plan_resolution;
    }
    plan_resolved_ = false;
    resolved_plan_identity_.clear();
    resolved_execution_plan_ = LlmMetalExecutionPlan{};
    weight_seed_ = 0;
    k_seed_ = 0;
    v_seed_ = 0;
    base_seed_ = 0;
    const auto* metal_plan = get_llm_metal_execution_plan(model_plan);
    const LlmMetalCapabilityEvidence& capability = metal_evidence().capability;
    const size_t runtime_host_mapping_granularity = get_system_page_size_bytes();
    if (!model_plan.valid || model_plan.backend != LlmMemoryBackend::Metal ||
        model_plan.phase != active_phase_ ||
        model_plan.kv_layout != active_kv_layout_ ||
        (active_phase_ == LlmPhase::Decode &&
         !model_plan.geometry.decode.has_value()) ||
        (active_phase_ == LlmPhase::Prefill &&
         (!model_plan.geometry.prefill.has_value() ||
          !model_plan.prefill_plan.has_value() ||
          !model_plan.prefill_plan->valid)) ||
        model_plan.phase != model_plan.geometry.phase || model_plan.kv_layout != model_plan.geometry.kv_layout ||
        model_plan.work_unit_kind != model_plan.geometry.work_unit_kind || metal_plan == nullptr ||
        !metal_plan->valid || metal_plan->identity.empty() || model_plan.plan_identity.empty() ||
        metal_plan->resources.argument_buffer_encoded_length != capability.argument_buffer_encoded_length ||
        metal_plan->resources.argument_buffer_alignment != capability.argument_buffer_alignment ||
        runtime_host_mapping_granularity == 0 ||
        metal_plan->resources.host_mapping_granularity_bytes != runtime_host_mapping_granularity ||
        metal_plan->resources.limits.segment_capacity_bytes != Constants::LLM_METAL_SEGMENT_CAPACITY_BYTES ||
        metal_plan->resources.limits.segment_slots_per_pool != Constants::LLM_METAL_SEGMENT_SLOTS_PER_POOL) {
      evidence_.plan_resolution = {LlmBackendStatus::Failed, LlmBackendReason::EXECUTION_PLAN_MISMATCH};
      return evidence_.plan_resolution;
    }

    LlmMetalResourcePlanRequest canonical_request;
    canonical_request.geometry = model_plan.geometry;
    if (model_plan.geometry.kv_layout == LlmKvLayout::Paged) {
      size_t sequence_tokens = 0;
      if (model_plan.geometry.phase == LlmPhase::Decode && model_plan.geometry.decode.has_value()) {
        sequence_tokens = model_plan.geometry.decode->visible_context_tokens;
      } else if (model_plan.geometry.phase == LlmPhase::Prefill && model_plan.geometry.prefill.has_value()) {
        sequence_tokens = model_plan.geometry.prefill->prompt_tokens;
      }
      canonical_request.paged_layout = build_llm_kv_layout_plan(
          {sequence_tokens, model_plan.geometry.kv_block_tokens, model_plan.geometry.layer_count,
           model_plan.geometry.batch_size, model_plan.geometry.k_or_v_record_bytes_per_layer});
    }
    canonical_request.argument_buffer_encoded_length = capability.argument_buffer_encoded_length;
    canonical_request.argument_buffer_alignment = capability.argument_buffer_alignment;
    canonical_request.max_buffer_length = capability.max_buffer_length;
    canonical_request.available_memory_bytes = metal_plan->resources.available_memory_bytes;
    canonical_request.host_mapping_granularity_bytes = runtime_host_mapping_granularity;
    canonical_request.additional_owned_bytes = metal_plan->resources.additional_owned_bytes;
    canonical_request.limits = metal_plan->resources.limits;
    LlmMetalExecutionPlan canonical = build_llm_metal_execution_plan(canonical_request);
    if (!canonical.valid || canonical.identity != metal_plan->identity ||
        !same_resource_plan(canonical.resources, metal_plan->resources) ||
        canonical.msl_revision != metal_plan->msl_revision ||
        canonical.msl_source_sha256 != metal_plan->msl_source_sha256 ||
        canonical.msl_source_sha256 != capability.kernel_source_sha256 ||
        canonical.msl_revision != capability.kernel_revision) {
      evidence_.plan_resolution = {LlmBackendStatus::Failed, LlmBackendReason::PLAN_RESOURCE_IDENTITY_MISMATCH};
      return evidence_.plan_resolution;
    }
    resolved_execution_plan_ = std::move(canonical);
    resolved_plan_identity_ = model_plan.plan_identity;
    weight_seed_ = model_plan.weight_buffer_seed;
    k_seed_ = model_plan.k_buffer_seed;
    v_seed_ = model_plan.v_buffer_seed;
    base_seed_ = model_plan.base_seed;
    plan_resolved_ = true;
    evidence_.plan_resolution = {LlmBackendStatus::Ready, LlmBackendReason::VALID};
    return evidence_.plan_resolution;
  }

  LlmMetalResourceMetadata build_resource_metadata(id<MTLBuffer> buffer, const LlmMetalPlannedResource& planned) const {
    LlmMetalResourceMetadata metadata;
    metadata.label = ns_string(buffer.label);
    metadata.pool = llm_metal_resource_pool_to_string(planned.pool);
    metadata.pool_index = planned.pool_index;
    metadata.storage_mode = storage_mode_string(buffer.storageMode);
    metadata.cpu_cache_mode = cpu_cache_mode_string(buffer.cpuCacheMode);
    metadata.hazard_tracking_mode = hazard_mode_string(buffer.hazardTrackingMode);
    metadata.resource_options = static_cast<uint64_t>(buffer.resourceOptions);
    metadata.length_bytes = static_cast<size_t>(buffer.length);
    if ([buffer respondsToSelector:@selector(allocatedSize)]) {
      metadata.allocated_size_available = true;
      metadata.allocated_size_bytes = static_cast<size_t>(buffer.allocatedSize);
      metadata.committed_bytes = metadata.allocated_size_bytes;
    } else {
      metadata.committed_bytes = metadata.length_bytes;
    }
    if (metadata.committed_bytes >= metadata.length_bytes) {
      metadata.allocation_rounding_bytes = metadata.committed_bytes - metadata.length_bytes;
    }
    return metadata;
  }

  id<MTLBuffer> find_buffer(NSArray<id<MTLBuffer>>* buffers, const std::vector<LlmMetalPlannedResource>& planned,
                            LlmMetalResourcePool pool, size_t pool_index = 0) const noexcept {
    for (size_t index = 0; index < planned.size(); ++index) {
      if (planned[index].pool == pool && planned[index].pool_index == pool_index) {
        return buffers[static_cast<NSUInteger>(index)];
      }
    }
    return nil;
  }

  bool dispatch_for_bytes(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline,
                          size_t byte_count) const noexcept {
    if (encoder == nil || pipeline == nil || byte_count == 0 || pipeline.threadExecutionWidth == 0) {
      return false;
    }
    const size_t width = static_cast<size_t>(pipeline.threadExecutionWidth);
    const size_t thread_limit = std::min(static_cast<size_t>(pipeline.maxTotalThreadsPerThreadgroup),
                                         Constants::LLM_METAL_THREADS_PER_THREADGROUP_CAP);
    const size_t threads = (thread_limit / width) * width;
    size_t vector_count = 0;
    size_t required_threadgroups = 0;
    if (threads == 0 || !checked_ceil_divide(byte_count, Constants::LLM_METAL_VECTOR_WIDTH_BYTES, vector_count) ||
        !checked_ceil_divide(std::max<size_t>(vector_count, 1), threads, required_threadgroups)) {
      return false;
    }
    const size_t threadgroups =
        std::max<size_t>(1, std::min(required_threadgroups, Constants::LLM_METAL_MAX_THREADGROUPS_PER_GRID));
    [encoder dispatchThreadgroups:MTLSizeMake(threadgroups, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    return true;
  }

  bool commit_and_wait(id<MTLCommandBuffer> command_buffer, LlmMetalErrorDiagnostic& error) const {
    if (command_buffer == nil) {
      error = internal_error("commandBuffer returned nil");
      return false;
    }
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
      error = error_diagnostic(command_buffer.error);
      return false;
    }
    return true;
  }

  bool preparation_stop_requested() const {
    return hooks_.stop_requested ? hooks_.stop_requested() : signal_received();
  }

  bool mark_preparation_interrupted(LlmMetalErrorDiagnostic& error) {
    if (!preparation_stop_requested()) {
      return false;
    }
    preparation_interrupted_ = true;
    error = internal_error(LlmBackendReason::PREPARATION_INTERRUPTED);
    return true;
  }

  uint64_t seed_for_pool(LlmMetalResourcePool pool) const noexcept {
    switch (pool) {
      case LlmMetalResourcePool::Weight:
        return weight_seed_;
      case LlmMetalResourcePool::K:
        return k_seed_;
      case LlmMetalResourcePool::V:
        return v_seed_;
      case LlmMetalResourcePool::BlockTable:
      case LlmMetalResourcePool::ArgumentBuffer:
      case LlmMetalResourcePool::Status:
      case LlmMetalResourcePool::Staging:
        return base_seed_;
    }
    return 0;
  }

  uint8_t expected_data_byte(LlmMetalResourcePool pool, const LlmMetalResourcePlan& plan,
                             uint64_t absolute_byte) const noexcept {
    const uint64_t seed = seed_for_pool(pool);
    if ((pool == LlmMetalResourcePool::K || pool == LlmMetalResourcePool::V) && plan.paged_layout.has_value()) {
      return paged_pattern_byte(seed, plan.paged_layout->block_bytes,
                                static_cast<uint32_t>(plan.paged_layout->physical_blocks_per_layer), absolute_byte);
    }
    return contiguous_pattern_byte(seed, absolute_byte);
  }

  bool encode_argument_buffer(NSArray<id<MTLBuffer>>* candidate, const LlmMetalResourcePlan& plan,
                              LlmMetalErrorDiagnostic& error) {
    id<MTLBuffer> argument = find_buffer(candidate, plan.planned_resources, LlmMetalResourcePool::ArgumentBuffer);
    if (argument == nil || argument.contents == nullptr || argument.length != plan.argument_buffer_encoded_length) {
      error = internal_error("argument buffer resource is invalid");
      return false;
    }
    for (const LlmMetalPlannedResource& resource : plan.planned_resources) {
      const size_t resource_id = argument_resource_id(resource.pool, resource.pool_index);
      if (resource_id != std::numeric_limits<size_t>::max() &&
          resource_id >= LlmMetalKernelContract::kArgumentBufferResourceCount) {
        error = internal_error("argument resource ID exceeds ABI");
        return false;
      }
    }
    [argument_encoder_ setArgumentBuffer:argument offset:0];
    for (size_t resource_id = 0; resource_id < LlmMetalKernelContract::kArgumentBufferResourceCount; ++resource_id) {
      [argument_encoder_ setBuffer:nil offset:0 atIndex:resource_id];
    }
    for (size_t index = 0; index < plan.planned_resources.size(); ++index) {
      const LlmMetalPlannedResource& resource = plan.planned_resources[index];
      const size_t resource_id = argument_resource_id(resource.pool, resource.pool_index);
      if (resource_id == std::numeric_limits<size_t>::max()) {
        continue;
      }
      [argument_encoder_ setBuffer:candidate[static_cast<NSUInteger>(index)] offset:0 atIndex:resource_id];
    }
    detach_argument_buffer();
    return true;
  }

  bool initialize_and_validate_data(NSArray<id<MTLBuffer>>* candidate, const LlmMetalResourcePlan& plan,
                                    LlmMetalErrorDiagnostic& error) {
    id<MTLBuffer> status = find_buffer(candidate, plan.planned_resources, LlmMetalResourcePool::Status);
    if (status == nil || status.contents == nullptr || status.length < sizeof(uint32_t)) {
      error = internal_error("status buffer resource is invalid");
      return false;
    }
    std::memset(status.contents, 0, static_cast<size_t>(status.length));
    id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
    if (command_buffer == nil) {
      error = internal_error("initialization commandBuffer returned nil");
      return false;
    }
    command_buffer.label = @"membenchmark.llm-metal.command.initialize-validate";
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    if (encoder == nil) {
      error = internal_error("initialization compute encoder returned nil");
      return false;
    }
    encoder.label = @"membenchmark.llm-metal.encoder.initialize-validate";

    std::array<size_t, 3> logical_bases{};
    for (size_t index = 0; index < plan.planned_resources.size(); ++index) {
      if (mark_preparation_interrupted(error)) {
        [encoder endEncoding];
        return false;
      }
      const LlmMetalPlannedResource& resource = plan.planned_resources[index];
      size_t logical_base_index = 0;
      if (resource.pool == LlmMetalResourcePool::K) {
        logical_base_index = 1;
      } else if (resource.pool == LlmMetalResourcePool::V) {
        logical_base_index = 2;
      } else if (resource.pool != LlmMetalResourcePool::Weight) {
        continue;
      }
      id<MTLBuffer> buffer = candidate[static_cast<NSUInteger>(index)];
      LlmMetalFoundationParams parameters;
      parameters.byte_count = resource.length_bytes;
      parameters.logical_base_bytes = logical_bases[logical_base_index];
      parameters.pattern_seed = seed_for_pool(resource.pool);
      if (resource.pool != LlmMetalResourcePool::Weight && plan.paged_layout.has_value()) {
        parameters.pattern_kind = LlmMetalKernelContract::kPagedPatternKind;
        parameters.block_bytes = plan.paged_layout->block_bytes;
        parameters.physical_blocks_per_layer = static_cast<uint32_t>(plan.paged_layout->physical_blocks_per_layer);
      } else {
        parameters.pattern_kind = LlmMetalKernelContract::kContiguousPatternKind;
      }
      [encoder setComputePipelineState:pipelines_.initialize];
      [encoder setBuffer:buffer offset:0 atIndex:LlmMetalKernelContract::kInitializeDestinationBufferIndex];
      [encoder setBytes:&parameters
                 length:sizeof(parameters)
                atIndex:LlmMetalKernelContract::kInitializeParametersBufferIndex];
      if (!dispatch_for_bytes(encoder, pipelines_.initialize, resource.length_bytes)) {
        [encoder endEncoding];
        error = internal_error("invalid initialization dispatch geometry");
        return false;
      }
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      [encoder setComputePipelineState:pipelines_.validate_bytes];
      [encoder setBuffer:buffer offset:0 atIndex:LlmMetalKernelContract::kByteValidationSourceBufferIndex];
      [encoder setBuffer:status offset:0 atIndex:LlmMetalKernelContract::kByteValidationStatusBufferIndex];
      [encoder setBytes:&parameters
                 length:sizeof(parameters)
                atIndex:LlmMetalKernelContract::kByteValidationParametersBufferIndex];
      if (!dispatch_for_bytes(encoder, pipelines_.validate_bytes, resource.length_bytes)) {
        [encoder endEncoding];
        error = internal_error("invalid byte-validation dispatch geometry");
        return false;
      }
      if (!NumericUtils::checked_add(logical_bases[logical_base_index], resource.length_bytes,
                                     logical_bases[logical_base_index])) {
        [encoder endEncoding];
        error = internal_error("initialization logical-base overflow");
        return false;
      }
    }
    [encoder endEncoding];
    if (!commit_and_wait(command_buffer, error)) {
      return false;
    }
    if (mark_preparation_interrupted(error)) {
      return false;
    }
    const uint32_t status_word = *static_cast<const uint32_t*>(status.contents);
    if (status_word != 0 || hooks_.force_initialization_mismatch) {
      return false;
    }

    size_t data_resource_count = 0;
    for (const LlmMetalPlannedResource& resource : plan.planned_resources) {
      if (resource.pool == LlmMetalResourcePool::Weight || resource.pool == LlmMetalResourcePool::K ||
          resource.pool == LlmMetalResourcePool::V) {
        ++data_resource_count;
      }
    }
    size_t sample_bytes = 0;
    if (!NumericUtils::checked_multiply(data_resource_count, 2, sample_bytes) ||
        sample_bytes > static_cast<size_t>(status.length)) {
      error = internal_error("sample readback exceeds status buffer");
      return false;
    }
    id<MTLCommandBuffer> sample_command = [queue_ commandBuffer];
    if (sample_command == nil) {
      error = internal_error("sample readback commandBuffer returned nil");
      return false;
    }
    sample_command.label = @"membenchmark.llm-metal.command.sample-readback";
    id<MTLComputeCommandEncoder> sample_encoder =
        [sample_command computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    if (sample_encoder == nil) {
      error = internal_error("sample readback compute encoder returned nil");
      return false;
    }
    sample_encoder.label = @"membenchmark.llm-metal.encoder.sample-readback";
    [sample_encoder setComputePipelineState:pipelines_.copy];
    size_t sample_index = 0;
    for (size_t index = 0; index < plan.planned_resources.size(); ++index) {
      const LlmMetalPlannedResource& resource = plan.planned_resources[index];
      if (resource.pool != LlmMetalResourcePool::Weight && resource.pool != LlmMetalResourcePool::K &&
          resource.pool != LlmMetalResourcePool::V) {
        continue;
      }
      if (mark_preparation_interrupted(error)) {
        [sample_encoder endEncoding];
        return false;
      }
      id<MTLBuffer> buffer = candidate[static_cast<NSUInteger>(index)];
      LlmMetalFoundationParams parameters;
      parameters.byte_count = 1;
      parameters.destination_offset_bytes = sample_index * 2;
      [sample_encoder setBuffer:buffer offset:0 atIndex:LlmMetalKernelContract::kCopySourceBufferIndex];
      [sample_encoder setBuffer:status offset:0 atIndex:LlmMetalKernelContract::kCopyDestinationBufferIndex];
      [sample_encoder setBytes:&parameters
                        length:sizeof(parameters)
                       atIndex:LlmMetalKernelContract::kCopyParametersBufferIndex];
      if (!dispatch_for_bytes(sample_encoder, pipelines_.copy, parameters.byte_count)) {
        [sample_encoder endEncoding];
        error = internal_error("invalid first-byte sample dispatch geometry");
        return false;
      }
      parameters.source_offset_bytes = resource.length_bytes - 1;
      parameters.destination_offset_bytes = sample_index * 2 + 1;
      [sample_encoder setBytes:&parameters
                        length:sizeof(parameters)
                       atIndex:LlmMetalKernelContract::kCopyParametersBufferIndex];
      if (!dispatch_for_bytes(sample_encoder, pipelines_.copy, parameters.byte_count)) {
        [sample_encoder endEncoding];
        error = internal_error("invalid last-byte sample dispatch geometry");
        return false;
      }
      ++sample_index;
    }
    [sample_encoder endEncoding];
    if (!commit_and_wait(sample_command, error)) {
      return false;
    }
    if (mark_preparation_interrupted(error)) {
      return false;
    }

    const auto* samples = static_cast<const uint8_t*>(status.contents);
    logical_bases = {};
    sample_index = 0;
    for (const LlmMetalPlannedResource& resource : plan.planned_resources) {
      size_t logical_base_index = 0;
      if (resource.pool == LlmMetalResourcePool::K) {
        logical_base_index = 1;
      } else if (resource.pool == LlmMetalResourcePool::V) {
        logical_base_index = 2;
      } else if (resource.pool != LlmMetalResourcePool::Weight) {
        continue;
      }
      size_t last_absolute_byte = 0;
      if (!NumericUtils::checked_add(logical_bases[logical_base_index], resource.length_bytes - 1,
                                     last_absolute_byte) ||
          samples[sample_index * 2] != expected_data_byte(resource.pool, plan, logical_bases[logical_base_index]) ||
          samples[sample_index * 2 + 1] != expected_data_byte(resource.pool, plan, last_absolute_byte) ||
          !NumericUtils::checked_add(logical_bases[logical_base_index], resource.length_bytes,
                                     logical_bases[logical_base_index])) {
        error = internal_error("CPU sample readback mismatch");
        return false;
      }
      ++sample_index;
    }
    return sample_index == data_resource_count;
  }

  bool upload_and_validate_table(NSArray<id<MTLBuffer>>* candidate,
                                 const LlmMemoryWorkPlan& model_plan,
                                 const LlmMetalResourcePlan& plan,
                                 LlmMetalResourceEvidence& evidence,
                                 LlmMetalErrorDiagnostic& error) {
    if (!plan.paged_layout.has_value()) {
      paged_checksum_summary_ = LlmMetalPagedChecksumSummary{};
      evidence.table_upload_completed = true;
      evidence.table_validation_completed = true;
      return true;
    }
    id<MTLBuffer> status = find_buffer(candidate, plan.planned_resources, LlmMetalResourcePool::Status);
    id<MTLBuffer> staging = find_buffer(candidate, plan.planned_resources, LlmMetalResourcePool::Staging);
    if (status == nil || status.contents == nullptr || staging == nil || staging.contents == nullptr ||
        staging.length != plan.staging_buffer_length) {
      error = internal_error("table upload resources are invalid");
      return false;
    }
    MmapPtr host_table =
        allocate_buffer(plan.host_permutation_mapping_bytes, "LLM Metal paged KV host permutation table");
    if (host_table == nullptr) {
      error = internal_error("host permutation mapping failed");
      return false;
    }
    const LlmKvInPlaceBlockTableMaterialization materialization = materialize_llm_kv_block_table_in_place(
        *plan.paged_layout, derive_llm_kv_permutation_seed(base_seed_), static_cast<uint32_t*>(host_table.get()),
        plan.paged_layout->block_table_entries, Constants::LLM_KV_BLOCK_TABLE_HASH_CHUNK_ENTRIES,
        [this]() { return preparation_stop_requested(); });
    if (!materialization.valid) {
      preparation_interrupted_ = materialization.reason_code == LlmKvLayoutReason::PREPARATION_INTERRUPTED;
      error = internal_error(materialization.reason_code);
      return false;
    }
    evidence.table_permutation = materialization.permutation;

    uint32_t* const mutable_entries = static_cast<uint32_t*>(host_table.get());
    const LlmMetalPagedChecksumSummary checksum_summary =
        build_llm_metal_decode_paged_checksum_summary(
            model_plan, mutable_entries, plan.paged_layout->block_table_entries,
            [this]() { return preparation_stop_requested(); });
    if (!checksum_summary.valid) {
      preparation_interrupted_ = preparation_stop_requested();
      error = internal_error(preparation_interrupted_
                                 ? LlmBackendReason::PREPARATION_INTERRUPTED
                                 : "paged checksum summary failed");
      return false;
    }
    if (hooks_.force_wrong_paged_table_permutation) {
      size_t first = std::numeric_limits<size_t>::max();
      size_t second = std::numeric_limits<size_t>::max();
      for (size_t logical = 0;
           logical < plan.paged_layout->block_table_entries; ++logical) {
        if (logical % plan.paged_layout->blocks_per_sequence + 1 ==
            plan.paged_layout->blocks_per_sequence) {
          continue;
        }
        if (first == std::numeric_limits<size_t>::max()) {
          first = logical;
        } else {
          second = logical;
          break;
        }
      }
      if (second == std::numeric_limits<size_t>::max()) {
        error = internal_error("wrong paged-table permutation hook requires two non-terminal entries");
        return false;
      }
      std::swap(mutable_entries[first], mutable_entries[second]);
    }

    const uint8_t* const host_bytes = static_cast<const uint8_t*>(host_table.get());
    constexpr size_t kUploadChunkBytes = Constants::LLM_KV_PREPARATION_POLL_INTERVAL_ENTRIES * sizeof(uint32_t);
    size_t table_byte_offset = 0;
    for (size_t segment_index = 0; segment_index < plan.table_segments->segment_count; ++segment_index) {
      const size_t length = plan.table_segments->segment_lengths[segment_index];
      id<MTLBuffer> table =
          find_buffer(candidate, plan.planned_resources, LlmMetalResourcePool::BlockTable, segment_index);
      if (table == nil || table.length != length || std::min(length, kUploadChunkBytes) > staging.length) {
        error = internal_error("table segment identity mismatch");
        return false;
      }
      table_segment_first_bytes_[segment_index] = host_bytes[table_byte_offset];
      table_segment_first_byte_valid_[segment_index] = true;
      size_t segment_byte_offset = 0;
      while (segment_byte_offset < length) {
        @autoreleasepool {
          if (mark_preparation_interrupted(error)) {
            return false;
          }
          const size_t chunk_bytes = std::min(kUploadChunkBytes, length - segment_byte_offset);
          std::memcpy(staging.contents, host_bytes + table_byte_offset, chunk_bytes);
          std::memset(status.contents, 0, static_cast<size_t>(status.length));
          LlmMetalFoundationParams parameters;
          parameters.byte_count = chunk_bytes;
          parameters.destination_offset_bytes = segment_byte_offset;

          id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
          id<MTLComputeCommandEncoder> encoder =
              [command_buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
          if (command_buffer == nil || encoder == nil) {
            error = internal_error("table upload command creation failed");
            return false;
          }
          command_buffer.label = @"membenchmark.llm-metal.command.table-upload-validate";
          encoder.label = @"membenchmark.llm-metal.encoder.table-upload-validate";
          [encoder setComputePipelineState:pipelines_.copy];
          [encoder setBuffer:staging offset:0 atIndex:LlmMetalKernelContract::kCopySourceBufferIndex];
          [encoder setBuffer:table offset:0 atIndex:LlmMetalKernelContract::kCopyDestinationBufferIndex];
          [encoder setBytes:&parameters
                     length:sizeof(parameters)
                    atIndex:LlmMetalKernelContract::kCopyParametersBufferIndex];
          if (!dispatch_for_bytes(encoder, pipelines_.copy, chunk_bytes)) {
            [encoder endEncoding];
            error = internal_error("invalid table-copy dispatch geometry");
            return false;
          }
          [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
          [encoder setComputePipelineState:pipelines_.validate_table];
          [encoder setBuffer:staging offset:0 atIndex:LlmMetalKernelContract::kTableValidationReferenceBufferIndex];
          [encoder setBuffer:table offset:0 atIndex:LlmMetalKernelContract::kTableValidationCandidateBufferIndex];
          [encoder setBuffer:status offset:0 atIndex:LlmMetalKernelContract::kTableValidationStatusBufferIndex];
          [encoder setBytes:&parameters
                     length:sizeof(parameters)
                    atIndex:LlmMetalKernelContract::kTableValidationParametersBufferIndex];
          if (!dispatch_for_bytes(encoder, pipelines_.validate_table, chunk_bytes)) {
            [encoder endEncoding];
            error = internal_error("invalid table-validation dispatch geometry");
            return false;
          }
          [encoder endEncoding];
          if (!commit_and_wait(command_buffer, error) || *static_cast<const uint32_t*>(status.contents) != 0) {
            if (error.description.empty()) {
              error = internal_error("GPU table validation mismatch");
            }
            return false;
          }

          id<MTLCommandBuffer> readback = [queue_ commandBuffer];
          id<MTLBlitCommandEncoder> blit = [readback blitCommandEncoder];
          if (readback == nil || blit == nil) {
            error = internal_error("table readback command creation failed");
            return false;
          }
          [blit copyFromBuffer:table
                   sourceOffset:segment_byte_offset
                       toBuffer:staging
              destinationOffset:0
                           size:chunk_bytes];
          [blit endEncoding];
          if (!commit_and_wait(readback, error) ||
              std::memcmp(staging.contents, host_bytes + table_byte_offset, chunk_bytes) != 0) {
            if (error.description.empty()) {
              error = internal_error("table readback mismatch");
            }
            return false;
          }
          if (!NumericUtils::checked_add(segment_byte_offset, chunk_bytes, segment_byte_offset) ||
              !NumericUtils::checked_add(table_byte_offset, chunk_bytes, table_byte_offset)) {
            error = internal_error("table upload offset overflow");
            return false;
          }
        }
      }
    }
    if (mark_preparation_interrupted(error)) {
      return false;
    }
    if (table_byte_offset != plan.paged_layout->memory.block_table_bytes) {
      error = internal_error("table upload byte count mismatch");
      return false;
    }
    evidence.table_upload_completed = true;
    evidence.table_validation_completed = true;
    paged_checksum_summary_ = checksum_summary;
    return true;
  }

  bool run_decode_layout_probe(id<MTLBuffer> status, LlmMetalErrorDiagnostic& error) {
    if (active_phase_ == LlmPhase::Prefill) {
      if (status == nil || status.contents == nullptr ||
          status.length < sizeof(LlmMetalPrefillLayoutProbeWords)) {
        error = internal_error(
            "prefill-contiguous layout-probe output resource is invalid");
        return false;
      }
      const size_t probe_width =
          static_cast<size_t>(pipelines_.probe_decode.threadExecutionWidth);
      if (probe_width == 0 ||
          probe_width > static_cast<size_t>(
                            pipelines_.probe_decode
                                .maxTotalThreadsPerThreadgroup)) {
        error = internal_error(
            "invalid prefill-contiguous layout-probe dispatch geometry");
        return false;
      }
      LlmMetalPrefillContiguousParams parameters;
      parameters.weight_bytes = UINT64_C(0x0102030405060708);
      parameters.k_bytes = UINT64_C(0x1112131415161718);
      parameters.v_bytes = UINT64_C(0x2122232425262728);
      parameters.segment_capacity_bytes = UINT64_C(0x3132333435363738);
      parameters.prompt_tokens = UINT64_C(0x4142434445464748);
      parameters.attention_query_tile_tokens =
          UINT64_C(0x5152535455565758);
      parameters.tile_count = UINT64_C(0x6162636465666768);
      parameters.layer_count = UINT64_C(0x7172737475767778);
      parameters.batch_size = UINT64_C(0x8182838485868788);
      parameters.record_bytes = UINT64_C(0x9192939495969798);
      parameters.work_units = UINT64_C(0xA1A2A3A4A5A6A7A8);
      parameters.weight_seed = UINT64_C(0xB1B2B3B4B5B6B7B8);
      parameters.k_seed = UINT64_C(0xC1C2C3C4C5C6C7C8);
      parameters.v_seed = UINT64_C(0xD1D2D3D4D5D6D7D8);
      parameters.scenario_seed = UINT64_C(0xE1E2E3E4E5E6E7E8);
      parameters.weight_segment_count = UINT32_C(0x11121314);
      parameters.k_segment_count = UINT32_C(0x21222324);
      parameters.v_segment_count = UINT32_C(0x31323334);
      parameters.reserved_zero = UINT32_C(0x41424344);

      std::memset(status.contents, 0, static_cast<size_t>(status.length));
      id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> encoder =
          [command_buffer
              computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
      if (command_buffer == nil || encoder == nil) {
        error = internal_error(
            "prefill-contiguous layout-probe command creation failed");
        return false;
      }
      command_buffer.label =
          @"membenchmark.llm-metal.command.prefill-contiguous-layout-probe";
      encoder.label =
          @"membenchmark.llm-metal.encoder.prefill-contiguous-layout-probe";
      [encoder setComputePipelineState:pipelines_.probe_decode];
      [encoder setBytes:&parameters
                 length:sizeof(parameters)
                atIndex:LlmMetalKernelContract::kDecodeProbeParametersBufferIndex];
      [encoder setBuffer:status
                  offset:0
                 atIndex:LlmMetalKernelContract::kDecodeProbeOutputBufferIndex];
      [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(probe_width, 1, 1)];
      [encoder endEncoding];
      if (!commit_and_wait(command_buffer, error)) {
        return false;
      }
      LlmMetalPrefillLayoutProbeWords words{};
      std::memcpy(words.data(), status.contents, sizeof(words));
      if (!validate_llm_metal_prefill_layout_probe(parameters, words)) {
        error = internal_error(
            "prefill-contiguous parameter layout probe mismatch");
        return false;
      }
      return true;
    }
    if (active_kv_layout_ == LlmKvLayout::Paged) {
      if (status == nil || status.contents == nullptr ||
          status.length < sizeof(LlmMetalDecodePagedLayoutProbeWords)) {
        error = internal_error("decode-paged layout-probe output resource is invalid");
        return false;
      }
      const size_t probe_width =
          static_cast<size_t>(pipelines_.probe_decode.threadExecutionWidth);
      if (probe_width == 0 ||
          probe_width > static_cast<size_t>(
                            pipelines_.probe_decode.maxTotalThreadsPerThreadgroup)) {
        error = internal_error("invalid decode-paged layout-probe dispatch geometry");
        return false;
      }
      LlmMetalDecodePagedParams parameters;
      parameters.weight_bytes = UINT64_C(0x0102030405060708);
      parameters.context_tokens = UINT64_C(0x1112131415161718);
      parameters.layer_count = UINT64_C(0x2122232425262728);
      parameters.batch_size = UINT64_C(0x3132333435363738);
      parameters.record_bytes = UINT64_C(0x4142434445464748);
      parameters.work_units = UINT64_C(0x5152535455565758);
      parameters.block_bytes = UINT64_C(0x6162636465666768);
      parameters.last_block_valid_bytes = UINT64_C(0x7172737475767778);
      parameters.append_offset_in_last_block = UINT64_C(0x8182838485868788);
      parameters.blocks_per_sequence = UINT64_C(0x9192939495969798);
      parameters.physical_blocks_per_layer = UINT64_C(0xA1A2A3A4A5A6A7A8);
      parameters.blocks_per_segment = UINT64_C(0xB1B2B3B4B5B6B7B8);
      parameters.table_entries_per_segment = UINT64_C(0xC1C2C3C4C5C6C7C8);
      parameters.segment_capacity_bytes = UINT64_C(0xD1D2D3D4D5D6D7D8);
      parameters.weight_seed = UINT64_C(0xE1E2E3E4E5E6E7E8);
      parameters.k_seed = UINT64_C(0xF1F2F3F4F5F6F7F8);
      parameters.v_seed = UINT64_C(0x0101010102020202);
      parameters.scenario_seed = UINT64_C(0x0303030304040404);
      parameters.weight_segment_count = UINT32_C(0x11121314);
      parameters.k_segment_count = UINT32_C(0x21222324);
      parameters.v_segment_count = UINT32_C(0x31323334);
      parameters.table_segment_count = UINT32_C(0x41424344);
      parameters.reserved_zero = UINT32_C(0x51525354);
      parameters.padding_zero = UINT32_C(0x61626364);

      std::memset(status.contents, 0, static_cast<size_t>(status.length));
      id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> encoder =
          [command_buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
      if (command_buffer == nil || encoder == nil) {
        error = internal_error("decode-paged layout-probe command creation failed");
        return false;
      }
      command_buffer.label = @"membenchmark.llm-metal.command.decode-paged-layout-probe";
      encoder.label = @"membenchmark.llm-metal.encoder.decode-paged-layout-probe";
      [encoder setComputePipelineState:pipelines_.probe_decode];
      [encoder setBytes:&parameters
                 length:sizeof(parameters)
                atIndex:LlmMetalKernelContract::kDecodeProbeParametersBufferIndex];
      [encoder setBuffer:status
                  offset:0
                 atIndex:LlmMetalKernelContract::kDecodeProbeOutputBufferIndex];
      [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(probe_width, 1, 1)];
      [encoder endEncoding];
      if (!commit_and_wait(command_buffer, error)) {
        return false;
      }
      LlmMetalDecodePagedLayoutProbeWords words{};
      std::memcpy(words.data(), status.contents, sizeof(words));
      if (!validate_llm_metal_decode_paged_layout_probe(parameters, words)) {
        error = internal_error("decode-paged parameter layout probe mismatch");
        return false;
      }
      return true;
    }
    if (status == nil || status.contents == nullptr || status.length < sizeof(LlmMetalDecodeLayoutProbeWords)) {
      error = internal_error("decode layout-probe output resource is invalid");
      return false;
    }
    const size_t probe_width = static_cast<size_t>(pipelines_.probe_decode.threadExecutionWidth);
    if (probe_width == 0 ||
        probe_width > static_cast<size_t>(pipelines_.probe_decode.maxTotalThreadsPerThreadgroup)) {
      error = internal_error("invalid decode layout-probe dispatch geometry");
      return false;
    }
    LlmMetalDecodeContiguousParams parameters;
    parameters.weight_bytes = UINT64_C(0x0102030405060708);
    parameters.k_bytes = UINT64_C(0x1112131415161718);
    parameters.v_bytes = UINT64_C(0x2122232425262728);
    parameters.segment_capacity_bytes = UINT64_C(0x3132333435363738);
    parameters.context_tokens = UINT64_C(0x4142434445464748);
    parameters.layer_count = UINT64_C(0x5152535455565758);
    parameters.batch_size = UINT64_C(0x6162636465666768);
    parameters.record_bytes = UINT64_C(0x7172737475767778);
    parameters.work_units = UINT64_C(0x8182838485868788);
    parameters.weight_seed = UINT64_C(0x9192939495969798);
    parameters.k_seed = UINT64_C(0xA1A2A3A4A5A6A7A8);
    parameters.v_seed = UINT64_C(0xB1B2B3B4B5B6B7B8);
    parameters.scenario_seed = UINT64_C(0xC1C2C3C4C5C6C7C8);
    parameters.weight_segment_count = UINT32_C(0xD1D2D3D4);
    parameters.k_segment_count = UINT32_C(0xE1E2E3E4);
    parameters.v_segment_count = UINT32_C(0xF1F2F3F4);
    parameters.reserved_zero = UINT32_C(0xA5A6A7A8);

    std::memset(status.contents, 0, static_cast<size_t>(status.length));
    id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    if (command_buffer == nil || encoder == nil) {
      error = internal_error("decode layout-probe command creation failed");
      return false;
    }
    command_buffer.label = @"membenchmark.llm-metal.command.decode-contiguous-layout-probe";
    encoder.label = @"membenchmark.llm-metal.encoder.decode-contiguous-layout-probe";
    [encoder setComputePipelineState:pipelines_.probe_decode];
    [encoder setBytes:&parameters
               length:sizeof(parameters)
              atIndex:LlmMetalKernelContract::kDecodeProbeParametersBufferIndex];
    [encoder setBuffer:status offset:0 atIndex:LlmMetalKernelContract::kDecodeProbeOutputBufferIndex];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(probe_width, 1, 1)];
    [encoder endEncoding];
    if (!commit_and_wait(command_buffer, error)) {
      return false;
    }
    LlmMetalDecodeLayoutProbeWords words{};
    std::memcpy(words.data(), status.contents, sizeof(words));
    if (!validate_llm_metal_decode_layout_probe(parameters, words)) {
      error = internal_error("decode parameter layout probe mismatch");
      return false;
    }
    return true;
  }

  bool run_layout_probe(NSArray<id<MTLBuffer>>* candidate, const LlmMetalResourcePlan& plan,
                        LlmMetalCapabilityEvidence& capability, LlmMetalErrorDiagnostic& error) {
    id<MTLBuffer> argument = find_buffer(candidate, plan.planned_resources, LlmMetalResourcePool::ArgumentBuffer);
    id<MTLBuffer> status = find_buffer(candidate, plan.planned_resources, LlmMetalResourcePool::Status);
    constexpr size_t kProbeOutputOffset = 256;
    if (argument == nil || status == nil || status.contents == nullptr ||
        status.length < kProbeOutputOffset + sizeof(LlmMetalLayoutProbeWords)) {
      error = internal_error("layout-probe resources are invalid");
      return false;
    }
    capability.layout_probe_evaluated = true;
    if (!run_decode_layout_probe(status, error)) {
      return false;
    }
    const size_t probe_width = static_cast<size_t>(pipelines_.probe.threadExecutionWidth);
    if (probe_width == 0 || probe_width > static_cast<size_t>(pipelines_.probe.maxTotalThreadsPerThreadgroup)) {
      error = internal_error("invalid layout-probe dispatch geometry");
      return false;
    }

    struct ProbeCase {
      LlmMetalResourcePool pool;
      size_t pool_index;
      uint32_t resource_kind;
      uint64_t expected_value;
    };
    std::array<ProbeCase, 9> probe_cases{};
    size_t probe_case_count = 0;
    const auto append_case = [&](LlmMetalResourcePool pool, size_t pool_index, uint32_t resource_kind,
                                 uint64_t expected_value) {
      if (probe_case_count >= probe_cases.size()) {
        return false;
      }
      probe_cases[probe_case_count++] = {pool, pool_index, resource_kind, expected_value};
      return true;
    };
    const auto append_data_pool_cases = [&](LlmMetalResourcePool pool, uint32_t resource_kind,
                                            const LlmKvSegmentPlan& segments) {
      if (!segments.valid || segments.segment_count == 0 || segments.segment_lengths.size() != segments.segment_count ||
          !append_case(pool, 0, resource_kind, expected_data_byte(pool, plan, 0))) {
        return false;
      }
      if (segments.segment_count == 1) {
        return true;
      }
      size_t logical_base = 0;
      for (size_t index = 0; index + 1 < segments.segment_count; ++index) {
        if (!NumericUtils::checked_add(logical_base, segments.segment_lengths[index], logical_base)) {
          return false;
        }
      }
      return append_case(pool, segments.segment_count - 1, resource_kind, expected_data_byte(pool, plan, logical_base));
    };
    if (!append_data_pool_cases(LlmMetalResourcePool::Weight, LlmMetalKernelContract::kProbeWeightResourceKind,
                                plan.weight_segments) ||
        !append_data_pool_cases(LlmMetalResourcePool::K, LlmMetalKernelContract::kProbeKeyResourceKind,
                                plan.k_segments) ||
        !append_data_pool_cases(LlmMetalResourcePool::V, LlmMetalKernelContract::kProbeValueResourceKind,
                                plan.v_segments)) {
      error = internal_error("layout-probe segment plan is invalid");
      return false;
    }
    if (plan.table_segments.has_value()) {
      if (!plan.table_segments->valid || plan.table_segments->segment_count == 0 ||
          plan.table_segments->segment_count > table_segment_first_bytes_.size()) {
        error = internal_error("table probe segment plan is invalid");
        return false;
      }
      const size_t last_table_segment = plan.table_segments->segment_count - 1;
      if (!table_segment_first_byte_valid_[0] ||
          !append_case(LlmMetalResourcePool::BlockTable, 0, LlmMetalKernelContract::kProbeTableResourceKind,
                       table_segment_first_bytes_[0])) {
        error = internal_error("table probe value is unavailable");
        return false;
      }
      if (last_table_segment != 0 && (!table_segment_first_byte_valid_[last_table_segment] ||
                                      !append_case(LlmMetalResourcePool::BlockTable, last_table_segment,
                                                   LlmMetalKernelContract::kProbeTableResourceKind,
                                                   table_segment_first_bytes_[last_table_segment]))) {
        error = internal_error("table probe value is unavailable");
        return false;
      }
    }
    if (!append_case(LlmMetalResourcePool::Status, 0, LlmMetalKernelContract::kProbeStatusChecksumResourceKind, 0)) {
      error = internal_error("layout-probe case capacity exceeded");
      return false;
    }

    for (size_t case_index = 0; case_index < probe_case_count; ++case_index) {
      @autoreleasepool {
        const ProbeCase& probe_case = probe_cases[case_index];
        id<MTLBuffer> resource = find_buffer(candidate, plan.planned_resources, probe_case.pool, probe_case.pool_index);
        if (resource == nil) {
          error = internal_error("layout-probe resource is unavailable");
          return false;
        }
        std::memset(status.contents, 0, static_cast<size_t>(status.length));
        LlmMetalFoundationParams parameters;
        parameters.byte_count = UINT64_C(0x0102030405060708);
        parameters.source_offset_bytes = UINT64_C(0x1112131415161718);
        parameters.destination_offset_bytes = UINT64_C(0x2122232425262728);
        parameters.logical_base_bytes = UINT64_C(0x3132333435363738);
        parameters.pattern_seed = UINT64_C(0x4142434445464748);
        parameters.block_bytes = UINT64_C(0x5152535455565758);
        parameters.physical_blocks_per_layer = UINT32_C(0x61626364);
        parameters.pattern_kind = UINT32_C(0x71727374);
        parameters.probe_resource_kind = probe_case.resource_kind;
        parameters.probe_resource_slot = static_cast<uint32_t>(probe_case.pool_index);

        id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
        if (command_buffer == nil) {
          error = internal_error("layout-probe command creation failed");
          return false;
        }
        id<MTLComputeCommandEncoder> encoder =
            [command_buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
        if (encoder == nil) {
          error = internal_error("layout-probe encoder creation failed");
          return false;
        }
        command_buffer.label = @"membenchmark.llm-metal.command.layout-probe";
        encoder.label = @"membenchmark.llm-metal.encoder.layout-probe";
        [encoder setComputePipelineState:pipelines_.probe];
        [encoder setBuffer:argument offset:0 atIndex:LlmMetalKernelContract::kProbeResourcesBufferIndex];
        [encoder setBytes:&parameters
                   length:sizeof(parameters)
                  atIndex:LlmMetalKernelContract::kProbeParametersBufferIndex];
        [encoder setBuffer:status offset:kProbeOutputOffset atIndex:LlmMetalKernelContract::kProbeOutputBufferIndex];
        [encoder useResource:resource usage:MTLResourceUsageRead];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(probe_width, 1, 1)];
        [encoder endEncoding];
        if (!commit_and_wait(command_buffer, error)) {
          return false;
        }
        LlmMetalLayoutProbeWords words{};
        std::memcpy(words.data(), static_cast<const uint8_t*>(status.contents) + kProbeOutputOffset, sizeof(words));
        if (!validate_llm_metal_layout_probe(parameters, words, probe_case.expected_value)) {
          error = internal_error("parameter layout probe mismatch");
          return false;
        }
      }
    }
    capability.layout_probe_valid = !hooks_.force_layout_probe_mismatch;
    capability.layout_probe_resource_count = probe_case_count;
    return capability.layout_probe_valid;
  }

  id<MTLComputePipelineState> workload_pipeline(LlmScenario scenario) const noexcept {
    switch (scenario) {
      case LlmScenario::WeightsOnly:
        return pipelines_.decode_weights_only;
      case LlmScenario::KvOnly:
        return pipelines_.decode_kv_only;
      case LlmScenario::Mixed:
        return pipelines_.decode_mixed;
    }
    return nil;
  }

  const char* workload_pipeline_label(LlmScenario scenario) const noexcept {
    const bool paged = active_kv_layout_ == LlmKvLayout::Paged;
    const bool prefill = active_phase_ == LlmPhase::Prefill;
    switch (scenario) {
      case LlmScenario::WeightsOnly:
        return prefill
                   ? "membenchmark.llm-metal.pipeline.prefill-contiguous.weights-only"
               : paged
                   ? "membenchmark.llm-metal.pipeline.decode-paged.weights-only"
                   : "membenchmark.llm-metal.pipeline.decode-contiguous.weights-only";
      case LlmScenario::KvOnly:
        return prefill
                   ? "membenchmark.llm-metal.pipeline.prefill-contiguous.kv-only"
               : paged
                   ? "membenchmark.llm-metal.pipeline.decode-paged.kv-only"
                   : "membenchmark.llm-metal.pipeline.decode-contiguous.kv-only";
      case LlmScenario::Mixed:
        return prefill
                   ? "membenchmark.llm-metal.pipeline.prefill-contiguous.mixed"
               : paged
                   ? "membenchmark.llm-metal.pipeline.decode-paged.mixed"
                   : "membenchmark.llm-metal.pipeline.decode-contiguous.mixed";
    }
    return prefill ? "membenchmark.llm-metal.pipeline.prefill-contiguous.unknown"
           : paged ? "membenchmark.llm-metal.pipeline.decode-paged.unknown"
                   : "membenchmark.llm-metal.pipeline.decode-contiguous.unknown";
  }

  bool task_inputs_match(const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan& scenario_plan,
                         const LlmRunnerTaskContext& context) const noexcept {
    const LlmMetalExecutionPlan* execution = get_llm_metal_execution_plan(model_plan);
    return initialized_ && plan_resolved_ && resources_prepared_ &&
           evidence_.initialization.status == LlmBackendStatus::Ready &&
           evidence_.preparation.status == LlmBackendStatus::Ready && model_plan.valid &&
           model_plan.backend == LlmMemoryBackend::Metal &&
           model_plan.phase == active_phase_ &&
           model_plan.kv_layout == active_kv_layout_ &&
           model_plan.plan_identity == resolved_plan_identity_ &&
           execution != nullptr && execution->valid && execution->identity == resolved_execution_plan_.identity &&
           scenario_plan.valid && scenario_plan.reason_code == LlmWorkPlanReason::VALID &&
           scenario_plan.model_plan_identity == model_plan.plan_identity && scenario_plan.work_units != 0 &&
           scenario_plan.work_units <= resolved_execution_plan_.resources.limits.maximum_work_units_per_dispatch &&
           scenario_plan.work_unit_kind == model_plan.work_unit_kind &&
           context.scenario == scenario_plan.scenario;
  }

  bool build_decode_parameters(const LlmMemoryWorkPlan& model_plan, const LlmScenarioWorkPlan& scenario_plan,
                               LlmMetalDecodeContiguousParams& parameters) const noexcept {
    const LlmGeometry& geometry = model_plan.geometry;
    const LlmMetalResourcePlan& resources = resolved_execution_plan_.resources;
    if (!geometry.decode.has_value() || resources.weight_segments.segment_count > UINT32_MAX ||
        resources.k_segments.segment_count > UINT32_MAX || resources.v_segments.segment_count > UINT32_MAX) {
      return false;
    }
    parameters.weight_bytes = geometry.active_weight_bytes_per_work_unit;
    parameters.k_bytes = geometry.k_mapping_bytes;
    parameters.v_bytes = geometry.v_mapping_bytes;
    parameters.segment_capacity_bytes = resources.limits.segment_capacity_bytes;
    parameters.context_tokens = geometry.decode->visible_context_tokens;
    parameters.layer_count = geometry.layer_count;
    parameters.batch_size = geometry.batch_size;
    parameters.record_bytes = geometry.k_or_v_record_bytes_per_layer;
    parameters.work_units = scenario_plan.work_units;
    parameters.weight_seed = model_plan.weight_buffer_seed;
    parameters.k_seed = model_plan.k_buffer_seed;
    parameters.v_seed = model_plan.v_buffer_seed;
    parameters.scenario_seed = scenario_plan.scenario_seed;
    parameters.weight_segment_count = static_cast<uint32_t>(resources.weight_segments.segment_count);
    parameters.k_segment_count = static_cast<uint32_t>(resources.k_segments.segment_count);
    parameters.v_segment_count = static_cast<uint32_t>(resources.v_segments.segment_count);
    return parameters.weight_bytes != 0 && parameters.k_bytes != 0 && parameters.v_bytes != 0 &&
           parameters.segment_capacity_bytes != 0 && parameters.context_tokens != 0 && parameters.layer_count != 0 &&
           parameters.batch_size != 0 && parameters.record_bytes != 0 && parameters.work_units != 0;
  }

  bool build_decode_paged_parameters(
      const LlmMemoryWorkPlan& model_plan,
      const LlmScenarioWorkPlan& scenario_plan,
      LlmMetalDecodePagedParams& parameters) const noexcept {
    const LlmGeometry& geometry = model_plan.geometry;
    const LlmMetalResourcePlan& resources = resolved_execution_plan_.resources;
    if (!geometry.decode.has_value() || !resources.paged_layout.has_value() ||
        !resources.table_segments.has_value() ||
        resources.weight_segments.segment_count > UINT32_MAX ||
        resources.k_segments.segment_count > UINT32_MAX ||
        resources.v_segments.segment_count > UINT32_MAX ||
        resources.table_segments->segment_count > UINT32_MAX) {
      return false;
    }
    const LlmKvLayoutPlan& layout = *resources.paged_layout;
    parameters.weight_bytes = geometry.active_weight_bytes_per_work_unit;
    parameters.context_tokens = geometry.decode->visible_context_tokens;
    parameters.layer_count = geometry.layer_count;
    parameters.batch_size = geometry.batch_size;
    parameters.record_bytes = geometry.k_or_v_record_bytes_per_layer;
    parameters.work_units = scenario_plan.work_units;
    parameters.block_bytes = layout.block_bytes;
    parameters.last_block_valid_bytes = layout.last_block_valid_bytes;
    parameters.append_offset_in_last_block =
        layout.decode_append_offset_in_last_block;
    parameters.blocks_per_sequence = layout.blocks_per_sequence;
    parameters.physical_blocks_per_layer = layout.physical_blocks_per_layer;
    parameters.blocks_per_segment = resources.k_segments.elements_per_segment;
    parameters.table_entries_per_segment =
        resources.table_segments->elements_per_segment;
    parameters.segment_capacity_bytes = resources.limits.segment_capacity_bytes;
    parameters.weight_seed = model_plan.weight_buffer_seed;
    parameters.k_seed = model_plan.k_buffer_seed;
    parameters.v_seed = model_plan.v_buffer_seed;
    parameters.scenario_seed = scenario_plan.scenario_seed;
    parameters.weight_segment_count =
        static_cast<uint32_t>(resources.weight_segments.segment_count);
    parameters.k_segment_count =
        static_cast<uint32_t>(resources.k_segments.segment_count);
    parameters.v_segment_count =
        static_cast<uint32_t>(resources.v_segments.segment_count);
    parameters.table_segment_count =
        static_cast<uint32_t>(resources.table_segments->segment_count);
    return parameters.weight_bytes != 0 && parameters.context_tokens != 0 &&
           parameters.layer_count != 0 && parameters.batch_size != 0 &&
           parameters.record_bytes != 0 && parameters.work_units != 0 &&
           parameters.block_bytes != 0 &&
           parameters.last_block_valid_bytes != 0 &&
           parameters.last_block_valid_bytes <= parameters.block_bytes &&
           parameters.append_offset_in_last_block + parameters.record_bytes ==
               parameters.last_block_valid_bytes &&
           parameters.blocks_per_sequence != 0 &&
           parameters.physical_blocks_per_layer != 0 &&
           parameters.blocks_per_segment != 0 &&
           parameters.table_entries_per_segment != 0;
  }

  bool build_prefill_parameters(
      const LlmMemoryWorkPlan& model_plan,
      const LlmScenarioWorkPlan& scenario_plan,
      LlmMetalPrefillContiguousParams& parameters) const noexcept {
    const LlmGeometry& geometry = model_plan.geometry;
    const LlmMetalResourcePlan& resources = resolved_execution_plan_.resources;
    if (!geometry.prefill.has_value() ||
        !model_plan.prefill_plan.has_value() ||
        !model_plan.prefill_plan->valid ||
        resources.weight_segments.segment_count > UINT32_MAX ||
        resources.k_segments.segment_count > UINT32_MAX ||
        resources.v_segments.segment_count > UINT32_MAX) {
      return false;
    }
    const LlmPrefillGeometry& prefill = *geometry.prefill;
    parameters.weight_bytes = geometry.active_weight_bytes_per_work_unit;
    parameters.k_bytes = geometry.k_mapping_bytes;
    parameters.v_bytes = geometry.v_mapping_bytes;
    parameters.segment_capacity_bytes =
        resources.limits.segment_capacity_bytes;
    parameters.prompt_tokens = prefill.prompt_tokens;
    parameters.attention_query_tile_tokens =
        prefill.attention_query_tile_tokens;
    parameters.tile_count = prefill.tile_count;
    parameters.layer_count = geometry.layer_count;
    parameters.batch_size = geometry.batch_size;
    parameters.record_bytes = geometry.k_or_v_record_bytes_per_layer;
    parameters.work_units = scenario_plan.work_units;
    parameters.weight_seed = model_plan.weight_buffer_seed;
    parameters.k_seed = model_plan.k_buffer_seed;
    parameters.v_seed = model_plan.v_buffer_seed;
    parameters.scenario_seed = scenario_plan.scenario_seed;
    parameters.weight_segment_count =
        static_cast<uint32_t>(resources.weight_segments.segment_count);
    parameters.k_segment_count =
        static_cast<uint32_t>(resources.k_segments.segment_count);
    parameters.v_segment_count =
        static_cast<uint32_t>(resources.v_segments.segment_count);
    return parameters.weight_bytes != 0 && parameters.k_bytes != 0 &&
           parameters.v_bytes != 0 &&
           parameters.segment_capacity_bytes != 0 &&
           parameters.prompt_tokens != 0 &&
           parameters.attention_query_tile_tokens != 0 &&
           parameters.attention_query_tile_tokens <=
               parameters.prompt_tokens &&
           parameters.tile_count != 0 && parameters.layer_count != 0 &&
           parameters.batch_size != 0 && parameters.record_bytes != 0 &&
           parameters.work_units != 0;
  }

  LlmMetalGridPlan build_task_grid(const LlmMemoryWorkPlan& model_plan,
                                   const LlmScenarioWorkPlan& scenario_plan,
                                   id<MTLComputePipelineState> pipeline) const {
    const LlmGeometry& geometry = model_plan.geometry;
    if (model_plan.kv_layout == LlmKvLayout::Paged &&
        scenario_plan.scenario != LlmScenario::WeightsOnly) {
      const auto& resources = resolved_execution_plan_.resources;
      if (!resources.paged_layout.has_value()) {
        return {};
      }
      const LlmKvLayoutPlan& layout = *resources.paged_layout;
      size_t owner_count = 0;
      if (!NumericUtils::checked_multiply(layout.layer_count,
                                          layout.batch_size, owner_count) ||
          !NumericUtils::checked_multiply(owner_count,
                                          layout.blocks_per_sequence,
                                          owner_count)) {
        LlmMetalGridPlan failed;
        failed.reason_code = LlmMetalPlanReason::OWNER_COUNT_OVERFLOW;
        return failed;
      }
      LlmMetalGridRequest request;
      request.owner_count = owner_count;
      const size_t maximum_weight_layer_bytes =
          geometry.active_weight_bytes_per_work_unit / geometry.layer_count +
          (geometry.active_weight_bytes_per_work_unit % geometry.layer_count != 0
               ? 1U
               : 0U);
      request.visit_bytes =
          scenario_plan.scenario == LlmScenario::Mixed
              ? std::max(layout.block_bytes, maximum_weight_layer_bytes)
              : layout.block_bytes;
      request.work_units = scenario_plan.work_units;
      request.paged_semantic_lookups =
          scenario_plan.layout_metadata_lookup_count;
      request.pipeline = {
          static_cast<size_t>(pipeline.threadExecutionWidth),
          static_cast<size_t>(pipeline.maxTotalThreadsPerThreadgroup)};
      request.limits = resources.limits;
      LlmMetalGridPlan grid = build_llm_metal_grid_plan(request);
      if (!grid.valid || grid.actual_threadgroups == 0) {
        return grid;
      }
      try {
        grid.threadgroup_accounted_bytes.assign(grid.actual_threadgroups, 0);
      } catch (const std::bad_alloc&) {
        grid.valid = false;
        grid.reason_code = LlmMetalPlanReason::PLANNER_ALLOCATION_FAILED;
        return grid;
      } catch (const std::length_error&) {
        grid.valid = false;
        grid.reason_code = LlmMetalPlanReason::PLANNER_ALLOCATION_FAILED;
        return grid;
      }
      const size_t weight_layer_base =
          geometry.active_weight_bytes_per_work_unit / geometry.layer_count;
      const size_t weight_layer_remainder =
          geometry.active_weight_bytes_per_work_unit % geometry.layer_count;
      size_t accounted_sum = 0;
      for (size_t owner = 0; owner < owner_count; ++owner) {
        const size_t logical_block = owner % layout.blocks_per_sequence;
        const size_t sequence = owner / layout.blocks_per_sequence;
        const size_t batch = sequence % layout.batch_size;
        const size_t layer = sequence / layout.batch_size;
        const bool terminal = logical_block + 1 == layout.blocks_per_sequence;
        const size_t valid_bytes = terminal ? layout.last_block_valid_bytes
                                            : layout.block_bytes;
        size_t owner_bytes_per_work_unit = 0;
        size_t append_bytes = 0;
        if (!NumericUtils::checked_multiply(valid_bytes, 2,
                                            owner_bytes_per_work_unit) ||
            (terminal &&
             (!NumericUtils::checked_multiply(
                  layout.k_or_v_record_bytes_per_layer, 2, append_bytes) ||
              !NumericUtils::checked_add(owner_bytes_per_work_unit,
                                         append_bytes,
                                         owner_bytes_per_work_unit))) ||
            !NumericUtils::checked_add(owner_bytes_per_work_unit,
                                       terminal ? 3 * sizeof(uint32_t)
                                                : 2 * sizeof(uint32_t),
                                       owner_bytes_per_work_unit)) {
          grid.valid = false;
          grid.reason_code = LlmMetalPlanReason::OWNER_COUNT_OVERFLOW;
          return grid;
        }
        if (scenario_plan.scenario == LlmScenario::Mixed && batch == 0 &&
            logical_block == 0 &&
            !NumericUtils::checked_add(
                owner_bytes_per_work_unit,
                weight_layer_base + (layer < weight_layer_remainder ? 1U : 0U),
                owner_bytes_per_work_unit)) {
          grid.valid = false;
          grid.reason_code = LlmMetalPlanReason::OWNER_COUNT_OVERFLOW;
          return grid;
        }
        size_t owner_bytes = 0;
        const size_t threadgroup = owner % grid.actual_threadgroups;
        if (!NumericUtils::checked_multiply(owner_bytes_per_work_unit,
                                            scenario_plan.work_units,
                                            owner_bytes) ||
            !NumericUtils::checked_add(
                grid.threadgroup_accounted_bytes[threadgroup], owner_bytes,
                grid.threadgroup_accounted_bytes[threadgroup]) ||
            !NumericUtils::checked_add(accounted_sum, owner_bytes,
                                       accounted_sum)) {
          grid.valid = false;
          grid.reason_code = LlmMetalPlanReason::OWNER_COUNT_OVERFLOW;
          return grid;
        }
      }
      if (accounted_sum != scenario_plan.task_accounted_bytes) {
        grid.valid = false;
        grid.reason_code = LlmMetalPlanReason::OWNER_COST_COUNT_MISMATCH;
        return grid;
      }
      const auto [minimum, maximum] = std::minmax_element(
          grid.threadgroup_accounted_bytes.begin(),
          grid.threadgroup_accounted_bytes.end());
      grid.minimum_threadgroup_accounted_bytes = *minimum;
      grid.maximum_threadgroup_accounted_bytes = *maximum;
      grid.threadgroup_accounted_imbalance_bytes = *maximum - *minimum;
      append_identity_field(grid.identity, "paged_owner_cost_version",
                            "llm-metal-paged-owner-cost-v1");
      append_identity_field(grid.identity,
                            "paged_minimum_threadgroup_accounted_bytes",
                            grid.minimum_threadgroup_accounted_bytes);
      append_identity_field(grid.identity,
                            "paged_maximum_threadgroup_accounted_bytes",
                            grid.maximum_threadgroup_accounted_bytes);
      append_identity_field(grid.identity,
                            "paged_threadgroup_accounted_imbalance_bytes",
                            grid.threadgroup_accounted_imbalance_bytes);
      for (size_t bytes : grid.threadgroup_accounted_bytes) {
        append_identity_field(grid.identity,
                              "paged_threadgroup_accounted_bytes", bytes);
      }
      return grid;
    }
    size_t maximum_visit_bytes = 0;
    size_t serial_range_visits_per_lane = 0;
    if (model_plan.phase == LlmPhase::Prefill &&
        !calculate_llm_metal_prefill_serial_range_visits_per_lane(
            model_plan, scenario_plan, serial_range_visits_per_lane)) {
      LlmMetalGridPlan failed;
      failed.reason_code =
          LlmMetalPlanReason::SERIAL_RANGE_VISIT_COUNT_OVERFLOW;
      return failed;
    }
    if (scenario_plan.scenario != LlmScenario::KvOnly) {
      const size_t weight_base = geometry.active_weight_bytes_per_work_unit / geometry.layer_count;
      maximum_visit_bytes = weight_base +
                            (geometry.active_weight_bytes_per_work_unit % geometry.layer_count != 0 ? 1U : 0U);
    }
    if (scenario_plan.scenario != LlmScenario::WeightsOnly) {
      size_t sequence_bytes = 0;
      const size_t sequence_tokens =
          geometry.phase == LlmPhase::Prefill
              ? geometry.prefill->prompt_tokens
              : geometry.decode->visible_context_tokens;
      if (!NumericUtils::checked_multiply(sequence_tokens,
                                          geometry.k_or_v_record_bytes_per_layer, sequence_bytes)) {
        return {};
      }
      maximum_visit_bytes = std::max(maximum_visit_bytes, sequence_bytes);
    }
    if (model_plan.phase == LlmPhase::Prefill &&
        !calculate_llm_metal_prefill_maximum_range_vector_span_bytes(
            geometry, scenario_plan.scenario, maximum_visit_bytes)) {
      LlmMetalGridPlan failed;
      failed.reason_code = LlmMetalPlanReason::OWNER_COUNT_OVERFLOW;
      return failed;
    }
    constexpr size_t kTargetOwnerBytes = Constants::BYTES_PER_MB;
    size_t owner_count = 0;
    if (maximum_visit_bytes == 0 || !checked_ceil_divide(maximum_visit_bytes, kTargetOwnerBytes, owner_count)) {
      return {};
    }
    owner_count = std::max<size_t>(owner_count, 1);
    const size_t predicted_threadgroups =
        std::min(owner_count, resolved_execution_plan_.resources.limits.maximum_threadgroups_per_grid);
    size_t visit_bytes_per_owner = 0;
    if (!checked_ceil_divide(maximum_visit_bytes, predicted_threadgroups, visit_bytes_per_owner)) {
      return {};
    }
    LlmMetalGridRequest request;
    request.owner_count = model_plan.phase == LlmPhase::Prefill
                              ? predicted_threadgroups
                              : owner_count;
    request.visit_bytes = visit_bytes_per_owner;
    request.work_units = scenario_plan.work_units;
    request.serial_range_visits_per_lane = serial_range_visits_per_lane;
    request.pipeline = {static_cast<size_t>(pipeline.threadExecutionWidth),
                        static_cast<size_t>(pipeline.maxTotalThreadsPerThreadgroup)};
    request.limits = resolved_execution_plan_.resources.limits;
    return build_llm_metal_grid_plan(request);
  }

  void declare_task_residency(id<MTLComputeCommandEncoder> encoder, LlmScenario scenario,
                              id<MTLBuffer> argument, id<MTLBuffer> status) const {
    [encoder useResource:argument usage:MTLResourceUsageRead];
    [encoder useResource:status usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    const bool use_weight = scenario != LlmScenario::KvOnly;
    const bool use_kv = scenario != LlmScenario::WeightsOnly;
    const LlmMetalResourcePlan& plan = resolved_execution_plan_.resources;
    for (const LlmMetalPlannedResource& resource : plan.planned_resources) {
      const bool active = (resource.pool == LlmMetalResourcePool::Weight && use_weight) ||
                          ((resource.pool == LlmMetalResourcePool::K || resource.pool == LlmMetalResourcePool::V) &&
                           use_kv) ||
                          (resource.pool == LlmMetalResourcePool::BlockTable &&
                           use_kv && active_kv_layout_ == LlmKvLayout::Paged);
      if (!active) {
        continue;
      }
      id<MTLBuffer> buffer = find_buffer(owned_buffers_, plan.planned_resources, resource.pool, resource.pool_index);
      const MTLResourceUsage usage =
          resource.pool == LlmMetalResourcePool::Weight ||
                  resource.pool == LlmMetalResourcePool::BlockTable
              ? MTLResourceUsageRead
              : MTLResourceUsageRead | MTLResourceUsageWrite;
      [encoder useResource:buffer usage:usage];
    }
  }

  bool reset_task_status(id<MTLBuffer> status, LlmMetalTaskEvidence& task, LlmMetalErrorDiagnostic& error) const {
    id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
    id<MTLBlitCommandEncoder> encoder = [command_buffer blitCommandEncoder];
    if (command_buffer == nil || encoder == nil) {
      error = internal_error("status reset command creation failed");
      return false;
    }
    ++task.reset_command_buffer_count;
    command_buffer.label = @"membenchmark.llm-metal.command.task-status-reset";
    encoder.label = @"membenchmark.llm-metal.encoder.task-status-reset";
    [encoder fillBuffer:status range:NSMakeRange(0, static_cast<NSUInteger>(status.length)) value:0];
    [encoder endEncoding];
    const bool completed = commit_and_wait(command_buffer, error);
    task.reset_command_status = completed ? "completed" : "error";
    return completed;
  }

  bool execute_timed_command(id<MTLComputePipelineState> pipeline, id<MTLBuffer> argument, id<MTLBuffer> status,
                             const void* parameters, size_t parameter_bytes,
                             const LlmMetalGridPlan& grid,
                             LlmScenario scenario, LlmMetalTaskEvidence& task,
                             LlmMetalErrorDiagnostic& error) const {
    id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    if (command_buffer == nil || encoder == nil) {
      error = internal_error("timed command creation failed");
      return false;
    }
    ++task.timed_command_buffer_count;
    ++task.timed_compute_encoder_count;
    const bool paged = active_kv_layout_ == LlmKvLayout::Paged;
    const bool prefill = active_phase_ == LlmPhase::Prefill;
    command_buffer.label = prefill
                               ? @"membenchmark.llm-metal.command.prefill-contiguous.timed"
                           : paged
                               ? @"membenchmark.llm-metal.command.decode-paged.timed"
                               : @"membenchmark.llm-metal.command.decode-contiguous.timed";
    encoder.label = prefill
                        ? @"membenchmark.llm-metal.encoder.prefill-contiguous.timed"
                    : paged
                        ? @"membenchmark.llm-metal.encoder.decode-paged.timed"
                        : @"membenchmark.llm-metal.encoder.decode-contiguous.timed";
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:argument offset:0 atIndex:LlmMetalKernelContract::kWorkloadResourcesBufferIndex];
    [encoder setBytes:parameters
               length:parameter_bytes
              atIndex:LlmMetalKernelContract::kWorkloadParametersBufferIndex];
    size_t reduction_bytes = 0;
    if (!NumericUtils::checked_multiply(grid.threads_per_threadgroup,
                                        static_cast<size_t>(LlmMetalKernelContract::kReductionLaneCount) *
                                            sizeof(uint32_t),
                                        reduction_bytes)) {
      [encoder endEncoding];
      error = internal_error("threadgroup reduction length overflow");
      return false;
    }
    [encoder setThreadgroupMemoryLength:reduction_bytes
                                atIndex:LlmMetalKernelContract::kWorkloadReductionThreadgroupIndex];
    if (paged && scenario != LlmScenario::WeightsOnly) {
      [encoder setThreadgroupMemoryLength:sizeof(uint32_t)
                                  atIndex:LlmMetalKernelContract::kWorkloadPhysicalIdThreadgroupIndex];
    }
    declare_task_residency(encoder, scenario, argument, status);
    [encoder dispatchThreadgroups:MTLSizeMake(grid.actual_threadgroups, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(grid.threads_per_threadgroup, 1, 1)];
    ++task.timed_workload_dispatch_count;
    [encoder endEncoding];

    using SteadyClock = std::chrono::steady_clock;
    const auto submit_start = SteadyClock::now();
    [command_buffer commit];
    const auto wait_start = SteadyClock::now();
    [command_buffer waitUntilCompleted];
    const auto completion = SteadyClock::now();
    task.host_timing_evaluated = true;
    task.host_submit_to_completion_seconds = std::chrono::duration<double>(completion - submit_start).count();
    task.host_wait_seconds = std::chrono::duration<double>(completion - wait_start).count();
    if (command_buffer.status != MTLCommandBufferStatusCompleted || hooks_.force_timed_command_failure) {
      task.timed_command_status = "error";
      error = hooks_.force_timed_command_failure ? internal_error("injected timed command failure")
                                                 : error_diagnostic(command_buffer.error);
      return false;
    }
    task.timed_command_status = "completed";
    task.timing_evaluated = true;
    task.gpu_start_seconds = command_buffer.GPUStartTime;
    task.gpu_end_seconds = command_buffer.GPUEndTime;
    if (hooks_.force_invalid_gpu_timestamps) {
      task.gpu_start_seconds = 0.0;
      task.gpu_end_seconds = 0.0;
    }
    task.gpu_elapsed_seconds = task.gpu_end_seconds - task.gpu_start_seconds;
    task.timing_valid = std::isfinite(task.gpu_start_seconds) && std::isfinite(task.gpu_end_seconds) &&
                        std::isfinite(task.gpu_elapsed_seconds) && task.gpu_start_seconds > 0.0 &&
                        task.gpu_end_seconds > task.gpu_start_seconds && task.gpu_elapsed_seconds > 0.0;
    return true;
  }

  bool run_task_post_validation(id<MTLBuffer> argument, id<MTLBuffer> status,
                                const void* parameters, size_t parameter_bytes,
                                const LlmMetalGridPlan& grid,
                                LlmScenario scenario, LlmMetalTaskEvidence& task,
                                LlmMetalErrorDiagnostic& error) const {
    id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
    id<MTLBlitCommandEncoder> reset_encoder = [command_buffer blitCommandEncoder];
    if (command_buffer == nil || reset_encoder == nil) {
      error = internal_error("post-validation command creation failed");
      return false;
    }
    ++task.post_validation_command_buffer_count;
    const bool paged = active_kv_layout_ == LlmKvLayout::Paged;
    const bool prefill = active_phase_ == LlmPhase::Prefill;
    command_buffer.label = prefill
                               ? @"membenchmark.llm-metal.command.prefill-contiguous.post-validation"
                           : paged
                               ? @"membenchmark.llm-metal.command.decode-paged.post-validation"
                               : @"membenchmark.llm-metal.command.decode-contiguous.post-validation";
    [reset_encoder fillBuffer:status range:NSMakeRange(0, sizeof(uint32_t)) value:0];
    [reset_encoder endEncoding];
    const bool validate_kv_writes = scenario != LlmScenario::WeightsOnly;
    if (validate_kv_writes) {
      id<MTLComputeCommandEncoder> encoder =
          [command_buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
      if (encoder == nil) {
        error = internal_error("post-validation compute encoder creation failed");
        return false;
      }
      encoder.label = prefill
                          ? @"membenchmark.llm-metal.encoder.prefill-contiguous.post-validation"
                      : paged
                          ? @"membenchmark.llm-metal.encoder.decode-paged.post-validation"
                          : @"membenchmark.llm-metal.encoder.decode-contiguous.post-validation";
      [encoder setComputePipelineState:pipelines_.validate_decode_appends];
      [encoder setBuffer:argument offset:0 atIndex:LlmMetalKernelContract::kPostValidationResourcesBufferIndex];
      [encoder setBytes:parameters
                 length:parameter_bytes
                atIndex:LlmMetalKernelContract::kPostValidationParametersBufferIndex];
      declare_task_residency(encoder, scenario, argument, status);
      [encoder dispatchThreadgroups:MTLSizeMake(grid.actual_threadgroups, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(grid.threads_per_threadgroup, 1, 1)];
      [encoder endEncoding];
    }
    if (!commit_and_wait(command_buffer, error) || hooks_.force_post_validation_command_failure) {
      task.post_validation_command_status = "error";
      if (hooks_.force_post_validation_command_failure) {
        error = internal_error("injected post-validation command failure");
      }
      return false;
    }
    task.post_validation_command_status = "completed";
    task.post_validation_evaluated = true;
    task.kv_write_validation_evaluated = validate_kv_writes;
    const uint32_t flags = *static_cast<const uint32_t*>(status.contents);
    task.kv_write_validation_valid =
        !validate_kv_writes ||
        ((flags &
          (LlmMetalKernelContract::kKvWriteValidationMismatchBit |
           LlmMetalKernelContract::kValidationInvalidParametersBit)) == 0 &&
         !hooks_.force_kv_write_validation_mismatch);
    const bool padding_applicable =
        paged && validate_kv_writes &&
        resolved_execution_plan_.resources.paged_layout.has_value() &&
        resolved_execution_plan_.resources.paged_layout->last_block_valid_bytes <
            resolved_execution_plan_.resources.paged_layout->block_bytes;
    task.padding_canary_applicable = padding_applicable;
    task.padding_canary_evaluated = padding_applicable;
    task.padding_canary_valid =
        padding_applicable &&
        (flags & (LlmMetalKernelContract::kPaddingCanaryMismatchBit |
                  LlmMetalKernelContract::kValidationInvalidParametersBit)) == 0 &&
        !hooks_.force_padding_canary_mismatch;
    task.post_validation_valid =
        task.kv_write_validation_valid &&
        (!padding_applicable || task.padding_canary_valid);
    return true;
  }

  LlmTaskExecutionResult execute_task_impl(const LlmMemoryWorkPlan& model_plan,
                                           const LlmScenarioWorkPlan& scenario_plan,
                                           const LlmRunnerTaskContext& context) {
    LlmTaskExecutionResult result{LlmTaskExecutionStatus::NotStarted, LlmBackendReason::NOT_INITIALIZED};
    result.identity = metal_task_identity(model_plan, scenario_plan, context);
    result.completion.planned_work_units = scenario_plan.work_units;
    LlmMetalTaskEvidence task;
    const bool supported_profile =
        (model_plan.phase == LlmPhase::Decode &&
         (model_plan.kv_layout == LlmKvLayout::Contiguous ||
          model_plan.kv_layout == LlmKvLayout::Paged)) ||
        (model_plan.phase == LlmPhase::Prefill &&
         model_plan.kv_layout == LlmKvLayout::Contiguous);
    if (model_plan.backend == LlmMemoryBackend::Metal &&
        !supported_profile) {
      result.status = LlmTaskExecutionStatus::Unsupported;
      result.reason_code = LlmBackendReason::TASK_UNSUPPORTED;
      result.backend_evidence = std::move(task);
      return result;
    }
    if (!task_inputs_match(model_plan, scenario_plan, context)) {
      result.status = LlmTaskExecutionStatus::Failed;
      result.reason_code = LlmBackendReason::TASK_IDENTITY_MISMATCH;
      result.backend_evidence = std::move(task);
      return result;
    }
    id<MTLComputePipelineState> pipeline = workload_pipeline(scenario_plan.scenario);
    task.timed_pipeline_available = pipeline != nil;
    task.pipeline_label = workload_pipeline_label(scenario_plan.scenario);
    if (pipeline != nil) {
      task.pipeline_thread_execution_width = static_cast<size_t>(pipeline.threadExecutionWidth);
      task.pipeline_max_total_threads_per_threadgroup =
          static_cast<size_t>(pipeline.maxTotalThreadsPerThreadgroup);
    }
    LlmMetalDecodeContiguousParams contiguous_parameters;
    LlmMetalDecodePagedParams paged_parameters;
    LlmMetalPrefillContiguousParams prefill_parameters;
    const bool paged = model_plan.kv_layout == LlmKvLayout::Paged;
    const bool prefill = model_plan.phase == LlmPhase::Prefill;
    const bool parameters_valid =
        prefill
            ? build_prefill_parameters(model_plan, scenario_plan,
                                       prefill_parameters)
            : paged ? build_decode_paged_parameters(model_plan, scenario_plan,
                                                     paged_parameters)
                    : build_decode_parameters(model_plan, scenario_plan,
                                              contiguous_parameters);
    const void* const parameters =
        prefill ? static_cast<const void*>(&prefill_parameters)
        : paged ? static_cast<const void*>(&paged_parameters)
                : static_cast<const void*>(&contiguous_parameters);
    const size_t parameter_bytes =
        prefill ? sizeof(prefill_parameters)
        : paged ? sizeof(paged_parameters) : sizeof(contiguous_parameters);
    const LlmMetalGridPlan grid = build_task_grid(model_plan, scenario_plan, pipeline);
    task.grid_plan = grid;
    task.grid_plan_available = grid.valid;
    if (pipeline == nil || !parameters_valid || !grid.valid ||
        grid.actual_threadgroups == 0 || grid.threads_per_threadgroup == 0) {
      result.status = LlmTaskExecutionStatus::Failed;
      result.reason_code = !grid.reason_code.empty() && grid.reason_code != LlmMetalPlanReason::VALID
                               ? grid.reason_code
                               : LlmBackendReason::EXECUTION_PLAN_MISMATCH;
      result.backend_evidence = std::move(task);
      return result;
    }
    const LlmMetalChecksumOracle oracle =
        prefill
            ? calculate_llm_metal_prefill_contiguous_checksum(model_plan,
                                                              scenario_plan)
            : paged ? calculate_llm_metal_decode_paged_checksum(
                          model_plan, scenario_plan, paged_checksum_summary_)
                    : calculate_llm_metal_decode_contiguous_checksum(
                          model_plan, scenario_plan);
    if (!oracle.valid) {
      result.status = LlmTaskExecutionStatus::Failed;
      result.reason_code =
          !oracle.reason_code.empty() &&
                  oracle.reason_code != LlmMetalPlanReason::INVALID_GEOMETRY
              ? oracle.reason_code
              : LlmBackendReason::EXECUTION_PLAN_MISMATCH;
      result.backend_evidence = std::move(task);
      return result;
    }
    const LlmMetalResourcePlan& resource_plan = resolved_execution_plan_.resources;
    id<MTLBuffer> argument = find_buffer(owned_buffers_, resource_plan.planned_resources,
                                         LlmMetalResourcePool::ArgumentBuffer);
    id<MTLBuffer> status =
        find_buffer(owned_buffers_, resource_plan.planned_resources, LlmMetalResourcePool::Status);
    if (argument == nil || status == nil || status.contents == nullptr ||
        status.length < LlmMetalKernelContract::kTimedStatusWordCount * sizeof(uint32_t)) {
      result.status = LlmTaskExecutionStatus::Failed;
      result.reason_code = LlmBackendReason::RESOURCES_NOT_PREPARED;
      result.backend_evidence = std::move(task);
      return result;
    }
    LlmMetalErrorDiagnostic task_error;
    if (!reset_task_status(status, task, task_error)) {
      task.error = std::move(task_error);
      result.status = LlmTaskExecutionStatus::Failed;
      result.reason_code = LlmBackendReason::STATUS_RESET_COMMAND_FAILED;
      result.backend_evidence = std::move(task);
      return result;
    }
    if (!execute_timed_command(pipeline, argument, status, parameters,
                               parameter_bytes, grid, scenario_plan.scenario, task,
                               task_error)) {
      task.error = std::move(task_error);
      result.status = LlmTaskExecutionStatus::Failed;
      result.reason_code = LlmBackendReason::TIMED_COMMAND_BUFFER_ERROR;
      result.backend_evidence = std::move(task);
      return result;
    }

    const auto* status_words = static_cast<const uint32_t*>(status.contents);
    task.expected_checksum = oracle.checksum;
    task.actual_checksum = {{status_words[LlmMetalKernelContract::kWeightChecksumAIndex],
                             status_words[LlmMetalKernelContract::kWeightChecksumBIndex]},
                            {status_words[LlmMetalKernelContract::kKeyChecksumAIndex],
                             status_words[LlmMetalKernelContract::kKeyChecksumBIndex]},
                            {status_words[LlmMetalKernelContract::kValueChecksumAIndex],
                             status_words[LlmMetalKernelContract::kValueChecksumBIndex]}};
    if (hooks_.force_timed_checksum_mismatch) {
      ++task.actual_checksum.weight.a;
    }
    const size_t actual_lookup_count =
        status_words[LlmMetalKernelContract::kLayoutMetadataLookupCountIndex];
    size_t actual_lookup_bytes = 0;
    const bool lookup_bytes_valid = NumericUtils::checked_multiply(
        actual_lookup_count, sizeof(uint32_t), actual_lookup_bytes);
    task.checksum_evaluated = true;
    task.checksum_valid =
        status_words[LlmMetalKernelContract::kStatusFlagsIndex] == 0 &&
        actual_lookup_count == scenario_plan.layout_metadata_lookup_count &&
        lookup_bytes_valid &&
        actual_lookup_bytes == scenario_plan.layout_metadata_read_bytes &&
        equal_llm_metal_checksum(task.expected_checksum, task.actual_checksum);
    result.timing.evaluated = task.timing_evaluated;
    result.timing.valid = task.timing_valid;
    result.timing.elapsed_seconds = task.gpu_elapsed_seconds;
    result.completion.completed_work_units = scenario_plan.work_units;
    result.completion.completed_effective_model_payload_bytes = scenario_plan.effective_model_payload_bytes;
    result.completion.completed_layout_metadata_lookup_count = actual_lookup_count;
    result.completion.completed_layout_metadata_read_bytes =
        lookup_bytes_valid ? actual_lookup_bytes : 0;
    result.completion.completed_task_accounted_bytes = scenario_plan.task_accounted_bytes;

    if (!run_task_post_validation(argument, status, parameters,
                                  parameter_bytes, grid,
                                  scenario_plan.scenario, task, task_error)) {
      task.error = std::move(task_error);
      result.status = LlmTaskExecutionStatus::Failed;
      result.reason_code = LlmBackendReason::POST_VALIDATION_COMMAND_FAILED;
      result.backend_evidence = std::move(task);
      return result;
    }
    result.validation.evaluated = true;
    result.validation.valid = task.post_validation_valid;
    if (!task.timing_valid) {
      result.status = LlmTaskExecutionStatus::Invalid;
      result.reason_code = LlmBackendReason::INVALID_GPU_TIMESTAMPS;
    } else if (!task.checksum_valid) {
      result.status = LlmTaskExecutionStatus::Invalid;
      result.reason_code = LlmBackendReason::TIMED_CHECKSUM_MISMATCH;
    } else if (!task.kv_write_validation_valid) {
      result.status = LlmTaskExecutionStatus::Invalid;
      result.reason_code = LlmBackendReason::KV_WRITE_VALIDATION_MISMATCH;
    } else if (task.padding_canary_applicable &&
               !task.padding_canary_valid) {
      result.status = LlmTaskExecutionStatus::Invalid;
      result.reason_code = LlmBackendReason::PADDING_CANARY_MISMATCH;
    } else {
      result.status = LlmTaskExecutionStatus::Complete;
      result.reason_code = LlmBackendReason::VALID;
    }
    result.backend_evidence = std::move(task);
    return result;
  }

  LlmBackendLifecycleResult prepare_resources_impl(const LlmMemoryWorkPlan& model_plan) {
    if (!initialized_ || !plan_resolved_ || evidence_.initialization.status != LlmBackendStatus::Ready ||
        model_plan.backend != LlmMemoryBackend::Metal || model_plan.plan_identity != resolved_plan_identity_) {
      evidence_.preparation = {LlmBackendStatus::Failed, LlmBackendReason::EXECUTION_PLAN_MISMATCH};
      return evidence_.preparation;
    }
    const auto* submitted = get_llm_metal_execution_plan(model_plan);
    if (submitted == nullptr || !submitted->valid || submitted->identity != resolved_execution_plan_.identity ||
        submitted->resources.identity != resolved_execution_plan_.resources.identity) {
      evidence_.preparation = {LlmBackendStatus::Failed, LlmBackendReason::PLAN_RESOURCE_IDENTITY_MISMATCH};
      return evidence_.preparation;
    }
    if (resources_prepared_) {
      evidence_.preparation = {LlmBackendStatus::Ready, LlmBackendReason::VALID};
      return evidence_.preparation;
    }
    const LlmMetalResourcePlan& plan = resolved_execution_plan_.resources;
    LlmMetalResourceEvidence resource_evidence;
    preparation_interrupted_ = false;
    metal_evidence().resources = resource_evidence;
    LlmMetalErrorDiagnostic phase_error;
    if (mark_preparation_interrupted(phase_error)) {
      return fail_preparation(LlmBackendReason::PREPARATION_INTERRUPTED, std::move(phase_error));
    }
    resource_evidence.allocation_attempted = true;
    resource_evidence.persistent_resource_length_bytes = plan.persistent_resource_length_bytes;
    if (plan.paged_layout.has_value() && !NumericUtils::checked_add(plan.paged_layout->memory.k_layout_padding_bytes,
                                                                    plan.paged_layout->memory.v_layout_padding_bytes,
                                                                    resource_evidence.layout_padding_bytes)) {
      evidence_.preparation = {LlmBackendStatus::Failed, LlmBackendReason::PLAN_RESOURCE_IDENTITY_MISMATCH};
      return evidence_.preparation;
    }
    resource_evidence.transient_peak_bytes = plan.transient_peak_bytes;
    resource_evidence.additional_owned_bytes = plan.additional_owned_bytes;
    resource_evidence.admitted_budget_bytes = plan.admitted_budget_bytes;
    resource_evidence.current_allocated_size_before = static_cast<uint64_t>(device_.currentAllocatedSize);
    resource_evidence.current_allocated_size_peak = resource_evidence.current_allocated_size_before;
    const uint64_t recommended = device_.recommendedMaxWorkingSetSize;
    if (recommended != 0) {
      resource_evidence.recommended_working_set_available = true;
    }
    metal_evidence().resources = resource_evidence;

    __strong NSMutableArray<id<MTLBuffer>>* candidate =
        [[NSMutableArray alloc] initWithCapacity:plan.planned_resources.size()];
    std::vector<LlmMetalAllocatedResource> actual_allocations;
    actual_allocations.reserve(plan.planned_resources.size());
    LlmMetalResourceEvidence& retained = metal_evidence().resources;
    if (candidate == nil) {
      retained.current_allocated_size_after_release = static_cast<uint64_t>(device_.currentAllocatedSize);
      retained.candidate_cleanup_completed = true;
      retained.error = internal_error("candidate container allocation failed");
      evidence_.preparation = {LlmBackendStatus::Failed, LlmBackendReason::METAL_RESOURCE_ALLOCATION_FAILED};
      return evidence_.preparation;
    }
    retained.resources.reserve(plan.planned_resources.size());
    for (size_t index = 0; index < plan.planned_resources.size(); ++index) {
      if (mark_preparation_interrupted(phase_error)) {
        candidate = nil;
        return fail_preparation(LlmBackendReason::PREPARATION_INTERRUPTED, std::move(phase_error));
      }
      if (index == hooks_.fail_allocation_after) {
        candidate = nil;
        retained.current_allocated_size_after_release = static_cast<uint64_t>(device_.currentAllocatedSize);
        retained.candidate_cleanup_completed = true;
        retained.error = internal_error("injected candidate allocation failure");
        evidence_.preparation = {LlmBackendStatus::Failed, LlmBackendReason::METAL_RESOURCE_ALLOCATION_FAILED};
        return evidence_.preparation;
      }
      const LlmMetalPlannedResource& planned = plan.planned_resources[index];
      const MTLResourceOptions options = is_private_pool(planned.pool)
                                             ? MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeTracked
                                             : MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked;
      id<MTLBuffer> buffer = [device_ newBufferWithLength:planned.length_bytes options:options];
      if (buffer == nil || (!is_private_pool(planned.pool) && buffer.contents == nullptr)) {
        candidate = nil;
        retained.current_allocated_size_after_release = static_cast<uint64_t>(device_.currentAllocatedSize);
        retained.candidate_cleanup_completed = true;
        retained.error = internal_error("newBufferWithLength returned nil");
        evidence_.preparation = {LlmBackendStatus::Failed, LlmBackendReason::METAL_RESOURCE_ALLOCATION_FAILED};
        return evidence_.preparation;
      }
      const std::string label = std::string("membenchmark.llm-metal.") +
                                llm_metal_resource_pool_to_string(planned.pool) + "-" +
                                std::to_string(planned.pool_index);
      buffer.label = [NSString stringWithUTF8String:label.c_str()];
      [candidate addObject:buffer];
      retained.current_allocated_size_peak =
          std::max(retained.current_allocated_size_peak, static_cast<uint64_t>(device_.currentAllocatedSize));
      LlmMetalResourceMetadata metadata = build_resource_metadata(buffer, planned);
      LlmMetalAllocatedResource actual;
      actual.pool = planned.pool;
      actual.pool_index = planned.pool_index;
      actual.length_bytes = metadata.length_bytes;
      if (metadata.allocated_size_available) {
        actual.allocated_size_bytes = metadata.allocated_size_bytes;
      }
      retained.resources.push_back(std::move(metadata));
      actual_allocations.push_back(std::move(actual));
    }
    retained.allocation_completed = true;
    retained.current_allocated_size_peak =
        std::max(retained.current_allocated_size_peak, static_cast<uint64_t>(device_.currentAllocatedSize));
    const LlmMetalCommittedAdmission committed = evaluate_llm_metal_committed_admission(plan, actual_allocations);
    retained.committed_resource_bytes = committed.committed_resource_bytes;
    retained.resource_rounding_bytes = committed.resource_rounding_bytes;
    retained.known_owned_peak_bytes = committed.known_owned_peak_bytes;
    if (retained.recommended_working_set_available) {
      const long double headroom = static_cast<long double>(recommended) -
                                   static_cast<long double>(retained.current_allocated_size_before) -
                                   static_cast<long double>(retained.known_owned_peak_bytes);
      const long double clamped =
          std::max(static_cast<long double>(std::numeric_limits<int64_t>::min()),
                   std::min(headroom, static_cast<long double>(std::numeric_limits<int64_t>::max())));
      retained.recommended_working_set_headroom_bytes = static_cast<int64_t>(clamped);
      retained.recommended_working_set_headroom_fraction =
          static_cast<double>(headroom / static_cast<long double>(recommended));
      retained.exceeds_recommended_working_set = headroom < 0;
    }
    if (!committed.valid) {
      candidate = nil;
      retained.current_allocated_size_after_release = static_cast<uint64_t>(device_.currentAllocatedSize);
      retained.candidate_cleanup_completed = true;
      evidence_.preparation = {LlmBackendStatus::Failed, committed.reason_code};
      return evidence_.preparation;
    }

    if (!encode_argument_buffer(candidate, plan, phase_error)) {
      candidate = nil;
      retained.current_allocated_size_after_release = static_cast<uint64_t>(device_.currentAllocatedSize);
      retained.candidate_cleanup_completed = true;
      retained.error = std::move(phase_error);
      evidence_.preparation = {LlmBackendStatus::Failed, LlmBackendReason::METAL_ARGUMENT_BUFFER_LAYOUT_INVALID};
      return evidence_.preparation;
    }
    if (!upload_and_validate_table(candidate, model_plan, plan, retained,
                                   phase_error)) {
      candidate = nil;
      retained.current_allocated_size_after_release = static_cast<uint64_t>(device_.currentAllocatedSize);
      retained.candidate_cleanup_completed = true;
      retained.error = std::move(phase_error);
      evidence_.preparation = {LlmBackendStatus::Failed, preparation_interrupted_
                                                             ? LlmBackendReason::PREPARATION_INTERRUPTED
                                                             : LlmBackendReason::METAL_RESOURCE_INITIALIZATION_FAILED};
      return evidence_.preparation;
    }
    if (!initialize_and_validate_data(candidate, plan, phase_error)) {
      candidate = nil;
      retained.current_allocated_size_after_release = static_cast<uint64_t>(device_.currentAllocatedSize);
      retained.candidate_cleanup_completed = true;
      retained.error =
          phase_error.description.empty() ? internal_error("private initialization mismatch") : std::move(phase_error);
      evidence_.preparation = {LlmBackendStatus::Failed, preparation_interrupted_
                                                             ? LlmBackendReason::PREPARATION_INTERRUPTED
                                                             : LlmBackendReason::METAL_RESOURCE_INITIALIZATION_FAILED};
      return evidence_.preparation;
    }
    retained.initialization_completed = true;
    retained.post_validation_completed = true;
    retained.cpu_sample_readback_validation_completed = true;
    if (mark_preparation_interrupted(phase_error)) {
      candidate = nil;
      return fail_preparation(LlmBackendReason::PREPARATION_INTERRUPTED, std::move(phase_error));
    }
    if (!run_layout_probe(candidate, plan, metal_evidence().capability, phase_error)) {
      candidate = nil;
      retained.current_allocated_size_after_release = static_cast<uint64_t>(device_.currentAllocatedSize);
      retained.candidate_cleanup_completed = true;
      retained.error =
          phase_error.description.empty() ? internal_error("parameter layout probe mismatch") : std::move(phase_error);
      evidence_.preparation = {LlmBackendStatus::Failed, LlmBackendReason::METAL_ARGUMENT_BUFFER_LAYOUT_INVALID};
      return evidence_.preparation;
    }

    if (mark_preparation_interrupted(phase_error)) {
      candidate = nil;
      return fail_preparation(LlmBackendReason::PREPARATION_INTERRUPTED, std::move(phase_error));
    }

    const size_t persistent_count = static_cast<size_t>(
        std::count_if(
            plan.planned_resources.begin(),
            plan.planned_resources.end(),
            [](const LlmMetalPlannedResource& resource) {
              return resource.persistent;
            }));
    __strong NSMutableArray<id<MTLBuffer>>* published =
        [[NSMutableArray alloc] initWithCapacity:persistent_count];
    if (published == nil) {
      candidate = nil;
      retained.current_allocated_size_after_release = static_cast<uint64_t>(device_.currentAllocatedSize);
      retained.candidate_cleanup_completed = true;
      retained.error = internal_error("published container allocation failed");
      evidence_.preparation = {LlmBackendStatus::Failed, LlmBackendReason::METAL_RESOURCE_ALLOCATION_FAILED};
      return evidence_.preparation;
    }
    for (size_t index = 0; index < plan.planned_resources.size(); ++index) {
      if (plan.planned_resources[index].persistent) {
        [published addObject:candidate[static_cast<NSUInteger>(index)]];
      }
    }
    if (published.count != persistent_count) {
      published = nil;
      candidate = nil;
      retained.current_allocated_size_after_release = static_cast<uint64_t>(device_.currentAllocatedSize);
      retained.candidate_cleanup_completed = true;
      retained.error = internal_error("published resource count mismatch");
      evidence_.preparation = {LlmBackendStatus::Failed, LlmBackendReason::METAL_RESOURCE_ALLOCATION_FAILED};
      return evidence_.preparation;
    }
    owned_buffers_ = published;
    candidate = nil;
    resources_prepared_ = true;
    retained.resources_published = true;
    retained.candidate_cleanup_completed = true;
    metal_evidence().timed_results_available =
        (model_plan.phase == LlmPhase::Decode &&
         (model_plan.kv_layout == LlmKvLayout::Contiguous ||
          model_plan.kv_layout == LlmKvLayout::Paged)) ||
        (model_plan.phase == LlmPhase::Prefill &&
         model_plan.kv_layout == LlmKvLayout::Contiguous);
    evidence_.preparation = {LlmBackendStatus::Ready, LlmBackendReason::VALID};
    return evidence_.preparation;
  }

  LlmMetalBackendTestHooks hooks_;
  bool initialized_ = false;
  bool plan_resolved_ = false;
  bool resources_prepared_ = false;
  bool preparation_interrupted_ = false;
  LlmPhase active_phase_ = LlmPhase::Decode;
  LlmKvLayout active_kv_layout_ = LlmKvLayout::Contiguous;
  std::string resolved_plan_identity_;
  LlmMetalExecutionPlan resolved_execution_plan_;
  LlmMetalPagedChecksumSummary paged_checksum_summary_;
  uint64_t weight_seed_ = 0;
  uint64_t k_seed_ = 0;
  uint64_t v_seed_ = 0;
  uint64_t base_seed_ = 0;
  std::array<uint8_t, Constants::LLM_METAL_SEGMENT_SLOTS_PER_POOL> table_segment_first_bytes_{};
  std::array<bool, Constants::LLM_METAL_SEGMENT_SLOTS_PER_POOL> table_segment_first_byte_valid_{};
  __strong id<MTLDevice> device_ = nil;
  __strong id<MTLCommandQueue> queue_ = nil;
  __strong id<MTLLibrary> library_ = nil;
  FoundationPipelines pipelines_;
  __strong id<MTLArgumentEncoder> argument_encoder_ = nil;
  __strong NSMutableArray<id<MTLBuffer>>* owned_buffers_ = nil;
  LlmBackendEvidence evidence_;
};

}  // namespace

std::unique_ptr<LlmBackend> create_llm_metal_backend() {
  return std::make_unique<LlmMetalBackend>(LlmMetalBackendTestHooks{});
}

std::unique_ptr<LlmBackend> create_llm_metal_backend_for_testing(const LlmMetalBackendTestHooks& hooks) {
  return std::make_unique<LlmMetalBackend>(hooks);
}
